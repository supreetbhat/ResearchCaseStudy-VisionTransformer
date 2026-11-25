# Step 1: Research FLUX (Supreet)

## Our focus
The RCS is about **understanding** how a state-of-the-art vision-transformer image generator works. Our case is **FLUX.1** by Black Forest Labs (BFL). We work backwards from FLUX through the papers it builds on until we reach a core we can fully explain.

- **In scope:** FLUX's anatomy (sampling pipeline, transformer blocks, conditioning, training objective), where each piece comes from in the literature, and how this differs from U-Net diffusion (last year's RCS).
- **Out of scope:** building, replicating or training a model. Our deliverable is explanation and a literature map, not code. *(The supervisor gave "replicate and train" as a minimum bar at the kickoff, so we will confirm this scope at the next meeting.)*
- **Also out of scope:** text-to-image. Following the kickoff meeting, we only care about how the image is generated, not how the prompt is understood.

**Short answer from my research so far:** FLUX's image generator is an **MMDiT/DiT-style transformer trained with rectified flow (flow matching)**. Remove the text side and what is left is essentially a **class- or unconditional DiT with a rectified-flow objective**.

---

## Why we are not focusing on the text encoders
1. **Scope.** The supervisor asked us to drop text-to-image. The interesting question is how the image is generated, e.g. picking a class like "shoe" and getting a new image.
2. **They are not trained as part of FLUX.** The text encoders are separate, **frozen, pre-trained models** made by other organisations:
   - FLUX.1: Google's T5-XXL and OpenAI's CLIP.
   - FLUX.2: Mistral's Small 3.2.

   FLUX only reads their output embeddings as conditioning input.
3. **They are swappable.** BFL replaced T5 + CLIP with a Mistral VLM between FLUX.1 and FLUX.2 and kept the same generator design. A component you can swap without changing how images are made is not the core of the image generator.
4. **They don't create pixels.** Every step from noise to image happens in the transformer (and the VAE decoder). The text encoders run **once**, before sampling starts. The transformer runs at **every** step.
5. **Their role is easy to describe in one line.** In FLUX.1:
   - T5 gives a sequence of per-token text embeddings, which join the image tokens in attention.
   - CLIP gives one pooled vector, which is added to the timestep embedding and steers every block through adaLN.

   That is all we need to know to "cut them away". In a class-conditional model, a learned class embedding takes over CLIP's place (exactly as in DiT), and there are no text tokens at all.

---

## No, Mistral is not the core of FLUX
At the kickoff, the quick search suggested that FLUX's core is the **Mistral (Small 3.x) 24B** model. Our reading shows this is a mix-up between two different parts of the system:

| | FLUX.1 (Aug 2024), our main case | FLUX.2 (Nov 2025) |
|---|---|---|
| **Text encoder** (reads the prompt) | T5-XXL + CLIP ViT-L | **Mistral Small 3.2, 24B** (vision-language model) |
| **Image generator** (makes the image) | 12B rectified-flow transformer: 19 double-stream + 38 single-stream blocks | 32B rectified-flow transformer: 8 double-stream + 48 single-stream blocks |
| **Latent space** | 16-channel VAE | New VAE (128 latent channels reported) |

- **FLUX.1 contains no Mistral at all.**
- In **FLUX.2**, Mistral is the *text encoder*. It turns the prompt into embeddings and is also used for prompt upsampling. It is an LLM-type model that predicts text. It does **not** generate images.
- In both versions, the image is made by a **rectified-flow transformer with MMDiT-style double-stream blocks followed by single-stream blocks**. The generator design stays the same across versions, while the text encoder changes. That is strong evidence for where the core is.
- This is actually good news for the project. Mistral is exactly the text part we were told to remove, and the supervisor's hope that "the text side can be cut away" holds.

*FLUX.2 numbers are from BFL's FLUX.2 blog and NVIDIA's model documentation. Check them against the Hugging Face model card before citing.*

---

## The image generator is MMDiT / DiT-style
### What the generator is
FLUX.1 is a **rectified-flow transformer that works in the latent space of an autoencoder**. Its lineage has three strands:

```
Transformer (Vaswani 2017) → ViT (Dosovitskiy 2020): image = sequence of patch tokens
        │
        ▼
DiT (Peebles & Xie 2023): a transformer replaces the U-Net as the denoiser;
        conditioning (timestep + class) through adaLN; class-conditional on ImageNet
        │
        ▼
MMDiT / SD3 (Esser et al. 2024, the authors who later founded BFL):
        separate weights for text and image tokens, joint attention over both,
        trained with rectified flow
        │
        ▼
FLUX.1 (BFL 2024): 19 MMDiT-style "double-stream" blocks
        + 38 "single-stream" blocks (shared weights, parallel attention + MLP)
        + RoPE positions, in a 16-channel latent space

Training objective:  Flow Matching (Lipman 2022) / Rectified Flow (Liu 2022)
Latent space:        Latent Diffusion (Rombach 2022)
```

### One generation step, piece by piece (FLUX.1)
1. **Start from pure noise.** A random latent of shape (H/8 × W/8 × 16) is drawn from a Gaussian.
2. **Patchify.** Each 2×2 patch of the latent becomes one token (16 × 4 = 64 values), projected to 3072 dims. A 1024×1024 image gives 4096 image tokens. *(ViT/DiT idea)*
3. **Conditioning vector `vec`** = timestep embedding + guidance embedding (dev only) + pooled CLIP vector. It drives **adaLN modulation** (scale, shift, gate) in every block. *(DiT's adaLN)*
4. **Double-stream blocks (×19).** Image tokens and T5 text tokens each have their own weights, but attend **jointly** over the concatenated sequence. *(MMDiT from SD3)*
5. **Single-stream blocks (×38).** All tokens are concatenated and processed with shared weights. Attention and MLP run in parallel. *(parallel blocks as in ViT-22B)*
6. **Output = a velocity.** The network predicts the direction from noise towards the image. The Euler sampler takes a small step along it. *(rectified flow)*
7. **Repeat** for ~28–50 steps (dev) or 1–4 steps (schnell, distilled). Then **un-patchify** and decode with the VAE.

### Why "DiT-style" is the right label
- The building block is a **transformer block with adaLN conditioning**, exactly DiT's design. Double-stream blocks are DiT blocks with a second set of weights for text. Single-stream blocks are DiT blocks with a parallel layout.
- **Without text**, the double-stream blocks have nothing to do on the text side and collapse into plain DiT blocks. The pooled CLIP vector inside `vec` is replaced by a class embedding. The result is **DiT with a rectified-flow objective**.

### Answer to the supervisor's question: does it also start from noise and denoise step by step?
**Yes.** Like the U-Net diffusion model, FLUX starts from pure Gaussian noise and refines it iteratively. The differences are these:

| | U-Net diffusion (DDPM, last year's RCS) | FLUX.1 |
|---|---|---|
| Denoiser network | Convolutional U-Net (+ attention) | Transformer on patch tokens (MMDiT/DiT-style) |
| What the network predicts | The added noise ε | A velocity v (noise → data direction) |
| Path from noise to data | Curved, set by a noise schedule | Straight line (rectified flow) |
| Sampler | Stochastic reverse diffusion, many steps | Deterministic ODE (Euler), fewer steps |
| Where it runs | Pixel space (or latent, for Stable Diffusion) | Latent space (16-channel VAE) |
| Conditioning | Timestep embedding (+ cross-attention for text) | adaLN on `vec` + joint attention with text tokens |

---

## Folder contents
| Path | What it is |
|---|---|
| `findings/01_demystifying_flux_review.md` | Can *Demystifying Flux Architecture* (Greenberg 2025) be cited? Verdict, errors found, what it gets right |
| `findings/02_flux1_anatomy_reading_guide.md` | How to study FLUX.1's anatomy: a six-step reading order with what to read in each source |
| `research-papers/` | Paper list with links, plus the Demystifying Flux PDF (CC-BY 4.0) |
| `code/` | Reference implementations to read (BFL, diffusers, DiT) |
| `blogs/` | Official announcements, model cards, tutorials |

## Key references
- Black Forest Labs (2024). FLUX reference code: https://github.com/black-forest-labs/flux
- Black Forest Labs et al. (2025). *FLUX.1 Kontext*. arXiv:2506.15742: https://arxiv.org/abs/2506.15742
- Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3/MMDiT). arXiv:2403.03206: https://arxiv.org/abs/2403.03206
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers* (DiT). arXiv:2212.09748: https://arxiv.org/abs/2212.09748
- Liu et al. (2022). *Flow Straight and Fast* (Rectified Flow). arXiv:2209.03003: https://arxiv.org/abs/2209.03003
- Lipman et al. (2022). *Flow Matching for Generative Modeling*. arXiv:2210.02747: https://arxiv.org/abs/2210.02747
- Greenberg (2025). *Demystifying Flux Architecture* (unofficial). arXiv:2507.09595: https://arxiv.org/abs/2507.09595
- BFL (2025). FLUX.2 announcement: https://bfl.ai/blog/flux-2

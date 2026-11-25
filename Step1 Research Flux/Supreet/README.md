# Step 1: Understanding FLUX.1 (Supreet)

## Our focus
How FLUX.1 turns noise into an image. That is the job of the **transformer** at its centre (MMDiT/DiT-style, trained with rectified flow), and we trace it back to the papers it comes from. The text encoders only turn the prompt into numbers for the transformer, so we treat them as an input and not as the core.

## FLUX.1 architecture at a glance
```
Prompt ──► T5 + CLIP (frozen text encoders) ──► text tokens + one pooled text vector
                                                          │
Random noise (latent) ──► patchify ──► image tokens ──────┤
                                                          ▼
              Transformer: 19 double-stream blocks (MMDiT) + 38 single-stream blocks
              conditioned on timestep + pooled text vector (adaLN)
                                                          │
                                         predicts velocity▼
              Euler step: move a little from noise towards the image
              (repeat ~28–50 steps)
                                                          │
                                                          ▼
                                    VAE decoder ──► final image
```
**Where the ideas come from:** Transformer → ViT (image as patch tokens) → DiT (transformer replaces the U-Net) → MMDiT / SD3 (text and image tokens attend jointly) → FLUX.1. The training objective is rectified flow / flow matching.

## Why no Mistral here
- **Mistral Small 3.2 (24B)** is the **text encoder of FLUX.2**. It reads the prompt. It does not generate the image.
- **FLUX.1 has no Mistral at all.** It uses T5 + CLIP as text encoders.
- In both versions the image is made by the **same kind of rectified-flow transformer** (double-stream + single-stream blocks). The text encoder changed between versions, but the generator design did not.

So the core of FLUX is the transformer, not Mistral.

---

## Reading plan
6 steps, about 11 hours in total. Go in order.

---

### 1. Demystifying Flux (~1.5 h)
**Greenberg (2025). *Demystifying Flux Architecture*.** arXiv:2507.09595

- Paper: https://arxiv.org/abs/2507.09595
- HTML version: https://arxiv.org/html/2507.09595v1
- Read sections 2.2–2.3 and Figures 4–10 only.

### 2. The official FLUX code (~2–3 h), the ground truth
**Black Forest Labs code:** https://github.com/black-forest-labs/flux

Read these files in order:
1. `src/flux/model.py`
2. `src/flux/modules/layers.py`
3. `src/flux/sampling.py`
4. `src/flux/util.py`

**The same model in Hugging Face diffusers.** Its names match the paper's figures, so it's easier to read side by side:
- Model: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py
- Pipeline: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

**Model cards and launch post:**
- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://bfl.ai/announcements/24-08-01-bfl

### 3. FLUX.1 Kontext report (~30 min), the official source to cite
**Black Forest Labs (2025). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space*.** arXiv:2506.15742

- Paper: https://arxiv.org/abs/2506.15742
- Read only the background and architecture section.

### 4. SD3 (~3 h)
**Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*.** ICML 2024

- Paper: https://arxiv.org/abs/2403.03206
- Read four parts:
  - the rectified-flow setup and how training timesteps are sampled
  - the MMDiT architecture
  - QK-norm
  - the timestep shift

### 5. Flow matching / rectified flow (~2 h)
**Start here (easier):** Holderrieth & Erives (2025), *An Introduction to Flow Matching and Diffusion Models* (MIT 6.S184 lecture notes)
- https://arxiv.org/abs/2506.02070

**Then one of the original papers, core method section only:**
- Liu et al. (2022), *Flow Straight and Fast* (rectified flow), ICLR 2023: https://arxiv.org/abs/2209.03003
- Lipman et al. (2022), *Flow Matching for Generative Modeling*, ICLR 2023: https://arxiv.org/abs/2210.02747

**For looking things up later:** Lipman et al. (2024), *Flow Matching Guide and Code*
- https://arxiv.org/abs/2412.06264

### 6. DiT (~2 h)
**Peebles & Xie (2023). *Scalable Diffusion Models with Transformers*.** ICCV 2023

- Paper: https://arxiv.org/abs/2212.09748
- Code: https://github.com/facebookresearch/DiT (`models.py` is short and readable)
- Read for patchify, adaLN and class conditioning.

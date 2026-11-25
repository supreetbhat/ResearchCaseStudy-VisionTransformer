# Review: *Demystifying Flux Architecture* (Greenberg 2025)

Paper: Or Greenberg (Hebrew University / GM R&D), arXiv:2507.09595v1, 13 Jul 2025, CC-BY 4.0, 20 pages. PDF in `../research-papers/`. Reviewed 2026-11-25.

## 1. Can it be referenced?
**Yes, as a secondary source, but not as your only evidence for a fact.**

| Check | Finding |
|---|---|
| Status | arXiv preprint, v1 only, single author. No venue found and not peer-reviewed. Self-declared "unofficial … not endorsed by" BFL |
| Method | Reverse-engineered from public inference code (BFL repo + diffusers). Says nothing about training, which BFL never disclosed |
| Scope | FLUX.1 only. No FLUX.2. Kontext described as "not yet released" (content frozen 5 Jun 2025) |
| Value | The only written walk-through of FLUX.1's block structure. Figures 4–10 are good diagrams to adapt (CC-BY allows it with attribution) |

**How to cite it:** as "an unofficial reverse-engineering report [Greenberg 2025]". Back each architectural claim with a primary source: BFL code (`github.com/black-forest-labs/flux`), the FLUX.1 Kontext tech report (arXiv 2506.15742, BFL's own paper), SD3 (Esser et al. 2024) and DiT (Peebles & Xie 2023).

### Errors and gaps found (do not copy these)
1. **Inconsistent time convention.** §1.4 and §2.2.2 say z0 = noise and z1 = clean. §2.3.1 says timesteps run 1→0, and the Fig. 5 caption says z0 = clean. The BFL code follows SD3: **t = 1 is noise, t = 0 is data**, and sampling goes 1→0. §2.2.1 also mixes in DDPM notation (t ∈ [0, T], T = 1000).
2. **Overclaim in §1.4.** It says rectified flow "eliminates the need for iterative … reverse denoising". That is wrong: FLUX still integrates an ODE step by step from pure noise (dev runs about 28–50 Euler steps). This matters for the supervisor's question. The answer is **yes, it starts from noise and refines iteratively**. Don't cite this sentence.
3. **adaLN attributed to the wrong paper.** It cites Keddous 2024 [21]. In diffusion, adaLN(-Zero) comes from **DiT (Peebles & Xie 2023), which the report never cites**. The idea itself goes back to AdaIN and FiLM.
4. **Missing details:**
   - QK-norm (RMSNorm on Q and K).
   - The count of single-stream blocks, which is 38 in the code (verify).
   - The origin of parallel attention + MLP blocks (ViT-22B, Dehghani 2023).
   - The resolution-dependent timestep shift in the sampler.
   - The 2×2 patchify, which gives tokens of 16 ch × 4 = 64 dims.
5. **Small slips:**
   - The attention formula (Eq. 3) puts V inside the softmax.
   - The guidance embedding exists only in dev, because it is guidance-distilled. schnell has none.
   - T5 max length is 512 for dev and 256 for schnell. Verify.
   - Timesteps are called "integer" but are continuous.

### What it gets right (consistent with the code and SD3)
- 12B rectified-flow transformer working in a **16-channel VAE latent** (8× downsampling).
- Text encoders: **T5** gives per-token embeddings and **CLIP** gives one pooled vector.
- The conditioning vector `vec` = timestep embedding + guidance embedding + pooled CLIP. It drives **adaLN** in every block.
- **19 double-stream blocks** (MMDiT as in SD3: separate weights per modality, joint attention over the concatenated text and image tokens), then **single-stream blocks** (shared weights, parallel attention + MLP).
- 3-axis **RoPE** from `img_ids` / `txt_ids`. Hidden size 3072.
- Euler flow-matching sampler, then VAE decode.
- Variants: pro is API-only; dev is guidance-distilled and non-commercial; schnell is timestep-distilled (LADD) and Apache-2.0.

## 2. What is left of FLUX.1 without text (conceptual only, nothing is built)
This is a thought exercise: what is left of FLUX.1 once the text side is removed, i.e. what is its core. Our scope is understanding only, with no building or training.

| FLUX.1 part | Role once the text is removed |
|---|---|
| T5 text stream + double-stream blocks | **Text-only.** With no text tokens, a double-stream block reduces to a standard DiT block |
| Pooled CLIP vector inside `vec` | In a class-conditional model this would be a **class embedding** (as in DiT) |
| Guidance embedding | **Not core** (it only exists because of distillation) |
| VAE latent | **Efficiency step from LDM, not core.** Small images like Fashion-MNIST would not need it |
| Patchify, adaLN modulation, attention + MLP blocks | **Core:** this is the DiT backbone |
| RoPE (2D) | **Core idea, interchangeable detail:** any positional encoding plays the same role |
| Rectified-flow velocity loss + Euler sampler | **Core:** this is the objective |

Conclusion: FLUX.1 without its text side is a **class-conditional DiT trained with rectified flow**. See `../README.md` for the full discussion.

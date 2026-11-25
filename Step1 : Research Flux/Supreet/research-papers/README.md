# Research papers

`Greenberg2025_DemystifyingFluxArchitecture_arXiv2507.09595.pdf`: Or Greenberg, *Demystifying Flux Architecture*, arXiv:2507.09595 (2025). Redistributed under its CC-BY 4.0 licence (https://creativecommons.org/licenses/by/4.0/), unmodified. It is unofficial and not peer-reviewed; see `../findings/01_demystifying_flux_review.md`.

## Core reading (in order; see `../findings/02_flux1_anatomy_reading_guide.md`)
| Step | Paper | Link | What to read |
|---|---|---|---|
| 1 | Greenberg (2025). *Demystifying Flux Architecture* (unofficial) | https://arxiv.org/abs/2507.09595 | §2.2–2.3, Figs. 4–10 |
| 3 | Black Forest Labs et al. (2025). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space* | https://arxiv.org/abs/2506.15742 | Background/architecture section. BFL's only paper, the official source to cite |
| 4 | Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3/MMDiT), ICML 2024 | https://arxiv.org/abs/2403.03206 | Rectified-flow setup, timestep sampling, MMDiT, QK-norm, timestep shift |
| 5 | Liu, Gong, Liu (2022). *Flow Straight and Fast: Rectified Flow*, ICLR 2023 | https://arxiv.org/abs/2209.03003 | Core method |
| 5 | Lipman et al. (2022). *Flow Matching for Generative Modeling*, ICLR 2023 | https://arxiv.org/abs/2210.02747 | Core method |
| 5 (easier start) | Holderrieth & Erives (2025). *An Introduction to Flow Matching and Diffusion Models* (MIT 6.S184 notes) | https://arxiv.org/abs/2506.02070 | Intro chapters |
| 5 (reference) | Lipman et al. (2024). *Flow Matching Guide and Code* | https://arxiv.org/abs/2412.06264 | Look things up as needed |
| 6 | Peebles & Xie (2023). *Scalable Diffusion Models with Transformers* (DiT), ICCV 2023 | https://arxiv.org/abs/2212.09748 | Patchify, adaLN-Zero, class conditioning, CFG |

## Map nodes (one line each on the literature map)
| Node | Paper | Link |
|---|---|---|
| Latent space | Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR | https://arxiv.org/abs/2112.10752 |
| RoPE | Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding* | https://arxiv.org/abs/2104.09864 |
| Parallel blocks | Dehghani et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters* | https://arxiv.org/abs/2302.05442 |
| schnell distillation | Sauer et al. (2024). *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation* | https://arxiv.org/abs/2403.12015 |
| dev distillation | Meng et al. (2023). *On Distillation of Guided Diffusion Models*, CVPR | https://arxiv.org/abs/2210.03142 |
| CFG | Ho & Salimans (2022). *Classifier-Free Diffusion Guidance* | https://arxiv.org/abs/2207.12598 |
| Text encoder (not core) | Radford et al. (2021). *CLIP* | https://arxiv.org/abs/2103.00020 |
| Text encoder (not core) | Raffel et al. (2020). *T5* | https://arxiv.org/abs/1910.10683 |
| U-Net diffusion baseline | Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models* | https://arxiv.org/abs/2006.11239 |
| Foundations | Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words* (ViT) | https://arxiv.org/abs/2010.11929 |
| Foundations | Vaswani et al. (2017). *Attention Is All You Need* | https://arxiv.org/abs/1706.03762 |
| Dataset | Xiao, Rasul, Vollgraf (2017). *Fashion-MNIST* | https://arxiv.org/abs/1708.07747 |

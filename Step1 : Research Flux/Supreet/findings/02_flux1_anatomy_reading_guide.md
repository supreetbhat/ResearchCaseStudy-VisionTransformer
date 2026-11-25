# FLUX.1 anatomy: reading guide

Goal: be able to draw one forward pass with tensor shapes, and say which paper each piece comes from.

**Step 1: Demystifying Flux, §2.2–2.3 only (~1.5 h).** Skip §1, §3 and the appendix. Work through Figs. 4–10. Use it as the map, keeping the errors listed in `01_demystifying_flux_review.md` in mind.

**Step 2: BFL reference code (~2–3 h). This is the ground truth.** Repo `black-forest-labs/flux`, folder `src/flux/` (verify paths in the repo; they can move):
- `model.py`: the `Flux` class. Find `img_in`, `txt_in`, `time_in`, `vector_in`, `guidance_in`, `pe_embedder`, `double_blocks`, `single_blocks`, `final_layer`.
- `modules/layers.py`: `EmbedND` (RoPE), `MLPEmbedder`, `Modulation` (adaLN), `QKNorm`, `DoubleStreamBlock`, `SingleStreamBlock`, `LastLayer`.
- `sampling.py`: `get_noise`, `prepare` (the 2×2 patchify rearrange plus building `img_ids`/`txt_ids`), `get_schedule` (time shift), `denoise` (the Euler loop).
- `util.py`: the configs (hidden 3072, heads 24, depth 19 / 38, `axes_dim`, `in_channels` 64, `guidance_embed`).

Exercise: trace a 1024×1024 image.
- The VAE gives a 128×128×16 latent.
- Patchify gives 64×64 = 4096 tokens of 64 dims, which are projected to 3072.
- Add 512 T5 tokens, for 4608 tokens in joint attention.
- Write down the shape after every line of `Flux.forward`.

**Step 3: FLUX.1 Kontext report (arXiv 2506.15742), architecture/background section only (~30 min).** It is the only paper by BFL itself, so it is the **citable official source** that FLUX.1 is a rectified-flow transformer with double- and single-stream blocks.

**Step 4: SD3 (Esser et al. 2024) (~3 h).** Read:
- the rectified-flow formulation and timestep sampling (logit-normal);
- the MMDiT figure and section;
- QK-norm;
- the resolution-dependent timestep shift.

This fills the gaps Demystifying leaves (see `01_demystifying_flux_review.md`).

**Step 5: Rectified Flow (Liu et al. 2022) or Flow Matching (Lipman et al. 2022), core method section only (~2 h).** Read for the loss, the straight-line path and the ODE sampling.

**Step 6: DiT (Peebles & Xie 2023) (~2 h).** Read for patchify, adaLN-Zero, class conditioning and CFG. This is the core that FLUX reduces to once the text is removed.

**One-line nodes for the map (no deep reading):**
- LDM (VAE latent space)
- RoFormer (RoPE)
- ViT-22B (parallel blocks)
- LADD (schnell)
- Guidance distillation (Meng 2023, dev)
- CLIP and T5 (text side, not part of the core)
- ViT and Transformer (foundations)

Links for every source are in `../research-papers/README.md`, `../code/README.md` and `../blogs/README.md`.

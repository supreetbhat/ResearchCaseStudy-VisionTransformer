# Reference code to read (we read it to understand the model; nothing is built here)

| Source | Link | Files to read |
|---|---|---|
| Black Forest Labs: official FLUX.1 inference code (the ground truth) | https://github.com/black-forest-labs/flux | `src/flux/model.py` (`Flux` class), `src/flux/modules/layers.py` (`EmbedND`, `MLPEmbedder`, `Modulation`, `QKNorm`, `DoubleStreamBlock`, `SingleStreamBlock`, `LastLayer`), `src/flux/sampling.py` (`get_noise`, `prepare`, `get_schedule`, `denoise`), `src/flux/util.py` (model configs) |
| Hugging Face diffusers: FLUX transformer (names match Demystifying Flux's figures) | https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py | Whole file |
| Hugging Face diffusers: FLUX pipeline | https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py | `__call__`, latent packing, timestep schedule |
| Meta: DiT reference code | https://github.com/facebookresearch/DiT | `models.py` (short and readable) |
| Black Forest Labs: FLUX.2 inference code | https://github.com/black-forest-labs/flux2 | For comparison only (text encoder = Mistral) |

File paths are from reading the repos; they can move, so search the repo if a path breaks.

**Exercise:** trace one 1024×1024 image through `Flux.forward`.
- VAE latent: 128×128×16.
- 2×2 patchify: 4096 tokens × 64, projected to 3072.
- Add 512 T5 tokens: 4608 tokens in joint attention.
- Write down the shape after each line.

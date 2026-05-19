# LoRA Fine-Tuning of Stable Diffusion Models for Style Transfer
### A documented design-rationale study, extended with a custom-autoencoder integration experiment, under consumer-hardware constraints

Parameter-efficient adaptation of large diffusion models is a well-
established space (LoRA, DreamBooth, Textual Inversion, and variants),
and methodological novelty is not what this project claims. What
off-the-shelf training tools expose is LoRA configuration as form
fields: rank, network alpha, target modules, learning rate per
submodule. The numerical defaults (rank 16–32, alpha equal to rank or
half of rank, lr ~3e-5) circulate as community lore in guides and
forum threads, but the *rationale* for those defaults — and how to
change them for a specific objective — is rarely traced end-to-end in
user-facing materials. The community has even built dedicated
reverse-engineering tools to recover what settings a finished LoRA was
trained with, which is its own evidence that the rationale layer is
missing.

This project documents one careful instantiation of those decisions
across two diffusion-model scales (SDXL 1.0, SD 1.5), uses trigger-token
activation to avoid catastrophic forgetting at the style level, and
includes a custom-autoencoder training-loop feasibility study. The
intended contribution is **documented rationale** — for which submodules
get LoRA, what rank suffices at each scale, why one of SDXL's two text
encoders is left untouched, what mixed precision does to the VAE
numerics, and what swapping the VAE reveals about pipeline modularity —
not new methodology.

---

## Claims

**C1 — Trigger-token activation cleanly preserves baseline generation
while gaining new style.** Binding the LoRA-tuned style to a rare
two-piece subword token (`xkz` → `['x', 'kz</w>']`) and prefixing every
training caption makes the new style *addressable* rather than
overwriting. With the trigger: anime style. Without it: baseline
generation preserved. Verified qualitatively across both SDXL and SD 1.5
at every save_every checkpoint. [→ §3]

**C2 — Rank asymmetry between style adoption (UNet) and trigger
binding (text encoder).** UNet attention projections need substantially
higher LoRA rank — r=32 for SDXL, r=24 for SD 1.5 — than the text
encoder needs to bind the trigger token (r=8 for both). Trainable
parameter ratio: **1.78% / 0.55% / 0.24%** (UNet SDXL / UNet SD 1.5 /
text encoder). Documents what "low-rank" empirically looks like across
two model scales for the same task family. [→ §2.1, §3]

**C3 — Selective adaptation in SDXL's dual-encoder architecture.**
Only `text_encoder_1` (CLIP ViT-L/14) receives LoRA; `text_encoder_2`
(OpenCLIP ViT-bigG/14) is deliberately frozen to preserve compositional
grounding. A documented design choice, not a library default —
`text_encoder_2` carries the broader semantic load and adapting it
risks degrading compositional understanding on non-anime prompts.
[→ §2.1]

**C4 — Mixed-precision interaction with VAE numerics: FP16-only does
not work.** The VAE's encode step overflows in FP16 and produces NaN
loss; the empirically working pattern is FP16 base + lifted-FP32 VAE.
UNet and text encoders run under `torch.amp.autocast(dtype=float16)` to
fit in 24GB VRAM; the VAE is explicitly cast `.to(torch.float32)`
because its numeric range cannot survive FP16. Documents one
consumer-hardware mixed-precision pattern. [→ §4]

**C5 — Evaluation is qualitative by structural necessity, not by
shortcut.** LoRA leaves base weights unchanged (`W' = W + BA`); FID /
CLIP score on trigger-off output measures the base model exactly, and
on trigger-on output is dominated by the base model with the ~2% LoRA
delta on top. Meaningful metrics here are style adherence, capability
preservation, and trigger-leakage rate — the first two are intrinsically
visual (and were inspected per save_every checkpoint), the third is
quantifiable but not done. [→ §5]

**C6 — Custom autoencoder: documenting training-loop feasibility and
pretrained-VAE asymmetry.** A simplified AE (3 DownBlocks / MidBlock-
with-attention / 3 UpBlocks, 50 epochs on 1,200 256×256 images, AdamW +
MSE) converges cleanly — loss curves show conventional descent, unlike
LoRA training where loss is nearly flat. `SimpleAutoencoderKL` mocks the
diffusers `AutoencoderKL` interface so it drops into
`StableDiffusionPipeline` with no pipeline-side changes. **What's
documented**: end-to-end VAE training + pipeline integration both work;
the drop-in swap reveals that the pretrained VAE carries substantial
off-distribution generality (color reconstruction) that a 1,200-image
custom AE cannot recover. **What's not delivered**: a production-quality
VAE replacement — that is gated by training scale (billions of images),
not by the path itself. [→ §6]

**Limitations** — see §8.

---

## 1. Why LoRA, briefly

Updating all ~3.5B parameters of SDXL is infeasible on consumer
hardware, and even when feasible would catastrophically forget the
model's general-purpose generation ability. LoRA injects a low-rank
matrix `BA` into target layers such that the effective weight becomes:

```
W' = W + BA,  where rank(BA) << rank(W)
```

Only `B` and `A` are trained. Combined with a trigger-token gating
scheme (§2.2), this gives a fine-tune that is *additive* and
*addressable* — base behavior is preserved, the new style is on a
switch.

---

## 2. Method

### 2.1 LoRA configurations (supports C2, C3)

```python
# UNet attention projections — the core style/content predictor
# SDXL: r=32, alpha=32. SD 1.5: r=24, alpha=24 (smaller UNet, lower rank suffices)
LoraConfig(
    r=32, lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)

# Text encoder — to bind the trigger token (r=8 for both SDXL and SD 1.5)
LoraConfig(
    r=8, lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
)
```

Resulting trainable-parameter accounting:

| Model | Total | LoRA trainable | Ratio |
|---|---|---|---|
| SDXL 1.0 (UNet) | 2.61B | 46.4M | **1.78%** |
| SD 1.5 (UNet) | 864M | 4.78M | **0.55%** |
| Text encoder (each) | 123M | 295K | **0.24%** |

For SDXL, only `text_encoder_1` (CLIP ViT-L/14) is LoRA-adapted;
`text_encoder_2` (OpenCLIP ViT-bigG/14) is frozen. The rationale:
`text_encoder_2` carries the broader semantic and compositional load,
and adapting it risks degrading compositional grounding on prompts
outside the anime training distribution. Whether adapting both encoders
would improve style consistency on multi-subject prompts is left
untested (see Limitations).

### 2.2 Trigger token design (supports C1)

A common failure mode of style fine-tuning: train on anime images, the
model *overwrites* general capability — prompt for a realistic portrait,
still get anime. The fix is to bind the style to a rare token that does
not collide with existing vocabulary. `xkz` was chosen for two reasons:

```python
>>> tokenizer.tokenize("xkz")
['x', 'kz</w>']
```

(i) it tokenizes into two rarely-seen subword pieces, and (ii) neither
piece overlaps with common words. During training every caption is
prefixed `xkz, `. The model learns:

- **With `xkz`** → anime style
- **Without `xkz`** → original behavior preserved

This is why the text encoder also needs LoRA, not just the UNet — the
text encoder is where the token-to-style binding actually lives.

---

## 3. Results (supports C1, C2)

### 3.1 SDXL 1.0

Training: 1,200 image-prompt pairs (filtered from 337K by prompt token
length ≤ 65, leaving 12-token headroom under the CLIP-77 cap for the
trigger + special tokens), 40 validation pairs, resolution 1024×1024,
AdamW with cosine-annealed LR (`T_max=10000` set deliberately longer
than the ~6,000 expected total steps so LR doesn't collapse to
`eta_min` mid-training).

Loss curves are **flat rather than sharply descending** — expected when
LoRA-tuning a heavily pretrained model. Visual inspection of checkpoint
samples at each 200-step interval (via TensorBoard) was the primary
model-selection signal. **Optimal checkpoint at step 3,200**:
trigger-on samples have converted style while trigger-off samples
retain the original realistic style.

**Prompt: "a photo of a woman walking on the street"**

| With trigger (`xkz, ...`) | No trigger |
|---|---|
| ![SDXL with trigger](assets/sdxl_woman_with.png) | ![SDXL no trigger](assets/sdxl_woman_no.png) |

**Prompt: "a young girl, white t-shirt, blue hat"**

| With trigger | No trigger |
|---|---|
| ![SDXL girl with](assets/sdxl_girl_with.png) | ![SDXL girl no](assets/sdxl_girl_no.png) |

### 3.2 SD 1.5

SD 1.5 has ~27% of SDXL's parameter count and uses a single text
encoder. Training was correspondingly faster; **optimal checkpoint at
step 1,200**.

**Prompt: "two young women having dinner on the table"**

| With trigger | No trigger |
|---|---|
| ![SD15 women with](assets/sd15_women_with.png) | ![SD15 women no](assets/sd15_women_no.png) |

Style transfer is less consistent than SDXL — some trigger-on samples
retain semi-realistic features. Consistent with SD 1.5's weaker
semantic grounding (single CLIP ViT-L/14 encoder, no OpenCLIP
ViT-bigG/14 pair).

---

## 4. Mixed precision: FP16 base + FP32 VAE (supports C4)

The pipeline and UNet load and train in FP16 via
`torch.amp.autocast("cuda", dtype=torch.float16)`. This is what allows
SDXL 1.0 (2.6B params) to fit on a 24GB consumer GPU at batch size 2,
resolution 1024×1024.

But the VAE cannot stay in FP16. Its encode step — projecting a
1024×1024 RGB image down to a 128×128 latent — **overflows in FP16 and
produces NaN loss**. The working pattern:

```python
# train.py
vae = pipe.vae.to(torch.float32)                  # VAE explicitly FP32

# engine.py — training step
latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()  # FP32 encode
latents = (latents * vae_scaling).to(dtype=torch.float16)              # back to FP16 for UNet

with torch.amp.autocast("cuda", dtype=torch.float16):
    model_pred = unet(noisy_latents, timesteps, ...)                   # UNet in FP16
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")  # loss in FP32
```

The same pattern applies to SD 1.5 training. This is one documented
consumer-hardware pattern: lift the precision-fragile component (VAE),
keep the VRAM-hungry components (UNet, text encoders) at FP16. The
split is what makes SDXL training viable on 24GB at full resolution.

---

## 5. Why no FID / CLIP score (supports C5)

LoRA is an *additive* adapter — `W' = W + BA`. Base weights are not
modified. This has direct consequences for evaluation choice:

- **Trigger-off output is the base model unchanged.** Running SDXL or
  SD 1.5 with the LoRA loaded but no `xkz` prefix produces the original
  model's output (modulo residual leakage). FID or CLIP score on
  trigger-off output measures the base model, not this project's
  contribution.
- **Trigger-on output is the base model plus a ~2% additive delta.**
  The LoRA contributes a small residual on top of base-model outputs.
  FID or CLIP score on trigger-on output is still dominated by base-
  model quality.

The metrics that *would* be meaningful for this work:

1. **Style adherence** — does the trigger-on output look like anime?
   Intrinsically visual, inspected per save_every checkpoint.
2. **Capability preservation** — is trigger-off output indistinguishable
   from the unmodified base model? Could be quantified by per-pixel diff
   against the base model under matched prompt + seed; not done.
3. **Trigger leakage rate** — does the model produce anime-style output
   even *without* the trigger? Quantifiable (e.g. a binary anime / non-
   anime classifier over a held-out prompt set) but not done.

Model selection was done via TensorBoard inspection of side-by-side
trigger-on / trigger-off sample pairs at every save_every interval (200
steps for SDXL / SD 1.5; 5 epochs for the autoencoder).

---

## 6. Custom autoencoder — training-loop feasibility and pretrained-VAE asymmetry (supports C6)

### 6.1 Architecture

`SimpleAutoencoderKL` (in `sd_lora_anime/models.py`):

- **3 DownBlocks** (single ResBlock each, stride-2 Conv2d downsample),
  channel widths 128 / 256 / 512
- **MidBlock** with self-attention (preserved — critical for global
  coherence on small datasets)
- **3 UpBlocks** (single ResBlock each, nearest-neighbor upsample +
  Conv2d), mirroring the encoder widths
- GroupNorm + SiLU throughout
- 4-channel latent via `quant_conv` (matching SD 1.5 VAE channel count)

Trained 50 epochs on 1,200 images at 256×256, AdamW lr=1e-4, MSE loss.
Best reconstruction at **epoch 35**.

### 6.2 Drop-in compatibility with `StableDiffusionPipeline`

This is the non-trivial engineering bit. `SimpleAutoencoderKL` mocks the
diffusers `AutoencoderKL` interface so it works inside the unmodified
pipeline:

```python
self.config = SimpleNamespace(
    scaling_factor=0.18215,
    use_quant_conv=True,
    use_post_quant_conv=True,
)

def encode(self, x, return_dict=True):
    ...
    return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(h))

def decode(self, z, return_dict=True, generator=None):
    ...
    return DecoderOutput(sample=dec)
```

With the mock interface, the custom AE drops straight into
`StableDiffusionPipeline.vae` without any pipeline-side changes.

### 6.3 Reconstruction in-distribution

![Autoencoder reconstruction](assets/ae_reconstruction.png)

Loss curves show conventional convergence — unlike LoRA training where
loss is nearly flat. The custom AE is learning a real reconstruction
signal on the training distribution.

### 6.4 Pipeline swap: SD 1.5 VAE → custom AE

**Prompt: "anime style, a little girl playing in the park"**

| Original SD 1.5 VAE | Custom Autoencoder |
|---|---|
| ![SD15 original VAE](assets/sd15_vae_original.png) | ![SD15 custom AE](assets/sd15_vae_custom.png) |

Custom AE preserves **structurally recognizable output** — character
outline, trees, path — but color reconstruction is severely degraded
with large blocky artifacts.

### 6.5 Reading this experiment

Two things are documented:

1. **The training-loop scaffolding works end-to-end.** Encoder, decoder,
   and integration with `StableDiffusionPipeline` are all functional.
   The training loop converges. There is no infrastructure obstacle to
   scaling this up, only a compute obstacle.
2. **Pretrained-VAE asymmetry.** The pretrained SD VAE carries
   substantial off-distribution generality — color reconstruction beyond
   the training distribution — that a 1,200-image custom AE cannot
   recover. The swap shows where pipeline modularity has its limit:
   latent spaces are not interoperable between a VAE trained on a narrow
   distribution and a UNet trained on billions of images.

What is *not* delivered: a production-quality VAE replacement.
Achieving that would require training-scale equivalence to the original
SD VAE (orders of magnitude more data and compute) — gated by hardware,
not by the path.

---

## 7. Hardware and environment

| | |
|---|---|
| GPU | single NVIDIA RTX 4090 (24GB VRAM) |
| Precision | FP16 base + FP32 VAE (see §4) |
| Batch size | 2 (SDXL / SD 1.5 at native resolution), 4 (custom AE at 256×256) |
| Optimizer | AdamW with cosine-annealed LR (`T_max=10000`, `eta_min=1e-7`) |
| Learning rates | SDXL 1e-5; SD 1.5 5e-5; custom AE 1e-4 |
| Training wall-clock | SDXL ~2.5 h; SD 1.5 ~30 min; custom AE ~25 min (from TensorBoard event timestamps) |
| Run multiplicity | each stage executed once; no multi-seed averaging |
| Inference | deterministic per `TEST_SEED=22333`, 40 inference steps |

Validation loss is computed at every evaluation interval (200 steps for
SDXL / SD 1.5, every epoch for the custom AE) and logged to TensorBoard
alongside training loss. TensorBoard also receives generated samples at
each checkpoint for visual model selection.

---

## 8. Limitations

- **Single domain.** Anime style only. Generalization to other style
  domains (oil painting, photograph stylization, etc.) is untested.

- **Small training set.** 1,200 image-prompt pairs is orders of
  magnitude below dataset scales typical in LoRA fine-tuning research.
  The trigger mechanism works, but the learned style concentrates around
  *anime characters* (the dominant content in this dataset slice);
  generalizing to anime-style *scenes without characters* would require
  dataset rebalancing.

- **Failure trajectories not preserved.** During design iteration,
  multiple alternative configurations were tried — different LoRA ranks
  (both lower and higher than the final r=32 / 24 / 8 set), different
  `target_modules` choices, different learning rates per submodule, an
  FP16-only attempt before the VAE-precision issue (§4) was diagnosed —
  but logs and intermediate checkpoints from those failing runs were not
  preserved. The decisions documented in §2 and §4 describe the final
  working configuration; the alternative trajectories that motivated
  those decisions cannot be quantitatively retraced. A research-grade
  follow-up would log all attempted configurations to a structured
  experiment tracker (e.g. Weights & Biases) from the start.

- **Qualitative evaluation by structural necessity.** See §5. Style
  adherence and capability preservation were inspected visually at every
  checkpoint; trigger-leakage rate was not quantitatively measured.

- **Single run per stage.** SDXL, SD 1.5, and the custom AE were each
  trained once. No multi-seed averaging, no LR sweep, no rank sweep.
  Variance estimates over reruns are not available.

- **`text_encoder_2` adaptation not tested.** Deliberately untouched per
  C3, but it remains untested whether adapting both SDXL encoders would
  improve style consistency on complex multi-subject prompts.

- **Custom AE off-distribution generality unverified.** See §6 — the
  swap experiment shows the AE has *not* learned a VAE-equivalent latent
  space; how to close that gap with consumer-hardware-feasible methods
  is not addressed.

---

## 9. Reproducing

The best LoRA weights and the custom AE checkpoint are shipped in
`weights/` via Git LFS, so inference works without retraining.

### Quickstart

```bash
git lfs install
git clone https://github.com/hansjohn819-commits/consumer-sd-pipeline-study.git
cd consumer-sd-pipeline-study
pip install -r requirements.txt

# Inference with shipped weights
python inference.py --model sdxl --prompt "a photo of a woman walking on the street" --trigger --output out_sdxl.png
python inference.py --model sd15 --prompt "two young women having dinner on the table" --trigger --output out_sd15.png
python inference.py --model ae   --input_image assets/sdxl_girl_with.png --output recon.png
```

### Retraining

```bash
# All three stages sequentially
python train.py --stage all

# Or single stage with overrides
python train.py --stage sdxl_train --lr 1e-5 --epochs 10
python train.py --stage sd15_train
python train.py --stage vae_train
```

Hyperparameters live at the top of `train.py`; CLI flags override
per-stage values. TensorBoard logs land in `runs/<stage>/`.

### Verifiability

Components needed to reproduce results:

- **Training data**: public HuggingFace dataset
  [`none-yet/anime-captions`](https://huggingface.co/datasets/none-yet/anime-captions);
  deterministic 1,240-sample selection via `DATASET_SEED=321`
- **Best weights**: committed via Git LFS in `weights/` (186 MB SDXL
  adapter, ~40 MB SD 1.5 adapter, 157 MB custom AE checkpoint)
- **Inference determinism**: `TEST_SEED=22333` + 40 inference steps
- **What is not reproducible without a 24 GB GPU**: the training runs
  themselves

---

## 10. References

- Hu, E. J., et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR.
- Podell, D., et al. (2023). [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952).
- Rombach, R., et al. (2022). [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). CVPR.
- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS.
- Ruiz, N., et al. (2023). [DreamBooth: Fine-Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242). CVPR.
- Gal, R., et al. (2022). [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618). ICLR 2023.
- Radford, A., et al. (2021). [Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf). ICML.
- Ilharco, G., et al. (2021). [OpenCLIP](https://doi.org/10.5281/zenodo.5143773).
- HuggingFace [PEFT library](https://github.com/huggingface/peft) — `LoraConfig` and `get_peft_model` used throughout this project.

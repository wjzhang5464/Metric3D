# Fine-tuning protocol: Metric3D v2-S on TartanAir v2

Archive of the rules, sources and decisions for the Metric3D v2-S fine-tuning baseline. The paper's
supplementary material describes this baseline from this file. Every claim about the papers or the code
cites its source, so it can be checked. Decisions are dated in the decision log (section 9).

Status (2026-10-14): protocol agreed; no fine-tuning code written or run yet.

## 1. Scope

- Model: **Metric3D v2-S only** (DINOv2-reg ViT-S + RAFT/GRU decoder). We use **metric depth only**; the
  normal outputs are neither evaluated nor delivered.
- Purpose: a baseline for our paper, next to the zero-shot v2-S result
  (`reports/zeroshot_v2s_tartanair/README.md`).
- Data: TartanAir v2 package (`/depot/chan129/data/TartanAir_v2/`, read-only). Train 69,779 frames and
  validation 1,944 frames (`tartanair_v2_trainval/`); test 10,868 frames (`tartanair_v2_test/`).
  All frames are 640x640 with fx = fy = 320 and cx = cy = 320.

## 2. Mandatory rules (from our collaborator, received 2026-10-14)

Verbatim. They were written for both UniDepthV2 and Metric3Dv2; only the Metric3Dv2 parts apply to us.
They override the official recipe wherever the two differ.

> Initialize from the official pretrained ViT-S weights.
>
> Follow the official fine-tuning script or example if one exists. If not, unfreeze the entire encoder and
> give it a lower learning rate than the decoder (e.g. 0.1x).
>
> Train for 20 epochs on the train split only, to match our model, even if the official recipe uses a
> different schedule. Please save a checkpoint after every epoch.
>
> Select the best checkpoint by depth MAE on the validation split. Never use the test set for any choice.
> This also applies to choosing between the two Metric3Dv2 runs in point 5.
>
> During training, treat sky pixels (use the provided sky mask) and pixels with depth > 200 m as invalid.
> For Metric3Dv2 only: its canonical-space depth limit (depth_normalize=(0.1, 200)) corresponds to only
> about 62 m in our data, because labels are scaled by 1000 / 308 ≈ 3.25. Please fine-tune it twice,
> keeping focal_length=1000 in both runs:
> (a) depth_normalize=(0.1, 700), which covers 200 m. Please run this one first.
> (b) depth_normalize=(0.1, 200), the official setting.
>
> At test time, save the de-canonicalized depth before the < 300 zeroing in postprocess_per_image.
>
> At test time, provide the ground-truth intrinsics K and do not apply any scale alignment. Save one
> prediction per frame for all 10,868 test frames, metric depth at 640x640, as
> <pred_dir>/<subset>/<frame_key>.npy, as described in the test package README. We will compute all
> metrics on our side.
>
> Please also rerun the zero-shot (pretrained, not fine-tuned) models with the official settings and the
> same inference script, and save their predictions in the same format.

**Test-set rule in practice.** No decision, smoke test, sanity check or debugging step during fine-tuning
looks at test frames. Validation frames serve all of these purposes. The test set is touched exactly once
per model, to write the final predictions.

## 3. What the official sources say about fine-tuning

These three sources disagree, and none of them gives a ViT-S fine-tuning setting.

| Source | What it says |
|---|---|
| v2 paper (arXiv 2404.15506v4), §IV "Implementation details" | "Fine-tuning experiments on KITTI and NYU are conducted on 8 GPUs with 20K further steps." |
| v2 paper, Supp §1.2 "Finetuning protocols" | ADAM, learning rate 1e-6 decayed linearly to 1e-7 within 6K steps; batch 16 (ViT-g) / 32 (ViT-L); "converge quickly in approximately 2K steps"; adds pseudo-labels from the pre-fine-tuned model, L_ft = 0.01 L_d-n + L_d + L_n + 0.01 (L_d(D_c, D̃_c) + L_n(N, Ñ)). |
| Code: `training/README.md` "Example: Finetune on KITTI" → `train_kitti.sh` → `training/mono/configs/RAFTDecoder/vit.raft5.large.kitti.py` | AdamW; encoder lr 5e-7 and decoder lr 1e-5, weight decay 0, eps 1e-10, `strict_match=True`; poly(0.9) schedule with a 20-iteration linear warmup, min_lr 1e-8; 20,010 iterations; 2 images per GPU; crop 616x1064; losses VNL 0.1 + GRUSequence (L1) 1.0 + DeNoConsistency 0.001. The pseudo-label term is **not implemented**. There are also `vit.raft5.giant2.kitti.py` and `vit.raft5.giant2.nyu.py`; no ViT-S config exists. |

Further facts:
- In the v2 paper, fine-tuned ("FT") rows exist only for ViT-L and ViT-g (Tables I, II); there are none
  for ViT-S. The fine-tuned checkpoints were never released (GitHub issue #202, unanswered).
- The maintainers point to `train_kitti.sh` for fine-tuning: "Not need to change any settings
  additionally" (#173).
- Issue #119 suggests freezing the backbone first. Issue #173 says freezing the encoder "will impair some
  performance". Both come from the same maintainer.
- **TartanAir is not in Metric3D's training data.** It is absent from both papers. The only synthetic
  sets in v2 Table V are Virtual KITTI and Hypersim, so our zero-shot result is genuinely zero-shot.
- No training seed is stated in the papers. `train.sh` does not pass `--seed`, so `train.py`'s argparse
  default **88** applies.

## 4. Chosen recipe

**Principle.** Rule 2 applies: an official fine-tuning example exists (`train_kitti.sh` with
`vit.raft5.large.kitti.py`), so we follow it, adapted to ViT-S. Rules 3-5 override it on schedule length,
masking and depth range. The Supp §1.2 protocol is not a script, and its pseudo-label term is not in the
code, so we do not use it; the supplementary material states this difference. The official script already
unfreezes the whole encoder with 0.05x the decoder learning rate (5e-7 vs. 1e-5), so the rule's 0.1x
fallback does not apply.

| Setting | Value | Source |
|---|---|---|
| Initialisation | official `metric_depth_vit_small_800k.pth` (sha256 `b34b2a2b…`) via `--load-from` | rule 1 |
| Backbone / decoder | `dino_vit_small_reg.dpt_raft`, `RAFTDepthNormalDPT5`, `iters=4`, `n_downsample=2` | official v2-S (inference config `vit.raft5.small.py`; the KITTI config's `iters=8` belongs to ViT-L/g) |
| Trainable parameters | everything (encoder unfrozen) | official KITTI config |
| Optimizer | AdamW; encoder lr 5e-7, decoder lr 1e-5; betas (0.9, 0.999); weight decay 0; eps 1e-10; `strict_match=True` (keeps the encoder/decoder parameter groups correct) | official KITTI config |
| LR schedule | linear warmup 20 iterations (ratio 1e-6), then poly power 0.9 down to min_lr 1e-8, stretched over the full 20 epochs | official shape + rule 3 |
| Batch | 16 images per optimizer step (official: 2 per GPU x 8 GPUs). Target: 1 A100 with batch 16; if it does not fit, gradient accumulation (`acc_batch`) keeps the effective batch at 16. Verified in the smoke test. | official + decision |
| Length | 20 epochs over the train split only. 1 epoch = floor(69,779 / 16) = 4,361 iterations, so 87,220 iterations in total (the official 20,010 iterations are about 4.6 of our epochs). `drop_last` drops 3 random frames per epoch; the sampler reshuffles every epoch. | rule 3 |
| Checkpoints | one per epoch (every 4,361 iterations), 20 per run | rule 3 |
| Losses | VNLoss 0.1, GRUSequenceLoss 1.0 (gamma 0.9, `stereo_sup=0`), DeNoConsistencyLoss 0.001 (`CEL`, scale 2) | official KITTI config |
| Precision / clipping | bf16 autocast, gradient-norm clipping 2.5 (official training loop, `IterBasedRunner_AMP`) | official code |
| Canonical space | focal_length 1000, labels scaled by 1000 / fx (CSTM_label) | official + rule 5 |
| Crop and augmentation | official KITTI training pipeline, unchanged: BGR2RGB, LabelScaleCononical, RandomResize (prob 0.5, ratio 0.85-1.15), RandomCrop 616x1064 (random; pads with RGB 0 and label -1), RandomEdgeMask (0.2), RandomHorizontalFlip (0.4), PhotoMetricDistortion, Weather (0.05), RandomBlur (0.05), RGBCompresion (0.1), Normalize | official KITTI config |
| Invalid pixels | GT set invalid (-1) wherever the package's `valid_mask.png` is 0. Its definition is exactly `depth > 1e-3 & depth <= 200 & ~sky`, i.e. rule 5. No sky regression loss (the official KITTI config has none either). | rule 5 |
| Dataset `data_type` | `denselidar_syn` (as for the official synthetic sets VKITTI / Hypersim / Replica). All three losses above are active for it. | decision |
| Seed | `--seed 3047` (our method's seed). The sampler's hard-coded seeds (136 / 639, incremented per epoch) and the dataset's `random.seed(100)` stay unchanged. | decision |
| Runs | (a) `depth_normalize=(0.1, 700)` first, then (b) `depth_normalize=(0.1, 200)`. Everything else identical. | rule 5 |

**Normals.** TartanAir has no GT normals. `NormalBranchLoss` is not in the recipe, and it would return
zero anyway, because it masks on non-zero GT normals. `DeNoConsistencyLoss` needs no labels: it compares
normals computed from the predicted depth with the predicted normals. It is kept because the official
script uses it.

## 5. Facts from the code that affect the runs

1. **Training and test use the same canonical scale.** During training the image keeps its size and the
   label is multiplied by 1000 / 320 = 3.125 (`training/mono/utils/transform.py`, `LabelScaleCononical`).
   RandomResize by a factor r multiplies the label by a further 1/r. At test time the 640 image is resized
   to 616, i.e. r = 0.9625, which gives 3.125 / 0.9625 = 3.247, the inference label scale factor.
2. **The decoder clamps canonical depth to `depth_normalize`.** It computes `y = max(min_val, min(y, max_val))`
   (`RAFTDepthNormalDPTDecoder5.py`, `clamp`), so there is no gradient beyond the limit. The metric cap at
   test time is max / 3.247: **61.6 m** for run (b) and **215.6 m** for run (a).
3. **`depth_normalize` also sets the decoder's output parametrisation, not just the cap.** The depth bins
   run from log(min) to log(max), and the GRU input is (d - max) / 100. Changing max from 200 to 700
   stretches the bins (far bins by up to 3.5x) and shifts the GRU input by -5. Run (a) therefore loads
   the official weights but does not reproduce the zero-shot predictions at step 0. This is inherent to
   rule 5. We record the step-0 validation MAE of both runs for reference.
4. **Run (a) still clips the farthest GT under strong downscaling.** A 200 m label becomes
   200 x 3.125 / r canonical, which exceeds 700 when r < 0.893. That happens in about 7% of samples
   (prob 0.5 x (0.893 - 0.85) / 0.3), and then only for GT above about 190 m. We accept this and mention
   it in the supplementary material.
5. **The `< 300` zeroing never fires for either run or for zero-shot.** The decoder's output range maps to
   [0.031, 61.6] m (b, zero-shot) or [0.031, 215.6] m (a). We still save before the zeroing, as rule 6
   requires.
6. **Padding differs between training and inference.** RandomCrop pads the 640-wide image to 1064 with
   RGB 0 (black) and label -1; inference pads with the ImageNet mean. This is official behaviour (the KITTI
   fine-tune pads 352 to 616 rows the same way). It is documented here and not changed.
7. **Inference for each run uses a config with the same `depth_normalize` as its training.** Run (a)
   needs an inference config with (0.1, 700); the inference script is the same for all models.
8. **The training and inference trees (`training/mono/` vs. root `mono/`) have diverged.** Fine-tuned
   checkpoints are evaluated with the root `mono/` code. Before any evaluation we check that the checkpoint
   loads with no unexpected keys. We also check that, at step 0, the training-tree and inference-tree
   forward passes agree.

## 6. Checkpoint selection (validation only)

- Candidates: the 20 epoch checkpoints (epochs 1-20) of each run. The un-fine-tuned weights are not a
  candidate.
- Each candidate is run on the 1,944 validation frames with **the same inference script and settings as
  the test predictions**: full 640x640 frame, GT intrinsics, no alignment, saved before the `< 300`
  zeroing.
- Score: depth MAE as defined in the official `metrics/eval_depth.py`. It is computed per frame on the
  package's valid mask and averaged over frames with equal weight. The checkpoint with the lowest MAE wins.
- Runs (a) and (b) are each selected independently, and **both** selected models are delivered. If one
  of the two has to be named as "the" Metric3Dv2 baseline, the choice is also made by validation MAE.
- We keep a record of the per-epoch validation MAE for both runs.

## 7. Test-time output (delivery)

Three prediction sets, each with one file per test frame (10,868), produced by the same inference script:

| Model | Weights | Inference config |
|---|---|---|
| zero-shot | official `metric_depth_vit_small_800k.pth` | official `mono/configs/HourglassDecoder/vit.raft5.small.py` |
| fine-tuned (a) | best validation epoch of run (a) | v2-S config with `depth_normalize=(0.1, 700)` |
| fine-tuned (b) | best validation epoch of run (b) | official v2-S config, `depth_normalize=(0.1, 200)` |

Format (test package README, "Metric script"): `<pred_dir>/<subset>/<frame_key>.npy`, written with
`numpy.save`, float32, 640x640, metric depth in metres, de-canonicalised and saved **before** the `< 300`
zeroing. GT intrinsics come from `frames.tsv`; no scale alignment. The collaborator computes all metrics.
We may also run the official `eval_depth.py` ourselves as a format check (it reports failures for
missing, non-640x640 or non-positive predictions).

### Zero-shot: conformance of the existing run and the required rerun

Checked 2026-10-14 against `$RCAC_SCRATCH/metric3d/outputs/zeroshot_v2s/full/pred/`, produced by
`tartanair_eval/infer_tartanair.py`:

| Requirement | Existing run |
|---|---|
| `<pred_dir>/<subset>/<frame_key>.npy`, all 10,868 frames | yes: 10,868 files, exactly the `frames.tsv` keys |
| float32, 640x640, metres | yes (sampled files: float32 (640, 640), 0.33-59.8 m) |
| GT intrinsics, no scale alignment | yes (fx/fy/cx/cy from `frames.tsv`; alignment exists only in our DIAGNOSTIC analyses, never in saved predictions) |
| official settings | yes (unmodified config and weights; bit-exact with the official pipeline on the demos) |
| saved before the `< 300` zeroing | **no, literally**: `predict()` applies `(d > 0) * (d < 300) * d` before saving. Numerically this changes nothing (fact 5), so the saved values equal the pre-zeroing values bit for bit. |

**Tip for the rerun.** Once the inference script is final (saving before the zeroing, and able to load
fine-tuned weights and configs), rerun zero-shot with that exact script into a new directory, e.g.
`$RCAC_SCRATCH/metric3d/outputs/zeroshot_v2s/final/pred`. The rerun takes about 17 minutes on an A100.
Then compare it to the old predictions: they should be bit-identical, which confirms fact 5. The
**delivered** zero-shot set is the rerun, so all three sets come from the same script version. The old
run and the zero-shot report remain valid as they are.

## 8. Compute, storage and layout

- Slurm account **`stanchan`**. Smoke tests use the `standby` queue. Long runs either go through
  `sbatch`, or are started by hand on an interactive node (up to 7 days). If a run outlives its job, it
  continues with `--resume-from`.
- **Never compute on the login node** (check `hostname`).
- Everything stays on scratch until both fine-tuning runs and the zero-shot rerun are finished. Then the
  results are moved to depot together (not yet decided what exactly is kept).
- Planned scratch layout (`$RCAC_SCRATCH/metric3d/...`):
  - `checkpoints/ft_v2s_ta2_dn700/`, `checkpoints/ft_v2s_ta2_dn200/` (`ckpt/stepXXXXXXXX.pth`, logs,
    tensorboard)
  - `outputs/finetune_v2s/<run>/val/epXX/pred/` (validation predictions per epoch) and
    `outputs/finetune_v2s/<run>/test/pred/` (delivered test predictions)
  - `outputs/zeroshot_v2s/final/pred/` (delivered zero-shot rerun)
- Before first use, verify the sha256 of all 11 trainval zips against `tartanair_v2_trainval/README.md`
  (on 2026-10-14 the zips were still being copied).

## 9. Decision log

| Date | Decision | By |
|---|---|---|
| 2026-10-14 | Only v2-S; metric depth only, normals ignored | user |
| 2026-10-14 | Collaborator's rules (section 2) are mandatory | user + collaborator |
| 2026-10-14 | Recipe = official KITTI fine-tune example adapted to ViT-S, overridden by the rules (section 4) | user (accepted proposal) |
| 2026-10-14 | Effective batch 16 on one A100 (gradient accumulation if needed); confirm in a smoke test | user |
| 2026-10-14 | Validation MAE = `eval_depth.py` MAE, frame-weighted mean over the 1,944 validation frames | user (accepted proposal) |
| 2026-10-14 | Keep DeNoConsistencyLoss as in the official code, including its behaviour on sky pixels: it uses no labels, so it does not supervise sky | user + collaborator (accepted proposal) |
| 2026-10-14 | Selection candidates are epochs 1-20 only | user + collaborator (accepted proposal) |
| 2026-10-14 | Deliver both runs (a) and (b), each with its own best checkpoint; run (a) first | user |
| 2026-10-14 | Seed 3047 (our method's seed); official hard-coded sampler seeds unchanged | user (accepted proposal) |
| 2026-10-14 | Slurm account `stanchan`; `standby` for smoke tests; long runs by sbatch or on an interactive node | user |
| 2026-10-14 | All outputs stay on scratch until all experiments are done; move to depot together later | user |
| 2026-10-14 | Delivered zero-shot set = rerun with the final inference script (section 7) | user (accepted proposal) |

## 10. Plan

1. Data: verify the zip checksums; write the `TartanAirDataset` training class; build the annotations
   from `frames.tsv`; check dataloader samples visually and numerically on train/validation frames
   (label scale 3.125 / r, masks, padding).
2. Configs: the training config for ViT-S runs (a) and (b); the inference config for (a).
3. Smoke test (standby): pretrained weights load with only `mask_token` missing; training- and
   inference-tree forward passes agree at step 0; a few hundred iterations with finite, decreasing losses;
   memory and speed at batch 16.
4. Inference script: save before the zeroing; support validation frames and fine-tuned weights/configs.
   Then the zero-shot rerun and its bit-identity check.
5. Run (a): 20 epochs; validation MAE for every epoch; select; test predictions.
6. Run (b): the same.
7. Write-up for the supplementary material: hyperparameter table, deviations from the official
   sources, commit hashes, checkpoint sha256s.

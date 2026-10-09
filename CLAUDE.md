# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Metric3D v1/v2: zero-shot monocular metric depth and surface-normal estimation. v1 = ConvNeXt encoder + Hourglass decoder (depth only); v2 = DINOv2-reg ViT (S/L/giant2) encoder + RAFT-style GRU decoder (depth + normals).

## Project goals

1. Get original inference and training running.
2. Fine-tune on TartanAir v2 (`/depot/chan129/data/TartanAir_v2/`).
3. Evaluate with our own eval script.

## Rules

- **Check `hostname` before running any compute.** Login nodes must never run training, inference or GPU jobs; submit with `sbatch` instead. On compute nodes (`gilbreth-k*`) small tests may run directly; long training always goes through sbatch scripts in `slurm/`.
- **Environment:** use only `/depot/chan129/users/zhan5056/virtual_env/Metric3D_ViT`. Ask the user before installing, upgrading or removing any package in it. Never touch the colleague's env at `/depot/chan129/users/bole/virtual_env/Metric3D_ViT` (the source it was cloned from).
- **Dataset is read-only:** never modify anything under `/depot/chan129/data/TartanAir_v2/`.
- **Large outputs go to scratch** (layout below), never into the repo or home dir. Scratch is purged periodically: copy final checkpoints and results to depot.
- **Git:** commit every meaningful change on branch `gilbreth-run` with a clear message, so the diff from upstream is always traceable. Upstream base commit is recorded in `ORIGINAL_COMMIT.txt` (`eb5b6fa`).

## Storage layout

| What | Path |
|---|---|
| Code (this repo) | `~/MMDE/metric3d-v2`, branch `gilbreth-run` |
| TartanAir v2 (read-only) | `/depot/chan129/data/TartanAir_v2/` (`tartanair_v2_trainval/`, `tartanair_v2_test/`) |
| Pretrained weights | `$RCAC_SCRATCH/metric3d/weights` (v2-S `metric_depth_vit_small_800k.pth` present, sha256 `b34b2a2b…`) |
| Fine-tuned checkpoints | `$RCAC_SCRATCH/metric3d/checkpoints` |
| Training and Slurm logs | `$RCAC_SCRATCH/metric3d/logs` |
| Predictions, eval results | `$RCAC_SCRATCH/metric3d/outputs` |

`$RCAC_SCRATCH` = `/scratch/gilbreth/zhan5056`. Tools default to writing in the repo (`show_dirs/`, `work_dirs/`), so always pass `--show-dir` / `--work-dir` pointing to scratch.

## Environment status

Checked 2026-10-09 on `gilbreth-k046` (NVIDIA A100 80GB PCIe, driver 595.91.07):

- Python 3.10.20, torch 2.1.0+cu121 (CUDA 12.1, cuDNN 8.9.2), `torch.cuda.is_available()` True, torchvision 0.16.0, xformers 0.0.22.post7 (memory-efficient attention works on GPU), mmcv 2.2.0 (compiled ops OK), mmengine 0.10.7, numpy 1.23.1, timm 0.9.12, opencv-python 4.8.0.74, h5py 3.16.0.
- vs `requirements_v2.txt`: torch/torchvision/xformers are newer than pinned (2.0.1/0.15.2/0.0.21) but mutually consistent; everything else present.
- mmcv 2.x has no `mmcv.utils.Config`; code falls back via `try/except` to `mmengine.Config`. Keep that pattern when adding imports.
- `imagecorruptions` fails to import (needs `pkg_resources`, removed in setuptools 82); not imported by any repo code, so harmless.
- **Inference:** all `mono/` modules and `hubconf.py` import. v2-S forward pass on random 1×3×616×1064 input: checkpoint loads with 0 unexpected keys and only `depth_model.encoder.mask_token` missing (expected; unused at inference); depth/confidence/normal outputs finite; ~46 ms per pass after warm-up, 0.76 GB peak GPU memory.
- **Training:** `mmseg` is not installed and `training/mono/utils/comm.py:14` imports it unconditionally, which breaks every training module. With only `mmseg` stubbed in-process, all 81 `training/mono/{model,utils,datasets}` modules and `mono/tools/train.py` import successfully in this env (h5py now satisfies `hypersim_dataset.py`).

## Pending decisions and known issues

- **mmseg (awaiting user confirmation; do not change yet):** plan is to make the `import mmseg` in `training/mono/utils/comm.py` optional (only used for env-info logging at line 59) rather than install mmseg, which likely conflicts with mmcv 2.2.0.
- **Upstream bug:** `training/mono/tools/test.py:29` imports `do_test_check_data` from `mono.utils.do_test`, which is not defined anywhere (its only use, line 157, is commented out). `test.py` will fail with ImportError until that name is dropped from the import. Not fixed yet.
- `training/mono/utils/raindropper/*` cannot be imported as standalone modules (uses `from raindropper...` absolute imports); its consumer `weather_aug_utils.py` imports fine. Repo quirk, not an env problem.

## Two separate code trees, both named `mono`

- `mono/` (repo root): inference only. Entry point `mono/tools/test_scale_cano.py`; also used by `hubconf.py` and `onnx/`.
- `training/mono/`: a separate, larger copy (adds `datasets/`, `model/losses/`, `criterion.py`, `do_train.py`, train/test tools). It is **not** a package extension of the root `mono/`; it shadows it.
- Scripts insert their own tree root into `sys.path` and `chdir` there, so run inference commands from the repo root and training commands from `training/`. Never import both trees in one process.
- Model files (backbones, decode heads) are duplicated and have diverged between the trees; a fix in one usually needs porting to the other.

## Commands

No test suite, linter or build step exists. "Testing" means running inference/eval.

Inference (from repo root, on a compute node):
```bash
# In-the-wild image folder (no intrinsics -> tries several default focal lengths)
python mono/tools/test_scale_cano.py mono/configs/HourglassDecoder/vit.raft5.small.py \
    --load-from $RCAC_SCRATCH/metric3d/weights/metric_depth_vit_small_800k.pth \
    --test_data_path ./data/wild_demo --launcher None \
    --show-dir $RCAC_SCRATCH/metric3d/outputs/<run_name>
# Dataset mode: pass a json annotation (see data/gene_annos_kitti_demo.py), e.g. ./data/kitti_demo/test_annotations.json
```
`test.sh`, `test_vit.sh`, `test_kitti.sh`, `test_nyu.sh` are the upstream examples (they expect weights in `./weight/`). `--launcher` defaults to `slurm`; pass `None` for single-process runs.

Minimal programmatic use is in `hubconf.py` (`metric3d_vit_small(pretrain=True)` etc., then `model.inference({'input': rgb})` → `(pred_depth, confidence, output_dict)`; normals at `output_dict['prediction_normal'][:, :3]`, normal confidence at `[:, 3]`). `pretrain=True` downloads via `torch.hub` into `~/.cache`; prefer loading the file from `$RCAC_SCRATCH/metric3d/weights` with `torch.load(...)['model_state_dict']`.

ONNX: `python onnx/metric3d_onnx_export.py metric3d_vit_small` then `python onnx/test_onnx.py metric3d_vit_small.onnx` (README's `metric_3d_onnx_export.py` name is a typo).

Training (from `training/`; upstream shell scripts in `training/mono/scripts/train_scripts/` `cd ../../../` so must be run from inside that dir, and the README wrongly calls it `training_scripts`). Long runs go through sbatch scripts in `slurm/`:
```bash
python mono/tools/train.py mono/configs/RAFTDecoder/vit.raft5.small.sanity_check.py \
    --use-tensorboard --launcher slurm --experiment_name set1 \
    --work-dir $RCAC_SCRATCH/metric3d/checkpoints/<run_name> [--load-from ckpt.pth] [--resume-from ckpt.pth]
python mono/tools/test.py mono/configs/test_configs_vit_small/ibims.vit.dpt.raft.py --load-from ckpt.pth   # see test.py bug above
```
`train.py` writes the log file and checkpoints into `--work-dir` and tensorboard into `<work-dir>/tensorboard` unless `--tensorboard-dir` is given.

## Architecture

**Configs**: mmengine-style Python configs with `_base_` inheritance. Leaf configs (`mono/configs/HourglassDecoder/*.py`, `training/mono/configs/RAFTDecoder/*.py`) override `model`, `data_basic`, and (training) `losses`, `data_array`, schedule/optimizer. Bases live in `configs/_base_/{models,datasets,losses,schedules}`.

**String-based module registry**: classes are resolved with `mono.utils.comm.get_func('mono.model.' + prefix + type)`. The model is `DepthModel` → `BaseDepthModel` → `model_pipelines.<cfg.model.type>` (`DensePredModel`), which builds `backbones.<backbone.type>` (a factory function like `vit_small_reg`) and `decode_heads.<decode_head.type>` (`RAFTDepthNormalDPT5` for v2, `HourglassDecoder` for v1). Losses are resolved the same way (`training/mono/model/criterion.py`) from the `losses.decoder_losses`/`gru_losses` config lists. Adding a component means adding the module, exporting it in that subpackage's `__init__.py`, and referencing it by name in a config.

**Data/checkpoint path registry**: `data_info/` (inference) and `training/data_server_info/` (training) hold Python dicts of dataset roots/annotation paths and pretrained-backbone paths (placeholders like `/mnt/nas/...`, `tbd_data_root`). At startup `load_data_info` collects them and `reset_ckpt_path` rewrites `cfg.model.backbone.checkpoint` from these. For inference the missing DINOv2/ConvNeXt backbone file is harmless (prints "NO pretrained imagenet ckpt") because the full checkpoint is loaded afterwards. For training, set these paths first. Training datasets are selected in a config's `data_array` by names keyed into `public_datasets.py`; annotations are json lists of `{rgb, depth, depth_scale, cam_in=[fx,fy,cx,cy]}`.

**Canonical camera space (core idea)**: the network predicts depth as if taken by a canonical camera with focal length 1000 (`data_basic.canonical_space.focal_length`). Pre-processing (`mono/utils/do_test.py: transform_test_data_scalecano`, `resize_for_input`) rescales the image to `crop_size` (ViT: 616×1064, multiples of 14; ConvNeXt: 544×1216), pads with the ImageNet mean (padding area must be cropped from outputs), and tracks a `label_scale_factor`. Metric depth = canonical prediction × (real_focal / 1000), adjusted for the resize. Wrong intrinsics give distorted point clouds even when depth maps look fine. Inputs use mean `[123.675,116.28,103.53]` / std `[58.395,57.12,57.375]` on 0–255 RGB. The v2 RAFT decoder clamps canonical depth to `data_basic.depth_normalize` (0.1–200 for v2 configs).

**Checkpoints**: state dict is under `['model_state_dict']`; keys are prefixed `depth_model.`. `mono/utils/running.py: load_ckpt` loads into `model.module`, so the model must already be wrapped in `DataParallel`/DDP (as `test_scale_cano.py` does); for an unwrapped model call `load_state_dict` directly as `hubconf.py` does.

"""Zero-shot Metric3D inference on the TartanAir v2 test package, via the official repo-root `mono/` code.

Model building/loading mirrors mono/tools/test_scale_cano.py; pre-processing and the forward pass call the
official functions in mono/utils/do_test.py unchanged; post-processing replicates the math of
do_test.postprocess_per_image exactly. Frames are read straight from the subset zips (nothing extracted).

Outputs (eval_depth.py layout):
    <out-dir>/pred/<subset>/<frame_key>.npy      float32 depth, metres, 640x640
    <out-dir>/normals/<subset>/<frame_key>.npy   float16 normals, [3, 640, 640] (unless --no-normals)
    <out-dir>/timing.tsv                         per-frame timing (seconds)

    python tartanair_eval/infer_tartanair.py --subsets test_scene --limit 10 --out-dir <dir>
"""
import argparse
import csv
import os
import os.path as osp
import sys
import time
import zipfile

import cv2
import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # repo-root `mono/` only; never `training/`

try:
    from mmcv.utils import Config
except ImportError:
    from mmengine import Config
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.utils.mldb import load_data_info, reset_ckpt_path

DATA_ROOT = '/depot/chan129/data/TartanAir_v2/tartanair_v2_test'
DEFAULT_CONFIG = osp.join(REPO_ROOT, 'mono/configs/HourglassDecoder/vit.raft5.small.py')
DEFAULT_WEIGHTS = osp.join(os.environ.get('RCAC_SCRATCH', ''), 'metric3d/weights/metric_depth_vit_small_800k.pth')
SUBSETS = ('test_matched', 'test_scene', 'pose_combination', 'pose_extrapolation',
           'height_holdout', 'combination_height', 'extrapolation_height')


def build_model(config_path=DEFAULT_CONFIG, ckpt_path=DEFAULT_WEIGHTS):
    """Same steps as test_scale_cano.py (non-distributed, --launcher None)."""
    cfg = Config.fromfile(config_path)
    cfg.load_from = ckpt_path
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    reset_ckpt_path(cfg.model, data_info)
    cfg.distributed = False

    model = get_configured_monodepth_model(cfg, )
    model = torch.nn.DataParallel(model).cuda()
    model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    return model, cfg


def predict(model, cfg, rgb, intrinsic):
    """rgb: HxWx3 uint8 RGB (as cv2.imread(...)[:, :, ::-1]); intrinsic: [fx, fy, cx, cy].
    Returns (depth [H,W] float32 tensor on GPU, metres; normal [3,H,W] tensor on GPU)."""
    normalize_scale = cfg.data_basic.depth_range[1]
    rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(rgb, list(intrinsic), cfg.data_basic)
    pred_depths, outputs = get_prediction(
        model=model,
        input=torch.stack([rgb_input]),
        cam_model=None,
        pad_info=[pad],
        scale_info=None,
        gt_depth=None,
        normalize_scale=None,
    )

    # --- replicates do_test.postprocess_per_image (depth) ---
    pred_depth = pred_depths[0, :].squeeze()
    pred_depth = pred_depth[pad[0]: pred_depth.shape[0] - pad[1], pad[2]: pred_depth.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], [rgb.shape[0], rgb.shape[1]], mode='bilinear').squeeze()
    pred_depth = pred_depth * normalize_scale / label_scale_factor
    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth

    # --- replicates do_test.postprocess_per_image (normals) ---
    pred_normal = outputs['prediction_normal'][0, :][:3, :, :]
    H, W = pred_normal.shape[1:]
    pred_normal = pred_normal[:, pad[0]:H - pad[1], pad[2]:W - pad[3]]
    pred_normal = torch.nn.functional.interpolate(pred_normal[None, :], size=[rgb.shape[0], rgb.shape[1]], mode='bilinear', align_corners=True).squeeze()
    return pred_depth, pred_normal


def read_rgb(zf, key):
    """Decode rgb.png like the official path: cv2.imread(path)[:, :, ::-1]."""
    buf = np.frombuffer(zf.read(key + '/rgb.png'), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def load_frames(data_root, subsets=None, keys_file=None, limit=None):
    """Rows of frames.tsv to run, in file order, optionally restricted to subsets / a key list."""
    with open(osp.join(data_root, 'frames.tsv'), newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if subsets:
        rows = [r for r in rows if r['subset'] in subsets]
    if keys_file:
        with open(keys_file) as f:
            wanted = [ln.strip().split('\t')[-1] for ln in f if ln.strip() and not ln.startswith('#')]
        by_key = {r['frame_key']: r for r in rows}
        missing = [k for k in wanted if k not in by_key]
        if missing:
            raise SystemExit(f'{len(missing)} keys not found in frames.tsv (first: {missing[0]})')
        rows = [by_key[k] for k in wanted]
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--subsets', nargs='+', choices=SUBSETS, default=None, help='default: all')
    ap.add_argument('--keys-file', default=None, help='file with one frame_key per line (optionally "subset<TAB>key")')
    ap.add_argument('--out-dir', required=True, help='writes pred/, normals/ and timing.tsv here')
    ap.add_argument('--limit', type=int, default=None, help='only the first N selected frames')
    ap.add_argument('--skip-existing', action='store_true', help='skip frames whose depth .npy already exists')
    ap.add_argument('--no-normals', action='store_true', help='do not save normals (depth only)')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    a = ap.parse_args()

    if osp.abspath(a.out_dir).startswith('/depot/chan129/data'):
        raise SystemExit('refusing to write under /depot/chan129/data')
    rows = load_frames(a.data_root, a.subsets, a.keys_file, a.limit)
    print(f'{len(rows)} frames selected', flush=True)

    t0 = time.time()
    model, cfg = build_model(a.config, a.weights)
    print(f'model ready in {time.time() - t0:.1f}s', flush=True)

    timing_path = osp.join(a.out_dir, 'timing.tsv')
    os.makedirs(a.out_dir, exist_ok=True)
    new_file = not osp.exists(timing_path)
    tlog = open(timing_path, 'a')
    if new_file:
        tlog.write('subset\tframe_key\tread_s\tinfer_s\tsave_s\ttotal_s\n')

    zips, done, t_start = {}, 0, time.time()
    try:
        for n, row in enumerate(rows, 1):
            subset, key = row['subset'], row['frame_key']
            depth_path = osp.join(a.out_dir, 'pred', subset, key + '.npy')
            normal_path = osp.join(a.out_dir, 'normals', subset, key + '.npy')
            if a.skip_existing and osp.exists(depth_path):
                continue
            t_a = time.time()
            if subset not in zips:
                zips[subset] = zipfile.ZipFile(osp.join(a.data_root, subset + '.zip'))
            rgb = read_rgb(zips[subset], key)
            intrinsic = [float(row['fx']), float(row['fy']), float(row['cx']), float(row['cy'])]
            torch.cuda.synchronize()
            t_b = time.time()
            depth, normal = predict(model, cfg, rgb, intrinsic)
            depth = depth.float().cpu().numpy()
            if not a.no_normals:
                normal = normal.cpu().numpy().astype(np.float16)
            t_c = time.time()
            os.makedirs(osp.dirname(depth_path), exist_ok=True)
            np.save(depth_path, depth.astype(np.float32))
            if not a.no_normals:
                os.makedirs(osp.dirname(normal_path), exist_ok=True)
                np.save(normal_path, normal)
            t_d = time.time()
            tlog.write(f'{subset}\t{key}\t{t_b - t_a:.4f}\t{t_c - t_b:.4f}\t{t_d - t_c:.4f}\t{t_d - t_a:.4f}\n')
            tlog.flush()
            done += 1
            print(f'[{n}/{len(rows)}] {subset}/{key} read {t_b - t_a:.3f}s infer {t_c - t_b:.3f}s '
                  f'save {t_d - t_c:.3f}s | depth {depth.min():.2f}-{depth.max():.2f} m', flush=True)
    finally:
        tlog.close()
        for zf in zips.values():
            zf.close()
    print(f'done: {done} frames in {time.time() - t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()

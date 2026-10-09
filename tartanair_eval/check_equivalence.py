"""Equivalence check: official Metric3D test pipeline vs. infer_tartanair.predict on the same images.

Runs the unmodified do_test.do_scalecano_test_with_custom_data on a demo annotation json, capturing the
official post-processed depth/normals by wrapping (not editing) three functions in mono.utils.do_test:
  - postprocess_per_image  -> grabs the metric meters (dam / dam_median / dam_global) it updates
  - save_val_imgs          -> receives the official final pred_depth
  - save_normal_val_imgs   -> receives the official final pred_normal
Then runs predict() on the same images/intrinsics and reports the differences, plus the official metrics
(which this repo version computes but never prints) and depth ranges.

    python -B tartanair_eval/check_equivalence.py --annos data/kitti_demo/test_annotations.json --out <json>
"""
import argparse
import json
import logging
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from infer_tartanair import REPO_ROOT, DEFAULT_CONFIG, DEFAULT_WEIGHTS, build_model, predict  # noqa: E402
import mono.utils.do_test as dt  # noqa: E402
from mono.utils.custom_data import load_from_annos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annos', required=True, help='annotation json, relative to repo root')
    ap.add_argument('--show-dir', required=True, help='official pipeline writes its vis/ and pcd/ here')
    ap.add_argument('--out', required=True, help='result json')
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    a = ap.parse_args()

    os.chdir(REPO_ROOT)  # as test_scale_cano.py does; annotation paths are relative to it
    model, cfg = build_model(a.config, a.weights)
    cfg.show_dir = a.show_dir
    test_data = load_from_annos(a.annos)

    captured = {'depth': [], 'normal': [], 'meters': None}
    orig_post, orig_save, orig_save_n = dt.postprocess_per_image, dt.save_val_imgs, dt.save_normal_val_imgs

    def post_wrap(*args, **kw):
        captured['meters'] = args[8:11]  # dam, dam_median, dam_global
        return orig_post(*args, **kw)

    def save_wrap(i, pred_depth, *args, **kw):
        captured['depth'].append(pred_depth.detach().clone())
        return orig_save(i, pred_depth, *args, **kw)

    def save_n_wrap(it, pred_normal, *args, **kw):
        captured['normal'].append(pred_normal.detach().clone())
        return orig_save_n(it, pred_normal, *args, **kw)

    dt.postprocess_per_image, dt.save_val_imgs, dt.save_normal_val_imgs = post_wrap, save_wrap, save_n_wrap
    try:
        dt.do_scalecano_test_with_custom_data(model, cfg, test_data, logging.getLogger(), False, 0, 1)
    finally:
        dt.postprocess_per_image, dt.save_val_imgs, dt.save_normal_val_imgs = orig_post, orig_save, orig_save_n
    assert len(captured['depth']) == len(test_data) == len(captured['normal'])

    dam, dam_median, dam_global = captured['meters']
    result = {'annos': a.annos, 'official_metrics': {
        'metric': {k: float(v) for k, v in dam.get_metrics().items()},
        'median_aligned': {k: float(v) for k, v in dam_median.get_metrics().items()},
        'global_aligned': {k: float(v) for k, v in dam_global.get_metrics().items()}},
        'frames': []}

    for an, off_d, off_n in zip(test_data, captured['depth'], captured['normal']):
        rgb = cv2.imread(an['rgb'])[:, :, ::-1].copy()
        gt = cv2.imread(an['depth'], -1) / an['depth_scale']
        assert gt.shape == rgb.shape[:2], 'official resizes pred to GT shape; equality assumed below'
        ours_d, ours_n = predict(model, cfg, rgb, an['intrinsic'])
        dd = (ours_d - off_d).abs()
        dn = (ours_n - off_n).abs()
        od = off_d.cpu().numpy()
        valid_gt = gt[gt > 0]
        result['frames'].append({
            'rgb': an['rgb'], 'shape': list(od.shape),
            'depth_absdiff_max': float(dd.max()), 'depth_absdiff_mean': float(dd.mean()),
            'normal_absdiff_max': float(dn.max()), 'normal_absdiff_mean': float(dn.mean()),
            'pred_min': float(od.min()), 'pred_p1': float(np.percentile(od, 1)),
            'pred_p99': float(np.percentile(od, 99)), 'pred_max': float(od.max()),
            'gt_min': float(valid_gt.min()), 'gt_max': float(valid_gt.max()),
            'median_ratio_gt_over_pred': float(np.median(valid_gt) / np.median(od[gt > 0])),
        })
    result['depth_absdiff_max'] = max(f['depth_absdiff_max'] for f in result['frames'])
    result['depth_absdiff_mean'] = float(np.mean([f['depth_absdiff_mean'] for f in result['frames']]))
    result['normal_absdiff_max'] = max(f['normal_absdiff_max'] for f in result['frames'])

    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

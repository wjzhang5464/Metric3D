"""Step-0 check: the training tree (training/mono) and the inference tree (root mono/) give the same prediction
for the same weights and input, and the checkpoint's keys match each tree's model exactly.

The two trees must never be imported in one process, so this runs twice:
  --tree root   : preprocess N validation frames with the official inference code (as infer_tartanair.py does),
                  run the root-tree model, save inputs and canonical predictions to <out>/root.npz.
  --tree train  : build the training-tree model from a training config, load the same weights, run the saved
                  inputs, compare with root.npz.
Both report missing / unexpected checkpoint keys (the official loaders use strict=False, which hides them).
Use --weights to check a fine-tuned checkpoint the same way (e.g. that it loads into the root tree).

    python -B tartanair_finetune/step0_equivalence.py --tree root  --out <dir> [--config C] [--weights W]
    python -B tartanair_finetune/step0_equivalence.py --tree train --out <dir> --train-config C [--weights W]
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys
import zipfile

import cv2
import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TRAINVAL = '/depot/chan129/data/TartanAir_v2/tartanair_v2_trainval'
DEFAULT_WEIGHTS = osp.join(os.environ.get('RCAC_SCRATCH', ''), 'metric3d/weights/metric_depth_vit_small_800k.pth')


def key_report(model, state_dict):
    mk, ck = set(model.state_dict().keys()), set(state_dict.keys())
    return {'model_keys': len(mk), 'ckpt_keys': len(ck),
            'missing_in_ckpt': sorted(mk - ck), 'unexpected_in_ckpt': sorted(ck - mk)}


def validation_frames(n):
    with open(osp.join(TRAINVAL, 'frames.tsv'), newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['role'] == 'validation']
    step = max(len(rows) // n, 1)
    return rows[::step][:n]  # evenly spread, deterministic


def run_root(a):
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, osp.join(REPO_ROOT, 'tartanair_eval'))
    from infer_tartanair import build_model, read_rgb, DEFAULT_CONFIG
    from mono.utils.do_test import transform_test_data_scalecano
    config = a.config or DEFAULT_CONFIG
    model, cfg = build_model(config, a.weights)
    sd = torch.load(a.weights, map_location='cpu')['model_state_dict']
    rep = key_report(model.module, sd)
    inputs, preds, keys = [], [], []
    with zipfile.ZipFile(osp.join(TRAINVAL, 'validation.zip')) as zf:
        for r in validation_frames(a.n):
            rgb = read_rgb(zf, r['frame_key'])
            intr = [float(r['fx']), float(r['fy']), float(r['cx']), float(r['cy'])]
            x, _, pad, scale = transform_test_data_scalecano(rgb, intr, cfg.data_basic)
            with torch.no_grad():
                pred, _, _ = model.module.inference({'input': x[None].cuda(), 'cam_model': None})
            x = x.cpu()
            inputs.append(x.numpy())
            preds.append(pred[0, 0].float().cpu().numpy())
            keys.append(r['frame_key'])
    np.savez(osp.join(a.out, 'root.npz'), inputs=np.stack(inputs), preds=np.stack(preds), keys=np.array(keys),
             pad=np.array(pad), scale=scale)
    rep.update(config=config, weights=a.weights, frames=len(keys), pad=list(map(int, pad)), label_scale=float(scale),
               pred_canonical_min=float(np.min(preds)), pred_canonical_max=float(np.max(preds)))
    return rep


def run_train(a):
    train_root = osp.join(REPO_ROOT, 'training')
    sys.path.insert(0, train_root)
    os.chdir(train_root)
    try:
        from mmcv.utils import Config
    except ImportError:
        from mmengine import Config
    from mono.model.criterion import build_criterions
    from mono.model.monodepth_model import get_configured_monodepth_model
    cfg = Config.fromfile(a.train_config)
    model = get_configured_monodepth_model(cfg, build_criterions(cfg)).cuda()
    sd = torch.load(a.weights, map_location='cpu')['model_state_dict']
    rep = key_report(model, sd)
    model.load_state_dict(sd, strict=False)
    model.eval()
    ref = np.load(osp.join(a.out, 'root.npz'))
    diffs, rel = [], []
    for x, p_root in zip(ref['inputs'], ref['preds']):
        with torch.no_grad():
            out = model.inference({'input': torch.from_numpy(x)[None].cuda(), 'cam_model': None})
        p = out['prediction'][0, 0].float().cpu().numpy()
        diffs.append(float(np.abs(p - p_root).max()))
        rel.append(float(np.median(np.abs(p - p_root) / p_root)))
    rep.update(train_config=a.train_config, weights=a.weights, frames=len(diffs),
               depth_normalize=list(cfg.data_basic.depth_normalize),
               max_abs_diff_canonical=max(diffs), per_frame_max_abs_diff=diffs, median_rel_diff_max=max(rel))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--tree', choices=['root', 'train'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=8, help='validation frames')
    ap.add_argument('--config', default=None, help='root-tree inference config (default: official v2-S)')
    ap.add_argument('--train-config', default='mono/configs/RAFTDecoder/vit.raft5.small.tartanair.dn200.py',
                    help='training-tree config, relative to training/')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--tag', default='', help='suffix of the report file name')
    a = ap.parse_args()
    a.out = osp.abspath(a.out)
    a.weights = osp.abspath(a.weights)
    os.makedirs(a.out, exist_ok=True)
    rep = run_root(a) if a.tree == 'root' else run_train(a)
    with open(osp.join(a.out, f'report_{a.tree}{a.tag}.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    short = {k: (v if not isinstance(v, list) or len(v) <= 12 else f'{len(v)} items') for k, v in rep.items()}
    print(json.dumps(short, indent=1))


if __name__ == '__main__':
    main()

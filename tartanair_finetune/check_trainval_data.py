"""Integrity and content checks of the TartanAir v2 trainval package (read-only; CPU).

Checks on a seeded sample of training frames plus (by default) every validation frame:
  * rgb.png decodes with cv2 to 640x640x3 uint8;
  * depth.npy is little-endian float16 640x640 and finite;
  * how cv2 / PIL decode the 1-bit masks (values seen), and that both decoders agree;
  * valid_mask == (depth > 1e-3) & (depth <= 200) & ~sky_mask exactly (the README rule);
  * mask pixel counts equal frames.tsv valid_pixels / sky_pixels;
  * depth statistics relevant to fine-tuning (valid pixels beyond the canonical caps, frames without
    valid pixels).

Outputs: <out>/per_frame.tsv and <out>/summary.json.

    python -B tartanair_finetune/check_trainval_data.py --out $RCAC_SCRATCH/metric3d/outputs/data_checks/content
"""
import argparse
import csv
import json
import os
import os.path as osp
import random
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
import io

DATA_ROOT = '/depot/chan129/data/TartanAir_v2/tartanair_v2_trainval'
# Metric caps of the canonical clamp at the training-time label scale 1000/320 = 3.125 (no RandomResize)
# and at the test-time label scale 3.2468 (640 -> 616 resize).
CAPS_M = {'cap_train_dn200': 200 / 3.125, 'cap_test_dn200': 200 / 3.2467532467532463,
          'cap_train_dn700': 700 / 3.125, 'cap_test_dn700': 700 / 3.2467532467532463}
BINS_M = [0, 2, 5, 10, 20, 40, 64, 100, 150, 200]

_zips = {}


def _zip(path):
    if path not in _zips:
        _zips[path] = zipfile.ZipFile(path)
    return _zips[path]


def check_frame(args):
    root, row = args
    zf = _zip(osp.join(root, row['zip']))
    key = row['frame_key']
    out = {'role': row['role'], 'zip': row['zip'], 'frame_key': key, 'env': row['env'], 'errors': []}

    rgb = cv2.imdecode(np.frombuffer(zf.read(key + '/rgb.png'), np.uint8), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape != (640, 640, 3) or rgb.dtype != np.uint8:
        out['errors'].append(f'rgb {None if rgb is None else (rgb.shape, rgb.dtype)}')
    rgb_raw = cv2.imdecode(np.frombuffer(zf.read(key + '/rgb.png'), np.uint8), cv2.IMREAD_UNCHANGED)
    out['rgb_raw_shape'] = 'x'.join(map(str, rgb_raw.shape))

    depth = np.load(io.BytesIO(zf.read(key + '/depth.npy')))
    if depth.dtype != np.dtype('<f2') or depth.shape != (640, 640):
        out['errors'].append(f'depth {depth.dtype} {depth.shape}')
    d = depth.astype(np.float32)
    out['depth_nonfinite'] = int((~np.isfinite(d)).sum())

    masks = {}
    for name in ('valid_mask', 'sky_mask'):
        buf = zf.read(f'{key}/{name}.png')
        m_cv = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
        m_pil = np.array(Image.open(io.BytesIO(buf)))
        out[f'{name}_cv2'] = f'{m_cv.dtype}:' + ','.join(map(str, np.unique(m_cv)))
        out[f'{name}_pil'] = f'{m_pil.dtype}:' + ','.join(map(str, np.unique(m_pil).astype(int)))
        if not np.array_equal(m_cv > 0, m_pil > 0):
            out['errors'].append(f'{name} cv2/PIL disagree')
        masks[name] = m_cv > 0
    valid, sky = masks['valid_mask'], masks['sky_mask']

    rule = (d > 1e-3) & (d <= 200) & ~sky
    out['rule_mismatch_px'] = int((rule != valid).sum())
    out['valid_px'] = int(valid.sum())
    out['sky_px'] = int(sky.sum())
    if out['valid_px'] != int(row['valid_pixels']) or out['sky_px'] != int(row['sky_pixels']):
        out['errors'].append('mask counts differ from frames.tsv')
    if out['rule_mismatch_px']:
        out['errors'].append('valid_mask != README rule')

    v = d[valid]
    out['valid_frac'] = out['valid_px'] / valid.size
    out['valid_max_m'] = float(v.max()) if v.size else float('nan')
    for k, cap in CAPS_M.items():
        out[f'px_above_{k}'] = int((v > cap).sum())
    out['hist'] = np.histogram(v, bins=BINS_M)[0].tolist()
    # Non-sky, non-valid pixels: depth > 200 m (far but not sky) or <= 1e-3
    out['px_far_not_sky'] = int(((d > 200) & ~sky).sum())
    out['px_near_invalid'] = int(((d <= 1e-3) & ~sky).sum())
    out['errors'] = ';'.join(out['errors'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--n-train', type=int, default=3000, help='seeded random sample of training frames')
    ap.add_argument('--n-val', type=int, default=-1, help='-1: all validation frames')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    if osp.abspath(a.out).startswith('/depot/chan129/data'):
        raise SystemExit('refusing to write under /depot/chan129/data')
    os.makedirs(a.out, exist_ok=True)

    with open(osp.join(a.data_root, 'frames.tsv'), newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    rng = random.Random(a.seed)
    train = [r for r in rows if r['role'] == 'train']
    val = [r for r in rows if r['role'] == 'validation']
    sel = rng.sample(train, min(a.n_train, len(train))) + (val if a.n_val < 0 else rng.sample(val, a.n_val))

    with Pool(a.workers) as pool:
        res = pool.map(check_frame, [(a.data_root, r) for r in sel], chunksize=8)

    cols = [k for k in res[0] if k != 'hist']
    with open(osp.join(a.out, 'per_frame.tsv'), 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in res:
            f.write('\t'.join(str(r[c]) for c in cols) + '\n')

    summary = {'seed': a.seed, 'bins_m': BINS_M, 'caps_m': CAPS_M}
    for role in ('train', 'validation'):
        rr = [r for r in res if r['role'] == role]
        if not rr:
            continue
        valid_total = sum(r['valid_px'] for r in rr)
        hist = np.sum([r['hist'] for r in rr], axis=0)
        s = {
            'frames': len(rr),
            'frames_with_errors': sum(bool(r['errors']) for r in rr),
            'error_examples': [f"{r['frame_key']}: {r['errors']}" for r in rr if r['errors']][:10],
            'rule_mismatch_px_total': sum(r['rule_mismatch_px'] for r in rr),
            'depth_nonfinite_total': sum(r['depth_nonfinite'] for r in rr),
            'rgb_raw_shapes': sorted({r['rgb_raw_shape'] for r in rr}),
            'mask_decodings': sorted({f"valid cv2 {r['valid_mask_cv2']} | pil {r['valid_mask_pil']}" for r in rr}
                                     | {f"sky cv2 {r['sky_mask_cv2']} | pil {r['sky_mask_pil']}" for r in rr}),
            'frames_without_valid_px': sum(r['valid_px'] == 0 for r in rr),
            'frames_valid_frac_lt_5pct': sum(r['valid_frac'] < 0.05 for r in rr),
            'mean_valid_frac': float(np.mean([r['valid_frac'] for r in rr])),
            'valid_px_total': valid_total,
            'share_valid_px_above': {k: sum(r[f'px_above_{k}'] for r in rr) / valid_total for k in CAPS_M},
            'frames_with_px_above': {k: sum(r[f'px_above_{k}'] > 0 for r in rr) for k in CAPS_M},
            'valid_px_share_per_bin': dict(zip([f'{lo}-{hi}' for lo, hi in zip(BINS_M[:-1], BINS_M[1:])],
                                               (hist / valid_total).round(5).tolist())),
            'far_not_sky_px_share_of_all': sum(r['px_far_not_sky'] for r in rr) / (len(rr) * 640 * 640),
            'near_invalid_px_share_of_all': sum(r['px_near_invalid'] for r in rr) / (len(rr) * 640 * 640),
            'sky_px_share_of_all': sum(r['sky_px'] for r in rr) / (len(rr) * 640 * 640),
        }
        summary[role] = s
    with open(osp.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

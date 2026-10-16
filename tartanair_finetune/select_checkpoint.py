"""Checkpoint selection on the TartanAir v2 validation split (never the test set), by depth MAE.

For every epoch checkpoint of a fine-tuning run (ckpt/stepXXXXXXXX.pth, one per epoch), runs the 1,944 validation
frames through exactly the delivery inference path (tartanair_eval/infer_tartanair.py: build_model, read_rgb,
predict_raw; GT intrinsics, no alignment, depth before the `< 300` zeroing), and scores each frame with the official
metric functions of the test package's metrics/eval_depth.py (frame_metrics on the frame's valid_mask; the same
float32 cast as when a saved .npy is scored). Score = mean per-frame MAE (each frame weighted equally), as in
eval_depth.py. The lowest mean MAE wins (ties: the earlier epoch). See reports/finetune_v2s_tartanair/PROTOCOL.md.

Resumable: epochs already scored (<out>/epXX.jsonl complete) are skipped, so it can be re-run while training is
still writing checkpoints. --reference also scores the official pretrained weights (step 0; not a candidate).

    python -B tartanair_finetune/select_checkpoint.py --run-dir $RCAC_SCRATCH/metric3d/checkpoints/ft_v2s_ta2_dn700 \
        --config mono/configs/HourglassDecoder/vit.raft5.small.dn700.py --reference

Outputs in <out> (default <run-dir>/val_selection/): epXX.jsonl (per-frame records), summary.tsv (one row per
scored checkpoint), selection.json (best epoch, its checkpoint and sha256; written when all --epochs are scored).
"""
import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import re
import sys
import time
import zipfile

import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, 'tartanair_eval'))
sys.dont_write_bytecode = True  # never write __pycache__ next to eval_depth.py under /depot
from infer_tartanair import (TRAINVAL_ROOT, DATA_ROOT, DEFAULT_WEIGHTS, build_model, load_frames,  # noqa: E402
                             predict_raw, read_rgb, zero_like_official)
from score_pilot import load_eval_module  # noqa: E402

ITERS_PER_EPOCH = 4361  # 69,779 // 16, see training/mono/configs/RAFTDecoder/vit.raft5.small.tartanair.dn200.py
METRICS_SHOWN = ('MAE', 'AbsRel', 'RMSE', 'd1')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def check_keys(model, ckpt_path):
    """The official loader uses strict=False; refuse checkpoints whose keys do not match the model exactly."""
    sd = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    mk, ck = set(model.module.state_dict()), set(sd)
    if mk != ck:
        raise SystemExit(f'{ckpt_path}: key mismatch, missing {sorted(mk - ck)[:5]}, unexpected {sorted(ck - mk)[:5]}')


def score(weights, config, rows, ev, out_jsonl, tag='', every=250):
    model, cfg = build_model(config, weights)
    check_keys(model, weights)
    tmp = out_jsonl + '.part'
    n_zeroing = 0
    t0 = time.time()
    with zipfile.ZipFile(osp.join(TRAINVAL_ROOT, 'validation.zip')) as zf, open(tmp, 'w') as f:
        maes = []
        for i, r in enumerate(rows, 1):
            key = r['frame_key']
            rgb = read_rgb(zf, key)
            intr = [float(r['fx']), float(r['fy']), float(r['cx']), float(r['cy'])]
            depth, _ = predict_raw(model, cfg, rgb, intr)
            n_zeroing += int((zero_like_official(depth) != depth).sum())
            pred = depth.float().cpu().numpy().astype(np.float32)  # exactly what infer_tartanair.py saves
            gt, valid = ev.read_gt(zf, key)
            status, res = ev.frame_metrics(pred, gt, valid)
            rec = {'frame_key': key, 'status': status, ('metrics' if status == 'scored' else 'reason'): res}
            f.write(json.dumps(rec) + '\n')
            if status == 'scored':
                maes.append(res['MAE'])
            if every and (i % every == 0 or i == len(rows)):
                el = time.time() - t0
                print(f'  {tag}: {i}/{len(rows)} frames, running mean MAE {np.mean(maes):.4f} m, '
                      f'{el:.0f} s elapsed, ~{el / i * (len(rows) - i):.0f} s left', flush=True)
    os.replace(tmp, out_jsonl)
    del model
    torch.cuda.empty_cache()
    return n_zeroing, time.time() - t0


def summarize(jsonl, ev):
    with open(jsonl) as f:
        recs = [json.loads(ln) for ln in f]
    return ev.aggregate('validation', recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run-dir', required=True, help='fine-tuning work dir (holds ckpt/)')
    ap.add_argument('--config', required=True, help='root-tree inference config matching the run (dn700 or official)')
    ap.add_argument('--out', default=None, help='default: <run-dir>/val_selection')
    ap.add_argument('--epochs', type=int, default=20, help='candidates are epochs --first-epoch..N')
    ap.add_argument('--first-epoch', type=int, default=1,
                    help='first candidate epoch (12 for the first runs, whose epochs 1-11 were deleted; PROTOCOL.md)')
    ap.add_argument('--iters-per-epoch', type=int, default=ITERS_PER_EPOCH)
    ap.add_argument('--reference', action='store_true',
                    help='also score the official weights under this run\'s --config (= the run at step 0; reference only)')
    ap.add_argument('--limit', type=int, default=None, help='first N validation frames only (testing; no selection)')
    a = ap.parse_args()
    out = a.out or osp.join(a.run_dir, 'val_selection')
    os.makedirs(out, exist_ok=True)
    ev = load_eval_module(DATA_ROOT)  # the test package's official metrics/eval_depth.py
    rows = load_frames(TRAINVAL_ROOT, split='validation', limit=a.limit)
    print(f'{len(rows)} validation frames; config {a.config}', flush=True)

    jobs = []
    if a.reference:
        jobs.append(('ep00_official', 0, DEFAULT_WEIGHTS))
    for e in range(a.first_epoch, a.epochs + 1):
        ck = osp.join(a.run_dir, 'ckpt', 'step%08d.pth' % (e * a.iters_per_epoch))
        jobs.append((f'ep{e:02d}', e, ck))
    unexpected = [p for p in sorted(os.listdir(osp.join(a.run_dir, 'ckpt'))) if re.match(r'step\d+\.pth$', p)
                  and int(p[4:12]) % a.iters_per_epoch] if osp.isdir(osp.join(a.run_dir, 'ckpt')) else []
    if unexpected:
        print(f'note: checkpoints off the epoch grid are ignored: {unexpected}', flush=True)

    summary_path = osp.join(out, 'summary.tsv')
    rows_out = {}
    if osp.exists(summary_path):
        with open(summary_path, newline='') as f:
            rows_out = {r['tag']: r for r in csv.DictReader(f, delimiter='\t')}
    for tag, epoch, ck in jobs:
        jsonl = osp.join(out, f'{tag}.jsonl')
        if tag in rows_out and osp.exists(jsonl) and a.limit is None:
            continue
        if not osp.exists(ck):
            print(f'{tag}: {ck} not written yet, skipped', flush=True)
            continue
        print(f'{tag}: scoring {ck}', flush=True)
        n_zero, secs = score(ck, a.config, rows, ev, jsonl, tag)
        agg = summarize(jsonl, ev)
        rows_out[tag] = {'tag': tag, 'epoch': epoch, 'step': epoch * a.iters_per_epoch, 'checkpoint': ck,
                         'frames': agg['frames'], 'failures': agg['failures'], 'unscorable': agg['unscorable'],
                         **{m: f'{agg[m]:.6f}' for m in ev.METRICS}, 'zeroing_px': n_zero, 'seconds': f'{secs:.0f}'}
        print(f'{tag}: ' + '  '.join(f'{m} {agg[m]:.4f}' for m in METRICS_SHOWN)
              + f'  failures {agg["failures"]}  ({secs:.0f} s)', flush=True)
        if a.limit is None:
            cols = list(rows_out[tag].keys())
            with open(summary_path, 'w') as f:
                f.write('\t'.join(cols) + '\n')
                for r in sorted(rows_out.values(), key=lambda r: int(r['epoch'])):
                    f.write('\t'.join(str(r[c]) for c in cols) + '\n')

    cands = [r for r in rows_out.values() if a.first_epoch <= int(r['epoch']) <= a.epochs]
    n_cand = a.epochs - a.first_epoch + 1
    if a.limit is not None or len(cands) < n_cand:
        print(f'{len(cands)}/{n_cand} candidate epochs scored; selection is written once all are scored', flush=True)
        return
    if any(int(r['failures']) for r in cands):
        raise SystemExit('some validation frames failed scoring; see summary.tsv')
    best = min(cands, key=lambda r: (float(r['MAE']), int(r['epoch'])))
    sel = {'criterion': 'mean per-frame MAE on the validation split (eval_depth.py definition), lowest wins',
           'best_epoch': int(best['epoch']), 'best_step': int(best['step']), 'best_MAE': float(best['MAE']),
           'checkpoint': best['checkpoint'], 'checkpoint_sha256': sha256(best['checkpoint']),
           'inference_config': a.config,
           'candidate_epochs': [a.first_epoch, a.epochs],
           'best_at_lower_boundary': int(best['epoch']) == a.first_epoch and a.first_epoch > 1,
           'all_epochs_MAE': {int(r['epoch']): float(r['MAE']) for r in sorted(cands, key=lambda r: int(r['epoch']))}}
    if 'ep00_official' in rows_out:
        sel['reference_official_weights_MAE'] = float(rows_out['ep00_official']['MAE'])
    if a.first_epoch > 1:
        sel['note'] = (f'epochs 1-{a.first_epoch - 1} are not available (deleted by the upstream checkpoint '
                       'retention; PROTOCOL.md incident 2026-10-15). Pre-registered rule: if the best epoch is the '
                       'lower boundary, the selection is not accepted and the run is repeated with all epochs kept.')
    with open(osp.join(out, 'selection.json'), 'w') as f:
        json.dump(sel, f, indent=1)
    print(json.dumps(sel, indent=1), flush=True)
    if sel.get('best_at_lower_boundary'):
        print(f'WARNING: best epoch = first candidate epoch {a.first_epoch}; by the pre-registered rule this '
              'selection is NOT accepted (an earlier epoch may be better).', flush=True)


if __name__ == '__main__':
    main()

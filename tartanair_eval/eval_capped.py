"""DIAGNOSTIC metrics: eval_depth.py's own functions with the valid mask further restricted to GT <= the model's
depth cap. Measures what the cap costs; never replaces the PRIMARY (official, unmodified eval_depth.py) numbers.

Per frame:  mask = official valid_mask & (float32 GT <= cap), cap derived from the official code + intrinsics.
Writes, like eval_depth.py:  <out>/<subset>.json  and  <out>/<subset>_per_frame.jsonl
Each subset json / per-frame record also carries a "diagnostic" block (computed on the OFFICIAL mask):
frac_valid_gt_above_cap, median_scale (median(gt)/median(pred)), AbsRel_median_aligned.

With --official-dir, also writes <out>/all_frames_ours.json: frame-weighted means over all scored frames of all
subsets for PRIMARY (from eval_depth.py's per-frame jsonl) and DIAGNOSTIC. These pooled rows are ours, not official.

    python -B tartanair_eval/eval_capped.py --pred <F>/pred --out <F>/eval_capped --official-dir <F>/eval_official
"""
import sys

sys.dont_write_bytecode = True  # never write __pycache__ next to eval_depth.py under /depot

import argparse
import csv
import json
import multiprocessing as mp
import os
import os.path as osp
import zipfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from score_pilot import DATA_ROOT, DEFAULT_CONFIG, load_eval_module, depth_cap  # noqa: E402

SUBSETS = ('test_matched', 'test_scene', 'pose_combination', 'pose_extrapolation',
           'height_holdout', 'combination_height', 'extrapolation_height')


def intrinsics_by_key(data_root):
    with open(osp.join(data_root, 'frames.tsv'), newline='') as f:
        return {r['frame_key']: (float(r['fx']), float(r['fy']), float(r['cx']), float(r['cy']))
                for r in csv.DictReader(f, delimiter='\t')}


def eval_subset(job):
    """Runs in a worker process (no CUDA). job = (data_root, pred_dir, subset, caps_by_key)."""
    data_root, pred_dir, subset, caps = job
    ev = load_eval_module(data_root)
    records = []
    with zipfile.ZipFile(osp.join(data_root, subset + '.zip')) as zf:
        for key in ev.frame_keys(zf):
            gt16, valid = ev.read_gt(zf, key)
            pred = ev.load_prediction(osp.join(pred_dir, subset, key + '.npy'))
            gt = gt16.astype(np.float32)
            cap = caps[key]
            capped = valid & (gt <= cap)
            status, res = ev.frame_metrics(pred, gt16, capped)
            rec = {'frame_key': key, 'status': status}
            rec['metrics' if status == 'scored' else 'reason'] = res

            diag = {'cap_m': cap, 'n_valid_official': int(valid.sum()), 'n_valid_capped': int(capped.sum()),
                    'officially_unscorable': not valid.any()}
            if valid.any():
                diag['frac_valid_gt_above_cap'] = float(1.0 - capped.sum() / valid.sum())
                if ev.check_prediction(pred, valid) is None:  # extras need a pred valid on the official mask
                    p = pred.astype(np.float32)
                    scale = float(np.median(gt[valid]) / np.median(p[valid]))
                    diag['median_scale'] = scale
                    diag['AbsRel_median_aligned'] = ev.compute_metrics(p * scale, gt, valid)['AbsRel']
            rec['diagnostic'] = diag
            records.append(rec)

    out = ev.aggregate(subset, records)
    diag_frames = [r['diagnostic'] for r in records]
    n_valid = sum(d['n_valid_official'] for d in diag_frames)
    scales = [d['median_scale'] for d in diag_frames if 'median_scale' in d]
    med_ar = [d['AbsRel_median_aligned'] for d in diag_frames if 'AbsRel_median_aligned' in d]
    out['diagnostic'] = {
        'note': 'DIAGNOSTIC (ours): mask = official valid & GT <= cap. Not the official metric.',
        'cap_m': sorted({d['cap_m'] for d in diag_frames}),
        'frac_valid_pixels_above_cap': float(1.0 - sum(d['n_valid_capped'] for d in diag_frames) / n_valid) if n_valid else float('nan'),
        'unscorable_due_to_cap': [r['frame_key'] for r in records
                                  if r['status'] == 'unscorable' and not r['diagnostic']['officially_unscorable']],
        'median_of_median_scale': float(np.median(scales)) if scales else float('nan'),
        'mean_AbsRel_median_aligned_official_mask': float(np.mean(med_ar)) if med_ar else float('nan'),
        'frames_with_median_extras': len(scales),
    }
    return subset, out, records


def pooled_row(per_frame_paths, metrics):
    """Frame-weighted mean over all scored frames in the given per-frame jsonl files (ours, not official)."""
    recs = [json.loads(ln) for p in per_frame_paths for ln in open(p)]
    scored = [r['metrics'] for r in recs if r['status'] == 'scored']
    row = {'frames_total': len(recs), 'frames': len(scored),
           'unscorable': sum(r['status'] == 'unscorable' for r in recs),
           'failures': sum(r['status'] == 'failure' for r in recs)}
    row.update({m: float(np.mean([s[m] for s in scored])) for m in metrics})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--official-dir', default=None, help='eval_depth.py --per-frame output, for the pooled PRIMARY row')
    ap.add_argument('--subsets', nargs='+', choices=SUBSETS, default=list(SUBSETS))
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    if osp.abspath(a.out).startswith('/depot/chan129/data'):
        raise SystemExit('refusing to write under /depot/chan129/data')
    os.makedirs(a.out, exist_ok=True)

    # cap per distinct intrinsics, from the official transform (needs CUDA; done once in the parent)
    intr = intrinsics_by_key(a.data_root)
    cap_of = {k: depth_cap(a.config, intrinsic=k)['cap_m'] for k in set(intr.values())}
    caps = {key: cap_of[k] for key, k in intr.items()}
    print('caps by intrinsics:', cap_of, flush=True)

    ev = load_eval_module(a.data_root)
    jobs = [(a.data_root, a.pred, s, caps) for s in a.subsets]
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs))) as pool:
        for subset, out, records in pool.imap_unordered(eval_subset, jobs):
            with open(osp.join(a.out, subset + '.json'), 'w') as f:
                json.dump(out, f, indent=2)
            with open(osp.join(a.out, subset + '_per_frame.jsonl'), 'w') as f:
                for r in records:
                    f.write(json.dumps(r) + '\n')
            print(f"{subset}: {out['frames']}/{out['frames_total']} scored, unscorable {out['unscorable']} "
                  f"(due to cap {len(out['diagnostic']['unscorable_due_to_cap'])}), failures {out['failures']}, "
                  f"AbsRel {out['AbsRel']:.4f}, d1 {out['d1']:.4f}", flush=True)

    pooled = {'note': 'All-frames rows are OURS (frame-weighted mean over all scored frames of the listed subsets), '
                      'computed from per-frame files; they are not produced by eval_depth.py.',
              'subsets': a.subsets,
              'DIAGNOSTIC_capped': pooled_row([osp.join(a.out, s + '_per_frame.jsonl') for s in a.subsets], ev.METRICS)}
    if a.official_dir:
        pooled['PRIMARY_official'] = pooled_row(
            [osp.join(a.official_dir, s + '_per_frame.jsonl') for s in a.subsets], ev.METRICS)
    with open(osp.join(a.out, 'all_frames_ours.json'), 'w') as f:
        json.dump(pooled, f, indent=2)
    print(json.dumps(pooled, indent=2))


if __name__ == '__main__':
    main()

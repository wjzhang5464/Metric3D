"""Score a subset of TartanAir v2 test frames with the dataset's own eval_depth.py, plus pilot diagnostics.

Only the frames in --keys-file are scored (eval_depth.py's CLI would count every other frame of a subset as a
failure). Metric functions are imported from eval_depth.py by path, unmodified. Adds per frame:
median-aligned AbsRel (pred * median(gt)/median(pred) over valid pixels), the effective depth cap derived
from the official code, the fraction of valid GT above it, 5-panel visuals and contact sheets.

    python -B tartanair_eval/score_pilot.py --keys-file <keys.txt> --run-dir <dir with pred/ normals/ timing.tsv>
"""
import argparse
import csv
import importlib.util
import json
import os
import os.path as osp
import re
import sys
import zipfile

sys.dont_write_bytecode = True  # never write __pycache__ next to eval_depth.py under /depot

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from infer_tartanair import DATA_ROOT, DEFAULT_CONFIG, SUBSETS, Config, read_rgb  # noqa: E402
from mono.utils.do_test import transform_test_data_scalecano  # noqa: E402

FULL_TEST_FRAMES = 10868


def load_eval_module(data_root):
    path = osp.join(data_root, 'metrics', 'eval_depth.py')
    spec = importlib.util.spec_from_file_location('eval_depth', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def depth_cap(config_path, h=640, w=640, intrinsic=(320.0, 320.0, 320.0, 320.0)):
    """Max metric depth the pipeline can output: decoder clamp max (canonical) * normalize_scale / label_scale_factor,
    with the label scale factor taken from the official transform_test_data_scalecano."""
    cfg = Config.fromfile(config_path)
    _, _, pad, lsf = transform_test_data_scalecano(np.zeros((h, w, 3), np.uint8), list(intrinsic), cfg.data_basic)
    clamp_max = cfg.data_basic.depth_normalize[1]  # RAFTDepthNormalDPT5.max_val
    normalize_scale = cfg.data_basic.depth_range[1]
    return {'label_scale_factor': float(lsf), 'canonical_clamp_max': float(clamp_max),
            'normalize_scale': float(normalize_scale), 'pad': [int(p) for p in pad],
            'cap_m': float(clamp_max * normalize_scale / lsf)}


def make_figure(path, rgb, gt, valid, pred, normal, cap, title):
    vmax = float(np.percentile(gt[valid], 98)) if valid.any() else float(pred.max())
    depth_cmap = plt.get_cmap('turbo').copy()
    depth_cmap.set_bad('0.5')
    err_cmap = plt.get_cmap('magma').copy()
    err_cmap.set_bad('0.5')
    gt_m = np.ma.masked_where(~valid, gt)
    err = np.ma.masked_where(~valid, np.abs(pred - gt) / np.where(valid, gt, 1.0))
    above = valid & (gt > cap)

    fig, axs = plt.subplots(1, 5, figsize=(25, 5.6))
    axs[0].imshow(rgb)
    axs[0].set_title('RGB')
    im = axs[1].imshow(gt_m, cmap=depth_cmap, vmin=0, vmax=vmax)
    if above.any():
        axs[1].contour(above.astype(float), levels=[0.5], colors='white', linewidths=1.0)
    axs[1].set_title(f'GT depth (gray=invalid, white contour: GT>{cap:.1f} m, {above.sum() / max(valid.sum(), 1):.1%})')
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02, label='m')
    im = axs[2].imshow(pred, cmap=depth_cmap, vmin=0, vmax=vmax)
    axs[2].set_title(f'Pred depth (max {pred.max():.1f} m)')
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.02, label='m')
    im = axs[3].imshow(err, cmap=err_cmap, vmin=0, vmax=0.5)
    axs[3].set_title('AbsRel |p-g|/g (gray=invalid, clipped 0.5)')
    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.02)
    axs[4].imshow(np.clip((normal.transpose(1, 2, 0) + 1) / 2, 0, 1))
    axs[4].set_title('Pred normals ((n+1)/2)')
    for ax in axs:
        ax.axis('off')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def contact_sheet(paths, out_path, cols, thumb_w):
    ims = [Image.open(p).convert('RGB') for p in paths]
    ims = [im.resize((thumb_w, int(im.height * thumb_w / im.width)), Image.LANCZOS) for im in ims]
    th = max(im.height for im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * thumb_w, rows * th), 'white')
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * thumb_w, (i // cols) * th))
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys-file', required=True)
    ap.add_argument('--run-dir', required=True, help='holds pred/, normals/, timing.tsv (from infer_tartanair.py)')
    ap.add_argument('--vis-dir', default=None, help='where figures and sheets go (default: <run-dir>/vis)')
    ap.add_argument('--sheet-prefix', default='sheet_', help='contact sheets are <vis-dir>/<prefix><subset>.png and <prefix>all.png')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    a = ap.parse_args()

    ev = load_eval_module(a.data_root)
    cap = depth_cap(a.config)
    with open(a.keys_file) as f:
        keys = [ln.rstrip('\n').split('\t') for ln in f if ln.strip()]
    with open(osp.join(a.run_dir, 'timing.tsv'), newline='') as f:
        timing = {r['frame_key']: r for r in csv.DictReader(f, delimiter='\t')}
    vis_dir = a.vis_dir or osp.join(a.run_dir, 'vis')

    frames, zips = [], {}
    for subset, key in keys:
        if subset not in zips:
            zips[subset] = zipfile.ZipFile(osp.join(a.data_root, subset + '.zip'))
        zf = zips[subset]
        gt16, valid = ev.read_gt(zf, key)
        pred = ev.load_prediction(osp.join(a.run_dir, 'pred', subset, key + '.npy'))
        status, res = ev.frame_metrics(pred, gt16, valid)
        rec = {'subset': subset, 'frame_key': key, 'status': status, 'n_valid': int(valid.sum())}
        if status != 'scored':
            rec['reason'] = res
            frames.append(rec)
            continue
        gt = gt16.astype(np.float32)
        p = pred.astype(np.float32)
        scale = float(np.median(gt[valid]) / np.median(p[valid]))
        med = ev.compute_metrics(p * scale, gt, valid)
        above = valid & (gt > cap['cap_m'])
        rec.update(metrics=res, median_scale=scale, AbsRel_median_aligned=med['AbsRel'],
                   frac_valid_gt_above_cap=float(above.sum() / valid.sum()), n_valid_gt_above_cap=int(above.sum()),
                   gt_valid_max=float(gt[valid].max()), pred_max=float(p.max()), pred_min=float(p.min()),
                   pred_valid_at_cap=float((p[valid] >= cap['cap_m'] * (1 - 1e-6)).mean()))
        if key in timing:
            rec['timing_s'] = {k: float(timing[key][k]) for k in ('read_s', 'infer_s', 'save_s', 'total_s')}

        normal = np.load(osp.join(a.run_dir, 'normals', subset, key + '.npy')).astype(np.float32)
        rgb = read_rgb(zf, key)
        os.makedirs(osp.join(vis_dir, subset), exist_ok=True)
        rec['vis'] = osp.join(vis_dir, subset, key + '.png')
        title = (f"{subset} | {key} | AbsRel {res['AbsRel']:.3f}  d1 {res['d1']:.3f}  "
                 f"median-aligned AbsRel {med['AbsRel']:.3f} (scale x{scale:.2f})")
        make_figure(rec['vis'], rgb, gt, valid, p, normal, cap['cap_m'], title)
        frames.append(rec)
    for zf in zips.values():
        zf.close()

    scored = [f for f in frames if f['status'] == 'scored']
    names = list(ev.METRICS) + ['AbsRel_median_aligned']

    def mean_of(fs):
        out = {m: float(np.mean([f['metrics'][m] for f in fs])) for m in ev.METRICS}
        out['AbsRel_median_aligned'] = float(np.mean([f['AbsRel_median_aligned'] for f in fs]))
        out['frames'] = len(fs)
        return out

    per_subset = {s: mean_of([f for f in scored if f['subset'] == s]) for s in SUBSETS
                  if any(f['subset'] == s for f in scored)}

    # contact sheets
    sheets = {}
    for s in per_subset:
        ps = [f['vis'] for f in scored if f['subset'] == s]
        sheets[s] = osp.join(vis_dir, f'{a.sheet_prefix}{s}.png')
        contact_sheet(ps, sheets[s], cols=1, thumb_w=2000)
    sheets['all'] = osp.join(vis_dir, f'{a.sheet_prefix}all.png')
    contact_sheet([f['vis'] for f in scored], sheets['all'], cols=3, thumb_w=1000)

    # timing: first frame includes CUDA warm-up, so exclude it from the steady-state rate
    t_all = [f['timing_s'] for f in scored if 'timing_s' in f]
    steady = t_all[1:] if len(t_all) > 1 else t_all
    model_load_s = None
    log_path = osp.join(a.run_dir, 'infer_stdout.log')
    if osp.exists(log_path):
        m = re.search(r'model ready in ([\d.]+)s', open(log_path).read())
        model_load_s = float(m.group(1)) if m else None
    mean_total = float(np.mean([t['total_s'] for t in steady]))
    timing_summary = {
        'frames_timed': len(t_all), 'first_frame_total_s': t_all[0]['total_s'] if t_all else None,
        'steady_mean_s': {k: float(np.mean([t[k] for t in steady])) for k in ('read_s', 'infer_s', 'save_s', 'total_s')},
        'model_load_s': model_load_s,
        'full_run_estimate': {
            'frames': FULL_TEST_FRAMES,
            'estimate_s': FULL_TEST_FRAMES * mean_total + (model_load_s or 0) + (t_all[0]['total_s'] if t_all else 0),
            'basis': 'steady-state mean total_s per frame (read+infer+save, batch 1) x frames + model load + warm-up; '
                     'excludes Python/CUDA start-up and job queueing'},
    }
    timing_summary['full_run_estimate']['estimate_min'] = timing_summary['full_run_estimate']['estimate_s'] / 60

    all_valid = sum(f['n_valid'] for f in scored)
    cap_stats = dict(cap)
    cap_stats.update(
        pred_max_observed=float(max(f['pred_max'] for f in scored)),
        frames_with_pred_at_cap=sum(f['pred_max'] >= cap['cap_m'] * (1 - 1e-6) for f in scored),
        frac_valid_gt_above_cap_overall_pixels=float(sum(f['n_valid_gt_above_cap'] for f in scored) / all_valid),
        frac_valid_gt_above_cap_mean_of_frames=float(np.mean([f['frac_valid_gt_above_cap'] for f in scored])),
        frames_with_gt_above_cap=sum(f['n_valid_gt_above_cap'] > 0 for f in scored),
        per_frame={f['frame_key']: f['frac_valid_gt_above_cap'] for f in scored},
    )

    summary = {
        'model': 'Metric3D v2-S (metric_depth_vit_small_800k.pth), zero-shot',
        'config': a.config, 'keys_file': a.keys_file, 'run_dir': a.run_dir,
        'eval_script': osp.join(a.data_root, 'metrics', 'eval_depth.py'),
        'frames': len(frames), 'scored': len(scored),
        'not_scored': [{k: f[k] for k in ('subset', 'frame_key', 'status', 'reason')} for f in frames if f['status'] != 'scored'],
        'overall_mean_frame_weighted': mean_of(scored),
        'per_subset_mean': per_subset,
        'cap_stats': cap_stats,
        'timing': timing_summary,
        'visuals': {'per_frame_dir': vis_dir, 'sheets': sheets},
        'per_frame': frames,
    }
    out = osp.join(a.run_dir, 'pilot_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"scored {len(scored)}/{len(frames)}")
    print(f"{'subset':22s} {'n':>2s} {'AbsRel':>7s} {'d1':>6s} {'RMSE':>7s} {'medAbsRel':>9s}")
    for s, m in list(per_subset.items()) + [('ALL', summary['overall_mean_frame_weighted'])]:
        print(f"{s:22s} {m['frames']:2d} {m['AbsRel']:7.4f} {m['d1']:6.4f} {m['RMSE']:7.3f} {m['AbsRel_median_aligned']:9.4f}")
    print('cap:', {k: v for k, v in cap_stats.items() if k != 'per_frame'})
    print('timing:', json.dumps(timing_summary))
    print('summary:', out)


if __name__ == '__main__':
    main()

"""DIAGNOSTIC field-of-view experiment: does a narrower FoV bring the scale ratio median(GT)/median(pred) closer to 1?

Conditions, all through infer_tartanair.predict with correctly adjusted intrinsics [fx, fy, cx, cy]:
  C0 full      640x640, [320, 320, 320, 320]                 (90 deg FoV; must equal the Stage 2 prediction)
  C1 crop448   centre crop 448x448, [320, 320, 224, 224]     (~70 deg)
  C2 crop320   centre crop 320x320, [320, 320, 160, 160]     (~53 deg)
  C3 resize    full image resized to 320x320 (INTER_AREA), [160, 160, 160, 160]  (90 deg; same down/up path as C2)
All are scored on the SAME pixels: the central 320x320 region of the official valid mask (C0/C1 predictions cropped,
C3 prediction bilinearly upsampled to 640 then cropped). Secondary score: that region & GT <= the C0 cap (61.6 m),
which removes the effect of the per-condition depth cap.

    python -B tartanair_eval/fov_experiment.py --keys-file <keys> --run-dir <F> --out <S3>/fov --fig <png>
"""
import sys

sys.dont_write_bytecode = True  # never write __pycache__ next to eval_depth.py under /depot

import argparse
import csv
import json
import math
import os
import os.path as osp
import time
import zipfile

import cv2
import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from infer_tartanair import build_model, predict, read_rgb, DEFAULT_WEIGHTS  # noqa: E402
from score_pilot import DATA_ROOT, DEFAULT_CONFIG, load_eval_module, depth_cap  # noqa: E402
from eval_capped import SUBSETS  # noqa: E402
from analyze_full import write_csv  # noqa: E402

R = slice(160, 480)  # central 320x320 region of the 640x640 frame
CONDS = {
    'C0_full':     dict(crop=None, size=640, intr=[320.0, 320.0, 320.0, 320.0]),
    'C1_crop448':  dict(crop=(96, 544), size=448, intr=[320.0, 320.0, 224.0, 224.0]),
    'C2_crop320':  dict(crop=(160, 480), size=320, intr=[320.0, 320.0, 160.0, 160.0]),
    'C3_resize320': dict(crop=None, size=320, intr=[160.0, 160.0, 160.0, 160.0], resize=True),
}


def run_condition(model, cfg, rgb, name):
    c = CONDS[name]
    if c.get('resize'):
        img = cv2.resize(rgb, (c['size'], c['size']), interpolation=cv2.INTER_AREA)
    elif c['crop']:
        lo, hi = c['crop']
        img = np.ascontiguousarray(rgb[lo:hi, lo:hi])
    else:
        img = rgb
    depth, _ = predict(model, cfg, img, c['intr'])
    if name == 'C3_resize320':  # back to 640 with the same bilinear op the official post-processing uses
        depth = torch.nn.functional.interpolate(depth[None, None], [640, 640], mode='bilinear').squeeze()
    full = depth.float().cpu().numpy()
    if name == 'C0_full' or name == 'C3_resize320':
        return full, full[R, R]
    if name == 'C1_crop448':
        return full, full[64:384, 64:384]  # 448 crop starts at 96 -> central 160..480 is 64..384
    return full, full  # C2 already is the central region


def region_scores(ev, p, g, m):
    if not m.any():
        return None
    scale = float(np.median(g[m]) / np.median(p[m]))
    met = ev.compute_metrics(p, g, m)
    return {'scale': scale, 'AbsRel': met['AbsRel'], 'd1': met['d1'],
            'AbsRel_median_aligned': ev.compute_metrics(p * scale, g, m)['AbsRel']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys-file', required=True)
    ap.add_argument('--run-dir', required=True, help='Stage 2 dir with pred/ (C0 reference)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fig', required=True, help='example figure path (3 frames: GT | C0 | C1 | C2 | C3)')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    a = ap.parse_args()
    if osp.abspath(a.out).startswith('/depot/chan129/data'):
        raise SystemExit('refusing to write under /depot/chan129/data')
    os.makedirs(osp.join(a.out, 'crops'), exist_ok=True)

    ev = load_eval_module(a.data_root)
    caps = {n: depth_cap(a.config, c['size'], c['size'], c['intr'])['cap_m'] for n, c in CONDS.items()}
    cap0 = caps['C0_full']
    print('caps (m):', caps, flush=True)
    with open(a.keys_file) as f:
        keys = [ln.rstrip('\n').split('\t') for ln in f if ln.strip()]

    rows, zips, t0, c0_identical = [], {}, time.time(), []
    model, cfg = build_model(a.config, a.weights)
    for n, (subset, key) in enumerate(keys, 1):
        if subset not in zips:
            zips[subset] = zipfile.ZipFile(osp.join(a.data_root, subset + '.zip'))
        zf = zips[subset]
        rgb = read_rgb(zf, key)
        gt16, valid = ev.read_gt(zf, key)
        g, m = gt16.astype(np.float32)[R, R], valid[R, R]
        m_cap = m & (g <= cap0)
        row = {'subset': subset, 'frame_key': key, 'env': key.split('__')[0], 'n_valid_region': int(m.sum()),
               'n_valid_region_capped': int(m_cap.sum())}
        crops = {'rgb': rgb[R, R], 'gt': gt16[R, R], 'valid': m}
        for name in CONDS:
            full, p = run_condition(model, cfg, rgb, name)
            if name == 'C0_full':
                ref = np.load(osp.join(a.run_dir, 'pred', subset, key + '.npy'))
                c0_identical.append(bool(np.array_equal(full, ref)))
            crops[name] = p.astype(np.float16)
            for tag, mask in (('', m), ('capped_', m_cap)):
                sc = region_scores(ev, p, g, mask)
                for k in ('scale', 'AbsRel', 'd1', 'AbsRel_median_aligned'):
                    row[f'{tag}{name}_{k}'] = sc[k] if sc else float('nan')
        np.savez_compressed(osp.join(a.out, 'crops', key + '.npz'), **crops)
        rows.append(row)
        if n % 25 == 0 or n == len(keys):
            print(f'[{n}/{len(keys)}] {time.time() - t0:.0f}s, C0 bit-identical so far: {sum(c0_identical)}/{len(c0_identical)}', flush=True)
    for zf in zips.values():
        zf.close()

    names = list(CONDS)
    write_csv(osp.join(a.out, 'fov_per_frame.csv'), rows, list(rows[0]))

    def summarize(rs, tag):
        out = []
        ok = [r for r in rs if r['n_valid_region' + ('_capped' if tag else '')] > 0]
        for name in names:
            s = np.array([r[f'{tag}{name}_scale'] for r in ok])
            s0 = np.array([r[f'{tag}C0_full_scale'] for r in ok])
            out.append({'mask': 'central320 & GT<=%.1f m' % cap0 if tag else 'central320', 'condition': name,
                        'cap_m': caps[name], 'frames': len(ok),
                        'median_scale_ratio': float(np.median(s)),
                        'median_abs_log_scale': float(np.median(np.abs(np.log(s)))),
                        'mean_AbsRel': float(np.mean([r[f'{tag}{name}_AbsRel'] for r in ok])),
                        'mean_d1': float(np.mean([r[f'{tag}{name}_d1'] for r in ok])),
                        'mean_AbsRel_median_aligned': float(np.mean([r[f'{tag}{name}_AbsRel_median_aligned'] for r in ok])),
                        'frac_toward_1_vs_C0': float(np.mean(np.abs(np.log(s)) < np.abs(np.log(s0)))) if name != 'C0_full' else float('nan'),
                        'frac_scale_gt_1': float(np.mean(s > 1))})
        return out

    summary = summarize(rows, '') + summarize(rows, 'capped_')
    write_csv(osp.join(a.out, 'fov_summary.csv'), summary, list(summary[0]))
    per_subset = []
    for s in SUBSETS:
        rs = [r for r in rows if r['subset'] == s and r['n_valid_region'] > 0]
        per_subset.append({'subset': s, 'frames': len(rs), **{f'{n}_median_scale': float(np.median([r[f'{n}_scale'] for r in rs])) for n in names}})
    write_csv(osp.join(a.out, 'fov_per_subset.csv'), per_subset, list(per_subset[0]))

    # paired comparisons on the same frames: is condition A's ratio closer to 1 than B's? (central320 mask)
    ok_pairs = [r for r in rows if r['n_valid_region'] > 0]
    rng = np.random.default_rng(0)
    paired = []
    for A, B in (('C1_crop448', 'C0_full'), ('C2_crop320', 'C0_full'), ('C3_resize320', 'C0_full'), ('C2_crop320', 'C3_resize320')):
        la = np.log(np.array([r[f'{A}_scale'] for r in ok_pairs]))
        lb = np.log(np.array([r[f'{B}_scale'] for r in ok_pairs]))
        d = la - lb  # change in log scale ratio (negative = ratio decreased)
        dabs = np.abs(la) - np.abs(lb)  # negative = closer to 1
        boot = np.median(d[rng.integers(0, len(d), (2000, len(d)))], axis=1)
        paired.append({'A': A, 'B': B, 'frames': len(d), 'frac_A_closer_to_1': float(np.mean(dabs < 0)),
                       'median_delta_log_scale': float(np.median(d)),
                       'median_delta_log_scale_ci95_lo': float(np.percentile(boot, 2.5)),
                       'median_delta_log_scale_ci95_hi': float(np.percentile(boot, 97.5)),
                       'median_ratio_factor_A_over_B': float(np.exp(np.median(d)))})
    write_csv(osp.join(a.out, 'fov_paired.csv'), paired, list(paired[0]))
    with open(osp.join(a.out, 'fov_summary.json'), 'w') as f:
        json.dump({'note': 'DIAGNOSTIC FoV experiment (ours).', 'frames': len(rows), 'envs': len({r['env'] for r in rows}),
                   'caps_m': caps, 'C0_bit_identical_to_stage2': f'{sum(c0_identical)}/{len(c0_identical)}',
                   'summary': summary, 'paired': paired, 'per_subset': per_subset}, f, indent=2)

    # ---- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r['n_valid_region'] > 0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
    data = [[r[f'{n}_scale'] for r in ok] for n in names]
    axs[0].boxplot(data, tick_labels=names, showfliers=False)
    for i, d in enumerate(data, 1):
        axs[0].scatter(np.random.default_rng(i).normal(i, 0.06, len(d)), d, s=3, alpha=0.3)
    axs[0].axhline(1.0, color='k', ls=':')
    axs[0].set_yscale('log'); axs[0].set_ylabel('median(GT)/median(pred), central 320x320')
    axs[0].set_title(f'Scale ratio per condition (n={len(ok)} frames)')
    bins = np.logspace(np.log10(0.1), np.log10(30), 60)
    for i, (n, d) in enumerate(zip(names, data)):
        axs[1].hist(np.clip(d, bins[0], bins[-1]), bins=bins, histtype='step', lw=2, color=f'C{i}',
                    label=f'{n} (median {np.median(d):.2f})')
    axs[1].axvline(1.0, color='k', ls=':'); axs[1].set_xscale('log'); axs[1].legend()
    axs[1].set_xlabel('scale ratio')
    fig.suptitle('FoV experiment: C0 90deg | C1 ~70deg crop | C2 ~53deg crop | C3 90deg resize-only control. DIAGNOSTIC')
    fig.tight_layout()
    fig.savefig(osp.join(a.out, 'fov_ratio_distribution.png'), dpi=90)
    plt.close(fig)

    # examples: frames at the 25th/50th/75th percentile of C0 scale ratio among frames with >= 50% valid region
    cand = sorted([r for r in ok if r['n_valid_region'] >= 0.5 * 320 * 320], key=lambda r: r['C0_full_scale'])
    picks = [cand[int(q * (len(cand) - 1))] for q in (0.25, 0.5, 0.75)]
    cmap = plt.get_cmap('turbo').copy(); cmap.set_bad('0.5')
    fig, axs = plt.subplots(3, 6, figsize=(24, 12.5))
    for row_ax, r in zip(axs, picks):
        c = np.load(osp.join(a.out, 'crops', r['frame_key'] + '.npz'))
        gv, mv = c['gt'].astype(np.float32), c['valid']
        vmax = float(np.percentile(gv[mv], 98))
        row_ax[0].imshow(c['rgb']); row_ax[0].set_title(f"{r['subset']}\n{r['frame_key']}", fontsize=9)
        row_ax[1].imshow(np.ma.masked_where(~mv, gv), cmap=cmap, vmin=0, vmax=vmax); row_ax[1].set_title('GT (central 320x320)')
        for ax, n in zip(row_ax[2:], names):
            im = ax.imshow(c[n].astype(np.float32), cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f"{n}: scale x{r[f'{n}_scale']:.2f}, AbsRel {r[f'{n}_AbsRel']:.3f}", fontsize=10)
        fig.colorbar(im, ax=row_ax[-1], fraction=0.046, pad=0.02, label='m')
        for ax in row_ax:
            ax.axis('off')
    fig.suptitle('FoV experiment examples (25th / 50th / 75th percentile of C0 scale ratio); shared colour range per row', fontsize=13)
    fig.tight_layout()
    os.makedirs(osp.dirname(osp.abspath(a.fig)), exist_ok=True)
    fig.savefig(a.fig, dpi=70)
    plt.close(fig)

    print(f"C0 bit-identical to Stage 2: {sum(c0_identical)}/{len(c0_identical)}")
    for s in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    for p_ in paired:
        print('paired', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p_.items()})
    print('per subset median scale:', [(p['subset'], *[round(p[f'{n}_median_scale'], 3) for n in names]) for p in per_subset])


if __name__ == '__main__':
    main()

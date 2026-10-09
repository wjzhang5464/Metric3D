"""DIAGNOSTIC error analysis of the full zero-shot run (CPU only; reads existing predictions and per-frame files).

  1. grouped tables (environment, difficulty, face, |pitch|, |roll|, subset) from PRIMARY per-frame metrics plus the
     diagnostic extras of eval_capped.py (median scale ratio, median-aligned AbsRel)
  2. per-frame median scale ratio: histograms, scatter vs per-frame median GT depth, Spearman correlation
  3. pixel-level error vs GT depth bins, pooled over all frames on the official valid mask
  4. mean frame luminance vs AbsRel / median-aligned AbsRel; dark frames (< 30/255) vs the rest
Optionally (--extremes-out) writes the 2 best + 2 worst frames per subset by PRIMARY AbsRel (>= 20% valid pixels).

    python -B tartanair_eval/analyze_full.py --run-dir <F> --out <S3>/analysis [--extremes-out <keys.txt>]
"""
import sys

sys.dont_write_bytecode = True  # never write __pycache__ next to eval_depth.py under /depot

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import zipfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from score_pilot import DATA_ROOT, DEFAULT_CONFIG, load_eval_module, depth_cap  # noqa: E402
from eval_capped import SUBSETS  # noqa: E402

LOG_RATIO_LIM = math.log(100.0)  # log(pred/gt) histogram range [-lim, lim]
LOG_RATIO_BINS = 4000            # resolution 2*lim/bins ~= 0.0023 in log space (~0.23% in ratio)
DARK_LUMA = 30.0                 # mean luminance threshold (0-255) for "dark" frames
ANGLE_EDGES_DEG = [0, 10, 20, 30, 40, 45.0001]


def frame_pass(job):
    """Worker (no CUDA): per-frame median GT and luminance, plus pooled per-depth-bin pixel accumulators."""
    data_root, pred_dir, subset, edges = job
    from infer_tartanair import read_rgb
    ev = load_eval_module(data_root)
    nb = len(edges) - 1
    count = np.zeros(nb, np.int64)
    absrel_sum = np.zeros(nb, np.float64)
    d1_count = np.zeros(nb, np.int64)
    ratio_hist = np.zeros((nb, LOG_RATIO_BINS), np.int64)
    frames = []
    with zipfile.ZipFile(osp.join(data_root, subset + '.zip')) as zf:
        for key in ev.frame_keys(zf):
            gt16, valid = ev.read_gt(zf, key)
            pred = np.load(osp.join(pred_dir, subset, key + '.npy'), allow_pickle=False).astype(np.float32)
            rgb = read_rgb(zf, key).astype(np.float32)
            luma = float((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).mean())
            rec = {'frame_key': key, 'subset': subset, 'luma': luma, 'valid_frac': float(valid.mean())}
            if valid.any():
                g = gt16.astype(np.float32)[valid]
                p = pred[valid]
                rec['median_gt'] = float(np.median(g))
                b = np.clip(np.searchsorted(edges, g, side='left') - 1, 0, nb - 1)  # g in (edges[b], edges[b+1]]
                count += np.bincount(b, minlength=nb)
                absrel_sum += np.bincount(b, weights=(np.abs(p - g) / g).astype(np.float64), minlength=nb)
                d1_count += np.bincount(b, weights=(np.maximum(p / g, g / p) < 1.25), minlength=nb).astype(np.int64)
                lr = np.clip(np.log(p / g), -LOG_RATIO_LIM, LOG_RATIO_LIM * (1 - 1e-9))
                h = ((lr + LOG_RATIO_LIM) / (2 * LOG_RATIO_LIM) * LOG_RATIO_BINS).astype(np.int64)
                ratio_hist += np.bincount(b * LOG_RATIO_BINS + h, minlength=nb * LOG_RATIO_BINS).reshape(nb, LOG_RATIO_BINS)
            frames.append(rec)
    return frames, count, absrel_sum, d1_count, ratio_hist


def hist_median_ratio(hist_row):
    """Median of pred/gt from a log-ratio histogram row (bin centre of the median bin)."""
    c = np.cumsum(hist_row)
    if c[-1] == 0:
        return float('nan')
    i = int(np.searchsorted(c, c[-1] / 2.0))
    return float(math.exp(-LOG_RATIO_LIM + (i + 0.5) * 2 * LOG_RATIO_LIM / LOG_RATIO_BINS))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def group_table(recs, keyfn, label):
    groups = {}
    for r in recs:
        groups.setdefault(keyfn(r), []).append(r)
    rows = []
    for k, rs in groups.items():
        rows.append({label: k, 'frames': len(rs),
                     'mean_AbsRel': float(np.mean([r['AbsRel'] for r in rs])),
                     'mean_d1': float(np.mean([r['d1'] for r in rs])),
                     'median_scale_ratio': float(np.median([r['median_scale'] for r in rs])),
                     'mean_AbsRel_median_aligned': float(np.mean([r['AbsRel_median_aligned'] for r in rs])),
                     'subsets': len({r['subset'] for r in rs})})
    return rows


def angle_bin(rad):
    deg = abs(math.degrees(rad))
    for lo, hi in zip(ANGLE_EDGES_DEG[:-1], ANGLE_EDGES_DEG[1:]):
        if lo <= deg < hi:
            return f'{lo:g}-{round(hi):g}'
    return 'out_of_range'


def bootstrap_ci(x, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='Stage 2 dir with pred/, eval_official/, eval_capped/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--extremes-out', default=None, help='write 2 best + 2 worst frames/subset (>= 20%% valid) here')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    if osp.abspath(a.out).startswith('/depot/chan129/data'):
        raise SystemExit('refusing to write under /depot/chan129/data')
    os.makedirs(a.out, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    cap = depth_cap(a.config)['cap_m']  # same derivation as Stage 1/2 (fx = 320 for every frame)
    edges = np.array([0, 2, 5, 10, 20, 40, cap, 100, 200], np.float64)

    # ---- merge frames.tsv + PRIMARY per-frame metrics + DIAGNOSTIC extras
    with open(osp.join(a.data_root, 'frames.tsv'), newline='') as f:
        tsv = {r['frame_key']: r for r in csv.DictReader(f, delimiter='\t')}
    recs = []
    for s in SUBSETS:
        cap_pf = {r['frame_key']: r for r in map(json.loads, open(osp.join(a.run_dir, 'eval_capped', s + '_per_frame.jsonl')))}
        for r in map(json.loads, open(osp.join(a.run_dir, 'eval_official', s + '_per_frame.jsonl'))):
            if r['status'] != 'scored':
                continue
            t, d = tsv[r['frame_key']], cap_pf[r['frame_key']]['diagnostic']
            recs.append({'frame_key': r['frame_key'], 'subset': s, 'env': t['env'], 'difficulty': t['difficulty'],
                         'face': t['face'], 'abs_pitch_deg': abs(math.degrees(float(t['gt_pitch_rad']))),
                         'abs_roll_deg': abs(math.degrees(float(t['gt_roll_rad']))),
                         'pitch_bin': angle_bin(float(t['gt_pitch_rad'])), 'roll_bin': angle_bin(float(t['gt_roll_rad'])),
                         'AbsRel': r['metrics']['AbsRel'], 'd1': r['metrics']['d1'], 'RMSE': r['metrics']['RMSE'],
                         'median_scale': d['median_scale'], 'AbsRel_median_aligned': d['AbsRel_median_aligned'],
                         'frac_valid_gt_above_cap': d['frac_valid_gt_above_cap']})
    by_key = {r['frame_key']: r for r in recs}

    # ---- pixel/frame pass over zips + predictions (parallel over subsets)
    jobs = [(a.data_root, osp.join(a.run_dir, 'pred'), s, edges) for s in SUBSETS]
    nb = len(edges) - 1
    count = np.zeros(nb, np.int64); absrel_sum = np.zeros(nb); d1_count = np.zeros(nb, np.int64)
    ratio_hist = np.zeros((nb, LOG_RATIO_BINS), np.int64)
    with mp.get_context('spawn').Pool(min(a.workers, len(jobs))) as pool:
        for frames, c, s_, d, h in pool.imap_unordered(frame_pass, jobs):
            count += c; absrel_sum += s_; d1_count += d; ratio_hist += h
            for fr in frames:
                if fr['frame_key'] in by_key:
                    by_key[fr['frame_key']].update(fr)
            print(f"pixel pass done: {frames[0]['subset']} ({len(frames)} frames)", flush=True)

    # ---- 1. grouped tables
    fields = ['frames', 'mean_AbsRel', 'mean_d1', 'median_scale_ratio', 'mean_AbsRel_median_aligned', 'subsets']
    tables = {
        'by_env': sorted(group_table(recs, lambda r: r['env'], 'env'), key=lambda r: r['mean_AbsRel']),
        'by_difficulty': sorted(group_table(recs, lambda r: r['difficulty'], 'difficulty'), key=lambda r: r['difficulty']),
        'by_face': sorted(group_table(recs, lambda r: r['face'], 'face'), key=lambda r: r['face']),
        'by_abs_pitch': sorted(group_table(recs, lambda r: r['pitch_bin'], 'abs_pitch_deg'), key=lambda r: float(r['abs_pitch_deg'].split('-')[0])),
        'by_abs_roll': sorted(group_table(recs, lambda r: r['roll_bin'], 'abs_roll_deg'), key=lambda r: float(r['abs_roll_deg'].split('-')[0])),
    }
    subset_rows = group_table(recs, lambda r: r['subset'], 'subset')
    for row in subset_rows:
        lo, hi = bootstrap_ci([r['AbsRel'] for r in recs if r['subset'] == row['subset']])
        row['AbsRel_ci95_lo'], row['AbsRel_ci95_hi'] = lo, hi
    tables['by_subset'] = sorted(subset_rows, key=lambda r: SUBSETS.index(r['subset']))
    for name, rows in tables.items():
        label = next(k for k in rows[0] if k not in fields and not k.startswith('AbsRel_ci95'))
        extra = ['AbsRel_ci95_lo', 'AbsRel_ci95_hi'] if name == 'by_subset' else []
        write_csv(osp.join(a.out, name + '.csv'), rows, [label] + fields + extra)

    # ---- 2. scale ratio analysis
    sc = np.array([r['median_scale'] for r in recs])
    mg = np.array([r['median_gt'] for r in recs])
    sp_rows = [{'group': 'all', 'frames': len(recs), **dict(zip(['spearman_rho', 'p_value'], map(float, spearmanr(mg, sc))))}]
    for s in SUBSETS:
        m = np.array([r['subset'] == s for r in recs])
        sp_rows.append({'group': s, 'frames': int(m.sum()), **dict(zip(['spearman_rho', 'p_value'], map(float, spearmanr(mg[m], sc[m]))))})
    write_csv(osp.join(a.out, 'spearman_scale_vs_median_gt.csv'), sp_rows, ['group', 'frames', 'spearman_rho', 'p_value'])
    ratio_stats = {'frac_scale_gt_1': float((sc > 1).mean()), 'median': float(np.median(sc)),
                   'p10': float(np.percentile(sc, 10)), 'p90': float(np.percentile(sc, 90))}

    bins = np.logspace(np.log10(0.1), np.log10(30), 70)
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True)
    for ax, s in zip(axs.flat, ('all',) + SUBSETS):
        v = sc if s == 'all' else np.array([r['median_scale'] for r in recs if r['subset'] == s])
        ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, color='C0' if s == 'all' else 'C1')
        ax.axvline(1.0, color='k', lw=1)
        ax.axvline(np.median(v), color='r', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_title(f'{s} (n={len(v)}, median {np.median(v):.2f}, >1: {(v > 1).mean():.0%})')
    for ax in axs[1]:
        ax.set_xlabel('per-frame median scale ratio  median(GT)/median(pred)')
    fig.suptitle('Per-frame median scale ratio (black: 1 = correct scale; red dashed: median). DIAGNOSTIC', fontsize=13)
    fig.tight_layout()
    fig.savefig(osp.join(a.out, 'scale_ratio_hist.png'), dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, s in enumerate(SUBSETS):
        m = np.array([r['subset'] == s for r in recs])
        ax.scatter(mg[m], sc[m], s=3, alpha=0.35, color=f'C{i}', label=s)
    gb = np.logspace(np.log10(mg.min()), np.log10(mg.max()), 25)
    idx = np.digitize(mg, gb)
    cen = [np.sqrt(gb[i - 1] * gb[i]) for i in range(1, len(gb)) if (idx == i).sum() >= 20]
    med = [np.median(sc[idx == i]) for i in range(1, len(gb)) if (idx == i).sum() >= 20]
    ax.plot(cen, med, 'k-o', ms=4, lw=2, label='binned median')
    ax.axhline(1.0, color='k', lw=1, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('per-frame median GT depth (m, official valid mask)')
    ax.set_ylabel('median scale ratio median(GT)/median(pred)')
    ax.set_title(f"Scale ratio vs scene depth (Spearman rho = {sp_rows[0]['spearman_rho']:.3f}, n = {len(recs)})")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(osp.join(a.out, 'scale_vs_median_gt.png'), dpi=100)
    plt.close(fig)

    # ---- 3. pixel-level error vs GT depth
    bin_rows = []
    for i in range(nb):
        bin_rows.append({'gt_bin_m': f'{edges[i]:g}-{edges[i + 1]:.4g}', 'bin_centre_m': float((edges[i] + edges[i + 1]) / 2),
                         'pixels': int(count[i]), 'pixel_share': float(count[i] / count.sum()),
                         'AbsRel': float(absrel_sum[i] / count[i]) if count[i] else float('nan'),
                         'median_pred_over_gt': hist_median_ratio(ratio_hist[i]),
                         'd1': float(d1_count[i] / count[i]) if count[i] else float('nan')})
    write_csv(osp.join(a.out, 'depth_bins.csv'), bin_rows, list(bin_rows[0]))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [r['bin_centre_m'] for r in bin_rows]
    ax.plot(xs, [r['median_pred_over_gt'] for r in bin_rows], 'o-', lw=2)
    for r in bin_rows:
        ax.annotate(f"{r['pixel_share']:.1%}", (r['bin_centre_m'], r['median_pred_over_gt']), textcoords='offset points', xytext=(4, 6), fontsize=8)
    ax.axhline(1.0, color='k', lw=1, ls=':')
    ax.axvline(cap, color='r', lw=1, ls='--', label=f'model cap {cap:.1f} m')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('GT depth bin centre (m)'); ax.set_ylabel('median pred/GT (pooled pixels)')
    ax.set_title('Pixel-level median pred/GT vs GT depth (labels: pixel share). DIAGNOSTIC')
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(a.out, 'depth_bin_ratio.png'), dpi=100)
    plt.close(fig)

    # ---- 4. brightness
    lum = np.array([r['luma'] for r in recs])
    dark = lum < DARK_LUMA
    def stats(mask):
        rs = [r for r, m in zip(recs, mask) if m]
        return {'frames': len(rs), 'share': len(rs) / len(recs),
                'mean_AbsRel': float(np.mean([r['AbsRel'] for r in rs])), 'mean_d1': float(np.mean([r['d1'] for r in rs])),
                'median_scale_ratio': float(np.median([r['median_scale'] for r in rs])),
                'mean_AbsRel_median_aligned': float(np.mean([r['AbsRel_median_aligned'] for r in rs]))}
    br_rows = [{'group': f'dark (luma < {DARK_LUMA:g})', **stats(dark)}, {'group': 'rest', **stats(~dark)}]
    for name, key in [('luma vs AbsRel', 'AbsRel'), ('luma vs AbsRel_median_aligned', 'AbsRel_median_aligned')]:
        rho, p = spearmanr(lum, [r[key] for r in recs])
        br_rows.append({'group': f'spearman {name}', 'frames': len(recs), 'share': float(rho), 'mean_AbsRel': float(p)})
    write_csv(osp.join(a.out, 'brightness.csv'), br_rows[:2], list(br_rows[0]))
    write_csv(osp.join(a.out, 'brightness_spearman.csv'),
              [{'pair': r['group'][9:], 'spearman_rho': r['share'], 'p_value': r['mean_AbsRel']} for r in br_rows[2:]],
              ['pair', 'spearman_rho', 'p_value'])
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, key, lab in [(axs[0], 'AbsRel', 'PRIMARY AbsRel'), (axs[1], 'AbsRel_median_aligned', 'median-aligned AbsRel (diagnostic)')]:
        ax.scatter(lum, [r[key] for r in recs], s=3, alpha=0.3)
        ax.axvline(DARK_LUMA, color='r', ls='--', lw=1, label=f'dark < {DARK_LUMA:g}')
        ax.set_yscale('log'); ax.set_xlabel('mean frame luminance (0-255)'); ax.set_ylabel(lab)
        ax.legend()
    fig.suptitle(f'Brightness vs error (dark frames: {dark.mean():.1%})')
    fig.tight_layout()
    fig.savefig(osp.join(a.out, 'brightness.png'), dpi=90)
    plt.close(fig)

    # ---- per-frame table (all merged fields) + summary json
    pf_fields = ['frame_key', 'subset', 'env', 'difficulty', 'face', 'abs_pitch_deg', 'abs_roll_deg', 'AbsRel', 'd1', 'RMSE',
                 'median_scale', 'AbsRel_median_aligned', 'frac_valid_gt_above_cap', 'median_gt', 'luma', 'valid_frac']
    write_csv(osp.join(a.out, 'per_frame_analysis.csv'), recs, pf_fields)
    summary = {'note': 'DIAGNOSTIC analysis (ours); PRIMARY metrics are the unmodified eval_depth.py per-frame values.',
               'frames': len(recs), 'cap_m': cap, 'depth_bin_edges_m': edges.tolist(),
               'log_ratio_hist': {'lim': LOG_RATIO_LIM, 'bins': LOG_RATIO_BINS},
               'scale_ratio': ratio_stats, 'spearman_scale_vs_median_gt': sp_rows,
               'brightness': br_rows, 'depth_bins': bin_rows, 'tables': tables}
    with open(osp.join(a.out, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- extremes for visuals
    if a.extremes_out:
        lines = []
        for s in SUBSETS:
            ok = sorted([r for r in recs if r['subset'] == s and int(tsv[r['frame_key']]['valid_pixels']) >= 0.2 * 640 * 640],
                        key=lambda r: r['AbsRel'])
            lines += [f"{s}\t{r['frame_key']}" for r in ok[:2] + ok[-2:]]
        with open(a.extremes_out, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    print(json.dumps({'scale_ratio': ratio_stats, 'spearman_all': sp_rows[0]}, indent=1))
    for name in ('by_difficulty', 'by_face', 'by_abs_pitch', 'by_abs_roll'):
        print(name, [(list(r.values())[0], r['frames'], round(r['mean_AbsRel'], 4)) for r in tables[name]])
    print('envs (best 5):', [(r['env'], r['frames'], round(r['mean_AbsRel'], 4)) for r in tables['by_env'][:5]])
    print('envs (worst 5):', [(r['env'], r['frames'], round(r['mean_AbsRel'], 4)) for r in tables['by_env'][-5:]])
    for r in bin_rows:
        print('bin', r['gt_bin_m'], f"share {r['pixel_share']:.4f} AbsRel {r['AbsRel']:.4f} med p/g {r['median_pred_over_gt']:.4f} d1 {r['d1']:.4f}")
    print('brightness', br_rows)


if __name__ == '__main__':
    main()

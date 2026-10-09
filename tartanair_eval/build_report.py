"""Build reports/zeroshot_v2s_tartanair/README.md from tartanair_eval/report_template.md and the result files.

Every number in the report is read or recomputed here from result files (Stage 1/2/3 outputs on scratch, frames.tsv,
the weights file, git); the template only holds prose and {{placeholders}}. A placeholder without a value is an error.
Also copies compressed key figures into reports/zeroshot_v2s_tartanair/figs/ (max 1600 px wide).

    python -B tartanair_eval/build_report.py [--dump]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import os.path as osp
import re
import subprocess

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from infer_tartanair import REPO_ROOT, DATA_ROOT, DEFAULT_CONFIG, DEFAULT_WEIGHTS, SUBSETS  # noqa: E402
from score_pilot import depth_cap  # noqa: E402
from fov_experiment import CONDS  # noqa: E402

OUT_ROOT = osp.join(os.environ['RCAC_SCRATCH'], 'metric3d/outputs/zeroshot_v2s')
S1, F, S3 = osp.join(OUT_ROOT, 'stage1'), osp.join(OUT_ROOT, 'full'), osp.join(OUT_ROOT, 'stage3')
REPORT_DIR = osp.join(REPO_ROOT, 'reports/zeroshot_v2s_tartanair')
TEMPLATE = osp.join(osp.dirname(osp.abspath(__file__)), 'report_template.md')
METRICS = ('AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'd1', 'd2', 'd3', 'MAE')
FMT = {'AbsRel': '.4f', 'SqRel': '.3f', 'RMSE': '.3f', 'RMSElog': '.4f', 'SILog': '.2f', 'd1': '.4f', 'd2': '.4f',
       'd3': '.4f', 'MAE': '.3f'}


def jload(p):
    with open(p) as f:
        return json.load(f)


def csv_rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def f(x, spec):
    return format(float(x), spec)


def git(*args):
    return subprocess.run(['git', '-C', REPO_ROOT] + list(args), capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def compress(src, dst_stem, photo):
    im = Image.open(src).convert('RGB')
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    if photo:
        dst = dst_stem + '.jpg'
        im.save(dst, quality=82, optimize=True, progressive=True)
    else:
        dst = dst_stem + '.png'
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(dst, optimize=True)
    return osp.basename(dst)


def build_context():
    c, T = {}, {}
    # ---------------- setup
    cap = depth_cap(DEFAULT_CONFIG)
    from infer_tartanair import Config
    cfg = Config.fromfile(DEFAULT_CONFIG)
    c['config'] = osp.relpath(DEFAULT_CONFIG, REPO_ROOT)
    c['weights_path'] = DEFAULT_WEIGHTS.replace(os.environ['RCAC_SCRATCH'], '$RCAC_SCRATCH')
    c['weights_sha256'] = sha256(DEFAULT_WEIGHTS)
    c['crop_h'], c['crop_w'] = cfg.data_basic.crop_size
    c['canon_focal'] = f(cfg.data_basic.canonical_space['focal_length'], 'g')
    c['clamp_min'], c['clamp_max'] = (f(v, 'g') for v in cfg.data_basic.depth_normalize)
    c['lsf'] = f(cap['label_scale_factor'], '.3f')
    c['lsf_full'] = repr(cap['label_scale_factor'])
    c['cap'] = f(cap['cap_m'], '.1f')
    c['cap_min'] = f(cfg.data_basic.depth_normalize[0] / cap['label_scale_factor'], '.4f')
    pad = cap['pad']
    c['pad_desc'] = f'{pad[2]}+{pad[3]} px (left+right)'
    c['wasted_pct'] = f(100 * (pad[2] + pad[3]) / c['crop_w'], '.1f')
    c['resize_side'] = c['crop_h'] - pad[0] - pad[1]
    with open(osp.join(DATA_ROOT, 'frames.tsv'), newline='') as fh:
        tsv = list(csv.DictReader(fh, delimiter='\t'))
    c['frames_total'] = len(tsv)
    intr = sorted({(r['fx'], r['fy'], r['cx'], r['cy']) for r in tsv})
    c['intr_desc'] = '; '.join(f'fx={a}, fy={b}, cx={cc}, cy={d}' for a, b, cc, d in intr)
    c['n_subsets'] = len({r['subset'] for r in tsv})
    c['n_envs'] = len({r['env'] for r in tsv})
    T['subset_sizes'] = md_table(['subset', 'frames'], [[s, sum(r['subset'] == s for r in tsv)] for s in SUBSETS])
    c['fov_c1_deg'] = f(2 * math.degrees(math.atan(224 / 320)), '.1f')
    c['fov_c2_deg'] = f(2 * math.degrees(math.atan(160 / 320)), '.1f')
    c['canon_mult'] = c['lsf']
    c['fx'] = f(float(tsv[0]['fx']), 'g')
    c['src_side'] = 640
    c['raft_iters'] = cfg.model.decode_head.iters

    # ---------------- correctness checks
    smoke_rows, range_rows = [], []
    for ds in ('kitti', 'nyu'):
        e = jload(osp.join(S1, 'equiv', f'{ds}_equivalence.json'))
        c[f'eq_{ds}_dmax'] = repr(e['depth_absdiff_max'])
        c[f'eq_{ds}_nmax'] = repr(e['normal_absdiff_max'])
        c[f'eq_{ds}_n'] = len(e['frames'])
        for kind in ('metric', 'median_aligned', 'global_aligned'):
            m = e['official_metrics'][kind]
            smoke_rows.append([ds.upper(), kind.replace('_', '-')] + [f(m[k], '.4f') for k in ('abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3')])
        pr = [fr for fr in e['frames']]
        range_rows.append([ds.upper(), f(min(x['pred_p1'] for x in pr), '.2f') + ' - ' + f(max(x['pred_p99'] for x in pr), '.2f'),
                           f(min(x['gt_min'] for x in pr), '.2f') + ' - ' + f(max(x['gt_max'] for x in pr), '.2f'),
                           ', '.join(f(x['median_ratio_gt_over_pred'], '.3f') for x in pr)])
    c['kitti_pred_max'] = f(max(x['pred_max'] for x in jload(osp.join(S1, 'equiv', 'kitti_equivalence.json'))['frames']), '.1f')
    T['smoke'] = md_table(['demo', 'alignment', 'AbsRel', 'RMSE', 'SILog', 'd1', 'd2', 'd3'], smoke_rows)
    T['smoke_ranges'] = md_table(['demo', 'pred depth p1 - p99 (m)', 'GT depth range (m)', 'per-image median GT/pred'], range_rows)

    pil = jload(osp.join(S1, 'pilot', 'pilot_summary.json'))['per_frame']
    off = {}
    for s in SUBSETS:
        for line in open(osp.join(F, 'eval_official', s + '_per_frame.jsonl')):
            r = json.loads(line)
            off[r['frame_key']] = r
    c['pilot_n'] = len(pil)
    c['pilot_maxdiff'] = repr(max(abs(off[p['frame_key']]['metrics'][m] - p['metrics'][m]) for p in pil for m in p['metrics']))

    pooled = jload(osp.join(F, 'eval_capped', 'all_frames_ours.json'))
    P = {s: jload(osp.join(F, 'eval_official', s + '.json')) for s in SUBSETS}
    D = {s: jload(osp.join(F, 'eval_capped', s + '.json')) for s in SUBSETS}
    for name, src, key in (('primary', P, 'PRIMARY_official'), ('diag', D, 'DIAGNOSTIC_capped')):
        N = sum(src[s]['frames'] for s in SUBSETS)
        c[f'pooled_diff_{name}'] = f(max(abs(sum(src[s][m] * src[s]['frames'] for s in SUBSETS) / N - pooled[key][m]) for m in METRICS), '.1e')
    # prediction coverage
    exp = {}
    for r in tsv:
        exp.setdefault(r['subset'], set()).add(r['frame_key'])
    have = {s: {osp.basename(p)[:-4] for p in glob.glob(osp.join(F, 'pred', s, '*.npy'))} for s in SUBSETS}
    c['pred_files'] = sum(len(v) for v in have.values())
    c['pred_coverage_exact'] = 'yes' if all(have[s] == exp[s] for s in SUBSETS) and set(os.listdir(osp.join(F, 'pred'))) == set(SUBSETS) else 'NO'
    c['pred_gb'] = f(sum(osp.getsize(p) for s in SUBSETS for p in glob.glob(osp.join(F, 'pred', s, '*.npy'))) / 1e9, '.1f')
    fov = jload(osp.join(S3, 'fov', 'fov_summary.json'))
    c['fov_c0_identical'] = fov['C0_bit_identical_to_stage2']
    ident = []
    for d in (osp.join(F, 'vis_frames'), osp.join(S3, 'vis_frames')):
        for p in glob.glob(osp.join(d, 'pred', '*', '*.npy')):
            ident.append(np.array_equal(np.load(p), np.load(osp.join(F, 'pred', osp.relpath(p, osp.join(d, 'pred'))))))
    c['vis_identical'] = f'{sum(ident)}/{len(ident)}'
    c['eval_official_status'] = re.search(r'EXIT_STATUS=(\d+)', open(osp.join(F, 'logs', 'eval_official.log')).read()).group(1)

    # ---------------- results
    def mrow(name, j):
        return [name, j['frames'], j['unscorable'], j['failures']] + [f(j[m], FMT[m]) for m in METRICS]
    hdr = ['subset', 'frames', 'unscorable', 'failures'] + list(METRICS)
    T['primary'] = md_table(hdr, [mrow(s, P[s]) for s in SUBSETS] + [mrow('**all frames (ours)**', pooled['PRIMARY_official'])])
    cap_pf = {}
    for s in SUBSETS:
        for line in open(osp.join(F, 'eval_capped', s + '_per_frame.jsonl')):
            r = json.loads(line)
            cap_pf[r['frame_key']] = r['diagnostic']
    tot_valid = sum(d['n_valid_official'] for d in cap_pf.values())
    tot_cap = sum(d['n_valid_capped'] for d in cap_pf.values())
    c['frac_above_cap_all'] = f(100 * (1 - tot_cap / tot_valid), '.2f')
    c['ts_frac_above_cap'] = f(100 * D['test_scene']['diagnostic']['frac_valid_pixels_above_cap'], '.2f')
    c['other_frac_above_cap_max'] = f(100 * max(D[s]['diagnostic']['frac_valid_pixels_above_cap'] for s in SUBSETS if s != 'test_scene'), '.2f')
    T['diag'] = md_table(hdr + ['valid px > cap'],
                         [mrow(s, D[s]) + [f(100 * D[s]['diagnostic']['frac_valid_pixels_above_cap'], '.2f') + '%'] for s in SUBSETS]
                         + [mrow('**all frames (ours)**', pooled['DIAGNOSTIC_capped']) + [c['frac_above_cap_all'] + '%']])
    drows = [[s, f(D[s]['AbsRel'] - P[s]['AbsRel'], '+.4f'), f(D[s]['d1'] - P[s]['d1'], '+.4f'), f(D[s]['RMSE'] - P[s]['RMSE'], '+.3f')] for s in SUBSETS]
    pa, da = pooled['PRIMARY_official'], pooled['DIAGNOSTIC_capped']
    drows.append(['**all frames (ours)**', f(da['AbsRel'] - pa['AbsRel'], '+.4f'), f(da['d1'] - pa['d1'], '+.4f'), f(da['RMSE'] - pa['RMSE'], '+.3f')])
    T['delta'] = md_table(['subset', 'Δ AbsRel', 'Δ d1', 'Δ RMSE (m)'], drows)
    c['d_absrel_all'], c['d_d1_all'], c['d_rmse_all'] = drows[-1][1:]
    for m in ('AbsRel', 'd1', 'RMSE', 'SILog', 'd2', 'd3'):
        c[f'all_{m}'] = f(pa[m], FMT[m])
    c['all_frames_scored'] = pa['frames']
    c['subset_absrel_min'] = f(min(P[s]['AbsRel'] for s in SUBSETS), '.4f')
    c['subset_absrel_max'] = f(max(P[s]['AbsRel'] for s in SUBSETS), '.4f')
    c['subset_absrel_max_name'] = max(SUBSETS, key=lambda s: P[s]['AbsRel'])

    A = jload(osp.join(S3, 'analysis', 'analysis_summary.json'))
    pfa = csv_rows(osp.join(S3, 'analysis', 'per_frame_analysis.csv'))
    srows = []
    for s in SUBSETS:
        dg = D[s]['diagnostic']
        mg = [float(r['median_gt']) for r in pfa if r['subset'] == s]
        srows.append([s, f(dg['median_of_median_scale'], '.3f'), f(dg['mean_AbsRel_median_aligned_official_mask'], '.4f'),
                      f(P[s]['AbsRel'], '.4f'), f(np.median(mg), '.2f')])
    sc_all = A['scale_ratio']
    srows.append(['**all frames (ours)**', f(sc_all['median'], '.3f'),
                  f(np.mean([d['AbsRel_median_aligned'] for d in cap_pf.values()]), '.4f'), c['all_AbsRel'],
                  f(np.median([float(r['median_gt']) for r in pfa]), '.2f')])
    T['scale'] = md_table(['subset', 'median of per-frame median scale ratio', 'mean median-aligned AbsRel', 'PRIMARY AbsRel',
                           'median of per-frame median GT depth (m)'], srows)
    c['scale_median'] = f(sc_all['median'], '.3f')
    c['scale_frac_gt1'] = f(100 * sc_all['frac_scale_gt_1'], '.1f')
    c['scale_p10'], c['scale_p90'] = f(sc_all['p10'], '.2f'), f(sc_all['p90'], '.2f')
    c['medar_all'] = srows[-1][2]

    # ---------------- analysis tables
    def gtab(rows, label, title):
        return md_table([title, 'frames', 'mean AbsRel', 'mean d1', 'median scale ratio', 'mean median-aligned AbsRel'],
                        [[r[label], r['frames'], f(r['mean_AbsRel'], '.4f'), f(r['mean_d1'], '.4f'), f(r['median_scale_ratio'], '.3f'),
                          f(r['mean_AbsRel_median_aligned'], '.4f')] for r in rows])
    tb = A['tables']
    T['env_top5'] = gtab(tb['by_env'][:5], 'env', 'environment (best 5)')
    T['env_bottom5'] = gtab(tb['by_env'][-5:], 'env', 'environment (worst 5)')
    T['env_full'] = gtab(tb['by_env'], 'env', 'environment')
    T['difficulty'] = gtab(tb['by_difficulty'], 'difficulty', 'difficulty')
    T['face'] = gtab(tb['by_face'], 'face', 'face')
    T['pitch'] = gtab(tb['by_abs_pitch'], 'abs_pitch_deg', 'abs. pitch (deg)')
    T['roll'] = gtab(tb['by_abs_roll'], 'abs_roll_deg', 'abs. roll (deg)')
    # share of per-frame AbsRel variance explained by each grouping (eta squared = between-group SS / total SS)
    from analyze_full import angle_bin
    y = np.array([float(r['AbsRel']) for r in pfa])
    groupings = {'environment': [r['env'] for r in pfa], 'subset': [r['subset'] for r in pfa],
                 'difficulty': [r['difficulty'] for r in pfa], 'face': [r['face'] for r in pfa],
                 'abs. pitch bin': [angle_bin(math.radians(float(r['abs_pitch_deg']))) for r in pfa],
                 'abs. roll bin': [angle_bin(math.radians(float(r['abs_roll_deg']))) for r in pfa]}
    eta = []
    for name, labels in groupings.items():
        labels = np.array(labels)
        ss_b = sum((labels == g).sum() * (y[labels == g].mean() - y.mean()) ** 2 for g in set(labels))
        eta.append([name, len(set(labels)), f(ss_b / ((y - y.mean()) ** 2).sum(), '.3f')])
    T['eta'] = md_table(['grouping', 'groups', 'eta squared (share of per-frame AbsRel variance)'], eta)
    c['eta_env'] = eta[0][2]
    c['eta_other_max'] = f(max(float(e[2]) for e in eta[1:]), '.3f')
    c['n_env_analyzed'] = len(tb['by_env'])
    best, worst = tb['by_env'][0], tb['by_env'][-1]
    c['env_range'] = f"{f(best['mean_AbsRel'], '.3f')} ({best['env']}) to {f(worst['mean_AbsRel'], '.3f')} ({worst['env']})"
    rl = tb['by_abs_roll']
    c['roll_first_bin'], c['roll_last_bin'] = rl[0]['abs_roll_deg'], rl[-1]['abs_roll_deg']
    c['roll_first_absrel'], c['roll_last_absrel'] = f(rl[0]['mean_AbsRel'], '.3f'), f(rl[-1]['mean_AbsRel'], '.3f')
    c['roll_first_medar'], c['roll_last_medar'] = f(rl[0]['mean_AbsRel_median_aligned'], '.3f'), f(rl[-1]['mean_AbsRel_median_aligned'], '.3f')
    medars = [r['mean_AbsRel_median_aligned'] for r in rl]
    c['roll_medar_monotonic'] = 'rises monotonically' if all(b > a for a, b in zip(medars, medars[1:])) else 'rises, though not monotonically,'
    pt = tb['by_abs_pitch']
    c['pitch_absrel_lo'], c['pitch_absrel_hi'] = f(min(r['mean_AbsRel'] for r in pt), '.3f'), f(max(r['mean_AbsRel'] for r in pt), '.3f')
    fc = tb['by_face']
    c['face_absrel_lo'], c['face_absrel_hi'] = f(min(r['mean_AbsRel'] for r in fc), '.3f'), f(max(r['mean_AbsRel'] for r in fc), '.3f')
    df = {r['difficulty']: r for r in tb['by_difficulty']}
    c['easy_absrel'], c['hard_absrel'] = f(df['easy']['mean_AbsRel'], '.3f'), f(df['hard']['mean_AbsRel'], '.3f')
    ci = {r['subset']: r for r in tb['by_subset']}
    c['ci_overlap_note'] = ', '.join(f"`{s}` {f(ci[s]['AbsRel_ci95_lo'], '.3f')}-{f(ci[s]['AbsRel_ci95_hi'], '.3f')}" for s in ('test_scene', 'pose_extrapolation', 'test_matched'))
    T['subset_ci'] = md_table(['subset', 'frames', 'mean AbsRel', 'bootstrap 95% CI (frames resampled)'],
                              [[r['subset'], r['frames'], f(r['mean_AbsRel'], '.4f'),
                                f(r['AbsRel_ci95_lo'], '.4f') + ' - ' + f(r['AbsRel_ci95_hi'], '.4f')] for r in tb['by_subset']])
    sp = A['spearman_scale_vs_median_gt']
    c['spearman_all'] = f(sp[0]['spearman_rho'], '.3f')
    T['spearman'] = md_table(['group', 'frames', 'Spearman rho (scale ratio vs median GT depth)'],
                             [[r['group'], r['frames'], f(r['spearman_rho'], '.3f')] for r in sp])
    bins = A['depth_bins']
    T['depth_bins'] = md_table(['GT depth bin (m)', 'pixel share', 'AbsRel', 'median pred/GT', 'd1'],
                               [[b['gt_bin_m'], f(100 * b['pixel_share'], '.2f') + '%', f(b['AbsRel'], '.4f'),
                                 f(b['median_pred_over_gt'], '.3f'), f(b['d1'], '.4f')] for b in bins])
    c['bin_first'], c['bin_last'] = bins[0]['gt_bin_m'], bins[-1]['gt_bin_m']
    c['bin_first_ratio'], c['bin_last_ratio'] = f(bins[0]['median_pred_over_gt'], '.3f'), f(bins[-1]['median_pred_over_gt'], '.3f')
    c['share_under_5'] = f(100 * (bins[0]['pixel_share'] + bins[1]['pixel_share']), '.1f')
    c['bin_2_5_ratio'] = f(bins[1]['median_pred_over_gt'], '.3f')
    c['bin_5_10_ratio'] = f(bins[2]['median_pred_over_gt'], '.3f')
    c['bin_10_20_ratio'] = f(bins[3]['median_pred_over_gt'], '.3f')
    c['ratio_hist_res'] = f(100 * (math.exp(2 * A['log_ratio_hist']['lim'] / A['log_ratio_hist']['bins']) - 1), '.2f')
    br = A['brightness']
    dark, rest = br[0], br[1]
    c['dark_n'], c['dark_share'] = dark['frames'], f(100 * dark['share'], '.1f')
    T['brightness'] = md_table(['group', 'frames', 'share', 'mean AbsRel', 'mean d1', 'median scale ratio', 'mean median-aligned AbsRel'],
                               [[b['group'], b['frames'], f(100 * b['share'], '.1f') + '%', f(b['mean_AbsRel'], '.4f'), f(b['mean_d1'], '.4f'),
                                 f(b['median_scale_ratio'], '.3f'), f(b['mean_AbsRel_median_aligned'], '.4f')] for b in (dark, rest)])
    c['luma_rho_absrel'] = f(br[2]['share'], '.3f')
    c['luma_rho_medar'] = f(br[3]['share'], '.3f')

    # ---------------- FoV experiment
    c['fov_n'], c['fov_envs'] = fov['frames'], fov['envs']
    def frow(r):
        return [r['condition'], f(r['cap_m'], '.1f'), r['frames'], f(r['median_scale_ratio'], '.3f'), f(r['median_abs_log_scale'], '.3f'),
                f(r['mean_AbsRel'], '.4f'), f(r['mean_d1'], '.4f'), f(r['mean_AbsRel_median_aligned'], '.4f'),
                '-' if math.isnan(r['frac_toward_1_vs_C0']) else f(100 * r['frac_toward_1_vs_C0'], '.1f') + '%']
    fhdr = ['condition', 'model cap (m)', 'frames', 'median scale ratio', 'median abs(log ratio)', 'mean AbsRel', 'mean d1',
            'mean median-aligned AbsRel', 'frames closer to 1 than C0']
    T['fov'] = md_table(fhdr, [frow(r) for r in fov['summary'] if r['mask'] == 'central320'])
    T['fov_capped'] = md_table(fhdr, [frow(r) for r in fov['summary'] if r['mask'] != 'central320'])
    for r in fov['summary']:
        if r['mask'] == 'central320':
            c[f"fov_{r['condition'][:2].lower()}_med"] = f(r['median_scale_ratio'], '.3f')
            c[f"fov_{r['condition'][:2].lower()}_cap"] = f(r['cap_m'], '.1f')
            c[f"fov_{r['condition'][:2].lower()}_absrel"] = f(r['mean_AbsRel'], '.4f')
    T['fov_paired'] = md_table(['A', 'B', 'frames', 'A closer to 1 than B', 'median ratio factor A/B', '95% CI of median Δlog (A-B)'],
                               [[p['A'], p['B'], p['frames'], f(100 * p['frac_A_closer_to_1'], '.1f') + '%', f(p['median_ratio_factor_A_over_B'], '.3f'),
                                 f(p['median_delta_log_scale_ci95_lo'], '.3f') + ' to ' + f(p['median_delta_log_scale_ci95_hi'], '.3f')] for p in fov['paired']])
    pd = {(p['A'][:2], p['B'][:2]): p for p in fov['paired']}
    for (a_, b_), p in pd.items():
        k = f'{a_.lower()}{b_.lower()}'
        c[f'pair_{k}_factor'] = f(p['median_ratio_factor_A_over_B'], '.3f')
        c[f'pair_{k}_frac'] = f(100 * p['frac_A_closer_to_1'], '.1f')
    T['fov_subset'] = md_table(['subset', 'frames'] + [n for n in CONDS],
                               [[p['subset'], p['frames']] + [f(p[f'{n}_median_scale'], '.3f') for n in CONDS] for p in fov['per_subset']])
    c['fov_c2_log_remaining_pct'] = f(100 * math.log(float(c['fov_c2_med'])) / math.log(float(c['fov_c0_med'])), '.0f')

    # ---------------- timing / repro
    log = open(osp.join(F, 'logs', 'infer.log')).read()
    c['infer_s'] = re.search(r'done: \d+ frames in ([\d.]+)s', log).group(1)
    c['infer_min'] = f(float(c['infer_s']) / 60, '.1f')
    with open(osp.join(F, 'timing.tsv'), newline='') as fh:
        tim = list(csv.DictReader(fh, delimiter='\t'))
    c['sec_per_frame'] = f(np.mean([float(r['total_s']) for r in tim[1:]]), '.3f')
    c['infer_per_frame'] = f(np.mean([float(r['infer_s']) for r in tim[1:]]), '.3f')
    c['base_commit'] = open(osp.join(REPO_ROOT, 'ORIGINAL_COMMIT.txt')).read().strip()[:7]
    commits = git('log', '--reverse', '--format=%h %s', f"{c['base_commit']}..HEAD", '--', 'tartanair_eval').splitlines()
    T['commits'] = '\n'.join(f'- `{ln[:7]}` {ln[8:]}' for ln in commits)
    c['out_root'] = OUT_ROOT.replace(os.environ['RCAC_SCRATCH'], '$RCAC_SCRATCH')
    c['gpu'] = 'NVIDIA A100 80GB PCIe (Gilbreth gilbreth-k046)'
    return c, T


def render(template, c, T):
    def sub(m):
        key = m.group(1)
        if key.startswith('TABLE:'):
            return T[key[6:]]
        if key not in c:
            raise KeyError(f'no value for placeholder {{{{{key}}}}}')
        return str(c[key])
    return re.sub(r'\{\{([A-Za-z0-9_:]+)\}\}', sub, template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', action='store_true', help='print the values and tables instead of writing the report')
    a = ap.parse_args()
    c, T = build_context()
    if a.dump:
        for k, v in c.items():
            print(f'{k} = {v}')
        for k, v in T.items():
            print(f'\n### TABLE:{k}\n{v}')
        return
    figs = osp.join(REPORT_DIR, 'figs')
    os.makedirs(figs, exist_ok=True)
    for old in glob.glob(osp.join(figs, '*')):
        os.remove(old)
    fig_srcs = {
        'scale_ratio_hist': (osp.join(S3, 'analysis', 'scale_ratio_hist.png'), False),
        'scale_vs_median_gt': (osp.join(S3, 'analysis', 'scale_vs_median_gt.png'), False),
        'depth_bin_ratio': (osp.join(S3, 'analysis', 'depth_bin_ratio.png'), False),
        'brightness': (osp.join(S3, 'analysis', 'brightness.png'), False),
        'fov_ratio_distribution': (osp.join(S3, 'fov', 'fov_ratio_distribution.png'), False),
        'fov_examples': (osp.join(S3, 'vis', 'fov_examples.png'), True),
        'extremes_test_scene': (osp.join(S3, 'vis', 'sheet_extremes_test_scene.png'), True),
        'extremes_all': (osp.join(S3, 'vis', 'sheet_extremes_all.png'), True),
        'random_all': (osp.join(F, 'vis', 'sheet_all.png'), True),
    }
    for name, (src, photo) in fig_srcs.items():
        c[f'fig_{name}'] = 'figs/' + compress(src, osp.join(figs, name), photo)
    with open(TEMPLATE) as fh:
        text = render(fh.read(), c, T)
    with open(osp.join(REPORT_DIR, 'README.md'), 'w') as fh:
        fh.write(text)
    size = sum(osp.getsize(p) for p in glob.glob(osp.join(REPORT_DIR, '**', '*'), recursive=True) if osp.isfile(p))
    print(f'wrote {osp.join(REPORT_DIR, "README.md")}; report dir size {size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()

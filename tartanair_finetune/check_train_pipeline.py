"""Check TartanAirV2Dataset + the official training pipeline on train frames (CPU only; no model).

Part A, exact checks: deterministic variants of the pipeline (RandomResize at a fixed ratio or off, centre crop)
on seeded train frames, compared with labels computed independently from the zip members:
  target == crop(pad(nearest_resize(depth * 1000 / fx) / r)) on valid pixels, -1 elsewhere (incl. padding),
  input == crop(pad(resized RGB)), sky -> semantic id 142, scale == (1000 / fx) / r, pad as expected.
  Also measures how many valid label pixels RandomResize(is_lidar=True) (the KITTI setting, a scatter resize
  meant for sparse LiDAR) loses compared with is_lidar=False (nearest neighbour, used for dense datasets).
Part B, the full random pipeline from configs/_base_/datasets/tartanair_v2.py through a DataLoader: output
  shapes, label-scale distribution, canonical label range vs. the decoder caps 200 / 700, throughput.
Part C: a few visualisations (input after de-normalisation, target, sky mask).

Run from the repo root (it switches to training/ itself; never imports the root `mono/`):
    python -B tartanair_finetune/check_train_pipeline.py --out $RCAC_SCRATCH/metric3d/outputs/data_checks/pipeline
"""
import argparse
import copy
import io
import json
import os
import os.path as osp
import random
import sys
import time
import zipfile

import cv2
import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TRAIN_ROOT = osp.join(REPO_ROOT, 'training')
sys.path.insert(0, TRAIN_ROOT)
os.chdir(TRAIN_ROOT)  # build_data_array / get_func resolve modules relative to training/

try:
    from mmcv.utils import Config
except ImportError:
    from mmengine import Config
import data_server_info  # noqa: F401  (registers db_info)
from mono.utils.db import load_data_info
from mono.datasets.tartanair_v2_dataset import TartanAirV2Dataset

DATA_BASIC = dict(canonical_space=dict(focal_length=1000.0), depth_range=(0, 1), depth_normalize=(0.1, 200),
                  crop_size=(616, 1064), clip_depth_range=(0.1, 200))
MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])


def build(cfg_ds, pipeline=None):
    cfg_ds = copy.deepcopy(cfg_ds)
    if pipeline is not None:
        cfg_ds.data.train.pipeline = pipeline
    db = {}
    load_data_info('data_server_info', data_info=db)
    return TartanAirV2Dataset(cfg_ds, 'train', db_info=db['TartanAirV2'], **DATA_BASIC)


def det_pipeline(ratio=None, is_lidar=False):
    p = [dict(type='BGR2RGB'), dict(type='LabelScaleCononical')]
    if ratio is not None:
        p.append(dict(type='RandomResize', prob=1.0, ratio_range=(ratio, ratio), is_lidar=is_lidar))
    p += [dict(type='RandomCrop', crop_size=(0, 0), crop_type='center', ignore_label=-1, padding=[0, 0, 0]),
          dict(type='ToTensor'), dict(type='Normalize', mean=MEAN.tolist(), std=STD.tolist())]
    return p


def read_raw(root, anno):
    """Independent read of one frame from its zip (no dataset code)."""
    with zipfile.ZipFile(osp.join(root, anno['zip'])) as zf:
        k = anno['frame_key']
        rgb = cv2.imdecode(np.frombuffer(zf.read(k + '/rgb.png'), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        depth = np.load(io.BytesIO(zf.read(k + '/depth.npy'))).astype(np.float64)
        valid = cv2.imdecode(np.frombuffer(zf.read(k + '/valid_mask.png'), np.uint8), cv2.IMREAD_UNCHANGED) > 0
        sky = cv2.imdecode(np.frombuffer(zf.read(k + '/sky_mask.png'), np.uint8), cv2.IMREAD_UNCHANGED) > 0
    return rgb, depth, valid, sky


def expected(rgb, depth, valid, sky, fx, ratio, crop=(616, 1064)):
    """Expected (input RGB, target, sem, scale, pad) of det_pipeline(ratio, is_lidar=False)."""
    s = 1000.0 / fx
    lab = np.where(valid, depth * s, -1.0)
    sem = np.where(sky, 142, -1)
    if ratio is not None:
        h, w = int(rgb.shape[0] * ratio + 0.5), int(rgb.shape[1] * ratio + 0.5)
        rgb = cv2.resize(np.ascontiguousarray(rgb), (w, h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST) / ratio
        sem = cv2.resize(sem.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
        s = s / ratio
    H, W = lab.shape
    ch, cw = crop
    ph, pw = max(ch - H, 0), max(cw - W, 0)
    t, l = ph // 2, pw // 2
    rgb = cv2.copyMakeBorder(np.ascontiguousarray(rgb), t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    lab = cv2.copyMakeBorder(lab, t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=-1)
    sem = cv2.copyMakeBorder(sem.astype(np.int32), t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=-1)
    ho, wo = int((H + ph - ch) / 2), int((W + pw - cw) / 2)
    sl = (slice(ho, ho + ch), slice(wo, wo + cw))
    lab = lab[sl].copy()
    lab[lab < 1e-6] = -1
    return rgb[sl], lab, sem[sl], s, [t, ph - t, l, pw - l]


def part_a(cfg_ds, n_frames, seed):
    res = {}
    ds = build(cfg_ds, det_pipeline())
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n_frames)
    for ratio in (None, 0.85, 0.9, 1.1, 1.15):
        ds_n = build(cfg_ds, det_pipeline(ratio, is_lidar=False))
        ds_l = build(cfg_ds, det_pipeline(ratio, is_lidar=True)) if ratio is not None else None
        st = dict(frames=0, target_max_abs_err=0.0, target_invalid_mismatch_px=0, input_max_abs_err=0.0,
                  sem_mismatch_px=0, scale_err=0.0, pad_mismatch=0, lidar_lost_valid_px_share=[],
                  lidar_value_mismatch_px=0)
        for i in idxs:
            anno = ds_n.annotations['files'][i]
            rgb, depth, valid, sky = read_raw(ds_n.data_root, anno)
            e_rgb, e_lab, e_sem, e_s, e_pad = expected(rgb, depth, valid, sky, anno['cam_in'][0], ratio)
            d = ds_n[i]
            tgt = d['target'].numpy()[0]
            inp = d['input'].numpy().transpose(1, 2, 0) * STD + MEAN
            v_e, v_t = e_lab > 0, tgt > 0
            st['target_invalid_mismatch_px'] += int((v_e != v_t).sum())
            both = v_e & v_t
            if both.any():
                st['target_max_abs_err'] = max(st['target_max_abs_err'], float(np.abs(tgt[both] - e_lab[both]).max()))
            st['input_max_abs_err'] = max(st['input_max_abs_err'], float(np.abs(inp - e_rgb).max()))
            st['sem_mismatch_px'] += int((d['sem_mask'].numpy()[0] != e_sem).sum())
            st['scale_err'] = max(st['scale_err'], abs(float(d['scale']) - e_s))
            st['pad_mismatch'] += int(list(d['pad'].numpy()) != e_pad)
            if ds_l is not None:
                tl = ds_l[i]['target'].numpy()[0]
                st['lidar_lost_valid_px_share'].append(float((v_e & ~(tl > 0)).sum() / max(v_e.sum(), 1)))
                bl = v_e & (tl > 0)
                # Scatter resize keeps source values but may move them by up to a pixel; count disagreements.
                st['lidar_value_mismatch_px'] += int((np.abs(tl[bl] - e_lab[bl]) > 1e-4).sum())
            st['frames'] += 1
        if st['lidar_lost_valid_px_share']:
            st['lidar_lost_valid_px_share'] = float(np.mean(st['lidar_lost_valid_px_share']))
        else:
            st.pop('lidar_lost_valid_px_share'), st.pop('lidar_value_mismatch_px')
        res['none' if ratio is None else f'r={ratio}'] = st
    return res


def part_b(cfg_ds, n_batches, batch_size, workers, seed):
    ds = build(cfg_ds)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(ds, generator=g)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers,
                                     drop_last=True, pin_memory=False, generator=g)
    scales, valid_frac, above200, above700, samples_above700, vmax, n_valid = [], [], 0, 0, 0, 0.0, 0
    shapes, nonfinite = set(), 0
    t0 = time.time()
    first = None
    for b, data in enumerate(dl):
        if b == 0:
            first = time.time() - t0
            t1 = time.time()
            vis = {k: data[k][:8].clone() for k in ('input', 'target', 'sem_mask')}
            vis['filename'] = data['filename'][:8]
        if b >= n_batches:
            break
        tgt = data['target']
        shapes.add((tuple(data['input'].shape[1:]), tuple(tgt.shape[1:])))
        nonfinite += int((~torch.isfinite(data['input'])).sum() + (~torch.isfinite(tgt)).sum())
        scales += data['scale'].tolist()
        v = tgt > 0
        valid_frac += v.float().mean(dim=(1, 2, 3)).tolist()
        n_valid += int(v.sum())
        above200 += int((tgt > 200).sum())
        above700 += int((tgt > 700).sum())
        samples_above700 += int(((tgt > 700).flatten(1).any(1)).sum())
        vmax = max(vmax, float(tgt.max()))
    elapsed = time.time() - t1
    n = len(scales)
    sc = np.array(scales)
    return dict(
        samples=n, batch_size=batch_size, workers=workers,
        shapes=[str(s) for s in shapes], nonfinite=nonfinite,
        first_batch_s=round(first, 2), frames_per_s_after_first=round((n - batch_size) / elapsed, 1),
        scale_min=float(sc.min()), scale_max=float(sc.max()),
        share_scale_eq_3125=float(np.mean(np.isclose(sc, 3.125))),
        mean_valid_frac_of_crop=float(np.mean(valid_frac)),
        canonical_target_max=vmax,
        share_valid_px_above_200=above200 / n_valid, share_valid_px_above_700=above700 / n_valid,
        share_samples_with_px_above_700=samples_above700 / n), vis


def part_c(vis, out):
    os.makedirs(out, exist_ok=True)
    for i in range(vis['input'].shape[0]):
        inp = (vis['input'][i].numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        tgt = vis['target'][i, 0].numpy()
        v = tgt > 0
        lg = np.zeros_like(tgt)
        lg[v] = np.log(tgt[v])
        lo, hi = (np.percentile(lg[v], 1), np.percentile(lg[v], 99)) if v.any() else (0, 1)
        col = cv2.applyColorMap((np.clip((lg - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8),
                                cv2.COLORMAP_TURBO)
        col[~v] = 128
        sky = np.where(vis['sem_mask'][i, 0].numpy() == 142, 255, 0).astype(np.uint8)
        sky = cv2.cvtColor(sky, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(osp.join(out, f'sample{i}_{vis["filename"][i][:-4]}.jpg'), np.concatenate([inp, col, sky], 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-exact', type=int, default=24, help='frames for the exact checks (part A)')
    ap.add_argument('--n-batches', type=int, default=40, help='batches for part B')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=3047)
    a = ap.parse_args()
    out = osp.abspath(a.out)
    if out.startswith('/depot/chan129/data'):
        raise SystemExit('refusing to write under /depot/chan129/data')
    os.makedirs(out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    cfg_ds = Config.fromfile('mono/configs/_base_/datasets/tartanair_v2.py').TartanAirV2_dataset
    summary = {'data_basic': DATA_BASIC}
    t = time.time()
    summary['A_exact'] = part_a(cfg_ds, a.n_exact, a.seed)
    summary['A_seconds'] = round(time.time() - t, 1)
    summary['B_full_pipeline'], vis = part_b(cfg_ds, a.n_batches, a.batch_size, a.workers, a.seed)
    part_c(vis, osp.join(out, 'vis'))
    with open(osp.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

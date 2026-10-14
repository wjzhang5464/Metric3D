"""Compare two prediction directories (<dir>/<subset>/<frame_key>.npy) file by file.

Used to confirm that the zero-shot rerun with the final inference script (depth saved before the `< 300` zeroing)
is bit-identical to the original zero-shot predictions (saved after it), as expected because that zeroing never
fires (reports/finetune_v2s_tartanair/PROTOCOL.md, fact 5).

    python -B tartanair_finetune/compare_preds.py --a <old>/pred --b <new>/pred [--out report.json]
"""
import argparse
import glob
import json
import os.path as osp
from multiprocessing import Pool

import numpy as np


def compare(args):
    pa, pb = args
    a, b = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
    if a.shape != b.shape or a.dtype != b.dtype:
        return 'meta', float('inf')
    same = a.tobytes() == b.tobytes()
    return ('identical' if same else 'different'), (0.0 if same else float(np.abs(a.astype(np.float64) - b).max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    rel = lambda d: {osp.relpath(p, d) for p in glob.glob(osp.join(d, '*', '*.npy'))}
    fa, fb = rel(a.a), rel(a.b)
    common = sorted(fa & fb)
    with Pool(a.workers) as pool:
        res = pool.map(compare, [(osp.join(a.a, r), osp.join(a.b, r)) for r in common], chunksize=32)
    rep = {'a': a.a, 'b': a.b, 'files_a': len(fa), 'files_b': len(fb), 'only_in_a': len(fa - fb),
           'only_in_b': len(fb - fa), 'compared': len(common),
           'bit_identical': sum(s == 'identical' for s, _ in res),
           'different': sum(s == 'different' for s, _ in res), 'shape_or_dtype_differs': sum(s == 'meta' for s, _ in res),
           'max_abs_diff': max((d for _, d in res), default=0.0)}
    rep['all_identical'] = (rep['bit_identical'] == len(fa) == len(fb))
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rep, f, indent=1)


if __name__ == '__main__':
    main()

"""Deterministic, evenly spaced frame selection: k frames per subset from frames.tsv.

Within each subset's rows (frames.tsv order), picks row indices floor((i + 0.5) * n / k), i = 0..k-1.
Writes "subset<TAB>frame_key" lines (readable by infer_tartanair.py --keys-file).

    python tartanair_eval/select_keys.py --per-subset 3 --out <keys.txt>
"""
import argparse
import csv
import os
import os.path as osp

from infer_tartanair import DATA_ROOT, SUBSETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-subset', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=DATA_ROOT)
    a = ap.parse_args()

    with open(osp.join(a.data_root, 'frames.tsv'), newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    lines = []
    for subset in SUBSETS:
        sub = [r for r in rows if r['subset'] == subset]
        n, k = len(sub), a.per_subset
        for i in range(k):
            lines.append(f"{subset}\t{sub[int((i + 0.5) * n / k)]['frame_key']}")
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

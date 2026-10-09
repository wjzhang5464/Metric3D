"""Deterministic frame selection: k frames per subset from frames.tsv.

Default (evenly spaced): within each subset's rows (frames.tsv order), row indices floor((i + 0.5) * n / k).
--seed S: seeded random pick of k distinct environments per subset (fewer if the subset has fewer), one random
frame from each, topped up with random frames from the rest of the subset if needed.
Writes "subset<TAB>frame_key" lines (readable by infer_tartanair.py --keys-file).

    python tartanair_eval/select_keys.py --per-subset 3 --out <keys.txt>
    python tartanair_eval/select_keys.py --per-subset 4 --seed 0 --out <keys.txt>
"""
import argparse
import csv
import os
import os.path as osp
import random

from infer_tartanair import DATA_ROOT, SUBSETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-subset', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None, help='seeded random pick from distinct environments')
    ap.add_argument('--data-root', default=DATA_ROOT)
    a = ap.parse_args()

    with open(osp.join(a.data_root, 'frames.tsv'), newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    lines = []
    for subset in SUBSETS:
        sub = [r for r in rows if r['subset'] == subset]
        n, k = len(sub), a.per_subset
        if a.seed is None:
            picks = [sub[int((i + 0.5) * n / k)] for i in range(k)]
        else:
            rng = random.Random(f'{a.seed}:{subset}')
            envs = sorted({r['env'] for r in sub})
            picks = [rng.choice([r for r in sub if r['env'] == e]) for e in rng.sample(envs, min(k, len(envs)))]
            rest = [r for r in sub if r not in picks]
            picks += rng.sample(rest, k - len(picks))
        lines += [f"{subset}\t{r['frame_key']}" for r in picks]
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

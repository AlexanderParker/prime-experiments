"""Iterated local search on the covering form (v2 of ea_cover.py).

Best-response coordinate ascent: sweep gears, set each gear's offset to the best of its g
choices given the others; on a plateau accept sideways moves; when stuck, kick 2-4 random
gears and continue; keep the best. Targets F (no holes), F2 (1 hole), F3 (2 holes),
G2 (<= 2 openings per q'-window). Lower bounds only; calibrated where exact values exist.

Usage: ils_cover.py q qnext [--target F3] [--iters 400] [--R 768] [--seed 0] [--procs 1]
"""
import argparse
import time

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_rows(gears, R, extra=None):
    rows = []
    i = np.arange(R)
    for g in gears:
        d = (2 * pow(6, -1, g)) % g
        if extra is not None and g == extra[0]:
            d = extra[1]
        tab = np.zeros((g, R), dtype=bool)
        for c in range(g):
            m = (i - c) % g
            tab[c] = (m == 0) | (m == d)
        rows.append(tab)
    return rows


def longest_with_holes(b, h):
    op = np.flatnonzero(~b)
    R = len(b)
    if len(op) <= h:
        return R
    pad = np.concatenate([[-1], op, [R]])
    return int((pad[1 + h:] - pad[:-1 - h] - 1).max())


def longest_sparse(b, q2, t=2):
    R = len(b)
    cs = np.concatenate([[0], np.cumsum(~b)])
    cq = cs[q2:] - cs[:-q2]
    ok = (cq <= t).astype(np.int8)
    d = np.diff(np.concatenate([[0], ok, [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    if len(s) == 0:
        return 0
    return int((e - s).max()) + q2 - 1


def fit_of(b, target, q2):
    if target == "F":
        return longest_with_holes(b, 0)
    if target == "F2":
        return longest_with_holes(b, 1)
    if target == "F3":
        return longest_with_holes(b, 2)
    return longest_sparse(b, q2, 2)


def search(gears, q2, target, R, iters, rng, extra=None):
    rows = build_rows(gears, R, extra)
    n = len(gears)
    genome = np.array([rng.integers(g) for g in gears])
    # blocked-count array: how many gears block each slot (so removing one gear is cheap)
    cnt = np.zeros(R, dtype=np.int16)
    for tab, c in zip(rows, genome):
        cnt += tab[c]
    best_f = fit_of(cnt > 0, target, q2)
    best_g = genome.copy()
    cur_f = best_f
    for it in range(iters):
        improved = True
        while improved:
            improved = False
            for k in rng.permutation(n):
                base = cnt - rows[k][genome[k]]
                cands = []
                for c in range(gears[k]):
                    f = fit_of((base + rows[k][c]) > 0, target, q2)
                    cands.append(f)
                cands = np.array(cands)
                mx = int(cands.max())
                choices = np.flatnonzero(cands == mx)
                c = int(rng.choice(choices))  # sideways moves allowed
                if mx > cur_f:
                    improved = True
                cur_f = mx
                cnt = base + rows[k][c]
                genome[k] = c
            if cur_f > best_f:
                best_f, best_g = cur_f, genome.copy()
        # kick
        genome = best_g.copy() if rng.random() < 0.5 else genome
        cnt = np.zeros(R, dtype=np.int16)
        for k in rng.choice(n, size=rng.integers(2, 5), replace=False):
            genome[k] = rng.integers(gears[k])
        for tab, c in zip(rows, genome):
            cnt += tab[c]
        cur_f = fit_of(cnt > 0, target, q2)
    return best_f, best_g, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", type=int)
    ap.add_argument("qnext", type=int)
    ap.add_argument("--target", default="F3")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--R", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default=None, help="q2:delta - add gear q2 with tooth spacing delta")
    a = ap.parse_args()
    gears = [g for g in PR if g <= a.q]
    extra = None
    if a.extra:
        e = a.extra.split(":")
        extra = (int(e[0]), int(e[1]))
        gears = gears + [extra[0]]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    f, genome, rows = search(gears, a.qnext, a.target, a.R, a.iters, rng, extra)
    b = np.zeros(a.R, dtype=bool)
    for tab, c in zip(rows, genome):
        b |= tab[c]
    # locate the stretch and its holes
    if a.target == "G2":
        cs = np.concatenate([[0], np.cumsum(~b)])
        cq = cs[a.qnext:] - cs[:-a.qnext]
        ok = (cq <= 2).astype(np.int8)
        d = np.diff(np.concatenate([[0], ok, [0]]))
        s_, e_ = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        i = int((e_ - s_).argmax())
        s = int(s_[i])
    else:
        h = {"F": 0, "F2": 1, "F3": 2}[a.target]
        op = np.flatnonzero(~b)
        pad = np.concatenate([[-1], op, [a.R]])
        L = pad[1 + h:] - pad[:-1 - h] - 1
        s = int(pad[int(L.argmax())] + 1)
    seg = b[s:s + f]
    holes = [int(x) for x in np.flatnonzero(~seg)]
    gaps = [b_ - a_ for a_, b_ in zip([-1] + holes, holes + [f])]
    tag = f" extra {extra[0]} delta={extra[1]}" if extra else ""
    print(f"{'+'.join(map(str, gears))} q'={a.qnext} target={a.target}{tag}: best={f} holes at {holes} gaps {gaps}  time {time.time() - t0:.0f}s  offsets {genome.tolist()}")


if __name__ == "__main__":
    main()

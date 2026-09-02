"""Memetic search on the covering form: a genetic algorithm whose every child is polished by
the best-response sweep of ils_cover.py (offset genome, tournament, uniform crossover,
mutation, local optimisation, replace-worst). Aim: exact F where plain ILS fell short
({5..43}: 99 vs 102; {5..47}: 116 vs 117) so the F_2 / F_3 / G_2 lower bounds can be trusted.

Usage: memetic.py q qnext [--target F] [--pop 16] [--gens 200] [--R 768] [--seed 0]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ils_cover import PR, build_rows, fit_of


def polish(rows, gears, genome, target, q2, rng, sweeps=3):
    R = rows[0].shape[1]
    cnt = np.zeros(R, dtype=np.int16)
    for tab, c in zip(rows, genome):
        cnt += tab[c]
    cur = fit_of(cnt > 0, target, q2)
    for _ in range(sweeps):
        improved = False
        for k in rng.permutation(len(gears)):
            base = cnt - rows[k][genome[k]]
            cands = np.array([fit_of((base + rows[k][c]) > 0, target, q2) for c in range(gears[k])])
            mx = int(cands.max())
            c = int(rng.choice(np.flatnonzero(cands == mx)))
            if mx > cur:
                improved = True
            cur = mx
            cnt = base + rows[k][c]
            genome[k] = c
        if not improved:
            break
    return cur, genome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", type=int)
    ap.add_argument("qnext", type=int)
    ap.add_argument("--target", default="F")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--R", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gears = [g for g in PR if g <= a.q]
    rng = np.random.default_rng(a.seed)
    rows = build_rows(gears, a.R)
    t0 = time.time()
    pop, fit = [], []
    for _ in range(a.pop):
        gnm = np.array([rng.integers(g) for g in gears])
        f, gnm = polish(rows, gears, gnm, a.target, a.qnext, rng)
        pop.append(gnm); fit.append(f)
    best = max(fit)
    print(f"init best {best}  time {time.time() - t0:.0f}s", flush=True)
    for gen in range(a.gens):
        i, j = rng.integers(a.pop, size=2)
        p1 = pop[i] if fit[i] >= fit[j] else pop[j]
        i, j = rng.integers(a.pop, size=2)
        p2 = pop[i] if fit[i] >= fit[j] else pop[j]
        mask = rng.random(len(gears)) < 0.5
        child = np.where(mask, p1, p2)
        for _ in range(rng.integers(1, 4)):
            k = rng.integers(len(gears))
            child[k] = rng.integers(gears[k])
        f, child = polish(rows, gears, child, a.target, a.qnext, rng)
        w = int(np.argmin(fit))
        if f >= fit[w] and not any(np.array_equal(child, p) for p in pop):
            pop[w], fit[w] = child, f
        if max(fit) > best:
            best = max(fit)
            print(f"gen {gen}: best {best}  time {time.time() - t0:.0f}s", flush=True)
    b = int(np.argmax(fit))
    print(f"{'+'.join(map(str, gears))} q'={a.qnext} target={a.target} memetic: best={fit[b]}  "
          f"time {time.time() - t0:.0f}s  offsets {pop[b].tolist()}")


if __name__ == "__main__":
    main()

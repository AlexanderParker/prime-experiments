"""Evolutionary / local search on the covering form of the record gap.

Machine {5..q}: gear g blocks residues c_g and c_g + d_g (mod g), d_g = 2 * 6^{-1} mod g,
offset c_g free (CRT). Genome = offsets. Fitness = longest stretch in [0, R) with
  h = 0 holes (F), 1 hole (F_2), 2 holes (F_3), or 3-sparse w.r.t. q' (G_2).
Population + mutation (resample / +-1 one gear) + hill climb. Reports best found per target
and the offsets / which gears cover the stretch. Lower bounds only.

Usage: ea_cover.py q qnext [--target F|F2|F3|G2] [--pop 64] [--gens 3000] [--R 512] [--seed 0]
"""
import argparse
import sys
import time

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def build_rows(gears, R):
    """rows[g_index][c] = bool array over [0, R): blocked by gear g at offset c."""
    rows = []
    i = np.arange(R)
    for g in gears:
        d = (2 * pow(6, -1, g)) % g
        tab = np.zeros((g, R), dtype=bool)
        for c in range(g):
            m = (i - c) % g
            tab[c] = (m == 0) | (m == d)
        rows.append(tab)
    return rows


def blocked(rows, genome):
    b = np.zeros(rows[0].shape[1], dtype=bool)
    for tab, c in zip(rows, genome):
        b |= tab[c]
    return b


def longest_with_holes(b, h):
    """longest stretch (slots) containing <= h openings; returns (len, start)."""
    op = np.flatnonzero(~b)
    R = len(b)
    if len(op) <= h:
        return R, 0
    # stretch between opening j-1 and opening j+h (exclusive), padded ends
    pad = np.concatenate([[-1], op, [R]])
    best, bs = 0, 0
    for j in range(1, len(pad) - h):
        L = pad[j + h] - pad[j - 1] - 1
        if L > best:
            best, bs = int(L), int(pad[j - 1] + 1)
    return best, bs


def longest_sparse(b, q2, t=2):
    """longest stretch where every q2-window inside holds <= t openings; returns (len, start)."""
    R = len(b)
    if R < q2:
        return 0, 0
    cs = np.concatenate([[0], np.cumsum(~b)])
    cq = cs[q2:] - cs[:-q2]  # openings in [k, k+q2)
    ok = cq <= t
    best, bs, run, rs = 0, 0, 0, 0
    for k, o in enumerate(ok):
        if o:
            if run == 0:
                rs = k
            run += 1
            if run + q2 - 1 > best:
                best, bs = run + q2 - 1, rs
        else:
            run = 0
    return best, bs


def fitness(rows, genome, target, q2):
    b = blocked(rows, genome)
    if target == "F":
        return longest_with_holes(b, 0)
    if target == "F2":
        return longest_with_holes(b, 1)
    if target == "F3":
        return longest_with_holes(b, 2)
    return longest_sparse(b, q2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", type=int)
    ap.add_argument("qnext", type=int)
    ap.add_argument("--target", default="F3")
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--R", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=4)
    a = ap.parse_args()
    gears = [g for g in PR if g <= a.q]
    rng = np.random.default_rng(a.seed)
    rows = build_rows(gears, a.R)
    t0 = time.time()
    gbest = (0, None, None)
    for rs in range(a.restarts):
        pop = [np.array([rng.integers(g) for g in gears]) for _ in range(a.pop)]
        fit = [fitness(rows, p, a.target, a.qnext)[0] for p in pop]
        stall = 0
        for gen in range(a.gens):
            # tournament parent, mutate 1-3 gears, replace worst if better
            i, j = rng.integers(a.pop, size=2)
            par = pop[i] if fit[i] >= fit[j] else pop[j]
            child = par.copy()
            for _ in range(rng.integers(1, 4)):
                k = rng.integers(len(gears))
                if rng.random() < 0.5:
                    child[k] = rng.integers(gears[k])
                else:
                    child[k] = (child[k] + rng.choice([-1, 1])) % gears[k]
            # crossover sometimes
            if rng.random() < 0.2:
                other = pop[rng.integers(a.pop)]
                mask = rng.random(len(gears)) < 0.5
                child[mask] = other[mask]
            f = fitness(rows, child, a.target, a.qnext)[0]
            w = int(np.argmin(fit))
            if f >= fit[w]:
                pop[w], fit[w] = child, f
            mx = max(fit)
            if mx > gbest[0]:
                gbest = (mx, pop[int(np.argmax(fit))].copy(), gen)
                stall = 0
            else:
                stall += 1
            if stall > a.gens // 2:
                break
    L, genome, gen = gbest
    b = blocked(rows, genome)
    val, s = fitness(rows, genome, a.target, a.qnext)
    seg = b[s:s + val]
    holes = [int(x) for x in np.flatnonzero(~seg)]
    # which gears cover which slot in the stretch (first gear that blocks it)
    cover = []
    for i in range(s, s + val):
        who = [g for g, tab, c in zip(gears, rows, genome) if tab[c][i]]
        cover.append(who)
    counts = {g: sum(1 for w in cover if g in w) for g in gears}
    sole = {g: sum(1 for w in cover if w == [g]) for g in gears}
    print(f"{'+'.join(map(str, gears))} q'={a.qnext} target={a.target}: best={val} (blocked-count {val - 1 - len(holes) if a.target != 'G2' else val - 1}) "
          f"holes at offsets {holes}  time {time.time() - t0:.0f}s")
    print(f"    hits per gear inside stretch: {counts}")
    print(f"    slots covered by that gear alone: {sole}")


if __name__ == "__main__":
    main()

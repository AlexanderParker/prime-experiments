"""Round 14 lateral: BOUNDING PADDED RUNS.

A padded link joins two kills at the SAME tooth, so its spacing is = 0 mod q',
i.e. a gap of M of size exactly q', 2q', ... (<= F(M)). Padding is therefore
possible only from the step where F(M) >= q'.

THE PADDING LEMMA (spectrum argument, exact).  A legal killed run of k kills
occupies k+1 CONSECUTIVE gaps of M (the k-1 links plus the two flanks), so its
merged gap is G <= F_{k+1}(M).  Now suppose a run carries TWO padded links with
j literal links between them (j >= 0).  Those j+2 links are j+2 consecutive
gaps of M summing to at least

        2q' + j*L,      L = min(s, q'-s) = the cheapest literal link,

hence two padded links REQUIRE  F_{j+2}(M) >= 2q' + j*L  for some j >= 0.
Contrapositive - the bound this round delivers:

    if  F_{j+2}(M) < 2q' + j*L  for every j >= 0,
    then EVERY legal killed run carries AT MOST ONE padded link.

j = 0 (two adjacent padded links) gives the headline criterion  F_2(M) < 2q'.

This file: checks the criterion at every step from the measured F_j spectra,
finds where it first fails, and censuses the padded links empirically.
"""
import sys
from math import prod

import numpy as np

from split_gap_law import primes

# full-period F_j spectra (mechanic, machines 13..31); machine 37 partial (>=)
SPEC = {13: [11, 16, 23, 26, 28, 31], 17: [18, 25, 28, 33, 35, 40],
        19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
        29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97]}
SPEC_LB = {37: [88, 90]}
STEPS = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37), (37, 41)]

def L_of(qp):
    u = pow(6, -1, qp)
    s = (2 * u) % qp
    return min(s, qp - s)

def criterion(y, qp):
    spec = SPEC.get(y) or SPEC_LB.get(y)
    lb = y in SPEC_LB
    L = L_of(qp)
    F = spec[0]
    rows = []
    ok = True
    for j in range(0, len(spec) - 1):
        need = 2 * qp + j * L
        have = spec[j + 1]
        rows.append((j, have, need, have < need))
        if have >= need:
            ok = False
    return F, L, rows, ok, lb

def part1():
    print("=" * 74)
    print("PART 1: the padding lemma, checked from the F_j spectra")
    print("  (padding possible at all only when F(M) >= q')")
    for y, qp in STEPS:
        if y not in SPEC and y not in SPEC_LB:
            continue
        F, L, rows, ok, lb = criterion(y, qp)
        poss = "possible" if F >= qp else "IMPOSSIBLE (F(M) < q')"
        mark = ">=" if lb else "="
        print(f"  step {y:>2}->{qp:<2}: F(M){mark}{F:>3}, q'={qp:>2}, L={L:>2}"
              f"  padding {poss}")
        for j, have, need, good in rows[:3]:
            print(f"      j={j}: F_{j+2}(M) {mark} {have:>3}  vs  2q'+jL = "
                  f"{need:>3}   {'excluded' if good else 'NOT EXCLUDED'}")
        print(f"      => at most one padded link per run: "
              f"{'YES (proved)' if ok else 'NOT PROVED'}"
              + ("  [lower-bound spectrum: exclusion cannot be claimed]" if lb else ""))

def part2(lim=29, chunk=100_000_000):
    print("=" * 74)
    print("PART 2: empirical census - padded links per legal killed run")
    for y, qp in STEPS:
        if y > lim:
            continue
        gears = primes(5, y)
        P = prod(gears)
        u = pow(6, -1, qp)
        A, B = (2 * u) % qp, (-2 * u) % qp
        padcnt = {}
        runhist = {}
        adj = 0
        gapq = {}
        tail = None
        a = 0
        while a < P:
            S = min(chunk, P - a)
            killed = np.zeros(S, bool)
            for q in gears:
                uq = pow(6, -1, q)
                for t in (uq, q - uq):
                    killed[(t - a) % q::q] = True
            o = np.flatnonzero(~killed).astype(np.int64) + a
            if tail is not None:
                o = np.concatenate((tail, o))
            d = np.diff(o)
            pad = (d % qp == 0)
            for v in np.unique(d[pad]):
                gapq[int(v)] = gapq.get(int(v), 0) + int((d[pad] == v).sum())
            if len(pad) > 1:
                adj += int((pad[:-1] & pad[1:]).sum())
            r = d % qp
            cls = np.where(r == 0, 0, np.where(r == A, 1, np.where(r == B, 2, 3)))
            pos = np.arange(len(cls))
            letter = cls != 0
            lastl = np.maximum.accumulate(np.where(letter, pos, -1))
            prev = np.full(len(cls), -1)
            prev[1:] = lastl[:-1]
            bad = letter & (prev >= 0) & (cls[np.maximum(prev, 0)] == cls)
            good = ~((cls == 3) | bad)
            idx = np.flatnonzero(good)
            if len(idx):
                cut = np.flatnonzero(np.diff(idx) != 1)
                starts = np.concatenate(([idx[0]], idx[cut + 1]))
                ends = np.concatenate((idx[cut], [idx[-1]]))
                for st, en in zip(starts, ends):
                    p = int((cls[st:en + 1] == 0).sum())
                    k = en - st + 2
                    padcnt[p] = padcnt.get(p, 0) + 1
                    runhist[k] = runhist.get(k, 0) + 1
            tail = o[-400:]
            a += S
        print(f"  step {y}->{qp}: gaps = 0 mod q' by value {gapq if gapq else '{} (none)'}; "
              f"adjacent padded pairs {adj}")
        print(f"      run length (kills) histogram {dict(sorted(runhist.items()))}")
        print(f"      padded links per run {dict(sorted(padcnt.items()))} "
              f"-> MAX = {max(padcnt) if padcnt else 0}")

if __name__ == "__main__":
    part1()
    part2(int(sys.argv[1]) if len(sys.argv) > 1 else 29)

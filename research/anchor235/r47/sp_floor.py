"""Item 2 / item 6: WHY s = 0 never happens on an extremal run -- the counting floor.

For a run, let a_g = the number of LEFT-flank columns gear g strikes and b_g = the number of
RIGHT-flank columns it strikes.  A separation (s = 0) needs a partition G = A u B with

    sum_{g in A} a_g >= L-1     and     sum_{g in B} b_g >= R-1,

because each flank column must be taken by a gear on its own side and a gear takes at most
a_g (resp. b_g) of them.  This is a knapsack test that uses only the COUNTS, not which columns.
If no partition passes it, s = 0 is impossible for a counting reason and no arrangement of the
teeth could rescue it.

Reported: how often the counting test alone already forbids separation, on the real machine's
attaining 3-runs at m13..m31 and on the family's, split trivial / hard.  Also the harmonic
capacity sum S(M) = sum_g 2/g, which is what makes the test bite: a flank of length n needs
gears of total capacity sum 2/g >= 1 to cover it, and both flanks together need 2, while
S(M) = 1.02, 1.14, 1.24, 1.33, 1.40, 1.47 at m13..m31 -- always less than 2.
"""
import sys, os, json, random
from math import prod
import numpy as np
from sp_core import (gears_of, us_of, sieve, gap_stats, attaining_runs, run_masks,
                     separability)

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PR = [5, 7, 11, 13, 17, 19]


def counting_ok(ml, mr, nl, nr):
    """is there a partition of the gears passing the count test?"""
    n = len(ml)
    a = [bin(m).count('1') for m in ml]
    b = [bin(m).count('1') for m in mr]
    for A in range(1 << n):
        sa = sum(a[i] for i in range(n) if (A >> i) & 1)
        if sa < nl:
            continue
        sb = sum(b[i] for i in range(n) if not (A >> i) & 1)
        if sb >= nr:
            return True
    return False


def sieve_w(gears, ws):
    P = prod(gears)
    bl = np.zeros(P, dtype=bool)
    for g, w in zip(gears, ws):
        bl[w % g::g] = True
        bl[(-w) % g::g] = True
    return P, bl


def main(out):
    out.write("harmonic capacity S(M) = sum 2/g:  " + "  ".join(
        f"m{t}: {sum(2/g for g in gears_of(t)):.4f}" for t in (13, 17, 19, 23, 29, 31)) + "\n")
    tot = [0, 0, 0, 0]      # hard: countfail, hardn, trivial: countfail, trivn
    for top in (13, 17, 19, 23):
        gears, us = gears_of(top), us_of(gears_of(top))
        P, bl = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, bl)
        cnt = {}
        for (x0, L, v, R) in attaining_runs(opens, gaps, N):
            ml, mr, nl, nr = run_masks(gears, us, x0, L, v, R)
            ok = counting_ok(ml, mr, nl, nr)
            s = separability(ml, mr, nl, nr)['s']
            k = ('HARD' if v < min(L, R) else 'TRIV')
            c = cnt.setdefault(k, [0, 0, 0])
            c[0] += 1
            c[1] += (not ok)
            c[2] += (s == 0)
        out.write(f"  m{top}: " + "; ".join(
            f"{k} n={c[0]}, counting forbids separation at {c[1]}, actually separable {c[2]}"
            for k, c in sorted(cnt.items())) + "\n")
    for top, path in ((29, 'sep_deep_m29.json'), (31, 'sep_deep_m31.json')):
        gears, us = gears_of(top), us_of(gears_of(top))
        with open(os.path.join(RES, path)) as f:
            rows = json.load(f)
        cnt = {}
        for r in rows:
            ml, mr, nl, nr = run_masks(gears, us, r['x0'], r['L'], r['v'], r['R'])
            ok = counting_ok(ml, mr, nl, nr)
            k = 'HARD' if r['hard'] else 'TRIV'
            c = cnt.setdefault(k, [0, 0, 0])
            c[0] += 1
            c[1] += (not ok)
            c[2] += (r['s'] == 0)
        out.write(f"  m{top}: " + "; ".join(
            f"{k} n={c[0]}, counting forbids separation at {c[1]}, actually separable {c[2]}"
            for k, c in sorted(cnt.items())) + "\n")
    # the family, same test
    rng = random.Random(5150)
    for top in (17, 19):
        gears = [p for p in PR if p <= top]
        realw = tuple(min(u, g - u) for g, u in zip(gears, us_of(gears)))
        cnt = {}
        seen = {realw}
        nm = 0
        while nm < 100:
            ws = tuple(rng.randrange(1, (g + 1) // 2) for g in gears)
            if ws in seen:
                continue
            seen.add(ws)
            nm += 1
            P, bl = sieve_w(gears, ws)
            opens, gaps, F, F2, N = gap_stats(P, bl)
            for (x0, L, v, R) in attaining_runs(opens, gaps, N):
                ml, mr, nl, nr = run_masks(gears, ws, x0, L, v, R)
                ok = counting_ok(ml, mr, nl, nr)
                s = separability(ml, mr, nl, nr)['s']
                k = 'HARD' if v < min(L, R) else 'TRIV'
                c = cnt.setdefault(k, [0, 0, 0])
                c[0] += 1
                c[1] += (not ok)
                c[2] += (s == 0)
        out.write(f"  FAMILY m{top} ({nm} members): " + "; ".join(
            f"{k} n={c[0]}, counting forbids separation at {c[1]}, actually separable {c[2]}"
            for k, c in sorted(cnt.items())) + "\n")
        out.flush()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()

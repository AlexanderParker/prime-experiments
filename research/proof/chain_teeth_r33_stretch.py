"""Prover C r33 -- print the coverage of a J-run: per gear, which interior columns it strikes,
sole-coverer counts per flank, and whether a CRT recombination of the two flanks exists
(a partition A|B of the gears with A alone covering the left flank and B alone the right)."""
import itertools, sys, json
import numpy as np
from math import prod
sys.path.insert(0, 'research/proof')
from chain_teeth_r33 import *

def stretch_report(y, teeth, q1, a, which, verbose=True):
    gears = gears_of(y); P = prod(gears)
    mk = open_mask(gears, teeth, P); ops = np.flatnonzero(mk); g = gaps_of(mk)
    b = q1 - a
    gp = np.roll(g, 1); gn = np.roll(g, -1); gnn = np.roll(g, -2)
    if which == 'A':   sel = g == a; fl = gp + gn; m = 1
    elif which == 'Q': sel = g == q1; fl = gp + gn; m = 1
    elif which == 'B': sel = g == b; fl = gp + gn; m = 1
    else:              sel = ((g == a) & (gn == b)) | ((g == b) & (gn == a)); fl = gp + gnn; m = 2
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        print(f"  {which}: no occurrence"); return
    k = int(fl[idx].argmax()); i = int(idx[k])
    x0 = int(ops[i - 1]); word = [int(g[(i + t) % g.size]) for t in range(m)]
    gL = int(g[i - 1]); gR = int(g[(i + m) % g.size]); span = gL + sum(word) + gR
    cols = [(x0 + t) % P for t in range(span + 1)]
    strikes = {q: np.array([(c % q) in (v % q, (-v) % q) for c in cols]) for q, v in zip(gears, teeth)}
    cover = sum(s.astype(int) for s in strikes.values())
    F = int(g.max())
    print(f"  {which}: ({gL}) + {word} + ({gR}) = {span}; F={F}; teeth {tuple(teeth)}; openings at offsets {[t for t in range(span + 1) if cover[t] == 0]}")
    left = range(1, gL); right = range(gL + sum(word) + 1, span)
    for q in gears:
        sl = [t for t in left if strikes[q][t]]; sr = [t for t in right if strikes[q][t]]
        soleL = [t for t in sl if cover[t] == 1]; soleR = [t for t in sr if cover[t] == 1]
        if verbose:
            print(f"     gear {q:2d}: left strikes {sl} sole {soleL} | right strikes {sr} sole {soleR}")
    # CRT recombination: partition of gears
    found = None
    for r in range(len(gears) + 1):
        for A in itertools.combinations(gears, r):
            B = [q for q in gears if q not in A]
            okL = all(any(strikes[q][t] for q in A) for t in left)
            okR = all(any(strikes[q][t] for q in B) for t in right)
            if okL and okR:
                found = (A, tuple(B)); break
        if found: break
    print(f"     CRT recombination of the two flanks (A covers left, B covers right): {found if found else 'NONE'};"
          f" sole-coverer gears left {[q for q in gears if any(strikes[q][t] and cover[t]==1 for t in left)]},"
          f" right {[q for q in gears if any(strikes[q][t] and cover[t]==1 for t in right)]}")

if __name__ == '__main__':
    print("REAL MACHINE binding occurrences")
    for y in [13, 17, 19, 23]:
        gears = gears_of(y); teeth = [real_tooth(q) for q in gears]; q1 = next_prime(y); a = letter_a(q1, real_tooth(q1))
        print(f"m{y} q'={q1} a={a}")
        for which in ['A', 'B', 'Q', 'AB']:
            stretch_report(y, teeth, q1, a, which, verbose=(y == 19))
    print("\nm17 PINNED VIOLATORS")
    for teeth in [(1, 3, 2, 1, 6), (2, 3, 3, 2, 4), (2, 3, 3, 3, 3)]:
        print(f"teeth {teeth}")
        for which in ['A', 'B']:
            stretch_report(17, list(teeth), 19, 6, which, verbose=True)

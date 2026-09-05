"""Branch 2f.i - mechanism: the violating chain, and the pair overlaps it uses.

For a member and its maximal word-legal run: the stretch, the kills by q', the letters, and
then, pair of gears by pair of gears, the columns of the stretch struck by BOTH gears and the
distances between them.  The four residues two gears strike together are a translate of
{0, S_g, S_h, S_g + S_h} mod gh; the two diagonals are d+ = CRT(s_g, s_h) and
d- = CRT(s_g, -s_h) folded.  For the real machine's separation s = 2 x 6^{-1} the coherent
diagonal is 3^{-1} mod gh, never below gh/3 (W3 N-S3), so a double-strike pair at a distance
below gh/3 on the coherent diagonal is a configuration the machine's own separations forbid.

Usage: uv run python research/anchor235/r46/cp_mech.py
"""
import os
import sys
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "proof"))
from cp_compat import admissible, best_core, incompat, real_tooth  # noqa: E402
from chain_family_r32 import (gaps_of, gears_of, letter_a, open_mask,  # noqa: E402
                              qstar_table, summarize)

B = 30


def struck(q, v, x):
    return x % q in ((v) % q, (-v) % q)


def report(y, q1, teeth, v1, tag):
    gears = gears_of(y)
    gs = gears + [q1]
    P = prod(gears)
    mask = open_mask(gears, teeth, P)
    g = gaps_of(mask)
    a = letter_a(q1, v1)
    F, F2, tab = qstar_table(g, q1, a)
    s = summarize(F, F2, tab, q1)
    rats = admissible(gs, B)
    k, rc, core = best_core(gs, list(teeth) + [v1], rats)
    print("=" * 100)
    print("%s   M={5..%d}, q'=%d, teeth=%s, v'=%d, a=%d, b=%d" % (tag, y, q1, teeth, v1, a, q1 - a))
    print("  separations s_q = %s ; real would be %s"
          % ([(2 * v) % q for q, v in zip(gears, teeth)],
             [(2 * real_tooth(q)) % q for q in gears]))
    print("  best rational c/r = %d/%d, coherent core %s (k=%d of %d), incompatible pairs I=%d"
          % (rc[1], rc[0], core, k, len(gs), incompat(len(gs), k)))
    print("  F=%d  F_2=%d  budget F+q'=%d  max_J Q*_J=%d  %s"
          % (F, F2, F + q1, s["chain"], "VIOLATES by %d" % (s["chain"] - F - q1)
             if s["chain"] > F + q1 else "holds"))
    # the maximising run
    Jb = max(tab, key=lambda J: tab[J]["Q"])
    rec = tab[Jb]
    print("  maximising run: J=%d  (%d) + %s + (%d) = %d   %s"
          % (Jb, rec["gL"], list(rec["word"]), rec["gR"], rec["Q"],
             "literal" if rec["literal"] else "padded"))
    # locate an occurrence
    ops = [int(x) for x in (mask.nonzero()[0])]
    import numpy as np
    gg = np.array(g)
    n = gg.size
    w = list(rec["word"])
    found = None
    for i in range(n - len(w) - 1):
        if list(gg[i + 1:i + 1 + len(w)]) == w and gg[i] == rec["gL"] and gg[i + 1 + len(w)] == rec["gR"]:
            found = i
            break
    if found is None:
        print("  (no occurrence located)")
        return
    x0 = ops[found]
    cols = [x0]
    for gap in [rec["gL"]] + w + [rec["gR"]]:
        cols.append(cols[-1] + gap)
    print("  occurrence: openings of M at %s" % cols)
    print("  offsets    : %s" % [c - cols[0] for c in cols])
    print("  interior openings and their residues mod q'=%d: %s"
          % (q1, [(c - cols[0], (c) % q1) for c in cols[1:-1]]))
    lo, hi = cols[0], cols[-1]
    span = hi - lo
    print("  stretch [%d, %d], span %d" % (lo, hi, span))
    # pair analysis
    print("  pair analysis over the stretch (columns struck by BOTH gears):")
    print("   pair      gh   d+   d-   gh/3   double-struck offsets in the stretch   distances")
    flagged = []
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            gq, hq = gears[i], gears[j]
            gh = gq * hq
            sg, sh = (2 * teeth[i]) % gq, (2 * teeth[j]) % hq
            ig, ih = pow(hq, -1, gq), pow(gq, -1, hq)
            dp = (sg * hq * ig + sh * gq * ih) % gh
            dm = (sg * hq * ig + (-sh % hq) * gq * ih) % gh
            dp, dm = min(dp, gh - dp), min(dm, gh - dm)
            ds = [x - lo for x in range(lo, hi + 1)
                  if struck(gq, teeth[i], x) and struck(hq, teeth[j], x)]
            dist = sorted({b - a2 for ai, a2 in enumerate(ds) for b in ds[ai + 1:]})
            short = [d for d in dist if d < gh / 3]
            if short:
                flagged.append((gq, hq, short))
            print("   (%2d,%2d) %5d %4d %4d %6.1f   %-32s %s%s"
                  % (gq, hq, gh, dp, dm, gh / 3, ds, dist,
                     "   <- distance below gh/3" if short else ""))
    print("  pairs realising a double-strike closer than gh/3: %s" % flagged)


if __name__ == "__main__":
    # the 2f refuting member (prover C's 23 -> 29 sweep)
    report(23, 29, [1, 1, 4, 2, 7, 1, 5], 5, "2f REFUTING MEMBER (incompatible, k=4)")
    # the fully compatible violators
    report(17, 19, [1, 3, 4, 4, 4], 4, "FULLY COMPATIBLE VIOLATOR c/r = 8/1")
    report(17, 19, [2, 3, 3, 3, 3], 3, "FULLY COMPATIBLE VIOLATOR c/r = 6/1")
    # the real machine as control
    report(17, 19, [1, 1, 2, 2, 3], 3, "REAL MACHINE m17 (control, c/r = 1/3)")
    report(23, 29, [1, 1, 2, 2, 3, 3, 4], 5, "REAL MACHINE m23 (control, c/r = 1/3)")

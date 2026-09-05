"""Item 4: the record's separability across rungs.

R3.h: a record stretch of M' = {5..q'} is a row of ORDINARY lower gaps whose junctions the top
three gears strike.  Read the record interval [X, Y] (Y = X + F) as a J-run of a LOWER machine
M = {5..q}: the survivors of M inside are X = s_0 < s_1 < ... < s_k = Y, giving gaps
g_1 .. g_k.  The J-run's flanks are the interior of g_1 and the interior of g_k.  This script
computes the shared number s and the shared gears of that J-run at every layer q, and asks
whether the shared gears are the top gears -- i.e. whether "made at the top" and "separable
except at the top" are the same fact.
"""
import sys, os
from sp_core import gears_of, us_of, sieve, gap_stats, separability, run_masks, letters

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# record starts on record (ends_or_middles.md section 1); m13..m23 are re-derived here.
KNOWN = {29: [200906185, 877375977], 31: [1468940242, 21844264615]}


def strikes(g, u, k):
    return k % g in (u % g, (-u) % g)


def blocked(gears, us, k):
    return any(strikes(g, u, k) for g, u in zip(gears, us))


def record_starts(top):
    if top in KNOWN:
        return KNOWN[top]
    gears, us = gears_of(top), us_of(gears_of(top))
    P, b = sieve(gears, us)
    opens, gaps, F, F2, N = gap_stats(P, b)
    import numpy as np
    return [int(opens[i]) for i in np.flatnonzero(gaps == F).tolist()]


def survivors(gears, us, X, Y):
    return [k for k in range(X, Y + 1) if not blocked(gears, us, k)]


def jrun_sep(gears, us, S):
    """S = survivor positions X = s_0 < ... < s_k = Y.  Flanks: interior of the first gap and
    of the last gap.  Returns the separability dict plus the two flank lengths."""
    L = S[1] - S[0]
    R = S[-1] - S[-2]
    x0 = S[0]
    v = S[-2] - S[1]        # the span of the middles
    ml, mr, nl, nr = run_masks(gears, us, x0, L, v, R)
    return separability(ml, mr, nl, nr), L, v, R


def main(out, tops=(17, 19, 23, 29, 31)):
    for top in tops:
        G = gears_of(top)
        U = us_of(G)
        F = {17: 18, 19: 25, 23: 34, 29: 43, 31: 58}[top]
        starts = record_starts(top)
        out.write(f"\n===== m{top}: F={F}, {len(starts)} record stretch(es) "
                  f"(first two shown)\n")
        for X in starts[:2]:
            Y = X + F
            assert not blocked(G, U, X) and not blocked(G, U, Y), "ends not open"
            assert all(blocked(G, U, k) for k in range(X + 1, Y)), "interior not blocked"
            out.write(f"  record at x={X}\n")
            for q in G[:-1]:
                gears = [g for g in G if g <= q]
                us = [u for g, u in zip(G, U) if g <= q]
                S = survivors(gears, us, X, Y)
                if len(S) < 3:
                    out.write(f"    layer {q:2d}: {len(S)-1} pieces -- too few for a J-run\n")
                    continue
                r, L, v, R = jrun_sep(gears, us, S)
                sh = [gears[i] for i in range(len(gears)) if (r['sharedmask'] >> i) & 1]
                word = [S[i + 1] - S[i] for i in range(len(S) - 1)]
                # which gears of the FULL machine strike the interior survivors (the gluers)
                glu = {}
                for k in S[1:-1]:
                    for g, u in zip(G, U):
                        if g > q and strikes(g, u, k):
                            glu.setdefault(g, []).append(k - X)
                out.write(f"    layer {q:2d}: {len(word)} pieces {word}; flanks "
                          f"(g_1,g_J)=({L},{R}) span v={v}; s={r['s']} u={r['u']} "
                          f"sigma={r['sigma']:.3f} ov={r['ov']} loss={r['loss']}; "
                          f"shared {sh} classes {[letters(g, v) for g in sh]}; "
                          f"gluers above {q}: "
                          f"{ {g: v2 for g, v2 in sorted(glu.items())} }\n")
            out.flush()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()

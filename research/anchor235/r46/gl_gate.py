"""G1: reproduce 2g.i's glue table from the covering formulation alone (no sieve in the test).

For each machine m13..m23: sieve the full period once to FIND the attaining 3-runs, then decide
each one by the covering condition (which uses only x0 mod g).  Also cross-check the covering
verdict against a direct machine lookup at the CRT point on a sample, so the two definitions are
shown to agree.
"""
import sys
import numpy as np
from gl_glue import (gears_of, us_of, sieve, gap_stats, glue, runs_with, covbits, cov_pair)


def crt(mods, rems):
    z, Mm = 0, 1
    for m, r in zip(mods, rems):
        t = ((r - z) * pow(Mm, -1, m)) % m
        z += Mm * t
        Mm *= m
    return z % Mm


def main(out):
    tot = totok = 0
    for top in (13, 17, 19, 23):
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        att = {v: s for v, s in N.items() if v >= 6}
        n = ok = 0
        fails = []
        checked = 0
        for (x0, L, v, R) in runs_with(opens, gaps, vmin=6, only_attaining=att):
            n += 1
            sig = glue(gears, us, x0, L, v, R)
            if sig is None:
                fails.append((v, L, R, x0))
            else:
                ok += 1
                if checked < 40:       # machine cross-check of the covering verdict
                    checked += 1
                    rems = [(x0 + 1 + v) % g if s else (x0 + 1) % g
                            for g, s in zip(gears, sig)]
                    z = crt(gears, rems)
                    T = L + R - 1
                    pat = [bool(blocked[(z + j) % P]) for j in range(T)]
                    assert all(pat[:L - 1]) and not pat[L - 1] and all(pat[L:]), (top, x0, L, v, R)
        tot += n
        totok += ok
        byv = {}
        for v, L, R, x0 in fails:
            byv.setdefault(v, []).append((L, R, x0))
        out.write(f"m{top}: F={F} F_2={F2}  attaining 3-runs v>=6: {n}  glue(C2) ok {ok} "
                  f"({100*ok/n:.1f}%)  fail {n-ok} at v={sorted(byv)} "
                  f"counts={{ {', '.join(f'{v}:{len(byv[v])}' for v in sorted(byv))} }}\n")
        for v in sorted(byv):
            for L, R, x0 in byv[v]:
                out.write(f"    FAIL m{top} v={v} (L,v,R)=({L},{v},{R}) sum={L+R} "
                          f"(F{L+R-F:+d}, F_2{L+R-F2:+d}) x0={x0}\n")
        out.flush()
    out.write(f"TOTAL m13..m23: {totok}/{tot} = {100*totok/tot:.1f}%\n")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, "w") if dest else sys.stdout
    main(o)
    if dest:
        o.close()

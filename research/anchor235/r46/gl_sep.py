"""Item 6 / item 5: the separation profile and the reach of the glue.

The glue never uses the fact that x_1 and x_2 are CONSECUTIVE openings.  For any two openings
p and p' = p + S it glues the gap ENDING at p to the gap STARTING at p', certifying
F_2(M) >= leftgap(p) + rightgap(p').  So the natural object is

    N*(S) = max over openings p with p+S an opening of ( leftgap(p) + rightgap(p+S) ),

with N*(v) >= N(v) (the 3-run profile is the sub-case where p, p+S are consecutive) and
N*(0) = F_2.  For a J-run, S is the span of the middles, so the J = 4 version of the glue
bounds g_1 + g_4 and hence Q*_4 <= F_2 + (middle sum).

Reported per machine: N*(S) for S up to SMAX, the least S >= 6 where N*(S) > F_2 (where the
covering statement must fail, since a valid colouring would certify F_2 >= N*(S)), and the
C2 rate on the extremal pairs at a few separations.
"""
import sys
import numpy as np
from gl_glue import gears_of, us_of, sieve, gap_stats, cov_pair, solve_cover


def main(out, tops=(13, 17, 19, 23), smax=120):
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        sm = min(smax, 80) if top >= 23 else smax
        rg = np.zeros(P, dtype=np.int16)
        rg[opens] = gaps
        lg = np.roll(gaps, 1)
        Ns = {}
        wit = {}
        for S in range(1, sm + 1):
            q = opens + S
            q[q >= P] -= P
            b = rg[q]
            sel = b > 0
            if not sel.any():
                continue
            tot = lg[sel].astype(np.int64) + b[sel]
            k = int(np.argmax(tot))
            Ns[S] = int(tot.max())
            idx = np.flatnonzero(sel)[k]
            wit[S] = (int(opens[idx]), int(lg[idx]), S, int(b[sel][k]))
        first = [S for S in sorted(Ns) if S >= 6 and Ns[S] > F2]
        out.write(f"\nm{top}: F={F} F_2={F2}; N*(S) for S=1..{sm}\n")
        out.write("   " + " ".join(f"{S}:{Ns[S]}" for S in sorted(Ns)) + "\n")
        out.write(f"   least S >= 6 with N*(S) > F_2: {first[0] if first else 'none'} "
                  f"(all such S: {first[:12]}{'...' if len(first) > 12 else ''})\n")
        out.write(f"   max_(S>=6) N*(S) = {max(Ns[S] for S in Ns if S >= 6)} "
                  f"vs F_2={F2}, 2F={2*F}\n")
        # C2 on the extremal pair at selected separations
        for S in sorted(Ns):
            if S < 6 or (S > 24 and S % 10):
                continue
            p, L, _, RR = wit[S]
            T, h, covL, covR = cov_pair(gears, us, p - L, L, RR, p + S)
            sol = solve_cover(covL, covR, T, h)
            out.write(f"     S={S:3d} extremal pair L={L} R={RR} sum={Ns[S]} "
                      f"({'<=' if Ns[S] <= F2 else '> '} F_2)  C2 "
                      f"{'OK' if sol else 'FAIL'}\n")
        out.flush()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, "w") if dest else sys.stdout
    main(o)
    if dest:
        o.close()

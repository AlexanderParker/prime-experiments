"""Constructor round 19: THE WINDOW COMPOSITION PROFILE - a new construct.

Object nobody has built: for a machine M and next gear q', the JOINT profile
of a gap-window - its composition (which gap VALUES sit where), its sum, and
whether its interiors qualify (values = 0, +-2c mod q'). Built as one object
so the relationship between "how big a window is" and "whether it can merge"
can be measured instead of assumed.

THE DECISIVE TEST (luck vs structure). If the qualifying condition were
independent of the window sum - the "arithmetic luck" reading - then the
qualifying windows would be a random ~p-sample of all windows, and the top of
the spectrum would almost surely contain one. So measure the EXCLUSION ZONE:
    Z = #{windows with sum > (max qualifying sum)}
and the luck probability (1-p)^Z. If that is astronomically small, the
suppression is STRUCTURE, and the structure is the mechanism (D) needs.

Also measured: the composition migration - max element / sum of the extremal
j-window as j grows (does the top of the spectrum move from "one huge gap" to
"several medium gaps"?).
"""
import numpy as np
import sys
sys.path.insert(0, "research")
from fuel_bound import gapword
from word_ceiling import FK, STEPS

CH = 30_000_000


def profile(y, q1, depths=(3, 4, 5)):
    g = gapword(y).astype(np.int16)
    n = len(g)
    c = pow(6, -1, q1)
    Q = {0, (2 * c) % q1, (-2 * c) % q1}
    F = FK[y]
    print(f"\n=== machine {y}, q'={q1}  (F={F}, {n:,} gaps)  "
          f"qualifying residues mod {q1}: {sorted(Q)}")
    for j in depths:
        # rolling sums and qualifying mask, chunked
        gmax = qmax = 0
        nq = ntot = 0
        best_comp = None
        for lo in range(0, n - j, CH):
            hi = min(lo + CH, n - j)
            idx = np.arange(lo, hi)
            s = np.zeros(len(idx), np.int32)
            for t in range(j):
                s += g[idx + t]
            ok = np.ones(len(idx), bool)
            for t in range(1, j - 1):               # interiors
                ok &= np.isin(g[idx + t] % q1, list(Q))
            ntot += len(idx)
            nq += int(ok.sum())
            m = int(s.max())
            if m > gmax:
                gmax = m
                a = int(idx[int(np.argmax(s))])
                best_comp = g[a:a + j].tolist()
            if ok.any():
                mq = int(s[ok].max())
                if mq > qmax:
                    qmax = mq
        # exclusion zone: windows with sum > qmax
        Z = 0
        for lo in range(0, n - j, CH):
            hi = min(lo + CH, n - j)
            idx = np.arange(lo, hi)
            s = np.zeros(len(idx), np.int32)
            for t in range(j):
                s += g[idx + t]
            Z += int((s > qmax).sum())
        p = nq / ntot
        luck = Z * np.log10(1 - p) if p < 1 else 0
        print(f"  j={j}: F_{j}={gmax} comp={best_comp} (max/sum="
              f"{max(best_comp)/gmax:.2f})  qualifying max={qmax} "
              f"(gap {gmax-qmax})")
        print(f"        p(qualify)={p:.4f}  exclusion zone Z={Z:,}  "
              f"luck prob = 10^{luck:.1f}"
              f"{'   <== STRUCTURE' if luck < -6 else '   (luck plausible)'}")


if __name__ == "__main__":
    for y, q1 in STEPS[2:]:                          # 17->19 .. 29->31
        profile(y, q1)

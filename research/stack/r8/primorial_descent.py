"""The primorial descent (owner, 2026-09-14): from home flip up on the full primorial q#, then
flip back down on partial primorials (q# without q, then without q', ...), k periods each, until
the partial primorial P_s is the first one above q/2.  Every landing is 2 t P_s - 1 with t chosen
by the periods; it is open to every gear of P_s by construction (residue -1 carried), and its
residue at a gear above the base is set by t.  The landing is in the window iff
(q + 1) / (2 P_s) <= t <= (q^2 - 1) / (2 P_s).

Per machine: P_s, the base it carries, the range of t, the t whose landing is a twin, the first,
and which gear strikes each failing t (the smallest gear dividing a member).  Also the descent
itself for the first twin: the periods k_i at each partial primorial.

usage: uv run python research/stack/r8/primorial_descent.py 2000
"""
import sys
import numpy as np
from sympy import primerange, factorint
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    tot = 0; has = 0; nopass = []; first_t = {}
    for q in qs:
        ps = list(primerange(2, q + 1))
        prims = []; P = 1
        for p in ps: P *= p; prims.append((p, P))
        # P_s = first primorial above q/2
        s = next(i for i, (p, P) in enumerate(prims) if P > q / 2)
        Ps = prims[s][1]; base = [p for p, _ in prims[:s + 1]]
        t_lo = -(-(q + 1) // (2 * Ps)); t_hi = (q * q - 1) // (2 * Ps)
        ts = range(t_lo, t_hi + 1)
        good = [t for t in ts if sv[2 * t * Ps - 1] and sv[2 * t * Ps + 1]]
        tot += 1; has += bool(good)
        if not good: nopass.append(q)
        if good:
            first_t[q] = good[0]
        if q in (11, 13, 31, 101, 499, 997, 1999) or (not good):
            # the descent for the first twin: from 2 q# down; at each partial primorial P_i = q#/(top gears) choose k_i
            desc = []
            if good:
                t = good[0]; want = t * Ps    # we need q# - sum k_i P_i = t P_s
                rem = prims[-1][1]              # q#
                for i in range(len(prims) - 2, s - 1, -1):
                    Pi = prims[i][1]
                    # remove as much as keeps rem >= want and rem = want mod Pi... rem is a multiple of Pi already
                    k = (rem - want) // Pi if i > s else (rem - want) // Pi
                    rem -= k * Pi; desc.append((prims[i][0], k))
                assert rem == want, (q, rem, want)
            fails = []
            for t in list(ts)[:12]:
                if t in good: continue
                L = 2 * t * Ps - 1
                g = min([p for m in (L, L + 2) for p in factorint(m) if p >= 5 and (len(factorint(m)) > 1 or factorint(m)[p] > 1)] or [0])
                fails.append((t, g))
            out.append(f"q = {q}: P_s = {Ps} (base {base}), t in [{t_lo}, {t_hi}] ({len(ts)} landings); twins at t = {good[:12]}{'...' if len(good) > 12 else ''}; "
                       f"first twin {2 * good[0] * Ps - 1 if good else None}; failing t and the striking gear: {fails}; descent to the first twin (partial primorial without gears down to p, periods k): {desc}")
    out.append("")
    out.append(f"machines {tot}; a twin among the landings 2 t P_s - 1 in the window at {has}; none at {nopass}")
    out.append(f"first t by machine (sample): {[(q, first_t[q]) for q in (13, 31, 101, 499, 997, 1999) if q in first_t]}")
    Path("research/stack/r8/results_primorial_descent.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

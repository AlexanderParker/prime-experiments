"""The teeth on the t-line of the primorial descent.  Landing 2 t P_s - 1.  A gear g above the
base strikes the left member at t = (2 P_s)^{-1} (mod g) and the right member at t = -(2 P_s)^{-1}
(mod g): two teeth per gear, fixed by g and P_s.  For P_s = 210 and 2310: every gear g above the
base up to a bound, its two teeth, and on the t-line 1..T which t each gear takes (first taker
in gear order), the t left standing (twins as long as every gear up to sqrt(2 t P_s + 1) is in
the list).

usage: uv run python research/stack/r8/descent_classes.py 2310 100
"""
import sys
import numpy as np
from sympy import primerange, factorint
from pathlib import Path

def main():
    Ps, T = int(sys.argv[1]), int(sys.argv[2])
    base = []
    for p in primerange(2, 100):
        if Ps % p == 0: base.append(p)
    top = 2 * T * Ps + 1; N = top + 10
    sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    gears = [g for g in primerange(base[-1] + 1, int(top ** 0.5) + 1)]
    out = [__doc__.strip(), "", f"P_s = {Ps}, base {base}, t = 1..{T}, landings up to {top}; gears above the base up to sqrt: {len(gears)} (to {gears[-1]})", ""]
    teeth = {}
    for g in gears:
        inv = pow(2 * Ps, -1, g); teeth[g] = (inv % g, (-inv) % g)
    out.append("teeth (left, right) mod g: " + ", ".join(f"{g}: {teeth[g]}" for g in gears[:24]) + " ...")
    live = list(range(1, T + 1)); taken = {}
    for g in gears:
        a, b = teeth[g]
        tk = [t for t in live if t % g in (a, b)]
        for t in tk: taken[t] = g
        live = [t for t in live if t % g not in (a, b)]
        if tk: out.append(f"   gear {g:>3} teeth {a:>3},{b:>3}: takes {tk}")
    out.append(f"standing t: {live}")
    for t in live: assert sv[2 * t * Ps - 1] and sv[2 * t * Ps + 1], t
    out.append(f"landings of the standing t: {[(2 * t * Ps - 1, 2 * t * Ps + 1) for t in live[:10]]}")
    # the classes as positions: for the first few gears, the tooth positions on the t line
    out.append("")
    out.append("tooth positions on the t-line for the first gears above the base:")
    for g in gears[:8]:
        a, b = teeth[g]
        out.append(f"   gear {g}: left tooth at t = {[t for t in range(1, T + 1) if t % g == a]}; right tooth at t = {[t for t in range(1, T + 1) if t % g == b]}")
    Path(f"research/stack/r8/results_descent_classes_{Ps}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

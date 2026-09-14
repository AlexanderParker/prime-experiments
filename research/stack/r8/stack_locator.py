"""The stack locator (owner, 2026-09-15: can we build a locator).

Base = the largest primorial P (2, 2*3, 2*3*5, ...) with 2 P g_min - 1 <= q^2, g_min the next prime
after the base; so span 0 = (q, 2 P g_min - 1] is as long as the window allows and every layer
is active on it.  Locator: walk the slots (6j - 1, 6j + 1) upward from q; stop at the first slot
no layer marks (no gear of base + any layer divides a member).  A hole in span 0 is a twin by
construction.  Reported per machine: base, span 0, the located slot, its offset above q in
slots, and a check; failures where span 0 has no hole.

usage: uv run python research/stack/r8/stack_locator.py 5000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    fails = []; offs = []; rows = []
    for q in qs:
        ps = list(primerange(2, q + 1))
        base = []; P = 1
        for i, p in enumerate(ps[:-1]):
            if 2 * P * p * ps[i + 1] - 1 <= q * q: P *= p; base.append(p)
            else: break
        gmin = ps[len(base)]
        end = 2 * P * gmin - 1
        gears = [g for g in ps if g >= 5]           # base gears from 5 and every layer gear
        j = q // 6 + 1; found = None; steps = 0
        while 6 * j - 1 <= end:
            n = 6 * j - 1; steps += 1
            if all(n % g and (n + 2) % g for g in gears if g <= end):
                found = n; break
            j += 1
        if found is None: fails.append((q, base, end))
        else:
            assert sv[found] and sv[found + 2], (q, found)
            offs.append(steps)
        if q in (11, 13, 31, 59, 101, 419, 499, 997, 1999, 4603, 4999):
            rows.append(f"   q = {q}: base {base} (P = {P}), span 0 = ({q}, {end}]; located {(found, found + 2) if found else None} after {steps} slots")
    out.append(f"machines 11..{Q}: {len(qs)}; located a twin at {len(qs) - len(fails)}; failures {fails}")
    out.append(f"slots walked to the located twin: max {max(offs)}, at the median {sorted(offs)[len(offs) // 2]}")
    out += rows
    Path("research/stack/r8/results_stack_locator.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

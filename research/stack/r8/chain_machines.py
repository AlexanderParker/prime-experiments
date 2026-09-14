"""Loop, iteration 6: hand the residues forward from machine to machine.

Origin for machine q = the landing of the previous machine p (the prime below q), which is open
to every gear up to p when it is a twin.  One flip up about a mirror holding q: landing =
origin + 2 k M, M in {6q (mirror {2,3,q}), 30q, P_q q (the spiral base with q), q# / p... }.
The flip keeps the phase of every gear dividing M; every other gear shifts by 2kM mod gear.
Blind rule: k = the smallest periods putting the landing above q.  Chain from machine 5 with
origin (11, 13), or (17, 19).  Reported: how far the chain stays on twins (the first machine
whose landing is not a twin), and, letting the chain continue from non-twin landings too, the
twin rate over the machines to 5000.  usage: uv run python research/stack/r8/chain_machines.py 5000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(7, Q + 1))
    N = 2 * Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    def mirrors(q):
        ps = list(primerange(2, q + 1)); base = []; P = 1
        for p in ps:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        return {'{2,3,q}': 6 * q, '{2,3,5,q}': 30 * q, '{base,q}': P * q}
    out = [__doc__.strip(), ""]
    for seed in (11, 17):
        for mname in ('{2,3,q}', '{2,3,5,q}', '{base,q}'):
            origin = seed; broken = None; twins = 0; inwin = 0
            for q in qs:
                M = mirrors(q)[mname]
                k = 1
                while origin + 2 * k * M <= q: k += 1
                E = origin + 2 * k * M
                ok = E + 2 <= q * q
                inwin += ok
                tw = ok and sv[E] and sv[E + 2]
                twins += tw
                if not tw and broken is None: broken = (q, E)
                origin = E
            out.append(f"seed ({seed}, {seed + 2}), mirror {mname}: chain stays on twins up to the machine before {broken}; continuing from every landing, twin at {twins} of {inwin} in the window over {len(qs)} machines")
    Path("research/stack/r8/results_chain_machines.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

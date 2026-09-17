"""Loop, entry 64: the one-flip locator's margin as q grows.  From home (-1), the columns
-1 + 12 g k d for g among the first A gears above sqrt q, k = 1..K, d = +-1: how many are open
to every gear of the machine (hence twins inside the window).  A and K scaled several ways.
usage: uv run python research/stack/r8/oneflip_margin.py
"""
import math
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    out = [__doc__.strip(), ""]
    for q in (1000, 2000, 5000, 10000, 20000):
        N = q * q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
        for i in range(2, int(N ** 0.5) + 1):
            if sv[i]: sv[i * i::i] = False
        gears = [p for p in primerange(5, q + 1)]
        hi = [p for p in gears if p * p > q]
        lgq = math.log(q)
        for name, A, K in (("A = 8, K = (ln q)^2", 8, int(lgq ** 2)),
                           ("A = 8, K = 40", 8, 40),
                           ("A = ln q, K = (ln q)^2", max(2, int(lgq)), int(lgq ** 2)),
                           ("A = 1, K = (ln q)^3", 1, int(lgq ** 3))):
            cnt = 0; first = None
            for g in hi[:A]:
                for k in range(1, K + 1):
                    for d in (1, -1):
                        x = -1 + 12 * g * k * d
                        if q < x <= q * q - 2 and sv[x] and sv[x + 2]:
                            cnt += 1
                            if first is None: first = (g, k, d)
            out.append(f"q = {q:>6} {name:<24}: candidates {2*A*K:>5}, open {cnt:>4}, first {first}")
        del sv
    print("\n".join(out[2:]))
    Path("research/stack/r8/results_oneflip_margin.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

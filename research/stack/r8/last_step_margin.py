"""Loop, entry 40: the last step's margin against q.  The walk's last flip is about {2, 3, 5}
from a column L open to 2, 3, 5 (L = 11, 17 or 29 mod 30); the candidates are L +- 60 k,
k = 1..K, inside the window (q, q^2 - 2]; the keeping moves are the candidates that are twins.
Proxy for the walk's own L: 200 columns per q drawn uniformly from the window with the right
residue mod 30.  Reported per q: the mean and the minimum number of twins among the 2K
candidates for K = 40, and the K needed so that every sampled column has at least one twin
among its first K periods (the largest first-hit k over the samples).  Twins by isprime.
usage: uv run python research/stack/r8/last_step_margin.py
"""
import random
from sympy import isprime
from pathlib import Path

def main():
    random.seed(3); out = [__doc__.strip(), ""]
    for q in (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 300000, 1000000):
        lo, hi = q + 1, q * q - 2; margins = []; firsts = []
        for _ in range(200):
            while True:
                L = random.randrange(lo, hi)
                if L % 30 in (11, 17, 29): break
            cnt = 0; first = None
            for k in range(1, 41):
                for d in (1, -1):
                    n = L + 60 * k * d
                    if lo <= n <= hi and isprime(n) and isprime(n + 2):
                        cnt += 1
                        if first is None: first = k
            margins.append(cnt); firsts.append(first)
            if first is None:
                k = 41
                while True:
                    if any(lo <= L + 60 * k * d <= hi and isprime(L + 60 * k * d) and isprime(L + 60 * k * d + 2) for d in (1, -1)): break
                    k += 1
                firsts[-1] = k
        line = f"q = {q:>8}: twins among the 80 candidates (K = 40): mean {sum(margins) / 200:.2f}, min {min(margins)}, samples with none {sum(1 for c in margins if c == 0)}; K needed for every sample to have a twin within K periods: {max(firsts)}"
        out.append(line); print(line, flush=True)
    Path("research/stack/r8/results_last_step_margin.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

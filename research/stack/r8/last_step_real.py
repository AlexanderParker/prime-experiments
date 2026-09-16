"""Loop, entry 41: the walk's OWN last step, not a proxy.  Run the settle walk (mirror {2,3,g},
K = 40, full memory, descending) to the step before gear 5, record the column L reached, then
report for the last flip about {2,3,5}: where L sits in the window, the first k with L +- 60k a
twin, and the number of twins among the first 80 candidates.
usage: uv run python research/stack/r8/last_step_real.py lo hi
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); K = 40
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    qs = list(primerange(lo, hi + 1))
    out = [__doc__.strip(), ""]; firsts = []; cnts = []; fracs = []
    for q in qs:
        m = E.Machine(q, sv); P = 6; seq = [g for g in m.ps if g >= 5][::-1]
        visited = []; L = -1
        for i, g in enumerate(seq[:-1]):        # every gear but the last (5)
            visited.append(g); nxt = seq[i + 1]
            best = None
            for k in range(1, K + 1):
                for d in (1, -1):
                    n = L + 2 * k * P * g * d
                    if n < 0 or n > q * q - 2: continue
                    on = sum(1 for h in visited if n % h in (0, h - 2))
                    md = min(dist(n % h, h) for h in visited)
                    fl = 0
                    if on == 0:
                        fl = sum(1 for k2 in range(1, 21) for d2 in (1, -1) if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                    sc = (-on, fl, md)
                    if best is None or sc > best[0]: best = (sc, n)
            L = best[1]
        first = None; cnt = 0
        k = 1
        while k <= 4000:
            for d in (1, -1):
                n = L + 60 * k * d
                if q < n <= q * q - 2 and sv[n] and sv[n + 2]:
                    if first is None: first = k
                    if k <= K: cnt += 1
            k += 1
            if first is not None and k > K: break
        firsts.append(first if first else 9999); cnts.append(cnt); fracs.append(L / (q * q))
        if q in (997, 1009, 1997, 1999, 2999, 3989):
            out.append(f"   q = {q}: column before the last flip L = {L} ({L / (q * q) * 100:.1f}% up the window); first twin at k = {first}; twins among the first 80 candidates {cnt}")
    out.insert(2, f"machines {qs[0]}..{qs[-1]} ({len(qs)}): first k with a twin: max {max(firsts)}, mean {sum(firsts) / len(firsts):.1f}; twins among the first 80 candidates: min {min(cnts)}, mean {sum(cnts) / len(cnts):.2f}; the column before the last flip sits at {sum(fracs) / len(fracs) * 100:.1f}% of the window on average")
    Path(f"research/stack/r8/results_last_step_real_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:10]))

if __name__ == "__main__":
    main()

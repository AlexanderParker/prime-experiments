"""Loop, iteration 14: the scoring pick-up walk, improved (phases only, no primality).
Variants on entry 13 (K = 15):
  order: 'desc' (q down to the smallest), 'desc-mid' (large gears descending, then the gears at
         most sqrt q ascending), 'small-first-then-desc' (gears at most sqrt q ascending first,
         then the rest descending)
  score: 'dist' (entry 13) or 'flex' (prefer the move after which the NEXT gear's step has the
         most invariant-keeping options; ties by dist)
  repair: after the walk, up to R extra flips about {base, s} for s among the gears at most
          sqrt q, each chosen by the same rule over all gears (R = 0, 2, 5)
Reported: machines 11 to 2000: twin (= landing open to every gear) count.
usage: uv run python research/stack/r8/pickup_walk_v2.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def options(n, P, g, visited, K, q):
    outs = []
    for k in range(1, K + 1):
        for d in (1, -1):
            m = n + 2 * k * P * g * d
            if m < 0 or m > q * q - 2: continue
            onteeth = sum(1 for h in visited if m % h in (0, h - 2))
            outs.append((m, onteeth, min(dist(m % h, h) for h in visited), k))
    return outs

def choose(n, P, g, visited, K, q, score, nxt):
    opts = options(n, P, g, visited, K, q)
    if not opts: return n
    if score == 'flex' and nxt is not None:
        def flex(m):
            vis2 = visited + [nxt]
            return sum(1 for k in range(1, K + 1) for d in (1, -1)
                       if 0 <= m + 2 * k * P * nxt * d <= q * q - 2 and all((m + 2 * k * P * nxt * d) % h not in (0, h - 2) for h in vis2))
        return max(opts, key=lambda o: (-o[1], flex(o[0]) if o[1] == 0 else -1, o[2], -o[3]))[0]
    return max(opts, key=lambda o: (-o[1], o[2], -o[3]))[0]

def run(q, order, score, R, K=15):
    ps = list(primerange(2, q + 1)); base = []; P = 1
    for p in ps:
        if P * p <= max(q // 2, 6): P *= p; base.append(p)
        else: break
    rest = [p for p in ps if p not in base]; r = int(q ** 0.5)
    small = [g for g in rest if g <= r]; large = [g for g in rest if g > r]
    if order == 'desc': seq = rest[::-1]
    elif order == 'desc-mid': seq = large[::-1] + small
    else: seq = small + large[::-1]
    n = -1; visited = [x for x in base if x >= 5]
    for i, g in enumerate(seq):
        visited.append(g)
        n = choose(n, P, g, visited, K, q, score, seq[i + 1] if i + 1 < len(seq) else None)
    allg = [x for x in ps if x >= 5]
    for _ in range(R):
        if all(n % h not in (0, h - 2) for h in allg): break
        best = None
        for s in (small or rest[:3]):
            for k in range(1, K + 1):
                for d in (1, -1):
                    m = n + 2 * k * P * s * d
                    if not (q < m <= q * q - 2): continue
                    onteeth = sum(1 for h in allg if m % h in (0, h - 2))
                    sc = (-onteeth, min(dist(m % h, h) for h in allg), -k)
                    if best is None or sc > best[0]: best = (sc, m)
        if best: n = best[1]
    return n, all(n % h not in (0, h - 2) for h in allg) and q < n <= q * q - 2

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    for order in ('desc', 'desc-mid', 'small-first-then-desc'):
        for score in ('dist', 'flex'):
            for R in (0, 2, 5):
                ok = 0; fails = []
                for q in qs:
                    n, good = run(q, order, score, R)
                    if good: assert sv[n] and sv[n + 2]; ok += 1
                    else: fails.append(q)
                out.append(f"order {order:<22} score {score:<4} repair {R}: twin at {ok} of {len(qs)}; fails {fails[:10]}{'...' if len(fails) > 10 else ''}")
                print(out[-1])
    Path("research/stack/r8/results_pickup_walk_v2.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

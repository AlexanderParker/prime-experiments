"""Loop, entry 53: the margin at the tail's first step, against q.  Running the whole prefix at
large q is expensive, so the column handed over is drawn as measured: a column of the window,
open to every gear above the cut (which the prefix guarantees) and to 2, 3 and 5, sitting in the
lower part of the window.  At the tail's first step the stride is 12 g with g the first gear
below the cut (about q / 4.5); the candidates are c + 12 g k d, k = 1..K.  Measured per q over
40 draws: candidates open to every gear of the machine, with K fixed and K scaled.
usage: uv run python research/stack/r8/first_tail_margin.py K_mode
"""
import sys, math, random
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "scaled"
    random.seed(17)
    out = [__doc__.strip(), ""]
    for q in (1000, 2000, 5000, 10000, 20000):
        N = q * q + 10
        sv = np.ones(N + 1, dtype=bool); sv[:2] = False
        for i in range(2, int(N ** 0.5) + 1):
            if sv[i]: sv[i * i::i] = False
        gears = [p for p in primerange(5, q + 1)]
        ps = gears[::-1]; cut = 0
        while cut < len(ps) and ps[cut] > 2 * (cut + 1): cut += 1
        g = ps[cut] if cut < len(ps) else 7                       # the first gear below the cut
        above = set(ps[:cut])                                      # gears the prefix has settled
        K = 40 if mode == "40" else max(20, int(math.log(q) ** 2))
        lo, hi = q + 1, q * q - 2
        cnts = []
        for _ in range(40):
            # a column of the lower window open to the settled gears and to 5
            for _try in range(20000):
                c = random.randrange(lo, lo + (hi - lo) // 3)
                if c % 6 != 5: continue
                if c % 5 in (0, 3): continue
                if all(c % h not in (0, h - 2) for h in above): break
            cnt = 0
            for k in range(1, K + 1):
                for d in (1, -1):
                    x = c + 12 * g * k * d
                    if lo <= x <= hi and sv[x] and sv[x + 2]: cnt += 1
            cnts.append(cnt)
        line = f"q = {q:>6}: cut at gear {g} (step {cut} of {len(ps)}), K = {K:>3}, candidates {2*K:>4}: open min {min(cnts)}, mean {sum(cnts)/len(cnts):.2f}, draws with none {sum(1 for c in cnts if c == 0)} of 40"
        out.append(line); print(line, flush=True)
        del sv
    Path(f"research/stack/r8/results_first_tail_margin_{mode}.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

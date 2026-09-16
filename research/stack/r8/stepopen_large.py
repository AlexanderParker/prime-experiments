"""Loop, entry 49: StepOpen's margin at larger machines.  The walk's own last step is expensive
to reach at large q, so the column L is replaced by its measured statistics: L is a column of
the window open to every gear of the machine except the ones the last step re-settles, and it
sits low in the window (29 to 38 percent measured).  Here the margin is measured directly on
the family: from a column L drawn in the window with L = 11, 17 or 29 mod 30 (open to 2, 3, 5)
and low in the window, the candidates L + 12 g k d for the first A gears above sqrt q,
k = 1..K, both directions: how many are open to every gear up to q (i.e. are twins, by the
square-root rule, since the candidates are below q^2).  Reported per q: the minimum and mean
over 60 draws, and the share of draws with none.
usage: uv run python research/stack/r8/stepopen_large.py A K
"""
import sys, random
from sympy import primerange, isprime
from pathlib import Path

def main():
    import math
    A = int(sys.argv[1]); Karg = sys.argv[2]; random.seed(11)
    out = [__doc__.strip(), "", f"family: {A} strides x K = {Karg} periods x 2 directions"]
    for q in (1000, 3000, 10000, 30000, 100000, 300000):
        K = int(Karg) if Karg.isdigit() else max(4, int(math.log(q) ** 2 / 4))
        gs = [p for p in primerange(int(q ** 0.5) + 1, q + 1)][:A]
        lo, hi = q + 1, q * q - 2
        cnts = []
        for _ in range(60):
            while True:
                L = random.randrange(lo, lo + (hi - lo) // 3)      # low in the window, as measured
                if L % 30 in (11, 17, 29): break
            c = 0
            for g in gs:
                for k in range(1, K + 1):
                    for d in (1, -1):
                        x = L + 12 * g * k * d
                        if lo <= x <= hi and isprime(x) and isprime(x + 2): c += 1
            cnts.append(c)
        line = f"q = {q:>7}: K = {K:>3}, candidates {2*A*K:>4}: open min {min(cnts)}, mean {sum(cnts)/len(cnts):.2f}, draws with none {sum(1 for c in cnts if c == 0)} of 60"
        out.append(line); print(line, flush=True)
    Path(f"research/stack/r8/results_stepopen_large_{A}_{Karg}.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

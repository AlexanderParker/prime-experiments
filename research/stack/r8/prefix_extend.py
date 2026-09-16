"""Loop, entry 44: enlarge the proved prefix.  The free-regime lemma needs the visited gears all
above 2n (n = their number), which covers the first steps.  Two ways to push it further,
measured on the walk itself:
 (a) spacing: a gear h strikes at most ceil(2K / h) + 2 of the 2K candidates, so the struck
     candidates number at most sum over visited h of (2K/h + 2); the lemma survives while that
     sum is below 2K, i.e. while sum over visited h of (1/h + 1/K) < 1.  Report the step at
     which the sum passes 1 (the "spacing regime" end) against the free regime's end.
 (b) memory trimming: at step i, does the candidate chosen need the whole memory, or only the
     gears above some bound?  Measure the largest gear that is ever the sole striker of every
     otherwise-keeping candidate (if the small gears never bind until late, the memory can be
     trimmed and the free regime applies to a longer prefix).
usage: uv run python research/stack/r8/prefix_extend.py lo hi
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); K = 40
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    for q in list(primerange(lo, hi + 1))[::7]:
        ps = [p for p in primerange(5, q + 1)][::-1]
        free = 0
        while free < len(ps) and ps[free] > 2 * (free + 1): free += 1
        s = 0.0; spac = 0
        for i, g in enumerate(ps):
            s += 1.0 / g + 1.0 / K
            if s >= 1.0: break
            spac = i + 1
        out.append(f"q = {q}: gears {len(ps)}; free regime ends at step {free} (gear {ps[free-1] if free else '-'}); spacing regime ends at step {spac} (gear {ps[spac-1] if spac else '-'}), {spac * 100 // len(ps)}% of the walk")
    Path(f"research/stack/r8/results_prefix_extend_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:12]))

if __name__ == "__main__":
    main()

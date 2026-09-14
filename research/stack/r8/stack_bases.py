"""The one-mirror stack across machine sizes and base sets (owner, 2026-09-15).

Base sets: null {} (P = 1), {2}, {2,3}, {2,3,5}, {2,3,5,7}, and 'spiral' (the lowest gears with
product at most q/2).  Layers = the gears from 5 up not in the base; layer g = the sieve of
base + {g} on the slot line (6j - 1, 6j + 1), from the origin to its landing 2 P g - 1 (capped at
the window top).  Stack = layers over each other; holes = slots no layer marks; a hole is true
(a twin) or false.  Spans = between consecutive landings; span 0 runs from the window's start to
the first landing, where every layer is active.

Per base, over the machines: (a) machines whose span 0 holds at least one twin (a twin found
before any layer ends); (b) per span index k: slots, twins, holes, true holes summed over the
machines; (c) the twin share inside spans that end at the second gear of a twin-gear pair
(13, 19, 31, 43, 61, 73, ...) against the other spans.

usage: uv run python research/stack/r8/stack_bases.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
from collections import defaultdict

BASES = {'null': [], '{2}': [2], '{2,3}': [2, 3], '{2,3,5}': [2, 3, 5], '{2,3,5,7}': [2, 3, 5, 7], 'spiral': None}

def main():
    Q = int(sys.argv[1]); qs = [p for p in primerange(31, Q + 1)]
    qs = qs[::max(1, len(qs) // 60)] + [499, 997, 1999]
    qs = sorted(set(q for q in qs if q <= Q))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), "", f"machines: {len(qs)} primes from {qs[0]} to {qs[-1]}", ""]
    for bname, bset in BASES.items():
        agg = defaultdict(lambda: [0, 0, 0, 0]); span0_twin = 0; twinpair = [0, 0]; other = [0, 0]; nm = 0
        for q in qs:
            ps = list(primerange(2, q + 1))
            if bset is None:
                base = []; P = 1
                for p in ps:
                    if P * p <= q // 2: P *= p; base.append(p)
                    else: break
            else:
                base = [p for p in bset if p <= q]; P = 1
                for p in base: P *= p
            gears = [g for g in ps if g >= 5 and g not in base]
            if not gears: continue
            nm += 1
            ends = [min(2 * P * g - 1, q * q) for g in gears]
            slots = [j for j in range(q // 6 + 1, (q * q - 1) // 6 + 1)]
            arr_n = np.array([6 * j - 1 for j in slots])
            stack = np.zeros(len(slots), dtype=bool)
            marks = {}
            bg = [x for x in base if x >= 5]
            for g, end in zip(gears, ends):
                active = arr_n <= end
                hit = np.zeros(len(slots), dtype=bool)
                for h in bg + [g]:
                    hit |= (arr_n % h == 0) | ((arr_n + 2) % h == 0)
                stack |= hit & active
            twin = sv[arr_n] & sv[arr_n + 2]
            bounds = [6 * slots[0] - 1 - 1] + ends
            first = True
            for k in range(len(bounds) - 1):
                lo, hi = bounds[k], bounds[k + 1]
                if hi <= lo: continue
                sel = (arr_n > lo) & (arr_n <= hi)
                s = int(sel.sum()); t = int((twin & sel).sum()); h = int((~stack & sel).sum()); th = int((~stack & twin & sel).sum())
                a = agg[min(k, 12)]; a[0] += s; a[1] += t; a[2] += h; a[3] += th
                if k == 0 and t > 0: span0_twin += 1
                g = gears[k] if k < len(gears) else None
                if g is not None and g - 2 in gears:  # span ending at the second of a twin-gear pair
                    twinpair[0] += s; twinpair[1] += t
                else:
                    other[0] += s; other[1] += t
        out.append(f"base {bname}: machines {nm}; span 0 (every layer active) holds a twin at {span0_twin} of {nm}")
        out.append(f"   span k: slots / twins / holes / true holes   (k = 12 collects 12 and beyond)")
        for k in sorted(agg):
            a = agg[k]; out.append(f"      {k:>2}: {a[0]:>7} / {a[1]:>6} / {a[2]:>7} / {a[3]:>6}")
        out.append(f"   twin share: spans ending at the second gear of a twin-gear pair {twinpair[1]} of {twinpair[0]} slots; other spans {other[1]} of {other[0]} slots")
        out.append("")
    Path("research/stack/r8/results_stack_bases.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

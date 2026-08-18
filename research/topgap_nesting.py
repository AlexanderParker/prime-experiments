"""Round 9 lateral, part 2: NESTING of maximal gaps across consecutive machines.

For each step M_y -> M_y' (add gear q = next prime): take every maximal gap of
M_y', and look at the SAME address in M_y (local recomputation, no full
period): which old gap does the new maximum grow from (size, rank vs F_old)?
How many old openings inside the new gap window did q kill (the chain length
k), and at which teeth? Plus exact addresses of new maxima mod 35, 385, 5005.

This answers "is the top of the gap spectrum corridor-forced the way the
L*=13 landmark's address was": the landmark lived at the corridor mouth; do
maximal gaps live at fixed addresses in the previous machine's corridor
structure?

Run: uv run python research/topgap_nesting.py    (repo root; numpy)
"""
from math import prod

import numpy as np

from split_gap_law import primes
from topgap_corridor import chunk_openings

def max_gaps(y, chunk=20_000_000):
    """All (leftpos, gap) with gap == F for machine y, plus F."""
    gears = primes(5, y)
    P = prod(gears)
    F = 0
    tops = []
    carry = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        ops = chunk_openings(gears, a, S)
        ext = ops if carry is None else np.concatenate((carry, ops))
        d = np.diff(ext)
        if len(d):
            m = int(d.max())
            if m > F:
                F = m
                tops = []
            for i in np.flatnonzero(d == F):
                tops.append(int(ext[i]))
        carry = ext[-2:]
        a += S
    return F, sorted(set(tops))

def local_openings(y, lo, hi):
    gears = primes(5, y)
    S = hi - lo
    killed = np.zeros(S, bool)
    for q in gears:
        u = pow(6, -1, q)
        for t in (u, q - u):
            killed[(t - lo) % q::q] = True
    return np.flatnonzero(~killed).astype(np.int64) + lo

def main():
    ys = [13, 17, 19, 23, 29]
    data = {y: max_gaps(y) for y in ys}
    for y, (F, tops) in data.items():
        print(f"y={y}: F={F}, maximal gaps at {len(tops)} positions; "
              f"addresses mod 35 {sorted(set(t % 35 for t in tops))}, "
              f"mod 385 {sorted(set(t % 385 for t in tops))}, "
              f"mod 5005 {sorted(set(t % 5005 for t in tops))[:8]}")
    print("=" * 72)
    print("NESTING: each new maximal gap, seen inside the previous machine")
    for yo, yn in zip(ys, ys[1:]):
        q = primes(yo + 1, yn)[-1]
        u = pow(6, -1, q)
        Fo, _ = data[yo]
        Fn, tops = data[yn]
        print(f"--- step {yo} -> {yn} (gear {q}): F {Fo} -> {Fn} ---")
        seen = set()
        for t in tops:
            lo, hi = t - 3 * Fo, t + Fn + 3 * Fo
            old = local_openings(yo, lo, hi)
            # old gap containing the new gap's interior start t+1
            j = np.searchsorted(old, t, side='right')
            oldgap_at = (int(old[j] - old[j - 1]), int(old[j - 1]))
            inside = old[(old > t) & (old < t + Fn)]
            kills = [(int(o), 'L' if o % q == u else
                      'R' if o % q == (q - u) % q else '??') for o in inside]
            word = tuple(np.diff(np.concatenate(([t], inside, [t + Fn]))).tolist())
            key = (oldgap_at[0], len(inside), word,
                   tuple(s for _, s in kills))
            if key in seen:
                continue
            seen.add(key)
            print(f"  new max at {t} (mod 35 {t%35}): sits in old gap of size "
                  f"{oldgap_at[0]} (= {oldgap_at[0]/Fo:.2f} F_old); chain k = "
                  f"{len(inside)} old openings killed by {q}: sides "
                  f"{[s for _, s in kills]}, merged gap word {word}")
            if '??' in [s for _, s in kills]:
                print("    WARNING: an interior opening not killed by the new gear!")
    print("=" * 72)
    print("F2 nesting check (which stratum does F2 use): see topgap_corridor")

if __name__ == "__main__":
    main()

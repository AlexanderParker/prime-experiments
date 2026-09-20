"""E2: the longest run of columns painted jointly by a set of gears, exact by exhaustion.

Gear g paints the columns n with 6n = +-1 mod g, i.e. n = +-inv6(g) mod g; the two teeth sit at
distance d_g = 2 inv6 = inv3 mod g (or g - inv3). By CRT every phase vector is a window of the real
pattern (RigidShift.lean), so the record over all phases is the record over all window positions;
here we take phases directly: for each phase vector (shift per gear) find the longest run.

Pre-registered (tree node R5.f.xxxiv.a, E2): F({g, h}) = 4 iff {g, h} = {5, 7}; 3 iff exactly one
of g, h is in {5, 7}; 2 otherwise - because a run of 4 needs both tooth distances equal to 2 and a
run of 3 needs one of them equal to 2, and inv3 = +-2 mod g iff g in {5, 7}.
Then triples and the initial segments {5..p} for p <= 23, to see how the record grows (E3).
"""
import sys, itertools
from math import prod

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})

def longest_run(gears):
    """Exact: iterate over all phase vectors (product of gears), scan one period of the joint
    pattern for the longest run of painted columns (cyclic)."""
    P = prod(gears)
    best = 0; arg = None
    T = {g: teeth(g) for g in gears}
    for shifts in itertools.product(*[range(g) for g in gears]):
        painted = bytearray(P)
        for g, s in zip(gears, shifts):
            for t in T[g]:
                start = (t + s) % g
                painted[start::g] = b"\x01" * len(range(start, P, g))
        # longest cyclic run of 1s
        run = 0; mx = 0
        for x in painted + painted[:min(P, 4096)]:
            if x: run += 1; mx = max(mx, run)
            else: run = 0
        mx = min(mx, P)
        if mx > best: best, arg = mx, shifts
    return best, arg

def longest_run_shiftfree(gears):
    """Same record, using RigidShift: the record over phases equals the record of the real pattern
    over one full period P (all shift vectors occur). Faster: one scan of the real pattern."""
    P = prod(gears)
    painted = bytearray(P)
    for g in gears:
        for t in teeth(g):
            painted[t::g] = b"\x01" * len(range(t, P, g))
    run = 0; mx = 0
    for x in painted + painted:
        if x: run += 1; mx = max(mx, run)
        else: run = 0
    return min(mx, P)

primes = [p for p in range(5, 200) if all(p % d for d in range(2, int(p**0.5) + 1))]
if __name__ == "__main__":
    # E2: pairs
    bad = []
    for g, h in itertools.combinations(primes[:25], 2):
        F = longest_run_shiftfree([g, h])
        pred = 4 if {g, h} == {5, 7} else (3 if (g in (5, 7)) != (h in (5, 7)) else 2)
        if F != pred: bad.append((g, h, F, pred))
    print(f"pairs among the first 25 gears: {len(list(itertools.combinations(primes[:25], 2)))}, "
          f"prediction violations: {len(bad)} {bad[:5]}")
    print("tooth distance inv3 mod g for g <= 43:", [(g, inv(3, g), (-inv(3, g)) % g) for g in primes[:12]])
    # cross-check the two methods on small pairs
    for g, h in [(5, 7), (5, 11), (7, 13), (11, 13)]:
        print(f"  {g},{h}: exhaustive {longest_run([g, h])[0]}, period scan {longest_run_shiftfree([g, h])}")
    # triples
    print("triples of the first 8 gears (g, h, k, F):")
    for tr in itertools.combinations(primes[:8], 3):
        print("  ", tr, longest_run_shiftfree(list(tr)))
    # initial segments
    print("initial segments {5..p}:")
    acc = []
    for p in primes:
        acc.append(p)
        if prod(acc) > 3 * 10**8: break
        print(f"  p = {p}: F = {longest_run_shiftfree(acc)}  (period {prod(acc)})")

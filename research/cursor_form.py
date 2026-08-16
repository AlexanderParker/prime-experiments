"""The cursor (odometer) form of the algorithm, ported from rust2/src/main.rs.

get_next_prime_gap in rust2 keeps one running bucket per trial divisor, holding
that divisor's next blocked gap, and advances a bucket only when the gap under
test passes it. There is no window and no cycling bound: each divisor's blocked
slots are produced lazily, forever, in step with the gap being tested. This is
the form the rest of the twin work should build on, so it is ported here exactly
and checked against direct division.

Two variants are provided:

  next_prime_gap_faithful - divisor set fixed at primes <= ceil(sqrt(p)), as in
      the Rust code.
  next_prime_gap_extended - divisor set grown to primes <= ceil(sqrt(p + gap))
      as the tested gap grows. This is the same rule with the divisor bound
      tracking the candidate instead of the starting point.

The two agree unless the returned gap reaches a square of a prime just above
sqrt(p); the test below looks for that case rather than assuming it away.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.twin_gap import trial_divisors
from verify_twin_gap import is_prime


def next_prime_gap_faithful(p):
    """Port of get_next_prime_gap: fixed divisor set, lazy per-divisor cursors."""
    bound = math.isqrt(p)
    if bound * bound != p:
        bound += 1
    divisors = trial_divisors(bound)
    # cursor[i] = next gap blocked by divisors[i]
    cursor = [(-p) % q for q in divisors]

    gap = 2
    if gap not in cursor:
        return gap
    while True:
        gap += 2
        blocked = False
        for i, q in enumerate(divisors):
            if q == 2:
                continue  # divisor 2 only ever blocks odd gaps for odd p
            while cursor[i] < gap:
                cursor[i] += q
            if cursor[i] == gap:
                blocked = True
                break
        if not blocked:
            return gap


def next_prime_gap_extended(p):
    """Same rule, with the divisor bound following the candidate p + gap."""
    divisors = []
    cursor = []

    def ensure(bound):
        ds = trial_divisors(bound)
        while len(divisors) < len(ds):
            q = ds[len(divisors)]
            divisors.append(q)
            cursor.append((-p) % q)

    gap = 0
    while True:
        gap += 2
        need = math.isqrt(p + gap)
        if need * need != p + gap:
            need += 1
        ensure(need)
        blocked = False
        for i, q in enumerate(divisors):
            if q == 2:
                continue
            while cursor[i] < gap:
                cursor[i] += q
            if cursor[i] == gap:
                blocked = True
                break
        if not blocked:
            return gap


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    disagreements = []
    wrong_faithful = []
    wrong_extended = []
    checked = 0

    p = 3
    while p < limit:
        if is_prime(p):
            true_gap = 2
            while not is_prime(p + true_gap):
                true_gap += 2
            gf = next_prime_gap_faithful(p)
            ge = next_prime_gap_extended(p)
            if gf != ge:
                disagreements.append((p, gf, ge))
            if gf != true_gap:
                wrong_faithful.append((p, gf, true_gap))
            if ge != true_gap:
                wrong_extended.append((p, ge, true_gap))
            checked += 1
        p += 2

    print(f"primes checked below {limit}: {checked}")
    print(f"faithful port wrong: {len(wrong_faithful)} {wrong_faithful[:5]}")
    print(f"extended version wrong: {len(wrong_extended)} {wrong_extended[:5]}")
    print(f"the two variants disagree: {len(disagreements)} {disagreements[:5]}")


if __name__ == "__main__":
    main()

"""Halved coordinates: the same algorithm with prime 2 removed and gaps halved.

Every prime gap after the 2 -> 3 step is even, so the algorithm's natural
coordinate is the half-gap h = g / 2. Substituting g = 2h into the blocking
condition q | p + g gives

    2h = -p (mod q)   <=>   h = -p * inverse(2) (mod q)

so in halved coordinates the divisor 2 disappears from the system entirely and
each odd prime q blocks exactly one residue

    d_q(p) = -p * (q + 1) / 2   (mod q)

The next prime gap is then 2 * (least h >= 1 avoiding every d_q).

The twin search halves the same way. From an even base n, blocking both
n + 2h and n + 2h + 2 gives

    h = d_q(n)        and        h = d_q(n) - 1

Two *adjacent* residues. So the twin problem is the original problem with each
blocked point widened into a blocked interval of length 2. That is the single
structural difference between "find the next prime" and "find the next twin
pair" in this framework, and it suggests the interval length L as the knob to
push on: L = 1 is the next-prime algorithm, L = 2 is twins.

This script checks both identities against the unhalved algorithm and against
direct division.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.twin_gap import next_twin_gap, trial_divisors
from verify_twin_gap import is_prime


def blocked_residue(n, q):
    """The single residue d_q that q blocks in halved coordinates."""
    return (-n * ((q + 1) // 2)) % q


def half_gap_to_next_prime(p):
    """Least h >= 1 with h != d_q (mod q) for every odd q <= sqrt(p + 2h)."""
    window = max(16, 4 * int(math.log(p)) ** 2)
    while True:
        bound = math.isqrt(p + 2 * window) + 1
        blocked = bytearray(window + 1)
        for q in trial_divisors(bound):
            if q == 2:
                continue
            r = blocked_residue(p, q)
            if r <= window:
                blocked[r :: q] = b"\x01" * ((window - r) // q + 1)
        for h in range(1, window + 1):
            if not blocked[h]:
                return h
        window *= 2


def half_gap_to_next_twin(n):
    """Least h >= 1 avoiding the adjacent pair {d_q, d_q - 1} for every odd q."""
    # Twin members are odd, so the base must be odd for the gap to be even and
    # the half-gap h to be an integer.
    assert n % 2 == 1, "halved twin search needs an odd base"
    window = max(16, 4 * int(math.log(n)) ** 2)
    while True:
        bound = math.isqrt(n + 2 * window + 2) + 1
        blocked = bytearray(window + 1)
        for q in trial_divisors(bound):
            if q == 2:
                continue
            d = blocked_residue(n, q)
            for r in (d % q, (d - 1) % q):
                if r <= window:
                    blocked[r :: q] = b"\x01" * ((window - r) // q + 1)
        for h in range(1, window + 1):
            if not blocked[h]:
                return h
        window *= 2


def main():
    # Identity 1: halved next-prime search reproduces the real prime gap.
    checked = 0
    p = 1009
    while p < 40000:
        if is_prime(p):
            h = half_gap_to_next_prime(p)
            q = p + 2
            while not is_prime(q):
                q += 2
            assert 2 * h == q - p, (p, 2 * h, q - p)
            checked += 1
        p += 2
    print(f"halved next-prime identity verified on {checked} primes")

    # Identity 2: halved twin search reproduces the unhalved twin gap.
    checked = 0
    for n in range(1001, 6000, 2):
        assert 2 * half_gap_to_next_twin(n) == next_twin_gap(n), n
        checked += 1
    for e in (6, 8, 10, 12):
        n = 10**e + 1
        assert 2 * half_gap_to_next_twin(n) == next_twin_gap(n), n
        checked += 1
    print(f"halved twin identity verified on {checked} bases")

    # The adjacency claim, spelled out for one case.
    n = 10**6
    rows = [(q, blocked_residue(n, q), (blocked_residue(n, q) - 1) % q)
            for q in trial_divisors(30) if q > 2]
    print(f"n={n} blocked residue pairs (adjacent by construction):")
    for q, d, d_minus in rows:
        print(f"  q={q:>3}: blocks h = {d} and h = {d_minus} (mod {q})")


if __name__ == "__main__":
    main()

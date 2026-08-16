"""n-wise combinations of gears, and each combination measured against the 6-cycle.

Pairwise is not enough: a gear set of `n` gears has `2^n - 1` non-empty sub-machines, and the
alignment structure lives in all of them. This module tabulates the whole lattice and compares
each level back to the 6-cycle.

Two different objects must be kept apart, and their ratio is the 6-cycle relationship.

    exposed NUMBERS   n in one period 6P with n = +/-1 mod 6 and divisible by no gear of S.
                      Count 2 prod (q - 1) - these are single members surviving.
    exposed SLOTS     m in one period P with both 6m - 1 and 6m + 1 surviving.
                      Count prod (q - 2) - these are the twin candidates.

A slot needs two exposed numbers straddling a multiple of 6, so the numbers belonging to a
slot number `2 prod (q - 2)`, and

    fraction of exposed numbers that belong to an exposed slot = prod (q - 2)/(q - 1)

exactly, for every sub-machine. That is the n-wise comparison to the 6-cycle: adding a gear
removes members at rate `(q - 1)/q` but removes slots at rate `(q - 2)/q`, so slots thin out
faster than members, by exactly that product.

Threat multiplicity. By CRT the slots threatened by *precisely* the gears of a subset `T` and
by no other gear of `S` number

    2^|T| * prod over q in S \\ T of (q - 2)

since each threatening gear contributes 2 residues and each non-threatening gear contributes
`q - 2`. The pairwise "coincidence is always 4" is the case `T = S`, `|T| = 2`: `2^2 * 1 = 4`.
Summing over all `T` of size `j` gives the number of slots threatened by exactly `j` gears.
"""

import itertools
import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, tooth


def threat_mask(gear_set):
    """Per-slot count of how many gears of the set threaten it, over one period."""
    P = prod(gear_set)
    counts = np.zeros(P, dtype=np.int16)
    for q in gear_set:
        u = tooth(q)
        for t in (u, q - u):
            counts[t::q] += 1
    return counts


def exposed_numbers(gear_set):
    """Members surviving in one period 6P: n = +/-1 mod 6, divisible by no gear."""
    P = prod(gear_set)
    total = 0
    for m in range(P):
        for member in (6 * m - 1, 6 * m + 1):
            if all(member % q for q in gear_set):
                total += 1
    return total


def multiplicity_distribution(gear_set):
    """Observed and predicted counts of slots threatened by exactly j gears."""
    counts = threat_mask(gear_set)
    n = len(gear_set)
    observed = [int((counts == j).sum()) for j in range(n + 1)]
    predicted = []
    for j in range(n + 1):
        total = 0
        for T in itertools.combinations(gear_set, j):
            rest = [q for q in gear_set if q not in T]
            total += (2**j) * prod(q - 2 for q in rest)
        predicted.append(total)
    return observed, predicted


def lattice_row(gear_set):
    """One row of the n-wise table."""
    P = prod(gear_set)
    n = len(gear_set)
    p1 = prod(q - 1 for q in gear_set)
    p2 = prod(q - 2 for q in gear_set)
    p4 = prod(max(q - 4, 0) for q in gear_set)
    return {
        "gears": tuple(int(q) for q in gear_set), "n": n, "period": P,
        "slip_vs_6": P % 6,
        "exposed_numbers": 2 * p1,
        "exposed_slots": p2,
        "numbers_in_slots": 2 * p2,
        "slot_fraction": p2 / p1,
        "runs": p2 - p4,
        "pair_alignments": p4,
        "all_threatened": 2**n,
    }


if __name__ == "__main__":
    G = gears_upto(19)

    print("n-WISE LATTICE, each level compared to the 6-cycle")
    print(f"  {'gears':>22} {'n':>2} {'period P':>9} {'P mod 6':>8} "
          f"{'exposed numbers':>16} {'exposed slots':>14} {'slot fraction':>14} "
          f"{'runs':>7} {'pair aligns':>12}")
    for r in range(1, len(G) + 1):
        for S in itertools.combinations(G, r):
            row = lattice_row(list(S))
            print(f"  {str(row['gears']):>22} {row['n']:>2} {row['period']:>9} "
                  f"{row['slip_vs_6']:>8} {row['exposed_numbers']:>16} "
                  f"{row['exposed_slots']:>14} {row['slot_fraction']:>14.6f} "
                  f"{row['runs']:>7} {row['pair_alignments']:>12}")
        if r >= 3:
            break

    print("\nverify exposed-number count = 2 prod(q-1) by direct enumeration")
    for S in [(5,), (5, 7), (7, 11), (5, 7, 11)]:
        got = exposed_numbers(list(S))
        want = 2 * prod(q - 1 for q in S)
        print(f"  {str(S):>14}: direct {got:>8}, 2 prod(q-1) = {want:>8}, "
              f"agree {got == want}")

    print("\nslot fraction prod((q-2)/(q-1)) as gears accumulate")
    print(f"  {'gears up to':>12} {'n':>3} {'slot fraction':>14} "
          f"{'exposed numbers':>16} {'exposed slots':>14}")
    for y in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        S = gears_upto(y)
        row = lattice_row(S)
        print(f"  {y:>12} {row['n']:>3} {row['slot_fraction']:>14.6f} "
              f"{row['exposed_numbers']:>16} {row['exposed_slots']:>14}")

    print("\nTHREAT MULTIPLICITY: slots threatened by exactly j gears")
    for S in [(5, 7), (5, 7, 11), (5, 7, 11, 13), (7, 11, 13, 17)]:
        obs, pred = multiplicity_distribution(list(S))
        ok = obs == pred
        print(f"  gears {str(S):>18}: observed {obs}")
        print(f"  {'':>24} predicted {pred}   agree {ok}")

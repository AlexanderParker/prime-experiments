"""The record as a field (owner: go the field; 2026-09-13). The wheel of the gears 5..y on the
m-line (column k = the pair (12k - 1, 12k + 1); gear h paints k = -+12^-1 mod h) at its worst
phase: the longest painted run R_y over a full period. Drawn as rows (gears) against the
columns of the run with margins; read: which rows paint each column of the run, columns with a
single painter, the run's mirror structure (the wheel is symmetric about every multiple of h,
hence the record runs come in mirror pairs about the multiples of P/2), the overpaint (sum of
paint over the run against its length), and how the run's start sits against each gear's
period. Also the first few record positions per y and whether each run is its own mirror
image (palindrome of painter sets).
Usage: uv run python record_field.py ymax   (y up to 29 exact; 31 is too large for a full period)
"""
import sys
import numpy as np
from sympy import primerange
sys.path.insert(0, 'research/stack/r8')
from submachine_phase import opens_mline


def painters(k, gears):
    return [h for h in gears if (12 * k - 1) % h == 0 or (12 * k + 1) % h == 0]


def main():
    ymax = int(sys.argv[1]); primes = list(primerange(5, ymax + 1))
    starts = {}
    for y in primes:
        gears = [h for h in primes if h <= y]
        ks, P = opens_mline(gears)
        gaps = np.diff(np.concatenate([ks, [ks[0] + P]])) - 1
        R = int(gaps.max()); where = np.nonzero(gaps == R)[0]
        runs = [(int(ks[i]) + 1, int(ks[i]) + R) for i in where]  # painted k from a to b inclusive
        starts[y] = (R, P, runs)
        a, b = runs[0]
        cols = list(range(a, b + 1)); pm = [painters(k, gears) for k in cols]
        single = sum(1 for p in pm if len(p) == 1); over = sum(len(p) for p in pm)
        pal = all(sorted(pm[i]) == sorted(pm[-1 - i]) for i in range(len(pm)))
        centre = (a + b) / 2
        mirror_pairs = sum(1 for (a1, b1) in runs for (a2, b2) in runs if (a1 + b2) % P == (P - 2) % P or (a1 + b2 + 2) % P == 0)
        print(f"y = {y}: record R = {R}, period {P}, record runs {len(runs)} (first {runs[:3]}); run {a}..{b}: single-painter columns {single} of {R}, total paint {over} ({over/R:.2f} per column); palindrome of painter sets: {pal}; centre {centre}, centre mod gears {[(h, round(centre % h, 1)) for h in gears]}")
        print("   painters per column: " + " ".join("/".join(str(x) for x in p) for p in pm))


if __name__ == "__main__":
    main()

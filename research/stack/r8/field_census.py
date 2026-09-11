"""Each field on its own: a census per field j on one section, no grouping.

For field j on the section [lo, hi) (numbers on S = +-1 mod 6):
  size; left/right split (class); the least-factor profile (which gear carries the member,
  n = g * m with g the least prime factor: the count per g and the share of the smallest gear);
  the largest least factor present (confinement); the profile of the second factor's field
  (m's index is j-1: by D1; report how many members have m prime / semiprime ... as a check);
  the longest range of S free of field j, and where; the first member and the last;
  hits per column: columns hit by field j on the left only, right only, both members;
  the count of columns where field j is the ONLY field hitting (its exclusive kills);
  the mirror axes 6 g m (g = 5, 7) : number of symmetric pairs of field j about them.

Usage: uv run python field_census.py lo hi jmax
"""
import sys
from collections import Counter
from sympy import factorint, primerange


def main():
    lo, hi, jmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    S = [n for n in range(lo, hi) if n % 6 in (1, 5)]
    om, lpf = {}, {}
    for n in S:
        f = factorint(n); om[n] = sum(f.values()); lpf[n] = min(f)
    Sset = set(S)
    cols = [k for k in range((lo + 1) // 6 + 1, (hi - 2) // 6 + 1) if 6 * k - 1 >= lo and 6 * k + 1 < hi]
    top = max(primerange(2, int(hi ** 0.5) + 2))
    print(f"section [{lo}, {hi}): |S| = {len(S)}, columns {len(cols)}, top gear below sqrt(hi) = {top}")
    for j in range(1, jmax + 1):
        F = [n for n in S if om[n] == j]
        if not F:
            print(f"\nfield {j}: empty"); continue
        left = sum(1 for n in F if n % 6 == 5); right = len(F) - left
        lp = Counter(lpf[n] for n in F); g_max = max(lp)
        first, last = F[0], F[-1]
        # longest free range on S
        Fs = set(F); best = (0, None, None); cnt = 0; start = None; prev = None
        for n in S:
            if n in Fs:
                if start is not None and cnt > best[0]: best = (cnt, start, prev)
                start = None; cnt = 0
            else:
                if start is None: start = n
                cnt += 1; prev = n
        if start is not None and cnt > best[0]: best = (cnt, start, prev)
        # per column
        L_only = R_only = both = excl = 0
        for k in cols:
            a, b = 6 * k - 1, 6 * k + 1
            hl, hr = om[a] == j, om[b] == j
            if hl and hr: both += 1
            elif hl: L_only += 1
            elif hr: R_only += 1
            if (hl or hr) and not ((om[a] >= 2 and om[a] != j) or (om[b] >= 2 and om[b] != j)):
                excl += 1
        # mirror pairs about 6 g m for g = 5, 7 (members n, n' of field j with n + n' = 12 g m, both multiples of g)
        mir = {}
        for g in (5, 7):
            c = 0
            byg = [n for n in F if n % g == 0]
            s = set(byg)
            for n in byg:
                m = (n // g + 1) // 6 if (n // g) % 6 == 5 else (n // g - 1) // 6  # column of n/g
                partner = 12 * g * m - n
                if partner > n and partner in s: c += 1
            mir[g] = c
        print(f"\nfield {j}: size {len(F)} ({100*len(F)/len(S):.1f}% of S); left {left} / right {right}; first {first} = {factorint(first)}, last {last}")
        print(f"  least factor: max {g_max}; top five gears by count {lp.most_common(5)}; share of gear 5: {lp[5]/len(F):.3f}, of gears <= 13: {sum(v for g,v in lp.items() if g <= 13)/len(F):.3f}")
        print(f"  longest field-{j}-free range on S: {best[0]} members, {best[1]}..{best[2]}")
        print(f"  columns hit: left only {L_only}, right only {R_only}, both {both}; columns where field {j} is the only striking field: {excl}")
        print(f"  mirror pairs about 6*5*m: {mir[5]}, about 6*7*m: {mir[7]}")


if __name__ == "__main__":
    main()

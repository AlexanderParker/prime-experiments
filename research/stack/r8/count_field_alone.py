"""Each factor-count field in isolation: its own gap spectrum and class pattern, no other field.

Field j = members of S (n = +-1 mod 6) with exactly j prime factors (multiplicity), all >= 5.
On [lo, hi): the gaps between consecutive members measured in S-steps (positions in the
sequence of S), the five commonest gaps with counts, the largest gap and where, the class
pattern (the count of same-class consecutive pairs against opposite-class, and the longest
run of one class), the density in successive tenths of the section, and self-similarity: the
share of members divisible by 5 (which equals the dilate 5 . field_{j-1}) and by 7.
Usage: uv run python count_field_alone.py lo hi jmax
"""
import sys
from collections import Counter
from sympy import factorint


def main():
    lo, hi, jmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    S = [n for n in range(lo, hi) if n % 6 in (1, 5)]
    pos = {n: i for i, n in enumerate(S)}
    om = {n: sum(factorint(n).values()) for n in S}
    print(f"section [{lo}, {hi}): |S| = {len(S)}")
    for j in range(1, jmax + 1):
        F = [n for n in S if om[n] == j]
        if len(F) < 2: print(f"field {j}: {len(F)} member(s)"); continue
        gaps = [pos[b] - pos[a] for a, b in zip(F, F[1:])]
        gc = Counter(gaps); mx = max(gaps); at = F[gaps.index(mx)]
        same = sum(1 for a, b in zip(F, F[1:]) if a % 6 == b % 6); opp = len(F) - 1 - same
        run = best = 1
        for a, b in zip(F, F[1:]):
            run = run + 1 if a % 6 == b % 6 else 1; best = max(best, run)
        tenths = [0] * 10
        for n in F: tenths[min(9, (n - lo) * 10 // (hi - lo))] += 1
        d5 = sum(1 for n in F if n % 5 == 0); d7 = sum(1 for n in F if n % 7 == 0)
        print(f"field {j}: {len(F)} members; mean gap {sum(gaps)/len(gaps):.2f} S-steps; commonest gaps {gc.most_common(5)}; largest gap {mx} after {at}")
        print(f"   class pattern: same-class consecutive {same}, opposite {opp} (ratio {same/max(opp,1):.2f}); longest one-class run {best}")
        print(f"   members per tenth of the section: {tenths}")
        print(f"   divisible by 5: {d5} ({100*d5/len(F):.1f}%), by 7: {d7} ({100*d7/len(F):.1f}%)")


if __name__ == "__main__":
    main()

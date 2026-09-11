"""Each gear field in isolation: its own structure, no other field involved.

Gear field of g on the whole line (not a section): F_g = {g m : m in S, m has no prime factor
below g, m >= g}. In columns: n = g m hits column (n -+ 1)/6.
Claims tested for each g in 5..23:
  (P) periodicity: the set of hit columns is periodic with period P_g = product of the gears
      h with 5 <= h <= g (the dilate of the lower machine's open set), beyond the first
      period; hits per period = 2 * product over 5 <= h < g of (h - 2);
  (R) the residues hit mod P_g, listed for g = 5, 7 (and counted for the rest);
  (M) mirror symmetry: r hit iff P_g - r hit (in the appropriate coordinate: the members
      g m and g (P' - m) ... we test the column set directly: r hit iff (-r - c) mod P_g hit
      for the constant c that the fold's offset imposes; we search c in 0..2);
  (S) the gap spectrum of hit columns within one period (multiset of gaps), and the maximal gap
      = g times the lower machine's record gap on S (the wheel's record), compared with the
      record F_top of the machine {5..g-1} from the register (F_top of the wheel below g).
Usage: uv run python gear_field_alone.py
"""
from math import prod
from collections import Counter
from sympy import primerange


def main():
    gears = list(primerange(5, 24))
    for g in gears:
        lower = [h for h in gears if h < g]
        P = prod(gears[: gears.index(g) + 1])  # product of gears 5..g
        # hit columns over two periods starting at a safe offset (columns >= g^2/6)
        start = (g * g) // 6 + 1
        hits = []
        for k in range(start, start + 2 * P + 1):
            for n in (6 * k - 1, 6 * k + 1):
                if n % g == 0:
                    m = n // g
                    if m >= g and all(m % h for h in lower):
                        hits.append(k); break
        hs = set(hits)
        # periodicity check: k in hs iff k + P in hs for k in the first period
        first = [k for k in hits if k < start + P]
        per_ok = all(((k in hs) == ((k + P) in hs)) for k in range(start, start + P))
        expected = 2 * prod(h - 2 for h in lower)
        # mirror: find c with r hit iff (c - r) mod P hit
        res = sorted(k % P for k in first)
        rs = set(res); mirror_c = None
        for c in range(P):
            if all(((c - r) % P) in rs for r in res):
                mirror_c = c; break
        gaps = Counter(b - a for a, b in zip(first, first[1:]))
        maxgap = max(gaps) if gaps else None
        print(f"gear {g}: period {P} columns; hits per period {len(first)} (expected 2*prod(h-2) = {expected}); periodic: {per_ok}; mirror centre c = {mirror_c}; max gap {maxgap} columns; gap spectrum {dict(sorted(gaps.items()))}")
        if g in (5, 7):
            print(f"   residues hit mod {P}: {res}")


if __name__ == "__main__":
    main()

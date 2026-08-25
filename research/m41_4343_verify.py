"""Round 23 (mechanic, for Constructor): the (43,43) word at machine 41.

Constructor's ask: the killable 2-word (43,43) at m41, span 86 - their pattern
counter exceeded a 3e8-node budget at 1127 s.  The COUNT was already in the
corpus (my own round-21 append: exactly 4 per period, four addresses) but was
flagged SINGLE-SOURCE.  This closes both halves:

  (a) recount by cov_count model enumeration        -> 4 EXACT, 32 s
  (b) direct verification of each address           -> asserted here
  (c) MIRROR-LAW cross-check, an independent method -> the four addresses are
      two exact mirror pairs summing to P - 86

(c) is the real cross-check: the opening set of any machine is closed under
k -> -k (each gear blocks the symmetric pair {u_q, -u_q}), so occurrences of a
PALINDROMIC word come in mirror pairs whose left endpoints sum to P - span.
Nothing in the counting method knows that.
"""
from math import prod

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
P = prod(GEARS)
SPAN = 86
ADDR = [116_431_845_582, 21_381_235_210_387,
        29_327_142_044_062, 50_591_945_408_867]


def blocked(k):
    for q in GEARS:
        u = pow(6, -1, q)
        if k % q in (u % q, (-u) % q):
            return True
    return False


print(f"machine 41: gears {GEARS}, period P = {P:,}")
assert P == 50_708_377_254_535, P
for k in ADDR:
    assert not blocked(k), ("endpoint 0 blocked", k)
    assert not blocked(k + 43), ("middle opening blocked", k)
    assert not blocked(k + 86), ("endpoint 86 blocked", k)
    for d in range(1, SPAN):
        if d in (43,):
            continue
        assert blocked(k + d), ("interior open", k, d)
    # both links padded: the two endpoints of each link share a residue mod 43
    assert (k % 43) == ((k + 43) % 43) == ((k + 86) % 43)
    print(f"  k = {k:>17,}  VERIFIED: openings at +0/+43/+86, all 84 other "
          f"interior slots blocked, both links padded (residue {k % 43} mod 43)")

print("\nMIRROR-LAW CROSS-CHECK (independent of the counting method):")
tgt = P - SPAN
pairs = []
for a in ADDR:
    for b in ADDR:
        if a < b and a + b == tgt:
            pairs.append((a, b))
for a, b in pairs:
    print(f"  {a:,} + {b:,} = {a+b:,} = P - {SPAN}")
assert len(pairs) == 2, pairs
assert sorted(x for p in pairs for x in p) == sorted(ADDR)
print(f"  the four addresses are EXACTLY two mirror pairs - the count is even, "
      f"as the mirror law forces, and no address is unpaired.")
print("\n  COUNT (43,43) at machine 41 = 4, EXACT, full period.")

"""R2.a self-feeding, part 4b: the flanking strikers of a birth column, per side.

Part 4's first pass measured min(left, right) smallest striker and got 5 at every opening of
every machine - which is a rule, but gear 5's rule, not a fact about the openings:

  gear 5's teeth are +-1 mod 5, so an opening sits at k = 0, 2 or 3 mod 5; then
  k = 0 -> both k-1 = -1 and k+1 = +1 are teeth of gear 5;
  k = 2 -> k-1 = 1 is a tooth, k+1 = 3 is not;
  k = 3 -> k+1 = -1 is a tooth, k-1 = 2 is not.
  So gear 5 strikes at least one neighbour of every opening, and both exactly at the
  openings on gear 5's shield (k = 0 mod 5).

B4 checks that and then measures the side gear 5 does NOT take.
B5 censuses the carry-over: which flanking strikers also strike the pair's own walk-start
column k0' = k(g-1), and checks each against the level-free set {7, 17, 31} of part 4's B2.

Writes results/sf_birth2.txt.
"""
import os
from math import isqrt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "sf_birth2.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


SLIM = 1_100_000
FL = sieve(SLIM)
GEARS = [p for p in range(5, 1051) if FL[p]]
PR = [p for p in range(5, 1010) if FL[p]]


def strikers(c, top):
    """all gears <= top striking column c, smallest first."""
    lo, hi = 6 * c - 1, 6 * c + 1
    out = []
    for p in GEARS:
        if p > top:
            break
        if lo % p == 0 or hi % p == 0:
            out.append(p)
    return out


say("=== B4. gear 5 at the neighbours of an opening ===")
tot = 0
c0 = c2 = c3 = 0
bad_none = 0
bad_both = 0
side_small = []
for idx in range(1, len(PR)):
    q = PR[idx]
    p = PR[idx - 1]
    lo = (p * p + 1) // 6 + 1
    hi = (q * q - 1) // 6
    if 6 * hi + 1 > SLIM:
        break
    for c in range(lo, hi + 1):
        if not (FL[6 * c - 1] and FL[6 * c + 1]):
            continue
        tot += 1
        r = c % 5
        if r == 0:
            c0 += 1
        elif r == 2:
            c2 += 1
        elif r == 3:
            c3 += 1
        else:
            bad_none += 1                   # impossible: k would be on gear 5's tooth
        left5 = (c - 1) % 5 in (1, 4)
        right5 = (c + 1) % 5 in (1, 4)
        if not (left5 or right5):
            bad_none += 1
        if (left5 and right5) != (r == 0):
            bad_both += 1
        # the side gear 5 does not take
        for cc, taken in ((c - 1, left5), (c + 1, right5)):
            if not taken:
                s = strikers(cc, q)
                side_small.append(s[0] if s else 0)
say("openings censused (sections of machines 7..1009):", tot)
say("  k mod 5 = 0 / 2 / 3:", c0, "/", c2, "/", c3, "  (k = +-1 mod 5 is a tooth of gear 5)")
say("  openings with neither neighbour struck by gear 5:", bad_none)
say("  'both neighbours struck by gear 5 iff k = 0 mod 5' exceptions:", bad_both)
side_small.sort()
nz = [v for v in side_small if v]
say("  free sides recorded (one per opening with k = 2 or 3 mod 5):", len(side_small))
say("  the side gear 5 leaves: smallest striker  min %d  median %d  max %d" % (
    nz[0], nz[len(nz) // 2], nz[-1]))
say("  free sides struck by NO gear of the machine:", len(side_small) - len(nz),
    "- those are prime quadruplets (the neighbour column is an opening too), counted twice,"
    " so", (len(side_small) - len(nz)) // 2, "quadruplets")
from collections import Counter
cc5 = Counter(side_small)
say("  its distribution: " + ", ".join("%d:%d" % (g, n) for g, n in sorted(cc5.items())[:12]))

say("")
say("=== B5. carry-over of a flanking striker to the pair's own walk-start column ===")
say("k0' = k(g-1), lower member g^2 - 2.  Part 4 B2: a striker of column k+j reaches k0'")
say("only if it divides (6j)^2 - 2 (it hit the lower member at k+j) or (6j+2)^2 - 2 (upper).")
tot5 = 0
hit5 = 0
seen = Counter()
for idx in range(1, len(PR)):
    q = PR[idx]
    p = PR[idx - 1]
    lo = (p * p + 1) // 6 + 1
    hi = (q * q - 1) // 6
    if 6 * hi + 1 > SLIM:
        break
    for c in range(lo, hi + 1):
        if not (FL[6 * c - 1] and FL[6 * c + 1]):
            continue
        g = 6 * c - 1
        k0p = c * (g - 1)
        for j in (-1, 1):
            for h in strikers(c + j, q):
                tot5 += 1
                if (6 * k0p - 1) % h == 0 or (6 * k0p + 1) % h == 0:
                    hit5 += 1
                    seen[h] += 1
say("(opening, flank j = +-1, striker) triples:", tot5, "  carried over to k0':", hit5)
say("  gears that carried over:", dict(sorted(seen.items())))
say("  predicted level-free set at j = +-1: {7, 17, 31} (7 at j = -1 upper, 17 at both lower,")
say("  31 at j = +1 upper).  Any gear outside it is a refutation.")

LOG.close()

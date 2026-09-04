"""R2.a self-feeding, part 4: where an opening is born against what it does as a gear.

An opening k of a machine's window is a twin pair (g, g+2) = (6k-1, 6k+1); as a gear pair its
teeth are +-k mod g and mod g+2, so k is the only thing that crosses levels (P3.1).  The
question that remains is whether a gear sitting near the birth column can act near the pair's
own working region one level up.

TRANSFER RULE (derived, then checked here).  Let h strike column k+j at the birth level, and
let k0' = k(g-1) be the first column of the pair's own walk (its lower member is g^2 - 2).
Then h strikes column k0' + i if and only if

    h | 36 j^2 + 6 i - 2   or   h | 36 j^2 + 6 i        (h hit k+j on the LOWER member)
    h | (6j+2)^2 + 6 i - 2 or   h | (6j+2)^2 + 6 i      (h hit k+j on the UPPER member)

The right-hand sides contain neither k nor g: the pair of offsets (j, i) alone decides which
gears can carry over.  B1 checks the rule; B2 prints the j = +-1, i = 0 case (the square gate);
B3 measures how often a flanking striker actually carries over, on the section of real machines.

Writes results/sf_birth.txt.
"""
import os
from math import isqrt, gcd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "sf_birth.txt"), "w")


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


FL = sieve(4_100_000)
GEARS = [p for p in range(5, 20001) if FL[p]]
say("gears used:", len(GEARS), "up to", GEARS[-1])

# ------------------------------------------------------------------ B1
say("")
say("=== B1. the transfer rule ===")
JS = range(-8, 9)
IS = range(0, 41)
pairs = 0
tested = 0
bad = 0
for g in range(5, 20001, 6):
    if not (FL[g] and FL[g + 2]):
        continue
    k = (g + 1) // 6
    pairs += 1
    k0p = k * (g - 1)
    for j in JS:
        c = k + j
        if c < 1:
            continue
        lo, hi = 6 * c - 1, 6 * c + 1
        fac = set()
        for n in (lo, hi):
            m = n
            for h in GEARS:
                if h * h > m:
                    break
                while m % h == 0:
                    fac.add(h)
                    m //= h
            if m >= 5:
                fac.add(m)
        for h in sorted(fac):
            side = 0
            if lo % h == 0:
                side |= 1
            if hi % h == 0:
                side |= 2
            if side == 0:
                continue
            for i in IS:
                cc = k0p + i
                actual = (6 * cc - 1) % h == 0 or (6 * cc + 1) % h == 0
                pred = False
                if side & 1:
                    v = 36 * j * j + 6 * i
                    pred = pred or v % h == 0 or (v - 2) % h == 0
                if side & 2:
                    v = (6 * j + 2) ** 2 + 6 * i
                    pred = pred or v % h == 0 or (v - 2) % h == 0
                tested += 1
                if actual != pred:
                    bad += 1
                    if bad < 5:
                        say("  MISMATCH g=%d j=%d i=%d h=%d side=%d" % (g, j, i, h, side))
say("twin pairs with g <= 20000:", pairs)
say("(pair, j, striker h, i) checks:", tested, "  mismatches:", bad)

# ------------------------------------------------------------------ B2
say("")
say("=== B2. the square-gate case: which flanking strikers can strike the walk start ===")
say("i = 0: h | 36 j^2 - 2 = 2(18 j^2 - 1)  [lower-member flank] or")
say("       h | (6j+2)^2 - 2 = 2(18 j^2 + 12 j + 1)  [upper-member flank]")
for j in range(-4, 5):
    a = 18 * j * j - 1
    b = 18 * j * j + 12 * j + 1
    fa = [p for p in GEARS if p <= abs(a) and a % p == 0] if a else []
    fb = [p for p in GEARS if p <= abs(b) and b % p == 0] if b else []
    say("  j = %+d: lower-member flank gears dividing %-5d -> %s ;  upper-member flank gears"
        " dividing %-5d -> %s" % (j, a, fa or "none", b, fb or "none"))
say("  so a gear striking the column immediately after the birth column (j = +1) reaches the")
say("  pair's own walk start only if it is 17 (lower member) or 31 (upper member); at j = -1")
say("  only 17 (lower) or 7 (upper).  Level-free.")

# ------------------------------------------------------------------ B3
say("")
say("=== B3. flanking strikers on the section of real machines ===")


def minstrike(c, top):
    lo, hi = 6 * c - 1, 6 * c + 1
    for p in GEARS:
        if p > top:
            return 0
        if lo % p == 0 or hi % p == 0:
            return p
    return 0


PR = [p for p in range(5, 5001) if FL[p]]
say("machine q, section (p^2, q^2], #openings, min/median/max smallest flanking striker,")
say("openings whose two flanking strikers are both > q^(1/2), carry-overs to the walk start")
carry_tot = 0
carry_hit = 0
for idx in range(1, len(PR)):
    q = PR[idx]
    p = PR[idx - 1]
    if q > 300 and q not in (307, 401, 503, 701, 1009, 2003, 3001, 4999):
        continue
    lo = (p * p + 1) // 6 + 1
    hi = (q * q - 1) // 6
    if hi > 4_000_000:
        continue
    ops = []
    for c in range(lo, hi + 1):
        if FL[6 * c - 1] and FL[6 * c + 1]:
            ops.append(c)
    if not ops:
        continue
    mf = []
    both_big = 0
    for c in ops:
        a = minstrike(c - 1, q)
        b = minstrike(c + 1, q)
        mf.append(min(a, b) if a and b else max(a, b))
        if a * a > q and b * b > q:
            both_big += 1
        # carry-over: does either flanking striker also strike the pair's own walk start?
        g = 6 * c - 1
        k0p = c * (g - 1)
        for h in (a, b):
            if h:
                carry_tot += 1
                if (6 * k0p - 1) % h == 0 or (6 * k0p + 1) % h == 0:
                    carry_hit += 1
    mf.sort()
    say("  q=%-5d section %d..%d  openings %-5d  smallest flank striker min %d med %d max %d"
        "  both flanks > sqrt(q): %d" % (q, lo, hi, len(ops), mf[0], mf[len(mf) // 2], mf[-1],
                                          both_big))
say("flanking strikers tested for carry-over to the pair's own walk start:", carry_tot,
    " carried over:", carry_hit)
say("  every carry-over must be one of the level-free gears of B2 (7, 17, 31 at j = +-1).")

LOG.close()

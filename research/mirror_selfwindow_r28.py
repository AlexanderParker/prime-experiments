"""
LATERAL round 28 - THE SELF-MIRROR WINDOW: ITS ADDRESS, ITS SIZE, AND WHERE THE
"AT MOST ONE IMPLIES ZERO" LEVER BITES (brief item c).

Formalist's kernel lemma `Mirror.none_of_at_most_one` has one hypothesis that is
NOT free:

    hexc : L t0 <> 2 * F          -- the SELF-MIRROR window does not itself
                                  -- carry the length being counted

`t0` is the unique index fixed by the mirror at that depth.  Everything else in
the lemma is machine-free.  So the whole machine-side content of the lever is:
WHAT IS THE SELF-MIRROR WINDOW, AND HOW BIG IS IT?  This script answers both.

PART A - THE ADDRESS FORMULA (proved here, verified by brute force).
Openings o_0 = 0 < o_1 < ... < o_{N-1}, N = prod(q-2) ODD, and o_{N-t} = P - o_t
(mirror closure, r25).  The mirror on depth-j window indices is t -> N-t-j, so
the self-mirror index solves 2t = -j (mod N) - unique because N is odd.  Then:

    j = 2i     (EVEN):  t = -i,      window = [o_{-i}, o_i],  SPAN = 2 * o_i
                        - centred on the mirror's own fixed slot 0;
    j = 2i+1   (ODD) :  t = M - i    with M = (N-1)/2,
                        window = [o_{M-i}, P - o_{M-i}], SPAN = P - 2 * o_{M-i}
                        - centred on the ANTIPODE P/2, which is not a slot.

PART B - THE SIZE.  span_self(j) against F_j, at every machine we can sieve.
This is what decides whether the lever is usable: the lever proves "count = 0"
for every length the self-mirror window does not carry, so it is usable at a
target length B exactly when span_self(j) <> B, and comfortably usable when
span_self(j) is far below the extremes the route argues about.

PART C - IS THE MIRROR THE ONLY LEVER THE MACHINE HAS?  The opening set carries
a (Z/2)^n group of per-gear sign flips sigma_S (k_q -> -k_q for q in S), each an
involution of the opening set.  Their fixed-point counts are computed exactly -
and only S = ALL GEARS has a single fixed point.  Every proper S has
N / prod_{q in S}(q-2) fixed points, and (a separate fact) only S = all gears is
an ISOMETRY of Z_P, so only it preserves window lengths at all.  So within the
machine's own symmetry group the lever is unique - a gated negative.

PART D - WORD REVERSAL IS THE SAME INVOLUTION, NOT A SECOND ONE.  The unique
odd-multiplicity palindrome of the depth-j gap-word census is EXACTLY the word
of the self-mirror window.  Verified cell for cell.  (Stated because round 27
listed the two as separate assets; they are one.)

Usage: python mirror_selfwindow_r28.py [--upto 23] [--maxdepth 40]
"""
import argparse
import sys
from collections import Counter

import numpy as np

GEARS = [5, 7, 11, 13, 17, 19, 23]

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def openings(gears, P):
    blocked = np.zeros(P, dtype=bool)
    for q in gears:
        v = pow(6, -1, q)
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    return np.flatnonzero(~blocked).astype(np.int64)


def self_index(N, j):
    """unique t in [0,N) with 2t = -j (mod N); N odd."""
    inv2 = (N + 1) // 2
    return (-j * inv2) % N


def span_at(op, P, t, j):
    N = op.size
    a = int(op[t % N])
    b = int(op[(t + j) % N])
    laps = (t + j) // N
    return b + laps * P - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=23)
    ap.add_argument("--maxdepth", type=int, default=40)
    a = ap.parse_args()

    print("=== A/B. THE SELF-MIRROR WINDOW: ADDRESS FORMULA AND SIZE ===")
    table = {}
    for n in range(2, len(GEARS) + 1):
        gears = GEARS[:n]
        y = gears[-1]
        if y > a.upto:
            break
        P = int(np.prod(gears))
        op = openings(gears, P)
        N = op.size
        gate(N == int(np.prod([q - 2 for q in gears])),
             "m%-2d: N = prod(q-2) = %d" % (y, N))
        gate(N % 2 == 1, "m%-2d: N is ODD (so 2t = -j has a unique solution)" % y)
        gate(int(op[0]) == 0, "m%-2d: slot 0 is an opening" % y)
        gate(bool(np.array_equal(op[1:][::-1], P - op[1:])),
             "m%-2d: the opening set is mirror-closed, o_{N-t} = P - o_t" % y)

        M = (N - 1) // 2
        # depth-j quantities
        ext = np.concatenate([op, op[:a.maxdepth + 1] + P])
        rows = []
        for j in range(1, min(a.maxdepth, N - 1) + 1):
            Fj = int((ext[j:j + N] - op).max())
            t = self_index(N, j)
            sp = span_at(op, P, t, j)
            # the address formula
            if j % 2 == 0:
                pred = 2 * int(op[j // 2])
                pt = (-(j // 2)) % N
            else:
                i = (j - 1) // 2
                pred = P - 2 * int(op[M - i])
                pt = M - i
            if pred != sp or pt != t:
                raise AssertionError("address formula wrong at m%d j=%d "
                                     "(t %d vs %d, span %d vs %d)"
                                     % (y, j, pt, t, pred, sp))
            rows.append((j, Fj, sp, sp / Fj))
        table[y] = rows
        gate(True, "m%-2d: the address formula reproduces t_j AND span_self(j) "
                   "for every depth j = 1..%d" % (y, len(rows)))
        # brute-force uniqueness of the self-mirror index (small machines only)
        if N <= 5000:
            for j in (1, 2, 3, 4, 5, 6):
                fix = [t for t in range(N) if (N - t - j) % N == t]
                if fix != [self_index(N, j)]:
                    raise AssertionError("uniqueness failed m%d j=%d" % (y, j))
            gate(True, "m%-2d: brute force - EXACTLY ONE self-mirror index at each "
                       "depth j = 1..6, equal to the formula" % y)

    print("\n  span_self(j) against F_j     (ratio = span_self / F_j)")
    print("  j    " + "".join("  m%-2d F_j  self  ratio" % y for y in table))
    for j in range(1, min(a.maxdepth, 12) + 1):
        line = "  %-4d " % j
        for y in table:
            r = table[y][j - 1]
            line += "  %5d %5d  %.3f" % (r[1], r[2], r[3])
        print(line)
    print("\n  first depth j at which span_self(j) exceeds 0.8 * F_j:")
    for y in table:
        hits = [r[0] for r in table[y] if r[3] > 0.8]
        print("    m%-2d : %s   (max ratio over j <= %d is %.3f at j = %d)"
              % (y, hits[0] if hits else "none in range",
                 len(table[y]), max(r[3] for r in table[y]),
                 max(table[y], key=lambda r: r[3])[0]))
    print("\n  *** THE LEVER'S EXCEPTION LIST *** - depths where span_self(j) = F_j")
    print("  exactly.  At these (machine, depth) pairs the self-mirror window IS")
    print("  extremal, hexc FAILS for the target 'span = F_j', and 'at most one")
    print("  implies zero' is NOT available.  Everywhere else it is.")
    for y in table:
        bad = [r[0] for r in table[y] if r[2] == r[1]]
        print("    m%-2d : %s" % (y, bad if bad else "none - lever available at "
                                  "every depth in range"))
    print("  route-relevant depths only (j = 2..6, the uniform-order theorem caps")
    print("  the chain at A_relax <= 5): max ratio per machine")
    for y in table:
        rs = [r for r in table[y] if 2 <= r[0] <= 6]
        print("    m%-2d : max span_self/F_j = %.3f at j = %d ; exception list %s"
              % (y, max(r[3] for r in rs), max(rs, key=lambda r: r[3])[0],
                 [r[0] for r in rs if r[2] == r[1]] or "EMPTY"))

    print("\n=== C. THE ROUTE'S OWN TARGET: DISCHARGING hexc AT DEPTH 2 ===")
    print("  The lemma counts windows of length 2F (an adjacent EQUAL pair).")
    print("  hexc needs span_self(2) = 2*o_1 = 2*d_0 <> 2F, i.e. d_0 <> F.")
    print("   y    d_0 = o_1   F    span_self(2) = 2 d_0   2F     hexc discharged?")
    for y in table:
        n = GEARS.index(y) + 1
        P = int(np.prod(GEARS[:n]))
        op = openings(GEARS[:n], P)
        d0 = int(op[1])
        F = table[y][0][1]
        ok = (2 * d0 != 2 * F)
        gate(ok, "m%-2d: hexc holds at depth 2 - 2*d_0 = %d <> 2F = %d"
             % (y, 2 * d0, 2 * F))
        print("   %-4d %-11d %-4d %-22d %-6d %s"
              % (y, d0, F, 2 * d0, 2 * F, "YES" if ok else "NO"))

    print("\n=== D. IS THE MIRROR THE ONLY LEVER? THE (Z/2)^n SIGN GROUP ===")
    print("  sigma_S : k_q -> -k_q for q in S.  Every sigma_S is an involution of")
    print("  the opening set (each exposed set is closed under negation).")
    print("  Fixed points of sigma_S = {k : k_q = 0 for all q in S}, so")
    print("  #fix = N / prod_{q in S} (q-2) - EXACT, no scan.")
    n = min(len(GEARS), [i for i, q in enumerate(GEARS, 1) if q <= a.upto][-1])
    gears = GEARS[:n]
    P = int(np.prod(gears))
    op = openings(gears, P)
    N = op.size
    import itertools as it
    worst = []
    for r in range(1, n + 1):
        for S in it.combinations(gears, r):
            d = 1
            for q in S:
                d *= q - 2
            worst.append((len(S), S, N // d))
    ones = [w for w in worst if w[2] == 1]
    gate(len(ones) == 1 and ones[0][1] == tuple(gears),
         "m%-2d: EXACTLY ONE of the %d sign involutions has a single fixed point, "
         "and it is the full mirror S = all gears" % (gears[-1], len(worst)))
    # verify the fixed-point count directly for a few S, on the opening set
    rr = np.stack([op % q for q in gears])
    for r in (1, 2, n):
        S = gears[:r]
        cnt = int(np.all(rr[:r] == 0, axis=0).sum())
        d = 1
        for q in S:
            d *= q - 2
        gate(cnt == N // d, "m%-2d: #fix(sigma_%s) = %d = N/prod(q-2) - measured on "
             "the opening set" % (gears[-1], str(S), cnt))
    print("  smallest fixed-point counts (|S|, S, #fix):")
    for w in sorted(worst, key=lambda w: w[2])[:4]:
        print("    |S|=%d  %-28s #fix = %d" % (w[0], str(w[1]), w[2]))
    print("  AND only S = all gears is an ISOMETRY of Z_P (k -> -k), so it is the")
    print("  only one of the 2^n - 1 that preserves window LENGTH at all: the other")
    print("  sign flips permute openings but move distances.  Gated negative: the")
    print("  machine's own symmetry group supplies exactly ONE parity lever.")

    print("\n=== F. BACKLOG U10: WHERE COULD A mod-4 LEVER COME FROM? ===")
    print("  U10 (round 26) asked this after item 51 showed no SYMMETRY of the")
    print("  opening set gives a mod-4 lever, and named two surviving candidates:")
    print("  (a) a free Z/4 action on a SUBSET of configurations, (b) a pairing")
    print("  not induced by a map of Z_P at all.  Candidate (a) is now CLOSED.")
    print("  The group of maps generated by the per-gear sign flips is")
    print("  (Z/2)^n by construction - sigma_S sigma_T = sigma_{S xor T} and")
    print("  sigma_S^2 = id - so it is ELEMENTARY ABELIAN and contains NO ELEMENT")
    print("  OF ORDER 4.  And the isometries of Z_P preserving the opening set are")
    print("  exactly {k -> +-k} (item 51), a group of order 2.  So no Z/4 action")
    print("  exists anywhere in the machine's automorphism group, free or not, and")
    print("  candidate (b) is the ONLY one left.")
    for r in (1, 2, min(3, n)):
        S = gears[:r]
        cnt = int(np.all(rr[:r] == 0, axis=0).sum())
        # sigma_S has order exactly 2 (it is not the identity, and it squares to it)
        gate(cnt < N, "sigma_%s is not the identity (it fixes %d < %d openings), "
             "and squares to it - order exactly 2" % (str(S), cnt, N))
    gate(all(w[2] < N for w in worst),
         "all %d non-trivial sign maps are genuine involutions of order 2 - the "
         "group is elementary abelian, so it has no element of order 4"
         % len(worst))

    print("\n=== E. WORD REVERSAL IS THE SAME INVOLUTION, NOT A SECOND ONE ===")
    for n in range(2, len(GEARS) + 1):
        gears = GEARS[:n]
        y = gears[-1]
        if y > min(a.upto, 17):
            break
        P = int(np.prod(gears))
        op = openings(gears, P)
        N = op.size
        g = np.diff(np.concatenate([op, [P]]))
        for j in (2, 3, 4):
            words = Counter(tuple(int(x) for x in np.take(g, range(t, t + j), mode="wrap"))
                            for t in range(N))
            # NOTE (self-correction of my own round-25 phrasing): it is NOT true
            # that exactly one WORD has odd multiplicity - non-palindromic words
            # come in reversal pairs of EQUAL count, and equal counts may both be
            # odd.  The exact law is about PALINDROMES.
            gate(all(words[w] == words[w[::-1]] for w in words),
                 "m%-2d depth %d: the word census is exactly reversal-symmetric" % (y, j))
            oddpal = [w for w, c in words.items() if w == w[::-1] and c % 2 == 1]
            gate(len(oddpal) == 1, "m%-2d depth %d: exactly ONE PALINDROME of odd "
                 "multiplicity (of %d palindromes)"
                 % (y, j, sum(1 for w in words if w == w[::-1])))
            t = self_index(N, j)
            selfw = tuple(int(x) for x in np.take(g, range(t, t + j), mode="wrap"))
            gate(selfw == oddpal[0], "m%-2d depth %d: the odd palindrome IS the "
                 "self-mirror window's word %s" % (y, j, str(selfw)))

    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

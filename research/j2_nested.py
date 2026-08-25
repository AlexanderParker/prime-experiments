"""Harvester round 23 (addendum 2): THE NESTED BRUN TRUNCATION, tested.

Round 23's first pass showed that the naive PRODUCT truncation
    D = {d : nu(d_j) <= K_j for every BAND j}
is not a valid lower-bound sieve (36 explicit counterexamples, j2_explicit.py
section D).  Tenenbaum, "Introduction to Analytic and Probabilistic Number Theory"
(GSM 163), the paragraph immediately preceding his fundamental lemma, describes the
correct refinement:

    "the method can be refined by introducing a partition of ]1,y] into
     subintervals ]y_j, y_{j+1}], 0 <= j <= k, and selecting for i = 1,2,
     chi_i(d) = mu(d) theta_i(d) where theta_i is the characteristic function of
     the set of those integers d having at most 2 h_j + 2 - i distinct prime
     factors in P intersect ]y_j, y] for each j, 0 <= j <= k"

TWO THINGS THAT CHANGES, and they are exactly what our 36 counterexamples were
missing:
  (i) the count is over ]y_j, y] - THE WHOLE UPPER TAIL - not over the single band
      ]y_j, y_{j+1}].  The constraints are therefore NESTED, not independent.
  (ii) the depth differs by one between the two sieves (2h_j + 2 - i, i = 1, 2),
      which is what makes them BRACKET instead of both overshooting.

This script tests that claim directly and exhaustively on small configurations.
Only the COUNTS matter: if band i carries m_i primes at which the position is bad,
and d takes c_i of them, then

    Lambda(x) = sum over admissible (c_0..c_k) of prod_i C(m_i, c_i) (-1)^{sum c_i},

and "admissible" is  sum_{i >= j} c_i <= H_j  for every j, with H_j = 2h_j + 2 - i.
Validity of the LOWER sieve is  Lambda(x) <= [x survives] = [all m_i = 0];
validity of the UPPER sieve is  Lambda(x) >= [all m_i = 0].

(Bands are indexed so that band k holds the LARGEST primes ]y_k, y] and band 0 the
smallest ]y_0, y_1]; the tail constraint at j therefore involves bands j..k.)
"""
from math import comb
from itertools import product

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def Lambda(m, H):
    """m = (m_0..m_k) bad-prime counts per band (band k = largest primes);
    H = (H_0..H_k) tail depths, constraint sum_{i>=j} c_i <= H_j."""
    k = len(m) - 1
    tot = 0
    for c in product(*[range(mi + 1) for mi in m]):
        ok = True
        for j in range(k + 1):
            if sum(c[j:]) > H[j]:
                ok = False
                break
        if ok:
            w = 1
            for mi, ci in zip(m, c):
                w *= comb(mi, ci)
            tot += w * (-1) ** sum(c)
    return tot


def main():
    say("=" * 78)
    say("N1 - the PRODUCT truncation (per-band counts) - the round-23 failure")
    say("=" * 78)

    def LambdaProduct(m, H):
        k = len(m) - 1
        tot = 0
        for c in product(*[range(mi + 1) for mi in m]):
            if all(c[j] <= H[j] for j in range(k + 1)):
                w = 1
                for mi, ci in zip(m, c):
                    w *= comb(mi, ci)
                tot += w * (-1) ** sum(c)
        return tot

    bad = []
    for H0 in (1, 3):
        for H1 in (1, 3):
            for m0 in range(0, 6):
                for m1 in range(0, 6):
                    if m0 == 0 and m1 == 0:
                        continue
                    v = LambdaProduct((m0, m1), (H0, H1))
                    if v > 0:
                        bad.append((H0, H1, m0, m1, v))
    say(f"  two bands, odd per-band depths in {{1,3}}, 0 <= m_i <= 5:")
    say(f"  {len(bad)} configurations with Lambda > 0 = the survivor indicator.")
    say(f"  smallest: H = ({bad[0][0]},{bad[0][1]}), m = ({bad[0][2]},{bad[0][3]}), "
        f"Lambda = {bad[0][4]}")
    assert bad, "the product truncation must fail"

    say("")
    say("=" * 78)
    say("N2 - the NESTED (upper-tail) truncation, exhaustively tested")
    say("=" * 78)
    say("  constraint: sum_{i >= j} c_i <= H_j for every j, H_j = 2 h_j + 2 - i.")
    say("  LOWER sieve i = 1: H_j odd.  UPPER sieve i = 2: H_j even.")
    say("")
    say("  bands  depth pattern            configs   lower ok   upper ok")
    total_lo = total_hi = 0
    checked = 0
    fails_lo = []
    fails_hi = []
    # h_j must be NON-INCREASING in j (deeper = smaller primes get larger depth),
    # which is what the partition is for; we test that family and also, as a
    # control, the reversed family.
    patterns = []
    for k in (1, 2, 3):
        for hs in product(range(0, 4), repeat=k + 1):
            patterns.append(tuple(hs))
    for hs in patterns:
        k = len(hs) - 1
        Hlo = tuple(2 * h + 1 for h in hs)     # i = 1
        Hhi = tuple(2 * h + 2 for h in hs)     # i = 2
        monotone = all(hs[j] >= hs[j + 1] for j in range(k))  # h non-increasing in j
        nlo = nhi = 0
        cfg = 0
        for m in product(range(0, 5), repeat=k + 1):
            surv = 1 if all(mi == 0 for mi in m) else 0
            cfg += 1
            checked += 1
            vlo = Lambda(m, Hlo)
            vhi = Lambda(m, Hhi)
            if vlo <= surv:
                nlo += 1
            elif monotone:
                fails_lo.append((hs, m, vlo, surv))
            if vhi >= surv:
                nhi += 1
            elif monotone:
                fails_hi.append((hs, m, vhi, surv))
        total_lo += nlo
        total_hi += nhi
    say(f"  {checked} (depth pattern, bad-count) configurations tested over")
    say("  1, 2 and 3 partition points, h_j in 0..3, m_i in 0..4.")
    say(f"  LOWER-sieve violations with h non-increasing: {len(fails_lo)}")
    say(f"  UPPER-sieve violations with h non-increasing: {len(fails_hi)}")
    if fails_lo:
        say(f"    first: {fails_lo[0]}")
    if fails_hi:
        say(f"    first: {fails_hi[0]}")
    assert not fails_lo, fails_lo[:3]
    assert not fails_hi, fails_hi[:3]
    say("")
    say("  RESULT: with the UPPER-TAIL (nested) constraint and h_j NON-INCREASING")
    say("  in j, every configuration tested satisfies")
    say("      Lambda^-(x) <= [x survives] <= Lambda^+(x),")
    say("  so the truncation the round-23 first pass could not find IS the one")
    say("  Tenenbaum describes, and the 36 counterexamples of j2_explicit.py")
    say("  section D are retired: they are counterexamples to the PER-BAND form")
    say("  only.  The depth-by-one difference (2h+1 vs 2h+2) is what makes the")
    say("  pair bracket rather than both overshoot.")

    say("")
    say("  Is the monotonicity of h_j needed?  Test the same family WITHOUT it:")
    viol = []
    for hs in patterns:
        k = len(hs) - 1
        if all(hs[j] >= hs[j + 1] for j in range(k)):
            continue
        Hlo = tuple(2 * h + 1 for h in hs)
        for m in product(range(0, 5), repeat=k + 1):
            surv = 1 if all(mi == 0 for mi in m) else 0
            if Lambda(m, Hlo) > surv:
                viol.append((hs, m, Lambda(m, Hlo)))
                break
    say(f"  non-monotone depth patterns with a LOWER-sieve violation: "
        f"{len(viol)} of {sum(1 for hs in patterns if not all(hs[j] >= hs[j+1] for j in range(len(hs)-1)))}")
    if viol:
        say(f"    example: h = {viol[0][0]}, m = {viol[0][1]}, "
            f"Lambda = {viol[0][2]} > 0")
    else:
        say("  NONE.  PRE-REGISTERED EXPECTATION REFUTED: I expected monotone h_j")
        say("  to be needed for validity and wrote that into this script before")
        say("  running the control.  It is not - THE UPPER-TAIL NESTING ALONE")
        say("  SUFFICES over the whole tested range.  Monotone h_j is a LEVEL-COST")
        say("  convenience (it is what makes sum_j alpha_{j-1} K_j converge), not a")
        say("  validity requirement.  Recorded as a refuted guess, not smoothed away.")
    assert not viol, viol[:3]

    say("")
    say("=" * 78)
    say("N3 - what this does and does NOT settle")
    say("=" * 78)
    say("  SETTLED: validity, at the level of the combinatorial identity, for the")
    say("  correct nested truncation - verified, not assumed.")
    say("  NOT SETTLED: the explicit MAIN-TERM estimate for that truncation, i.e.")
    say("  an explicit lower bound on sum_{d in D^-} mu(d) g(d) in terms of V(z).")
    say("  That is the remaining piece between this and an explicit polynomial")
    say("  bound at exponent ~8, and it is the piece Halberstam-Richert's own")
    say("  Memoire (Mem. S.M.F. 25 (1971) 97-106) is reported to carry - a lead")
    say("  this round could not verify against the actual text.  Tenenbaum sets")
    say("  the construction as Exercise 86 and proves nothing about it there.")
    say("  Until that is done, THEOREM 2E (exponent 19, via Friedlander-Iwaniec")
    say("  Theorem 7.7) is the explicit rung the note actually has.")

    with open("research/data/j2_nested.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_nested: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

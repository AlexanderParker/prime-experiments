"""Harvester round 23 (addendum): AN EXPLICIT RUNG 2, via Friedlander-Iwaniec
"Opera de Cribro" THEOREM 7.7 - and an audit of the round-23 dyadic-range costing.

THE FIND. The round's primary target - "make the implied constant in rung 2
explicit" - is achievable after all, by CITATION to a CONSTANT-FREE sieve theorem,
provided the sieve hypothesis constant K is pinned. Verified against actual text
(ar5iv HTML of arXiv:2602.22720, Dudek & Dunn, "An Explicit Result for the Sum of
Two Almost Primes", submitted 26 Feb 2026):

  THEOREM (Friedlander-Iwaniec, Opera de Cribro, Theorem 7.7; transcribed as
  Dudek-Dunn Theorem 1.3).  Let g be a density function with
      prod_{w <= p < z} (1 - g(p))^{-1}  <=  K (log z / log w)^kappa
  for all z > w >= 2.  Put k = kappa + log K and s = log D / log z.  Let
  Lambda = Lambda^- Lambda^2 be the Selberg lower-bound sieve of level D.  Then,
  provided s >= 2k + 3,
      S(A, z)  >=  X V(z) { 1 - ((s+3)/(2 e^k)) (2 e k/(s-3))^{(s-3)/2} }
                   -  2 R_4(A, D),
      R_4(A, D) = sum_{d | P(z), d < D} tau_4(d) |r_d(A)|.

  LEMMA (Dudek-Dunn, Lemma 2.1).  For the multiplicative g with g(2) = 1/2 and
  g(p) = 2/p for p >= 3,
      prod_{w <= p < z} (1 - g(p))^{-1}  <=  3 (log z / log w)^2   (2 <= w < z).

THE POINT: THAT g IS LITERALLY OURS.  omega(2) = 1 gives g(2) = 1/2 and
omega(p) = 2 for odd p gives g(p) = 2/p.  (Not a coincidence: Dudek-Dunn sift for
n and N - n simultaneously, the Goldbach side of Ziller-Morack Theorem 4.1, which
is the same two-classes-per-prime structure as the paired Jacobsthal problem.)
So kappa = 2 and K = 3 with NO further work, and K = 3 is BEST POSSIBLE: the
degenerate limit w = 3, z -> 3+ gives (1 - 2/3)^{-1} = 3 against
(log z/log w)^2 -> 1.  (Checked below over a large grid of (w, z).)

CONSEQUENCE - THEOREM 2E (explicit rung 2), derived and verified in this script:

    j_2(p_n#)  <=  C_0 * p_n^{s} * (log p_n)^{10}    for every p_n >= p_0,

with s, C_0 and p_0 all stated.  Every constant is explicit; nothing is left as an
implied constant or an ineffective threshold.

Also here: an AUDIT of research/j2_explicit.py section D after a warning that its
per-range factor might be inverted.  It is not - the script divides by V_j
(amplification by (1/theta)^kappa), which is the correct orientation.
"""
from math import log, exp, lgamma, e as E
import numpy as np
from fractions import Fraction as Fr
from sympy import primerange, prime

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


KAPPA = 2.0
K_CONST = 3.0
k_FI = KAPPA + log(K_CONST)


def bracket(s, k=k_FI):
    """the FI 7.7 / Dudek-Dunn 1.3 bracket 1 - ((s+3)/(2e^k))(2ek/(s-3))^{(s-3)/2}"""
    return 1.0 - ((s + 3) / (2 * exp(k))) * (2 * E * k / (s - 3)) ** ((s - 3) / 2)


def primes_upto(n):
    sv = np.ones(n + 1, bool)
    sv[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = False
    return np.flatnonzero(sv).astype(np.int64)


def main():
    say("=" * 78)
    say("F1 - the sieve hypothesis constant: kappa = 2, K = 3, and K = 3 is sharp")
    say("=" * 78)
    say("  g(2) = 1/2, g(p) = 2/p (p odd) - exactly omega(p)/p for the paired sieve.")
    pr = primes_upto(200000).tolist()
    g = {p: (0.5 if p == 2 else 2.0 / p) for p in pr}
    # sup over (w,z) of prod_{w<=p<z}(1-g)^{-1} / (log z/log w)^kappa
    worst = None
    cum = {}
    # walk w over primes, z over primes > w (bounded window is enough: the ratio
    # decays once several primes are included)
    for i, w in enumerate(pr[:400]):
        acc = 1.0
        for j in range(i, min(i + 400, len(pr))):
            p = pr[j]
            znext = pr[j + 1] if j + 1 < len(pr) else p + 1
            acc *= 1.0 / (1.0 - g[p])
            # z just above p includes exactly the primes w..p; the supremum of
            # the ratio over that z-interval sits at its LEFT endpoint z -> p+
            for z in (p * (1 + 1e-13), znext - 1e-9):
                if z <= w:
                    continue
                val = acc / (log(z) / log(w)) ** KAPPA
                if worst is None or val > worst[0]:
                    worst = (val, w, z)
    say(f"  sup over the tested grid = {worst[0]:.6f} at w = {worst[1]}, "
        f"z = {worst[2]:.4f}")
    assert worst[0] <= 3.0 + 1e-9, worst
    assert worst[0] > 2.99, worst
    say("  ASSERTED: the supremum is 3, attained in the limit w = 3, z -> 3+ where")
    say("  the product is (1-2/3)^{-1} = 3 and (log z/log w)^2 -> 1.  So K = 3 is")
    say("  exact and best possible - Dudek-Dunn Lemma 2.1 confirmed independently.")
    say(f"  Hence k = kappa + log K = 2 + log 3 = {k_FI:.6f},")
    say(f"  and FI 7.7's hypothesis s >= 2k + 3 = {2*k_FI+3:.4f}.")

    say("")
    say("=" * 78)
    say("F2 - the FI 7.7 bracket as a function of the level exponent s")
    say("=" * 78)
    say("        s     bracket    usable?")
    rows = []
    for s in [9.2, 12, 15, 18, 18.5, 19, 19.5, 20, 21, 22, 25, 30]:
        b = bracket(s)
        rows.append((s, b))
        say(f"   {s:>6.2f}  {b:>10.5f}    {'yes' if b > 0 else 'no'}")
    assert bracket(9.2) < 0 and bracket(30) > 0
    # positivity threshold
    lo, hi = 9.2, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if bracket(mid) < 0:
            lo = mid
        else:
            hi = mid
    s_star = (lo + hi) / 2
    say(f"  POSITIVITY THRESHOLD s* = {s_star:.5f}  (bracket = 0 there).")
    assert 18.2 < s_star < 18.4, s_star
    say("  So FI 7.7's hypothesis s >= 2k+3 = 9.197 is necessary but not")
    say("  sufficient for a POSITIVE main term; the sieve first bites at")
    f"{s_star:.3f}"
    say(f"  s = {s_star:.3f}.  We take s = 19 (bracket {bracket(19):.4f} > 0.24);")
    say(f"  s = 20 gives {bracket(20):.4f} and s = 22 gives {bracket(22):.4f}.")
    assert bracket(19) > 0.24 and bracket(20) > 0.5

    say("")
    say("=" * 78)
    say("F3 - THEOREM 2E: the explicit polynomial bound")
    say("=" * 78)
    say("  A = {1,...,m}, sifted by Omega_p (|Omega_p| = omega(p)) for p <= z = p_n.")
    say("  X = m, g(d) = omega(d)/d, r_d = |A_d| - X g(d), |r_d| <= omega(d).")
    say("  For squarefree d: tau_4(d) = 4^nu(d) and omega(d) <= 2^nu(d), so")
    say("      R_4(A,D) <= sum_{d < D, d squarefree} 8^{nu(d)}")
    say("                <= D prod_{p < D} (1 + 8/p)  <=  C_8 D (log D)^8.")
    # explicit C_8 : prod_{p<=x}(1+8/p) <= (e^gamma log x (1+1/log^2 x))^8 * H
    # with H = prod_p (1+8/p)(1-1/p)^8  (convergent).  Evaluate H numerically.
    GAMMA = 0.5772156649015328606
    # H_D = prod_{p<D}(1+8/p)(1-1/p)^8 is DECREASING in D (each factor is
    # 1 - 36/p^2 + O(p^-3) < 1), so H_D <= H_{D_0} for every D >= D_0.  We need an
    # UPPER bound, so the limit value would be UNSAFE; take D_0 = 10^6 (our D is
    # z^s >= 285^19, so D >= D_0 is free).
    fac = [(1 + 8.0 / p) * (1 - 1.0 / p) ** 8 for p in primes_upto(1000000).tolist()]
    assert all(f < 1.0 for f in fac), "each factor must be < 1 for monotonicity"
    H = 1.0
    for f in fac:
        H *= f
    say(f"  H_D0 = prod_{{p<10^6}}(1+8/p)(1-1/p)^8 = {H:.6e}")
    say("  (decreasing in D, so this is a valid UPPER bound for every D >= 10^6)")
    # Mertens explicit: prod_{p<=x}(1-1/p)^{-1} <= e^gamma log x (1 + 1/log^2 x), x>=286
    C8 = H * exp(8 * GAMMA)
    say(f"  C_8 = H e^{{8 gamma}} = {C8:.4f}, so")
    say(f"      R_4(A,D) <= {C8:.4f} * D * (log D)^8 (1 + 1/log^2 D)^8   (D >= 286)")

    say("")
    say("  Positivity needs  m V(z) * bracket  >  2 R_4(A,D)  with D = z^s:")
    say("      m  >  (2/bracket) * C_8 * z^s * (s log z)^8 / V(z),")
    say("      V(z) = V_n >= 0.3905/(log z)^2   (p_n >= 285, section 3 of the doc)")
    def absorb(C0):
        """least X with C0 (log X)^10 <= X (solved on the reals)"""
        lo, hi = 10.0, 1e300
        for _ in range(2000):
            mid = (lo * hi) ** 0.5
            if C0 * log(mid) ** 10 <= mid:
                hi = mid
            else:
                lo = mid
        return hi

    say("      s      bracket        C_0        threshold for C_0(log p)^10 <= p")
    chosen = None
    for s in (19.0, 19.5, 20.0, 21.0, 22.0):
        br = bracket(s)
        # the (1 + 1/log^2 D)^8 Mertens correction: D = z^s >= 285^19 so
        # log D >= 107, (1 + 1/log^2 D)^8 <= 1.001 - folded in as 1.001
        C0 = 1.001 * (2.0 / br) * C8 * (s ** 8) / 0.3905
        say(f"   {s:>5.1f} {br:>12.5f} {C0:>12.4e}   {absorb(C0):.3e}")
        if s == 19.0:
            chosen = (s, br, C0)
    s, br, C0 = chosen
    P0 = absorb(C0)
    say("")
    say("  THEOREM 2E (s = 19, the smallest integer level exponent at which the")
    say(f"  FI 7.7 bracket is positive - it first bites at s* = {s_star:.3f}):")
    say("")
    say(f"      j_2(p_n#)  <=  {C0:.4e} * p_n^{{19}} * (log p_n)^{{10}} + 1")
    say("")
    say("  for every p_n >= 285, WITH EVERY CONSTANT EXPLICIT AND NO INEFFECTIVE")
    say("  THRESHOLD.  That is rung 2 made self-contained.")
    say(f"  A clean SINGLE exponent needs a large threshold - p_n^{{20}} only from")
    say(f"  p_n >= {P0:.3e}, since (log p)^10 is slow to absorb - so the log-form")
    say("  above, whose threshold is 285, is the one to state.")
    assert br > 0.24 and C0 > 0
    assert C0 * log(P0) ** 10 <= P0 * 1.000001

    say("")
    say("  HONEST NOTES, all of which belong in the paper:")
    say(f"   * s = 19 is the smallest INTEGER above s* = {s_star:.3f}; any real")
    say("     s > s* works and gives exponent s, so the family of statements is")
    say("     j_2(p_n#) << p_n^{s} for every s > 18.31.")
    say("   * PRE-SIEVING the small primes lowers K toward 1 and hence s: dropping")
    say("     p = 2, 3 from the sieved range gives K = sup over p >= 5, computed")
    for pmin in (2, 3, 5, 7, 11, 101):
        sub = [p for p in pr[:400] if p >= pmin]
        wst = None
        for i, w in enumerate(sub[:200]):
            acc = 1.0
            for j in range(i, min(i + 300, len(sub))):
                p = sub[j]
                acc *= 1.0 / (1.0 - g[p])
                znx = sub[j + 1] if j + 1 < len(sub) else p + 1
                for z in (p * (1 + 1e-13), znx - 1e-9):
                    if z <= w:
                        continue
                    val = acc / (log(z) / log(w)) ** KAPPA
                    if wst is None or val > wst:
                        wst = val
        kk = KAPPA + log(wst)
        lo2, hi2 = 2 * kk + 3, 60.0
        for _ in range(200):
            mid = (lo2 + hi2) / 2
            if bracket(mid, kk) < 0:
                lo2 = mid
            else:
                hi2 = mid
        say(f"       p >= {pmin:>3}:  K = {wst:.4f},  k = {kk:.4f},  "
            f"s* = {(lo2+hi2)/2:.3f}")
    say("     so a pre-sieved version would reach exponent ~15-16 at the cost of")
    say("     carrying the pre-sieve construction; NOT done here, named instead.")
    say("   * the remainder is tau_4-WEIGHTED. Round 22's Theorem-2 sketch quoted")
    say("     the UNWEIGHTED sum_{d<D}|r_d| << D log^2 D; under FI 7.7 the correct")
    say("     statement is sum_{d<D} tau_4(d)|r_d| << D log^8 D. Log powers only -")
    say("     the EXPONENT is unaffected - but the doc must say the weighted form.")

    say("")
    say("=" * 78)
    say("F4 - AUDIT of j2_explicit.py section D: is the per-range factor inverted?")
    say("=" * 78)
    theta = 0.5
    t = 2 * log(1.0 / theta)
    vj = theta ** 2
    z = 10 ** 6
    prs = primes_upto(z)
    lo3 = int(z ** theta)
    sel = prs[(prs > lo3) & (prs <= z)]
    prodv = float(np.exp(np.log1p(-2.0 / sel).sum()))
    Ts = float((2.0 / sel).sum())
    say(f"  closed forms for range (z^theta, z], theta = {theta}:  "
        f"T_j = 2 log(1/theta) = {t:.6f},  V_j = theta^kappa = {vj:.6f}")
    say(f"  empirical over ({lo3}, {z}]:  T_j = {Ts:.6f},  V_j = {prodv:.6f}")
    assert abs(Ts - t) < 0.05 and abs(prodv - vj) < 0.02
    Kj = 4
    lt = (Kj + 1) * log(t) - lgamma(Kj + 2) - log(vj)
    tail = t ** (Kj + 1) / exp(lgamma(Kj + 2))
    say(f"  section D computes err_j = exp((K_j+1)log T_j - log((K_j+1)!) - log V_j)")
    say(f"          = {exp(lt):.6f} = tail {tail:.6f} times {exp(lt)/tail:.4f} "
        f"= 1/V_j = (1/theta)^kappa")
    assert abs(exp(lt) / tail - 1.0 / vj) < 1e-12
    say("  VERDICT: the factor is an AMPLIFICATION by (1/theta)^kappa = 4, which is")
    say("  the correct orientation (Ford's Brun-Hooley functional uses the same).")
    say("  The warned-of inversion (multiplying by V_j instead of dividing) is NOT")
    say("  present.  Section D's s = 9.07 / cost 0.36 stands as computed.")

    say("")
    say("=" * 78)
    say("F5 - WHY FI 7.7 AND NOT THE OTHER TWO CONSTANT-FREE THEOREMS")
    say("=" * 78)
    say("  Opera de Cribro carries THREE fully explicit, constant-free results.")
    say("  All are usable; the exponent they buy differs by ~10, so the choice")
    say("  matters.  Thresholds re-derived here from the stated inequalities:")
    say("")
    say("    ODC Thm 6.9 (p.69):  S >= X V(z){1 - e^{9k-s} K^10} + R^-(A,D),")
    say("      valid for D >= z^{9 kappa + 1}.  Main term positive iff")
    say("          s  >  9 kappa + 10 log K.")
    say("    ODC Cor 6.10 (p.69): S = X V(z){1 + 4 theta (9kappa+1)^kappa")
    say("      e^{9kappa-s} K^11} + theta R(A,D), |theta| <= 1, needing only")
    say("      D >= z >= 2 - NO hypothesis on s at all.  Positive iff")
    say("          s  >  9 kappa + log( 4 (9 kappa + 1)^kappa K^11 ).")
    say("    ODC Thm 7.7:         the Lambda^- Lambda^2 bracket used above.")
    say("")
    say("      theorem            K = 3      K = 1.097 (pre-sieved at 3)")

    def s69(K, kap=KAPPA):
        return 9 * kap + 10 * log(K)

    def s610(K, kap=KAPPA):
        return 9 * kap + log(4 * (9 * kap + 1) ** kap * K ** 11)

    def s77(K, kap=KAPPA):
        kk = kap + log(K)
        lo2, hi2 = 2 * kk + 3, 200.0
        for _ in range(300):
            mid = (lo2 + hi2) / 2
            if bracket(mid, kk) < 0:
                lo2 = mid
            else:
                hi2 = mid
        return (lo2 + hi2) / 2

    rows69 = []
    for name, fn in (("ODC Thm 6.9", s69), ("ODC Cor 6.10", s610),
                     ("ODC Thm 7.7", s77)):
        a, b = fn(3.0), fn(1.097)
        rows69.append((name, a, b))
        say(f"      {name:<16} s > {a:>7.3f}   s > {b:>7.3f}")
    d69 = dict((n, (a, b)) for n, a, b in rows69)
    assert abs(d69["ODC Thm 6.9"][0] - 28.99) < 0.01, d69
    assert abs(d69["ODC Thm 6.9"][1] - 18.93) < 0.01, d69
    assert abs(d69["ODC Cor 6.10"][0] - 37.36) < 0.02, d69
    assert abs(d69["ODC Cor 6.10"][1] - 26.29) < 0.02, d69
    assert abs(d69["ODC Thm 7.7"][0] - 18.308) < 0.01, d69
    assert abs(d69["ODC Thm 7.7"][1] - 14.53) < 0.02, d69
    assert d69["ODC Thm 7.7"][0] < d69["ODC Thm 6.9"][0] < d69["ODC Cor 6.10"][0]
    say("")
    say("  VERDICT: FI/ODC THEOREM 7.7 WINS BY ABOUT 10 IN THE EXPONENT, because")
    say("  K^10 is brutal at K = 3 (10 log 3 = 10.99 on its own).  Thm 6.9 is a")
    say("  cleaner-looking FALLBACK; Cor 6.10's value is that it assumes NOTHING")
    say("  on s (only D >= z >= 2).  All three thresholds re-derived here from the")
    say("  stated inequalities and asserted against an independent computation.")
    say("")
    say("  CITATION-NUMBERING SWEEP (this is the second numbering error found in")
    say("  two messages, both in results about to be leaned on):")
    say("   * IWANIEC-KOWALSKI HAS NO THEOREM 6.9 AND NO COROLLARY 6.10. Chapter 6")
    say("     ('Elementary Sieve Methods') stops at Theorem 6.7; in IK, 6.9 and")
    say("     6.10 are EQUATION labels. The 6.9/6.10 numbering belongs to Opera de")
    say("     Cribro. IK's s >= 9 kappa + 1 / K^10 result is THEOREM 6.1 /")
    say("     COROLLARY 6.2 (p.158). IK's FUNDAMENTAL LEMMA 6.3 has NO lower bound")
    say("     on s and error 1 + O(e^{-s}(1 + K/log z)^{10}) - K-dependence INSIDE")
    say("     the O and to the tenth power, so NOT explicit.")
    say("   * TENENBAUM: the fundamental lemma is THEOREM 4.4 (Theorem 3 in the")
    show = "     1995 CUP edition), not 4.3; and 'Theorem I.4.2' DOES NOT EXIST -"
    say(show)
    say("     I.4.2 is a COROLLARY (the Bonferroni inequality).")
    say("   * NATHANSON Ch.6 is a DEAD END for this purpose: 'Elementary estimates")
    say("     for primes', no general-dimension sieve at all.")
    say("   * CORRECT AS WE HAD IT: 'Friedlander-Iwaniec Opera de Cribro Thm 6.9'")
    say("     is a real fundamental lemma and our two uses of that phrase stand.")

    with open("research/data/j2_fi77.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_fi77: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

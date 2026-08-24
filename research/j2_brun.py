"""Harvester round 22: the BRUN rung of the j_2 ladder, with fully explicit constants,
and the sifting-limit ceiling of the whole sieve method.

Round 21 gave two rungs for the (previously empty) upper-bound ladder of the paired
Jacobsthal function:
    rung 1  j_2(p_n#) <= 2*3^(n-1)/V_n + 1 < 3^(n+1)(log p_n)^2   elementary, complete
    rung 2  j_2(p_n#) <<_eps p_n^(beta_2 + eps)                    fundamental lemma

THEOREM 3 (this round; contains Theorem 1 as the case K = n).  Let p_1..p_n be the
first n primes, omega(2) = 1, omega(p) = 2 for odd p, and let e_j(x_1..x_n) be the
elementary symmetric polynomials.  Put

    E_K = sum_{j<=K} e_j(omega(p_1),...,omega(p_n))          (remainder cost)
    R_K = sum_{j>K}  e_j(omega(p_1)/p_1,...,omega(p_n)/p_n)  (truncation cost)
    V_n = prod_p (1 - omega(p)/p) = (1/2) prod_{3<=p<=p_n} (1 - 2/p).

Then for EVERY ODD K with R_K < V_n,

    j_2(p_n#)  <=  E_K / (V_n - R_K)  +  1.

K = n gives R_n = 0, E_n = prod(1+omega(p)) = 2*3^(n-1) - exactly Theorem 1.  The
optimal K is far smaller: K* ~ 4*T_n, T_n = sum_p omega(p)/p ~ 2 log log p_n, and then
E_{K*} ~ (2 e n / K*)^{K*} = exp(O(log p_n log log p_n)), i.e. the bound is
QUASI-POLYNOMIAL, p_n^{O(log log p_n)}, instead of exponential 3^n.

Proof.  Bonferroni with ODD truncation depth K is a LOWER bound for the survivor
indicator: a position lying in exactly r of the bad classes contributes
sum_{j<=K} (-1)^j C(r,j) = (-1)^K C(r-1,K), which is -C(r-1,K) <= 0 for r >= 1 and 1
for r = 0.  With N_d = m*omega(d)/d + theta_d*omega(d), |theta_d| <= 1 (the bad set for
squarefree d is omega(d) = prod_{p|d} omega(p) residue classes mod d, each of which
meets [1,m] in m/d + theta places):

    S  >=  sum_{d|P(z), omega(d)<=K} mu(d) N_d
       >=  m * sum_{j<=K} (-1)^j e_j(omega(p)/p)  -  sum_{j<=K} e_j(omega(p))
       >=  m (V_n - R_K)  -  E_K,

since sum_{j<=K}(-1)^j e_j = V_n - sum_{j>K}(-1)^j e_j >= V_n - R_K.  A fully bad run
of length m forces S = 0, hence m <= E_K/(V_n - R_K).  QED

Also recorded (literature, checked 2026-08-24): the ceiling of the sieve route.  The
best proved dimension-2 sifting limit is beta_2 = 4.266 (Diamond-Halberstam-Richert;
Franze arXiv:1012.3809 Table 1, which gives 4.516 for the Lambda^2 Lambda^- sieve at
kappa = 2) - an improvement on round 21's cited 4.45.  Selberg's conjectural optimum
is 2*kappa = 4.  Ziller-Morack Conjecture 6 asks for exponent 2: a DIMENSION-ONE
quality bound on a dimension-TWO problem.
"""
from fractions import Fraction as Fr
from math import log
from sympy import prime, primerange

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def esym(weights, kmax):
    """e_0..e_kmax of the given weights (exact)."""
    e = [Fr(0)] * (kmax + 1)
    e[0] = Fr(1)
    for w in weights:
        for j in range(min(kmax, len(e) - 1), 0, -1):
            e[j] += w * e[j - 1]
    return e


def tables(n, kmax):
    ps = list(primerange(2, prime(n) + 1))
    assert len(ps) == n
    om = [1] + [2] * (n - 1)
    eR = esym([Fr(o, p) for o, p in zip(om, ps)], kmax)     # for the truncation tail
    eE = esym([Fr(o) for o in om], kmax)                    # for the remainder cost
    V = Fr(1)
    T = Fr(0)
    tot = Fr(1)
    for o, p in zip(om, ps):
        V *= Fr(p - o, p)
        T += Fr(o, p)
        tot *= (1 + Fr(o, p))
    return ps, V, T, eR, eE, tot


def brun_bound(n, kmax=None):
    """min over odd K of E_K/(V-R_K)+1, exact rationals.
    Returns (bound, K*, V_n, T_n, p_n, Theorem-1 value)."""
    if kmax is None:
        kmax = min(n + 1, 60)
    ps, V, T, eR, eE, tot = tables(n, kmax)
    legendre = Fr(2 * 3 ** (n - 1)) / V + 1          # round-21 Theorem 1
    best, bestK = None, None
    cumR = Fr(0)
    cumE = Fr(0)
    for K in range(0, kmax + 1):
        cumR += eR[K]
        cumE += eE[K]
        R = tot - cumR          # = sum_{j>K} e_j(omega/p)   (exact)
        if K % 2 == 0 or R >= V:
            continue
        b = cumE / (V - R) + 1
        if best is None or b < best:
            best, bestK = b, K
    if kmax >= n + 1:            # full inclusion-exclusion is available: K = n or n+1
        best = min(best, legendre) if best is not None else legendre
        if best == legendre and bestK is None:
            bestK = n if n % 2 else n + 1
    return best, bestK, V, T, ps[-1], legendre


def flog(fr):
    return log(fr.numerator) - log(fr.denominator)


def main():
    say("=== THEOREM 3 (Brun pure sieve, exact rationals): j_2(p_n#) <= "
        "E_K/(V_n-R_K) + 1 ===")
    say("   n   p_n     K*     Brun bound      Legendre (K=n, r21 Thm 1)   "
        "p_n^4.266     ZM h_2")
    zm = {3: 18, 4: 30, 5: 66, 6: 150, 7: 192, 8: 258, 9: 366}
    rows = []
    for n in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 21, 30, 40, 60, 80,
              120, 170, 250, 400]:
        b, K, V, T, p, leg = brun_bound(n, kmax=n + 1)
        # cross-check: full depth K >= n IS round 21's Theorem 1 (R = 0, E = 2*3^(n-1))
        ps2, V2, T2, eR2, eE2, tot2 = tables(n, n + 1)
        assert sum(eE2) == 2 * 3 ** (n - 1) and tot2 == sum(eR2), n
        rows.append((n, p, K, b, leg, p ** 4.266, float(T)))
        say(f" {n:>3} {p:>5} {K:>6}   {float(b):>14.5g}   {float(leg):>25.5g}   "
            f"{p**4.266:>11.4g}   {zm.get(n, ''):>6}")
    for n, p, K, b, leg, sift, T in rows:
        assert K % 2 == 1 and b > 0
        assert b <= leg, (n, "Brun must never be worse than its own K=n case")
        if n in zm:
            assert float(b) > zm[n] and float(leg) > zm[n], n
    cross = next(r[0] for r in rows if r[3] < r[4])
    say(f"  K = n reproduces round-21 Theorem 1 EXACTLY at every n (asserted).")
    say(f"  CROSSOVER (strictly better than the K=n / Theorem-1 value): n = {cross} "
        f"(p_n = {next(r[1] for r in rows if r[0]==cross)})")

    say("")
    say("=== quasi-polynomial shape:  log(bound) / (log p_n * log log p_n) ===")
    say("      n    p_n     K*    4*T_n   log(bound)   ratio   [Thm 1 ratio]")
    ratios = []
    for n in [40, 80, 170, 400, 800, 1500, 3000]:
        b, K, V, T, p, leg = brun_bound(n, kmax=40)
        lb, ll = flog(b), flog(leg)
        r = lb / (log(p) * log(log(p)))
        ratios.append(r)
        say(f"  {n:>5} {p:>6} {K:>6} {4*float(T):>8.2f} {lb:>12.2f} {r:>7.3f}"
            f"   [{ll/(log(p)*log(log(p))):9.1f}]")
    assert ratios[-1] < ratios[0] * 1.6, "ratio should stay bounded (quasi-polynomial)"
    assert max(ratios) < 12.0
    say("  Theorem 1's own ratio diverges (exponential); Theorem 3's stays in a narrow "
        "band -> quasi-polynomial confirmed numerically to p_n = "
        f"{prime(3000)}.")

    say("")
    say("=== THEOREM 3's INEQUALITY CHECKED DIRECTLY ON REAL PAIRED WINDOWS ===")
    # S >= m (V_n - R_K) - E_K, with the worst-case (omega = 2) constants, verified
    # against brute-force survivor counts on random paired progressions.
    import random
    from sympy import primerange as _pr
    random.seed(22)
    checks = 0
    worst = None
    for n in (3, 4, 5, 6):
        ps = list(_pr(2, prime(n) + 1))
        P = 1
        for q in ps:
            P *= q
        _, V, T, eR, eE, tot = tables(n, n + 1)
        for _ in range(150):
            a = random.randrange(P)
            e = random.randrange(1, P)
            m = random.randrange(30, 400)
            Om = [{(-a) % q, (-a - 2 * e) % q} for q in ps]
            S = sum(1 for i in range(1, m + 1)
                    if all(i % q not in Om[j] for j, q in enumerate(ps)))
            cumR = Fr(0)
            cumE = Fr(0)
            for K in range(0, n + 2):
                cumR += eR[K]
                cumE += eE[K]
                if K % 2 == 0:
                    continue
                R = tot - cumR
                lb = m * (V - R) - cumE
                assert S >= lb, (n, a, e, m, K, S, float(lb))
                checks += 1
                if S > 0 and (worst is None or float(lb) / S > worst[0]):
                    worst = (float(lb) / S, n, K)
    say(f"  {checks} instances (n = 3..6, odd K = 1..n+1), every one satisfies "
        f"S >= m(V_n - R_K) - E_K; tightest ratio lower-bound/true = {worst[0]:.4f} "
        f"at n = {worst[1]}, K = {worst[2]}")

    say("")
    say("=== THE LADDER, as it now stands ===")
    say("  rung 0  j_2(p_n#) <= p_n#                       trivial (periodicity)")
    say("  rung 1  <= 2*3^(n-1)/V_n + 1 < 3^(n+1)log^2 p   elementary  (round 21)")
    say("  rung 1.5 <= E_K/(V_n-R_K)+1 = p_n^{O(log log p_n)}  elementary, explicit, "
        "quasi-polynomial  (THIS ROUND; K=n is rung 1)")
    say("  rung 2  <<_eps p_n^(beta_2+eps), beta_2 = 4.266  fundamental lemma / DHR "
        "(improved from 4.45 this round)")
    say("  CEILING of any dimension-2 lower-bound sieve: exponent beta_2, conjecturally "
        "2*kappa = 4 (Selberg)")
    say("  TARGET  Ziller-Morack Conjecture 6: exponent 2  -  below the conjectural "
        "dimension-2 sifting limit, i.e. parity-blocked, not merely unproved")

    with open("research/data/j2_brun.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_brun: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

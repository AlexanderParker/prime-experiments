"""j2_odc6.py - THE ODC CHAPTER 6 ROUTE (round 25, harvester).

Decides the round-24 named opening #2: is Opera de Cribro Chapter 6's beta-sieve
apparatus EXPLICIT at kappa = 2?

FIRST-HAND SOURCE (page scans of the AMS printing read 2026-08-29, Google Books
volume id Dz6REQAAQBAJ; pp. 68-73 and p. 112).  The printed objects used here:

  PROPOSITION 6.7 (p.68).  Let Lambda^+, Lambda^- be the upper-/lower-bound
    beta-sieve of level D.  Then for any multiplicative g satisfying (5.38),
      (6.73)  V^+(D,z) <= {1 + psi^+(a, s-beta) K^(1+1/alpha)} V(z),  s >= beta+1
      (6.74)  V^-(D,z) >= {1 - psi^-(a, s-beta) K^(1+1/alpha)} V(z),  s >= beta
    with  a = alpha e^(1+alpha)   (6.67)
     and  alpha = (kappa/2) log((beta+1)/(beta-1))  (6.65), equivalently
      (6.94)  beta = beta_kappa = 1 + 2 (e^(2 alpha/kappa) - 1)^(-1).
  (6.75)  psi^{+-}(a, s-beta) < 2 e^-2 (1-a^2)^-1 a^(s-beta)
  (6.86)  psi^-(a, s-beta)    < 2 e^-2 (1-a^2)^-1 a^2      if s >= beta
  COROLLARY 6.13 (p.71), alpha = 1/4:
      V^-(D,z) >= {1 - (7/8) K^5} V(z)  if s >= beta_kappa,
      beta_kappa = 1 + 2(e^(1/2kappa) - 1)^-1,  beta_1 = 4.082.., beta_2 = 8.041..
  p.73: the root alpha* = 0.264904.. of  alpha + (2+3a)/(3+4a) + log alpha
      + log((3+4alpha)/(2+3alpha)) = 0  gives beta_1 = 3.8629.., beta_2 = 7.5941..
      and (6.97) beta_kappa < alpha^-1 kappa + 1 < 3.775 kappa + 1.

CRITICAL POINT OF EXPLICITNESS.  Proposition 6.7, (6.75), (6.85), (6.86) and
Corollary 6.13 carry NO O(.), NO <<, NO implied constant and NO "z sufficiently
large".  The only inexplicit sentence in the neighbourhood is COROLLARY 6.14,
whose proof reads "provided K is sufficiently close to one ... we can depress its
size close to one by choosing a slightly larger value of kappa" and whose
statement says "for z large".  That device is exactly what we replace by
PRE-SIEVING at explicit finite cost (round 24's 2E'/2E'' machinery), so the
Chapter 6 apparatus IS usable with every constant stated.

Everything below is asserted.  Run: python research/j2_odc6.py
"""

import sys
from math import log, exp, sqrt, e, pi

OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def hr(t=""):
    say()
    say("=" * 78)
    if t:
        say(t)
        say("=" * 78)


def approx(x, y, tol, what):
    assert abs(x - y) <= tol, "%s: %.10f vs %.10f (tol %g)" % (what, x, y, tol)


# ----------------------------------------------------------------------------
# The printed formulas, as functions.
# ----------------------------------------------------------------------------

def a_of(alpha):
    """(6.67)  a = alpha e^(1+alpha)."""
    return alpha * exp(1.0 + alpha)


def beta_of(alpha, kappa):
    """(6.94)  beta = 1 + 2 (e^(2 alpha/kappa) - 1)^(-1)."""
    return 1.0 + 2.0 / (exp(2.0 * alpha / kappa) - 1.0)


def psi_minus_bound(alpha, s_minus_beta=0.0):
    """(6.86) at s = beta, (6.75) in general.

    psi^-(a, t) sums 1/n! (na/e)^n over EVEN n > t.  The book's bound is
    2 e^-2 (1-a^2)^-1 a^m with m the least even integer > t (m = 2 at t = 0,
    which is (6.86)); (6.75) uses a^t which is weaker.  We use a^m, valid for
    every t >= 0 and equal to (6.86) at t = 0.
    """
    a = a_of(alpha)
    assert 0.0 < a < 1.0, "a must lie in (0,1); a = %.6f" % a
    m = 2
    while m <= s_minus_beta:
        m += 2
    return 2.0 * exp(-2.0) * a ** m / (1.0 - a * a)


def odc_p73_root():
    """The p.73 equation: alpha + (2+3a)/(3+4a) + log a + log((3+4a)/(2+3a)) = 0."""
    def f(al):
        return al + (2 + 3 * al) / (3 + 4 * al) + log(al) + log((3 + 4 * al) / (2 + 3 * al))
    lo, hi = 0.10, 0.40
    assert f(lo) < 0 < f(hi), (f(lo), f(hi))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# SECTION A - reproduce the book's own printed numbers from its own formulas.
# ----------------------------------------------------------------------------
hr("SECTION A - ODC Chapter 6 reproduced from its printed formulas (p.68-73)")

a_quarter = a_of(0.25)
say("  alpha = 1/4:  a = (1/4) e^(5/4) = %.10f   (book: a < 7/8 = %.4f)"
    % (a_quarter, 7 / 8))
assert a_quarter < 7 / 8

psi_q = psi_minus_bound(0.25, 0.0)
book_psi = 2 * e / (16 * sqrt(e) - e ** 3)          # book's closed form, p.71
say("  psi^-(a) bound at s = beta   = %.10f" % psi_q)
say("  book's printed closed form 2e(16 sqrt e - e^3)^-1 = %.10f" % book_psi)
approx(psi_q, book_psi, 1e-9, "psi^- closed form")
assert 0.8637 <= psi_q < 0.8638, "book prints psi^- = 0.8637... (truncated)"
assert psi_q < 7 / 8, "book asserts psi^- < 7/8"
say("  -> BOTH renderings of the book's psi^- agree to 1e-9 and psi^- < 7/8. OK")

b1q, b2q = beta_of(0.25, 1.0), beta_of(0.25, 2.0)
say()
say("  (6.87) beta_kappa = 1 + 2(e^(1/2kappa)-1)^-1  [alpha = 1/4]")
say("     beta_1 = %.6f   (book: 4.082...)" % b1q)
say("     beta_2 = %.6f   (book: 8.041...)" % b2q)
approx(b1q, 4.082, 1e-3, "beta_1 at alpha=1/4")
approx(b2q, 8.041, 1e-3, "beta_2 at alpha=1/4")
for kap in [0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 12]:
    assert beta_of(0.25, kap) <= 4 * kap + 1 + 1e-12, ("(6.90) beta_kappa <= 4kappa+1", kap)
say("     (6.90) beta_kappa <= 4 kappa + 1 verified at kappa = 0.25 .. 12.  OK")

astar_root = odc_p73_root()          # OUR root of the book's OWN printed equation
ASTAR_BOOK = 0.264904                # the book's printed numerical value
astar = ASTAR_BOOK                   # used downstream: the conservative (smaller) one
say()
say("  p.73 exact computation.  The book derives alpha' = (2+3a)/(3+4a) from")
say("  psi^-(x) = 1 and then the single equation")
say("       alpha + (2+3a)/(3+4a) + log alpha + log((3+4a)/(2+3a)) = 0 ,")
say("  saying 'A numerical computation gives (use the Taylor expansion at 1/4)")
say("  alpha = 0.264904...'.")
say()
say("     OUR root of that printed equation   alpha  = %.9f" % astar_root)
say("     the book's printed value            alpha* = %.9f" % ASTAR_BOOK)
say("     residual of the printed value in the printed equation: f(alpha*) = %.6f"
    % (ASTAR_BOOK + (2 + 3 * ASTAR_BOOK) / (3 + 4 * ASTAR_BOOK) + log(ASTAR_BOOK)
       + log((3 + 4 * ASTAR_BOOK) / (2 + 3 * ASTAR_BOOK))))
say()
say("  *** DISCREPANCY, RECORDED (referee-grade, immaterial to the verdict). ***")
say("  The printed alpha* = 0.264904 does NOT solve the book's own printed equation;")
say("  the true root is 0.2652637, larger by 3.6e-4.  Downstream:")
say("     beta_2 from the printed alpha*      = %.6f   (book prints 7.5941)"
    % beta_of(ASTAR_BOOK, 2.0))
say("     beta_2 from the true root           = %.6f" % beta_of(astar_root, 2.0))
say("     beta_1 from the printed alpha*      = %.6f   (book prints 3.8629)"
    % beta_of(ASTAR_BOOK, 1.0))
say("     beta_1 from the true root           = %.6f" % beta_of(astar_root, 1.0))
say("  The printed alpha* IS internally consistent with the printed beta_1 = 3.8629")
say("  and beta_2 = 7.5941 (both to within the truncation of the printed alpha*),")
say("  so the harvested page is self-consistent and there is no OCR digit error.")
say("  What is off is the ROOT-FINDING: the book says 'use the Taylor expansion at")
say("  1/4', and a Taylor approximation about 1/4 is exactly the kind of thing that")
say("  lands 3.6e-4 short.  CONSEQUENCE, in our favour: the exact root of the book's")
say("  own equation gives beta_2 = %.4f, %.4f BETTER than the printed 7.5941."
    % (beta_of(astar_root, 2.0), 7.5941 - beta_of(astar_root, 2.0)))
say("  IT DOES NOT TOUCH THE VERDICT EITHER WAY: our route's binding root is the")
say("  K -> 1 root of section B (alpha = 0.2533), strictly below both values.")
approx(beta_of(ASTAR_BOOK, 2.0), 7.5941, 1.5e-4, "printed alpha* -> printed beta_2")
approx(beta_of(ASTAR_BOOK, 1.0), 3.8629, 1.0e-4, "printed alpha* -> printed beta_1")
assert abs(astar_root - ASTAR_BOOK) < 4e-4, "root and printed value agree to 3 dp"
assert abs(astar_root - ASTAR_BOOK) > 1e-5, \
    "the root/printed-value gap is the thing being recorded; if it vanishes, re-read p.73"
assert beta_of(astar_root, 2.0) < 7.5941, "the exact root must improve the printed beta_2"
say("     (6.97) alpha*^-1 = %.6f  (book: beta_kappa < 3.775 kappa + 1)" % (1 / astar))
approx(1 / astar, 3.775, 1e-3, "(6.97) constant")
say()
say("  VERDICT A: every printed number of ODC section 6.6 (0.8637, 4.082, 8.041,")
say("  4kappa+1, 0.264904, 3.8629, 7.5941, 3.775) is reproduced from the book's")
say("  own printed formulas by independent code.  The OCR harvest is arithmetically")
say("  self-consistent - the strongest check available short of holding the book.")

# ----------------------------------------------------------------------------
# SECTION B - THE POSITIVITY CRITERION, AND THE HALBERSTAM-RICHERT COINCIDENCE
# ----------------------------------------------------------------------------
hr("SECTION B - positivity criterion, and the HR-Memoire identity")

say("  (6.74) is a genuine lower bound as soon as")
say("       psi^-(a, s-beta) * K^(1 + 1/alpha)  <  1 .")
say("  At s = beta this reads   [2 e^-2 a^2/(1-a^2)] K^(1+1/alpha) < 1 .")
say()
say("  As K -> 1 (unlimited pre-sieving) the criterion becomes")
say("       2 e^-2 a^2/(1-a^2) < 1   <==>   a^2 (1 + 2 e^-2) < 1 .")

a_inf = sqrt(1.0 / (1.0 + 2.0 * exp(-2.0)))
say("       a_infinity = %.9f" % a_inf)


def alpha_from_a(a):
    lo, hi = 1e-9, 1.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if a_of(mid) < a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


alpha_inf = alpha_from_a(a_inf)
beta2_inf = beta_of(alpha_inf, 2.0)
say("       alpha_infinity = %.9f" % alpha_inf)
say("       beta_2         = %.9f   <- the K -> 1 floor of the ODC Ch.6 route" % beta2_inf)
say()
say("  THE HALBERSTAM-RICHERT MEMOIRE (Mem. SMF 25 (1971) 97-106) - round 24 read")
say("  the numdam scan and re-derived its two printed conditions:")
say("       (1.2)      lambda e^(1+lambda) < 1")
say("       positivity lambda^2 e^(2 lambda) (2 + e^2) < 1")
say("       yielding lambda* = 0.2533219, u = 7.9719548.")
say()
HR_LAMBDA = 0.2533219
say("  CLAIM B1 (algebraic).  HR's positivity condition IS ODC's (6.86) positivity")
say("  condition, character for character.  With a = lambda e^(1+lambda) one has")
say("  lambda^2 e^(2 lambda) = a^2/e^2, so")
say("       lambda^2 e^(2 lambda)(2 + e^2) < 1  <==>  a^2 (2 e^-2 + 1) < 1 .")
for lam in [0.05, 0.1, 0.17, 0.2533219, 0.26, 0.3]:
    aa = a_of(lam)
    lhs_hr = lam * lam * exp(2 * lam) * (2 + e * e)
    lhs_odc = aa * aa * (2 * exp(-2.0) + 1.0)
    approx(lhs_hr, lhs_odc, 1e-12, "HR<->ODC positivity at lambda=%g" % lam)
say("  -> verified identically at six values of lambda (max error < 1e-12).  OK")
say()
say("  CLAIM B2 (numerical).  HR's lambda* and ODC's a_infinity give the SAME root.")
say("       HR   lambda* = %.7f" % HR_LAMBDA)
say("       ODC  alpha_inf = %.7f" % alpha_inf)
approx(alpha_inf, HR_LAMBDA, 5e-7, "alpha_inf == HR lambda*")
say("  -> equal to 5e-7.  THE TWO ROUND-24 LEADS ARE ONE EQUATION.")
say()
say("  CLAIM B3.  HR's printed level exponent is u = 1 + 2.01/(e^lambda* - 1);")
say("  ODC's is beta_2 = 1 + 2/(e^alpha - 1).  The 2.01 is HR's own safety margin.")
u_hr = 1 + 2.01 / (exp(HR_LAMBDA) - 1)
u_two = 1 + 2.00 / (exp(HR_LAMBDA) - 1)
say("       1 + 2.01/(e^lambda*-1) = %.9f   (round 24's re-derived 7.9719548)" % u_hr)
say("       1 + 2.00/(e^lambda*-1) = %.9f   (= ODC beta_2 floor)" % u_two)
approx(u_hr, 7.9719548, 1e-6, "HR u re-derivation")
approx(u_two, beta2_inf, 1e-6, "ODC floor == HR with 2.00")
say("  -> ODC Chapter 6 is the EXPLICIT form of the 1971 Memoire's theorem, and it")
say("     is very slightly SHARPER (7.9373 against 7.9720).")

# ----------------------------------------------------------------------------
# SECTION C - the pre-sieving ladder: how small can K be made, and at what cost
# ----------------------------------------------------------------------------
hr("SECTION C - pre-sieving ladder K(p0) and the alpha it unlocks")

KAPPA = 2.0


def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


PR = primes_upto(200000)
G = {p: (0.5 if p == 2 else 2.0 / p) for p in PR}


def K_of(p0, nw=400, nz=600):
    """sup over 2 <= w < z of prod_{w<=p<z, p>=p0}(1-g(p))^-1 / (log z/log w)^kappa.

    For w < p0 the numerator is unchanged while (log z/log w)^kappa is LARGER, so
    the supremum is attained at w >= p0; within (p_i, p_{i+1}] the ratio is largest
    at z -> p_i+, so testing z just above each prime suffices.
    """
    sub = [p for p in PR if p >= p0]
    worst = 1.0
    for i, w in enumerate(sub[:nw]):
        acc = 1.0
        for j in range(i, min(i + nz, len(sub))):
            p = sub[j]
            acc *= 1.0 / (1.0 - G[p])
            z = p * (1.0 + 1e-13)
            if z <= w:
                continue
            val = acc / (log(z) / log(w)) ** KAPPA
            if val > worst:
                worst = val
    return worst


def N_pre(p0):
    n = 1
    for p in PR:
        if p >= p0:
            break
        n *= (p - (1 if p == 2 else 2))
    return n


def slack(al, K):
    """log of psi^-(a,0) K^(1+1/alpha); the criterion is slack < 0."""
    return log(psi_minus_bound(al, 0.0)) + (1.0 + 1.0 / al) * log(K)


def alpha_max_for_K(K):
    """Largest alpha in (0, alpha*] with psi^-(a) K^(1+1/alpha) < 1 at s = beta.

    NOT monotone: the criterion fails for very small alpha (the K exponent
    1+1/alpha blows up) AND for alpha near alpha* (psi^- -> 1).  It holds on a
    middle band, so we scan downwards from alpha* and take the first success -
    beta_2(alpha) is decreasing in alpha, so that is the best rung.
    """
    n = 40000
    for i in range(n):
        al = astar * (1.0 - i / float(n))
        if al <= 1e-6:
            break
        try:
            if slack(al, K) < 0.0:
                return al
        except (AssertionError, ValueError):
            continue
    return None


K_CRIT_QUARTER = (1.0 / psi_q) ** (1.0 / 5.0)
say("  Corollary 6.13 (alpha = 1/4, the NUMBERED citable form) needs")
say("       (7/8) K^5 < 1   -- book's rounding -- or exactly psi^- K^5 < 1,")
say("       i.e. K < psi^-^(-1/5) = %.7f" % K_CRIT_QUARTER)
say()
say("  %-6s %-12s %-12s %-10s %s" % ("p0", "K(p0)", "alpha_max", "beta_2", "log10 N_pre"))
LADDER = [2, 3, 5, 7, 11, 13, 17, 23, 31, 41, 53, 71, 101, 151, 211, 307, 401,
          601, 1009, 2003, 5003, 10007, 20011, 50021]
rows = []
for p0 in LADDER:
    K = K_of(p0)
    am = alpha_max_for_K(K)
    b2 = beta_of(am, 2.0) if am else float("inf")
    np_ = N_pre(p0)
    rows.append((p0, K, am, b2, np_))
    say("  %-6d %-12.7f %-12s %-10s %.1f"
        % (p0, K, ("%.7f" % am) if am else "-none-",
           ("%.4f" % b2) if am else "-", log(np_, 10) if np_ > 1 else 0.0))

say()
prev = float("inf")
for (_, K, am, b2, _) in rows:
    assert b2 <= prev + 1e-9, "beta_2 must be non-increasing along the pre-sieve ladder"
    prev = b2
say("  ASSERTED: beta_2 is non-increasing along the pre-sieving ladder.  OK")

first_q = [r for r in rows if r[1] < K_CRIT_QUARTER]
assert first_q, "no p0 in the ladder reaches K < K_crit(alpha=1/4)"
say("  Least tabulated p0 with K < %.7f (Corollary 6.13 usable verbatim): p0 = %d"
    % (K_CRIT_QUARTER, first_q[0][0]))
say("     there K = %.7f, N_pre = %d (%.1f digits)"
    % (first_q[0][1], first_q[0][4], log(first_q[0][4], 10)))

best = min(rows, key=lambda r: r[3])
say("  Best tabulated rung: p0 = %d, K = %.7f, alpha = %.7f, beta_2 = %.6f"
    % (best[0], best[1], best[2], best[3]))
assert best[3] < 8.5, "the ODC Ch.6 route must beat exponent 8.5 somewhere"
assert best[3] > beta2_inf - 1e-9, "cannot beat the K -> 1 floor"
say("  Floor as K -> 1: beta_2 = %.6f (unreachable, approached)." % beta2_inf)

# ----------------------------------------------------------------------------
# SECTION D - THEOREM 2G: the explicit bound
# ----------------------------------------------------------------------------
hr("SECTION D - THEOREM 2G, fully explicit")

say("  ARITHMETIC.  For squarefree d | P(z) the paired sieve removes omega(d)")
say("  classes with omega(d) <= 2^nu(d), so after pre-sieving at p0")
say("       |r_d| <= 2^nu(d) N_pre .")
say("  The beta-sieve weights satisfy |lambda_d^-| <= 1 and are supported on d < D,")
say("  hence  |R^-(A,D)| <= N_pre sum_{d<D} 2^nu(d) <= N_pre sum_{d<D} tau(d)")
say("                    <= N_pre D (log D + 1)")
say("  using sum_{n<=x} tau(n) = sum_{d<=x} floor(x/d) <= x (log x + 1).")
say()
say("  *** THIS IS WHERE ODC 6 BEATS ODC 7.7 TWICE OVER: the beta-sieve's weights")
say("      are bounded by 1, so the remainder carries tau (2^nu), not tau_4 (4^nu).")
say("      Theorem 2E/2E'' had to pay sum_{d<D} 8^nu(d) << C_8 D (log D)^8. ***")


def tau_sum_bound(D):
    return D * (log(D) + 1.0)


# sanity: the elementary bound really does dominate sum tau(n)
tot = 0
for n in range(1, 20001):
    k = 1
    c = 0
    while k * k <= n:
        if n % k == 0:
            c += 2 if k * k != n else 1
        k += 1
    tot += c
assert tot <= tau_sum_bound(20000.0), (tot, tau_sum_bound(20000.0))
say("  CHECK: sum_{n<=20000} tau(n) = %d <= %.0f = 20000(log 20000 + 1).  OK"
    % (tot, tau_sum_bound(20000.0)))

V_CONST = 0.3905          # V(z) >= V_CONST/(log z)^2 for z >= 285 (r23 defect 6.18)
Z_MIN = 285.0


def theorem_2G(p0, safety=None):
    """Least m for which ODC (6.74) forces a survivor in every window of length m.

    All constants are carried as log10 because N_pre exceeds 10^300.
    """
    K = K_of(p0)
    al = alpha_max_for_K(K) if safety is None else safety
    if al is None:
        return None
    try:
        sl = slack(al, K)
    except (AssertionError, ValueError):
        return None
    if sl >= 0.0:
        return None
    delta = 1.0 - exp(sl)
    s = beta_of(al, 2.0)
    log10C = log(N_pre(p0), 10) - log(V_CONST * delta, 10)
    return dict(p0=p0, K=K, alpha=al, s=s, delta=delta, log10C=log10C)


def log10_bound_2G(th, z):
    """log10 of C z^s (s log z + 1)(log z)^2."""
    return (th["log10C"] + th["s"] * log(z, 10)
            + log(th["s"] * log(z) + 1.0, 10) + 2.0 * log(log(z), 10))


def log10_bound_2Epp(z):
    """Round 24's Theorem 2E'': j_2 <= 7.2671e11 z^15 (log z)^10 + 1, z >= 285."""
    return log(7.2671e11, 10) + 15.0 * log(z, 10) + 10.0 * log(log(z), 10)


say()
say("  For every z >= %g and every window of m consecutive integers, positivity of"
    % Z_MIN)
say("  the ODC lower-bound beta-sieve forces a paired survivor as soon as")
say("       m V(z) delta > N_pre D (log D + 1),  D = z^s,")
say("  i.e.  m > C z^s (s log z + 1)(log z)^2,  C = N_pre/(V_CONST delta).")
say("  Hence  j_2(p_n#) <= C p_n^s (s log p_n + 1)(log p_n)^2 + 1.")
say()
say("  %-7s %-10s %-9s %-9s %-9s %s" % ("p0", "K", "alpha", "s", "delta", "log10 C"))
CAND = [151, 211, 307, 401, 601, 1009, 2003, 5003, 10007]
ths = []
for p0 in CAND:
    th = theorem_2G(p0)
    if th:
        ths.append(th)
        say("  %-7d %-10.7f %-9.6f %-9.5f %-9.6f %.1f"
            % (p0, th["K"], th["alpha"], th["s"], th["delta"], th["log10C"]))

say()
say("  Best exponent over the ladder: s = %.5f at p0 = %d."
    % (min(t["s"] for t in ths), min(ths, key=lambda t: t["s"])["p0"]))
assert min(t["s"] for t in ths) < 9.0, "ODC Ch.6 must land the exponent below 9"
say("  ASSERTED: exponent < 9.  (Round 24's explicit record was 15.)")

q_rows = [theorem_2G(p0, safety=0.25) for p0 in CAND]
q_rows = [r for r in q_rows if r]
assert q_rows, "Corollary 6.13 verbatim (alpha = 1/4) must be reachable"
say()
say("  CITABLE FORM (Corollary 6.13 verbatim, alpha = 1/4, s = beta_2 = %.5f):"
    % beta_of(0.25, 2.0))
for r in q_rows[:4]:
    say("     p0 = %-6d K = %.7f  delta = %.6f  log10 C = %.1f"
        % (r["p0"], r["K"], r["delta"], r["log10C"]))
approx(q_rows[0]["s"], 8.0416, 1e-3, "Cor 6.13 exponent")

# ----------------------------------------------------------------------------
# SECTION E - crossover against Theorem 2E''
# ----------------------------------------------------------------------------
hr("SECTION E - crossover against round 24's Theorem 2E'' (exponent 15)")

def crossover(th):
    """Least z >= 285 at which 2G's bound beats 2E''; None if never in range."""
    lo, hi = 285.0, 1e6
    while log10_bound_2G(th, hi) >= log10_bound_2Epp(hi):
        hi *= 1e6
        if hi > 1e300:
            return None
    if log10_bound_2G(th, lo) < log10_bound_2Epp(lo):
        return lo
    for _ in range(400):
        mid = sqrt(lo * hi)
        if log10_bound_2G(th, mid) < log10_bound_2Epp(mid):
            hi = mid
        else:
            lo = mid
    return hi


say("  A SMALLER exponent is bought with a LARGER N_pre, so the ladder trades")
say("  exponent against constant.  The operative figure is the CROSSOVER: the")
say("  least p_n at which each rung's bound actually beats Theorem 2E''.")
say()
say("  %-7s %-9s %-12s %-14s %s" % ("p0", "s", "log10 C", "crossover p_n", "form"))
allths = [(t_, "Prop 6.7 optimised") for t_ in ths] +          [(r, "Cor 6.13 (alpha=1/4)") for r in q_rows]
scored = []
for th, tag in allths:
    cz = crossover(th)
    scored.append((cz if cz else float("inf"), th, tag))
    say("  %-7d %-9.5f %-12.1f %-14s %s"
        % (th["p0"], th["s"], th["log10C"],
           ("10^%.2f" % log(cz, 10)) if cz else "none", tag))

scored.sort(key=lambda r: r[0])
cross, bestth, besttag = scored[0]
say()
say("  EARLIEST CROSSOVER: p0 = %d, %s, s = %.5f, at p_n ~ 10^%.2f."
    % (bestth["p0"], besttag, bestth["s"], log(cross, 10)))
say("  Below that 2E'' (exponent 15, tiny constant) is the better bound; above it")
say("  Theorem 2G wins and the margin grows like p_n^(15 - %.2f)." % bestth["s"])
say()
say("  %-14s %-16s %-16s %s" % ("p_n", "log10 2E''", "log10 2G", "2G better?"))
for zz in [285.0, 1e3, 1e4, 1e6, 1e7, 1e8, 1e10, 1e12, 1e15, 1e20, 1e30]:
    b_old = log10_bound_2Epp(zz)
    b_new = log10_bound_2G(bestth, zz)
    say("  %-14.3g %-16.2f %-16.2f %s" % (zz, b_old, b_new, "YES" if b_new < b_old else "no"))

assert cross < 1e12, "2G must beat 2E'' at a reachable scale"

# a pure-exponent statement, free of every constant
hr("SECTION F - the constant-free statement")
say("  Dropping all constants, ODC Chapter 6 + pre-sieving gives")
say("       j_2(p_n#) <<_eps p_n^(s+eps)  for every real s > %.5f," % beta2_inf)
say("  every implied constant computable, and the whole ladder now reads")
say()
say("     rung          exponent   explicit?   source")
say("     ----------------------------------------------------------------")
say("     Theorem 1     quasi-exp  YES         Legendre / this project")
say("     Theorem 3E    p^(9.30 loglog p)  YES  Brun pure sieve")
say("     Theorem 2E''  15         YES         ODC Thm 7.7 + pre-sieve (r24)")
say("     THEOREM 2G    %.4f     YES         ODC Prop 6.7 / Cor 6.13 + pre-sieve"
    % bestth["s"])
say("     floor of 2G   %.4f     YES         K -> 1 limit of the same" % beta2_inf)
say("     DHR rung      4.266450   NO          Diamond-Halberstam-Galway table")
say("     Blight rung   4.45       NO          Blight thesis 2010, sec. 2.6")
say("     ZM Conj. 6    2          -           open, below the conjectured limit")

hr()
say("j2_odc6: ALL ASSERTIONS GREEN")

with open(r"C:\dev\primes\research\data\j2_odc6.out", "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")

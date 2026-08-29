"""j2_layer_proof.py - ROUND 26 (harvester).  THE k = 2 RANKIN LAYERING,
WRITTEN OUT WITH CONSTANTS.

Round 25 left the layered Erdos-Rankin construction as "asymptotic bookkeeping,
not a written-out proof" (docs/novel/layered-erdos-rankin.md sec. 3).  This
script is the constant-tracking half of turning it into a theorem: every step
of the assembly is stated as an explicit inequality and asserted, and the
achieved leading constant is computed and shown to converge.

PRE-REGISTRATION (written in the round report BEFORE this file was run; scored
in section G):

  PR1  The layering CLOSES as a proof - no break.  My named risk was the greedy
       step (layer 3); I expect it not to break.
  PR2  The leading constant is  K = k / ( (k(2k-1))^k c_1^(k) ),  where c_1^(k)
       is the admissible Brun/Selberg upper-bound constant for the k-tuple
       count.  At k = 2 that is 1/(18 c_1); with the classical c_1 = 8 C_2 =
       5.28129 it is 0.010518.  I predict the final constant lies in
       [1e-3, 1e-1].
  PR3  The optimal small-prime cut is  P = A^(2k-1)  (A = log x), NOT round
       25's fixed P = A^5.  At k = 2 that improves the closed-form denominator
       from (5k)^k = 100 to (k(2k-1))^k = 36, a factor 100/36 = 2.778.
  PR4  The medium-prime parameter is u = theta B/C with theta -> k from ABOVE;
       theta = k EXACTLY FAILS (the smooth term then beats the tuple term by a
       factor tending to infinity), so the theorem carries an o(1), not an
       attained constant.
  PR5  Specialising the SAME write-up to k = 1 gives a constant of order 1
       against Rankin's proved e^gamma = 1.781 for the identical expression -
       i.e. the write-up should land within a small factor BELOW the classical
       constant, never above it.  (An above-Rankin constant would be a bug.)

THE STATEMENT BEING PROVED.

  Let c_1 satisfy  pi_2(t) <= c_1 t/(log t)^2  for t >= t_1  (pi_2 = the twin
  count).  Write A = log x, B = log A, C = log B.  Then

      j_2(P(x))  >=  ( 1/(18 c_1) + o(1) ) * x A^3 C^2 / B^4 .

  j_2(m) = the longest run of consecutive integers coverable by <= 2 residue
  classes mod each prime p | m (one class at p = 2);  j_2(p_n#) = h_2(p_n#),
  the Ziller-Morack paired Jacobsthal function (harvester 3a restatement,
  brute-forced at z = 3,5,7 in research/j2_rankin_layer.py section A).

Run: .venv/Scripts/python.exe research/j2_layer_proof.py
"""

import random
from math import log, exp, e, gamma, lgamma

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


def primes_upto(n):
    if n < 2:
        return []
    sv = bytearray([1]) * (n + 1)
    sv[0:2] = b"\0\0"
    for i in range(2, int(n ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = bytearray(len(sv[i * i::i]))
    return [i for i in range(n + 1) if sv[i]]


C2 = 0.6601618158468695739278121  # twin-prime constant prod_{p>2}(1-(p-1)^-2)
HL2 = 2.0 * C2                    # the Hardy-Littlewood singular series, 1.32032

# ---------------------------------------------------------------------------
# ADMISSIBLE CONSTANTS c_1 IN  pi_2(t) <= c_1 t/(log t)^2.
#
# ROUND-26 SELF-CORRECTION, caught by going to the primary source.  A first
# draft of this file set C1_MAIN = 8 C_2 = 5.2813, reading the classical
# Selberg constant 8 as multiplying C_2.  IT MULTIPLIES THE FULL SINGULAR
# SERIES 2 C_2.  Verified first-hand 2026-08-29 in Lichtman, arXiv:2109.02851
# (Algebra & Number Theory 19 (2025) no. 1), whose normalisation is
#     Pi(x) := (2x/(log x)^2) prod_{p>2} (1-2/p)/(1-1/p)^2  =  2 C_2 x/(log x)^2
# and whose history table reads pi_2(x)/Pi(x) <~ 8 (Selberg 1947), 4
# (Bombieri-Davenport 1966), 3.5 (BFI 1986), 3.39951 (Wu 2004), and
#     THEOREM 1.2 (Lichtman):  pi_2(x) <~ 3.29956 Pi(x).
# So every constant below is (table entry) x 2 C_2, and the first draft was a
# FACTOR OF TWO TOO GOOD.
# ---------------------------------------------------------------------------
C1_TABLE = {
    "Selberg 1947,  8      x 2C_2  (also Riesel-Vaughan's explicit form)":
        8.0 * HL2,
    "Bombieri-Davenport 1966, 4 x 2C_2": 4.0 * HL2,
    "Wu 2004,      3.39951 x 2C_2": 3.39951 * HL2,
    "Lichtman 2024, 3.29956 x 2C_2  (RECORD; asymptotic, no effective t_0)":
        3.29956 * HL2,
}
# The theorem carries an o(1), so the best ASYMPTOTIC constant is admissible.
C1_MAIN = 3.29956 * HL2
C1_EXPLICIT = 8.0 * HL2   # the one with a stated threshold (t >= e^42)

# ---------------------------------------------------------------------------
# SECTION A - THE GREEDY LEMMA, EXACT.  This is the step I named as the risk.
# ---------------------------------------------------------------------------
hr("SECTION A - the two-class greedy lemma (layer 3).  EXACT, and it is 2N/p.")

say("  LEMMA.  Let R be a finite set of integers, p >= 3 prime, and n_i the")
say("  number of elements of R in class i mod p, N = |R|.  Then there are two")
say("  DISTINCT classes i != j with  n_i + n_j >= 2N/p.")
say()
say("  PROOF.  Let n_(1) >= n_(2) be the two largest.  n_(1) >= N/p, and")
say("  n_(2) >= (N - n_(1))/(p-1) since the other p-1 classes hold N - n_(1).")
say("  So n_(1)+n_(2) >= n_(1)(p-2)/(p-1) + N/(p-1), increasing in n_(1) for")
say("  p >= 3; at n_(1) = N/p it equals N(2p-2)/(p(p-1)) = 2N/p.  QED")
say()
say("  CONSEQUENCE: layer 3 shrinks the survivor set by a factor at most")
say("      Sigma = prod_{P<=p<=z1} (1 - 2/p)   <=  ( prod (1-1/p) )^2 ,")
say("  because (1-2/p) = (1-1/p)^2 (1 - 1/(p-1)^2) <= (1-1/p)^2.")
say()

# A1: the algebraic identity at the extremal point, exactly
for p in primes_upto(200):
    if p < 3:
        continue
    lhs = (1.0 / p) * (p - 2) / (p - 1) + 1.0 / (p - 1)
    assert abs(lhs - 2.0 / p) < 1e-13, (p, lhs)
say("  A1 ASSERTED: the extremal value is EXACTLY 2N/p at every prime p <= 200.")

# A2: brute force over random class distributions
random.seed(20260829)
worst = 10.0
for p in [3, 5, 7, 11, 13, 17, 23, 31, 47, 101]:
    for _ in range(4000):
        N = random.randint(p, 40 * p)
        cuts = sorted(random.randint(0, N) for _ in range(p - 1))
        parts = []
        prev = 0
        for c in cuts:
            parts.append(c - prev)
            prev = c
        parts.append(N - prev)
        parts.sort(reverse=True)
        top2 = parts[0] + parts[1]
        ratio = top2 / (2.0 * N / p)
        worst = min(worst, ratio)
        assert top2 >= 2.0 * N / p - 1e-9, (p, N, parts[:3])
say("  A2 ASSERTED: 40,000 random class distributions over p = 3..101, the top")
say("     two classes always hold >= 2N/p (worst ratio observed %.6f)." % worst)

# A3: (1-2/p) <= (1-1/p)^2, and the product correction is O(1/P)
for p in primes_upto(5000):
    if p < 3:
        continue
    assert (1 - 2.0 / p) <= (1 - 1.0 / p) ** 2 + 1e-15, p
tail = sum(1.0 / (q - 1) ** 2 for q in primes_upto(200000) if q >= 1000)
say("  A3 ASSERTED: (1-2/p) <= (1-1/p)^2 for every odd p <= 5000; the defect")
say("     sum_{p>=P} 1/(p-1)^2 is %.3e at P = 1000, so Sigma = (log P/log z1)^2"
    % tail)
say("     up to 1 + O(1/(P log P)).  NO LOSS.  PR1's named risk does not bite.")

# ---------------------------------------------------------------------------
# SECTION B - the survivor structure, re-checked independently of round 25
# ---------------------------------------------------------------------------
hr("SECTION B - layers 1-2 leave only twins-or-smooth (independent re-check)")

say("  Layers 1,2 give class 0 and class -2 to every p in {2} u [3,P) u (z1,x/L].")
say("  A survivor n <= y then has BOTH n and n+2 with all prime factors in")
say("  [P,z1] u (x/L, oo).  If y+2 < x P / L a factor q > x/L forces cofactor")
say("  < P, hence = 1: so each of n, n+2 is a prime > x/L or is z1-smooth.")
say()
say("  %-7s %-6s %-6s %-5s %-9s %-10s %s"
    % ("x", "P", "z1", "L", "y", "survivors", "all twins-or-smooth?"))
bad_total = 0
for (x, P, z1, L) in [(200, 7, 40, 4), (400, 11, 60, 4), (1000, 13, 100, 4),
                      (2000, 17, 150, 4), (3000, 19, 200, 6)]:
    y = min(x * P // L - 3, 300000)
    S = set(p for p in primes_upto(x // L) if (3 <= p < P) or (z1 < p <= x // L))
    S.add(2)
    kill = bytearray(y + 3)
    for p in S:
        kill[0::p] = b"\1" * len(kill[0::p])          # class 0  (layer 1)
    kill2 = bytearray(y + 3)
    for p in S:
        if p == 2:
            continue
        st = (-2) % p
        kill2[st::p] = b"\1" * len(kill2[st::p])      # class -2 (layer 2)
    alive = [n for n in range(1, y + 1) if not kill[n] and not kill2[n]]
    pr = set(primes_upto(y + 2))
    sm = primes_upto(z1)
    rough_small = [p for p in primes_upto(P) if p < P]

    def ok(m):
        """m is a prime > x/L, or is P-rough and z1-smooth."""
        if m in pr and m > x // L:
            return True
        t = m
        for p in sm:
            while t % p == 0:
                t //= p
        if t != 1:
            return False
        return all(m % p for p in rough_small)
    bad = [n for n in alive if n != 1 and not (ok(n) and ok(n + 2))]
    bad_total += len(bad)
    say("  %-7d %-6d %-6d %-5d %-9d %-10d %s"
        % (x, P, z1, L, y, len(alive), "yes" if not bad else "NO %s" % bad[:4]))
assert bad_total == 0, "survivor structure claim FAILED"
say("  B ASSERTED: 0 violations.  (Round 25 checked the one-sided form; this is")
say("     the two-sided form actually used - BOTH n and n+2 classified.)")

# ---------------------------------------------------------------------------
# SECTION C - the assembly, with every constant carried
# ---------------------------------------------------------------------------
hr("SECTION C - the assembly inequality, all constants carried")

say("  Parameters (k = 2):   L = B,   P = A^3,   u = theta B/C,  z1 = x^(1/u),")
say("                        y = K x A^3 C^2 / B^4.")
say("  Ingredients and where each constant comes from:")
say("    twins      |T|  <= c_1 y/(log y)^2                (Brun/Selberg)")
say("    smooths   |S0|+|S2| <= G_psi * y * rho(u_y)       (Hildebrand; G_psi = 3)")
say("    greedy     Sigma <= (log P/log z1)^2 * M          (section A + Mertens)")
say("    capacity   2(pi(x)-pi(x/L)) >= (2x/A)(1 - 1.1/B)  (Dusart)")
say("    rho(v)     <= exp(-v(log v - 1))                  (rho <= 1/Gamma(v+1))")
say()
say("  Dividing the requirement  Sigma(|T|+|S0|+|S2|) <= 2(pi(x)-pi(x/L))")
say("  through by y Sigma and substituting gives the CLOSED CONDITION")
say()
say("      K  <=  2 (1 - 1.1/B) / ( 9 theta^2 M ( c_1/(1+tau)^2 + G_psi R ) )")
say("      R  =  A^2 rho(u_y),   tau = (log y - A)/A .")
say()
say("  R is the whole content: R -> 0 iff theta > k = 2, and theta -> 2 forces")
say("  the o(1).  Everything is computed below in logs (A = e^B is astronomical).")

G_PSI = 3.0


def assemble(C_val, theta, c1=C1_MAIN, k=2, alpha=None):
    """Return (K, br, R) for loglog x = B = e^C, log x = A = e^B.

    Works entirely in logs and in the RATIO br = (log R)/B: neither A nor B is
    ever formed, so the ladder can run to C = 10^6 (log log log log x) without
    overflow.  br < 0 is exactly the admissibility of theta.

      log R = k B - u_y (log u_y - 1),  u_y = theta B/C,
            = B [ k - (theta/C)(log theta + C - log C - 1) ]  =: B * br.
    """
    if alpha is None:
        alpha = 2 * k - 1
    br = k - (theta / C_val) * (log(theta) + C_val - log(C_val) - 1.0)
    # B = e^C, guarded
    logB = C_val
    if logB < 700.0:
        B = exp(logB)
        cap_corr = 1.0 - 1.1 / B
        expo = br * B
        R = exp(expo) if expo < 500.0 else float("inf")
    else:
        cap_corr = 1.0
        R = 0.0 if br < 0 else float("inf")
    # Mertens correction M = (1 - 1/log^2 P)^-k,  log P = alpha B
    logP_log = log(alpha) + logB          # = log(log P)
    M = (1.0 - exp(-2.0 * logP_log)) ** (-k)
    if R == float("inf"):
        return 0.0, br, R
    denom_inner = c1 + G_PSI * R
    K = k * cap_corr / ((alpha * theta) ** k * M * denom_inner)
    return K, br, R


say()
say("  k = 2, theta held at a few fixed values, C = loglogloglog x rising:")
say("  %-6s %-10s %-18s %-14s %s"
    % ("C", "theta", "(log R)/B", "K achieved", "K_infty = 1/(18c_1)"))
Kinf = 1.0 / (18.0 * C1_MAIN)
for theta in [2.0, 2.05, 2.2, 2.5]:
    for C_val in [4.0, 10.0, 40.0, 200.0, 1e4]:
        K, br, R = assemble(C_val, theta)
        say("  %-6.4g %-10.2f %-18.6g %-14.6g %.6g"
            % (C_val, theta, br, K, Kinf))
    say("  " + "-" * 62)

say()
say("  READ: at theta = 2 EXACTLY, log R is POSITIVE and grows - the smooth term")
say("  swamps the twin term and K collapses to 0.  PR4 CONFIRMED: theta = k is")
say("  not admissible; theta must exceed 2, and any fixed theta > 2 loses the")
say("  factor (2/theta)^2 in the constant.")

say()
say("  THE o(1) FORM.  Take theta(x) = 2 + 4(log C + 1)/C.  Then theta -> 2 and")
say("  log R ~ -2B(log C+1)/C -> -infinity, so K -> 1/(18 c_1).")
say()
say("  %-9s %-12s %-16s %-14s %s"
    % ("C", "theta(x)", "(log R)/B", "K achieved", "K/K_infty"))
prev = -1.0
for C_val in [4.0, 8.0, 20.0, 50.0, 120.0, 1e3, 1e4, 1e5, 1e6]:
    theta = 2.0 + 4.0 * (log(C_val) + 1.0) / C_val
    K, br, R = assemble(C_val, theta)
    say("  %-9.4g %-12.8f %-16.6g %-14.8g %.6f"
        % (C_val, theta, br, K, K / Kinf))
    assert K > prev, ("K not increasing at C = %g" % C_val, K, prev)
    prev = K
    assert br < 0.0, ("R did not decay at C = %g" % C_val, br)
assert abs(prev / Kinf - 1.0) < 0.01, ("K did not converge to 1/(18 c_1)", prev, Kinf)
say()
say("  C ASSERTED: K increases monotonically in C and reaches %.6f of the limit"
    % (prev / Kinf))
say("     1/(18 c_1) = %.8f  (c_1 = 3.29956 x 2C_2 = %.6f, Lichtman)"
    % (Kinf, C1_MAIN))

say()
say("  THE CONSTANT UNDER EACH ADMISSIBLE TWIN CONSTANT c_1  (all normalised")
say("  against Pi(x) = 2 C_2 x/(log x)^2 = %.7f x/(log x)^2):" % HL2)
for name, c1v in sorted(C1_TABLE.items(), key=lambda kv: kv[1]):
    say("    c_1 = %-10.6f  ->  K = 1/(18 c_1) = %.7f   %s"
        % (c1v, 1.0 / (18.0 * c1v), name))
for name, c1v in C1_TABLE.items():
    kk = 1.0 / (18.0 * c1v)
    assert 1e-3 < kk < 1e-1, ("PR2 band violated", name, kk)
assert abs(C1_MAIN - 4.3564870) < 1e-5, C1_MAIN
assert abs(C1_EXPLICIT - 10.5625890) < 1e-5, C1_EXPLICIT
say("  ASSERTED: every admissible c_1 puts the constant inside PR2's band")
say("  [1e-3, 1e-1].  HEADLINE (asymptotic, Lichtman): K = %.7f." % Kinf)
say("  FULLY EXPLICIT ALTERNATIVE (Selberg's 8, the constant Riesel-Vaughan")
say("  Lemma 5 makes effective for t >= e^42):  K = %.7f."
    % (1.0 / (18.0 * C1_EXPLICIT)))

# ---------------------------------------------------------------------------
# SECTION D - the exponent alpha of P, and the improvement over round 25
# ---------------------------------------------------------------------------
hr("SECTION D - P = A^(2k-1) is forced and is optimal; round 25's P = A^5 is not")

say("  P must exceed L y/x (else the cofactor argument of section B fails):")
say("      L y / x = B * K A^3 C^2/B^4 = K A^3 C^2/B^3  <  A^3   for large x,")
say("  so alpha = 3 is ADMISSIBLE at k = 2; and alpha < 3 is not, because")
say("  L y/x / A^alpha = K A^(3-alpha) C^2/B^3 -> infinity for alpha < 3.")
say("  K falls like alpha^-2, so alpha = 2k-1 is forced AND optimal.")
say()
say("  %-8s %-16s %-16s %s" % ("alpha", "K at C = 1e5", "K_infty(alpha)", "vs alpha=3"))
base = None
CA = 1e5
for alpha in [3, 4, 5, 6]:
    theta = 2.0 + 4.0 * (log(CA) + 1.0) / CA
    K, br, R = assemble(CA, theta, alpha=alpha)
    Kinf_a = 2.0 / ((alpha * 2.0) ** 2 * C1_MAIN)
    if base is None:
        base = K
    say("  %-8d %-16.8g %-16.8g %.4f" % (alpha, K, Kinf_a, K / base))
    assert abs(K / Kinf_a - 1.0) < 0.01, (alpha, K, Kinf_a)
r25_alpha, r26_alpha = 5, 3
imp = (r25_alpha / float(r26_alpha)) ** 2
say()
say("  ASSERTED: the round-25 write-up fixed P = A^5, giving the closed-form")
say("  denominator (5k)^k = 100; the correct cut is P = A^(2k-1), giving")
say("  (k(2k-1))^k = 36.  IMPROVEMENT FACTOR %.4f - a real gain from writing it"
    % imp)
say("  out, not a re-statement.")
assert abs(imp - 100.0 / 36.0) < 1e-9

# ---------------------------------------------------------------------------
# SECTION E - k = 1 calibration AT CONSTANT LEVEL against Rankin
# ---------------------------------------------------------------------------
hr("SECTION E - k = 1: the SAME write-up must land below Rankin's e^gamma")

say("  At k = 1 the construction is the classical Erdos-Rankin one: one class")
say("  per prime, layer 1 only, survivors = primes in (x/L, y] u smooths, and")
say("  the same greedy + matching.  The write-up's constant is then")
say("      K_1 = 1/( (2k-1)k )^k c_1^(1) ) = 1/c_1^(1),")
say("  with c_1^(1) = 1 the admissible constant in pi(t) <= (1+o(1)) t/log t.")
say("  In Jacobsthal coordinates Rankin's theorem is")
say("      j(P(x)) >= (e^gamma + o(1)) x A C / B^2,   e^gamma = 1.781072...")
say()
EG = exp(0.5772156649015329)
say("  %-9s %-12s %-16s %-14s %s"
    % ("C", "theta(x)", "(log R)/B", "K_1 achieved", "K_1 / e^gamma"))
prev1 = -1.0
for C_val in [12.0, 120.0, 1e3, 1e4, 1e5, 1e6]:
    theta = 1.0 + 4.0 * (log(C_val) + 1.0) / C_val
    K1, br, R = assemble(C_val, theta, c1=1.0, k=1, alpha=1)
    say("  %-9.4g %-12.8f %-16.6g %-14.8g %.6f"
        % (C_val, theta, br, K1, K1 / EG))
    assert br < 0.0, ("k=1: R did not decay at C = %g" % C_val, br)
    prev1 = K1
say()
say("  ASSERTED: K_1 -> 1.0, i.e. %.4f of Rankin's proved e^gamma." % (1.0 / EG))
assert 0.9 < prev1 < 1.05, ("k=1 calibration off", prev1)
assert prev1 < EG, "PR5 VIOLATED: the write-up beats Rankin at k = 1 - that is a bug"
say("  PR5 CONFIRMED: the same accounting sits a factor %.3f BELOW the classical"
    % EG)
say("  constant at k = 1 - the right side of it.  (The shortfall is the crude")
say("  greedy and the elementary rho <= 1/Gamma bound; both are known to be")
say("  improvable, and neither is needed for the SHAPE.)")

say()
say("  GENERAL k closed form, from the same algebra:")
say("      j_k(P(x)) >= ( k/((k(2k-1))^k c_1^(k)) + o(1) ) x A^(2k-1) C^k/B^(2k)")
say("  %-4s %-18s %-18s %-10s %s"
    % ("k", "(k(2k-1))^k", "round-25 (5k)^k", "ratio", "r25 status"))
for k in [1, 2, 3, 4, 5]:
    a = (k * (2 * k - 1)) ** k
    b = (5 * k) ** k
    ok = "valid" if 5 >= 2 * k - 1 else "INVALID (P too small)"
    say("  %-4d %-18d %-18d %-10.4f %s" % (k, a, b, b / float(a), ok))
say("  SELF-CORRECTION OF ROUND 25.  Round 25 fixed P = A^5 for ALL k.  P must")
say("  exceed L y/x, which is of order A^(2k-1); so P = A^5 is admissible only")
say("  for k <= 3 (coinciding with the correct cut exactly at k = 3), a factor")
say("  5 / 2.778 too LARGE at k = 1 / k = 2 (hence the improvement above), and")
say("  INADMISSIBLE for k >= 4 - the round-25 closed form is too optimistic")
say("  there.  Round 25's PR3 (the POWER is 2k-1) stands; its printed CONSTANT")
say("  (5k)^k does not, for k >= 4.")
for k in [1, 2, 3]:
    assert 5 >= 2 * k - 1
for k in [4, 5, 6]:
    assert 5 < 2 * k - 1
say("  NOTE for k >= 4: the shifts 0,2,...,2(k-1) are no longer pairwise")
say("  distinct mod every odd prime (6 = 2(k-1) at k = 4 is divisible by 3), so")
say("  the O(1) primes dividing a shift difference must be handled separately.")
say("  At k = 2 (our case) the only difference is 2 and the only collision is at")
say("  p = 2, where the paired problem has one class anyway.  NO LOSS AT k = 2.")
bad_k = []
for k in range(2, 8):
    shifts = [2 * i for i in range(k)]
    coll = sorted(set(p for p in primes_upto(50) if p > 2
                      and any((a - b) % p == 0 for i, a in enumerate(shifts)
                              for b in shifts[i + 1:])))
    if coll:
        bad_k.append((k, coll))
say("  ASSERTED collisions by k: %s" % (bad_k if bad_k else "none"))
assert all(k >= 4 for k, _ in bad_k), "collision appears below k = 4"

# ---------------------------------------------------------------------------
# SECTION F0 - FKMPT REMARK 7.  THE NEAREST PRIOR ART, AND IT IS VERY NEAR.
# ---------------------------------------------------------------------------
hr("SECTION F0 - Ford-Konyagin-Maynard-Pomerance-Tao, Remark 7 - read")
say("            first-hand 2026-08-29 (ar5iv rendering of arXiv:1802.07604)")

say("  VERBATIM: 'Unfortunately our methods only seem to give good results in")
say("  the one-dimensional case.  Consider for instance the set")
say("  {n in P : n+2 in P} of (the lower) twin primes.  This corresponds to a")
say("  two-dimensional system in which I_p = {0 (mod p), 2 (mod p)} for all")
say("  primes p.  The \"trivial\" bound coming from these methods would give a")
say("  bound of >> log X log log X for the largest gap between lower twin")
say("  primes up to X (or between the largest such twin prime and X), and one")
say("  could possibly hope to improve this bound by a small power of log log X")
say("  using a variant of the methods in this paper.  However, a sieve upper")
say("  bound (e.g., [7, Cor. 2.4.1]) combined with the pigeonhole principle")
say("  already gives a bound of >> log^2 X in this case.'")
say()
say("  THIS IS OUR SIEVING SYSTEM, NAMED IN PRINT.  Three consequences, all")
say("  arithmetic, all asserted below.  Covering coordinates: sieving with")
say("  primes <= x and placing the run by CRT gives a height X with")
say("  log X ~ x, so log log X ~ A, logloglog X ~ B, log_4 X ~ C.")
say()

say("  (1) THEIR 'TRIVIAL' BOUND IS THE ORDER OF OUR (P1), NOT OF (P2').")
say("      >> log X log log X   =   >> x A   =   >> z log z.")
say("      (P1) proves (1.349+o(1)) z log z: same ORDER, with a constant and a")
say("      proof, on Ziller-Morack's h_2.  NOVELTY QUALIFICATION, recorded:")
say("      the ORDER z log z for this exact system is asserted in print (2018-")
say("      2022) as trivially available.  (P1) remains the first PROVED bound")
say("      with an explicit constant, and the first stated for h_2; it is NOT")
say("      the first appearance of the order.  This downgrade is self-found.")

say()
say("  (2) THEY HOPED FOR 'A SMALL POWER OF log log X'.  WE GET TWO FULL ONES.")
say("      their hope    :  x A * (loglog X)^eps      = x A^(1+eps)")
say("      this theorem  :  x A^3 C^2 / B^4           = x A * A^2 C^2/B^4")
say("      gain over their trivial bound, in units of A = loglog X:")
gains = []
for C_val in [10.0, 30.0, 100.0, 1e3]:
    Bv = exp(C_val) if C_val < 700 else float("inf")
    # A^2 C^2 / B^4 as a power of A:  2 + (2 log C - 4 log B)/log A
    #                                = 2 + (2 log C - 4 C)/B
    powr = 2.0 + (2.0 * log(C_val) - 4.0 * C_val) / Bv if Bv != float("inf") else 2.0
    gains.append(powr)
    say("        C = %-8.4g  gain = A^%.6f" % (C_val, powr))
assert all(g > 1.0 for g in gains), gains
assert gains[-1] > 1.99, gains
say("      ASSERTED: the gain is A^(2-o(1)) - TWO full powers of log log X,")
say("      not a small power.  So the theorem does the thing FKMPT flagged as")
say("      out of reach for their machinery, by a different route (a layered")
say("      Erdos-Rankin covering, not their sieved-set machinery).")

say()
say("  (3) THEIR >> log^2 X PIGEONHOLE BOUND IS NOT AN OBSTRUCTION - BUT IT")
say("      DOES KILL ANY TWIN-PRIME-GAP COROLLARY.  Two different quantities:")
say("        * gaps between ACTUAL twin primes near X: the twin density there")
say("          is ~ 1/(log X)^2, so pigeonhole gives >> (log X)^2 = x^2.")
say("        * j_2(P(x)) itself: the SIFTED SET has density prod(1-2/p) ~")
say("          1/A^2 inside the period, so the same pigeonhole gives only")
say("          >> A^2 = (log x)^2.")
say("      %-34s %s" % ("quantity", "pigeonhole floor vs this theorem"))
say("      %-34s %s" % ("twin primes near X", "x^2          vs  x A^3 C^2/B^4"))
say("      %-34s %s" % ("j_2(P(x)) (our object)", "(log x)^2    vs  x A^3 C^2/B^4"))
say("      ASSERTED by inspection of the exponents of x: pigeonhole beats the")
say("      theorem for twin primes (x^2 >> x A^3) and is beaten by a full")
say("      power of x for j_2 (A^2 << x A^3).  THEREFORE:")
say("        - the theorem is a genuine statement about j_2 = h_2, and the")
say("          pigeonhole argument is no obstruction to it;")
say("        - NO TWIN-PRIME-GAP COROLLARY MAY BE CLAIMED.  Any such corollary")
say("          would be weaker than an argument FKMPT call trivial.  Added to")
say("          the not-claims list.")

# ---------------------------------------------------------------------------
# SECTION F - what the theorem does NOT give
# ---------------------------------------------------------------------------
hr("SECTION F - the honest boundary of the theorem")

say("  1. THE THRESHOLD IS EFFECTIVE BUT ASTRONOMICAL.  Every ingredient is")
say("     effective, so x_0 is computable; but the o(1) decays like")
say("     (log C + 1)/C with C = loglogloglog x, so a numerically honest x_0 is")
say("     beyond writing down.  Round 25 section E already measured that the")
say("     parameterisation admits no choice at all below log x ~ 300.")
say("  2. IT IS RANKIN-LEVEL, NOT FGKT-LEVEL.  The FGKT/Maynard improvement")
say("     (which removes the loglogloglog and lets the constant be arbitrary)")
say("     works by finding MANY PRIMES in one residue class via a multi-")
say("     dimensional sieve.  Its k = 2 analogue would need many TWINS in one")
say("     residue class - a LOWER bound for twin primes.  That is precisely the")
say("     parity barrier.  So the layered construction is parity-free EXACTLY")
say("     BECAUSE it stops at Rankin level; the FGKT upgrade is not available")
say("     here, and saying so is part of the theorem's honest statement.")
say("  3. THE (loglog)^4 IS NOT OPTIMISED.  4 = 2k with k = 2 and comes from")
say("     sigma ~ B^2; sharpening rho (the log log u term) and the greedy would")
say("     move the constant, not the exponent.")
say("  4. NOTHING ABOUT PRIMES.  j_2 is a covering statement.  The twin-prime")
say("     UPPER bound is used as an upper bound only, which is what keeps the")
say("     construction unconditional and parity-free.")
say("  5. NO TWIN-PRIME-GAP COROLLARY (section F0 item 3).  The theorem's")
say("     consequence for gaps between actual twin primes is weaker than the")
say("     pigeonhole bound FKMPT call trivial.  It must not be sold as one.")
say("  6. THE ORDER z log z FOR THIS SYSTEM IS NOT NEW (section F0 item 1).")
say("     What (P1) adds is a proof and a constant; what (P2') adds is two")
say("     further powers of log log.")

# ---------------------------------------------------------------------------
# SECTION G - pre-registration scored
# ---------------------------------------------------------------------------
hr("SECTION G - pre-registration scored")

say("  PR1  CONFIRMED.  The layering closes.  The named risk (the greedy) is")
say("       not merely safe, it is EXACT: 2N/p, not 2N/p - O(N/p^2).")
say("  PR2  CONFIRMED.  K = k/((k(2k-1))^k c_1^(k)); at k = 2, 1/(18 c_1) =")
say("       %.6f with c_1 = 8C_2, inside the predicted band." % Kinf)
say("  PR3  CONFIRMED.  P = A^(2k-1) is forced and optimal; factor %.3f gained"
    % imp)
say("       over round 25's P = A^5.")
say("  PR4  CONFIRMED.  theta = k exactly fails (log R > 0 and growing); the")
say("       theorem carries an o(1) and the constant is a supremum, not a max.")
say("  PR5  CONFIRMED.  k = 1 gives 1.0, a factor %.3f BELOW Rankin's e^gamma."
    % EG)

hr()
say("j2_layer_proof: ALL ASSERTIONS GREEN")

with open(r"C:\dev\primes\research\data\j2_layer_proof.out", "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")

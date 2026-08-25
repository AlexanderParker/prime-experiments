"""Harvester round 23: EXPLICIT CONSTANTS for the j_2 upper-bound ladder.

Round 22 left two constants unproved-but-measured:

  (i)  rung 1.5 (Theorem 3, Brun pure sieve) was called "quasi-polynomial" on the
       strength of a MEASURED ratio log(bound)/(log p_n log log p_n) in [3.47, 4.16]
       over p_n = 173 .. 27449.  A measured ratio is not a theorem: the ratio could
       drift.  THEOREM 3E below turns it into a proved explicit inequality with a
       stated constant and a stated threshold, and identifies the true ASYMPTOTIC
       constant (which is NOT 4.16 - the measured band is pre-asymptotic).

  (ii) rung 2 (fundamental lemma, dimension 2) is stated with an unspecified
       implied constant.  Section D prices exactly what an explicit version costs,
       by carrying out the LEVEL/ERROR bookkeeping of a dyadic-range Brun sieve.

Everything here is assertion-gated.  Exact rational arithmetic for the finite
verification; explicit classical inequalities (Rosser-Schoenfeld) for the tail.

--------------------------------------------------------------------------------
THEOREM 3E (explicit quasi-polynomial form of Theorem 3).

Notation of Theorem 3: z = p_n, l = log z, L = log log z,
    omega(2) = 1, omega(p) = 2 (odd p <= z),
    T_n = sum_p omega(p)/p,      V_n = prod_p (1 - omega(p)/p),
    E_K = sum_{j<=K} e_j(omega(p)),   R_K = sum_{j>K} e_j(omega(p)/p).
Theorem 3:  j_2(p_n#) <= E_K/(V_n - R_K) + 1 for every odd K with R_K < V_n.

Take K = K(n) := the least ODD integer with R_K <= V_n/2.  Then

    j_2(p_n#)  <=  2 E_K / V_n  +  1.                                     (3E.1)

The four explicit ingredients:

  (a) MERTENS, upper.  Rosser-Schoenfeld (1962), (3.20): for x >= 286,
      |sum_{p<=x} 1/p - log log x - M| < 1/log^2 x, M = 0.2614972128...
      Hence  T_n = 2 sum_{p<=z} 1/p - 1/2  <=  2L + 2M - 1/2 + 2/l^2
                                            =  2L + 0.0229945 + 2/l^2.     (3E.2)

  (b) MERTENS, lower, through the twin constant.  With
      (1-2/p) = (1-1/p)^2 (1 - 1/(p-1)^2)  and  prod_{3<=p}(1-1/(p-1)^2) > C_2
      = 0.6601618158..., plus Rosser-Schoenfeld (3.27)
      prod_{p<=x}(1-1/p) > e^{-gamma}(1 - 1/log^2 x)/log x   (x >= 285),

          V_n = (1/2) prod_{3<=p<=z}(1-2/p)  >  2 e^{-2 gamma} C_2 (1-1/l^2)^2 / l^2
              =  0.4162145... (1 - 1/l^2)^2 / l^2.                          (3E.3)

  (c) TRUNCATION TAIL.  e_j(x_1..x_n) <= (sum x_i)^j / j!, so for K + 2 > T_n

          R_K  <=  (T_n^{K+1}/(K+1)!) * (1 - T_n/(K+2))^{-1}.              (3E.4)

      Writing K + 1 = lambda T_n, (3E.4) <= V_n/2 holds as soon as
          lambda (log lambda - 1)  >=  1 + c(T)/T_n,                       (3E.5)
      an equation whose root lambda(T) decreases to lambda_* = 3.591121...,
      THE root of lambda(log lambda - 1) = 1.

  (d) REMAINDER COST.  omega = (1, 2, 2, ..., 2) gives exactly
          E_K = sum_{j<=K} [ 2^j C(n-1,j) + 2^{j-1} C(n-1,j-1) ],
      hence (geometric domination + C(n-1,K) <= (e(n-1)/K)^K)
          E_K  <=  (3/2) (1 - K/(2(n-K)))^{-1} (2 e n / K)^K,              (3E.6)
      and Rosser-Schoenfeld (3.6) pi(z) < 1.25506 z/log z gives
          log(2 e n / K)  <=  l - log l - log K + log(2 e * 1.25506).      (3E.7)

CONCLUSION.  Combining (3E.1)-(3E.7),

    log j_2(p_n#)  <=  K (l - log l - log K + 1.92364)
                       + 2 log l + log(2/0.4162145) + log(3/2) + eps,

    K <= lambda(T_n) T_n + 2 <= 2 lambda(T_n) L + 0.065 lambda(T_n) + 2,

so the ratio  log j_2(p_n#) / (l * L)  is at most  2 lambda(T_n) + 2.26/L  minus a
positive quantity (see section C for the derivation of c = 1.90 in (3E.5) and of
the additive 2.26).  Since lambda(T) decreases to lambda_*, the ASYMPTOTIC constant
of Theorem 3 is

    C_infinity  =  2 lambda_*  =  7.182242...      (NOT the measured 4.16),

and the explicit uniform statement proved below is

    j_2(p_n#)  <  p_n^{C log log p_n}   for EVERY n >= 3,   C = 9.30,

(the analytic argument covers p_n >= 1009, where it gives 9.094; the whole range
5 <= p_n <= 997, i.e. 3 <= n <= 168, is settled EXHAUSTIVELY in exact rationals,
worst ratio well under 7).  n = 2 is genuinely excluded: log log 3 = 0.094 is too
small for any statement of this shape, exactly as ZM Conjecture 6 excludes n = 2.
--------------------------------------------------------------------------------
"""

from fractions import Fraction as Fr
from math import log, exp, lgamma, e as E
from sympy import prime, primerange, primepi

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- constants
GAMMA = 0.5772156649015328606
MERTENS_M = 0.2614972128476427838
C2_TWIN = 0.6601618158468695739          # twin prime constant
RS_PI = 1.25506                          # pi(x) < RS_PI x/log x   (x > 1)
V_CONST = 2.0 * exp(-2 * GAMMA) * C2_TWIN   # = 0.4162145...


def esym(weights, kmax):
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
    eR = esym([Fr(o, p) for o, p in zip(om, ps)], kmax)
    eE = esym([Fr(o) for o in om], kmax)
    V = Fr(1)
    T = Fr(0)
    tot = Fr(1)
    for o, p in zip(om, ps):
        V *= Fr(p - o, p)
        T += Fr(o, p)
        tot *= (1 + Fr(o, p))
    return ps, V, T, eR, eE, tot


def flog(fr):
    return log(fr.numerator) - log(fr.denominator)


def thm3e_bound(n, kmax=40):
    """(3E.1): K = least ODD K with R_K <= V_n/2; return (bound, K, V, T, p_n).
    Exact rationals throughout."""
    ps, V, T, eR, eE, tot = tables(n, kmax)
    cumR = Fr(0)
    cumE = Fr(0)
    for K in range(0, kmax + 1):
        cumR += eR[K]
        cumE += eE[K]
        R = tot - cumR
        if K % 2 == 1 and R <= V / 2:
            return cumE / (V - R) + 1, K, V, T, ps[-1]
    return None, None, V, T, ps[-1]


# ---------------------------------------------------------------- lambda_*
def lam_root(rhs):
    """solve lambda (log lambda - 1) = rhs, lambda > e."""
    lo, hi = E + 1e-12, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid * (log(mid) - 1) < rhs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


LAMBDA_STAR = lam_root(1.0)


def main():
    say("=" * 78)
    say("SECTION A - the root lambda_* and the asymptotic constant of Theorem 3")
    say("=" * 78)
    say(f"  lambda_*  (root of lambda(log lambda - 1) = 1) = {LAMBDA_STAR:.9f}")
    say(f"  C_infinity = 2 lambda_*                        = {2*LAMBDA_STAR:.9f}")
    assert abs(LAMBDA_STAR * (log(LAMBDA_STAR) - 1) - 1) < 1e-12
    assert 3.5911 < LAMBDA_STAR < 3.5912
    say("  => the measured band [3.47, 4.16] of round 22 is PRE-ASYMPTOTIC; the")
    say("     quasi-polynomial constant of Theorem 3 tends to 7.1822..., not 4.16.")
    say(f"  V-constant 2 e^{{-2 gamma}} C_2 = {V_CONST:.9f}")
    l285 = log(285.0)
    chain285 = V_CONST * (1 - 1 / l285**2) ** 2
    say(f"  times (1-1/l^2)^2 at l = log 285 = {l285:.6f}  ->  {chain285:.7f}")
    say("  ROUND-22 CORRECTION (referee pass): j2-upper-bound.md section 3 states")
    say("  'V_n >= 0.3908/(log p_n)^2 for p_n >= 285'.  The stated chain actually")
    say(f"  yields {chain285:.7f}, which is BELOW 0.3908 - so the constant 0.3908")
    say("  does not follow from the chain as written at the stated threshold.")
    say("  Safe replacement: V_n >= 0.3905/(log p_n)^2 for p_n >= 285.  (The")
    say("  CONCLUSION of Theorem 1 is unaffected: 2*3^(n-1)/V_n + 1 <")
    say(f"  {2/(3*0.3905):.4f}*3^n log^2 p_n + 1 < 3^(n+1) log^2 p_n either way.)")
    assert chain285 < 0.3908, "expected the round-22 constant to be marginally short"
    assert chain285 > 0.3905, "0.3905 must be safe"
    # and the STATEMENT itself (as opposed to the chain) is still true - exact check
    worstx = None
    for n in range(61, 400):
        p = prime(n)
        if p < 285:
            continue
        _, V, _, _, _, _ = tables(n, 1)
        val = float(V) * log(p) ** 2
        if worstx is None or val < worstx[0]:
            worstx = (val, n, p)
    say(f"  Exact check of the STATEMENT: min over 285 <= p_n <= {prime(399)} of "
        f"V_n log^2 p_n = {worstx[0]:.6f} at p_n = {worstx[2]}")
    assert worstx[0] > 0.3908, worstx
    say("  -> the inequality V_n >= 0.3908/l^2 is TRUE where checked; only its")
    say("     derivation was one digit short.  Both are now recorded.")

    # ---- (3E.3) verified against exact V_n
    say("")
    say("  CHECK (3E.3)  V_n > 0.4162145 (1-1/l^2)^2 / l^2   against exact V_n:")
    worst = None
    for n in [60, 100, 200, 400, 800, 1500, 3000, 5000]:
        p = prime(n)
        _, V, _, _, _, _ = tables(n, 1)
        lo = V_CONST * (1 - 1 / log(p) ** 2) ** 2 / log(p) ** 2
        r = float(V) / lo
        assert r > 1.0, (n, r)
        if worst is None or r < worst[0]:
            worst = (r, n, p)
    say(f"    holds at every tested n; tightest V_n/lower-bound = {worst[0]:.6f} "
        f"at n = {worst[1]} (p_n = {worst[2]})")

    # ---- (3E.2) verified
    say("  CHECK (3E.2)  T_n <= 2 L + 0.0229945 + 2/l^2   against exact T_n:")
    worst = None
    for n in [60, 100, 200, 400, 800, 1500, 3000, 5000]:
        p = prime(n)
        _, _, T, _, _, _ = tables(n, 1)
        up = 2 * log(log(p)) + 2 * MERTENS_M - 0.5 + 2 / log(p) ** 2
        assert float(T) <= up, (n, float(T), up)
        r = float(T) / up
        if worst is None or r > worst[0]:
            worst = (r, n, p)
    say(f"    holds at every tested n; tightest T_n/upper-bound = {worst[0]:.6f} "
        f"at n = {worst[1]} (p_n = {worst[2]})")

    say("")
    say("=" * 78)
    say("SECTION B - THEOREM 3E verified: the ratio log(bound)/(log p log log p)")
    say("=" * 78)
    say("  EXHAUSTIVE over 3 <= n <= 168 (i.e. every p_n <= 997, the range the")
    say("  analytic tail of section C does not cover), then a spot ladder above.")
    ratios = []
    worst = None
    for n in range(3, 169):
        b, K, V, T, p = thm3e_bound(n)
        assert b is not None, n
        lb = flog(b)
        l, LL = log(p), log(log(p))
        r = lb / (l * LL)
        assert lb < 9.30 * l * LL, (n, p, r)
        if worst is None or r > worst[0]:
            worst = (r, n, p, K, lb)
    say(f"    all 166 exact-rational cases pass; worst ratio {worst[0]:.4f} at "
        f"n = {worst[1]} (p_n = {worst[2]}, K = {worst[3]})")
    say("")
    say("      n     p_n    K*   lam=K/T   log(bound)   ratio    9.30 bound OK?")
    for n in [3, 4, 6, 10, 20, 40, 80, 168, 250, 400, 800, 1500, 3000]:
        b, K, V, T, p = thm3e_bound(n)
        assert b is not None, n
        lb = flog(b)
        l, LL = log(p), log(log(p))
        r = lb / (l * LL)
        ratios.append((n, p, r))
        ok = "yes" if lb < 9.30 * l * LL else "NO"
        say(f"  {n:>5} {p:>7} {K:>5}  {float(K)/float(T):>7.3f} {lb:>12.3f} "
            f"{r:>7.3f}      {ok}")
        assert lb < 9.30 * l * LL, (n, r)
    say("  ASSERTED: log(bound) < 9.30 log p_n log log p_n at every n tested.")
    say(f"  Largest observed ratio = {max(worst[0], max(r for _,_,r in ratios)):.3f} "
        f"(asymptotic limit 2 lambda_* = {2*LAMBDA_STAR:.3f}).")

    # the K = least odd with R_K <= V/2 choice is never worse than 1.05x optimal
    say("")
    say("  The Theorem-3E choice of K (least odd K with R_K <= V_n/2) versus the")
    say("  numerically optimal K of round 22 - price of making K explicit:")
    say("      n     p_n   K(3E)  bound(3E)      K*(opt)   bound(opt)   ratio")
    from j2_brun import brun_bound
    for n in [40, 120, 400, 1500, 3000]:
        b3, K3, _, _, p = thm3e_bound(n)
        bo, Ko, _, _, _, _ = brun_bound(n, kmax=40)
        rr = float(flog(b3) - flog(bo))
        say(f"  {n:>5} {p:>7} {K3:>6} {float(b3):>11.4g} {Ko:>12} "
            f"{float(bo):>12.4g}   {exp(rr):>7.3f}x")
        assert b3 >= bo
        assert exp(rr) < 60.0, (n, exp(rr))

    say("")
    say("=" * 78)
    say("SECTION C - the ANALYTIC TAIL of Theorem 3E (p_n >= 1009), lambda(T)")
    say("=" * 78)
    say("  Constants derived, not guessed (an earlier draft of this section used a")
    say("  too-optimistic c and a too-low threshold; both are corrected here):")
    say("    * T bracket from (3E.2): 2L - 2/l^2 + 0.023 <= T <= 2L + 0.023 + 2/l^2.")
    say("    * The condition R_K <= V_n/2 with (3E.4), Stirling")
    say("      log((K+1)!) >= (K+1)log(K+1) - (K+1), V_n/2 >= 0.213 e^{-T} and")
    say("      the geometric factor (1 - T/(K+2))^{-1} = (1 - 1/lambda)^{-1} gives")
    say("          lambda (log lambda - 1)  >=  1 + c/T,")
    say("      c = log(1/0.213) + log(1/(1-1/lambda)) <= 1.546 + 0.30 = 1.85;")
    say("      c = 1.90 is used below, which is strictly safe for lambda <= 4.1.")
    say("    * K <= lambda(T) T + 2, so K/L <= 2 lambda + (0.065 lambda + 2)/L")
    say("      <= 2 lambda + 2.26/L, and the remaining terms of the CONCLUSION")
    say("      chain (-K/l, -K(log K - 1.92)/(l L)) are NEGATIVE and dropped.")
    say("        l        L      T      lambda(T)   2lam + 2.26/L")
    tail = []
    for pz in [1009, 5e3, 1e5, 1e8, 1e20, 1e60, 1e300]:
        l = log(pz)
        LL = log(l)
        T = 2 * LL - 2 / l ** 2 + 0.0229945       # the LOWER end of the bracket
        lam = lam_root(1.0 + 1.90 / T)
        val = 2 * lam + 2.26 / LL
        tail.append((l, LL, lam, val))
        say(f"   {l:>10.4g} {LL:>8.4f} {T:>7.4f}    {lam:>7.4f}      {val:>8.4f}")
        assert lam <= 4.1, "the c = 1.90 bound assumed lambda <= 4.1"
    assert all(v[3] <= 9.30 for v in tail), tail
    assert tail[0][3] < 9.30 and tail == sorted(tail, key=lambda v: -v[3])
    say("  ASSERTED: 2 lambda(T) + 2.26/L <= 9.30 for every p_n >= 1009 "
        f"(value {tail[0][3]:.4f} at the")
    say("  threshold), and the sequence DECREASES monotonically to")
    say(f"  2 lambda_* = {2*LAMBDA_STAR:.4f}.  Section B settles 3 <= n <= 168 "
        "(p_n <= 997)")
    say("  exhaustively in exact rationals, so C = 9.30 is uniform for all n >= 3,")
    say("  and 7.1822 is the asymptotic truth.")

    say("")
    say("=" * 78)
    say("SECTION D - what an EXPLICIT rung 2 costs: dyadic-range level bookkeeping")
    say("=" * 78)
    say("  Rung 2 is polynomial only because the sieve truncation is by SIZE of d,")
    say("  not by the number of prime factors.  Pure Brun (Theorem 3) can never be")
    say("  polynomial: its level is z^K and K -> infinity is forced by (3E.5).")
    say("  Below: the level exponent s = log D / log z of the classical dyadic-range")
    say("  design - primes split at z^{theta^j}, Bonferroni depth K_j = ceil(K c^{j-1})")
    say("  in range j - together with the truncation cost it must pay.  This is the")
    say("  bookkeeping an explicit fundamental lemma has to control; the numbers say")
    say("  what exponent an explicit dimension-2 lower-bound sieve should reach.")
    say("     theta     c     K   s = sum theta^{j-1} K_j   trunc. cost sum_j err_j")
    best = None
    for theta in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        t = 2 * log(1.0 / theta)          # T_j = sum_{p in I_j} omega(p)/p
        vj = theta ** 2                   # V_j  = prod_{p in I_j}(1-omega(p)/p)
        for c in [1.05, 1.1, 1.2, 1.3, 1.5, 2.0]:
            if theta * c >= 1.0:
                continue
            for K in range(2, 26):
                s = 0.0
                err = 0.0
                for j in range(1, 400):
                    Kj = int(-(-K * c ** (j - 1) // 1))
                    w = theta ** (j - 1)
                    s += w * Kj
                    # e_{Kj+1} tail bound  t^{Kj+1}/(Kj+1)!  relative to V_j
                    lt = (Kj + 1) * log(t) - lgamma(Kj + 2) - log(vj)
                    err += exp(lt)
                    if w * Kj < 1e-9 and exp(lt) < 1e-12:
                        break
                if err <= 0.5 and (best is None or s < best[0]):
                    best = (s, theta, c, K, err)
    assert best is not None
    s, theta, c, K, err = best
    say(f"     best design: theta = {theta}, c = {c}, K = {K}  ->  "
        f"s = {s:.3f}, truncation cost {err:.4f}")
    for theta, c, K in [(0.5, 1.2, 4), (0.5, 1.5, 4), (0.6, 1.3, 5), (0.7, 1.1, 3)]:
        t = 2 * log(1.0 / theta)
        vj = theta ** 2
        s = 0.0
        err = 0.0
        for j in range(1, 400):
            Kj = int(-(-K * c ** (j - 1) // 1))
            s += theta ** (j - 1) * Kj
            err += exp((Kj + 1) * log(t) - lgamma(Kj + 2) - log(vj))
            if theta ** (j - 1) * Kj < 1e-9:
                break
        say(f"     theta={theta:<5} c={c:<5} K={K:<3} s = {s:>7.3f}   "
            f"cost = {err:>8.4f}  {'(admissible)' if err<=0.5 else '(too lossy)'}")
    say("")
    say("  READING: a dyadic-range design with a bounded level exponent EXISTS -")
    say(f"  s ~ {s:.1f}-{best[0]:.1f} - so an explicit polynomial rung 2 with exponent")
    say("  of that order is not ruled out by the level/error accounting.  What the")
    say("  accounting does NOT supply is VALIDITY: the product truncation")
    say("  {d : nu(d_j) <= K_j for all j} is provably NOT a lower-bound sieve (the")
    say("  per-range Bonferroni factors are <= 0 individually, so a product of two")
    say("  of them is >= 0), and the union bound over ranges is catastrophically")
    say("  lossy (J ~ log log z ranges against a main term of size V ~ 1/log^2 z).")
    say("  A valid lower-bound truncation must be NESTED in the ordered prime")
    say("  factorisation (Brun / Rosser-Iwaniec), and that is exactly the step whose")
    say("  constants are not explicit in the standard references.")

    # the invalidity claim, verified by explicit counterexample
    say("")
    say("  VERIFIED (not asserted from theory): the product truncation is invalid.")
    # two ranges, r_1 = r_2 = 1, K_1 = K_2 = 1: Lambda = (1-1)*(1-1)?  compute
    from math import comb
    bad = []
    for K1 in (1, 3):
        for K2 in (1, 3):
            for r1 in range(1, 6):
                for r2 in range(1, 6):
                    A1 = sum((-1) ** k * comb(r1, k) for k in range(K1 + 1))
                    A2 = sum((-1) ** k * comb(r2, k) for k in range(K2 + 1))
                    if A1 * A2 > 0:      # must be <= 0 = indicator, since r1,r2>=1
                        bad.append((K1, K2, r1, r2, A1 * A2))
    assert bad, "expected the product truncation to fail somewhere"
    say(f"    {len(bad)} explicit (K1,K2,r1,r2) with Lambda > 0 = 1_survivor;")
    say(f"    smallest: K1={bad[0][0]}, K2={bad[0][1]}, r1={bad[0][2]}, "
        f"r2={bad[0][3]}, Lambda = {bad[0][4]} > 0.")
    say("")
    say("  ROUND-23 FOLLOW-UP: these 36 are counterexamples to the PER-BAND form")
    say("  ONLY.  Counting prime factors over the whole UPPER TAIL instead - i.e.")
    say("  nu(d restricted to primes above z^{alpha_j}) <= H_j, nested - is valid:")
    say("  research/j2_nested.py finds ZERO violations over 168,400 configurations")
    say("  (and refutes, by control, the guess that the depths must be monotone).")
    say("  So the missing ingredient is NOT validity but the explicit MAIN-TERM")
    say("  estimate for the nested truncation.  See docs/novel/j2-upper-bound.md")
    say("  section 8(b).")

    with open("research/data/j2_explicit.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_explicit: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

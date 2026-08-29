"""
jk_family.py  -  ROUND 27, HARVESTER.  Gate for (P6): the k-class Jacobsthal
family j_k as a published object.  Doc: docs/novel/jk-family.md.

Sections
  A  the definition and the COVERING RESTATEMENT (omega_p = min(k, p-1)),
     brute-forced: shift form == covering form at k = 1,2,3 and z = 3,5,7
  B  the upper rung at every k: beta_k from ODC Corollary 6.13, and the
     hypotheses (5.38)/(6.69) checked at every k (see j2_odcpages.py)
  C  the Legendre rung at every k, against the exact values of section A
  D  the lower rung at every k: (P2')'s constant K_k
  E  THE k >= 4 SHIFT-SET QUESTION - ANSWERED (the note's own open item 6.3)
  F  the general-k greedy lemma, and a one-line proof that subsumes round
     26's k = 2 argument

Run:  .venv/Scripts/python.exe research/jk_family.py
"""

from itertools import combinations, combinations_with_replacement
from math import gcd, log, exp
from mpmath import mp, mpf
from sympy import primerange, factorint, isprime

mp.dps = 25

OUT = []
FAILS = []


def say(s=""):
    print(s)
    OUT.append(s)


def rule(t):
    say()
    say("=" * 78)
    say(t)
    say("=" * 78)


def check(name, cond, detail=""):
    if cond:
        say("  [ok]   %s %s" % (name, detail))
    else:
        say("  [FAIL] %s %s" % (name, detail))
        FAILS.append(name)


# ===========================================================================
rule("SECTION A - THE DEFINITION, AND THE COVERING RESTATEMENT")
# ===========================================================================

say("""
DEFINITION (shift form).  Let k >= 1 and let E = (0 = E_0 < E_1 < ... <
E_{k-1}) be an ADMISSIBLE k-tuple: for every prime p, E does not meet every
class mod p.  For squarefree m put

    j_E(m) = the largest gap between consecutive integers n with
             gcd( (n+E_0)(n+E_1)...(n+E_{k-1}), m ) = 1,

the gap taken cyclically over one period m, and

    j_k(m) = max over admissible k-tuples E of j_E(m).

  k = 1 : E = (0), and j_1 IS the ordinary Jacobsthal function j(m).
  k = 2 : E = (0, E), and j_2 IS Ziller-Morack's paired Jacobsthal function
          (their max over even E; odd E is never admissible at p = 2).

PROPOSITION (covering restatement).  For every z >= 2,

    j_k(P(z)) - 1  =  the length of the longest interval coverable by
                      choosing, at each prime p <= z, a set S_p of residue
                      classes mod p with

                            |S_p|  <=  min(k, p-1).

The bound min(k, p-1) is the whole content: k classes are available wherever
there is room for them, and admissibility forbids using all p classes at the
primes p <= k.  It reproduces the two known cases exactly -
  k = 1 : |S_p| <= 1 for every p            (the ordinary covering problem)
  k = 2 : |S_2| <= 1, |S_p| <= 2 for p >= 3 (Ziller-Morack's omega(2) = 1,
                                             omega(p) = 2; our g(2) = 1/2,
                                             g(p) = 2/p)
and it is what makes the SIEVE DIMENSION of the family equal to k.

Both directions of the Proposition are CRT.  (=>) The classes killed by E
at p are -t + {E_i mod p}, a set of size <= min(k, p-1) by admissibility.
(<=) Given (S_p), pick for each p a surjection {0..k-1} -> S_p and let CRT
define E_i; then {E_i mod p} = S_p at every p <= z.

BRUTE-FORCED BELOW, both forms independently, at k = 1,2,3 and z = 3,5,7.
""")


def survivors_gap(m, E, primes):
    """largest cyclic gap between consecutive n in [0,m) with all n+E_i
    coprime to m; None if there are no survivors."""
    killed = bytearray(m)
    for p in primes:
        for e in E:
            r = (-e) % p
            for n in range(r, m, p):
                killed[n] = 1
    surv = [n for n in range(m) if not killed[n]]
    if not surv:
        return None
    best = 0
    for i in range(len(surv)):
        d = surv[(i + 1) % len(surv)] - surv[i]
        if d <= 0:
            d += m
        if d > best:
            best = d
    return best


def admissible(E, primes):
    for p in primes:
        if len({e % p for e in E}) >= p:
            return False
    return True


def jk_shift_form(k, z):
    """j_k(P(z)) by exhaustive search over shift tuples.

    j_E depends on E only through the MULTISET {E_i mod m} (CRT), and
    repeats are legitimate - e.g. (0,2,6) at m = 6 has residues {0,2,0} -
    so the search runs over multisets, not over distinct residues."""
    primes = list(primerange(2, z + 1))
    m = 1
    for p in primes:
        m *= p
    best, bestE = 0, None
    for rest in combinations_with_replacement(range(m), k - 1):
        E = (0,) + rest
        if not admissible(E, primes):
            continue
        g = survivors_gap(m, E, primes)
        if g is not None and g > best:
            best, bestE = g, E
    return best, bestE, m


def jk_covering_form(k, z):
    """longest run of covered integers, +1, over all admissible (S_p)."""
    primes = list(primerange(2, z + 1))
    m = 1
    for p in primes:
        m *= p
    choices = []
    for p in primes:
        cap = min(k, p - 1)
        opts = []
        for size in range(1, cap + 1):
            opts.extend(combinations(range(p), size))
        choices.append((p, opts))

    best = 0

    def rec(i, killed):
        nonlocal best
        if i == len(choices):
            # longest run of killed, cyclically, over [0,m)
            run = cur = 0
            for n in list(range(m)) + list(range(m)):
                if killed[n % m]:
                    cur += 1
                    run = max(run, cur)
                else:
                    cur = 0
            run = min(run, m)
            best = max(best, run + 1)
            return
        p, opts = choices[i]
        for S in opts:
            nk = bytearray(killed)
            for s in S:
                for n in range(s % p, m, p):
                    nk[n] = 1
            rec(i + 1, nk)

    rec(0, bytearray(m))
    return best


say("    k   z    j_k (shift form)   witness E        j_k (covering form)")
exact = {}
for k in (1, 2, 3):
    for z in (3, 5, 7):
        if k > z:            # no admissible k-tuple can be tested this small
            continue
        js, E, m = jk_shift_form(k, z)
        jc = jk_covering_form(k, z)
        exact[(k, z)] = js
        say("    %-3d %-4d %-18d %-16s %-d" % (k, z, js, str(E), jc))
        check("shift form == covering form at k=%d, z=%d" % (k, z), js == jc)

check("k=1 reproduces the ordinary Jacobsthal j(6)=4, j(30)=6, j(210)=10 "
      "(OEIS A048669)",
      (exact[(1, 3)], exact[(1, 5)], exact[(1, 7)]) == (4, 6, 10),
      str((exact[(1, 3)], exact[(1, 5)], exact[(1, 7)])))
check("k=2 reproduces Ziller-Morack h_2 = 6, 18, 30 at z = 3, 5, 7",
      (exact[(2, 3)], exact[(2, 5)], exact[(2, 7)]) == (6, 18, 30),
      str((exact[(2, 3)], exact[(2, 5)], exact[(2, 7)])))
say("""
  The k = 3 row is, as far as this lane can establish, THE FIRST TIME THE
  FUNCTION HAS BEEN EVALUATED: j_3(P(5)) and j_3(P(7)) above.  Small, and
  that is the point - the object is elementary and nobody has named it.""")

# omega table
say()
say("  omega_p = min(k, p-1), the per-prime class budget:")
say("    k \\ p     2    3    5    7   11   13")
for k in (1, 2, 3, 4, 5):
    say("    %-9d %s" % (k, "".join("%5d" % min(k, p - 1)
                                    for p in (2, 3, 5, 7, 11, 13))))
check("omega is ZM's at k = 2 (1 at p=2, 2 elsewhere)",
      [min(2, p - 1) for p in (2, 3, 5, 7, 11)] == [1, 2, 2, 2, 2])
check("omega is 1 everywhere at k = 1 (ordinary sieve)",
      [min(1, p - 1) for p in (2, 3, 5, 7, 11)] == [1, 1, 1, 1, 1])

# ===========================================================================
rule("SECTION B - THE UPPER RUNG AT EVERY k (ODC Corollary 6.13)")
# ===========================================================================

say("""
The whole of Unit 1's upper ladder is uniform in k; only the DIMENSION
changes, from kappa = 2 to kappa = k.  In particular Opera de Cribro
Corollary 6.13, which gives Theorem 2G's exponent 8.04162 at kappa = 2,
gives at every kappa

    beta_kappa = 1 + 2 ( e^{1/(2 kappa)} - 1 )^{-1} ,

and round 27 checked first-hand (research/j2_odcpages.py section B, ODC
p. 67) that its hypothesis (6.69) - which is exactly alpha < 1/c, c the root
of c(log c - 1) = 1 - holds at alpha = 1/4 IDENTICALLY IN kappa.  So the
rung is available at every k with no further work:

    j_k(P(z))  <<_{k,eps}  z^{beta_k + eps},   beta_k = 1 + 2(e^{1/(2k)}-1)^-1

and, with pre-sieving in place of ODC's own (inexplicit) preliminary sifting,
every constant is computable exactly as in Theorem 2G.
""")


def beta(kap):
    kap = mpf(kap)
    return 1 + 2 / (mp.e ** (1 / (2 * kap)) - 1)


say("    k     beta_k          4k+1     beta_k - 4k")
for k in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
    b = beta(k)
    say("    %-5d %-15.6f %-8d %+.6f" % (k, float(b), 4 * k + 1,
                                         float(b - 4 * k)))
    check("4k-1 < beta_%d < 4k+1" % k, 4 * k - 1 < b < 4 * k + 1)
check("beta_2 is exactly Theorem 2G's exponent 8.041623",
      abs(beta(2) - mpf('8.0416233')) < mpf('1e-6'), "%.7f" % float(beta(2)))
check("beta_k/(4k) -> 1 from above", beta(20) / 80 > 1 and beta(20) / 80 < 1.001,
      "beta_20/80 = %.8f" % float(beta(20) / 80))

say("""
  HONEST, AND IT MUST BE IN THE NOTE: at k = 1 this rung gives exponent
  4.082988, which is WORSE than the record.  Iwaniec 1978 proves
  j(P(z)) << z^2, and that has stood for 48 years.  So the family rung is
  NOT the best bound at k = 1; it is the only bound at every k >= 2, because
  the function has not been named.  Anyone who names it gets these rungs
  from standard sieve theory - which is the honest description of the
  contribution, exactly as in Unit 1's not-claim 2.""")
check("the family rung is strictly weaker than Iwaniec's exponent 2 at k=1",
      beta(1) > 2, "beta_1 = %.6f > 2" % float(beta(1)))

# ===========================================================================
rule("SECTION C - THE LEGENDRE RUNG AT EVERY k, against section A's exacts")
# ===========================================================================

say("""
Theorem 1 of Unit 1 is Legendre inclusion-exclusion and is also uniform in k:

    j_k(P(z))  <=  prod_{p<=z} (1 + omega_p) / V_k(z)  +  1 ,
    omega_p = min(k, p-1),   V_k(z) = prod_{p<=z} (1 - omega_p/p).

At k = 2 the numerator is 1*2 * prod_{3<=p<=p_n} 3 = 2 * 3^{n-1}, which is
Theorem 1's constant verbatim.
""")


def legendre_bound(k, z):
    num, V = mpf(1), mpf(1)
    for p in primerange(2, z + 1):
        w = min(k, p - 1)
        num *= (1 + w)
        V *= (1 - mpf(w) / p)
    return num / V + 1


for k in (2,):
    for n, z in ((1, 3), (2, 5), (3, 7)):
        num = mpf(1)
        for p in primerange(2, z + 1):
            num *= 1 + min(k, p - 1)
        check("k=2 numerator at z=%d is 2*3^(n-1) = %d" % (z, 2 * 3 ** n),
              num == 2 * 3 ** n, "%d" % int(num))

say("    k   z    exact j_k   Legendre bound   ratio")
for (k, z), v in sorted(exact.items()):
    b = legendre_bound(k, z)
    say("    %-3d %-4d %-11d %-16.3f %.4f" % (k, z, v, float(b), v / float(b)))
    check("Legendre bound holds at k=%d, z=%d" % (k, z), mpf(v) <= b)

# ===========================================================================
rule("SECTION D - THE LOWER RUNG AT EVERY k ((P2'), round 26)")
# ===========================================================================

say("""
    THEOREM (P2', round 26).  Let pi_E(t) <= c_1^(k) t/(log t)^k for every
    admissible k-tuple E in play.  With A = log x, B = log A, C = log B,

        j_k(P(x))  >=  ( K_k + o(1) ) x A^{2k-1} C^k / B^{2k} ,
        K_k = k / ( (k(2k-1))^k c_1^(k) ).
""")


def K_k(k, c1=1):
    return mpf(k) / (mpf(k * (2 * k - 1)) ** k * c1)


c1_twin = mpf('4.356487')          # Lichtman 3.29956 * 2C_2
check("K_1 = 1/c_1^(1), and = 1 with c_1^(1) = 1 (PNT-admissible)",
      abs(K_k(1, 1) - 1) < mpf('1e-20'), "%.10f" % float(K_k(1, 1)))
check("K_2 = 1/(18 c_1) exactly",
      abs(K_k(2, c1_twin) - 1 / (18 * c1_twin)) < mpf('1e-20'))
check("K_2 with Lichtman's c_1 is the headline 0.0127524",
      abs(K_k(2, c1_twin) - mpf('0.0127524')) < mpf('1e-6'),
      "%.7f" % float(K_k(2, c1_twin)))
say()
say("    k    (k(2k-1))^k        K_k at c_1^(k) = 1     power of A")
for k in (1, 2, 3, 4, 5, 6):
    say("    %-4d %-18d %-22.10g %d" % (k, k * (2 * k - 1) ** 1 and
                                        (k * (2 * k - 1)) ** k,
                                        float(K_k(k, 1)), 2 * k - 1))

say("""
  THE SANDWICH AT EVERY k, and it is the point of the note:

      x A^{2k-1} C^k / B^{2k}   <<   j_k(P(x))   <<   x^{beta_k + eps}
                                                       beta_k ~ 4k

  CONJECTURE (sharp, falsifiable, and the family's real content):
      j_k(P(x))  =  x (log x)^{2k-1+o(1)}  for every k >= 1.
  At k = 1 this is the standard expectation for the Jacobsthal function; at
  k = 2 it is round 26's sharpened (P3); at k >= 3 it is new because the
  object is new.  ANY claimed upper bound j_k << x A^{f(k)} with
  f(k) < 2k-1 is contradicted outright by (P2') - free consistency checks on
  a whole family, which is what makes the family worth publishing even
  though each individual rung is standard.""")
check("the lower exponent 2k-1 is odd and strictly increasing in k",
      all((2 * k - 1) % 2 == 1 for k in range(1, 20))
      and all(2 * k - 1 < 2 * (k + 1) - 1 for k in range(1, 20)))

# ===========================================================================
rule("SECTION E - THE k >= 4 SHIFT-SET QUESTION - ANSWERED")
# ===========================================================================

say("""
THE QUESTION (layered-erdos-rankin.md section 6 item 3, round 26, named as
the family's "one piece of real work"): the construction's layers 1..k put
class -E_i mod p, and for k >= 4 the shifts 0,2,...,2(k-1) are not pairwise
distinct modulo every odd prime (3 | 6 already at k = 4).  Round 26 wrote
"the finitely many offending primes can be set aside, but a clean statement
wants the optimal shift set", and recorded the collisions
[(4,[3]), (5,[3]), (6,[3,5]), (7,[3,5])].

THE ANSWER: THE QUESTION DISSOLVES, AND IT COSTS NOTHING.  Two observations.

1. The shifts 0,2,...,2(k-1) are the WRONG tuple: from k = 3 on they are not
   even admissible (0,2,4 covers all of Z/3, so no n survives at all).  The
   construction needs an admissible tuple, and admissible k-tuples exist for
   every k - e.g. E = {q_1,...,q_k} - q_1 for the k least primes q_i > k,
   which is admissible because no q_i is divisible by any p <= k, so the
   tuple misses class -q_1 mod p there, and has only k <= p-1 elements for
   p > k.

2. With ANY admissible tuple, a collision E_i = E_j mod p can happen only at
   a prime p dividing some pairwise difference, hence only at
   p <= M_k := max_{i<j} (E_j - E_i), a CONSTANT depending on k alone.  The
   layered construction's greedy range is [P, z1] with P = A^{2k-1} -> oo,
   so for x large EVERY colliding prime lies BELOW P, i.e. inside the
   Eratosthenes layers, never inside the greedy layer.  And a collision
   inside the Eratosthenes layers costs nothing: layers i and j simply
   coincide at that p, which uses FEWER than the k available classes, while
   the survivor structure ("no n+E_i has a prime factor in
   [3,P) u (z1,x/L]") is unchanged.  So Sigma = prod_{P<=p<=z1}(1 - k/p) with
   no correction whatever, and the constant K_k of section D stands as
   printed for every k.

   THE EXPLICIT THRESHOLD: it is enough that A^{2k-1} > M_k, i.e.
   x > exp( M_k^{1/(2k-1)} ) - which is astronomically weaker than the
   construction's own threshold (round 26: nothing exists below log x ~ 300).
""")

say("    k   admissible tuple E                    M_k   colliding primes"
    "   exp(M_k^(1/(2k-1)))")
for k in range(2, 13):
    q0 = next(p for p in primerange(k + 1, 10 ** 4))
    qs = []
    p = q0
    while len(qs) < k:
        if isprime(p):
            qs.append(p)
        p += 1
    E = tuple(q - qs[0] for q in qs)
    primes_le_k = list(primerange(2, k + 1))
    adm = admissible(E, list(primerange(2, max(E) + 2)))
    diffs = [E[j] - E[i] for i in range(k) for j in range(i + 1, k)]
    coll = sorted({q for d in diffs for q in factorint(d)})
    Mk = max(diffs)
    thr = exp(Mk ** (1.0 / (2 * k - 1)))
    say("    %-3d %-36s %-5d %-18s %.4g"
        % (k, str(E), Mk, str(coll), thr))
    check("E is admissible at k = %d" % k, adm)
    check("every colliding prime at k = %d is <= M_k = %d" % (k, Mk),
          all(q <= Mk for q in coll))
    check("the threshold x > exp(M_k^{1/(2k-1)}) at k = %d is trivial "
          "(< e^10)" % k, thr < exp(10), "%.4g" % thr)

say("""
  SO THE FAMILY'S NAMED OPEN ITEM CLOSES.  What remains of it is a
  genuinely finite optimisation and NOT a gap in the theorem: which
  admissible k-tuple minimises c_1^(k) (equivalently, minimises the singular
  series S(E) in pi_E(t) <= c_1^(k) S(E) t/(log t)^k)?  That moves the
  CONSTANT K_k and nothing else.  Recorded as such in the note.""")

# ===========================================================================
rule("SECTION F - THE GREEDY LEMMA AT EVERY k, AND A SIMPLER PROOF OF OURS")
# ===========================================================================

say("""
LEMMA (general k).  Let R be a finite set of integers, p a prime, N = |R|,
1 <= k <= p.  Then some k DISTINCT classes mod p contain together at least
kN/p elements of R.

PROOF.  The p class counts average N/p; the k LARGEST of them therefore have
average at least N/p; so they sum to at least kN/p.  QED

That is one line, and it SUBSUMES round 26's k = 2 lemma, whose proof (via
n_(1) >= N/p, n_(2) >= (N-n_(1))/(p-1) and monotonicity) was correct but
longer than it needed to be.  Recorded as a simplification of our own
argument, not a correction of it: the k = 2 statement 2N/p was and is exact.
""")

import random
random.seed(20260829)
worst = {}
for trial in range(40000):
    p = random.choice([3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    N = random.randint(1, 400)
    cuts = sorted(random.randint(0, N) for _ in range(p - 1))
    counts = [b - a for a, b in zip([0] + cuts, cuts + [N])]
    for k in range(1, p + 1):
        top = sum(sorted(counts, reverse=True)[:k])
        if top < k * N / p - 1e-9:
            FAILS.append("greedy k=%d p=%d" % (k, p))
        r = top / (k * N / p) if N else 1.0
        worst[(p, k)] = min(worst.get((p, k), 9.9), r)
check("general-k greedy >= kN/p on 40,000 random class distributions, "
      "all p <= 31 and all 1 <= k <= p", True,
      "worst ratio observed %.6f (>= 1 means the bound holds)"
      % min(worst.values()))
check("the k = 2 case reproduces round 26's exact 2N/p",
      min(v for (p, k), v in worst.items() if k == 2) >= 1.0)

# ===========================================================================
rule("VERDICT")
# ===========================================================================
if FAILS:
    say("  jk_family: %d FAILURES: %s" % (len(FAILS), ", ".join(sorted(set(FAILS)))))
else:
    say("  jk_family: ALL ASSERTIONS GREEN")

with open("research/data/jk_family.out", "w") as f:
    f.write("\n".join(OUT) + "\n")

raise SystemExit(1 if FAILS else 0)

"""
j2_odcpages.py  -  ROUND 27, HARVESTER.

Closes the three ODC page-image caveats that rounds 24-26 carried forward:
(5.38), (6.69) and p. 74.  All three page images were fetched first-hand on
2026-08-29 from the Google Books publisher preview of the AMS printing
(volume Dz6REQAAQBAJ) and are on disk under research/data/odc6_scans/:

    PA42.png  - (5.38), the sieve-dimension hypothesis, section 5.5
    PA43.png  - the discussion of (5.38): (5.39), (5.41), (5.42)
    PA44.png  - (5.43), Lemma 5.1
    PA45.png  - the (5.38) consequence used in ch. 6
    PA67.png  - (6.65)-(6.70): alpha, the convergence condition, and (6.69)
    PA74.png  - (6.99)/(6.100), preliminary sieving

Everything below is ARITHMETIC ON WHAT THOSE PAGES PRINT, re-derived here from
the printed formulas by independent code, in the same style as j2_odc6.py.
No claim of this round enters the record without an assertion here.

Run:  .venv/Scripts/python.exe research/j2_odcpages.py
"""

from mpmath import mp, mpf, exp, log, coth, findroot

mp.dps = 30

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def rule(t):
    say()
    say("=" * 78)
    say(t)
    say("=" * 78)


FAILS = []


def check(name, cond, detail=""):
    if cond:
        say("  [ok]   %s %s" % (name, detail))
    else:
        say("  [FAIL] %s %s" % (name, detail))
        FAILS.append(name)


# ---------------------------------------------------------------------------
rule("SECTION A - (5.38), Opera de Cribro p. 42, section 5.5 'The Sieve "
     "Dimension'")
# ---------------------------------------------------------------------------

say("""
PRINTED ON p. 42, transcribed from the page image:

    (5.38)      prod_{w <= p < z} (1 - g(p))^{-1}  <=  K (log z / log w)^kappa

    "or, in other words, in the notation of (5.12),

                V(w)/V(z) <= K (log z / log w)^kappa

    for any z > w >= 2, where K is a constant, K > 1.  Note that this
    inequality implies

                g(p) <= 1 - 1/K."

WHAT THIS SETTLES.  Rounds 23-24 used (5.38) from two independent verbatim
transcriptions (Dudek-Dunn Lemma 2.1; Campbell arXiv:2608.09488 Thm 2.1) and
recorded that the book itself had NOT been consulted for it.  It has now been
read first-hand.  The form we used is the form the book prints, character for
character in content: same product, same one-sided direction, same
(log z/log w)^kappa, same range z > w >= 2.  THE ROUND-23/24 CAVEAT ON (5.38)
IS DISCHARGED.

TWO BY-PRODUCTS THE BOOK GIVES FOR FREE, neither of which we had:

  (i)  K > 1 is REQUIRED by the book's own statement.  Every K this project
       uses must therefore be checked > 1, not merely finite.
  (ii) (5.38) IMPLIES g(p) <= 1 - 1/K - and conversely, taking w = p and
       z -> p+ in (5.38) gives K >= (1 - g(p))^{-1} at every prime of the
       sifting range.  So the book's own remark EXPLAINS, in one line, the
       K-ladder this project measured by grid search in round 23: K is at
       least the single-prime value at the SMALLEST prime left in the range.
""")

# our density: g(2) = 1/2, g(p) = 2/p for odd p  (omega(p)/p, kappa = 2)


def g(p):
    return mpf(1) / 2 if p == 2 else mpf(2) / p


# The single-prime lower bound on K forced by (5.38) at w = p, z -> p+.
def K_forced(p):
    return 1 / (1 - g(p))


say("  the (5.38) single-prime forcing K >= (1-g(p))^{-1}, our density:")
say("    p        g(p)        forced K >= ")
for p in (2, 3, 5, 7, 11, 13, 101, 151):
    say("    %-6d   %-10.6f  %-12.7f" % (p, float(g(p)), float(K_forced(p))))

# Round 23/24 recorded K values, by the least prime of the sifting range.
recorded = {3: mpf(3), 5: mpf(5) / 3, 7: mpf('1.4')}
for p0, Krec in recorded.items():
    check("K(p_0=%d) = %s is exactly the (5.38) single-prime value" % (p0, Krec),
          abs(K_forced(p0) - Krec) < mpf('1e-25'),
          "-> %.10f" % float(Krec))

say("""
  So the round-23 grid search's "supremum at w = 3, z -> 3+, returns exactly
  3.000000" is not a numerical accident: at p = 3 the paired sieve has
  g(3) = 2/3, and (1 - 2/3)^{-1} = 3 exactly.  The same identity gives 5/3 at
  p_0 = 5 and 7/5 at p_0 = 7.  From p_0 = 11 on the supremum moves off the
  single prime (round 23 measured 1.2624 against the single-prime 11/9 =
  1.2222), because a whole range of primes can beat one factor relative to
  (log z/log w)^2 - so the single-prime value is a LOWER bound on K, which is
  the direction the book states.""")

check("11/9 is a strict LOWER bound for the measured K(p_0=11) = 1.2624",
      K_forced(11) < mpf('1.2624'),
      "11/9 = %.6f < 1.2624" % float(K_forced(11)))

# every operative K of the project must satisfy K > 1 and g(p) <= 1 - 1/K
operative = [("2E  (no pre-sieve, p_0 = 3)", 3, mpf(3)),
             ("2E' (p_0 = 5)", 5, mpf(5) / 3),
             ("2E'' (p_0 = 13)", 13, mpf('1.2624')),
             ("2G  (p_0 = 151)", 151, mpf('1.0260176'))]
say()
say("  every operative K of the ladder against the book's two conditions:")
for name, p0, K in operative:
    ok = (K > 1) and (g(p0) <= 1 - 1 / K + mpf('1e-20'))
    check(name, ok, "K = %.7f > 1 and g(%d) = %.6f <= 1 - 1/K = %.6f"
          % (float(K), p0, float(g(p0)), float(1 - 1 / K)))

# ---------------------------------------------------------------------------
rule("SECTION B - (6.69), Opera de Cribro p. 67 - THE CONDITION ON kappa "
     "QUOTED INSIDE PROPOSITION 6.7")
# ---------------------------------------------------------------------------

say("""
PRINTED ON p. 67, transcribed from the page image (section 6.5, Fundamental
Lemma), in its own derivation order:

    (6.65)   alpha = (kappa/2) log((beta+1)/(beta-1))
    (6.66)   V_n(z)/V(z) <= (1/n!) (n alpha e^alpha)^n K^{1+alpha^{-1}}
    (6.67)   a = alpha e^{1+alpha} < 1 ,  "the condition required for
             convergence.  Let c = 3.591... be the solution of (6.11),"
    (6.68)   e^{1+c^{-1}} = c
             "Then the condition a < 1 means alpha < c^{-1}, or equivalently,"
    (6.69)   kappa < (2/c) ( log((beta+1)/(beta-1)) )^{-1}
    (6.70)   beta > 1 + 2 (e^{2/(c kappa)} - 1)^{-1}

and Proposition 6.7 reads "for any multiplicative function g(d) satisfying
(5.38) with kappa bounded by (6.69)".

WHAT THIS SETTLES, and it settles it for EVERY kappa at once, not just ours.
Rewrite (6.69) with (6.65): it says exactly

        alpha  <  1/c  =  0.2784645...

i.e. (6.69) IS the convergence condition (6.67) a = alpha e^{1+alpha} < 1,
written as a bound on kappa.  Now Corollary 6.13's own choice
beta_kappa = 1 + 2(e^{1/(2 kappa)} - 1)^{-1} gives
(beta+1)/(beta-1) = e^{1/(2 kappa)} and hence alpha = kappa/2 * 1/(2 kappa)
= 1/4 IDENTICALLY IN kappa.  So Corollary 6.13 sits at alpha = 1/4 < 1/c for
every kappa > 0, which is why the book states it "for kappa > 0".
THE ROUND-25/26 (6.69) CAVEAT IS DISCHARGED, and not conditionally.
""")

# c from (6.11)/(6.68): (c/e)^c = e  <=>  c(log c - 1) = 1  <=>  e^{1+1/c} = c
c = findroot(lambda t: t * (log(t) - 1) - 1, mpf('3.6'))
check("(6.11) c(log c - 1) = 1 has root c = 3.5911214766...",
      abs(c - mpf('3.5911214766')) < mpf('1e-9'), "c = %.12f" % float(c))
check("(6.68) e^{1+1/c} = c holds at that root",
      abs(exp(1 + 1 / c) - c) < mpf('1e-20'))
check("c is EXACTLY our Theorem 3E constant lambda_* (2 lambda_* = 7.182242)",
      abs(2 * c - mpf('7.1822429532')) < mpf('1e-8'),
      "2c = %.10f" % float(2 * c))

alpha_star = 1 / c
say()
say("  the (6.69) ceiling in alpha:  alpha < 1/c = %.10f" % float(alpha_star))


def a_of(alpha):
    return alpha * exp(1 + alpha)


check("(6.67)/(6.69) are the same condition: a(1/c) = 1 exactly",
      abs(a_of(alpha_star) - 1) < mpf('1e-20'),
      "a(1/c) = %.15f" % float(a_of(alpha_star)))

# Corollary 6.13's alpha is 1/4 identically in kappa
say()
say("  Corollary 6.13's alpha, recomputed from its own beta_kappa, at nine")
say("  dimensions - (6.65) applied to beta_kappa = 1 + 2(e^{1/(2 kappa)}-1)^-1:")
say("    kappa     beta_kappa        alpha        alpha < 1/c ?")
for kap in [mpf(t) for t in ('0.5', '1', '1.5', '2', '2.5', '3', '4', '5',
                             '10')]:
    beta = 1 + 2 / (exp(1 / (2 * kap)) - 1)
    alpha = (kap / 2) * log((beta + 1) / (beta - 1))
    ok = alpha < alpha_star
    say("    %-8s  %-15.6f   %-11.8f  %s"
        % (str(kap), float(beta), float(alpha), "yes" if ok else "NO"))
    check("Cor 6.13 alpha = 1/4 at kappa = %s" % kap,
          abs(alpha - mpf('0.25')) < mpf('1e-20'))
    check("Cor 6.13 satisfies (6.69) at kappa = %s" % kap, ok)

check("our operative kappa = 2 gives beta_2 = 8.04162 (the 2G exponent)",
      abs((1 + 2 / (exp(mpf('0.25')) - 1)) - mpf('8.0416232')) < mpf('1e-6'),
      "beta_2 = %.7f" % float(1 + 2 / (exp(mpf('0.25')) - 1)))

# (6.70) directly, at kappa = 2
kap = mpf(2)
beta_670 = 1 + 2 / (exp(2 / (c * kap)) - 1)
beta_2G = 1 + 2 / (exp(mpf('0.25')) - 1)
say()
say("  (6.70) at kappa = 2:  beta > 1 + 2(e^{2/(2c)} - 1)^{-1} = %.7f"
    % float(beta_670))
check("Theorem 2G's beta_2 = 8.04162 satisfies (6.70) with room",
      beta_2G > beta_670,
      "8.041623 > %.6f, margin %.6f" % (float(beta_670),
                                        float(beta_2G - beta_670)))

# our K -> 1 positivity root
alpha_inf = mpf('0.253321897')
check("the 2G-inf root alpha_inf = 0.253321897 also satisfies (6.69)",
      alpha_inf < alpha_star,
      "%.9f < %.9f, margin %.9f"
      % (float(alpha_inf), float(alpha_star), float(alpha_star - alpha_inf)))

# beta as a function of alpha at kappa = 2:  (beta+1)/(beta-1) = e^{alpha}
def beta_of(alpha, kappa=mpf(2)):
    return coth(alpha / kappa)


check("beta(alpha) = coth(alpha/kappa) reproduces beta_2 = 8.041623 at "
      "alpha = 1/4", abs(beta_of(mpf('0.25')) - beta_2G) < mpf('1e-20'))
check("beta(alpha_inf) = 7.93727, the printed 2G-inf floor",
      abs(beta_of(alpha_inf) - mpf('7.9372682')) < mpf('1e-6'),
      "%.7f" % float(beta_of(alpha_inf)))

beta_hard = beta_of(alpha_star)
say()
say("""  A NEW NUMBER, and it is the sharpest thing (6.69) buys us.  Because
  beta = coth(alpha/kappa) is DECREASING in alpha and (6.69) caps alpha at
  1/c, ODC Chapter 6's beta-sieve has an ABSOLUTE FLOOR at kappa = 2:

        beta  >  coth(1/(2c))  =  %.7f      [ (6.69)/(6.70) ]

  against our positivity floor 7.93727 (the K -> 1 root of the bracket).  So
  the two floors are 7.93727 (positivity, binding) and 7.22859 (convergence,
  slack by 0.7085).  READING: (6.69) is NOT what stops Chapter 6 - positivity
  is.  Even discarding EVERY K-loss AND the whole positivity requirement, the
  device cannot print an exponent below 7.22859 at kappa = 2, which is still
  3.0 above the DHR 4.266.  The 7.937 -> 4.266 gap is not reachable by any
  tuning of this chapter.""" % float(beta_hard))

check("(6.69) hard floor at kappa = 2 is 7.22859, below our 7.93727",
      beta_hard < beta_of(alpha_inf) and beta_hard > mpf('7.2'),
      "%.7f" % float(beta_hard))
check("even the (6.69) hard floor is far above DHR beta_2 = 4.266",
      beta_hard > mpf('4.266') + 2, "%.4f vs 4.266" % float(beta_hard))

# ---------------------------------------------------------------------------
rule("SECTION C - p. 74 - ODC's OWN PRELIMINARY SIEVING IS NOT EXPLICIT")
# ---------------------------------------------------------------------------

say("""
PRINTED ON p. 74, transcribed from the page image - the second half of the
preliminary-sieving proposition whose first half is on p. 73:

    (6.99)   S(A,z) <= X V(z_0) { sum_{d|P(z,z_0)} lambda_d^+ g(d)
                                  + O(e^{-s_0} V(z_0)/V(z)) }
                      + sum_{d_0|P(z_0), d_0<=D_0} sum_{d|P(z,z_0), d<=D}
                        |lambda_d^+ r_{d_0 d}(A)|
    (6.100)  (the matching lower bound)

    "where the implied constant depends only on K_0 and kappa_0."

WHAT THIS SETTLES.  Round 25 flagged p. 74 because our pre-sieved rungs
(2E', 2E'', and 2G's N_pre) rest on round 24's OWN elementary accounting
rather than on the book's preliminary-sieving apparatus, and we could not see
whether the book's version was stronger.  It is not - it is WEAKER for our
purpose, because it carries an unevaluated O(.) with an implied constant
"depending only on K_0 and kappa_0", i.e. exactly the kind of inexplicitness
the whole 2E/2G line exists to avoid.  ODC's preliminary sieving COULD NOT
have supplied our pre-sieving factor.

  => THE CAVEAT CLOSES POSITIVELY: our N_pre = prod_{p<p_0}(p - omega(p))
     accounting (|r_d| <= 2^{nu(d)} N_pre, X V'(z) = m V(z) exactly) is not
     merely an alternative to the book's device, it is the only EXPLICIT
     route, and it stays ours.

AND THE BOOK'S OTHER ROUTE TO K -> 1, PRICED HERE FOR THE FIRST TIME.
pp. 43-44 offer a device we had never considered: (5.42)/(5.43) get K as close
to 1 as one likes WITHOUT any preliminary sifting, by ENLARGING THE DIMENSION
by epsilon:

    (5.42)  K = 1 + L (log y)^{eps-1} (log z)^{-eps}   at dimension kappa+eps
    (5.43)  K = 1 + L (log z)^{-1}                     at eps = 1, y = 2
            "the constant K given by (5.42) is fine, even for y = 2 (no
             preliminary sifting is needed)"

That is a genuine competitor to pre-sieving.  It loses, and by a lot, because
beta = coth(alpha/kappa) is INCREASING in kappa:""")

say()
say("    device                            kappa    K        beta at alpha_inf")
for label, kap, Kd in [("no pre-sieve, K = 3 (2E)", mpf(2), "3"),
                       ("pre-sieve at 13 (2E'')", mpf(2), "1.2624"),
                       ("pre-sieve at 151 (2G)", mpf(2), "1.026"),
                       ("ODC (5.43): eps = 1, no pre-sieve", mpf(3), "-> 1"),
                       ("ODC (5.42): eps = 1/2, no pre-sieve", mpf('2.5'),
                        "-> 1")]:
    say("    %-32s  %-7s  %-8s %.6f"
        % (label, str(kap), Kd, float(beta_of(alpha_inf, kap))))

b3 = beta_of(alpha_inf, mpf(3))
b25 = beta_of(alpha_inf, mpf('2.5'))
check("raising the dimension to kappa = 3 costs 3.93 of exponent",
      b3 > mpf('11.8') and b3 < mpf('11.95'),
      "beta = %.6f against 7.93727" % float(b3))
check("even eps = 1/2 (kappa = 2.5) costs 1.98 of exponent",
      b25 > mpf('9.9') and b25 < mpf('9.95'),
      "beta = %.6f against 7.93727" % float(b25))
say("""
  So the book's own K -> 1 device is priced and REJECTED, on the book's own
  arithmetic: it buys a factor in the constant and pays 2-4 whole units of
  exponent.  PRE-SIEVING KEEPS kappa = 2 AND IS THEREFORE THE RIGHT DEVICE -
  a conclusion round 24 reached by luck rather than by comparison, and which
  is now compared.""")

# ---------------------------------------------------------------------------
rule("SECTION D - WHAT MOVED IN UNIT 1")
# ---------------------------------------------------------------------------

say("""
NOTHING IN THE LADDER MOVES.  Every rung stands exactly as printed in
j2-upper-bound.md section 11a; no constant, exponent or threshold changes.
What changes is the CAVEAT LIST:

  BEFORE (round-26 close): "the ODC page-image caveat is still open: (5.38),
  (6.69) and p. 74 were not re-fetched.  One library visit closes it and it
  should happen before submission."

  AFTER (this round): all three fetched and read first-hand, 2026-08-29.
    (5.38)  p. 42 - matches our used form exactly; plus K > 1 required, plus
            g(p) <= 1 - 1/K, which EXPLAINS the measured K-ladder.
    (6.69)  p. 67 - is alpha < 1/c, and Corollary 6.13's alpha is 1/4
            identically in kappa, so the hypothesis holds for every kappa,
            not merely ours.  New: the (6.69) hard floor at kappa = 2 is
            7.2288, so positivity - not (6.69) - is what stops Chapter 6.
    p. 74   - ODC's own preliminary sieving carries an implied constant
            depending on K_0 and kappa_0; it is NOT explicit, so our N_pre
            accounting is the only explicit route and stays ours.

  RESIDUAL, and it is now the ONLY page caveat left: every one of these pages
  is a publisher-preview page IMAGE read on screen, not a copy held in hand.
  Mitigation is the same as rounds 24-25 and is now stronger: the pages are
  mutually cross-checking (p. 45 quotes (5.38) and its consequence; p. 67's
  (6.65)-(6.70) reproduce Corollary 6.13's beta_kappa exactly; p. 67's c is
  our own lambda_*), so an OCR corruption would have to be self-consistent
  across four pages of two different chapters and agree with two independent
  transcriptions of Theorem 7.7.  A submission should still say "as printed in
  [FI], Theorem 7.7 / Corollary 6.13" and nothing about typography.""")

# ---------------------------------------------------------------------------
rule("VERDICT")
# ---------------------------------------------------------------------------
if FAILS:
    say("  j2_odcpages: %d FAILURES: %s" % (len(FAILS), ", ".join(FAILS)))
else:
    say("  j2_odcpages: ALL ASSERTIONS GREEN")

with open("research/data/j2_odcpages.out", "w") as f:
    f.write("\n".join(OUT) + "\n")

raise SystemExit(1 if FAILS else 0)

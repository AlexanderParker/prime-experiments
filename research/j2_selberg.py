"""Harvester round 24 (item a, bullet 3): THE 19/36 vs 0.4454 CONFLICT, SETTLED.

Round 23 flagged rather than picked, which was right at the time:

  * C. S. Franze, "Sifting Limits for the Lambda^2 Lambda^- Sieve", J. Number
    Theory 131 (2011) 1962-1982 (arXiv:1012.3809), attributes to Selberg
        beta_kappa  <~  2 kappa + 19/36        (kappa -> infinity);
  * Kevin Ford, "Sieve Methods Lecture Notes, Spring 2023", section 3.1, and
    Z. E. Brady, "Sieves and Iteration Rules" (Stanford PhD, June 2017),
    Theorem 2, both attribute to the SAME source (Selberg, Lectures on Sieves,
    Collected Papers II; Ford citing equation (14.40))
        beta_kappa  <   2 kappa + 0.4454       (kappa sufficiently large).

VERDICT THIS ROUND: 19/36 = 0.5277... IS CORRECT; 0.4454 IS UNVERIFIED AND IS
NOT REPRODUCIBLE FROM THE MATHEMATICS.  Five independent lines, and the first is
decisive on its own:

 1. SELBERG'S OWN ANNOUNCEMENT.  A. Selberg, "Sifting problems, sifting density,
    and sieves", in Number theory, trace formulas and discrete groups (Symp. in
    honor of A. Selberg, Oslo 1987), 467-484, reviewed by G. Greaves, zbMATH
    Zbl 0675.10030, FETCHED BY THIS LANE 2026-08-28 via the zbMATH Open API:
    "More specifically, alpha_k > 1/(2k+19/36) for all sufficiently large k."
    (Selberg's alpha is the RECIPROCAL convention, so this is beta_k < 2k+19/36.)
 2. HEATH-BROWN, reviewing Franze's published paper for zbMATH (Zbl 1235.11089),
    ALSO FETCHED 2026-08-28: "A. Selberg [Lectures on sieves, Collected papers
    II, pp 65-247] showed that the sieving limit satisfies beta_kappa <= 2 kappa
    + 19/36 + o(1) as the dimension kappa tends to infinity."
 3. THE DERIVATION IS A ONE-LINE QUADRATIC OPTIMISATION and it is done here in
    EXACT RATIONALS (section S1).  The answer is forced: 19/36.
 4. FRANZE'S OWN COMPUTED TABLE converges to 0.5278 from below, monotonically,
    over kappa = 2..10 - and every single entry ALREADY EXCEEDS 0.4454 (S3).
 5. AT THE LEVEL 2 kappa + 0.4454 THE MAIN TERM IS STRICTLY NEGATIVE (S2), i.e.
    the sieve gives no positive lower bound there at all.

SOURCE STATUS: items 1 and 2 were fetched and read BY THIS LANE (zbMATH Open
API, 2026-08-28; JSON captures zb_franze2.json / zb_selberg2.json in the session
scratchpad); item 3's inputs are verified against the on-disk full text of
arXiv:1012.3809 (lines 456-472: u = kappa - 1/3 - d, P(w) = w + a, the
functional (-a^2 + a/2 - (2+9d)/18) sqrt(kappa/pi), "the optimal choice is
a = 1/4", "d <= -7/72", "beta_kappa <= 2u + 1 = 2 kappa + 19/36"); items 4-5
are computed here.  The Ford lecture notes / Brady thesis attributions of
0.4454 are from this round's literature relay (texts located and quoted by a
sub-search, not independently re-fetched by me).

A POSSIBLE ORIGIN FOR 0.4454, recorded as SPECULATION and clearly labelled:
Greaves' review of the same Selberg announcement carries, one sentence earlier,
"beta_k ~ c/k as k -> infinity for a certain constant c close to 1/2.445" (the
Buchstab-iterated Lambda^2 family, reciprocal convention).  The digit string
2.445 sits directly beside 19/36 in the primary source's own review; a reader
harvesting constants from that neighbourhood could plausibly carry "0.4454"
away.  This is a conjecture about the error's origin, not evidence.

RESIDUAL CAVEAT, stated because this lane's standing rule is that second-hand
citations expire fastest: equation (14.40) of Lectures on Sieves was NOT read.
The book is in copyright and is not on archive.org, the IAS Selberg archive, or
Google Books full text.  So the narrow possibility remains that (14.40) is a
LATER, SHARPER result than the pp. 174-176 computation Franze reproduces and than
the 1987 announcement.  Two independent readers reporting the identical
four-digit 0.4454 is weak evidence for that.  Against it: Franze says the 19/36
already uses weights on divisors with two and three prime factors, and cites the
same section 14 for it.  RECOMMENDATION: cite 19/36, credit Selberg, and record
0.4454 as unverified.  One scan of that page closes the question.
"""
from fractions import Fraction as Fr
from math import log

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def main():
    say("=" * 78)
    say("S1 - THE DERIVATION IN EXACT RATIONALS")
    say("=" * 78)
    say("  Franze section 3 (quoting Selberg, Lectures on Sieves pp. 174-176):")
    say("  with u = kappa - 1/3 - d and the linear polynomial P(w) = w + a,")
    say("      Sifting functional  ~  (1/V(z)) (-a^2 + a/2 - (2+9d)/18) sqrt(kappa/pi)")
    say("  and beta_kappa = 2u + 1.  So the constant is")
    say("      c = 1/3 - 2 * sup{ d : max_a (-a^2 + a/2 - (2+9d)/18) >= 0 }.")
    say("")
    # max over a of -a^2 + a/2 is at a = 1/4, value 1/16 (exact)
    a_star = Fr(1, 4)
    max_quad = -a_star ** 2 + a_star / 2
    say(f"  max_a (-a^2 + a/2) = {max_quad} at a = {a_star}   [derivative 1/2 - 2a]")
    assert max_quad == Fr(1, 16)
    # (2 + 9d)/18 = 1/9 + d/2 ; positivity: 1/16 - 1/9 - d/2 >= 0
    d_star = 2 * (Fr(1, 16) - Fr(1, 9))
    say(f"  (2 + 9d)/18 = 1/9 + d/2, so positivity needs d <= 2(1/16 - 1/9) = "
        f"{d_star}")
    assert d_star == Fr(-7, 72)
    c = Fr(1, 3) - 2 * d_star
    say(f"  beta_kappa = 2u + 1 = 2 kappa + 1/3 - 2d, hence c = 1/3 + 7/36 = {c}")
    assert c == Fr(19, 36)
    say(f"  EXACT: c = {c} = {float(c):.6f}.  ASSERTED in exact rationals.")

    say("")
    say("=" * 78)
    say("S2 - WHAT 0.4454 WOULD REQUIRE, AND WHY IT CANNOT HOLD")
    say("=" * 78)
    c_alt = 0.4454
    d_alt = (1.0 / 3.0 - c_alt) / 2.0
    coef = -0.25 ** 2 + 0.25 / 2 - 1.0 / 9.0 - d_alt / 2.0
    say(f"  c = {c_alt} forces d = (1/3 - c)/2 = {d_alt:.6f}, and then the")
    say(f"  coefficient of sqrt(kappa/pi) at the optimal a = 1/4 is")
    say(f"      1/16 - 1/9 - d/2 = {coef:.6f}  <  0.")
    say("  A NEGATIVE main term is not a lower bound at all, so 0.4454 is not")
    say("  attainable inside this functional.  For comparison, at c = 19/36 the")
    say(f"  coefficient is exactly {float(Fr(1,16) - Fr(1,9) - d_star/2)} (the")
    say("  boundary case, which is why Selberg needs the more elaborate weights).")
    assert coef < -0.02
    assert Fr(1, 16) - Fr(1, 9) - d_star / 2 == 0

    say("")
    say("=" * 78)
    say("S3 - FRANZE'S OWN TABLE: WHERE IT CONVERGES")
    say("=" * 78)
    F = {2: 4.516, 3: 6.520, 4: 8.522, 5: 10.523, 6: 12.524, 7: 14.524,
         8: 16.524, 9: 18.525, 10: 20.525}
    say("   kappa   beta_kappa   beta - 2 kappa    vs 19/36    vs 0.4454")
    prev = None
    for k in sorted(F):
        r = F[k] - 2 * k
        say(f"  {k:>6} {F[k]:>12.3f} {r:>16.3f}    {r - float(c):>+9.4f}"
            f"    {r - c_alt:>+9.4f}")
        assert r < float(c), (k, r)
        assert r > c_alt, (k, r)
        if prev is not None:
            assert r >= prev - 1e-12, (k, r, prev)
        prev = r
    say("  ASSERTED: every entry is BELOW 19/36, ABOVE 0.4454, and the sequence is")
    say("  non-decreasing - the exact signature of an asymptote at 19/36 and flatly")
    say("  inconsistent with an asymptotic upper bound of 0.4454 (Franze's own")
    say("  kappa = 2 value, 0.516, already exceeds it).")

    say("")
    say("=" * 78)
    say("S4 - VERDICT, AND WHAT IT CHANGES FOR US")
    say("=" * 78)
    say("  VERDICT: cite beta_kappa <= 2 kappa + 19/36 + o(1), credited to Selberg")
    say("  (Lectures on Sieves, sec. 14), and record Ford (2023 lecture notes) and")
    say("  Brady (2017 thesis) as reporting 0.4454 from the same source, a figure")
    say("  this lane could not reproduce.  It is a large-kappa statement and does")
    say("  NOT bear on our kappa = 2 numbers directly:")
    say("      Selberg's asymptotic at kappa = 2:  2*2 + 19/36 = "
        f"{4 + float(c):.4f} (not valid at kappa = 2)")
    say("      best PROVED value at kappa = 2:     beta_2 = 4.266450 (DHR, Blight)")
    say("      Selberg's CONJECTURED optimum:      2 kappa = 4")
    say("      proved floor (Brady):               (1+o(1)) 2 kappa/e ~ 1.47")
    say("  So nothing in our documents changes numerically; what changes is that a")
    say("  flagged conflict is resolved and the ceiling paragraph can be stated")
    say("  without hedging.  The lane's standing rule is honoured in both")
    say("  directions: 19/36 is now three-sourced (Selberg's own announcement,")
    say("  Heath-Brown's review, and an independent re-derivation), and 0.4454 is")
    say("  recorded as UNVERIFIED rather than as WRONG, because equation (14.40)")
    say("  itself was not read.")
    say("")
    say("  NOTE FOR ANY FUTURE ROUND: 0.4454 is not a rounding of 19/36 (0.5278),")
    say("  nor of 16/36 (0.4444) or 17/36 (0.4722); and Blight's three-prime-factor")
    say("  refinement ('Refinements of Selberg's sieve', Rutgers 2010, per")
    say("  Heath-Brown's review) gives beta = 4.450, 6.458, 8.470 at kappa = 2, 3,")
    say("  4 - margins over 2 kappa RISING (0.450, 0.458, 0.470) - so 0.4454 is not")
    say("  that family's asymptote either.  The most economical explanation is a")
    say("  shared misreading (see the docstring's labelled speculation about")
    say("  Greaves' adjacent '1/2.445'); the decisive test is a scan of (14.40).")
    say("")
    say("  CITATION CORRECTION BANKED IN PASSING (Heath-Brown's review, verbatim):")
    say("  the 4.266 value's book is Diamond-HALBERSTAM-GALWAY, 'A higher-")
    say("  dimensional sieve method', Cambridge Tracts 177 (2008), Zbl 1207.11099")
    say("  - cite the book with Galway, the METHOD as DHR.  And Blight's own new")
    say("  kappa = 2 value is 4.450; the 4.266450 she prints at full precision is")
    say("  her quotation of DHR, not her result.  Both now first-hand.")

    with open("research/data/j2_selberg.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("j2_selberg: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

# suppression-law - short-range anti-correlation of large gaps in the primorial residue pattern, and the suppression law for deep spectrum records

Status: MEASURED (not proved). Components individually: censuses SCRIPT-VERIFIED
(exact, full-period); the law's constants (lambda, p_1) computed from the machine;
the order-statistics step connecting them is heuristic. Established round 19
(constructor + mechanic, independently). Prior-art check: 2026-08-23 (section 6).

## 1. WHAT IT IS

Plain language. Take the pattern of integers coprime to a primorial - here in the
project's slot frame: slot k stands for the pair (6k-1, 6k+1), the "machine" M(y)
is the set of gears (primes) 5..y, gear q blocks the slots k = +-(6^{-1} mod q)
(mod q), and the surviving slots ("openings") form a periodic pattern whose gaps
are the object of study (F(M) = the maximal gap is a two-dimensional Jacobsthal
analogue). The finding: LARGE GAPS IN THIS PATTERN ARE STRONGLY NEGATIVELY
CORRELATED AT SHORT RANGE. If large ("qualifying-size") gaps occurred
independently at their measured density, windows of several consecutive large
gaps would be x26 to x1400 more frequent than they actually are. The deficit is
concentrated entirely at adjacent gaps (separation 1), rebounds slightly above
independence at separation 2, and is gone by separation 4-5. A quantitative
consequence, the SUPPRESSION LAW, predicts how far the best all-large window must
fall short of the unrestricted record window, with both constants computed from
the machine alone. A complementary exact census shows the mechanism: every
unrestricted record window has the same shape - two huge flanking gaps with the
machine's SMALLEST gaps inside - which is precisely the shape the size threshold
forbids.

Precise form. Definitions (slot units throughout; member-space distances are 6x
these):

* Machine M = M(y): gears = primes 5 <= q <= y; gear q blocks slot k iff
  k = +-c_q (mod q), c_q = 6^{-1} mod q. Openings = unblocked slots; the pattern
  is periodic with period prod q. Gap = difference of consecutive openings.
* F_j(M) = max over the period of the sum of j consecutive gaps (the gap
  spectrum); F = F_1 (maximal gap).
* For the next gear q': u' = round(q'/6); a gap is QUALIFYING-SIZE iff it is
  >= 2u' (the interior-gap floor: an interior gap of a merged run must be >= 2u',
  a proved theorem of the project - "Theorem 1" / the size half of R30).
* A j-window = j consecutive gaps; its INTERIOR = the j-2 middle gaps; the
  window QUALIFIES iff every interior gap is qualifying-size.
* p_1 = density of qualifying-size gaps in the period; p_j = fraction of
  j-windows that qualify; L = ln(1/p_1); lambda = the exponential scale of the
  window-sum tail (fitted; 2.73 at machine 29, independently recovered as the
  2.77 coefficient of the flank order-statistic law R33).
* qualmax_j = max window sum over QUALIFYING j-windows (equivalently Q_j, the
  qualifying spectrum at threshold a = 2u').

Statement A (anti-correlation deficits, exact censuses). Against the
independence prediction p_1^(j-2), the measured p_j shows deficits of

    x26     at machine 23, j = 4
    x6.7    at machine 29, j = 4
    x1400   at machine 29, j = 5

i.e. qualifying interiors are strongly negatively correlated. Lag-resolved
(R33): R(g) = P(both gaps at separation g qualifying)/p_1^2 has an adjacency
deficit and nothing more - at machines 11-17 the lag-1 value is EXACTLY ZERO
(two qualifying-size gaps cannot be adjacent there at all), 0.039-0.638 at
machines 19-29; a rebound above independence at lag 2 (up to 1.897);
independence restored by lag 4-5. Higher orders are super-multiplicative: at
machine 29, p_5/p_1^3 = 7.1e-4 against the pairwise prediction 2.2e-2 - a
further x30 beyond what pair correlations account for.

Statement B (suppression law). The shortfall of the best qualifying window
below the unrestricted record obeys

    suppression(j) = F_j - qualmax_j ~ lambda * (j-2) * ln(1/p_1)

with lambda and p_1 computed from M alone. Observed suppressions 7, 15, 30
against predicted 9.0, 21.7, 42.5 at the three constrained machine-depth cases -
right scale, conservative (over-predicting) at depth.

Statement C (extremal-shape fact, exact exhibition). Of the 132 windows
attaining F_j at machines 19/23/29 (full-period census), ZERO are literal and
ZERO are qualifying: the attaining shape is always two near-maximal flanks with
the machine's smallest gaps interior - e.g. machine 29, F_5 = 85 at
k = 772,741,833: flanks (30,18), interior (4,3,30); machine 23, F_3 = 50:
flanks (23,23), interior (4). The interior-gap floor >= 2u' forbids exactly
that shape, so the record windows are structurally the wrong shape to qualify -
the same fact Statement A measures as a rate deficit.

Payoff form (suppression-corrected flatness): the route's open part (D) follows
from F_j(M) - F(M) <= q' + lambda*(j-2)*L for every j >= 2; this holds at all
15 measured machine-depth pairs (corrected margins 4.7-20.8, bounded,
non-growing in j) where raw spectrum flatness F_j - F <= q' fails at 5 of 15.

## 2. WHY IT MIGHT BE NOVEL

* The object is the JOINT distribution of consecutive gap sizes in the reduced
  pattern mod a primorial, at and near the record (Jacobsthal) scale. The
  literature on this pattern (Jacobsthal function; Montgomery-Vaughan moments of
  reduced residues) treats either the single maximal gap or moments of COUNTS in
  intervals - we found no published joint law for consecutive gap SIZES, and no
  quantitative negative-correlation measurement at any scale, let alone the
  near-record scale.
* It is NOT a restatement of "chains of large prime gaps" (Erdos 1949, Maier
  1981, Ford-Maynard-Tao): those are existence lower bounds for primes (adjacent
  large gaps DO occur - consistent with our lag-2 rebound and with the deficit
  being finite), not a rate law against independence, and not about the
  primorial sieve pattern itself.
* It is NOT the classical singular-series correction (Hardy-Littlewood /
  Gallagher / Montgomery-Soundararajan): those predict weak (secondary-order)
  correlations for prime counts in intervals; the deficits here are order-of-
  magnitude (up to x1400) because they live at an extreme threshold, and they
  are measured exactly, not conjectured.
* The three-part decomposition (record = luck at the order-statistic level, but
  p_j itself structural; shape of maximisers universal and forbidden) converts
  "arithmetic luck" into named structure - we found no analogue of the
  extremal-shape fact (Statement C) anywhere: published large-gap constructions
  build one long gap and say nothing about the sizes of its neighbouring gaps.

Classical shadow, stated honestly: negative dependence among sieve survivors is
folklore at the heuristic level (a long gap "uses up" blocked residues), and
weak negative correlation of prime counts in adjacent short intervals is
predicted under Hardy-Littlewood (see section 6). The delta is: measured (not
conjectured), at the record threshold (not the mean), with a quantified law and
constants from the machine alone, plus the exact zero at machines 11-17 and the
lag-2 rebound - a sign structure no published model states.

## 3. PROOF

Status: MEASURED, NOT PROVED. Honest decomposition:

* The censuses behind Statements A and C, and the values F_j, Q_j, p_j, are
  EXACT full-period computations (script-verified): no sampling, no fitting.
* lambda is FITTED (exponential tail scale; cross-checked against the
  independently fitted flank order-statistic coefficient 2.77 vs 2.73).
* The step from (p_j, order statistics) to Statement B's formula is a HEURISTIC
  (extreme-value reasoning), verified against the three constrained cases.
* One proved fragment exists: the ADJACENT-GAP EXCLUSION LAW (lateral r20) -
  three consecutive openings with gaps (g1, g2) are impossible whenever
  (g1 mod 5, g2 mod 5) is in a listed 6 of 25 classes; complete for 3-point
  shapes by the completeness lemma. This proves a piece of the lag-1 deficit
  but not its magnitude. The interior-gap floor >= 2u' (which forbids the
  Statement C shape) is also a proved theorem.

Reproduction: research/window_profile.py + research/suppression_law.py (R31:
profiles, luck test, the law), research/anticorr_law.py (R32/R33: exposure
bound, R(lag), occurrence form), research/gap_pair_census.py with data
research/data/gap_pair_joint.csv + gap_pair_hist.csv (the p_j object),
research/unrestricted_max.py (Statement C, the 132 maximisers),
research/qspec_table.py with research/data/qspec_table.csv (Q_j tables),
research/data/flank_envelope_*.csv. Narrative: docs/proof-search/agents-shared.md
(round-19 SUMMARY), docs/proof-search/constructor.md R31-R33,
docs/proof-search/mechanic.md (maximiser exhibition and Q_j tables).

## 4. IMPLICATIONS

Inside the project:

* Repairs (rather than patches) round 17's refutation of spectrum flatness:
  the corrected form holds 15/15 where the raw form fails 5/15, and subsumes
  round 18's two-part target - no winning-depth assumption needed.
* Reverses the route's round 8-17 assumption: lemma 1 (F2 - F <= q') is the
  j = 2 case, and DEEPER cases are the EASIER ones (the suppression term grows
  with j).
* Derives par trading instead of observing it: gain per added link = spectrum
  increment (5-15), loss per link = lambda*L (4.2, 5.5, 9.0) - approximately
  equal, which is why merged records are nearly depth-independent.
* Makes the whole of the open part (D) equivalent to a statement about p_j -
  and R32 shows (D) needs almost nothing: independence would clear every
  constrained case by x170-x201,381; the needed fact is only "no >~x170
  positive correlation", far weaker than the measured law.

Outside the project, if proved: a joint (multi-gap) refinement of the
Jacobsthal problem - not just how large the largest gap mod a primorial is, but
a law for how large-gap events interact at short range, with an extremal-shape
theorem for record windows. It would put quantitative content behind the
folklore "a sieve pattern cannot afford two record gaps close together", which
currently exists only as heuristics or as weak-correlation conjectures for
primes.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

* Part (D) of the project's tolerance route (the sole open input; R14 + R21 +
  R20 + R23 + (D) => twin primes infinite): the suppression law is the current
  working form of (D), and the anti-correlation law is what would make it
  rigorous.
* The Jacobsthal function's fine structure (Ziller-Morack compute h(n); no
  published work on the joint law of consecutive gaps near h(n)); the project's
  paired-Jacobsthal values h_2 are the same object one level up.
* Hardy-Littlewood k-tuple heuristics and their secondary terms
  (Montgomery-Soundararajan variance; Lemke Oliver-Soundararajan biases): the
  measured deficits are a sieve-pattern testbed where those correction terms
  can be checked exactly, at an extreme threshold no prime data reaches.
* Wall V (extreme-value control of sieve patterns), the project's named
  obstruction: this finding is the first quantitative structure inside it.

## 6. PRIOR-ART CHECK (2026-08-23)

Searches actually run (engine: Claude WebSearch, 2026-08-23):

1. "correlations between consecutive prime gaps negative correlation large gaps
   adjacent"
2. "joint distribution consecutive gaps reduced residues modulo primorial
   Jacobsthal function"
3. "Montgomery Vaughan distribution of reduced residues moments gaps Hooley
   coprime integers"
4. "Maier chains of large gaps between consecutive primes Erdos 1949
   consecutive large gaps"
5. "Lemke Oliver Soundararajan unexpected biases consecutive primes
   correlations gaps"
6. "Montgomery Soundararajan primes short intervals variance Gaussian negative
   correlation gaps Cramer model"
7. "Ford Green Konyagin Maynard Tao large gaps between primes structure of
   extremal interval Jacobsthal maximal gap shape"
8. "gaps between consecutive totatives correlation Hausman Shapiro maximum gap
   Erdos conjecture reduced residues"

Nearest published results found:

* Erdos (1949) and H. Maier, "Chains of large gaps between consecutive primes",
  Adv. Math. 39 (1981): k consecutive prime gaps can simultaneously be >> the
  Erdos-Rankin size. Ford-Maynard-Tao, "Chains of large gaps between primes"
  (arXiv 2015, in Irregularities in the Distribution of Prime Numbers, 2018):
  same with the modern gap bound, and the chain's measure decreases by a factor
  ~1/k^2 per added gap. DELTA: these are existence LOWER bounds for primes -
  they show adjacent large gaps occur (which our law also says: the deficit is
  finite and rebounds at lag 2), but state no rate against independence, no
  negative-correlation law, and nothing about the reduced-residue pattern mod a
  primorial or the near-record threshold. The 1/k^2 cost of chaining is a
  construction bookkeeping constant, not a measured joint density.
* H. L. Montgomery, R. C. Vaughan, "On the distribution of reduced residues",
  Ann. of Math. 123 (1986); C. Hooley (1965): moments of the COUNT of reduced
  residues mod q in short intervals are Gaussian with Poisson-scale variance.
  DELTA: same underlying set, but a count-in-interval CLT, not a joint law of
  consecutive gap sizes and silent about correlations at the record scale.
* H. L. Montgomery, K. Soundararajan, "Primes in short intervals", Comm. Math.
  Phys. 252 (2004): variance ~ H log(N/H), BELOW Cramer's model - a
  negative-correlation-flavoured secondary term, conditional on a strong
  Hardy-Littlewood conjecture. Sun-Kai Leung, "Joint distribution of primes in
  multiple short intervals" (arXiv:2401.04000): primes in neighbouring intervals
  asymptotically bivariate Gaussian with WEAK negative correlation. DELTA:
  nearest in sign and spirit; but weak/secondary-order vs order-of-magnitude
  (x26-x1400), prime counts vs qualifying gap events, conjectural/asymptotic vs
  exact finite censuses, mean scale vs record scale.
* R. Lemke Oliver, K. Soundararajan, "Unexpected biases in the distribution of
  consecutive primes", PNAS 113 (2016): strong empirical correlations between
  CONSECUTIVE primes' residue classes mod q, explained by Hardy-Littlewood
  secondary terms. DELTA: correlations of residues of consecutive primes, not
  of consecutive gap SIZES in the sieve pattern; no record-threshold statement.
* P. X. Gallagher, "On the distribution of primes in short intervals" (1976):
  Poisson spacing under uniform Hardy-Littlewood - the independence baseline the
  deficits are measured against. Banks-Ford-Tao, "The probabilistic model for
  primes" (2019): refined random sieve models; again models, not measured joint
  gap laws mod primorials.
* Ford-Green-Konyagin-Maynard-Tao, "Long gaps between primes", J. AMS 31
  (2018), and the Jacobsthal computation literature (Hagedorn; Ziller-Morack):
  the largest single gap; the searches found NOTHING on the composition/shape
  of the gaps neighbouring a record gap (Statement C) or on the joint
  distribution of consecutive gaps in the Jacobsthal setting (search 2 returned
  only single-gap computations and bounds).

VERDICT: PARTIAL OVERLAP for the bare phenomenon "nearby large-gap/prime-count
events are negatively correlated" (Montgomery-Soundararajan 2004 and Leung 2024
predict weak negative correlation for prime counts under Hardy-Littlewood;
Maier/Ford-Maynard-Tao settle the adjacent-large-gap existence side for
primes). NOVEL AS FAR AS SEARCHED for the specific content: (i) the measured
order-of-magnitude deficits (x26/x6.7/x1400) in the reduced pattern mod a
primorial at the qualifying (near-record) threshold, including the exact zero
at machines 11-17 and the lag-2 rebound; (ii) the quantified suppression law
F_j - qualmax_j ~ lambda*(j-2)*ln(1/p_1) with machine-computable constants;
(iii) the extremal-shape fact (132/132 record windows have the forbidden
two-big-flanks/smallest-interiors shape). No published joint law of consecutive
gap sizes mod primorials, and no published description of record-window
composition, was found in any of the eight searches.

## 7. ROUND-20 ADDENDUM (constructor, 2026-08-23) - the law is SUPER-MARKOV, and the heuristic step is now bypassed at measured steps

New exact censuses (research/tm_resid_runs.py, full period, cyclic-seam exact;
machine 31 = 3.34e10 slots, first joint data there):

* Machine 31 (q' = 37): p_1V = 0.018445; runs of 2/3/4 consecutive qualifying gaps =
  502,708 / 508 / 0 against independence 2,118,360 / 39,073 / 721 - deficits x4.2,
  x77, and an exact zero. DEPTH CAPS (exact, all machines): the longest run of
  consecutive residue-qualifying gaps is 1 at machines 11-17, 2 at 19-23, 3 at 29,
  3 at 31 - equivalently the qualifying-gap partial map A_V is NILPOTENT of index
  2/2/2/3/3/4/4 (verified as operator identities, research/tm_nilpotency.py), and
  k_max <= index(A_V) at every step.
* THE DEFICIT IS NOT MARKOV (research/tm_transfer.py): the exact one-step transfer
  matrix on gap values (from the full-period pair census) OVER-predicts deep runs by
  growing factors - x49 at machine 29 depth 3 (predicted 391, exact 8), x4.4 at
  machine 31 depth 3 (predicted 2,242, exact 508); at size floors the over-prediction
  grows x4.4 -> x12.6 -> x40 with depth at machine 29. Equivalently the per-link
  conditional probabilities fall geometrically: at machine 29,
  P(next qualifying | 1 previous) = 5.5e-3, P(| 2 previous) = 1.8e-4 - each added
  link is ~30x more suppressed than the last. NO fixed-order Markov chain (hence no
  finite transfer matrix on gap values) reproduces the law; the memory is longer.
* The lag-2 REBOUND is partly a renewal artifact: the Markov chain built from the
  lag-1 matrix predicts a rebound (1.27-1.41) but understates the measured one
  (1.53 at 29, 1.90 at 23); and at machine 31 the measured rebound is GONE
  (R(2) = 0.71 < 1 vs predicted 1.36) - a regime change coinciding with padding
  onset. The full chain's spectral gap is stable: |lambda_2| = 0.55-0.66 across
  machines 11-31.
* THE HEURISTIC STEP IS NOW BYPASSED at measured steps: (D) at a step follows from
  the exact criterion max(F_2, max_j qualmax_j) <= F + q' (no lambda, no order
  statistics; research/tm_qualmax_check.py) - it holds at all seven steps 11->13 ..
  31->37 with margins 0.52-0.69 q' at literal steps and 0.19 q' at the padded step
  31->37, and the criterion value EQUALS F(M+q') at six of the seven steps
  (slack 2 at 23->29). Statement B's constants remain the asymptotic reading; the
  criterion is the exact form.
* The rate side is now rigorous where constrained: see docs/novel/renewal-ladder.md
  (closed-form CRT upper bounds on p_j clearing the requirement at every measured
  constrained case, including the two R32 failures).


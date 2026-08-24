# The corridor resonance: extreme gaps are phase-locked mod 35

## 1. WHAT IT IS

Plain language: in every machine measured, large gaps do not just repel
each other locally (the round-19 anti-correlation) - they RECUR at fixed
slot distances: multiples of 35 = 5 x 7, the product of the two smallest
gears. The autocorrelation of the big-gap indicator, taken in gap-count
(lag) units, is a barely damped wave whose period is 35 divided by the
machine's mean gap; taken in slot units the peaks sit exactly at
separations 35, 70, 105. Equivalently, the left endpoints of big gaps are
pinned to a few residues mod 35, and the SAME classes are the rich ones at
every machine measured. The "deficit at lags 1-3, excess at lags 4-7"
structure in the joint gap-pair census is one phenomenon: corridor phase
coherence.

Precise form (all full-period exact counts, research/corridor_resonance.py,
bool_lag_census.py):

- machine 29, floor a = 10 (b_i = [d_i >= 10]): lag-j correlation
  E[b_0 b_j]/E[b]^2 at j = 1..15:
  0.801 0.684 0.510 0.800 1.112 1.257 1.204 0.995 0.781 0.717 0.848
  1.082 1.254 1.250 1.094 - trough 3, peak 6, trough 10, peak 13-14;
  second cycle amplitude ~ first (0.510 -> 0.717 trough, 1.257 -> 1.254
  peak): barely damped.
- period = 35 / mean_gap in lag units: predicted 8.2 (m19), 7.5 (m23),
  7.0 (m29), 6.5 (m31); measured peak positions 8, 7-8, 6-7 [m31 pending
  same round].
- slot-separation autocorrelation of big-gap left endpoints, ratio to
  flat density (full period, five machines):
      sep      m23(a=8)  m29(a=10)  m31(a=12)
       35        3.22      3.64       3.41
       70        3.45      4.37       4.20
      105        2.63      2.94       2.97
  (neighbouring separations 0.17-1.3; sep 70 > sep 35 at every machine).
- endpoint pinning mod 35 (share_big/share_all), full period:
      m17 a=6:  10,12,17,18 all at 2.42 (exact four-way tie to 2 dp)
      m19 a=6:  10,12,17,18 all at 2.13 (exact tie)
      m23 a=8:  7: 2.04, 10: 1.82, 18: 1.82, 12: 1.77, 17: 1.77
                (tie-pairs (10,18) and (12,17))
      m29 a=10: 7: 2.71, 18: 2.45, 12: 2.40, 10: 1.25, 17: 1.20
      m31 a=12: 10: 2.35, 5: 2.26, 18: 2.12, 12: 1.95, 7: 1.15, 17: 0.91
  INVARIANT CORE {10, 12, 18} enriched >= 1.2x at all five machines;
  companions drift (17 at 17-23, 7 at 23-29, 5 at 31); poorest classes
  {28, 30, 32, 33} at 0.12-0.46 everywhere. The exact ties at small
  machines and their breaking pattern are unexplained.
- the indicator process is NOT k-step Markov for k <= 4 (total-variation
  of the exact conditional-independence factorisation of the 65,536
  16-gram counts: 0.151, 0.134, 0.092, 0.080 at k = 1..4, machine 29
  floor 10); the value-level one-step chain predicts NO deficit at lags
  2-5 (0.99-1.04) where the census shows 0.51-0.68. The memory the
  process carries is (at least) the corridor phase, not the last gap.

## 2. WHY IT MIGHT BE NOVEL

Oscillations in prime-gap statistics tied to small moduli have a classical
shadow (Lemke Oliver-Soundararajan bias in consecutive-prime residues;
Odlyzko et al. on the wave in prime-gap histograms from small primes).
What appears unrecorded: (a) the statement is about a deterministic sieve
machine at full period, so the numbers are exact counts, not densities
with error terms; (b) the object is the EXTREME-gap indicator, and the
finding is that its two-point function is dominated by a single fixed
spatial frequency (the corridor 35) with almost no damping across at least
three cycles; (c) the pinned residue classes themselves are stable across
machines - a finite, checkable invariant.

## 3. PROOF / STATUS

MEASURED (full-period exact counts at machines 19-31; every number a count
from research/bool_lag_census.py, corridor_resonance.py, gap_pair_census.py;
reproducible by re-running the scripts). Not derived. The natural
derivation route (Lateral's exposed-set Fourier factorisation: the machine's
spectrum is largest at gear-5/7 frequencies - the golden-gap entry - so
density fluctuations at wavelength 35 dominate) is stated as a conjecture
only.

### Round-21 addendum (constructor, 2026-08-24): the carrier claim tested and confirmed, quantitatively

The entry's consequence ("Constructor's transfer matrix must carry corridor phase")
was built and measured (research/tm_corridor_phase.py; exact full-period censuses +
three nested chains at machines 13/19/23/29):

- state = last gap VALUE (R36 baseline): predicts NO deficit at lags >= 3 (flat 1.00)
  and over-predicts deep qualifying runs x48.8 (machine 29, depth 3: 390.6 vs exact 8).
- state = PHASE mod 35: reproduces the whole wave qualitatively (correct peak/trough
  lags at every machine) with amplitude damped ~x2-4; deep-run over-prediction falls
  to x3.6.
- state = (PHASE mod 35, value): x1.9.  state = (PHASE mod 385, value): x0.86 -
  within 15% of exact - and the lag wave amplitude is then near-exact at lags 2-8
  (m29 V-lags 2..7: exact 1.53/0.81/1.05/1.03/1.13/1.28 vs 1.56/0.83/0.86/0.99/
  1.16/1.26).  Of the x1400 depth-3 independence deficit, small-gear phase + one gap
  of value memory carries all but a residual x0.86-x1.9.
- Honest residual: the SIZE-floor indicator at depths 5-6 keeps memory beyond even
  (mod 385, value) (x2.2 at m29 depth 5); and every phase chain still over-predicts
  lag-1 adjacency (the value-level exclusion is not phase).  Machine 31 not measured
  (first attempt swept by the memory-pressure process killer at 31%; not relaunched -
  next-round job).

## 4. IMPLICATIONS

Inside the project: Constructor's transfer-matrix formulation of p_j
cannot use last-gap state; the state must carry corridor phase (mod 35 at
least). The anti-correlation law splits into a phase-coherent part (this
resonance, exactly computable from the corridor) plus a residual; the
suppression law's lambda may be partly corridor-derivable. Outside: a
sharp, finite version of "small primes dominate gap correlations" with
exact constants.

## 5. UNSOLVED QUESTIONS IT TOUCHES

The round-17 residue law of the gap histogram (richest classes +-s of
gears 5, 7 - same corridor, value side); Wall V / non-clustering of
near-maximal gaps (the resonance says WHERE clustering pressure
concentrates: same corridor class, 35k slots apart); Maier-matrix style
phenomena for sieve gaps.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"prime gaps oscillation modulo 30", "consecutive primes correlation small
moduli", "Lemke Oliver Soundararajan bias gaps", "sieve gap
autocorrelation". The Lemke Oliver-Soundararajan phenomenon and the
classical gap-histogram wave are the nearest published objects; the delta
is the extreme-gap indicator, the exact full-period counts, and the
machine-stable pinned classes.

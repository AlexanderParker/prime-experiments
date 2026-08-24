# The pole-phase law: why the gap histogram's mod-5 phase is +126 degrees

Status: MEASURED (full-period exact counts, machines 11-31, machine 37 at
12.9% prefix) + PROVED (the pole identity and the depth sum rule - one-line
algebra) + MODEL (the closed-form predictor's phases, floats, labeled).
Established round 21 (Lateral; resolves Mechanic's round-20 unexplained
handoff). Script: `research/c14_phase.py`. Prior-art check: NOT YET CHECKED
(section 6).

## 1. WHAT IT IS

Plain language. Mechanic's round-20 measurement: the DFT of the full-period
gap-value histogram at gear 5, H_5(1) = sum_v W1(v) e(2 pi i v/5), keeps a
constant phase of +126 degrees (+-2) across seven machines while its
amplitude falls - and 126 is not the 0/180 that concentration at the +-s
residue classes would give. Resolution: 126 degrees is the phase of the
POLE of the one-sided integer lattice at frequency 1/5, and the measured
constancy is the statement that the DIFFERENCED histogram carries no mod-5
phase at all.

Precise form. Let omega = e(k/p) and abbreviate the histogram transform
H_p(k) = sum_{g>=1} W1(g) omega^g. Abel summation gives the exact identity

    H_p(k) = [omega/(1-omega)] * B_p(k),
    B_p(k) = W1(1) + sum_{g>=2} (W1(g) - W1(g-1)) omega^{g-1},

and    arg( omega/(1-omega) ) = 90 + 180 k/p   degrees, EXACTLY.

For (p,k) = (5,1): 90 + 36 = 126. So

    arg H_5(1) = 126 deg  <=>  the difference transform B_5(1) is REAL > 0.

Measured (research/c14_phase.py parts 1, 2, 6; class counts exact integers,
angles float):

- arg B_5(1) (equivalently arg H_5(1) - 126): +3.63, +3.65, +1.82, +0.33,
  +0.35, +0.06, -0.23 at machines 11..31 full period; -0.34 at the machine-37
  12.9% prefix. From machine 19 on, 100.00% of the frequency-1 energy of the
  residue-class deviation vector lies in the 126-degree direction.
- SECOND FREQUENCY CONFIRMS THE LAW (new measurement, predicted by the
  frame): the pole phase at (5,2) is 90 + 72 = 162 == -18 (mod 180; the
  bracket's sign is negative here). Measured arg B_5(2): -31.7, -22.6,
  -17.4, -13.9, -11.2, -9.0, -7.0, -5.7 - monotone toward 0 at every step.
- GEAR-7 CONTRAST: arg B_7(1) = -3.1, +5.1, +9.6, +12.7, +14.3, +15.4,
  +16.2, +17.0 - the bracket is NOT real and drifts: no pin, exactly the
  "slow drift" Mechanic observed for the mod-7 phase. Being at the pole
  phase is a property gear 5 has and gear 7 does not.

Equivalent exact reformulations of "arg H_5(1) = 126":

- GOLDEN CONSTRAINT on the residue-class counts N_r = #{gaps == r mod 5}:
      phi^2 (N_0 + N_1) = (N_2 + N_4) + 2 phi N_3,     phi = (1+sqrt5)/2.
- REFLECTION ANTISYMMETRY: the frequency-1 component of the deviation
  vector is antisymmetric under v -> 1 - v (mod 5), the reflection that
  swaps classes 0 <-> 1 and 2 <-> 4 and fixes 3 (each difference
  omega^r - omega^{1-r} has argument exactly 126 deg).

Anchor sum rule (PROVED, verified exactly): summing the transform over ALL
window depths j (What_j(omega) = sum over j-windows of omega^{window sum}),

    sum_{j>=1} What_j(omega) = |sum_a omega^a|^2 - N
                             = (2 - phi) prod_{q != 5} (q-2)^2  -  prod_q (q-2),

a REAL number - because every ordered pair of openings is the endpoint pair
of exactly one window (the round-20 depth-sum identity) and the openings
are EXACTLY uniform over A_5 = {0,2,3} mod 5 (CRT; asserted as integers at
machines 13-23: class counts prod_{q!=5}(q-2), 0, prod, prod, 0). So the
depth family's phases must close a polygon in C; W_1's arm at 126 deg is
the first edge (the spiral is measured for j <= 25 in part 3; it is
irregular, and W_2's arm climbs toward the pole phase as machines grow:
66.5 -> 87.7 -> 113.2 deg at 17/19/23).

MECHANISM AND THE HONEST LIMIT STATEMENT (model, floats): the closed-form
predictor W1_pred(g) = N2(g) prod_t (1 - N3(0,t,g)/N2(g)) reproduces the
measured phase within 1.5 deg at every machine 11-31 (gear 7: within
2.5 deg, including the drift) - the phase is CRT arithmetic, not noise.
Within that model, pushed beyond all data (machines 37..499, pure closed
form), the phase is NOT asymptotically pinned at 126: it crosses 126
between machines 31 and 47 and drifts slowly on (-0.1 deg/machine there;
124.6 at y = 97, 117.6 at y = 499). So the measured "+126 +- 2 machine-
independent" is the pole phase plus a PLATEAU where the bracket's phase
crosses zero - not (on present evidence) an arithmetic invariant. The
model's own error has a trend at gear 7, so the pin-vs-drift question at
gear 5 stays open; it is decidable at the next machines: the model
predicts arg H_5(1) = 125.5-125.9 at machines 41/43 - a measured return
to 126.0 +- 0.1 would falsify the drift and establish a genuine pin.

Amplitude observation (recorded, unexplained): |H_5(1)|/H_0 * mean_gap =
1.037, 1.015, 1.014, 1.019, 1.016, 1.010, 1.014 at machines 13..37 - the
mod-5 ripple amplitude is 1/mean_gap to +-1% with no trend.

## 2. WHY IT MIGHT BE NOVEL

The amplitude side of residue oscillations in gap histograms has classical
shadows (the mod-6/mod-30 wave in prime-gap histograms; Lemke Oliver-
Soundararajan for consecutive primes). What appears unrecorded is the PHASE
side: (a) that the phase of the mod-p line of a sieve gap histogram is the
universal pole phase 90 + 180k/p of the one-sided integer lattice, with the
arithmetic content confined to the residual (the differenced histogram's
phase); (b) the two-frequency confirmation (the freq-2 line converging to
-18 deg); (c) the golden linear constraint on residue-class counts as the
exact form of the phase statement. Honest caveat: the pole identity itself
is one line of Abel summation - classical technique; the content is the
measured reality of the bracket across machines and the resolution of which
gears/frequencies are pinned.

## 3. PROOF / STATUS

- Pole identity + pole phase: PROVED (Abel summation; arg(omega/(1-omega))
  = 90 + 180k/p since 1 - e(x) = -2i sin(pi x) e(x/2)).
- Depth sum rule: PROVED (depth-sum identity + CRT class uniformity);
  class-count integer assertions green at machines 13-23.
- Phase measurements: exact integer class counts from full-period censuses
  (research/data/gap_pair_hist.csv, Mechanic), angles floats.
- Models M0/M1 and the asymptotic sweep: floats, labeled; every table in
  research/c14_phase.py (parts 4, 5). Model M2 (corridor-hardness beta-
  model as the sole mechanism) REFUTED: its phases sit at -163..-169 deg
  for gear 5 at every beta - recorded as a dead end.

## 4. IMPLICATIONS

Inside the project: Mechanic's C14 unexplained phase is closed - the
constant is not an arithmetic invariant to chase; the arithmetic object is
the BRACKET (the differenced histogram's transform), whose phase is the
honest residual and whose zero-crossing near machine 29-31 is a measured
event. Any future residue-law modeling should difference the histogram
first. Outside: a clean, checkable phase law for residue oscillations of
gap distributions of sifted sets (and plausibly of prime gaps themselves -
the same pole argument applies to any one-sided slowly-varying histogram;
the prime-gap histogram mod small p is an immediate test target).

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- WHY the gear-5 bracket is real to +-0.4 deg across machines 19-37 while
  gear 7's drifts - reproduced by the closed-form model, not conceptually
  derived. Open micro-question with an exact finite form per machine.
- Pin vs drift at gear 5: decidable at machines 41/43 (prediction above).
- The 1/mean_gap amplitude near-law (constant 1.015 +- 1%): unexplained.
- Whether the classical prime-gap mod-6/30 wave obeys the same pole-phase
  law (test outside the machine).

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager: "phase
prime gap histogram residue classes oscillation"; "Abel summation
characteristic function gap distribution sieve"; "Lemke Oliver
Soundararajan phase"; "golden ratio linear relation residue class counts
gaps". Nearest expected art: amplitude-level treatments of the gap-
histogram wave (Odlyzko et al.); no phase-level statement expected but the
pole argument is elementary enough that a signal-processing-literate
treatment may exist.

## 7. ROUND-21 ADDENDUM (Mechanic): THE PIN IS FALSIFIED - DRIFT CONFIRMED

The decision measurement named in sections 1 and 5 was taken (2026-08-24,
research/ghist_prefix.py, logs research/data/ghist41_prefix.log,
ghist43_prefix.log): 2e9-slot prefix gap histograms at machines 41 and 43.

    machine 41: arg H_5(1) = +125.70 deg   (335,220,558 gaps)
    machine 43: arg H_5(1) = +125.76 deg   (319,628,891 gaps)

Both inside the drift model's predicted band 125.5-125.9 and outside the
pin's 126.0 +- 0.1: the phase is a PLATEAU crossing 126, not an arithmetic
invariant - the model's drift is confirmed at the first machines beyond its
data. Bonus: the amplitude near-law holds at both new machines to 0.1%:
|H_5(1)|/H_0 = 0.16998 vs 1.015/mean_gap = 0.17012 (m41); 0.16199 vs
0.16221 (m43). (Prefix caveat, validated: phases are from 2e9-slot prefixes of
periods 5.07e13 / 2.18e15; at machine 31 the same 2e9-slot prefix
reproduces the full-period phase and amplitude to displayed precision -
+125.77 deg and 0.18813 both ways (ghist31_prefix2e9.log vs
ghist31_full.log, full 33.4e9-slot scan). Bonus exact value from the
validation: arg H_5(1) at machine 31 FULL PERIOD = +125.77 deg - the
plateau's crossing of 126 has already happened by m31.)

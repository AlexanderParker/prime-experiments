# The twin machine is a low-F outlier among its own counterfactuals

Lateral, round 27 (2026-08-29). Status: SCRIPT-VERIFIED (exhaustive, exact) for
machines 11, 13, 17, 19. Mechanism: NOT explained (one candidate pre-registered
and refuted in the sign).

## 1. WHAT IT IS

The machine has two kinds of input: WHICH gears (the primes 5..y) and WHERE each
gear's teeth sit. The gears are the problem; the tooth positions are FORCED by
the twin constellation, since gear q blocks slot k exactly when 6k = +-1 mod q,
i.e. at k = +-v_q with v_q = 6^{-1} mod q. Nothing in the project had ever asked
what happens if the teeth move.

Define the counterfactual family: keep the gears, keep the mirror symmetry
(teeth at +-v_q, which every twin-type constellation has), and let the half-width
v_q range freely:

    V(y) = prod_{q <= y} {1, 2, ..., (q-1)/2},   |V| = 30, 180, 1440, 12960
                                                 at y = 11, 13, 17, 19.

Every member of V(y) has the SAME period P, the SAME number of openings
prod (q-2) (the sharing law: phases move where survivors are, never how many),
and the same per-gear kill density. Only the POSITIONS differ. F is invariant
under k -> +-k + b but NOT under k -> ck (scaling is not an isometry of Z_P), so
F genuinely varies over V(y). The family is small enough to ENUMERATE
EXHAUSTIVELY.

RESULT (exact, all of V(y), research/tooth_counterfactual.py):

    y    |V|     F(twin)  min   median   max   twin's percentile in V(y)
    11   30      7        6     8        11    20.0%
    13   180     11       10    13       25    18.1%
    17   1440    18       14    19       32    26.4%
    19   12960   25       20    28       43    17.1%

THE TWIN MACHINE'S RECORD GAP IS IN THE BOTTOM FIFTH TO QUARTER OF ITS OWN
COUNTERFACTUAL DISTRIBUTION AT EVERY MACHINE TESTED, and roughly 10-15% below
the median - but it is never the minimum. The maximum is 1.6-1.9x the twin value
(43 vs 25 at m19), so the family is wide and the twin's position inside it is a
real fact, not a narrow band.

## 2. WHY IT MIGHT BE NOVEL

Jacobsthal-type quantities are always studied for the actual reduced residue
system (or for a fixed admissible constellation). The COUNTERFACTUAL
DISTRIBUTION of the maximal gap over all symmetric two-tooth sievings with the
same gears and the same survivor count appears not to be a studied object, and
the twin machine's position inside it is a new kind of statement about the twin
problem: it says the twin constellation is not a generic sieve, and that the
non-genericity is in the DIRECTION OF SMALLER GAPS.

It is also the first quantity this project has found on which the real phase
vector IS distinguished. Round 2's enumeration (lateral Refuted 3) scored the
real phase vector on WASTE metrics and found it in the top 10-25% with "no
variational handle". This is the same parameter space with F itself as the
objective, and it separates: the real vector sits low, consistently.

## 3. PROOF / STATUS

SCRIPT-VERIFIED, exhaustive and exact: research/tooth_counterfactual.py builds
every one of the 30 / 180 / 1440 / 12960 sievings by direct full-period sieve
(P up to 1,616,615), computes the exact cyclic maximal gap, and asserts (a) the
true tooth vector is a member of the family, (b) every member has exactly
prod (q-2) openings. Log research/data/tooth_counterfactual.log, 10 gates, exit 0.

NOT PROVED, and not extrapolated: four machines is four machines. m23 would be
|V| = 12960 * 11 = 142,560 sievings over P = 37,182,145 - about an hour of
single-core work and the next honest rung.

INDEPENDENCE CAVEAT, stated because it matters: the four rows are NOT four
independent observations. The twin tooth vector at m19 is the m17 vector with one
coordinate appended, and so on down, so the four percentiles are nested and a
naive "0.264^4 = 0.005" significance calculation is WRONG. What the data support
is "consistently below the median at four nested machines with the deficit
neither growing nor shrinking", not a p-value.

## 4. IMPLICATIONS

(i) It is a POSITIVE fact of the kind the project keeps failing to find: the
arithmetic of the twin constellation makes F smaller than a generic same-density
sieve, which is the right direction for the conjecture. Every upper-bound
argument that treats the machine as "some sieve with these densities" is
therefore leaving something on the table, and the measurement says how much: the
median counterfactual F is 10-15% above the truth, and the counterfactual maximum
is 60-90% above it.

(ii) It gives a NEW FALSIFIABLE OBJECT for the extreme-value question: instead of
asking how F grows, ask how the twin's PERCENTILE moves. If the percentile is
stable (~20%) the twin machine is a fixed distance into the tail of its own
family; if it drifts to 50% the twin's advantage is a small-machine effect.

(iii) It reframes "arithmetic selection" concretely. The project's standing
verdict on erratic quantities is "arithmetic luck, not structure". Here the
arithmetic luck has a sign and a size.

## 5. UNSOLVED QUESTIONS IT TOUCHES

Jacobsthal's function and its two-teeth analogue; the extremal problem "which
symmetric tooth vector maximises / minimises the maximal gap" (a covering-design
question with a Jacobsthal flavour, and the minimum is attained away from the
twin vector at every machine tested); the general question of how much of the
twin problem's difficulty is generic-sieve difficulty.

MECHANISM: OPEN, and one candidate is already dead. Pre-registered P11 (round 27)
predicted that the explanation is ANGULAR COHERENCE - the twin vector has
v_q/q ~ 1/6 at every gear, the smallest angular dispersion in the family, and
coherent teeth should pack better. REFUTED, and refuted in the SIGN: Spearman
correlation between F and angular dispersion is -0.14 / -0.20 / -0.11 at
m13/m17/m19 (higher dispersion goes with slightly LOWER F), and the twin sits in
the LOWEST-dispersion quartile, which is the quartile with the HIGHEST mean F
(28.56 vs 27.69 at m19). Within that quartile alone the twin is at the 15.6% /
20.8% / 10.5% percentile. So the twin vector is a low-F outlier INSIDE the
high-F coherence class - the effect is real and its cause is not coherence.

SECOND CANDIDATE, ALSO DEAD. By CRT every symmetric tooth vector is
v_q = m^{-1} mod q for some integer m, and the twin machine is m = 6. P13
predicted the feature is "m is small". REFUTED (research/tooth_msweep.py, log
research/data/tooth_msweep.log): over m = 1..60 coprime to the gears at m19 the
F values have median 28.0 - EXACTLY the full family's median - with m = 1 giving
33, m = 2 giving 34 and m = 4 giving 32, while the sweep's minimum F = 20 is at
m = 12, not at the twin's m = 6 (F = 25). Small m is not low-F, and 6 is not
distinguished among small m.

So two natural mechanisms are refuted and the effect stands unexplained. That is
the honest state: a real, exactly-measured, consistently-signed anomaly with no
mechanism, which by the project's own measurement directive is a target rather
than a wall.

## 6. PRIOR-ART CHECK

Not yet checked. Terms to run: "Jacobsthal function admissible tuple dependence";
"maximal gap reduced residue system varying residue classes extremal"; "covering
systems two residues per prime largest uncovered run"; "Jacobsthal function
g(n,k) / Erdos-Rankin tooth placement extremal". NOTE: the closest classical
object is the extremal question "choose one (or two) residue classes per prime to
maximise the longest uncovered run", which IS studied (Erdos-Rankin,
Ford-Green-Konyagin-Tao style constructions). The DELTA to check is the
DISTRIBUTION over all choices and the LOCATION of the arithmetically-forced
choice inside it, which is a different question from the extremum.

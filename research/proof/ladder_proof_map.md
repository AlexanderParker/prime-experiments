# The twin ladder: proof map, audit and certificate (2026-09-19)

Laboratory rounds 1-9 (tree nodes R5.f.i-x; math lane on Opus with fresh context, PM the manager,
Lean closing behind). This page is the map a reader should hold: what is proved, what is measured,
what is open, and what was withdrawn.

## 1. The objects

- **Twin centre** `s`: `6 | s`, `s - 1` and `s + 1` both prime (the twin `(P, P+2)`, `s = P + 1 = 6c`).
- **Stretch of a twin** (normal form, PROVED, `proofs/TwinLadder.lean`): the columns strictly between
  `P^2` and `(P+2)^2` are exactly `6c^2 + j`, `|j| <= 2c - 1`, with members `s^2 + 6j - 1` and
  `s^2 + 6j + 1`; the centre member is `P(P+2) = s^2 - 1`; the twin gears `P`, `P+2` strike the stretch
  only at the centre (`twin_gear_strikes_centre_only`).
- **Rung** `s -> s'`: a twin centre `s' = s^2 + 6j` inside the stretch (`proofs/TwinLadderTheorem.lean`).
- **Base-open offset at depth x**: neither member has a prime factor `<= x`; canonical depth `x = floor(sqrt s)`.
- **Ladder hypothesis** (`LadderHyp`): every twin centre has a rung. **Near twin hypothesis**
  (`NearTwinHyp B`): a rung among the first `B(s)` base-open offsets ordered by `|j|`.

## 2. The map

```
[D1] columns / members / gears                        KERNEL (pre-existing)
[D2] TwinCentre s                                     KERNEL  TwinLadderTheorem.lean
[D3] stretch in centre coordinates                    KERNEL  TwinLadder.lean          <- L1
[D4] base-open offset at depth x                      definition
[D5] Rung s s'                                        KERNEL  TwinLadderTheorem.lean
[D6] LadderHyp                                        KERNEL  TwinLadderTheorem.lean
[D7] NearTwinHyp B                                    KERNEL  TwinLadderTheorem.lean

[L1] normal form                                      KERNEL, PROVED
[L2] twin gears strike only the centre                KERNEL, PROVED
[L3] record pigeonhole: L columns -> >= L/(F+1) open  KERNEL, PROVED (LadderBase.lean), conditional on a run bound F
[L3a] run bound F(x) < 2x^2/3 at depth x = sqrt s     *** OPEN *** exact to x = 61 (F = 179); measured 0.7 x ln x;
                                                       best proved two-class exponent 4.266 - vacuous here
[L4] min-gap: strikes of one gear >= (g-1)/3 apart    PROVED (kernel: ManyBody.teeth_separation); off the critical path
[L5] square-phase eligibility (g | s^2 - A => A square mod g)  PROVED (KillPositions.lean form); off the critical path
[L6] rung_gt: a rung climbs                           KERNEL, PROVED
[L7] twinCentre_unbounded                             KERNEL, PROVED
[L8] twins_unbounded_of_ladder                        KERNEL, PROVED
[L9] NearTwinHyp -> LadderHyp                         KERNEL, PROVED
[C]  canonical ladder certified, six rungs to 48 digits   KERNEL (LadderCertificate.lean, LadderPratt.lean)
[H]  the ladder hypothesis                            *** OPEN *** measured at every twin lower to 10^6
```

Critical path: `H --L9--> D6 --L8--> twin primes above every bound`. `L3` is not on it: it turns the
hypothesis from an existence statement into a bounded per-rung check and drags in `L3a`. `L4`, `L5`
constrain which gear plugs which offset; they closed the covering and rigidity routes (rounds 6-8)
and are used by nothing in the theorem.

## 3. The statement, correctly delimited

Let `(P, P+2)` be twin primes and `s = P + 1 = 6c`. The integers strictly between `P^2` and
`(P+2)^2` are exactly the pairs `s^2 + 6j -+ 1`, `|j| <= 2c - 1`, and the only member of that range
divisible by `P` or `P + 2` is `P(P+2) = s^2 - 1` (proved, formalised). A rung is a twin pair strictly
between `P^2` and `(P+2)^2`; the ladder hypothesis asserts every twin pair has a rung. Formalised and
proved: the ladder hypothesis implies twin primes above every bound, by induction from `(5, 7)`,
since a rung strictly increases the centre. Also proved and formalised: a window of `L` columns
contains at least `L/(F(x)+1)` base-open offsets, `F(x)` the longest run of consecutive columns
struck by the primes `<= x` - so each rung's search is a bounded explicit list PROVIDED
`F(x) = O(x^(2-e))`, a two-class Jacobsthal bound stronger than any proved. The single open
ingredient of the ladder itself is the ladder hypothesis, verified for every twin pair with
`P <= 10^6`; in its bounded form (a twin among the first `ceil(4 ln s)` base-open offsets, verified
over the same range, max ratio 3.29) it is a twin-prime existence statement in an interval of length
about `(log N)^3` around `N = s^2`; in its weakest form it is the twin-prime analogue of Legendre's
conjecture between `P^2` and `(P+2)^2`. Nothing in the chain reduces either open ingredient to a
known theorem; the contribution is the reduction of the infinitude of twin primes to one localised
short-interval statement, with the candidate list, the phase structure and the plug constraints
proved, and the ladder certified in the kernel to a 48-digit twin centre.

## 4. The audit (lane round 9, unsparing)

| round | claim | status |
|---|---|---|
| 1 | square-scale transfer | MEASURED-HELD; its use as an inductive step WITHDRAWN (lands one level short) |
| 1 | single-plug column | MEASURED-HELD; gate not established |
| 1 | near pairs, separation (g-2)/6 | MEASURED-HELD; constant superseded by (g-1)/3 |
| 2 | normal form; twin gears at the centre only | PROVED, kernel |
| 2 | the ladder (every twin's stretch holds a twin) | MEASURED-HELD to 10^6; OPEN = LadderHyp |
| 2 | 6|j| < 20 (ln P)^2 | MEASURED-HELD (worst 10.3); fitted constant, factor-2 margin |
| 2 | >= 2 twins per rung; top band idle | MEASURED (min 3); idleness PROVED |
| 3 | inheritance lemma | PROVED |
| 3 | locator with the y-rule | MEASURED-HELD to 10^6 but FAILS ASYMPTOTICALLY (yield -> 0); fixed y REFUTED |
| 3 | clean rungs at 13 | REFUTED as stated (17, 29, 41, 269); holds from 271 |
| 3 | count ratio >= 1/2 | REFUTED at P = 71 (0.472) |
| 4 | progression form M <= 4 sqrt c | REFUTED (19 short progressions); capture identity is a mean-level identity (0.19 to 5x per twin) |
| 4 | sieve form (index law, mean 4) | MEASURED-HELD; became the rung |
| 4 | steering s' = 0 mod Q | MEASURED-HELD above thresholds |
| 4 | algebraic offsets | difference-of-squares failure PROVED; enhancement predictions PARTLY WRONG (0.97/0.82 asymmetry unexplained; -(2c-1) is 0.03 by 5 | P^2 + 4 or P^2 + 6, found by measurement) |
| 5 | record pigeonhole | PROVED, kernel - CONDITIONAL on a run bound (see flag) |
| 5 | index <= 24 | REFUTED (29, 34); a constant cannot bound a geometric tail's maximum |
| 5 | canonical ladder predictions | held on nine rungs (too few to test a law) |
| 6 | index <= ceil(4 ln P) | MEASURED-HELD to 10^6 (max ratio 3.29) |
| 6 | square-phase eligibility | PROVED; calibration 0.245 |
| 6 | reuse (2/g)^2 per pair | REFUTED by a factor 2; wrong model, and the refutation line was set around the wrong model |
| 7 | exact reuse expectation | HELD after the statistic was corrected; residual +5 to +7% (two to three sigma) OPEN |
| 7 | universal list | PROVED; steered index 4.53 +- 0.21 against a stated band 3.6-4.4: marginally above |
| 7 | no impossible block | MEASURED-HELD; the CRT sentence was heuristic, not a proof |
| 8 | deficit is the statistic | HELD |
| 8 | bound tiers, s0 ~ 6.8 x 10^4 | MEASURED-HELD (8 exceptions, last s = 12,162); s0 and the interval lengths are estimates from measured laws |

**The serious flag.** "The record pigeonhole supplies the candidate list unconditionally" (rounds
5-8) is wrong as written: the pigeonhole is a theorem, but instantiating it at depth `x` needs an
upper bound on `F(x)`; exact values stop at `x = 61`, and to guarantee even one base-open offset
one needs `F(x) < 2x^2/3`, a two-class Jacobsthal bound of exponent 2 against the proved 4.266.
Read every "unconditional" of those rounds as "conditional on `F(x) = O(x^(2-e))`, open".

**General.** Every density used (W(x), 2C_2/ln^2 N, Buchstab, the 1/delta^2 law) is a heuristic; the
index law cannot be proved as stated (it implies twins in short intervals); several tests were
called exact when the computation was exact and the model was not.

## 5. The certificate

Per rung: `j_k`, `s_{k+1} = s_k^2 + 6 j_k`, the bound `|j_k| <= s_k/3 - 1` (equivalent to the containment
by the normal form), and Lucas/Pratt certificates for both members. Digits by rung: 2, 3, 6, 12, 24,
48, 96, 190. Certified in the kernel (`proofs/LadderCertificate.lean` with `LadderPratt.lean`, generated
by `research/stack/r8/gen_pratt.py`): six rungs, `6 -> 30 -> 882 -> 777978 -> 605249768610 ->
366327282402458541331692 -> 134195677832370611289419659133121237872763586710`; the 48-digit members'
`p - 1` factored in one second each (largest prime cofactor 30 digits). Rung 7 (96 digits) is a GNFS
computation; rung 8 (190) needs ECPP and has no Lean checker. The composition theorem
`twins_unbounded_of_ladder_above`: the ladder hypothesis restricted to twin centres at least the
48-digit one gives twin primes above every bound - so the hypothesis is assumed only above `10^47`,
where the supply lemma and the index law have their widest measured margins.

## 6. Round 10: the map at fixed depth, the cube bound, the name, the path (2026-09-19 14:10)

- **L3@61**: the pigeonhole instantiated at depth 61 with the exact record F(61) = 179 - every
  twin's stretch holds >= (4c - 1)/180 61-rough offsets, no open input. L3a leaves the map.
- **The rate, exact**: W(61) = prod_{5<=g<=61}(g-2)/g = 0.137565; twin centres among offsets near
  s^2 at 3 C_2/(ln s)^2; mean first-twin index 0.0695 (ln s)^2 (measured 0.85-0.94 of it from
  ln s >= 7).
- **The bound is a cube**: the index is geometric, so its maximum over the twins below a cap is
  the mean times ln N ~ ln s; B(s) = ceil(0.0695 (ln s)^3). Predicted maximum to 10^6: 85-120 and
  i/(ln s)^2 in [0.45, 0.63]; observed 107 and 0.598. Refutation line i > 0.07 (ln s)^3: no twin to
  10^6 breaches it. Supply 0.0695 (ln s)^3 <= (2s/3 - 1)/180 from s ~ 1.8 x 10^4; the prefix ends at
  1.34 x 10^47.
- **The name**: the first B(s) 61-rough offsets span 43.6 B = 3.03 (ln s)^3 = 0.379 (log N)^3
  integers about N = s^2. NTH_61 is the twin analogue of Cramer's conjecture (consecutive twin
  pairs near N at most O((log N)^3) apart) restricted to N = s^2, s a twin centre. The mean tier
  0.76 (log N)^2 equals the average twin gap (log N)^2/(2 C_2) = 0.757 (log N)^2 to three digits.
- **Depth cancels**: H = 6B/W(x) is independent of the depth x; ordering, steering and the
  one-level-down candidates move nothing from measured to proved. The reduction is complete as a
  reduction; the open node is the conjecture localised to the squares of twin centres.
- **The weakest form that composes is a path**: `twins_unbounded_of_path` (a sequence of twin
  centres each a rung of the previous gives twins unbounded) and `path_of_ladderHyp_above` (the
  universal hypothesis above a bound yields a path by dependent choice) - kernel,
  TwinLadderTheorem.lean. By Konig the induction needs only that the rung tree rooted at the
  certified top is infinite (every measured stretch has >= 3 twin centres).
- **One sentence**: infinitude of twin primes follows, by an induction formalised in Lean from
  (5, 7) through a Pratt-certified prefix of six rungs reaching 1.34 x 10^47, from the single
  statement that every twin centre s >= 1.34 x 10^47 (or merely every centre on one path) has
  another twin pair strictly between (s-1)^2 and (s+1)^2 - the twin analogue of Cramer's conjecture
  localised to the squares of twin centres - with the candidate list at each step supplied
  unconditionally by the exact record at depth 61.

# Branch 2f.i - separation compatibility as the chain statement's ingredient

Parent: node 2f of `research/proof/theory_tree.md` (the adjacent-teeth sub-family), REFUTED by
prover C's 23 -> 29 sweep. The observation that spawned this branch is the untouched half of
that refutation, written into `research/proof/dead_branches_reopened.md` entry 2f: *what does
the refuting member have that the real machine lacks?* Weak point W3 of
`research/proof/the_wall.md` answered one half of that question for the cover number - the real
machine's separations are compatible under CRT, `r (S_g + S_h) = c (mod gh)`, 32,490 pair-checks
with 0 exceptions - and thin place 4 of the unstick file names the untried test: compatibility as
the ingredient of the CHAIN statement, not of `K`.

Scripts: `research/anchor235/r46/cp_*.py`. Result outputs (untracked):
`research/anchor235/r46/results/`. Every number this document relies on is written into the
document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 What this branch could find that is not already known

Known already: (a) the real machine's separations are CRT-compatible and no other family the
record has built is (W3, `separation_drives_K.md` N-S2/N-S3); (b) compatibility does **not**
drive `F` (branch 6) and does **not** drive `K` (W3); (c) the chain statement fails on the
counterfactual family and every ingredient set short of the real higher gears' teeth has a
counterexample (2f). Nobody has asked whether the family's chain violators are exactly its
INCOMPATIBLE members. If they are, the chain statement's missing ingredient is an arithmetic
property of the residues (which face C explicitly allows: "the specific teeth as arithmetic,
not as a symmetry") rather than "the real teeth"; if they are not, the branch returns a
counterexample and closes the last property the record has that is the real machine's own.

### 0.2 Setup: the exact definition of incompatibility

The tooth-counterfactual family (`docs/proof-search/alignment-rules.md` section 5): machine
`M = {5..p}`, gears `q`, teeth at `+-v_q` with `v_q in 1..(q-1)/2`; incoming gear `q'` with tooth
`v'`, letters `a = min(2v', q' - 2v')`, `b = q' - a`. The real machine is `v_q = round(q/6)`.

**Separation.** Gear `q`'s two struck classes are `+-v_q`, so their difference is `+-2 v_q`. Write
`s_q = 2 v_q (mod q)`. The sign is a gauge: which of the two teeth is called "first" is the
mirror, and the record's unsigned form `s_g = 2 u_g` with `u_g = 6^{-1}` picks one. So every
statement below is invariant under `s_q -> -s_q` per gear, and the real machine has
`3 s_q = +-1 (mod q)` at every gear (`6 u_q = q -+ 1`), i.e. `s_q` is the same rational one third
at every gear once the gauge is fixed.

**Compatibility at a rational.** Fix a bound `B`. An *admissible rational* is a pair of integers
`(r, c)` with `1 <= r, c <= B`, `gcd(r, c) = 1`, and both `r` and `c` coprime to every gear of the
machine (`r` must be invertible mod `gh`, as the record's identity requires; `c = 0 mod q` would
put the two teeth on top of each other). Gear `q` is **(r,c)-coherent** iff

        r s_q = c  or  r s_q = -c   (mod q).

A pair of gears `(g, h)` is **compatible at (r,c)** iff both are `(r,c)`-coherent, which is
exactly the record's identity at that pair: `r (S_g + S_h) = c (mod gh)` with
`S_g = CRT(s_g, 0)`, `S_h = CRT(0, s_h)`, up to the per-gear gauge.

**The incompatibility count.** For a member with gear set `G` (`n = |G|`),

        k(member) = max over admissible (r,c) of #{q in G : q is (r,c)-coherent}
        I(member) = C(n,2) - C(k,2)
                  = the number of gear pairs not compatible under the best single rational,

and the **incompatible pairs** are the pairs with at least one gear outside a maximum core. `I = 0`
iff every gear takes the same rational separation. `B = 30` is the branch's default (it covers
every coherent family of W3, whose rationals are `c/r` with `r <= 13`); `B = 200` is run as a
robustness row. Both `I` including the incoming gear `q'` and `I` on the old gears alone are
reported.

**A structural note fixed before computing** (it decides which coherent families exist at all):
`r` and `c` must be coprime to every gear, so for a machine `{5..q}` the admissible rationals are
exactly those whose numerator and denominator have all prime factors in `{2, 3}` or above `q`.
The coherent families `r = 5, 7, 11` of W3 therefore do NOT exist as full-machine families here -
their denominator is one of the machine's own gears - and are swept below as members of the
"one gear off the rational" sub-family, with that gear free. The machine's own one third,
`c/r = 1/3`, is 3-smooth and always admissible.

### 0.3 Predictions, with numbers, and what refutes each

**PK1 - the gate.** `I = 0` on the real machine at m11..m29 (both with and without `q'`), and
`I = 0` on every coherent family member built from an admissible rational, by construction and by
direct check. REFUTED, and the branch stops, if the real machine has `I > 0`.

**PK2 - every violator is incompatible.** Every recorded violator - the budget violators (chain
or pair) and the chain violators on the full family at m11..m19, plus the two 23 -> 29 refuting
members - has `I >= 1`. **Stated in advance as nearly vacuous**: the coherent members number a few
hundred against 142,560 at m19, so a violator drawn at random is incompatible with probability
above 0.99. The prediction that is not vacuous, and the one scored: at the 2f refuting member
`(1,1,4,2,7,1,5)` the maximum core is exactly the set of gears carrying real teeth, `{5, 7, 13}`,
so `k = 3`, `I = 18` of 21 pairs, and every incompatible pair contains one of the moved gears
`11, 17, 19, 23`. REFUTED if the best core is larger than the real-teeth set at that member, or if
some violator anywhere has an incompatible pair with both gears real.

**PK3 - the graded form (the test with power).** On the full m19 family (142,560 rows on disk,
exhaustive) the chain-violation rate is monotone decreasing in the core size `k`: with `n = 7`
gears (`5..19` plus `q' = 23`) the rate at `k = 1, 2, 3, 4, ...` falls, and in particular the rate
at `k = n-1` (one gear off the rational) is **below one third of the rate at `k <= 2`**. REFUTED
if the rate at `k = n-1` is at or above the overall family rate 0.135%, or if the sequence is
non-monotone at two or more steps.

**PK4 - the converse, the coherent sub-family.** Exhaustive sweep of every admissible-rational
coherent member at 23 -> 29 with the incoming tooth pinned: **0 chain violators, 0 pair
violators**. Stated in advance as weak evidence: the sub-family has of order 100 members against a
base rate near 0.01-0.35%, so 0 is what the null predicts too. It is a necessary condition, not
evidence.

**PK5 - the deciding test: does ONE incompatible gear already permit violation?** The sub-family
"compatible in all but one gear" at 23 -> 29, pinned incoming tooth, exhaustive (every admissible
rational x every gear x every value of that gear's tooth), with a **size-matched random pinned
control** run in the same script so the base rate is measured, not assumed. Prediction: **at least
one chain violator in the one-off sub-family**, and its rate statistically indistinguishable from
the control's - i.e. one incompatible pair already permits violation and compatibility is not the
ingredient. REFUTED, and compatibility becomes the candidate ingredient, if the one-off sub-family
has 0 violators over at least 3,000 rows while the matched control has at least 2.

**PK6 - mechanism: does the violation use a pair overlap compatible separations forbid?** For a
pair `(g, h)` write `D_{gh} = CRT(s_g, s_h) (mod gh)` (the diagonal of the four struck residues
`{0, S_g, S_h, S_g + S_h}`) and `delta_{gh} = min(D, gh - D)`. N-S3: for compatible separations at
the rational `c/r`, `delta = min(c, r-c)/r * gh` up to `O(1)`, so for the machine `delta >= gh/3`
at every pair, 0 exceptions in 4,095. Prediction: at the 23 -> 29 refuting member's violating
stretch at least one pair of gears realises a **double-strike pair at distance `delta < gh/3`**
inside the stretch - a configuration no compatible member can have at that pair - and that pair
involves a moved gear. REFUTED if every double-struck pair inside the violating stretch is at a
distance a compatible member could also realise (`delta >= gh/3`), i.e. the violation uses nothing
compatibility forbids.

**PK7 - the alternative property, if compatibility does not separate.** The tail gears' tooth
distance (W3's N-S4: real `sep_q / q = 1/3` at every gear, against a random mean 1/4 and a range
`(0, 1/2]`). On the full m19 family, stratify the chain-violation rate by `min_q sep_q / q` and by
`mean_q sep_q / q` over the tail gears `q >= 17`. Prediction, from prover C's finding that
separation at `q >= 11` is not the carrier: **no threshold on either statistic isolates the
violators** - every threshold leaving at least 10,000 members above it still contains a violator.
REFUTED by a threshold with 0 violators above it and at least 10,000 members above it.

**PK8.** Everything that holds without exception over the computed range, with counts.

### 0.4 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| PK1 | `I = 0` on the real machine m11..m29 and on every coherent member | **CONFIRMED**: `k = n` at all twelve rows, best rational `(r,c) = (3,1)` at every one; the CRT form `r(S_g+S_h) = c mod gh` checked directly at the 36 pairs of m29+31, 0 failures; 71 / 47 coherent members at m19 / m23, 0 with `I > 0` (2.1) |
| PK2 | every recorded violator has `I >= 1`; the 2f member has core `{5,7,13}`, `I = 18` | **REFUTED on the first clause**: three recorded budget violators are FULLY COMPATIBLE, `I = 0` - one at m11 (rational 29/18) and two at m17 (rationals **8/1** and **6/1**), all three re-verified by direct sieve of `M + q'` (2.2, 2.3). CONFIRMED on the second: the 2f member's core is `{5, 7, 13}` plus the coherent incoming gear 29, `k = 4` of 8, `I = 22` of 28, and every incompatible pair contains one of the moved gears 11, 17, 19, 23 |
| PK3 | violation rate monotone in `k`; rate at `k = n-1` below a third of the rate at `k <= 2` | **REFUTED, and in the opposite direction at m17**: rate by `k` is 0.000 / 0.273 / 0.291 / **2.062**% at `k = 3..6` at m17 (the fully coherent cell is 7.4x the family rate 0.278%) and 0.107 / 0.145 / 0.117 / 0.076 / 0.000% at `k = 3..7` at m19 (2.4). Robust to the bound: at `B = 10` the m17 coherent set is 25 members with 2 violators, 8.0%, a factor 29 (2.5) |
| PK4 | coherent sub-family at 23 -> 29: 0 violators (necessary, not evidence) | **CONFIRMED and uninformative as pre-registered**: 71 members (old gears coherent, incoming tooth pinned) and 47 (all gears coherent), 0 budget violators; expected under the null below 0.1 (3.1) |
| PK5 | one-off sub-family: at least one violator, rate matching the control | **CONFIRMED**: 2,627 members "coherent in all but one gear", **1 chain violator** (teeth `(2,2,1,2,8,7,5)`, `v' = 5`, odd gear 19, `Q*_4 = 65 > 63`), against **0** in a size-matched random pinned control of 2,627 (3.2) |
| PK6 | the violating stretch uses a short diagonal (`delta < gh/3`) at a moved-gear pair | **REFUTED**: 7 of 21 pairs realise a double-strike below `gh/3` inside the 2f member's violating stretch, but so does the REAL machine inside its own maximal run (5 of 21 at m23, distances 2, 5, 7 at the pair (5,7)); the short distances come from the ANTI-diagonal `d- = CRT(s_g, -s_h)`, which coherence does not constrain (4.1) |
| PK7 | no tooth-distance threshold isolates the violators | **CONFIRMED**: an m19 violator has every tooth distance `sep_q/q >= 0.2941`, above the real machine's minimum 0.2857 - teeth `(1,2,2,2,6,3)`, `v' = 6`, `F = 25`, direct sieve `F(M+23) = 52 > 48` (5.1) |
| PK8 | exception-free statements with counts | five, listed in section 6 |

---

## 1. Setup (exact ranges)

No sampling except the one row that says so. Scripts in `research/anchor235/r46/` (prefix `cp_`);
outputs (untracked) in `research/anchor235/r46/results/`.

| object | range | script |
|---|---|---|
| the incompatibility count `I`, the core `k`, the best rational | real machine m11..m29 with and without `q'`; every coherent member at m17, m19, m23 | `cp_compat.py gate` |
| `I` at every recorded violator | the full tooth-counterfactual family at m11, m13, m17, m19 (180 / 1,440 / 12,960 / 142,560 rows, prover C's `chain_teeth_r33_fam_m*.json`) and the m23 `(T)+(L)` sweep (22,400 rows, `chain_teeth_r33_sub_m23.json`) | `cp_compat.py viol`, `cp_detail.py` |
| independent re-verification of the compatible violators by DIRECT sieve of `M + q'` | 3 violators + 2 real controls | `cp_verify.py` (reuses `chain_family_r32.direct_F_new`) |
| the full family stratified by `k`, by `(T)`, by tooth distance, by the two pair diagonals, by resonance | m17 (12,960) and m19 (142,560), exhaustive | `cp_strat.py` |
| bound robustness | `B = 10, 30, 60, 120, 240` at m17 and m19, exhaustive | inline, 2.5 |
| the coherent sub-family at 23 -> 29, incoming tooth pinned | 71 members, one lower period (37,182,145) each | `cp_sweep.py coh` |
| the fully coherent sub-family at 23 -> 29 (incoming gear coherent too) | 47 members | `cp_sweep.py cohfull` |
| "coherent in all but one gear" at 23 -> 29, incoming tooth pinned | 2,627 members, exhaustive over every admissible rational x every gear x every other tooth value | `cp_sweep.py oneoff` |
| size-matched random pinned control at 23 -> 29 | 2,627 members, fixed seed 46 (the only sampled row) | `cp_sweep.py ctrl` |
| the violating chain and its pair overlaps | 2f refuting member, both m17 compatible violators, 2 real controls | `cp_mech.py` |

**The family** is prover B's and prover C's unchanged: gears `{5..p}` with teeth at `+-v_q`,
`v_q in 1..(q-1)/2`, incoming gear `q'` with tooth `v'`; `F`, `F_2`, `Q*_J` computed on one lower
period from the cyclic gap sequence (`chain_family_r32.qstar_table`), the budget inequality being
`max(F_2, max_J Q*_J) = F(M + q') <= F(M) + q'` by the attainment identity. Every violator quoted
below was re-derived from the recorded rows, and the four headline ones were re-verified by a
direct sieve of `M + q'`.

## 2. Results: the incompatibility count

### 2.1 The gate (PK1)

`I = 0` on the real machine at every level, with and without the incoming gear, always at the same
rational `(r, c) = (3, 1)`:

| machine | gears | teeth | admissible rationals at `B = 30` | `k` | `I` |
|---|---|---|---|---|---|
| m11 | 5, 7, 11 | 1, 1, 2 | 203 | 3 of 3 | 0 |
| m13 + q' | 5..17 | 1, 1, 2, 2, 3 | 125 | 5 of 5 | 0 |
| m17 + q' | 5..19 | 1, 1, 2, 2, 3, 3 | 97 | 6 of 6 | 0 |
| m19 + q' | 5..23 | 1, 1, 2, 2, 3, 3, 4 | 71 | 7 of 7 | 0 |
| m23 + q' | 5..29 | 1, 1, 2, 2, 3, 3, 4, 5 | 47 | 8 of 8 | 0 |
| m29 + q' | 5..31 | 1, 1, 2, 2, 3, 3, 4, 5, 5 | 47 | 9 of 9 | 0 |

and the identity in its CRT form, `r (S_g + S_h) = c (mod gh)` with `(r, c) = (3, 1)`, holds at all
36 gear pairs of m29 + 31 with 0 failures, `min(D, gh - D) >= gh/3` at each - W3's N-S2 / N-S3
reproduced inside the family's parameterisation, with the per-gear sign gauge made explicit.

**A structural fact fixed by the gate, and new.** `r` and `c` must be coprime to every gear, so the
rationals available to a machine `{5..q}` are exactly those whose numerator and denominator are
3-smooth or have all prime factors above `q`: the machine eats its own denominators as it grows.
The count at `B = 30` falls 203, 155, 125, 97, 71, 47 at m11, m11+13, m13+17, m17+19, m19+23,
m23+29. The machine's own rational `1/3` survives every level because 3 belongs to the ANCHOR, not
to the gears - the only reason a coherent separation exists at all. Distinct admissible rationals
give distinct coherent members with no collision up to `B = 30` (25/25, 97/97, 71/71, 47/47); the
first collisions appear at `B = 60` (463 members from 469 rationals at m17).

### 2.2 Every recorded violator (PK2)

Budget violators = chain violators (`max_J Q*_J > F + q'`) plus pair violators (`F_2 > F + q'`).

| level | rows | budget violators (chain / pair) | core size `k` of the violators (`n` gears) | violators with `I = 0` |
|---|---|---|---|---|
| m11 + 13 | 180 | 1 (1 / 0) | `k = 4` of 4 | **1** |
| m13 + 17 | 1,440 | 1 (1 / 0) | `k = 4` of 5 | 0 |
| m17 + 19 | 12,960 | 36 (36 / 0) | `k = 4` x27, `5` x7, `6` x2 | **2** |
| m19 + 23 | 142,560 | 203 (193 / 11, one member both) | `k = 3` x5, `4` x153, `5` x43, `6` x2 | 0 |
| m23 + 29, `(T)+(L)` | 22,400 | 4 (3 / 1) | `k = 4` x4 of 8 | 0 |

The 2f refuting member, in the branch's terms: teeth `(1, 1, 4, 2, 7, 1, 5)`, `v' = 5`, separations
`(2, 2, 8, 4, 14, 2, 10)` against the real `(2, 2, 4, 4, 6, 6, 8)`; best rational `1/3` with
coherent core `{5, 7, 13, 29}`, `k = 4` of 8, `I = 22` of 28 pairs. The moved gears are exactly
`11, 17, 19, 23` and every incompatible pair contains one of them. **The second clause of PK2 is
confirmed**: the refutation of 2f is an incompatible member, and its incompatibility is carried by
the gears it moved.

The m23 `(T)+(L)` sweep also carries a PAIR violator that prover C's log did not print
(`F_2 > F + 29` at teeth `(1,2,3,3,1,3,9)`), so that sub-family has 4 budget violators, not 2.

### 2.3 The counterexample: compatible members that violate (PK2 refuted)

Three recorded budget violators have `I = 0` - every gear on one rational, the same property the
real machine has. All three re-verified by a direct sieve of the full `M + q'` period
(`cp_verify.py`, the attainment gate G2 agreeing cell for cell):

| level | rational `c/r` | teeth (old, `v'`) | separations | `a` | `F` | `F_2` | budget | `max_J Q*_J` | direct `F(M+q')` | excess |
|---|---|---|---|---|---|---|---|---|---|---|
| m11 + 13 | 29/18 | `(1,1,5)`, 1 | 2, 2, 10 | 2 | 11 | 13 | 24 | 25 (`J=5`, `(2,11,2)`) | 25 | +1 |
| m17 + 19 | **8/1** | `(1,3,4,4,4)`, 4 | 2, 6, 8, 8, 8 | 8 | 19 | 27 | 38 | 40 (`J=4`, padded `(8,19)`) | 40 | +2 |
| m17 + 19 | **6/1** | `(2,3,3,3,3)`, 3 | 4, 6, 6, 6, 6 | 6 | 18 | 24 | 37 | 38 (`J=5`, `(6,13,6)`) | 38 | +1 |

The two m17 members are the cleanest coherent families there are: **the same integer separation at
every gear**, `s_q = +-8` and `s_q = +-6`, rationals `8/1` and `6/1`, both inside the bound
`B = 10`. They are coherent in exactly the sense W3 measured, and they break the budget inequality.
The real m17 machine at the same incoming tooth is the control: `F = 18`, `max_J Q*_J = 25`,
budget 37, direct `F(M+19) = 25`.

Note the shape: in all three the incoming letter `a` EQUALS the common separation (`a = 8 = s_q`;
`a = 6 = s_q`; `a = 2 = s_5 = s_7 = s_13`). Section 4.2 takes that up and finds it is not a law.

### 2.4 The graded form (PK3 refuted)

Chain-violation rate by core size, exhaustive, `B = 30`:

| `k` | m17: members / violators / rate | m19: members / violators / rate |
|---|---|---|
| 3 | 572 / 0 / 0.000% | 4,666 / 5 / 0.107% |
| 4 | 9,888 / 27 / 0.273% | 99,381 / 144 / 0.145% |
| 5 | 2,403 / 7 / 0.291% | 35,815 / 42 / 0.117% |
| 6 | 97 / 2 / **2.062%** | 2,627 / 2 / 0.076% |
| 7 | - | 71 / 0 / 0.000% |
| all | 12,960 / 36 / 0.278% | 142,560 / 193 / 0.135% |

Monotone decrease is refuted at both levels. At m17 the fully coherent cell is 7.4 times the family
rate; at m19 the `k = 7` cell has 71 members and an expected violator count of 0.10, so its zero is
not evidence of anything. Crossed with `(T)` (no adjacent teeth) the same holds: at m19 the cell
`(k = 7, (T))` has 6 members and `(k = 6, (T))` has 280, expected violators 0.01 and 0.6.

### 2.5 Bound robustness

Fully coherent members and the violators among them, exhaustive at each bound:

| `B` | m17 rationals / coherent members / violators | m19 rationals / coherent members / violators |
|---|---|---|
| 10 | 25 / 25 / **2** (8.0%) | 25 / 25 / 0 |
| 30 | 97 / 97 / **2** (2.06%) | 71 / 71 / 0 |
| 60 | 469 / 463 / 3 (0.65%) | 395 / 395 / 2 (0.51%) |
| 120 | 2,347 / 2,237 / 8 (0.36%) | 2,049 / 2,045 / 5 (0.24%) |
| 240 | 10,271 / 7,491 / 24 (0.32%) | 9,263 / 9,069 / 11 (0.12%) |
| family rate | 0.278% | 0.135% |

The refutation is strongest at the TIGHTEST bound - 2 of the 25 smallest-rational coherent members
at m17 violate. As `B` grows the coherent set fills the family and its rate falls to the family
rate, exactly as a property with no content should.

## 3. The converse sweeps at 23 -> 29

Machine `{5..23}`, `q' = 29`, one lower period of 37,182,145 columns per member; incoming tooth
pinned to `v' = 5` unless the row says otherwise (`a = 10`, `b = 19`, `3a = q' + 1`).

### 3.1 The coherent sub-families (PK4)

| sub-family | members | budget violators | with `(T)` | violators with `(T)` |
|---|---|---|---|---|
| old gears coherent, `v'` pinned | 71 | 0 | 6 | 0 |
| all gears coherent, incoming gear included | 47 | 0 | 4 | 0 |

Confirmed, and as pre-registered it is **not evidence**: at the m23 pinned base rate measured in
3.2 (0 in 2,627) the expected violator count over 118 members is below 0.1. The families for
`r = 5, 7, 11` named in the brief do not exist at this machine - their denominator is one of its
own gears (2.1) - and appear instead inside 3.2 with that gear free.

### 3.2 One incompatible gear (PK5), against a matched control

| sub-family | members | budget violators | rate | with `(T)` | violators with `(T)` |
|---|---|---|---|---|---|
| coherent in all but one gear, `v'` pinned | 2,627 | **1** (chain) | 0.038% | 280 | 0 |
| random pinned control, seed 46 | 2,627 | 0 | 0.000% | 394 | 0 |

The violator: teeth `(2, 2, 1, 2, 8, 7, 5)`, `v' = 5`, `a = 10`; the base coherent member is
`(2, 2, 1, 2, 8, 3, 5)` and the odd gear is **19**, moved from `v = 3` to `v = 7`. `F = 34`,
`F_2 = 45`, budget 63, `Q*_4 = 65` - a literal depth-4 violation with excess 2. `k = 6` of 8,
`I = 13`. So **one incompatible gear already permits a chain violation**, and the one-off
sub-family's rate is at or above the random control's, not below it. The pre-registered refutation
condition for PK5 (0 violators over at least 3,000 one-off rows while the control has at least 2)
is not met in either direction, but the direction observed is the one that kills the hypothesis:
compatibility of all but one gear buys nothing measurable.

## 4. Mechanism

### 4.1 What the violating chain uses (PK6 refuted)

The 2f refuting member's violating run, located on the period (`cp_mech.py`): openings of `M` at
8,676,260 + offsets `(0, 18, 28, 47, 62)`, gaps `(18, 10, 19, 15)`, span 62 against budget 61. The
two interior openings that `q'` kills sit at residues `0` and `10` mod 29 with the middle opening at
`10`: the word is the literal `(a, b) = (10, 19)` and the kills alternate teeth exactly as T3
requires. The flanks are 18 and 15.

Pair by pair over the stretch, the columns struck by BOTH gears and the distances between them:
7 of the 21 pairs realise a double-strike at a distance below `gh/3` - `(5,7)` at 2, 5, 7;
`(5,11)` at 3, 8; `(5,13)` at 13, 17; `(5,19)` at 17; `(7,11)` at 14; `(7,13)` at 26, 30;
`(13,17)` at 17. **But the real machine does the same inside its own maximal run**: at m23 the real
`(10) + [10] + (23)` run has `(5,7)` double-struck at distances 2, 5, 7, 12, `(5,11)` at 15, 18,
`(5,13)` at 13, `(5,17)` at 23 - five pairs below `gh/3`.

The reason is exact, and it is the limit of N-S3. Two gears strike four residues mod `gh`, a
translate of `{0, S_g, S_h, S_g + S_h}`; that rectangle has TWO diagonals,
`d+ = CRT(s_g, s_h)` and `d- = CRT(s_g, -s_h)`, and coherence fixes only `d+ = c r^{-1} (mod gh)`.
For the machine `d+ = 3^{-1} = (gh +- 1)/3`, never short - that is N-S3 - but `d-` satisfies
`3 d- = CRT(1, -1)` and is an arbitrary residue: at `(5, 7)` the real machine's `d-` is 12 and the
folded distances inside its pattern are 2, 5, 7, 12. **The short overlaps a violating chain uses
live on the anti-diagonal, which no compatibility statement constrains.** So the mechanism the
branch was looking for does not exist: there is no pair configuration inside a violating stretch
that compatible separations forbid, and therefore no lemma about compatible pairs that could bound
the flanks.

For contrast, the compatible violator `8/1` has `d+ = 8` at EVERY pair (the coherent diagonal is the
integer `c` itself when `r = 1`), the shortest possible - the opposite extreme from the machine's
`1/3` - and it violates; the `1/2` coherent member, whose `d+ = (gh+1)/2` is LONGER than the
machine's, does not. Within the coherent set the diagonal ratio
`rho = min_pairs min(d+, gh - d+)/gh` orders the members (real machine `rho = 0.3273`; `8/1` has
`rho = 0.0324`), but it does not order the violation rate: at m17, `B = 240`, the rate by `rho` bin
is 0.279% on `[0, 0.05)`, 0.562% on `[0.05, 0.10)`, 0.901% on `[0.10, 0.15)`, then 0 over the 31
members above 0.15 - rising where there is support, and zero only where the expected count is 0.1.

### 4.2 The shape of the compatible violators, and why it is not a law

All three compatible violators have the incoming letter equal to the common separation, so a step
of `a` carries one tooth of every gear onto its other tooth simultaneously. Define the resonance
count `res = #{old gears q : a = +-2 v_q (mod q)}`. On the m17 family the cell `res = 5` (every old
gear resonant) has 7 members and 2 violators, 28.6% against the family's 0.278% - a factor 103, and
those 2 are exactly the compatible violators. On the m19 family the same cells are empty of
violators: `res = 5` 201 members 0, `res = 6` 7 members 0, `res = 4` 2,289 members 1 (0.044%), while
`res = 2` carries the highest rate (0.202%). So resonance describes the m17 counterexamples and is
not a law; it is recorded as the shape, not as a rule.

## 5. What else does not separate

### 5.1 The tail gears' tooth distance (PK7)

W3's N-S4 - the real tail gear's tooth distance is 0.688-0.697 of the arc, outside the entire
random range - is the brief's alternative property. Inside the family the analogue is
`sep_q / q = min(s_q, q - s_q)/q`, which the real machine holds at 0.2857-0.4783 (minimum 2/7 at
gear 7). It does not separate:

| `min_q sep_q/q` | m19 members | chain violators | rate |
|---|---|---|---|
| `[0.00, 0.05)` | 12,960 | 9 | 0.069% |
| `[0.05, 0.15)` | 102,720 | 158 | 0.154% |
| `[0.15, 0.25)` | 24,720 | 18 | 0.073% |
| `[0.25, 0.30)` | 1,800 | 8 | 0.444% |
| `[0.30, 0.45)` | 360 | 0 | 0.000% |

The bin containing the real machine's own value carries the HIGHEST rate, and the counterexample is
explicit: teeth `(1, 2, 2, 2, 6, 3)`, `v' = 6`, `a = 11` has every tooth distance at or above
0.2941 - above the real machine's minimum 0.2857 - with `F = 25`, budget 48, and a direct sieve
giving `F(M + 23) = 52`, an excess of 4. Restricting to the tail gears `q >= 17` gives no
separation either (rates 0.13, 0.17, 0.16, 0.09, 0.20, 0.07, 0.04, 0.12, 0.10% across the nine bins
of `min_{q >= 17} sep_q/q`).

### 5.2 The pair-diagonal statistics, and the power of the whole experiment

`rmax = min_pairs max(d+, d-)/gh` (the gauge-free form of N-S3; real machine 0.3273) has no violator
above 0.2471 at m19 and none above 0.2605 at m17 - but only 598 and 302 members sit above 0.25,
expected violators 0.8 and 0.8. `rmin = min_pairs min(d+, d-)/gh` puts the real machine at 0.0186,
INSIDE the violators' range `[0, 0.042]`, so it separates in the wrong direction.

**The general shape of every negative result here**: the region of the family where no violator
lives is the region where almost no members live. Every statistic tried - core size, tooth distance,
both diagonals, resonance - has a top bin of a few hundred members out of 142,560 and an expected
violator count below one. The family experiment cannot decide these questions by frequency; only a
construction can.

## 6. What holds without exception (PK8)

1. **The real machine is compatible at `(3,1)` at every level, with the incoming gear included.**
   `k = n`, `I = 0` at m11..m29, 12 rows, 0 exceptions; the CRT form at all 36 pairs of m29+31.
   (This is `6 u_q = q -+ 1` restated at the composite modulus - W3's N-S2, kernel-backed; noted,
   not new.)
2. **The machine eats its own denominators.** The admissible rationals of `{5..q}` are exactly
   those whose numerator and denominator are 3-smooth or have all prime factors above `q`; the
   count at `B = 30` is 203, 155, 125, 97, 71, 47 at m11 .. m29+31. `1/3` survives at every level
   only because 3 is in the anchor. Exact, 0 exceptions, and new.
3. **Distinct admissible rationals give distinct coherent members** up to `B = 30`: 25/25, 97/97,
   71/71, 47/47 at m17, m19, m23. First collisions at `B = 60` (463 of 469).
4. **Every coherent member at 23 -> 29 satisfies the budget inequality**: 0 of 118 (71 pinned +
   47 fully coherent). True, and by the measured base rate not evidence of anything.
5. **Every incompatible pair of the 2f refuting member contains a moved gear**: 22 of 28 pairs,
   core `{5, 7, 13, 29}`, moved gears `11, 17, 19, 23`, 0 exceptions. The same holds at all four
   m23 `(T)+(L)` budget violators.

## 7. Verdict

**Compatibility is not the chain statement's missing ingredient, and it is not even correlated with
the statement in the right direction.** The hypothesis is refuted by counterexample at the tightest
form of the definition: two members of the m17 family whose separations are the same integer at
every gear - the rationals `8/1` and `6/1`, both inside `B = 10`, both `I = 0`, both coherent in
exactly the sense W3 measured on the real machine - break the budget inequality, verified by a
direct sieve of the full `M + q'` period. At m17 the fully coherent members violate at 2 of 25
(`B = 10`), 8.0% against a family rate of 0.278%: coherence is a **liability** there, not a
protection. At m19 it is neutral (0 of 71, expected 0.1). One incompatible gear does not protect
either: the "coherent in all but one gear" sub-family at 23 -> 29 has a chain violator in 2,627 rows
where a size-matched random control has none.

The two positive halves stand and are worth keeping. First, the 2f refutation IS an incompatible
member and its incompatibility is exactly the gears it moved (`k = 4` of 8, core `{5, 7, 13, 29}`,
`I = 22`), so at that member "the real higher gears' teeth" and "compatible separations" name the
same set - which is why the test had to be run on the family and not on the counterexample. Second,
the reason coherence cannot be the ingredient is exact and is the limit of N-S3: two gears strike a
rectangle mod `gh` with TWO diagonals, and coherence fixes only one of them. The machine's `1/3`
makes `d+` long at every pair, but `d-` satisfies `3 d- = CRT(1, -1)` and is an arbitrary residue -
the real machine's own record run uses double-strikes at distances 2, 5, 7 at the pair (5, 7). There
is no pair configuration inside a violating stretch that compatible separations forbid, so the
smallest lemma about compatible pairs that the brief asked for does not exist.

The alternative the brief named, the tail gears' tooth distance, is refuted with a verified
counterexample: an m19 member whose every tooth distance exceeds the real machine's minimum
violates the budget inequality by 4.

**For the tree.** 2f.i is DEAD. What survives it: the two exception-free facts of section 6 (the
anchor is what makes a coherent separation available at all - the machine's gears exclude
themselves as denominators, so `1/3` is available only because 3 is anchor and not gear); and the
methodological finding of 5.2, that no statistic of the separations can be established as
protective on this family, because every candidate's protective region holds a few hundred members
against an expected violator count below one. Face C's opening - "the specific teeth as arithmetic,
not as a symmetry" - is narrowed by one more candidate: the arithmetic is not the compatibility of
the separations.

## 8. Dead ends, with the refuting instance

- **Compatibility as the ingredient.** Refuting instances: m17 `(1,3,4,4,4)`, `v' = 4`, rational
  `8/1`, `F(M+19) = 40 > 38`; m17 `(2,3,3,3,3)`, `v' = 3`, rational `6/1`, `38 > 37`; m11
  `(1,1,5)`, `v' = 1`, rational `29/18`, `25 > 24`. All `I = 0`, all direct-sieve verified.
- **Compatibility in all but one pair-group.** Refuting instance: 23 -> 29 teeth
  `(2,2,1,2,8,7,5)`, `v' = 5`, odd gear 19, `Q*_4 = 65 > 63`.
- **The coherent diagonal `d+` as the forbidden configuration.** Refuting instance: the real
  machine's own m23 record run, double-struck at distances 2, 5, 7 at the pair (5,7) - the same
  configuration the violating stretch uses, supplied by `d-`.
- **The tail gears' tooth distance.** Refuting instance: m19 `(1,2,2,2,6,3)`, `v' = 6`, every
  `sep_q/q >= 0.2941 > 0.2857`, `F(M+23) = 52 > 48`.
- **Resonance (`a = +-2 v_q` at every gear) as a rule.** Describes all three compatible violators
  and the m17 cell (2 of 7, 28.6%); refuted at m19, where the same cells hold 208 members and 0
  violators.

## 9. Files

- `research/anchor235/r46/cp_compat.py` - the definitions, the gate, the violator classification.
- `research/anchor235/r46/cp_detail.py` - the compatible violators in full.
- `research/anchor235/r46/cp_verify.py` - direct-sieve re-verification (reuses `chain_family_r32`).
- `research/anchor235/r46/cp_strat.py` - the full-family stratification by every statistic.
- `research/anchor235/r46/cp_sweep.py` - the 23 -> 29 sweeps (`coh`, `cohfull`, `oneoff`, `ctrl`).
- `research/anchor235/r46/cp_mech.py` - the violating chain and its pair overlaps.
- Outputs (untracked): `research/anchor235/r46/results/` - `gate.log`, `viol.log`, `detail.log`,
  `verify.log`, `mech.log`, `cp_strat_m17.log`, `cp_strat_m19.log`, `sweep_*_m23.json`,
  `strat_m*_B30.json`, `viol_compat.json`, `coh_members_m*.json`.
- Compute: 3 processes, largest array 37M bool (m23); the two 2,627-member m23 sweeps 434 s and
  347 s, the m19 stratification 4 min, everything else under a minute.

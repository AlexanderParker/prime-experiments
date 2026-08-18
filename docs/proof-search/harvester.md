# Harvester workstream - side theorems and adjacent conjectures

Round 1 (2026-08-18). Mission: statements weaker or adjacent to twin primes where this
session's machinery yields actual results. Everything below is priced honestly;
"not reachable" always means "not reachable with currently published methods" - an
imported corpus limit, stated as such, distinct from any event in the machine itself.

## 1. Survey of candidates

### C1. Legendre / Oppermann / Brocard band statements
Statement class: a prime (Legendre: in (n^2, (n+1)^2); Oppermann: both halves; Brocard:
>= 4 primes between consecutive prime squares) in every band. This is the team's own T1,
the terminus of the layer-band descent.
- Machinery that applies: margin census (M(t) law, li-model to 0.1%), band tiling,
  exact censuses to 6.67e9 slots. All DIAGNOSTIC: the margin is gear-blind (mechanic
  round 3) and the bands are invisible to it at 1e-4.
- Honest distance: the published localisation exponent stops at 0.525 (Alweiss-Luo, anchored
  at Baker-Harman-Pintz); needed 0.5. Not implied by RH. IMPORTED CORPUS LIMIT - a fact about
  existing methods, not about the machine; the gears frame is a sieve reparametrisation and
  inherits it unchanged. The machine-side event underneath (thinnest bands sit exactly at twin
  endpoints) remains uninterrogated as a mechanism.
- Pricing: nothing reachable beyond known WITH PUBLISHED METHODS. Value enormous,
  reachability ~0 by current technology.

### C2. Polignac for a fixed even gap 2d - the conjecture itself
- Machinery: everything transfers (see C3); research/general_gap.py already derives the
  class count prod(q - r_q), r_q = 1 iff q | d - the Hardy-Littlewood factor from the
  blocking rule alone.
- Honest distance: the parity barrier per fixed d - an imported corpus limit (about sieve
  methods as published, not the machine). Maynard-Tao/Polymath give SOME d <= 246 infinitely
  often, never a chosen d. Same imported limit as twins for every single d.
- Pricing: conjecture not reachable for any fixed d by published methods. (The machinery does
  not even prefer small d: the blocked-residue structure is isomorphic up to the q | d
  collapse.)

### C3. The per-gap reduction as a standalone kernel-checked theorem  <- TOP PICK (with C4)
Statement: for EVERY d, {p : p, p+2d both prime} is infinite IFF every scale has a window
(y, y^2] containing a gap-2d survivor of the gears <= y. Plus the transfer law: an odd
gear blocks both members of a gap-2d slot iff q | d (slot-cap generalised).
- Machinery: BlockedSlots.lean's twin proof is gap-agnostic in all but bookkeeping;
  Horizon's per-member argument never sees the gap. general_gap.py has the numerics.
- Honest distance: DAYS. First bite executed this round (proofs/Polignac.lean, below).
- Value: moderate and real. Ziller-Morack (arXiv:1706.00317, Thm 4.1) prove their
  paired-Jacobsthal bound sufficient for Goldbach + all even differences at once; per-
  difference EQUIVALENCES, machine-checked, are new (the review already established the
  d=1 iff is sharper than ZM). It also makes the session's whole corpus formally
  applicable to every even gap: the horizon, layer, supply, census files compose with
  SurvivorGap d unchanged. Harvestable as a formalisation note.

### C4. Goldbach via the paired-Jacobsthal frame
Two very different objects here:
- The CONJECTURE (or ZM Conjecture 6, the h_2 bound): Reduction-A class or harder (h_2 is
  a max over all differences). Not reachable by published methods.
- The windowed REDUCTION: "if some n in (sqrt N, N - sqrt N) has n and N - n free of prime
  factors <= sqrt N, then N is a sum of two primes", with the converse exact on the
  representations with both parts above sqrt N. Same horizon argument, reachable NOW,
  kernel-checked this round in the same file. Value: modest, citable; completes the
  ZM-frame trio (twins / Polignac-d / Goldbach) in Lean.

### C5. Prime quadruplets and k-tuples
- Machinery: found live as width-2 umbrellas (six quadruplets of the 47-window, 180504 at
  y=100003, share ~1.7% steady); exact alignment-count laws already session-proven:
  p(x) = prod(q - 2 + 2x), positions with k consecutive exposed = prod(q - 2k) with exact
  validity condition q >= 6(k-1).
- Honest distance: the counting laws are provable elementary CRT facts (and effectively
  proven in-corpus); their external novelty is low (standard sieve-support counting).
  Infinitude of quadruplets is strictly beyond twins. The k-tuple version of C3's
  reduction is a mechanical extension (pattern = finite offset set) if ever wanted.
- Pricing: no standalone publishable theorem beyond a k-tuple version of C3. Low value.

### C6. The overcount census theorem (Lateral rounds 2-3)
Statement: machine overcount = SAME + PAIRSPLIT - CORR, with SAME = squarefree-product
floor counting, PAIRSPLIT = the gap-graded split-class law, verified exactly at three
scales; plus the closed-form split representative x = (q'(b0 + iq) - 1)/6.
- Honest distance: fully provable, elementary; Lateral already calls the census identity
  "mechanisable". A finite-identity Lean target of the same genre as Supply.lean.
- Value: mostly internal (upgrades D(t) in the X-consistency equation to closed form via
  CORR). As standalone publication: a remark-level identity. Medium-low.

### C7. The g=2 pinning corollary as a stated theorem (Lateral round 3)
Statement: among all prime pairs (q, q+g) below y, ONLY g = 2 has m0 = 0, i.e. only twin
pairs have their split double-kill class pinned at depth u' <= (y+1)/6 in every window at
every scale; all other gaps enter at depth ~P/(6g) conditionally on mod-6 alignment.
- Honest distance: provable now by the CRT computation already verified on all 2850 pairs
  to 400. The quantified self-reference ("twins below y are the unique unconditionally
  guaranteed line item of the level-y^2 doubles ledger") is the session's most
  distinctive exact structural fact.
- Value: small-moderate; the natural centrepiece lemma if the team ever writes the
  ledger paper. Reachable; second bite candidate.

### C8. The constant-2 fragile law (Mechanic rounds 1-2)
Statement (measured): fragile * pi_win / (twins * W1) -> 2, exact to 0.43% at y=50021,
Poisson-clean per gear band with the size-corrected form.
- Honest distance: this is a Hardy-Littlewood-class asymptotic (its ingredients are
  twin-HL and semiprime counting); unconditional proof sits beyond currently published
  technology - an imported corpus limit, not a fact about the law itself, which the data
  match to 0.43%. A CONDITIONAL derivation (under HL for pairs) looks writable and would
  explain the constant 2 as the two ways a lone-composite member pairs with a prime; that
  is an exercise in heuristic bookkeeping, not a theorem of independent standing.
- Pricing: not harvestable unconditionally by published methods. Low.

### C9. Universal double-onset bound (Constructor round 2)
L0(y) <= 27129 for every y via Montgomery-Vaughan. Already proved in-corpus; it is a
Brun-Titchmarsh corollary, known-class. Diagnostic value only. Low.

### C10. The F(2,y) table as data (review section 7a)
The difference-2 paired-Jacobsthal values are genuinely new data (ZM compute none);
F(2,53) >= 420 stands unfinished. Finishing it and publishing the table (OEIS + note) is
reachable pure compute with small permanent value. Compute-bound (Rust search, tens of
minutes per increment); not this round's bite.

## 2. Ranking (reachability x value)

| rank | candidate | reachability | value | product |
|------|-----------|--------------|-------|---------|
| 1 | C3 per-gap reduction iff + slot-cap transfer (Lean) | high (done this round) | moderate | HIGH |
| 2 | C4 Goldbach window reduction (Lean) | high (done this round) | modest | HIGH |
| 3 | C7 g=2 pinning theorem | high | small-moderate | MED |
| 4 | C6 overcount census identity | high | low-moderate | MED |
| 5 | C10 F(2,y) data / OEIS | high (compute) | small | LOW-MED |
| 6 | C8 fragile law (conditional only) | medium | low | LOW |
| 7 | C5 k-tuple counting laws | high | very low | LOW |
| 8 | C9 onset bound | done | low | LOW |
| 9 | C1 band statements | ~0 | huge | ~0 |
| 10 | C2 fixed-gap Polignac | ~0 | huge | ~0 |

Top pick: C3 + C4 together, one Lean file - they share the horizon lemma and together
turn the kernel ledger from a twin-specific artifact into the general ZM frame.

## 3. First bite: proofs/Polignac.lean (executed)

New file, registered in lakefile.toml (targets now include Polignac). Contents:

- `prime_of_no_factor_le_sqrt`: sqrt-graded horizon lemma (per-member, gap-blind).
- `SurvivorGap d y m`: the gap-2d survivor predicate; `survivorGap_one_iff`: d = 1 is
  definitionally `BlockedSlots.Survivor`.
- `slot_cap_gap`: odd prime blocks both members of a gap-2d slot => q | d; corollary
  `slot_cap_twin` recovers Layer.slot_cap's content. This is the exact transfer
  condition for the whole corpus: every law whose proof uses slot-cap holds verbatim
  for gap 2d at the gears coprime to d, and the q | d gears collapse to one residue
  (the HL factor, mechanically).
- `survivorGap_iff_pair`: windowed equivalence survivor <=> prime pair at gap 2d.
- `gapPairs_infinite_iff_survivor_in_window (d)`: THE per-gap iff - Polignac for 2d
  is equivalent to the windowed survivor statement, both directions, every d (d = 0
  degenerates gracefully to infinitude of primes).
- `goldbach_of_survivor` / `goldbach_rep_of_survivor` / `survivor_of_goldbach_rep`:
  the Goldbach window reduction with its exact converse on central representations.

Verification discipline: all three statement families checked computationally first
(research/polignac_transfer_check.py: windowed iff for d in {0,1,2,3,5,6}, y in
{13,23,47}, zero fails; Goldbach frame exact for all even N < 2000; slot cap exact for
d < 20, q < 100). Lean build status recorded below.

BUILD STATUS: BUILDS CLEAN. `lake build Polignac` succeeds (one deprecation warning on
push_neg, same as the existing files), zero sorry. Axiom audit (lake env lean, #print
axioms on all nine theorems): standard axioms only - [propext, Classical.choice,
Quot.sound]; `survivorGap_one_iff` needs only [propext] and `survivor_of_goldbach_rep`
only [propext, Quot.sound]. Registered in lakefile.toml (defaultTargets + lean_lib).
One rename during build: this mathlib has `Nat.dvd_sub` where older versions had
`Nat.dvd_sub'`. The kernel ledger is now 6 files.

## 4. What this buys the team

- Every kernel-checked file now has a stated generalisation path: Horizon and Supply are
  already gap-blind (per-member arguments); Layer's slot_cap is the d = 1 case of
  slot_cap_gap; the one genuinely twin-specific object in the corpus is the phase vector
  +-u' (self-blocking at u' = round(q/6)) and its g=2 pinning (C7).
- Polignac vocabulary for the X-consistency programme: Condition X for gap 2d has the
  same zero-slack census with doubles counted by the d-pattern's split classes; the
  "unique guaranteed supply line" story (C7) is a statement ABOUT d = 2 from inside the
  general frame - the first structural fact that distinguishes twins from other gaps.
- A publishable unit exists if the manager wants one: "Machine-checked reductions of
  Polignac-type and Goldbach-type statements to paired-Jacobsthal window bounds"
  (Polignac.lean + BlockedSlots.lean + the F(2,y) data of C10). Modest but real; no
  overselling - it contains no progress on any conjecture, it is the frame made formal.

## 5. Next bites (in order)

1. C7: state and prove the g=2 pinning theorem (paper-form proof from the split law's
   closed form; optional Lean of the m0 = 0 iff g = 2 step, which is one mod-g inverse).
   [EXECUTED round 2 - see section 6.]
2. C6: CORR formula-ization (also requested by Constructor round 4) - harvest and
   flagship coincide there.
3. C10: restart the F(2,53) search if compute budget allows; package table for OEIS.

## 6. Round 2 (coordinator-approved bite): the g=2 pinning theorem, kernel-checked

Target (the g=2 slice of Lateral's split-gap law): only twin pairs have their split
double-kill class pinned unconditionally at the bottom of every window; every other gap's
class sits at alignment-conditional depth. Formalised in proofs/Polignac.lean (same file,
new section "The g = 2 pinning"), concrete twin-pair case per the coordinator's steer:

- `twin_mod_six`: p, p+2 prime, p > 3 => p = 5 mod 6 (slot coordinate exact).
- `twin_pin`: the pair IS slot u = (p+1)/6: 6u-1 = p, 6u+1 = p+2, p | left member,
  p+2 | right member - the split representative in closed form, existence trivialised
  by the self-block identity (the pin is the pair).
- `twin_pin_le`: u <= (y+1)/6 for EVERY y >= p - bottom band of every window, every
  scale, unconditionally. This is the formal statement of "twins below y are the
  guaranteed line item of the level-y^2 doubles ledger" (location half).
- `twin_split_class_iff`: slot k is split-killed by {p, p+2} (p left, p+2 right)
  IFF k = u mod p(p+2) - the full CRT class as an iff (g=2 case of the roots-of-unity
  law). Proof: coprimality to 6 cancels the slot map; distinct primes lift by CRT;
  below-the-pin slots are impossible because p+2 would divide a positive number
  smaller than itself.
- `twin_mirror_slot`: the second split class at P - u, both divisibilities explicit.
- `twin_product_slot`: the same-member double at kp = u(p+1), where 6kp - 1 = p(p+2)
  exactly - the machine re-ingesting its own output, formal.
- `own_slot_pin_gap_two` (uniqueness): a prime pair (q, q+g), both odd, that
  split-kills the slot holding q itself forces g = 2. Only twins pin at their own
  slot; every other gap's split representative is strictly deeper (in the full law:
  ~P/(6g), mod-6-alignment-conditional - that quantitative half stays paper-side,
  research/split_gap_law.py).

Verification discipline: research/twin_pin_check.py ran first - all 81 twin pairs to
3000 (pin, class iff exhaustive over two periods to p = 150, mirror, product slot) and
the uniqueness scan over all prime pairs q < q' <= 400 (20 own-slot pins found, all
g = 2): zero fails.

BUILD STATUS ROUND 2: BUILDS CLEAN ("Build completed successfully", zero sorry; only
pre-existing push_neg deprecation warnings + one unused-binder lint). Axiom audit on all
16 Polignac theorems: standard axioms only; the seven twin-pin theorems need just
[propext, Quot.sound] except twin_split_class_iff ([propext, Classical.choice,
Quot.sound]).

Lean notes for the team: omega does NOT combine congruences across moduli (2, 3, 6) -
decompose to a single modulus with explicit witnesses; this mathlib needs
`import Mathlib.Data.Nat.ModEq` for the [MOD n] notation (BlockedSlots does not pull
it in); `Nat.dvd_sub` here is the old `Nat.dvd_sub'`.

## 7. Round 3 (coordinator-approved): the SAME-side census, kernel-checked

First layer of the master supply formula formalised - the SAME-side pair census as one
CRT class plus its floor count, with the composite root law's "exactly once if it fits"
as a windowed corollary with explicit hypotheses. New section in proofs/Polignac.lean
(now also imports Census - first file composing with the formalist's census):

- `six_mul_class`: slot-map inversion - for any m coprime to 6 and target residue c,
  {k : 6k = c mod m} is exactly one class mod m (existence via
  Nat.exists_mul_mod_eq_of_coprime, uniqueness via cancel_left_of_coprime).
- `left_dvd_iff` / `right_dvd_iff`: member divisibility = residue condition
  (6k = 1 for the left member, 6k = m-1 for the right).
- `card_class_Ico` (THE FLOOR COUNT): #{k in [1,t] : k = a mod m} = (t + m - a)/m
  for 1 <= a <= m - proved by induction with Nat.succ_div_of_dvd/not_dvd; this is
  the count primitive every floor-arithmetic term of SAME/PAIRSPLIT reduces to.
- `same_left_census` / `same_right_census`: for distinct primes q, r >= 5, the slots
  whose left (resp. right) member both gears divide are ONE CRT class mod qr, count
  (t + qr - a)/qr over the first t slots - the SAME-side pair term, exact.
- `same_census_once` (COMPOSITE ROOT LAW, windowed): a <= t < a + P => count exactly 1.
  "Exactly once if it fits", with the fit hypotheses explicit.
- `same_left_own_value`: when qr = 5 mod 6 the class representative IS the slot
  holding qr itself ((qr+1)/6, member qr) - "acts at its own value" explicit.
- `class_rep_unique`, `not_dvd_six`: small reusable pieces.

Second bite (coordinator's): `twin_pin_self_block` - the pin slot u of a twin pair
(p, p+2) satisfies Census.slotComps u = 0 (a REAL twin slot, both members prime) AND
is never a BlockedSlots.Survivor of any machine with divisor bound >= p: the machine
is blind to its own pair. This is the formal reason the U-pin list is invisible to n2.

Verification discipline: research/same_census_check.py ran first - 105 prime pairs
(5 <= q < r < 60): class membership iff exhaustive over two periods (left + right),
floor count at 11 t-values per pair, window "exactly once", own-value reps: zero fails.

BUILD STATUS ROUND 3: BUILDS CLEAN - whole ledger green: `lake build` all 8 targets
(incl. Bridge + Gear), "Build completed successfully" (988 jobs), zero sorry. Axiom
audit on all 28 Polignac theorems: standard axioms only ([propext, Classical.choice,
Quot.sound]; several need only [propext, Quot.sound]).

Additional Lean notes for the team (this mathlib): `Finset.card_insert_of_not_mem` is
now `Finset.card_insert_of_notMem` (not_mem -> notMem rename);
`Ico_succ_right_eq_insert_Ico` lives in namespace `Nat`, not `Finset`; beware
`rwa [show m = ... ] at h` when m is also the ModEq modulus - the rewrite hits the
modulus occurrence too; rewrite in the goal with the equation oriented the other way.
Count primitive proof pattern: induction + Nat.succ_div_of_dvd/not_dvd avoids all
division-by-variable omega limitations.

## 8. Round 4 (coordinator-approved): the PAIRSPLIT class - the master formula's
formal core complete

The split (cross-member) layer, closing the SAME + PAIRSPLIT pair. New theorems in
proofs/Polignac.lean (built clean on first compile):

- `split_class`: for distinct primes q, r >= 5, the slots where q strikes the LEFT
  member and r the RIGHT (q | 6k-1, r | 6k+1) are exactly ONE CRT class mod qr, with
  the floor count (t + qr - a)/qr over the first t slots. Construction: the joint
  target residue c = CRT(1 mod q, r-1 mod r) via Nat.chineseRemainder, funneled
  through six_mul_class at modulus qr; the and-to-product step is
  Nat.modEq_and_modEq_iff_modEq_mul; modulus descent via Nat.ModEq.of_dvd. The
  mirror class (r left, q right) is the same theorem with roles swapped.
- `split_rep_twin_eq_pin` (g=2 LOOP-CLOSER): for a twin pair (p, p+2), any
  representative below the modulus of the split class equals the pin u = (p+1)/6 -
  the PAIRSPLIT representative of a twin pair IS its own slot. Together with
  twin_pin_le this is the formal "twins below y are the unique unconditionally
  guaranteed line item of the doubles ledger"; the two Polignac sections
  (pinning, PAIRSPLIT) now meet in one statement.
- `twin_split_count`: the twin pair's split count over the first t slots in closed
  form anchored at the pin - equal to 1 exactly on u <= t < u + p(p+2).

With round 3's SAME census, both structural layers of Lateral's master supply
formula (overcount = SAME + PAIRSPLIT - CORR) now have their class-and-count core
kernel-checked; what remains formal-side is the signed multi-gear combination
itself (CORR, >= 3-gear products) - bookkeeping over the same two primitives
(six_mul_class + card_class_Ico), scoped as future work.

Verification discipline: research/pairsplit_check.py ran first - 210 ordered prime
pairs (5 <= q, r < 60, both orientations): split-class membership iff exhaustive
over two periods, floor count on split reps, mirror role-swap, and the g=2
loop-closer (split rep == pin on all 5 twin pairs in range): zero fails. Also
cross-consistent with Lateral's closed form (split_gap_law.py: g=2 has m0 = 0,
b0 = 1, x = u').

BUILD STATUS ROUND 4: BUILDS CLEAN on first compile; whole ledger green - `lake build`
all 8 targets, "Build completed successfully" (988 jobs), zero sorry. Axiom audit:
split_class and twin_split_count on [propext, Classical.choice, Quot.sound];
split_rep_twin_eq_pin needs only [propext, Quot.sound]. Polignac.lean now holds 31
theorems across four sections: ZM-frame reductions, g=2 pinning, SAME census +
self-block, PAIRSPLIT + loop-closer.

## 9. Round 5 (coordinator-approved): the CORR triple - the signed layer lands

CHOICE MADE: CORR triple over F(2,53). Reason: the triple reduces entirely to the
round-3/4 primitives (six_mul_class + card_class_Ico + chineseRemainder) - a clean,
bounded Lean bite - while the F(2,53) search is open-ended compute (tens of minutes
per increment, uncertain termination inside a round; the expensive step is the final
uncoverable proof). F(2,53) stays shelved as the rank-5 data item.

New theorems in proofs/Polignac.lean (built clean on first compile):

- `twoSided_class` (THE GENERAL BOTH-SIDED TERM): for coprime moduli mL, mR > 1,
  both coprime to 6, the slots with mL | left member and mR | right member are ONE
  CRT class mod mL*mR with floor count (t + M - a)/M. Subsumes split_class (both
  prime) and yields EVERY both-sided term of the master formula in one statement -
  the moduli need only be coprime-to-6 coprime pairs, which all squarefree gear
  products are.
- `corr_triple_class`: the first genuinely new CORR case - distinct primes
  q, r, s >= 5, the triple (qr | left, s | right) is one class mod qrs, count
  closed-form. Ten lines: pure instantiation of twoSided_class. Other role splits
  (q | left, rs | right; etc.) are further instantiations.
- `corr_triple_signed` (THE SIGN, subtraction-free): distinct slots hit by either
  of two split classes sharing right gear s, PLUS the triple class, EQUAL the two
  split incidence counts. Only hypothesis: Coprime q r. The inclusion-exclusion
  step formal: the triple class is exactly what the signed sum removes when
  incidences become distinct slots. (Uses Finset.card_union_add_card_inter; the
  overlap-equals-triple step is the coprime divisibility glue.)
- `six_coprime_prime`: helper (prime >= 5 coprime to 6).

PAPER-SIDE: shape of the general CORR term and what remains for full CORR.
Every term of Lateral's master formula is N(s_L, s_R; t) for coprime squarefree
gear products (s_L, s_R), total >= 2 gears, sign (-1)^{#gears}; twoSided_class
already gives every such N as one class + floor count (s_R = 1 side is the SAME
census, done in round 3). What is NOT yet formal is the ASSEMBLY: (i) the general
union-to-alternating-sum conversion (n-ary inclusion-exclusion over the split
incidence classes - mathlib has Finset.inclusion_exclusion machinery to build on;
my corr_triple_signed is the n = 2 case), and (ii) the statement that the assembled
signed sum equals the census overcount (Lateral verified it exact at every prefix
at two scales). Both are combinatorial bookkeeping over the two proven primitives -
no new number theory anywhere in the remaining gap. Estimated as one further
formalist-scale round if the team wants full CORR; the per-term core is complete
as of this round.

Verification discipline: research/corr_triple_check.py ran first - 20 triples from
{5..19}, all 3 role splits each (60 two-sided cases): class membership exhaustive
over one period, floor counts, and the signed identity |A or B| + |triple| =
|A| + |B| with overlap == triple at 5 t-values per triple: zero fails.

BUILD STATUS ROUND 5: BUILDS CLEAN on first compile; whole ledger green - `lake build`
"Build completed successfully" (990 jobs), zero sorry. Axiom audit: all four new
theorems on [propext, Classical.choice, Quot.sound]. Polignac.lean = 35 theorems,
five sections: ZM-frame reductions, g=2 pinning, SAME census + self-block,
PAIRSPLIT + loop-closer, CORR two-sided/triple/signed.

## 10. Round 6 (coordinator-approved): the assembly - inclusion-exclusion over
incidence classes

The step from per-term core to assembled formula. Scope taken: the full n = 3
assembly proven GENERALLY (not just for {5,7,11} on a finite range - the concrete
window milestone is subsumed as an instance), plus the two bridges that convert the
assembled terms into CRT-class counts. New theorems in proofs/Polignac.lean (built
clean on first compile):

- `three_sets_ie` (n = 3 INCLUSION-EXCLUSION, subtraction-free, ANY finsets):
  |A u B u C| + |A^B| + |A^C| + |B^C| = |A| + |B| + |C| + |A^B^C|. Proof: three
  instances of card_union_add_card_inter + distributivity + omega.
- `three_preds_ie`: the filter/predicate form over one slot range.
- `three_gear_assembly` ("ASSEMBLED SUM = SIEVE OVERCOUNT"): three_preds_ie at the
  mark sets M_q = {k : q | 6k-1 or q | 6k+1} - distinct marked slots + the three
  pairwise terms = per-gear marks + the triple. Set-level, NO primality hypotheses;
  overcount := marks - distinct is a rearrangement. Extends corr_triple_signed
  (n = 2) to the full 3-gear window.
- `card_marks_eq` (PER-GEAR BRIDGE): |M_q| = |left class| + |right class| -
  disjoint by slot cap (mark_side_unique, 3 lines from slot_cap_twin).
- `card_pair_inter_eq` (PAIR BRIDGE): |M_q ^ M_r| = |LL| + |LR| + |RL| + |RR| -
  the four disjoint side classes (SAME-left, split, mirror split, SAME-right),
  each ONE CRT class with a floor count via six_mul_class / twoSided_class. This
  is where the set-level assembly meets the class-and-count layer.
- `card_filter_or_of_excl`: reusable primitive (filter of exclusive disjunction
  splits the count).

Remaining for FULL closed-form CORR (paper-side, priced): (i) the triple
intersection's 8-way side decomposition - mechanically identical to the pair
bridge (2^3 cases instead of 2^2, same mark_side_unique discharges); (ii) n > 3
gears - either iterated three_sets_ie or mathlib's inclusion-exclusion machinery.
Verified numerically THIS round including the full pipeline: overcount =
sum(12 pair classes) - sum(8 triple classes) with EVERY term equal to its floor
formula (research/assembly_check.py: 4 gear triples x 6 window lengths, zero
fails). No new number theory anywhere in the remaining gap.

F(2,53) watch (coordinator's ask): research/data/maxgap53.log exists with header
only ("y = 53, divisors [3..53]") at round end - the manager's detached run has
produced no increments yet; nothing to fold in. Needs <= 486 for the tolerance
constant; last known bound F(2,53) >= 420.

BUILD STATUS ROUND 6: BUILDS CLEAN on first compile; whole ledger green - `lake build`
"Build completed successfully" (992 jobs), zero sorry. Axiom audit: all seven new
theorems on [propext, Classical.choice, Quot.sound]. Polignac.lean = 42 theorems,
six sections (+ assembly).

## 11. Round 7 (coordinator-approved): the assembly line CLOSED - triple bridge +
the 26-term master theorem

(1) `card_triple_inter_eq` (TRIPLE BRIDGE): |M_q ^ M_r ^ M_s| = the eight disjoint
side classes LLL..RRR, each ONE CRT class with its floor count. Identical mechanics
to the pair bridge: 8-way predicate flatten (16 rintro cases) + 7 exclusivity peels,
every clash discharged by mark_side_unique. Built clean on first compile.

(2) `three_gear_master` (END-TO-END, 26 filter-card terms, subtraction-free):

    distinct + 12 pair side classes = 6 single side classes + 8 triple side classes

over the first t slots, for any distinct odd primes q, r, s. Every term beyond
"distinct" is one CRT class whose count is closed-form floor arithmetic
(six_mul_class / twoSided_class + card_class_Ico). Overcount = marks - distinct
rearranges it to overcount = pairs - triples: this is the formal statement of what
assembly_check.py verified numerically with zero fails. Proof: three_gear_assembly
+ the three bridges, rewritten term-by-term, omega. With this, THE ASSEMBLY LINE
FOR 3 GEARS IS CLOSED formally end to end; n > 3 was assessed and not forced
(needs either iterated three_sets_ie with 2^n-way flattens - mechanical but
voluminous - or mathlib's signed inclusion-exclusion over ℤ; nothing conceptually
new, deferred until the team needs it).

Verification discipline: research/master3_check.py ran first (5 gear triples x 5
window lengths to t = 5005: triple 8-way bridge + the 26-term identity: zero fails);
assembly_check.py had already verified the floor forms of every class term.

(3) F(2,53) WATCH + PRUNING ASSESSMENT (analysis only, per brief):
- State: TWO maxgap.exe processes are running (tasklist confirms, ~14 MB and
  ~16 MB working sets); research/data/maxgap53.log still contains ONLY the header
  line at round end. Since the header flushed, line buffering works - the likely
  reading is that no L-increment has completed since launch: at L >= 420 each
  increment cost was already "tens of minutes" at L = 416 (review 7a) and grows
  with L; hours per increment is plausible now. Manager should double-check the
  resume/state file mtime and why there are two processes (parallel L-split or
  accidental duplicate?).
- Pruned restart economics: the endpoint law (both endpoints in the 15-residue
  exposed set mod 35; left endpoint in A(G), as small as 3 residues) gives a 2-5x
  per-increment cut, INCLUDING the final expensive uncoverable certificate. The
  corpus search supports resume (review 7a resumed from L = 356), so a pruned
  rebuild restarts at the current verified L with zero lost work. Break-even is
  roughly ONE increment at current L; remaining distance is plausibly 20+
  increments (quadratic-law prediction ~441 vs standing >= 420; geometric law
  predicts more).
- ASSESSMENT: a pruned restart beats continuing unpruned unless the run is within
  ~1 increment of termination, for which there is no evidence. Recommend: verify
  the two processes' state, implement the endpoint-law filter in rust/maxgap,
  restart from the current resume state. (Not implemented this round, per brief.)

BUILD STATUS ROUND 7: Polignac BUILDS CLEAN on first compile (973 jobs), zero sorry;
axiom audit on both new theorems: [propext, Classical.choice, Quot.sound]. Ledger
note: plain `lake build` currently fails in Corridor.lean - ANOTHER workstream's
new file, mid-edit this round (3 errors, its own file); all nine other targets
(BlockedSlots Horizon Layer Supply Census Bridge Polignac Gear Placement) build
green together: "Build completed successfully" (988 jobs). Mirror of the round-2
situation in reverse; flagged for the manager, owner to fix. Polignac.lean = 44
theorems, six sections.

What this buys: the finite u'-pin list U in the round-4 master formula (n2 = B - U,
U confined to the bottom y/6 slots) now has its kernel formally characterised: existence
(twin_pin), location bound (twin_pin_le), exactness of the class (twin_split_class_iff),
and twin-uniqueness (own_slot_pin_gap_two). The remaining unformalised half of the
distinguishing fact is quantitative (depth ~P/(6g) for g > 2 and the alignment rate) -
priced as paper-side, not kernel-side, for now.

## 12. Round 8 (coordinator-authorized implementation): the pruned F(2,53) restart

DELIVERED: rust2/src/bin/maxgap_pruned.rs - the endpoint law in covering-search form,
verified identical to the original on six exact values, launched detached from the
current resume point.

DERIVATION (the law had to be re-derived soundly for the FREE-OFFSET covering search;
the machine-frame law of research/topgap_endpoint_law.py is about fixed offsets):
- MOD-3 ENDPOINT SKIP: at the maximal coverable run M, both bounding positions -1 and
  M are uncovered (else the run extends) and gear 3 is always used at the max (an
  unused gear 3 could cover position M); gear 3's single uncovered residue class must
  contain both -1 and M, so M = 2 (mod 3) and F = 0 (mod 3) UNCONDITIONALLY. All
  thirteen known exact values comply (33..309, all = 0 mod 3). Coverability is
  monotone in L, so every L != 0 (mod 3) below F is coverable WITHOUT SEARCH; the
  first uncoverable multiple of 3 is exactly F. Cuts 2/3 of all coverable increments.
- LEFT-TAUT OFFSET EXCLUSION (per-L EQUIVALENCE, not just max-valid): for every L,
  coverable(L) <=> coverable(L) with position -1 uncovered. Proof: the maximal run's
  witness cannot cover -1 (else an (M+1)-run is covered) and restricts to a left-taut
  witness of every prefix length. So every gear bars its two offsets covering -1
  (o = q-2, q-1): gear q never covers positions = -1 (mod q), collapsing the branch
  factor at every leftmost-uncovered position = -1 or -2 (mod q).
- INTERACTION FIXED: the original's mirror-canonical o5 halving maps left-taut to
  RIGHT-taut coverings under reflection - the two prunings are UNSOUND together.
  Removed the canonicalisation; left-tautness itself restricts o5 to {0,1,2} - the
  same root branch count, so nothing is lost.
- NOT USED (deliberately): the A(G) mod-35 right-endpoint refinement - it conditions
  on the gap length G and is only valid at the maximum, not per-L; using it in the
  incremental loop would be unsound.

VERIFICATION (identity before any long run, per discipline):
  y        11   13   17   19   23   29   37
  original 21   33   54   75  102  129  264
  pruned   21   33   54   75  102  129  264     <- EXACT MATCH, all = 0 mod 3
  Timing at y=37 (from L=250): pruned 1.12s vs original 1.74s; the mod-3 skip's full
  3x applies to the long coverable climb at y=53 (only 2 increments were skippable in
  the y=37 tail); left-taut cuts apply inside every search including the final
  uncoverable certificate.

LAUNCH: maxgap_pruned.exe 53 420, detached via Start-Process,
  PID 94812, log research/data/maxgap53_pruned.log (stdout; .err.log alongside).
  Resume point per ROUND-11 NEWS: unpruned log shows "run of 420 is coverable", so
  the pruned run STARTS at L=420 - its first increment re-verifies the fresh fact
  (the required consistency check), then proceeds 423, 426, ... (421, 422 skipped by
  law). The two unpruned processes (PIDs 32784, 89404) were NOT touched - manager
  retires them after this report.

## 13. Round 9: pruning theorems formalized; the literal cap transfers to Polignac d

### (1) The two pruning theorems' number-theory cores

VERIFIED FIRST, exhaustively over ALL offset tuples (research/lefttaut_check.py -
not the pruned search, so the check is independent of the code being justified):
y = 11/13/17, F = 21/33/54 (corpus match), all = 0 mod 3, and the left-taut
equivalence holds at EVERY L from 1 to F+2, zero mismatches. Plus the mod-3
core over all offsets and positions (literal_cap_gap_d.py T3, zero fails).

KERNEL-CHECKED (mine, proofs/Polignac.lean - no collision with Formalist's
Machine13; built clean on first compile, axioms [propext, Quot.sound] only):
- `AdjBlocked q o i`: the covering search's blocking relation (gear q at offset
  o blocks the ADJACENT pair {o, o+1} mod q) - the adjacent-frame counterpart of
  BlockedSlots.Blocked, now a definition the search's proofs can cite.
- `free_class_three`: gear 3's pair covers two of three classes, so an unblocked
  position sits in the single free class o+2 mod 3.
- `free_class_unique_three`: gear 3 cannot leave two incongruent positions
  uncovered.
- `endpoint_run_mod_three` (THE ENDPOINT LAW): if a run of M positions has both
  flanks unblocked by gear 3, then 3 | (M+1). Since F(2,y) = M+1 at the maximal
  run, this IS "F(2,y) = 0 mod 3" - the mod-3 skip's justification, and the
  reason all thirteen known values are divisible by 3.

HANDED TO FORMALIST (exact statement, in agents-shared; NOT taken myself because
it quantifies over coverings and wants the search formalized, which is their
Machine-file machinery, not a one-lemma job):
  LEFT-TAUT EQUIVALENCE. Fix gears Q and L >= 1. Write Cov(L) for "there is an
  offset assignment o : Q -> N with every position of [0, L) AdjBlocked by some
  gear". Then Cov(L) <=> there is such an assignment additionally leaving
  position -1 unblocked by every gear.
  (=>) Let M >= L be maximal with Cov(M) (finite: M < F). Its witness cannot
  block position -1, else the run [-1, M) of length M+1 is covered, contra
  maximality. Restricting that witness to [0, L) proves the taut form.
  (<=) Trivial. Verified exhaustively y <= 17, every L.
  Consequence used by the search: every gear may drop its two offsets q-2, q-1
  (those blocking -1), so gear q never blocks positions = -1 mod q.

### (2) HARVEST: the literal cap transfers to Polignac gap d (tool:
research/literal_cap_gap_d.py)

Question posed: does the SUMMARY's literal cap ("literal chains have at most 6
members, for every gear, forever" - a 48-class finite check) survive for
separation d != 2? ANSWER: YES for every d tested, with one honest exclusion.

Computed the analog of Constructor's table for d = 2, 4, 8, 10, 14, 16, 20, 28
(all d != 0 mod 6), over every prime q' in (d, 2000]:

    d    e=d/2   |E_d| mod 35   cap spectrum by class      max cap  invariance
    2      1        15        {2:24, 3:4, 4:14, 6:6}          6      OK 48 cls
    4      2        15        {2:24, 3:4, 4:14, 6:6}          6      OK 48 cls
    8      4        15        {2:24, 3:4, 4:14, 6:6}          6      OK 48 cls
   16      8        15        {2:24, 3:4, 4:14, 6:6}          6      OK 48 cls
   10      5        20        {4:24, 6:24}                    6      OK 48 cls
   20     10        20        {4:24, 6:24}                    6      OK 48 cls
   14      7        18        {2:24, 4:12, 6:12}              6      OK 48 cls
   28     14        18        {2:24, 4:12, 6:12}              6      OK 48 cls

d = 2 reproduces Constructor's published table exactly ({2:24, 3:4, 4:14, 6:6}),
which validates the re-implementation.

WHAT TRANSFERS VERBATIM (no d-specific input at all):
- the architecture: a literal chain is an interleaved two-phase walk with
  PERIOD 70 mod 35, so the cap is a function of q' mod 210 ONLY - the same
  48-invertible-class finite check, per d;
- class invariance itself: verified per d over ~300 primes each, zero
  mismatches, exactly as in the twin case;
- THE CEILING: max cap = 6 for every d tested. "Literal chains have at most 6
  members, for every gear, forever" appears to be separation-INDEPENDENT.

WHAT NEEDS THE d-SPECIFIC INPUT (two scalars, both closed-form):
- the exposed set E_d mod 35: gears 5, 7 block +-e*6^{-1}, and gear q's two
  blocked residues COLLAPSE TO ONE exactly when q | e. That collapse condition
  is my kernel-checked `Polignac.slot_cap_gap`, so the sizes are forced:
  |E_d| = prod over {5,7} of (q - r_q), r_q = 1 if q | e else 2 - giving
  15 (generic), 20 (5 | e), 18 (7 | e), 24 (35 | e). The measured column above
  matches this formula exactly. The Hardy-Littlewood factor and the exposed-set
  size are the same object.
- the walk step u'_d(q') = least positive representative of +-e*6^{-1} mod q'
  (twin case: round(q'/6)).
So the per-d fuel bound is a two-line specialisation, not a new theory.

WHAT DOES NOT TRANSFER AS STATED (the honest exclusion): d = 0 mod 6, i.e.
3 | e. There gear 3 keeps TWO free classes instead of one (the same free-class
count that drives my endpoint law above), the single-slot-frame collapse fails,
and the walk lives mod 105 with two subframes. Not computed here; flagged as
the one genuine d-specific gap. Note this is exactly the class containing the
densest Polignac gaps (d = 6, 12, 18...), so it is worth someone's round.

CONSEQUENCE FOR THE PROGRAMME: every separation d != 0 mod 6 gets its own fuel
bound with the same ceiling 6, so the per-d tolerance route generalizes - the
Wall-V pricing is not twin-specific. Combined with round 1's per-gap reduction
(gapPairs_infinite_iff_survivor_in_window) the whole tolerance apparatus is now
stated for every even gap.

### (3) F(2,53) WATCH - PRUNED RUN REPRODUCES 420, UNPRUNED PAIR RETIRED

*** The pruned run REPRODUCED "run of 420 is coverable" *** (log line 2), then
skipped 421 and 422 by the mod-3 law and is now searching L = 423. The manager
retired the unpruned pair during this round: tasklist now shows ONLY
maxgap_pruned.exe PID 94812 (13.5 MB). Unpruned log's last line before
retirement was "run of 421 is coverable"; the pruned run is ahead of it in
effective progress (421, 422 disposed of by theorem rather than by search).
Log: research/data/maxgap53_pruned.log. Next expected line: 423.

## 14. Round 10: the excluded case closed, and the word identity's transfer priced

Tools: research/literal_cap_mod105.py (chunk 1), research/word_identity_gap_d.py
(chunk 2). Both work in HALVED COORDINATES - position n, pair (2n+1, 2n+1+2e),
gear q blocks n = 0, -e mod q, e = d/2 - the universal frame in which gear 3 is
EXPLICIT. (The twin slot frame quotients gear 3 away only because 3 blocks two of
three classes when 3 does not divide e. That is precisely why d = 0 mod 6 needed
its own treatment.) The frame change is validated by reproducing Constructor's
twin cap table exactly.

### (1) THE EXCLUDED CASE d = 0 mod 6: cap finite, 48-class check survives,
### ceiling 6 breaks only when 15 | e

Definition used (frame-free): a literal chain is a maximal run of CONSECUTIVE
frame-admissible q'-kills (n = 0 or -e mod q', n admissible for gear 3) that are
all exposed to gears 5, 7. Computed exactly over a full period 105*q', doubled
for wrap, for every prime q' <= 1200 coprime to 105.

    gcd(e,105)   |E_d| mod 105   cap spectrum by class        max cap
        1             15         {2:24, 3:4, 4:14, 6:6}          6
        5             20         {4:24, 6:24}                    6
        7             18         {2:24, 4:12, 6:12}              6
        3             30         {4:36, 5:4, 6:8}                6      <- d = 0 mod 6
       21             36         {4:36, 6:12}                    6
       35             24         {6:48}                          6
       15             40         {6:8, 7:8, 8:24, 10:8}         10      <- ceiling breaks
      105             48         {12:48}                        12      <- ceiling breaks

ANSWERS to the three questions asked:
- IS THERE STILL A FINITE CAP for d = 0 mod 6? YES. For gcd(e,105) = 3 (i.e.
  d = 6, 12, 18, 24 - the densest Polignac gaps) the cap spectrum is
  {4:36, 5:4, 6:8} with MAX 6 - the same ceiling as twins, though the FLOOR
  rises from 2 to 4 and a cap of 5 appears (never present in the twin table).
- DOES 48-CLASS INVARIANCE SURVIVE? YES, as a mod-105 invariance: cap is a
  function of q' mod 105 only, zero mismatches for every d tested, and
  phi(105) = 48 - the same 48 classes. Unification worth recording: for ODD q',
  q' mod 210 is determined by q' mod 105, so Constructor's mod-210 statement and
  this mod-105 one are THE SAME CHECK. One law, one class count, all d.
- DOES |E_d| STILL TRACK THE HL FACTOR? YES, exactly, now including gear 3:
  |E_d| = prod over q in {3,5,7} of (q - r_q), r_q = 1 if q | e else 2. Every row
  of the table matches the prediction (column 2 vs the formula). The collapse
  condition r_q = 1 iff q | e is my kernel-checked `Polignac.slot_cap_gap`.
- BONUS (exhaustive, not sampled): the cap SPECTRUM depends only on gcd(e, 105) -
  verified by e = 45 reproducing e = 15 and e = 7 reproducing d = 14. Since 105
  has exactly 8 divisors and all 8 are computed above, THE TABLE IS COMPLETE OVER
  ALL EVEN d. So: for every even gap d, literal chains are capped by a 48-class
  check on q' mod 105; the cap is <= 6 for six of the eight gcd classes, 10 when
  gcd = 15, 12 when 105 | e; and 12 IS THE ABSOLUTE CEILING OVER ALL POLIGNAC
  GAPS. The fuel bound is universal over Polignac, and honest about where the
  constant grows: exactly where e absorbs the small gears (denser exposed set).

### (2) THE WORD IDENTITY: shape and firing transfer verbatim; ALTERNATION does not

Tested against EXACT F values (full-period computation, all q' CRT phases - the
phase loop IS Constructor's "the q' copies realize every residue shift"), 13
configurations: d = 2, 4, 6, 10, 12, 30 and the degenerate q' | e, machines
{3,5,7,11} up to {3,5,7,11,13,17}.

- W1 (merge/tier decomposition reproduces F(M+q') exactly): Y in 13/13.
- W2 (THE IDENTITY'S SHAPE, F(M+q') = max(F2(M), max over k>=2 tiers)):
  Y in 13/13, INCLUDING every d = 0 mod 6 case and both degenerate cases.
  Moreover tier_1 = F2(M) exactly in every row (33/48/75/30/48/22/30/45/27/18/35)
  - Constructor's "the 1-letter word always fires" verified for every d. The
  lower-bound mechanism rests on gcd(P_M, q') = 1, which contains NO d at all:
  it transfers verbatim, and so does the identity's shape.
- W3 (TOOTH ALTERNATION inside chains): Y in all 3 does-not-divide cases
  (d = 2, 4, 10, 30 rows); N in 3 of 5 tested 3 | e cases (d = 6 at q' = 17, 19;
  d = 12 at q' = 17). MECHANISM (diagnosed, not guessed): the frame letter
  SEQUENCE is strictly two-letter alternating when 3 does not divide e
  (e.g. twins q'=17: 18, 33, 18, 33, ...) but has FOUR letters per cycle with a
  SHORT letter when 3 | e (e.g. e=6, q'=17: 6, 11, 6, 28, ...). A short letter
  makes single-kill skips cheap, and an odd skip flips the tooth parity - so
  same-tooth adjacency occurs. Alternation is a twin-FRAME fact, not a
  separation-independent one.
- W4 (realized chain letters are sums of consecutive frame letters): holds; the
  two apparent failures in the table were WRAP-AROUND ARTIFACTS of my letter
  extractor (differences taken mod P across the period end), confirmed by direct
  diagnostic: twins q'=17 realized letters {18,33} = the frame letters exactly;
  e=6 q'=17 realized {6,11,23} with 23 = 11+6+6 a padded link. Recorded rather
  than hidden, per discipline.
- W5 (degenerate q' | e, gear q' has ONE tooth): the frame letter set collapses to
  the single value 3q' (39 at q'=13, 51 at q'=17), chains become plain arithmetic
  progressions, no k >= 2 tier is needed, and F(M+q') = F2(M) exactly in both
  cases. The identity survives trivially; the word grammar degenerates.

VERDICT for the programme. The tolerance route's growth law is parameterized by d
with TWO ingredients of different status: the identity's shape and its firing
mechanism are d-agnostic (gcd(P_M,q') = 1 only), so every Polignac gap gets the
same exact growth law; but the alternation ingredient - and therefore the
2-candidate-word grammar that makes the word list SHORT - is specific to
3 does not divide e. For d = 0 mod 6 the grammar has a richer letter alphabet
(3 letters, one of them short) and needs its own word list before the tolerance
route can be quoted there. That is the honest boundary: identity universal,
grammar not.

### (3) F(2,53) WATCH

Unchanged this round: PID 94812 alive (27 MB), log still at
"420 coverable / 421 skipped / 422 skipped", i.e. still inside the L = 423
search - the first genuinely new increment past the retired unpruned pair's
reach. Ledger green independently: `lake build` 998 jobs, zero sorries.

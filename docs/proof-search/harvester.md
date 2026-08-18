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

What this buys: the finite u'-pin list U in the round-4 master formula (n2 = B - U,
U confined to the bottom y/6 slots) now has its kernel formally characterised: existence
(twin_pin), location bound (twin_pin_le), exactness of the class (twin_split_class_iff),
and twin-uniqueness (own_slot_pin_gap_two). The remaining unformalised half of the
distinguishing fact is quantitative (depth ~P/(6g) for g > 2 and the alignment rate) -
priced as paper-side, not kernel-side, for now.

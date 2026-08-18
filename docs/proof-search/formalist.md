# Formalist workstream log

## Round 1 — Horizon theorem (2026-08-18)

### What was done

Formalised the horizon theorem as `proofs/Horizon.lean` (namespace `Horizon`,
standalone — imports only mathlib, nothing from `BlockedSlots`). Registered it
in `proofs/lakefile.toml` as a second `lean_lib` and added it to
`defaultTargets`; lake does not glob the directory, each root file needs its
own `[[lean_lib]]` entry. Added the three theorems to `proofs/AxiomCheck.lean`
(run with `~/.elan/bin/lake.exe env lean AxiomCheck.lean` from `proofs/`).

### Final theorem statements

```lean
theorem exists_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : ∃ p, p.Prime ∧ p < y ∧ p ∣ m

theorem prime_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m) : m.Prime

theorem twin_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hwin : m + 2 < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m ∧ ¬ p ∣ (m + 2)) :
    m.Prime ∧ (m + 2).Prime
```

Note the strict bound `p < y` throughout: this is the interior form — gears
strictly below `y` decide the open window `(y, y*y)`, sharper than
`BlockedSlots.survivor_iff_twin` which uses `q ≤ y` and the closed window
`m + 2 ≤ y * y`.

### Proof route

`Nat.minFac m` is prime (`Nat.minFac_prime`, needs `m ≠ 1`, which follows from
`y < m < y*y` forcing `m ≥ 2`), divides `m` (`Nat.minFac_dvd`), and satisfies
`minFac m ^ 2 ≤ m` for composite `m` (`Nat.minFac_sq_le_self`). Then
`minFac² ≤ m < y²` gives `minFac < y` by contradiction with
`Nat.mul_le_mul` + `linarith` — no `Nat.sqrt` needed at all, cleaner than the
sketch. The pair corollary is two applications of the contrapositive form.
Adapted from `prime_of_not_blocked` in `BlockedSlots.lean:74-83`.

One iteration detail: `omega` cannot see through the nonlinear atom `y * y`,
so the window inequalities are discharged with `linarith` (which treats
`y * y` as an opaque atom) rather than `omega`. Avoided the deprecated
`push_neg` (mathlib now wants `push Not`) by using `Nat.lt_or_ge` /
`Nat.le_of_not_lt` instead.

### Build status

`~/.elan/bin/lake.exe build` from `proofs/`: **Build completed successfully
(968 jobs)**, zero `sorry`, zero warnings in `Horizon.lean`.
`BlockedSlots.lean` still builds clean (only its pre-existing `push_neg`
deprecation warnings).

### Axiom audit

```
'Horizon.exists_prime_factor_lt'      depends on axioms: [propext, Classical.choice, Quot.sound]
'Horizon.prime_of_no_prime_factor_lt' depends on axioms: [propext, Classical.choice, Quot.sound]
'Horizon.twin_of_no_prime_factor_lt'  depends on axioms: [propext, Classical.choice, Quot.sound]
```

Only the three standard axioms — nothing beyond ordinary mathematics.

### Proposed next target

The layer law's arithmetic core: one layer's novelty in stepping the horizon
from `y` to `y'` is `{y^2} ∪ {y*c : c prime, y < c, y*c < y'^2}` — i.e. a
composite that first becomes visible when the horizon passes `y` has least
prime factor exactly `y` (when `y` is prime). This is again a `minFac`
argument (`minFac m = y ↔ y ∣ m ∧ no smaller prime divides`), composes
directly with `exists_prime_factor_lt`, and needs no new machinery.
Second choice: the `h(2) ≥ d` product inequality.

## Round 2 — Layer law arithmetic core + slot-cap lemma (2026-08-18)

### What was done

New file `proofs/Layer.lean` (namespace `Layer`, mathlib-only imports),
registered as a third `[[lean_lib]]` in `proofs/lakefile.toml` and added to
`defaultTargets`. AxiomCheck.lean extended with the four new theorems.
Compiled first try, zero sorry, zero warnings.

### Final theorem statements

```lean
theorem slot_cap {q m : ℕ} (hq : 3 ≤ q) : ¬ (q ∣ m ∧ q ∣ m + 2)

theorem minFac_lt_or_eq {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (h1 : 1 < m) (hnp : ¬ m.Prime) (hm : m < y' * y') :
    m.minFac < y ∨ m.minFac = y

theorem eq_mul_prime_of_minFac_eq {y m : ℕ} (h1 : 1 < m)
    (hfac : m.minFac = y) (hlow : y * y < m) (hhigh : m < y * y * y) :
    ∃ c, c.Prime ∧ y < c ∧ m = y * c

theorem layer_novelty {y y' m : ℕ}
    (hnext : ∀ q, q.Prime → y < q → q < y' → False)
    (hthin : y' * y' ≤ y * y * y)
    (hnp : ¬ m.Prime) (hlow : y * y < m) (hhigh : m < y' * y') :
    (∃ p, p.Prime ∧ p < y ∧ p ∣ m) ∨ ∃ c, c.Prime ∧ y < c ∧ m = y * c
```

### Design decisions

- Took the STRONGEST form: `c` is proved prime, not merely `minFac c ≥ y`.
  Bertrand was avoided entirely by carrying the thin-layer bound
  `y'^2 ≤ y^3` as an explicit hypothesis (`hthin`). For consecutive primes
  it holds from y = 3 on; the caller discharges it, this file never needs
  Bertrand. Inside the proof the range collapses to `c < y*y`, and a
  composite `c` with all prime factors `≥ y` (via `Nat.minFac_le_of_dvd`
  applied to `minFac c ∣ m` against `minFac m = y`) would need
  `y*y ≤ minFac c ^ 2 ≤ c` — contradiction.
- `hnext` phrased exactly like `BlockedSlots.survivor_step`'s gap
  hypothesis (`∀ q, q.Prime → y < q → q < y' → False`), so the two compose
  without adapters. `y` prime is NOT assumed — it falls out of
  `minFac m = y` in the semiprime branch.
- `layer_novelty`'s open bounds `y*y < m < y'*y'` deliberately exclude the
  boundary novelty `{y^2}` itself, matching "novelty = {y^2} ∪ {y*c}".
- slot_cap stated for `3 ≤ q` (no primality needed): `q ∣ m ∧ q ∣ m+2`
  gives `q ∣ 2` by `Nat.dvd_sub`, five lines total.

### Build status

`lake build`: **Build completed successfully (970 jobs)**, zero sorry.
BlockedSlots.lean and Horizon.lean unchanged and still building clean.

### Axiom audit (extended AxiomCheck.lean)

```
'Layer.slot_cap'                 depends on axioms: [propext, Quot.sound]
'Layer.minFac_lt_or_eq'          depends on axioms: [propext, Classical.choice, Quot.sound]
'Layer.eq_mul_prime_of_minFac_eq' depends on axioms: [propext, Classical.choice, Quot.sound]
'Layer.layer_novelty'            depends on axioms: [propext, Classical.choice, Quot.sound]
```

Standard axioms only; slot_cap needs neither choice nor anything classical.

### Proposed next target

The zero-slack theorem's finite core from the Constructor ledger: with
slot_cap and lpf attribution both now kernel-checked, the overlap-free
supply identity sum_q R(q) = 2N - P over a window is a Finset.card
partition argument — the natural next Lean chunk, and it is what C2
(prefix pigeonhole) formalisation would sit on. Alternative: h(2) ≥ d's
product inequality.

## Round 3 — Supply identity as a Finset partition (2026-08-18)

### What was done

New file `proofs/Supply.lean` (namespace `Supply`), the first file that
IMPORTS the earlier ones — `import Horizon` discharges the fiber-membership
obligation, `import Layer` supplies slot_cap for the slot-level corollary.
Registered as fourth `[[lean_lib]]` in lakefile.toml, added to
defaultTargets, AxiomCheck.lean extended. Compiled first try, zero sorry,
zero warnings.

### Final theorem statements

```lean
theorem minFac_mem_gears {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : m.minFac ∈ (Finset.range y).filter Nat.Prime

theorem card_composites_eq_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    (S.filter fun m => ¬ m.Prime).card =
      ∑ p ∈ (Finset.range y).filter Nat.Prime,
        (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card

theorem card_eq_primes_add_sum_roots {y : ℕ} (S : Finset ℕ)
    (hS : ∀ m ∈ S, y < m ∧ m < y * y) :
    S.card = (S.filter Nat.Prime).card
      + ∑ p ∈ (Finset.range y).filter Nat.Prime,
          (S.filter fun m => ¬ m.Prime ∧ m.minFac = p).card

theorem roots_ne {m : ℕ} (h1 : 1 < m) (hodd : ¬ 2 ∣ m) :
    m.minFac ≠ (m + 2).minFac
```

### Proof route

- The partition is `Finset.card_eq_sum_card_fiberwise` (current mathlib
  signature takes `(↑s).MapsTo f t`) with `f = Nat.minFac`,
  `t = (range y).filter Nat.Prime`; the MapsTo obligation is exactly
  `minFac_mem_gears`, which is Horizon's `exists_prime_factor_lt` plus
  `Nat.minFac_le_of_dvd` (least factor ≤ the found factor < y). `m ≠ 1`
  falls out of `p ∣ m` for a prime p (`Nat.dvd_one`). The fibers are
  reshaped from nested filters to the `∧` form by `Finset.filter_filter`.
- Ledger form via `Finset.card_filter_add_card_filter_not`.
- `roots_ne`: a shared root would be a gear ≥ 3 (odd m rules out 2)
  dividing both m and m+2, killed by `Layer.slot_cap`. `3 ≤ minFac` comes
  from `lt_of_le_of_ne hp.two_le` — definitionally `2 < x` is `3 ≤ x`.
- The window hypothesis is per-member (`∀ m ∈ S, y < m ∧ m < y * y`), so S
  need not be an interval: it composes with any window Finset the
  constructor formalisation later chooses (e.g. the ±1 mod 6 members),
  and the `2N − P` reading is call-site arithmetic once S comes in pairs.

### Build status

`lake build`: **Build completed successfully (974 jobs)**, zero sorry.
BlockedSlots, Horizon, Layer unchanged, still building.

### Axiom audit (extended AxiomCheck.lean)

```
'Supply.minFac_mem_gears'            depends on axioms: [propext, Classical.choice, Quot.sound]
'Supply.card_composites_eq_sum_roots' depends on axioms: [propext, Classical.choice, Quot.sound]
'Supply.card_eq_primes_add_sum_roots' depends on axioms: [propext, Classical.choice, Quot.sound]
'Supply.roots_ne'                    depends on axioms: [propext, Classical.choice, Quot.sound]
```

### Proposed next target

The zero-slack direction: under Condition X (no twin in the window), every
slot has ≥ 1 composite member, so pairing the ledger form over slots gives
n1 + 2*n2 = C and n1 + n2 = N together with P = n1 — the census pinning
n1(t) = P(t), n2(t) = N(t) − P(t) as Finset statements over prefixes.
That is the formal substrate C2 (prefix pigeonhole) sits on. Alternative:
h(2) ≥ d's product inequality.

## Round 4 — Zero-slack census pinning (2026-08-18)

### What was done

New file `proofs/Census.lean` (namespace `Census`), the demand side of the
round-4 flagship (X-consistency equation), formalized. Sixth lakefile
target; AxiomCheck extended. One build iteration (a missing `smul_eq_mul`
under minimal imports — replaced by `Finset.sum_const_nat`, which is the
right ℕ-native lemma anyway); second build clean, zero sorry, zero warnings.

### Setup

Slot `k` carries the pair `(6k−1, 6k+1)` (`lo`/`hi`). Per-slot counters
`slotPrimes k`, `slotComps k` (0/1/2 via decidable `Nat.Prime` ifs — tied to
real primality, not abstract blocking, so Horizon/Supply compose directly).
Over an arbitrary `T : Finset ℕ` of slots: `primesIn` (= P), `compsIn`
(= C), `n0`/`n1`/`n2` = card of the filter `slotComps k = 0/1/2`, N = `T.card`.

### Final theorem statements (all over arbitrary T)

```lean
theorem census_partition : n0 T + n1 T + n2 T = T.card
theorem comps_eq         : compsIn T = n1 T + 2 * n2 T
theorem primes_add_comps : primesIn T + compsIn T = 2 * T.card
theorem primes_eq        : primesIn T = n1 T + 2 * n0 T
theorem n0_eq_zero_iff   : n0 T = 0 ↔ ∀ k ∈ T, ¬ ((lo k).Prime ∧ (hi k).Prime)
theorem census_pinned (h0 : n0 T = 0) :
    n1 T = primesIn T ∧ n2 T = T.card - primesIn T
theorem census_pinned_add (h0 : n0 T = 0) : n2 T + primesIn T = T.card
theorem census_pinned_prefix (t) (hX : ∀ k < t, ¬ ((lo k).Prime ∧ (hi k).Prime)) :
    n1 (range t) = primesIn (range t) ∧ n2 (range t) = t - primesIn (range t)
```

### Proof route

- Partition: `Finset.card_eq_sum_card_fiberwise` with fibering function
  `slotComps` into `range 3` (MapsTo from `slotComps_le_two`), then
  `sum_range_succ` twice + `sum_range_one` + `rfl`.
- `comps_eq`: `Finset.sum_fiberwise_of_maps_to`, each fiber's sum collapsed
  by `Finset.sum_const_nat` (constant value b on the fiber) to `card * b`;
  the b = 0 term vanishes.
- `primes_add_comps`: `sum_add_distrib` + per-slot `slotPrimes + slotComps
  = 2` (four-way `by_cases` + simp) + `sum_const_nat`.
- `primes_eq`, `census_pinned`: pure `omega` over the three identities —
  omega handles the card/sum terms as opaque atoms, including the ℕ
  subtraction in `n2 = N − P`. `census_pinned_add` is the subtraction-free
  form for downstream composition.
- `n0_eq_zero_iff` makes the hypothesis exactly Condition X:
  `card_eq_zero` + `filter_eq_empty_iff` + the per-slot iff.

### Build status

`lake build`: **Build completed successfully (976 jobs)**, zero sorry.
All five earlier libs unchanged and building.

### Axiom audit

All seven checked Census theorems: `[propext, Classical.choice, Quot.sound]`.

### Proposed next target

The demand side is now pinned; the natural next chunk is the BRIDGE to the
supply side: identify `compsIn (range t)` with Supply's root-partitioned
count over the corresponding member Finset (S = image of slots' members),
giving Σ_q R_q(t) = n1(t) + 2·n2(t) kernel-checked end to end — the formal
skeleton of the X-consistency equation's left-hand side. Alternative:
h(2) ≥ d's product inequality.

## Round 5 — The bridge identity (2026-08-18)

### What was done

New file `proofs/Bridge.lean` (namespace `Bridge`), importing Supply and
Census — the file that connects the supply side (root partition over
members) to the demand side (slot census). Registered in lakefile,
AxiomCheck extended. Compiled first try, zero sorry, zero warnings.

### Final theorem statements

```lean
def members (T : Finset ℕ) : Finset ℕ := T.image lo ∪ T.image hi

theorem card_members        : (members T).card = 2 * T.card
theorem card_comps_members  : ((members T).filter fun m => ¬ m.Prime).card = Census.compsIn T
theorem card_primes_members : ((members T).filter fun m => m.Prime).card = Census.primesIn T

theorem sum_roots_eq_census {y} (T) (hwin : ∀ k ∈ T, y < lo k ∧ hi k < y * y) :
    (∑ p ∈ (Finset.range y).filter Nat.Prime,
      ((members T).filter fun m => ¬ m.Prime ∧ m.minFac = p).card)
      = Census.n1 T + 2 * Census.n2 T

theorem sum_roots_pinned {y} (T) (hwin) (h0 : Census.n0 T = 0) :
    Σ_p R_p = Census.primesIn T + 2 * (T.card - Census.primesIn T)

theorem slot_roots_ne {k} (hk : 1 ≤ k) : (lo k).minFac ≠ (hi k).minFac
```

### Proof route

- `members T = T.image lo ∪ T.image hi`; `lo`/`hi` injective (omega through
  the truncated subtraction), images disjoint (`5 ≢ 1 mod 6`, omega). So
  each slot contributes its two members distinctly — the flagged subtlety
  is discharged at the count level by disjointness.
- Count bridge: `filter_union` + `card_union_of_disjoint` +
  `disjoint_filter_filter`, then `Finset.filter_image` swaps the member
  filter into a slot filter through each injection, `Finset.card_filter`
  turns the two filter cards into indicator sums, and a four-way `by_cases`
  matches their sum to `slotComps k` (resp. `slotPrimes k`) pointwise.
- `sum_roots_eq_census` is then three rewrites: Supply's identity backwards
  (window hypothesis transferred slot→member by `members_window`), Census's
  `comps_eq` backwards, and the count bridge. Nothing else.
- `sum_roots_pinned`: bridge + `census_pinned`, closed by omega (handles
  the ℕ subtraction).
- `slot_roots_ne` = `Supply.roots_ne` at `m = 6k−1` (odd, and
  `hi k = lo k + 2` for `k ≥ 1`) — root distinctness inside a slot.

### Build status

`lake build BlockedSlots Horizon Layer Supply Census Bridge`: **Build
completed successfully (981 jobs)**, zero sorry, all six formalist libs
green. Note: a seventh lib `Polignac` (added to the lakefile by another
workstream this round) currently has five compile errors in its own file —
pre-existing, untouched by me, and isolated: it does not affect the six
libs above. Flagged to the manager via agents-shared.

### Axiom audit

All six checked Bridge theorems: `[propext, Classical.choice, Quot.sound]`.

### Proposed next target

The X-consistency equation now has its LHS skeleton formal end to end.
Natural next chunk: per-gear decomposition of the bridge — split Σ_p R_p by
the fiber at a single gear q (R_q alone) and prove the ledger's per-gear
cap R_q(T) ≤ (number of multiples of q among members), connecting to the
deletion-spacing law; that is the first step toward the supply side's
freedom-free semiprime arithmetic. Alternative: h(2) ≥ d.

## Round 6 — Per-gear fiber and caps (2026-08-18)

### What was done

New file `proofs/Gear.lean` (namespace `Gear`), eighth lakefile target.
First file to import BlockedSlots for reuse (`card_blocked_by_le` becomes
the interval bound). Compiled first try, zero sorry, zero warnings.

### Final theorem statements

```lean
def R (q : ℕ) (S : Finset ℕ) : ℕ :=
  (S.filter fun m => ¬ m.Prime ∧ m.minFac = q).card

theorem supply_eq_sum_R (hS : window) :
    (S.filter fun m => ¬ m.Prime).card = ∑ p ∈ (range y).filter Nat.Prime, R p S
theorem sum_R_eq_census (hwin : slot window) :
    (∑ p ∈ (range y).filter Nat.Prime, R p (Bridge.members T))
      = Census.n1 T + 2 * Census.n2 T
theorem R_le_card_multiples : R q S ≤ (S.filter fun m => q ∣ m).card
theorem R_prefix_le (hq : 0 < q) :
    R q (Bridge.members (Finset.range t)) ≤ 6 * t / q + 2
theorem sq_le_of_minFac_eq (h1 : 1 < m) (hnp : ¬ m.Prime)
    (hfac : m.minFac = q) : q * q ≤ m
theorem R_eq_zero_of_below_sq (hS : ∀ m ∈ S, 1 < m ∧ m < q * q) : R q S = 0
```

### Proof route / design notes

- `R` is a def, and the two restatements (`supply_eq_sum_R`,
  `sum_R_eq_census`) are definitional repackagings of Supply/Bridge —
  zero-cost, but they give downstream files a named handle on one gear's
  ledger line.
- Set cap: `minFac m = q → q ∣ m` (rewrite `Nat.minFac_dvd`), filter-subset,
  `card_le_card`. Four lines.
- Interval cap: members of `range t` sit below `6t`, so the multiples
  filter injects into `(range (6t)).filter (q ∣ 0 + ·)` — literally
  `BlockedSlots.card_blocked_by_le 0 q (6t)`, giving `6t/q + 2`. Sharpness
  not fought for (per instructions): the bound composes and the `6t/q` term
  is what matters.
- Shadow law: `Nat.minFac_sq_le_self` — a gear's ledger line is empty below
  `q²`. Found and guarded a genuine edge case: `minFac 0 = 2`, so without
  `1 < m` the number 0 lands in gear 2's class and the law is false. The
  window hypotheses used everywhere else already give `1 < m`, so the guard
  costs callers nothing.

### Build status

`lake build` (all 8 targets, Polignac now green upstream): **Build
completed successfully (988 jobs)**, zero sorry.

### Axiom audit

All six checked Gear theorems: `[propext, Classical.choice, Quot.sound]`.

### Proposed next target

The gear ledger now has: its line (`R`), its total (bridge), its cap
(interval), and its onset (`q²`, shadow law). The natural next chunk is the
SEMIPRIME REFINEMENT of one line: in the window `(y, y²)` with `q < y ≤ q²`,
every member of gear q's class is `q * c` with `c` prime (this is exactly
`Layer.eq_mul_prime_of_minFac_eq` applied member-wise), giving
`R q S = #(primes c with q*c ∈ S)` — the freedom-free supply arithmetic's
first exact formula. Alternative: h(2) ≥ d.

## Round 7 — Semiprime refinement: one gear's line, exactly (2026-08-18)

### What was done

Extended `proofs/Gear.lean` (no new lakefile target — the refinement
belongs to the gear namespace and composes with `R` directly). One
iteration for two warnings (deprecated `Set.mem_setOf_eq` →
`Set.mem_ofPred_eq`; an unused hypothesis that revealed
`R_eq_card_partners` needs no positivity at all). Zero sorry, zero
warnings.

### Final theorem statements

```lean
theorem semiprime_of_fiber (hq : q.Prime) (h1 : 1 < m) (hnp : ¬ m.Prime)
    (hfac : m.minFac = q) (hcube : m < q * q * q) :
    ∃ c, c.Prime ∧ q ≤ c ∧ m = q * c

theorem not_prime_mul (hq : q.Prime) (hc : c.Prime) : ¬ (q * c).Prime
theorem minFac_mul (hq : q.Prime) (hc : c.Prime) (hqc : q ≤ c) :
    (q * c).minFac = q

def partners (q : ℕ) (S : Finset ℕ) : Finset ℕ :=
  (S.filter fun m => ¬ m.Prime ∧ m.minFac = q).image (· / q)

theorem R_eq_card_partners (q S) : R q S = (partners q S).card   -- unconditional
theorem mem_partners (hq : q.Prime) (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    c ∈ partners q S ↔ c.Prime ∧ q ≤ c ∧ q * c ∈ S
theorem window_bounds (hwin) (hy : 1 ≤ y) (hthin : y * y ≤ q * q * q) :
    ∀ m ∈ S, 1 < m ∧ m < q * q * q
```

### Design notes

- The coordinator's suggested regime `q < y ≤ q²` is not sufficient for
  the c-prime conclusion (counterexample: q = 5, y = 25, m = 175 = 5·35 is
  rooted at 5 with composite cofactor 35). The honest regime is every
  member `< q³` (window form: `y² ≤ q³`, i.e. gears `q ≥ y^(2/3)`), which
  is what `Layer.eq_mul_prime_of_minFac_eq` needs. Stated per-member
  (`hS`), with `window_bounds` as the window adapter.
- Second boundary case found: the square. `m = q²` is rooted at q but is
  not `q·c` with `c > q` — so the decomposition is stated with `q ≤ c`,
  equality exactly at the square (the shadow-law onset). The partner-set
  membership then comes out clean with no special-casing: the square's
  partner is `q` itself.
- `R_eq_card_partners` (the bijection `m ↦ m / q`) is UNCONDITIONAL — no
  primality, no positivity, no range: injectivity only needs that fiber
  members are multiples of their root (`Nat.mul_div_cancel'` twice).
  All the regime hypotheses live in `mem_partners` where they belong.
- Reverse inclusion needs `q * c ≠ 1` for `minFac_prime`; got it from
  `Nat.dvd_one` (q divides 1) rather than product arithmetic — omega
  cannot see variable products, a recurring theme.

### Build status

`lake build` (all 8 targets): **Build completed successfully (988 jobs)**,
zero sorry.

### Axiom audit

`semiprime_of_fiber`, `R_eq_card_partners`, `mem_partners` (and the round-6
theorems, re-audited): `[propext, Classical.choice, Quot.sound]`.

### Proposed next target

With `mem_partners`, R_q in the large-gear regime IS a prime count:
`R q S = #{c prime : q ≤ c, q*c ∈ S}`. Next natural chunk: specialise S to
the member set of a slot interval and identify WHICH slot `q*c` lands in
(slot arithmetic: `q*c = 6k ± 1` determines k = the semiprime slot of the
lateral workstream's pinned classes) — connecting the supply formula to
slot positions, the first step toward the placement (not just count) side
of the X-equation. Alternative: h(2) ≥ d.

## Round 8 — Placement: where the supply line sits (2026-08-18)

### What was done

New file `proofs/Placement.lean` (namespace `Placement`), ninth lakefile
target, importing Gear (whole stack transitively). Written before the
session cut, registered and verified after: typechecked clean on the first
`lake env lean` run, zero sorry, zero warnings. AxiomCheck extended.

### Final theorem statements

```lean
theorem prime_mod_six (hp : p.Prime) (h5 : 5 ≤ p) : p % 6 = 1 ∨ p % 6 = 5
theorem sign_law (ha : a % 6 = 1 ∨ a % 6 = 5) (hb : b % 6 = 1 ∨ b % 6 = 5) :
    ((a * b) % 6 = 1 ↔ a % 6 = b % 6)
theorem unit_mul (ha) (hb) : (a * b) % 6 = 1 ∨ (a * b) % 6 = 5

def slotOf (m : ℕ) : ℕ := (m + 1) / 6
theorem lo_slotOf (hm : m % 6 = 5) : Census.lo (slotOf m) = m
theorem hi_slotOf (hm : m % 6 = 1) : Census.hi (slotOf m) = m
theorem mem_members_iff_slot (hm : m % 6 = 1 ∨ m % 6 = 5) :
    m ∈ Bridge.members T ↔ slotOf m ∈ T

theorem slot_injOn_partners (hq : q.Prime) (h5 : 5 ≤ q)
    (hS : ∀ m ∈ S, 1 < m ∧ m < q * q * q) :
    Set.InjOn (fun c => slotOf (q * c)) (Gear.partners q S)
theorem card_slots_of_line (hq) (h5) (hS) :
    ((Gear.partners q S).image fun c => slotOf (q * c)).card = Gear.R q S
theorem R_slots_eq (hq : q.Prime) (h5 : 5 ≤ q) (hcube : 6 * t ≤ q * q * q) :
    Gear.R q (Bridge.members (Finset.Ico 1 t))
      = ((Finset.range (6 * t)).filter fun c =>
          c.Prime ∧ q ≤ c ∧ slotOf (q * c) ∈ Finset.Ico 1 t).card
```

### Design notes

- One simplification over the brief: `slotOf m = (m + 1) / 6` works for
  BOTH sign classes — `(6k−1+1)/6 = (6k+1+1)/6 = k` — so no case-split
  function, and every slot-arithmetic goal stays omega-friendly (no `if`).
- The sign law is a four-case `decide` after `Nat.mul_mod`; `prime_mod_six`
  is two divisor-exclusions plus omega (which handles `∣` and `%` by
  literals natively).
- Injectivity of placement is the slot cap in action: two partners in one
  slot means q divides both members at distance 2 — the mixed-sign cases
  land exactly on `Layer.slot_cap`, the same-sign cases are cancellation.
- The count corollary uses slot interval `Ico 1 t`, not `range t`: slot 0
  is degenerate (members 0 and 1, and 1 < m fails). Census identities are
  unaffected (any Finset), but placement statements should prefer `Ico 1 t`.
- Axiom notes: `sign_law` depends only on `propext`; `prime_mod_six` on
  `[propext, Quot.sound]` — the arithmetic core is nearly axiom-free.

### Build status

`lake build` (all 9 targets): **Build completed successfully (990 jobs)**,
zero sorry.

### Axiom audit

All six checked Placement theorems standard; see notes above for the
two that need less.

### Proposed next target

The ledger now knows count AND position of every large-gear line. Two
candidates: (a) the twin-product pin — the lateral workstream's closed form
`(p+1)² − 1 = p(p+2)`: for a twin pair (p, p+2), the product slot
`slotOf (p*(p+2))` is `6u'²`-structured and its membership claims are pure
arithmetic, a small file connecting Placement to Polignac's pinning
theorems; (b) h(2) ≥ d's product inequality (long-standing alternative).

## Round 9 — The (5,7) corridor: 32-cap + the pin unified (2026-08-18)

### What was done

New file `proofs/Corridor.lean` (namespace `Corridor`), tenth lakefile
target, importing Placement and Polignac (first formalist file to import
another workstream's lib — read-only composition, Polignac untouched).
Typechecked clean first try, zero sorry, zero warnings.

### Final theorem statements

```lean
-- The 32-cap
theorem exists_class_in_run (a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ (k % 35 = 1 ∨ k % 35 = 34)
theorem both_composite_of_class (hk : 2 ≤ k) (h : k % 35 = 1 ∨ k % 35 = 34) :
    ¬ (Census.lo k).Prime ∧ ¬ (Census.hi k).Prime
theorem both_composite_in_run (ha : 2 ≤ a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ ¬ (lo k).Prime ∧ ¬ (hi k).Prime
theorem double_slot_in_run (ha : 2 ≤ a) :
    ∃ k, a ≤ k ∧ k < a + 33 ∧ Census.slotComps k = 2
theorem prime_adjacent_run_le (ha : 2 ≤ a)
    (hrun : ∀ k, a ≤ k → k < a + L → (lo k).Prime ∨ (hi k).Prime) : L ≤ 32

-- The pin unified
theorem product_slotOf (hu : 6 * u = p + 1) :
    Placement.slotOf (p * (p + 2)) = u * (p + 1)
theorem product_slotOf_sq (hu) : slotOf (p * (p + 2)) = 6 * (u * u)
theorem twin_product_pin (hu) :
    slotOf (p*(p+2)) = u*(p+1) ∧ Census.lo (u*(p+1)) = p*(p+2)
      ∧ p ∣ lo (u*(p+1)) ∧ (p+2) ∣ lo (u*(p+1))
```

### Proof route

- The cap is exactly three moves, as Lateral predicted: (1) the class-gap
  lemma is a witness construction (`a + (1 − a%35)` or `a + (34 − a%35)`)
  with all checks omega (literal moduli); (2) both-composite at the classes
  is four applications of "proper divisor ≥ 2 kills primality", each an
  omega pair (5 ∣ 6k−1 from k ≡ 1 mod 35, and size); (3) assembly. The
  contrapositive `prime_adjacent_run_le` is the headline form.
- `k ≥ 2` guard: slot 1 IS the twin (5,7) — the unique class slot where
  both members are prime. The classes force k ≥ 34 or ≥ 36 anyway.
- Pin: `slotOf (p(p+2)) = ((p+1)² )/6 = u(p+1)` by `mul_div_cancel_left`
  after the ring identity; `twin_product_pin` then re-exports
  `Polignac.twin_product_slot` through `Census.lo` — the equation
  `6·(u(p+1)) − 1 = p(p+2)` IS `lo (u(p+1)) = p(p+2)` definitionally.

### Build status

`lake build` (all 10 targets): **Build completed successfully (992
jobs)**, zero sorry.

### Axiom audit

Every Corridor theorem except `double_slot_in_run` needs only
`[propext, Quot.sound]` — no Classical.choice anywhere in the cap or the
pin. (`double_slot_in_run` picks up choice through Census's decidable
counters; still standard.)

### Proposed next target

The cap gives every 33-window a double slot unconditionally; Census gives
n2 = N − P under X. A natural next chunk: window-count corollary — over
any range of W slots, n2 ≥ ⌊W/33⌋-ish lower bound by packing disjoint
33-windows (Finset counting, no new number theory), giving the formal
floor on doubles that the X-consistency demand side must meet. Alternative:
the tolerance lemmas' arithmetic skeletons (top-gap anti-clustering
inequality shell) if the Constructor lands the statement shape.

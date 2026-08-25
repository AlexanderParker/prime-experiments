# Formalist workstream - cumulative findings

Compacted 2026-08-23; full verbatim rounds 1-19 log at
archive/formalist-full-r1-19.md. (Rounds 12, 14 and 19 have no entries in
that log - the lane was not briefed those rounds; round-19 draft files
proofs/Machine19Core.lean and proofs/Machine19Probe.lean exist,
unregistered.)

Mandate: kernel-check the twin-prime proof-search results in Lean 4 /
mathlib - exact statements, zero sorries, axiom-audited; verify every claim
against the research/*.py tooling BEFORE formalising; record honest
"will not close" verdicts as first-class outputs.

## Infrastructure facts

- Lake does NOT glob proofs/: every root file needs its own `[[lean_lib]]`
  entry in `proofs/lakefile.toml` plus `defaultTargets`. Sibling modules
  imported by a root (the PolignacCap split) must ALSO each be declared as
  a `lean_lib` or imports fail with "unknown module prefix". Ledger at
  round 18: 17 targets + 9 module libs (25 libs), 1254 jobs (1252 at
  round 17), zero sorries, zero warnings.
- Build: `~/.elan/bin/lake.exe build` from `proofs/`. Axiom audit:
  `proofs/AxiomCheck.lean`, run with `~/.elan/bin/lake.exe env lean
  AxiomCheck.lean` from `proofs/`. No `native_decide`, no
  `Lean.ofReduceBool` anywhere in the ledger.
- The mathlib cache is PARTIAL: "mathlib has it" is not "we can use it".
  `Mathlib.Data.Finite.Basic` is not built (no `Finite (Fin n)` instance);
  `Mathlib.Data.ZMod.Basic` is.
- Kernel-scan scaling rules (measured): ~5e3 tuples per DECLARATION max,
  and only a handful of heavy `decide +kernel` declarations per MODULE -
  lake elaborates each module in its own process; both limits are
  per-process state, not total work.
- Load-bearing build times: machine-13 period scans 12.4 s;
  `forbidden_pairs_count` 22 s (`decide +kernel`, maxRecDepth 8192);
  machine-17 slices ~16 s per 5005-tuple slice, whole lib ~2 min;
  PolignacCap classes 17-60 s after the 38x speedup (10 min 48 s before).

## Kernel-checked theorems

Axiom footprint is the standard three `[propext, Classical.choice,
Quot.sound]` unless noted. Signatures verbatim from the round logs (a few
were logged in abbreviated form, marked `...`; full statements are in the
files).

### 1. proofs/Horizon.lean (968 jobs)

Interior form: gears strictly below y decide the open window (y, y*y) -
sharper than BlockedSlots.survivor_iff_twin (q <= y, closed window).

```lean
theorem exists_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (hnp : ¬ m.Prime) : ∃ p, p.Prime ∧ p < y ∧ p ∣ m

theorem prime_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hmyy : m < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m) : m.Prime

theorem twin_of_no_prime_factor_lt {y m : ℕ} (hym : y < m) (hwin : m + 2 < y * y)
    (h : ∀ p, p.Prime → p < y → ¬ p ∣ m ∧ ¬ p ∣ (m + 2)) :
    m.Prime ∧ (m + 2).Prime
```

### 2. proofs/Layer.lean (970 jobs)

Bertrand avoided by carrying the thin-layer bound y'^2 <= y^3 as an
explicit hypothesis (holds for consecutive primes from y = 3 on; the
caller discharges it). `slot_cap` needs only `[propext, Quot.sound]`.

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

### 3. proofs/Supply.lean (974 jobs)

Root partition by minFac. The window hypothesis is per-member, so S need
not be an interval.

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

### 4. proofs/Census.lean (976 jobs)

Slot k carries (6k-1, 6k+1) as `lo`/`hi`; counters tied to real
`Nat.Prime`. All over arbitrary `T : Finset Nat`; `n0_eq_zero_iff` is
exactly Condition X.

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

### 5. proofs/Bridge.lean (981 jobs)

Supply side (root partition over members) connected to demand side (slot
census) - the X-consistency equation's LHS skeleton, end to end.

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

### 6. proofs/Gear.lean (rounds 6-7, 988 jobs)

Per-gear ledger line R, cap, onset (shadow law), and the semiprime
refinement: below q^3 one gear's line IS a prime count.

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

Boundary facts: the square m = q^2 is rooted at q with partner q itself
(hence `q <= c`, not `q < c`); `minFac 0 = 2`, so the `1 < m` guard is
required or gear 2's fiber absorbs 0.

### 7. proofs/Placement.lean (990 jobs)

Where each large-gear line sits. `sign_law`: `[propext]` only;
`prime_mod_six`: `[propext, Quot.sound]`. `slotOf m = (m+1)/6` covers
both sign classes with no case split. Placement statements prefer
`Ico 1 t` (slot 0 is degenerate).

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

### 8. proofs/Corridor.lean (rounds 9-10, 992 jobs)

The (5,7) corridor: 32-cap, twin-product pin, endpoint/adjacency laws,
packing floor. Round-9 theorems except `double_slot_in_run` need only
`[propext, Quot.sound]`. Slot 1 IS the twin (5,7) - the unique class slot
with both members prime, hence the `k >= 2` guards. Round-10 claims
cross-verified against research/topgap_endpoint_law.py.

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

-- The pin unified (re-exports Polignac.twin_product_slot through Census.lo)
theorem product_slotOf (hu : 6 * u = p + 1) :
    Placement.slotOf (p * (p + 2)) = u * (p + 1)
theorem product_slotOf_sq (hu) : slotOf (p * (p + 2)) = 6 * (u * u)
theorem twin_product_pin (hu) :
    slotOf (p*(p+2)) = u*(p+1) ∧ Census.lo (u*(p+1)) = p*(p+2)
      ∧ p ∣ lo (u*(p+1)) ∧ (p+2) ∣ lo (u*(p+1))

-- Endpoint / adjacency laws (round 10)
def Exposed (k) : Prop := ¬5∣lo k ∧ ¬5∣hi k ∧ ¬7∣lo k ∧ ¬7∣hi k
def exposedSet : Finset ℕ := {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33}

theorem exposed_iff_mem (hk : 1 ≤ k) : Exposed k ↔ k % 35 ∈ exposedSet
theorem endpoint_law (ha : 1 ≤ a) (h1 : Exposed a) (h2 : Exposed (a+G)) :
    a % 35 ∈ exposedSet.filter fun r => (r + G) % 35 ∈ exposedSet
theorem endpoint_law_34 (hG : G % 35 = 34) ... :
    a % 35 = 3 ∨ a % 35 = 18 ∨ a % 35 = 33

def allowed3 (g1 g2) : Finset ℕ :=
  exposedSet.filter fun r => (r+g1)%35 ∈ exposedSet ∧ (r+g1+g2)%35 ∈ exposedSet
theorem adjacency_law ... : a % 35 ∈ allowed3 (g1 % 35) (g2 % 35)
theorem no_chain_of_forbidden (hf : allowed3 ... = ∅) ... : False
theorem forbidden_first_examples : allowed3 1 1 = ∅ ∧ ...
theorem forbidden_pairs_count :
    ((range 35 ×ˢ range 35).filter fun p => allowed3 p.1 p.2 = ∅).card = 294

theorem n2_packing (ha : 2 ≤ a) : W / 33 ≤ Census.n2 (Finset.Ico a (a + W))
```

`n2_packing` uses Classical.choice via `choose`; a Nat.find variant would
make it choice-free if ever needed. `forbidden_pairs_count` is kernel
`decide` - no ofReduceBool.

### 9. proofs/Machine13.lean (round 11, 996 jobs) - the y=13 alpha1 certificate

Verified against research/strata_adjacency.py on all 5005 residues first.
Tiers A + B + C ALL closed (at fixed y the period scan subsumes B and C;
tier A kept separate: machine-free, scales). Logged in shorthand:

```lean
theorem gap_le      ... : b - a <= 11          -- F_k(13) <= 11
theorem pair_sum_le ... : c - a <= 16          -- F2_k(13) <= 16
theorem gap11_realized   : openings 122,133 with nothing between  -- F  = 11
theorem pair16_realized  : openings 117,122,133 (gaps 5,11)       -- F2 = 16
theorem alpha1_certificate ... : 3 * (c - a) <= 3 * 11 + 1 * 17
theorem lemma1_at_13       ... : (c - a) - 11 <= 1 * 17
theorem tierA_forbidden : allowed3 of (6,11),(8,11),(11,6),(11,8),(11,11) = empty
theorem tierA_kills / no_11_11_chain : those chains cannot exist at all
```

`Machine13.w11` and `w16` (the two period scans) depend on NO axioms at
all - pure kernel computation.

### 10. proofs/MaxGap.lean (round 11) - F = 0 mod 3

`uncovered_span_mod_three` (two distinct blocked classes mod 3 leave one,
so any two survivors are congruent), `F_zero_mod_three` (3 | M+1 = F),
`M_two_mod_three`, `not_max_of_mod_three` (the pruning rule: a length not
= 2 mod 3 can never be maximal). Search bookkeeping - maximality forcing
both bounding positions uncovered, gear 3 active - taken as hypotheses;
all four need only `[propext, Quot.sound]`.

### 11. proofs/LiteralCap.lean (round 13, 998 jobs) - the twin literal cap

Verified against research/literal_cap_gap_d.py first (48 invertible
classes mod 210, cap spectrum {2:24, 3:4, 4:14, 6:6}).

```lean
def sOf (c : ℕ) : ℕ := (if c % 6 = 1 then (c - 1) / 3 else (c + 1) / 3) % 35
def wpos (t s r ph i : ℕ) : ℕ :=
  (r + ((i + ph) / 2) * t + (if (i + ph) % 2 = 1 then s else 0)) % 35

theorem no_run_seven :          -- THE FINITE CHECK
    ∀ c < 210, Nat.gcd c 210 = 1 →
      ∀ r < 35, ∀ ph < 2, run7 (c % 35) (sOf c) r ph = false

theorem s_eq (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) :
    (2 * u) % 35 = sOf (q % 210)

theorem literal_chain_le_six    -- THE CAP
    (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 ≤ r)
    (hE : ∀ i < L, Corridor.Exposed (member r q u ph i)) : L ≤ 6

theorem cap_six_classes_sharp : -- SHARPNESS
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ hasRun6 c = true)
      = {37, 53, 83, 127, 157, 173}
```

Literal chains have at most 6 members, at every gear, NO bound on q'; 6
is attained at exactly six classes so it cannot be lowered. Stated as "no
class admits SEVEN consecutive exposed walk members" - the sharpest form
that stays linear (48 x 35 x 2 x 7 tests, not a 140-step max-run).

### 12. proofs/Machine17.lean (round 15, 1002 jobs)

Landed by chunking the 85085-tuple scan into 34 explicit slice theorems
assembled with `interval_cases`. Logged in shorthand:

```lean
theorem gap_le      ... : b - a <= 18            -- F_k(17) = 18
theorem pair_sum_le ... : c - a <= 25            -- F2_k(17) = 25
theorem alpha1_certificate ... : 9 * (c - a) <= 9 * 18 + 4 * 19   -- 225 <= 238
theorem lemma1_at_17       ... : 3 * ((c - a) - 18) <= 4 * 19
```

`Machine17.w18All` and `w25All` depend on `[propext]` ONLY - the entire
85085-tuple period scan rests on one axiom. The 25 is tight (24 fails).

### 13. proofs/TierA.lean (rounds 15, 16, 18) - corridor law for chains of any length

`carrier` generalises `Corridor.allowed3` to a chain of any length; cost
independent of the machine - the piece that scales past the scans.
Round-15 core, logged in shorthand:

```lean
def offsets : List ℕ → List ℕ                    -- partial sums
def carrier (steps : List ℕ) : Finset ℕ          -- residues carrying the chain
theorem mem_carrier_of_chain : chain of openings → base residue in carrier
theorem no_chain_of_carrier_empty : carrier = ∅ → no such chain, anywhere
def flanked (F) (w) : List ℕ := F :: (w ++ [F])
theorem no_maximal_flanks : carrier (flanked F w) = ∅ → no both-maximal flanks
theorem padding_count_le / padding_at_most_one_below_onset
```

Flank steps closed by corridor arithmetic alone (constructor 24.3):
11->13 (w=(4), F=7), 13->17 (w=(6), F=11), 17->19 (w=(13), F=18),
23->29 (w=(19), F=34), 29->31 (w=(10), F=43). `flanks_17_19` is the sharp
one: each flank alone feasible mod 35, both together not. The honest
exception is itself a theorem: `flanks_19_23_nonempty : carrier (flanked
25 [8]) = {0, 5, 7, 12}`. All carriers checked against
research/flank_tierA_fix.py.

Round 16, lateral's padding corridor law (checked against lateral.md):

```lean
theorem no_adjacent_equal_padded (hc : carrier [q, q] = ∅) ... : False
theorem no_adjacent_padded_41 : carrier [41, 41] = ∅
theorem equal_padding_forbidden_classes :
    ((Finset.range 35).filter fun g => Nat.gcd g 35 = 1 ∧ carrier [g, g] = ∅)
      = {1, 4, 6, 9, 11, 16, 19, 24, 26, 29, 31, 34}
theorem equal_padding_forbidden_card : ... .card = 12
theorem padding_shape_dichotomy : ∀ g < 35, Nat.gcd g 35 = 1 →
    (carrier [g, g] = ∅ ↔
      carrier [g, (2*g) % 35] ≠ ∅ ∧ carrier [(2*g) % 35, g] ≠ ∅)
```

Two adjacent equal padded links are impossible at q' = 41 by the (5,7)
corridor alone - no spectrum input needed. Round 18, onset gate and
padding budget:

```lean
theorem onset_gate (hg : 0 < g) (hdvd : q ∣ g) (hF : g ≤ F) : q ≤ F
theorem padding_three_not_excluded : 13 * q ≤ 6 * F → 6 * (3*q) ≤ 6*F + 5*q
```

`onset_gate` (`[propext]` only): a padded link's interior gap is a
positive multiple of q' and is one of M's gaps, so q' <= F(M) - padding
cannot exist below onset. `padding_count_le` (`p <= F/q + 5/6`, a bound
that GROWS) needs NO axioms; `padding_at_most_one_below_onset`
`[propext, Quot.sound]`, says nothing at or above onset.
`padding_three_not_excluded`: once F >= (13/6) q the budget stops
excluding three padded links - lateral's p = 3 at 41->43.

### 14. PolignacCap (round 17, 1252 jobs) - the all-d literal cap

Files: `proofs/PolignacCapCore.lean` (defs + coprime lemma),
`PolignacCap{1,3,5,7,15,21,35,105}.lean` (one gcd class each),
`PolignacCap.lean` (root: `capOf`, `capOf_le_twelve`).

Harvester's halved-coordinate frame: position n is the pair
(2n+1, 2n+1+2e) for d = 2e; gear q blocks n = 0, -e (mod q); gear 3
FILTERS the candidate list (does not break runs). The cap depends only on
gcd(e,105), so eight theorems cover EVERY even gap:

    gcd(e,105)    1    5    7    3   21   35   15  105
    cap           6    6    6    6    6    6   10   12

All eight `cap_gcd_*` and `capOf_le_twelve` depend on NO AXIOMS AT ALL.
Each cap checked numerically sharp (scan fails at cap - 1); all eight
spectra reproduced independently first; the twin row reproduces
constructor's mod-35 table, cross-validating the frame change.
**12 is the absolute ceiling over all Polignac gaps**. gcd = 3 (the
d = 0 mod 6 case, densest gaps) still caps at 6. |E_e| matches
Hardy-Littlewood: prod over q in {3,5,7} of (q - r_q), r_q = 1 if q | e
else 2.

The coprime lemma (standard three axioms):

```lean
theorem exists_mul_mod_eq {n t : ℕ} (hn : 0 < n) (h : Nat.Coprime t n)
    {r : ℕ} (hr : r < n) : ∃ j, j < n ∧ (j * t) % n = r
```

It is the prerequisite of the single-cycle reduction (one orbit-length
walk replaces the whole start set - a 37x cut, verified exact) - on the
shelf for machines past 23.

### 15. proofs/Spectrum.lean (round 18, 1254 jobs) - THE BRIDGE IDENTITY

The load-bearing formal step of constructor's decomposition of (D).

```lean
def windowSum (g : ℕ → ℕ) (a j : ℕ) : ℕ := ∑ i ∈ Finset.range j, g (a + i)
def SpectrumBound (g : ℕ → ℕ) (j Fj : ℕ) : Prop := ∀ a, windowSum g a j ≤ Fj

theorem merged_eq (g a l) :
    g a + windowSum g (a+1) l + g (a+l+1) = windowSum g a (l+2)
theorem merged_le_spectrum (h : SpectrumBound g (l+2) Fj) :
    g a + windowSum g (a+1) l + g (a+l+1) ≤ Fj
theorem merged_le_spectrum_succ (h : SpectrumBound g ((l+1)+1) Fk) : ...
theorem merged_le_of_shallow (hl : l + 2 ≤ 4)
    (h4 : SpectrumBound g 4 F4) (hflat : F4 ≤ F + q) :
    g a + windowSum g (a+1) l + g (a+l+1) ≤ F + q
```

`merged_eq`: a word of l consecutive gaps plus its two flanks spans
exactly l + 2 = k + 1 CONSECUTIVE gaps, so merged length is a window sum
bounded by the spectrum value. `merged_le_of_shallow` derives (D) at
alpha = 3 from the two empirical halves - k_win <= 3 and shallow flatness
F_4 <= F + q' - with NO machinery in the statement. Both halves stay
hypotheses; nothing empirical is assumed inside the file.

### Five-part factorisation, formal status (round-18 audit)

- (A) finite word list from q' mod 210: PARTIAL - the class-reduction core
  is kernel-checked (`LiteralCap.s_eq`) and the length bound is
  `literal_chain_le_six`; the word-list ENUMERATION is computed, not
  checked.
- (B) literal span: FULLY kernel-checked, universally -
  `literal_chain_le_six` (twins) and `capOf_le_twelve` (every even d).
- (C) padded span: count bound (`padding_count_le`) and onset gate
  (`onset_gate`) both checked.
- (E) both-flanks-maximal exclusion: kernel-checked (`TierA.flanks_*`,
  `carrier`) but OFF-TARGET for (D) - see verdicts below.

## Honest "will not close" verdicts

1. **Tier A does not close 19->23.** `flanks_19_23_nonempty : carrier
   (flanked 25 [8]) = {0, 5, 7, 12}` - recorded as a theorem, not
   omitted. The mod-385 and direct tiers are genuinely needed there.
   Anyone building on tier A must carry this.
2. **(E) both-flanks-maximal exclusion is off-target for (D).**
   Constructor measured FS_max is attained at MID-SIZE flanks, never
   maximal ones (at 29->31, max FS = 48 at flanks (18,30) with F = 43).
   The exclusion theorems stand as corridor facts but rule out a
   configuration that never binds.
3. **The monotone-envelope / F_j spectrum route will not close (D).**
   Spectrum flatness FAILS at 29->31 (the 5-window max sits 42 above F
   where 31 is allowed) - so F_j was deliberately NOT formalised. If ever
   wanted: replace the literal 2 in `Machine17.pair25T` by j; the
   two-witness extraction generalises via the same Nodup-filtered-list
   argument.
4. **Residue laws cannot cap sizes** (constructor 20.2) - the mod-105 /
   mod-385 corridor transfer is not a route to size bounds.
5. **"Cap <= 6 for ALL (t,s) pairs mod 35" is FALSE** - over all 1225
   pairs the spectrum runs {2,3,4,5,6,8,10,140}. The restriction to
   invertible classes mod 210 does real work: the cap is not a property
   of the exposed set alone, it needs the arithmetic of q'.
6. **The regime q < y <= q^2 is insufficient for the c-prime semiprime
   conclusion** (counterexample: q = 5, y = 25, m = 175 = 5*35). The
   honest regime is m < q^3 (`window_bounds` is the window adapter).
7. **Tier-C wall, measured then REVISED.** Round 15, with the encoding
   of the day: machine 19 (period 1,616,615) = 323 slices ~ 86 min;
   machine 23 (37.2M) ~ 33 h - "tier C formalisable up to about machine
   19 and no further by this route". Round 18 revision after the 38x
   speedup stack: machine 19 lands in single-digit minutes, machine 23
   becomes an overnight job. The wall was an artefact of the encoding,
   not the mathematics; the single-cycle reduction via
   `exists_mul_mod_eq` removes the scan entirely if a machine is ever
   truly out of reach.
8. **Round-18 correction of round-15 padding claims:** the count bound is
   budget arithmetic, NOT constant; F < q is not "the onset condition"
   but (by `onset_gate`) precisely the regime where NO padded link
   exists. The theorems were hypothesis-explicit and never false;
   headings/docstrings overclaimed and were restated.

## Failed approaches and standing lessons

Tactic-level:

- `omega` cannot see nonlinear atoms (y*y, variable products) - use
  `linarith` (treats y*y as opaque) for window inequalities; derive
  `q*c != 1` from `Nat.dvd_one`, not product arithmetic.
- One-shot `omega` dies at 5 simultaneous dvd atoms. Fix: per-gear iffs
  (one dvd <-> one residue each), bridge k%5 = k%35%5 etc., generalize
  r = k%35, then `interval_cases r <;> decide`.
- An 8-conjunct Bool/Prop bridge iff times out at 1M heartbeats under
  `tauto` OR `omega` even when each half is fast. Fix: normalise instead
  of searching -
  `simp only [expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]`.
- When a product of two variables is bounded on one side,
  `interval_cases` the bounded one first: `((i+ph)/2) * q` becomes
  `literal * q`, linear, and omega closes each case.
- Index-shift goals over window sums: do the shifts as explicit `rw`s -
  `congr 1; omega` and `norm_num` both fail; omega cannot close until the
  g-atoms are syntactically identical.
- This mathlib's `Finset.range_subset` has a different shape - supply
  subsets pointwise. `Finset.sum_const_nat` is the Nat-native collapse
  (`smul_eq_mul` is missing under minimal imports).
  `windowSum_mono` needs `Mathlib.Algebra.Order.BigOperators.Group.Finset`.

Kernel-computation-level:

- Direct `decide` over residues mod 5005 does not terminate (both
  `Nat.decidableBallLT` and `List.all` shapes killed after 5+ min). Scan
  the CRT TUPLE instead ((all a < 5, b < 7, c < 11, d < 13), shifts
  mod each gear separately): same 5005 cases, single-digit moduli, 12.4 s.
  The general recipe for any machine whose period is a product of small
  primes.
- At 85085 tuples the limit is tuples PER DECLARATION, not total:
  `decidableBallLT` over all coords blows the proof TERM (2 GB+); one
  Bool with 5 nested `List.all` makes the term `rfl` but evaluation
  never finishes; (all e < 17, slice e = true) by `decide +kernel` is
  still > 600 s (a Prop quantifier over Bool slices does NOT behave like
  separate declarations). What works: explicit slice theorems +
  `interval_cases` assembly.
- Eight `decide +kernel` calls in ONE file: memory past 2.3 GB, 20+ min,
  even though each alone is 17-60 s. Split into separate modules under
  one root - lake gives each its own process.
- Big literal tables (`forbidden_pairs_count`, 1225 pairs): plain
  `decide` hits elaborator maxRecDepth; `set_option maxRecDepth 8192` +
  `decide +kernel` keeps it axiom-clean.
- Speed stack (38x, general): allocation-free fuel-recursive Bool over
  Nat state (list allocation dominates kernel time); restrict starts to
  the exposed/opening set (2-7x); measure fuel instead of guessing.

Modelling-level (the kernel caught real errors):

- First pairT formulation quantified over ALL window starts, not
  openings - `decide` reported it FALSE (1296 counterexamples confirmed).
  The real F2 statement requires the window to start at an opening.
- Gear-3 skip semantics in the halved frame: gear 3 filters the candidate
  list; a 3-inadmissible kill is SKIPPED and the run continues across it.
  Treating gear 3 like gears 5/7 gives max caps 2/4 instead of 6/10/12.
- `minFac 0 = 2`: without `1 < m`, 0 lands in gear 2's fiber and the
  shadow law is false.
- Verify every claim against the research/*.py tooling BEFORE
  formalising (caught the above and validated every table).
- Test a new lemma in a scratch file before placing it upstream of
  multi-minute decides. This caught the dead
  `Finite.injective_iff_surjective` route for `exists_mul_mod_eq`
  (missing `Finite (Fin n)` instance); the working route is `ZMod n`
  units (`ZMod.unitOfCoprime` etc.).
- Deprecations in this mathlib: `push_neg` -> `push Not`;
  `Set.mem_setOf_eq` -> `Set.mem_ofPred_eq`.

## Open formalisation targets (priority order)

1. **The (A) word-list enumeration gap**: the word LIST as a function of
   q' mod 210 is computed, not kernel-checked. Same shape as the
   LiteralCap 48-class check; affordable with the round-17 encoding.
2. **Suppression-corrected flatness as a hypothesis-explicit theorem**:
   wire `merged_le_of_shallow` to a concrete machine by proving a
   `SpectrumBound g 4 F4` instance from a period scan - the certificates
   already produce F_1 and F_2; F_4 is the same encoding with the count
   threshold raised. (Contingent on mechanic's two halves surviving at
   machines 31/37/41.)
3. **Tier-C machine 19** (period 1,616,615): single-digit minutes under
   the current encoding (verdict 7). Draft files proofs/Machine19Core.lean
   and proofs/Machine19Probe.lean exist, unregistered.
4. **CRT collapse / single-cycle reduction**: one orbit-length walk
   replaces the whole start set (37x) whenever the step is invertible mod
   the modulus; the prerequisite `exists_mul_mod_eq` is already proved in
   PolignacCapCore. The named construct for pushing past machine 23.

## Round 20

Ledger: 29 targets + 26 module libs, **1276 jobs** (1254 at round 19), zero
sorries, zero warnings in owned files. All three briefed targets landed plus
one pickup from lateral's round-20 list. New libs: `LiteralCapTable`,
`Machine19` + `Machine19Core` + `Machine19S0..S16` (all registered, root in
defaultTargets). Axiom audit run: every new theorem on the standard three or
fewer; **`Machine19.sliceAll` depends on `[propext]` ONLY** - the whole
1,616,615-slot period scan on one axiom, like `Machine17.w18All`.

### 16. proofs/LiteralCapTable.lean - the (A) word-list enumeration, closed

Verified before formalising: per-class caps recomputed in the corridor frame
and cross-checked against research/literal_cap_gap_d.py's 140-step max-run
computation (48/48 classes, zero mismatches; equivalence of start-anchored
and anywhere-in-walk runs holds because a shifted start is another (r, ph)
pair); every realized chain length in research/data/fuel_census.csv respects
its class cap, saturating it at q' = 19 and 31.

```lean
def runL (t s r ph L : N) : Bool        -- run6/run7 at any length
def hasRunL (c L : N) : Bool
def capC (c : N) : N                    -- the explicit 48-class table

theorem cap_table_maximal :             -- upper: NO run of capC c + 1
    forall c < 210, Nat.gcd c 210 = 1 ->
      forall r < 35, forall ph < 2,
        runL (c % 35) (LiteralCap.sOf c) r ph (capC c + 1) = false
theorem cap_table_realized :            -- exact: a run of capC c EXISTS
    forall c < 210, Nat.gcd c 210 = 1 -> hasRunL c (capC c) = true

theorem literal_chain_le_capC {q u r ph L : N}
    (hu : 6 * u + 1 = q or 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 <= r)
    (hE : forall i < L, Corridor.Exposed (LiteralCap.member r q u ph i)) :
    L <= capC (q % 210)
theorem word_length_lt_capC ... : ell < capC (q % 210)  -- (A) in word form

theorem cap_two_classes   : ... = {11, 13, ..., 199}    -- 24 classes
theorem cap_three_classes : ... = {29, 59, 151, 181}
theorem cap_four_classes  : ... = {1, 23, 31, 61, 67, 89, 97, 113, 121,
                                   143, 149, 179, 187, 209}
theorem cap_six_classes   : ... = {37, 53, 83, 127, 157, 173}
theorem no_cap_five       : ... = empty  -- the spectrum {2,3,4,6} has a hole
theorem cap_spectrum_counts : cards 24 / 4 / 14 / 6
theorem hasRunL_mono, capC_le_six
```

(A) status change: the word list of R21/R26 - alternating words over
`{2u', q'-2u'}`, two per length, lengths `1 .. capC-1` - is now COMPLETE as
a kernel-checked function of `q' mod 210` alone, exact in both directions.
The five-part audit line becomes: (A) FULLY kernel-checked.
docs/novel/literal-cap.md status upgraded accordingly.

Also here (pickup from lateral round 20): **`tripled_teeth_antipode`** - the
T3 law. For `6u = q -+ 1`: `{3u, q - 3u} = {(q-1)/2, (q+1)/2}` in exact
integer form, every gear forever (lateral had asserted it numerically to
100,000). Two-line omega proof; status upgraded in
docs/novel/golden-spectral-gap.md.

### Spectrum.lean round-20 block - the qualifying spectrum

Mechanic's Q_j (research/qspec_table.py: max sum of j consecutive gaps whose
j-2 MIDDLE gaps are all >= the floor a = 2u') is now a formal object, and
suppression-corrected flatness is hypothesis-explicit in the form the
censuses discharge:

```lean
def Qualifying (g : N -> N) (u a j : N) : Prop :=
  forall i, 1 <= i -> i + 1 < j -> 2 * u <= g (a + i)
def QualBound (g : N -> N) (u j Qj : N) : Prop :=
  forall a, Qualifying g u a j -> windowSum g a j <= Qj

theorem qualifying_of_word (hw : forall i < l, 2 * u <= g (a + 1 + i)) :
    Qualifying g u a (l + 2)
theorem merged_le_qual (hQ : QualBound g u (l + 2) Qj) (hw) :
    g a + windowSum g (a + 1) l + g (a + l + 1) <= Qj
theorem merged_le_of_qual_flat (hQ) (hflat : Qj <= F + q) (hw) :
    merged <= F + q
theorem merged_le_of_qual_flat_all (Q : N -> N)
    (hQ : forall j, QualBound g u j (Q j)) (hflat : forall j, Q j <= F + q) :
    forall a l, (forall i < l, 2 * u <= g (a + 1 + i)) -> merged <= F + q
theorem merged_le_of_corrected            -- R31's two-part lambda form,
    (hflat : Fj <= F + q + lam * l * L)   -- Qualifying hypothesis explicit
    (hsupp : forall b, Qualifying g u b (l+2) ->
               windowSum g b (l+2) + lam*l*L <= Fj)
    (hw) : merged <= F + q
theorem alphabet_ge_floor : 2 * u <= q - 2 * u  -- both literal letters
theorem padded_ge_floor   : 2 * u <= q          -- and padded letters qualify
```

`merged_le_of_qual_flat_all` is the word-free criterion: `Q_j <= F + q'` at
every depth gives (D) for every floor-respecting word of every length - NO
k_win, NO fuel, NO word list in the statement; `Q_j = 0` (no qualifying
window that deep) discharges deep depths for free, which is exactly how
mechanic's tables behave.

### 17. Machine19 - third machine certified, and the FIRST WIRED INSTANCE

proofs/Machine19Core.lean (defs), Machine19S0..S16.lean (323 slices of 5005
CRT tuples, 19 per file), Machine19.lean (assembly + consequences). Round
15's "tier C caps at machine 19" wall is formally dead. Verified over the
full period numerically first: F_1..F_5 = 25, 31, 35, 38, 47; openings
378,675; fuel row (19,23) N3 = 62, k_max = 3.

```lean
theorem sliceAll : forall e < 17, forall f < 19, slice e f = true
                                          -- [propext] ONLY
theorem gap_le      ... : b - a <= 25     -- F_k(19)  = 25
theorem pair_sum_le ... : c - a <= 31     -- F2_k(19) = 31
theorem quad_sum_le ... : e - a <= 38     -- F4_k(19) = 38 (NEW: depth 4)
theorem alpha1_certificate ... : 9 * (c - a) <= 9 * 25 + 4 * 23  -- 279<=317
theorem lemma1_at_19       ... : 3 * ((c - a) - 25) <= 4 * 23
theorem shallow_flatness   ... : e - a <= 25 + 23  -- F_4 <= F+q' (38 <= 48)

-- the machine's REAL gap sequence, formal:
theorem exists_exposed_above (k) : exists m, k < m and Exposed19 m
def nextOp (k) := Nat.find (exists_exposed_above k)
def opSeq : N -> N                       -- the openings in increasing order
def g19 (n) := opSeq (n + 1) - opSeq n   -- the gap word
theorem windowSum_g19 :
    Spectrum.windowSum g19 a j = opSeq (a + j) - opSeq a
theorem spectrum_four      : Spectrum.SpectrumBound g19 4 38
theorem spectrum_four_flat : Spectrum.SpectrumBound g19 4 (25 + 23)
theorem D_of_shallow_word {a l : N} (hl : l + 2 <= 4) :
    g19 a + Spectrum.windowSum g19 (a + 1) l + g19 (a + l + 1) <= 25 + 23
```

`D_of_shallow_word` is (D) at alpha = 3 at machine 19 as a theorem about the
machine's own gap word: `merged_le_of_shallow`'s flatness half is discharged
by the kernel scan, and the ONLY remaining hypothesis is the word's
shallowness. Census facts for context: k_max = 3 at 19->23 and the winning
word (8,15) has l = 2 (depth 4) - covered. A deep word (l >= 3) is not
covered; measured this round for the record (full-period Python, floor 8):
Q_4(19) = 37, Q_5(19) = 38, Q_6(19) = 0 - the qualifying criterion holds at
EVERY depth with margin >= 10 and the fuel cap arrives free at depth 6.

Scan engineering (measured): ~13 s per 5005-tuple slice with the round-18
encoding plus the third window fact (the F4 walk costs ~40% over F+F2 alone
- 246 s per 19-slice file); whole machine ~70 min of kernel time.

### New failed-approach / infrastructure lessons

- **Parallel slice-family builds die on MEMORY, not CPU.** A 16-target
  `lake build` invocation ran ~5 concurrent module processes on a 16 GB
  machine (~2-3 GB each, 1.7 GB free system-wide) and 10 of 16 targets
  failed; every failed target succeeds standalone. Lake (5.0.0 here) has no
  jobs flag: bound concurrency by invoking `lake build` with at most 2-3
  targets at a time, sequentially. The failure mode in the log is just
  "error: build failed" with the failed targets listed - read the WHOLE
  list, not the tail (this round initially rebuilt 3 of 10 because the list
  was truncated by `tail -5`).
- **The sorry'd-assembly dry-check pattern works.** Copy the root file, swap
  the slice imports for the core import, replace the assembly theorem's
  proof by `sorry`, and `lake env lean` it: the entire 300-line root
  (witness extraction with 4 witnesses, the opSeq/Nat.find development, the
  wired instance) elaborated before ANY slice had finished, so the root
  compiled first try when the slices landed. Cost: zero kernel time.
- `lake env lean` can fail transiently with "failed to read ...
  .olean.private" while a parallel `lake build` is running - retry, don't
  debug.
- 4-witness extraction from a `countP` fact: convert with
  `List.countP_eq_length_filter`, rcases the filtered list to
  `w::x::y::z::rest` (length contradictions close the short cases), get
  pairwise distinctness by three `List.nodup_cons.mp` unpackings, derive
  per-witness 4-way disjunctions (= b, = c, = d, or >= e) by the by_contra
  cascade, and let one final `omega` do the 4-distinct-values-in-3-slots
  pigeonhole (256 implicit cases - fine).
- `Nat.find` is fully usable here: the opening predicate is decidable, so
  the gap sequence `opSeq`/`g19` is computable and its API (`find_spec`,
  `find_min`) gives consecutiveness for free. No choice needed beyond the
  standard footprint.

### Open formalisation targets (re-prioritised after round 20)

1. **R39's inequality** (constructor's request):
   `F(M+q') <= max(F2, max_j qualmax_j)` - needs the merge law as a
   two-machine statement (every new gap is a window sum of old gaps whose
   interiors are q'-killed, hence residue-qualifying). The
   Qualifying/QualBound vocabulary landed this round is its target language;
   this is the route's live criterion and the top target.
2. **Q_5(19) <= 48 kernel scan** - would remove even the shallowness
   hypothesis from `D_of_shallow_word` at machine 19 (a 49-step-walk variant
   of the current encoding, ~2x cost, same slice recipe).
3. **Machine 23** (period 37.2M): overnight at ~10 h kernel time with the
   F4 walk included; extends the certificate ladder and the wired instance
   to the next step (needs F4(23) <= 34 + 29 = 63).
4. **Lateral's depth-sum identity at a fixed machine**:
   `sum_j W_j(g) = prod_q c_q(g)` at machine 13 - finite, medium design
   work (the window <-> endpoint-pair bijection plus a CRT count).
5. **Harvester's paired-Holt coef rung** (5005 -> 85085): coef
   position-freeness is near-definitional; the rung verification is a
   machine-17-scale scan with word extraction.
6. **Constructor's renewal-ladder validity** (finite IE + CRT, per step).

## Round 21

Ledger: +22 modules (MergeLaw, TwoTeeth, Machine19QCore, Machine19QProbe,
Machine19QS0..16, Machine19Q, Machine23), all registered with `[[lean_lib]]`,
4 new defaultTargets (MergeLaw, TwoTeeth, Machine19Q, Machine23). Build green
at **1302 jobs** (1276 at round 20), zero sorries, zero warnings in owned
files. Axiom audit: every new theorem on the standard three or fewer;
**`Machine19.qsliceAll` depends on `[propext]` ONLY** - the whole
1,616,615-slot qualifying scan on one axiom, like round 20's `sliceAll`.
All census inputs verified against full-period Python BEFORE formalising
(scratchpad verify_m19_r21.py: F ladder 25/31/35/38/47, ZERO 4-runs of gaps
>= 8, Q_3..Q_6(floor 8) = 35/37/38/0, 19->23 letters exactly {8,15,23},
merge-depth histogram j=1..4 = 7206695/733672/11746/62, F(23) = 34;
check_two_teeth.py: spacing law for every prime gear 5..199). Every job the
round launched finished before this write-up.

### 22. proofs/MergeLaw.lean - R39 as a two-machine kernel statement

Constructor's exact qualmax criterion, abstract in the machine: `pos` = the
old machine's opening enumeration, `kap` = the kill predicate on opening
indices, teeth `{u, q-u}`.

```lean
theorem sub_mod_eq (hxy : x <= y) (hal : al < q) (hbe : be < q)
    (hx : x % q = al) (hy : y % q = be) : (y - x) % q = (q + be - al) % q
def MergedWindow (kap : N -> Prop) (a j : N) : Prop :=
  0 < j and not kap a and not kap (a + j) and forall i, 0 < i -> i < j -> kap (a + i)
theorem interior_gap_mod ... :        -- RESIDUE NECESSITY
    g (a+i) % q = 0 or g (a+i) % q = 2*u or g (a+i) % q = q - 2*u
theorem floor_of_mod (hG : 0 < G) (h4u : 4*u <= q) (h : ...) : 2*u <= G
theorem newgap_le            -- THE CORE: merged window sum <= B whenever
    (hF2 : SpectrumBound g 2 F2) (hF2B : F2 <= B)      -- F2 <= B and
    (hQ : forall j, 3 <= j -> QualBound g u j (Q j)) (hQB : forall j, Q j <= B)
    (hmw : MergedWindow kap a j) : Spectrum.windowSum g a j <= B
theorem newgap_le_max ... : windowSum g a j <= max F2 Qmax   -- R39 verbatim
theorem D_of_qualmax  ... : windowSum g a j <= F + qp        -- (D) form
```

The consumers are `Spectrum.SpectrumBound` / `Spectrum.QualBound` instances
(the round-20 vocabulary), which the per-machine scans provide. Nothing
empirical inside. NOTE for the route: with Mechanic's F_3(37) = 97 and
Constructor's R44, R39 is now DECIDED at 37->41 - this file is the abstract
statement any such decision instantiates.

### 23. proofs/TwoTeeth.lean - the kill-spacing law, T1-T5 kernel-checked

Constructor's docs/novel/two-teeth-kill-spacing.md T1-T5, all kernel-checked
(their doc updated with pointers; my duplicate draft doc folded into it):

```lean
def Kill (q u x : N) : Prop := x % q = u or x % q = q - u
theorem next_kill_of_lo ... : y - x = q - 2*u and y % q = q - u  -- exact form
theorem next_kill_of_hi ... : y - x = 2*u and y % q = u
theorem kill_spacing    ... : y - x = 2*u or y - x = q - 2*u   -- {2u', q'-2u'}
theorem kill_spacing_min .. : 2*u <= y - x                     -- min ~ q'/3
theorem kill_period     ... : z - x = q       -- alternation: pairs sum to q'
theorem gear_side (hu6 : 6*u+1 = q or 6*u = q+1) (hq : 5 <= q) : 0 < u and 4*u < q
theorem teeth_letters   ... : 2*u + (q - 2*u) = q and ((6*u)%q = q-1 or = 1) -- T1
theorem spacing_from_lo ... : ((y-x)%q = 0 and y%q = u) or
    ((y-x)%q = q-2*u and y%q = q-u)           -- T2+T3, padding-transparent
theorem spacing_from_hi ... : (stay) or ((y-x)%q = 2*u and y%q = u)
theorem kills_gap_ge    ... : 2*u <= y - x    -- T4 general: ANY two kills
theorem fuel_span_cap   ... : 2*u*(k-1) <= x (k-1) - x 0       -- T5
theorem fuel_le         ... : k <= 1 + (x (k-1) - x 0) / (2*u) -- ~3L/q'
```

Verified numerically first for every prime gear 5..199 (spacings,
alternation, minimum; scratchpad check_two_teeth.py, zero mismatches).

### 24. Machine19Q - the qualifying spectrum closed at EVERY depth

The scan (`Machine19QCore.lean` + `Machine19QS0..16`, 323 slices): ONE
five-step `seekT` walk per opening reads off `F_3 <= 35` (third next
opening within 35), `F_5 <= 47` (fifth within 47) and the `Q_6 = 0` carrier
(the next four gaps never all >= 8). 12x faster than a first countP-based
encoding (97 s vs 1150 s per slice under identical load), and cheaper than
round 20's `okT` (the walk stops at o5 <= 47 vs 25+31+38 slots in three
passes). Extraction needs NO witness pigeonhole: `seek_next` proves the
walk computes `nextOp` exactly (fuel 25 suffices by round 20's `gap_le`),
so the chain IS the consecutive openings.

```lean
theorem seek_next (hx : 1 <= x) (hE : Exposed19 (x + s)) :
    x + seekT (x%5) (x%7) (x%11) (x%13) (x%17) (x%19) 25 s = nextOp (x + s)
theorem chain_facts (n : N) :
    opSeq (n+3) - opSeq n <= 35 and opSeq (n+5) - opSeq n <= 47 and
      not (8 <= g19 n and 8 <= g19 (n+1) and 8 <= g19 (n+2) and 8 <= g19 (n+3))
theorem no_big_run (n) : not (four consecutive g19 gaps all >= 8)  -- Q_6 = 0
theorem spectrum_ladder : F_1..F_5 <= 25, 31, 35, 38, 47 over g19  -- kernel-fed
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g19 4 j 47
theorem qual_five_flat : Spectrum.QualBound g19 4 5 (25 + 23)   -- the brief's
                                                 -- Q_5(19) <= 48, subsumed
theorem D_of_word {a l : N} (hw : forall i < l, 8 <= g19 (a + 1 + i)) :
    g19 a + Spectrum.windowSum g19 (a+1) l + g19 (a+l+1) <= 25 + 23
theorem opSeq_surj (hm : 1 <= m) (hE : Exposed19 m) : exists n, opSeq n = m
```

`D_of_word` is (D) at alpha = 3 at machine 19 for EVERY word length - round
20's shallowness hypothesis is GONE (depths 2-5 flat under 48 by the kernel
ladder, F_5 = 47 needing no qualifying constraint at all; depths >= 6 empty
by `no_big_run`). Only the letter floor remains - and `Machine23.lean`
discharges even that. `opSeq_surj` (strong induction on the distance via
`nextOp` minimality) makes the enumeration complete - the missing piece for
instantiating MergeLaw on a real machine.

### 25. proofs/Machine23.lean - (D) at 19->23, END TO END, NO HYPOTHESES

```lean
def Killed23 (k : N) : Prop := k % 23 = 4 or k % 23 = 19   -- teeth {u', 23-u'}
def Exposed23 (k : N) : Prop := Exposed19 k and not(23 | lo k) and not(23 | hi k)
def g23 (n : N) : N := opSeq23 (n+1) - opSeq23 n    -- machine 23's own gaps
theorem merge_alphabet (hk1 : Killed23 x) (hk2 : Killed23 y) (hxy : x < y)
    (hle : y - x <= 25) : y - x = 8 or y - x = 15 or y - x = 23
theorem g23_le (n : N) : g23 n <= 47
theorem D_at_19_23 (n : N) : g23 n <= 25 + 23
```

`g23_le` = `MergeLaw.newgap_le` instantiated with machine 19's kernel
bounds through `opSeq_surj`: every machine-23 gap is a merged window of
machine 19, its interiors killed by gear 23's teeth, hence residue- and
size-qualifying, hence <= max(F2, max_j Q_j) = 47. `D_at_19_23` is the
first machine step where (D) at alpha = 3 is FULLY kernel-checked with no
hypotheses at all: flatness, the qualifying spectrum, the fuel cap
(Q_6 = 0) and the floor are all discharged by the period scans + the merge
law. (No machine-23 period scan was needed - the merge law replaced a 37.2M
scan; census cross-check F(23) = 34, so 47 is a true, untight bound and
(D)'s 48 clears it.)

### Infrastructure lessons (round 21)

- SHARED-MACHINE CONTENTION: other lanes' live sessions (SAT refutations,
  full-period censuses) ran throughout - lake/lean slowed 5-40x, and lean
  processes started in background get starved; raising them to AboveNormal
  restored ~full speed (a persistent booster loop handled new leans). A
  killed lake leaves `.lake/config/N/lakefile.olean.lock` stale - remove it
  or later invocations hang. Lake runs `git rev-parse` + `git diff
  --exit-code HEAD` per invocation (slow on a busy repo). lakefile.toml
  edits trigger a big trace/replay pass. The round-20 process sweep struck
  TWICE (killed both slice loops mid-run); skip-if-built resume loops meant
  zero loss both times.
- ENCODING: read MULTIPLE window facts off ONE walk. countP-per-fact
  (rounds 18-20) re-walks the window per fact; a seekT chain visits each
  slot once, stops at the last needed opening, and turns extraction into
  equations (no Nodup/pigeonhole). 12x measured. General recipe for future
  scans: walk to the k-th next opening, assert positions.
- The sorry'd-assembly dry-check generalises to a MEGA-DRY file: concatenate
  all new modules (imports stripped) over already-built imports, one
  `lake env lean` - four files' full elaboration at zero kernel cost, no
  lake-lock conflict with a running build. Caught 3 real bugs (a Nat.find
  vs nextOp atom mismatch for omega; two goals closed early by defeq).
- `if_pos`/`if_neg` are deprecated in this mathlib: prove the two unfolding
  equations once with `split` (`seekT_succ_pos`/`_neg`) and `rw` with them.
- `ring` is not available under minimal imports - use `Nat.mul_succ` for the
  product step; `Nat.modEq_iff_dvd'` needs `Mathlib.Data.Nat.ModEq`
  (cached); omega cannot mix `Nat.find` with a def wrapping it (state the
  find-fact in the def's terms explicitly), and cannot relate
  variable*variable products (link them with an explicit
  `Nat.mul_succ`/`mul_comm` equation).

### Not taken (with reasons)

- Depth-sum identity at machine 13 (priority 4): the window half needs a
  machine-13 `opSeq` development + the window<->pair bijection; the CRT
  half alone (N2(g) = prod c_q(g)) is cheap but carries none of the
  content. The opSeq/opSeq_surj recipe built this round makes it affordable
  - named next-round target.
- Machine-23 period scan: deliberately NOT run - the merge law made it
  unnecessary for (D) (see 25). A direct F(23) = 34 certificate remains an
  overnight option if exactness is ever wanted.
- `Machine19QProbe` (one-slice canary) kept in the ledger: it is the cheap
  timing probe for future encoding changes.

### Open formalisation targets (re-prioritised after round 21)

1. **Extend the two-machine instance up the ladder**: 23->29 needs a
   machine-23 qualifying ladder (Q_j(23) <= F(23) + 29 at all depths) -
   same chain-scan recipe; overnight-scale. MergeLaw + opSeq_surj machinery
   now make each new step mechanical: scan ladder -> instantiate.
2. **Depth-sum identity at m13** (recipe above).
3. **Machine 23 exact certificate** (F, F2, F4) if the route wants
   exactness rather than bounds.
4. **Harvester's paired-Holt coef rung** (5005 -> 85085), unchanged.

## Round 22

Ledger: +13 libs (Machine11, Machine13QCore, Machine13QS, Machine13Q,
Machine17QCore, Machine17QS0/1/2, Machine17Q, Ladder, DepthSum, Potential,
Potential19; 7 new defaultTargets), all registered with `[[lean_lib]]`. Build
green at **1322 jobs** (1302 at round 21), zero sorries, zero warnings in owned files, no
`native_decide` / `ofReduceBool` anywhere. Axiom audit run over every new
theorem: the standard three or fewer; **`Machine11.qasm`, `Machine13.qasm`,
`Ladder.criterion_arith`, and every `DepthSum` kernel fact depend on NO AXIOMS AT
ALL**, and `Machine17.qsliceAll` (85,085 tuples) on `[propext, Quot.sound]`.
All census inputs verified over full periods in Python BEFORE formalising
(scratchpad ladder_verify.py, predsim2.py, m23_verify.py, depthsum13.py - the
kernel predicates themselves were simulated exhaustively, zero failures). Every
job the round launched finished before this write-up.

### 26. THE (D) LADDER - four consecutive steps, hypothesis-free

`proofs/Ladder.lean`, on three new machine developments. Round 21 established
that the per-step recipe is mechanical; this round ran it on the three steps
BELOW 19->23, so the ladder is contiguous from the bottom of the machine
sequence. Every conjunct is a theorem about that machine's OWN gap sequence with
no hypotheses at all.

```lean
theorem D_at_11_13 (n : N) : Machine13.g13 n <= 7 + 13
theorem D_at_13_17 (n : N) : Machine17.g17 n <= 11 + 17
theorem D_at_17_19 (n : N) : Machine19.g19 n <= 18 + 19
theorem D_ladder :
    (forall n, Machine13.g13 n <= 7 + 13) and (forall n, Machine17.g17 n <= 11 + 17) and
      (forall n, Machine19.g19 n <= 18 + 19) and (forall n, Machine23.g23 n <= 25 + 23)
-- R39's own form, per rung (the value the criterion actually produces):
theorem g13_le       (n : N) : Machine13.g13 n <= 20   -- max(F2=11, maxQ=20)
theorem g17_le       (n : N) : Machine17.g17 n <= 26   -- max(F2=16, maxQ=26)
theorem g19_le_of_17 (n : N) : Machine19.g19 n <= 35   -- max(F2=25, maxQ=35)
```

    step     criterion max(F2, max_j Q_j)   budget F+q'   margin   floor 2u'
    11->13   max(11, 20) = 20               20             0 TIGHT   4
    13->17   max(16, 26) = 26               28             2         6
    17->19   max(25, 35) = 35               37             2         6
    19->23   max(31, 47) = 47               48             1         8

The 11->13 rung is EXACTLY tight: `Q_5(11; 4) = 20 = F(11) + 13`. Note
`g19_le_of_17` derives a machine-19 gap bound from machine 17's period scan
ALONE (35, vs the sharp 25 that machine 19's own scan gives) - the point of a
rung is that the merge law reaches the next machine without seeing it.

**`MergeLaw.newgap_le_step`** is the new load-bearing lemma - the per-step
bookkeeping factored out once, so a rung is now a 15-line instantiation:

```lean
theorem pos_le_add (hmono : forall m, pos m <= pos (m+1)) (a j) : pos a <= pos (a + j)
theorem windowSum_telescope (hg) (hmono) (a j) :
    Spectrum.windowSum g a j = pos (a + j) - pos a
theorem newgap_le_step {ExO ExN Kap : N -> Prop} {posO posN g : N -> N}
    (hg) (hOmono) (hOpos) (hOex) (hOsurj)          -- old machine's enumeration
    (hNpos) (hNmono) (hNex) (hNempty)              -- new machine's enumeration
    (hnk) (hkn) (hsub)                             -- the step relation
    (hteeth : forall x, Kap x -> x % q = u or x % q = q - u)
    (hu : 0 < u) (h4u : 4 * u <= q)
    (hF2 : Spectrum.SpectrumBound g 2 F2) (hF2B : F2 <= B)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g u j (Q j)) (hQB : forall j, Q j <= B)
    (n : N) : posN (n + 1) - posN n <= B
```

Above the scannable range, hypothesis-explicit instantiations with the census
values named in the statement (so exactly what is assumed is visible):

```lean
theorem D_at_23_29 ... (hteeth : forall x, Kap x -> x % 29 = 5 or x % 29 = 24)
    (hF2 : Spectrum.SpectrumBound g 2 39)                   -- F_2(23) = 39
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g 5 j (Q j)) (hQm : forall j, Q j <= 60)
    (n : N) : posN (n + 1) - posN n <= 34 + 29
theorem D_at_37_41 ... (hteeth : forall x, Kap x -> x % 41 = 7 or x % 41 = 34)
    (hF2 : Spectrum.SpectrumBound g 2 90)                   -- F_2(37) = 90
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g 7 j (Q j)) (hQm : forall j, Q j <= 91)
    (n : N) : posN (n + 1) - posN n <= 88 + 41
theorem criterion_arith : max 39 60 <= 34 + 29 and max 90 91 <= 88 + 41  -- no axioms
```

INDEPENDENT CONFIRMATION OF THE CORRECTED C13 ROW (the brief's critical input).
I re-derived machine 23's spectra myself over the full 37,182,145-slot period
before writing `D_at_23_29`: **F(23) = 34, F_2(23) = 39, Q_j(23; 10) = 43, 50,
55, 60, 0 for j = 3..7, longest run of gaps >= 10 is 4**. Mechanic's corrected
row 43/50/55/60/0 reproduces EXACTLY; the pre-2026-08-24 row 50/50/49/0/0 is
confirmed wrong. Criterion 60 <= 63, margin 3.

### 27. Machine11 / Machine13Q / Machine17Q - three new machines certified

Each is the round-21 `seekT`-walk recipe at a new machine: one walk per opening
reads the whole ladder plus the depth refutation, `seek_next` proves the walk
computes `nextOp` exactly, `opSeq_surj` makes the enumeration complete. Each
scan's FIRST check is `o1 <= F`, which re-derives `F_1` from the same walk and
is what makes the fuel provably sufficient (round 21 imported that fact from a
separate scan; folding it in removes the dependency).

```lean
-- Machine11 (gears {5,7,11}, period 385; one 385-tuple kernel check, NO AXIOMS)
theorem qasm : qslice = true
theorem chain_facts (n) : opSeq (n+1) - opSeq n <= 7 and opSeq (n+2) - opSeq n <= 11
  and opSeq (n+3) - opSeq n <= 16 and opSeq (n+4) - opSeq n <= 18
  and ((4 <= g11 (n+1) and 4 <= g11 (n+2) and 4 <= g11 (n+3)) -> opSeq (n+5) - opSeq n <= 20)
  and not (4 <= g11 n and 4 <= g11 (n+1) and 4 <= g11 (n+2) and 4 <= g11 (n+3))
theorem spectrum_ladder : F_1..F_4 <= 7, 11, 16, 18   -- over g11
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g11 2 j 20
theorem opSeq_surj (hm : 1 <= m) (hE : Exposed11 m) : exists n, opSeq n = m

-- Machine13Q (period 5005; one 5005-tuple kernel check, NO AXIOMS)
theorem chain_facts (n) : opSeq (n+1) - opSeq n <= 11 and opSeq (n+3) - opSeq n <= 23
  and opSeq (n+4) - opSeq n <= 26 and not (6 <= g13 n and 6 <= g13 (n+1) and 6 <= g13 (n+2))
theorem spectrum_ladder : F_1..F_4 <= 11, 16, 23, 26  -- F_2 = round-11 certificate
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g13 3 j 26
theorem opSeq_surj ...

-- Machine17Q (period 85085; 17 slices of 5005, [propext, Quot.sound] only)
theorem qsliceAll : forall e < 17, qslice e = true
theorem chain_facts (n) : ... F_1 <= 18, F_3 <= 28, F_4 <= 33, F_5 <= 35,
  the qualifying depth-6 bound Q_6 <= 34, and no five consecutive gaps all >= 6
theorem spectrum_ladder : F_1..F_5 <= 18, 25, 28, 33, 35
theorem qual_bound_all : forall j, 3 <= j -> Spectrum.QualBound g17 3 j 35
theorem opSeq_surj ...
```

WHERE THE QUALIFYING RESTRICTION EARNS ITS KEEP, measured per machine: at
machine 13 the unconditional ladder already clears the budget at both live
depths (F_3 = 23, F_4 = 26 <= 28) and the qualifying structure only kills
j >= 5; at machine 11 it first bites at depth 5 (F_5 = 23 > 20, Q_5 = 20); at
machine 17 at depth 6 (F_6 = 40 > 37, Q_6 = 34). So the criterion is NOT a
uniform improvement - it is a one-or-two-depth patch on the unconditional
spectrum, and the depth it patches moves UP with the machine (5, -, 5, 6, then
19->23's depth 5-6 pattern). Full-period ladders behind these (all newly
measured this round):

    machine   F_1..F_8                             Q_j(floor)              max run
    11        7, 11, 16, 18, 23, 26, 28, 30        16, 18, 20, 0 (fl 4)    3
    13        11, 16, 23, 26, 28, 31, 34, 38       18, 23, 0     (fl 6)    2
    17        18, 25, 28, 33, 35, 40, 43, 48       28, 31, 32, 34, 0 (6)   4
    23        34, 39, 50, 58, 65, 77, 83, 88       43, 50, 55, 60, 0 (10)  4

Scan cost, measured this round (with ~7 python jobs of another lane running):
machine 11, 385 tuples, seconds; machine 13, 5005 tuples with a 4-step walk at
fuel 11, **37 s**; machine 17, 17 slices of 5005 with a 6-step walk at fuel 18,
**213 s for 6 slices** (~35 s/slice), 17 slices in three parallel files, ~10 min
wall. Assemblies 19-26 s each.

### 28. proofs/DepthSum.lean - Lateral's depth-sum identity at machine 13

Both halves of `sum_j W_j(g) = prod_q c_q(g)`, kernel-checked; the glue is not
(honest gap, below).

```lean
theorem window_depth_unique (hg) (hmono : forall m, pos m < pos (m+1))
    (h1 : Spectrum.windowSum g a j1 = gap) (h2 : Spectrum.windowSum g a j2 = gap) :
    j1 = j2
def reachSet (g) (gap N J) : Finset N
theorem mem_reachSet : a in reachSet g gap N J <->
    a < N and exists j, 1 <= j and j < J and Spectrum.windowSum g a j = gap
theorem depth_partition (hg) (hmono) (gap N J) :
    sum over j in Finset.Ico 1 J of
        ((Finset.range N).filter fun a => Spectrum.windowSum g a j = gap).card
      = (reachSet g gap N J).card
-- the RHS at machine 13, over the whole period (all of these: NO AXIOMS)
theorem local_factor_5  : forall gap < 40, cq 5 1 4 gap   = 5 - nuq 5 gap
theorem local_factor_7  : forall gap < 40, cq 7 6 1 gap   = 7 - nuq 7 gap
theorem local_factor_11 : forall gap < 40, cq 11 2 9 gap  = 11 - nuq 11 gap
theorem local_factor_13 : forall gap < 40, cq 13 11 2 gap = 13 - nuq 13 gap
theorem depth_sum_at_13 : forall gap < 40,
    pairCount13 gap = cq 5 1 4 gap * cq 7 6 1 gap * cq 11 2 9 gap * cq 13 11 2 gap
theorem depth_sum_hl_form : forall gap < 40, pairCount13 gap
    = (5 - nuq 5 gap) * (7 - nuq 7 gap) * (11 - nuq 11 gap) * (13 - nuq 13 gap)
```

`window_depth_unique` IS Lateral's one-line bijection ("every opening pair at
lag g is the endpoint pair of exactly one window") in its load-bearing form, and
it is abstract - no machine, no arithmetic, just strict monotonicity.
`local_factor_*` is Harvester's identity `c_q(g) = q - nu_q({0, 2, 6g, 6g+2})`,
the one they listed as a kernel candidate: the machine's transfer diagonal IS
the Hardy-Littlewood prime-quadruplet local factor, now checked at four gears.
`depth_sum_hl_form` states the machine-13 pair population directly in HL form.
Verified first over the full period for g = 0..59 (depthsum13.py, zero
mismatches on both halves).

### 29. proofs/Potential.lean + Potential19.lean - (D) WITH NO DEPTH QUANTIFIER

Picked up mid-round from the coordinator's cross-lane routing of Constructor's
R46 (docs/novel/kleene-generator.md): their Kleene generator
`F(M+q') = L^T (x) K* (x) R` has as its corollary that (D) holds IFF a POTENTIAL
`h` exists satisfying three ONE-STEP, ONE-OPENING inequalities. That is the
first form of (D) that is not an infinite family indexed by depth, so it is the
better formal target, and I took it without deferring anything else.

KERNEL-CHECKED: the direction that does proof work - a potential CERTIFIES the
bound, at every chain length, by one induction.

```lean
def IsPotential {St : Type*} (Step : St -> St -> Prop) (d e h : St -> N) (B : N) : Prop :=
  (forall x, d x <= h x) and (forall x y, Step x y -> d x + h y <= h x)
    and (forall x, e x + h x <= B)
theorem chain_le_potential (hC1) (hC2) :
    forall (l : N) (p : N -> St), (forall k, k < l -> Step (p k) (p (k+1))) ->
      sum over k in Finset.range (l+1) of d (p k) <= h (p 0)
theorem D_of_potential (hP : IsPotential Step d e h B) (l) (p) (hstep) :
    e (p 0) + sum over k in Finset.range (l+1) of d (p k) <= B
theorem windowSum_succ_left (g b l) :
    Spectrum.windowSum g b (l+1) = g b + Spectrum.windowSum g (b+1) l
theorem tail_le_potential (hC1) (hC2) : forall l b, (forall i < l, 2*u <= g (b+i)) ->
    Spectrum.windowSum g b l + g (b+l) <= h b
theorem merged_le_of_potential {g h u F q}
    (hC1 : forall i, g i <= h i)
    (hC2 : forall i, 2*u <= g i -> g i + h (i+1) <= h i)
    (hC3 : forall i, g i + h (i+1) <= F + q)
    {a l} (hw : forall i < l, 2*u <= g (a+1+i)) :
    g a + Spectrum.windowSum g (a+1) l + g (a+l+1) <= F + q
```

`merged_le_of_potential`'s hypotheses contain NO quantifier over `l`; its
conclusion holds for every `l`. The abstract form keeps the state type general
because Constructor's states are `(opening, tooth)` pairs, not indices.

AND THE FIRST EXHIBITED CERTIFICATE (`Potential19.lean`) - so the potential form
is not just a definition:

```lean
def h19 (i : N) : N :=            -- the qualifying tail from i
  if 8 <= g19 i then
    (if 8 <= g19 (i+1) then
      (if 8 <= g19 (i+2) then g19 i + g19 (i+1) + g19 (i+2) + g19 (i+3)
        else g19 i + g19 (i+1) + g19 (i+2))
      else g19 i + g19 (i+1))
    else g19 i
theorem h19_C1 (i) : g19 i <= h19 i
theorem h19_C2 (i) (hq : 8 <= g19 i) : g19 i + h19 (i+1) <= h19 i
theorem h19_C3 (i) : g19 i + h19 (i+1) <= 25 + 23
theorem D_of_word_potential {a l} (hw : forall i < l, 8 <= g19 (a+1+i)) :
    g19 a + Spectrum.windowSum g19 (a+1) l + g19 (a+l+1) <= 25 + 23
```

The three clauses are discharged by machine 19's kernel ladder ALONE: (C1) is
syntactic; (C2) holds with EQUALITY in every branch, and its deepest branch is
exactly `Machine19.no_big_run` (`Q_6 = 0` - three floor gaps in a row force the
fourth not to qualify, so the tail terminates); (C3)'s four cases are precisely
the four rungs `F_2, F_3, F_4, F_5 <= 31, 35, 38, 47`, all under the budget 48.
So `D_of_word_potential` re-proves `Machine19.D_of_word` through a finite object
one can write down, with no depth analysis in the hypotheses.

THE RECIPE GENERALISES, and I state it because it is what a future rung reuses:
at any machine whose qualifying runs are bounded - which is exactly what
`Q_J = 0` says, and every machine scanned so far has such a `J` (11: J=6,
13: J=5, 17: J=7, 19: J=6, 23: J=7) - the TAIL FUNCTION unfolded to depth `J-2`
IS a potential, (C2)'s deepest branch is the `Q_J = 0` refutation, and (C3)'s
cases are the machine's own spectrum ladder. What is NOT known, and what
Constructor's negative at 29->31 is about, is a potential valid at every machine
at once.

### Honest "will not close" verdicts (round 22)

9. **23->29 CANNOT BE MADE HYPOTHESIS-FREE BY THIS ROUTE, and the reason is
   structural, not budgetary: THE MERGE LAW IS ONE-STEP.** R39 consumes an
   `F_2` and a qualifying spectrum of the OLD machine and produces a bound on
   the NEW machine's single gaps - which is not of the form the next rung
   needs. Quantified at the live step: R39 gives `g23 <= 47`, so the best
   merge-law-only bound on `F_2(23)` is `2 * 47 = 94`, against the `<= 63` the
   23->29 rung requires (true value 39). Chaining the depth-j bounds is worse,
   not better: a depth-j window of machine 23 relaxes to a machine-19 window
   with `j - 1` unconstrained interior points, and the loss compounds linearly
   in j (three qualifying blocks of machine 19 admit `47 + 10 + 47 = 104`
   against the true `Q_3(23; 10) = 43`). This is Constructor's counting
   boundary (R41) in its formal-lane form: **no function of the old machine's
   marginal data supplies the next rung's input; each rung needs its own scan.**
   So 23->29 needs machine 23's own period scan: 37,182,145 CRT tuples = 7,434
   slices of 5005, and at this round's measured 35 s/slice for a 5-gear 6-step
   walk (machine 23 needs 7 gears and a 7-step walk at fuel 34, ~2.6x the
   per-slice work) that is **~150-200 hours of kernel time** - a multi-day job,
   not a round-scale one. Deliberately not started (job-completion rule).
   THE CONSTRUCT THAT WOULD REMOVE IT, named: a MARKED QUALIFYING SPECTRUM of
   the old machine - `Q^[j]` = max window sum over old-machine windows carrying
   `j - 1` MARKED interior openings at mutual distance `>= 2u''`, all unmarked
   interiors killed. `Q_j(new) <= Q^[j](old)` by construction, and `Q^[j]` is
   scannable at the OLD machine, so it would make the ladder chainable from one
   scan. Measured obstruction: the relaxation forgets WHICH openings the new
   gear kills, and the estimates above say the loss already exceeds the budget
   at `j = 2`. Worth one census (Mechanic or Constructor) before anyone
   formalises it.
10. **A rung's bound is not the machine's F.** Every rung produces a true but
    untight bound (20 vs F(13) = 11; 26 vs 18; 35 vs 25; 47 vs 34). (D) at
    alpha = 3 is all that is claimed, and all that the route needs; anyone
    wanting exact F still needs the machine's own certificate.
11. **The depth-sum identity's glue was not built.** `depth_partition` counts
    window STARTS in an index range; `depth_sum_at_13` counts openings in a
    SLOT range. Relating them needs "one period of `Machine13.opSeq` = one
    period of residues", i.e. a periodicity / re-indexing bridge (`opSeq (n +
    1485) = opSeq n + 5005` for the 1485 openings per period). Routine but real;
    it was not affordable alongside the ladder and is named rather than
    half-done.

12. **The Kleene identity itself was NOT formalised, nor the converse of the
    potential form.** `F(M+q') = L^T (x) K* (x) R` is an EQUALITY and needs
    max-plus matrix machinery plus the machine's own `K`; the converse ("a
    potential always exists") is where nilpotency of `K` is used, `h` being the
    least super-solution (a max over tails), which needs the finite path bound
    as a Finset construction. Only the certificate direction is claimed - it is
    the one a proof consumes, and `Potential19` shows it is not vacuous. Also
    recorded verbatim from Constructor so nobody reads more into these files
    than is there: the generator is arity-free but NOT YET machine-free -
    bounded-state certificates certify 19->23 (45 <= 48) and FAIL at 29->31
    (99 / 99 / 91 against a budget of 74). The files make the target statement
    precise; they do not prove (D).

### Infrastructure lessons (round 22)

- **State gap facts as `opSeq` differences, never as walk offsets.** `have g3 :
  g11 (n+3) = o4 - o3 := by simp only [g11]; omega` FAILS: `simp only` unfolds
  `g11 (n+3)` to `opSeq (n+3+1) - opSeq (n+3)` and omega sees `opSeq (n+3+1)`
  as an atom distinct from the `opSeq (n+4)` of the chain equations. The
  round-21 shape `have g3 : g11 (n+3) = opSeq (n+4) - opSeq (n+3) := by simp
  only [g11]` works because simp's index normalisation closes it by rfl and
  leaves omega only atoms it already has. This will bite any future rung.
- **`rcases h with a | b; . tac1; . tac2` inside a term-mode `by` is wrong** -
  `;` sequences over ALL goals, so the focusing bullets misfire ("No goals to
  be solved" plus "unsolved goals" on the same line). Use `<;>` with a tactic
  that closes both branches (`rcases hk with h | h <;> omega`).
- **Component projections into a 12-conjunct `Exposed` are position-sensitive,
  and the kernel catches swaps immediately** - gear q's `lo` tooth and `hi`
  tooth are DIFFERENT residues (19: `lo` at `k % 19 = 16`, `hi` at 3), so
  `Or.inl`/`Or.inr` must match the order in `killed_iff`. Two swapped pairs were
  caught by application-type-mismatch, not by a false theorem.
- **`seek_next` needs `hnE : Exposed (nextOp (x+s))` in scope** - the trailing
  `rwa` closes with `assumption`. Dropping that one line from the round-21
  template cost two rebuild cycles.
- **The mega-dry-check was NOT used this round and should have been**: three of
  the four failures above would have been caught at zero kernel cost. The scans
  at these machines are cheap enough (37-213 s) that the discipline slipped; at
  machine-19 scale it would not have.
- **Scan-first-check trick, general**: make the FIRST clause of a chain scan
  `Nat.ble o1 F`. It re-derives `F_1 <= F` from the same walk, which is exactly
  the fuel-sufficiency fact `seek_next` needs - so a new machine's chain scan is
  self-contained and imports no bound from an earlier scan.
- Slice families: 6 heavy `decide +kernel` per file, three files built in one
  `lake build` invocation, stayed inside the memory rule with no failures.

### Open formalisation targets (re-prioritised after round 22)

1. **A potential at 17->19 and at 13->17 by the recipe of section 29**: each is
   ~60 lines now that `Potential.merged_le_of_potential` exists and each machine
   has both its `Q_J = 0` refutation and its ladder. Cheapest way to make the
   depth-quantifier-free form of (D) the PRIMARY statement of every rung.
2. **The depth-sum glue at m13** (verdict 11): `opSeq (n + 1485) = opSeq n +
   5005` and the re-indexing it enables. Finishes the identity at one machine.
3. **The marked qualifying spectrum** (verdict 9): census FIRST (does
   `Q^[2](19) <= 63`?), formalise only if the numbers survive. The only named
   route that makes the ladder chainable without a scan per rung.
4. **Machine 23's period scan** (37.2M tuples, ~150-200 h): the brute-force
   route to the 23->29 rung. Only worth starting as a deliberately-scoped
   multi-round job, and only if 3 fails.
5. **Harvester's paired-Holt coef rung** (5005 -> 85085), unchanged from
   round 21.

## Round 23

Ledger: +6 libs (`Machine23QCore`, `Machine23Q`, `Machine29`, `CoveringCert`, `CoveringCert2`,
`PotentialLadder`), all registered with `[[lean_lib]]` and in `defaultTargets`.
Build green at **1334 jobs** (1322 at round 22; 1332 before the post-routing section 34), zero sorries, zero warnings in
owned files, no `native_decide` / `ofReduceBool` anywhere. Axiom audit run over
every new theorem: the standard three or fewer;
**`CoveringCert.cert_signs` depends on NO AXIOMS** and
`Machine29.merge_alphabet` on `[propext, Quot.sound]`. Everything was verified
over full periods in Python BEFORE formalising (scratchpad m23_qspec.py,
mirror23.py, gencert.py, marked_check.py, marked_check2.py, marked29.py). Every
job the round launched finished before this write-up.

Briefed items 2 and 3 landed in full. Item 1 (the 23->29 rung) did NOT close,
and the reason is a MEASURED kernel-cost fact, not a mathematical one. The
round's most consequential output is a correction to the census item 1 was
built on, which turned out to be reached independently in two other lanes the
same round.

LABEL, corrected (Constructor R49, adopted): the step everyone had been calling
"23->29" in the failure discussion is 29->31 (budget `F(29) + 31 = 74`); both
objects are indexed by their OLD machine, 29 for Constructor's abstraction and
23 for the marked spectrum, so one step carried two names. **The 23->29 rung
(budget `F(23) + 29 = 63`) was never in doubt** - it is the rung sections 30-31
below are about, and it is the one my `Machine29.lean` states.

### 30. THE MARKED QUALIFYING SPECTRUM: THE PUBLISHED NUMBERS ARE INFLATED, THE
### DISCREPANCY IS ONE LINE OF A DP, AND ONE VERDICT REVERSES

My brief said to verify Mechanic's `Q_j(new) <= Q^[j](old)` myself at the steps
where both sides are known before building on it. I did, from their written
DEFINITION rather than their code, and the re-derivation disagrees with their
published table.

    step (floor)     J:      2     3     4     5     6     7
    11->13 (a=6)  Q_J(13)   16    18    23     0     0     0    exact
                  published 16    23    23     0     0     0
                  corrected 16    18    23     0     0     0
    13->17 (a=6)  Q_J(17)   25    28    31    32    34     0    exact
                  published 25    28    32    33     -     -
                  corrected 25    28    31    32    34     0
    17->19 (a=8)  Q_J(19)   31    35    37    38     0     0    exact
                  published 31    35    38    38     -     -
                  corrected 31    35    37    38     0     0
    19->23 (a=10) Q_J(23)   39    43    50    55    60     0    exact
                  published 39    50    50    55    60     0
                  corrected 39    43    50    55    60     0
    29->31 (a=10) Q_J(29)   55    65    68    71    71    71    (Mechanic's exact)
                  published 55    65    68    85    73    73
                  corrected 55    65    68    71    71    71

**THE CORRECTED MARKED SPECTRUM EQUALS THE EXACT `Q_J(new)` IN ALL 30 ENTRIES OF
ALL FIVE STEPS.** The construct is not merely a tight relaxation; here it is
exact, entrywise.

THE DISCREPANCY IS LOCATED EXACTLY, and it is one line of a dynamic program.
`research/marked_qspec.py`'s feasibility search places `J-1` marks and returns
success the moment the count is reached; it never checks that the interiors
AFTER the last mark are killed. So it accepts windows whose tail contains an
opening that is neither marked nor killed - a configuration the definition
forbids. Re-running MY code with that one check disabled REPRODUCES THE
PUBLISHED ROW DIGIT FOR DIGIT at every step (16/23/23/0, 25/28/32/33,
31/35/38/38, 39/50/50/55/60/0). That is the proof of diagnosis; nothing else
differs between the two implementations.

CONSEQUENCES:
- The headline verdict at 19->23 STANDS and was conservative: published
  `max_J Q^[J](19) = 60 <= 63`, corrected also 60 (the error bites at J = 3, not
  at the maximum). The 23->29 rung's arithmetic is unaffected.
- **The verdict at 29->31 REVERSES.** Published `Q^[J](23) = 55 65 68 85 73 73`,
  `max = 85` against budget 74, concluding "the construct buys EXACTLY ONE
  RUNG". My corrected recomputation over machine 23's full 37,182,145-slot
  period (7,952,175 openings, 191 s) gives **`55 65 68 71 71 71`, `max = 71 <=
  74` - the rung is NOT lost by this route**, and the J = 5 entry that carried
  the whole verdict was the DP artefact.

TRIPLE-SOURCED, and I am recording the convergence rather than claiming
priority: Mechanic retracted the row in their own early post, Constructor
audited it from the opposite direction ("a sound relaxation CANNOT report 85")
and proved the SANDWICH LEMMA that makes the equality forced -
`Q_J(new) <= Q^[J](old) <= max_{j<=J} Q_j(new)`, hence
`max_J Q^[J](old) = max_J Q_J(new)` always - and I re-derived the numbers from
the definition and identified the offending DP line. Three lanes, three methods,
one answer. What my half adds that the others do not have is the exact
reproduction of the published rows from the disabled check, which is what turns
"their numbers are wrong" into "their numbers are THIS bug".

FORMALISATION NOTE. The `<=` half of the sandwich is a one-line relaxation
argument. The `>=` half is the content, and it is what makes the construct worth
formalising at all: with it, the marked spectrum supplies EVERY rung, not one.
See target 2 below.

### 31. THE 23->29 RUNG: EVERYTHING BUT TWO DECIDABLE FACTS

`proofs/Machine23Q.lean` + `proofs/Machine29.lean`. Round 22 recorded
`Ladder.D_at_23_29` as an instantiation over an ABSTRACT pair of machines. This
round it becomes a statement about two CONCRETE machines with exactly two named
hypotheses, both decidable facts about machine 23's own gap word:

```lean
-- Machine23Q.lean: machine 23's enumeration is complete (no new scan)
theorem opSeq23_strict_mono {a b : N} (h : a < b) : opSeq23 a < opSeq23 b
theorem windowSum_g23 (a j : N) :
    Spectrum.windowSum g23 a j = opSeq23 (a + j) - opSeq23 a
theorem opSeq23_surj {m : N} (hm : 1 <= m) (hE : Exposed23 m) : exists n, opSeq23 n = m

-- Machine29.lean: gear 29's teeth are {5, 24}  (6 * 5 = 30 = 29 + 1)
def Killed29 (k : N) : Prop := k % 29 = 5 or k % 29 = 24
def Exposed29 (k : N) : Prop :=
  Exposed23 k and not (29 | Census.lo k) and not (29 | Census.hi k)
def g29 (n : N) : N := opSeq29 (n + 1) - opSeq29 n
theorem killed29_iff {k : N} (hk : 1 <= k) :
    Killed29 k <-> (29 | Census.lo k or 29 | Census.hi k)
theorem merge_alphabet {x y : N} (hk1 : Killed29 x) (hk2 : Killed29 y)
    (hxy : x < y) (hle : y - x <= 34) : y - x = 10 or y - x = 19 or y - x = 29
theorem g29_le (hF2 : Spectrum.SpectrumBound g23 2 39)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g23 5 j 60) (n : N) : g29 n <= 60
theorem D_at_23_29 (hF2 : Spectrum.SpectrumBound g23 2 39)
    (hQ : forall j, 3 <= j -> Spectrum.QualBound g23 5 j 60) (n : N) : g29 n <= 34 + 29
```

Everything else is discharged: machine 29's own opening enumeration
(`exists_exposed29_above` at period `1,078,282,205 = 37,182,145 * 29`,
`nextOp29`, `opSeq29`, `opSeq29_gap_empty`), the teeth, the containment
`Exposed29 -> Exposed23`, machine 23's enumeration completeness, and the
merge-law wiring. So the rung is reduced from "an abstract instantiation" to
"`F_2(23) <= 39` and `Q_j(23; 10) <= 60`", both decidable.

Verified this round over the FULL machine-23 period, independently of round 22's
run (scratchpad m23_qspec.py): `F_1..F_8 = 34, 39, 50, 58, 65, 77, 83, 88`;
`Q_j(23; 10) = 39, 43, 50, 55, 60, 0` for `j = 2..7`; longest run of gaps `>= 10`
is 4. Criterion `max(39, 60) = 60`, budget 63, margin 3. `merge_alphabet` is the
concrete content at this step: the 23->29 merge letters are exactly
`{10, 19, 29}`, all at or above the floor `2u'' = 10`.

### 32. proofs/CoveringCert.lean - A (D) RUNG PROVED A SECOND WAY, SCAN-FREE

The briefed item 2, landed whole: `F(19) <= 37 = F(17) + 19` from THIRTY-SEVEN
INTEGERS, with no period of machine 19 built and nothing shared with the
merge-law route.

```lean
def ywList : List N := [115, 169, 265, ..., 265, 169, 115]     -- 37 weights
def totY : N := sum over i in Finset.range 37 of yw i
def S (bq : N -> N -> Bool) (r : N) : N                        -- single-gear sum
def PP (bq : N -> N -> Bool) (r5 rq : N) : N                   -- pair sum
theorem tot_eq : totY = 9757                                   -- decide +kernel
theorem S5_le  : forall r < 5,  S b5 r <= 3905                 -- six of these
theorem S19_le : forall r < 19, S b19 r <= 1115
theorem P7_ge  : forall a < 5, forall b < 7, 1101 <= PP b7 a b -- five of these
theorem P19_ge : forall a < 5, forall b < 19, 272 <= PP b19 a b
theorem cert_signs :
    3905+2796+1821+1648+1204+1115 < 9757 + (1101+552+548+276+272)  -- NO AXIOMS
theorem kounias (a b c d e f : Bool) (h : (a || b || c || d || e || f) = true)
    (w : N) : w + (the five (if a && x then w else 0))
                <= (the six (if x then w else 0))
theorem cover_bound (hcov : forall i < 37, the six-way or at i) :
    totY + (PP b7 r5 r7 + ... + PP b19 r5 r19)
      <= S b5 r5 + S b7 r7 + S b11 r11 + S b13 r13 + S b17 r17 + S b19 r19
theorem no_cover (h5 : r5 < 5) ... (hcov) : False
theorem no_37_run {p : N} (hp : 1 <= p) : exists i < 37, Machine19.Exposed19 (p + i)
theorem F19_le_37 (n : N) : Machine19.g19 n <= 37
theorem D_17_19_lp (n : N) : Machine19.g19 n <= 18 + 19
```

`no_37_run` and `F19_le_37` depend on NOTHING except the certificate arithmetic
and `Machine19.exposed19_iff` (the definition of an opening as a CRT tuple): no
slice, no `sliceAll`, no `qsliceAll`, no merge law, no `Spectrum`. `D_17_19_lp`
additionally reads the budget `F(17) = 18` off `Machine17`, exactly as the
merge-law route does. So the 17->19 rung of the (D) ladder now has TWO kernel
proofs whose only common ancestor is the definition of the machine.

THREE FACTS ABOUT THE CERTIFICATE that only appeared on formalising it (all
recorded in docs/novel/covering-lp-certificates.md, whose status I upgraded):
- **It is supported on ONE distinguished gear.** All 37 nonzero dual weights sit
  on rows `(i, 5)`; the Kounias cut is used with `k = 5` at every position and
  never with any other. The 222-row, 7-pair LP optimum uses 37 rows and 5 pairs
  (`(7,11)` and `(7,13)` pass the visibility test and then get weight 0). That
  is what makes the Lean statement small - six maxima and five minima, and no
  sum over gears anywhere in the file.
- **It is a palindrome**, `y_i = y_{36-i}` exactly - the machine's mirror
  symmetry `k -> -k` appearing in the dual.
- Scaled to integers (common denominator 1101) it reads `12489 < 9757 + 2749`,
  margin 17 out of 12489 = 0.14%.

Kernel cost: 11 `decide +kernel` declarations, seconds - against a 1,616,615-slot
period scan that costs hours. `F(19) <= 37` is WEAKER than `Machine19.gap_le`'s
exact 25; the point is the method, and the method is the only upper bound on a
Jacobsthal-type maximal gap in this development that a kernel checks without
enumerating a period.

### 33. proofs/PotentialLadder.lean - THE DEPTH-QUANTIFIER-FREE FORM AT EVERY
### SCANNED RUNG

Round 22's own top open target, and the brief's item 3 in the form that
discharges rungs rather than restating the target. The recipe of round 22's
section 29 run at the three rungs below 19->23:

```lean
def h11 (i : N) : N   -- machine 11's qualifying tail, floor 4, unfolds <= 4
def h13 (i : N) : N   -- machine 13's, floor 6, unfolds <= 3
def h17 (i : N) : N   -- machine 17's, floor 6, unfolds <= 5
theorem h11_C1 / h11_C2 / h11_C3, h13_C1/C2/C3, h17_C1/C2/C3
theorem D_of_word_11 {a l : N} (hw : forall i < l, 4 <= Machine11.g11 (a+1+i)) :
    Machine11.g11 a + Spectrum.windowSum Machine11.g11 (a+1) l
      + Machine11.g11 (a+l+1) <= 7 + 13
theorem D_of_word_13 ... <= 11 + 17
theorem D_of_word_17 ... <= 18 + 19
theorem potential_ladder : the three, collected
```

    rung      potential   floor 2u'   tail depth   budget F + q'
    11 -> 13  h11             4           4          7 + 13 = 20
    13 -> 17  h13             6           3         11 + 17 = 28
    17 -> 19  h17             6           5         18 + 19 = 37
    19 -> 23  h19 (r22)       8           4         25 + 23 = 48

(C2) holds with EQUALITY in every branch at every machine, and its deepest
branch is always exactly that machine's `no_big_run`; (C3)'s cases are always
that machine's own spectrum ladder, with the DEEPEST case at machines 11 and 17
supplied by the CONDITIONAL rung of `chain_facts` (`Q_5(11) <= 20`,
`Q_6(17) <= 34`) rather than by an unconditional `F_j` - which is exactly where
the qualifying restriction earns its keep, and it lines up with round 22's
per-machine measurement of which depth it patches. THE TAIL DEPTHS DO NOT GROW
WITH THE MACHINE: 4, 3, 5, 4. Each is a separate finite object; a potential
valid at every machine at once is still not known.

### Honest "will not close" verdicts (round 23)

13. **THE 23->29 RUNG'S SCAN IS NOT ROUND-SCALE ON THIS MACHINE, AND THE REASON
    IS THAT A LEAN KERNEL CANNOT SHARE A WALK ACROSS A PHASE LOOP.** Round 22
    priced the rung at a 7,434-slice machine-23 period scan (~150-200 h). I
    found a better factorisation; it is still too slow, for a reason worth
    recording because it will recur at every future rung.
    THE FACTORISATION (`proofs/Machine23QCore.lean`, kept in the ledger as the
    encoding): machine 23's period is `1,616,615 * 23`, so scan the 323 slices
    machine 19 ALREADY uses, each with an inner 23-fold loop over the gear-23
    PHASE `g = k % 23`. This is EXACT - not the marked relaxation - because for
    a fixed machine-19 tuple the phase is the only remaining freedom. One walk
    per (tuple, phase) reads `F_1 <= 34`, `F_2 <= 39`, four guarded qualifying
    rungs `Q_3..Q_6 <= 60`, and the five-run refutation; the guards are Bool
    `&&` / `||`, which the kernel evaluates LAZILY, so a tuple whose second gap
    is below the floor never walks past `o2`. Simulated exhaustively first: the
    `chain23` Bool is true at all 7,952,175 machine-23 openings (mirror23.py,
    two independent implementations, one numpy over the whole period and one a
    literal transcription of the Lean defs).
    MEASURED COST (mini-slices of 143 tuples = 1/35 of a slice, priority-boosted
    - see the infrastructure note):
        machine-19-style walk, no phase loop (`Machine19.qokT`)   0.7 s
        full `chain23`, ONE phase                                 1.0 s
        full `chain23`, 23 phases                                12   s
    so a real 5005-tuple slice is ~420 s and the 323 slices are **~38 h
    sequential, ~13 h at the 2-3 parallel targets the memory rule allows**. Not
    round-scale, and not honestly startable under the job-completion rule.
    WHY THE PHASE LOOP COSTS ITS FULL 21x - THE TRANSFERABLE FACT: the
    machine-19 walking is IDENTICAL for 21 of the 23 phases (gear 23 kills only
    2 of 23 phases at any one opening), so nearly all the work is repeated, and
    it cannot be shared. I hoisted the phase-free walk out explicitly - replacing
    the 7-residue slot test by `Machine19.seekT`, which does not mention `g` -
    and measured NO improvement (11 s vs 12 s). A control settles why: making
    the loop body `g`-INDEPENDENT so all 23 iterations reduce the SAME closed
    term collapses the cost to 1.35 s. So **the kernel DOES share structurally
    identical subterms; the walks after the first hop are not identical - they
    are indexed by `g` even where they compute the same number** - and a pure
    term-rewriting kernel has no way to say "evaluate once, then branch".
    THE CONSTRUCT THAT WOULD REMOVE IT, named and priced: index the machine-19
    opening chain by POSITION rather than offset (`w19 a b c d e f k` = the k-th
    machine-19 opening after the base, which is `g`-free for literal `k` and
    therefore shared), and let the phase loop only select indices into it.
    Estimated 5x from the measurements above (~7 h sequential, ~2.5 h at
    3-parallel), at the cost of an index-based extraction. Worth doing on an
    UNCONTENDED machine, or by Mechanic; not worth gambling a round on.
    NOTE the corrected marked spectrum does NOT remove this: it replaces the
    phase quantifier by a phase MAXIMUM, which in a functional kernel encoding
    still costs one walk per phase. Mechanic's Python avoids that with an
    incremental mutable coverage array; a kernel term cannot.
14. **`F_2(23) <= 63` cannot be got from `F(23) <= 34` by doubling.** Recorded
    because it is the obvious shortcut and it misses by five: `2 * 34 = 68 > 63`
    (and with round 21's `g23 <= 47` it is 94). The depth-2 fact genuinely needs
    its own scan; there is no free ride from the single-gap bound.
15. **A_4's soundness is not itself kernel-checkable at machine 29, and any
    Lean statement of it must carry that as a hypothesis.** Constructor's A_4
    (state = last three GAP VALUES, phase-free, 14,368 states / 3,513 edges at
    m29, exact at all seven scannable steps) has an edge relation "this
    4-tuple of consecutive gaps is REALISED in the period". Realisability is a
    full-period claim about machine 29 - 1,078,282,205 slots - so the edge set
    cannot be certified in the kernel by any method now in the lane. What CAN
    be: the longest-path value over an EXPLICIT edge set, with "E contains every
    realised 4-tuple" as a named hypothesis the census discharges. That is the
    right shape (it is the shape `Ladder.D_at_23_29` had), and it is target 3.
    Any theorem must not be worded so as to suggest (D) is proved in general:
    A_4 is per-machine, and Constructor's own machine-free result in the same
    report is a NEGATIVE (MF_3 mod 35 = MF_3 mod 385 = MF_4 mod 35 at all seven
    steps, 15/31/47/111/105/125/211 against budgets 20/28/37/48/63/74/95).

### Infrastructure lessons (round 23)

- **PRIORITY BOOSTING IS WORTH 2.3x AND SHOULD BE THE DEFAULT ON THIS MACHINE.**
  Every measurement was taken twice, once at normal priority and once with the
  `lean.exe` children raised to `High` by a PowerShell poll loop running
  alongside the build: same mini-slice, 28 s -> 12 s. The machine ran at 100%
  CPU throughout from ~8 other-lane `python3.12` processes on 14 cores / 20
  threads. Round 21 recorded the same effect; it is still the cheapest speedup
  available and it costs one line:
  `for i in $(seq 1 N); do powershell -NoProfile -Command "Get-Process lean -ErrorAction SilentlyContinue | ForEach-Object { $_.PriorityClass = 'High' }"; sleep 2; done`
  run in parallel with the build and stopped when it exits.
- **THE KERNEL SHARES STRUCTURALLY IDENTICAL SUBTERMS - MEASURE IT, DO NOT
  ASSUME EITHER WAY.** The control in verdict 13 (23 identical iterations cost
  1x, 23 nearly-identical ones cost 21x) is the cheapest diagnostic I have found
  for "is my encoding paying for repetition?", it costs one mini-slice, and it
  decided this round's biggest engineering question. Corollary for encodings:
  what must be shared has to be SYNTACTICALLY free of the loop variable, not
  merely equal in value at every iteration.
- **A DIAGNOSTIC LADDER BEATS A SINGLE PROBE.** Four mini-slice variants
  (calibration / one phase / two clauses / everything) in one file with
  `set_option profiler true` and `profiler.threshold 20` gave per-declaration
  `type checking took ...` lines and located the cost in one 2.5-minute run. The
  round-20 single-canary probe ran 25 minutes and said only "too slow"; I made
  that mistake first and it cost most of an hour.
- **`decide +kernel` handles `Finset.sum` fine.** All eleven `CoveringCert`
  numeric facts - sums over `Finset.range 37` containing `%`, `==` and `if`,
  bounded-quantified over up to 95 phase pairs - go through with
  `maxRecDepth 20000`. I had budgeted a `List.foldl` rewrite and did not need it.
- **`omega` on 12 simultaneous `%`-atoms times out** (200,000 heartbeats), even
  though each gear's residue shift is individually trivial. Fix, which is the
  round-19 five-dvd lesson in a new costume: introduce the six shifts as
  separate one-gear `have q5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega` facts,
  rewrite with them, and finish the propositional step with `tauto`, not `omega`.
- The mega-dry-check earned its keep again (`Machine23Q` + `Machine29`
  concatenated over their built imports elaborated in one `lake env lean` at
  zero kernel cost; `PotentialLadder` and `CoveringCert` needed one dry pass
  each). Build the dry file with `grep -v "^import "` on each source plus the
  shared imports - trimming by line offsets silently eats `namespace` lines,
  which cost me one wasted 3-minute pass.
- `push_neg` is deprecated in this mathlib in favour of `push Not`; it now emits
  a warning, which fails the zero-warning invariant.
- Imports must all precede the first command: appending `#print axioms` blocks
  to `AxiomCheck.lean` requires their imports to be spliced at the TOP.

### Open formalisation targets (re-prioritised after round 23)

1. **The machine-23 chain scan by the index encoding** (verdict 13): the only
   thing between `Machine29.D_at_23_29` and a hypothesis-free fifth rung.
   `Machine23QCore.lean` already holds the predicate, verified exhaustively;
   what is needed is the `w19`-indexed rewrite (~5x, measured) and ~2.5 h of
   uncontended parallel build.
2. **The SANDWICH LEMMA, formalised** (Constructor R51):
   `Q_J(new) <= Q^[J](old) <= max_{j<=J} Q_j(new)`. With it the marked spectrum
   supplies EVERY rung from the OLD machine's period, which is the general
   version of target 1 and would retire the per-rung scan entirely. The `<=`
   half is a relaxation argument; the `>=` half (extend a relaxed window to the
   nearest survivor on each side) is the content. Abstract, machine-free, and
   it does not need a scan - the best value-per-line target in the lane.
3. **A_4 at machine 29 or 31 as a longest-path certificate** (verdict 15):
   `theorem D_of_A4 (E : Finset (N x N x N x N)) (hE : every realised 4-tuple of
   consecutive gaps is in E) (hlp : longest path over E <= 58) : (D) at 29->31`.
   14,368 states / 3,513 edges at m29 is well inside what this lane has
   kernel-checked; the hypothesis `hE` is discharged by Constructor's census,
   exactly as `Ladder.D_at_23_29`'s hypotheses were.
4. **A level-2 covering certificate at machine 23 at width 63.** The LP thread's
   degree ceiling says degree 2 goes vacuous from machine 29 on, so machine 23
   is the LAST machine at which the `CoveringCert` vehicle can be tried at all.
   If a certificate exists at `W = 63` it proves the 23->29 rung outright,
   scan-free, and target 1 becomes unnecessary. Sharp, finite, cheap to ask.
5. **The depth-sum glue at m13** (round 22 verdict 11), unchanged.
6. **Harvester's paired-Holt coef rung** (5005 -> 85085), unchanged.

### 34. CONSISTENT COVERING CERTIFICATES: 11->13 AND 13->17 IN THE KERNEL TOO
### (post-routing, after the LP-duality thread's round-23 filing)

Build green at **1334 jobs**; +1 lib `CoveringCert2`, registered, axiom audit
clean (standard three).

The thread's finding is that round 22's relaxation - and the whole classical
Bonferroni/Kounias family - drops MARGINAL CONSISTENCY, and that restoring it at
degree 2 closes 11->13 and 13->17, which no amount of DEGREE does (at machine 13
the inconsistent relaxation is feasible at degree 2, 3 AND 4). Their coordinator
routing was: keep the round-22 object at 17->19 (cheap, already done) and take
the consistent form only at 11->13 and 13->17, where it costs 464 and 2,868
rational operations.

Taken - and the formal side turns out to need NO DUAL MULTIPLIERS AT ALL, which
makes the certificates an order of magnitude smaller than the thread's.

WHERE THE INCONSISTENCY IS, IN MY OWN FILE. `CoveringCert.cover_bound` produces

    sum y + sum_j P_j(r5, rj)  <=  S_5(r5) + sum_j S_j(rj)

with the TRUE phases in it. Round 22's shape then bounds each block separately -
`max_r S_5(r)`, `max_r S_j(r)`, `min_(r5,rj) P_j(r5,rj)` - which lets gear 5 use
one phase in `S_5` and a different one in the pair minima. That IS the missing
consistency, in this development, in one line.

THE FIX IS TO KEEP THE PHASES UNDER ONE QUANTIFIER. Since the left-hand side is
literally `sum_i y_i * Kounias_i`, the sharp bound is

    sum y  <=  max over PHASE TUPLES of [ S_5(r5) + sum_j (S_j(rj) - P_j(r5,rj)) ]

which is finite, decidable, and strictly stronger. It is the `k = 5` STAR case of
marginal consistency (only gear 5's marginal is tied, because every row of the
certificate uses gear 5 as its distinguished event - my round-23 finding that the
optimum is supported on one gear is what makes this enough). No dual variable for
a consistency equation ever appears.

```lean
theorem cert13 : forall a < 5, forall b < 7, forall c < 11, forall d < 13,
    S13 b5 a + (S13 b7 b - P13 b7 a b) + (S13 b11 c - P13 b11 a c)
      + (S13 b13 d - P13 b13 a d) < 22                      -- decide +kernel
theorem cover13 (hcov : forall i < 20, the four-way or at i) :
    T13 + (P13 b7 r5 r7 + P13 b11 r5 r11 + P13 b13 r5 r13)
      <= S13 b5 r5 + S13 b7 r7 + S13 b11 r11 + S13 b13 r13
theorem no_20_run {p : N} (hp : 1 <= p) : exists i < 20, Machine13.Exposed13 (p + i)
theorem F13_le_20 (n : N) : Machine13.g13 n <= 20
theorem D_11_13_lp (n : N) : Machine13.g13 n <= 7 + 13
theorem cert17 : forall a < 5, forall b < 7, forall c < 11, forall d < 13,
    forall e < 17, ... < 94                                 -- decide +kernel
theorem F17_le_28 (n : N) : Machine17.g17 n <= 28
theorem D_13_17_lp (n : N) : Machine17.g17 n <= 11 + 17
theorem lp_ladder :   -- three consecutive rungs, one vehicle
    (forall n, Machine13.g13 n <= 7 + 13) and (forall n, Machine17.g17 n <= 11 + 17)
      and (forall n, Machine19.g19 n <= 18 + 19)
```

THE CERTIFICATES, and the size is the point:

    rung      width  weights                     sum  max over tuples  margin  tuples
    11 -> 13    20   20 integers, EIGHTEEN 1s     22        21           1      5,005
    13 -> 17    28   28 integers, all in [2,5]    94        92           2     85,085
    17 -> 19    37   37 integers (round 22)     9757     12489-2749=9740  17  1,616,615

Both new ones are PALINDROMES again. Against the thread's fully-consistent dual
at 11->13 - 106 integers over a common denominator 37, 2,868 rational operations
- the phase-tied form is **20 small integers, eighteen of them 1**, and the
verification is integer arithmetic with no denominators anywhere. Searched and
verified exactly over every phase tuple before formalising (scratchpad
consistent_cert.py, gen_consistent.py; 3,850 and 59,767 DISTINCT coefficient
vectors among the 5,005 and 85,085 tuples).

WHY THE 85,085-FOLD QUANTIFIER IS CHEAP - and this is where this round's own
infrastructure finding pays off. `cert17` quantifies over 85,085 phase tuples,
which by the round-20 rule of thumb (~5e3 tuples per declaration) should not fit.
It takes seconds, because the kernel SHARES STRUCTURALLY IDENTICAL SUBTERMS (the
control experiment in verdict 13): `S17 b7 b` is the same closed term whatever
`a, c, d, e` are, so only 53 distinct `S` sums and 240 distinct `P` sums are ever
evaluated and the 85,085 iterations are integer comparisons on cached values.
The rule of thumb is about DISTINCT sub-computations, not about the quantifier's
range - worth knowing before sizing any future check.

STATUS OF THE VEHICLE AFTER THIS: three consecutive (D) rungs kernel-proved by
covering certificates - 11->13, 13->17, 17->19 - sharing nothing with the merge
law. 11->13 and 13->17 need consistency; 17->19 does not. The thread's fourth
rung, 7->11, was not stated (it is the cheapest and least interesting).

CORRECTIONS ADOPTED FROM THE ROUTING:
- `F(29) = 43`, not 46. Checked against every statement in my lane: no
  contamination. The only place F(29) enters is the 29->31 budget
  `F(29) + 31 = 74` in section 30's table, which is 43 + 31 and was already
  right.
- **Round-23 open target 4 is WITHDRAWN.** I had named "a level-2 covering
  certificate at machine 23 at width 63" as the cheap way to make the whole
  13-hour machine-23 scan unnecessary. The thread settled it in the negative:
  23->29 is VACUOUS at degree 2 (the uniform product measure is a global
  distribution, hence feasible for the CONSISTENT relaxation too), so
  "consistency buys WIDTH, not MACHINES" and every round-22 ceiling machine is
  unchanged. The covering vehicle's honest range is 7->11 .. 17->19, with 19->23
  undecided. It cannot reach the rung my scan is for.
- Their exact rung-ratio row `B(y)/F(y)` = 2.29, 1.82, 1.56, 1.48, 1.41, 1.47,
  1.28, 1.08, 1.42 is the reason, and it is worth carrying in this lane too: a
  certificate vehicle must be NEAR-TIGHT at every step, so any bound whose gap
  grows with the machine stops proving rungs long before it stops proving
  anything. My margins are 4.5%, 2.1% and 0.14% at the three rungs - shrinking
  exactly as that row predicts.

### Open formalisation targets (final, after the routing)

1. **The machine-23 chain scan by the index encoding** (verdict 13) - unchanged,
   and now the ONLY route to the fifth rung, since target 4 is withdrawn.
2. **The SANDWICH LEMMA, formalised** (Constructor R51) - unchanged, still the
   best value-per-line object in the lane.
3. **A_4 at machine 29 or 31 as a longest-path certificate** (verdict 15) -
   unchanged.
4. ~~A level-2 covering certificate at machine 23 at width 63~~ - WITHDRAWN,
   refuted by the LP-duality thread (above).
5. **The depth-sum glue at m13** (round 22 verdict 11), unchanged.
6. **Harvester's paired-Holt coef rung** (5005 -> 85085), unchanged.

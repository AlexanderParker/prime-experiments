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

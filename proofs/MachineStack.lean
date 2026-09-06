/-
THE STACK OF MACHINES AND THE EXHAUST (Formalist, round 35).

Branch: `research/proof/theory_tree.md` node R4.b.viii (the owner's stack of
machines), `research/proof/top_machine_4.md` section 4 (the zone law L46).

THE OBJECT.  The machines are stacked.  TIER 1 (the motor) is the primes up
to `q`; its period is `cut q 1 = q#`.  TIER 2 (the wheels) is the primes in
`(q, q#]`; its period is `cut q 2`.  In general TIER `k + 1` is the primes in
`(cut q (k-1), cut q k]`, so the CUTS are `cut q 0 = q`, `cut q 1 = q#`, and
`cut q k =` tier `k`'s period `= ` the lower edge of tier `k + 2`.  Everything
above the wheels is the EXHAUST.

Three facts about the stack are formalised here, and one law of the wheels
alone.

1. STRIDE CONTAINMENT.  Every gear of tier `k + 2` exceeds tier `k`'s period,
   so it SPANS tier `k`: inside any window of that period it has at most one
   multiple, hence strikes at most two pair positions (one per tooth), while
   tier `k`'s own pattern repeats in full over the same window.

2. NON-CONTAINMENT.  A gear never spans its own tier (a product of two or more
   gears exceeds each of them), and a gear of tier `k + 2` never spans tier
   `k + 1` either: it is at most tier `k + 1`'s period, and equal only in the
   degenerate case of a one-gear tier.  Only two tiers down is there room.

3. THE EXHAUST CAP.  On the quiet zone `(C, C^2]` above a cut `C`, every
   strike of an exhaust gear is a HOME STRIKE (the gear IS the number) or an
   ECHO (a number some gear `<= C` already strikes).  Hence a pair left open
   there by tiers `1 .. k + 1` - whose union is exactly the primes `<= cut q k`
   - is a twin prime.

4. THE ZONE LAW of the wheels alone: on the smooth zone a pair is open iff
   both members are `q`-smooth; on the quiet zone a number is open iff it is
   `q`-smooth times at most one prime above `Q`.

Nothing here is a `decide`; every theorem carries the exact hypothesis it
needs.  The one place where the stack needs an input from outside its own
arithmetic is the MONOTONICITY of the cut sequence (`CutMono`): `cut q 1` is
above `cut q 0` by Bertrand (`cut_zero_le_one` below), but from there on
"the next period is above the last cut" is a statement about the density of
primes in `(cut q k, cut q (k+1)]`, and it is FALSE at `q = 2, 3` (there the
stack degenerates: at `q = 3` the cuts are `3, 6, 5, 1, ...` - tier 3 is the
single gear `{5}`, and tier 4 is EMPTY.  At `q = 5` they are
`5, 30, 215656441, ...` and climb).  So it is carried as a hypothesis exactly
where it is used.
-/

import TopMachineWalk
import Mathlib.NumberTheory.Bertrand

namespace TopMachine

/-! ## 0. Tiers and cuts -/

/-- The primes in the half-open interval `(a, b]`: one tier's gear set. -/
def gearsIoc (a b : ℕ) : Finset ℕ := (Finset.Ioc a b).filter Nat.Prime

theorem mem_gearsIoc {a b p : ℕ} : p ∈ gearsIoc a b ↔ p.Prime ∧ a < p ∧ p ≤ b := by
  simp only [gearsIoc, Finset.mem_filter, Finset.mem_Ioc]
  tauto

/-- All the primes up to `C` - the bottom of the stack, and (`stack_eq_primesLE`)
the union of a whole prefix of tiers. -/
def primesLE (C : ℕ) : Finset ℕ := gearsIoc 1 C

theorem mem_primesLE {C p : ℕ} : p ∈ primesLE C ↔ p.Prime ∧ p ≤ C := by
  rw [primesLE, mem_gearsIoc]
  constructor
  · rintro ⟨hp, _, h⟩; exact ⟨hp, h⟩
  · rintro ⟨hp, h⟩; exact ⟨hp, hp.one_lt, h⟩

/-- The cuts of the stack on base `q`: `cut q 0 = q`, `cut q 1 = q#`, and
`cut q (k+2)` is the period of tier `k + 2`, the primes in
`(cut q k, cut q (k+1)]`. -/
def cut (q : ℕ) : ℕ → ℕ
  | 0 => q
  | 1 => ∏ g ∈ gearsIoc 1 q, g
  | (k + 2) => ∏ g ∈ gearsIoc (cut q k) (cut q (k + 1)), g

/-- Tier `k` of the stack: tier 1 is the motor (primes `<= q`), tier 2 the
wheels (primes in `(q, q#]`), tier `k + 2` the primes in
`(cut q k, cut q (k+1)]`.  `tier q 0` is empty: the stack starts at 1. -/
def tier (q : ℕ) : ℕ → Finset ℕ
  | 0 => ∅
  | 1 => gearsIoc 1 q
  | (k + 2) => gearsIoc (cut q k) (cut q (k + 1))

/-- The cut sequence IS the sequence of tier periods. -/
theorem tier_prod (q k : ℕ) : ∏ g ∈ tier q (k + 1), g = cut q (k + 1) := by
  cases k <;> rfl

theorem tier_gear_prime {q k g : ℕ} (hg : g ∈ tier q (k + 1)) : g.Prime := by
  cases k with
  | zero => exact (mem_gearsIoc.mp hg).1
  | succ j => exact (mem_gearsIoc.mp hg).1

theorem tier_gear_two_le {q k g : ℕ} (hg : g ∈ tier q (k + 1)) : 2 ≤ g :=
  (tier_gear_prime hg).two_le

/-- Every gear of tier `k + 2` is above the cut `cut q k`, which is tier `k`'s
period when `k >= 1`. -/
theorem gear_gt_cut {q k g : ℕ} (hg : g ∈ tier q (k + 2)) : cut q k < g :=
  (mem_gearsIoc.mp hg).2.1

/-- Every gear of tier `k + 2` is at most `cut q (k+1)`, tier `k + 1`'s period. -/
theorem gear_le_cut {q k g : ℕ} (hg : g ∈ tier q (k + 2)) : g ≤ cut q (k + 1) :=
  (mem_gearsIoc.mp hg).2.2

theorem tier_prod_pos (q k : ℕ) : 0 < ∏ g ∈ tier q (k + 1), g :=
  Finset.prod_pos fun _ hg => (tier_gear_prime hg).pos

/-! ## 1. Stride containment: a gear two tiers up spans the tier below

A gear `g` SPANS a machine when its stride `g` is at least that machine's
period.  Spanning is exactly the containment statement: over one period of the
spanned machine the gear has at most one multiple. -/

/-- `g` spans the machine `M`: its stride is at least `M`'s period. -/
def Spans (g : ℕ) (M : Finset ℕ) : Prop := (∏ h ∈ M, h) ≤ g

instance decStrikesInt (g : ℕ) (n : ℤ) : Decidable (Strikes g n) := by
  unfold Strikes; infer_instance

/-- A gear larger than the window length has at most one multiple in it. -/
theorem card_dvd_window_le_one {g P : ℕ} (hgP : P < g) (x : ℤ) :
    (((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ x + (j : ℤ)))).card ≤ 1 := by
  have hg0 : 0 < g := lt_of_le_of_lt (Nat.zero_le P) hgP
  rw [Finset.card_le_one]
  intro a ha b hb
  rw [Finset.mem_filter, Finset.mem_range] at ha hb
  obtain ⟨ha1, ha2⟩ := ha
  obtain ⟨hb1, hb2⟩ := hb
  have hd : (g : ℤ) ∣ ((a : ℤ) - (b : ℤ)) := by
    have h := dvd_sub ha2 hb2
    have : x + (a : ℤ) - (x + (b : ℤ)) = (a : ℤ) - (b : ℤ) := by ring
    rwa [this] at h
  have hlt1 : (a : ℤ) < (g : ℤ) := by exact_mod_cast lt_trans ha1 hgP
  have hlt2 : (b : ℤ) < (g : ℤ) := by exact_mod_cast lt_trans hb1 hgP
  have hnn1 : (0 : ℤ) ≤ (a : ℤ) := Int.natCast_nonneg a
  have hnn2 : (0 : ℤ) ≤ (b : ℤ) := Int.natCast_nonneg b
  have := eq_zero_of_dvd_of_abs_lt hg0 hd (by linarith) (by linarith)
  have : (a : ℤ) = (b : ℤ) := by linarith
  exact_mod_cast this

/-- A gear larger than the window length strikes at most two pair positions in
it - one per tooth. -/
theorem card_strikes_window_le_two {g P : ℕ} (hgP : P < g) (x : ℤ) :
    (((Finset.range P).filter (fun (j : ℕ) => Strikes g (x + (j : ℤ))))).card ≤ 2 := by
  have hsub :
      ((Finset.range P).filter (fun (j : ℕ) => Strikes g (x + (j : ℤ)))) ⊆
        ((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ x + (j : ℤ))) ∪
          ((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ (x + 2) + (j : ℤ))) := by
    intro j hj
    rw [Finset.mem_filter] at hj
    obtain ⟨hj1, hj2⟩ := hj
    rcases hj2 with h | h
    · exact Finset.mem_union_left _ (Finset.mem_filter.mpr ⟨hj1, h⟩)
    · refine Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨hj1, ?_⟩)
      have : x + (j : ℤ) + 2 = (x + 2) + (j : ℤ) := by ring
      rwa [this] at h
  calc ((Finset.range P).filter (fun (j : ℕ) => Strikes g (x + (j : ℤ)))).card
      ≤ (((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ x + (j : ℤ))) ∪
          ((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ (x + 2) + (j : ℤ)))).card :=
        Finset.card_le_card hsub
    _ ≤ ((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ x + (j : ℤ))).card +
          ((Finset.range P).filter (fun (j : ℕ) => (g : ℤ) ∣ (x + 2) + (j : ℤ))).card :=
        Finset.card_union_le _ _
    _ ≤ 1 + 1 := Nat.add_le_add (card_dvd_window_le_one hgP x)
        (card_dvd_window_le_one hgP (x + 2))
    _ = 2 := rfl

/-- The spanned machine's own pattern repeats in full over every stride: the
open set is invariant under a shift by any common multiple of its gears. -/
theorem strikes_add_period {g : ℕ} {P : ℤ} (h : (g : ℤ) ∣ P) (n : ℤ) :
    Strikes g (n + P) ↔ Strikes g n := by
  unfold Strikes
  have h1 : ((g : ℤ) ∣ n + P) ↔ ((g : ℤ) ∣ n) :=
    dvd_iff_of_dvd_sub (by simpa using h)
  have h2 : ((g : ℤ) ∣ n + P + 2) ↔ ((g : ℤ) ∣ n + 2) := by
    refine dvd_iff_of_dvd_sub ?_
    have : n + P + 2 - (n + 2) = P := by ring
    rwa [this]
  rw [h1, h2]

theorem isOpen_add_period {G : Finset ℕ} {P : ℤ} (hP : ∀ g ∈ G, (g : ℤ) ∣ P) (n : ℤ) :
    IsOpen G (n + P) ↔ IsOpen G n :=
  forall_congr' fun g => forall_congr' fun hg => not_congr (strikes_add_period (hP g hg) n)

/-- One period of a tier carries the tier's whole pattern. -/
theorem tier_pattern_repeats (q k : ℕ) (n : ℤ) :
    IsOpen (tier q (k + 1)) (n + ((∏ h ∈ tier q (k + 1), h : ℕ) : ℤ)) ↔
      IsOpen (tier q (k + 1)) n :=
  isOpen_add_period (fun _ hg =>
    Int.natCast_dvd_natCast.mpr (Finset.dvd_prod_of_mem _ hg)) n

/-- STRIDE CONTAINMENT, the size half: every gear of tier `k + 3` spans tier
`k + 1`. -/
theorem spans_two_below {q k g : ℕ} (hg : g ∈ tier q (k + 3)) : Spans g (tier q (k + 1)) := by
  have h := gear_gt_cut (k := k + 1) hg
  rw [Spans, tier_prod]
  exact le_of_lt h

/-- STRIDE CONTAINMENT, the counting half: a gear two tiers up strikes at most
two positions of any window of the lower tier's period. -/
theorem stride_containment {q k g : ℕ} (hg : g ∈ tier q (k + 3)) (x : ℤ) :
    (((Finset.range (∏ h ∈ tier q (k + 1), h)).filter
      (fun (j : ℕ) => Strikes g (x + (j : ℤ)))).card) ≤ 2 := by
  refine card_strikes_window_le_two ?_ x
  have h := gear_gt_cut (k := k + 1) hg
  rwa [tier_prod]

/-! ## 2. Non-containment: no gear spans its own tier, or the tier below -/

/-- A product of two or more gears, each at least 2, exceeds each of them. -/
theorem lt_prod_of_two_le {G : Finset ℕ} (h2 : ∀ g ∈ G, 2 ≤ g) (hcard : 2 ≤ G.card)
    {g : ℕ} (hg : g ∈ G) : g < ∏ h ∈ G, h := by
  obtain ⟨h, hh, hne⟩ : ∃ h ∈ G, h ≠ g := by
    by_contra hc
    push Not at hc
    have hsub : G ⊆ {g} := fun x hx => Finset.mem_singleton.mpr (hc x hx)
    have := Finset.card_le_card hsub
    rw [Finset.card_singleton] at this
    omega
  have hprod : ∏ x ∈ G, x = g * ∏ x ∈ G.erase g, x := (Finset.mul_prod_erase G _ hg).symm
  have hmem : h ∈ G.erase g := Finset.mem_erase.mpr ⟨hne, hh⟩
  have hrest : 2 ≤ ∏ x ∈ G.erase g, x :=
    le_trans (h2 h hh)
      (Finset.single_le_prod' (fun i hi => le_trans (by norm_num)
        (h2 i (Finset.mem_of_mem_erase hi))) hmem)
  have hg2 : 2 ≤ g := h2 g hg
  calc g < g * 2 := by omega
    _ ≤ g * ∏ x ∈ G.erase g, x := Nat.mul_le_mul_left g hrest
    _ = ∏ x ∈ G, x := hprod.symm

/-- NO SELF-SPAN: a gear never spans its own tier. -/
theorem not_spans_self {q k g : ℕ} (hcard : 2 ≤ (tier q (k + 1)).card)
    (hg : g ∈ tier q (k + 1)) : ¬ Spans g (tier q (k + 1)) := by
  rw [Spans]
  exact not_le.mpr (lt_prod_of_two_le (fun x hx => tier_gear_two_le hx) hcard hg)

/-- A gear of tier `k + 2` is at most tier `k + 1`'s period - one tier down is
never enough room. -/
theorem gear_le_period_below {q k g : ℕ} (hg : g ∈ tier q (k + 2)) :
    g ≤ ∏ h ∈ tier q (k + 1), h := by
  rw [tier_prod]
  exact gear_le_cut hg

/-- NO SPAN ONE TIER DOWN: a gear of tier `k + 2` spans tier `k + 1` only if
that tier is a single gear (its silent top).  With two or more gears below,
never. -/
theorem not_spans_below {q k g : ℕ} (hcard : 2 ≤ (tier q (k + 1)).card)
    (hg : g ∈ tier q (k + 2)) : ¬ Spans g (tier q (k + 1)) := by
  rw [Spans]
  intro hspan
  have hle := gear_le_period_below hg
  have heq : g = ∏ h ∈ tier q (k + 1), h := le_antisymm hle hspan
  obtain ⟨h, hh⟩ : ∃ h, h ∈ tier q (k + 1) := by
    rcases Finset.card_pos.mp (by omega : 0 < (tier q (k + 1)).card) with ⟨h, hh⟩
    exact ⟨h, hh⟩
  have hdvd : h ∣ g := heq ▸ Finset.dvd_prod_of_mem _ hh
  have hlt : h < g := heq ▸ lt_prod_of_two_le (fun x hx => tier_gear_two_le hx) hcard hh
  have hgp : g.Prime := tier_gear_prime (k := k + 1) hg
  rcases (Nat.Prime.eq_one_or_self_of_dvd hgp h hdvd) with h1 | h1
  · have := tier_gear_two_le hh; omega
  · omega

/-! ## 3. The exhaust cap: home strikes and echoes on the quiet zone -/

theorem strikesN_natCast (g n : ℕ) : StrikesN g (n : ℤ) ↔ g ∣ n :=
  Int.natCast_dvd_natCast

/-- The pair view and the single-number view agree: a pair is open iff both of
its members are. -/
theorem isOpen_iff_openNum (G : Finset ℕ) (n : ℤ) :
    IsOpen G n ↔ OpenNum G n ∧ OpenNum G (n + 2) := by
  constructor
  · intro h
    exact ⟨fun g hg hs => h g hg (Or.inl hs), fun g hg hs => h g hg (Or.inr hs)⟩
  · rintro ⟨h1, h2⟩ g hg (hs | hs)
    · exact h1 g hg hs
    · exact h2 g hg hs

/-- THE EXHAUST CAP (the redundancy cap).  Above a cut `C >= 2`, on the quiet
zone `(C, C^2]`, a prime `p > C` dividing `n` leaves only two possibilities:
`n = p` (a HOME STRIKE) or `n` has a prime factor `<= C` (an ECHO of a gear at
or below the cut). -/
theorem exhaust_home_or_echo {C n p : ℕ} (h1 : C < n) (h2 : n ≤ C ^ 2)
    (hpC : C < p) (hpn : p ∣ n) :
    n = p ∨ ∃ r ∈ primesLE C, r ∣ n := by
  obtain ⟨m, rfl⟩ := hpn
  have hCC : C < C ^ 2 := lt_of_lt_of_le h1 h2
  have hC0 : 0 < C := Nat.pos_of_ne_zero (by rintro rfl; simp at hCC)
  rcases Nat.lt_or_ge m 2 with hm | hm
  · interval_cases m
    · simp at h1
    · left; omega
  · right
    have hmC : m < C := by
      have hstep : m * (C + 1) ≤ C * C := by
        calc m * (C + 1) ≤ m * p := Nat.mul_le_mul_left m (by omega)
          _ = p * m := Nat.mul_comm _ _
          _ ≤ C ^ 2 := h2
          _ = C * C := by ring
      have hCC : C * C < C * (C + 1) := by
        have : 0 < C := by omega
        nlinarith
      by_contra hcon
      push Not at hcon
      have : C * (C + 1) ≤ m * (C + 1) := Nat.mul_le_mul_right _ hcon
      omega
    have hm0 : m ≠ 1 := by omega
    refine ⟨m.minFac, mem_primesLE.mpr ⟨Nat.minFac_prime hm0, ?_⟩, ?_⟩
    · exact le_trans (Nat.minFac_le (by omega)) (le_of_lt hmC)
    · exact Dvd.dvd.mul_left (Nat.minFac_dvd m) p

/-- On the quiet zone `(C, C^2]` a number is open under all the primes `<= C`
iff it is prime. -/
theorem openNum_iff_prime {C n : ℕ} (hC : 2 ≤ C) (h1 : C < n) (h2 : n ≤ C ^ 2) :
    OpenNum (primesLE C) (n : ℤ) ↔ n.Prime := by
  constructor
  · intro hopen
    have hn2 : 2 ≤ n := by omega
    have hp : (n.minFac).Prime := Nat.minFac_prime (by omega)
    have hdvd : n.minFac ∣ n := Nat.minFac_dvd n
    have hbig : C < n.minFac := by
      by_contra hcon
      push Not at hcon
      exact hopen n.minFac (mem_primesLE.mpr ⟨hp, hcon⟩)
        ((strikesN_natCast _ _).mpr hdvd)
    rcases exhaust_home_or_echo h1 h2 hbig hdvd with heq | ⟨r, hr, hrn⟩
    · rw [heq]; exact hp
    · exact absurd ((strikesN_natCast _ _).mpr hrn) (hopen r hr)
  · intro hn g hg hs
    have hgn : g ∣ n := (strikesN_natCast _ _).mp hs
    have hgp := (mem_primesLE.mp hg).1
    have hgC := (mem_primesLE.mp hg).2
    rcases (Nat.Prime.eq_one_or_self_of_dvd hn g hgn) with h | h
    · exact absurd h (by have := hgp.two_le; omega)
    · omega

/-- THE CLUTCH ON THE QUIET ZONE: a pair left open there by all the gears up to
the cut is a twin prime, and conversely. -/
theorem open_iff_twin {C n : ℕ} (hC : 2 ≤ C) (h1 : C < n) (h2 : n + 2 ≤ C ^ 2) :
    IsOpen (primesLE C) (n : ℤ) ↔ (n.Prime ∧ (n + 2).Prime) := by
  rw [isOpen_iff_openNum]
  have hcast : ((n : ℤ) + 2) = ((n + 2 : ℕ) : ℤ) := by push_cast; ring
  rw [hcast]
  exact and_congr (openNum_iff_prime hC h1 (by omega))
    (openNum_iff_prime hC (by omega) h2)

/-! ### The stack form -/

/-- The union of tiers `1 .. k + 1`. -/
def stack (q k : ℕ) : Finset ℕ := (Finset.range (k + 1)).biUnion (fun j => tier q (j + 1))

/-- The cuts are nondecreasing up to level `k`.  True from `q = 5` on, false at
`q = 2, 3`; it is a statement about the density of primes in
`(cut q j, cut q (j+1)]`, not about the stack's own arithmetic, so it is
carried as a hypothesis. -/
def CutMono (q k : ℕ) : Prop := ∀ j < k, cut q j ≤ cut q (j + 1)

theorem cut_mono_le {q k : ℕ} (h : CutMono q k) : ∀ {i j : ℕ}, i ≤ j → j ≤ k → cut q i ≤ cut q j := by
  intro i j hij hjk
  induction j with
  | zero =>
    have hi : i = 0 := by omega
    subst hi; exact le_refl _
  | succ n ih =>
    rcases Nat.lt_or_ge i (n + 1) with hlt | hge
    · exact le_trans (ih (by omega) (by omega)) (h n (by omega))
    · have : i = n + 1 := by omega
      subst this; exact le_refl _

/-- THE UNION OF A PREFIX OF THE STACK is exactly the primes up to the cut:
tiers `1 .. k + 1` are the primes `<= cut q k`. -/
theorem stack_eq_primesLE (q : ℕ) : ∀ k, CutMono q k → stack q k = primesLE (cut q k) := by
  intro k
  induction k with
  | zero =>
    intro _
    show (Finset.range 1).biUnion (fun j => tier q (j + 1)) = primesLE (cut q 0)
    rw [Finset.range_one, Finset.singleton_biUnion]
    rfl
  | succ n ih =>
    intro hmono
    have hstep : cut q n ≤ cut q (n + 1) := hmono n (by omega)
    have hprev : stack q n = primesLE (cut q n) := ih (fun j hj => hmono j (by omega))
    have hsplit : stack q (n + 1) = stack q n ∪ tier q (n + 2) := by
      rw [stack, stack, Finset.range_add_one, Finset.biUnion_insert]
      exact Finset.union_comm _ _
    rw [hsplit, hprev]
    ext p
    have htier : (p ∈ tier q (n + 2)) ↔ (p.Prime ∧ cut q n < p ∧ p ≤ cut q (n + 1)) :=
      mem_gearsIoc
    rw [Finset.mem_union, mem_primesLE, mem_primesLE, htier]
    constructor
    · rintro (⟨hp, hle⟩ | ⟨hp, _, hle⟩)
      · exact ⟨hp, le_trans hle hstep⟩
      · exact ⟨hp, hle⟩
    · rintro ⟨hp, hle⟩
      rcases Nat.lt_or_ge (cut q n) p with hlt | hge
      · exact Or.inr ⟨hp, hlt, hle⟩
      · exact Or.inl ⟨hp, hge⟩

/-- Every gear of the exhaust - every tier from `k + 2` upwards - is above the
cut `cut q k`, tier `k`'s period. -/
theorem exhaust_gear_gt_cut {q k j g : ℕ} (hmono : CutMono q j) (hjk : k ≤ j)
    (hg : g ∈ tier q (j + 2)) : cut q k < g :=
  lt_of_le_of_lt (cut_mono_le hmono hjk (le_refl j)) (gear_gt_cut hg)

/-- THE EXHAUST IS SILENT ON THE QUIET ZONE, in stack form: on
`(cut q k, (cut q k)^2]` every strike of an exhaust gear is a home strike or an
echo of a gear of tiers `1 .. k + 1`. -/
theorem exhaust_silent {q k j n g : ℕ} (hmono : CutMono q j) (hjk : k ≤ j)
    (hmk : CutMono q k) (hg : g ∈ tier q (j + 2)) (hgn : g ∣ n)
    (h1 : cut q k < n) (h2 : n ≤ (cut q k) ^ 2) :
    n = g ∨ ∃ r ∈ stack q k, r ∣ n := by
  have hgt : cut q k < g := exhaust_gear_gt_cut hmono hjk hg
  rw [stack_eq_primesLE q k hmk]
  exact exhaust_home_or_echo h1 h2 hgt hgn

/-- THE STACK'S WINDOW STATEMENT: a pair left open by tiers `1 .. k + 1` on the
quiet zone `(cut q k, (cut q k)^2]` is a twin prime, and every twin prime there
is such a pair.  The exhaust cannot change it (`exhaust_silent`). -/
theorem stack_open_iff_twin {q k n : ℕ} (hmono : CutMono q k) (hC : 2 ≤ cut q k)
    (h1 : cut q k < n) (h2 : n + 2 ≤ (cut q k) ^ 2) :
    IsOpen (stack q k) (n : ℤ) ↔ (n.Prime ∧ (n + 2).Prime) := by
  rw [stack_eq_primesLE q k hmono]
  exact open_iff_twin hC h1 h2

/-! ## 4. The zone law of the wheels -/

/-- `n` is `q`-smooth: every prime factor is at most `q`. -/
def Smooth (q n : ℕ) : Prop := ∀ p : ℕ, p.Prime → p ∣ n → p ≤ q

/-- THE SMOOTH ZONE, single-number form: for `0 < n <= Q`, `n` is open under the
gears `(q, Q]` iff `n` is `q`-smooth. -/
theorem smooth_zone_num {q Q n : ℕ} (hn : 0 < n) (hnQ : n ≤ Q) :
    OpenNum (gearsIoc q Q) (n : ℤ) ↔ Smooth q n := by
  constructor
  · intro hopen p hp hpn
    by_contra hcon
    push Not at hcon
    have hpQ : p ≤ Q := le_trans (Nat.le_of_dvd hn hpn) hnQ
    exact hopen p (mem_gearsIoc.mpr ⟨hp, hcon, hpQ⟩) ((strikesN_natCast _ _).mpr hpn)
  · intro hsm g hg hs
    have := mem_gearsIoc.mp hg
    have hgn : g ∣ n := (strikesN_natCast _ _).mp hs
    have := hsm g this.1 hgn
    omega

/-- THE ZONE LAW (top_machine_4.md L46): for `n + 2 <= Q` the pair `n` is open
under the wheels `(q, Q]` iff `n` and `n + 2` are both `q`-smooth. -/
theorem smooth_zone {q Q n : ℕ} (hn : 0 < n) (hnQ : n + 2 ≤ Q) :
    IsOpen (gearsIoc q Q) (n : ℤ) ↔ (Smooth q n ∧ Smooth q (n + 2)) := by
  rw [isOpen_iff_openNum]
  have hcast : ((n : ℤ) + 2) = ((n + 2 : ℕ) : ℤ) := by push_cast; ring
  rw [hcast]
  exact and_congr (smooth_zone_num hn (by omega)) (smooth_zone_num (by omega) hnQ)

/-- The wheels are the tier-2 gears, so the zone law is a law of tier 2. -/
theorem wheels_smooth_zone {q n : ℕ} (hn : 0 < n) (hnQ : n + 2 ≤ cut q 1) :
    IsOpen (tier q 2) (n : ℤ) ↔ (Smooth q n ∧ Smooth q (n + 2)) :=
  smooth_zone hn hnQ

/-- THE QUIET ZONE: for `0 < n <= Q^2`, `n` is open under the gears `(q, Q]` iff
`n` is `q`-smooth times at most one prime above `Q`. -/
theorem quiet_zone {q Q n : ℕ} (hn : 0 < n) (hnQ : n ≤ Q ^ 2) :
    OpenNum (gearsIoc q Q) (n : ℤ) ↔
      ∃ s P : ℕ, n = s * P ∧ Smooth q s ∧ (P = 1 ∨ (P.Prime ∧ Q < P)) := by
  constructor
  · intro hopen
    by_cases hbig : ∃ p, p.Prime ∧ Q < p ∧ p ∣ n
    · obtain ⟨p, hp, hpQ, s, rfl⟩ := hbig
      refine ⟨s, p, by ring, ?_, Or.inr ⟨hp, hpQ⟩⟩
      intro r hr hrs
      have hrn : r ∣ p * s := Dvd.dvd.mul_left hrs p
      rcases Nat.lt_or_ge q r with hqr | hqr
      · exfalso
        rcases Nat.lt_or_ge Q r with hQr | hQr
        · -- two prime factors above Q: the product already exceeds Q^2
          have hpr : p * r ∣ p * s := Nat.mul_dvd_mul_left p hrs
          have hle : p * r ≤ p * s := Nat.le_of_dvd hn hpr
          have h1 : Q + 1 ≤ p := hpQ
          have h2 : Q + 1 ≤ r := hQr
          have : Q ^ 2 < p * r := by nlinarith
          omega
        · exact hopen r (mem_gearsIoc.mpr ⟨hr, hqr, hQr⟩) ((strikesN_natCast _ _).mpr hrn)
      · exact hqr
    · push Not at hbig
      refine ⟨n, 1, by ring, ?_, Or.inl rfl⟩
      intro p hp hpn
      rcases Nat.lt_or_ge q p with hqp | hqp
      · rcases Nat.lt_or_ge Q p with hQp | hQp
        · exact absurd hpn (hbig p hp hQp)
        · exact absurd ((strikesN_natCast _ _).mpr hpn)
            (hopen p (mem_gearsIoc.mpr ⟨hp, hqp, hQp⟩))
      · exact hqp
  · rintro ⟨s, P, rfl, hsm, hP⟩ g hg hs
    have hmem := mem_gearsIoc.mp hg
    have hgn : g ∣ s * P := (strikesN_natCast _ _).mp hs
    rcases (Nat.Prime.dvd_mul hmem.1).mp hgn with h | h
    · have := hsm g hmem.1 h
      omega
    · rcases hP with rfl | ⟨hPp, hPQ⟩
      · have := Nat.le_of_dvd (by norm_num) h
        have := hmem.1.two_le
        omega
      · rcases (Nat.Prime.eq_one_or_self_of_dvd hPp g h) with h1 | h1
        · have := hmem.1.two_le; omega
        · omega

/-! ## 5. The one step of the cut sequence that is unconditional

`CutMono` is carried as a hypothesis because "the next period is above the last
cut" is a prime-density statement, and it FAILS at the bottom of the stack for
`q = 2, 3`.  The FIRST step is different: `cut q 0 = q` is below
`cut q 1 = q#` for every `q`, by Bertrand's postulate.  So the motor-and-wheels
case - the stack prefix the project actually uses - needs no hypothesis. -/

/-- The primorial is at least its argument: `n <= n#`.  Bertrand supplies, for
`n = p * m` with `m >= 2`, a prime in `(m, 2m] ⊆ (m, n]` which is not counted in
`m#`, and `p <= m` because `p` is the least prime factor. -/
theorem le_prod_primesLE (n : ℕ) : n ≤ ∏ p ∈ primesLE n, p := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rcases Nat.lt_or_ge n 2 with hn | hn
    · interval_cases n
      · exact Nat.zero_le _
      · simp [primesLE, gearsIoc]
    · have hpp : (n.minFac).Prime := Nat.minFac_prime (by omega)
      obtain ⟨m, hm⟩ := Nat.minFac_dvd n
      have hp2 : 2 ≤ n.minFac := hpp.two_le
      have hm0 : 0 < m := by
        rcases Nat.eq_zero_or_pos m with rfl | h
        · omega
        · exact h
      rcases Nat.lt_or_ge m 2 with hm1 | hm2
      · -- `m = 1`: `n` is itself prime, and it is one of the factors of `n#`.
        have hmeq : m = 1 := by omega
        subst hmeq
        rw [Nat.mul_one] at hm
        refine Finset.single_le_prod' (fun _ hi => (mem_primesLE.mp hi).1.one_lt.le) ?_
        exact mem_primesLE.mpr ⟨by rw [hm]; exact hpp, le_refl n⟩
      · have hpm : n.minFac ≤ m :=
          Nat.minFac_le_of_dvd hm2 ⟨n.minFac, by rw [Nat.mul_comm]; exact hm⟩
        have hmlt : m < n := by nlinarith
        have ihm : m ≤ ∏ p ∈ primesLE m, p := ih m hmlt
        obtain ⟨p', hp'p, hp'gt, hp'le⟩ := Nat.exists_prime_lt_and_le_two_mul m (by omega)
        have hp'n : p' ≤ n := le_trans hp'le (by nlinarith)
        have hnotmem : p' ∉ primesLE m := fun h => by
          have := (mem_primesLE.mp h).2; omega
        have hsub : insert p' (primesLE m) ⊆ primesLE n := by
          intro x hx
          rcases Finset.mem_insert.mp hx with rfl | hx
          · exact mem_primesLE.mpr ⟨hp'p, hp'n⟩
          · exact mem_primesLE.mpr ⟨(mem_primesLE.mp hx).1,
              le_trans (mem_primesLE.mp hx).2 (le_of_lt hmlt)⟩
        have hins : ∏ x ∈ insert p' (primesLE m), x = p' * ∏ x ∈ primesLE m, x :=
          Finset.prod_insert hnotmem
        have hle : ∏ x ∈ insert p' (primesLE m), x ≤ ∏ x ∈ primesLE n, x :=
          Finset.prod_le_prod_of_subset_of_one_le' hsub
            (fun _ hi _ => (mem_primesLE.mp hi).1.one_lt.le)
        rw [hins] at hle
        have hstep : n ≤ p' * ∏ x ∈ primesLE m, x := by nlinarith
        exact le_trans hstep hle

/-- The motor's period is above the motor's top gear: `cut q 0 <= cut q 1`. -/
theorem cut_zero_le_one (q : ℕ) : cut q 0 ≤ cut q 1 := le_prod_primesLE q

/-- The first step of the cut sequence is monotone with no hypothesis. -/
theorem cutMono_one (q : ℕ) : CutMono q 1 := by
  intro j hj
  interval_cases j
  exact cut_zero_le_one q

theorem two_le_cut_one {q : ℕ} (hq : 2 ≤ q) : 2 ≤ cut q 1 :=
  Finset.single_le_prod' (fun _ hi => (mem_primesLE.mp hi).1.one_lt.le)
    (mem_primesLE.mpr ⟨Nat.prime_two, hq⟩)

/-- MOTOR AND WHEELS ARE THE PRIMES UP TO `q#`, with no hypothesis. -/
theorem stack_one (q : ℕ) : stack q 1 = primesLE (cut q 1) :=
  stack_eq_primesLE q 1 (cutMono_one q)

/-- THE HEADLINE, unconditional: on the quiet zone `(q#, (q#)^2]` a pair left
open by the motor and the wheels together is a twin prime, and every twin prime
there is such a pair.  Everything above the wheels - the whole exhaust - can
only home-strike or echo there (`exhaust_silent`). -/
theorem wheels_open_iff_twin {q n : ℕ} (hq : 2 ≤ q) (h1 : cut q 1 < n)
    (h2 : n + 2 ≤ (cut q 1) ^ 2) :
    IsOpen (stack q 1) (n : ℤ) ↔ (n.Prime ∧ (n + 2).Prime) :=
  stack_open_iff_twin (cutMono_one q) (two_le_cut_one hq) h1 h2


end TopMachine

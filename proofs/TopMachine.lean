/-
THE TOP MACHINE ON THE RAW LINE (Formalist, round 32).

Branch document: `research/proof/top_machine_1.md`, sections 0 and 4; scripts
`research/topmachine/r1/*.py`.  Every statement below was checked against a
brute-force model first (arcs, counts, mirror, clump, runs, chains, chain
law, merge law, parity law, conjugacy - zero exceptions on the wheels the
branch reports).

THE OBJECT.  The line is the integers, unfolded; no anchor, no columns.  A
PAIR is `(n, n + 2)`, indexed by its lower member, so the pair coordinate IS
the raw line.  A GEAR `g` strikes the pair `n` iff `g | n` or `g | n + 2` -
two teeth, at `0` and at `-2`, of separation 2.  A pair struck by no gear of
a finite gear set `G` is OPEN.  The wheel is `W = prod G`.

The gears of the owner's construction are the primes above `q`, so in
practice `g >= 7`; the laws hold under weaker hypotheses and each theorem
below carries the exact one it needs (`3 <= g`, `5 <= g`, `g` odd, ...).
Nothing here assumes primality: only size, oddness and - for the wheel count,
in `TopMachineWheel.lean` - pairwise coprimality.

The law numbers in the section headers are the branch document's (section 4).
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Finset.Range
import Mathlib.Data.Finset.Max
import Mathlib.Algebra.Ring.Divisibility.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

namespace TopMachine

/-! ## 0. The machine -/

/-- **Gear `g` strikes the pair `n`** iff it divides either member. -/
def Strikes (g : ℕ) (n : ℤ) : Prop := (g : ℤ) ∣ n ∨ (g : ℤ) ∣ (n + 2)

/-- **The pair `n` is open for the gear set `G`** iff no gear strikes it. -/
def IsOpen (G : Finset ℕ) (n : ℤ) : Prop := ∀ g ∈ G, ¬ Strikes g n

/-- The residue form of `Strikes`, on `ℕ`, used for the wheel count. -/
def StrikesR (g r : ℕ) : Prop := r % g = 0 ∨ (r + 2) % g = 0

/-- The residue form of `IsOpen`. -/
def OpenN (G : Finset ℕ) (n : ℕ) : Prop := ∀ g ∈ G, ¬ StrikesR g n

instance decStrikesR (g r : ℕ) : Decidable (StrikesR g r) := by
  unfold StrikesR; infer_instance

instance decOpenN (G : Finset ℕ) (n : ℕ) : Decidable (OpenN G n) := by
  unfold OpenN; infer_instance

/-- The two forms agree on natural arguments. -/
theorem strikes_natCast (g n : ℕ) : Strikes g (n : ℤ) ↔ StrikesR g n := by
  unfold Strikes StrikesR
  have h1 : ((g : ℤ) ∣ (n : ℤ)) ↔ g ∣ n := Int.natCast_dvd_natCast
  have h2 : ((g : ℤ) ∣ ((n : ℤ) + 2)) ↔ g ∣ (n + 2) := by
    rw [show ((n : ℤ) + 2) = ((n + 2 : ℕ) : ℤ) by push_cast; ring]
    exact Int.natCast_dvd_natCast
  rw [h1, h2, Nat.dvd_iff_mod_eq_zero, Nat.dvd_iff_mod_eq_zero]

theorem open_natCast (G : Finset ℕ) (n : ℕ) : IsOpen G (n : ℤ) ↔ OpenN G n :=
  forall_congr' fun g => forall_congr' fun _ => not_congr (strikes_natCast g n)

/-! ## Arithmetic helpers -/

/-- A multiple of `g` of absolute value below `g` is zero. -/
theorem eq_zero_of_dvd_of_abs_lt {g : ℕ} (hg : 0 < g) {z : ℤ} (h : (g : ℤ) ∣ z)
    (h1 : -(g : ℤ) < z) (h2 : z < (g : ℤ)) : z = 0 := by
  obtain ⟨c, rfl⟩ := h
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg
  rcases lt_trichotomy c 0 with hc | hc | hc
  · have hle : (g : ℤ) * c ≤ (g : ℤ) * (-1) :=
      mul_le_mul_of_nonneg_left (by omega) (le_of_lt hgz)
    rw [mul_neg_one] at hle; linarith
  · rw [hc, mul_zero]
  · have hle : (g : ℤ) * 1 ≤ (g : ℤ) * c :=
      mul_le_mul_of_nonneg_left (by omega) (le_of_lt hgz)
    rw [mul_one] at hle; linarith

/-- A nonzero integer of absolute value below `g` is not a multiple of `g`. -/
theorem not_dvd_of_abs_lt {g : ℕ} (hg : 0 < g) {z : ℤ} (hz : z ≠ 0)
    (h1 : -(g : ℤ) < z) (h2 : z < (g : ℤ)) : ¬ (g : ℤ) ∣ z :=
  fun h => hz (eq_zero_of_dvd_of_abs_lt hg h h1 h2)

/-- Divisibility transports along a difference the gear divides. -/
theorem dvd_iff_of_dvd_sub {g : ℕ} {x y : ℤ} (h : (g : ℤ) ∣ x - y) :
    (g : ℤ) ∣ x ↔ (g : ℤ) ∣ y := by
  constructor
  · intro hx
    have := dvd_sub hx h
    rwa [show x - (x - y) = y by ring] at this
  · intro hy
    have := dvd_add hy h
    rwa [show y + (x - y) = x by ring] at this

/-- An odd gear dividing `2 c` divides `c`: `2` is invertible mod `g`. -/
theorem dvd_of_dvd_two_mul {g : ℕ} (hodd : g % 2 = 1) {c : ℤ} (h : (g : ℤ) ∣ 2 * c) :
    (g : ℤ) ∣ c := by
  obtain ⟨s, hs⟩ : ∃ s, g = 2 * s + 1 := ⟨g / 2, by omega⟩
  have hgz : (g : ℤ) = 2 * (s : ℤ) + 1 := by rw [hs]; push_cast; ring
  have h1 : (g : ℤ) ∣ (2 * c) * ((s : ℤ) + 1) := h.mul_right _
  have h2 : (g : ℤ) ∣ c * (g : ℤ) := Dvd.intro_left c rfl
  have h3 : (2 * c) * ((s : ℤ) + 1) - c * (g : ℤ) = c := by rw [hgz]; ring
  have := dvd_sub h1 h2
  rwa [h3] at this

/-! ## L1 (two teeth, separation 2) and L2 (the arcs and the shield)

Every gear has exactly two teeth, at `n = 0` and `n = -2 (mod g)`, hence
`g - 2` slots; the slots form the arcs `1 .. g-3` and the singleton `{g-1}`,
the SHIELD. -/

/-- **L1, the teeth.**  Struck residues below `g` are exactly `0` and `g - 2`. -/
theorem mod_add_two_eq_zero_iff {g r : ℕ} (hg : 3 ≤ g) (hr : r < g) :
    (r + 2) % g = 0 ↔ r = g - 2 := by
  constructor
  · intro h
    obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero h
    have hk2 : k < 2 := by
      by_contra hc
      have h2 : g * 2 ≤ g * k := Nat.mul_le_mul_left g (by omega)
      omega
    interval_cases k <;> omega
  · intro h; subst h
    have hgg : g - 2 + 2 = g := by omega
    rw [hgg, Nat.mod_self]

theorem strikesR_iff {g r : ℕ} (hg : 3 ≤ g) (hr : r < g) :
    StrikesR g r ↔ (r = 0 ∨ r = g - 2) := by
  unfold StrikesR
  rw [Nat.mod_eq_of_lt hr, mod_add_two_eq_zero_iff hg hr]

/-- **L2, the arcs.**  The open residues of one gear are the long arc
`1 .. g-3` together with the shield `g - 1`. -/
theorem open_residues {g : ℕ} (hg : 3 ≤ g) :
    (Finset.range g).filter (fun r => ¬ StrikesR g r)
      = Finset.Ico 1 (g - 2) ∪ {g - 1} := by
  ext r
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_union, Finset.mem_Ico,
    Finset.mem_singleton]
  constructor
  · rintro ⟨hrg, hns⟩
    rw [strikesR_iff hg hrg] at hns
    omega
  · intro h
    have hrg : r < g := by omega
    refine ⟨hrg, ?_⟩
    rw [strikesR_iff hg hrg]
    omega

/-- **L1, the count.**  Exactly `g - 2` open residues modulo `g`. -/
theorem card_open_residues {g : ℕ} (hg : 3 ≤ g) :
    ((Finset.range g).filter (fun r => ¬ StrikesR g r)).card = g - 2 := by
  rw [open_residues hg, Finset.card_union_of_disjoint, Nat.card_Ico, Finset.card_singleton]
  · omega
  · rw [Finset.disjoint_singleton_right, Finset.mem_Ico]
    omega

/-- **L2, the shield.**  The pair `n = -1` is open for every gear `g >= 2`:
its members are `-1` and `1`. -/
theorem not_strikes_neg_one {g : ℕ} (hg : 2 ≤ g) : ¬ Strikes g (-1 : ℤ) := by
  have hgz : (2 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hg
  have hg0 : 0 < g := by omega
  rintro (h | h)
  · exact not_dvd_of_abs_lt hg0 (by norm_num) (by linarith) (by linarith) h
  · rw [show (-1 : ℤ) + 2 = 1 by ring] at h
    exact not_dvd_of_abs_lt hg0 (by norm_num) (by linarith) (by linarith) h

/-! ## L3 (the partner law) and L4 (the forbidden gap)

Every strike has a partner strike of the same gear exactly 2 away: the struck
set of one gear is a disjoint union of dominoes `{x, x + 2}`.  A gap of
exactly 4 is therefore impossible. -/

/-- **L3, the partner law.**  A strike is never alone. -/
theorem partner {g : ℕ} {n : ℤ} (h : Strikes g n) :
    Strikes g (n - 2) ∨ Strikes g (n + 2) := by
  rcases h with h | h
  · left; right; rwa [show n - 2 + 2 = n by ring]
  · right; left; exact h

/-- **L3, the domino form.**  The struck set of `g` is exactly the union of
the dominoes `{x, x + 2}` over the multiples `x` of `g`. -/
theorem strikes_iff_domino {g : ℕ} {n : ℤ} :
    Strikes g n ↔ ∃ x : ℤ, (g : ℤ) ∣ x ∧ (n = x ∨ n = x - 2) := by
  constructor
  · rintro (h | h)
    · exact ⟨n, h, Or.inl rfl⟩
    · exact ⟨n + 2, h, Or.inr (by ring)⟩
  · rintro ⟨x, hx, rfl | rfl⟩
    · exact Or.inl hx
    · exact Or.inr (by rwa [show x - 2 + 2 = x by ring])

/-- **L4, the forbidden gap.**  If `n` and `n + 4` are open then so is
`n + 2` - the striker of the middle pair would have to strike one of the
two bounding pairs. -/
theorem open_of_open_add_four {G : Finset ℕ} {n : ℤ}
    (h0 : IsOpen G n) (h4 : IsOpen G (n + 4)) : IsOpen G (n + 2) := by
  intro g hg hs
  rcases hs with h | h
  · exact h0 g hg (Or.inr h)
  · exact h4 g hg (Or.inl (by rwa [show n + 2 + 2 = n + 4 by ring] at h))

/-- **L4 as the branch states it**: a gap of exactly 4 never occurs. -/
theorem no_gap_four {G : Finset ℕ} {n : ℤ}
    (h0 : IsOpen G n) (h4 : IsOpen G (n + 4))
    (hmid : ∀ z : ℤ, n < z → z < n + 4 → ¬ IsOpen G z) : False :=
  hmid (n + 2) (by linarith) (by linarith) (open_of_open_add_four h0 h4)

/-! ## L6 (the always-open pair, the antipode, the origin clump) -/

/-- **L6, the always-open pair.**  `n = -1` is open for every gear set whose
gears are at least 2. -/
theorem shield_open {G : Finset ℕ} (hG : ∀ g ∈ G, 2 ≤ g) : IsOpen G (-1 : ℤ) :=
  fun g hg => not_strikes_neg_one (hG g hg)

/-- The clump lemma: a pair both of whose members are nonzero and smaller in
absolute value than every gear is open. -/
theorem open_of_small {G : Finset ℕ} {q : ℕ} {n : ℤ} (hq : 1 ≤ q)
    (hG : ∀ g ∈ G, q ≤ g)
    (h1 : -(q : ℤ) + 1 ≤ n) (h2 : n ≤ (q : ℤ) - 3)
    (hn0 : n ≠ 0) (hn2 : n + 2 ≠ 0) : IsOpen G n := by
  intro g hg hs
  have hqg : (q : ℤ) ≤ (g : ℤ) := by exact_mod_cast hG g hg
  have hg0 : 0 < g := by have := hG g hg; omega
  have hq1 : (1 : ℤ) ≤ (q : ℤ) := by exact_mod_cast hq
  rcases hs with h | h
  · exact not_dvd_of_abs_lt hg0 hn0 (by linarith) (by linarith) h
  · exact not_dvd_of_abs_lt hg0 hn2 (by linarith) (by linarith) h

/-- **L6, the antipode analogue.**  `n = 2` (members 2, 4) is open for every
gear set with gears `>= 5`. -/
theorem two_open {G : Finset ℕ} (hG : ∀ g ∈ G, 5 ≤ g) : IsOpen G (2 : ℤ) := by
  intro g hg hs
  have hgz : (5 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hG g hg
  have hg0 : 0 < g := by have := hG g hg; omega
  rcases hs with h | h
  · exact not_dvd_of_abs_lt hg0 (by norm_num) (by linarith) (by linarith) h
  · rw [show (2 : ℤ) + 2 = 4 by ring] at h
    exact not_dvd_of_abs_lt hg0 (by norm_num) (by linarith) (by linarith) h

/-- **L6, the antipode's mirror image.**  `n = -4` (members -4, -2) is open
for every gear set with gears `>= 5`. -/
theorem neg_four_open {G : Finset ℕ} (hG : ∀ g ∈ G, 5 ≤ g) : IsOpen G (-4 : ℤ) := by
  intro g hg hs
  have hgz : (5 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hG g hg
  have hg0 : 0 < g := by have := hG g hg; omega
  rcases hs with h | h
  · exact not_dvd_of_abs_lt hg0 (by norm_num) (by linarith) (by linarith) h
  · rw [show (-4 : ℤ) + 2 = -2 by ring] at h
    exact not_dvd_of_abs_lt hg0 (by norm_num) (by linarith) (by linarith) h

/-- **L6, the origin clump.**  With smallest gear `q'`, every `n` in
`[-(q'-1), q'-3]` other than `0` and `-2` is open: two maximal runs of
`q' - 3` slots separated by struck / shield / struck. -/
theorem clump_open {G : Finset ℕ} {q : ℕ} {n : ℤ} (hq : 3 ≤ q)
    (hG : ∀ g ∈ G, q ≤ g)
    (h1 : -(q : ℤ) + 1 ≤ n) (h2 : n ≤ (q : ℤ) - 3)
    (hn0 : n ≠ 0) (hn2 : n ≠ -2) : IsOpen G n :=
  open_of_small (by omega) hG h1 h2 hn0 (by omega)

/-- **L6, the clump above the origin**: every `n` in `[1, q' - 3]` is open. -/
theorem origin_clump {G : Finset ℕ} {q : ℕ} {n : ℤ} (hq : 3 ≤ q)
    (hG : ∀ g ∈ G, q ≤ g) (h1 : 1 ≤ n) (h2 : n ≤ (q : ℤ) - 3) : IsOpen G n := by
  have hqz : (3 : ℤ) ≤ (q : ℤ) := by exact_mod_cast hq
  exact clump_open hq hG (by linarith) h2 (by omega) (by omega)

/-! ## L7 (the mirror) -/

/-- **L7, one gear.**  `n |-> -n - 2` exchanges the two members of a pair. -/
theorem strikes_mirror {g : ℕ} {n : ℤ} : Strikes g (-n - 2) ↔ Strikes g n := by
  unfold Strikes
  constructor
  · rintro (h | h)
    · right; rwa [show -n - 2 = -(n + 2) by ring, dvd_neg] at h
    · left; rwa [show -n - 2 + 2 = -n by ring, dvd_neg] at h
  · rintro (h | h)
    · right; rw [show -n - 2 + 2 = -n by ring, dvd_neg]; exact h
    · left; rw [show -n - 2 = -(n + 2) by ring, dvd_neg]; exact h

/-- **L7, the mirror.**  `n |-> -n - 2` maps the open set onto itself. -/
theorem open_mirror {G : Finset ℕ} {n : ℤ} : IsOpen G (-n - 2) ↔ IsOpen G n :=
  forall_congr' fun _ => forall_congr' fun _ => not_congr strikes_mirror

/-- The mirror's unique fixed point is the shield `n = -1`. -/
theorem mirror_fixed_iff {n : ℤ} : -n - 2 = n ↔ n = -1 := by omega

/-! ## L8 (the symmetry group)

Sufficiency for every gear set, and per-gear necessity. -/

/-- **L8, sufficiency, one gear.**  If `c = +-1 (mod g)` then
`n |-> c(n+1) - 1` preserves gear `g`'s struck set. -/
theorem strikes_affine {g : ℕ} {c n : ℤ} (hc : (g : ℤ) ∣ c - 1 ∨ (g : ℤ) ∣ c + 1) :
    Strikes g (c * (n + 1) - 1) ↔ Strikes g n := by
  rcases hc with hc | hc
  · obtain ⟨t, ht⟩ := hc
    have d1 : (g : ℤ) ∣ (c * (n + 1) - 1) - n :=
      ⟨t * (n + 1), by linear_combination (n + 1) * ht⟩
    have d2 : (g : ℤ) ∣ (c * (n + 1) - 1 + 2) - (n + 2) :=
      ⟨t * (n + 1), by linear_combination (n + 1) * ht⟩
    unfold Strikes
    rw [dvd_iff_of_dvd_sub d1, dvd_iff_of_dvd_sub d2]
  · obtain ⟨t, ht⟩ := hc
    have d1 : (g : ℤ) ∣ (c * (n + 1) - 1) - (-(n + 2)) :=
      ⟨t * (n + 1), by linear_combination (n + 1) * ht⟩
    have d2 : (g : ℤ) ∣ (c * (n + 1) - 1 + 2) - (-n) :=
      ⟨t * (n + 1), by linear_combination (n + 1) * ht⟩
    unfold Strikes
    rw [dvd_iff_of_dvd_sub d1, dvd_iff_of_dvd_sub d2, dvd_neg, dvd_neg]
    exact or_comm

/-- **L8, sufficiency.**  The maps `n |-> c(n+1) - 1` with `c = +-1` modulo
every gear preserve the open set.  `c = 1` is the identity; `c = -1` is the
mirror of L7. -/
theorem open_affine {G : Finset ℕ} {c n : ℤ}
    (hc : ∀ g ∈ G, (g : ℤ) ∣ c - 1 ∨ (g : ℤ) ∣ c + 1) :
    IsOpen G (c * (n + 1) - 1) ↔ IsOpen G n :=
  forall_congr' fun g =>
    ⟨fun h hg => (not_congr (strikes_affine (hc g hg))).mp (h hg),
     fun h hg => (not_congr (strikes_affine (hc g hg))).mpr (h hg)⟩

/-- `c = 1` is the identity. -/
theorem affine_one (n : ℤ) : (1 : ℤ) * (n + 1) - 1 = n := by ring

/-- `c = -1` is exactly the mirror of L7. -/
theorem affine_neg_one (n : ℤ) : (-1 : ℤ) * (n + 1) - 1 = -n - 2 := by ring

/-- **L8, adjacency.**  The map `n |-> c(n+1) - 1` moves consecutive pairs by
exactly `c`, so it preserves adjacency precisely when `c = +-1` in `ℤ_W`; of
the `(Z/2)^m` symmetries only two - the identity and the mirror - qualify. -/
theorem affine_step (c n : ℤ) :
    (c * ((n + 1) + 1) - 1) - (c * (n + 1) - 1) = c := by ring

/-- **L8, necessity, one gear.**  An affine map `n |-> c n + b` whose `c` is a
unit mod `g` and which preserves gear `g`'s struck set must permute the tooth
pair `{0, -2}`, so `(c, b) = (1, 0)` or `(-1, -2)` modulo `g`.  The branch's
one-line proof: evaluate at the two teeth, and use that an odd gear dividing
`2c` divides `c`. -/
theorem affine_teeth {g : ℕ} (hodd : g % 2 = 1) {c b : ℤ}
    (hcu : ¬ (g : ℤ) ∣ c)
    (h : ∀ n : ℤ, Strikes g (c * n + b) ↔ Strikes g n) :
    ((g : ℤ) ∣ c - 1 ∧ (g : ℤ) ∣ b) ∨ ((g : ℤ) ∣ c + 1 ∧ (g : ℤ) ∣ b + 2) := by
  have key : ∀ z : ℤ, (g : ℤ) ∣ 2 * z → (g : ℤ) ∣ z := fun z hz =>
    dvd_of_dvd_two_mul hodd hz
  have h0 : Strikes g b := by
    have hz := (h 0).mpr (Or.inl (by norm_num))
    rwa [show c * 0 + b = b by ring] at hz
  have h2 : Strikes g (c * (-2) + b) := (h (-2)).mpr (Or.inr (by norm_num))
  rcases h0 with hb | hb
  · rcases h2 with hs | hs
    · exact absurd (key c (by
        have hd := dvd_sub hb hs
        rwa [show b - (c * (-2) + b) = 2 * c by ring] at hd)) hcu
    · refine Or.inl ⟨?_, hb⟩
      have hd := dvd_sub hs hb
      rw [show (c * (-2) + b + 2) - b = 2 * (1 - c) by ring] at hd
      have hd2 := key _ hd
      rwa [show (1 : ℤ) - c = -(c - 1) by ring, dvd_neg] at hd2
  · rcases h2 with hs | hs
    · refine Or.inr ⟨?_, hb⟩
      have hd := dvd_sub hb hs
      rw [show (b + 2) - (c * (-2) + b) = 2 * (c + 1) by ring] at hd
      exact key _ hd
    · exact absurd (key c (by
        have hd := dvd_sub hb hs
        rwa [show (b + 2) - (c * (-2) + b + 2) = 2 * c by ring] at hd)) hcu

/-! ## L10 (the alignment law)

The longest run of consecutive open pairs is exactly `q' - 3`, and the
longest step-2 chain is exactly `q' - 2`. -/

/-- Upper bound, one gear: gear `g` strikes somewhere in any `g - 2`
consecutive pairs. -/
theorem no_long_run {g : ℕ} (hg : 5 ≤ g) (n : ℤ)
    (h : ∀ i : ℤ, 0 ≤ i → i < (g : ℤ) - 2 → ¬ Strikes g (n + i)) : False := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast (by omega : 0 < g)
  have hg5 : (5 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hg
  have hr0 : 0 ≤ n % (g : ℤ) := Int.emod_nonneg n (by omega)
  have hrg : n % (g : ℤ) < (g : ℤ) := Int.emod_lt_of_pos n hgz
  have hdvd : (g : ℤ) ∣ n - n % (g : ℤ) := ⟨n / (g : ℤ), by rw [Int.emod_def]; ring⟩
  by_cases hr : n % (g : ℤ) = 0
  · refine h 0 le_rfl (by linarith) (Or.inl ?_)
    rw [add_zero]
    rwa [hr, sub_zero] at hdvd
  · have hrpos : 0 < n % (g : ℤ) := lt_of_le_of_ne hr0 (Ne.symm hr)
    by_cases hr2 : n % (g : ℤ) ≤ 2
    · refine h ((g : ℤ) - n % (g : ℤ) - 2) (by linarith) (by linarith) (Or.inr ?_)
      rw [show n + ((g : ℤ) - n % (g : ℤ) - 2) + 2 = (n - n % (g : ℤ)) + (g : ℤ) by ring]
      exact dvd_add hdvd dvd_rfl
    · refine h ((g : ℤ) - n % (g : ℤ)) (by linarith) (by linarith) (Or.inl ?_)
      rw [show n + ((g : ℤ) - n % (g : ℤ)) = (n - n % (g : ℤ)) + (g : ℤ) by ring]
      exact dvd_add hdvd dvd_rfl

/-- **L10, upper bound.**  No machine has `q' - 2` consecutive open pairs,
`q'` any gear at least 5.  With attainment below, the longest run is exactly
`q' - 3`, `q'` the smallest gear. -/
theorem run_lt {G : Finset ℕ} {q : ℕ} (hq : 5 ≤ q) (hqG : q ∈ G) (n : ℤ)
    (h : ∀ i : ℤ, 0 ≤ i → i < (q : ℤ) - 2 → IsOpen G (n + i)) : False :=
  no_long_run hq n fun i h1 h2 => h i h1 h2 q hqG

/-- **L10, attainment.**  The origin clump realises a run of `q' - 3`
consecutive open pairs at `1, 2, ..., q' - 3`. -/
theorem run_attained {G : Finset ℕ} {q : ℕ} (hq : 3 ≤ q) (hG : ∀ g ∈ G, q ≤ g) :
    ∀ i : ℤ, 0 ≤ i → i < (q : ℤ) - 3 → IsOpen G (1 + i) := by
  intro i h1 h2
  exact origin_clump hq hG (by linarith) (by linarith)

/-- Upper bound, one gear, for the step-2 chain: an odd gear `g` strikes
somewhere in any `g - 1` pairs spaced 2 apart. -/
theorem no_long_chain2 {g : ℕ} (hg : 3 ≤ g) (hodd : g % 2 = 1) (n : ℤ)
    (h : ∀ j : ℤ, 0 ≤ j → j < (g : ℤ) - 1 → ¬ Strikes g (n + 2 * j)) : False := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast (by omega : 0 < g)
  obtain ⟨t, ht⟩ : ∃ t, g = 2 * t + 1 := ⟨g / 2, by omega⟩
  have htz : (g : ℤ) = 2 * (t : ℤ) + 1 := by rw [ht]; push_cast; ring
  have hj00 : 0 ≤ (-(n * ((t : ℤ) + 1))) % (g : ℤ) :=
    Int.emod_nonneg _ (by omega)
  have hj0g : (-(n * ((t : ℤ) + 1))) % (g : ℤ) < (g : ℤ) :=
    Int.emod_lt_of_pos _ hgz
  have hdvdA : (g : ℤ) ∣ (-(n * ((t : ℤ) + 1))) - (-(n * ((t : ℤ) + 1))) % (g : ℤ) :=
    ⟨(-(n * ((t : ℤ) + 1))) / (g : ℤ), by rw [Int.emod_def]; ring⟩
  have hkey : (g : ℤ) ∣ n + 2 * ((-(n * ((t : ℤ) + 1))) % (g : ℤ)) := by
    rw [show n + 2 * ((-(n * ((t : ℤ) + 1))) % (g : ℤ))
        = (-n) * (2 * (t : ℤ) + 1)
          - 2 * ((-(n * ((t : ℤ) + 1))) - (-(n * ((t : ℤ) + 1))) % (g : ℤ)) by ring,
      ← htz]
    exact dvd_sub (Dvd.intro_left _ rfl) (Dvd.dvd.mul_left hdvdA 2)
  by_cases hlast : (-(n * ((t : ℤ) + 1))) % (g : ℤ) < (g : ℤ) - 1
  · exact h _ hj00 hlast (Or.inl hkey)
  · have hj0eq : (-(n * ((t : ℤ) + 1))) % (g : ℤ) = (g : ℤ) - 1 := by omega
    refine h ((g : ℤ) - 2) (by linarith) (by linarith) (Or.inr ?_)
    rw [show n + 2 * ((g : ℤ) - 2) + 2 = n + 2 * ((g : ℤ) - 1) by ring, ← hj0eq]
    exact hkey

/-- **L10, the chain, upper bound.**  No machine has `q' - 1` open pairs in
step-2 progression. -/
theorem chain2_lt {G : Finset ℕ} {q : ℕ} (hq : 3 ≤ q) (hodd : q % 2 = 1) (hqG : q ∈ G)
    (n : ℤ) (h : ∀ j : ℤ, 0 ≤ j → j < (q : ℤ) - 1 → IsOpen G (n + 2 * j)) : False :=
  no_long_chain2 hq hodd n fun j h1 h2 => h j h1 h2 q hqG

/-- **L10, the chain, attainment.**  The odd members of the origin clump form
a step-2 chain of `q' - 2` open pairs, from `-(q' - 2)`. -/
theorem chain2_attained {G : Finset ℕ} {q : ℕ} (hq : 3 ≤ q) (hodd : q % 2 = 1)
    (hG : ∀ g ∈ G, q ≤ g) :
    ∀ j : ℤ, 0 ≤ j → j < (q : ℤ) - 2 → IsOpen G (-((q : ℤ) - 2) + 2 * j) := by
  intro j h1 h2
  obtain ⟨t, ht⟩ : ∃ t, q = 2 * t + 1 := ⟨q / 2, by omega⟩
  have htz : (q : ℤ) = 2 * (t : ℤ) + 1 := by rw [ht]; push_cast; ring
  have hform : -((q : ℤ) - 2) + 2 * j = 2 * (j - (t : ℤ)) + 1 := by rw [htz]; ring
  refine clump_open hq hG (by linarith) (by linarith) ?_ ?_
  · rw [hform]; omega
  · rw [hform]; omega

/-! ## L12 (the chain law) -/

/-- **L12, the chain law.**  Two openings `x`, `y` of a machine are both
struck by a new gear `g` in some translate iff `y - x = 0, +2, -2 (mod g)`.
Translating the machine and translating the gear's phase are the same thing,
so the shift `s` ranges over `ℤ`. -/
theorem chain_law {g : ℕ} (x y : ℤ) :
    (∃ s : ℤ, Strikes g (x + s) ∧ Strikes g (y + s)) ↔
      ((g : ℤ) ∣ y - x ∨ (g : ℤ) ∣ y - x - 2 ∨ (g : ℤ) ∣ y - x + 2) := by
  constructor
  · rintro ⟨s, hx, hy⟩
    rcases hx with hx | hx <;> rcases hy with hy | hy
    · exact Or.inl (by
        have hd := dvd_sub hy hx
        rwa [show (y + s) - (x + s) = y - x by ring] at hd)
    · exact Or.inr (Or.inr (by
        have hd := dvd_sub hy hx
        rwa [show (y + s + 2) - (x + s) = y - x + 2 by ring] at hd))
    · exact Or.inr (Or.inl (by
        have hd := dvd_sub hy hx
        rwa [show (y + s) - (x + s + 2) = y - x - 2 by ring] at hd))
    · exact Or.inl (by
        have hd := dvd_sub hy hx
        rwa [show (y + s + 2) - (x + s + 2) = y - x by ring] at hd)
  · rintro (h | h | h)
    · exact ⟨-x, Or.inl (by rw [show x + -x = (0:ℤ) by ring]; exact dvd_zero _),
        Or.inl (by rwa [show y + -x = y - x by ring])⟩
    · exact ⟨-x - 2, Or.inr (by rw [show x + (-x - 2) + 2 = (0:ℤ) by ring]; exact dvd_zero _),
        Or.inl (by rwa [show y + (-x - 2) = y - x - 2 by ring])⟩
    · exact ⟨-x, Or.inl (by rw [show x + -x = (0:ℤ) by ring]; exact dvd_zero _),
        Or.inr (by rwa [show y + -x + 2 = y - x + 2 by ring])⟩

/-! ## L13 (the merge law) -/

/-- **L13, the merge law.**  If `x < y` are consecutive openings of the
machine `G + g`, then both are openings of `G` and every opening of `G`
strictly between them is struck by `g`: a gap of `M + g` is a gap of `M` or a
merge of consecutive gaps of `M` whose interior openings `g` strikes. -/
theorem merge_law {G : Finset ℕ} {g : ℕ} {x y : ℤ}
    (hx : IsOpen (insert g G) x) (hy : IsOpen (insert g G) y)
    (hmid : ∀ z : ℤ, x < z → z < y → ¬ IsOpen (insert g G) z) :
    IsOpen G x ∧ IsOpen G y ∧
      ∀ z : ℤ, x < z → z < y → IsOpen G z → Strikes g z := by
  refine ⟨fun a ha => hx a (Finset.mem_insert_of_mem ha),
    fun a ha => hy a (Finset.mem_insert_of_mem ha), ?_⟩
  intro z h1 h2 hz
  by_contra hgs
  refine hmid z h1 h2 ?_
  intro a ha
  rcases Finset.mem_insert.mp ha with rfl | ha'
  · exact hgs
  · exact hz a ha'

/-! ## L17 (the parity law), upper bound

If every gear of `G` (`m = |G|`) is odd and exceeds `2m + 1` then no run of
consecutive struck pairs is longer than `2m - (m mod 2)`.  The branch's
tiling argument: inside a short window each gear's strikes form one
distance-2 domino, which never crosses parity, so the evens and the odds of
the run are covered by DISJOINT sets of gears, at most two positions each. -/

/-- Inside a window of length `L` with `L + 2 <= g`, two positions struck by
the same gear differ by exactly `0`, `2` or `-2`. -/
theorem window_pair {g : ℕ} {n : ℤ} {i j L : ℕ} (hL : L + 2 ≤ g)
    (hi : i < L) (hj : j < L)
    (hsi : Strikes g (n + (i : ℤ))) (hsj : Strikes g (n + (j : ℤ))) :
    (i : ℤ) - (j : ℤ) = 0 ∨ (i : ℤ) - (j : ℤ) = 2 ∨ (i : ℤ) - (j : ℤ) = -2 := by
  have hg0 : 0 < g := by omega
  have hgz : (L : ℤ) + 2 ≤ (g : ℤ) := by exact_mod_cast hL
  have hiz : (i : ℤ) < (L : ℤ) := by exact_mod_cast hi
  have hjz : (j : ℤ) < (L : ℤ) := by exact_mod_cast hj
  have hi0 : (0 : ℤ) ≤ (i : ℤ) := Int.natCast_nonneg i
  have hj0 : (0 : ℤ) ≤ (j : ℤ) := Int.natCast_nonneg j
  rcases hsi with hsi | hsi <;> rcases hsj with hsj | hsj
  · refine Or.inl (eq_zero_of_dvd_of_abs_lt hg0 ?_ (by linarith) (by linarith))
    have hd := dvd_sub hsi hsj
    rwa [show (n + (i : ℤ)) - (n + (j : ℤ)) = (i : ℤ) - (j : ℤ) by ring] at hd
  · refine Or.inr (Or.inl ?_)
    have hd := dvd_sub hsi hsj
    rw [show (n + (i : ℤ)) - (n + (j : ℤ) + 2) = (i : ℤ) - (j : ℤ) - 2 by ring] at hd
    have := eq_zero_of_dvd_of_abs_lt hg0 hd (by linarith) (by linarith)
    linarith
  · refine Or.inr (Or.inr ?_)
    have hd := dvd_sub hsi hsj
    rw [show (n + (i : ℤ) + 2) - (n + (j : ℤ)) = (i : ℤ) - (j : ℤ) + 2 by ring] at hd
    have := eq_zero_of_dvd_of_abs_lt hg0 hd (by linarith) (by linarith)
    linarith
  · refine Or.inl (eq_zero_of_dvd_of_abs_lt hg0 ?_ (by linarith) (by linarith))
    have hd := dvd_sub hsi hsj
    rwa [show (n + (i : ℤ) + 2) - (n + (j : ℤ) + 2) = (i : ℤ) - (j : ℤ) by ring] at hd

/-- `(range L).filter (· % 2 = 0)` has `(L+1)/2` elements. -/
theorem card_even_range (L : ℕ) :
    ((Finset.range L).filter (fun i => i % 2 = 0)).card = (L + 1) / 2 := by
  induction L with
  | zero => simp
  | succ k ih =>
      rw [Finset.range_add_one, Finset.filter_insert]
      split_ifs with h
      · rw [Finset.card_insert_of_notMem (by simp)]; omega
      · omega

/-- `(range L).filter (· % 2 = 1)` has `L/2` elements. -/
theorem card_odd_range (L : ℕ) :
    ((Finset.range L).filter (fun i => i % 2 = 1)).card = L / 2 := by
  induction L with
  | zero => simp
  | succ k ih =>
      rw [Finset.range_add_one, Finset.filter_insert]
      split_ifs with h
      · rw [Finset.card_insert_of_notMem (by simp)]; omega
      · omega

/-- **L17, the core count.**  A run of `L <= 2m + 1` consecutive struck pairs
forces `L + (m mod 2) <= 2m`. -/
theorem parity_core {G : Finset ℕ} {n : ℤ} {L : ℕ}
    (hodd : ∀ g ∈ G, g % 2 = 1) (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hL : L ≤ 2 * G.card + 1)
    (hstruck : ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))) :
    L + G.card % 2 ≤ 2 * G.card := by
  classical
  have hex : ∀ i : ℕ, i < L → ∃ g ∈ G, Strikes g (n + (i : ℤ)) := by
    intro i hi
    by_contra hc
    exact hstruck i hi fun g hg hs => hc ⟨g, hg, hs⟩
  have hex' : ∀ i : ℕ, ∃ g : ℕ, i < L → g ∈ G ∧ Strikes g (n + (i : ℤ)) := by
    intro i
    by_cases hi : i < L
    · obtain ⟨g, hg, hs⟩ := hex i hi
      exact ⟨g, fun _ => ⟨hg, hs⟩⟩
    · exact ⟨0, fun h => absurd h hi⟩
  choose f hfmem using hex'
  have hwin : ∀ g ∈ G, L + 2 ≤ g := by
    intro g hg
    have h1 := hbig g hg
    have h2 := hodd g hg
    omega
  have hpair : ∀ i j : ℕ, i < L → j < L → f i = f j →
      (i : ℤ) - (j : ℤ) = 0 ∨ (i : ℤ) - (j : ℤ) = 2 ∨ (i : ℤ) - (j : ℤ) = -2 := by
    intro i j hi hj hij
    obtain ⟨hgi, hsi⟩ := hfmem i hi
    obtain ⟨hgj, hsj⟩ := hfmem j hj
    rw [hij] at hsi
    exact window_pair (hwin (f j) hgj) hi hj hsi hsj
  set E := (Finset.range L).filter (fun i => i % 2 = 0) with hE
  set O := (Finset.range L).filter (fun i => i % 2 = 1) with hO
  have hEsub : ∀ i ∈ E, i < L := by
    intro i hi
    exact Finset.mem_range.mp (Finset.mem_filter.mp hi).1
  have hOsub : ∀ i ∈ O, i < L := by
    intro i hi
    exact Finset.mem_range.mp (Finset.mem_filter.mp hi).1
  have hfib : ∀ (S : Finset ℕ), (∀ i ∈ S, i < L) → ∀ b ∈ S.image f,
      (S.filter (fun i => f i = b)).card ≤ 2 := by
    intro S hS b _
    rcases Finset.eq_empty_or_nonempty (S.filter (fun i => f i = b)) with hTe | hTne
    · rw [hTe]; simp
    · set a := (S.filter (fun i => f i = b)).min' hTne with hadef
      have ha : a ∈ S.filter (fun i => f i = b) := Finset.min'_mem _ hTne
      have hsub : S.filter (fun i => f i = b) ⊆ ({a, a + 2} : Finset ℕ) := by
        intro x hx
        have hxa : a ≤ x := Finset.min'_le _ x hx
        have hx1 := Finset.mem_filter.mp hx
        have ha1 := Finset.mem_filter.mp ha
        have hp := hpair x a (hS x hx1.1) (hS a ha1.1) (by rw [hx1.2, ha1.2])
        have hxaz : (a : ℤ) ≤ (x : ℤ) := by exact_mod_cast hxa
        simp only [Finset.mem_insert, Finset.mem_singleton]
        omega
      calc (S.filter (fun i => f i = b)).card
          ≤ ({a, a + 2} : Finset ℕ).card := Finset.card_le_card hsub
        _ ≤ 2 := by
            refine le_trans (Finset.card_insert_le _ _) ?_
            simp
  have hEcard : E.card ≤ 2 * (E.image f).card :=
    Finset.card_le_mul_card_image E 2 (hfib E hEsub)
  have hOcard : O.card ≤ 2 * (O.image f).card :=
    Finset.card_le_mul_card_image O 2 (hfib O hOsub)
  have hdisj : Disjoint (E.image f) (O.image f) := by
    rw [Finset.disjoint_left]
    intro b hbE hbO
    obtain ⟨i, hi, hib⟩ := Finset.mem_image.mp hbE
    obtain ⟨j, hj, hjb⟩ := Finset.mem_image.mp hbO
    have hie : i % 2 = 0 := (Finset.mem_filter.mp hi).2
    have hjo : j % 2 = 1 := (Finset.mem_filter.mp hj).2
    have hp := hpair i j (hEsub i hi) (hOsub j hj) (by rw [hib, hjb])
    omega
  have hsubG : (E.image f) ∪ (O.image f) ⊆ G := by
    intro b hb
    rcases Finset.mem_union.mp hb with h | h
    · obtain ⟨i, hi, hib⟩ := Finset.mem_image.mp h
      rw [← hib]; exact (hfmem i (hEsub i hi)).1
    · obtain ⟨i, hi, hib⟩ := Finset.mem_image.mp h
      rw [← hib]; exact (hfmem i (hOsub i hi)).1
  have hsum : (E.image f).card + (O.image f).card ≤ G.card := by
    rw [← Finset.card_union_of_disjoint hdisj]
    exact Finset.card_le_card hsubG
  have hEv : E.card = (L + 1) / 2 := card_even_range L
  have hOv : O.card = L / 2 := card_odd_range L
  omega

/-- **L17, the parity law, upper bound.**  If every gear of `G` is odd and
exceeds `2m + 1`, `m = |G|`, then the longest run of consecutive struck pairs
is at most `2m - (m mod 2)`: `2m` for an even number of gears, `2m - 1` for an
odd number.  The record's SIZE is decided by the parity of the number of
gears and by nothing else - not by the sizes of the gears. -/
theorem parity_upper {G : Finset ℕ} {n : ℤ} {L : ℕ}
    (hodd : ∀ g ∈ G, g % 2 = 1) (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hstruck : ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))) :
    L + G.card % 2 ≤ 2 * G.card := by
  by_cases hL : L ≤ 2 * G.card + 1
  · exact parity_core hodd hbig hL hstruck
  · exfalso
    have h := parity_core (L := 2 * G.card + 1) hodd hbig le_rfl
      (fun i hi => hstruck i (by omega))
    omega

end TopMachine

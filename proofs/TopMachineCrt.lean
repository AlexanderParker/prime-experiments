/-
THE TOP MACHINE: THE FINSET-INDEXED CRT, AND THE TWO LAWS IT UNBLOCKS
(Formalist, round 33).

Round 32 left exactly two laws of `research/proof/top_machine_1.md` section 4
open, both for the same missing reason: there was no `Finset`-indexed Chinese
remainder theorem in the project, only the two-modulus counting form
`TopMachine.card_filter_crt`.

  * **L17, attainment** - the record `F_top = 2m - (m mod 2)` needs an explicit
    phase vector: one distance-2 domino per gear, `ceil(m/2)` of them tiling
    the even positions of `[0, L)` and `floor(m/2)` tiling the odd ones.  Each
    gear's phase is a residue, and the run exists iff the residue vector is
    realisable - which is what CRT says.

  * **L8, the exact group** - going from "the affine map preserves the OPEN SET
    of `G`" to "it preserves EACH GEAR's struck set" needs, for the gear under
    test, an `n` in a prescribed class mod `g` whose other gears miss both `n`
    and its image; that is a CRT with a per-gear avoidance choice (4 forbidden
    residues out of `g >= 5`).

This file proves the CRT (`exists_crt`, `crt_unique`) and then closes both.

Nothing here assumes primality except `affine_group`, the convenience corollary
of `affine_group_of_unit`: primality is used ONLY to turn "no gear divides the
multiplier `c`" (which is proved for arbitrary coprime gears, `symm_not_dvd_mul`)
into "`c` is invertible modulo each gear".
-/

import TopMachineWheel
import Mathlib.Data.Int.GCD
import Mathlib.RingTheory.Coprime.Basic
import Mathlib.RingTheory.Coprime.Lemmas
import Mathlib.RingTheory.Int.Basic
import Mathlib.Data.Int.ModEq

namespace TopMachine

/-! ## 1. The Finset-indexed Chinese remainder theorem -/

/-- Bezout, in the form used throughout: if `a` and `P` are coprime then `P` is
invertible modulo `a`. -/
theorem exists_inv_of_coprime {a P : ℕ} (h : Nat.Coprime a P) :
    ∃ v : ℤ, (a : ℤ) ∣ (P : ℤ) * v - 1 := by
  have h1 : Nat.gcd a P = 1 := h
  have hbez : (1 : ℤ) = (a : ℤ) * Nat.gcdA a P + (P : ℤ) * Nat.gcdB a P := by
    have h0 := Nat.gcd_eq_gcd_ab a P
    rw [h1] at h0
    exact_mod_cast h0
  exact ⟨Nat.gcdB a P, ⟨-(Nat.gcdA a P), by linear_combination -hbez⟩⟩

/-- The same over `ℤ`: an integer coprime to the gear is invertible mod it. -/
theorem exists_inv_of_isCoprime {g : ℕ} {c : ℤ} (h : IsCoprime (g : ℤ) c) :
    ∃ c' : ℤ, (g : ℤ) ∣ c * c' - 1 := by
  obtain ⟨u, v, huv⟩ := h
  exact ⟨v, ⟨-u, by linear_combination huv⟩⟩

/-- **THE LEMMA.  Finset-indexed CRT, existence.**  For a finite set `G` of
pairwise coprime moduli and any family of residues `r : ℕ → ℤ` there is a single
integer `n` with `n = r g (mod g)` for every `g ∈ G`.

Proved by induction on `G`: the new modulus `a` is coprime to the product of the
old ones, so the old solution can be corrected by a multiple of that product. -/
theorem exists_crt : ∀ (G : Finset ℕ),
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ (r : ℕ → ℤ),
    ∃ n : ℤ, ∀ g ∈ G, (g : ℤ) ∣ n - r g := by
  classical
  intro G
  induction G using Finset.induction_on with
  | empty => intro _ r; exact ⟨0, by simp⟩
  | insert a s hasnot ih =>
      intro hcop r
      have hmemA : a ∈ insert a s := Finset.mem_insert_self a s
      have hscop : ∀ g ∈ s, ∀ h ∈ s, g ≠ h → Nat.Coprime g h := fun g hg h hh hne =>
        hcop g (Finset.mem_insert_of_mem hg) h (Finset.mem_insert_of_mem hh) hne
      obtain ⟨n0, hn0⟩ := ih hscop r
      have hcopr : Nat.Coprime a (∏ g ∈ s, g) :=
        Nat.Coprime.prod_right fun i hi =>
          hcop a hmemA i (Finset.mem_insert_of_mem hi) (by rintro rfl; exact hasnot hi)
      obtain ⟨v, hv⟩ := exists_inv_of_coprime hcopr
      refine ⟨n0 + ((∏ g ∈ s, g : ℕ) : ℤ) * (v * (r a - n0)), ?_⟩
      intro g hg
      rcases Finset.mem_insert.mp hg with rfl | hg'
      · have hstep : n0 + ((∏ h ∈ s, h : ℕ) : ℤ) * (v * (r g - n0)) - r g
            = (((∏ h ∈ s, h : ℕ) : ℤ) * v - 1) * (r g - n0) := by ring
        rw [hstep]
        exact hv.mul_right _
      · have h1 : (g : ℤ) ∣ n0 - r g := hn0 g hg'
        have h2 : (g : ℤ) ∣ ((∏ h ∈ s, h : ℕ) : ℤ) :=
          Int.natCast_dvd_natCast.mpr (Finset.dvd_prod_of_mem _ hg')
        have hstep : n0 + ((∏ h ∈ s, h : ℕ) : ℤ) * (v * (r a - n0)) - r g
            = (n0 - r g) + ((∏ h ∈ s, h : ℕ) : ℤ) * (v * (r a - n0)) := by ring
        rw [hstep]
        exact dvd_add h1 (h2.mul_right _)

/-- **THE LEMMA, uniqueness.**  Two solutions of the same system agree modulo
the product of the moduli - so the solution of `exists_crt` is unique mod the
wheel. -/
theorem crt_unique : ∀ (G : Finset ℕ),
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ n n' : ℤ,
    (∀ g ∈ G, (g : ℤ) ∣ n - n') → ((∏ g ∈ G, g : ℕ) : ℤ) ∣ n - n' := by
  classical
  intro G
  induction G using Finset.induction_on with
  | empty => intro _ n n' _; simp
  | insert a s hasnot ih =>
      intro hcop n n' h
      have hmemA : a ∈ insert a s := Finset.mem_insert_self a s
      have hscop : ∀ g ∈ s, ∀ h ∈ s, g ≠ h → Nat.Coprime g h := fun g hg h2 hh hne =>
        hcop g (Finset.mem_insert_of_mem hg) h2 (Finset.mem_insert_of_mem hh) hne
      have hcopr : Nat.Coprime a (∏ g ∈ s, g) :=
        Nat.Coprime.prod_right fun i hi =>
          hcop a hmemA i (Finset.mem_insert_of_mem hi) (by rintro rfl; exact hasnot hi)
      have hA : (a : ℤ) ∣ n - n' := h a hmemA
      have hP : ((∏ g ∈ s, g : ℕ) : ℤ) ∣ n - n' :=
        ih hscop n n' fun g hg => h g (Finset.mem_insert_of_mem hg)
      have hiso : IsCoprime ((a : ℕ) : ℤ) ((∏ g ∈ s, g : ℕ) : ℤ) :=
        Nat.isCoprime_iff_coprime.mpr hcopr
      have hmul := hiso.mul_dvd hA hP
      rw [Finset.prod_insert hasnot]
      push_cast at hmul ⊢
      exact hmul

/-! ## 2. A residue avoiding finitely many classes

The per-gear choice that makes the CRT useful for L8: modulo a gear bigger than
the number of forbidden classes there is always a free residue. -/

theorem card_le_four (x y z w : ℤ) : ({x, y, z, w} : Finset ℤ).card ≤ 4 := by
  classical
  have h1 := Finset.card_insert_le x ({y, z, w} : Finset ℤ)
  have h2 := Finset.card_insert_le y ({z, w} : Finset ℤ)
  have h3 := Finset.card_insert_le z ({w} : Finset ℤ)
  have h4 : ({w} : Finset ℤ).card = 1 := Finset.card_singleton w
  omega

/-- Fewer forbidden classes than residues leaves a residue free. -/
theorem exists_avoiding (g : ℕ) (V : Finset ℤ) (hV : V.card < g) :
    ∃ x : ℤ, ∀ v ∈ V, ¬ (g : ℤ) ∣ x - v := by
  classical
  have hg0 : 0 < g := lt_of_le_of_lt (Nat.zero_le _) hV
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg0
  set B : Finset ℕ := V.image (fun v => (v % (g : ℤ)).toNat) with hBdef
  have hBcard : B.card < g := lt_of_le_of_lt Finset.card_image_le hV
  have hex : ∃ r ∈ Finset.range g, r ∉ B := by
    by_contra hc
    have hsub : Finset.range g ⊆ B := by
      intro r hr
      by_contra hrB
      exact hc ⟨r, hr, hrB⟩
    have hle := Finset.card_le_card hsub
    rw [Finset.card_range] at hle
    omega
  obtain ⟨r, hr, hrB⟩ := hex
  have hrg : r < g := Finset.mem_range.mp hr
  refine ⟨(r : ℤ), fun v hv hdvd => hrB ?_⟩
  have hcong : v % (g : ℤ) = (r : ℤ) % (g : ℤ) := Int.modEq_iff_dvd.mpr hdvd
  have hrz : (r : ℤ) % (g : ℤ) = (r : ℤ) := by
    refine Int.emod_eq_of_lt (Int.natCast_nonneg r) ?_
    exact_mod_cast hrg
  have hmod : (v % (g : ℤ)).toNat = r := by
    rw [hcong, hrz]
    simp
  rw [hBdef]
  exact Finset.mem_image.mpr ⟨v, hv, hmod⟩

/-! ## 3. L17, attainment: the domino assignment

`m` gears, `L = 2m - (m mod 2)` positions.  Each gear gets ONE distance-2
domino `{A, A + 2}`, placed by fixing its residue: if `g | n + A + 2` then gear
`g` strikes both position `A` and position `A + 2` of the window at `n`.

The assignment (`anchor`), matching `research/topmachine/r1/cover.py`'s pool
pieces `{x, x + 2}` and its `minpieces` count `ceil(len/2)` per parity chain:

  * the first `ceil(m/2)` gears take the anchors `0, 4, 8, ...`, tiling the
    EVEN positions of `[0, L)`;
  * the remaining `floor(m/2)` gears take `1, 5, 9, ...`, tiling the ODD ones.

Checked before formalising on all 389 gear sets of `m = 1..11` consecutive
primes from `[5, 200)` with `q' > 2m + 1`: the CRT solution `n` makes every
position of `[0, L)` struck - 0 failures. -/

/-- The domino anchor of the `j`-th gear of an `m`-gear set. -/
def anchor (m j : ℕ) : ℕ :=
  if j < (m + 1) / 2 then 4 * j else 4 * (j - (m + 1) / 2) + 1

/-- The assignment covers the whole record window: every position of
`[0, 2m - (m mod 2))` lies on some gear's domino. -/
theorem anchor_covers {m i : ℕ} (hi : i < 2 * m - m % 2) :
    ∃ j, j < m ∧ (i = anchor m j ∨ i = anchor m j + 2) := by
  by_cases hpar : i % 2 = 0
  · refine ⟨i / 4, by omega, ?_⟩
    unfold anchor
    split_ifs with hc <;> omega
  · refine ⟨(m + 1) / 2 + i / 4, by omega, ?_⟩
    unfold anchor
    split_ifs with hc <;> omega

/-- **L17, attainment.**  A gear set of pairwise coprime gears has a run of
`2m - (m mod 2)` consecutive STRUCK pairs: place one domino per gear by CRT.

No size hypothesis is needed for the lower bound - a gear always strikes both
ends of the domino its residue names.  (The matching upper bound `parity_upper`
is where oddness and `g > 2m + 1` are used.) -/
theorem parity_attained {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    ∃ n : ℤ, ∀ i : ℕ, i < 2 * G.card - G.card % 2 → ¬ IsOpen G (n + (i : ℤ)) := by
  classical
  -- the residue demanded of each gear: its domino anchor, read off its index
  have hpick : ∀ h : ℕ, ∃ z : ℤ, ∀ hh : h ∈ G,
      z = -(((anchor G.card ((G.equivFin ⟨h, hh⟩ : Fin G.card)).val : ℕ) : ℤ) + 2) := by
    intro h
    by_cases hh : h ∈ G
    · exact ⟨-(((anchor G.card ((G.equivFin ⟨h, hh⟩ : Fin G.card)).val : ℕ) : ℤ) + 2),
        fun _ => rfl⟩
    · exact ⟨0, fun hh' => absurd hh' hh⟩
  choose r hr using hpick
  obtain ⟨n, hn⟩ := exists_crt G hcop r
  refine ⟨n, fun i hi hopen => ?_⟩
  obtain ⟨j, hj, hij⟩ := anchor_covers hi
  set x := G.equivFin.symm ⟨j, hj⟩ with hxdef
  have hx : G.equivFin x = ⟨j, hj⟩ := Equiv.apply_symm_apply _ _
  have hgj : (x : ℕ) ∈ G := x.2
  have hidx : ((G.equivFin ⟨(x : ℕ), hgj⟩ : Fin G.card)).val = j := by
    have h1 : (⟨(x : ℕ), hgj⟩ : {y // y ∈ G}) = x := rfl
    rw [h1, hx]
  have hdvd := hn (x : ℕ) hgj
  rw [hr (x : ℕ) hgj, hidx] at hdvd
  have hA : ((x : ℕ) : ℤ) ∣ n + ((anchor G.card j : ℕ) : ℤ) + 2 := by
    have hstep : n - -(((anchor G.card j : ℕ) : ℤ) + 2)
        = n + ((anchor G.card j : ℕ) : ℤ) + 2 := by ring
    rwa [hstep] at hdvd
  refine hopen (x : ℕ) hgj ?_
  rcases hij with h | h
  · refine Or.inr ?_
    have hstep : n + (i : ℤ) + 2 = n + ((anchor G.card j : ℕ) : ℤ) + 2 := by rw [h]
    rw [hstep]; exact hA
  · refine Or.inl ?_
    have hstep : n + (i : ℤ) = n + ((anchor G.card j : ℕ) : ℤ) + 2 := by
      rw [h]; push_cast; ring
    rw [hstep]; exact hA

/-- **L17, THE PARITY LAW, as an equality.**  For gears odd, pairwise coprime
and bigger than `2m + 1`, the longest run of consecutive struck pairs is
EXACTLY `2m - (m mod 2)`: `2m` for an even number of gears, `2m - 1` for an odd
number.  Upper bound `parity_upper` (round 32), lower bound `parity_attained`. -/
theorem parity_law {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsOpen G (n + (i : ℤ))}
      (2 * G.card - G.card % 2) := by
  constructor
  · exact parity_attained hcop
  · rintro L ⟨n, hL⟩
    have := parity_upper hodd hbig hL
    omega

/-! ## 4. L8, the exact symmetry group

An affine map `n ↦ c n + b` preserving the open set of `G`.  Three steps:

  1. `symm_not_dvd_mul`: no gear divides `c` (else openness is invariant under
     a shift that the gear can move onto its own tooth);
  2. `isolate` + `affine_gear`: the map preserves EACH GEAR's struck set
     (CRT with an avoidance choice at every other gear);
  3. `affine_teeth` (round 32): per gear, `(c, b) = (1, 0)` or `(-1, -2)`.

Then `sign_count` counts the multipliers: exactly `2 ^ m` residues mod the
wheel, and `exists_symmetry` realises every one of the `2 ^ m` sign vectors. -/

/-- `Strikes g` depends only on the residue mod `g`. -/
theorem strikes_congr {g : ℕ} {x y : ℤ} (h : (g : ℤ) ∣ x - y) :
    Strikes g x ↔ Strikes g y := by
  unfold Strikes
  have h2 : (g : ℤ) ∣ (x + 2) - (y + 2) := by
    have hre : (x + 2) - (y + 2) = x - y := by ring
    rwa [hre]
  rw [dvd_iff_of_dvd_sub h, dvd_iff_of_dvd_sub h2]

/-- **L8, step 1.**  No gear divides the multiplier of a symmetry.  If `g | c`
then `n` and `n + tP` (`P` the product of the OTHER gears) have images congruent
modulo every gear, so openness is invariant under `n ↦ n + tP`; but `P` is
invertible mod `g`, so some `t` slides the always-open shield `-1` onto `g`'s
own tooth. -/
theorem symm_not_dvd_mul {G : Finset ℕ} (h2 : ∀ g ∈ G, 2 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {c b : ℤ}
    (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n)
    {g : ℕ} (hg : g ∈ G) : ¬ (g : ℤ) ∣ c := by
  classical
  intro hgc
  set P : ℕ := ∏ h ∈ G.erase g, h with hPdef
  have hPcop : Nat.Coprime g P :=
    Nat.Coprime.prod_right fun i hi =>
      hcop g hg i (Finset.mem_of_mem_erase hi) (Ne.symm (Finset.ne_of_mem_erase hi))
  have hinv : ∀ (n t : ℤ), IsOpen G (n + t * (P : ℤ)) ↔ IsOpen G n := by
    intro n t
    have hstep : ∀ h ∈ G, (h : ℤ) ∣ (c * (n + t * (P : ℤ)) + b) - (c * n + b) := by
      intro h hh
      have hre : (c * (n + t * (P : ℤ)) + b) - (c * n + b) = c * t * (P : ℤ) := by ring
      rw [hre]
      by_cases hhg : h = g
      · subst hhg
        exact Dvd.dvd.mul_right (Dvd.dvd.mul_right hgc t) _
      · have hdP : (h : ℤ) ∣ (P : ℤ) :=
          Int.natCast_dvd_natCast.mpr
            (Finset.dvd_prod_of_mem _ (Finset.mem_erase.mpr ⟨hhg, hh⟩))
        exact Dvd.dvd.mul_left hdP _
    have h1 : IsOpen G (c * (n + t * (P : ℤ)) + b) ↔ IsOpen G (c * n + b) :=
      forall_congr' fun h => forall_congr' fun hh => not_congr (strikes_congr (hstep h hh))
    rw [← hpres (n + t * (P : ℤ)), ← hpres n]
    exact h1
  obtain ⟨v, hv⟩ := exists_inv_of_coprime hPcop
  have hshield : IsOpen G (-1 + v * (P : ℤ)) := by
    rw [hinv (-1) v]
    exact shield_open h2
  refine hshield g hg (Or.inl ?_)
  have hre : (-1 : ℤ) + v * (P : ℤ) = (P : ℤ) * v - 1 := by ring
  rw [hre]
  exact hv

/-- **L8, step 2 (the isolation).**  Given a target residue `a` mod the gear
`g`, CRT produces an `n` in that class for which EVERY other gear misses both
`n` and its image `c n + b`: at each other gear only four residues are
forbidden (`0`, `-2` for `n`; their two preimages for the image), and the gear
has at least five. -/
theorem isolate {G : Finset ℕ} (h5 : ∀ g ∈ G, 5 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {c b : ℤ}
    (hunit : ∀ g ∈ G, ∃ c' : ℤ, (g : ℤ) ∣ c * c' - 1)
    {g : ℕ} (hg : g ∈ G) (a : ℤ) :
    ∃ N : ℤ, (g : ℤ) ∣ N - a ∧
      ∀ h ∈ G, h ≠ g → ¬ Strikes h N ∧ ¬ Strikes h (c * N + b) := by
  classical
  have hchoice : ∀ h : ℕ, ∃ x : ℤ,
      h ∈ G → h ≠ g → (¬ Strikes h x ∧ ¬ Strikes h (c * x + b)) := by
    intro h
    by_cases hh : h ∈ G
    · obtain ⟨c', hc'⟩ := hunit h hh
      have hcard : ({0, -2, -(c' * b), -(c' * (b + 2))} : Finset ℤ).card < h := by
        have hle := card_le_four (0 : ℤ) (-2) (-(c' * b)) (-(c' * (b + 2)))
        have := h5 h hh
        omega
      obtain ⟨x, hx⟩ := exists_avoiding h _ hcard
      have hx0 : ¬ (h : ℤ) ∣ x - 0 := hx 0 (by simp)
      have hx2 : ¬ (h : ℤ) ∣ x - (-2) := hx (-2) (by simp)
      have hx3 : ¬ (h : ℤ) ∣ x - (-(c' * b)) := hx _ (by simp)
      have hx4 : ¬ (h : ℤ) ∣ x - (-(c' * (b + 2))) := hx _ (by simp)
      refine ⟨x, fun _ _ => ⟨?_, ?_⟩⟩
      · rintro (hs | hs)
        · exact hx0 (by rwa [sub_zero])
        · exact hx2 (by rwa [show x - (-2) = x + 2 by ring])
      · rintro (hs | hs)
        · refine hx3 ?_
          have hd : (h : ℤ) ∣ c' * (c * x + b) := hs.mul_left _
          have hd2 := dvd_sub hd (Dvd.dvd.mul_left hc' x)
          have hre : c' * (c * x + b) - x * (c * c' - 1) = x - (-(c' * b)) := by ring
          rwa [hre] at hd2
        · refine hx4 ?_
          have hd : (h : ℤ) ∣ c' * (c * x + b + 2) := hs.mul_left _
          have hd2 := dvd_sub hd (Dvd.dvd.mul_left hc' x)
          have hre : c' * (c * x + b + 2) - x * (c * c' - 1)
              = x - (-(c' * (b + 2))) := by ring
          rwa [hre] at hd2
    · exact ⟨0, fun h1 _ => absurd h1 hh⟩
  choose ρ hρ using hchoice
  obtain ⟨N, hN⟩ := exists_crt G hcop (fun h => if h = g then a else ρ h)
  refine ⟨N, ?_, ?_⟩
  · have hd := hN g hg
    simpa using hd
  · intro h hh hne
    have hd := hN h hh
    simp only [hne, ite_false] at hd
    obtain ⟨hs1, hs2⟩ := hρ h hh hne
    refine ⟨fun hc => hs1 ((strikes_congr hd).mp hc), fun hc => hs2 ?_⟩
    refine (strikes_congr (?_ : (h : ℤ) ∣ (c * N + b) - (c * ρ h + b))).mp hc
    have hre : (c * N + b) - (c * ρ h + b) = c * (N - ρ h) := by ring
    rw [hre]
    exact hd.mul_left c

/-- **L8, step 2 (the assembly).**  A symmetry of the whole open set preserves
each single gear's struck set. -/
theorem affine_gear {G : Finset ℕ} (h5 : ∀ g ∈ G, 5 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {c b : ℤ}
    (hunit : ∀ g ∈ G, ∃ c' : ℤ, (g : ℤ) ∣ c * c' - 1)
    (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n)
    {g : ℕ} (hg : g ∈ G) (n : ℤ) : Strikes g (c * n + b) ↔ Strikes g n := by
  obtain ⟨N, hNa, hNo⟩ := isolate h5 hcop hunit hg n
  have hopen1 : IsOpen G N ↔ ¬ Strikes g N := by
    constructor
    · intro h; exact h g hg
    · intro h h' hh'
      by_cases he : h' = g
      · subst he; exact h
      · exact (hNo h' hh' he).1
  have hopen2 : IsOpen G (c * N + b) ↔ ¬ Strikes g (c * N + b) := by
    constructor
    · intro h; exact h g hg
    · intro h h' hh'
      by_cases he : h' = g
      · subst he; exact h
      · exact (hNo h' hh' he).2
  have e1 : Strikes g N ↔ Strikes g n := strikes_congr hNa
  have e2 : Strikes g (c * N + b) ↔ Strikes g (c * n + b) := by
    refine strikes_congr ?_
    have hre : (c * N + b) - (c * n + b) = c * (N - n) := by ring
    rw [hre]
    exact hNa.mul_left c
  have hkey := hpres N
  rw [hopen1, hopen2] at hkey
  rw [← e2, ← e1]
  exact not_iff_not.mp hkey

/-- **L8, NECESSITY, the whole gear set.**  Every affine map `n ↦ c n + b` of
`ℤ_W` that preserves the open set of `G` and is invertible modulo each gear has
`(c, b) = (1, 0)` or `(-1, -2)` modulo every gear - i.e. it is
`n ↦ c(n + 1) - 1` with `c = ±1` mod every gear. -/
theorem affine_group_of_unit {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (h5 : ∀ g ∈ G, 5 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) {c b : ℤ}
    (hunit : ∀ g ∈ G, ∃ c' : ℤ, (g : ℤ) ∣ c * c' - 1)
    (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n) :
    ∀ g ∈ G, ((g : ℤ) ∣ c - 1 ∧ (g : ℤ) ∣ b) ∨ ((g : ℤ) ∣ c + 1 ∧ (g : ℤ) ∣ b + 2) := by
  intro g hg
  refine affine_teeth (hodd g hg) ?_ (affine_gear h5 hcop hunit hpres hg)
  intro hgc
  obtain ⟨c', hc'⟩ := hunit g hg
  have hone : (g : ℤ) ∣ 1 := by
    have hd := dvd_sub (Dvd.dvd.mul_right hgc c') hc'
    have hre : c * c' - (c * c' - 1) = 1 := by ring
    rwa [hre] at hd
  have hle : (g : ℤ) ≤ 1 := Int.le_of_dvd (by norm_num) hone
  have h5g : 5 ≤ g := h5 g hg
  have hgz : (5 : ℤ) ≤ (g : ℤ) := by exact_mod_cast h5g
  linarith

/-- **L8, NECESSITY for a set of primes.**  With prime gears the invertibility
hypothesis is not needed: `symm_not_dvd_mul` shows no gear divides `c`, and for
a prime gear that IS invertibility. -/
theorem affine_group {G : Finset ℕ} (hp : ∀ g ∈ G, Nat.Prime g) (h5 : ∀ g ∈ G, 5 ≤ g)
    {c b : ℤ} (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n) :
    ∀ g ∈ G, ((g : ℤ) ∣ c - 1 ∧ (g : ℤ) ∣ b) ∨ ((g : ℤ) ∣ c + 1 ∧ (g : ℤ) ∣ b + 2) := by
  have hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h := fun g hg h hh hne =>
    (Nat.coprime_primes (hp g hg) (hp h hh)).mpr hne
  have hodd : ∀ g ∈ G, g % 2 = 1 := by
    intro g hg
    have hpg := hp g hg
    have h5g := h5 g hg
    have hne2 : g ≠ 2 := by omega
    exact Nat.odd_iff.mp (hpg.odd_of_ne_two hne2)
  have h2 : ∀ g ∈ G, 2 ≤ g := fun g hg => by have := h5 g hg; omega
  have hunit : ∀ g ∈ G, ∃ c' : ℤ, (g : ℤ) ∣ c * c' - 1 := by
    intro g hg
    have hnd : ¬ (g : ℤ) ∣ c := symm_not_dvd_mul h2 hcop hpres hg
    have hnd' : ¬ g ∣ c.natAbs := by
      intro hdd
      refine hnd ?_
      have hab : ((g : ℤ)).natAbs ∣ c.natAbs := by rwa [Int.natAbs_natCast]
      exact Int.natAbs_dvd_natAbs.mp hab
    have hgc : Nat.gcd g c.natAbs = 1 := (Nat.Prime.coprime_iff_not_dvd (hp g hg)).mpr hnd'
    have hiso : IsCoprime ((g : ℕ) : ℤ) c := by
      rw [Int.isCoprime_iff_gcd_eq_one]
      simpa [Int.gcd, Int.natAbs_natCast] using hgc
    exact exists_inv_of_isCoprime hiso
  exact affine_group_of_unit hodd h5 hcop hunit hpres

/-- **L8, the shape of the group.**  The branch's form: every symmetry is
`n ↦ c(n + 1) - 1` (that is, `b = c - 1` modulo every gear) with `c = ±1`
modulo every gear. -/
theorem affine_group_form {G : Finset ℕ} (hp : ∀ g ∈ G, Nat.Prime g) (h5 : ∀ g ∈ G, 5 ≤ g)
    {c b : ℤ} (hpres : ∀ n : ℤ, IsOpen G (c * n + b) ↔ IsOpen G n) :
    (∀ g ∈ G, (g : ℤ) ∣ c - 1 ∨ (g : ℤ) ∣ c + 1) ∧ (∀ g ∈ G, (g : ℤ) ∣ b - (c - 1)) := by
  have h := affine_group hp h5 hpres
  constructor
  · intro g hg
    rcases h g hg with ⟨h1, _⟩ | ⟨h1, _⟩
    · exact Or.inl h1
    · exact Or.inr h1
  · intro g hg
    rcases h g hg with ⟨h1, hb⟩ | ⟨h1, hb⟩
    · exact dvd_sub hb h1
    · have hd := dvd_sub hb h1
      rwa [show (b + 2) - (c + 1) = b - (c - 1) by ring] at hd

/-! ## 5. L8, the count: exactly `2 ^ m` symmetries

The multipliers are the residues `c` mod the wheel with `c = ±1` modulo every
gear.  There are two per gear (`1` and `g - 1`, distinct once `g ≥ 3`), so
`2 ^ m` in all - the same CRT count that gives `wheel_count`, run on the sign
predicate instead of the open predicate. -/

/-- `c = ±1` modulo `g`, in residue form. -/
def SignR (g n : ℕ) : Prop := n % g = 1 ∨ (n + 1) % g = 0

/-- `c = ±1` modulo every gear. -/
def SignsN (G : Finset ℕ) (n : ℕ) : Prop := ∀ g ∈ G, SignR g n

instance decSignR (g n : ℕ) : Decidable (SignR g n) := by unfold SignR; infer_instance

instance decSignsN (G : Finset ℕ) (n : ℕ) : Decidable (SignsN G n) := by
  unfold SignsN; infer_instance

theorem signR_congr {g x y : ℕ} (h : x % g = y % g) : SignR g x ↔ SignR g y := by
  unfold SignR
  rw [Nat.add_mod x 1 g, Nat.add_mod y 1 g, h]

theorem signsN_congr {G : Finset ℕ} {W x y : ℕ} (hdvd : ∀ g ∈ G, g ∣ W)
    (h : x % W = y % W) : SignsN G x ↔ SignsN G y := by
  unfold SignsN
  refine forall_congr' fun g => forall_congr' fun hg => signR_congr ?_
  rw [← Nat.mod_mod_of_dvd x (hdvd g hg), ← Nat.mod_mod_of_dvd y (hdvd g hg), h]

theorem signsN_insert {a : ℕ} {s : Finset ℕ} {n : ℕ} :
    SignsN (insert a s) n ↔ (SignR a n ∧ SignsN s n) := by
  constructor
  · intro h
    exact ⟨h a (Finset.mem_insert_self a s), fun g hg => h g (Finset.mem_insert_of_mem hg)⟩
  · rintro ⟨h1, h2⟩ g hg
    rcases Finset.mem_insert.mp hg with rfl | hg'
    · exact h1
    · exact h2 g hg'

/-- Modulo one gear there are exactly two signs: `1` and `g - 1`. -/
theorem sign_residues {g : ℕ} (hg : 3 ≤ g) :
    (Finset.range g).filter (fun r => SignR g r) = ({1, g - 1} : Finset ℕ) := by
  ext r
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton]
  constructor
  · rintro ⟨hrg, hs⟩
    unfold SignR at hs
    rcases hs with hs | hs
    · left; rwa [Nat.mod_eq_of_lt hrg] at hs
    · right
      obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero hs
      have hk2 : k < 2 := by
        by_contra hc
        have hmul : g * 2 ≤ g * k := Nat.mul_le_mul_left g (by omega)
        omega
      interval_cases k <;> omega
  · intro h
    have hrg : r < g := by omega
    refine ⟨hrg, ?_⟩
    unfold SignR
    rcases h with rfl | rfl
    · left; exact Nat.mod_eq_of_lt (by omega)
    · right
      have hgg : g - 1 + 1 = g := by omega
      rw [hgg, Nat.mod_self]

theorem card_sign_residues {g : ℕ} (hg : 3 ≤ g) :
    ((Finset.range g).filter (fun r => SignR g r)).card = 2 := by
  rw [sign_residues hg, Finset.card_insert_of_notMem (by simp; omega), Finset.card_singleton]

/-- **L8, THE COUNT.**  Exactly `2 ^ m` residues mod the wheel are `±1` modulo
every gear: the symmetry group of the top machine is `(Z/2)^m`. -/
theorem sign_count : ∀ (G : Finset ℕ), (∀ g ∈ G, 3 ≤ g) →
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) →
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => SignsN G n)).card = 2 ^ G.card := by
  classical
  intro G
  induction G using Finset.induction_on with
  | empty => intro _ _; simp [SignsN]
  | insert a s hasnot ih =>
      intro hG3 hcop
      have hmemA : a ∈ insert a s := Finset.mem_insert_self a s
      have ha3 : 3 ≤ a := hG3 a hmemA
      have ha0 : 0 < a := by omega
      have hs3 : ∀ g ∈ s, 3 ≤ g := fun g hg => hG3 g (Finset.mem_insert_of_mem hg)
      have hscop : ∀ g ∈ s, ∀ h ∈ s, g ≠ h → Nat.Coprime g h := fun g hg h hh hne =>
        hcop g (Finset.mem_insert_of_mem hg) h (Finset.mem_insert_of_mem hh) hne
      have hP0 : 0 < ∏ g ∈ s, g := by
        have h1 : 1 ≤ ∏ g ∈ s, g :=
          Finset.one_le_prod' fun i hi => by have := hs3 i hi; omega
        omega
      have hcopr : Nat.Coprime a (∏ g ∈ s, g) :=
        Nat.Coprime.prod_right fun i hi =>
          hcop a hmemA i (Finset.mem_insert_of_mem hi) (by rintro rfl; exact hasnot hi)
      have hdvd : ∀ g ∈ s, g ∣ (∏ g ∈ s, g) := fun g hg => Finset.dvd_prod_of_mem _ hg
      rw [Finset.prod_insert hasnot, Finset.card_insert_of_notMem hasnot]
      have hfil : (Finset.range (a * ∏ g ∈ s, g)).filter (fun n => SignsN (insert a s) n)
          = (Finset.range (a * ∏ g ∈ s, g)).filter
              (fun n => SignR a n ∧ SignsN s n) :=
        Finset.filter_congr fun n _ => signsN_insert
      rw [hfil,
        card_filter_crt ha0 hP0 hcopr (SignR a) (SignsN s)
          (fun x y h => signR_congr h)
          (fun x y h => signsN_congr hdvd h),
        card_sign_residues ha3, ih hs3 hscop]
      ring

/-- The residue form of the sign condition is the divisibility form. -/
theorem signR_iff_dvd {g n : ℕ} (hg : 2 ≤ g) :
    SignR g n ↔ ((g : ℤ) ∣ (n : ℤ) - 1 ∨ (g : ℤ) ∣ (n : ℤ) + 1) := by
  unfold SignR
  have hgz : (2 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hg
  constructor
  · rintro (h | h)
    · left
      have hn1 : 1 ≤ n := by
        rcases Nat.eq_zero_or_pos n with rfl | hp
        · rw [Nat.zero_mod] at h; omega
        · exact hp
      have hdn : g ∣ n - 1 := by
        refine ⟨n / g, ?_⟩
        have hdm := Nat.div_add_mod n g
        rw [h] at hdm
        omega
      have hcast : ((n - 1 : ℕ) : ℤ) = (n : ℤ) - 1 := by
        push_cast [Nat.cast_sub hn1]; ring
      rw [← hcast]
      exact Int.natCast_dvd_natCast.mpr hdn
    · right
      have hdn : g ∣ (n + 1) := Nat.dvd_of_mod_eq_zero h
      have hcast : ((n + 1 : ℕ) : ℤ) = (n : ℤ) + 1 := by push_cast; ring
      rw [← hcast]
      exact Int.natCast_dvd_natCast.mpr hdn
  · rintro (h | h)
    · left
      have h1 : (1 : ℤ) % (g : ℤ) = (n : ℤ) % (g : ℤ) := Int.modEq_iff_dvd.mpr h
      have h2 : (1 : ℤ) % (g : ℤ) = 1 := Int.emod_eq_of_lt (by norm_num) (by linarith)
      have h3 : ((n % g : ℕ) : ℤ) = 1 := by push_cast; rw [← h1]; exact h2
      exact_mod_cast h3
    · right
      have hd : (g : ℤ) ∣ ((n + 1 : ℕ) : ℤ) := by push_cast; exact h
      have hdn : g ∣ (n + 1) := Int.natCast_dvd_natCast.mp hd
      rwa [Nat.dvd_iff_mod_eq_zero] at hdn

/-- **L8, REALISABILITY.**  Every one of the `2 ^ m` sign vectors is realised by
an actual symmetry `n ↦ c(n + 1) - 1` of the open set (CRT for the multiplier,
`open_affine` for the symmetry). -/
theorem exists_symmetry {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (ε : ℕ → ℤ)
    (hε : ∀ g ∈ G, ε g = 1 ∨ ε g = -1) :
    ∃ c : ℤ, (∀ g ∈ G, (g : ℤ) ∣ c - ε g) ∧
      ∀ n : ℤ, IsOpen G (c * (n + 1) - 1) ↔ IsOpen G n := by
  obtain ⟨c, hc⟩ := exists_crt G hcop ε
  have hpm : ∀ g ∈ G, (g : ℤ) ∣ c - 1 ∨ (g : ℤ) ∣ c + 1 := by
    intro g hg
    have hd := hc g hg
    rcases hε g hg with h | h
    · left; rwa [h] at hd
    · right
      rw [h] at hd
      rwa [show c - (-1 : ℤ) = c + 1 by ring] at hd
  exact ⟨c, hc, fun n => open_affine hpm⟩

end TopMachine

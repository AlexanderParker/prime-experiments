/-
MirrorWalk: the mirror walk's steps as theorems (proof_skeleton.md Part V, 2026-09-13).

A flip about the axis `a` sends the column `(n, n+2)` to `(2a - n - 2, 2a - n)`.  A gear `g`
dividing `2a` (the axis product) keeps the openness of the column: `g` divides a member of the
column iff it divides a member of the image.  A walk of flips composes to a flip about the
alternating sum of the axes (odd length) or a slide by twice it (even length), so the end of a
walk is certified for exactly the gears dividing that sum.  From home `(-1, 1)` every walk lands
on `(2A - 1, 2A + 1)`; with `2A = 12m` the landing is the column `2m` of the line, and the
record route of SquareColumn (S0) makes it a twin prime pair whenever the gears not dividing
the axis miss it and the landing lies below the square of a prime `P` with every prime of
`[5, P)` in the gear set.
-/
import SquareColumn
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace MirrorWalk

open SquareColumn

/-- The flip of the column `n` about the axis `a`: the left member of the mirror image. -/
def flip (a n : ℤ) : ℤ := 2 * a - n - 2

/-- Column `n` is open to `g`: `g` divides neither member. -/
def OpenTo (g : ℕ) (n : ℤ) : Prop := ¬ (g : ℤ) ∣ n ∧ ¬ (g : ℤ) ∣ n + 2

theorem flip_flip (a b n : ℤ) : flip a (flip b n) = n + 2 * (a - b) := by
  unfold flip; ring

theorem flip_home (a : ℤ) : flip a (-1) = 2 * a - 1 := by
  unfold flip; ring

private theorem dvd_sub_iff_of_dvd {g x y : ℤ} (h : g ∣ x) : g ∣ x - y ↔ g ∣ y :=
  ⟨fun hd => by simpa using dvd_sub h hd, fun hd => dvd_sub h hd⟩

/-- **The axis rule.**  A gear dividing the axis product `2a` divides a member of the column
`n` iff it divides a member of the flipped column. -/
theorem flip_carries {g : ℕ} {a n : ℤ} (h : (g : ℤ) ∣ 2 * a) :
    ((g : ℤ) ∣ n ∨ (g : ℤ) ∣ n + 2) ↔ ((g : ℤ) ∣ flip a n ∨ (g : ℤ) ∣ flip a n + 2) := by
  have e1 : flip a n = 2 * a - (n + 2) := by unfold flip; ring
  have e2 : flip a n + 2 = 2 * a - n := by unfold flip; ring
  rw [e2, e1, dvd_sub_iff_of_dvd h, dvd_sub_iff_of_dvd h]
  exact or_comm

/-- The axis rule in the language of openness. -/
theorem openTo_flip_iff {g : ℕ} {a n : ℤ} (h : (g : ℤ) ∣ 2 * a) :
    OpenTo g (flip a n) ↔ OpenTo g n := by
  unfold OpenTo
  have := flip_carries (n := n) h
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun hd => ?_, fun hd => ?_⟩
    · rcases this.mp (Or.inl hd) with h' | h' <;> contradiction
    · rcases this.mp (Or.inr hd) with h' | h' <;> contradiction
  · rintro ⟨h1, h2⟩
    refine ⟨fun hd => ?_, fun hd => ?_⟩
    · rcases this.mpr (Or.inl hd) with h' | h' <;> contradiction
    · rcases this.mpr (Or.inr hd) with h' | h' <;> contradiction

/-- A walk: flips about the axes in order (first axis first). -/
def walk : List ℤ → ℤ → ℤ
  | [], n => n
  | a :: rest, n => walk rest (flip a n)

/-- The alternating sum of the axes, last axis positive. -/
def alt : List ℤ → ℤ
  | [] => 0
  | a :: rest => if rest.length % 2 = 0 then a + alt rest else alt rest - a

/-- **The composition law.**  A walk of odd length is the flip about the alternating sum; a
walk of even length is the slide by twice it. -/
theorem walk_eq (as : List ℤ) (n : ℤ) :
    walk as n = if as.length % 2 = 0 then n + 2 * alt as else flip (alt as) n := by
  induction as generalizing n with
  | nil => simp [walk, alt]
  | cons a rest ih =>
    simp only [walk, alt, List.length_cons]
    rw [ih]
    by_cases hr : rest.length % 2 = 0
    · have hl : (rest.length + 1) % 2 ≠ 0 := by omega
      simp only [hr, hl, ite_true, ite_false]
      unfold flip; ring
    · have hl : (rest.length + 1) % 2 = 0 := by omega
      simp only [hr, hl, ite_true, ite_false]
      unfold flip; ring

/-- The even case is two flips: about `0` then about the alternating sum. -/
theorem slide_eq_two_flips (S n : ℤ) : n + 2 * S = flip S (flip 0 n) := by
  unfold flip; ring

/-- **What a walk carries.**  A gear dividing twice the alternating sum keeps the openness of
the column across the whole walk, whatever the intermediate landings. -/
theorem openTo_walk_iff {g : ℕ} (as : List ℤ) {n : ℤ} (h : (g : ℤ) ∣ 2 * alt as) :
    OpenTo g (walk as n) ↔ OpenTo g n := by
  rw [walk_eq]
  by_cases he : as.length % 2 = 0
  · simp only [he, ite_true]
    rw [slide_eq_two_flips, openTo_flip_iff h, openTo_flip_iff (by simp)]
  · simp only [he, ite_false]
    exact openTo_flip_iff h

/-- From home `(-1, 1)` a walk of odd length lands on `(2A - 1, 2A + 1)`, `A` the alternating
sum. -/
theorem walk_home_odd (as : List ℤ) (hodd : as.length % 2 ≠ 0) :
    walk as (-1) = 2 * alt as - 1 := by
  rw [walk_eq]; simp only [hodd, ite_false]; exact flip_home _

/-- And a walk of even length lands on `(2A - 1, 2A + 1)` as well. -/
theorem walk_home_even (as : List ℤ) (heven : as.length % 2 = 0) :
    walk as (-1) = 2 * alt as - 1 := by
  rw [walk_eq]; simp only [heven, ite_true]; ring

/-- Home is open to every gear `g ≥ 2`. -/
theorem home_open {g : ℕ} (hg : 2 ≤ g) : OpenTo g (-1) := by
  unfold OpenTo
  constructor
  · intro hd
    have h1 : (g : ℤ) ∣ 1 := by simpa using (dvd_neg.mp hd)
    have := Int.eq_one_of_dvd_one (by positivity) h1
    omega
  · intro hd
    have h1 : (g : ℤ) ∣ 1 := by simpa using hd
    have := Int.eq_one_of_dvd_one (by positivity) h1
    omega

/-- **The certified gears.**  At the end of any walk from home, every gear `g ≥ 2` dividing
twice the alternating sum finds the landing open. -/
theorem landing_open_of_dvd {g : ℕ} (hg : 2 ≤ g) (as : List ℤ) (h : (g : ℤ) ∣ 2 * alt as) :
    OpenTo g (walk as (-1)) :=
  (openTo_walk_iff as h).mpr (home_open hg)

/-! ## The landing on the line: column `2m`, the m-line -/

/-- The gears of a column struck on the m-line: `StruckBy G (2m)` says some gear of `G` divides
`12m - 1` or `12m + 1`. -/
theorem struckBy_mline {G : Finset ℕ} {m : ℕ} :
    StruckBy G (2 * m) ↔ ∃ g ∈ G, g ∣ 12 * m - 1 ∨ g ∣ 12 * m + 1 := by
  unfold StruckBy
  constructor
  · rintro ⟨g, hg, h⟩
    refine ⟨g, hg, ?_⟩
    have e1 : 6 * (2 * m) - 1 = 12 * m - 1 := by omega
    have e2 : 6 * (2 * m) + 1 = 12 * m + 1 := by omega
    rw [e1, e2] at h; exact h
  · rintro ⟨g, hg, h⟩
    refine ⟨g, hg, ?_⟩
    have e1 : 6 * (2 * m) - 1 = 12 * m - 1 := by omega
    have e2 : 6 * (2 * m) + 1 = 12 * m + 1 := by omega
    rw [e1, e2]; exact h

/-- A gear dividing the axis product `12m` misses both members `12m ± 1` (they are adjacent to
a multiple). -/
theorem not_dvd_landing_of_dvd_axis {g m : ℕ} (hg : 2 ≤ g) (hm : 1 ≤ m) (h : g ∣ 12 * m) :
    ¬ g ∣ 12 * m - 1 ∧ ¬ g ∣ 12 * m + 1 := by
  constructor
  · intro hd
    have h1 : g ∣ 12 * m - (12 * m - 1) := Nat.dvd_sub h hd
    have e : 12 * m - (12 * m - 1) = 1 := by omega
    rw [e] at h1
    have := Nat.eq_one_of_dvd_one h1
    omega
  · intro hd
    have h1 : g ∣ 12 * m + 1 - 12 * m := Nat.dvd_sub hd h
    have e : 12 * m + 1 - 12 * m = 1 := by omega
    rw [e] at h1
    have := Nat.eq_one_of_dvd_one h1
    omega

/-- **The landing lemma** (S0 on the m-line).  If the landing `(12m - 1, 12m + 1)` lies below
`P^2`, the gear set contains every prime of `[5, P)`, and every gear of `G` that does not divide
the axis product `12m` misses both members, then the landing is a twin prime pair.  The gears
dividing `12m` need no check: the mirror carries them. -/
theorem landing_twin {G : Finset ℕ} {P m : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (hG : ∀ g ∈ G, 2 ≤ g)
    (hm : 1 ≤ m) (hlt : 12 * m + 1 < P ^ 2)
    (hroots : ∀ g ∈ G, ¬ g ∣ 12 * m → ¬ g ∣ 12 * m - 1 ∧ ¬ g ∣ 12 * m + 1) :
    (12 * m - 1).Prime ∧ (12 * m + 1).Prime := by
  have hk : 1 ≤ 2 * m := by omega
  have hlt' : 6 * (2 * m) + 1 < P ^ 2 := by omega
  have hun : ¬ StruckBy G (2 * m) := by
    rw [struckBy_mline]
    rintro ⟨g, hg, h⟩
    by_cases hd : g ∣ 12 * m
    · have := not_dvd_landing_of_dvd_axis (hG g hg) hm hd
      rcases h with h | h
      · exact this.1 h
      · exact this.2 h
    · have := hroots g hg hd
      rcases h with h | h
      · exact this.1 h
      · exact this.2 h
  have := section_twin_of_unstruck hfull hk hlt' hun
  have e1 : 6 * (2 * m) - 1 = 12 * m - 1 := by omega
  have e2 : 6 * (2 * m) + 1 = 12 * m + 1 := by omega
  rw [e1, e2] at this
  exact this

/-- **Termination from the record on the m-line.**  If every run of `F + 1` consecutive
columns of the m-line holds a column unstruck by `G` (the m-line record is at most `F`), and the
stretch of `m` in `[a, a + l)` with `F + 1 ≤ l` lies below `P^2` (`12 (a + l) + 1 ≤ P^2`), then
some `m` in the stretch lands on a twin prime pair. -/
theorem walk_lands_of_record {G : Finset ℕ} {P a l F : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) (ha : 1 ≤ a)
    (hlt : 12 * (a + l) + 1 ≤ P ^ 2) (hF : ∀ x, ∃ j, j ≤ F ∧ ¬ StruckBy G (2 * (x + j)))
    (hl : F + 1 ≤ l) :
    ∃ m, a ≤ m ∧ m < a + l ∧ (12 * m - 1).Prime ∧ (12 * m + 1).Prime := by
  obtain ⟨j, hj, hun⟩ := hF a
  refine ⟨a + j, by omega, by omega, ?_⟩
  have hk : 1 ≤ 2 * (a + j) := by omega
  have hlt' : 6 * (2 * (a + j)) + 1 < P ^ 2 := by
    have : 12 * (a + j) + 1 < 12 * (a + l) + 1 := by omega
    omega
  have := section_twin_of_unstruck hfull hk hlt' hun
  have e1 : 6 * (2 * (a + j)) - 1 = 12 * (a + j) - 1 := by omega
  have e2 : 6 * (2 * (a + j)) + 1 = 12 * (a + j) + 1 := by omega
  rw [e1, e2] at this
  exact this

end MirrorWalk

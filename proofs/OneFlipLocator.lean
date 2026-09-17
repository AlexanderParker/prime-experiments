/-
OneFlipLocator (rounds 60-62, 2026-09-17): the construction in its final form.

Sixty-odd rounds of search over walks ended where they began: the best construction is one flip
from home.  The flip about the mirror `{2, 3, …g}` with `k` periods sends the home pair
`(-1, +1)` to

    (12 g k d - 1,  12 g k d + 1),      column `m = 2 g k d`,   `d = ±1`,

which carries the phases of exactly the gears dividing the mirror; the machine's own gears
decide whether that column is open, and the square-root rule turns an open column below `q²`
into a twin prime pair.

  `OneFlipOpen G g K P`: some column `2 g k d`, `k = 1 … K`, `d = ±1`, lies in the window
  `(P, P²]` and is struck by no gear of `G`.

  `oneflip_twin`: with `G` holding every prime from 5 below `P`, `OneFlipOpen` gives a twin
  prime pair in the window.

The teeth law below says how rigid the family is.  A gear dividing the mirror never strikes it
at all (`mirror_gear_never_strikes`): it would have to divide 1.  Every other gear `h` strikes
at exactly two periods, `k ≡ v` and `k ≡ -v` modulo `h`, where `v` inverts the stride `12 g`
(`oneflip_teeth`) - a symmetric pair about `k = 0`, which is home itself.

Measured (research/stack/r8/oneflip_symmetric_teeth.py, round 62).  The mirror to use is the
smallest one that fits, `{2, 3, 5}` (stride 60): it leaves the most open columns at every
machine tested (72 of 638 at `q = 20011`, against 41 for the mirror at the first gear above
`√q`), and no machine from 11 to 20011 is without one.  A larger mirror carries more gear
phases but spaces its candidates further apart, so they land past the twins - the trade lemma
(`mirror_times_candidates`) read on the mirror instead of the walk.  The periods needed are
those of the window's own start, `k ≈ q / 60`, plus a small offset: the first open period sits
0 to 34 past the start at every machine measured to 2000003.
-/
import MirrorWalkConditional

namespace MirrorWalk

open SquareColumn

/-- The column a single flip about `{2, 3, …g}` reaches from home in `k` periods: the flip sends
the pair `(-1, +1)` to `(12 g k d - 1, 12 g k d + 1)`, whose column is `2 g k d`. -/
def oneFlip (g : ℤ) (k : ℕ) (d : ℤ) : ℤ := 2 * g * k * d

/-- The members of that column are the stride's multiple either side of 1. -/
theorem oneFlip_members (g : ℤ) (k : ℕ) (d : ℤ) :
    6 * oneFlip g k d - 1 = 12 * g * k * d - 1 ∧ 6 * oneFlip g k d + 1 = 12 * g * k * d + 1 := by
  unfold oneFlip; constructor <;> ring

/-- **The one-flip hypothesis.**  Some column of the family is a column of the window that no
gear of `G` strikes. -/
def OneFlipOpen (G : Finset ℕ) (g : ℤ) (K P : ℕ) : Prop :=
  ∃ (k : ℕ) (d : ℤ) (m : ℕ), 1 ≤ k ∧ k ≤ K ∧ (d = 1 ∨ d = -1) ∧
    (m : ℤ) = oneFlip g k d ∧ 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧
    ∀ h ∈ G, ¬ (h ∣ 6 * m - 1) ∧ ¬ (h ∣ 6 * m + 1)

/-- **The locator's theorem.**  With `G` holding every prime from 5 below `P`, the one-flip
hypothesis gives a twin prime pair inside the window `(P, P²]`. -/
theorem oneflip_twin {G : Finset ℕ} {g : ℤ} {K P : ℕ}
    (hfull : ∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G)
    (hopen : OneFlipOpen G g K P) :
    ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  obtain ⟨k, d, m, hk1, hkK, hd, hcast, hm1, hmlt, hstruck⟩ := hopen
  have hns : ¬ StruckBy G m := by
    rintro ⟨h, hh, hdvd | hdvd⟩
    · exact (hstruck h hh).1 hdvd
    · exact (hstruck h hh).2 hdvd
  obtain ⟨hp1, hp2⟩ := section_twin_of_unstruck hfull hm1 hmlt hns
  exact ⟨m, hm1, hmlt, hp1, hp2⟩

/-- **The window statement, from the locator.**  If every machine's one-flip family holds an
open column, every machine's window holds a twin prime pair. -/
theorem window_statement_of_oneflip
    (H : ∀ (P : ℕ), P.Prime → 5 ≤ P → ∃ (G : Finset ℕ) (g : ℤ) (K : ℕ),
        (∀ r, r.Prime → 5 ≤ r → r < P → r ∈ G) ∧ OneFlipOpen G g K P) :
    ∀ (P : ℕ), P.Prime → 5 ≤ P →
      ∃ m : ℕ, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime := by
  intro P hP h5
  obtain ⟨G, g, K, hfull, hopen⟩ := H P hP h5
  exact oneflip_twin hfull hopen

/-- **A gear striking a scaled line.**  With `u` an inverse of the stride `s` modulo `h`, the
gear `h` divides `s * k - c` exactly when `k ≡ c * u`. -/
theorem strike_iff_scaled {h : ℕ} {s u c k : ℤ} (hu : s * u ≡ 1 [ZMOD h]) :
    (h : ℤ) ∣ s * k - c ↔ k ≡ c * u [ZMOD h] := by
  constructor
  · intro hd
    have h0 : s * k ≡ c [ZMOD h] := Int.ModEq.symm (Int.modEq_iff_dvd.mpr (by simpa using hd))
    have h1 : k * (s * u) ≡ k * 1 [ZMOD h] := hu.mul_left k
    have h2 : k * (s * u) = (s * k) * u := by ring
    have h3 : (s * k) * u ≡ c * u [ZMOD h] := h0.mul_right u
    have h4 : k * 1 ≡ c * u [ZMOD h] := by
      have h5 : k * 1 ≡ (s * k) * u [ZMOD h] := by rw [← h2]; exact h1.symm
      exact h5.trans h3
    rwa [mul_one] at h4
  · intro hk
    have h1 : s * k ≡ s * (c * u) [ZMOD h] := hk.mul_left s
    have h3 : s * (c * u) ≡ c [ZMOD h] := by
      have h3' : c * (s * u) ≡ c * 1 [ZMOD h] := hu.mul_left c
      calc s * (c * u) = c * (s * u) := by ring
        _ ≡ c * 1 [ZMOD h] := h3'
        _ = c := by ring
    exact Int.ModEq.dvd (Int.ModEq.symm (h1.trans h3))

/-- **The teeth of a gear on the one-flip family.**  A gear `h` whose inverse of the stride
`12 g` is `v` strikes the family at exactly two periods, `k ≡ v` and `k ≡ -v`: a symmetric pair
about home. -/
theorem oneflip_teeth {h : ℕ} {g v k : ℤ} (hv : (12 * g) * v ≡ 1 [ZMOD h]) :
    ((h : ℤ) ∣ 12 * g * k - 1 ↔ k ≡ v [ZMOD h]) ∧
    ((h : ℤ) ∣ 12 * g * k + 1 ↔ k ≡ -v [ZMOD h]) := by
  constructor
  · have := strike_iff_scaled (s := 12 * g) (u := v) (c := 1) (k := k) hv
    simpa using this
  · have := strike_iff_scaled (s := 12 * g) (u := v) (c := -1) (k := k) hv
    have e : 12 * g * k - (-1) = 12 * g * k + 1 := by ring
    rw [e] at this
    simpa using this

/-- **The mirror's own gears never strike.**  A gear dividing the stride misses the whole family:
it would have to divide 1. -/
theorem mirror_gear_never_strikes {h : ℕ} (hh : 1 < h) {s k : ℤ} (hdvd : (h : ℤ) ∣ s) :
    ¬ (h : ℤ) ∣ s * k - 1 ∧ ¬ (h : ℤ) ∣ s * k + 1 := by
  have hk : (h : ℤ) ∣ s * k := hdvd.mul_right k
  have h1 : (1 : ℤ) < (h : ℤ) := by exact_mod_cast hh
  constructor
  · intro hd
    have : (h : ℤ) ∣ 1 := by
      have := hk.sub hd
      simpa using this
    have := Int.le_of_dvd one_pos this
    omega
  · intro hd
    have : (h : ℤ) ∣ 1 := by
      have := hd.sub hk
      simpa using this
    have := Int.le_of_dvd one_pos this
    omega

end MirrorWalk

/-
RigidShift (round 112, 2026-09-19): the re-phasing adversary never leaves the machine.

Entries 117-119 measure the machine against an adversary who may shift each gear's rigid tooth
pair (the classes `s + u` and `s - u` modulo `g`, `u = 6⁻¹ mod g`) to any phase `s` it likes.  By
the Chinese remainder theorem every such shift vector is realised by ONE window position `x` of
the real pattern: relative to `x`, gear `g` strikes offset `i` at its real tooth `t` exactly when
`i ≡ s g + t (mod g)`.  So the best run over all shift vectors is the machine's own record
`F(M)`, and every rigid killer the adversary finds is a real window of the real pattern
somewhere else in its period.
-/
import Mathlib.Data.Nat.ChineseRemainder
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.Ring

namespace RigidShift

/-- **Every shift vector is a window of the real pattern.**  For distinct primes `G` and any
shifts `s`, there is a window position `x` such that, for every gear `g ∈ G`, every tooth offset
`t` and every relative column `i`: the real tooth `t` strikes the column `x + i` iff the shifted
tooth `s g + t` strikes the relative column `i`. -/
theorem window_realises_shift (G : Finset ℕ) (hG : ∀ g ∈ G, g.Prime) (s : ℕ → ℕ) :
    ∃ x : ℕ, ∀ g ∈ G, ∀ i t : ℕ, (x + i ≡ t [MOD g] ↔ i ≡ s g + t [MOD g]) := by
  classical
  have hs : ∀ g ∈ G, id g ≠ 0 := fun g hg => (hG g hg).ne_zero
  have pp : Set.Pairwise (↑G : Set ℕ) (Function.onFun Nat.Coprime id) := by
    intro i hi j hj hij
    exact (Nat.coprime_primes (hG i hi) (hG j hj)).mpr hij
  obtain ⟨x, hx⟩ := Nat.chineseRemainderOfFinset (fun g => (g - 1) * s g) id G hs pp
  refine ⟨x, ?_⟩
  intro g hg i t
  have hx' : x ≡ (g - 1) * s g [MOD g] := hx g hg
  have hg1 : 1 ≤ g := (hG g hg).one_lt.le
  have e : (g - 1) * s g + s g = g * s g := by
    obtain ⟨k, rfl⟩ : ∃ k, g = k + 1 := ⟨g - 1, by omega⟩
    rw [Nat.add_sub_cancel]
    ring
  have h0 : x + s g ≡ 0 [MOD g] := by
    calc x + s g ≡ (g - 1) * s g + s g [MOD g] := Nat.ModEq.add_right _ hx'
      _ = g * s g := e
      _ ≡ 0 [MOD g] := Nat.modEq_zero_iff_dvd.mpr ⟨s g, rfl⟩
  constructor
  · intro h
    have h1 : x + i + s g ≡ t + s g [MOD g] := Nat.ModEq.add_right _ h
    have h2 : x + i + s g ≡ i [MOD g] := by
      calc x + i + s g = (x + s g) + i := by ring
        _ ≡ 0 + i [MOD g] := Nat.ModEq.add_right _ h0
        _ = i := zero_add i
    calc i ≡ x + i + s g [MOD g] := h2.symm
      _ ≡ t + s g [MOD g] := h1
      _ = s g + t := add_comm _ _
  · intro h
    calc x + i ≡ x + (s g + t) [MOD g] := Nat.ModEq.add_left _ h
      _ = (x + s g) + t := by ring
      _ ≡ 0 + t [MOD g] := Nat.ModEq.add_right _ h0
      _ = t := zero_add t

/-- The rigid pair of gear `g` at shift `s`: the relative columns `i ≡ s + u` and `i ≡ s + (g - u)`
modulo `g`, `u` the gear's tooth (`6⁻¹ mod g` on the machine). -/
def PairStrikes (g s u i : ℕ) : Prop := i ≡ s + u [MOD g] ∨ i ≡ s + (g - u) [MOD g]

/-- **The shifted machine is the real machine at another window**: the real pair of every gear,
read from window `x`, is the shifted pair. -/
theorem shifted_pattern_is_window (G : Finset ℕ) (hG : ∀ g ∈ G, g.Prime) (s u : ℕ → ℕ) :
    ∃ x : ℕ, ∀ g ∈ G, ∀ i : ℕ, PairStrikes g 0 (u g) (x + i) ↔ PairStrikes g (s g) (u g) i := by
  obtain ⟨x, hx⟩ := window_realises_shift G hG s
  refine ⟨x, fun g hg i => ?_⟩
  unfold PairStrikes
  simp only [zero_add]
  rw [hx g hg i (u g), hx g hg i (g - u g)]

end RigidShift

/-
ManyBody (round 89, 2026-09-18): the interactions the machine has, by logic.

A body is a gear.  An interaction among gears is a relation between where their teeth fall on the
column line - a relation among their phases at a column.  The machine has exactly the following
interactions, each a statement about a column `N = 6 m` and its members `N - 1`, `N + 1`.

  1. Member coprimality (`no_gear_both_members`): no gear strikes both members of one column,
     because it would divide their difference 2.  So a column's killers split into two disjoint
     sets, one per member.
  2. Stacking is bounded by size (`no_three_large_on_member`): three gears whose product exceeds
     `q²` cannot all divide one member at most `q²`.  Large gears cannot pile onto one member;
     with `top_gear_cofactor` (round 84), above `q/2` at most one can.
  3. Joint strikes are a single class (`joint_strike_class`): two gears striking chosen members of
     the same column do so on exactly one class modulo their product.  This is the Chinese
     remainder theorem, and it says the lattices carry NO interaction: over a full joint period
     the coincidences are exactly the product of the shares.
  4. Truncation: the window is shorter than a joint period, so a joint class appears in it a whole
     number of times, one more or one fewer than its share.  This is the only source of deviation
     between coincidences and shares, and it is fixed, per tuple, by the window's endpoints modulo
     the tuple's period - not by anything the gears do to each other.
  5. The cofactor recursion: a gear's strike lands on a survivor of the smaller gears exactly when
     its cofactor is free of them.  The kills of a field on the survivors are indexed by the rough
     cofactors, which are the survivors of a smaller machine.  Self-similar, and again no
     coordination across gears.

None of the five can coordinate gears across a window: 1, 2 and 4 are bounded per column or per
tuple, 3 is exactly zero, and 5 is a recursion into the same structure.  A block would need a
sixth: a relation between the residues of ONE integer modulo different gears.  There is none
beyond size - the residues of an integer below the joint period are free (CRT surjectivity), and
the only thing "size" says is that the window's start is one such integer.  Whether a small
integer's residue vector can be adversarial is the free-configuration question, decided
negatively by exact search to `q = 23` and open beyond as the exponent-2 question.
-/
import Mathlib

namespace MirrorWalk

/-- **1. No gear strikes both members of a column.**  A divisor of both `N - 1` and `N + 1`
divides 2, so a gear of size at least 3 cannot. -/
theorem no_gear_both_members {g N : ℕ} (hg : 3 ≤ g) (hN : 1 ≤ N)
    (h1 : g ∣ N - 1) (h2 : g ∣ N + 1) : False := by
  have hd : g ∣ (N + 1) - (N - 1) := Nat.dvd_sub h2 h1
  have e : (N + 1) - (N - 1) = 2 := by omega
  rw [e] at hd
  have := Nat.le_of_dvd (by norm_num) hd
  omega

/-- **2. Three gears whose product exceeds `q²` cannot all divide one member at most `q²`.** -/
theorem no_three_large_on_member {g₁ g₂ g₃ n q : ℕ} (hn : n ≤ q ^ 2) (hn0 : 0 < n)
    (hbig : q ^ 2 < g₁ * g₂ * g₃) (hc12 : Nat.Coprime g₁ g₂) (hc13 : Nat.Coprime g₁ g₃)
    (hc23 : Nat.Coprime g₂ g₃)
    (h1 : g₁ ∣ n) (h2 : g₂ ∣ n) (h3 : g₃ ∣ n) : False := by
  have h12 : g₁ * g₂ ∣ n := Nat.Coprime.mul_dvd_of_dvd_of_dvd hc12 h1 h2
  have hc : Nat.Coprime (g₁ * g₂) g₃ := Nat.Coprime.mul hc13 hc23
  have h123 : g₁ * g₂ * g₃ ∣ n := Nat.Coprime.mul_dvd_of_dvd_of_dvd hc h12 h3
  have := Nat.le_of_dvd hn0 h123
  omega

/-- **3. A joint strike is one class modulo the product.**  If gears `g₁` and `g₂` are coprime,
the columns `N` with `N ≡ a [MOD g₁]` and `N ≡ b [MOD g₂]` - the ones where `g₁` strikes its
chosen member and `g₂` strikes its chosen member - form exactly one residue class modulo
`g₁ g₂`: any two such columns are congruent modulo `g₁ g₂`, and one exists. -/
theorem joint_strike_class {g₁ g₂ a b : ℕ} (hc : Nat.Coprime g₁ g₂) :
    (∃ N : ℕ, N ≡ a [MOD g₁] ∧ N ≡ b [MOD g₂]) ∧
    (∀ N M : ℕ, N ≡ a [MOD g₁] → N ≡ b [MOD g₂] → M ≡ a [MOD g₁] → M ≡ b [MOD g₂] →
      N ≡ M [MOD g₁ * g₂]) := by
  constructor
  · obtain ⟨N, hN1, hN2⟩ := Nat.chineseRemainder hc a b
    exact ⟨N, hN1, hN2⟩
  · intro N M hN1 hN2 hM1 hM2
    have e1 : N ≡ M [MOD g₁] := hN1.trans hM1.symm
    have e2 : N ≡ M [MOD g₂] := hN2.trans hM2.symm
    exact (Nat.modEq_and_modEq_iff_modEq_mul hc).mp ⟨e1, e2⟩

end MirrorWalk

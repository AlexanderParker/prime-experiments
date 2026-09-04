/-
THE 31->37 RUNG BY CASE-SPLIT LP DUALITY - TIER 18 of 35 (GENERATED).

Held gears [5, 7] at residues [2, 4]; this tier imports the 11 cases
with gear 11 at every residue and proves that no window of 95
consecutive slots is fully blocked when the held residues are
these.  Tiered because the flat 385-import root of this rung
exhausted the machine's commit charge (formalist.md R29.5).
-/
import CaseCert37C198
import CaseCert37C199
import CaseCert37C200
import CaseCert37C201
import CaseCert37C202
import CaseCert37C203
import CaseCert37C204
import CaseCert37C205
import CaseCert37C206
import CaseCert37C207
import CaseCert37C208

namespace CaseCert37

set_option maxHeartbeats 4000000

theorem nocase198 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov198 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q198 t) (plt198 t ht)
  rw [e5, e7, e11, pfree198_5 t ht, pfree198_7 t ht, pfree198_11 t ht] at h3
  exact h3

theorem nocase199 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 1)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov199 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q199 t) (plt199 t ht)
  rw [e5, e7, e11, pfree199_5 t ht, pfree199_7 t ht, pfree199_11 t ht] at h3
  exact h3

theorem nocase200 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 2)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov200 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q200 t) (plt200 t ht)
  rw [e5, e7, e11, pfree200_5 t ht, pfree200_7 t ht, pfree200_11 t ht] at h3
  exact h3

theorem nocase201 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 3)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov201 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q201 t) (plt201 t ht)
  rw [e5, e7, e11, pfree201_5 t ht, pfree201_7 t ht, pfree201_11 t ht] at h3
  exact h3

theorem nocase202 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 4)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov202 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q202 t) (plt202 t ht)
  rw [e5, e7, e11, pfree202_5 t ht, pfree202_7 t ht, pfree202_11 t ht] at h3
  exact h3

theorem nocase203 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 5)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov203 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q203 t) (plt203 t ht)
  rw [e5, e7, e11, pfree203_5 t ht, pfree203_7 t ht, pfree203_11 t ht] at h3
  exact h3

theorem nocase204 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 6)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov204 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q204 t) (plt204 t ht)
  rw [e5, e7, e11, pfree204_5 t ht, pfree204_7 t ht, pfree204_11 t ht] at h3
  exact h3

theorem nocase205 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 7)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov205 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q205 t) (plt205 t ht)
  rw [e5, e7, e11, pfree205_5 t ht, pfree205_7 t ht, pfree205_11 t ht] at h3
  exact h3

theorem nocase206 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 8)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov206 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q206 t) (plt206 t ht)
  rw [e5, e7, e11, pfree206_5 t ht, pfree206_7 t ht, pfree206_11 t ht] at h3
  exact h3

theorem nocase207 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 9)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov207 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q207 t) (plt207 t ht)
  rw [e5, e7, e11, pfree207_5 t ht, pfree207_7 t ht, pfree207_11 t ht] at h3
  exact h3

theorem nocase208 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4) (e11 : p % 11 = 10)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov208 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q208 t) (plt208 t ht)
  rw [e5, e7, e11, pfree208_5 t ht, pfree208_7 t ht, pfree208_11 t ht] at h3
  exact h3

/-- Tier 18: the 11 residues of gear 11, each closed by its case. -/
theorem nopair18 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 4)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  have d11 : p % 11 = 0 ∨ p % 11 = 1 ∨ p % 11 = 2 ∨ p % 11 = 3 ∨ p % 11 = 4 ∨ p % 11 = 5 ∨ p % 11 = 6 ∨ p % 11 = 7 ∨ p % 11 = 8 ∨ p % 11 = 9 ∨ p % 11 = 10 := by omega
  rcases d11 with e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11
  · exact nocase198 e5 e7 e11 hall
  · exact nocase199 e5 e7 e11 hall
  · exact nocase200 e5 e7 e11 hall
  · exact nocase201 e5 e7 e11 hall
  · exact nocase202 e5 e7 e11 hall
  · exact nocase203 e5 e7 e11 hall
  · exact nocase204 e5 e7 e11 hall
  · exact nocase205 e5 e7 e11 hall
  · exact nocase206 e5 e7 e11 hall
  · exact nocase207 e5 e7 e11 hall
  · exact nocase208 e5 e7 e11 hall

end CaseCert37

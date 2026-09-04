/-
THE 31->37 RUNG BY CASE-SPLIT LP DUALITY - TIER 7 of 35 (GENERATED).

Held gears [5, 7] at residues [1, 0]; this tier imports the 11 cases
with gear 11 at every residue and proves that no window of 95
consecutive slots is fully blocked when the held residues are
these.  Tiered because the flat 385-import root of this rung
exhausted the machine's commit charge (formalist.md R29.5).
-/
import CaseCert37C77
import CaseCert37C78
import CaseCert37C79
import CaseCert37C80
import CaseCert37C81
import CaseCert37C82
import CaseCert37C83
import CaseCert37C84
import CaseCert37C85
import CaseCert37C86
import CaseCert37C87

namespace CaseCert37

set_option maxHeartbeats 4000000

theorem nocase77 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov77 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q77 t) (plt77 t ht)
  rw [e5, e7, e11, pfree77_5 t ht, pfree77_7 t ht, pfree77_11 t ht] at h3
  exact h3

theorem nocase78 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 1)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov78 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q78 t) (plt78 t ht)
  rw [e5, e7, e11, pfree78_5 t ht, pfree78_7 t ht, pfree78_11 t ht] at h3
  exact h3

theorem nocase79 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 2)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov79 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q79 t) (plt79 t ht)
  rw [e5, e7, e11, pfree79_5 t ht, pfree79_7 t ht, pfree79_11 t ht] at h3
  exact h3

theorem nocase80 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 3)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov80 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q80 t) (plt80 t ht)
  rw [e5, e7, e11, pfree80_5 t ht, pfree80_7 t ht, pfree80_11 t ht] at h3
  exact h3

theorem nocase81 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 4)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov81 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q81 t) (plt81 t ht)
  rw [e5, e7, e11, pfree81_5 t ht, pfree81_7 t ht, pfree81_11 t ht] at h3
  exact h3

theorem nocase82 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 5)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov82 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q82 t) (plt82 t ht)
  rw [e5, e7, e11, pfree82_5 t ht, pfree82_7 t ht, pfree82_11 t ht] at h3
  exact h3

theorem nocase83 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 6)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov83 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q83 t) (plt83 t ht)
  rw [e5, e7, e11, pfree83_5 t ht, pfree83_7 t ht, pfree83_11 t ht] at h3
  exact h3

theorem nocase84 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 7)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov84 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q84 t) (plt84 t ht)
  rw [e5, e7, e11, pfree84_5 t ht, pfree84_7 t ht, pfree84_11 t ht] at h3
  exact h3

theorem nocase85 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 8)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov85 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q85 t) (plt85 t ht)
  rw [e5, e7, e11, pfree85_5 t ht, pfree85_7 t ht, pfree85_11 t ht] at h3
  exact h3

theorem nocase86 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 9)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov86 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q86 t) (plt86 t ht)
  rw [e5, e7, e11, pfree86_5 t ht, pfree86_7 t ht, pfree86_11 t ht] at h3
  exact h3

theorem nocase87 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0) (e11 : p % 11 = 10)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov87 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q87 t) (plt87 t ht)
  rw [e5, e7, e11, pfree87_5 t ht, pfree87_7 t ht, pfree87_11 t ht] at h3
  exact h3

/-- Tier 7: the 11 residues of gear 11, each closed by its case. -/
theorem nopair7 {p : ℕ} (e5 : p % 5 = 1) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  have d11 : p % 11 = 0 ∨ p % 11 = 1 ∨ p % 11 = 2 ∨ p % 11 = 3 ∨ p % 11 = 4 ∨ p % 11 = 5 ∨ p % 11 = 6 ∨ p % 11 = 7 ∨ p % 11 = 8 ∨ p % 11 = 9 ∨ p % 11 = 10 := by omega
  rcases d11 with e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11
  · exact nocase77 e5 e7 e11 hall
  · exact nocase78 e5 e7 e11 hall
  · exact nocase79 e5 e7 e11 hall
  · exact nocase80 e5 e7 e11 hall
  · exact nocase81 e5 e7 e11 hall
  · exact nocase82 e5 e7 e11 hall
  · exact nocase83 e5 e7 e11 hall
  · exact nocase84 e5 e7 e11 hall
  · exact nocase85 e5 e7 e11 hall
  · exact nocase86 e5 e7 e11 hall
  · exact nocase87 e5 e7 e11 hall

end CaseCert37

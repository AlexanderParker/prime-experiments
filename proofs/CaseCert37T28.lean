/-
THE 31->37 RUNG BY CASE-SPLIT LP DUALITY - TIER 28 of 35 (GENERATED).

Held gears [5, 7] at residues [4, 0]; this tier imports the 11 cases
with gear 11 at every residue and proves that no window of 95
consecutive slots is fully blocked when the held residues are
these.  Tiered because the flat 385-import root of this rung
exhausted the machine's commit charge (formalist.md R29.5).
-/
import CaseCert37C308
import CaseCert37C309
import CaseCert37C310
import CaseCert37C311
import CaseCert37C312
import CaseCert37C313
import CaseCert37C314
import CaseCert37C315
import CaseCert37C316
import CaseCert37C317
import CaseCert37C318

namespace CaseCert37

set_option maxHeartbeats 4000000

theorem nocase308 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov308 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q308 t) (plt308 t ht)
  rw [e5, e7, e11, pfree308_5 t ht, pfree308_7 t ht, pfree308_11 t ht] at h3
  exact h3

theorem nocase309 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 1)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov309 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q309 t) (plt309 t ht)
  rw [e5, e7, e11, pfree309_5 t ht, pfree309_7 t ht, pfree309_11 t ht] at h3
  exact h3

theorem nocase310 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 2)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov310 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q310 t) (plt310 t ht)
  rw [e5, e7, e11, pfree310_5 t ht, pfree310_7 t ht, pfree310_11 t ht] at h3
  exact h3

theorem nocase311 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 3)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov311 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q311 t) (plt311 t ht)
  rw [e5, e7, e11, pfree311_5 t ht, pfree311_7 t ht, pfree311_11 t ht] at h3
  exact h3

theorem nocase312 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 4)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov312 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q312 t) (plt312 t ht)
  rw [e5, e7, e11, pfree312_5 t ht, pfree312_7 t ht, pfree312_11 t ht] at h3
  exact h3

theorem nocase313 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 5)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov313 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q313 t) (plt313 t ht)
  rw [e5, e7, e11, pfree313_5 t ht, pfree313_7 t ht, pfree313_11 t ht] at h3
  exact h3

theorem nocase314 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 6)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov314 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q314 t) (plt314 t ht)
  rw [e5, e7, e11, pfree314_5 t ht, pfree314_7 t ht, pfree314_11 t ht] at h3
  exact h3

theorem nocase315 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 7)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov315 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q315 t) (plt315 t ht)
  rw [e5, e7, e11, pfree315_5 t ht, pfree315_7 t ht, pfree315_11 t ht] at h3
  exact h3

theorem nocase316 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 8)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov316 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q316 t) (plt316 t ht)
  rw [e5, e7, e11, pfree316_5 t ht, pfree316_7 t ht, pfree316_11 t ht] at h3
  exact h3

theorem nocase317 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 9)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov317 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q317 t) (plt317 t ht)
  rw [e5, e7, e11, pfree317_5 t ht, pfree317_7 t ht, pfree317_11 t ht] at h3
  exact h3

theorem nocase318 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0) (e11 : p % 11 = 10)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov318 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q318 t) (plt318 t ht)
  rw [e5, e7, e11, pfree318_5 t ht, pfree318_7 t ht, pfree318_11 t ht] at h3
  exact h3

/-- Tier 28: the 11 residues of gear 11, each closed by its case. -/
theorem nopair28 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  have d11 : p % 11 = 0 ∨ p % 11 = 1 ∨ p % 11 = 2 ∨ p % 11 = 3 ∨ p % 11 = 4 ∨ p % 11 = 5 ∨ p % 11 = 6 ∨ p % 11 = 7 ∨ p % 11 = 8 ∨ p % 11 = 9 ∨ p % 11 = 10 := by omega
  rcases d11 with e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11
  · exact nocase308 e5 e7 e11 hall
  · exact nocase309 e5 e7 e11 hall
  · exact nocase310 e5 e7 e11 hall
  · exact nocase311 e5 e7 e11 hall
  · exact nocase312 e5 e7 e11 hall
  · exact nocase313 e5 e7 e11 hall
  · exact nocase314 e5 e7 e11 hall
  · exact nocase315 e5 e7 e11 hall
  · exact nocase316 e5 e7 e11 hall
  · exact nocase317 e5 e7 e11 hall
  · exact nocase318 e5 e7 e11 hall

end CaseCert37

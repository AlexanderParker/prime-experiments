/-
THE 31->37 RUNG BY CASE-SPLIT LP DUALITY - TIER 30 of 35 (GENERATED).

Held gears [5, 7] at residues [4, 2]; this tier imports the 11 cases
with gear 11 at every residue and proves that no window of 95
consecutive slots is fully blocked when the held residues are
these.  Tiered because the flat 385-import root of this rung
exhausted the machine's commit charge (formalist.md R29.5).
-/
import CaseCert37C330
import CaseCert37C331
import CaseCert37C332
import CaseCert37C333
import CaseCert37C334
import CaseCert37C335
import CaseCert37C336
import CaseCert37C337
import CaseCert37C338
import CaseCert37C339
import CaseCert37C340

namespace CaseCert37

set_option maxHeartbeats 4000000

theorem nocase330 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov330 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q330 t) (plt330 t ht)
  rw [e5, e7, e11, pfree330_5 t ht, pfree330_7 t ht, pfree330_11 t ht] at h3
  exact h3

theorem nocase331 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 1)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov331 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q331 t) (plt331 t ht)
  rw [e5, e7, e11, pfree331_5 t ht, pfree331_7 t ht, pfree331_11 t ht] at h3
  exact h3

theorem nocase332 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 2)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov332 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q332 t) (plt332 t ht)
  rw [e5, e7, e11, pfree332_5 t ht, pfree332_7 t ht, pfree332_11 t ht] at h3
  exact h3

theorem nocase333 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 3)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov333 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q333 t) (plt333 t ht)
  rw [e5, e7, e11, pfree333_5 t ht, pfree333_7 t ht, pfree333_11 t ht] at h3
  exact h3

theorem nocase334 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 4)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov334 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q334 t) (plt334 t ht)
  rw [e5, e7, e11, pfree334_5 t ht, pfree334_7 t ht, pfree334_11 t ht] at h3
  exact h3

theorem nocase335 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 5)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov335 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q335 t) (plt335 t ht)
  rw [e5, e7, e11, pfree335_5 t ht, pfree335_7 t ht, pfree335_11 t ht] at h3
  exact h3

theorem nocase336 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 6)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov336 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q336 t) (plt336 t ht)
  rw [e5, e7, e11, pfree336_5 t ht, pfree336_7 t ht, pfree336_11 t ht] at h3
  exact h3

theorem nocase337 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 7)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov337 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q337 t) (plt337 t ht)
  rw [e5, e7, e11, pfree337_5 t ht, pfree337_7 t ht, pfree337_11 t ht] at h3
  exact h3

theorem nocase338 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 8)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov338 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q338 t) (plt338 t ht)
  rw [e5, e7, e11, pfree338_5 t ht, pfree338_7 t ht, pfree338_11 t ht] at h3
  exact h3

theorem nocase339 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 9)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov339 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q339 t) (plt339 t ht)
  rw [e5, e7, e11, pfree339_5 t ht, pfree339_7 t ht, pfree339_11 t ht] at h3
  exact h3

theorem nocase340 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2) (e11 : p % 11 = 10)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov340 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q340 t) (plt340 t ht)
  rw [e5, e7, e11, pfree340_5 t ht, pfree340_7 t ht, pfree340_11 t ht] at h3
  exact h3

/-- Tier 30: the 11 residues of gear 11, each closed by its case. -/
theorem nopair30 {p : ℕ} (e5 : p % 5 = 4) (e7 : p % 7 = 2)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  have d11 : p % 11 = 0 ∨ p % 11 = 1 ∨ p % 11 = 2 ∨ p % 11 = 3 ∨ p % 11 = 4 ∨ p % 11 = 5 ∨ p % 11 = 6 ∨ p % 11 = 7 ∨ p % 11 = 8 ∨ p % 11 = 9 ∨ p % 11 = 10 := by omega
  rcases d11 with e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11
  · exact nocase330 e5 e7 e11 hall
  · exact nocase331 e5 e7 e11 hall
  · exact nocase332 e5 e7 e11 hall
  · exact nocase333 e5 e7 e11 hall
  · exact nocase334 e5 e7 e11 hall
  · exact nocase335 e5 e7 e11 hall
  · exact nocase336 e5 e7 e11 hall
  · exact nocase337 e5 e7 e11 hall
  · exact nocase338 e5 e7 e11 hall
  · exact nocase339 e5 e7 e11 hall
  · exact nocase340 e5 e7 e11 hall

end CaseCert37

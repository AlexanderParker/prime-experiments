/-
THE 31->37 RUNG BY CASE-SPLIT LP DUALITY - TIER 20 of 35 (GENERATED).

Held gears [5, 7] at residues [2, 6]; this tier imports the 11 cases
with gear 11 at every residue and proves that no window of 95
consecutive slots is fully blocked when the held residues are
these.  Tiered because the flat 385-import root of this rung
exhausted the machine's commit charge (formalist.md R29.5).
-/
import CaseCert37C220
import CaseCert37C221
import CaseCert37C222
import CaseCert37C223
import CaseCert37C224
import CaseCert37C225
import CaseCert37C226
import CaseCert37C227
import CaseCert37C228
import CaseCert37C229
import CaseCert37C230

namespace CaseCert37

set_option maxHeartbeats 4000000

theorem nocase220 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 0)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov220 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q220 t) (plt220 t ht)
  rw [e5, e7, e11, pfree220_5 t ht, pfree220_7 t ht, pfree220_11 t ht] at h3
  exact h3

theorem nocase221 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 1)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov221 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q221 t) (plt221 t ht)
  rw [e5, e7, e11, pfree221_5 t ht, pfree221_7 t ht, pfree221_11 t ht] at h3
  exact h3

theorem nocase222 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 2)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov222 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q222 t) (plt222 t ht)
  rw [e5, e7, e11, pfree222_5 t ht, pfree222_7 t ht, pfree222_11 t ht] at h3
  exact h3

theorem nocase223 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 3)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov223 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q223 t) (plt223 t ht)
  rw [e5, e7, e11, pfree223_5 t ht, pfree223_7 t ht, pfree223_11 t ht] at h3
  exact h3

theorem nocase224 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 4)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov224 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q224 t) (plt224 t ht)
  rw [e5, e7, e11, pfree224_5 t ht, pfree224_7 t ht, pfree224_11 t ht] at h3
  exact h3

theorem nocase225 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 5)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov225 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q225 t) (plt225 t ht)
  rw [e5, e7, e11, pfree225_5 t ht, pfree225_7 t ht, pfree225_11 t ht] at h3
  exact h3

theorem nocase226 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 6)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov226 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q226 t) (plt226 t ht)
  rw [e5, e7, e11, pfree226_5 t ht, pfree226_7 t ht, pfree226_11 t ht] at h3
  exact h3

theorem nocase227 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 7)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov227 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q227 t) (plt227 t ht)
  rw [e5, e7, e11, pfree227_5 t ht, pfree227_7 t ht, pfree227_11 t ht] at h3
  exact h3

theorem nocase228 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 8)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov228 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q228 t) (plt228 t ht)
  rw [e5, e7, e11, pfree228_5 t ht, pfree228_7 t ht, pfree228_11 t ht] at h3
  exact h3

theorem nocase229 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 9)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov229 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q229 t) (plt229 t ht)
  rw [e5, e7, e11, pfree229_5 t ht, pfree229_7 t ht, pfree229_11 t ht] at h3
  exact h3

theorem nocase230 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6) (e11 : p % 11 = 10)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  refine nocov230 (r0 := p % 13) (r1 := p % 17) (r2 := p % 19) (r3 := p % 23) (r4 := p % 29) (r5 := p % 31) (r6 := p % 37) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) ?_
  intro t ht
  have h3 := hall (q230 t) (plt230 t ht)
  rw [e5, e7, e11, pfree230_5 t ht, pfree230_7 t ht, pfree230_11 t ht] at h3
  exact h3

/-- Tier 20: the 11 residues of gear 11, each closed by its case. -/
theorem nopair20 {p : ℕ} (e5 : p % 5 = 2) (e7 : p % 7 = 6)
    (hall : ∀ i, i < 95 → (gb5 (p % 5) i || gb7 (p % 7) i || gb11 (p % 11) i || gb13 (p % 13) i || gb17 (p % 17) i || gb19 (p % 19) i || gb23 (p % 23) i || gb29 (p % 29) i || gb31 (p % 31) i || gb37 (p % 37) i) = true) : False := by
  have d11 : p % 11 = 0 ∨ p % 11 = 1 ∨ p % 11 = 2 ∨ p % 11 = 3 ∨ p % 11 = 4 ∨ p % 11 = 5 ∨ p % 11 = 6 ∨ p % 11 = 7 ∨ p % 11 = 8 ∨ p % 11 = 9 ∨ p % 11 = 10 := by omega
  rcases d11 with e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11 | e11
  · exact nocase220 e5 e7 e11 hall
  · exact nocase221 e5 e7 e11 hall
  · exact nocase222 e5 e7 e11 hall
  · exact nocase223 e5 e7 e11 hall
  · exact nocase224 e5 e7 e11 hall
  · exact nocase225 e5 e7 e11 hall
  · exact nocase226 e5 e7 e11 hall
  · exact nocase227 e5 e7 e11 hall
  · exact nocase228 e5 e7 e11 hall
  · exact nocase229 e5 e7 e11 hall
  · exact nocase230 e5 e7 e11 hall

end CaseCert37

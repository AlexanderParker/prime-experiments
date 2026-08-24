/-
The merge law as a two-machine kernel statement - Constructor's R39.

When gear `q'` is added to a machine `M`, every gap of the new machine
`M + q'` is a MERGED WINDOW of the old machine: a window of `j` consecutive
old gaps whose `j - 1` interior openings are all killed by the new gear.
Gear `q'` has exactly two teeth, at slot residues `u'` and `q' - u'`
(`6u' = q' -+ 1`), so any two killed openings differ by `0`, `2u'` or
`q' - 2u'` mod `q'` - RESIDUE NECESSITY - and a positive gap in one of
those classes is at least `2u'`, the qualifying floor. Hence every interior
gap of a merged window qualifies, the window is a QUALIFYING window of the
old machine, and

    F(M + q')  <=  max (F2, max_j qualmax_j)          (R39)

with no firing analysis, no order statistics, no lambda: merge law +
residue necessity only. `(D)` at `alpha = 3` follows whenever the
right-hand side is at most `F + q'`.

Everything here is abstract: `pos` is the old machine's opening sequence,
`g` its gap word, `kap` the kill predicate ON OPENING INDICES. The
statements consume `Spectrum.SpectrumBound` / `Spectrum.QualBound`
instances, which the per-machine kernel scans provide (`Machine19Q.lean`),
and are consumed in turn by the concrete two-machine instance
(`Machine23.lean`). Nothing empirical is assumed in this file.
-/

import Spectrum

namespace MergeLaw

/-! ## Modular difference arithmetic -/

/-- Difference of two residues: if `x <= y` with `x % q = al`, `y % q = be`
(`al, be < q`), then `(y - x) % q = (q + be - al) % q`. -/
theorem sub_mod_eq {q x y al be : ℕ} (hxy : x ≤ y) (hal : al < q) (hbe : be < q)
    (hx : x % q = al) (hy : y % q = be) : (y - x) % q = (q + be - al) % q := by
  have hq : 0 < q := by omega
  obtain ⟨d, rfl⟩ : ∃ d, y = x + d := ⟨y - x, by omega⟩
  have hr : d % q < q := Nat.mod_lt _ hq
  have h1 : (al + d % q) % q = be := by
    rw [← hx, ← Nat.add_mod]; exact hy
  have hd : x + d - x = d := by omega
  rw [hd]
  rcases Nat.lt_or_ge (al + d % q) q with hlt | hge
  · have hbe' : al + d % q = be := by rw [← Nat.mod_eq_of_lt hlt]; exact h1
    have e : q + be - al = q + d % q := by omega
    rw [e, Nat.add_mod_left, Nat.mod_eq_of_lt hr]
  · have hsub : (al + d % q) % q = al + d % q - q := by
      rw [Nat.mod_eq_sub_mod hge]
      exact Nat.mod_eq_of_lt (by omega)
    have hbe' : al + d % q - q = be := by rw [← hsub]; exact h1
    have e : q + be - al = d % q := by omega
    rw [e, Nat.mod_eq_of_lt hr]

/-! ## Merged windows -/

/-- A merged window of the new machine, over the OLD machine's opening
indices: the openings at `a` and `a + j` survive gear `q'`, every interior
opening is killed. The new gap is then the window sum of the `j` old gaps
from `a` (`Spectrum.windowSum g a j`) - the merge law by construction. -/
def MergedWindow (kap : ℕ → Prop) (a j : ℕ) : Prop :=
  0 < j ∧ ¬ kap a ∧ ¬ kap (a + j) ∧ ∀ i, 0 < i → i < j → kap (a + i)

/-- **Residue necessity.** If killed openings sit on gear `q`'s two teeth
`{u, q - u}` (slot residues), every INTERIOR gap of a merged window - both
of whose endpoints are killed - is `0`, `2u` or `q - 2u` mod `q`. -/
theorem interior_gap_mod {kap : ℕ → Prop} {pos g : ℕ → ℕ} {q u a j : ℕ}
    (hg : ∀ n, g n = pos (n + 1) - pos n) (hmono : ∀ n, pos n ≤ pos (n + 1))
    (hteeth : ∀ n, kap n → pos n % q = u ∨ pos n % q = q - u)
    (hu : 0 < u) (h2u : 2 * u < q)
    (hmw : MergedWindow kap a j) {i : ℕ} (h1 : 1 ≤ i) (h2 : i + 1 < j) :
    g (a + i) % q = 0 ∨ g (a + i) % q = 2 * u ∨ g (a + i) % q = q - 2 * u := by
  obtain ⟨hj, _hka, _hkb, hint⟩ := hmw
  have k1 : kap (a + i) := hint i (by omega) (by omega)
  have k2 : kap (a + i + 1) := by
    have h := hint (i + 1) (by omega) h2
    rwa [show a + (i + 1) = a + i + 1 by omega] at h
  have r1 := hteeth _ k1
  have r2 := hteeth _ k2
  have hle : pos (a + i) ≤ pos (a + i + 1) := hmono _
  rw [hg]
  rcases r1 with h | h <;> rcases r2 with h' | h'
  · -- u -> u : difference 0 mod q
    left
    have hs := sub_mod_eq hle (by omega) (by omega) h h'
    have e : q + u - u = q := by omega
    rw [e, Nat.mod_self] at hs
    exact hs
  · -- u -> q - u : difference q - 2u mod q
    right; right
    have hs := sub_mod_eq hle (by omega) (by omega) h h'
    have e : q + (q - u) - u = 2 * q - 2 * u := by omega
    rw [e] at hs
    have e2 : (2 * q - 2 * u) % q = q - 2 * u := by
      rw [Nat.mod_eq_sub_mod (by omega)]
      have e3 : 2 * q - 2 * u - q = q - 2 * u := by omega
      rw [e3]
      exact Nat.mod_eq_of_lt (by omega)
    rw [e2] at hs
    exact hs
  · -- q - u -> u : difference 2u mod q
    right; left
    have hs := sub_mod_eq hle (by omega) (by omega) h h'
    have e : q + u - (q - u) = 2 * u := by omega
    rw [e, show (2 * u) % q = 2 * u from Nat.mod_eq_of_lt (by omega)] at hs
    exact hs
  · -- q - u -> q - u : difference 0 mod q
    left
    have hs := sub_mod_eq hle (by omega) (by omega) h h'
    have e : q + (q - u) - (q - u) = q := by omega
    rw [e, Nat.mod_self] at hs
    exact hs

/-- A positive gap in one of the three necessary residue classes meets the
qualifying floor `2u` (`4u <= q` holds for every gear: `6u' = q' -+ 1`). -/
theorem floor_of_mod {G q u : ℕ} (hG : 0 < G) (h4u : 4 * u ≤ q)
    (h : G % q = 0 ∨ G % q = 2 * u ∨ G % q = q - 2 * u) : 2 * u ≤ G := by
  have hGq : G % q ≤ G := Nat.mod_le _ _
  rcases h with h | h | h
  · rcases Nat.eq_zero_or_pos q with hq | hq
    · omega
    · have hdvd : q ∣ G := Nat.dvd_of_mod_eq_zero h
      have := Nat.le_of_dvd hG hdvd
      omega
  · omega
  · omega

/-! ## R39 -/

/-- **R39, the core form.** Merge law + residue necessity: every merged
window sum of the old machine is at most `B` as soon as `F2 <= B` and every
qualifying spectrum value `Q_j <= B` (`j >= 3`). Instantiating
`B = max F2 (max_j Q_j)` gives `F(M+q') <= max(F2, max_j qualmax_j)`
verbatim; instantiating `B = F + q'` gives (D) at `alpha = 3`. -/
theorem newgap_le {g pos : ℕ → ℕ} {kap : ℕ → Prop} {q u B F2 : ℕ} {Q : ℕ → ℕ}
    (hg : ∀ n, g n = pos (n + 1) - pos n) (hmono : ∀ n, pos n < pos (n + 1))
    (hteeth : ∀ n, kap n → pos n % q = u ∨ pos n % q = q - u)
    (hu : 0 < u) (h4u : 4 * u ≤ q)
    (hF2 : Spectrum.SpectrumBound g 2 F2) (hF2B : F2 ≤ B)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g u j (Q j)) (hQB : ∀ j, Q j ≤ B)
    {a j : ℕ} (hmw : MergedWindow kap a j) :
    Spectrum.windowSum g a j ≤ B := by
  rcases Nat.lt_or_ge j 3 with hj3 | hj3
  · exact le_trans (le_trans (Spectrum.windowSum_mono g a (by omega)) (hF2 a)) hF2B
  · have hqual : Spectrum.Qualifying g u a j := by
      intro i hi1 hi2
      have hm := interior_gap_mod hg (fun n => le_of_lt (hmono n)) hteeth hu
        (by omega) hmw hi1 hi2
      have hGpos : 0 < g (a + i) := by
        rw [hg]; have := hmono (a + i); omega
      exact floor_of_mod hGpos h4u hm
    exact le_trans (hQ j hj3 a hqual) (hQB j)

/-- **R39 in Constructor's exact shape**: every new gap is at most
`max (F2, Qmax)` where `Qmax` dominates the qualifying spectrum. -/
theorem newgap_le_max {g pos : ℕ → ℕ} {kap : ℕ → Prop} {q u F2 Qmax : ℕ}
    {Q : ℕ → ℕ}
    (hg : ∀ n, g n = pos (n + 1) - pos n) (hmono : ∀ n, pos n < pos (n + 1))
    (hteeth : ∀ n, kap n → pos n % q = u ∨ pos n % q = q - u)
    (hu : 0 < u) (h4u : 4 * u ≤ q)
    (hF2 : Spectrum.SpectrumBound g 2 F2)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g u j (Q j)) (hQm : ∀ j, Q j ≤ Qmax)
    {a j : ℕ} (hmw : MergedWindow kap a j) :
    Spectrum.windowSum g a j ≤ max F2 Qmax :=
  newgap_le hg hmono hteeth hu h4u hF2 (le_max_left _ _) hQ
    (fun j => le_trans (hQm j) (le_max_right _ _)) hmw

/-- **(D) at `alpha = 3` from R39**: if `max(F2, max_j Q_j) <= F + q'`,
every merged window - every gap of the new machine - is within tolerance. -/
theorem D_of_qualmax {g pos : ℕ → ℕ} {kap : ℕ → Prop} {q u F2 F qp : ℕ}
    {Q : ℕ → ℕ}
    (hg : ∀ n, g n = pos (n + 1) - pos n) (hmono : ∀ n, pos n < pos (n + 1))
    (hteeth : ∀ n, kap n → pos n % q = u ∨ pos n % q = q - u)
    (hu : 0 < u) (h4u : 4 * u ≤ q)
    (hF2 : Spectrum.SpectrumBound g 2 F2) (hF2D : F2 ≤ F + qp)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g u j (Q j))
    (hQD : ∀ j, Q j ≤ F + qp)
    {a j : ℕ} (hmw : MergedWindow kap a j) :
    Spectrum.windowSum g a j ≤ F + qp :=
  newgap_le hg hmono hteeth hu h4u hF2 hF2D hQ hQD hmw

end MergeLaw

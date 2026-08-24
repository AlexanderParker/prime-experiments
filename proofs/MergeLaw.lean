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

/-! ## The step law: R39 applied to two concrete machines

`newgap_le` speaks about merged windows over the OLD machine's opening
indices. The lemma below does the bookkeeping that turns it into a statement
about the NEW machine's own gap sequence, once and for all: locate both
endpoints of a new gap in the old enumeration, check that everything strictly
between is killed, and telescope. Each further rung of the (D) ladder is then
an instantiation - `Ladder.lean` climbs 11->13, 13->17, 17->19 with it, and
`Machine23.lean` is the same argument written out by hand at 19->23. -/

/-- Monotone position sequences do not decrease over jumps. -/
theorem pos_le_add {pos : ℕ → ℕ} (hmono : ∀ m, pos m ≤ pos (m + 1)) (a j : ℕ) :
    pos a ≤ pos (a + j) := by
  induction j with
  | zero => rfl
  | succ j ih =>
    have := hmono (a + j)
    rw [show a + (j + 1) = (a + j) + 1 by omega]
    omega

/-- Window sums of a gap word telescope to position differences. -/
theorem windowSum_telescope {g pos : ℕ → ℕ}
    (hg : ∀ m, g m = pos (m + 1) - pos m) (hmono : ∀ m, pos m ≤ pos (m + 1))
    (a j : ℕ) : Spectrum.windowSum g a j = pos (a + j) - pos a := by
  induction j with
  | zero => simp [Spectrum.windowSum]
  | succ j ih =>
    have hs : Spectrum.windowSum g a (j + 1)
        = Spectrum.windowSum g a j + g (a + j) := Finset.sum_range_succ _ _
    have h1 := pos_le_add hmono a j
    have h2 := hmono (a + j)
    rw [hs, ih, hg, show a + (j + 1) = (a + j) + 1 by omega]
    omega

/-- **The step law.** `ExO`/`posO` are the old machine's openings and their
enumeration, `ExN`/`posN` the new machine's; `Kap` is the new gear's kill
predicate on slots, with teeth `{u, q - u}`. Given the old machine's
`F_2 <= B` and qualifying spectrum `Q_j <= B` (`j >= 3`), EVERY gap of the
new machine is at most `B`. This is R39, `F(M + q') <= max(F2, max_j
qualmax_j)`, as a statement about two concrete machines. -/
theorem newgap_le_step {ExO ExN Kap : ℕ → Prop} {posO posN g : ℕ → ℕ}
    {q u B F2 : ℕ} {Q : ℕ → ℕ}
    (hg : ∀ m, g m = posO (m + 1) - posO m)
    (hOmono : ∀ m, posO m < posO (m + 1)) (hOpos : ∀ m, 1 ≤ posO m)
    (hOex : ∀ m, ExO (posO m))
    (hOsurj : ∀ x, 1 ≤ x → ExO x → ∃ m, posO m = x)
    (hNpos : ∀ m, 1 ≤ posN m) (hNmono : ∀ m, posN m < posN (m + 1))
    (hNex : ∀ m, ExN (posN m))
    (hNempty : ∀ m x, posN m < x → x < posN (m + 1) → ¬ ExN x)
    (hnk : ∀ x, 1 ≤ x → ExN x → ¬ Kap x)
    (hkn : ∀ x, 1 ≤ x → ExO x → ¬ Kap x → ExN x)
    (hsub : ∀ x, ExN x → ExO x)
    (hteeth : ∀ x, Kap x → x % q = u ∨ x % q = q - u)
    (hu : 0 < u) (h4u : 4 * u ≤ q)
    (hF2 : Spectrum.SpectrumBound g 2 F2) (hF2B : F2 ≤ B)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g u j (Q j)) (hQB : ∀ j, Q j ≤ B)
    (n : ℕ) : posN (n + 1) - posN n ≤ B := by
  have hOmono' : ∀ m, posO m ≤ posO (m + 1) := fun m => le_of_lt (hOmono m)
  have hOlt : ∀ a b, a < b → posO a < posO b := by
    intro a b hab
    have h1 := hOmono a
    have h2 := pos_le_add hOmono' (a + 1) (b - (a + 1))
    rw [show a + 1 + (b - (a + 1)) = b by omega] at h2
    omega
  obtain ⟨a, ha⟩ := hOsurj (posN n) (hNpos n) (hsub _ (hNex n))
  obtain ⟨b, hb⟩ := hOsurj (posN (n + 1)) (hNpos (n + 1)) (hsub _ (hNex (n + 1)))
  have hNlt := hNmono n
  have hab : a < b := by
    by_contra hc
    have hle : posO b ≤ posO a := by
      rcases Nat.lt_or_ge b a with h | h
      · exact le_of_lt (hOlt b a h)
      · have he : b = a := by omega
        rw [he]
    omega
  have hmw : MergedWindow (fun i => Kap (posO i)) a (b - a) := by
    refine ⟨by omega, ?_, ?_, ?_⟩
    · show ¬ Kap (posO a)
      rw [ha]; exact hnk _ (hNpos n) (hNex n)
    · show ¬ Kap (posO (a + (b - a)))
      rw [show a + (b - a) = b by omega, hb]
      exact hnk _ (hNpos (n + 1)) (hNex (n + 1))
    · intro i hi0 hij
      show Kap (posO (a + i))
      have hv1 : posN n < posO (a + i) := by rw [← ha]; exact hOlt _ _ (by omega)
      have hv2 : posO (a + i) < posN (n + 1) := by
        rw [← hb]; exact hOlt _ _ (by omega)
      by_contra hK
      exact hNempty n _ hv1 hv2
        (hkn _ (hOpos (a + i)) (hOex _) hK)
  have hgap : posN (n + 1) - posN n = Spectrum.windowSum g a (b - a) := by
    rw [windowSum_telescope hg hOmono', show a + (b - a) = b by omega, ha, hb]
  rw [hgap]
  exact newgap_le hg hOmono (fun i hk => hteeth (posO i) hk) hu h4u hF2 hF2B
    hQ hQB hmw

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

/-
THE SURVIVOR GENERATOR IS SOUND AT 11 -> 13 (round 26) - round-25 verdict 20
discharged, and machine 13's low spectrum PROVED from machine 11's word.

`Gen11.lean` computed `gen 0 = 11`, `gen 1 = 16` and left the honest gap:
"this file states that the generator COMPUTES the right integers; it does not
yet prove that it MUST".  Verdict 20 named the two missing pieces:

  (i)  `gw11` certified as machine 11's own opening sequence;
  (ii) the periodicity glue `opSeq11 (n + 135) = opSeq11 n + 385`.

(ii) is `Machine11Per.opSeq_shift`, an instance of the abstract
`Periodic.op_shift`.  (i) is `gAt_succ` below - and it comes with a
correction: `gw11`'s base is ONE OPENING EARLIER than the enumeration's, so
the exact identity is `gAt (i + 1) = g11 i`, not `gAt i = g11 i`.  (The
generator's VALUE is unaffected: `gen` maximises over all 135 bases, and a
rotation permutes them.)

With those two, the walk can be shown to SIMULATE the machine:

    walk_sound : if the walk does not bail, the offset it returns is exactly
                 `opSeq13 (n + ns + 1) - opSeq13 n`

- proved by induction on the fuel against the invariant "`x + d` is the
`k`-th machine-11 opening after `x`, and exactly `surv` machine-13 openings
lie in `(x, x + d]`".  The bail value is the SENTINEL 999 (round 26's change
to `Gen11.walk`), so `gen ns < 999` is itself the proof that no walk bailed -
which is what turns a computed maximum into a bound.

CONSEQUENCE (`generator_sound`):

    F_1(13) <= 11,  F_2(13) <= 16,  F_3(13) <= 23,  F_4(13) <= 26

with the ONLY period scanned anywhere being machine 11's 385 slots.  Machine
13's own 5,005-slot scan (`Machine13.qasm`) proves the same four integers and
is not used by any theorem in this file - the two derivations are
independent, and they agree.
-/

import Gen11
import Machine11Per

namespace Gen11

/-! ## 1. `gw11` is machine 11's gap word -/

/-- The letters of `gw11`, checked one by one against machine 11's own
opening walk.  `gw11`'s base is one opening earlier than the enumeration's,
which is why the index carries a `+ 1`. -/
theorem word_check : ∀ i < 135,
    gw11.getD ((i + 1) % 135) 0 = Machine11.ow (i + 1) - Machine11.ow i := by
  decide +kernel

/-- **`gw11` IS machine 11's gap word** (verdict 20, item (i)): at EVERY
index, not merely inside one period - that is what the periodicity glue
buys. -/
theorem gAt_succ (i : ℕ) : gAt (i + 1) = Machine11.g11 i := by
  have hlt : i % 135 < 135 := Nat.mod_lt _ (by omega)
  have h := word_check (i % 135) hlt
  have h2 : Machine11.g11 (i % 135)
      = Machine11.ow (i % 135 + 1) - Machine11.ow (i % 135) := Machine11.g11_eq_ow _
  have h3 : (i + 1) % 135 = (i % 135 + 1) % 135 := by omega
  show gw11.getD ((i + 1) % 135) 0 = _
  rw [h3, h, ← h2, ← Machine11.g11_mod]

/-! ## 2. Bookkeeping: the walk's step, its base index, and the max -/

/-- One step of the walk, unfolded. -/
theorem walk_succ (i c ns fuel k d surv : ℕ) :
    walk i c ns (fuel + 1) k d surv =
      (if 30 < d + gAt (i + k) then 999
       else if kil13 (c + (d + gAt (i + k))) then
         walk i c ns fuel (k + 1) (d + gAt (i + k)) surv
       else if surv == ns then d + gAt (i + k)
       else walk i c ns fuel (k + 1) (d + gAt (i + k)) (surv + 1)) := rfl

/-- The walk depends on its base only mod 135. -/
theorem walk_mod (i c ns : ℕ) : ∀ fuel k d surv,
    walk (i % 135) c ns fuel k d surv = walk i c ns fuel k d surv := by
  intro fuel
  induction fuel with
  | zero => intro k d surv; rfl
  | succ fuel ih =>
    intro k d surv
    have hg : gAt (i % 135 + k) = gAt (i + k) := by
      show gw11.getD ((i % 135 + k) % 135) 0 = gw11.getD ((i + k) % 135) 0
      rw [show (i % 135 + k) % 135 = (i + k) % 135 by omega]
    rw [walk_succ, walk_succ, hg]
    simp only [ih]

/-- The fold's seed is a lower bound for the fold. -/
theorem le_foldl_max (l : List ℕ) : ∀ b : ℕ, b ≤ l.foldl max b := by
  induction l with
  | nil => intro b; exact Nat.le_refl b
  | cons a t ih =>
    intro b
    simp only [List.foldl_cons]
    exact le_trans (le_max_left b a) (ih (max b a))

/-- Every member of the list is a lower bound for the fold. -/
theorem mem_le_foldl_max {x : ℕ} (l : List ℕ) : ∀ b : ℕ, x ∈ l → x ≤ l.foldl max b := by
  induction l with
  | nil => intro b h; simp at h
  | cons a t ih =>
    intro b h
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp h with he | ht
    · subst he
      exact le_trans (le_max_right b x) (le_foldl_max t (max b x))
    · exact ih (max b a) ht

/-- **Every individual walk is bounded by the generator's value.** -/
theorem walk_le_gen {i c ns : ℕ} (hi : i < 135) (hc : c < 13)
    (hkc : kil13 c = false) : walk i c ns 13 0 0 0 ≤ gen ns := by
  have h1 : (if kil13 c then 0 else walk i c ns 13 0 0 0)
      = walk i c ns 13 0 0 0 := by simp [hkc]
  have hin : walk i c ns 13 0 0 0 ≤
      ((List.range 13).map fun c => if kil13 c then 0 else walk i c ns 13 0 0 0).foldl
        max 0 := by
    rw [← h1]
    exact mem_le_foldl_max _ 0 (List.mem_map_of_mem (List.mem_range.mpr hc))
  have hout :
      ((List.range 13).map fun c => if kil13 c then 0 else walk i c ns 13 0 0 0).foldl
        max 0 ≤ gen ns :=
    mem_le_foldl_max _ 0 (List.mem_map_of_mem (List.mem_range.mpr hi))
  exact le_trans hin hout

/-! ## 3. Machine 13's openings are machine 11's, minus gear 13's teeth -/

/-- Every machine-13 opening is a machine-11 opening. -/
theorem exposed11_of_13 {k : ℕ} (h : Machine13.Exposed13 k) :
    Machine11.Exposed11 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1⟩

/-- `kil13` sees only the residue. -/
theorem kil13_congr {a b : ℕ} (h : a % 13 = b % 13) : kil13 a = kil13 b := by
  simp only [kil13, h]

/-- **Gear 13's teeth, exactly**: a machine-11 opening survives to machine 13
iff `kil13` spares it (`6 * 11 = 66 = 5 * 13 + 1`, `6 * 2 = 12 = 13 - 1`). -/
theorem exposed13_iff_11 {k : ℕ} (hk : 1 ≤ k) :
    Machine13.Exposed13 k ↔ (Machine11.Exposed11 k ∧ kil13 k = false) := by
  have h1 : (13 ∣ Census.lo k) ↔ k % 13 = 11 := by simp only [Census.lo]; omega
  have h2 : (13 ∣ Census.hi k) ↔ k % 13 = 2 := by simp only [Census.hi]; omega
  unfold Machine13.Exposed13 Machine11.Exposed11
  rw [h1, h2]
  simp only [kil13, Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
  tauto

/-! ## 4. The walk simulates the machine -/

/-- **THE SOUNDNESS BRIDGE.**  From a machine-13 opening `opSeq13 n`, sitting
at machine-11 index `j`, the walk's state `(k, d, surv)` means "`x + d` is the
`k`-th machine-11 opening after `x`, and exactly `surv` machine-13 openings
lie in `(x, x + d]`".  Under that invariant, any value the walk returns other
than the sentinel is the offset of the `(ns+1)`-st machine-13 opening. -/
theorem walk_sound (ns n j : ℕ) :
    ∀ fuel k d surv,
      Machine11.opSeq (j + k) = Machine13.opSeq n + d →
      Machine13.opSeq (n + surv) ≤ Machine13.opSeq n + d →
      Machine13.opSeq n + d < Machine13.opSeq (n + surv + 1) →
      surv ≤ ns →
      walk (j + 1) (Machine13.opSeq n % 13) ns fuel k d surv ≠ 999 →
      Machine13.opSeq n + walk (j + 1) (Machine13.opSeq n % 13) ns fuel k d surv
        = Machine13.opSeq (n + ns + 1) := by
  intro fuel
  induction fuel with
  | zero =>
    intro k d surv _ _ _ _ hne
    exact absurd rfl hne
  | succ fuel ih =>
    intro k d surv h11 hlo hhi hsn hne
    -- the next machine-11 opening after `x + d`
    have hstep : gAt (j + 1 + k) = Machine11.g11 (j + k) := by
      rw [show j + 1 + k = (j + k) + 1 by omega]; exact gAt_succ (j + k)
    have hmono := Machine11.opSeq_lt_succ (j + k)
    have hg : Machine11.g11 (j + k)
        = Machine11.opSeq (j + k + 1) - Machine11.opSeq (j + k) := rfl
    have hnext : Machine11.opSeq (j + k + 1)
        = Machine13.opSeq n + (d + gAt (j + 1 + k)) := by
      rw [hstep]; omega
    have hdpos : 0 < gAt (j + 1 + k) := by rw [hstep]; omega
    rw [walk_succ] at hne ⊢
    by_cases hcap : 30 < d + gAt (j + 1 + k)
    · rw [if_pos hcap] at hne; exact absurd rfl hne
    rw [if_neg hcap] at hne ⊢
    -- the new point is a machine-11 opening, and it is positive
    have hE11 : Machine11.Exposed11 (Machine13.opSeq n + (d + gAt (j + 1 + k))) := by
      rw [← hnext]; exact Machine11.opSeq_exposed _
    have hpos : 1 ≤ Machine13.opSeq n + (d + gAt (j + 1 + k)) := by
      have := Machine13.opSeq_pos n; omega
    have hkc : kil13 (Machine13.opSeq n % 13 + (d + gAt (j + 1 + k)))
        = kil13 (Machine13.opSeq n + (d + gAt (j + 1 + k))) :=
      kil13_congr (by omega)
    -- no machine-13 opening hides strictly between the two machine-11 openings
    have hzE13 : Machine13.Exposed13 (Machine13.opSeq (n + surv + 1)) :=
      Machine13.opSeq_exposed _
    have hzge : Machine13.opSeq n + (d + gAt (j + 1 + k))
        ≤ Machine13.opSeq (n + surv + 1) := by
      rcases Nat.lt_or_ge (Machine13.opSeq (n + surv + 1))
        (Machine13.opSeq n + (d + gAt (j + 1 + k))) with hcon | hok
      · exact absurd (exposed11_of_13 hzE13)
          (Machine11.opSeq_gap_empty (j + k) _ (by omega) (by omega))
      · exact hok
    rcases Bool.eq_false_or_eq_true
      (kil13 (Machine13.opSeq n % 13 + (d + gAt (j + 1 + k)))) with hk1 | hk0
    · -- the new point is KILLED by gear 13: walk on, invariant preserved
      rw [if_pos hk1] at hne ⊢
      have hyN : ¬ Machine13.Exposed13 (Machine13.opSeq n + (d + gAt (j + 1 + k))) := by
        intro hc
        have h2 := ((exposed13_iff_11 hpos).mp hc).2
        rw [← hkc] at h2
        rw [h2] at hk1
        exact Bool.noConfusion hk1
      have hzne : Machine13.opSeq (n + surv + 1)
          ≠ Machine13.opSeq n + (d + gAt (j + 1 + k)) := by
        intro hc; exact hyN (hc ▸ hzE13)
      exact ih (k + 1) (d + gAt (j + 1 + k)) surv
        (by rw [show j + (k + 1) = j + k + 1 by omega]; exact hnext)
        (by omega) (by omega) hsn hne
    · -- the new point SURVIVES: it is the (surv+1)-st machine-13 opening
      have hnot : ¬ (kil13 (Machine13.opSeq n % 13 + (d + gAt (j + 1 + k))) = true) := by
        rw [hk0]; simp
      rw [if_neg hnot] at hne ⊢
      have hyE13 : Machine13.Exposed13 (Machine13.opSeq n + (d + gAt (j + 1 + k))) :=
        (exposed13_iff_11 hpos).mpr ⟨hE11, by rw [← hkc]; exact hk0⟩
      have hzle : Machine13.opSeq (n + surv + 1)
          ≤ Machine13.opSeq n + (d + gAt (j + 1 + k)) := by
        show Machine13.nextOp (Machine13.opSeq (n + surv)) ≤ _
        exact Nat.find_min' (Machine13.exists_exposed_above _) ⟨by omega, hyE13⟩
      have hzeq : Machine13.opSeq (n + surv + 1)
          = Machine13.opSeq n + (d + gAt (j + 1 + k)) := by omega
      by_cases hsurv : surv = ns
      · rw [if_pos (show (surv == ns) = true by simp [hsurv])]
        rw [← hsurv]
        exact hzeq.symm
      · rw [if_neg (show ¬ ((surv == ns) = true) by simp [hsurv])] at hne ⊢
        have hmono13 := Machine13.opSeq_lt_succ (n + surv + 1)
        exact ih (k + 1) (d + gAt (j + 1 + k)) (surv + 1)
          (by rw [show j + (k + 1) = j + k + 1 by omega]; exact hnext)
          (by rw [show n + (surv + 1) = n + surv + 1 by omega]; omega)
          (by rw [show n + (surv + 1) + 1 = (n + surv + 1) + 1 by omega]; omega)
          (by omega) hne

/-! ## 5. The spectrum of machine 13, from machine 11's word -/

/-- **The generator bounds machine 13's spectrum.**  `gen ns < 999` says no
walk bailed at the span cap, and then every window of `ns + 1` consecutive
machine-13 gaps is one of the walks. -/
theorem spectrum_of_gen {ns : ℕ} (hgen : gen ns < 999) :
    Spectrum.SpectrumBound Machine13.g13 (ns + 1) (gen ns) := by
  intro n
  rw [Machine13.windowSum_g13, show n + (ns + 1) = n + ns + 1 by omega]
  have hx1 : 1 ≤ Machine13.opSeq n := Machine13.opSeq_pos n
  have hE13 : Machine13.Exposed13 (Machine13.opSeq n) := Machine13.opSeq_exposed n
  obtain ⟨j, hj⟩ := Machine11.opSeq_surj hx1 (exposed11_of_13 hE13)
  have hc : Machine13.opSeq n % 13 < 13 := Nat.mod_lt _ (by omega)
  have hkc : kil13 (Machine13.opSeq n % 13) = false := by
    rw [kil13_congr (show (Machine13.opSeq n % 13) % 13 = Machine13.opSeq n % 13 by omega)]
    exact ((exposed13_iff_11 hx1).mp hE13).2
  have hjm : (j + 1) % 135 < 135 := Nat.mod_lt _ (by omega)
  have hle := walk_le_gen (ns := ns) hjm hc hkc
  rw [walk_mod] at hle
  have hne : walk (j + 1) (Machine13.opSeq n % 13) ns 13 0 0 0 ≠ 999 := by omega
  have hs := walk_sound ns n j 13 0 0 0 (by simpa using hj)
    (by simp) (by simpa using Machine13.opSeq_lt_succ n) (Nat.zero_le ns) hne
  omega

/-- **THE GENERATOR IS SOUND AT 11 -> 13.**  Machine 13's whole low spectrum
ladder, derived from machine 11's 135-letter word over its 385-slot period -
no machine-13 period anywhere in the derivation. -/
theorem generator_sound :
    Spectrum.SpectrumBound Machine13.g13 1 11 ∧
      Spectrum.SpectrumBound Machine13.g13 2 16 ∧
      Spectrum.SpectrumBound Machine13.g13 3 23 ∧
      Spectrum.SpectrumBound Machine13.g13 4 26 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have h := spectrum_of_gen (ns := 0) (by rw [gen_zero]; omega)
    rwa [gen_zero] at h
  · have h := spectrum_of_gen (ns := 1) (by rw [gen_one]; omega)
    rwa [gen_one] at h
  · have h := spectrum_of_gen (ns := 2) (by rw [gen_two]; omega)
    rwa [gen_two] at h
  · have h := spectrum_of_gen (ns := 3) (by rw [gen_three]; omega)
    rwa [gen_three] at h

end Gen11

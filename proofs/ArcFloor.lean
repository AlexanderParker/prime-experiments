/-
THE ARC FLOOR FOR ANY SEPARATIONS (E4), AND THE +4 LAWS IN BOTH DIRECTIONS
(Formalist, round 38).

Source: `research/proof/rich_half.md` section 5 (E4, the arc floor with its true
hypothesis `arc >= 2`; E5, the coincidence law) and `docs/proofs/21-collision-laws.md`
(Theorem 1, the linear deficit law `c(g, h; L + gh) = c(g, h; L) + 4`; Theorem 3, the arc
floor, on record there as a CERTIFICATE for the real teeth with no proof and the note
"any correct proof must use `3 a_g = g -+ 1`" - refuted here: the proof uses only
`arc >= 2`).  Neither file 20's Lemma 2 nor any theorem of file 21 was in the kernel
before this file (both files say "Kernel: none"), so nothing is duplicated; the pieces
of file 20's Lemma 2 that E4 and Theorem 1 actually consume are proved here directly
(`card_le_one_of_arc`: a gear with arc `>= L` strikes at most one column of a run of
`L`; `card_strikeSet_add`: one period carries exactly two strikes).

THE MODEL.  A gear is a modulus `g` with two teeth at cyclic separation `s`; with phase
`c` it strikes column `k` iff `k = c` or `k = c + s (mod g)`.  A run of `L` is the
columns `0, ..., L - 1` (phases are free, so any run can be slid to `0`; the document's
`joint_max` quantifies over independent phase pairs, which is exactly what is written
here - no one-orbit / CRT reduction is used for E4).  `maxStrike` / `minStrike` are the
most / fewest columns of the run one gear can strike over its phases; `jointMax` /
`jointMin` the same for the union of two gears over independent phase pairs;
`collision = maxStrike + maxStrike - jointMax` is file 21's `c(g, h; L)` and
`coincidence = minStrike + minStrike - jointMin` is rich_half.md's `k(g, h; n)`.

Phases range over `range (g + 1)`, a full residue system mod `g` for every `g`, so the
definitions are total; only `c % g` matters (`strikeSet_mod`), so the sup / inf over
`range (g + 1)` is the sup / inf over ALL phases (`card_le_maxStrike`, `minStrike_le`,
`card_le_jointMax`, `jointMin_le`, each needing only `0 < g`).

`arc g s = min s (g - s)`, the shorter gap between the teeth around the cycle; it is the
document's `a_g` when `s < g`, and `0` when `s = 0` or `s >= g`.

Imports `TopMachineWheel` only for the two-modulus CRT count
`TopMachine.card_filter_crt` (the manifold library's counting engine); everything else
is elementary.  Zero sorries; no `native_decide`, no `decide`, no `Lean.ofReduceBool`.
-/
import TopMachineWheel
import Mathlib.Data.Nat.Periodic
import Mathlib.Data.Nat.Count

namespace ArcFloor

/-! ## The objects -/

/-- Gear `g` with separation `s` at phase `c` strikes column `k`. -/
def Hit (g s c k : ℕ) : Prop := k ≡ c [MOD g] ∨ k ≡ c + s [MOD g]

instance (g s c : ℕ) : DecidablePred (Hit g s c) := fun _ => by unfold Hit; infer_instance

/-- The columns of the run `[0, L)` struck by gear `g` at phase `c`. -/
def strikeSet (g s c L : ℕ) : Finset ℕ := (Finset.range L).filter (Hit g s c)

/-- The short arc `a_g = min(s, g - s)`: the shorter of the two gaps between the teeth. -/
def arc (g s : ℕ) : ℕ := min s (g - s)

theorem range_succ_nonempty (g : ℕ) : (Finset.range (g + 1)).Nonempty :=
  Finset.nonempty_range_iff.mpr (Nat.succ_ne_zero g)

/-- `max_g(L)`: the most columns of a run of `L` that gear `g` can strike, over its phases. -/
def maxStrike (g s L : ℕ) : ℕ :=
  (Finset.range (g + 1)).sup' (range_succ_nonempty g) fun c => (strikeSet g s c L).card

/-- `min_g(L)`: the fewest columns of a run of `L` that gear `g` must strike, over its phases. -/
def minStrike (g s L : ℕ) : ℕ :=
  (Finset.range (g + 1)).inf' (range_succ_nonempty g) fun c => (strikeSet g s c L).card

/-- Independent phase pairs of two gears. -/
def phasePairs (g h : ℕ) : Finset (ℕ × ℕ) := Finset.range (g + 1) ×ˢ Finset.range (h + 1)

theorem phasePairs_nonempty (g h : ℕ) : (phasePairs g h).Nonempty :=
  (range_succ_nonempty g).product (range_succ_nonempty h)

/-- `joint_max(g, h; L)`: the most columns of a run of `L` two gears can strike between them. -/
def jointMax (g s h t L : ℕ) : ℕ :=
  (phasePairs g h).sup' (phasePairs_nonempty g h) fun p =>
    (strikeSet g s p.1 L ∪ strikeSet h t p.2 L).card

/-- `joint_min(g, h; L)`: the fewest columns of a run of `L` two gears must strike between them. -/
def jointMin (g s h t L : ℕ) : ℕ :=
  (phasePairs g h).inf' (phasePairs_nonempty g h) fun p =>
    (strikeSet g s p.1 L ∪ strikeSet h t p.2 L).card

/-- File 21's collision deficit `c(g, h; L) = max_g + max_h - joint_max`. -/
def collision (g s h t L : ℕ) : ℕ := maxStrike g s L + maxStrike h t L - jointMax g s h t L

/-- rich_half.md's coincidence `k(g, h; L) = min_g + min_h - joint_min`. -/
def coincidence (g s h t L : ℕ) : ℕ := minStrike g s L + minStrike h t L - jointMin g s h t L

/-! ## Phases: only the residue matters, and every phase is represented -/

theorem hit_congr {g s c c' k : ℕ} (h : c ≡ c' [MOD g]) : Hit g s c k ↔ Hit g s c' k := by
  unfold Hit
  constructor
  · rintro (h1 | h1)
    · exact Or.inl (h1.trans h)
    · exact Or.inr (h1.trans (h.add_right s))
  · rintro (h1 | h1)
    · exact Or.inl (h1.trans h.symm)
    · exact Or.inr (h1.trans (h.symm.add_right s))

theorem strikeSet_mod (g s c L : ℕ) : strikeSet g s (c % g) L = strikeSet g s c L :=
  Finset.filter_congr fun _ _ => hit_congr (Nat.mod_modEq c g)

theorem mem_strikeSet {g s c L k : ℕ} : k ∈ strikeSet g s c L ↔ k < L ∧ Hit g s c k := by
  simp only [strikeSet, Finset.mem_filter, Finset.mem_range]

theorem card_strikeSet_le (g s c L : ℕ) : (strikeSet g s c L).card ≤ L :=
  (Finset.card_filter_le _ _).trans (Finset.card_range L).le

/-- The phase equal to the column strikes it: every column can be struck. -/
theorem hit_self (g s k : ℕ) : Hit g s k k := Or.inl (Nat.ModEq.refl k)

theorem mod_mem_range_succ {g : ℕ} (hg : 0 < g) (c : ℕ) : c % g ∈ Finset.range (g + 1) :=
  Finset.mem_range.mpr (by have := Nat.mod_lt c hg; omega)

theorem mem_phasePairs {g h : ℕ} {p : ℕ × ℕ} :
    p ∈ phasePairs g h ↔ p.1 ∈ Finset.range (g + 1) ∧ p.2 ∈ Finset.range (h + 1) :=
  Finset.mem_product

theorem card_le_maxStrike {g : ℕ} (hg : 0 < g) (s c L : ℕ) :
    (strikeSet g s c L).card ≤ maxStrike g s L := by
  rw [← strikeSet_mod g s c]
  exact Finset.le_sup' (fun c => (strikeSet g s c L).card) (mod_mem_range_succ hg c)

theorem minStrike_le {g : ℕ} (hg : 0 < g) (s c L : ℕ) :
    minStrike g s L ≤ (strikeSet g s c L).card := by
  rw [← strikeSet_mod g s c]
  exact Finset.inf'_le (fun c => (strikeSet g s c L).card) (mod_mem_range_succ hg c)

theorem card_le_jointMax {g h : ℕ} (hg : 0 < g) (hh : 0 < h) (s t c d L : ℕ) :
    (strikeSet g s c L ∪ strikeSet h t d L).card ≤ jointMax g s h t L := by
  rw [← strikeSet_mod g s c, ← strikeSet_mod h t d]
  exact Finset.le_sup' (fun p : ℕ × ℕ => (strikeSet g s p.1 L ∪ strikeSet h t p.2 L).card)
    ((mem_phasePairs (p := (c % g, d % h))).mpr ⟨mod_mem_range_succ hg c, mod_mem_range_succ hh d⟩)

theorem jointMin_le {g h : ℕ} (hg : 0 < g) (hh : 0 < h) (s t c d L : ℕ) :
    jointMin g s h t L ≤ (strikeSet g s c L ∪ strikeSet h t d L).card := by
  rw [← strikeSet_mod g s c, ← strikeSet_mod h t d]
  exact Finset.inf'_le (fun p : ℕ × ℕ => (strikeSet g s p.1 L ∪ strikeSet h t p.2 L).card)
    ((mem_phasePairs (p := (c % g, d % h))).mpr ⟨mod_mem_range_succ hg c, mod_mem_range_succ hh d⟩)

theorem exists_maxStrike (g s L : ℕ) : ∃ c, maxStrike g s L = (strikeSet g s c L).card := by
  obtain ⟨c, -, hc⟩ := Finset.exists_mem_eq_sup' (range_succ_nonempty g)
    (fun c => (strikeSet g s c L).card)
  exact ⟨c, hc⟩

theorem exists_minStrike (g s L : ℕ) : ∃ c, minStrike g s L = (strikeSet g s c L).card := by
  obtain ⟨c, -, hc⟩ := Finset.exists_mem_eq_inf' (range_succ_nonempty g)
    (fun c => (strikeSet g s c L).card)
  exact ⟨c, hc⟩

theorem exists_jointMax (g s h t L : ℕ) :
    ∃ c d, jointMax g s h t L = (strikeSet g s c L ∪ strikeSet h t d L).card := by
  obtain ⟨p, -, hp⟩ := Finset.exists_mem_eq_sup' (phasePairs_nonempty g h)
    (fun p : ℕ × ℕ => (strikeSet g s p.1 L ∪ strikeSet h t p.2 L).card)
  exact ⟨p.1, p.2, hp⟩

theorem exists_jointMin (g s h t L : ℕ) :
    ∃ c d, jointMin g s h t L = (strikeSet g s c L ∪ strikeSet h t d L).card := by
  obtain ⟨p, -, hp⟩ := Finset.exists_mem_eq_inf' (phasePairs_nonempty g h)
    (fun p : ℕ × ℕ => (strikeSet g s p.1 L ∪ strikeSet h t p.2 L).card)
  exact ⟨p.1, p.2, hp⟩

theorem maxStrike_le_of {g s L n : ℕ} (hn : ∀ c, (strikeSet g s c L).card ≤ n) :
    maxStrike g s L ≤ n := by
  unfold maxStrike
  exact Finset.sup'_le _ _ fun c _ => hn c

theorem le_minStrike_of {g s L n : ℕ} (hn : ∀ c, n ≤ (strikeSet g s c L).card) :
    n ≤ minStrike g s L := by
  unfold minStrike
  exact Finset.le_inf' _ _ fun c _ => hn c

theorem jointMax_le_of {g s h t L n : ℕ}
    (hn : ∀ c d, (strikeSet g s c L ∪ strikeSet h t d L).card ≤ n) :
    jointMax g s h t L ≤ n := by
  unfold jointMax
  exact Finset.sup'_le _ _ fun p _ => hn p.1 p.2

theorem le_jointMin_of {g s h t L n : ℕ}
    (hn : ∀ c d, n ≤ (strikeSet g s c L ∪ strikeSet h t d L).card) :
    n ≤ jointMin g s h t L := by
  unfold jointMin
  exact Finset.le_inf' _ _ fun p _ => hn p.1 p.2

/-- `c(g, h; L) >= 0`: the union is at most the sum. -/
theorem jointMax_le_add {g h : ℕ} (hg : 0 < g) (hh : 0 < h) (s t L : ℕ) :
    jointMax g s h t L ≤ maxStrike g s L + maxStrike h t L := by
  obtain ⟨c, d, hcd⟩ := exists_jointMax g s h t L
  rw [hcd]
  exact (Finset.card_union_le _ _).trans
    (Nat.add_le_add (card_le_maxStrike hg s c L) (card_le_maxStrike hh t d L))

/-- `k(g, h; L) >= 0`: the fewest joint strikes are at most the two separate minima. -/
theorem jointMin_le_add {g h : ℕ} (hg : 0 < g) (hh : 0 < h) (s t L : ℕ) :
    jointMin g s h t L ≤ minStrike g s L + minStrike h t L := by
  obtain ⟨c, hc⟩ := exists_minStrike g s L
  obtain ⟨d, hd⟩ := exists_minStrike h t L
  rw [hc, hd]
  exact (jointMin_le hg hh s t c d L).trans (Finset.card_union_le _ _)

theorem jointMax_comm (g s h t L : ℕ) : jointMax g s h t L = jointMax h t g s L := by
  apply le_antisymm
  · unfold jointMax
    exact Finset.sup'_le _ _ fun p hp => by
      rw [Finset.union_comm]
      have hp' := mem_phasePairs.mp hp
      exact Finset.le_sup' (fun q : ℕ × ℕ => (strikeSet h t q.1 L ∪ strikeSet g s q.2 L).card)
        ((mem_phasePairs (p := (p.2, p.1))).mpr ⟨hp'.2, hp'.1⟩)
  · unfold jointMax
    exact Finset.sup'_le _ _ fun p hp => by
      rw [Finset.union_comm]
      have hp' := mem_phasePairs.mp hp
      exact Finset.le_sup' (fun q : ℕ × ℕ => (strikeSet g s q.1 L ∪ strikeSet h t q.2 L).card)
        ((mem_phasePairs (p := (p.2, p.1))).mpr ⟨hp'.2, hp'.1⟩)

/-! ## E4, step (i): a gear whose arc is at least `L` strikes at most one column of a run of `L` -/

/-- Two struck columns of one gear are at least an arc apart (trivial when `arc = 0`). -/
theorem arc_le_dist {g s c a b : ℕ} (hg : 0 < g) (hab : a < b)
    (ha : Hit g s c a) (hb : Hit g s c b) : arc g s ≤ b - a := by
  unfold arc
  have hsame : ∀ x y : ℕ, x < y → x ≡ y [MOD g] → g ≤ y - x := fun x y hxy hmod =>
    Nat.le_of_dvd (by omega) ((Nat.modEq_iff_dvd' hxy.le).mp hmod)
  rcases ha with ha | ha <;> rcases hb with hb | hb
  · have := hsame a b hab (ha.trans hb.symm)
    omega
  · -- `a ≡ c`, `b ≡ c + s`: `b ≡ a + s`
    have h1 : a + s ≡ b [MOD g] := (ha.add_right s).trans hb.symm
    by_cases hsg : s < g
    · rcases le_or_gt (a + s) b with h2 | h2
      · omega
      · have := hsame b (a + s) h2 h1.symm
        omega
    · omega
  · -- `a ≡ c + s`, `b ≡ c`: `a ≡ b + s`
    have h1 : b + s ≡ a [MOD g] := (hb.add_right s).trans ha.symm
    have := hsame a (b + s) (by omega) h1.symm
    omega
  · have := hsame a b hab (ha.trans hb.symm)
    omega

/-- Step (i): with `L <= a_h`, gear `h` strikes at most one column of the run, at any phase. -/
theorem card_le_one_of_arc {h t d L : ℕ} (hh : 0 < h) (hL : L ≤ arc h t) :
    (strikeSet h t d L).card ≤ 1 := by
  rw [Finset.card_le_one]
  intro a ha b hb
  obtain ⟨ha1, ha2⟩ := mem_strikeSet.mp ha
  obtain ⟨hb1, hb2⟩ := mem_strikeSet.mp hb
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hab | hab
  · have := arc_le_dist hh hab ha2 hb2
    omega
  · have := arc_le_dist hh hab hb2 ha2
    omega

/-- Step (i), the value: `max_h(L) = 1` for `1 <= L <= a_h`. -/
theorem maxStrike_eq_one {h t L : ℕ} (hL1 : 1 ≤ L) (hL : L ≤ arc h t) : maxStrike h t L = 1 := by
  have hh : 0 < h := by unfold arc at hL; omega
  apply le_antisymm
  · exact maxStrike_le_of fun d => card_le_one_of_arc hh hL
  · calc 1 ≤ (strikeSet h t 0 L).card :=
          Finset.card_pos.mpr ⟨0, mem_strikeSet.mpr ⟨hL1, hit_self h t 0⟩⟩
      _ ≤ maxStrike h t L := card_le_maxStrike hh t 0 L

/-! ## E4, step (iii) / (iv): the other gear leaves a column of the run unstruck -/

/-- For `L >= 3` any gear `g >= 3` misses a column of the run (three consecutive columns
have three distinct residues, and there are two teeth); for `L = 2` it misses one unless
its teeth are adjacent (`arc = 1`).  Both at every phase. -/
theorem exists_unstruck {g s L : ℕ} (hg : 3 ≤ g) (hL2 : 2 ≤ L)
    (hside : 3 ≤ L ∨ (s < g ∧ arc g s ≠ 1)) (c : ℕ) :
    ∃ k, k < L ∧ ¬ Hit g s c k := by
  by_contra hcon
  have hcon : ∀ k, k < L → Hit g s c k := fun k hk => by
    by_contra h
    exact hcon ⟨k, hk, h⟩
  have h0 := hcon 0 (by omega)
  have h1 := hcon 1 (by omega)
  unfold Hit Nat.ModEq at h0 h1
  rw [Nat.zero_mod] at h0
  rw [Nat.mod_eq_of_lt (show 1 < g by omega)] at h1
  rcases hside with hL3 | ⟨hsg, harc⟩
  · have h2 := hcon 2 (by omega)
    unfold Hit Nat.ModEq at h2
    rw [Nat.mod_eq_of_lt (show 2 < g by omega)] at h2
    omega
  · unfold arc at harc
    rw [Nat.add_mod, Nat.mod_eq_of_lt hsg] at h0 h1
    rcases h0 with h0 | h0
    · rw [← h0, Nat.zero_add, Nat.mod_eq_of_lt hsg] at h1
      omega
    · rcases h1 with h1 | h1
      · rw [← h1] at h0
        have hdvd := Nat.dvd_of_mod_eq_zero h0.symm
        have := Nat.le_of_dvd (by omega) hdvd
        omega
      · omega

/-! ## E4, the arc floor -/

/-- **E4, the arc floor, one-sided form.**  If gear `h` has arc `a_h >= L >= 2` and the other
gear `g >= 3` cannot fill the run (`L >= 3`, or `L = 2` and `g`'s teeth are not adjacent),
then the two gears do not collide at `L`: `joint_max = max_g + max_h`, i.e. `c(g, h; L) = 0`.
Steps (i)-(iv) of rich_half.md 5.1: `h` strikes at most one column and can strike any;
`g` at a maximising phase leaves a column free; put `h`'s strike there. -/
theorem arc_floor_of_arc_ge {g s h t L : ℕ} (hg : 3 ≤ g) (hL2 : 2 ≤ L) (hL : L ≤ arc h t)
    (hside : 3 ≤ L ∨ (s < g ∧ arc g s ≠ 1)) :
    jointMax g s h t L = maxStrike g s L + maxStrike h t L := by
  have hh : 0 < h := by unfold arc at hL; omega
  have hg0 : 0 < g := by omega
  rw [maxStrike_eq_one (by omega) hL]
  apply le_antisymm
  · have := jointMax_le_add hg0 hh s t L
    rwa [maxStrike_eq_one (by omega) hL] at this
  · obtain ⟨c, hc⟩ := exists_maxStrike g s L
    obtain ⟨k, hkL, hk⟩ := exists_unstruck hg hL2 hside c
    have hnot : k ∉ strikeSet g s c L := fun hmem => hk (mem_strikeSet.mp hmem).2
    calc maxStrike g s L + 1 = (insert k (strikeSet g s c L)).card := by
          rw [Finset.card_insert_of_notMem hnot, hc]
      _ ≤ (strikeSet g s c L ∪ strikeSet h t k L).card :=
          Finset.card_le_card (Finset.insert_subset
            (Finset.mem_union_right _ (mem_strikeSet.mpr ⟨hkL, hit_self h t k⟩))
            Finset.subset_union_left)
      _ ≤ jointMax g s h t L := card_le_jointMax hg0 hh s t c k L

/-- **E4, the arc floor (rich_half.md 5.1), symmetric form.**  Two gears `g, h >= 3` with ANY
separations and short arcs `a_g, a_h`; for every `2 <= L <= max(a_g, a_h)`, the pair does not
collide at `L` - provided, when `L = 2`, that neither gear has adjacent teeth. -/
theorem arc_floor {g s h t L : ℕ} (hg : 3 ≤ g) (hh : 3 ≤ h) (hL2 : 2 ≤ L)
    (hL : L ≤ max (arc g s) (arc h t))
    (hside : 3 ≤ L ∨ (s < g ∧ t < h ∧ arc g s ≠ 1 ∧ arc h t ≠ 1)) :
    jointMax g s h t L = maxStrike g s L + maxStrike h t L := by
  rcases le_max_iff.mp hL with hL' | hL'
  · rw [jointMax_comm, arc_floor_of_arc_ge hh hL2 hL' (by
      rcases hside with h3 | ⟨_, h1, _, h2⟩
      · exact Or.inl h3
      · exact Or.inr ⟨h1, h2⟩), add_comm]
  · exact arc_floor_of_arc_ge hg hL2 hL' (by
      rcases hside with h3 | ⟨h1, _, h2, _⟩
      · exact Or.inl h3
      · exact Or.inr ⟨h1, h2⟩)

/-- E4 in the document's words, first clause: `c(g, h; L) = 0` for every `3 <= L <= max(a_g, a_h)`. -/
theorem collision_eq_zero_of_three_le {g s h t L : ℕ} (hg : 3 ≤ g) (hh : 3 ≤ h) (hL3 : 3 ≤ L)
    (hL : L ≤ max (arc g s) (arc h t)) : collision g s h t L = 0 := by
  unfold collision
  rw [arc_floor hg hh (by omega) hL (Or.inl hL3)]
  exact Nat.sub_self _

/-- E4 in the document's words, second clause: `c(g, h; 2) = 0` when neither arc is `1`.
The document's convention has both arcs `>= 1` (distinct teeth); the proof needs distinct
teeth for ONE gear only (`0 < t < h` gives `a_h >= 2` and so `2 <= max`), and `s < g`. -/
theorem collision_two_eq_zero {g s h t : ℕ} (hg : 3 ≤ g) (hh : 3 ≤ h)
    (hsg : s < g) (ht : 0 < t) (hth : t < h)
    (h1 : arc g s ≠ 1) (h2 : arc h t ≠ 1) : collision g s h t 2 = 0 := by
  have hmax : 2 ≤ max (arc g s) (arc h t) := by unfold arc at h1 h2 ⊢; omega
  unfold collision
  rw [arc_floor hg hh le_rfl hmax (Or.inr ⟨hsg, hth, h1, h2⟩)]
  exact Nat.sub_self _

/-- E4, "in particular": the floor of file 21's Theorem 3 holds whenever `a_g, a_h >= 2`,
with no other hypothesis (`arc >= 2` already forces `g >= 4` and `s < g`). -/
theorem collision_eq_zero_of_arcs {g s h t L : ℕ} (ha : 2 ≤ arc g s) (hb : 2 ≤ arc h t)
    (hL2 : 2 ≤ L) (hL : L ≤ max (arc g s) (arc h t)) : collision g s h t L = 0 := by
  have hg : 3 ≤ g := by unfold arc at ha; omega
  have hh : 3 ≤ h := by unfold arc at hb; omega
  unfold collision
  rw [arc_floor hg hh hL2 hL (by
    rcases Nat.lt_or_ge L 3 with h3 | h3
    · exact Or.inr ⟨by unfold arc at ha; omega, by unfold arc at hb; omega, by omega, by omega⟩
    · exact Or.inl h3)]
  exact Nat.sub_self _

/-- The real separation `3 s = 1 (mod g)` has arc `>= 2` for every `g >= 5`: the real teeth
satisfy E4's hypothesis, which is all file 21 saw in them. -/
theorem arc_real_ge_two {g s : ℕ} (hg : 5 ≤ g) (hsg : s < g) (hreal : 3 * s ≡ 1 [MOD g]) :
    2 ≤ arc g s := by
  unfold arc
  unfold Nat.ModEq at hreal
  rw [Nat.mod_eq_of_lt (show 1 < g by omega)] at hreal
  have h0 : s ≠ 0 := by
    rintro rfl
    rw [Nat.mul_zero, Nat.zero_mod] at hreal
    omega
  have h1 : s ≠ 1 := by
    rintro rfl
    rw [Nat.mul_one, Nat.mod_eq_of_lt (show 3 < g by omega)] at hreal
    omega
  have h2 : s ≠ g - 1 := by
    rintro rfl
    have e : 3 * (g - 1) = 2 * g + (g - 3) := by omega
    rw [e, Nat.mul_add_mod_of_lt (by omega)] at hreal
    omega
  omega

/-! ## The `L = 2` exception: exactly one arc equal to `1` gives `c(g, h; 2) = 1` -/

theorem maxStrike_two_of_arc_one {g s : ℕ} (h1 : arc g s = 1) : maxStrike g s 2 = 2 := by
  unfold arc at h1
  have hg : 0 < g := by omega
  apply le_antisymm
  · exact maxStrike_le_of fun c => card_strikeSet_le g s c 2
  · obtain ⟨c, hc0, hc1⟩ : ∃ c, Hit g s c 0 ∧ Hit g s c 1 := by
      rcases (show s = 1 ∨ s = g - 1 by omega) with rfl | rfl
      · exact ⟨0, Or.inl (Nat.ModEq.refl 0), Or.inr (Nat.ModEq.refl 1)⟩
      · refine ⟨1, Or.inr ?_, Or.inl (Nat.ModEq.refl 1)⟩
        show 0 % g = (1 + (g - 1)) % g
        rw [show 1 + (g - 1) = g by omega, Nat.mod_self, Nat.zero_mod]
    have hset : strikeSet g s c 2 = Finset.range 2 := Finset.filter_true_of_mem fun k hk => by
      rcases (show k = 0 ∨ k = 1 by have := Finset.mem_range.mp hk; omega) with rfl | rfl
      · exact hc0
      · exact hc1
    calc 2 = (strikeSet g s c 2).card := by rw [hset, Finset.card_range]
      _ ≤ maxStrike g s 2 := card_le_maxStrike hg s c 2

/-- The failure at `L = 2` is exactly as the document says: an arc-`1` gear against a gear
of arc `>= 2` collides at `L = 2` with deficit exactly `1`. -/
theorem collision_two_of_arc_one {g s h t : ℕ} (h1 : arc g s = 1) (h2 : 2 ≤ arc h t) :
    collision g s h t 2 = 1 := by
  have hg : 0 < g := by unfold arc at h1; omega
  have hh : 0 < h := by unfold arc at h2; omega
  unfold collision
  rw [maxStrike_two_of_arc_one h1, maxStrike_eq_one (by norm_num) h2]
  have hJ : jointMax g s h t 2 = 2 := by
    apply le_antisymm
    · exact jointMax_le_of fun c d =>
        (Finset.card_le_card (Finset.union_subset (Finset.filter_subset _ _)
          (Finset.filter_subset _ _))).trans (Finset.card_range 2).le
    · obtain ⟨c, hc⟩ := exists_maxStrike g s 2
      rw [maxStrike_two_of_arc_one h1] at hc
      calc 2 = (strikeSet g s c 2).card := hc
        _ ≤ (strikeSet g s c 2 ∪ strikeSet h t 0 2).card :=
            Finset.card_le_card Finset.subset_union_left
        _ ≤ jointMax g s h t 2 := card_le_jointMax hg hh s t c 0 2
  rw [hJ]

/-! ## The period: one gear carries exactly two strikes per `g` columns, at every phase -/

theorem hit_periodic (g s c : ℕ) : Function.Periodic (Hit g s c) g := by
  intro k
  apply propext
  unfold Hit Nat.ModEq
  rw [Nat.add_mod_right]

/-- Two teeth: exactly two residues of a full period are struck. -/
theorem card_filter_hit_range {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (c : ℕ) :
    ((Finset.range g).filter (Hit g s c)).card = 2 := by
  have hset : (Finset.range g).filter (Hit g s c) = {c % g, (c + s) % g} := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert, Finset.mem_singleton,
      Hit, Nat.ModEq]
    constructor
    · rintro ⟨hk, h | h⟩
      · left; rw [← h, Nat.mod_eq_of_lt hk]
      · right; rw [← h, Nat.mod_eq_of_lt hk]
    · rintro (rfl | rfl)
      · exact ⟨Nat.mod_lt _ hg, Or.inl (Nat.mod_mod _ _)⟩
      · exact ⟨Nat.mod_lt _ hg, Or.inr (Nat.mod_mod _ _)⟩
  rw [hset, Finset.card_pair_eq_two_iff]
  intro heq
  have := Nat.sub_mod_eq_zero_of_mod_eq heq.symm
  rw [Nat.add_sub_cancel_left, Nat.mod_eq_of_lt hsg] at this
  omega

theorem card_filter_not_hit_range {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (c : ℕ) :
    ((Finset.range g).filter (fun k => ¬ Hit g s c k)).card = g - 2 := by
  have := Finset.card_filter_add_card_filter_not (s := Finset.range g) (Hit g s c)
  rw [card_filter_hit_range hg hs hsg, Finset.card_range] at this
  omega

theorem card_filter_hit_Ico {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (c n : ℕ) :
    ((Finset.Ico n (n + g)).filter (Hit g s c)).card = 2 := by
  rw [Nat.filter_Ico_card_eq_of_periodic n g (Hit g s c) (hit_periodic g s c),
    Nat.count_eq_card_filter_range]
  exact card_filter_hit_range hg hs hsg c

theorem card_filter_hit_Ico_mul {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (c n : ℕ) :
    ∀ m : ℕ, ((Finset.Ico n (n + g * m)).filter (Hit g s c)).card = 2 * m := by
  intro m
  induction m with
  | zero => simp
  | succ m ih =>
    have hsplit : Finset.Ico n (n + g * (m + 1))
        = Finset.Ico n (n + g * m) ∪ Finset.Ico (n + g * m) (n + g * m + g) := by
      rw [Finset.Ico_union_Ico_eq_Ico (by omega) (by omega)]
      congr 1
      ring
    rw [hsplit, Finset.filter_union, Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (Finset.Ico_disjoint_Ico_consecutive _ _ _)), ih,
      card_filter_hit_Ico hg hs hsg]
    ring

/-- File 20's Lemma 2 in the form Theorem 1 uses: one gear's count on a run grows by exactly
`2m` per `m` periods, at every phase. -/
theorem card_strikeSet_add {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (c L m : ℕ) :
    (strikeSet g s c (L + g * m)).card = (strikeSet g s c L).card + 2 * m := by
  unfold strikeSet
  rw [Finset.range_eq_Ico, Finset.range_eq_Ico,
    ← Finset.Ico_union_Ico_eq_Ico (Nat.zero_le L) (Nat.le_add_right L (g * m)),
    Finset.filter_union, Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (Finset.Ico_disjoint_Ico_consecutive _ _ _)),
    card_filter_hit_Ico_mul hg hs hsg c L m]

/-! ## The union of two coprime gears carries exactly `2g + 2h - 4` marks per period `gh` -/

theorem hitU_periodic (g s c h t d : ℕ) :
    Function.Periodic (fun k => Hit g s c k ∨ Hit h t d k) (g * h) := by
  intro k
  have e1 : Hit g s c (k + g * h) = Hit g s c k := by
    rw [mul_comm]; exact (hit_periodic g s c).nat_mul h k
  have e2 : Hit h t d (k + g * h) = Hit h t d k := (hit_periodic h t d).nat_mul g k
  simp only [e1, e2]

/-- Step 3 of file 21's Theorem 1: `|U| = gh - (g - 2)(h - 2) = 2g + 2h - 4`, by CRT. -/
theorem card_filter_union_range {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (c d : ℕ) :
    ((Finset.range (g * h)).filter (fun k => Hit g s c k ∨ Hit h t d k)).card + 4
      = 2 * g + 2 * h := by
  have hnot : ((Finset.range (g * h)).filter (fun k => ¬ (Hit g s c k ∨ Hit h t d k))).card
      = (g - 2) * (h - 2) := by
    have hfil : (Finset.range (g * h)).filter (fun k => ¬ (Hit g s c k ∨ Hit h t d k))
        = (Finset.range (g * h)).filter (fun k => ¬ Hit g s c k ∧ ¬ Hit h t d k) :=
      Finset.filter_congr fun _ _ => not_or
    rw [hfil, TopMachine.card_filter_crt hg hh hcop (fun k => ¬ Hit g s c k)
      (fun k => ¬ Hit h t d k)
      (fun x y hxy => by simp only [Hit, Nat.ModEq, hxy])
      (fun x y hxy => by simp only [Hit, Nat.ModEq, hxy]),
      card_filter_not_hit_range hg hs hsg, card_filter_not_hit_range hh ht hth]
  have hall := Finset.card_filter_add_card_filter_not (s := Finset.range (g * h))
    (fun k => Hit g s c k ∨ Hit h t d k)
  rw [hnot, Finset.card_range] at hall
  obtain ⟨g', rfl⟩ := Nat.exists_eq_add_of_le (show 2 ≤ g by omega)
  obtain ⟨h', rfl⟩ := Nat.exists_eq_add_of_le (show 2 ≤ h by omega)
  have e : (2 + g') * (2 + h') = g' * h' + 2 * g' + 2 * h' + 4 := by ring
  simp only [Nat.add_sub_cancel_left] at hall
  omega

theorem card_filter_union_Ico {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (c d n : ℕ) :
    ((Finset.Ico n (n + g * h)).filter (fun k => Hit g s c k ∨ Hit h t d k)).card + 4
      = 2 * g + 2 * h := by
  rw [Nat.filter_Ico_card_eq_of_periodic n (g * h) _ (hitU_periodic g s c h t d),
    Nat.count_eq_card_filter_range]
  exact card_filter_union_range hg hs hsg hh ht hth hcop c d

/-- Step 2 of Theorem 1, at every phase pair: a run of `L + gh` is a run of `L` plus one full
period, and the period carries exactly `2g + 2h - 4` marks of the union pattern. -/
theorem card_union_add {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (c d L : ℕ) :
    (strikeSet g s c (L + g * h) ∪ strikeSet h t d (L + g * h)).card + 4
      = (strikeSet g s c L ∪ strikeSet h t d L).card + 2 * g + 2 * h := by
  unfold strikeSet
  rw [← Finset.filter_or, ← Finset.filter_or, Finset.range_eq_Ico, Finset.range_eq_Ico,
    ← Finset.Ico_union_Ico_eq_Ico (Nat.zero_le L) (Nat.le_add_right L (g * h)),
    Finset.filter_union, Finset.card_union_of_disjoint
      (Finset.disjoint_filter_filter (Finset.Ico_disjoint_Ico_consecutive _ _ _))]
  have := card_filter_union_Ico hg hs hsg hh ht hth hcop c d L
  omega

/-! ## The +4 laws: max direction (file 21, Theorem 1) and min direction (rich_half.md, E5) -/

/-- `max_g(L + gm) = max_g(L) + 2m`. -/
theorem maxStrike_add {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (L m : ℕ) :
    maxStrike g s (L + g * m) = maxStrike g s L + 2 * m := by
  apply le_antisymm
  · exact maxStrike_le_of fun c => by
      rw [card_strikeSet_add hg hs hsg c L m]
      exact Nat.add_le_add_right (card_le_maxStrike hg s c L) _
  · obtain ⟨c, hc⟩ := exists_maxStrike g s L
    rw [hc, ← card_strikeSet_add hg hs hsg c L m]
    exact card_le_maxStrike hg s c _

/-- `min_g(L + gm) = min_g(L) + 2m`. -/
theorem minStrike_add {g s : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g) (L m : ℕ) :
    minStrike g s (L + g * m) = minStrike g s L + 2 * m := by
  apply le_antisymm
  · obtain ⟨c, hc⟩ := exists_minStrike g s L
    rw [hc, ← card_strikeSet_add hg hs hsg c L m]
    exact minStrike_le hg s c _
  · exact le_minStrike_of fun c => by
      rw [card_strikeSet_add hg hs hsg c L m]
      exact Nat.add_le_add_right (minStrike_le hg s c L) _

/-- `joint_max(L + gh) = joint_max(L) + 2g + 2h - 4`. -/
theorem jointMax_add {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    jointMax g s h t (L + g * h) + 4 = jointMax g s h t L + 2 * g + 2 * h := by
  apply le_antisymm
  · obtain ⟨c, d, hcd⟩ := exists_jointMax g s h t (L + g * h)
    rw [hcd, card_union_add hg hs hsg hh ht hth hcop c d L]
    have := card_le_jointMax hg hh s t c d L
    omega
  · obtain ⟨c, d, hcd⟩ := exists_jointMax g s h t L
    rw [hcd]
    have h1 := card_union_add hg hs hsg hh ht hth hcop c d L
    have h2 := card_le_jointMax hg hh s t c d (L + g * h)
    omega

/-- `joint_min(L + gh) = joint_min(L) + 2g + 2h - 4` (E5, second clause). -/
theorem jointMin_add {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    jointMin g s h t (L + g * h) + 4 = jointMin g s h t L + 2 * g + 2 * h := by
  apply le_antisymm
  · obtain ⟨c, d, hcd⟩ := exists_jointMin g s h t L
    rw [hcd]
    have h1 := card_union_add hg hs hsg hh ht hth hcop c d L
    have h2 := jointMin_le hg hh s t c d (L + g * h)
    omega
  · obtain ⟨c, d, hcd⟩ := exists_jointMin g s h t (L + g * h)
    rw [hcd, card_union_add hg hs hsg hh ht hth hcop c d L]
    have := jointMin_le hg hh s t c d L
    omega

/-- **File 21, Theorem 1 (the linear deficit law), any separations:**
`c(g, h; L + gh) = c(g, h; L) + 4`. -/
theorem collision_add {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    collision g s h t (L + g * h) = collision g s h t L + 4 := by
  unfold collision
  have h1 := maxStrike_add hg hs hsg L h
  have h2 := maxStrike_add hh ht hth L g
  rw [mul_comm h g] at h2
  have h3 := jointMax_add hg hs hsg hh ht hth hcop L
  have h4 := jointMax_le_add hg hh s t L
  rw [h1, h2]
  omega

/-- **E5 (the coincidence law, rich_half.md 5.3), any separations:**
`k(g, h; n + gh) = k(g, h; n) + 4` - Theorem 1 with `max` replaced by `min`. -/
theorem coincidence_add {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    coincidence g s h t (L + g * h) = coincidence g s h t L + 4 := by
  unfold coincidence
  have h1 := minStrike_add hg hs hsg L h
  have h2 := minStrike_add hh ht hth L g
  rw [mul_comm h g] at h2
  have h3 := jointMin_add hg hs hsg hh ht hth hcop L
  have h4 := jointMin_le_add hg hh s t L
  rw [h1, h2]
  omega

/-- Theorem 1 iterated: slope exactly `4` per period. -/
theorem collision_add_mul {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    ∀ n : ℕ, collision g s h t (L + g * h * n) = collision g s h t L + 4 * n := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show L + g * h * (n + 1) = (L + g * h * n) + g * h by ring,
      collision_add hg hs hsg hh ht hth hcop, ih]
    ring

/-- Theorem 1's consequence: `c(g, h; L) >= 4 floor(L / gh)`; every zero of `c` lies in `[1, gh]`. -/
theorem four_mul_div_le_collision {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    4 * (L / (g * h)) ≤ collision g s h t L := by
  have := collision_add_mul hg hs hsg hh ht hth hcop (L % (g * h)) (L / (g * h))
  rw [Nat.mod_add_div L (g * h)] at this
  omega

theorem coincidence_add_mul {g s h t : ℕ} (hg : 0 < g) (hs : 0 < s) (hsg : s < g)
    (hh : 0 < h) (ht : 0 < t) (hth : t < h) (hcop : Nat.Coprime g h) (L : ℕ) :
    ∀ n : ℕ, coincidence g s h t (L + g * h * n) = coincidence g s h t L + 4 * n := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show L + g * h * (n + 1) = (L + g * h * n) + g * h by ring,
      coincidence_add hg hs hsg hh ht hth hcop, ih]
    ring

end ArcFloor

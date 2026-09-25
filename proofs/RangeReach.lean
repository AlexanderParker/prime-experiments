import RangeHandoff
import RangeChain

/-!
# The node chain on twin lower legs

A *node* is the lower leg `p` of a twin pair: `p` and `p + 2` are both prime.
For a node `s`, its *successor node* `s⁺` is the least node strictly above `s`, and `s⁺⁺` is the
successor node of `s⁺`. A *rung* of `s` is a node `r` in the range of machine `s`:
`s < r` and `r + 2 ≤ primorial s`.

`s` is *good* when `s⁺` exists and `s⁺ + 2 ≤ primorial s`.
RANGE (`RangeAll`) is the range statement `RangeStatement q` at every bound `q ≥ 7`.

This file proves:

* `nextNode_unique`, `nextNode_le`, `exists_nextNode`, `exists_prevNode`: the successor node is
  unique, lies at or below every node above `s`, exists once some node lies above `s`; every
  node `s > 29` is the successor node of a node `a` with `29 ≤ a < s`.
* `rung_transport`, `overlap_agree`: a rung of `s` lying above `t ≥ s` is a rung of `t`;
  machines `s ≤ t` spare the same nodes on the overlap of their ranges.
* `rungs_initial`, `rung_ge_next`: the rungs of `s` are an initial run of the nodes above `s`,
  starting at `s⁺`.
* `good_iff_exists_rung`, `good_iff_rangeStatement`: `s` is good iff it has a rung iff
  `RangeStatement s`.
* `good_29`: `29` is good (`29⁺ = 41`, `43 ≤ 210 ≤ primorial 29`).
* `range_iff_all_good`, `rangePrime_iff_all_good`: RANGE (at every `q ≥ 7`, or only at every
  prime `q ≥ 7`) iff every node `s ≥ 29` is good.
* `GoodReach` (nodes reachable from `29` by good steps `s ↦ s⁺`), `goodReach_iff`,
  `GoodReach.initial`: a node is reachable iff every node from `29` up to it (exclusive) is good;
  the reachable nodes are an initial run of the nodes from `29`.
* `reach_unbounded_iff`, `range_iff_reach_unbounded`: the reachable nodes are unbounded iff every
  node `s ≥ 29` is good iff RANGE.
* `NS` (whenever a node `s ≥ 29` is good, `s⁺⁺` exists and `s⁺⁺ + 2 ≤ primorial s`),
  `ns_step`, `all_good_of_ns`, `range_of_ns`, `twins_unbounded_of_ns`: NS alone (the base `29`
  is proved good) makes every node `s ≥ 29` good, hence RANGE, hence twin pairs are unbounded.
  `NSAll` (the unconditional form) implies `NS`.
* `ns_iff_two_rungs`, `ns_iff_no_single_rung`: NS iff every node `s ≥ 29` has two distinct
  rungs iff no node `s ≥ 29` has exactly one rung.
* `drop_at_most_one`, `first_bad`, `bad_iff_terminal`: passing from `a` to `a⁺` drops at most the
  rung `a⁺`; a node `s ≥ 29` that is not good exists iff some node `a ≥ 29` has exactly one rung
  `t` and `t` is not good.
* `not_parent_unique`: a node can be a rung of two nodes (`149` is a rung of `29` and of `59`).
-/

namespace RangeLine

/-- A *node*: the lower leg `p` of a twin pair, `p` and `p + 2` both prime. -/
def TwinNode (p : ℕ) : Prop := p.Prime ∧ (p + 2).Prime

/-- `t` is the *successor node* `s⁺` of `s`: the least node strictly above `s`. -/
def NextNode (s t : ℕ) : Prop := TwinNode t ∧ s < t ∧ ∀ u, s < u → u < t → ¬ TwinNode u

/-- `r` is a *rung* of `s`: a node in the range of machine `s`, `s < r` and
`r + 2 ≤ primorial s`. -/
def Rung (s r : ℕ) : Prop := TwinNode r ∧ s < r ∧ r + 2 ≤ primorial s

/-- `s` is *good*: its successor node `s⁺` exists and `s⁺ + 2 ≤ primorial s`. -/
def Good (s : ℕ) : Prop := ∃ t, NextNode s t ∧ t + 2 ≤ primorial s

/-- RANGE: the range statement at every bound `q ≥ 7`. -/
def RangeAll : Prop := ∀ q, 7 ≤ q → RangeStatement q

/-- `29` is a node (`29` and `31` are prime). -/
theorem twinNode_29 : TwinNode 29 := ⟨by norm_num, by norm_num⟩

/-- Any `x ≤ 210` is at most `primorial q` for `q ≥ 7`, since `primorial 7 = 210`. -/
theorem le_primorial_of_le_210 {x q : ℕ} (hx : x ≤ 210) (hq : 7 ≤ q) : x ≤ primorial q := by
  have h := primorial_le_of_le hq
  rw [primorial_seven] at h
  omega

/-- The successor node is unique. -/
theorem nextNode_unique {s t t' : ℕ} (h : NextNode s t) (h' : NextNode s t') : t = t' := by
  rcases lt_trichotomy t t' with hlt | heq | hgt
  · exact absurd h.1 (h'.2.2 t h.2.1 hlt)
  · exact heq
  · exact absurd h'.1 (h.2.2 t' h'.2.1 hgt)

/-- Every node strictly above `s` is at or above the successor node `s⁺`. -/
theorem nextNode_le {s t u : ℕ} (h : NextNode s t) (hu : TwinNode u) (hsu : s < u) : t ≤ u := by
  by_contra hlt
  exact h.2.2 u hsu (by omega) hu

/-- If some node lies strictly above `s`, the successor node `s⁺` exists. -/
theorem exists_nextNode {s r : ℕ} (hr : TwinNode r) (hsr : s < r) : ∃ t, NextNode s t := by
  classical
  have hex : ∃ u, s < u ∧ TwinNode u := ⟨r, hsr, hr⟩
  have hspec := Nat.find_spec hex
  exact ⟨Nat.find hex, hspec.2, hspec.1, fun u hsu hut hu => Nat.find_min hex hut ⟨hsu, hu⟩⟩

/-- Every node `s > 29` is the successor node of a node `a` with `29 ≤ a` (so `a < s`). -/
theorem exists_prevNode {s : ℕ} (hs : TwinNode s) (h29 : 29 < s) :
    ∃ a, TwinNode a ∧ 29 ≤ a ∧ NextNode a s := by
  classical
  refine ⟨Nat.findGreatest TwinNode (s - 1), ?_, ?_, hs, ?_, ?_⟩
  · exact Nat.findGreatest_spec (P := TwinNode) (m := 29) (by omega) twinNode_29
  · exact Nat.le_findGreatest (P := TwinNode) (by omega) twinNode_29
  · have := Nat.findGreatest_le (P := TwinNode) (s - 1)
    omega
  · intro u hau hus
    exact Nat.findGreatest_is_greatest hau (by omega)

/-- Transport: a rung of `s` lying above `t ≥ s` is a rung of `t`. -/
theorem rung_transport {s t r : ℕ} (hst : s ≤ t) (h : Rung s r) (htr : t < r) : Rung t r :=
  ⟨h.1, htr, le_trans h.2.2 (primorial_le_of_le hst)⟩

/-- Overlap agreement: above `t` and with upper leg `≤ primorial s`, machines `s ≤ t` have the
same rungs. -/
theorem overlap_agree {s t r : ℕ} (hst : s ≤ t) (htr : t < r) (hr : r + 2 ≤ primorial s) :
    Rung s r ↔ Rung t r :=
  ⟨fun h => rung_transport hst h htr, fun h => ⟨h.1, lt_of_le_of_lt hst htr, hr⟩⟩

/-- The rungs of `s` are an initial run of the nodes above `s`: a node between `s` and a rung
of `s` is a rung of `s`. -/
theorem rungs_initial {s a b : ℕ} (hb : Rung s b) (ha : TwinNode a) (hsa : s < a)
    (hab : a ≤ b) : Rung s a :=
  ⟨ha, hsa, by have := hb.2.2; omega⟩

/-- Every rung of `s` is at or above the successor node `s⁺`. -/
theorem rung_ge_next {s t r : ℕ} (h : NextNode s t) (hr : Rung s r) : t ≤ r :=
  nextNode_le h hr.1 hr.2.1

/-- `s` is good iff it has a rung. -/
theorem good_iff_exists_rung {s : ℕ} : Good s ↔ ∃ r, Rung s r := by
  constructor
  · rintro ⟨t, ht, hle⟩
    exact ⟨t, ht.1, ht.2.1, hle⟩
  · rintro ⟨r, hr⟩
    obtain ⟨t, ht⟩ := exists_nextNode hr.1 hr.2.1
    exact ⟨t, ht, (rungs_initial hr ht.1 ht.2.1 (rung_ge_next ht hr)).2.2⟩

/-- `s` is good iff the range statement holds at `s`. -/
theorem good_iff_rangeStatement {s : ℕ} : Good s ↔ RangeStatement s := by
  rw [good_iff_exists_rung]
  constructor
  · rintro ⟨r, hr, hsr, hle⟩
    exact ⟨r, hsr, hle, hr.1, hr.2⟩
  · rintro ⟨r, hsr, hle, hp, hp2⟩
    exact ⟨r, ⟨hp, hp2⟩, hsr, hle⟩

/-- `29` is good: `41` is a rung of `29` (`43 ≤ 210 ≤ primorial 29`). -/
theorem good_29 : Good 29 :=
  good_iff_exists_rung.2 ⟨41, ⟨by norm_num, by norm_num⟩, by norm_num,
    le_primorial_of_le_210 (by norm_num) (by norm_num)⟩

/-- RANGE at every prime `q ≥ 7` makes every node `s ≥ 29` good. -/
theorem all_good_of_rangePrime (h : ∀ q, q.Prime → 7 ≤ q → RangeStatement q) :
    ∀ s, TwinNode s → 29 ≤ s → Good s :=
  fun s hs h29 => good_iff_rangeStatement.2 (h s hs.1 (by omega))

/-- Every node `s ≥ 29` good gives the range statement at every `q ≥ 7`. -/
theorem rangeAll_of_all_good (h : ∀ s, TwinNode s → 29 ≤ s → Good s) : RangeAll := by
  classical
  intro q hq
  by_cases hq29 : q < 29
  · exact twin_serves (p := 29) (by norm_num) (by norm_num) hq29
      (le_primorial_of_le_210 (by norm_num) hq)
  · have haN : TwinNode (Nat.findGreatest TwinNode q) :=
      Nat.findGreatest_spec (P := TwinNode) (m := 29) (by omega) twinNode_29
    have ha29 : 29 ≤ Nat.findGreatest TwinNode q :=
      Nat.le_findGreatest (P := TwinNode) (by omega) twinNode_29
    have haq : Nat.findGreatest TwinNode q ≤ q := Nat.findGreatest_le q
    obtain ⟨r, hr⟩ := good_iff_exists_rung.1 (h _ haN ha29)
    have hqr : q < r := by
      by_contra hle
      exact Nat.findGreatest_is_greatest hr.2.1 (by omega) hr.1
    exact twin_serves hr.1.1 hr.1.2 hqr (le_trans hr.2.2 (primorial_le_of_le haq))

/-- RANGE iff every node `s ≥ 29` is good. -/
theorem range_iff_all_good : RangeAll ↔ ∀ s, TwinNode s → 29 ≤ s → Good s :=
  ⟨fun h => all_good_of_rangePrime (fun q _ hq => h q hq), rangeAll_of_all_good⟩

/-- RANGE at every prime `q ≥ 7` iff every node `s ≥ 29` is good (so the prime form and the
all-`q` form of RANGE are equivalent). -/
theorem rangePrime_iff_all_good :
    (∀ q, q.Prime → 7 ≤ q → RangeStatement q) ↔ ∀ s, TwinNode s → 29 ≤ s → Good s :=
  ⟨all_good_of_rangePrime, fun h q _ hq => rangeAll_of_all_good h q hq⟩

/-- RANGE gives unbounded twin pairs: for every `N` a prime `p > N` with `p + 2` prime. -/
theorem twins_unbounded_of_rangeAll (h : RangeAll) :
    ∀ N, ∃ p, N < p ∧ p.Prime ∧ (p + 2).Prime :=
  range_implies_unbounded (fun q _ hq => h q hq)

/-- Nodes reachable from `29` by good steps: from a reachable `s` with `s⁺ + 2 ≤ primorial s`,
step to `s⁺`. -/
inductive GoodReach : ℕ → Prop
  | root : GoodReach 29
  | step {s t : ℕ} : GoodReach s → NextNode s t → t + 2 ≤ primorial s → GoodReach t

/-- A reachable number is a node `≥ 29`. -/
theorem GoodReach.node {s : ℕ} (h : GoodReach s) : TwinNode s ∧ 29 ≤ s := by
  induction h with
  | root => exact ⟨twinNode_29, le_refl 29⟩
  | step _ ht _ ih => exact ⟨ht.1, by have := ht.2.1; omega⟩

/-- Every node `s` with `29 ≤ s` below a reachable node is good. -/
theorem GoodReach.good_below {b : ℕ} (h : GoodReach b) :
    ∀ s, TwinNode s → 29 ≤ s → s < b → Good s := by
  induction h with
  | root =>
    intro s _ h29 hlt
    omega
  | @step s₀ t _ ht hle ih =>
    intro s hs h29 hlt
    rcases lt_trichotomy s s₀ with h | h | h
    · exact ih s hs h29 h
    · rw [h]
      exact ⟨t, ht, hle⟩
    · exact absurd hs (ht.2.2 s h hlt)

/-- A number is reachable from `29` by good steps iff it is a node `≥ 29` and every node from
`29` up to it (exclusive) is good. -/
theorem goodReach_iff {s : ℕ} :
    GoodReach s ↔ TwinNode s ∧ 29 ≤ s ∧ ∀ u, TwinNode u → 29 ≤ u → u < s → Good u := by
  constructor
  · intro h
    exact ⟨h.node.1, h.node.2, h.good_below⟩
  · induction s using Nat.strong_induction_on with
    | _ s ih =>
      rintro ⟨hs, h29, hgood⟩
      rcases Nat.eq_or_lt_of_le h29 with h | h
      · rw [← h]
        exact GoodReach.root
      · obtain ⟨a, ha, ha29, has⟩ := exists_prevNode hs h
        have hRa : GoodReach a :=
          ih a has.2.1 ⟨ha, ha29, fun u hu hu29 hua => hgood u hu hu29 (lt_trans hua has.2.1)⟩
        obtain ⟨t, hat, hle⟩ := hgood a ha ha29 has.2.1
        have hts : t = s := nextNode_unique hat has
        rw [hts] at hat hle
        exact GoodReach.step hRa hat hle

/-- The reachable nodes are an initial run of the nodes from `29`: a node `a` with
`29 ≤ a ≤ b` for a reachable `b` is reachable. -/
theorem GoodReach.initial {b : ℕ} (h : GoodReach b) :
    ∀ a, TwinNode a → 29 ≤ a → a ≤ b → GoodReach a :=
  fun _ ha h29 hab => goodReach_iff.2
    ⟨ha, h29, fun u hu hu29 hua => h.good_below u hu hu29 (lt_of_lt_of_le hua hab)⟩

/-- The nodes reachable from `29` by good steps are unbounded iff every node `s ≥ 29` is good. -/
theorem reach_unbounded_iff :
    (∀ N, ∃ s, N < s ∧ GoodReach s) ↔ ∀ s, TwinNode s → 29 ≤ s → Good s := by
  constructor
  · intro h s hs h29
    obtain ⟨b, hb, hR⟩ := h s
    exact hR.good_below s hs h29 hb
  · intro h N
    induction N with
    | zero => exact ⟨29, by norm_num, GoodReach.root⟩
    | succ n ih =>
      obtain ⟨s, hs, hR⟩ := ih
      obtain ⟨t, ht, hle⟩ := h s hR.node.1 hR.node.2
      exact ⟨t, by have := ht.2.1; omega, GoodReach.step hR ht hle⟩

/-- RANGE iff the nodes reachable from `29` by good steps are unbounded. -/
theorem range_iff_reach_unbounded : RangeAll ↔ ∀ N, ∃ s, N < s ∧ GoodReach s :=
  range_iff_all_good.trans reach_unbounded_iff.symm

/-- NS (second successor, range form): whenever a node `s ≥ 29` is good, the second successor
`s⁺⁺` exists and `s⁺⁺ + 2 ≤ primorial s`. -/
def NS : Prop :=
  ∀ s, TwinNode s → 29 ≤ s → Good s →
    ∃ t u, NextNode s t ∧ NextNode t u ∧ u + 2 ≤ primorial s

/-- NS, unconditional form: for every node `s ≥ 29`, `s⁺` and `s⁺⁺` exist and
`s⁺⁺ + 2 ≤ primorial s`. -/
def NSAll : Prop :=
  ∀ s, TwinNode s → 29 ≤ s → ∃ t u, NextNode s t ∧ NextNode t u ∧ u + 2 ≤ primorial s

/-- The unconditional form implies NS. -/
theorem ns_of_nsAll (h : NSAll) : NS :=
  fun s hs h29 _ => h s hs h29

/-- The step with NS: if a node `s ≥ 29` is good, then `s⁺` is a rung of `s` and `s⁺` is good
(`s⁺⁺` is a rung of `s⁺`, since `s⁺⁺ + 2 ≤ primorial s ≤ primorial s⁺`). -/
theorem ns_step (hNS : NS) {s t : ℕ} (hs : TwinNode s) (h29 : 29 ≤ s) (hg : Good s)
    (ht : NextNode s t) : Rung s t ∧ Good t := by
  obtain ⟨t', u, ht', hu, hle⟩ := hNS s hs h29 hg
  have htt : t' = t := nextNode_unique ht' ht
  rw [htt] at hu
  have htu := hu.2.1
  refine ⟨⟨ht.1, ht.2.1, by omega⟩, good_iff_exists_rung.2 ⟨u, hu.1, htu, ?_⟩⟩
  exact le_trans hle (primorial_le_of_le (le_of_lt ht.2.1))

/-- NS alone (the base `29` is proved good) makes every node `s ≥ 29` good. -/
theorem all_good_of_ns (hNS : NS) : ∀ s, TwinNode s → 29 ≤ s → Good s := by
  intro s
  induction s using Nat.strong_induction_on with
  | _ s ih =>
    intro hs h29
    rcases Nat.eq_or_lt_of_le h29 with h | h
    · rw [← h]
      exact good_29
    · obtain ⟨a, ha, ha29, has⟩ := exists_prevNode hs h
      exact (ns_step hNS ha ha29 (ih a has.2.1 ha ha29) has).2

/-- NS gives RANGE. -/
theorem range_of_ns (hNS : NS) : RangeAll :=
  rangeAll_of_all_good (all_good_of_ns hNS)

/-- NS gives the range statement at every prime `q ≥ 7`. -/
theorem rangePrime_of_ns (hNS : NS) : ∀ q, q.Prime → 7 ≤ q → RangeStatement q :=
  fun q _ hq => range_of_ns hNS q hq

/-- NS gives unbounded twin pairs: for every `N` a prime `p > N` with `p + 2` prime. -/
theorem twins_unbounded_of_ns (hNS : NS) : ∀ N, ∃ p, N < p ∧ p.Prime ∧ (p + 2).Prime :=
  range_implies_unbounded (rangePrime_of_ns hNS)

/-- NS gives that the nodes reachable from `29` by good steps are unbounded. -/
theorem reach_unbounded_of_ns (hNS : NS) : ∀ N, ∃ s, N < s ∧ GoodReach s :=
  reach_unbounded_iff.2 (all_good_of_ns hNS)

/-- Two distinct rungs of `s` give `s⁺`, `s⁺⁺` with `s⁺⁺ + 2 ≤ primorial s`. -/
theorem second_of_two_rungs {s r r' : ℕ} (hr : Rung s r) (hr' : Rung s r') (hne : r ≠ r') :
    ∃ t u, NextNode s t ∧ NextNode t u ∧ u + 2 ≤ primorial s := by
  have key : ∃ a b, Rung s a ∧ Rung s b ∧ a < b := by
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · exact ⟨r, r', hr, hr', hlt⟩
    · exact ⟨r', r, hr', hr, hgt⟩
  obtain ⟨a, b, ha, hb, hab⟩ := key
  obtain ⟨t, ht⟩ := exists_nextNode ha.1 ha.2.1
  have hta : t ≤ a := rung_ge_next ht ha
  have htb : t < b := lt_of_le_of_lt hta hab
  obtain ⟨u, hu⟩ := exists_nextNode (s := t) hb.1 htb
  have hub : u ≤ b := nextNode_le hu hb.1 htb
  have hsu : s < u := lt_trans ht.2.1 hu.2.1
  exact ⟨t, u, ht, hu, (rungs_initial hb hu.1 hsu hub).2.2⟩

/-- NS iff every node `s ≥ 29` has two distinct rungs. -/
theorem ns_iff_two_rungs :
    NS ↔ ∀ s, TwinNode s → 29 ≤ s → ∃ r r', Rung s r ∧ Rung s r' ∧ r ≠ r' := by
  constructor
  · intro hNS s hs h29
    obtain ⟨t, u, ht, hu, hle⟩ := hNS s hs h29 (all_good_of_ns hNS s hs h29)
    have htu := hu.2.1
    exact ⟨t, u, ⟨ht.1, ht.2.1, by omega⟩, ⟨hu.1, lt_trans ht.2.1 htu, hle⟩, ne_of_lt htu⟩
  · intro h s hs h29 _
    obtain ⟨r, r', hr, hr', hne⟩ := h s hs h29
    exact second_of_two_rungs hr hr' hne

/-- NS iff no node `s ≥ 29` has a single rung: beside any rung of `s` there is another. -/
theorem ns_iff_no_single_rung :
    NS ↔ ∀ s, TwinNode s → 29 ≤ s → ∀ r, Rung s r → ∃ r', Rung s r' ∧ r' ≠ r := by
  constructor
  · intro hNS s hs h29 r _
    obtain ⟨r₁, r₂, h1, h2, hne⟩ := ns_iff_two_rungs.1 hNS s hs h29
    by_cases h1r : r₁ = r
    · exact ⟨r₂, h2, fun h2r => hne (h1r.trans h2r.symm)⟩
    · exact ⟨r₁, h1, h1r⟩
  · intro h s hs h29 hg
    obtain ⟨r, hr⟩ := good_iff_exists_rung.1 hg
    obtain ⟨r', hr', hne⟩ := h s hs h29 r hr
    exact second_of_two_rungs hr hr' (Ne.symm hne)

/-- Drop at most one: every rung of `a` other than `a⁺` is a rung of `a⁺`. -/
theorem drop_at_most_one {a t r : ℕ} (hat : NextNode a t) (hr : Rung a r) (hne : r ≠ t) :
    Rung t r :=
  rung_transport (le_of_lt hat.2.1) hr (lt_of_le_of_ne (rung_ge_next hat hr) (Ne.symm hne))

/-- First bad node: if `a` is good and `a⁺` is not good, then `a⁺` is the only rung of `a`. -/
theorem first_bad {a t : ℕ} (ha : Good a) (hat : NextNode a t) (hbad : ¬ Good t) :
    Rung a t ∧ ∀ r, Rung a r → r = t := by
  obtain ⟨r₀, hr₀⟩ := good_iff_exists_rung.1 ha
  refine ⟨rungs_initial hr₀ hat.1 hat.2.1 (rung_ge_next hat hr₀), ?_⟩
  intro r hr
  by_contra hne
  exact hbad (good_iff_exists_rung.2 ⟨r, drop_at_most_one hat hr hne⟩)

/-- A node `≥ 29` that is not good exists iff some node `a ≥ 29` has exactly one rung `t` and
`t` is not good. -/
theorem bad_iff_terminal :
    (∃ t, TwinNode t ∧ 29 ≤ t ∧ ¬ Good t) ↔
      ∃ a t, TwinNode a ∧ 29 ≤ a ∧ Rung a t ∧ (∀ r, Rung a r → r = t) ∧ ¬ Good t := by
  classical
  constructor
  · intro hex
    have ht := Nat.find_spec hex
    have hleast : ∀ u, u < Nat.find hex → ¬ (TwinNode u ∧ 29 ≤ u ∧ ¬ Good u) :=
      fun u hu => Nat.find_min hex hu
    have ht29 : Nat.find hex ≠ 29 := by
      intro h
      rw [h] at ht
      exact ht.2.2 good_29
    have hlt : 29 < Nat.find hex := lt_of_le_of_ne ht.2.1 (Ne.symm ht29)
    obtain ⟨a, ha, ha29, hat⟩ := exists_prevNode ht.1 hlt
    have hgood : Good a := by
      by_contra hno
      exact hleast a hat.2.1 ⟨ha, ha29, hno⟩
    obtain ⟨h1, h2⟩ := first_bad hgood hat ht.2.2
    exact ⟨a, _, ha, ha29, h1, h2, ht.2.2⟩
  · rintro ⟨a, t, -, ha29, hat, -, hbad⟩
    exact ⟨t, hat.1, by have := hat.2.1; omega, hbad⟩

/-- A node can be a rung of two nodes: `149` is a rung of both `29` and `59`. -/
theorem not_parent_unique :
    ¬ ∀ s t r, TwinNode s → TwinNode t → Rung s r → Rung t r → s = t := by
  intro h
  have h149 : TwinNode 149 := ⟨by norm_num, by norm_num⟩
  have h59 : TwinNode 59 := ⟨by norm_num, by norm_num⟩
  have r1 : Rung 29 149 := ⟨h149, by norm_num, le_primorial_of_le_210 (by norm_num) (by norm_num)⟩
  have r2 : Rung 59 149 := ⟨h149, by norm_num, le_primorial_of_le_210 (by norm_num) (by norm_num)⟩
  have := h 29 59 149 twinNode_29 h59 r1 r2
  norm_num at this

end RangeLine

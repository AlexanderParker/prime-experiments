/-
THE DEPTH-SUM IDENTITY AT MACHINE 13 (round 22).

Lateral's round-20 identity (docs/novel/depth-sum-identity.md):

    sum_{j >= 1} W_j(g)  =  prod_q c_q(g),

`W_j(g)` = the number of `j`-windows of consecutive gaps summing to `g`,
`c_q(g)` = gear `q`'s local pair factor. Their one-line proof has two
independent halves, and this file kernel-checks both of them:

* THE DEPTH PARTITION (abstract, any machine, any gap): every opening pair at
  lag `g` is the endpoint pair of EXACTLY ONE window, because a strictly
  increasing position sequence makes `j |-> pos (a + j) - pos a` injective.
  `window_depth_unique` is that injectivity; `depth_partition` is the sum
  rule it produces - the sum over depths of the per-depth window counts is
  the count of starts that reach `g` at SOME depth. No arithmetic, no
  machine: this is the whole content of "sum over j".

* THE LOCAL FACTOR at machine 13, in Harvester's closed form
  (docs/novel/paired-hlb-cycles.md): `c_q(g) = q - nu_q({0, 2, 6g, 6g+2})` -
  the machine's transfer diagonal IS the Hardy-Littlewood prime-quadruplet
  local factor. Kernel-checked at all four gears of machine 13 for every
  `g < 40` (`local_factor_5/7/11/13`), together with the product form
  `pairCount13 g = c_5 c_7 c_11 c_13` over the whole 5005-slot period
  (`depth_sum_at_13`) - the identity's right-hand side, exact.

HONEST GAP (see formalist.md round 22): the two halves are checked, the GLUE
is not. Turning `depth_partition` into a statement about `pairCount13`
requires "count over one period of the enumeration = count over residues" -
a periodicity/re-indexing bridge for `Machine13.opSeq` that this file does
not build. What is stated here is exactly what is proved.

All numbers verified over the full period first (scratchpad depthsum13.py:
g = 0..59, zero mismatches on both halves).
-/

import Machine13Q
import MergeLaw

namespace DepthSum

/-! ## The depth partition - the "sum over j" half, abstract -/

/-- **At most one depth reaches a given lag.** For a strictly increasing
position sequence, `j |-> windowSum g a j` is strictly increasing in `j`, so
a window from `a` summing to `gap` has a UNIQUE length. (This is the
bijection "every opening pair at lag `g` is the endpoint pair of exactly one
window", with the pair's left endpoint fixed at `pos a`.) -/
theorem window_depth_unique {g pos : ℕ → ℕ}
    (hg : ∀ m, g m = pos (m + 1) - pos m) (hmono : ∀ m, pos m < pos (m + 1))
    {a j1 j2 gap : ℕ} (h1 : Spectrum.windowSum g a j1 = gap)
    (h2 : Spectrum.windowSum g a j2 = gap) : j1 = j2 := by
  have hmono' : ∀ m, pos m ≤ pos (m + 1) := fun m => le_of_lt (hmono m)
  have ht : ∀ j, Spectrum.windowSum g a j = pos (a + j) - pos a :=
    fun j => MergeLaw.windowSum_telescope hg hmono' a j
  have hlt : ∀ x y : ℕ, x < y → pos (a + x) < pos (a + y) := by
    intro x y hxy
    have h3 := hmono (a + x)
    have h4 := MergeLaw.pos_le_add hmono' (a + x + 1) (y - x - 1)
    rw [show a + x + 1 + (y - x - 1) = a + y by omega] at h4
    omega
  have hle : pos a ≤ pos (a + j1) := MergeLaw.pos_le_add hmono' a j1
  have hle2 : pos a ≤ pos (a + j2) := MergeLaw.pos_le_add hmono' a j2
  rw [ht] at h1 h2
  by_contra hne
  rcases Nat.lt_or_ge j1 j2 with h | h
  · have := hlt j1 j2 h; omega
  · have hgt : j2 < j1 := by omega
    have := hlt j2 j1 hgt; omega

/-- The set of window starts below `N` that reach `gap` at SOME depth in
`[1, J)`, as a union over depths. -/
def reachSet (g : ℕ → ℕ) (gap N J : ℕ) : Finset ℕ :=
  (Finset.Ico 1 J).biUnion fun j =>
    (Finset.range N).filter fun a => Spectrum.windowSum g a j = gap

/-- Membership in `reachSet`, spelled out. -/
theorem mem_reachSet {g : ℕ → ℕ} {gap N J a : ℕ} :
    a ∈ reachSet g gap N J ↔
      a < N ∧ ∃ j, 1 ≤ j ∧ j < J ∧ Spectrum.windowSum g a j = gap := by
  simp only [reachSet, Finset.mem_biUnion, Finset.mem_Ico, Finset.mem_filter,
    Finset.mem_range]
  constructor
  · rintro ⟨j, ⟨hj1, hj2⟩, haN, hw⟩
    exact ⟨haN, j, hj1, hj2, hw⟩
  · rintro ⟨haN, j, hj1, hj2, hw⟩
    exact ⟨j, ⟨hj1, hj2⟩, haN, hw⟩

/-- **The depth partition.** The per-depth window counts are disjoint, so
`sum_j W_j(gap)` counts exactly the window STARTS that reach `gap` at some
depth - no double counting at any depth, at any machine. This is the whole
content of the identity's left-hand side. -/
theorem depth_partition {g pos : ℕ → ℕ}
    (hg : ∀ m, g m = pos (m + 1) - pos m) (hmono : ∀ m, pos m < pos (m + 1))
    (gap N J : ℕ) :
    ∑ j ∈ Finset.Ico 1 J,
        ((Finset.range N).filter fun a => Spectrum.windowSum g a j = gap).card
      = (reachSet g gap N J).card := by
  rw [reachSet, Finset.card_biUnion]
  intro x _ y _ hxy
  simp only [Finset.disjoint_left, Finset.mem_filter]
  rintro a ⟨_, hx⟩ ⟨_, hy⟩
  exact hxy (window_depth_unique hg hmono hx hy)

/-! ## The local factor at machine 13 - the `prod_q c_q(g)` half -/

/-- Gear `q` with teeth `t1, t2` leaves residue `r` open. -/
def openR (q t1 t2 r : ℕ) : Bool := r % q != t1 && r % q != t2

/-- **`c_q(g)`**: the number of residues mod `q` at which BOTH ends of a lag-`g`
pair are open - gear `q`'s local pair factor, i.e. one entry of the machine's
transfer diagonal. -/
def cq (q t1 t2 gap : ℕ) : ℕ :=
  ((List.range q).filter fun r => openR q t1 t2 r && openR q t1 t2 (r + gap)).length

/-- **`nu_q({0, 2, 6g, 6g+2})`**: the number of DISTINCT residues mod `q` in
the Hardy-Littlewood prime-quadruplet pattern of the lag-`g` pair. -/
def nuq (q gap : ℕ) : ℕ :=
  (([0, 2, 6 * gap, 6 * gap + 2].map fun x => x % q) : List ℕ).dedup.length

/-- The whole 5005-slot period: openings at lag `gap`, counted. -/
def pairCount13 (gap : ℕ) : ℕ :=
  ((List.range 5005).filter fun k =>
    Machine13.atT (k % 5) (k % 7) (k % 11) (k % 13) 0 &&
      Machine13.atT (k % 5) (k % 7) (k % 11) (k % 13) gap).length

set_option maxRecDepth 40000 in
/-- **Harvester's local-factor identity at gear 5**, kernel-checked:
`c_5(g) = 5 - nu_5({0, 2, 6g, 6g+2})`. -/
theorem local_factor_5 : ∀ gap < 40, cq 5 1 4 gap = 5 - nuq 5 gap := by
  decide +kernel

set_option maxRecDepth 40000 in
/-- Gear 7. -/
theorem local_factor_7 : ∀ gap < 40, cq 7 6 1 gap = 7 - nuq 7 gap := by
  decide +kernel

set_option maxRecDepth 40000 in
/-- Gear 11. -/
theorem local_factor_11 : ∀ gap < 40, cq 11 2 9 gap = 11 - nuq 11 gap := by
  decide +kernel

set_option maxRecDepth 40000 in
/-- Gear 13. -/
theorem local_factor_13 : ∀ gap < 40, cq 13 11 2 gap = 13 - nuq 13 gap := by
  decide +kernel

set_option maxRecDepth 100000 in
/-- **The depth-sum identity's right-hand side at machine 13**, over the whole
5005-slot period: the number of opening pairs at lag `gap` is exactly the
product of the four gears' local factors - the CRT factorisation, checked
term by term for every `gap < 40`. -/
theorem depth_sum_at_13 : ∀ gap < 40,
    pairCount13 gap
      = cq 5 1 4 gap * cq 7 6 1 gap * cq 11 2 9 gap * cq 13 11 2 gap := by
  decide +kernel

/-- **The identity's right-hand side in Hardy-Littlewood form**: the lag-`g`
pair population of machine 13 is `prod_q (q - nu_q({0, 2, 6g, 6g+2}))` - the
prime-quadruplet local factors, exactly. -/
theorem depth_sum_hl_form : ∀ gap < 40,
    pairCount13 gap
      = (5 - nuq 5 gap) * (7 - nuq 7 gap) * (11 - nuq 11 gap)
          * (13 - nuq 13 gap) := by
  intro gap hgap
  rw [depth_sum_at_13 gap hgap, local_factor_5 gap hgap, local_factor_7 gap hgap,
    local_factor_11 gap hgap, local_factor_13 gap hgap]

end DepthSum

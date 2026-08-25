/-
A COVERING LP DUAL CERTIFICATE, KERNEL-CHECKED: `F(19) <= 37`, hence the (D)
step 17->19, with NO period scan (round 23).

This is a SECOND, fully independent proof vehicle for a rung of the (D)
ladder. The ladder in `Ladder.lean` climbs 17->19 through the merge law fed
by machine 17's 85,085-tuple period scan. Here the same rung is proved from
THIRTY-SEVEN INTEGERS and finitely many comparisons, sharing nothing with
that route (docs/novel/covering-lp-certificates.md; the certificate vector
was produced by `research/lp_dual_certs.py` in exact rationals and rescaled
to integers by scratchpad gencert.py, which also re-verified it in exactly
the aggregated form checked below).

THE ARGUMENT, in one paragraph. Suppose 37 consecutive slots `p, .., p+36`
are all blocked. Put `r_q = p % q`; then for each offset `i < 37` at least
one gear `q` has `(r_q + i) mod q` on one of its two teeth. The KOUNIAS
inequality with the distinguished gear 5 says, pointwise,

    1 + #{j != 5 : gear j and gear 5 both block i}  <=  #{q : gear q blocks i}

(an identity when gear 5 blocks, and `1 <= #q` otherwise). Weight position
`i` by `y i >= 0` and sum: the left side is `sum y` plus five PAIR sums, the
right side is six SINGLE sums. Each single sum is at most its maximum over
that gear's phases, each pair sum at least its minimum over phase pairs, so

    sum y + sum_j min P_(5,j)  <=  sum_q max_r S_q(r).

With the weights below that reads `9757 + 2749 <= 12489`, i.e.
`12506 <= 12489` - FALSE. So no 37 consecutive slots are all blocked:
every window of 37 slots contains a machine-19 opening, `F(19) <= 37`.

WHAT IS AND IS NOT USED. `no_37_run` and `F19_le_37` below depend on NOTHING
except the certificate arithmetic and `Machine19.exposed19_iff` (the
definition of an opening as a CRT tuple). No slice, no `sliceAll`, no
`qsliceAll`, no merge law. `D_17_19_lp` additionally reads the budget
`F(17) + 19 = 18 + 19` off `Machine17`'s certificate, which is where the
"17" enters - exactly as in the merge-law route, and nowhere else.

HONEST STATUS. `F(19) <= 37` is WEAKER than `Machine19.gap_le`'s exact
`F(19) = 25`; the point is the METHOD - a scan-free, polynomial-size,
kernel-checkable certificate for a Jacobsthal-type maximal gap, which the
period scans are not. Its cost here is 37 integers, 72 phase evaluations and
335 phase-pair evaluations of a 37-term sum, against a 1,616,615-slot period.
-/

import Machine19
import Machine17

namespace CoveringCert

/-! ## The six gears as blocking predicates on (phase, offset)

Teeth read off `Machine19.expT`: gear `q` blocks the slot whose residue is
`u_q` or `q - u_q`, where `6 u_q = q -+ 1`. -/

def b5 (r i : ℕ) : Bool := ((r + i) % 5 == 1) || ((r + i) % 5 == 4)
def b7 (r i : ℕ) : Bool := ((r + i) % 7 == 6) || ((r + i) % 7 == 1)
def b11 (r i : ℕ) : Bool := ((r + i) % 11 == 2) || ((r + i) % 11 == 9)
def b13 (r i : ℕ) : Bool := ((r + i) % 13 == 11) || ((r + i) % 13 == 2)
def b17 (r i : ℕ) : Bool := ((r + i) % 17 == 3) || ((r + i) % 17 == 14)
def b19 (r i : ℕ) : Bool := ((r + i) % 19 == 16) || ((r + i) % 19 == 3)

/-! ## The certificate -/

/-- **The certificate**: 37 nonnegative integer weights, one per position of
the window (the exact rational Farkas vector of the level-2 covering LP,
scaled by its common denominator 1101). It is a palindrome - the machine's
mirror symmetry `k -> -k`. -/
def ywList : List ℕ :=
  [115, 169, 265, 272, 280, 276, 276, 276, 272, 276, 283, 280, 283, 276, 295,
   283, 276, 272, 307, 272, 276, 283, 295, 276, 283, 280, 283, 276, 272, 276,
   276, 276, 280, 272, 265, 169, 115]

def yw (i : ℕ) : ℕ := ywList.getD i 0

/-- Total weight. -/
def totY : ℕ := ∑ i ∈ Finset.range 37, yw i

/-- The single-gear sum at phase `r`: the weight this gear can absorb. -/
def S (bq : ℕ → ℕ → Bool) (r : ℕ) : ℕ :=
  ∑ i ∈ Finset.range 37, (if bq r i then yw i else 0)

/-- The pair sum for `(5, q)` at phases `(r5, rq)`: the weight gear 5 and
gear `q` block in common. -/
def PP (bq : ℕ → ℕ → Bool) (r5 rq : ℕ) : ℕ :=
  ∑ i ∈ Finset.range 37, (if b5 r5 i && bq rq i then yw i else 0)

/-! ## The finite checks -/

set_option maxRecDepth 20000

theorem tot_eq : totY = 9757 := by decide +kernel

theorem S5_le : ∀ r < 5, S b5 r ≤ 3905 := by decide +kernel
theorem S7_le : ∀ r < 7, S b7 r ≤ 2796 := by decide +kernel
theorem S11_le : ∀ r < 11, S b11 r ≤ 1821 := by decide +kernel
theorem S13_le : ∀ r < 13, S b13 r ≤ 1648 := by decide +kernel
theorem S17_le : ∀ r < 17, S b17 r ≤ 1204 := by decide +kernel
theorem S19_le : ∀ r < 19, S b19 r ≤ 1115 := by decide +kernel

theorem P7_ge : ∀ a < 5, ∀ b < 7, 1101 ≤ PP b7 a b := by decide +kernel
theorem P11_ge : ∀ a < 5, ∀ b < 11, 552 ≤ PP b11 a b := by decide +kernel
theorem P13_ge : ∀ a < 5, ∀ b < 13, 548 ≤ PP b13 a b := by decide +kernel
theorem P17_ge : ∀ a < 5, ∀ b < 17, 276 ≤ PP b17 a b := by decide +kernel
theorem P19_ge : ∀ a < 5, ∀ b < 19, 272 ≤ PP b19 a b := by decide +kernel

/-- **The certificate signs**: `sum_q max_r S_q(r) < sum y + sum_j min P_(5,j)`.
Margin 17 out of 12,489. NO AXIOMS. -/
theorem cert_signs :
    3905 + 2796 + 1821 + 1648 + 1204 + 1115
      < 9757 + (1101 + 552 + 548 + 276 + 272) := by decide

/-! ## Kounias, pointwise -/

/-- **The Kounias inequality with a distinguished event `a`** (here gear 5),
over six events, one of which holds. An identity when `a` holds; the trivial
`1 <= #events` otherwise. -/
theorem kounias (a b c d e f : Bool) (h : (a || b || c || d || e || f) = true)
    (w : ℕ) :
    w + ((if a && b then w else 0) + (if a && c then w else 0) +
         (if a && d then w else 0) + (if a && e then w else 0) +
         (if a && f then w else 0))
      ≤ (if a then w else 0) + (if b then w else 0) + (if c then w else 0) +
        (if d then w else 0) + (if e then w else 0) + (if f then w else 0) := by
  revert h
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> cases f <;>
    simp <;> omega

/-! ## The aggregate -/

/-- Summing the pointwise inequality over the window. -/
theorem cover_bound {r5 r7 r11 r13 r17 r19 : ℕ}
    (hcov : ∀ i < 37, (b5 r5 i || b7 r7 i || b11 r11 i || b13 r13 i ||
      b17 r17 i || b19 r19 i) = true) :
    totY + (PP b7 r5 r7 + PP b11 r5 r11 + PP b13 r5 r13 + PP b17 r5 r17 +
        PP b19 r5 r19)
      ≤ S b5 r5 + S b7 r7 + S b11 r11 + S b13 r13 + S b17 r17 + S b19 r19 := by
  have key : ∀ i ∈ Finset.range 37,
      yw i + ((if b5 r5 i && b7 r7 i then yw i else 0) +
        (if b5 r5 i && b11 r11 i then yw i else 0) +
        (if b5 r5 i && b13 r13 i then yw i else 0) +
        (if b5 r5 i && b17 r17 i then yw i else 0) +
        (if b5 r5 i && b19 r19 i then yw i else 0))
      ≤ (if b5 r5 i then yw i else 0) + (if b7 r7 i then yw i else 0) +
        (if b11 r11 i then yw i else 0) + (if b13 r13 i then yw i else 0) +
        (if b17 r17 i then yw i else 0) + (if b19 r19 i then yw i else 0) := by
    intro i hi
    exact kounias _ _ _ _ _ _ (hcov i (Finset.mem_range.mp hi)) (yw i)
  have hL : totY + (PP b7 r5 r7 + PP b11 r5 r11 + PP b13 r5 r13 +
      PP b17 r5 r17 + PP b19 r5 r19)
      = ∑ i ∈ Finset.range 37,
        (yw i + ((if b5 r5 i && b7 r7 i then yw i else 0) +
          (if b5 r5 i && b11 r11 i then yw i else 0) +
          (if b5 r5 i && b13 r13 i then yw i else 0) +
          (if b5 r5 i && b17 r17 i then yw i else 0) +
          (if b5 r5 i && b19 r19 i then yw i else 0))) := by
    simp only [totY, PP, Finset.sum_add_distrib]
  have hR : S b5 r5 + S b7 r7 + S b11 r11 + S b13 r13 + S b17 r17 + S b19 r19
      = ∑ i ∈ Finset.range 37,
        ((if b5 r5 i then yw i else 0) + (if b7 r7 i then yw i else 0) +
          (if b11 r11 i then yw i else 0) + (if b13 r13 i then yw i else 0) +
          (if b17 r17 i then yw i else 0) + (if b19 r19 i then yw i else 0)) := by
    simp only [S, Finset.sum_add_distrib]
  rw [hL, hR]
  exact Finset.sum_le_sum key

/-- **No phase combination covers 37 consecutive slots.** -/
theorem no_cover {r5 r7 r11 r13 r17 r19 : ℕ} (h5 : r5 < 5) (h7 : r7 < 7)
    (h11 : r11 < 11) (h13 : r13 < 13) (h17 : r17 < 17) (h19 : r19 < 19)
    (hcov : ∀ i < 37, (b5 r5 i || b7 r7 i || b11 r11 i || b13 r13 i ||
      b17 r17 i || b19 r19 i) = true) : False := by
  have hb := cover_bound hcov
  have a5 := S5_le r5 h5
  have a7 := S7_le r7 h7
  have a11 := S11_le r11 h11
  have a13 := S13_le r13 h13
  have a17 := S17_le r17 h17
  have a19 := S19_le r19 h19
  have c7 := P7_ge r5 h5 r7 h7
  have c11 := P11_ge r5 h5 r11 h11
  have c13 := P13_ge r5 h5 r13 h13
  have c17 := P17_ge r5 h5 r17 h17
  have c19 := P19_ge r5 h5 r19 h19
  rw [tot_eq] at hb
  omega

/-! ## Back to the machine -/

set_option maxHeartbeats 1000000 in
/-- A slot that is not a machine-19 opening is blocked by one of the six
gears, in the certificate's `(phase, offset)` coordinates. -/
theorem blocked_of_not_exposed {p i : ℕ} (hp : 1 ≤ p)
    (h : ¬ Machine19.Exposed19 (p + i)) :
    (b5 (p % 5) i || b7 (p % 7) i || b11 (p % 11) i || b13 (p % 13) i ||
      b17 (p % 17) i || b19 (p % 19) i) = true := by
  have q5 : (p % 5 + i) % 5 = (p + i) % 5 := by omega
  have q7 : (p % 7 + i) % 7 = (p + i) % 7 := by omega
  have q11 : (p % 11 + i) % 11 = (p + i) % 11 := by omega
  have q13 : (p % 13 + i) % 13 = (p + i) % 13 := by omega
  have q17 : (p % 17 + i) % 17 = (p + i) % 17 := by omega
  have q19 : (p % 19 + i) % 19 = (p + i) % 19 := by omega
  simp only [b5, b7, b11, b13, b17, b19, q5, q7, q11, q13, q17, q19,
    Bool.or_eq_true, beq_iff_eq]
  by_contra hc
  apply h
  rw [Machine19.exposed19_iff (show 1 ≤ p + i by omega)]
  simp only [Machine19.expT, Bool.and_eq_true, bne_iff_ne, ne_eq, and_assoc]
  tauto

/-- **`F(19) <= 37`, by LP duality alone**: every window of 37 consecutive
slots contains a machine-19 opening. No period scan, no merge law - only the
37 certificate weights and `Machine19.exposed19_iff`. -/
theorem no_37_run {p : ℕ} (hp : 1 ≤ p) :
    ∃ i < 37, Machine19.Exposed19 (p + i) := by
  by_contra hc
  push Not at hc
  exact no_cover (r5 := p % 5) (r7 := p % 7) (r11 := p % 11) (r13 := p % 13)
    (r17 := p % 17) (r19 := p % 19) (by omega) (by omega) (by omega)
    (by omega) (by omega) (by omega)
    (fun i hi => blocked_of_not_exposed hp (hc i hi))

/-- **`F(19) <= 37` over machine 19's own gap word**, from the certificate. -/
theorem F19_le_37 (n : ℕ) : Machine19.g19 n ≤ 37 := by
  by_contra hc
  obtain ⟨i, hi, hE⟩ := no_37_run (p := Machine19.opSeq n + 1)
    (by have := Machine19.opSeq_pos n; omega)
  have hgap : Machine19.g19 n = Machine19.opSeq (n + 1) - Machine19.opSeq n := rfl
  have hlt := Machine19.opSeq_lt_succ n
  exact Machine19.opSeq_gap_empty n (Machine19.opSeq n + 1 + i)
    (by omega) (by omega) hE

/-- **(D) at `alpha = 3` at the 17->19 step, PROVED A SECOND WAY**: every gap
of machine 19 is at most `F(17) + 19 = 18 + 19`. The budget's `F(17) = 18` is
`Machine17.gap_le`; the `<= 37` itself is the covering certificate, and shares
nothing with `Ladder.D_at_17_19` (merge law + machine-17 qualifying scan). -/
theorem D_17_19_lp (n : ℕ) : Machine19.g19 n ≤ 18 + 19 := F19_le_37 n

end CoveringCert

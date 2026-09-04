# 2. The tooth rule and its corollaries

## In plain words

Every gear strikes columns in a perfectly regular pattern: two columns in each turn, always the
same two positions, one where it divides the lower number of the pair and one where it divides
the upper. The two struck positions are never next to each other, they sit a fixed distance
apart, and the column containing the gear's own prime is one of them. Between the two strikes
lie two stretches of untouched columns, a short one and a long one about twice as long, and the
column at the exact centre of the short stretch can never be struck, because the gear divides
the number halfway between the pair. When two gears are themselves twin primes they strike the
same low position, and the column where their product sits is struck by both.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; a gear is a prime `q >= 5`; gear `q` strikes column `k`
if `q | 6k-1` or `q | 6k+1`.  For a gear `q` write `u_q` for the integer `u` with
`6u = q - 1` (when `q = 1 mod 6`) or `6u = q + 1` (when `q = 5 mod 6`); so `u_q = round(q/6)`,
`1 <= u_q`, and `4 u_q < q`.  The residues `+-u_q mod q` are the **teeth** of `q`; the other
`q - 2` residues are its **openings**.

Classical translation: `k = +-u_q (mod q)` says exactly that `q` divides one member of the pair
`(6k-1, 6k+1)`; the two arcs of open residues are the two runs of consecutive `k` for which
neither member is a multiple of `q`.

## Statement

Fix a gear `q`.

**(a) Tooth rule.**  `q` strikes `k` iff `k = u_q` or `k = -u_q (mod q)`.  Precisely: if
`q = 5 (mod 6)` then `q | 6k-1 <=> k = u_q` and `q | 6k+1 <=> k = -u_q`; if `q = 1 (mod 6)` the
two are exchanged.  So `q` strikes exactly two residues per period and leaves `q - 2` open.

**(b) Arcs and spacing.**  The open residues form two arcs: the **long arc**
`u_q + 1, ..., q - u_q - 1` of `q - 2u_q - 1` residues, and the **short arc**
`-(u_q - 1), ..., u_q - 1` of `2u_q - 1` residues, centred at `0`.  The distance from one tooth
to the other is `2u_q` one way and `q - 2u_q` the other; consecutive struck columns of `q` are
spaced alternately `q - 2u_q` (from a column `= u_q`) and `2u_q` (from a column `= -u_q`), two
consecutive spacings sum to `q`, and no two struck columns are closer than `2u_q`.

**(c) The shield and self-blocking.**  `k = 0 (mod q)` is never struck; it is the centre of the
short arc.  The low tooth `u_q` is the column that contains `q` itself: `q = 6u_q -+ 1`.

**(d) Teeth are never adjacent.**  If `q` strikes `k` then `q` does not strike `k + 1`.

**(e) Twin gears.**  Let `p >= 5` and `p + 2` both be prime.  Then `u := (p+1)/6` is an
integer and `u_p = u_{p+2} = u`: twin gears share their tooth value.  Column `k` is struck by
both `p` and `p+2` iff `k = +-u` or `k = +-u(p+1) (mod p(p+2))`.  At `k = u(p+1)` the lower
member is `p(p+2)` itself, struck by both gears; at `k = u` the members are `p` and `p+2`.  The
split class "`p | 6k-1` and `(p+2) | 6k+1`" is exactly `k = u (mod p(p+2))`.  And if `q > 3`
and `q + g` are prime, `g > 0`, `6k - 1 = q` and `(q+g) | 6k+1`, then `g = 2`.

## Proof

**(a)**

1. `6` is invertible mod `q` since `q >= 5`.  So `q | 6k -+ 1` iff `6k = +-1` iff
   `k = +-6^{-1} (mod q)`.
2. `6u_q = q -+ 1 = -+1 (mod q)`, so `6^{-1} = -+u_q`, and `{6^{-1}, -6^{-1}} = {u_q, -u_q}`.
   Sign bookkeeping: for `q = 5 (mod 6)`, `6u_q = q+1 = 1`, so `k = u_q` gives `6k = 1`, i.e.
   `q | 6k-1`; and `k = -u_q` gives `q | 6k+1`.  For `q = 1 (mod 6)`, `6u_q = -1` and the roles
   swap.
3. `u_q` and `-u_q` are distinct residues because `0 < 2u_q < q`.  So exactly two residues are
   struck.

**(b)**

4. `1 <= u_q`: `6u_q = q -+ 1 >= 4`.  `4u_q < q`: `6u_q <= q + 1` gives `4u_q <= (2q+2)/3 < q`
   for `q > 2`.  Hence `0 < u_q < q - u_q < q`.
5. The two teeth `u_q < q - u_q` cut the residues `0..q-1` into the run
   `u_q + 1, ..., q - u_q - 1` (`q - 2u_q - 1` residues) and the run
   `q - u_q + 1, ..., q - 1, 0, 1, ..., u_q - 1` (`2u_q - 1` residues, containing `0` at its
   centre).  Since `4u_q < q`, `2u_q - 1 < q - 2u_q - 1`: short and long.
6. Distances between the teeth: `(q - u_q) - u_q = q - 2u_q` and `u_q + q - (q - u_q) = 2u_q`,
   summing to `q`.  If `x = u_q (mod q)` is struck, the columns `x+1, ..., x + (q - 2u_q) - 1`
   have residues `u_q + 1, ..., q - u_q - 1`, none a tooth, and `x + (q - 2u_q) = q - u_q` is a
   tooth: the next struck column is exactly `q - 2u_q` on.  From `x = -u_q` the same reasoning
   gives the next struck column `2u_q` on, at residue `u_q`.  Spacings therefore alternate and
   any two distinct struck columns differ by a positive number congruent to `0`, `2u_q` or
   `q - 2u_q`, hence by at least `2u_q`.

**(c)**

7. If `k = 0 (mod q)` then `q | 6k`, so `q | 6k +- 1` would give `q | 1`.  Its position in the
   short arc `-(u_q - 1), ..., u_q - 1` is the centre.
8. `q = 6u_q -+ 1` says `q` is a member of column `u_q`; `q` divides itself, so `q` strikes
   `u_q`, in agreement with (a).

**(d)**

9. Suppose `k = +-u_q` and `k + 1 = +-u_q (mod q)`.  Subtracting, `1 = 0`, `1 = 2u_q` or
   `1 = -2u_q (mod q)`.  The first is impossible.  If `1 = 2u_q`, multiply by 3:
   `3 = 6u_q = -+1`, so `q | 2` or `q | 4`.  If `1 = -2u_q`, then `3 = -6u_q = +-1`, the same.
   Both are impossible for `q >= 5`.

**(e)**

10. `p >= 5` prime is odd and not a multiple of 3; `p + 2` prime and `> 3` is not a multiple of
    3, so `p = 2 (mod 3)`, hence `p = 5 (mod 6)` and `6 | p + 1`.  With `u = (p+1)/6`:
    `6u = p + 1` is the defining equation of `u_p` (case `p = 5 mod 6`), and
    `6u = (p+2) - 1` is the defining equation of `u_{p+2}` (case `p + 2 = 1 mod 6`).
11. Both gears strike `k` iff `k = +-u (mod p)` and `k = +-u (mod p+2)`.  Since `p` and `p+2`
    are coprime, the Chinese remainder theorem gives four classes mod `p(p+2)`:
    `(u, u) -> u`, `(-u, -u) -> -u`, `(u, -u) -> u(p+1)` (check: `p + 1 = 1 (mod p)` and
    `p + 1 = -1 (mod p+2)`), `(-u, u) -> -u(p+1)`.
12. `6 u(p+1) - 1 = (p+1)^2 - 1 = p(p+2)`, divisible by both gears.  At `k = u`,
    `6u - 1 = p` and `6u + 1 = p + 2`.
13. By step 2, `p | 6k-1` iff `k = u (mod p)` (`p = 5 mod 6`), and `(p+2) | 6k+1` iff
    `k = u (mod p+2)` (`p + 2 = 1 mod 6`, where `k = u_{p+2}` is the `6k+1` tooth).  CRT gives
    `k = u (mod p(p+2))`.
14. If `6k - 1 = q` then `6k + 1 = q + 2`, so `(q+g) | q + 2` forces `q + g <= q + 2`, i.e.
    `g <= 2`; both `q` and `q + g` are odd primes, so `g` is even, so `g = 2`.

## Status

Kernel: `TwoTeeth.kill_spacing`, `TwoTeeth.kill_spacing_min`, `TwoTeeth.kill_period`,
`TwoTeeth.next_kill_of_lo`, `TwoTeeth.next_kill_of_hi`, `TwoTeeth.teeth_letters`,
`TwoTeeth.gear_side`, `TwoTeeth.kill_spacing_gear`, `TwoTeeth.kill_spacing_min_gear` (arcs and
spacing, (b)); `AnchorChain.neighbour_of_hit` (d); `Polignac.twin_product_slot`,
`Polignac.twin_mirror_slot`, `Polignac.twin_split_class_iff`, `Polignac.twin_pin_le`,
`Polignac.twin_pin_self_block`, `Polignac.own_slot_pin_gap_two`, `Corridor.product_slotOf`,
`Corridor.product_slotOf_sq`, `Corridor.twin_product_pin` (e); `Layer.slot_cap` (a gear never
strikes both members of one column).  The tooth rule (a) is not a named theorem: the kernel
takes it as the definition of a strike (`TwoTeeth.Kill q u x := x % q = u or x % q = q - u`,
`AnchorChain.OnTeeth`, the per-machine tests such as `Machine17.expT` and the generated
`CaseCert37.gb5 .. gb37`), and `TwoTeeth.teeth_letters` carries `6u mod q in {1, q-1}`.
Steps 1-3 and 7-8 are written here.

Verified computationally: every gear `5..199` and to `q = 100000`
(`research/check_two_teeth.py`); twin pairs to 3000 (60 pairs to 2000, 81 to 3000).

## Relationship to the conjecture

Bookkeeping (exact machinery): a complete description of one gear.  It supplies the letters
`a`, `b` and the never-adjacent fact that every later file uses; it establishes nothing about
how the openings of many gears space out; it depends on nothing measured.

## Where it is used

Every file below: the letters `a = 2u_q'`, `b = q' - 2u_q'` of file 05 are the two spacings of
(b); (d) is what makes the `x + 1` restart of the nested formula (file 09) legitimate; (e) is the
one place two gears strike a column deterministically inside every window.

## Source

`docs/proof-search/alignment-rules.md` 1.1-1.3; `docs/umbrellas-and-shields.md`;
`docs/novel/two-teeth-kill-spacing.md`; `docs/novel/tooth-sharing-pinning.md`.

# 18. The 31 -> 37 rung by case-split LP duality: `F(37) <= 95 = F(31) + 37`

## In plain words

For the machine with gears up to 37 the kernel proves that among any 95 consecutive columns
there is an opening, which is the budget inequality at that step. The proof fixes where the
three smallest gears are in their cycles, 385 possibilities, and for each one checks an exact
arithmetic certificate showing that the remaining seven gears cannot cover the columns those
three leave open. The certificates were found by a linear-programming search, but the kernel
only checks the arithmetic; no census of the machine is used.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; machine 37 is `{5, 7, 11, 13, 17, 19, 23, 29, 31, 37}`;
column `k` is an opening iff no gear strikes it, gear `q` striking `k` iff `k = +-u_q (mod q)`
with teeth `(1,4), (6,1), (2,9), (11,2), (3,14), (16,3), (4,19), (24,5), (26,5), (6,31)` for the
ten gears (file 02).  `F(37)` is the largest gap between consecutive openings; `F(31) = 58`.
A **stretch** of width `W` at column `p >= 1` (any run of `W` consecutive columns; the round notes call it a window, which in this series means only the certified range) is `{p, p+1, ..., p + W - 1}`; it is **fully
blocked** if it contains no opening.  In **(phase, offset) coordinates** gear `q` blocks offset
`i` of the stretch at `p` iff `(p mod q + i) mod q` is a tooth of `q`; so whether a stretch is
fully blocked depends only on the **phase vector** `(p mod 5, ..., p mod 37)`.

Classical translation: `F(37) <= 95` says that among any 95 consecutive columns there is a pair
`(6k-1, 6k+1)` with no prime factor `<= 37`; the budget inequality at this rung,
`F(37) <= F(31) + 37`, is exactly this width.

## Statement

**Theorem.**  Every stretch of 95 consecutive columns (from column 1 on) contains an opening of
machine 37.  Hence every gap of machine 37 is at most `95 = 58 + 37`: the budget inequality
holds at the step 31 -> 37.

The inputs are the primes up to 37 and, per case, 1049 integers; no census of any machine and no
period scan enter.

## Proof

**A. The case split (why 385 cases are exhaustive).**

1. Suppose the stretch at `p` is fully blocked.  Then for every offset `i < 95` some gear
   blocks `i` in (phase, offset) coordinates (`CaseCert37.blocked`: a non-opening is struck by
   one of the ten gears, unfolded gear by gear from machine 37's opening predicate).
2. The **held gears** are 5, 7, 11.  Their phases `(p mod 5, p mod 7, p mod 11)` take exactly
   `5 * 7 * 11 = 385` values.  The root theorem `CaseCert37.no_run` splits on `p mod 5`
   (5 cases) and `p mod 7` (7 cases), each landing in one of 35 tier files `CaseCert37T0 ..
   T34`, each of which splits on `p mod 11` (11 cases) into the leaf theorems
   `CaseCert37.nocase0 .. nocase384`.  Every `p` falls in exactly one leaf.
3. In leaf `c` with held phases `(h_5, h_7, h_11)`, let `Pos_c` be the list of offsets
   `i < 95` blocked by none of the held gears at those phases (the kernel checks
   `plt_c`: every listed offset is `< 95`, and `pfree_c`: the held gears block none of them;
   `n_c := |Pos_c|`, e.g. 34 in case 0).  Since the stretch is fully blocked, every `t in Pos_c`
   is blocked by one of the seven **free gears** `13, 17, 19, 23, 29, 31, 37` at their phases
   `r_0, ..., r_6` (`r_a = p mod q_a`).  The leaf theorem `nocov_c` derives `False` from that
   covering hypothesis for every phase tuple `(r_0, ..., r_6)`.  So no `p` has a fully blocked
   stretch of width 95.

**B. What one leaf certificate is.**  Fix a case `c`; write `c_a(r, t) in {0, 1}` for "free
gear `a` at phase `r` blocks position `t in Pos_c`" and `D(t) := sum_a c_a(r_a, t)` for the
number of free gears blocking `t`.  The certificate consists of integers: weights `w_t >= 0`
(one per position), a vector `u` (the dual multipliers of the phase-consistency links), and
the numbers below computed from them.

4. **The lowest-blocker identity.**  For a blocked position `t` (i.e. `D(t) >= 1`), let
   `lo(t)` be the smallest index `a` with `c_a(r_a, t) = 1`.  Then

       1 + #{ (a, b) : a < b, a = lo(t), c_b(r_b, t) = 1 } = D(t),

   because the lowest blocker pairs with each of the other `D(t) - 1` blockers exactly once
   (`CaseSplit.lowest7`, checked over all `2^7` blocking patterns).  Also `D(t) >= 1`
   (`CaseSplit.degpos7`).
5. **Summing with weights.**  Multiply the identity by 1 and sum over `Pos_c`, then add
   `sum_t w_t D(t) >= sum_t w_t` (as `w_t >= 0`, `D(t) >= 1`):

       sum_t (w_t + 1) D(t)  >=  sum_t w_t + n_c + sum_{a<b} N_ab,

   where `N_ab := #{t : a = lo(t), c_b(r_b, t) = 1}`.  The left side equals
   `sum_a S_a(r_a)` with `S_a(r) := sum_t (w_t + 1) c_a(r, t)`, a function of gear `a`'s phase
   alone.  This is the **recursion row**:

       sum_a S_a(r_a) - sum_{a<b} N_ab(r_a, r_b, ...)  >=  sum_t w_t + n_c  =: rhs_c.     (R)

6. **Lower bounds on the pair terms that depend on two phases only.**  For `a = 0` (gear 13,
   the lowest free gear), `lo(t) = 0` iff `c_0(r_0, t) = 1`, so
   `N_0b = P_0b(r_0, r_b) := #{t : c_0(r_0,t) = c_b(r_b,t) = 1}` exactly.  For `a = 1`,
   `N_1b = P_1b(r_1, r_b) - #{t : c_1 = c_b = 1 = c_0(r_0, t)} >= P_1b(r_1, r_b) - M_1b(r_1, r_b)`
   where `M_1b(r_1, r_b) := max_{r_0} #{t : c_1(r_1,t) = c_b(r_b,t) = c_0(r_0,t) = 1}`
   (`CaseSplit.ind_low2`, `le_mxr`); the certificate uses this bound only on a listed set
   `E_1b` of phase pairs and `0` elsewhere (`N_ab >= 0` always: `CaseSplit.ind_nonneg`).  For
   `a >= 2` the certificate uses `N_ab >= 0`.  Call the resulting two-phase lower bounds
   `N'_ab(r_a, r_b)`; (R) holds with `N'` in place of `N`.
7. **Decoupling with the `u` vector.**  For each free gear `a` define
   `L_a(r) := sum over the six pairs containing a of u(offset_{ab} + r)` and
   `aS_a(r) := S_a(r) - L_a(r)`; for each pair define
   `aP_ab(r_a, r_b) := -N'_ab(r_a, r_b) + u(offset_{ab} + r_b) + u(offset_{ba} + r_a)`.  Every
   `u` entry occurs once with `+` (in some `aP`) and once with `-` (in the corresponding `L_a`),
   so for every phase tuple

       sum_a aS_a(r_a) + sum_{a<b} aP_ab(r_a, r_b) = sum_a S_a(r_a) - sum_{a<b} N'_ab(r_a, r_b)

   (the identity `hid`, by `ring`).  Combined with (R): the left side is `>= rhs_c` whenever a
   covering phase tuple exists.
8. **Block maxima.**  `MS_a := max_{r < q_a} aS_a(r)` and
   `MP_ab := max_{r_a < q_a, r_b < q_b} aP_ab(r_a, r_b)` are integers the kernel computes
   (`CaseSplit.mxr`, `mxr2`; the values `MSv_c`, `MPv_c` by `decide +kernel`), and every term
   of step 7's left side is at most the corresponding maximum (`CaseSplit.le_mxr`,
   `le_mxr2`).  So for a covering phase tuple, `sum_a MS_a + sum_{a<b} MP_ab >= rhs_c`.
9. **The certificate inequality.**  The kernel evaluates `sum_a MS_a + sum_{a<b} MP_ab` and
   `rhs_c` as integers and checks `cert_c : sum_a MS_a + sum_{a<b} MP_ab < rhs_c` (case 0:
   `32 < 34`, margin 2).  With step 8 this is a contradiction: no phase tuple of the free gears
   covers `Pos_c`.  The leaf proof `nocov_c` is exactly the chain 4-9 discharged by `linarith`
   from the named facts.

**C. From stretches to gaps.**

10. `CaseCert37.F_le`: if consecutive openings `o_n < o_{n+1}` of machine 37 had
    `o_{n+1} - o_n > 95`, the stretch of width 95 at `o_n + 1` would contain no opening
    (`Machine37.opSeq37_gap_empty`), contradicting `no_run`.  So every gap `g37 n <= 95`, and
    `CaseCert37.D_31_37_case` restates it as `g37 n <= 58 + 37`.

**D. Where the numbers come from, and what the kernel does not check.**  The weights `w`, the
vector `u`, the lists `E_ab` and the position lists were produced by the LP-duality thread as an
exact rational dual solution of the composed level-2 covering relaxation at width 95 with held
gears (5, 7, 11), scaled by a common denominator per case (`research/data/r29/
cert_31_37_h*.json`, generated into Lean by `research/gen_case_lean.py`, which re-derives every
number from the primes alone).  The kernel does not know or need that provenance: it checks the
inequalities of steps 3-9 on the integers as given.  That `k = 3` held gears is the smallest
number that certifies this rung is a proof on the LP side (exact in-polytope refutations at
`k = 1, 2`), not a kernel statement.

## Status

Kernel: `CaseCert37.gb5 .. gb37` (the gears as blocking predicates), `CaseCert37.blocked`,
`CaseCert37.no_run`, `CaseCert37.F_le`, `CaseCert37.D_31_37_case` (`proofs/CaseCert37.lean`);
`CaseCert37.nocase0 .. nocase384` (`proofs/CaseCert37T0.lean .. T34.lean`);
`CaseCert37.nocov0 .. nocov384`, `cert0 .. cert384`, `plt_c`, `pfree_c_5/7/11`, `wnn_c`,
`MSv_c_*`, `MPv_c_*`, `rhsv_c` (`proofs/CaseCert37C0.lean .. C384.lean`);
`CaseSplit.mxr`, `CaseSplit.le_mxr`, `CaseSplit.mxr2`, `CaseSplit.le_mxr2`,
`CaseSplit.lowest7`, `CaseSplit.degpos7`, `CaseSplit.ind_low2`, `CaseSplit.ind_nonneg`
(`proofs/CaseSplit.lean`); `Machine37.Exposed37`, `Machine37.opSeq37`, `Machine37.g37`,
`Machine37.opSeq37_gap_empty`.  The root is tiered (35 sub-roots of 11 cases) because the flat
385-import root exhausted the machine (Formalist R29.5, R30).  `CaseCert23` and `CaseCert31`
are the same construction at 19->23 (5 cases) and 29->31 (35 cases).

Verified computationally: the 385 certificates with margins `1/5 .. 3` (`manifest_31_37.json`,
`research/lp_rungs_r29.txt`); `F(37) = 88` exactly by scan, so the width 95 has slack 7.

## Prior art, and what is new

**Leverages.**  Ordinary LP duality (an exact rational Farkas certificate per case), and the
Bonferroni family the record names for the pair terms: Kounias (1968), Hunter (1976), Worsley
(1982) and Prekopa, whose degree-2 bound is the shape of the recursion row -- the record notes
that the row is numerically almost entirely a Kounias row at the smallest free gear.  The case
split over held phases is one level of the Sherali-Adams / Lovasz-Schrijver / Lasserre
consistency hierarchies, cited as machinery in `docs/novel/consistency-over-degree.md`.

**New.**  The rung itself: `F(37) <= 95` in the kernel, hypothesis-free, with no census and no
period scan.  And the composition -- hold three gears' phases (385 cases) and discharge each by
an exact integer dual certificate, with the `u`-vector decoupling that turns a seven-phase
covering into per-gear and per-pair block maxima -- is what makes a rung checkable by arithmetic
alone, the provenance of the numbers being irrelevant to the check.

**Not new.**  The dual certificate is ordinary LP duality and the pair bound is the classical
degree-2 Bonferroni inequality in the machine's coordinates; the consistency machinery is
Sherali-Adams.  What the register's own check found unpublished is the application: no
measurement of consistency against degree on a Jacobsthal-type covering problem, and no use of
either to certify a step (`docs/novel/consistency-over-degree.md`, PARTIAL OVERLAP, 2026-08-25).
Prior art for the case-split strengthening specifically is not checked --
`docs/novel/restricted-covering-certificates.md` section 7 records it as not yet run.

## Relationship to the conjecture

A rung of the budget inequality in the kernel, hypothesis-free: progress on the certified
ladder (rung ten), not on the general statement.  The method costs a primorial in the number of
held gears and supplies no induction step.  Nothing measured enters the proof; the
certificates' provenance (an LP) is irrelevant to their check.

## Where it is used

The tenth rung of the budget inequality in the kernel, and the only rung past 29->31 that is
kernel-checked; the template for any further rung the LP thread emits.

## Source

LP-duality thread rounds 26-29 (`docs/proof-search/lp-duality.md`); Formalist rounds 27-30
(`proofs/CaseSplit.lean`, `proofs/CaseCert37*.lean`, `research/gen_case_lean.py`).

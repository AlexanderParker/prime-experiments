# 9. The record law (phase reduction) and the nested next-opening formula

## In plain words

Instead of building the bigger machine, keep the old machine's cycle and, for each possible
placement of the new gear, cross out the old openings that placement would strike; the longest
gap over all placements is the bigger machine's record. This is what makes records hundreds of
times cheaper to compute. The kernel confirms it fully for one machine, the gears up to 17: the
seventeen placements give a largest gap of 18, and the machine's true record is 18. A companion
formula finds the next opening of the bigger machine by walking the old openings and skipping
the struck ones; how many it may have to skip in a row is a measured number, not a proved one.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; `M` is a finite set of gears with period `P` and
`P`-periodic opening set `O_M subset N`; `g` is a gear not in `M`, `u = u_g` its tooth value
(`6u = g -+ 1`), `d := 2u`, teeth `{u, -u} mod g`; `M' := M + g` has period `Pg`.  `F(M')` is
the largest gap between consecutive openings of `M'`.  A **phase** is a residue `r in Z_g`, and
the **two-class set** of phase `r` is `{r, r + d} subset Z_g`.

Classical translation: `O_M` is the set of `k` for which no prime of `M` divides `6k +- 1`;
the record law computes the largest run of `k` with a `g`-divisible member or an `M`-struck
member, using only the residues of the `M`-openings modulo `g` on one period of `M`.

## Statement

**Theorem 1 (phase reduction).**  For each phase `r` define the **phase-`r` sequence**

    S_r := { x in N : x mod P in O_M  and  x mod g not in {r, r + d} }.

(i)  `S_r` is a translate of the openings of `M'`: with `t_r := (-u - r) P^{-1} (mod g)`,

    x in S_r   iff   x + t_r P  is an opening of M',

and `r -> t_r` is a bijection of `Z_g`; the phase `r = -u` is `M'` itself (`t = 0`).

(ii)  `F(M') = max over r in Z_g of max{ z - y : y < z consecutive in S_r, 0 <= y < P }`.
That is, the record of `M'` is a maximum over `g` phases of largest gaps computed on ONE period
of the lower machine, from the lower openings and their residues mod `g` alone.

(iii)  Every gap of `S_r` is either a gap of `M` (between two openings whose residues avoid the
two-class set) or, for a maximal run `x_1 < ... < x_k` (`k >= 1`) of consecutive openings of
`M` whose residues all lie in `{r, r + d}`, with neighbours `x_0 < x_1` and `x_{k+1} > x_k`,
the sum `(x_1 - x_0) + (x_k - x_1) + (x_{k+1} - x_k)` = gap before + run span + gap after.

**Theorem 2 (the record law at machine 17, kernel-checked at both ends).**  For
`M = {5, 7, 11, 13}` (`P = 5005`, 1485 openings per period), `g = 17`, `u = 3`, `d = 6`: the
largest merged gap of the phase-`r` sequence with left endpoint in `[0, 5005)` is
`16, 16, 18, 18, 18, 16, 18, 18, 16, 15, 16, 18, 18, 16, 18, 18, 18` for `r = 0..16`, maximum
`18`; and `F({5, 7, 11, 13, 17}) = 18` exactly, attained at the openings `117` and `135`.

**Theorem 3 (the nested next-opening formula).**  Let `M` be any set of columns such that
above every `x` there is an element of `M` ("lower openings"), and `H` any predicate ("hit")
such that above every `x` there is an element of `M` not in `H`.  Let `next_M(x)` be the least
element of `M` above `x`, and `next_G(x)` the least element of `M` above `x` not in `H`.  If
the first `k` lower openings above `x`, namely `next_M(x), next_M^2(x), ..., next_M^k(x)`, are
all hits and `next_M^{k+1}(x)` is not, then

    next_G(x) = next_M^{k+1}(x).

`k = 0` is the hit law's "no hit, no move" (file 05 (B)); `k = 1` is the two-term form.  The
number of terms is bounded by the longest run of consecutive hits, the **chain depth** `D_g` --
a measured per-machine quantity, not a theorem (see Status).

## Proof

**Theorem 1.**

1. (i)  For `x in N` and `t in Z_g`: `x + tP` is on a tooth of `g` iff `x + tP = +-u (mod g)`
   iff `x = -u - tP` or `x = u - tP = (-u - tP) + 2u`, i.e. iff `x mod g in {r, r + d}` with
   `r = -u - tP`.  Solving for `t`: `t = (-u - r) P^{-1} =: t_r`, a bijection `r -> t_r`
   since `P` is a unit mod `g`.  And `x + t_r P` is an opening of `M'` iff it is an opening of
   `M` (iff `x mod P in O_M`, as `P | t_r P`) and not on a tooth of `g` (iff
   `x mod g not in {r, r + d}`).  At `r = -u`, `t_r = 0`.
2. (ii)  Both `S_r` and the opening set of `M'` are `Pg`-periodic, and by (i) the gaps of
   `S_r` are exactly the gaps of `M'` shifted by `t_r P`.  Every gap of `M'` has a left
   endpoint `X = x + tP` with `0 <= x < P` and some `t`; choose `r` with `t_r = t (mod g)`
   (possible by the bijection); then `x` is the left endpoint of a gap of `S_r` of the same
   length, and `0 <= x < P`.  Conversely every gap of `S_r` is a gap of `M'`.  So the set of
   gap lengths of `M'` equals the union over `r` of the lengths of gaps of `S_r` whose left
   endpoint lies in `[0, P)`, and taking maxima gives (ii).
3. (iii)  Let `y < z` be consecutive in `S_r`.  Both are openings of `M` with residues outside
   `{r, r+d}`; the openings of `M` strictly between them all have residues inside it (else they
   would be in `S_r`).  If there are none, `z - y` is a gap of `M`.  Otherwise they form a
   maximal run `x_1 < ... < x_k` of consecutive `M`-openings with residues in the two-class
   set, with `x_0 = y` and `x_{k+1} = z`, and `z - y` telescopes as stated.

**Theorem 2.**  This is Theorem 1 instantiated and computed.

4. The lower opening test is `lowOpen(k) := k mod 5 not in {1, 4}, k mod 7 not in {6, 1},
   k mod 11 not in {2, 9}, k mod 13 not in {11, 2}` (the tooth rule, file 02, at
   `u = 1, 1, 2, 2`), and the phase-`r` survivor test is
   `surv r y := lowOpen(y mod 5005) and y mod 17 not in {r, r + 6}`.
5. `5005 = 7 (mod 17)` and `7^{-1} = 5 (mod 17)`, so `t_r = (-3 - r) * 5 = (14 - r) * 5 (mod 17)`;
   the kernel's `tOf r = ((31 - r) * 5) mod 17` is the same number (`31 = 14 (mod 17)`).  The
   kernel proves `surv r y = openT17(y + tOf r * 5005)` for every `r < 17` and every `y`
   (`AnchorRecord17.surv_shift`), where `openT17` is machine 17's own opening test written in the
   same shape, and `openT17 k = true iff Machine17.Exposed17 k` for `k >= 1`
   (`AnchorRecord17.openT17_iff`).  That is (i) at this machine.
6. For each `r < 17` the kernel walks `surv r` over `[0, 5005 + 64)` (one lower period plus a
   look-ahead longer than any gap) and records the largest gap whose left endpoint is below
   `5005`: the seventeen values listed, each by `decide +kernel`
   (`AnchorRecord17.mg0 .. mg16`), so `mg r <= 18` for all `r` and `mg 2 = 18`
   (`AnchorRecord17.record_max`).
7. Independently, `Machine17.gap_le` (a period scan of machine 17) gives every gap of machine
   17 `<= 18`, and `117`, `135` are openings with none between
   (`AnchorRecord17.gap18_realized`), so `F(17) = 18` (`AnchorRecord17.F17_eq_18`).

Honest scope, from the kernel file's own header: the kernel does NOT prove that the walk `mg`
correctly computes the largest gap of the phase-`r` sequence; it computes the seventeen numbers
and, separately, proves `F(17) = 18`.  The identity "max over phases = `F(17)`" is therefore
verified at both ends (`18` and `18`) rather than derived from one to the other.  Theorem 1
above is the written derivation; `research/anchor235/r29_record17_gate.py` gates the same
identity outside the kernel.

**Theorem 3.**

8. `next_G(x)` is an element of `M` above `x` and not in `H`.  Let `m_i := next_M^i(x)`
   (`m_1 < m_2 < ...` are the successive elements of `M` above `x`; each `m_{i+1}` is the least
   element of `M` above `m_i`).
9. `next_G(x) >= m_{k+1}`: by induction on `i <= k+1`, `next_G(x) >= m_i`.  For `i = 1` it is
   an element of `M` above `x`.  If `next_G(x) >= m_i` with `i <= k`, then `next_G(x) != m_i`
   (because `m_i` is a hit and `next_G(x)` is not), so `next_G(x) > m_i`, hence
   `next_G(x) >= m_{i+1}` by minimality of `m_{i+1}`.
10. `next_G(x) <= m_{k+1}`: `m_{k+1}` is an element of `M` above `x` and not a hit, so it
    competes in the minimum defining `next_G(x)`.

The term cap: the nested formula uses `k + 1` applications of `next_M`, where `k` is the length
of the run of consecutive hits starting at `next_M(x)`; `k` is at most the longest run of
consecutive lower openings all on the teeth of `g` in one copy -- the chain depth `D_g`.  By the
chain law (file 05 (C)) such a run alternates freely between the two classes `r` and `r + d`,
so nothing in the residue arithmetic bounds its length; `D_g` is a fact about the SIZES of the
lower gaps (the admissible gaps are the elements of `{d, g - d, g, g + d, ...}` at most
`F(M) + 1`), measured per machine: `D_17 = D_19 = 2`, `D_23 = 3`, `D_29 = 2`, `D_31 = 4`,
`D_37 = 4`, `D_41 = 3`.  `D_g` bounded is OPEN; by file 10 it equals `A_kill = L + 1`.

## Status

Kernel: Theorem 2 -- `AnchorRecord17.lowOpen`, `AnchorRecord17.surv`, `AnchorRecord17.walk`,
`AnchorRecord17.mg`, `AnchorRecord17.mg0 .. mg16`, `AnchorRecord17.record_max`
(`proofs/AnchorRecord17Core.lean`); `AnchorRecord17.openT17`, `AnchorRecord17.tOf`,
`AnchorRecord17.shift_res`, `AnchorRecord17.surv_shift`, `AnchorRecord17.openT17_iff`,
`AnchorRecord17.phase_is_machine`, `AnchorRecord17.gap18_realized`, `AnchorRecord17.F17_eq_18`
(`proofs/AnchorRecord17.lean`), with `Machine17.gap_le`.  Theorem 1 in the abstract:
`AnchorChain.copy_phase`, `AnchorChain.phase_bijective` (step 1); steps 2-3 written.  Theorem 3:
`AnchorChain.nextM`, `AnchorChain.nextG`, `AnchorChain.nextM_le_nextG`, `AnchorChain.hop_zero`,
`AnchorChain.lt_iterate`, `AnchorChain.M_iterate`, `AnchorChain.iterate_le_of_hits`,
`AnchorChain.hop_iter`, `AnchorChain.hop_one` (all abstract in the machine).

Verified computationally: the phase reduction reproduces the records at `{5..7} .. {5..29}`
from full lower periods, at 31 and 37, and at 41 on a partial sweep (`F = 42` for `{5..29}`
from 7,952,175 lower openings instead of a `6.5e9`-column period); the nested formula equals the
walk at every column on the full periods `{5, 7} .. {5..19}` (`research/anchor235/`).

## Prior art, and what is new

**Leverages.**  Standard (CRT) and file 05 (A).  One-class prior art, read first-hand in the
round-30 check of `docs/novel/anchor-235-layer-laws.md`: Holt & Rudd, arXiv:1408.6002, Lemma 2.1
(the concatenated copies, whose elementwise product is the phase walk) and Theorem 2.3 (each
closure exactly once, by CRT, which is the phase bijection here).

**New.**  Phase reduction read as a computation of the record: the maximum over `g` phases of
largest gaps taken on ONE period of the lower machine, with the two-class set `{r, r + d}` where
the one-class picture has a single class.  That is what makes records cheap enough to reach
machines no scan reaches, and machine 17 is certified in the kernel at both ends.  The check
found none of four items in print -- the two-class chain law, neighbour-of-hit as a theorem for
every gear, the record as a maximum over phases, and `D_g = A_kill` -- and the nested
next-opening formula with its iterate is the walk's closed form.

**Not new.**  The copies-and-phases picture is Holt-Rudd's one-class recursion in gear language;
the register's recorded verdict for this entry is PARTIAL OVERLAP on exactly that.  The honest
scope is in the file itself: the kernel computes the seventeen phase maxima and separately
proves `F(17) = 18`, rather than deriving one from the other, and the chain depth `D_g` that
caps the nested formula is measured, not proved.

## Relationship to the conjecture

Computational machinery: it makes records cheap to compute (this is how `F(59) = 161` was
obtained) and certifies one record in the kernel.  No bearing on the open size statement; the
chain depth that caps the nested formula is measured, not proved.

## Where it is used

The scan-free computation of records (`58, 88, 91` at 31/37/41 on machine 29's period, `161`
at 59 on machine 23's period); the `L = 1` row of the phase table is `F_2` of the lower
machine; the nested formula is the layered walk's closed form (`anchor-235.md` 9f).

## Source

`docs/proof-search/anchor-235.md` sections 9d-9f (the phase form and the nested form);
`research/anchor235/chain_depth.py`; Formalist round 29 (`AnchorRecord17`);
`docs/proof-search/alignment-rules.md` 3.1.

# The anchor-2,3,5 layer laws: chain law, neighbour-of-hit, the phase-reduction record law, and D_g = A_kill

Register entry written by the harvester, round 30, on the manager's instruction, for
results established by the manager's anchor-235 line (`docs/proof-search/anchor-235.md`
sections 9d-9g), kernel-checked by Formalist (round 29, `proofs/AnchorChain.lean`,
`proofs/AnchorRecord17.lean`, `AnchorRecord17Core.lean`) and measured by Mechanic
(round 29, C48-C50).  Status is given per item in section 3; the prior-art verdict is in
section 6.

---

## 1. WHAT IT IS

**Plain language.**  Build the machine one gear at a time.  The new gear `g` sits on top
of the lower machine `M = {5..g-}` and can only delete lower openings that land on its
two teeth.  Four facts decide everything the new gear does, and they are about
RESIDUES MOD `g` of the lower openings, not about the slots: (i) which pairs of
consecutive lower openings can both be deleted (the chain law); (ii) that the neighbour
of a deleted slot is never deleted (neighbour-of-hit); (iii) that the `g` copies of the
lower period inside the new period run through every deletion phase exactly once, so the
new machine's record gap is a maximum over `g` phases of "gap before + run span + gap
after" on ONE lower period (the phase-reduction record law); and (iv) that the longest run
of consecutive lower openings a single phase can delete is exactly the project's kill
arity `A_kill(M -> g)`.

**Precise form.**  Gear `g >= 5`, tooth `u = 6^{-1} mod g`, `d = 2u mod g`
(`d_g` in the anchor doc; `s` / `2u'` in the twin-route documents).  A slot `x` is a HIT of
`g` iff `x = +-u mod g`; the two-class set at phase `r` is `{r, r + d} mod g`, and the
teeth ARE the two-class set at phase `-u` (`AnchorChain.teeth_eq_phase`).

> **(L1) CHAIN LAW** (`AnchorChain.chain_law`, every `g`).  Two slots lie in a common
> two-class set `{r, r + d}` iff their difference is `0`, `d` or `-d` mod `g`.  Hence two
> consecutive lower openings `x < y` are both deleted by `g` (at some phase) iff
> `y - x = 0` or `+-d_g mod g`; and the gap sizes that can carry a second hit are the
> classes `{d_g, g - d_g, g, g + d_g, 2g - d_g, ...}` cut at `F_M + 1`.
> **T3 half** (`no_two_up`, `no_two_down`): three slots in one two-class set cannot be
> reached by two `+d` steps or two `-d` steps unless `2d = 0` - a run in a two-class set
> never steps the same way twice.

> **(L2) NEIGHBOUR-OF-HIT** (`AnchorChain.neighbour_of_hit`): for every `g >= 5`, if `x`
> is a hit then `x + 1` is not, because `d = 2u = 3^{-1}` and `3^{-1} = +-1 mod g`
> would force `g | 2` or `g | 4`.  This is what licenses the nested formula's `x + 1`
> restart after a hit.

> **(L3) PHASE-REDUCTION RECORD LAW** (`AnchorChain.copy_phase` +
> `phase_bijective`; `anchor-235.md` 9f; `chain_depth.py`).  In the copy of the lower
> period shifted by `j P_M`, the hits of `g` are exactly the lower slots deleted at phase
> `-u - j P_M`; when `P_M` is a unit mod `g` the map `j -> -u - j P_M` is a bijection of
> `Z_g`, so the `g` copies realise every deletion phase exactly once.  Therefore, on ONE
> lower period with the lower opening residues mod `g`,
>
>     D_g          = longest run of consecutive lower openings with residues in one set
>                    {r, r + d_g}
>     F_bc(M+g) + 1 = max over such runs (all phases r) of
>                    (gap before) + (run span) + (gap after)
>
> where `F_bc` is the anchor doc's BLOCKED-COUNT record (`F_bc = 17` at machine 17) and
> `F_bc + 1` is the merged gap = the corpus max-gap record `F(M+g)` (`F(17) = 18`; the
> corpus ladder 5, 7, 11, 18, 25, 34, 43 is the anchor ladder 4, 6, 10, 17, 24, 33, 42
> plus one).  At machine 17 this is in the kernel at both ends:
> `AnchorRecord17.record_max` (`mg r <= 18` for all 17 phases, `mg 2 = 18`),
> `surv_shift` / `phase_is_machine` (phase `r` IS machine 17 shifted by `tOf r` lower
> periods) and `F17_eq_18` (the corpus attainment, new at 17).

> **(L4) `D_g = A_kill(M -> g)`** (Mechanic C49, 7 for 7).  The chain depth of 9f and the
> twin route's kill arity are one object: both count co-deletable runs of consecutive
> `M`-openings, and C10's word legality ("prefix-sum range `<= 1`") is exactly "all in one
> two-class set".  With Constructor's R89, `D_g = A_kill = L(M) + 1`, `L` the longest
> realised legal word.

**The nested formula** (9f), as a theorem abstract in the machine
(`AnchorChain.hop_zero` / `hop_iter` / `hop_one`): with `M` the lower opening predicate
and `H` the hit predicate, the enlarged machine's next opening after `x` is `nextM`
iterated once past the run of hits, `nextG x = nextM^[k+1] x` when the first `k` lower
openings after `x` are hits and the `(k+1)`-st is not.  Its `D_g`-term cap is the `k` of
this statement.

---

## 2. WHY IT MIGHT BE NOVEL

* The four laws are elementary residue arithmetic, and the copies-of-the-lower-period
  picture is Holt-Rudd's (arXiv:1408.6002, Lemma 2.1 / Theorem 2.3) for the ONE-class
  sieve.  What is not in that literature: the two-class chain law (a common two-class set
  iff difference in `{0, +-d}`), which has no one-class analogue (one class gives
  "difference `= 0 mod g`" only), and the T3 alternation half.
* (L3) is a genuine REDUCTION: the new machine's record is computed on the lower period's
  opening list with residues mod `g` only - `F = 42` for `{5..29}` from 7,952,175 lower
  openings instead of a 6.5e9-slot period (819x smaller), and Mechanic's round-29 record
  law at 31/37/41 (58, 88, 91) walked a 1.24e12-slot period with no array beyond machine
  29.  The Jacobsthal-computation literature (Hagedorn; Ziller-Morack; Ziller) assigns
  residues per prime and searches coverings of a WINDOW; none walks the lower sieve's
  gap sequence by phase.
* (L4) identifies two constructs built four rounds apart in different languages.

---

## 3. PROOF / STATUS

| item | status | pointer |
|---|---|---|
| (L1) chain law, both directions, every `g` | **KERNEL-CHECKED**, `[propext, Quot.sound]` / with `Classical.choice` (manager's audit, round 29) | `proofs/AnchorChain.lean`: `teeth_eq_phase`, `chain_law`, `no_two_up`, `no_two_down` |
| (L1) the admissible-gap list and its realisation | **SCRIPT-VERIFIED**, full periods `{5..23}`, "holds without exception" | `research/anchor235/layer_law.py`, `results/layer_law.txt` (anchor-235.md 9d table) |
| (L2) neighbour-of-hit | **KERNEL-CHECKED** for every `g >= 5` from `6u = 1` alone | `AnchorChain.neighbour_of_hit` |
| (L3) copies realise every phase once | **KERNEL-CHECKED**, machine-free | `AnchorChain.copy_phase`, `phase_bijective` |
| (L3) record law at machine 17: phase table 16 16 18 18 18 16 18 18 16 15 16 18 18 16 18 18 18, max 18; phase `r` = machine 17 shifted | **KERNEL-CHECKED** at both ends (phase table by `decide +kernel` per phase; corpus record by `Machine17.gap_le` + witness `(117, 135)`); NOT derived one from the other (no correctness proof of the walk against `Machine17.nextOp`) | `AnchorRecord17Core.mg0..mg16`, `AnchorRecord17.record_max`, `surv_shift`, `phase_is_machine`, `gap18_realized`, `F17_eq_18`; gate `research/anchor235/r29_record17_gate.py` |
| (L3) record law `F_bc = 4, 6, 10, 17, 24, 33, 42` (blocked count; corpus max-gap `5, 7, 11, 18, 25, 34, 43`) at `{5..7}..`{5..29}`; corpus `58, 88, 91` at 31/37/41 | **SCRIPT-VERIFIED, exact** (corpus values reproduced; the 31/37 rows over FULL lower periods, the 41 row a deliberate 36.9% sweep whose two answers are exact because the sample gives the lower half and C14's COV-SAT `F(41) = 91` the upper) | `research/anchor235/chain_depth.py`; Mechanic `research/chain_depth_r29.py gate`, `gate_mechanic_r29.py` C |
| (L3) the nested formula's recursion | **KERNEL-CHECKED**, abstract in the machine | `AnchorChain.hop_zero`, `hop_iter`, `hop_one` |
| (L3) the nested formula equals the walk at every slot | **SCRIPT-VERIFIED** on full periods `{5,7}` .. `{5..19}` | `research/anchor235/nested_form.py` |
| (L4) `D_g = A_kill(M -> g)` | **SCRIPT-VERIFIED**, 7 for 7 (`D_17 = D_19 = 2`, `D_23 = 3`, `D_29 = 2`, `D_31 = 4`, `D_37 = 4`, `D_41 = 3`), identity by argument, not proved in the kernel | Mechanic C49, `research/chain_depth_r29.py`; harvester `research/hr_twoclass_r30.py` E reproduces `A_kill = 2, 2, 2, 3, 2` at m11..m23 from the `q'`-copy concatenation |
| `D_g` bounded | **OPEN** - Formalist's honest boundary: "the chain DEPTH `D_g` is not an algebraic consequence"; a run alternates freely, and `D_g` is a fact about lower gap SIZES | `AnchorChain.lean` header; even-j-mechanism.md |

---

## 4. IMPLICATIONS

* Inside the project: the whole per-layer behaviour of the machine is decided by
  residues of the lower gaps mod `g` (L1), so a layer is a finite formula of depth `D_g`
  (9f), and the record of the next machine is a one-lower-period scan (L3) - the vehicle
  behind `F(M+g)` at 31, 37, 41 and behind the rung-eleven measurement.  (L4) turns the
  anchor line's `D_g` and the twin route's `A_kill` / `L(M)` into one target: whatever
  bounds one bounds the other.
* Outside: the phase reduction is a two-class instance of "the next stage of the sieve is
  `g` phases of the current stage" (Holt-Rudd's copies), used for the MAXIMAL gap rather
  than for populations - the one-class analogue (compute `h(k+1)` by walking `G(p_k#)`
  under `p_{k+1}` phases) is implicit in Holt-Rudd's recursion but, as far as searched,
  never used as a computation of Jacobsthal's function.

---

## 5. UNSOLVED QUESTIONS IT TOUCHES

* Is `D_g` (= `A_kill` = `L(M) + 1`) bounded?  Open; `L(47) = 4` is the current maximum
  and the row `1,1,1,2,1,3,3,2,2,2,4,3` (m11..m53) is non-monotone.  Half of the
  derivation target (`agents-shared.md` STATE OF THE DERIVATION, round 29).
* Whether the phase-reduction record law can be derived in the kernel from the walk
  (a correctness proof of `chain_depth.py`'s walk against `Machine17.nextOp`), which
  would make (L3) a single kernel theorem at 17 rather than two verified ends.
* Jacobsthal-type computation: (L3) is a per-prime incremental algorithm for the
  two-class maximal gap; whether it beats covering search at scale is priced in
  `merge-law-h2-test.md` (it does not, beyond a few rungs).

---

## 6. PRIOR-ART CHECK

**Checked 2026-09-03 (harvester, round 30).  Verdict: PARTIAL OVERLAP - the copies-and-
phases picture is Holt-Rudd's one-class recursion (KNOWN); the two-class chain law, the
neighbour-of-hit identity as a theorem for every gear, the phase-reduction record law as
a computation of the maximal gap, and `D_g = A_kill` are NOVEL AS FAR AS SEARCHED.**

| item | exact statement | source | relation |
|---|---|---|---|
| Holt-Rudd Lemma 2.1 | "R2. Concatenate `p_{k+1}` copies of `G(p_k#)`. R3. Add adjacent gaps as indicated by the elementwise product `p_{k+1} * G(p_k#)`" | arXiv:1408.6002 p. 5 (READ) | the one-class version of (L3)'s copies; the "elementwise product" is the phase walk |
| Holt-Rudd Theorem 2.3 | "Each possible closure of adjacent gaps in the cycle `G(p_k#)` occurs exactly once in the recursive construction of `G(p_{k+1}#)`", by CRT | p. 8 (READ) | one-class `phase_bijective`: each lower generator is removed in exactly one copy; two-class: exactly two (harvester `hr_twoclass_r30.py` A) |
| Holt-Rudd Lemma 3.1 | constellations of span `< 2p_{k+1}` have their `j+1` closures in distinct copies | p. 11 (READ) | the one-class "no chain" regime; the two-class threshold is `s_min(g) = min(d_g, g - d_g)`, sharp (harvester follow-on, round 30) |
| Holt-Rudd remark (vi) | a run of `m+1` equal gaps `g` forces `g = 0 mod p` for all `p <= m+2` | p. 7 (READ) | a residue constraint on runs of consecutive gaps, by SMALL primes; (L1) is the constraint by the NEXT prime |
| Ziller-Morack 2016 | GPA / RPA: assign one residue per prime, cover a window; Prop. 2.1 / RPA2 canonical form | arXiv:1611.03310 (lane record, READ round 29) | residues per prime, window covering; no gap-sequence walk, no phase reduction |
| Ziller 2020 Prop. 2.7 | `m in D(k) => m in D(k+1)` | arXiv:2007.01808 (READ) | one-class propagation of a single gap; (L3) computes the NEXT record from the lower sequence |
| Hagedorn 2009 | `h(n)` for `n < 50`, backtracking with capacity bounds | Math. Comp. 78 (2009) - NOT OBTAINED (HTTP 403 twice) | one-class covering search; SECONDARY |
| TwoTeeth.kill_spacing_min (project) | `2u <= y - x` at a fixed gear | `proofs/` (Formalist) | the fixed-gear form of (L2); (L2) is the mod-`g` statement for every gear |

NONE FOUND: any published statement of the two-class chain law, of the neighbour-of-hit
identity, of the maximal gap as a maximum over phases of "gap before + run span + gap
after" on the previous stage, or of the identity between a chain depth and a kill arity.
Searches run 2026-09-03: "Holt Rudd cycle recursion Jacobsthal maximal gap next prime";
"Jacobsthal function upper bound adding a prime recursion"; "consecutive gaps between
reduced residues modulo primorial residue class next prime run length"; "propagation
of coverings Jacobsthal primorial".  Nearest relatives inside the project:
`two-teeth-kill-spacing.md` (T1-T5), `merge-law.md` (the same reduction from the
twin-route side), `even-j-mechanism.md` (`L(M)`), `dictionary-monotonicity-onset.md`
(the depth-0 lemma, the `>= 1 copy` clause of the same count).

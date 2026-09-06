# The law register: the wheels and the exhaust

One project-wide numbering for the laws of the **wheels** (the top machine, the primes in
`(q, q#]` on the raw line, teeth at `0` and `-2`, dominoes `{x, x + 2}`) and of the **exhaust**
(every tier above the wheels).  Created 2026-09-06 by the harvester lane, discharging
`objects_ledger.md` items **O-W7** and **O-X5**.

**Why it exists.**  Six documents numbered their laws independently and three numberings clash:
`top_machine_2.md`'s L30-L38 against `top_machine_3.md`'s, `top_machine_4.md`'s L50-L56 against
`top_machine_5.md`'s, `top_machine_5.md`'s L57-L59 against `top_machine_6.md`'s.  A single
citation like "L53" was ambiguous.  The W-numbers below are unambiguous and permanent.

**The map.**

| document | its numbering | register |
|---|---|---|
| `top_machine_1.md` = `docs/proofs/22-top-machine-laws.md` | L1-L21 | **W1-W21** |
| `top_machine_2.md` | L22-L38 | **W22-W38** |
| `top_machine_3.md` | L30-L45 | **W39-W54** |
| `top_machine_4.md` | L46-L56 | **W55-W65** |
| `top_machine_5.md` | L50-L59 | **W66-W75** |
| `top_machine_6.md` | L57-L66 | **W76-W85** |
| `docs/proofs/23-stack-and-exhaust.md` | S1(a-d), S2, S3, S4(a-b) | **X1-X8**, unchanged |

**Future documents number from W86.**  `top_machine_7.md` (branch R7) is a pre-registration at
the time of writing and carries no laws yet; when it lands it numbers from W86.

**One warning about names.**  `top_machine_1.md` section 4 ends with two unmechanised facts it
calls **"W1"** and **"W2"**.  Those are *not* register entries W1 and W2.  They are closed at
register **W24** (the gap-3 / gap-5 identity) and **W32** (the range record; and doc 1's reading
"the wheel record is a hard ceiling never reached" is *refuted* there).

## Status vocabulary

- **KERNEL** - proved in Lean with zero sorries; the Lean name is given.  See
  `research/proof/top_machine_lean.md` for hypotheses and axiom footprints.
- **PROOF** - a written proof on paper, not in the kernel.
- **MEASURED** - exact computation over the stated range, no proof; exception count given.
- **ROOT** - the conjecture in disguise: the statement is equivalent to, or as hard as, the thing
  the project is trying to prove.
- **REFUTED** - false as stated; the row stays, with the correction named.

## Prior-art vocabulary

- **KNOWN** - the statement is in print.
- **KNOWN VARIANT** - a close statement is in print; the delta is stated.
- **NEW** - no prior art located after the search described in section 3.
- **STANDARD TOOL** - the proof is a routine application of a named tool and the statement is not
  itself a result.

Prior art checked **2026-09-06**, by five parallel literature sweeps (search terms in section 3).
A result found known is not a loss: an independent exact replication validates the machine.

---

# 1. The wheels

## W1-W21 - `top_machine_1.md` / `docs/proofs/22-top-machine-laws.md`

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **W1** | d1 L1 | two teeth at `0`, `-2`, separation 2; `g - 2` slots | KERNEL `strikesR_iff`, `card_open_residues` | **STANDARD TOOL** - the definition of the twin sieve's local condition; written down in every twin-sieve paper (Ziller-Morack arXiv:1706.00317) | none; it is the definition |
| **W2** | d1 L2 | the slots form two arcs, `g - 3` and `1`; the singleton is the shield `n = -1` | KERNEL `open_residues`, `not_strikes_neg_one` | **STANDARD TOOL** - restatement of W1.  The shield half (that `W - 1` and `W + 1` are both coprime to `W`) is Ziller arXiv:2007.01808's proof that `2 in D(k)` | fixes the metric; the (g-3):1 asymmetry is what makes the wheels not the motor |
| **W3** | d1 L3 | **the partner law**: every strike has a partner strike of the same gear at distance exactly 2; a gear's struck set is a disjoint union of dominoes `{x, x+2}` | KERNEL `partner`, `strikes_iff_domino`, no hypothesis | **NEW** as a stated property; the ingredient (`{0, g-2}` are 2 apart) is everywhere.  Nearest: Holt-Rudd arXiv:1408.6002 treat the *one-class* sieve as a dynamical system on the cycle of gaps and never isolate a piece inventory | substrate for W4 and W29; on its own it proves nothing about windows |
| **W4** | d1 L4 | **the forbidden gap 4**: `n`, `n + 4` open implies `n + 2` open; a gap of exactly 4 is impossible | KERNEL `open_of_open_add_four`, `no_gap_four`, no hypothesis | **NEW**.  The one-class analogue question is Ziller arXiv:2007.01808 (`D(k)`, which even differences occur; he shows 2 and 2k always do and conjectures all up to `h(k-1)`); the two-class question appears never to have been asked.  Honest: the proof is two lines and a referee will call it "immediate but unrecorded" | local rigidity: a length-5 window with one open pair is forced to contain a second.  A constraint on the complement, not on primality |
| **W5** | d1 L5 | the wheel count `prod (g - 2)` | KERNEL `wheel_count`, engine `card_filter_crt` | **KNOWN** - Schemmel 1869 `S_2`; the Hardy-Littlewood local factor for `(0,2)`; Holt-Rudd's `N_2(p#) = prod (q - 2)` | bookkeeping |
| **W6** | d1 L6 | shield open; antipodes `n = 2`, `n = -4` open; **the origin clump** - every `n` in `[-(q'-1), q'-3]` open except `0` and `-2`, two maximal runs of `q' - 3` | KERNEL `shield_open`, `two_open`, `neg_four_open`, `clump_open`, `origin_clump` | **KNOWN VARIANT** - the shield is Ziller arXiv:2007.01808's `p# +- 1` argument; the clump is an elementary extension of it and nothing close was located | the clump attains the run ceiling with no search, so W10's lower bound needs no CRT |
| **W7** | d1 L7 | **the mirror** `n -> -n - 2` preserves the open set; unique fixed point the shield | KERNEL `strikes_mirror`, `open_mirror`, `mirror_fixed_iff`, no hypothesis | **KNOWN VARIANT** - Holt-Rudd arXiv:1408.6002 Remark 2.2(v): the one-class cycle of gaps is symmetric, `a` is an `n`-prime iff `p# - a` is.  Delta: two classes, and the fixed point shifts from 0 to `-1` | halves every exhaustive search; forbids any drift argument (see W35) |
| **W8** | d1 L8 | the affine maps preserving the open set are exactly `n -> c(n+1) - 1`, `c = +-1` mod every gear: `(Z/2)^m`; adjacency subgroup `Z/2` | KERNEL `affine_group`, `affine_group_form`, `exists_symmetry`, `sign_count` | **NEW** for the full `2^m` stabiliser and the adjacency subgroup; the mirror half is W7's KNOWN VARIANT.  Corrected at **W70** (the order is `2^{#odd gears}`) | warns that apparent coincidences across gear phases may be symmetry artefacts, not mechanism |
| **W9** | d1 L9 | every gap length has an even count except length 1 | **REFUTED** as a universal statement - false as soon as `N_1 = 0`; PROOF for gears `>= 5` at W27 | superseded, see **W67** | - |
| **W10** | d1 L10 | **the alignment law**: longest run of open pairs `= q' - 3`; longest step-2 chain `= q' - 2`; start counts `prod(g - 2 - L)`, `prod(g - 1 - L)` | KERNEL `run_lt`, `run_attained`, `chain2_lt`, `chain2_attained` | **KNOWN VARIANT** - the run half is the standard "CRT realises every relative phase" alignment argument and the count is Schemmel; the **step-2 chain ceiling `q' - 2` has no located analogue** (it has none in the bottom machine either, whose anchor contains 3) | the chain ceiling is the wheels' own; the run ceiling fixes the metric |
| **W11** | d1 L11 | **the run spectrum** is the second difference of `A(L) = prod(g - 2 - L)`: a polynomial of degree `m - 2`, an arithmetic progression of common difference exactly 6 at three gears, second difference 24 at four | PROOF + MEASURED, 12 wheels every `L`, 0 mismatches | `A(L)` itself **STANDARD TOOL** (Schemmel / admissible-tuple CRT count).  The second-difference law, the degree `m - 2`, the AP-of-difference-6 and the exact top-length count `prod(g - q' + 1)`: **NEW**, nothing located.  Elementary but unpublished | dual to W45; with W45 it makes `F_top` the support edge of an exactly computable function, which the Jacobsthal literature only bounds |
| **W12** | d1 L12 | **the chain law**: `x < y` both struck by a new gear `g` iff `y - x = 0, +2, -2 (mod g)` | KERNEL `chain_law`, no hypothesis | **STANDARD TOOL** - a one-line residue statement | the grammar of adding a gear |
| **W13** | d1 L13 | **the merge law**: every gap of `M + g` is a gap of `M` or a merge of consecutive gaps whose interior openings `g` strikes | KERNEL `merge_law`, no hypothesis | **KNOWN VARIANT** - Holt-Rudd arXiv:1408.6002 Lemma 2.1: concatenate `p` copies of the cycle, then close adjacent gaps.  That *is* the merge, in the one-class cycle.  Delta: two classes, arbitrary `M` | the ladder step |
| **W14** | d1 L14 | letters `{2, g - 2}`, strict alternation of the nonzero classes; the fuel cap becomes `k <= (x_k - x_0)/2` and is **vacuous** | PROOF + MEASURED, 816 struck runs, 0 exceptions | **STANDARD TOOL** | the negative half is the content: the wheels' grammar puts no brake on one gear's sweep, unlike the motor's |
| **W15** | d1 L15 | adjacent open pairs `prod(g - 4)`; member-sharing pairs `prod(g - 3)` | KERNEL as `pair_corr` at `d = 1, 2` | **KNOWN** - Schemmel `S_4` / the Hardy-Littlewood local factors for `(0,2,3,5)` and `(0,2,4)` | bookkeeping |
| **W16** | d1 L16 | **the record is an exact cover**: `F_top(G)` is the largest `L` such that `[0, L)` is covered by one phase per gear, each contributing a singleton, a domino `{x, x+2}`, or (if `g <= L+1`) the long letter `{x, x+g-2}`.  No flanks | PROOF (exact, by CRT); validated against the full-period scan 15 of 15 | **KNOWN VARIANT** - Ziller arXiv:2007.01808 Definition 2.4 ("restricted covering") and Proposition 1.8 are exactly this equivalence for **one** class per prime, with a flank condition; the same reduction is the engine of Erdos-Rankin and of FGKMT arXiv:1412.5029.  Ziller-Morack arXiv:1706.03668 say they "extended and adapted these approaches" for paired progressions but state no theorem.  Delta: the two-class version stated, the explicit piece inventory, and the exactness in both directions with **no** flank condition | this is how the record is computable at all.  **Correction 2026-09-06**: the object it computes is `F(M)`, the `D = 2` cover, **not** `h_2`, whose classes are free (verdict 1); it is the route that left Ziller-Morack at a conjecture verified to `p = 73`, so it is a vehicle, not a lever |
| **W17** | d1 L17 | **the parity law**: every gear `> 2m + 1` implies `F_top = 2m - (m mod 2)`.  The record is decided by the parity of the gear COUNT and by nothing else - not by the gears' sizes at all | KERNEL, **as an equality**: `parity_core`, `parity_upper`, `parity_attained`, `parity_law` (`IsGreatest`) | **NEW**.  Nearest: the classical one-class proposition (if every prime factor of `n` exceeds `k = omega(n)` then `j(n) = k + 1`), background in Erdos, Math. Scand. 10 (1962) 163-170 and Ziller-Morack arXiv:1611.03310.  The size-independence half is that result's direct analogue; **the parity defect `-(m mod 2)` has no analogue in print** - in the one-class case there is no defect, so nobody has had reason to notice one | the exactly-solved boundary case: the base of any induction that adds small gears to a large-gear core.  Its hypothesis is the opposite of the window regime |
| **W18** | d1 L18 | **universal record multiplicity**: the number of record blocks per wheel depends only on `m` - 18, 24, 480, 720 at `m = 3,4,5,6` - and so do the counts just below | MEASURED, 14 large-gear wheels, 0 exceptions; derived from W25, hence conditional on its unproved vanishing | **NEW**.  Nearest: Holt-Rudd arXiv:1408.6002 / arXiv:1503.00231 count occurrences of gap constellations per period by recursion - one class, stage-dependent, no records.  Nothing located claims prime-independence of any such multiplicity | a multiplicity that is a pure function of `m` is an exact per-period identity, which survives being combined with a density estimate.  More promising than the extremal statement alone |
| **W19** | d1 L19 | **the conjugacy**: `n -> 6^{-1}(n + 1)` carries the wheels' open set exactly onto the same gears' opening set in the motor's column coordinate.  Counting and symmetry laws transfer; metric laws do not | KERNEL `conjugacy`, `strikes_iff_col`, `exists_column`, `conjugacy_census` | **STANDARD TOOL** for the map (the affine action on admissible tuples is in every account of admissible `k`-tuples).  The counting-vs-metric organising principle is **NEW** but low-value - a hygiene rule, not a result | negative-space: it says `F_top` cannot be imported from any other pattern-sieve, which kills a class of shortcuts.  It does **not** reduce Polignac `2k` to the twin case, because scaling changes the difference |
| **W20** | d1 L20 | **no fold**: the open pairs are equidistributed mod 2, mod 3 and mod 6 (within 2 per class in every wheel) | MEASURED, 12 wheels | **STANDARD TOOL** - CRT on coprime moduli.  The corollary that the sixfold `6k +- 1` belongs to the primes 2 and 3 alone is folklore (it is how every account parameterises twins) | closes off "the wheels have their own anchor"; see W34, W75 |
| **W21** | d1 L21 | **the gear zone**: for `n <= Z` the pair `n` is open iff both members are `q`-smooth; the in-use record lies there | PROOF; sharpened and kernel-checked at **W55** | **KNOWN** - Stormer 1897 and Lehmer, Illinois J. Math. 8 (1964) 57-69 (which treats difference 2 explicitly); the object is OEIS A002071/A002072 | see W55-W57 |

## W22-W38 - `top_machine_2.md`

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **W22** | d2 L22 | **the gap census law**: `N_d(G) = sum_{S subset [1,d-1]} (-1)^{|S|} prod_g (g - |E_g(S)|)`, `E_g(S) = ({0,-2,-d,-d-2} u {-j,-(j+2) : j in S}) mod g` | PROOF (CRT + inclusion-exclusion); **not in the kernel** | **KNOWN VARIANT** - Steven Brown, *Distance between consecutive elements of the multiplicative group of integers modulo n*, arXiv:2311.06873 / Notes Number Theory Discrete Math. 30(1) (2024) 81-99, has exactly this shape for **one** class per prime: the CRT product `nu(T) = prod (p - card(T mod p))` (Thm 3.1) and the inclusion-exclusion `kappa(T) = sum_X (-1)^{|X|} nu(T u X)` (Thm 2.1).  **Delta**: ours is the two-class instantiation, where each interior position needs a gear on *one of its two teeth*; and ours allows arbitrary pairwise-coprime odd gears, not squarefree primes.  Holt's cycle-of-gaps machinery is **recursive** (verified across arXiv:1408.6002, 1510.00743, 2502.20470) with only `N_2 = prod(q-2)` in closed form, so this is not Holt's result | the only exact object here that says *which gap lengths are impossible*, which is what "no long twin-free stretch" needs.  **Cite Brown in any write-up** |
| **W23** | d2 L23 | **the universal signature**: every gear `> d + 2` gives `|E_g(S)| = e(S)` independent of `g`, so `N_d` is a universal polynomial with signature `c_e(d)`; departures need a gear dividing a difference `<= d + 2` | PROOF (mechanism) + MEASURED `d <= 16` | **KNOWN VARIANT** - Brown arXiv:2311.06873 Lemma 3.1 + Remark 3.1 proves exactly the threshold universality, one class | the mechanism behind W24, W25 |
| **W24** | d2 L24 | **W1 closed**: `N_3` and `N_5` share the universal signature `prod(g-4) - 2 prod(g-5) + prod(g-6)`; the gap-3 classes collapse only for `g | 3` or `5`, the gap-5 classes for `g = 7`.  Hence equal counts iff every gear exceeds 7; `(3,5)` the only coincident pair and `4` the only identically zero length, `d <= 16` | PROOF; 16 wheels including a `6.7e9` full period | **KNOWN VARIANT for the phenomenon, NEW for the instance and its mechanism.**  Brown arXiv:2311.06873 eqs (16)-(17) has `K(2,P) = K(4,P) = prod(q-2)` exactly, and Holt-Rudd arXiv:1510.00743 has `n_{2,1}(p#) = n_{4,1}(p#)` at every stage - but **for a different reason**: Brown's comes from a parity annihilation (Prop 3.2), ours from *equal class counts*.  Nobody publishes the polynomial, the failure at exactly gear 7, or the uniqueness claims.  Note `N_4 = 0` has the one-line proof W4, so treat that half as folklore | an exact identity that breaks at exactly one gear is the kind of rigid constraint that survives into a proof; the "same class count implies same polynomial" mechanism generalises |
| **W25** | d2 L25 | **the degree law**: `N_d = sum_k (-1)^k sigma_{m-k} M_k(d)`, `M_k(d) = sum_e c_e(d) e^k`; degree `m - r(d)` in the gears, gear-independent exactly when `r(d) = m`, value `(-1)^m M_m(d)` - which derives W18 | MEASURED; rests on the **unproved** vanishing `M_k(d) = 0` for `k < r(d)`, verified to `d = 16` (open item O-W1) | **NEW**.  Brown has the threshold (W23) but no elementary-symmetric expansion, no moments, no degree structure; Holt has no polynomial at all | the only located mechanism that makes a gap count *not grow* with the gear set - a genuine lever on record-stretch questions |
| **W26** | d2 L26 | `r(d)` is the parity covering number `ceil(ceil((d-1)/2)/2) + ceil(floor((d-1)/2)/2)` for every `d <= 16` except `d = 4`; hence `F_top(m) = max{d : r(d) <= m} - 1` reproduces the parity law.  `d = 4` is the unique exception because the gap's cover has closed boundaries and the record's has free ones | MEASURED, `d = 1..16`; census record = cover record 15 of 15 | **NEW** | ties the census (W22) to the record (W16, W17) - the two halves of the machine in one identity |
| **W27** | d2 L27 | `prod (g - e)` is odd iff `e` is even; hence `N_d` is odd only at `d = 1` | PROOF (second, independent proof of W9), `d <= 16` | **STANDARD TOOL** - parity of a product of odd factors | - |
| **W28** | d2 L28 | the joint census is the CRT product; all `m` gears strike on exactly `2^m` classes, of which exactly two (`n = 0`, `n = -2`) have every gear on the same tooth - **the origin is the unique total collision, and the clump is its shadow** | PROOF (CRT) + MEASURED, 14 tuples, deviation 0 | **KNOWN** - textbook CRT.  The "unique total collision" reading is presentation, not new mathematics; do not put it forward as a finding | explains why the clump near 0 is not evidence of anything global |
| **W29** | d2 L29 | **the collision law**: with every gear `> L + 1`, no three distinct traces pairwise intersect (pairwise-intersecting distance-2 dominoes lie in a span of 2).  Hence the record cover is a **perfect tiling for even `m` and carries exactly one unit of waste for odd `m`** - the parity law's `-(m mod 2)` in geometric form | PROOF (one line) + MEASURED, 1,354 triples, `L <= 12`; waste 11 of 11, `m = 2..12` | **NEW - the strongest novelty claim in the wheels.**  Nothing resembling either half was located.  The covering-systems literature (Mirsky-Newman; Znam; Hough, Ann. of Math. 181 (2015) 361-382; Balister-Bollobas-Morris-Sahasrabudhe-Tiba) is about *infinite* covers of `Z` with distinct moduli and reciprocal-sum constraints, and never studies the waste parity of a finite interval cover with two classes per modulus | the only item that converts a structural observation into an **exact** extremal value rather than a bound.  Two cautions: the hypothesis is the easy regime (no long letters), and it bounds where twins *can* be, not that one *is* |
| **W30** | d2 L30 | the record of a triple is 5 without gear 7 and 6 with; of a quadruple 8 without and 9 with.  The other gears' sizes never matter | MEASURED, exhaustive: 1,540 triples, 7,315 quadruples, 0 exceptions | **NEW as data** - Ziller-Morack arXiv:1706.03668 compute `h_2` for **primorials only** (`p <= 73`, OEIS A288815), never for an arbitrary sub-multiset of odd primes; and (correction 2026-09-06) `h_2` is the free-class record, a different function of the same gear set - see verdict 1.  But it is *corroborating* W17/W71, not independent: at `m = 3` the W71 threshold is `2m + 3 = 9`, so 7 is the only sub-threshold gear in 7..97 | evidential |
| **W31** | d2 L31 | **the sub-threshold reduction**: `F_top(G)` is a function of `m` and of `{g <= F_top + 1}` alone; larger gears enter only by their number | MEASURED, 90 comparable cases, `m = 3..8`, 0 exceptions; formula at **W62** | **KNOWN VARIANT** - "primes exceeding the window can each remove at most one position, so the search is over the small primes and the large ones enter as a count" is the standard engine of every Jacobsthal computation: Hagedorn, Math. Comp. 78 (2009); Costello-Watts arXiv:1208.5342; Ziller-Morack arXiv:1611.03310; Ziller arXiv:2007.01808 Cor 3.2.  **Delta**: the two-class version, where a large gear removes a *domino* rather than a point | this is where the leverage is: it is the only statement in W17-W31 that applies with small gears present, i.e. in the actual window regime |
| **W32** | d2 L32 | **the range record is a first hit on the wheel's census**: `F_range(N) = max{d : W/c(d) <= N} - 1`; the wheel record IS reached, at 0.005% to 10.9% of the period | MEASURED, within one unit at 19 of 21 checkpoints | **KNOWN VARIANT** - Kourbatov, J. Integer Seq. 16 (2013), arXiv:1301.2242; Kourbatov-Wolf arXiv:1901.03785; Kourbatov arXiv:2002.02115 give exactly this "set the expected count to 1 and solve" first-occurrence heuristic, with a Gumbel law, **for the primes**.  **Delta**: ours asserts it *exactly*, for a fixed periodic sifted set with a known deterministic census `c(d)`.  Honest: it cannot be a theorem in general - `W/c(d)` is a mean spacing and long gaps are not evenly spread - so **the exactness is the finding**, and it is the thing to try to break | high.  It converts a per-period statistic into a statement about a finite range - the join between the census (W22-W25) and existence.  It also **refutes** doc 22's W2 ("the wheel record is a hard ceiling approached slowly"), which was an artefact of stopping at `N = 10^7` |
| **W33** | d2 L33 | the record blocks are pinned modulo the small gears: eight blocks of `{7..31}` in four mirror pairs summing to `W - 33`, two residues mod 1001 | MEASURED, one wheel, full period | **NEW** (measured on one wheel; too thin to claim) | location, not existence |
| **W34** | d2 L34 | **the corridor**: an anchoring gear needs `g - 2 <= 2`, i.e. `g <= 4`, so **no set of top gears can anchor, for any split**; the smallest gears give a corridor of density `>= 0.58`, uniformly filled (`prod_{larger}(g-2)` per slot) | PROOF (a count) + MEASURED, 7 wheels; uniform descent 5 of 5 | **STANDARD TOOL** - the arithmetic identity "#open classes `= g - 2`" plus CRT descent.  Worth citing as a *comparison* rather than prior art: Balister-Bollobas-Morris-Sahasrabudhe-Tiba prove "every distinct covering system has a modulus divisible by 2 or 3" - the same shape of statement, where it is a hard theorem | closes a whole family of approaches (no large-prime set anchors).  A negative result, and a real service to the wall map |
| **W35** | d2 L35 | **no direction**: the cyclic gap word read from the shield is a palindrome | MEASURED, 6 wheels (23 in d5); mechanism = W7 | **KNOWN VARIANT** - Holt-Rudd arXiv:1408.6002 Remark 2.2(v) (restated as 2.3(v) in arXiv:1510.00743): "except for the final 2, the cycle of differences is symmetric, `g_{k,j} = g_{k,phi(p_k#)-j}`".  **Delta**: two-class, mirror `n -> -n-2` with fixed point `-1` rather than the `r/-r` pairing, and arbitrary gear set rather than an initial segment.  Publish as a lemma, never as a discovery | forbids any directional or drift argument - kills a family of would-be proofs cheaply |
| **W36** | d2 L36 | **the metric anchor**: run `q'-3`, chain `q'-2`, clump `2(q'-3)+1` are functions of `q'` alone; the record is a function of `m` alone.  Neither fixes the other | MEASURED, 14 wheels; components proved at W6/W10/W17 | **STANDARD TOOL** - a restatement of proved laws.  The *separation* is presentation | tells you which knob does what |
| **W37** | d2 L37 | **the removal law**: removing `q'` divides `W` by `q'`, the open count by `q'-2`, `N_1` by `q'-4`; the ceilings grow; and `F_top` falls by 3 at even `m`, by 1 at odd `m` | MEASURED, 16 steps over four chains | **KNOWN VARIANT** - Holt-Rudd arXiv:1408.6002 eq. (2) / arXiv:1510.00743 Cor 3.2 have `N_2(p_{k+1}#) = (p_{k+1}-2) N_2(p_k#)` and the constellation factor `(p - j - 1)`; read backwards these are ours, and because the counts are CRT products the order is irrelevant.  The `prod(g-4)` factor is the same at `j = 3`.  **The `F_top` step (`-3` even, `-1` odd) is NEW** - nothing located - though it is a corollary of W17.  Caveat to state: the `(q'-4)` factor is exact only for `q' >= 5` | the count laws are the arithmetic backbone of any window census; the `F` step is a constraint on the record |
| **W38** | d2 L38 | **removal independence**: in the large-gear regime `F_top(G \ {g})` is the same whichever `g` leaves; fails exactly outside the regime | MEASURED, 9 of 9; fails at `{7,11,13,17}` | **NEW**, but an immediate corollary of W17 (the record depends only on `m`).  Present it as a corollary; claiming it independently overstates.  Nearest: Ziller-Morack arXiv:1706.00317 Remark 2.2 gives only `j_2(n_1 n_2) > j_2(n_1)` | restatement of size-independence |

## W39-W54 - `top_machine_3.md`

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **W39** | d3 L30 | **THE MEX FORM**: if every gear `> 2m`, the next open pair after `x` is `x + mex{(-x) mod g, (-x-2) mod g : g in G}`.  `O(m)` operations, no scan, no period | KERNEL `mex_form` (`IsLeast`), `mexS_le` | **NEW as stated.**  "mex" / minimal excludant appears in combinatorial game theory (Sprague-Grundy) and in partitions (Andrews-Newman 2019); **no number-theoretic sieve use was located**.  The nearest functional prior art is the wheel sieve (Pritchard), which gets the next coprime by table lookup over a period, not by an `O(m)` formula from residues.  Elementary: it is the constructive content of W17's threshold | an exact `O(m)` local model wherever the smallest gear is large - the right primitive for an induction that adds small gears one at a time, and cheap to formalise.  Not a route to the window: `F_top < q'` fails badly at the real anchor |
| **W40** | d3 L31 | the location bound `L(x) <= 2m - (m mod 2)` in mex form | KERNEL `mexS_le_parity` | **NEW** - re-proof of W17 from the closed form | - |
| **W41** | d3 L32 | the general mex form `L(x) = mex(union_g ({a_g,b_g} + g Z_{>=0}))`, truncatable at any proved bound `B` | PROOF; 24,000 in-use walks, 160-443 gears, 0 mismatches | **STANDARD TOOL** - the definition unrolled | the practical oracle at in-use gear sets |
| **W42** | d3 L33 | **the counting bound** `L <= 2m/(1 - 2H_S)`, `H_S = sum_{g <= L} 1/g`, valid when `H_S < 1/2` | PROOF | **STANDARD TOOL** - the trivial union bound; the one-class "each large prime kills one point" is folklore (Erdos, Math. Scand. 10 (1962)) | dies exactly where it is needed - see W59, W60, W61 |
| **W43** | d3 L34 | **the twin-candidate mex form**: three teeth `0, -1, -2`, `R(x) = mex` of `3m` residues, if every gear `> 3m` | KERNEL `triple_mex_form`, `mexT_le` | **NEW as stated**; same pigeonhole-plus-CRT as W39 | the control for W44 |
| **W44** | d3 L35 | **THE TRIPLE RECORD**: every gear `>= 3m + 3` implies `F_3(G) = 3m` **exactly**, with no parity defect - a solid triomino tiles, a gapped domino does not | KERNEL `triple_law` (`IsGreatest`), `triple_upper`, `triple_attained` | **NEW as stated**, close to STANDARD TOOL on its own (one line of pigeonhole plus CRT).  **The genuinely new content is the contrast**: the defect `-(m mod 2)` is caused by the tooth *separation*, not the tooth *count* - located nowhere.  Nearest parity-flavoured lemma in the area is Ziller arXiv:2007.01808 Lemma 3.1, which is a parity *reduction*, not a *defect* | isolates exactly what twinness costs over the consecutive-triple sieve, at the large-gear end |
| **W45** | d3 L36 | **the `C`-identities**: `#{L = j} = C(j) - C(j+1)`; `N_d = C(d-1) - 2C(d) + C(d+1)`; `F_top = max{j : C(j) > 0}`; `sum_x L(x) = sum_j C(j)`.  The gap census is the second difference of the all-struck count, dual to W11's run spectrum | PROOF; all five identities, 10 wheels, 0 mismatches | **NEW** - no located statement of the duality, and none of `F_top` as the support edge of `C` | **the sharpest re-expression located**: it makes the real-teeth (`D = 2`) two-class record the support edge of an exactly computable function, where the whole Jacobsthal literature only bounds it.  It does **not** reach `h_2`, whose classes are free (verdict 1) |
| **W46** | d3 L37 | **the closed form of `C(j)`**: `C(j) = sum_{k,e} (-1)^k T(j,k,e) prod_g (g - 2k + e)`, `T` a convolution of path-subset counts.  "All open" is a per-gear condition and gives one product; "all struck" is not, and gives a signed sum | PROOF; 6 wheels, 0 mismatches; `T` brute-forced to `j = 12` | **NEW as assembled**; ingredients standard - `binom(k-1,e) binom(n-k+1,k-e)` is the classical Kaplansky/Riordan subsets-with-adjacencies count.  The structural moral is the standard multiplicative-vs-complement duality; state it as an observation | second only to W45 and W22: an exact handle on the quantity the literature can only bound, and the structural reason the record resists a product formula (see W50) |
| **W47** | d3 L38 | **the hop law and the collapse**: `g` hops at the landing `y` iff `y = 0` or `-2 (mod g)`; if `g > F_G + 3` the hop chain is at most 2, and a double hop occurs iff `y = -2 (mod g)` and the lower gap is exactly 2 - a one-line non-recursive layer | PROOF; 12 layers, 391,048 positions, 0 exceptions; sharp in the other order | **KNOWN VARIANT** - Holt-Rudd arXiv:1408.6002 Lemma 2.2: at stage `p_{k+1}` the cycle is concatenated `p_{k+1}` times and "each possible closure of adjacent gaps occurs exactly once".  **Delta**: two classes, and the explicit chain-length-2 collapse with its iff | the layer that makes the walk computable gear by gear |
| **W48** | d3 L39 | the nested form and its lazy cost (`m` layer tests plus fewer than one gap lookup); smallest gear first | PROOF + MEASURED, 7 wheels | **STANDARD TOOL** - an algorithm | computational |
| **W49** | d3 L40 | **the per-gear transform**: `u_g^(a) = -(1 + omega_g^{2a})/g`, which in the shield coordinate `n + 1` is the real `-(2/g) cos(2 pi a/g)`.  The wheels are the `u = 1` machine; the motor's fold is replaced by one translation | PROOF + exact DFT, 5 wheels, 32,077 frequencies, max error 1.1e-16 | **STANDARD TOOL** - Ramanujan sums (1918), E. Cohen's generalisations, and the DFT-of-even-functions-mod-`r` theory (Haukkanen arXiv:1708.04507); CRT multiplicativity of the full spectrum is standard.  Sieve-side use: Gadiyar-Padma arXiv:math/0601574 | W19's conjugacy seen on the Fourier side |
| **W50** | d3 L41 | **full spectral support**, and what the spectrum cannot decide: `O^(a) != 0` always, so the spectrum **decides the run record** (a product vanishes iff a factor does) and **cannot decide `F_top`** (a signed sum's positivity is not a support question) | PROOF; minima 3.3e-07 to 3.1e-05, never 0 | non-vanishing is **KNOWN VARIANT / a parity exercise** (`cos = 0` needs `4a = g mod 2g`, impossible for odd `g`).  **The dichotomy is NEW as a stated principle**, though it is a re-expression of the widely understood reason circle-method analysis gives no tuple lower bounds | wall-mapping: do not spend rounds on a spectral attack on the blocked record |
| **W51** | d3 L42 | **the striker-parity bit**: `#even - #odd = prod(g - 4)` exactly, the same polynomial as the domino count | PROOF (one character sum per gear); 10 wheels, exact | **KNOWN** - Schemmel `S_4` plus a one-line multiplicative character sum.  Present as an identity, cite Schemmel 1869 | - |
| **W52** | d3 L43 | **the XOR run reaches the record**: the longest run of ones of the striker-parity bit equals `F_top` exactly, because a record block struck exactly once at every cell always exists | MEASURED, 10 of 10 wheels, both parities of `m` | **NEW** - nothing in the Liouville / parity-problem literature is a finite-period run statement.  **But not independent of W29**: its proof needs the zero-waste statement, so W29 and W52 stand or fall together and must not be reported as separate evidence | attractive because the parity bit has a product formula while the record does not.  Caution: this is pushing on the parity problem from the inside (Selberg 1949; Tao, *Open question: the parity problem in sieve theory*, 2007) - expect it to bound the sieve record, not the primes |
| **W53** | d3 L44 | **the correlation is a true product**: `B(d) = prod_g c_g(d)`, `c_g = g-2, g-3, g-4` at `d = 0, +-2, else`.  Never vanishes, so every distance occurs | KERNEL `corr_prod`, `pair_corr` | **KNOWN / STANDARD TOOL** - the exact per-period count `prod (g - |H mod g|)` for `H = {0,2,d,d+2}` is the Hardy-Littlewood (1923) local factor; Brown arXiv:2311.06873 Thm 3.1 writes it verbatim; non-vanishing is Hensley-Richards admissibility.  No delta.  Related: Aryan, Mathematika (2014) arXiv:1302.2296; Montgomery-Vaughan, Ann. of Math. 123 (1986) 311-333 | the *contrast* with W54 is the useful part - but note it is essentially Ziller arXiv:2007.01808's premise, so the framing is validated by the literature rather than new to it |
| **W54** | d3 L45 | **the holes**: in the pair view the only gap hole below the record is `d = 4`; in the triple view they are exactly `d = 2` and `d = 3` | KERNEL `no_pair_gap_four`, `no_start_gap`, no hypothesis | **NEW** (with W4).  The one-class analogue question is Ziller arXiv:2007.01808's `D(k)`; the two-class version was never asked | with W4, the local rigidity of the open set |

## W55-W65 - `top_machine_4.md`

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **W55** | d4 L46 | **THE GEAR-ZONE IDENTITY**: for `1 <= n <= Q - 2` the pair `n` is open iff `n` and `n + 2` are both `q`-smooth | KERNEL `smooth_zone`, `smooth_zone_num`, `wheels_smooth_zone` (= X7) | **KNOWN.**  The equivalence is one line, and the resulting object - `q`-smooth pairs at difference 2 - is exactly the Stormer-Lehmer problem: Stormer 1897; **D. H. Lehmer, *On a problem of Stormer*, Illinois J. Math. 8 (1964) 57-69**, which reduces to `~2^k - 1` Pell equations, computes all 869 41-smooth neighbours and **explicitly extends to difference 2**; OEIS A002071/A002072 (A002071's own comment states the difference-2 form).  Still actively recomputed for isogeny cryptography (Costello et al. arXiv:2211.04315; arXiv:2509.17699).  Delta: only the framing, "the bottom of a sieve by an interval of primes is a Stormer list" | negative-direction: the bottom of the range contains **no twin-prime information at all**, and it fixes the constant in W57 |
| **W56** | d4 L47 | **THE IN-USE RECORD IS THE LARGEST GAP OF A FINITE LIST**: `F_zone = max_i (s_{i+1} - s_i - 1)` over the `q`-smooth-pair list below `Q`, value **and** position | PROOF (from W55); 36 machines + 24 sweep points, 0 exceptions | **KNOWN VARIANT with a NEW consequence.**  Every ingredient is published - Stormer/Lehmer (the list), Tijdeman 1973 and Heath-Brown, Mathematika (2019) arXiv:1808.02947 (gaps between smooth numbers), FKMPT, JEMS (2021) arXiv:1802.07604 (gaps in sieved sets).  **What was not located anywhere: any study of the gap structure of the Stormer-Lehmer list itself.**  Exact delta against FKMPT, worth stating: theirs is a full-period theorem driven by density; ours is confined to `[1, Q]`, where the sieve is emphatically not equidistributed and the answer is Diophantine, not probabilistic | it proves the in-use record is not a statistic of the gear set, so **any theory that predicts the record from densities is measuring the wrong thing** |
| **W57** | d4 L48 | **THE PROVED LOWER BOUND**, no sieve estimate: `F_range(N) >= max(list maxgap, Q - 2 - s_k)`; for `Q > s(q)`, `F_range >= sqrt(N) - s(q) - 2` | PROOF (finiteness is Stormer/Lehmer; the members used are computed exactly, so unconditional) | **NEW as far as searched** for the bound; the constants are classical | a floor, and floors do not deliver twins.  Diagnostic value |
| **W58** | d4 L49 | **no bound of the parity kind**: `Q ~ (m log m)/2`, so `F_range >= (m log m)/2 (1+o(1)) - s(q)` and no bound linear in `m` can hold | PROOF + MEASURED, 36 machines | **NEW as a statement**; a corollary of W57 | kills "extend the parity law to the in-use machine" |
| **W59** | d4 L50 | **where the covering bound dies, in closed form**: alive exactly while `F < L*(q) = exp exp(1/2 + sum_{p<=q} 1/p - M)`; classifies 35 of 36 machines | PROOF (Mertens) + MEASURED | **STANDARD TOOL** (Mertens' theorem) | it never sees the gear-zone record |
| **W60** | d4 L51 | **the ceiling on union bounds**: a union bound over covering patterns controls lengths only to `d <= 2 log N / log q'`; the in-use record exceeds it by 4 to 520 | PROOF | **STANDARD TOOL / KNOWN VARIANT** - Brun-truncation growth is the classical shadow; the project's own `moment-degree-ceiling` is the LP-side analogue | **the in-use next-opening bound is not reachable by counting** - a wall edge, stated exactly |
| **W61** | d4 L52 | **in use the tail is empty**: certified covers give `F_top >= Q`, so every gear is a core gear; the parity apparatus measures nothing | MEASURED, 26 of 27 in-use machines | **NEW** (a structural observation about the in-use regime; nothing located) | explains *why* W42 goes vacuous in use: the parity law and the covering bound are statements about a tail that does not exist here |
| **W62** | d4 L53 | **THE CORE/TAIL RULE**: `F_top(G) = max{L : min over core phase vectors of D(U) <= t}`, `D(U)` the domino cost (sum of `ceil(run/2)` over step-2 runs per parity class) | PROOF from W16 (sketch; being closed in `top_machine_7.md`); 0 mismatches on 13 known records, 89 sets decided | **KNOWN VARIANT** - the core/tail principle is the standard engine of every Jacobsthal computation (Hagedorn 2009; Costello-Watts arXiv:1208.5342; Ziller-Morack arXiv:1611.03310).  **Delta**, and it is real: the two-class version, where a large gear removes a *domino* not a point, plus the explicit cost `D(U)` and the max-min certificate.  State it as "the standard core/tail reduction, instantiated for the paired sieve, with an explicit cost function" | **the highest-leverage item in the wheels**: it converts an infinite family into a finite core computation plus one integer `t`.  If anything here reaches the target, it goes through W31/W62 |
| **W63** | d4 L54 | **the additive form** `F_top = F_core + 2t - (t mod 2)` holds at exactly the sets with an empty core, and fails at all 25 with a core | PROOF (mechanism) + MEASURED, 64 of 89; pre-registered as refuted with `{13..41}` named in advance | **NEW** (a sharpness statement; nothing located) | says exactly which sets the parity apparatus can decide |
| **W64** | d4 L55 | **no saturation**: `F_range` is linear in the largest gear (`Q - 160 - u` at `q = 5`, `Q` from 316 to `10^6`); gears far above `sqrt(N)` are not harmless | PROOF (two exact causes) + MEASURED, 24 + 12 points | **KNOWN - and demoted.  The measured constant 160 IS `s(5)`**, the largest `n` with `n` and `n + 2` both `{2,3,5}`-smooth (`160 = 2^5 * 5`, `162 = 2 * 3^4`): the last element of the Stormer-Lehmer difference-2 list, Lehmer 1964 / OEIS A002072.  So the "law" is W57's bound running at equality, and the constant can be predicted in advance for every `q` from the published tables (Luca-Najman, Math. Comp. 80 (2011) to 100-smooth).  The second half - added gears keep merging blocks - is a standard observation about sieving past the square root | as a **check**: any model of the record that does not reproduce `s(q)` exactly at small `q` is wrong.  Predictive value beyond that: low |
| **W65** | d4 L56 | **the two regimes and the crossover**: `F_range = max(F_zone, A)`; the zone wins in 26 of 36, and each `q` has one crossover `N` after which it wins for good | MEASURED, 36 machines; the pre-registered ceiling `A <= 400` refuted (419) | **NEW as a description**; the first-hit half of it is Kourbatov-style heuristic (see W32) | tells you which of the two objects a bound must attack at a given `N` |

## W66-W75 - `top_machine_5.md`

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **W66** | d5 L50 | **the sharp mex criterion**: `M(x) < q'` implies `L(x) = M(x)`; hence `F_top(G) < q'` makes the mex form exact everywhere.  Strictly weaker than W39's `q' > 2m` | PROOF (one line) + MEASURED, 94,774 positions, 18 wheels, 0 exceptions; predicts "0 failures" at 26 of 26 gear sets | **NEW as stated**; elementary.  The useful half is that it is checkable *at each `x`* rather than requiring a global hypothesis | makes W39 usable at gear sets where the global hypothesis fails |
| **W67** | d5 L51 | **exactly one gap length has an odd count**, and it is the mirror-self-paired gap (`2a + d + 2 = 0 mod W`) - no hypothesis at all | PROOF (from W7) + MEASURED, 15 wheels | **NEW**; corrects **W9**, which is false as soon as `N_1 = 0` | - |
| **W68** | d5 L52 | **every counting law is a tooth-count law**: replacing 2 by `t_g = |{0,-2} mod g|` carries the slot count, wheel count, run and chain counts, correlation, census, parity bias, transform and all-struck classes down to gear 2 | PROOF + MEASURED, 23 wheels, 0 mismatches everywhere | **STANDARD TOOL** - it is `|E mod g|` bookkeeping; Brown arXiv:2311.06873 Thm 3.1 already writes `prod (p - card(T mod p))` in exactly this generality | it is what lets the wheels' laws be tested at `q' = 2, 3`, i.e. against the motor |
| **W69** | d5 L53 | **the two ceilings, corrected**: the run ceiling is `max(q'-3, 1)`; the **chain** ceiling is `q_odd - 2`, the smallest **odd** gear minus two, because gear 2 is invisible to a step-2 chain | PROOF (mechanism) + MEASURED, 14 of 14 | **NEW** as a correction; standard in method.  Corrects **W10** at `q' = 2` | - |
| **W70** | d5 L54 | **the symmetry group, corrected**: order `2^{#odd gears}`, not `2^m`, because the two signs coincide mod 2 | PROOF + brute force at `W = 30, 70, 105, 165, 210, 286, 385` | **NEW** (with W8).  Corrects **W8** at `q' = 2` | - |
| **W71** | d5 L55 | **THE SHARP PARITY THRESHOLD**: `F_top = 2m - (m mod 2)` **iff** the record cover is a tiling by free distance-2 dominoes, **iff** `q' >= 2m + 1` (even `m`) or `q' >= 2m + 3` (odd `m`) | PROOF + MEASURED, 35 gear sets, 7 boundary pairs, 0 exceptions; tiling/parity equivalence 28 of 28 | **NEW.**  Nearest: the classical one-class threshold `q_1 > k` implies `j(n) = k+1`, and the admissible-tuple fact that admissibility need only be checked at `p <= k` (Polymath8; Thangadurai, INTEGERS 14 (2014)).  **Delta**: prior art gives a *sufficient* one-class threshold; nothing gives an *iff*, nothing gives two thresholds split by the parity of `m`, and nothing characterises the extremal configuration as a free domino tiling.  Corrects **W17**'s hypothesis, which is sufficient at both parities and **not necessary** at even `m` | it tells you exactly which small gears are the whole difficulty - the precise boundary where the elementary analysis stops |
| **W72** | d5 L56 | **THE ANCHOR RESCALING LAW**: `F(G+{2}) = 2F_2(G)+1`, `F(G+{3}) = 3F_3(G)+2`, `F(G+{2,3}) = 6 F_col(G) + 5` | PROOF (mechanism) + MEASURED, **30 of 30 exact** | **KNOWN VARIANT.**  The one-class analogue `j(2n) = 2j(n)` (and `j(p^k n) = j(pn)`) is the basic reduction of every Jacobsthal computation - Ziller-Morack arXiv:1611.03310, Hagedorn 2009 - and the composed `6F + 5` is the record-level form of the `6k +- 1` folklore.  Ziller-Morack could not have reached `p = 73` without collapsing 2 and 3 somehow, so read this as *used but unstated*.  **Delta**: the exact identity with the sub-lattice coordinate (teeth `{0, -2r^{-1}}`), making it an equality rather than an inequality | practical: a 6x reduction of the search space.  **ACTION DISCHARGED AND CORRECTED, 2026-09-06** (`research/harvest/r1/jacobsthal_check.md`): the rescaling does **not** put `F` on `h_2`'s normalisation - `h_2` frees the two classes and `F` fixes them at `D = 2` (verdict 1), so Table 1 / A288815 is the wrong comparison and would have reported a machine-wide failure (A288815 gives 18, 66, 150, 192, 258, 366 at `n = 3, 5..9` against `6F` = 12, 42, 66, 108, 150, 204).  The validation that exists: `6 F(M)` against the **real** twin-candidate max gap at `p_n#` by direct sieve, **7 of 7 exact**, `n = 3..9`, to `p_9# = 223 092 870` |
| **W73** | d5 L57 | **the certified column mex** for the motor: `M_B(x) <= B` certifies the next open column, at `2 sum_g (1 + floor(B/g))` numbers | PROOF (self-certifying) + MEASURED, 890,501 walks, 0 mismatches | **NEW as stated**; the certificate form is the useful half | the motor's next-opening formula, obtained by transporting W39 through W19 |
| **W74** | d5 L58 | **the hop collapse transfers**; the double-hop rule becomes "the lower gap equals the forward letter of the landing's tooth" | PROOF + MEASURED, 18 pair layers, 4 column layers, 10,860 hits, 0 exceptions | **NEW** (with W47) | the chain bound survives the change of coordinate; the rule naming the number 2 does not |
| **W75** | d5 L59 | **the anchoring dichotomy**: a gear folds the line iff `g - t_g = 1`, i.e. `g = 3` or `g = 2`; with an exact list of what each destroys and what survives | PROOF + MEASURED, 23 wheels | **STANDARD TOOL** - the arithmetic identity of W34, plus the observation that `{0,-2}` collapses mod 2.  The comparison to Balister et al. ("every distinct covering system has a modulus divisible by 2 or 3") is the interesting neighbour, not prior art | the sharpest form of "the sixfold belongs to the anchor" |

## W76-W85 - `top_machine_6.md`

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **W76** | d6 L57 | **THE ZONE RULE**: for every `n <= Q^2`, `n` is admissible iff `n = s P` with `s` `q`-smooth and `P` either 1 or a prime `> Q` | PROOF; KERNEL as `quiet_zone` (= X8); 45 machines, every cell of `[1, Q^2]`, 0 exceptions | **KNOWN / STANDARD TOOL.**  It is the Legendre-Meissel-Lehmer partial sieve: `phi(x, a) = pi(x) - a + 1` whenever `x < p_{a+1}^2` (Lagarias-Miller-Odlyzko, Math. Comp. 44 (1985)), with a smooth factor bolted on; also the "large-prime variation" folklore of QS/NFS (a cofactor `<= B^2` must be prime).  The ambient set is named in print: **Weingartner, *Integers free of prime divisors from an interval, I / II*, Acta Arith. 98 (2001) 117-131 and 104 (2002) 309-343**; Tenenbaum, Ann. Sci. ENS (4) 19 (1986) 1-30.  **NAMING WARNING**: do **not** call this "semismooth" - Bach-Peralta, Math. Comp. 65 (1996) 1701-1715 use it for the *opposite* inequality (one prime bounded above) | scaffolding, not leverage: it is the reason the window's arithmetic above `Q` is entirely prime-distribution arithmetic - precisely the difficulty being routed around |
| **W77** | d6 L58 | **the two edges**: the first admissible non-smooth `n` is `p_1 = nextprime(Q)`; the rule first fails at `p_1^2`; **there is no transition band** | PROOF; 45 of 45 | **KNOWN** - an immediate restatement of `x < p_{a+1}^2` | it names the zone's edges in the machine's own vocabulary; `Q - sqrt(Q)` is an edge of nothing |
| **W78** | d6 L59 | **the stratification**: the smooth cofactor at height `x` is `<= x/p_1`, attained; so in `(Q, 2Q]` admissible = prime or `q`-smooth | PROOF; every dyadic stratum of `(10^4, 10^8]`, 0 violations among 5.4 million | **STANDARD TOOL** - the classical bottom-stratum statement of the combinatorial sieve (Legendre / Meissel-Lehmer `P_2`, `P_3` strata) | identifies the only place in `(Q, 2Q]` a non-trivial open pair can come from |
| **W79** | d6 L60 | **THE FAMILY DECOMPOSITION**: the zone's open pairs are the disjoint union, over `q`-smooth pairs `(s, s')` with `gcd(s,s') | 2`, of `{(sP, s'P') : s'P' - sP = 2}` - a finite union of **linear twin-prime problems**, at most `Psi(Q,q)^2` of them | PROOF; 1,510 families at `q = 5`, `Q = 10^4`; 0 forbidden gcds among 5.4 million pairs | **NEW as far as searched.**  Each individual family is a classical open binary problem (Hardy-Littlewood 1923 Conjecture B, Bateman-Horn, Schinzel-Sierpinski), but no published statement of the twin sieve on `(Q, Q^2]` as this disjoint union was located.  Honest caveat: it is elementary once W76 is in hand, and a referee will call it bookkeeping unless something is *done* with the family structure.  Accuracy note: "polylog in `Q`" is right for **fixed** `q` (`Psi(Q,q) ~ (log Q)^{pi(q)}`), not uniformly in `q` | **the strongest of the zone laws for the target**: it converts "does the window contain a twin" into "does *at least one* of finitely many binary problems have a solution in a bounded range" - a disjunction, strictly weaker than the twin problem alone.  This is where to push |
| **W80** | d6 L61 | **the exact count** `#{admissible n <= X} = Psi(X,q) + sum_{s q-smooth <= X/p_1} (pi(X/s) - pi(Q))`, and **each family's count is a property of the integers, not of `q`**: raising `q` adds families and changes none; family `(1,1)` contributes `pi_2(Q^2) - pi_2(Q)` at every `q` | PROOF + MEASURED, 13 machines, 0 mismatches | **KNOWN VARIANT** - the counting identity is Legendre/Meissel-Lehmer convolved with the smooth part (equivalently one-level Buchstab), the same shape as Bach-Peralta's semismooth count `sigma(u,v)`.  **The `q`-independence of each family is NEW as stated** | the `q`-independence is what makes the decomposition a reduction rather than a re-parameterisation |
| **W81** | d6 L62 | **THE WALK IN THE ZONE**: `nextadm(x) = min(least q-smooth >= x, min over q-smooth s of s * nextprime(max(Q, ceil(x/s))))`.  The mex over residues becomes a minimum over smooth scalings of the next-prime function | PROOF (from W76) + MEASURED, 11,489,920 positions, 0 mismatches; 600,000 pair walks | **NEW as far as searched**, but elementary - W76 read as an algorithm.  Nearest: the Meissel-Lehmer `phi(x,a)` algorithms enumerate the same `n = sP` objects (primecount); Najman arXiv:1108.3710 and Stormer/Lehmer are the algorithmic analogues on the smooth side.  Honest: it is scan-free only relative to a **nextprime oracle**, so the claim is "reduces the walk to `pi`-evaluations", not "no prime computation" | computational: it lets the zone be probed at `Q` far past any sieve scan - exactly what is needed to test W79's family predictions at large `Q` |
| **W82** | d6 L63 | **ALIGNMENTS ALWAYS OCCUR**: any prime gap in `(Q, 2Q]` free of `q`-smooth numbers is a run of struck pairs, so the zone record is at least the largest such gap - a proved lower bound from **prime gaps alone** | PROOF; 48 machines, 0 exceptions; truth is 3.22 to 24.00 times the floor | mechanism **KNOWN** (lower-bounding a sifted-set gap by a prime gap is textbook: FGKMT, Annals 183 (2016); Ford et al., JAMS 31 (2018)); the **statement for this object is NEW as far as searched**.  Closest published instance of the same move - reasoning about rough survivors *inside* a prime gap rather than about the primes - is Gafni-Tao, *Rough numbers between consecutive primes* (2025) | the honest wall-map entry on the lower side |
| **W83** | d6 L64 | **the record profile is a U**: the record per dyadic stratum falls then rises; the bottom stratum makes the record at 28 of 48 machines | MEASURED, stratum tables at 7 machines, positions at 48 | **NEW** (measured) | tells you which end of the zone a bound must attack |
| **W84** | d6 L65 | **NO UPPER BOUND ON THE ZONE RECORD IS AVAILABLE, AND THE EXACT REASON**: an upper bound needs a lower bound on the density of open pairs in a short interval above `Q`, which is a linear twin-prime problem.  Bounded below by ordinary prime gaps, unbounded above | **ROOT** - the conjecture in disguise, and the record says so exactly | **KNOWN as an obstruction** - it is the parity problem (Selberg 1949; Tao, *Open question: the parity problem in sieve theory*, 2007), and a **strictly stronger** statement than the project's target is already a published conjecture: **Ziller-Morack arXiv:1706.00317 Theorem 4.1 / Conjecture 6**, `h_2(n) < p_n^2 - p_n` for `n >= 3`, which they prove implies both Goldbach and the infinitude of prime pairs at every even difference, verified to `p = 73`.  **Correction 2026-09-06**: Conjecture 6 is the *adversarial* window statement - `h_2` frees the two residue classes, the project's `F` fixes them at `D = 2` - so it **implies** the project's window statement and is not the same statement (verdict 1).  **CORRECTION from the sweep**: "exactly as hard as bounding the gaps of the family `(1,1)`" over-states it - an upper bound needs an open pair from *any one* of W79's families, i.e. a **finite disjunction** of binary prime problems, which is weaker than the twin problem itself.  Still parity-blocked, so the reading survives; the sharp phrasing does not | this is the root.  The useful content is the correction: W79 + W84 say the target is a disjunction, not the twin problem |
| **W85** | d6 L66 | **W4 and W24 survive onto the range**: no distance-4 gap in the zone (0 in 49,433,381 gaps); the distance-3 / distance-5 counts agree to 0.02% exactly when 7 is not a gear | PROOF (W4 inherited, the argument is local) + MEASURED (the near-equality) | **KNOWN VARIANT** - inherits W4 (NEW) and W24 (KNOWN VARIANT of Brown / Holt-Rudd) | the wheel laws that are local survive the loss of periodicity - a small but real transfer |

---

# 2. The exhaust

`docs/proofs/23-stack-and-exhaust.md`.  Kernel: `proofs/MachineStack.lean`, 48 declarations,
zero sorries, no `native_decide`.

| # | source | statement | status | prior art (2026-09-06) | use toward the root |
|---|---|---|---|---|---|
| **X1** | S1(a) | a gear `g` exceeding a window length `P` has at most one multiple in the window | KERNEL `card_dvd_window_le_one` | **STANDARD TOOL** - division.  The one-strike-per-copy fact is Holt-Rudd arXiv:1408.6002 Lemma 2.2; the segmented-sieve version ("a prime larger than the segment marks at most one bit") is standard implementation folklore (Sorenson arXiv:1712.09130) | bookkeeping |
| **X2** | S1(b) | hence it strikes at most **two** pair positions - one per tooth | KERNEL `card_strikes_window_le_two` | **STANDARD TOOL** - X1 plus the definition of the tooth pair | bounds what an unmodelled gear can do to a certified window |
| **X3** | S1(c) | **stride containment**: a gear two tiers up strikes at most two pair positions of any window of the lower tier's period | KERNEL `spans_two_below`, `stride_containment` | **STANDARD TOOL**; the *tier indexing* is a relabelling that is new (Holt's stack advances one prime at a time, so his one-strike-per-copy sits one level up; the tier construction buys the two-floor gap) | an upper bound on damage, not a lower bound on survivors |
| **X4** | S1(d) | over that same window the lower tier's pattern repeats **in full** | KERNEL `strikes_add_period`, `isOpen_add_period`, `tier_pattern_repeats` | **STANDARD TOOL** - periodicity | the visitor from two floors up sees the complete pattern and touches two cells |
| **X5** | S2 | **non-containment**: `g < prod G` for `|G| >= 2`, all elements `>= 2`; so no gear spans its own tier, and a tier-`k+2` gear spans tier `k+1` only if it equals that period - impossible.  **Room appears exactly two floors up and not before** | KERNEL `lt_prod_of_two_le`, `not_spans_self`, `not_spans_below` | **NEW as a stated observation, mathematically trivial.**  No published statement that a sieve hierarchy has a containment threshold at exactly two levels was located; the nearest is the wheel-sieve hierarchy `W_k` with period `Pi_k` (Pritchard 1981/82), where `p_{k+1} < Pi_k` is used constantly and never remarked on as a threshold.  **Do not present it as a theorem** | the sharpness half of X3: the two-floor gap cannot be improved to one |
| **X6** | S3 | **THE EXHAUST CAP**: for `C < n <= C^2` and any divisor `p > C`, either `n = p` (home strike) or `n` has a prime factor `<= C` (echo) - **no primality of `p` is used**.  Hence on `(C, C^2]` a pair the primes `<= C` leave open **is** a twin prime | KERNEL `exhaust_home_or_echo`, `openNum_iff_prime`, `open_iff_twin`, `stack_open_iff_twin`, `wheels_open_iff_twin` | **KNOWN** - the sieve of Eratosthenes to the square root; Legendre's formula (~1808); Pomerance, *The Sieve of Eratosthenes and Rough Numbers*.  The claimed novelty, "which machine in the stack can reach which", is **KNOWN VARIANT** of **Fred B. Holt, *Surviving Eratosthenes sieve I: quadratic density and Legendre's conjecture*, arXiv:2603.25915 (March 2026)**, whose *interval of survival* `[p_k^2, p_{k+1}^2]` is this dichotomy as a stack-of-stages object.  **Delta**: tier indexing rather than consecutive primes, plus the explicit "nothing higher can help *or* hurt" clause, which was not located anywhere.  Read Holt 2026 before writing this up | **handle with care.**  The cap is real, but it is also exactly the shape of the parity obstruction: "capped to the motor plus one machine" restates *why* the classical route stalls.  Any claim that the cap *helps* must say what it does that a sieve weight does not |
| **X7** | S4(a) | the zone law: for `0 < n <= Q` an admissible `n` is `q`-smooth | KERNEL `smooth_zone_num`, `smooth_zone`, `wheels_smooth_zone` | **KNOWN** - identical to **W55**; Stormer 1897 / Lehmer 1964 | see W55-W57 |
| **X8** | S4(b) | the quiet zone: for `0 < n <= Q^2` an admissible `n` is `q`-smooth times at most one prime above `Q` | KERNEL `quiet_zone` | **KNOWN** - identical to **W76**; Legendre / Meissel-Lehmer `phi(x,a) = pi(x) - a + 1` for `x < p_{a+1}^2` | see W76-W82 |

**Also in `docs/proofs/23`, and not a numbered fact**: the cut recursion `cut_k =` the period of
tier `k` `=` the lower edge of tier `k + 2`.  No reference was located for this two-step recursion
as a named sequence (checked OEIS A002110, A018239, A034386, and Euclid-Mullin including Booker
arXiv:1605.08929, which is a different recursion): **NEW as an object, STANDARD TOOL for every
ingredient**, and organisational only.  One correction from the sweep: `q <= q#` is trivial
(`q` divides `q#`); **Bertrand is what is actually needed for tier non-emptiness**, and the file
should say so.  The degeneration at `q = 3` (cuts `3, 6, 5, 1, ...`) was independently confirmed.

---

# 3. Verdict counts and the searches

## Counts

One primary verdict per row.  Where a row carries a split reading ("KNOWN VARIANT for the
phenomenon, NEW for the mechanism"), it is counted under the first, and the split is stated in
the cell.

| verdict | wheels (W1-W85) | exhaust (X1-X8) | total |
|---|---|---|---|
| **KNOWN** | 11 | 3 | **14** |
| **KNOWN VARIANT** | 18 | 0 | **18** |
| **NEW** | 37 | 1 | **38** |
| **STANDARD TOOL** | 18 | 4 | **22** |
| **REFUTED** (W9) | 1 | 0 | **1** |
| total rows | 85 | 8 | **93** |

By document: d1 (W1-W21) 3 KNOWN / 5 KNOWN VARIANT / 6 NEW / 6 STANDARD / 1 REFUTED;
d2 (W22-W38) 1 / 7 / 6 / 3; d3 (W39-W54) 2 / 1 / 9 / 4; d4 (W55-W65) 2 / 2 / 5 / 2;
d5 (W66-W75) 0 / 1 / 7 / 2; d6 (W76-W85) 3 / 2 / 4 / 1.

W84 is counted under KNOWN (the obstruction is the parity problem and the target statement is
Ziller-Morack's published conjecture) and is additionally flagged **ROOT**.

## The five verdicts that matter most

1. **A stronger, adversarial window statement is in print; the project's is its real-teeth
   specialisation.**  *(Corrected 2026-09-06 by computation; see
   `research/harvest/r1/jacobsthal_check.md`.  The first version of this verdict said
   `F_top + 1` **is** `h_2` and Conjecture 6 **is** the project's window statement.  Both
   identifications are false from the third primorial on.)*

   Ziller-Morack's `j_2(n)` (arXiv:1706.00317 Def. 2.1-2.2) is the least `m` such that **every**
   paired progression `<a,b>_m` with `2 | b - a` carries a coprime pair.  The even difference
   `D = b - a` is quantified **over**: by CRT, `a` and `D` give **any** two residues mod each
   prime, independently.  So `h_2` at primorials is the **free-residue two-class covering
   record** - `A288815 = 6 A072753 + 6`, with A072753 (Resta) the largest `m` coverable by two
   arbitrary classes per prime - which is the adversary of `the_wall.md` face 5a and of
   `docs/proofs/20`, **not** the real teeth.  The project's `F(M)` is the single instance
   `D = 2` (teeth `0, -2`; separation `3^{-1} (mod g)` in columns).  Verified exhaustively:
   A072753 = 2, 4, 10, 24, 31 reproduced at `{5}..{5..17}`, and `F(M) - 1` = 1, 4, 6, 10, 17 on
   the same sets - **equal at `{5,7}` only, strictly below at `{5}` and at every set from
   `{5,7,11}` on** (6 against 10; 87 against 117 at `{5..37}`).  The domination
   `F(M) - 1 <= A072753` is a theorem: `D = 2` is one competitor in the maximum.

   **Conjecture 6, `h_2(n) < p_n^2 - p_n`, is therefore the adversarial window statement and is
   strictly stronger than the project's**: in column units it gives
   `F(M) <= A072753 + 1 < (p_n^2 - p_n)/6 < (p_{n+1}^2 - 1)/6`, the project's window, so it
   **implies** the project's statement (and, by their Theorem 4.1, Goldbach and prime pairs at
   every even difference).  The converse fails - the project's statement constrains one even
   difference out of the family `h_2` maximises over.  Verified to `p_21 = 73`.

   **W72's "free external check" does not exist as written.**  `6 F(M)` must be compared against
   the **real** twin-candidate gap at `p_n#`, not against Table 1: by direct sieve of
   `{k : gcd(k(k+2), p_n#) = 1}` the max cyclic gap is **12, 30, 42, 66, 108, 150, 204** at
   `n = 3..9`, equal to `6 F(M)` seven times out of seven, against A288815's 18, 30, 66, 150,
   192, 258, 366.  A072753 validates the **free** record instead, and the project does compute
   that object - `docs/novel/jk-growth-discriminator.md`'s `j_k` engine reproduces nine
   published A288815 values (`z = 2..31`); `research/harvest/r1/` adds an independent check of
   the first five.

   **Prior art for `docs/proofs/20`, recorded here.**  Doc 20's `A(K)` frees the **primes** and
   keeps `D = 2`; Conjecture 6 keeps the initial segment and frees `D`.  They strengthen the
   project's window statement along different axes and **neither implies the other**, so doc
   20's Theorem A (`K = 1..10`) is *not* a partial result toward Conjecture 6 and must not be
   filed as one; at `K = 1..10` Conjecture 6's adversary is the larger at every `K` but `K = 2`
   (117 against 87 at `K = 10`) and its window the smaller (222 columns against 280).  What doc
   20 **should** cite is **A072753 itself** as the published table, to 19 gears, of the record
   of the free-residue adversary it compares itself against.
2. **W22-W24, the census, are the two-class version of a 2024 paper.**  Steven Brown,
   arXiv:2311.06873 / *Notes on Number Theory and Discrete Mathematics* **30**(1) (2024) 81-99,
   has the same inclusion-exclusion-over-CRT-products shape, the same threshold universality
   (Lemma 3.1), and even the same kind of coincidence (`K(2,P) = K(4,P)`, eqs 16-17) - all for
   **one** class per prime, and for a *different* reason than ours.  KNOWN VARIANT; cite him.
   Holt is **not** the blocker: his machinery is recursive with only `N_2 = prod(q-2)` closed.
3. **W29 (the collision law / zero waste) is the strongest novelty in the wheels.**  The
   covering-systems literature works in a different regime entirely - infinite covers of `Z`,
   distinct moduli, reciprocal sums (Mirsky-Newman, Znam, Hough Ann. of Math. 181 (2015),
   Balister et al.) - and would not have found the waste parity of a finite two-class interval
   cover.  With W17, W44 and W71 it is the cluster where the new mathematics is.  **W52 is not
   independent of it** and must not be reported as separate evidence.
4. **W64's constant is Lehmer's.**  The measured `F_range = Q - 160 - u` at `q = 5` has
   `160 = s(5)`, the largest `{2,3,5}`-smooth number whose successor-plus-two is also smooth
   (`160, 162`) - the last element of the Stormer-Lehmer difference-2 list (Lehmer, Illinois J.
   Math. **8** (1964) 57-69; OEIS A002072).  "No saturation" is not a phenomenon; it is W57's
   bound running at equality with a Diophantine constant, predictable in advance for every `q`
   from published tables (Luca-Najman, Math. Comp. **80** (2011)).  The register entry is demoted
   accordingly.
5. **W79 + W84 restate the target as a disjunction, and that is the one place to push.**  The
   zone's open pairs are a finite union of linear twin-prime problems (W79, NEW), so an upper
   bound on the zone record needs a solution to *any one* of them, not to the twin problem itself
   - a strictly weaker requirement than `top_machine_6.md` claims.  It remains parity-blocked
   (Selberg 1949; Tao 2007), and FKMPT arXiv:1802.07604 **Remark 7** already names the
   `I_p = {0,2}` sieving system, so any asymptotic phrasing lands inside their theorem.  A live
   neighbour worth reading: **van Doorn & Tang, *Consecutive integers free of certain prime
   factors*, arXiv:2606.19863 (June 2026)**, settling Erdos Problem #451 on the same ambient
   object with the complementary extremal question.

## Naming corrections carried out of this pass

- The ambient set of W76-W82 is **"integers free of prime divisors from an interval"**
  (Weingartner, Acta Arith. 98 (2001) / 104 (2002)) or "integers with no prime factor in `(y,z]`"
  (Tenenbaum 1986).  It is **not** "semismooth": Bach-Peralta, Math. Comp. 65 (1996), bound the
  exceptional prime *above*, the opposite inequality.  Using their word will read as an error.
- **CORRECTED 2026-09-06.**  `F_top` is **not** the paired Jacobsthal function minus one, and
  the name must **not** be adopted.  `h_2 = j_2(p_n#)` maximises over the even difference `D`,
  i.e. over free residue classes (`A288815 = 6 A072753 + 6`); `F_top` is the `D = 2` instance.
  They agree at `{5,7}` and nowhere else beyond it.  Call `F_top` the **real-teeth record** (or
  the `D = 2` paired Jacobsthal value) and `h_2` the **free-class record**; `A072753` is the
  published table of the latter, to 19 gears.  See `research/harvest/r1/jacobsthal_check.md`.
- The counts `prod(g-2)`, `prod(g-3)`, `prod(g-4)`, `prod(g-2-L)` are **Schemmel totients**
  (Schemmel 1869) / Hardy-Littlewood local factors.  Never claimed.

## Searches run (2026-09-06)

Five parallel sweeps, roughly 110 distinct web searches and 40 full-text retrievals.

- **Jacobsthal group** - Ziller-Morack paired progressions and `j_2`/`h_2`; Jacobsthal `g`/`h`
  bounds (Kanold, Stevens, Iwaniec 1978, Erdos 1962, Hajdu-Saradha, Costello-Watts, Hagedorn);
  covering systems (Jordan 1967, Filaseta et al.); restricted coverings; admissible and narrow
  `k`-tuples; Holt-Rudd cycle of gaps; long gaps in sieved sets; Schemmel totient; wheel sieve /
  next coprime to a primorial; "mex" in sieve contexts; jumping champions; twin primes between
  prime squares; interval covering by distance-2 pairs; OEIS A288815, A048669, A048670.
- **Covering/tiling group** - exact and disjoint covering systems (Mirsky-Newman, Znam,
  Berger-Felzenbaum-Fraenkel); Hough's minimum modulus; Balister et al. distortion method;
  Coven-Meyerowitz; de Bruijn; ILP/SAT covering computations (INTEGERS 24A, 2024); `j(2n) = 2j(n)`;
  Erdos-Rankin; Liouville sign patterns; the parity problem.
- **Census/spectra group** - Holt (1408.6002, 1510.00743, 2502.20470, 2603.25915) read in full;
  Hooley; Montgomery-Vaughan reduced residues; Hausman-Shapiro; Aryan; Hardy-Littlewood singular
  series for `{0,2,d,d+2}`; closed-form gap counts over primorials (which located **Brown**);
  Ziller arXiv:2007.01808; Ramanujan sums and the DFT of even functions mod `r`; Grob-Schmitt.
- **Smooth-number group** - Stormer/Lehmer and the difference-2 list; A002071/A002072;
  Luca-Najman; Conrey-Holmstrom-McLaughlin; twin-smooth cryptography (B-SIDH/SQIsign, Pell,
  lattice); Tijdeman; Heath-Brown arXiv:1808.02947; Najman arXiv:1108.3710; Weingartner;
  Tenenbaum; Bach-Peralta; Meissel-Lehmer `phi(x,a)`; Buchstab; FKMPT; van Doorn-Tang;
  Ford (divisor in an interval); Gafni-Tao.
- **Stack group** - Pritchard's wheel sieve; segmented sieves (Sorenson); Legendre's formula;
  Pomerance on rough numbers; primorial sequences and Euclid-Mullin; Kourbatov on maximal gaps
  and first occurrences; covering systems with restricted divisibility.

**Coverage limits, stated honestly.**  Three sources could not be read: Hagedorn's 2009 paper
(403 from tcnj.edu - its "large primes enter by counting" lemma is very likely stated there in
some form, so W31/W62's principle must not be claimed as new), Eckford Cohen's *An analogue of a
result of Jacobsthal*, and the OEIS pages A288815/A048669/A048670 (403; A288815's comments may
carry formulas not seen here).  **That last limit is exactly what broke verdict 1**: A288815's
own formula `a(n) = 6 A072753(n) + 6`, and A072753's Resta comment naming the free-residue
two-class covering record, were not in front of the sweep.  Both were read on 2026-09-06 (see
verdict 1); the project's own first-hand reading of the same two pages, dated 2026-08-29, was
already on the record in `docs/novel/jk-growth-discriminator.md` section 6 and was not consulted.
Several arXiv PDFs (1706.00317, 1706.03668, 2107.06950,
1909.02205) would not text-extract and were read via ar5iv HTML, reliable on definitions and
theorem statements but possibly missing remarks buried in proofs.

---

# 4. Where this goes

- `docs/novel/README.md` carries an index entry per NEW and KNOWN VARIANT cluster, dated
  2026-09-06, pointing back here.
- `docs/proofs/22-top-machine-laws.md` and `docs/proofs/23-stack-and-exhaust.md` have their
  "Prior art, and what is new" sections updated with the verdicts; `22`'s "prior art not checked
  for the machine as an object" is discharged.
- `research/proof/objects_ledger.md` items **O-W7** and **O-X5** are answered by this file.
- The register is the citation surface from here on: cite **W-numbers**, not `L`-numbers.

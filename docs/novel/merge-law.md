# merge-law - F(M+q') is computable from the old machine alone

## 1. WHAT IT IS

Plain language. The machine M(y) is the set of "gears" q = 5, 7, ..., y (odd primes), acting on
slot space: slot k stands for the pair (6k-1, 6k+1), and gear q blocks slot k exactly when
k = +-6^{-1} mod q (its two "teeth"). The unblocked slots ("openings") repeat with period
P = product of the gears, and F(M) is the largest gap between consecutive openings - the paired /
twin Jacobsthal function in the 6k+-1 frame. Adding the next gear q' multiplies the period by q',
so computing F(M+q') by rebuilding the pattern costs q' times the old period. The merge law says
the rebuild is unnecessary: the new record gap can be read off the OLD machine's gap word plus q'
alone. Adding a gear only deletes openings, deletions merge the gaps either side, and there is an
exact, checkable criterion for which runs of consecutive old openings get deleted together
somewhere in the new period.

Precise form. Let M have period P, exposed (opening) set E, and gap word g_1, g_2, ... (the cyclic
sequence of differences of consecutive openings). Let q' be a prime coprime to P with teeth
{c, q'-c} mod q', c = 6^{-1} mod q'. Then E' = { x in [0, Pq') : x mod P in E, x mod q' not a
tooth }, and:

(a) **Lap structure.** Walking x upward walks E around q' times. Lap l covers [lP, (l+1)P) and is
    E with two residue classes mod q' deleted (the teeth as seen from lap l); the deleted pair
    shifts by -P mod q' per lap. Since gcd(P, q') = 1 the shift generates, so every phase of gear
    q' occurs in exactly one lap, and **each opening of M is deleted in exactly 2 of the q' laps**
    (once per tooth).

(b) **Chain condition.** Deleting k consecutive openings merges the k+1 gaps they separate, so
    every new gap is a sum of consecutive old gaps. A run of k consecutive openings is deleted
    together in some lap iff, walking along it, each successive opening lands on a tooth: the
    partial sums of the k-1 interior gaps all lie in {0, +-2c} mod q' (0 mod q' = both kills on
    the same tooth, a "padded" link; +-2c mod q' = opposite teeth, a "literal" link). By (a)
    every run meeting the condition is realised in some lap, so the condition is exact, not
    merely necessary.

(c) **The law.** With F2(M) = the largest sum of two adjacent old gaps and, for a word w (a
    chain's interior-gap sequence), span(w) = its sum and FS_max(w; M) = the largest sum of the
    two flanking gaps over occurrences of w in the old gap word:

        F(M+q') = max( F2(M),  max over compatible w of [ span(w) + FS_max(w; M) ] )

    (constructor.md R21 - "an identity, not a ceiling"; the word list and compatibility depend on
    q' mod 210 alone, only occurrences and flanks come from M). Everything on the right is
    computed from the old machine's gap word and q' - the new period is never constructed, a
    factor ~q' saving. The k = 1 case (no interior gaps, always available) gives
    F(M+q') >= F2(M) unconditionally.

The same statements hold verbatim in the adjacent frame (gear 3 included, teeth {o, o+1} mod q,
all lengths x3); there the tooth-difference set is {0, +-1} mod q'.

## 2. WHY IT MIGHT BE NOVEL

The classical shadow is the computation of Jacobsthal's function at primorials. Every published
computation - Hagedorn (2009), Costello-Watts (2015), Ziller-Morack (2016, 2017) - searches each
modulus from scratch (sequential array-filling, permutation search, ILP, greedy variants); none
uses the gap structure of the previous primorial. The closest structural relative, Holt & Rudd's
cycle-of-gaps recursion (section 6), constructs the ENTIRE new cycle (concatenate p copies, close
gaps at p * old generators) and proves each closure occurs exactly once - but it stays in the
ordinary one-residue-class sieve, it materialises the full new cycle rather than answering an
extremal question from the old word, and its authors state no formula for the maximal gap. The
two genuinely non-obvious pieces here are (i) the exactness of the chain condition -
realisability of a deletion run is decided by partial sums of the old gap word mod q', with no
reference to the new period - and (ii) that this turns the extremal statistic F into a recursion
on the gear set: the paired Jacobsthal value of the next machine from the old machine's gap word
alone. It is not a restatement of "sieving is periodic + CRT"; CRT gives the lap structure, but
the extraction of F without construction is the content.

## 3. PROOF

Status: **PROVED (elementary paper proof) + SCRIPT-VERIFIED (finite, exact); the law itself is
not kernel-checked** (kernel-checked fragments exist, see below).

KERNEL-CHECKED since round 21 (formalist): the law's BOUND form - R39's two-machine
inequality - is `MergeLaw.newgap_le` / `newgap_le_max` / `D_of_qualmax`
(proofs/MergeLaw.lean): every gap of M+q' is a window sum of old gaps whose interiors
are killed (merged window), hence residue-qualifying (`interior_gap_mod`,
`floor_of_mod`), hence `F(M+q') <= max(F2, max_j Q_j)` - merge law + residue
necessity, abstract in the machine.  Instantiated end to end at 19->23:
`Machine23.g23_le` (every machine-23 gap <= 47) and `Machine23.D_at_19_23`
((D) at alpha=3, no hypotheses), on machine 19's kernel-fed ladder
(proofs/Machine19Q.lean).  The exact-computation form of the law (the full
histogram transform) remains paper+script only.

ROUND 22 (formalist): the bound form is now a LADDER of FOUR CONSECUTIVE
STEPS, all hypothesis-free (proofs/Ladder.lean, on new period scans in
proofs/Machine11.lean, Machine13Q.lean, Machine17Q.lean):

    11->13  g13 <= 20 = F(11)+13   `Ladder.D_at_11_13`   (criterion 20, TIGHT)
    13->17  g17 <= 28 = F(13)+17   `Ladder.D_at_13_17`   (criterion 26)
    17->19  g19 <= 37 = F(17)+19   `Ladder.D_at_17_19`   (criterion 35)
    19->23  g23 <= 48 = F(19)+23   `Machine23.D_at_19_23` (criterion 47)

collected as `Ladder.D_ladder`.  The per-step bookkeeping is factored out
once as `MergeLaw.newgap_le_step` (locate both endpoints of a new gap in the
old enumeration, check the interior is killed, telescope), so a rung costs
only the old machine's `F_2` and qualifying-spectrum scan.  Steps above the
scannable range are recorded hypothesis-explicitly: `Ladder.D_at_23_29`
(Mechanic's corrected F_2(23)=39, max_j Q_j(23;10)=60, budget 63) and
`Ladder.D_at_37_41` (F_2(37)=90, max_j Q_j(37;41)=91, budget 129).

HONEST LIMIT recorded with the ladder: the law is ONE-STEP.  It consumes an
F_2 and a qualifying spectrum and produces neither, so rungs cannot be
chained without a fresh scan of each machine in turn - the merge law alone
gives only F_2(23) <= 2*47 = 94 against the 63 the next rung needs.

- Proof of (a), (b): `docs/gear-recursion.md` sections 3-4a (the transform, the deletion-merge
  mechanism, the chain condition and its exactness via "each opening deleted in exactly 2 laps").
  Same argument in the module docstring of `research/gear_recursion.py`.
- Transform verified against direct construction of the new machine for four extensions (gears to
  7 plus 11, to 11 plus 13, to 13 plus 17, to 17 plus 19), matching not only the maximum but the
  **entire gap histogram** (`research/gear_recursion.py`).
- Chain-condition value verified against the transform in 15 cases spanning gear sets to 19 and
  added gears to 31 - exact agreement every time (`docs/gear-recursion.md` section 4a).
- Word-indexed identity (c) verified exactly at all six consecutive steps 11->13 .. 29->31, and
  consistent with the padded winner at 31->37 (`docs/proof-search/constructor.md` R21).
- At scale: chain condition exact at each new size - predicted F(M+q') = 58 (29->31), 88 (31->37),
  90/91 (37->41 probes), 92 (q = 53) (`docs/proof-search/mechanic.md` C10).
- Production reimplementation: `rust3/src/machine.rs` `f_next` (returns F(M+q') plus a witness
  Merge), tests `merge_law_matches_known_ladder` (5 ladder steps) and
  `merge_law_agrees_with_direct_construction` (brute-force cross-check at 4 steps); wired into
  `rust3/src/bin/gapsuite.rs`. 24 gearsuite tests green.
- Kernel-checked fragment: `proofs/Spectrum.lean` - `merged_eq` (a merged run of l interior gaps
  plus two flanks spans exactly l+2 consecutive gaps, i.e. merged length IS a window sum),
  `merged_le_spectrum`, `merged_le_of_shallow`; axiom footprint per the round-19 ledger. These
  formalise "new gap = window sum of old gaps" and F(M+q') <= F_{k_max+1}(M); the lap structure
  and chain-condition exactness are NOT formalised.

## 4. IMPLICATIONS

Inside the project: this is the engine of the round-13+ programme. It reduces the increment
question F(M+q') - F(M) <= alpha*q' (which, by the tolerance theorem, constructor.md R14, closes
the route for any fixed alpha at the measured scale) to a statement about the old gap word mod
q'. It yields the spectrum reduction F(M+q') <= F_{k_max+1}(M), the fuel/literal-cap analysis
(literal chains have at most 6 members, every gear, forever - constructor.md R20), the saturation
theorem (separate entry), the deletion-spacing lemma (separate entry), and F(2,y) ladder
extensions at 1/q' of the naive cost (`f_next` in rust3/gearsuite).

Outside: it gives an incremental algorithm for paired-Jacobsthal values (h_2-type functions,
where published work computes each modulus from scratch), and the identical argument with one
tooth instead of two computes the ordinary Jacobsthal g(p_{k+1}#) from the gap word of p_k# -
which, as far as searched, no published algorithm does either.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Twin prime conjecture, via the route F_k(y) <= (y^2 - y)/6: the merge law is the tool for the
  one open link (a proved bound on the aggregate increment constant C).
- Ziller-Morack Conjecture 6 (paired-progression bound, g_2-type <= p_n^2 - p_n): the machine's F
  IS the paired Jacobsthal object; the merge law is a new computational route to its values.
- Jacobsthal-function computation generally (Hagedorn's h(n), OEIS A048670): the one-tooth
  specialisation is an incremental algorithm the literature lacks.
- Polignac gap questions: the same machinery runs for any even difference class (see the
  project's polignac-cap and twin-percentile entries).

## 6. PRIOR-ART CHECK

Checked 2026-08-23. Searches actually run (Claude Code WebSearch; full-text reads via ar5iv):

1. "Jacobsthal function g(n) primorial maximal gap consecutive coprime integers Hagedorn
   computation" - found Hagedorn, Ziller-Morack, OEIS A048669/A048670, Ziller arXiv:2007.01808.
2. "Ziller Morack 'Jacobsthal function' computation primorial h(k) algorithm" - found
   arXiv:1611.03310 and arXiv:1706.03668 (paired progressions, values to prime 73).
3. "'Jacobsthal function' recursion 'adding a prime' incremental computation g(P*q) from g(P)" -
   no incremental method found; the literature computes each modulus from scratch.
4. "'reduced residues' primorial 'maximal gap' structure 'adjacent gaps' merge OR merging OR
   incremental prime added" - surfaced the Holt-Rudd trail (gap copies + merging language).
5. "Holt Rudd 'cycle of gaps' Eratosthenes sieve primorial recursion copies closures" - found
   arXiv:1408.6002, 1503.00231, 1510.00743, 1402.1970.
6. Full-text read of arXiv:1611.03310 (ar5iv): algorithms reviewed are BSA, BPA, RPA, ILP, DSA,
   CRPDSA, GPA - all from-scratch searches; **no incremental computation from previous primorial
   data, no use of gap structure**.
7. Full-text read of arXiv:1408.6002 (ar5iv): Lemma 2.1 (recursion R1-R3: next prime, concatenate
   p_{k+1} copies of G(p_k#), close adjacent gaps at positions p_{k+1} * G(p_k#)); Theorem 2.3
   (each possible closure occurs exactly once, by CRT). **No formula for the maximal gap; no way
   to get the new maximum without constructing the new cycle; ordinary one-class sieve only.**
8. Full-text read of arXiv:2007.01808 (ar5iv): which even values occur as gaps at each primorial;
   no structural recursion, no maximal-gap formula (closest: Conjecture 4.1, h(k-1) <= N_min(k)).
9. "Pritchard wheel sieve 'gap structure' wheel W_k recursion rolling wheel next prime
   properties" - Pritchard's wheel sieve uses the wheel as an enumeration device
   (O(N/log log N) sieving); no extremal-gap recursion.
10. "mathoverflow OR mathstackexchange Jacobsthal function adding a prime modulus maximal gap
    coprime residues recursion" - nothing beyond the above.
11. "Costello Watts Jacobsthal function bound Iwaniec Montgomery Vaughan distribution reduced
    residues gaps" - bounds literature only, no recursion.
12. "twin prime sieve '6k-1' '6k+1' pairs wheel maximal gap 'Jacobsthal' paired two residue
    classes per prime" - only from-scratch twin sieves (e.g. segmented sieve of Zakiya); no
    paired-frame gap recursion.

Nearest published results:

- **F. B. Holt & H. Rudd**, "Eratosthenes sieve and the gaps between primes", arXiv:1408.6002
  (2014); "Constellations of gaps in Eratosthenes sieve", arXiv:1503.00231; "Combinatorics of the
  gaps between primes", arXiv:1510.00743 (2015); "On Polignac's conjecture", arXiv:1402.1970.
  The cycle-of-gaps recursion G(p_k#) -> G(p_{k+1}#) is the exact one-residue-class analogue of
  the lap/merge transform (concatenated copies, closures = gap merges, each closure exactly once
  by CRT). They use it for constellation POPULATION dynamics, not for the maximal gap.
- **T. R. Hagedorn**, "Computation of Jacobsthal's function h(n) for n < 50", Math. Comp. 78
  (2009), 1073-1087; **F. Costello & P. Watts**, "A computational upper bound on Jacobsthal's
  function", Math. Comp. 84 (2015), arXiv:1208.5342; **M. Ziller & J. F. Morack**, "Algorithmic
  concepts for the computation of Jacobsthal's function", arXiv:1611.03310 (2016), and "A short
  note on the computation of the generalised Jacobsthal function for paired progressions",
  arXiv:1706.03668 (2017, values to prime 73). All from-scratch per modulus.
- Background bounds, no recursion: E. Jacobsthal, Norske Vid. Selsk. Forh. Trondheim 33 (1960);
  P. Erdos, Math. Scand. 10 (1962), 163-170; H. Iwaniec, "On the problem of Jacobsthal",
  Demonstratio Math. 11 (1978), 225-231; H. L. Montgomery & R. C. Vaughan, "On the distribution
  of reduced residues", Ann. of Math. 123 (1986), 311-333.

**Verdict: PARTIAL OVERLAP** (Holt & Rudd, cited above). The delta: (i) their recursion is the
ordinary one-tooth sieve - the paired frame with two teeth per gear, each opening deleted in
exactly 2 laps, and the literal/padded link structure appear neither in their work nor anywhere
else found; (ii) they construct the whole new cycle, whereas the chain condition here answers the
extremal question from the old gap word alone, factor ~q' cheaper, and they state no maximal-gap
formula at all; (iii) the word-indexed identity F(M+q') = max(F2, span + FS_max) has no published
counterpart found in either frame. The transform itself (laps + closures + CRT) must be called a
known idea in the one-class frame; the computation of F(M+q') without constructing the new period
is NOVEL AS FAR AS SEARCHED.

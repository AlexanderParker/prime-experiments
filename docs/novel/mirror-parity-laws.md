# Mirror parity laws for the twin machine

Lateral, round 25 (2026-08-29). Script: `research/mirror_cells.py` parts A, B
(log `research/data/mirror_cells.log`); the m11..m29 cell/parity columns are
also emitted by `research/spiral29.py`.

ROUND-26 ADDENDUM (Formalist): **THE ARITHMETIC HALVES ARE KERNEL-CHECKED**
(`proofs/Mirror.lean`, build green at 1426 jobs; axiom footprint
`[propext, Quot.sound]` - not even `Classical.choice`).

* `Mirror.mirror_gear` - M0 for ONE GEAR, any period: with `q | P` and
  `1 <= k < P`, `q | lo (P - k) <-> q | hi k` and `q | hi (P - k) <-> q | lo k`.
  The mirror EXCHANGES the slot's two members, and blocking is symmetric in
  them, so the opening set is closed under the involution. Instantiated at
  `Mirror.mirror_exposed11` (3 gears) and `Mirror.mirror_exposed29` (8 gears).
* `Mirror.antipode_open` - **`g_1* = 1`, in five lines and with no residues**:
  if `2s = P + 1` and `q | P`, `q >= 5`, then `6s = 3P + 3`, so the antipodal
  slot's members are `3P + 2` and `3P + 4` and gear `q` would have to divide
  `2` or `4`. Instantiated as `antipode_exposed11 : Exposed11 193` and
  `antipode_exposed29 : Exposed29 539141103` - both by ARITHMETIC, no scan and
  no `decide`, at a machine whose period no kernel will ever see.
* `Mirror.self_mirror_unique` - at most ONE self-mirror window per depth:
  `2 t_1 + j = 2 t_2 + j = 0 (mod N)` with `t_1, t_2 < N` and `N` odd forces
  `t_1 = t_2`. With `Mirror.periods_odd` (`N` odd at m11..m31) this is the
  half of the parity law the live route consumes - "fewer than two" proves
  "none".

NOT KERNEL-CHECKED, and it is the interesting half: "every count is EVEN
except the exceptional one" needs the counting step (a fixed-point-free
involution on a `Finset` has even cardinality), which `Mirror.lean` does not
build. What is proved there is the involution and the uniqueness of its fixed
point, not the parity of the counts.

## 1. WHAT IT IS

The machine's opening set is exactly symmetric about slot 0. That is obvious.
What is not obvious is that this single symmetry pins the **parity** of every
window count and every gap-word count in the machine, completely and for ever:
in each family, all counts are even except exactly one, and the exceptional one
is named in advance.

Definitions (project frame). Slot `k` = the pair `(6k-1, 6k+1)`. Machine `M_y`
has gears the primes `5..y`; gear `q` blocks slot `k` iff `6k = -+1 (mod q)`.
`P = prod q`, `N = prod (q-2)` = number of openings per period, `o_0 = 0 <
o_1 < ... < o_{N-1}` the openings, `W_j(g)` = number of depth-`j` windows
(`j` consecutive gaps) of total length `g`, `F` = maximal gap, `k_1 = o_1` =
the first opening after 0.

**M0 (the involution).** `k` is blocked iff some gear divides `6k-1` or `6k+1`;
that condition is invariant under `k -> -k`, so the opening set is exactly
closed under `k -> -k`. `k = 0` is always an opening, and `P` is odd, so `0` is
the involution's only fixed slot. On indices the map is `o_t -> o_{N-t}`.

**M1 (window parity law).** Mirror sends the depth-`j` window starting at
opening index `t` to the one starting at `N - t - j`. Because `N = prod (q-2)`
is odd, `2t = -j (mod N)` has exactly one solution, so **each depth has exactly
one self-mirror window**. Hence, for every machine and every depth `j`,

>   `W_j(g)` is EVEN for every `g` except the single length `g_j*` of the
>   window starting at `t_j = -j/2 (mod N)`, where it is ODD.

At `j = 1` that window is the gap straddling the antipode of 0, and it always
has `g_1* = 1` at the machines checked. Corollary: `W_1(F)` is EVEN unless `F`
is the antipodal gap - **the maximal gap always occurs an even number of
times** at every machine tested.

**M2 (gap-word reversal law).** Mirror sends the gap word `(g_1,...,g_j)` read
at openings `(k_0,...,k_j)` to the REVERSED word read at `(-k_j,...,-k_0)`. So

>   the depth-`j` gap-word census is EXACTLY reverse-symmetric,
>   `W_j(g_1,...,g_j) = W_j(g_j,...,g_1)`,
>   and every PALINDROMIC word count is EVEN except exactly one word per depth.

At `j = 2` the exceptional word is forced: it is `(k_1, k_1)`, the pair of gaps
flanking slot 0 (equal because the openings around 0 are `-+k_1`). Measured
exceptional palindromes: `(3,3)`, `(5,1,5)`, `(2,3,3,2)` at m11/m13;
`(5,5)`, `(10,1,10)`, `(2,5,5,2)` at m17/m19.

**M3 (the adjacent-gap corollary - the reason this was written).**
Since `k_1 < F` at every machine, the unique self-mirror adjacent pair is never
`(F,F)`. Therefore

>   any adjacent configuration with `g_1 = g_2` - in particular an `(F,F)` pair,
>   the configuration that would realise `F_2 = 2F` - occurs an EVEN number of
>   times in every period.

A "big next to big" event of equal size can never happen exactly once.

## 2. WHY IT MIGHT BE NOVEL

The involution itself is folklore (the Jacobsthal/prime-gap literature routinely
notes that a reduced residue system mod `P` is symmetric). What appears not to
be recorded is the **parity consequence for window and word counts**: that the
full gap-word census of a primorial sieve is an exactly reverse-symmetric
multiset with a *single* odd palindrome per depth, and that the odd class is
located in advance by `t = -j/2 (mod N)`. Standard treatments of `g(P)` /
Jacobsthal's function use the symmetry to halve a search, not to derive parity
invariants of the whole count family.

The corollary M3 is the interesting direction: it converts a *counting* bound
into an *impossibility*. Parity arguments of this shape ("at most one, therefore
none") are common in combinatorics but have not, as far as this project's
records go, been applied to maximal-gap adjacency in a sieve.

Honest shadow: for `j = 1` the statement "each gap length occurs an even number
of times except one" is close to the observation that reduced residues mod `P`
come in `+-` pairs, and a careful reader of Hagedorn or Ziller-Morack might
regard it as immediate. The depth-`j` and word-level versions, and the located
exceptional class, are the parts with content.

## 3. PROOF

SCRIPT-VERIFIED (finite, exact integers) + PROVED (elementary; the proofs above
are complete as written - they are three lines each and use only that `N` and
`P` are odd and that `0` is an opening).

`research/mirror_cells.py`:

- part A asserts, at m11/m13/m17/m19: mirror closure of the opening set; that
  for every `j = 1..12` the odd class of `W_j` is unique; and that it sits at
  the length of the window predicted by `t = -j/2 (mod N)`.
- part B asserts, at the same machines and depths `j = 2,3,4`: exact equality of
  the word census with its reversal (compared as sorted `(code, count)` arrays),
  and that exactly one palindromic word has odd count, equal to `(k_1,k_1)` at
  `j = 2`.
- `research/spiral29.py` re-derives the `j = 1` consequences (`N_2`, `N_3` even;
  see the companion document) independently at m11..m29 including the
  full-period machine 29 (`N = 214,708,725`).

Run: `python research/mirror_cells.py --parts AB --maxy 19` - 9 assertion gates
green, exit 0.

Kernel handoff (proposed, not done): M1 at `j = 1` is a natural Lean target -
"the opening set is closed under negation, 0 is its only fixed point, hence the
gap multiset is a disjoint union of mirror pairs plus one self-paired gap."

## 4. IMPLICATIONS

Inside the project:

- **A parity lever for the two-gap law.** The live route's obligation is a
  statement of the form "`F_2(M) <= F(M) + q'`", whose worst case is two
  adjacent near-maximal gaps. M3 says the symmetric extreme case `(F,F)` cannot
  occur an odd number of times. Any counting/covering argument that caps the
  number of `(F,F)` adjacencies at ONE therefore proves there are NONE. That is
  a strictly cheaper obligation than proving the cap is zero directly.
- **A free consistency check on every census.** M1/M2 are exact integer
  identities, so any full-period census that violates them is wrong. Applying it
  immediately caught a real defect: every full-period `ghist` row in
  `research/data/gap_pair_hist.csv` carries `N-1` gaps, not `N` - the census
  closed the period linearly and dropped the wrap-around gap (size `k_1` =
  3,3,5,5,5,7,7 at m11..m31). Relative error `1e-9`, harmless for densities, but
  it breaks every exact integer identity downstream. `mirror_cells.load_ghist`
  repairs it and asserts the repair.
- It supplies the missing structural reason for round 9's "mirror pairing exact
  at every machine" (item 12) and round 7's "word distributions closest to
  revcomp-symmetric" (item 9b): for GAP words the symmetry is not approximate,
  it is exact, and the small observed asymmetry in the round-7 SIDE-word study
  must therefore come from the L/R labelling, not from position.

Outside: any primorial-sieve gap census inherits the same parity laws.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Jacobsthal's function `g(P)` and its two-gap analogue: M1 says the count of
  maximal gaps in a primorial residue system is even except in the single case
  where the maximal gap straddles the antipode.
- Polignac / admissible-tuple counting: M2 says the ordered-tuple census of a
  primorial sieve is a reverse-symmetric multiset, which halves any exhaustive
  word enumeration exactly (not heuristically).
- The project's own two-gap obligation (R55-R59), via M3.

## 6. PRIOR-ART CHECK

**Not yet checked** (this lane has no web access). Suggested search terms for
the manager: "Jacobsthal function symmetry reduced residues primorial parity";
"gap sequence of reduced residue system palindromic"; "prime gap word census
reversal symmetry"; Hagedorn "Computation of Jacobsthal's function"; Ziller &
Morack tables; Holt (arXiv 2502.20470) - his cycle-of-gaps framework is the
nearest formal neighbour and may already record the reversal symmetry of
`G(p#)` (his gap cycles ARE reverse-symmetric; whether the PARITY corollary and
the located odd class appear is the question). Standing lesson: prior-art checks
expire - recheck before publication.

---

## 7. ROUND-26 EXTENSION: THE LEVER, THE ADDRESS, AND ITS EXACT CEILING

Lateral, round 26 (2026-08-29). Script: `research/mirror_lever2.py`
(log `research/data/mirror_lever2.log`, 52 assertion gates, exit 0).

Round 25 located the exceptional window by its **index** `t_j = -j/2 (mod N)`.
An index is only usable after the period has been enumerated. Round 26 replaces
it by an **address**, and the whole family becomes computable at machines no
scan will ever reach.

### 7.1 The full symmetry group, and why the lever is worth exactly one unit

**THEOREM A.** The affine maps `k -> ck + b` of `Z_P` that carry the opening
set onto itself are exactly

>   `Aff(O) = { k -> ck : c = +-1 (mod q) for every gear q }  =  (Z/2)^m`,

`m` = number of gears; and of its `2^m` elements **only `c = +-1 (mod P)`
preserve adjacency of openings**, i.e. only the identity and the mirror act on
windows at all.

*Proof.* An affine map preserves `O` iff for every gear it carries the tooth
pair `{+-u_q}` onto itself. Adding the two requirements `cu + b = -+u` gives
`2b = 0 (mod q)`; `q` is odd so `b = 0`. Then `cu = +-u` with `u` invertible
gives `c = +-1 (mod q)`. Conversely each such `c` works. The element flipping
the gears in `S` fixes `k` iff `q | k` for `q in S`, i.e. `P/prod_S q` slots -
one when `S` is everything. Adjacency: brute-forced. []

Gated: at `m11` over **ALL 240 units x 385 shifts = 92,400 affine maps**, at
`m13` over all 2,880 units; predicted group, predicted fixed-point counts, and
the adjacency verdict all exact.

**THEOREM A2 (the ceiling without the affine assumption).** Anything acting on
windows preserves the *circular order* of `Z_P`, and the order-preserving
bijections of a cycle are exactly the rotations `k -> k+b`, the reversing ones
exactly the reflections `k -> b-k`. Rotations: `O+b = O` needs `{+-u}+b = {+-u}`
per gear, and adding the two equations gives `2b = 0`, so `b = 0`. Reflections:
`b-u = -u` with `b+u = u` gives `b = 0`, while `b-u = u` with `b+u = -u` gives
`4u = 0 (mod q)`, impossible. So

>   **the full symmetry group of the opening set inside the circle `Z_P` is
>   `{identity, mirror} = Z/2`, exactly** - brute-forced over all `2P` rotations
>   and reflections at m11 and m13.

**This is a ceiling, not a bonus.** The window census carries a `Z/2` action and
nothing larger - `2^m` affine symmetries exist but `2^m - 2` of them destroy
consecutiveness, and outside the affine world there is nothing at all. So the
lever "cap at one, parity gives zero" is worth **exactly one unit** - a factor of
two in a counting argument, never a factor of four. A finer parity must come from
something that is not a symmetry of the opening set at all.

### 7.2 The exceptional window, in slot space

**THEOREM B.** A depth-`j` window with endpoints `x < y = x+g` (taken in
`[0,P)`) is self-mirror iff `x + y = 0 (mod P)`, i.e. iff `2x + g in {P, 2P}`.
So it is **centred on the antipode** (`2x+g = P`, forcing `g` odd) or
**centred on slot 0** (`2x+g = 2P`, forcing `g` even). Counting openings on
each arc:

>   `j` even:  `g_j* = 2 o_{j/2}`            (the window through slot 0)
>   `j` odd :  `g_j* = 2 b_{(j+1)/2} - P`    (the window through the antipode)

where `o_1 < o_2 < ...` are the openings just above `0` and
`b_1 < b_2 < ...` those just above `(P-1)/2`. **COROLLARY: `g_j* = j (mod 2)`,**
so `W_j(g)` is EVEN for every `g` of the wrong parity with no computation at
all - half the entire window spectrum, free.

Both endpoint lists come from sieving a few dozen slots, so `g_j*` is available
**scan-free at every machine**. Table (`j = 1..12`, machines 11..53) in the log;
verified against the exact full-period `W_j` census at m11..m29 for every depth
`j <= 12`: the set of `g` with `W_j(g)` odd is exactly `{g_j*}`, no exceptions.

### 7.3 `g_1* = 1` ALWAYS - the antipodal gap is a theorem, not an observation

Section 1 recorded `g_1* = 1` "at the machines checked". It is universal.
Since `P = 0 (mod q)`, the antipodal slot `s = (P+1)/2` reduces mod every gear
to `inverse(2) = (q+1)/2`. Multiply by 6: `6s = 3(q+1) = 3 (mod q)`, while
`6(+-u_q) = +-1` by the tooth law. So `s` is a tooth iff `3 = +-1 (mod q)`,
i.e. `q | 2` or `q | 4` - impossible for `q >= 5`. (This is the T3 law
`3u = +-(q+1)/2` of `golden-spectral-gap.md` item (b) wearing a different hat.)

>   **THE ANTIPODAL SLOTS `(P+-1)/2` ARE OPENINGS AT EVERY MACHINE, so the
>   antipodal gap has length 1 and `W_1(g)` is EVEN for EVERY `g >= 2`.**

Only the number of gaps of size 1 is odd. In particular the number of **maximal
gaps is even at every machine, unconditionally** - the round-25 statement's side
condition is discharged - so the maximal gap never occurs exactly once.

### 7.4 The fixed-point criterion: parity by one membership test

For a **palindromic** tuple `w` of span `s` the occurrence set is
mirror-invariant, and an occurrence at `k` is self-mirror iff `2k = -s (mod P)`.
`P` is odd, so there is exactly one candidate address.

>   **THEOREM C.** `#occ(w)` is ODD iff `w` occurs at `k_w = -s * inverse(2)
>   (mod P)`, and EVEN otherwise - an `O(#gears)` test.

Specialising to `w = (g,g)`: `k_w = -g`, and the occurrence needs openings at
`-g, 0, g` with nothing between, i.e. `g = k_1`. That rederives round 25's
"the unique odd depth-2 palindrome is `(k_1,k_1)`" in one line and generalises
it to every arity. Gated at m11..m23 against the exact period census: the
criterion predicts the parity of **every** palindromic 2- and 3-tuple count.

### 7.5 The lever has no side condition on the (D) family

The merge law quantifies only over **qualifying** windows - those whose middle
gaps all clear the next gear's tooth floor `a = 2u'(q')`. The exceptional window
sits against slot 0 or the antipode, where the gaps are the machine's shortest
(`k_1 = 3..10`), so it is **never qualifying**: checked at every ladder rung
`11->13 .. 47->53` and every depth `j <= 7`, verdict `n` in all 66 cells.

>   So on the qualifying family the lever is unconditional: **an exact bound
>   "at most one qualifying depth-`j` window exceeds the budget" proves there
>   are none.**

(Reported for the route, not developed here - mandate.)

### 7.6 Reversal, and what not knowing it cost

**THEOREM D.** The mirror sends an occurrence of the gap word `w` at address
`k` to an occurrence of `reverse(w)` at `-(k + span w)`, bijectively. Hence
`#occ(w) = #occ(reverse w)` exactly, at every machine and every arity, and
realisability is reverse-invariant. The same argument covers **merge kill
words** (the old machine's openings and the new gear's teeth are both
negation-symmetric). So every realisability census - dictionary build, SAT
refutation, CRT decision - need only decide ONE WORD PER REVERSE CLASS.

Gated: the realised 4-tuple dictionaries at m23/m29/m31/m37 are exactly
reverse-closed (15,696 / 45,854 / 115,193 / 291,675 tuples). Audit of this
project's own arity censuses (`research/data/r24/akillp_*.log`): **82 word
decisions, every reverse pair agreeing** - a falsifiable gate the theorem
passes - and **12,877 s of 27,946 s (46%) was spent deciding the second member
of a reverse pair**, including two of the four span-141 words at 47->53 that
cost 20,005 s between them.

### 7.7 Status

THEOREMS A-D are elementary and proved. Every numeric claim in 7.1-7.6 is
assertion-gated in `research/mirror_lever2.py` (52 gates, exit 0). Prior-art
check for the round-26 material: **not yet checked**; suggested terms in
addition to section 6 - "affine automorphism group of reduced residue system",
"symmetry group of a primorial covering system", "palindromic gap words
primorial", "Jacobsthal antipodal residue".

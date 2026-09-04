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

ROUND-27 ADDENDUM (Formalist): **THE COUNTING HALF IS NOW KERNEL-CHECKED**
(same file, same build). Round 26's named gap is closed:

* `Mirror.even_card_involution` - a map `f` that sends a `Finset s` into
  itself, is an involution there and fixes nothing has `s.card % 2 = 0`.
  Structural induction on a cardinality bound; the step removes `a` and `f a`
  and restricts the hypotheses, which works because `f x = f a` forces
  `x = a` and `f x = a` forces `x = f a`. No machine, no arithmetic.
* `Mirror.window_count_even` - **EVERY WINDOW LENGTH OCCURS AN EVEN NUMBER OF
  TIMES**, except possibly the one carried by the self-mirror window: for any
  `m` that is an involution of `range N` preserving a length function `L`, and
  any `g` that no fixed point of `m` carries,
  `#{t < N : L t = g}` is even.
* `Mirror.adjacent_equal_even` - the endpoint lever at depth 2: an adjacent
  EQUAL pair `(F, F)` (length `2F`) occurs an even number of times whenever
  the exceptional window's length is not `2F`.
* `Mirror.none_of_at_most_one` - the form the live route quotes: parity plus
  a counting bound of `<= 1` gives `= 0`.

HONEST SCOPE, and it is the reason this is an addendum and not a closure.
What is kernel-checked is the LEVER, stated over an abstract index involution:
the counting argument, and its specialisation to the `(F,F)` endpoint. What is
NOT kernel-checked is the INSTANTIATION at a particular machine - that the
machine's own depth-`j` window family is mirror-equivariant with `L` invariant
requires composing `mirror_gear`/`mirror_exposed29` with the opening
ENUMERATION (the `Periodic.lean` machinery), and that composition is not
built. So the lever is available to any argument that supplies its three
hypotheses; the project does not yet supply them from a machine.

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

## 8. ROUND-28 EXTENSION: THE SELF-MIRROR WINDOW'S SIZE, AND THE LEVER'S EXCEPTION LIST

Lateral, round 28. Status: PROVED (elementary) + SCRIPT-VERIFIED exactly at
m7..m23, all depths j = 1..30 (`research/mirror_selfwindow_r28.py`, 83 assertion
gates, exit 0, log `research/data/r28/mirror_selfwindow.log`), plus
`research/tuple_reversal_r28.py` (18 gates).

Formalist's kernel lemma `Mirror.none_of_at_most_one` is machine-free except for
ONE hypothesis:

    hexc : L t0 <> 2 * F        -- the self-mirror window does not itself carry
                                -- the length being counted

`t0` is the unique mirror-fixed window index at that depth. So the entire
machine-side content of the "at most one implies zero" lever is: WHAT IS THE
SELF-MIRROR WINDOW, AND HOW BIG IS IT? This section answers both, exactly.

### 8.1 THE ADDRESS FORMULA (theorem)

Openings `o_0 = 0 < o_1 < ... < o_{N-1}`, `N = prod (q-2)` ODD, mirror closure
`o_{N-t} = P - o_t` (section 1). The mirror carries the depth-`j` window
`W_t = [o_t, o_{t+j}]` to `[-o_{t+j}, -o_t] = W_{N-t-j}`, so on indices it is the
involution `t -> -(t+j)`, span-preserving; its fixed points solve `2t = -j`
(mod `N`), unique because `N` is odd. Writing `M = (N-1)/2`:

    j = 2i   (EVEN):  t_j = -i,     W = [o_{N-i}, o_i],       SPAN = 2 * o_i
    j = 2i+1 (ODD) :  t_j = M - i,  W = [o_{M-i}, P - o_{M-i}],
                                              SPAN = P - 2 * o_{M-i}

Proof of the even case: `W = [o_{N-i}, o_i]` and `o_{N-i} = P - o_i`, so the
forward span is `o_i + P - (P - o_i) = 2 o_i`. Odd case: `2(M-i) = N-1-2i =
-(2i+1)` (mod `N`), and `N - (M-i) = M+i+1`, so the far endpoint is
`P - o_{M-i}`. []

**THE GEOMETRIC STATEMENT.** `Z_P` has exactly two mirror centres: the slot `0`
(an opening, the involution's only fixed slot) and the antipode `P/2` (not a
slot, `P` odd). THE SELF-MIRROR WINDOW AT DEPTH `j` IS THE BALL OF `j+1`
CONSECUTIVE OPENINGS CENTRED ON A MIRROR CENTRE - on slot `0` when `j` is even,
on the antipode when `j` is odd. Both formulas are "twice the distance from a
mirror centre to the `i`-th opening outward".

**THIS GEOMETRIC HALF IS NOT NEW** - section 7 (round 26) already relocated the
exceptional window from an index to exactly this address, and proved the
stronger route-facing statement that it is never QUALIFYING, over more rungs
(to `47->53`) and depths (`j <= 7`) than round 28 covers. What sections 8.1-8.3
add is the QUANTITATIVE half: the closed span formulas, the size table against
`F_j`, the exception list, and the literal discharge of the kernel lemma's own
hypothesis.

### 8.2 THE SIZE, AND THE LEVER'S EXCEPTION LIST

Because the self-mirror window is centred on a FIXED point of the geometry
rather than chosen for its size, its span is a TYPICAL `j`-window span, i.e.
about `j * P/N`, while `F_j` is the MAXIMUM over all `N` windows. So the lever
should fail only where typical and extremal nearly coincide - small machines.
Measured, exactly (`span_self(j) / F_j`):

    j     m7     m11    m13    m17    m19    m23
    1    0.200  0.143  0.091  0.056  0.040  0.029
    2    0.571  0.545  0.375  0.400  0.323  0.256
    3    1.000  0.688  0.478  0.750  0.600  0.420
    4    0.462  0.556  0.385  0.424  0.368  0.241
    5    0.938  0.913  0.750  0.714  0.660  0.600
    6    0.556  0.538  0.452  0.500  0.400  0.260

**THE EXCEPTION LIST** - the `(machine, depth)` pairs with `span_self(j) = F_j`
exactly, i.e. where `hexc` FAILS for the target "span `= F_j`" and the lever is
NOT available - over all depths `j <= 30`:

    m7  : j = 3, 7, 9, 11, 14        m17 : none
    m11 : j = 11                     m19 : none
    m13 : none                       m23 : none

**So from m13 upward the lever is available at every depth up to 30, and the
only failures anywhere are at the two smallest machines.** Restricted to the
route-relevant depths `j = 2..6` (the uniform-order theorem caps the alternation
at `A_relax <= 5`), the exception list is EMPTY from m11 up and the worst ratio
falls monotonically in the machine: 0.913 (m11), 0.750 (m13), 0.750 (m17),
0.660 (m19), 0.600 (m23).

### 8.3 hexc DISCHARGED AT THE ROUTE'S OWN TARGET

The lemma as stated counts windows of length `2F` (an adjacent EQUAL pair, the
`(F,F)` configuration). At depth 2 the address formula gives
`span_self(2) = 2 * o_1 = 2 * d_0`, twice the FIRST GAP - so

    hexc  <=>  d_0 <> F,

and `d_0 = 2, 3, 3, 5, 5, 5` against `F = 5, 7, 11, 18, 25, 34` at
m7..m23. **The hypothesis is discharged by a one-line inequality at every
machine, and `d_0` is itself already a closed form (Mechanic's wrap-gap
identity, r25).** This is the machine instantiation the round-27 handover asked
for, in the only place it was not free.

### 8.4 THE LEVER IS UNIQUE - A REPLICATION OF SECTION 7's ITEM 51, NOT A NEW RESULT

**Recorded honestly: this subsection restates a round-26 theorem in different
coordinates.** Round 26 (item 51) already proved that the affine maps carrying
the opening set onto itself are exactly `(Z/2)^m`, that the element flipping the
gears in `S` has `P / prod_{q in S} q` fixed slots - ONE when `S` is everything -
and that only `c = +-1` sends consecutive openings to consecutive openings,
brute-force gated over 92,400 affine maps at m11. What follows is the same
statement counted on the OPENING set rather than on `Z_P`; the net new content is
the restatement in terms of the exposed sets.

The opening set carries a whole group of involutions, not one: for each subset
`S` of the gears, `sigma_S : k_q -> -k_q` for `q in S` (identity elsewhere) is an
involution of the opening set, because each exposed set `A_q` is closed under
negation. There are `2^n - 1` non-trivial ones. Their fixed-point counts are
exact and need no scan:

    #fix(sigma_S) = N / prod_{q in S} (q-2)

(verified directly on the opening set at m23 for `|S| = 1, 2, 7`). Hence
**EXACTLY ONE of the `2^n - 1` sign involutions has a single fixed point, and it
is the full mirror `S = all gears`** (m23: 127 subsets, next-smallest
fixed-point counts 3, 5, 9). Moreover only `S = all gears` is an ISOMETRY of
`Z_P`: the others permute openings but move distances, so they do not preserve
window length and cannot support a length-counting parity argument at all.

**Conclusion, stated as a negative with its proof: the machine's own symmetry
group supplies exactly ONE parity lever.** Round 26's item 51 showed no symmetry
of the opening set gives a mod-4 lever; this says more - within the natural
`(Z/2)^n` there is not even a second mod-2 lever.

### 8.5 WORD REVERSAL IS THE SAME INVOLUTION, NOT A SECOND ONE

Round 25 listed the mirror on `Z_P` and the reversal symmetry of the gap-word
census as two assets. They are one. Verified cell for cell at m7/m11/m13,
depths 2, 3, 4: the census is exactly reversal-symmetric, exactly ONE PALINDROME
has odd multiplicity, and **it is the self-mirror window's own word** (e.g.
`(2,2)`, `(5,1,5)`, `(1,2,2,1)` at m7).

SELF-CORRECTION of my own round-25 phrasing, which said "exactly one odd
palindrome per depth" in a way that reads as "exactly one word of odd
multiplicity". That is FALSE and my own gate caught it: non-palindromic words
come in reversal pairs of EQUAL count, and equal counts may both be odd - m7
depth 2 has five words of odd multiplicity. The exact law is about PALINDROMES
only, and the safe universal form is "AT MOST one palindrome of odd
multiplicity, and only the self-mirror word can be it": the self-mirror word's
own multiplicity can itself be even (it is 3 at m7 depth 2 and 1 at m7 depth 4,
but nothing forces it odd).

### 8.6 THE LEVER REACHES THE TRANSFER SUPERSETS

Section 7.6 gated reversal-closure on the four EXACT 4-tuple dictionaries. Round
28 re-derives those independently (a replication, not a new fact) and extends
the check to the two CRT TRANSFER supersets, which ARE new here: they are built
by a completely different route (emission from the previous machine, no scan)
and had no a-priori reason to inherit the symmetry unless the emission is itself
mirror-faithful. IT IS. Palindrome counts are new at every row:

    dictionary                          #tuples    #palindromes   orbits
    m23 4-tuples (exact)                 15,696          28        7,862
    m29 4-tuples (exact)                 45,854          50       22,952
    m31 4-tuples (exact)                115,193          79       57,636
    m37 4-tuples (exact)                291,675         139      145,907
    m37 4-tuples (31->37 transfer)    2,435,140         546    1,217,843
    m41 4-tuples (37->41 transfer)    4,239,676         874    2,120,275

Operational payoff, ~50% at every file: any census, LP enumeration or SAT sweep
over one of these dictionaries whose predicate is reversal-invariant need only
visit the orbit representatives.

### 8.7 A CLAIM ABOUT BACKLOG U10 THAT WAS WITHDRAWN

This subsection originally claimed to close half of backlog U10 ("where could a
mod-4 lever come from?"). **The claim was wrong and is withdrawn.** The argument
was: the sign group is elementary abelian, so every non-identity element has
order exactly 2 (gated at all 127 subsets at m23), and the isometries of `Z_P`
preserving the opening set are `{k -> +-k}`, of order 2; hence no `Z/4` action
can be restricted from the machine's automorphism group.

That is true, and it does not touch U10. U10 was posed KNOWING item 51 rules out
symmetries; its surviving candidate (a) is a free `Z/4` action on a SUBSET OF
CONFIGURATIONS, which by construction need not be induced by any map of `Z_P` -
that is what "not a symmetry" means there. The argument above only rules out the
sub-case already covered. **U10 remains open on both candidates.**

### 8.8 STATUS AND PRIOR ART

8.1 and 8.4 are elementary theorems with proofs above; 8.2, 8.3, 8.5, 8.6 are
exact script-verified measurements. PRIOR-ART CHECK: **not yet checked**.
Suggested terms beyond section 6: "fixed point of reflection on a difference
set", "self-conjugate gap in a reduced residue system", "palindromic window
primorial sieve", "Burnside orbit counting covering system enumeration".

## 9. ROUND-29 EXTENSION: THE LEVER'S HYPOTHESIS IS A THEOREM AT EVERY DEPTH >= 3

Lateral, round 29 (2026-09-03).  Script `research/evenj_reversal_r29.py`
(185 assertion gates, exit 0, log `research/data/r29/evenj_reversal.log`).
Status: PROVED (elementary) + SCRIPT-VERIFIED at m11..m23, J = 2..7.

Sections 7.5 and 8 left the lever usable but hypothesis-carrying: round 26
checked at 66 cells that the self-mirror window is never QUALIFYING, and round
28 gave its span formula and an EXCEPTION LIST.  Both are measurements.  This
section replaces them with a theorem, on the sharper family the live route
actually uses (WORD-LEGAL windows, i.e. the family `Q*_J` maximises over -
middles `0` or `+-2u'` mod `q'` with the nonzero classes strictly alternating,
padded middles transparent).

### 9.1 THE MAP, AND WHAT IT PRESERVES

With openings `o_0 = 0 < ... < o_{N-1}`, `N = prod(q-2)` ODD, the mirror carries
the depth-`J` window `W_t = [o_t, o_{t+J}]` to `W_{N-t-J}`, so on indices it is
the involution

>   `R_J(t) = -(t + J)  (mod N)`,   on addresses `k -> -(k + span)`,
>   on gap words `w -> reverse(w)`, on killing residues `r -> -(r + span)`.

`R_J` preserves DEPTH, SPAN, the INTERIOR OPENING COUNT `J-1`, condition T2
(value-wise, hence reversal-invariant) and condition T3 (strict alternation of
the nonzero classes is reversal-invariant).  **So the word-legal family is
`R_J`-invariant and `R_J` preserves span on it**, and `N` odd gives `R_J`
exactly one fixed point per depth - the self-mirror window, centred on slot `0`
for `J` even and on the antipode for `J` odd (section 8.1).

### 9.2 THEOREM.  FOR EVERY `J >= 3` THE SELF-MIRROR WINDOW IS NEVER WORD-LEGAL

*`J` ODD.*  The window has `J+1` openings placed symmetrically about the
antipode, so its central gap is the one straddling the antipode - and by 7.3
both antipodal slots `(P+-1)/2` are OPENINGS, so that gap has length `1`.  For
`J >= 3` the central gap is a MIDDLE, so T2 requires `1 = 0` or `1 = +-2u'`
(mod `q'`).  Now `2u' = 2*6^{-1} = 3^{-1}` (mod `q'`), so `1 = +-2u'` needs
`3 = +-1` (mod `q'`), i.e. `q' | 2` or `q' | 4` - impossible for `q' >= 5`, and
`1 = 0` is impossible.  []

*`J` EVEN, `J >= 4`.*  The window has `J+1` openings placed symmetrically about
slot `0`, which is itself an opening, so its two CENTRAL gaps are both equal to
`d_0 = o_1`, the machine's first gap; for `J >= 4` both are MIDDLES.  Two equal
adjacent middles are either both of the same NONZERO class - which T3 forbids -
or both padded, which needs `q' | d_0`, impossible since `0 < d_0 < q'`.  []

*`J = 2`.*  There are no middles, so every 2-window is word-legal and the
self-mirror one is `(d_0, d_0)`.  **This is the only depth at which the lever
needs a hypothesis, and there it is exactly `d_0 != F`** (round 28, 8.3), with
`d_0 = 2,3,3,5,5,5` against `F = 5,7,11,18,25,34` at m7..m23.

> **COROLLARY.**  `R_J` is FIXED-POINT-FREE on the word-legal depth-`J` family
> for every `J >= 3`, so **every span count over that family is EVEN, with no
> exceptional class, no exception list and no census**.  "At most one word-legal
> `J`-window exceeds the budget" therefore proves there are NONE,
> unconditionally, at every depth `J >= 3` and every machine.

This is strictly stronger than 7.5's route (which excluded the self-mirror window
by SIZE - it is never qualifying - a per-rung check) and than 8.2's exception
list (a span table).  The `J` odd branch needs nothing at all about the machine;
the `J` even branch needs only `0 < d_0 < q'`, one line per rung.

### 9.3 THE GATE

`research/evenj_reversal_r29.py --upto 23 --maxj 7`, 185 assertions at
m11/m13/m17/m19/m23 (`q' = 13, 17, 19, 23, 29`), `J = 2..7`: the self-mirror
word is a palindrome at every depth; it is NOT word-legal at every `J >= 3`,
with the two central middles exhibited (`d_0, d_0` for even `J`, the antipodal
`1` for odd `J`); the word-legal family is `R_J`-closed; `R_J` preserves span on
it; every span count is EVEN at `J >= 3`; exactly ONE odd span count at `J = 2`,
at `2 d_0`; the number of windows attaining `Q*_J` is EVEN at every `J >= 3`.

    machine  q'   d_0  |legal J=2|  |J=3|    |J=4|  Q*_2 Q*_3 Q*_4
    m11      13   3      135          6        0     11    8   -
    m13      17   3     1485         72        0     16   18   -
    m17      19   5    22275       1088        0     25   25   -
    m19      23   5   378675      11784       62     31   33  34
    m23      29   5  7952175     243816        0     39   43   -

Every `Q*_J` reproduces Constructor's independently computed R68 table exactly,
by a different vehicle, as do the emptiness verdicts `Q*_4(23)` and `Q*_5(19)`.
ZERO literal palindromes at even `J >= 4` (Constructor's Theorem B), verified.

### 9.4 WHAT EVEN `J` DOES NOT GIVE, AND WHY

Round 28 observed that at even `J` the maximisers come in REVERSAL PAIRS.  That
is now explained rather than observed: Theorem B forbids a literal word-legal
even-`J` palindrome, so an attaining window is never its own mirror image and
the attaining set has even cardinality (gated at every non-empty cell).

**It yields no inequality on `F_J` or `Q*_J`.**  `R_J` is span-preserving, so the
only object it adds to a counting argument over the word-legal family is the
QUOTIENT by an involution, of cardinality `|family|/2`; every consequence is
therefore the same ONE UNIT the odd-`J` route already gives - "fewer than two
proves none" in place of "fewer than one" - and section 7.1's Theorem A2 already
proved that one unit is the ceiling (the full symmetry group is `Z/2`, exactly).
What 9.2 changes is not the size of the lever but its PRICE: at every depth
`>= 3` the hypothesis is discharged by arithmetic instead of by a machine check.
(PROVED for counting arguments over the word-legal family; **JUDGMENT, NOT
RESULT** for "no argument of any kind".)

### 9.5 STATUS AND PRIOR ART

9.1 and 9.2 are elementary theorems with the proofs above; 9.3 is exact
script-verification.  PRIOR-ART CHECK: **not yet checked**.  Terms beyond
sections 6 and 8.8: "fixed-point-free involution on admissible gap words",
"palindromic constraint alternating residue classes sieve", "antipodal gap
reduced residue system parity".

## 10. ROUND-30 EXTENSION: THE MIRROR GATED ON EVERY RECORD, AND IN TRANSFER COORDINATES

Lateral, round 30 (2026-09-03).  Script `research/mirror_records_r30.py` (150
assertion gates, exit 0, log `research/data/r30/mirror_records.log`).
Status: PROVED (one line) + SCRIPT-VERIFIED on all 24 exact record windows
on file.

### 10.1 THE STATEMENT

For a window of machine `y` (address `k` an opening, span `s`, interior
offsets `0 < o_1 < ... < o_{J-1} < s` the openings strictly inside, every
other slot of the span blocked):

>   `k' = (P - k - s) mod P` is an opening; the openings in `[k', k'+s]` are
>   exactly the reversed offsets `s - o_{J-1} < ... < s - o_1`; the flanks
>   reverse (gap below `k'` = gap above `k+s`, gap above `k'+s` = gap below
>   `k`); for any gear `q''` not dividing `P` the residue of an interior
>   opening maps `r -> (P - r) mod q''`; and `k + k' + s = P` when both
>   addresses are taken in `[0, P)`.

*Proof.*  `k + t` is blocked by `q` iff `k + t = +-u_q (mod q)`; the tooth pair
is closed under negation and `q | P`, so `k + t` is open iff `P - k - t` is
open.  []

Gated at their own machines from the definition, partner always a DIFFERENT
slot: `F_2(41) = 103`, `F_2(53) = 159`, `F_2(59) = 173` A and B (B IS A's
mirror: `k_B = P(59) - k_A - 173`), `F(59) >= 161` (an m53 3-window),
`F_2(43) = 116`, `F_3(43) = 125`, `F_4(43) = 132`, the `F_5(41) = 128` pair (a
mirror pair in MACHINE-41 coordinates, not only in machine-23 ones),
`F_3(47) = 145`, the LP thread's `F_2(37) >= 90` phases (slot 90816580900),
Mechanic's `F_6(47) = 177` (slot 46615676895423125) and the eleven record-law
rows of `chain_31/37/41.json` (`F(31) = 58`, `F(37) = 88`, `F(41) = 91` among
them).

### 10.2 THE MIRROR IN TRANSFER COORDINATES

A transfer search lifts a window of machine `y0` (period `P0`, start `k`) by
phases `c_q` of the new gears: `x = k + j P0` with `c_q = -j P0 (mod q)`.  The
mirror `x -> P - x - s` has `k' = P0 - k - s` and `j' = P/P0 - 1 - j`, hence

>   `(k, c_q)  ->  (P0 - k - s,  (P0 - c_q) mod q)`,   marks reversed.

Gated on `F_6(47)` (`K = 26216680`, phases `(3,21,29,26,26,27)` ->
`K' = 10965288`, `(23,8,3,20,29,9)`, lifting to exactly `P(47) - x - 177`), on
the `F_5(41)` pair (machine-23 starts 4,834,937 / 32,347,080, phases mirrored)
and on the `F_2(59)` pair from machine 23 with eight new gears.  Every transfer
sweep can therefore decide one representative per orbit and copy the verdict.

### 10.3 WHAT IT BUYS, AND ONE NEW CONSEQUENCE

Exactly two things: the factor 2 on every reversal-invariant search, and the
parity constraint - maximisers of `Q*_J` at `J >= 3` (section 9) and of `F_J`
wherever `span_self(J) != F_J` (section 8) come in pairs, so a search that
has found ONE maximiser is provably incomplete and the partner's address is
`P - k - s`.  No inequality on `Q*_J` or `F_J` (section 9.4; JUDGMENT, NOT
RESULT for "no argument of any kind").

The new consequence is at the one depth the lever does not cover.  The
self-mirror 2-window is `(d_0, d_0)` (8.3), so `F_2 >= 2 d_0` at every
symmetric two-tooth sieve, and on the tooth-counterfactual family the ONLY
member (of 14,616 exhaustively enumerated) at which the depth-2 half of (D),
`F_2 <= F + q'`, fails is one where `F_2 = 2 d_0` with `d_0 = 25`, `F = 26`
(tooth-counterfactual-percentile.md 5C.3; `research/d0_family_r30.py`, 13
gates).  Excluding wrap-pair members the family's depth-2 slack is positive at
every step.  So the hypothesis `d_0 != F` that the lever needs at `J = 2` is
also the depth-2 half's arithmetic input, and on the real machine `d_0` is a
closed form.

Prior-art check: **not yet checked**; terms beyond 8.8 and 9.5: "reflection
symmetry of covering-system search space", "symmetric residue system window
pairing", "CRT lift of a mirrored window".

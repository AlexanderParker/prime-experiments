# Mirror parity laws for the twin machine

Lateral, round 25 (2026-08-29). Script: `research/mirror_cells.py` parts A, B
(log `research/data/mirror_cells.log`); the m11..m29 cell/parity columns are
also emitted by `research/spiral29.py`.

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

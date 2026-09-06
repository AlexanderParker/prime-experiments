# The valves, reviewed from the record (review lane of the R4.c hybrid, 2026-09-07)

Mandate: sort everything the project recorded about the interaction between the engine and the
manifold into interface objects, flag the coordinate each fact was measured in, and list every
assumption inherited from the engine's view that the manifold's own laws have since contradicted
or refined. No new computation beyond five small spot checks (`uv run python`, Q = 1000, noted
where used). Nothing here is a new claim; every fact is cited to its file and its number in
`refiled_by_object.md` (C1-C36, M-, W-), `objects_ledger.md` (d1 L1 .. d6 L66, X1-X24) or
`the_wall.md` (faces A-E, 5a-5l). Vocabulary is the owner's (`agents-shared.md`): ENGINE, MANIFOLD
with its smooth zone `[1, Q]` and quiet zone `(Q, Q^2]`, VALVES, EXHAUST; charge `s x P`, air `s`,
fuel `P`, the pure charge `(1, 1)`, back pressure = the manifold's strikes, valve timing = the
placement residue law, knocking. Status words: KERNEL, PROOF, EXACT (with count), MEASURED (with
count), ROOT (the conjecture in disguise), REFUTED, PARTIAL.

The scratch lane's files (`valves_scratch.md`, `research/valves/scratch/`) were not read and are
not written to.

---

## 0. One frame for the whole record: the split parameter `Q`, and three coordinates

Everything the record says about the interaction was measured at one of three values of the cut
`Q` that separates manifold from exhaust, and in one of two coordinates. Naming this first makes
the sorting mechanical.

- **`Q = q` (the manifold empty).** The whole window line (`the_wall.md` 5j, 5k; C1, C2, C17-C24,
  C28-C34). With no manifold gear, the quiet zone `(q, q^2]` is the engine's window; a manifold-open
  pair is any pair; the valves are the engine acting on every pair; the pure charge is the twin
  primes of the window. Every "window" fact is a valve fact at the degenerate split. Coordinate:
  the engine's column `k = (6k - 1, 6k + 1)`.
- **`Q = Z = isqrt(6P + 1)`, the square root of the engine's period** (`period_scale.md`, C2-C15,
  W2-W12). The arena `[0, P)` in columns is the numbers up to `6P = Z^2 = Q^2`: exactly the
  manifold's smooth zone `[1, Q]` (period_scale's "placement prefix" `[1, ceil(Z/6)]`) followed by
  its quiet zone `(Q, Q^2]`, with the exhaust absent by construction (every prime above `Z` is
  silent on the range: the exhaust cap X5). Coordinate: columns.
- **`Q = 10^5, 3 x 10^5` with `q = 5..13`** (`manifold_census_large.md`; `top_machine_6.md` at
  `Q <= 10^4`). Coordinate: the raw line, the manifold's own.

The two coordinates are related by the fold `k = (n + 1)/6` on `n = 5 (mod 6)`, and the fold is
not a definition of the valves but their first act (object I1 below). The engine's records were
all written in columns; the manifold's laws in the raw line; the interface is where the two meet,
and most of the inherited assumptions found below are one coordinate's convention read as the
other's fact.

---

## 1. THE INTERFACE OBJECTS

### I1. The conjugacy: the raw coordinate against the column coordinate

**Definition.** Two maps, and the record conflates them. (a) The manifold's L19 (top_machine_1.md
3.3; KERNEL `conjugacy`, `exists_column`, `conjugacy_census`): on residues modulo a wheel `W`
whose gears leave 6 invertible (the record states it for gears `>= 7`; any gear `>= 5` will do),
`n -> 6^{-1}(n + 1) (mod W)` carries the manifold's open-pair set exactly onto the
same gears' opening set in the column coordinate (teeth `{0, -2}` go to `{6^{-1}, -6^{-1}}`;
0 mismatches in 12 wheels, 2.2 million residues). (b) The literal fold `k = (n + 1)/6` on the
integers `n = 5 (mod 6)`, which is the map the valves use.

**Facts belonging here.**
- L19 as stated: counting and symmetry laws are common property of the two coordinates, metric
  laws (runs, gaps, records, arcs, letters, alignment) are not, because the residue map is not an
  isometry. KERNEL for the map; PROOF for the transfer principle.
- M3 / Appendix B.1: the engine's separation `d_g = 3^{-1} (mod g)`, "one third at every gear", is
  the manifold's separation 2 seen through the fold: `2 x 6^{-1} = 3^{-1}`. EXACT 1,200 of 1,200
  cells (pinned_arithmetic.md); PROOF, one line. Coordinate: columns. Belongs to the interface,
  not to the engine.
- d5 L59, the anchoring dichotomy: a gear folds the line iff `g - t_g = 1`, i.e. `g = 3` (its two
  teeth `0, -2 = 1` adjacent) or `g = 2` (one tooth); no other gear folds. PROOF + EXACT, 23 wheels.
  Coordinate: raw line.
- d5 L56, the rescaling law: `F(G + {2, 3}) = 6 F_col(G) + 5`, `F(G + {2}) = 2 F_2(G) + 1`,
  `F(G + {3}) = 3 F_3(G) + 2`. PROOF (mechanism) + EXACT, 30 of 30 wheels. This is the fold as a
  metric identity: the manifold with the engine's two folding gears added has, on the raw line,
  exactly six times the column record plus five.
- d1 L20, no fold in the manifold: its open pairs are equidistributed mod 2, 3, 6 to within 2 in
  every class; d2 L34, no manifold anchor (a folding gear needs `g <= 4`). PROOF + EXACT.
- The two letters unmoved by the change of coordinate are gears 5 and 7 (`u_g = +-1`): their
  letters `{2, 3}` and `{2, 5}` are the same in both coordinates; every larger gear's short letter
  stretches from 2 to `2u_g ~ g/3` in columns (top_machine_1.md 3.3).

**Coordinate flag.** L19 is a residue statement and needs 6 invertible, so it excludes exactly the
two gears that fold, 2 and 3 - the engine's own first two gears. The valves' conjugacy is (b): on
the raw line, valve 2 strikes every even `n` (one tooth, `n` and `n + 2` both even), valve 3
strikes `n = 0, 1 (mod 3)` (its two teeth adjacent, L59), and what survives both is exactly
`n = 5 (mod 6)`, on which `k = (n + 1)/6` is a bijection onto the columns and an isometry up to the
factor 6. So on the valves' own domain the metric DOES transfer, at scale 6: adjacent columns are
raw pairs `n, n + 6`; the engine's dominoes (M14) are manifold pairs six apart.

**Inherited assumptions found.**
1. M1 records "the anchor 2, 3, 5 folded into the ruler" as a definition (KERNEL, definitional).
   The manifold's L59 and L56 make it the first valve: 2 and 3 acting on the manifold's open set,
   the only two gears in the whole stack that fold, and the fold's cost is the identity L56. The
   column coordinate is the manifold's open set after valves 2 and 3 have fired.
2. R4's REFINEMENT (theory_tree.md, owner 2026-09-06) defined the top machine as "the primes
   above `q` up to the range's edge, teeth `+-6^{-1}`, every gear starting at column 0": the
   manifold in the engine's coordinate. The construction rule and L19 retired it; every metric
   fact period_scale.md states about "the top machine alone" (W7 open runs, W8 closed-run records
   7, 24, 30, 58, 104; C5 runs inside cells) is a fact about the manifold after valves 2 and 3,
   not about the manifold. Its own run ceiling is `q' - 3` (d1 L10, KERNEL), its own record is
   L16/L17/L69, and the two sets of numbers are related by L56, not equal.
3. The one-third separation was carried as the engine's defining arithmetic (node 6,
   separability.md, collision_laws.md, wall W3 "the separation `2 x 6^{-1}` is the SAME rational
   one third at every gear"). It is the manifold's 2 through the fold, and W3's question (does
   the coherent separation drive `K`) was answered no (C23) - as it had to be, because on the raw
   line there is nothing to cohere: every gear's teeth are `{0, -2}` by the definition of a strike.
4. The manifold's dominoes (adjacent open pairs `n, n + 1`, d1 L15 `prod(g - 4)`), its step-2
   chains (`n, n + 2` both open, ceiling `q' - 2`), and its forbidden gap 4 (d1 L4) never reach
   the valves' domain: `n` and `n + 1` differ in parity, and among `n, n + 2, n + 4` one is a
   multiple of 3. Valves 2 and 3 burn every one of them. The record never said so because the
   record never saw the manifold below raw distance 6.

### I2. The anchoring valves and the corridor mod 35 (mod 210 on the raw line)

**Definition.** The engine's four smallest gears 2, 3, 5, 7 acting on the manifold's open set. In
columns this is the corridor `E_35 = {0, 2, 3, 5, 7, 10, 12, 17, 18, 23, 25, 28, 30, 32, 33}`
(docs/proofs/14): spot-checked here, `E_35` is exactly the opening set of the sub-engine `{5, 7}`
in columns (`k` not `+-1 (mod 5)`, not `+-6^{-1} = {1, 6} (mod 7)`; `3 x 5 = 15` classes). On the
raw line it is `n = 5 (mod 6)`, `n` not `0, -2 (mod 5)`, `n` not `0, -2 (mod 7)`: three classes
mod 30 (`{11, 17, 29}`) and fifteen mod 210.

**Facts belonging here.**
- M24 the corridor: endpoint and adjacency laws (294 forbidden pairs), tier A carriers, the
  completeness lemma, the 32-cap, the adjacent-gap exclusion mod 5, the AP lemma, padding onset.
  Partly KERNEL (`Corridor.exposed_iff_mem`, `forbidden_pairs_count`, `prime_adjacent_run_le`,
  `TierA.*`); constrains where, never how big (face B1). Coordinate: columns.
- M41 the gear-5 lock (PROOF, five cases; exhaustive to `L = 2000`; 1.7 million window stretches)
  and M42 the slot rule (EXACT, eight full periods): every maximal blocked stretch has gear 5 at its
  coverage-maximal phase. Coordinate: columns.
- M49 the flanks are coupled by the anchor: 931 of 1,225 pair classes mod 35 admissible. EXACT, 8.8
  million openings. Columns.
- M66 corridor resonance (big gaps recur at slot separations 35, 70, 105 with left endpoints in
  `{10, 12, 18} mod 35`). MEASURED. Columns.
- C21 the reachability landscape: the islands for bound 7 are the offsets `i = 5, 10, 12, 17
  (mod 35)` from the column of `q^2` that no gear `<= 7` strikes at any `q` (the quadratic-residue
  bar); `|Bar(g)| = (g + 1 - chi_g(2) - chi_g(-2))/4`; the doubling law. PROOF + EXACT. Columns,
  offsets from `q^2`.
- C26 the anchor's rigidity in the window: the openings of `{5..13}` sorted modulo any higher gear
  miss fair share by fewer than 30 in every window to `Q = 5000`, and the rigidity is exhausted at
  the first gear above the anchor. PROOF + EXACT, 400,000 gear-rows. Columns; a counting law.
- M14 the alignment law with gear 5: openings are isolated points and dominoes, `prod(q - 4)`
  dominoes. PROOF (docs/proofs/04). Columns.

**Translation into charge words.** Every family `(s, s')` occupies a definite set of classes of
`P mod 30` (and mod 210), namely the solutions of `s' P' - s P = 2` with `P, P'` units mod 30
(mod 210). Spot check at `q = 5`, `Q = 1000`: for the six families `(1,1), (3,1), (1,3), (1,5),
(2,4), (1,9)` the observed classes of `P mod 30` are exactly the predicted sets - `(1,1)`:
`{11, 17, 29}`; `(3,1)`: `{7, 13, 17, 19, 23, 29}`; `(1,3)`: `{1, 7, 19}`; `(1,5)`: `{23}`;
`(2,4)`: `{1, 7, 13}`; `(1,9)`: `{1, 7, 19}`. The pure charge's classes are the corridor pulled
back through the fold; every other family's classes are the corridor's image under `(s, s')`.
The gear-5 lock and the slot rule translate word for word (valve 5 = gear 5) but they are about
the phase of valve 5 inside a twin-free stretch of columns, all columns, charges or not: they say
nothing about charges and are kept as PARTIAL, positional. The alignment law M14 translates as: in
the pure charge, twins at raw distance 6 occur (prime quadruplets) and never three in a row (one of
`P, P + 6, P + 12` is a multiple of 5 or has a partner that is): the smallest-distance metric of the
valves' open set belongs to the engine's 5, not to the manifold's `q'`.

**Inherited assumption.** The corridor was filed as an engine object (M24) and face B's whole
list (B1: corridor, slot rule, lock, phase pinning, the 15-class walk-length law) as "the engine's
positional interface". On the raw line the corridor is simply the residue structure of the pure
charge modulo 210, i.e. the four anchoring valves. Nothing is contradicted; but "constrains where,
never how big" reads differently once the corridor is the pure charge's residue class: it is the
statement that a twin's residues mod 210 are fixed and its position is free, and face B is then
the observation that fixed residues do not bound gaps - a valve fact of the empty manifold.

### I3. The family decomposition and the pure charge

**Definition (d6 L60, PROOF).** In the quiet zone every open pair carries the label `(s, s')` of the
smooth parts of its members, `gcd(s, s') | 2`; the zone's open pairs with both rough parts large are
the disjoint union over labels of `{(sP, s'P') : P, P' prime > Q, s'P' - sP = 2}`. Each family is
one valve; the engine strikes the air `s` or `s'` of every family but `(1, 1)`; the pure charge is
the twin primes above `Q`.

**Facts belonging here.**
- d6 L57 the zone rule (`n` admissible iff `n = sP`, `P` 1 or a prime `> Q`): PROOF, 45 machines, 0
  exceptions; KERNEL `quiet_zone`. L58 the two edges (`p_1` and `p_1^2`, no transition band): PROOF,
  45 of 45. L59 the stratification (`s <= x/p_1`, attained in every stratum; in `(Q, 2Q]`
  admissible = prime or `q`-smooth): PROOF, 0 violations in 5.4 million pairs. Raw line.
- d6 L61 the exact count and the `q`-independence of every family: PROOF + EXACT, 13 machines, 0
  mismatches; family `(1, 1)` contributes `pi_2(Q^2) - pi_2(Q)` at every `q`. Raw line.
- The census at large `Q`: family `(1, 1)` is 27,411,455 at `Q = 10^5` and 203,707,420 at
  `Q = 3 x 10^5` for every engine `q = 5, 7, 11, 13`, while the number of families grows 3,824 to
  147,100. EXACT. Raw line. Spot check here at `Q = 1000`: `(1, 1) = 8,134 = pi_2(10^6) -
  pi_2(10^3)` at `q = 5` and `q = 7`; all 831 families of `q = 5` keep their counts at `q = 7`
  (2,398 families).
- W103: the manifold's quiet-zone record at `Q` is the largest gap between consecutive twin primes
  in its low strata, shortened only where a `q`-smooth number with a prime neighbour lands inside
  the gap; identical across the four engines at 14 of 19 values of `Q`. MEASURED, mechanism stated.
  ROOT for its size. Raw line.
- The air pair splitting a record (census item 3): at `Q = 3 x 10^5` the twin gap 1,452 after
  850,349 is split to `151 + 1,301` for `q = 7` by the pair `(850500, 850502) = (2^2 3^5 5^3 7, 2 x
  425251)`, family `(850500, 2)`, which for `q = 5` the manifold gear 7 strikes. EXACT. This is one
  charge burnt by one valve, seen.
- d6 L63 the prime-gap floor (every `q`-smooth-free prime gap in `(Q, 2Q]` is a run of struck pairs):
  PROOF, 48 machines; truth 3.2 to 24 times the floor. L64 the U-profile (the record is a competition
  between the family-starved bottom stratum and the long top; the bottom wins 28 of 48): MEASURED.
  L65: no upper bound on the zone record, because the bottom stratum is family `(1, 1)`: ROOT.
- d6 L62 the walk in the zone: `nextadm(x) = min(least q-smooth >= x, min_s s x nextprime(max(Q,
  ceil(x/s))))`, the mex over residues replaced by a minimum over smooth scalings of the next-prime
  function. PROOF + EXACT, 11,489,920 positions and 600,000 pair walks, 0 mismatches. Raw line.
- The mirror. d1 L7 (KERNEL, no hypothesis): `n -> -n - 2` preserves the manifold's open set; on
  families it swaps `(s, s') <-> (s', s)`. The census shows the swap as near-equality on ranges:
  313,608 against 312,952 for `(3,1)/(1,3)` at `Q = 10^4`; spot check `(1,3)/(3,1)` 5,890 / 5,970,
  `(1,5)/(5,1)` 2,473 / 2,500 at `Q = 1000`.

**Coordinate flag.** All raw-line. The only column-coordinate facts that are about this object are
C4's both-open cell and C3, sorted under I9 and I4; they are this object seen at `Q = Z`.

**Inherited assumptions found.**
1. R4's REFINEMENT expected "the clutch's own patterns (joint runs, correlations, what the shared
   origin forces) are where the solution space lives". The family decomposition says the valves'
   open set in the quiet zone is one family, `(1, 1)`, whose count is a property of the integers
   and not of the engine (L61); a joint run is a twin gap; the correlation is the coupling 0.83 of
   the both-open cell (C4), which is the `s = 2` handicap on the pure charge's count. The solution
   space is not there; the family decomposition locates the obstruction in the one family no valve
   touches (census "Use").
2. "Fuel = a manifold prime `P`" (glossary). In the quiet zone `P > Q`, so the fuel is a prime above
   the cut - an exhaust gear, whose home strike is exactly the charge (object I4). The glossary's
   word is kept; the flag is that the fuel of every charge in `(Q, Q^2]` is a gear of the tier
   above, not of the manifold.
3. C27 (the structured-families identity, PROOF + EXACT: every family of columns defined by residues
   modulo the engine's period carries twins at the window's own rate) was read as "no family beats
   existence". The `(s, s')` families are not residue families: they are defined by exact
   valuations at the engine's primes, and they carry their own rates (spot check: `(3, 1)` at 0.734
   of `(1, 1)`, `(1, 5)` at 0.30, at `Q = 1000`). C27 does not cover them and does not contradict
   them; the two are different objects with one word.

### I4. The echo set and the home strikes: the outer boundary of the valves

**Definition.** On `(C, C^2]` above a cut `C`, a strike by a gear `p > C` on `n` is a HOME STRIKE
(`n = p`) or an ECHO of a gear at or below the cut (`n` has a prime factor `<= C`). The exhaust's
whole action on the manifold's quiet zone is home strikes and echoes; its first strike that is
neither is `p_1^2`.

**Facts belonging here.**
- X5 the exhaust cap: KERNEL (`exhaust_home_or_echo`, `open_iff_twin`, `wheels_open_iff_twin`,
  unconditional for `2 <= q`; primality of the exhaust gear never used). X15 the redundancy lemma in
  range form (for `g^2 > N` every multiple of `g` in `[1, N]` is `g` or has a factor below `g`):
  PROOF, 7,357,725 strikes, 0 exceptions. X17: the exhaust's action on an open pair of the window is
  exactly two home strikes (57,344 incidences, 2.000 per open pair): MEASURED. X18: the first strike
  neither home nor echo is exactly `p_1^2`: MEASURED (961 at `Q = 30`, 44,521 at `Q = 210`) and
  PROOF by L58. X19: the exhaust's share of the open pairs is 0.000% on `(Q, Q^2]`, then 36.9, 62.8,
  75.0, 82.0, 86.6% by decade. MEASURED. Raw line.
- C1 the route (KERNEL, docs/proofs/01) and C16 the layer law (KERNEL, docs/proofs/15): the same
  fact at `Q = q`, in columns.
- C2 the window is the zero-interaction region (object I10, below): the same fact at `Q = Z`.
- W9 placement geometry: a manifold gear is placed at its home column `h(g) = round(g/6)`, all
  placements in `[1, ceil(Z/6)]`, no column carries three gears. EXACT, 2,338 placements. Columns.
  W10 double occupancy: doubly occupied placements number the twin pairs in `(q, Z]`, and "for every
  `q` there is one" is equivalent to the conjecture. PROOF. Columns. C10: a home column is never
  struck by an engine gear on the gear's own side (the member is the prime). EXACT, 15,549 checks.
  C3: twins = both-open + home-only, exact over 38,889,216 columns.
- R4's ZONES (repeating / non-repeating / silent gears on a range of `K` columns; the effective top
  machine on the range is `{q'..sqrt(6K)}`) and W12 the redundancy lemma: the same object stated in
  columns; NAMED, not formalised there; formalised since as X15 (range) and X5 (window).

**Translation.** In charge words the home strike is the charge itself: the exhaust gear `P`'s home
pair `(P, P + 2)` or `(P - 2, P)` in the quiet zone IS the family `(1, s')` or `(s, 1)` charge with
fuel `P`; the pure charge is the exhaust's doubly occupied placement (X17: two home strikes per open
pair). The same sentence one tier down: the manifold gear `g`'s home pair in the smooth zone is the
column `h(g)`, and a twin in `(q, Q]` is the manifold's doubly occupied placement (W10). And one tier
further down: theorem (E)'s exception set is the engine's doubly occupied placements, the twins
`<= q` (object I6). Home strikes exist at every tier, and at every tier a doubly occupied placement
is a twin of that tier's own range.

**Inherited assumptions found.**
1. "A home strike is a strike but not a kill" (period_scale.md 0, W9) was an exemption convention
   in the engine's coordinate, and period_scale.md itself corrected it ("not a nuisance to exempt:
   the clutch's marker for a twin whose member the top machine happens to own", 3.2). The manifold's
   and the exhaust's own laws have no exemption: the manifold strikes its own twins in `(q, Q]` (both
   members are its gears), the exhaust strikes its own twins in `(Q, Q^2]` (X17). Consequence for the
   valves: the valves' open set never contains a smooth-zone twin. Twins of the period = the pure
   charge of the quiet zone + the manifold's doubly occupied placements of the smooth zone (C3), and
   the second set is manifold-closed.
2. R4's ZONES put the manifold's effective boundary on a range at `sqrt(6K)`; the manifold's own
   laws put the tier boundary at `p_1^2` (L58, X18) whatever the range, and say the exhaust is silent
   on the quiet zone of every split (X5, X6, the ladder of windows). Refinement, not contradiction:
   the zone statement is the cap read at the range's square root.
3. "The clutch's own period is the primorial to `Z`; within `[0, P)` it never repeats and its only
   exact self-similarity is the mirror" (period_scale.md 3.4). The manifold on a range has no period
   at all (d1 W2, ledger 18: non-periodicity belongs to the in-use machine), but it has an exact rule
   (L57) and an exact enumeration (L60): the replacement for periodicity is the smooth/rough split,
   not a longer period.

### I5. The twisted copies: back pressure in the cofactor coordinate

**Definition (C6).** A manifold gear `g` strikes an engine-open column iff, writing the struck member
as `g m`, the cofactor `m` is `q`-rough and the partner `g m -+ 2` is `q`-rough; in the `m`
coordinate that is the opening set of a twisted engine with teeth `{0, -+2 g^{-1}}` at every engine
gear. The manifold's action on the engine's openings is a union of `2(pi(Z) - pi(q))` coherent
twisted copies of the engine at separation `2/g`.

**Facts belonging here.**
- C6: EXACT, 4,676 copies, 17,035,903 cofactors, 0 mismatches; each copy has exactly `prod(h - 2)`
  openings per `m`-period. Columns (the cofactor of a column member).
- C7 level of distribution 1: `|X_g - 2N/g| < 2 x 3^m` at 2,338 cells, pairs at 19,956 with
  `gh < P`, triples the same; the true growth `2^m`. EXACT, 0 exceptions. Coordinate-free (a count
  of openings in classes of a modulus coprime to the engine's period).
- C14 the switching identity `E = 2D + Q` (ordered prime-cofactor strikes = twice the both-prime
  pairs plus the squares): EXACT, five machines; an identity, not an inequality. C15 Brun on the
  valves: the order-2 truncation error equals the order-3 term; exactness buys nothing. EXACT.
  C8 the survivor curve: 1.0000 at `s >= 4.27`, minimum 0.8603 at `s = 2.09`; it is the classical
  `f_2`. EXACT. All counting; coordinate-free.
- C25 each gear's in-window take is one curve in `t = ln g / ln Q'` (fair share below `t = 0.55`,
  1.87 as `t -> 1`), white residual over 105,919 gear-rows, the same for every family of columns.
  EXACT. Columns.
- C19 the near-twins, at most three per rung: the new gear `q'` bites at most three isolated spots
  of its section, spaced `>= (q' - 1)/3`. MEASURED, 666 rungs. Columns (Appendix B.9: a ladder fact).
- B.6: the family `c/r` was introduced as an engine counterfactual (separation_drives_K.md) and is
  the manifold's natural coordinate on the engine's openings.

**Translation.** On the raw line the engine's teeth on the pair `(n, n + 2)` are `{0, -2} (mod h)`
for every engine gear `h >= 5`, the same domino teeth as the manifold's (the one-third is
`2 x 6^{-1}`, I1). Dividing by `g` sends `{0, -2}` to `{0, -2 g^{-1}}`. So the twist is an artefact
of the column coordinate: a twisted copy is the engine's own raw-line domino pattern read in the
coordinate `m = n/g`, and coherence (the same `2 g^{-1}` at every gear) is the statement that the
engine's separation is 2 at every gear on the raw line. In charge words the object is back pressure:
the manifold's strikes on the engine's open set, which in the quiet zone are the rough pairs (air 1
on both sides) whose fuel is not a single prime above `Q`. C7 says back pressure is exactly
proportionate at every manifold gear and every pair of gears over the period. C25 says the back
pressure of gear `g` on the quiet zone of split `Q'` follows one curve in `ln g / ln Q'`, and its
mechanism ("where the multiplier columns `m` sit relative to `g^2`") is the stratification L59 read
per gear. C19 says the back pressure of the one-gear manifold `{q'}` on its own quiet zone
`(q, q'^2]` is at most three strikes: the twisted copy of `g = q'` cut to `m < q'`.

**Inherited assumptions found.**
1. "Twisted" and "coherent at separation `2/g`" were the engine's coordinate describing the
   manifold's plain teeth. Nothing measured is wrong; the description was the wrong coordinate's.
2. R4 (owner) asked whether the union of twisted copies carries bilinear (Type II) structure the
   sieve cannot see; C14 answered that switching gives an identity. In charge words: counting
   two-fuel members `g m` with `m` prime by `g` or by `m` counts the same set. The question was the
   right one; the answer is that the back pressure has no asymmetry to exploit, at `q <= 23`.

### I6. The effective-machine theorem (E) and its exception set: the inner boundary

**Definition (position_frontier.md).** For every column with `6k - 1 > q`, `k` is blocked under
`{5..q}` iff it is blocked under `{5..floor(sqrt(6k + 1))}`. PROOF, one line. The exception set is
exactly the twin gear pairs striking their own home columns, all below `(q + 1)/6`: 7 of 7 count
matches at `q = 23..997`. Columns.

**Facts belonging here.**
- (E) itself, PROOF; M7 a gear exposes nothing below its own square, KERNEL (`Gear.R_eq_zero_of_
  below_sq`, `Layer.slot_cap`).
- M53 the position-length frontier: `R_min(L) = 1` for `L < d_0`, `R_min(L) >= 3.25 L` for
  `L >= d_0`, 0 exceptions in 113 period cells and 8,375 window cells; `c = 1.25` unconditional
  from the ladder. PROOF + EXACT. Columns.
- C28: from `q = 1427` the longest blocked run of `[1, W]` is the initial run from column 1 (2,038
  of 2,038 rungs), so the window statement is exactly `d_0 <= W`, and by (E) the manifold is
  irrelevant inside the window. PROOF + EXACT. Columns. C30: `d_0` is the column of the first twin
  above the top gear, `(d_0 - 1)/(q/6)` in `[0.97, 1.35]`, median 1.005, over 2,254 rungs. EXACT.
- Position_frontier.md 3(c): the initial run `[1, d_0 - 1]` is a chain of effective-machine bricks
  fused at junctions by the engine's twin gear pairs, one per junction, in order (28 pieces at
  `q = 997`). EXACT. 4: at `x = 1` the spectrum-plus-depth bound degenerates to an identity; "the run
  ends at the first `E`-opening that is not a twin gear pair, which is `d_0` by definition". ROOT.
- C29 the window has at most two junctions, the column of `q'` and the column of `q'^2`. EXACT, 152
  rungs. C33 `F_W` is the largest twin gap in `(q, q'^2)`. EXACT. C32 (holders of a window stretch
  against a period record). EXACT. Columns.

**Translation.** (E) says: above its own placement prefix `[1, (q + 1)/6]`, every strike of the
engine is an echo of a gear below the column's square root - the engine is its own exhaust above
`(q + 1)/6`, exactly as the exhaust is home-or-echo above `Q` (X5). The exception set is the engine's
home strikes: the columns `h(g)` for `g <= q`, exclusive precisely when `g` and `g +- 2` are both
prime - the engine's doubly occupied placements. So the valves' inner boundary is the engine's own
placement prefix, and inside it the engine's twins sit as its own home strikes, invisible to any
larger gear's law. The window statement `d_0 <= W` in charge words: the first pure charge above the
cut `q` (the first twin above `q`) lies below `q'^2`; since no fuel exists below `p_1 = q'` (L58),
`d_0 >= q'/6` trivially and the measured median `(d_0 - 1)/(q/6) = 1.005` says the first pure charge
sits essentially at the smooth zone's edge. The initial run is the manifold's origin clump
(`[-(q' - 1), q' - 3]` all open, d1 L6) plus the engine's home strikes, in the engine's coordinate.

**Inherited assumptions found.**
1. (E) was proved and filed as an engine fact ("the effective machine at a column is exact") and its
   exception set as a curiosity. Read with X5 and W10 it is the same law as the exhaust cap and the
   placement law, one tier down: every tier's gears are placed on their home pairs, and every tier's
   twins are its doubly occupied placements. The valves have an inner boundary (the engine's
   placement prefix) and an outer boundary (`p_1^2`) of the same kind.
2. 5k's sentence "the manifold is irrelevant inside the window" is true at `Q = q` and only there:
   at `Q = q` the manifold is empty. At any larger split the manifold's strikes on the engine's open
   set begin at `q'^2 > q^2` (C2), so the sentence is the zero-interaction region (I10) restated, not
   a property of the manifold.

### I7. The placement residue law: valve timing, and the parity barrier in valve terms

**Definition (C9).** As `r` runs over the residues mod `6h` coprime to 6 and nonzero mod `h`, the
home column `k = (r -+ 1)/6 mod h` takes each of the `h - 2` non-tooth classes of engine gear `h`
exactly twice and each of its two tooth classes exactly once; hence the placement density is
exactly `prod(1 - 1/(h - 1))`. PROOF (one line) + EXACT, 25 (machine, gear) pairs, all residues;
real top primes at `q = 23` give per-class ratios 1.93-2.05 against 2.000. Columns.

**Facts belonging here.**
- C9, C10 (the prime member is never struck), W9, W10 (double occupancy is the conjecture), C3,
  C11 the origin law (from `q = 19` the longest both-open stretch of the period starts at column 0
  and ends at the first twin both of whose members exceed `Z`; the top machine at 3% of its density
  in the prefix), C4's home-only sub-cell (3, 9, 26, 78, 268 = the twin pairs in `(q, Z]`), C8, C15.
- R4.a (period_scale.md 4.4): placement is a dimension-1 event (one class per engine gear), double
  occupancy dimension 2, and the step between them is the `E_1` versus `E_2` question of Chen's
  theorem, measured at the sieve's own prediction to 0.01% (C14). EXACT. The parity barrier named in
  the machine's own terms.

**Translation.** Spot check on the raw line, `h = 5, 7`: for a prime `g = 5 (mod 6)`, whose home
pair is `(g, g + 2)`, the engine gear `h` strikes the home pair iff `g = -2 (mod h)`; for `g = 1
(mod 6)`, home pair `(g - 2, g)`, iff `g = 2 (mod h)`: exactly one forbidden class of `g` per gear
per sign. The 2 : 1 column law is the image of "one class per sign" under the fold (the two signs
land on the two tooth classes once each and on every other class together). In charge words: valve
timing is the law of the engine striking the PARTNER of a placed prime. The placed prime is fuel
(air 1, unburnable); the partner is `s' P'`; the engine burns the air `s'` in one class of `g` per
gear; the density of placed primes with an engine-open partner is `prod(1 - 1/(h - 1))`, dimension 1,
provable, and it is the same law whether the placed prime is a manifold gear in the smooth zone or
an exhaust gear in the quiet zone. The pure charge needs the partner's fuel `P'` to be a single
prime as well, i.e. the partner free of every manifold gear too; that is the dimension-2 step. In
valve words the parity barrier is: the engine's timing law on the partner is exact (level 1, the
engine's period invertible modulo every larger prime), and the manifold's timing law on the same
partner - the manifold's gears striking the exhaust's home pairs - is the same one-class-per-gear
law but with moduli up to the square root of the range, where the record has nothing (C34's second
moment dead by proof; C8 is the classical `f_2`).

**Inherited assumptions found.**
1. Period_scale.md pre-registered that home columns are equidistributed over the permitted classes
   (its P8 working expectation) and refuted it with the 2 : 1 law. On the raw line there was never a
   distribution to expect: one class per sign is a definition. The 2 : 1 is the fold.
2. C9 was filed first as a wheels fact (Appendix B.3) because it was about placements; its content
   is the engine's action on the partner and it is a valve fact, as the re-filing already said.

### I8. The island witness and the cover number `K(d)`

**Definition (C20, C22).** For `q` coprime to 30 above 2,849, some island offset `i = 5, 10, 12, 17
(mod 35)` in `[1, d)`, `d = 2 u_q`, is an opening of `{5..q}` past the column of `q^2`; one class
`i = 12 (mod 35)` suffices from `q = 5477`, the free island sits in `[1, 0.152 d)` and its absolute
offset never exceeds 2,392. EXACT, 0 exceptions in 17,748 primes and 52,574 integers to 200,000.
`K(d)` is the fewest gears above 7, each at one phase of its choosing with its two classes at the
fixed separation `2 x 6^{-1}`, that strike every island of `[1, d)`: exact at 23 arcs to `d = 1330`
(3 .. 22), ILP-certified, growth `d/(ln d)^3`. Columns; offsets from the column of `q^2`; the phase
vector `q^2 mod g`.

**Facts belonging here.** C20, C21, C22, C23 (the real separation does not drive `K`: `K_real` is
the mode of the random-separation distribution, and the mean pairwise overlap is `4m/(gh)` for every
separation - the same identity as C7's `4N/(gh)`), C24 (the square phase vector is irrelevant),
N-C6 the class count (a cover with phases is realised by exactly `2^K` classes of `q` modulo the
product of its gears, PROOF, 324 million residues), N-C7 the square pin (`P > q^2` at every
`d >= 70`, so a failure is "the CRT lift of a prescribed residue vector is exactly `q^2`"), C34
(the first moment with the `s = 2` correction predicts 16.51 failures against 17; the second moment
dead by proof), wall faces D1-D3 and E4, W1 (dead: on islands the target is met by one gear at
`d >= 560`), W2 (whole columns: the root in covering language, 5a corrected by 5d).

**Translation.** At `Q = q` the section `(q^2, q'^2)` lies inside the zone rule's validity (L58: the
rule holds up to `p_1^2 = q'^2`), so an engine-open pair there is a pure charge and the witness reads:
a pure charge `(q^2 + 6i - 2, q^2 + 6i)` exists with `i = 12 (mod 35)` below 2,392 - a twin within
about `q/3` numbers of `q^2`, face E4, twin-Bertrand at scale `q/3`. The islands are the residue
classes (I2) shifted to the square edge: offsets that valves 5 and 7 can never burn at any `q`. The
walk from `q^2` in the manifold's own words is L62 at `Q = q`: `nextadm(x) = min(least q-smooth
>= x, min_s s x nextprime(x/s))`, a next-prime walk over smooth scalings, not a residue mex; the
engine's nested formula M29 and the certified column mex d5 L57 compute the same walk in columns. The
cover number does not translate into families: it quantifies over phase vectors the integers do not
realise (N-C7: at most one `q` per cover), which is face D, and the manifold has no phase (its gears'
positions at `n = 0` are fixed by the definition of a strike). `K(d)` is an engine-family object with
a valve target (Appendix B.2), and it stays in columns.

**Inherited assumptions found.** The walk from `q^2` was first filed as a clutch fact and then
re-filed as an engine path with a valve landing (B.2). With L62 the path itself has a raw-line form
that contains no residue: the walk in the quiet zone of any split is smooth-times-next-prime. The
"phase vector `q^2 mod g`" is the engine's coordinate for "where `q^2` sits"; on the raw line the
same information is `q^2` itself, and C24 (real, locally-square and random vectors fail alike) is
the statement that the raw position carries nothing the residues do not.

### I9. The four cells and the level of distribution at period scale

**Definition (C4, `Q = Z`).** Every column classified by (engine state, manifold state); at `q = 23`:
both open 895,791; engine-open / manifold-closed 7,056,384 (home strikes only 268, at least one
proper strike 7,056,116); engine-closed / manifold-open 4,150,311; both closed 25,079,659. Coupling
of the both-open cell 1.0197, 0.9455, 0.8659, 0.8451, 0.8300 at `q = 11..23`, the other three cells
within 5% of independence. EXACT. Columns.

**Facts belonging here.** C4, C5 (both cells needing the engine open have longest run exactly 2 -
gear 5 alone; the both-closed cell reaches the engine's record to `q = 19` and falls short at 23),
C7, C8, C12 (the twin-free record is a joint object: 24, 82, 153, 254, 501 columns, 1.85-3.66 times
the sum of the two machines' own closed records, with the engine closed at 1.005 of its average
rate inside it), C13 (nothing at the period scale gives the window: twin-free stretches of 502
columns against a window of 83), W4-W8 (the top machine in columns: density falling 41% across the
period, 6-9% Buchstab excess, fair-share kills `round(2P/g)`, open runs to 8, closed runs above the
prefix 7, 24, 30, 58, 104).

**Translation into zones.** Both-open = the pure charges of the quiet zone `(Z, Z^2]` (the smooth
zone has none: every Stormer pair is all air, burnt by the engine). Engine-open / manifold-closed,
home-only = the manifold's own twins in the smooth zone (I4); proper = back pressure on the rough
pairs (I5). Engine-closed / manifold-open = the burnt charges: every family with `s > 1` or
`s' > 1`, and the smooth zone's Stormer pairs. Both closed = the rest. So the coupling 0.83 is the
pure charge's count against the product of the engine's and the manifold's open densities, and it
is the `s = 2` handicap on the twin count (R4.a). C5's "run 2, gear 5 alone" is the alignment law
M14 on the valves' domain (I2). C12's twin-free record at `Q = Z` is the largest twin gap below
`6P = Q^2`, in columns (501 columns = 3,006 numbers at `q = 23`); "made by the manifold covering the
engine's ordinary leftovers" is the back pressure filling every rough pair inside a twin gap while
the engine works at its average rate. C13 is the statement that the valves' record at the split
`Q = Z` (a twin gap below `Q^2`) is far larger than the window of the split `Q = q`; nothing at the
large split constrains the small one. W4's monotone fall across `[0, P)` is L64's U-profile seen in
twenty blocks each larger than `Q^{1.5}`, where the density is past its peak; W5's Buchstab excess
is the family mixture; W6's fair share is C7 for the manifold alone.

**Inherited assumptions found.**
1. W8 "the top machine's closed-run records 7, 24, 30, 58, 104" and W7 "its open runs are short"
   are the manifold after valves 2 and 3 (I1, assumption 2); the manifold's own record on a range
   is in the smooth zone (d4 L47, the largest gap of the Stormer list) and its quiet-zone record is
   a twin gap (W103). The record kept two different objects under "the top machine's record"
   (Appendix B.5 already warns of three records under one word).
2. C12 was read as the record being "a joint object 3.7 times either machine's own". In zone words
   both machines' quiet-zone records are twin gaps (W103 for the manifold, C33/C12 for the valves),
   each made by the other machine's strikes on its leftovers; the factor 3.7 compares a twin gap
   with a Stormer gap plus a folded manifold run, which are not the objects the manifold's laws
   name.

### I10. The zero-interaction region

**Definition (C2).** A proper strike of a manifold gear on an engine-open column has member `g m`
with `m` `q`-rough and `m > 1`, hence the member is at least `q'^2 > q^2`; the manifold cannot kill
anything in the window. PROOF + EXACT: first proper kill at columns 28, 60, 60, 140, 140 against
window tops 20, 28, 48, 60, 88 (`q = 11..23`), i.e. at the column of `q'^2` or of the first
engine-open product above it. Columns.

**Facts belonging here.** C2, C1 (the route, KERNEL), C16, C17 the square gate (the deepest hopping
layer of the walk from `q^2` is the top gear iff `q^2 - 2` is prime; 153 open, 514 shut), C18 (the
walk starts on a tooth of `q`, `q` strikes it once and is inert after; `L < d` at every `q` above
53), C29 (at most two junctions, the column of `q'` and of `q'^2`), C31 the self-feeding transfer
rule, X18, X19.

**Translation.** The region is a property of every split, not of the window: below `p_1^2` the
tier above the cut is silent except for home strikes (X5, X18). At `Q = q` it is the window
`(q, q^2]` (with the rule valid to `q'^2`, L58); at `Q = Z` it is `[1, Q^2]` for the exhaust and
`[1, q'^2)` for the manifold's strikes on the engine's open set. C29's two junctions are the one-gear
manifold `{q'}`'s home strike (its home pair, the column of `q'`) and its first non-echo strike (the
column of `q'^2`, X18 at tier 1): the multiples `q' m` with `m` `q`-rough and `m < q'` do not exist
because `q'` is the next prime. C17 the square gate is X18 for the engine's own top gear: `q^2` is
`q`'s first strike that is not an echo, and the walk from `q^2` needs gear `q` iff no smaller gear
strikes that column, i.e. iff `q^2 - 2` is prime. C18's `d = 2c (mod q)` columns is the raw distance
to `q(q + 2)` or `q(q + 4)` divided by 6, the top gear's next multiple on a column; `L < d` is then
"a pure charge before the top gear's next strike", face E4.

**Inherited assumption.** Period_scale.md 3.3: "the window is the ONLY region of the period where
one machine decides alone, and it is 2.4e-6 of the period". At the split `Q = Z` the quiet zone
`(Z, Z^2]` is also decided by engine + manifold alone (the exhaust is silent there, X5), and it is
most of the period. The sentence is true of the engine alone (only in the window does the engine
alone decide) and was read as a statement about the interaction. The stack's ladder of windows (X6)
is the corrected form: every tier's quiet zone is the zero-interaction region of the tier above.

---

## 2. THE COORDINATE FLAG: every column-only fact, and what it becomes

Facts measured only in the column coordinate, one line each: does it survive the fold into the
charge picture, and as what. "Transfers" means the statement is about residues or counts and is the
same on the raw line; "scales" means it is metric on the valves' domain and carries over at scale
6; "does not translate" means it is about an object the raw line does not contain.

- C1 the route: transfers verbatim (KERNEL on the raw member `m = 6k - 1` already).
- C2 zero-interaction: transfers; becomes "the tier above the cut is silent below `p_1^2`" (X5,
  X18), a property of every split.
- C3 twins = both-open + home-only: transfers; becomes "twins of `[1, Q^2]` = the pure charge of
  the quiet zone + the manifold's doubly occupied placements of the smooth zone".
- C4 the four cells: transfers into zones (I9): pure charge / manifold's own twins and back
  pressure / burnt charges and Stormer pairs / the rest.
- C5 runs inside cells: scales; the run of 2 forced by gear 5 is M14 on the valves' domain (twin
  quadruplets, never three twins six apart). Not the manifold's `q' - 3`.
- C6 twisted copies: transfers; the twist disappears (the engine's raw-line teeth are `{0, -2}`,
  the copy is that pattern in the coordinate `m = n/g`). Becomes: back pressure per gear.
- C7 level of distribution 1: transfers; back pressure is exactly proportionate at every gear and
  gear pair over the period.
- C8 survivor curve, C15 Brun, C14 switching: transfer (counting); C14 becomes "two-fuel members
  counted by either fuel give one set".
- C9 placement residue law: transfers; becomes "one forbidden class of the placed prime per engine
  gear per sign" (spot-checked); the 2 : 1 is the fold.
- C10 home column unstruck on its own side: transfers trivially (the member is the fuel).
- C11 origin law: transfers; becomes "the smooth zone holds no pure charge; the first pure charge is
  the first twin above `Q`" (= census item 6, d4 L47's gap at the top of the Stormer list).
- C12 twin-free record: scales (x6); becomes the largest twin gap below `Q^2` at `Q = Z`. ROOT.
- C13 nothing at period scale gives the window: transfers; becomes "the valves' record at a large
  split does not bound the valves' record at a small split".
- C16 layer law: transfers; it is L57/L58 at the split `y`.
- C17 square gate: transfers; X18 for the engine's top gear (`q^2` is its first non-echo strike).
- C18 the walk's frame (`L < d`, the top gear inert): scales; `d` is the raw distance to `q(q+2)` or
  `q(q+4)` over 6; the content is face E4 and does not change.
- C19 near-twins at most three: transfers; the back pressure of the one-gear manifold `{q'}` on its
  own quiet zone, cut to cofactors below `q'`.
- C20 island witness: scales; becomes "a pure charge at `q^2 + 6i`, `i = 12 (mod 35)`, `i < 2392`".
- C21 reachability landscape: transfers as residues (the islands are the anchoring valves' classes
  at the square edge); the doubling law transfers (it is the two roots of `q^2 = r`).
- C22 `K(d)`, C23, N-C6, N-C7: do not translate into families - they quantify over phase vectors
  that no integer realises; the manifold has no phase. Kept in columns as an engine-family object
  with a valve target.
- C24 the square phase vector is irrelevant: transfers as "the raw position of `q^2` carries nothing
  its residues do not".
- C25 in-window take, one curve: transfers; back pressure per gear on the quiet zone of split `Q'`
  as a function of `ln g / ln Q'`; the mechanism is L59 per gear.
- C26 anchor rigidity: transfers (a discrepancy count); becomes "the back pressure of any gear on
  the anchoring valves' classes is fair to within 30, and after the first gear above the anchor the
  survivors are the engine's pattern, not the anchor's".
- C27 structured-families identity: transfers for residue families; does NOT cover the `(s, s')`
  families, which are valuation-defined and carry their own rates (I3).
- C28 `d_0 <= W`: scales; "the first pure charge above `q` lies below `q'^2`"; ROOT.
- C29 two junctions: transfers; the home strike and the first non-echo strike of `{q'}`.
- C30 `d_0`: scales; the first twin above `q`, at about `q` (median 1.005).
- C31 self-feeding (the next level's walk starts at `6k^2 - 2k`; the level-free transfer rule):
  does not translate simply - it is a quadratic identity in the column of a twin member; kept in
  columns as an engine walk fact.
- C32 holders of a window stretch: does not translate usefully (which engine gears hold a stretch);
  kept in columns, PARTIAL.
- C33 `F_W` is the largest twin gap in `(q, q'^2)`: scales; the valves' record at `Q = q` is a twin
  gap, the same statement as W103 (`Q = 10^5`) and C12 (`Q = Z`).
- C34 moments: transfers as a count; face D unchanged.
- M3 one-third separation: transfers into "separation 2" and stops being an engine property.
- M24 corridor, M41 lock, M42 slot rule, M49 flank coupling, M66 resonance: transfer as residues
  (the anchoring valves' classes mod 30, 210); the lock and slot rule are about all columns of a
  stretch and say nothing about charges (PARTIAL).
- M14 alignment law: scales; the smallest-distance law of the pure charge (I2).
- W7, W8 (the top machine's runs and closed records in columns): do not translate to the manifold's
  own metric; they are the manifold after valves 2 and 3, related to the raw record by L56.
- W4, W5, W6 (density fall, Buchstab excess, fair share): transfer; L64's profile past its peak, the
  family mixture, C7 for the manifold alone.
- W9, W10 (placement geometry, double occupancy): transfer; the manifold's own twins as its doubly
  occupied placements, and W10's equivalence is X17's sentence one tier down.
- The tooth-counterfactual family (M67-M70, M74, face C): does not translate. On the raw line a
  gear's teeth are `{0, -2}` by definition (d1 L1, L3 the partner law); moving a tooth to `+-v`
  makes a machine that removes two arbitrary classes, Ziller-Morack's free `h_2` adversary, with no
  partner law. See 4.2.

---

## 3. THE PARTIAL INTERFACES

Kept as PARTIAL, with what each explains and what it leaves. Nothing filed dead for incompleteness.

- **C12 / W103, who makes the record.** Explains agency: the valves' twin-free record in the quiet
  zone is made by the manifold's back pressure on the engine's ordinary leftovers (engine at 1.005
  of its rate), and the manifold's own record is a twin gap shortened only by air pairs with a prime
  neighbour. Leaves: the length. ROOT for the length, PARTIAL for the mechanism.
- **C5, the run of 2.** Explains the both-open cell's run length (gear 5). Leaves: gaps.
- **C25, one curve per gear.** Explains the rate of back pressure per gear in the quiet zone and that
  it belongs to the range, not to any family of columns. Leaves: the maximum (face A4).
- **C26, the anchor's rigidity.** Explains the anchoring valves' discrepancy (below 30) against every
  higher gear. Leaves: everything after the first gear above the anchor, by its own statement.
- **C27, the identity.** Explains why no residue-defined family of columns beats existence. Leaves
  the `(s, s')` families, which are not residue families (I3, assumption 3).
- **C19, at most three near-twins.** Explains the one-gear manifold's bites on its own quiet zone.
  Leaves the twins between the bites.
- **C17 / C18 / C29, the square gate, the single strike, the two junctions.** Explain the top gear's
  and the next gear's whole action on the window: one home strike, one square, nothing else. Leave
  the walk length between them (E4).
- **C31, self-feeding.** Explains which gears carry over between a twin's column and the walk of its
  square (`{7, 17, 31}` at `j = +-1`). Leaves the chain of landings, which has no rule, as
  pre-registered.
- **C11, the origin law.** Explains the smooth-zone stretch (no pure charge below `p_1`, the first
  pure charge at the first twin above `Q`). Leaves the quiet-zone record.
- **L63, the prime-gap floor.** Explains a lower bound on the manifold's record from prime gaps and
  the smooth list. Leaves a factor 3.2 to 24 to the truth.
- **L64, the U-profile.** Explains which stratum wins the manifold's record and why (family
  starvation at the bottom, the long logarithm at the top). Leaves how long.
- **C34, the first moment.** Explains the failure count of the island witness (16.51 against 17)
  with the `s = 2` correction. Leaves the transfer (the second moment is dead by proof).
- **The placement residue law as timing (I7).** Explains the engine's action on a placed prime's
  partner completely, dimension 1. Leaves the manifold's action on the same partner, dimension 2.
- **The knock (M74).** The real teeth glue at 2.4x the matched family. Explains nothing yet; not a
  contradiction of anything; and it is measured against a family the raw line does not admit (4.2).
  Kept as the standing knock, with that flag.

---

## 4. THE SHADOWS: the wall's faces in valve words, and which manifold laws bear on each

### 4.1 Face A, the count that cannot see position

In valve words: the pure charge's count. L61 gives every family's count as a property of the
integers; for `(1, 1)` that count is `pi_2(Q^2) - pi_2(Q)`, and every sieve estimate of it at
`s = 2` has lower function zero (C8: 1.0000 at `s >= 4.27`, 0.86 at `s = 2.09`). C7 says the count
is exact to `2^m` (level 1) and C15 says exactness buys nothing because the main terms alternate at
`s = 2`. The placement law names the step: the partner's air is 1 (dimension 1, proved) against the
partner's fuel is a single prime (dimension 2).

What bears on it from the manifold's proved laws:
- The census law (L61) and the census table: raising `q` adds valves and changes no family's count.
  So no engine size helps the pure charge's count; the count that face A cannot see is the same
  count at every split. Bears negatively and exactly.
- The gap census law (d2 L22, KERNEL) and the cover polynomial (W86): the manifold's own gap census
  on a wheel is an exact inclusion-exclusion product, and its record is `max{d : C(d) > 0}` (d3
  L36) - a sign question of an alternating sum that the spectrum cannot decide (L41). This is face A
  inside the manifold: the same alternating-sum obstruction, proved to be one.
- The mex closed form (L30/L32/L50, and L62 in the zone): the walk in the quiet zone is a minimum
  over smooth scalings of the next-prime function. It says the valves' walk is a prime-gap object,
  not a residue object: face A at the walk, in the manifold's own words.
- The zone laws (L57-L59): the pure charge is the only fuel-only family; in the bottom stratum
  `(Q, 2Q]` every open pair is a twin or has a smooth member (L59), so the count in the bottom
  stratum is the twin count plus a Stormer-type count. Locates face A; does not move it.
- The record rule (L69, KERNEL) and the parity law (L17): bear only on wheels with all gears large;
  in use the tail is empty (d4 L52) and the rule reduces to a scan of the whole core. Do not bear.

### 4.2 Faces C and D, typical and the transfer from typical to real

Face C in valve words: the engine's real teeth are typical among the tooth-counterfactual family
(M70, C2, C4 of the wall), except the knock. Face D: rare among all phase vectors is not never for
real `q` (D1-D3); the transfer needed is equidistribution of `q^2` in structured sets modulo products
far above `q^2`.

What the manifold's laws say about the reference classes:
- The partner law (d1 L3, KERNEL, no hypothesis) and the anchoring dichotomy (L59): on the raw line
  every gear's teeth are `{0, -2}`; there is no family of tooth positions. The tooth-counterfactual
  family is a column-coordinate construction that, pulled back, removes two arbitrary classes per
  gear with no partner: it is Ziller-Morack's free adversary `h_2` (wall 5a), whose window statement
  is their Conjecture 6, strictly stronger than the root. Face C's "typical" is typical within a
  family that contains no valve. The families the raw line does admit are gear SETS (`A(K)`, the arc
  multiset) and the split `Q`; against gear sets the real machine is on the good side (M71: twin
  gears help; M72: an optimal 10-gear blocker). The knock is measured against the inadmissible
  family; whether it survives against an admissible one is not on the record.
- Face D: the phase vector `q^2 mod g` is the engine's coordinate for the raw integer `q^2`; C24 says
  real, locally-square and random vectors fail alike, i.e. the raw position carries nothing. The
  transfer face is then: a statement about all integers `q^2` against a statement about the squares.
  L62 (the walk is smooth-times-next-prime) bears in one way only: it says the walk from `q^2` in the
  zone is decided by the next primes above `q^2/s` for the smooth `s`, so "typical vector" and "real
  `q`" differ exactly by whether those next primes are the next primes of an actual square's
  neighbourhood - which is the transfer restated, not reduced.
- 5i: at the period scale the transfer face disappears because the valves' coupling is exact (C7).
  In valve words: at the split `Q = Z` there is no transfer to make, and face A stands alone. The
  transfer is a face of the small split `Q = q` only.

### 4.3 Face O (5h), the order growing with the gear count

In valve words: the least order of gear interaction whose joint maximum falls below the window grows
like `K - 3` (1, 2, 2, 3, 4, 5, 6, 7, 8, 8 at `K = 3..12`); pairwise laws prove the adversarial
lemma at `K = 4` and no bounded order reaches all `K`; a new gear's net contribution to coverage
turns negative once the small gears' strike sum passes one half. An engine-family statement in
columns; the valves enter only as the target (`A(K) < (p_{K+1}^2 - 1)/6`).

What the manifold's laws say:
- The collision law (d2 L29, PROOF): in a window with every gear `> L + 1` no three distinct traces
  pairwise intersect, so the record cover is a perfect tiling for even `m` and wastes one unit for
  odd `m`. The manifold's own interaction order in the free regime is two, and its record is linear
  in the gear COUNT (L17 `2m - (m mod 2)`; increments `+3, +1`). So 5h's growing order is a property
  of the LOADED regime (small gears present), which the engine is always in (its smallest gear is 5,
  never large: L17's remark) and the manifold in use is also in (L52, the tail is empty).
- The loaded record rule (L69, KERNEL) and the core/tail rule (d4 L53): the record is a function of
  `m` and the core `{g <= F_top + 1}`; when the core is everything, the rule is a scan. That is 5h
  from the manifold's side: no closed form below a full scan of the core wheel, and the core is the
  whole machine in use.
- L49 (no bound linear in `m` on a range: `F_range >= (m log m)/2 - s(q)`) and L55 (no saturation:
  `F_range` linear in the largest gear because the smooth zone is `[1, Q]` whatever `N`): in valve
  words the manifold's in-use record is the Stormer gap, which grows with `Q`, so any "order" law on
  a range is dominated by the smooth zone, not by interactions at all.
- The rescaling law (L56) makes the engine a manifold-type machine on the raw line with 2 and 3
  added: `F_engine = (F_raw({2, 3} + {5..q}) - 5)/6`. The manifold's record laws apply to that gear
  set in principle and reduce, in the loaded regime, to L69's scan. So 5h is the same wall in both
  coordinates, and the manifold's laws prove that it is the loaded regime's wall, not a property of
  the fold or the anchor.

### 4.4 Faces B and E in one line each

Face B (position cannot see length) in valve words: the pure charge's residues mod 210 are fixed and
its position is free (I2); every positional law is a law of the anchoring valves and constrains
where a twin can sit, never how far the next one is. The manifold's L41 (full spectral support:
the spectrum decides the run record and cannot decide `F_top`) is face B on the manifold's side.
Face E (every local formulation over-asks): each of `d_0 <= W`, the section statement, `L < d`, the
arc witness is "a pure charge within a short distance of a named point" (the cut, the square, the
top gear's next multiple); the manifold's L65 is the same sentence for its own record.

---

## 5. PREDICTIONS ON RECORD for the reconcile step

Statements the recorded facts imply, which the scratch lane, working from the definitions alone,
should rediscover. Each carries the recorded fact it rests on and the spot check made here.

**P1 (residues).** Each family `(s, s')` of the quiet zone occupies exactly the classes of `P mod 30`
(and of the pair mod 30, and mod 210) that solve `s' P' - s P = 2` with `P, P'` units; the pure
charge occupies `P = 11, 17, 29 (mod 30)` and the fifteen classes mod 210 that are the engine's
corridor `E_35` pulled back through the fold; and the manifold's mirror `n -> -n - 2` (d1 L7) swaps
`(s, s')` with `(s', s)`, so mirror families have the same number of classes and nearly equal
censuses on a range. Rests on L60, d1 L7, docs/proofs/14; spot-checked at `Q = 1000` for six families
(classes exact) and three mirror pairs (5,890 / 5,970; 2,473 / 2,500; 2,415 / 2,432).

**P2 (the pure charge is `q`-free).** At fixed `Q`, the count of the pure charge - and of every
family - does not depend on `q`; raising `q` adds families and changes none. Rests on L61 and the
census (27,411,455 and 203,707,420 at four engines); spot-checked at `Q = 1000` (8,134 at `q = 5`
and 7, equal to the twin count in `(Q, Q^2]`; all 831 families of `q = 5` unchanged at `q = 7`).

**P3 (the valves' record is a twin gap at every split).** The longest run of the quiet zone with no
pure charge is the largest twin gap there, at `Q = q` (C33), at `Q = 10^5` (W103) and at `Q = Z`
(C12); inside it the engine's openings sit at their average rate (1.005) and the manifold's back
pressure closes every one; the manifold's own record is the same twin gap in the bottom stratum,
shortened only where an air pair with a prime neighbour lands inside (the family `(850500, 2)` at
`Q = 3 x 10^5`). Rests on C12, C33, W103, L59, L64.

**P4 (valve timing on the raw line, and home strikes at every tier).** The engine strikes the home
pair of a placed prime `g` in exactly one class of `g` per engine gear per sign (`g = -2 (mod h)` for
`g = 5 (mod 6)`, `g = 2 (mod h)` for `g = 1 (mod 6)`), so the fraction of placed primes with an
engine-open partner is `prod_{5 <= h <= q} (1 - 1/(h - 1))`; the column 2 : 1 law is this under the
fold. And the same placement structure exists at every tier: a twin in a tier's own range is that
tier's doubly occupied placement - the exception set of (E) for the engine, W10 for the manifold,
X17 for the exhaust. Rests on C9, C10, position_frontier 3(b), W10, X17; spot-checked for `h = 5, 7,
11` (2 : 1 in columns) and `h = 5, 7` (one class per sign on the raw line).

**P5 (RED FLAG: a claim of this would be suspect).** That any metric law of the manifold - the
run ceiling `q' - 3`, the step-2 chain ceiling `q' - 2`, the dominoes `prod(g - 4)`, the forbidden
gap 4 by the partner law, the parity record `2m - (m mod 2)` - survives into the valves' open set, or
that the L19 conjugacy `n -> 6^{-1}(n + 1)` is the map between the manifold's open set and the
valves' domain. The recorded facts say the opposite: valves 2 and 3 burn every domino (parity) and
every step-2 chain (mod 3) before any larger valve acts; the valves' domain is `n = 5 (mod 6)` and
its smallest-distance metric is the engine's alignment law with gear 5 (twins six apart occur in
pairs, never in threes); L19 excludes exactly the two folding gears and is not an isometry; the
manifold's column-coordinate "records" 7, 24, 30, 58, 104 (W8) are the manifold after the fold,
related to its raw record by `F(G + {2, 3}) = 6 F_col(G) + 5` (L56), and are not the manifold's `F_top`.
A secondary red flag of the same kind: that the pure charge's count is the engine's open density
times the manifold's open density times the range. It is 17% below that at `q = 23` and the deficit
grows (C4, coupling 0.83 and falling).

---

## 6. The inherited assumptions, collected

One line each, with the object that carries the correction.

1. The column `(6k - 1, 6k + 1)` was a definition (M1); it is the first valve, gears 2 and 3 acting on
   the manifold, the only folding gears (L59), at the cost L56. [I1]
2. The manifold was first defined in the engine's coordinate with teeth `+-6^{-1}` (R4 REFINEMENT);
   every metric fact about "the top machine alone" in period_scale.md (W7, W8, C5) is the manifold
   after the fold, not the manifold. [I1, I9]
3. The one-third separation was the engine's arithmetic (node 6, W3, collision_laws.md); it is the
   manifold's 2 through the fold, and there is nothing to cohere on the raw line. [I1, I5]
4. The manifold's dominoes, chains and forbidden gap were expected to matter to the interaction;
   valves 2 and 3 burn all of them below raw distance 6. [I1, P5]
5. The corridor was an engine object; it is the pure charge's residue class mod 210, the four
   anchoring valves. [I2]
6. "The clutch's own patterns are where the solution space lives" (R4 REFINEMENT); the valves' open
   set is one family whose count is `q`-free, and the obstruction sits in it. [I3]
7. "A home strike is not a kill" was an exemption; every tier strikes its own twins as home strikes,
   so the valves' open set never contains a smooth-zone twin. [I4]
8. The zones' boundary at `sqrt(6K)` was a range statement; the tier boundary is `p_1^2` at every
   split. [I4, I10]
9. "The clutch never repeats within `[0, P)`" looked for a period; the manifold on a range has an
   exact rule and enumeration instead (L57, L60). [I4]
10. "Twisted, coherent copies at separation `2/g`" described the engine's plain raw-line teeth in
    the wrong coordinate. [I5]
11. Theorem (E) was an engine curiosity with an exception set; it is the exhaust cap one tier down,
    and the exception set is the engine's own doubly occupied placements. [I6]
12. "The manifold is irrelevant inside the window" (5k) holds because at `Q = q` the manifold is
    empty; it is the zero-interaction region, a property of every split. [I6, I10]
13. Home columns were expected equidistributed (period_scale P8); on the raw line the timing law is
    one class per sign by definition, and the 2 : 1 is the fold. [I7]
14. The walk from `q^2` was a residue object (nested mex); in the zone it is smooth-times-next-prime
    (L62), a prime-gap object. [I8]
15. "The window is the ONLY region where one machine decides alone" is true of the engine and was
    read of the interaction; every tier's quiet zone is such a region for the tier above. [I10]
16. C27's "no family beats existence" was read to cover the `(s, s')` families; they are not
    residue families and carry their own rates. [I3]
17. Face C's reference family (teeth moved to `+-v`) has no raw-line existence (the partner law);
    the knock is measured against it. [4.2]
18. Three objects shared the word "record" (B.5); in the quiet zone both machines' records are twin
    gaps made by the other machine's strikes. [I9]

Spot-check script: `scratchpad/spot.py` (session scratchpad), Q = 1000, q = 5 and 7; no project
file written by it.

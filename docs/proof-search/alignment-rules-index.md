# Alignment of openings: the merged rule index

The companion index to `docs/proof-search/alignment-rules.md`.  That document is the picture; this
one is the ledger.  Read section 0 of `alignment-rules.md` first -- it fixes the vocabulary (column,
gear, tooth, opening, machine, window, section, stretch, record), the status tags, and the two
translation hazards ("window" = J-run; `F_2(M)` versus `F(2,y)`).

Every entry from the three harvests, deduplicated, in the schema *Statement / Calculates / Status /
Where / Limits*.  Where two harvests state the same rule the sharper statement is kept and both
pointers are given.  Sources are abbreviated: **[S]** `harvest_shared.md`, **[La]**
`harvest_lanes.md`, **[Ar]** `harvest_archive.md`.

### A. One gear, the columns, and the frames

**A1. Tooth rule.**  *Statement:* gear `q` strikes `k` iff `k = +-u_q (mod q)`, `u_q = 6^{-1} mod q
= round(q/6)`, `6u_q = q -+ 1`; the other `q-2` residues are openings; tooth-to-tooth distances are
`2u_q` and `q - 2u_q`, summing to `q`.  *Calculates:* a gear's exact opening set from `q` alone, and
the two literal letters of every merge word.  *Status:* `[kernel]` (`TwoTeeth.kill_spacing`,
`kill_spacing_min`, `kill_period`, `teeth_letters`, `gear_side`; `LiteralCapTable.
tripled_teeth_antipode`), asserted for every gear 5..199 and to `q = 100000`.  *Where:* [La] A;
[S] A T1; [Ar] Part 1; mechanic Definitions, constructor R40, formalist 2.18, lateral 29(b).
*Limits:* says nothing about which columns survive all gears jointly.

**A2. Shield and umbrellas.**  *Statement:* `k = 0 mod q` is a shield (q divides the midpoint), at
the exact centre of the short umbrella (length `2u'-1`); the long umbrella has length `q-2u'-1`;
the two arcs are locked at ratio 1:2 (`~q/3`, `~2q/3`).  *Calculates:* the longest run one gear
leaves open (`~2q/3`).  *Status:* `[exact: gears 5..47; tabulated at 5,11,17,23,29,41,101,1009]`.
*Where:* [Ar] `umbrellas-and-shields.md`.  *Limits:* one gear; a big gear alone never makes a long
blocked stretch.

**A3. Self-blocking (low tooth = the gear's own column).**  *Statement:* the low tooth is
`u' = round(q/6)`, exactly the column containing the gear's own pair; `q = 5 mod 6` puts the
left-kill tooth low, `q = 1 mod 6` the right-kill.  *Calculates:* both teeth from `q` with no
inverse computation.  *Status:* `[exact: gear table 5..47]`.  *Where:* [Ar]
`umbrellas-and-shields.md`; `twin-prime-program.md` 17c.  *Limits:* the two other observed tooth
patterns (alternation; differences 3,5,7,9,...) BREAK at prime gaps of 6.

**A4. The `u'` column is the twin sequence.**  *Statement:* low teeth run 1,1,2,2,3,3,4,5,5,6,7,7,8
and a value appears twice exactly when column `u'` is a twin pair.  *Calculates:* which gear pairs
strike the same column at the bottom of the window -- twin gears only.  *Status:* `[exact: gear list
to 47]`.  *Where:* [Ar] `umbrellas-and-shields.md`.  *Limits:* a coincidence of TEETH, not of
openings; it removes no opening.

**A5. Teeth are never adjacent / neighbour-of-a-hit.**  *Statement:* `d_q = 2u_q = 3^{-1} mod q` is
never `+-1`, so a hit's neighbour is `g`-free, and `P(x+1 open | x a g-hit) = P(open) g/(g-2)`
exactly.  *Calculates:* the `x+1` restart in the nested next-opening formula; the conditional
opening rate (0.2342 = 0.2139 x 23/21 at `g=23`, 0.2390 at 19, 0.2994 at 7).  *Status:* `[kernel]`
(`AnchorChain.neighbour_of_hit`, every gear from `6u = 1`); `[exact: full period {5..23}]`.
*Where:* [S] A; [La] F; `anchor-235.md` 9e.  *Limits:* one gear; a hit's neighbour is open LESS
often than a random blocked column's (0.2481) -- not a way to find openings.

**A6. Column 0 is always open; the sharing law.**  *Statement:* gear `q` strikes `+-6^{-1}`, never
0; survivors per period are `prod(q-2)` regardless of phases.  *Calculates:* the period identity
every census must satisfy (standing rule 25).  *Status:* `[exact: elementary, asserted at every
machine]`.  *Where:* [S] E; [La] B; [Ar] Part 1.  *Limits:* a count, not a position; column 0 is a
primorial multiple, the alignment that is useless for the window.

**A7. The wrap gap equals the first gap.**  *Statement:* by mirror closure the largest opening is
`P - x_1`, so `wrap = d_0 = 2,3,3,5,5,5,7,7,7,10` at m7..m41.  *Calculates:* the cyclic-seam gap in
closed form; every full-period census total.  *Status:* `[exact: nine machines,
cyclic_close_r25.py]`.  *Where:* [La] B; [S] D.  *Limits:* per-machine; nothing predicts `d_0`
beyond the sieve.

**A8. Antipodal columns are open.**  *Statement:* `(P +- 1)/2` are openings at every machine and the
antipodal gap is 1; hence `W_1(g)` is even for every `g >= 2` and the record gap never occurs once.
*Calculates:* an opening exhibited by arithmetic at a machine no kernel can scan
(`antipode_exposed29 : Exposed29 539141103`).  *Status:* `[kernel]` (`Mirror.antipode_open`).
*Where:* [S] D; [La] B; formalist R26.4; lateral 53.  *Limits:* two columns; it is where the gaps
are shortest.

**A9. The alignment law.**  *Statement:* the longest run of consecutive openings of a gear set is
the long arc of its smallest gear, `q_min - 2u_min - 1` (5->2, 7->4, 11->6, 13->8, 17->10), because
CRT realises every relative phase somewhere.  *Calculates:* that longest run for any gear set from
its smallest gear alone; corollary -- the pattern supports prime quadruplets at every level.
*Status:* `[exact: 103 gear sets, research/alignment.py]`.  *Where:* [Ar]
`twin-prime-program.md` 26c-26d.  *Limits:* it says SOMEWHERE IN THE PERIOD; the period is the
primorial, the window is `y^2/6`.  This limitation IS Reduction A.

**A10. Points and dominoes.**  *Statement:* gear 5 forbids three consecutive open columns, so the
opening set is a disjoint union of isolated points and dominoes; `n_1 = prod(q-4)` dominoes,
`prod(q-2) - 2prod(q-4)` singletons; the `prod(q-2k)` family collapses at `k = 3` and needs
`q >= 6(k-1)`.  *Calculates:* the complete local description of the openings.  *Status:*
`[exact: corollary; checked at {3,5,7}]`.  *Where:* [Ar] `ideas-from-the-session.md` 2;
`twin-prime-program.md` 31c.  *Limits:* local only; it says nothing about spacing BETWEEN the
objects, which is the whole problem.

**A11. Anchor-open columns (gear 5's frame).**  *Statement:* the anchor 2,3,5 leaves open exactly
`k mod 5 in {0,2,3}` (numbers `1,11,13,17,19,29 mod 30`); every machine's openings are a subset.
*Calculates:* the base density 6/30 and the frame in which every gear's hits are counted; the AP
lemma and the adjacent-gap exclusion law are consequences.  *Status:* `[exact: definitional;
kernel-adjacent through Machine19.expT / Corridor.Exposed]`.  *Where:* [La] A; `anchor-235.md`
conventions.  *Limits:* gear 5's frame only.

**A12. One gear over the 2,3,5 anchor.**  *Statement:* a gear `q >= 7` hits exactly six anchor-open
numbers per run of `30q`, one per anchor-open class, hence each twin-slot type twice per run, and
leaves `q-6` cycles wholly untouched; `q mod 30` fixes only WHERE the six hits sit.  *Calculates:*
the six `m`-values in four classes, the untouched-run lengths `q x (gap in m)`, and the cycle index
`((qm - 11) div 30) mod q`.  *Status:* `[exact: every prime 11 <= q <= 5000]`.  *Where:* [La] A;
`anchor-235.md` 1-2.  *Limits:* single gear; clean ends hold from `q = 37` on with listed
exceptions at 11,13,17,19,29,31.

**A13. The +-1 walk.**  *Statement:* every gear and every combination walks the 6-cycle at exactly
`+-1` per rotation, never faster; kills are spaced 4-then-2 rotations (or 2-then-4), which is what
makes the long and short umbrellas.  *Calculates:* the direction a gear walks the columns.
*Status:* `[exact: measured for every sub-machine of up to four gears from 5 to 59]`.  *Where:*
[Ar] `gear-at-infinity.md`.  *Limits:* a rate statement about presentation, not about where
coincidences fall.

**A14. Gear-3 and gear-5 blocking laws.**  *Statement:* gear 3 blocks one of any two adjacent
positions (so every admissible gap is a multiple of 3 and `F_h = 0 mod 3`); gear 5 blocks one of any
three positions spaced 3 apart.  *Calculates:* the shape of the opening set at the finest scale, and
a mod-3 law on the record.  *Status:* `[exact: proved; the mod-3 law checked against all thirteen
known F_h]`.  *Where:* [Ar] `covering-bound-route.md` 18a.  *Limits:* h-frame statements about the
two fastest gears.

**A15. The complete mod-3 dichotomy.**  *Statement:* `3 | F_d(y)` for every gear set **iff** `3` does
not divide `e` **iff** `d != 0 mod 6`; mechanism -- a gear blocks `n = 0` and `n = -e`, two residues
collapsing to one exactly when `q | e`, and at `q = 3` that leaves a SINGLE class.  No gear above 3
can pin openings to one residue.  *Calculates:* whether the record is forced to be a multiple of 3;
the twin form `F(2,y) = 0 mod 3` cuts 2/3 of all coverable increments in a record search.
*Status:* `[kernel]` (`three_survivors_congr`, `three_dvd_gap`, `no_mod_law_above_three`,
`F_zero_mod_three`, ...); `[exact: full periods y = 11..23, 15 gap classes]`.  *Where:* [Ar]
harvester r8/r9/r15, formalist r11, `proofs/MaxGap.lean`.  *Limits:* gear 3 only; no analogue higher.

**A16. The two frames are one machine, scaled by 3.**  *Statement:* the adjacent frame is the column
machine scaled by 3: `F(2,y) = 3 F_k(y)`, `F2_adjacent = 3 F2_k`.  *Calculates:* transfer of every
result with `L -> 3L`; verified 15, 21, 33, 54, 75, 102, 129 against 5, 7, 11, 18, 25, 34, 43.
*Status:* `[exact: seven gear sets for F, six for F_2]`.  *Where:* [Ar] `gear-recursion.md` 1;
`handover.md` 0.5.  *Limits:* the adjacent-frame `L = 1` has NO column-frame counterpart, so
`h(1) = d/(1-d)` is a grid artefact.

**A17. The separation family.**  *Statement:* generalise to gear `q` blocking two residues at
arbitrary separation `s_q`; the adjacent frame is `s_q = 1`, the column frame `s_q = 3^{-1}`; `F`
depends on the separation VECTOR.  *Calculates:* which claims are fragile under generalisation.
*Status:* `[exact: frame definition, used to refute the gear-3 lemma]`.  *Where:* [Ar]
`handover.md` 0.5.  *Limits:* none.

**A18. The corridor `E_35`.**  *Statement:* gears 5 and 7 leave open exactly 15 residues mod 35;
every opening of every machine lies in `E_35`, and a stretch's partial sums must all stay in `E_35`
(the carrier).  *Calculates:* for any prescribed gap word, the admissible base residues mod 35; an
empty carrier forbids the configuration at every machine forever.  *Status:* `[kernel]`
(`Corridor.exposed_iff_mem`, `endpoint_law`, `adjacency_law`, `forbidden_pairs_count = 294`,
`no_chain_of_forbidden`; `TierA.carrier`, `mem_carrier_of_chain`, `no_chain_of_carrier_empty`).
*Where:* [La] A; [S] E; [Ar] Part 9.  *Limits:* corridors constrain WHERE, never HOW BIG (escape
distance 1).

**A19. Completeness lemma.**  *Statement:* a shape with `n` prescribed openings can be blocked by
gear `q` only if `q <= 2n`.  *Calculates:* which gears must be consulted at all.  *Status:*
`[exact: proved]`.  *Where:* [La] A/item 20; [Ar] lateral r17; [S] E.  *Limits:* **harvests
disagree on the reach** -- [La] and [Ar] say the mod-35 test IS the entire corridor for `n <= 5`
(gear 11 enters at `n = 6`, gear 13 at `n = 7`); [S] states `n <= 3` and flags `n <= 5` as claimed
in one place.  Necessary condition only (the exposed half).

**A20. The 32-cap / horizon theorem.**  *Statement:* the `(5,7)` both-composite classes `{1,34}` mod
35 have max cyclic gap 33, so any 33 consecutive columns contain a both-composite column and a
prime-carrying run is at most 32 long, at every scale, forever.  *Calculates:* an absolute ceiling
on saturated runs; `n2_packing : W/33 <= n2`.  *Status:* `[kernel]` (`Corridor.exists_class_in_run`,
`both_composite_in_run`, `double_slot_in_run`, `prime_adjacent_run_le`, `n2_packing`);
escalation-checked through gear 23.  *Where:* [La] A; [Ar] lateral r8, formalist r9.  *Limits:*
about prime-carrying columns, not machine openings; column 1 is the unique class exception (the twin
5,7); whether `lim L0 = 32` is a finite check nobody has run.

**A21. Corridor mouth pins the landmark.**  *Statement:* the `(5,7)` corridor starts at
`k = 2 mod 35` and the `L* = 13` landmark sits at column 2452 `= 2 mod 35`, at the corridor mouth;
at gears `<= 17` and `<= 19` the extremal corridor's absolute start IS column 2452.  *Calculates:*
the address of the longest saturated stretch from small-gear geometry.  *Status:* `[exact: through
gears <= 23]`.  *Where:* [Ar] lateral r8.  *Limits:* through gear 23.

**A22. Unconditional load ceiling past the horizon.**  *Statement:* on a twin-free stretch,
`P_run <= L - minB(L)`: `L = 33/50/100/200/252` gives 0.970/0.920/0.910/0.880/0.873, asymptote
`1 - 730/5005 = 0.854`.  *Calculates:* a ceiling on prime load for any stretch length.  *Status:*
`[exact: proved from the B-class census]`.  *Where:* [Ar] lateral r8.  *Limits:* reality sits far
below (0.52 at `L = 100`), so it closes the `L > 32` frontier without a contradiction.

**A23. The endpoint law.**  *Statement:* a gap of length `G` runs between `a` and `a+G`, so
`a mod 35 in A(G) = {r in E : (r+G) mod 35 in E}`, `|A|` in 3..15; `G = 34 mod 35` forces
`a mod 35 in {3,18,33}`.  *Calculates:* candidate left-endpoint residues of a record stretch; a
`15/|A|` = 2-5x prune.  *Status:* `[kernel]` (`Corridor.endpoint_law`, `endpoint_law_34`);
`[exact: every gap in five full periods]`.  *Where:* [Ar] constructor r9, formalist r10.  *Limits:*
constrains WHERE only; measured concentration EXCEEDS the forcing (at m19 all twenty records sit at
the single residue 5 of nine allowed).

**A24. Adjacency law / 294 forbidden pairs.**  *Statement:* adjacent gaps `(G1,G2)` force
`a, a+G1, a+G1+G2` into `E`; the allowed set is empty for 294 of the 1,225 pairs mod 35.
*Calculates:* a machine-free table of impossible adjacent stretch-length pairs.  *Status:*
`[kernel]` (`decide +kernel`, no `native_decide`, 22 s).  *Where:* [Ar] constructor r9, formalist
r10.  *Limits:* 931 of 1,225 remain allowed; every observed `F_2`-realising pair sits inside its
allowed set.

**A25. Adjacent-gap exclusion law (mod 5), complete.**  *Statement:* three consecutive openings with
`(g1 mod 5, g2 mod 5) in {(1,1),(1,3),(2,4),(3,1),(4,2),(4,4)}` are impossible, 6 of 25 classes, in
every machine containing gear 5; by completeness only gear 5 can block a 3-point shape, so the list
is complete.  *Calculates:* a free prune on any adjacent gap pair (24% of classes).  *Status:*
`[exact: proved; full-period censuses m11..m31, 1,589 populated lag-1 cells, zero forbidden]`.
*Where:* [La] C; [Ar] lateral r20.  *Limits:* adjacent gaps only; at lag `>= 2` the same classes
carry up to 35.8M counts.

**A26. AP lemma (mod 5, scale-free).**  *Statement:* no four openings in arithmetic progression with
common difference `q'`, for every prime `q' > 5`; generalised -- four openings at pure multiples
`i q'` with the four `i` distinct mod 5 are impossible.  *Calculates:* kills `j = 2` and `j = 4`
literal links between two padded links, and `p = 3` all-adjacent padding, for every gear.  *Status:*
`[exact: proved, verified over all (r,g) mod 5 with g invertible]`.  *Where:* [La] C; [Ar] lateral
r16/r17.  *Limits:* silent on shapes that are not pure `q'`-multiples.

**A27. Openings AP theorem.**  *Statement:* an AP of `L` openings has common difference divisible by
every gear `q < L+2`; 3 equal gaps need `5 | g`, 5 need `35 | g`, 9 need `385 | g`, `L >= y+2` needs
the primorial.  *Calculates:* the longest equal-gap run (measured 3-4 everywhere, `g = 5` every
time).  *Status:* `[exact: proved; full periods m13..m29, zero violations]`.  *Where:* [La] C;
[Ar] lateral r18.  *Limits:* equal gaps only.

**A28. Strict-alternation cap = 6 (gear 5 alone).**  *Statement:* strict `LRLR...` saturated runs
need primes at alternating gaps 8,4,8,4,...; the offset residues cover `Z/5` at length 7 (L-first)
and 6 (R-first), so gear 5 caps strict alternation at 6 columns.  *Calculates:* an unconditional cap
on strictly alternating stretches.  *Status:* `[exact: proved; max strict alternation 6, at column
19125 at both scales, letters LRLRLR]`.  *Where:* [Ar] lateral r7.  *Limits:* strict alternation
only; repeats like the landmark's `LLLL` are the norm -- the constraint is CRT, not alternation.
### B. Two and more gears: the coincidence lattice

**B1. Pairwise coincidence is always exactly 4.**  *Statement:* for any two gears the joint machine
has period `qr`, `(q-2)(r-2)` openings and exactly 4 double-kill columns -- the CRT lifts of the
four sign choices; two same-sign (product blocks at `qr` and its mirror), two mixed-sign (crossed
kills).  *Calculates:* the exact positions of every jointly struck column, by CRT, with no scan.
*Status:* `[exact: worked in full for all six pairs from {5,7,11,13}]`.  *Where:* [Ar]
`pair-anatomy.md`; `twin-prime-program.md` 31b, 32b, 28b/28c.  *Limits:* counts are
pair-independent theorems; PLACEMENT is slip arithmetic -- the open question at pair level.

**B2. Composite-gear lift.**  *Statement:* every single-gear law lifts verbatim to the composite gear
`qr` (both-left tooth `6^{-1} mod qr`, low tooth `round(qr/6)`, joint shield at 0, family by sign
multiplication); the genuinely new object is the crossed pair, at
`X_crossed = 1 + q((-2 q^{-1}) mod r)`.  *Calculates:* both product-block columns and the crossed
columns in closed form.  *Status:* `[exact: pair table 5x7 .. 11x13]`.  *Where:* [Ar]
`pair-anatomy.md`.  *Limits:* only the same-side half is a lift; the crossed cloud is not reducible
to a single gear.

**B3. The accumulation model (composite gear + a crossed cloud of `2^n - 2`).**  *Statement:* a
machine of `n` gears is one composite gear plus `2^n - 2` mixed lifts, mirror-paired with sign
complement; all four triples from `{5,7,11,13}` have exactly 8 coincidence columns.  *Calculates:*
the complete coincidence set of any machine as a CRT sign lattice, with addresses.  *Status:*
`[exact: verified to n = 3, with worked lists]`.  *Where:* [Ar] `pair-anatomy.md`;
`twin-prime-program.md` 28c.  *Limits:* it gives the cloud's cardinality and addresses, not its
distribution inside the window.

**B4. Coincidence hubs.**  *Statement:* many-factor columns recur across machines -- column
`141 = (845,847) = (5x13^2, 7x11^2)` carries teeth of 5, 7, 11, 13 at once, so every triple drawn
from them coincides there; likewise columns 24 and 596.  *Calculates:* which columns are struck by
the most gears, i.e. the anchors that bracket long blocked stretches.  *Status:* `[exact: the four
triples from {5,7,11,13}]`.  *Where:* [Ar] `pair-anatomy.md`, `state-walk.md` finding 1.  *Limits:*
descriptive; hub-rate at the binding loci was later measured as generic (REFUTED, section 8.5).

**B5. The pair record falls; the machine record grows by accumulation.**  *Statement:* pair records
5, 4, 4, 4, 4, 3; triples 7, 7, 6, 6.  *Calculates:* the record is a property of the SET, not of the
largest gear.  *Status:* `[exact: all pairs and triples from {5,7,11,13}]`.  *Where:* [Ar]
`pair-anatomy.md`.  *Limits:* `n = 2` and `n = 3` only.

**B6. Two gears already generate every twin in their window.**  *Statement:* the `{5,7}` window is
columns `<= 8`; its openings 2, 3, 5, 7 are `(11,13), (17,19), (29,31), (41,43)` -- every twin in
the window.  *Status:* `[exact: machine {5,7}]`.  *Where:* [Ar] `pair-anatomy.md`.  *Limits:* the
base of the ladder, not evidence about larger machines.

**B7. The 5x7 map, mirror-folded.**  *Statement:* period 35, 15 openings, cyclic gap word
`2,1,2,2,3,2,5,1,5,2,3,2,2,1,2`, record 5; folding at 17.5 the map matches itself.  *Status:*
`[exact]`.  *Where:* [Ar] `pair-anatomy.md`.  *Limits:* the mirror halves the search, never more.

**B8. Roots-of-unity law.**  *Statement:* column `k` is hit by the pair `{q,q'}` iff
`36k^2 = 1 mod qq'`; trivial roots `+-1` are same-member columns, nontrivial roots are cross-member.
*Calculates:* the double-struck columns of a window by semiprime arithmetic with no primality tests
-- "doubles are one fixed subset of N".  *Status:* `[exact: verified both directions on the full
y=47 window]`.  *Where:* [Ar] agents-shared r2; [S] E; constructor R6.  *Limits:* it addresses double
kills, the complement of openings, pairwise.

**B9. Gap-graded split law.**  *Statement:* for `(q, q' = q+g)`, `m0 = (-2 q^{-1}) mod g`,
`b0 = (2 + m0 q)/g`, `i = (q'-b0) q^{-1} mod 6`, `x = (q'(b0+iq)-1)/6`, mirror at `P - x`; depth
`x ~ P(m0/g + i)/6`.  *Calculates:* the exact column at which any two gears cross-kill, in closed
form.  *Status:* `[exact: all 2850 pairs q < q' <= 400; 753,378 pairs at y = 10007, zero failures]`.
*Where:* [La] I; [Ar] lateral r3.  *Limits:* pairwise only; the multi-gear terms need the master
formula.

**B10. `g = 2` is the unique gap that pins at the bottom of every window.**  *Statement:* `m0 = 0`
iff `g = 2`, so `b0 = 1` identically and the split pins at `x = u' <= K` at every scale; every other
gap has floor depth `~P/(6g)`, reached only when the mod-6 alignment `i = 0` lands.  *Calculates:*
which gear pairs are guaranteed to coincide INSIDE the window (only twin gears), and where.
*Status:* `[kernel]` (`twin_pin`, `twin_pin_le`, `twin_split_class_iff`, `twin_mirror_slot`,
`twin_product_slot`, `own_slot_pin_gap_two`); `[exact: 81 twin pairs to 3000; uniqueness scan over
all pairs q < q' <= 400 found 20 own-column pins, ALL g = 2]`.  *Where:* [Ar] lateral r3, harvester
r2.  *Limits:* it pins a DOUBLE KILL, not an opening; and it is the machine's blind spot -- the pin
column has zero composite members yet is never a survivor of any machine with bound `>= p`.

**B11. Tooth-sharing classes of a twin pair.**  *Statement:* the four within-pair double-kill classes
mod `p(p+2)` are `{+-u', +-u'(p+1)}`; the mixed class IS the twin-product column
(`6u'(p+1) - 1 = p(p+2)`).  *Calculates:* the two deterministic marks a twin pair below `y` leaves on
the window above.  *Status:* `[exact: 60/60 twin pairs to 2000]`; `[kernel]` for the product slot.
*Where:* [S] E; [La] I; [Ar] lateral r1.  *Limits:* both guaranteed wasted kills land on
already-decided columns -- zero new openings; net gain `O(T(y))` per window against the needed
`~K/log^2`.

**B12. Redistribution law (sharing is positional, never cardinal).**  *Statement:* sharing a pair's
tooth phase changes expected in-window wasted kills by `1 - 2R/P` with `R = K mod P`; over full
periods sharing changes nothing.  *Calculates:* the in-window effect of a phase change, with the
full-period effect proved zero.  *Status:* `[exact/tested: 400 draws per configuration]`.  *Where:*
[Ar] lateral r1.  *Limits:* it is the reason no counting argument based on phase can work.

**B13. Slot-level sharing between two gears.**  *Statement:* two gears share a column in exactly 4
residue classes mod `qq'`; `12/5` of these per `qq'` are anchor-open; three gears share only mod
`qq'q''`; hit points coincide only at multiples of `qq'` (first `1517 = 37 x 41`).  *Calculates:* the
exact overlap and the waste: tooth density `sum 2/q` over 7..47 is 1.257 against a blocked fraction
0.745, so 0.512 of the teeth land on already-blocked columns, entirely by this rigid sharing.
*Status:* `[exact: slot_interact.py; nine gears 37..71 below 300000 give untouched 50,141 against
prod(1-1/q) x anchor-open = 50,165]`.  *Where:* [La] I; [Ar] `anchor-235.md` 4.  *Limits:* sharing
moves WHERE, never HOW MANY.

**B14. End-zone alignment of all the gears (a negative).**  *Statement:* gear `q`'s clean end zone is
`+-h_q` around every multiple of `30q` (`h_q = q`, or `7q` for the class `+-7`); zones drift by
`30(q'-q)` per period and realign at `30qq'`; a set stacks at `30 prod q`.  EXACT SEARCH over
`n in [1369, 10^7]` for all gears `37 <= q <= sqrt(n)`: **zero solutions**; closest calls -- missing
one gear's zone never past `n = 1680`, two never past 2478, three never past 3550.  *Calculates:*
where all gears' end zones would coincide.  *Status:* `[exact: search; expected fraction 3.1e-2 with
2 gears, 3.8e-14 with 15]`.  *Where:* [La] I; `anchor-235.md` 3.  *Limits:* a negative about one
particular alignment, not about openings generally.

**B15. Square roots of unity.**  *Statement:* the columns struck by every gear at once are exactly
the square roots of unity mod the period, scaled by `6^{-1}`; `2^n` of them.  *Calculates:* the
full-coincidence positions of any gear set.  *Status:* `[exact: sets of one to four gears; crossed
formula for all 28 pairs to 29]`.  *Where:* [Ar] `twin-prime-program.md` 28b-28c.  *Limits:* this is
the TOTAL-coincidence set -- the opposite extreme from an opening.

**B16. The whole machine as one gcd.**  *Statement:* `m` is open to every gear in `S` iff
`gcd(36m^2 - 1, prod S) = 1`; a twin column iff `gcd(36m^2-1, primorial(sqrt(6m+1))) = 1`.
*Calculates:* the opening condition as one arithmetic statement about one quadratic.  *Status:*
`[exact: 11,996 checks across five gear sets, zero disagreements]`; `[kernel]`
(`centreSurvivor_iff_twin`).  *Where:* [Ar] `twin-prime-program.md` 28d, 29a.  *Limits:* evaluating
the gcd needs the primorial -- a VALIDATOR, not a constructor.

**B17. The refinement law.**  *Statement:* for disjoint gear sets, `open(A u B) = CRT(open A,
open B)`, verified identical as sets (`15 x 99 = 1485`).  *Calculates:* the machine's openings by
combining sub-machines.  *Status:* `[exact]`.  *Where:* [Ar] `twin-prime-program.md` 17b.  *Limits:*
combining does not reduce cost -- the CRT is the primorial.

**B18. The generating polynomial.**  *Statement:* `p(x) = prod_q (q - 2 + 2x)`; the coefficient of
`x^j` counts columns struck by exactly `j` gears; `p(1) = P`, `p(0) = prod(q-2)`,
`p(1-k) = prod(q-2k)`, `p(0) - p(-1)` = number of opening runs.  *Calculates:* the whole `n`-wise
alignment lattice in closed form.  *Status:* `[exact: 30 gear sets of sizes 1-5, zero mismatches]`.
*Where:* [Ar] `twin-prime-program.md` 32b-32c.  *Limits:* `p(x)` is invariant under any relabelling
of residues -- it depends only on the multiset of gears, NOT on where the teeth sit, while the
distance to the next opening depends on phase.

**B19. The class tree, the turn law, non-extinction, the sound prune.**  *Statement:* adding gear `q`
splits each class into `q` children and kills exactly two, at
`t = (+-u_q - k_0) P^{-1} mod q`; `prod(q-2) >= 1` always; smallest-representative-first search is
correct and complete.  *Calculates:* the opening set of `{5..q}` from `{5..p}` by CRT, class by
class.  *Status:* `[exact: verified against brute force over all sub-machines of up to three gears
from 5..29]`; `[exact: proved]` for non-extinction and the prune.  *Where:* [Ar] `class-tree.md`;
`twin-prime-program.md` 17a-17b, 17e.  *Limits:* it controls OPENNESS, not POSITION.

**B20. The sideways-step obstruction.**  *Statement:* when a branch dies and the search steps to a
sibling, the sibling's smallest representative can jump by primorial-scale amounts; bounding the
sideways distance to the nearest open branch inside the window IS Reduction A.  *Calculates:*
nothing -- it names the quantity every alignment argument must bound.  *Status:*
`[exact: statement of the open problem, equivalence-strength]`.  *Where:* [Ar] `class-tree.md`;
`twin-prime-program.md` 1h.  *Limits:* the target, not a tool.  "Every route in the programme is an
attempt to bound the sideways step."

**B21. Machine slip versus cycle slip.**  *Statement:* the CYCLE SLIP `|P_S - P_T|` and the MACHINE
SLIP `P_S mod q` must be kept apart; the second is what composes and is the input to the turn law's
`P^{-1} mod q`.  *Status:* `[standing naming rule]`.  *Where:* [Ar] `handover.md` 0.3.  *Limits:* a
naming rule.

### C. Adding a gear: copies, merge, grammar

**C1. Lap/copy structure and the phase bijection.**  *Statement:* the new period is `q'` copies of
the old; copy `j` deletes the openings whose residue sits on a tooth as seen from that lap; the pair
shifts by `-P mod q'` per lap; `gcd(P,q') = 1` makes `j -> -u' - jP` a bijection, so every deletion
phase occurs exactly once and each opening dies in exactly 2 copies.  *Calculates:* which columns the
new gear's openings coincide with the machine's openings at, for every phase at once -- turning
"does this alignment occur somewhere" into "is there an admissible residue".  *Status:* `[kernel]`
(`AnchorChain.copy_phase`, `phase_bijective`, machine-free); the 2-copy count `[exact: m11..m23]`.
*Where:* [S] A; [La] D/F; [Ar] `gear-recursion.md` 3.  *Limits:* says where alignments CAN occur,
not which the old machine realises.

**C2. Merge law.**  *Statement:* every gap of `M+q'` is either a gap of `M` or the merge of a maximal
run of consecutive `M`-openings all struck by one phase of `q'`; deleting `k` merges `k+1` gaps, so
every new gap is a sum of consecutive old gaps.  With `span(w)` and `FS_max(w;M)`:
`F(M+q') = max(F_2(M), max over compatible w of [span(w) + FS_max(w;M)])` -- an identity, not a
ceiling.  *Calculates:* the new record from the old gap word plus `q'`, at `1/q'` of the cost of
rebuilding; the word list and compatibility depend on `q' mod 210` alone.  *Status:*
`[exact: proved + the whole gap histogram reproduced at four extensions; F = 18,25,34,43,58,88 at
six steps]`; `[kernel]` for the bound form (`MergeLaw.newgap_le`, `newgap_le_step`, `newgap_le_max`,
`Spectrum.merged_eq`) plus a four-rung hypothesis-free ladder (`Ladder.D_ladder`, 11->13 .. 19->23).
*Where:* [S] B; [La] D; [Ar] Part 8, lateral r13, constructor r12.  *Limits:* one-step -- it consumes
an `F_2` and a qualifying spectrum and produces neither, so rungs do not chain without a fresh input.

**C3. T1 -- the letters are the tooth differences.**  *Statement:*
`{2c, -2c} mod q' = {a, b}`, `a + b = q'`, `a = 2u'`, `3a = q' -+ 1`.  *Calculates:* the whole legal
alphabet at a step from `q'` alone: `Lambda(M) = {v <= F(M) : v = 0 or +-2c mod q'}`, about `3F/q'`
letters.  *Status:* `[kernel]` (`TwoTeeth.teeth_letters`); asserted at all 2,258 primes 11..20000.
*Where:* [S] A; [La] D.  *Limits:* the alphabet, not which letters occur as gaps of `M`.

**C4. T2 -- residue necessity.**  *Statement:* consecutive struck openings sit on `{+-c} mod q'`, so
every interior gap of a struck run is `0, +2c, -2c mod q'`, and a positive such gap is `>= 2u'`.
*Calculates:* a necessary test on every interior gap of a candidate merge, machine-free; the
qualifying floor `a = 2u'` for free.  *Status:* `[kernel]` (`MergeLaw.interior_gap_mod`,
`floor_of_mod`, `Machine23/29/31/37.merge_alphabet`).  *Where:* [S] A; [La] D.  *Limits:* necessary
only; the cover half is separate.

**C5. T3 -- strict alternation, padded letters transparent.**  *Statement:* nonzero-class spacings
strictly alternate, `|#a - #b| <= 1`; padded spacings keep the tooth; two consecutive nonzero letters
sum to `>= q'`.  *Calculates:* the grammar of alignment -- exactly two alternating words per length.
*Status:* `[kernel]` (`TwoTeeth.spacing_from_lo/_hi`, `next_kill_of_lo/_hi`,
`WordLegal.legal_iff_noRepeat`, `alt_iff_prefixSum`, `AnchorChain.no_two_up`, `no_two_down`);
asserted on every run of every full joint period at 11->13 .. 29->31.  *Where:* [S] A; [La] D;
[Ar] lateral r13.  *Limits:* grammar only -- the residue arithmetic alone bounds no length.

**C6. T4 -- deletion spacing, and its adjacent-frame form.**  *Statement:* every nonzero-class
spacing is `>= a = 2u'`, every padded spacing `>= q'`; in the adjacent frame two consecutive
deletions inside one lap are at least `q-1` apart, and that is tight (attained at 13 and 19); in the
column frame the minimum qualifying distance is `(q -+ 1)/3`.  *Calculates:* a stretch of length `G`
carries at most `1 + G/(q-1)` deletions; a chain of `k` needs span `>= (k-1)(q-1)`.  *Status:*
`[exact: proved (three-line mod-q case analysis); tightness measured 12/12, 18/16, 18/18, 24/22]`;
`[kernel]` (`TwoTeeth.kills_gap_ge`, `kill_spacing_min`).  *Where:* [S] A; [Ar] `gear-recursion.md`
4, `chain-conditions.md`.  *Limits:* a lower bound on spacing, not on how many spacings chain; it
bounds `k` only when `F_k(M) < q/2`, and the regime that matters has `F_k(M) >> q`.

**C7. T5 -- the fuel-span cap.**  *Statement:* `k <= 1 + span/(2u') <= 1 + 3 span/(q'-1)`.
*Calculates:* the arity ceiling of a merge from its span alone.  *Status:* `[kernel]`
(`TwoTeeth.fuel_span_cap`, `fuel_le`).  *Where:* [S] A; [La] D.  *Limits:* saturated only at 11->13
and 19->23; and `span/q'` grows along the ladder, so the cap grows.

**C8. The chain condition / chain law.**  *Statement:* adding gear `q` merges `k+1` old stretches
exactly when the `k` interior openings all lie in `{phi, phi+s} mod q`, `s = 3^{-1} mod q`;
equivalently two consecutive openings are both struck iff their gap is `0` or `+-d_g mod g`, and a
set lies in one two-class set iff every pairwise difference is `0` or `+-d`.  *Calculates:* the
record of the bigger machine from the old gap word (predictions 18, 25, 34 verified); the chain depth
per layer from the admissible gap list.  *Status:* `[kernel]` both directions, every gear
(`AnchorChain.chain_law`, `teeth_eq_phase`); `[exact: full periods {5..23}; independent k-frame
implementation at machines 13,17,19,23]`.  *Where:* [S] A; [La] F; [Ar] `chain-conditions.md`.
*Limits:* a condition on ONE gap; the DEPTH is not an algebraic consequence -- a run alternates
freely, so `D_g` is a fact about lower gap SIZES.

**C9. Hit law.**  *Statement:* gear `g` hops at the lower landing `x` iff `x = +-u_g mod g`; the
fraction of landings with no hit is exactly `1 - 2/g` (0.8182, 0.8462, 0.8824, 0.8947, 0.9130 at
`g = 11..23`).  *Calculates:* the layer's contribution to the walk as one residue test.  *Status:*
`[kernel]` (`AnchorChain.teeth_eq_phase`, `hop_zero`); `[exact: full period of every machine
{5..23}]`.  *Where:* [La] F; `anchor-235.md` 9d.  *Limits:* one layer at a time; the residual the
search cannot express is the lower walk.

**C10. The frame trap.**  *Statement:* the adjacent-frame chain condition `{phi, phi+1}` gives a
`k=2` count of `prod(q-4)` -- the domino count -- which exposed the error; k-frame teeth are never
adjacent.  *Calculates:* a sanity check on any chain/merge count.  *Status:* `[recorded error,
corrected]`.  *Where:* [Ar] `chain-conditions.md`.  *Limits:* a bookkeeping caution.

**C11. The maximal chains are the minimal alternation.**  *Statement:* all 62 `k=3` runs at
(gears `<= 19`, `q = 23`) have interior word `(8,15)` or `(15,8)`, span exactly `q`, 31 of each
orientation; a `k=4` run would need `(s, q-s, s)`, span `q+s`, or a gap of exactly `q` beside the
pattern.  *Calculates:* exactly which gap words permit a deeper merge.  *Status:* `[exact: complete
anatomy at (gears <= 19, q = 23)]`.  *Where:* [Ar] `chain-conditions.md`.  *Limits:* at that size.

**C12. The fuel census and the first `k = 4`.**  *Statement:* at gears `<= 23` the complete inventory
of 3-gap words permitting `k=4` is EMPTY; at gears `<= 29` and `q = 31` there are exactly FOUR
qualifying triples, all the word `(10,21,10)`, span `41 = q + 10`, in two mirror pairs.
*Calculates:* whether a deeper merge is possible at a given step by looking up the required word in
the current gap word.  *Status:* `[exact: machines <= 23, <= 29, <= 31, the last streamed over
period 3.34e10]`.  *Where:* [Ar] `chain-conditions.md` addendum.  *Limits:* "there is no universal
bound `k <= 3`"; chain length grows with the machine.

**C13. Fuel is sharply non-monotone in the new gear.**  *Statement:* at gears `<= 31` the `k=3` fuel
pairs are 70,964 / 2 / 0 / 0 / 224 and the `k=4` triples 216 / 0 / 0 / 0 / 0 at
`q = 37/41/43/47/53`; adjacency of the specific lift values is pure word-arithmetic (gap 16 next to
31 never occurs; 18 next to 35 occurs 224 times).  *Calculates:* which steps of the recursion will be
lumpy.  *Status:* `[measured/exact: machine <= 31]`.  *Where:* [Ar] `chain-conditions.md`.  *Limits:*
a hypothesis about the increment anomaly, supported by data, not proved.

**C14. Saturation theorem.**  *Statement:* if `q - 1 > F(M)` then `F(M+q) = F_2(M)` exactly; above
the threshold the increment does not depend on `q` at all.  *Calculates:* the new record with no scan
whenever the added gear is far (`{5,7}` plus any of 11..53 gives `F = 21` in the adjacent frame).
*Status:* `[exact: proved + 48 (M,q) pairs, zero violations]`.  *Where:* [S] B; [Ar]
`gear-recursion.md` 4b.  *Limits:* the compliant regime is PROVABLY DISJOINT from the consecutive
chain the route needs (`q' < F(M)` throughout).

**C15. Why chain length cannot be bounded mechanically.**  *Statement:* `(k-1)(q-1) <= (k-1) F(M)` is
vacuous whenever `F(M) >= q-1`, precisely the regime the consecutive chain lives in.  *Calculates:*
nothing -- it closes an approach.  *Status:* `[exact: proved-negative]`.  *Where:* [Ar]
`gear-recursion.md` 4b.  *Limits:* "bounding `k` needs the ARITHMETIC of which gaps fall within 1 of
a multiple of `q`".

**C16. Anatomy of the maximising chain.**  *Statement:* below the threshold every interior gap is
what the condition demands: `+17`: `[18] = 17+1`; `+19`: `[18] = 19-1`; `13+17`: `[33] = 2x17-1`;
`13+23`: `[24] = 23+1`; `17+19`: `[39] = 2x19+1`; `17+29`: `[30] = 29+1`; `19+23` `k=3`:
`[45,24]`; `19+31` `k=3`: `[30,63]`.  Excess `F(M+q) - F_2(M)` reads 15, 6, 6, 0, 3, 9, 18.
*Status:* `[measured/exact: seven steps]`.  *Where:* [Ar] `gear-recursion.md` 4b.  *Limits:* seven
steps; `k` never exceeds 3 in any maximum observed there.

**C17. The realisability CSP.**  *Statement:* a gap tuple occurs as consecutive gaps of `M` iff the
(open) and (cover) system on phase vectors is feasible; the period never appears.  *Calculates:*
exactly where alignments occur, from the list of primes alone; decided by `crt_dict.decide_cover`.
*Status:* `[exact: proved (CRT); gated on 2,013 tuples at m11/13/17 against a pruned
inclusion-exclusion counter, set-equal to full-period censuses (D_4(23), 15,696 tuples)]`.  *Where:*
[S] A; [La] J; `scanfree-certificate.md` 1.  *Limits:* the cover half costs `2^{|Y|}`; SHALLOW
queries are the dear end and DEEP queries the cheap end.

**C18. Phase saturation.**  *Statement:* with `FREE_q(X) = Z_q \ ((X mod q) u ((X - s_q) mod q))`,
`s_q = -2 x 6^{-1} mod q`: if `FREE_q(X)` is empty for some gear `q <= y` the word has NO occurrence
anywhere; `|FREE_q(X)| >= q - 2|X|`, so only gears `q < 2|X|` can fire.  *Calculates:* solver-free
zero verdicts (Constructor's m41 arity-4 superset screened `4,239,676 -> 2,814,574` in seconds, gear
5 killing 780,486 and gear 7 644,616).  *Status:* `[exact: proved, two lines; gated sound against
the entire realised-word record -- 37 words at five steps, 0 false kills on 291,675 realised m37
4-tuples]`.  *Where:* [S] C; [La] C; `phase-saturation-arity.md`.  *Limits:* it IS the corridor
condition mod 35 by CRT, so it adds NOTHING to a corridor-built abstraction -- it answered ZERO of
the 27,197 superset-YES queries at 41->43.

**C19. The walk screen.**  *Statement:* every point of the transfer's WALK -- struck interiors
included -- is an `M`-opening, so the whole walk must have an admissible phase at every gear; sound,
strictly stronger than the emission screen, and a prefix prune.  *Calculates:* superset sizes
`2,435,140 -> 1,182,475 -> 1,153,814` at 31->37 against a truth of 291,675.  *Status:*
`[exact: proved sound; asserted at every step that no realised tuple is removed; walk-screened ==
walk+emission at all six steps]`.  *Where:* [S] E; [La] C.  *Limits:* 2.4-11.7% of the superset; a
screen, not a decision.

**C20. Depth-0 / dictionary-monotonicity lemma.**  *Statement:* `D_m(M) subset D_m(M+q')` for every
prime `q' > 2(m+1)`.  *Calculates:* 145,907 of 874,087 reverse classes of the m41 arity-4 superset
are YES BY THEOREM (16.7%), at every span, with no solver.  *Status:* `[exact: proved, three lines;
seven pairs and arities 2,3,4,5; hypothesis SHARP -- first failure at m = 6,7,8,9 for
q' = 11,13,17,19]`.  *Where:* [S] E; [La] D.  *Limits:* PRIOR ART -- Ziller 2020 Prop. 2.7 is the
one-class arity-1 case; says nothing about which NEW tuples appear.

**C21. The two-class copy picture (Holt-Rudd in two classes).**  *Statement:* (a) each lower opening
is hit in exactly two copies; (b) a run of `j+1` consecutive lower openings with offsets `X` is
spared in exactly `q' - |X u (X+s)|` copies, and below `s_min` all `2(j+1)` hitting copies are
distinct; (c) a run of `k >= 2` is hit entirely in 0 copies if illegal, 1 if legal with a literal
letter, 2 if legal and all padded.  *Calculates:* the survival factor `q' - 2(j+1)` below the
threshold; the exact number of aligned phases for a candidate word with no search.  *Status:*
`[exact: proposition, three-line CRT; exhaustive at m11/m13/m17 and sampled at m19/m23; every
maximal run of >= 2 hits at m11..m23 -- 8/72/1,088/11,722/243,816 runs]`.  *Where:* [S] A; [La] D;
harvester r30.  *Limits:* the threshold is SHARP (4,6,6,8,10 at m11..m23) but every stretch that
matters is above it; and the multiplicity does not decrease with `k`, so the count can never bound
`L(M)`.

**C22. Firing law.**  *Statement:* the spacing word's first entry fixes the orientation and hence a
single firing residue, density `1/q'` per lap; across the new period every fuel site fires exactly
once, at `j = (fire - p) P_old^{-1} mod q'`.  *Calculates:* realised `k`-chains per new period
`= N_k` exactly.  *Status:* `[exact: zero violations over 13,062 sites]`.  *Where:* [La] D; [Ar]
lateral r12.  *Limits:* alignment is a DENSITY factor, never a COUNT factor -- there is no
suppression multiplier to find.

**C23. Firing is binary.**  *Statement:* every occurrence of a compatible word fires in exactly
`|valid starts|` of the copies, and incompatible words never fire anywhere.  *Calculates:* the
lower-bound half of the word-indexed identity -- "which is precisely why it is an IDENTITY rather
than an inequality".  *Status:* `[exact]`.  *Where:* [Ar] constructor r12.  *Limits:* none.

**C24. Layer law and horizon theorem.**  *Statement:* gears strictly below `y` decide the open
interior `(y, y^2)` exactly; the new gear's entire novel workload is `y^2` (owed iff `y^2-2` prime)
plus `y c` for primes `c in (y, y'^2/y)` -- one to three explicit columns per layer.  *Calculates:*
which gear can possibly strike a given column of the window; the short explicit list of semiprime
columns a layer owes.  *Status:* `[kernel]` (`Horizon.exists_prime_factor_lt`,
`prime_of_no_prime_factor_lt`, `twin_of_no_prime_factor_lt`; `Layer.layer_novelty`,
`minFac_lt_or_eq`, `eq_mul_prime_of_minFac_eq`); `[exact: verified y = 13..79; nine layers 13->17 ..
43->47, seven owing nothing in-band]`.  *Where:* [S] F; [Ar] `class-tree.md`.  *Limits:* an
existence/attribution statement, not a positional one; beyond `y^2` openness is not twinhood.

**C25. Shadow law.**  *Statement:* a gear supplies nothing below `q^2`; territory sizes collapse
upward (in the 13-window: gear 5 has 9 kills, 7 has 6, 11 has 2, 13 has 0 interior kills).
*Calculates:* the first column at which a new gear can act.  *Status:* `[kernel]`
(`Gear.sq_le_of_minFac_eq`, `Gear.R_eq_zero_of_below_sq`).  *Where:* [Ar] `class-tree.md`,
`proofs/Gear.lean`.  *Limits:* full-set sieving is provably equivalent to graded sieving inside a
window.

**C26. Composite root law.**  *Statement:* every squarefree product of set gears acts unshadowed
exactly once per window, at its own value, if it fits; six products fit the 13-window (35, 55, 65,
77, 91, 143) and no triple product does.  *Calculates:* the complete list of multi-gear coincidence
columns inside a window.  *Status:* `[kernel]` (`same_census_once`, `same_left_own_value`);
`[exact: 13-window worked in full]`.  *Where:* [Ar] `class-tree.md`, harvester r3.  *Limits:*
same-member coincidences only.

**C27. Graded depth and the necessity law.**  *Statement:* no subset covers the whole window (every
gear's square is an in-window root kill), but the window is GRADED -- gears `<= z` are exact on
columns whose members stay below `nextprime(z)^2`; gear `q` is NEEDED iff it owns a pseudo-twin in
the window, and droppability is transient.  *Calculates:* the exact minimal certifying set of a
window (`y=13 {5,7,13}`; `y=17 {5,7,11,13}`; `y=23 {5..19}`; `y=31 {5..29}`; `y=41 {5..37}`;
`y=47 {5..37,43,47}`; `y=59 {5..47}`); measured depth needed 7, 11, 15, 20 at `y = 41, 109, 197,
389`, averaging `0.42 sqrt(6y)`.  *Status:* `[exact: verified y = 13..59]`.  *Where:* [Ar]
`class-tree.md`, `research/minimal_subset.py`.  *Limits:* "the first twin sits close above `y`" is
an empirical input; proving that closeness would be STRONGER than Reduction A.

**C28. Square gate.**  *Statement:* descending exclusion of gears halts at the first `q` with
`q^2 - 2` prime (prime at 5, 7, 13, 19, 29, 37, 43, 47; composite at 11, 17, 23, 31, 41, 53, 59).
*Calculates:* how many top gears may be dropped from a window's certifying set.  *Status:*
`[exact: computation, correcting the earlier coprime-stopper hypothesis]`.  *Where:* [Ar]
`class-tree.md`.  *Limits:* about which gears are needed, not where openings sit.
### D. Padding: the second alignment mode

**D1. The padded link.**  *Statement:* a link is padded iff its two openings share a residue mod
`q'` -- the same tooth, one lap apart -- so its interior gap is `0 mod q'`, hence `>= q'`.
*Calculates:* the extra way a new gear's teeth align with two openings below.  *Status:*
`[exact: two verified witnesses at machine 31, q' = 37, with addresses]`.  *Where:* [Ar] mechanic
r14-15, harvester r12.  *Limits:* the shared residue is NOT `+-u'` (15 vs `u' = 31` in the example)
-- "the phase decides where they fire, not whether"; and SUPPLY is not LINKS (26,366 supply gaps at
31->37 give about 1,400 links).

**D2. Onset gate.**  *Statement:* `(0 < g) -> (q | g) -> (g <= F) -> q <= F`; so `q' <= F(M)` is
necessary for any padded link.  *Calculates:* the exact step at which padding can first appear;
`supply(M,q') = hist_M[q']` is one lookup.  *Status:* `[kernel]` (`TierA.onset_gate`,
`padding_at_most_one_below_onset`, `[propext]` only).  *Where:* [La] D; [Ar] formalist r18-19,
mechanic r14-16.  *Limits:* sufficiency is FALSE (`supply(29,41) = 0` despite `F(29) = 43`); a
prefix bounds the histogram only from below; `F < q` is the NO-PADDING regime, not "the onset
condition"; the padding COUNT bound `p <= F/q + 5/6` GROWS.

**D3. Padded cost per frame.**  *Statement:* a padded link costs `q'` in column units, `3q'` halved,
`6q'` in members; if `3 | e` the cheapest padded link costs `q'`, otherwise `3q'`.  *Calculates:*
the per-gap padding cost `c_d`, hence `p <= F(M+q')/c_d` and the onset condition `F(M) >= c_d`.
*Status:* `[exact: proved one line each way; measured for d = 6, 12, 30]`.  *Where:* [Ar] harvester
r11-12.  *Limits:* padding is 3x cheaper in absolute terms for `3 | e` but 1.5x in scale-relative
terms.

**D4. The padding lemma (two padded links need a spectrum value).**  *Statement:* two padded links
with `j` literal links between them occupy `j+2` consecutive gaps summing to at least `2q' + jL`, so
they require `F_{j+2}(M) >= 2q' + jL` for some `j`; headline `j=0`: `F_2(M) < 2q'` says two padded
links can never be adjacent.  Companion: if `2q' > F(M)` every padded link has size exactly `q'`.
*Calculates:* `p <= 1` per step from the spectrum; with `p <= 1` the run is `[literal] --q'--
[literal]`, so `k <= 12` and `span <= 5q' + 2s <= 6.35 q'`.  *Status:* `[exact: every computed step,
confirmed over full periods -- padded gaps 0, 0, 86, 6, 2090, 26367 at 13->17 .. 31->37, all of size
exactly q', 0 adjacent pairs, max 1 per run]`.  *Where:* [Ar] lateral r14.  *Limits:* it EXPIRES at
37->41 (`F/2q'` and `F_2/2q'` cross 1 there); it bounds the SPAN, not the increment.

**D5. The shape law.**  *Statement:* two padded links separated by `j` literal links: `j=0` feasible
50%, `j=1` 32%, `j=2` ALWAYS IMPOSSIBLE, `j=3` 4% of abstract pairs but 0 of 546 actual primes,
`j=4` ALWAYS IMPOSSIBLE; feasibility is a function of `q' mod 210`.  *Calculates:* the finite,
scale-free family of padded-run shapes; `span <= (4+p)q' + 2s`.  *Status:* `[exact: every prime to
4000; j = 2 and j = 4 proved by the AP lemma]`.  *Where:* [Ar] lateral r16.  *Limits:* the SHAPE law
is permanent, the COUNT `p` is not.

**D6. The 50/50 residue law (adjacent equal padded links).**  *Statement:* two adjacent equal padded
links need a 3-term AP `r, r+g, r+2g` inside `E_35` with `g = q' mod 35`; impossible for exactly the
12 classes `{1,4,6,9,11,16,19,24,26,29,31,34}`, with a perfect DICHOTOMY against the unequal shapes.
*Calculates:* for any new gear, whether adjacent padding is structurally possible, from `q' mod 35`.
*Status:* `[kernel]` (`TierA.no_adjacent_padded_41`, `equal_padding_forbidden_classes`, `_card = 12`,
`padding_shape_dichotomy` as an iff).  *Where:* [S] E; [Ar] lateral r15, formalist r16.  *Limits:*
adjacent EQUAL padded links only; it marks 41->43 as the first step with no obstruction of any kind.

**D7. Gear 3 forbids adjacent padded links for `d = 0 mod 6`.**  *Statement:* for `3 | e` the three
openings occupy all three classes mod 3 and gear 3 blocks one, unconditionally.  *Calculates:* an
unconditional grammar restriction for the densest Polignac gaps.  *Status:* `[exact: proved one
line; computed over all probes q' < 400 -- d=2: 34/74; d=4: 40/74; d=6 and 12: 74/74; d=30: 72/72]`.
*Where:* [Ar] harvester r13.  *Limits:* structural compensation -- padding is cheaper for
`d = 0 mod 6` but can never repeat there.

**D8. The 37->41 knife-edge.**  *Statement:* the `j=1` shape at 37->41 has two variants; literal 14
gives mod-35 offsets `[0,6,20,26]` (phases 12 and 32 OK), literal 27 gives `[0,6,33,4]`
(IMPOSSIBLE), so the census turns on `F_3(37) >= 96` against a measured prefix of 95.  *Calculates:*
reduces a full census question to one unit of one spectrum value.  *Status:* `[exact]`.  *Where:*
[Ar] lateral r15-16.  *Limits:* the corridor cannot settle the knife-edge itself.

**D9. The obstruction table.**  *Statement:* a shape is unobstructed iff corridor-feasible AND
spectrum-affordable; per step: 19->23 `j=0` cost 46 need `F_2` have 31 (short by 15); 19->23 `j=1`
54 vs 35 (short 19); 23->29 and 29->31 corridor EXCLUDES; 31->37 `j=0` 74 vs 68 (short 6); 31->37
`j=1` 86 vs 85 (short ONE); 37->41 `j=0` corridor EXCLUDES; 37->41 `j=1` 96 vs `>= 95` (short ONE);
41->43 and 43->47 `j=0` `F_2` OK.  *Status:* `[exact/measured]`.  *Where:* [Ar] lateral r17.
*Limits:* the two one-unit near-misses in a row are flagged as an observation, not a law -- "I have
no mechanism for that".

**D10. Padding is tier-blind.**  *Statement:* a run of `k` killed openings merges `k+1` gaps whatever
its letters, so `F_{k+1} >= F(M+q')` is padding-blind; what padding changes is FEASIBILITY.  TIER =
how many gaps merge; PADDING = whether the links connect.  *Status:* `[exact/structural]`.  *Where:*
[Ar] mechanic r14.  *Limits:* none; the 31->37 record needs BOTH `k = 3` and one padded link.

**D11. Padding is the gear-37 anomaly.**  *Statement:* at 31->37 the run census splits by padded-link
count: `z=0` 114,750,740 runs, max flanked span 71; `z=1` 26,366 runs, max 88 -- the true record
(`k=2`: 26,030 max 85; `k=3`: 336 max 88); `z >= 2`: 0.  Literal-only would give 71, not 88.  Winner
anatomy `[kill]--37--[kill]--12--[kill]` at `k = 9,463,664,103`, span `49 = q' + B`, flanks 28+11,
excess `20 = +0.541 q'`.  *Calculates:* the record of the bigger machine and which 336 runs in a
3.34e10-column period produce it.  *Status:* `[exact: full period, found independently by two
lanes]`.  *Where:* [Ar] lateral r13, mechanic r14.  *Limits:* the corpus's unexplained gear-37 spike
(2.432q against neighbours 0.22q and 0.84q) is exactly the first step whose winning word carries a
padded link -- a structurally different tier switching on, not a fluctuation.

**D12. Padded-lag law (`3q' +- 1` factorisation).**  *Statement:* the enhancement condition
`g = +-2u_q` is `3g = +-1`, so gear `q` is enhanced at lag `g` iff `q | 3g - 1` or `q | 3g + 1`;
`q' = 23: 68/70 -> 5,7,17`; `29: 86/88 -> 11,43`; `31: 92/94 -> 23,47`; `37: 110/112 -> 5,7,11`;
`41: 122/124 -> 31,61`; `43: 128/130 -> 5,13`.  *Calculates:* which gears enhance a given padded lag,
from a factorisation.  *Status:* `[exact: zero mismatches]`.  *Where:* [Ar] lateral r20.  *Limits:*
`sigma(q')` spans 3.3x while the measured supply share spans 330x -- endpoint arithmetic accounts
for about a tenth of the erraticity; the rest is the interior.

### E. The record: identities, algorithms, genealogy

**E1. Attainment theorem (R68) and the record law.**  *Statement:* a legal middle-gap word implies
`x_J - x_0 <= F(M+q')`; with the converse, `max(F_2, max_J Q*_J) = F(M+q')` exactly.  *Calculates:*
the record of the bigger machine from the smaller one, with `M+q'` never built.  *Status:*
`[exact: proved both ways; eight steps m11..m37; two out-of-scan confirmations 118 = F(47) and
145 = F(53); F(59) = 161 computed on machine 23's period, ratio 5.3e11; 27,570 counterfactual
machines, zero exceptions]`.  *Where:* [S] B; [La] D; constructor R68/R46, mechanic C24/C27/C35/C51.
*Limits:* `Q*_max` EQUALS `F(M+q')`, so the criterion is not a relaxation -- there is no slack in it
to exploit; its whole value is being computed on the old machine.

**E2. The exact record algorithm.**  *Statement:* `F(M+q') = max over k >= 1, over k-sites, of
o[i+k] - o[i-1]`; residues drop out entirely because every site fires once per new period; `k=1`
reproduces `F_2`.  *Calculates:* `F` of the next machine from the old opening sequence alone.
*Status:* `[exact: six steps -- 18, 25, 34, 43, 58, 88]`.  *Where:* [Ar] lateral r13,
`research/merge_correct.py`.  *Limits:* costs a scan of the old machine's full period.

**E3. The word-indexed identity.**  *Statement:* `F(M+q') = max(F_2(M), max over compatible w of
[span(w) + FS_max(w;M)])`, with `W(q')` determined by `q' mod 210`.  *Calculates:* the exact new
record from a word list depending on `q' mod 210` alone plus occurrence and flank data.  *Status:*
`[exact: 6/6 steps; binding words (4), (6), (13), (8,15), (10), (10); transfers verbatim to every
even d, 13/13, with tier_1 = F_2(M) exactly in every row]`.  *Where:* [Ar] constructor r12,
harvester r10.  *Limits:* `FS_max` is the sole open input.

**E4. The general-gap merge law.**  *Statement:* with teeth at `n = 0` and `n = -e`, `g = 0 mod q'`
is padded, `g = +-e mod q'` literal, anything else illegal, nonzero letters alternate; `F(M+q') =
max over legal runs of span`.  "Lateral's law with `2u` replaced by `e` -- the ONLY `d`-dependence."
*Status:* `[exact: 14 of 14 configurations, identity exact, 0 soundness violations, 0 firing
misses]`.  *Where:* [Ar] harvester r11.  *Limits:* wrap-around must be handled at absolute positions
over two periods.

**E5. Degenerate gear (`q' | e`).**  *Statement:* it has ONE tooth, the frame letter set collapses to
`3q'`, chains become plain APs, and `F(M+q') = F_2(M)` exactly.  *Status:* `[exact: both degenerate
cases]`.  *Where:* [Ar] harvester r10.  *Limits:* only the `q' | e` case.

**E6. Compatibility and the corridor never interact.**  *Statement:* `gcd(35,q') = 1`, so the tooth
condition is CRT-independent of the mod-35 carrier.  *Calculates:* the two filters may be applied
independently.  *Status:* `[exact: holds because q' > 7]`.  *Where:* [Ar] constructor r13.
*Limits:* none.

**E7. Phase-reduction record law (anchor form).**  *Statement:* on one lower period,
`D_g` = longest run of consecutive lower openings with residues in one two-class set, and
`F_bc(M+g) + 1 = max over such runs of (gap before) + (run span) + (gap after)`.  *Calculates:*
`F = 42` for `{5..29}` from 7,952,175 lower openings instead of a 6.5e9-column period (819x
smaller); 58, 88, 91 at 31/37/41 walking a 1.24e12-column period with no array beyond machine 29.
*Status:* `[exact: {5..7} .. {5..29}, and 31/37 full lower periods, 41 at 36.9% partial with both
headline answers still exact]`; `[kernel]` at machine 17 at both ends (`AnchorRecord17.record_max`,
`surv_shift`, `phase_is_machine`, `F17_eq_18`).  *Where:* [S] B; [La] F; `anchor-235.md` 9f.
*Limits:* the phase loop is avoided by mapping residues by `d^{-1}`; the reduction is CONCEPTUAL,
not economic, in a slot-walk kernel encoding (1.01x); the two ends are verified, not derived one
from the other.

**E8. The nested next-opening formula.**  *Statement:* `nextG x = nextM^[k+1] x` when the first `k`
lower openings after `x` are hits; term cap `D_g`.  *Calculates:* the next opening of the bigger
machine without materialising it; measured lazy cost `1 + hops above gear 5`, mean 2.4 at `{5..19}`
against 162 static terms.  *Status:* `[kernel]` (`AnchorChain.hop_zero/hop_iter/hop_one`);
`[exact: equal to the walk at every column on full periods {5,7} .. {5..19}]`.  *Where:* [S] B;
[La] F.  *Limits:* `prod(1 + D_g)` terms as a flat/nested form -- exponential in layers, no
cross-layer cancellation found; the scan form is quadratic but needs `F+1` as its term bound.

**E9. `D_g = A_kill`.**  *Statement:* the anchor line's chain depth equals the twin route's kill
arity at every gear where both exist (`D_17 = D_19 = 2`, `D_23 = 3`, `D_29 = 2`, `D_31 = 4`,
`D_37 = 4`, `D_41 = 3`); with R89, `D_g = L + 1`.  *Calculates:* either quantity from the other; a
streamed partial pass gives `D_g >= v`, a decided arity level `D_g <= A_kill`, and the halves meet
(`D_41 = 3` exact from 0.1% coverage).  *Status:* `[exact: identity by argument; 7 for 7 by two
vehicles built four rounds apart in different languages]`.  *Where:* [S] B; [La] F.  *Limits:*
`D_g` bounded is OPEN.

**E10. The word reduction R89.**  *Statement:* `Q*_J > -inf` iff `L(M) >= J-2`; `J_max = L+2`,
`A_kill = L+1`.  *Calculates:* the depth cap of the whole per-`J` family from the shallowest
dictionary the project has; every empty cell becomes a one-line dictionary fact.  *Status:*
`[kernel]` over an abstract opening enumeration (`WordLegal.chain_iff_word`, `qstar_iff_word`,
`jmax`, `akill`, one named periodicity hypothesis), instantiated at m11/m13/m17;
`[exact: 16/16 against the recorded rows]`.  *Where:* [S] B; [La] D.  *Limits:* it moves the open
question, it does not close it.

**E11. Same-tooth lemma R90.**  *Statement:* the middle span is `0 mod q'` exactly when the number of
non-padded middles is even; a literal even-`J` chain starts and ends on the same tooth with middle
span `>= ((J-2)/2) q'`.  *Calculates:* the even/odd split by arithmetic rather than by counting
parity.  *Status:* `[kernel]` (`WordLegal.same_tooth`, `same_tooth_window`, `literal_even_span`);
`[exact: 38 realised legal words, 0 violations]`.  *Where:* [S] B; [La] D.  *Limits:* literal chains;
a padded middle breaks the parity bookkeeping.

**E12. Middle-sum lemma (Theorem A) / flank-envelope collapse.**  *Statement:* middle sum `>= k q'`
(`J` even) or `>= k q' + a` (`J` odd), hence `Phi_J <= F_2 + s_min - m_min(J)`; at `J=5` the flanks
may sum to at most `F_2 - q'`, at `J=6` to `F_2 + a - 2q'`.  *Calculates:* how much room the two free
flanks of a deep alignment have -- and why the deep layers are the CHEAP ones.  *Status:*
`[exact: proved from T1-T3]`; measured `Phi_J <= F_2 - b` at every non-empty literal even-`J` cell,
margins +5, +10, +9.  *Where:* [S] B; [La] D; constructor R82.  *Limits:* literal middles only.

**E13. Peel bound (Theorem D) / free flank reduction.**  *Statement:* `Q*_J <= Q*_{J-1} + min(g_L,
g_R)`; equivalently `g_L + w + g_R <= F_2 + min(g_L,g_R)` with no hypothesis; read backwards,
`Q*_3 > F_2 + s_min` forces min flank `> s_min`.  *Calculates:* a hypothesis-free reduction of every
depth to the one below; discharges 6 of 8 steps; the residue is triples with both flanks above
`s_min` (0, 0, 16, 4, 24, 131, 205, 317 at m11..m37).  *Status:* `[exact: proved, hypothesis-free;
the flank consequence asserted at all 27,570 counterfactual machines]`.  *Where:* [S] B; [La] D.
*Limits:* does NOT reach `J >= 4` -- short by exactly `F_2 - a` (R55's 2F wall).

**E14. The par-trading residual `eps`.**  *Statement:* `eps(v) = Phi(u) - Phi(v) - x`;
`Delta_J = Delta_{J-1} - eps`, `Delta_2 = 0`; `eps(v) = d - g_out` with `d >= 0`.  *Calculates:*
exactly what one more aligned strike costs the merged record; "`Delta_J = O(1)`" is exactly "`eps` is
`O(1)` per letter AND `L` is bounded".  *Status:* `[exact: identity proved; decomposition asserted
30/30]`; `|eps| <= s_min` `[measured: 14/14 literal cells]` and REFUTED at 10/16 padded cells;
`max |eps| = 4` along maximising chains over 12 cells against `s_min` running 4..14.  *Where:*
[S] B; `even-j-mechanism.md`.  *Limits:* the six failures all carry the padded letter `q'`; unproved
in every direction; `eps = O(1)` is a CANCELLATION, not a smallness (`d = 27`, `g_out = 28` at m31).

**E15. The `F_3` wall.**  *Statement:* `Phi(q') + q' <= F_3(M)` trivially, with EQUALITY at m31,
whose `F_3 = 85` maximisers `(18,37,30)`/`(30,37,18)` have the padded letter as their middle; excess
`F_3 - (F_2 + s_min)` is `+1,+1,-3,-4,+1,0,+5,-7` at m11..m37.  *Calculates:* a decidable per-step
condition predicting where the increment law fails, and by how much.  *Status:* `[exact:
script-verified and gated, f3_middles_r30.py]`.  *Where:* [S] B; constructor r30.  *Limits:* a
residue event with base rate `3/q'`, so it WILL recur -- arithmetic luck per step, not a law;
`Phi(12,37) = 39` and `Phi(37) = 48` each rest on ONE occurrence.

**E16. Spectrum-plus-depth certificate.**  *Statement:* `F(M+q') <= max_{2 <= J <= J_max} F_J(M)`,
`J_max = A_kill + 1`; margin exactly `F(M) + q' - F_{A_kill+1}(M)`.  *Calculates:* the rung from the
OLD machine's spectrum over a finite depth range -- no word list, no flank envelope, no oracle.
Certifies 41->43 (+16) and 43->47 (+18).  *Status:* `[exact: proved; table verified at ten steps;
certifies 8 of 9 steps whose spectrum is complete]`.  *Where:* [S] B; [La] J;
`spectrum-depth-certificate.md`.  *Limits:* every `A_kill <= 3` step certifies (+10 to +24) and both
failures (29->31 by -11, 47->53 by -6) plus the single +3 squeaker are `A_kill >= 4` steps; and it is
circular below m59 (the `F_J` values are exhaustive only via deletion-ladder caps taken from `F`
above the step).

**E17. Deletion ladder.**  *Statement:* `F_j(M) <= F(M + the next j-1 primes)`.  *Calculates:* free
exact caps past the scan wall (`F_2(41) <= 103`, `F_2(53) <= 161`, `F_4(43) <= 161`).  *Status:*
`[exact: proved (three lines, CRT); asserted at all 32 (M,j) pairs where both sides are known, one
equality F_2(17) = 25 = F(19)]`.  *Where:* [S] B; [La] D; mechanic K3, constructor R93.  *Limits:*
LOGICALLY CIRCULAR as an induction step, and its slack thins to 0 at 41->43.

**E18. Lap-phase transfer.**  *Statement:* a maximal run of `M'`-openings is exactly a pair (run of
consecutive `M`-openings, phase tuple) meeting the endpoint/survivor/cover conditions.  *Calculates:*
`Q_J(M')` and `F_J(M')` exactly on `M`'s period at `1/(q_1...q_r)` of the cost -- the vehicle behind
`F(59) = 161`, `F_2(53) = 159`, `F_4(41) = 118`, `F_6(47) = 177`.  *Status:* `[exact: proved (CRT) +
two-sided anchors, m31 ladder 68/85/90/91/90/88 reproduced entrywise; every witness CRT'd to the
target machine and re-verified slot by slot]`.  *Where:* [S] B; `old-machine-spectrum.md`.  *Limits:*
a CERTIFICATION is conditional on the span cap, a FAILURE is not; with `r >= 2` the survivor-count
lower bound is not monotone, so the walk must stop on its RUNNING MAXIMUM.

**E19. The survivor identity / generator.**  *Statement:* with the SKIP letter `SIGMA`,
`F_2(M+q') = L K* SIGMA K* R` and `F_j(M+q') = L K* (SIGMA K*)^{j-1} R`; between two aligned runs
there is exactly ONE surviving opening.  *Calculates:* the whole low spectrum of the new machine from
the old machine's word, with the automaton fixed and only the word growing.  *Status:*
`[exact: proved for every j; verified exactly at j = 2 at six steps, matching an independent lag-1
pair census; F_2(M+q') = 16,25,31,39,55,68,90]`; `[kernel]` at 11->13 (`Gen11Sound.generator_sound`,
`F_1..F_4(13) <= 11,16,23,26`, machine 13's period nowhere in the derivation, `DepAudit.lean`).
*Where:* [S] B; [La] J; constructor R56/R57/R59.  *Limits:* `A_5` is needed where `A_4` suffices for
the plain system -- one more order of history per skip.

**E20. Kleene identity.**  *Statement:* `F(M+q') = L (x) K* (x) R` in max-plus; `K` nilpotent of
index `k_max`, `m`-th layer exactly `Q*_{m+2}`.  *Calculates:* every depth's aligned maximum from one
algebra, and the arity-free dual certificate (C1)-(C3).  *Status:* `[exact: verified at m11..m29]`;
the certificate direction `[kernel]`; the identity itself is not kernel-checked.  *Where:* [S] B;
`kleene-generator.md`.  *Limits:* it names NO truncation depth; at a fixed machine it is a finite
longest-path computation, and the open part is a closed form for `h` valid at every machine.

**E21. The potential.**  *Statement:* the budget inequality holds IFF a potential `h` exists with
(C1) `h(i,s) >= d_i`, (C2) `h(i,s) >= d_i + h(i+1,s')`, (C3) `d_{i-1} + h(i,s) <= F(M)+q'`.
*Calculates:* the whole ladder from one function per machine; `h11, h13, h17, h19` exhibited, tail
depths 4, 3, 5, 4 -- not growing.  *Status:* `[kernel]` for the certificate direction
(`Potential.IsPotential`, `chain_le_potential`, `D_of_potential`, `merged_le_of_potential`,
`Potential19`, `PotentialLadder`).  *Where:* [La] J; constructor R46; formalist 2.24.  *Limits:* the
CONVERSE is not formalised; a potential valid at every machine at once is not known.

**E22. Word-free (qualifying) criterion `Q_j`.**  *Statement:* `F(M+q') <= max(F_2, max_{j>=3}
qualmax_j)` with middles `>= 2u'` and T3-legal.  *Calculates:* the rung from three exact census
quantities of the OLD machine; margins `+4, +10, +9, +10, +13, +3` at 11->13 .. 29->31; and
`Q_j = 0` delivers the fuel cap for free.  *Status:* `[exact: full period m11..m31; equality at 7 of
8 steps]`; `[kernel]` (`MergeLaw.newgap_le_max`, `D_of_qualmax`, `Spectrum.merged_le_of_qual_flat_
all`), instantiated at five hypothesis-free rungs plus two dictionary rungs.  *Where:* [La] J; [Ar]
mechanic r17.  *Limits:* the ALL-DEPTHS form FAILS from 43->47 on (152 > 150; 177 > 171) with
exhibited witnesses; the margin collapses from `~0.45q'` to `0.10-0.11q'` at m29/m31; and it needs an
UPPER bound on `Q_j`, which prefixes cannot give.

**E23. The crux at 29->31.**  *Statement:* `F_5 = 85 = F + 42` fails; `Q_5 = 71 = F + 28` passes.
*Calculates:* the interior-gap floor alone -- one inequality, no compatibility, no residues, no
corridor -- brings 42 down to 28 and clears the budget with margin 3.  *Status:* `[exact: full
period]`.  *Where:* [Ar] mechanic r17.  *Limits:* margin `0.10q'` is the honest caveat in the other
direction.

**E24. `Q_j` is attained at the binding step.**  *Statement:* `Q_7(31) = 88 = F(37)` exactly; machine
31 `F_j = 85 90 92 97 104 110` against `Q_j = 85 90 91 90 88 0` for `j = 3..8`; machine 41 (prefix)
`max_j Q_j = 110` against `F + q' = 133`.  *Calculates:* how much the interior-gap floor removes at
each depth.  *Status:* `[exact: machine 31 full period, 3.343e10, 725 s; m41 from a 0.08% prefix]`.
*Where:* [Ar] mechanic r17 addendum 2.  *Limits:* prefix rows are lower bounds.

**E25. The tier table.**  *Statement:* per-step minimum chain length: 13->17 `k=2`; 17->19 `k=1`;
19->23 `k=2`; 23->29 `k=2`; 29->31 `k=2`; 31->37 `k=3` (the record 88 EXCEEDS `F_3(31) = 85`, and the
`k=4` chains reach only `<= 87`, so it is carried by `k=3` exactly).  *Calculates:* the minimum chain
depth that can carry the record, from the spectrum alone.  *Status:* `[exact: full period]`.
*Where:* [Ar] mechanic r13.  *Limits:* a threshold, not a density.

**E26. `k_win` and par trading.**  *Statement:* each extra link buys about `q'/2` of span and costs
about the same in flank sum, so the merged maximum is nearly independent of depth (spreads 0%, 8%,
6%, 14%, 12%; restated at round 20 as a band of ~25%); `k_win <= 3` at all seven measured steps,
winning words `(4), (6), (13), (8,15), (10), (10), (37,12)`; gain per link = spectrum increment
(5-15), loss per link = `lambda L` (4.2, 5.5, 9.0).  *Calculates:* which chain depth wins at a step,
with the exact address (`k = 137,307`; `14,995,460`; `278,620,515`).  *Status:* `[measured/derived
7/7, independently confirmed]`.  *Where:* [Ar] constructor r18-19, mechanic r17.  *Limits:* at
29->31 the `k=3` and `k=4` chains TIE at 55 while `k=2` wins at 58 -- fuel exists and LOSES by 3;
`k_win = 3` at m31 and `k_win = 1` at m41, so winners get SHALLOWER as machines grow.

**E27. Merge availability is not a function of fuel population.**  *Statement:* correlation(excess
share, `N3` per opening) `= -0.03` over seven steps; zero long-chain fuel still yields substantial
excess (23->29: `N3 = 0`, share 0.44) and huge fuel yields small excess (29->31: `N3 = 13000`,
`N4 = 4`, share 0.20).  *Calculates:* separates availability of merges from size of flanks -- `N2` is
ubiquitous (2-5% of openings at every step), so excess MAGNITUDE is set by flank quality and chain
length enters as a THRESHOLD.  *Status:* `[measured: seven steps]`.  *Where:* [Ar] mechanic r13.
*Limits:* seven steps.

**E28. The bridge / flank identity.**  *Statement:* an occurrence of a length-`ell` word plus its two
flanks IS `ell+2` consecutive gaps, so `span(w) + FS(occurrence) <= F_{ell+2}(M)` identically.
*Calculates:* converts the target from a statement about flanks into spectrum flatness at bounded
depth.  *Status:* `[kernel]` (`Spectrum.merged_eq`, `merged_le_spectrum`, `merged_le_of_shallow`);
`[exact: identity]`.  *Where:* [Ar] constructor r17, formalist r18-19.  *Limits:* needs an UPPER
bound on `F_j`; scans give lower bounds only.  The length-only ceiling is ATTAINED (machine 19, word
`(10,)`, `k = 137,328`, flanks (21,4): `21+10+4 = 35 = F_3(19)`), so no better length-only bound
exists.

**E29. Span-resolved envelope `H_ell(s)`.**  *Statement:* `span(w) + H_ell(span(w)) <= F + q'`
implies the budget inequality using only the word's LENGTH and SPAN; implied at all 44 measured
(step, word) pairs, including the former residual (`29->31, w = (10,21,10)`, span 41:
`41 + 24 = 65 <= 74`).  *Status:* `[exact: 44 pairs]`.  *Where:* [Ar] mechanic r17.  *Limits:* still
needs upper bounds on the envelope at prefix-only machines.

**E30. The record's shape and genealogy.**  *Statement:* of 132 stretches attaining `F_j` at
m19/m23/m29, ZERO are literal and ZERO are qualifying -- the shape is always two near-maximal flanks
with the machine's smallest gaps interior; and the ancestor of a record is a RUNNER-UP at 7 of 8
steps (deficit 2-14), whose largest gap was itself merged one machine down at 7 of 8 (12 of 12 for
`F_J` records), running 2-5 generations deep.  *Calculates:* "records do not recruit records" is
FALSE in the spectrum sense (1 of 8) and TRUE in the depth sense (7 of 8); the m31 tree is given in
full.  *Status:* `[exact: full-period censuses; every record column re-verified at its own machine]`.
*Where:* [S] E; [La] H; mechanic C54, `suppression-law.md` C.  *Limits:* ancestor RANK by span is
8-219, so `F(M+q')` cannot be computed from `M`'s top-`k` runs; `F(M+q')` is scan-free exactly when
the record is carried at depth `>= 3`.

**E31. Record gaps are isolated; the record grows from MEDIUM old gaps.**  *Statement:* neighbours of
every record gap are small (`(1,2)`, `(1,3)`, `(2,2)`, `(2,5)`, `<= 7` at the larger machines);
`F_2 - F = 2,2,4,5,7,6,5` against `q' - s_min = 5,7,9,11,13,15,19`; old-gap sizes under new maxima
run 0.16-0.68 `F_old` with chains `k = 2-3`.  *Calculates:* the depth-2 slack directly, and which
stratum to search.  *Status:* `[exact: full period at seven machines; measured y = 17..29]`.
*Where:* [La] H; [Ar] lateral r9, mechanic C25.  *Limits:* NO explanation is on record for why a
record-size gap of a CRT word has only small neighbours; and isolation does NOT explain `F_2`.

**E32. The record stretch is ordinary at the bottom, made at the top.**  *Statement:* survivors track
a random stretch at the bottom layers and the record is made at the top three or four layers, where
each gear removes 2-3 survivors a random stretch would keep; the top layer carries 1-3 survivors, ALL
on that gear's teeth, with differences in the chain classes.  *Calculates:* the layer at which a
record becomes a record; example `{5..23}` record 33: 19, 13, 10, 7, 5, 3, 0 against 19.8, 14.2,
11.6, 9.8, 8.7, 7.7, 7.0.  *Status:* `[exact: full period {5..11} .. {5..23}]`.  *Where:* [La] H;
`anchor-235.md` 9d-9e.  *Limits:* a description of the alignment, not a criterion for it; the
top-layer pattern does not repeat from rung to rung, only the shape does.

**E33. Where the record first shows.**  *Statement:* the records of `{5..13}` through `{5..19}` start
at columns 123, 118, 111 (numbers about 670-740) -- right after `19^2` the periodic word already
shows its record.  *Status:* `[exact: three machines]`.  *Where:* [La] H.  *Limits:* three machines.

**E34. The lower side of the record is forced.**  *Statement:* `F(M+q') >= F_2(M)` with no
computation; the best one-kill run equals `F_2` at every rung.  *Status:* `[exact: every rung
{5}+7 .. {5..23}+29]`.  *Where:* [La] H; mechanic K3.  *Limits:* a floor; `F' - F_2` is the whole
content.

**E35. Chain skeleton at the maxima.**  *Statement:* kill sides strictly alternate (R,L / L,R,L /
R,L,R in every case) and interior kill spacings are exactly `{2u', q-2u'}` (17: 6/11; 19: 13;
23: 8/15; 29: 10).  *Calculates:* the internal structure of a record from the new gear's teeth alone.
*Status:* `[exact: y = 17..29]`.  *Where:* [Ar] lateral r9.  *Limits:* literal chains only.

**E36. Interior gaps of the record chain are minimal-stride.**  *Statement:* they are exactly
`+-2u' mod q'`, never a multiple of `q'`; `F' - F_2 = 1,0,0,2,0,3,4` against `s_min = 2,4,4,6,6,8,10`
over the full period at all eight rungs.  *Calculates:* the increment over every kill chain, not only
the record.  *Status:* `[exact: full period, eight rungs, with chain counts]`.  *Where:* [La] H;
`anchor-235.md` 9.  *Limits:* about the REAL teeth; the delta-uniform form fails at small delta.

**E37. Increment law.**  *Statement:* `F(M+q') - F_2(M) <= s_min(q')`.  *Calculates:* an upper bound
on the new record from the old depth-2 record plus one residue.  *Status:* `[kernel]` at all six
LITERAL steps, both halves, hypothesis-free (`Increment.increment_law_literal_steps`, 1,749 jobs;
sharpness `f2_19_sharp`, `f2_23_sharp`, `f2_29_sharp`); `[measured: 11 of 12 corpus steps, failing at
the padded 31->37 by +8; confirmed out of sample at 53->59]`.  *Where:* [S] G; [La] D; formalist r28,
constructor R76, LP thread.  *Limits:* the base cases are kernel facts, the INDUCTION STEP is not,
and the LP vehicle cannot supply it; violated by 13-22% of the counterfactual family; the decisive
quantity `W_inc - F(q')` is negative at exactly one corpus step (31->37).

**E38. Triple inequality and `Delta_3`.**  *Statement:* `g_L + w + g_R <= F_2 + min(g_L,g_R)` with no
hypothesis; `Delta_3 = -3, 2, 0, 2, 4, 3, 2, 0`, bounded while `s_min` grows linearly.  *Calculates:*
the shape to aim at is `Delta_3 = O(1)`.  *Status:* `[exact: the reduction proved; the table
verified]`.  *Where:* [S] G; constructor r27.  *Limits:* the padded half is the tight one.

**E39. `Delta_J` bounded uniformly in `M` and `J`.**  *Statement:* every literal cell lies in
`[-3,+4]` at m11..m41 and the excess shrinks with depth; `J_max = A_kill + 1` at all eight censused
machines; confirmed out of sample at m53 (every `Delta_J` is `+2`).  *Calculates:* the whole
depth-`>=3` half of the increment law at a step is three inequalities.  *Status:* `[measured: 13
cells, exact per cell, three independent vehicles; NOT proved]`.  *Where:* [S] G;
`per-j-window-analogues.md` 1.5.  *Limits:* at m41 two cells are bounds, not values.

**E40. Palindrome dichotomy of the maximisers.**  *Statement:* unique up to reversal at every
measured cell; a reversal PAIR and never a palindrome at `J = 3, 4`; unique and SELF-REVERSE at
`J = 5` (`(7,10,21,10,7)` at m29, `(3,25,12,25,3)` at m31, each `Delta_5 = 0`).  *Calculates:* the
palindrome route applies at odd `J` only.  *Status:* `[measured: exhaustive per cell]`.  *Where:*
[S] G.  *Limits:* at even `J` the mirror lever has nothing to bite on.

**E41. Composition migration.**  *Statement:* the extremal `j`-stretch migrates to several medium
gaps as `j` grows: max element / sum at the argmax is 0.64 -> 0.51 (m17), 0.51 -> 0.53 (m19),
0.46 -> 0.43 (m23), 0.54 -> 0.35 (m29), with argmax compositions `[3,7,18]`, `[10,7,18]`,
`[23,4,23]`, `[35,20,10]`.  *Calculates:* why the isolation law does not control deep runs -- the
deep extremal runs never contain the record gap at all.  *Status:* `[measured: four machines]`.
*Where:* [Ar] constructor r19.  *Limits:* four machines.
### F. How deep an alignment can go: the caps

**F1. Spectrum bound on `L`.**  *Statement:* `L(M) <= 2T+1` (SIMPLE), `<= 2T+1-p` (letter-aware),
`<= max(2T, 2 floor((G-2-a_min)/q') + 1)` (PARITY), `T = floor((G-2)/q')`, `G = F(M+q')`; i.e.
`L <= 2F(M+q')/q' + 1`.  *Calculates:* the alignment depth from the record and the gear, metrically;
corpus row (PARITY) `1,1,2,3,3,3,5,4,5,5,5,5` against `L = 1,1,1,2,1,3,3,2,2,2,4,3`; beats EXPCAP at
m19/m37/m41/m43/m53.  *Status:* `[exact: PROVED, unconditional given R68 and T3; 12 corpus machines
(173 gates) and 165,584 counterfactual machines, zero violations]`.  *Where:* [S] C; [La] E;
`spectrum-bound-on-L.md`, lateral item 84.  *Limits:* `L` is `O(F/q')`, not `O(1)`; it closes the
chain only under `8F <= q'^2 - (eps+12)q' + 16`, true at 8 of 13 corpus steps, and that closure is
conditional on the open padded case (`c_A = 4` is a literal-letter constant).

**F2. Bare-word uniform cap.**  *Statement:* `L_bare(M) <= PSORD(q' mod 210) <= 5` at every machine;
`PSORD = 1` on 24 classes, 2 on 4, 3 on 14, never 4, 5 on `{37,53,83,127,157,173}`; `S = {PSORD <=
2}` has 28 of 48 classes.  *Calculates:* the bare half of the alignment depth from `q' mod 210`
alone; at m41/m43 every bare decision is FREE.  *Status:* `[kernel]` (`BareAlt.
bareAlt_inadmissible_iff`, `no_gapWord`, `no_bare_run`, `S_card = 28`, `psord_le_five`,
`psord_ne_four`, `inadmissible_iff_capC`), instantiated at m23/m37/m41/m43, three independent
vehicles agreeing element for element; also `c in S <-> capC c <= 3`.  *Where:* [S] C; [La] E;
`bare-word-uniform-cap.md`, constructor R103.  *Limits:* bounds `L_bare`, NOT `L` -- at m37/m41/m43
`PSORD = 1` while `L = 2`, at m53 `PSORD = 2` while `L = 3`; the deep words there are provably not
bare.

**F3. `L = max(L_bare, L_pad)`.**  *Statement:* trivially true; so requirement (B) is exactly
"`L_pad` bounded".  *Calculates:* splits the open crux and identifies m37, m41, m43, m53 as the
machines where `L > L_bare`, each excess the padded letter `q'`.  *Status:* `[exact: theorem]`;
`[measured: L_pad = 0,0,0,1,1,1,2,2,2,2,3,3 at m11..m53]`; `L_pad(53) = 3` is DERIVED.  *Where:*
[S] C; [La] E; constructor R104/R105.  *Limits:* nothing on record bounds `L_pad`; the term that
makes it the cover half is the ALPHABET SIZE `~3F/q'`.

**F4. Literal cap theorem.**  *Statement:* a literal chain has at most 6 members at every gear
forever; the cap is a function of `q' mod 210` -- 2 on 24 classes, 3 on 4, 4 on 14, 6 on
`{37,53,83,127,157,173}`, never 5.  *Calculates:* the fuel cap at any step from one residue, and the
finite word list of clause (A).  *Status:* `[kernel]` both ways (`LiteralCap.no_run_seven`, `s_eq`,
`literal_chain_le_six`, `cap_six_classes_sharp`; `LiteralCapTable.cap_table_maximal`,
`cap_table_realized`, `no_cap_five`, `cap_spectrum_counts`); verified against every prime `<= 5000`,
0 mismatches.  *Where:* [S] C; [La] E; [Ar] constructor r11, formalist r13.  *Limits:* LITERAL
chains only (padded runs escape it; "killed runs are bounded by 6" is FALSE); NOT a density fact
(over all 1,225 `(t,s)` pairs mod 35 the spectrum reaches 140); and it never predicts realised arity.

**F5. Polignac cap.**  *Statement:* the literal cap depends only on `gcd(e,105)`: 6 for six classes,
10 for `gcd = 15`, 12 for `gcd = 105`; **12 is the absolute ceiling over all Polignac
configurations, for every gear, forever.**  *Calculates:* the fuel bound for any even gap from one
gcd.  *Status:* `[kernel]` (all eight `cap_gcd_*` and `capOf_le_twelve` depend on NO axioms at all;
each cap checked sharp).  *Where:* [S] C; [Ar] harvester r10, formalist r17.  *Limits:* gears 3, 5, 7
only; gear 3 FILTERS the candidate list rather than breaking runs.

**F6. `mod-105 = mod-210`, and the two-phase walk.**  *Statement:* the cap is a function of
`q' mod 105` only, and for odd `q'` that is the same check as `q' mod 210`; `phi(105) = 48`.  A
literal chain is an interleaved two-phase walk of period 70 mod 35.  *Calculates:* reduces any gear
to one of 48 classes for alignment purposes.  *Status:* `[exact: exhaustive over every prime
q' <= 1200 coprime to 105; class invariance per d over ~300 primes each]`.  *Where:* [Ar] harvester
r9-r10.  *Limits:* odd `q'`; requires the gear-3 skip semantics.

**F7. Single-cycle reduction.**  *Statement:* the walk's state space `(position mod 105, parity)` is
a SINGLE cycle of length 210, so one walk of 260 steps from one start sees every state.
*Calculates:* replaces `105 x 2` starts by one (a 37x cut).  *Status:* `[exact: single-walk max run
= all-starts max run, zero mismatches over all 8 gap classes x 48 classes]`; the enabling lemma
`exists_mul_mod_eq` is `[kernel]`.  *Where:* [Ar] formalist r16-17.  *Limits:* not used in the end
(restricting starts to the exposed set was cheaper); kept as a reusable piece.

**F8. `A_relax` / PSORD (uniform alternation order).**  *Statement:* `A_relax(M) <= 5` for every
machine with `y >= 7`, and `<= 4` unless `q' = 37,53,83,127,157,173 (mod 210)`; the six exceptional
classes are exactly the litcap-6 classes.  *Calculates:* a machine-free cap on the alternation order
from one residue.  *Status:* `[kernel]` (`AlternationOrder.ps_min_le_five`, `ps_min_five_iff`,
`ps_min_four_iff`, `ps_min_counts 24/16/2/6`, `ps_max_eq_capC`); `[exact: all 48 classes,
cross-checked by a direct sweep of every prime q' < 20000 with all gears to 100; adding gears 11 and
13 refutes nothing further, 60/60 and 720/720]`.  *Where:* [S] C; [La] E; constructor R74, formalist
R29.2.  *Limits:* it caps a PROXY -- `A_relax` tests ONE candidate cycle, while `A_m` is nilpotent
only when every legal cycle is broken; `N(37) = 3 > A_relax(37) = 2` and `N(41) = 3 > 2`.

**F9. `psMax = capC` -- two invariants, one object.**  *Statement:* the phase-saturation order
(maximising convention) equals the corridor literal cap at all 48 classes.  *Calculates:* either
object from the other; distribution `{2:24, 3:4, 4:14, 6:6}`.  *Status:* `[kernel]`
(`AlternationOrder.ps_max_eq_capC`).  *Where:* [La] E; formalist R29.2.  *Limits:* the two differ
only in the QUANTIFIER over start letters, so `S` (28 classes) is not R74's order-2 set (24).

**F10. Alternation ceiling (closed form, no solver).**  *Statement:* for the pure alternation `A_k`,
`ceil(M -> q') = min{k : FREE_q(X(A_k)) = {} for some q <= y} - 1`, a closed form in
`(q' mod q, s mod q)` for the two or three smallest gears; ceilings `6, 2, 2, 2, 5, 3, 3, 4` at
31->37 .. 61->67 against measured `A_kill = 4, 3, 3, 3, 5, 4`.  *Status:* `[exact: script-verified,
gated; A_kill(47->53) = 5 sat EXACTLY at its ceiling]`.  *Where:* [S] C;
`phase-saturation-arity.md` Corollary.  *Limits:* bounds the ALTERNATION only; at every step whose
ceiling is 2 or 3 the answer is `ceiling + 1` and the lifting word is PADDED -- four steps, recorded
as a pattern, not a law.

**F11. CORRCAP.**  *Statement:* the longest T3-legal word with values `<= F` whose prefix-sum walk
stays in `E mod 35`: `4, 2, 3, 5, 25, 25, 11, 5` at 19->23 .. 47->53, and **INFINITE from 53->59
on**.  *Calculates:* the exact machine at which a bounded set of small gears stops capping alignment
depth -- and since `F/q'` grows without bound, no fixed set of small gears can ever cap the order
again.  *Status:* `[exact: an explicit automaton on the 35 x 3 corridor states with cycle detection;
R75's row reproduced 9/9 by two lanes]`.  *Where:* [S] C; [La] E; `uniform-order-bound.md`.
*Limits:* the term that makes it infinite is the ALPHABET SIZE `~3F/q'`; the bare alphabet stays two
letters forever, which is why `PSORD <= 5` is uniform.

**F12. EXPCAP and the sub-machine lemma.**  *Statement:* a word of length `m` survives phase
saturation at `M` iff it survives at `{g in M : g <= 2m+2}`; `EXPCAP` row `1,1,1,4,2,3,5,18,13,10,
5,21` at m11..m53.  *Calculates:* the exposure-only cap on alignment depth at any machine.
*Status:* `[exact: proved; asserted at every (M,m) cell m11..m53]`.  *Where:* [S] C;
`cover-half-counter-ladder.md`.  *Limits:* `EXPCAP - L` is NOT bounded (16, 11, 8, 18 at
m37/m41/m43/m53).

**F13. Cover-half order `N(M)`.**  *Statement:* the smallest `m` at which `A_m` is acyclic;
`max(2, A_relax) <= N <= A_res`; measured `N = 2,2,2,3,2,3,4,3,3` at m11..m41; the cycles that push
`N` above `A_relax` are PADDED 2-cycles that die at order 3 because T3-transparency is not
T3-legality once the run sees two literal letters.  *Calculates:* the abstraction order a
certificate needs, scan-free by CRT.  *Status:* `[exact: decided by CRT at m11..m41,
research/cover_order.py, reproducing R75's hand-computed row by a different vehicle]`.  *Where:*
[La] E; constructor R75/R85.  *Limits:* the vehicle stops at m43; whether `N` is bounded is open.

**F14. Tail-run cap.**  *Statement:* a `k`-chain's `k-1` interior gaps are consecutive gaps of `M`
all `>= 2u'`, so `k_max <= T(M, 2u') + 1` with `T` the longest run of consecutive gaps `>= 2u'`;
`T = 3, 2, 4, 3, 4, 5` across 11->13 .. 29->31.  *Status:* `[exact: theorem, one line]`.  *Where:*
[Ar] constructor r11.  *Limits:* loose (realised `k_max` 2,2,3,2,4) because chains also need residue
alignment.

**F15. The span law.**  *Statement:* same-residue openings are `>= q` apart and alternating distance
pairs sum to `>= q`, so a run of `k` openings has `span >= floor((k-1)/2) q`.  *Calculates:* an upper
bound on `k` from the record, in the saturated regime only.  *Status:* `[exact: proved]`.  *Where:*
[Ar] `chain-conditions.md`.  *Limits:* it bounds `k` only when `F_k(M) < q/2` -- "gap structure alone
cannot bound `k`".

**F16. Literal span law.**  *Statement:* the two primitive literal letters sum to the frame period,
so `k` letters span `ceil(k/2)` periods; with the cap table, literal span `<= ceil((cap_d - 1)/2)
q'` -- `<= 3q'` at cap 6, `<= 5q'` at cap 10, `<= 6q'` at cap 12.  *Status:* `[exact: verified,
route_transfer_audit.py]`.  *Where:* [Ar] harvester r13.  *Limits:* the constant degrades by at most
a factor 2 across ALL Polignac gaps.

**F17. Fixed-depth counters cannot bound `L`.**  *Statement:* `S^(2)_m = S^(4)_m = S_m` at all 21
measured cells (m19..m37) -- fixed-depth Bonferroni kills nothing -- while `min E_0/N` is 6..16 at
`m=1`, 845..10,742 at `m=2`, 145,158/312,151 at `m=3`, 4,344,055 at m37 `m=2`, growing in both `m`
and `M`.  *Calculates:* a uniform bound on `L` needs the cover half at FULL depth on a candidate set
unbounded in `M`.  *Status:* `[exact: proved for fixed-depth truncations given the measured EXPCAP
growth]`; "no counter of any kind" is labelled JUDGMENT, NOT RESULT.  *Where:* [S] H;
`cover-half-counter-ladder.md`.  *Limits:* the closed form `A_m = sum_k C(m,k) p^k T(m-k)` is proved
and asserted to `m = 6`; the first-moment threshold is 4,5,6,6,6 at m19..m37 against `L = 2,1,3,3,2`.

**F18. The killer profile.**  *Statement:* every decided kill of a one-letter extension is either
cover-only (`y* = 0`) or a corridor kill (`y* = 7` for the pure alternations, `y* = 5` from m37 on);
**no extension at any machine was attributed to the open constraint of a gear above 7**; the profile
is bimodal and empty in the middle.  *Calculates:* which half of the realisability CSP does the work
at each machine, hence where a proof of `L` bounded must go.  *Status:* `[measured: exact CRT
decisions at m19..m41, 0 realised and 0 undecided at the full machine; cover-only verdicts at
m19/m23 re-derived by a direct period scan]`.  *Where:* [S] C; [La] J; mechanic C53,
`legal-word-length-mechanism.md`.  *Limits:* m43/m47 not delivered; 2 at m37 and 10 at m41 refuted
but unattributed.

**F19. `L` is a density statistic; its last unit is not.**  *Statement:* an independent-letter model
with the real class densities and the T3 transfer matrix predicts the longest legal run to within one
unit at every scanned machine (3.7/3, 4.0/3, 4.0/2), while the COUNT of legal runs collapses at the
top (4 vs 279 at m29; 216 vs 1,610 and 0 vs 2.5 at m31; 27 vs 10,500 at length 2 at m37).
*Calculates:* `L` splits into a histogram statement plus a one-unit arithmetic collapse; the free
screens are exactly ONE LENGTH too generous at 7 of 8 next-prime cells.  *Status:* `[measured: exact
censuses, exact CRT decisions, one model used as an instrument]`.  *Where:* [S] C; mechanic r30.
*Limits:* the next prime is usually but not always the maximising gear.

**F20. The exact null for `L`.**  *Statement:* the legal-class probability under the machine's own
gap distribution is 0.19-0.39 of `3/q'` at m11..m37 (0.105 at m37); alternation costs a uniform
12-14%; dependence between consecutive gaps is a factor 0.43-1.00, not monotone.  *Calculates:* the
right null -- `I-actA/L` = 1.14, 1.31, 1.96 at m29/m31/m37 against 4.3-5.2 for the equidistributed
proxy.  *Status:* `[exact: two independent routes agreeing to three decimals; m41..m47 rows labelled
PROXY]`.  *Where:* [S] C; harvester r30.  *Limits:* at m13/m17/m23 the order-1 Markov null already
gives `E[L] = 1.00`; at m19 it moves only 3% of the way, so the dependence there is beyond lag 1.

**F21. Interior grammar: exactly two candidate words per chain length.**  *Statement:* a merge word
with `k` interior kills is side-alternating with spacing alternating `sigma`, `q - sigma`, so there
are exactly 2 candidates per `k`; abstracting parts to `c` classes, `|shapes(k)| <= 2 c^{k+1}`.
*Calculates:* the complete literal-chain shape list at each fuel level.  *Status:* `[exact]`.
*Where:* [Ar] lateral r11.  *Limits:* the interior grammar is finite iff `k_max` is bounded, and
`k_max` grows.

**F22. Boundary grammar: a finite a-priori superset with no stabilisation.**  *Statement:*
CRT-admissibility cuts the boundary shapes to a machine-independent superset of 3,798 half-shapes;
cross-machine full-shape recurrence is ZERO (0/24, 0/20, 0/102, 0/30, 0/22); observed halves are
3.2% of admissible and essentially disjoint per machine.  *Status:* `[exact: superset; negative on
stabilisation]`.  *Where:* [Ar] lateral r11.  *Limits:* a finite a-priori SUPERSET yes, an a-priori
list of OCCURRING shapes no.

**F23. The `k = 4` fuel-site census.**  *Statement:* over machine 29's full period there are exactly
4 sites with word `(10,21,10)` -- positions 220,171,102 (flanks 7,7), 406,081,827 (4,7),
672,200,337 (7,4), 858,111,062 (7,7), two mirror pairs -- and ZERO sites for the grammar's other
permitted word `(21,10,21)`.  The grammar allowed two words; arithmetic selection realises one.  At
31->37 there are 216 `k=4` sites in BOTH orientations.  *Calculates:* the exact addresses of the
deepest measured chains.  *Status:* `[exact: machine 29 and machine 31 full periods]`.  *Where:*
[Ar] lateral r11, mechanic r11.  *Limits:* `N5 = 0` everywhere scanned.

**F24. Word admissibility criterion and the finite word language.**  *Statement:* a side-word is
admissible iff some phase makes every prime side avoid every small-gear tooth; each position forbids
exactly one residue per gear and the per-gear allowed sets combine freely by CRT.  Language census
(gears `<= 13`): all `2^L` words admissible through `L = 4`, first exclusions at `L = 5` (`LLLLL`,
`RRRRR` -- gear 5's law), growth NOT exponential, plateau by `L = 18`, EMPTY FROM `L = 33` ON,
matching the B-gap horizon computed independently.  *Calculates:* how many side-words are possible at
each length -- a finite tree with a wall.  *Status:* `[exact: gears <= 13]`.  *Where:* [Ar] lateral
r8.  *Limits:* gears `<= 13`.

**F25. Word recurrence is CRT alignment.**  *Statement:* identical `L=8` words recur at position
differences divisible by 5 in 86% of duplicate pairs (baseline 20%), by 7 in 63% (17%), by 35 in 55%
(~3%); the forced-letter fraction is 0.729 measured against a crude CRT prediction 0.703; the six
`L=13` runs have six DISTINCT words at six different residues mod 35.  *Status:* `[measured: y =
3163, 10007; all 757 observed words admissible, zero failures]`.  *Where:* [Ar] lateral r7-r8.
*Limits:* gears `<= 13`.

**F26. Mirror and parity laws on words.**  *Statement:* the positional mirror reverses order and
swaps L/R, so REVERSE-COMPLEMENT is the machine symmetry (TV distances 0.328 vs 0.564 vs 0.600 at
`L=8`); PARITY THEOREM -- an odd-length word cannot equal its own reverse-complement, so odd-length
saturated runs are NEVER self-mirror (0 observed, forced), while even-length self-mirror runs are
common (16 of 250 at `L=8`).  *Status:* `[exact: parity proved; the distribution measured]`.
*Where:* [Ar] lateral r7.  *Limits:* parity only for the proved half.

### G. The mirror: the one exact symmetry

**G1. The involution.**  *Statement:* the opening set is exactly closed under `k -> -k`; `k = 0` is
its only fixed column; on indices `o_t -> o_{N-t}`.  *Calculates:* a free exact consistency check on
every census and a factor 2 on every reversal-invariant search.  *Status:* `[kernel]`
(`Mirror.mirror_gear`), instantiated at m11 and m29.  *Where:* [S] D.  *Limits:* `Z/2` and nothing
more.

**G2. The full symmetry group is `Z/2`, exactly.**  *Statement:* the affine maps preserving the
opening set are the `2^m` sign multiplications, only `c = +-1 mod P` preserving adjacency; dropping
affineness, only the identity and the mirror survive; `#fix(sigma_S) = N / prod_{q in S}(q-2)`.
*Calculates:* the ceiling on any parity lever -- exactly one unit, a factor of two, never four.
*Status:* `[exact: proved + brute-forced over all 92,400 affine maps at m11 and all 2P rotations and
reflections at m11/m13]`.  *Where:* [S] D.  *Limits:* a finer parity must come from something that
is NOT a symmetry of the opening set.

**G3. Maximal gaps occur an even number of times.**  *Statement:* `W_1(g)` is even for every
`g >= 2`; only the count of gaps of size 1 is odd; record gaps come in mirror pairs summing to
`P - F`.  *Calculates:* parity of every gap-length count -- a counting argument that caps a
configuration at ONE proves there are NONE.  *Status:* `[exact: proved + measured; record
multiplicities 12,20,20,4,2,4,[2],[4] at m13..m41, all even]`; kernel halves in
`Mirror.even_card_involution`, `window_count_even`, `none_of_at_most_one`, instantiated at m11.
*Where:* [La] B; [S] D.  *Limits:* worth exactly one unit.

**G4. The self-mirror stretch, located.**  *Statement:* exactly one self-mirror run per depth, at
`t_j = -j/2 (mod N)`, span `2 o_i` (`j` even) or `P - 2 o_{M-i}` (`j` odd); corollary
`g_j* = j (mod 2)`.  *Calculates:* `g_j*` scan-free at every machine from a few dozen sieved columns
(table to m53, `j <= 12`); at depth 2 it is `2 d_0`.  *Status:* `[exact: proved; verified against the
exact full-period `W_j` censuses at m11..m29 for every j <= 12]`.  *Where:* [S] D; [La] B.  *Limits:*
the LEVER is kernel-checked over an abstract index involution; the INSTANTIATION exists only at m11.

**G5. The self-mirror stretch is never word-legal at depth `>= 3`.**  *Statement:* odd `J` -- the
central middle is the antipodal gap, length 1, and 1 is legal only if `3 = +-1 mod q'`; even
`J >= 4` -- the two central middles are both `d_0`, forbidden by T3 and by `0 < d_0 < q'`; `J = 2` is
the one depth needing `d_0 != F`.  *Calculates:* `R_J` is fixed-point-free on the word-legal family
at every `J >= 3`, so every span count is EVEN with no exception list and no census.  *Status:*
`[exact: proved; gated at m11..m23, J = 2..7, 185 assertions]`.  *Where:* [S] D; [La] B.  *Limits:*
buys ONE UNIT, never four.

**G6. The mirror on records and in transfer coordinates.**  *Statement:* `k' = (P - k - s) mod P`
with reversed interior offsets and flanks, residues `r -> (P - r) mod q''`, `k + k' + s = P`.
*Calculates:* a factor 2 on every transfer sweep, and a PARITY CONSTRAINT -- a search that has found
one maximiser is provably incomplete.  *Status:* `[exact: proved + all 24 exact record stretches on
file (150 gates); the two `F_2(59)` maximisers an exact mirror pair including flanks, kernel-checked
as `CrtSlots.mirror_59`]`.  *Where:* [S] D.  *Limits:* no inequality on `Q*_J` or `F_J`.

**G7. Word reversal and the fixed-point criterion.**  *Statement:* `#occ(w) = #occ(reverse w)`
exactly; for a palindromic tuple of span `s`, `#occ(w)` is odd iff `w` occurs at
`k_w = -s/2 (mod P)`.  *Calculates:* decide one word per reverse class and copy the verdict; the
audit found 46% of 27,946 s had been spent deciding the SECOND member of a reverse pair.  *Status:*
`[exact: proved; gated on the exact 4-tuple dictionaries at m23/29/31/37 and on two CRT transfer
supersets]`.  *Where:* [S] D.  *Limits:* word reversal is the SAME involution, not a second one.

**G8. `F_2 >= 2 d_0`.**  *Statement:* the two gaps around column 0 are `(d_0,d_0)`.  *Calculates:* a
lower bound on `F_2` from one sieved column, and the exact missing gap a linearly-closed census
drops.  *Status:* `[exact: theorem + closed form; gated at 15,217 counterfactual machines]`.
*Where:* [S] D; [La] K.  *Limits:* on the counterfactual family the ONLY depth-2 failure in 14,616
exhaustively enumerated machines is exactly this stretch (`d_0 = 25` against `F = 26`).

**G9. Mirror closure of record stretches.**  *Statement:* the set of maximal-gap intervals is closed
under `k -> -k` at every machine tested; no gap straddles 0; merged gap words appear mirrored
(`(4,8,15,7)`/`(7,15,8,4)` at y=23; `(10,10,23)`/`(23,10,10)` at y=29); 4-20 maximal gaps per period.
*Status:* `[exact: full periods to y=23; streamed 1.078e9 for y=29]`.  *Where:* [Ar] lateral r9,
constructor r9.  *Limits:* mirror pairing of SITES does not imply mirror pairing of realised chains
-- the machine-29 mirror does not commute with gear 31's teeth.
### H. Counting identities, histograms and addresses

**H1. Depth-sum identity.**  *Statement:* `sum_{j >= 1} W_j(g) = prod_q c_q(g) = N2(g)` -- every
ordered pair of openings at lag `g` is the endpoint pair of exactly one run.  *Calculates:* an exact
CRT count of ordered opening pairs at every lag, hence the depth-uniform bound
`W_j(g) <= prod_q c_q(g)` with no period scan, at machines far beyond any scan.  *Status:*
`[exact: proved, one line; integer-exact m11..m29 for g = 1..64]`; `[kernel]` in both halves at m13
(`DepthSum.window_depth_unique`, `depth_partition`, `local_factor_{5,7,11,13}`, `depth_sum_at_13`,
`depth_sum_hl_form`).  *Where:* [S] E; [La] I.  *Limits:* **PRIOR ART -- Holt, arXiv:2502.20470,
Corollary 1**, specialised to the constellation `(2, 6g-2, 2)`; the identity and derivation stand,
the novelty label was wrong.  The kernel GLUE (index range vs residue range) is not built.

**H2. The local factor `c_q(g)`.**  *Statement:* `c_q(g) = q-2` if `q | g`; `q-3` if
`g = +-2u_q mod q` (the literal-link lag); `q-4` otherwise -- **the three cases of the
autocorrelation are the three tooth-relationships.**  Equivalently
`c_q(g) = q - nu_q({0,2,6g,6g+2})`, the Hardy-Littlewood prime-quadruplet local factor.  The
`n`-point form is `c_q(d_1..d_n) = q - 2n + O`, exact whenever `q >= 2n`, and "`c_q > 0` forced when
`q > 2n`" IS the completeness lemma.  *Calculates:* the CRT count of any prescribed alignment; the
admissible endpoint phases mod 35 are exactly `c_5(g) c_7(g)` in `{3..15}`, a five-fold swing driven
by the two smallest gears.  *Status:* `[exact: proved; gears 5..31 at all lags, zero mismatches;
16,500 brute-force checks for the n-point form, gears 5..43, n = 1..5]`; `[kernel]` at four gears.
*Where:* [La] I; [Ar] lateral r18/r20; harvester 5f.  *Limits:* ENDPOINT arithmetic only -- endpoint
exposure is a CONJUNCTION and factorises by CRT; the INTERIOR condition is a DISJUNCTION and does
NOT.  "That is why it stopped there."

**H3. What `c_5 c_7` explains about missing stretch lengths.**  *Statement:* machine 23,
`g = 24..31 -> counts 0/1404/310/170/322/6/112/20` with `c_5 c_7 = 3/9/4/6/10/3/12/3`; gap 24
(absent at m19 and m23) and gap 29 (count 6 between neighbours 322 and 112) both carry the MINIMUM
value 3; three of the four gap values absent below `F` across three machines carry the minimum.
*Calculates:* which stretch lengths are structurally suppressed; adding `log(c_5 c_7)` to a
smooth-decay fit raises `R^2` from 0.449 to 0.463 (m19), 0.856 to 0.896 (m23), 0.913 to 0.934 (m29).
*Status:* `[measured: full periods m19, m23, m29]`.  *Where:* [Ar] lateral r18.  *Limits:* residual
demand vs purchasable supply leaves slack 8-16 at every `g`, so gap 24's absence is selection plus
rarity, NOT a covering obstruction.

**H4. The paired Holt recursion.**  *Statement:* `n_g(M+q') = sum over words w with sum(w) = g of
coef(w) n_w(M)`, `coef(w)` position-free, so the map is linear with universal coefficients; at
`j = 1`, `coef(g)` is exactly the exposed-set autocorrelation `c_{q'}(g)`.  Generic word survival is
`q' - 2(j+1)` against Holt's one-class `q' - j - 1`: **the paired system contracts twice as fast per
unit of word length.**  *Calculates:* the whole new gap histogram from the old word census, exactly.
*Status:* `[exact: four rungs including the full WORD-census level (6,714 words at 5005->85085;
10,489 at 85085->1616615); paper proof elementary]`.  *Where:* [S] E; `paired-holt-recursion.md`.
*Limits:* population dynamics, not extremes; the `q' - 2j - 2` diagonal is a consequence of Holt's
point count in his 2025 general dynamics.

**H5. The interior disjunction, expanded and self-pruning.**  *Statement:*
`density(gap exactly g) = sum_{T subset interior} (-1)^{|T|} D({0,g} u T)`, `D(S) = prod_q c_q(S)/q`
-- an alternating sum of exposed-set correlations, every term in closed form, with Bonferroni
truncation giving rigorous two-sided bounds.  It prunes its own expansion: `c_5(S) = 0` whenever the
point set occupies 4 or more residues mod 5 (`g=20`: 36% pruned at depth 2, 70% at 3, 89% at 4, 97%
at 5).  *Calculates:* the exact density of a gap of length `g`, term by term.  *Status:*
`[exact: object; measured convergence against exact full-period counts at m19]`.  *Where:* [Ar]
lateral r20.  *Limits:* depth needed grows like `g/4` -- "Brun's problem in the machine's own
language, quantified".

**H6. The renewal ladder.**  *Statement:* for ANY subset `Y` of the interior offsets,
`#{k mod P : X open, Y blocked} = sum_{T subset Y} (-1)^{|T|} prod_q c_q(X u T)`; every choice gives
a VALID upper bound, and nesting gives a monotone ladder from the exposure bound to the exact count
at cost `2^{|Y|}`.  *Calculates:* a rigorous bound on how often a prescribed alignment occurs, at any
budget; three rungs (`s = 3` points per gap) already clear the route's requirement at every
constrained case.  *Status:* `[exact: proved (inclusion-exclusion + CRT) + every bound asserted `>=`
the exact full-period census where one exists]`.  *Where:* [S] H; `renewal-ladder.md`.  *Limits:* the
requirement it is checked against still carries a fitted constant `lambda`.

**H7. The multi-lag exposure bound.**  *Statement:*
`p_j <= (1/rho) sum over qualifying tuples prod_q c_q(0, v_1, v_1+v_2, ...)/q`, dropping only the
"no opening strictly between" condition.  *Calculates:* a rigorous upper bound on the
qualifying-stretch rate from exposure arithmetic alone -- the only inequality in the route with no
heuristic step.  *Status:* `[exact: rigorous inequality; measured SHORT by a factor 2-29]`.  *Where:*
[Ar] constructor r20.  *Limits:* the missing factor is exactly the dropped condition -- THE RENEWAL
FACTOR, named as the entire remaining gap and not built.  The `1/rho` conditioning correction
(stretches are per-opening, the exposure product per-column) is a bookkeeping law that reverses the
verdict.

**H8. The anti-correlation law.**  *Statement:* `R(lag) = P(both gaps qualifying)/p_1^2` is an
ADJACENCY EFFECT and nothing more: a strong deficit at lag 1 (EXACT ZERO at m11..m17), a rebound
above independence at lag 2, independence restored by lag 4-5; higher orders are
super-multiplicative (`p_5/p_1^3 = 7.1e-4` at m29 against `0.148^2 = 2.2e-2`).  *Calculates:* the
joint qualifying probability at any lag -- the object the suppression law needs.  *Status:*
`[measured: joint gap-pair census, m11..m29]`.  *Where:* [Ar] constructor r19-r20.  *Limits:* no
closed form; and the target needs far less -- independence alone clears every constrained case by
170x to 201,381x.

**H9. Suppression law and suppression-corrected flatness.**  *Statement:*
`suppression(j) = F_j - qualmax_j ~ lambda (j-2) ln(1/p_1)`, so the budget inequality follows from
`F_j - F <= q' + lambda (j-2) L` for every `j`.  ALL 15 machine-depth pairs hold (corrected values
4.7-15.1, bounded and non-growing) where RAW flatness fails at 5 of 15.  *Calculates:* the target as
a single depth-indexed inequality over quantities computable from `M`'s gap word alone.  *Status:*
`[measured/derived: r19; machine 29 observed suppression 7, 15, 30 against predicted 9.0, 21.7, 42.5
-- right scale, conservative at depth]`.  *Where:* [Ar] constructor r19.  *Limits:* `lambda` is
fitted, `p_1` measured, and the order-statistics step heuristic.  Consequence on record: the `j = 2`
case IS lemma 1, so lemma 1 and the deep-run problem are ONE statement at different depths -- and the
deep cases are the EASIER ones, reversing what the route assumed from round 8 to round 17.

**H10. Shallow flatness.**  *Statement:* a winner with `k <= 3` spans at most 4 consecutive gaps, so
the target follows from `k_win <= 3` and `F_4 - F <= q'`; measured `F_4 - F = 11, 15, 15, 13, 24, 27,
32` against `q' = 13, 17, 19, 23, 29, 31, 37`, ratios 0.57-0.88, flat at 0.79-0.88 for six of seven
machines with no downward trend.  *Status:* `[measured 6/6 and 7/7; neither half proven; m37/m41 are
prefix lower bounds]`.  *Where:* [Ar] constructor r18, mechanic r17.  *Limits:* subsumed by the
suppression law at r19.

**H11. Strict ordering of the three lemmas.**  *Statement:* Wall V clustering (`F_2 - F = O(q')`)
implies SPECTRUM FLATNESS implies the budget inequality (whose family has relative density
`~(3/q')^{k-1}`).  *Calculates:* places any candidate lemma in the hierarchy -- and since the middle
one is measured FALSE, the target "cannot be weakened further by dropping position information".
*Status:* `[exact: implications; the middle one measured false]`.  *Where:* [Ar] constructor r17.
*Limits:* none.

**H12. The compatibility restriction is load-bearing.**  *Statement:* deep flatness gives `F + 42`
where only `q' = 31` is allowed; the true increment is 15.  *Calculates:* what the residue
restriction is worth -- "any attempt to prove (D) by discarding the restriction loses the step it
most needs".  *Status:* `[exact: from the measured spectra plus the fuel census]`.  *Where:* [Ar]
constructor r17.  *Limits:* it refutes deep spectrum flatness.

**H13. The flank order-statistic law.**  *Statement:* `maxflank(w) ~ 2.05 ln(occ(w))` (sd 0.27) and
`FS_max(w) ~ 2.77 ln(occ(w))` (sd 0.24), with 2.77 matching `lambda = 2.73` fitted independently from
the stretch-sum tail at m29.  *Calculates:* the maximal flank sum of a word from its occurrence count
alone; counts fall 2-5 orders across a step's compatible spans (29->31: 7,815,766 / 205,068 / 6,500 /
4 at spans 10/21/31/41).  *Status:* `[measured: 12 word-steps]`.  *Where:* [La] I; [Ar] constructor
r20.  *Limits:* fitted, not derived; it is a LITERAL-letter law -- the two padded letters at m19/m23
are the two extremes of the band (1.80 and 6.14 against 2.39-2.96), so inverting it at m31 gives
`occ(37;m31)` in `[2.5e3, 4.0e11]`, eight orders wide.

**H14. Structural suppression on well-sampled words, and its exception.**  *Statement:* against an
exact zero-parameter rarity null every well-sampled compatible word sits BELOW the null at
`p = 0.0000`, by a deficit that GROWS with the machine (-1..-5 at m11-19, -7..-15 at m23 and m29).
THE EXCEPTION: `(10,21,10)` at 29->31 sits at obs 14 against null 15, `p = 0.4732` -- its four
occurrences behave exactly like four independent draws, and its margin of +19 is a pure sample-size
effect.  *Calculates:* decomposes the observed envelope into (i) the spectrum ceiling (an identity,
attained at m19), (ii) the rarity order statistic, (iii) a structural suppression of 7-15 gap units.
ONLY (i) IS A THEOREM.  *Status:* `[measured: full period, 14-row table]`.  *Where:* [Ar] mechanic
r17.  *Limits:* a derivation for the long words cannot come from the envelope or the ceiling -- "it
has to come from RARITY".

**H15. The target in occurrence form.**  *Statement:* `span(w) + lambda ln(occ(w)) <= F + q'` for
every compatible `w`, with `occ(w)` censusable and bounded above in closed form by
`N x (exposure product)`.  *Calculates:* "the first form in which every term is a counting quantity
with a closed-form upper bound: no extremes, no residue lottery, no fuel."  *Status:*
`[measured/derived: r20; four tests]`.  *Where:* [Ar] constructor r20.  *Limits:* the prediction
under-shoots the actual at two of four tests.

**H16. The counted word census.**  *Statement:* exact occurrence counts `occ(w)` of every run of
legal letters of length `<= 4`, with flank envelope `Phi(w)` and (length `<= 3`) the whole flank-sum
distribution, over the full cyclic period at m11..m37: `occ(23;m19) = 86`, `occ(29;m23) = 6`,
`occ(31;m29) = 2090`, `occ(37;m31) = 26,366`, `occ(41;m37) = 61,460`; two-letter words at m31
(`(12,25)` 35,314 each, `(12,37)` 150, `(25,37)` 18) and three-letter (`(12,25,12)` 188,
`(25,12,25)` 28).  *Status:* `[exact: full period, gated five ways -- count = prod(q-2), weighted
sum = P, max = F, max pair = F_2, every table mirror-symmetric]`.  *Where:* [La] I; constructor R102.
*Limits:* m41 is ~40x m37 and is not a one-round job.

**H17. The hole list, and holes heal.**  *Statement:* holes (gap values absent below the record) are
`{}`, `{9}`, `{17}`, `{19,24}`, `{24}`, `{41,42}` at m11..m29 (m31: 54, 56, 57), rare (0-2 per
machine against 7-41 realised values) and at the TOP of the spectrum (0.82F..0.98F) with one
exception (`v = 24` at m23, 0.71F).  `13->17: 9 healed; 17->19: 17 healed; 19->23: 19 healed, 24
inherited; 23->29: 24 healed` -- five of six holes are filled by the very next gear, no hole is ever
CREATED below the previous machine's `F`, so the spectrum fills in monotonically from below.
*Calculates:* which stretch lengths never occur, hence which padding steps are impossible.
*Status:* `[exact: full period m11..m29, four transitions]`.  *Where:* [Ar] mechanic r17 addendum 2.
*Limits:* machine 37's prefix at 4.85% coverage is INCONCLUSIVE, not a hole.

**H18. The residue law of the gap histogram.**  *Statement:* `hist_M[v]` is strongly non-flat in
`v mod p` and the shape is stable across machines and converging; **the richest classes are the
letter values of the small gears** (mod 7 the two richest are `v = 2` and `v = 5 = +-s` for gear 7;
mod 5 the richest is `v = 2 = s`).  *Calculates:* which stretch lengths the machine prefers.
*Status:* `[measured: full period, machines 11, 17, 23, 29, every entry moving monotonically and
settling]`.  *Where:* [Ar] mechanic r17 addendum 2; [La] gap 20.  *Limits:* it is NOT the naive
endpoint-survival count (`v = 2 mod 5` at 1.70 beats `v = 0` at 1.16) -- UNEXPLAINED; and it does NOT
predict the holes.

**H19. The corridor resonance.**  *Statement:* large gaps recur at fixed column distances --
multiples of 35; the slot-separation autocorrelation of big-gap left endpoints is 3.2-4.4x flat at
separations 35, 70, 105 (0.17-1.3 at neighbouring separations, and separation 70 exceeds separation
35 at every machine); left endpoints are PINNED to a few residues mod 35, the SAME classes rich at
every machine (m17/m19: `10,12,17,18` in an exact four-way tie).  *Calculates:* where the machine's
biggest gaps sit relative to one another, mod 35.  *Status:* `[measured: full-period exact counts,
five machines]`; closed form in `corridor-eigenvalue-closed-form.md` (the corridor-phase chain's
spectrum is the image of the roots of unity under a Moebius map with one parameter `rho`).  *Where:*
[S] E.  *Limits:* the closed form has one modelling step (columns independent given their phases).

**H20. The pinning law (LAW A).**  *Statement:* the neighbourhood word of a near-top gap pins its
address mod 385 to `<= 4` offsets, uniformly in `y`: gear 5 pinned to exactly one offset by every
near-top word (206/206 across five machines), gear 7 unique for 94%, gear 11 for 90%, gear 13 1-5
offsets; the full mod-385 address is unique for 87% and `<= 4` ALWAYS.  *Calculates:*
`#top-stratum classes <= 4 x #words`; observed 6-14 classes, flat, while gap counts swing 20-106.
*Status:* `[exact: containment 0 fails in 206 words, machines 13-29, full periods; tightness
71-85%]`.  *Where:* [S] E; [Ar] lateral r10.  *Limits:* the honest law is LOCAL,
`address = pin(word)`, NOT inherited; the open piece is uniformity of the near-top word grammar.

**H21. Address pinning of the record stretch.**  *Statement:* maximal gaps concentrate into 1-2
endpoint classes mod 35 (`y=19`: all twenty at left 5, right 30; `y=23`: `{3,33}`; `y=29`: `{2,25}`)
and 2-6 classes mod 385 out of 135 available (~30x over baseline); at `y = 23` and 29 the maximal gap
is UNIQUE up to mirror.  *Status:* `[exact: per machine, y = 13..29, streamed 1,078,282,205 for
y=29]`.  *Where:* [Ar] lateral r9.  *Limits:* unlike the absolute `L* = 13` landmark, the pinned
address DRIFTS with the machine -- gaps are machine-relative, saturated runs absolute.

**H22. Minimum separation of record stretches.**  *Statement:* 4-20 maximal gaps per period,
mirror-paired, with minimum separation 0.45-2.29% of the entire primorial period (851,695 columns at
gears `<= 23`).  *Calculates:* the empirical separation floor between records inside one period --
five-plus orders beyond what the route needs.  *Status:* `[measured: full periods, gears <= 11..23]`.
*Where:* [Ar] constructor r9.  *Limits:* measured only -- this IS the "Wall V" statement the route
cannot prove.

**H23. Two maximal stretches are never adjacent.**  *Statement:* at every machine `y = 13, 17, 19,
23` the top stratum occupies 4-6 classes mod 385 and the class-level adjacency test returns EMPTY;
the `alpha_1` certificate closes with a three-tier check (machine-free A3 / mod-385 class
disjointness / direct).  *Status:* `[exact: machines 13, 17, 19, 23; the tier-A piece kernel-checked
as `Machine13.no_11_11_chain` with no period scan]`.  *Where:* [Ar] constructor r10, formalist r11.
*Limits:* the tier-C residual GROWS (4 at y=13 to 96 at y=23), so scale needs mod-5005; uniformity in
`y` still open.

**H24. Near-top gap grammar.**  *Statement:* not finite in absolute terms (gap values grow with `y`,
no 32-cap analogue at the top), but the RELATIVE grammar is stable: flanks of near-top gaps in
`{1,2,3,4,5}` columns, chain interior spacings exactly `{2u', q-2u'}`, and near-top neighbourhood
word counts small and non-growing (14-42 distinct 5-gap words per machine, no trend y=13 to 29).
Top-gap neighbourhood `= [small flank][medium gaps][rigid chain skeleton]`.  *Status:*
`[measured: y = 13..29]`.  *Where:* [Ar] lateral r9.  *Limits:* the `{1..5}` flank alphabet was later
SCOPED DOWN to a first-flank fact; the max flank part grows 7 -> 13 with `y`.

**H25. Two-`n` gap reordering (openings in odometer order).**  *Statement:* sorting openings by CRT
phase vector (lex), the adjacent differences take EXACTLY `2n` distinct values at `n` gears, with
`mult(D(i,d)) = s_i(d) prod_{i'<i}(q_{i'}-2)`; CRT-lex order IS the mixed-radix odometer and
`d_i = 2` at every gear because the teeth are never adjacent.  *Calculates:* an explicit bijection
`Phi: [0,N) -> O` (a generalised van der Corput point set), so `F` is literally `P` times a digital
sequence's dispersion.  *Status:* `[exact: PROVED; prior art recorded as KNOWN IN MECHANISM
(Langevin's lex-successor theorem; Fried-Sos)]`.  *Where:* [S] E.  *Limits:* CLOSED LINE -- over 60
admissible re-choices of the teeth the count stays `2n = 8` while `F` ranges over `[10,18]`.

**H26. The exact-count route (COV-SAT / COV-COUNT).**  *Statement:* a phase vector corresponds by CRT
to exactly one column per period, so the per-period occurrence count of an alignment is the number of
models of the CNF projected to phase variables; count 0 is one UNSAT.  *Calculates:* exact
per-period counts where the count is small -- validated `(8,15)@19: 31`, `(10,21)@23: 138`,
`(10,21,10)@29: 4` with all four addresses reproduced, `(21,10,21)@29: 0`.  **RECORD MULTIPLICITY
LADDER: the record gap occurs exactly 4, 2, 4, 2, 4 times per period at m23, m29, m31, m37, m41** --
`O(1)` per period while the period grows six orders of magnitude.  *Status:* `[exact: every model
CRT'd to its column and machine-verified; op-count event -- `(10,21,10)@29` at 9,204 solver
propagations against `2^38 x 9` inclusion-exclusion ops]`.  *Where:* [S] H;
`cov-sat-exact-spectra.md`.  *Limits:* abundant patterns (counts `>~1e5`) stay out of enumeration
reach; cost scales with the COUNT, not with `2^{|Y|}`.

**H27. Inflation-onset law.**  *Statement:* `onset(M -> q') = min span of [(D_4(q'') \ D_4(q'))
INTERSECT the transfer's own emissions]` -- the transfer first over-generates exactly where the NEXT
machine's new repertoire begins; measured 13, 15, 17, 25, 31, 41, 53, 68 at 11->13 .. 37->41.
*Calculates:* the "certainly exact" region of a superset before any decision is paid for; it
PREDICTED `onset(37->41) = 68` out of sample.  *Status:* `[measured: refined form 31 of 31 across six
output arities and two screens; simple form 16 of 25; causal version 8/8]`.  *Where:* [S] E;
`dictionary-monotonicity-onset.md`.  *Limits:* not proved; the mechanism is closure failure x phase
saturation x a near-constant factor (`onset/Y_5` in a band of width 0.042), and the third factor is
unexplained.

**H28. The delta-profile.**  *Statement:* `delta_q(e) = min(e mod q, q - e mod q)`; the twin
difference has `delta_q = 1` for EVERY gear -- twins are the maximally clustered member of the
family at every scale at once.  Winners: gears `3,5,7,11 -> maxF 33`, profile `(1,1,1,3)`;
`+13 -> 75`, `(1,1,1,3,6)`; `+17 -> 96`, `(1,1,2,4,6,8)` and `(1,1,2,3,4,3)`.  At gears `<= 13` the
rule is exactly "`e` is extremal iff `(delta_3..delta_13) = (1,1,1,3,6)`", satisfied by 16 of 7,507
differences, all 16 attaining `F = 75`.  *Status:* `[exact: exhaustive; precision and recall 100% up
to gears <= 13]`.  *Where:* [Ar] harvester r17.  *Limits:* "maximally spread at the top" describes
some maximisers, not all.

**H29. Extension versus compromise.**  *Statement:* the winning profile at 13 is the winning profile
at 11 with the new gear's entry appended AT ITS MAXIMUM (the optimum EXTENDS, `maxF x2.27`), while at
17 it cannot (`delta_7` moves 1 -> 2 and `delta_11` 3 -> 4, `maxF x1.28`).  *Calculates:* predicts
whether a step gains its full record increment or a compromised one.  *Status:* `[exact: for the
computed machines; persistence measured -- the champions of gears <= 13 land at the 99.3-99.8th
percentile at gears <= 17 but none stays maximal]`.  *Where:* [Ar] harvester r17.  *Limits:* three or
four machines.

**H30. Paired-Jacobsthal values, and where twins sit.**  *Statement:* first exact `h_2` values
`y = 5: 18; 7: 30; 11: 66; 13: 150; 17: 192` against the Ziller-Morack Conjecture 6 bound
`p_n^2 - p_n = 20, 42, 110, 156, 272` -- holds at all five, but the margin is non-monotone with a
one-off dip at 13 (3.8% against 10.0, 28.6, 40.0, 29.4).  **Twins are the easy end**: at gears
`<= 13`, among the 2,880 differences coprime to `P`, `F` ranges 30..75 and the twin difference gives
33 -- the 13.3rd percentile, with 77.2% of coprime differences having a LARGER record and the
extremal one 2.27x the twin value (1.78x at gears `<= 17`, 21st percentile).  *Status:*
`[exact: exhaustive to y = 17; y = 19 only a lower bound]`.  *Where:* [Ar] harvester r16.  *Limits:*
"prove it for twins" is strictly the easy end of "prove it for every even difference".

**H31. Density does not determine the record.**  *Statement:* over the 31 `gcd(e,P)` classes at gears
`<= 13`, `F_max/lambda` ranges 2.88 (`gcd = 5005`) to 7.52 (`gcd = 3`) -- a factor 2.6 spread; two
difference classes with the same mean gap can differ by more than 2x in their record.  *Status:*
`[exact: all 31 classes at gears <= 13]`.  *Where:* [Ar] harvester r16.  *Limits:* one machine.

**H32. Openings per period = the Hardy-Littlewood factor.**  *Statement:*
`|E_d| = prod_{q in {3,5,7}} (q - r_q)`, `r_q = 1` if `q | e` else 2 -- giving
15/20/18/30/36/24/40/48 across the eight `gcd(e,105)` classes.  "The Hardy-Littlewood factor and the
exposed-set size are the same object."  *Status:* `[exact: every prime q' <= 1200 coprime to 105, all
8 classes; independently reproduced in the kernel]`.  *Where:* [Ar] harvester r9-r10, formalist r17.
*Limits:* openings counted, not located.

**H33. Slot-cap at gap `2d`.**  *Statement:* an odd prime blocking BOTH members of a gap-`2d` column
forces `q | d`; `slot_cap_twin` is the `d = 1` case.  "This is THE transfer condition: every corpus
law whose proof rests on slot-cap holds verbatim for gap `2d` at gears coprime to `d`, and the gears
`q | d` collapse to one residue -- the Hardy-Littlewood factor, mechanically."  *Status:* `[kernel]`
(`proofs/Polignac.lean`); verified computationally first.  *Where:* [Ar] harvester r1.  *Limits:*
about one gear's teeth, not about coincidence across gears.

**H34. Per-gap reduction.**  *Statement:* `gapPairs_infinite_iff_survivor_in_window (d)` -- the
per-gap IFF, both directions, every `d` (`d = 0` degenerating to infinitude of primes).
*Calculates:* turns every even-gap conjecture into a statement about openings inside the window.
*Status:* `[kernel]` (`survivorGap_one_iff` needs only `[propext]`); verified computationally first.
*Where:* [Ar] harvester r1.  *Limits:* an equivalence, not progress.

**H35. Placement, injectivity, and the counting primitives.**  *Statement:* `slotOf m = (m+1)/6`
recovers the column from either member; `sign_law : (ab) % 6 = 1 <-> a % 6 = b % 6`;
`slot_injOn_partners` (a gear never strikes the same column twice); `six_mul_class` (`{k : 6k = c mod
m}` is exactly one class mod `m`); `card_class_Ico`.  *Calculates:* which column a gear-multiple
occupies and on which side, and how many columns of a prefix lie in a class -- every floor term of
the supply formula.  *Status:* `[kernel]` (`proofs/Placement.lean`, `proofs/Polignac.lean`).
*Where:* [Ar] formalist r8, harvester r3.  *Limits:* regime `q >= 5` prime, members `< q^3`; column 0
degenerate; requires coprimality to 6.

**H36. Same-side, split and two-sided classes; the three-gear master identity.**  *Statement:* for
distinct primes the same-left / same-right / split columns are each ONE CRT class mod `qr` with
closed-form counts; `twoSided_class` subsumes them; and `three_gear_master` gives, over the first `t`
columns, `distinct + 12 pair side classes = 6 single side classes + 8 triple side classes`, every
term a closed-form floor count.  *Calculates:* the exact number of distinct columns three gears leave
open over a prefix.  *Status:* `[kernel]`; verified first over 105 pairs, 210 ordered pairs, 60
two-sided triples, and 5 gear triples x 5 window lengths to `t = 5005`.  *Where:* [Ar] harvester
r3-r7.  *Limits:* three gears; `n > 3` assessed and deliberately deferred.

**H37. Semiprime refinement.**  *Statement:* for `q` prime and `1 < m < q^3` composite with
`minFac m = q`, `m = qc` with `c` prime and `q <= c`; hence `R q S = #{c prime : q <= c, qc in S}`.
*Calculates:* exactly which columns a large gear strikes inside the window -- one per partner prime.
*Status:* `[kernel]`.  *Where:* [Ar] formalist r7.  *Limits:* the boundary case is the square.

**H38. The one-kill lemma.**  *Statement:* gear `q` contributes at most `2(floor(L/q)+1)` teeth to a
stretch of `L`; in the maximal stretch of `{5,7,11,13}` gears 11 and 13 sit exactly at their
ceilings, and gear 5 kills 9 of 24 columns -- its 2-per-5 rhythm, tight.  *Calculates:* an upper
bound on how much of a stretch a gear can block.  *Status:* `[exact]`; `[kernel]` in R-form
(`Gear.R_prefix_le`).  *Where:* [Ar] `chain-conditions.md`, `state-walk.md`.  *Limits:* summed over
gears the capacity is abundant from `y = 13` on (`sum 2/q >= 1`), so the cap alone never bounds the
record.

**H39. The mex of the tooth schedules.**  *Statement:* the stride from any open column is the MEX of
the union of the gears' tooth schedules; the certificate is complete and per-column (the maximal
stretch of `{5,7,11,13}`, 11 columns from column 122, with every interior column's blocker listed).
*Calculates:* the exact distance to the next opening, and the per-column attribution of a blocked
stretch, with no stepping.  *Status:* `[exact: verified to agree with the walk]`.  *Where:* [Ar]
`chain-conditions.md`, `state-walk.md`.  *Limits:* per-position; gives the next opening but no bound
on it.

**H40. Anatomy of a blocked stretch.**  *Statement:* maximal stretches are bracketed by coincidence
hubs (entry and exit are TRIPLE kills, the mid-run column a deep hub -- worked end to end for
`{5..19}`, 25 columns, entry 111, mid 119 with `715 = 5x11x13`, exit 134, then 135 all umbrellas ->
the twins `(809,811)`); 18 of the 24 blocked columns die to EXACTLY ONE gear; and twelve of the 24
have some gear SHIELDING while another kills.  *Calculates:* how fragile a record stretch is, and why
isolated shields never make twins.  *Status:* `[exact: machine {5..19}, full walk recorded]`.
*Where:* [Ar] `state-walk.md`.  *Limits:* one machine; hub-rate at the binding loci was later
measured as generic.
### I. The window, the section and the walk

**I1. The window identity.**  *Statement:* the admissible pattern and the twins coincide exactly on
`(y, y^2]`: `survivors(y,K) = T(6K+1) - T(y)`.  *Calculates:* the certified translation from openings
to twins.  *Status:* `[exact: y = 11..1009]`; `[kernel]` (`BlockedSlots.survivor_iff_twin`,
`twins_infinite_iff_survivor_in_window`, `Horizon.twin_of_no_prime_factor_lt`).  *Where:* [La] A;
[Ar] `twin-prime-program.md` 17d.  *Limits:* it is the definition of the target, not a step toward
it; the kernel needs one survivor anywhere in `(q, q'^2)`, and the SECTION-only statement is strictly
stronger than the conjecture.

**I2. The window is the opening stretch of the periodic pattern.**  *Statement:* at gear `q` the
kernel's window `(q/6, W]`, `W = (q'^2-1)/6`, is the opening stretch of the pattern of gears `7..q`
and includes every lower section.  Full-period figures (blocked-column counts): `q=7`: period 35,
worst run 4, `W=12`, 4 open in window, run entering the window 2; `11`: 385, 6, 8, 2, 4; `13`: 5005,
10, 20, 7, 4; `17`: 85085, 17, 12, 2, 4; `19`: 1616615, 24, 28, 4, 11; `23`: 37182145, 33, 52, 8, 7.
*Status:* `[exact: full-period computation]`.  *Where:* [La] G; `anchor-235.md` 7.  *Limits:* the
worst run already exceeds the current SECTION at `q = 17` (17 against 12) -- existence in the section
is POSITIONAL.

**I3. Where the worst stretches sit.**  *Statement:* deep in the period, in mirror pairs at `k` and
`P - k` (fractions 0.3-0.7, or at the period's ends), **never at the window**; the run at `q^2/6` is
at most 0.663 of the section (worst at `q = 137`) to `q = 5000`.  *Calculates:* the provenance of the
section's survivor -- "it is not that the record is small, it is that the record is somewhere else".
*Status:* `[measured: to q = 5000; mirror pairing exact]`.  *Where:* [S] F; [La] G;
`anchor-235.md` 7, `word-tree.md` 7.3.  *Limits:* positional -- a statement about the section, not
about `F`.

**I4. Against the whole window the position drops out.**  *Statement:* `F(q) < W(q) - q/6` forces an
opening in the window whatever the pattern does at `q^2`; measured `F/W = 0.25` flat from `q = 5` to
53, `F(59) = 161` against `W = 620`; and if the budget inequality holds at every rung then
`F/W <= 3/ln y < 1` for `y > 20`.  *Calculates:* the reduction of "an opening lands inside the
window" to the record law plus the budget inequality, with the positional question removed entirely.
*Status:* `[measured: the ratio]`; `[exact: the implication]`; the budget inequality holds at every
computable rung through 59 (203 against 204 at 53->59).  *Where:* [S] F; [La] G.  *Limits:* this is
the whole open part.

**I5. The tolerance theorem.**  *Statement:* if `F(M+q) - F(M) <= alpha q` at every consecutive step
with `q > 47`, for any fixed `alpha` at or below the `alpha*(y)` scale, then
`F(2,y) <= 354 + alpha(S(y) - 328) < (y^2-y)/2` for every prime `y >= 53`.  *Calculates:*
`alpha*(y) = [(y^2-y)/2 - 354]/[S(y) - 328]` = 5.64 at `y=101`, 8.71 at `1e4`, 13.3 at `1e6` --
asymptotically `ln y`.  *Status:* `[exact: conditional theorem; checked at every prime y in
[53, 1e6], zero failures, worst ratio 0.6557 at y = 113; beyond 1e6 by Rosser-Schoenfeld]`.
*Where:* [La] G; [Ar] constructor r8.  *Limits:* the open link is "no consecutive step ever exceeds
`2.5q`"; the observed maximum is `2.432q` at gear 37.

**I6. The multiplicative route and the fourth wall.**  *Statement:* the two missing lemmas are TOP-GAP
ANTI-CLUSTERING (`F_2 - F = O(q)`) and FUEL-MERGE CONTROL (`excess = O(q)`), measured `<= 1.24q` and
`<= 1.62q` at their separate maxima; the obstruction is a wall distinct from abundance, localisation
and parity -- EXTREME-VALUE CONTROL OF SIEVE PATTERNS.  *Calculates:* classifies any candidate route:
"an attack belongs to this event iff its missing input is a statement about the machine's own gap
word, with no prime-counting content."  *Status:* `[exact: the tolerance arithmetic; the two lemmas
measured, not proved]`.  *Where:* [Ar] constructor r8, attempts-map amendment.  *Limits:* it names an
obstruction; it does not remove it.  The route is not the parity wall by the dimension-1 test.

**I7. The record fits the window with a factor 2.3-3, falling.**  *Statement:* `F_k(y) <= (y^2-y)/6`
holds with a factor of 2.3 to 3 for every gear set measured, and the ratio is falling.  *Status:*
`[measured: every gear set to the corpus limit]`.  *Where:* [Ar] `gear-at-infinity.md`,
`gear-recursion.md` 2.  *Limits:* a measured margin, not a bound.

**I8. The record grows polylogarithmically while the window grows quadratically.**  *Statement:* the
maximal blocked stretch tracks `~0.45-0.49 log^3(member)/6` columns across the whole measured range,
and stretch/window collapses from `2.1e-2` (`y=101`) to `6.0e-7` (`y=100003`, members to `1e10`).
*Calculates:* the measured slack of Reduction A at any scale.  *Status:* `[measured: y = 101..100003;
27.4M twins generated and verified]`.  *Where:* [Ar] `class-tree.md`.  *Limits:* a trend, explicitly
NOT a wall and not a proof.

**I9. Section aggregates.**  *Statement:* every one of 667 sections to `q' = 5003` holds a twin;
`G_S/|S|` is below 1 everywhere, max 0.684 at 29->31, falling 0.352, 0.221, 0.177, 0.092 by band;
twin counts are Hardy-Littlewood (observed/predicted 1.0028 over `1000 <= q' <= 5003`).
*Calculates:* the section is a Mertens word at every scale, sections differing only by scale.
*Status:* `[measured: section_probe_r29.py, 8 of 9 gates; the failure S4 is a bookkeeping error]`.
*Where:* [S] F; `word-tree.md` 7.2-7.3.  *Limits:* nothing here is provable by the machine: twin gaps
are unbounded in principle and the `ln^2` scale is heuristic.  **Harvest disagreement on the minimum
counts:** [S] gives 2, 6, 10, 21, 51 across its bands (always at a gap-2 rung); [La] gives 2, 3, 6,
7, 19, 21, 42, 51, 68 across bins 5-50 .. 4000-5000, with run/section falling median 0.235 -> 0.020
and worst 0.544 -> 0.085.

**I10. Inside the section the machine below is exact and the new gear is silent.**  *Statement:*
every composite below `q'^2` has a prime factor `<= p`, so gears `5..p` are exact in the section and
`q'` does nothing in its own section; the previous gear `p` enters only through `p m` with `m` prime
in `(p, q'^2/p)`.  *Calculates:* the section's word from the machine below, with no new arithmetic.
*Status:* `[exact: forced]`; `[measured: 667 sections -- gear p is the death rung of at most 3
columns, and of NONE at 77% of sections with q' >= 500]`.  *Where:* [S] F; `word-tree.md` 7.1-7.2.
*Limits:* as NUMBER-strikes `p` reaches up to six; the "at most three" is about DEATH RUNGS.

**I11. The section's blocked word is the divisibility lattice.**  *Statement:*
`blocked(p -> q') = union over s of s * open_{<s}((p^2/s, q'^2/s))`, so a new twin is a column no
scaled open word reaches on either side.  *Calculates:* the section as a stitch of the open words of
every smaller machine, each scaled by its next gear.  *Status:* `[measured: gated over 666 sections,
per-gear bands contiguous]`.  *Where:* [S] F; `word-tree.md` 9.2.  *Limits:* **no section-specific
feature of gear interactions was found** -- both which vectors survive and where the strikes come
from reduce to CRT and to the smaller machines' open words.

**I12. The residue vectors are uniform in the section.**  *Statement:* over 122,546 new twins with
`q' >= 1000`, uniform over the tooth-avoiding classes to TV 0.0026 (mod 5), 0.0033 (mod 35), 0.0097
(mod 385).  *Calculates:* "killing twins for ever would need a rung from which no tooth-avoiding
vector lands in the section, and the vectors that land are the generic ones -- the kill would have to
remove every class at once, not a pattern."  *Status:* `[measured: 667 sections, 8/10 gates]`.
*Where:* [S] F; `word-tree.md` 8.2-8.3.  *Limits:* a distributional statement, not an existence
proof.

**I13. Provenance of a new opening.**  *Statement:* (i) the two sides are INDEPENDENT (TV 0.024 from
the product of marginals; left marginal `5: 0.665, 7: 0.134, 11: 0.045`, `(5,5)` alone 44%); (ii) the
number of gears touching a new twin's word grows like the RECORDS of an iid Mertens sequence (2.2 at
`q' < 100`, 3.7 at `q' ~ 5000`, model within 6%); (iii) the top of the provenance is a gear `> p/2`
about half the time (46-48%).  *Status:* `[measured: 130,664 new twins over 667 sections]`.  *Where:*
[S] F.  *Limits:* `q'` itself appears in NO provenance.

**I14. The blocked run inside a section is the generic Mertens tree.**  *Statement:* depth 6.6 ->
36.8 across the bands while run length grows 15.6 -> 197.8; single-kill levels 58-63% of the depth,
top single-kill chain 46-48%, in every band; the top five levels single-kill in 100% of trees pooled
over `q' >= 1000`; 60% extensions and 40% joins in every band; median join ratio exactly 1/2 through
the middle; the top of the tree unbalanced (~1:3).  *Status:* `[measured: 667 sections, exploratory,
not pre-registered]`.  *Where:* [S] F; `word-tree.md` 7.4-7.5.  *Limits:* nothing repeats at the
tuple level (no top 3-tuple pattern reaches 3% of sections); only the statistics are universal.

**I15. Section view: every section holds an aligned column.**  *Statement:* per section
`(q^2, q'^2)`, taking nothing from lower windows, every section to `q = 5000` holds a twin; aligned
count `= anchor-open x prod_{7<=g<=q}(1-2/g) x 0.66-1.0`.  *Status:* `[exact: to q = 5000,
section_trend.py]`.  *Where:* [La] G; `anchor-235.md` 4.  *Limits:* STRONGER than the twin
conjecture -- recorded as an overstatement corrected.

**I16. The walk from `q^2`.**  *Statement:* stepping by residue tests only from the column holding
`q^2`, the walk lands on an opening of `{5..q}`, and that opening IS a twin pair; landings from
`q = 37, 97, 499, 997, 4999, 10007, 100003` at 10, 2, 22, 10, 40, 12, 79 columns.  *Calculates:* the
address of the first twin above `q^2` by residue arithmetic alone.  *Status:* `[exact: slot_walk.py,
all landings verified twins]`.  *Where:* [La] G; `anchor-235.md` 6.  *Limits:* existence for a FIXED
gear set is CRT; for the GROWING gear set it is the conjecture.

**I17. The layered walk from `q^2`, for every prime to 5000.**  *Statement:* 667 walks, EVERY landing
a twin pair; walk length median 19, maximum 265 at `q = 4637` (second 187 at 2593 and 4003); 1 to 44
layers hop per walk; total hops equal the walk length in every walk (an identity -- each traversed
column is counted once, at the layer of its smallest blocking gear).  *Status:* `[exact: 667 primes,
layered_walk.py]`.  *Where:* [La] G; `anchor-235.md` 9c.  *Limits:* closed by a recursion of depth
`pi(q)`; no formula collapsing the recursion has been found.

**I18. The first opening past `q^2`.**  *Statement:* the first twin sits a median 18 columns past
`q^2`, maximum 264 at `q = 4637`, to `q = 5000`; the position of an open whole cycle inside a section
is uniform (quartiles 0.24, 0.48, 0.74).  *Status:* `[measured: to q = 5000]`.  *Where:* [La] G.
*Limits:* a measurement, not a law; the maximum is a record on a curve.

**I19. Whole anchor cycles inside sections.**  *Statement:* a cycle is untouched by gear `q` iff
`j mod q` avoids six residues; under gears `7..Q` the open cycles have period `prod q` with
`prod(q-6)` per period.  Against the window sections to `10^8`: 1,226 sections, 1,088 with no open
cycle, 121 with one, 16 with two, 1 with three; share rising 0% to 13%; longest dry stretch 50
sections (`q = 7079..7549`).  *Status:* `[exact: below 10^8, 156 such cycles]`.  *Where:* [La] G;
`anchor-235.md` 5.  *Limits:* the section is NOT the natural unit for whole cycles; existence for the
growing gear set is the Hardy-Littlewood sextuplet conjecture.

**I20. Onset law `L0(y) <= 27129`.**  *Statement:* the lag from the window's start to the first
column with both members composite is bounded unconditionally via Montgomery-Vaughan.  *Calculates:*
an absolute bound on how deep into a window the first double column can be; measured over 442 windows
`13 <= y <= 3163`: max `L0 = 17`, `L0 = 0` in 153/442, a twin precedes the first double in 132/442.
*Status:* `[exact: unconditional theorem; measurements exact]`.  *Where:* [La] G; constructor R7.
*Limits:* 310 of 442 real windows have NO twin in the onset prefix, so the onset scale is not itself
a contradiction.

**I21. The first double column is `k = 20`.**  *Statement:* `(119,121)`; every column `k <= 19` has a
prime member.  *Status:* `[exact]`.  *Where:* [La] G; constructor R5.  *Limits:* a prefix statement;
the prefix-pigeonhole refutation reaches only `t <= 4`.

**I22. Rule capacity forces an opening in the section, up to `y = 88`.**  *Statement:* for 75 values
of `y` between 6 and 88 there is a `T <= y^2` where the candidate columns outnumber the maximum
possible strikes, so a twin MUST exist in `(y, T]` -- proved by counting rules, with no sieve theory;
the largest is `y = 88`, window `(88,115]`, 5 candidates against at most 4 strikes.  *Status:*
`[exact: corrected figure]`.  *Where:* [Ar] `twin-prime-program.md` 1e-1f.  *Limits:* the mechanism
dies exactly when the sixth odd gear becomes root-active, near `y = 90`, and never succeeds again.

**I23. Umbrella jumping and the stack certificate.**  *Statement:* inside a window a twin IS a column
whose joint umbrella exists over the certifying set; each gear's distance to its next tooth is
`min((u_q - m) mod q, (-u_q - m) mod q)`.  Umbrella-jumping pinpointed all 55 twins of the 47-window,
with the six prime quadruplets appearing automatically as width-2 umbrellas; each twin carries a
stack certificate naming the binding gears (column 23 = `(137,139)`: gears 5, 7, 11 all at room 0 --
"twins sit in needle's eyes").  *Calculates:* THE NEXT OPENING from any position without stepping.
*Status:* `[exact: scaled to y = 100003, 1.67e9 columns, 27,412,929 twins, 180,504 quadruplets]`.
*Where:* [Ar] `class-tree.md`, `gear-at-infinity.md`.  *Limits:* it computes the next opening; it
does not BOUND the distance to it.

**I24. The gap has no closed form.**  *Statement:* offset `t` is open iff
`gcd(n+t, primorial(R)) = 1`, so the distance to the next opening is the least `t >= 1` with `n+t`
coprime to the primorial -- by CRT a union of residue classes modulo the primorial, exponential in
`R`.  Anything producing the gap in time polynomial in `log n` would bound it and settle the
question.  *Calculates:* the ledger -- per-gear next tooth is closed form; per-offset openness is
closed form; THE GAP is not.  *Status:* `[exact: three implementations cross-checked on 28,000
consecutive odd n, zero disagreements]`.  *Where:* [Ar] `gap-without-lattice.md`.  *Limits:* the
floor on certifying one opening is `pi(R)` consultations.

**I25. The closed-form target, stated without gear bookkeeping.**  *Statement:* generating the next
opening in closed form is exactly: minimise `CRT(v)` above a given bound subject to
`v_q not in {0, -2}` for every gear `q <= y`.  *Calculates:* the exact statement any closed form must
solve; the unit-pair count equals the twin count below `y^2` exactly at `y = 5..29` (2, 4, 7, 9, 15,
17, 21, 28).  *Status:* `[exact: restatement]`.  *Where:* [Ar] `twin-prime-program.md` 29c-29d.
*Limits:* "the closed form will not come from refining the slip arithmetic, because the slip
arithmetic is now fully solved and expressed".

**I26. Forcing channels and the forcing budget.**  *Statement:* gear `q` cannot strike column `k`
whenever `q | 6k + c` for `c` not `+-1 mod q`; widening from the midpoint alone to
`c in {0,+-2,+-3}` takes the self-certifying family from 4 columns to 10
(`k = 1,2,3,5,7,10,12,17,18,33`), every one a genuine twin, with nothing further to `k = 300000`.
THE BUDGET: certifying `k` requires `pi(sqrt(6k))` gears forced open while each fixed channel
supplies `omega(6k+c) ~ log log(6k)`, and one channel protects at most `log_5(6k+c)` gears (14 at
`k = 10^9` against 7,606 demanded).  *Calculates:* the exact numerical requirement on any closed
form.  *Status:* `[exact/measured to k = 300000; the deficit becomes permanent at k = 50]`.  *Where:*
[Ar] `twin-prime-program.md` 22b-22c.  *Limits:* any route to a closed form needs a forcing mechanism
that is not divisibility by a product; the three tried give a dichotomy.

**I27. Bounded-depth state never determines the position of the next opening.**  *Statement:* for any
depth `z`, twin columns and non-twin columns share every residue up to `z` (explicit copies at
`z = 13, 29, 97, 199, 997`).  *Calculates:* refutes in advance any navigation rule using gears only
up to a fixed depth.  *Status:* `[exact: explicit construction at five depths]`.  *Where:* [Ar]
`twin-prime-program.md` 23b-23c.  *Limits:* FRAME-INDEPENDENT; it does not rule out a rule consuming
gears to `sqrt(6k)`.  Empirically at `k0 = 10^12 + 1` the step count reads 1, 1, 1, 2, 19, 19, 19, 19
for gears to 20..1e5 and 86 for the full set -- "stopping early gives a SPECIFIC WRONG ANSWER".

**I28. Constraint concentration.**  *Statement:* a gear can only matter over the next `W` columns if
`d_q <= W`; at `k0 = 10^12+1` with 179,643 gears, `W=16` gives 60 constraining gears, `W=128` gives
299, `W=512` gives 940 -- precisely the gears dividing one of the `2W` window numbers.  *Calculates:*
the answer is determined by a few hundred gears (600:1 at `k = 10^12`), the biting count growing like
`2W log log y`.  *Status:* `[measured at k0 = 1e9, 1e12, 1e14, 1e16]`.  *Where:* [Ar]
`twin-prime-program.md` 24b.  *Limits:* knowing WHICH gears bite requires factoring the window
numbers.

**I29. Inside the window every event is a root event.**  *Statement:* for a column in the window
every opening-ending has cofactor at least the gear; measured over 1,024 columns at `k0 = 10^12+1`:
4,327 events, 4,327 root, 0 redundant.  *Calculates:* "a second characterisation of the window: it is
exactly the region in which the gear set does no redundant work".  *Status:* `[exact: measured
4327/4327]`.  *Where:* [Ar] `twin-prime-program.md` 25a.  *Limits:* not a statement about overlap.

**I30. Fragile columns.**  *Statement:* in the same 1,024-column window, 9 columns have zero endings
(twins) and 55 have exactly ONE, with closing gears from 5 up to 353,057.  *Calculates:* the concrete
form of the bounded-depth obstruction -- to a gear set stopping below 150,697, column +19 is
indistinguishable from a twin.  *Status:* `[measured: one window at one scale]`.  *Where:* [Ar]
`twin-prime-program.md` 25b.  *Limits:* one window.

**I31. The small-gear phase floor, and where the phase relation dies.**  *Statement:* the least open
offset of a sub-machine is a valid LOWER bound on the distance to the next opening (a table lookup),
but weak: at `m = 10^12+1` the floor from `(5,7,11,13)` is 1 against a true step of 86.  The one
structure that would relate phases across gears dies at the square root: on the hyperbolic block
where `c = floor(6m/q)` is constant the phase is linear in the gear, but `q <= sqrt(6m)` forces block
length `< 1` -- measured over all 179,643 gears: MAXIMUM GEARS SHARING A BLOCK = 1.  *Status:*
`[exact/measured]`.  *Where:* [Ar] `twin-prime-program.md` 33a-33b.  *Limits:* the dual (cofactor)
side supplies no relation either; the phase relation and the certification window fail at the SAME
place, recorded as "not a coincidence".

**I32. The spectrum factorises, but no beat may be dropped.**  *Statement:*
`Ehat(k) = prod_q ehat_q(k t_q mod q)` with `t_q = (P/q)^{-1} mod q`; the L1 norm factorises too and
grows about 2.06 per gear (per-gear factors 1.494 .. 2.136 for gears 5..29, approaching `1 + 4/pi`),
so at `m = 10^12` with 179,643 gears it is around `10^56000` against a required resolution of 1/2.
*Calculates:* the entire Fourier transform from a product of `n` cosines per frequency, with no FFT.
*Status:* `[exact: agrees with a direct FFT to 1.1e-16 across all 5005 frequencies for
(5,7,11,13)]`.  *Where:* [Ar] `twin-prime-program.md` 35a, 36b-36d.  *Limits:* "the frequency domain
is a faithful and exact description that is exponentially less compact than the machine it
describes"; the `t_q` twist is essential and easy to miss.

**I33. Absolute landmarks versus machine-relative addresses.**  *Statement:* saturated runs are
ABSOLUTE (primality-only), so every window sees the same integers -- `L* = 13` sits at columns
2452-2464 at every scale, and a window excluding a landmark inherits the next instance.  Record GAPS
are machine-relative and their pinned addresses drift.  *Status:* `[exact: one absolute segmented
scan of k = 1..1.2e10]`.  *Where:* [Ar] mechanic r7, lateral r6/r9.  *Limits:* exhaustive to member
7.2e10 (later 1.67e11).

**I34. Record-run addresses.**  *Statement:* `L=10` at `k=59` (member 353); `L=13` first at
`k=2452` (member 14711, word `RLLRRLLLLRLRL`), recurring at 61,501,443; 874,166,593; 1,909,351,447;
8,472,005,085; 9,599,932,213; the first `L=14` at `k = 46,133,660,494`, members
276,801,962,963 .. 276,801,963,043, word `LRRLRLRRRRLLRL`, both boundary columns both-composite,
Miller-Rabin verified.  *Status:* `[exact: exhaustive scan, independently verified]`.  *Where:* [Ar]
mechanic r7/r9.  *Limits:* `L* = 13` stood from member 1.5e4 to 2.8e11 -- "a record on the curve,
never a wall".

**I35. Inside a record stretch: P-rate plus n2-rate = 1.**  *Statement:* per column, `0.80 + 0.20` at
`L=25`, `0.52 + 0.48` at `L=100`; `L <= 13` records have zero interior doubles -- pure `n1`, every
column a pseudo-twin column.  *Status:* `[exact: within the examined records]`.  *Where:* [Ar]
lateral r6.  *Limits:* descriptive of records only.

**I36. The load-length frontier is ABSOLUTE.**  *Statement:* `maxload(L)` = 1.0000 for `L <= 13`,
then .9286, .875, .85, .80, .7188, .60, .52, .43, .32 at `L = 14, 16, 20, 25, 32, 50, 100, 200, 478`
-- IDENTICAL at `y = 1009, 3163, 10007` because the record-holders are the same absolute integers.
*Status:* `[exact/measured at three scales]`.  *Where:* [Ar] lateral r6.  *Limits:* renewability at
depth is measured, not forced.

**I37. Two extremal families.**  *Statement:* load-extremal runs (short, absolute, prime-dense,
constellation-governed) and length-extremal stretches (deep, load ~0.3, gap-word-governed) are
DIFFERENT families, merging only at `L = maxstride`.  *Calculates:* which instrument applies to which
regime -- chain analysis cannot see the binding region `L ~ 14-32`.  *Status:* `[measured: y = 1009,
3163, 10007]`.  *Where:* [Ar] lateral r6; attempts-map 5.  *Limits:* it says the chain machinery is
the wrong tool for the load frontier, and vice versa.

**I38. Persistence as a Bertrand-type statement.**  *Statement:* "every level-`y` open interior
contains a saturated run of length `L`" is EQUIVALENT to `6 r_{n+1} - 1 < (6 r_n + 1)^2` for the
positions of `L`-saturated runs; `persistence(1)` is a THEOREM (Brun), `persistence(2)` is
disjunctive Polignac (OPEN), `persistence(L >= 3)` disjunctive Hardy-Littlewood.  *Calculates:* the
exact provability frontier of the frontier curve.  *Status:* `[exact at L=1; conjectural for
L >= 2]`.  *Where:* [Ar] lateral r7.  *Limits:* the frontier is a DESCRIPTIVE upper envelope, never a
premise.

**I39. The fresh-block recursion.**  *Statement:* in any band, gear `q`'s fresh blocks sit at `qr`
with `r` running over the PRIMES in `band/q`, so the machine's blockers in each band are its own
output from lower bands, re-entered as structure; deaths in band `h` draw root gears only from bands
up to the tower-half -- a hard lower-triangular cutoff.  *Calculates:* the band-by-band attribution
matrix.  *Status:* `[exact: by law L3, computed over all 1e6 candidate pairs to midpoint 6e6, 168
bands]`.  *Where:* [Ar] `band-attribution.md`.  *Limits:* "the matrix quantifies the cascade; it does
not bound anything" -- any by-construction argument must control the PLACEMENT of the products `qr`,
a bilinear statement about pairs of machine outputs.

**I40. The gear-at-infinity frame.**  *Statement:* the machine is fully constructed to infinity, the
gears return after the primorial, at 0 the machine is completely aligned, and no gear can outpace the
6-cycle -- so the structure near 0 recurs.  Four of six steps are theorems.  *Calculates:* nothing;
it is the frame that produced the `+-1` walk law, the two blocking laws, the mod-3 law and the
closed-form next-twin method.  *Status:* `[exact: steps 1-5 proved; step 6 open and equal to
Reduction A]`.  *Where:* [Ar] `gear-at-infinity.md`.  *Limits:* "the frame gives: the configuration
exists, recurs, and recurs at the fastest rate the machine allows.  It does not give: it recurs
within `y^2` of where the gear set was assembled."

**I41. Reduction A in umbrella language.**  *Statement:* a twin column is a column standing under
every relevant gear's umbrella at once; the umbrellas provably overlap somewhere in every period (the
alignment law); the single open question is WHERE.  *Calculates:* nothing -- it is the exact
statement of the alignment question in the human's own vocabulary.  *Status:* `[exact: the alignment
law proved; the localisation open, kernel-checked equivalent to the conjecture]`.  *Where:* [Ar]
`umbrellas-and-shields.md`.  *Limits:* existence in a period of size `~e^y` against a window of size
`~y^2` -- "localisation, not existence".

**I42. The README's original alignment observation (the oldest record).**  *Statement:* "each new odd
prime `p` has its first multiple after `p` at `2p`, which is even and therefore already blocked by
prime 2 -- new primes predominantly ALIGN their blocking patterns with existing blocked positions
rather than creating entirely novel constraints."  *Status:* `[informal observation, predating
everything else]`.  *Where:* [Ar] `README.md`.  *Limits:* the same README records "no clear mechanism
emerges from the modular arithmetic for simultaneous blocking of adjacent odd positions" as its own
open question.
### J. The tooth-counterfactual family

**J1. The family itself.**  *Statement:* keep the gears and the mirror symmetry (teeth `+-v_q`), let
`v_q` range over `{1..(q-1)/2}`; every member has the same period, the same `prod(q-2)` openings and
the same per-gear density -- only positions move.  `|V(y)| = 30 / 180 / 1440 / 12960` at
m11/13/17/19.  *Calculates:* a clean null model for every alignment statistic.  *Status:*
`[exact: exhaustive at m11..m19; m23 exhaustive in the pinned family (12,960)]`.  *Where:* [S] G;
[La] K; `tooth-counterfactual-percentile.md`.  *Limits:* the rows are NESTED, so they are not
independent draws and no p-value is claimed.

**J2. The record law is family-wide.**  *Statement:* `max(F_2(M), max_{J>=3} Q*_J(M;q')) = F(M+q')`
EXACTLY at every one of 27,570 counterfactual machines, zero exceptions.  *Calculates:* **the
identity that computes `F(M+q')` from the machine below is STRUCTURAL; only the SIZE of `Q*_J` is
arithmetic** -- so the counterfactual obstruction is an obstruction to BOUNDING `Q*_J`, a strictly
smaller target.  *Status:* `[exact: 27,570 members]`.  *Where:* [S] G; [La] K (item 69).  *Limits:*
the family is mirror-SYMMETRIC; whether the record law needs the mirror at all is untested (U18).

**J3. The increment law and the budget inequality are not generic.**  *Statement:* the increment law
is violated by 13.3 / 13.9 / 14.5 / 21.7 / 22.3 per cent of members at 7->11 .. 19->23, GROWING with
the machine; pinning `v_{q'} = round(q'/6)` drops it to 0 / 0 / 1.1 / 6.5 / 5.7 per cent; the budget
inequality itself fails at 0.00-0.56%.  *Calculates:* the decomposition of where the difficulty lives
-- THE NEW GEAR'S TOOTH POSITION carries most of it and the old machine's arithmetic the rest.
*Status:* `[exact: exhaustive m11..m19 and the full 142,560-member 19->23 family; a 601-member SAMPLE
at 23->29]`.  *Where:* [S] G; [La] K.  *Limits:* no argument using only "same gears, same density,
symmetric teeth" can prove the increment law.

**J4. `L` is not capped on the family by the real machine's constant.**  *Statement:* max `L` over the
full family is 1, 3, 3, 3, 5 at 7->11 .. 19->23 against the real machine's 0, 1, 1, 1, 2; the `L = 5`
member is `V(19)`'s `(1,2,5,2,1,5)` with `v_23 = 9`, word `[5,18,5,18,5]`, residues mod 23
alternating 16, 21; EVERY deepest word at every step is LITERAL.  *Calculates:* any proof that `L` is
bounded must use the teeth -- CRT, the mirror, T2/T3, R89/R90 and the record law all hold at every
member.  *Status:* `[exact: exhaustive at five steps, 165,584 rows]`.  *Where:* [S] G; [La] K (items
78, 86).  *Limits:* the real machine is at or below the family median at every step.

**J5. The residual violators are not a congruence on `F(M)`.**  *Statement:* `F(M) mod q' in {0,a,b}`
has sensitivity 34.0% at 17->19 and 5.6% at 19->23; the best predictor of that form reaches 57.9%
balanced accuracy; 94.4% of residual violators are not congruent to a legal letter; the depth-3
attaining middle is the old record in 0.0% of 19->23 violators.  What DOES describe the residual set
is a DEPTH-4 word-legal run (70% of 19->23 violators are invisible at depth 3) plus min flank
`> s_min`.  *Status:* `[exact: pinned family, three steps]`.  *Where:* [S] refuted list; [La] K
(items 70, 71).  *Limits:* `H1` HOLDS at m31 while all three of m31's failing rows fail there.

**J6. Where the teeth enter `L`.**  *Statement:* `P(L>=3 | bare alternation {5,7}-admissible)` =
0.006 / 0.101 / 0.272 / 0.320 and `P(L>=3 | not admissible)` = 0.0000 / 0.0000 / 0.0001 / 0.0000 at
13->17 .. 23->29, with 0 of 4 / 605 / 19,408 / 1,340 bare-letter `L>=3` words inadmissible.  The real
machine's alternation is NOT admissible at 13->17 `(6,11,6)`, 17->19 `(6,13,6)`, 23->29 `(10,19,10)`
-- so its `L <= 2` there is decided by gears 5 and 7 alone -- and IS admissible at 19->23 `(8,15,8)`.
*Calculates:* gear 5's tooth explains 17.3% of `L`'s variance at 17->19 (more than the incoming
tooth's 12.5%), while every old gear above 7 explains under 1%.  *Status:* `[exact: exhaustive on the
family; the necessity direction kernel-checked (`BareAlt.no_gapWord`)]`; an OBSERVATION with one
exception class (the shifted letter `a+q'`), not a theorem.  *Where:* [S] G; [La] K (item 79).
*Limits:* the two channels are near-orthogonal and explain together only 36-42% of `L`'s variance;
`L` is NOT monotone in the letter size.

**J7. The twin machine is a low-`F` outlier.**  *Statement:* `F(twin)` sits at the 20.0 / 18.1 / 26.4
/ 17.1 / 11.9 percentile at m11..m23, ~10-15% below the median, never the minimum, in a family whose
maximum is 1.6-1.9x the truth; the placement STRENGTHENS WITH DEPTH at the two largest machines
(m19: 17.1 / 12.3 / 6.3 for `F`/`F_2`/`F_3`; m23: `F` 11.9%, `F_2` 3.1%).  *Calculates:* the real
phase vector IS distinguished, in the favourable direction; the increment law's own margin puts the
twin at the 66.8-83.3 percentile.  *Status:* `[exact: exhaustive m7..m23, pinned at m23]`.  *Where:*
[S] G; [La] K.  *Limits:* the BUDGET SLACK is UNDISTINGUISHED (59.0 / 37.2 / 49.3 percentile); the
mechanism is OPEN with three candidates dead.

**J8. The depth-2 slack and its one family failure.**  *Statement:* `F_2 >= 2 d_0` always, so
`F_2 <= F + q'` can fail by that self-mirror 2-run alone; over 14,616 exhaustively enumerated old
machines it fails at exactly ONE, `V(19)`'s `(1,1,4,3,5,2)` with `F = 26`, `F_2 = 50`, `d_0 = 25`;
excluding wrap-pair members the minimum slack is 8/6/6/5/4/9.  *Calculates:* the only depth-2 failure
mode found is the self-mirror 2-run -- exactly the one depth at which the mirror lever needs a
hypothesis.  *Status:* `[exact: gated at 15,217 old machines]`.  *Where:* [La] K (items 80, 81).
*Limits:* the real machine's depth-2 slack is ORDINARY (23.7-86.5 percentile).

**J9. The section-view counterfactual: the real teeth sit on the densest class.**  *Statement:*
pooled over sections `q' >= 1000`, moving gear 13's teeth to ANY other `v` leaves 3.6-3.8% MORE
survivors than the real teeth, gear 7's 6.4%, and all moved positions agree with each other.
Mechanism, parameter-free: among the columns no gear but `s*` touches, the tooth class is richer than
every other by 1.160 (`s*=7`), 1.202 (13), 1.273 (31), against the cofactor model
`ln n / ln(n/s*)` giving 1.138, 1.190, 1.271.  *Calculates:* the real machine removes more survivors
than any counterfactual teeth would, in every section.  *Status:* `[measured: 667 sections; a
pre-registered prediction REFUTED]`.  *Where:* [S] F; `word-tree.md` 9.3.  *Limits:* it explains the
SIGN, not the size; it is the section-view face of the period result.

**J10. The real phase vector is not extremal for anything else.**  *Statement:* `+-u'` ranks 1716th
of 11,550 on overcount and 2536th on lone in the `{5,7,11}` two-teeth space; argmax/argmin only in the
degenerate `{5,7}` mirror space.  *Calculates:* "special point of phase space" means only "the census
is deterministic".  *Status:* `[exact: full enumeration]`.  *Where:* [Ar] lateral r2.  *Limits:* a
negative.

### K. Certificates: LP duals, CSP, potentials, convexity

**K1. The case-split covering certificate.**  *Statement:* "machine `y` has a fully blocked stretch of
width `W`" is a covering polytope with one 0/1 variable per phase tuple of every gear and pair; an
infeasibility certificate is an exact rational DUAL proving `F(y) <= W` from the primes alone -- no
census, no period, no word list.  Fixing the phases of the `k` smallest gears (the CASE SPLIT)
shrinks the position set; a certificate in every case certifies the rung.  *Calculates:* ten rungs
7->11 .. 41->43, the eleventh 47->53 at `W = 171` by 8,077 certificates, tight `F` at five machines
(`F(19) <= 25`, `F(23) <= 34`, `F(29) <= 43`, `F(31) <= 58`, `F(37) <= 88`), and the increment-law
upper half at seven steps.  *Status:* `[exact: rational duals re-checked from their own integers;
kernel-transcribed at 19->23 (`CaseCert23`), 29->31 (`CaseCert31`), 31->37 (`CaseCert37`, 385 cases
in 35 tiers) and three increment rungs (`IncCert23/29/31`)]`.  *Where:* [La] J; lp-duality.md r29-30;
formalist R27.1-R30.0.  *Limits:* cost is a PRIMORIAL in `k` (1, 5, 35, 385, 5005, 85085 cases); and
`W_inc - F(q')` is negative at exactly one corpus step (31->37), where no sound method can certify at
any `k`.

**K2. The restricted covering vehicle (prescribing OPEN columns).**  *Statement:*
`RelaxStar(gears, W, held, ws, openpts)` runs the composed covering LP on `[0,W)` minus what the held
gears strike minus the required-open positions.  Prescribing open positions does not just shrink the
obligation, **it DELETES BRANCHES OF THE CASE SPLIT for free**: at machine 23, span 40, gear 5 held,
three of the five cases are vacuous; at machine 19, of 1,680 windowed cells, 413 are killed outright
because gear 5 has NO phase leaving all three required-open positions open.  *Calculates:*
adjacent-gap-pair realisability by LP duality -- scan-free exact `F_2(19) = 31` and `F_2(23) = 39`;
and with `openpts = {0,W}`, spectrum HOLES.  *Status:* `[exact: rational arithmetic on every verdict;
every certified verdict carries an exact dual re-verified from a clean rebuild, every refuted verdict
an exact primal point verified in the polytope]`; kernel-checked at the 19->23, 29->31 and 31->37
rungs.  *Where:* [S] H; `restricted-covering-certificates.md`.  *Limits:* NOT exact -- nine unrealised
cells at m19 spans 28 and 30 are not certified, four with exact in-polytope witnesses (genuine
integrality gaps); one of the nine is `(15,15)`, the self-mirror split.

**K3. The lowest-blocker identity.**  *Statement:* if some gear strikes column `x`, then
`1 + #{(a,b) : a<b, both strike x, no gear below a strikes x} = #{a : a strikes x}`.  *Calculates:*
summed over the position set, `sum_a |A_a| >= |pos| + sum n_ab` -- the whole recursion row of the
covering certificate, as a `decide` over `2^m` Booleans, without evaluating an 8.2-million-term
max-cover.  *Status:* `[kernel]`, NO AXIOMS (`CaseSplit.lowest6`, `lowest7`, `CaseSplit5.lowest5`,
`degpos5/6/7`).  *Where:* [S] H; [La] J.  *Limits:* a pointwise counting identity; `n_ab = 0` for
96.4% of the gear-index-1 columns at 29->31, so the row is numerically almost entirely a Kounias row
at the smallest free gear.

**K4. The mirror as a symmetry of the certificate.**  *Statement:*
`reflect(hits(q,r,W)) = hits(q, (1-W-r) mod q, W)` with `reflect(i) = W-1-i`, so the case at `ws` and
the case at `(1-W-ws) mod q` have reflected position sets, isomorphic relaxations and equal `V*`,
`|pos|` and certificate cost.  *Calculates:* decide one case per mirror orbit and TRANSCRIBE the
other -- a free factor 2; every level has exactly one self-mirror case.  *Status:*
`[exact: theorem with proof; gated at every gear of m11..m47 at W = 74, 95, 104, 132; 385/385
transcribed certificates re-verified from JSON alone at 31->37]`.  *Where:* [S] H; [La] J;
lp-duality r29-30.  *Limits:* the transcription is a genuine SECOND certificate -- the float solver
found a different dual 124 times of 385.

**K5. The boundary-blocked translation.**  *Statement:* if `pos(ws+t) = pos(ws) - t` as subsets of
`[0,W)` then the five claims of the mirror theorem hold verbatim with `rho(i) = i - t`.
*Calculates:* the VALUE CLASSES of a case split are exactly the orbits of {mirror, translation},
matching the measured class counts (11 of 35 at m37 `W=95 k=2`; 14 of 35 at m41 `W=104 k=2`) and
giving a further 1.8x at m53 `W=171 k=4` (1,391 classes against 2,503 orbits).  It EXPLAINS round
29's unnamed "value classes coarser than mirror orbits" to the unit.  *Status:* `[exact: theorem;
484 translation transcriptions from 330 of 385 certificates, every one re-verified from JSON]`.
*Where:* [S] H; [La] J.  *Limits:* prior-art check not run; the boundary condition is invisible to a
test of "every case", which is why round 29 wrongly recorded "it is not a translation".

**K6. Pairwise convexity computes the record through m17 and provably stops at m19.**  *Statement:*
`L*(y)` (the level-2 Sherali-Adams threshold) equals `F` exactly at m11, m13, m17 (7, 11, 18); at m19
`L* = 27` against `F(19) = 25`, `V(25) = V(26) = 0`, and PSD does not repair it.  **Every certificate
of `F(19) <= 26` must use THREE-gear information.**  Vacuity ratio `L*/F` = 1.000, 1.000, 1.000,
1.080, 1.647, `>= 1.721` at m11..m29.  *Status:* `[exact: soundness proved; every claimed bound
carries an exact rational dual verified in integer arithmetic; the m19 SDP verdicts are numerical and
flagged]`.  *Where:* [S] H; `covering-hierarchy-exactness.md`.  *Limits:* three certificate families
(potentials, covering duals, moment hierarchies) fail along the SAME axis -- arity, not convexity.

**K7. Overlap counting.**  *Statement:* `f(L) = maxgroupkills(group,L) + sum over remainder of
maxkills(q,L)`, and `f(L) >= L` is necessary for a run of `L` to be coverable, so any single `L` with
`f(L) < L` certifies `F_k <= L`.  *Calculates:* `y=23: 34 true vs 50`; `y=29: 43 vs 135`;
`y=31: 58 vs 1043`; vacuous from `y = 37`.  *Status:* `[exact: certificate, measured to y = 43]`.
*Where:* [Ar] `twin-prime-program.md` 19.  *Limits:* the group must contain all but a `1/log y`
fraction of the gears and its period is still exponential -- "overlap counting done this way is the
period scan wearing a disguise".

**K8. The extremal covering exhausts every gear.**  *Statement:* at `y = 17, 19, 23, 29` the extremal
covering leaves NO gear idle; efficiencies all 1.000 at `y=19, L=74` (146 incidences, 1.973 per
position) and 0.833-1.000 at `y=31, L=173` (370 incidences, 2.139); multiplicity is SPREAD
(`{1:56, 2:63, 3:37, 4:10, 5:5, 6:2}` at `y=31`), and average multiplicity matches `2 sum 1/q` almost
exactly.  *Calculates:* why capacity bounds fail -- a counting bound needs `mult` to exceed
`2 sum 1/q ~ 2 log log y` while measured `mult ~ 2`.  *Status:* `[exact: witnesses at y = 19, 23, 29,
31]`.  *Where:* [Ar] `twin-prime-program.md` 37b-37c.  *Limits:* "the capacity bound is close to
achievable and its failure is not slack".

**K9. Translation and reflection symmetry of coverings (the record search).**  *Statement:* shifting
every offset by `t` maps coverings to coverings, so `t` can be chosen to put gear 3 at offset 0 (a
factor of 3); reversing the run maps `q`'s blocked pair to another adjacent pair, so coverability is
reversal invariant, with the residual involution `R(o) = (L-2-o) - s (mod q)` broken by pre-assigning
gear 5.  *Calculates:* the record for `y = 29, 31, 37, 41, 43` (`F(2,y) = 129, 174, 264, 273, 309`)
where the period scan cannot reach; `F(2,47) = 354`, `F_k(47) = 118`.  *Status:* `[exact: y <= 47;
all eleven values y = 5..41 recomputed from L = 1 with the break in place, every one matching]`.
*Where:* [Ar] `twin-prime-program.md` 5.  *Limits:* `y = 53` is out of reach this way.

**K10. Elementary lower bound on the record.**  *Statement:* pin every gear up to `z` at offset 0 and
spend one gear from `(z,y]` on each remaining position:
`F(2,y) >= max{L : |S(2,z) cap [0,L)| <= pi(y) - pi(z)}`.  *Calculates:* `y=29 -> 53` (true 129),
`31 -> 59` (174), `37 -> 74` (264), `41 -> 83` (273), `43 -> 107` (309) -- better than the trivial
pairing bound by a factor of 4.  *Status:* `[exact: computable by finite check]`.  *Where:* [Ar]
`twin-prime-program.md` 14a.  *Limits:* still a factor of 3 below the truth -- "the algorithm's
structure yields elementary lower bounds easily and upper bounds not at all".

**K11. Left-taut equivalence.**  *Statement:* `Cov(L)` holds iff there is a covering additionally
leaving column `-1` unstruck by every gear; consequently every gear may drop its two offsets `q-2,
q-1`.  *Calculates:* a per-`L` pruning rule collapsing the branch factor at every leftmost-uncovered
column.  *Status:* `[exact: exhaustive over ALL offset tuples, y = 11/13/17, at every L from 1 to
F+2, zero mismatches]`.  *Where:* [Ar] harvester r8-r9.  *Limits:* UNSOUND combined with the
mirror-canonical `o5` halving; the canonicalisation was removed.

**K12. The CRT-tuple scan technique.**  *Statement:* a direct `decide` over residues mod 5005 does
not terminate; quantifying over the CRT TUPLE with per-gear shifts gives the same 5,005 cases with
every modulus a single digit and decision-tree depth `<= 13`.  *Calculates:* makes "is this column
open under all gears" a per-gear coordinate test.  *Status:* `[kernel-validated technique]`.
*Where:* [Ar] formalist r11.  *Limits:* keep each declaration at or below roughly `5x10^3` tuples.

**K13. Machine certificates in the kernel.**  *Statement:* machine 13 -- `gap_le : b - a <= 11`,
`pair_sum_le : c - a <= 16`, `gap11_realized`, `pair16_realized`, `alpha1_certificate`,
`lemma1_at_13`, `tierA_forbidden`, `no_11_11_chain`; machine 17 -- `gap_le <= 18`,
`pair_sum_le <= 25` (tight, 24 fails), `alpha1_certificate : 9(c-a) <= 9x18 + 4x19`, `lemma1_at_17`.
*Calculates:* the exact record and pair-record of two whole machines, with explicit witness columns.
*Status:* `[kernel]` (`Machine13.w11` and `w16` depend on NO axioms; machine 17's `w18All`/`w25All`
-- an 85,085-tuple period scan -- need only `[propext]`).  *Where:* [Ar] formalist r11/r15.
*Limits:* kernel-evaluation cost, not mathematics, is the barrier past this size.

**K14. The covering-count conjecture.**  *Statement:* `N(L) <= P(1-d)^L`, whence
`F_h(y) <= ceil(log P / -log(1-d))`, of order `y log^2 y`, below the `y^2/2` the window requires for
every `y >= 23`; with exact values settling `y <= 43` the two ranges overlap.  *Status:*
`[conjectured]`; verified with zero violations for every gear set containing 3 up to `P = 4.8e6`, and
exhaustively at `y = 19, 23, 29, 31` over 4.8e6, 1.1e8, 3.2e9 and 1e11 offset vectors, worst ratio
exactly 1.000000.  *Where:* [Ar] `covering-bound-route.md` 1-3.  *Limits:* it FAILS for gear sets
omitting 3, and the mechanism is adjacency annihilation at `q = 3`, so any proof must use that and
the SEPARATIONS; the bound actually needed is far weaker (`rho <= exp(-2 log P / y^2)`).

### L. The covering / hazard line, and the early-route objects

**L1. The CRT collapse: there is no design freedom.**  *Statement:* the offset vector is exactly a
single residue mod `P`, so the uncovered set is a TRANSLATE of the single pattern
`{n : n, n+1 both coprime to P}`; hence `F(2,y) = 1 + the maximum gap of that pattern`.
*Calculates:* reduces the whole target to the maximum gap of ONE explicit pattern, with no design
search.  *Status:* `[exact: proved (short CRT); the all-offsets-1 configuration attains `F(2,y)`
exactly at y = 7..29, ratio 1.000 in all seven cases, the last by segmented sieve over 3.2e9
positions]`.  *Where:* [Ar] `status.md` 4a, `handover.md` 3.17.  *Limits:* "the exhaustive offset
searches explore TRANSLATIONS, not designs"; any argument appealing to choosing offsets cleverly is
appealing to a translation.

**L2. The exposure criterion (the master rule).**  *Statement:* with
`W_q(S) = {s-1, s : s in S} mod q`, gear `q` is FORCED to strike one of `S` exactly when
`W_q(S) = Z_q`; hence `#{m : every position of m+S is open} = prod_q (q - |W_q(S)|)`.  *Calculates:*
everything else in the covering line -- forbidden configurations, the `c_j` decomposition, the hazard.
*Status:* `[exact]`.  *Where:* [Ar] `forbidden-configurations.md`.  *Limits:* adjacent (halved) frame.

**L3. The minimal size law.**  *Statement:* gear `q` can be forced to strike one of `S` only if
`|S| >= (q+1)/2`, and the bound is ATTAINED; exposure form -- ANY `(q-1)/2` positions can be
simultaneously open to gear `q`, whatever their spacing.  *Calculates:* how many columns you may
demand open before a given gear must strike one; gear 3's and gear 5's blocking laws are its first
two cases.  *Status:* `[exact: proved with an attaining construction; exhaustive to q = 19, the
construction checked for all 45 odd primes below 200]`.  *Where:* [Ar]
`forbidden-configurations.md` 1.  *Limits:* existence of a simultaneously-openable set, not its
position inside the window.

**L4. Minimal span grows like `1.9q`; large gears force nothing new.**  *Statement:* restricting
positions to multiples of 3 and minimising span gives `span = 6, 12, 18, 24, 30, 36, 42, 54, 60` at
`q = 5..31`, ratio 1.20-1.94; gears 29 through 47 contribute ZERO minimal forbidden configurations
beyond those from gears `<= 23`.  *Calculates:* which gears can add new local obstructions -- only
the small ones.  *Status:* `[exact: minima by bitmask DP; the gear census exhaustive inside a box of
word length 16, reproduced at a smaller box]`.  *Where:* [Ar] `forbidden-configurations.md` 2, 5.
*Limits:* THE ANTIDICTIONARY IS NOT FINITE -- at both box widths the longest minimal forbidden word
equals the box length and the per-length count is still rising.

**L5. Minimal forbidden gap words of the small gears.**  *Statement:* gear 5 forbids ten
(`11, 13, 16, 24, 31, 42, 61, 121, 151, 222`); gear 7 seventeen; gear 11 one hundred and seventy, of
lengths 5 to 9; gear 13 has one of span 24.  (Letters are gap/3.)  *Status:* `[exact: enumeration]`.
*Where:* [Ar] `forbidden-configurations.md` 2.  *Limits:* small gears only.

**L6. Admissibility is factor-closed.**  *Statement:* `S' subset S` implies
`W_q(S') subset W_q(S)`, so every factor of an admissible gap word is admissible.  *Calculates:* a
level-by-level search -- extend admissible words by one letter.  *Status:* `[exact]`.  *Where:* [Ar]
`forbidden-configurations.md` 5.  *Limits:* none.

**L7. The factorisation law.**  *Statement:* `|W_q(S)| = w(S)` for every gear `q > span(S) + 1` (the
threshold is `span+1`, not `span`), so gears above `span+1` contribute `prod (q - w(S))`, seeing only
size and adjacency, never placement.  **All placement dependence lives in the gears at or below
`span(S)+1`.**  *Calculates:* separates the gear set from the run length completely.  *Status:*
`[exact: verified with zero exceptions for L = 9, 12, 15, 18 against gear sets to 31 once the
threshold was corrected]`.  *Where:* [Ar] `forbidden-configurations.md` 3.  *Limits:* caught by
check, not inspection.

**L8. The `c_j(L)` decomposition.**  *Statement:*
`N(L) = sum_j c_j(L) prod_{L < q <= y} (q - j)` with `c_j(L)` independent of `y`; tables given for
`L = 2, 6, 12, 21, 24`.  *Calculates:* `N(L)` for any `y` from a `y`-independent table.  *Status:*
`[exact: verified by reassembling N(L) at y = 13, 19, 23, 31]`.  *Where:* [Ar]
`forbidden-configurations.md` 4.  *Limits:* VALIDITY CONDITION -- `c_j(L)` is built from gears
`q <= L`, so the hazard at `L` needs `y >= L+1` (at `y=13, L=21` the formula returns 406,008 against
a true `N(21) = 312`).

**L9. The per-`J` recipe does scale.**  *Statement:* the `2^L`-subset sum for `c_j(L)` is zero the
moment one gear has `W_q(T) = Z_q`, and a depth-first scan pruning on the first fully covered gear
visits only 2, 4, 10, 19, 61, 181, 289, 721, 2548 subsets at `L = 1..39` against `2^39 = 5.5e11`;
EVERY visited subset contributes.  *Calculates:* `c_j(L)` in closed form to `L = 39`; each tight case
becomes an explicit finite inequality between products over the gear set.  *Status:* `[exact: built
and measured -- an explicit REVERSAL of the earlier "does not scale" judgement]`.  *Where:* [Ar]
`forbidden-configurations.md` 9; `handover.md` method audit.  *Limits:* one condition per `L`, and
the tight `L` do not terminate -- the NUMBER of conditions still grows like `F_k ~ 0.055 y^2`.

**L10. The tight `L` are a short fixed list of small absolute values.**  *Statement:* the minima of
`h/d` sit at block starts `L = 1, 3, 6, 9, ...`; ranked, the tight starts are `1, 6, 3, 21, 15, 9,
24` with 30, 39, 45, 54 behind -- the same small absolute values for every gear set, stable from
`y = 13` through `y = 401` and FROZEN in order from `y = 199`.  The four tightest are `1, 6, 3, 9`,
exactly the four already proved.  *Status:* `[measured: y = 13..401; y=23 computed twice
independently; y=29 and 31 over 3.2e9 and 1e11 offset vectors]`.  *Where:* [Ar]
`forbidden-configurations.md` 8.  *Limits:* the falsification test is whether the tight list keeps
growing with `y`; it has not through `y = 29`.

**L11. The gap identity and the hazard form.**  *Statement:* `N(L) = sum over gaps g of max(0, g-L)`,
so `G(L) = N(L) - N(L+1)` and `h(L) = 1 - N(L+1)/N(L)`; the target `N(L) <= P(1-d)^L` is exactly
`h(L) >= d` for every `L` -- the hazard rate of the gap distribution is at least `d` everywhere.
*Calculates:* "the cleanest form the open problem has taken: no offset vectors, no coverings, no
separations, and no sub-problems".  *Status:* `[exact: zero mismatches for gear sets {3,5,7} through
{3,5,7,11,13,17}, 22,275 gaps in a period of 255,255]`.  *Where:* [Ar] `covering-bound-route.md` 15a,
15c, 17a.  *Limits:* the hazard is NOT monotone -- `h` dips at `k = 3, 6, 8, 9` in every set tested.

**L12. The proved hazard cases, and the extremal gear set.**  *Statement:* `L=1` is
`C_y = prod(1 - 4/(q-2)^2) <= 1`, true term by term; `L=3` reduces to
`prod(1-2/q)^2 >= prod(1-4/q)`, holding factor by factor; `L=6` uses `C = (8/3)B` exactly and `Y = 0`
by the gear-5 law; `L=9` holds from `y = 17` with `y = 7, 11, 13` checked exactly in integers.
`{3,5,7}` IS the extremal gear set -- tight at `L = 1, 6, 9` with exact equality at 6 and 9.
*Calculates:* `kappa(L)` in closed form in rationals for small `L`; the k-frame limit
`kappa(2) = 2 - (11/3)C = 0.5448`.  *Status:* `[exact: proved for every y at L = 1, 2, 3 in the
column frame -- "to my knowledge, the first cases of form (b) proved in the k-frame"]`.  *Where:*
[Ar] `covering-bound-route.md` 18b-18d, 20, 23.  *Limits:* no uniformity in `L` -- "they are the easy
sliver".

**L13. The pair weight `psi`.**  *Statement:* only `delta = 0 mod 3` contributes (the gear-3 law
reappearing term by term rather than imposed), and
`psi(delta) = 3C prod_{q | delta} (q-2)/(q-4) prod_{q | delta^2-1} (q-3)/(q-4)` with
`C = 0.396880415`; the MEAN of `psi` is exactly 3 (running means 2.6926 .. 2.99976 to `L = 3e5`).
*Calculates:* the exact coincidence multiplicity of two columns at separation `delta`;
`kappa(L) = L - sum_{delta <= L} psi(delta) + small`.  *Status:* `[exact: closed form checked against
direct pattern counts for 150 deltas at y = 19 and against measured kappa at y = 100003 for thirteen
L; the closing inequality `sum psi <= L-1` verified over every block start to L = 5e6 (1.67M cases),
minimum 1.6343 at L = 6]`.  *Where:* [Ar] `forbidden-configurations.md` 8c.  *Limits:* an expansion
in `d` at fixed `L` -- it controls `L << 1/d` while the records live at `L d ~ 8-18 and growing` (THE
REGIME GAP); closing it needs control of an AVERAGE, not a mechanism.

**L14. The repulsion form.**  *Statement:* `kappa(L) >= 1` says exactly that conditioning on an open
column at 0 REDUCES the expected number of open columns in `(0,L]` by at least `d`; `v(1) = v(2) = 0`
outright.  *Calculates:* the requirement is `mean psi <= 3 - 3/L`, measured `3 - 104/L` at `L = 5e6`
-- a margin of about 30x at large `L`, and at the tight point `L = 6` mean `psi = 2.183` against
2.5 required.  *Status:* `[exact: identity; measured margins]`.  *Where:* [Ar]
`forbidden-configurations.md` 8d.  *Limits:* "the repulsion is measurable but not yet bounded".

**L15. The helper condition.**  *Statement:* one offset serves two positions exactly when their
distance equals that gear's tooth separation, i.e. `q | 3 delta - 1` or `q | 3 delta + 1` -- so the
helpers at distance `delta` are exactly the prime divisors of `3delta -+ 1`, at most
`2 log_2(3delta+1)` of them (six at `delta = 2`, ten at 10, twenty-four at 1000), NEVER all `pi(y)`.
*Calculates:* the helper count at any distance.  *Status:* `[exact: zero mismatches over gears 5 to
299 and delta 1 to 39]`.  *Where:* [Ar] `covering-bound-route.md` 8b.  *Limits:* it governs the
`L = 2` violation but not the general step.

**L16. Adjacency is annihilated at gear 3.**  *Statement:* two positions at distance `>= 2` forbid
four offsets (factor `1 - 4/q`, negatively correlated) while ADJACENT positions forbid only three
(factor `1 - 3/q`, positively correlated for `q >= 5`) -- but at `q = 3` that factor is exactly 0.
*Calculates:* `Pr[adjacent both open] = 0.000000` for `{3,5,7}` and `{3,5,7,11}` against 0.166234 for
`{5,7,11}`; hence `h(1) = d/(1-d)` exactly.  *Status:* `[exact; measured]`.  *Where:* [Ar]
`covering-bound-route.md` 4.  *Limits:* it controls the bound at `L = 2` exactly, not at every `L`;
in the column frame five of the twelve distances 1..12 are positively correlated for `{5,7,11,13}`.

**L17. The domino.**  *Statement:* `rho(L) <= rho(1)` -- the first step of a blocked run is the cheap
one, because gear `q` blocks the adjacent pair `{o,o+1}` and no later position can share a domino
with position 0.  *Calculates:* the mechanism behind the peak of the hazard at `L = 1`.  *Status:*
`[exact: verified at every L up to F_h for y = 7..19, peak always at L = 1]`.  *Where:* [Ar]
`forbidden-configurations.md` 6b.  *Limits:* "the statement is true jointly and false per gear" --
per-gear conditional marginals RISE under conditioning, by up to 63%.

**L18. The spread lemma.**  *Statement:* for gear `q` with tooth separation `s`, every offset blocks
between `2 floor(i/q)` and `2 floor(i/q) + 2` positions of a run of length `i` -- a spread of at most
2, whatever `i` and `q` -- and when `q | i` the spread is exactly 0.  *Calculates:* in the window
regime no offset of any gear is materially more useful than another, so conditioning cannot
concentrate.  *Status:* `[exact: proved; measured for q = 5, 7, 11, 29]`.  *Where:* [Ar]
`covering-bound-route.md` 9c.  *Limits:* "usefulness is not the only thing conditioning responds to:
WHERE an offset blocks matters as well as how much".

**L19. Smallest-gear-first splitting.**  *Statement:* removing the smallest gear splits `[0,L)` into
its `q_1 - 2` open residue classes, each reindexing to an interval of length `L/q_1` with the
remaining separations rescaled to `s_q q_1^{-1} mod q`; gear 3 leaves exactly ONE sub-problem, gear 5
three, gear 7 five.  *Calculates:* the closing chain `f(L) <= (1-d')^{L(1-2/q_1)} <= (1-d)^L`, so the
whole bound follows from one correlation statement.  *Status:* `[exact: algebra; sub-problem negative
correlation measured with zero violations for {7,11,13}, {7,11,13,17}]`.  *Where:* [Ar]
`covering-bound-route.md` 12a-12b.  *Limits:* the correlation statement it needs is unproved, and the
sub-problem's own bound genuinely FAILS for some induced separations.

**L20. The divisor law for induced adjacency.**  *Statement:* gear `q` becomes adjacent at level
`q_1` iff `q | 3q_1 - 1` (`q_1=5 -> 7`; `q_1=7 -> 5`; `q_1=11 -> none`; `q_1=29 -> 43`;
`q_1=1009 -> 17, 89`), at most `log_2(3q_1)` of them.  *Calculates:* which gears are pushed to
adjacency at each level of the recursion.  *Status:* `[exact: zero mismatches for every `q_1` up to
60 against all gears to 200; explicitly retained as correct arithmetic after the conclusion built on
it was refuted]`.  *Where:* [Ar] `covering-bound-route.md` 12d.  *Limits:* the CONCLUSION drawn from
it was refuted.

**L21. The hole mechanism (why `F_2 - F` is small).**  *Statement:* covering `[0,L)` leaves gear 3
free to choose its offset, but in the HOLE problem gear 3 must AVOID the hole, leaving exactly one
admissible offset, so the class left to gears `>= 5` is FORCED and the hole itself is one position of
that class no longer needing coverage.  **The hole buys exactly one position of slack**, at the cost
of losing the choice of class.  *Calculates:* `F(y) = 1 + max coverable run`;
`F_2(y) = 1 + max run coverable except for one interior position`; `F_2 = 21, 33, 48, 75, 93, 117,
165` at `y = 7..29`, exact in all seven cases.  *Status:* `[exact: mechanical account, validated
against the pattern]`.  *Where:* [Ar] `gear-recursion.md` 4c.  *Limits:* "it is not an argument for
boundedness, and the `y = 29` value shows why one should not claim boundedness from it".

**L22. Maximal gaps are strongly isolated (early form).**  *Statement:* the gaps either side of a
maximal gap are minimal -- `(2,2)` at `y=29`; `(2,2), (2,3), (2,5)` at `y=19`; `(1,5), (3,3), (5,1)`
at `y=23`.  *Calculates:* the trade-off in
`F(M+q) - F(M) = (F_2 - F) + (F(M+q) - F_2)`.  *Status:* `[measured: y = 19, 23, 29]`.  *Where:* [Ar]
`gear-recursion.md` 4c.  *Limits:* CORRECTED -- isolation does NOT explain `F_2` (at `y=29`,
`F_2 = 55` comes from `(30,25)`), and `F_2 - F` doubles at `y = 29`.

**L23. The increment constant `C`.**  *Statement:* `F_adjacent(y) ~ C sum_{3<=q<=y} q` with `C`
measured between 0.808 and 1.354; since the odd primes are a subset of the odd numbers,
`sum q <= (y^2+2y-3)/4` with NO prime counting, so `C <= 1.8` suffices for every `y >= 29`.
*Calculates:* the whole target reduced to one constant.  *Status:* `[exact: steps 2 and 3
elementary; C measured over thirteen exact values]`.  *Where:* [Ar] `gear-recursion.md` 5-6.
*Limits:* `C` is NOT monotone and its supremum is not established; per-step bounds cannot deliver it
(the gear-37 step reaches `2.432q`), so the constant must be argued in aggregate.

**L24. The disjunction, priced.**  *Statement:* minimal admissible diameters from the machine's own
admissibility rule are `w(k) = 2, 6, 8, 12, 16, 20, 26, 30, 32, 36, 42` for `k = 2..12`, matching the
known values; over `(43, 1849]` an 8-tuple of diameter 26 averages 1.192 primes, so some `m` must
carry two -- verified directly, 666 such `m`, the best at `m = 1601` carrying 7 primes among the 8
offsets.  *Status:* `[exact]`.  *Where:* [Ar] `twin-prime-program.md` 14g.  *Limits:* in that same
range the truth is a gap of 2 with 50 twin pairs present, while the framework yields only a gap of
26.

**L25. The route (A)(B)(C)(D)(E), as the archive left it.**  *Statement:* (A) word list finite and
computable from `q' mod 210`: PROVEN, formally partial (the enumeration is computed, not checked);
(B) literal span, chains `<= 6` members, `span < (10/3)q'`: PROVEN and now universal over every even
gap; (C) padded span, `p <= F/q' + 5/6`, onset `F >= q'`: PROVEN, closed in the kernel at r19;
(D) flank bound: OPEN, the sole gap; (E) both-flanks-maximal: PROVEN but later recorded OFF-TARGET.
*Calculates:* the whole increment `incr_k(M,q') <= (alpha/3) q'`, sufficient at `alpha = 2.5` and 3.
*Status:* `[as recorded at the end of round 19]`.  *Where:* [Ar] constructor r15, formalist r18-19.
*Limits:* (D) is open for twins, hence for every even gap -- it is THE SAME open lemma, `d` entering
only through explicit finite constants.

**L26. (D) is the hypothesis localised, not weaker.**  *Statement:* by the word-indexed identity it is
equivalent to `incr_k <= q'`, localised to `<= 6` pinned words per step; `alpha = 3` buys room --
the allowance rises by `q'/6` per word (17%) and the minimum margin over all measured word-steps
rises from +0.83 to +7.  *Status:* `[exact: equivalence; margins measured]`.  *Where:* [Ar]
constructor r16.  *Limits:* none.

**L27. (D) as a mid-tail x mid-tail pair-sum bound.**  *Statement:* what is needed is that the sum of
two gaps at pinned separation `span(w)`, each observed at most `0.81F`, is at most `F + q' -
span(w)` -- weaker in kind than the extreme-value statements the route needed at rounds 8-13, and the
weakest form the requirement has taken.  *Status:* `[open at every step; margin >= 0.19q' measured]`.
*Where:* [Ar] constructor r16.  *Limits:* still Wall V, no prime input.

**L28. Margin trajectory (literal words).**  *Statement:* `min over each step's compatible words of
F + q' - span - FS_max` = +12 (0.923q', word (4)), +10 (0.588, (6)), +12 (0.632, (13)), +14 (0.609,
(8,15)), +20 (0.690, (10)), +16 (0.516, (10)) at 11->13 .. 29->31; the absolute margin GROWS, the
relative margin sits in a flat band `[0.52, 0.92]q'` with no downward trend.  *Status:*
`[exact: full period, six steps]`.  *Where:* [Ar] mechanic r17.  *Limits:* the PADDED tier at 31->37
has its own minimum, `+7 = 0.19q'` -- a different object; neither is shrinking.

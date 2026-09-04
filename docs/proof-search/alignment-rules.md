# Alignment of openings: what the project already knows

Editor's merge of three harvests -- `research/alignment/harvest_shared.md` (85 entries, the
shared log and `docs/novel/`), `research/alignment/harvest_lanes.md` (92 entries, the six lane
files plus the Lean headers) and `research/alignment/harvest_archive.md` (253 entries, rounds
1-19 and the early design documents).  Nothing here is new.  Every statement is one the record
already makes; every one carries its status and a pointer.  Where two harvests state the same
rule the sharper statement is kept and both pointers are given; where they disagree, both are
printed and the disagreement is named.

---

## 0. How to read this

### The vocabulary (fixed)

A **column** `k` is the pair `6k-1, 6k+1`.  Gears 2 and 3 are built into the columns, so the
machine starts at gear 5.

A **gear** is a prime `q >= 5`.  Gear `q` **strikes** column `k` iff `k = +u_q` or `k = -u_q`
(mod `q`), where `u_q = 6^{-1} mod q = round(q/6)` and `6 u_q = q -+ 1`.  Those two residues are
the gear's **teeth**.  The other `q-2` residues are the gear's **openings**.

The **machine** `M = {5..y}` has one gear per prime up to `y`.  Its **openings** are the columns
every gear leaves open: `prod (q-2)` of them per period `P = prod q`.

The **window** of machine `{5..y}` is its certified range -- the columns whose numbers lie below
the square of the next prime.  The **kernel route** is that a machine opening lying in `(y, y^2]`
IS a twin pair, because a composite there has a prime factor strictly below `y`.

The **section** is the window's new part, `p^2 < 6k+1 < q'^2`, where `p` is the previous gear.

A **stretch** is a run of consecutive columns anywhere in the period.

The **record** `F(M)` is the longest stretch with no opening, in the max-gap convention: the
distance between consecutive openings.  `F_j(M)` is the largest sum of `j` consecutive gaps --
the longest stretch spanning `j-1` interior openings.  `F_1 = F`, `F_2 = F_2`.

**Alignment** is the thing this document is about: the gears' openings coinciding at a column.
The live question is where and when a new gear's openings line up with the openings of the
machine below it, inside the window.

The **budget inequality** is `F(M + q') <= F(M) + q'`.  It is the **target**.  It is accepted as
true here; the project is not entertaining that it is false.  It is never called a law, and
nothing measured in this document is called a law.  (Where the docs already carry a name --
"hit law", "chain law", "merge law", "alignment law" -- those names are kept, because each of
them is a kernel-checked or proved theorem.  The one exception flagged in the text is the
"increment law", which is kernel-checked only at the literal steps.)

### Status tags

- `[kernel]` -- checked in Lean, with the declaration named.
- `[exact: <scope>]` -- a proof, or an exact computation over the stated scope (full periods,
  every prime in a range, an exhaustive enumeration).
- `[measured: <scope>]` -- a measurement over the stated scope.  A measurement never becomes a
  law here, however many machines it holds at.
- `[conjectured]` -- stated, believed, not established.

### Two translation hazards, both load-bearing

**Hazard 1 -- "window".**  The lane documents and `docs/novel/` say *window* for a run of `J`
consecutive gaps of `M` (equivalently a stretch spanning `J+1` consecutive openings).  In this
document that object is a **J-run**, and *window* means only the certified range.  Every
statement below has been translated; the harvests had already done the translation, and
`harvest_shared.md` records it as the load-bearing note.

**Hazard 2 -- `F_2(M)` versus `F(2,y)`.**  `F_2(M)` is the depth-2 spectrum: the longest stretch
spanning three consecutive openings.  `F(2,y) = 3 F(y)` is the member-unit twin ladder in the
adjacent (halved) frame.  They are different objects and they collide in the notation.  This
document uses `F_j` throughout and never writes `F(2,y)` except where the adjacent frame is
explicitly the subject.  (Recorded as a hazard by Harvester r30, `agents-shared.md` "THE
COLLISION LIST".)

A third, smaller, convention note: `anchor-235.md` section 9 uses a **blocked-count** convention
`F_bc = F - 1`, and says so.  Everything here is in max-gap units unless flagged.

### Letters used throughout

For the incoming gear `q'`: `u' = round(q'/6)`, `c = 6^{-1} mod q'`, `d' = 2u' = 2c`,
`a = d'`, `b = q' - a`, so `a + b = q'` and `3a = q' -+ 1`; `s_min(q') = min(a,b) = a`.
A gap value `v` is a **legal letter** iff `v mod q' in {0, +a, -a}`; `0 mod q'` is **padded**,
`+-a` **literal**.  `L(M)` is the length of the longest realised legal word (a T3-alternating run
of legal letters occurring as consecutive gaps of `M`).

---

## 1. The columns and one gear

### 1.1 The tooth rule

Gear `q >= 5` strikes column `k` iff `k = +-u_q (mod q)`, `u_q = 6^{-1} mod q = round(q/6)`,
with `6 u_q = q -+ 1`.  The two struck residues are the teeth; the other `q-2` residues are the
gear's openings.  The two arcs between the teeth have lengths `2u_q - 1` and `q - 2u_q - 1`, so
every distance between two teeth of one gear is `2u_q` or `q - 2u_q`, and the two sum to `q`.
`[kernel]` (`TwoTeeth.kill_spacing`, `kill_spacing_min`, `kill_period`, `teeth_letters`,
`gear_side`), asserted numerically for every prime gear `5..199` and to `q = 100000`
(`research/check_two_teeth.py`).  *(lanes A "tooth rule"; shared A T1; archive Part 1.)*

Three corollaries the record keeps separately:

- **The shield.**  `k = 0 mod q` is not a tooth: gear `q` divides the midpoint `6k` there, so it
  provably divides neither member.  The shield sits at the exact centre of the **short umbrella**
  (length `2u'-1`, `u' = min(u, q-u)`); the **long umbrella** has length `q - 2u' - 1`.  The two
  arcs grow linearly with the ratio locked at 1:2 (short `~ q/3`, long `~ 2q/3`), because the
  tooth separation is `3^{-1} mod q`.  `[exact: gears 5..47, tabulated to 1009]`
  *(archive `umbrellas-and-shields.md`.)*
- **The self-blocking law.**  The low tooth is `u' = round(q/6)`, which is exactly the column
  containing the gear's own pair: every gear strikes the column that contains itself.
  `q = 5 mod 6` puts the left-kill tooth low, `q = 1 mod 6` the right-kill tooth.
  `[exact: gear table 5..47]` *(archive; the two other "observed tooth patterns" -- alternation
  of low/high on successive gears, and tooth differences running 3,5,7,9,... -- BREAK at prime
  gaps of 6 (23->29, 31->37); only the `u'` identity survives.)*
- **Teeth are never adjacent.**  `d_q = 2u_q = 3^{-1} mod q`, and `3^{-1} = +-1 mod q` would
  force `q | 2` or `q | 4`.  So if column `x` is struck by `q` then `x+1` is not.  `[kernel]`
  (`AnchorChain.neighbour_of_hit`, from `6u = 1` alone, every gear at once).  This is what makes
  the `x+1` restart correct in the nested next-opening formula, and it gives exactly
  `P(x+1 open | x is a g-hit) = P(open) * g/(g-2)` -- measured `0.2342 = 0.2139 * 23/21` at
  `{5..23}` for `g = 23`, `0.2390` for `g = 19`, `0.2994` for `g = 7`, exact to four places.
  `[exact: full period {5..23}]` *(lanes F; shared A.)*

### 1.2 The incoming gear's tooth, and the letters

For the gear `q'` about to be added, the tooth value is `u' = round(q'/6)` and the two literal
letters are `a = 2u'` and `b = q' - 2u'`, with `a + b = q'` and `3a = q' -+ 1` exactly
(`a = (q'-1)/3` if `q' = 1 mod 3`, `(q'+1)/3` if `q' = 2 mod 3`), so `s_min(q') = a ~ q'/3`.
`[kernel]` (`TwoTeeth.teeth_letters`); `a = 2 round(q'/6)` with `3a = q' -+ 1` asserted at all
2,258 primes `11..20000` (`bare_lemma_r31.py` GATE A1).  For a general even gap `2d` the walk
step is the least positive representative of `+-e * 6^{-1} mod q'`; the twin case is
`round(q'/6)`, and the discarded multiple of 210 contributes a multiple of 70, hence nothing
mod 35 (`LiteralCap.s_eq`).  *(shared A T1; lanes A; archive Part 12.)*

### 1.3 Tooth sharing between twin gears

A twin gear pair `(p, p+2)` carries the **same** tooth value `u' = (p+1)/6`.  The columns both
gears strike are exactly the four CRT classes `k = +u', -u', +u'(p+1), -u'(p+1) (mod p(p+2))`;
at `k = u'(p+1)` the lower member is `6u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2)`, the twin product
itself.  So each twin gear pair marks the window above it at exactly two deterministic columns
and wastes at least two strikes per window on already-dead columns.  `[kernel]`
(`Polignac.twin_product_slot`, `Corridor.product_slotOf`, `twin_product_pin`,
`twin_pin_self_block`, `twin_pin_le`, `twin_split_class_iff`, `twin_mirror_slot`,
`own_slot_pin_gap_two`); `[exact: 60/60 twin pairs to 2000, 81 pairs to 3000]`.
The general form (R6, the roots-of-unity law): column `k` is struck by both `q` and `q'` iff
`36 k^2 = 1 (mod q q')`.  The uniqueness half is sharp -- an odd prime pair `(q, q+g)`
split-killing the column holding `q` itself forces `g = 2`.  *(shared E; lanes I; archive 2, 6.)*

Honest negative attached, on record from round 1: the wasted strikes land on already-decided
columns, so tooth-sharing **counts** gain only `O(T(y))` per window, and over full periods
sharing changes nothing at all -- survivors per period are `prod(q-2)` regardless of phases.
Sharing moves WHERE the waste lands, never HOW MANY survive.  `[exact: proved; asserted at every
machine]`

### 1.4 The columns that are always open

- **Column 0.**  Gear `q` strikes `+-6^{-1} mod q`, never 0, so column 0 is an opening at every
  machine.  It is the anchor of the mirror.  `[exact: elementary]`  Its uselessness is on record:
  column 0 is a primorial multiple, i.e. as far from the window as a column can be.
- **The antipodal columns `(P +- 1)/2`.**  `2s = P+1` gives `6s = 3P+3`, so the members are
  `3P+2` and `3P+4`; a gear striking either would divide 2 or 4.  Equivalently `6s = 3 mod q`
  against teeth at `6u = +-1`, and `3 = +-1 mod q` is impossible for `q >= 5`.  So the antipodal
  gap has length 1 at every machine.  `[kernel]` (`Mirror.antipode_open`, five lines, no
  residues; instantiated `antipode_exposed11`, `antipode_exposed29 : Exposed29 539141103` -- an
  opening exhibited by arithmetic at a machine whose period no kernel will see).
  Consequence: `W_1(g)` (the count of gaps of length `g` per period) is EVEN for every `g >= 2`,
  unconditionally, so the record gap never occurs exactly once.  *(shared D; lanes B.)*
### 1.5 The alignment law, and its recorded limitation

**The longest run of consecutive openings of a gear set equals the long arc of its smallest
gear**, `q_min - 2u_min - 1`, no matter how many other gears are added: smallest gear
`5 -> 2`, `7 -> 4`, `11 -> 6`, `13 -> 8`, `17 -> 10`.  The reason is alignment: by CRT every
relative phase of the other gears occurs somewhere in the period, so there is a position at which
none of their teeth falls inside the smallest gear's long arc, leaving it fully open; adding
gears cannot shorten it because it cannot remove that position.  The shortest run is always 1 --
every gear set has isolated openings.  `[exact: zero failures over 103 gear sets,
research/alignment.py]` *(archive `twin-prime-program.md` 26c-26d.)*

This is the closest the corpus comes to a positive alignment theorem, and its limitation is
exactly the open problem: **the law says SOMEWHERE IN THE PERIOD**, and the period is the
primorial while the certification window is the first `~y^2/6` columns.

Two corollaries the record keeps:

- With gear 5 present the longest run is 2, so in column space **the opening set is exactly a
  disjoint union of isolated points and dominoes** -- nothing longer.  Counts follow:
  dominoes `n_1 = prod(q-4)`, runs `prod(q-2) - prod(q-4)`, singletons
  `prod(q-2) - 2 prod(q-4)`; validity of the `prod(q-2k)` family needs `q >= 6(k-1)`.
  `[exact: corollary; checked at {3,5,7}: 9 singletons, 3 dominoes, 15 points in 12 runs]`
  The pattern therefore structurally supports prime quadruplets at every level.  What it says
  nothing about is the spacing BETWEEN the objects, which is the whole problem.
- Gear 3 blocks one of any two adjacent columns and gear 5 one of any three spaced 3 apart, which
  is where the mod-3 law on the record comes from (`3 | F_h(y)` in the adjacent frame).
  `[exact: proved; checked against all thirteen known F_h values]`

### 1.6 What two gears and three gears do jointly

For any two gears `(q,r)` the joint machine has period `qr`, exactly `(q-2)(r-2)` openings and
exactly **4** double-kill columns: the CRT lifts of the four sign choices `(+-u_q, +-u_r)`.  The
two same-sign lifts are product blocks (both gears on the same member, at the column of `qr` and
its mirror); the two mixed-sign lifts are crossed kills, positioned by the slip inverse
`X_crossed = 1 + q((-2 q^{-1}) mod r)`.  A machine of `n` gears is ONE composite gear (all the
single-gear laws lifted verbatim: both-left tooth `6^{-1} mod P`, low tooth `round(P/6)`, joint
shield at 0) plus a **crossed cloud** of `2^n - 2` mixed lifts, mirror-paired with sign
complement; the columns struck by every gear at once are exactly the square roots of unity
modulo the period, scaled by `6^{-1}`.  `[exact: verified for sets of one to four gears; all six
pairs and all four triples from {5,7,11,13} worked in full; crossed formula verified for all 28
pairs to 29]` *(archive Parts 2 and 6; `pair-anatomy.md`, `twin-prime-program.md` 28b-28c.)*

The record of a lone **pair** falls as the pair grows (5 at 5x7, 4 at 5x11 and 7x11, 3 at 11x13)
while triples give 7, 7, 6, 6: **gaps grow through accumulation of gears, never through the size
of one pair.**  `[exact: all pairs and triples from {5,7,11,13}]`

And the closed-form address of a crossed coincidence for any pair `(q, q' = q+g)`:
`m0 = (-2 q^{-1}) mod g`, `b0 = (2 + m0 q)/g`, `i = (q' - b0) q^{-1} mod 6`,
`x = (q'(b0 + iq) - 1)/6`, mirror at `P - x`.  `g = 2` is the unique gap with `b0 = 1`, so its
split representative `x = u' <= K` is in-window at every scale, unconditionally; every other gap
has floor depth `~P/(6g)` and is alignment-conditional.  `[exact: all 2850 pairs
5 <= q < q' <= 400, and 753,378 pairs at y = 10007, zero failures]`

### 1.7 The corridor: what gears 5 and 7 forbid, anywhere, forever

Gears 5 and 7 jointly leave open exactly 15 residues mod 35:
`E_35 = {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33}`.  Every opening of every machine containing
5 and 7 lies in `E_35`; a stretch of openings with gaps `g_1..g_l` can sit only at base residues
whose partial sums all stay in `E_35` (the **carrier**), and an empty carrier forbids that
configuration at every machine forever, with no scan.  `[kernel]` (`Corridor.exposedSet`,
`exposed_iff_mem`, `endpoint_law`, `adjacency_law`, `forbidden_pairs_count = 294`,
`no_chain_of_forbidden`; `TierA.carrier`, `mem_carrier_of_chain`, `no_chain_of_carrier_empty`).

**The completeness lemma.**  A shape with `n` prescribed open columns can be blocked by gear `q`
only if `q <= 2n` (two teeth forbid at most `2n` of `q` phases, and CRT makes the gears
independent).  *Harvest disagreement, printed as found:* `harvest_lanes.md` and
`harvest_archive.md` state that for `n <= 5` the mod-35 test IS the entire corridor, with gear 11
first entering at `n = 6` and gear 13 at `n = 7` (archive lateral r17,
`research/corridor_complete.py`; lanes item 20); `harvest_shared.md` states completeness for
`n <= 3` and explicitly flags that `n <= 5` is claimed in one place with the same `q <= 2n`
bound (`docs/novel/corridor-law.md` LIMITS).  Both readings agree on the bound itself; they
differ on how far it is claimed to have been pushed.

Two permanent residue facts from the same source:

- **The 32-cap.**  Gears 5 and 7 have both-composite classes at `k = 1` and `k = 34 (mod 35)`,
  whose largest cyclic gap is 33.  So any 33 consecutive columns contain a column with both
  members composite, and a stretch of columns each carrying a prime member is at most 32 long --
  unconditionally, at every scale, from two gears.  `[kernel]` (`Corridor.exists_class_in_run`,
  `both_composite_in_run`, `double_slot_in_run`, `prime_adjacent_run_le`, `n2_packing`).
  Escalation-checked: adding gears does not lower it through gear 23.  Whether `lim L0 = 32` over
  all gears is a Jacobsthal-type question, finitely checkable per gear set and monotone
  non-increasing; nobody has run it.
- **The adjacent-gap exclusion law.**  Three consecutive openings with gaps `(g1, g2)` are
  impossible whenever `(g1 mod 5, g2 mod 5) in {(1,1),(1,3),(2,4),(3,1),(4,2),(4,4)}` -- 6 of 25
  classes, 24% -- at every scale in every machine containing gear 5, and by the completeness
  lemma only gear 5 can block a 3-point shape, so the list is COMPLETE.  `[exact: proved;
  cross-checked against full-period censuses m11..m31, 1,589 populated lag-1 cells, zero in a
  forbidden class]`  Scope: ADJACENT gaps only; at separation `j >= 2` the same classes carry up
  to 35.8M counts.

And two exclusions on regularly-spaced openings:

- **The AP lemma (mod 5, scale-free).**  Openings have `k mod 5 in {0,2,3}`; four terms of an
  arithmetic progression with difference coprime to 5 occupy four distinct residues mod 5, and
  three residues cannot hold four.  So there are NO four openings in arithmetic progression with
  common difference `q'`, for every prime `q' > 5`.  `[exact: proved, verified exhaustively over
  all (r,g) mod 5 with g invertible]`
- **The openings AP theorem.**  An AP of `L` openings has common difference divisible by every
  gear `q < L+2`: 3 equal consecutive gaps require `5 | g`, 5 require `35 | g`, 9 require
  `385 | g`, and `L >= y+2` needs the full primorial.  `[exact: proved; full periods m13..m29,
  zero violations; the longest run of equal consecutive gaps is 3-4 at every machine, with
  `g = 5` exactly every time -- the theorem's minimal witness, realised]`

**The decisive negative on corridors, on record since round 9: escape distance = 1.**  Over all
1,225 gap pairs mod 35, EVERY `(G1,G2)` is within L1 distance 1 of a corridor-allowed pair.  A
near-maximal gap has about 35 candidate lengths in its range, so any residue exclusion is evaded
by a `+-1` slide in one component.  **Corridor arithmetic constrains WHERE top-gap configurations
sit, never HOW BIG they are** -- at modulus 35 and, by the same argument (the exposed set's own
max gap stays `O(1)`), at ANY bounded modulus.  `[exact: all 1225 pairs, plus a general
argument]`  Confirmed from the other side: lifting the modulus `35 -> 385 -> 5005 -> 85085 ->
1616615` adds EXACTLY ZERO exclusions anywhere tier A did not already give zero, at all 16
word-step pairs, and structurally so (`S_m` and `E_m` are unions of lifts).
---

## 2. Adding a gear to a machine

This section is the mechanism: what gear `q'` does to the openings of `M` below it.  Everything
here is one-step -- it takes the machine below as given.

### 2.1 The `q'` copies, the phase bijection, the turn law

Adding gear `q'` to a machine of period `P` gives period `P q'`, which is `q'` **copies** (laps)
of the old pattern laid end to end.  In copy `j` the openings are `o + jP`, and copy `j` deletes
exactly the old openings whose residue mod `q'` sits on a tooth as seen from that lap; the
deleted pair shifts by `-P mod q'` per lap.  Since `gcd(P, q') = 1` the map
`j -> -u' - jP (mod q')` is a **bijection** of `Z_{q'}`.  Hence:

**the `q'` copies realise every deletion phase exactly once, and each opening of `M` is deleted in
exactly 2 of the `q'` copies, one per tooth.**

`[kernel]` (`AnchorChain.copy_phase`, `phase_bijective`, machine-free); the "exactly 2 copies"
count `[exact: m11..m23, hr_twoclass_r30.py A, 2N hits]`.  *(shared A "lap/copy structure";
lanes D "the two-class copy picture", F "phase reduction"; archive Part 8 "the merge transform".)*

The same fact, class by class, is the **turn law**: an open class `k_0` of the machine below
spawns `q'` daughter classes `k_0 + tP`, of which exactly two are struck -- at
`t = (+-u_{q'} - k_0) P^{-1} mod q'` -- and `q'-2` survive, the shield-child among them.
`[exact: verified against brute force over all sub-machines of up to three gears from 5..29 and
all their open classes, zero mismatches, research/slip_algebra.py]`

The **class tree** is the turn law iterated: a node at level `i` is a residue class mod `P_i`, a
description of a column under every umbrella so far; adding a gear splits each node into `q'`
children and kills exactly 2.  The tree is **never extinct** (`prod(q-2) >= 3^n > 0`), and
smallest-representative-first search of it is correct and complete (`k mod P_i <= k`).
`[exact: proved]`

And the tree names the open problem in its own terms -- **the sideways step**: following open
branches controls OPENNESS, not POSITION.  When a branch dies and the search steps to a sibling,
the sibling class's smallest representative can jump by primorial-scale amounts (changing one
level's residue moves the representative by that level's idempotent).  "The tree provably always
has open branches, and one within `F_k(y)` of any point -- but bounding the sideways distance to
the nearest open branch inside the window is Reduction A itself.  Every route in the programme is
an attempt to bound the sideways step."  *(archive `class-tree.md`.)*

What the copies say and do not say: they say where alignments CAN occur, for every phase at once,
turning "does this alignment occur somewhere" into "is there an admissible residue", with no
search over position.  They do not say which of those alignments the old machine realises.

### 2.2 The hit law and the chain law

- **Hit law.**  Gear `g` hops at the lower landing `x` iff `x = +-u_g (mod g)` -- the walk to the
  next opening of `{5..g}` moves past `x` exactly when `g` strikes `x`.  The fraction of landings
  with no hit is exactly `1 - 2/g` (0.8182, 0.8462, 0.8824, 0.8947, 0.9130 at
  `g = 11,13,17,19,23`).  `[kernel]` (`AnchorChain.teeth_eq_phase`, `hop_zero`);
  `[exact: full period of every machine {5..23}, no exception]`.
- **Chain law.**  Two columns lie in a common two-class set `{r, r+d_g} mod g` iff their
  difference is `0`, `+d_g` or `-d_g mod g`.  So two consecutive openings of the lower machine are
  both struck by gear `g` at some phase **iff their gap is `0` or `+-d_g mod g`**, and the gap
  sizes that can carry a second strike are the classes `{d_g, g-d_g, g, g+d_g, 2g-d_g, ...}` cut
  at `F(M)+1`.  A set of openings lies in one two-class set iff every pairwise difference is `0`
  or `+-d`.  `[kernel]`, both directions, every gear at once (`AnchorChain.chain_law`,
  `teeth_eq_phase`); admissible-gap list `[exact: full periods {5..23}]`.

  The measured chain table (layer, `d_g`, lower `F`, admissible gaps `<= F+1`, realised, depth):
  `7, 5, 1, {2}, {2}, 2`; `11, 4, 4, {4}, none, 1`; `13, 9, 6, {4}, {4}, 2`;
  `17, 6, 10, {6,11}, {6,11}, 2`; `19, 13, 17, {6,13}, {6,13}, 2`;
  `23, 8, 24, {8,15,23}, {8,15,23}, 3`.  `[exact: full periods {5..23}]`

  The honest boundary is written into `proofs/AnchorChain.lean`'s own header: **the depth is not
  an algebraic consequence.**  A run inside a two-class set alternates freely, so `D_g` is a fact
  about the lower gap SIZES -- a per-machine measurement, not a residue theorem.

- **Neighbour-of-a-hit** is section 1.1: knowing one gear's hit buys the factor `g/(g-2)` and
  nothing else.  The neighbour of a hit is open LESS often than the neighbour of a random blocked
  column (0.2481) and much more often than the neighbour of an opening (0.0881 -- openings repel).
  "One side of a hit is open" is not a way to find openings.  `[exact: full period {5..23}]`

### 2.3 The merge transform and the merge law

Every gap of `M + q'` is either a gap of `M`, or the merge of a MAXIMAL run of consecutive
`M`-openings all struck by one phase of `q'`.  Deleting `k` consecutive openings merges `k+1`
gaps, so **every new gap is a sum of consecutive old gaps and the record grows only by merging.**
`[kernel]` for the bound form (`MergeLaw.newgap_le`, `newgap_le_step`, `newgap_le_max`,
`Spectrum.merged_eq`); `[exact: the ENTIRE gap histogram reproduced against direct construction
at four extensions, and `F = 18, 25, 34, 43, 58, 88` at the six steps 13->17 .. 31->37]`.
*(archive `gear-recursion.md` 3, `chain-conditions.md`; lanes D "merge law"; shared B.)*

The grammar of a merged run -- five statements, all kernel-checked, which together are the whole
local theory of alignment at one step:

- **T1 (the alphabet).**  `{2c mod q', -2c mod q'} = {a, b}`: the residues by which two struck
  openings can differ ARE the literal alphabet.  Hence
  `Lambda(M) = {v <= F(M) : v = 0 or +-2c mod q'}`, about `3F/q'` letters.
- **T2 (residue necessity).**  Consecutive struck openings sit on `{+-c} mod q'`, so every
  interior gap of a struck run is `0`, `+2c` or `-2c mod q'`; a positive gap in one of those
  classes is at least `2u'`.  `[kernel]` (`MergeLaw.interior_gap_mod`, `floor_of_mod`,
  `Machine23/29/31/37.merge_alphabet`).  Necessary only -- the cover half (everything else in the
  stretch blocked) is a separate condition.
- **T3 (strict alternation, padded letters transparent).**  A spacing `= +2c` moves the struck
  residue `-c -> +c`, `-2c` moves `+c -> -c`, `0` keeps the tooth.  So the nonzero-class spacings
  of an aligned run STRICTLY ALTERNATE and `|#a - #b| <= 1`; padded spacings are transparent.  Two
  equal nonzero classes in a row would need `3c` or `-3c in {+-c}`, i.e. `q' | 2` or `q' | 4`.
  Consequence: two consecutive nonzero letters sum to `>= a + b = q'`.  `[kernel]`
  (`TwoTeeth.spacing_from_lo/_hi`, `next_kill_of_lo/_hi`, `WordLegal.legal_iff_noRepeat`,
  `alt_iff_prefixSum`, `AnchorChain.no_two_up`, `no_two_down`).
- **T4 (deletion spacing).**  Every nonzero-class spacing is `>= a = 2u'`, every padded spacing is
  `>= q'`.  In the adjacent frame (gear 3 included, teeth `{o, o+1} mod q`) the same argument
  gives: two consecutive deletions inside one lap are at least `q-1` apart, and that is TIGHT
  (attained at `q = 13` and `q = 19`).  So a stretch of length `G` carries at most `1 + G/(q-1)`
  deletions.  `[kernel]` (`TwoTeeth.kills_gap_ge`, `kill_spacing_min`); tightness
  `[exact: measured minima 12/12, 18/16, 18/18, 24/22]`.  In the column frame the minimum
  qualifying distance is `min(s, q-s) = (q -+ 1)/3`.
- **T5 (the fuel-span cap).**  `k <= 1 + span/(2u') <= 1 + 3 span/(q'-1)`: at most `~3L/q'`
  aligned strikes in an interior span of `L`, closed form, every gear, forever.  `[kernel]`
  (`TwoTeeth.fuel_span_cap`, `fuel_le`).  Saturated only at 11->13 and 19->23; one below
  elsewhere.  And `span/q'` grows along the ladder, so the cap grows.

Two recorded failure modes of the merge condition, kept so they are not repeated: the
literal-only version MISSES padded links (undershoot 71 against `>= 88` at 31->37), and "all
spacings in `{0, +-2u}` without alternation" is TOO PERMISSIVE (overshoot 45 against 43 at
23->29, on the illegal word `(10,10)`).  The corrected condition is T2 + T3 together.

### 2.4 Saturation: when a far gear always aligns the same way

If `q - 1 > F(M)` then `F(M+q) = F_2(M)` EXACTLY.  No chain of two or more strikes can exist (its
interior gap would need to be `>= q-1 > F(M)`), and every adjacent pair of old gaps IS merged
somewhere, because each opening is struck in some lap.  Above the threshold the increment does not
depend on `q` at all: `{5,7}` plus any of `q = 11,13,17,19,23,29,37,41,53` all give `F = 21`
(adjacent frame), increment 6 every time.  `[exact: proved + 48 (M,q) pairs, zero violations]`

The limitation is structural and is the reason the theorem does not close the ladder: **along the
consecutive chain `q'` is always the next prime and `q' < F(M)` throughout** (47 against 354), so
the compliant regime and the needed regime are PROVABLY DISJOINT.

And the recorded reason chain length cannot be bounded by span arithmetic alone: deleted points
are consecutive openings, so `p_k - p_1 >= (k-1)(q-1)` and `<= (k-1) F(M)`; together
`(k-1)(q-1) <= (k-1) F(M)`, VACUOUS whenever `F(M) >= q-1` -- precisely the regime the chain
lives in.  "Bounding `k` needs the ARITHMETIC of which gaps fall within 1 of a multiple of `q`."
### 2.5 Padded links: the second way a new gear's teeth align

A link is **padded** iff its two openings share a residue mod `q'` -- the same tooth, one lap
apart -- so its interior gap is `0 mod q'`, hence `>= q'`.  Worked address on record (machine 31,
`q' = 37`): openings `k = 634158` and `k = 634195`, both `k = 31 mod 37`, interior column-gap 37,
members differing by `222 = 6 x 37`.  `[exact: two verified witnesses]`

- **Onset gate.**  A padded link's interior gap is a positive multiple of `q'` and is one of `M`'s
  own gaps, so `q' <= F(M)` is NECESSARY.  `[kernel]` (`TierA.onset_gate`,
  `padding_at_most_one_below_onset`).  **Sufficiency is FALSE**: machine 29 has `F = 43 >= 41` yet
  `supply(29,41) = 0` exactly -- 41 is not realised as a gap of machine 29 while 43 is (twice).
  Availability is governed by the gap-value SPECTRUM, not by `F`; `supply(M,q') = hist_M[q']` is
  one lookup, and one gap histogram answers the onset question for every future gear at once
  (machine 29: 2090, 84, 0, 2 at `q' = 31, 37, 41, 43`; machine 31: 26,366 at `q' = 37`).
  Boundary case, sharp: at `q' = F(M)` exactly the supply is 2, precisely the number of maximal
  gaps in the period.
- **Padded cost per frame.**  A padded link costs `q'` in column units, `3q'` halved, `6q'` in
  member units (recorded as NOT a frame conflict).  For a general gap with `3 | e` the cheapest
  padded link costs `q'`; for `3` not dividing `e` all openings lie in one class mod 3, every gap
  is divisible by 3, and the cheapest padded link costs `3q'`.
- **The padding count bound GROWS.**  Each padded link consumes a gap `= 0 mod q'`, so a run of
  span `<= F(M+q')` carries `p <= F(M+q')/c_d`, in the kernel `p <= F/q + 5/6`.  `[kernel]`
  (`padding_count_le`, no axioms).  `padding_three_not_excluded` records that once
  `F >= (13/6) q` the budget stops excluding three padded links.
- **The shape law.**  Two padded links separated by `j` literal links: `j = 0` feasible for 50% of
  the 840 invertible `(g,v)` pairs mod 35, `j = 1` for 32%, `j = 2` ALWAYS IMPOSSIBLE (by the AP
  lemma), `j = 3` 4% of abstract pairs but 0 of 546 actual primes `11..4000`, `j = 4` ALWAYS
  IMPOSSIBLE.  Feasibility is a function of `q' mod 210`.  `[exact: every prime to 4000;
  j = 2, 4 proved]`
- **Adjacent equal padded links: a permanent 50/50 residue criterion.**  Two adjacent padded links
  put three consecutive openings at `r, r+g, r+2g mod 35` with `g = q' mod 35` -- a 3-term AP
  inside `E_35`.  Impossible for exactly the 12 classes `{1,4,6,9,11,16,19,24,26,29,31,34}`
  mod 35, with a perfect DICHOTOMY: the equal shape `(1,1)` is infeasible iff both unequal shapes
  `(1,2)`/`(2,1)` are feasible.  `[kernel]` (`TierA.equal_padding_forbidden_classes`, `_card = 12`,
  `padding_shape_dichotomy` as an iff, `no_adjacent_padded_41`).  Instance: at 37->41 (`g = 6`)
  there are ZERO solutions.  Unlike the round-14 padding lemma (whose spectrum threshold expires
  at 37->41) this never expires.
- **Gear 3 forbids adjacent padded links for every gap `d = 0 mod 6`**, unconditionally, by gear 3
  alone: for `3 | e` the step is `q'` with `3` not dividing `q'`, so `r, r+q', r+2q'` occupy all
  three classes mod 3.  `[exact: proved; all probes q' < 400]`  Structural compensation on record:
  padding is 3x cheaper in absolute terms for `d = 0 mod 6` but can never repeat consecutively
  there.

### 2.6 Firing: alignment is a density factor, never a count factor

Inside a chain, kills sit at the two teeth alternately, so the spacing word's FIRST entry fixes
the orientation and hence a SINGLE firing residue (a word starting with `s` fires iff
`p = -u mod q'`, starting with `q'-s` iff `p = +u`), density `1/q'` per lap -- half the naive
`2/q'`.  Across the new machine's full period **every fuel site fires exactly once**, at the
closed-form address `j = (fire - p) P_old^{-1} (mod q')`.  `[exact: derived and verified with zero
violations over 13,062 sites at 19->23 and 29->31; all four `k=4` sites of machine 29 fire, at
`j = 12, 30, 0, 18`]`

Consequence, and it retired a whole line of hoped-for suppression factors: realised `k`-chains per
new period `= N_k` exactly.  There is no surviving fraction to multiply a ceiling by -- **which is
precisely why the word-indexed statement is an IDENTITY rather than an inequality.**  Incompatible
words never fire anywhere.

### 2.7 Counting the alignments a new gear makes: sparing, multiplicity, monotonicity

- **Sparing count and the sharp `s_min` threshold.**  A run of `j+1` consecutive lower openings
  with offsets `X` is SPARED in exactly `q' - |X u (X+s)| (mod q')` copies; if its span is
  `< s_min(q') = min(a,b)` then `|X u (X+s)| = 2(j+1)` and all `2(j+1)` hitting copies are
  distinct (the two-class form of Holt-Rudd Lemma 3.1).  The threshold is SHARP: the smallest span
  at which two points of one run are hit in the same copy is `4, 6, 6, 8, 10` at m11..m23 --
  exactly the smallest realised legal letter each time.  `[exact: proposition + exhaustive at
  m11/m13/m17 (945/10,395/155,925 runs, j <= 7), sampled at m19/m23]`
  Scope note that matters: `F(M) >= s_min` at every machine from m11 on and `F/q'` grows to 2.5 by
  53->59, so **every stretch that matters is above the threshold** -- which is exactly where the
  one-class literature is silent by construction.
- **Multiplicity of a chain.**  The number of copies in which a run of `k >= 2` consecutive lower
  openings is struck ENTIRELY is `0` if the gap word is illegal, `1` if it is legal and contains a
  literal letter, `2` if it is legal and every letter is padded (both tooth assignments work).
  `[exact: every maximal run of >= 2 hits at m11..m23 -- 8 / 72 / 1,088 / 11,722 / 243,816 runs]`
  Gated negative attached: **the multiplicity does not decrease with `k`**, so the count alone can
  never bound `L(M)`.
- **The depth-0 (dictionary-monotonicity) lemma.**  For every prime `q' > 2(m+1)`,
  `D_m(M) subset D_m(M + q')`: a realised `m`-tuple of consecutive gaps SURVIVES adding a gear,
  because the pattern forbids at most `2(m+1) < q'` phases and CRT supplies a lap with an
  admissible one, and the `m+1` openings are still consecutive.  `[exact: proved, three lines;
  arities 2,3,4 at all six exact pairs 13->17 .. 31->37, arities 5,6,7 at the small steps]`
  The hypothesis is SHARP (first failure at `m = 6,7,8,9` for `q' = 11,13,17,19`).  Prior art to
  cite: Ziller 2020, arXiv:2007.01808, Prop. 2.7 is the one-class arity-1 case, framing attributed
  to de Polignac 1849.  Payoff: 145,907 of 874,087 reverse classes of the m41 arity-4 superset are
  YES BY THEOREM (16.7%), at every span, with no solver.

### 2.8 The realisability CSP: alignment as feasibility, with no period

Column `k` is struck by gear `q` iff `k = +-u_q mod q`, so by CRT a column IS a phase vector
`(a_q)`.  A tuple of gap values with prefix-sum points `X` and interior points
`Y = (0, span) \ X` occurs as consecutive gaps of `M` **iff**

- (open) `a_q not in {+-u_q - x mod q : x in X}` for every gear, and
- (cover) for every `t in Y` some gear has `a_q = +-u_q - t (mod q)`.

The period never appears: `pi(y)-2` variables, domains `<= q`.  `[exact: proved (CRT, one line);
decider `research/crt_dict.py` gated on 2,013 tuples of arity 1,2,3 at m11/13/17 against an
independent pruned inclusion-exclusion counter, and set-equal to the full-period censuses
(`D_4(23)`, 15,696 tuples, tuple for tuple); the corpus ladders `F = 7,11,18,25,34,43,58,88` and
`F_2 = 11,16,25,31,39,55,68` recovered with no period]`

The cost profile is the opposite of the intuition the project carried for five rounds
(Constructor r28): the cover half costs `2^{|Y|}` in the worst case, so **shallow queries are the
dear end and deep queries the cheap end** -- an arity-1 refutation costs 13.2 s at m31, 10-20 s at
m37 and was undecided past 250 s at m41, while an arity-4 decision at m29 costs 3 ms.

### 2.9 The layer law: what the new gear owes in its own section

One layer is one prime retiring into the working set, horizon advancing `y^2 -> y'^2`.  The newly
activated gear's ENTIRE novel workload is (1) retro-closing the old horizon square `y^2` (owed iff
`y^2 - 2` is prime) and (2) the columns `y c` for primes `c in (y, y'^2/y)` -- one to three
explicit numbers per layer, since `y'^2/y < 4y` by Bertrand -- each owed iff its partner member is
prime.  Everything else in the fresh band is closed by the old gears.  `[kernel]`
(`Layer.layer_novelty`, `Layer.minFac_lt_or_eq`, `Layer.eq_mul_prime_of_minFac_eq`);
`[exact: the nine layers 13->17 .. 43->47; seven of nine owe nothing in-band at all, the
exceptions being 221 = 13x17 beside prime 223, and 437 = 19x23 beside prime 439]`
"The tower's complexity lives in the number of layers, never inside one."

Companion, the **shadow law**: a gear supplies nothing below `q^2` -- its ledger line opens at
`q^2`.  `[kernel]` (`Gear.sq_le_of_minFac_eq`, `Gear.R_eq_zero_of_below_sq`).  So a gear first
matters at all in its own section, and full-set sieving is provably equivalent to graded sieving
inside a window.
---

## 3. The record of the bigger machine, from the smaller one

Everything in this section computes `F(M + q')` -- or bounds it -- without building `M + q'`.
That is the practical content of alignment: the new record is an old stretch whose interior
openings the new gear's teeth all reach.

### 3.1 The record law (phase reduction): the widest two-class run, plus its flanks

On ONE lower period, with the lower opening residues mod `g`:

- `D_g` = the longest run of consecutive lower openings whose residues lie in one two-class set
  `{r, r + d_g}`;
- `F_bc(M+g) + 1 = max over such runs (over all phases r) of (gap before) + (run span) + (gap
  after)`, where `F_bc` is the blocked count and `F_bc + 1` the max-gap record.

So the record of the next machine is a maximum, over `g` phases, of "gap before + run span + gap
after" computed on one lower period.  `[exact: {5..7} .. {5..29} and at 31/37 (full lower periods)
and 41 (a deliberate 36.9% partial sweep whose two headline answers are still exact)]`;
`[kernel]` at machine 17 at BOTH ENDS (`AnchorRecord17.record_max` -- phase table
`16 16 18 18 18 16 18 18 16 15 16 18 18 16 18 18 18`, max 18; `surv_shift`, `phase_is_machine`,
`gap18_realized`, `F17_eq_18`; 17 per-phase `decide +kernel` in `AnchorRecord17Core`).

What it buys, in the project's own numbers: `F = 42` for `{5..29}` from 7,952,175 lower openings
instead of a `6.5e9`-column period, **819x smaller**; and the records 58, 88, 91 at 31/37/41
walked a `1.24e12`-column period with no array beyond machine 29.  The phase is not looped over in
practice -- mapping residues by `d^{-1}` turns "`{r, r+d}` for some `r`" into "two adjacent
values", so one rolling max/min per length decides all `g` phases at once and the winning phase is
read back.

Two riders on record: the `L = 1` row is `F_2` of the lower machine every time
(`55 = F_2(29)`, `68 = F_2(31)`, `90 = F_2(37)`), because a run of one deleted opening merges
exactly two lower gaps; and the reduction is CONCEPTUAL, not economic, in a slot-walk kernel
encoding (86,173 kernel column tests against the direct scan's 85,085 -- 1.01x).  To collect the
saving the kernel needs the opening LIST as an object.

The companion **nested next-opening formula** computes the enlarged machine's next opening without
materialising it: with `M` the lower opening predicate and `H` the hit predicate,
`nextG x = nextM^[k+1] x` when the first `k` lower openings after `x` are hits and the `(k+1)`-st
is not; its term cap is `D_g`.  `[kernel]` as a theorem abstract in the machine
(`AnchorChain.hop_zero`, `hop_iter`, `hop_one`); `[exact: equal to the walk at every column on
full periods {5,7} .. {5..19}]`.  Cost: `prod (1 + D_g)` terms as a flat/nested form (3, 6, 18,
54, 162, 648, 1944 to `{5..29}` -- exponential in layers, no cross-layer cancellation found),
against a lazy cost of `1 + (crossed columns not on gear 5's teeth)`, mean 1.37..2.36 at `{5,7}`
.. `{5..19}`; the scan form is quadratic but needs `F+1` as its term bound, which is the unknown.

### 3.2 The attainment identity: a legal word is always aligned somewhere

**Theorem (R68).**  If consecutive openings `x_0 < ... < x_J` of `M` have a legal middle-gap word
then `x_J - x_0 <= F(M + q')`.  Proof: legality gives a tooth assignment `t_1..t_{J-1}` with
`x_{i+1} - x_i = (t_{i+1} - t_i) c (mod q')`; fix one, set `r = t_1 c - x_1`; the joint period is
`P(M) q'` with `gcd(P(M), q') = 1`, so some translate `x + jP(M)` with `jP(M) = r (mod q')` is a
run of `M` with the same gaps in which `q'` strikes EVERY interior.  With the converse
(`Q*_J <= F_J` and the Kleene identity):

**`max( F_2(M), max_{J >= 3} Q*_J(M ; legal for q') ) = F(M + q')`, exactly.**

`[exact: proved both ways (r22 Kleene identity; r26 standalone CRT proof); computed exactly at
eight steps m11..m37 (`qstar.py`), two out-of-scan confirmations `Q*_max(43;47) = 118 = F(47)` and
`Q*_max(47;53) = 145 = F(53)`, and the same vehicle then computed `F(59) = 161` on machine 23's
period, a cost ratio of 5.3e11]`; and `[exact: 27,570 tooth-counterfactual machines, zero
exceptions]` -- see section 5.

The negative half travels with it and must be repeated wherever the identity is used: because
`Q*_max` EQUALS `F(M+q')`, "the word-legal criterion certifies the budget inequality" is the SAME
statement as "the budget inequality holds".  **There is no slack in it to exploit.**  Its whole
value is that it is computed on the old machine.

### 3.3 The exact record algorithm, and the word-indexed identity

Two equivalent operational forms of the same identity, both on record:

- **The exact record algorithm.**  `F(M+q') = max over k >= 1, over all k-sites, of
  `o[i+k] - o[i-1]``, where a `k`-site is `k` consecutive old openings whose spacing word is a
  legal killed word of `q'`.  Because every site fires exactly once per new period, **residues
  drop out of the record question entirely** -- no new-period scan, no residue bookkeeping;
  `k = 1` reproduces `F_2` identically.  `[exact: six steps, 18, 25, 34, 43, 58, 88]`
- **The word-indexed identity.**  With `W(q')` the alternating words in `{a,b}` of length
  `<= L-1` (L = litcap(q' mod 210)) plus the padded words, and `w` COMPATIBLE if some tooth
  residue `r in {c, q'-c}` has all partial sums `r + prefix(w)` again in `{c, q'-c}`:
  `F(M+q') = max( F_2(M), max over compatible w of [ span(w) + FS_max(w; M) ] )`, with
  `FS_max(w;M) = max over occurrences of w of (gap before + gap after)`.  `[exact: 6/6 steps;
  binding words (4), (6), (13), (8,15), (10), (10)]`  The word list and compatibility depend on
  `q' mod 210` ALONE; only occurrences and flanks come from `M`.  `FS_max` is the sole open input.

The transfer to any even gap is verbatim: with the gear's two teeth at `n = 0` and `n = -e`,
`g = 0 mod q'` is a padded link, `g = +-e mod q'` a literal link, anything else illegal, non-zero
letters alternate, and `F(M+q') = max over legal runs of span`.  "This is the same law with `2u`
replaced by `e` -- the ONLY `d`-dependence in the law."  `[exact: 14 of 14 configurations
(d = 2,4,6,10,12,30; machines {3,5,7,11} .. {3..17}; q' = 13,17,19; all CRT phases), identity
exact, 0 soundness violations, 0 firing misses]`  And `tier_1 = F_2(M)` exactly in every row --
"the 1-letter word always fires" -- for every `d`.  A degenerate gear (`q' | e`) has ONE tooth,
the frame letter set collapses to `3q'`, and `F(M+q') = F_2(M)` exactly.

Compatibility and the corridor never interact: `gcd(35, q') = 1`, so the tooth condition is
CRT-independent of the mod-35 carrier.  `[exact: holds because q' > 7]`

### 3.4 The deletion ladder

`F_j(M) <= F(M + the next j-1 primes)`.  Proof, three lines: take the stretch realising `F_j(M)`;
it has `j-1` interior openings; `P(M)` is invertible mod each of the next `j-1` primes, so CRT
gives a translate in which interior `i` is congruent to gear `q'_i`'s own tooth, for every `i` at
once -- every interior dies.  It is the `r`-gear generalisation of `F(M+q') >= F_2(M)`: `r` new
gears buy `r` rungs of the `F_j` ladder, one designated strike each, because the `r` phases are
independent.  `[exact: proved; asserted at all 32 (M,j) pairs where both sides are known exactly,
one equality (`F_2(17) = 25 = F(19)`), tightest non-equality `F_2(37) = 90` against
`F(41) = 91`]`

Free caps past the scan wall: `F_2(41) <= F(43) = 103`, `F_2(43) <= F(47) = 118`,
`F_3(43) <= F(53) = 145`, `F_4(43) <= F(59) = 161`, `F_2(53) <= F(59) = 161`.

**It is logically circular as an induction step** -- it prices `F_2(M)` by the very `F` the rung is
certifying -- and its slack thins along the ladder: `F(M+q') - F_2(M)` is 3 at 29->31, 1 at 37->41,
0 at 41->43.

### 3.5 The lap-phase transfer (a distant machine's alignments from a small one)

`k |-> (k mod P, (k mod q_1, ..., k mod q_r))` is a bijection, and a maximal run of `M'`-openings
is exactly a pair (run of consecutive `M`-openings, phase tuple) such that the endpoints and the
chosen survivors avoid every new gear's teeth and every other interior `M`-opening is struck by at
least one new gear.  So `Q_J(M')` and `F_J(M')` are computable exactly on `M`'s period, at
`1/(q_1...q_r)` of the cost.  `[exact: proved (CRT) + two-sided anchors -- the m31 ladder
`68/85/90/91/90/88` reproduced entrywise, `F_2(37) = 90` and `F_3(37) = 97` from three gears
below; every witness CRT'd to the target machine and re-verified slot by slot]`  This is the
vehicle behind `F(59) = 161`, `F_2(53) = 159`, `F_4(41) = 118`, `F_6(47) = 177`.

Two riders: a CERTIFICATION is conditional on the span cap, a FAILURE is not; and a soundness trap
is on record -- with `r >= 2` the survivor-count lower bound is not monotone, so the walk must stop
on its RUNNING MAXIMUM.
### 3.6 The word reduction: alignment depth IS a word-length question

**R89.**  `Q*_J(M; q') > -inf` **iff** `L(M) >= J - 2`.  Hence `J_max(M) = L(M) + 2` and
`A_kill(M -> q') = L(M) + 1`.  Forward half: the `J-2` middles of a word-legal `J`-run are a
realised legal word.  Converse: an occurrence of a realised legal word plus its two flanking gaps
IS a word-legal `J`-run, because legality constrains only the middles.  `[kernel]` over an
abstract opening enumeration (`WordLegal.chain_iff_word`, `word_of_window`, `window_of_word`,
`qstar_iff_word`, `jmax`, `akill`), with one named hypothesis (periodicity of the gap residues),
instantiated at m11/m13/m17 (`L = 1` at each); `[exact: 16/16 against the recorded `J_max` and
`A_kill` rows]`.

Measured corpus row: `L = 1,1,1,2,1,3,3,2,2,2,4,3` at m11..m53.  `[measured: m11..m53]`

Every EMPTY cell of the per-`J` table becomes a one-line dictionary fact.  The reduction moves the
open question, it does not close it: `L(M)` bounded is still open.

**`D_g = A_kill(M -> g)` -- two constructs, one object.**  The anchor line's chain depth equals the
twin route's kill arity at every gear where both exist (`D_17 = D_19 = 2`, `D_23 = 3`, `D_29 = 2`,
`D_31 = 4`, `D_37 = 4`, `D_41 = 3`): both count co-deletable runs of consecutive `M`-openings, and
word legality ("prefix-sum range `<= 1`") IS "all in one two-class set".  With R89,
`D_g = A_kill = L + 1`.  `[exact: identity by argument; 7 for 7 by two vehicles built four rounds
apart in different languages]`  A streamed partial pass gives `D_g >= v`, a decided arity level
gives `D_g <= A_kill`, and the two halves meet -- `D_41 = 3` is exact from 0.1% coverage.
`D_g` bounded is OPEN.

**Same-tooth lemma (R90).**  A padded middle leaves the tooth fixed, a literal middle flips it, so
the middle span `x_{J-1} - x_1` is `= 0 mod q'` exactly when the number of non-padded middles is
even, and `+-2c mod q'` otherwise.  A LITERAL even-`J` chain therefore starts and ends ON THE SAME
TOOTH, and its middle span is `>= ((J-2)/2) q'`.  `[kernel]` (`WordLegal.same_tooth`,
`same_tooth_window`, `literal_even_span`, hypothesis `2c != 0` discharged from `6c = 1`);
`[exact: 38 realised legal words, 0 violations]`.  Literal chains only -- the two padded even-`J`
maximisers `(12,37)` at m31 and `(41,14)` at m37 have middle sums `12 mod 37` and `14 mod 41`.

### 3.7 The per-`J` layer: what each depth can contribute

- **`F_j` (the spectrum).**  Full-period spectra on record: machine 13 `11 16 23 26 28 31`;
  17 `18 25 28 33 35 40`; 19 `25 31 35 38 47 50`; 23 `34 39 50 58 65 77`; 29 `43 55 65 70 85 90`;
  31 `58 68 85 90 92 97`; 41 (prefix, lower bounds, indexed j = 3..8, NOT j = 1..6: F(41) = 91,
  F_2(41) = 103) `110 112 118 123 130 138`.
  `[exact: full period m13..m31; m37/m41 rows are prefix LOWER bounds]`
- **`Q_j` (the word-free / qualifying spectrum).**  `Q_j(M; a)` = max sum of `j` consecutive gaps
  whose `j-2` middles are all `>= a = 2u'`.  Every qualifying merged stretch is such a sum, so
  the budget inequality follows from `Q_{ell+2} <= F(M) + q'`.  Margins `F + q' - max_j Q_j`:
  `+4, +10, +9, +10, +13, +3` at 11->13 .. 29->31.  `[exact: full period m11..m31]`  The interior
  floor `2u'` is a THEOREM, and at 29->31 it alone does the work: `F_5 = 85 = F + 42` fails while
  `Q_5 = 71 = F + 28` passes.  Honest counterweight recorded in the same entry: the word-free
  margin collapses from `~0.45q'` to `0.10-0.11q'` at machines 29 and 31, while the
  word-restricted margin does not (0.52q' at 29->31).  And the ALL-DEPTHS hypothesis-free form
  FAILS from 43->47 on (152 > 150; 177 > 171) with machine-verified witnesses -- which kills that
  form, not the budget inequality.
- **`Q*_J` (the word-legal spectrum).**  The same object with the sharp predicate -- each middle
  in `V = {0, +s, -s} mod q'` AND the letter word T3-alternating -- instead of the size shadow
  `>= 2u'`.  `Q*_J <= Q_J` pointwise and `max_J Q*_J = F(M+q')` exactly.  It certifies at every
  step the plain criterion loses: 43->47 at `<= 149 <= 150`, 47->53 at `<= 170 <= 171`, at every
  depth `J = 2..7`, consuming NO arity hypothesis.  The failing 47->53 stretch's middles
  `[22,28,30,67]` all clear the floor `a = 18` but not one is congruent mod 53 to a legal letter,
  so it can never be merged: **the plain criterion was failing on a relaxation the merge law never
  needed.**  `[exact: two-sided anchors 88 = F(37) at 31->37 and 58 = F(31) at 29->31]`
- **The spectrum-plus-depth certificate.**  `F(M+q') <= max_{2 <= J <= J_max} F_J(M)` with
  `J_max = A_kill + 1`, from attainment plus `Q*_J <= F_J` plus "emptiness is upward closed".  The
  margin at a step is exactly `F(M) + q' - F_{A_kill+1}(M)`.  `[exact: proved; table verified at
  ten steps]`  Rung nine: `F(43) <= max(103,117,118) = 118 < 134` (margin +16).  Rung ten:
  `F(47) <= max(116,125,132) = 132 < 150` (margin +18).  **Every `A_kill <= 3` step certifies
  (+10 to +24); both failures (29->31 by -11, 47->53 by -6) and the single +3 squeaker are the
  `A_kill >= 4` steps.**  Mechanism: one extra unit of `A_kill` admits one more level of the `F`
  ladder, costing 7-16 units, while the budget gains only `q' - q'_prev` (4 to 6).
  Circularity recorded (Constructor r29 self-correction): below m59 the `F_J(M)` values are
  exhaustive only because of deletion-ladder caps taken from `F` at machines ABOVE the step, and at
  `j = 2` that cap is the very quantity the rung bounds -- so those rungs are method
  demonstrations, not independent bounds.
- **The peel bound (Theorem D).**  Deleting either flank of a word-legal `J`-run leaves a
  word-legal `(J-1)`-run, so `Q*_J <= Q*_{J-1} + min(g_L, g_R)` at the argmax; equivalently
  `g_L + w + g_R <= F_2(M) + min(g_L, g_R)` with NO hypothesis.  `[exact: proved,
  hypothesis-free]`  Read backwards it forces a violating depth-3 run to have MIN FLANK `> s_min`
  (asserted at all 27,570 counterfactual machines).  It discharges the triple inequality at any
  triple whose smaller flank is `<= s_min` -- 6 of 8 steps outright; the residue is triples with
  both flanks above `s_min`, measured 0, 0, 16, 4, 24, 131, 205, 317 at m11..m37.  It does NOT
  reach `J >= 4`: the free reduction gives only `2F_2 - q'`, short by exactly `F_2 - a` (R55's
  2F wall).
- **The middle-sum lemma (Theorem A).**  In a literal word-legal `J`-run the `J-2` middles
  alternate between classes `a` and `b`, so with `k = floor((J-2)/2)` the middle sum is `>= k q'`
  (`J` even) or `>= k q' + a` (`J` odd).  Hence `Phi_J <= F_2(M) + s_min(q') - m_min(J)`: at
  `J = 5` the two flanks may sum to at most `F_2 - q'`, at `J = 6` to at most `F_2 + a - 2q'`.
  **The flank envelope collapses at rate `q'` per two levels -- which is why the deep layers are
  the cheap ones.**  `[CORRECTION 2026-09-05, docs/proofs/16: the envelope form silently assumes the MEASURED
  Delta_J <= s_min; what T1-T3 prove is Phi_J <= Q*_J - m_min(J) <= F(M+q') - m_min(J)]`; measured
  `Phi_J <= F_2 - b` at every non-empty literal even-`J` cell, margins +5 (m19), +10 (m29), +9 (m31).
  Literal middles only.
- **`Delta_J` and par trading.**  `Delta_J = Q*_J - F_2(M)`: every LITERAL cell lies in `[-3, +4]`
  at m11..m41, and the excess SHRINKS with depth (`Delta_5 = 0` exactly at both machines where
  `J = 5` exists); confirmed out of sample at machine 53 (every `Delta_J` is `+2`).
  `[measured: 13 cells, exact per cell, three independent vehicles; NOT proved; at m41 two cells
  are bounds (`Q*_3 <= 116`, `Q*_4 <= 100`)]`  The residual `eps(v) = Phi(u) - Phi(v) - x` for a
  realised legal word `v = u.x` satisfies `Delta_J = Delta_{J-1} - eps` with `Delta_2 = 0`, so
  "`Delta_J = O(1)` uniformly in `J`" is EXACTLY "`eps` is `O(1)` per letter AND `L(M)` is
  bounded".  Decomposition lemma: `eps(v) = d - g_out` with `d = Phi(u) - x - g_kept >= 0`, so
  `eps = O(1)` is a CANCELLATION, not a smallness (`d = 27`, `g_out = 28` at m31).
  `[exact: identity proved, decomposition asserted 30/30]`; `|eps| <= s_min` is
  `[measured: 14/14 literal cells]` and **REFUTED at 10 of 16 padded cells**; `max |eps| = 4` along
  maximising chains over 12 cells against `s_min` running 4..14.
- **The palindrome dichotomy.**  At every measured cell the maximising word is unique up to
  reversal.  At `J = 3` (11 cells) and `J = 4` (4 cells) it is a reversal PAIR and NEVER a
  palindrome (Theorem B forbids a literal even-`J` palindrome outright); at `J = 5` (2 cells) it is
  unique and SELF-REVERSE: `(7,10,21,10,7)` at m29 and `(3,25,12,25,3)` at m31, each with
  `Delta_5 = 0` exactly.  `[measured: exhaustive per cell]`
- **The `F_3` wall.**  `Phi(q') + q' <= F_3(M)` trivially.  At m31 it is EQUALITY: the
  `F_3(31) = 85` maximisers are `(18,37,30)`/`(30,37,18)` -- the old machine's depth-3 record has
  the padded letter as its middle.  At every other machine m11..m37 the `F_3` maximiser's middle is
  not a legal letter of any class.  The excess `F_3 - (F_2 + s_min)` is `+1,+1,-3,-4,+1,0,+5,-7` at
  m11..m37: four machines exceed the increment budget at depth 3, and only m31's exceeding run is
  word-legal.  `[exact: script-verified and gated, f3_middles_r30.py]`  The recurrence is a residue
  event with base rate `3/q'`, so it WILL recur; it is labelled arithmetic luck per step, never a
  law.  Prediction on record: `F_3(37)`'s `(37,23,37)`, `F_3(43)`'s `(67,28,30)` and `F_3(47)`'s
  `(28,33,84)` have non-legal middles.  Rider: `Phi(12,37) = 39` and `Phi(37) = 48` each rest on
  ONE occurrence (a mirror pair), and with that one stretch removed par trading holds at the padded
  letter too (`eps = +3`).
### 3.8 Bounds on the alignment depth `L`

- **The spectrum bound on `L`** (the theorem that retired requirement (B) as posed).  With
  `G = F(M+q')`, `T = floor((G-2)/q')` and `p` padded letters:
  `(SIMPLE) L(M) <= 2T + 1`; letter-aware `L <= 2T + 1 - p`;
  `(PARITY) L(M) <= max(2T, 2 floor((G - 2 - a_min)/q') + 1)`.  I.e. **`L(M) <= 2 F(M+q')/q' + 1`.**
  Proof: class minima are `a`, `b`, `q'`; T3 makes two consecutive nonzero letters sum to `>= q'`;
  attainment gives `span <= G - 2` (an `m`-letter word occupies `m+1` openings plus one gap each
  side).  `[exact: PROVED, unconditional given R68 and T3; verified at 12 corpus machines
  (173 gates) and 165,584 counterfactual machines with ZERO violations, including the family's
  `L = 5` member where (PARITY) equals 5 exactly]`  Corpus row (PARITY) `1,1,2,3,3,3,5,4,5,5,5,5`
  against `L = 1,1,1,2,1,3,3,2,2,2,4,3`; TIGHT at m11, m13, m29.
  The consequence is that `L` is `O(F/q')`, not `O(1)`, and `F/q'` is measured
  `0.54 .. 2.64` and growing.  Substituted into the chain it gives
  `G <= (q'(F_2 + c_A) - 4 c_A)/(q' - 2 c_A)` for `q' > 2c_A`, and the budget inequality whenever
  `8F <= q'^2 - (F_2 - F + 12) q' + 16` -- true at 8 of 13 corpus steps, failing only at the five
  small ones.  `c_A = 4` is a LITERAL-letter constant, so that closure is conditional on the open
  padded case.
- **The bare-word uniform cap** -- the first bound on part of `L` that does not grow.  A BARE word
  (every letter `a` or `b`) is forced by T3 to be one of the two alternations, and a realised
  word's prefix-sum offsets are OPENINGS of `M`, so they fit inside the exposed set of every gear,
  in particular inside the corridor `E_35`.  Hence `L_bare(M) <= PSORD(q' mod 210) <= 5` at EVERY
  machine, uniformly, with `PSORD = 1` on 24 classes, 2 on 4, 3 on 14, **never 4**, and 5 on
  exactly `{37,53,83,127,157,173}`; `S = {PSORD <= 2}` has 28 of the 48 classes (density 7/12).
  `[kernel]` (`BareAlt.bareAlt_inadmissible_iff`, `no_gapWord`, `no_bare_run`, `no_bare_run_ge`,
  `S_card = 28`, `psord_le_five`, `psord_ne_four`, `psord_eq_one_iff/_two_iff/_five_iff`,
  `fitsB_of_open`, `open_of_gapWord`, `inadmissible_iff_capC`), instantiated at m23/m37/m41/m43;
  three vehicles sharing no code agree element for element.  Also kernel-checked:
  `c in S <-> LiteralCapTable.capC c <= 3`, so the bare cap and the literal cap are ONE object at
  every class.  At m41 and m43 every bare decision is FREE.
- **The literal cap.**  A literal chain (built purely from the gear's own two teeth, every member
  in the corridor `E mod 35`) never has more than 6 members, for every gear forever, and the exact
  cap is a function of `q' mod 210`: 2 on 24 classes, 3 on 4, 4 on 14, 6 on exactly
  `{37, 53, 83, 127, 157, 173}`; there is no class of cap 5.  `[kernel]` both ways
  (`LiteralCap.no_run_seven`, `s_eq`, `literal_chain_le_six`, `cap_six_classes_sharp`;
  `LiteralCapTable.cap_table_maximal`, `cap_table_realized`, `no_cap_five`, `cap_spectrum_counts`),
  verified against every prime `<= 5000`, 0 mismatches.  The mechanism is that a literal chain is
  an interleaved two-phase walk of period 70 mod 35.  Scope: LITERAL chains only -- padded runs
  escape it, "killed runs are bounded by 6" is FALSE, and it is NOT a density fact (over all 1,225
  `(t,s)` pairs mod 35 the run spectrum reaches 140, so the restriction to invertible classes does
  real work).  It also never predicts realised arity: litcap 4 at m41 where the literal 2-word
  count is exactly 0; litcap 2 at m37 where `A_kill = 3` (forced padded).
- **The Polignac cap.**  For gap `d = 2e` the literal-chain cap depends only on `gcd(e,105)`:
  6 for six of the eight classes, 10 for `gcd = 15`, 12 for `gcd = 105`.  **12 is the absolute
  ceiling over all Polignac configurations, for every gear, forever.**  `[kernel]`
  (`PolignacCap.cap_gcd_*`, `capOf_le_twelve`, no axioms at all; each cap checked sharp).  Modelling
  trap the kernel caught: gear 3 FILTERS the candidate list rather than breaking a run (the wrong
  model gives caps 2/4 instead of 6/10/12).  Companion: `mod-105 = mod-210` for odd `q'`, so one
  48-class check covers every gap.
- **The decomposition `L = max(L_bare, L_pad)`.**  With `L_pad` the longest realised legal word
  using at least one non-bare letter, requirement (B) is EXACTLY "`L_pad` bounded".  Measured
  `L_pad = 0,0,0,1,1,1,2,2,2,2,3,3` at m11..m53 -- it takes every value 0..3 and it GROWS.
  `[measured: m11..m53; L_pad(47) = 3 measured ((18,35,53), (18,53,35), (35,18,53) realised;
  (35,71,35) undecided at 6e7 nodes); L_pad(53) = 3 DERIVED from the bare-word theorem plus the
  recorded L(53) = 3, at a machine no census reaches]`  The four machines where `L > L_bare`
  (m37, m41, m43, m53) are exactly the `S`-machines whose record is carried by a word containing
  the padded letter `q'`.  **Nothing on record bounds `L_pad`.**
- **EXPCAP (the exposure half's own cap).**  A word of length `m` survives phase saturation at `M`
  iff it survives at the sub-machine `{g in M : g <= 2m+2}`.  `EXPCAP(M) = max{m : S_m > 0}`, row
  `1,1,1,4,2,3,5,18,13,10,5,21` at m11..m53 against `L = 1,1,1,2,1,3,3,2,2,2,4,3`.
  `[exact: proved; asserted numerically at every (M,m) cell m11..m53]`  `EXPCAP - L` is NOT bounded
  (16, 11, 8, 18 at m37/m41/m43/m53): the exposure half over-caps by an arithmetic-selected amount.
- **The killer profile: which half actually kills an extension.**  For every realised longest legal
  word and every T3-legal one-letter extension, the kill is attributed either to the SAT screen or
  to `y*`, the smallest gear prefix whose OPEN constraints make the CSP infeasible.  The profile is
  BIMODAL and EMPTY IN THE MIDDLE: every decided kill is either **cover-only (`y* = 0`) -- no
  column of `M` blocks the punctured interior, the pattern does not occur in `M` at all** -- or a
  CORRIDOR kill (`y* = 7` for the pure alternations `(10,19)` at m23 and `(12,25,12,25)` at m31,
  whose blocked patterns DO occur, gate-verified by a period scan; `y* = 5` from m37 on).  **No
  extension at any machine was attributed to the open constraint of a gear above 7.**
  `[measured: exact CRT decisions at m19..m41; 0 realised, 0 undecided at the full machine; at
  19->23 all 4 classes die at `y* = 0`; 23->29 three at 0 and one at 7; 29->31 both at 0;
  31->37 three at 0 and one at 7; 37->41 eight at 0 and five at gear 5; 41->43 nine at gear 5]`
  Where this points: the corridor `{5,7}` bounds exactly the pure-alternation family, and every
  other extension is refuted by the COVER half in its purest form -- an `F_J`-type statement about
  `M`'s blocked runs, not a teeth statement about `q'`.  Not delivered: m43 and m47 profiles; 2 at
  m37 and 10 at m41 are refuted but unattributed.
- **`L` as a density statistic, with a one-unit arithmetic collapse.**  With the REAL class
  densities of the legal alphabet in `M`'s exact gap histogram and the T3 alternation transfer
  matrix (growth rate `p0 + sqrt(p+ p-)`), an independent-letter model predicts the longest legal
  run to WITHIN ONE UNIT at every scanned machine (3.7/3 at m29, 4.0/3 at m31, 4.0/2 at m37) -- but
  the COUNT of legal runs collapses at the top (4 against 279 at m29, 216 against 1,610 and 0
  against 2.5 at m31, 27 against 10,500 already at length 2 at m37).  The free screens (alphabet +
  spectrum + phase saturation) are exactly ONE LENGTH too generous at 7 of 8 next-prime cells.
  `[measured: exact censuses and exact CRT decisions]`  Rider: the next prime is USUALLY but not
  always the maximising gear (m23: `L_31 = 2 > L_29 = 1`; m37: `L_53 = 3 > L_41 = 2`); what sets
  `L_g` is the alphabet size, which `q'` usually maximises.
- **The exact null for `L`.**  Exact finite-automaton expectations of the longest legal run over
  `N = prod(q-2)` gaps.  The class probability is the largest single suppression and it is NOT
  `3/q'`: under the machine's own gap distribution it is 0.19-0.39 of `3/q'` at m11..m37 (0.105 at
  m37), a factor 2.5-9.5 below equidistribution, because gap values pile up below the smallest
  legal letter and `p0 = 0` at seven of eight machines.  Alternation costs a uniform 12-14%;
  dependence between consecutive gaps is a factor 0.43-1.00, NOT monotone in the machine.
  `[exact: two independent routes agreeing to three decimals; m41..m47 rows labelled PROXY]`
  So `I-actA / L` is 1.14, 1.31, 1.96 at m29, m31, m37 against 4.3-5.2 for the equidistributed
  proxy: the "18 against 4" gap at m47 is mostly the estimate, not the machine.  At m13/m17/m23 the
  ORDER-1 Markov null already gives `E[L] = 1.00` exactly; at m19 the order-1 null moves only 3% of
  the way, so the dependence that limits `L` there is beyond lag 1.

### 3.9 Automata: the whole depth family in one algebra

- **The Kleene identity.**  On states `(opening index i, tooth s)`, with
  `K[(i,s),(i+1,s')] = d_i` when `d_i mod q' in {0,a,b}` and `(s -> s')` is the T3 transition, and
  flanks `L(i) = d_{i-1}`, `R(i,s) = d_i`: `F(M+q') = L (x) K* (x) R` in max-plus.  `K` is
  nilpotent of index `k_max`, so `K*` is a finite sum, but **the identity names no truncation
  depth**; its `m`-th layer is exactly `Q*_{m+2}`.  `[exact: verified at m11..m29; the identity
  itself is not kernel-checked]`
- **The potential (depth-quantifier-free).**  The budget inequality holds IFF a potential `h`
  exists with three ONE-STEP, ONE-OPENING inequalities: `(C1) h(i,s) >= d_i`;
  `(C2) h(i,s) >= d_i + h(i+1,s')` for every legal qualifying transition;
  `(C3) d_{i-1} + h(i,s) <= F(M) + q'`.  Necessity `h = K* (x) R`; sufficiency because any
  super-solution dominates the star.  `[kernel]` for the certificate direction
  (`Potential.IsPotential`, `chain_le_potential`, `D_of_potential`, `merged_le_of_potential`;
  `Potential19`, `PotentialLadder`), with `h11, h13, h17, h19` exhibited and tail depths NOT
  growing with the machine (4, 3, 5, 4); `(C2)` holds with EQUALITY in every branch at every
  machine and its deepest branch is always that machine's own `no_big_run`.  The CONVERSE is not
  formalised, and a potential valid at every machine at once is not known: the generator is
  arity-free but NOT machine-free.
- **The survivor generator.**  A run of two consecutive NEW gaps is a run of old openings all
  struck at one phase EXCEPT ONE SURVIVOR, and the spacing straddling the survivor is
  `d_i + d_{i+1}`; the survivor lives iff `cls(d_i)` is ILLEGAL out of the current tooth.  Adding
  that one SKIP transition `SIGMA`: `F_2(M+q') = L (x) K* (x) SIGMA (x) K* (x) R`, and generally
  `F_j(M+q') = L (x) K* (x) (SIGMA (x) K*)^{j-1} (x) R`.  Between two aligned runs there is exactly
  ONE surviving opening.  `[exact: proved for every j; verified exact, full period, seam-stitched
  at every scannable step (`F_2(M+q') = 16, 25, 31, 39, 55, 68, 90`), against an independent pair
  census and against CRT+SAT at 31->37]`; `[kernel]` at 11->13 (`Gen11.gen_zero = 11`,
  `gen_one = 16`, `Gen11Sound.generator_sound` giving `F_1..F_4(13) <= 11,16,23,26` from machine
  11's 135-letter word with machine 13's period nowhere in the derivation, independence gated by
  `DepAudit.lean`).  So the two-gap statement at `M+q'` DESCENDS -- it is layer 0 of the same
  algebra one gear down -- rather than being an extra hypothesis.  Cost: the survivor system needs
  ONE MORE order of history than the plain system (`A_4` exact 7/7 plain, `A_5` on the survivor
  side).
### 3.10 The increment law -- kernel-checked only at literal steps

**Statement.**  `F(M + q') - F_2(M) <= s_min(q') = min(2u', q' - 2u') = 2u'`.  Reading, and the
reading is a labelled hypothesis: `2u'` is the smallest positive legal letter, so one more aligned
link buys at most one small letter over the old two-gap maximum -- unless the link is padded, when
it is worth a full `q'`.

**Status, stated exactly as the record has it.**  `[kernel]` at all six LITERAL steps, both halves,
hypothesis-free (`Increment.increment_law_literal_steps`, 1,749 jobs; the lower halves are CRT
columns of the real machine, e.g. `F_2(29) >= 55` from the single column 858386140 in 35 s;
sharpness also kernel-checked -- `f2_19_sharp`, `f2_23_sharp`, `f2_29_sharp`).  It is NOT
kernel-checked as a general statement, and it is **not called a law here beyond those steps.**
`[measured: 11 of 12 testable corpus steps, failing only at the padded 31->37 by +8]`; confirmed
OUT OF SAMPLE at 53->59 (predicted `F(59) <= 179`, measured `<= 178`, now `= 161`).
Differences against caps: `0,2,0,3,4,3,20,1,0,...,+2` at 53->59 against
`4,6,6,8,10,10,12,14,14,16,18,20`.

The base cases are kernel facts; **the induction step is not**, and the LP vehicle cannot supply it
(cost is a primorial in the number of held gears).  The quantity that decides certifiability is
`W_inc - F(q')`, negative at EXACTLY ONE corpus step -- the padded 31->37, where the increment
width asks for something FALSE since `F(37) = 88 > 80`.

It is also **not generic**: 13-22% of the tooth-counterfactual family violate it (0-6.5% once the
incoming gear's tooth is pinned to `round(q'/6)`), so no proof from "same gears, same density,
symmetric teeth" can exist.  See section 5.

Two companions:

- **The triple inequality and `Delta_3`.**  Both `(g_L + w)` and `(w + g_R)` are 2-runs of `M`, so
  with NO hypothesis `g_L + w + g_R <= F_2(M) + min(g_L, g_R)` -- automatic whenever the smaller
  flank is `<= s_min`, which discharges 6 of the 8 steps outright.  `Delta_3 = -3, 2, 0, 2, 4, 3,
  2, 0` -- **bounded by a constant while `s_min` grows linearly**, so the shape to aim at is
  `Delta_3 = O(1)`, not `Delta_3 <= s_min`.  `[exact: the reduction proved; the table verified,
  literal middles separated from padded]`  The literal triple inequality holds at EVERY step
  including 31->37 (70 <= 80); the padded half is the tight one (+1 against the literal half's +10
  at 41->43).
- **The anchor line's own increment row.**  The interior gaps of the record chain are exactly
  `+-2u' mod q'`, NEVER a multiple of `q'` (kills alternate teeth at the minimum stride), and since
  `3 x 2u' = 1 mod q'`, `s_min = (q' +- 1)/3`, so a chain of `m` kills spends at least
  `(m-1)(q'-1)/3` columns on its interior.  Measured over the full period at all eight rungs:
  `F' - F_2 = 1,0,0,2,0,3,4` against `s_min = 2,4,4,6,6,8,10` -- the increment over every kill
  chain, not only the record (rung 23: 733,670 one-kill chains max 30, 11,746 two-kill max 32, 62
  three-kill max 33; rung 29: 15.4M one-kill max 38, 243,822 two-kill max 42, no three-kill chain).
  `[exact: full period, eight rungs]`  Note the indexing: this row is indexed by the INCOMING gear
  (so the last entry is the machine `{5..23}`), while the archive's `F_2 - F = 2,4,5,7,6,5,12` row
  is indexed by the machine itself (so its last entry is m29).  The two are consistent.

### 3.11 What a record stretch is made of

- **The lower side is forced.**  `F(M+q') >= F_2(M)` with no computation: the middle opening of any
  two consecutive old gaps dies in exactly 2 of its `q'` lifts.  The best one-kill run equals `F_2`
  at every rung.  `[exact: every rung {5}+7 .. {5..23}+29]`  The upper side `F' - F_2` is the whole
  content.
- **What the new gear must do.**  The record of `M + q'` is an old stretch `x_0 < ... < x_J` with
  every interior struck by `q'` and both END openings surviving; kills `= J-1`; consecutive kills
  differ by `0` (same tooth) or `+-2u'` (opposite teeth) mod `q'`.  Decompositions on record:
  `{5,7,11}+13`: `7 -> 11`, old gaps `[6,5]`, 1 kill; `{..13}+17`: `11 -> 18`, `[5,11,2]`, 2 kills,
  residues `[3,14]`; `{..17}+19`: `18 -> 25`, `[7,18]`, 1 kill; `{..19}+23`: `25 -> 34`,
  `[4,8,15,7]`, 3 kills; `53->59`: `145 -> 161`, `[10,118,33]`, 2 kills, same tooth
  (`118 = 2 x 59`).  `[exact: every computable rung]`  **Parity is not a constraint** -- old gaps
  are mixed parity and `q'` is odd, so any gap parity has a tooth arrangement; the obstacle is
  arithmetic mod `q'`, never parity.
- **The shape of the extremal stretch is the wrong shape to qualify.**  Of the 132 stretches
  attaining `F_j` at m19/m23/m29 (full-period census), ZERO are literal and ZERO are qualifying:
  the attaining shape is always two near-maximal flanks with the machine's SMALLEST gaps interior
  (2,3,4,5,7) -- which the interior floor `>= 2u'` forbids exactly.  Exhibited with addresses, e.g.
  machine 29 `F_5 = 85`, flanks (30,18), interior (4,3,30), at `k = 772,741,833`.
  `[exact: 132 maximisers, full period]`
- **Record gaps are isolated.**  The neighbours of every record gap are small: `(1,2)` at `{5,7}`;
  `(1,3)` at `{5..11}`; `(2,2),(2,5)` at `{5..13}`; up to 7 at `{5..17}`; `<= 5` at `{5..19}`;
  `<= 7` next to any gap `>= 0.8F` at `{5..23}`; `<= 7` at `{5..29}`.  `F_2 - F = 2,2,4,5,7,6,5`
  against `q' - s_min = 5,7,9,11,13,15,19`.  `[exact: full period, seven machines]`  Also
  `F_2(47) = 134` is attained at `[54,80]`, containing NEITHER a maximal gap.  **No explanation is
  on record for why a record-size gap of a CRT word has only small neighbours** -- it is named as
  "the part with no teeth in it at all".  And the isolation does NOT explain `F_2`: at m29
  `F_2 = 55` comes from `(30,25)`, two large-but-not-maximal gaps.
- **Ordinary at the bottom, made at the top.**  Survivors of a record stretch under `{5..g}` track
  a random stretch of the same length at the bottom layers (`S_5/mean = 0.96-1.0`,
  `S_7/mean = 0.92` at `{5..23}`) and the record is MADE at the top three or four layers, where
  each gear removes 2-3 survivors a random stretch would keep.  Example `{5..23}`, record 33:
  survivors 19, 13, 10, 7, 5, 3, 0 against random-stretch means 19.8, 14.2, 11.6, 9.8, 8.7, 7.7,
  7.0.  The three longest gaps of every machine share the profile to within one survivor.
  `[exact: full period {5..11} .. {5..23}]`  The top layer carries 1 to 3 survivors, ALL on the top
  gear's teeth, with differences in the chain classes (`{5..17}`: 2 at 4, 15, teeth `+,-`,
  difference `11 = -d`; `{5..23}`: 3 at 3, 11, 26, teeth `-,+,-`, differences `8 = +d`,
  `15 = -d`).  One layer down the survivors are 3,3,3,3,5 and only some sit on that gear's teeth.
  What repeats from rung to rung is the SHAPE, not the pattern.
- **Record genealogy: records recruit runner-ups.**  For a record of `M + q'` at column `k`, the
  `M`-openings inside `(k, k+F)` are the deleted chain and the ancestor is the `(L+1)`-gap run they
  cut.  The ancestor is ALMOST NEVER the `F_J(M)` maximiser (1 of 8) -- it is a RUNNER-UP, by 2 to
  14 -- but its largest gap is itself a merged run one level down (7 of 8), and that continues for
  1-5 generations.  For the `F_J` records the largest gap is merged one machine down in 12 of 12.
  The m31 record's whole tree: `58 <- m29 [18,10,30] (phase 8, teeth '+-') <- 30 = m23 [7,23] <-
  23 = m19 [5,15,3] <- 15 = m17 [2,6,7] <- 6 = m13 [5,1]`, five generations, deficits 7, 9, 12, 13,
  10, 9.  `[exact: computed by residue arithmetic on the column, no scan; every record column
  re-verified at its own machine]`  The ancestor's RANK among `M`'s own `J`-runs by span is 8 to
  219, so `F(M+q')` cannot be computed from the top-`k` `J`-runs of `M`, nor from `M`'s spectrum
  records.
- **When the record is scan-free.**  By attainment it is `max over the realised legal words w
  (1-4 per machine) of max over the OCCURRENCES of w of (gap before + span + gap after)`.  For long
  words the occurrences are few (4 for `(10,21,10)` at m29, two mirror pairs) and each is a CRT
  solution enumerable scan-free; for short words (8e6 and 1.3e4 occurrences at m29) the flank order
  statistic `Phi(w)` is exactly what a scan supplies and no enumeration reaches.  **So `F(M+q')` is
  scan-free precisely when the record is carried at depth `L >= 3`** -- which it is at 31->37,
  37->41, 47->53, 53->59.  `[exact: consequence of R68 plus the counted census; named, not built]`
  Winners get shallower as machines grow: `k_win = 3` at 31->37 and 37->41, `k_win = 1` at m41.

### 3.12 The mirror, and the one exact symmetry

- **The involution.**  Column `k` is struck iff some gear divides `6k-1` or `6k+1`, which is
  invariant under `k -> -k`.  So the opening set is exactly closed under negation, `k = 0` is
  always an opening and (P odd) its ONLY fixed column; on indices the map is `o_t -> o_{N-t}`.
  `[kernel]` (`Mirror.mirror_gear`, instantiated at m11 and m29).
- **The full symmetry group is `Z/2`, exactly.**  The affine maps preserving the opening set are
  the `2^m` multiplications by `c = +-1 mod every gear` (`b = 0` forced), of which only
  `c = +-1 mod P` preserves ADJACENCY; dropping affineness, the only rotations and reflections of
  `Z_P` preserving the openings are the identity and the mirror.  Fixed-point counts:
  `#fix(sigma_S) = N / prod_{q in S}(q-2)`, so exactly ONE of the `2^n - 1` sign involutions has a
  single fixed point.  `[exact: proved + brute-forced over all 92,400 affine maps at m11 and all
  `2P` rotations and reflections at m11/m13]`  **Any parity lever from a symmetry of the opening
  set is worth EXACTLY ONE UNIT, a factor of two, never four; there is no mod-4 version to hope
  for.**  A finer parity must come from something that is not a symmetry of the opening set.
- **The self-mirror stretch, located.**  A depth-`j` run is self-mirror iff its endpoints sum to
  `0 (mod P)`, i.e. it is centred on column 0 (`j` even) or on the antipode (`j` odd).  `N` odd
  gives exactly ONE self-mirror run per depth, at index `t_j = -j/2 (mod N)`, with span `2 o_i` for
  `j = 2i` and `P - 2 o_{M-i}` for `j = 2i+1`, `M = (N-1)/2`.  Corollary `g_j* = j (mod 2)`:
  `W_j(g)` is EVEN for every `g` of the wrong parity with no computation -- half the entire
  spectrum, free.  `[exact: proved; verified against exact full-period `W_j` censuses at m11..m29
  for every `j <= 12`, the odd column exactly `{g_j*}`]`; the counting half `[kernel]`
  (`Mirror.even_card_involution`, `window_count_even`, `adjacent_equal_even`,
  `none_of_at_most_one`, `self_mirror_unique`), instantiated only at m11.
- **The self-mirror stretch is NEVER word-legal at depth `>= 3`.**  `J` odd: its central middle is
  the antipodal gap, of length 1, and 1 is a legal letter only if `3 = +-1 mod q'`.  `J` even
  `>= 4`: its two central middles are both `d_0`, and T3 forbids two equal nonzero classes while
  `0 < d_0 < q'` forbids both being padded.  `J = 2` is the one depth needing a hypothesis, and
  there it is exactly `d_0 != F`.  `[exact: proved; gated at m11..m23, J = 2..7, 185 assertions]`
  Consequence: the reversal map is FIXED-POINT-FREE on the word-legal family at every `J >= 3`, so
  every span count is EVEN with no exceptional class and no census -- and "at most one word-legal
  `J`-run exceeds the budget" therefore proves there are NONE.
- **The mirror on records, and in transfer coordinates.**  For a stretch at address `k` with span
  `s` and interior offsets `o_i`: `k' = (P - k - s) mod P` is an opening, its interior offsets are
  the reversed `s - o_i`, its flanks are the reversed flanks, residues map `r -> (P - r) mod q''`,
  and `k + k' + s = P`.  In transfer coordinates `(k, c_q) -> (P0 - k - s, (P0 - c_q) mod q)` with
  marks reversed.  `[exact: proved + verified on all 24 exact record stretches on file (150 gates),
  partner always a DIFFERENT column; the two `F_2(59)` maximisers are an exact mirror pair
  INCLUDING their flanks, kernel-checked as `CrtSlots.mirror_59`]`  A search that has found one
  maximiser is provably incomplete, and the partner's address is `P - k - s`.  No inequality on
  `Q*_J` or `F_J` follows.
- **Word reversal.**  The mirror sends an occurrence of `w` at `k` to an occurrence of
  `reverse(w)` at `-(k + span w)`, bijectively, so `#occ(w) = #occ(reverse w)` exactly and
  realisability is reverse-invariant, kill words included.  For a PALINDROMIC tuple of span `s`,
  `#occ(w)` is ODD iff `w` occurs at the single candidate address `k_w = -s/2 (mod P)`.
  `[exact: proved; gated on the exact 4-tuple dictionaries at m23/29/31/37 (15,696 / 45,854 /
  115,193 / 291,675, reverse-closed) and on two CRT transfer supersets (2,435,140 and 4,239,676
  tuples) which had no a-priori reason to inherit the symmetry]`  Audit of the project's own logs:
  82 word decisions, every reverse pair agreeing, and **12,877 s of 27,946 s (46%) had been spent
  deciding the SECOND member of a reverse pair.**  Word reversal is the SAME involution, not a
  second one (r28 self-correction).
- **`F_2 >= 2 d_0`, and `d_0` in closed form.**  By the mirror the two gaps around column 0 are
  `(d_0, d_0)`, so `F_2 >= 2 d_0` at every symmetric two-tooth sieve; and the wrap gap of a period
  equals the FIRST gap, `wrap = P - x_{N-1} = x_1 = d_0`, `= 2,3,3,5,5,5,7,7,7,10` at m7..m41.
  `[exact: theorem + closed form; gated at 15,217 counterfactual machines]`  This is how a real
  defect in `gap_pair_hist.csv` (a full-period census taken linearly over a circle) was found and
  repaired without a rescan.
---

## 4. Where the openings are inside the window

### 4.1 The window identity, and the kernel route

Inside `(y, y^2]` the gears below `y` decide primality exactly -- a composite there has a prime
factor strictly below `y` -- so a column of the machine `{5..y}` that is open and lies in the
window IS a twin pair.  `[kernel]` (`Horizon.exists_prime_factor_lt`,
`prime_of_no_prime_factor_lt`, `twin_of_no_prime_factor_lt` with the strict bound `p < y`;
`BlockedSlots.survivor_iff_twin`, `twins_infinite_iff_survivor_in_window`); `[exact: the admissible
pattern and the twins coincide on `(y, y^2]`, `survivors(y,K) = T(6K+1) - T(y)`, verified at
`y = 11..1009`]`.  The kernel's window is `k in (q/6, W]`, `W = (q'^2-1)/6`, and it is the OPENING
STRETCH of the machine's periodic pattern, containing every lower section.

Two structural riders, both on record:

- The top gear's whole unique contribution is boundary: its self-pair at the bottom edge and its
  square at the horizon, which false-positives exactly when `y^2 - 2` is prime.  Downward exclusion
  of gears therefore halts at the first `q` with `q^2 - 2` prime (the **square gate**), and a gear
  is NEEDED iff it owns a pseudo-twin in the window -- droppability is transient.  `[exact:
  minimal certifying sets verified y = 13..59]`
- Beyond the horizon openness is NOT twinhood: nudge home 595 of the `{5,7,11,13}` machine is
  `(3569,3571)` with `3569 = 43 x 83`.

### 4.2 Against the whole window, the position drops out

`F(q) < W(q) - q/6` forces an opening in the window **whatever the pattern does at `q^2`.**
Measured `F/W = 0.25` flat from `q = 5` to 53; `F(59) = 161` against `W = 620`.  And if the budget
inequality holds at every rung then `F(y) <= sum_{q <= y} q ~ y^2/(2 ln y)` against `W ~ y^2/6`, so
`F/W <= 3/ln y < 1` for `y > 20`: an opening in every window, twins infinite.
`[measured: the ratio, q = 5..59]`; `[exact: the implication is arithmetic]`; the budget inequality
holds at every computable rung through 59 (203 against budget 204 at 53->59).

The same arithmetic in adjacent-frame units: `F(2,y) <= 354 + alpha(S(y) - 328)`, checked at every
prime in `[53, 10^6]`, zero failures, worst ratio 0.6557 at `y = 113` (`alpha = 3`); beyond `10^6`
by Rosser-Schoenfeld.  `alpha*(y) = [(y^2-y)/2 - 354]/[S(y) - 328]` is 5.64 at `y = 101`, 8.71 at
`10^4`, 13.3 at `10^6` -- asymptotically `ln y`, so ANY fixed per-step constant `alpha <= 2.5`
delivers.  `[exact: conditional theorem, checked to 10^6]`  The open link is the per-step statement
itself; the observed maximum is `2.432 q` at gear 37.

**This is the whole open part.**  Everything else in this section is positional information that
the whole-window statement does not need.

### 4.3 Where the worst stretches actually sit

The worst run of the machine's pattern is longer than the current SECTION already at `q = 17` (17
against a section of 12; 144 runs `>= W` covering 2.3% of the period), and `F(59) = 161` against a
section of 40.  Full-period figures (blocked-column counts): `q = 7`: period 35, worst run 4,
`W = 12`, 4 open in window, run entering the window 2; `11`: 385, 6, 8, 2, 4; `13`: 5005, 10, 20,
7, 4; `17`: 85085, 17, 12, 2, 4; `19`: 1616615, 24, 28, 4, 11; `23`: 37182145, 33, 52, 8, 7.
`[exact: full-period computation, research/anchor235/period_vs_window.py]`

**So existence in the section is POSITIONAL**: the worst runs sit DEEP in the period, in mirror
pairs at positions `k` and `P - k` (fractions 0.3-0.7 of the period, or at the period's ends),
**never at the window**.  The run the pattern happens to have AT `q'^2/6` is short.
`[measured: to q = 5000; mirror pairing exact at every machine]`

Three different "how short" statistics are on record, and they are three different quantities --
do not read them as one number:

- the run of the machine's own pattern at `q'^2/6` is at most **0.663** of the section (worst at
  `q' = 137`), to `q' = 5000` (`docs/proof-search/anchor-235.md` 7, via harvest_shared F);
- `G_S/|S|`, the largest gap between twins inside the section (or edge to nearest twin), is below 1
  everywhere with max **0.684** at 29->31, falling 0.352, 0.221, 0.177, 0.092 by band
  (`word-tree.md` 7.3);
- the longest blocked run of ANCHOR-OPEN columns inside a section grows like `q^0.51` while the
  section grows like `2q ln q`, so run/section FALLS: median 0.235 -> 0.020, worst **0.544**
  (29->31) -> 0.085 (`anchor-235.md` 4).

### 4.4 The section: every one on record holds an aligned column

`[measured: 667 sections to q' = 5003]`  Every section holds a twin.  Two binnings are on record
and they report different minima, so both are printed: `harvest_shared.md` (from `word-tree.md`
7.2/7.3) gives minimum counts rising **2, 6, 10, 21, 51** across its bands, always at a gap-2 rung
(whose section is the shortest, `|S| = (4q'-4)/6`); `harvest_lanes.md` (from `anchor-235.md` 4,
`section_trend.py`) gives **2, 3, 6, 7, 19, 21, 42, 51, 68** across the bins 5-50 .. 4000-5000.
The two agree in kind, not in bin structure.

Aligned count per section `= anchor-open x prod_{7 <= g <= q}(1 - 2/g) x 0.66-1.0`.  Twin counts
are Hardy-Littlewood: observed/predicted 1.0028 over `1000 <= q' <= 5003`.  The section is a
Mertens word at every scale, with sections differing from one another only by scale.

Recorded overstatement, corrected: "every section holds an aligned column" is STRONGER than the
twin conjecture -- a dead section would be a twin gap `>= 4 sqrt(x)`.  And nothing here is provable
by the machine: twin gaps are unbounded in principle and the `ln^2` scale is heuristic.

### 4.5 Inside the section the machine below is exact and the new gear is silent

Every composite below `q'^2` has a prime factor `<= p`.  So inside the section `p -> q'` the gears
`5..p` are EXACT -- the periodic word of `m_p` restricted to the section IS the twin-prime
indicator there -- and the new gear `q'` does nothing in its own section, its first strike being
`q'^2`, the far edge.  **The section attributed to machine `q'` is the last stretch where the
PREVIOUS machine is still telling the truth.**  `[exact: forced]`; `[measured: 667 sections to
q' = 5003 -- gear `p` is the death rung of at most 3 columns in its section, and of NONE at 77% of
sections with q' >= 500]`  As NUMBER-strikes `p` reaches up to six (`p q'`, `p q'_2`, ...); the "at
most three" is about DEATH RUNGS.

The section's blocked word is the divisibility lattice: for every gear `s <= p` the strikes in the
section are `K_s(p -> q') = s x { m in (p^2/s, q'^2/s) : no prime factor below s }`, and the set on
the right is the OPEN WORD of the sub-machine with gears below `s`, read at numbers.  So
`blocked(p -> q') = union over s of s * open_{<s}((p^2/s, q'^2/s))`, and **a new twin is a column
that no such scaled open word reaches on either side.**  `[measured: gated over 666 sections, with
the per-gear bands contiguous]`  Recorded negative from the same run: **no section-specific feature
of gear interactions was found** -- both which vectors survive and where the strikes come from
reduce to CRT and to the smaller machines' open words.

### 4.6 What enables a new opening in the section

- **The residue vectors are uniform.**  Over the 122,546 new twins with `q' >= 1000`, the residue
  classes are uniform over the tooth-avoiding classes to total variation 0.0026 (mod 5), 0.0033
  (mod 35, 15 open classes, least 0.0658 most 0.0675 against 0.0667) and 0.0097 (mod 385, 135
  classes).  **The enabling alignment is the CRT product, nothing finer**: there is no preferred
  combination and no gear whose position in its own word makes a new twin more or less likely.
  `[measured: 667 sections, twin_provenance_r29.py, 8/10 gates]`  For the proof this says: killing
  twins for ever would need a rung from which no tooth-avoiding vector lands in the section, and
  the vectors that land are the generic ones -- the kill would have to remove every class at once,
  not a pattern.
- **Provenance.**  (i) The two sides of a new twin are INDEPENDENT (framing-pair joint within TV
  0.024 of the product of its marginals; left marginal `5: 0.665, 7: 0.134, 11: 0.045, ...`,
  `(5,5)` alone 44%).  (ii) The number of gears touching a new twin's word grows like the RECORDS
  of an iid Mertens sequence: 2.2 at `q' < 100`, 3.7 at `q' ~ 5000` -- about one more gear per
  factor 10 in `q'`, model within 6%.  (iii) The top of a new twin's provenance is a gear `> p/2`
  about HALF the time (46-48% at `q' >= 1000`), because a new twin lives at numbers `~p^2` where
  near-twins have density of the same order as twins.  `[measured: 130,664 new twins over 667
  sections]`  `q'` itself appears in NO provenance.
- **How a blocked run inside a section is assembled.**  Depth (distinct death rungs) 6.6 -> 36.8
  across the bands while run length grows 15.6 -> 197.8; single-kill levels are 58-63% of the depth
  and the top single-kill chain 46-48%, in every band from `q' = 5` to 5003; pooled over the 502
  sections with `q' >= 1000` the top five levels are single-kill in 100% of trees.  In tuple
  coordinates 60% of merges are EXTENSIONS and 40% JOINS in every band, the median join ratio is
  exactly 1/2 through the middle of the tree (the `5,7` comb leaves pieces of lengths 1, 2, 4
  only), and the top of the tree is UNBALANCED (last merges join pieces in ratio ~1:3).
  `[measured: 667 sections, exploratory, not pre-registered]`  Nothing repeats at the tuple level
  (no top 3-tuple pattern reaches 3% of sections); only the statistics are universal.

### 4.7 The walk from `q^2`

Starting at the column holding `q^2` and stepping by residue tests only -- no primality test
anywhere -- the walk lands on an opening of `{5..q}`, and that opening IS a twin pair.  From
`q = 37, 97, 499, 997, 4999, 10007, 100003` it lands at `1427|1429` (10 columns), `9419|9421` (2),
`249131|249133` (22), `994067|994069` (10), `24990239|24990241` (40), `100140119|100140121` (12),
`10000600481|10000600483` (79 columns, in a section of 533,392).  `[exact:
research/anchor235/slot_walk.py; all landings verified twins]`

Run as the layered closure `W_g = W_{g-} + hits of g` from `q^2` under gears `5..q` for every prime
`q <= 5000` (667 walks): **every landing is a twin prime pair**; walk length median 19, maximum 265
at `q = 4637` (second 187 at 2593 and 4003); between 1 and 44 layers hop per walk; and total hops
equal the walk length in every walk -- an identity, since each traversed column is counted once, at
the layer of its smallest blocking gap.  `[exact: 667 primes]`  In the other convention the first
twin sits a median 18 columns past `q^2`, maximum 264 at `q = 4637`; the position of an open whole
cycle inside a section is uniform (quartiles 0.24, 0.48, 0.74).

Existence for a FIXED gear set is CRT (`prod(q-2) x 3/5` open columns per period, never zero); for
the GROWING gear set it is the conjecture.  The walk is closed by a recursion of depth `pi(q)` and
no formula collapsing the recursion has been found.

### 4.8 The onset of the window

- **The onset law.**  `L0(y)`, the lag from the window's start to the first column with BOTH members
  composite, satisfies `L0(y) <= L* = 27129` for every `y`, unconditionally (via
  Montgomery-Vaughan `pi(x+H) - pi(x) < 2H/ln H`, since `6L* + 2 > e^12`).  `[exact: unconditional
  theorem of the programme]`  Measured over 442 windows `13 <= y <= 3163`: max `L0 = 17` (at
  `y = 13`), `L0 = 0` in 153/442, a twin precedes the first double in 132/442, and the first double
  sits at column ~2-4 with no growth in `y`.  Scope: 310 of 442 real windows have NO twin in the
  onset prefix, so the onset scale is not itself a contradiction.
- **The first double column is `k = 20`** (`119, 121`); every column `k <= 19` has a prime member.
  `[exact]`
- **Whole anchor cycles.**  A cycle `j` (three whole twin slots) is untouched by gear `q` iff
  `j mod q` avoids six residues `((q m - 11) div 30) mod q`; under gears `7..Q` the open cycles are
  a fixed pattern of period `prod q` with `prod (q-6)` open cycles per period.  Against the window
  sections to `10^8`: 1226 sections, 1088 with no open cycle, 121 with one, 16 with two, 1 with
  three; the share holding one rises 0% (`q < 100`) to 13% (3000-10000); longest dry stretch 50
  sections (`q = 7079..7549`).  `[exact: below 10^8, 156 such cycles, all on the rule]`  The
  section is NOT the natural unit for whole cycles, and existence for the growing gear set is the
  Hardy-Littlewood sextuplet conjecture -- stronger than twin primes.
---

## 5. What needs the real teeth: the counterfactual family

The machine has two inputs -- WHICH gears, and WHERE the teeth are.  The **tooth-counterfactual
family** keeps the gears and the mirror symmetry (teeth at `+-v_q`) and lets `v_q` range over
`{1..(q-1)/2}`.  Every member has the SAME period, the SAME `prod(q-2)` openings and the same
per-gear density; only positions move.  `|V(y)| = 30 / 180 / 1440 / 12960` at m11/13/17/19.
This is the project's clean null model for every alignment statement, and it sharply localises
where arithmetic enters.

### 5.1 What holds on every tooth-moved machine

- **The record law / attainment identity is STRUCTURAL.**
  `max(F_2(M), max_{J >= 3} Q*_J(M; q')) = F(M+q')` holds EXACTLY at every one of 27,570
  counterfactual machines, zero exceptions.  `[exact: 27,570 members]`
- The alignment law, CRT, the mirror, T2/T3, R89/R90 and the peel bound's flank consequence all
  hold at every member (the last asserted at all 27,570).
- The spectrum bound on `L` holds at every one of 165,584 counterfactual rows, zero violations,
  including the family's `L = 5` member where the PARITY form equals 5 exactly.

**So the counterfactual obstruction is an obstruction to BOUNDING `Q*_J`, not to the record law --
a strictly smaller target.**  Only the SIZE of `Q*_J` is arithmetic.

### 5.2 What does not hold

- **The budget inequality itself fails at 0.00-0.56% of members.**  `[exact: exhaustive m11..m19]`
- **The increment law fails at 13.3 / 13.9 / 14.5 / 21.7 / 22.3 per cent** of members at 7->11 ..
  19->23, and the rate GROWS with the machine.  Pinning the incoming gear's tooth to
  `v_{q'} = round(q'/6)` and letting the old teeth range freely drops the violations to
  0 / 0 / 1.1 / 6.5 / 5.7 per cent: **the new gear's tooth carries most of the law.**
  `[exact: exhaustive m11..m19 and the full 142,560-member 19->23 family]`
  Consequence, stated by the record: no argument using only "same gears, same density, symmetric
  teeth" can prove the increment law.
- **`L` is not capped by the real machine's constant.**  Max `L` over the FULL family is
  `1, 3, 3, 3, 5` at 7->11 .. 19->23 against the real machine's `0, 1, 1, 1, 2`.  The `L = 5`
  member (`J_max = 7`, `A_kill = 6`, beyond anything the corpus shows below m47) is `V(19)`'s
  `(1,2,5,2,1,5)` with `v_23 = 9`, word `[5,18,5,18,5]`, residues mod 23 alternating 16, 21.  EVERY
  deepest word at every step is LITERAL.  `[exact: exhaustive at five steps, 165,584 rows]`
  **Any proof that `L` is bounded must use the teeth.**
- **The residual violators are not a congruence on `F(M)`.**  `F(M) mod q' in {0,a,b}` has
  sensitivity 34.0% at 17->19 and 5.6% at 19->23; the best predictor of that form reaches 57.9%
  balanced accuracy; 94.4% of residual violators have `F(M)` NOT congruent to a legal letter, and
  the depth-3 attaining middle is the old record in 0.0% of 19->23 violators.  What DOES describe
  the residual set is a DEPTH-4 word-legal run (70% of 19->23 violators are invisible at depth 3)
  plus the flank condition `min flank > s_min`.  `[exact: pinned family, three steps]`
- **The depth-2 half has exactly one family failure mode.**  `F_2 >= 2 d_0` always, so
  `F_2 <= F + q'` can fail by that self-mirror 2-run alone whenever `2 d_0 > F + q'` -- and over
  14,616 exhaustively enumerated old machines it fails at exactly ONE, `V(19)`'s `(1,1,4,3,5,2)`
  with `F = 26`, `F_2 = 50`, `d_0 = 25`.  Excluding wrap-pair members the minimum slack is
  8/6/6/5/4/9, positive at every step.  `[exact: gated at 15,217 old machines]`  It is exactly the
  one depth at which the mirror lever needs a hypothesis (`d_0 != F`), and `d_0` is a closed form on
  the real machine.

### 5.3 Where the teeth enter: gears 5 and 7, through the bare alternation

Call `(a,b,a)` **admissible** if some residue mod 5 (and mod 7) carries `r, r+a, r+a+b, r+2a+b`
outside the gear's tooth pair.  Then on the family
`P(L >= 3 | admissible) = 0.006 / 0.101 / 0.272 / 0.320` and
`P(L >= 3 | NOT admissible) = 0.0000 / 0.0000 / 0.0001 / 0.0000` at 13->17 .. 23->29, with
**0 of 4 / 605 / 19,408 / 1,340 bare-letter `L >= 3` words inadmissible** (0 exceptions in 21,357
rows).  `[exact: exhaustive on the family]`  The necessity direction is `[kernel]`
(`BareAlt.no_gapWord`).

The REAL machine's alternation is NOT admissible at 13->17 `(6,11,6)`, 17->19 `(6,13,6)` and
23->29 `(10,19,10)` -- so its `L <= 2` there is decided by gears 5 and 7 ALONE -- and IS admissible
at 19->23 `(8,15,8)`, where `L = 2` is a fact about the higher gears.  Gear 5's tooth explains
17.3% of `L`'s variance at 17->19 (more than the incoming tooth's 12.5%; all 22 pinned `L = 3` rows
have `v_5 = 2`), while every old gear above 7 explains under 1% at every step.  This is an
OBSERVATION with one exception class (the shifted letter `a + q'`), not a theorem -- and it is the
round-31 bare-word lemma seen on the family.  The two channels (letter size and `{5,7}`
admissibility) are near-orthogonal and together explain only 36-42% of `L`'s variance; `L` is NOT
monotone in the letter size, and the SMALLEST letter gives the SHORTEST words.

### 5.4 The twin machine is a low-`F` outlier among its own counterfactuals

`F(twin)` sits at the 20.0 / 18.1 / 26.4 / 17.1 / 11.9 percentile of `V(y)` at m11..m23, about
10-15% below the median, never the minimum, in a family whose maximum is 1.6-1.9x the truth.  The
placement STRENGTHENS WITH DEPTH at the two largest machines (m19: 17.1 / 12.3 / 6.3 for
`F`/`F_2`/`F_3`; m23: `F` 11.9%, `F_2` 3.1%) -- and the route consumes `F_2`, not `F`.  The
increment law's own margin `s_min - increment` puts the twin at the 66.8-83.3 percentile.
`[exact: exhaustive m7..m23, pinned at m23; the rows are NESTED, so no p-value is claimed]`

Honest negatives attached: the BUDGET SLACK `F(M+q') - F(M) - q'` is UNDISTINGUISHED (59.0 / 37.2 /
49.3 percentile at the three largest steps); the real machine's depth-2 slack is ORDINARY (23.7-86.5
percentile); and the MECHANISM of the outlier is OPEN with three candidates dead (section 8).

### 5.5 The section-view counterfactual: the real teeth sit on the densest class

Pooled over sections `q' >= 1000`, moving gear 13's teeth to ANY other value `v` leaves 3.6-3.8%
MORE survivors than the real teeth; gear 7's 6.4%; and all moved positions agree with each other --
**the real class is the odd one out.**  Mechanism, parameter-free: among the columns no gear but
`s*` touches, the tooth class is richer than every other class by 1.160 (`s* = 7`), 1.202 (13),
1.273 (31), and the cofactor model `ln n / ln(n/s*)` gives 1.138, 1.190, 1.271.  A number `s* x m`
in the tooth class is clean iff its cofactor `m`, smaller by the factor `s*`, is prime -- likelier
than for a full-size number.  `[measured: 667 sections; a pre-registered prediction REFUTED -- the
section CAN tell real teeth from moved ones]`

So: **the real teeth of every gear sit exactly on the residue class where the relaxed machine's
survivors are densest**, and the real machine therefore removes more of them than any counterfactual
teeth would, in every section, by `ln n / ln(n/s*)`.  This is the section-view face of the period
result (the real machine a low-`F` outlier under moved teeth); it explains the SIGN, not the size.

### 5.6 A bookkeeping note on the family sizes

The 27,570 total for the record-law check is `30 + 180 + 1440 + 12960 + 12960`, i.e. the fifth row
(19->23) counted as a pinned 12,960-member row; the increment-law statistic at the same step used
the full 142,560-member family, and the 23->29 row in `harvest_shared.md` is a 601-member SAMPLE.
Different family checks therefore quote different totals (27,570 / 142,560 / 165,584 / 15,217 /
14,616).  Each number is correct for its own check; they are not interchangeable.
---

## 6. The ceilings on the record: what stops, and where

This section collects the results that say a whole class of instrument cannot bound the record or
the alignment depth.  They are the most valuable entries in the record, because each one closes a
direction permanently.

### 6.1 Escape distance = 1 (bounded-modulus arithmetic constrains where, never how big)

Over all 1,225 gap pairs mod 35, every `(G1,G2)` is within L1 distance 1 of a corridor-allowed
pair.  A near-maximal gap has about 35 candidate lengths in its range, so any residue exclusion is
evaded by a `+-1` slide in one component.  **Corridor arithmetic constrains WHERE top-gap
configurations sit, never HOW BIG they are** -- at modulus 35 and, by the same argument, at ANY
bounded modulus, since the exposed set's own max gap stays `O(1)`.  `[exact: all 1225 pairs plus a
general argument]`  Corroborated: tier B (lifting the modulus to 385, 5005, 85085, 1616615) adds
exactly zero exclusions at all 16 word-step pairs; and `MF_3 mod 35`, `MF_3 mod 385` and
`MF_4 mod 35` are identical at every step -- neither a finer modulus nor more history buys one
unit.

### 6.2 CORRCAP is infinite from 53 -> 59

`CORRCAP(q', F)` is the longest T3-legal word with values `<= F` whose prefix-sum walk stays inside
`E mod 35` -- the strongest cap gears 5 and 7 can EVER give.  It is `4, 2, 3, 5, 25, 25, 11, 5` at
19->23 .. 47->53 and **INFINITE from 53 -> 59 on**, and at every larger `F/q'`.  Mechanism: padded
letters step by `j q' mod 35` and `gcd(q',35) = 1`, so once `F/q'` is large the steps fill `Z_35`
and the corridor acquires a cycle.  `[exact: explicit automaton on the 35 x 3 corridor states with
cycle detection, GATE B5 r31; R75's row reproduced 9/9]`  Since `F/q'` grows without bound
(1.1, 1.2, 1.4, 1.6, 2.1, 2.1, 2.2, 2.2, 2.5 at 19->23 .. 53->59), **no fixed set of small gears
can ever cap the alignment order again.**

The term that makes it infinite is the ALPHABET SIZE `~3F/q'`.  The bare alphabet is two letters at
every machine forever, which is exactly why `PSORD <= 5` is uniform and `L_bare` is capped while
`L_pad` is not.

### 6.3 No fixed-depth counter can bound `L`

With `A_m >= S_m = S^(0)_m >= S^(2)_m >= S^(4)_m >= D_m` (abstract T3 words; exposure survivors;
Bonferroni depth-`s`; realised), **`S^(2)_m = S^(4)_m = S_m` at all 21 measured cells (m19..m37)**
-- fixed-depth Bonferroni kills NOTHING -- while the exact count `N(w)` sits far below the depth-0
term: `min E_0/N` is 6..16 at `m = 1`, 845..10,742 at `m = 2`, 145,158 / 312,151 at `m = 3`,
4,344,055 at m37 `m = 2`, growing in both `m` and `M`.  Verdict: `E_0(w) = prod_g c_g(X)` is a
`P`-scale count of columns with the pattern's points open and the higher terms are bounded-ratio
corrections, so `E_s < 1` cannot happen at fixed `s` until the exposure half has already killed the
word.  **A uniform bound on `L` needs the cover half at FULL depth (`2^{|Y|}` per word) on a
candidate set that is itself unbounded in `M`.**  `[exact: proved for fixed-depth truncations given
the measured EXPCAP growth]`  "No counter of any kind" is labelled JUDGMENT, NOT RESULT.

Adjacent dead ends of the same shape:

- **Pairwise convexity computes the record through m17 and provably stops at m19.**
  `L*(y) = min{L : the level-2 (Sherali-Adams) covering relaxation proves RUN(L) impossible}` equals
  `F` exactly at m11, m13, m17 (7, 11, 18).  At m19, `L* = 27` against `F(19) = 25`
  (`V(25) = V(26) = 0`), and PSD does not repair it: **every certificate of `F(19) <= 26` must use
  THREE-gear information; no pairwise-consistent reasoning, linear or semidefinite, suffices.**
  Vacuity ratio `L*/F` = 1.000, 1.000, 1.000, 1.080, 1.647, `>= 1.721` at m11..m29.
  `[exact: soundness proved; every claimed bound carries an exact rational dual verified in integer
  arithmetic; the m19 SDP verdicts are numerical and flagged]`  Three certificate families
  (potentials, covering duals, moment hierarchies) now fail along the SAME axis -- arity, not
  convexity.
- **Congruence-class potentials certify nothing at any modulus**: a potential that is a function of
  `k mod m` for a proper divisor `m` of the period forces `0 >= m`, because every class mod `m`
  contains a blocked column.
- **Moment / PSD / Chebyshev routes**: margins 67.6 .. 4.3e10 and GROWING; the L2 instrument's
  spread is real but essentially uncorrelated with `F` (spearman -0.038, +0.023, -0.186), and
  NEGATIVELY correlated at the largest machine.  The L1 character bound is provably BLIND to the
  teeth: `sum_m |Shat(m)|/P = prod_q S_q/q` does not depend on `v_q`, identical at all
  30/180/1440 counterfactual tooth vectors while `F` spreads 1.8-2.5x.
- **Capacity and overlap counting.**  The extremal covering leaves NO gear idle and runs at
  near-perfect efficiency (all 1.000 at `y = 19, L = 74`; 0.833-1.000 at `y = 31, L = 173`), with
  average multiplicity matching `2 sum 1/q` almost exactly (1.973 vs 1.911; 2.139 vs 2.131).  A
  counting bound needs `mult` to exceed `2 sum 1/q ~ 2 log log y` while measured `mult ~ 2` -- the
  required forced overlap is a factor `log log y` above what occurs, so **the capacity bound is
  close to achievable and its failure is not slack.**  The overlap-counting certificate
  (`f(L) >= L` necessary) gives `y=23: 34 true vs 50; y=29: 43 vs 135; y=31: 58 vs 1043` and is
  vacuous from `y = 37`; and it is "the period scan wearing a disguise".  The corridor's only size
  statement, the local capacity cap `F2_k(y) <= (2#K+1)/(rho - 2 sum 1/q)`, dies two to three gears
  above ANY base (tight once: `F2_k(11) <= 12` against 11).

### 6.4 The 2F wall (the depth-2 slack)

`S_2 = F + q' - F_2`, measured 9..49 and growing roughly with `q'`.  The free peel reduction gives
only `2F_2 - q'` at `J = 4`, short by exactly `F_2 - a`.  The tight rearrangement bound is
`F + G_2 = 2F` (maximal gaps are mirror-paired), which is over budget from 19->23 on -- and since
every unitary invariant of the machine is a function of the gap histogram, that is a THEOREM that
the invariant route dies.  **No instrument on record supplies `S_2`.**  `[measured: the row;
the wall itself is a theorem about what cannot supply it]`

Related, and recorded as a refutation of an assumption rather than of the object: "`F_2` needs
slack below the budget" is FALSE -- every `U <= 74` certifies, `U in [75,85]` stalls at `U`,
`U >= 86` stalls at 86, so the obligation is EXACTLY the two-gap statement itself, with zero
further slack.

### 6.5 Two named constructs that would supply the missing upper bounds, neither built

- **The coverability spectrum `COV(M)`.**  A gap of exactly `v` at machine `M` means `v-1`
  consecutive columns ALL struck with both endpoints spared.  That is not a residue-marginal
  question about `v`, it is a covering-feasibility question about the gear set, so the hole set is
  the complement of `COV(M) = { L : an interval of L consecutive columns is coverable by the gears
  5..M, with both flanking columns spared }`.  It is CRT arithmetic on the gear set, computable
  WITHOUT SCANNING THE PERIOD, so it reaches machines 37, 41, 43, 53 whose periods (1.2e12, 5.1e13,
  2.2e15) are beyond any scan -- and it therefore yields **the UPPER bounds on `F` and on the `F_j`
  that every prefix row lacks**, which is the single missing input for the qualifying-spectrum
  criterion at those steps.  `[BUILT in round 20: research/cov_sat.py, mechanic.md K1 - exact spectra with
  complete hole lists at m11..m37, m41 complete; the harvest's NOT BUILT tag was stale. Round 32:
  research/cov_sat_r32.py adds F(61) >= 171, F(67) >= 175, F(71) >= 185 as verified lower bounds; UNSAT
  cost grows 6-11x per rung, so no upper bound past m41.]`
- **The renewal factor.**  The multi-lag exposure bound is the only step in the whole route with no
  heuristic: "gap `= v`" is (both endpoints exposed) AND (no opening strictly between); dropping the
  second only increases the probability, and exposure is a CONJUNCTION so it factorises by CRT,
  giving `p_j <= (1/rho) sum over qualifying tuples prod_q c_q(0, v_1, v_1+v_2, ...)/q`.  Measured
  SHORT by a factor 2-29.  **The missing factor is exactly the dropped condition -- a closed-form
  lower bound on `P(no opening strictly between | both endpoints exposed)` at separation `v`.  That
  single factor is the entire remaining gap between the rigorous exposure bound and sufficiency.**
  `[rigorous inequality; the factor is NAMED and NOT BUILT]`
  The related renewal LADDER is built and rigorous: for ANY subset `Y` of the interior offsets,
  `#{k mod P : X open, Y blocked} = sum_{T subset Y} (-1)^{|T|} prod_q c_q(X u T)`, exact
  closed-form CRT arithmetic, every choice a VALID upper bound, nesting the chosen points giving a
  monotone ladder from the exposure bound to the exact count at cost `2^{|Y|}`; three rungs already
  clear the route's requirement at every constrained case.  What it is checked against still carries
  a fitted constant `lambda`.
---

## 7. The picture put together

**What is proved, in the kernel.**  A column of the machine `{5..y}` that is open and lies in the
window IS a twin pair.  Each gear has exactly two teeth, at `+-round(q/6)`, never adjacent, never
at column 0, never at the antipode.  Adding a gear is `q'` copies of the old pattern, one per
deletion phase, each opening dying in exactly two of them; so every new gap is a sum of consecutive
old gaps, and the record grows only by merging.  Which runs can merge is decided by residues alone:
interior gaps lie in `{0, +a, -a} mod q'`, the nonzero classes strictly alternate, padded letters
are transparent, consecutive strikes are at least `a = 2u'` apart.  Literal chains have at most six
members at every gear forever, and the bare part of the alignment depth obeys
`L_bare <= PSORD(q' mod 210) <= 5`, uniformly, from gears 5 and 7 alone.  The machine's only
symmetry is the mirror, and it is worth exactly one factor of two.

**What is exact at every computed machine.**  The record of the bigger machine is computable from
the smaller one, three ways that agree: as the widest run of lower openings lying in one two-class
set, plus its two flanking gaps, maximised over the `q'` phases on ONE lower period; as
`max(F_2(M), max_J Q*_J(M))`, which EQUALS `F(M+q')`; and as
`max over compatible words w of [span(w) + FS_max(w)]`, the word list depending on `q' mod 210`
alone.  That identity is structural: it holds at all 27,570 tooth-moved machines.  The depth of an
alignment is a word-length question -- `J_max = L + 2`, `A_kill = L + 1 = D_g` -- and `L` obeys the
proved bound `L <= 2F(M+q')/q' + 1`.  Every computable rung satisfies the budget inequality, the
last being 53->59 at 203 against 204.  Inside the window, `F/W = 0.25` flat from `q = 5` to 53; the
worst stretches sit deep in the period in mirror pairs, never at the window; every one of 667
sections to `q' = 5003` holds a twin, the first one a median 18 columns past `q^2`; and the residue
vectors that enable a new twin are uniform over the tooth-avoiding classes -- the enabling
alignment is the CRT product, nothing finer.

**The single statement that remains.**  It is an alignment statement about openings inside the
window: *for every `y`, the machine's longest opening-free stretch stays below the growth of its
own window* -- `F(y) < W(y) - y/6`, which forces an opening in the window whatever the pattern does
at `y^2`.  Equivalently, and this is the form the ladder uses, the budget inequality
`F(M + q') <= F(M) + q'` holds UNIFORMLY, at every rung rather than at every rung computed so far.
Equivalently again, in the form Lateral round 31 wrote it, the Jacobsthal-square condition

`8F <= q'^2 - (eps + 12) q' + 16`,  `eps = F_2(M) - F(M)`,

which the spectrum bound on `L` turns into the budget inequality outright, and which is true at 8
of the 13 corpus steps, failing only at the five small ones.

**What the record says bears on it directly.**  The attainment identity, because it makes the
target exactly "bound `Q*_J`", with no slack to give away.  The counterfactual family, because it
shows the identity is structural while the SIZE is arithmetic, and that any proof must use the
teeth -- the incoming gear's tooth carrying most of it.  The spectrum bound on `L`, because it is
the only proved bound on alignment depth at every machine, and it is what produces the
Jacobsthal-square condition; and its two open ends: `L_pad` (the bare half is closed, the padded
half is not) and whether the bound is tight infinitely often.  The `A_kill <= 3` pattern in the
spectrum-plus-depth certificate, because every step with shallow kill arity certifies with margin
+10 to +24 and every failure is a deep one.  The peel bound and the triple inequality, because they
discharge depth 3 with no hypothesis and leave `J >= 4` -- the 2F wall -- as the depth-2 slack
`S_2 = F + q' - F_2`, which nothing on record supplies.  The `F_3` wall, because it is the one
identified per-step event (the old machine's depth-3 record carrying the padded letter as its
middle, base rate `3/q'`) at which the increment law is known to fail, by exactly
`F_3 - F_2 - s_min`.  And the ceilings, because they say where not to look: bounded-modulus
arithmetic constrains where and never how big (escape distance 1); gears 5 and 7 stop capping the
order at 53->59 (CORRCAP infinite); no fixed-depth counter can bound `L`; no pairwise-consistent
certificate reaches m19.

Two constructs, both named in the record and neither built, would supply the missing upper bounds:
the coverability spectrum `COV(M)`, which is CRT arithmetic with no period scan and would give the
`F_j` upper bounds every prefix row lacks; and the renewal factor, a closed-form lower bound on
`P(no opening strictly between | both endpoints exposed)`, which is the entire remaining gap
between the rigorous exposure bound and sufficiency.
---

## 8. Refuted alignment claims

Merged from the three harvests, deduplicated, one line each with a pointer.  **Do not rederive
these.**  Where two harvests refute the same claim from different sides, both pointers are kept.

### 8.1 Chain depth, fuel and arity

- **`k <= 3` as a universal chain bound** -- REFUTED: `k = 4` exists at (gears `<= 29`, `q = 31`),
  four qualifying triples, word `(10,21,10)`.  (`chain-conditions.md` addendum.)
- **`k_max <= 3` everywhere** -- CORRECTED: it held only through 23->29; `k_max = 4` at 29->31 and
  31->37, and the `k=4` absence at 37->41 was self-demoted as evidence (`N3` is suppressed 830x
  there, so the conditioned expectation was 0.91).  (Mechanic r11-r12.)
- **`A_kill(M -> q') <= 3` as a universal fuel cap** -- FALSE at 47->53, where `A_kill = 5` exactly,
  the project's only 5-chain.  (Mechanic r25.)
- **"`k_max <= 4` at 47->53" and the fuel-cap repair of the word-free criterion there** -- DEAD, not
  deferred: `A_kill = 5` forces depth `>= 6` and `Q_6(47;18) = 174 > 171`.  (Mechanic r25, R24/C23.)
- **"Nothing seen contradicts `k_max = 3`" at 47->53** -- FALSE; the first 5-chain `(18,35,18,35)`
  is realised.  (Mechanic R23.)
- **`L <= 3`** -- REFUTED: `L(47) = 4`, decided in FOUR CRT calls (`(18,35,18,35)` realised, the
  first realised legal 4-word in the project).  (Constructor r29.)
- **`L_pad <= 2` persists** -- REFUTED: `L_pad(47) = 3` measured, and `L_pad(53) = 3` follows from
  the bare-word theorem plus the recorded `L(53) = 3`.  (Constructor r31, P6 scored.)
- **(B), "`L(M)` bounded by an absolute constant"** -- RETIRED as probably false in the limit and
  never needed: `L = O(F/q')` is a theorem and `F/q'` is measured 0.54..2.64 and growing.
  (`docs/novel/spectrum-bound-on-L.md`; Lateral r31.)
- **P7's mechanism, "`L_pad` is the cover half because padded letters are invisible mod 35"** -- the
  CONCLUSION stands, the MECHANISM is refuted: padded letters are FULLY VISIBLE to gears 5 and 7
  (they refute 13 of 26 non-bare 2-words at m47).  What makes `L_pad` the cover half is the
  ALPHABET SIZE `~3F/q'`.  (Constructor r31, corrected mid-round.)
- **The alternation-pair predictor "`A_kill >= 5` iff the pair `(s, q'-s)` is realised"** -- REFUTED
  by its own pre-registered test at 53->59: `(20,39)` IS realised with two machine-verified
  witnesses, yet `(20,39,20)`, `(39,20,39)` and all longer alternations are ZERO with no SAT call.
  Pair realisability is necessary and NOT sufficient.  (Mechanic r25/r26.)
- **R49's identity `N = max(2, A_relax)`** -- REFUTED at m37 (`A_relax = 2`, `N = 3`, bought by the
  padded cycle `14 -> 41 -> 27 -> 41 -> 14`) and again at m41 (padded 2-cycle `[43] -> [29] ->
  [43]`, dying at order 3).  (Constructor r27/r28, R75/R85.)
- **Litcap as a predictor of realised arity** -- it is a proved cap on the LITERAL part only: litcap
  4 at m41 where the literal 2-word count is exactly 0; litcap 2 at m37 where `A_kill = 3`.
  (Constructor X29.)
- **"cap `<= 6` for ALL `(t,s)` residue pairs mod 35"** -- FALSE: over all 1,225 pairs the spectrum
  is `{2,3,4,5,6,8,10,140}`; the restriction to invertible classes mod 210 does real work.
  (Formalist r13.)
- **"The truncation arity grows"** -- self-corrected: the growing sequence was the RESIDUE arity, and
  a residue-qualifying run is not a kill chain (T3 forbids two same-class letters in a row).
  (Constructor X28.)
- **Holt-Rudd's counting bounds `L(M)`** -- PROVABLY NOT, from the count alone: a `k`-run occupies at
  most 2 copies and exactly 1 unless every letter is padded, WHATEVER `k` is.  The term that breaks
  the one-class argument is the minimal in-copy hit distance `s_min ~ q'/3`, and every stretch that
  matters has span above it.  (Harvester r30 follow-on.)
- **`R45`'s `A_relax(37) = 3`** -- WRONG, it is 2: `arity_ladder.py` HARDCODED the `m=1` and `m=2`
  entries at m29/31/37 as "realised" instead of looking them up, and gear 5 refutes `(14,27)` by
  phase saturation.  (Constructor r27 self-correction.)
- **`R61`'s scan-free `D_2 = 1,254` / `D_3 = 15,020` at m31** -- should read 1,253 / 15,019; the run
  predated the `decide_cover` fix and counted the phantom `(1,1)` and `(1,1,1)`.  (Constructor r26.)
- **"1 of 4 `k=4` fuel sites is phase-aligned" / "site 858111062 is sterile forever" / a
  fuel x alignment "double rarity" multiplier** -- ALL a one-window artefact: every fuel site fires
  exactly once per new-machine period (all four fire, at `j = 12, 30, 0, 18`), and alignment is a
  DENSITY factor, never a COUNT factor.  (Lateral r11 -> r12.)
- **"Fuel `k_max` and the record are decoupled"** -- WITHDRAWN, same one-window artefact.
  (Lateral r11 -> r12.)
- **The asymptotic safety argument for lemma 2 (`excess <= span_max = 2q' + s <= 2.67q'`)** --
  WITHDRAWN: it used the cap-6 theorem, which is stated for LITERAL chains; padded runs are not
  capped by it.  (Lateral r13.)

### 8.2 The merge law, the criteria, and the record

- **M1, "every realised legal spacing value is exactly `a`, `b` or `q'`"** -- REFUTED: the exact
  legal alphabet is `{v <= F : v = 0 or +-2c mod q', v realised}` and it contains `49 = a+q'` at
  m31, `55, 68` at m37, `57, 72, 86 = 2q'` at m41; a small-machine phenomenon, alphabet growing
  1,2,2,3,3,3,4,5,6.  (Constructor r28/R86; Mechanic C43b; `two-teeth-kill-spacing.md` M1.)
- **The all-depths word-free (hypothesis-free) criterion** -- FAILS from 43->47 on with
  machine-verified witnesses (`Q_7(43;16) >= 152` at `k = 110,350,776,715,218`;
  `Q_7(47;18) >= 177` at `k = 41,120,916,229,562,503`).  It kills that form, not the budget
  inequality.  (Mechanic C20.)
- **The plain (size-floor) criterion at 43->47 and 47->53** -- FAILS (152 vs 150, 177 vs 171), and
  the failure is in the CRITERION: the m47 witness's four middles `[22,28,30,67]` all clear the
  floor `a = 18` but not one is congruent mod 53 to a legal letter.  Word legality repairs both.
  (`old-machine-spectrum.md` 8.)
- **The literal-only merge algorithm** -- incomplete, misses padded links (undershoot 71 vs `>= 88`
  at 31->37); and the fully permissive version (all spacings `{0, +-2u}` without alternation) is too
  permissive (overshoot 45 vs 43 at 23->29 on the illegal word `(10,10)`).  (Lateral r13.)
- **The adjacent-frame chain condition `{phi, phi+1}`** -- WRONG FRAME: its `k=2` count is
  `prod(q-4)`, the domino count; the k-frame teeth are never adjacent.  Superseded by
  `{phi, phi+s}`, `s = 3^{-1} mod q`.  (`chain-conditions.md`, "the frame trap".)
- **`G_2 <= F + q'` (the 3-sparse sufficient criterion)** -- fails from rung 47 (152 > 149,
  174 > 170); the stretches that beat it are not two-progression patterns, so the budget inequality
  is untouched and the relaxation is simply too loose past 43.  (`anchor-235.md` 9a.)
- **`SPEC_3` and `SPEC_4`** (the spectrum-plus-depth certificate truncated below `J_max`) -- UNSOUND
  on the counterfactual family (30 and 5 unsound cells); the depth range genuinely has to reach
  `J_max`.  `SPEC_5` is sound but certifies 0.3-1.2% where word-legality certifies 96-100%.
  (Lateral r29.)
- **Round-28's framing "the spectrum-plus-depth certificate closes rungs independently"** --
  CORRECTED: the old machine's `F_J` values are exhaustive only because of deletion-ladder caps
  taken from `F` at machines ABOVE the step, and at `j = 2` that cap is the very quantity the rung
  bounds.  Rungs below m59 are method demonstrations, not independent bounds.  (Constructor r29.)
- **The deletion ladder as the induction's supplier of `F_2`** -- numerically sufficient but
  LOGICALLY CIRCULAR, and its slack thins to 0 at 41->43.  (Constructor X36.)
- **"`F_2` needs slack below the budget"** -- REFUTED by the slack sweep: every `U <= 74` certifies,
  `U in [75,85]` stalls at `U`, `U >= 86` stalls at 86 -- the obligation is EXACTLY the two-gap
  statement itself.  (Constructor X37.)
- **Round-22's "no bounded state certifies at 29->31"** -- OVERTURNED by the history ladder:
  `A_3 + phase 385` certifies (72 <= 74) and `A_4` (three gap values, phase-free, 14,368 states) is
  EXACT at all seven scannable steps.  The missing object was the machine's dictionary of realised
  4-tuples, not a finer congruence.  (`kleene-generator.md` 4b.)
- **`A_4` as the carrier of the qualifying-tail potential** -- the qualifying sub-digraph has a CYCLE
  at m23, m29, m31, so the longest qualifying path is INFINITE; `A_5` would not fix m29 either.
  (Formalist verdict 19.)
- **The marked spectrum `Q^[J]` and the "J=5 object"** -- an implementation artefact (the DP returned
  success as soon as the mark quota was filled, never checking interiors after the last mark); the
  corrected marked spectrum is exact in all 30 entries and the 29->31 verdict REVERSES.
  (Mechanic R15; Constructor X33; Formalist verdict 12c.)
- **The C13 qualifying-spectrum table (4 of 7 rows)** -- wrong, built before the vacuity fix; the
  criterion column was always right, but any earlier use of an individual `Q_j` from that table must
  be re-checked.  (Mechanic R14.)
- **"The open part of (D) is four addresses"** -- SUPERSEDED within the same round by the qualifying
  spectrum and the span-resolved envelope.  (Mechanic r17.)
- **"Peak qualifying depth is non-decreasing in `M`"** -- REFUTED by Mechanic's own table: the peak
  is terminal at `M <= 23` and INTERIOR at m31 (5 of 7) and m37 (7 of 8).  (Mechanic r28, E1/C42.)
- **The drift recursion "new max address = f(old top-stratum address)"** -- REFUTED (reachability
  18/20 -> 0/4 and 1/2); the honest law is LOCAL, `address = pin(word)`.  (Lateral r10, refuted
  angle 5.)
- **A-priori stabilisation of the near-top word-SHAPE family** -- cross-machine full-shape recurrence
  is ZERO at all five machines; observed halves are 3.2% of admissible and disjoint per machine.
  (Lateral r11, refuted angle 6.)
- **The 2n-gap reordering as a route to `F`** -- REFUTED BY ITS OWN PROOF: over 60 admissible
  re-choices of the teeth the distinct-difference count stays `2n = 8` while `F` ranges over
  `[10,18]`, and `F` is not a statistic of the order permutation at all (a permutation records
  order; `F` needs the metric).  Marked a CLOSED LINE.  (`two-n-gap-reordering.md`; Lateral r27.)
- **The mex law (the first opening = mex of `{u_q} u {q-u_q}`)** -- held to `y = 37`, FAILED at 41
  (gave 20 against the truth 87); any rule carrying a bounded number of teeth per gear must
  under-block.  (`twin-prime-program.md` 31d.)
### 8.3 The corridor, residues and padding

- **"The corridor caps alignment length at every machine"** -- REFUTED: `CORRCAP` is INFINITE from
  53->59 on, and at every larger `F/q'`.  No fixed set of small gears can ever cap the order again.
  (`uniform-order-bound.md`.)
- **Bounded-modulus residue laws capping SIZES** -- every `(G1,G2)` pair is within L1 distance 1 of a
  corridor-allowed pair at ANY bounded modulus: corridors constrain WHERE, never HOW BIG.
  (Constructor X11; Formalist verdict 4.)
- **Tier B (moduli 385 .. 1616615) as a source of exclusions** -- adds EXACTLY ZERO exclusions
  anywhere tier A did not, at all 16 word-step pairs; "B is not a tier at all here".
  (Constructor X13/r13 25.4; Lateral X11.)
- **The machine-free corridor certificate** -- `MF_3 mod 35`, `MF_3 mod 385` and `MF_4 mod 35` are
  IDENTICAL at every step; neither a finer modulus nor more history buys one unit.
  (Constructor X34, R52.)
- **"(D) might be corridor-forced at `n = 4`"** -- 0 of the 1,225 `(span, flank-sum)` classes mod 35
  are blocked; every flank-sum value above the requirement is corridor-feasible for every span.
  (Lateral item 31.)
- **`p <= 2` (at most two padded links)** -- NOT provable from the AP lemma: the `(0,1)` and `(1,0)`
  triples survive and are corridor-feasible first at `q' = 43`, so `p = 3` is structurally permitted
  from 41->43 on.  (Lateral r17.)
- **The round-14 padding lemma as a general ceiling** -- EXPIRES exactly at 37->41: `F/2q'` climbs
  0.32, 0.47, 0.54, 0.59, 0.69, 0.78 then 1.07, and `F_2/2q'` 0.47, 0.66, 0.67, 0.67, 0.89, 0.92
  then 1.10.  "Yes for machines up to 31, and no asymptotically."  (Lateral r14.)
- **Round-16's "the ceiling stands on structure"** -- TOO STRONG: the SHAPE law is permanent but the
  COUNT `p` is not, and `span <= F + O(q')`, not `O(q')`.  (Lateral r16 -> r17.)
- **Sufficiency of the padding onset rule (`F(M) >= q'` implies supply `> 0`)** -- FALSE: machine 29
  has `F = 43 >= 41` yet `supply(29,41) = 0` exactly.  (Mechanic r15.)
- **"`F < q` is the onset condition"** -- WRONG NAMING: by `onset_gate` it is precisely the
  NO-PADDING regime, and the padding count bound `p <= F/q + 5/6` GROWS.  (Formalist r19.)
- **The smooth `supply^2/gaps` prediction of padding events** -- padding switches on and off with
  `q' mod 35`; the model predicted ~5 double-padded runs at 37->41 where the corridor forbids the
  adjacent shape outright, and was RETRACTED by its own author when `hist_37[41]` gave supply
  ~6.08e4 not ~1e6 (expected double-padded runs 0.017).  (Lateral r15; Mechanic r14 -> r16.)
- **The smooth `e^{-(q'/lambda)}` model for padding share** -- REFUTED: measured 2.27e-4, 7.54e-7,
  9.73e-6, 4.23e-6 is erratic and non-monotone, off the exponential by 20-1000x.  (Mechanic r14.)
- **"Longer literal words become profitable as lambda grows" as the 31->37 crossover mechanism** --
  at best half the story: the crossover is a PADDING ONSET (the best literal configuration reaches
  only 71 against the record 88).  (Lateral r13.)
- **Covering/capacity explanation of absent gaps** -- residual interior demand has positive slack
  (8-16 spare strikes) at every `g`; gap 24's absence is arithmetic selection plus rarity, NOT
  impossibility.  "Don't hunt one."  (Lateral r19, refuted angle 11.)
- **"No smooth law, only the histogram"** (about gap-value erraticity) -- itself wrong: the law is
  multiplicative and arithmetic, with a three-line closed form (`c_5 c_7`), accounting for about a
  quarter of what was called noise.  (Lateral r18.)
- **Phase saturation as a general instrument** -- it IS the corridor condition mod 35 by CRT, so it
  adds NOTHING to a corridor-built abstraction: it answered ZERO of the 27,197 superset-YES queries
  the 41->43 loop asks.  (`phase-saturation-arity.md` LIMITS.)
- **`R_J` giving an inequality on `F_J` or `Q*_J` at even `J` via the mirror** -- NO: `R_J` is
  span-preserving, so the only object it adds is the quotient by an involution, the SAME one unit
  the odd-`J` route gives, and the group `Z/2` is proved to be the ceiling.
  (`mirror-parity-laws.md` 9.4, 7.1.)
- **"Extremal implies palindromic"** -- FALSE at `J = 3` and `J = 4` (maximisers are reversal PAIRS,
  and Theorem B forbids a literal even-`J` palindrome outright); TRUE only at `J = 5`.
  (`per-j-window-analogues.md` 1.6.)
- **The self-mirror run's span `<= 0.8 F_j`** -- REFUTED: `span_self(j) = F_j` exactly at m7
  (`j = 3,7,9,11,14`) and m11 (`j = 11`); the correction is an explicit exception list, empty from
  m13 up.  (Lateral refuted angle 42.)
- **"Every hole lies in the top half of the gap range" as `> 0.71 F`** -- the tightest is 0.7059 at
  m23, so `> 0.70 F` holds and `> 0.71 F` would FAIL.  (Lateral item 67.)
- **"Gear 5 is the ONLY parity-obstructed gear for `p <= 37`"** -- WRONG AS WORDED: because `W_1(1)`
  is the only odd histogram entry and `1 = 1 mod p` for every `p`, `alpha_1(p)` is ODD at every
  gear, so the pole phase is unattainable at EVERY gear.  (Lateral r26 self-correction.)

### 8.4 Spectrum, flatness, flanks and envelopes

- **Raw / deep spectrum flatness `F_{k_max+1} - F <= q'`** -- FALSE at 29->31 (`F_5 - F = 42` against
  `q' = 31`); raw flatness fails 5 of 15 machine-depth pairs.  Later re-read as "a refutation of the
  WRONG DEPTH" and repaired by the suppression law.  (Constructor X17/r17 -> r19; Mechanic R13.)
- **Round-18's two-part target ((D-a) `k_win <= 3` plus (D-b) `F_4 - F <= q'`)** -- SUBSUMED: no
  separate assumption about winning depth is needed, because the suppression term kills deep runs
  automatically.  (Constructor r19.)
- **Monotone flank envelope as a machine law** -- FALSE, six violations with addresses (m29 span 25
  -> flank 30 beats span 21 -> 27, `k = 133,490,560`; m29 span 31 -> 22 beats span 29 -> 15; m19
  span 10 -> 21 beats span 8 -> 20; m23 span 29 -> 8 beats span 27 -> 7; m17 span 8 -> 14 beats span
  6 -> 12).  It holds WITHIN every step (19/19); the envelope follows OCCURRENCE COUNT, not span.
  (Mechanic R11/r17; Constructor X19.)
- **`FS <= F`** -- FALSE: `FS/F = 1.09` at 13->17 and 1.12 at 29->31; and round 14's
  `FS < F - q'/6` was a REQUIREMENT misread as a derived fact.  (Constructor X18/r15.)
- **Both-flanks-maximal exclusion as a route** -- correct and kernel-worthy (machine-free forbidden
  at 14 of 16 word-step pairs) but OFF-TARGET: the binding flank pairs are MID-SIZE, never maximal
  (largest single flank 0.16F to 0.81F, never F).  (Constructor X16/r16; Formalist verdict 2.)
- **The histogram / any unitary invariant as a supplier of the two-gap fact** -- the tight
  rearrangement bound is `F + G_2 = 2F` (maximal gaps mirror-paired), over budget from 19->23 on;
  since every unitary invariant of `N = BS` is a function of the gap histogram, this is a THEOREM
  that the invariant route dies.  (Constructor X35; Lateral item 39.)
- **Congruence-class potentials at any modulus** -- a potential that is a function of `k mod m` for a
  proper divisor `m` of the period certifies nothing: every class mod `m` contains a blocked column,
  so `h(k) >= h(k-1)+1` forces `0 >= m`.  (Constructor X32; Lateral item 40.)
- **Moment / PSD / Chebyshev routes** -- margins 67.6 .. 4.3e10 and GROWING; the L2 instrument's
  spread is essentially uncorrelated with `F` (spearman -0.038, +0.023, -0.186) and NEGATIVELY
  correlated at the largest machine.  (Lateral items 34, 77.)
- **The L1 character bound** -- provably BLIND to the teeth: `sum_m |Shat(m)|/P = prod_q S_q/q` does
  not depend on `v_q`, identical at all 30/180/1440 counterfactual tooth vectors while `F` spreads
  1.8-2.5x.  (Lateral item 76.)
- **Round-22's "#distinct eigenvalues = |Farey(F+1)| - 2"** -- assumed every gap length `1..F` is
  realised; HOLES break it (true counts 21/41/113/183/363/549/981/1813/2467 at m11..m41), with the
  exact loss rule `loss = sum phi(hole+1)`.  (Lateral r26 self-correction.)
- **`alpha_1/alpha_2 -> -1/phi` (the golden direction)** -- REFUTED: it is a CROSSING, not a limit
  (`-0.5778` at m37, `+0.0403` past `-1/phi` and still rising).  (Lateral r27 item 3.)
- **`arg H_5(1) = 126 deg` as a machine-independent constant** -- REFUTED twice: unattainable exactly
  (a parity obstruction) and, on exact cyclic data, a monotone DOWNWARD ladder
  `129.776 -> 125.659` at m13..m37.  (Mechanic r26; Lateral r25.)
- **Lateral item 29(a)'s machine-DFT formula `prod_q hat_q(m mod q)`** -- WRONG AS WRITTEN; the
  correct form is `prod_q hat_q(m c_q)` with `c_q = (P/q)^{-1} mod q`.  Everything item 29 concludes
  survives.  (Lateral r29 self-correction.)
- **Round-19's named target (the anti-correlation law as NECESSARY for (D))** -- OVER-SPECIFIED:
  independence alone clears every constrained case by 170x to 201,381x.  (Constructor r20.)
- **The exposure bound on `p_j` without the `1/rho` conditioning** -- "uncorrected, the bound appeared
  to clear the requirement everywhere tested; corrected, it does not."  (Constructor r20 sec 43.)
- **Round-27's "second frontier: a convergence frontier with no closed form"** -- the object DOES NOT
  EXIST at the named cell: the lifted LP's limit polytope is EMPTY there, so the decelerating loop
  was converging to a certificate.  "I had built a frontier out of a symptom."  (LP thread r28.)
- **E10, "the `k = 3` case split is tight on `F` at m41"** -- REFUTED: 32 of 92 decided cases carry
  EXACT in-polytope refutations; the error was reading "case 0 is already empty at 92" as a
  statement about the SPLIT.  (LP thread r29.)

### 8.5 The onset, the transfer, and the teeth

- **"The transfer superset is exact at span `<= 80`"** -- REFUTED: the inflation onset is at span 68,
  sharply (every reverse class of span `<= 67` realised, first refutation at 68).  (Mechanic r27.)
- **Three pre-registered closed forms for the onset** (`F_2` one machine back; `2F` two machines
  back; a constant ratio to `F`) -- ALL THREE FAIL out of sample; each had fitted the single
  round-27 data point.  Only the recursion form, intersected with the transfer's own emissions,
  survives.  (Mechanic r28, D1-D3.)
- **H1, "`F(M) mod q'` not in `{0,a,b}` is the teeth-sensitive separator"** -- KILLED: it holds 11/12
  against a base rate of `3 sum(1/q') = 1.291`, i.e. one observed against 1.29 expected, and it
  HOLDS at m31 while all three of m31's failing rows FAIL there.  (Constructor r29.)
- **`Pcong := F(M) mod q' in {0,A,B}` as a characterisation of the increment law's residual
  violators** -- REFUTED on the counterfactual family: sensitivity 34.0% / 5.6% at the two largest
  steps, 94.4% of residual violators not congruent to a legal letter, and the depth-3 attaining
  middle is the old record in 0.0% of 19->23 violators; the best predictor of that form reaches
  57.9% balanced accuracy.  (Lateral r29.)
- **Lateral's own lemma A0, "a depth-3 violator cannot have middle `s_min` since `g_L + g_R > F_2`
  is impossible"** -- FALSE: `g_L` and `g_R` are at LAG 2, not adjacent, so `F_2` does not bound
  their sum; 41-100% of depth-3 violators have middle exactly `s_min`.  The correct elementary
  statement is the peel bound.  (Lateral r29 self-correction.)
- **The three "mechanism" candidates for the low-`F` outlier** -- ALL REFUTED, all three in the same
  shape (the twin is a low-`F` outlier INSIDE the high-`F` class of the proposed variable): angular
  coherence (refuted IN THE SIGN -- the twin sits in the lowest-dispersion quartile, which has the
  HIGHEST mean `F`); "the teeth are the reciprocal of a small integer" (the `m = 1..60` sweep's
  median is exactly the family median, argmin at `m = 12`, not the twin's `m = 6`); and localisation
  in `(v_5, v_7)` (the top-variance gear is 7 or 11, never 5; no gear explains more than 9%).
  (`tooth-counterfactual-percentile.md` 5, 5A.3.)
- **"The twin machine's advantage shows in the budget slack `F(M+q') - F(M) - q'`"** -- NO: 59.0 /
  37.2 / 49.3 percentile at the three largest steps, undistinguished.  The favourable quantity is the
  INCREMENT LAW'S OWN MARGIN (66.8-83.3 percentile).  (Lateral r28.)
- **The real phase vector `+-u'` as EXTREMAL for something** -- REFUTED by exact full enumeration: it
  ranks 1716th of 11550 on overcount and 2536th on lone in the `{5,7,11}` two-teeth space.  "Special
  point of phase space" means only "the census is deterministic".  (Lateral r2.)
- **Tooth-sharing changing the survivor count** -- REFUTED: over full periods sharing changes nothing
  (`prod(q-2)` conservation), and both guaranteed wasted kills land on already-decided columns, so
  ZERO new openings.  (Lateral r1; attempts-map I.4.)
- **Umbrella nesting as a separate mechanism** -- REFUTED: any two gears' short umbrellas are
  concentric at joint shields; only the coinciding edges are twin-specific, and those are the `+-u'`
  pinned classes.  One mechanism total.  (Lateral r1.)
- **Hub enrichment at the binding loci** -- REFUTED: hub-rate/ambient 0.999 / 1.006; near-binding
  stretches are NOT hub-enriched, despite record stretches being bracketed by hubs.  (Lateral r5.)
- **Mirror-awareness at moment level** -- REFUTED by a two-line theorem: `k -> -k` swaps
  `omega_L/omega_R` and fixes `m`, so all mirror-augmented moments double and every ratio is
  invariant.  VACUOUS at any order.  (Constructor r6.)
- **Reflection symmetry in centred coordinates as a positional constraint** -- "the mirror of the
  survivor set is the survivor set, and the symmetry constrains nothing about position.  Closed."
  (`twin-prime-program.md` 14.)

### 8.6 The section and the window

- **Section pre-registration S4, "gear `p` is the death rung of 0.35-0.70 of its section"** --
  REFUTED as stated and EXPLAINED by the boundary: `p^2` is the excluded near edge, so the candidates
  are 1-3 columns each credited with probability `~1.7/ln p`, giving 75-80% zero.
  (`word-tree.md` 7.2.)
- **The pre-registered "1 to 3" for gear `p`'s own strikes in its section** -- REFUTED as
  NUMBER-strikes (up to six at wide rungs); "at most three" is about DEATH RUNGS.
  (`word-tree.md` 9.2.)
- **Section pre-registration V4, "the largest gear interacting with a new twin exceeds `p/2` for
  5-25%"** -- REFUTED: 46-48% at `q' >= 1000`; the 11.7% was a WINDOW AVERAGE dominated by the low
  part of the window where near-twin columns cannot exist.  Also refuted: "the fraction of new twins
  whose word is final by level 13" is 0.7-2%, not 5-25%.  (`word-tree.md` 8.2.)
- **Section pre-registration C, "the section cannot tell the real teeth from moved ones"** --
  REFUTED, and the refutation is the finding: moving gear 13's teeth to ANY `v` leaves 3.6-3.8% more
  survivors than the real teeth (gear 7's 6.4%), with the cofactor model `ln n / ln(n/s*)` matching.
  (`word-tree.md` 9.3.)
- **"Every section holds an aligned column IS the twin prime conjecture"** -- overstated: it is
  STRONGER (a dead section is a twin gap `>= 4 sqrt(x)`).  (`anchor-235.md` 10.)
- **"The real machine grows, untouched happens exactly once"** -- WITHDRAWN as overstepping the line
  of enquiry.  (`anchor-235.md` 10.)
- **The primorial-scale unwind always yields twins** -- REFUTED: nudge home 595 of the `{5,7,11,13}`
  machine is `(3569,3571)` with `3569 = 43 x 83`.  Openness beyond the horizon is not twinhood.
  (`class-tree.md`.)
- **Downward exclusion halting at the first gear with a coprime in the window** -- SUPERSEDED by the
  SQUARE GATE: it halts at the first `q` with `q^2 - 2` prime, because a gear's square is its first
  root kill.  (`class-tree.md`.)
- **Extending a joint umbrella rightward from the certifying set of its first column** -- BUG: it
  claims columns where the tower has activated a NEW gear inside the interval, one false twin per
  large window.  Fix: judge every column at its own graded depth.  (`class-tree.md`.)
- **The double-onset route as a contradiction** -- the required superdense bound is FALSE as a
  universal statement: 310 of 442 real windows have a twin-free onset prefix.  (Constructor r2.)
- **"Thinnest layer bands sit exactly at twin endpoints" as a machine obstruction** -- EXACT BUT
  TRIVIAL (`T = g(2p+g)/6`), and the "hostile thin band" reading is a DENSITY ARTEFACT.
  (Mechanic r10.)
- **"L* = 13 is a wall"** -- REFUTED: `L = 14` at `k = 46,133,660,494`, Poisson-consistent with the
  constellation model, no deficit against Hardy-Littlewood.  "Do not build bounds on 13."
  (Mechanic r7/r9.)
- **"Perfect alternation" of the record-run side word** -- CORRECTED: none of the six `L=13`
  instances is strictly L/R alternating (the landmark reads `RLLRRLLLLRLRL`); perfect alternation is
  LOAD-only, side words are blocky.  (Mechanic r7.)

### 8.7 The covering-bound line and the early routes

- **"Uniform adjacency is the only failure mode of the covering bound"** -- REFUTED: `{5,7,11}` has
  56 failures of which 55 are NOT uniformly adjacent; 511 of 512 for `{5,7,11,13}`.
  (`covering-bound-route.md` 13a.)
- **The gear-3 lemma ("gear 3 present implies the bound for ANY separations")** -- REFUTED by
  `{3,5,7,11,13,17}` with separations `(1,3,3,3,3,3)` at `L = 6`: `N(6) = 148485 > 147584.435`,
  ratio 1.006102, verified twice.  "The third overclaim in this area."
  (`covering-bound-route.md` 14a.)
- **Per-gear conditional marginals falling under conditioning** -- FALSE and badly: they RISE, by up
  to 63% above `2/q`, failing at 36 of 53 values of `L` at `y = 17`.  "Gear exhaustion is not the
  mechanism"; any proof must treat the gears jointly.  (`forbidden-configurations.md` 7.)
- **Weak negative association `h(L) >= prod (1 - marginal_q)`** -- fails narrowly: once at `y = 17`
  (`L = 2`, short by 2.6%), twice each at `y = 11` and 13.  (`forbidden-configurations.md` 7.)
- **Negative association / Harris / FKG as tools** -- the wrong tools: the escape indicators for one
  prime come from a CYCLIC SHIFT of a fixed pattern, and cyclic-shift families are not negatively
  associated in general.  (`covering-bound-route.md` 7.)
- **The universal hazard bound `h(L) >= 1/(F_h - L)`** -- TRUE but CIRCULAR: it presupposes `F_h`, so
  "an apparent complete proof for `{3,5,7}` built on it was hollow".  (`covering-bound-route.md`
  22b.)
- **The finite-automaton / transfer-matrix route over the gap word** -- closed for two independent
  reasons: the antidictionary is infinite, and recovering the counts needs the automaton WEIGHTED by
  the CRT measure, whose weights are the counts themselves.  (`forbidden-configurations.md` 5.)
- **Sub-multiplicativity `N(a+b) P <= N(a) N(b)`** -- FALSE: `{3,5,7}` at `a = b = 6` gives
  `N(12) P = 630 > N(6)^2 = 576`.  (`covering-bound-route.md` 10a.)
- **Size-only induction on the gear count** -- the hypothesis is false: if `S` lies inside two
  residue classes mod `q_n` the last gear covers it alone; the induction must be SPREAD-AWARE.
  (`covering-bound-route.md` 11b.)
- **Log-concavity of `N`** -- FALSE, fails at `L = 3`; and tail-fraction bounds from
  `N(L) <= N(1) - (L-1)G(L)` are too crude from `L = 6`.  (`covering-bound-route.md` 27.)
- **Monotonicity of the margin in the gear set** -- FALSE: `R_1` peaks at 1.705697 near `q = 83` then
  falls to 1.679767.  (`covering-bound-route.md` 24a.)
- **The collapsing margin** -- a NORMALISATION ARTEFACT: the exact integer differences were GROWING
  (51,555,900 at `y=19`; 350,759,640 at `y=23`).  (`covering-bound-route.md` 26b.)
- **The `3q` / `6q^2` reading of the tight block starts** -- refuted from both directions: at
  `y = 71` the predicted 51 and 57 rank 19th and 21st of 21, the LOOSEST.
  (`forbidden-configurations.md` 8a.)
- **"The per-J recipe does not scale"** -- ITSELF REFUTED: 2,548 contributing terms at `L = 39`
  against a predicted `5.5e11`.  "A scaling judgement made without building the thing cost a working
  route for most of the programme."  (`handover.md` method audit.)
- **`min_L h(L) = h(1)` as the target, and `h(1) = d/(1-d)` as the free base case** -- SUPERSEDED:
  the adjacent-frame `L = 1` is a grid artefact with no column-frame counterpart; what is needed is
  only `h(L) >= d`.  (`gear-recursion.md` 1.)
- **"kappa settles near 0.68"** -- SUPERSEDED: it drifts down to the exactly computable
  `kappa(2) -> 2 - (11/3)C = 0.54477`.  (`review-2026-08-17.md` 2.)
- **Pathways 7.2 and 7.3 held together** -- THEY CONTRADICT: form (b) implies
  `F_k(y) <= 2 + log P_k/(-log(1-d)) ~ y log^2 y / 2.1`, which the quadratic law
  `F ~ 0.68 y^2/log y` crosses at `y ~ 400`.  At most one of the two is true.
  (`review-2026-08-17.md` 4.)
- **Per-step multiplicative bound `r <= (q'/q)^2`** -- FALSE at 6 of 12 steps.  (Constructor r8.)
- **"`C <= 1.10`" for the increment constant** -- refuted by filling in `y = 37`, where `C = 1.354`;
  and PER-STEP increment bounds cannot deliver the constant at all, since the gear-37 step reaches
  `2.432 q`.  (`gear-recursion.md` 5.)
- **"`F_2` is attained at a maximal gap, and `F_2 - F` is bounded"** -- BOTH FALSE: at `y = 29`
  `F_2 = 55` comes from the adjacent pair `(30,25)`, and `F_2 - F` reads 2, 4, 5, 7, 6, 5, 12,
  doubling at `y = 29`.  (`handover.md` 6.14.)
- **Any argument resting only on "gaps are multiples of 3 and at least 3"** -- `{3,3,15}` satisfies
  both and violates the claim.  (`handover.md` 6.15.)
- **The two-scale bound** -- not valid, decisively: the rigorous version's side condition
  `u < 1/(2 F(2,z))` fails at every step and worsens with `z`.  (`twin-prime-program.md` 14c.)
- **Single-to-twin transfer `F(2,y) <= K F(1,y)`** -- `F_2/F_1` climbs 3.00 to 5.61 with no ceiling.
  (`twin-prime-program.md` 14e.)
- **The multiplicity / `omega` extension as the parity input** -- "the extension exists, and provably
  does not help": `lambda(m)` is not a function of `m mod P(y)` for any `y`.
  (`twin-prime-program.md` 14i.)
- **Low-order spectral truncation as a localisation shortcut** -- it localises perfectly, but the
  quantity thresholded is just "do all gears expose"; the spectrum contains no compression of the
  localisation information.  (`twin-prime-program.md` 35c.)
- **The `q = 3` pinning as a descent** -- self-similar but buys nothing further: pinning needs
  `q - 2 = 1`, which happens only at gear 3.  (`twin-prime-program.md` 13a.)
- **The flank alphabet `{1..5}` as a general fact** -- SCOPED DOWN to a first-flank fact only; the
  max flank part grows 7 -> 13 with `y`, and no corridor CAP on `F_2 - F` was found.
  (Lateral r9 -> r11.)

### 8.8 Errors caught by the kernel or by a gate (kept so the trap is not re-entered)

- The `F_2` encoding quantifying over ALL window starts rather than openings -- a REAL ERROR caught
  by `decide` (Python confirmed 1,296 counterexamples); the corrected statement requires the run to
  start at an OPENING.  (Formalist r11.)
- Modelling gear 3 as a run-breaker like gears 5 and 7 in the cap walk -- WRONG: gear 3 FILTERS the
  candidate list, and the wrong model gives caps 2/4 instead of 6/10/12.  (Formalist r16.)
- The linear-close defect -- every full-period census taken linearly over a CIRCLE was short by its
  seam structures; found by the mirror parity law on first use.  (Mechanic C26; Lateral 46(d).)
- "Tier C caps at machine 19" -- an ARTEFACT OF THE OLD ENCODING: restricting starts to openings
  takes machine 19 from 86 min to ~20 min and machine 23 from 33 h to ~7 h.  (Formalist r18/r19.)
- The mirror-canonical `o5` halving in the pruned record search -- UNSOUND combined with
  left-tautness (reflection maps left-taut to RIGHT-taut coverings); removed.  (Harvester r8.)
- "The compatible-word set is not a function of `q' mod 105`" (73/73 mismatches) -- own bug: letters
  are `q'`-sized VALUES while the claim is about RESIDUES.  Corrected, zero mismatches.
  (Harvester r13.)
- The "frame conflict" over the padded-link cost ("exactly `q'`" vs "at least `3q'`") -- NOT A
  CONFLICT: `q'` in column units, `3q'` halved, `6q'` in members.  (Harvester r12.)
- "26,366 of them" as a count of padded LINKS -- it is the SUPPLY (gaps of `M` equal to exactly
  `q'`); a link additionally needs its endpoint on a tooth, about 1,400 of them.  (Harvester r12.)
- "Tooth alternation FAILS for `3 | e`" -- the observation was real but the law it was tested against
  was the pre-padding one; a same-tooth adjacency is a legal PADDED link, not a violation.
  (Harvester r10 -> r11.)
- The rotation floor `ceil(6m/q)` (must be `(6m-1)//q`; the ceiling emitted `1000000001` as a twin);
  the spectral CRT component `k mod q` (must be `k t_q mod q` with `t_q = (P/q)^{-1} mod q` -- "it
  happened to agree at `k = 0, 143, 1001`, exactly where the twist is trivial"); and the inverse
  branch swap for `q = 1 mod 6`, which drops a gear and reports a false twin at `k = 10^12`.
  (`twin-prime-program.md` 27b, 35a, 21b.)
- The `L=6` inclusion-exclusion pruning rule discarding subsets whose first interior point is 3 -- a
  BUG (`{0,3,9}` holds no three points spaced 3 apart), producing negative gap counts and hazard
  -1.43.  (`covering-bound-route.md` 19a.)
- The `psi` factoriser bug -- `odd_prime_factors(6)` returned `[6]`, injecting a spurious
  `(6-2)/(6-4) = 2` into `psi(6)`.  (`forbidden-configurations.md` 8c.)
- The harvester's own prediction that Ziller-Morack Conjecture 6 would be breached at `y = 17` --
  REFUTED by its own computation in the same round (`h_2(17) = 192`, holds with 29.4% margin).
  Recorded rather than dropped.  (Harvester r16.)
- The flag that `gcd(e,105) = 15` and `105` are "exactly where a budget could fail" -- the
  computation says the OPPOSITE: those two classes have the SMALLEST increments of all `d` tested
  (max 0.632 and 0.483 against 1.235 for twins).  DENSITY WINS.  (Harvester r14.)
---

## 9. Gaps the record names

Only items the record itself flags as open, unbuilt or unstated.  Merged from the three harvests
and deduplicated.

### 9.1 The crux, in its current decomposition

1. **Is `L_pad(M)` bounded?**  This is now the WHOLE of requirement (B).  The bare half is closed
   (`L_bare <= PSORD <= 5`, kernel); gears 5 and 7 cannot supply the padded half past 53->59
   (CORRCAP infinite); no fixed-depth counter can supply it; the exposure half over-caps by 16-18.
   What is left is the COVER half at full depth on the NON-BARE words only.  Row
   `0,0,0,1,1,1,2,2,2,2,3,3` at m11..m53, and it grows.  (Constructor R105, U22.)
2. **Is `L(M) = A_kill - 1 = D_g - 1` bounded at all?**  Open; `L(47) = 4` is the current maximum and
   the row is non-monotone.  Formalist's honest boundary, written into `AnchorChain.lean`'s header:
   "the chain DEPTH `D_g` is not an algebraic consequence -- a run alternates freely, and `D_g` is a
   fact about lower gap SIZES."
3. **(A-pad): the `F_3`-wall event.**  `|eps| <= 4` is measured on LITERAL letters; the padded letter
   at m31 has `eps = -17`, and the closure of the whole chain rests on exactly this.  Named
   question: does the `F_3`-wall event recur, and when it does, does the increment law fail by
   exactly `F_3 - F_2 - s_min`?  (Constructor R91, R101/C6, r30/r31.)
4. **(D2), the depth-2 slack `S_2 = F + q' - F_2`,** measured 9..49 and growing roughly with `q'` --
   "no instrument on record supplies it".  It is R55's 2F wall: no rearrangement invariant and no
   congruence potential supplies it.  (Constructor R99.)
5. **Is the spectrum bound on `L` tight infinitely often?**  Tight at m11, m13, m29 and at 13-88% of
   counterfactual rows.  If `L = 2F/q' + 1 - o(1)` along the corpus then (B) is definitively false;
   if `L` stalls, a better bound exists.  Cheap test named: the full 23->29 family.  (Lateral U23.)
6. **The uniform order.**  Nothing says which `m` makes `A_m` nilpotent and tight; `A_relax` is
   non-monotone (`1,2,2,3,2,3,4,2,2` at m11..m41) and is a lossy proxy for `N(M)`, whose
   boundedness is open; the cover half is the only supplier.  (Constructor R67(i), R75, R100.)

### 9.2 Named constructs that would supply the missing upper bounds, not built

7. **The coverability spectrum `COV(M)`** -- CRT arithmetic on the gear set with no period scan,
   reaching machines 37/41/43/53, yielding the UPPER bounds on `F` and `F_j` that every prefix row
   lacks.  Proposed at the end of round 17/19, never built.  (Mechanic r17/r19.)
8. **The renewal factor** -- a closed-form lower bound on `P(no opening strictly between | both
   endpoints exposed)` at separation `v`.  "That single factor is the ENTIRE remaining gap between
   the rigorous exposure bound and sufficiency.  I did not build it."  (Constructor r20 sec 49.)
9. **The counted occurrence census of every realised legal word by CRT enumeration at m41..m47,**
   with the flank sum per column (`occ(q'; M)` as a list of addresses) -- NAMED, NOT BUILT; it is
   what makes `F(M+q')` scan-free at the short-word steps, and what would decide the three failing
   `Phi` rows at m31.  (Mechanic C54/r30; Constructor R96.)
10. **Is `Ghat` computable below a scan?**  The whole layered walk -- and therefore the anchor-235
    floor -- reduces to that ONE object, the gap-weighted opening transform
    `Ghat(m) = sum_o g(o) e(-mo/P)`; its mean-field part is closed form and carries 69-77% of the
    energy, and the residual is depth-1 adjacency, the same open term as the depth-sum identity's.
    (Lateral U17, item 75.)
11. **The phase-saturation ceiling of the PADDED alternation family** `(s, q'+(q'-s), s, ...)` -- the
    same pigeonhole on a different exposed set, therefore another closed form in the small gears with
    no solver.  NAMED, NOT BUILT.  (Mechanic r27.)
12. **`c_q(g1,g2)` at the padded lag, and the full gap correlation function** (the `(g+1)`-point
    object: two exposed ends plus `g-1` covered interiors) -- "bigger build, and where the complexity
    actually lives", explicitly not built.  (Lateral r18.)

### 9.3 Open questions the record states in words

13. **Why is `Phi(q')` twice as large relative to `F_2` at m31 as anywhere else?**  The counted
    padded-gap census would say; the flank order-statistic law is a LITERAL-letter law and inverting
    it at m31 gives an eight-orders-wide interval.  No teeth-arithmetic separator of the three open
    `Phi` rows was found -- none of `q'/F`, `q' mod 210`, litcap, `F mod q'`, `a/q'` orders the
    machines so that m31 is extreme.  (Constructor r29/r30.)
14. **Is there a statement "the padded letter's flank envelope obeys a DIFFERENT law", or is the
    padded letter simply where the literal analysis stops applying?**  Named, unanswered.
    (Constructor r29.)
15. **Why does a record-size gap of a CRT word have only small neighbours?**  Named as "the part with
    no teeth in it at all".  (`anchor-235.md` 9.)
16. **`spearman(L, n_0) = +0.311` beats both `a_min` and `min(n_a, n_b)`,** and `n_0` does not depend
    on `v_{q'}` at all -- so this is an unstated statement about the OLD machine's gap histogram at
    multiples of `q'`.  UNCLAIMED.  (Lateral U24.)
17. **The residue law of the gap histogram and its machine-independent phase** -- the richest classes
    are `+-s` of the SMALL gears (the small gears' letters are legible in the whole machine's gap
    histogram), and this is NOT the naive endpoint-survival prediction (`v = 2 mod 5` at 1.70 beats
    `v = 0` at 1.16).  UNEXPLAINED.  And the residue law does NOT predict the holes.
    (Mechanic C14, C17.5, r17.)
18. **Does the family-wide record law survive the ASYMMETRIC family** (teeth at arbitrary
    `{t_q, t'_q}`, not `+-v_q`)?  The attainment proof does not obviously use the mirror; the answer
    says which hypothesis the derivation may assume.  Cheap at m11/m13, NOT RUN.  (Lateral U18.)
19. **Is `d_0` the whole depth-2 story?**  The non-wrap depth-2 slack is positive (min 4-9) at every
    step; is `F_2 <= max(2 d_0, F + c)` for a small `c` on the family?  (Lateral U20.)
20. **The `{5,7}` admissibility cap at the corpus rungs** -- a one-page table of the rungs at which
    the twin alternation is `{5,7}`-admissible to length `k`, against `L = 1,1,1,2,1,3,3,2,2,2,4,3`.
    Named as the next step, UNCLAIMED.  (Lateral U21.)
21. **Prove the onset law, or find its first failure.**  The natural next test is 41->43, needing
    `D_4(43)` or a further extension of the m41 shard; the multiplicity residue (`onset/Y_5`, a
    near-constant factor 1.37-1.41) is the only part still unexplained.
    (`dictionary-monotonicity-onset.md` 5.)
22. **The mechanism of the low-`F` outlier is OPEN** with three candidates dead; the effect is an
    INTERACTION spread over the whole tooth vector, not a main effect of any gear.
    (`tooth-counterfactual-percentile.md` 5A.3.)
23. **Does partial level 3 (triple moments on selected gear triples) certify `L = 25` at m19, and
    WHICH triple is the obstruction?**  Named construct, not built.
    (`covering-hierarchy-exactness.md` 7.)
24. **A mod-4 lever** -- proved impossible from any SYMMETRY of the opening set (the group is exactly
    `Z/2`); the two remaining candidates (a free `Z/4` action on a subset of configurations, or a
    pairing not induced by a map of `Z_P`) are untouched.  (Lateral U10.)
25. **The first-moment transfer** -- the independence model already gets the two-gap law right with a
    polylog-versus-linear margin, so what is missing is not a sharper inequality but an unconditional
    TRANSFER of that first moment (a large-deviation bound on the machine's own covering system),
    which no rearrangement invariant and no congruence potential can perform.  That is Wall V with
    its scale named.  (Constructor R64, R67.)
26. **The query count / decision cost** -- the CEGAR query count is bounded by a machine-free
    `T_4 + T_2 <= F^4 + F^2`, but the RATIO to that cap is unbounded and the count is
    strategy-dependent; a single arity-1 realisability refutation grows about 15x per added gear.
    (Constructor R67(ii)(iii), R69.)
27. **The 41->43 and 43->47 rungs by the realisability chain** -- pinned to the ORACLE's information
    content, not to cost or strategy: the transfer superset stalls at 222 under three settings.
    (Constructor R79.)
28. **Whether `lim L0 = 32`** over all gears (the 32-cap) -- a finite check per gear set, monotone
    non-increasing, run only through gear 23.  (Lateral r8, B2.)
29. **Uniformity of the near-top word grammar** -- named as "the one open piece" of a
    machine-independent `alpha_1`; never closed.  And the mod-5005 / uniformity-in-`y` extension of
    the adjacency certificate: the tier-C residual GROWS (4 at `y=13` to 96 at `y=23`).
    (Lateral r10; Constructor r10.)
30. **The self-reference question, left live:** whether the self-blocked set at a level -- which is
    PRECISELY the previous level's twins -- CONSTRAINS that level's opening set, rather than merely
    being subtracted from it.  (`twin-prime-program.md` 17e.)
31. **Identifying a factor-restriction mechanism that is neither order-type nor congruence-type** --
    or showing the two types are exhaustive: the concrete open problem the constructor line left
    behind.  (`twin-prime-program.md` 30d.)

### 9.4 Formalisation gaps the record names

32. **The phase-reduction record law derived in the kernel from the walk** (a correctness proof of
    `chain_depth.py`'s walk against `Machine17.nextOp`), which would make the record law a single
    kernel theorem at 17 rather than two verified ends.  (`anchor-235-layer-laws.md` 5.)
33. **The depth-sum identity's kernel GLUE** (the `Finset` re-indexing between period window starts
    and residues) -- the only kernel step still missing there.  (`depth-sum-identity.md`.)
34. **The generator's soundness bridge above 11->13** -- the same lemma at two machines; done at
    m11/m13 by `Periodic.lean`, not at m17+ (machine 17's `ow` base case is a 19,305-step `decide`).
    (Formalist verdicts 11, 20; R31.3.)
35. **`Census29P` / `Census31P` are one-period claims and will not be kernel-checked** -- what round
    26 shrank is only their finiteness; removing them needs either the dictionary transfer from
    machine 23 or the sandwich lemma.  (Formalist verdicts 21, 25.)
36. **The (A) word-list ENUMERATION as a kernel object** -- currently computed, not checked; proposed
    as a target and not executed.  (Formalist r18.)
37. **`F_2(59) <= 173` carries a span condition** -- unconditional only as `>= 173`; the upper half
    needs `F(61)`, which the project does not own, and is deliberately NOT stated in the kernel.
    (Formalist verdict 51; Mechanic C48.)
38. **The mirror lever where it pays** -- instantiated at machine 11, where the direct bound is
    cheaper anyway; at machine 29 the base case `opSeq(N-1)` is not reachable by a walk and the lever
    needs a different route to it.  (Formalist R28.8.)
39. **Formalising the spectrum `F_j` / per-`J` runs** -- assessed and deliberately NOT built
    ("formalising `F_j` would formalise a route already known not to close (D)"); the reusable piece
    named is that replacing the literal 2 in `Machine17.pair25T` by `j` gives exactly
    `F_j(M) <= B`.  (Formalist r17.)
40. **The `n > 3` gear inclusion-exclusion** -- assessed and not forced (needs iterated
    `three_sets_ie` with `2^n`-way flattens, or mathlib's signed inclusion-exclusion over `Z`);
    nothing conceptually new, deferred.  (Harvester r6-r7.)
41. **The quantitative half of the `g = 2` uniqueness theorem** (depth `~P/(6g)` for `g > 2` and the
    mod-6 alignment rate) -- stays paper-side, PRICED, NOT FORMALISED.  (Harvester r2, r7.)
42. **The exponential-witness construction** -- stated but not machine-checked; it wants `ZMod q` as
    a field, and an attempt in raw `Nat.mod` arithmetic was REMOVED from `BlockedSlots.lean` rather
    than left broken.  (`twin-prime-program.md` 14b.)

### 9.5 Computations named and not run

43. **The exact m41 4-tuple census** -- complete at every span `<= 80`; the remaining paid population
    is 711,279 reverse classes, span 81-90 alone being 23 h at five workers.  (Mechanic C41.)
44. **`F(2,53)` / `F(2,59)`** -- named as the decisive measurement and never completed: standing at
    `>= 416` (review), `>= 420` (synthesis), `>= 426` (pruned run).  Every strategic choice in the
    growth-law question depends on it.  (`review-2026-08-17.md` 7a; Harvester r14.)
45. **The `y = 19` decision for Ziller-Morack Conjecture 6** -- honest status UNDECIDED; the full
    scan is `~1.2e13` operations, and a sharper attack would enumerate candidate delta-profiles
    directly by CRT.  (Harvester r17.)
46. **The `k_win >= 4` hunt at machines 31, 37, 41** -- running at filing, outcome not on record.
    (Mechanic r17.)
47. **The `L = 15` saturated-run hunt** -- never landed; last status 69.5%, members to `~1.4e12`, the
    `L = 14` record unbeaten.  (Mechanic r9.)
48. **The 31->37 site-residue histogram over the full 3.34e10 period** -- still running at write-up,
    no result reported later.  (Lateral r12.)
49. **The double-padding hunt in general** -- "plausibly COMPUTATIONALLY OUT OF RANGE rather than
    merely unobserved -- an honest limit, and a case where only a structural argument can decide."
    (Mechanic r16.)
50. **Extending the AP lemma to gear 7** ("do SIX openings in `q'`-AP become forbidden?") -- offered
    at round 16; no round reports it.  (Lateral r16.)
51. **The medium-medium adjacency question for `alpha_1`** -- a concrete next target, converted at
    round 10 into a word-level CRT check and never reported as closed.  (Lateral r9.)
52. **Late steps of the per-gap budget audit** -- NOT VERIFIED: steps beyond 23->29 for any `d`, and
    `gcd` classes 7, 21, 35 (`d = 14, 42, 70`).  (Harvester r14.)
53. **The 37->41 and 41->43 next-step padded tests** -- priced but not run; confirming that their
    winners are literal would show the padded tier is intermittent rather than growing.
    (Lateral r14.)
54. **The falsification criteria (a)-(e) for the `k_max` census**, including "the 31->37 census:
    literal `k = 5` or 6 there is CONSISTENT (cap-6 gear); `k = 7+` anywhere falsifies the absolute
    cap" -- filed, not resolved.  (Constructor r11.)
55. **The closed form for the multiplicity growth `S_pair/n2`** -- no closed form fitted (candidate:
    second moment of active-pair density); never returned to.  (Mechanic r4.)
56. **The layer-band descent** -- STOPPED after one page at limit event T1, and flagged as an
    UNEXAMINED MACHINE EVENT: the thinnest bands sit exactly at twin endpoints, "so far treated only
    as an obstacle; uninterrogated as a mechanism".  (Constructor r2-r3; attempts-map II.4.)
57. **`add_gear` iterating on its own** -- it does not iterate, because computing `F` two gears ahead
    needs the new gap WORD rather than its maximum; `add_gear` supplies the new histogram exactly but
    the ordering costs `A q` to materialise.  (`gear-recursion.md` 4a.)
58. **Reflection pairing and window-and-divisor lockstep** -- two routes named "worth trying next" in
    the twin-prime-program and never returned to under those names.  (`twin-prime-program.md` 9.)

### 9.6 Absences worth reporting

- **No matrix or transfer-matrix formulation of the machine exists** anywhere in the archive or the
  early design documents.  The nearest objects are the single-cycle walk reduction (the cap walk's
  state space `(position mod 105, parity)` is a single 210-cycle) and the transfer-matrix route over
  the gap word, which is recorded as REFUTED in both its forms.  The matrix frame first appears as a
  HUMAN DIRECTIVE for round 20.
- **No gear-order sorting rule exists.**  The nearest statements are the root ordering
  ("shadow `< q^2`, square, then coprimes"), the smallest-gear-first splitting recursion, the "large
  gears force nothing new" census, and the scan-start restriction to openings.
- The words CORRIDOR and LITERAL CAP do not appear in the early design documents at all -- they are
  proof-search vocabulary, introduced in rounds 8-11.
- The lateral archive file has NO ROUND 19; the constructor file has NO ROUND 7 (its round-7
  deliverable is `docs/proof-search/attempts-map.md`); the mechanic file ends at ROUND 17.

---

## Appendix

The full merged rule index -- every entry from the three harvests, deduplicated, in the schema
*Statement / Calculates / Status / Where / Limits* -- is a separate file:
**`docs/proof-search/alignment-rules-index.md`**.

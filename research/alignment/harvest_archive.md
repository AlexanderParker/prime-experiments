# Alignment of openings - harvest of the early record

Scope: `docs/proof-search/archive/` (verbatim logs of rounds 1-19: agents-shared, constructor,
formalist, harvester, lateral, mechanic) plus every early design document under `docs/` and the
repo root that predates the proof-search lanes (`twin-prime-program.md`, `covering-bound-route.md`,
`forbidden-configurations.md`, `gear-recursion.md`, `handover.md`, `status.md`,
`ideas-from-the-session.md`, `review-2026-08-17.md`, `synthesis-2026-08-18.md`,
`umbrellas-and-shields.md`, `pair-anatomy.md`, `class-tree.md`, `chain-conditions.md`,
`state-walk.md`, `gear-at-infinity.md`, `band-attribution.md`, `gap-without-lattice.md`,
`README.md`), and the round-7 attempts map.

Vocabulary used throughout (the human's): the machine {5..y} has one gear per prime; gears 2 and 3
are built into the columns (column k = the pair 6k-1, 6k+1). A gear's OPENINGS are the columns it
does not strike (g-2 of every g). The machine's openings are the columns every gear leaves open;
per period there are prod(g-2). The WINDOW is the certified range (columns with 6k+1 below the
square of the next prime). The SECTION is the new part of the window, p^2 < 6k+1 < q^2. A STRETCH
is a run of consecutive columns anywhere in the period. The RECORD F(M) is the longest stretch
with no opening. ALIGNMENT = the openings of the gears coincide at a column.

Where a source says "window" but means a run of columns, the entry says STRETCH and flags it.

---

## PART 1 - What a single gear does to the columns

### Tooth rule (teeth, shield, umbrellas)
- STATEMENT Per gear q >= 5 every residue of k mod q has exactly one role: two TEETH (the strikes) -
  the left-kill tooth at `k = u` where `u = 6^{-1} mod q` (q | 6k-1) and the right-kill tooth at
  `k = q-u` (q | 6k+1); one SHIELD at `k = 0 mod q` (q divides the midpoint 6k so it provably
  cannot divide either member); and q-3 openings in two runs, the SHORT UMBRELLA of length
  `2u'-1` with `u' = min(u, q-u)`, which has the shield at its exact centre, and the LONG UMBRELLA
  of length `q-2u'-1`. So a gear's openings are q-2 columns per period = shield + (q-3) umbrella
  columns.
- CALCULATES: every gear's opening set in closed form from one inverse, `u = 6^{-1} mod q`; the two
  umbrella lengths tend to q/3 and 2q/3 because the tooth separation is `3^{-1} mod q`.
- STATUS: exact (table given for gears 5,7,11,13,17,19; extended table to 47).
- WHERE: docs/umbrellas-and-shields.md, "The table" and "Why the teeth sum to q";
  twin-prime-program.md section 17c.
- LIMITS: single gear only; says nothing about where two gears' openings coincide.

### Teeth sum to q (the mirror)
- STATEMENT The teeth are `+-u`: the left tooth solves `6k = +1 mod q`, the right solves `6k = -1`,
  so right tooth `= q - u` and the two sum to q. One single symmetry `k -> -k` yields all of: teeth
  summing to q, the shield centred in the short umbrella, the whole opening pattern symmetric about
  column 0, and a mirror axis at half of every machine's period.
- CALCULATES: the second tooth from the first; the mirror partner of any opening (column c <-> P-c).
- STATUS: exact / theorem.
- WHERE: docs/umbrellas-and-shields.md, "Why the teeth sum to q".
- LIMITS: a symmetry of the pattern, not a localisation of any particular opening.

### Low tooth = u' = round(q/6) = the gear's own column (self-blocking law)
- STATEMENT The low tooth of gear q is `u' = round(q/6)`, which is exactly the column of the gear's
  own pair - every gear strikes the column that contains itself. Corollaries: q = 5 mod 6 puts the
  LEFT-kill tooth low, q = 1 mod 6 puts the RIGHT-kill tooth low; the tooth difference is
  `(2q -+ 1)/3` = long umbrella length + 1; the min tooth is u' and the max tooth's distance from
  the last residue is u'-1.
- CALCULATES: both teeth of any gear from q alone (`u' = round(q/6)`, family by `q mod 6`), hence
  the gear's whole opening set without any inverse computation.
- STATUS: exact (verified over the gear table 5..47).
- WHERE: docs/umbrellas-and-shields.md, "Three observed tooth patterns, and the one identity behind
  them"; twin-prime-program.md section 17c.
- LIMITS: the observed patterns (a) alternation and (b) consecutive odd differences BREAK at prime
  gaps of 6 (23->29, 31->37); only the u' identity survives.

### The u' column is the twin sequence (self-reference in one column)
- STATEMENT The list of low teeth u' runs 1,1,2,2,3,3,4,5,5,6,7,7,8 and a value appears TWICE
  exactly when column u' is a twin pair (both 6u' +- 1 prime): doubles at 1 (5,7), 2 (11,13),
  3 (17,19), 5 (29,31), 7 (41,43); singles at 4 (25 = 5^2), 6 (35 = 5*7), 8 (49 = 7^2). Twin gears
  share their low tooth - they spend their overlap on their own pair, early, inside the window.
- CALCULATES: which pairs of gears strike the same column at the bottom of the window (twin gears
  only), i.e. the one guaranteed place two gears' teeth align near column 0.
- STATUS: exact (gear list to 47).
- WHERE: docs/umbrellas-and-shields.md, "The extended table"; twin-prime-program.md section 17d.
- LIMITS: the sharing is a coincidence of TEETH, not of openings; it removes no opening the
  machine would otherwise have.

### Umbrella ratio locked at 1:2
- STATEMENT Two teeth cut a gear's circle into exactly two arcs; the arc holding residue 0 is
  always the SMALL one (it is the shield-centred arc with teeth at +-u' around it). Both arcs grow
  linearly with ratio locked 1:2 (short ~ q/3, long ~ 2q/3), the remaining 3 residues being the
  two teeth and the shield inside the short arc.
- CALCULATES: the length of the longest run of consecutive columns a SINGLE gear leaves open
  (2q/3 - roughly), for any gear.
- STATUS: exact (tabulated at gears 5, 11, 17, 23, 29, 41, 101, 1009).
- WHERE: docs/umbrellas-and-shields.md, "Umbrella growth".
- LIMITS: one gear at a time; large gears are almost pure opening, so a big gear alone never
  creates a long blocked stretch.

### The +-1 walk (no gear can outpace the 6-cycle)
- STATEMENT Every gear and every combination of gears walks the 6-cycle at exactly `+/-1` per
  rotation, never faster: successive multiples of q land in successive columns stepping by
  `q mod 6`, and every prime gear is 1 or 5 mod 6. The same holds for composite sub-machines since
  the units mod 6 are closed under multiplication.
- CALCULATES: the direction a gear walks the columns (gears = 1 mod 6 kill right-then-left, gears
  = 5 mod 6 kill left-then-right); the kills are spaced 4-then-2 rotations (or 2-then-4), which is
  what produces the long and short umbrellas.
- STATUS: exact, and measured for every sub-machine of up to four gears from 5 to 59 (periods mod 6
  always 1 or 5, never anything else).
- WHERE: docs/gear-at-infinity.md "What of this is now proved" (step 5); twin-prime-program.md
  sections 26a, 32a, 18a.
- LIMITS: a rate statement about presentation, not a statement about where the coincidences fall.

### Column 0 is open in every machine (complete alignment at 0)
- STATEMENT At column 0 every gear divides the midpoint, so every gear SHIELDS rather than strikes:
  column 0 is an opening in every gear set, always, because a gear's teeth sit at `+- 6^{-1} mod q`
  and `6^{-1}` is never 0.
- CALCULATES: one guaranteed opening of every machine, and the anchor of the mirror symmetry
  (the pattern is symmetric about column 0 as well as periodic).
- STATUS: exact / theorem.
- WHERE: docs/gear-at-infinity.md, steps 3 and 4.
- LIMITS: column 0 is a primorial multiple, i.e. as far from the window as possible - it is the
  alignment that is useless for the window.

### Gear 3 blocks one of any two adjacent columns; gear 5 blocks one of any three spaced 3 apart
- STATEMENT Gear 3 blocks one of any two adjacent positions, so every gap in the admissible pattern
  is a multiple of 3 and `F_h(y) = 0 mod 3`; gear 5 blocks one of any three positions spaced 3
  apart, so exposed runs have length at most 2 and the pattern is exactly ISOLATED POINTS AND
  DOMINOES.
- CALCULATES: the shape of the opening set at the finest scale (points and dominoes), and a mod-3
  constraint on the record.
- STATUS: proved; the mod-3 law checked against all thirteen known values of F_h.
- WHERE: docs/covering-bound-route.md section 18a; recorded in docs/gear-at-infinity.md.
- LIMITS: these are statements in the h-frame (adjacent frame) about the two fastest gears only.

---

## PART 2 - What two and three gears do jointly (where openings coincide)

### Pairwise coincidence is always exactly 4, at the four sign lifts
- STATEMENT For any two gears (q,r) the joint machine has period qr, exactly `(q-2)(r-2)` openings,
  and exactly 4 double-kill columns - the CRT lifts of the four sign choices `(+-u_q, +-u_r)`.
  Taxonomy: the two SAME-SIGN lifts are product blocks (both gears hit the same member, at the
  columns of `qr` and its mirror), the two MIXED-SIGN lifts are crossed kills (one member killed by
  each gear); for TWIN gears one crossed lift collapses onto the gears' own pair.
- CALCULATES: the exact positions of every column two gears jointly strike, by CRT, with no scan;
  and hence the opening count `(q-2)(r-2)` per period.
- STATUS: exact (worked in full for 5x7 period 35, 5x11 period 55, 5x13, 7x11 period 77, 7x13,
  11x13 period 143; "the 2^n law of the threat lattice at n = 2").
- WHERE: docs/pair-anatomy.md, "Taxonomy of the four double-kills" and "The pair table";
  twin-prime-program.md sections 31b, 32b, 28b/28c.
- LIMITS: counts are pair-independent theorems; PLACEMENT is slip arithmetic - which is exactly the
  programme's open question at pair level.

### Composite-gear lift: a pair's same-side kills ARE the composite gear qr
- STATEMENT Every single-gear law lifts verbatim to the composite gear qr: both-left tooth
  `= 6^{-1} mod qr` and both-right `=` its mirror, summing to qr; the family law extends by sign
  multiplication (`qr mod 6` = product of the factors' +-1; family 5 -> both-left low, family 1 ->
  both-right low); self-blocking extends (low tooth `= round(qr/6)` = the column whose pair contains
  qr); joint shield at 0. The genuinely NEW object is the crossed pair, positioned by the slip
  inverse `X_crossed = 1 + q((-2 q^{-1}) mod r)`.
- CALCULATES: the position of both product-block columns of any gear pair from `round(qr/6)`, and
  the crossed columns from the slip inverse - all in closed form.
- STATUS: exact (pair table 5x7 .. 11x13, all six pairs).
- WHERE: docs/pair-anatomy.md, "The pair table: single-gear laws lifted to composites";
  twin-prime-program.md section 28b.
- LIMITS: only the same-side (composite) half is a lift; the crossed cloud is the part that is not
  reducible to a single gear.

### The accumulation model: composite gear + a crossed cloud of 2^n - 2
- STATEMENT A machine of n gears = ONE composite gear (2 teeth, all classical laws lifted verbatim)
  plus a CROSSED CLOUD of `2^n - 2` mixed lifts, mirror-paired with sign complement; the cloud
  doubles per gear. All four triples from {5,7,11,13} have exactly `2^3 = 8` coincidence columns:
  2 all-same-side (the composite gear qrs, all-left at `6^{-1} mod P`, all-right at its mirror, low
  tooth `= round(P/6)`) plus 6 mixed lifts (three minority-gear flavours x two orientations),
  mirror-paired with complementary L/R signatures.
- CALCULATES: the complete coincidence set of any machine as a CRT sign lattice - the exact
  addresses of every column struck by two or more gears.
- STATUS: exact, verified to n = 3 (worked lists given: (5,7,11) P=385 all-R low at 64, mixed
  134/251, 141/244, 174/211; (5,7,13) P=455 all-L low at 76, mixed 41/414, 106/349, 141/314;
  (5,11,13) P=715 all-R low at 119, mixed 24/691, 141/574, 284/431; (7,11,13) P=1001 all-L low at
  167, mixed 141/860, 288/713, 405/596).
- WHERE: docs/pair-anatomy.md, "Triples: the 8-lift prediction confirmed"; twin-prime-program.md
  section 28c (square-roots-of-unity lattice).
- LIMITS: "its placement is the programme's single open question" - the model gives the cloud's
  cardinality and CRT addresses, not the cloud's distribution inside the window.

### Coincidence hubs: many-factor columns recur across machines
- STATEMENT Deep columns recur across machines: column 141 = (845,847) = (5*13^2, 7*11^2) carries
  teeth of 5, 7, 11 and 13 all at once, so EVERY triple drawn from them coincides there; likewise
  columns 24 and 596. Many-factor columns are the coincidence hubs.
- CALCULATES: which columns are struck by the most gears at once - the anchors that bracket long
  blocked stretches (see "maximal stretches are bracketed by hubs").
- STATUS: exact (worked at the four triples from {5,7,11,13}).
- WHERE: docs/pair-anatomy.md, "Triples"; docs/state-walk.md finding 1.
- LIMITS: descriptive; hub-rate at the binding loci was later measured as generic (see refuted list).

### The pair record falls, the machine record grows only by accumulation
- STATEMENT The record of a single PAIR falls as the pair grows: 5 (5x7) -> 4 (5x11, 7x11) -> 4
  (5x13, 7x13) -> 3 (11x13). But triples give 7, 7, 6, 6. "A lone pair of bigger gears covers less;
  gaps grow only through accumulation of gears, never through the size of one pair" - adding gears
  grows the record even as each gear individually weakens.
- CALCULATES: the direction of the record under gear addition, and that the record is a property of
  the SET, not of the largest gear.
- STATUS: exact (all six pairs and all four triples from {5,7,11,13}).
- WHERE: docs/pair-anatomy.md, "Max gap falls as the pair grows" and "Triples".
- LIMITS: n = 2 and n = 3 only.

### Two gears already generate every twin in their window
- STATEMENT The {5,7} window is columns <= 8 (6k+1 <= 49). The openings there are 2, 3, 5, 7 ->
  (11,13), (17,19), (29,31), (41,43) - EVERY twin in the window, generated by two gears' umbrellas.
- CALCULATES: the window content of the smallest machine directly from the joint opening set.
- STATUS: exact (machine {5,7}).
- WHERE: docs/pair-anatomy.md, "Certified twins from two gears".
- LIMITS: the smallest case; it is the base of the ladder, not evidence about larger machines.

### The 5x7 map, mirror-folded
- STATEMENT The 5x7 period-35 map has 15 openings = (5-2)(7-2), cyclic gap word
  `2,1,2,2,3,2,5,1,5,2,3,2,2,1,2`, record 5; folding at 17.5 the map matches itself - openings pair
  2<->33, 3<->32, 5<->30, 7<->28, 10<->25, 12<->23, 17<->18 and the double-kill columns pair
  1<->34, 6<->29. The half-period mirror at pair level.
- CALCULATES: half the period determines the other half; the record of {5,7} read off the gap word.
- STATUS: exact (machine {5,7}).
- WHERE: docs/pair-anatomy.md, "The 5 x 7 map (period 35)".
- LIMITS: mirror symmetry halves the search, never more (see "half-winding" note in class-tree).

---

## PART 3 - Adding a gear: what happens to the openings

### The turn law (adding a gear splits each class into q children, kills exactly 2)
- STATEMENT A node at level i is a residue class mod P_i - a description of a column under every
  umbrella so far. ADDING GEAR q SPLITS EACH NODE INTO q CHILDREN (the q phases the new gear can
  present, one per lap of the old machine); the turn law kills exactly 2 of them (the children
  landing on the new gear's teeth, positions closed-form from the slip); q-2 survive, the
  shield-child (= 0 mod q) among them.
- CALCULATES: the opening set of the machine {5..q} from that of {5..p} by CRT, class by class, and
  the count prod(q-2) directly.
- STATUS: proven (twin-prime-program.md section 17e turn law; nested classes section 1h).
- WHERE: docs/class-tree.md, "The class tree" bullets 1-3.
- LIMITS: it controls OPENNESS, not POSITION - see the sideways-step obstruction below.

### The tree is never extinct
- STATEMENT Every node has `>= q-2 >= 3` children, so `prod(q-2) >= 1` always: the machine's
  openings never vanish. The fully-shielded branch is `k = 0 mod everything` (column 0, primorial
  multiples); generic twins are all-umbrella branches.
- CALCULATES: the unconditional existence of openings in every machine, per period.
- STATUS: proven.
- WHERE: docs/class-tree.md, "The tree is never extinct".
- LIMITS: existence per PERIOD (~e^y), not per window (~y^2). This is the whole gap.

### The sideways-step obstruction (the exact form of the alignment question)
- STATEMENT Following open branches controls openness, not position: when a branch dies and the
  search steps to a sibling, the sibling class's smallest representative can jump by
  PRIMORIAL-SCALE amounts (the CRT-dial lesson - changing one level's residue moves the
  representative by that level's idempotent). The tree provably always has open branches, and one
  within F_k(y) of any point, but bounding the SIDEWAYS DISTANCE to the nearest open branch inside
  the window is Reduction A itself. "Every route in the programme is an attempt to bound the
  sideways step."
- CALCULATES: nothing directly; it names the quantity every alignment argument must bound.
- STATUS: exact statement of the open problem (equivalence-strength).
- WHERE: docs/class-tree.md, "The obstruction, stated in tree terms"; twin-prime-program.md 1h
  ("the tree's infinite paths are profinite integers; only the paths that stay small are twins").
- LIMITS: this is the target, not a tool.

### Sound prune (smallest-representative-first search is correct and complete)
- STATEMENT The level-i ancestor of column k is `k mod P_i <= k`, so discarding branches whose
  smallest representative exceeds the search bound never loses an answer within the bound.
  Smallest-representative-first search of the tree is correct and complete - it is the constructor,
  tree-shaped, and finds every twin.
- CALCULATES: a complete search for openings inside a bound, ordered by position.
- STATUS: proven (twin-prime-program.md section 17e).
- WHERE: docs/class-tree.md, "Sound prune".
- LIMITS: correctness of a search, not a bound on where the first opening lands.

### The chain condition (when adding a gear MERGES stretches)
- STATEMENT Adding gear q merges k+1 old stretches into one exactly when the k interior openings
  all lie in `{phi, phi + s} mod q, with s = 3^{-1} mod q` - the two teeth of the new gear at their
  true separation. The new record is then computable from the old gap word alone.
- CALCULATES: the record of the bigger machine from the smaller machine's gap word: find the runs of
  interior openings that sit on the new gear's two teeth, merge them. Verified exactly: predictions
  18, 25, 34 for F_k(17), F_k(19), F_k(23) against their true values.
- STATUS: exact, verified from an independent implementation in the k-frame (machines 13, 17, 19,
  23); gear-recursion.md section 4a.
- WHERE: docs/chain-conditions.md, "Level 2: what determines stride growth when a gear is added";
  research/slip_path.py `chain_prediction`.
- LIMITS: it is a per-step recipe requiring the old gap word; it does not by itself bound k.

### The frame trap (k-frame vs adjacent frame)
- STATEMENT A first version of the chain condition used the ADJACENT window `{phi, phi+1}`; its k=2
  count came out exactly `prod(q-4)` (the domino count) for every q, which exposed the error -
  dominoes fit an adjacent window trivially, but the k-frame teeth are NEVER adjacent (s = +-1 mod q
  would need q | 2 or q | 4).
- CALCULATES: a sanity check on any chain/merge count (if it equals prod(q-4) the frame is wrong).
- STATUS: recorded error, corrected; the handover's section 0.5 warning about confusing the two
  frames is cited as earned.
- WHERE: docs/chain-conditions.md, "The frame trap, recorded".
- LIMITS: a bookkeeping caution, not a law.

### Deletion spacing (q +- 1)/3 - the minimum distance between qualifying openings
- STATEMENT Qualifying interior distances are `= 0 or +-s (mod q)`, so the MINIMUM qualifying
  distance is `min(s, q-s) = (q +- 1)/3` - the k-frame deletion-spacing law (the adjacent-frame
  version, `>= q-1`, proved in gear-recursion.md section 4; divide by 3).
- CALCULATES: an absolute lower bound on how far apart two openings must be to be merged by the same
  new gear; and the SATURATION THRESHOLD - chains die entirely once `(q +- 1)/3 > F_k(M)`, visible
  as zeros in the census.
- STATUS: exact (census table at machines <=13, <=17, <=19 against new gears q = 17..43).
- WHERE: docs/chain-conditions.md, "The census (correct window)".
- LIMITS: it bounds k only when F_k(M) < q/2, and the regime that matters has F_k(M) >> q.

### The span law and what it cannot do
- STATEMENT Provable by pigeonhole on the two residue values: same-residue openings are `>= q`
  apart and alternating distance pairs sum to `>= q`, so a run of k openings has
  `span >= floor((k-1)/2) * q`. Combined with "consecutive openings <= F_k(M) apart" this bounds k
  only when `F_k(M) < q/2`.
- CALCULATES: an upper bound on chain length k from the record, in the saturated regime only.
- STATUS: proven.
- WHERE: docs/chain-conditions.md, "The span law, and what it cannot do"; corpus section 5.5.
- LIMITS: explicitly cannot bound k in the regime that matters; "gap structure alone cannot bound k".

### The maximal chains are the minimal alternation (s, q-s)
- STATEMENT All 62 k=3 runs at (gears<=19, q=23) have interior distance word `(s, q-s) = (8,15)` or
  its mirror `(15,8)` - residues `a -> a+s -> a`, span exactly q, 31 of each orientation. Maximal
  chains are the minimal alternation, nothing else occurs. A k=4 run would require the exact
  consecutive gap word `(s, q-s, s)`, span `q+s`, or a distance `= 0 mod q` (a single gap of exactly
  q) adjacent to the pattern.
- CALCULATES: exactly which gap words permit a deeper merge - an enumerable condition on the old
  machine's gap word.
- STATUS: exact, complete anatomy at (gears<=19, q=23).
- WHERE: docs/chain-conditions.md, "Anatomy of the maximal chains".
- LIMITS: at that size only; the fuel question was then pushed up (next entry).

### The fuel census: k=4 first exists at machine 29 with the word (10,21,10)
- STATEMENT At gears<=23 (7.95M openings, F_k = 34) the complete inventory of 3-gap words that
  would permit k=4 - all orientations, all 0-step variants, fifteen word shapes - is EMPTY: the fuel
  does not exist at all. At gears<=29 (214.7M openings, F_k = 43) the fuel appears: at q = 31 there
  are exactly FOUR qualifying triples, all the same word `(10, 21, 10)`, the minimal alternation
  `(-s, +s, -s)` with span `41 = q + 10`, forming two mirror pairs about the period midpoint (the
  four starting positions sum pairwise to exactly P). So k = 4 chains exist at (gears<=29, q=31) -
  the first k > 3 anywhere in the programme.
- CALCULATES: whether a deeper merge is possible at a given step, by looking up the required word in
  the current machine's gap word - "the lumpy increments of the gear recursion would then be
  readable in advance from the fuel table of the current gap word".
- STATUS: exact (machines <=23, <=29, <=31 - the last streamed over period 3.34e10).
- WHERE: docs/chain-conditions.md, "Addendum: the fuel census at gears<=23 and gears<=29, and the
  first k=4".
- LIMITS: "there is no universal bound k <= 3"; chain length grows with the machine.

### Fuel is sharply non-monotone in the new gear
- STATEMENT At gears<=31 the fuel spreads and is sharply non-monotone in q: k=3 fuel pairs
  70964 / 2 / 0 / 0 / 224 and k=4 fuel triples 216 / 0 / 0 / 0 / 0 at q = 37 / 41 / 43 / 47 / 53.
  Adjacency of the specific lift values is pure word-arithmetic - gap 16 next to 31 (the q=47
  requirement) never occurs, while 18 next to 35 (q=53) occurs 224 times.
- CALCULATES: a prediction of which steps of the gear recursion will be lumpy (the gear-37 anomaly
  = the fuel-rich case with 70,964 qualifying pairs; the tiny +9 increment at gear 41 matches its
  near-empty fuel, 2 pairs).
- STATUS: measured/exact at machine <=31 for q in {37,41,43,47,53}.
- WHERE: docs/chain-conditions.md, addendum.
- LIMITS: a hypothesis about the increment anomaly, stated as supported by the data, not proved.

---

## PART 4 - Where the openings are relative to the window

### Horizon / event-horizon theorem (gears strictly below y decide the interior)
- STATEMENT Any composite member strictly inside (y, y^2) has a prime factor `<= sqrt(M) < y`, so
  the top gear is never the root cause of an interior kill: gears STRICTLY BELOW y decide the open
  interior exactly. The top gear's whole unique contribution is the boundary - its self-pair at the
  bottom edge and its square at the horizon, which false-positives precisely when `y^2 - 2` is prime.
- CALCULATES: which gears you must consult to certify an opening inside the window; the exclusion
  works exactly once per window (the second gear's square lies strictly inside).
- STATUS: theorem, verified y = 13..79; later kernel-checked
  (`Horizon.exists_prime_factor_lt`, `Horizon.prime_of_no_prime_factor_lt`,
  `Horizon.twin_of_no_prime_factor_lt`, with the STRICT bound p < y).
- WHERE: docs/class-tree.md, "The event horizon and the layer law"; proofs/Horizon.lean
  (formalist round 1).
- LIMITS: exact only on the open interior; beyond y^2 an opening need not be a twin ("openness
  beyond the horizon is not twinhood" - nudge home 595 of the {5,7,11,13} machine is (3569,3571)
  with 3569 = 43*83).

### Layer law (the new gear's whole novel workload in the section)
- STATEMENT One layer = one prime retiring into the working set, horizon advancing `y^2 -> y'^2`.
  The newly activated gear's ENTIRE novel workload is (1) retro-closing the old horizon square
  `y^2` (owed iff `y^2-2` prime), and (2) the columns `y*c` for primes `c` in `(y, y'^2/y)` - one to
  three explicit numbers per layer (Bertrand: `y'^2/y < 4y`) - each owed iff its partner member is
  prime. Everything else in the fresh band is closed by the old gears.
- CALCULATES: exactly which columns of the SECTION the new gear is responsible for - a short
  explicit list of semiprime columns, enumerable in advance. "The tower's complexity lives in the
  number of layers, never inside one."
- STATUS: verified for the nine layers 13->17 .. 43->47 (seven of nine owe nothing in-band at all;
  the exceptions are 221 = 13*17 beside prime 223, and 437 = 19*23 beside prime 439); later
  kernel-checked as `Layer.layer_novelty`, `Layer.minFac_lt_or_eq`,
  `Layer.eq_mul_prime_of_minFac_eq`.
- WHERE: docs/class-tree.md, "Layer law"; proofs/Layer.lean (formalist round 2).
- LIMITS: it describes the new gear's kills in the section, not the position of the section's
  openings.

### Shadow law (a gear supplies nothing below its square)
- STATEMENT A gear supplies nothing below `q^2` - its ledger line opens at `q^2`. Territory sizes
  collapse upward: in the 13-window, gear 5 has 9 kills, gear 7 has 6, gear 11 has 2, gear 13 has 0
  interior kills (its square 169 sits at the horizon).
- CALCULATES: the first column at which a new gear can act; hence the section is where a new gear
  first matters at all.
- STATUS: kernel-checked (`Gear.sq_le_of_minFac_eq`, `Gear.R_eq_zero_of_below_sq`; guard: minFac 0
  = 2, so the law needs 1 < m); the territory map is exact for the 13-set.
- WHERE: docs/class-tree.md, "Territory map of the 13-set"; proofs/Gear.lean (formalist round 6).
- LIMITS: below q^2 a gear's strikes are always shadowed, so "full-set sieving is provably
  equivalent to graded inside a window".

### Composite root law (every squarefree gear product acts exactly once per window, at its own value)
- STATEMENT Every squarefree product of set gears acts UNSHADOWED exactly once per window - at its
  own value - if it fits at all. Same-member joint hits of a pair are `qr*j` with `j = +-1 mod 6`;
  `j = 5` already overflows the 13-window (5*35 = 175 > 169), so each pair scores exactly one
  product coincidence (35, 55, 65, 77, 91, 143 - six pairs, six products, no more), and no triple
  product fits (385 > 169), so zero same-member triples.
- CALCULATES: the complete list of multi-gear coincidence columns inside a given window, by
  enumerating squarefree gear products that fit.
- STATUS: verified (13-window worked in full); later kernel-checked as `same_census_once` ("exactly
  once if it fits", hypotheses explicit) and `same_left_own_value` (`qr = 5 mod 6` => the class rep
  IS column `(qr+1)/6`, member qr itself - "acts at its own value").
- WHERE: docs/class-tree.md, "The overlap map and the composite root law"; harvester round 3,
  proofs/Polignac.lean.
- LIMITS: same-member (product) coincidences only; the crossed cloud is separate.

### Graded depth: the subset needed depends on where in the window you look
- STATEMENT No subset covers the whole window - every gear's square is an in-window root kill on a
  candidate (`q^2 = 1 mod 6` always, so gear squares are always right members), so dropping gear q
  falsely opens column `(q^2-1)/6` whenever `q^2-2` is prime (drop 13 from the y=13 set and
  column 28 = (167,169) reports as a twin). BUT the window is GRADED: gears <= z are exact on
  columns whose members stay below `nextprime(z)^2`. Inside a window, a twin IS a column whose joint
  umbrella exists over the certifying set (graded depth: gears <= sqrt(member)).
- CALCULATES: for a given column, the exact gear set that certifies it; measured consequence for the
  first opening above y - depth needed 7, 11, 15, 20 at y = 41, 109, 197, 389, gears kept 2/11,
  3/27, 4/43, 6/75, "needed depth averages 0.42*sqrt(6y)".
- STATUS: exact (worked at y = 41..389; the 13-window fully worked).
- WHERE: docs/class-tree.md, "Sufficient sub-sets: which gears the window actually needs".
- LIMITS: "the first twin sits close above y" is an empirical input (within 169 of y for all
  y <= 3163); proving that closeness would be STRONGER than Reduction A.

### Square gate (downward exclusion stops at the first gear with q^2 - 2 prime)
- STATEMENT Descending exclusion of gears halts at the first q with `q^2 - 2` prime, because a
  gear's square is its first root kill (`q^2 < q*r`), met before any coprime. Governing sequence:
  primality of `q^2 - 2` down the gear list (prime at 5, 7, 13, 19, 29, 37, 43, 47; composite at 11,
  17, 23, 31, 41, 53, 59 in range). Gears with `q^2 - 2` prime are permanent floor-setters, each
  owning an eternal square pseudo-twin.
- CALCULATES: how many top gears may be dropped from a window's certifying set (exclusion depth =
  the run of consecutive composites from the top - one or two gears, except the 13-window's run down
  to {5,7}). Table: 13-window drops 13,11 stop at 7 with (47,49); 17/19 drops 17(,19) stop at 13
  with (167,169); 23/29 stop at 19 with (359,361); 31/37 stop at 29 with (839,841); 41/43 stop at 37
  with (1367,1369); 47..59 stop at 43/47 with (1847,1849)/(2207,2209).
- STATUS: exact computation (corrects the earlier "coprime stopper" hypothesis - see refuted list).
- WHERE: docs/class-tree.md, "Downward exclusion across windows: the square gate".
- LIMITS: a fact about which gears are needed, not about where openings sit.

### Necessity law (a gear is needed iff it owns a pseudo-twin in the window)
- STATEMENT Gear q is needed iff one of its root kills pairs with a PRIME partner in the window - a
  pseudo-twin like (209,211) = (11*19, prime) that only q can unmask. The minimal set is all gears
  minus the newest one or two, and droppability is TRANSIENT (11 returns at y=17 when the window
  reaches (209,211); 41 survives dropped through y=47).
- CALCULATES: the exact minimal certifying gear set of a window (verified exact for y = 13..59:
  y=13 {5,7,13}; y=17 {5,7,11,13}; y=23 {5..19}; y=31 {5..29}; y=41 {5..37}; y=47 {5..37,43,47};
  y=59 {5..47}).
- STATUS: exact, verified y = 13..59.
- WHERE: docs/class-tree.md, "The exact minimal subset"; research/minimal_subset.py.
- LIMITS: "q is necessary" = "q owns a lone-killer fragile column in the window" - a statement about
  the deciding kills, not about alignment of openings.

### Umbrella jumping (compute the next opening in closed form, no stepping)
- STATEMENT Inside a window, a twin IS a column whose joint umbrella exists over the certifying set;
  umbrella-jumping - next joint umbrella, read its interval in closed form, hop past - pinpointed
  all 55 twins of the 47-window, every one verified prime, with the six prime quadruplets of the
  range appearing automatically as width-2 umbrellas ((101..109), (191..199), (821..829),
  (1481..1489), (1871..1879), (2081..2089)) - the points-and-dominoes law operating live.
- CALCULATES: THE NEXT OPENING from any position without stepping; each gear's distance to its next
  tooth is `min((u_q - m) mod q, (-u_q - m) mod q)` - "read the gear's phase" (closed-form next-twin
  method, verified to k = 10^16).
- STATUS: exact; scaled to y = 100003 (1.67e9 columns, 27,412,929 twins, 180,504 quadruplets).
- WHERE: docs/class-tree.md, "Pinpointing twins in the umbrella stack"; research/umbrella_tools.py,
  research/jump_distance.py; docs/gear-at-infinity.md.
- LIMITS: computes the next opening, does not BOUND the distance to it - "the gap is not closed
  form, and equivalent to the open problem" (docs/gap-without-lattice.md).

### The stack certificate: twins sit in needle's eyes
- STATEMENT Each twin carries a stack certificate - per-gear rooms, minima = the binding gears.
  Column 23 = (137,139): gear 5 room right 0, gear 7 room left 0, gear 11 room right 0 - the twin
  pinched to width 1 from three directions. "Twins sit in needle's eyes, and the certificate names
  the needles."
- CALCULATES: for any opening, which gears are the tight constraints on either side.
- STATUS: exact (47-window worked).
- WHERE: docs/class-tree.md, "Pinpointing twins in the umbrella stack".
- LIMITS: descriptive per opening.

### A bug worth keeping: judge every column at its own graded depth
- STATEMENT Computing a joint umbrella with the certifying set of its FIRST column and extending
  rightward claims columns where the tower has activated a NEW gear inside the interval (a square
  crossing mid-umbrella) - one false twin per large window, exposed by full verification. The fix
  judges every column at its own graded depth; "the failure mode is the horizon law in miniature".
- CALCULATES: the correct depth rule for reading openings across a section boundary.
- STATUS: recorded and fixed.
- WHERE: docs/class-tree.md, "Bug caught and fixed en route".
- LIMITS: an implementation law, but it is exactly the section boundary effect.

---

## PART 5 - The record, and the structure of a maximal blocked stretch

### The stretch is the mex of the gears' tooth schedules
- STATEMENT The stride (distance to the next joint opening) from any open column is the MEX of the
  union of the gears' tooth schedules - the walk ends at the first forward offset no tooth reaches.
  The certificate is complete and per-column; e.g. the maximal stretch of gears {5,7,11,13} (11
  columns, from column 122) with every interior column's blocker:
  `123:[11] 124:[5] 125:[7] 126:[5] 127:[7] 128:[13] 129:[5] 130:[11] 131:[5] 132:[7,13]`.
- CALCULATES: the exact distance to the next opening, and the per-column attribution of a blocked
  stretch, with no stepping (`mex_jump` reproduces the walk).
- STATUS: exact (verified to agree with the walk); research/slip_path.py `state_walk`, `mex_jump`.
- WHERE: docs/chain-conditions.md, "Level 1"; docs/state-walk.md.
- LIMITS: per-position; gives the next opening but no bound on it.

### The one-kill lemma (supply cap inside a stretch)
- STATEMENT Gear q contributes at most `2(floor(L/q)+1)` teeth to a stretch of L; in the maximal
  stretch of {5,7,11,13} gears 11 and 13 sit exactly at their ceilings. Long blocked stretches
  require near-maximal efficiency from EVERY gear.
- CALCULATES: an upper bound on how much of a stretch a given gear can block, hence a capacity
  condition on the record.
- STATUS: exact; kernel-checked in R-form as `Gear.R_prefix_le` (R q (members (range t)) <= 6t/q+2).
- WHERE: docs/chain-conditions.md, "Level 1"; corpus section 37b; attempts-map I.1(c).
- LIMITS: summed over gears the capacity is abundant from y = 13 on (sum 2/q >= 1), so the cap alone
  never bounds the record.

### Maximal stretches are bracketed by coincidence hubs
- STATEMENT Entry and exit of a maximal blocked stretch are TRIPLE kills, and the mid-run column is
  a deep hub. Worked end to end for the machine {5..19} (25 columns, open column 110 to 135):
  entry 111 (665,667) [5L,7L,19L]; mid 119 (713,715) [5R,11R,13R] with 715 = 5*11*13; exit
  134 (803,805) [5R,7R,11L]; then 135 (809,811) is ALL UMBRELLAS -> the twins (809,811).
- CALCULATES: the shape of a record stretch - "long strides anchor on many-factor columns".
- STATUS: exact (machine {5..19}, full walk recorded).
- WHERE: docs/state-walk.md, "The maximal stride of the y=19 machine, walked end to end", finding 1.
- LIMITS: one machine; the later census found hub-rate at binding loci generic (see refuted list).

### Most of a blocked stretch is single-point failure
- STATEMENT 18 of the 24 blocked columns in the {5..19} maximal stretch die to EXACTLY ONE gear: the
  stretch is a chain of fragile links held together by two heavy anchors - the same shape the
  chain-condition/fuel analysis found from the gap-word side.
- CALCULATES: how fragile a record stretch is - which columns would become openings if one gear were
  removed or shifted.
- STATUS: exact (machine {5..19}).
- WHERE: docs/state-walk.md, finding 4.
- LIMITS: one machine.

### Shields appear uselessly all through a blocked stretch
- STATEMENT Twelve of the 24 blocked columns in the {5..19} maximal stretch have some gear SHIELDING
  while another kills - a shield protects its pair from one gear only. "The picture of why isolated
  shields never make twins and only full umbrella stacks do."
- CALCULATES: nothing to compute; it rules out shield-counting as a route to openings.
- STATUS: exact (machine {5..19}).
- WHERE: docs/state-walk.md, finding 3.
- LIMITS: a negative structural fact.

### The smallest gear carries its ceiling load
- STATEMENT In the {5..19} maximal stretch gear 5 kills 9 of the 24 columns - exactly its 2-per-5
  rhythm, the extremal-efficiency law visible line by line.
- CALCULATES: confirms the one-kill lemma is TIGHT for the smallest gear inside a record stretch.
- STATUS: exact (machine {5..19}).
- WHERE: docs/state-walk.md, finding 2; corpus section 37b.
- LIMITS: one machine.

### The record fits the window with a factor 2.3-3, and the ratio is falling
- STATEMENT The whole content of the remaining gap is `F_k(y) <= (y^2 - y)/6` - the record of the
  admissible pattern, in column units, fitting inside the validity window. It holds with a factor of
  2.3 to 3 for every gear set measured, and the ratio is falling.
- CALCULATES: the numeric margin of Reduction A at every computed machine.
- STATUS: measured (every gear set measured up to the corpus limit); docs/gear-recursion.md
  section 2.
- WHERE: docs/gear-at-infinity.md, "Where the argument does not close".
- LIMITS: measured margin, not a bound.

### The record grows polylogarithmically while the window grows quadratically
- STATEMENT The maximal blocked stretch tracks `~ 0.45-0.49 * log^3(member)/6` columns across the
  whole measured range, and stretch/window collapses from 2.1e-2 (y=101) to 6.0e-7 (y=100003,
  members to 1e10) - five orders of magnitude across three orders of y. "The maximal twin-free
  stretch grows polylogarithmically while the horizon grows quadratically; the widening gap between
  those two growth rates is Reduction A's slack."
- CALCULATES: the measured slack of Reduction A at any scale.
- STATUS: measured (y = 101 .. 100003; 27.4M twins generated and verified).
- WHERE: docs/class-tree.md, "Stride/window collapse to y = 100003"; SUMMARY carry-over facts.
- LIMITS: a trend, explicitly NOT a wall and not a proof.

### Two extremal families: load-extremal vs length-extremal
- STATEMENT Frontier (load-extremal) runs and chain/fuel maximal stretches are DIFFERENT extremal
  families. Load-extremal: short, ABSOLUTE (fixed integer landmarks), prime-dense,
  constellation-governed. Length-extremal: deep, load ~0.3, gap-word-governed. They merge only at
  L = maxstride, so chain analysis cannot see the binding region L ~ 14-32.
- CALCULATES: which instrument applies to which regime.
- STATUS: measured (y = 1009 / 3163 / 10007; the L* = 13 landmark at columns 2452-2464, primes
  14713..14783, identical at every scale).
- WHERE: lateral round 6 (SUMMARY); attempts-map section 5.
- LIMITS: says the chain machinery is the wrong tool for the load frontier, and vice versa.

### An AP of openings has difference divisible by every gear below it
- STATEMENT An arithmetic progression of L openings has difference divisible by every gear q < L+2:
  3 equal gaps need `5|g`, 5 need `35|g`, and `L >= y+2` needs `>= P(y)`.
- CALCULATES: an exclusion on regularly-spaced openings - the longest equal-gap run is 3-4 with
  g = 5 exactly every time.
- STATUS: full-period verified on machines 13-29, zero violations.
- WHERE: lateral (SUMMARY, "SIDE THEOREM", round 19).
- LIMITS: about equal-gap runs only.

---

## PART 6 - Phases, pins and CRT addresses of coincidences

### Roots-of-unity law (a column is doubly struck iff 6k is a nontrivial root of 1)
- STATEMENT Column k is hit by gear pair {q,q'} iff `36k^2 = 1 mod qq'`; the trivial roots `+-1` are
  same-member (semiprime-multiple) columns and the nontrivial roots `+-r` (with
  `r = CRT(+1 mod q, -1 mod q')`) are cross-member. So a column is DOUBLE iff 6k lands on a
  nontrivial root of unity mod some active semiprime. Twin-pair gears recover `r = p+1`.
- CALCULATES: the double-struck columns of a window by semiprime arithmetic only, with no primality
  tests and zero freedom - "doubles are one fixed subset of N".
- STATUS: exact, verified both directions on the full y=47 window (constructor round 2).
- WHERE: archive/agents-shared r2; research/double_onset.py.
- LIMITS: it addresses double kills, i.e. the complement of openings, pairwise.

### The gap-graded split law (closed-form address of a crossed coincidence)
- STATEMENT The split class of pair `(q, q' = q+g)` - q kills left, q' kills right - has least
  representative `x = (q'(b0 + i*q) - 1)/6` with `m0 = (-2 q^{-1}) mod g`, `b0 = (2 + m0*q)/g`,
  `i = (q'-b0)*q^{-1} mod 6`; the other class is `P - x` with `P = qq'`. Depth `x ~ P(m0/g + i)/6`.
- CALCULATES: the exact position of a crossed coincidence for any gear pair, in closed form - the
  roots-of-unity law made explicit.
- STATUS: exact, verified against brute CRT for all 2850 prime pairs q < q' <= 400, zero fails.
- WHERE: archive lateral round 3; research/split_gap_law.py.
- LIMITS: pairwise only; the multi-gear (CORR) terms need the master formula.

### g = 2 is the unique gap that pins at the bottom of every window
- STATEMENT `m0 = 0` iff `g = 2`, so g=2 is the UNIQUE gap with `b0 = 1` identically: its split pins
  at `x = u' <= K` in every window at every scale. Other gaps have floor depth `~P/(6g)`, reached
  only when the mod-6 alignment `i=0` lands. Twins below y are therefore the unique gap class with
  unconditionally guaranteed contribution to the level-y^2 doubles ledger; everything else is
  residue-alignment-conditional.
- CALCULATES: which pairs of gears are guaranteed to coincide INSIDE the window (only twin gears),
  and where (at column u' <= (y+1)/6, i.e. in the bottom band).
- STATUS: exact; kernel-checked as `twin_pin` (the pair IS column u = (p+1)/6), `twin_pin_le`
  (u <= (y+1)/6 for EVERY y >= p), `twin_split_class_iff` (column k split-killed by {p,p+2} IFF
  k = u mod p(p+2)), `twin_mirror_slot` (second class at P-u), `twin_product_slot` (same-member
  double at u(p+1), member = p(p+2) exactly), and `own_slot_pin_gap_two` (UNIQUENESS: an odd prime
  pair (q,q+g) split-killing the column holding q itself forces g = 2).
- WHERE: archive lateral round 3; harvester round 2, proofs/Polignac.lean;
  research/twin_pin_check.py (81 twin pairs to 3000; uniqueness scan over all prime pairs
  q < q' <= 400 found 20 own-column pins, ALL g=2).
- LIMITS: it pins a DOUBLE KILL, not an opening; and it is the machine's blind spot -
  `twin_pin_self_block` records that the pin column u of twin (p,p+2) has zero composite members
  yet is never a survivor of any machine with bound >= p ("the machine is blind to its own pair").

### Tooth-sharing classes of a twin pair
- STATEMENT A twin pair's four within-pair double-kill CRT classes mod `P = p(p+2)` are exactly
  `{+-u', +-u'(p+1)}`. The `+-u'` columns are split kills (own pair + mirror); the mixed class IS
  the twin-product column: `6*u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2)`. So each level-sqrt(N) twin pair
  marks the level-N window at exactly 2 deterministic columns: its own column and its product
  column. Generalisation: for ANY two real gears the cross classes are pinned at the semiprime qq'
  columns.
- CALCULATES: the deterministic marks a twin pair below y leaves on the window above.
- STATUS: exact, verified 60/60 twin pairs to 2000.
- WHERE: archive lateral round 1; research/tooth_sharing.py.
- LIMITS: both guaranteed wasted kills land on already-decided columns, so this creates zero new
  openings (see refuted list).

### Redistribution law: sharing is positional, never cardinal
- STATEMENT Sharing a pair's tooth phase changes expected in-window wasted kills by `1 - 2R/P` with
  `R = K mod P`; over FULL PERIODS sharing changes nothing (`prod(q-2)` conservation) - the
  mechanism is purely POSITIONAL, never cardinal.
- CALCULATES: the in-window effect of a phase change, with the full-period effect proved zero.
- STATUS: exact / tested (400 draws per config, matches per pair including sign flips).
- WHERE: archive lateral round 1; attempts-map I.4.
- LIMITS: it is the reason no counting argument based on phase can work; the content is entirely in
  placement.

### Exposed-set autocorrelation: the three cases are the three tooth-relationships
- STATEMENT For gear q with exposed set `A_q = Z_q` minus its two teeth,
  `c_q(g) = |{r in A_q : r+g in A_q}|` has the closed form: `c_q(g) = q-2` if `q | g` (same tooth),
  `q-3` if `g = +-2u_q` (opposite teeth), `q-4` otherwise. THE THREE CASES OF THE AUTOCORRELATION
  ARE THE THREE TOOTH-RELATIONSHIPS.
- CALCULATES: how many phases of gear q keep BOTH ends of a stretch of length g open - i.e. the
  count of alignments of two openings at a given separation, for one gear; and by CRT the admissible
  endpoint phases mod 35 = exactly `c_5(g)*c_7(g)` in {3..15}.
- STATUS: exact closed form, brute-force verified over gears 5-31 at ALL lags, zero mismatches.
- WHERE: archive lateral round 19 (SUMMARY).
- LIMITS: ENDPOINT exposure is a CONJUNCTION so it factorises by CRT; the INTERIOR condition is a
  DISJUNCTION and does NOT factorise - "that is why it stopped there".
- NOTE: `g = +-2u_q` is exactly the literal-link lag that rounds 12-17 spent on padding, reached
  here from an unrelated direction.

### The pinning law (the neighbourhood word pins the address to <= 4 offsets)
- STATEMENT The neighbourhood word pins the mod-385 address to `<= 4` offsets, UNIFORMLY in y
  (206/206 words, five machines; gear 5 unique always). #top-stratum classes `<= 4 x #words`;
  observed 6-14 classes, flat, while gap counts swing 20-106. The address is LOCAL:
  `address = pin(word)`, not inherited.
- CALCULATES: the possible CRT addresses of a near-record stretch from its gap word alone.
- STATUS: measured/exact over 206 words across five machines.
- WHERE: archive lateral round 10 (SUMMARY, "THE PINNING LAW").
- LIMITS: the drift recursion was REFUTED (reachability 18/20 -> 0/4) - addresses are not inherited
  from the smaller machine.

### Corridor laws (kernel-checked constraints on where a record stretch can sit mod 35)
- STATEMENT `endpoint_law` and `endpoint_law_34` (G = 34 mod 35 forces a mod 35 in {3,18,33});
  `adjacency_law` with `forbidden_pairs_count = 294` (full 35x35 table by `decide` + kernel, no
  `native_decide`); `no_chain_of_forbidden`; `n2_packing` (W/33 <= n2).
- CALCULATES: which mod-35 addresses a stretch of a given length can occupy - the corridor.
- STATUS: kernel-checked (proofs/Corridor.lean, formalist round 10).
- WHERE: archive/agents-shared r10 formal ledger.
- LIMITS: mod-35 only (gears 5 and 7); scaling needs mod-385 / mod-5005.

### Onset gate (a gear that divides a gap must be at most the record)
- STATEMENT `onset_gate : (0 < g) -> (q | g) -> (g <= F) -> q <= F` - one line, `[propext]` only.
  Restated consequence: `F < q` is precisely the NO-PADDING regime, not "the onset condition" as
  an earlier heading had it.
- CALCULATES: when the new gear can appear as a letter (a gap of exactly q') at all.
- STATUS: kernel-checked (formalist round 19).
- WHERE: archive/agents-shared r19 AUDIT (C).
- LIMITS: the padding COUNT bound is `p <= F/q + 5/6`, which GROWS;
  `padding_three_not_excluded` records that `F >= (13/6)q` stops excluding three links.

---

## PART 7 - The route and the spectrum

### The window identity (openings and twins coincide exactly on (y, y^2])
- STATEMENT The admissible pattern and the twins coincide exactly on `(y, y^2]`:
  `survivors(y, K) = T(6K+1) - T(y)`, verified exactly at y = 11 through 1009. Outside it they
  diverge, and that divergence is where the argument needs the missing bound.
- CALCULATES: the certified translation from openings to twins, i.e. what the WINDOW means.
- STATUS: exact (y = 11..1009); kernel-checked as `BlockedSlots.survivor_iff_twin` (q <= y) and
  sharper `Horizon.twin_of_no_prime_factor_lt` (strict p < y).
- WHERE: twin-prime-program.md section 17d; docs/gear-at-infinity.md; proofs/BlockedSlots.lean.
- LIMITS: it is the definition of the target, not a step toward it.

### Reduction A in umbrella language
- STATEMENT "A twin column is a column standing under every relevant gear's umbrella at once
  (shield-centre or plain miss). The umbrellas provably overlap somewhere in every period (the
  ALIGNMENT LAW: the smallest gear's long umbrella survives intact at some phase, whatever the other
  gears do). The single open question of the programme is WHERE: whether an all-umbrella column
  always occurs inside the certification window `6k+1 <= y^2`."
- CALCULATES: nothing; it is the exact statement of the alignment question in the human's own
  vocabulary, and the one place the archive uses the phrase "the alignment law".
- STATUS: the alignment law (overlap somewhere in every period) is proven; the localisation is
  Reduction A, open, kernel-checked equivalent to the conjecture.
- WHERE: docs/umbrellas-and-shields.md, "The machine statement in this vocabulary" (worked example:
  column 12 = (71,73): mod 5 -> 2 long, mod 7 -> 5 long, mod 11 -> 1 short, mod 13 -> 12 short -
  covered everywhere).
- LIMITS: existence in a period of size ~e^y vs a window of size ~y^2 - "localisation, not
  existence" (docs/gear-at-infinity.md).

### The spectrum F_j and spectrum flatness
- STATEMENT With `F_j` = the max sum of j consecutive gaps (the gap spectrum), rigorously
  `excess <= F_{k_max+1} - F_2`, and lemma 1 is the first spectrum increment. The whole tolerance
  hypothesis = SPECTRUM FLATNESS (increments are q/3-scale, not F-scale) + FUEL BOUND
  (`k_max = o(ln y)` suffices; measured `k_max <= 3` everywhere, 62 k=3 chains matching the corpus
  fuel census exactly).
- CALCULATES: the record of the bigger machine from the smaller one's spectrum, via the merge
  bookkeeping.
- STATUS: exact reduction (constructor round 10); flatness measured, later REPAIRED (see next).
- WHERE: archive/agents-shared r10 "THE UNIFICATION".
- LIMITS: raw flatness FAILS at 5 of 15 machine-depth pairs (round 17 refutation).

### The suppression law and suppression-corrected flatness
- STATEMENT `suppression(j) = F_j - qualmax_j ~ lambda*(j-2)*ln(1/p_1)`, with lambda and p_1
  computed from M alone. Payoff: (D) follows from `F_j - F <= q' + lambda*(j-2)*L` for every j. ALL
  15 MACHINE-DEPTH PAIRS HOLD (corrected values 4.7-15.1, bounded and non-growing) where raw
  flatness fails at 5 of 15.
- CALCULATES: a corrected bound on the spectrum increments of the bigger machine from the smaller
  machine's own quantities.
- STATUS: measured/derived (round 19; observed suppression 7, 15, 30 vs predicted 9.0, 21.7, 42.5 -
  right scale, conservative at depth).
- WHERE: archive/agents-shared r19 SUMMARY (constructor).
- LIMITS: it repairs round 17's refutation of raw flatness; the underlying anti-correlation law
  (the formula for p_j) was still open at the end of round 19.

### The qualifying spectrum Q_j closes every measured step, word-free
- STATEMENT At 29->31 `F_5 = F+42` fails but `Q_5 = F+28` passes - the size threshold alone takes 42
  to 28. The same object gives the fuel cap free: `Q_j = 0` iff no word that long exists.
- CALCULATES: the per-step ceiling on the record increment without any word enumeration.
- STATUS: measured, closes every measured step (machines up to 31).
- WHERE: archive/agents-shared r19 SUMMARY (mechanic).
- LIMITS: HONEST COUNTERWEIGHT recorded in the same entry - the margin collapses from ~0.45q' to
  0.10-0.11q' at machines 29 and 31: "the criterion is running out of room exactly where the
  machines get big".

### The extremal stretches are structurally the wrong shape to qualify
- STATEMENT Of 132 stretches attaining `F_j` at machines 19/23/29, ZERO are literal and ZERO are
  qualifying. The shape is always two near-maximal flanks plus the machine's SMALLEST gaps inside
  (interiors 4, 3, 30 at machine 29's `F_5 = 85`, `k = 772,741,833`) - and the interior-gap floor
  `2u'` forbids exactly that shape.
- CALCULATES: an exclusion on the shape of a record-attaining stretch - it names the interior-gap
  floor `2u'` as the mechanism.
- STATUS: exact census (132 stretches at machines 19, 23, 29; the exact record position k given).
- WHERE: archive/agents-shared r19 SUMMARY (mechanic, "THE MECHANISM EXHIBITED, NOT ARGUED").
- LIMITS: at those three machines.

### The bridge identity: a word plus its two flanks IS a window sum
- STATEMENT `merged_eq`: a word occupying l consecutive gaps plus its two flanks spans exactly
  `l+2 = k+1` CONSECUTIVE gaps, so merged length IS a window sum. With `merged_le_spectrum` and
  `merged_le_of_shallow`, (D) at alpha=3 follows from `k_win <= 3` and `F_4 <= F + q'`. The
  statement mentions NO fuel, NO k_max, NO words, NO residues, NO padding - only `g : N -> N`.
- CALCULATES: the record of the bigger machine as a spectrum value of the smaller one.
- STATUS: kernel-checked (proofs/Spectrum.lean, formalist round 19); both empirical halves remain
  hypotheses inside the file.
- WHERE: archive/agents-shared r19 FORMAL LEDGER.
- LIMITS: the two empirical halves (k_win <= 3, F_4 <= F + q') are census inputs, not theorems.

### The envelope follows occurrence count, not span
- STATEMENT The monotone envelope (bigger span -> bigger max flank) holds within every step (19/19)
  but is FALSE as a machine law: at machine 29, span 21 -> max flank 27 (205,068 occurrences) vs
  span 25 -> max flank 30 (88,548 occurrences, k = 133,490,560). THE ENVELOPE FOLLOWS OCCURRENCE
  COUNT, NOT SPAN.
- CALCULATES: corrects any argument that orders flanks by span.
- STATUS: exact (machine 29, the violating pair exhibited with counts and position).
- WHERE: archive/agents-shared r19 SUMMARY (mechanic, "THE HUNTED VIOLATION, FOUND").
- LIMITS: it is a refutation of the envelope-as-law, kept as a positive rule about occurrence count.

### Par trading (gain per link = spectrum increment, loss per link = lambda*L)
- STATEMENT PAR TRADING IS DERIVED, NOT OBSERVED: gain per link = spectrum increment (5-15), loss
  per link = `lambda*L` (4.2, 5.5, 9.0) - approximately equal. Confirmed independently (spreads
  8.8%, 9.3%, 5.2%), `k_win = 3, 2, 2`; at 29->31 the k=3 and k=4 chains TIE at 55 while k=2 wins at
  58 - "fuel exists and LOSES by 3".
- CALCULATES: which chain depth wins at a given step, hence the winning merge.
- STATUS: measured/derived (machines up to 31).
- WHERE: archive/agents-shared r19 SUMMARY (constructor and mechanic, independently).
- LIMITS: a trade-off at the measured steps, not a proof of a bound.

### Hole structure and the coverability spectrum
- STATEMENT First enumeration of HOLES (gap values absent below the record): {9}, {17}, {19,24},
  {24}, {41,42} at machines 13-29. ABSENCE IS TRANSIENT - 5 of 6 heal at the next gear; only v = 24
  survives a step. `hist_M[v]` has a stable, converging residue law whose richest classes are `+-s`
  of gears 5 and 7 - THE CORRIDOR TEETH ARE LEGIBLE IN THE WHOLE MACHINE'S GAP HISTOGRAM. But the
  residue law does NOT predict the holes.
- CALCULATES: which stretch lengths do and do not occur in a machine; named construct COV(M), the
  COVERABILITY SPECTRUM - CRT arithmetic, no period scan, reaching machines 37/41/43/53, yielding
  the UPPER bounds on F_j that every prefix row lacks.
- STATUS: measured (machines 13-29); COV(M) was a NAMED NEXT CONSTRUCT at the end of round 19.
- WHERE: archive/agents-shared r19 SUMMARY (mechanic, "THE HOLE STRUCTURE").
- LIMITS: the residue law does not predict the holes; and a clean negative from lateral - residual
  demand vs purchasable supply leaves slack 8-16 at every g, so gap 24's absence is selection plus
  rarity, NOT a covering obstruction ("don't hunt one").

### The route (A)(B)(C)(D) and its audit at the end of round 19
- STATEMENT (B) fully checked and now UNIVERSAL; (E) checked, off-target; (A) PARTIAL - the
  class-reduction core checked, the list ENUMERATION only computed (the remaining gap); (C) had an
  unchecked half, now CLOSED by `onset_gate`. (D) is the part that the suppression-corrected
  flatness statement delivers.
- CALCULATES: the state of the derivation as of the last archived round.
- STATUS: as recorded (formalist round 19 audit).
- WHERE: archive/agents-shared r19 "AUDIT (A)/(B)/(C)/(E)".
- LIMITS: this is the round-19 state; the current SUMMARY records a different, later decomposition.

### The multiplicative route and the 2.5q constant (limit event V)
- STATEMENT With the sharp prime sum the tolerance is `alpha*(y) ~ ln y` (5.6 at y=101, 13.3 at
  10^6), so ANY fixed per-step constant delivers: `incr <= 2.5q` at every consecutive step beyond 47
  implies twins infinite (verified exactly to y = 10^6, worst ratio 0.656; observed maximum 2.432q
  at gear 37). The two missing lemmas are both gap-word extreme-value statements: TOP-GAP
  ANTI-CLUSTERING (`F2 - F = O(q)`) and FUEL-MERGE CONTROL (`excess = O(q)`); measured `<= 1.24q`
  and `<= 1.62q` at their separate maxima. The saturation regime (`q > F(M)`) is already a theorem
  with alpha = 1, but the consecutive chain never enters it.
- CALCULATES: a sufficient per-step bound on how much the record may grow when one gear is added.
- STATUS: verified exactly to y = 10^6; the two lemmas measured, not proved.
- WHERE: docs/proof-search/attempts-map.md, "Amendment, round 8: Limit event V"; constructor round
  8, research/multiplicative_route.py.
- LIMITS: "an attack belongs to event V iff its missing input is a statement about the machine's own
  gap word (adjacency and alignment at the top of the gap distribution) with no prime-counting
  content."

### Adjacency: two maximal gaps can never be adjacent
- STATEMENT At y = 13/17/19/23 two maximal gaps can NEVER be adjacent - certified by class
  arithmetic plus one period scan. Per-machine alpha1 closes with a three-tier check (A3
  machine-free / mod-385 strata disjointness / direct), written out at y=13.
- CALCULATES: an exclusion on adjacent record stretches - the top-gap anti-clustering input of the
  multiplicative route, at the measured machines.
- STATUS: exact (machines 13, 17, 19, 23).
- WHERE: archive/agents-shared r10 "ADJACENCY: NO" (constructor, answering lateral's target).
- LIMITS: honest limit recorded - the tier-C residual GROWS (4 at y=13 -> 96 at y=23), so scale
  needs mod-5005; uniformity in y still open.

### The fresh-block recursion: the machine's blockers are its own earlier output
- STATEMENT In any band, gear q's fresh blocks sit at `q*r` with r running over the PRIMES in
  `band/q`, so the machine's blockers in each band are its own output from lower bands, re-entered
  as structure. Deaths in band h draw root gears only from bands up to the tower-half - a hard
  LOWER-TRIANGULAR cutoff.
- CALCULATES: the band-by-band attribution matrix; which gear bands can act in which band.
- STATUS: exact by law L3 (computed over all 10^6 candidate pairs to midpoint 6e6, 168 bands).
- WHERE: docs/band-attribution.md; twin-prime-program.md section 17d.
- LIMITS: "The matrix quantifies the cascade; it does not bound anything." Any by-construction
  argument must control the PLACEMENT of the products q*r against the 1,5 columns - "a bilinear
  statement about pairs of machine outputs, not a per-gear statement".

### The gap has no closed form (and why)
- STATEMENT Offset t is open iff `gcd(n + t, primorial(R)) = 1`, so the distance to the next
  opening is "least t >= 1 with n + t coprime to primorial(R)", the joint condition across all gears
  at once. By CRT the open offsets are a union of residue classes modulo the primorial, exponential
  in R, and locating the least element above a given point IS the localisation problem. Anything
  producing the gap in time polynomial in log n would bound it and settle the question.
- CALCULATES: the ledger - PER-GEAR NEXT TOOTH is closed form; PER-OFFSET OPENNESS is closed form;
  THE GAP is not closed form and is equivalent to the open problem.
- STATUS: exact reasoning; three implementations cross-checked on 28,000 consecutive odd n with zero
  disagreements.
- WHERE: docs/gap-without-lattice.md; rust/src/bin/closedgap.rs.
- LIMITS: the floor on certifying one opening is `pi(R)` consultations - "exactly the window
  identity".

### The gear-at-infinity frame: recurrence is proved, localisation is not
- STATEMENT The machine is fully constructed to infinity; the gears return to their start after the
  primorial; at 0 the machine is completely aligned; no gear can outpace the 6-cycle; therefore the
  structure near 0 recurs. Four of the six steps are theorems. The gap is step 6, and the difficulty
  is LOCALISATION, NOT EXISTENCE: the recurrence period is the primorial (~e^y) while the region in
  which those gears can decide primality is only `(y, y^2]`.
- CALCULATES: nothing; it is the frame that produced the +-1 walk law, the two blocking laws, the
  mod-3 law for F_h, and the closed-form next-twin method.
- STATUS: steps 1-5 proved; step 6 open and equal to Reduction A.
- WHERE: docs/gear-at-infinity.md.
- LIMITS: "the frame gives: the configuration exists, recurs, and recurs at the fastest rate the
  machine allows. It does not give: it recurs within y^2 of where the gear set was assembled."

### The README's original alignment observation (the oldest record)
- STATEMENT "Each new odd prime p has its first multiple after p at 2p, which is even and therefore
  already blocked by prime 2. This means new primes predominantly ALIGN their blocking patterns with
  existing blocked positions rather than creating entirely novel constraints." Hierarchical coverage:
  within the 10-position gaps between consecutive multiples of 11, the blocking patterns from
  {2,3,5,7} provide complete coverage.
- CALCULATES: nothing; it is the first statement in the repository of the idea that a new gear's
  strikes mostly land on already-struck columns.
- STATUS: informal observation (the repository README, predating everything else).
- WHERE: README.md, "Alignment Property" and "Hierarchical Coverage Property".
- LIMITS: the same README records "No clear mechanism emerges from the modular arithmetic for
  simultaneous blocking of adjacent odd positions" as an open question, and lists "Formal analysis of
  how period expansions (2 -> 6 -> 30 -> 210 -> ...) preserve twin prime configurations" as an area
  for further investigation.

---

## PART 8 - The merge law in full (how the record of the bigger machine is built)

### The merge transform: adding a gear = q phased copies of the old pattern
- STATEMENT Let M have period P and opening set E. Adding gear q gives period Pq and
  `E' = { x in [0, Pq) : x mod P in E, x mod q not in {0,1} }`. Lap l keeps the points whose
  `e mod q` avoids `{ -lP mod q, (1-lP) mod q }`, and THAT PAIR SHIFTS BY `-P mod q` PER LAP. Since
  `gcd(P,q) = 1` the shift generates, so EVERY PHASE OF THE NEW GEAR OCCURS IN EXACTLY ONE LAP and
  EVERY OPENING OF M IS DELETED IN EXACTLY 2 OF THE q LAPS. "Adding a gear is q copies of the old
  pattern, each thinned at a different phase, laid end to end."
- CALCULATES: the whole new gap histogram from the old pattern and q; "deleting k consecutive
  points merges k+1 gaps. EVERY NEW GAP IS A SUM OF CONSECUTIVE OLD GAPS, so the record grows only
  by merging."
- STATUS: exact - verified against direct construction for four extensions (gears to 7 plus 11, to
  11 plus 13, to 13 plus 17, to 17 plus 19), matching the ENTIRE gap histogram, not only the record.
- WHERE: docs/gear-recursion.md section 3; research/gear_recursion.py.
- LIMITS: stated in the adjacent frame `{0,1}`; needs gcd(P,q) = 1.

### Deletion-spacing lemma, proved and tight
- STATEMENT Within one lap consecutive deleted points are at least `q-1` apart: deleted points lie
  in `{phi, phi+1}` mod q so any two differ by 0 or +-1 mod q, and old gaps are at least 3, so
  `delta = 0 mod q` gives `>= q`, `= 1 mod q` gives `>= q+1`, `= -1 mod q` gives `>= q-1`.
- CALCULATES: a stretch of length G holds at most `1 + G/(q-1)` deleted points.
- STATUS: proved; TIGHT - measured minimum spacing exactly q-1 at q = 13 and q = 19 (minima 12/12,
  18/16, 18/18, 24/22).
- WHERE: docs/gear-recursion.md section 4.
- LIMITS: it is the k=2 case of the chain condition.

### Saturation theorem: a far gear gives F2 of the old machine, independent of q
- STATEMENT If `q - 1 > F(M)` then `F(M + q) = F2(M)` exactly (F2 = largest sum of two adjacent old
  gaps), because a chain with k >= 2 needs an interior gap `0 or +-1 mod q` and at least 3, hence at
  least q-1, and no gap of M reaches q-1. ABOVE THE THRESHOLD THE INCREMENT DOES NOT DEPEND ON q AT
  ALL (gears to 7 plus any of 11..53 all give F = 21, increment 6 every time).
- CALCULATES: the new record with no scan whenever the added gear is far.
- STATUS: proved; checked over 48 pairs, zero violations.
- WHERE: docs/gear-recursion.md section 4b; archive constructor round 8 sec 19.3.
- LIMITS: "along the CONSECUTIVE chain q is always the next prime and q < F(M) throughout
  (47 < 354): the compliant regime and the needed regime are DISJOINT."

### Why chain length cannot be bounded mechanically (recorded so it is not retried)
- STATEMENT The deleted points are consecutive openings, so `p_k - p_1 = sum h_j >= (k-1)(q-1)` and
  `<= (k-1) F(M)`; together `(k-1)(q-1) <= (k-1) F(M)`, VACUOUS whenever `F(M) >= q-1` - precisely
  the regime the consecutive chain lives in. "Bounding k needs the ARITHMETIC of which gaps fall
  within 1 of a multiple of q."
- CALCULATES: nothing - it closes an approach.
- STATUS: proved-negative.
- WHERE: docs/gear-recursion.md section 4b; corpus 5.5.
- LIMITS: "a chain of length k needs k-1 consecutive gaps that are all both unusually large and
  pinned to within 1 of a multiple of q."

### Anatomy of the maximising chain: interiors sit within 1 of a multiple of q
- STATEMENT Below the threshold the maximising chain is short and every interior gap is what the
  condition demands: gears to 11 + 17 `[18] = 17+1`; to 11 + 19 `[18] = 19-1`; to 13 + 17
  `[33] = 2*17-1`; to 13 + 23 `[24] = 23+1`; to 17 + 19 `[39] = 2*19+1`; to 17 + 29 `[30] = 29+1`;
  to 19 + 23 k=3 `[45,24] = 2*23-1, 23+1`; to 19 + 31 k=3 `[30,63] = 31-1, 2*31+1`. k never exceeds
  3 in any maximum observed; excess `F(M+q) - F2(M)` reads 15, 6, 6, 0, 3, 9, 18.
- CALCULATES: the exact composition of the record-realising configuration at each step.
- STATUS: measured/exact at those steps.
- WHERE: docs/gear-recursion.md section 4b.
- LIMITS: seven steps.

### The exact record algorithm (residues drop out)
- STATEMENT `F(M+q') = max over k >= 1, over all k-sites, of (o[i+k] - o[i-1])`, where a k-site is k
  consecutive OLD openings whose spacing word is a legal killed word of q'. Because every site fires
  exactly once per new period, RESIDUES DROP OUT OF THE RECORD QUESTION ENTIRELY - no new-period
  scan, no residue bookkeeping. k=1 reproduces F2 identically.
- CALCULATES: F of the machine with one more gear from the smaller machine's opening sequence alone.
  Verified at SIX steps: 18, 25, 34, 43, 58, 88 for 13->17 .. 31->37.
- STATUS: exact (six steps), with the corrected legality condition.
- WHERE: archive lateral round 13; research/merge_correct.py.
- LIMITS: costs a scan of the old machine's full period.

### The legal killed-run condition (literal letters alternate, zeros free)
- STATEMENT Spacings `= 0 or +-2u mod q'`, with the NON-ZERO letters ALTERNATING and 0's insertable
  freely. Equivalently in mechanic's alphabet: `+1 = spacing s = 2u mod q'`, `-1 = spacing q'-s`,
  `0 = spacing 0 mod q'`. Two failure modes on record: literal-only MISSES PADDED LINKS (undershoot
  71 vs >= 88 at 31->37); all spacings without alternation is TOO PERMISSIVE (a +2u step goes
  -u -> +u so it is legal only FROM tooth -u; overshoot 45 vs 43 at 23->29 on the illegal word
  (10,10)).
- CALCULATES: which runs of consecutive old openings the new gear can merge.
- STATUS: exact, re-verified at all six steps; mechanic's validation `N3 = 62` at 19->23 with
  anatomy (8,15)/(15,8) reproduces the corpus fuel census exactly.
- WHERE: archive lateral round 13 correction; archive mechanic rounds 11 and 14,
  research/fuel_census.py.
- LIMITS: N_k counts co-deletable TUPLES, which differs from maximal-run counts once k_max > k.

### The general-gap form of the merge law (the only d-dependence is e)
- STATEMENT Gear q' has two teeth at `n = 0` and `n = -e (mod q')`, separated by e. Between adjacent
  members of a killed run sits a single M-gap g, and `g = 0 mod q'` is a PADDED link (same tooth),
  `g = +-e mod q'` is a LITERAL link (opposite teeth), anything else is ILLEGAL; non-zero letters
  alternate (forced, not assumed). Then `F(M+q') = max over LEGAL runs of span`. "This is Lateral's
  law with 2u replaced by e - the ONLY d-dependence in the law."
- CALCULATES: the record of the bigger machine for ANY even gap from the old machine alone.
- STATUS: exact over 14 configurations (d = 2, 4, 6, 10, 12, 30; machines {3,5,7,11}..{3..17};
  q' = 13, 17, 19; all CRT phases): soundness 0 violations, firing 0 misses, converse 0, IDENTITY
  14 of 14 EXACT.
- WHERE: archive harvester round 11 sec 1.
- LIMITS: wrap-around must be handled by computing kills at absolute positions over two periods.

### The firing law (one residue, not two)
- STATEMENT Inside a chain, kills sit at the two teeth `{u, -u}` alternately, so a kill at u is
  followed by `-2u = q'-s` and a kill at -u by `+2u = s`. The spacing word's FIRST entry fixes the
  orientation and hence a SINGLE firing residue: starts with s -> fires iff `p = -u (mod q')`;
  starts with q'-s -> fires iff `p = +u (mod q')`. Per-window firing density `1/q'`, HALF the naive
  `2/q'` (k=1 kills fire at both teeth).
- CALCULATES: whether a given fuel site fires in a given phase window, from the word's first letter.
- STATUS: derived then verified with ZERO violations over 13,062 sites at 19->23 and 29->31.
- WHERE: archive lateral round 12; research/firing_ratio.py, firing_law_check.py, firing3137.py.
- LIMITS: literal chains.

### Every fuel site fires exactly once per new-machine period
- STATEMENT The new period is `q'*P_old` and `P_old` is invertible mod q', so each site recurs at q'
  distinct residues across the q' phase windows: EVERY FUEL SITE FIRES EXACTLY ONCE PER NEW PERIOD,
  at the closed-form address `j = (fire - p) * P_old^{-1} (mod q')`, position `p + j*P_old`.
- CALCULATES: the exact absolute column at which a fuel site fires. All four k=4 sites of machine
  29: `j = 12, 30, 0, 18` -> positions 13,159,557,562 / 32,754,547,977 / 672,200,337 /
  20,267,190,752, chain residues [26,5,26,5].
- STATUS: exact.
- WHERE: archive lateral round 12.
- LIMITS: consequence - realized k-chains per NEW period = N_k exactly; ALIGNMENT IS A DENSITY
  FACTOR, NEVER A COUNT FACTOR.

### Firing is binary
- STATEMENT `gcd(P_M, q') = 1`, so the q' CRT copies realize every residue shift; every occurrence
  of a compatible word fires in exactly `|valid starts|` of the copies, and INCOMPATIBLE WORDS NEVER
  FIRE ANYWHERE. "There is no surviving fraction to multiply the ceiling by - which is precisely why
  the word-indexed statement is an IDENTITY rather than an inequality."
- CALCULATES: the lower-bound half of the word-indexed identity.
- STATUS: exact.
- WHERE: archive constructor round 12 secs 24.1, 24.4.
- LIMITS: none.

### THE WORD-INDEXED IDENTITY (exactly where the new record comes from)
- STATEMENT With `u' = round(q'/6)`, `a = 2u'`, `b = q'-a`, `L = litcap(q' mod 210)`, let `W(q')` be
  the alternating words in `{a,b}` of length `<= L-1` plus the padded words; w is COMPATIBLE if some
  tooth residue `r in {c, q'-c}` has all partial sums `r + prefix(w)` again in `{c, q'-c}`. Then
  `F(M+q') = max( F2(M), max over COMPATIBLE w of [ span(w) + FS_max(w; M) ] )` with
  `FS_max(w;M) = max over occurrences of w of (gap before + gap after)`.
- CALCULATES: the exact new record from a word list depending on `q' mod 210` ALONE plus occurrence
  and flank data of the old gap word. Verified at all six steps (11, 18, 25, 34, 43, 58); binding
  words (4), (6), (13), (8,15), (10), (10).
- STATUS: exact identity, 6/6; transfers verbatim to every even d (13/13, `tier_1 = F2(M)` exactly
  in every row - "the 1-letter word always fires").
- WHERE: archive constructor round 12 secs 24.1-24.2; archive harvester round 10 sec 2.
- LIMITS: `FS_max` is the sole open input.

### Compatibility and the corridor never interact (CRT independence)
- STATEMENT `gcd(35, q') = 1`, so the tooth condition defining COMPATIBILITY is CRT-independent of
  the mod-35 carrier: firing and tier A never interact.
- CALCULATES: lets the two filters be applied independently.
- STATUS: exact (holds because q' > 7).
- WHERE: archive constructor round 13 sec 25.1.
- LIMITS: none.

### Tail-run cap: every qualifying interior is at least 2u'
- STATEMENT Every qualifying interior gap is `= 0 or +-2c mod q'` and positive, hence `>= 2u'`. A
  k-chain's k-1 interior gaps are CONSECUTIVE gaps of M all `>= 2u'`, so
  `k_max(M, q') <= T(M, 2u') + 1` with T the longest run of consecutive gaps `>= 2u'`.
- CALCULATES: a residue-free cap on chain length from the old gap word alone; T = 3, 2, 4, 3, 4, 5
  across steps 11->13 .. 29->31.
- STATUS: exact theorem, one line.
- WHERE: archive constructor round 11 sec 23.1.
- LIMITS: loose (realized k_max 2,2,3,2,4) because chains also need residue alignment.

### THE LITERAL CAP THEOREM (chains of at most 6 members, at every gear, forever)
- STATEMENT A literal chain has member positions `r, r+2u', r+q', r+q'+2u', ...` - an interleaved
  two-phase walk of period 70 mod 35 that must stay inside the 15-residue exposed set. Its maximal
  run is a function of `q' mod 210` ONLY: cap 2 on 24 classes, cap 3 on 4, cap 4 on 14, cap 6 on 6
  (`q' = 37, 53, 83, 127, 157, 173 mod 210`). LITERAL CHAINS HAVE AT MOST 6 MEMBERS, FOR EVERY GEAR,
  FOREVER; `k=5 at q'=31 is FORBIDDEN mod 35`.
- CALCULATES: the maximum literal chain length for any gear from its class mod 210.
- STATUS: kernel-checked (`LiteralCap.no_run_seven`, `literal_chain_le_six`, `s_eq`,
  `cap_six_classes_sharp`; no native_decide); verified as a class function against every prime to
  5000, zero mismatches.
- LIMITS: LITERAL chains only - explicitly withdrawn for padded chains ("padded links have no
  analogue"); and "cap <= 6 for ALL (t,s) pairs mod 35" is FALSE (spectrum {2,3,4,5,6,8,10,140}),
  so the invertible-class restriction does real work.
- WHERE: archive constructor round 11 sec 23.2; archive formalist round 13, proofs/LiteralCap.lean.

### The cap explains the realized selection
- STATEMENT k_max by step = 2, 2, 3, 2, 4 at gears with caps 2, 2, 4, 3, 4 - saturated at
  q' = 17, 19, 31; the k=4 event sits exactly at a cap-4 gear. Prediction: the first literal k = 5
  or 6 can only occur at a cap-6 gear.
- CALCULATES: which gears can host the deepest chains.
- STATUS: measured agreement at six steps; the prediction is falsifiable and unconfirmed in the
  archive.
- WHERE: archive constructor round 11 sec 23.2.
- LIMITS: prediction only.

### The padded link (two openings struck at the SAME tooth, one lap apart)
- STATEMENT A link is padded iff its two openings share ANY residue mod q' - the same tooth, one lap
  apart - so its interior gap is `0 mod q'`, hence `>= q'`. Worked address (machine 31, q'=37):
  openings k = 634158 (3804947, 3804949) and k = 634195 (3805169, 3805171), flanks at 634153 and
  634197, interior column-gap 37 = 111 adjacent = 222 integers, both endpoints `k = 31 mod 37` (the
  SAME residue), `3805169 - 3804947 = 222 = 6 x 37` exactly. Second exhibited witness at machine 31:
  slot gap `8,288,105 - 8,288,068 = 37`, halved `24,864,314 - 24,864,203 = 111`, members
  `49,728,629 - 49,728,407 = 222`.
- CALCULATES: the extra way a new gear's teeth align with two openings of the machine below.
- STATUS: exact, two verified witnesses.
- WHERE: archive mechanic rounds 14-15; archive harvester round 12 sec 1.
- LIMITS: the shared residue is NOT `+-u'` (15 vs u' = 31 in the example) - "the phase decides where
  they fire, not whether". SUPPLY is not LINKS: a link also needs its endpoint on a tooth, `2/q'` of
  supply (26,366 supply gaps at 31->37 give about 1,400 links).

### Padding onset: necessity is a theorem, sufficiency is FALSE
- STATEMENT Padding supply > 0 requires `F(M) >= q'` - a gap of exactly q' must fit at all (ZERO by
  structure at 13->17 and 17->19). Kernel form `TierA.onset_gate : (0 < g) -> (q | g) -> (g <= F)
  -> q <= F`. SUFFICIENCY IS FALSE: machine 29 has `F = 43 >= 41` yet `supply(29, 41) = 0 EXACTLY` -
  41 is not realized as a gap of machine 29 while 43 is (twice). Availability is governed by the
  gap-value SPECTRUM, not by F.
- CALCULATES: `supply(M, q') = hist_M[q']` exactly - one gap histogram answers the onset question
  for every future gear at once.
- STATUS: necessity kernel-checked ([propext] only); counterexample exact at full period; the
  histogram identity reproduces the full-period censuses exactly (machine 29: 2090, 84, 0, 2 at
  q' = 31, 37, 41, 43; machine 31: 26366 at q' = 37).
- WHERE: archive mechanic rounds 14-16, research/hist_probe.py; archive formalist round 18.
- LIMITS: a PREFIX bounds the histogram from below - a positive entry is definitive, a zero is not.
  BOUNDARY CASE, sharp: at `q' = F(M)` exactly (machine 29, q'=43) the supply is 2, precisely the
  number of maximal gaps in the period.

### Padding is tier-blind: two independent axes
- STATEMENT Padding does NOT change the tier bound - a run of k killed openings merges k+1 gaps
  whatever its letters, so `F_{k+1} >= F(M+q')` is padding-blind. What padding changes is
  FEASIBILITY: it makes runs legal that literal letters would break. TIER = how many gaps merge;
  PADDING = whether the links connect. The 31->37 record needs BOTH: k = 3 AND one padded link.
- STATUS: exact / structural.
- WHERE: archive mechanic round 14 sec (3).
- LIMITS: none.

### PADDING IS THE GEAR-37 ANOMALY
- STATEMENT At 31->37 the run census splits by padded-link count z: `z=0` 114,750,740 runs, max
  flanked span 71; `z=1` 26,366 runs, max 88 - THE TRUE RECORD (k=2: 26,030, max 85; k=3: 336, max
  88); `z>=2: 0`. LITERAL-ONLY WOULD GIVE 71, NOT 88. Winner anatomy
  `[kill]--37--[kill]--12--[kill]` at `k = 9,463,664,103`, span 49 = q' + B, flanks 28+11, excess 20
  = +0.541 q'. The corpus's unexplained gear-37 spike (2.432q against neighbours 0.22q and 0.84q) is
  exactly the first step whose winning word carries a padded link - A STRUCTURALLY DIFFERENT TIER
  SWITCHING ON, NOT A FLUCTUATION.
- CALCULATES: the record of the bigger machine and which 336 runs in a 3.34e10-column period produce
  it. The first five steps have LITERAL winners (spans 11, 13, 23, 10, 10).
- STATUS: exact, full period 3.343e10, found independently by lateral and by mechanic.
- WHERE: archive lateral round 13 final; archive mechanic round 14.
- LIMITS: "the route's tightest constraint is not a length effect but an AVAILABILITY effect:
  whether M carries a gap of exactly q'."

### THE PADDING LEMMA (two padded links require a spectrum value)
- STATEMENT A legal killed run of k kills occupies k+1 CONSECUTIVE gaps of M, so `G <= F_{k+1}(M)`.
  Two padded links with j literal links between them occupy j+2 consecutive gaps summing to at least
  `2q' + j*L` (L = min(s, q'-s)). Hence two padded links require `F_{j+2}(M) >= 2q' + j*L` for some
  j; contrapositively, if `F_{j+2}(M) < 2q' + j*L` for every j then every run carries AT MOST ONE
  padded link. Headline j=0: `F_2(M) < 2q'` says two padded links can never be adjacent. Companion:
  if `2q' > F(M)` then every padded link has size EXACTLY q'.
- CALCULATES: `p <= 1` per step from the spectrum; with p <= 1 and size exactly q' the run is
  `[literal chain] --q'-- [literal chain]`, so cap-6 applies to each segment and `k <= 12`,
  `span <= 5q' + 2s <= 6.35 q'`.
- STATUS: exact at every computed step and confirmed over full periods - padded gaps 0, 0, 86, 6,
  2090, 26367 at 13->17 .. 31->37, ALL of size exactly q', 0 adjacent padded pairs, max 1 per run.
- WHERE: archive lateral round 14; research/padding_bound.py, padding_horizon.py, padding31.py.
- LIMITS: it EXPIRES - `F(M)/2q'` climbs 0.32, 0.47, 0.54, 0.59, 0.69, 0.78 then 1.07, and
  `F2(M)/2q'` 0.47, 0.66, 0.67, 0.67, 0.89, 0.92 then 1.10, so the ceiling dies exactly at 37->41.
  It bounds the SPAN, not the increment.

### Padded-link cost per frame, and the cheapest padded link per gap class
- STATEMENT A padded link costs `q'` in COLUMN units, `3q'` in halved/adjacent units, `6q'` in
  member units. Cheapest cost by gap class: if `3 | e` gear 3 leaves TWO classes, gaps take all
  residues mod 3, and the cheapest padded link costs exactly `q'`; if `3` does not divide e gear 3
  blocks two classes so ALL openings lie in ONE class mod 3, every gap is divisible by 3, and the
  cheapest padded link costs `3q'`.
- CALCULATES: the per-gap padding cost `c_d`, hence `p <= F(M+q')/c_d`, and the onset condition
  `F(M) >= c_d`.
- STATUS: proved one line each way; measured (d=6: min padded gap 17 at q'=17, 19 at q'=19; d=12: 13
  at q'=13, 19 at q'=19; d=30: 17 at q'=17; d=2, 4, 10: none at these machine sizes). For twins the
  first padded winner is at 31->37 (sixth step); for d = 12 a padded run WINS at 11->13, the FIRST
  step tested.
- WHERE: archive harvester rounds 11-12.
- LIMITS: padding is 3x cheaper in absolute terms for `3 | e` but 1.5x in scale-relative terms.

### Padding count bound (it grows)
- STATEMENT Each padded link consumes a gap `= 0 mod q'`, so a run of span `<= F(M+q')` carries
  `p <= F(M+q')/c_d`, which in the kernel reads `p <= F/q + 5/6` - A BOUND THAT GROWS. Companion:
  `padding_three_not_excluded : 13*q <= 6*F -> 6*(3*q) <= 6*F + 5*q` records that once
  `F >= (13/6)q` the budget stops excluding three padded links.
- CALCULATES: the maximum number of padded links in a record-length run.
- STATUS: kernel-checked (`padding_count_le` needs NO axioms at all); measured over 8 configurations
  (d = 2, 4, 6, 12, 30), no violations.
- WHERE: archive harvester round 12 sec 3; archive formalist rounds 15 and 18.
- LIMITS: `padding_at_most_one` was renamed `padding_at_most_one_below_onset` - `F < q` is precisely
  the NO-PADDING regime, not "the onset condition".

---

## PART 9 - The corridor: what gears 5 and 7 forbid, anywhere, forever

### The exposed set mod 35
- STATEMENT `exposedSet = {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33}` and `Exposed k <-> k mod 35 in
  exposedSet` (for k >= 1): every opening of any machine lies in these 15 residues mod 35.
- CALCULATES: an immediate residue filter on every opening address, machine-free.
- STATUS: kernel-checked (`Corridor.exposedSet`, `exposed_iff_mem`).
- WHERE: archive formalist round 10, proofs/Corridor.lean.
- LIMITS: two gears only; residue laws cannot cap sizes (see escape distance).

### THE HORIZON THEOREM: no stretch of one-prime columns exceeds 32
- STATEMENT Gear pair (5,7) has B-classes `{1, 34}` mod 35 (both members composite there); the max
  cyclic gap between them is 33, so ANY 33 consecutive columns contain a both-composite column.
  Hence every saturated run - every stretch of consecutive columns each carrying at least one prime
  - has length `<= 32`, at every scale, forever.
- CALCULATES: an unconditional cap from two residue classes mod 35.
- STATUS: PROVED, two lines; kernel-checked as `Corridor.exists_class_in_run`,
  `both_composite_of_class`, `both_composite_in_run`, `double_slot_in_run`,
  `prime_adjacent_run_le` ([propext, Quot.sound] only). Escalation-checked: adding gears does NOT
  lower it through gear 23 (L0 = 32 for {5,7}, {5,7,11}, {5,7,11,13} (period 5005, 730 B-slots, 4
  surviving corridor phases), {..17} (85085), {..19} (1.6M), {..23} (37.2M, 8.6M B-slots)).
- WHERE: archive lateral round 8; archive formalist round 9.
- LIMITS: column 1 IS the twin (5,7) - the unique class exception, excluded by a `k >= 2` guard.
  Whether `lim L0 = 32` over all gears is a Jacobsthal-type question, finitely checkable per gear
  set, monotone non-increasing.

### The corridor mouth pins the landmark
- STATEMENT The (5,7) corridor starts at `k = 2 mod 35`, and the L* = 13 landmark sits at column
  2452 = 2 mod 35 - AT THE CORRIDOR MOUTH; at gears <= 17 and <= 19 the extremal corridor's absolute
  start IS column 2452. "The landmark lives where it does because that is the widest small-gear
  corridor."
- CALCULATES: the address of the longest saturated stretch from the small-gear geometry.
- STATUS: exact, verified through gears <= 23.
- WHERE: archive lateral round 8.
- LIMITS: through gear 23.

### n2_packing (a forced closed column in every 33)
- STATEMENT `n2_packing : 2 <= a -> W / 33 <= Census.n2 (Ico a (a+W))` - packing disjoint 33-windows
  gives a floor on double-composite columns in any range.
- CALCULATES: the formal floor on closed columns any stretch must contain.
- STATUS: kernel-checked (uses Classical.choice via `choose`; a `Nat.find` variant would remove it).
- WHERE: archive formalist round 10.
- LIMITS: a floor with no matching ceiling.

### Unconditional load ceiling past the horizon
- STATEMENT On any twin-free stretch, B-columns carry zero primes, so `P_run <= L - minB(L)`:
  `L = 33/50/100/200/252 -> ceiling 0.970/0.920/0.910/0.880/0.873`, asymptote
  `1 - 730/5005 = 0.854`. The first unconditional load ceiling below 1 beyond the horizon.
- CALCULATES: a ceiling on prime load for any stretch length from the B-column count of the
  (5,7,11,13) period.
- STATUS: proved from the B-class census.
- WHERE: archive lateral round 8.
- LIMITS: reality sits far below (0.52 at L=100), so the ceiling closes the L > 32 frontier without
  creating a contradiction.

### The endpoint law (which residues can start a stretch of given length)
- STATEMENT A gap of length G runs between openings a and a+G, so
  `a mod 35 in A(G) = {r in E : (r+G) mod 35 in E}`; `|A|` ranges 3..15, and `G = 34 mod 35` forces
  `a mod 35 in {3, 18, 33}`.
- CALCULATES: candidate left-endpoint residues of any record stretch; prunes a covering search by
  `15/|A| = 2-5x`; transfers to the adjacent frame mod 105. MEASURED CONCENTRATION EXCEEDS THE
  FORCING: at gears<=19 (F = 25, nine residues allowed) ALL TWENTY records sit at the SINGLE residue
  5; at gears<=23 the four record gaps sit at {3, 33} of the three allowed.
- STATUS: kernel-checked (`Corridor.endpoint_law`, `endpoint_law_34`); verified at every gap in five
  full periods.
- WHERE: archive constructor round 9 sec 20.1; archive formalist round 10.
- LIMITS: constrains WHERE only.

### The adjacency law A3 and the 294 forbidden pairs
- STATEMENT Adjacent gaps `(G1, G2)` force `a, a+G1, a+G1+G2` into E; the allowed set is
  `allowed3 (G1 mod 35) (G2 mod 35)`, and it is EMPTY for 294 of the 1225 length-pairs mod 35 - the
  first examples being (1,1), (1,3), (1,6). `no_chain_of_forbidden`: such a chain of three openings
  can never exist, anywhere.
- CALCULATES: a machine-free table of impossible adjacent stretch-length pairs.
- STATUS: kernel-checked (`Corridor.adjacency_law`, `forbidden_pairs_count = 294` by
  `decide +kernel`, no `native_decide`, 22 s).
- WHERE: archive constructor round 9; archive formalist round 10.
- LIMITS: 931 of 1225 pairs remain allowed. Every observed F2-realising pair sits inside its allowed
  set, at machines where that set has as few as 2 residues.

### ESCAPE DISTANCE = 1 (the decisive negative on corridors)
- STATEMENT Computed over all 1225 pairs: EVERY `(G1,G2)` is within L1 distance 1 of a
  corridor-ALLOWED pair. A near-maximal gap has ~35 candidate lengths in its range, so any residue
  exclusion is evaded by a +-1 slide in one component. CORRIDOR ARITHMETIC CONSTRAINS WHERE
  TOP-GAP CONFIGURATIONS SIT, NEVER HOW BIG THEY ARE - at modulus 35 and, by the same argument, at
  ANY bounded modulus (the exposed set's own max gap stays O(1), so escape distance stays O(1)).
- CALCULATES: proves no size bound can follow from bounded-modulus arithmetic, however many corridor
  levels are stacked.
- STATUS: exact over all 1225 pairs, plus a general argument.
- WHERE: archive constructor round 9 sec 20.2; carried as "escape distance = 1" in the SUMMARY.
- LIMITS: a negative about the corridor method, not about the machine.

### Local capacity cap on F2 (the corridor's only size statement)
- STATEMENT Refining the corridor to a density statement (base gears B with exposed density rho;
  killers `q in (B, y]` supply `2*ceil(S/q)` deletions per span S) gives
  `rho*S - 1 <= sum 2*ceil(S/q)`, hence `F2_k(y) <= (2#K + 1)/(rho - 2 sum 1/q)` when the margin is
  positive. Base {5,7} -> `F2_k(11) <= 12` (actual 11, TIGHT); `F2_k(13) <= 54` (actual 16); y=17
  VACUOUS; base {5..17} -> `F2_k(23) <= 72` (actual 39); y=31 VACUOUS.
- CALCULATES: an exact upper bound on the two-stretch record two or three gears above any base.
- STATUS: exact where the margin is positive.
- WHERE: archive constructor round 9 sec 20.3.
- LIMITS: "the margin `rho - 2 sum 1/q` dies two to three gears above ANY base."

### Tier A: the carrier of a chain of openings
- STATEMENT A word occurrence with both flanks is a chain of openings
  `p0, p1 = p0+gL, p1+w1, ..., p1+span, p1+span+gR` - ALL exposed, hence all in E mod 35. With
  `S_m(w) = {r in E_m : every partial sum r + w_1..w_j in E_m}` (the CARRIER), the flank pair
  `(gL, gR)` is tier-A-feasible iff some `r in S_m(w)` has `r - gL in E_m` and
  `r + span + gR in E_m`. Interior non-openings give no tier-A constraint.
- CALCULATES: machine-free feasibility of any (word, flank-pair) configuration; `l = 0` recovers A3.
- STATUS: kernel-checked (`TierA.carrier`, `mem_carrier_of_chain`, `no_chain_of_carrier_empty`,
  `flanked`, `no_maximal_flanks`); cost independent of the machine - "this is the piece that scales
  past the scans".
- WHERE: archive constructor round 13 sec 25.1; archive formalist round 15, proofs/TierA.lean.
- LIMITS: SIZE-BLIND - it forbids exact value combinations, never a size range.

### Both-flanks-maximal is machine-free forbidden at 14 of 16
- STATEMENT "Both flanks maximal" is FORBIDDEN machine-free at 14 of 16 word-step pairs; "one flank
  maximal" at 9 of 16. The two joint exceptions are `w = (8)` and `w = (15)` at 19->23. Decidable
  from `(q' mod 210, w, F mod 35)` alone. Kernel instances closing by corridor arithmetic alone:
  11->13 (w=(4), F=7), 13->17 ((6), 11), 17->19 ((13), 18), 23->29 ((19), 34), 29->31 ((10), 43).
  `flanks_17_19` is the sharp one - each flank ALONE is feasible mod 35, both together are not.
- CALCULATES: excludes the extreme flank configuration by residue arithmetic; at 29->31 every
  compatible word except (10) has `L0 R0` at modulus 35 already.
- STATUS: exact (14/16); kernel-checked for the listed steps.
- WHERE: archive constructor round 13 sec 25.2; archive formalist round 15.
- LIMITS: HONEST EXCEPTION, recorded as a theorem rather than omitted -
  `flanks_19_23_nonempty : carrier (flanked 25 [8]) = {0, 5, 7, 12}`. Tier A does NOT close 19->23.
  And the whole line was later recorded OFF-TARGET (the binding flanks are mid-size).

### Tier B is not a tier (lifting the modulus buys nothing)
- STATEMENT Lifting `35 -> 385 -> 5005 -> 85085 -> 1616615` (gears 5,7 / +11 / +13 / +17 / +19),
  carrier and feasibility counts scale up proportionally and NEVER reach zero where tier A did not
  already give zero - at all 16 word-step pairs tier B adds EXACTLY ZERO new exclusions. Structural:
  `S_m` and `E_m` are unions of lifts, so a mod-35 feasible configuration stays feasible at every
  multiple modulus. "B is not a tier at all here; the hierarchy is A (machine-free, scalable) versus
  C (period, unscalable)."
- CALCULATES: tells you not to lift the modulus.
- STATUS: exact (structural argument + 16/16 empirical, verified through gear 19).
- WHERE: archive constructor round 13 sec 25.4.
- LIMITS: the residual is a pure period-scan fact (5,005 to 1.08e9 columns; 3.3e10 at 31->37,
  "already past kernel reach").

### THE COMPLETENESS LEMMA (mod 35 IS the entire corridor for small shapes)
- STATEMENT A shape with n openings can be blocked by gear q only if `q <= 2n`: gear q has two
  teeth, so it forbids at most 2n phases out of q, and if `2n < q` some phase always survives.
  Constraints from distinct gears are independent by CRT, so a shape is corridor-feasible iff it is
  feasible gear by gear. Consequence: for `n = 4 or 5` ONLY GEARS 5 AND 7 CAN BLOCK, so the mod-35
  test IS the entire corridor and NO LARGER MODULUS CAN EVER HELP. Gear 11 first enters at n = 6,
  gear 13 at n = 7.
- CALCULATES: settles "would mod 385 / mod 1155 help?" structurally, with no computation, and
  retroactively certifies every mod-35 verdict of rounds 15-16 as complete.
- STATUS: PROVED.
- WHERE: archive lateral round 17, research/corridor_complete.py.
- LIMITS: consequence - the 37->41 j=1 shape (n = 4 openings) is GENUINELY FEASIBLE; no corridor at
  any modulus kills it.

### THE ADJACENT-GAP EXCLUSION LAW (24% of adjacent gap-pairs are impossible)
- STATEMENT For a lag pair the correlation ratio against independence is
  `Lambda(g1,g2) = prod_q [ c_q(0,g1,g1+g2)*(q-2) / (c_q(0,g1)*c_q(0,g2)) ]`, and it vanishes
  exactly when some gear has `c_q = 0` on the three-point shape. By the completeness lemma only
  gears `q <= 6` can do it, so ONLY GEAR 5 EVER DOES. Working gear 5 out (exposed residues {0,2,3}):
  THREE CONSECUTIVE OPENINGS WITH GAPS (g1, g2) ARE IMPOSSIBLE WHENEVER
  `(g1 mod 5, g2 mod 5) in {(1,1), (1,3), (2,4), (3,1), (4,2), (4,4)}` - 6 of 25 classes, 24%.
  Forced, at every scale, in every machine containing gear 5, AND THE LIST IS COMPLETE.
- CALCULATES: 24% of adjacent gap-pair classes are forbidden outright anywhere in any machine.
- STATUS: PROVED (a proof, not a statistic); cross-checked against the independent joint gap-pair
  census over six machines y = 11..29 - at lag 1, 1,589 populated cells, ZERO in a forbidden class;
  at lag >= 2 the SAME classes are heavily populated (up to 35,798,770 counts at y=29), exactly as
  the scope predicts.
- WHERE: archive lateral round 20 sec B, research/npoint_autocorr.py.
- LIMITS: ADJACENT GAPS ONLY. At separation `j >= 2` the intervening openings are free, the offsets
  are not determined, and no exclusion follows.

### THE AP LEMMA (mod 5, scale-free)
- STATEMENT Gear 5 exposes only 3 of its 5 residues (`k mod 5 in {0,2,3}`, teeth at 1 and 4). Four
  terms of an arithmetic progression with common difference coprime to 5 occupy four DISTINCT
  residues mod 5, and three residues cannot hold four. Hence NO RUN OF OPENINGS EVER CONTAINS FOUR
  OPENINGS IN ARITHMETIC PROGRESSION WITH COMMON DIFFERENCE q', FOR EVERY PRIME q' > 5.
- CALCULATES: forbids `j = 2` between two padded links for every q' (offsets 0, q', q'+v, 2q', 3q'
  contain the 4-term AP {0, q', 2q', 3q'}); forbids `p = 3` all-adjacent.
- STATUS: PROVED, verified exhaustively over all (r, g) mod 5 with g invertible, zero exceptions;
  flagged as a two-line kernel target.
- WHERE: archive lateral round 16, research/corridor_ap_lemma.py.
- LIMITS: silent on shapes that are not pure q'-multiples.

### THE OPENINGS AP THEOREM (arbitrary common difference)
- STATEMENT Gear q leaves q-2 residues and an L-term AP with `gcd(d,q)=1` occupies `min(L,q)`
  distinct residues, so `L > q-2` forces a tooth. Hence AN AP OF L OPENINGS HAS COMMON DIFFERENCE
  DIVISIBLE BY EVERY GEAR `q < L + 2`: 3 equal consecutive gaps require `5 | g`, 5 require
  `35 | g`, 9 require `385 | g`, and `L >= y+2` needs difference at least the full primorial `P(y)`.
- CALCULATES: forbids equally spaced openings unless the spacing carries the small-gear primorial.
- STATUS: PROVED and verified on full periods of machines 13, 17, 19, 23, 29 - zero violations; the
  longest run of equal consecutive gaps is 3-4 at every machine WITH g = 5 EXACTLY IN EVERY CASE -
  the theorem's minimal witness, realised.
- WHERE: archive lateral round 18, research/openings_ap.py, openings_ap2.py.
- LIMITS: none stated.

### THE SHAPE LAW (two padded links can only be separated by j in {0,1})
- STATEMENT Exhaustive residue check over all 840 invertible `(g, v)` pairs mod 35:
  `j=0 feasible 50%; j=1 32%; j=2 0% ALWAYS IMPOSSIBLE; j=3 4% of abstract pairs but 0 of 546 actual
  primes 11..4000; j=4 0% ALWAYS IMPOSSIBLE`. Feasibility is a function of `q' mod 210` (42 distinct
  residues, zero clashes) - the same modulus as the word list.
- CALCULATES: the finite, scale-free family of padded-run shapes; `span <= (4+p)q' + 2s`.
- STATUS: verified for every prime to 4000; `j = 2` and `j = 4` PROVEN outright by the AP lemma.
- WHERE: archive lateral round 16, research/corridor_shapes.py.
- LIMITS: the SHAPE law is permanent, the COUNT p is not (see the correction in the refuted list).

### THE GENERALISED AP LEMMA (which p=3 arrangements survive)
- STATEMENT Four openings at pure q'-multiples `i*q'` whose four values of i are DISTINCT mod 5 are
  impossible. For three padded links with j-patterns `(j1, j2)`, j in {0,1}:
  `(0,0): i = {0,1,2,3}` 4 distinct mod 5 -> IMPOSSIBLE; `(1,1): i = {0,1,3,4}` -> IMPOSSIBLE;
  `(0,1): i = {0,1,2}` only - lemma silent; `(1,0): i = {0,1}` - silent. The two survivors are
  corridor-feasible for 4 of 27 primes tested, FIRST AT `q' = 43` (also 47, 103).
- CALCULATES: kills the cheap p=3 arrangements at every scale; forces surviving p=3 shapes to spend
  literals.
- STATUS: PROVED (the lemma) / computed (the feasibility scan).
- WHERE: archive lateral round 17.
- LIMITS: `p <= 2` does NOT follow - p = 3 is structurally permitted from 41->43 on.

### THE 50/50 RESIDUE LAW (adjacent equal padded links switch on and off with q' mod 35)
- STATEMENT Two adjacent padded links of sizes `a q'`, `b q'` put three consecutive openings at
  `r, r+a g, r+(a+b) g` mod 35 with `g = q' mod 35`. For `q' = 41`, `g = 6`, `r, r+6, r+12 all in E`
  has ZERO solutions over all 15 `r in E` - so two adjacent equal padded links are impossible at
  37->41 by the (5,7) corridor alone, with no spectrum input. GENERAL LAW: feasibility depends only
  on `q' mod 35` - POSSIBLE for q' = 23, 37, 43, 47, 53, 67, 73, 83, 97 ...; IMPOSSIBLE for
  q' = 29, 31, 41, 59, 61, 71, 79, 89 ... - exactly 12 of the 24 invertible classes, a 50/50
  property of `q' mod 35`, NOT a trend in scale. PERFECT DICHOTOMY: whenever the (1,1) shape is
  feasible the unequal shapes (1,2) and (2,1) are infeasible, and conversely.
- CALCULATES: for any new gear, whether adjacent padding is structurally possible, from `q' mod 35`.
- STATUS: exact; kernel-checked as `TierA.no_adjacent_padded_41 : carrier [41,41] = empty`,
  `equal_padding_forbidden_classes = {1,4,6,9,11,16,19,24,26,29,31,34}`,
  `equal_padding_forbidden_card = 12`, and `padding_shape_dichotomy` PROVED AS AN IFF.
- WHERE: archive lateral round 15, research/padding_corridor_law.py; archive formalist round 16.
- LIMITS: this is why a smooth `supply^2/gaps` model cannot predict padding - arithmetic selection
  beats the smooth law.

### Gear 3 forbids adjacent padded links for every gap d = 0 mod 6
- STATEMENT For `3 | e` the padded cost is `c = q'` with q' not divisible by 3, so
  `r, r+q', r+2q'` occupy ALL THREE classes mod 3 and gear 3 blocks one. HENCE FOR EVERY q' AND
  EVERY `d = 0 mod 6`, TWO PADDED LINKS CAN NEVER BE ADJACENT - unconditionally, by gear 3 alone.
  For `3` not dividing e the step is `3q' = 0 mod 3`, all three openings share the class, gear 3
  says nothing, and the exclusion must come from gears 5 and 7 - which is why it holds for only 34
  of 74 twin probes.
- CALCULATES: an unconditional grammar restriction for the densest Polignac gaps.
- STATUS: proved (one line); computed over all probes q' < 400 - d=2: 34/74 INCLUDING q'=41
  (reproducing the corridor result); d=4: 40/74; d=6 and d=12: 74/74; d=30: 72/72.
- WHERE: archive harvester round 13 sec 17.
- LIMITS: STRUCTURAL COMPENSATION - padding is 3x cheaper in absolute terms for `d = 0 mod 6` but
  can never repeat consecutively there; the two effects pull opposite ways.

### The 37->41 knife-edge (the corridor kills the expensive variant)
- STATEMENT The j=1 shape at 37->41 has two variants: literal 14 gives offsets 0, 41, 55, 96 ->
  mod 35 `[0,6,20,26]`, phases 12 and 32 OK; literal 27 gives offsets 0, 41, 68, 109 -> mod 35
  `[0,6,33,4]`, IMPOSSIBLE. So the cheap variant survives and the census turns on
  `F_3(37) >= 96` against a measured prefix of 95.
- CALCULATES: reduces a full census question to one unit of one spectrum value.
- STATUS: exact.
- WHERE: archive lateral rounds 15-16.
- LIMITS: the corridor cannot settle the knife-edge itself.

### The obstruction table (which steps are corridor- or spectrum-blocked)
- STATEMENT A shape is unobstructed iff corridor-feasible AND spectrum-affordable (`cost <= F_j(M)`,
  necessary because the run's gaps are consecutive gaps of M):
  `19->23 j=0 cost 46 need F_2 have 31 corridor OK short by 15; 19->23 j=1 cost 54 F_3 35 short by
  19; 23->29 j=0/1 58/68 corridor EXCLUDES; 29->31 j=0/1 62/72 corridor EXCLUDES; 31->37 j=0 74 F_2
  68 short by 6; 31->37 j=1 86 F_3 85 short by ONE; 37->41 j=0 82 corridor EXCLUDES; 37->41 j=1 96
  F_3 >=95 short by ONE; 41->43 j=0 86 F_2 OK; 43->47 j=0 94 F_2 OK`.
- CALCULATES: per-step feasibility of double padding.
- STATUS: exact/measured.
- WHERE: archive lateral round 17, research/padding_onset.py.
- LIMITS: the two one-unit near-misses in a row are flagged as an observation, not a law - "I have
  no mechanism for that".

### THE EXPOSED-SET AUTOCORRELATION c_q(g) (the three tooth-relationships)
- STATEMENT For gear q with exposed set `A_q = Z_q` minus its two teeth (`|A_q| = q-2`),
  `c_q(g) = |{ r in A_q : r + g in A_q }|` has the closed form
  `c_q(g) = q-2 if q | g` (same tooth), `= q-3 if g = +-2u_q (mod q)` (opposite teeth - THE
  LITERAL-LINK LAG), `= q-4 otherwise`. THE THREE CASES OF THE AUTOCORRELATION ARE THE THREE
  TOOTH-RELATIONSHIPS. Gear 5 (u=1, 2u=2) gives 3 / 2 / 1: lags `= +-1 mod 5` are suppressed
  threefold by gear 5 alone.
- CALCULATES: the number of phases keeping BOTH ends of a lag-g pair open; and by CRT the admissible
  endpoint phases mod 35 for a lag-g pair are exactly `c_5(g) * c_7(g)`, ranging over {3,...,15} - a
  five-fold swing driven entirely by the two smallest gears.
- STATUS: derived then brute-force verified over gears 5..31 at ALL lags, zero mismatches.
- WHERE: archive lateral round 18, research/exposed_autocorr.py.
- LIMITS: ENDPOINT half only - endpoint exposure is a CONJUNCTION so it factorises by CRT; the
  INTERIOR condition is a DISJUNCTION and does not factorise. "That is why it stopped there."

### What c_5*c_7 explains about missing stretch lengths
- STATEMENT Machine 23: `g 24/25/26/27/28/29/30/31 -> count 0/1404/310/170/322/6/112/20`, with
  `c5*c7 = 3/9/4/6/10/3/12/3`. Gap 24 (absent at machines 19 AND 23) and gap 29 (count 6 between
  neighbours 322 and 112) both carry the MINIMUM possible value 3; three of the four gap values
  absent below F across three machines carry the minimum.
- CALCULATES: which stretch lengths are structurally suppressed; adding `log(c_5 c_7)` to a
  smooth-decay fit raises R^2 from 0.449 to 0.463 (machine 19), 0.856 to 0.896 (23), 0.913 to 0.934
  (29) - about a quarter of what was called noise.
- STATUS: measured over full periods of machines 19, 23, 29.
- WHERE: archive lateral round 18.
- LIMITS: "no smooth law, only the histogram" is not right - the law is multiplicative and
  arithmetic rather than smooth. But residual demand vs purchasable supply leaves slack 8-16 at
  every g, so gap 24's absence is selection plus rarity, NOT a covering obstruction.

### THE n-POINT CORRELATION (one closed form subsuming three constructs)
- STATEMENT n points at offsets `d_1..d_n` are all exposed to gear q iff the phase avoids the 2n
  classes `{t - d_i}`; two classes coincide iff `d_i - d_j = 0 or +-2u`. Hence
  `c_q(d_1..d_n) = q - 2n + O`, `O = #{pairs with d_i - d_j = +-2u mod q}`, exact whenever
  `q >= 2n`. It subsumes n=1 (the exposed set, q-2), n=2 (the three-case form), and
  "`c_q > 0` forced when `q > 2n`" IS the completeness lemma.
- CALCULATES: the number of phases keeping any n named columns simultaneously open.
- STATUS: verified brute force, 16,500 checks over gears 5..43 and n = 1..5, zero mismatches.
- WHERE: archive lateral round 20, research/npoint_autocorr.py.
- LIMITS: exact only when `q >= 2n`.

### The padded-lag law (3q' +- 1 factorisation)
- STATEMENT Since `2u_q = 2*6^{-1} = 3^{-1} (mod q)`, the enhancement condition `g = +-2u_q` becomes
  `3g = +-1`, i.e. GEAR q IS ENHANCED AT LAG g IFF `q | 3g - 1` OR `q | 3g + 1`. So the arithmetic
  of a padded link is governed by the factorisation of `3q' +- 1`: `q'=23: 68/70 -> 5,7,17;
  29: 86/88 -> 11,43; 31: 92/94 -> 23,47; 37: 110/112 -> 5,7,11; 41: 122/124 -> 31,61;
  43: 128/130 -> 5,13`.
- CALCULATES: which gears enhance a given padded lag, from a factorisation.
- STATUS: verified, zero mismatches.
- WHERE: archive lateral round 20, research/padded_lag.py.
- LIMITS: HONEST - `sigma(q')` spans 3.3x while the measured supply share spans 330x, so endpoint
  arithmetic accounts for roughly a tenth of the erraticity; the rest is the interior.

### The interior disjunction, expanded and self-pruning
- STATEMENT `density(gap exactly g) = sum over T subset of interior of (-1)^|T| D({0,g} u T)` with
  `D(S) = prod_q c_q(S)/q` - an ALTERNATING SUM OF EXPOSED-SET CORRELATIONS, every term in closed
  form. Bonferroni truncation gives rigorous bounds (even depth upper, odd lower). THE CONSTRUCT
  PRUNES ITS OWN EXPANSION: `c_5(S) = 0` whenever the point set occupies 4 or more residues mod 5,
  so most subsets contribute exactly zero - g=20: depth 2 36% pruned, depth 3 70%, depth 4 89%,
  depth 5 97%; g=16: depth 3 90%, depth 4 98%, depth 5 99.6%.
- CALCULATES: the exact density of a gap of length g, term by term, rigorously truncatable.
- STATUS: exact object, measured convergence against exact full-period counts at machine 19 (g=8
  exact 10462, converging by depth 4; g=20 exact 142, depth 6 gives 402).
- WHERE: archive lateral round 20, research/bonferroni_gap.py, ie_pruning.py.
- LIMITS: depth needed grows roughly like `g/4` - "Brun's problem in the machine's own language,
  quantified".

### The word admissibility criterion (per-position forbidden phase)
- STATEMENT A side-word w is admissible iff some phase makes every prime side avoid every small-gear
  tooth. Letter L at position i forbids phase `u_q - i (mod q)`; letter R forbids `-u_q - i`. Each
  position forbids exactly one residue per gear and the per-gear allowed sets combine freely by CRT,
  so `w admissible <=> for every q the chosen residues do not cover Z_q`. Phase view: a column where
  the small machine hits BOTH sides admits no letter at all - and those B-columns are exactly the
  split/Bezout classes of gear pairs.
- CALCULATES: whether any side-word can occur, by residue covering per gear; the language is
  nonempty at length L iff the CRT period has an L-stretch free of B-columns.
- STATUS: exact (censused for gears <= 13; the criterion itself is gear-general).
- WHERE: archive lateral round 8, research/word_grammar.py.
- LIMITS: censused at gears <= 13.

### The word language is finite with a wall at 33
- STATEMENT Language census (gears <= 13, exact): `L 1/4/5/10/13/14/17/18..26/31/32/33 -> |lang|
  2/16/30/235/474/579/1176/~1140-1570/2560/2560/0`. All `2^L` words admissible through L = 4; first
  exclusions at L = 5 (LLLLL, RRRRR) - same-letter blocks cap at 4, gear 5's law. Growth is NOT
  exponential: the ratio falls to ~1.0 by L = 18 and the language PLATEAUS while `2^L` passes 10^9;
  EMPTY FROM L = 33 ON, matching the horizon computed independently from B-gaps.
- CALCULATES: how many side-words are possible at each length - a finite tree with a wall, the
  opposite in kind to the infinite gap-word antidictionary.
- STATUS: exact for gears <= 13.
- WHERE: archive lateral round 8.
- LIMITS: gears <= 13.

### Strict-alternation cap = 6 (gear 5 alone)
- STATEMENT Strict LRLR... saturated runs correspond to primes at alternating gaps 8,4,8,4,...; the
  offset residues mod 5 cover all of Z/5 at length 7 (L-first) and length 6 (R-first), so gear 5
  alone caps strict alternation at 6 columns (L-first) / 5 (R-first). Data: max strict alternation =
  6, at column 19125 at BOTH scales (another absolute landmark), letters LRLRLR - the L-first phase,
  exactly as the theorem requires.
- CALCULATES: an unconditional cap on strictly alternating stretches at any scale.
- STATUS: PROVED.
- WHERE: archive lateral round 7.
- LIMITS: strict alternation only; repeats like the landmark's LLLL are the norm - the constraint is
  CRT, not alternation.

### Word recurrence is CRT alignment, not chance
- STATEMENT Identical L=8 words recur at position differences divisible by 5 in 86% of duplicate
  pairs (baseline 20%), by 7 in 63% (baseline 17%), by 35 in 55% (baseline ~3%). The forced-letter
  fraction is 0.729 measured for gears <= 13 (crude CRT prediction 0.703). The landmark word
  RLLRRLLLLRLRL occurs exactly once in 1.67e7 columns; the six L=13 runs have six DISTINCT words at
  six different residues mod 35 (2, 3, 13, 17, 5, 18) - each uses a different corridor phase.
- CALCULATES: where a given side-word can repeat - only at congruent positions.
- STATUS: measured (y = 3163, 10007); all 757 observed words are admissible, zero failures.
- WHERE: archive lateral rounds 7-8.
- LIMITS: gears <= 13.

### Mirror and parity laws on words
- STATEMENT The positional mirror `k -> -k` reverses order and swaps L/R, so REVERSE-COMPLEMENT is
  the machine symmetry: TV distances at L=8, N=250 are 0.328 (reverse-complement) vs 0.564 (reverse)
  vs 0.600 (complement); letter marginal 0.4996 against the mirror's prediction 0.5. PARITY THEOREM
  (proved, two lines): an odd-length word cannot equal its own reverse-complement (its middle letter
  would equal its own complement), so ODD-LENGTH SATURATED RUNS ARE NEVER SELF-MIRROR - 0 odd
  palindromes observed (forced), while even-length self-mirror runs are common (16 of 250 at L=8).
- CALCULATES: which word symmetry to expect in any saturated-stretch census.
- STATUS: parity PROVED; the distribution claim measured.
- WHERE: archive lateral round 7, research/alternation_words.py.
- LIMITS: parity only for the proved half.

---

## PART 10 - Addresses: where a record stretch can sit

### LAW A - word-pinning (the neighbourhood word pins the address)
- STATEMENT The neighbourhood word of a near-top gap (openings within 20 columns each side)
  determines its address mod 385 almost uniquely: each opening must avoid both teeth of each small
  gear, forbidding 2 offsets per gear, and the ~10 openings around a top gap leave almost nothing.
  GEAR 5 IS PINNED TO EXACTLY ONE OFFSET BY EVERY NEAR-TOP WORD - 206/206 across all five machines;
  gear 7 unique for 94% (never more than 2); gear 11 unique for 90% (never more than 4); gear 13
  1-5 offsets; the full mod-385 address is UNIQUE for 87% and `<= 4` ALWAYS, at every machine.
- CALCULATES: `#top-stratum classes <= sum over near-top words of #phases(word) <= 4 x #words`; it
  converts "can two top-stratum classes be adjacent" into a finite grammar-level CRT check per word
  pair, with no period scan.
- STATUS: containment exact (0 fails in 206 words, machines 13-29, full periods), tightness 71-85%;
  UNIFORM in y. Observed class counts 6-14, flat, while gap counts swing 20-106.
- WHERE: archive lateral round 10, research/address_drift.py.
- LIMITS: the honest law is LOCAL - `address = pin(word)`, NOT `address = f(previous address)`. The
  remaining open piece is UNIFORMITY OF THE NEAR-TOP WORD GRAMMAR.

### Address pinning of the record stretch (machine-relative)
- STATEMENT Maximal gaps concentrate into 1-2 endpoint classes mod 35 (y=19: all twenty at left = 5,
  right = 30 = -5; y=23: {3,33}; y=29: {2,25}) and 2-6 classes mod 385 out of 135 available (~30x
  over baseline; top-200 gaps' endpoint classes concentrate up to 5.7x). At y = 23 and 29 the
  maximal gap is UNIQUE up to mirror.
- CALCULATES: a short candidate list of residue classes in which to look for a machine's record.
- STATUS: exact per machine (y = 13..29; streamed period 1,078,282,205 for y=29).
- WHERE: archive lateral round 9, research/topgap_corridor.py.
- LIMITS: unlike the absolute L*=13 landmark, the pinned address DRIFTS with the machine - gaps are
  machine-relative objects, saturated runs are absolute.

### Mirror closure of record stretches
- STATEMENT The set of maximal-gap intervals is closed under `k -> -k` at every machine tested:
  column 0 is a universal opening (the all-gears shield), so no gap straddles 0 and maximal gaps
  come in PROPER MIRROR PAIRS, with merged gap words appearing mirrored - e.g. (4,8,15,7)/(7,15,8,4)
  at y=23, (10,10,23)/(23,10,10) at y=29.
- CALCULATES: halves the record search - every record stretch has a mirror twin.
- STATUS: exact at every machine tested (full periods to y=23; streamed 1.078e9 for y=29). Record
  censuses show 4-20 maximal gaps per period, mirror-paired.
- WHERE: archive lateral round 9; archive constructor round 9 sec 20.4.
- LIMITS: mirror pairing of SITES does not imply mirror pairing of realized chains - the machine-29
  mirror does not commute with gear 31's teeth.

### Chain skeleton at the maxima: the interior spacings are the new gear's teeth
- STATEMENT Every new maximum `M_y -> M_y'` is a merge of old gaps by an alternating chain of the
  new gear: KILL SIDES STRICTLY ALTERNATE (R,L / L,R,L / R,L,R in every case) and the interior kill
  spacings are EXACTLY `{2u', q-2u'}` of the new gear (17: 6/11; 19: 13; 23: 8/15; 29: 10) - the
  chain condition's `{phi, phi+s}` law reconfirmed independently at the extreme tail.
- CALCULATES: the internal structure of a record stretch of the bigger machine from the new gear's
  teeth alone.
- STATUS: exact, y = 17..29.
- WHERE: archive lateral round 9.
- LIMITS: literal chains only - padded links (spacing 0 mod q') are outside it.

### The record of the bigger machine grows from MEDIUM old gaps
- STATEMENT Old-gap sizes under new maxima run 0.16-0.68 `F_old` (chains k = 2-3), except two y=19
  cases where an old MAXIMAL gap extends by k=1 (18+7). "F2 lives at medium gap pairs" is the
  generic regime; max-extends-max is the exception, not the rule.
- CALCULATES: which stratum of the smaller machine's gaps to search when predicting the next record.
- STATUS: measured, y = 17..29.
- WHERE: archive lateral round 9.
- LIMITS: it is exactly why the drift recursion fails - no near-top stratum tracks the medium
  spectrum.

### Near-top gap grammar: finite in structure, infinite in values
- STATEMENT The near-top gap language is NOT finite in absolute terms - gap values grow with y, with
  no 32-cap analogue for the top of the gap spectrum. What IS finite and stable is the RELATIVE
  grammar, three alphabets: FLANKS of near-top gaps in {1,2,3,4,5} columns at every machine tested
  (top flank pairs (2,2), (2,3), (1,3), (2,5)); CHAIN INTERIOR SPACINGS exactly
  `{2u'_q, q - 2u'_q}`; and near-top NEIGHBOURHOOD WORD COUNTS small and non-growing (14-42 distinct
  5-gap words per machine, no trend y=13 to 29). Top-gap neighbourhood =
  `[small flank] [medium gaps] [rigid chain skeleton]`.
- CALCULATES: the shape family to enumerate around a record.
- STATUS: measured (y = 13..29).
- WHERE: archive lateral round 9.
- LIMITS: the {1..5} flank alphabet was later SCOPED DOWN to a first-flank fact only - deeper parts
  track typical gap sizes and the max flank part grows 7 -> 13 with y. No corridor CAP on `F2 - F`
  was found.

### Two maximal stretches are never adjacent (class-level, mod 385)
- STATEMENT At every machine y = 13, 17, 19, 23 the top stratum occupies 4-6 classes mod 385, and
  the class-level adjacency test - is any r and r+F both a top-stratum left endpoint mod 385 -
  returns EMPTY at all four machines. Two maximal gaps can NEVER be adjacent, certified by class
  arithmetic alone (given the address census, one period scan each).
- CALCULATES: excludes the extreme adjacent configuration from residue data; the alpha1 certificate
  closes with a three-tier check - at y=13 with alpha1=1, 14 dangerous pairs die 5 by machine-free
  A3, 5 by mod-385 class disjointness, 4 by direct check (residual pairs (7,11), (8,10) and mirrors
  "class-compatible but unrealized"; budget `F2_k <= 16.67`, actual 16).
- STATUS: exact at machines 13, 17, 19, 23; the tier-A piece kernel-checked as
  `Machine13.no_11_11_chain` ("two maximal gaps are never adjacent at y=13" with NO period scan).
- WHERE: archive constructor round 10 sec 21; archive formalist round 11.
- LIMITS: the tier-C residual GROWS (4 at y=13 -> 96 at y=23), so scale needs mod-5005; uniformity
  in y still open.

### Minimum separation of record stretches (the measured non-clustering)
- STATEMENT Full-period record censuses give 4-20 maximal gaps per period (mirror-paired), and the
  MINIMUM SEPARATION between record gaps is 0.45-2.29% OF THE ENTIRE PRIMORIAL PERIOD (851,695
  columns at gears <= 23) - near-maximal gaps are astronomically anti-clustered in reality, five-plus
  orders beyond what the route needs.
- CALCULATES: the empirical separation floor between records inside one period.
- STATUS: measured (full periods, gears <= 11..23).
- WHERE: archive constructor round 9 sec 20.4.
- LIMITS: measured only - this IS the "Wall V" statement the route cannot prove.

### INTERIOR grammar: exactly 2 candidate words per chain length
- STATEMENT A merge word with k interior kills is side-alternating with spacing word alternating
  `sigma = 2u'_q` and `sigma-bar = q - sigma`, so the spacing pattern is determined by its initial
  type: EXACTLY 2 CANDIDATES PER k (`(s,q-s,s,...)` or its swap). Abstracting parts to c classes,
  `|shapes(k)| <= 2 c^(k+1)` - finite for each k, machine-independently. The interior grammar is
  finite iff `k_max` is bounded.
- CALCULATES: the complete literal-chain shape list at each fuel level.
- STATUS: exact.
- WHERE: archive lateral round 11, research/word_shapes.py.
- LIMITS: `k_max` grows (2,2,3,2,4 by step), so only the graded form is honest.

### BOUNDARY grammar: a finite a-priori superset, with no stabilisation
- STATEMENT Finite a priori but trivially so (compositions, `2^20 - 1` per half); CRT-admissibility
  cuts it to a machine-independent superset of 3798 half-shapes (enumerated exactly; pruning valid
  by monotonicity - extensions of inadmissible words stay inadmissible). CROSS-MACHINE FULL-SHAPE
  RECURRENCE IS ZERO at every machine (0/24, 0/20, 0/102, 0/30, 0/22); observed halves number 123 =
  3.2% of admissible and are essentially disjoint per machine. Mirror closure exact everywhere.
- CALCULATES: a fixed admissible family inside which extreme-value selection roams without
  repeating.
- STATUS: exact (superset), negative on stabilisation.
- WHERE: archive lateral round 11.
- LIMITS: a finite a-priori SUPERSET yes, an a-priori list of OCCURRING shapes no.

### The k=4 fuel-site census: arithmetic selects one of two permitted words
- STATEMENT Phase-free site census over machine 29's full period (1.078e9): EXACTLY 4 SITES with
  spacing word `(10,21,10)` - positions 220,171,102 (flanks 7,7), 406,081,827 (4,7), 672,200,337
  (7,4), 858,111,062 (7,7), two mirror pairs under the machine-29 mirror - and ZERO SITES for the
  grammar's other permitted k=4 word `(21,10,21)`. The grammar allowed two words; arithmetic
  selection realizes one. Pinning HOLDS for the k=4 object: the neighbourhood word pins the address
  to 3 phases mod 385 (`<= 4`), and the observed address is in the set. At 31->37 there are 216 k=4
  sites in BOTH orientations `(12,25,12)` and `(25,12,25)`, flanks in {1,2,3,5,6,10,11,13}.
- CALCULATES: the exact addresses of the deepest measured chains.
- STATUS: exact (machine 29 and machine 31 full periods).
- WHERE: archive lateral round 11, research/k4_pinning.py; archive mechanic round 11.
- LIMITS: `N5 = 0` everywhere scanned.

### Absolute landmarks versus machine-relative addresses
- STATEMENT Saturated runs are ABSOLUTE objects (primality-only), so every window sees the same
  integers and a window census is a truncation of the absolute list: `L* = 13` sits at columns
  2452-2464 (primes 14713..14783) at every scale, and a window whose bottom EXCLUDES a landmark
  inherits the next instance (y=50021 and 200003 both max at k = 61,501,443). Record GAPS, by
  contrast, are machine-relative and their pinned addresses drift.
- CALCULATES: the record saturated stretch in any window from one absolute scan.
- STATUS: exact (one absolute segmented scan of k = 1..1.2e10, zero truncation events beyond the
  flagged ones).
- WHERE: archive mechanic round 7; archive lateral rounds 6 and 9.
- LIMITS: exhaustive to member 7.2e10 (later extended to 1.67e11).

### Record-run addresses (exact column values)
- STATEMENT `L=10` at k=59 (member 353); `L=13` first at k=2452 (member 14711, word RLLRRLLLLRLRL),
  recurring at k=61,501,443; 874,166,593; 1,909,351,447; 8,472,005,085; 9,599,932,213. THE FIRST
  `L=14`: `k_start = 46,133,660,494`, members 276,801,962,963 .. 276,801,963,043, word
  LRRLRLRRRRLLRL, both boundary columns both-composite (maximality), Miller-Rabin verified.
- CALCULATES: the exact positions of the longest one-prime-per-column stretches found.
- STATUS: exact, exhaustive scan, independently verified.
- WHERE: archive mechanic rounds 7 and 9.
- LIMITS: L*=13 stood from member 1.5e4 to 2.8e11 - "a record on the curve, never a wall".

### Inside a record stretch: P-rate plus n2-rate = 1
- STATEMENT Inside every record run the twin-free identity is visible exactly: `P-rate + n2-rate = 1`
  per column (L=25 record: 0.80 + 0.20; L=100: 0.52 + 0.48). `L <= 13` records have zero interior
  doubles - pure n1, every column one prime plus one lone composite, so record runs are
  fragile-dense (every column a pseudo-twin column).
- CALCULATES: the composition of a saturated stretch column by column.
- STATUS: exact within the examined records.
- WHERE: archive lateral round 6.
- LIMITS: descriptive of records only.

### The load-length frontier is ABSOLUTE
- STATEMENT `maxload(L)` = max prime members per column over twin-free L-stretches:
  `L 1..13 -> 1.0000; 14 -> .9286; 16 -> .875; 20 -> .85; 25 -> .80; 32 -> .7188; 50 -> .60;
  100 -> .52; 200 -> .43; 478 -> .32`. IDENTICAL at y = 1009, 3163, 10007 because the record-holders
  are the SAME absolute integer landmarks. The frontier is a property of the integers; the window
  only truncates it from below at `s0 ~ y/6`.
- CALCULATES: the maximum prime load a twin-free stretch of a given length can carry.
- STATUS: exact/measured at three scales.
- WHERE: archive lateral round 6, research/load_frontier.py.
- LIMITS: renewability at depth is measured, not forced - persistence is a prime-constellation
  statement (imported corpus limit).

### Persistence as a Bertrand-type statement
- STATEMENT Because tower bands tile - interior(y) is columns `(y/6, y^2/6)` and interior(y^2)
  starts where it ends - `persistence(L)` ("every level-y open interior contains a saturated run of
  length L") is EQUIVALENT to: the increasing sequence `r_1 < r_2 < ...` of L-saturated-run positions
  satisfies `6 r_{n+1} - 1 < (6 r_n + 1)^2` ("the next run arrives before the square of the last").
  `persistence(1)` is a THEOREM (Brun); `persistence(2)` is disjunctive Polignac, OPEN;
  `persistence(L >= 3)` is disjunctive Hardy-Littlewood at tuple size L.
- CALCULATES: the exact provability frontier of the frontier curve; for each FIXED y the statement
  is a finite computation and verified bands stay verified.
- STATUS: theorem at L=1, conjectural for L >= 2.
- WHERE: archive lateral round 7.
- LIMITS: the frontier is a DESCRIPTIVE upper envelope, never a premise.

---

## PART 11 - The spectrum, per-J stretches, and how (D) was attacked

### The gap spectrum F_j (per-J stretch sums)
- STATEMENT `F_j(M)` = max sum of j consecutive gaps (`F_1 = F`, `F_2 = F2`). Rigorously
  `F(M+q') <= F_{k_max+1}(M)` and `excess <= F_{k_max+1}(M) - F2(M)`. Full-period spectra:
  machine 13: `11 16 23 26 28 31`; 17: `18 25 28 33 35 40`; 19: `25 31 35 38 47 50`;
  23: `34 39 50 58 65 77`; 29: `43 55 65 70 85 90`; 31: `58 68 85 90 92 97`; machine 41 (prefix):
  `110 112 118 123 130 138`.
- CALCULATES: bounds the new record purely from the old machine's gap word; increments stay
  `q/3`-scale (2-17) at every depth, no F-scale jump anywhere.
- STATUS: rigorous inequality; spectra exact at full period for machines 13..31 (machine 37 and 41
  rows are prefix LOWER bounds).
- WHERE: archive constructor round 10 sec 22; archive mechanic rounds 11 and 13.
- LIMITS: the criterion needs an UPPER bound on F_j, which every prefix row lacks - "which is why
  every prefix row reads 'not falsified' instead of 'verified'".

### THE TIER TABLE (which depth each step needs)
- STATEMENT Deleting k openings merges k+1 gaps, so a record `F(M+q')` is realizable only from
  chains with `F_{k+1} >= F(M+q')`. Per-step minimum chain length: 13->17 needs k=2; 17->19 k=1;
  19->23 k=2; 23->29 k=2; 29->31 k=2; 31->37 k=3. At 31->37 the record 88 EXCEEDS `F3(31) = 85`, so
  no k<=2 chain can reach it, and since the k=4 chains there reach only `<= 87` it is carried by a
  k=3 chain EXACTLY.
- CALCULATES: the minimum chain depth that can carry the record, from the spectrum alone - no word
  enumeration.
- STATUS: exact, full period.
- WHERE: archive mechanic round 13 sec (3).
- LIMITS: a threshold, not a density; tier and padding are independent axes.

### Merge availability: excess is not a function of fuel population
- STATEMENT `Correlation(excess share, N3 per opening) = -0.03` over seven steps. Zero long-chain
  fuel still yields substantial excess (23->29: `N3 = 0`, share 0.44 - pure k=2 merges) and huge
  fuel yields small excess (29->31: `N3 = 13000`, `N4 = 4`, share 0.20). MECHANISM: N2 is ubiquitous
  (2-5% of openings at every step), so k=2 merges are always available and excess MAGNITUDE is set
  by flank quality; chain length enters as a THRESHOLD (the tier table), never as a density.
- CALCULATES: separates the two channels - availability of merges versus size of flanks.
- STATUS: measured, negative result, seven steps.
- WHERE: archive mechanic round 13.
- LIMITS: seven steps.

### THE FLANK IDENTITY / bridge identity (a word plus its flanks IS a stretch)
- STATEMENT An occurrence of a length-`ell` word is `ell+2` CONSECUTIVE GAPS: left flank, the ell
  letters, right flank. Therefore `span(w) + FS(occurrence) <= F_{ell+2}(M)` IDENTICALLY, for every
  word, at every occurrence. Kernel form `Spectrum.merged_eq`:
  `g a + windowSum g (a+1) l + g (a+l+1) = windowSum g a (l+2)`, with `merged_le_spectrum` and
  `merged_le_of_shallow` deriving (D) at alpha=3 from `k_win <= 3` and `F_4 <= F + q'`.
- CALCULATES: converts (D) from a statement about flanks into SPECTRUM FLATNESS AT BOUNDED DEPTH.
- STATUS: identity (exact); kernel-checked (proofs/Spectrum.lean) - "its statement mentions NO fuel,
  NO k_max, NO words, NO residues, NO padding, only `g : N -> N`". Both empirical halves remain
  hypotheses inside the file, so the censuses decide the conclusion without the formal step being at
  risk.
- WHERE: archive constructor round 17 sec 34.1; archive mechanic round 17 sec (0),
  research/flank_envelope.py; archive formalist round 18.
- LIMITS: needs an UPPER bound on F_j; scans give lower bounds only.

### The length-only ceiling is ATTAINED
- STATEMENT At machine 19, word `(10,)`, over 9,452 occurrences, address `k = 137,328`, flanks
  `(21, 4)`: `span + FS = 21 + 10 + 4 = 35 = F_3(19)` EXACTLY. "Any attempt to sharpen
  `span + FS <= F_{ell+2}` into something smaller must use the word's letters, not just its length."
- CALCULATES: proves no better length-only bound exists.
- STATUS: exact, full period, single attaining address.
- WHERE: archive mechanic round 17 sec (2).
- LIMITS: one address.

### THE QUALIFYING SPECTRUM Q_j (the word-free criterion)
- STATEMENT `Q_j(M; a) = max sum of j consecutive gaps whose j-2 MIDDLE gaps are all >= a`, with
  `a = 2u' = 2*round(q'/6)`. Every qualifying word's merged stretch is such a sum, so
  `span + FS <= Q_{ell+2}`, and (D) is implied by the purely spectral, word-free inequality
  `Q_{ell+2}(M; a) <= F(M) + q'`. Full-period criterion margins `F + q' - max_j Q_j`: 11->13 +4,
  13->17 +10, 17->19 +9, 19->23 +10, 23->29 +13, 29->31 +3, 29->37 +9.
- CALCULATES: closes (D) at every measured step without enumerating words, residues or a corridor.
- STATUS: exact at full period (research/qualifying_spectrum.py), machines 11..31.
- LIMITS: needs an upper bound on Q_j, so prefixes can only falsify. HONEST COUNTERWEIGHT: the
  word-free margin COLLAPSES from ~0.45q' to 0.10-0.11q' at machines 29 and 31 (criterion margins
  +4/0.31, +10/0.59, +9/0.47, +10/0.43, +13/0.45, +3/0.10, +4/0.11, +23/0.17 prefix). The
  word-restricted margin does NOT collapse (0.52q' at 29->31).
- WHERE: archive mechanic round 17 addendum.

### THE CRUX AT 29->31: the size threshold alone does it
- STATEMENT `29->31: F_5 = 85 = F + 42` unrestricted - FAILS (42 > 31); `Q_5 = 71 = F + 28`
  qualifying only - PASSES (28 <= 31). "The interior-gap floor alone - one inequality, no
  compatibility, no residues, no corridor - brings 42 down to 28 and clears the budget with margin
  3. At the depth that actually binds, the suppression is carried by the SIZE THRESHOLD, which is a
  theorem, and it is already sufficient. Residue coincidence is not needed at this step."
- CALCULATES: a direct answer to the "arithmetic luck" caveat.
- STATUS: exact, full period.
- WHERE: archive mechanic round 17 addendum.
- LIMITS: margin 0.10q' is the honest caveat in the other direction.

### Q_j delivers the fuel cap for free
- STATEMENT `Q_j = 0` exactly when no qualifying word of length `j-2` exists, so the SAME object
  delivers the fuel cap: `Q_j = 0` for j > 5 at machine 19, j > 6 at machines 17, 23 and 29+q'=37,
  j > 7 at machine 29 + q'=31, j > 8 at machine 41. "The route's part (D) and its fuel bound are one
  measurement, not two."
- CALCULATES: chain-length cap and flank bound from one spectral pass.
- STATUS: exact, full period.
- WHERE: archive mechanic round 17 addendum; archive constructor round 20.
- LIMITS: `Q_j = 0` must be read as vacuous, not violated (a tool bug that read it as failure was
  caught and fixed; no published number was affected).

### Q_j is ATTAINED at the binding step
- STATEMENT `Q_7(31) = 88 = F(31->37)` EXACTLY - the qualifying spectrum at the winning depth equals
  the true record, so the bound is attained, not slack. Machine 31 full period (F=58, F+q'=95,
  a=12): `F_j = 85 90 92 97 104 110` versus `Q_j = 85 90 91 90 88 0` for j = 3..8 (drops 0, 0, 1, 7,
  16). Machine 41 (prefix, a=14): `F_j = 110 112 118 123 130 138`, `Q_j = 110 112 110 117 122 121`,
  giving `max_j Q_j = 110` against `F + q' = 133` (margin +23).
- CALCULATES: how much the interior-gap floor removes at each depth.
- STATUS: exact at machine 31 (full period 3.343e10, 725 s); machine 41 from a 0.08% prefix.
- WHERE: archive mechanic round 17 addendum 2.
- LIMITS: prefix rows are lower bounds.

### THE SPAN-RESOLVED ENVELOPE H_ell(s)
- STATEMENT With `H_ell(s)` = max flank sum over ALL runs of `ell` gaps with interior span exactly s
  (any letters), `span(w) + H_ell(span(w)) <= F + q'` implies (D) for w using only the word's LENGTH
  and SPAN. Evaluated on all 44 measured (step, compatible word) pairs: IMPLIED AT EVERY ONE,
  including the former residual - `29->31, w = (10,21,10), span 41: 41 + H_3(41) = 41 + 24 = 65 <=
  74`.
- CALCULATES: a second, sharper word-free criterion at no extra cost; it closes the residual word
  without looking at its occurrences at all.
- STATUS: exact on all 44 measured pairs.
- WHERE: archive mechanic round 17 sec (10).
- LIMITS: still needs upper bounds on the underlying envelope at prefix-only machines.

### THE INTERIOR-GAP FLOOR AND THE SHAPE OF THE UNRESTRICTED MAXIMISER
- STATEMENT A qualifying interior gap is `0 or +-2c mod q'` and positive, hence `>= 2u' = a` (a
  THEOREM, not a measurement). OF THE 132 MAXIMISERS CENSUSED AT MACHINES 19, 23 AND 29, ZERO ARE
  LITERAL AND ZERO ARE QUALIFYING: the shape is always TWO NEAR-MAXIMAL FLANKS PLUS THE MACHINE'S
  SMALLEST GAPS IN THE INTERIOR (2, 3, 4, 5, 7). Exhibited: machine 23 `F_3 = 50` flanks (23,23)
  interior (4,) at k = 2,082,580; `F_4 = 58` flanks (28,23) interior (4,3) at k = 29,098,935;
  `F_5 = 65` flanks (28,10) interior (5,2,20) at k = 36,845,450; machine 29 `F_3 = 65` flanks
  (39,23) interior (3,) at k = 407,599,253; `F_4 = 70` flanks (31,12) interior (4,23) at
  k = 717,564,717; `F_5 = 85` flanks (30,18) interior (4,3,30) at k = 772,741,833 and flanks (27,18)
  interior (3,7,30) at k = 725,859,998.
- CALCULATES: answers HOW the qualifying restriction suppresses the unrestricted maximum - by
  exhibition: the unrestricted maximum is attained by a shape the interior-gap floor forbids
  outright.
- STATUS: exact census (132 maximisers), research/unrestricted_max.py; the floor is a theorem.
- WHERE: archive mechanic round 17 sec (8); archive/agents-shared r19 SUMMARY.
- LIMITS: machines 19, 23, 29.

### Composition migration (the deep extremal stretch does not contain the record)
- STATEMENT The extremal j-stretch is not a huge gap with small neighbours - it MIGRATES to several
  medium gaps as j grows. Max element / sum at the argmax stretch: machine 17: 0.64 (j=3), 0.55,
  0.51, argmax composition `[3, 7, 18]`; machine 19: 0.51, 0.61, 0.53, `[10, 7, 18]`; machine 23:
  0.46, 0.48, 0.43, `[23, 4, 23]`; machine 29: 0.54, 0.44, 0.35, `[35, 20, 10]`. "This is why the
  isolation law does not control deep windows: the deep extremal windows never contain the record
  gap at all."
- CALCULATES: the composition of each machine's maximal j-stretch, position by position.
- STATUS: measured (machines 17, 19, 23, 29).
- WHERE: archive constructor round 19 sec 40.
- LIMITS: four machines.

### PAR TRADING (why deep chains never win)
- STATEMENT Each additional chain link buys about `q'/2` of span and costs about the same in flank
  sum, so the merged maximum is nearly INDEPENDENT OF DEPTH: merged spreads 0% (13->17), 8%
  (17->19), 6% (19->23), 14% (23->29), 12% (29->31). DERIVED, not observed: gain per link = spectrum
  increment (5-15), loss per link = `lambda*L` (4.2, 5.5, 9.0 at machines 19, 23, 29) - two
  independently computed quantities. Consequence `k_win <= 3` at all seven measured steps; winning
  words `(4), (6), (13), (8,15), (10), (10), (37,12)`.
- CALCULATES: predicts that the winning word is shallow, and explains the k_max/record decoupling.
- STATUS: measured 7/7; independently confirmed (spreads 8.8%, 9.3%, 5.2%; `k_win = 3, 2, 2`).
- WHERE: archive constructor rounds 18-19; archive mechanic round 17 sec (11).
- LIMITS: restated in round 20 as a BAND - "merged is depth-independent to within ~25%" (machine 31
  spread 22.7%, machine 41 9.3%). At 29->31 the k=3 and k=4 chains TIE at 55 while k=2 wins at 58 -
  fuel exists and LOSES by 3.

### k_win: the winner's depth and its exact address
- STATEMENT Maximum flanked merged span `ops[i+k] - ops[i-1]` per depth k: `19->23: k=1 31, k=2 33,
  k=3 34` (k_max 3, k_win 3); `23->29: 39, 43` (k_max 2, k_win 2); `29->31: 55, 58, 55, 55` (k_max
  4, k_win 2). WINNER ADDRESSES: `k = 137,307` (19->23, word (15,8)); `k = 14,995,460` (23->29, word
  (10,)); `k = 278,620,515` (29->31, word (10,)). The last is exactly the envelope census's word
  (10,) with `span 10 + FS_max 48 = 58` - two independent censuses agreeing on the record AND its
  address.
- CALCULATES: which chain depth carries the record, with the exact column.
- STATUS: exact, full period; research/kwin_census.py reproduces F = 34, 43, 58 and the round-11
  tuple counts.
- WHERE: archive mechanic round 17 sec (11).
- LIMITS: three steps; "the falsifying event to hunt is a single `k_win >= 4`". Later: `k_win = 3` at
  machine 31 and `k_win = 1` at machine 41 - winners get SHALLOWER as machines grow.

### The maximising flank pairs are MID-SIZE, never maximal
- STATEMENT The flank pairs that attain `FS_max` are MID-SIZE. At 29->31 the maximum `FS = 48` is
  attained at `(gL, gR) = (18, 30)` with F = 43 - neither flank maximal - and the largest single
  flank occurring anywhere in that word's occurrences is 35 = 0.81F. Across all 15 word-steps the
  largest single flank runs 0.16F to 0.81F and NEVER reaches F.
- CALCULATES: identifies the actual binding configuration for (D), and makes the both-flanks-maximal
  theorem off-target.
- STATUS: measured across all 15 word-steps.
- WHERE: archive constructor round 16 sec 32.
- LIMITS: it retires a line of work rather than opening one.

### The monotone envelope: true within a step, FALSE as a machine law
- STATEMENT (a) WITHIN a step's compatible word list the largest single flank falls monotonically
  with span - 19 of 19 measured word-steps, ZERO violations, and the fall is steep (0.81F at span 10
  to 0.16F at span 41 at 29->31). (b) AS A PROPERTY OF THE MACHINE it is FALSE - six violations with
  addresses, e.g. machine 29 span 21 -> max flank 27 (205,068 occurrences) versus span 25 -> max
  flank 30 (88,548 occurrences, `k = 133,490,560`); machine 29 span 29 -> 15 versus span 31 -> 22
  (`k = 661,321,007`); machine 19 span 8 -> 20 versus span 10 -> 21 (`k = 137,328`); machine 23 span
  27 -> 7 versus span 29 -> 8 (`k = 15,554,598`); machine 17 span 6 -> 12 versus span 8 -> 14.
  (c) UNCONDITIONALLY it is massively false.
- CALCULATES: kills "monotone envelope" as a route, and locates the real variable.
- STATUS: exact, all three verdicts, full period where reachable.
- WHERE: archive mechanic round 17 sec (3).
- LIMITS: "the within-step monotonicity is real but it is an ORDERING OF RARITY, not a law of
  position" - occurrence counts fall by 2-5 orders of magnitude across consecutive compatible spans
  (29->31: 7,815,766 / 205,068 / 6,500 / 4 across spans 10/21/31/41).

### THE FLANK ORDER-STATISTIC LAW (the flank follows occurrence count, not span)
- STATEMENT `maxflank(w) ~ 2.05 * ln(occ(w))` (mean 2.05, sd 0.27 over 12 word-steps) and
  `FS_max(w) ~ 2.77 * ln(occ(w))` (mean 2.77, sd 0.24), and 2.77 matches `lambda = 2.73` fitted
  INDEPENDENTLY from the window-sum tail at machine 29. "Longer words have fewer occurrences, and
  the largest flank grows like the log of the occurrence count; the apparent span-monotonicity was
  occurrence-count monotonicity all along."
- CALCULATES: predicts the maximal flank sum of any word from its occurrence count alone.
- STATUS: measured over 12 word-steps.
- WHERE: archive constructor round 20 sec 46.
- LIMITS: fitted, not derived; the one outlier is the 4-occurrence word (10,21,10), where order
  statistics has nothing to work with.

### Structural suppression on well-sampled words, and its exception
- STATEMENT Against an exact zero-parameter rarity null (2*occ flanks drawn from the machine's OWN
  gap histogram, max taken; effective null = min(rarity null, ceiling `F_{ell+2} - span`)), every
  well-sampled compatible word sits BELOW the null at p = 0.0000 and below the effective null too,
  by a deficit that GROWS with the machine: -1..-5 at machines 11-19, -7..-15 at machines 23 and 29.
  THE EXCEPTION: the word `(10,21,10)` at 29->31 sits at obs = 14 against null 15, `p = 0.4732` -
  its four occurrences behave EXACTLY like four independent draws; there is no structural
  suppression there at all, and its margin of +19 is a pure sample-size effect.
- CALCULATES: decomposes the observed envelope into (i) the spectrum ceiling (an identity, attained
  at machine 19), (ii) the rarity order statistic, (iii) a structural suppression of 7-15 gap units.
  ONLY (i) IS A THEOREM.
- STATUS: measured, full period; 14-row table.
- WHERE: archive mechanic round 17 sec (4).
- LIMITS: "a derivation of (D) for the long words cannot come from the monotone envelope and cannot
  come from the ceiling (too weak there: 44 vs the needed 33). It has to come from RARITY - an upper
  bound on the number of occurrences of a long compatible word, times a tail bound."

### (D) IN OCCURRENCE FORM
- STATEMENT Since `merged(w) = span(w) + FS_max(w)`, (D) reads
  `span(w) + lambda * ln(occ(w)) <= F + q'` for every compatible w, with lambda the flank-sum tail
  scale (~2.7, machine-computable) and `occ(w)` censusable directly and bounded above in closed form
  by `N x (exposure product)`, i.e. by the multi-lag `c_q`.
- CALCULATES: "the first form of (D) in which every term is a counting quantity with a closed-form
  upper bound: no extremes, no residue lottery, no fuel." Tests - machine 29 w=(10): span 10, occ
  7,815,766 -> predicted 53.3 (actual 58, budget 74); w=(21): 21, 205,068 -> 54.4 (51, 74);
  w=(10,21): 31, 6,500 -> 55.0 (55, 74); machine 23 w=(10): 10, 243,370 -> 29.7 (43, 63).
- STATUS: measured/derived at round 20.
- WHERE: archive constructor round 20 sec 47.
- LIMITS: the prediction under-shoots the actual at two of four tests.

### Suppression law and suppression-corrected flatness
- STATEMENT With lambda = the exponential scale of the stretch-sum tail and `L = ln(1/p_1)`:
  `suppression(j) := F_j - qualmax_j ~ lambda * (j-2) * L`, so
  `merged_max(j) ~ F_j - lambda (j-2) L`. PAYOFF: (D) follows from
  `F_j(M) - F(M) <= q' + lambda (j-2) L` for every `j >= 2`. Checked at 15 machine-depth pairs -
  machine 19 corrected 6.0, 5.8, 4.7, 9.5, 8.3; machine 23 5.0, 10.5, 12.9, 14.4, 20.8; machine 29
  12.0, 13.0, 9.1, 15.1, 11.1 - ALL 15 HOLD, BOUNDED AND NON-GROWING IN j (4.7 to 15.1), where RAW
  flatness fails at 5 of 15 and the raw values grow.
- CALCULATES: converts (D) into a single depth-indexed inequality over quantities computable from
  M's gap word alone - no words, residues, fuel, padding or extremes.
- STATUS: constructed/measured; machine 29 observed suppression 7, 15, 30 at j = 3, 4, 5 against
  predicted 9.0, 21.7, 42.5 - right scale, conservative at depth.
- WHERE: archive constructor round 19 sec 42.
- LIMITS: lambda is fitted from the tail, p_1 is measured, and the order-statistics step is
  heuristic. CONSEQUENCE ON RECORD: the j=2 case IS lemma 1 (suppression zero), so lemma 1 and the
  deep-window problem are ONE statement at different depths - AND THE DEEP CASES ARE THE EASIER
  ONES, reversing what the route assumed from round 8 to round 17.

### Shallow flatness (the fixed-depth test)
- STATEMENT A winner with `k <= 3` spans at most 4 consecutive gaps, so (D) at alpha = 3 follows
  from `[k_win <= 3] AND [F_4(M) - F(M) <= q']`. Measured `F_4 - F` = 11, 15, 15, 13, 24, 27, 32
  against q' = 13, 17, 19, 23, 29, 31, 37 - ratios 0.57-0.88, FLAT at 0.79-0.88 for six of seven
  machines with no downward trend. SHALLOW FLATNESS HOLDS AT ALL SIX MACHINES where deep flatness
  fails at three: "the round-17 refutation was a refutation of the WRONG DEPTH, not of the flatness
  idea."
- CALCULATES: a fixed-depth stretch test independent of fuel, k_max, words, residues and padding.
- STATUS: measured 6/6 and 7/7; neither half proven; machine 37 and 41 rows are prefix lower bounds
  ("not falsified").
- WHERE: archive constructor round 18 sec 39; archive mechanic round 17 sec (12).
- LIMITS: subsumed by the suppression law at round 19 - no separate winning-depth assumption needed.

### Strict ordering of the three lemmas
- STATEMENT `Wall V clustering (extreme x anything: F2 - F = O(q')) ==> SPECTRUM FLATNESS (all
  stretches of k+1 consecutive gaps) ==> (D) (only stretches whose interiors are qualifying values
  at compatible residues - a subfamily of relative density ~ (3/q')^{k-1})`. Hence "(D) cannot be
  weakened further by dropping position information, because the first such weakening is already
  false."
- CALCULATES: places any candidate lemma in the hierarchy.
- STATUS: exact implications; the middle one measured FALSE.
- WHERE: archive constructor round 17 sec 35.
- LIMITS: none.

### The compatibility restriction is load-bearing
- STATEMENT "The envelope does not follow from the spectrum. The qualifying/compatibility
  restriction is load-bearing, not cosmetic - it is precisely the difference between 42 and 15 at the
  one step where fuel is deepest. Any attempt to prove (D) by discarding the restriction loses the
  step it most needs."
- CALCULATES: quantifies what the residue restriction is worth (deep flatness gives F+42 where only
  q'=31 is allowed; the true increment is 15).
- STATUS: exact from the measured spectra plus the fuel census.
- WHERE: archive constructor round 17 sec 34.2.
- LIMITS: it refutes deep spectrum flatness.

### THE ANTI-CORRELATION LAW (an adjacency effect and nothing more)
- STATEMENT `R(lag) = P(both gaps qualifying at that lag) / p_1^2` from the landed census: machine
  11 (q'=13, p_1=0.0448) 0.000 at every lag 1-5; machine 13 (17) 0,0,0,0,1.149; machine 17 (19)
  0,0,0,0.979,1.355; machine 19 (23) 0.638, 0.622, 0.311, 0.365, 0.540; machine 23 (29) 0.039,
  1.897, 0.696, 1.043, 1.048; machine 29 (31) 0.148, 1.534, 0.807, 1.050, 1.033. THE LAW IS AN
  ADJACENCY EFFECT AND NOTHING MORE: a strong deficit at lag 1 (EXACT ZERO at machines 11-17 -
  qualifying gaps cannot be adjacent there at all), a rebound ABOVE independence at lag 2, and
  independence restored by lag 4-5. Arithmetic, not smooth. Higher orders are super-multiplicative:
  at machine 29 `p_5/p_1^3 = 7.1e-4` against the pairwise prediction `0.148^2 = 2.2e-2`, a further
  30x. Against independence `p_1^(j-2)` the deficits are x1.0, x1.6, x26 (m23 j=4), x6.7 (m29 j=4),
  x1400 (m29 j=5).
- CALCULATES: the joint qualifying probability at any lag - the object the suppression law needs.
- STATUS: measured from the joint gap-pair census, machines 11..29.
- WHERE: archive constructor rounds 19-20 secs 41, 45.
- LIMITS: measured, no closed form. AND (D) needs far less: independence alone clears every
  constrained case by 170x to 201,381x, so "(D) does not need the anti-correlation law - only that
  p_j is not POSITIVELY correlated by more than ~170x".

### The multi-lag exposure bound (the only step with no heuristic)
- STATEMENT "gap = v" is (both endpoints exposed) AND (no opening strictly between); dropping the
  second only increases the probability, and exposure is a CONJUNCTION so it factorises by CRT.
  Hence `p_j <= (1/rho) * sum over qualifying tuples (v_1..v_{j-2}) of prod_q c_q(0, v_1, v_1+v_2,
  ...) / q`.
- CALCULATES: a rigorous upper bound on the qualifying-stretch rate from exposure arithmetic alone -
  "the only inequality in the route with no heuristic step", offered as kernel-ready.
- STATUS: rigorous inequality; measured SHORT by a factor 2-29 (x2.0 at machine 29 j=5 and j=6,
  x28.8 at machine 23 j=6).
- WHERE: archive constructor round 20 sec 43.
- LIMITS: the missing factor is exactly the dropped "no opening strictly between" condition - THE
  RENEWAL FACTOR, named as the entire remaining gap and not built.

### The 1/rho conditioning correction (columns versus openings)
- STATEMENT `p_j` counts STRETCHES (it is per-opening) while the exposure product is per-COLUMN, so
  the bound must be divided by the machine density `rho = prod(1 - 2/q)`. Uncorrected, the bound
  appeared to clear the requirement everywhere tested; corrected, it does not.
- CALCULATES: the correct normalisation between opening-indexed and column-indexed quantities.
- STATUS: exact correction, self-reported - "reporting the uncorrected version would have claimed
  (D) closed word-free; it is not."
- WHERE: archive constructor round 20 sec 43.
- LIMITS: a bookkeeping law.

### THE HOLE LIST (which stretch lengths never occur)
- STATEMENT Exact, full period, first enumeration: machine 11 F=7 holes none; 13 F=11 {9}; 17 F=18
  {17}; 19 F=25 {19, 24}; 23 F=34 {24}; 29 F=43 {41, 42} (machine 31, from a separate run: 54, 56,
  57). Holes are RARE (0-2 per machine against 7-41 realised values) and sit at the TOP of the
  spectrum - 0.82F, 0.94F, 0.76F, 0.96F, 0.95F, 0.98F - with ONE exception, `v = 24` at machine 23,
  at 0.71F.
- CALCULATES: exactly which stretch lengths never occur between consecutive openings, hence which
  padding steps are impossible.
- STATUS: exact, full period, machines 11..29, research/hole_structure.py.
- WHERE: archive mechanic round 17 addendum 2.
- LIMITS: machine 37's prefix has not seen 69, but at 4.85% coverage that is INCONCLUSIVE, not a
  hole.

### HOLES HEAL: the spectrum fills monotonically from below
- STATEMENT `13 -> 17: 9 HEALED; 17 -> 19: 17 HEALED; 19 -> 23: 19 HEALED, 24 INHERITED; 23 -> 29:
  24 HEALED`. Five of six holes are filled by the very next gear; exactly one (v = 24) survives a
  step, and it survives the step where the two machines' F differ most. NO hole is ever CREATED
  below the previous machine's F, so the spectrum fills in monotonically from below as gears are
  added, and holes are a boundary effect the next gear repairs.
- CALCULATES: predicts that a hole blocking padding at one step is gone at the next.
- STATUS: exact, full period, four transitions.
- WHERE: archive mechanic round 17 addendum 2.
- LIMITS: four transitions.

### THE RESIDUE LAW OF THE GAP HISTOGRAM (the corridor teeth are legible in the whole machine)
- STATEMENT `hist_M[v]` is strongly non-flat in `v mod p` and the SHAPE IS STABLE ACROSS MACHINES
  AND CONVERGING (entries are class share x p, so 1.00 = flat): machine 11 mod 3 `0.58 0.67 1.75`,
  machine 29 mod 3 `0.65 0.93 1.42`; mod 5 machine 11 `0.83 0.90 2.22 0.83 0.21`, machine 29
  `1.16 0.80 1.70 0.93 0.41`; machine 29 mod 7 `0.78 0.90 1.64 1.15 0.67 1.40 0.45`. THE RICHEST
  CLASSES ARE THE LETTER VALUES OF THE SMALL GEARS: mod 7 the two richest are `v = 2` and `v = 5` =
  `+-s` for gear 7 (`s = 2*6^{-1} = 5 mod 7`); mod 5 the richest is `v = 2 = s`.
- CALCULATES: which stretch lengths the machine prefers, from the small gears' letter values.
- STATUS: measured, full period, machines 11, 17, 23, 29; every entry moves monotonically with the
  machine and is settling.
- WHERE: archive mechanic round 17 addendum 2.
- LIMITS: it is NOT the naive endpoint-survival count (which predicts `v = 0 mod p` richest and
  `+s`/`-s` equal; measured, `v = 2 mod 5` at 1.70 beats `v = 0` at 1.16). UNEXPLAINED. And the
  residue law does NOT predict the holes - scoring by `R(v) = prod_p share_p(v mod p)` for p <= 7
  hits at machines 13, 19, 23 but MISSES at 29 (holes 41, 42 rank 7 and 10 of 23) and flatly misses
  at 17 (hole 17 ranks 10 of 10, the HIGHEST score).

### THE COVERABILITY SPECTRUM COV(M) (the named construct, not built)
- STATEMENT A gap of exactly v at machine M means `v-1` CONSECUTIVE COLUMNS ALL STRUCK with both
  endpoints spared. That is not a residue-marginal question about v, it is a COVERING-FEASIBILITY
  question about the gear set, so the hole set is the complement of
  `COV(M) = { L : an interval of L consecutive columns is coverable by the gears 5..M, with both
  flanking columns spared }`.
- CALCULATES: (1) it is CRT arithmetic on the gear set, computable WITHOUT SCANNING THE PERIOD, so
  it reaches machines 37, 41, 43, 53 whose periods (1.2e12, 5.1e13, 2.2e15) are beyond any scan;
  (2) it therefore yields the UPPER bounds on F(M) and on the F_j that every prefix row lacks - "the
  single missing input for the qualifying-spectrum criterion at those steps"; (3) it joins the gap
  census to the pruned F(2,53) record search (which answers "is a run of length L coverable" one L
  at a time) and to the corridor, in one object.
- STATUS: NAMED CONSTRUCT, proposed at the end of round 17/19, NOT BUILT in the archive.
- WHERE: archive mechanic round 17 addendum 2; archive/agents-shared r19 SUMMARY.
- LIMITS: "That is my proposal for the next round ... not a bigger scan, a coverability spectrum."

### The route (A)(B)(C)(D)(E) and how each part was closed
- STATEMENT (A) WORD LIST - finite, computable from `q' mod 210` alone: PROVEN (rounds 11-12);
  formally PARTIAL - the class-reduction core is kernel-checked (`LiteralCap.s_eq`,
  `literal_chain_le_six`) but the ENUMERATION of the list is computed, not checked (the remaining
  gap). (B) LITERAL SPAN - literal chains have `<= 6` members, so `<= 5` letters, `span < (10/3)q'`:
  PROVEN (round 11), and now UNIVERSAL over every even gap (`PolignacCap.capOf_le_twelve`).
  (C) PADDED SPAN - each padded letter `>= q'`, count `p <= (F + (alpha/3)q')/q' ~ F/q'`, onset needs
  `F >= q'`: PROVEN (round 14 plus the onset gate, closed in the kernel at round 19).
  (D) FLANK BOUND - `FS_max(w) <= F + (alpha/3)q' - span(w)` for every compatible qualifying w:
  OPEN, the sole gap. (E) partial toward (D) - "both flanks maximal" is machine-free forbidden at 14
  of 16 word-step pairs: PROVEN (round 13), later recorded OFF-TARGET.
- CALCULATES: the whole increment `incr_k(M, q') = F_k(M+q') - F_k(M) <= (alpha/3) q'`, sufficient
  for the conjecture at alpha = 2.5 and alpha = 3.
- STATUS: as recorded at the end of round 19.
- WHERE: archive constructor round 15 sec 27.1; archive formalist round 18/19 audit.
- LIMITS: (D) is open for twins, hence for every even gap - it is THE SAME open lemma, d entering
  only through explicit finite constants.

### (D) is the hypothesis localised, not weaker
- STATEMENT By the word-indexed identity (D) is equivalent to `incr_k <= q'` - so (D) is NOT
  logically weaker than the hypothesis, it IS the hypothesis localised to `<= 6` pinned words per
  step. What alpha = 3 buys is room: the allowance rises by `q'/6` per word (17%), and the minimum
  margin over all measured word-steps rises from +0.83 to +7.
- CALCULATES: per-word margins at alpha = 3 (15 word-steps, margins +7.0 to +26.0); relative room
  0.19q' at the padded step, `>= 0.52q'` at every literal step.
- STATUS: exact equivalence; margins measured.
- WHERE: archive constructor round 16 sec 30.
- LIMITS: none.

### (D) as a mid-tail x mid-tail pair-sum bound
- STATEMENT NEEDED: the sum of two gaps at pinned separation `span(w)`, each observed at most 0.81F,
  is at most `F + q' - span(w)`. "This is a MID-TAIL x MID-TAIL pair-sum bound - weaker in kind than
  the extreme-value statements the route needed at rounds 8-13 (lemma 1 was extreme x anything;
  round 14's padded form was mid x extreme). It is the weakest form the requirement has taken."
- CALCULATES: nothing new - it restates the open target in its weakest form.
- STATUS: open at every step; margin `>= 0.19q'` measured.
- WHERE: archive constructor round 16 sec 33.
- LIMITS: still Wall V, no prime input.

### Margin trajectory (literal words)
- STATEMENT Minimum over each step's compatible words of `F + q' - span - FS_max`: 11->13 +12
  (0.923q', word (4)); 13->17 +10 (0.588, (6)); 17->19 +12 (0.632, (13)); 19->23 +14 (0.609,
  (8,15)); 23->29 +20 (0.690, (10)); 29->31 +16 (0.516, (10)). The absolute margin GROWS (+10 to
  +20); the relative margin sits in a flat band [0.52, 0.92]q' with no downward trend over six steps.
- CALCULATES: how close each step comes to violating (D), with the binding word named.
- STATUS: exact, full period, six steps.
- WHERE: archive mechanic round 17 sec (5).
- LIMITS: the PADDED tier at 31->37 has its own minimum, +7 = 0.19q' - a different object; both are
  recorded, neither is shrinking.

### The tolerance theorem (a per-step increment law closes the conjecture)
- STATEMENT If the increment law `F(M+q) - F(M) <= alpha*q` holds at every consecutive-gear step
  with q > 47, for ANY fixed alpha at or below `alpha*(y)` scale - in particular alpha = 2.5 or 3 -
  then `F(2,y) <= 354 + alpha*(S(y) - 328) < (y^2 - y)/2` for every prime `y >= 53` (S = prime sum).
  With `y <= 47` known directly this gives an opening in every window - twins infinite.
- CALCULATES: `alpha*(y) = [(y^2-y)/2 - 354]/[S(y) - 328]` = 5.64 at y=101, 8.71 at 1e4, 13.3 at
  1e6 - asymptotically `ln y`.
- STATUS: exact conditional theorem; checked at every prime y in [53, 1e6], zero failures, worst
  ratio 0.6557 at y = 113 (alpha = 3); beyond 1e6 by Rosser-Schoenfeld.
- WHERE: archive constructor round 8 sec 19.2, research/multiplicative_route.py.
- LIMITS: the open link is the mechanical statement "no consecutive step ever exceeds 2.5q"; the
  observed maximum is 2.432q at gear 37.

### The fourth wall (extreme-value control of sieve patterns)
- STATEMENT The obstruction is a wall distinct from abundance, localisation and parity: EXTREME-VALUE
  CONTROL OF SIEVE PATTERNS - "a statement about the machine's own gap word". The route evades the
  other three outright, and is not the parity wall by the dimension-1 test (the same increment
  statement for the ordinary one-residue Jacobsthal function would be sharper than Iwaniec's
  theorem, where parity does not obstruct). Classification rule: "an attack belongs to this event
  iff its missing input is a statement about the machine's own gap word (adjacency and alignment at
  the top of the gap distribution) with no prime-counting content."
- CALCULATES: classifies any candidate route.
- STATUS: verdict of round 8, filed as an amendment to the attempts map.
- WHERE: archive constructor round 8 sec 19.4; docs/proof-search/attempts-map.md amendment.
- LIMITS: names an obstruction; does not remove it. The two missing lemmas are TOP-GAP
  ANTI-CLUSTERING (`F2 - F = O(q)`) and FUEL-MERGE CONTROL (`excess = O(q)`), measured `<= 1.24q`
  and `<= 1.62q` at their separate maxima.

---

## PART 12 - The same alignment laws for every even gap, and the kernel forms

### Slot-cap at gap 2d (the transfer condition)
- STATEMENT `slot_cap_gap`: an odd prime blocking BOTH members of a gap-2d column forces `q | d`;
  `slot_cap_twin` is the d=1 case. "This is THE transfer condition: every corpus law whose proof
  rests on slot-cap holds verbatim for gap 2d at gears coprime to d, and the gears `q | d` collapse
  to one residue - the Hardy-Littlewood factor, mechanically."
- CALCULATES: how many teeth a gear has in the gap-2d pattern: `r_q = 1 if q | e else 2`.
- STATUS: kernel-checked (proofs/Polignac.lean); verified computationally first (d < 20, q < 100).
- WHERE: archive harvester round 1 sec 3.
- LIMITS: about one gear's teeth, not about coincidence across gears.

### Openings per period = the Hardy-Littlewood factor
- STATEMENT `|E_d| = prod over q in {3,5,7} of (q - r_q)`, `r_q = 1 if q | e else 2` - giving
  15/20/18/30/36/24/40/48 across the eight `gcd(e,105)` classes. "The Hardy-Littlewood factor and
  the exposed-set size are the same object."
- CALCULATES: the number of columns left open by the small gears, for any even gap.
- STATUS: exact (measured over every prime `q' <= 1200` coprime to 105, all 8 classes);
  independently reproduced in the kernel for all eight classes.
- WHERE: archive harvester rounds 9-10; archive formalist round 17.
- LIMITS: openings counted, not located.

### The universal literal cap over all Polignac gaps
- STATEMENT The cap depends only on `gcd(e,105)`: `gcd 1/5/7/3/21/35/15/105 -> cap 6/6/6/6/6/6/10/12`.
  Since 105 has exactly eight divisors, eight theorems cover EVERY even gap. `gcd = 3` is the
  `d = 0 mod 6` case - the densest Polignac gaps - and it still caps at 6. The ceiling breaks only at
  `gcd = 15` (10) and `gcd = 105` (12), exactly where e absorbs the small gears and enlarges the
  exposed set. 12 IS THE ABSOLUTE CEILING OVER ALL POLIGNAC GAPS.
- CALCULATES: the fuel bound for every even gap from `gcd(e,105)` alone.
- STATUS: kernel-checked - all eight `cap_gcd_*` AND `capOf_le_twelve` DEPEND ON NO AXIOMS AT ALL;
  each cap also checked numerically to be SHARP (the scan fails at cap - 1).
- WHERE: archive harvester round 10; archive formalist round 17, proofs/PolignacCap*.lean.
- LIMITS: gears 3, 5, 7 only; gear 3 FILTERS the candidate list rather than breaking runs (modelling
  it like gears 5 and 7 gives wrong caps 2/4 instead of 6/10/12).

### mod-105 = mod-210 (one class check for all gaps)
- STATEMENT The cap is a function of `q' mod 105` only, zero mismatches for every d tested, and
  `phi(105) = 48` - the same 48 classes. For ODD q', `q' mod 210` is determined by `q' mod 105`, so
  the mod-210 statement and the mod-105 one are THE SAME CHECK. One law, one class count, all d.
- CALCULATES: reduces any gear to one of 48 classes for alignment purposes.
- STATUS: exact, exhaustive over every prime `q' <= 1200` coprime to 105.
- WHERE: archive harvester round 10 sec 1.
- LIMITS: odd q' only.

### The walk step (the new gear's tooth offset)
- STATEMENT The walk step is `u'_d(q') = least positive representative of +-e*6^{-1} mod q'`; the
  twin case is `round(q'/6)`. Kernel: `6u' = q' -+ 1` gives `2u' = (q' -+ 1)/3`, and the discarded
  multiple of 210 contributes a multiple of 70, hence nothing mod 35 (`LiteralCap.s_eq`).
- CALCULATES: where the new gear's teeth land in the column frame, as one closed-form residue.
- STATUS: exact / kernel-checked.
- WHERE: archive harvester round 9; archive formalist round 13.
- LIMITS: the `round(q'/6)` form is the d = 2 case.

### Literal chain = an interleaved two-phase walk of period 70 mod 35
- STATEMENT A literal chain is an interleaved two-phase walk with PERIOD 70 mod 35, so the cap is a
  function of `q' mod 210` ONLY - the same 48-invertible-class finite check, per gap; class
  invariance verified per d over ~300 primes each, zero mismatches.
- CALCULATES: reduces an unbounded chain question to a finite residue walk.
- STATUS: exact.
- WHERE: archive harvester round 9 sec 2.
- LIMITS: requires the gear-3 skip semantics.

### Single-cycle reduction (one walk sees every alignment state)
- STATEMENT The walk's state space `(position mod 105, parity)` is a SINGLE CYCLE OF LENGTH 210,
  because two steps advance the position by t and `gcd(t,105) = 1`. So ONE walk of 260 steps from a
  single start sees every state, replacing `105 x 2` starts by one. Prerequisite lemma
  `exists_mul_mod_eq : 0 < n -> Coprime t n -> r < n -> exists j < n, (j*t) % n = r`.
- CALCULATES: every opening residue is reached by some multiple of the tooth step, so a single orbit
  enumerates all alignments (a 37x cut in the cap computation).
- STATUS: reduction verified numerically EXACT (single-walk max run = all-starts max run, zero
  mismatches over all 8 gap classes x 48 classes of t); the enabling lemma is kernel-checked.
- WHERE: archive formalist rounds 16-17.
- LIMITS: not used for the cap in the end (restricting starts to the exposed set was cheaper); kept
  as a reusable piece for any machine out of scan reach.

### The word identity's shape transfers verbatim, and the 1-letter word always fires
- STATEMENT W1 (the merge/tier decomposition reproduces `F(M+q')` exactly): Y in 13/13. W2 (the
  identity's shape, `F(M+q') = max(F2(M), max over k>=2 tiers)`): Y in 13/13, INCLUDING every
  `d = 0 mod 6` case and both degenerate cases; and `tier_1 = F2(M)` EXACTLY in every row
  (33/48/75/30/48/22/30/45/27/18/35) - "the 1-letter word always fires", verified for every d. "The
  lower-bound mechanism rests on `gcd(P_M, q') = 1`, which contains NO d at all: it transfers
  verbatim, and so does the identity's shape."
- CALCULATES: the record of the next machine as a max over tiers of the current one, for any gap.
- STATUS: measured exact, 13 configurations (d = 2, 4, 6, 10, 12, 30 plus degenerate `q' | e`).
- WHERE: archive harvester round 10 sec 2.
- LIMITS: needs the phase/copies condition `gcd(P_M, q') = 1`.

### Realized chain letters are sums of consecutive frame letters
- STATEMENT W4: the realized chain letters are sums of consecutive frame letters - twins q'=17
  realized letters `{18,33}` = the frame letters exactly; e=6 q'=17 realized `{6,11,23}` with
  `23 = 11+6+6` a padded link.
- CALCULATES: the alphabet of realizable chain steps from the frame letters.
- STATUS: measured, after correcting a wrap-around extraction artefact.
- WHERE: archive harvester round 10 sec 2.
- LIMITS: extraction must use absolute positions over two periods.

### A degenerate gear (q' dividing e) has ONE tooth
- STATEMENT W5: when `q' | e` gear q' has ONE tooth, the frame letter set collapses to the single
  value `3q'` (39 at q'=13, 51 at q'=17), chains become plain arithmetic progressions, no `k >= 2`
  tier is needed, and `F(M+q') = F2(M)` exactly in both cases. The identity survives trivially; the
  word grammar degenerates.
- CALCULATES: the record of the extended machine directly when the new gear divides e.
- STATUS: measured, both degenerate cases exact.
- WHERE: archive harvester round 10 sec 2.
- LIMITS: only the `q' | e` case.

### Literal span law (two primitive letters sum to the frame period)
- STATEMENT The two primitive literal letters sum to the frame period (twins, q'=41: `42 + 81 = 123
  = 3q'`; d=6, q'=41: `3 + 38 = 41 = q'`), so k letters span `ceil(k/2)` periods. With the universal
  cap table: `literal span <= ceil((cap_d - 1)/2) x q'` - `cap_d = 6` gives `<= 5` letters and
  `<= 3q'`; `cap_d = 10` gives `<= 9` letters, `<= 5q'`; `cap_d = 12` gives `<= 11` letters,
  `<= 6q'`.
- CALCULATES: the total span a literal chain can occupy, from the cap.
- STATUS: exact/verified (research/route_transfer_audit.py).
- WHERE: archive harvester round 13 sec (B).
- LIMITS: "the constant degrades by at most a factor 2 across ALL Polignac gaps."

### THE COMPLETE MOD-3 DICHOTOMY (when the record is forced to be a multiple of 3)
- STATEMENT `3 | F_d(y) for every gear set <=> 3 does not divide e <=> d != 0 mod 6`. MECHANISM: a
  gear blocks `n = 0` and `n = -e`, two residues COLLAPSING to one exactly when `q | e`. At `q = 3`
  that is decisive because `3 - 2 = 1`: when `3` does not divide e gear 3 leaves a SINGLE class, so
  every opening is congruent mod 3 and every gap - the maximal one included - is a multiple of 3.
  When `3 | e` it leaves two classes and openings one apart exist. For `q >= 5` at least `q - 2 >= 3`
  classes survive, so NO GEAR ABOVE 3 CAN PIN OPENINGS TO ONE RESIDUE. The dichotomy is complete and
  sharp at gear 3.
- CALCULATES: for any even gap, whether the record is forced to be a multiple of 3; the twin form is
  `F(2,y) = 0 mod 3`, and every length not `= 0 mod 3` below F is coverable without search, so the
  first uncoverable multiple of 3 IS F - cutting 2/3 of all coverable increments in a record search.
- STATUS: kernel-checked (`three_survivors_congr`, `three_dvd_gap`, `three_survivors_adjacent`,
  `no_mod_law_above_three`, `endpoint_run_mod_three`, `F_zero_mod_three`, `M_two_mod_three`,
  `not_max_of_mod_three`, all on [propext, Quot.sound]); verified exhaustively over full periods,
  machines y = 11..23, 15 gap classes (F_2 = 21, 33, 54, 75, 102 all divisible by 3 against
  F_6 = 16, 28, 39, 57, 65, none forced); all thirteen known F(2,y) values comply.
- WHERE: archive harvester rounds 8, 9, 15; archive formalist round 11, proofs/MaxGap.lean.
- LIMITS: gear 3 only; no analogue at any higher gear.

### Left-taut equivalence (a gear never strikes the column just left of a covered run)
- STATEMENT Fix gears Q and `L >= 1`. `Cov(L)` (some offset assignment strikes every column of
  `[0,L)`) holds IFF there is such an assignment additionally leaving column `-1` unstruck by every
  gear. Proof: take M >= L maximal with Cov(M); its witness cannot strike `-1`, else the run
  `[-1, M)` of length M+1 is covered, contradicting maximality. CONSEQUENCE: every gear may drop its
  two offsets `q-2, q-1`, so gear q never strikes columns `= -1 mod q`.
- CALCULATES: a per-L pruning rule that collapses the branch factor at every leftmost-uncovered
  column `= -1` or `-2 mod q`, including inside the final uncoverable certificate.
- STATUS: verified exhaustively over ALL offset tuples, y = 11/13/17, at EVERY L from 1 to F+2, zero
  mismatches.
- WHERE: archive harvester rounds 8-9, research/lefttaut_check.py.
- LIMITS: it is UNSOUND combined with the mirror-canonical `o5` halving (reflection maps left-taut
  to RIGHT-taut coverings); the canonicalisation was removed.

### Machine certificates: the record and pair-record of a whole machine, in the kernel
- STATEMENT Machine 13: `gap_le : b - a <= 11` (F_k(13) <= 11); `pair_sum_le : c - a <= 16`;
  `gap11_realized` (openings 122, 133 with nothing between); `pair16_realized` (openings 117, 122,
  133, gaps 5, 11); `alpha1_certificate : 3*(c-a) <= 3*11 + 1*17`; `lemma1_at_13 : (c-a) - 11 <= 17`;
  `tierA_forbidden` (allowed3 of (6,11), (8,11), (11,6), (11,8), (11,11) all empty);
  `no_11_11_chain`. Machine 17: `gap_le : b - a <= 18`; `pair_sum_le : c - a <= 25`;
  `alpha1_certificate : 9*(c-a) <= 9*18 + 4*19` (225 <= 238); `lemma1_at_17`. The 25 is tight - 24
  fails.
- CALCULATES: the exact record and pair-record of two whole machines, with explicit witness columns.
- STATUS: kernel-checked; `Machine13.w11` and `w16` DEPEND ON NO AXIOMS AT ALL, and Machine 17's
  `w18All`/`w25All` (an 85085-tuple period scan) need ONLY `[propext]`.
- WHERE: archive formalist rounds 11 and 15.
- LIMITS: kernel-evaluation cost, not mathematics, is the barrier past this size.

### The CRT-tuple scan technique (how alignment questions get decided in the kernel)
- STATEMENT A direct `decide` over residues mod 5005 DOES NOT TERMINATE. The fix that made the round
  possible: quantify over the CRT TUPLE `forall a < 5, b < 7, c < 11, d < 13` with the opening test
  `expT a b c d` and shifts taken modulo each gear separately. Same 5005 cases, but every modulus is
  a single digit and the decision tree has depth `<= 13` instead of 5005. "This is the general recipe
  for any machine whose period is a product of small primes."
- CALCULATES: makes "is this column open under all gears" a per-gear coordinate test.
- STATUS: technique, kernel-validated.
- WHERE: archive formalist round 11.
- LIMITS: keep each DECLARATION at or below roughly `5x10^3` tuples, and bound the number of heavy
  declarations per module.

### Placement: which column a gear-multiple occupies, and injectivity
- STATEMENT `prime_mod_six`, `sign_law : (a*b) % 6 = 1 <-> a % 6 = b % 6` on unit classes, and
  `slotOf m = (m+1)/6` - ONE formula recovers the column from EITHER member. `slot_injOn_partners`:
  `c -> slotOf (q*c)` is injective on a gear's partner set, so one gear's line occupies exactly
  `R_q` distinct columns; the mixed-sign collisions die on `Layer.slot_cap`.
- CALCULATES: given q and c, which column `q*c` sits in and on which side (left member iff the two
  factors have opposite residue mod 6); and that a gear never strikes the same column twice.
- STATUS: kernel-checked (proofs/Placement.lean; `sign_law` needs only `propext`).
- WHERE: archive formalist round 8.
- LIMITS: regime `q` prime `>= 5`, members `< q^3`; column 0 is degenerate.

### Semiprime refinement: in the large-gear regime a gear's line IS a prime count
- STATEMENT `semiprime_of_fiber`: `q` prime, `1 < m < q^3`, m composite with `minFac m = q` implies
  `m = q*c` with c prime and `q <= c`. Hence `R q S = #{c prime : q <= c, q*c in S}`.
- CALCULATES: exactly which columns a large gear strikes inside the window - one per partner prime.
- STATUS: kernel-checked; regime is every member `< q^3` (window form `y^2 <= q^3`, gears
  `q >= y^(2/3)`).
- WHERE: archive formalist round 7.
- LIMITS: the boundary case is the square - `m = q^2` is rooted at q but is not `q*c` with `c > q`,
  so the decomposition is stated with `q <= c`, equality exactly at the shadow-law onset.

### six_mul_class and card_class_Ico (the two primitives under every position law)
- STATEMENT `six_mul_class`: for any m coprime to 6 and any target residue c, `{k : 6k = c mod m}`
  is EXACTLY ONE class mod m. `left_dvd_iff` / `right_dvd_iff`: member divisibility is a residue
  condition (`6k = 1` left, `6k = m-1` right). `card_class_Ico`:
  `#{k in [1,t] : k = a mod m} = (t + m - a)/m` for `1 <= a <= m`.
- CALCULATES: turns "gear m strikes the left/right member of column k" into a single residue class,
  and counts how many columns of a prefix lie in it. Every floor term of the supply formula reduces
  to these two.
- STATUS: kernel-checked; verified computationally first over 105 prime pairs.
- WHERE: archive harvester round 3, proofs/Polignac.lean.
- LIMITS: requires coprimality to 6.

### SAME-side and split classes (both gears on one member, or one each)
- STATEMENT `same_left_census` / `same_right_census`: for distinct primes `q, r >= 5` the columns
  whose left (resp. right) member both gears divide are ONE CRT class mod qr with count
  `(t + qr - a)/qr`. `split_class`: the columns where q strikes the LEFT member and r the RIGHT are
  exactly ONE CRT class mod qr, with the joint target residue `c = CRT(1 mod q, r-1 mod r)`; the
  mirror class swaps roles. `twoSided_class`: for coprime moduli `mL, mR > 1` both coprime to 6, the
  columns with `mL |` left and `mR |` right are ONE class mod `mL*mR` - this subsumes `split_class`
  and yields EVERY both-sided term of the master formula in one statement.
- CALCULATES: the exact columns at which two (or more) gears jointly close a column, from either
  side, with a closed-form count.
- STATUS: kernel-checked; verified first over 105 pairs, 210 ordered pairs, and 60 two-sided triple
  cases, zero fails.
- WHERE: archive harvester rounds 3-5.
- LIMITS: requires coprimality to 6 and between the two sides.

### The three-gear master identity (26 terms, subtraction-free)
- STATEMENT `three_gear_master`: for any distinct odd primes q, r, s over the first t columns,
  `distinct + 12 pair side classes = 6 single side classes + 8 triple side classes`. Every term
  beyond "distinct" is ONE CRT class whose count is closed-form floor arithmetic. "With this, THE
  ASSEMBLY LINE FOR 3 GEARS IS CLOSED formally end to end." Supporting bridges: `card_marks_eq`
  (per-gear, disjoint by slot cap), `card_pair_inter_eq` (four disjoint side classes LL/LR/RL/RR),
  `card_triple_inter_eq` (eight disjoint side classes LLL..RRR), plus `three_sets_ie` (n=3
  inclusion-exclusion, subtraction-free, any finsets).
- CALCULATES: the exact number of distinct columns three gears leave open over a prefix, as closed
  floor arithmetic.
- STATUS: kernel-checked; verified first over 5 gear triples x 5 window lengths to t = 5005, zero
  fails.
- WHERE: archive harvester rounds 5-7.
- LIMITS: three gears; `n > 3` was assessed and deliberately deferred.

### Per-gap reduction (every even-gap conjecture is a statement about openings in a window)
- STATEMENT `SurvivorGap d y m` (the gap-2d opening predicate); `survivorGap_one_iff` (d=1 is
  definitionally the twin case); `survivorGap_iff_pair` (windowed opening `<=>` prime pair at gap
  2d); `gapPairs_infinite_iff_survivor_in_window (d)` - the PER-GAP IFF, both directions, every d
  (d=0 degenerates gracefully to infinitude of primes). Companion: the Goldbach window reduction
  with its exact converse on central representations.
- CALCULATES: turns every even-gap conjecture into a statement about openings inside the window.
- STATUS: kernel-checked, standard axioms (`survivorGap_one_iff` needs only `[propext]`); verified
  computationally first (d in {0,1,2,3,5,6}, y in {13,23,47}; all even N < 2000 for Goldbach).
- WHERE: archive harvester round 1 sec 3, proofs/Polignac.lean.
- LIMITS: an equivalence, not progress.

### The delta-profile (recognising a record-maximising gap from its tooth separations)
- STATEMENT For a difference e and gear q, `delta_q(e) = min(e mod q, q - e mod q)` is the
  separation between gear q's two struck residues 0 and -e - `delta_q = 1` means the teeth are
  ADJACENT (blocking clustered), `delta_q ~ q/2` means maximally SPREAD. THE TWIN DIFFERENCE HAS
  `delta_q = 1` FOR EVERY GEAR: twins are the maximally clustered member of the family, at every
  scale at once. Winners: `3,5,7,11 -> maxF 33, 8 winners, profile (1,1,1,3)`;
  `+13 -> 75, 16 winners, (1,1,1,3,6)`; `+17 -> 96, 64 winners, (1,1,2,4,6,8) and (1,1,2,3,4,3)`. At
  gears <= 13 the rule is exactly "e is extremal iff `(delta_3..delta_13) = (1,1,1,3,6)`", satisfied
  by 16 of 7507 differences, all 16 attaining F = 75.
- CALCULATES: recognises a record-maximising difference from its tooth-separation profile alone.
- STATUS: exact, exhaustive; precision 100% in all three cases, recall 100% up to gears <= 13.
- WHERE: archive harvester round 17 sec 1.
- LIMITS: "maximally spread at the top" describes some maximisers, not all - the co-winning shape
  (1,1,2,3,4,3) has top entry 3 against a maximum of 8.

### Extension versus compromise (why the record jumps at some steps)
- STATEMENT The winning profile at 13, `(1,1,1,3,6)`, is the winning profile at 11, `(1,1,1,3)`,
  with the new gear's entry appended AT ITS MAXIMUM value 6 - the optimum EXTENDS with no concession
  below. At 17 it cannot: the winning profiles move `delta_7` from 1 to 2 and `delta_11` from 3 to 4
  (or to 3). So adding gear 13 bought the family maximum its full value (maxF x2.27, 33 -> 75) while
  adding gear 17 bought much less (x1.28, 75 -> 96).
- CALCULATES: predicts whether a machine step gains its full record increment or a compromised one.
- STATUS: exact for the computed machines; PERSISTENCE measured - the champions of gears <= 13 land
  at the 99.3-99.8th percentile at gears <= 17 but none stays maximal, while the champion e = 344 of
  gears <= 11 becomes THE maximiser at gears <= 13.
- WHERE: archive harvester round 17 sec 2.
- LIMITS: three or four machines.

### The paired-Jacobsthal values, and where twins sit in their own family
- STATEMENT First exact `h_2` values, exhaustive over every even difference:
  `y=5: 18; 7: 30; 11: 66; 13: 150; 17: 192` against the Ziller-Morack Conjecture 6 bound
  `p_n^2 - p_n = 20, 42, 110, 156, 272` - HOLDS at all five, but the MARGIN IS NON-MONOTONE WITH A
  ONE-OFF DIP AT 13 (3.8% against 10.0, 28.6, 40.0, 29.4). Maximising differences at y = 13:
  e = 344, 734, 839, 916, 2164 - all coprime to P, none small, none structured.
  TWINS ARE THE EASY END: at gears <= 13, among the 2,880 differences coprime to P, F ranges 30..75
  and the twin difference gives 33 - THE 13.3rd PERCENTILE, with 77.2% of coprime differences having
  a LARGER record and the extremal one 2.27x the twin value (1.78x at gears <= 17, 21st percentile).
- CALCULATES: locates the twin problem inside the paired-Jacobsthal family; "prove it for twins" is
  strictly the easy end of "prove it for every even difference", by a factor > 2.
- STATUS: exact, exhaustive (e and P-e are reflections, so e = 1..P/2 is complete).
- WHERE: archive harvester round 16.
- LIMITS: up to y = 17; y = 19 only a lower bound.

### Density does not determine the record
- STATEMENT Opening density is a pure function of which gears divide e, so the mean gap is exactly
  computable per class. If the Jacobsthal heuristic captured the d-dependence, `F_max/lambda` would
  be near-constant across classes. IT IS NOT: over the 31 `gcd(e,P)` classes at gears <= 13 it ranges
  `2.88 (gcd = 5005) .. 7.52 (gcd = 3)` - a factor 2.6 spread. TWO DIFFERENCE CLASSES WITH THE SAME
  MEAN GAP CAN DIFFER BY MORE THAN 2x IN THEIR RECORD.
- CALCULATES: rules out predicting the record from opening density alone.
- STATUS: exact over all 31 classes at gears <= 13.
- WHERE: archive harvester round 16 sec 3.
- LIMITS: one machine.

---

## PART 13 - Frames, the covering view, and the counting routes

### The CRT collapse: there is no design freedom
- STATEMENT By CRT the offset vector `(o_q)` is exactly a single residue `c mod P`, so the uncovered
  set is a TRANSLATE of the single pattern `{ n : n and n+1 both coprime to P }`. "The P offset
  vectors are precisely the P translations of one pattern, and nothing else." Hence
  `F(2,y) = 1 + the maximum gap of { n : n, n+1 both coprime to P(y) }` - the Jacobsthal-type
  function for the pair {0,1} modulo the primorial.
- CALCULATES: reduces the whole target to the maximum gap of ONE explicit pattern, with no design
  search.
- STATUS: proved (a short CRT argument); VERIFIED - the all-offsets-1 configuration attains `F(2,y)`
  exactly at y = 7, 11, 13, 17, 19, 23, 29, ratio 1.000 in all seven cases, the last by segmented
  sieve over all 3.2e9 positions.
- WHERE: docs/status.md section 4a; docs/handover.md section 3 item 17.
- LIMITS: consequences - "the exhaustive offset searches explore TRANSLATIONS, not designs";
  "'extremal configurations are efficient' is automatic - they are all the same configuration";
  "the obstruction is not combinatorial fitting". Any argument appealing to choosing offsets cleverly
  is appealing to a translation.

### The two frames are one machine, scaled by 3
- STATEMENT Gear 3 blocks one of any two adjacent positions, confining the exposed set to a single
  class mod 3; rescaling that class by 3 leaves each gear `q >= 5` blocking two residues separated
  by `3^{-1} mod q` - which IS the column-frame separation, since `2 * 6^{-1} = 1/3`. So THE
  ADJACENT FRAME IS THE COLUMN MACHINE, SCALED BY 3: `F(2,y) = 3 F_k(y)` and
  `F2_adjacent = 3 F2_k`.
- CALCULATES: transfer of every result between frames with `L -> 3L`.
- STATUS: exact, verified for seven gear sets (F) and six (F2): 15, 21, 33, 54, 75, 102, 129 in the
  adjacent frame against 5, 7, 11, 18, 25, 34, 43 in columns.
- WHERE: docs/gear-recursion.md section 1; docs/handover.md section 0.5.
- LIMITS: consequence - the adjacent-frame case `L = 1` has NO counterpart in the column frame, so
  `h(1) = d/(1-d)` is a grid artefact and `min_L h(L) = h(1)` is STRONGER than the conjecture needs;
  in column space the minimum of `h/d` sits at `L = 2`.

### The separation family (which claims survive generalisation)
- STATEMENT Generalise: gear q blocks two residues at arbitrary separation `s_q`. The adjacent frame
  is `s_q = 1`, the column frame is `s_q = 3^{-1}`. F depends on the SEPARATION VECTOR, which is why
  results proved "for any separations" are stronger - and why the gear-3 lemma was refuted with an
  explicit separation vector `(1,3,3,3,3,3)`.
- CALCULATES: which claims are fragile under generalisation.
- STATUS: frame definition, used to refute the gear-3 lemma.
- WHERE: docs/handover.md section 0.5 frame 4.
- LIMITS: none.

### THE EXPOSURE CRITERION (the master rule)
- STATEMENT For a set S of positions write `W_q(S) = { s-1, s : s in S } mod q`. Gear q is FORCED to
  strike one of S exactly when `W_q(S) = Z_q`; if `|W_q(S)| < q` there is an offset leaving every
  position of S open at once. Hence
  `#{m : every position of m + S is open} = prod_q (q - |W_q(S)|)`.
- CALCULATES: everything else in the covering line - forbidden configurations, the c_j decomposition,
  the hazard.
- STATUS: exact.
- WHERE: docs/forbidden-configurations.md header; research/minimal_forbidden.py,
  covering_decomposition.py, gap_automaton.py.
- LIMITS: adjacent (halved) frame.

### THE MINIMAL SIZE LAW: any (q-1)/2 columns can be simultaneously open to gear q
- STATEMENT Gear q can be forced to strike one of S only if `|S| >= (q+1)/2`, and the bound is
  ATTAINED - take S at residues `0, 2, 4, ..., q-1`, whose dominoes tile `Z_q` with one overlap, and
  integer positions with those residues all `= 0 mod 3` exist by CRT. EXPOSURE FORM: ANY `(q-1)/2`
  POSITIONS CAN BE SIMULTANEOUSLY OPEN TO GEAR q, WHATEVER THEIR SPACING. A gear only starts
  constraining once the configuration is half its circumference. Gear 3's and gear 5's blocking laws
  are its first two cases.
- CALCULATES: how many columns you may demand open before a given gear must strike one.
- STATUS: proved with an attaining construction; exhaustive to q = 19, construction checked for all
  45 odd primes below 200, zero failures.
- WHERE: docs/forbidden-configurations.md section 1; docs/handover.md section 3 item 8.
- LIMITS: existence of a simultaneously-openable set, not its position inside the window.

### Minimal span grows like 1.9q, and large gears force nothing new
- STATEMENT Restricting positions to multiples of 3 and minimising span: `q = 5..31` gives
  `span = 6, 12, 18, 24, 30, 36, 42, 54, 60`, ratio `span/q = 1.20 .. 1.94` - the span grows like
  `1.9 q` while the number of positions grows like `q/2`. Consequence: LARGE GEARS FORCE NOTHING NEW
  - gears 29 through 47 contribute ZERO minimal forbidden configurations beyond those from gears
  `<= 23` (25060, 25270, 25270, 25270, 25270, 25270, 25270 at gears to 19, 23, 29, 31, 37, 41, 47),
  and not as a box artefact, since 29's and 31's own minimal configurations fit inside the box.
- CALCULATES: which gears can add new local obstructions - only the small ones.
- STATUS: exact minima by bitmask dynamic programme; the gear census exhaustive inside a box of word
  length 16, letters to 6, reproduced at a smaller box.
- WHERE: docs/forbidden-configurations.md sections 2 and 5.
- LIMITS: gear 3 has NO forbidden configuration inside a single class mod 3; gears from 37 up need
  18+ letters and are untested at this width. THE ANTIDICTIONARY IS NOT FINITE - at both box widths
  the longest minimal forbidden word equals the box length and the per-length count is still rising.

### Minimal forbidden gap words of the small gears
- STATEMENT Gear 5 forbids ten minimal gap words: `11, 13, 16, 24, 31, 42, 61, 121, 151, 222`. Gear
  7 forbids seventeen: `121, 131, 213, 312, 314, 333, 413, 1111, 1113, 1233, 2112, 2114, 3111, 3113,
  3321, 4112, 12321`. Gear 11 forbids 170, of lengths 5 to 9. Gear 13 has a forbidden configuration
  of span 24. (Letters are gap/3.)
- CALCULATES: the explicit local obstructions of each small gear.
- STATUS: exact enumeration.
- WHERE: docs/forbidden-configurations.md section 2.
- LIMITS: small gears only.

### Admissibility is factor-closed
- STATEMENT If `S' subset S` then `W_q(S') subset W_q(S)`, so a configuration that fails to force
  has no sub-configuration that forces. Hence EVERY FACTOR OF AN ADMISSIBLE GAP WORD IS ADMISSIBLE,
  and a word is minimally forbidden exactly when it is forbidden while both of its length-(n-1)
  factors are admissible.
- CALCULATES: a level-by-level search - extend admissible words by one letter, keep the admissible.
- STATUS: exact.
- WHERE: docs/forbidden-configurations.md section 5; research/gap_automaton.py.
- LIMITS: none.

### THE FACTORISATION LAW (all placement dependence lives below span+1)
- STATEMENT With `w(S) = |{s-1, s : s in S}|` counted over the INTEGERS, `|W_q(S)| = w(S)` for every
  gear `q > span(S) + 1`. THE THRESHOLD IS `span+1`, NOT `span`: the extreme members `min(S)-1` and
  `max(S)` differ by `span(S)+1` and collide mod q exactly at `q = span+1`. Consequence: in any
  product `prod_q (q - |W_q(S)|)` the gears above `span(S)+1` contribute `prod (q - w(S))`, which
  sees only the size and adjacency of S, never its placement. ALL PLACEMENT DEPENDENCE LIVES IN THE
  GEARS AT OR BELOW `span(S) + 1`.
- CALCULATES: separates the gear set from the run length completely.
- STATUS: exact - verified with zero exceptions for L = 9, 12, 15, 18 against gear sets to 31 once
  the threshold was corrected.
- WHERE: docs/forbidden-configurations.md section 3.
- LIMITS: caught by check, not inspection - for `S = {0,12}` and `q = 13`, `W_13 = {0,11,12}` has
  size 3 while `2|S| = 4`.

### The c_j(L) decomposition (the gear set enters only through prod (q - j))
- STATEMENT `N(L) = sum_j c_j(L) * prod_{L < q <= y} (q - j)` with
  `c_j(L) = sum over T with w(T)=j of (-1)^{|T|} prod_{q <= L} (q - |W_q(T)|)`, and THE `c_j(L)` DO
  NOT DEPEND ON y. Table: `L=2: c_0=1, c_2=-2, c_3=1`; `L=6: c_0=15, c_2=-18, c_4=3`;
  `L=12: 1155, -1620, 651, -60`; `L=21: 4849845, -7952175, 4454838, -1038864, 94284, -1920`;
  `L=24: 111546435, -190852200, 118032579, -33045960, 4138290, -167856`.
- CALCULATES: `N(L)` for any y from a y-independent table.
- STATUS: exact - verified by reassembling `N(L)` at y = 13, 19, 23, 31.
- WHERE: docs/forbidden-configurations.md section 4.
- LIMITS: VALIDITY CONDITION - `c_j(L)` is built from gears `q <= L`, so it describes the machine
  only when the gear set contains all of them; the hazard at L needs `y >= L + 1`. At y=13, L=21 the
  formula returns 406008 against a true `N(21) = 312`.

### The per-J recipe does scale (the head gears annihilate nearly everything)
- STATEMENT `c_j(L)` is a sum over `2^L` subsets, but the product is zero the moment ONE gear has
  `W_q(T) = Z_q`, and a depth-first scan pruning on the first fully covered gear never visits the
  rest. Visited subsets at `L = 1, 3, 6, 9, 15, 21, 24, 30, 39` are
  `2, 4, 10, 19, 61, 181, 289, 721, 2548` against `2^39 = 5.5e11`. EVERY VISITED SUBSET CONTRIBUTES.
- CALCULATES: `c_j(L)` in closed form to L = 39; each tight case becomes an explicit finite
  inequality between products over the gear set, all checking out from closed forms for y = 23..199.
- STATUS: built and measured - an explicit REVERSAL of the earlier "does not scale" judgement,
  recorded as a method-audit lesson ("a scaling judgement made without building the thing cost a
  working route for most of the programme").
- WHERE: docs/forbidden-configurations.md section 9; docs/handover.md section 3 item 21 and the
  method audit note.
- LIMITS: one condition per L, and the tight L do not terminate - the NUMBER of conditions still
  grows like `F_k ~ 0.055 y^2`.

### The tight L are a short fixed list of small absolute values
- STATEMENT `h` rises within each block `{1,2}, {3,4,5}, {6,7,8}, ...`, so the minima of `h/d` sit
  at the BLOCK STARTS `L = 1, 3, 6, 9, ...`. Ranking them, the tight starts are `1, 6, 3, 21, 15, 9,
  24` with 30, 39, 45, 54 behind - THE SAME SMALL ABSOLUTE VALUES FOR EVERY GEAR SET, not a fixed
  fraction of the record, stable from y = 13 through y = 401, and FROZEN in order from y = 199 on.
  The four tightest are `1, 6, 3, 9`, exactly the four already proved.
- CALCULATES: which L need checking - a short fixed list, not a growing one.
- STATUS: measured (y = 13 .. 401; y=23 computed twice by independent implementations; y=29 and 31
  over 3.2e9 and 1e11 offset vectors).
- WHERE: docs/forbidden-configurations.md section 8; docs/ideas-from-the-session.md section 3a.
- LIMITS: "the falsification test is whether the tight list keeps growing with y; the measurement
  says it has not through y = 29."

### The gap identity and the hazard form
- STATEMENT By CRT, `N(L) = sum over gaps g of max(0, g - L)` - the number of positions from which L
  consecutive columns are all struck. A gap of length g contributes to `N(L) - N(L+1)` exactly once
  when `g > L`, so `G(L) = N(L) - N(L+1)` and `h(L) = G(L)/N(L) = 1 - N(L+1)/N(L)`. The target
  `N(L) <= P(1-d)^L` is exactly `h(L) >= d` for every L - THE HAZARD RATE OF THE GAP DISTRIBUTION IS
  AT LEAST d EVERYWHERE. "This is the cleanest form the open problem has taken: no offset vectors,
  no coverings, no separations, and no sub-problems."
- CALCULATES: recovers the record as the largest gap; `h(1) = d/(1-d) > d` always, free from the
  mod-3 law.
- STATUS: exact - verified with zero mismatches for gear sets `{3,5,7}` through `{3,5,7,11,13,17}`
  (22275 gaps in a period of 255255).
- WHERE: docs/covering-bound-route.md sections 15a, 15c, 17a; docs/forbidden-configurations.md
  section 6.
- LIMITS: the hazard is NOT monotone - `h` dips at k = 3, 6, 8, 9 in every set tested, so the
  distribution is not IHR.

### The proved hazard cases, and the extremal gear set
- STATEMENT `L=1`: `h(1) >= d` is `C_y = prod (1 - 4/(q-2)^2) <= 1`, true term by term. `L=3`:
  reduces to `prod (1-2/q)^2 >= prod (1-4/q)`, holding factor by factor since
  `(1-2/q)^2/(1-4/q) = 1 + 4/(q(q-4)) > 1`. `L=6`: uses `C = (8/3) B` exactly (collision factors at
  q = 5, 7) and `Y = 0` by the gear-5 law, giving `(11/3) C_y + E_y <= 2`, left side `86/45 = 1.9111`
  at y = 7. `L=9`: sufficient condition `(17/3) C_y + (14/3) E_y <= 3`, decreasing, holding from
  y = 17; y = 7, 11, 13 checked exactly in integers. `{3,5,7}` IS THE EXTREMAL GEAR SET - tight at
  L = 1, 6 and 9 (exact equality at both 6 and 9).
- CALCULATES: `kappa(L)` in closed form in rationals for small L at any y; the k-frame limit
  `kappa(2) = 2 - (11/3)C = 0.5448`.
- STATUS: proved for every y at L = 1, 2, 3 in the column frame; "to my knowledge, the first cases
  of form (b) proved in the k-frame".
- WHERE: docs/covering-bound-route.md sections 18b-d, 20, 23; docs/review-2026-08-17.md section 3.
- LIMITS: no uniformity in L - "they are the easy sliver"; the general block-start condition
  degenerates at the top (`0 >= 0`) exactly where `N(L)` vanishes.

### The pair weight psi, and only multiples of 3 contribute
- STATEMENT For a pair at distance `delta` the four values `t-1, t, t+delta-1, t+delta` collide mod q
  exactly when `q | delta` (`|W_q| = 2`) or `q | delta -+ 1` (`|W_q| = 3`), else 4. Gear 3 divides
  one of the three, and if it divides `delta +- 1` the factor is `1 - 3/3 = 0`. SO ONLY
  `delta = 0 mod 3` CONTRIBUTES - the gear-3 law reappearing TERM BY TERM rather than imposed. And
  `psi(delta) = 3C * prod_{q | delta} (q-2)/(q-4) * prod_{q | delta^2-1} (q-3)/(q-4)` with
  `C = 0.396880415`, `3C = 1.190641246`, `psi(3) = 3C` exactly.
- CALCULATES: the exact coincidence multiplicity of two columns at separation delta, gear by gear;
  `kappa(L) = L - sum_{delta <= L} psi(delta) + (small)`. THE MEAN OF psi IS EXACTLY 3 (running means
  2.6926, 2.9320, 2.9858, 2.9980, 2.99976 up to L = 63, 300, 3000, 30000, 300000).
- STATUS: exact algebra; the closed form checked against direct pattern counts for 150 deltas at
  y = 19 (exact agreement) and against measured kappa at y = 100003 for thirteen L. The closing
  inequality `sum psi <= L - 1` verified over every block start to `L = 5e6` (1.67 million cases),
  minimum `1.6343` at L = 6 throughout, no block start below 1.
- WHERE: docs/forbidden-configurations.md section 8c; docs/handover.md section 3 item 22;
  research/kappa_expansion.py.
- LIMITS: an expansion in d at fixed L - it controls `L << 1/d`, while the records live at
  `L*d ~ 8-18 and growing` (THE REGIME GAP). Closing it needs control of an AVERAGE, not a mechanism.

### The repulsion form (openings repel)
- STATEMENT With `v(delta) = P(0 and delta both open)`, `kappa(L) >= 1` says exactly: conditioning on
  an open column at 0 REDUCES the expected number of open columns in `(0, L]` by at least d - one
  column's worth of density. `v(1) = v(2) = 0` OUTRIGHT, since gear 3 blocks one of any two positions
  less than 3 apart, a deficit of `2d^2` against the unconditional `L d^2`.
- CALCULATES: the requirement is `mean psi <= 3 - 3/L`, measured `3 - 104/L` at `L = 5e6` - a margin
  of about 30x at large L, and at the tight point L = 6 mean psi = 2.183 against 2.5 required.
- STATUS: exact identity; measured margins.
- WHERE: docs/forbidden-configurations.md section 8d; docs/handover.md section 3 item 23.
- LIMITS: "the repulsion is measurable but not yet bounded".

### The helper condition (which gears can serve two columns from one offset)
- STATEMENT One offset serves two positions exactly when their distance equals that gear's TOOTH
  SEPARATION: gear q blocks `{r, r + s_q}`, so it covers both i and `i + delta` from a single offset
  precisely when `delta = +- s_q mod q`. In the column frame (`s_q = 3^{-1} mod q`) that is
  `q | 3delta - 1 or q | 3delta + 1` - THE HELPERS AT DISTANCE delta ARE EXACTLY THE PRIME DIVISORS
  OF `3delta - 1` AND `3delta + 1`, at most `omega(3delta-1) + omega(3delta+1) <= 2 log_2(3delta+1)`
  of them - six at delta = 2, ten at 10, twenty-four at 1000. NEVER ALL pi(y) OF THEM.
- CALCULATES: the helper count at any distance.
- STATUS: exact - verified with zero mismatches over gears 5 to 299 and delta 1 to 39.
- WHERE: docs/covering-bound-route.md section 8b.
- LIMITS: it governs the L = 2 violation but NOT the general step, which involves every earlier
  position rather than one distance.

### Adjacency is annihilated at gear 3
- STATEMENT For a single gear, two positions at distance `>= 2` forbid four offsets (factor
  `1 - 4/q`, negatively correlated), while ADJACENT positions forbid only three (factor `1 - 3/q`,
  positively correlated for `q >= 5`) - but at `q = 3` that factor is `1 - 3/3 = 0` EXACTLY. THE
  POSITIVE CORRELATION IS ANNIHILATED BY THE 6-CYCLE ITSELF, giving `h(1) = d/(1-d)` exactly.
- CALCULATES: `Pr[adjacent both open] = 0.000000` for `{3,5,7}` and `{3,5,7,11}` against `0.166234`
  for `{5,7,11}`.
- STATUS: exact; measured.
- WHERE: docs/covering-bound-route.md section 4; docs/forbidden-configurations.md section 6a.
- LIMITS: it controls the bound at L = 2 exactly, not at every L - in the column frame five of the
  twelve distances 1..12 are positively correlated for `{5,7,11,13}` (distances 2, 5, 7, 10, 12,
  corresponding to gears dividing `3delta -+ 1`), and the bound holds regardless.

### The domino: only the first step of a blocked run is cheap
- STATEMENT `rho(L) <= rho(1)` says the FIRST step of a blocked run is the cheap one. The reason is
  the domino: gear q blocks the adjacent pair `{o, o+1}`, so given position 0 blocked by q, position
  1 comes free whenever that gear's domino points right - half of the offsets that block 0. NO LATER
  POSITION CAN SHARE A DOMINO WITH POSITION 0, so no later step is cheap in the same way.
- CALCULATES: the mechanism behind the peak of the hazard at L = 1.
- STATUS: mechanism statement; `rho(L) <= rho(1)` verified at every L up to F_h for y = 7..19, peak
  always at L = 1.
- WHERE: docs/forbidden-configurations.md section 6b.
- LIMITS: "The statement is true jointly and false per gear" - per-gear conditional marginals RISE
  under conditioning, by up to 63%.

### The spread lemma (no offset of any gear is much more useful than another)
- STATEMENT For gear q with tooth separation s, every offset blocks between `2 floor(i/q)` and
  `2 floor(i/q) + 2` positions of a run of length i - A SPREAD OF AT MOST 2, whatever i and q - and
  WHEN q DIVIDES i THE SPREAD IS EXACTLY 0, so all q offsets are perfectly interchangeable.
- CALCULATES: in the window regime, where the run is far longer than any gear, no offset of any gear
  is materially more useful than another, so conditioning cannot concentrate.
- STATUS: proved and exact; measured for q = 5, 7, 11, 29 - spread 0 at every multiple of q, relative
  spread falling like `q/i`.
- WHERE: docs/covering-bound-route.md section 9c.
- LIMITS: "usefulness is not the only thing conditioning responds to: WHERE an offset blocks matters
  as well as how much."

### Smallest-gear-first splitting (the run-length recursion on gears)
- STATEMENT Removing the smallest gear `q_1` splits `[0,L)` into the `q_1 - 2` residue classes it
  leaves open; each class reindexes to an interval of length `L/q_1`, with the remaining gears'
  SEPARATIONS RESCALED to `s_q q_1^{-1} mod q`. Gear 3 leaves EXACTLY ONE sub-problem (which is why
  the gear-3 conditioning was clean); gear 5 leaves three, gear 7 five.
- CALCULATES: the closing chain `f(L) <= (1-d')^{L(1 - 2/q_1)} <= (1-d)^L`, so the entire bound
  follows from one correlation statement, and the reduction is genuine - each level consumes a gear
  and shortens the interval.
- STATUS: exact algebra; sub-problem negative correlation measured with zero violations for
  `{7,11,13}`, `{7,11,13,17}` and for removing gear 7 from `{11,13,17}`.
- WHERE: docs/covering-bound-route.md sections 12a-12b.
- LIMITS: the correlation statement it needs is unproved, and the sub-problem's own bound genuinely
  FAILS for some induced separations.

### The divisor law for induced adjacency
- STATEMENT Under the recursion, gear q reaches separation 1 exactly when `s_q q_1^{-1} = 1 mod q`,
  that is `s_q = q_1`, and with `s_q = 3^{-1} mod q` that is `3 q_1 = 1 mod q`:
  GEAR q BECOMES ADJACENT AT LEVEL q_1 IFF `q | 3 q_1 - 1`. Examples: `q_1=5: 14 = 2*7` -> gear 7;
  `q_1=7: 20 = 2^2*5` -> gear 5; `q_1=11: 32 = 2^5` -> none; `q_1=29: 86 = 2*43` -> gear 43;
  `q_1=1009: 3026 = 2*17*89` -> gears 17, 89. At most `omega(3q_1-1) <= log_2(3q_1)` of them.
- CALCULATES: which gears are pushed to adjacency at each level of the recursion.
- STATUS: exact - verified with zero mismatches for every `q_1` up to 60 against all gears to 200;
  explicitly retained as correct arithmetic after the conclusion built on it was refuted.
- WHERE: docs/covering-bound-route.md section 12d.
- LIMITS: the CONCLUSION drawn from it ("uniform adjacency never occurs, therefore failure is
  impossible") was refuted.

### THE GENERATING POLYNOMIAL (every alignment law as one substitution)
- STATEMENT By CRT the columns struck by precisely the gears of a subset T number
  `2^{|T|} prod_{q not in T} (q-2)`; summing over subsets gives one product
  `p(x) = prod over q in S of (q - 2 + 2x)`, whose coefficient of `x^j` is the number of columns
  struck by exactly j gears. Evaluations: `p(1) = P` (all columns); `p(0) = prod(q-2)` (openings);
  `p(1-k) = prod(q-2k)` (positions with k consecutive openings); `p(0) - p(-1)` (number of opening
  runs). Examples: `(5,7) [15, 16, 4]`; `(5,7,11) [135, 174, 68, 8]`;
  `(5,7,11,13) [1485, 2184, 1096, 224, 16]`.
- CALCULATES: the whole n-wise alignment lattice in closed form, with no enumeration.
- STATUS: exact - verified over 30 gear sets of sizes 1 to 5 with zero mismatches; the derived counts
  verified over 66 pairs to 45 and 67 gear sets with zero failures.
- WHERE: docs/twin-prime-program.md sections 32b-32c; docs/handover.md section 3 item 11;
  research/nwise.py, research/alignment.py.
- LIMITS: `p(x)` is invariant under any relabelling of the residues - it depends only on the multiset
  of gears, NOT on where their teeth sit. The distance to the next opening depends on phase, and the
  generating polynomial has none.

### Points and dominoes: the openings have no run longer than 2
- STATEMENT The gear-5 law forbids three consecutive open columns, so in column space THE OPENING SET
  IS EXACTLY A DISJOINT UNION OF ISOLATED POINTS AND DOMINOES - nothing longer. Counts follow:
  dominoes number `n_1 = prod (q-4)`, total runs `prod(q-2) - prod(q-4)`, singletons
  `prod(q-2) - 2 prod(q-4)`. Consequence: the `prod(q-2k)` family COLLAPSES at k = 3 and carries
  exactly two numbers, A and n_1, and no more. Validity of `prod(q-2k)`: `q >= 6(k-1)`, because for
  q = 7 the two teeth are only `2u = 2` apart so the forbidden residues collide (`(7,11)` has 10
  triple-alignments, not `prod(q-6) = 5`).
- CALCULATES: the complete local description of the openings, and the counts of each object type -
  checked at `{3,5,7}`: 9 singletons and 3 dominoes, 15 points in 12 runs.
- STATUS: exact corollary.
- WHERE: docs/ideas-from-the-session.md section 2; docs/twin-prime-program.md section 31c.
- LIMITS: local only - it says nothing about spacing BETWEEN the objects, which is the whole problem.
  "The gap distribution is the distribution of spacings between these objects, a one-dimensional
  arrangement problem with two object types."

### THE ALIGNMENT LAW (the longest run of openings is set by the smallest gear alone)
- STATEMENT The longest run of consecutive OPENINGS of a gear set equals the LONG ARC OF ITS
  SMALLEST GEAR, `q_min - 2u_min - 1`, no matter how many other gears are added (smallest gear 5 ->
  2; 7 -> 4; 11 -> 6; 13 -> 8; 17 -> 10). THE REASON IS ALIGNMENT: by CRT every relative phase of the
  other gears occurs somewhere in the period, so there is a position at which none of their teeth
  falls inside the smallest gear's long arc, leaving it fully open. Adding gears cannot shorten it
  because it cannot remove that position. The shortest run is always 1 - every gear set has isolated
  openings.
- CALCULATES: the longest run of consecutive openings for any gear set, from its smallest gear only.
  COROLLARY: the pattern of all gears up to y always contains two consecutive openings, so the
  admissible pattern structurally supports prime quadruplets at every level.
- STATUS: exact, zero failures over 103 gear sets (research/alignment.py).
- WHERE: docs/twin-prime-program.md sections 26c-26d.
- LIMITS: "What it withholds is LOCATION. The law says SOMEWHERE IN THE PERIOD, and the period is the
  primorial while the validity window is the first `y^2/6` columns." This is the closest the corpus
  comes to a positive alignment theorem, and its limitation is exactly Reduction A.

### The turn law in closed form (which copies a new gear strikes)
- STATEMENT Adding gear q to a machine of period P, an open class `k_0` spawns q daughter classes
  `k_0 + tP`, of which EXACTLY TWO ARE STRUCK - at `t = (+-u_q - k_0) * P^{-1} mod q` - and `q-2`
  survive. Because `gcd(P,q) = 1`, gear q's tooth visits all q residues in exactly q turns of the
  sub-machine.
- CALCULATES: for any opening class of a sub-machine, exactly which of the next q copies a new gear
  kills, without running the machine. It derives `prod(q-2)` from slip arithmetic alone, and is why
  no class can ever be closed out - every gear is at least 5, so `q - 2 >= 3 > 0`.
- STATUS: exact - verified against brute force over all sub-machines of up to three gears from 5..29
  and all their open classes, zero mismatches (research/slip_algebra.py).
- WHERE: docs/twin-prime-program.md sections 17a-17b; docs/handover.md section 0.3.
- LIMITS: gives the surviving turns of ONE added gear, not the joint answer over all gears.

### Machine slip versus cycle slip
- STATEMENT Two quantities kept apart: the CYCLE SLIP `|P_S - P_T|` (how far two cycles drift per
  revolution) and the MACHINE SLIP `P_S mod q` (the phase a composite machine of period P_S presents
  to a new gear). THE SECOND IS WHAT COMPOSES, and it is the input to the turn law's `P^{-1} mod q`.
- CALCULATES: the phase a machine hands to the next gear.
- STATUS: standing.
- WHERE: docs/handover.md section 0.3; docs/twin-prime-program.md section 17a.
- LIMITS: a naming rule; "both were called slip in this work and must be kept apart".

### Refinement law (the openings of a union are the CRT of the openings)
- STATEMENT For disjoint gear sets A, B: `open(A union B) = CRT(open(A), open(B))`, verified
  identical AS SETS, with `|open(5,7)| * |open(11,13)| = 15 * 99 = 1485 = |open(5,7,11,13)|`.
- CALCULATES: the machine's openings by combining sub-machines', exactly.
- STATUS: exact (verified as sets).
- WHERE: docs/twin-prime-program.md section 17b.
- LIMITS: combining does not reduce cost - the CRT is the primorial.

### Square roots of unity (where every gear's teeth align at once)
- STATEMENT For any gear set the joint threats are the CRT lifts of the sign vectors, `2^n` of them,
  and the set of admissible X is exactly `{ X : X = +-1 mod q for every q in S } = { X : X^2 = 1
  mod P }`. SO THE COLUMNS STRUCK BY EVERY GEAR AT ONCE ARE THE SQUARE ROOTS OF UNITY MODULO THE
  PERIOD, SCALED BY `6^{-1}`. For a pair the crossed threats are governed by the SLIP INVERSE:
  `X_crossed = 1 + q((-2 q^{-1}) mod q')`.
- CALCULATES: the full-coincidence positions of any gear set - count `2^n`, positions
  `6^{-1} X mod P`.
- STATUS: exact - verified for sets of one to four gears (count `2^n` and the scaled root set equals
  the direct threat set exactly); the crossed formula verified for all 28 pairs of gears up to 29,
  zero failures.
- WHERE: docs/twin-prime-program.md sections 28b-28c.
- LIMITS: this is the TOTAL-coincidence set - the opposite extreme from an opening.

### The whole machine as one gcd
- STATEMENT `m is open to every gear in S <=> gcd(36 m^2 - 1, prod S) = 1`; taking S to be the gears
  up to the certifying bound, `m is a twin column <=> gcd(36 m^2 - 1, primorial(sqrt(6m+1))) = 1`.
  Gear q strikes m exactly when `6m` is a square root of unity mod q, and SHIELDS when `6m = 0`. The
  36 is `6^2`, the square of the block period; writing `N = 6m` removes the constant entirely.
- CALCULATES: the machine's opening condition as one arithmetic statement about one quadratic,
  rather than `pi(y)` gear congruences.
- STATUS: exact - 11996 checks across five gear sets with zero disagreements; `m = 1..3999` against
  direct primality, zero mismatches; kernel-checked as `centreSurvivor_iff_twin`.
- WHERE: docs/twin-prime-program.md sections 28d, 29a; docs/handover.md sections 0.6, 3a.
- LIMITS: evaluating the gcd needs the primorial - it is a VALIDATOR, not a constructor. "These are
  exact but say nothing about WHERE the next twin is."

### The closed-form target, stated without gear bookkeeping
- STATEMENT `N-1` and `N+1` both coprime to `P(y)` means both are UNITS mod the primorial, so
  "twin in the window `<=>` two units of `U(P(y))` differing by 2, inside the first `y^2` residues".
  Units are exactly the CRT vectors with every coordinate nonzero, and a difference-2 pair is a
  vector v with `v_q != 0` and `v_q != -2` for every gear - two forbidden residues per coordinate,
  which is `prod(q-2)` arriving from the group side. SO GENERATING THE NEXT OPENING IN CLOSED FORM IS
  EXACTLY: minimise `CRT(v)` above a given bound subject to `v_q not in {0, -2}` for every gear
  `q <= y`. ANY CLOSED FORM MUST SOLVE THIS MINIMISATION.
- CALCULATES: the exact statement any closed form for the position of the next opening must solve.
- STATUS: exact restatement; the unit-pair count equals the twin count below `y^2` exactly at
  y = 5..29 (2, 4, 7, 9, 15, 17, 21, 28).
- WHERE: docs/twin-prime-program.md sections 29c-29d.
- LIMITS: "the closed form will not come from refining the slip arithmetic, because the slip
  arithmetic is now fully solved and expressed."

### Forcing channels, and the forcing budget
- STATEMENT Gear q cannot strike column k whenever `q | 6k + c` for any c not congruent to
  `+-1 mod q`. `q | 6k` (the shield) is one channel among many - `q | 2k+1`, `q | 3k +- 1` force
  openings just as well. Widening from the midpoint alone to `c in {0, +-2, +-3}` takes the
  self-certifying family from 4 columns to 10: `k = 1, 2, 3, 5, 7, 10, 12, 17, 18, 33` - every one a
  genuine twin, and nothing further to `k = 300000`. THE BUDGET: certifying column k requires every
  gear to `sqrt(6k+1)` to be forced open - `pi(sqrt(6k))` of them - while each fixed channel supplies
  only `omega(6k+c) ~ log log(6k)`, and a single channel covering many gears needs `M | 6k+c`, so it
  protects at most `log_5(6k+c)` gears (14 at `k = 10^9` against 7606 demanded). `sqrt k` AGAINST
  `log log k`; equivalently, forcing the gears up to z needs `6k + c >= primorial(z)`, so k is of
  order `e^z`, whose certification demands gears to `e^{z/2}`.
- CALCULATES: the complete list of columns at which every gear's opening is forced by divisibility,
  and the exact numerical requirement on any closed form.
- STATUS: exact / measured to k = 300000; the deficit becomes permanent at k = 50 and widens forever.
- WHERE: docs/twin-prime-program.md sections 22b-22c.
- LIMITS: "Any route to a closed form has to supply a forcing mechanism that is not divisibility by
  a product." The three restriction mechanisms tried (order `N = a^n`, congruence `N = x^2+2`,
  cyclotomic `q | Phi_k(x)`) give the dichotomy: strong enough to close the budget implies the family
  is `{(3,5)}`; nonempty family implies only a constant or polynomial factor.

### Bounded-depth state never determines the position of the next opening
- STATEMENT For any depth z with `P = prod(gears <= z)`, take any twin column `k1` whose members
  exceed z; then every `k1 + mP` has THE SAME RESIDUE AT EVERY GEAR AT MOST z, for every m, while not
  all are twin columns. Hence the state of the gears up to z does not determine whether a column is a
  twin. Explicit copies: `z=13, k1=3 (17,19)` vs `5008 (30047,30049)` broken by 151; `z=29, k1=7` vs
  `1078282212` broken by 1567; `z=97, 17` vs `3.8e35` broken by 191; `z=199, 38` vs `1.3e81` broken
  by 9619; `z=997, 170` vs `3.3e336` broken by 38185123.
- CALCULATES: refutes in advance any navigation rule using gears only up to a fixed depth - "no
  navigation using only the gears up to a fixed depth can compute the step count, however cleverly it
  combines their positions, cycle lengths, slips, arcs and sub-machine periods".
- STATUS: exact, verified by explicit construction at five depths (research/navigate.py).
- WHERE: docs/twin-prime-program.md sections 23b-23c, 24d.
- LIMITS: FRAME-INDEPENDENT - if two columns agree at every gear up to z they agree on every exposure
  quantity too (same arc, same room, same shield status, same schedule), since all are functions of
  `k mod q`. It does NOT rule out a rule that consumes the gear set all the way to `sqrt(6k)`, or one
  expressed directly in k. Empirical form: at `k0 = 10^12 + 1` the step count reads 1, 1, 1, 2, 19,
  19, 19, 19 for gears to 20, 50, 100, 200, 500, 1000, 1e4, 1e5 and 86 for the full set - "stopping
  early does not give an incomplete answer, it gives a SPECIFIC WRONG ANSWER, 19 instead of 86".

### The spectrum factorises, but no beat may be dropped
- STATEMENT The opening signal is a PRODUCT `E(m) = prod_q e_q(m mod q)`, so by CRT its transform
  factorises: `Ehat(k) = prod_q ehat_q(k t_q mod q)` with `t_q = (P/q)^{-1} mod q`,
  `ehat_q(0) = (q-2)/q`, `ehat_q(c != 0) = -(2/q) cos(2 pi c u_q / q)`. BUT the L1 norm factorises
  too and grows about 2.06 per gear (per-gear factors 1.494, 1.713, 1.914, 1.969, 2.040, 2.064,
  2.100, 2.136 for gears 5..29, approaching `1 + 4/pi`), so at `m = 10^12` with 179643 gears it is
  around `10^56000` against a required resolution of 1/2: NO BEAT CAN BE DROPPED. Exact evaluation
  organised by beat order IS inclusion-exclusion over gear subsets, costing `2^n` terms.
- CALCULATES: the entire Fourier transform of the openings signal from a product of n cosines per
  frequency, with no FFT.
- STATUS: exact - agrees with a direct FFT to `1.1e-16` across all 5005 frequencies for
  `(5,7,11,13)`; full-spectrum reconstruction exact to `1.9e-15`.
- WHERE: docs/twin-prime-program.md sections 35a, 36b-36d; docs/handover.md section 3 item 12.
- LIMITS: "The frequency domain is a faithful and exact description that is exponentially less
  compact than the machine it describes." The `t_q` twist is essential and easy to miss.

### The small-gear phase floor, and where the phase relation dies
- STATEMENT For a sub-machine S the openings depend only on `m mod P_S`, so the least open offset is
  a valid LOWER BOUND on the distance to the next opening - a pure table lookup with no stepping. It
  is weak: at `m = 10^12 + 1` the floor from `(5,7,11,13)` is 1 against a true step of 86, and
  enlarging to `(5,7,11,13,17,19)` does not move it. THE ONE STRUCTURE THAT WOULD RELATE PHASES
  ACROSS GEARS DIES AT THE SQUARE ROOT: on the hyperbolic block where `c = floor(6m/q)` is constant
  the phase is LINEAR in the gear (`6m mod q = 6m - cq`), but gears satisfy `q <= sqrt(6m)`, forcing
  `c >= sqrt(6m)` and block length `< 1` - measured at `m = 10^12 + 1` over all 179643 gears: 179643
  distinct values of c, MAXIMUM GEARS SHARING A BLOCK = 1, blocks with two or more gears = 0.
- CALCULATES: a valid but weak floor; and the exact reason no phase-based rule can relate two gears
  by the divisor hyperbola.
- STATUS: exact, measured (research/phase.py).
- WHERE: docs/twin-prime-program.md sections 33a-33b; docs/handover.md section 2.6.
- LIMITS: the dual (cofactor) side supplies no relation between different gears either - "the linear
  relation the cofactor side makes available IS that gear's arithmetic progression of multiples,
  restated". The phase relation and the certification window fail at the SAME place, which the
  corpus records as "not a coincidence".

### Constraint concentration (which gears decide the next opening)
- STATEMENT A gear can only matter over the next W columns if its current opening ENDS inside the
  window, `d_q <= W`. At `k0 = 10^12 + 1` with 179643 gears: `W=16` -> 60 gears constrain; `W=128` ->
  299; `W=512` -> 940 - and those are precisely the gears dividing one of the `2W` window numbers.
  "A large gear's open run is about `2q/3`, vastly longer than any window of interest, so it can only
  constrain when we happen to sit within W of one of its two teeth."
- CALCULATES: the answer is DETERMINED by a few hundred gears and every other gear is consulted only
  to prove it does not bite - a 600 to 1 ratio at `k = 10^12`; the biting count grows like
  `2 W log log y`.
- STATUS: measured at `k0 = 10^9, 10^12, 10^14, 10^16` (research/exposure.py).
- WHERE: docs/twin-prime-program.md sections 24b, 22a.
- LIMITS: knowing WHICH gears bite requires factoring the window numbers.

### Inside the window every event is a root event
- STATEMENT For a column in the validity window, every opening-ending has cofactor at least the gear:
  `j = m/q >= y^2/y = y >= q`. Measured over 1024 columns at `k0 = 10^12 + 1` with 179643 gears:
  4327 events, 4327 root, 0 redundant. "This is a second characterisation of the window: it is
  exactly the region in which the gear set does no redundant work."
- CALCULATES: no gear meeting the window is doing work another gear already did.
- STATUS: exact law, measured 4327/4327.
- WHERE: docs/twin-prime-program.md section 25a.
- LIMITS: it is NOT a statement about overlap - two gears both at most `sqrt(m)` can divide the same
  member, and both events are root events.

### Fragile columns (closed by exactly one gear)
- STATEMENT In the same 1024-column window, 9 columns have zero endings and are twins, and 55 HAVE
  EXACTLY ONE - a single gear is all that closes them, with closing gears from 5 up to 353057:
  column +16 gear 127 cofactor 47244094489; column +19 gear 150697 cofactor 39814993; column +184
  gear 353057 cofactor 16994423. "The fragile columns closed by a LARGE gear are precisely what
  defeats bounded-depth navigation" - to a gear set stopping below 150697, column +19 is
  indistinguishable from a twin.
- CALCULATES: the concrete form of the bounded-depth obstruction.
- STATUS: measured (one window at one scale).
- WHERE: docs/twin-prime-program.md section 25b.
- LIMITS: one window.

### Overlap counting: an exact certificate that dies at gear 37
- STATEMENT `f(L) = maxgroupkills(group, L) + sum over remainder of maxkills(q, L)`, and
  `f(L) >= L` is necessary for a run of L consecutive columns to be coverable, so ANY SINGLE L WITH
  `f(L) < L` CERTIFIES `F_k <= L` - for each L independently, with no monotonicity assumed. It bites
  exactly when `2 * sum over excluded gears of 1/q < prod over group of (1 - 2/q)`.
- CALCULATES: an upper bound on the record - `y=23: 34 true vs 50; y=29: 43 vs 135; y=31: 58 vs
  1043`; vacuous from y=37. Overlap accounting moves the vacuum point from 13 to 37.
- STATUS: exact certificate, measured to y = 43 (research/overlap_bound.py).
- WHERE: docs/twin-prime-program.md section 19.
- LIMITS: "the group must contain all but a `1/log y` fraction of the gears, and its period is still
  exponential in y. OVERLAP COUNTING DONE THIS WAY IS THE PERIOD SCAN WEARING A DISGUISE."

### The extremal covering exhausts every gear at near-perfect efficiency
- STATEMENT At y = 17, 19, 23, 29 THE EXTREMAL COVERING LEAVES NO GEAR IDLE - it exhausts all gears
  up to y. Measured efficiencies: `y=19, L=74`: all 1.000, 146 incidences, 1.973 per position;
  `y=31, L=173`: 0.833-1.000, 370 incidences, 2.139. Coverage multiplicity is SPREAD, not
  concentrated: at y = 31 the distribution is `{1: 56, 2: 63, 3: 37, 4: 10, 5: 5, 6: 2}`, so 56 of
  173 positions are covered exactly once and PIN their gear's offset. Average multiplicity matches
  `2 sum_{q <= y} 1/q` almost exactly (1.973 vs 1.911; 2.139 vs 2.131).
- CALCULATES: why capacity bounds fail - a counting bound needs
  `L <= 2 pi(y)/(mult - 2 sum 1/q)`, requiring `mult` to exceed `2 sum 1/q ~ 2 log log y`, while
  measured `mult ~ 2` - THE REQUIRED FORCED OVERLAP IS A FACTOR `log log y` ABOVE WHAT OCCURS.
- STATUS: exact witnesses at y = 19, 23, 29, 31.
- WHERE: docs/twin-prime-program.md sections 37b-37c.
- LIMITS: "the capacity bound is close to achievable and its failure is not slack"; no counting
  argument of this shape can close it.

### The covering-count conjecture (and why it needs gear 3)
- STATEMENT `N(L) <= P (1 - d)^L`, where `N(L)` counts offset vectors covering a run of length L and
  `d = prod (1 - 2/q)`. Given it, `F_h(y) <= ceil(log P / -log(1-d))`, of order `y log^2 y`, below
  the `y^2/2` the window requires for every `y >= 23`; since exact values settle `y <= 43`, the two
  ranges overlap and the union is complete. "The count is an integer, so nothing probabilistic
  enters."
- CALCULATES: a bound on the record from a count of offset vectors.
- STATUS: CONJECTURED; verified with zero violations for every gear set containing 3 up to
  `P = 4.8` million, and exhaustively at y = 19, 23, 29, 31 over 4.8e6, 1.1e8, 3.2e9 and 1e11 offset
  vectors, worst ratio exactly 1.000000 throughout.
- WHERE: docs/covering-bound-route.md sections 1-3, 14c; docs/twin-prime-program.md section 38.
- LIMITS: it FAILS for gear sets omitting 3, and the mechanism is the adjacency annihilation at
  `q = 3`. "Any proof must use the `q=3` adjacency annihilation rather than a generic correlation
  inequality - precisely because it is false without 3", and must use the SEPARATIONS, not merely
  gear 3's presence. The bound needed is far weaker than the one conjectured: only
  `rho <= exp(-2 log P / y^2)`, a decay rate about `2/y` against the conjectured `d ~ 1/log^2 y`,
  with slack growing like `y/(2 log^2 y)`.

### Translation and reflection symmetry of coverings (the record search)
- STATEMENT TRANSLATION: shifting every offset by the same t maps coverings to coverings of a
  translated run and the largest gap is translation invariant, so t can be chosen to put gear 3 at
  offset 0 - a factor of 3, covering positions 0 and 1 before the search starts. REFLECTION:
  reversing the run maps q's blocked pair `{o, o+1}` to `{L-2-o, L-1-o}`, another adjacent pair, so
  coverability is reversal invariant; with 3 pinned the residual involution is
  `R(o) = (L-2-o) - s (mod q)`, `s = (L-2) mod 3`, broken by pre-assigning gear 5 and keeping only
  offsets with `o5 <= R(o5)`.
- CALCULATES: the record for y = 29, 31, 37, 41, 43 (`F(2,y) = 129, 174, 264, 273, 309`) where the
  period scan cannot reach; `F(2,47) = 354`, `F_k(47) = 118`.
- STATUS: exact for `y <= 47`; all eleven values from y = 5 to 41 recomputed from L = 1 with the
  break in place, every one matching.
- WHERE: docs/twin-prime-program.md section 5; docs/gear-recursion.md section 2.
- LIMITS: `y = 53` is out of reach this way; the record table is
  `F_h = 6, 15, 21, 33, 54, 75, 102, 129, 174, 264, 273, 309, 354` at y = 5..47, all `= 0 mod 3`.

### Elementary lower bound on the record, by pinning every gear at offset 0
- STATEMENT Pin every gear up to z at offset 0 so it blocks positions 0 and 1 mod q; the positions
  still uncovered in `[0,L)` are exactly the openings of `S(2,z)`, a finite computation. Then spend
  one gear from `(z,y]` on each remaining position:
  `F(2,y) >= max { L : |S(2,z) cap [0,L)| <= pi(y) - pi(z) }`.
- CALCULATES: lower bounds - `y=29 -> 53` (true 129), `31 -> 59` (174), `37 -> 74` (264),
  `41 -> 83` (273), `43 -> 107` (309); better than the trivial pairing bound by a factor of 4.
- STATUS: exact / computable by finite check.
- WHERE: docs/twin-prime-program.md section 14a.
- LIMITS: still a factor of 3 below the truth - "the algorithm's structure yields elementary lower
  bounds easily and upper bounds not at all".

### The increment law and the proof skeleton
- STATEMENT `F_adjacent(y) ~ C sum_{3<=q<=y} q` with C measured between 0.808 and 1.354 (values
  1.000, 0.808, 0.846, 0.964, 1.000, 1.041, 1.016, 1.101, 1.354, 1.157, 1.108 at y = 7..43). The
  skeleton: (1) prove `F_adjacent(y) <= C * sum q`; (2) elementary - the odd primes are a subset of
  the odd numbers, so `sum_{3<=q<=y} q <= (y^2+2y-3)/4`, with NO prime counting; (3) conclude the
  chain closes as soon as `C <= 2(y^2-y)/(y^2+2y-3)`, which is 1.8125 at y = 29 and rises to 2 - SO
  `C <= 1.8` SUFFICES FOR EVERY `y >= 29`.
- CALCULATES: the whole target reduced to one constant.
- STATUS: steps 2 and 3 elementary/exact; step 1 open; C measured over thirteen exact values.
- WHERE: docs/gear-recursion.md sections 5-6.
- LIMITS: C is NOT monotone and its supremum is not established; the largest measured is 1.354 at
  y = 37 against a threshold of 1.85 - "this is the weak point of the skeleton, not the elementary
  step". PER-STEP increment bounds CANNOT deliver the constant: the gear-37 step reaches 2.432q, so
  no per-step bound `<= 1.8q` is true, and the constant must be argued in aggregate.

### Maximal gaps are strongly isolated
- STATEMENT The gaps immediately either side of a maximal gap are minimal - at y = 29 the only
  flanking pair is `(2,2)`, the smallest possible; flanking pairs are `(2,2) (2,3) (2,5)` at y=19,
  `(1,5) (3,3) (5,1)` at y=23, `(2,2)` at y=29. "A long blocked run consumes the gears in its
  neighbourhood, so what follows it is dense." The isolation strengthens with y.
- CALCULATES: the trade-off in `F(M+q) - F(M) = (F2(M) - F(M)) + (F(M+q) - F2(M))` - a large F forces
  small neighbours, capping `F2 - F`, while a large `F2 - F` means two medium gaps are adjacent,
  limiting how much a chain can add on top.
- STATUS: measured (y = 19, 23, 29).
- WHERE: docs/gear-recursion.md section 4c; docs/handover.md sections 6 item 14 and 7.1.
- LIMITS: CORRECTED - isolation does NOT explain F2. At y = 29, `F2 = 55` comes from the adjacent
  pair `(30, 25)`, two large-but-not-maximal gaps; same at y = 19 (`F2 = 31` from `(21,10)` while the
  maximal gap is 25). Only at y = 23 does F2 sit at a maximal gap.

### The hole mechanism (why F2 - F is small)
- STATEMENT Covering `[0,L)` leaves gear 3 free to choose its offset, so gears `>= 5` must cover one
  whole class mod 3 and may pick the smallest. In the HOLE problem gear 3 must AVOID the hole h,
  which leaves it exactly one admissible offset out of 3, so the class left to gears `>= 5` is FORCED
  to be `h mod 3` - and the hole itself is one position of that class no longer needing coverage. SO
  THE HOLE BUYS EXACTLY ONE POSITION of slack, at the cost of losing the choice of class and of every
  gear `>= 5` losing 2 of its q offsets.
- CALCULATES: the two covering problems as one family - `F(y) = 1 + max coverable run`;
  `F2(y) = 1 + max run coverable except for one interior position`.
- STATUS: mechanical account; F2 values validated against the pattern -
  `F2 = 21, 33, 48, 75, 93, 117, 165` at y = 7..29, exact in all seven cases.
- WHERE: docs/gear-recursion.md section 4c.
- LIMITS: "It is not an argument for boundedness, and the y = 29 value shows why one should not claim
  boundedness from it." `F2 - F` reads 2, 4, 5, 7, 6, 5, 12 - DOUBLING at y = 29.

### The disjunction, priced (minimal admissible diameters from the machine's own rule)
- STATEMENT Minimal admissible diameters computed from the machine's own admissibility rule (the
  offsets must not cover every residue class modulo any prime), all matching the known values:
  `w(k) = 2, 6, 8, 12, 16, 20, 26, 30, 32, 36, 42` for `k = 2..12`. A position survives the
  disjunction unless EVERY d fails there - a conjunction the blocking must arrange simultaneously.
- CALCULATES: over `(43, 1849]` the prime density is 0.1489, so an 8-tuple of diameter 26 averages
  1.192 primes - above 1, so some m must carry two; verified directly, 666 such m, the best at
  m = 1601 carrying 7 primes among the 8 offsets.
- STATUS: exact, matching known values.
- WHERE: docs/twin-prime-program.md section 14g.
- LIMITS: "In that same range the truth is a gap of 2, with 50 twin pairs present, while the
  strongest statement the framework yields without sieve input is a gap of 26."

### Rule capacity forces an opening in the section, up to y = 88
- STATEMENT For 75 values of y between 6 and 88 there is a `T <= y^2` where the candidate columns
  outnumber the maximum possible strikes, so a twin pair MUST exist in `(y, T]` - proved by counting
  rules, with no sieve theory. The largest is y = 88, window `(88, 115]`: 5 candidates against at
  most 4 strikes. The mechanism dies exactly when the sixth odd gear becomes root-active, at
  `sqrt(T)` near 13, i.e. y near 90; past 88 it never succeeds again, because 11 becomes root-active
  once the window reaches 121 and no window can stay above y while below 121.
- CALCULATES: an unconditional forced opening inside the window for y = 6..64, 67-70, 73-76, 79-82,
  85-88.
- STATUS: exact (corrected figure).
- WHERE: docs/twin-prime-program.md sections 1e-1f.
- LIMITS: from y about 90 the rule set has enough capacity to cover the candidates.

### The class tree, and the sideways step
- STATEMENT A node at level i is a residue class mod `P_i` - a description of a column under every
  umbrella so far. ADDING GEAR q SPLITS EACH NODE INTO q CHILDREN (the q phases the new gear can
  present), and the turn law kills exactly 2 of them; `q-2` survive, the shield-child among them. The
  tree is NEVER EXTINCT (`prod(q-2) >= 1`). But following open branches controls OPENNESS, NOT
  POSITION: when a branch dies and the search steps to a sibling, the sibling class's smallest
  representative can jump by PRIMORIAL-SCALE amounts (changing one level's residue moves the
  representative by that level's idempotent). "The tree provably always has open branches, and one
  within `F_k(y)` of any point - but bounding the SIDEWAYS DISTANCE to the nearest open branch inside
  the window is Reduction A itself. Every route in the programme is an attempt to bound the sideways
  step."
- CALCULATES: the opening set of `{5..q}` from that of `{5..p}` by CRT, class by class.
- STATUS: proven (turn law, non-extinction, sound prune); the sideways bound is the open problem.
- WHERE: docs/class-tree.md; docs/twin-prime-program.md sections 1h, 17e.
- LIMITS: "the tree's infinite paths are PROFINITE integers; only the paths that stay small are
  twins" - and EVERY INTEGER IS EVENTUALLY BLOCKED BY ITSELF (at level `y >= m` the gear `q = m`
  divides m), so no integer is admissible at all levels.

---

## REFUTED OR SUPERSEDED ALIGNMENT CLAIMS

- Tooth-sharing changes the survivor count: REFUTED - over full periods sharing changes nothing
  (prod(q-2) conservation); the mechanism is purely positional. Both guaranteed wasted kills land on
  already-decided columns (self-block column, product column), so ZERO new openings.
  [archive lateral round 1; attempts-map I.4]
- The real phase vector +-u' is EXTREMAL for something: REFUTED by exact full enumeration - it ranks
  1716th of 11550 on overcount and 2536th on lone in the {5,7,11} two-teeth space; argmax/argmin only
  in the degenerate {5,7} mirror space. "Special point of phase space" means only "the census is
  deterministic". [archive lateral round 2]
- Umbrella nesting is a separate mechanism: REFUTED - any two gears' short umbrellas are concentric
  at joint shields; only the coinciding edges are twin-specific, and those are the +-u' pinned
  classes. One mechanism total. [archive lateral round 1]
- The adjacent-frame chain condition {phi, phi+1}: WRONG FRAME - its k=2 count is prod(q-4), the
  domino count; the k-frame teeth are never adjacent. Superseded by {phi, phi+s},
  s = 3^{-1} mod q. [docs/chain-conditions.md, "The frame trap"]
- k <= 3 as a universal chain bound: REFUTED - k = 4 exists at (gears<=29, q=31), four qualifying
  triples, word (10,21,10). "There is no universal bound k <= 3." [docs/chain-conditions.md addendum]
- Downward exclusion halts at the first gear with a coprime in the window: SUPERSEDED by the SQUARE
  GATE - it halts at the first q with q^2 - 2 prime, because a gear's square is its first root kill
  (q^2 < q*r), met before any coprime. [docs/class-tree.md]
- "Perfect alternation" of the record-run side word: CORRECTED - none of the six L=13 instances is
  strictly L/R alternating (the landmark reads RLLRRLLLLRLRL, with an LLLL block); perfect
  alternation is LOAD-only (one prime per column), side words are blocky. [archive mechanic round 7]
- L* = 13 as a wall: REFUTED as a wall, kept as a record - it stands to member 7.2e10 but RECURS six
  times, and the L->L+1 rate ratio ~0.3 puts the first L=14 within members ~1e11-1e12. "Do not build
  bounds on 13." [archive mechanic round 7]
- The drift recursion for near-top addresses: REFUTED (reachability 18/20 -> 0/4). The address is
  LOCAL - address = pin(word) - not inherited from the smaller machine. [archive lateral round 10]
- Hub enrichment at the binding loci: REFUTED - hub-rate/ambient 0.999 / 1.006; near-binding
  stretches are NOT hub-enriched, despite record stretches being bracketed by hubs.
  [archive lateral round 5; attempts-map section 5]
- Mirror-awareness buys anything at moment level: REFUTED by a two-line theorem - k -> -k swaps
  omega_L/omega_R and fixes m, so all mirror-augmented moments double and every ratio is invariant.
  Mirror-awareness is VACUOUS at moment level, at any order. [archive constructor round 6]
- Raw spectrum flatness: REFUTED at 5 of 15 machine-depth pairs in round 17; REPAIRED (not patched)
  in round 19 by suppression-corrected flatness, which holds at all 15.
  [archive/agents-shared r17 -> r19]
- The monotone envelope as a machine law (bigger span -> bigger max flank): REFUTED at machine 29
  (span 21 -> flank 27 vs span 25 -> flank 30, k = 133,490,560). It holds WITHIN every step (19/19)
  but the envelope follows OCCURRENCE COUNT, not span. [archive mechanic round 19]
- Gap 24's absence is a covering obstruction: REFUTED - residual demand vs purchasable supply leaves
  slack 8-16 at every g; the absence is selection plus rarity. "Don't hunt one."
  [archive lateral round 19]
- Caps at machine 19 in the tier-C scan: REFUTED as an ARTEFACT OF THE OLD ENCODING; an
  allocation-free scan plus restricting starts to openings (density 0.234, a 4.3x cut) plus tight
  fuel takes machine 19 from 86 min to ~20 min. [archive formalist round 19, correcting round 15]
- "F < q is the onset condition": WRONG NAMING - by onset_gate it is precisely the NO-PADDING
  regime; heading and docstring had overclaimed, and the padding count bound p <= F/q + 5/6 GROWS.
  [archive formalist round 19]
- The primorial-scale unwind always yields twins: REFUTED - nudge home 595 of the {5,7,11,13}
  machine is (3569,3571) with 3569 = 43*83. Openness beyond the horizon is not twinhood.
  [docs/class-tree.md]
- Observed tooth patterns (a) low/high alternation on successive gears and (b) tooth differences
  running 3,5,7,9,11,13: BOTH BREAK - at 23->29 and 31->37 (prime gaps of 6) the mod-6 family
  repeats, the alternation stalls, and the difference jumps by 4, skipping an odd. Only the
  u' = round(q/6) identity survives. [docs/umbrellas-and-shields.md]
- Extending a joint umbrella rightward from the certifying set of its first column: BUG - it claims
  columns where the tower has activated a new gear inside the interval (a square crossing
  mid-umbrella); one false twin per large window. Fix: judge every column at its own graded depth.
  [docs/class-tree.md]

- The mex law (the first opening = mex of `{u_q} u {q-u_q}`): held to y = 37, FAILED at 41 (gave 20
  against the truth 87). Structural reason: any rule carrying a bounded number of teeth per gear must
  under-block. [twin-prime-program.md section 31d; handover.md section 6 item 1]
- "Uniform adjacency is the only failure mode of the covering bound": REFUTED - `{5,7,11}` has 56
  failures of which 55 are NOT uniformly adjacent (e.g. separations (1,1,2), ratio 1.0534); 511 of
  512 for `{5,7,11,13}`. [covering-bound-route.md section 13a]
- The gear-3 lemma ("gear 3 present implies the bound for ANY separations"): REFUTED by
  `{3,5,7,11,13,17}` with separations `(1,3,3,3,3,3)` at L = 6: `N(6) = 148485 > 147584.435`, ratio
  1.006102, verified twice. "This is the third overclaim in this area."
  [covering-bound-route.md section 14a]
- Monotonicity of the margin in the gear set: FALSE - `R_1` peaks at 1.705697 near q = 83 then falls
  to 1.679767. [covering-bound-route.md section 24a]
- The collapsing margin: a NORMALISATION ARTEFACT - the exact integer differences were GROWING
  (51555900 at y=19, 350759640 at y=23). [covering-bound-route.md section 26b]
- Log-concavity of N: FALSE, fails at L = 3. Tail-fraction bounds from `N(L) <= N(1) - (L-1)G(L)`:
  too crude from L = 6. [covering-bound-route.md sections 27a-27b]
- The universal hazard bound `h(L) >= 1/(F_h - L)`: TRUE but CIRCULAR - it presupposes F_h, so "an
  apparent complete proof for {3,5,7} built on it was hollow". [covering-bound-route.md section 22b]
- The finite-automaton / transfer-matrix route over the gap word: closed for two independent reasons
  - the antidictionary is infinite, and recovering the counts needs the automaton WEIGHTED by the CRT
  measure, whose weights are the counts themselves (circular in the same way as the hazard bound).
  Measured at y = 13: letter 1 at 0.127 in the pattern against 0.044 in the automaton, moving in
  opposite directions with y. [forbidden-configurations.md section 5]
- Per-gear conditional marginals fall under conditioning: FALSE and badly - they RISE, by up to 63%
  above `2/q`, failing at 36 of 53 values of L at y = 17. "Gear exhaustion is not the mechanism"; any
  proof must treat the gears jointly. [forbidden-configurations.md section 7]
- Weak negative association `h(L) >= prod (1 - marginal_q)`: fails narrowly - once at y = 17 (L = 2,
  short by 2.6%), twice each at y = 11 and 13. [forbidden-configurations.md section 7]
- The `3q` / `6q^2` reading of the tight block starts: refuted from both directions - at y = 71 the
  predicted 51 and 57 rank 19th and 21st of 21 (the LOOSEST). [forbidden-configurations.md section 8a]
- "F2 is attained at a maximal gap, and F2 - F is bounded": BOTH FALSE - at y = 29 `F2 = 55` comes
  from the adjacent pair (30, 25), and `F2 - F` reads 2, 4, 5, 7, 6, 5, 12, doubling at y = 29.
  [handover.md section 6 item 14]
- Any argument resting only on "gaps are multiples of 3 and at least 3": `{3,3,15}` satisfies both
  and violates the claim. [handover.md section 6 item 15]
- Sub-multiplicativity `N(a+b) * P <= N(a) * N(b)`: FALSE - `{3,5,7}` at a = b = 6 gives
  `N(12)*P = 630 > N(6)^2 = 576`. [covering-bound-route.md section 10a]
- Size-only induction on the gear count: the hypothesis is false - if S lies inside two residue
  classes mod `q_n` the last gear covers it alone; the induction must be SPREAD-AWARE.
  [covering-bound-route.md section 11b]
- "The per-J recipe does not scale": ITSELF REFUTED - 2548 contributing terms at L = 39 against a
  predicted `5.5e11`. "A scaling judgement made without building the thing cost a working route for
  most of the programme." [handover.md method audit note]
- `min_L h(L) = h(1)` as the target, and `h(1) = d/(1-d)` as the free base case: SUPERSEDED - the
  adjacent-frame `L = 1` is a grid artefact with no column-frame counterpart; what is needed is only
  `h(L) >= d`. [gear-recursion.md section 1; handover.md section 3 item 17a]
- "kappa settles near 0.68": SUPERSEDED - it is drifting down to the exactly computable
  `kappa(2) -> 2 - (11/3)C = 0.54477`. [review-2026-08-17.md section 2]
- Pathways 7.2 and 7.3 held together: THEY CONTRADICT - form (b) implies
  `F_k(y) <= 2 + log P_k/(-log(1-d)) ~ y log^2 y / 2.1`, which the quadratic law `F ~ 0.68 y^2/log y`
  crosses at `y ~ 400`. AT MOST ONE OF THE TWO IS TRUE, and the handover held both without noticing.
  [review-2026-08-17.md section 4]
- Per-step multiplicative bound `r <= (q'/q)^2`: FALSE at 6 of 12 steps. [archive constructor r8]
- The double-onset route as a contradiction: the required superdense bound is FALSE as a universal
  statement - 310 of 442 real windows have a twin-free onset prefix, so X's forced alternation is
  actually realised there. [archive constructor r2 sec 9]
- Deep spectrum flatness `F_{k_max+1}(M) - F(M) <= q'`: FALSE at 29->31 (`F_5 - F = 42` against
  q' = 31); later re-read as "a refutation of the WRONG DEPTH" and repaired by the suppression law.
  [archive constructor r17 -> r18 -> r19]
- Round-18's two-part target `(D-a) k_win <= 3` plus `(D-b) F_4 - F <= q'`: SUBSUMED - "no separate
  assumption about winning depth is needed, because the suppression term kills deep windows
  automatically". [archive constructor r19]
- Round-19's named target (the anti-correlation law as NECESSARY for (D)): OVER-SPECIFIED -
  independence alone clears every constrained case by 170x to 201,381x. [archive constructor r20]
- The exposure bound on `p_j` without the `1/rho` conditioning: "Uncorrected, the bound appeared to
  clear the requirement everywhere tested. Corrected, it does not." [archive constructor r20 sec 43]
- The literal-only merge algorithm (`merge_decompose.py`): incomplete, misses padded links -
  undershoot 71 vs `>= 88` at 31->37. The fully permissive version (`merge_general.py`, all spacings
  `{0, +-2u}` without alternation): too permissive - overshoot 45 vs 43 at 23->29 on the illegal word
  (10,10). [archive lateral r13 correction]
- "Longer literal words become profitable as lambda grows" as the 31->37 crossover mechanism: at best
  half the story - the crossover is a PADDING ONSET (the best literal configuration reaches only 71).
  [archive lateral r13]
- The asymptotic safety argument for lemma 2 (`excess <= span_max = 2q' + s <= 2.67q'`): WITHDRAWN -
  it used the cap-6 theorem, which is stated for LITERAL chains; padded runs are not capped by it.
  "Constructor should not build on it." [archive lateral r13]
- "1 of 4 k=4 fuel sites is phase-aligned" and "site 858111062 is sterile forever": BOTH WRONG, a
  one-window artefact - every fuel site fires exactly once per new-machine period, and all four fire
  (j = 12, 30, 0, 18). The associated hope of a firing multiplier on the ceiling does not exist:
  alignment is a DENSITY factor, never a count factor. [archive lateral r11 -> r12]
- "Fuel k_max and the record are decoupled" / "the realized k=4 merge gives G=52 while F(31)=58 comes
  from a k=3 site": WITHDRAWN, same one-window artefact. [archive lateral r11 -> r12]
- The round-14 padding lemma as a general ceiling: EXPIRES exactly at 37->41 (`F/2q'` and `F2/2q'`
  cross 1 there) - "yes for machines up to 31, and no asymptotically". [archive lateral r14]
- Round-16's "the ceiling stands on structure": TOO STRONG - the SHAPE law is permanent but the COUNT
  p is not, and `span <= F + O(q')`, NOT `O(q')`. [archive lateral r16 -> r17]
- `p <= 2`: NOT provable from the AP lemma - the (0,1) and (1,0) triples survive and are
  corridor-feasible first at q' = 43, so p = 3 is structurally permitted from 41->43 on.
  [archive lateral r17]
- The `supply^2/gaps ~ 5` estimate for double padding at 37->41: contradicted by the pre-registered
  prediction of NO double-padded run - that model counts pairs without asking whether the corridor
  admits the shape. RETRACTED by its own author when `hist_37[41]` gave supply ~6.08e4 not ~1e6
  (expected double-padded runs 0.017, not ~5). [archive lateral r15; archive mechanic r14 -> r16]
- Sufficiency of the padding onset rule (`F(M) >= q'` implies supply > 0): FALSE - machine 29 has
  `F = 43 >= 41` yet `supply(29,41) = 0` exactly. [archive mechanic r15]
- `k_max <= 3` everywhere: CORRECTED - it held only through 23->29; `k_max = 4` at 29->31 and 31->37.
  And the k=4 absence at 37->41 was SELF-DEMOTED as evidence, since N3 is suppressed 830x there so
  the conditioned expectation was 0.91 - "the test did not probe the cap; it re-measured arithmetic
  selection". [archive mechanic rounds 11-12]
- The smooth `e^-(q'/lambda)` model for padding share: REFUTED - measured 2.27e-4, 7.54e-7, 9.73e-6,
  4.23e-6 is erratic and non-monotone, off the exponential by 20-1000x. [archive mechanic r14]
- "L* = 13 is a wall": REFUTED - `L = 14` found at `k = 46,133,660,494`, Poisson-consistent with the
  constellation model, no deficit against Hardy-Littlewood. [archive mechanic r9]
- "Thinnest layer bands sit exactly at twin endpoints" as a machine obstruction: EXACT BUT TRIVIAL
  (`T = g(2p+g)/6`), and the "hostile thin band" reading is a DENSITY ARTEFACT - twin density in
  twin-endpoint bands equals generic density at every matched height, with the only deterministic
  obstruction being ONE dead centre column. [archive mechanic r10]
- "The open part of (D) is four addresses": SUPERSEDED within the same round by the qualifying
  spectrum and the span-resolved envelope - "there is now NO residual at any measured step under
  either refined criterion". [archive mechanic r17]
- Tier B (modulus lifting) as a source of new exclusions: ZERO new exclusions at all 16 word-step
  pairs; "B is not a tier at all here". [archive constructor r13 sec 25.4]
- Both-flanks-maximal exclusion as route-advancing: correct but OFF-TARGET - the binding flank pairs
  are mid-size. [archive constructor r16; archive formalist r16]
- `FS <= F` as a clean flank bound: REFUTED outright by the literal steps (`FS/F = 1.12` at 29->31,
  1.09F at 13->17); and round 14's `FS < F - q'/6` was a REQUIREMENT misread as a derived fact.
  [archive constructor r15]
- "cap <= 6 for ALL (t,s) residue pairs mod 35": FALSE - over all 1225 pairs the spectrum is
  {2,3,4,5,6,8,10,140}; the restriction to invertible classes mod 210 does real work.
  [archive formalist r13]
- Modelling gear 3 as a run-breaker like gears 5 and 7 in the cap walk: WRONG - gear 3 FILTERS the
  candidate list (a 3-inadmissible kill is skipped and the run continues); the wrong model gives caps
  2/4 instead of 6/10/12. [archive formalist r16]
- The F2 encoding quantifying over ALL window starts rather than openings: a REAL ERROR caught by the
  kernel (`decide` reported the proposition false; Python confirmed 1296 counterexamples). The
  corrected statement requires the window to start at an OPENING. [archive formalist r11]
- "Tier C caps at machine 19": an ARTEFACT OF THE OLD ENCODING - restricting starts to openings takes
  machine 19 from 86 min to ~20 min and machine 23 from 33 h to ~7 h. [archive formalist r18]
- "Tooth alternation FAILS for `3 | e`" (W3): the OBSERVATION was real but the law it was tested
  against was the pre-padding one - a same-tooth adjacency is a legal PADDED link, not a violation.
  [archive harvester r10 -> r11]
- "The compatible-word set is not a function of `q' mod 105`" (73/73 mismatches): own bug - letters
  are q'-sized VALUES while the claim is about RESIDUES; corrected, zero mismatches.
  [archive harvester r13]
- The "frame conflict" over the padded-link cost ("exactly q'" vs "at least 3q'"): NOT A CONFLICT -
  q' in COLUMN units, 3q' halved, 6q' in members. [archive harvester r12]
- "26,366 of them" as a count of padded LINKS: it is the SUPPLY (gaps of M equal to exactly q'); a
  link additionally needs its endpoint on a tooth, about 1,400 of them. [archive harvester r12]
- The flag that `gcd(e,105) = 15` and 105 are "exactly where a budget could fail": the computation
  says the OPPOSITE - those two classes have the SMALLEST increments of all d tested (max 0.632 and
  0.483 against 1.235 for twins). DENSITY WINS. [archive harvester r14]
- The harvester's own prediction that Ziller-Morack Conjecture 6 would be breached at y = 17
  (extrapolated ~330 against the bound 272): REFUTED by its own computation in the same round
  (`h_2(17) = 192`, holds with 29.4% margin). Recorded rather than dropped. [archive harvester r16]
- The mirror-canonical `o5` halving in the pruned record search: UNSOUND combined with left-tautness
  (reflection maps left-taut to RIGHT-taut coverings); removed. [archive harvester r8]
- The `A(G)` mod-35 endpoint refinement inside the incremental covering loop: deliberately NOT used -
  it conditions on the gap length G and is valid only at the maximum, not per-L.
  [archive harvester r8]
- Reflection symmetry in centred coordinates as a positional constraint: "The mirror of the survivor
  set is the survivor set, and the symmetry constrains nothing about position. Closed."
  [twin-prime-program.md section 14]
- The `q = 3` pinning as a descent: self-similar but buys nothing further - pinning needs `q - 2 = 1`,
  which happens only at gear 3. Recorded as a closed lead. [twin-prime-program.md section 13a]
- The two-scale bound: not valid, and decisively so - the rigorous version's side condition
  `u < 1/(2 F(2,z))` fails at every step and worsens with z. [twin-prime-program.md section 14c]
- Single-to-twin transfer `F(2,y) <= K F(1,y)`: `F2/F1` climbs 3.00 to 5.61 with no ceiling.
  [twin-prime-program.md section 14e]
- The multiplicity / `omega` extension as the parity input: "the extension exists, and provably does
  not help" - `lambda(m)` is not a function of `m mod P(y)` for any y.
  [twin-prime-program.md section 14i]
- Low-order spectral truncation as a localisation shortcut: it localises perfectly but the quantity
  being thresholded is just "do all gears expose" - the spectrum contains no compression of the
  localisation information. [twin-prime-program.md section 35c]
- The rotation floor `ceil(6m/q)`: must be `(6m - 1) // q`; the ceiling emitted `1000000001` as a
  twin. The spectral CRT component `k mod q`: must be `k t_q mod q` with `t_q = (P/q)^{-1} mod q` -
  "it happened to agree at k = 0, 143, 1001, which is exactly where the twist is trivial". The
  inverse branch swap for `q = 1 mod 6` drops a gear and reports a false twin at `k = 10^12`.
  [twin-prime-program.md sections 27b, 35a, 21b]
- The `L=6` inclusion-exclusion pruning rule discarding subsets whose first interior point is 3: a
  BUG - `{0,3,9}` holds no three points spaced 3 apart; the error produced negative gap counts (-4 at
  L = 21) and hazard -1.43. The correct test is for an actual chain `s, s+3, s+6` inside the set.
  [covering-bound-route.md section 19a]
- The `psi` factoriser bug: `odd_prime_factors(6)` returned `[6]`, injecting a spurious
  `(6-2)/(6-4) = 2` into `psi(6)` and doubling it. [forbidden-configurations.md section 8c]
- "`C <= 1.10`" for the increment law: refuted by filling in y = 37, where C = 1.354.
  [gear-recursion.md section 5]
- The flank alphabet `{1..5}` as a general fact: SCOPED DOWN to a first-flank fact only - the max
  flank part grows 7 -> 13 with y. [archive lateral r9 -> r11]
- "No smooth law, only the histogram" (about gap-value erraticity): "is not right" - there is a law,
  multiplicative and arithmetic, with a three-line closed form, accounting for about a quarter of what
  was called noise. [archive lateral r18]
- Near-top word-shape families stabilising across machines: NEGATIVE - cross-machine full-shape
  recurrence is ZERO at all five machines; observed halves are 3.2% of admissible and essentially
  disjoint per machine. [archive lateral r11]

## EARLY IDEAS NEVER FOLLOWED UP (only those the record itself names as dropped)

- CORR formula-isation to distinct-double counts in closed form: "mechanically expandable to floor
  arithmetic via higher (s_L,s_R) product-pair terms if needed - DEFERRED".
  [archive lateral round 3]
- The quantitative half of the g=2 uniqueness theorem (depth ~P/(6g) for g > 2 and the mod-6
  alignment rate): "stays paper-side (research/split_gap_law.py) - PRICED, NOT FORMALISED".
  [archive harvester round 2]
- The layer-band descent: "STOPPED" after one page, per caution, at limit event T1 (a prime in every
  band, Legendre class). Flagged as an UNEXAMINED MACHINE EVENT: the thinnest bands sit exactly at
  twin endpoints, "so far treated only as an obstacle; uninterrogated as a mechanism".
  [archive constructor rounds 2-3; attempts-map II.4 and open avenue 6]
- The interior-condition factorisation for the exposed-set autocorrelation: named as WHY IT STOPPED
  THERE - endpoint exposure is a conjunction so it factorises by CRT; the interior condition is a
  DISJUNCTION and does not. Named next constructs `c_q(g1,g2)` (gear x lag-pair) and the
  autocorrelation at the padded lag q'. [archive lateral round 19]
- The COVERABILITY SPECTRUM COV(M): named construct for the next round, not built inside rounds
  1-19. [archive mechanic round 19]
- The frequency-space / phase look: "the earlier frequency-space / phase look was ABANDONED EARLY" -
  recorded in the round-20 human directive, which asks for it to be re-entered with the round-19
  objects in hand. [docs/proof-search/agents-shared.md, HUMAN DIRECTIVE 2026-08-23]
- Half-winding (using the mirror to fix a subset's behaviour in half its primorial): "true, but the
  mismatch under attack is e^y against y^2, so the factor 2 is CONCEPTUAL rather than asymptotic".
  [docs/class-tree.md]
- THE RENEWAL FACTOR - "a closed-form lower bound on `P(no opening strictly between | both endpoints
  exposed)` at separation v. That single factor is the ENTIRE remaining gap between the rigorous
  exposure bound and sufficiency. I did not build it this round." Still open at the end of the
  archive. [archive constructor r20 sec 49]
- The anti-correlation law as a formula for `p_j`: "Not built this round because it needs a joint
  gap-pair census at separations 1..5 over whole periods." [archive constructor r19]
- `c_q(g1, g2)` (the gear x lag-pair autocorrelation) and the autocorrelation at the padded lag q':
  named at round 18 as "two cheap unbuilt relatives, flagged for anyone"; both were built in round 20
  as the n-point form and the `3q' +- 1` padded-lag law. [archive lateral r18]
- The full gap correlation function (the `(g+1)`-point object: 2 exposed ends plus g-1 covered
  interiors) - "Bigger build, and where the complexity actually lives", explicitly not built.
  [archive lateral r18]
- Extending the AP lemma to gear 7 ("do SIX openings in q'-AP become forbidden?"): offered as the
  next step at round 16; no round in the archive reports it. [archive lateral r16]
- The 31->37 site-residue histogram over the full 3.34e10 period: "Still running at write-up time...
  Not load-bearing now that firing is once-per-period by the law"; no result reported later.
  [archive lateral r12]
- The medium-medium adjacency question for alpha1: "a concrete next target but NOT closed this
  round"; converted at round 10 into a word-level CRT check and never reported as closed anywhere.
  [archive lateral r9]
- The Jacobsthal push "does the 32-corridor survive gears <= 100": proposed at round 8; the log only
  ever records the check through gear 23. [archive lateral r8]
- The mod-5005 / uniformity-in-y extension of the adjacency certificate: "the class tier needs
  mod-5005 at scale; uniformity-in-y still open". [archive constructor r10]
- Uniformity of the near-top word grammar: named as "the one open piece" of machine-independent
  alpha1; never closed. [archive lateral r10]
- The 37->41 and 41->43 next-step tests: priced but not run - "confirming that its winners are
  literal would show the padded tier is intermittent rather than growing". [archive lateral r14]
- The falsification criteria (a)-(e) for the k_max census, including "(e) the 31->37 census: literal
  k = 5 or 6 there is CONSISTENT (cap-6 gear); k = 7+ anywhere falsifies the absolute cap": filed,
  not resolved. [archive constructor r11]
- The `k_win >= 4` hunt at machines 31, 37, 41: running at filing, outcome not on record.
  [archive mechanic r17]
- The pre-registered predictions for 31->37 (that the length-4 and length-5 compatible words have
  ZERO occurrences, since `N_5 = 0`), for 37->41 and for 41->43: filed as tests, outcome not on
  record in the archive. [archive mechanic r17 sec 7]
- The L=15 saturated-run hunt: never landed within the archive; last status 69.5%, members to
  ~1.4e12, the L=14 record unbeaten. [archive mechanic r9]
- The closed form for the multiplicity growth `S_pair/n2`: "no closed form fitted this round
  (candidate: second moment of active-pair density)"; never returned to. [archive mechanic r4]
- The mechanism behind the residue law of the gap histogram: "It is not the naive endpoint-survival
  count ... Unexplained." [archive mechanic r17]
- The double-padding hunt in general: "plausibly COMPUTATIONALLY OUT OF RANGE rather than merely
  unobserved - an honest limit, and a case where only a structural argument can decide."
  [archive mechanic r16]
- Formalising the spectrum `F_j` / per-J windows: "assessed, deliberately not built - formalising
  `F_j` would formalise a route already known not to close (D)". The reusable piece named: replacing
  the literal 2 in `Machine17.pair25T` by j gives exactly `F_j(M) <= B`, and the two-witness
  extraction generalises to j witnesses. [archive formalist r17]
- The `n > 3` gear inclusion-exclusion: "assessed and not forced (needs either iterated three_sets_ie
  with `2^n`-way flattens - mechanical but voluminous - or mathlib's signed inclusion-exclusion over
  Z; nothing conceptually new, deferred until the team needs it)." [archive harvester rounds 6-7]
- The (A) word-list ENUMERATION as a kernel object: "currently computed, not checked" - proposed as
  the next target and not executed within the archive. [archive formalist r18]
- Tier B (mod 385) for the 19->23 gap that tier A leaves open: dropped mid-round by the coordinator;
  nothing had been started. [archive formalist r16]
- The `h(2) >= d` product inequality: offered as the standing "alternative" next target in formalist
  rounds 1 through 7 and never taken. [archive formalist rounds 1-7]
- The quantitative half of the g > 2 pinning depth (`~P/(6g)` and the mod-6 alignment rate): "stays
  paper-side ... priced, not formalised". [archive harvester rounds 2 and 7]
- Late steps of the per-gap budget audit: "NOT VERIFIED (stated plainly): steps beyond 23->29 for any
  d ... Also unchecked: gcd classes 7, 21, 35 (d = 14, 42, 70)." [archive harvester r14]
- The y = 19 decision for Conjecture 6: "the honest status is UNDECIDED. Cost: the full scan is
  ~1.2e13 operations (out of reach here); a sharper attack would enumerate candidate delta-profiles
  directly by CRT - well within reach for a future round." [archive harvester r17]
- F(2,53) / F(2,59): the decisive measurement, named as outstanding and never completed - standing at
  `>= 416` (review), `>= 420` (synthesis), `>= 426` (pruned run). Every strategic choice in the
  growth-law question depends on it. [review-2026-08-17.md section 7a; archive harvester r14]
- The multiplicative accounting `F(M+q)/F(M)` rather than `F(M+q) - F(M)`: named by the review as
  NEVER TRIED in the corpus ("5.4 and 7.1 are both additive"); opened in round 8 as the tolerance
  route. [review-2026-08-17.md section 7 point 3]
- Reflection pairing ("pair each killed opening with its mirror image") and window-and-divisor
  lockstep ("if the bias has a fixed sign, the growth form closes"): two routes named "worth trying
  next" in the twin-prime-program and never returned to under those names.
  [twin-prime-program.md section 9]
- The self-reference question left live: "whether the self-blocked set at a level - which is PRECISELY
  the previous level's twins - CONSTRAINS that level's opening set, rather than merely being
  subtracted from it." [twin-prime-program.md section 17e]
- The sweep's efficiency fix ("a queue keyed on when each gear next shuts, costing one `d_q` per gear
  to build"): stated, not built. [twin-prime-program.md section 24c]
- The easier sibling: "forbidding one residue per coordinate instead of two gives the gaps between
  consecutive units mod P, which is Jacobsthal's function. Measured: largest consecutive-unit gap is
  6, 10, 14, 22, 26, 34 for y = 5..19. No closed form is known for that either."
  [twin-prime-program.md section 29e]
- Identifying a factor-restriction mechanism that is neither order-type nor congruence-type - or
  showing the two types are exhaustive: the concrete open problem the constructor line left behind.
  [twin-prime-program.md section 30d]
- Cofactor reindexing, exposure-window relationship tables, phase, frequency-space localisation, the
  gcd/order/unit-group validators, and forbidden configurations as an automaton route: each recorded
  as CLOSED as a route while its objects were kept ("the tables are correct and reusable"; "useful as
  a consistency check"). [handover.md sections 2.4, 2.6, 2.7, 2.8, 2.10, 2.11, 2.13]
- Negative association / Harris / FKG as tools: "the wrong tools; the escape indicators for one prime
  come from a CYCLIC SHIFT of a fixed pattern, and cyclic-shift families are not negatively
  associated in general." [covering-bound-route.md section 7]
- Generalising the machine further: abandoned after three failures - "there is no
  separation-independent generalisation, and the attempt to find one has now failed in both
  directions." [covering-bound-route.md sections 12-15]
- `add_gear` iterating on its own: "it does not yet iterate, because computing F two gears ahead needs
  the new gap WORD rather than its maximum; `add_gear` supplies the new histogram exactly but the
  ordering costs `A q` to materialise." [gear-recursion.md section 4a]
- Extending the exact record table beyond y = 47: "further grinding of this kind has low value ...
  going further needs a better algorithm than pruned depth-first search, not more patience."
  [twin-prime-program.md section 5]
- The exponential-witness construction claim: "stated but not machine-checked. It wants `ZMod q` as a
  field; an attempt in raw `Nat.mod` arithmetic did not go through and was REMOVED from
  proofs/BlockedSlots.lean rather than left broken there." [twin-prime-program.md section 14b]

## ABSENCES WORTH REPORTING

- NO MATRIX OR TRANSFER-MATRIX FORMULATION of the machine exists anywhere in the archive or the early
  design docs. The nearest objects are (a) the single-cycle walk reduction - the cap walk's state
  space `(position mod 105, parity)` is a SINGLE 210-cycle - and (b) the transfer-matrix route over
  the gap word, which is recorded as REFUTED in both its forms (infinite antidictionary; and letter
  statistics count which words CAN occur, y-independent, while the counts scale with P). The
  matrix/transfer-matrix frame first appears as a HUMAN DIRECTIVE for round 20, i.e. after the
  archived rounds.
- NO GEAR-ORDER SORTING RULE exists. The nearest statements are the ROOT ORDERING ("shadow < q^2,
  square, then coprimes"), the smallest-gear-first splitting recursion, the "large gears force
  nothing new" census, and the scan-start restriction to openings.
- The words CORRIDOR and LITERAL CAP do not appear in covering-bound-route.md,
  forbidden-configurations.md, gear-recursion.md, handover.md, status.md, ideas-from-the-session.md,
  review-2026-08-17.md or synthesis-2026-08-18.md - they are proof-search vocabulary, introduced in
  rounds 8-11.
- The lateral archive file has NO ROUND 19 (it runs round 18 then round 20); the constructor file has
  NO ROUND 7 (the round-7 deliverable is docs/proof-search/attempts-map.md); the mechanic file ends
  at ROUND 17.

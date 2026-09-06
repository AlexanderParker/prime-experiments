# The top machine on its own terms (branch R4.b)

> Law numbers in this document map to the project-wide register
> (`research/proof/law_register.md`): **L1-L21 = W1-W21**.  The two unmechanised facts
> called "W1" and "W2" in section 4 below are NOT register entries W1 and W2; they are
> closed at register W24 and W32 respectively.

Parent: R4, the period-scale formulation, and specifically the owner's CONSTRUCTION RULE of
2026-09-06: study the top machine on its own, on the raw line, not in the bottom's coordinate
and not against the bottom's kills. The observation that spawned this branch is the owner's
analogy: the bottom machine is the motor, understood in depth; the top machine is the wheels,
so far inferred only from the motor's odd behaviour. This branch builds the wheels.

Prior data used (the only prior data permitted here): research/proof/period_scale.md 3.1, 3.5,
3.11 - the top machine's mirror about the origin, its in-range density, its longest closed runs
above the placement prefix (7, 24, 30, 58, 104 at q = 11..23) and the placement residue law.
Those were all measured in the BOTTOM's column coordinate; this branch is on the raw line, so
they are context, not results to reproduce.

Inspiration only (bottom machine, one line each, never compared during construction):
docs/proofs/02-tooth-rule.md (two teeth at +-6^-1, arcs 2u and q - 2u, the shield at 0),
03-always-open-columns.md (column 0, the antipode (P +- 1)/2, the mirror k -> -k, the affine
symmetry group (Z/2)^m, even gap counts), 04-alignment-law.md (longest run of consecutive
openings = the long arc of the smallest gear; dominoes prod(q - 4)), 05-adding-a-gear.md
(copies and phases, hit law, chain law, merge law, the grammar T1-T5 with letters
a = 2u_g, b = g - 2u_g).

---

## 0. Setup: what the top machine is

**The line.** The integers, unfolded. No anchor 2, 3, 5; no 6-fold; no columns.

**The gears.** The primes in (q, Q]. Gear g strikes every multiple of g.

**The object.** A *pair* is (n, n + 2). The pair is indexed by its lower member n, so the pair
coordinate IS the raw line. Gear g strikes the pair n iff g | n or g | n + 2, i.e. iff

        n = 0 (mod g)   or   n = -2 (mod g).

So in pair coordinates every gear has exactly **two teeth**, at 0 and at -2, of **separation 2**,
and g - 2 open residues. A pair struck by no gear of the machine is **open**.

**The wheel.** The wheel of a gear set G is W = prod_{g in G} g; the open pairs are a
W-periodic set. The wheel of the smallest three (or four) gears is the top machine's own anchor:
7 x 11 x 13 = 1001, 11 x 13 x 17 = 2431, 13 x 17 x 19 = 4199, and the four-gear wheels
7 x 11 x 13 x 17 = 17017, 11 x 13 x 17 x 19 = 46189, 13 x 17 x 19 x 23 = 96577.

**Vocabulary for this branch** (the top machine's own): gear, tooth, pair, open pair, wheel,
slot (= an open residue class of the wheel), run (maximal block of consecutive open pairs),
**record F_top** (longest block of consecutive n with NO open pair), gap (difference of
consecutive open pairs; gap = record block length + 1), letter (a spacing class of a gear's
teeth), domino (two adjacent open pairs n, n + 1, i.e. the quadruple (n, n+2, n+3)).

**The mirror.** n -> -n - 2 exchanges the two members of a pair; it is the top machine's
reflection.

**Ranges.** In use the machine's period never fits the range, so the second half of the branch
is the machine on [1, N] with N far below W.

---

## 1. Pre-registered predictions and scorecard

Written before any computation. Each is followed by what would refute it.

**P1 (wheel count).** The wheel of G has exactly prod_{g in G}(g - 2) open pairs per turn.
Refuted by any wheel whose exact count differs.

**P2 (arcs, the top machine's asymmetry).** The open residues of a single gear g form exactly
two arcs: a **long arc** of length g - 3 (residues 1 .. g - 3) and a **short arc of length
exactly 1** (the residue -1). Hence the top machine's own asymmetry between its two arcs is
(g - 3) : 1, not the bottom's roughly 2 : 1, and the alignment-law analogue reads: the longest
run of consecutive open pairs of any gear set with smallest gear q' is exactly q' - 3.
Refuted by a run longer than q' - 3 in any wheel, or by an arc structure other than
(g - 3, 1).

**P3 (the always-open pair and the origin clump).** n = -1 (mod W) is open for every gear (its
members are -1 and +1). More: every n whose two members have no prime factor in G is open, so
near the origin the wheel carries a forced clump: all n in [-(q' - 1), q' - 3] are open except
n = -2 and n = 0. Predicted shape: a run of q' - 3 open pairs, then struck, open, struck, then
a run of q' - 3 open pairs. Refuted by a struck pair strictly inside either predicted run.

**P4 (the mirror and the antipode).** n -> -n - 2 maps the open set of any gear set onto itself
with 0 mismatches. Its unique fixed point mod W is n = -1. The antipode analogue: n = 2 and
n = -4 (mod W) are open for every gear set (members 2, 4 and -4, -2), and they are adjacent
under the mirror in the sense that the mirror exchanges them. Predicted symmetry group of the
open-pair set among affine maps n -> c n + b: exactly the maps n -> c(n + 1) - 1 with
c = +-1 (mod g) for every gear g, i.e. (Z/2)^{|G|}; of these only c = +-1 (mod W) preserve
adjacency, so the adjacency-preserving group is Z/2. Refuted by an extra symmetry or a missing
one.

**P5 (the conjugacy: what the top machine IS).** The map n -> k = 6^{-1}(n + 1) (mod W) carries
the top machine's open-pair set exactly onto the opening set of the same-gear machine written in
the bottom's column coordinate (teeth +-6^{-1}), with 0 mismatches. More generally the machine
with teeth {0, -2t} is affinely conjugate to the machine with teeth {0, -2} for every t coprime
to W. CONSEQUENCE PREDICTED: every counting/symmetry law transfers unchanged between the two
coordinates, and every metric law (runs, gaps, records, alignment) does NOT, because the
conjugating map is not an isometry. Refuted by a mismatch in the conjugacy, or by a counting law
that fails to transfer.

**P6 (chain law in pair coordinates).** Two open pairs n < n' of a machine M are both struck by
a new gear g (in some copy of the period) iff n' - n = 0, +2 or -2 (mod g). 0 exceptions
predicted. Refuted by one exception.

**P7 (letters and alternation).** The letters of gear g are a = 2 and b = g - 2 (a + b = g). In
a run of consecutive open pairs of M all struck by g, the nonzero letter classes strictly
alternate 2, g - 2, 2, ... . 0 exceptions predicted. CONSEQUENCE PREDICTED: the fuel cap becomes
"k <= (x_k - x_0)/2", which is nearly vacuous, so unlike the bottom machine the short letter
puts almost no brake on how many open pairs one gear can sweep - the top machine's grammar is
cheap.

**P8 (merge law).** Every gap of M + g is either a gap of M or a sum of consecutive gaps of M
whose interior open pairs are all struck by g. 0 exceptions predicted.

**P9 (dominoes and the run spectrum).** The number of adjacent open pairs (n and n + 1 both
open) per wheel is exactly prod(g - 4). Unlike the bottom machine (where gear 5 forces runs of
at most 2, so openings are points and dominoes), the top machine has runs up to q' - 3, so its
run spectrum is a real spectrum. Predicted: the number of runs of length exactly L falls
geometrically in L until the hard ceiling q' - 3.

**P10 (records and the ladder).** Adding gear g to a machine M increases the record by less than
g: F_top(M + g) < F_top(M) + g. Predicted at every step of every ladder tested. Also predicted:
the record is NOT at the origin (the origin is the clump), and the record's block is swept
mostly by the LARGEST gears of the machine, measured as sole-striker counts.

**P11 (the machine on a range).** On [1, N] with N far below W the density of open pairs exceeds
the product prod(1 - 2/g) (a Buchstab-type excess) and falls monotonically across the range; the
longest pair-free run inside [1, N] is far below the wheel's record.

**P12 (near the origin on a range).** On [1, N] the neighbourhood of every multiple of the
product of the smallest gears is unusually open (the origin clump repeats at every multiple of
the small gears' product only when the larger gears also miss), and the very first stretch above
0 is the most open stretch of the range.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| P1 | wheel count = prod(g - 2) | |
| P2 | arcs (g - 3, 1); longest run = q' - 3 | |
| P3 | always-open n = -1; origin clump | |
| P4 | mirror 0 mismatches, group (Z/2)^m, adjacency group Z/2 | |
| P5 | conjugacy to the column coordinate, 0 mismatches | |
| P6 | chain law 0 exceptions | |
| P7 | letters {2, g - 2}, alternation 0 exceptions | |
| P8 | merge law 0 exceptions | |
| P9 | dominoes prod(g - 4); run spectrum to q' - 3 | |
| P10 | F_top(M + g) < F_top(M) + g; record away from origin; top gears make it | |
| P11 | range density above the product, falling; range record below wheel record | |
| P12 | first stretch above 0 the most open | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r1/`, results (untracked) in `.../results/`:
`wheel.py` (the wheel and its slots, mirror, symmetry group, conjugacy),
`pairwise.py` (partner law, chain, merge, alternation, run spectrum, two-gear cells),
`cover.py` (the record as an exact covering problem), `validate.py` (the covering
formulation against a full-period scan, 15 of 15), `ladder.py` (the ladder,
the parity law, record composition), `range.py` (the machine on [1, N]),
`extras.py` (step-2 chains, gap spectrum, near-wheel density).
Every count below is exact over a full wheel period, or exact over the stated range.

---

## 3. Results

### 3.1 The wheel and its slots

Twelve wheels, exact over the full period.

| gears | W | open pairs | = prod(g-2) | longest run | = q'-3 | record F_top | at | multiplicity | dominoes | = prod(g-4) |
|---|---|---|---|---|---|---|---|---|---|---|
| 7,11,13 | 1,001 | 495 | yes | 4 | yes | 6 | 217 | 2 | 189 | yes |
| 11,13,17 | 2,431 | 1,485 | yes | 8 | yes | 5 | 49 | 18 | 819 | yes |
| 13,17,19 | 4,199 | 2,805 | yes | 10 | yes | 5 | 150 | 18 | 1,755 | yes |
| 17,19,23 | 7,429 | 5,355 | yes | 14 | yes | 5 | 456 | 18 | 3,705 | yes |
| 19,23,29 | 12,673 | 9,639 | yes | 16 | yes | 5 | 112 | 18 | 7,125 | yes |
| 23,29,31 | 20,677 | 16,443 | yes | 20 | yes | 5 | 2,229 | 18 | 12,825 | yes |
| 7,11,13,17 | 17,017 | 7,425 | yes | 4 | yes | 9 | 1,526 | 12 | 2,457 | yes |
| 11,13,17,19 | 46,189 | 25,245 | yes | 8 | yes | 8 | 1,779 | 24 | 12,285 | yes |
| 13,17,19,23 | 96,577 | 58,905 | yes | 10 | yes | 8 | 453 | 24 | 33,345 | yes |
| 17,19,23,29 | 215,441 | 144,585 | yes | 14 | yes | 8 | 5,127 | 24 | 92,625 | yes |
| 7,11,13,17,19 | 323,323 | 126,225 | yes | 4 | yes | 12 | 2,594 | 48 | 36,855 | yes |
| 11,13,17,19,23 | 1,062,347 | 530,145 | yes | 8 | yes | 10 | 24,354 | 24 | 233,415 | yes |

**The arcs.** Every gear, without exception in the twelve wheels: its open residues form
exactly two arcs, of lengths **g - 3** and **1**. The singleton arc is the residue -1. This is
the top machine's asymmetry, (g - 3) : 1, and the short side has collapsed to a single slot.

**The slot structure modulo small numbers.** The open pairs are equidistributed modulo 2, 3 and
6: in every wheel the six classes mod 6 differ from equality by at most 2 (11,13,17:
248, 247, 248, 248, 247, 247 of 1,485; 17,19,23,29: 24,098, 24,096, 24,096, 24,098, 24,099,
24,098 of 144,585). **There is no fold.** The top machine has no residue preference of the kind
2 and 3 impose on the bottom; the "one third" and the six-fold are properties of the anchor, not
of a gear machine.

**The run spectrum.** The number of maximal runs of exactly L consecutive open pairs is the
second difference of A(L) = prod_g (g - 2 - L), the number of starts of L consecutive open
pairs: 0 mismatches in 12 wheels over every L. Because A is a polynomial of degree m in L, the
run spectrum is a polynomial of degree m - 2: for a **three-gear wheel it is an arithmetic
progression of common difference exactly 6, whatever the gears** (17,19,23: 88, 82, 76, 70, 64,
58, 52, 46, 40, 34, 28, 22 for L = 2..13), and for a four-gear wheel a quadratic (second
difference 24: 1,162, 934, 730, 550, 394, 262). The top length L = q' - 3 is the exception, with
count exactly prod_g (g - q' + 1) (21, 35, 189, 385 at the four wheels checked).

**The gap spectrum.** Gaps between consecutive open pairs take the values 1, 2, 3, 5, 6, ... :
**the gap 4 never occurs, in any wheel or any range** (12 wheels, plus 27 range machines to
N = 10^7, 0 occurrences). The counts of gap 3 and gap 5 are exactly equal in every wheel whose
gears all exceed 7 (52 = 52, 68 = 68, 88 = 88, 112 = 112, 136 = 136, 1,162 = 1,162, 1,978 =
1,978, 3,386 = 3,386, 28,196 = 28,196) and unequal exactly when 7 is a gear (32 vs 34, 538 vs
590, 9,604 vs 10,766). Every gap length has an even count except length 1, in every wheel.

### 3.2 The always-open pair, the mirror, the symmetry group

- **n = -1 (mod W) is open for every gear set** (members -1 and +1): 12 of 12.
- **The mirror n -> -n - 2 maps the open-pair set onto itself: 0 mismatches in 12 wheels**
  (2.2 million residues). Its unique fixed point mod W is n = -1 (W odd, so 2n = -2 has one
  solution), and that fixed point is the always-open pair.
- **The antipode.** n = 2 (members 2, 4) and n = -4 (members -4, -2) are open for every gear
  set: 12 of 12. They are exchanged by the mirror. They are the images of the bottom's
  antipodal columns (P +- 1)/2 - but on the raw line they are **not adjacent**; the bottom's
  "gap of length 1 at the antipode" is a fact about the column coordinate, not about the
  machine.
- **The origin clump.** For gears >= q', every pair n with -(q' - 1) <= n <= q' - 3 is open
  except n = 0 and n = -2 (whose pairs contain 0). Measured at 7,11,13: n = -6..-3 open,
  -2 struck, -1 open, 0 struck, 1..4 open, 5 struck. The origin carries two runs of the maximal
  length q' - 3 separated by struck / shield / struck. The top machine's "column 0" is not one
  slot but a clump of 2(q' - 3) + 1 slots.
- **The symmetry group.** Brute force over all affine maps n -> c n + b of Z_W (W = 1,001):
  exactly 8 = 2^3 preserve the open set, and every one has the form
  **n -> c(n + 1) - 1 with c = +-1 (mod g) for each gear g**; of these exactly two,
  c = +-1 (mod W), preserve adjacency: the identity and the mirror. So the symmetry group is
  (Z/2)^m and the adjacency-preserving group is Z/2. One-line proof: an affine map preserves the
  open set iff it permutes each gear's tooth pair {0, -2}, which forces (c, b) = (1, 0) or
  (-1, -2) modulo that gear.

### 3.3 What the top machine IS: the conjugacy

**The map n -> k = 6^{-1}(n + 1) (mod W) carries the top machine's open-pair set exactly onto
the opening set of the same gears written with teeth +-6^{-1}: 0 mismatches in all twelve
wheels (2.2 million residues).** Proof in a line: n = 0 (mod g) iff 6k = 1, and n = -2 (mod g)
iff 6k = -1, so the teeth {0, -2} go to {6^{-1}, -6^{-1}}; 6 is invertible mod W because every
gear is at least 7.

The consequence is the branch's organising principle:

> Every **counting or symmetry** law of a gear machine is coordinate-free and transfers between
> the two coordinates unchanged. Every **metric** law - runs, gaps, records, arcs, letters,
> alignment - does not, because the conjugating map is not an isometry. The top machine's own
> laws are exactly the metric ones, and they are the ones the bottom machine cannot supply.

The only gears unmoved by the change of coordinate are g = 5 and g = 7, the two with u_g = 1:
their letters are {2, 3} and {2, 5} in both coordinates. For every larger gear the column
coordinate stretches the short letter from **2** to **2u_g ~ g/3**.

### 3.4 Pairwise laws

**Two gears.** Over the period gh the pairs both strike number exactly 4, for every pair tested
- (7,11), (11,13), (13,17), (17,19), (29,31), (7,13), (11,17) - **including twin gears**: the
four classes are n = 0/-2 (mod g) crossed with n = 0/-2 (mod h). For twin gears g, g + 2 one of
the four is n = g, the pair (g, g + 2): the two gears striking the pair that they themselves
are. Exactly one strikes: 2(h - 2) for g and 2(g - 2) for h; open (g - 2)(h - 2). All matched.

**The partner law (new, and the top machine's most characteristic fact).** If a gear strikes the
pair n it also strikes the pair n - 2 (if the tooth is 0) or n + 2 (if the tooth is -2). The
struck set of any single gear is a disjoint union of **distance-2 dominoes {x, x + 2}**, never
an isolated strike. **0 exceptions in 8 wheels** (5.6 million struck residues). Proof: the teeth
are 0 and -2, so n = 0 forces n - 2 = -2 and n = -2 forces n + 2 = 0.

**Corollary (the forbidden gap).** A gap of exactly 4 is impossible. Proof: it would need n + 2
struck with n and n + 4 both open, but the striker of n + 2 also strikes n or n + 4. Verified:
0 gaps of length 4 in 12 wheels and in every range machine to N = 10^7.

**Chain law.** For a machine M and a new gear g, two openings x < y of M are both struck by g in
some copy of M's period iff y - x = 0, +2 or -2 (mod g). **0 exceptions in 118,341 tested
opening pairs** over M = {7,11}+13, {7,11,13}+17, {11,13}+17, {11,13,17}+19.

**Merge law.** Every gap of M + g is a gap of M or a sum of consecutive gaps of M whose interior
openings are all struck by g. **0 exceptions in 34,646 gaps** over the same four steps.

**Letters and alternation.** The letters of gear g are a = 2 and b = g - 2 (a + b = g). In a run
of consecutive openings of M all struck by g, a spacing = 0 (mod g) keeps the tooth, a spacing
= 2 goes from tooth -2 to tooth 0, a spacing = g - 2 goes from tooth 0 to tooth -2, and the
nonzero classes strictly alternate. **0 exceptions in 816 struck runs.**

**The fuel cap is gone.** The bottom's T5 reads x_k - x_0 >= k a with a = 2u_g ~ g/3, a real
brake on how many openings one gear can sweep. On the raw line a = 2, so the cap reads
k <= (x_k - x_0)/2 and is vacuous: **the top machine's grammar is cheap**, and a single gear can
sweep half of any stretch.

**Dominoes, two kinds.** Adjacent open pairs n, n + 1 (the quadruple n, n+2, n+3): exactly
prod(g - 4) per wheel, 12 of 12. Open pairs sharing a member, n and n + 2 (the triple
n, n+2, n+4): exactly prod(g - 3) per wheel, 12 of 12. More generally the number of starts of a
step-2 chain of L open pairs is prod(g - 1 - L) and of a step-1 run of L open pairs is
prod(g - 2 - L): **0 mismatches over all L in 12 wheels**. Hence two ceilings,

        longest run of consecutive open pairs        = q' - 3
        longest chain n, n+2, n+4, ... of open pairs = q' - 2

both attained in all 12 wheels. The second has no bottom-machine analogue at all: the bottom's
anchor contains 3, so a triple of open columns cannot become a prime triple. The top machine
supports chains of length q' - 2.

### 3.5 The record and the ladder

**The record is a covering problem (exact).** A run of L consecutive struck pairs exists in the
period iff [0, L) can be covered by choosing, for each gear g, a phase s_g and taking
S_g = {x in [0,L) : (x + s_g) mod g in {0, g-2}}. CRT makes every phase vector realisable, so
this is an exact characterisation. **Validated against the full-period scan on 15 wheels:
15 agreements, 0 disagreements** (F = 6, 5, 5, 5, 5, 5, 9, 8, 8, 8, 12, 10, 9, 9, 9).

**The pieces.** A gear g > L + 1 can only contribute a **domino {x, x + 2}** or a singleton
inside the window (its teeth are 2 apart; the other route between them, g - 2, is longer than
the window). A gear g <= L + 1 can also contribute the **long letter {x, x + g - 2}**, and a
gear g <= L repeats. So the record block is a tiling of L consecutive integers by the gears'
letters: the record's composition is letters and nothing else - there are no flanks.

**THE PARITY LAW (new, exact, with proof).** If every gear exceeds 2m + 1, where m = |G|, then

        F_top(G) = 2m - (m mod 2),

that is 2m for an even number of gears and 2m - 1 for an odd number. **170 cases, 0 exceptions**
(all sets of m consecutive primes, m = 2..11, starting anywhere in 7..199 with q' > 2m + 1).
Proof: with all gears large every piece is a distance-2 domino or a singleton, so L <= 2m; a
distance-2 domino never crosses parity, so the evens of [0, L) and the odds of [0, L) must each
be partitioned by same-parity dominoes plus singletons, needing
ceil(ceil(L/2)/2) + ceil(floor(L/2)/2) pieces; at L = 2m that is 2 ceil(m/2), which exceeds m
exactly when m is odd, and at L = 2m - 1 it is m. **The record of a large-gear top machine is
decided by the parity of the number of gears and by nothing else** - not by the sizes of the
gears at all.

**The increment law in the large-gear regime.** The parity law forces increments alternating
**+3, +1, +3, +1, ...**, which is exactly what the ladders show while the gears stay large
(q' = 17: 3, 1, 3, 1, 3, 1, 3; q' = 19: 3, 1, 3, 1, 3, 1, 3; q' = 23: 3, 1, 3, 1, 3, 1, 3, 1).

**The ladders** (gears added one at a time from q'; every value exact except the last rung of
the first two, marked >=, where the covering search was cut off):

| q' | F_top by top gear Q |
|---|---|
| 7 | 7:1, 11:4, 13:6, 17:9, 19:12, 23:19, 29:25, 31:32, 37: >= 39 |
| 11 | 11:1, 13:4, 17:5, 19:8, 23:10, 29:16, 31:18, 37:24, 41:28, 43:34, 47: >= 37 |
| 13 | 13:1, 17:4, 19:5, 23:8, 29:9, 31:12, 37:16, 41:18, 43:24, 47:27, 53:33, 59:36 |
| 17 | 17:1, 19:4, 23:5, 29:8, 31:9, 37:12, 41:13, 43:16, 47:21, 53:24, 59:27, 61:32, 67:35 |
| 19 | 19:1, 23:4, 29:5, 31:8, 37:9, 41:12, 43:13, 47:16, 53:18, 59:21, 61:24, 67:27, 71:32, 73:35, 79:40 |
| 23 | 23:1, 29:4, 31:5, 37:8, 41:9, 43:12, 47:13, 53:16, 59:17, 61:20, 67:22, 71:25, 73:28, 79:33, 83:36, 89:40, 97:42 |

**The budget analogue is enormously slack.** Over all 69 exact ladder steps the increment
F_top(M + g) - F_top(M) never exceeds **7**, against new gears of size up to 97. There is no
sign of a budget of size g: the top machine's record grows **linearly in the number of gears**,
about 2 per gear, not with the gear.

**Which gears make the record.** The opposite of the bottom machine. In the optimal cover the
strike count per gear is about 2L/g, so the **smallest** gears do the work: at {7,...,31},
L = 32, the strikes are 7:9, 11:6, 13:6, 17:4, 19:4, 23:4, 29:2, 31:2. Each top gear contributes
exactly one domino. The record is made at the bottom of the top machine.

**Record multiplicity is universal.** The number of record blocks per wheel depends only on the
number of gears, not on the gears: **18 (m = 3, five wheels), 24 (m = 4, three wheels), 480
(m = 5, four wheels), 720 (m = 6, two wheels)** - 0 exceptions in 14 large-gear wheels. So are
the counts just below the record for even m (m = 4: 96, 24, 24 at gaps 7, 8, 9 in all three
wheels; m = 6: 6,480, 1,440, 720 at gaps 11, 12, 13 in both).

### 3.6 The machine on a range

**(a) A fixed gear set, wheel far above the range.** Eight gears; N = 10^5, 10^6, 10^7:

| gears | W | density at N = 10^5 | prod(1 - 2/g) | ratio | F in range (10^5 / 10^6 / 10^7) | wheel record |
|---|---|---|---|---|---|---|
| 7..31 | 6.69e9 | 0.310320 | 0.310458 | 0.9996 | 21 / 24 / 27 | 32 |
| 13..41 | 1.32e11 | 0.477890 | 0.478000 | 0.9998 | 15 / 16 / 17 | 18 |
| 19..47 | 1.20e12 | 0.584660 | 0.584475 | 1.0003 | 12 / 14 / 14 | 16 |

A top machine with a fixed gear set is, on a range 10^4 to 10^7 times shorter than its wheel,
**indistinguishable from periodic in density** (four to five figures), and its longest pair-free
run climbs slowly toward the wheel record without reaching it. Its 20 block densities are flat
to 0.1%. Non-periodicity is not a property of the top machine as such.

**(b) The machine as used, gears (q, sqrt(N)].** Here the gear set grows with the range:

| q | N | Z | gears | density | prod(1-2/g) | ratio | longest pair-free run | at |
|---|---|---|---|---|---|---|---|---|
| 5 | 10^7 | 3,137 | 443 | 0.072339 | 0.063963 | 1.131 | 3,006 | 161 |
| 7 | 10^7 | 3,137 | 442 | 0.103745 | 0.089549 | 1.159 | 2,718 | 449 |
| 11 | 10^7 | 3,137 | 441 | 0.128041 | 0.109449 | 1.170 | 2,088 | 1,079 |
| 13 | 10^7 | 3,137 | 440 | 0.151833 | 0.129348 | 1.174 | 1,166 | 2,001 |
| 17 | 10^7 | 3,137 | 439 | 0.171810 | 0.146595 | 1.172 | 618 | 2,549 |
| 19 | 10^7 | 3,137 | 438 | 0.191254 | 0.163841 | 1.167 | 227 | 2,661 |

Three facts. (i) The density **exceeds** the CRT product by 5-17% at every q and every N: the
Buchstab excess of a set defined by "no prime factor in (q, Z]", not a two-tooth sieve deficit.
(ii) The block densities fall monotonically after the first block (q = 11, N = 10^7: 0.1473 down
to 0.1167, a 26% fall), so the in-use machine, unlike the fixed one, is strongly non-uniform -
because its gear set is tied to the range, not because a gear machine is non-uniform.
(iii) **Every record sits below Z**: at N = 10^7 the record positions are 161, 449, 1,079,
2,001, 2,549, 2,661 against Z = 3,137.

**The gear zone (new).** For n <= Z the pair n is open iff both n and n + 2 are q-smooth,
because any prime factor of a member that exceeds q is itself a gear. So [1, Z] is almost
entirely struck - the gears strike their own homes - and **the longest pair-free run of the
in-use machine is always in the gear zone, immediately above the origin clump**. This is the
exact reverse of the wheel, where the origin is the most open place in the period. The origin
clump survives only out to q' - 3; then the gear zone begins.

**Near multiples of the small gears' wheel** (fixed 8-gear machines, N = 10^7, up to 3,000
multiples of q'q''q'''): the density in a window of +-5 is 1.62, 1.20, 1.05 times the mean; at
+-20 it is 1.16, 1.09, 1.15; at +-100 and +-500 it is 1.00. Each multiple of the small wheel
carries a short local clump about as wide as the small gears - the origin clump, repeated - and
nothing at any larger scale.

---

## 4. Laws

Numbered, with proof or exception count, in the top machine's own vocabulary.

**L1 (two teeth, separation 2).** In pair coordinates every gear g has exactly two teeth, at
n = 0 and n = -2 (mod g), and g - 2 slots. Proof: g | n or g | n + 2. *Counting; transfers.*

**L2 (the arcs).** The slots of one gear form two arcs, of lengths g - 3 and 1; the singleton
arc is n = -1, the **shield**. Proof: the teeth 0 and g - 2 cut Z_g into the runs 1..g-3 and
{g-1}. Verified: 12 wheels, every gear, 0 exceptions. *Metric; the bottom's arcs are
(2u_g - 1, g - 2u_g - 1).*

**L3 (the partner law).** Every strike of a gear has a partner strike of the same gear at
distance exactly 2 - at n - 2 if the tooth is 0, at n + 2 if the tooth is -2. The struck set of
one gear is a disjoint union of dominoes {x, x + 2}. Proved in one line; 0 exceptions in
5.6 million struck residues over 8 wheels. *New in form: the bottom's corresponding distance is
2u_g ~ g/3, so no such local law exists there.*

**L4 (the forbidden gap).** A gap of exactly 4 between consecutive open pairs is impossible in
any top machine. Proof: the striker of the middle pair would have to strike one of the two
bounding open pairs. 0 occurrences in 12 wheels and 27 range machines to N = 10^7. *New; no
bottom analogue.*

**L5 (the wheel count).** A wheel of gears G has exactly prod(g - 2) open pairs. 12 of 12.
*Counting; transfers.*

**L6 (the always-open pair and the origin clump).** n = -1 (mod W) is open for every gear set;
and with gears >= q' every n in [-(q'-1), q'-3] except n = 0 and n = -2 is open, so the origin
carries 2(q' - 3) + 1 slots in two maximal runs separated by struck / shield / struck. n = 2 and
n = -4 are also open for every gear set (the antipode analogue). 12 of 12. *The clump is new;
the single always-open slot transfers.*

**L7 (the mirror).** n -> -n - 2 maps the open-pair set onto itself; 0 mismatches in 12 wheels.
Its unique fixed point mod W is the shield n = -1. *Transfers.*

**L8 (the symmetry group).** The affine maps preserving the open-pair set are exactly
n -> c(n + 1) - 1 with c = +-1 (mod g) for every gear, a group (Z/2)^m; only c = +-1 (mod W)
preserve adjacency, so the adjacency-preserving group is Z/2. Proved in a line; brute-force
verified at W = 1,001 (8 of 8 maps). *Transfers.*

**L9 (mirror parity of the gap census).** Every gap length has an even count per wheel except
length 1. 12 of 12. *Transfers.*

**L10 (the alignment law).** The longest run of consecutive open pairs is exactly q' - 3,
whatever the other gears; the longest step-2 chain of open pairs is exactly q' - 2. Both
attained in 12 of 12. Counts: prod(g - 2 - L) starts of a run of L, prod(g - 1 - L) starts of a
chain of L; 0 mismatches over all L. *The law transfers; its value does not (the bottom's is
(2q'-2)/3 or (2q'-4)/3), and the chain statement has no bottom analogue.*

**L11 (the run spectrum).** The number of maximal runs of exactly L open pairs is the second
difference of prod(g - 2 - L): a polynomial of degree m - 2 in L, hence an arithmetic
progression of common difference exactly 6 for three-gear wheels whatever the gears; the top
length q' - 3 occurs exactly prod(g - q' + 1) times. 0 mismatches, 12 wheels. *New.*

**L12 (chain law).** Two openings x < y of M are both struck by a new gear g in some copy iff
y - x = 0, +2, -2 (mod g). 0 exceptions in 118,341 pairs. *Transfers, with d = 2 in place of
d = 2u_g.*

**L13 (merge law).** Every gap of M + g is a gap of M or a merge of consecutive gaps of M whose
interior openings g strikes. 0 exceptions in 34,646 gaps. *Transfers.*

**L14 (letters and alternation).** The letters are {2, g - 2}; in a struck run the nonzero
letter classes strictly alternate, with the tooth reading 2: (-2 -> 0) and g - 2: (0 -> -2).
0 exceptions in 816 runs. The fuel cap becomes k <= (x_k - x_0)/2 and is vacuous. *Transfers in
form; the letters' sizes, and with them the cap, change completely.*

**L15 (dominoes).** Adjacent open pairs (n, n+1) number prod(g - 4); open pairs sharing a member
(n, n+2) number prod(g - 3). 12 of 12 each. *The first transfers; the second is new.*

**L16 (the record is a cover).** F_top(G) = the largest L such that [0, L) can be covered by
choosing one phase per gear, each gear then contributing a singleton, a domino {x, x + 2}, or,
if g <= L + 1, a long piece {x, x + g - 2} (and more if g <= L). Exact; validated against the
full-period scan 15 times out of 15. *New.*

**L17 (the parity law).** If every gear exceeds 2m + 1 then F_top = 2m - (m mod 2). Proved;
170 cases, 0 exceptions. Increments therefore alternate +3, +1 in that regime. *New; nothing in
the bottom machine resembles it, because the bottom's smallest gear is 5 and never large.*

**L18 (universal record multiplicity).** For large-gear machines the number of record blocks per
wheel depends only on m: 18, 24, 480, 720 for m = 3, 4, 5, 6. 0 exceptions in 14 wheels. *New.*

**L19 (the conjugacy).** n -> 6^{-1}(n + 1) carries the top machine's open-pair set exactly onto
the same gears' opening set in the column coordinate; 0 mismatches in 12 wheels, 2.2 million
residues. Counting and symmetry laws are therefore common property; metric laws are not.
*New as a statement about the two coordinates.*

**L20 (no fold).** The open pairs are equidistributed mod 2, mod 3 and mod 6 to within 2 in
every class, in every wheel. The top machine has no parity or mod-3 structure. *New; it says the
bottom's six-fold belongs to the anchor, not to a gear machine.*

**L21 (the gear zone).** On a range, for n <= Z the pair n is open iff both members are
q-smooth; the machine's longest pair-free run on [1, N] lies in [1, Z] at every q and N tested
(18 of 18 range machines). *New.*

Two measured facts kept without mechanism:

**W1.** The counts of gap 3 and gap 5 are exactly equal in every wheel whose gears all exceed 7
(9 wheels) and unequal exactly when 7 is a gear (3 wheels). Residue characterisation found: a
gap of 3 at n needs a gear with n = -1 (its shield) and a gear with n = -4; a gap of 5 needs a
gear with n = -3 and a gear with n = -4. No shift or reflection of Z_W carries one set to the
other (all W shifts and all W reflections tested at three wheels). Mechanism open.

**W2.** The range record of a fixed-gear machine climbs 21, 24, 27 (gears 7..31) and 15, 16, 17
(gears 13..41) as N goes 10^5, 10^6, 10^7, against wheel records 32 and 18: the range approaches
the wheel record slowly from below, and the wheel record is a hard ceiling.

---

## 5. What is new

Which bottom-machine laws transfer. The organising fact is L19.

| bottom law | in the top machine |
|---|---|
| two teeth, q - 2 openings | **unchanged** (teeth 0, -2 instead of +-6^{-1}) |
| opening count prod(q - 2), domino count prod(q - 4) | **unchanged** |
| column 0 always open; the mirror; unique fixed point | **unchanged** (the shield n = -1) |
| affine symmetry group (Z/2)^m, adjacency group Z/2 | **unchanged** |
| gap counts even except one | **unchanged** |
| copies and phases; hit, chain, merge laws; legal words | **unchanged in form** |
| alignment law "longest run = long arc of the smallest gear" | **form unchanged, value changes**: q' - 3 instead of about 2q'/3 |
| arcs (2u - 1, q - 2u - 1) ~ (q/3, 2q/3) | **changes**: (1, g - 3); the short arc collapses to the shield |
| letters {2u, q - 2u} ~ {q/3, 2q/3}; fuel cap a real brake | **changes**: {2, g - 2}; the cap is vacuous |
| teeth never adjacent | **replaced by the partner law**: teeth always at distance exactly 2 |
| the antipode (P +- 1)/2 gives an adjacent open pair | **counting part survives** (n = 2, n = -4 always open); **adjacency does not** |
| gear 5 caps runs at 2; openings are points and dominoes | **no analogue**: runs to q' - 3, step-2 chains to q' - 2 |
| the six-fold: every twin is 6k +- 1 | **no analogue**: the open pairs are flat mod 6 (L20) |
| the record made at the top three gears, growing like log^2 | **reversed**: made at the smallest gears, growing linearly, F = 2m - (m mod 2) when the gears are large |
| - | **new: the forbidden gap 4 (L4)** |
| - | **new: the origin clump, 2(q'-3)+1 slots (L6)** |
| - | **new: the record as an exact cover; the parity law; universal multiplicity (L16-L18)** |
| - | **new: the gear zone, where the range record always lies (L21)** |

**The one-sentence difference.** The bottom machine's gears have their two teeth a third of a
turn apart, so a gear's strikes are far apart and its contribution to a long blocked stretch is
spread; the top machine's gears have their teeth **two apart**, so each gear contributes a
domino and a blocked stretch is a *tiling by dominoes* - which is why the top machine's record
is 2m - (m mod 2), decided by counting and parity, and not by the sizes of the gears at all.

Prior art met and stopped: the longest pair-free run of a set of primes is the two-class
Jacobsthal problem (one class: Jacobsthal 1961; Iwaniec 1978 for the bound). Nothing above
re-derives an asymptotic for it; the results here are exact structure at fixed gear sets
(L16-L18), which the asymptotic literature does not address. The counts prod(g - 2),
prod(g - 3), prod(g - 4) are the standard Hardy-Littlewood / Schemmel local factors for the
patterns (0,2), (0,2,4), (0,2,3,5); they are used here as bookkeeping and not claimed.

---

## 6. Verdict: what the top machine is, as a machine

**A domino machine.** Each gear strikes the line in dominoes of width 2 - a strike never comes
alone, and its partner is always exactly two away (L3). Everything characteristic follows from
that one fact: the arcs are (g - 3, 1) rather than roughly (g/3, 2g/3); the short arc is a
single slot, the shield, which is the always-open pair and the mirror's only fixed point; a gap
of 4 is impossible; a blocked stretch is a tiling of consecutive integers by dominoes and by the
long letters of gears small enough to reach across it; and so the record is a counting-and-parity
quantity, 2m - (m mod 2) whenever the gears are large, made by the smallest gears, growing by
about 2 per gear and never by the size of a gear.

**It is the same machine as the bottom one for every counting question and a different machine
for every metric question** (L19). Its own anchor - the wheel of its three or four smallest
gears - folds nothing: the open pairs are flat mod 2, 3 and 6. What its smallest gear does
instead is set two ceilings, q' - 3 for a run of open pairs and q' - 2 for a step-2 chain, and
supply the long letters that let the record exceed 2m.

**On a range it splits into two zones.** Below Z the gears strike their own homes and almost
everything is closed: the gear zone, where the record always lives, sitting immediately above
the origin clump of 2(q' - 3) + 1 forced slots. Above Z the machine is dense, its density above
the CRT product by 5-17%, and its longest pair-free run far below the wheel's record and
climbing toward it slowly.

No interpretation against the twin conjecture is offered here; the clutch comes later.

---

## 7. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| P1 | wheel count = prod(g - 2) | **held**, 12 of 12 |
| P2 | arcs (g - 3, 1); longest run = q' - 3 | **held**, 12 of 12, every gear |
| P3 | always-open n = -1; origin clump | **held**, 12 of 12 |
| P4 | mirror 0 mismatches, group (Z/2)^m, adjacency Z/2 | **held**; 0 mismatches, group verified at W = 1,001 |
| P5 | conjugacy to the column coordinate, 0 mismatches | **held**, 2.2 million residues |
| P6 | chain law 0 exceptions | **held**, 118,341 pairs |
| P7 | letters {2, g - 2}, alternation, vacuous fuel cap | **held**, 816 runs |
| P8 | merge law 0 exceptions | **held**, 34,646 gaps |
| P9 | dominoes prod(g - 4); run spectrum to q' - 3 | **held**; spectrum sharpened to the exact second-difference law L11 |
| P10 | F(M+g) < F(M) + g; record away from origin; **top** gears make it | budget **held with vast slack** (max increment 7 in 69 steps); record away from origin **held**; "top gears make it" **REFUTED** - the smallest gears do the work |
| P11 | range density above the product, falling; range record below wheel record | **held for the in-use machine** (ratio 1.05-1.17, 26% fall); **refuted for a fixed gear set**, which is flat to four figures and behaves as if periodic |
| P12 | the first stretch above 0 is the most open | **refuted for the in-use machine**: the gear zone above the origin clump is the *least* open stretch and carries the record. Held for a fixed gear set. |

---

## 8. Dead ends

- **Scanning the period for the ladder.** Abandoned at seven gears (period 10^9 and rising).
  Replaced by the covering formulation (L16), which is exact and cheap and was validated against
  the scan 15 times out of 15.
- **A bijection for W1.** The exact equality of the gap-3 and gap-5 counts is not induced by any
  shift or reflection of Z_W: all W shifts and all W reflections were tested at {11,13,17},
  {13,17,19}, {17,19,23} and none carries one set to the other. The residue characterisation
  (shield-plus-(-4) against (-3)-plus-(-4)) is as far as this branch got.
- **"The record is made at the top gears" (P10).** Refuted by the composition of the optimal
  covers: strikes per gear go as 2L/g. What survived is the useful half - the record is a tiling
  by letters with no flanks, which is what made L16 and L17 possible.
- **"The top machine is non-periodic on a range."** Refuted for a fixed gear set: density flat
  to four or five figures on a range 10^4 times below the wheel. The non-uniformity belongs to
  the *in-use* machine, whose gear set grows with the range, not to a gear machine as such.
- **The largest ladder rungs.** The covering search was cut off at one length per ladder
  (L = 40, 38, 39, 41, 44, 46); those rungs are lower bounds, not records, and are marked so.

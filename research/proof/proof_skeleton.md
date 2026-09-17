# Twin primes by construction: the one document (rewritten 2026-09-13)

Everything is built, not named. Each claim says why the construction makes it so, and carries
its status: PROVED (built here or in the kernel, proofs/, Lean 4, zero sorries, axioms propext /
Classical.choice / Quot.sound only), EXACT (checked with zero exceptions on a stated range and
provable by the same argument, not yet in the kernel), MEASURED (zero exceptions on a stated
range, no proof), OPEN (the one statement left). The earlier document (sections, cuts, the stack
by squares, the three faces) is kept as research/proof/proof_skeleton_2026-09-11.md; the tree
(research/proof/theory_tree.md) holds the history.

# Part I. The construction

## 1. The line, a gear, the fold. PROVED

The line is 1, 2, 3, ... A gear of size g strikes g, 2g, 3g, ... and nothing else. Gears 2 and 3
leave, in every strip of six, only 6j - 1 and 6j + 1: pair them as the column j. A gear g >= 5
strikes column j iff g divides 6j - 1 or 6j + 1, two residues of j per gear (the teeth, j = -+6^-1
mod g). A twin prime pair is a column with both members prime. [kernel TopMachine.strikesR_iff]

## 2. The square-root rule. PROVED

A number below P^2 that no prime below P divides is prime. So a column whose members lie below
P^2 and which no gear below P strikes is a twin prime pair. [kernel OneStepE.blocked_iff_sqrt,
CoreLeftover.twin_of_rough, SquareColumn.section_twin_of_unstruck]

## 3. The machine and its window. PROVED

Machine q (q prime): the gears 5 .. q. Its window: the numbers in (q, q^2]. Inside the window the
machine is complete: every composite there has a gear factor at most q (by 2), so no gear above q
acts below q^2, and a column of the window that the machine does not strike is a twin. The
machine is used at cycle 1 only; to go higher, add gears. [kernel as in 2]

## 4. The statement. PROVED reduction

If every window (q, q^2], q prime, holds a twin prime pair, then twin prime pairs are infinite: a
largest pair T would leave the window of any prime above T empty. The whole proof is therefore
the window statement: for every prime q the machine 5 .. q leaves a column of (q, q^2] unstruck.

## 5. What can strike inside the window. EXACT

- No gear above q (3). Nothing above q^2 enters.
- Squares: one per gear, g^2 for g <= q, each the right member of its column (g^2 = 1 mod 6).
- Composites of order j (j prime factors with multiplicity) exist in the window iff 5^j <= q^2,
  so the order is at most floor(2 log_5 q) (exact at 18 machines to 401); orders 4 and up are
  the smooth field (rows 5 and 7 nearly alone); orders 2 and 3 hold 90 to 99 percent of the
  kills, and the order-2 kills are mostly a gear times a prime of the window itself.
- The fields (research/proof/fields_construction.md, docs/fields_view.html): the strikes split
  by which gears take part; each field's rows and periodicity are exact (kernel Fields,
  FieldsB, FieldsC).

# Part II. The mirror walk

## 6. Mirror axes. PROVED

For a set S of gears containing 2 and 3, with product M, the pattern of the gears of S repeats
every M and is symmetric about 0, hence about every multiple of M: these multiples are the
mirror axes of S. A flip about the axis a sends the column (n, n+2) to (2a - n - 2, 2a - n).
The axis rule: a gear dividing 2a divides a member of the column iff it divides a member of its
image, so the flip carries the openness of every gear dividing the axis product. [kernel
MirrorWalk.flip_carries, openTo_flip_iff]

## 7. Composition. PROVED

A walk of flips about a_1, a_2, ... ends at 2A - n - 2 after an odd number of flips and at
n + 2A after an even number, A the alternating sum of the axes (last positive). The end of the
walk is open to every gear dividing 2A, whatever the intermediate landings were; tracking the
carried gears flip by flip undercounts (42 k_2 - 30 k_1 = 66 carries 11 though neither axis
does). [kernel MirrorWalk.walk_eq, openTo_walk_iff]

## 8. The landing family. PROVED

Home is the column (-1, 1), open to every gear. Every walk from home ends on (2A - 1, 2A + 1),
A a multiple of 6, and every gear dividing A finds it open. Writing 2A = 12m: the landing is
column 2m of the line, the m-line; a gear h strikes it iff m = -+12^-1 (mod h), two classes of
m per gear. [kernel MirrorWalk.walk_home_odd, walk_home_even, home_open, landing_open_of_dvd,
struckBy_mline]

## 9. The landing lemma. PROVED

If 12m + 1 < P^2, the gear set holds every prime of [5, P), and every gear not dividing 12m
misses 12m - 1 and 12m + 1, then (12m - 1, 12m + 1) is a twin prime pair. The gears dividing
12m need no check: the mirror carries them. [kernel MirrorWalk.landing_twin,
not_dvd_landing_of_dvd_axis]

## 10. Termination from the record. PROVED implication

Let R(q) be the m-line record of machine q: the longest run of consecutive m, all struck by
some gear up to q. If R(q) is below the window length (q^2 - q)/12, some m with 12m in the
window is unstruck, and by 9 the walk lands on a twin. [kernel MirrorWalk.walk_lands_of_record;
the same shape as SquareColumn.section_twin_of_record]

# Part III. The walk as an algorithm

## 11. The locator. MEASURED to 20000

Rule: axis A = k M with M = 6 (or 30); k the smallest value with 2kM in the window whose class
mod every remaining gear avoids the two teeth. One flip from home. Every machine from 11 to
20000 lands on a twin prime pair (2258 of 2258 for each M), the landing a fraction of a percent
into the window. Carrying more gears in M buys nothing: a mirror carries only the gears whose
primorial stays below q^2/2 (seven gears by q = 3000 against hundreds in the machine); the
choice of k does the work at every size. Multi-step walks reach the same landings by 7.
[research/proof/locator.md; research/stack/r8/true_mirror_walk.py, multi_mirror_walk.py]

## 12. The same walk at the square. EXACT lemma, MEASURED existence

Let g be the first prime above sqrt q (the gears below g are exactly the gears up to sqrt q).
Every column in (g^2, g g'), g' the next prime, missed by the gears below g and by g is a twin
(a composite there would need two factors of at least g); 8194 such columns to g = 1500, 0
exceptions. The strike law: with r = g mod h, gear h strikes the column i places after g^2 iff
h divides r^2 + 6i - 2 or r^2 + 6i (45,150 checks, 0 violations), so the gears just below g
(small r) cannot strike near g^2. Taking i as the smallest offset the law allows lands on a twin
at every machine to 20000, offset at most 27 columns. This is the window statement read at the
first square above q; it is not used in the chain 6 to 10 and is kept as the second form.

# Part IV. What is open

## 13. The one statement. OPEN

For every prime q, the m-line record R(q) of the machine 5 .. q is below (q^2 - q)/12.

Equivalent and sufficient forms: the record form of the earlier document (F(q) < q^2/6 on all
columns) implies it; the run form (no run of struck columns covers the window) is it. Measured
exactly on the m-line to q = 3001: R(q) = 4, 13, 43, 80, 191, 278 at q = 11, 31, 101, 401,
1009, 3001 against window lengths 9, 77, 841, 13366, 84756, 750250; the open m in the window
number 3, 14, 100, 906, 4179, 26960; the record's share of the window falls from 0.40 to
0.0004. [research/stack/r8/mline_records.py]

## 14. What stops the walk, by the fields. EXACT where stated

The walk lands on the first unpainted k at or after the zone start k_0 (the first k with
12k - 1 > q). It is stopped only by a painted run anchored at k_0 spanning the whole zone;
nothing else can stop it. On the landing zone the only fields that paint are the multiples
rows, one per gear (two teeth per period h; row h leaves h - 2 of every h consecutive k); the
squares, the higher fields and the product fields are relabellings of that paint, and no gear
is blind on the m-line. The rows are independent modulo the product of the gears (exactly
prod (h - 2) unpainted k per full period), the zone being one phase of that period. Let L(q) be
the painted run at k_0: the walk lands iff L(q) is below the zone length. L(q) is the distance
from q to the first twin above q with midpoint a multiple of 12, in m-line columns; machines
11 to 20000: mean 8.7, median 6, largest 55 (q = 13007, zone 14,097,420 long).
[research/proof/walk_fields.md]

So the one open statement is, in its sharpest form: for every prime q, the first twin above q
with midpoint a multiple of 12 lies below q^2. What decides it is the paint just above q, laid
by the small composites there (every number in (q, 2q) is prime or has a factor below q).

# Part IV b. The construction in its final form: one flip from home (2026-09-17)

Sixty-four rounds of walk construction (research/proof/loop_algorithms.md) ended at the simplest
form available.  Every walk of more than one step spends room in the window on the mirrors it
carries, and the trade is exact: candidates spaced 2M apart inside a window of length q^2 - q
number at most (q^2 - q) / (2M), so carried gears and candidates divide the same window
(`mirror_times_candidates`, round 56).  Measured directly (round 63): at the last step of the
settle walk the open candidates number 4.92 with the full proved prefix and 7.31 with no prefix
at all when the stride is chosen from eight; the prefix costs more than it gives, because the
gears above the cut still strike 22.5 of 75.8 candidates at the handover.  So the construction
is one flip.

## 15. The one-flip locator. STATED, one hypothesis

From home (-1, 1), open to every gear, flip about the mirror {2, 3, g} with period k in either
direction.  The landing is the column

    -1 + 12 g k d,   k = 1..K,   d = +1 or -1.

`OneFlipOpen G g K P` says one of those columns lies in the window (q, q^2] and is struck by no
gear of G.  With G holding every prime from 5 below P:

    OneFlipOpen  =>  a twin prime pair inside (q, q^2]        `oneflip_twin`
    OneFlipOpen for every machine  =>  the window statement   `window_statement_of_oneflip`

[proofs/OneFlipLocator.lean, round 60; 0 sorries, axioms propext / Classical.choice / Quot.sound]

The choices are fixed, not searched: the mirror is the smallest one that fits, g = 5 (stride
360), and K = (ln q)^3.  MEASURED (oneflip_margin.py, oneflip_classes.py, oneflip_small_mirror.py):
the family holds 25 to 58 open columns at every machine from 19 to 20011, the first at a period
between 6 and 289.  Two things decide it.  The candidate count must grow: with a fixed K = 40 the
margin thins (23 open at q = 1000, 15 at q = 20000), and the arrangement of the candidates does
not matter beyond their number.  The mirror must be small: over every admissible mirror at
q = 5000 the open columns run from 55 (g = 5) down to 0 (g = 2843), mean 8.3, and the mirror at
the first gear above sqrt(q) leaves none at q = 101 and about half as many as g = 5 at every
larger machine.  A larger mirror carries more gear phases but spaces the candidates further
apart, which is the trade again, read on the mirror instead of the walk.

The teeth law makes the family rigid.  Its members are 72 g k - 7 and 72 g k - 5, so a gear h
that inverts the stride at u strikes exactly at k = 7u and k = 5u modulo h: every gear's two
teeth are the fixed pair (7, 5), scaled by that gear's own unit, and nothing else enters.
PROVED: `strike_iff_scaled`, `oneflip_teeth`, `oneflip_members`
[proofs/OneFlipLocator.lean, round 61].

Part III's locator (11) is the same shape with the mirror {2, 3} or {2, 3, 5} and the period
chosen by the residues; the form stated here takes the larger mirror {2, 3, g} and lets the
period run over a fixed range, so the family is explicit and finite without a residue search.

## 16. The settle walk: what it proved, kept as a recorded result

The settle walk (visit the gears q down to 7, at gear g flip about {2, 3, g}, choose the period
that keeps every gear visited so far off its two teeth) is no longer the construction, but two
of its parts are proved outright and stay in the kernel as results about mirror walks in
general:

  (a) The walk stays in the window.  PROVED: an in-range move exists at every step whose mirror
      fits (`in_range_move`), a move into the window exists at the last step (`window_move`),
      the spiral base fits for every gear with 2g + 3 <= q (`base_fits`), and the induction is
      immediate (`stays_in_range`).  [proofs/MirrorWalkInWindow.lean, round 56]
  (b) The keeping move exists at every step of the first cut.  PROVED: a gear coprime to the
      stride strikes the candidates in two classes of the period (`strikes_iff_offA`), n gears
      strike at most 2n classes (`resA_card_le`), so with every visited gear above 2n some
      period at most 2n gives a candidate open to all of them (`keeping_move_free`).
      [proofs/MirrorWalkSettleFree.lean, round 57]  The cut is the steps where the visited gears
      all exceed twice their number: 70.5% of the steps at q = 1000, 76.6% at 10^4, 84.3% at
      10^6.  The lemma is maximal: no stride count, visiting order, carrying, inheritance or
      residue choice moves the cut (round 57).

Part (b) applies to the one-flip locator too, at n = 1: it is why a period exists keeping g
itself off its teeth.  What it cannot do is cover the small gears, and that is the whole of the
open statement.

## 17. The one hypothesis. OPEN, named

`OneFlipOpen` is the window statement restricted to one explicit arithmetic progression of
modulus 12 g.  Nothing in the sixty-four rounds removed it, and the trade above says why: room
in the window buys either carried gears or candidates, never both.  The general step form
`StepOpen G c s K P` (any column c, any stride s) is the same statement for a walk of any shape
and gives the same conclusion: `walk_twin_of_stepOpen`
[proofs/MirrorWalkConditional.lean, round 58].  `oneflip_twin` is its instance at c = -1,
s = 12 g d.

# Part V. Standing

- PROVED: 1, 2, 3, 4, 6, 7, 8, 9, 10, 16(a), 16(b), the implications of 15 and 17 (kernel names given).
- EXACT: 5, the lemma and the strike law of 12, the blind-gear laws (locator.md), the field facts of 14, the trade of Part IV b.
- MEASURED: 11 to 20000; 12 to 20000; 13 to 3001; L(q) of 14 to 20000; 15 to 20000; the settle walk of 16 to 2000.
- OPEN: 13, and equivalently OneFlipOpen of 15 (StepOpen of 17 in its general form). The
  construction is one flip from home; everything around that flip is proved, and the one thing
  left is that some column of an explicit finite family is open.

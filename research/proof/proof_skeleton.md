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

## 14. What a proof of 13 must produce

A reason, from the construction, that the teeth m = -+12^-1 (mod h) of the gears 5 .. q cannot
cover (q^2 - q)/12 consecutive m. What is known about the teeth: two per gear, mirror-symmetric
about every multiple of h; the machine's pattern on the m-line has period the product of its
gears, far longer than the window, so the window sees one phase of it; the record grows far
slower than the window (13). What every earlier route met (proof_skeleton_2026-09-11.md, Parts
III and IV): a count of open m gives the right order but no guarantee; a machine with the same
teeth counts but free phases can cover a window (the counter-machines), so the reason must use
the real teeth, which are set by the primes themselves (the hand-up: the gears are the survivors
of the lower windows). The walk adds the exact form of the target (10): a run of struck m below
q^2/12 shorter than the window.

# Part V. Standing

- PROVED: 1, 2, 3, 4, 6, 7, 8, 9, 10 (kernel names given).
- EXACT: 5, the lemma and the strike law of 12, the blind-gear laws (locator.md).
- MEASURED: 11 to 20000; 12 to 20000; 13 to 3001.
- OPEN: 13.

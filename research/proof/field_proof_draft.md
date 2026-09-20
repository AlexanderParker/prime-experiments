# The field proof: a draft in lemmas (owner's argument, 2026-09-20)

Owner's direction (2026-09-20): the machine always produces twin slots; no future state can prevent
them, because every field of residues has a fixed shape and phase that can never enter a permanent
covering state, and no set of gears creates a cycle that always blocks. Write the proof as if these
facts hold, as a draft hypothesis, then solidify each lemma one at a time.

This page is that draft. Every statement is marked **PROVED** (with the kernel file), **PROVED
(elementary, not yet in the kernel)**, or **LEMMA** (to be established). The chain from the lemmas to
the theorem is complete: if every LEMMA is proved, the twin prime conjecture follows. Nothing here
is a count, a density or a sieve bound; every statement is about which columns which gears strike.

## Plain words

Columns hold the pairs (6n-1, 6n+1). Gears are the primes from 5 up; a gear strikes a column when
it divides one of the two members. Below the square of a gear, the only columns it can strike are
ones a smaller gear already struck, or its own home column. So once a column has survived every
gear below the square root of its members, it is a twin pair and stays one forever: later gears
widen the machine but cannot reach back. The whole proof therefore rests on one thing: the gears up
to q, with the phases they actually have, never paint every column of the window between q and
q^2. The fields tell us exactly where each gear paints; the lemma to establish is that those
paintings never close the window.

## 0. Objects

- **Column** n >= 1 holds the pair (6n - 1, 6n + 1). **Gear** g: a prime g >= 5. **Machine** q: the
  gears 5..q (q prime). **Window** of q: the columns n with q < 6n - 1 and 6n + 1 <= q^2, i.e.
  n from (q + 7)/6 to (q^2 - 1)/6, about q^2/6 columns.
- **Strike**: gear g strikes column n when g | 6n - 1 or g | 6n + 1. **Twin slot** of machine q: a
  column of the window that no gear of the machine strikes.
- **Fields** (ids of the fields explorer): `multiples` (row g marks every multiple of g),
  `squares` (row g marks g^2), `products:j` (members with exactly j prime factors, marked in the
  row of each factor), `higher:g` (composite members whose smallest gear factor is g),
  `lower:g` (composite members whose largest gear factor is g). The painted set of the window is
  the union of the `multiples` rows; `squares`, `products:j`, `higher:g`, `lower:g` are
  relabellings of that paint by the shape of the member (research/proof/walk_fields.md).

## 1. The square-root rule - PROVED

**Lemma A.** A twin slot of machine q is a twin prime pair: both members are prime.

*Proof.* A member m of a window column satisfies q < m <= q^2 and gcd(m, 6) = 1. If m were
composite, its least prime factor f would satisfy f^2 <= m <= q^2, so f <= q, and f >= 5 because
gcd(m, 6) = 1; then f is a gear of the machine dividing m, so the column is struck. Kernel:
`prime_of_unstruck_member`, `window_twin_of_maxGap` (proofs/LadderMaxGap.lean, LadderCovering.lean).

## 2. Widening never reaches back - PROVED (elementary; kernel entry to add)

**Lemma B (the widening rule).** Let g be a gear and n a column with 6n + 1 < g^2. If g strikes
column n, then either 6n - 1 = g or 6n + 1 = g (the gear's home column), or a gear h < g already
strikes column n.

*Proof.* g divides a member m < g^2, so m = g k with 1 <= k < g. If k = 1, m = g: the home column.
If k > 1, k is coprime to 6 (as m is) and k < g, so k has a prime factor h with 5 <= h <= k < g,
and h divides m: gear h strikes the column.

*Consequence.* A twin slot of machine q is a twin slot of every larger machine: no gear added later
can strike it (its members are below the new gear's square and are not the new gear itself, being
primes below it). "No future state can prevent them" is this lemma, and it is exact.

## 3. The shapes of the fields - PROVED (elementary)

**Lemma C1 (multiples).** In every run of g consecutive columns, row g marks exactly two columns
(the classes n = +-6^{-1} mod g, at distance 3^{-1} mod g from each other), and it never marks two
adjacent columns: adjacency would need 3^{-1} = +-1 mod g, i.e. g | 2 or g | 4. (In the 12k
coordinates of research/proof/walk_fields.md gear 5 has adjacent teeth 2, 3 mod 5; in the 6n
coordinates used here its teeth are 1, 4 mod 5, at distance 2.) Kernel: `two_strike_classes`
(proofs/TwinLadder.lean), `struck_classes` (LadderCovering.lean).

**Lemma C2 (squares).** Row g marks the column of g^2 exactly once in the window, at
n = (g^2 - 1)/6 (the right member; 6n - 1 is never a square), and only when g^2 <= q^2, i.e.
always for g <= q. So `squares` contributes exactly one column per gear, at a known position.

**Lemma C3 (products).** A composite member with exactly two prime factors g <= r (both gears) is
g r, and g^2 <= g r <= r^2: the product of an odd pair strikes between the squares of the pair.
Its column is (g r -+ 1)/6 according as g r = +-1 mod 6. A member with j >= 3 prime factors is a
multiple of its smallest factor g with cofactor >= 25, so it lies in row g at a multiple g k,
k >= 25 coprime to 6; it is already marked by the `multiples` row of g and adds no new paint.

**Lemma C4 (higher and lower).** `higher:g` is row g restricted to members whose other factors
are all >= g; `lower:g` is row g restricted to members whose other factors are all <= g. Their
unions over g are each the whole painted set. They add no paint; they say which gear is
responsible for each painted column (the smallest, or the largest).

**Lemma C5 (no blind gears).** Every gear g >= 5 acts on the window: 6 is invertible mod g, so
both classes of Lemma C1 exist. (Kernel: `invSix_spec`, LadderCovering.lean.)

## 4. No permanent covering - PROVED, and where its reach ends

**Lemma D1 (no blocking cycle).** For any finite set G of gears, the joint strike pattern has
period P = product of G, P is coprime to 6, and each period contains exactly prod_{g in G} (g - 2)
unstruck columns, at least one. In particular no set of gears blocks every column, and no joint
period is congruent to 0 mod 2 or mod 3: the only gears whose period aligns with the column
lattice are 2 and 3, which are not gears. *Proof.* Chinese remainder theorem: each gear leaves
g - 2 classes free, and the classes combine independently. (Kernel: `window_realises_shift`,
proofs/RigidShift.lean, for the phase statement.)

**Where D1 stops.** D1 says a covering state is never permanent: within every full period there
are unstruck columns. The window of q has about q^2/6 columns, while the period of the machine q
is the product of its gears, larger than e^{q/2}. The unstruck columns that D1 guarantees may all
lie outside the window. So D1 alone does not give a twin slot in the window. This is the exact
point where the draft needs its one open lemma.

## 5. The window lemma - LEMMA (the one to establish)

**Lemma E (the window is never painted over).** For every prime q >= 5 the gears 5..q, at the
phases they actually have, leave at least one column of the window of q unstruck.

Kernel forms already available: `MaxGapHyp` (proofs/LadderMaxGap.lean) - "the longest run of
struck columns anywhere in the pattern of the gears 5..q is shorter than the window" - implies
Lemma E; `CoveringHyp` (LadderCovering.lean) - "no choice of two classes per gear covers the
window" - implies it too. Both are stronger than E; E itself is exactly "machine q has a twin
slot". Measured: E holds for every prime q up to 10^8 by the twin-gap tables (largest maximal
twin gap 35,640 near 7 x 10^16 against a window of q^2/6 columns), and the stronger MaxGapHyp
holds with the machine's exact record F(q) = 34, 88, 91, 103, 118, 145, 160, 179, 213 at
q = 23..67 against windows of 84..737 columns.

**Sub-lemmas proposed for E, one field at a time** (the owner's programme):

- **E1 (one row).** Row g alone never paints two adjacent columns, so its longest run is 1.
  PROVED (Lemma C1).
- **E2 (two rows).** For gears g < h the longest run both rows together can paint is exactly 4 if
  {g, h} = {5, 7}, 3 if exactly one of g, h is 5 or 7, and 2 otherwise. *Proof.* A gear's two teeth
  sit at distance 3^{-1} mod g (or g minus it); this distance is 2 exactly when 6 = +-1 mod g, i.e.
  g in {5, 7}. No gear paints two adjacent columns alone (E1). A run of 4 needs the two gears'
  tooth pairs interleaved as {n, n+2} and {n+1, n+3}, so both distances are 2; a run of 3 needs
  one tooth pair {n, n+2} with the other gear at n+1, so one distance is 2; a run of 2 is always
  available by the Chinese remainder theorem; a run of 5 or more would need a gear to paint 3
  columns within 5, impossible below its period, and above it the two gears paint fewer than
  2/5 + 2/7 of any long run. PROVED (exact by exhaustion for all 300 pairs of gears up to 101,
  research/stack/r8/two_rows_exact.py; kernel entry to add).
- **E3 (how a new gear extends the record).** Let F(q) be the longest run the machine q can paint
  anywhere in its period. A record run of machine q' consists of runs of machine q joined at
  columns painted only by q' (holes of the old machine filled by the new gear's teeth). If h such
  columns lie in the run, the run minus them is at most h + 1 old runs, each at most F(q), so
  F(q') <= (h + 1) F(q) + h, with h <= 2 ceil(F(q')/q') and the h teeth at mutual distances 0 or
  +-3^{-1} modulo q'. PROVED (elementary). Measured (exact scans, 5..23): h = 2, 1, 1, 2, 1-2, 3
  for q' = 7, 11, 13, 17, 19, 23; the old runs used are 1, 4, 5, 10, 17, 14 against old records
  1, 4, 6, 10, 17, 24 - from 17 on the record is assembled from several medium runs, not from the
  old record extended. The inequality alone allows F to triple per gear; E4 needs the spacing
  structure of the old machine's holes.
- **E4 (the record is below the window).** F(q) < (q^2 - q)/6 for every prime q. LEMMA. This is
  `MaxGapHyp`. By E3 it is a statement about the holes of machine q (its twin candidates): the
  gear q' can join old runs only at holes spaced 0 or +-3^{-1} mod q' apart; E4 says no chain of
  such joins reaches the window's length. Measured: F = 1, 4, 6, 10, 17, 24, 33, 42, 57 (run
  convention, q = 5..31), then 88, 91, 103, 118, 145, 160, 179, 213 (gap convention, q = 37..67),
  about 0.17 q ln^2 q, against a window of q^2/6; the ratio falls as 1/q.

If E4 is proved, Lemma E follows (the window is a run of the pattern, so it is not fully painted),
Lemma A makes the unstruck column a twin pair, Lemma B keeps it a twin pair in every larger
machine, and the theorem follows.

## 6. The theorem

**Theorem (conditional on Lemma E).** For every prime q >= 5 the window of q holds a twin prime
pair; hence there are twin primes above every bound.

*Proof.* By E the window has an unstruck column; by A its members are prime; by B nothing later
changes that; the windows of the primes q are unbounded (Euclid), so the twins are unbounded.
Kernel: `windowStatement_of_maxGapHyp`, `twins_unbounded_of_maxGapHyp` (LadderMaxGap.lean) give
this from the stronger form E4; the E-form needs only the two lines of Lemma A.

## 7. Standing of the parts

| part | statement | standing |
|---|---|---|
| A | an unstruck window column is a twin pair | PROVED (kernel) |
| B | a later gear never strikes a twin slot | PROVED (elementary); kernel entry to add |
| C1-C5 | the shapes of the five fields | PROVED (C1, C5 kernel; C2-C4 elementary) |
| D1 | no set of gears blocks permanently | PROVED (CRT) |
| E1 | one row paints no two adjacent columns (g >= 7) | PROVED |
| E2 | two rows: exact longest joint run 4 / 3 / 2 | PROVED (2026-09-20) |
| E3 | a record is old runs joined at the new gear's teeth, F(q') <= (h+1)F(q) + h | PROVED (2026-09-20) |
| E3' | the sandwich G_1(q) <= F(q') <= G_{2 ceil(F/q')}(q); exact recursion by alignable chains | PROVED (2026-09-20) |
| E4c | in a covered window of q' the interior hole gaps of q are >= (q'-1)/3 and alternate in two residues | PROVED (2026-09-20) |
| E4d | F(q') <= S_t*(q), the longest stretch of q with hole gaps >= (q'-1)/3 | PROVED (2026-09-20) |
| E4 | S_t*(q) < (q'^2 - q')/6 for consecutive primes q < q' (implies `MaxGapHyp`) | LEMMA |
| E | the window is never painted over | LEMMA (follows from E4) |

E2 is proved. Three gears (exact, first eight gears): 6 with both 5 and 7; 5 with one of them
and two gears of tooth distance 4 (11, 13); 4 with one of them and gears of distance 6 or more;
3 with neither - the record of a small set is a function of the multiset of tooth distances.
Initial segments 5..p (run convention): 1, 4, 6, 10, 17, 24, 33 for p = 5, 7, 11, 13, 17, 19, 23.
E3 is proved in its exact form, and the record obeys an exact recursion (verified for every step
7..23 over the joint period): F(q') is the larger of F(q) and, over every maximal chain of
consecutive holes of machine q that all lie in the two tooth classes of q', the span from the hole
before the chain to the hole after it, less one. Two consecutive holes are both in the classes only
when their gap is exactly 3^{-1} mod q' or q' minus it (or differs from these by a multiple of q');
so the record is decided by the hole-gap spectrum of machine q at two specific gap values and by
the runs flanking those gaps. Measured: aligned consecutive pairs are 0.1 to 0.3 percent of the
holes at q' = 13..23, three to five times rarer than independence would give; chains of at most 3
holes; the hole-gap spectrum is the project's wheel gap census (docs/novel, W11, W22-W26, W45).

**The sandwich (PROVED).** Let G_k(q) be the longest window of machine q holding at most k
holes (G_0 = F(q); G_1 the largest run-hole-run). A single hole can always be aligned with the
next gear q', and q' fills at most 2 ceil(L/q') columns of any run of length L; hence

    G_1(q) <= F(q') <= G_k(q),  k = 2 ceil(F(q')/q').

Exact at every step 7..23: G_1 = 3, 6, 10, 15, 24, 30 against F(q') = 4, 6, 10, 17, 24, 33, with
equality at 11, 13, 19 - the next record is often just the largest run-hole-run of the current
machine. The exact recursion: F(q') is the largest span-plus-flanks of a chain of consecutive
holes whose partial gap sums all lie in {0, +3^{-1}} or all in {0, -3^{-1}} modulo q'; verified at
all six steps. Aligned pairs are counted exactly by the hole-gap census at the two values 3^{-1}
and q' - 3^{-1} divided by q'.

**E4 in its sharpest form.** E4 for q' follows from G_k(q) < q'^2/6 with k about q'/3: every window
of machine q of length q'^2/6 holds more than q'/3 holes. Iterating the worst case per gear (each
gear removes at most 2 ceil(L/g) holes) is the union bound and loses, because new paint lands
mostly on painted columns. So the lemma to establish is the OVERLAP statement: in every window of
length q'^2/6, the teeth of the new gear q' cover fewer than all the holes left by 5..q. That is
the window statement for q' itself, now with the machine's exact growth law beside it: the record
never jumps past the old machine's k-hole windows, and the alignment of a chain is decided by the
hole-gap census at two residues.

**E4c (PROVED).** Since 3 x 3^{-1} = 1 mod q', the tooth distance 3^{-1} mod q' is (q'+1)/3 or
(2q'+1)/3, so the two gaps at which consecutive holes can both be filled by q' are {(q'+1)/3,
(2q'-1)/3} or {(q'-1)/3, (2q'+1)/3}, up to multiples of q', and along a chain they alternate
(every second hole is in the same class). Hence in a window fully painted by machine q', every
interior gap between consecutive holes of machine q is at least (q'-1)/3, every interior run of
machine q is at least (q'-4)/3 long, and the holes inside number at most 3L/(q'-1) + 1. A covered
window is a stretch of machine q with hole density at most 3/q', against the machine's 2.5/ln^2 q.

**E4, final form for this draft.** Machine q has no stretch of q'^2/6 columns in which every
interior hole gap is at least (q'-1)/3 and the gaps alternate between the two residues 3^{-1} and
-3^{-1} modulo q'. Equivalently: every window of machine q of length q'^2/6 has a pair of
consecutive holes closer than (q'-1)/3, or two consecutive gaps that do not alternate. This is
the lemma the field programme must now establish; it asks for about q'/2 holes in a window that
generically holds q'^2/(2.4 ln^2 q'). The counting form of it is a sifted-set lower bound; the
field form asks why the small gaps 1, 2, 3, 5 of machine q - the most common gaps it has - can
never all be absent from a stretch of q'^2/6 columns.

**E4d (PROVED) and the number.** Let S_t(q) be the longest stretch of machine q whose interior
consecutive hole gaps are all at least t. By E4c, F(q') <= S_t*(q) with t* = ceil((q'-1)/3).
Measured (exact, q -> q'): S_t*(q) = 5, 7, 19, 22, 33, 37 against windows 7, 18, 26, 45, 57, 84
(ratios 0.71, 0.39, 0.73, 0.49, 0.58, 0.44) and against F(q') = 4, 6, 10, 17, 24, 33. The S_t
table falls fast to t about 6 and then sits on a plateau near F(q) plus one or two isolated
flanks. So the lemma to establish, with its number, is

    E4:  S_ceil((q'-1)/3)(q) < (q'^2 - q')/6   for every pair of consecutive primes q < q' >= 7,

the longest stretch of machine q whose holes are all at least q'/3 apart is shorter than the
window of q'. Its holes are the isolated twin candidates of level q; E4 says isolated candidates
never line up sparsely enough, for long enough, to span the window. The data instrument is the fields explorer and `rigid_record_bisect.py`;
the exact rule is W88's loaded record rule. What must not enter: any count of columns, density or
sieve bound - E4 is to be proved from the shapes and phases alone.

## 8. What is already known about E4 from the record (for honesty, one paragraph)

The record F(q) is the two-class Jacobsthal-type function of the twin sieve. Its free-class
relative (any two classes per gear, OEIS A072753) is bounded above only by sieve methods, which
give exponent 4.27 in q, and the one-class Jacobsthal function only by Iwaniec's exponent 2 with an
uncomputed constant (docs/novel/j2-upper-bound.md, rounds 22-27). So E4 asks for an exponent-2
bound with the constant 1/6 for the actual pattern - a bound no sieve gives. The draft's wager is
that the shapes and phases of the actual pattern (not free classes) carry it; E2 and E3 are the
places to test that wager, exactly, one gear at a time.

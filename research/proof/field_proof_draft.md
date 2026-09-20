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

## 2. Widening never reaches back - PROVED (kernel: proofs/LadderWidening.lean, `widening`, `twin_slot_persists`)

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

## 5a. The numerator from the shapes (owner's direction, 2026-09-20 16:30)

The stretch is built from the rows, whose shapes we know; this section derives what can be derived
from the shapes alone, verifying only endpoints by table.

**Self-similarity of the fields (exact).** In the window of q every painted member is a composite
coprime to 6, and `higher:g` (composites whose smallest gear factor is g) is exactly g times the
set of numbers in [g, q^2/g] coprime to 6 and to every gear below g - g times the rough set of the
machine below g. So the painted set of the window is the union over gears of scaled copies of the
smaller machines' own hole patterns: the fields are self-similar, and the count of holes of
machine q is the Buchstab identity read column by column. This is the shape calculation of the
painted set; it is exact and it is the sieve's identity, no more and no less.

**The thin-band bound (PROVED from the shapes).** Split the gears into a base 5..B, whose pattern
is held exactly, and a top band (B, q]. Let h_B(L) be the least number of base holes in any window
of length L (from one base period). A covered stretch of length L must have every base hole inside
it painted by a top gear, and a top gear g paints at most 2 ceil(L/g) columns of the stretch. Hence

    F(q) <= max{ L : 2 sum_{B < g <= q} ceil(L/g) >= h_B(L) }.

Evaluated (research/stack/r8/thin_band_bound.py, table used only for h_B): base 17, top {19}:
F(19) <= 42 (true 24), and 42 < 57 = window of 19, so the window statement for machine 19 follows
from shapes alone; base 23, top {29}: F(29) <= 89 (true 42), 89 < 135, likewise for 29. With a band
two gears thick the bound is 143 for q = 23 (true 33, window 84) and above 160 for q = 31 (true 57):
useless. The bound is exactly the upper half of the sandwich (E3') with the base's sparsity
function written as h_B, and it is the union bound on the band.

**Why it loses, in shape terms.** The bound charges every tooth of a top gear as a hit on a base
hole. In a stretch of length L the gear has 2 ceil(L/g) teeth, but the base holes are a fraction
h_B(L)/L of the stretch, and only those teeth that fall on holes do any work: the true number of
hits is the number of base holes in the two classes of g, about 2 h_B(L)/g when the holes are
spread over the classes. The bound is tight only if the holes of the base inside the stretch all sit
in the two classes of the new gear - which is the alternation law of E4c, i.e. a covered window.
So every bound the row shapes give on their own charges the overlap at its worst case, and the
worst case is the statement to be excluded. The alternation attempt reduces to the same point:
the holes of machine q inside a covered window lie in two classes mod q', every second one in the
same class; excluding a stretch of q^2/6 columns with that property from the shapes of the rows
5..q is E4, and the rows' individual shapes constrain gaps between holes (each residue class of
each row leaves g - 2 free classes, whose differences cover every residue) but not their joint
residues modulo a gear that is not in the machine.

**What the shapes have given, and what they have not.** Given: the two-row and three-row records by
position analysis (E2, and F({5,7,g}) = 6 for every g); the alternation and spacing laws of a
covered window (E4c); the sandwich and the recursion (E3'); the thin-band bound and with it the
window statement for machines 19 and 29 without any table of the machine itself. Not given: any
bound on the record that does not charge the new gears' overlap at its worst case. The lemma E4 is
that overlap statement and nothing else; the direct counting form of it (the number of covering
phase vectors is at most P (1 - d)^L) was tried in docs/covering-bound-route.md and refuted.

## 5b. The residue-collapse census (shape calculation, verified 2026-09-20 17:10)

Take a window W of machine q of length L, and a new gear q' with teeth t_1, t_2 (t_2 - t_1 =
3^{-1} mod q'). Over the joint period every translate of W by a multiple of the period of q
appears with every residue mod q', so by CRT the number of translates in which q' paints every
hole of W is exactly: q' if W has no hole; 2 if all holes of W are congruent mod q'; 1 if the holes
occupy exactly two residues at difference +-3^{-1} mod q'; 0 otherwise. Summing over the windows of
one period of q gives the number C_{q'}(L) of covered windows of length L of machine q', and

    F(q') = max{ L : C_{q'}(L) > 0 }.

Verified exactly (research/stack/r8/residue_collapse.py) at q = 11 -> 13 and 13 -> 17 for
L = 6..20 (e.g. 20 covered windows of length 17 for machine 17, from the census: 20). For L above
F(q) the first term is empty, and the second needs every hole gap to be a multiple of q' - with
gaps at most F(q) + 1 that means a single hole - so

    C_{q'}(L) = 2 W_1(q, L) + N_alt(q, q', L)      (L > F(q)),

W_1 the number of windows of q with exactly one hole, N_alt the number whose holes form two
interleaved sub-progressions of difference q' at offset 3^{-1} mod q' (gaps alternating between
3^{-1} and q' - 3^{-1} modulo q'). Hence F(q') = max(G_1(q), largest alternating window), which is
the sandwich again, now as an equality, and E4 for q' reads: G_1(q) < (q'^2 - q')/6 and machine q
has no alternating window of that length.

What this settles about shapes. The census is the complete shape calculation of the record of
the next gear: nothing about q' enters except its teeth, and everything else is a census of
machine q's windows by the residues of their holes. Each row of machine q constrains those
residues not at all (a row leaves g - 2 free classes whose differences cover every residue), so
the alternating windows are governed by the joint pattern only. The attempt to bound an
alternating window through the smallest gear fails because the holes in a class of q' need not be
consecutive terms of the progression (a gap that is a multiple of q' is allowed), so gear 5's
free arcs along the progression bound nothing.

## 5d. The twisted translates (shape calculation, 2026-09-20 18:40)

Write a column as n = c + k q' (c its class mod q'). Gear g strikes it iff k lies in two classes mod g
at distance (3q')^{-1} mod g; so along each class c the machine q appears as one two-class pattern T
(the twisted machine, same gears, tooth distances (3q')^{-1} mod g) translated by c times q'^{-1}
mod the period. A covered window of q' of length L is therefore: T fully painted on about L/q'
consecutive terms at q' - 2 of q' translates, the translates forming an arithmetic progression of
shifts with difference q'^{-1} (exact, by CRT).

**Proved consequence.** Each non-tooth class alone gives F(q') <= q' (F_T(q; q') + 1), F_T the record
of the twisted machine. Measured (research/stack/r8/twisted_record.py): F_T = 3, 7, 10, 16, 28 against
F(q) = 4, 6, 10, 17, 24 for q = 7..19 - the twisted record is the ordinary record's size - so the
bound is 44, 104, 187, 323, 667 against windows 18, 26, 45, 57, 84: q' times too weak. The shifts are
equidistributed modulo every gear (as c runs over q' consecutive values, c q'^{-1} mod g covers each
residue floor(q'/g) or ceil(q'/g) times), which fixes how often each translate meets each tooth of
each gear but not whether the runs coincide.

**What the form shows.** The single-class constraint is as weak as one row's shape: it charges one
class and ignores that the q' - 2 runs are runs of one pattern at coupled positions. The coupling
is the whole content of E4, exactly as the overlap statement in 5a and the alternating window in 5b:
three coordinate systems, one fact. The row shapes fix every individual constraint exactly; they do
not fix the simultaneous one.

## 5e. The structure at the origin: squares are the only new kills (owner's "find the structure", 2026-09-20 19:20)

The record F(q) is a maximum over every phase of the pattern; the window of q sits at one phase,
the origin, where the fields have their factorisation meaning. There the shapes give an exact law
the generic phase does not have.

**Lemma E5 (origin square lemma, PROVED, elementary).** Let q < q' be consecutive primes. A member
m <= q'^2 of a column of the window of q' with no prime factor <= q is either prime or equal to
q'^2 (a product of two primes above q is at least q'^2, with equality only for q' x q'). Hence the
columns of the window of q' left unstruck by machine q are exactly the twin columns of that window,
plus the single column (q'^2 - 1)/6 when q'^2 - 2 is prime. The new gear q' therefore fills at most
ONE hole of machine q inside the window of q', and that hole was never a twin slot. Verified for
all 75 consecutive prime pairs to q' = 400 (research/stack/r8/origin_square_lemma.py): 0
violations, 26 square columns present, no other extra hole.

**The window count law (exact).** Write T(q) for the number of twin slots of machine q (twin pairs
with lower member in (q, q^2]). Then

    T(q') = T(q) - [q' + 2 prime] + N(q^2, q'^2],

N the number of twin pairs with lower member in (q^2, q'^2]: the only slot a machine can lose is
the pair (q', q'+2) at the bottom boundary, and every slot gained is a new twin at the top. The
squares field (E5) contributes no loss; the products field (composites g p with q < g <= p) paints
the new region (q^2, q'^2] and decides N. So at the origin the overlap of the new gear with the
old holes is total but for the square column, and the covering question of sections 5a-5d does not
arise: the window statement fails at q' only if T(q) <= 1 and N(q^2, q'^2] = 0, i.e. only after a
descent of the walk T by one per prime step with no new twin at the top over the whole descent -
a twin-free interval (q_0^2, q'^2] at least 4 q_0 (T(q_0) - 1) long past the last healthy window.

**What the structure says about the two targets.** MaxGapHyp (E4) asks that the record beat the
window at every phase; the window statement asks it at the origin only, where the shapes reduce
the new gear's action to one square column and put all the content into N(q^2, q'^2] - twins
between consecutive prime squares, RegionHyp of the ladder proof map. The field programme and the
ladder programme meet here exactly: the window count law is the ladder's region statement written
as a conservation law for slots, and the products field is the mechanism that fills the new
region. The lemma left is N(q^2, q'^2] >= 1 for every consecutive pair, or the weaker
"T never descends to 0", which is the same lemma in walk form.

## 5f. The shortest region: the twin stretch in the products field (2026-09-20 19:50)

The region between consecutive prime squares with gap 2 is the stretch of the twin centre s = q + 1:
columns 6c^2 + j, |j| <= 2c - 1, members s^2 + 6j -+ 1, 4c - 1 columns at height s^2. Its
shapes are on the record (normal form; the twin gears strike only the centre; every gear strikes
two classes of offsets; the top-band k-rule; the composite-forcing difference-of-squares families;
universal clearance). In the products field the region lemma reads as follows.

**Base and plugs (exact).** Fix a base level B (here B = floor(sqrt(2s))). A column is base-open
when both members are B-rough. A base-open column is a twin unless a member is a product of primes
above B - a P_2 = h p with B < h <= p, or a P_3 (no P_4 fits below (s+1)^2 with all factors above
sqrt(2s)). So

    base-open = twins + plugged,   plugged = columns with a P_2 or P_3 member and a B-rough partner.

The region lemma (a rung exists) is: the plugs do not exhaust the base-open columns. Measured
(research/stack/r8/stretch_products_split.py, twin centres 102..2970, exact factorisation): plugged
share 0.63, 0.72, 0.71, 0.73, 0.76, 0.70, 0.76, 0.71, 0.76, 0.75, 0.77, 0.81, 0.77 - rising slowly;
plug members almost all P_2 (P_3 a few percent), never P_4; the top band h in (s/2, s] paints 233 of
1,247 columns at s = 1872, of which 38 are base-open - a share e^{-gamma}/ln B of its paint lands
on base-open columns, as the partner's roughness predicts.

**What the shapes give here.** A plug is a pair (h p, h p -+ 2) with h p a product of two primes above
B and the partner B-rough: the products field lands h p at one column between h^2 and p^2 (the
owner's law), and the partner condition is a roughness condition on a shifted prime product. The
number of plugs is therefore bounded ABOVE by a sieve on the partner (an upper-bound sieve, which
exists at every level), and the number of base-open columns is bounded BELOW only by a lower-bound
sieve, which exists in a stretch of length 4s only for levels below (4s)^{1/4.27}, far below
sqrt(2s). This is exactly Chen's switching frame, and it is why Chen's theorem holds on the whole
line and not in a stretch: the plugs can be counted, the base cannot be bounded from below at the
level where the plugs are few. The measured share 0.7-0.8 says the plugs take about three quarters
of the base-open columns and leave a quarter, uniformly in s.

**Region lemma in field terms.** For every twin centre s and B = floor(sqrt(2s)): the products h p
(B < h <= p) whose partner h p -+ 2 is B-rough number fewer than the base-open columns of the
stretch. The lemma is a comparison of two counts that the shapes define exactly; the count of plugs
is accessible, the count of base-open columns in a stretch is the sifted-set lower bound that the
whole record turns on.

## 5g. The products field of the stretch is Goldbach (2026-09-20 20:30)

**Identity (exact).** A member of the stretch that is a product of two primes h <= p equals
m^2 - d^2 with m = (h + p)/2 and d = (p - h)/2: it is a Goldbach partition of the even number 2m,
and it lies in the stretch iff (s-1)^2 < m^2 - d^2 < (s+1)^2, at offset j = (m^2 - d^2 - s^2 -+ 1)/6.
By AM-GM, s - 1 < m <= (h + (s+1)^2/h)/2, so the partitioned numbers 2m run from 2s up to about
s^2/B for the smallest gear h = B in play. Verified on all 54,027 two-prime members of the stretches
of the 74 twin centres in [102, 3000] (research/stack/r8/stretch_goldbach.py), 0 violations.

**The top band is the short Goldbach problem of the twin centre.** The two-prime members with
m = s are exactly the partitions 2s = (s - d) + (s + d) with 1 <= d < sqrt(2s), lying at the lower
member of offset j = (1 - d^2)/6 - the record's difference-of-squares family, now read as the
Goldbach partitions of 2s whose parts lie within sqrt(2s) of s; d = 1 is the twin pair itself,
plugging the centre column with s^2 - 1 (the twin gears strike only the centre). The members with
m = s + 1 are the partitions of 2s + 2 with d = 6t, at the upper member of offset j = c - 6t^2;
2s - 2 contributes nothing (its products fall below the stretch). Verified exactly at every one of
the 74 twin centres: the set of such members equals the set of such partitions, 0 violations; per
centre 2 to 4 short partitions of 2s and 0 to 2 of 2s + 2.

**What the structure says.** The stretch is the Goldbach-partition graph of the even numbers 2m
between 2s and s^2/B: each column is plugged by the partitions (h, p) whose product falls on it
(and by three-prime products); the twins are the columns no partition reaches while both members
stay rough. The twin problem in the stretch and the Goldbach problem for the numbers just above 2s
are the two faces of one machine: the composite-forcing families of the record are Goldbach
partitions of 2s and 2s + 2, and the base-open count against the partition count is the region
lemma. The partition count of 2s carries the factorisation of s through its singular series
(prod (p-1)/(p-2) over p | s), while the twin count of the stretch carries none (the inherited
local factor is identically 1, section 5 of the tree): the plugs know the arithmetic of s, the
survivors do not.

**Relationship to establish (the lemma, third field form).** For every twin centre s and
B = floor(sqrt(2s)): the number of Goldbach partitions (h, p), B < h <= p, of the even numbers
2m in (2s - 2, s^2/B] whose product lies in the stretch with a B-rough partner, plus the
three-prime plugs, is smaller than the number of base-open columns.

## 5h. The level analysis of the comparison (2026-09-20 21:00)

The lemma of 5f-5g compares two counts at a base level B. Both sides are known exactly in law:
twins per column 6 x 1.3203/(4 ln^2 s) = 1.98/ln^2 s; base-open columns per column
prod_{g<=B}(1 - 2/g) ~ 2.5/ln^2 B. Hence the plug share is

    share(B) = 1 - T/U(B) ~ 1 - 0.79 ln^2 B / ln^2 s,

verified (research/stack/r8/share_by_level.py, three twin centres near 1300-3000): B = s^0.234:
measured 0.94-0.95, law 0.96; s^0.5: 0.78-0.80, law 0.80; s^0.8: 0.46-0.56, law 0.49-0.51 (the
exact product in place of 2.5/ln^2 B).

**What a proof through this comparison would need, by level.** To conclude plugs < U(B) one needs
(i) a lower bound for U(B) and (ii) an upper bound for the plugs with constant below 1/share(B).
- (i) U(B) is a set sifted by two classes per gear up to B in an interval of length 4s; a
  lower-bound sieve exists only for B <= (4s)^{1/4.266} ~ s^0.234 (the dimension-2 sifting limit),
  and there its constant is small. At B = sqrt(2s) there is no lower bound for U by any sieve,
  whatever equidistribution is assumed: the interval has perfect remainders already and the limit
  is intrinsic (Selberg's examples).
- (ii) The plug bound is a one-class sieve on the partners h p -+ 2 over the products; at level of
  distribution D the constant is F(ln D / ln B): with D = x^{1/2} (Bombieri-Vinogradov range,
  x = s^2) and B = s^0.5 that is F(2) = e^gamma = 1.78; the share 0.80 needs a constant below
  1.25. With D = x^{1-eps} (an Elliott-Halberstam-type level for the bilinear sequence) F(4) ~ 1.02
  would do at B = s^0.5 - but (i) is then missing. At B = s^0.234 where (i) exists, the share is
  0.96 and (ii) needs a constant below 1.04, which no level gives.

So the two requirements never hold at the same level: the comparison has no admissible B. This is
the exact shape of the obstruction in the products frame - not a missing idea about any one field,
but the fact that the base must be counted from below at the level where the plugs are counted
from above, and the two counts sit on opposite sides of the sifting limit. It is Chen's switching
frame with the interval too short, quantified: the twin conjecture would follow from a lower bound
for U(sqrt(2s)) that beats 1.98 s/(0.79 ln^2 s) x (1 - 1/1.02) - a non-sieve lower bound for
pairs of numbers both prime-or-P_2 in the stretch - together with an EH-level bound for the plugs.

## 5i. Where the base-open count comes from: the machine at level sqrt(2s) (2026-09-20 21:30)

The base-open columns U of the stretch at level B = sqrt(2s) are the holes of machine B (gears
5..B) in a window of 4c - 1 ~ B^2/3 columns at a generic phase (the stretch sits at height s^2 ~
B^4/4, far from machine B's origin, so no square-lemma simplification applies). Two exact facts
about that count, from the sandwich functions of section 5 (E3'):

- U >= k whenever G_{k-1}(B) < 4c - 1, where G_k(B) is the longest window of machine B with at most
  k holes. So a lower bound for U is a statement about the k-hole windows of machine B at scale B^2.
- Measured on the machines 5..23, G_k grows linearly in k at the mean hole spacing (slope 4-5 at
  q = 19, mean spacing 4.27): G_k(B) ~ F(B) + k x ln^2 B / 2.5. If that law held as a theorem up to
  k ~ B^2 / ln^2 B, it would give U >= (B^2/3 - F(B)) x 2.5 / ln^2 B ~ 6.6 s / ln^2 s, which is the
  mean of U (6.7 s/ln^2 s) - the lower bound the comparison of 5h needs at level sqrt(2s).

So the region lemma, in the products frame, rests on two statements: (a) the holes of machine B are
uniform at scale B^2 - no window of about B^2/3 columns holds fewer than half the mean number of
holes (the G_k law for large k); (b) the plugs, a one-class sieve on the bilinear partner sequence
h p -+ 2, are bounded with constant near 1, which needs an Elliott-Halberstam-type level of
distribution for that sequence. Statement (a) is the joint alignment of sections 5a-5d in density
form: E4 asks for one hole in a window of B^2/6, (a) asks for a proportional number in a window of
B^2/3. Both are statements about one sifted set at one scale, and neither follows from the rows'
individual shapes. Known in the direction of (a): for one class per prime Montgomery-Vaughan (1986)
bound the moments of the count of reduced residues in short intervals, which controls almost every
window but not the worst window; the two-class analogue would say the same.

**Statement (a) measured at small scale** (research/stack/r8/hole_uniformity.py, machines 11..23,
exact over the period; worst window's holes against the mean delta L). At L = B^2: 0.92, 0.88,
0.89, 0.88, 0.87 - the worst window holds seven eighths of the mean, stable in B. At L = B^2/3 (the
stretch's scale at level B): 0.86, 0.72, 0.72, 0.71, 0.69 - two thirds, falling slowly. At L = B^2/6
(the E4 scale): 0.57, 0.60, 0.64, 0.50, 0.43 in ratio, but 4, 5, 8, 7, 8 in holes against means 7,
8, 13, 14, 19: the worst window at the E4 scale keeps a nearly flat handful of holes while the mean
grows. So (a) holds in proportional form at the stretch's scale with constant about 0.7 on this
range, and E4's margin in absolute holes at scale B^2/6 is small and slow-growing - the record
F(B) ~ 0.17 B ln^2 B against B^2/6 is the same fact seen from the gap side.

**State of the lemma after the field programme.** Every proved line of the draft is in the kernel.
The one lemma has been carried through four exact forms (sparse stretch, alternating window, twisted
translates, base-open against plugs) and its obstruction has been located in each: the joint
alignment of a sifted set at the scale of its own square. The products frame adds the structure that
the plugs are Goldbach partitions and gives the comparison a number (share 1 - 0.79 ln^2 B/ln^2 s)
and a level analysis (no admissible level for the sieve). What would finish the proof is either a
lower bound for the holes of machine B in every window of length B^2/3 proportional to the mean, or
a direct argument for a hole in the window of length B^2/6 - the same statement at two strengths.

## 5j. The whole-range form (owner's direction, 2026-09-20 22:10)

Beyond q^2/6 an unstruck column of machine q is a twin candidate, not a twin (its members exceed q^2
and may have prime factors above q); the window is where the machine's verdict is final. But the
owner's instinct to use the whole period is right in this form: for consecutive primes q < q' the
new band is one gear, and the proved thin-band bound (kernel `thin_band`) reads

    F(q') <= max{ L : 2 ceil(L/q') >= h_q(L) },

h_q(L) the least number of holes of machine q in ANY window of length L over its whole period.
Hence

    (U')  every window of (q'^2 - q')/6 columns of machine q holds more than 2 ceil(q'/6) holes
          ==>  E4(q')  ==>  the window statement for q'.

(U') is a statement about machine q across its entire range, and it asks for about q'/3 holes
where the mean is q^2/(2.4 ln^2 q): a factor q/ln^2 q of slack. Verified at every computable
consecutive pair (research/stack/r8/thin_band_bound.py, base = the previous machine): F(19) <= 42
< 57, F(23) <= 57 < 84, F(29) <= 89 < 135 (true records 24, 33, 42). So the window statement for
machines 19, 23, 29 follows from the shapes plus the previous machine's worst-window counts.

**What (U') needs.** The record alone gives h_q(L) >= L/(F(q)+1) ~ q/ln^2 q holes in a window of
q^2/6, short of q/3 by ln^2 q/3. (U') is therefore the statement that record-sized gaps do not
cluster: over any q^2/6 columns the gaps average at most q/2, i.e. at most a fraction about 3/ln^2 q
of consecutive gaps can be near the record. Measured to q = 23 the worst window at scale q^2/6 holds
4 to 8 holes against the needed 2 ceil(q'/6) = 4 to 10 - the margin is real but thin at these
sizes, and the mean sits far above. This is the sharpest whole-range form on record: it removes the
window from the hypothesis and leaves a statement about how the machine's long gaps space
themselves over its period.

## 5c. Kernel status of the field lemmas (2026-09-20 17:40, proofs/LadderFields.lean)

`StrikesBy p n`; `no_adjacent` (E1); `strike_distance`, `strike_distance_ge` (E4c: two columns
struck by one gear are congruent mod p or at least (p-1)/3 apart); `class_count`, `gear_count`,
`gear_count_prime` (a gear strikes at most 2 ceil(L/p) columns of a run of length L);
`holes_in_covered_run` (E3' upper half: a run covered by machine q' holds at most 2 ceil(L/q')
holes of machine q, q' the next prime); `thin_band` (E4e); `five_consecutive`, `four_consecutive`
(E2, with the {5, 7} exception). 0 sorries; axioms propext, Classical.choice, Quot.sound; built and
audited by the manager. proofs/LadderWidening.lean (2026-09-20 18:20) adds Lemma B (`widening`,
`twin_column_strikers`, `twin_slot_persists`) and the sandwich's lower half (`struck_periodic`,
`tooth_of_class`, `align_single_hole`, `not_maxGapBelow_of_single_hole`): a run of the old machine
with one hole becomes a fully struck run of the next machine at a translate found by CRT. Every
proved line of this draft is now kernel-checked; the only lemma outside the kernel is E4.

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
| B | a later gear never strikes a twin slot | PROVED (kernel `widening`, `twin_slot_persists`) |
| C1-C5 | the shapes of the five fields | PROVED (C1, C5 kernel; C2-C4 elementary) |
| D1 | no set of gears blocks permanently | PROVED (CRT) |
| E1 | one row paints no two adjacent columns | PROVED (kernel `no_adjacent`) |
| E2 | two rows: exact longest joint run 4 / 3 / 2 | PROVED (kernel `five_consecutive`, `four_consecutive`) |
| E3 | a record is old runs joined at the new gear's teeth, F(q') <= (h+1)F(q) + h | PROVED (2026-09-20) |
| E3' | the sandwich G_1(q) <= F(q') <= G_{2 ceil(F/q')}(q); exact recursion by alignable chains | PROVED (kernel: upper `holes_in_covered_run`, lower `align_single_hole`) |
| E4c | in a covered window of q' the interior hole gaps of q are >= (q'-1)/3 and alternate in two residues | PROVED (kernel `strike_distance`, `strike_distance_ge`) |
| E4d | F(q') <= S_t*(q), the longest stretch of q with hole gaps >= (q'-1)/3 | PROVED (2026-09-20) |
| E4e | thin-band bound F(q) <= max{L : 2 sum_top ceil(L/g) >= h_B(L)}; window statement for 19 and 29 by shapes | PROVED (kernel `thin_band`) |
| E4f | per-class bound F(q') <= q' (F_T(q;q') + 1), F_T the twisted record (too weak by a factor q') | PROVED (2026-09-20) |
| E5 | origin square lemma: the next gear fills at most one hole in the window, the square column; window count law T(q') = T(q) - [q'+2 prime] + N(q^2, q'^2] | PROVED (elementary; kernel entry to add) |
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

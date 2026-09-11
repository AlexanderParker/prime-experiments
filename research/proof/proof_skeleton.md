# The proof by construction: the one document (2026-09-11)

This is the single place where the argument lives. Everything is built, not named; each claim
says why the construction makes it so. Labels: PROVED (the construction below, or a kernel
theorem in proofs/, named), MEASURED (no exceptions over a stated range, no proof), THEORISED
(not proved; if true it closes the conjecture). The history of how each part was found is in
research/proof/theory_tree.md (the tree) and research/proof/proof_skeleton_history.md (the
earlier diary form of this document). Nothing here depends on either.

---

# Part I. The construction

## 1. The line and a gear

Take the counting numbers 1, 2, 3, ... in a row: the line. A gear of size g is the rule "strike
every g-th number": it strikes g, 2g, 3g, ... and nothing else. A number is struck by a set of
gears if at least one of them strikes it. A prime is a number above 1 that no smaller number
above 1 divides, so a prime p is struck by no gear of size below p, and by its own gear at p.

## 2. The fold: what the gears 2 and 3 leave

Run the gears 2 and 3. Every number is struck except those of the form 6j - 1 and 6j + 1 (check
one strip of six: 6j, 6j+1, 6j+2, 6j+3, 6j+4, 6j+5: gear 2 takes the even ones, gear 3 takes
6j and 6j+3, and 6j+1 and 6j+5 = 6(j+1) - 1 survive). Pair the survivors as slots
(6j - 1, 6j + 1): two numbers at distance 2; call j the slot's column. Every prime above 3 sits
in a slot, and a twin prime pair (two primes at distance 2) is a slot with both members prime.
A gear g >= 5 strikes column j exactly when g divides 6j - 1 or 6j + 1, i.e. when
j = +-6^{-1} mod g: two residues per gear, the gear's teeth. [PROVED by the check; kernel
TopMachine.strikesR_iff]

## 3. What decides whether a number is prime: only the gears up to its square root

Take a number n and a prime P with n < P^2. If no prime below P strikes n, then n is prime.
Reason: if n were composite, n = a x b with 1 < a <= b, then a x a <= n < P^2 so a < P, and the
smallest prime factor of a is below P and strikes n. Contradiction. [PROVED; kernel
OneStepE.blocked_iff_sqrt, CoreLeftover.twin_of_rough for a slot]

## 4. The cuts, the sections, the machines

Set c_1 = 3. Let p_k be the smallest prime at or above c_k, and set c_{k+1} = p_k^2. So the cuts
are 3, 9, 121, 16129, 260,467,321, ... (p_k = 3, 11, 127, 16139, ...). Section k is the numbers
from c_k up to but not including c_{k+1}: a section runs from a number to about its square.
Machine k is the set of gears whose sizes are the primes in section k: machine 1 = {3, 5, 7},
machine 2 = {11, ..., 113}, machine 3 = {127, ..., 16127}. Every prime above 2 belongs to exactly
one machine, because the sections tile the line from 3 on. [construction; kernel
MachineStack.stack_eq_primesLE, cut_mono_le]

## 5. What the higher machines do inside a section: home strikes and echoes

Take a number n in section k+1, so p_k^2 <= n < p_{k+1}^2. By 3, n is prime exactly when no
prime below p_{k+1} strikes it, and the primes below p_{k+1} are precisely the gears of machines
1 .. k. Now take a gear p of machine k+1 or higher (p >= p_{k+1}) and suppose it strikes n, so
n = p x c. Then c = n / p < p_{k+1}^2 / p_{k+1} = p_{k+1}. Two cases: c = 1, n is the gear's own
size (a home strike); c > 1, c has a prime factor below p_{k+1}, a gear of machines 1 .. k that
already strikes n (an echo). So inside section k+1 the machines above k add nothing except
their own sizes, and a slot in section k+1 left unstruck by machines 1 .. k has both members
prime: it is a twin prime pair. [PROVED; kernel OneStepE.new_iff,
MachineStack.exhaust_home_or_echo, MachineStack.stack_open_iff_twin]

## 6. The hand-up: survivors become gears

By 4, the gears of machine k+1 are the primes in section k+1. By 5, those are exactly the
numbers in section k+1 that machines 1 .. k leave unstruck. So machine k+1 is BUILT from the
survivors of machines 1 .. k, and a twin prime pair in section k+1 is the same thing as two
gears of machine k+1 at distance 2, and the same thing as a slot machines 1 .. k left open.
[PROVED, an identity of the construction; verified with 0 mismatches at 16 sections to q = 23]

## 7. The base

Machine 1 = {3, 5, 7} contains the pairs (3, 5) and (5, 7) at distance 2. Section 2 = [9, 121):
the slots machine 1 leaves open are (11,13), (17,19), (29,31), (41,43), (59,61), (71,73),
(101,103), (107,109), eight twin pairs, the twin gears of machine 2. [PROVED by inspection]

## 8. The step (THEORISED)

For every k: the gears of machines 1 .. k leave at least one slot in section k+1 unstruck. By 6
that is the same as: machine k+1 has two gears at distance 2. In plain words: between every
cut and its square there is a twin prime pair. [MEASURED: every section along the chain from 3
to c_5 = 260,467,321, and along every chain from every base to 23; PROVED at three links, see
Part III.4]

## 9. The conclusion

Assume 8. Machine 1 has a twin-gear pair (7). By 8 machines 1 .. k leave a slot open in section
k+1, which by 5 is a twin prime pair and by 6 a twin-gear pair of machine k+1. So every section
holds a twin prime pair, the sections tile the line from 3 on, and there are infinitely many
twin primes. [PROVED given 8]

---

# Part II. The one statement, in its exact forms

Everything above 8 is built and everything after 8 follows, so the conjecture is exactly
statement 8. Its equivalent and sufficient forms, each proved to be so:

- (8a, the section form) Some column j with p_k^2 <= 6j - 1 and 6j + 1 < p_{k+1}^2 is struck by
  no gear g <= q, where q is the largest prime below p_{k+1} (q is just under p_k^2). [8 <=> 8a
  by 2 and 5]
- (8b, the run form) Let a be the column of the cut, 6a + 1 = p_k^2, and l = (p_{k+1}^2 -
  p_k^2)/6 the section's length in columns. Let L_a be the number of consecutive columns from
  a on that are each struck by some gear g <= q. Then 8 <=> L_a < l. [PROVED; kernel
  SquareColumn.twin_in_section_iff_L_lt, L_lt_iff]
- (8c, the record form, SUFFICIENT) Let F(q) be the largest number of consecutive columns
  anywhere on the line each struck by some gear of {5..q} (the machine's record). If
  F(q) < l then 8 holds at that link, since the section is a stretch of l columns and no
  stretch longer than F(q) is fully struck. Since l is about q^2/6, the uniform statement
  F(q) < q^2/6 for every prime q gives 8 at every link. [PROVED; kernel
  SquareColumn.section_twin_of_unstruck, section_twin_of_record, section_twin_of_record_W:
  the record enters only as a hypothesis, and the gear set need only contain the primes of
  [5, P)]
- (8d, the number form of 8c) Every interval of q^2 consecutive numbers contains a slot
  (n, n+2), n = 5 mod 6, with no prime factor <= q in either member. [8c restated on the line]
- (8e, the finer statement, SUFFICIENT, NOT NEEDED) A twin pair between every pair of
  consecutive prime squares [p^2, p'^2). It implies 8 and is far stronger; it is measured true
  for every prime to 10^7 (research/proof/frontier_floor_1e7.md). Its own study
  (research/proof/first_realisation.md) is Part IV.3.

---

# Part III. What is proved about 8

## III.1 Facts of the machine that bear on a run of struck columns

- The offset-strike law at a square: with p^2 = 6a + 1, a gear g strikes column a + i iff
  p^2 = -6i or 2 - 6i mod g. So near a square every strike is a statement about p^2 mod g,
  and a gear can strike offset i for some p only if -6i or 2 - 6i is a square mod g (the blind
  classes). [PROVED; kernel SquareColumn.offset_strike, offset_strike_modEq, blind_class,
  strikesZ_iff_root; no hypothesis on p or g beyond p^2 = 6a + 1]
- The two-prime lemma: a number below t^3 with no prime factor <= t is a prime or a product of
  exactly two primes above t. [PROVED, kernel CoreLeftover.primeOrSemiprime_of_rough_lt_cube]
- The core / tail split on a stretch of L columns: the gears <= 6L + 1 (the core) decide the
  pattern; each gear above (the tail) strikes at most one column of the stretch, and every tail
  strike on a core-open column is a product of two tail primes. [PROVED, kernel
  TopMachineRecord.loaded_record_rule, CoreLeftover.leftover_eq_card_twins]
- The depth law: a core-open column below (6L + 1)^2 is a twin. [PROVED, kernel
  CoreLeftover.twin_of_rough]
- The record is an exact cover: F(q) is the largest L such that L consecutive columns are
  covered by one phase per gear. [PROVED, law register W16]
- The parity law for a machine whose every gear exceeds 2m + 1 (m gears): F = 2m - (m mod 2).
  [PROVED, kernel TopMachineCrt.parity_law] The primes never satisfy the hypothesis (2, 3, 5,
  7 are gears).
- The mex form of the next open column in the free regime and its exact hypothesis. [PROVED,
  kernel TopMachineWalk.mex_form; research/proof/next_gap_closed_form.md]

## III.2 What 8 cannot be built from (each shown by a machine that has the property and no open slot)

- Counting alone: shifted gears (progressions a + g, a + 2g, ...) can strike every slot of a
  section (776 of the 1,226 gears below 10^4 strike the first 60 cycles at q = 5). Any argument
  using only how many numbers each gear strikes applies to them. [PROVED by construction, V12]
- Multiples alone: the machine of all primes below p_{k+1} together with the twin primes of
  section k+1 strikes multiples, has the divisor property, obeys every law on record, and leaves
  no open slot. What excludes it is only 4: a machine's gears are the primes of its own section.
  [PROVED by construction, V17]
- The gear set alone (for the finer statement 8e): keep the gears of the machine and change only
  their teeth; at p = 17, 29 and every cut 37..53 some two-tooth machine strikes the whole
  interval between consecutive prime squares; at 7, 11, 13, 19, 23, 31 none does. So 8e needs
  the real teeth +-6^{-1} mod g. [PROVED by construction, first_realisation.md; the family
  and the real teeth as its special case: kernel SquareColumn.FamilyBlocked, real_teeth,
  blockedZ_eq_family] For 8 itself
- Dilation, hand-up and the square-root rule together, WITHOUT the finite fold: the monoid M
  generated by 5 and the primes = 1 mod 6 (research/proof/origin_mechanic.md 6.2). Its
  strikes are multiples, its gears are its own irreducibles (hand-up), its sections lie
  between irreducible squares (square-root rule), and 8 fails on it: [25, 961) has 20 columns
  with both members in M and no twin irreducibles. So the three axioms that make the real
  machine (dilation, the finite fold 2, 3, the hand-up) are each necessary; each of the three
  counter-machines breaks exactly one. The finite fold enters the construction only through
  2, the strike-class law, which is the sieve's input, and the sieve's input is insufficient
  at the origin (the parity twin, IV.1). [PROVED by construction]
- What the nesting is and is not (origin_mechanic.md D3, proved exact): the struck set is the
  disjoint union over gears g of the dilates g . R_g, R_g the open survivors of the machine
  below g (0 mismatches over 198,798 members in 19 sections), so the pattern at the origin is
  built of scaled copies of the lower patterns, as the owner said; but the union is the same
  set however it is labelled, so the nesting forbids no cover that the real teeth do not
  already forbid. It is a fact of the construction, not a constraint.
  Where it is a different tool: every strike in a real section is n = g_0 m with 5 <= g_0 <=
  sqrt(n) <= m and m a survivor (0 violations at seven cuts), which the tooth-family killers
  (4 to 156 phantom strikes per section) and V17 (quotient 1) violate; the smallest section
  the real teeth hold while free classes kill it is the finer section at p = 17, columns 49
  to 59, where 13 and 17 are pinned to 13 x 23, 13 x 25, 17 x 19.
- The gear set alone, in full (research/proof/fold_mechanic.md): thin the primes by exactly
  the twin lowers. The thinned set has both classes of the fold at every scale, is
  equidistributed in every admissible class (the twin lowers have density 0), keeps dilation,
  the square-root rule and the side-swap rule, and its monoid kills every section (361 of 361
  finer sections, all three construction sections). Every two-class monoid is transparent:
  8 holds on it iff its generators keep a twin lower in the section (proved, 0 mismatches
  over 7,309 sections). So no property of the gear set is the missing axiom. What the
  thinning breaks, in the integers, is the line: (101, 103) is open under it and 101 is not
  a gear. The axiom is 2's first half, that every survivor of the fold is on the line, a gear
  or a multiple of a smaller gear. The classes are invisible to the column cover, and the
  fold's own sign, n mod 3, is sieve-visible; the invisible part of Liouville is the count of
  class +1 factors. [PROVED by construction]
  For 8 itself
  (sections of q^2/6 columns) the same question is the free-phase record h_2 against the
  square, Part IV.1.

## III.3 The counts, all closed

- The core's leftover on a stretch of the section's record length is an extreme value of a
  count (z about -4.5), the same for real phases, random phases and integer gear sets; min
  K_L = 0 exactly when the core's own record reaches L (S12); supply never binds; the leftover
  members are primes or products of two tail primes (P against P1 P2); the twin-free runs sit
  on the independent-slot prediction at every length; the one band deviation (base 3, 400-450
  slots) has no counterpart on five other sections. [MEASURED, research/proof/core_leftover.md,
  dead_branches_reopened_4.md, leftover_depth.md]

## III.4 Where 8 is proved

At exactly three links, by 8c and the certified records (research/proof/tree_review.md 4):

| link | section | q | F(q) | section length l (columns) |
|---|---|---|---|---|
| base 3, link 1 | [9, 121) | 7 | 5 | 20 |
| base 5, link 1 | [25, 841) | 23 | 34 | 140 |
| base 7, link 1 | [49, 2809) | 47 | 118 | 468 |

No fourth link: link 2 of base 3 needs F({5..113}) < 2688 and no record is certified past 59.

---

# Part IV. What a proof of 8 must produce, and the three faces of the wall

The certified records against the square (6F/q^2): 0.39, 0.37, 0.42, 0.39, 0.31, 0.36, 0.39,
0.33, 0.33, 0.32, 0.31 at q = 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, and 0.28..0.31 at 59
(F(59) in [161, 178]). Falling. Testing larger q cannot prove anything: the proof is a
closed-form statement from proven mechanics, and the candidates are these.

## IV.1 The length face: F(q) < q^2/6 (form 8c)

What is needed: a bound of exponent 2 on the record of a two-tooth machine with the real teeth,
with constant 1/6. What is known in print: for one tooth per gear the dimension-1 sieve gives
exponent 2 (Iwaniec 1978, constant unspecified); for two teeth the dimension-2 sieve's sifting
limit is 4.2665, exponent 4.27; the free-phase form (any two classes per gear) is Ziller-Morack
Conjecture 6, h_2 < p_n^2 - p_n, open, measured true to p_n = 73; lower bounds of Erdos-Rankin
/ Ford-Green-Konyagin-Maynard-Tao type are of order q log q log log log q, far below q^2. The
gap between exponent 2 and 4.27 is the parity barrier in covering form.

Answered (research/proof/length_face.md, ROOT, the parity twin built): the real-teeth
structure does not break the symmetry anywhere a record run can sit.
- The strike pattern on a stretch depends only on the phase vector (x mod g) and every vector
  occurs, so F(q) is the record of a fixed-separation two-tooth family; the real teeth add one
  thing, the separation 2k_g = 3^{-1} mod g, worth a factor 1.3-1.8 in length over free
  classes (A072753 against F - 1: 60/33 ... 236/144), never an exponent. [PROVED, LF1/LF2]
- The parity twin: O^- = the open columns whose members have Liouville product -1. It is empty
  below (q'^2 - 1)/6 > q^2/6 (a rough number below q'^2 is prime, and two primes have product
  sign +1), and its sieve data on [1, (q'^2 - 1)/6) is indistinguishable from the open set's
  to square-root size (0 of 1,326 cells above 3 sd at q = 23..53). So at every cut a set with
  the same sieve inputs violates F < q^2/6 by at least the section's length: no argument
  from those inputs can prove the target. [PROVED, LF3, with the measurement]
- Where the real teeth DO break the sign symmetry: exactly one place, the origin, below q'^2,
  where the open columns are twins (Liouville +1 forced). Every section of the construction
  sits there (a / l is about 1 / p_{k+1}); every record run sits far above it (x / L >=
  384,679 at m23, 10^9 at m37), where multiplicativity is invisible inside the stretch.
- Any uniform bound F(q) <= C q^2 with C < 1/6 already implies twin primes; the band between
  exponent 2 and 4.27 is open and sieve-unreachable. The phase-zero instance of the missing
  lemma is "a twin prime pair in (q, q^2 + 1]" for every prime q.

## IV.2 The count face

Every count on record (density, leftover, supply, the signed census) is matched by a machine
with no open slot (III.2, III.3). A proof cannot be a count.

## IV.3 The position face (the finer statement 8e only)

For 8e the interval between consecutive squares is often shorter than the record (at 7 of 11
certified cuts, e.g. p = 41: 28 columns against F = 91), so 8e is about where the record's
runs sit; the first run of a given length has no floor (at p = 29 it lies below the square);
8e <=> the run through the square column is shorter than the interval; the gear set alone
cannot give it (III.2). ROOT for 8e. Not needed for 8.

## IV.3a The fields (the owner's decomposition; research/proof/fields.md)

Split the hits by the striker's kind: field 1 the primes, field j the products of exactly j
primes >= 5, the square field the diagonal of field 2. Exact and proved: every field is the
primes dilated (F_j intersect 5S = 5 . F_{j-1}), so no field has a location rule the primes
lack; no field is periodic though the overlay is; field j hits a right member iff its class -1
factor count is even; dilating a column by g lands its members mirror-symmetric about 6 g m,
and the symmetric pairs of field j about that axis are exactly the pairs of field j - 1 at
equal distance, the twin being radius 1; below p'^2 the deep fields are confined to the small
gears (F_3 on {5, 7, 11, 13} at p = 53); the square field is empty in every finer section and
equals the cuts of the construction. The growing mirror symmetry is real (pairs per axis 2,
47, 110,944 along the chain from 3) and is the pairing of the primes about 6m with the twin
as its innermost radius. The split is the one a sieve cannot make: the fields are the
Omega-strata of the overlay, the sieve sees only the overlay, the column sign of IV.1 is the
parity of the field-index sum, and a partition of the struck set does not see its
complement. ROOT. [PROVED facts; MEASURED symmetry; the obstruction by construction]

## IV.4 The shape of what would close 8

The three faces meet at one place. The count face and the length face are both parity-blocked
away from the origin, and the one place where the real teeth are not a free two-tooth choice
is the origin: below q'^2 every open column is a twin, the sign symmetry is broken there and
nowhere else, and that is exactly where every section sits. So what would close 8 is a
mechanic of the origin: a reason, built from the construction, that the primes below p_{k+1}
cannot strike every column between p_k^2 and p_{k+1}^2, which uses that these columns lie
below p_{k+1}^2 (the square-root rule, the one fact that distinguishes the origin) and not a
count. The counter-machines of III.2 say the reason must also use that the gears are the
survivors of the sections below (the hand-up, 6). After origin_mechanic.md the demand is
sharper: the reason must use the finite fold (that the gears sit in the two classes +-1 mod 6
and strike exactly the dilates of those classes) in a way that is not the strike-class law
alone, because the monoid counter-machine has dilation, hand-up and the square-root rule and
still kills a section. The one construct named and not yet excluded: a non-count use of the
tail pins, the strikes g x m of the section with m a small prime, which the origin forces
(below g^3 every quotient is prime) and height does not. After fold_mechanic.md the demand is
sharper again: the reason cannot be a property of the gear set (both classes, equidistribution,
dilation, hand-up in the monoid sense: a thinned prime set has them all and kills every
section); it must use that the line is complete, every survivor of the fold being a gear or a
multiple of a smaller gear, at phase zero, in a way that is neither a free-phase cover nor a
count. The teeth, the classes and the gear set are exhausted.

A closed-form statement of the form "the machine {5..q} with its real teeth cannot cover q^2/6
consecutive columns, because ...", where the reason is a mechanic already proved (Part III.1)
or a new one built from the construction, and not a count. The one mechanic that is specific
to the real teeth and proved is the offset-strike law (III.1, first item): every strike is a
divisibility of 6j +- 1, so a fully struck run of L columns is an interval of 6L numbers in
which every number = +-1 mod 6 has a prime factor <= q. Nothing on record turns that into a
bound of exponent 2.

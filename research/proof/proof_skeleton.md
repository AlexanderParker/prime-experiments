# The proof by construction, as it stands (2026-09-07, rewritten as a construction)

Everything below is built, not named. Each claim is followed by why the construction makes it
so. Labels: PROVED (the construction below, or a kernel proof on record), MEASURED (no
exceptions over a stated range, no proof), THEORISED (not proved; if true it closes the
conjecture).

## 1. The line and a gear

Take the counting numbers 1, 2, 3, ... in a row: the line. A gear of size g is the rule "strike
every g-th number": it strikes g, 2g, 3g, ... and nothing else. A number is struck by a set of
gears if at least one of them strikes it. A prime is a number above 1 that no smaller number
above 1 divides, so a prime p is struck by no gear of size below p, and by its own gear at p.

## 2. The fold: what the gears 2 and 3 leave

Run the gears 2 and 3. Every number is struck except those of the form 6j - 1 and 6j + 1 (check
one strip of six: 6j, 6j+1, 6j+2, 6j+3, 6j+4, 6j+5: gear 2 takes the even ones, gear 3 takes
6j and 6j+3, and 6j+1 and 6j+5 = 6(j+1) - 1 survive). Pair the survivors as slots
(6j - 1, 6j + 1): two numbers at distance 2. Every prime above 3 sits in a slot, and a twin prime
pair (two primes at distance 2) is a slot with both members prime. [PROVED by the check]

## 3. What decides whether a number is prime: only the gears up to its square root

Take a number n and a prime P with n < P^2. If no prime below P strikes n, then n is prime.
Construction of the reason: if n were composite, n = a x b with 1 < a <= b, then a x a <= n < P^2
so a < P, and the smallest prime factor of a is below P and strikes n. Contradiction. [PROVED]

## 4. The cuts, the sections, the machines

Set c_1 = 3. Let p_k be the smallest prime at or above c_k, and set c_{k+1} = p_k^2. So the cuts
are 3, 9, 121, 16129, 260,467,321... (p_k = 3, 11, 127, 16139, ...). Section k is the numbers from
c_k up to but not including c_{k+1}. Machine k is the set of gears whose sizes are the primes in
section k: machine 1 = {3, 5, 7}, machine 2 = {11, 13, ..., 113}, machine 3 = {127, ..., 16127}.
Every prime above 2 belongs to exactly one machine, because the sections tile the line from 3
on. [construction]

## 5. What the higher machines do inside a section: home strikes and echoes

Take a number n in section k+1, so p_k^2 <= n < p_{k+1}^2. By 3, n is prime exactly when no
prime below p_{k+1} strikes it, and the primes below p_{k+1} are precisely the gears of machines
1 .. k. Now take a gear p of machine k+1 or higher (p >= p_{k+1}) and suppose it strikes n, so
n = p x c. Then c = n / p < p_{k+1}^2 / p_{k+1} = p_{k+1}. Two cases:
- c = 1: n is the gear's own size. Call that a home strike: the gear striking itself.
- c > 1: c has a prime factor below p_{k+1}, which is a gear of machines 1 .. k and already
  strikes n. Call that an echo: a strike on a number a lower machine already struck.
So inside section k+1 the machines above k add nothing except their own sizes, and a slot in
section k+1 left unstruck by machines 1 .. k has both members prime: it is a twin prime pair.
[PROVED here; also kernel-checked as OneStepE.new_iff and MachineStack.exhaust_home_or_echo]

## 6. The hand-up: survivors become gears

By 4, the gears of machine k+1 are the primes in section k+1. By 5, those are exactly the
numbers in section k+1 that machines 1 .. k leave unstruck. So machine k+1 is BUILT from the
survivors of machines 1 .. k, and a twin prime pair in section k+1 is the same thing as two
gears of machine k+1 at distance 2, and the same thing as a slot machines 1 .. k left open.
[PROVED, an identity of the construction; verified with 0 mismatches at 16 sections to q = 23]

## 7. The base

Machine 1 = {3, 5, 7} contains the pairs (3, 5) and (5, 7) at distance 2. Section 2 = [9, 121):
the slots machine 1 leaves open are (11,13), (17,19), (29,31), (41,43), (59,61), (71,73),
(101,103), (107,109), eight twin pairs, which are the twin gears of machine 2. Every section
computed along this chain (to c_5 = 260,467,321) and along every chain from every base to
q = 23 holds at least one. [PROVED for machine 1 by inspection; MEASURED beyond]

## 8. The step (THEORISED)

For every k: the gears of machines 1 .. k leave at least one slot in section k+1 unstruck.
By 6 that is the same as: machine k+1 has two gears at distance 2. In plain words: between
every cut and its square there is a twin prime pair.

## 9. The conclusion

Assume 8. Machine 1 has a twin-gear pair (7). By 8 machines 1 .. k leave a slot open in section
k+1, which by 5 is a twin prime pair and by 6 a twin-gear pair of machine k+1. So every section
holds a twin prime pair, the sections tile the line from 3 on, and there are infinitely many
twin primes. [PROVED given 8]

## 10. What 8 cannot be built from (each shown by constructing a machine that has the
property and no open slot)

- Counting alone. Suppose each gear of machines 1 .. k were allowed to strike a shifted
  progression (a + g, a + 2g, ...) instead of its multiples. Such shifted gears can be arranged
  to strike every slot of a section: 776 of the 1,226 gears below 10^4 suffice for the first 60
  cycles of a section at q = 5 [PROVED by construction, V12]. Any argument that uses only how
  many numbers each gear strikes applies to the shifted gears too, so it cannot give 8. The
  real gears strike multiples; that is the only difference, and it must be used.
- Multiples alone. What "strikes multiples" gives is: if n is struck, every multiple of n is
  struck; if n is unstruck, every divisor of n is unstruck [PROVED, V14: this is the whole
  content of phase zero]. But build the machine whose gears are all primes below p_{k+1}
  together with the twin primes of section k+1 themselves. It strikes multiples, it has the
  divisor property, it obeys every law on record, and it leaves no open slot in section k+1
  [PROVED by construction, V17]. What excludes that machine from the construction is only 4:
  a machine's gears are the primes of ITS OWN section, the survivors of the machines below.
- So 8 must use the recursion of 6: the gears of machine k+1 are the survivors of machines
  1 .. k, not an arbitrary set of primes. Every argument so far treated the gears as given.

## 11. The shape of what is missing

A property of the survivor set of a section that (a) is carried to the next section when the
cut is squared, (b) is not a count, and (c) forces two survivors at distance 2. The properties
proved to carry across the squaring so far are counts: every machine removes half the numbers
and a quarter of the slots that reach it [PROVED as Mertens; measured to q = 23]. Two
candidates, neither built yet:

- THEORISED 11a. Two gears at distance 2 strike the same slot earlier than any other pair (at
  (g + 4)/3 in the folded coordinate; deficit growing by exactly 4 per common period; PROVED,
  ArcFloor). So a machine whose gears include twin pairs strikes the start of the next section
  less than its gear count predicts. If that shortfall is large enough, at the start of the
  section, to leave one slot open before the machine's smallest gear has engaged (height c x g),
  then 8 holds and the twin pairs propagate themselves. Not measured.
- THEORISED 11b. The survivor set of section k+1 is what the construction itself generates
  from the survivors below. The machine of 10's second bullet is not generated that way. The
  property that separates a generated survivor set from an arbitrary set of primes with the
  divisor property, on one section, is the invariant. The parity-defined set (numbers with an
  even number of prime factors) also has the divisor property and also kills the count (the
  signed slot census is 0.1% of the slots against an 8% twin share, MEASURED), so the invariant
  is finer than anything a signed count can see. Not identified.

Everything above 8 is built. Everything after 8 follows. The gap is one sentence about what
the survivors of a section do to the section above it.

## 12. Update of 2026-09-09 (local evidence and the core-leftover lane)

- 11a is measured and weak: twin-gear collisions waste 0.6 to 11% of a pair's strikes, mostly
  in (5, 7); a density correction, not a forcing.
- The step at a link, measured at the core: over every stretch of the section's record length,
  the core's leftover behaves like an extreme value of a count (z about -4.5), the same for the
  real phases, random phases and integer gear sets; the twin-free stretch is not the minimum
  leftover but the one the tail finishes exactly; supply never binds. The recursion's only trace
  is the density deficit next to the origin, which lowers the leftover and never raises it.
- min K_L = 0 exactly while the core's own longest fully-covered run reaches L (S12, exact).
- So 11b is not in the leftover's size. What remains of 8 is the exact finish: among the core's
  leftover slots on a stretch, at least one is a twin. The next pass looks at what the leftover
  slots are (their members are core-free: primes of the tail range, or products of two large
  primes) and at the exact finish as a property of the primes' pairing at distance 2, one level
  down.

## 13. Update of 2026-09-11: the one statement, and what cannot prove it

The proof has exactly one unproved statement, 8: for every k the gears of machines 1 .. k leave
a slot of section k+1 unstruck, i.e. between every cut and its square there is a twin pair.
Everything else is PROVED (2, 3, 5, 6, 9) or is the base (7).

What 8 is NOT provable from, now with the certified numbers: the machine's record. Section k+1
in slots is (p'^2 - p^2)/6 for the cut p and the next prime p'; the machine {5..p} has a
certified longest fully struck run F(p) somewhere in its period. If the section were longer
than the record, 8 would follow from the record alone. It is not:

| cut p | p' | section, slots | F(p) | section / record |
|---|---|---|---|---|
| 17 | 19 | 12 | 18 | 0.67 |
| 29 | 31 | 20 | 43 | 0.47 |
| 37 | 41 | 52 | 88 | 0.59 |
| 41 | 43 | 28 | 91 | 0.31 |
| 43 | 47 | 60 | 103 | 0.58 |
| 47 | 53 | 100 | 118 | 0.85 |
| 53 | 59 | 112 | 145 | 0.77 |

At 7 of the 11 cuts from 13 to 53 the machine CAN strike a run longer than the whole section;
it does so elsewhere in its period, never inside the section. The section's actual longest
twin gap is far shorter (28 slots on the window to 59^2 at p = 53, engine_laws_m37.md U5; the
ratio section / longest gap inside stays above 4.6 to 10^7, frontier_floor_1e7.md). So 8 is a
statement about WHERE the machine's long runs sit, not how long they are: the first column at
which the machine {5..p} realises a struck run of the section's length lies above p'^2 / 6.
Every count-based route (the record, the core's leftover, the tail's supply, the order law that
bounds the next record by the previous one) measures length, and length is the wrong quantity.
The bound on the record in flight (the order law past 41) is therefore engine structure, not a
route to 8, and is closed as FACT when its document lands.

What a proof of 8 must produce: a lower bound on the first column of a struck run of length l
for the engine {5..p}, as a function of p and l, exceeding p'^2 / 6 at l = (p'^2 - p^2)/6. The
one new instrument that speaks about positions is the covering-problem decision of a word's
realisation (order_law_37_41.md): a word is realised exactly on a set of residue classes of the
column modulo the gears it uses, and its first realisation is the least column in that set.

## 14. Update of 2026-09-11 (the first-realisation lane): the exact form of 8, and its obstruction

- The position form of 8 is exact and simple: 8 holds at the cut p iff the fully struck run
  THROUGH the square column p^2 / 6 is shorter than the section, L_a(p) < l_p. That run length
  is the first-twin offset above p^2 (research/proof/frontier_floor_1e7.md: 0 exceptions to
  10^7; ratio L_a / l_p between 0.05 and 0.375 at p = 11..53). [PROVED equivalence; the
  inequality MEASURED]
- A floor on where the machine's first long run sits does not exist: at p = 29 the machine's
  first run of the section's length lies below the square (columns 111..134; the square column
  is 140) and 8 holds there only because that run ends before the square. So 8 cannot be proved
  by bounding first positions from below; it is about the one run through one column.
- The obstruction, built: keep the gears of machine k and change only their teeth (the two
  residues each gear strikes). At p = 17, 29 and every cut from 37 to 53 some such machine
  strikes the whole section; at 7, 11, 13, 19, 23, 31 none does. So 8 is not a property of
  "primes up to p as gears" - it is a property of the actual teeth, the columns k = +-6^-1
  mod g, which is where the primes' own arithmetic enters. Any proof must use the teeth.
- What was tried and closed: the record (section 13), the leftover count (12), the first
  position (this section). The remaining object is L_1(p), the run through the square column,
  and what the teeth 6^-1 mod g do to it.

## 15. What the teeth do at the square column (manager, 2026-09-11)

The square p^2 is the right member of the column a with p^2 = 6a + 1 (p^2 = 1 mod 6). A gear g
strikes the column a + i, at offset i from the square, iff g divides 6(a + i) - 1 or
6(a + i) + 1, i.e. iff

    p^2 = -6i  (mod g)   or   p^2 = 2 - 6i  (mod g).

So the offsets a gear can strike near a square are fixed by quadratic residues: g can strike
offset i for SOME prime p only if -6i or 2 - 6i is a square mod g (the blind classes of
R4.d.i.a), and it strikes offset i for THIS p iff p lies in one of at most four classes mod g,
p = +-sqrt(-6i) or +-sqrt(2 - 6i). This is the whole content of "the real teeth": the tooth
family of section 14 replaces these four classes by two arbitrary residues per gear, and then
the square is nothing; with the real teeth every strike near the square is a statement about
p^2 mod g.

The exact form of step 8 in these terms: the section is fully struck iff for every offset
0 <= i < l_p some gear g <= p has p^2 in {-6i, 2 - 6i} mod g. Call K_p the set of residue
vectors (p mod g)_{g <= p} that cover every offset; step 8 at the cut p says the prime p's own
vector is not in K_p. The family fractions of section 14 (1% at 17, 0.3% at 29) are the size
of K_p relative to all two-tooth choices, not relative to the vectors a prime can have; the
question is whether a prime's vector can ever lie in K_p, and nothing on the record says why
not beyond the count (0 exceptions to 10^7). Formalisation of this section is in progress
(proofs/SquareColumn.lean: the square column, the offset-strike law, the blind corollary, the
twin conclusion below p'^2, the equivalence with L_a < l_p).

## 16. Correction (manager, 2026-09-11, 17:40): sections 13-15 concern a finer statement, not 8

Sections 13, 14 and 15 measured sections between CONSECUTIVE prime squares, [p^2, p'^2) with
p' the next prime after p. That is not the construction of section 4. There, c_{k+1} = p_k^2
with p_k the first prime AT OR ABOVE c_k, so section k+1 = [p_k^2, p_{k+1}^2) with
p_{k+1} = nextprime(p_k^2): a section runs from a number to about its square (121 to 16129;
16129 to 260,467,321), and the machines striking it are all the primes below p_{k+1}, whose
largest, q, is just below p_k^2. The statement of sections 13-15 (a twin between every pair of
consecutive prime squares) implies 8 and is far stronger; its wall (position; the teeth) is
real for it, and those sections stand as the map of THAT statement. For 8 itself the picture
is the one already on record (tree_review.md section 4; base_and_step.md Q7), restated here so
the skeleton is not misread:

- The record route is OPEN for 8. Section k+1 has about q^2 / 6 columns (q the top gear), and
  the machine {5..q} cannot strike more than F(q) consecutive columns anywhere. So 8 at link k
  follows from F(q) < (p_{k+1}^2 - p_k^2)/6, and the certified records are a third of that:

  | q | F(q) | q^2/6 | 6F/q^2 |
  |---|---|---|---|
  | 13 | 11 | 28 | 0.39 |
  | 23 | 34 | 88 | 0.39 |
  | 31 | 58 | 160 | 0.36 |
  | 41 | 91 | 280 | 0.33 |
  | 47 | 118 | 368 | 0.32 |
  | 53 | 145 | 468 | 0.31 |
  | 59 | 161..178 | 580 | 0.28..0.31 |

  falling with q. This proves 8 at three links (base 3, 5, 7, link 1: q = 7, 23, 47), and at
  no other, because no record is certified past 59 (link 2 of base 3 needs F({5..113}) < 2688).
- What a proof of 8 by the record needs: F(q) < q^2 / 6 for every q, a bound of exponent 2 on
  the paired record of the primes' own teeth. In the free-phase form (any two classes per gear)
  this is Ziller-Morack's Conjecture 6, h_2 < p_n^2 - p_n, open in print; the only proved
  bounds on a two-class record come from the dimension-2 sieve, exponent 4.27 (sifting limit
  beta_2 = 4.2665 against the 2 that the section has). That gap is the parity barrier in
  covering form: the record route to 8 is exactly as hard as the sieve's parity problem, and
  it needs no position at all.
- The teeth obstruction of section 14 does not apply at the section's true length: the
  two-tooth families killed 60 to 112 columns with the gears up to 53; the section at q = 53
  has 468. Whether any two-tooth choice of the gears 5..q can cover q^2 / 6 columns is the
  free-phase question above (h_2 against the square), measured false to p_n = 73 by the
  Ziller-Morack table and unproved.

So the one unproved statement 8 has three faces, all on record: length (the record below the
square; sieve-blocked at exponent 4.27 against 2), count (the leftover; the parity object),
and, for the finer statement only, position (the run through the square column; the teeth).

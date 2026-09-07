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
are 3, 9, 121, 16129, 260,532,... (p_k = 3, 11, 127, 16141, ...). Section k is the numbers from
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
computed along this chain (to c_5 = 260,532,881) and along every chain from every base to
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

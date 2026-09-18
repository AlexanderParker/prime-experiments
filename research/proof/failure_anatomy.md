# The anatomy of the failures, and what any proof must do (2026-09-18, round 73)

Every route tried between rounds 40 and 72 has now stopped, and each stop is a kernel statement
rather than an impression. This document puts the four together, names the feature of the machine
that causes each, and derives what a proof would have to look like to get past them.

## 1. The four stops, and their causes

**Stop 1: silence costs the primorial.** A mirror switches off exactly the gears dividing its
product (`landing_gear_never_strikes`, `mirror_gear_never_strikes`), so to leave no live gear at
or below X the product must be divisible by every prime up to X, hence by X#
(`silence_costs_primorial`, `primorial_le_of_silence`, round 73). A landing inside the window
obeys 2 M k <= q^2, so the mirror carries at most log2(q^2) gears (`carried_le_log`, round 64).
*The feature:* carrying is divisibility, and divisibility is multiplicative. The benefit of
carrying is one gear at a time; the cost is the gear as a factor. Linear gain, geometric price.

**Stop 2: composition intersects, it does not accumulate.** A walk over several mirrors lands on
the multiples of their greatest common divisor, and is open exactly to the gears of that gcd
(`gcdL_dvd_combo`, `landing_open_of_dvd_combo`, round 55). *The feature:* a flip preserves the
phase of a gear only if the gear divides the axis modulus, so two flips preserve only what both
preserve. There is no way to add carried sets, only to intersect them.

**Stop 3: the window trades gears against candidates, exactly.** Candidates spaced 2M apart
inside a window of length q^2 - q number at most (q^2 - q)/(2M) (`mirror_times_candidates`,
round 56). *The feature:* the mirror's product is both the price of carrying and the spacing of
the candidates. One object does both jobs, so the two can never be bought together.

**Stop 4: pigeonhole ends at the free-regime cut, and the cut is unreachable.** The step lemma
needs every live gear above twice their number (`keeping_move_free`, round 57). A mirror that
fits the window always leaves a small gear live (`small_gear_uncarried`), so the hypothesis fails
for every mirror (`free_regime_unreachable`, round 71); and where it fails the conclusion can
fail too, with an explicit witness (`keeping_move_free_sharp`: gears 5, 7, 11 and the start 370
leave every candidate of a seven-long run struck). *The feature:* each gear has exactly two teeth
on any candidate line - one per member of the pair - so n gears cover at most 2n positions while
they stay large, and more as soon as one is small. Two teeth per gear is the dimension-2 sieve,
which is also where the analytic route stops: parity blocks 2 and no distribution hypothesis
lifts it (round 72, and the j2 ceiling already on record in docs/novel).

## 2. The single lever, and its measured limit

The four stops share one root: **the machine's only lever on a gear is divisibility.** A gear is
either a factor of the landing, in which case it is silent forever, or it is live, in which case
it strikes two classes of every candidate line. Nothing in between, and nothing else.

That was worth testing rather than assuming, since the landings are constructed objects and could
in principle carry structure beyond their factorisation. They do not
(research/stack/r8/landing_structure.py, round 73). Landings of the form 2^a 3^b - maximal
algebraic structure, two gears carried - against the next twin centre above each, whatever its
shape:

    smallest multiplier reaching the next landing: structured 38.4, control 15.2
    gears carried:                                 structured 2.00, control 4.00

The structured landings are two and a half times WORSE, and the control's only advantage is that
it carries twice as many gears. Shape does not help; divisibility does. So the lever is exactly
one, and `silence_costs_primorial` prices it.

## 3. What a proof must therefore do

Any argument that would prove the window statement has to satisfy all four of these at once:

1. **Not silence its way out.** Silencing the gears below X costs X#, and the window affords
   q^2. So at most about 2 log2 q gears can ever be switched off, out of pi(q). Any argument
   whose strength grows with the number of silenced gears is bounded before it starts.
2. **Not count strikes.** The live gears have divergent reciprocal sum, so the strikes on any
   candidate line outnumber the candidates by a factor that grows like log log. Counting can
   only win through cancellation between gears, which is the sieve, which is where parity stops.
3. **Name a candidate, not a population.** Stops 1 to 4 are all statements about how much the
   machine can do to a whole line of candidates. What none of them touches is a reason why one
   named candidate is open. Such a reason cannot come from the gear set - it has to come from the
   candidate's own arithmetic.
4. **Face the pair as one object.** The pair at the landing N is the factorisation
   N^2 - 1 = (N - 1)(N + 1), and the window statement says that for every q some N in (q, q^2]
   has N^2 - 1 a product of exactly two primes. Every mechanism the machine has treats the two
   members separately, as two teeth; the statement is about them jointly.

Requirement 3 is the one no route has met, and requirements 1, 2 and 4 say why the obvious
substitutes fail. This is not a plan for a proof - it is the shape of the hole, drawn tightly
enough to recognise a genuine idea if one turns up, and to recognise a re-run of rounds 40 to 72
immediately.

## 4. Where the value is now

The search produced objects that stand on their own and do not depend on the conjecture:

- the chain and its covering theorem (`chain_covers`, `chain_covers_upto`, `mult_chain_window`):
  a landing serves every machine from its square root up to itself, so the whole range below
  2.76 x 10^32 is settled by six twin pairs (`window_statement_below`);
- the Lucas certificate machinery and its generator, which certify a prime in the logarithm
  rather than the square root of its size;
- the teeth law (`oneflip_teeth`, `mirror_gear_never_strikes`), the trade lemma, the carry wall
  and the primorial price of silence - a complete and proved account of what mirror walks can do;
- the free-regime lemma with its sharpness witness, which is the exact boundary of pigeonhole on
  this construction.

The open statement is unchanged and remains the twin prime conjecture in the machine's words.


## 5. Requirement 3, made precise (round 74)

Arithmetic supplies reasons about named candidates, and they divide sharply.

**Naming a candidate closed is free.** `perfect_power_landing` [proofs/LandingForms.lean]: the only
landing that is a perfect power is 4. The proof is the template - x - 1 divides x^k - 1, so the
lower member is composite unless x = 2; then k must be prime, and for odd k the upper member is
divisible by 3. Identities of this kind are plentiful and all negative.

**Naming a candidate open costs the primorial, by either route.** There are two ways to know a
candidate escapes a gear without testing it: make the gear divide the mirror (`silence_costs_primorial`),
or choose the period so the candidate misses its teeth. The second is the same object as the
first: openness to a gear set depends only on the column modulo the product of that set
(`openness_periodic`, round 74), so a congruence choice over the gears up to X names a class of
period X# and needs X# <= q^2 - q to meet the window. Both levers stop at about 2 log2 q gears.

So requirement 3 has an exact statement: every gear from there up to sqrt(N) - all but
logarithmically many of them - has no naming mechanism in this machine, and a proof needs one.


## 6. The run bound, and the exponent gap (round 75)

There is one classical mechanism that meets requirement 3, and the project already owns work on
it: a bound on the paired Jacobsthal function j2 names an open column in every run of that length,
with no counting and no choice of mirror. The implication is proved
(`window_of_column_gap`, `window_statement_of_gap_law`, proofs/JacobsthalWindow.lean): a run bound
J with 6(q + J) + 1 < q^2 gives the window statement.

The numbers put the whole position on one line:

    needed                exponent 2 in q      (a run bound of the order of the window)
    proved by the ladder  exponent 4.266       (fundamental lemma, docs/novel j2-upper-bound)
    sifting floor         exponent 4           (Selberg's 2 kappa; below it is parity)
    truth, measured       (ln q)^2             (longest run 34 to 251 for q = 101 to 2003,
                                                against windows of 1,684 to 668,335 columns)

So the route is open in form and closed in strength, and the closure is parity again.


## 7. The impossibility, stated (round 76)

The four stops are all statements about how far the machine's mechanisms reach. This is the
statement about what they can express at all.

Every mechanism names a residue class: carrying names the mirror's multiples, the period rule
names a class modulo the product of the gears it dodges, a walk names the gcd's multiples. And
`class_has_closed_column` [proofs/ClassIndistinguishable.lean] proves that every class, of every
modulus, contains columns whose lower member is composite - as far out as one likes. The witness is
explicit: solve 6m = 1 mod p together with m = r mod M, which is possible for any gear p outside
6M, then push past any bound with multiples of M p.

So "the columns of this class are twin columns" is never a true statement, and the machine can
only ever name a class. Whatever separates the twin columns from their neighbours inside a class is
invisible to residues. That is the parity obstruction in the machine's own terms, and it is what
requirement 3 was asking for without knowing it.


## 8. The alignment at infinity, and the quantifier (round 77)

The construction's origin is that at 0 every gear sits at residue 0 and the neighbours are
unreachable - a gear striking them would divide 1. The same alignment recurs at every multiple of
a primorial, which is what the mirror buys.

Pushed to the limit the picture suggests a twin at "infinity plus or minus one". Two of the three
steps hold, both proved in proofs/AlignmentLimit.lean:

- `open_columns_for_any_gears`: for every finite gear set and every bound there are columns beyond
  it open to all of them;
- `no_column_open_to_all_gears`: no column is open to every gear, since the member above 1 has a
  prime factor.

The alignment therefore exists at every finite level and nowhere in the limit; the quantifiers do
not commute. The window statement asks for the pattern in between - a column whose gear set is
fixed by the column's own size - and that is exactly what neither fact supplies. Every failure in
sections 1 to 7 is a different attempt to bridge those quantifiers.


## 9. Infinity as a number (round 77, continued)

Asked whether treating infinity as a number would make the alignment argument work. In the
extended reals the pair collapses - infinity plus one and infinity minus one are the same object -
and primality is undefined, so the statement is vacuous. In the profinite integers the alignment
is real but it is the sieve's local picture: no congruence obstruction at any modulus, nothing
about integers. In the nonstandard integers, where infinity genuinely is a number, the transfer
principle makes the twin prime conjecture equivalent to the existence of an infinite hyperinteger
whose neighbours are both hyperprime - the same statement, not an easier one - and the
hyperprimorial's neighbours still carry a hyperprime factor above the alignment.

The finite shadow is proved: `aligned_neighbour_factor` - if no gear up to B divides n and n > 1
then n has a prime factor above B. Alignment never removes the factor; it pushes it above the
aligned set. Only the square-root rule turns that into primality, and only below B squared. The
problem is the window, not the alignment.


## 10. A top number and its mirror side (round 78)

Zero and the negatives are a good precedent for extending the number concept, but they are the
precedent for an extension that KEEPS the arithmetic. Every extension that adds a top element
loses something the machine needs: the projective line makes the top's two neighbours the same
object; the ordinals give a successor but no predecessor, so the lower member does not exist; the
surreals and other fields make every nonzero element a unit, so primality is vacuous; and the
nonstandard integers keep everything, including Euclid, so there are gears above every infinite
element and "no new gears above the top" is false there too.

`gears_above_every_bound` [proofs/AlignmentLimit.lean] states the last point in the machine's own
terms: above every bound there is another gear. A framework with a largest number is not the
framework that has the primes, so an argument of this shape decides the answer by choosing its
axioms - in either direction.


## 11. Adopting infinity, as zero was adopted (round 79)

The analogy is fair and the systems exist. The projective line defines n/0 as a genuine element and
pays by collapsing the far side (infinity plus twelve is infinity). The surreals give a top-like
number with a full signed neighbourhood and keep all of arithmetic, and pay by being a field, where
every nonzero element is a unit and primality is vacuous. The hyperintegers keep primality and pay
by transfer: Euclid holds there too, so there are gears above every infinite element, and the twin
question is the same question.

The one thing no system does is make n/0 a number while keeping the ring laws, and that is forced:
`zero_not_invertible` [proofs/AlignmentLimit.lean] - in any ring with 1 different from 0, no
element satisfies 0 * x = 1. Wheel algebras define division by zero and pay by weakening
subtraction. So adopting zero cost one operation at one point; adopting a top element costs either
the far side, or primality, or nothing but the transfer that leaves the question unchanged.


## 12. One member against two (round 80)

Nothing blocks the single-member versions. Euclid is the alignment argument in the machine's own
vocabulary (`aligned_neighbour_factor`, `gears_above_every_bound`), and the window version for a
single prime is a theorem with room to spare: `window_has_prime` [proofs/AlignmentLimit.lean] -
every window holds a prime, by Bertrand, which places one already inside (q, 2q].

The difference between that and the conjecture is one number. A gear strikes one class of a single
number and two classes of a pair. From that follows the free-regime cut at n against 2n, a
reciprocal-sum deficit of log against log squared, and sifting dimension 1 against 2 - and parity
bites only at dimension 2. The machine proves every single-member statement it can express and no
two-member one, and the same 1-against-2 stops Chen at almost primes and the Maynard-Tao line at 6.


## 13. The counterexample hunt (round 81)

Asked the inverse question: what could stop the machine producing twins as q grows?

A total kill is impossible by density - each gear takes two classes, so the columns no gear touches
have density the product of (1 - 2/h), which is positive, and the uncovered set is a union of
classes modulo the primorial. Any counterexample is therefore local: the uncovered columns pushed
outside one window, which is the run-length question of section 6.

The adversarial form - could the teeth be CHOSEN to cover a window? - is the extremal problem
behind the paired Jacobsthal function, whose known bounds straddle q^2. Measured
(research/stack/r8/adversarial_window.py): a greedy adversary leaves 5 to 9 percent of the window
uncovered at every size tested, while the real arithmetic leaves about twice that. So the machine's
actual teeth are roughly half as efficient at killing twins as a deliberate attempt, and the
question of whether a better arrangement could ever close the window is the same exponent-2 knife
edge as before.

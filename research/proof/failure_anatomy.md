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

# 23. The stack of machines and the exhaust

## In plain words

The route uses two machines.  The first, the motor, is built from the small primes up to some
cut; it repeats after a certain number of steps, and that number -- the product of its gears --
is its period.  The second, the wheels, is built from the primes that start where the motor
stopped and run up to the motor's period.  That rule does not stop at two.  Apply it again and
there is a third machine, from the top of the wheels to the wheels' period; again, and there is a
fourth.  The machines are stacked, and each one's gears run from the top of the machine below to
the period of the machine below.

Stacking them that way has one immediate consequence, and it is the whole point of this file.
Take any machine and the machine **two** levels above it.  Every gear up there is bigger than the
whole period of the machine down here -- that is exactly what the rule says, because the lower
edge of a machine's gear list is the period of the machine two below.  A gear bigger than a
period has a stride longer than one full turn of that machine.  So inside one turn it can catch
at most one number, which in pair coordinates is at most two positions, one per tooth; and while
it manages that, the lower machine's entire pattern -- every one of its open slots -- has come
round in full.  The visitor from two floors up sees the complete pattern and touches two cells of
it.

One floor up is a different story, and the difference is arithmetic, not luck.  A machine's
period is the product of its gears, and a product of two or more numbers each at least two is
bigger than any one of them.  So no gear ever spans its own machine, and no gear of the next
machine up spans the machine below it either: those gears stop at the lower machine's period, and
only the very top of the list could reach it -- which needs the machine below to be a single
gear.  Room appears exactly two floors up and not before.

Then the second half.  Everything above the wheels, taken together, is the **exhaust**: smoke
produced by the tiers below, rising away, and the question is whether it ever comes back down
into the engine.  On one stretch it provably does not.  Fix a cut `C` and look at the numbers
between `C` and `C^2`.  Let some gear above `C` strike a number `n` there.  Write `n = p m`.
Either `m = 1`, and the gear has struck **its own number** -- a home strike, the one place a
large prime can appear -- or `m` is at least 2, and then `m` is smaller than `C` (else `p m`
would already be past `C^2`), so `m` brings a prime factor at or below the cut with it, and that
small gear strikes `n` too.  The big gear's strike is an **echo** of a strike already there.
Nothing above the cut removes anything on that stretch that was not already removed.

Put together: on the stretch `(C, C^2]` the only machines that decide anything are the ones at or
below the cut, and a pair left open by them is a pair of primes two apart.  This is the sieve to
the square root, which is old; what is new here is reading it as a statement about a stack of
machines and about which machine can reach which -- and the consequence, which is a limit on
where the proof can live.  The search is capped: the motor, the one machine in use above it, and
their clutch, on the quiet zone.  Nothing higher in the stack helps, and nothing higher hurts.
Climb to the next cut and the picture is identical, with the same missing instrument: a bound for
the in-use machine.

## Vocabulary

**Stack, tier, cut.**  The **stack** is the whole sequence of machines.  **Tier 1** is the motor,
the primes up to `q`.  **Tier `k + 1`** is the primes in `(cut_{k-1}, cut_k]`, where `cut_k` is
tier `k`'s **period** -- the product of its gears.  So the **cuts** are `cut_0 = q`,
`cut_1 = q#` (the primorial, tier 1's period), `cut_2` = the wheels' period, and in general

        cut_k  =  product of tier k's gears  =  the lower edge of tier k + 2 .

Tier 2 -- the primes in `(q, q#]` -- is the **wheels** of file 22.  "Cut" is not "rung": a rung is
one step `q -> q'` of the ladder.

**Exhaust.**  Every tier above the wheels, taken together.

**Home strike, echo.**  A **home strike** is a gear striking its own number (`n = g`).  An
**echo** is a strike on a number that a gear of a lower tier already strikes.

**Span.**  A gear **spans** a machine when its stride -- the gear itself -- is at least that
machine's period.

**Smooth zone, quiet zone.**  For gears `(q, Q]` on a range: the **smooth zone** `[1, Q]`, and the
**quiet zone** `(Q, Q^2]`.  `n` is **`q`-smooth** when every prime factor of `n` is at most `q`.

**Strike, open (as in file 22).**  Gear `g` strikes the pair `n` iff `g | n` or `g | n + 2`
(`Strikes`); a pair struck by no gear of `G` is **open** (`IsOpen`).  In the single-number view
`g` strikes `n` iff `g | n` (`StrikesN`), and `n` is open iff no gear strikes it (`OpenNum`).  A
pair is open iff both of its members are (`isOpen_iff_openNum`).

Classical translation.  A tier is the set of primes in an interval; a cut is a primorial-like
product; the exhaust cap is the statement that sieving `(C, C^2]` by the primes up to `C` leaves
exactly the primes -- the sieve of Eratosthenes stopped at the square root.

## Statement

Throughout, `G` is a `Finset` of naturals, `tier q k` and `cut q k` are the stack on base `q` as
defined above, and `primesLE C` is the set of primes at most `C`.

**S1 (STRIDE CONTAINMENT).**  Every gear of tier `k + 2` exceeds `cut q k`, the period of tier
`k`; hence

 (a) for `P < g` and any `x`, the set `{ j < P : g | x + j }` has at most one element;

 (b) for `P < g` and any `x`, the set `{ j < P : g strikes the pair x + j }` has at most two
     elements -- one per tooth;

 (c) with `P` the period of tier `k + 1` and `g` any gear of tier `k + 3`, (b) applies: a gear two
     tiers up strikes at most two pair positions of any window of the lower tier's period;

 (d) over that same window tier `k + 1`'s own pattern repeats in full: `IsOpen` for a gear set is
     invariant under a shift by any common multiple of its gears, in particular by its period.

**S2 (NON-CONTAINMENT).**  If every element of `G` is at least 2 and `|G| >= 2` then
`g < prod G` for every `g in G`.  Hence: no gear spans its own tier; and a gear of tier `k + 2` is
at most `cut q (k+1)`, tier `k + 1`'s period, so it spans tier `k + 1` only if it **equals** that
period -- impossible when tier `k + 1` has two or more gears.  Spanning starts exactly two tiers
down.

**S3 (THE EXHAUST CAP).**  Let `C < n <= C^2` and let `p` be any divisor of `n` with `p > C`.
Then `n = p` (a **home strike**) or `n` has a prime factor `r <= C` (an **echo**).  No primality
of `p` is needed.  Consequently, for `2 <= C` and `C < n <= C^2`,

        n open under all the primes <= C   iff   n is prime ,

and for `C < n`, `n + 2 <= C^2`,

        the pair n open under all the primes <= C   iff   n and n + 2 are twin primes .

In stack form: the union of tiers `1 .. k + 1` is exactly `primesLE (cut q k)` (the cuts being
nondecreasing up to `k`), every gear of every tier from `k + 2` upwards lies above `cut q k`, so
on the quiet zone `(cut q k, (cut q k)^2]` every exhaust strike is a home strike or an echo of a
gear of tiers `1 .. k + 1`, and a pair those tiers leave open there is a twin prime.

The first step needs no hypothesis: `q <= q#` for every `q` (Bertrand), so **the motor and the
wheels together are exactly the primes up to `q#`**, and on `(q#, (q#)^2]` a pair they leave open
is a twin prime, unconditionally for `q >= 2`.

**S4 (THE ZONE LAW, top_machine_4.md L46).**  Let `G` be the primes in `(q, Q]`.

 (a) For `0 < n <= Q`: `n` is open iff `n` is `q`-smooth.  Hence for `n + 2 <= Q` the pair `n` is
     open iff `n` and `n + 2` are both `q`-smooth.

 (b) For `0 < n <= Q^2`: `n` is open iff `n = s P` with `s` `q`-smooth and `P` either 1 or a prime
     above `Q` -- `q`-smooth times **at most one** prime above `Q`.

## Proof

**S1.**  (a) If `g | x + j` and `g | x + j'` with `j, j' < P` then `g | j - j'`, and
`|j - j'| < P < g`, so `j = j'`.  (b) The pairs struck by `g` in the window are those `j` with
`g | x + j` or `g | (x + 2) + j`; each set has at most one element by (a) applied at `x` and at
`x + 2`, and the union of two sets of size at most 1 has size at most 2.  (c) Every gear of tier
`k + 3` exceeds `cut q (k+1)`, which is tier `k + 1`'s period, by the definition of the cut
sequence; apply (b) with `P` that period.  (d) If `g | P` then `g | n + P` iff `g | n` and
`g | n + P + 2` iff `g | n + 2`, so `g` strikes `n + P` iff it strikes `n`; a tier's period is
divisible by each of its gears.

**S2.**  Let `g in G` and pick `h in G` with `h != g`, which exists because `|G| >= 2`.  Then
`prod G = g * prod (G \ {g})`, and `prod (G \ {g}) >= h >= 2` because every factor is at least 1
and `h` is one of them.  So `prod G >= 2g > g`.  For the tiers: a tier's gears are primes, hence
at least 2, so no gear spans its own tier.  A gear of tier `k + 2` lies in
`(cut q k, cut q (k+1)]` by definition, so it is at most tier `k + 1`'s period; if it spanned that
tier it would equal the period, and then a gear `h` of tier `k + 1` would divide it and be
strictly smaller than it -- impossible for a prime.

**S3.**  Write `n = p m`.  If `m = 0` then `n = 0`, contradicting `C < n`.  If `m = 1` then
`n = p`: the home strike.  Otherwise `m >= 2`, and from `m (C + 1) <= m p = n <= C^2 = C C` and
`C C < C (C + 1)` we get `m < C`.  The least prime factor `r` of `m` then satisfies `r <= m < C`
and `r | m | n`: the echo.  (Note `C >= 2` is forced by `C < n <= C^2`.)

For the "open iff prime" form, let `2 <= C < n <= C^2`.  If `n` is prime, no prime `<= C` divides
it, since such a divisor would have to be `n` itself.  Conversely let `n` be open and let `p` be
its least prime factor.  If `p <= C` then `p` is one of the gears and strikes `n`, so `p > C`; the
cap then gives `n = p`, hence `n` prime, or a prime factor `<= C`, which again strikes `n`.  The
pair form is the single-number form at `n` and at `n + 2`, using that a pair is open iff both
members are.

For the stack form, the union of tiers `1 .. k + 1` is
`(1, cut_0] u (cut_0, cut_1] u ... u (cut_{k-1}, cut_k]` intersected with the primes, which
telescopes to `(1, cut_k]` when the cuts are nondecreasing; that is `primesLE (cut q k)`.  Every
gear of tier `j + 2` exceeds `cut q j >= cut q k` for `j >= k`, so it is above the cut and the
cap applies.

For the unconditional first step, `q <= q#`: by strong induction on `q`.  For `q < 2` both sides
are immediate.  Otherwise write `q = p m` with `p` the least prime factor.  If `m = 1` then `q` is
prime and is one of the factors of `q#`.  If `m >= 2` then `p <= m` (the least prime factor of `q`
is at most any divisor of `q` that is at least 2), `m < q`, and by Bertrand there is a prime `p'`
with `m < p' <= 2m <= p m = q`.  That `p'` is a factor of `q#` and is not a factor of `m#`, so
`q# >= p' * m# >= p' * m > m * m >= p * m = q`, using the induction hypothesis `m <= m#`.

**S4.**  (a) If `n` is `q`-smooth then no prime above `q` divides it, and every gear exceeds `q`.
Conversely, if a prime `p > q` divides `n` then `p <= n <= Q`, so `p` is a gear and strikes `n`.
(b) If `n = s P` with `s` smooth and `P` = 1 or a prime above `Q`, a gear `g in (q, Q]` dividing
`n` divides `s` (so `g <= q`, false) or divides `P` (so `g = P > Q`, false).  Conversely let `n` be
open.  If no prime above `Q` divides `n`, then every prime factor is at most `Q`, and openness
rules out `(q, Q]`, so `n` is `q`-smooth and `P = 1`.  Otherwise pick a prime `P > Q` dividing
`n`, write `n = s P`, and let `r` be a prime factor of `s`.  Then `r | n`, so `r <= q` or `r > Q`
by openness; and `r > Q` is impossible, since then `P r | n` with both factors above `Q` would
force `n >= (Q + 1)^2 > Q^2`.  So `s` is `q`-smooth.

## Status

**Kernel-checked, zero sorries.**  `proofs/MachineStack.lean` (library `MachineStack`, namespace
`TopMachine`), 48 declarations.  **No `native_decide`, no `Lean.ofReduceBool`, no `decide`** --
every theorem is an ordinary proof, so nothing depends on any set being small enough to enumerate.

Definitions: `gearsIoc`, `primesLE`, `cut`, `tier`, `stack`, `Spans`, `Smooth`, `CutMono`; the
strike and open predicates are file 22's `Strikes`, `IsOpen`, `StrikesN`, `OpenNum`, reused
unchanged.

| statement | Lean name | hypothesis |
|---|---|---|
| cuts are the tier periods | `tier_prod` | none |
| gear of tier `k+2` above `cut q k` | `gear_gt_cut`, `gear_le_cut` | none |
| S1(a) at most one multiple per stride | `card_dvd_window_le_one` | `P < g` |
| S1(b) at most two pair positions | `card_strikes_window_le_two` | `P < g` |
| S1(c) a gear two tiers up spans, and is contained | `spans_two_below`, `stride_containment` | none |
| S1(d) the lower pattern repeats | `strikes_add_period`, `isOpen_add_period`, `tier_pattern_repeats` | gears divide the shift |
| S2 product beats each factor | `lt_prod_of_two_le` | gears `>= 2`, `\|G\| >= 2` |
| S2 no self-span | `not_spans_self` | tier has `>= 2` gears |
| S2 no span one tier down | `gear_le_period_below`, `not_spans_below` | tier below has `>= 2` gears |
| S3 home strike or echo | `exhaust_home_or_echo` | `C < n <= C^2`, `C < p`, `p \| n` |
| S3 open iff prime | `openNum_iff_prime` | `2 <= C < n <= C^2` |
| S3 open iff twin | `open_iff_twin` | `2 <= C < n`, `n + 2 <= C^2` |
| S3 a stack prefix is the primes below the cut | `stack_eq_primesLE` | `CutMono q k` |
| S3 the exhaust is above the cut | `exhaust_gear_gt_cut` | `CutMono q j`, `k <= j` |
| S3 exhaust silent on the quiet zone | `exhaust_silent` | as above |
| S3 the stack's window statement | `stack_open_iff_twin` | `CutMono q k`, `2 <= cut q k` |
| `q <= q#` (Bertrand) | `le_prod_primesLE`, `cut_zero_le_one`, `cutMono_one` | none |
| motor + wheels = primes `<= q#` | `stack_one` | none |
| motor + wheels open iff twin on `(q#, (q#)^2]` | `wheels_open_iff_twin` | `2 <= q` |
| S4(a) the zone law | `smooth_zone_num`, `smooth_zone`, `wheels_smooth_zone` | `0 < n`, `n + 2 <= Q` |
| S4(b) the quiet zone | `quiet_zone` | `0 < n <= Q^2` |
| pair view = two number views | `isOpen_iff_openNum` | none |

Build (from `proofs/`):

    ~/.elan/bin/lake.exe build TopMachine TopMachineWheel TopMachineCrt TopMachineWalk MachineStack

Result: green, 2244 jobs, no warnings, no errors; `MachineStack` 9.3 s cold, ordinary elaboration
(peak memory a normal `lean.exe`).  Axiom audit over all 48 declarations (`lake env lean` on a
`#print axioms` file, and the same block appended to `proofs/AxiomCheck.lean` behind
`import MachineStack`): every declaration is `[propext, Classical.choice, Quot.sound]` or smaller
(`decStrikesInt` is `[propext]`; `Smooth` is `[propext]`; `Spans`, `strikes_add_period` are
`[propext, Quot.sound]`); **no `sorryAx`, no `Lean.ofReduceBool`, no `Lean.trustCompiler`**.

**One hypothesis is not derived, and it is not a technicality.**  `CutMono q k` -- the cuts are
nondecreasing -- is a statement about the density of primes in `(cut q j, cut q (j+1)]`, not about
the stack's arithmetic, and it is **false at the bottom for `q = 2, 3`**: the cuts at `q = 3` are
`3, 6, 5, 1, ...`, so tier 3 is the single gear `{5}` and tier 4 is empty.  At `q = 5` they are
`5, 30, 215656441, ...` and climb, and the first step `cut_0 <= cut_1` is unconditional
(Bertrand).  So the theorems that need a whole prefix of the stack carry `CutMono` explicitly, and
the motor-and-wheels case -- the one the project uses -- carries nothing.

Verified computationally before and after formalising (bounded brute force, sympy): the cuts at
`q = 2, 3, 5, 7, 11` (the degeneracy at `q = 2, 3` as stated); `n <= n#` for `n < 400`, 0
failures; the open-iff-twin law at `C = 30` over all `n` in `(30, 898]`, 0 failures; the exhaust
cap at `C = 30` over all `n` in `(30, 900]` and all their prime factors, 0 failures; the zone law
and the quiet-zone form at `q = 5`, `Q = 30` over `n <= 900`, 0 failures; stride containment for
gears 31, 37, 101, 1009 over 250 windows of length 30, 0 failures.

## Prior art, and what is new

**Leverages.**  The exhaust cap is the **sieve to the square root**: a number in `(C, C^2]` with
no prime factor at or below `C` is prime (Eratosthenes; Legendre's form of it).  It is already in
this corpus twice, in the shapes the earlier files needed: file 01's Theorem 4
(`Horizon.exists_prime_factor_lt`) and file 15's layer law (`Layer.layer_novelty`,
`Layer.minFac_lt_or_eq`).  The containment statements are elementary arithmetic (a product of two
or more factors `>= 2` exceeds each of them; a gear larger than the window length has at most one
multiple in it).  The primorial bound `q <= q#` uses **Bertrand's postulate**
(`Nat.exists_prime_lt_and_le_two_mul` in mathlib).  The zone law is L46 of
`research/proof/top_machine_4.md`, this project's own, and is standard smooth-number bookkeeping.

**New.**  Not the arithmetic -- the framing and what it rules out.  (i) The **stack** as an
object: the cut recursion `cut_k = ` the period of tier `k` `= ` the lower edge of tier `k + 2`,
formalised as a two-step recursion with the tiers read off it.  (ii) **Stride containment two
tiers down, and its exact failure one tier down**: spanning is not a matter of degree but a
threshold that the construction rule places precisely two levels up, with the one-level statement
false except for a single-gear tier.  (iii) The **exhaust dichotomy in machine form** -- every
strike of a gear above the cut on the quiet zone is a home strike or an echo -- with the
observation, visible only once it is in the kernel, that **primality of the exhaust gear is never
used**: any divisor above the cut behaves the same way.  (iv) The consequence, which is the
reason the file exists: the search is **capped** to the motor, the one in-use machine above it,
and their clutch on the quiet zone.  (v) The honest converse: cut monotonicity is a prime-density
input, and the stack degenerates at `q = 2, 3`.

**Not new.**  The mathematics of S3 is Eratosthenes; the mathematics of S1 and S2 is counting
multiples in an interval and comparing a product with its factors.  No prior-art search has been
run for the stack framing itself; it is carried in `docs/novel` only through the theory tree node
R4.b.viii, not as a claimed novelty.

## Relationship to the conjecture

**It does not prove it, and it does not weaken it.**  Nothing in this file bounds a twin-free
run, produces an opening, or says anything about where openings are.  What it does is fix the
search space, in both directions:

* **Nothing above the wheels can hurt.**  On the quiet zone `(cut_k, cut_k^2]` every exhaust
  strike is a home strike or an echo, so no machine above the in-use one can close a pair that the
  machines at or below the cut leave open.  A pair open for tiers `1 .. k + 1` there **is** a twin
  prime -- an iff, not an implication.
* **Nothing above the wheels can help.**  For the same reason: the exhaust adds no structure to
  exploit.  Its gears are either silent on the zone or duplicating strikes already made.  A route
  that hoped to find leverage higher up the stack is closed.
* **The shape is fixed and it repeats.**  At every cut the question is the same: do the known
  machines plus one in-use machine leave an open pair in `(C, C^2]`?  Theorem (E) of
  `position_frontier.md` already says the effective machine at a column is exact, so the stack
  introduces no new interactions; it turns the window statement into a ladder of the same question
  at the primorial cuts `q, q#, ...`.  **The missing instrument is the same at every cut -- a
  bound for the in-use machine -- and would be found once and carried up.**

Nothing measured enters any of the statements.  The one non-arithmetic input is Bertrand's
postulate, used for a single unconditional step, and `CutMono` beyond that, carried as a
hypothesis and false at `q = 2, 3`.

## Where it is used

* **Caps the search** (theory tree R4.b.viii): the proof effort belongs to the motor, the in-use
  machine and their clutch, on the quiet zone; the exhaust is out of scope, provably.
* **Licenses file 22 to describe the wheels alone.**  The zone laws S4 say what the wheels do on
  the two zones of a range in closed form (`q`-smooth pairs below `Q`, smooth-times-one-prime
  above), which is where the in-use record lives (`top_machine_4.md` L46-L48).
* **Gives the clutch its exact window statement.**  `wheels_open_iff_twin` is the two-machine
  formulation with no hypothesis beyond `q >= 2`: motor plus wheels, open pair on `(q#, (q#)^2]`,
  iff twin prime.
* **Gives the counting face of the wall a precise form.**  S1 says an exhaust gear touches at most
  two of the `prod (g - 2)` slots per lower period, so "does the exhaust ever clear a whole
  period" is a sparse-covering question, not a structural one.
* **Warns off the tower-as-progress reading.**  The stack repeats the same difficulty at every
  cut; climbing it is not progress, and this file is the reason why.

## Source

`research/proof/theory_tree.md` node R4.b.viii (the owner's stack of machines, the stride
containment, the owner's cap and its consequence) and R4.b.vii (the zone of tranquillity);
`research/proof/top_machine_4.md` section 4, L46-L48 (the zone law and the in-use record);
`research/proof/top_machine_lean.md` (the kernel ledger, round 35 section: names, hypotheses,
build and audit); kernel source `proofs/MachineStack.lean`, on top of `proofs/TopMachine.lean` and
`proofs/TopMachineWalk.lean`.  Vocabulary: the glossary of the repository `README.md` (stack,
tier, cut, exhaust, home strike, echo, span, smooth and quiet zones).  Neighbouring files:
`docs/proofs/01-the-route.md` (Theorem 4, the same sieve fact in the route's shape),
`docs/proofs/15-layer-and-shadow-laws.md` (the same fact per layer),
`docs/proofs/22-top-machine-laws.md` (the wheels on their own terms).

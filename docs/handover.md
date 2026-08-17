# Twin primes: full research handover

## What this is

**The problem.** The twin prime conjecture: there are infinitely many primes `p` with `p + 2` also prime.

**The project.** An attempt on it from a single mechanical model, built from scratch rather than borrowed. Every
prime is treated as a *gear* - a wheel of circumference `q` that turns alongside all the others and blocks exactly
one position per rotation, leaving every other position of that rotation open. Composite gears, phases, and the
interaction of their cycles then determine which slots stay open, and a twin prime is a slot that every gear leaves
open. The whole programme asks one question in many forms: *by what mechanism do the turning cycles conspire to
leave two adjacent candidate slots open, and can that mechanism be shown never to stop?*

The model was developed deliberately without reaching for established sieve machinery. That was a constraint on
method, not a claim that the machinery is irrelevant - and it means the vocabulary here is home-grown and the
reader should expect to recognise some results under other names.

**What the project produced.** A large body of mechanical structure about the machine, most of it verified
computationally at explicit scale, and one artefact believed to be new: a **closed-form method for finding the next
twin prime without walking**, which computes the distance to the next open pair directly from the gear phases and
was verified to `k = 10^16`. Also several new computed values of the maximum-gap function, and seventeen lines of
enquiry each pushed to the point of a result or a counterexample - catalogued in section 2. And a
**machine-checked Lean formalisation of the reduction** (section 3a): the statement that this programme's target is
equivalent to the twin prime conjecture is verified by Lean's kernel, not merely argued.

**What it did not produce.** The conjecture. Every route reduces to a single bound - how far apart consecutive open
pairs can be, compared against the window in which the gears can certify them - and that bound resisted every
attempt made here. Section 5 sets out where each attempt stalls.

## The basis for the hunt

These are the project's premises, set out by the person directing the work rather than derived from it. They are
what make the hunt worth running rather than a search for a lottery ticket, and they are stated as arguments to be
evaluated on their merits, with the current status of each. Several have already turned into theorems, which is the
main reason for taking the rest seriously.

**1. The lower bound is 2, and the task is to reach it.** Bounded-gap results establish that *some* finite gap
recurs infinitely often. The floor of that ladder is 2. So the target is not an open question of existence in the
abstract - it is a matter of finding, among the mechanisms already in hand, the one that drives the bound down to
its floor. *Status: framing, and the framing under which every reduction in section 1 was derived.*

**2. There is a reason the open slots always admit a twin pair inside the current window.** Not a hope - a claim
that a mechanism exists and is discoverable, and that it lies in how the cycles interact to expose the `1` and `5`
slots of the 6-cycle. Everything in section 0 is an attempt to write that mechanism down. *Status: the mechanism is
partly written down - the turn law, the tooth budget, the arcs, the merge transform - and the outstanding step is
exactly the localisation claim of section 1. Unproved, but the object it refers to is now explicit rather than
vague.*

**3. The gears return to their starting point, and no gear can outpace the base cycle.** The primes are gears
joined in sequence. As they turn they diverge, but they are integers, so after the least common multiple every
gear is simultaneously back where it began. The only way the machine could stop presenting the twin configuration
would be for a new gear to run *faster* than the lower cycle - and that is impossible by construction, since a
larger circumference means a slower walk across the slots. *Status: **both halves are now theorems.** Every gear
and every combination of gears walks the 6-cycle at exactly `+-1` per rotation, never faster (section 3, item 2),
and the pattern is exactly periodic with the required symmetry (item 4). This argument predicted a result before
the result was found.*

**4. The machine is fully constructed up to infinity, and infinite rotation resets it to its state at 0.** Nothing
is assembled as one walks; all gears exist at once. At 0 every gear divides 0, so every gear shields rather than
threatens - the complete-shield position - and twins sit immediately beside it. To deny infinitely many twins is
therefore to claim the machine stops presenting a configuration it presents next to 0, against its own
periodicity. A corollary was argued explicitly: **if the infinite object is off the table then the conjecture is
meaningless** - it asks for a proof about something that does not exist - so reasoning with the completed infinite
machine is legitimate rather than a category error. *Status: slot 0 is always exposed and the pattern is periodic
and symmetric about 0, both theorems (items 3 and 4). The gap is **localisation, not existence**: the recurrence
period is the primorial, about `e^y`, while the gears only certify on `(y, y^2]`. See section 2.15.*

**5. Gears block once per rotation; every other position is open.** A correction that had to be made repeatedly
against drift in this work, and the single most productive instruction in it. The machine is not to be evaluated
for blockers but for **exposure**. Two teeth in pair-index space is a *derived* consequence - the two rotations
whose block happens to land on a candidate slot - not a second block. *Status: adopted as the definition in
section 0, and the source of the tooth budget, the arc structure and the minimal size law.*

**6. Efficiency is not the goal; a closed form is.** Walking forward to test whether a location is a twin is not a
deterministic answer, however fast the walk. What is wanted is a formula that yields the position. *Status:
achieved for the next twin (section 2.9), verified to `k = 10^16`. Not achieved for the bound on that formula's
output, which is the open problem.*

**7. A contradiction argument is acceptable, and does not need statistics.** The admissible shape was stated
directly: *this is the mechanism that generates twin primes; for it to stop, condition X would have to hold; X
cannot hold because of Y.* If the constructor works, then for it to fail some specific condition must occur, and
ruling that condition out is a proof - no density estimate required. *Status: the mechanism half exists and is
verified. No condition X with a provable impossibility has been found. The nearest approach is the repulsion form
of section 3, item 23. See section 2.16 - this is probably the framing most worth carrying forward.*

**8. Rare mechanisms still count.** A rule that fires seldom is not thereby disqualified, and a low hit rate is
not a defect. The machine is exact; a construct that produces a twin once in a great while still produces one.
*Status: a methodological premise, and the reason the low-energy beats were tested rather than discarded - see the
Method section, where the same principle appears in its sharpest form.*

**9. State-of-the-art difficulty is not evidence of impossibility.** The instruction was explicit: do not appeal
to the absence of a known proof, and do not conclude that a breakthrough is required before one has been looked
for. *Status: methodological. Worth noting that it was correct twice in this programme, where a route was
abandoned on a difficulty judgement and later found to work - see the audit note in the Method section.*

## What the handover is for

This document exists to transfer the entire context to a reader with stronger reasoning than the one who produced
it, so that the reader can:

1. **re-derive the mechanics from scratch** (sections 0 and 2) rather than inherit them;
2. **check the findings**, using the stated scales to re-run what was tested and push past what was not;
3. **judge the inferences separately from the data** - the measurements here are the reliable part, the reasoning
   is not;
4. **form its own target statement.** Section 1 gives the reduction this programme reached most recently, but that
   is the latest line of enquiry, not a settled thesis, and it may not be the right one to attack.

**Suggested reading order: the basis for the hunt, then the Method section, then section 2, then section 1.**
Reading section 1 first risks anchoring on one framing. Section 2 is the full inventory of lines explored, in the
order explored, with what each produced and where each stopped - several were abandoned while still viable, and at
least two were abandoned for reasons later shown to be wrong.

The model has no settled name; "gears" and "machine" are used throughout as the working vocabulary, defined in
section 0.

## Status of the claims below

**Nothing here should be taken as settled.** These are one programme's findings with the evidence attached so it
can be checked. Where something is called *proved*, that means a proof was written here and believed correct - not
that it has been independently verified, and the reasoning is the weakest part of this work. Where something is
called *refuted*, that means a counterexample or a measured divergence was found at a stated scale; the data
should be reproducible, but the inference drawn from it may not be sound. Several claims in this document were
written up as results and then overturned by the next data point, so the base rate of error here is not low.

Section 6 lists claims that were tried and appear to fail, with the evidence against each - not to rule anything
out, since some may fail for a fixable reason or not fail at all. Implementation defects found and fixed during the
work are omitted, but their existence is part of why the computational claims are better re-run than trusted.

**Notation.** `q` ranges over odd primes, `y` is the gear bound, `P(y) = prod_{3<=q<=y} q`,
`d = prod (1 - 2/q)` is the density of open slots, `A = prod (q-2)` their count per period. `F` denotes a maximum
gap, subscripted by frame; the frames are defined in section 0.5 and confusing two of them cost real time here.

**Sections.** The basis for the hunt · Method · 0 the mechanical model · 1 the most recent reduction and its
equivalent forms · 2 every line explored · 3 findings offered as established · 3a what is formalised in Lean ·
4 what was computed and at what scale · 5 where the argument stalls · 6 claims that appear to fail · 7 pathways not
formalised · 8 files.

---

## Method: the working rule that produced these results

Read this before working. It is not a stylistic preference: it repeatedly outperformed the instincts of the work
it was correcting, and every substantial result below came from following it.

> **Focus on the mechanics.** Work out how the machine actually works and how its components interact. Build
> and test, build and test. Use statistics only to *verify* a construction that is already built and tested -
> **never to rule out an approach that has not been tried.**
>
> **Be especially wary of statistics that show a quantity approaching zero.** The prime machine is a precise
> instrument. A hit that lands at a frequency approaching zero at infinity *still lands*. Discarding a
> construct because it is a statistical outlier is exactly the trap that would prevent the solution being
> found.

That second paragraph is not abstract caution. It was decisive twice here.

**Case 1 - the low-energy beats.** The factorised spectrum (section 2.8) has beats whose energy share is
negligible, and standard signal-processing practice discards them. Tested instead of assumed: the `L1` norm
grows about `2.06` per gear, so a single low-energy beat still closes a slot. Truncation is not lossy, it is
**wrong** - it loses exactness rather than precision. A statistical instinct would have thrown away the terms
that carry the answer.

**Case 2 - the normalisation that looked like a vanishing margin.** The hazard ratio `h(L)/d` tends to 1 for
every `L` as the gear set grows. Read as a margin, that says the slack vanishes everywhere and no lossy
argument can ever work - and that reading was written down, and was wrong. `h(1)/d = 1/(1-d)` also tends to 1,
so the whole scale is collapsing, not the margin. On the correct scale `kappa = (h/d - 1)/d`, the margin
converges to about `1.67`, comfortably away from zero. **A quantity approaching zero was an artefact of how it
was being measured, and taking it at face value nearly closed a live route.**

The same rule explains the failures. Every claim in section 6 was recorded because something was generalised from
a small sample before being tested at scale - a pattern in nine numbers, a boundedness claim from six data points,
an ordering that held to `y = 37` and broke at 41. Build-and-test caught all of those; reasoning alone had
endorsed several of them. That is the direct reason for the framing of this document: the reasoning done here is
its least reliable component, and the measurements are its most reliable, so a reader whose reasoning is stronger
should re-derive rather than inherit.

**An audit note the reader should act on.** Two closures in this document do not fully meet the standard above,
and should be re-examined rather than trusted:

* the claim that the forbidden-word antidictionary is **infinite** (section 6, item 9) rests on extrapolating a
  count that was still rising at the edge of a search box of length 16. That half is an extrapolation. The
  closure of that route does not depend on it - the second reason given there is exact - but the extrapolated
  half should not be reused;
* "the per-`j` recipe does not scale" was asserted early on a term-count estimate and **later refuted** by
  building it: `2548` contributing terms at `L = 39` against a predicted `5.5 * 10^11`. A scaling judgement made
  without building the thing is exactly the failure mode this rule guards against, and it cost a working route
  for most of the programme.

Where sections 5 and 7 use measurement to exclude something, they exclude a *proof technique* on exact
arithmetic grounds - `sum 2/q > 1` is an identity, not an estimate - never a construct or an untried approach.
If any exclusion below reads as ruling out a mechanism because its effect looks small, treat it as unproven and
test it.

---

## 0. The mechanical model: gears, teeth, and the machines investigated

The whole programme is built on one physical picture, and the formal statements later are all shadows of it. This
section is the vocabulary the rest of the document uses.

### 0.1 The machine

Each odd prime `q` is a **gear** of circumference `q`, turning at the same rate as every other gear. All gears
exist simultaneously - the machine is fully constructed, not assembled as you walk - and each one, **per
rotation, blocks exactly one position; every other position of that rotation is open.** This framing matters:
the object of study is the *openings*, not the blocks. Asking "what blocks slot `n`" recovers trial division
and nothing more; asking "what configuration of gears leaves a slot open" is the productive question, and it is
the one every result below answers.

Gears 2 and 3 together make a **6-cycle**. In it, only the residues `1` and `5 mod 6` can host a prime above
3, so the machine presents candidate slots in adjacent pairs `(6k-1, 6k+1)`. A twin pair is such a pair with
both members open to every gear. The base 6-cycle is the fastest thing in the machine: **every gear and every
combination of gears walks it at exactly `+-1` per rotation** (item 2 of section 3), so the `1, 5` slots keep
being presented at the maximum rate and later gears can only sample them.

### 0.2 Teeth, roles, arcs

Move to the **pair index** `k`, where slot `k` denotes the pair `(6k-1, 6k+1)`. Gear `q >= 5` divides one
member exactly when `6k = +-1 mod q`, that is

    k = +- 6^{-1} mod q.

So in pair-index space a gear presents **two teeth**, at `+u_q` and `-u_q` with `u_q = 6^{-1} mod q`. This is
*derived*, not a second block: the gear still blocks once per rotation, but the two rotations `j = +-1 mod 6`
are the ones whose block lands on a candidate slot. Tooth separation is `2u_q = 2 * 6^{-1} = 3^{-1} mod q`,
concretely `(q+1)/3` or `(q-1)/3`. The `+-` symmetry is forced by `6 * 6^{-1} = 1`.

Relative to gear `q`, every pair index has exactly one of three **roles**:

    killer   k = +- u_q mod q      2 of q residues   one member divisible by q
    shield   q | k                 1 of q residues   q divides the midpoint, so neither member
    miss     everything else       q - 3 residues    q lands elsewhere in the rotation

That is the **tooth budget**: one shield, two killers, `q-3` misses, universal in every gear. Between the two
teeth the gear leaves two **arcs** of open slots, of lengths `q - 2u_q - 1` (about `2q/3`) and `2u_q - 1`
(about `q/3`), and the short arc is centred exactly on the shield.

### 0.3 Slip, sub-machines, and the turn law

Two distinct things were both called **slip** in this work and must be kept apart. *Cycle slip* is `|P - Q|`
between two periods - how far two cycles drift per revolution, and the original sense of the word here. *Machine
slip* is `P mod q`, the phase a composite machine of period `P` presents to a new gear `q`. The second is what
composes.

A **sub-machine** is any subset `S` of gears, with period `P = prod_{q in S} q` and an exposed set that is a
union of complete residue classes mod `P`. Sub-machines compose by CRT, and the composition is governed by the
**turn law**: adding gear `q` to a machine of period `P`, an open class `k_0` spawns `q` daughter classes
`k_0 + jP`, of which **exactly two are struck** - at

    t = (+- u_q - k_0) * P^{-1} mod q

- and `q - 2` survive. Iterating the turn law is exactly the `prod (q-2)` survivor count, and it is the
mechanism behind the generating polynomial `prod (q - 2 + 2x)` of item 11.

Every gear also blocks itself, in a precise sense: the **lower tooth of gear `q` is the index of the pair
containing `q`**, and twin gears share their lower tooth (item 5). So the machine's own components sit at
identifiable teeth, which is why the certified window has a square-root shape - a gear can only decide slots
below its square.

### 0.4 The window: why any of this terminates

Slot `k` is *certified* by the gears up to `y` precisely when `6k + 1 <= y^2`, since a surviving composite
would need a prime factor above `y`. Inside that window, exposure to all gears `<= y` and genuine twinhood are
the same thing - the **window identity** `survivors(y,K) = T(6K+1) - T(y)`, exact and verified from `y = 11` to
`1009` (item 6). Outside it they diverge. This is the entire reason the problem is finite-dimensional at each
scale: **a bound on how far apart consecutive openings can be, compared against the window `(y, y^2]`, settles
the conjecture.** That comparison is Reduction A of section 1.

### 0.5 The machines investigated, and how they relate

Six coordinate systems were built and cross-checked. Two of them were confused with each other for a long stretch
of the work, so the relations are stated explicitly.

1. **`n`-space.** Positions are integers; gear `q` blocks its multiples, one per rotation. Base gears 2 and 3
   leave `n = +-1 mod 6`. Good for the closed-form next-twin method and for intuition; poor for counting,
   because the twin condition is a pair condition.
2. **Pair-index (`k`) space - the real frame.** Slot `k` is the pair `(6k-1, 6k+1)`; gear `q >= 5` blocks two
   residues `+- 6^{-1} mod q`, separation `3^{-1} mod q`; **gear 3 is inert**, since `6k +- 1` is never
   divisible by 3. This is where the conjecture actually lives, and `F_k(y)` is its maximum gap.
3. **The adjacent (halved) frame.** Every odd prime, gear 3 included, blocks the adjacent pair
   `{o, o+1} mod q`. This is the frame of `maxgap.rs`, `coverbound.rs`, `holegap.rs` and `hazard.py`, chosen
   because "two adjacent residues" makes the covering search uniform across gears.
   **Relation to frame 2, exactly:** gear 3 blocks one of any two adjacent positions, so it confines the
   exposed set to a single class mod 3; rescaling that class by `3` leaves each gear `q >= 5` blocking two
   residues separated by `3^{-1} mod q` - which *is* the pair-index separation, since `2 * 6^{-1} = 1/3`. So
   the adjacent frame with gear 3 **is** the pair-index machine, scaled by 3. Hence `F(2,y) = 3 F_k(y)` and
   `F2_adjacent = 3 F2_k`, verified for seven and six gear sets respectively. Lengths in the older documents
   are adjacent-frame; divide by 3 for `k`-space.
4. **The separation family.** Generalise: gear `q` blocks two residues at arbitrary separation `s_q`. Frame 3
   is `s_q = 1`, frame 2 is `s_q = 3^{-1}`. `F` depends on the separation vector, which is why results proved
   "for any separations" are stronger and why one refutation (item 3 of section 6) is stated with an explicit
   separation vector `(1,3,3,3,3,3)`.
5. **Cofactor reindexing.** Index by the cofactor `P/q` rather than by `q`. Blocks then correspond to gears,
   the phase slope of a block is the gear's own step, and the same square-root threshold appears from both
   sides. Useful as a duality check; it did not yield a bound.
6. **The frequency domain.** The indicator of the exposed set has an exactly factorised spectrum
   `Ehat(k) = prod_q ehat_q(k t_q mod q)` (item 12), agreeing with a direct FFT to `1.1e-16`. Every gear
   contributes a beat. **Truncation is not available**: the `L1` norm grows about `2.06` per gear, so a single
   low-energy beat still closes a slot, and discarding it loses exactness rather than precision.

A seventh view, the **offset-vector or covering view** - choose an offset per gear and ask which runs can be
covered - turned out not to be a separate machine at all. By CRT it is frame 3 with a single translation
parameter (item 17); the apparent design freedom is illusory.

### 0.6 Validator versus constructor

Two kinds of result are available and should not be confused.

A **validator** decides whether a given slot is a twin: the gcd form `gcd(36m^2 - 1, primorial(sqrt(6m+1))) = 1`
(item 10), the order form (gear `q` threatens iff `ord_q(6m) <= 2`), and the unit-group form (two units of
`U(P(y))` differing by 2 inside the first `y^2` residues). These are exact but say nothing about *where* the
next twin is.

A **constructor** produces the next twin without walking: each gear's distance to its next tooth is
`min((u_q - m) mod q, (-u_q - m) mod q)`, and the minimum over gears is the distance to the next bite. This
runs to `k = 10^16`, and no prior form of it was found. What it does **not** do is bound its own output - the bite
distance is computed, not bounded - and that gap between constructing and bounding is precisely the open problem.
Section 5 sets out why every attempt to bound it has failed.

---

## 1. The target, and the reductions arrived at most recently

The target is the twin prime conjecture. What follows is the *latest* reduction line, reached in the final
stretch of work; section 2 lists the others. Treat it as one candidate framing among several, not as settled.

**Two coordinate systems, one problem.**

*`k`-space (the real frame).* A twin pair is `(6k-1, 6k+1)`. A prime `q >= 5` divides one member exactly when
`k = +- 6^{-1} mod q`, so gear `q` blocks two residues mod `q`. Gear 3 never acts, since `6k +- 1` is never
divisible by 3. Write `F_k(y)` for the maximum gap of the set of `k` exposed to all gears `<= y`.

*The adjacent frame.* Every odd prime, gear 3 included, blocks the adjacent pair `{o, o+1} mod q`. Write
`F(2,y)` for the maximum gap. **`F(2,y) = 3 F_k(y)` exactly** - verified for seven gear sets - because gear 3
confines the exposed set to one class mod 3.

**The reduction.** Slot `k` is certified as a genuine twin by gears up to `y` exactly when it is exposed to
all of them and `6k+1 <= y^2`. Slot `k = 0` is exposed for every gear set, and gaps are at most `F_k(y)`, so an
exposed slot exists in `(y/6, y/6 + F_k(y)]`. Therefore:

> **REDUCTION A.** The twin prime conjecture follows from
>
>     F_k(y) <= (y^2 - y)/6      for all sufficiently large y.

**The equivalent form that removes all apparent combinatorial freedom** (section 3, item 17):

>     F(2,y) = 1 + max gap of { n : n and n+1 are both coprime to P(y) }.

This is the Jacobsthal-type function for the pair `{0,1}` modulo the primorial. If item 17 holds, there is **no**
covering-design freedom to exploit, and bounding the maximum gap of this one explicit pattern is the whole task.
Item 17 is a short CRT argument and should be among the first things checked, since a great deal rests on it.

**Measured slack.** `F_k(y) / ((y^2-y)/6)`:

    y         7     11     13     17     19     23     29     31     37     41     43
    F_k       5      7     11     18     25     34     43     58     88     91    103
    ratio 0.714  0.382  0.423  0.397  0.439  0.403  0.318  0.374  0.396  0.333  0.342

The requirement holds with a factor of roughly 2.5 and the ratio is flat, not climbing. So the truth of the
statement is not in doubt numerically; only the proof is missing.

### 1.1 Four interchangeable forms of Reduction A

All four are equivalent; different attacks prefer different ones.

**(a) Direct.** `F_k(y) <= (y^2-y)/6`.

**(b) Covering / hazard.** With `N(L) = #{m : m, ..., m+L-1 all blocked} = sum_g max(0, g-L)` and
`G(L) = #{gaps > L}`:

    G(L) = N(L) - N(L+1)              (proved, item 18)
    h(L) = G(L)/N(L) = 1 - N(L+1)/N(L)

and the bound follows from `N(L) <= P (1-d)^L`, equivalently `h(L) >= d` for every `L`. In the **`k`-frame this
is all that is needed.** In the adjacent frame the stronger `min_L h(L) = h(1)` was pursued; that is an
artefact of the finer grid (item 17a below) and is not required.

**(c) Second-order form.** `h(L)/d -> 1` for every `L` as the gear set grows, so `h/d` carries no margin.
The scale-free quantity is

    kappa(L) = (h(L)/d - 1)/d,     kappa(1) = 1/(1-d) exactly.

Adjacent frame: the bound needs `kappa(L) >= 1/(1-d)` for `L >= 2`. `k`-frame: it needs only `kappa(L) >= 0`.
Measured `k`-frame minimum of `kappa` settles near **0.68** (values `0.3889, 0.6167, 0.6749, 0.6824, 0.6840,
0.6805` at `y = 7..23`), so there is absolute room, not a knife edge.

**(d) Gear-recursion form.** `F(2,y) <= C * sum_{3<=q<=y} q` for a constant `C`. Since the odd primes sit
inside the odd numbers, `sum_{3<=q<=y} q <= (y^2+2y-3)/4` with **no prime counting** (note `pi(y) < y/2` is
false at `y = 3, 5, 7`), and Reduction A then follows from any proved

    C <= 2(y^2-y)/(y^2+2y-3),   which is 1.8125 at y=29, 1.85 at y=37, 1.88 at y=47, rising to 2.

So **`C <= 1.8` suffices for all `y >= 29`**, with smaller `y` checked directly. Measured `C` peaks at
**1.354** at `y = 37` and is not monotone.

---

## 2. Lines of research explored, and what each produced

In roughly the order explored. Status labels record **this programme's judgement, not a proof about the route**:
**live** (abandoned while still viable, or still viable now), **closed** (appeared to fail or to have no reach on
the evidence gathered - not shown impossible), **absorbed** (superseded by something taken to be stronger), or
**standing** (a result argued here and then relied on everywhere, so an error in it propagates widely). Two
closures were later found to have been wrong, one of them costing a working route for most of the programme, so a
"closed" label is a place to look rather than a place to stop.

**2.1 Slip algebra: periods and relative slips.** The founding question. Rather than turning the gears, use only
their periods and *relative slips* - gear 2 slips against gear 3 by 1 per 2-cycle, and so on - and ask
deterministically, with set algebra or a tree, when the `1` and `5` slots are simultaneously free. Produced the
distinction between *cycle slip* `|P - Q|` and *machine slip* `P mod q` (section 0.3), the **turn law** in
closed form, and CRT composition of sub-machines. **Standing.** The turn law underlies every later counting
result. Artefact: `research/slip_algebra.py`.

**2.2 Blocking positions, then the inverse: open slots.** The first pass asked which gear positions *block* the
`1, 5` slots. That was the wrong direction and was corrected repeatedly: **the object of study is the openings.**
The reframe was made concrete by aligning every gear so its block falls at the end of its cycle - gear 2 open at
1 blocked at 2, gear 3 open at 1-2 blocked at 3, gear 5 open at 1-4 blocked at 5 - so all slips start at offset
0, and then asking what *opens* `1` and `5` rather than what shuts them. Produced the role trichotomy
killer/shield/miss, the **tooth budget** (one shield, two killers, `q-3` misses), the **arc structure** with the
short arc centred on the shield, and the one-kill lemma. **Standing**, and the framing of everything after.
Artefacts: `research/open_runs.py`, `research/exposure.py`, `research/exposure_rotations.py`.

**2.3 Self-blocking and the square-root tower.** Each gear sits at an identifiable tooth of its own machine: the
lower tooth of gear `q` is the index of the pair containing `q`, and twin gears share it. Led to the **window
identity** `survivors(y,K) = T(6K+1) - T(y)` and the banded square-root tower. **Standing** - this is what makes
the problem finite-dimensional at each scale and defines the window `(y, y^2]`. Artefact:
`research/self_blocking.py`.

**2.4 Exposure-window relationship tables.** Built as a deliberate exercise in doing the tabular work rather than
reaching for a shortcut: for every gear, its exposure windows and their relation to every other gear and to the
`1, 5` slots. Table A gave each window and the slot it attacks; Table B found **pairwise coincidence is always
exactly 4**; Table C derived the run count. Also produced a localisation rule that was then refuted. **Closed as
a route** - pairwise data is not enough - but the tables are correct and reusable. Artefact:
`research/exposure_tables.py`.

**2.5 n-wise windows beyond pairwise.** Pairwise being insufficient, the natural next step was the full `n`-wise
lattice of window intersections against the 6-cycle. Produced the **generating polynomial** `prod (q - 2 + 2x)`,
whose `x^j` coefficient counts slots threatened by exactly `j` gears and whose evaluations reproduce every
alignment law as a single substitution. **Standing** as machinery. Artefacts: `research/nwise.py`,
`research/alignment.py`.

**2.6 Phase.** Pursued on the instinct that the answer is in the phase relationships. Produced an exact but weak
floor from the small-gear phase, and identified the one structure that would have related phases across scales -
which **dies exactly at the square root**, the same threshold as the window. **Closed**, and the reason it closes
is informative: the phase relation and the certification window fail at the same place, which is not a
coincidence. Artefact: `research/phase.py`.

**2.7 Cofactor reindexing.** Reindex by the cofactor `P/q` instead of by `q`. Blocks then correspond to gears,
the phase slope of a block is the gear's own step, and the same square-root threshold appears from both
directions - a genuine duality. **Closed** as a source of bounds; useful as a consistency check. Artefact:
`research/closed_form.py` and section 34 of `twin-prime-program.md`.

**2.8 Frequency-space and phase analysis.** Pursued on the observation that a phase relationship in the
signal-processing sense arises when signals share a frequency, that an FFT moves time to frequency, and that
beats should appear as overtones. This turned out to be exactly right structurally: the indicator of the exposed
set has an **exactly factorised spectrum** `Ehat(k) = prod_q ehat_q(k t_q mod q)`, matching a direct FFT to
`1.1e-16`. Low-order truncation localises the pattern beautifully - and is **forbidden**: the `L1` norm grows
about `2.06` per gear, so a single low-energy beat still closes a slot, and dropping it loses exactness rather
than precision. That was tested directly at the insistence that low-energy beats not be discarded, and the
insistence was correct. A full-fidelity spectral pipeline was then built with nothing discarded; it works, and
its cost is the reason it is not a shortcut. **Live**, in the specific sense of pathway 7.2 below: correlations
are spectral, `C(g) = P sum_k |Ehat(k)|^2 omega^{kg}` with `|Ehat|^2` factorising exactly, and the gap counts are
alternating sums of correlations. **Closed** as a localisation shortcut. Artefacts: `research/spectrum.py`,
`research/full_spectrum.py`.

**2.9 The constructor, and the closed-form next twin.** An openings-driven generator: from a known twin, compute
each gear's distance to its next tooth as `min((u_q - m) mod q, (-u_q - m) mod q)`, take the minimum, and jump.
Produced a **closed-form next-twin method with no walking**, verified to `k = 10^16` (192 s down to 0.081 s), plus
bulk gear generation, plus an explicit closed form `J(m0) = sum_J prod (1 - E(m0+i))` for the distance itself.
**Standing, and no prior form of it was found.** It is also the sharpest statement of the gap: the
constructor computes its own output but does not bound it. Artefacts: `research/twin_constructor.py`,
`research/jump_distance.py`, `research/closed_form.py`, `research/navigate.py`.

**2.10 Validators: gcd, order, unit group.** Three exact characterisations of twinhood - `gcd(36m^2 - 1,
primorial) = 1`, `ord_q(6m) <= 2`, and two units of `U(P(y))` differing by 2 within the first `y^2` residues.
**Standing** as identities. **Closed** as routes, since none localises. Also the origin of the `36 = 6^2`
question: it is the block period squared.

**2.11 Overlap and exact-group bounds.** Bound the exposed count by tracking exact overlap groups rather than
inclusion-exclusion. Produced a working scanner and exact counts. **Absorbed** into the generating polynomial and
the later decomposition. Artefact: `research/overlap_bound.py`.

**2.12 Bounding `F(2,y)`: the covering and hazard route.** The main sustained line. Reformulated the maximum gap
as a covering count `N(L)`, then as a hazard `h(L) = G(L)/N(L)`, with the bound equivalent to `h(L) >= d`.
Produced `G(L) = N(L) - N(L+1)`, the proof that `h(1) = d/(1-d)` is free because gear 3 annihilates the
third-order term, four proved cases `h(1), h(3), h(6), h(9)`, closed forms for the gap counts, and the
`kappa` normalisation. **Live** - this is Reduction A form (b)/(c). Four sub-routes inside it are closed;
see section 6. Artefacts: `research/hazard.py`, `research/covering_bound.py`,
`research/covering_decomposition.py`, `research/closed_hazard.py`, `rust2/src/bin/maxgap.rs`,
`rust2/src/bin/coverbound.rs`.

**2.13 Forbidden configurations and the constrained word.** Read the gap sequence as a word over an alphabet and
ask what factors are forbidden. Produced the **minimal size law** - gear `q` can be forced only in a
configuration of at least `(q+1)/2` positions, attained - with its exposure form, the minimal-span growth
`~1.9q`, and the fact that gears from 29 to 47 add no new minimal forbidden configuration. **Closed** as an
automaton route, for two independent reasons (item 9 of section 6). **Standing** as laws. Artefacts:
`research/minimal_forbidden.py`, `research/gap_automaton.py`.

**2.14 The gear-set recursion.** Recurse on the gear set rather than on the run length. Produced the exact
**merge transform** (adding gear `q` is `q` copies of the old pattern, each thinned at a different phase, and
every exposed point is deleted in exactly 2 of the `q` laps), the **chain condition** determining `F(M+q)` from
the old gap word alone, the **deletion-spacing lemma**, and the **saturation theorem**. **Live** - this is
Reduction A form (d), and pathway 7.1. Artefacts: `research/gear_recursion.py`, `rust2/src/bin/holegap.rs`.

**2.15 Infinite gears: the gear at infinity.** A conceptual line, and the source of several results. The machine
is fully constructed up to infinity; the gears are integers, so after the primorial every gear is simultaneously
back where it began, and infinite rotation resets the machine to its state at 0. At 0 every gear divides 0, so
every gear shields - the complete-shield position - and twins sit immediately beside it. No gear can outpace the
base 6-cycle. Therefore the configuration that produces twins recurs, and to deny infinitely many twins is to
claim the machine stops presenting a configuration it presents next to 0.
Four of its six steps are **theorems**: the `+-1` walk, slot 0 always exposed, exact periodicity with symmetry
about 0, and the gear model itself. The `+-1` walk law and both blocking laws were *found* by taking this frame
literally. Where it does not close is **localisation, not existence**: the recurrence period is the primorial,
about `e^y`, while the gears only certify on `(y, y^2]`. So the frame gives that the configuration recurs
forever, at the fastest rate the machine allows, but not that it recurs inside the window where it can be
certified - and that last step is exactly Reduction A. **Live as a frame**, and it states precisely what a proof
still owes. Artefact: `docs/gear-at-infinity.md`.

**2.16 Proof by contradiction on the mechanism.** A shape stipulated as acceptable and worth keeping open: *this
is the mechanism that generates twin primes; to stop it, condition X would have to hold; X cannot hold because
of Y.* The mechanism half exists and is verified - the constructor of 2.9 plus the turn law of 2.1. The
obstruction half is exactly what is missing: no condition X has been found whose impossibility is provable. The
nearest approach is the **repulsion form** (item 23), which says the mechanism fails only if exposed positions
cease to repel, and the repulsion is measurable but not yet bounded. **Live**, and probably the framing most
worth carrying forward.

**2.17 Two coordinate systems and the CRT collapse.** The final line. Established that the adjacent frame and
the pair-index frame are the same machine scaled by 3, and then that the apparent covering-design freedom does
not exist at all - every offset vector is one translation (item 17). **Standing**, and it simplified the target
substantially while withdrawing an earlier framing. This is the line section 1 reports.

A standing constraint on all of the above, which shaped what was and was not attempted, and which is stated in
full in the Method section: statistical and probabilistic reasoning was excluded as a *substitute* for
mechanism, on the grounds that twin primes are placed by exact modular residues and their interplay, and a
density estimate cannot see an exact placement. It was later admitted as a *validator* - legitimate to check a
mechanical construction already built and tested, not to pre-empt one. The lines that produced standing results
(2.1, 2.2, 2.3, 2.5, 2.8, 2.9, 2.13, 2.14) are exactly the ones pursued mechanically first and measured
afterwards; the lines that produced refutations are the ones where a pattern was generalised before it was
built.

---

## 3. Findings offered as established - each needs checking

Either argued to a proof here, or verified exhaustively at the stated scale. Several items are exhaustive only
inside a bounded search box, and the bound is given in each case. Item numbers are referenced from other sections.

**Machine mechanics.**

1. **Teeth.** Gear `q` threatens `k = +- 6^{-1} mod q`. With `tooth(q) = (q+1)/6` for `q = 5 mod 6` and
   `(q-1)/6` for `q = 1 mod 6`, the threatened set is `{tooth(q), q - tooth(q)}`; `6^{-1}` equals `tooth(q)` in
   the first case and `q - tooth(q)` in the second. The `+-` symmetry follows from `6 * 6^{-1} = 1`.
2. **The ±1 walk.** Every gear and every sub-machine steps the 6-cycle by exactly `+-1` per rotation:
   successive multiples of `q` step by `q mod 6`, every prime gear is `1` or `5 mod 6`, and the units mod 6 are
   closed under multiplication so composites inherit it. Verified for every sub-machine of up to four gears
   drawn from 5..59: `P mod 6` is always 1 or 5.
3. **Slot 0 is always exposed** - the complete-shield position - since `6^{-1}` is never `0`.
4. **Periodicity and symmetry.** The exposed set is a union of complete residue classes mod `P`, and the threat
   set is symmetric under `m -> -m`, so the pattern is symmetric about 0.
5. **Self-blocking law.** The lower tooth of gear `q` is the index of the pair containing `q`; twin gears share
   their lower tooth. Zero failures to `q = 2000`.
6. **Window identity.** `survivors(y, K) = T(6K+1) - T(y)` for `6K+1 <= y^2`, exact; verified `y = 11` to
   `1009`. This is what makes the certified window `(y, y^2]` precise.
7. **Tooth budget.** Per rotation each gear contributes exactly one shield, two killers and `q-3` misses.
   Universal.
8. **Minimal size law.** Gear `q` can be *forced* to block one of a set `S` only when `|S| >= (q+1)/2`, and the
   bound is attained - take `S` at residues `0, 2, 4, ..., q-1`, whose dominoes `{q-1,0}, {1,2}, ...` tile `Z_q`
   with one overlap; integer positions with those residues and all `= 0 mod 3` exist by CRT. Exhaustive to
   `q = 19`; the construction checked for all 45 odd primes below 200. **Exposure form: any `(q-1)/2` positions
   can be simultaneously exposed to gear `q`, whatever their spacing.** Gear 3's and gear 5's blocking laws are
   its first two cases. Minimal span with positions restricted to multiples of 3 grows like `1.9q`, by exact
   bitmask dynamic programme.
9. **Large gears force nothing new.** Within a search box of word length 16 and letters to 6, gears 29 through
   47 contribute zero minimal forbidden configurations beyond those from gears `<= 23`. Not a box artefact for
   29 and 31, whose own minimal configurations fit inside the box.
10. **gcd characterisation.** `m` is a twin slot iff `gcd(36m^2 - 1, primorial(floor(sqrt(6m+1)))) = 1`. The 36
    is `6^2`, the block period squared.
11. **Generating polynomial.** `p(x) = prod (q - 2 + 2x)`; the coefficient of `x^j` counts slots threatened by
    exactly `j` gears. `p(1) = P`, `p(0) = prod(q-2) = A`, `p(1-k) = prod(q-2k)`, `p(0) - p(-1) = ` run count.
12. **Factorised spectrum.** `Ehat(k) = prod_q ehat_q(k t_q mod q)` with `t_q = (P/q)^{-1} mod q`,
    `ehat_q(0) = (q-2)/q`, `ehat_q(c != 0) = -(2/q) cos(2 pi c u_q / q)`. Agrees with a direct FFT to
    `1.1e-16`. The `L1` norm grows about `2.06` per gear, so **spectral truncation is not available** - a single
    low-energy beat still blocks a slot.

**The gear-set recursion.**

13. **Merge transform (exact).** Let `M` have period `P` and exposed set `E`. Adding gear `q` gives period `Pq`
    and `E' = { x : x mod P in E, x mod q not in {0,1} }`. Walking `x` upward walks `E` around `q` times; lap
    `l` keeps the points whose `e mod q` avoids `{-lP mod q, (1-lP) mod q}`, and that pair shifts by `-P mod q`
    per lap. Since `gcd(P,q) = 1` the shift generates, so **every phase of gear `q` occurs in exactly one lap,
    and every exposed point of `M` is deleted in exactly 2 of the `q` laps.** Verified against direct
    construction on full gap *histograms*, not merely maxima.
14. **Chain condition (exact).** A new gap merges `k+1` consecutive old gaps when the `k` exposed points
    between them are deleted in one lap; those points lie in `{phi, phi+1} mod q`, so the partial sums of the
    interior gaps must all lie in `{0,1} mod q` or all in `{0,-1} mod q`. Since every point is deleted somewhere,
    **`F(M+q)` is determined by the old gap word and `q` alone.** Verified exact in 15 cases (gear sets to 19,
    added gears to 31).
15. **Deletion-spacing lemma (proved, tight).** Consecutive deleted points within a lap are at least `q-1`
    apart. *Proof:* they differ by `0` or `+-1 mod q`, and gaps are at least 3, so `delta = 0 mod q` gives
    `delta >= q`, `delta = 1 mod q` gives `delta >= q+1`, and `delta = -1 mod q` gives `delta >= q-1`. Attained
    at `q = 13` and `q = 19`.
16. **Saturation theorem (proved).** If `q - 1 > F(M)` then `F(M+q) = F2(M)`, where `F2` is the largest sum of
    two adjacent old gaps - because no interior gap can reach `q-1`, so only `k=1` chains exist. Checked over 48
    pairs, zero violations. Consequence: **above the threshold the increment is independent of `q`** (gears to 7
    plus any of 11..53 all give `F = 21`).

**The CRT collapse - the most important structural fact.**

17. **Every configuration is one translate.** The offset vector `(o_q)_{q<=y}` is, by CRT, exactly one residue
    `c mod P` with `c = o_q mod q`; gear `q` then blocks `i = c` or `c+1 mod q`, so the uncovered set is
    `{ i : i-c, i-c-1 both coprime to P }` - a translate of the single pattern `{ n : n, n+1 both coprime to P }`.
    The `P` offset vectors are precisely the `P` translations of one pattern. Hence `F(2,y) = 1 + ` the maximum
    gap of that pattern. **Verified:** the all-offsets-1 configuration (where `i` is uncovered exactly when
    `i-1` and `i-2` are both free of prime factors `<= y`) attains `F(2,y)` exactly at
    `y = 7, 11, 13, 17, 19, 23, 29` - ratio `1.000` in all seven cases, the last by segmented sieve over all
    `3.2 * 10^9` positions.
    - 17a. **Corollary.** The adjacent-frame `L = 1` has no `k`-space counterpart, so `h(1) = d/(1-d)` - the
      "free" case the covering route was built on - is a grid artefact. In `k`-space the minimum of `h/d` sits at
      `L = 2`, and `min_L h(L) = h(1)` is *stronger than the conjecture needs*.
    - 17b. **Corollary.** The exhaustive offset searches (`maxgap.rs`, `coverbound.rs`, `holegap.rs`) are sound
      but search a space isomorphic to `Z_P`: they explore translations, not designs. "Extremal configurations
      are efficient" is automatic - they are all the same configuration.

**Exact identities in the hazard formulation.**

18. `G(L) = N(L) - N(L+1)`, so `h(L) = 1 - N(L+1)/N(L)`.
19. `rho(1) = N(2)/N(1) = (1-2d)/(1-d)` and `h(1) = d/(1-d)`, **proved**: the third-order term is
    `prod (1 - 3/q)`, which vanishes because gear 3 contributes `1 - 3/3 = 0`. That is what makes `h(1)` free.
20. **Factorisation law.** With `w(S) = |{s-1, s : s in S}|` over the integers, `|W_q(S)| = w(S)` for every
    `q > span(S) + 1`. The threshold is `span+1`, not `span`: `min(S)-1` and `max(S)` differ by `span+1` and
    collide exactly at `q = span+1`. Hence
    `N(L) = sum_j c_j(L) prod_{q>L}(q-j)` with `c_j(L)` **independent of `y`**; verified by reassembly at
    `y = 13, 19, 23, 31`. Validity needs `y >= L+1`.
21. **The per-`j` recipe does scale.** `c_j(L)` is a sum over `2^L` subsets, but the head gears annihilate
    nearly all of them and depth-first pruning on the first fully covered gear visits only survivors:
    `2, 4, 10, 19, 61, 181, 289, 721, 2548` contributing terms at `L = 1, 3, 6, 9, 15, 21, 24, 30, 39` against
    `2^39 = 5.5e11`. Every visited subset contributes.
22. **Second-order expansion of `kappa`.** With `psi(delta) = v(delta)/d^2` the normalised pair weight,

        kappa(L) = L - sum_{delta <= L} psi(delta) + (small)
        psi(delta) = 3C * prod_{q | delta, q>=5} (q-2)/(q-4) * prod_{q | delta^2-1, q>=5} (q-3)/(q-4)
        C = prod_{q>=5} (1 - 4/(q-2)^2) = 0.396880415,    3C = 1.190641246

    Only `delta = 0 mod 3` contributes, because gear 3 divides one of `delta-1, delta, delta+1` and the two
    off-cases give the factor `1 - 3/3 = 0` - **the gear-3 law emerging term by term rather than imposed.**
    `psi(3) = 3C` exactly. Checked against measured `kappa` at `y = 100003` for thirteen `L`, agreeing to `0.03`
    at `L=3` rising to `0.42` at `L=63`. **`mean psi -> 3` exactly** (`2.6926, 2.9320, 2.9858, 2.9980, 2.99976`
    at `L = 63 .. 300000`), so `kappa = L - sum psi` is a bounded difference of two quantities each growing like
    `L`. Minimum of the leading-order `kappa` over all block starts to `L = 5 * 10^6` is **1.6343 at `L = 6`**,
    never below 1.
23. **Repulsion form.** `v(delta) = P(0 and delta both exposed)`, so `kappa(L) >= 1` says exactly: conditioning
    on an exposed position at 0 reduces the expected count of exposed positions in `(0, L]` by at least `d` -
    one position's worth of density. `v(1) = v(2) = 0` outright, since gear 3 blocks one of any two positions
    less than 3 apart.

**Proved special cases.** `h(1), h(3), h(6), h(9)` in the adjacent frame. Also `F_h = 0 mod 3` for all thirteen
known values, and exposed runs have length at most 2 in `k`-space (so the pattern is exactly isolated points and
dominoes, counts `prod(q-4)` and `prod(q-2) - 2 prod(q-4)`).

**New computed values produced here.** `F(2,37) = 264`, `F(2,41) = 273`, `F(2,43) = 309`; `F(2,29) = 129` and
`F(2,31) = 174` in halved coordinates; and `F2(2,y) = 21, 33, 48, 75, 93, 117, 165, 204, 270` for
`y = 7, 11, 13, 17, 19, 23, 29, 31, 37`. `F2_adjacent = 3 F2_k` verified for six gear sets.

**A closed-form next-twin method, which did not previously exist.** Per gear the distance to its next tooth is
`min((u_q - m) mod q, (-u_q - m) mod q)`; taking the minimum over gears gives the bite distance, and iterating
gives the next twin without walking. Verified to `k = 10^16` (192 s to 0.081 s against the naive walk). Also an
explicit closed form `J(m0) = sum_{J>=1} prod_{i=1..J} (1 - E(m0+i))` with
`E(m) = prod_q (1 - [q prime][q | 36m^2 - 1])`, verified at `m0 = 1, 5, 20, 50, 100, 200, 400, 1000`.

---

## 3a. What is formalised in Lean, and what is not

`proofs/BlockedSlots.lean` - 31 theorems, no `sorry` - formalises the reduction and the machinery around it
against mathlib.

> **Machine-checked.** `lake build` completes successfully, 966 jobs, warnings only - deprecated `push_neg` and
> unused-variable lints, no errors. Toolchain `leanprover/lean4:v4.34.0-rc1`, mathlib at
> `4819386ff6a6681a4321877b165ffe7a7d115fa6`.
>
> Checked further with `#print axioms`: the key results depend only on the three standard Lean axioms -
> `propext`, `Classical.choice`, `Quot.sound`. No `sorryAx`, no custom `axiom`, no `native_decide`. The file
> contains zero occurrences of `sorry`, `native_decide`, `axiom` and `@[implemented_by]`. `AxiomCheck.lean` in the
> same directory reproduces the audit.
>
> So this section is **not** in the same evidentiary category as the rest of the document. Section 3's findings are
> informal arguments plus computation; the statements below are verified by Lean's kernel, and what remains open
> about them is only whether they say what is intended - the definitions, not the proofs.

**What is formalised.**

* **The blocking relation** and its equivalence to the lazily-advanced cursor form the Python uses
  (`blocked_iff_cursor`) - so the algorithm and the arithmetic definition are the same object.
* **Soundness and completeness of a single opening**: an unblocked slot past `y` with `y` at least the square root
  is prime (`prime_of_not_blocked`), and conversely a prime past `y` is unblocked (`not_blocked_of_prime`).
* **The next-gap operation is well defined.** `exists_gapOK` proves the search terminates, `nextGap` is then
  `Nat.find` of that, and `nextGap_spec` with `no_prime_between` says it lands on the next prime and skips nothing.
  This is the formal counterpart of the constructor of section 2.9.
* **The twin version**, with two cursors per divisor: `twin_of_not_blockedTwin`, `not_blockedTwin_of_twin`,
  `twinGap_spec`, `no_twin_between`, plus the arithmetic lemma `sqrt_add_two_lt` that the induction needs.
* **The reduction, in both directions.** `Survivor y m` says no prime `q <= y` divides `m` or `m + 2`;
  `survivor_iff_twin` says that inside the certified window a survivor *is* a twin pair; and

      twins_infinite_iff_survivor_in_window :
        {p | p.Prime ∧ (p+2).Prime}.Infinite ↔
          ∀ N, ∃ y, N ≤ y ∧ ∃ m, y < m ∧ m + 2 ≤ y * y ∧ Survivor y m

  is an **iff**, so the target of this programme is equivalent to the conjecture rather than merely sufficient for
  it - and that equivalence is machine-checked. `survivor_in_window_of_gap_bound` then converts a gap bound into
  the window statement, which is the formal version of Reduction A. **What this settles:** the reduction is not
  where an error can be hiding. Whatever is wrong in this programme is downstream of it, in the informal work of
  sections 3 and 5.
* **The centred form.** Running the rule at the midpoint `c` removes the base entirely: `c` is blocked by `q`
  exactly when `q | c^2 - 1`, so the twin pattern is one fixed nested family. `twin_of_centreSurvivor` and
  `centreSurvivor_iff_twin`.
* **The contradiction shape of argument 7, formalised.** `covering_of_not_infinite` states what would have to be
  true for the twins to run out: beyond some `N`, *every* `c` has a prime `q <= sqrt(c+1)` dividing `c^2 - 1`. That
  is the condition X of section 2.16, written down precisely. What is missing is its impossibility.
* **Why constructed witnesses are never where they are needed.** `centreSurvivor_factorial` and
  `exists_centreSurvivor`: survivors are easy to produce - `y!` is one - and uselessly large. This formalises why
  the CRT-construction route cannot reach inside the window.
* **Gear 3 forces the midpoint**: `six_dvd_succ_of_survivor`, the formal version of the law that makes every gap a
  multiple of 3.
* **What counting can do, exactly**: `card_blocked_by_le` gives `L/q + 2` as the number of slots one divisor blocks
  in a run of `L`. Only the inequality is a theorem. *"No counting argument can succeed"* is a claim about proof
  strategies and is **not** formalised - section 5.1 is evidence for it, not a proof of it.
* **Lockstep**: `survivor_step` - advancing the divisor bound to the next prime `y'` can only destroy the survivor
  whose own member is `y'`, so the removal side of the accounting leaks nothing.

**What is not formalised.** Everything in section 3 from item 8 onward: the minimal size law, the factorisation
law, the merge transform, the chain condition, the deletion-spacing lemma, the saturation theorem, the CRT collapse
of item 17, `G(L) = N(L) - N(L+1)`, `h(1) = d/(1-d)`, and the `kappa` expansion. All of those are informal
arguments in the markdown plus computational verification. Several are short and would formalise easily.

**Suggested formalisation order**, by ratio of value to effort:

1. **item 17, the CRT collapse** - a few lines, and section 1's simplest form rests entirely on it;
2. **the deletion-spacing lemma and the saturation theorem** (items 15, 16) - both are short case analyses on
   `delta = 0, +-1 mod q` with `delta >= 3`, and the saturation theorem is the only clean structural theorem about
   the recursion;
3. **`h(1) = d/(1-d)`** (item 19) - it turns on `prod (1 - 3/q) = 0` because gear 3 contributes a zero factor, which
   is a one-line observation worth pinning down formally since a whole route was built on it;
4. **the minimal size law** (item 8) - the counting bound is immediate; the attaining construction needs CRT.

**Building it.** `proofs/.lake` is **already populated on this machine** - mathlib source, its dependencies, and
the `.olean` cache - so `lake build` from `proofs/` should be near-instant and needs no network. The tree is
several GB and is gitignored (`.gitignore:85`), so it exists locally but is not in the repository.

From a clean checkout the sequence is: `lake update`, then `lake exe cache get` to pull mathlib's `.olean` cache
rather than compiling it from source, then `lake build`. The cache download is roughly 8,700 files. Then
`lake env lean AxiomCheck.lean` prints the axiom dependencies.

Two practical notes. `lake` may not be on `PATH`; it is at `~/.elan/bin/lake.exe`. And the pinned toolchain
`v4.34.0-rc1` is installed alongside `v4.33.0` - elan will select the pinned one inside `proofs/`.

---

## 4. The computational anchor

Every structural claim in section 3 was computed two ways wherever two ways were
reachable, in Python (`research/*.py`) and Rust (`rust2/src/bin/*.rs`), and the independent routes agree
exactly: at `y = 23` the hazard table derived from the pattern matches the Rust enumeration of all 111,546,435
offset vectors to four decimals; at `y = 29` the hole-covering search returns `F2 = 165` against `3 * 55 = 165`
from the `k`-frame pattern, the pattern route being at its memory limit there; at `y = 29` a segmented sieve
over all `3.2 * 10^9` positions gives the single-pattern maximum gap as `129` against `F(2,29) = 129` from the
offset search, confirming the CRT collapse of item 17; the merge transform reproduces not just maxima but full
gap *histograms* against direct construction for four gear-set extensions; `chain_max` reproduces `F(M+q)`
exactly in 15 cases spanning gear sets to 19 and added gears to 31; the factorised spectrum agrees with a
direct FFT to `1.1e-16`; and `holegap.rs` reproduces `F2` exactly at all seven `y` where the pattern could be
built independently. Scales reached: `kappa` and the `psi` expansion to `y = 100003` (9592 gears) at 80-digit
precision, chosen because the integer form needs products with tens of thousands of digits while float64 loses
about eight digits to cancellation (`c_0` reaches `10^8` at `L = 24` while the answer is of order 1); the
leading-order `kappa` inequality over all 1.67 million block starts to `L = 5 * 10^6` by
smallest-prime-factor sieve; `F(2,y)` by exhaustive covering search to `y = 47`; the closed-form next-twin
method to `k = 10^16`; the minimal size law exhaustively to `q = 19` with its construction checked for all 45
odd primes below 200. The refutations in section 6 are the substantive output of the computation, not
incidental: each is a counterexample or a measured divergence at a stated scale, and several arrived as the
next data point after the claim looked settled.

---

## 5. The near-proof bottleneck

**Where it stalls, on this programme's reading.** Every route tried here reduces to a constant, and the constant
resisted, apparently because *every elementary bound attempted compares how much the gears can cover against how
much needs covering* - and capacity is abundant, not scarce. The arithmetic in 5.1 to 5.3 is exact and should
survive checking; the claim that it explains *all* the failures is an interpretation and may be too broad.

**5.1 The one-scale capacity barrier.** Gear `q` covers about `2L/q` positions of a run of length `L`, so a
capacity contradiction needs `sum 2/q < 1`:

    y                5      7     11     13     23     47    101
    sum 2/q (q>=5)  0.400  0.686  0.868  1.021  1.331  1.657  1.959
    sum 2/q (q>=3)  1.067  1.352  1.534  1.688  1.998  2.323  2.625
    overlap          6.7%  35.2%  53.4%  68.8%  99.8% 132.3% 162.5%

Capacity arguments work **only up to `y = 11`, and never again**. By `y = 47` the gears carry 132% more
covering capacity than a run needs.

**5.2 The two-scale repair also fails, and fails worse.** Split the gears at `z`. Those `<= z` can cover a run
of `F(z)-1` outright, so in a window of length `L` they leave at least `L/F(z) - 1` uncovered, and the gears in
`(z,y]` cover at most `2(L/q + 1)` each. A bound on `L` follows exactly when
`F(z) * 2 * sum_{z<q<=y} 1/q < 1`. Minimising over `z` (excluding the degenerate `z=y`):

    y             13     17     19     23     29     37     47
    best z         3      3      3      3      3      3      3
    product    3.064  3.417  3.733  3.994  4.201  4.556  4.970

The best threshold is always `z = 3`, where the condition reduces to `sum_{5<=q<=y} 1/q < 1/6` against an
actual `0.51` already at `y = 13`. Short by 3 to 5, and since `sum 1/q` diverges **the shortfall grows without
bound.**

**5.3 What the shortfall measures: clustering.** The gap between capacity and truth is the maximum gap against
the mean gap, `F(z) d_z`:

    z            3     5     7    11    13    17    19    23    29    31    37    43    47
    1/d_z     3.00  5.00  7.00  8.56 10.11 11.46 12.81 14.03 15.07 16.11 17.03 18.77 19.61
    F(z) d_z  1.00  1.20  2.14  2.45  3.26  4.71  5.86  7.27  8.56 10.80 15.51 16.46 18.06

Equal to the mean gap at `z = 3`, eighteen times it at `z = 47`, and growing like `y^2 / log^3 y` since
`F ~ C y^2/log y` against `1/d ~ log^2 y`. **The exposed set is far from uniform, and every bound that treats
it as uniform loses exactly this factor.** Note (item 17) that the difficulty is *not* a combinatorial fitting
or design question - there is no design freedom - so the clustering is the whole of it.

**5.4 The gear-recursion route stalls on an aggregate.** `F(M+q) - F(M)` splits exactly into
`(F2(M) - F(M)) + (F(M+q) - F2(M))`, both now computable:

    add q         11     13     17     19     23     29     31     37     41     43
    F(M)          15     21     33     54     75    102    129    174    264    273
    F2(M)         21     33     48     75     93    117    165    204    270      -
    increment      6     12     21     21     27     27     45     90      9     36
    F2 - F         6     12     15     21     18     15     36     30      6      -
    excess         0      0      6      0      9     12      9     60      3      -
    incr/q     0.545  0.923  1.235  1.105  1.174  0.931  1.452  2.432  0.220  0.837

**The gear-37 step reaches `2.432 q`, above the `1.8` the chain needs, so no per-step bound
`increment <= 1.8 q` is true and summing per-step bounds cannot give `C`.** Cumulative `C` is nevertheless
`1.354`, because the neighbours are `1.452` and `0.220`. The two pieces trade off against each other rather
than each staying small - which is why the aggregate is better behaved than either, and why the constant must
be argued as a property of the running total or the average increment.

**5.5 The chain length `k` is not boundable from gap structure.** The deleted points are consecutive exposed
points, so the span holds exactly `k` of them and the interior gaps are genuine gaps of `M`:
`(k-1)(q-1) <= sum h_j <= (k-1) F(M)`. These are compatible for **every** `k` as soon as `F(M) >= q-1`, which
is the regime that matters. Above the threshold they are incompatible for `k >= 2`, which is exactly the
saturation theorem - so the argument has that reach and no more. Bounding `k` needs the arithmetic of which
gaps fall within 1 of a multiple of `q`. Empirically `k <= 3` always: for gears to 19 with `q = 23`, runs of one
qualifying interior gap occur 11808 times, of two only 62 times, of three never.

**5.6 The remaining inequality, at its cleanest.** From item 22, the bound reduces at leading order to

    sum_{delta <= L, 3 | delta} psi(delta)  <=  L - 1     for every L >= 3,

verified over all 1.67 million block starts to `L = 5 * 10^6`, minimum margin `1.6343` at `L = 6`. What is
missing is (i) a rigorous error term for the second-order expansion, currently only measured, and (ii) control
of the average of `psi` with the error in the right direction. `mean psi -> 3` is what makes it delicate: the
inequality is a statement about the *second-order* term of that average.

---

## 6. Claims that appear to fail, with the evidence against each

Each of these was believed, then contradicted by a counterexample or a measured divergence, and each looked right
on a small sample first. The evidence is given so it can be reproduced and so the *inference* can be judged
separately from the data - a claim may fail as stated while a repaired version survives, and in at least one case
below the stated reason for failure was itself later found to be wrong. Treat the numbers as reproducible and the
verdicts as provisional.

1. **mex law** (first exposed slot = mex of `{u_q} union {q-u_q}`) - held to `y = 37`, failed at 41, stalled at
   20 against the truth 87.
2. **Uniform adjacency as the failure condition** - 55 of 56 failures for `{5,7,11}` were *not* uniformly
   adjacent.
3. **Gear-3 lemma** (gear 3 implies the bound for any separations) - counterexample `{3,5,7,11,13,17}` with
   separations `(1,3,3,3,3,3)`: `N(6) = 148485 > 147584.435`, ratio `1.006102`. Verified twice, the second time
   by direct enumeration of all 255255 tuples.
4. **Monotonicity of the margin in the gear set** - false; `R_1` peaks at `1.705697` near `q = 83` then falls to
   `1.679767`.
5. **Collapsing margin** - a normalisation artefact; the exact integer differences were *growing*
   (`51555900`, `350759640`).
6. **Log-concavity of `N`** - false, fails at `L = 3`.
7. **Tail-fraction bounds** from `N(L) <= N(1) - (L-1)G(L)` - too crude from `L = 6`.
8. **The universal bound `h >= 1/(F_h - L)`** - circular; it presupposes `F_h`, so it cannot appear in a chain
   deriving `F_h`. An apparent "complete proof for `{3,5,7}`" built on it was hollow.
9. **Finite automaton over the gap word**, in both forms - the antidictionary is infinite (minimal forbidden
   words still appearing at length 16 with the count rising), and even granting an automaton its letter
   statistics count which words *can* occur, a `y`-independent quantity, while `n_j` counts how often they *do*
   and scales with `P`. Measured side by side at `y = 13`: letter 1 at `0.127` in the pattern against `0.044` in
   the automaton, and the two move in opposite directions with `y`. Weighting it needs the `n_j` themselves.
10. **Per-gear conditional marginals fall under conditioning** - they *rise*, by up to 63% above `2/q`, failing
    at 36 of 53 values of `L` at `y = 17`. So gear exhaustion is **not** a per-gear effect, and the intuitive
    reading of the step form is wrong at that level.
11. **Weak negative association** `h(L) >= prod (1 - marginal_q)` - fails narrowly: once at `y = 17` (`L = 2`,
    short by 2.6%), twice each at `y = 11` and `13`.
12. **Per-gear usefulness for the step form** - the offsets that block position `L` are jointly below average at
    covering `[0,L)` only when `L mod q >= q/4`. Exact, 4525 pairs `(q,L)`, zero exceptions.
13. **The `3q` / `6q^2` reading of the tight block starts** - predicted `51, 57, 150` as tight; at `y = 71` `51`
    and `57` rank 19th and 21st of 21, the loosest, while `30` and `45` (in neither family) rank 8th and 10th,
    and at `y = 31` `L = 150` ranks 33rd of 58. Refuted from both directions by two independent routes.
14. **`F2` is attained at a maximal gap, and `F2 - F` is bounded** - both false. At `y = 29` (`k`-frame)
    `F2 = 55` comes from the adjacent pair `(30, 25)`, two large-but-not-maximal gaps, and `F2 - F` reads
    `2, 4, 5, 7, 6, 5, 12` in `k`-units, doubling at `y = 29` after appearing to settle. What *is* true and
    strengthens: maximal gaps are strongly isolated - at `y = 29` the only flanking pair is `(2,2)`.
15. **Any argument resting only on "gaps are multiples of 3 and at least 3"** - the multiset `{3,3,15}`
    satisfies both and violates the claim.
16. **Per-step increment bounds** - see 5.4.
17. **Covering-design freedom** - there is none; see item 17. Any argument that appeals to choosing offsets
    cleverly is appealing to a translation.

---

## 7. Promising pathways not yet formalised

**7.1 The aggregate increment.** The per-step route is closed (5.4) but the cumulative `C` is well behaved
because the two pieces of each increment trade off. Formalise the trade-off: when `F2(M) - F(M)` is large the
chain excess is small and vice versa. Concretely, both pieces are governed by the same object - the largest
gaps of `M` and their neighbourhoods - and the isolation of maximal gaps (item 14 above, the true half) says a
large `F` forces small neighbours, which caps `F2 - F`; while a large `F2 - F` means two medium gaps are
adjacent, which limits how much a chain can add on top. A bound on the *sum* of the two, uniform in `q`, gives
`C` and hence Reduction A. Tools in place: `chain_max` for the second piece, `holegap.rs` for the first, and the
saturation theorem for the regime boundary.

**7.2 A uniform lower bound on `kappa`, in the `k`-frame.** The `k`-frame needs only `kappa(L) >= 0` against a
measured minimum near `0.68` - absolute room, no minimisation required. Via item 22 this is
`sum_{delta<=L, 3|delta} psi(delta) <= L - 1`, an inequality about an explicit divisor product. Two sub-problems:
a rigorous error term for the second-order expansion (the neglected triples-and-above, measured at `0.03` to
`0.42` across `L = 3..63` at `y = 100003`), and a second-order estimate for the average of `psi` in the right
direction. The repulsion form (item 23) is the mechanical statement of the same thing and may be the better
handle: `v(1) = v(2) = 0` outright gives a deficit of `2d^2`, and the question is whether the surviving
multiples of 3 over-compensate - they need `mean psi <= 3 - 3/L` and measure `3 - 104/L` at `L = 5e6`.

**7.3 A clustering-aware bound.** Section 5.3 identifies what the failed routes appear to have been unable to
see: the ratio `F d` of maximum gap to mean gap, growing like `y^2/log^3 y`. On that reading, a bound that
*uses* the clustering rather than assuming uniformity is the kind to look for. Two concrete openings. First, the exact
`N(L) = sum_j c_j(L) prod_{q>L}(q-j)` decomposition (item 20) separates the gear set from the run length
completely, with `c_j` computable to `L = 39` in 2548 terms (item 21) - so the `L`-dependence is available in
closed form and only its large-`L` behaviour is missing. Second, the CRT collapse (item 17) means the target is
the maximum gap of one explicit pattern, `{ n : n, n+1 coprime to P(y) }`, so any technique for pair-Jacobsthal
bounds applies directly; the programme's contribution would be the explicit constant, since the requirement
`F <= (y^2-y)/2` in adjacent units is a factor of `log y` weaker than what the measured `F ~ 0.68 y^2/log y`
delivers, and that margin *widens* with `y`.

---

## 8. Files

    proofs/BlockedSlots.lean          the reduction and its machinery, 31 theorems, builds clean
    proofs/AxiomCheck.lean            #print axioms audit of the seven load-bearing theorems
    proofs/lakefile.toml              mathlib dependency, pinned in lake-manifest.json
    proofs/README.md                  what the formalisation covers

    docs/status.md                    consolidated status, capacity barrier, the CRT collapse
    docs/gear-recursion.md            merge transform, chain condition, saturation, frames
    docs/forbidden-configurations.md  minimal size law, factorisation, step form, kappa
    docs/covering-bound-route.md      the hazard route and its proved cases (adjacent frame)
    docs/twin-prime-program.md        the long-form record, sections 17-37
    docs/gear-at-infinity.md          the conceptual frame and what of it is proved
    docs/ideas-from-the-session.md    leads with named falsification tests, and the closed list

    research/gear_recursion.py        merge transform, chain_max, deletion spacing
    research/closed_hazard.py         c_j(L) by pruned enumeration, density and kappa at 80 digits
    research/kappa_expansion.py       the psi closed form and the second-order expansion
    research/minimal_forbidden.py     minimal size law, minimal span, factorisation law
    research/covering_decomposition.py  N(L) decomposition, step form
    research/negative_association.py  the two refuted association claims
    research/hazard.py                gap counts in closed form
    research/jump_distance.py         the closed-form next-twin method
    rust2/src/bin/maxgap.rs           F(2,y) by pruned covering search
    rust2/src/bin/holegap.rs          F2(2,y) by the same search with a mandatory hole
    rust2/src/bin/coverbound.rs       exhaustive offset enumeration, N(L) dump

Two cautions when reading the supporting files. Lengths quoted in `covering-bound-route.md` and
`forbidden-configurations.md` are **adjacent-frame**; divide by 3 for `k`-space. The `y^2/6` in
`gear-at-infinity.md` is `k`-space; the `y^2/2` in `covering-bound-route.md` is the same requirement in adjacent
units. And those files were written as the work proceeded, so each contains claims later corrected elsewhere -
where a supporting file and this document disagree, this document is the later view, but neither is authoritative
over a fresh derivation.

---

## Closing note

The most useful thing this programme produced may not be any single result but the map: a mechanical model written
down precisely enough that its consequences can be computed, seventeen routes traced to where each one stops, and
a single remaining bound identified in four equivalent forms with the measured slack in each. The conjecture is
not proved here and no claim is made that it nearly was. What is offered is that the remaining gap is now one
statement rather than a fog, that the statement is numerically true with a factor of roughly 2.5 to spare, and
that the reasoning standing between it and a proof is the part of this work least likely to be right.

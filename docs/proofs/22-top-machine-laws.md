# 22. The top machine on its own terms

## In plain words

The route builds one machine out of the small primes and asks where it leaves a hole.  There is a
second machine, made of the primes above the cut, and until now it has only ever been described
by what it does to the first one -- the way a car that will not move might be diagnosed by
listening to the engine and guessing at a gearbox.  This file takes the second machine off the
car and turns it over in the hand.

Written on its own, on the plain integers with no sixfold folding and no small primes, the object
it works on is a pair of numbers two apart, and a prime blocks that pair when it divides either
of the two.  So each prime blocks two positions out of every however-many, and those two
positions are exactly two apart.  That single fact -- **two apart** -- is the whole machine.  A
prime never blocks a position alone: its partner block is always exactly two away, so its blocked
set is a scatter of dominoes.  Because of that, the positions a prime leaves open are one long
unbroken stretch and then one single leftover slot, and that leftover slot is the same slot for
every prime, so it is open in every machine there is: it is the pair of numbers either side of a
multiple of everything.  Because of that, a run of exactly three blocked positions between two
open ones cannot happen -- whoever blocked the middle one has to have blocked one of the two ends.
And because of that, the longest blocked stretch the machine can make is a tiling of consecutive
integers by dominoes, which is a counting exercise and not a question about how big the primes
are: the answer is twice the number of primes, minus one if that number is odd, and nothing else.
The primes may be enormous and the answer does not change.

The first machine looks completely different -- its two blocked positions sit a third of a turn
apart, not two apart -- and the last part of the file shows that this is a change of ruler and
nothing more.  One fixed relabelling of the line carries either machine onto the other exactly.
So every law that counts things, or reflects them, is the same law twice over; and every law that
measures a distance -- how long a run is, how far apart two openings are, how big the record is --
belongs to one machine only, because the relabelling stretches distances.  The measuring laws are
what this file adds, and they are the ones the first machine could never have supplied.

Nothing here is about twin primes.  These are the wheels, described alone; the clutch that
engages them with the motor comes later.

## Vocabulary

**The line.** The integers, unfolded.  No anchor `2, 3, 5`, no sixfold, no columns.

**Gear, pair, strike.** A **gear** is a prime `g` (in use, a prime in `(q, Q]`).  A **pair** is
`(n, n + 2)`, indexed by its lower member, so the pair coordinate *is* the raw line.  Gear `g`
**strikes** the pair `n` iff `g | n` or `g | n + 2`, i.e. iff

        n = 0 (mod g)   or   n = -2 (mod g) .

In pair coordinates every gear therefore has exactly **two teeth**, at `0` and at `-2`, of
**separation 2**, and leaves `g - 2` open residues.  A pair struck by no gear of the set `G` is
**open**.

**Wheel, slot.** The **wheel** of `G` is `W = prod_{g in G} g`; the open set is `W`-periodic.  A
**slot** is an open residue class of the wheel.  The wheel of the smallest three or four gears is
the top machine's own anchor: `7*11*13 = 1001`, `11*13*17 = 2431`, `13*17*19 = 4199`,
`7*11*13*17 = 17017`, `11*13*17*19 = 46189`, `13*17*19*23 = 96577`.

**Run, chain, gap, record.** A **run** is a maximal block of consecutive open pairs; a **step-2
chain** is `n, n+2, n+4, ...` all open (the members overlap: an `L`-chain is `L + 2` consecutive
integers in arithmetic progression).  A **gap** is the difference of consecutive open pairs; the
**record** `F_top(G)` is the longest block of consecutive `n` carrying **no** open pair, so
`gap = record block + 1`.  `m = |G|` throughout, and `q'` is the smallest gear.

**Letters, domino, shield, clump.** The **letters** of `g` are the two spacings between its teeth,
`2` and `g - 2`.  A **domino** is two adjacent open pairs `n, n + 1` (the quadruple
`n, n+2, n+3`); a **member-sharing pair** is `n, n + 2` both open (the triple `n, n+2, n+4`).  The
**shield** is the residue `n = -1`, whose members are `-1` and `+1`.  The **origin clump** is the
forced block of open pairs around `0`.  The **mirror** is `n -> -n - 2`, which exchanges the two
members of a pair.

**In use.** On a range `[1, N]` with gears `(q, Z]`, `Z = floor(sqrt(N))` or thereabouts, the
wheel is astronomically larger than the range; the **gear zone** is `[1, Z]`, where a gear can
strike its own home.

Classical translation.  An open pair is an `n` with neither `n` nor `n + 2` having a prime factor
in `G`; `F_top(G)` is the longest interval free of such `n`, which is the **two-class Jacobsthal
function** of the set `G`.  The counts `prod (g - 2)`, `prod (g - 3)`, `prod (g - 4)` are the
standard Hardy-Littlewood / Schemmel local factors of the patterns `(0, 2)`, `(0, 2, 4)`,
`(0, 2, 3, 5)`.

## Statement

The branch's numbering is kept.  `G` is a finite set of gears, `m = |G|`, `q' = min G`,
`W = prod g`.  Every count below is exact over a full wheel period or exact over the stated range.

### Structural

**L1 (two teeth, separation 2).**  Every gear has exactly two teeth in pair coordinates, at
`n = 0` and `n = -2 (mod g)`, and exactly `g - 2` slots.

**L2 (the arcs, and the shield).**  The slots of one gear form exactly two arcs, of lengths
`g - 3` (the residues `1 .. g - 3`) and `1` (the residue `-1`, the **shield**).  The top machine's
arc asymmetry is `(g - 3) : 1`; the bottom machine's is `(2u_g - 1) : (g - 2u_g - 1)`, about
`1 : 2`.

**L3 (the partner law).**  Every strike of a gear has a partner strike of the same gear at
distance exactly `2` -- at `n - 2` if the tooth is `0`, at `n + 2` if the tooth is `-2`.  The
struck set of a single gear is a disjoint union of dominoes `{x, x + 2}`; a strike never comes
alone.

**L4 (the forbidden gap), in the stronger form.**  If `n` and `n + 4` are open then `n + 2` is
open.  Hence a gap of exactly `4` between consecutive open pairs is impossible in any top machine.

**L5 (the wheel count).**  A wheel of pairwise coprime gears, each at least 3, has exactly
`prod_{g in G} (g - 2)` open pairs per turn.

**L6 (the always-open pair, the antipodes, the origin clump).**  `n = -1` is open for every gear
set (members `-1`, `+1`).  `n = 2` and `n = -4` are open for every gear set (members `2, 4` and
`-4, -2`), and the mirror exchanges them.  If every gear is at least `q'` then every `n` with
`-(q' - 1) <= n <= q' - 3` is open except `n = 0` and `n = -2`: the origin carries
`2(q' - 3) + 1` slots, two maximal runs of the maximal length `q' - 3` separated by
struck / shield / struck.

**L7 (the mirror).**  `n -> -n - 2` maps the open set of any gear set onto itself, and its unique
fixed point mod `W` is the shield `n = -1`.

**L8 (the symmetry group).**  The affine maps `n -> c n + b` of `Z_W` preserving the open set are
exactly the maps `n -> c(n + 1) - 1` with `c = +-1 (mod g)` for every gear `g`, a group
`(Z/2)^m`; of these exactly the two with `c = +-1 (mod W)` -- the identity and the mirror --
preserve adjacency, so the adjacency-preserving group is `Z/2`.

### Metric

**L10 (the alignment law, and the chain ceiling).**  The longest run of consecutive open pairs is
exactly `q' - 3`, whatever the other gears are; the longest step-2 chain of open pairs is exactly
`q' - 2`.  The number of starts of a run of `L` is `prod (g - 2 - L)` and of a step-2 chain of `L`
is `prod (g - 1 - L)`.

**L16 (the record is a cover).**  `F_top(G)` is exactly the largest `L` such that `[0, L)` can be
covered by choosing one phase `s_g` per gear and taking
`S_g = {x in [0, L) : (x + s_g) mod g in {0, g - 2}}`.  A gear `g > L + 1` contributes a singleton
or a domino `{x, x + 2}`; a gear `g <= L + 1` may also contribute the long letter
`{x, x + g - 2}`; a gear `g <= L` repeats.  The record block is a tiling of `L` consecutive
integers by the gears' letters -- there are no flanks.

**L17 (the parity law).**  If every gear exceeds `2m + 1` then

        F_top(G) = 2m - (m mod 2) ,

that is `2m` for an even number of gears and `2m - 1` for an odd number.  Increments therefore
alternate `+3, +1, +3, +1, ...` in that regime.  **The record of a large-gear top machine is
decided by the parity of the number of gears and by nothing else -- not by the sizes of the gears
at all.**

### Dynamical

**L12 (the chain law, `d = 2`).**  Two openings `x < y` of a machine `M` are both struck by a new
gear `g`, in some copy of `M`'s period, iff `y - x = 0, +2` or `-2 (mod g)`.

**L13 (the merge law).**  Every gap of `M + g` is a gap of `M` or a sum of consecutive gaps of `M`
whose interior openings are all struck by `g`.

**L14 (letters and alternation).**  The letters of `g` are `a = 2` and `b = g - 2`, `a + b = g`.
In a run of consecutive openings of `M` all struck by `g`, a spacing `= 0 (mod g)` keeps the
tooth, `= 2` goes from tooth `-2` to tooth `0`, `= g - 2` goes from tooth `0` to tooth `-2`, and
the nonzero letter classes strictly alternate.  The bottom machine's fuel cap `x_k - x_0 >= k a`
becomes `k <= (x_k - x_0)/2` and is **vacuous**: the top machine's grammar is cheap, and one gear
can sweep half of any stretch.

### Counting

**L9 (mirror parity of the gap census).**  Every gap length has an even count per wheel except
length `1`, which has an odd count.

**L11 (the run spectrum).**  The number of maximal runs of exactly `L` open pairs is the second
difference of `A(L) = prod (g - 2 - L)`.  Since `A` is a polynomial of degree `m` in `L`, the run
spectrum is a polynomial of degree `m - 2`: for a **three-gear wheel it is an arithmetic
progression of common difference exactly `6`, whatever the gears** (`17, 19, 23`:
`88, 82, 76, 70, 64, 58, 52, 46, 40, 34, 28, 22` at `L = 2..13`), and for a four-gear wheel a
quadratic with second difference `24` (`1162, 934, 730, 550, 394, 262`).  The top length
`L = q' - 3` is the exception, occurring exactly `prod (g - q' + 1)` times.

**L15 (dominoes and member-sharing pairs).**  Adjacent open pairs `(n, n + 1)` number
`prod (g - 4)` per wheel; open pairs sharing a member, `(n, n + 2)`, number `prod (g - 3)`.

**L18 (universal record multiplicity).**  For large-gear machines the number of record blocks per
wheel depends only on `m`, not on the gears: `18, 24, 480, 720` at `m = 3, 4, 5, 6`.  So do the
counts just below the record for even `m` (`m = 4`: `96, 24, 24` at gaps `7, 8, 9` in all three
wheels; `m = 6`: `6480, 1440, 720` at gaps `11, 12, 13` in both).

### The bridge

**L19 (the conjugacy).**  The map `n -> k = 6^{-1}(n + 1) (mod W)` carries the top machine's
open-pair set exactly onto the opening set of the **same gears** written in the bottom machine's
column coordinate (teeth `+-6^{-1}`, slot `k` the pair `(6k - 1, 6k + 1)`).  `6` is invertible mod
`W` as soon as every gear is at least `5`, so the column exists for every `n`.  Consequently:

> Every **counting or symmetry** law of a gear machine is coordinate-free and transfers between
> the two coordinates unchanged.  Every **metric** law -- arcs, runs, gaps, records, letters,
> alignment -- does not, because the conjugating map is not an isometry.  The top machine's own
> laws are exactly the metric ones, and they are the ones the bottom machine cannot supply.

In particular the bottom machine's separation `3^{-1} (mod g)`, "one third of a turn", is the top
machine's separation `2` pushed through this conjugacy: `2 * 6^{-1} = 3^{-1}`.  The only gears
unmoved by the change of coordinate are `g = 5` and `g = 7`, the two with `u_g = 1`: their letters
are `{2, 3}` and `{2, 5}` in both coordinates.  For every larger gear the column coordinate
stretches the short letter from `2` to `2u_g ~ g/3`.

### In use, on a range

**L20 (no fold).**  The open pairs are equidistributed modulo `2`, `3` and `6`: in every wheel the
six classes mod 6 differ from equality by at most `2`.  The top machine has no parity and no mod-3
structure; the bottom's sixfold belongs to the anchor `2, 3, 5`, not to a gear machine.

**L21 (the gear zone).**  On a range `[1, N]` with gears `(q, Z]`, a pair `n <= Z` is open iff
both members are `q`-smooth -- any prime factor of a member exceeding `q` is itself a gear, and it
strikes its own home.  So `[1, Z]` is almost entirely struck, and the machine's longest pair-free
run on `[1, N]` lies in the gear zone, immediately above the origin clump.  This is the exact
reverse of the wheel, where the origin is the most open place in the period.

### Two measured facts kept without mechanism

**W1.**  The counts of gap `3` and gap `5` are exactly equal in every wheel whose gears all exceed
`7`, and unequal exactly when `7` is a gear.  Residue characterisation: a gap of 3 at `n` needs a
gear with `n = -1` (its shield) and a gear with `n = -4`; a gap of 5 needs a gear with `n = -3`
and a gear with `n = -4`.  No shift and no reflection of `Z_W` carries one set to the other.
Mechanism open.

**W2.**  The range record of a fixed-gear machine climbs slowly toward the wheel record from below
and does not reach it: `21, 24, 27` (gears `7..31`) and `15, 16, 17` (gears `13..41`) at
`N = 10^5, 10^6, 10^7`, against wheel records `32` and `18`.

## Proof

### Structural

1. **L1.**  `g` strikes `n` iff `g | n` or `g | n + 2`, which in residues mod `g` is
   `r = 0` or `r = g - 2`; these are distinct for `g >= 3`, so two teeth and `g - 2` slots.
   Kernel: `TopMachine.strikesR_iff` (`3 <= g`), `TopMachine.card_open_residues`.
2. **L2.**  Removing `0` and `g - 2` from the cycle `Z_g` leaves the interval `1 .. g - 3` and the
   singleton `{g - 1}`.  Kernel: `TopMachine.open_residues`.  The shield `n = -1` has members
   `-1` and `+1`, neither divisible by any `g >= 2`: kernel
   `TopMachine.not_strikes_neg_one`.
3. **L3.**  The teeth are `0` and `-2`, so `n = 0 (mod g)` forces `n - 2 = -2 (mod g)` and
   `n = -2 (mod g)` forces `n + 2 = 0 (mod g)`.  One line, no hypothesis on `g` at all.  Kernel:
   `TopMachine.partner`; the equivalent global form "the struck set is the union of the dominoes
   `{x, x + 2}` over the multiples `x` of `g`" is `TopMachine.strikes_iff_domino`.
4. **L4.**  Suppose `n` and `n + 4` are open and some gear strikes `n + 2`.  If its tooth is `0`
   at `n + 2` then it also strikes `n` (tooth `-2`), contradicting `n` open; if its tooth is `-2`
   at `n + 2` then `n + 4 = 0 (mod g)` and it strikes `n + 4`, contradicting `n + 4` open.  So
   `n + 2` is open.  A gap of exactly 4 would require `n`, `n + 4` open with `n + 2` struck, which
   this forbids.  Kernel: `TopMachine.open_of_open_add_four` and `TopMachine.no_gap_four`, both
   with no hypothesis on the gears.
5. **L5.**  Each gear leaves `g - 2` residues (L1); the gears are pairwise coprime, so CRT makes
   the open set a product of independent residue conditions and the count multiplies.  Kernel:
   `TopMachine.wheel_count`, by induction on the gear set over the two-modulus CRT counting lemma
   `TopMachine.card_filter_crt`; no primality is assumed, only `3 <= g` and pairwise coprimality.
6. **L6.**  The shield is item 2.  For `n = 2` the members are `2` and `4`, for `n = -4` they are
   `-4` and `-2`: no gear `>= 5` divides any of them.  For the clump, a pair both of whose members
   are nonzero and smaller in absolute value than every gear is open, and the members of the pairs
   `n in [-(q' - 1), q' - 3]` are the integers of `[-(q' - 1), q' - 1]`, all of absolute value
   below `q'`; the two exceptions are the pairs containing `0` itself, `n = 0` and `n = -2`.
   Kernel: `TopMachine.shield_open`, `two_open`, `neg_four_open`, `open_of_small`, `clump_open`,
   `origin_clump`.
7. **L7.**  `g | -n - 2` iff `g | n + 2`, and `g | (-n - 2) + 2 = -n` iff `g | n`: the mirror
   exchanges the two divisibility conditions, hence preserves both "struck" and "open".  Its fixed
   points solve `-n - 2 = n`, i.e. `2n = -2`, which has the unique solution `n = -1` mod any odd
   `W`.  Kernel: `TopMachine.strikes_mirror`, `open_mirror`, `mirror_fixed_iff`.
8. **L8, sufficiency and adjacency (proved).**  An affine map preserves the struck set of one gear
   iff it permutes that gear's tooth pair `{0, -2}`, which forces `(c, b) = (1, 0)` or `(-1, -2)`
   modulo that gear -- exactly `n -> c(n + 1) - 1` with `c = +-1 (mod g)`.  Kernel: sufficiency
   for every gear set is `TopMachine.strikes_affine` / `open_affine`; per-gear necessity is
   `TopMachine.affine_teeth` (`g` odd, `g` not dividing `c`).  Adjacency: the map sends `n` and
   `n + 1` to images differing by exactly `c` (`TopMachine.affine_step`), so adjacency survives
   iff `c = +-1` in `Z_W`; the two survivors are the identity and the mirror
   (`affine_one`, `affine_neg_one`).
9. **L8, the assembly (proved, round 33).**  Going from "preserves the open set of `G`" to
   "preserves each gear's struck set" uses a `Finset`-indexed CRT: for each gear, an open `n` is
   built whose other gears miss both `n` and its image, so the image is struck iff that gear
   strikes it (`TopMachine.isolate`, needing `g >= 5`; `affine_gear`).  With per-gear necessity
   this gives `c = +-1` and `b = c - 1` modulo every gear (`TopMachine.affine_group`,
   `affine_group_form`; gears prime); every sign vector is realised (`exists_symmetry`), and the
   count of realising residues mod `W` is exactly `2^m` (`sign_count`).  The CRT lemma itself is
   `TopMachine.exists_crt` with uniqueness `crt_unique`, by `Finset` induction and Bezout.  The
   one place primality enters the whole library: for a composite gear sharing a factor with `c`
   the isolation argument gives no contradiction; the owner's gears are primes above `q`.

### Metric

10. **L10, upper bounds.**  A run of consecutive open pairs contains no strike of `q'`, so it lies
    strictly inside one of `q'`'s arcs; the long arc has `q' - 3` slots (L2), so no run of
    `q' - 2` exists.  Kernel: `TopMachine.no_long_run` (one gear, `5 <= g`) and
    `TopMachine.run_lt`.  For the step-2 chain, the pairs `n, n + 2, ..., n + 2(L - 1)` hit
    `L` distinct residues mod an odd `g` (2 is invertible), and `g`'s teeth are two of the `g`
    residues, so `L = g - 1` forces a strike: kernel `TopMachine.no_long_chain2`,
    `TopMachine.chain2_lt`.
11. **L10, attainment.**  The origin clump (L6) supplies both: `1, 2, ..., q' - 3` is a run of
    `q' - 3` open pairs, and the odd members of the clump, `-(q' - 2), -(q' - 2) + 2, ...`, are a
    step-2 chain of `q' - 2` open pairs.  Kernel: `TopMachine.run_attained`,
    `TopMachine.chain2_attained`.  So both ceilings are exact, and both are attained at the origin
    in **every** machine -- no CRT search is needed for the lower bound, unlike the bottom
    machine's alignment law (file 04), whose lower bound is a CRT alignment somewhere in the
    period.
12. **L10, the counts.**  A run of `L` starting at `n` requires, for each gear, that none of
    `n, ..., n + L - 1` be `0` or `-2 (mod g)`, i.e. that `n` avoid the `L + 2` residues
    `0, -1, ..., -(L + 1)`; so `g - 2 - L` residues per gear and `prod (g - 2 - L)` starts by CRT.
    A step-2 chain of `L` requires `n` to avoid `0, -2, -4, ..., -2L`, which are `L + 1` distinct
    residues for odd `g`; so `prod (g - 1 - L)` starts.  Positivity of the two products gives the
    two ceilings `q' - 3` and `q' - 2` again, from the counting side.
13. **L16.**  A run of `L` consecutive **struck** pairs exists in the period iff `[0, L)` is
    covered by the gears' strike sets at some phase vector, and by CRT every phase vector is
    realised somewhere in the period; so the covering formulation is an exact characterisation,
    not a relaxation.  The piece list is L1 read inside a window: gear `g`'s strikes in a window
    of length `L` are the intersections of `[0, L)` with `{x, x + 2} + g Z`, which is a singleton
    or a domino when `g > L + 1`, and can additionally be `{x, x + g - 2}` when `g <= L + 1`, and
    can repeat when `g <= L`.  This is a written proof; it is **not** in the kernel.  Validated
    against full-period scans on 15 wheels, 15 agreements and 0 disagreements
    (`F = 6, 5, 5, 5, 5, 5, 9, 8, 8, 8, 12, 10, 9, 9, 9`).
14. **L17, the upper bound (proved).**  Let every gear be odd and exceed `2m + 1`, and let
    `[n, n + L)` be all struck with `L <= 2m + 1`.  Then `L + 2 <= g` for every gear, so by L3 all
    of one gear's strikes inside the window lie in a single domino `{x, x + 2}` -- and a
    distance-2 domino **never crosses parity**.  So each gear serves the even positions of the
    window or the odd positions, never both, and covers at most two of them.  Covering the
    `ceil(L/2)` even positions needs at least `ceil(ceil(L/2)/2)` gears and the `floor(L/2)` odd
    positions at least `ceil(floor(L/2)/2)`, from **disjoint** pools, so
    `ceil(ceil(L/2)/2) + ceil(floor(L/2)/2) <= m`.  At `L = 2m` the left side is `2 ceil(m/2)`,
    which exceeds `m` exactly when `m` is odd; at `L = 2m - 1` it is `m`.  Hence
    `L + (m mod 2) <= 2m`.  Kernel: `TopMachine.parity_core` and `TopMachine.parity_upper`, in
    exactly that sharp form, with the window lemma `TopMachine.window_pair` and the parity counts
    `card_even_range`, `card_odd_range`.
15. **L17, attainment (proved, round 33).**  The construction: `ceil(m/2)` gears tile the even
    positions of `[0, 2m)` with dominoes `{0, 2}, {4, 6}, ...` and `floor(m/2)` gears tile the odd
    positions with `{1, 3}, {5, 7}, ...` (`TopMachine.anchor`); each gear's residue is chosen by
    `exists_crt` so that its teeth land on its domino.  Then every position below
    `2m - (m mod 2)` is struck (`TopMachine.parity_attained`, no size or oddness hypothesis: a
    gear always strikes both ends of its own domino).  With the upper bound this is the equality
    `TopMachine.parity_law`: `2m - (m mod 2)` is the greatest run length, as an `IsGreatest`
    statement, for odd pairwise-coprime gears above `2m + 1`.  The assignment was checked against
    the cover search first: 389 gear sets (`m = 1..11`, consecutive primes in `[5, 200)`), 0
    failures.

### Dynamical

16. **L12.**  Translating the machine and translating the gear's phase are the same thing, so the
    question is whether some shift `s` has `g` striking both `x + s` and `y + s`.  Each of the two
    strikes uses tooth `0` or tooth `-2`; the four combinations give
    `y - x = 0, +2, -2, 0 (mod g)`, and conversely each of the three residues is realised by an
    explicit shift.  Kernel: `TopMachine.chain_law`, an iff, with **no hypothesis** on `g`.  This
    is the bottom machine's chain law with `d = 2` in place of `d = 2u_g`.
17. **L13.**  If `x < y` are consecutive openings of `M + g` then both are openings of `M` (fewer
    gears strike), and any opening `z` of `M` strictly between them must be struck by `g`, since
    otherwise `z` would be an opening of `M + g` inside the gap.  Kernel:
    `TopMachine.merge_law`, no hypothesis.  The bottom machine's `MergeLaw` / `TwoTeeth`
    infrastructure did **not** specialise to `d = 2`: those files carry teeth `{u, q - u}`
    symmetric about `0`, whereas the top machine's teeth `{0, -2}` are an offset pair, so the
    reuse would have cost more than the two- and six-line direct proofs.
18. **L14.**  The letters are the two spacings between consecutive teeth of `g` on the cycle
    `Z_g`, namely `2` (from `-2` up to `0`) and `g - 2` (from `0` round to `-2`), summing to `g`;
    alternation is the statement that consecutive nonzero spacings must alternate tooth, which is
    forced because there are only two teeth.  Recorded as **measured**: 0 exceptions in 816 struck
    runs.  The fuel cap is the bottom machine's T5, `x_k - x_0 >= k a` with `a` the short letter;
    here `a = 2`, so it reads `k <= (x_k - x_0)/2` and forbids nothing that the pigeonhole does
    not already allow.

### Counting

19. **L9.**  The mirror is an involution on the wheel that reverses the cyclic order of the open
    pairs, so it permutes the gaps and preserves their lengths; the gaps of each length therefore
    pair off except for the self-mirror ones.  The number of open pairs is `prod (g - 2)`, a
    product of odd numbers, hence odd, and the mirror fixes exactly one open pair (the shield,
    item 7); a reflection of an odd cycle fixes exactly one vertex and exactly one edge, so
    exactly one gap is self-mirror.  That gap is a domino: the pair `n` with `2n = -3 (mod W)` and
    its successor are both open for every gear (their members satisfy `2n = -3`, `2(n + 2) = 1`,
    `2(n + 1) = -1`, `2(n + 3) = 3`, and no gear divides `1` or `3`), they are adjacent, and the
    mirror exchanges them.  So length `1` carries the odd count and every other length is even.
    This is the involution argument of file 03 transported to the raw line; the branch records L9
    as measured (12 of 12), and the identification of the exceptional gap is written out here.
20. **L11.**  Write `s(L) = prod (g - 2 - L)` for the number of starts of `L` consecutive open
    pairs (item 12).  A maximal run of length exactly `L` is a start of `L` whose predecessor and
    successor are struck, so its count is the second difference `s(L) - 2 s(L + 1) + s(L + 2)`.
    `s` is a polynomial of degree `m` in `L` with leading coefficient `(-1)^m`, so the spectrum is
    a polynomial of degree `m - 2`: for `m = 3` an arithmetic progression with common difference
    `-6` (the second difference of `-L^3`), for `m = 4` a quadratic with second difference `24`.
    At `L = q' - 3` the successor count `s(q' - 2)` vanishes (its factor at `g = q'` is `0`), so
    the top length occurs `s(q' - 3) = prod (g - q' + 1)` times.  0 mismatches over every `L` in
    12 wheels.
21. **L15.**  `n` and `n + 1` both open requires `n` to avoid `{0, -2, -1, -3}` mod each gear,
    four distinct residues for odd `g >= 7`, so `prod (g - 4)` by CRT.  `n` and `n + 2` both open
    requires `n` to avoid `{0, -2, -4}`, three distinct residues, so `prod (g - 3)`.  Both are
    item 12 at `L = 2`.  12 of 12 each.
22. **L18.**  Recorded as **measured**, with no mechanism: 0 exceptions in 14 large-gear wheels.
    The natural reading is that L17 makes the record a combinatorial object of `m` alone, so its
    multiplicity should be too; that is a reading, not a proof, and the sub-record counts for even
    `m` are likewise unexplained.

### The bridge

23. **L19.**  Fix a gear `g` and let `6k = n + 1 (mod g)`.  Then `g | n` iff `g | 6k - 1`, and
    `g | n + 2` iff `g | 6k + 1`.  So `g` strikes the pair `n` exactly when it blocks the column
    `k = (6k - 1, 6k + 1)`, and the teeth `{0, -2}` go to `{6^{-1}, -6^{-1}}`.  Since every gear
    is at least `7`, `gcd(6, W) = 1` and the column `k = 6^{-1}(n + 1)` exists for every `n`.
    Kernel: `TopMachine.strikes_iff_col`, `TopMachine.conjugacy`, `TopMachine.exists_column`, and
    `TopMachine.conjugacy_census` against the project's own slot members `Census.lo k = 6k - 1`,
    `Census.hi k = 6k + 1`.  Measured independently: 0 mismatches in 12 wheels, 2.2 million
    residues.
24. **What L19 says.**  A bijection of residue sets preserves everything defined by counting and
    by the symmetries that commute with it, and destroys everything defined by distance.  So the
    wheel count `prod (g - 2)`, the domino count `prod (g - 4)`, the mirror and its fixed point,
    the affine symmetry group, the even gap census, and the hit / chain / merge laws **in form**
    are one set of facts written twice; while the arcs, the letters, the alignment value, the run
    and chain ceilings, the gap spectrum and the record are two different sets of facts.  The
    bottom machine's separation "one third of a turn" is not a separate phenomenon: it is the top
    machine's separation `2` seen through this map, `2 * 6^{-1} = 3^{-1} (mod g)`, which is
    `(g -+ 1)/3`.  Conversely, this file's metric laws are precisely what the bottom machine's
    record could never have told anyone about the top machine.

### In use

25. **L20.**  Recorded as **measured**: the six classes mod 6 differ from equality by at most 2 in
    every wheel (`11, 13, 17`: `248, 247, 248, 248, 247, 247` of 1485; `17, 19, 23, 29`:
    `24098, 24096, 24096, 24098, 24099, 24098` of 144585).  Read against L19 it says the bottom
    machine's sixfold is a fact about the anchor `2, 3, 5` and not about a gear machine.
26. **L21, the smoothness half (proved).**  Let the gears be the primes in `(q, Z]` and let
    `n <= Z`.  If a member of the pair has a prime factor `p > q`, then `p <= n + 2 <= Z + 2`, so
    `p` is a gear (up to the two-element edge), and it strikes `n`.  Hence for `n <= Z` the pair
    is open iff both members are `q`-smooth, which is rare: the gears strike their own homes, and
    the gear zone is almost entirely closed.
27. **L21, the location half (measured).**  The longest pair-free run of the in-use machine lies
    inside `[1, Z]` in 18 of 18 range machines.  At `N = 10^7` the record positions are
    `161, 449, 1079, 2001, 2549, 2661` at `q = 5, 7, 11, 13, 17, 19`, against `Z = 3137`; the runs
    themselves are `3006, 2718, 2088, 1166, 618, 227`.  The origin clump survives only out to
    `q' - 3`; then the gear zone begins.

## The wheels and the ladders, and how to reproduce them

From the repository root, `uv run python research/topmachine/r1/<script>`:

| script | what it computes |
|---|---|
| `wheel.py` | the wheel and its slots, arcs, mirror, affine symmetry group, the conjugacy |
| `pairwise.py` | partner law, chain law, merge law, alternation, run spectrum, two-gear cells |
| `cover.py` | the record as an exact covering problem (L16) |
| `validate.py` | the covering formulation against a full-period scan, 15 of 15 |
| `ladder.py` | the ladders, the parity law, the record's composition |
| `range.py` | the machine on `[1, N]`, densities, the gear zone |
| `extras.py` | step-2 chains, the gap spectrum, near-wheel density |

Results (untracked) land in `research/topmachine/r1/results/`.

The twelve wheels, exact over the full period (`m = 3, 4, 5`), give the record `F_top` and its
multiplicity:

| gears | `W` | open pairs | longest run | `F_top` | multiplicity | dominoes |
|---|---|---|---|---|---|---|
| 7,11,13 | 1,001 | 495 | 4 | 6 | 2 | 189 |
| 11,13,17 | 2,431 | 1,485 | 8 | 5 | 18 | 819 |
| 13,17,19 | 4,199 | 2,805 | 10 | 5 | 18 | 1,755 |
| 17,19,23 | 7,429 | 5,355 | 14 | 5 | 18 | 3,705 |
| 19,23,29 | 12,673 | 9,639 | 16 | 5 | 18 | 7,125 |
| 23,29,31 | 20,677 | 16,443 | 20 | 5 | 18 | 12,825 |
| 7,11,13,17 | 17,017 | 7,425 | 4 | 9 | 12 | 2,457 |
| 11,13,17,19 | 46,189 | 25,245 | 8 | 8 | 24 | 12,285 |
| 13,17,19,23 | 96,577 | 58,905 | 10 | 8 | 24 | 33,345 |
| 17,19,23,29 | 215,441 | 144,585 | 14 | 8 | 24 | 92,625 |
| 7,11,13,17,19 | 323,323 | 126,225 | 4 | 12 | 48 | 36,855 |
| 11,13,17,19,23 | 1,062,347 | 530,145 | 8 | 10 | 24 | 233,415 |

Every "open pairs" entry is `prod (g - 2)`, every "longest run" is `q' - 3`, every "dominoes" is
`prod (g - 4)`.  The rows with gear 7 are the ones outside L17's large-gear regime (`7` does not
exceed `2m + 1` at `m = 3, 4, 5`), and they are exactly the rows whose record and multiplicity
break the universal pattern.

The ladders, gears added one at a time from `q'` (`>=` marks the two rungs where the covering
search was cut off, which are lower bounds and not records):

| `q'` | `F_top` by top gear `Q` |
|---|---|
| 7 | 7:1, 11:4, 13:6, 17:9, 19:12, 23:19, 29:25, 31:32, 37: >= 39 |
| 11 | 11:1, 13:4, 17:5, 19:8, 23:10, 29:16, 31:18, 37:24, 41:28, 43:34, 47: >= 37 |
| 13 | 13:1, 17:4, 19:5, 23:8, 29:9, 31:12, 37:16, 41:18, 43:24, 47:27, 53:33, 59:36 |
| 17 | 17:1, 19:4, 23:5, 29:8, 31:9, 37:12, 41:13, 43:16, 47:21, 53:24, 59:27, 61:32, 67:35 |
| 19 | 19:1, 23:4, 29:5, 31:8, 37:9, 41:12, 43:13, 47:16, 53:18, 59:21, 61:24, 67:27, 71:32, 73:35, 79:40 |
| 23 | 23:1, 29:4, 31:5, 37:8, 41:9, 43:12, 47:13, 53:16, 59:17, 61:20, 67:22, 71:25, 73:28, 79:33, 83:36, 89:40, 97:42 |

Over all 69 exact ladder steps the increment `F_top(M + g) - F_top(M)` never exceeds **7**,
against new gears of size up to 97: the analogue of the budget inequality holds here with enormous
slack, and the record grows linearly in the number of gears, about 2 per gear, not with the gear.
In the optimal covers the strikes per gear go as `2L/g`, so the **smallest** gears do the work
(at `{7..31}`, `L = 32`: `7:9, 11:6, 13:6, 17:4, 19:4, 23:4, 29:2, 31:2`, each top gear
contributing exactly one domino).

## Status

Kernel: **`proofs/TopMachine.lean`** (core, no project dependency),
**`proofs/TopMachineWheel.lean`** (wheel count and conjugacy, imports `Census`) and
**`proofs/TopMachineCrt.lean`** (the `Finset` CRT lemma, the parity law's attainment, the exact
symmetry group; round 33), namespace `TopMachine`, registered in `proofs/lakefile.toml` and
audited from `proofs/AxiomCheck.lean`.  88 declarations, **zero sorries, no `native_decide`, no `Lean.ofReduceBool`, no `decide` at all**
-- every theorem is an ordinary proof, so nothing depends on a gear set being small enough to
enumerate.  No theorem assumes primality; each carries the exact hypothesis it needs (`3 <= g`,
`5 <= g`, `g` odd, pairwise coprime), all of which the owner's construction (primes above `q`, so
`g >= 7`) satisfies.  Build: `lake build TopMachine TopMachineWheel`, green at 1001 jobs, about
7 s each, ordinary elaboration and normal memory.  Axiom audit: every declaration is
`[propext, Classical.choice, Quot.sound]` or smaller; `partner`, `strikes_iff_domino` need
`[propext]` only; only `parity_core` uses choice essentially.

**In the kernel:**

| law | Lean name (`TopMachine.*`) | hypothesis |
|---|---|---|
| L1 teeth, count `g - 2` | `strikesR_iff`, `card_open_residues` | `3 <= g` |
| L2 arcs `(g - 3, 1)`, the shield | `open_residues`, `not_strikes_neg_one` | `3 <= g`, `2 <= g` |
| L3 partner law, domino form | `partner`, `strikes_iff_domino` | none |
| L4 forbidden gap, strong form | `open_of_open_add_four`, `no_gap_four` | none |
| L5 wheel count, CRT engine | `wheel_count`, `card_filter_crt` | gears `>= 3`, pairwise coprime |
| L6 shield, antipodes, clump | `shield_open`, `two_open`, `neg_four_open`, `clump_open`, `origin_clump` | gears `>= 2`, `>= 5`, `>= q' >= 3` |
| L7 mirror, unique fixed point | `strikes_mirror`, `open_mirror`, `mirror_fixed_iff` | none |
| L8 sufficiency, per-gear necessity, adjacency | `strikes_affine`, `open_affine`, `affine_teeth`, `affine_step` | `c = +-1` mod each gear; `g` odd |
| L10 run `< q' - 2`, attained `q' - 3` | `no_long_run`, `run_lt`, `run_attained` | `5 <= q'`, `q' in G` |
| L10 chain `< q' - 1`, attained `q' - 2` | `no_long_chain2`, `chain2_lt`, `chain2_attained` | `q'` odd, `3 <= q'` |
| L12 chain law (iff) | `chain_law` | none |
| L13 merge law | `merge_law` | none |
| L17 parity law, **upper bound** | `parity_core`, `parity_upper` | gears odd and `> 2m + 1` |
| L19 conjugacy, existence, `Census` form | `strikes_iff_col`, `conjugacy`, `exists_column`, `conjugacy_census` | `6k = n + 1` mod each gear |

**Written proof, not in the kernel:** L16 (the record as an exact cover, with its piece list);
L9 (the involution argument of file 03 transported, with the exceptional gap identified); L11 (the
second-difference identity and its degree); L15 and the run/chain start counts of L10 (one-line
CRT counts); L21's smoothness half; the reading of L19 in item 24.

**Measured, exhaustive over a stated range, not proved:**

| statement | range | exceptions |
|---|---|---|
| L3 partner law | 8 wheels, 5.6 million struck residues | **0** |
| L4 no gap of 4 | 12 wheels; 27 range machines to `N = 10^7` | **0** |
| L5, L10, L15 counts, arcs | 12 wheels, every gear, every `L` | **0** |
| L6 shield, antipodes, clump | 12 wheels | **0** |
| L7 mirror | 12 wheels, 2.2 million residues | **0** |
| L8 the group is exactly `(Z/2)^m` | brute force at `W = 1001` | 8 of 8 maps |
| L9 gap census even except length 1 | 12 wheels | **0** |
| L11 run spectrum = second difference | 12 wheels, every `L` | **0** |
| L12 chain law | 118,341 opening pairs, four ladder steps | **0** |
| L13 merge law | 34,646 gaps, the same four steps | **0** |
| L14 letters and alternation | 816 struck runs | **0** |
| L16 covering formulation vs full-period scan | 15 wheels | **0** |
| L17 attainment `F_top = 2m - (m mod 2)` | 170 cases (`m = 2..11`, `q'` in `7..199`, `q' > 2m + 1`) | **0** |
| L18 record multiplicity universal in `m` | 14 large-gear wheels | **0** |
| L19 conjugacy | 12 wheels, 2.2 million residues | **0** |
| L20 no fold (flat mod 2, 3, 6) | 12 wheels | max class deviation 2 |
| L21 record inside the gear zone | 18 range machines | **0** |
| the budget analogue `F_top(M + g) - F_top(M) <= 7` | 69 exact ladder steps, gears to 97 | **0** |

**Closed in round 33 (previously "will not close").**  Both were blocked on one lemma, the
`Finset`-indexed CRT existence statement, now `TopMachine.exists_crt` (with `crt_unique`).

* **L8, "the symmetry group is exactly `(Z/2)^m`"**: `TopMachine.affine_group` (necessity, gears
  prime and `>= 5`), `exists_symmetry` (every sign vector realised), `sign_count` (exactly `2^m`
  residues mod `W` realise a preserving map).
* **L17, attainment**: `TopMachine.parity_attained` (the explicit tiling assignment `anchor`),
  and the equality `TopMachine.parity_law` as an `IsGreatest` statement.

**Round 34 (`proofs/TopMachineWalk.lean`, 70 declarations)** added the walk laws of
`research/proof/top_machine_3.md`: `TopMachine.mex_form` (the next open pair after `x` is
`x + mex{off g x, off g (x+2)}` when every gear exceeds `2m`, as `IsLeast`), `mexS_le_parity`
(the location bound `2m - (m mod 2)` from `parity_upper`), `triple_mex_form` and `triple_law`
(the run-of-three record is exactly `3m`; attainment with no size hypothesis), `no_start_gap`
(no hypothesis), `pair_corr` (the pair-correlation product, gears `>= 5`).  Build with
`TopMachineWalk` added: green at 1394 jobs; audit of those six: standard axioms; zero sorries.

Build: `lake build TopMachine TopMachineWheel TopMachineCrt`, green at 1392 jobs; manager audit
of `exists_crt`, `crt_unique`, `parity_attained`, `parity_law`, `affine_group`,
`exists_symmetry`, `sign_count`: `[propext, Classical.choice, Quot.sound]` each; zero sorries,
no `native_decide`, no `decide`.

**Refuted, and recorded as such.**  Three pre-registered predictions of the branch failed: "the
record is made at the top gears" (the *smallest* gears do the work, `2L/g` strikes each); "the
first stretch above 0 is the most open" (for the in-use machine the gear zone above the origin
clump is the *least* open stretch and carries the record); and "the top machine is non-periodic on
a range" (for a *fixed* gear set the density is flat to four or five figures on a range `10^4`
times below the wheel, so non-uniformity belongs to the in-use machine, whose gear set grows with
the range, and not to a gear machine as such).

**Not established, and said so.**  W1, the exact equality of the gap-3 and gap-5 counts, has a
residue characterisation and no mechanism, and is not induced by any shift or reflection of `Z_W`
(all `W` shifts and all `W` reflections tested at three wheels).  L18 has no proof.  The two
`>=` ladder rungs are lower bounds.

## Prior art, and what is new

**Prior-art status: CHECKED 2026-09-06** (harvester, five parallel literature sweeps; full
per-law verdicts in `research/proof/law_register.md`, where this file's L1-L21 are W1-W21).  The
line that used to stand here -- "no literature search has been run for the top machine as an
object" -- is discharged.  What the check found, in one paragraph.  **The object has a published
name**: `F_top(G) + 1` is Ziller and Morack's *paired Jacobsthal function* `j_2(W)`
(arXiv:1706.00317 Definition 2.1; `h_2` at primorials, arXiv:1706.03668 Table 1, OEIS A288815),
and their Conjecture 6, `h_2(n) < p_n^2 - p_n`, **is** the window statement this project is
after -- already in print, with a proved Goldbach and prime-pair payoff (their Theorem 4.1) and
values to `p = 73`.  Of the 21 laws here, 3 are KNOWN (L5, L15, L21), 5 are KNOWN VARIANT
(L6, L7, L10, L13, L16), 6 are NEW as far as searched (L3, L4, L8, L11, L17, L18), 6 are
STANDARD TOOL (L1, L2, L12, L14, L19, L20), and **L9 is REFUTED** -- "every gap length has an
even count except length 1" is false as soon as `N_1 = 0`; the true statement, needing no
hypothesis, is register W67.  The nearest published relatives, by law: **L16** is Ziller
arXiv:2007.01808 Definition 2.4 and Proposition 1.8 (the *restricted covering* equivalence) in
one class per prime, and the same reduction is the engine of Erdos-Rankin and of
Ford-Green-Konyagin-Maynard-Tao; **L13** is Holt and Rudd arXiv:1408.6002 Lemma 2.1 (concatenate
`p` copies of the cycle, then close adjacent gaps); **L7** is their Remark 2.2(v) (the one-class
cycle of gaps is symmetric); **L5**, **L11**'s `A(L)` and **L15** are Schemmel totients
(Schemmel 1869) and Hardy-Littlewood local factors; **L21** is the Stormer-Lehmer difference-2
smooth-pair problem (Stormer 1897; Lehmer, Illinois J. Math. **8** (1964) 57-69; OEIS
A002071/A002072).  **L17's parity defect has no analogue in print** -- the classical one-class
proposition (`q_1 > omega(n)` implies `j(n) = omega(n) + 1`, Erdos, Math. Scand. **10** (1962)
163-170) gives size-independence but no defect, because the one-class case has none -- and the
covering-systems literature (Mirsky-Newman, Znam, Hough, Balister et al.) works in a different
regime entirely and would not have found the collision law behind it (register W29, the strongest
novelty claim in the wheels).  One measured claim of this file is **refuted** by the register:
`W2`, "the range record climbs slowly toward the wheel record and does not reach it", was an
artefact of stopping at `N = 10^7` -- register W32 reaches it at 0.005% to 10.9% of the period.
`W1` is **closed** at register W24, and is a KNOWN VARIANT: the one-class coincidence
`K(2,P) = K(4,P)` is in Steven Brown, arXiv:2311.06873 / *Notes on Number Theory and Discrete
Math.* **30**(1) (2024) 81-99, for a different reason (parity annihilation, not equal class
counts).  The old paragraphs below are kept as written, and are correct as far as they go.

**Standard, and used as bookkeeping.**  The wheel count `prod (g - 2)` and the start counts
`prod (g - 2 - L)`, `prod (g - 3)`, `prod (g - 4)` are the ordinary Hardy-Littlewood / Schemmel
local factors for the patterns `(0, 2)`, `(0, 2, 4)`, `(0, 2, 3, 5)`, and the CRT argument behind
them is standard; the record marks that family KNOWN (file 04, `docs/novel/matrix-formulation.md`
for Schemmel 1869).  The alignment law's *shape* -- "the longest run of consecutive openings is
the long arc of the smallest gear, uniform in the others" -- is file 04's, and its one-class
analogue is the usual "CRT realises every relative phase somewhere in the period"; here the lower
bound needs no CRT at all, because the origin clump attains it in every machine.

**The nearest classical object.**  The longest pair-free run of a set of primes is a
**Jacobsthal-type function**: the one-class case is Jacobsthal (1961), with Iwaniec (1978) for the
bound; the object here is its two-class analogue for the pattern `(0, 2)`.  Nothing in this file
re-derives or improves an asymptotic for it.  What is here instead is **exact structure at a fixed
gear set** -- L16, L17, L18 -- which the asymptotic literature does not address, and which goes in
the opposite direction: the parity law says the two-class Jacobsthal record of a large-gear set is
`2m - (m mod 2)`, independent of the gears' sizes.

**New as far as the record goes** (all prior art not checked):

* **L3, the partner law**, and the reading of the whole machine as a **domino machine**: a strike
  is never alone, its partner is exactly two away, and every characteristic fact follows.  The
  bottom machine's corresponding distance is `2u_g ~ g/3`, so no such local law exists there.
* **L4, the forbidden gap 4**, in the strong form `n`, `n + 4` open implies `n + 2` open.  No
  bottom-machine analogue.
* **L2 / L6, the collapse of the short arc to a single slot**, the shield `n = -1`, and the
  **origin clump** of `2(q' - 3) + 1` forced slots -- two maximal runs at the ceiling, in every
  machine, with no search.
* **L10's chain ceiling** `q' - 2` for step-2 chains.  This has no bottom-machine analogue at all:
  the bottom's anchor contains 3, so a triple of open columns cannot be a prime triple.
* **L11, the run spectrum as the second difference** of `prod (g - 2 - L)`, hence a polynomial of
  degree `m - 2`; in particular an arithmetic progression of common difference 6 for every
  three-gear wheel whatever the gears.
* **L16, the record as an exact cover** by the gears' letters, with no flanks; **L17, the parity
  law**; **L18, universal record multiplicity**.
* **L19, the conjugacy**, as a statement about the two coordinates and as the branch's organising
  principle: which laws are common property and which are one machine's own.
* **L20, no fold**: the top machine has no mod-2, mod-3 or mod-6 preference, so the bottom's
  sixfold is the anchor's and not a gear machine's.
* **L21, the gear zone**, and the fact that the in-use record always sits in it.

**Not new.**  The counting corollaries and their CRT proofs; the mirror and the affine-symmetry
argument, which are file 03's for the bottom machine; the chain, merge and alternation laws, which
are file 05's, restated with `d = 2` and letters `{2, g - 2}`.

## Relationship to the conjecture

**This file describes the wheels alone.**  In the two-machine formulation (theory tree R4) the
bottom machine `{5..q}` is the motor and the top machine, the primes in `(q, Q]`, is the wheels; a
twin is a column where **both** machines are open, and that joint condition is the clutch.  Under
the owner's construction rule this file studies the top machine on its own, on the raw line, not
in the bottom's coordinate and not against the bottom's kills.  **No interpretation against the
twin conjecture is offered here, and none should be read into it.**

**Nothing here bounds a twin-free run.**  Every record in this file is the top machine's own
pair-free record, `F_top`, and by L17 it is tiny: `2m - (m mod 2)` once the gears are large --
about 2 per gear, never the size of a gear -- and over 69 exact ladder steps to gear 97 the
increment never exceeded 7.  The record that matters for the conjecture is the joint one, and the
record already on file (`period_scale.md`, R4.a) is `1.9` to `3.7` times the *sum* of the two
machines' own closed records: the twin-free stretches are made by the two machines covering each
other's leftovers, which is a clutch fact and not in this file.

**What it does supply is a rulebook with a transfer rule attached.**  L19 is the useful part for
the route: it says exactly which of the bottom machine's laws are coordinate-free (counting,
symmetry) and therefore already known for the top machine, and which are not (arcs, letters, runs,
gaps, records) and therefore had to be established here.  It also removes a temptation: the
bottom's "one third" separation and its sixfold are not two more structures to explain -- the
first is separation 2 seen through `6^{-1}`, and the second belongs to the anchor `2, 3, 5` alone
(L20).

**What enters measured.**  Nothing measured enters the kernel-checked laws (L1-L8 in part, L10,
L12, L13, L17's upper bound, L19).  L16 and the counting laws are written proofs.  L14, L18, L20,
L21's location half, W1 and W2 are measurements over stated finite ranges.  L17's attainment and
L8's group count, measured only in round 32, are kernel-checked since round 33.

## Where it is used

* As the **wheels' rulebook** for the clutch work under R4: what the top machine does on its own,
  before any question about the bottom's openings is asked.
* **L19 as the transfer rule.**  Any counting or symmetry statement proved for one machine is
  available for free in the other; any metric statement is not, and must be re-derived.  This is
  what makes files 02-05's counting content usable on the raw line and warns against importing
  their metric content.
* **L3 and L4** are what constrain the local shape of a top machine's open set: dominoes of
  strikes, no gap of 4, arcs `(g - 3, 1)`.  Any statement about the top machine's behaviour on a
  stretch has to respect them.
* **L17 and the ladders** set the scale: the top machine's own record is a counting-and-parity
  quantity of size about `2m`, so it cannot by itself account for a long twin-free stretch.  That
  is the negative fact the clutch work starts from.
* **L21** says where to look on a range: the in-use machine's record is in the gear zone `[1, Z]`,
  immediately above the origin clump, and not out in the body of the range.
* **L16** is the computational tool: the record is an exact covering problem, cheap where the
  period scan is hopeless (abandoned at seven gears, period `10^9` and rising), and validated
  against the scan 15 times out of 15.

## Source

`research/proof/top_machine_1.md` (the branch document R4.b: the twelve pre-registered
predictions and their verdicts, sections 0 and 4 for the setup and the 21 laws, section 8 for the
dead ends); `research/proof/top_machine_lean.md` (the kernel ledger R4.b.i: what is proved, the
Lean names and hypotheses, and the round-33 closure of the two holes); kernel sources
`proofs/TopMachine.lean`, `proofs/TopMachineWheel.lean` and `proofs/TopMachineCrt.lean`; scripts and outputs in
`research/topmachine/r1/`.  Framing: `research/proof/theory_tree.md` node R4 (the period-scale
formulation, the construction rule, the analogy motor / wheels / clutch).  Context, in the
bottom's coordinate and used only as context: `research/proof/period_scale.md` 3.1, 3.5, 3.11.
Inspiration, taken as form and not as content: `docs/proofs/02-tooth-rule.md`,
`03-always-open-columns.md`, `04-alignment-law.md`, `05-adding-a-gear.md`.

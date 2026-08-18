# human.md - proof-search status snapshot

Maintained as a CURRENT-STATE summary, revised in place (user direction,
2026-08-18): no round-by-round log here. History and detail live in
agents-shared.md (round appends), the workstream docs, and attempts-map.md.

## ELI5: where things stand

We study twin primes through a gear machine: every prime q >= 5 is a gear
that blocks certain slots k (the pairs 6k-1, 6k+1), and a twin prime pair
is a slot no gear blocks. The conjecture says unblocked slots never run
out. The programme has built an exact, machine-checked ledger of how the
gears spend their blocking budget, measured everything measurable, and
recorded every attempted proof route with the exact event that limits it.

Current state in one paragraph: all counting-style attacks (capacity,
moments, densities) are recorded with their limiting events - the deep one
being that the whole gap between "what the gears must do" and "what they
can do" lives in a single number per window prefix (the twin count itself,
a zeroth moment invisible to every power moment). The live route is
different in kind: it prices the growth of the machine's biggest blocked
stretch (F) step by step, and needs just two finite-flavoured statements -
"the gap spectrum is flat" and "merge chains stay short" (the fuel bound).
Both are local, censusable objects, and the current round extended their
census by two machine sizes: chains of length 4 exist (exactly 4 per
period at the 29->31 step, all one word, mirror-paired) - so the fuel
story is "k_max grows, but glacially" - the needed bound is k_max = o(ln y).
Separately, the saturated-run programme found the first length-14 run
exactly where constellation statistics predicted, validating the model
that everything else is measured against, and the thin-band reopening
closed with a one-slot self-reference gem: every twin pre-blocks the exact
center of the thinnest band above it, and beyond that one slot the thin
bands are statistically ordinary.

## The map (attempts-map.md - read before proposing anything)

Every route is filed as attempt -> yield -> limiting event, in five event
classes: I Abundance (capacity arguments die by measured surplus),
II Superdensity/Localisation (prefix pigeonholes stop by slot ~4; the
0.525 localisation exponent is an imported corpus limit), III
Parity/Second-moment (the X-gap is zeroth-moment only; moment ladders
cannot see it), IV Equivalence-to-target (a ring of exact reformulations -
instant classifier for new proposals), V Extreme-value control of sieve
patterns (the one route whose missing lemma is about the machine's own
gap words, not about primes - the live front). Trend-based verdicts are
marked observations, not walls; imported literature limits are marked as
candidate reopenings.

## Kernel-checked (Lean, all standard axioms, ledger green)

10 targets: reduction iff (BlockedSlots), horizon theorem (strict p < y),
layer novelty (fresh composite = y*c, c prime), slot-cap, supply identity
(sum R_q = C, partition form), Bridge (sum R_p = n1 + 2n2), Census
pinning, Gear ledger lines (caps, onset at q^2), Placement (slot of every
supply member; injection; placed counts), Polignac.lean 44 theorems
(g=2 pinning + uniqueness, SAME/PAIRSPLIT/CORR master-formula terms,
three_gear_master end to end), Corridor.lean (endpoint/adjacency laws,
294-entry forbidden table by kernel decide, n2 packing).

## Exact laws and events (the machine facts everything rests on)

- Horizon: gears < y decide the open window (y, y^2) exactly.
- Drift-sign event at member e^6 ~ 403: the only absolute constant found;
  window margins climb from slot ~1 above it (X impossible outright below).
- Roots-of-unity law: slot double <=> 36k^2 = 1 mod an active semiprime;
  doubles are freedom-free arithmetic. Defect identity: the X-equation's
  defect IS the twin count, per slot, exactly.
- Twin self-reference (dead-center law): twin (6m-1, 6m+1) has product
  slot k = 6m^2 at exactly the center of its thin band (T = 4m); one dead
  slot per band, everything else density (9,591 bands checked to 1e10).
- Saturated runs: unconditionally capped at 32 by the (5,7) CRT corridor;
  records: L=13 at k=2452 (member 1.5e4), L=14 at k=4.6e10 (member
  2.77e11, found on the constellation model's schedule); words are blocky,
  never strictly alternating (strict alternation caps at 6 by gear 5).
- Deletion-spacing law (q+-1)/3; chain condition: the new record gap
  F(M+q) is predicted exactly by the old gap word (verified through the
  1e9-period step 29->31: pred 58 = actual 58).
- Fuel census (chains of co-deletable openings): k_max by consecutive
  step: 2, 2, 3, 2, 4 at steps 13->17 .. 29->31; the k=4 population is
  4 per period, one word class (10,21,10), mirror-paired. Fuel words are
  literal {2u', q'-2u'} alternations - a finite local object.
- Per-gear fragile law exact with 1/ln weights (2e-4); margin trajectory
  = t - li(...) to 0.1%, gear-blind; multiplicity distribution = product
  structure exactly (slot-cap covariance = primezeta(2) - 1/4 - 1/9).

## Live fronts (the funnel, narrowest first)

1. WORD-GRAMMAR UNIFORMITY (the door everything funnels through): a
   record gap's mod-385 address is pinned by its neighbourhood word (<= 4
   offsets, uniformly, 206/206 words); is the word-shape family finite a
   priori? If yes, the machine-independent alpha1 lemma closes.
2. TOLERANCE ROUTE (event class V): F(2,y) < (y^2-y)/2 for all y follows
   from spectrum flatness + fuel bound k_max = o(ln y). Increments
   measured q/3-scale vs budget 2.5q; k_max record now 4 (29->31);
   spectrum F_j censused to machine 29 (j<=6), machine 31+ in progress.
3. F(2,53) pricing run (manager, detached): decides the constant alpha
   (2.5 <=> F(2,53) <= 486; current partial 420).
4. L=15 hunt (mechanic, detached): members to 1.2e13, ~30% done, chunk-
   flushed; model predicts first L=15 near 5e12.
5. Lean assembly: n-ary inclusion-exclusion over incidence classes =
   the last formal gap of the master formula.

## Running jobs and data

- L=15 scan: research/data/satruns_L15.log (resumable, state file).
- Machine-31 full fuel census + machine-37 partial (fuel37.log): running.
- Data inventory: research/data/*.csv - fragile/prefix/margin/supply/
  multiplicity/zone/satruns/band/fuel censuses, all append-mode with
  schemas in headers; every count exact at stated scale.

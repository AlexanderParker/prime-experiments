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
Round 14 explained the one event this whole programme was originally
abandoned over. Years ago the route was dropped because a single gear
(37) produced a freak jump - three times its neighbours, breaking the
constant everything relied on. It is not a freak: it is the moment a
second, structurally different mechanism switches on. Ordinary chains
kill openings on alternating teeth; a "padded" link kills two on the SAME
tooth, which costs a gap the size of the whole gear but buys a stretch
just as big. Padded material simply does not exist at the small machines
(counts 0, 0, 0, then 86, 6, 2090), sits harmlessly above the record when
it first appears, and at gear 37 wins for the first time. The unexplained
spike was a new tier turning on, exactly on schedule.

Better still, padding turns out to be self-limiting - for now. Two padded
links can never sit in the same killed run at any machine we can reach
(proved, and measured: zero exceptions everywhere), which restores the
ceiling withdrawn last round, at a larger constant. The honest catch: the
two quantities that make that proof work are climbing, and they cross
exactly at the next step (37->41). So we have a dated prediction rather
than a theorem - at the next machine, either a double-padded run appears
(the ceiling dies as forecast) or something further is suppressing them,
which would be a bigger discovery than the ceiling.

The machine-checked side had its best result too: the six-link cap on
ordinary chains is now a real theorem, for every gear, with no bound on
size - plus a proof that six is exactly attained, so it cannot be
improved. And a useful negative: the cap is NOT a property of the exposed
positions alone (a sloppier version is false, with runs up to 140), so
the arithmetic restriction is doing real work.

One conflict is left open on purpose rather than papered over: two
workstreams measure a padded link costing one gear-size, a third proves
it should cost three, for twins specifically. That is almost certainly a
units difference between two coordinate frames, but it is load-bearing
now, so it is flagged for whoever picks this up rather than assumed away.
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

12 targets, 996 jobs: reduction iff (BlockedSlots), horizon theorem (strict p < y),
layer novelty (fresh composite = y*c, c prime), slot-cap, supply identity
(sum R_q = C, partition form), Bridge (sum R_p = n1 + 2n2), Census
pinning, Gear ledger lines (caps, onset at q^2), Placement (slot of every
supply member; injection; placed counts), Polignac.lean 44 theorems
(g=2 pinning + uniqueness, SAME/PAIRSPLIT/CORR master-formula terms,
three_gear_master end to end), Corridor.lean (endpoint/adjacency laws,
294-entry forbidden table by kernel decide, n2 packing), Machine13.lean
(the y=13 alpha1 certificate - tiers A/B/C all closed, nothing sorried;
F=11 and F2=16 both proven and realized, budget tight; w11/w16 depend on
NO axioms at all), MaxGap.lean (F(2,y) = 0 mod 3, incl. the pruning rule
as a theorem). Technique of record: decide over residues mod 5005 does
not terminate - quantify over the CRT tuple (a<5,b<7,c<11,d<13) instead,
same 5005 cases at single-digit moduli, 12.4s.

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
- Merge algorithm (verified 18, 25, 34, 43, 58, 88): F(M+q') = max over
  maximal legal killed runs of (o[i+k] - o[i-1]), from the OLD machine
  alone. Legal link: spacings = 0 or +-2u mod q', non-zero letters
  alternating, zeros free. Two earlier versions failed and are recorded
  (literal-only undershot 71 vs 88; all-{0,+-2u} overshot 45 vs 43).
- Padding onset = the gear-37 anomaly, EXPLAINED: padded gaps per step
  0,0,0,86,6,2090 (only ever the exact value q'); the padded tier sits a
  flat +6 above F and never binds through 29->31; at 31->37 the winner is
  [pad 37][literal 12], span 49, merged 88 = F(37) - the first padded
  winner, and exactly the corpus's unexplained 2.432q spike between
  neighbours at 0.220q and 0.837q. A new tier switching on.
- Padding lemma: a run of k kills occupies k+1 CONSECUTIVE gaps, so two
  padded links separated by j literal links need F_{j+2}(M) >= 2q' + jL.
  Where that fails, at most ONE padded link per run - measured: zero
  adjacent padded pairs at every machine. Then span <= 6.35q' (ceiling
  restored at a larger constant). Enabling ratios climb (F/2q' 0.32 ->
  1.07; F2/2q' 0.47 -> 1.10): the ceiling ends exactly at 37->41.
- Padding count bound: each padded link's interior gap is >= q' while the
  budget grants (5/6)q', so one padded link forces FS < F - q'/6 and
  p <= ~F/q' (at 31->37, p <= 2.40, so p=3 impossible - as measured).
- q'-gaps are a MID-TAIL object, not a common one: 0.001-0.023% of gaps,
  and q'/meangap -> infinity at every scale. The needed statement is a
  mid-tail x extreme-tail correlation - weaker than lemma 1's extreme x
  extreme form, but still Wall V.
- Tier table: a record needs F_{k+1} >= F(M+q'); minimum k per step
  2,1,2,2,2,3. At 31->37 the record 88 exceeds F_3(31) = 85, so it is
  carried by a k=3 chain exactly - lemma 2 load-bearing at one step.
  Excess share vs fuel population correlates at -0.03: excess magnitude
  is set by flank quality, chain length enters only as a threshold.
- Tier A (machine-free, scalable): both flanks maximal forbidden at 14 of
  16 word-step pairs, decidable from (q' mod 210, w, F mod 35). Tier B
  buys nothing - lifting the modulus to 1616615 adds zero exclusions.
  Tier A is size-blind, so it cannot supply the missing bound alone.
- Firing law: chain kills alternate between teeth {u, -u}; the word's
  first entry fixes orientation, so exactly one firing residue (density
  1/q'). Every fuel site fires once per new-machine period, address
  j = (fire - p)*P_old^-1 mod q'; realized k-chains per period = N_k.
- Literal cap, KERNEL-CHECKED (LiteralCap.lean): literal_chain_le_six -
  at most 6 members for any gear with gcd(q,210) = 1, no bound on q,
  forever; cap_six_classes_sharp - 6 attained at exactly {37,53,83,127,
  157,173} mod 210 (set equality, so unimprovable). NEGATIVE: "cap <= 6
  for all (t,s) pairs mod 35" is FALSE (spectrum {2,3,4,5,6,8,10,140}) -
  the class restriction does real work; any d != 2 transfer must keep it.
- Firing law, FULLY GENERAL (all d): teeth A: n = 0, B: n = -e mod q';
  between adjacent kills sits one gap g with g = 0 (padded) or g = +-e
  (literal), else illegal; non-zero letters alternate, FORCED. Lateral's
  law with 2u -> e as the only d-dependence; 14/14 configurations exact.
- Padding is where twins differ: for 3 not dividing e all gaps are
  divisible by 3 so the cheapest padded link costs 3q'; for 3 | e it
  costs q'. d = 12's first padded winner is its FIRST step; twins' is the
  sixth. Any "padding is expensive" argument is specific to d != 0 mod 6.
- Cap transfer, UNIVERSAL: in halved coordinates the mod-105 invariance
  (phi(105) = 48) IS the mod-210 law - one check, 48 classes, all even d.
  Spectrum depends only on gcd(e,105); all 8 divisor classes computed:
  ceiling 12 iff 105 | e, 10 when 15 | e, 6 otherwise (incl. twins and
  the densest gaps d = 6,12,18,24). |E_d| = HL factor via slot_cap_gap.
  Word-identity shape transfers 13/13; tooth alternation fails for 3 | e
  (four-letter cycle with a short letter) - needs d-specific restatement.
- Fuel census (chains of co-deletable openings): k_max by consecutive
  step: 2, 2, 3, 2, 4, 4 at steps 13->17 .. 31->37; N4 = 4 at 29->31
  (word (10,21,10)) and 216 at 31->37 ((12,25,12)/(25,12,25)); N5 = 0
  everywhere scanned. Fuel words are literal {2u', q'-2u'} alternations.
- Literal cap theorem: literal chains <= 6 members for every gear (max run
  in the exposed set mod 35 depends on q' mod 210 only; caps {2,3,4,6}
  over the 48 classes; verified to prime 5000, zero mismatches).
- Interior grammar finite <=> k_max bounded (2 candidate words per k);
  pinning holds at k=4; fuel length and record growth decoupled
  (k=4 spans <= 87 < 88 = F); graded tolerance (F_{k+1}-F)/q_next <= 1.52
  at k <= 5, under the 2.5 budget.
- F(2,y) = 0 mod 3 unconditionally (endpoint argument at gear 3; all 13
  known exact values comply) - new side theorem, powers the pruned search.
- Per-gear fragile law exact with 1/ln weights (2e-4); margin trajectory
  = t - li(...) to 0.1%, gear-blind; multiplicity distribution = product
  structure exactly (slot-cap covariance = primezeta(2) - 1/4 - 1/9).

## Live fronts (the funnel, narrowest first)

1. THE 37->41 TEST (dated, falsifiable): the padding lemma's enabling
   ratios (F/2q', F2/2q') cross 1 exactly there, so expect the first
   two-padded-link run and/or the first padded link of size 82. Finding
   neither means a further suppression mechanism - worth more than the
   lemma. A gap-filtered scan suffices (padded links need gaps >= 41).
   UNRESOLVED: padded-link cost for twins is measured at q' by two
   workstreams and proved 3q' by a third - a frame difference (slot vs
   halved coordinates) is the likely cause but it is NOT settled, and it
   is load-bearing for both the ceiling and the twins-vs-other-d story.
2. THE FLANK BOUND: FS_max(w) <= F + 2.5q'/3 - span(w); measured phi ~
   0.16q' against ~0.5q' allowed. Tier A cannot supply it (size-blind,
   escape slide 1); the only candidate supplier is "near-maximal gaps do
   not cluster at pinned addresses" - Wall V with bounded complexity.
3. TOLERANCE ROUTE margin, stated honestly: incr/q' by step = 1.24, 1.10,
   1.17, 0.93, 1.45, 2.43 against alpha = 2.5. The binding step 31->37
   clears by 2.7%; the rest by 42-91%. Asymptotic safety is NOT currently
   claimed (the cap-6 ceiling covers literal chains only).
3. F(2,53) pricing run (detached, now pruned 2-5x): decides the constant
   alpha (2.5 <=> F(2,53) <= 486; 420 proven coverable, search past 420).
4. L=15 hunt (mechanic, detached): members to 1.2e13, ~36% done, chunk-
   flushed; model predicts first L=15 near 5e12.
5. Lean assembly: n-ary inclusion-exclusion over incidence classes =
   the last formal gap of the master formula.

## Running jobs and data

- L=15 scan: research/data/satruns_L15.log (resumable, state file).
- Pruned F(2,53): maxgap53_pruned.log (PID 94812); unpruned pair retires
  once the pruned log reproduces "420 coverable".
- Machine-37 fuel partial (fuel37.log, k=5 watch) + machine-31 spectrum
  (spectrum31.log): running.
- Machine17.lean: written, constants verified (F=18, F2=25, the 25
  tight), DELIBERATELY UNREGISTERED - 85085 tuples exhausts the kernel
  (memory as a proof term; >10 min as nested Bool closures). The period
  scan stops being kernel-viable near 10^5 cases for evaluation reasons,
  not because the certificate changes. Fix: 17 chunks of 5005.
- Data inventory: research/data/*.csv - fragile/prefix/margin/supply/
  multiplicity/zone/satruns/band/fuel censuses, all append-mode with
  schemas in headers; every count exact at stated scale.

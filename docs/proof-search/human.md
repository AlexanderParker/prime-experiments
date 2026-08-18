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
Round 12 turned the route's growth law from an estimate into an exact
formula. The biggest blocked stretch after adding a new gear is not merely
bounded by the old data - it is COMPUTED from it: take every "word" the
new gear permits (a short list fixed by the gear's residue class alone,
never more than six), and the answer is the best of those words' spans
plus their two flanking gaps. Checked against all six known values: exact
every time. Every increment lands inside the budget with roughly three
times the room to spare - and round 11's overshoot at four of six steps
was an artifact of the cruder bound, now gone. What remains of the ENTIRE
route is one inequality about the flanks around those few words, and its
structural reason is already computed (a record-sized gap has never once
been found flanking one of these words, 0 for 17). That inequality has the
same shape as a question already answered "no" per machine last round,
just with a word in the middle - so the machinery that closed that one
transfers directly.

Two supporting results: the six-link chain cap generalises to every
Polignac gap not divisible by 6 - same ceiling, same 48-class check - and
the exposed set it lives on turns out to BE the Hardy-Littlewood factor,
the same object seen from two sides. And the first fully machine-checked
instance of the route's missing lemma is closed: at the smallest machine,
every tier verified, nothing assumed, with the budget shown to be tight.
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
- Word identity: F(M+q') = max(F2(M), max over compatible words w of
  span(w) + FS_max(w;M)) - an identity, not a bound (lower bound from
  gcd(P_M, q') = 1: every compatible word fires, incompatible never).
  Word list from q' mod 210 alone; reproduces all six known F values.
- Firing law: chain kills alternate between teeth {u, -u}; the word's
  first entry fixes orientation, so exactly one firing residue (density
  1/q'). Every fuel site fires once per new-machine period, address
  j = (fire - p)*P_old^-1 mod q'; realized k-chains per period = N_k.
- Cap transfer: max literal-chain cap = 6 for every Polignac d not = 0
  mod 6 (48 classes, primes to 2000, zero mismatches); |E_d| =
  15/20/18/24 governed by slot_cap_gap - exposed set = HL factor.
  Excluded: d = 0 mod 6 (gear 3 keeps two free classes, walk mod 105).
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

1. THE FLANK BOUND (the single missing piece of the tolerance route):
   FS_max(w) <= F + 2.5q'/3 - span(w) for the <= 6 compatible words per
   step. Measured margins +7.2 to +21.2; structural reason computed (no
   top-stratum gap has ever flanked a compatible-word occurrence, 0/17).
   Shape = round 10's adjacency question with a word in between, an
   (l+2)-point correlation; the A/B/C tier machinery transfers verbatim.
2. TOLERANCE ROUTE (event class V): now = word identity (PROVEN) + the
   flank bound (item 1). Increments/q' measured 0.31-0.81 vs budget 2.5;
   excess overtakes lemma 1 at the largest fuel population, so lemma 2 is
   not vacuous. Word-grammar uniformity remains the machine-independent
   form's open door (address pinned to <= 4 offsets, 206/206 words).
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
- Machine13.lean (y=13 alpha1 certificate): typechecking, in flight.
- Data inventory: research/data/*.csv - fragile/prefix/margin/supply/
  multiplicity/zone/satruns/band/fuel censuses, all append-mode with
  schemas in headers; every count exact at stated scale.

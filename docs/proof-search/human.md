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
Round 13 made the growth law fully computable and, in the same breath,
took back two comforting claims - including one of mine. The good news:
the biggest blocked stretch after adding a gear can now be computed from
the OLD machine alone, no scan of the new one, and it reproduces all six
known values exactly (the sixth, 88, was independently found by a partial
scan, so it is the true value and not a floor). The mechanism at that
sixth step turned out to be new: the winner is not a longer pattern of the
usual kind but a "padded" one - two kills landing on the same tooth,
buying a stretch equal to the gear itself. Every earlier step was won by
the ordinary kind, so step six is a genuine onset, and padding gets
cheaper as the machine's gaps grow.

The corrections. Our six-link cap covers only the ordinary chains, so it
does NOT cap padded ones - which means the claim that the route is safe
in the long run is withdrawn until someone bounds padding. And the margin
I reported as "three times the room to spare" was a units error on my
part: two workstreams normalised by different constants. The true margin
at the one binding step is 2.7 percent, against a budget the other six
steps clear by 42 to 91 percent. That binding step is the same freak gear
the original programme abandoned the route over - it still fits, but only
just, and honesty about that is the point.

What survives and is strong: the fuel ceiling is now UNIVERSAL - the same
48-class check, the same maximum of six, for every prime-gap size, with a
complete table over all even gaps (absolute ceiling 12 in the rarest
class). The route's remaining requirement is one bound on padded runs -
how often gaps of exactly one gear-size can chain - which is the same
"near-maximal gaps do not cluster" statement that the last three rounds
kept converging on, now aimed at a concrete new object.
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
- Padding onset: the 31->37 winner is [kill]-37-[kill]-12-[kill] - one
  padded link (two kills, same tooth, span q') plus one literal link;
  span 49 beats the longest literal span 37. Steps 1-5 all literal-won.
  The cap-6 theorem does NOT bound padded runs - bounding them is the
  route's live requirement.
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

1. BOUND THE PADDED RUNS (the route's live requirement): how often can
   gaps of exactly q' chain? Each padded link needs a top-gap of M, so
   this is the rounds 9-10 adjacency machinery on a new object. Until it
   is bounded, no constant ceiling on the excess follows.
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
- Machine13.lean (y=13 alpha1 certificate): typechecking, in flight.
- Data inventory: research/data/*.csv - fragile/prefix/margin/supply/
  multiplicity/zone/satruns/band/fuel censuses, all append-mode with
  schemas in headers; every count exact at stated scale.

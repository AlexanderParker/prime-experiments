# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State after round 20 - the frames round. The human's two directives (matrix/linear algebra,
complex numbers) both landed; (A) is now FULLY KERNEL-CHECKED, leaving (D) as the sole open
part of the route; (D) itself was stripped of its last heuristic; and the machine's laws
began unifying into one operator algebra.

THE EXACT QUALMAX CRITERION (constructor R39, the live target's new form): by the merge law,
F(M+q') <= max(F2, max_j qualmax_j), and (D) at alpha=3 follows from
max(F2, max_j qualmax_j) <= F + q' - three full-period census quantities, NO lambda, no order
statistics. Holds 7/7 measured steps, EQUALS F(M+q') at 6 of 7 (slack 2 at 23->29). Margins
0.52-0.69q' literal steps, 0.19q' at the padded 31->37 - but Mechanic's 37->41 test shows the
Q_j collapse does NOT continue (0.83q', 16%): it is a litcap-6 phenomenon; next real test q'=53.

(A) CLOSED AT THE KERNEL (formalist): cap_table_maximal + cap_table_realized - the 48-class
word-list enumeration exact both ways, per-class chain bound literal_chain_le_capC. Machine 19
certified ([propext] only): F=25, F2=31, F4=38, and the first END-TO-END (D) INSTANCE:
D_of_shallow_word - (D) at alpha=3 at machine 19 as a kernel theorem about the machine's own
gap word (only shallowness l+2<=4 remains hypothetical; census k_max=3 there). Build 1276 jobs.

THE OPERATOR ALGEBRA (matrix frame, all lanes + follow-up explorer): state = tensor of per-gear
spaces; F(M) = NILPOTENCY INDEX of the blocked-walk operator BS (verified 7 machines, index=F
exactly); BS = (tensor S_q) - (tensor E_qS_q) - WALL V IS TENSOR RANK 2; fuel cap = nilpotency
index of the qualifying map (2,2,2,3,3,4,4 at 11..31). REFUTED WITH STRUCTURE: no spectral gap
(renewal operator is a permutation), aggregated gap chain NOT Markov (over-predicts deep runs
x49 at m29 d3; each link ~30x more suppressed than the last - favourable for (D), fatal for
naive spectral bounds). Wiener-Khinchin: c-law = corridor = depth-sum = paired-Holt diagonal,
ONE exact matrix identity. Live tinkering with the human: nilpotency growth delta = 3,2,4,7,7,9
vs q' = 7..23 (~q'/2.5); TWO-TEETH KILL SPACING LAW: kill spacings within a window lie in
{2u', q'-2u'}, min ~ q'/3 (fuel <= ~3L/q' closed form); record windows merge SMALL old gaps
only ([4,8,15,7] at +23 vs F_old=25) - the anti-correlation watched live. Also: the open-walk
operator has index 3 FOREVER (gear 5's teeth at +-1 mod 5 forbid three consecutive
twin-eligible slots - an eternal cap found by pointing the matrix the other way).

THE FOURIER FRAME (lateral + mechanic): machine DFT factorises, real, closed form (verified vs
FFT all 85085 frequencies of m17); T3 law 3u = (q+1)/2 mod q kernel-checked; GOLDEN SPECTRAL
GAP phi/3 = 0.539 for every machine containing gear 5, hat_5(2) = phi EXACTLY (charpoly
(x-3)(x^2-x-1)^2); mechanic: subleading transfer eigenvalue 0.6273 -> 0.5425 (m13..37),
distance to phi/3 shrinking geometrically x0.6/machine - convergence conjectured on 7 exact
points, no fit. DEPTH-SUM IDENTITY (lateral, proved): sum_j W_j(g) = prod_q c_q(g) - every
depth at every g bounded by one closed form.

CORRIDOR RESONANCE (mechanic, new measured law): qualifying-gap autocorrelation is a barely
damped period-35 wave; big-gap left endpoints PINNED mod 35 (invariant core {10,12,18}, five
machines); separation autocorrelation peaks exactly at 35/70/105 (x4.4). The lag-1-3 deficit /
lag-4-7 excess is ONE phenomenon. NOT k-step Markov for k<=4: Constructor's transfer matrix
must carry CORRIDOR PHASE, not last-gap state. Unexplained: C14 residue law has a
machine-independent phase +126 deg +-2 (seven machines) - handed to Lateral.

COV-SAT (mechanic's centerpiece): every gap/window/fuel question as ~300-var CNF over gear
phases, witnesses CRT'd back and machine-verified. Beyond any scan: MACHINE 41 COMPLETE
(period 5.07e13): F(41)=91, holes {84,87,89}, healing law holds. F2(37)=90 exact (lemma-1
margin 2 of 41 - deep end easier again). F(2,53)=435 EXACT - alpha=3 budget passes with 15%
room. Fuel caps decided at FULL period: k_max=3 at 37->41 AND 41->43 (N4=0 both, exhaustive).
FIRST DOUBLE-PADDED RUN EXISTS: word (43,43) at 41->43, k=116,431,845,582 - the r16 prediction
decided with an address; gap 86=2q' also first. Standing: F3(37) in [97,163] - the (D)-decision
at 37->41 needs <=129, 34 refutations away (~15 min each). L=15 absent to 1.2e13 (sub-1sigma).
kwin31 full period: k_win=3, zero k>=5 to depth 8 - r17 pre-registered test CONFIRMED.

HARVESTER (own mandate): why-13 CLOSED as four exact events - slack quantised mod 6, minimum
attained only at p=5,13 through 73 (the dip is ONE QUANTUM above equality); margin falls at all
6 twin steps >= 13, rises at all 5 gap-6 steps; r=3.231 at 11->13 unique jump > 2.6; clean
extension dies after 11->13 forever (best 17-extension loses by exactly 9 - unexplained).
ROUTE-RELEVANT: fixed differences with single-step increments 3.231/3.947/4.435 q' exist - NO
UNIFORM alpha<=3 BUDGET OVER THE FULL FAMILY (twins' own budget unaffected; the family
generalisation is dead as stated). Twin percentile externally validated vs ZM (12 machines,
extreme 1.34-2.27x twin, median 1.70x). PAIRED HOLT RECURSION built and verified exact 4 rungs:
two-residue linear population dynamics, diagonal = c-law, eigen-scale (q'-2j-2)/(q'-2) -
paired contracts twice as fast as Holt's one-residue - a transfer matrix with KNOWN ENTRIES.
New N4: ANY proved bound on j_2 - zero published attempts exist.

RENEWAL LADDER (constructor, rigorous): #(X exposed, Y blocked) = sum over T subset Y of
(-1)^|T| prod_q c_q(X u T) - exact CRT closed form, no scan; monotone ladder clears the (D)
requirement at every constrained case incl. both R32 failures (x300, x91, x2000 clearance);
first joint-gap bounds beyond scan reach: p_5(37) <= 3.4e-2, p_6(37) <= 9.8e-3. Zero
certificate NOT reached (2^|Y| IE cost) - named blocker: a pattern counter cheaper than IE
(Mechanic's COV-SAT is the candidate supplier).

MERGE-LAW BENCHMARK (standalone, docs/novel/merge-law-h2-test.md): h_2(19)=258 and h_2(23)=366
replicated from the OLD machine's words alone - 962x fewer operations than construction;
values match Ziller-Morack to the digit. The h_2 family ladder cannot ride past ~29 (class
count x q' per rung); twin ladder verified through F(2,43)=309.

PRIOR-ART REGISTER (docs/novel/, 17 entries): all 10 round-1-19 candidates checked; matrix
formulation piecewise-checked (scaffolding KNOWN - Good-Thomas, circulants, HL local factors;
NOVEL* deltas: nilpotency-for-Jacobsthal, rank-2 Kronecker-difference, golden ratio in sieve
spectrum, two-residue transfer matrix). ZM's companion note 1706.03668 found: our h_2 values
are REPLICATION not first computation; their h_2(19)=258 settled our y=19 question.

HONEST NEGATIVES / CORRECTIONS this round: envelope41's "(D) PROVED" log line INVALID (tiny
prefix, F wrong by 18); r19's k_win=1-at-37->41 claim wrong (full period: k=3, padded run
carries the record - prefix k_win is a lower-quality object); per-cell endpoint factorisation
refuted; corridor-forcing of (D) at n=4 refuted (0/1225); kappa density-universality first
order only; a constructor seam double-count (+15 gaps) caught by cross-check and re-run; a
process sweep killed 15+ jobs in waves - per-instance logging + resume flags prevented all
data loss (two "killed" jobs had actually completed - check CSVs before rerunning).

THE HUMAN'S FRAMING (2026-08-24, now the route's stated philosophy): the basic rules never
change; the difficulty is the interactions of gear SETS slipping against each other; and the
goal is not an infinite ladder of per-arity rules but THE GENERATOR of the complexity - one
algebra whose expansion produces every layer. Pairs are conquered (c-law/corridor/spectrum
unified); the wall is proven to start at 3-point interactions (no 2-point object certifies the
depth cap from m19 on - tropical R37). Proof shape preferred: mechanism exhaustion - show no
mechanism or combination can block twin recurrence, converting statistical statements into
mechanical caps, as done for (A)/(B)/(C).

ROUND-21 (briefed, not launched - each in its own lane):
CONSTRUCTOR -> ONE ALGEBRA, NOT INFINITE RULES: nilpotency additivity as the (D) proto-law -
index growth under tensor-and-strike, using BS = rank-2 tensor difference, the two-teeth
spacing law {2u', q'-2u'}, and fuel = bridges; then rebuild the transfer matrix carrying
CORRIDOR PHASE (mechanic's resonance says last-gap state is the wrong state space). R39 at
37->41 decides when Mechanic lands F3(37).
MECHANIC -> finish F3(37) (34 refutations, ~15 min each - the (D)-decision at 37->41); then
q'=53 Q_j margins (the litcap-6 hypothesis's real test); COV-SAT as the pattern-counter
supplier for Constructor's ladder zero-certificates; machine-43/47 tails as checkpointed
background.
LATERAL -> the C14 +126 deg machine-independent phase (your own unexplained event); the
EIGENPHASE STATISTICS test (human's Riemann-bridge hunch): pair-correlation of machine-operator
eigenphases vs GUE vs Poisson as machines grow - exact, falsifiable; the PSD/large-sieve
constraint from the exact spectrum (does positive-definiteness BITE on violating windows?).
FORMALIST -> R39 as a two-machine kernel statement (QualBound is the vocabulary); Q_5(19)<=48
scan (removes D_of_shallow_word's last hypothesis); depth-sum identity at m13; T3 done - next
the two-teeth spacing law (elementary, closed-form, kernel-ready).
HARVESTER -> N4 (any proved j_2 bound - empty ladder, zero attempts in the literature); the
clean-extension death at 17 (loses by exactly 9 - unexplained); paired-Holt eigen-analysis
toward HL-B in paired cycles (N5).

NOVEL-FINDINGS RULE (all agents, standing - from the human, 2026-08-23):
Anything potentially novel to mathematics gets its own document in docs/novel/ (template and
rules in docs/novel/README.md) IN THE SAME ROUND it is established: what it is, why it might be
novel, proof or proof pointer with honest status (kernel-checked / script-verified / measured /
conjectured), implications, unsolved questions or conjectures it touches, and a prior-art check
(agents without web access write "not yet checked"; the manager runs the check and records the
verdict). Over-inclusion is fine - the check sorts it out. Update docs/novel/README.md's index.
This is an exception to the scope rule: docs/novel/ is writable by every agent.

CLEAN-CONTEXT RULE (from the human, 2026-08-23): round-20 agents start with CLEAN context.
Read the compacted workstream docs (this file's SUMMARY + rules, your own doc, and any doc your
brief names) - do NOT read docs/proof-search/archive/ except to verify one specific claim whose
compacted statement you need at full detail. The archives are verbatim rounds 1-19 logs kept so
nothing is lost; they are not round-20 reading.

HUMAN DIRECTIVE (2026-08-23) - TWO NEW FRAMES, work them into the round:

1. MATRIX / LINEAR ALGEBRA. Express the machine's discovered structure in matrix form and see
   what routes open. Entry points (suggestions, not limits): gear blocking as circulant /
   permutation matrices over Z_q; the merge transform as a linear operator on gap words; the
   corridor mod 35 as a 35x35 operator whose powers generate exposure; TRANSFER MATRICES for
   the gap-word grammar - p_j (the joint distribution of qualifying gaps at separations 1..j)
   as a product of transfer matrices, so the measured anti-correlation deficit (x26, x6.7,
   x1400) becomes a SPECTRAL statement (spectral gap / Perron-Frobenius bound) instead of a
   census; chain and literal caps as nilpotency / spectral-radius facts.
2. COMPLEX NUMBERS. The blocking indicator is exactly a sum of q-th roots of unity
   (1_{q blocks k} = (1/q) sum_j e^{2 pi i j (k -+ u')/q}), so every census the project has
   taken has an exponential-sum form. Gear alignment is a PHASE relation. Take the DFT of the
   corridor, of gap spectra, of the joint gap-pair distribution; the earlier frequency-space /
   phase look was abandoned early - re-enter it now WITH the round-19 objects in hand
   (suppression law, p_j, flank shapes), not the round-1 ones.

Lane assignment (consistent with the mandate rule - each frame lands in the lane it serves):
  CONSTRUCTOR - transfer-matrix formulation of p_j; this is the live route's own target
                restated, not a re-tasking.
  LATERAL     - the complex/Fourier frame is its native lane (reframings); also any matrix
                form the other lanes cannot reach.
  MECHANIC    - measure what the new frames predict as exact events: eigenvalues, spectral
                gaps, character/exponential sums against their census values. Events, not fits.
  HARVESTER   - literature adjacency on its own mandate: exponential sums over sieve residues,
                transfer-matrix sieves, where the named functions it tracks meet these frames.
  FORMALIST   - unchanged mandate; pick up any matrix identity that becomes exact and finite
                (a finite matrix product equalling a census number is kernel-checkable).

ROUND-20 BRIEFS (historical - all five filed 2026-08-24; see round-20 appends below):
CONSTRUCTOR -> the anti-correlation law: a formula for p_j, the joint distribution of
qualifying-size gaps at separations 1..j. That deficit (x26, x6.7, x1400) is what would make the
suppression law rigorous, and it is now the whole of (D). Mechanic owes you the joint gap-pair
census at separations 1-5; Lateral's c_q(g1,g2) is the same object from the corridor side -
three workstreams converging on one construct, so state precisely what you need from each.
MECHANIC -> (a) the joint gap-pair census at separations 1-5 over whole periods, for
Constructor's p_j; (b) the COVERABILITY SPECTRUM COV(M) you named - CRT arithmetic, no period
scan, reaching machines 37/41/43/53, giving the UPPER bounds every prefix row lacks; (c) the
k_win >= 4 falsification watch at 31/37/41. Your Q_j margin collapse (0.45q' -> 0.10q') is the
counterweight to watch - if it keeps falling, say so early.
LATERAL -> c_q(g1,g2), the gear x lag-pair autocorrelation: your own named next construct, and
independently the natural object for (D) since a flank sum IS a two-lag quantity. Also the
autocorrelation at the padded lag q'. Note the interior condition is a disjunction and does not
factorise - that obstruction is the interesting part, not a reason to stop.
FORMALIST -> (A)'s remaining gap (the word-list ENUMERATION, the only part of (A) not checked),
then the suppression-corrected flatness statement as a hypothesis-explicit theorem so the
censuses can discharge it. Your tier-C re-attack stands: machine 19 at ~20 min is now worth
doing.
HARVESTER -> "why is 13 extremal?" - your own sharp question, on your own mandate: the h_2
margin dips to 3.8% at y = 13 and recovers. That is a named-function anomaly with a literature
attached. Also worth stating for publication: twins are the 13.3rd percentile of their own
family, which reframes what Reduction A is.

JOB-COMPLETION RULE (all agents, standing - from the human, 2026-08-18):
A round is NOT finished while any job it launched is still running. Do not file a round report
on partial coverage and do not promise to "fold results when they land" - WAIT for your own
detached jobs to finish, then report once with complete data.

Consequences, all intended:
- Launch jobs EARLY in the round, not at the end. A job started in your last few actions will
  hold the whole round open.
- Size jobs to the round. If a census would take many hours, either narrow it so it completes,
  or split it so the part that completes this round is self-contained and the rest is a
  deliberately-scoped next-round job - do not start an open-ended run and report around it.
- If a job is genuinely stuck or dead, that is a finding: say so, with what you did to establish
  it, rather than treating it as still-pending.
- At the end of the round every process the round started is finished, so the round's data is
  complete when the write-up happens and nothing arrives afterwards to invalidate it.

(The manager gave the opposite latitude in round 20 - "report without them rather than blocking"
- and that was wrong. This rule supersedes it.)

BENCHMARK PROTOCOL (all agents, standing - from the human, 2026-08-23):
Performance comparisons COUNT OPERATIONS, not wall time - instrument both code paths with
explicit counters (letters scanned, deletions applied, strikes sieved, or a closed-form op
count when a run is infeasible) so results are decoupled from the machine. Report ops per step
and the ratio. At most one wall-time column as a secondary sanity check, and it must come from
runs executed ALONE - never run compared computations side by side (CPU/cache contention
invalidates the timings; this ruined a benchmark once already).

MEASUREMENT DIRECTIVE (all agents, standing - from the human, 2026-08-18, and it overrides the
default habit of this search):

"'Measured everything measurable' is obviously not true - if we had, we'd have solved the
conjecture. Focus on finding NEW measurements, by exploring new machine constructs derived from
RELATIONSHIPS BETWEEN ITS PARTS. When the analysis points at complexity - that's not a wall,
that's the solution space we need to push into."

What this changes, concretely:
- Every measurement so far was taken on an object we already knew to name: gears, teeth, slots,
  gaps, words, chains, flanks, spectra. The unmeasured space is the RELATIONSHIPS between those
  objects, and constructs built out of those relationships. Build the new object first, then
  measure it - do not re-measure known objects at larger scale and call it progress.
- "Still Wall V", "extreme-value control", "arithmetic luck not structure", "no smooth law, only
  the histogram" - these have been the terminal verdicts of many rounds. They are NOT stopping
  points. Each one names a region we have declined to enter because it looked complex. Enter it.
  If a quantity is erratic and arithmetically selected, that erraticity is itself an object with
  structure - measure THAT (its own histogram, its correlations, its generating relation), rather
  than reporting that no smooth law exists.
- A report that ends "this is the limiting event, still open" is incomplete unless it also names
  the construct that would have to be built to go further, and why it was not built this round.

MANDATE RULE (all agents, standing - added after a manager error, 2026-08-18):
Each workstream works ITS OWN MANDATE. The manager does not re-task a workstream to whatever the
live route needs that round; that is what happened over rounds 3-17 and it left two mandates
unserved while five agents crowded one inequality.

  MECHANIC   - empirical censuses at scale on the machine's real structure; EVENTS with exact
               counts, never fitted trends. Standing rule earned the hard way, three times:
               never extrapolate a per-step share - look it up.
  CONSTRUCTOR- build the proof; attack the target directly. Owns the live route.
  FORMALIST  - kernel-checked Lean, zero sorries, honest reporting of what will not close.
  LATERAL    - unorthodox angles, reframings, self-reference; the directions the other four
               cannot reach. NOT a second analyst on the live route.
  HARVESTER  - side theorems and ADJACENT CONJECTURES, per its own round-1 ranking. NOT
               twin-route support (formal work goes to Formalist, censuses to Mechanic).

If a brief from the manager reads as another workstream's lane, the agent should push back and
cite this rule. Drift is a coordination failure, not an agent failure.

SCOPE RULE (all agents, standing): write ONLY your own workstream doc, your round append here,
and files you created in research/ or proofs/. The SUMMARY, human.md, other workstreams' logs,
and all corpus docs (docs/*.md outside proof-search/) are off-limits without an explicit
manager instruction in your brief. (Rounds 9-10 compliance: all five agents clean.)

SUPERSEDED-ROUND-14: Lateral -> BOUND THE PADDED RUNS (their own next target, and now the route's live
question): how often can gaps of exactly q' chain? Each padded link needs a top-gap of M, so
this is the rounds 9-10 adjacency machinery aimed at a new object. Constructor -> the padding
question from the tolerance side: with tier A size-blind and tier B dead, what does a padded-run
bound have to look like to give phi, and is the near-max non-clustering statement (Wall V,
bounded complexity) genuinely the only supplier? Mechanic -> the padding census: how many gaps
of exactly q' does each machine carry, and how do they chain (the empirical side of lateral's
target); continue the k=5 hunt at any step Constructor nominates. Harvester -> the d-specific
firing restatement for 3 | e (four-letter cycle, short letter), and whether padding transfers to
general d. Formalist -> (when the in-flight work lands) the 48-class cap via CRT tuples, then
tier A's machine-free exclusion as a kernel theorem - per Constructor, tier A is the only
scalable piece and now has an exact statement: (q' mod 210, w, F mod 35) decides it.

SUPERSEDED-ROUND-13: Constructor -> the (l+2)-point correlation: transfer the A/B/C tier machinery to
FS_max(w) <= F + 2.5q'/3 - span(w). Tier A first (machine-free forbidden configurations around a
word occurrence - the generalisation of no_11_11_chain), then what the per-machine check costs
at each of the six steps. This is now the single missing bound of the whole tolerance route.
Lateral -> the excess share vs fuel population: does it saturate or climb (the 0.811 at 31->37
is the warning shape)? Needs machine 37/41 spectra - coordinate with Mechanic. Also: with
firing settled as density-not-count, restate the graded tolerance cleanly. Mechanic -> land
machine-37 fuel (the k=5 falsification test) and machine-31/37 spectra; then the excess-share
census Lateral needs. Formalist -> the 48-class literal cap via the CRT-tuple recipe (Constructor
23.2), then machine 17 (period 85085) where tiers B/C first genuinely separate from the scan.
Harvester -> the d != 0 mod 6 restriction: what does the mod-105 walk give for d = 6, 12, 18
(the densest gaps)? Plus: does the word identity itself transfer to general d?

SUPERSEDED-ROUND-12: Constructor -> the WORD-INDEXED TOLERANCE THEOREM: assemble the certified per-step
ceiling from the literal cap (<= 6 words/step) + flanks + pinned addresses, and test it against
every measured step - state exactly which flank-sum bound closes the route and what it costs.
Lateral -> the FIRING RATIO: fuel sites x phase alignment across all censused steps (mechanic's
216-site N4 at 31->37 is the sample); quantify double rarity and what it does to the graded
constant. Mechanic -> fold machine-37/spectrum-31 results when they land; verdicts on
Constructor's five falsification criteria; k=5 watch at 37->41. Formalist -> finish Machine13
certificate (in flight); then the literal cap's 48-class check and F(2,y) = 0 mod 3 as kernel
targets. Harvester -> monitor the pruned run; formalize its two pruning theorems' number-theory
cores (mod-3 endpoint, left-taut equivalence) or hand them to Formalist with exact statements;
resume related-conjecture harvesting with the new fuel machinery (Polignac per-d: does the
literal cap transfer to d != 2?).
## Toolbelt inventory (all verified this session)
- research/umbrella_tools.py: closed-form umbrella membership/edges for any gear set (min-rooms)
- research/slip_path.py: state_walk (per-slot gear states + kill attribution), mex_jump,
  chain_prediction (stride growth from gap word, correct k-frame window {phi, phi+s})
- research/slip_bezout.py: slip-chain->Bezout alignment, product sign law, nudge constructor
- research/chain_census.py: chain-length census + fuel words; research/band_attribution.py
- research/minimal_subset.py, sufficient_subset.py, event_horizon.py, layer_ledger.py,
  coprime_census.py, kappa_exact.py, kappa_profile.py, deficit_scan.py
- proofs/BlockedSlots.lean: kernel-checked reduction (iff), builds clean; lake at ~/.elan/bin/lake.exe

## Established laws (session-proven unless marked measured)
- Horizon theorem: gears < y decide the open interior (y, y^2) exactly; top gear's unique acts
  are boundary only. Layer law: one layer's novelty = {y^2} + {y*c : c prime in (y, y'^2/y)}.
- Composite root law: every squarefree product of set gears acts unshadowed exactly once per
  window (its own value), if it fits. Root ordering: shadow < q^2, square, then coprimes.
- Necessity law: gear q needed iff it owns a pseudo-twin (root kill beside a prime) in window.
  Square gate: q^2-2 primality (prime at 5,7,13,19,29,37,43,47).
- h(L) >= d proven for L = 1,2,3 at every y (k-frame); kappa(2) limit = 2 - (11/3)C = 0.5448.
- Deletion spacing (q+-1)/3; span law >= floor((k-1)/2)*q; chain condition verified (predictions
  18/25/34 exact); fuel words rare (k=4 first at y=29, word (10,21,10)); measured: fuel abundance
  may explain gear-37 increment anomaly.
- Measured at scale: in-window max stride ~ 0.47*log^3(member)/6 slots; stride/window collapses
  2.1e-2 (y=101) -> 6.0e-7 (y=100003, members to 1e10). 27.4M twins generated+verified.
- Every gear/composite: 2 teeth summing to modulus, shield centred in short umbrella, umbrellas
  1/3+2/3, self-blocks own pair (u' = round(q/6)); u'-column doubling = the twin sequence itself.
- The one open quantity (all equivalent forms): the all-umbrella slot recurs inside every window
  (Reduction A, kernel-checked equivalent to the conjecture).

## Round findings (rounds 1-19)

Compacted 2026-08-23 by human directive. The full verbatim round-by-round appends are preserved
at `docs/proof-search/archive/agents-shared-full-r1-19.md` (nothing deleted; also in git
history). Cumulative findings live in each workstream's own doc (also compacted, with full
verbatim copies in `archive/`). New round appends go below this line as before.

## Lateral round 20

Both briefs served: c_q applied to (D)/flanks/padded lag, and the complex frame entered with
round-19 objects. All jobs finished before write-up. Five results, two novel-register docs.

1. DEPTH-SUM IDENTITY (proved, one line; integer-exact machines 11-29, g = 1..64, zero
   mismatches): sum_{j>=1} W_j(g) = prod_q c_q(g), where W_j(g) = # j-windows of consecutive
   gaps summing to g. Every opening pair at lag g is the endpoint pair of exactly one window;
   CRT counts the pairs. COROLLARY: W_j(g) <= prod_q c_q(g) for EVERY depth j - a closed-form,
   depth-uniform upper bound on every window-sum count, no period scan, any machine. The whole
   F_j family sits inside one closed-form sum rule (complements Mechanic's COV(M) from the
   other side). docs/novel/depth-sum-identity.md; research/depth_identity.py writes exact W_j
   tables to research/data/depth_identity_<y>.csv (machines 11-29) for anyone's use.

2. RENEWAL DECOMPOSITION: W1(g) = N2(g) * prod_t(1 - N3(0,t,g)/N2(g)) * kappa(g) - closed-form
   endpoint arithmetic x closed-form interior-independent product x measured remainder. kappa
   is the interior DISJUNCTION isolated. Findings: kappa(1)=kappa(2)=1 trivially; kappa(4)=1
   EXACTLY at machines 13-29 (integer-exact vs full inclusion-exclusion) though the per-gear
   multiplicativity behind it is FALSE for q >= 7 - a cross-gear cancellation, open
   micro-question. kappa decays smoothly, log-CONVEXLY, slope stabilising ~ -0.16/slot
   (23/29/31); the 2-parameter kappa law + closed form explains 94.9-98.7% of the histogram's
   log-variance at machines 19-31. HONEST LIMIT: dividing out N2 alone removes only 11-30% of
   the post-trend wiggle (r18's c5*c7 got 24-28%) - the endpoint dividend saturates; the wiggle
   is INTERIOR arithmetic, which the full closed form does capture (see 3).

3. THE MACHINE IN FREQUENCY SPACE (human frame 2; all exact, script-asserted). The DFT
   factorises over gears, is REAL and closed form: hat_q(j) = -2cos(2 pi j u/q); verified
   against FFT at machine 17 over all 85085 frequencies (dev 4e-11). T3 LAW: 3u = (q+1)/2 mod q
   for every prime q >= 5 (asserted to 100000) - tripled teeth are ADJACENT residues at the
   antipode, hat_q(3) = -2cos(pi/q) -> -2: at local frequency 3 every gear is nearly a single
   point in phase (the Fourier avatar of the tooth law). GOLDEN SPECTRAL GAP: hat_5(2) = phi
   exactly, and for every machine containing gear 5, max non-DC |hat|/DC = phi/3 = 0.539345,
   attained only by gear 5's +-2 mode (full character enumeration, machines 13/17) - the
   spectral-gap form of gear-5 corridor dominance. Measured: the gap histogram's dominant
   oscillatory line at machines 29/31 IS the golden line (freq 2/5), and subtracting the full
   closed form removes 99.6%+ of its power (0.1687 -> 0.0006 at 29). "No smooth law, only the
   histogram" is now: histogram = closed-form arithmetic x smooth renewal, with named lines.
   docs/novel/golden-spectral-gap.md; research/machine_dft.py.

4. WHAT (D)'s ANTI-CORRELATION IS MADE OF (research/pair_renewal.py, on Mechanic's joint
   census incl. machine 31 = 6.2e9 pairs). (a) The adjacent-gap exclusion law extends to
   machine 31: ZERO counts in the 6 forbidden mod-5 classes. (b) Bulk pairs: the irreducible
   correction FACTORISES on count-weighted average (kappa2/(k1*k1) mean 0.99 at 29 and 31) but
   NOT per cell (log-sd ~1). (c) QUALIFYING pairs: measured R(1) sits x2.4-x6 BELOW the
   closed-form + factorising-renewal prediction (meas/pred 0.167, 0.30, 0.38-0.42 at 23/29/31)
   and the full 4-POINT interior predictor does NOT close the gap: THE ANTI-CORRELATION (D)
   NEEDS IS GENUINELY BEYOND 3- AND 4-POINT CRT ARITHMETIC - it lives in the interior
   condition at qualifying lags. The residual shrinks toward 1 with machine size (3 points -
   watch, not extrapolate). DEFINITION FLAG for Constructor: my measured machine-19 R(1) is
   1.28 (positive) under size+residue qualifying, 2.03 under size-only - neither matches R33's
   reported deficit range; please state R33's exact qualifying set. (d) PADDED LAG: the full
   predictor takes the padding-supply erraticity down to kappa(q') in [0.007, 0.107] across
   the four measured steps - a 15x residual where round 19's endpoint sigma explained ~1/10 of
   330x; the 23->29 supply (6) sits below even its own machine's kappa trend.

5. FLANK-SUM CORRIDOR LAW - (D) IS NEVER CORRIDOR-FORCED (research/fs_corridor.py). The
   4-point shape {0, gL, gL+s, s+T} (s = span, T = flank sum) can be blocked only by gears
   5,7 (completeness lemma), so mod 35 decides. Result: 0 of 1225 (s,T) classes are blocked -
   every flank-sum value above the (D) requirement is corridor-feasible for every span
   (checked against all 47 census word-steps: no (D)-critical interval touches a blocked T).
   Yet 51.6% of individual (gL,s,gR) mod-35 SPLITS are blocked - the sum always survives
   through the split disjunction. VERDICT FOR CONSTRUCTOR: do not spend anything on a corridor
   proof of (D); the only route is the counting/occurrence form (R33).

Refuted this round: per-cell endpoint-arithmetic factorisation of pair interaction (log-sd
~1.2, drifting bias); full-closed-form explanation of padding supply (15x remains);
kappa(g*density) universality (first order only); corridor-forcing of (D) at n=4 (0/1225).

Open, named: kappa(4)=1 mechanism (cross-gear cancellation, kernel-checkable); kappa's
log-convex tail as the Wall-V residence (fit top decile vs F - not done this round); lag>=2
R prediction = a transfer-matrix product over the closed-form 3-point kernels (Constructor's
frame-1 object - kernels are ready in my scripts); the qualifying suppression trend needs the
machine-37/41 joint census (Mechanic); a large-sieve inequality on W_j from the exact spectrum
(named, not built).

Needs from other lanes: CONSTRUCTOR - R33's exact qualifying-set definition (see 4c); the
transfer-matrix kernels N3(0,g1,g1+g2)/N2 are computed and closed-form if wanted. MECHANIC -
joint gap-pair census at 37/41 when COV work allows, to settle the qualifying-suppression
trend. FORMALIST - two cheap kernel targets if wanted: the depth-sum identity at a fixed
machine (finite), and T3 (3u = (q+1)/2, two lines).

## Harvester round 20 - why-13 closed, percentile externally validated, the paired Holt recursion

(Own lane throughout: named-function anomaly, publication statement, literature adjacency of
the two frames. Scripts assert-gated and green: research/zm_margin_mechanism.py,
family17_percentile.py, paired_holt_recursion.py; no detached jobs; nothing pending.)

1. WHY 13 IS EXTREMAL - CLOSED, as four exact events against ZM's full 19-point table
   (docs/novel/paired-jacobsthal-values.md 4a): (i) QUANTISATION - the Conjecture 6 slack is
   quantised mod 6 and attains its minimum admissible value ONLY at p = 5 (slack 2) and
   p = 13 (slack 6) through 73; the dip is "one quantum above equality" (omega_2 = 24 = cap
   25 - 1). (ii) STEP LAW - margin falls at ALL 6 twin steps >= 13, rises at ALL 5 gap-6
   steps, gap-4 mixed (3 up 2 down); ABSOLUTE slack falls only at twin steps (13, 31, 61).
   Crossover mechanism: d(B)/B ~ 2g/p vs d(h_2)/h_2 ~ 2r/p, sign flips at gap ~ r.
   (iii) UNIQUE JUMP - r = Delta(maxF)/q' = 3.231 at 11->13, the only step > 2.6 of all 18.
   (iv) LAST CLEAN EXTENSION - winners extend winners at 7->11 and 11->13 (16/16 keep the
   profile; the same fixed e gains 3.231 q' in one step) and never again: best 17-extension
   of a 13-winner reaches 87 vs max 96; merge-law round's 19-argmax rank at 17 (35,848 above,
   value = twin's own 54) verified from my full 17-scan. The dip = the family's only
   full-extension jump landing on a twin bound-step, one quantum from equality.

2. BUDGET EVENTS FOR CONSTRUCTOR (measured, verified by direct construction): fixed
   differences with single-step increments 3.231 q' (e=344, 11->13), 3.947 q' (e=1,532,627,
   17->19, 54->129), 4.435 q' (e=107,207,699, 19->23, 81->183) EXIST. Round-14 audit's
   structured-d worst was 1.846; twins' own 2.432. NO uniform alpha <= 3 increment budget
   holds over the full even-difference family - (D)'s constant is structured-d-specific;
   "closing (D) closes every d" needs per-d constants or an explicit family-argmax exclusion.

3. TWIN PERCENTILE, EXTERNALLY VALIDATED (docs/novel/twin-percentile.md 4a): twin/extreme
   now known at 12 machines y=5..43 using ZM's h_2 as external denominator - twin attains
   the family max only at y = 7; extreme = 1.34x-2.27x twin, median 1.70x. Exact tie-aware
   percentiles: coprime-class strictly-below 13.3% (13) / 21.3% (17), strictly-above 77.2% /
   68.6%. Per-class arrays saved (research/data/f13_family.npy, f17_family.npy) - stop
   recomputing the families.

4. THE PAIRED HOLT RECURSION (new, docs/novel/paired-holt-recursion.md; the frames
   directive's literature import DELIVERED): Holt's cycle-of-gaps population dynamics
   (arXiv:1510.00743 Thm 3.2, one residue per prime, transfer matrix diag (p-j-1)/(p-2),
   p-independent eigenvectors) has an exact TWO-RESIDUE analogue:
       n_g(M+q') = sum over words w with sum(w)=g of coef(w) * n_w(M),
       coef(w) = #{r in Z_q' : r,r+g not in T; every interior partial sum in T},
   position-free. VERIFIED EXACT for every gap value at 4 rungs (slot 5005->85085->1616615;
   family e=344 +17; gcd collapse e=102 +17 which IS Holt's one-residue case). The j=1
   diagonal is EXACTLY Lateral's round-19 autocorrelation c_q(g) in {q-2,q-3,q-4} - two
   lanes' constructs are one object - and length-j word survival is q'-2j-2, i.e. normalised
   eigenvalues (q'-2j-2)/(q'-2) vs Holt's (q'-j-1)/(q'-2): the paired system contracts TWICE
   as fast per word length. This is the transfer-matrix frame with KNOWN entries: the
   anti-correlation deficit / p_j is now poseable as a spectral statement over a matrix we
   can write down (Constructor); the gap-pair census is its length-2 input row (Mechanic);
   c_q(g1,g2) is its length-2 block, and the interior disjunction DOES factorise r-wise
   inside coef (Lateral); coef position-freeness + a fixed rung is finite and
   kernel-checkable (Formalist). Unused import flagged for next round: Holt's p-independent
   eigenvector analysis -> closed-form asymptotic paired-gap population ratios (HL
   Conjecture B in paired cycles).

5. LITERATURE STATE (settled this round, full-text reads): ZM prove NO upper bound of any
   strength on j_2 (only elementary monotonicity; no Iwaniec citation; no heuristic for
   p^2-p; no growth commentary; no remark on the 13 case) - the paired analogue of the
   Kanold/Stevens/Iwaniec upper-bound ladder is EMPTY, a zero-published-attempts open
   problem adjacent to our exact table (new candidate N4). Transfer-matrix sieves: Holt is
   alone in the frame (search with -Holt: nothing). One unreviewed Zenodo twin-prime claim
   (Ojaroudi 2026) located and assessed: no recursion, not prior art. h_2 ~ (p^2-p)/2
   empirical share recorded as observation + candidate conjecture, prior-art clean.

Needs: CONSTRUCTOR - state whether (D) is priced per-d in light of item 2; the coef formula
is 20 lines in paired_holt_recursion.py. MECHANIC - a paired-recursion falsification run at
29/31 scale (predict full histogram from old word counts) would be a strong event if COV
work allows. LATERAL - your c_q(g1,g2) = coef of 2-letter words; see item 4. FORMALIST -
coef position-freeness / one fixed rung, finite kernel targets if wanted.

## Constructor round 20

THE TRANSFER-MATRIX DIRECTIVE, EXECUTED HONESTLY - one exact criterion gained, one
spectral hope refuted with structure, one rigorous bound family delivered.

THE EXACT QUALMAX CRITERION (R39 - the round's centre). By the merge law every new gap
is a window sum whose interiors are residue-qualifying (g mod q' in {0, +-2c}), so
EXACTLY: F(M+q') <= max(F2, max_j qualmax_j), and (D) at alpha = 3 follows from
max(F2, max_j qualmax_j) <= F + q' - three census quantities, NO lambda, NO order
statistics: R31's corrected flatness with the heuristic stripped. Measured 7/7 steps
11->13 .. 31->37; the criterion value EQUALS F(M+q') at 6 of 7 (slack 2 at 23->29);
margins 0.52-0.69 q' literal, 0.19 q' at the padded step 31->37. New exact data behind
it: machine-31 FULL spectra F_j = 58, 68, 85, 90, 92, 97 (the envelope job never
delivered these; came free from a 3.34e10-slot cyclic-exact census), machine-31
qualifying runs 502,708 / 508 / 0 at depths 2/3/4, and the complete k=4 fuel inventory
of machine 29: exactly 8 windows, all permutations of {10,10,21}, window sums 47-55 vs
budget 74.

THE RENEWAL LADDER (R38, docs/novel/renewal-ladder.md - rigorous). X20's dropped
"no opening between" condition restored one interior point at a time: for any chosen
interior set Y, #(X exposed, Y blocked) = sum_{T subset Y} (-1)^|T| prod_q c_q(X u T)
- exact CRT closed form, no period scan; nested Y gives a monotone ladder from R32's
exposure bound toward exact. It CLEARS the (D) requirement at every constrained case
including both R32 failures: machine 23 j=6 (was short x28.8) now clears x300; machine
29 j=5 (was short x2.0) clears x91; machine 29 j=6 clears x2000. First joint-gap
bounds at an unscannable machine: p_5(37) <= 3.4e-2, p_6(37) <= 9.8e-3 (period
1.24e12). Honest limits: tightness above exact degrades with machine size (x40 to
x1.8e5); NO zero certificate reached (2^|Y| bars full exactness) - rate bounds only.

THE SPECTRAL HOPE, REFUTED WITH STRUCTURE (R35/R36). In the exact operator frame
(S, D = tensor of D_q, B = I-D, G_v = D(SB)^{v-1}SD) there is NO spectral gap: the
renewal operator is a permutation - decorrelation is an aggregation phenomenon. What
survives exactly is NILPOTENCY: F(M) = nilpotency index of BS, and the qualifying-gap
map A_V is nilpotent of index 2,2,2,3,3,4,4 at machines 11..31 (fuel cap = nilpotency
index; verified by operator iteration). And the AGGREGATED chain is NOT MARKOV: the
exact one-step transfer matrix over-predicts deep qualifying runs by GROWING factors
(x49 at machine 29 depth 3: predicted 391 vs exact 8; x4.4 at 31; size floors
x4.4/x12.6/x40 with depth) - each added link is ~30x more suppressed than the last, so
no fixed-order transfer matrix on gap values can carry the law; the deficit's memory
is longer than any fixed lag. The machine is MORE anti-correlated than any pair-based
spectral bound - favourable for (D), fatal for the naive spectral proof vehicle.
Constants for the record: rho(T_VV)/p_1V = 0.65/0.039/0.20/0.24 (machines 19-31),
|lambda_2| = 0.55-0.66 stable. NEW EVENT: the R33 lag-2 rebound is partly Markov
(predicted 1.27-1.41 vs measured 1.53-1.90) and VANISHES at machine 31 (obs 0.71 < 1
vs pred 1.36) - a regime change at padding onset.

TROPICAL SIDE (R37): F_j <= longest path in the pair-support graph - exact at j=2,
lossy from j=3; the V-subgraph is acyclic at machines 11-17 (pair table alone proves
the depth cap there) but cyclic from 19 on while realized depth still caps: THE DEPTH
CAP IS A >= 3-POINT PHENOMENON from machine 19 onward - no 2-point object (pair table,
corridor, c_q(g)) can certify it. Fits Lateral's completeness lemma from the other side.

NEEDS FROM OTHER LANES: (Mechanic) COV(M)-style exact pattern counter - #(X exposed,
all interiors blocked) cheaper than 2^|Y| IE - is now the named blocker for zero
certificates (qualmax_j = 0 without scan) and for the FLANKED LADDER (add g_L, g_R to
the tuple; certify span + flanks > F + q' to zero = the direct rigorous (D) per step);
also exact residue-qualifying run censuses at 37 would test R39's criterion at the
next step. (Lateral) the multi-lag c_q(offsets) IS your c_q(g1,g2) - the ladder is
its consumer; any closed form for the renewal factor at one lag plugs straight in.
(Formalist) two kernel-ready items: the ladder's validity (finite IE + CRT, per-step
finite statement) and R39's inequality F(M+q') <= max(F2, max_j qualmax_j) (merge law
+ residue necessity - no firing needed). All scripts asserted: tm_resid_runs.py,
tm_transfer.py, tm_tropical.py, tm_renewal_bound.py, tm_nilpotency.py,
tm_qualmax_check.py, tm_deepruns.py; data in research/data/tm_resid_runs.csv.

## Formalist round 20

All three briefed targets landed plus one pickup from Lateral's round-20 list. Build green
at **1276 jobs** (1254 at round 19), zero sorries, zero warnings in owned files; axiom audit
run - every new theorem on the standard three or fewer, and **Machine19.sliceAll (the whole
1,616,615-slot period scan) depends on [propext] ONLY**. Every census input verified against
the research tooling BEFORE formalising. All jobs this round finished before write-up.

1. (A) CLOSED - THE WORD LIST IS NOW KERNEL-CHECKED, BOTH WAYS (proofs/LiteralCapTable.lean).
   The exact per-class literal cap over the 48 invertible classes mod 210: cap_table_maximal
   (NO class admits a run of capC(c)+1) and cap_table_realized (every class realizes capC(c)
   in the corridor) - the table is exact, not just safe. literal_chain_le_capC replaces the
   uniform 6 by each class's own value; word_length_lt_capC is the word form: a literal word
   at gear q' has fewer than capC(q' mod 210) letters, so R21/R26's word list (alternating
   words, two per length, lengths 1..capC-1) is COMPLETE as a function of q' mod 210 alone.
   The census is now kernel theorems: {2: 24 classes, 3: 4, 4: 14, 6: 6}, all four class
   sets explicit, and NO class has cap 5 (no_cap_five). Cross-checked against
   literal_cap_gap_d.py (48/48, zero mismatches) and the fuel census (realized k_max <= its
   class cap everywhere; saturated at q' = 19 and 31). Five-part audit update: (A) FULLY
   kernel-checked - (A), (B), (C) now all closed, (E) closed-but-off-target, (D) open as
   ever. docs/novel/literal-cap.md status upgraded.

2. SUPPRESSION-CORRECTED FLATNESS, HYPOTHESIS-EXPLICIT (Spectrum.lean). Mechanic's Q_j is
   now a formal object over an abstract gap sequence: Qualifying g u a j (the j-2 interiors
   meet the floor 2u') and QualBound g u j Qj. qualifying_of_word: any word whose letters
   meet the floor makes its merged window QUALIFYING - and both alphabets do
   (alphabet_ge_floor: 2u' <= q'-2u'; padded_ge_floor: 2u' <= q'). The payoff,
   merged_le_of_qual_flat_all: Q_j <= F + q' AT EVERY DEPTH gives (D) for every
   floor-respecting word of every length - NO k_win, NO fuel, NO word list in the statement,
   and Q_j = 0 discharges deep depths for free (exactly how mechanic's qspec tables behave).
   merged_le_of_corrected is R31's two-part lambda form with the qualifying hypothesis
   explicit. This is the shape a per-machine census discharge should target: prove
   QualBound instances, conclude (D) at that machine.

3. TIER C RE-ATTACK EXECUTED - MACHINE 19 CERTIFIED (Machine19.lean + Machine19Core + 17
   slice files; 323 slices of 5005 CRT tuples). F_k(19) = 25, F2_k(19) = 31, and NEW:
   F4_k(19) = 38 (quad_sum_le, the depth-4 spectrum value) - F ladder 25/31/35/38/47
   verified over the full period numerically first. alpha1_certificate (9F2 <= 9F + 4q',
   279 <= 317) and lemma1_at_19, mirroring machines 13 and 17. Third machine certified;
   round 15's "tier C caps at machine 19" wall is formally dead (~13 s/slice, ~70 min of
   kernel time total, [propext] only).

4. THE FIRST END-TO-END WIRED INSTANCE (the round's centre). Machine 19's REAL gap sequence
   is now formal: opSeq (the openings in increasing order, built from Nat.find over the
   decidable opening predicate - period multiples witness existence), g19 n = opSeq(n+1) -
   opSeq n, and windowSum_g19 (window sums telescope to opening differences). Then:
       spectrum_four      : Spectrum.SpectrumBound g19 4 38          (kernel-fed)
       spectrum_four_flat : Spectrum.SpectrumBound g19 4 (25 + 23)   (F_4 <= F + q')
       D_of_shallow_word  : l + 2 <= 4 -> merged window of the word <= 25 + 23
   That is (D) at alpha = 3 at machine 19 AS A THEOREM about the machine's own gap word,
   with the flatness half discharged by the kernel scan - the first time the bridge identity
   is wired to a concrete machine end to end. The ONLY remaining hypothesis is shallowness
   (k_win <= 3; census: k_max = 3 at 19->23, winning word (8,15) has l = 2, depth 4 -
   covered). NOT covered: a hypothetical deep word (l >= 3). Measured for the record
   (full-period Python, floor a = 8): Q_4(19) = 37, Q_5(19) = 38, Q_6(19) = 0 - the
   qualifying criterion holds at EVERY depth at machine 19 with margin >= 10, and Q_6 = 0
   is the fuel cap arriving free. A Q_5 <= 48 kernel scan (a 49-step-walk variant, ~2x
   cost) would remove even the shallowness hypothesis - named next-round target.

5. PICKUP FROM LATERAL'S LIST: THE T3 LAW KERNEL-CHECKED
   (LiteralCapTable.tripled_teeth_antipode): {3u', q'-3u'} = {(q'-1)/2, (q'+1)/2} in exact
   integer form, every gear forever (numeric assertion to 100,000 now superseded by proof).
   Status upgraded in docs/novel/golden-spectral-gap.md and docs/novel/literal-cap.md.

NOT TAKEN THIS ROUND (named, with reasons - all real projects, not one-liners):
R39's inequality F(M+q') <= max(F2, max_j qualmax_j) needs the merge law formalized as a
two-machine statement (new gap = window sum of old gaps with q'-killed, residue-qualifying
interiors) - the QualBound vocabulary landed this round is its target language, and it is
next round's top target; the renewal ladder's validity (finite IE + CRT, per step);
Lateral's depth-sum identity at a fixed machine; Harvester's paired-Holt coef rung
5005 -> 85085. Priority after R39: machine 23 overnight (~10 h with the F4 walk; extends
the wired instance to 23->29), then Q_5(19), then depth-sum / coef rung.

INFRASTRUCTURE (details in formalist.md): parallel slice-family builds die on MEMORY -
~5 concurrent kernel-scan processes on this 16 GB machine failed 10 of 16 targets, every
one succeeds standalone; this Lake has no jobs flag, so launch big slice families at most
2-3 targets per invocation. The sorry'd-assembly dry-check pattern (root file with slice
imports swapped for the core and the assembly theorem sorry'd, run under lake env lean)
let the entire 300-line root elaborate before any slice finished - zero kernel cost, root
compiled first try.

## Mechanic round 20

FOLDED ROUND-19 TAIL (all jobs finished after r19 filed): F(2,53) = 435
EXACT - alpha=3's budget (needs <= 513) PASSES with 15% room. MACHINE 37
FULL PERIOD: F(37) = 88 exact; the definitive hole list is 13 values
{73..87 minus 77, 85}; supply(37,41) = 61,460; z>=2 = 0; and THE STEP
RECORD 37->41 IS 91, carried by a k=3 z=1 PADDED run (3,044 of them) -
padding carries the record again, and prefix k_win (which said 1) is a
lower-quality object. kwin31 full period: k_win = 3 at 31->37, record 88
= F(37) by two independent routes (merge law vs direct scan); ZERO k>=5
tuples to depth 8 - the r17 pre-registered test CONFIRMED. L=15 hunt
complete to member 1.2e13: NO L=15 (48 L=13s, 5 L=14s; expectation 0.52,
absence sub-1-sigma). qspec37 at 16%: margin 0.83q' - THE Q_j MARGIN
COLLAPSE DOES NOT CONTINUE at 37->41: the collapse is a litcap-6
phenomenon and 41/43 are litcap-2; next real test is q' = 53.

COV(M) DELIVERED AS CRT+SAT (the named construct, working): every
gap/window/pair/fuel occurrence question is a ~300-var CNF over gear
phases (two classes at separation -2u_q, one free phase per gear, CRT
realises any phase vector); witnesses are CRT'd back to explicit k and
machine-verified. VALIDATED EXACTLY on all 8 scanned machines (m37's 13
holes: 11,829 s scan reproduced in 123 s) and on F_j rows m23 j=2..6,
m29 j=2..6, m31 j=2..5 - witness addresses reproduce the r17 census
maximisers. NEW, BEYOND ANY SCAN: machine 41 (period 5.07e13) COMPLETE:
F(41) = 91, holes {84, 87, 89} (ladder of hole counts 11..41:
0,1,1,2,1,2,3,13,3 - the 37 explosion heals). F_2(37) = 90 exact:
lemma-1 margin at 37 is 2 of 41 available - the deep end keeps getting
easier. Maximal-gap ADJACENCY refuted at 31, 37, 41 (was y <= 23).
FUEL CAPS DECIDED AT FULL PERIOD: N_4(37->41) = 0 and N_4(41->43) = 0
by exhaustive refutation of all legal words - k_max = 3 at both steps.
THE FIRST DOUBLE-PADDED RUN EXISTS: word (43,43) at 41->43, witness
k = 116,431,845,582 - the r16 "plausibly decidable only structurally"
prediction, decided, with an address; gap 86 = 2q' also occurs (first
2q' padding anywhere). Standing bounds, resumable tools: F_3(37) in
[97, 163] - the (D)-decision at 37->41 (needs <= 129) is 34 refutations
away at ~15 min each, next round; F(43) >= 103 with 104 refuted, tail
[105,118] open; F(47) >= 118; F(53) <= 145 pinned by F(2,53).

THE CORRIDOR RESONANCE (new law, measured, docs/novel/): the
qualifying-gap indicator's autocorrelation is a barely damped WAVE -
m29 floor 10, lags 1..15: 0.801 0.684 0.510 0.800 1.112 1.257 1.204
0.995 0.781 0.717 0.848 1.082 1.254 1.250 1.094 - whose period is
35/mean_gap at every machine. Mechanism measured directly: big-gap left
endpoints are PINNED mod 35 (invariant core {10,12,18} at all five
machines; exact four-way ties at m17/m19) and their slot-separation
autocorrelation peaks EXACTLY at 35/70/105 (up to x4.4 at 70). The
gap-pair "deficit at lags 1-3, excess at 4-7" is this one phenomenon.
CONSEQUENCE FOR CONSTRUCTOR'S TRANSFER MATRIX: the process is NOT
k-step Markov for k <= 4 (exact factorisation TV 0.15/0.13/0.09/0.08 at
m29), and the value-level one-step chain predicts NO deficit at lags
2-5 where the census says 0.51-0.68 - the state must carry corridor
phase (mod 35 at least), not the last gap.

MATRIX MEETS COMPLEX FRAME: the subleading eigenvalue of the measured
lag-1 transfer matrix is real negative with |lam2| = 0.6273, 0.5959,
0.5722, 0.5583, 0.5515, 0.5462, 0.5425 (m13..m37); its distance to
Lateral's golden gap phi/3 = 0.53934 shrinks geometrically (factor
~0.6/machine). kappa(2) = 0.5448 is passed and dead as a limit;
convergence to phi/3 is CONJECTURED on 7 exact points, no fit. DFT
identity events all check: c_q(g) = inverse transform of the exposed-set
power spectrum (376 checks, 0 mismatches); corridor census = c5*c7 =
product-spectrum inverse DFT (35/35). The C14 gap-histogram residue law
has a machine-independent PHASE: arg H_5(1) = +126 deg +- 2 at all
seven machines (not the +-s ripple's 0/180) - UNEXPLAINED, for Lateral.

p_j FOR CONSTRUCTOR (deliverable a): full-period joint gap-pair census
at lags 1-5 and run-min m=2..6, machines 13..31, + machine 37 at 12.9%
(28.19e9 gaps): research/data/gap_pair_{hist,joint}.csv +
pj_deficits.csv. At each machine's own floor, p_m/p_1^m at m=3 is
0.15-0.32 (x3-x6.7 deficit) and x18-x190 at m=4-5, stable across
23..37. The deficits' lag structure and their non-Markov character are
the corridor resonance above - the anti-correlation law's carrier is
corridor phase coherence, exactly computable from c_5 x c_7.

HONEST NEGATIVES AND INCIDENTS: envelope31/37 died mid-scan (sweep) -
their one pre-registered claim was settled by kwin31 instead; the
envelope41 log's "(D) PROVED" line is INVALID (tiny prefix, F wrong by
18) - treat as prefix word data only. The exact F_3(37)/Q_3(37) and the
machine-43/47 tail refutations are 1.5-2 h PER INSTANCE at 12+ gears -
not reachable this round; per-S/per-v checkpoints are in the logs and
every tool resumes. A process sweep killed 15+ jobs in waves all night
(r19's phenomenon again); after per-instance logging + resume flags
were added, nothing further was lost. fuel_census/gap_pair/bool_lag now
take --start for sliced, checkpointed runs.

OWED / NEEDED: Constructor - pj_deficits.csv + the corridor-state
requirement (mod-35 phase in the transfer matrix); the exact pair
support (which (u,v) adjacent pairs occur) is SAT-decidable per pair on
request (cov_sat.py pair command). Lateral - the pinning classes
{10,12,18} mod 35, their exact ties/breaking, and the +126 deg
histogram phase, all corridor objects; also lam2 -> phi/3 (your golden
gap) as the conjectured limit of a measured matrix. Formalist - two
kernel-checkable finite claims now exist with witnesses: fuel-cap
refutations (N_4 = 0 at 37->41, 41->43: finite word lists, finite CRT
enumeration each) and the double-padded witness k = 116,431,845,582
(pure arithmetic check). Harvester - F(41) = 91 and the hole ladder are
new exact values in the F(2,y)-family (F(2,41) = 273); ZM-style tables
have no analogue of the hole lists.

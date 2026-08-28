# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State after round 24 - the machine-free round, run across a three-day API outage that split
every lane, with two lanes lost mid-round and completed from their disk state. Despite that:
the two-gap obligation was answered in BOTH directions (negative proved, positive verified
exact), a FIFTH (D) rung landed hypothesis-free in the kernel, the SDP door was closed by
theorem, the j_2 paper's exponent fell 19 -> 15 with a first lower-ladder theorem, and the
round's infrastructure failure produced the project's clearest cost-model lesson. Every lane
either corrected itself or was corrected by a gate.

MANAGER GATE-CHECK (first run of the standing rule; from clean processes, 2026-08-28/29):
- twogap_table.py (Constructor R55): GREEN - "OK - all assertions passed".
- m37_count_audit.py + junction analysis (Mechanic): GREEN - spectrum holds full-period,
  no junction caveat; COV-SAT cross-agreement printed.
- a_kill.py validate (Mechanic): GREEN - SAT enumeration N_3(37->41) = 3052 == scan, N_4 = 0.
- j2_referee.py (Harvester): GREEN - "ALL ASSERTIONS GREEN".
- lp_degree_range.py section G (LP thread): CAUGHT A CLAIM/GATE DESYNC - the lane corrected
  its round-23 thresholds (31 -> 30 at m17, 37 -> 35-or-36 at m19) but left the gate asserting
  the OLD values, so the gate fired against the corrected truth. Gate updated with provenance;
  rerun GREEN on all four corrected thresholds - AND THE RERUN SETTLED THE ROUND'S OPEN CELL:
  W*_indep(19) = 36 EXACTLY (35 feasible, 36 infeasible, 661 s bisection; the round-24 report
  had it as "35 or 36, deciding run starved").
- sdp_cover.py (Lateral): lane-verified in-round from a clean process (exact rational duals,
  nothing rests on floats); manager rerun stopped by the human mid-run - recorded as
  LANE-VERIFIED, NOT MANAGER-VERIFIED.

THE TWO-GAP VERDICT (constructor R55-R59, filed by the manager from the lane's completed
drafts after the agent was lost - text verbatim from research/data/r24/r24_draft.md):
- R55, THE NEGATIVE HALF, QUANTIFIED: both machine-free suppliers of the two-gap fact
  saturate at 2F. The histogram supplier is exactly killed by the MIRROR LAW (mirror-paired
  maximal gaps make F + G_2 = 2F tight), and by Lateral's Jordan=histogram theorem that kills
  every unitary invariant; the corridor layer-0 column equals 2F or 2F-1 row-for-row. Over
  budget from 19->23 on. The two-gap statement F_2(M) <= F(M) + q' is measured exact at all
  seven full-period machines with slacks +9, +12, +12, +17, +24, +19, +27 (m11..31).
- R56, THE SURVIVOR IDENTITY (positive half): F_2(M+q') = L (x) K* (x) Sigma (x) K* (x) R in
  max-plus - the two-gap object has its OWN generator with one skip transition through the
  unique surviving opening. Verified exact at every full-period step against the independent
  pair census (16, 25, 31, 39, 55, 68) - 6/6, with the m31 pass re-launched and finished
  post-outage. New doc: docs/novel/survivor-generator.md.
- R57: A_4 certifies the NEXT step's two-gap budget at every step; A_5 restores exactness
  where A_4 is loose (39, 55, 68 exact) - A_5(23) delivers the literal R53 integer
  F_2(29) = 55 WITH NO MACHINE-29 SCAN.
- R58: CEGAR WITH STATE+EDGE REFINEMENT NEEDS NO GIVEN INTEGER - 181/90/955 queries certify
  (D) at 19->23 / 23->29 / 29->31; edge-only controls stall at exactly the 2F wall. Slack
  sweep: every U <= 74 certifies, U in [75,85] stalls at U - THE OBLIGATION IS EXACTLY THE
  TWO-GAP STATEMENT, AND IT DESCENDS.
- R59, predictions scored: P1 confirmed (6/6), P2 half-refuted, P4 refuted (threshold =
  budget), P6 refuted (mechanic's [2,88] witness), P3/P5 undecided.
- NAMED NEXT CONSTRUCT: the scan-free dictionary - answer the 90-955 CEGAR queries by R43's
  pruned-IE CRT counter (every query is arity <= 5, span <= F_2, the cheap end of its cost
  curve), then CHAIN STEPS: A_5(23) -> F_2(29) -> certify 29->31 -> A_5(29) -> F_2(31) -> ...

THE FIFTH KERNEL RUNG - 23->29 HYPOTHESIS-FREE (formalist; lane lost mid-round, round
completed and independently re-verified by a successor agent from disk state):
  theorem D_23_29 (n : N) : Machine29.g29 n <= 34 + 29     -- proofs/Machine23Scan.lean
via spectrum23_one/two (F=34, F_2=39), qual23_all (QualBound g23 5 j 60), and qsliceIdxAll -
323 slices x 5005 tuples x 23 phases = 37,182,145 residues each exactly once; the scan
certifies its own fuel per clause, importing no outside bound. qsliceIdxAll is [propext]
ALONE; the rest standard three. BUILD GREEN AT 1372 JOBS, zero sorries, no native_decide.
THE MEASURED FIX AND TWO SELF-CORRECTIONS: position-indexing beat the round-23 encoding by
>= 22x PER SLICE (65-81 s vs >= 1780 s aborted, controlled A/B) - not the predicted 5x; end
to end 3 h 36 min vs the 13 h estimate. The finisher's own draft said "~0.4 GB per worker" on
plausibility - MEASURED 5.38 GB, WRONG BY 13x - and that number is the whole story of the
infrastructure failure: round 23's cost model had NO MEMORY COLUMN, so a six-way parallel
launch (32 GB of commit on a 16 GB box) looked reasonable and produced TEN AND THREE-QUARTER
HOURS OF ZERO COMPLETED SLICES (page-thrash is a LIVELOCK, not a slowdown). Serialised to 2
workers by research/lean_babysitter.py (suspend-not-kill + EmptyWorkingSet + RAM-scaled
runner count - the reusable artifact), the same work finished in 3 h 36 min. Also caught: its
timing wrapper wrote a finished-looking number for a KILLED run (recorded as bound, not
measurement); round 23's slice count was 7,429 not 7,434. PARALLELISM BUDGET FOR LEAN SCANS
IS 2. THE PER-RUNG SCAN VEHICLE ENDS HERE: the same factorisation at 29->31 costs ~170 h
(7,429 outer slices, no reusable family, 29-fold inner loop) - the sandwich-lemma route and
Mechanic's dictionary-transfer superset (which converts A_4-at-29->31 into a longest-path
decide over an explicit digraph) are now the required path. Scaffolding with a stub axiom
(DryScan2.lean, never in the build) deleted at close.

THE SDP DOOR CLOSED BY THEOREM (lateral): LP(MF_m) = the max-plus closure EXACTLY (12/12 at
all seven steps, integer-equal), hence EVERY convex relaxation sandwiched between LP and
truth - every Lasserre level, every SDP - returns exactly the closure value. The 125-vs-74
gap at 29->31 is 100% edge-set, 0% relaxation gap. The covering-side SDP that CAN exist is
exact at m11/13/17 (exact rational duals 479/1152, 1041/2081, 1673/19767), BREAKS AT m19
(L* = 27 vs F = 25), and PSD does not repair it - every certificate of F(19) <= 26 needs
three-gear information: the sharpest finite instance of the arity law. Vacuity ratios 1.000,
1.000, 1.000, 1.080, 1.647, >= 1.721 at m11..29 - a THIRD independent certificate family
obeying the arity law. Secondaries closed by reduction: the norm-cliff envelope constant is
identically 2^osc(h) (histogram-circular); the Maslov bridge transports bounds at equal
oscillation (isomorphism, not amplifier). Predictions: 5 confirmed, 2 refuted (both their
own), 1 unresolved.

MECHANIC (split by the outage; sanctioned undecided-with-checkpoint outcomes):
- A_kill handoff BUILT AND ANCHORED: a_kill.py enumerates kill-chain words by three theorems,
  decides each by CRT+SAT with assert-verified witnesses; reproduces N_3(37->41) = 3052 and
  N_4 = 0 exactly. DECIDED SO FAR: A_kill(43->47) >= 3 (5 realised words incl. double-padded
  (47,47)), A_kill(47->53) >= 3 (11 realised incl. (53,53) - THIRD consecutive step with the
  z=2 shape). Restoration thresholds exact: k_max <= 5 restores 43->47 (fails at J=7 only),
  k_max <= 4 restores 47->53 (fails at J=6,7). Nothing seen contradicts k_max = 3.
- THE m37 DISCREPANCY RESOLVED - THE SCAN WAS RIGHT, THE LABEL LIED: fuel_census.report()
  ignored --start; three chained runs tile the period and their counts sum to prod(q-2) =
  217,929,355,875 EXACTLY. Self-caught bonus: junction-straddling windows were seen by NEITHER
  run - repaired by direct junction examination (worst straddler 61 vs F_6 = 120). New
  standing rule 18.
- Dictionaries: m23 (15,696) and m29 (45,854) 4-tuples double-derived byte-identical; the
  DICTIONARY TRANSFER built - certified SUPERSET by order-m closure (0 tuples missing at both
  validation steps, inflation 4.15x/6.21x), 31->37 superset delivered (2,435,140 tuples) -
  exactly Formalist's hE : realised subset E shape and sound for the max-plus closure.
- F_3(41) = 110 EXACT and F_3(43) = 125 EXACT (first computations) via the floor-1 lap-phase
  transfer - which also explains the r23 SAT stalls as boundary refutations above the true max.
- Honest: the criterion repair itself is not delivered (k=4 finish checkpointed); commit
  exhaustion killed jobs twice (fixes now policy); a kill attempt on old memory hogs was
  correctly blocked by the permission layer and reported.

HARVESTER (paper lane):
- OPERA DE CRIBRO CONFIRMED from the book's own text (OCR of the AMS printing; caveat stated):
  Thm 7.7 statement, hypothesis, bracket and tau_4 remainder all match both transcriptions -
  three agreeing renderings, one the book itself. (7.122) is a loose sufficient condition
  (21.6 vs exact s* = 18.308); Cor 7.8 buys nothing at k ~ 3.1. Openings: ODC Ch.6 prints
  beta_2 = 7.5941 (explicitness unknown - THE decisive question), Blight's thesis unobtained.
- HR MEMOIRE OBTAINED (Mem. SMF 25 (1971), numdam scan): treats EXACTLY our density (worked
  example A = {n(n+2)}). The 7.972 exponent is DERIVED not printed - re-derived from the
  paper's two printed conditions and asserted (lambda* = 0.2533219, u = 7.9719548). All
  remainders are O(.): the exponent-8 route is an EXPLICITNESS problem in a known 1971
  theorem, converging with the ODC Ch.6 lead on one object.
- 19/36 vs 0.4454 SETTLED FOR 19/36: Selberg's own announcement (Greaves' review, zbMATH
  fetched first-hand) plus exact-rational re-derivation (a = 1/4, d = -7/72, constant exactly
  19/36); at 2kappa+0.4454 the functional's main term is negative. 0.4454 recorded unverified.
- EXPONENT 19 -> 17 FREE (Thm 2E', pre-sieve 2 and 3) -> 15 AT CONSTANT COST 135 (Thm 2E''),
  and 15 is PROVED THE FLOOR of FI 7.7 at kappa = 2 (s* -> 14.169 as K -> 1).
- THE LOWER LADDER EXECUTED, correcting the lane twice: THEOREM (P1) h_2(P(z)) >=
  (1.349+o(1)) z log z - the first lower bound using the paired structure (smooths are covered
  free, so only z-rough numbers need covering: one log thinner than the ordinary problem);
  certificates sieve-verified at z = 13..1e5. RETRACTED: round 23's "truth ~ p^2/2" (its own
  data does not support it - z^2 and z log^2 z fit ZM's table equally) and the round-23
  capacity argument (capacity is not scale-free) - so the round-23 open problem
  "h_2 >> p^{1+delta}" asked for something probably false and is REPLACED by the
  Rankin-layering and paired-Iwaniec problems. Two draft overstatements caught by their own
  gates in-round. Prior art: Kalmynin-Konyagin 2302.00459 nearest, no overlap. Decidable
  falsification target: one exact h_2 beyond p_n = 73 (models differ 2.6-3.6x by z = 151-251).

LP THREAD:
- THE REGRESSION PINNED AND CORRECTED (the round's must-have): round-23 section-G claim
  failed its own gate; truth is W*_indep(17) = 30 (not 31; independently re-verified by
  from-scratch code) and W*_indep(19) = 36 (settled exactly by the manager's gate-check rerun:
  35 feasible, 36 infeasible). Round-22's 8/21/31/37 stand only as KOUNIAS-FAMILY thresholds;
  the adaptive degree-2 cuts ARE strictly sharper. No other lane consumed the wrong values;
  the Lean certificates are unaffected. Cause honestly not reconstructed (round 23 did not
  save its feasible point) - NEW PROCESS RULE: FEASIBLE VERDICTS MUST SAVE THEIR WITNESS.
  Every standing verdict re-run through a hardened gate (exact re-assertion of A.nu = b,
  nu >= 0 on every extends verdict); all survive. The m17 degree-4 blank cell FILLED in 13 s
  ("fails" - feasible, completion exhibited) by a one-shot completion LP where round 23's cut
  loop ran 45 min without settling.
- THE COMPOSITION (consistency x Costello-Watts recursion): width and vacuity-escape YES,
  machines NO (as pre-registered; E1-E4 held, E6 split - the split is the finding: the row
  alone wanders, the full composition stays FLAT at 1.000/1.273/1.278/1.320). Same four (D)
  rungs proved with certificates 2-3x SMALLER (562/1,456/3,303/8,179 ops). NEW WIDTH at m19:
  the composition certifies width 33 where NO degree-2 cut certificate of any kind exists.
  E4 STRONGER THAN PREDICTED: the row cuts the uniform product measure THROUGH MACHINE 37
  (+3.46/+3.27/+2.01/+0.41 at m23/29/31/37; lost at m41) - the first certificate object to
  survive the product measure past the degree-2 vacuity ceiling. CONSEQUENTLY 23->29 MOVES
  FROM REFUTED TO OPEN. No new rung: 19->23 undecided (+0.037 converged, no certificate);
  the 23->29 decider was starved by other lanes' jobs (188 MB free, 28 workers).

MANDATE AUDIT (standing rule; drift named as the manager's own coordination failure):
- CONSTRUCTOR: on mandate (owns the route). MECHANIC: on mandate (censuses). HARVESTER: on
  mandate, and its best work is there. FORMALIST: on mandate. LP THREAD: a SANCTIONED
  dedicated route thread - correct location for route work.
- LATERAL: three consecutive rounds (22-24) on certificates/relaxations of the live route's
  system. Each brief had a native-frame justification (spectral, Jordan, SDP are its tools),
  but cumulatively this is "second analyst on the live route", which the mandate forbids.
  THE BRIEFS CAME FROM THE MANAGER; the drift is the manager's. ROUND 25 RESTORES LATERAL to
  genuinely lateral territory: its own backlog (m29 depth spiral, the 1.015/mean-gap
  amplitude law, gear-7's drifting bracket, Farey-spectrum consequences, the 613 cosine
  near-collisions) - and route-serving ideas from Lateral route through the manager.

INFRASTRUCTURE LESSONS OF THE ROUND (all now standing policy):
- MEMORY IS A LIVELOCK RISK, NOT A SLOWDOWN: 10.75 h of six-way lean building produced ZERO
  slices; the same work serialised finished in 3.6 h. Cost models MUST carry a memory column.
- Lean scan parallelism budget on this box: 2. Commit (not working set) is what exhausts -
  suspended processes park their full commit; Task Manager's default column hides it.
- research/lean_babysitter.py is the reusable guard (suspend-not-kill, EmptyWorkingSet trim,
  RAM-scaled runner count).
- The venv is at C:\dev\primes\.venv; invoke interpreters by absolute path.

ROUND-25 (briefed, NOT launched - runs under the MODEL POLICY: lanes on Opus, manager on
Fable, manager gate-check before write-up, escalation valve for derivation-grade reasoning;
spine = CHAIN THE LADDER AND DERIVE THE TWO-GAP LAW):
CONSTRUCTOR (Opus) -> execute your own named next construct: the scan-free dictionary -
CEGAR queries answered by the pruned-IE CRT counter, then CHAIN: A_5(23) -> F_2(29) ->
certify 29->31 -> A_5(29) -> F_2(31) -> ... Each chained step is a machine-free-INPUT
certification (only the previous machine's word is consumed). Report exactly where the chain
first needs a fact it cannot generate - that residue is the two-gap law's irreducible core.
ESCALATION VALVE: if the mirror law + kill-spacing + survivor identity start combining into
an actual derivation of "no big-next-to-big", STOP and hand the derivation to the manager.
MECHANIC (Opus) -> finish the A_kill k=4 decisions at 43->47 and 47->53 from the checkpoints
(the criterion repair depends on them); the m37 exact 4-tuple dictionary (3 ranges to rerun,
commands in handover-mechanic.md); then the m41 dictionary superset for the chain.
LATERAL (Opus) -> YOUR OWN MANDATE, restored (see audit): pick from your own backlog. The
one route-adjacent item you may keep, because it is yours and nobody else's: the mirror law
(openings closed under k -> -k) as a STRUCTURAL constraint - what else does exact mirror
symmetry force about adjacent gaps? That is a reframing question, not route support.
FORMALIST (Opus) -> (1) A_4 at 29->31 via Mechanic's dictionary-transfer superset (the hE
shape; longest-path decide over an explicit digraph - the sixth rung, and the first by the
new vehicle); (2) Constructor's A_5(23) closure = 55 as the cheaper alternative input; (3)
the 11->13 survivor identity (= 16) as the first kernel statement of the generator; (4) the
m13 covering dual (integer check, 793 rows). The per-rung scan vehicle is DEAD (170 h) - do
not attempt a 29->31 scan.
HARVESTER (Opus) -> Unit 1 to submission: obtain Blight's thesis (Rutgers 2010) and decide
the ODC Ch.6 beta_2 = 7.5941 explicitness question - the two named openings; re-run
j2_referee.py before any claim. Then the Rankin-layering problem on the lower ladder.
LP THREAD (Opus) -> the product-measure-surviving row is the first object past the vacuity
ceiling: push the composition at 23->29 on a QUIET box (the round-24 decider was starved,
not refuted); map where the row dies (m41) and why; new process rule applies (save every
feasible witness).
MANAGER (Fable) -> gate-check before write-up; the two-gap derivation attempt if the
escalation valve fires; compaction maintenance.

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

MODEL POLICY (standing - from the human, 2026-08-28, for round 25 on):
Lane agents run on OPUS; the manager (round write-ups, cross-lane routing, mandate audits,
novelty verdicts, round briefs) runs on FABLE. Rationale, from the rounds-23/24 experiment: lane
work is build-and-test, where the assertion gates and the Lean kernel do the guaranteeing - a
gated claim cannot drift silently. The compensating rules are therefore LOAD-BEARING, not
etiquette:
- Every claim must pass an assertion gate, the kernel, or an exact census before it enters the
  record. Ungated prose is not a result.
- Every NEGATIVE ("this route won't work", "not worth pursuing", "will not close") must carry a
  proof, an exact measurement, or the explicit label "JUDGMENT, NOT RESULT". Negatives are the
  one claim type with no gate to fail - label them or gate them.
- Pre-register predictions where possible; score them in the round report.
- MANAGER GATE-CHECK (new, standing): before each round write-up the manager re-runs every
  lane's headline assertion gate from a clean process and reports the result in the SUMMARY.
- ESCALATION VALVE (new, standing): if a lane hits an item where the REASONING ITSELF is the
  deliverable - e.g. the two-gap mechanism derivation beginning to crack - that item is pulled
  up to the manager (Fable) rather than finished in-lane. The finish-line derivation is not
  attempted on the budget model.

COMPUTE POLICY (all agents, standing - from the human, 2026-08-28):
The box has 20 CPU cores and ~16 GB RAM. Scripts SHOULD run multi-core where it makes sense and
is safe - but leave headroom: keep the TOTAL load across all lanes at <= 16 cores so the Windows
UI and VS Code stay responsive. Practical defaults: a lane doing heavy compute takes up to 8
workers when the box is quiet, 3-4 when other lanes are active; check before launching
(Get-Process count / CPU load), don't assume. MEMORY IS THE BINDING CONSTRAINT, not cores - the
project has now hit fork-table exhaustion once and pagefile/commit exhaustion twice (WinError
1455). Cap concurrent child processes (pool <= 3 for memory-heavy work), stagger launches, retry
Popen failures, and make every orchestrator resume from its own log. Use whatever tools fit the
job best; invoke interpreters by absolute path (venv activation does not persist across shells).

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

## Round findings (rounds 1-24)

Compacted 2026-08-23 (rounds 1-19) and 2026-08-29 (rounds 20-24) by human directive. Full
verbatim round-by-round appends are preserved in `docs/proof-search/archive/` as
`agents-shared-full-r1-19.md` and `agents-shared-full-r20-24.md` (nothing deleted; also in git
history). Cumulative findings live in each workstream's own doc (also compacted, verbatim
copies in `archive/`). New round appends go below this line as before.

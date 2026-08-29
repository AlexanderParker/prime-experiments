# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State after round 27 - THE ROUND THAT LEFT THE CORPUS. (D) is now decided TRUE at 53->59, a
step where NO upper bound on the new machine's record existed anywhere - computed on machine
23's period for a property of machine 59 (period ratio 5.3e11) through the record law. The
uniform-order meta-residue fell to a machine-free theorem. The increment law survived a
pre-registered test at a padded step it had never seen. The LP certificates entered the
kernel, giving a hypothesis-free D_29_31. And the human's sort-step idea was proved, honestly
deflated by its own prior-art-and-counterfactual analysis, and paid off sideways with the
project's first quantity distinguishing the real machine from its counterfactuals.

MANAGER GATE-CHECK (2026-08-31/09-01, clean processes): constructor uniform_order.py GREEN,
increment_law.py GREEN; lateral lex_odometer.py 145/145 GREEN; LP emit_certs_r27.py GATE
GREEN (3 s), increment_cert_r27.py GATE GREEN (42 s); formalist lp_cert_lean.py ALL PASSED
(incl. 400-random-tuple soundness gate); harvester j2_referee/citesweep/odcpages GREEN
(in-lane, clean); mechanic two-sided decider gates GREEN (in-lane); lake build GREEN AT 1521
JOBS, zero sorries, 367 audited footprints all standard-three or smaller.

(D) BEYOND THE CORPUS (mechanic): max_J Q*_J(53; legal for 59) <= 203 < 204 = F(53) + 59 -
(D) TRUE AT 53->59, the first step past the corpus ladder's end. Four empty bands + one
exhibited witness give 161 <= F(59) <= 178. THE INCREMENT LAW CONFIRMED at this never-seen
PADDED step (pre-registered genuine bet: predicts <= 179, measured <= 178) - so the law's one
failure (31->37) may be specific to that step, not to padding. Also: A_kill(53->59) = 4 EXACT
at the cost of ONE solver call (the refuted-span band table killed 63 words solver-free);
shape is the PADDED alternation (20,98,20); A_kill - ceiling now -2,+1,+1,+1,0,+1 across six
steps. Q_J(37;14) J=2..8 delivered with witnesses (reproducing the round-23 row four rounds
later; J=8 TURNS OVER; Constructor's lost Q_7 = 112). Predictor scorecard 4-3-1 with the
honest line "I extrapolated a pattern instead of computing it."

THE UNIFORM-ORDER THEOREM (constructor): A_relax(M) <= 5 AT EVERY MACHINE, <= 4 unless
q' = 37, 53, 83, 127, 157, 173 (mod 210) - proved by phase saturation at GEARS 5 AND 7 ONLY,
hence machine-free; all 48 classes enumerated; cross-checked against every prime q' < 20000.
THE SIX EXCEPTIONAL CLASSES ARE EXACTLY R20'S LITCAP-6 CLASSES - phase saturation at {5,7} IS
the literal cap's arithmetic under a different quantifier; two invariants five rounds apart
are one object. THE COMPANION NEGATIVE, sharper: the correctly-posed order question is N(M)
(every legal cycle broken); R49's N = max(2, A_relax) REFUTED at m37 (padded cycle); CORRCAP
goes INFINITE from 53->59 on - no fixed small-gear set caps the order the chain needs; the
open question is the COVER half of the CSP. Self-correction: R45's A_relax(37) was a
hardcoded value in their own script (3 -> 2); corrected ladder 1,2,2,3,2,3,4,2,2.

THE INCREMENT LAW AND THE TRIPLE INEQUALITY (manager + constructor + LP + mechanic):
- Now holds at ELEVEN OF TWELVE testable steps + the 53->59 confirmation; only 31->37 fails.
- THE TRIPLE INEQUALITY (manager's reduction: adjacent gap triples (a,w,b) of M with w = +-s
  mod q' satisfy a+w+b <= F_2(M) + s_min) holds at all eight censused steps FOR LITERAL
  MIDDLES INCLUDING 31->37 (70 <= 80 there - the padded middle alone carries that step's
  failure), and is CERTIFIED AT A NINTH STEP 41->43 SCAN-FREE by CRT (literal <= 107, padded
  <= 116, both < 117, zero undecided).
- CONSTRUCTOR'S FREE REDUCTION: span <= F_2 + min(g_L, g_R) with no hypothesis discharges 6
  of 8 steps outright; the residual Delta_3 = max(span - F_2) IS BOUNDED BY A CONSTANT
  (-3..4, no trend) while s_min grows linearly - THE DERIVATION TARGET IS Delta_3 = O(1),
  sharper than the s_min form.
- LP THREAD: THE INCREMENT LAW IS CERTIFIED AT ALL SIX LITERAL STEPS THROUGH 29->31 BY
  CERTIFICATE + WITNESS, NO PERIOD SCAN ANYWHERE - the dual half at the tighter increment
  width (strictly harder than the rung), the realisability half by scan-free CRT witnesses,
  each tight. The manager's induction has its base cases BY CERTIFICATE. New finding: TWO
  frontiers (width - closed form; cut-loop convergence - new open question, the 41->43
  increment width fails by a decelerating loop, not by width).

THE LP CERTIFICATES ARE IN THE KERNEL (formalist + LP): CaseCert23.D_19_23_case and
CaseCert31.D_29_31_case - HYPOTHESIS-FREE (D) RUNGS from case-split certificates; D_29_31 now
exists in TWO kernel forms (census-hypothesis and none). Three of the four feared soundness
lemmas were VACUOUS AT THE ARTEFACTS (every cut row of all 75 certified cases is the BASE
CUT - the LP thread's emission discovery, which collapsed the validity obligation from a
2^n subset-sum to ">= 1 by inspection"); exhaustiveness was one omega; the real work was the
recursion row via the lowest-blocker identity as a 2^m Boolean decide. Kernel wall = CASE
COUNT (k=3 ~ 8.6 h, k=4 ~ 5 days), not columns. Emission landed early, gated, independently
transcribed (agrees as exact rationals). THE LP LADDER IS TEN RUNGS (41->43 k=3 finished
385/385, incl. all six round-26 stalls - "budget-limited, not blocked" was right); the
vehicle is TIGHT ON F at four machines. MIRROR LEVER'S COUNTING CORE KERNEL-CHECKED
(even_card_involution, window_count_even, none_of_at_most_one - the "at most one implies
zero" corollary); machine instantiation is round-28 item 0.

THE SORT-STEP ARC CLOSED HONESTLY (lateral, the human's idea): THE 2n-GAP LAW IS PROVED
(mixed-radix odometer, carry argument, closed-form multiplicities, explicit van-der-Corput
bijection) - then DEFLATED BY ITS OWN ANALYSIS: prior art KNOWN IN MECHANISM (Langevin's
lex-successor theorem recovers three-gap; Fried-Sos generalise), and the gated counterfactual
test shows the 2n count is INSENSITIVE TO THE TEETH (60 re-choosings: always 2n, while F
moves over [10,18]; holds for non-prime moduli) - the phase coordinates discard exactly the
arithmetic F depends on; "F as a shuffle statistic" REFUTED (order vs metric). Closed line.
THE SIDEWAYS PAYOFF: THE TOOTH-COUNTERFACTUAL FAMILY - keeping gears, mirror symmetry and
survivor count fixed and moving only the teeth (exhaustive: 30/180/1440/12960 sievings), THE
REAL TWIN MACHINE IS A LOW-F OUTLIER at the 20.0/18.1/26.4/17.1 percentile (m11..19), in a
family whose max runs 1.6-1.9x the truth - THE FIRST QUANTITY ON WHICH THE REAL PHASE VECTOR
IS DISTINGUISHED, in the favourable direction. U13 routed to the manager: the same family as
a NULL MODEL for (D) - the twin's percentile in the budget-slack distribution; if favourable,
the first-moment transfer has measured room it is not using. Also: U5 closed by two lines of
field theory (real cyclotomic fields of coprime conductor are linearly disjoint => the
degeneracy group is exactly (Z/2)^#gears, NO accidental collision exists at any machine -
round 21's law upgraded measurement -> theorem); U6/U9 closed (-1/phi is a CROSSING between
m29 and m31, refuting their own item 48; the amplitude plateau oscillates in [1.0100,
1.0193]). Scorecard 9-4 with three of six real bets lost - "the shape last round's 10/10
lacked".

HARVESTER - UNIT 1 CLOSED OUT, MEMO ON THE HUMAN'S DESK: all three page caveats closed
first-hand (the fetch failure was a cookie-session issue - new hygiene clause 9: a page that
would not fetch is not a page that cannot be fetched); (5.38) explains the whole pre-sieved
K-ladder in one line (K >= (1-g(p))^-1); (6.69) was never at risk (positivity, not (6.69),
stops Chapter 6 - absolute floor 7.229 < our 7.937); p.74's implied constant means OUR N_pre
accounting is the only explicit route. THE SUBMISSION MEMO (docs/novel/unit1-submission-memo.md)
DOES NOT RECOMMEND SUBMITTING: strongest (empty nine-year ladder four explicit rungs deep
with the explicitness boundary itself proved; P2' genuinely new and parity-free structurally;
a self-auditing paper), weakest (audience: the anchor paper has ONE citation in nine years),
venue tiers, the write-to-Ziller-and-Morack suggestion, AI-disclosure facts laid out AND THE
DECISION LEFT TO THE HUMAN. (P6) WRITTEN with a FIRST EVALUATION: j_3 = 6, 24, 78 (covering
restatement reproduces A048669 at k=1 and ZM at k=2); uniform ladder in k; conjecture
j_k = x(log x)^{2k-1+o(1)}. Lower ladder's k>=4 defect resolved AT ZERO COST (colliding
primes sit inside the Eratosthenes layers where collisions cost nothing). New hygiene clause
10: a hypothesis cited by number is an unread hypothesis.

MECHANIC'S CENSUS PRICING (the finding IS the price): the exact m41 4-tuple census costs
>= 1,121 CORE-HOURS by the CRT route (round 26's ~85,000 core-s figure was for the HISTOGRAM,
a strictly easier object) - a multi-round object, deliberately not started. Delivered
instead: F_4(41) = 118 EXACT (first computation, 602 core-s) - and since a realised 4-tuple's
span is <= F_4 BY DEFINITION, 58.8% OF THE ARITY-4 SUPERSET FALLS BY THEOREM (4,239,676 ->
1,747,819; induced 3-tuples 130,942 -> 95,331); the exact shard complete at every span <= 77
(338,855 tuples, checkpointed); the two-sided decider gate green at three machines. The
INFLATION ONSET IS SHARP AT SPAN 68 - a fact about the order-4 closure. RUNG NINE remains
oracle-blocked; the screened+exact-shard oracle is the round-28 path.

SELF-CORRECTIONS THIS ROUND (every lane again): constructor's hardcoded A_relax entries and
two repeated process mistakes owned; mechanic's extrapolated predictor refuted + the
conflated Q_7 premise unpicked + four campaign restarts owned; lateral refuted its own item
48 and three predictions; LP refuted its own cost model (k=4 predicted, k=3 sufficed) with
"the sentence I had already written about other people's work"; formalist discarded 20 min
of possibly-raced build ("a kernel claim on a possibly-raced olean is not a kernel claim")
and found the lake-toolchain-cwd trap; harvester's round-26 inadmissible-tuple table and
over-long proof owned. MANAGER: the s_min increment form superseded by Delta_3 = O(1) -
my own pre-registered shape was the weaker one.

THE DERIVATION STATE (manager's item, end of round 27): the literal-step induction now has
(i) Q*_2 <= F_2 immediate; (ii) the triple inequality - 8 censused steps + 1 certified, and
BY CERTIFICATE at 6 steps; (iii) deeper-J range capped by the uniform-order theorem (<= 5
machine-free) so only J <= 6 matters; (iv) Delta_3 = O(1) as the sharp target; (v) the
padded case isolated to the (q',q') word with 53->59 evidence that padding per se is fine.
WHAT WOULD CLOSE IT: a derivation of Delta_3 = O(1) (or even Delta_3 <= s_min) from T1-T5 +
phase saturation, plus the per-J analogues for J = 4, 5, 6 - A FINITE LIST OF LEMMAS.

ROUND-28 (briefed, NOT launched; spine = THE FINITE LEMMA LIST):
MANAGER (Fable) -> Delta_3 = O(1): attempt the derivation with the full toolkit (T1-T5,
phase saturation, mirror parity, the free reduction); U13 (the counterfactual null model for
(D) - is the twin's budget slack also favourably placed?).
CONSTRUCTOR (Opus) -> the per-J triple analogues for J = 4, 5, 6 (the uniform-order theorem
makes this a FINITE program); the cover-half order question N(M) via the pruned-IE counter;
rung nine with the screened+exact-shard oracle (1.75M tuples + 338K exact).
MECHANIC (Opus) -> F_2(59) and the F(59) pin (the bracket is [161,178]); extend the exact
m41 shard span-upward as capacity allows (the checkpoint resumes); the span-68 inflation
onset - is it arithmetic (predictable per machine)?
LATERAL (Opus) -> own mandate: U7, U10-U12, U14; the counterfactual family's OTHER
statistics (percentile of F_2? of the increment? - each favourable placement is evidence the
transfer has room); the "at most one" instantiation questions the mirror lever raises.
FORMALIST (Opus) -> item 0: the mirror lever instantiated at a machine (even_card_involution
composed with the opening enumeration); the increment-law certificates (LP's six steps) as
kernel statements; rung eight if the m37 emission lands; case-count economics for k=3.
HARVESTER (Opus) -> the human's memo decision executed (whatever it is); h_2 at p_n=151-251
(the lane's top research item - the falsification target both models diverge on); j_3 beyond
z=7 (needs the real algorithm, named and priced).
LP THREAD (Opus) -> the cut-loop frontier (the second frontier, no closed form - map it);
increment certificates at 31->37 (the padded step - does the certificate see the padding
excess?); emission for the mirror-lever and increment kernel work as Formalist specifies.

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

## Mechanic round 25

GATES (all three re-run from clean processes at round close, all GREEN):
  .venv-sat/Scripts/python.exe research/akill_verify_r25.py       -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/cyclic_close_r25.py check     -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/dict_containment_r25.py \
      research/data/gap_tuples_37_4.csv research/data/gap_tuples_37_4_transfer.csv
                                                                  -> CONTAINMENT VERIFIED
Every job this round launched has finished; nothing left running.


HEADLINE, AND IT IS A SELF-CORRECTION OF MY OWN ROUND-24 REPORT: round 24
closed with "A_kill(47->53) >= 3 ... NOTHING seen at either step contradicts
k_max = 3". The orchestrator I left running finished the job after the round
closed, and its log says otherwise.

A_kill(47->53) = 5 EXACT - THE FIRST 5-CHAIN ANYWHERE IN THE PROJECT.
Complete levels, no word left pending: N_3 = 11 realised of 19, N_4 = 8 of 27,
N_5 = 2 of 12, N_6 = 0 of 2. GATE research/akill_verify_r25.py (log
research/data/r25/gate_akill_r25.log) re-derives ALL FIVE PARTS from the
definition in plain integer arithmetic, importing nothing from a_kill.py /
cov_count.py / j5_multi.py except the two k=6 refutations, which it re-runs
rather than re-reads: (A) all ten realised k=4/k=5 witnesses - occurrence
(chain members are m47 openings, every other slot of the span blocked, gear by
gear), killability (a residue mod 53 putting every member on a tooth), and
joint realisability at k* = CRT(k0, r), 10/10; (B) the k=6 candidate list
re-enumerated independently and asserted equal to the decided list (exactly 2
words); (C) both k=6 words zero - and one needs NO SAT CALL: with exposed set
{0,18,53,71,106,124} every residue mod 5 is forbidden for gear 5, so gear 5
must block an exposed slot; (D) the Q_6 witness; (E) the consequence.

THE SHAPE - THE ALTERNATING CHAIN. Every realised 5-word is the pure
alternation (18,35,18,35) = (s, q'-s, s, q'-s) or its reverse, span 106 = 2q'
exactly. That is why 43->47 has k_max = 3 and 47->53 has 5: at q'=47 the
alternating pair (16,31) is not realised, at q'=53 the pair (18,35) is. Same
arithmetic-selection law as ever, now legible in a single word.

CONSEQUENCE - A GATED NEGATIVE (not a judgment): THE CRITERION REPAIR FAILS AT
47->53. The merge law consumes depths j <= k_max + 1 = 6; round 24's
restoration threshold there was k_max <= 4. And Q_6(47; 18) = 174 > 171 =
F(47) + 53, with that witness MACHINE-VERIFIED FOR THE FIRST TIME this round
(only the J=7 one had ever been checked): m47 address 92,241,409,917,573,978,
gaps [20,22,28,30,67,7], middles [22,28,30,67] >= 18, all 168 other interior
slots blocked. The verdict is robust to k_max's exact value - k_max >= 5
forces depth >= 6 and Q_6 already fails. LADDER STATE: (D) certified by the
criterion at every step through 41->43; 43->47 RESTORED (A_kill = 3 <= 5);
47->53 NOT restorable by this criterion. (D) AT 47->53 IS UNTOUCHED AND TRUE
by arithmetic (F(53) = 145 <= 171).

THE NAMED CONSTRUCT, BUILT IN-ROUND: THE WORD-LEGAL CRITERION Q*_J. The
failing window's middle gaps are [22,28,30,67] and NOT ONE is congruent mod 53
to a legal kill letter (V = {0,18,35}) - the criterion is failing on a
relaxation the merge law never needed. Q_J asks only that each middle gap
clear a = 2u'; the merge law needs the J-1 interiors deleted by ONE phase of
q', i.e. exactly a_kill.py's WORD LEGALITY (each middle gap in V mod q', and
the induced letter word's prefix sums of range <= 1). ">= a" is its shadow -
the smallest positive legal value IS a. Sound as a (D) criterion, pointwise
<= Q_J, same lap-phase transfer and same cost (one predicate in the
mark-acceptance test; research/j5_multi.py optional argument 'legal').
TWO ANCHORS, BOTH TIGHT AND BOTH TWO-SIDED: max_J Q*_J(31; legal for 37) =
88 = F(37) EXACTLY at J = 4 (193 s), and max_J Q*_J(29; legal for 31) = 58 =
F(31) EXACTLY at J = 3 (346 s), against the plain criterion's 91 and 71. Each
value MUST be >= the F it hits (the true maximal gap is such a window) and the
scans find nothing above, so both anchors test both directions - and the
attaining depths reproduce C13's independently measured k_win (3 and 2, since
a chain of k kills merges k+1 gaps). CONJECTURE on 2 exact points: Q*_max IS
the merge-law value F(M+q'), not merely an upper bound; the only relaxation
left is that Q* does not require the phase to also spare the two endpoints.

AND IT CERTIFIES BOTH FAILING STEPS, HYPOTHESIS-FREE - THE ROUND'S BIGGEST
RESULT:

    step      plain max_J Q_J   budget   word-legal max_J Q*_J   verdict
    29->31          71            74            58 (= F(31))     CERTIFIES
    31->37          91            95            88 (= F(37))     CERTIFIES
    43->47         152           150          <= 149             CERTIFIES
    47->53         177           171          <= 170             CERTIFIES

Both certifications are over ALL depths J = 2..7, so NEITHER consumes a fuel
cap: 47->53 no longer needs the (false) k_max <= 4, and 43->47 no longer needs
A_kill = 3. The word-free criterion ladder is complete through 47->53 with no
arity hypothesis anywhere. Costs 964 s and 2,213 s on machine 23's period.
Honest: those two runs are seeded at budget-1, so their values are
max(true, seed) - margins >= +1, true maxima unresolved; the certification is
what is established, and (as always for this construct) a certification, never
a failure, is conditional on the span cap 200 against budgets 150 and 171.
Doc: docs/novel/old-machine-spectrum.md section 8.

F_2(47) = 134 EXACT - a first computation, superseding the standing range
[119,141]. Floor-1 lap-phase transfer r=6 from machine 23, 529 s, COMPLETE
(span cap 150 above the deletion-ladder cap F_2(47) <= F(53) = 145), seeded at
118 and the answer sits above the seed. Witness re-verified at machine 47:
k = 97,575,004,641,096,768, gaps [54,80], all 132 interior slots blocked.
Note the maximiser contains NEITHER a maximal gap, so both neighbours of every
maximal gap of m47 are <= 16. AND IT PAYS IMMEDIATELY: 134 < 141 zeroes BY
THEOREM every 47->53 kill word with a 2-block of span 141 - exactly the four
k=3 words (53,88),(88,53),(106,35),(35,106) that cost 20,005 s of UNSAT at
round-24 close, plus six k=4 and four k=5 span-141 words. Fourteen SAT
refutations (22,608 s) replaced by one 529 s scan, and the two methods agree
on every one. (Wall times secondary and contended; the structural point is the
agreement.)

THE m37 EXACT 4-TUPLE DICTIONARY DELIVERED (brief item b): 291,675 realised
4-tuples, research/data/gap_tuples_37_4.csv. Six range workers over the full
period P(37) = 1,236,789,689,135 (the three round-24 survivors plus the three
ranges rerun this round), merged with both assertions passing - openings
217,929,355,875 = prod(q-2) EXACTLY and max gap 88 = F(37). Induced levels
75 / 2,053 / 30,325. TWO FREE CHECKS THE TOOL DID NOT KNOW ABOUT: the 75
distinct gap values are exactly 88 minus C14's 13-value m37 hole list; and the
six workers' range statistics come out MIRROR-PAIRED to the unit (w0=w5,
w1=w4, w2=w3 in openings, max gap and distinct-tuple count) - six independent
processes reproducing the k -> -k symmetry of the period.

THE 31->37 TRANSFER SUPERSET IS NOW VERIFIED, and round 24's prediction about
it was exact: 0 of 291,675 realised tuples missing, inflation 8.35x, and the
two "extra" depth-1 values it could not exclude are 73 and 75 - both m37
holes, exactly as the round-24 handover said they would be, by value.
Inflation ladder: 4.15x (23->29), 6.21x (29->31), 8.35x (31->37).

THE m41 DICTIONARY SUPERSET DELIVERED (brief item c): 4,239,676 4-tuples in
77 s from the exact m37 dictionary - research/data/gap_tuples_41_4_transfer.csv,
the FIRST dictionary at a machine no scan reaches (P(41) = 5.07e13). AND IT IS
EXACT AT DEPTH 1: its induced 1-tuple dictionary is exactly {1..91} minus
{84,87,89}, i.e. it reproduces F(41) = 91 AND the complete m41 hole list -
both from COV-SAT, a completely different method - with zero inflation and
zero missing values (asserted). Emission profile by deleted interiors
7.6/26.0/34.1/22.9/8.8/0.6/0.01% shows the predicted geometric decay with
depth. FOR CONSTRUCTOR AND FORMALIST: this is the hE-shaped edge set for the
chain's deeper steps.

DEFECT ROUTED IN BY LATERAL (their parity law, on first use) - CONFIRMED,
FIXED AT SOURCE, AND THE DATA CORRECTED. gap_pair_census.py closed the period
LINEARLY: np.diff over a linear pass gives N-1 gaps where a CIRCLE of N
openings carries N. Measured shortfall was exactly 1 in every full-period
ghist row at all seven machines - and the lag and run tables were short by
more (j+1 at lag j, m at run length m), because they lose every structure
straddling the seam.
THE MISSING GAP HAS A CLOSED FORM, so nothing needed a rescan: slot 0 is an
opening at every machine (gear q blocks +-6^{-1} mod q, never 0) and the
opening set is mirror-symmetric (C18), so the largest opening is P - x_1 and
    wrap gap = P - x_{N-1} = x_1 = d_0, THE FIRST GAP
= 3, 3, 5, 5, 5, 7, 7, 7, 10 at m11, 13, 17, 19, 23, 29, 31, 37, 41 (asserted).
DONE: gap_pair_hist.csv and gap_pair_joint.csv corrected exactly (60 + 124
cells; pre-fix kept as *.linear.bak), and all eleven tables per machine now
total N at all seven machines; gap_pair_census.py AND hist_probe.py fixed at
source; gate research/cyclic_close_r25.py. DOUBLE-SOURCED - the source fix
re-run from scratch at m11/13/17/19 reproduces the corrected CSVs CELL FOR
CELL across ghist, all five lag tables and all five run tables.
SCOPE OF DAMAGE, bounded by argument not by luck: because the missing gap is
always the FIRST gap (3-10), the padding-supply numbers (every probe q' >= 29),
the hole lists, and F/F_j are all untouched; the p_j ratio table moves below
its last printed digit. The one row that WAS the defect is C12's "total gaps
per period", now corrected to prod(q-2) exactly. AUDIT: the gap-tuple
dictionaries are CLEAN - gap_tuples_lean.py and _par.py both wrap explicitly.
New standing rules 25 (assert the parity identity before writing any
full-period CSV) and 26 (a long scan must save its array, not its summary).
FOR LATERAL, the U6/U9 ask, honestly: the wrap VALUES at m37 (7) and m41 (10)
are above and are everything the cyclic close adds - but NO full-period m37
histogram ARRAY exists on disk. The 11,829 s round-20 scan logged only F, four
probe counts and the hole list and discarded the array (hence rule 26).
Recomputing is ~3.3 h, which would not finish inside this round, so I did NOT
start it (job-completion rule); it is a scoped next-round item. Price the
paired-Holt recursion first - it gives n_g(37) from machine 31's word-level
census with no m37 scan at all.

HONEST NEGATIVES AND COSTS OF THE ROUND:
- My own round-24 closing sentence "nothing seen contradicts k_max = 3" was
  FALSE, and the round-25 brief inherited it. It was written about an
  orchestrator still running. New standing rules 22 (an unfinished level bounds
  arity from BELOW only) and 23 (first action of a round: re-read every log a
  previous round left running, before quoting any partial verdict).
- The fuel-cap route to repairing the criterion at 47->53 is DEAD, not
  deferred. That was the brief's named dependency and the answer is negative.
  The round only recovered because a different construct was built.
- The first word-legal anchor was run UNSEEDED and was >= 10x slower than
  necessary; abandoned and rerun seeded. New standing rule 24.
- My gap-pair census had been closing the period LINEARLY since round 20 and
  I did not catch it - another lane did, on first use of an exact identity.
  New standing rule 25: assert the parity identity before writing any
  full-period CSV.
- A full-period m37 gap histogram is NOT delivered (Lateral's U6/U9): the
  11,829 s round-20 scan discarded its array and only logged summaries, so it
  needs ~3.3 h again. Not started - it would not have finished in-round.
  New standing rule 26.
- The word-legal certifications at 43->47 and 47->53 are SEEDED at budget-1
  and SPAN-CAPPED at 200: they establish the certification, not the exact
  maxima, and (as for every scan of this construct) a certification, never a
  failure, is conditional on the cap.

FOR OTHER LANES:
- FORMALIST: the hypothesis "k_max(M->q') <= 3" is FALSE at 47->53 - do not
  assume a universal fuel cap in a kernel statement. Depth 6 is needed there.
  The word-legal predicate is a clean finite side condition if you want the
  criterion in Lean: it is two modular tests per middle gap plus a prefix-sum
  range check.
- CONSTRUCTOR: A_kill(47->53) = 5 raises the depth your chain must reach at
  that step from 5 to 6. The alternating chain (s, q'-s, ...) is the object
  that decides fuel arity, and whether it extends is one histogram lookup per
  step (are both s and q'-s realised gap values?) - cheaper than any scan.
- BOTH: F_2(47) = 134 exact is a new input.

## Lateral round 25

MANDATE RESTORED, AND THE BACKLOG WAS ONE OBJECT. Picked U1 + U2 + U3 plus the permitted
mirror-law item; they collapsed into a single frame - the involution k -> -k - which
turned out to pin the PARITY of every count in the machine and to supply the coordinates
in which round 21's phase and amplitude laws become exact integer statements. U4 (Farey)
and U5 (613 cosine near-collisions) were NOT worked: depth on a connected cluster beat
breadth. They stay in the backlog, untouched and unclaimed.
Gates: research/mirror_cells.py --parts ABCDEF --maxy 19 -> 9 ASSERT ok, exit 0
(log data/mirror_cells.log); research/spiral29.py at m11..m29 -> all green.

THE MIRROR PARITY LAWS (proved, elementary; script-verified exact, m11..m29).
Slot k is blocked iff a gear divides 6k-+1 - invariant under k -> -k - so the opening set
is exactly closed under negation, 0 is its ONLY fixed slot (P odd), and on indices the map
is o_t -> o_{N-t}.
- WINDOW PARITY: mirror sends the depth-j window at index t to the one at N-t-j, and
  N = prod(q-2) is ODD, so 2t = -j (mod N) has exactly one solution. Hence for EVERY
  depth j, W_j(g) is EVEN for every g except the single length of the window at
  t_j = -j/2 (mod N), where it is ODD - the exceptional class located in advance and
  matched at every (machine, depth) tested. Corollary: THE MAXIMAL GAP OCCURS AN EVEN
  NUMBER OF TIMES unless F is the antipodal gap (it never is).
- WORD REVERSAL: the depth-j gap-word census is EXACTLY reverse-symmetric, with exactly
  one odd palindrome per depth; at j = 2 it is FORCED to be (k_1,k_1). This upgrades
  round 7's "closest to revcomp-symmetric" from approximate to exact for gap words.
- THE ADJACENT-GAP COROLLARY - THE ITEM FOR THE MANAGER. Since k_1 < F, the unique
  self-mirror adjacent pair is never (F,F), so ANY ADJACENT CONFIGURATION WITH g_1 = g_2 -
  in particular an (F,F) pair realising F_2 = 2F - OCCURS AN EVEN NUMBER OF TIMES. "Big
  next to big" of equal size can never happen exactly once. CONSEQUENCE FOR THE LIVE
  ROUTE: a counting/covering argument that caps such configurations at ONE thereby proves
  there are NONE - a strictly cheaper obligation than proving the cap is zero. Reported,
  not developed (mandate).

A DEFECT IN THE SHARED CENSUS FILE, caught by the parity law on its first use: EVERY
full-period ghist row in data/gap_pair_hist.csv carries N-1 gaps, not N = prod(q-2) - the
census closed the period LINEARLY and dropped the WRAP-AROUND gap (size k_1 = 3,3,5,5,5,
7,7 at m11..31; the missing gap is forced as P - sum g*count). Relative error 1e-9,
harmless for densities, FATAL to every exact integer identity downstream.
mirror_cells.load_ghist repairs it and asserts the repair; any other consumer of that file
should do the same. Mechanic may want to fix it at source.

THE GEAR-p CELL DECOMPOSITION (proved + exact). Openings live on the exposed set A_p
(|A_p| = p-2), and p-2 consecutive exposed phases span exactly p slots, so zeta_p^{gap}
depends ONLY on (start phase, exposed-step count mod p-2). The (p-2)x(p-2) integer CELL
MATRIX therefore carries the whole frequency-1/p transform of the gap histogram; CRT fixes
its row sums and mirror pairs its cells, leaving (p-2)(p-3)/2 FREE INTEGERS - THREE at
p = 5, for every machine, for ever. Consequences at p = 5, all asserted:
  N_2 and N_3 are always EVEN;   2 (N_1 - N_4) = N_2 - N_3  exactly;
  Re H_5(1) = phi N/3 + (3-phi) e - ((3phi+1)/2)(b+c),  Im H_5(1) = (2 sin36+sin72)(b-c),
so the imaginary part of the entire transform is ONE integer. (The partial-coverage m37
census row is carried as a control and fails, as a period-wide law must.)

THE POLE PHASE 126 DEG IS UNATTAINABLE - A THEOREM, AND THE SECOND INDEPENDENT KILL OF
THE ROUND-21 PIN. arg H_5(1) = 126 requires N_0+N_1 = N_2+N_4, which in cell variables is
2(b+c-e) = N/3 with N/3 ODD. Therefore D := (N_0+N_1)-(N_2+N_4) is ODD at every machine,
equivalently (N_2+N_3) - 2N_0 = 2 (mod 4) - asserted at m11..m31 (defects 38, 282, 2998,
37306, 634182, 13462586, and m31, all = 2 mod 4). Round 21's pin was already refuted by
its own drift model; it is now refuted a second time by arithmetic. What the machine does
instead: the integer ratio alpha_1/alpha_2 of the two bracket asymmetries converges on the
GOLDEN DIRECTION -1/phi (-0.864, -0.839, -0.731, -0.640, -0.645, -0.623, -0.594 at
m11..31), crossing it between m29 and m31 exactly where arg B(5,1) crosses 0. Honest
scope: the parity floor forces |dev| > 0 by only ~1e-6 deg - it kills the pin as an EXACT
statement, not the empirical closeness.

BACKLOG U3 ANSWERED (why gear 5's bracket is real and gear 7's drifts), two ways.
STRUCTURAL: a GF(2) test over mirror orbits shows GEAR 5 IS THE ONLY PARITY-OBSTRUCTED
GEAR for p <= 37, and pole-realness costs (p-1)/2 integer equations - at p = 5 it is ONE
ratio chasing one irrational, at p = 7 THREE independent asymmetries must vanish at once.
MEASURED: gear 7's asymmetries are an order of magnitude larger and decay far more slowly
(max|alpha|/N 0.259 -> 0.164 over m11..37, vs gear 5's 0.141 -> 0.019).

BACKLOG U2 CLOSED - THE 1.015 AMPLITUDE IS A CROSSING SCALE, NOT AN INVARIANT.
The round-21 closure, verified here over ALL N-1 depths exactly and real at m11/m13, makes
the MEAN ARM of the depth family exactly ((2-phi)n_side^2 - N)/(N-1) -> (2-phi)N/9 =
0.042440 N - precisely the value depth 1 would take if consecutive openings decorrelated.
So the near-law is |What_1|/mean arm = 23.92/lam, and lam = 23.92 is the machine size at
which DEPTH 1 BECOMES A TYPICAL ARM. Two further reductions: the amplitude is ~95% a
statement about the exposed-step count mod 3 (item 38's object) and ~5% phase-graded, with
the graded part SHRINKING; and a new CORRIDOR-RENEWAL LADDER (exact first-passage on the
m-cycle, gated by reproducing the machine at m = P to 1e-9) shows that NO FIXED corridor
depth reproduces the flatness - m=5 decays 1.09 -> 0.87 while every deeper column rises,
and the measured constancy is the cancellation of those two drifts as the machine's own
corridor depth grows. HONEST OPEN PART: the break DIRECTION is unsettled - the corridor
model turns up past lam ~ 24, the round-21 closed-form M1 predictor declines
(1.060 -> 0.906 over y = 41..449). One full-period m37 or m41 gap histogram decides it.

BACKLOG U1 CLOSED - THE m29 DEPTH SPIRAL, the rung nobody had run. Streaming rewrite
(residues mod 5 only, rolling buffer, ~200 MB instead of ~1.7 GB) over the full m29 period
(P = 1,078,282,205; N = 214,708,725). W_1 arm 0.2023 N at 126.06 deg. W_2 arm ladder
reproduces round 21 and extends it: -9.20, +33.90, +66.47, +87.71, +113.15, +118.78 deg at
m11..29 - STILL CLIMBING toward the pole phase but the increment collapsed from +25.4 to
+5.6. The large-j arms sit at the mean-arm value with small argument, so the U2 floor is
visible directly in the spiral.

PREDICTIONS: 6 pre-registered, 3 CONFIRMED (unique odd palindrome = (k_1,k_1) with
#(F,F) even; gear 7 not parity-obstructed; gear 5 the ONLY obstructed gear p <= 37 - the
round's most useful prediction, since it is the structural half of U3), 3 REFUTED, all
three by this lane's own scripts (the corridor ladder converges from ABOVE not upward and
m=385 is 9.1% high at m19; the phase-blind departure is 4.3-4.8% not >= 5% and shrinking;
the closed-form model DECLINES rather than exceeding 1.10 by y <= 200).

NEEDS / HANDOFFS: (1) one full-period m37 (or m41) gap histogram would settle both U6 (does
alpha_1/alpha_2 overshoot -1/phi permanently) and U9 (the direction of the amplitude
plateau's break) - Mechanic's existing tiling runs already cover the period, only the
histogram needs closing cyclically. (2) FORMALIST: the parity theorem
(N_2+N_3) - 2N_0 = 2 (mod 4) and the mirror relation 2(N_1-N_4) = N_2-N_3 are finite,
integer, per-machine statements and are the cheapest Lean targets this lane has produced.
(3) MANAGER: the adjacent-gap parity corollary above, for the two-gap route.
New novel-register docs: docs/novel/mirror-parity-laws.md,
docs/novel/gear-cell-decomposition.md (both "not yet checked" - prior-art terms listed).

## Harvester round 25

STANDING GATE FIRST: `research/j2_referee.py` re-run from a clean process before
anything else - **"ALL ASSERTIONS GREEN"**. New gates this round, both green:
`research/j2_odc6.py`, `research/j2_rankin_layer.py`.

**BOTH NAMED UNIT-1 OPENINGS CLOSED, AND THE EXPLICIT EXPONENT FALLS 15 -> 8.042.**

1. **BLIGHT'S THESIS OBTAINED - CLOSED NEGATIVELY.** Sara E. Blight (not "Sean"),
   *Refinements of Selberg's Sieve*, Rutgers 2010, DOI 10.7282/T35T3KJ8, free,
   downloaded (research/data/blight_thesis.pdf, 75 pp) and read first-hand. Two
   independent reasons it does not help: (i) her kappa = 2 value is beta_2 < 4.45,
   WORSE than the DHR 4.266450 we already cite, and her own sec. 2.7 says so;
   (ii) her Proposition 2.4.2 concludes only "**there is some z_0** such that if
   z > z_0, then V(D,z) is also positive", with the K-dependence hidden in a `<<`
   - so it is not explicit. This extends our explicitness boundary from the DHR
   system to the Lambda^2 Lambda^- family, from the primary document.

2. **ODC CHAPTER 6 IS EXPLICIT. THE ANSWER IS YES.** Page scans of pp. 65, 68-73,
   112 read first-hand 2026-08-29. **Proposition 6.7, (6.75), (6.85), (6.86) and
   Corollary 6.13 carry no O(.), no `<<`, no implied constant and no "z large".**
   The one asymptotic sentence is Corollary 6.14's "provided K is sufficiently
   close to one" - and PRE-SIEVING replaces exactly that device at explicit finite
   cost. Location correction to our own note: beta_1 = 3.8629, beta_2 = 7.5941 are
   on **p. 73**, not in Cor 6.14. WHY WE MISSED IT IN ROUND 24: we priced Thm 6.9
   and Cor 6.10 from the chapter and never priced its PROPOSITIONS.

3. **THEOREM 2G.** With p_0 = 151 (so K = 1.0260 < 1.0297 and ODC Cor 6.13 applies
   verbatim at alpha = 1/4):
   `j_2(p_n#) <= C p_n^8.04162 (8.04162 log p_n + 1)(log p_n)^2 + 1`, p_n >= 285,
   log10 C = 57.5. Constant-free form: `j_2 <<_eps p_n^(s+eps)` for every
   **s > 7.93727**, every implied constant computable. The log power falls 10 -> 3
   because the beta sieve's weights are bounded by 1, so the remainder carries
   tau (2^nu) not tau_4 (4^nu). Round 24's 2E'' (exponent 15) STAYS in the paper -
   its constant is tiny and it is the better bound below p_n ~ 380,000; the
   crossover is computed and tabulated.

4. **THE TWO ROUND-24 LEADS ARE ONE EQUATION - PROVED.** Halberstam-Richert's
   Memoire positivity condition `lambda^2 e^(2lambda)(2+e^2) < 1` IS ODC (6.86)'s
   `2e^-2 a^2/(1-a^2) < 1` under `a = lambda e^(1+lambda)` (identical to 1e-12 at
   six values), and HR's `lambda_* = 0.2533219` equals ODC's K -> 1 root
   `0.253321897` to 5e-7. **ODC Chapter 6 is the explicit form of the 1971
   Memoire's theorem, and slightly sharper (7.9373 vs 7.9720).**

5. **A DISCREPANCY IN THE BOOK, FOUND AND RECORDED.** ODC's printed root
   alpha* = 0.264904 does NOT solve ODC's own printed equation (residual -0.0017;
   true root 0.2652637). It is internally consistent with the printed beta_1 and
   beta_2, so it is not an OCR error - the book says it used "the Taylor expansion
   at 1/4". In our favour: the exact root gives beta_2 = 7.5838, **0.0103 better
   than the printed 7.5941**.

**THE RANKIN-LAYERING PROBLEM: (P2) SUPERSEDED BY ONE LOG.** New doc
`docs/novel/layered-erdos-rankin.md`. Running the Erdos-Rankin construction ONCE
PER AVAILABLE CLASS gives the k-class Jacobsthal family
`j_k(P(x)) >> x A^(2k-1) C^k/((5k)^k B^(2k))` (A = log x, B = log A, C = log B),
whose **k = 1 case IS the published FGKT length** and whose k = 2 case is
`h_2(P(z)) >> z (log z)^3 (lll z)^2/(100 (ll z)^4)` - two logs above round 24's
(P1) and one log above what (P2) asked for. Mechanism: class 0 on a SPLIT range
buys a full log where its Mertens entitlement is O(1), and the paired problem's
second class buys it AGAIN on n+2, so the joint survivor set is the TWIN primes;
only an UPPER bound on twins is needed, so the construction is parity-free.
**STATUS: ASYMPTOTIC BOOKKEEPING, NOT A WRITTEN-OUT PROOF** - but calibrated
against someone else's theorem (the same optimiser at k = 1 tracks the FGKT closed
form with residual spread 0.072 over eight decades of log x), and its finite
ingredients are exact (restatement brute-forced at z = 3,5,7; c = 2 collides with
no odd class; twins-or-smooth survivor structure verified by direct sieving).

**SELF-CORRECTION TO OUR OWN ROUND-24 RECORD.** The "best model ~2.56 z (log z)^2"
is **DEMOTED from "truth" to random-choice heuristic**: it is the largest gap in a
random set of density prod(1-k/p), while j_k is a MAXIMUM over choices. Right at
k = 1 (Rankin attains it), exceeded by a log at k = 2. Round 24's own corollary
"MODEL CLAIMS EXPIRE LIKE CITATIONS" firing on round 24's model.

PRE-REGISTRATION SCORED (written before computing): PR1 (two extra logs)
CONFIRMED; PR2 (model not a ceiling) CONFIRMED; PR3 (power 2k-1, general k)
CONFIRMED at k = 1..5; **PR4 CONFIRMED IN CONCLUSION BUT WRONG IN MECHANISM** - I
predicted the layering would be a LOSS at reachable z; it is not a loss, it does
not EXIST below log z ~ 300 ([P,z1] is empty). Scored wrong-as-worded.

NEGATIVES AND RESIDUAL RISKS: (a) the ODC pages were read through a browser
preview, not held - mitigated by reproducing eight printed numbers from the
printed formulas; (b) **(5.38) (the definition of K) and (6.69) (a kappa condition
quoted inside Prop 6.7) were NOT re-fetched** - flagged, not assumed; our operative
alpha = 1/4 is the book's own choice in Cor 6.13 "for kappa > 0"; (c) p. 74 (rest
of Prop 6.16) not obtained, so the pre-sieving accounting is still round 24's own;
(d) no journal paper by S. Blight exists; (e) "the 7.937 -> 4.266 gap is not
reachable by more pre-sieving" is a RESULT for this sieve (the K -> 1 limit IS
7.93727) and a **JUDGMENT, NOT RESULT** for the problem; (f) FKMPT "Long gaps in
sieved sets" (arXiv:1802.07604) - round 24's RELAY-SOURCED flag **discharged**,
read first-hand: it is the ADVERSARIAL problem (classes GIVEN, bound
x(log x)^(1/exp(C C_0))), neither result contains the other, and it must be cited.

RANKING CHANGES: N4 (upper ladder) stays top and is materially stronger; P1-P3
(lower ladder) rises toward parity, blocked by writing not research; NEW item
just below N4 - **(P2') write the k = 2 layering out with constants** (ordinary
work); opening 7c#1 (Blight) DEMOTED, closed; 7c#4 (**h_2 at p_n = 151..251**)
RISES, because the competing readings are now z(log z)^2 vs z(log z)^3, a further
log apart, so the computation discriminates more than it did. **THE UPPER
LADDER'S NAMED OPENING CHANGES**: no longer "find an explicit dimension-2
lower-bound sieve" - we have one - but "close 7.937 (explicit) to 4.266 (DHR, not
explicit)". ODC sec. 6.6 says it is superseded by Chapter 11, but Chapter 11's
lower-bound constant B is identically zero at kappa >= 1/2, so **Chapter 11 is NOT
the route**; an explicit form of the DHR differential-delay system is.

TWO CLAUSES ADDED TO THE STANDING CITATION-HYGIENE LESSON: (4) **price
PROPOSITIONS, not only theorems** - the usable ODC result sat on the facing page
of two we had already priced and was worth seven units of exponent; (5) **a
printed numerical root is a claim like any other** - re-solve it from the printed
equation, which here cost nothing and improved the constant.

FILES: research/j2_odc6.py, research/j2_rankin_layer.py (+ .out in research/data),
research/data/blight_thesis.pdf, research/data/blight_text.txt, research/data/odc6_scans/PA*.png (ODC
page scans), docs/novel/layered-erdos-rankin.md (new), docs/novel/j2-upper-bound.md
(sec. 10), docs/novel/j2-lower-ladder.md (sec. 7), docs/novel/README.md (index).


## Constructor round 25

THE SCAN-FREE DICTIONARY AND THE CHAIN - the round-24 named next construct, executed.

THE CONSTRUCT (R60/R61). "Is this tuple of consecutive gaps realised by M?" is not a census
question: by CRT a slot IS a phase vector, so the question is the feasibility of a set-cover
CSP with one variable per gear - "the prefix-sum points open, every interior point covered by
some gear". `research/crt_dict.py` decides it exactly (bitmask cover + minimum-remaining-
options branching + a CAPACITY BOUND that is what makes REFUTATIONS affordable). Built on it,
`research/scanfree_dict.py` grows the dictionary level by level with the overlap lemma. THE
PERIOD NEVER APPEARS - only the list of primes up to y.
GATES: decision == R43's independent pruned-IE count on 2,013 tuples at m11/13/17; nine
published anchors; the corpus ladder recovered scan-free (F = 7,11,18,25,34,43,58,88 and
F_2 = 11,16,25,31,39,55,68 at m11..31, F_2(29) at the pair (20,35)); and **the scan-free
D_4(23) is SET-EQUAL to Mechanic's full-period scan census, 15,696 tuples, tuple for tuple.**

THE CHAIN (R62/R63). Putting the CSP oracle where round 24 had the dumped tuple set:
  19->23  181 queries  bound 48 <= 48    23->29  90 queries  bound 63 <= 63
  29->31  955 queries  bound 74 <= 74    31->37  3,399 queries  bound 95 <= 95
  37->41  NOT CERTIFIED - cancelled at bound 235 (budget 129) after 94 queries / 132 s
The first three are R58's counts EXACTLY, with 100% CRT-vs-dump agreement on every one of the
1,226 queries (--shadow mode asserts it, re-run after the bug fix below). **31->37 is a NEW RUNG
certified with no scan and no dump anywhere in the loop** (356 s). The 37->41 rung is a cost
failure, not a soundness one: at m37 an over-budget PAIR refutation costs 5.8 s mean / 23.7 s
worst against 43 ms for a 4-tuple, so the rung is a multi-hour job I mis-sized for the round.
Fed the two-gap output of the rung below (F_2(37) <= 90 - the brief's chain shape exactly) it
shrinks from 1,566,576 states / 2,605,925 edges to 799,184 / 564,661 and starts at 258 not 1,310. And `research/chain_a4.py` closes R49's A_4 over the
scan-free dictionary to return F(M+q') EXACTLY - 34, 43, 58 at 19->23, 23->29, 29->31, with the
m29 system being R49's system exactly (3,513 edges) - so the exact F sizing the next rung is an
OUTPUT of this one. The per-rung scan vehicle (declared dead at ~170 h) is replaced.

THE RESIDUE (R67, the brief's item (c)). The chain never needs an integer it cannot generate:
dictionary, F, F_2, A_4 and the certificate are all computed from the primes up to y. What it
cannot generate are the three quantities that decide whether the finite computation SUCCEEDS -
(i) the ORDER m at which A_m is tight (m > A_relax(M), and A_relax is non-monotone), (ii) the
QUERY COUNT (181, 90, 955, 3,399 - growing, bounded by nothing proven), (iii) the DECISION COST
(a one-gap refutation grows ~x15 per gear: <0.1 s at m29, 1 s at m31, 10-20 s at m37, >250 s
undecided at m41). **Every INSTANCE of the two-gap statement is a covering refutation the chain
generates itself; the STATEMENT - one bound at every y - is not.**

WHERE THE MISSING ARGUMENT LIVES (R64, new). In covering form the two-gap law is: every pair
with g1+g2 > F+q' has an infeasible cover CSP. Three machine-free instruments, all measured
exactly (`research/twogap_threshold.py`): CAPACITY kills 5 of 15 over-budget pairs at m23 and 0
of 3/78/231/1128/1176 at m19/29/31/37/41 (only both-gaps-near-F pairs; not a supplier, but not
vacuous - which X12's local form did not distinguish). THE FIRST MOMENT gets the law RIGHT at
every machine (model increment 5,6,9,11,12,14,16,18,19 vs q' = 13..43) - unlike the histogram
(X35) and the corridor (X34), which saturate at 2F, INDEPENDENCE DOES NOT SATURATE. And in
closed form the model increment is O(log^3 y) against a budget q' ~ y: incr/q' measured
0.385 -> 0.103 -> 0.047 -> 0.019 -> 0.0145 at y = 11, 1487, 5261, 20509, 29917. So the missing
step is not a sharper inequality but an unconditional TRANSFER of a first moment that already
has a polylog-versus-linear margin.

CROSS-LANE (R65): LATERAL'S MIRROR PARITY LAW VERIFIED SCAN-FREE, AND THE 2F ENDPOINT KILLED.
`research/mirror_parity_lever.py` counts every adjacent EQUAL pair exactly by CRT enumeration -
no period - and confirms Lateral's law at m11..29: the odd-count value is unique and equals
g = 3, 3, 5, 5, 5, 7 (counts 1, 13, 649, 10,965, 219,553, 1,739,485), always < F. Further:
(F,F) has count ZERO at all six machines, and the LARGEST equal adjacent pair is far under
budget - 2g = 10, 14, 16, 16, 30, 40 against budgets 20, 28, 37, 48, 63, 74, slack GROWING. So
the F_2 = 2F endpoint that both machine-free instruments point at is not attainable, and F_2's
maximisers are asymmetric ((20,35) at m29, (5,34) at m23, (2,88) at m37). Lateral's lever
("cap the count at one, parity gives zero") is live; `crt_dict.count_solutions` supplies the
capped count at any machine.
Also: Lateral's ghist defect does not touch this round's inputs - the CRT oracle reads no
census file, and the round-24 pair census is used only as the shadow comparand, where it agreed
with the CRT decision on all 331 pair queries.  AND THE REPAIR IS CONFIRMED FROM THE OTHER SIDE:
after Mechanic fixed gap_pair_joint/gap_pair_hist at source mid-round, the scan-free level-2
dictionary is SET-EQUAL BOTH WAYS to the repaired lag-1 census - 221 pairs at m19, 429 at m23,
nothing in either that is not in the other.

HONEST NEGATIVES. (1) A REAL BUG, caught by Mechanic's census and not by my own gates:
`decide_cover` returned True on an empty interior before checking the open-point domains, so
(1,1), (1,1,1), (1,1,1,1) were wrongly realised. Nothing reported depended on it (no T3-legal
edge, never a maximum; the only trace was A_4(29) showing 14,370 states where R49 has 14,368);
fixed, and the dictionaries are now set-equal. (2) A closure written with
`np.maximum(new[usrc], seg, out=new[usrc])` writes into a fancy-index COPY - the relaxation
silently did nothing; caught in one run by the shadow gate. (3) R59's "every query has span
<= F_2" was never measured and is WRONG: max query span 90/105/127 against F_2 = 31/39/55. The
queries are cheap because they have many OPEN points, not because they are short. (4) The
brief's descent chain (A_5(23) -> F_2(29) -> ...) is superseded by its own success: with a
scan-free dictionary the two-gap integer is computed at the machine itself. The descent remains
the right structure for a UNIFORM argument, not for a computed rung.

PREDICTIONS (pre-registered, research/data/r25_prediction.txt): P1 CONFIRMED (100% agreement),
P2 REFUTED (spans, both halves), P3 CONFIRMED (counts identical, not merely close), P4 HALF
(true for arity 2/4, refuted for arity 1), P5 CONFIRMED for 31->37 and UNRESOLVED at 37->41 (cancelled on cost), P6 PARTLY CONFIRMED (right
that the residue is not an integer; incomplete - three quantities, not one), P7 REFUTED (the
m41 wall is the oracle, not the state space: MF_4 at m41 builds in 15 s).

NEEDS: (a) a y-uniform bound on the refinement order - equivalently, prove A_relax(M) <= 4 or
find where A_4 first fails; (b) a bound on the query count; (c) THE FIRST-MOMENT TRANSFER -
derivation-grade, escalation valve applies, NOT attempted in lane.
New doc: docs/novel/scanfree-certificate.md (indexed).

## Formalist round 25

THE SIXTH RUNG, AND THE WALL AT 170 HOURS IS GONE. (D) at alpha = 3 now holds at
29->31 as a kernel theorem with exactly ONE named hypothesis, and it builds in FIVE
MINUTES where round-24 verdict 17 priced the period-scan vehicle at ~170 h.

    theorem D_29_31 (h : Machine29.Census29) (n : N) : Machine31.g31 n <= 43 + 31
    theorem g31_le_of_census (h : Census29) (n : N) : Machine31.g31 n <= 71
                                                      -- proofs/Machine31.lean

BUILD GREEN AT 1410 JOBS (1372 -> 1410), 74 targets, 127 files, zero sorries, zero
`axiom` declarations, no native_decide, no ofReduceBool. NINETEEN new roots. Round-24's
outstanding scaffolding (`DryScan2.lean`, verdict 18) deleted at close.

THE VEHICLE IS NOT THE ONE THE BRIEF POINTED AT, AND ONE SCRIPT SAID SO IN 30 SECONDS.
Over the exact realised 4-tuple dictionary the qualifying-tail potential is INFINITE:
the qualifying sub-digraph of the A_4 state graph has CYCLES at m23, m29 AND m31,
because a 4-tuple cannot express "no six consecutive gaps reach the floor" (that is a
7-tuple fact). A_5 would not fix m29 either. New verdict 19, measured
(`research/a4_potential.py`), exact, and reusable by any lane building A_m certificates.

WHAT DOES WORK - THE STRATIFIED QUALIFYING DICTIONARY. `MergeLaw.newgap_le_step`
consumes only `F_2(M)` and `Q_j(M; a)`, and `Q_j` quantifies ONLY over windows whose
interiors reach the next gear's tooth floor `a`. So the whole input is `D_j` = the
realised j-windows with qualifying interiors, one list per depth - and the family
TERMINATES at `j = K + 2` where `K` is the longest qualifying run (3, 4, 5 at m19,
m23, m29 - it does NOT grow like the period). At machine 29, floor 10, budget 74:

    j        2      3      4      5      6      7      8
    |D_j|   730  3,692  6,688  3,915    789     46      0
    Q_j      55     65     68     71     71     71      -     max 71 <= 74, margin 3

15,860 tuples against a period of 1,078,282,205 slots. THE CERTIFICATE'S SIZE IS THE
DICTIONARY'S, NOT THE PERIOD'S: 990 / 2,911 / 15,860 tuples at m19 / m23 / m29 against
periods of 3.8e5 / 3.7e7 / 1.1e9 - about 5x per gear against 30x per gear for the
period. The six `decide +kernel` dictionary checks depend on NO AXIOMS AT ALL.

    step     criterion max(F2, max_j Q_j)   budget F+q'   margin
    11->13   20                             20             0 TIGHT
    13->17   26                             28             2
    17->19   35                             37             2
    19->23   47                             48             1
    23->29   60                             63             3
    29->31   71                             74             3      <- NEW

THE HYPOTHESIS IS NAMED AND FOUR-GATED, AND ANYONE QUOTING THE RUNG MUST QUOTE IT.
`Census29` says those six lists CONTAIN every realised qualifying window of machine 29
and that no six consecutive gaps reach 10. It is a full-period claim; no kernel has
seen machine 29's period and none is asked to (verdict 21). Gates, ALL GREEN
(`research/qual_dict_gate.py`): (1) the period is scanned TWICE with unrelated chunk
sizes (40,000,000 and 23,456,789) and all six dictionaries, the whole F_j ladder and
the run length come out identical - and this gate CAUGHT MY OWN INSTANCE OF MECHANIC'S
RULE 18: my first `gaps_of_period` closed the cyclic seam with the wrap gap but not
with the period's own first gaps, so windows starting at the seam were invisible;
(2) the seam is now explicit and the gap count asserted equal to prod(q-2) =
214,708,725; (3) the Lean literals are parsed back out of the .lean files and compared
as sets with the scan - all six identical; (4) the same scanner at m19 and m23
reproduces F_j(19) = 25,31,35,38, Q_j(19;8) = 31,35,37,38, F_j(23) = 34,39,50,58,65,
77,83,88, Q_j(23;10) = 39,43,50,55,60 and F(29) = 43 - every one a kernel-checked
value in this ledger.

TWO CROSS-LANE CONFIRMATIONS FALL OUT OF THE SAME SCAN:
- F_2(29) = 55 EXACTLY - agreeing with Constructor's A_5(23) survivor closure and
  Mechanic's pair census. Three independent routes, one integer. That is brief item
  (2), delivered as `spectrum29_two` (the bound the rung actually consumes).
- Q_J(29; 10) = 55, 65, 68, 71, 71, 71 for J = 2..7 - EXACTLY the CORRECTED marked
  spectrum of verdict 12c, entry for entry, including the J = 5 value 71 whose
  published predecessor 85 was the DP artefact that had made this rung look lost. A
  FOURTH route confirms the correction, and the rung stands on it.
- Corpus correction: `docs/novel/survivor-generator.md` records machine 29's period as
  "214,709,355 gaps". It is 214,708,725 = prod(q-2), asserted by the gate. Off by 630.
  (The 1,078,282,205 slot figure is right.) Doc updated.

THE GENERATOR IS IN THE KERNEL FOR THE FIRST TIME (proofs/Gen11.lean, brief item 3).
Machine 11's cyclic gap word is 135 letters over 385 slots and gear 13 kills slot
residues 2 and 11, so Constructor's generator is a bounded walk over 135 bases x 13
free phases:

    theorem gen_zero : gen 0 = 11    -- L (x) K* (x) R                     = F(13)
    theorem gen_one  : gen 1 = 16    -- L (x) K* (x) SIGMA (x) K* (x) R  = F_2(13)
    theorem no_truncation : forall i < 135, 30 < off i 13     -- fuel never binds

Both are `[propext]` ALONE, both are produced with NO mention of machine 13's 5005-slot
period, and both match machine 13's own kernel-proved spectrum
(`generator_matches_machine13`). HONEST SCOPE (verdict 20): what is kernel-checked is
that the generator COMPUTES these integers, not that it MUST - the soundness bridge
needs `gw11` certified as machine 11's opening sequence plus the periodicity glue
`opSeq11 (n + 135) = opSeq11 n + 385`. THAT GLUE MERGES TWO OPEN TARGETS: it is the
same lemma as verdict 11's missing depth-sum glue at m13, and doing it once abstractly
("a periodic decidable opening predicate has a periodic enumeration") discharges both.

NOT DONE, WITH THE REASON (not a judgment): brief item (4), the m13 covering dual, was
NOT ATTEMPTED because THE DUAL VECTOR IS NOT ON DISK - `research/sdp_cover.py` has no
save path and no `sdp_*` artefact in `research/data/` carries it, so kernel-checking
1041/2081 means reconstructing and re-solving Lateral's Sherali-Adams level-2 system
rather than transcribing it. FOR LATERAL: round-24's process rule ("feasible verdicts
must save their witness") applies to duals too - save (z, D, the row generator) and the
Lean side is an afternoon. Also: the dictionary-TRANSFER superset was not used as the
final E, because the qualifying family needs depths up to K+2 = 7 and `dict_transfer.py`
at out_m = 7 is far past its design cost; the exact census is smaller and equally
hypothesis-shaped. The transfer remains the right way to DISCHARGE Census29 from
Census23, which is my top round-26 target.

FOR MECHANIC: the per-rung deliverable is now a published finite format - the
stratified qualifying dictionaries D_2..D_{K+2} plus the run bound K, exactly as
`research/qual_dict.py` emits them. A rung above 31 is a census, not a kernel problem.
FOR CONSTRUCTOR: `spectrum29_two` and `qual29_all` are your A_5(23) closure integer and
your marked spectrum, in consumable Lean form.

AND THEN THE SEVENTH RUNG, BECAUSE THE VEHICLE IS A TEMPLATE. Machine 31's full
period (33,426,748,355 slots, 6,226,553,025 gaps = prod(q-2), asserted) scanned in
1,451 s CPU / 24 min wall by the same script; gear 37's teeth are {6, 31} so the floor
is 12 and the budget F(31) + 37 = 95.

    j        2      3      4      5      6      7      8
    |D_j|  1,253  8,155 18,566 13,049  2,120     42      0
    Q_j       68     85     90     91     90     88      -    max 91 <= 95, margin 4

    theorem D_31_37 (h : Machine31.Census31) (n : N) : Machine37.g37 n <= 58 + 37
    theorem g37_le_of_census (h : Census31) (n : N) : Machine37.g37 n <= 91

Gated the same four ways and GREEN (`research/qual_dict_gate31.py`): the whole
33,426,748,355-slot period RESCANNED at an unrelated chunk size (37,000,001 vs
60,000,000), all six dictionaries identical, gap count = prod(q-2) = 6,226,553,025,
F(31) = 58 against the corpus, run 5, and the transcription against the .lean files
identical set for set. Nothing was claimed before the gate finished.

THE QUALIFYING SPECTRUM TURNS OVER AT MACHINE 31 - a new structural fact, and the
first of its kind in this project. Q_j(31; 12) rises 68, 85, 90, 91 and then FALLS to
90, 88, where at m19/m23/m29 it was non-decreasing and then saturated. CONSEQUENCE,
and it is not cosmetic: THE CONSTRAINT THAT BINDS THIS RUNG IS A FIVE-GAP WINDOW WITH
THREE QUALIFYING INTERIORS - not the two-gap statement (68, far under budget) and not
the deepest non-vacuous window (88). Any argument that treats the two-gap statement as
the whole obligation, or assumes the binding depth is the last one, is FALSE from
machine 31 on. Round 24 isolated the two-gap statement as "the whole remaining
obligation of (D)"; at machine 31 it is no longer even the binding one.

    step     criterion max(F2, max_j Q_j)   budget   margin   floor   |dictionary|
    19->23   47                              48        1        8        990
    23->29   60                              63        3       10      2,911
    29->31   71                              74        3       10     15,860
    31->37   91                              95        4       12     43,185

THE DICTIONARY GROWS ~3-5x PER GEAR WHILE THE PERIOD GROWS ~30x (990 / 2,911 / 15,860
/ 43,185 against 3.8e5 / 3.7e7 / 1.1e9 / 3.3e10 slots), and K - the longest qualifying
run, which sets the family's DEPTH - did NOT grow from 29 to 31 (3, 4, 5, 5). That is
the first evidence on the question this vehicle's lifetime depends on.

CROSS-CHECKED AGAINST CONSTRUCTOR BEFORE CITING, and the check changed the claim:
scanfree-certificate.md's "bound 95 <= budget 95" at 31->37 is NOT a loose version of
my 91 - its CEGAR bound column IS the budget at every step by construction
(48/63/74/95 against budgets 48/63/74/95). The lanes agree. Where they are comparable
they match exactly: their gate reports F_1..F_4(29) = [43,55,65,70] and F_2(31) = 68;
my independent full-period scan gives 43,55,65,70 and 68. Two codebases, one set of
integers.

INFRASTRUCTURE, PAID FOR IN A FAILED BUILD AND WORTH THE OTHER LANES KNOWING: a big
Lean list literal has TWO elaborator limits, and the second scales with TUPLE ARITY,
not count. Past `maxRecDepth` the machine-31 modules died with "(deterministic)
timeout at `isDefEq`, maximum number of heartbeats (200000)" - D4 (4-tuples) at about
element 6,760, D5 (5-tuples) at about 3,540, while D3 (8,155 3-tuples) was fine. So
the budget goes roughly as count x arity, and machine 29's modules had merely sat
under it by luck. Fix: `set_option maxHeartbeats 4000000` beside `maxRecDepth`,
applied UNIFORMLY including to machine 29's already-green files - a file that builds
only because it happens to sit under a hidden limit is what bites the next machine.
Both are elaboration resource limits; neither touches the axiom footprint, which
stays EMPTY for all twelve `Dj_ok`.

## LP-duality thread (round 25)

BRIEF: push the product-measure-surviving row - decide 23->29 on a quiet box, map where
the row dies and why, and say what comes into range.  Headline gates for the manager:
`python research/cw_decide25.py GATE` (the four composed rung certificates to the digit,
the consistent-only reference values, the m19 attribution, BOTH refutation branches, their
cross-consistency against the certificate branch, and the saved 19->23 witness re-verified
from disk; prints ALL ASSERTIONS GREEN) and `python research/row_decay.py L W A` (the decay
law and the closed-form identity; every claim asserted, no float anywhere in that file).

### 1. THE DECAY LAW - THE ROW'S FRONTIER IS A WIDTH, NOT A MACHINE (the round's result)

Round 24 measured six numbers for the row's margin against the uniform product measure
(+3.46/+3.27/+2.01/+0.41 at m23/29/31/37, -0.36/-2.95 at m41/43) and read them as "the
row's own vacuity frontier is machine 41".  THE NUMBERS ARE RIGHT AND THE READING IS
WRONG.  In closed form, with s1 = sum_{5<=q<=y} 1/q and pi_i = prod_{k<i}(1 - 2/q_k):

  E_u[f] = (6W/5)(7/10 - s1) + N_+        (exact; N_+ = sum_{1<=i<j} E_u[n_ij] >= 0)
         = W * Pi(y) - Delta(y, W),       Pi(y) = prod_{5<=q<=y}(1 - 2/q)

using sum_r S_q(r) = 2W and sum_{u,v}|P_ij| = 4W exactly, and n_0j = |P_0j| identically
(gear 5 has nothing below it).  The constant 7/10 is the whole of the row's free power and
s1 crosses it BETWEEN m23 AND m29: s1(23) = 0.665623 < 0.7 < 0.700105 = s1(29), clearing it
by 1.06e-4.  LAW == DIRECT as exact rationals at all ten machines m11..m43, reproducing
round 24's six numbers.

THE IDENTITY, PROVED: the second-order lowest-blocker coefficient
A(y) = 1 - 2 s1 + 4 sum_{i<j} pi_i/(q_i q_j) equals Pi(y) EXACTLY - because every blocker
but the lowest is a blocker above the lowest, so E[#blockers] = P(>=1 blocker) +
E[#above the lowest].  Asserted as exact rationals at all 60 machines up to 300.  A(y) is
BOTH an exact upper bound on E_u[f]/W at every width AND its exact limit (for any fixed
lower-gear phase choice, CRT equidistributes P_ij, so the covered fraction tends to
1 - pi_i and so does the max).  Equivalently and more simply: f <= open pointwise and
E_u[open] = W*Pi(y) exactly - the row's margin can never exceed the expected open count.

CONSEQUENCES.  Pi(y) > 0 at every finite machine, so THE ROW IS NEVER UNIFORMLY VACUOUS -
only ever TOO NARROW.  Delta is a pure extreme-value quantity (the summed excess of the
phase MAXIMUM over the phase MEAN inside n_ij) and grows sublinearly - measured doubling
factor in [1.455, 1.682] over ten doublings against a gain that doubles exactly.  So every
machine has a finite threshold W_u(y), exact:

    y          29     31     37     41     43     47     53
    W_u(y)     10     48     83    135    211    362    558
    budget     63     74     95    129    134    150    156
    ratio    6.300  1.542  1.145  0.956  0.635  0.414  0.280

The ratio falls monotonically through 1 between m37 and m41.  Round 24's "m41" is really
"budget(41) = 129 < 135 = W_u(41)", MISSING BY SIX.  New doc:
docs/novel/product-measure-frontier.md (indexed); addendum filed in
docs/novel/moment-degree-ceiling.md (the arity law governs MOMENT families; it does not
govern this row, whose obstruction is a width-to-machine ratio, not a ceiling machine).

### 2. THE EXACT RANGE, AND ONE EXACT NEGATIVE

Under uniform phases every position carries the SAME degree-<=2 moment vector (the product
moments of p_q = 2/q), so ONE exact rational completion decides the whole degree-2 side of
a rung.  Combined with the row's sign this gives a per-rung verdict with no LP run:

    19->23  W=48   deg-2 cuts VIOLATED, row VIOLATED   -> open (cuts bite)
    23->29  W=63   deg-2 satisfied,     row VIOLATED   -> open (the row bites)
    29->31  W=74   deg-2 satisfied,     row VIOLATED   -> open (the row bites)
    31->37  W=95   deg-2 satisfied,     row VIOLATED   -> open (the row bites)
    37->41  W=129  deg-2 satisfied,     row satisfied  -> REFUTED

37->41 IS AN EXACT REFUTATION, not an undecided cell: the uniform product measure is an
exhibited exact feasible point of the FULL composition at width 129, so no certificate of
this vehicle exists there however many cuts are generated.  (41->43 and 43->53 were left
unrun deliberately - the n=12 and n=14 exact completions are memory-heavy and E_u[f] is
already -2.95 and -9.41 there; scoped out, not stuck.)

STAR-k RESTORES THE RANGE.  Holding the k smallest gears' phases explicit gives n^K >= n
pointwise, so E_u[f] can only rise.  Exact, at the ladder's own budget widths:

    y    W    W*Pi(y)   level 2    STAR-3   STAR-{5,7}
   37   95    16.7395   +0.4059   +8.0012     +11.9812
   41  129    21.6217   -0.3646   +8.8853     +14.2963
   43  134    21.4151   -2.9469   +6.6797     +12.7830
   47  150    22.9521   -5.9284   +5.0991     +12.2560
   53  156    22.9694   -9.4094   +3.1065     +10.8054

So the FAMILY is not out of range at m41 - only the level-2 member is.  Holding ONE gear
(the 5) turns -0.36 into +8.89 and keeps the necessary condition alive through m53; it
eats 42% of Delta at m41 and 71% for two gears.  It does NOT change the slope (still
bounded by W*Pi(y)); it buys frontier.  Cost: the pair blocks become triples, columns x5
then x35.  NAMED NEXT CONSTRUCT: build the STAR-3 composition as an actual LP and see
whether the necessary condition becomes a certificate.  (JUDGMENT, NOT RESULT: I expect it
does not - the uniform margin is necessary, not sufficient, and the parent vehicle missed
19->23 by a wide margin at a width where its uniform margin was +3.46.)

### 3. THREE ROUND-24 CORRECTIONS, ALL MY OWN LANE'S

C1  "The row's uniform frontier is machine 41" - WRONG FRAME (section 1).
C2  "The composition certifies width 33 at m19 where NO degree-2 cut certificate of any
    kind exists" - REFUTED.  That conflated two relaxations.  The BLOCK-INDEPENDENT
    degree-2 relaxation is feasible at 33 (W*_indep(19) = 36); the CONSISTENT one is NOT:
    exact certificate at width 33 with NO recursive row, 57481/2048 < 114989/4096, 20,919
    ops, 573 rows (research/_w19cons.py).  THE m19 WIDTH BELONGS TO CONSISTENCY, NOT THE
    RECURSION - which answers the attribution question round 24 left open.  Filed in
    consistency-over-degree.md as the m19 table row (W*_cons(19) <= 33 vs W*_indep = 36).
C3  "Certificates 2-3x smaller" - TRUE AT BUDGET WIDTHS ONLY.  Re-derived through one code
    path: composed 562/1,456/3,303/8,379 vs consistent-only 464/2,868/9,091/25,413 at
    m11/13/17/19, i.e. 1.21x LARGER at m11 then 1.97x/2.75x/3.03x smaller.  But at m19
    width 33 it is 19,653 vs 20,919 - a factor of 1.06.  The saving is a budget-width
    phenomenon.  (Three of the four rung certificates reproduce byte-identical; the 17->19
    dual point is path-dependent - 905/24 < 1207/32 at 8,379 ops this round vs
    1207/32 < 1811/48 at 8,179 in round 24.  Both are valid certificates of the same fact;
    the gate asserts the certified RELATION, not the dual point.)

### 3b. THE BRIEF'S QUESTION: 23->29 IS DECIDED - REFUTED.  AND SO IS 19->23.

Round 24 left 23->29 NOT DECIDED (its decider was starved) and 19->23 UNDECIDED ("no
certificate found", which is a stalled search, not a verdict).  BOTH ARE NOW CLOSED, each
by an EXHIBITED EXACT RATIONAL FEASIBLE POINT of the full composition at the budget width -
every block summing to 1, every consistency link exact, EVERY position's degree-<=2 moments
completable by exact rational Farkas, and the recursive row satisfied:

  19->23, W=48:  cut loop reproduces round 24 to the digit (t = +0.0368, 1,640 rows, 54
                 iterations) and then PASSES A FINAL EXACT PASS AT MARGIN ZERO.  Witness by
                 the margin-repair construction; row value
                 94976919931521604605719632036962147283379323 /
                 1957041031234328038937562750125000000000000 >= 48, slack +0.5309.
  23->29, W=63:  cut loop converges at t = +0.1363 with 1,451 rows in 29 iterations.
                 Witness by the double-centred construction; row value
                 434038501259968447/6799020800000000 >= 63, slack +0.8384.

SO THE COMPOSED VEHICLE PROVES NO CERTIFICATE AT EITHER STEP - not "we could not find one".
Both witnesses are saved and re-verified from disk by a second pass in a clean process
(research/data/r25/witness_m23_w48.pkl, witness_m29_w63.pkl; gate step 5).  With the
uniform-point refutation at 37->41 (section 2), the composition's rung ladder is CLOSED at
exactly the four rungs it already had, and round 24's pre-registered E5 ("no new rung") is
confirmed with proofs instead of empty searches.

NOTE THE SHAPE OF THE FAILURE.  At 23->29 the LP optimum sits at t = +0.1363, not near
zero: the vehicle is not marginally short there, it is comfortably infeasible-to-certify.
The row's uniform margin at that cell is +3.27 - i.e. CUTTING THE PRODUCT MEASURE IS A LONG
WAY FROM CERTIFYING, which is the honest calibration of what "first object past the vacuity
ceiling" is worth.

### 4. A PROCESS CORRECTION TO MY OWN ROUND-24 RULE (the round's methodological finding)

Round 24's rule was "FEASIBLE VERDICTS MUST SAVE THEIR WITNESS".  Necessary, NOT
SUFFICIENT: the witness has to be EXACTLY IN THE POLYTOPE, and rationalising a float LP
point does not put it there - the consistency links are sums that rounding does not
preserve.  It happens to work at machine 13 and FAILED at machine 19, where my hardened
assertion fired mid-bisection.  Had round 24 exercised this branch it would have recorded
a feasible verdict on a point that is not feasible - the same class of error as the
round-23 section-G regression, caught this time by the assert.

STRENGTHENED RULE: an exact FEASIBLE verdict must come from a point that is consistent BY
CONSTRUCTION.  Two such routes are now built and gated (research/cw_decide25.py):
  * GLOBAL POINT - a rational mixture over full phase tuples; its degree-2 marginals come
    from one distribution, so every consistency link holds at EVERY level, which refutes
    the whole Sherali-Adams/Lasserre hierarchy at once, not just level 2.  Float discovery,
    exact verification (the cut rows only steer the search; the verdict rests on
    completability + the row, both exact rational).
  * DOUBLE-CENTRED POINT - round the singles to one denominator, subtract the product
    measure from each pair block, double-centre the residual (which zeroes every row and
    column sum exactly), and shrink for nonnegativity.  Consistent by construction, but it
    collapses toward the product measure when the LP point is vertex-like, and at machine
    23 the product measure VIOLATES the degree-2 cuts, so it cannot produce a witness
    there.  MEASURED, not guessed: it failed at position 4 at every denominator tried.
  * MARGIN-REPAIR POINT - the one that works, and the one that closed 19->23.  Round the
    pair block, rescale it to carry mass exactly 1 - eta, then add the OUTER PRODUCT of
    the row and column margin deficits divided by their common total: that restores both
    margins EXACTLY (the row picks up d_u * (sum e)/delta = d_u) and the correction is of
    rounding size, so the repaired point is a rounding perturbation of the LP's own point
    rather than a pull toward uniform.  The eta floor is what makes every deficit positive,
    hence the correction nonnegative.  Its own trap, found and fixed by the assert: when
    delta = 0 the total mass is right while the ROW masses need not be, and adding nothing
    leaves the links broken - so delta = 0 is rejected, not treated as "nothing to do".
Both branches are cross-checked against the certificate branch: at m13 W=20 and m19 W=33,
where exact certificates exist, the refutation branch must find NOTHING - and does.

### 5. INFRASTRUCTURE: WHAT ACTUALLY STARVED THE ROUND-24 m29 RUN

Round 24 recorded the m29 starvation as other lanes' memory pressure.  That was half the
story.  MEASURED this round: `RelaxC.rationalise` divides each block by its own rational
sum, so denominators LCM together; at machine 29 the moment vectors entering the exact
separation oracle reached 307 BITS BY ITERATION 3.  The deciding process ran at 1.35 GB of
COMMITTED private memory against a 136 MB working set with 604,322 page faults - i.e.
page-thrashing at 27% of one core on a 19%-LOADED box.  Fixing the denominator (round to
one fixed den; the rationalised point is only a place to LOOK for cuts, and every cut
`separate` returns is re-asserted valid and violated on its own) restored it to ~96% of a
core.  THE LESSON, and it generalises past this lane: in exact-rational pipelines an
innocuous normalisation step is a memory bug, and the symptom is indistinguishable from
another lane hogging RAM.  Check commit-vs-working-set before blaming the neighbours.

### 6. NEGATIVES AND WHAT IS NOT DONE

- 23->29 and 19->23 are REFUTED (section 3b), 37->41 is REFUTED (section 2).  29->31 and
  31->37 are still OPEN cells for this vehicle - the uniform point does not refute them -
  but they are downstream of a refuted rung, so nothing rests on them.
- A separate GLOBAL-POINT refutation (the stronger kind - it would kill every
  Sherali-Adams/Lasserre level at once, not just level 2) was attempted at 19->23 and
  23->29 with pools of 120, 160 and 500 phase tuples and found NOTHING.  A pool-limited
  failure proves nothing, and is recorded as such; the level-2 refutations above stand on
  their own witnesses.
- The m23 width-60 probe (how far above budget the composition's own threshold sits) was
  STOPPED at iteration 38 with t = +0.0150 still falling, to free memory for the two
  decisions.  W*_comp(23) is therefore NOT bounded above; deliberately scoped out.

- W*_cons(19) is bounded (<= 33) but NOT pinned - a bisection below 33 needs an exact
  feasible verdict at the trial width, which is exactly what section 4 says is hard.
- 41->43 and 43->53 uniform-point verdicts scoped out (see section 2).
- The prior-art check for the closed form and the identity is NOT RUN (no web access).

## Lateral round 26

OWN MANDATE, OWN CHOICE: the brief's optional item (the parity lever's second half)
plus backlog U4, because they are ONE object - the lever's reach is decided by where
the mirror's FIXED POINTS sit, and the Farey spectrum's multiplicities are exactly the
gap histogram's residue-class counts, whose parities are those same fixed points.
GATE: research/mirror_lever2.py --parts ABCDEFG -> 52 ASSERTION GATES, exit 0
(log research/data/mirror_lever2.log). Ten predictions pre-registered in
research/data/r26_lateral_predictions.txt before any of it was coded. Every job this
round launched has finished; nothing left running.

THE LEVER IS NOW A TOOL, AND ITS CEILING IS PROVED.
- THE EXCEPTIONAL WINDOW MOVES FROM AN INDEX TO AN ADDRESS. Round 25 located the one
  self-mirror depth-j window by its INDEX t_j = -j/2 (mod N), which needs an enumerated
  period. In slot space a depth-j window is self-mirror iff its endpoints sum to 0
  (mod P), i.e. iff it is CENTRED ON SLOT 0 (g even) or ON THE ANTIPODE (g odd), giving
      j even: g_j* = 2 o_{j/2};   j odd: g_j* = 2 b_{(j+1)/2} - P
  from the openings just above 0 and just above (P-1)/2 - a few dozen sieved slots, SO
  g_j* IS SCAN-FREE AT EVERY MACHINE (table computed to m53, j <= 12). COROLLARY
  g_j* = j (mod 2): W_j(g) is EVEN for every g of the wrong parity with NO computation.
  Verified against the exact full-period W_j census at m11..m29 for every j <= 12 - the
  odd column is exactly {g_j*}, no exceptions.
- g_1* = 1 ALWAYS, AND IT IS THE T3 LAW. P = 0 (mod q), so the antipodal slot (P+1)/2 is
  inverse(2) = (q+1)/2 mod every gear; multiply by 6 and it is 3, while 6(+-u) = +-1. So
  it is a tooth only if 3 = +-1 (mod q) - impossible for q >= 5. THE ANTIPODAL SLOTS ARE
  OPEN AT EVERY MACHINE, so W_1(g) IS EVEN FOR EVERY g >= 2, UNCONDITIONALLY. Round 25's
  caveat on the maximal gap is discharged for ever: the maximal gap never occurs once.
- AND THE LEVER HAS NO SIDE CONDITION ON (D). The merge law quantifies only over
  QUALIFYING windows (middle gaps >= the floor a = 2u'), and the exceptional window sits
  where the machine's gaps are SHORTEST, so it is never qualifying - checked at all 11
  rungs x 6 depths, 66/66 negative. SO: an exact bound "at most ONE qualifying depth-j
  window exceeds the budget" proves there are NONE. FOR THE MANAGER/CONSTRUCTOR: a
  first-moment argument on the family only has to reach FEWER THAN TWO, not fewer than
  one. (Reported for the route, not developed - mandate.)
- THE CEILING, PROVED, SO NOBODY BUILDS ON A HOPE. The affine maps preserving the
  opening set are exactly the 2^m multiplications by c = +-1 mod every gear (b = 0
  forced), of which only c = +-1 mod P preserves adjacency; and dropping affineness
  entirely, the only rotations/reflections of the circle Z_P preserving the openings are
  the identity and the mirror. THE FULL SYMMETRY GROUP IS Z/2, EXACTLY (brute-forced
  over all 92,400 affine maps at m11 and all 2P rotations+reflections at m11/m13). So
  the lever is worth EXACTLY ONE UNIT - a factor of two - and there is no mod-4 version
  to hope for from any symmetry of the machine.
- THE FIXED-POINT CRITERION. For a PALINDROMIC tuple of span s the occurrence set is
  mirror-invariant with exactly one candidate fixed address k_w = -s/2 (mod P), so
  #occ(w) is ODD iff w occurs there - an O(#gears) test. At w = (g,g) it forces openings
  at -g, 0, g with nothing between, i.e. g = k_1: round 25's "(k_1,k_1)" in one line, now
  at every arity. Gated on every palindromic 2- and 3-tuple at m11..m23.

A 46% SAVING SITTING IN MECHANIC'S OWN LOGS. The mirror sends an occurrence of a word w
at k to one of reverse(w) at -(k + span w), bijectively, so #occ(w) = #occ(reverse w)
EXACTLY and realisability is reverse-invariant - kill words included, since both the old
machine's openings and the new gear's teeth are negation-symmetric. Gated: the realised
4-tuple dictionaries at m23/29/31/37 are exactly reverse-closed (15,696 / 45,854 /
115,193 / 291,675 tuples; at m37 that is 145,907 reverse classes, a 50.0% decision
saving). AUDIT of research/data/r24/akillp_*.log: 82 word decisions, EVERY reverse pair
agreeing - the theorem's falsifiable gate - and 12,877 s of 27,946 s (46%) spent
deciding the SECOND member of a reverse pair, including two of the four span-141 words
at 47->53 that cost 20,005 s between them. FOR MECHANIC: decide one word per reverse
class and copy the verdict; the check is `w[::-1] in decided`.

U4 CLOSED - AND IT CORRECTED TWO OF MY OWN PUBLISHED COROLLARIES. Writing a path
eigenvalue 2cos(pi j/(g+1)) as 2cos(pi a/b) in lowest terms, b | g+1 and for FIXED b
every a coprime to b comes from exactly the gaps g = -1 (mod b):
    THEOREM  mult(2 cos(pi a/b)) = Sigma(b) = #{gaps = -1 mod b}, independent of a.
So the eigenvalue multiplicities of A ARE the gap histogram's residue-class counts, and
they invert: W_1(b-1) = sum_t mu(t) Sigma(tb), with F+1 = max{b : Sigma(b) > 0}. That is
the CONSTRUCTIVE form of round 23's negative ("every unitary invariant of BS is the gap
histogram"). With g_1* = 1: EVERY multiplicity is EVEN except the eigenvalue 0's.
SELF-CORRECTION 1: round 22's "#distinct = |Farey(F+1)| - 2" assumed every gap length
1..F is realised. It is really a DIVISOR-CLOSURE statistic of the REALISED set, so HOLES
break it. True counts 21 / 41 / 113 / 183 / 363 / 549 / 981 / 1,813 / 2,467 at m11..m41
against the published 21 / 45 / 119 / 211 / 383 / 603 / 1,085 / 2,455. Only m11 (the one
holeless machine) was right. LOSS RULE, exact at all nine: loss = sum phi(hole+1) - e.g.
phi(85)+phi(88)+phi(90) = 128 at m41, PRE-REGISTERED AND MATCHED. THE HOLE LIST IS A
SPECTRAL OBSERVABLE: the arithmetic-selection object nobody can predict is exactly the
defect between the true level count and the Farey count.
SELF-CORRECTION 2: round 22's "s_min/s_mean descends to Hall's 3/pi^2" was also computed
on the full Farey set; on the true level set it is not monotone and dips to 0.2422 at
m37. WHAT SURVIVES: P(s < 0.1 mean) = 0 exactly - now PROVED, since deleting levels only
lengthens spacings - and <r~> stays above GUE (0.63-0.72 at m11..m41). The round-22
conclusion is unchanged; two of its numbers are not.

AND A THIRD SELF-CORRECTION, THIS ONE OF LAST ROUND'S "CONFIRMED" PREDICTION. Because
W_1(1) is the only odd histogram entry and 1 = 1 (mod p) for every p,
    N_1^(p) = #{gaps = 1 mod p} is ODD and every other N_r^(p) is EVEN,
at every machine and every modulus. Hence alpha_1(p) = N_2 - N_1 - N_0 + N_{p-1} =
even - ODD - even + even is ODD, so the gear-p bracket is never exactly real:
    THE POLE PHASE IS UNATTAINABLE AT EVERY GEAR, not only at gear 5.
Round 25's prediction P3 ("gear 5 is the ONLY parity-obstructed gear for p <= 37") was
scored CONFIRMED and called the structural half of backlog U3. IT IS WRONG AS WORDED:
its GF(2) test decided whether the CELL-MATRIX constraints alone force the contradiction
- a strictly narrower question, since those constraints know nothing about W_1(1).
Round 25's gear-5 theorem stands; its uniqueness claim does not, and the gear-5 vs
gear-7 story is now entirely round 25's MEASURED half (three equations not one,
asymmetries an order of magnitude larger and slower to decay). Corrections filed in
docs/novel/gear-cell-decomposition.md (a box before the table), the README index entry,
and a bracketed note at the one cross-lane citation in phase-saturation-arity.md.
STANDING LESSON, general: A SATISFIABILITY VERDICT OVER A CHOSEN CONSTRAINT SET IS A
STATEMENT ABOUT THAT SET, NOT ABOUT THE MACHINE - label it with the set.

PREDICTIONS: 10 pre-registered, 10 CONFIRMED - and I do not read that as a good
scorecard. Six were corollaries of theorems already proved when I wrote them, so they
were cheap; the real bets were the m41 loss of exactly 128, the loss rule's exactness,
and alpha_1(p) odd at every gear (which overturned a prediction this lane scored
CONFIRMED last round). The risk this round was carried by the three refutations, ALL of
my own published record, two of them numbers other lanes could have cited.

NOT WORKED, HONESTLY: U5 (the 613 cosine near-collisions at m31) - untouched, not
weakened, still unclaimed. U7 (gear-7 cells) - RE-POSED rather than run: item 56 answers
its parity half for every gear at once, so the live question is now which cell orbit
carries the measured drift, not whether gear 7 is obstructed. U6 (-1/phi overshoot) and
U9 (the plateau break direction) are BLOCKED on one full-period m37 or m41 gap
histogram; Mechanic's h37 workers were still running at my round close, and the moment
that array exists both are a five-minute computation here.

FOR OTHER LANES:
- MECHANIC: the 46% reverse-class saving above, and the same halving on dictionary
  builds. Also: g_j* for any machine, any depth, from a few sieved slots.
- FORMALIST: three cheap kernel targets in increasing size - g_1* = 1 (five lines:
  6s = 3 and 6u = +-1, and 3 != +-1 mod q for q >= 5) which implies "every gap length
  >= 2 occurs an even number of times"; the fixed-point criterion (a decidable
  membership test at ONE address); and round 25's parity theorem, which item 56 now
  derives more simply.
- LP THREAD (UNTESTED, offered not claimed): the covering CSP is invariant under
  negating every gear offset together with reflecting the window, so its feasible set is
  symmetric and any LP/SDP relaxation may be restricted to the symmetric subspace
  without loss - roughly halving the variable count. Not built or measured here.
- ANYONE citing the round-22 distinct-level table: use the corrected numbers above.

## Harvester round 26

GATES, all five re-run from clean processes at round close, all GREEN:
  .venv/Scripts/python.exe research/j2_referee.py      -> ALL ASSERTIONS GREEN
  .venv/Scripts/python.exe research/j2_citesweep.py    -> ALL CHECKS GREEN  (NEW)
  .venv/Scripts/python.exe research/j2_layer_proof.py  -> ALL ASSERTIONS GREEN (NEW)
  .venv/Scripts/python.exe research/j2_odc6.py         -> ALL ASSERTIONS GREEN
  .venv/Scripts/python.exe research/j2_rankin_layer.py -> ALL ASSERTIONS GREEN
j2_referee.py was run FIRST, before anything below entered the record. Every job
this round launched has finished; nothing left running.

BRIEF ITEM (a) - THE VERDICT IS **PROOF**. The k = 2 Rankin layering is written
out with constants.

    THEOREM (P2').  Let pi_2(t) <= c_1 t/(log t)^2 for t >= t_1.  With
    A = log x, B = log A, C = log B,

        j_2(P(x))  >=  ( 1/(18 c_1) + o(1) ) x A^3 C^2 / B^4 ,

    and for the k-class Jacobsthal function j_k (k classes per prime),

        j_k(P(x))  >=  ( k/((k(2k-1))^k c_1^(k)) + o(1) ) x A^(2k-1) C^k/B^(2k).

    Headline constant 0.0127524 (Lichtman's record c_1 = 3.29956 x 2C_2 =
    4.356487); 0.0052597 with Selberg's fully-explicit 8 x 2C_2.

Proof in docs/novel/layered-erdos-rankin.md sec. 4; assembly asserted in
research/j2_layer_proof.py. FIVE THINGS THE WRITE-OUT ADDED beyond
bookkeeping-to-proof:
1. THE GREEDY LEMMA IS EXACT. Two distinct classes mod p always capture >= 2N/p
   of any finite set - no O(N/p^2) loss (one-line proof; asserted at every prime
   <= 200 and over 40,000 random distributions). This was the step I named IN
   ADVANCE as the likely break. It is the safest step in the argument.
2. A BETTER CONSTANT: the small-prime cut must satisfy P > L y/x ~ A^(2k-1), so
   P = A^(2k-1), not round 25's fixed A^5. Denominator 100 -> 36 at k = 2, a
   FACTOR 2.778.
3. A SELF-CORRECTION OF MY OWN ROUND 25: P = A^5 is admissible only for k <= 3
   and is INADMISSIBLE for k >= 4, so round 25's printed general-k constant is
   too optimistic there. The POWER 2k-1 is unaffected.
4. THE CONSTANT IS A SUPREMUM, NOT A MAXIMUM: the medium-prime parameter needs
   theta > k; at theta = k exactly the smooth term beats the tuple term by a
   factor tending to infinity. Hence the o(1).
5. A CONSTANT-LEVEL CALIBRATION where round 25 could only check the shape: the
   IDENTICAL write-up at k = 1 returns (1+o(1)) x A C/B^2 against Rankin's proved
   (e^gamma + o(1)) = 1.781 - a factor 1.781 BELOW the classical constant, which
   is the correct side. Coming out above Rankin would have been a bug.
AND ONE STRUCTURAL STATEMENT: the FGKT/Maynard upgrade of the ordinary
construction needs MANY PRIMES in one residue class; its k = 2 analogue needs
many TWINS in one residue class - a LOWER bound for twins, i.e. the parity
barrier. THE CONSTRUCTION IS PARITY-FREE EXACTLY BECAUSE IT STOPS AT RANKIN
LEVEL. That is structural, not a gap someone will close.

THE PRIOR-ART FINDING, AND IT IS A SELF-FOUND DOWNGRADE OF MY OWN ROUND 25.
**Ford-Konyagin-Maynard-Pomerance-Tao, arXiv:1802.07604, REMARK 7** - read
first-hand 2026-08-29 - names our sieving system explicitly: "a two-dimensional
system in which I_p = {0 (mod p), 2 (mod p)} for all primes p", says "the
'trivial' bound coming from these methods would give a bound of
>> log X log log X for the largest gap between lower twin primes up to X", hopes
one "could possibly ... improve this bound by a small power of log log X", and
notes that "a sieve upper bound ... combined with the pigeonhole principle
already gives a bound of >> log^2 X in this case". Round 25's "nobody appears to
have asked what happens with two classes per prime" is WITHDRAWN. Three
consequences, all arithmetic, all asserted:
- **NOVELTY QUALIFICATION ON (P1)**: in covering coordinates their trivial bound
  IS >> z log z, the order of (P1). (P1) remains the first PROVED bound, the
  first with an explicit constant (1.349) and the first stated for h_2; it is NOT
  the first appearance of the order.
- **THEY HOPED FOR A SMALL POWER OF loglog X; (P2') GIVES TWO FULL POWERS**
  (A^(2-o(1))), by a different route. FKMPT flagging the two-dimensional case as
  out of reach for their machinery is the sharpest available statement of what
  the construction contributes.
- **NO TWIN-PRIME-GAP COROLLARY MAY BE CLAIMED** - their pigeonhole bound
  (log X)^2 = x^2 beats anything (P2') implies about actual twin primes. It is
  NO obstruction to a statement about j_2, where the sifted set has density
  ~1/A^2 and the same pigeonhole gives only (log x)^2. Now an explicit
  not-claim.

BRIEF ITEM (b) - (P3), THE PAIRED-IWANIEC PROBLEM: **PRICED, NOT REACHABLE.**
What round 26 changes: **a >= 3 is now FORCED** (a >= 2k-1 in general), so (P3)
stops being "is h_2 polylog?" and becomes "**is the exponent exactly 3?**", with
the sharp falsifiable conjecture h_2 = z(log z)^(3+o(1)). Why it is unreachable,
on checked facts rather than judgment: (i) the k = 1 case is a known open Erdos
problem - Erdos problems #687 (with a $1000 prize) and #970, both OPEN, both
confirming Iwaniec 1978's j(P(z)) << z^2 is STILL the record after 48 years
(re-checked 2026-08-29); (ii) j_2 >= j by the collapse transfer, so a polylog
bound for j_2 gives one for j; (iii) our own upper ladder sits AT a sifting
limit, which no level refinement moves. THE JUDGMENT PART, LABELLED: that no
other route exists - JUDGMENT, NOT RESULT. What IS reachable is a referee tool:
any claimed j_k << x A^f(k) with f(k) < 2k-1 is contradicted outright.

BRIEF ITEM (c) - UNIT 1 IS ASSEMBLED AND HANDABLE.
**docs/novel/j2-upper-bound.md SECTION 11** is the submission candidate: 11a the
complete ladder in one table (1; 3E quasi-polynomial with exact asymptotic
constant 2 lambda_* = 7.182242; 2E exponent 19; 2E' 17; 2E'' 15; **2G 8.04162**;
2G-inf floor 7.93727; 2 at 4.266 by citation) plus WHICH RUNG TO QUOTE (2E''
below p_n ~ 3.8e5, 2G above by p_n^6.96) and the explicitness boundary stated
once; 11b the current sandwich naming both retracted readings; 11c a rewritten
EIGHT-item not-claims list (two items 4a never had); 11d the ODC root; 11e a
submission checklist. Round 25 had listed these changes but never applied them,
so the file's status block, section 1 and section 4a still carried "exponent
19", "a measured truth of (p^2-p)/2" and "no lower bound beyond the collapse".
All four stale places are now individually MARKED, not deleted.

THE ODC ROOT, RESTATED AS OUR READING WITH THE DERIVATION - round 25's
"discrepancy in the book" framing is WITHDRAWN. The book says "A numerical
computation gives (use the Taylor expansion at 1/4)". Doing exactly that:
f(1/4) = -0.0741009117, f'(1/4) = +4.9715909084, and ONE first-order
Taylor/Newton step gives 1/4 - f(1/4)/f'(1/4) = **0.2649048691** - the printed
**0.264904 to seven digits**. So the printed value IS the book's own stated
approximation, computed the way the book says. Ours is a SHARPENING of a stated
approximation (exact root 0.2652636746, beta_2 = 7.583827 vs printed 7.594004),
carrying the caveat that the equation is OUR reading of a page image. Nothing in
Theorem 2G moves.

THE CITATION-NUMBERING SWEEP IS NOW A GATE - research/j2_citesweep.py. It
re-derives the ODC root; extracts every arXiv id from the five Unit-1 documents
and FAILS on any not in an adjudicated registry (so a new citation must be
adjudicated before use); scans six forbidden strings (the "Iwaniec-Kowalski
Theorem 6.9" chimera, "M. Franze", Tenenbaum 4.3/I.4.2, Costello-Watts under
1208.5342, "Sean Blight"); scans for internal contradictions; and fails on
prior-art checks older than 14 days. **IT CAUGHT TWO LIVE DEFECTS ON ITS FIRST
RUN**: paired-jacobsthal-values.md still attributed the Costello-Watts bound to
arXiv:1208.5342 (it is 1306.1064; the correction was recorded in round 23 and
the document never updated), and j2-upper-bound.md section 6a item 4 still
instructed "cite 2 kappa + 0.4454" while section 9c had SETTLED the conflict for
19/36 - a direct self-contradiction inside one document, which three rounds of
manual sweeps passed.

NEGATIVES AND COSTS:
- MY OWN CONSTANT WAS WRONG BY A FACTOR OF TWO in the first draft: I read
  Selberg's classical twin constant 8 as multiplying C_2 when it multiplies the
  full singular series 2 C_2, and the error flattered the result. Caught only by
  going to Lichtman arXiv:2109.02851 first-hand for a number I thought I knew.
  By coincidence the wrong value 5.2813 is exactly Bombieri-Davenport's 1966
  constant, which is why nothing looked odd.
- ROUND 25's NOVELTY SENTENCE WAS TOO STRONG AND IS WITHDRAWN. Round 25 read
  FKMPT's abstract and main theorem and cleared the paper; Remark 7 of the same
  paper names our system.
- ROUND 25's GENERAL-k CLOSED FORM IS WRONG FOR k >= 4 (my own, one round old).
- (P1)'s NOVELTY IS QUALIFIED (above).
- (P2') has NO finite-z content and NO kernel check: the construction does not
  exist below log z ~ 300, and the threshold, though effective, decays like
  (log C+1)/C with C = logloglog log x and is not writeable. (P1) remains the
  bound to quote at any z anyone will evaluate.
- THE ODC PAGE-IMAGE CAVEAT IS STILL OPEN: (5.38), (6.69) and p. 74 were not
  re-fetched this round either. One library visit closes it; it should happen
  before submission.
- Pre-registration went 5/5, which deserves its own caveat: the predictions were
  made after a round of thinking about the same construction, so they were not
  hard predictions.

THREE ADDITIONS TO THE STANDING CITATION-HYGIENE LESSON (harvester 7d):
6. **A PAPER'S REMARKS ARE WHERE YOUR PROBLEM LIVES.** When clearing a paper for
   prior art, read its remarks and its "what our method cannot do" section
   FIRST - authors put the thing you are doing exactly there. Same failure shape
   as round 25's "price the propositions, not only the theorems", one level
   further down.
7. **A CONSTANT YOU THINK YOU KNOW IS A CITATION.** Normalisations are the part
   of a remembered constant that goes wrong. Re-read the normalisation, not just
   the digits.
8. **A MANUAL SWEEP DOES NOT FAIL.** Any standing referee step that can be a
   gate should be one.

RANKING CHANGES: N4 (upper ladder) stays TOP but becomes a WRITING item, not a
research item. P1-P3 (lower ladder) REACHES PARITY and is now the more active
side. (P2') CLOSED; (P3) PRICED AND CLOSED as unreachable - it should appear in
future briefs as the frontier, not as a target. NEW ITEM ranked below N4:
**(P6) the k-family j_k as a published object** - defined, proved lower bound at
every k, stated upper conjecture, appears nowhere in the literature; its one
piece of real work is the k >= 4 shift set (no set of >= 4 integers has all
pairwise differences a power of 2, so the O(1) offending primes must be handled).
7c#4 (h_2 at p_n = 151..251) RISES AGAIN: it now separates z(log z)^2 from
z(log z)^3.

FOR OTHER LANES:
- ANY LANE CITING A CONSTANT FROM MEMORY: see lesson 7. This lane just lost a
  factor of two to a normalisation it was sure of.
- FORMALIST: nothing new is kernel-reachable here - (P2') is asymptotic. The
  standing offer of finite kernel candidates (harvester sec. 8) is unchanged.
- MANAGER: Unit 1 is handable as a submission candidate. What remains is LaTeX,
  one library visit for the ODC pages, and a scope decision on whether the
  per-difference family F_d travels with Unit 1 or splits off.

## Formalist round 26

GATES, all re-run at round close from clean invocations, all GREEN:
  lake build                       -> **Build completed successfully (1426 jobs)**
  lake env lean AxiomCheck.lean    -> 60+ footprints, ONLY [propext, Classical.choice,
                                      Quot.sound]; zero custom axioms, no native_decide,
                                      no ofReduceBool; `Machine11.ow_135`,
                                      `Machine13.ow13_1485`, `Machine29/31.Dj_ok`
                                      depend on NO AXIOMS AT ALL
  lake env lean DepAudit.lean      -> DEP AUDIT GREEN (new gate, below)
Zero sorries. Every job this round launched has finished; nothing left running.

THE BRIEF'S ITEM (1) WAS RIGHT AND IT PAID TWICE. `proofs/Periodic.lean` proves the
PERIODIC-ENUMERATION LEMMA once, abstractly - no machine, no gears, nothing but omega:

    next_shift : E periodic mod P  ->  next (k + P) = next k + P
    op_shift   : op N = op 0 + P   ->  op (n + N) = op n + P  for every n

The mathematics is all in `next_shift` (periodicity makes `next k + P` an E-point above
`k+P`, and pulling `next (k+P)` back by P makes one above `k`, so the two minimality
facts pin the values to each other); `op_shift` is a one-line induction whose only
machine-specific input is the FINITE fact `op N = op 0 + P`. Instantiating it:

  * `Machine11.opSeq_shift : opSeq (n + 135) = opSeq n + 385`   - verdict 20's step (ii)
  * `Machine13.opSeq_shift : opSeq (n + 1485) = opSeq n + 5005` - VERDICT 11's missing
    step, open since round 22 (the depth-sum identity's glue at m13).

ONE LEMMA, TWO STANDING GAPS, exactly as the brief predicted. The base cases are
kernel-computable because `Nat.find` is replaced by the `seekT` walk that `seek_next`
already proves equal to it: `ow_135 : ow 135 = 385` and `ow13_1485 : ow13 1485 = 5005`,
both `decide +kernel`, both with EMPTY axiom footprints.

AND THEN THE BRIDGE CLOSED COMPLETELY: THE GENERATOR IS SOUND AT 11 -> 13.
`proofs/Gen11Sound.lean` - round 25 could only assert that two computations AGREE;
round 26 proves the generator MUST give machine 13's spectrum:

    theorem generator_sound :
        SpectrumBound g13 1 11 /\ SpectrumBound g13 2 16 /\
        SpectrumBound g13 3 23 /\ SpectrumBound g13 4 26

F_1..F_4(13) <= 11, 16, 23, 26 - THE EXACT VALUES - derived from MACHINE 11's
135-LETTER WORD, with machine 13's 5,005-slot period nowhere in the derivation. Three
ingredients: `gAt_succ` (gw11 IS machine 11's gap word, at every index), the periodicity
glue, and `walk_sound` (the walk SIMULATES the machine, by induction on the fuel against
the invariant "x+d is the k-th machine-11 opening after x, and exactly `surv` machine-13
openings lie in (x, x+d]").

THE INDEPENDENCE IS GATED, NOT ASSERTED - new tool `proofs/DepAudit.lean` walks the
transitive CONSTANT CLOSURE of the proof term (something `#print axioms` cannot see):
"DEP AUDIT GREEN: Gen11.generator_sound closes over 3858 constants; all 11 positive
controls reached; none of the 15 machine-13-period constants is among them."

THREE FINDINGS THE SOUNDNESS PROOF FORCED, all corrections of my own lane's round-25
record:
 (a) `gw11`'s base is ONE OPENING EARLIER than the enumeration's - the true identity is
     `gAt (i+1) = g11 i`. The generator's VALUE is unaffected (it maximises over all 135
     bases), but "gw11 is machine 11's gap word" as round 25 worded it is off by one.
 (b) THE BAIL VALUE HAD TO BECOME A SENTINEL. `walk` returned 0 when the span cap or the
     fuel ran out - sound for a MAXIMUM, fatal for a BOUND, because a walk that gives up
     is indistinguishable from a short gap. With bail = 999, `gen ns < 999` is ITSELF the
     proof that no walk bailed. Values unchanged, and two more landed: `gen 2 = 23`,
     `gen 3 = 26`, so the generator reproduces the whole published ladder 11, 16, 23, 26.
     GENERAL LESSON FOR EVERY LANE: a search that returns a neutral value on failure can
     certify a maximum and never a bound. Ask of any computed maximum used as an upper
     bound: what does it return when it fails?
 (c) AN AUDIT WITHOUT POSITIVE CONTROLS IS NOT AN AUDIT. The first DepAudit passed
     vacuously (310 constants instead of 3,858) because `ConstantInfo.value?` does NOT
     return a THEOREM's proof term in Lean 4.34; the eleven required-reachable controls
     caught it in one run.

BRIEF ITEM (2) - THE CENSUS HYPOTHESIS IS NOW A ONE-PERIOD CLAIM. `Census29`/`Census31`
said "for EVERY index n"; the gates verify ONE PERIOD; nothing connected them. Now:

    Periodic.index_reduce      : every index's forward gap word is the forward gap word
                                 of an index whose opening lies in [1, P]
    Machine29.census29_of_period (h : Census29P) : Census29
    Machine31.census31_of_period (h : Census31P) : Census31
    LadderPeriod.D_29_31_period (h : Census29P) (n) : g31 n <= 43 + 31
    LadderPeriod.D_31_37_period (h : Census31P) (n) : g37 n <= 58 + 37

`index_reduce` needs NO walk and NO base case - only that the opening PREDICATE is
periodic (one `omega` per gear) plus surjectivity of the enumeration - which is why it
works at machines whose period a kernel will never enumerate. VERDICT 21 STANDS: the
census is still not kernel-checked. What changed is that the unverified part is now
FINITE as well as explicit: the 214,708,725 openings of one machine-29 period
(6,226,553,025 at m31), which is exactly the object `qual_dict.py` scans.

BRIEF ITEM (4) - LATERAL'S PARITY LAWS, THE ARITHMETIC HALVES, IN THE KERNEL
(`proofs/Mirror.lean`, footprint [propext, Quot.sound] - not even Classical.choice):
  * `mirror_gear` - M0 for ONE GEAR at ANY period: the mirror k -> P-k EXCHANGES the
    slot's two members, and blocking is symmetric in them. Instantiated at m11 (3 gears)
    and m29 (8 gears).
  * `antipode_open` - Lateral's `g_1* = 1`, and shorter than their residue argument:
    6*((P+1)/2) = 3P+3, so the antipodal slot's members are 3P+2 and 3P+4 and a gear
    would have to divide 2 or 4. `antipode_exposed29 : Exposed29 539141103` is an
    opening of machine 29 exhibited BY ARITHMETIC - no scan, no `decide`.
  * `self_mirror_unique` - at most ONE self-mirror window per depth (N odd), which is
    the half the live route consumes: "fewer than two" proves "none".
  NOT DONE, and it is the half the lever needs: "every count is EVEN except the
  exceptional one" needs a fixed-point-free-involution counting lemma. Named as a
  round-27 target, not claimed.

BRIEF ITEM (3), THE EIGHTH RUNG - NOT ATTEMPTED, and the reason is not a judgment. At my
close `agents-shared.md` carried round-26 blocks from LATERAL and HARVESTER only. Two
independent blockers anyway: (i) the vehicle needs MACHINE 37's qualifying family
D_2..D_7 at floor 14 (gear 41's teeth {7,34}), i.e. a full-period scan of 1.24e12 slots -
which the brief forbids; and (ii) Mechanic's m41 superset is the wrong object twice over:
it is a dictionary of the NEW machine (41), where the merge law consumes the OLD
machine's word, and it is 4-tuples where the family needs depths to K+2 = 7.
ALSO A CORRECTION OF MY OWN R25.7: "machine 37's period is about 15 min of CPU per pass"
is wrong by three orders of magnitude on this lane's own measured scaling (m31's 33.4e9
slots cost 1,451 s CPU; m37 is 37x larger).
FOR CONSTRUCTOR: the eighth rung's ONLY missing input is machine 37's qualifying windows
at depths 2..7, floor 14 - a CSP job for `crt_dict.py`/`scanfree_dict.py`, not a scan.
Emit them in `qual_dict.py`'s format and the rung is a transcription.

NEW STANDING INFRASTRUCTURE FACTS (both paid for in failed runs):
 * ONE DIVISIBILITY PER `omega` CALL. `unfold Exposed13; omega` over all eight gears at
   once ELABORATES and then fails IN THE KERNEL with "(deterministic) timeout" - the
   certificate for sixteen simultaneous divisibility constraints is too big to re-check.
   Split per gear it is instant. The elaborator is not the binding limit here; the kernel
   is.
 * `ConstantInfo.value?` does not expose theorem proofs (Lean 4.34) - match `.thmInfo`.

NEW VERDICTS 22-25 (in formalist.md R26.6): the value?/positive-controls lesson; the
kernel-side omega limit; the sentinel lesson for computed bounds; and "the census
hypotheses are now finite, and that is the whole of what round 26 could shrink".

DOCS: `docs/novel/survivor-generator.md` (round-26 addendum: the generator is SOUND, with
the gated independence), `docs/novel/depth-sum-identity.md` (the periodicity bridge is
built; only the Finset re-indexing remains), `docs/novel/qualifying-dictionary-rung.md`
(the one-period census), `docs/novel/mirror-parity-laws.md` (the kernel-checked halves).

## LP-duality thread (round 26)

BRIEF: STAR-k at budget widths m41-m53 - CERTIFICATES OR ONLY NECESSITY?  Plus: does the
frontier-is-a-width law open NARROWER (windowed) statements of route value?  Plus a cost
curve.  Headline gate for the manager:
`.venv/Scripts/python.exe research/star_case.py GATE` (eight items, ~50 s, prints ALL
ASSERTIONS GREEN); ground-truth cross-check `research/window_dict.py 19 20 31`.

### 0. THE ANSWER: CERTIFICATES - AND THE WHOLE (D) LADDER THROUGH 37->41

STAR-k WAS THE WRONG OBJECT.  Holding gear 5's phase at w needs no triple blocks at all:
CASE w IS THE COMPOSED VEHICLE ITSELF on the position set [0,W) minus what gear 5 blocks,
over the gears above 5 - and a certificate in EVERY case is a certificate of the rung.
That one observation turns round 25's closed four-rung ladder into a NINE-rung one.
Round 25 REFUTED 19->23 and 23->29 with exhibited exact witnesses and called 37->41 an
EXACT REFUTATION.  All three are certified this round, hypothesis-free, from the list of
primes alone:

  rung      W    held        cases   exact certificate ops   re-verified from disk
  19->23   48    (5)             5                  38,677   202/7   < 607/21
  23->29   63    (5,7)          35                 362,049   25      < 26
  29->31   74    (5,7)          35                 576,472   92/3    < 94/3
  31->37   95    (5,7,11)      385               8,388,426   32      < 34
  37->41  129    (5,7,11)      385              12,778,058   95/2    < 97/2

With round 24's four rungs (7->11, 11->13, 13->17, 17->19) that is EVERY (D) STEP THE
PROJECT HAS, certified by LP duality with no census hypothesis anywhere.  Round 25's
refutations are correct as written - they refute the LEVEL-2 MEMBER - but they do not
bound the family, and this round's headline is that the family has a ladder parameter
(the number of held gears) that the level-2 member does not.

Separately, the same "restrict the vehicle" move answers part (b): prescribing OPEN
positions turns the vehicle into a decision procedure for the ADJACENT-GAP-PAIR
DICTIONARY, and gives complete scan-free LP-duality proofs of the EXACT F_2 at machines
19 and 23.

New files: `research/star_case.py` (the restricted relaxation, both instances, the gate),
`research/window_dict.py` (the vehicle against a full-period scan).
New doc: `docs/novel/restricted-covering-certificates.md` (indexed).
Updated: `docs/novel/product-measure-frontier.md` section 5 (its own open question,
answered both ways).

### 1. THE CONSTRUCT (one object, two instances, and they compose)

`RelaxStar(gears, W, held, ws, openpts)` is round 25's composed vehicle with

  * pos = [0,W) minus what the HELD gears block at ws minus the required-OPEN positions,
  * dom(q) = the phases of q that block no required-open position,
  * n_ij taken over pos with the lower gears restricted to their domains,
  * the degree-2 cuts taken at the positions of pos only.

SOUNDNESS (proved, elementary): a real configuration with those held phases and those
open positions induces a 0/1 point of the polytope; restricting the lower gears' phases
only RAISES n_ij, and n_ij <= N_ij still holds at the actual tuple because that tuple is
in the domains.  GATED: with held = () and openpts = () the class is IDENTICAL to round
25's `RelaxCF` at m11/13/17/19 - columns, links, recursion row, rhs - so this is a strict
extension, not a reimplementation.

  (A) CASE SPLIT (openpts = (), held = the k smallest gears).  Strictly stronger than the
      STAR-3 LP: a family of case points always MIXES into a STAR-3 point (completability
      is convex; where a held gear blocks the position the moment vector is trivially
      completable), but a STAR-3 point does NOT condition into a family of case points,
      because its conditionals need not be pairwise consistent.
  (B) WINDOWED (held = (), openpts = {0, a, W} in ambient W+1).  A certificate says
      MACHINE M HAS NO ADJACENT GAP PAIR (a, W-a) - membership in the level-2 gap
      dictionary, decided by duality instead of by search or scan.
  (C) THEY COMPOSE, and the composition is strictly stronger than either (section 3).

### 2. (a) THE CASE-SPLIT VERDICT, AND THE LADDER PARAMETER

THE PRE-TEST, EXACT AND NOW CASE BY CASE.  The conditional uniform product measure
decides both sides of a case for free.  Its row value E_u[f_w] is exact and its MEAN over
the cases is exactly round 25's STAR-k number - asserted in the gate at m23/29/31, so the
case decomposition reproduces round 25's row identically.  At the budget widths
(min / max / mean):

     y    W   cases   min        max        mean (= round 25's STAR-k)
    41  129      5   +8.8304    +8.9661    +8.8853
    41  129     35  +13.6538   +14.7904   +14.2963
    41  129    385  +15.9739   +19.1761   +17.6640
    43  134      5   +6.6359    +6.7442    +6.6797
    43  134     35  +12.4104   +13.1870   +12.7830
    47  150      5   +5.0773    +5.1046    +5.0991
    47  150     35  +11.8010   +12.6057   +12.2560
    53  156      5   +3.0700    +3.1439    +3.1065
    53  156     35  +10.4790   +11.2061   +10.8054

NOT ONE case anywhere has E_u[f_w] <= 0.  An average could not have shown this.

AND A SECOND INGREDIENT ROUND 25 MISSED, WHICH IS WHY 37->41 REOPENED.  Round 25's exact
refutation of 37->41 rests on the uniform product measure's degree-2 moments being
COMPLETABLE at machine 41 (n = 11).  CONDITIONED ON GEAR 5 THEY ARE NOT: at n = 10 (drop
5) and n = 9 (drop 5 and 7) the conditional product moments carry an exactly-verified
violated degree-2 cut.  Holding one gear revives BOTH ingredients, so the round-25
witness is not a witness for the stronger species - and the cell then certified outright.

THE LADDER PARAMETER IS REAL, not decoration.  23->29 does NOT certify with one held gear
(the LP maximum of the recursion row stalls at 38.316 against the 38 it must beat, after
33 cut passes - 0.83% short); 31->37 does not certify with two (40.994 against 40, 2.5%);
37->41 does not certify with one (86.756 against 78, 11.2%) or two (57.281 against 55,
4.1%).  Each held gear roughly halves the residual and multiplies the case count by that
gear.  Above the ladder: 41->43 at k = 3 leaves a handful of hard cases (case (1,1,9)
stalls at 43.126 against 43) (see the negatives); 43->47 and 47->53 were SAMPLED at k = 4 -
m47 8/8 random cases certified, m53 6/8 (the two stalls at 47.08 and 46.30) - so 47->53
looks like a k = 5 problem.  Full k = 4 sweeps (5,005 cases) were not run this round.

AND THE VEHICLE IS NOW TIGHT ON F ITSELF.  With two held gears it certifies F(M) <= F(M)
- the exact value - at three machines: F(19) <= 25 (107,188 ops, 2 s), F(23) <= 34
(202,959 ops, 6 s), F(29) <= 43 (373,775 ops, 62 s).  m31 is not tight at k = 2 (19/35
cases at W = 58).  For comparison round 25's level-2 vehicle needed width 33 at m19 and
was REFUTED at width 48 at m23.

SOUNDNESS OF THE SPECIES, against ground truth the LP never sees: at a width BELOW F a
fully blocked window exists, so the case split must NOT certify.  m13 (F=11) at W=10,
m17 (F=18) at W=17, m19 (F=25) at W=24: never all cases.  m29 (F=43) at W=42 with 35
cases: 33 certify, 2 do not - sound, and sharp.

### 3. (b) THE WINDOWED STATEMENTS - AND THEY REACH THE LIVE CONJECTURE

The narrower statement the frontier-is-a-width law pointed at is exactly Constructor's
R64 covering form of the two-gap statement and the depth-2 member of Mechanic's Q*_J
family: "machine M has no configuration with 0, a and W open and everything between
blocked", i.e. "the adjacent gap pair (a, W-a) is not realised".

  * F_2(19) <= 31, LP-PROVED, SCAN-FREE, TIGHT.  Spans [32,66] x all splits: 1,680 cells,
    413 killed outright (gear 5 has NO phase leaving all three required-open positions
    open), 1,267 exact dual certificates, ZERO refuted, ZERO undecided, 2,760,053 ops.
    Every adjacent pair has both gaps <= F(19), and the case split now certifies
    F(19) <= 25, so F_2 <= 50 <= 66 and the sweep covers the whole obligation.  True
    value 31.
  * F_2(23) <= 39, LP-PROVED, SCAN-FREE, TIGHT.  Spans [40,68]: with gear 5 held, 1,537
    splits, 368 fully vacuous, 1,151 certified, 18 left (13 at span 40, 5 at span 42);
    those 18 all certify with gears 5 AND 7 held (1,009,730 ops, 28 s).  F(23) <= 34 is
    certified by the same vehicle, so F_2 <= 68 and the sweep is complete.  True value 39.
  * IT LOCATES THE MAXIMISER.  At span 31 = F_2(19) the vehicle fails on EXACTLY two
    splits, (10,21) and (21,10), each by an exact in-polytope witness; a full-period scan
    says the realised pairs of sum 31 are exactly those two.  (Gate item 6.)
  * AGAINST THE WHOLE DICTIONARY (`window_dict.py`, spans 20..31 vs the 221 realised
    adjacent pairs from a 1,616,615-slot period scan): 109 CERTIFIED (all genuinely
    unrealised), 94 correctly REFUTED, 72 DEAD (all unrealised), ZERO UNSOUND.  NOT
    EXACT, though: nine unrealised cells are not certified - (2,26),(26,2) at span 28 and
    (2,28),(4,26),(7,23),(15,15),(23,7),(26,4),(28,2) at span 30.  FOUR carry EXACT
    in-polytope witnesses - (2,26),(26,2) at row 107/4 >= 26 slack 3/4 and (4,26),(26,4)
    at row 28 >= 28 slack 0, all re-verified from disk - so they are genuine INTEGRALITY
    GAPS; the other five stall.  Holding gear 5 as well closes three of the nine
    ((2,28), (15,15), (28,2)).  Note (15,15)
    among them - the self-mirror split, exactly the palindromic case Lateral's mirror
    parity law singles out.
  * THE TWO RESTRICTIONS COMPOSE, STRICTLY.  At machine 23 span 40 the plain windowed
    vehicle cannot certify the split (2,38): after 61 passes and 91 s its LP maximum is
    39.7689 against 38 (4.7%), and (3,37) sits at 40.0943 (5.5%).  With gear 5 held,
    THREE OF THE FIVE CASES ARE VACUOUS - that phase of gear 5 blocks a required-open
    position, so the configuration is impossible outright - and the other two CERTIFY IN
    ONE SECOND.  Prescribing open positions does not just shrink the obligation, it
    DELETES BRANCHES OF THE CASE SPLIT for free.
  * THE SAME CONSTRUCT SEES SPECTRUM HOLES.  With openpts = {0,W} a certificate says "no
    gap of size exactly W".  Machine 19's exact gap-value set is {1..18,20,21,22,23,25};
    the vehicle certifies 24, 27 and 29 - all genuine holes, two BELOW F(19) = 25 - and
    refuses at 22, 23, 25, which are attained.

### 4. (c) THE COST CURVE

Closed-form op counts for the case split (LP columns, and the max-cover cells the
recursion row costs), exact:

     y   k   cases    n   cols/case   cols total   cells/case   cells total
    41   0       1   11      24,180       24,180       23,947        23,947
    41   1       5   10      23,035      115,175       22,807       114,035
    41   2      35    9      21,481      751,835       21,260       744,100
    41   3     385    8      19,160    7,376,600       18,950     7,295,750
    53   0       1   14      64,433       64,433       64,057        64,057
    53   1       5   13      62,573      312,865       62,202       311,010
    53   2      35   12      60,018    2,100,630       59,654     2,087,890
    53   3     385   11      56,124   21,607,740       55,771    21,471,835

THE LAW: holding gear q multiplies the TOTAL work by very nearly q and shrinks each case
by only ~5%, because the column count is dominated by the LARGE gears, which stay.
Measured multipliers at m41: x4.76 (k 0->1), x6.53 (1->2), x9.81 (2->3).  Certificate
cost per rung: 38,677 ops (5 cases) at 19->23; 362,049 (35) at 23->29; 576,472 (35) at
29->31; 8,388,426 (385) at 31->37; 12,778,058 (385) at 37->41 - i.e. ~33,000 exact
operations per case, nearly flat in the machine, with the case count carrying all the
growth.  Windowed certificates cost ~2,200 ops each at m19.

So the species buys reach at a cost that is a PRODUCT OF PRIMES: k = 4 is 5,005 cases and
affordable, k = 5 is 85,085 and marginal, k = 6 is 1,616,615 and not.  THAT IS THE REAL
LIMIT OF THIS VEHICLE, and it is a different limit from any the project has recorded: not
a degree ceiling and not a width frontier, but a primorial in the number of held gears.

### 5. INFRASTRUCTURE (three fixes, all gated, all reusable)

  * `zeta_fast` - the subset-sum transform in n 2^(n-1) exact additions where
    `zeta_values` loops over supersets (~150,000 steps at n = 10).  Asserted equal on
    random instances at n = 3,5,7,9.
  * `completable_fast` - completability decided by VERIFYING a float-discovered
    completion exactly (nu >= 0 and A nu = b on a small support, both asserted in exact
    rationals) instead of running the exact rational simplex on the (subsets x atoms)
    tableau.  MEASURED: n = 8, 1.86 s -> 0.02 s; n = 9, 46.45 s -> 0.27 s; n = 11, a call
    that did not finish in TEN MINUTES -> 16.7 s.  Verdicts agree with the exact oracle at
    n = 7, 8, 9 (gate item 4).  This is what made the whole round affordable.
  * THE LP REFORMULATION.  Round 25's loop maximised a COMMON additive slack t over all
    rows, which conflates the coverage rows (rhs ~1) with the recursion row (rhs |pos|).
    MEASURED at m41: t sat at exactly +0.221818 for six consecutive passes while 78 inert
    cut rows per pass accumulated and the float LP went 2.2 s -> 80.7 s.  The loop now
    maximises the recursion row itself - the quantity the certificate is about - with the
    cuts as hard rows.  The round-24 rung certificates still reproduce (m11 562 ops
    identical; m13/m17 differ only in the path-dependent dual point, as round 25 noted).

### 6. A CORRECTION TO MY OWN ROUND-25 PROCESS RULE

Round 25 strengthened "save your witness" to "the witness must be EXACTLY IN THE
POLYTOPE", and recorded that a rationalised LP point is not - hence two special
constructions.  That reading is too strong at level 2: `rationalise_star` normalises every
size-2 block to sum to exactly 1 at a single denominator, and `repair_links` then defines
each single block as an exact marginal of a pair block, so the repaired point is exactly
consistent BY CONSTRUCTION.  MEASURED: at machine 19 span 28 split (2,26) the loop stalls
with the repaired point satisfying every block sum, every link, the recursion row
(26.75 >= 26) and completability at all 26 positions - a perfectly good witness that round
25's candidate list discarded, turning a REFUTED cell into an UNDECIDED one.  Nine such
cells were mis-labelled before the fix.  THE RULE THAT STANDS is the one that always
mattered: THE WITNESS MUST BE VERIFIED EXACTLY IN THE POLYTOPE, whatever produced it -
"consistent by construction" is a good way to FIND one, not a requirement on one.

Second process item, also mine: round 25's deciding loop ABORTED on an assert when the
float LP said infeasible and no rounding of its dual closed the inequality exactly.  That
killed a whole sweep on one awkward dual.  It is now a recorded verdict (`NODUAL`) - a
failure to produce a certificate is an undecided cell, never a certificate.

### 7. NEGATIVES AND WHAT IS NOT DONE

- 41->43 AT k = 3 WAS NOT COMPLETED, and it is my one badly-sized job of the round.  I
  DELIBERATELY STOPPED it at 163 of 385 cases: 157 certified, six stalled at a 45 s/case
  budget - (1,1,1), (1,1,2), (1,1,9), (1,6,1), (1,6,8), (1,6,9), all with the held gear-5
  phase 1.  Re-running three of those six at 150-200 s, TWO CERTIFIED (73 s and 204 s)
  and the third stalls 0.17% short (43.0751 against 43 after six passes).  So 41->43 at
  k = 3 is BUDGET-LIMITED rather than blocked, and the honest statement is a PARTIAL
  SWEEP, not a rung.  The lesson is the standing one and it is mine: launch the
  multi-hour job early or narrow it - I launched this one late and had to stop it.
  43->47 and 47->53 were only SAMPLED at
  k = 4 (8/8 and 6/8 random cases).  Full sweeps at k = 4 (5,005 cases) were not run.
  The two m53 stalls sit at 47.08 and 46.30 against ~46, so 47->53 is plausibly a k = 5
  problem - and k = 5 is 85,085 cases, which is where this vehicle's cost curve bites.
- The windowed vehicle is NOT exact: nine unrealised cells at machine 19 spans 28 and 30
  are not certified by the plain form, four with exact feasible witnesses (genuine
  integrality gaps); holding gear 5 closes three of the nine.  Whether the residual gap
  has a structural description is not resolved.
- The F_2 ladder was run in full only at machines 19 and 23.
- Depth >= 3 windowed statements (the rest of Mechanic's Q*_J family) are a direct
  generalisation of the same construct and were NOT run: the split count is W^(J-1).
- EVERY CERTIFICATE HERE IS SCRIPT-VERIFIED, NOT KERNEL-CHECKED.  The arithmetic is exact
  and each certificate is re-verified against a rebuild of the relaxation from scratch,
  but that is this thread's own code checking its own object.  See section 9.
- The prior-art check for the case-split certificate species and for the integrality of
  the punctured-window relaxation is NOT RUN (no web access).

### 8. PRE-REGISTERED PREDICTIONS FOR ROUND 27 (score them next round)

E1  41->43 completes at k = 4 (5,005 cases) and 43->47 too; 47->53 needs k = 5.
E2  THE CASE-SPLIT LADDER IS MONOTONE IN k: no rung certified at k fails at k+1.  This is
    NOT automatic - the case problems are different LPs, not refinements of one - so it is
    a real prediction.
E3  THE VEHICLE IS TIGHT ON F AT EVERY MACHINE ONCE k IS LARGE ENOUGH: F(31) <= 58 (which
    fails at k = 2, 19/35) certifies at k = 3.
E4  THE INTEGRALITY-GAP CELLS OF THE WINDOWED VEHICLE CLUSTER JUST BELOW F_2.  At machine
    19 all nine sit at spans 28 and 30 against F_2 = 31, and at machine 23 the leftovers
    sat at spans 40 and 42 against F_2 = 39.  I predict the same at machine 29: the
    leftovers will sit at spans 56-58 against F_2(29) = 55, and none below 53.

### 9. FOR OTHER LANES

- FORMALIST, and this is the one that matters: THE CASE-SPLIT CERTIFICATES ARE FINITE
  EXACT RATIONAL OBJECTS AND THE CHECKING PREDICATE IS ARITHMETIC.  Per case: a
  nonnegative rational weight per cut row, one weight on the recursion row, a signed
  weight per consistency link, and the single inequality
  `sum_S max_{j in block S} a_j < sum_r y_r (1 - lam^r_0) + yff * |pos|`.  Validity of a
  cut row is "its subset-sums are >= 1 at every nonempty atom", and n <= 11 in every rung
  here, so <= 2,047 atoms per row.  Exhaustiveness of the case split is "the held gears'
  phases range over all residues".  A (D) rung with NO CENSUS HYPOTHESIS AT ALL looks
  reachable this way, and the rungs from 29->31 on - which are currently
  hypothesis-explicit - are the candidates.  The files are
  `research/data/r26/cert_rung*.pkl`; ask and I will emit them in whatever shape the
  Lean side wants.
- CONSTRUCTOR: the windowed vehicle answers your level-2 dictionary queries with a DUAL
  CERTIFICATE for the negative direction.  At machine 19 it agrees with the period scan
  on every one of 275 decided cells and contradicts nothing; cost ~2,200 exact ops per
  answer.  It is not complete (nine gap cells), so it is a cheap FILTER in front of the
  CSP oracle rather than a replacement.
- MECHANIC: the same construct is Q*_J at depth J with openpts = the prefix sums, and
  your word-legality condition just deletes cells from the sweep.  It gives F_2 exactly
  and scan-free at m19 and m23; if it scales it is the way to test Q*_max = F(M+q') at
  the steps your period scans cannot reach.
- LATERAL: the self-mirror split (15,15) is among the windowed vehicle's integrality-gap
  cells at machine 19 - the palindromic case your parity law singles out.  One machine,
  so possibly a coincidence; worth one look.

## Formalist round 26 - addendum (cross-lane pickup verdict, filed after my round closed)

Routed in by the manager: the LP thread's section 9, addressed to this lane - the
case-split certificates are finite exact rationals with an arithmetic checking predicate,
and a (D) rung with NO census hypothesis looks reachable in Lean.

I MEASURED THE ARTEFACTS BEFORE ANSWERING, because what was quoted for sizing are OP
COUNTS and those are not certificate sizes. `research/data/r26/cert_gate_m23_w48_h*.pkl`
(the 19->23 rung, five cases, gear 5 held):

    per case: 29 cut rows x 22 rational entries + 29 row weights + 450 link weights
              + 1 recursion weight = ~1,120 rationals, 5.4 KB
              DENOMINATORS <= 5 BITS (21, 7, 3); numerators <= 10 bits
              verdict record lhs 202/7 < rhs 607/21 - your published table row exactly
    rung:     5 cases, ~5,600 small rationals, ~27 KB

**THE DATA IS NOT THE OBSTACLE** - ~11,000 numerals is `Machine29D4` scale (6,688
4-tuples, 55 s, empty axiom footprint), and single fractions are cheaper per element than
tuples, so round 25's `count x arity` isDefEq limit is nowhere near binding. Your
"CoveringCert scale" instinct was right, and for a better reason than the op ratio: the
certificate is small even though the CHECK is 26x bigger.

**THE OBSTACLE IS SOUNDNESS, NOT ARITHMETIC.** The theorem is `certificate -> rung`.
This lane already has that scaffold for the UNRESTRICTED level-2 consistent vehicle
(`CoveringCert.lean`, `CoveringCert2.lean`). `RelaxStar` needs four more lemmas: `pos`
restricted by the held gears; `dom(q)` restricted (restricting lower gears only RAISES
n_ij while n_ij <= N_ij still holds at the actual tuple); cuts taken at `pos` only, with
validity "subset-sums >= 1 at every nonempty atom"; and CASE EXHAUSTIVENESS over the held
gears' phases - the last has no analogue in the existing files and is exactly what lets
the species escape round 25's refutations.

VERDICT: **ROUND-27 TARGET, TOP OF MY LIST - not a this-round pickup.** Not for lack of
appetite: it retires `Census29P`/`Census31P` outright, which is the residue my verdicts 21
and 25 name, and nothing else on my list does that. But it is four soundness lemmas plus
five transcribed modules, my round is filed and green at 1426 jobs, and the
job-completion rule says launch work of that size EARLY or narrow it. Starting it now
would do neither. Order of attack is fixed in formalist.md R26.8: one case first, then
19->23 (5 cases), then 29->31 (35 cases) - the first rung that REPLACES a
census-hypothesis rung.

ASKING FOR THE EMISSION YOU OFFERED, and this is the cheap thing that unblocks round 27 -
JSON, one file per case, integers only (no `Fraction` reprs): `rows` as
`[pos, [[num,den] x 22]]`; `y`, `nu`, `yff` as integer pairs; **the ATOM INDEXING made
explicit** (which gear subset each of the 22 entries is, as bitmasks) so the Lean side can
state cut validity without reverse-engineering your column order; `held`, `ws`, `W`,
`full`, and the claimed `lhs`/`rhs` as integer pairs; and for exhaustiveness the list of
held-phase tuples the files are indexed by, plus the assertion that it is all of
`prod (residues of held gears)`.

ONE THING TO BE AWARE OF ON YOUR SIDE: your section 7 says every certificate here is
script-verified by this thread's own code checking its own object. That is exactly the
class of claim my round-26 `DepAudit.lean` lesson applies to - **an audit needs positive
controls or it is not an audit**. Your soundness cross-check at widths BELOW F (m13 W=10,
m17 W=17, m19 W=24, and the sharp m29 W=42 case where 33 of 35 certify) IS that control,
and it is the strongest thing in the block for my purposes: it is what makes me willing to
spend a round formalising the species rather than checking it.

## Constructor round 26

GATE (headline, re-runnable from a clean process in ~4 minutes):
  .venv/Scripts/python.exe research/qstar.py     -> all assertions passed
                                        (log research/data/r26_qstar.log)
Supporting gates, all green this round:
  research/query_law.py    -> reproduces round 25's 181/90/955/3399 exactly
  research/query_cap.py    -> every measured query count under its proved cap
  research/f2_41.py        -> F_2(41) = 103 EXACT, scan-free
  research/chain_dict_oracle.py --verify 37 -> 12,587/12,587 CRT-confirmed

HEADLINE 1 - MECHANIC'S Q* CONJECTURE IS A THEOREM, AND THE PROJECT ALREADY
HAD IT; VERIFIED EXACTLY AT EIGHT STEPS. Q*_J(M; legal for q') is
definitionally identical to my R46 qualmax_J = LAYER J-2 OF THE KLEENE STAR:
Mechanic's condition (i) is K's edge condition d mod q' in {0,a,b}, their
condition (ii) is K's T3 tooth transition, and the two free flanks are L and
R. R46's identity F(M+q') = L (x) K* (x) R - PROVED BOTH WAYS in round 22 - is
therefore exactly the conjecture, and the direction Mechanic left open is
R46's (>=) half, the CRT choice of the killing copy. Standalone form:

  ATTAINMENT THEOREM. If consecutive openings x_0 < ... < x_J of M have a
  legal middle-gap word then x_J - x_0 <= F(M + q').
  PROOF. Legality = existence of a tooth assignment; fix it, read off the
  residue r mod q' that puts every interior on a tooth; the joint period is
  P(M) q' with gcd(P(M), q') = 1, so SOME translate of the window sits at that
  residue and has all its interiors killed; the containing gap of M + q' is
  then at least the span. [] (J = 2 is Mechanic's own deletion ladder
  F_2(M) <= F(M + one gear); this is its extension to every depth.)

COMPUTED AT EIGHT STEPS, exactly, with NO depth cap and NO span cap - the
"every J" quantifier is closed by the A_4 closure (which bounds max_J Q*_J
over all J at once and terminates):

    M    q'   F(M)  F(M+q')  Q*_2 Q*_3 Q*_4 Q*_5  max_J Q*_J  J*    verdict
    11   13     7      11      11    8    -    -         11    2     EXACT
    13   17    11      18      16   18    -    -         18    3     EXACT
    17   19    18      25      25   25    -    -         25    2,3   EXACT
    19   23    25      34      31   33   34    -         34    4     EXACT
    23   29    34      43      39   43    -    -         43    3     EXACT
    29   31    43      58      55   58   55   55         58    3     EXACT
    31   37    58      88      68   85   88   68         88    4     EXACT
    37   41    88      91      90   90   91    -         91    4     EXACT

J* = k_win + 1 at every step. The eighth row is new (Mechanic's m37 census
makes it decidable). Attaining windows are the project's own extremal objects
re-derived - (4,8,15,7), (7,10,21,10,7), (11,12,37,28), (2,88) - and F(41)=91
is attained at (21,14,41,15), a new datum.

FOR MECHANIC: your two SEEDED anchors now have their exact values by theorem -
max_J Q*_J(43; legal for 47) = F(47) = 118 and max_J Q*_J(47; legal for 53) =
F(53) = 145, so 43->47 and 47->53 certify with margins 32 and 26, not +1, and
the span cap is irrelevant. That is a falsifiable prediction for your j5_multi
at those steps.
FOR EVERYONE, THE NEGATIVE HALF: because Q*_max EQUALS F(M+q'), "the word-legal
criterion certifies (D)" is NOT weaker than "(D) holds" - it is the same
statement in another representation. The criterion's margins are (D)'s TRUE
margins, not slack to exploit. Q*'s value is that it never builds M + q'.
FOR FORMALIST: this is the exact complement of your kernel soundness result at
11->13. Soundness says the Q*-style maximum BOUNDS the record; attainment says
it EQUALS it. Together, at 11->13, that is the record law at kernel grade.
A CORRECTION TO THE RECORD: R39's "criterion equals F(M+q') at 6 of 7, slack 2
at 23->29" is wrong - the exact value is 43 = F(29) at J = 3, window
(10,10,23). Equality holds 7 of 7, as the theorem requires.
Docs: docs/novel/kleene-generator.md s4c; docs/novel/old-machine-spectrum.md s9.

HEADLINE 2 - THE EIGHTH RUNG 37->41 IS CERTIFIED, SCAN-FREE, WITH NO GIVEN
INTEGER. R62b's diagnosis was right (oracle cost, not state space) and the fix
is to split what the oracle does. PHASE 1: run the loop with Mechanic's EXACT
m37 4-tuple census as oracle (its induced level-2 projection is the exact
realised-pair set; its induced F = 88, F_2 = 90 asserted vs corpus first) -
CERTIFIED, bound 129 <= budget 129, 12,695 queries, 63 s. PHASE 2: re-prove all
12,587 deletions with the scan-free CRT decider, in parallel now that the list
is known in advance - 12,587 OF 12,587 CONFIRMED, ZERO contradictions, ZERO
undecided, 10,859 s CPU / 2,174 s wall on 5 workers. So the census only CHOSE
which refutations to attempt; every one is established from the gear list.
Cross-check: A_4 over the same census gives F(41) = 91 exactly, so the ninth
rung's input is an output of the eighth.

HEADLINE 3 - THE QUERY COUNT (R67 item ii): MY OWN NEGATIVE WAS TOO STRONG,
AND THE QUESTION WAS AIMED AT THE WRONG OBJECT.
(a) A PROVED CAP. The loop asks only about value 4-tuples carried by live MF_4
edges and (flank,base) pairs of live MF_4 states, and memoises, so for ANY
strategy queries <= T_4(F,q') + T_2(F,q') <= F^4 + F^2, where T_4, T_2 are
machine-free counts of the system at (F, q'). Measured usage: 0.00 / 1.40 /
0.69 / 1.19 / 0.33 / 1.36 / 1.45 / 0.25 PERCENT of that cap at the eight steps
- a 5.8x band, tighter than any other correlate tried (q/F^2 spans 20x). So
R67's "bounded by nothing proven" is superseded.
(b) THE COUNT IS A PROPERTY OF THE STRATEGY. The SAME rung 37->41 costs 2,879
queries (topk=1), 12,695 (topk=256), or 5,771 (topk=256 with F_2(37) given).
It is however ORACLE-INDEPENDENT, and that was checked not assumed: run with
Mechanic's censuses instead of the CRT decider the loop reproduces round 25's
181 / 90 / 955 / 3,399 DIGIT FOR DIGIT. Three new small steps: 11->13 needs
ZERO queries (machine-free MF_4 already certifies), 13->17 six, 17->19
eighteen. Ladder: 0, 6, 18, 181, 90, 955, 3399, 2879 - non-monotone TWICE.
(c) THE STRATEGY-FREE OBJECT IS THE CERTIFICATE, and it can be minimised
(restore a deletion, re-close, drop it if the bound still clears - exact,
because restoring can only raise a max-plus closure):

    step        queries  greedy deletions  IRREDUNDANT
    11 -> 13          0         0               0
    13 -> 17          6         6               2
    17 -> 19         18        18              11
    19 -> 23        181       163              76
    23 -> 29         90        90              58
    29 -> 31        955       905             712
    31 -> 37      3,399     3,235           2,189
    37 -> 41      2,879     2,769           2,077

THE NON-MONOTONICITY SURVIVES MINIMISATION (23->29 costs 58 against 19->23's
76; 37->41 costs 2,077 against 31->37's 2,189), so it is a fact about the
STEPS, not the search - the certificate cost tracks the added gear's
arithmetic, like A_kill, A_relax and litcap. FOR FORMALIST: the eighth rung is
2,077 finite CRT refutations plus one closure.

F_2(41) = 103 EXACT, SCAN-FREE AND NON-CIRCULAR - NOT a first computation, and
I nearly filed it as one. Mechanic's lap-phase transfer already pinned it (see
their docs/novel/README.md entry: "PINS F_2(41) = 103 with no descent (cap
F(43) = 103 free, witness at 103)"). The difference is that their pin uses the
DELETION-LADDER cap F_2(41) <= F(43) - legitimate as a computation, circular as
an induction step (X36) - while this sweep never mentions F(43): it refutes
every candidate pair ABOVE 103 outright. Sweeping Mechanic's m41 transfer
dictionary downwards by pair SUM (only 36 superset pairs exceed 103) and
deciding each by CRT: refutations at sums 115, 113, 110, 108, 107, 106, 105,
104 (36 pairs, 0 undecided), first realised pair at 103, witness (75,28). Two
vehicles, same integer.
NOTE: F(M+q') - F_2(M) is now 3 at 29->31, 1 at 37->41 and 0 at 41->43 - the
deletion ladder's slack is exhausted exactly where the budget slack grows
(X36 sharpened).

A SUPERSET DICTIONARY IS A SOUND ORACLE - worth stating, because it is not
obvious: the loop only ever ACTS on a NO, and absence from a superset of the
realised set proves absence from the realised set. So Mechanic's m41 transfer
dictionary licenses genuine deletions. Measured how inflated it is: of the 249
arity-4 tuples it calls realised on the 41->43 walk, the CRT decider refutes
12 of 12 sampled (mean 3.6 s).

CROSS-LANE FOR FORMALIST - the m37 QUALIFYING SPECTRUM AT FLOOR 14, no scan
(research/qual_spectrum.py, log research/data/r26_qualspec_37.log):
    Q_2 = 88  (2,88)      Q_3 = 90  (2,62,28)        Q_4 = 97 (2,37,23,37)
    Q_5 = 103 (4,26,14,35,28)   Q_6 = 110 (2,30,32,15,20,13)
all EXACT, against the budget F(37) + 41 = 129. Method: an A_4 closure with the
SIZE FLOOR as the edge predicate gives sound upper bounds at every depth
(90, 97, 103, 127, 146, 174, ...), then branch-and-bound over A_4 walks with
those layer potentials as an admissible heuristic gives the exact values, each
window decided by the realisability oracle. THE IMPORTANT CAVEAT FOR YOUR LANE:
unlike the word-legal family, THE SIZE-FLOOR FAMILY DOES NOT TERMINATE IN THE
ABSTRACTION - the floor-14 layer bounds keep growing to 380 at layer 16 - so no
free depth cap, and each depth needs its own exact computation. That is the
concrete price of the size floor over the kill-word predicate, and it is the
same distinction Headline 1 draws. Q_7 is NOT delivered - see the negatives.

ON THE MANAGER'S INCREMENT-LAW CANDIDATE (F(M+q') - F_2(M) <= min(2u',q'-2u')):
reproduced from my Q* witness table plus F_2(41) = 103 - differences
0, 2, 0, 3, 4, 3, 20, 1, 0 against caps 4, 6, 6, 8, 10, 10, 12, 14, 14 at
11->13 .. 41->43. Holds 8 of 9, fails by +8 at the padded 31->37. READING,
LABELLED HYPOTHESIS: 2u' is the smallest positive LEGAL letter, so the
candidate says "one more link buys at most one small letter over the old
two-gap maximum, unless the link is padded (worth a full q')".

NEGATIVES AND SELF-CORRECTIONS:
- MY OWN R61 TABLE IS WRONG AT MACHINE 31: scan-free D_2 = 1,254 and
  D_3 = 15,020 should read 1,253 and 15,019. That run predates round 25's own
  decide_cover fix and counted the phantom (1,1) and (1,1,1). Both are now
  refuted by the fixed decider AND absent from Mechanic's independent census.
  MECHANIC'S m31 CENSUS IS CLEAN - the defect was mine. F_1,F_2,F_3 unchanged.
- R67's "the query count is bounded by nothing proven" was too strong (see
  Headline 3a), and I asked for a law about a quantity that is a property of my
  search strategy, not of the problem.
- THE NINTH RUNG 41->43 IS NOT CERTIFIED. Attempts, all recorded:
  (i) the superset oracle ALONE stalls at bound 222 vs budget 134 after 2,452
  queries in 38 s - the transfer dictionary is too inflated at arity 4;
  (ii) the exact hybrid (superset then CRT) was mis-sized at topk = 16
  (pool.map blocks on the slowest member of a batch and m41 PAIR refutations
  cost 30-140 s) and was cancelled;
  (iii) the properly-sized run (topk = 256, F_2(41) = 103, 6 workers) was
  KILLED BY A MEMORY EVENT at 12:35 together with the qual_spectrum job (three
  Application fault records; commit 41.5 of 65 GB, 3.6 GB physical free, other
  lanes running). Cause found and FIXED AT SOURCE - the superset was a 4.2M
  entry Python int set (~500 MB); it is now a sorted numpy int64 array queried
  by searchsorted (34 MB);
  (iv) the lean rerun (topk = 256, F_2 = 103, 3 workers) ran clean for 95
  minutes and was CANCELLED AT ROUND CLOSE - 5,575 s of worker CPU (~1,550 m41
  refutations at the measured 3.6 s mean) against only 32 s in the main
  process, so the rung is entirely ORACLE-bound, exactly as R62b found one
  gear earlier. Closing record appended to research/data/r26_chain43_hyb3.log.
  Recorded as a CANCELLATION, not a result: nothing is unsound and nothing is
  stuck, it is a multi-hour job mis-sized for the round.
  Q_7(37;14) was lost in the same memory event and its recomputation (the
  longest realised run of m37 gaps >= 14) was still running at round close and
  was stopped; Q_2..Q_6 above are unaffected (already printed, exact), and
  Q_7 <= 174 by the layer bound.
- PROCESS: I piped two long background runs through `tail`, which buffers, so
  they showed no progress for 25 and 10 minutes and one had to be killed and
  relaunched. Cost real round time; my own rule now is that a background job
  writes to a file with -u, never through a pipe.

NEEDS: (1) FROM MECHANIC - an EXACT m41 4-tuple census (their transfer
dictionary is already exact at depth 1; the inflation is at arity >= 2 and I
measured it at 12/12). That is the cheap route to rung nine and it is their
vehicle. (2) The UNIFORM ORDER (R67 item i) is untouched and is the sharpest
single open question the chain has. (3) The first-moment transfer stays with
the manager - and Headline 1 sharpens what it must deliver: there is NO slack
anywhere in the criterion family to trade against, so the transfer has to
prove the record itself.

## Mechanic round 26

GATES (all re-run from clean processes at round close, all GREEN):
  .venv/Scripts/python.exe research/akill_verify_r26.py         -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/ghist_gate_r26.py <7 csv>   -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/screen_tuples_r26.py (x4)   -> 0 false kills / 468,418
  .venv/Scripts/python.exe research/akill_verify_r25.py --nosat -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/cyclic_close_r25.py check   -> ALL ASSERTIONS PASSED
Every job this round launched has finished or been deliberately stopped and
reported as such; nothing is left running.

THE HEADLINE: CONSTRUCTOR'S NEW THEOREM SURVIVED ITS HARDEST TEST, AT BOTH
DEEP ANCHORS, WITH WITNESSES. Their round-26 filing turned my Q* conjecture
into a theorem (Q*_J is the Kleene star's layer J-2; R46 plus a standalone
attainment proof) and handed this lane a falsifiable prediction: my two
seeded anchors must come back Q*_max = F(M+q') EXACTLY, 118 and 145, margins
32 and 26 - NOT the "<= 149 / <= 170" my round-25 seeded runs could see. Both
runs were already in flight and both landed on the predicted value:

    step     seed  cap   PREDICTED   MEASURED        round 25 said   budget
    43->47   117   200   118=F(47)   118  MATCH      <= 149          150
    47->53   144   171   145=F(53)   145  MATCH      <= 170          171

Ten range workers and fourteen range workers respectively; the 47->53 cap 171
composes with round 25's cap-200 seed-170 run so (144, 200] is covered with no
gap. FOUR WITNESSES, each re-verified AT THE TARGET MACHINE from the
definition (openings where claimed, EVERY other slot of the span blocked slot
by slot, middle gaps a legal kill word), and they arrived in MIRROR PAIRS from
workers that knew nothing of each other:
    m43 k = 18,497,829,635,337     gaps [85,31,2]      middle [31] = -s mod 47
    m43 k = 2,161,962,392,309,550  gaps [2,31,85]
    m47 k = 82,799,441,296,736,535 gaps [70,35,18,22]  middles [35,18]
    m47 k = 19,682,189,134,678,555 gaps [22,18,35,70]
THE 47->53 MAXIMISER'S MIDDLE WORD IS (35,18) = (q'-s, s) - THE ALTERNATION,
the same object that controls fuel arity below. The attainment theorem and the
arity obstruction meet on one window.
CONTROL: the ten 43->47 workers' window counts sum to 178,542,615 - digit for
digit the round-25 serial total at a completely different seed.
CONSEQUENCE: the attaining depth equalled k_win + 1 at both steps where k_win
was measured independently, so this PRE-REGISTERS k_win(43->47) = 2 and
k_win(47->53) = 3 - one kwin_census run each, and C13's census has never gone
past 37->41.

THE ROUND-25 ALTERNATION PREDICTOR: PRE-REGISTERED, TESTED, REFUTED - AND
REPLACED BY A THEOREM THAT NEEDS NO SOLVER (docs/novel/phase-saturation-arity.md).
Pre-registration written before any SAT call: research/data/r26/prereg_akill_53_59.md.
At 53->59 (s = 20, q'-s = 39):
  P1 CONFIRMED - the pair (20,39) IS realised at machine 53, witnesses
     5,408,553,654,414,421,963 and 1,522,353,991,400,668,678, both re-derived
     from the definition (occurrence, killability at r = 49/10 mod 59, joint
     realisability by CRT).
  P2 REFUTED - (20,39,20), (39,20,39), (20,39,20,39), (39,20,39,20) and
     (20,39,20,39,20) are ALL ZERO, every one with ZERO SAT CALLS.
  => THE 5-CHAIN SHAPE DOES NOT RECUR. Pair realisability is NECESSARY
  (overlap lemma) and NOT SUFFICIENT: round 25's inference was wrong although
  every number under it was right.
THE REPLACEMENT. Gear q blocks {a, a + s_q} mod q for a free phase a,
s_q = -2*6^{-1} mod q, so a pattern with exposed offsets X occurs only if
    FREE_q(X) = Z_q \ ( (X mod q) u ((X - s_q) mod q) )   is NON-EMPTY;
empty for some q means ZERO BY THEOREM. |FREE_q| >= q - 2|X|, so only gears
q < 2|X| can fire - all the content is at gears 5, 7, 11. Applied to the
alternation it is a CLOSED-FORM arity ceiling per step:

    step      31->37 37->41 41->43 43->47 47->53 53->59 59->61 61->67
    ceiling      6      2      2      2      5      3      3      4
    dead gear    5      5      5      5      5      7      5      7
    A_kill       4      3      3      3      5      -      -      -

A_kill(47->53) = 5 sat EXACTLY at its ceiling; at 53->59 the ceiling falls back
to 3 and for the first time the binding gear is 7. 53 WAS SPECIAL, and now for
a reason. GATED: sound against the project's ENTIRE realised-word record (37
words at five steps, none wrongly zeroed); reproduces the three structural
zeros already on record ((18,35,18,35,18) at 47->53, found by hand in round 25,
and (16,31),(31,16) at 43->47, which cost SAT); reverse-invariant; and it
closes two levels outright with no solver (N_6 = 0 at 41->43, N_7 = 0 at 43->47).

FOR CONSTRUCTOR, IMMEDIATELY USABLE ON THE RUNG-NINE BLOCKER. The same
obstruction applies to GAP TUPLES, and for a 4-tuple only gears 5 and 7 can
fire - two lookups per row. Soundness gate first: 468,418 exact 4-tuples at
machines 23/29/31/37, ZERO removed. Applied to your m41 arity-4 superset:

    4,239,676 tuples -> 2,814,574 (66.39%); 1,425,102 REMOVED BY THEOREM
    (gear 5 kills 780,486, gear 7 kills 644,616), in seconds
    induced 3-tuple dictionary 130,942 -> 111,899

research/data/r26/gap_tuples_41_4_screened.csv. A screened superset is still a
superset, so it drops into the same certificate slot with no soundness argument
to redo - it is simply tighter, and it attacks exactly the inflation that made
12/12 of your sampled superset-YES tuples CRT-refute.

FULL-PERIOD GAP HISTOGRAMS, CYCLICALLY CLOSED BY CONSTRUCTION (Lateral's
U6/U9; handover research/data/r26/handover-lateral-U6-U9.md). A CORRECTION
FIRST: the brief's premise "your tiling runs cover the period, only the close
was missing" is wrong - the m37 tiling runs produced DISTINCT-TUPLE SETS, not
counts, and there were never any m41 tiling runs. So research/ghist_transfer.py
was built: K2's lap-phase bijection used for COUNTING, where the new machine's
laps are the phase-filtered copies of the old machine's opening set and the T
lap-boundary gaps taken in lap order ARE the cyclic close - by build, not by
patch. Both period identities asserted at merge.
  m37 EXACT, FULL PERIOD: 217,929,355,875 gaps over 1,236,789,689,135 slots in
  4,764 core-seconds against the round-20 direct sieve's 11,829 s, and it
  reproduces that scan on everything the scan recorded (F = 88, the complete
  13-value hole list, all four padding supplies 61,460/144,162/48,722/10,390) -
  plus the array round 20 discarded, plus hist[59] = 28 and hist[61] = 108,
  non-monotone in q' as ever.
  m13..m31 gated CELL FOR CELL against the round-25 corrected census; m31 from
  TWO different base machines.
  THE GEAR-5 TRANSFORM LADDER IS NOW EXACT AND CYCLIC THROUGH m37:
  arg H_5(1) = 129.776, 127.808, 126.334, 126.352, 126.059, 125.768, 125.659 at
  m13..m37. C17.5's "+126 deg at all seven machines, machine-independent" is on
  exact data a monotone DOWNWARD ladder that has crossed below 126 and stayed
  there (increments -1.97, -1.47, +0.02, -0.29, -0.29, -0.11).
  AND IT CAUGHT C26 ONE MORE TIME, IN ANOTHER LANE'S NUMBERS: round 21's m31
  mod-5 class counts differ from the exact cyclic ones in EXACTLY ONE CELL by
  EXACTLY ONE - class 2, because the m31 wrap gap is 7 and 7 = 2 (mod 5). Size
  and location both predicted by "wrap gap = first gap". Nothing concluded
  moves (relative error 1.6e-10); exact-integer identities must use the cyclic
  row.

F_2(53) = 159 EXACT - a first computation (three range workers TILING the
period, floor-1 lap-phase transfer at r = 7, the deepest yet). Witness
re-verified at machine 53: k = 327,666,424,664,536,738, gaps [77, 82], all 157
interior slots blocked. TWO CONSEQUENCES: (i) the deletion ladder gives
F(59) >= 159, i.e. F(2,59) >= 477 - a NEW UNCONDITIONAL LOWER BOUND on the next
corpus rung, where the ladder stopped at F(2,53) = 435; (D) at 53->59 needs
F(59) <= 204, so at most 45 of room remains there. (ii) it prices the
A_kill(53->59) campaign: with the F_2 span cap, the phase screen and the mirror
law the levels collapse from 36/170/776 legal words to 6/11/10 SOLVER CALLS -
a 36x cut in which every step is a theorem.

CROSS-LANE, ADOPTED MID-ROUND: Lateral's mirror law #occ(w) = #occ(rev w).
a_kill_par.py now collapses every level to its reverse classes and copies the
verdict, and the law was CHECKED AGAINST THIS LANE'S OWN DATA in the gate - all
37 realised words are reverse-closed, the legal-word lists are reverse-closed
(5,636 words), and the phase-saturation screen is provably reverse-invariant
and agrees on every reverse pair. Their 46%-waste audit is now rule 27. It
showed up unbidden three more times this round: both Q* anchor witness pairs
and both 53->59 pair witnesses are exact mirrors.

HONEST NEGATIVES AND COSTS:
- A_kill(53->59) IS NOT DECIDED. Established: >= 3, and the alternation family
  supplies at most 3. The full campaign is priced (27 solver calls) and
  deferred. Rule 22: this is a LOWER bound only.
- THE m41 HISTOGRAM AND THE EXACT m41 4-TUPLE CENSUS ARE NOT DELIVERED. The
  brief's "nearly free from existing data" was wrong (see the correction
  above). Measured price: T = 1,363,783 laps at 0.062 s/lap alone = ~85,000
  core-seconds for the histogram, ~1.7x that for histogram-plus-tuples, at
  ~330 MB per worker. Launched at 15 workers, it drove free RAM from 6.0 GB to
  1.1 GB with the CPU counter at 38% - paging, not computing - completed no
  worker in 80 minutes and was killed (free RAM went straight back to 6.0 GB).
  At a memory-safe 6-8 workers it is a 3-5 hour job. Not restarted, because a
  run that cannot finish must not be reported around.
- THE DELTA OPTIMISATION IS A WASH. O(n) -> O(2n/q) on the innermost gear
  should have been ~20x; measured ALONE and SEQUENTIALLY on the same slice it
  is 71 s vs 65 s, 1.09x - 25 numpy calls on 400-750k arrays cost what 4 calls
  on 7M cost. Kept only because it is BIT-IDENTICAL to the simple path, which
  makes it a permanent equivalence gate.
- I MIS-PARSED MY OWN CLI AND LOST A RUN. The F_2(53) launch omitted the
  argv[8] slot, so every worker read its own HI as its I0 and walked a SUFFIX
  instead of a tile - caught because one worker reported "windows walked 0" and
  another an index past its own range. Re-tiled and completed. j5_multi.py now
  prints the walked range UNCONDITIONALLY.
- FORTY MINUTES BLIND, TWICE. Histogram workers printed every 20,000 laps with
  5,735 each; Q* workers every 200,000 indices with 795,217 each. Healthy jobs
  were indistinguishable from stalled ones and I nearly killed one twice. Rule 28.
- BELOWNORMAL PRIORITY IS STARVATION, NOT POLITENESS, ON A LOADED BOX. Five
  47->53 workers at BelowNormal did under 200,000 indices in 100 minutes;
  relaunched at Normal they did 28,400 in 190 seconds. Rule 30.

STANDING ADDITIONS: rules 27 (one word per reverse class), 28 (progress stride
from the worker's own share), 29 (run the arithmetic screen before the solver),
30 (process count is not load - measure % processor time and free RAM).


## Manager note (2026-08-31, post round-26 close): the human's sort-step idea - first probe

The human suggested viewing the sliding windows in a SORTED order (a sort step per turn)
under which the gap pattern becomes obvious. Manager probe confirms on contact: in CRT-LEX
ORDER (sort openings by phase vector) the distinct adjacent-difference count is EXACTLY 2n
at n gears - 6, 8, 10, 12, 14 at machines [5,7,11] .. [5..23], verified exact. Natural
order: 7, 10, 17, irregular. Registered as docs/novel/two-n-gap-reordering.md (MEASURED,
prior art not yet checked - the lex fact may be folklore; the shuffle framing is the delta).
ROUTED TO LATERAL (round 27, optional addition to their own-mandate list): prove the 2n law
(odometer/carry analysis), check prior art, and ask their question - what shuffle statistic
is the natural-order record gap? Deliberately NOT added to any route-critical path per the
human's "don't shift direction" instruction.


## LP-duality thread (round 27)

### 0. FOR FORMALIST - THE JSON IS ON DISK AND GATED (posted as soon as it landed)

`research/data/r27/` : `cert_19_23_h<w>.json` (5 cases) and `cert_29_31_h<w1>_<w2>.json`
(35 cases), one file per case, integers only; `layout_<rung>.json` (the case-independent
column/link/atom layout - PROVED identical across cases, since with no required-open
positions every phase domain is all of Z_q); `manifest_<rung>.json` (held-phase tuple list
+ the exhaustiveness assertion `= prod Z_q`).  Emitter and gate:
`.venv/Scripts/python.exe research/emit_certs_r27.py GATE` (~5 s, ALL ASSERTIONS GREEN) -
it re-loads every JSON from disk, rebuilds the layout FROM THE PRIMES, recomputes every
O_j, re-checks every cut row by the exact zeta transform, and recomputes lhs/rhs from the
file's own integers.  Sizes: 15.1 KB per case at 19->23, 30.0 KB at 29->31.

**AND ONE FACT THAT SHRINKS YOUR OBLIGATION 3.  EVERY CUT ROW IN BOTH RUNGS IS THE BASE
CUT** (`rows_all_base_cut: true` in every file; both rungs certify at iteration zero, so no
degree-2 separating cut was ever generated).  So "cut validity" is not a 2^n subset-sum
check at all here - it is `sum_i x_i >= 1`, valid by inspection.  Two corollaries you can
use: `lam_0 = 0` in every row, so `rhs = sum_r y_r + yff*|pos|`; and a PAIR column's mask
is not a singleton, so cut rows contribute NOTHING to it - `a_j = yff*frow_j + (link terms)`
on pair columns, and only single columns see `y`.

### 0b. GATES (all four green this round, clean processes)

  research/star_case.py GATE                  ALL ASSERTIONS GREEN   91 s  (round 26's
                                              headline gate, unchanged and still green)
  research/emit_certs_r27.py GATE             ALL ASSERTIONS GREEN    5 s  (40 JSON files)
  research/gate_rung_41_43_r27.py <i> 5       385/385 re-verified   ~65 s per stripe
  research/increment_cert_r27.py GATE         ALL ASSERTIONS GREEN   75 s  (120 certs +
                                              6 witnesses rebuilt)
New files: `emit_certs_r27.py`, `increment_cert_r27.py`, `gate_rung_41_43_r27.py`,
`_m43k3_r27.py` (the resumable striped worker), `_predscore_r27.py`; data in
`research/data/r27/` (47 JSON, 1,118 certificates, 18 MB).

### 1. (b) THE NINTH RUNG IS CLOSED: 41 -> 43 CASE-SPLIT CERTIFIED AT k = 3, 385/385

Round 26's one badly-sized job is finished, and it certifies.  228 remaining cases on ten
striped resumable workers at 240 s/case (the round-26 sweep used 45 s and stalled on six);
ALL SIX ROUND-26 STALLS CERTIFY at the larger budget, so "budget-limited, not blocked" was
the right reading.

  rung      W    held        cases   exact certificate ops   re-verified from disk
  41->43  134    (5,7,11)      385              18,649,193   all 385, 5 stripes, ~65 s

  * case (0,0,0) closes 3523/128 < 1763/64; the SMALLEST margin over all 385 cases is
    19/100000, so width 134 is only just enough - the same knife-edge as every other rung.
  * iteration histogram: 371 cases certify at ITERATION ZERO (level-1 coverage rows plus
    the recursion row, NO cut generation), 14 need 2-7 cut passes.
  * gate: `.venv/Scripts/python.exe research/gate_rung_41_43_r27.py <i> 5` (five stripes,
    ~65 s each) - rebuilds each relaxation FROM THE PRIMES, re-checks every cut row's
    validity by the exact zeta transform, re-closes lhs < rhs in exact rationals, and
    asserts the 385 held-phase tuples are exactly Z_5 x Z_7 x Z_11.

WHY IT MATTERS BEYOND ARITHMETIC: 41 -> 43 is the step CONSTRUCTOR reported in round 26 as
NOT certified (oracle-bound at arity 4).  So the LP ladder is now TEN rungs - 7->11, 11->13,
13->17, 17->19, 19->23, 23->29, 29->31, 31->37, 37->41, 41->43 - every (D) step the project
has, plus one it did not have.

### 2. (c) THE VEHICLE REACHES THE INCREMENT WIDTH - THE MANAGER'S BASE CASES BY CERTIFICATE

THE QUESTION, restated precisely.  The increment law is
F(M+q') - F_2(M) <= s_min(q') = min(2u' mod q', (-2u') mod q').  A DUAL certificate can
carry the UPPER half only, and that half is exactly this vehicle run at the INCREMENT WIDTH
W_inc = F_2(M) + s_min(q') in place of the ladder's budget width F(M) + q'.  W_inc is
strictly smaller at every step, so this is a STRICTLY HARDER obligation than the (D) rung
and is NOT implied by it.

ANSWER: YES, at every literal step the vehicle reaches.

  step     s_min  F_2(M)  W_inc   budget   k   cases   exact ops   secs
  11->13     4      11      15       20    1     5        4,416      2
  13->17     6      16      22       28    1     5       10,620     <1
  17->19     6      25      31       37    1     5       22,409      1
  19->23     8      31      39       48    2    35      203,921      5
  23->29    10      39      49       63    2    35      365,473     23
  29->31    10      55      65       74    2    35      574,172     55

THE TIGHTER WIDTH COSTS EXACTLY ONE HELD GEAR, AND ONLY WHERE THERE WAS ROOM FOR IT TO.
19 -> 23 certifies at k = 1 at the budget width 48 (five cases at iteration zero) and does
NOT at W_inc = 39 - case w = 0 stalls - certifying only at k = 2.  23 -> 29 and 29 -> 31
already needed k = 2 at their budget widths and still need exactly k = 2 at W_inc, so the
extra difficulty is absorbed.  The ladder parameter is measuring difficulty, not serving as
a knob.

THE OTHER HALF IS A WITNESS OBLIGATION, AND IT IS DISCHARGED TOO.  F_2(M) >= its claimed
value is a REALISABILITY statement and no dual certificate can carry it.  `witness_f2`
builds an explicit phase vector by an exact-cover BACKTRACK over the gears - NO PERIOD SCAN
- and `check_witness` re-checks it by CRT arithmetic on [0, s]:

  F_2(11) >= 11  phases [2,0,10]                 openings [0,5,11]   split (5,6)
  F_2(13) >= 16  phases [2,0,10,0]               openings [0,5,16]   split (5,11)
  F_2(17) >= 25  phases [0,5,0,6,8]              openings [0,7,25]   split (7,18)
  F_2(19) >= 31  phases [2,2,8,7,11,7]           openings [0,10,31]  split (10,21)
  F_2(23) >= 39  phases [3,5,10,4,1,7,15]        openings [0,5,39]   split (5,34)
  F_2(29) >= 55  phases [0,3,7,1,6,17,12,2]      openings [0,20,55]  split (20,35)

Each is TIGHT (equals the recorded F_2), so the increment law holds at every literal step
through 29 -> 31 by CERTIFICATE + WITNESS with NO PERIOD SCAN ANYWHERE IN THE CHAIN.  Two
cross-checks worth noting: the m19 witness has split (10,21), which is exactly the maximiser
this vehicle located from the DUAL side in round 26; and the m29 witness reproduces the
project's F_2(29) = 55 independently and scan-free.

GATE: `.venv/Scripts/python.exe research/increment_cert_r27.py GATE` - ALL ASSERTIONS GREEN
(75 s); re-verifies all 120 increment certificates from disk and rebuilds every witness.

FOR THE MANAGER: this is the base-case supply your induction asked for.  What it does NOT
give you is the induction STEP - every one of these is a finite certificate at one machine,
and the vehicle's cost is a primorial in k, so it cannot be run "for all M".

### 3. NEGATIVE, AND IT IS THE INTERESTING ONE: A SECOND FRONTIER

41 -> 43 AT THE INCREMENT WIDTH W_inc = 117 (F_2(41) = 103 + s_min(43) = 14) IS NOT REACHED,
and the reason is new.  The pre-test PASSES comfortably - E_u[f_w] over sampled cases has
min +5.62 (k=1), +10.80 (k=2), +14.01 (k=3) at W = 117, so the product-measure width
frontier of `product-measure-frontier.md` does not obstruct it.  The CUT LOOP is what fails:
at k = 3, case (0,0,0), the LP maximum of the recursion row falls

  44.2578, 44.2083, 44.1398, 44.0282, 43.9540, 43.9020, 43.8266, 43.7705, 43.7399,
  43.6940, 43.6339, 43.5601, 43.5196, 43.4856   against the 43 it must fall below

(fifteen passes, 654 cut rows, 377 s), about 0.05 per pass and DECELERATING - at that rate
the crossing is ~10 more passes away and each pass is lengthening.  One case did not decide
in 35 minutes against 10-40 s per case at
the budget width 134.  SO THE VEHICLE'S COST IS NOT A SMOOTH FUNCTION OF THE WIDTH: it
explodes as W approaches the value being proved, while the necessary condition stays
healthy.  There are TWO frontiers here, and only the first has a closed form: the WIDTH
frontier (where the vehicle can work at all) and a CONVERGENCE frontier (where it does so
affordably).  Recorded in `product-measure-frontier.md` section 5 as a new open question.

### 4. SCORING MY ROUND-26 PRE-REGISTERED PREDICTIONS

E1 ("41->43 completes at k = 4 (5,005 cases) and 43->47 too; 47->53 needs k = 5")
   FIRST CLAUSE REFUTED, IN THE OPTIMISTIC DIRECTION - AND IT IS MY OWN COST MODEL THAT
   WAS WRONG.  41->43 completes at k = 3 (385 cases), not k = 4.  I over-priced my own
   vehicle by a factor of 13 in case count, because I read round 26's six stalls as
   "k = 3 is not enough" when they were "45 s is not enough".  A stall is a budget
   verdict, never a reach verdict, and I had already written that sentence about someone
   else's work.  The 43->47 and 47->53 clauses were NOT tested this round.

E2 ("the case-split ladder is monotone in k: no rung certified at k fails at k+1")
   CONFIRMED at the tested rung.  29->31 at budget width 74 - certified at k = 2 in round
   26 - re-run at k = 3: 385/385 CERTIFIED, 5,220,357 exact ops, 144 s on four workers,
   zero failures.  These are genuinely different LPs (the k = 3 case problems are not
   refinements of the k = 2 ones), so the test had content.  ONE RUNG ONLY - not a proof
   of monotonicity.

E3 ("the vehicle is tight on F at every machine once k is large enough: F(31) <= 58,
   which fails at k = 2 with 19/35, certifies at k = 3")
   CONFIRMED, EXACTLY AS PRE-REGISTERED.  385/385 cases certified, 5,294,517 exact ops,
   ~180 s on four workers, zero failures.  So the vehicle is now TIGHT ON F at FOUR
   machines: F(19) <= 25 and F(23) <= 34 and F(29) <= 43 at k = 2, F(31) <= 58 at k = 3 -
   each the exact value, each scan-free, each hypothesis-free.

E4 ("the windowed vehicle's integrality-gap cells at machine 29 cluster at spans 56-58
   against F_2(29) = 55, none below 53")
   NOT TESTED - and the probe says why it is expensive.  At machine 29, span 60, the
   PLAIN windowed vehicle (no held gear) leaves 8 of 22 sampled splits STUCK at a 40 s
   budget while 14 certify.  So the m29 F_2 ladder is a HELD-GEAR job throughout, not the
   one-second-per-cell sweep m19 and m23 were, and the full ladder (spans 56..86) needs
   sizing before it is launched.  Deferred deliberately rather than started late.

### 5. FOR OTHER LANES

- FORMALIST: section 0 above.  The two rungs you named are emitted and gated; the
  base-cut fact removes your obligation 3 for both of them.
- CONSTRUCTOR: 41 -> 43 - the rung you recorded as NOT certified - is certified here at
  the same budget width 134, hypothesis-free, 385 exact dual certificates.  Independent
  vehicle, same conclusion; if your closure route later reaches it, the two are a
  cross-check rather than a duplication.
- MANAGER: your increment law's LITERAL-STEP BASE CASES now exist as certificates plus
  scan-free witnesses at 11->13, 13->17, 17->19, 19->23, 23->29, 29->31 (section 2).  The
  vehicle gives you base cases, not the induction step - its cost is a primorial in k, so
  it cannot be run "for all M", and the width at which it stops converging (section 3) is
  a second obstacle with no closed form.
- MECHANIC: `witness_f2` reproduces F_2(29) = 55 with an explicit phase vector and no
  period scan, in about two minutes.  The same backtrack should reach F_2(41) = 103 and
  F_2(53) = 159 as independent scan-free confirmations of your exact values.

### 6. PRE-REGISTERED PREDICTIONS FOR ROUND 28 (score them next round)

E5  The vehicle is tight on F at machine 37 too: F(37) <= 88 certifies at k = 3 or k = 4.
E6  THE CONVERGENCE FRONTIER IS ABOUT THE MARGIN, NOT THE WIDTH: at machine 43, k = 3,
    the cut loop converges for every W >= 128 and for no W <= 120.  (The budget width 134
    converges in 10-40 s; W = 117 does not converge in 35 minutes.)
E7  41 -> 43 at the increment width 117 CERTIFIES at k = 4 (5,005 cases).  Recorded
    knowing E1's lesson: this is a reach claim, and if it fails at k = 4 within a proper
    budget I will say the reach claim was wrong rather than that the budget was.
E8  The m29 F_2 ladder completes with gear 5 held throughout, and its leftover cells sit
    at spans 56-58 (E4 restated at the k it actually needs).



## Harvester round 27

GATES, all four re-run from clean processes at round close, all GREEN:
  .venv/Scripts/python.exe research/j2_referee.py    -> ALL ASSERTIONS GREEN
  .venv/Scripts/python.exe research/j2_citesweep.py  -> ALL CHECKS GREEN
  .venv/Scripts/python.exe research/j2_odcpages.py   -> ALL ASSERTIONS GREEN (NEW)
  .venv/Scripts/python.exe research/jk_family.py     -> ALL ASSERTIONS GREEN (NEW)
j2_referee.py was run FIRST, before anything below entered the record. Every
job this round launched has finished; nothing left running.

BRIEF ITEM (a) - **THE ODC PAGE-IMAGE CAVEAT IS CLOSED.** Rounds 24, 25 and 26
each ended with the same sentence - "(5.38), (6.69) and p. 74 were not
re-fetched; one library visit closes it". All three are now read first-hand
(2026-08-29), plus pp. 43, 44, 45 for context: research/data/odc6_scans/PA42,
PA43, PA44, PA45, PA67, PA74. **NOTHING IN THE LADDER MOVES** - no constant,
exponent or threshold changes - but each page paid something:

- **(5.38), p. 42** is `prod_{w<=p<z}(1-g(p))^{-1} <= K (log z/log w)^kappa`
  for z > w >= 2, "where K is a constant, K > 1" - exactly the form rounds
  23-24 took from two transcriptions, now confirmed against the book. TWO
  BY-PRODUCTS: K > 1 is REQUIRED, and the book prints `g(p) <= 1 - 1/K`, whose
  converse `K >= (1-g(p))^{-1}` **EXPLAINS OUR WHOLE PRE-SIEVED K-LADDER IN ONE
  LINE** (K = 3 at p_0 = 3 because g(3) = 2/3 exactly; 5/3 at 5; 7/5 at 7).
  Round 23 found those by grid search and could not say why.
- **(6.69), p. 67** turns out to be `alpha < 1/c` with c the root of
  c(log c - 1) = 1 - i.e. the convergence condition a = alpha e^{1+alpha} < 1,
  and **c IS our own Theorem 3E constant lambda_* = 3.591121**. Corollary
  6.13's own beta_kappa forces **alpha = 1/4 IDENTICALLY IN kappa**, so the
  hypothesis holds at EVERY dimension, not merely ours (checked at nine).
  NEW NUMBER: since beta = coth(alpha/kappa) decreases in alpha, (6.69) puts an
  ABSOLUTE FLOOR of **7.22859** under ODC Ch. 6 at kappa = 2, below our
  positivity floor 7.93727 - **so (6.69) is not what stops Chapter 6,
  POSITIVITY IS**, and even discarding both the chapter cannot print an
  exponent under 7.229, still 3.0 above DHR's 4.266.
- **p. 74** shows ODC's own preliminary sieving (6.99)/(6.100) carrying an
  implied constant "depending only on K_0 and kappa_0" - **NOT EXPLICIT**. So
  it could never have supplied our pre-sieving factor: round 24's elementary
  N_pre accounting is THE ONLY EXPLICIT ROUTE and stays ours. AND the book's
  OTHER route to K -> 1, which we had never considered - (5.42)/(5.43),
  enlarging the dimension by eps instead of pre-sieving - is now priced and
  REJECTED on the book's own arithmetic: eps = 1 costs 3.93 of exponent
  (11.871 vs 7.937), eps = 1/2 costs 1.98. Pre-sieving keeps kappa = 2 and is
  the right device; round 24 chose it without the comparison.

METHOD NOTE FOR EVERY LANE: round 25's fetch failed because the publisher
preview serves page images only to a session holding its cookies - the naked
URL returns a 9,103-byte placeholder. Driving a browser to the volume, reading
the jscmd=click3 page list for the SIGNED image URLs and fetching them inside
that session returns the pages. **A source recorded as "not obtainable" was a
missing cookie, and three rounds inherited the verdict.**

BRIEF ITEM (a) - **THE SUBMISSION MEMO IS WRITTEN:
docs/novel/unit1-submission-memo.md.** One page, and it deliberately does NOT
recommend submitting - THE DECISION IS THE HUMAN'S. It carries: what the paper
claims and what it does not; the three strongest points a referee will see (an
empty ladder now four explicit rungs deep, WITH the explicitness boundary
proved so the obvious improvement is pre-answered; (P2') as a new construction
that is parity-free for a structural reason; a visibly self-auditing paper);
the three weakest ("this is an exercise"; the audience; an asymptotic lower
half over a replicated computational half); a venue-class assessment; and the
AI-assistance disclosure question, flagged with the facts and NOT decided.
THE AUDIENCE NUMBER, stated plainly because it decides the venue:
**arXiv:1706.00317 has EXACTLY ONE CITATION IN NINE YEARS** (its own companion
note), and zbMATH returns no records for "paired Jacobsthal". Venue reading:
arXiv math.NT first whatever else is decided; JNT / Ramanujan J / Acta Arith /
Mathematika in range IF (P2') travels with it; INTEGERS or JIS if it stays
elementary; not a general-audience venue. Plus one free suggestion: write to
Ziller and Morack, who are the prior readership, the natural referees, and the
people who can compute h_2 at p_n = 151.

BRIEF ITEM (b) - **(P6), THE k-FAMILY, IS WRITTEN: docs/novel/jk-family.md.**
The object: j_k(m) = max over ADMISSIBLE k-tuples E of the largest gap between
consecutive n with all n+E_i coprime to m. k = 1 is the ordinary Jacobsthal
function, k = 2 is Ziller-Morack's h_2.

  PROPOSITION (covering restatement, and it is the whole content):
    j_k(P(z)) - 1 = the longest interval coverable by choosing at each prime
    p <= z a set S_p of classes mod p with  |S_p| <= min(k, p-1).

Both directions are CRT. `min(k, p-1)` gives 1 everywhere at k = 1 and
**ZM's omega(2) = 1, omega(p) = 2 at k = 2** - our own g(2) = 1/2, g(p) = 2/p -
and it is what makes THE SIFTING DIMENSION EQUAL TO k. Both forms were
brute-forced independently and agree at k = 1,2,3 x z = 3,5,7: k = 1 returns
A048669's 4, 6, 10; k = 2 returns ZM's 6, 18, 30; and **k = 3 returns
j_3(P(3)) = 6, j_3(P(5)) = 24, j_3(P(7)) = 78 - a first evaluation.**
THE LADDER IS UNIFORM IN k, which is the reason to publish the family:
Legendre with omega_p = min(k,p-1); **the explicit polynomial rung
j_k <<_{k,eps} z^{beta_k+eps} with beta_k = 1 + 2(e^{1/(2k)}-1)^{-1} in
(4k-1, 4k+1)** - whose k = 2 entry IS Theorem 2G's 8.041623, so the family
rung CONTAINS Unit 1's best explicit bound - and (P2')'s
x A^{2k-1} C^k/B^{2k}. CONJECTURE: j_k(P(x)) = x (log x)^{2k-1+o(1)}.
HONEST AND IN THE NOTE: at k = 1 the family rung (4.083) is WORSE than
Iwaniec's record 2; the upper rungs are standard sieve theory applied to a new
object, not new sieve theory. WHY IT MATTERS: it converts Unit 1's weakest
structural point ("one function, standard tools") into "a family, and the
family is the contribution", and it locates ZM Conjecture 6 inside the family
- exponent k at dimension k, the level at which a survivor in (y, y^2] IS a
prime k-tuple, so **THE PARITY CEILING IS UNIFORM IN k**, not special to twins.

BRIEF ITEM (c) - **THE LOWER LADDER'S NAMED NEXT QUESTION IS ANSWERED, AND IT
COSTS NOTHING.** layered-erdos-rankin.md sec. 6 item 3 (the k >= 4 shift set):
(i) `0,2,...,2(k-1)` is the WRONG tuple - from k = 3 it is not even admissible
(0,2,4 covers Z/3), which round 26 did not notice while tabulating its
collisions; (ii) with ANY admissible tuple, a collision needs p | E_j - E_i so
p <= M_k, a constant in k, while the greedy layer runs over [P, z1] with
P = A^{2k-1} -> oo - so for large x EVERY colliding prime is BELOW P, inside
the Eratosthenes layers, where a collision merely means two layers coincide
and the survivor structure is untouched. Sigma = prod(1-k/p) needs no
correction and K_k stands as printed; (iii) threshold x > exp(M_k^{1/(2k-1)}),
under e^4 for every k <= 12 against the construction's own log x ~ 300. What
is left is a finite optimisation (which admissible tuple minimises c_1^{(k)}),
not a gap.
ALSO, A SIMPLIFICATION OF OUR OWN: the greedy lemma at general k has a
ONE-LINE proof - the p class counts average N/p, so the k largest average at
least N/p and sum to at least kN/p - which subsumes round 26's k = 2 argument.
The k = 2 statement 2N/p was and is exact; this is a simplification, not a
correction.

NEGATIVES AND COSTS:
- ROUND 26 TABULATED COLLISIONS FOR AN INADMISSIBLE TUPLE (mine, one round
  old). Harmless at k = 2, but it is the lane's recurring shape: carrying a
  small-k object into general k without re-checking the definition.
- ROUND 26's GREEDY PROOF WAS LONGER THAN NECESSARY (also mine).
- (6.69) TURNS OUT NEVER TO HAVE BEEN AT RISK - it holds at every kappa. Three
  rounds carried it as an open caveat because we priced Proposition 6.7 and
  did not read the condition it cites BY NUMBER.
- THE PAGE IMAGES ARE STILL IMAGES, not a copy in hand. Mitigated four ways
  (the pages cross-check each other and agree with two transcriptions of Thm
  7.7), not removed.
- j_3 BEYOND z = 7 NOT COMPUTED: the covering-form search is exponential in
  the number of primes and z = 11 needs a real algorithm. Named, priced, and
  deliberately not started - it would not have finished in-round.
- NO PRE-REGISTRATION THIS ROUND. Fetch-and-write work; the one outcome I did
  not know in advance was (6.69) and I wrote down no prediction. A miss.

TWO ADDITIONS TO THE STANDING CITATION-HYGIENE LESSON (harvester 7d):
9.  **A PAGE THAT WOULD NOT FETCH IS NOT A PAGE THAT CANNOT BE FETCHED.** When
    a source is recorded as unobtainable, record HOW the attempt was made, so
    the next round attacks the method instead of inheriting the verdict.
10. **A HYPOTHESIS CITED BY NUMBER IS AN UNREAD HYPOTHESIS.** Follow every
    numbered reference inside a result you are using, to the page. Clause 4
    (price the propositions) one level further down again.

RANKING CHANGES: N4 (upper ladder) stays TOP and its LAST research-shaped item
is gone - every blocker, opening and caveat of rounds 23-26 is closed, and
what remains is LaTeX, a scope decision, and a decision that is the human's.
**(P6) RISES to sit beside N4**: written, gated, prior-art-checked, and it is
the piece that answers the strongest referee objection to N4. P1-P3 unchanged
in rank, one item lighter. **7c#4 (h_2 at p_n = 151..251) becomes the lane's
top RESEARCH item by default** - the only purchasable number separating
z(log z)^2 from z(log z)^3. DEMOTED: nothing.

FOR OTHER LANES:
- ANY LANE THAT HAS RECORDED A SOURCE AS UNOBTAINABLE: see lesson 9. Ours was
  a missing cookie and it cost three rounds.
- FORMALIST: still nothing new that is kernel-reachable - the family's rungs
  are asymptotic. But the COVERING RESTATEMENT (|S_p| <= min(k, p-1)) is a
  finite, decidable statement at any fixed (k, z), and the k = 1,2,3 values at
  z = 3,5,7 are exact integers with witnesses; if a finite kernel candidate is
  ever wanted from this lane, that is the cleanest one it has ever had.
- MANAGER: Unit 1 needs no more research. The memo is the handover;
  the submission decision, the scope decision (does F_d travel? does j_k
  travel inside Unit 1 or separately?) and the AI-disclosure decision are all
  the human's, and the memo is written to make them makeable rather than to
  make them.

## Lateral round 27

GATES (all re-run from clean processes at round close, all exit 0):
  research/lex_odometer.py --parts ABCDEFGH   -> 145 assertion gates
  research/ghist37_u69.py                     ->  45 assertion gates
  research/u5_collisions.py --y 29            ->  10 assertion gates (needs mpmath)
  research/tooth_counterfactual.py --upto 19  ->  10 assertion gates
Predictions P1-P13 pre-registered in research/data/r27_lateral_predictions.txt,
each block written before the code it scores. Every job launched has finished;
nothing left running.

CHOSE: the routed 2n law in full, then U6 + U9 (unblocked by Mechanic's exact
m37 histogram), then U5 - untouched for three rounds and it fell to two lines of
field theory. NOT WORKED, unclaimed: U7, U10, U11.

1. THE ROUTED 2n LAW IS PROVED - AND THEN REFUTED AS A ROUTE, BY ITS OWN PROOF.
   CRT-lex order IS the mixed-radix odometer, so the lex successor increments the
   last non-maximal digit and wraps the rest, giving difference
   D(i,delta) = CRT(0 below i ; delta at i ; the wrap w above i). The carry
   position is recoverable (coordinates below i are 0), so
       #distinct differences = sum_i d_i,  d_i = #distinct consecutive
                                                differences of the sorted A_i,
   and d_i = 2 at every gear because the teeth are NEVER adjacent (adjacency
   needs 3 = +-1 mod q). Hence 2n. Closed-form multiplicities
   mult(D(i,delta)) = s_i(delta) prod_{i'<i}(q_{i'}-2); the CYCLIC closure is
   FREE for the machine's own teeth (w_1 = -max(A_1) is 1 or 2, so the wrap is
   already D(1,w_1)); the count holds for EVERY gear ordering. Also an explicit
   bijection Phi: [0,N) -> O (a generalised van der Corput point set), so F is
   literally P times a digital sequence's dispersion.
   PRIOR ART, checked this round on the web: KNOWN IN MECHANISM. Langevin's
   lex-successor theorem for planar lattices (successor in {w+u, w+v, w+u+v},
   and it RECOVERS the three-distance/three-gap theorems) is the same carry
   argument; Fried-Sos extend it to ordered abelian groups. The finite CRT
   version is folklore-grade, as the first probe guessed. Surviving delta: the
   multiplicity table, the free wrap, order-independence.
   THE DEFLATION, and it is gated, not a judgment: the count depends on each gear
   ONLY through "how many distinct interior run-lengths does the removed set
   have", which is 1 for every two-point removal bar a degenerate terminal pair.
   So over 60 admissible RE-CHOICES OF THE TEETH the count stays 2n = 8 while F
   ranges over [10,18] - a factor 1.8 - and the law even holds for coprime
   NON-prime moduli. THE COORDINATES DISCARD EXACTLY THE ARITHMETIC F DEPENDS ON.
   And F is not a statistic of the order permutation at all: a permutation
   records order, F needs the metric. The dual count (distinct lex-index
   displacements between natural-order neighbours) is 5, 25, 95, 368, 1362 at
   n = 2..6 - complexity moved, not reduced. docs/novel/two-n-gap-reordering.md
   rewritten: PROVED, prior-art verdict recorded, and marked a CLOSED LINE.

2. U5 CLOSED AFTER THREE ROUNDS - AND THE ROUND-21 DEGENERACY LAW IS NOW A
   THEOREM AT EVERY MACHINE. The circulant's eigenvalues are products of one
   factor per gear from S_q = {q-2} u {-2cos(2 pi r/q)}, none zero. Equality of
   two such products forces prod (f_q/f'_q) = 1 with the ratios in the real
   cyclotomic fields Q(zeta_q)^+, which have PAIRWISE COPRIME CONDUCTORS and are
   therefore linearly disjoint - so every ratio is rational, and a rational ratio
   inside S_q forces equality (norms kill -1; 2(r+-r') = q is impossible for odd
   q). Hence the degeneracy group is exactly (Z/2)^{#gears}: #distinct =
   prod (q+1)/2, ties = P - prod (q+1)/2, EVERYWHERE, and NO accidental exact
   collision exists at any machine. Round 21's measurement at three machines is
   upgraded to a proof at all. DECISIVE TEST at m29 (where round 21 logged 6
   near-collisions at 1e-12): all 8,164,800 desymmetrized levels rebuilt, exactly
   those 6 found, each recomputed at 60 digits - ALL SIX SEPARATE, smallest
   8.635e-14. Crowding measured (median spacing 1.30e-05). m31's 613 are covered
   by the theorem and NOT re-measured - 1.3e8 labelled levels is not memory-safe
   here, said plainly. Free double-source: the script re-derives round 21's tie
   counts 313 / 4501 / 80549 at m11/13/17 by brute force.

3. U6 AND U9 CLOSED ON MECHANIC'S EXACT m37 HISTOGRAM - AND -1/phi IS A CROSSING,
   NOT A LIMIT. Exact m37: alpha_1 = 4,107,707,379, alpha_2 = -7,109,650,222,
   ratio -0.577765. Ladder m11..m37: -0.8636, -0.8393, -0.7305, -0.6402,
   -0.6448, -0.6231, -0.5943, -0.5778. It crosses -1/phi = -0.61803 between m29
   and m31 and is +0.0403 past it at m37, still rising (increments +0.0288,
   +0.0166 - decaying, not turning). SO MY OWN ITEM 48's "the machine drives the
   ratio TO the golden direction" IS REFUTED as an asymptotic claim; every exact
   identity under it survives. Same event as arg H_5(1) crossing 126 deg.
   AMPLITUDE (U9): |H_5(1)|/N * lam exact at m11..m37 = 1.125953, 1.036230,
   1.015003, 1.013946, 1.019315, 1.016081, 1.009970, 1.014085 - it does NOT
   break, it OSCILLATES in [1.0100, 1.0193] from m17 on. The m31 -> m37 move is
   UP, siding with the corridor ladder against M1 - and against my own
   pre-registered P8, which called DOWN. The "which model" question was mis-posed.
   CROSS-GATES: total = prod(q-2) and gap sum = P at all eight machines; gap 1
   the only odd entry at all eight; alpha_1 odd at all eight; arg H_5(1)
   reproduces Mechanic's exact ladder to 5e-3 deg; the amplitude column
   reproduces my round-25 table to 6e-4.

4. THE ROUND'S MOST INTERESTING NUMBER, AND IT WAS NOT IN THE BRIEF - THE TWIN
   MACHINE IS A LOW-F OUTLIER AMONG ITS OWN COUNTERFACTUALS. Keep the gears, keep
   the mirror symmetry, and move the teeth: v_q ranges over {1..(q-1)/2}, every
   member has the SAME period, the SAME survivor count prod(q-2) and the same
   per-gear density - only positions move. F is invariant under k -> +-k+b but
   NOT under k -> ck, so F genuinely varies, and the family (30 / 180 / 1440 /
   12960 members at m11/13/17/19) is small enough to ENUMERATE EXHAUSTIVELY:

       y    |V|     F(twin)  min  median  max   twin's percentile
       11   30      7        6    8       11    20.0%
       13   180     11       10   13      25    18.1%
       17   1440    18       14   19      32    26.4%
       19   12960   25       20   28      43    17.1%

   THE REAL MACHINE SITS IN THE BOTTOM FIFTH TO QUARTER OF ITS OWN F
   DISTRIBUTION at every machine tested, ~10-15% below the median, never the
   minimum, in a family whose maximum is 1.6-1.9x the truth. This is the FIRST
   quantity on which the real phase vector is distinguished: round 2's
   enumeration (my Refuted 3) scored it on WASTE metrics and found "no
   variational handle"; F itself separates, and in the FAVOURABLE direction.
   MECHANISM: OPEN, with two of my own candidates already dead - angular
   coherence REFUTED IN THE SIGN (spearman(F, dispersion) = -0.14/-0.20/-0.11;
   the twin sits in the lowest-dispersion quartile, which has the HIGHEST mean F,
   and is at the 10.5-20.8 percentile INSIDE it), and "the teeth are the
   reciprocal of a small integer" REFUTED (median over m = 1..60 is exactly the
   family median; the sweep's argmin is m = 12, not the twin's m = 6).
   CAVEAT I insist on: the four rows are NESTED, so they are not four independent
   draws and no p-value is claimed. docs/novel/tooth-counterfactual-percentile.md.

SCORECARD: 9 confirmed, 4 refuted, all four refutations my own (P8, P11, P13 and
the routed item's own framing). Last round I called my 10/10 suspicious; this
round four of thirteen were cheap corollaries and I lost three of the six real
bets, which is the right shape.

FOR OTHER LANES:
- MANAGER: the routed 2n item is proved, prior-art-checked and CLOSED as a route.
  What is worth your attention instead is item 4 and its route-side question,
  which I am routing rather than working (mandate): the counterfactual family
  fixes prod(q-2), so it is a clean null model for (D) too - what is the twin
  machine's percentile in the counterfactual distribution of the budget slack
  F(M+q') - F(M) - q'? If it is favourably placed there as well, the first-moment
  transfer has a MEASURED amount of room it is not using.
- ANY LANE citing "alpha_1/alpha_2 -> -1/phi": it is a crossing, not a limit.
- FORMALIST (offered, not claimed): "#distinct eigenvalues = prod (q+1)/2" and
  the 2n law are both finite per machine and smaller than the mirror even-count
  lemma already queued.
- MECHANIC: nothing owed to me. Your exact cyclic m37 histogram closed two
  backlog items in one pass and my script re-derives your arg ladder to 5e-3 deg
  as an independent check. An exact m41 histogram would add one rung to both
  ladders and make "does the overshoot decelerate to a limit" decidable.
## Constructor round 27

GATES (all re-run from clean processes at round close, all GREEN):
  .venv/Scripts/python.exe research/uniform_order.py   -> all assertions passed
  .venv/Scripts/python.exe research/increment_law.py   -> all assertions passed
  .venv/Scripts/python.exe research/triple_41.py --y 37 --q 41 --floor 89
      -> MAX = 90, witness (28,14,48) - reproduces the exact m37 census answer,
         value and witness, by a completely different vehicle
  research/chain_ps.py -> 0 false kills on all 291,675 realised m37 4-tuples
Logs: research/data/r27_uniform_order.log, r27_increment_law.log,
r27_triple_41.log, r27_chain_ps_41.log, r27_chain_ps_41_f2.log.
Pre-registration: research/data/r27_prediction.txt (written first, scored in
docs/proof-search/constructor.md).
Every job this round launched has finished; nothing is left running.


HEADLINE, AND IT IS TWO-SIDED. R67's residue (i) - THE ORDER, "nothing says
which m makes A_m nilpotent at machine M", asked for in two consecutive rounds
as "a bound, any bound, valid at every machine" - now has one for the object
the project has always used as its proxy, and a PROOF THAT NO SUCH BOUND EXISTS
FOR THE OBJECT THE CHAIN ACTUALLY NEEDS.

THEOREM (uniform alternation order). For every machine M = {5..y}, y >= 7,
    A_relax(M) <= 5,
and <= 4 unless q' = 37, 53, 83, 127, 157, 173 (mod 210). Proved by phase
saturation at GEARS 5 AND 7 ALONE, so the whole statement is a function of
q' mod 210 with no machine in it: X for the m-letter alternation is
{0, a, q', q'+a, 2q', ...}, so X mod g depends only on (a mod g, q' mod g), and
3a = q' -+ 1 fixes a mod g from q' mod 6g. All 48 invertible classes enumerated:
orders 2 / 3 / 4 / 5 at 24 / 16 / 2 / 6 classes. Gears 11 and 13 refute NOTHING
further (60 of 60 and 720 of 720 refinements stay at 5). Cross-checked by a
direct sweep of every prime q' < 20000 with all gears up to 100 - same
distribution, same exceptional residues.

AND THE SIX EXCEPTIONAL CLASSES ARE EXACTLY R20's LITCAP-6 CLASSES. By CRT a
translate of a point set fits inside the exposed sets of gears 5 and 7
separately iff it fits inside the corridor E mod 35 - so MECHANIC'S PHASE
SATURATION AT {5,7} AND THE LITERAL CAP ARE THE SAME ARITHMETIC, and the only
difference is the quantifier over starting letters (litcap MAXIMISES, being
about chain existence; the order MINIMISES, since one broken window kills the
cycle). Two invariants found independently five rounds apart are one object.

THE COMPANION NEGATIVE, and it is the sharper half. A_relax tests ONE candidate
cycle; A_m is nilpotent only when EVERY legal cycle is broken, and padded
letters (= 0 mod q') are T3-transparent. With N(M) = smallest m at which A_m is
acyclic, measured exactly from the same dictionaries:

    machine   11 13 17 19 23 29 31 37
    A_relax    1  2  2  3  2  3  4  2      (corrected - see the self-correction)
    N          2  2  2  3  2  3  4  3

R49's identity N = max(2, A_relax), which held 7 of 7, is REFUTED at the eighth
machine, and the extra order is bought by a PADDED cycle (m37's legal values are
{14,27,41,55,68,82}; 41 and 82 are transparent, so 14 -> 41 -> 27 -> 41 -> 14
is legal where the pure alternation (14,27) does not even occur).
AND THE METHOD DIES AT A LOCATABLE STEP. CORRCAP(q',F) = longest T3-legal word
with values <= F whose prefix-sum walk stays in E mod 35 - the strongest cap
gears 5 and 7 can EVER give - is 4, 2, 3, 5, 25, 25, 11, 5 at 19->23 .. 47->53
and INFINITE FROM 53 -> 59 ON (F/q' = 2.5), and at every larger ratio tested.
Mechanism: padded letters step by j*q' mod 35 and gcd(q',35) = 1, so once F/q'
is large the steps fill Z_35 and the corridor stops constraining. F/q' grows
without bound along the chain, so NO FIXED SET OF SMALL GEARS CAN EVER CAP THE
ORDER AGAIN. The correctly-posed uniform-order question is about the COVER half
of the realisability CSP, which no bounded gear set can supply (bigger machines
make covering EASIER).

THE INCREMENT LAW AT THE THREE DEEP STEPS: HOLDS 3 OF 3.
    41->43   F(43) - F_2(41) = 103 - 103 =  0  <= 14
    43->47   F(47) - F_2(43) <= 118 - 103 = 15  <= 16
    47->53   F(53) - F_2(47) = 145 - 134 = 11  <= 18
The middle row needs NO COMPUTATION: F_2 is non-decreasing in the machine
(proof in the append), so F_2(43) >= F_2(41) = 103 without ever computing
F_2(43), which the corpus only brackets in [103,118]. With R68's witness table
the increment law now holds at ELEVEN of the twelve testable steps, failing only
at the padded 31->37 by +8.

THE MANAGER'S TRIPLE INEQUALITY, EXACT AT EIGHT STEPS PLUS A NINTH SCAN-FREE,
AND THE PADDED STEP'S FAILURE IS ENTIRELY IN THE PADDED MIDDLE. Separating
LITERAL middles (w = +-s mod q') from PADDED ones (w = 0 mod q'), which the
manager's five-step probe never had to do (no padded middle exists below
19->23):

     M    q'  s_min  F_2  rhs | literal max slack witness    | padded max slack
    11    13     4    11   15 |     8       +7  (1,4,3)      |  none
    13    17     6    16   22 |    18       +4  (5,11,2)     |  none
    17    19     6    25   31 |    25       +6  (5,13,7)     |  none
    19    23     8    31   39 |    33       +6  (5,8,20)     |    31   +8
    23    29    10    39   49 |    43       +6  (10,10,23)   |    40   +9
    29    31    10    55   65 |    58       +7  (18,10,30)   |    49  +16
    31    37    12    68   80 |    70      +10  (5,25,40)    |    85   -5
    37    41    14    90  104 |    90      +14  (28,14,48)   |    83  +21

THE LITERAL TRIPLE INEQUALITY HOLDS AT EVERY STEP INCLUDING THE PADDED 31->37
(70 <= 80). My literal column reproduces the manager's independent probe digit
for digit at their five steps (8, 18, 25, 33, 43). Per depth, against R68's
exact Q* table, Q*_J - F_2 <= s_min holds 7 of 8 and every failing cell contains
the padded letter 37.

A FREE REDUCTION OF THE DEPTH-3 OBLIGATION, FOR THE MANAGER'S DERIVATION. Both
(g_L + w) and (w + g_R) are 2-gap windows of M, so with no hypothesis at all
    g_L + w + g_R <= F_2(M) + min(g_L, g_R),
i.e. the triple inequality is AUTOMATIC whenever the smaller flank is <= s_min -
which discharges 6 of the 8 steps outright. And the quantity the law must cap,
Delta_3 = max over legal triples of (span - F_2), is -3, 2, 0, 2, 4, 3, 2, 0:
BOUNDED BY A CONSTANT, no trend, while s_min grows linearly (4, 6, 6, 8, 10, 10,
12, 14). THE SHAPE TO AIM AT IS "Delta_3 = O(1)", NOT "Delta_3 <= s_min".

AND A NINTH STEP, SCAN-FREE. research/triple_41.py sweeps Mechanic's m41
transfer SUPERSET downward by span (a superset is a sound candidate source),
halves it by Lateral's reversal theorem, prefilters by phase saturation for
free, and decides the survivors with crt_dict. GATED at machine 37, where it
returns MAX = 90 with witness (28,14,48) - the exact census answer, value and
witness, by a completely different vehicle. AT MACHINE 41 (budget
F_2(41) + s_min(43) = 117), every candidate decided, ZERO undecided:
    LITERAL middles: spans 144 down to 108 all REFUTED  => max <= 107
    PADDED  middles: spans 130 down to 117 all REFUTED  => max <= 116
so Q*_3(41; legal for 43) <= 116 < 117 - THE TRIPLE INEQUALITY IS CERTIFIED AT
41 -> 43, the ninth step and the first with no census of any kind behind it.
The PADDED half is the tight one (+1 against the literal half's +10) - the same
asymmetry that makes 31->37 the family's single failure, seen one gear later
without a failure.
HONEST LIMITS: the exact maximum is NOT delivered (the literal descent was
stopped at the span-108 level boundary after one arity-3 instance at span 107
ran past 20 minutes against a 30-70 s norm; every completed level is a
self-contained refutation set, so "<= 107" is exact and "= 107" is not
claimed), and the stop decision was made on a STALE POLL - the level had
completed 156 s earlier and the job was healthy. New measurement for the
oracle-cost curve: m41 arity-3 CRT refutation cost is NOT uniform in span -
mean ~40 s with a heavy tail a node budget does not bound in practice.

RUNG NINE: NOT CERTIFIED, and the failure is now pinned to the ORACLE'S
INFORMATION CONTENT rather than to cost or strategy. No exact m41 census
appeared on disk, so item (c)'s precondition never arrived. What was done:
- PHASE SATURATION AS A FREE ORACLE TIER (research/chain_ps.py), gated sound
  (0 false kills on all 291,675 realised m37 4-tuples).
- IT ANSWERS ZERO of the 27,197 superset-YES queries the 41->43 loop asks. Not
  few - zero. Reason, structural: MF_4 is built MOD 35, so every edge it carries
  is already corridor-admissible, and phase saturation at gears 5 and 7 IS the
  corridor condition. FOR MECHANIC: "screens 4,239,676 -> 2,814,574 of the
  dictionary" is real as a screen of the DICTIONARY and worth nothing on the
  loop's own query stream; those are two different numbers and only the second
  is the oracle's cost.
- THE STALL IS AT 222 UNDER THREE SETTINGS: superset alone (R72), + phase
  saturation, and + phase saturation + the exact F_2(41) = 103 (which cuts the
  system to 1,058,228 states / 819,629 edges and 3,209 queries). 222 is the
  transfer superset's INFORMATION LIMIT. Only the CRT tier moves it. Route (ii)
  of R72 - Mechanic's exact m41 4-tuple census - remains the cheap route.

SELF-CORRECTIONS AND HONEST NEGATIVES:
- R45's A_relax(37) = 3 IS WRONG (it is 2), and the defect is MINE: my own
  arity_ladder.py HARDCODES the m = 1 and m = 2 entries at machines 29/31/37 as
  "realised" instead of looking them up. Second round running that a hardcoded
  convenience in my own script entered the record as a measurement (R61's m31
  counts were the first). New lane rule: a cell a script FILLS IN rather than
  LOOKS UP is printed as such, or not printed.
- My pre-registered P1 (phase saturation caps A_relax at 4) is REFUTED - it caps
  at 5. P6 (the triple inequality at 31->37) is REFUTED AS WRITTEN, because I
  predicted against a quantifier the manager's own message had already drawn.
  P7 refuted too (the maximisers ARE realised windows; at depth 3 the relaxation
  costs nothing).
- I PIPED A LONG BACKGROUND RUN THROUGH `tail` AGAIN - the exact mistake in the
  round-26 negatives, same lane, one round later.
- increment_law.py's own verdict line calls the m41 SUPERSET row a failure. It
  is not: 144 is a superset maximum and R72 measured that superset as heavily
  inflated at arity >= 2. Flagged, not silently patched; the corrected reading
  is the CRT sweep.

FOR OTHER LANES:
- MANAGER: the depth-3 half of your increment law is a CONSTANT against a LINEAR
  function (Delta_3 <= 4 at all eight steps), the free flank reduction already
  discharges 6 of 8, and the literal/padded separation makes the 31->37 exception
  a statement about the padded middle only - all three are in your derivation's
  favour. The per-J family is NOT uniform in the same way: Q*_J - F_2 exceeds
  s_min at 31->37 for J = 3 AND J = 4, always via the padded letter.
- MECHANIC: your phase-saturation theorem is EXACTLY the corridor condition mod
  35 (CRT), which explains both why it fires only at gears 5, 7, 11 and why it
  adds nothing to a corridor-built abstraction; and its alternation ceiling is
  litcap's own arithmetic under a different quantifier. Also: the exact m41
  4-tuple census is the only thing that moves rung nine, measured.
- FORMALIST: A_relax(M) <= 5 is a finite, per-class, integer statement (48
  classes mod 210, each a small phase-saturation check) - kernel-checkable in the
  same shape as LiteralCapTable.lean, and it would be the first UNIFORM order
  statement in the Lean corpus. Do NOT lift it to "A_m is nilpotent for m >= 6":
  that is false in general (see the padded-cycle refutation above).
- LATERAL: your reversal theorem halves every sweep in triple_41.py, and the
  round used it as a matter of course.

## Formalist round 27

GATES, all re-run at round close from clean invocations, all GREEN:
  cd proofs && lake build            -> **Build completed successfully (1521 jobs)**
                                        (1426 -> 1521; 45 new modules: 40 case
                                        transcriptions, 2 gear bases, 2 rung
                                        roots, 1 shared soundness module)
  cd proofs && lake env lean AxiomCheck.lean
                                     -> 367 footprints, ONLY [propext,
                                        Classical.choice, Quot.sound]; zero custom
                                        axioms, no native_decide, no ofReduceBool;
                                        `CaseSplit.lowest6/7`, `degpos6/7`,
                                        `ind_low2`, `ind_nonneg` depend on NO
                                        AXIOMS AT ALL
  .venv/Scripts/python.exe research/lp_cert_lean.py GATE
                                     -> ALL ASSERTIONS PASSED
Zero sorries. Every job this round launched has finished; nothing left running.

THE HEADLINE: **THE LP THREAD'S CASE-SPLIT CERTIFICATES ARE IN THE KERNEL, AND
`D_29_31` NOW EXISTS IN A FORM WITH NO CENSUS HYPOTHESIS AT ALL.**

    theorem CaseCert23.D_19_23_case (n : ℕ) : Machine23.g23 n ≤ 25 + 23
    theorem CaseCert31.D_29_31_case (n : ℕ) : Machine31.g31 n ≤ 43 + 31

Both hypothesis-free: no period, no census, no `native_decide`, standard-three
axiom footprint. The second is the point of the exercise - `Machine31.D_29_31`
consumes `Machine29.Census29`, a full-period claim about 214,708,725 openings
that no kernel has seen or will see (verdicts 21/25, the ledger's standing
residue). The same rung now has a second proof that consumes NOTHING but the
primes 5..31 and 35 case certificates.

R26.8's fixed order of attack was followed - (i) one case, (ii) exhaustiveness
-> the 19->23 rung, (iii) 29->31 - and step (iii) was REACHED, not stopped at
(ii). New files: `proofs/CaseSplit.lean`, `CaseCert23B/C0..C4/CaseCert23.lean`,
`CaseCert31B/C0..C34/CaseCert31.lean`; generator `research/gen_case_lean.py`;
transcription + gate `research/lp_cert_lean.py`.

MY OWN R26.8 SIZING WAS WRONG IN AN INSTRUCTIVE WAY - THREE OF THE FOUR NAMED
SOUNDNESS LEMMAS WERE VACUOUS AT THE ARTEFACTS, AND THE REAL WORK WAS NOT ON
THE LIST.
- `pos` restricted: trivial (`pos` is a literal list; one `decide` per case).
- `dom(q)` restricted: VACUOUS - domains only shrink when OPEN positions are
  prescribed, and a case split prescribes none.
- Cut validity: VACUOUS at every rung on disk. **EVERY CUT ROW OF ALL 75
  CERTIFIED CASES OF 19->23, 23->29 AND 29->31 IS THE BASE CUT** - the
  separation loop never fired at these widths - so the coverage row is
  literally the hypothesis "some gear blocks this position", and `lam_0 = 0`.
  (The LP thread found this independently and posted it mid-round; two
  codebases, same finding, same round.)
- Case exhaustiveness, which I called "the one with no analogue in the existing
  files": ONE `omega`.
THE ACTUAL WORK WAS THE RECURSION ROW, whose coefficients `n_ab` are defined in
the LP code as a MAX-COVER over the lower gears' phases - 8.2 million
evaluations to certify literally at 19->23. It has a clean lemma:

    THE LOWEST-BLOCKER IDENTITY (CaseSplit.lowest6 / lowest7, NO AXIOMS)
    if some gear blocks x, then
      1 + #{(a,b) : a<b, both block x, no gear below a blocks x}
        =  #{a : a blocks x}
    - only the LOWEST blocker can be the `a` of such a pair, and it pairs with
    each of the other blockers exactly once.

Summed over the position set that is `sum_a |A_a| >= |pos| + sum n_ab` for ANY
`n_ab` bounded by the "a lowest, b also blocks" count - which is what the
vehicle's `n_ab` is. In Lean it is a `decide` over 2^m Boolean assignments.

A KERNEL-SIZING FACT THAT MADE THE 35-CASE RUNG AFFORDABLE, and it is a fact
about the VEHICLE: **`n_ab = 0` for 96.4% of the gear-index-1 columns at
29->31** (52,173 of 54,145 over the 35 cases) - one gear below suffices to
cover the whole two-gear overlap - and `n_ab = 0` is sound with NO evaluation.
Listing the exceptions cut the per-case kernel check from 9m01 to 4m10 solo;
the whole 35-case rung then built in 47 MINUTES WALL at two concurrent workers.
Structurally the recursion row is, numerically, almost entirely a KOUNIAS ROW
AT THE SMALLEST FREE GEAR. General kernel tactic, filed as verdict 28: look
for a certificate's SUPPORT before formalising its DEFINITION.

THE EMISSION - I BUILT MY OWN AND IT AGREES WITH THEIRS. The LP thread's JSON
was not on disk at my first look, so per my brief's contingency I wrote
`research/lp_cert_lean.py`, which does more than transcribe: it rebuilds the
relaxation FROM THE PRIMES, asserts every row equals `base_cut`, recomputes
`n_ab` from the kernel-cheap closed form and asserts it equal to
`RelaxStar.frow` COLUMN BY COLUMN (3,381 columns/case at 19->23, 7,201 at
29->31), recomputes `lhs`/`rhs` from its own formulas in exact scaled integers
and asserts agreement with the recorded verdicts, and runs a random-tuple
SOUNDNESS GATE on the recursion row. Their emission then landed mid-round,
matching my R26.8 spec exactly; CROSS-CHECKED: their `cert_19_23_h*.json` and
my independent transcription agree as exact rationals on `pos`, `y`, `nu`,
`yff`, `lhs`, `rhs` for all five cases. This lane's JSON (`cert_19_23.json`,
`cert_23_29.json`, `cert_29_31.json`) also covers 23->29, which theirs does not.

ITEM 1 - THE MIRROR'S COUNTING HALF, round 26's named gap, CLOSED
(`proofs/Mirror.lean`; the standard three - `Classical.choice` enters through the
`Finset` machinery, where round 26's arithmetic halves needed only
`[propext, Quot.sound]`):
  `even_card_involution` - a fixed-point-free involution of a `Finset` gives
    even cardinality (structural induction; remove `a` and `f a`).
  `window_count_even` - EVERY window length occurs an EVEN number of times
    except the one the self-mirror window carries.
  `adjacent_equal_even` / `none_of_at_most_one` - the endpoint lever verbatim:
    an adjacent EQUAL pair `(F,F)` occurs an even number of times, so a
    counting bound of "at most one" proves "none".
HONEST SCOPE: what is kernel-checked is the LEVER over an ABSTRACT index
involution. INSTANTIATING it at a machine - proving the depth-j window family
is mirror-equivariant with the length invariant - needs `mirror_exposed29`
composed with the opening ENUMERATION (`Periodic.lean`), and that composition
is NOT built. Named as round-28 item 0, not claimed.

ITEM 2 - RUNG EIGHT (37->41): NOT ATTEMPTED, PRECONDITION ABSENT, and this is
a missing input rather than a judgment. My brief made it conditional on
Constructor's emission landing in the `hE` shape; their round-27 block is filed
and carries no machine-37 qualifying dictionary (depths 2..7, floor 14) and no
`qual_dict.py`-format emission of the 12,587 deletions. The ask is unchanged
and cheap for them: emit those windows and the rung is a transcription.

A TOOLING TRAP WORTH THE OTHER LANES KNOWING (verdict 26): `~/.elan/bin/lake.exe`
is an ELAN PROXY that reads `lean-toolchain` from the CURRENT WORKING DIRECTORY,
not from `-d`/`--dir`. Invoked from the repo root it silently picks elan's
default toolchain (4.33.1 here, against the project's 4.34.0-rc1) and starts
REBUILDING MATHLIB FROM SOURCE, failing on version-skewed `Batteries` lemmas.
The symptom is indistinguishable from a corrupt cache and is neither. Any agent
shell that resets cwd between calls must chain `cd proofs && lake ...`.

A SECOND PROCESS FINDING, MINE, AND IT IS A CORRECTNESS ONE (verdict 30):
STOPPING A BACKGROUND TASK DID NOT STOP THE SCRIPT IT LAUNCHED. I stopped a
3-wide build driver (the tool reported success), rewrote it 2-wide and
relaunched; the process list later showed FOUR copies alive, interleaving into
one log and racing on the same modules. A `[ ! -f x.olean ]` resume guard does
NOT stop two builders starting the same module at once, lake does not lock a
module against another lake process, and `.olean`s are TRUSTED on load rather
than re-checked - so a raced file is precisely what this lane must not build a
claim on. Response: killed every driver and worker, VERIFIED FROM THE PROCESS
LIST rather than from the tool's success message, DELETED all 35 case oleans
and rebuilt from clean under a single driver (~20 min of work discarded).
Standing rules: confirm a stopped build is really gone from the process list;
never run two builders over one build tree; if it happens anyway, delete the
artefacts.

FOR OTHER LANES:
- LP THREAD: your emission is exactly right and the base-cut fact shrinks the
  Lean obligation as you said. What the kernel needs NEXT, if you want 31->37
  kernel-checked, is the SMALLEST k that certifies it - at the measured 1.34
  min/case throughput (two workers), 35 cases is ~45 min and 385 cases is
  ~8.6 h - so k = 2 is a comfortable round and k = 3 is the whole round. Also: the exception-list fact
  above (96.4% of gear-index-1 `n_ab` are zero) is a statement about your
  vehicle, not about Lean, and may be worth a line in your own cost law.
- CONSTRUCTOR: `A_relax(M) <= 5` as 48 classes mod 210 is ACCEPTED as a
  round-28 target - it is exactly the `LiteralCapTable.lean` shape and would be
  the first UNIFORM order statement in the Lean corpus; your warning against
  lifting it to "A_m nilpotent for m >= 6" is recorded with it. The m37
  qualifying dictionary remains the only missing input to rung eight here.
- MANAGER: the ladder's hypothesis inventory changed. 29->31 no longer NEEDS
  `Census29`; the census hypotheses now bind only at 31->37 and above.

## Mechanic round 27

GATES (all re-run from clean processes at round close, all GREEN):
  .venv/Scripts/python.exe research/m41_census_r27.py gate      -> ALL ASSERTIONS PASSED
  .venv-sat/Scripts/python.exe research/akill_verify_r27.py     -> ALL ASSERTIONS PASSED
  .venv-sat/Scripts/python.exe research/akill_bands_r27.py      -> 6 tilings re-asserted
  .venv-sat/Scripts/python.exe research/akill_verify_r26.py     -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/ghist_gate_r26.py <7 csv>   -> ALL ASSERTIONS PASSED
  .venv/Scripts/python.exe research/cyclic_close_r25.py check   -> ALL ASSERTIONS PASSED
  .venv-sat/Scripts/python.exe research/akill_verify_r25.py --nosat -> ALL ASSERTIONS PASSED
Pre-registration written before any solver call:
research/data/r27/prereg_mechanic_r27.md - EIGHT predictions, THREE REFUTED by
my own runs (scored below).

HEADLINE: (D) AT 53 -> 59 IS DECIDED TRUE - the first step of the ladder with no
upper bound on the new machine's F anywhere in the project or in the published
corpus.  The corpus F(2,y) ladder stops at y = 53; round 26 could only say
F(59) >= F_2(53) = 159 with "at most 45 of room".  Round 27 computes the other
side, on MACHINE 23's period (37,182,145 slots) for a property of MACHINE 59
(period 1.96e19, ratio 5.3e11), through Constructor's round-26 attainment
theorem F(M+q') = max_J Q*_J(M; legal for q'):

    max_J Q*_J(53; legal for 59) <= 203  <  204 = F(53) + 59     ==>  (D) HOLDS
    and, after four empty bands and one exhibited window,  161 <= F(59) <= 178.

AND THE BRACKET DECIDES THE MANAGER'S INCREMENT LAW AT A STEP IT HAD NEVER
SEEN.  F(M+q') - F_2(M) <= s_min(q') = min(2u', q'-2u') gives F(59) <= 179 at
53 -> 59 (s_min = 20, F_2(53) = 159).  MEASURED F(59) <= 178: THE LAW HOLDS,
increment in [2, 19] against a cap of 20.  Pre-registered as B2 before the scan
and it was a real bet - the law's one known failure is the PADDED step 31 -> 37,
and this step has padding (the lower-bound witness itself has a 2q' interior).

This is the record law's FIRST USE AS A COMPUTATIONAL INSTRUMENT where the
answer was not already known.  The lower bound 161 is up from 159 and comes with
a machine-53 witness re-verified from the definition:
k = 2,505,673,933,219,103,747, gaps [10, 118, 33], span 161, all 158 other slots
of the span blocked, middle gap 118 = 2q' a legal (padded) kill letter.

THE TECHNIQUE THAT MADE IT AFFORDABLE, and it generalises (new rule 31): a
DESCENDING LADDER OF BANDS.  A transfer run seeded at lo with span cap hi
decides exactly "the maximum in (lo, hi], or lo if empty", so bands with
hi_{i+1} = lo_i compose with no gap, each one finishes, each tightens the bound
the moment it lands, and the first non-empty band from the top IS the answer.
MEASURED: lowering the seed by ten costs about four times the run, while
lowering the cap makes the walk cheaper - so a single run seeded at the floor
pays for every band at once and reports nothing until it is done.  Also new
rule 32: a DEPTH cap is a cost control, not just a scope choice - the same band
at JMAX = 7 was projected at nine hours per worker and killed; at JMAX = 3,
which was all the question needed, it ran in 45 minutes.

AND THE BANDS PAID FOR A SECOND LANE ITEM.  A realised k-chain kill word is a
word-legal window of J = k-1 gaps, so an EMPTY band refutes every kill word
whose span lands in it, at every depth it covers, with no solver call.  Six
completed scans give a refuted-span band table (research/akill_bands_r27.py,
every tiling re-asserted), and it decided 63 of the A_kill(53->59) words
outright:

    A_kill(53 -> 59) = 4 EXACT, every level complete.
      N_3 = 8 of 36: (20,39) (20,59) (20,98) (20,118) and their four reverses -
        EVERY realised 3-chain carries the letter s = 20, paired with q'-s = 39,
        q' = 59, q'+(q'-s) = 98 or 2q' = 118.  Nothing pairs two non-s letters.
      N_4 = 1 of 18: the palindrome (20, 98, 20), witness
        5,179,823,167,446,585,215, re-verified from the definition.
      N_5 = 0, and the k=5 level is EMPTY BEFORE ANY DECISION - a 4-letter word
        needs both 3-letter sub-words realised and the only one is a palindrome
        whose two overlaps cannot both be it.  The overlap lemma closes it.
      THE WHOLE CAMPAIGN COST ONE UNSAT (2,666 s); 63 words fell to the band
      table, 7 to the screen, the rest to the mirror law.
      THE SHAPE IS THE PADDED ALTERNATION: the ceiling for the PURE alternation
      is 3 and (20,39,20) IS zero as C29's theorem says; what carries arity 4 is
      (20,98,20) = (s, q'+(q'-s), s).  A_kill - ceiling is now
      -2, +1, +1, +1, 0, +1 at 31->37 .. 53->59: at every step whose ceiling is
      2 or 3 the answer is ceiling+1 and the lifting word is PADDED.  NAMED NEXT
      CONSTRUCT: the phase-saturation ceiling of the PADDED alternation family,
      a closed form in the small gears exactly like C29's.

C30 priced this campaign at 27 solver calls; the k=3 level cost FOUR, all of
them satisfiable hits of 0-49 s, and ZERO UNSATs.  The cost of not doing it this
way was measured, not imagined: the first launch put three of those words on
pysat at fourteen gears and they were still running after two hours each -
every one of them refuted by a scan already on disk or costing 1,300-1,800 s.
THE SCOPE POINT THAT MAKES IT SOUND: a span cap conditions claims about spans
ABOVE it, never claims INSIDE the scanned interval.  "F_2(53) <= 159" is
cap-conditional; "no 2-window has span in (159, 200]" is not.  Only the second
kind is used.
NEW DATUM FROM THE SAME MACHINERY: machine 53's adjacent-pair span spectrum has
a SIX-WIDE HOLE at 153..158, immediately below its maximum F_2(53) = 159 (the
largest 2-window span at or below 158 is 152, from two range workers tiling the
period).  That hole is what killed the two words SAT could not.

BRIEF ITEM (a) - THE m41 EXACT 4-TUPLE CENSUS: PRICED FIRST, AND THE PRICE IS
THE FINDING.  Span-stratified timing over the round-26 screened superset
(1,407,543 reverse classes after mirror halving) gives >= 4,036,276 CORE-SECONDS
= 1,121 core-hours by the CRT-decision route, and >= 2e5 core-seconds by the
period route (round 26's ~85,000 s figure was for the HISTOGRAM, a strictly
easier object; a tuple pass needs a 6.2M-element scatter per lap on top, and the
route enumerates all prod(q-2) = 8.499e12 openings of m41's period).  BOTH
VEHICLES EXCEED THE ROUND BY ONE TO THREE ORDERS OF MAGNITUDE.  Delivered
instead, in three parts:

 (1) F_4(41) = 118 EXACT, a first computation (602 core-seconds, r = 4 floor-1
     transfer, cap 150 ABOVE the deletion-ladder cap 145 so the value is NOT
     span-capped).  The standing entry was "F_4(41) <= 145", which is the
     superset's own maximum span, i.e. nothing.  Three controls the tool was not
     told about: J = 3 returns 110 = F_3(41) seeded one below, AT ROUND 24's OWN
     SAT ADDRESS k = 30,382,499,692,410; the two workers' J = 4 maximisers are
     an exact MIRROR PAIR (4,834,947 + 32,347,080 = P(23) - 118) verified at
     machine 41 with gaps [51,2,50,15] and [15,50,2,51]; and that maximiser is
     present in the screened superset, so the cap is attained, i.e. tight.
     => A realised 4-tuple span is at most F_4(41) BY DEFINITION, so
     4,239,676 -> 2,814,574 (r26 phase saturation) -> 1,747,819:
     58.8% OF CONSTRUCTOR'S ARITY-4 SUPERSET REMOVED BY THEOREM ALONE, in
     seconds, no solver.  Induced 3-tuple dictionary 130,942 -> 95,331.
     research/data/r27/gap_tuples_41_4_screened_spancap.csv
 (2) THE EXACT SHARD, BY ASCENDING SPAN:
     research/data/r27/gap_tuples_41_4_exact_le77.csv - COMPLETE AT EVERY SPAN
     <= 77, 169,981 reverse classes decided, ZERO undecided, 338,855 tuples
     (178,886 classes decided in all, 868 refuted; two waves, the second
     resuming from the first's checkpoint and moving the frontier 75 -> 77).
     Reverse-closure of the emitted set asserted, containment in the superset
     asserted.  Checkpointed: each worker resumes from its own log with one
     command line.
 (3) THE DECIDER GATE, TWO-SIDED, at the three machines whose exact 4-tuple
     dictionary this lane scanned in full: m23 15,696/15,696 YES and 2,000/2,000
     NO, m29 3,000+3,000, m31 2,000+2,000 (90 s).  The negative controls are
     4-tuples built from each machine own realised gap VALUES but absent from
     its exact dictionary - Formalist's round-26 lesson applied in both
     directions.

AND A STRUCTURAL FINDING THE SHARD PRODUCED: THE INFLATION ONSET IS SHARP AND IT
IS AT SPAN 68.  Every one of the ~137,000 reverse classes of span <= 67 is
REALISED; the first refutation anywhere is at span 68, and the refuted count
climbs 2, 0, 6, 14, 26, 17, 68, 117, 117, 105 over spans 68..77, against 20/24 refuted at
span 81-100 and 24/24 at 101-140 in the timed sample.  So the dictionary
transfer (K4) is EXACT below 68 and collapses over the next ~30 units of span -
a fact about the ORDER-4 CLOSURE, not about the machine: a machine-37 walk short
enough to be pinned by its 4-windows is realised, and past ~68 the closure stops
determining it.

CROSS-LANE, FOR CONSTRUCTOR - THE m37 QUALIFYING SPECTRUM, AND YOUR LOST Q_7.
research/data/r27/qspec37_w{0,1}.log, two range workers tiling the period, both
reporting the same row, with witnesses re-verified at machine 37:

    J             2    3    4    5    6    7    8
    Q_J(37; 14)  90   97  103  110  112  114  112     budget F(37) + 41 = 129

J = 2..7 REPRODUCES C20's round-23 row exactly, four rounds later, by an
independent run; J = 8 is new AND IT TURNS OVER (114 -> 112), the second machine
at which the qualifying spectrum is non-monotone in depth (Formalist found it at
m31).  Your round-26 row Q_2..Q_6 = 88, 90, 97, 103, 110 matches mine SHIFTED BY
ONE (your Q_3..Q_6 = my Q_2..Q_5), which reads as your J counting OPENINGS where
mine counts GAPS.  On that reading YOUR MISSING Q_7 IS 112, far under your layer
bound 174 and under budget.  I am not adjudicating your index convention - the
row above is in mine, with witnesses, so it can be re-indexed unambiguously.
(Rule 5 exists because this project has been bitten by exactly this before.)

HONEST NEGATIVES AND COSTS OF THE ROUND:
- THE m41 EXACT CENSUS IS NOT DELIVERED, and after pricing it is a MULTI-ROUND
  object rather than a next-round one.  What would change that is a cheaper
  decision vehicle, not more cores: at 3-4 s per refutation the remaining
  ~1,229,000 reverse classes are ~1,100 core-hours.
- F(59) IS BRACKETED, NOT PINNED: 161 <= F(59) <= 178.  The remaining spans are
  (161, 178] and, by the depth-3 band, only at depths J >= 4.  Each band gets
  dearer as it descends (1,038 / 3,256 / 4,930 s x 7 workers for 11, 11 and 6
  units of width), and a single wide band (161, 184] at JMAX = 7 was launched
  and KILLED at a four-hour projection.
- F_2(59) NOT COMPUTED (brief item c, second half).  It wants F(59) as a seed
  and the round spent its cores getting the first side of that.
- I MIS-SIZED A BAND AND KILLED IT: the word-legal band (152, 158] at JMAX = 7
  was under 8% after 40 minutes (a nine-hour projection) because seeding six
  units below a low span expands an enormous number of windows.  Re-run at
  JMAX = 3 - all the question needed - it finished in 45 minutes.  Rule 32.
- I READ A HEALTHY JOB AS A STALLED ONE TWICE, because j5_multi.py's progress
  stride is a GLOBAL constant (198,804 start indices) rather than a share of the
  worker's own range, so a worker can run an hour before its first line.  This
  is rule 28 recurring inside a tool rather than an orchestrator; recorded as
  rule 33.
- I RESTARTED THE A_kill CAMPAIGN FOUR TIMES as each new scan landed.  The
  driver's resume-from-own-log made that free, but the right shape would have
  been to compute the band table FIRST and hand the solver only what no scan
  owns.
- MY OWN PRE-REGISTRATION WENT 3 REFUTED / 4 CONFIRMED / 1 UNRESOLVED:
    A1 "A_kill(53->59) = 3"            REFUTED - (20,98,20) is realised at k=4.
       I extrapolated a per-step pattern from three steps instead of computing
       it: standing rule 1 in a new costume.  C29's alternation ceiling itself
       is untouched - the pure alternation (20,39,20) IS zero, as the theorem
       says; what lifts A_kill past it is padding.  The value is 4, EXACTLY
       ONE above the ceiling - which is what four of the six measured steps do.
    A2 "(59,59) is realised"           REFUTED - it is ZERO, by the phase-
       saturation screen with no solver call.  The run of DOUBLE-PADDED 3-chains
       that held at 41->43, 43->47 and 47->53 ENDS at 53->59.
    C1 "the superset is exact at span <= 80"  REFUTED - the onset is at 68.
    A3 (the nine k=3 words above the F_2(53) cap are all zero - an independent
       SAT-free cross-check of C30's upper direction) CONFIRMED.
    B1 ((D) holds at 53->59) CONFIRMED.  B3 (F(59) > 159 strictly) CONFIRMED
       (>= 161).  C3 (F_4(41) in [113,125]) CONFIRMED (118).
    B2 (the manager's INCREMENT LAW at this step: F(59) <= F_2(53) + s_min(59)
       = 179) CONFIRMED - band (178, 184] came back EMPTY, so F(59) <= 178.
       This is the round's one prediction that was a bet rather than a
       corollary.
    B4 (the attaining depth J* = k_win + 1, predicted 3 or 4) UNRESOLVED while
       F(59) is a bracket; the deepest word-legal window found is at J = 3.

FOR OTHER LANES:
- CONSTRUCTOR: your rung-nine oracle input is 58.8% smaller by theorem alone
  (gap_tuples_41_4_screened_spancap.csv), F_4(41) = 118 is a new exact input,
  and the exact shard at span <= 77 is 338,855 tuples with zero inflation
  below span 68.  Your Q_7(37;14) is 112 on the index reading above.
- MANAGER: your increment law is CONFIRMED at 53 -> 59, a step it had never
  seen - F(59) <= 178 against the law's 179, with the increment in [2, 19]
  against a cap of 20.  It was pre-registered before the scan.
- FORMALIST: F_4(41) = 118, F(59) <= 203 and the m37 row Q_2..Q_8 are finite
  integers with exhibited witnesses; the A_kill(53->59) k=3 level is complete
  and its refutations are BAND MEMBERSHIPS, i.e. arithmetic, not solver runs.

# agents-shared.md - findings exchange for the proof-search team

## SUMMARY (manager-rewritten each round - read this first; details below and in workstream docs)

State after round 19 - the directive worked. Two workstreams independently converted "arithmetic
luck" into measured structure, the route's requirement was repaired rather than patched, and the
deep cases turn out to be the EASY ones - reversing what the route assumed from round 8 to 17.

THE SUPPRESSION LAW (constructor, from a NEW OBJECT - the window composition profile: a j-window
as one thing carrying its composition, its sum, and the qualifying status of its interiors):
"luck" splits into two questions that answer differently. (1) THE MAXIMUM IS LUCKY - the
exclusion zone is tiny (8-60 windows), luck probability 10^-0.1 to 10^-1.3; given p, the
qualifying max sits where a random p-sample's max would. (2) BUT p ITSELF IS NOT. Against
independence p_1^(j-2), measured p_j shows deficits of x26, x6.7 and x1400 - qualifying interiors
are STRONGLY NEGATIVELY CORRELATED. That is Wall V's non-clustering statement appearing as a
measured correlation deficit in a built object, not as an assumed need. The structure is in p,
not in the order statistic.
SUPPRESSION LAW: suppression(j) = F_j - qualmax_j ~ lambda*(j-2)*ln(1/p_1), with lambda and p_1
computed from M alone. Observed 7, 15, 30 vs predicted 9.0, 21.7, 42.5 - right scale,
conservative at depth. PAR TRADING IS NOW DERIVED, NOT OBSERVED: gain per link = spectrum
increment (5-15), loss per link = lambda*L (4.2, 5.5, 9.0) - approximately equal.
THE PAYOFF - SUPPRESSION-CORRECTED FLATNESS: (D) follows from F_j - F <= q' + lambda*(j-2)*L for
every j. ALL 15 MACHINE-DEPTH PAIRS HOLD (corrected values 4.7-15.1, bounded and non-growing)
where raw flatness fails at 5 of 15. This REPAIRS round 17's refutation rather than patching it,
SUBSUMES round 18's two-part target (no winning-depth assumption needed), and shows lemma 1 is
just the j=2 case - THE DEEP CASES ARE THE EASIER ONES.

THE MECHANISM EXHIBITED, NOT ARGUED (mechanic, independently): of 132 windows attaining F_j at
machines 19/23/29, ZERO are literal and ZERO are qualifying. The shape is always two near-maximal
flanks plus the machine's SMALLEST gaps inside (interiors 4, 3, 30 at machine 29's F_5 = 85,
k = 772,741,833) - and the interior-gap floor 2u' forbids exactly that shape. So the
extremal windows are structurally the wrong shape to qualify, which is the same fact
Constructor's correlation deficit measures from the other side.
THE QUALIFYING SPECTRUM Q_j closes every measured step, WORD-FREE: at 29->31, F_5 = F+42 fails
but Q_5 = F+28 passes - the size threshold alone takes 42 to 28, a direct answer to the
"arithmetic luck" caveat. The same object gives the fuel cap free (Q_j = 0 iff no word that
long). HONEST COUNTERWEIGHT: the margin collapses from ~0.45q' to 0.10-0.11q' at machines 29 and
31 - the criterion is running out of room exactly where the machines get big.
THE HUNTED VIOLATION, FOUND: the monotone envelope holds within every step (19/19) but is FALSE
as a machine law - machine 29, span 21 -> max flank 27 (205,068 occurrences) vs span 25 -> max
flank 30 (88,548 occurrences, k = 133,490,560). THE ENVELOPE FOLLOWS OCCURRENCE COUNT, NOT SPAN.
PAR TRADING CONFIRMED independently (spreads 8.8%, 9.3%, 5.2%), k_win = 3, 2, 2; at 29->31 the
k=3 and k=4 chains tie at 55 while k=2 wins at 58 - fuel exists and LOSES by 3.
THE HOLE STRUCTURE (new measurement, from the directive): first enumeration - holes {9}, {17},
{19,24}, {24}, {41,42} at machines 13-29. ABSENCE IS TRANSIENT (5 of 6 heal at the next gear;
only v = 24 survives a step). hist_M[v] has a stable, converging residue law whose richest
classes are +-s of gears 5 and 7 - THE CORRIDOR TEETH ARE LEGIBLE IN THE WHOLE MACHINE'S GAP
HISTOGRAM. But the residue law does NOT predict the holes. Named construct for next round: the
COVERABILITY SPECTRUM COV(M) - CRT arithmetic, no period scan, reaches machines 37/41/43/53, and
yields the UPPER bounds on F_j that every prefix row currently lacks.

THE EXPOSED-SET AUTOCORRELATION (lateral, back in its own lane, new object): for gear q with
exposed set A_q = Z_q minus its two teeth, c_q(g) = |{r in A_q : r+g in A_q}|. CLOSED FORM,
brute-force verified over gears 5-31 at all lags, zero mismatches: c_q(g) = q-2 if q | g (same
tooth), q-3 if g = +-2u_q (opposite teeth - EXACTLY THE LITERAL-LINK LAG that rounds 12-17 spent
on padding, reached here from an unrelated direction), q-4 otherwise. THE THREE CASES OF THE
AUTOCORRELATION ARE THE THREE TOOTH-RELATIONSHIPS.
IT PARTLY DISSOLVES "NO SMOOTH LAW, ONLY THE HISTOGRAM": admissible endpoint phases mod 35 =
exactly c_5(g)*c_7(g) in {3..15}. Machine 23: gap 24 (absent at machines 19 AND 23) and gap 29
(count 6, between neighbours 322 and 112) both carry the MINIMUM possible value 3; three of four
absent-below-F values carry the minimum. Adding log(c_5 c_7) to a smooth-decay fit removes 28%
(machine 23) and 24% (machine 29) of residual variance - a real law, multiplicative and
arithmetic, not smooth.
CLEAN NEGATIVE: residual demand vs purchasable supply leaves slack 8-16 at every g, so gap 24's
absence is selection plus rarity, NOT a covering obstruction - don't hunt one.
SIDE THEOREM: an AP of L openings has difference divisible by every gear q < L+2 (3 equal gaps
need 5|g, 5 need 35|g, L >= y+2 needs >= P(y)). Full-period verified on machines 13-29, zero
violations, longest equal-gap run 3-4 with g = 5 exactly every time.
WHY IT STOPPED THERE: endpoint exposure is a CONJUNCTION so it factorises by CRT - that is why
c_q(g) has a closed form. The interior condition is a DISJUNCTION and does not factorise. Named
next constructs: c_q(g1,g2) (gear x lag-pair - the natural object for (D), since a flank sum IS a
two-lag quantity) and the autocorrelation at the padded lag q'.

FORMAL LEDGER (26 libs, 1254 jobs, zero sorries, zero warnings): PolignacCap confirmed - all
eight cap_gcd_* AND capOf_le_twelve depend on NO AXIOMS AT ALL; |E_e| matches the HL product
prod(q - r_q) for all eight classes. THE BRIDGE IDENTITY LANDED (proofs/Spectrum.lean): merged_eq
(a word occupying l consecutive gaps plus its two flanks spans exactly l+2 = k+1 CONSECUTIVE
gaps, so merged length IS a window sum), merged_le_spectrum, and merged_le_of_shallow deriving
(D) at alpha=3 from k_win <= 3 and F_4 <= F + q' - ITS STATEMENT MENTIONS NO FUEL, NO k_max, NO
WORDS, NO RESIDUES, NO PADDING, only g : N -> N. Both empirical halves remain hypotheses inside
the file, so the censuses decide the conclusion without the formal step being at risk.
AUDIT (A)/(B)/(C)/(E): (B) fully checked and now universal; (E) checked, off-target; (A) partial
- class-reduction core checked, list ENUMERATION only computed (the remaining gap); (C) had an
unchecked half, now CLOSED: onset_gate (0 < g) (q | g) (g <= F) : q <= F, one line, [propext]
only. PADDING RESTATED: the count bound is documented as p <= F/q + 5/6 (it GROWS);
padding_three_not_excluded records that F >= (13/6)q stops excluding three links; renamed
padding_at_most_one_below_onset. The heading and docstring had overclaimed and wrongly called
F < q "the onset condition" when by onset_gate it is precisely the NO-PADDING regime.
PER THE DIRECTIVE, the tier-C wall was re-attacked rather than restated: round 15's "caps at
machine 19" was an ARTEFACT OF THE OLD ENCODING. Allocation-free scan + restricting starts to
openings (density 0.234, a 4.3x cut) + tight fuel takes machine 19 from 86 min to ~20 min, and
machine 23 from 33 h to ~7 h. If a machine is truly out of reach the named construct is the
single-cycle reduction, whose prerequisite exists_mul_mod_eq is now proved - off the shelf.

HARVESTER, BACK ON MANDATE - THE PAIRED JACOBSTHAL FAMILY MEASURED: first exact values of h_2
(Ziller-Morack compute none), exhaustive over every even difference: 18, 30, 66, 150, 192 at
y = 5, 7, 11, 13, 17 against the Conjecture 6 bound p_n^2 - p_n = 20, 42, 110, 156, 272 - HOLDS
at all five, but THE MARGIN IS NON-MONOTONE WITH A ONE-OFF DIP AT 13 (3.8% vs 10.0, 28.6, 40.0,
29.4). Conjecture 6 is not approached steadily from below but in a single outlying case, making
"why is 13 extremal?" a sharp attackable question. A FAILED PREDICTION RECORDED: from the first
four points harvester predicted extrapolation (~330) would BREACH the bound at y = 17; it ran the
computation in the same round and was refuted (192, holds).
TWINS SIT NEAR THE EASY END OF THEIR OWN FAMILY: at gears <= 13, among the 2,880 differences
coprime to P, F ranges 30..75 and the twin difference gives 33 - the 13.3rd PERCENTILE, with
77.2% of coprime differences having a LARGER maximal gap and the extremal one 2.27x the twin
value (1.78x at gears <= 17). Since the twin case of Conjecture 6 IS Reduction A, "prove it for
twins" is strictly the easy end of "prove it for every even difference" - by a factor > 2, not a
constant.
DENSITY DOES NOT DETERMINE THE EXTREME: F_max/lambda ranges 2.88-7.52 across the 31 gcd classes;
two difference classes with the same mean gap differ by > 2x in maximal gap. The d-dependence is
genuine second-order structure the density heuristic misses.

SEARCHES: a process sweep killed the entire detached set mid-round; NO FINDINGS LOST (all
chunk-flushed), coverage was. All relaunched: satruns_L15 resumed at 69.8%, padding37, hist37,
fuel37_k5hunt, plus envelope31/37/41, kwin31/37/41, qspec37. Pruned F(2,53) at >= 426.
NEW TOOL OUTSIDE THE SEARCH: rust3/gearsuite - a slot-frame prime and twin-gap suite built only
from proven laws (tooth law, slot cap, onset, horizon, corridor-as-wheel, merge law), 24 tests
green against outside values (pi(10^8), the F ladder, the r13 spectra, maximal gap 132 after
1357201), 94M slots/s, and f_next computing F(M+q') from the old machine alone.

State after round 10 (carry-over) - the tolerance route reduced to two named statements, the
adjacency question answered NO, and the T1 reopening closed with an exact self-reference law.

THE UNIFICATION (constructor): both tolerance lemmas are now ONE structure. With F_j = max sum
of j consecutive gaps (the gap spectrum), rigorously excess <= F_{k_max+1} - F2, and lemma 1 is
the first spectrum increment. The whole tolerance hypothesis = SPECTRUM FLATNESS (increments are
q/3-scale, not F-scale) + FUEL BOUND (k_max = o(ln y) suffices; measured k_max <= 3 everywhere,
62 k=3 chains matching the corpus fuel census exactly). Fuel is LOCAL - genuinely
corridor-approachable; flatness inherits the escape-distance obstruction (Wall V).

ADJACENCY: NO (constructor, answering lateral's target): at y = 13/17/19/23 two maximal gaps
can never be adjacent - certified by class arithmetic + one period scan. Per-machine alpha1
closes with a three-tier check (A3 machine-free / mod-385 strata disjointness / direct), written
out at y=13. Honest limit: the tier-C residual grows (4 at y=13 -> 96 at y=23), so scale needs
mod-5005; uniformity in y still open.

THE PINNING LAW (lateral): the neighbourhood word pins the mod-385 address to <= 4 offsets,
UNIFORMLY in y (206/206 words, five machines; gear 5 unique always). #top-stratum classes <= 4
x #words; observed 6-14 classes, flat, while gap counts swing 20-106. Drift recursion REFUTED
(reachability 18/20 -> 0/4): the address is local - address = pin(word), not inherited.
Machine-independent alpha1 now needs exactly one open piece: UNIFORMITY OF THE NEAR-TOP WORD
GRAMMAR (is the word-shape family finite a priori from flank alphabet {1..5} + chain skeleton +
pinning?).

T1 REOPENING CLOSED (mechanic): the exact content is two laws - (trivial) thickness T is
monotone in the gap so g=2 bands are thinnest, T = 4m at a twin; (real) every twin dead-centers
the thinnest band above it: its product slot k = 6m^2 sits at offset 2m = T/2 exactly, one dead
slot per band, 1223/1223 verified. Everything else is density artifact (9,591 bands to 10^10:
decade-matched g=2/all ratios 0.984-1.018; zero twin-empty bands; min primes/band = 6). The
descent's binding case binds by length alone - the imported Legendre-class problem, no added
machine hostility.

FORMAL LEDGER (green, 992 jobs, 10 targets): Corridor.lean extended - endpoint_law,
endpoint_law_34 (G = 34 mod 35 forces a mod 35 in {3,18,33}), adjacency_law +
forbidden_pairs_count (= 294, full 35x35 table by decide +kernel, no native_decide),
no_chain_of_forbidden, n2_packing (W/33 <= n2, choice flagged removable). Harvester: the
assembly line CLOSED - card_triple_inter_eq (8 CRT side classes) + three_gear_master (26
filter-card terms, subtraction-free, any distinct odd primes, any prefix) - the formal master
formula for 3 gears end to end. Polignac.lean = 44 theorems. Proof note for mod-105/385
attempts: one-shot omega dies at 5 dvd atoms; use per-gear iffs + interval_cases (formalist).

SEARCHES: F(2,53) log still header-only (>= 420 standing; needs <= 486). Harvester's
assessment: a pruned restart (endpoint-law filter, 2-5x, resume support) beats continuing
unpruned - implementation authorized round 11. L=15 hunt running detached (satruns_L15.log,
~15h to 1.2e13, chunk-flushed).

CARRY-OVER FACTS (round 9, still load-bearing): escape distance = 1 - bounded-modulus corridors
constrain position, never magnitude (Wall V for global flatness). First L=14 at k =
46,133,660,494 (member 2.768e11), HL model validated; ladder L=15 ~ 5e12, L=32 ~ 3e42.
prime_adjacent_run_le (32-cap) on [propext, Quot.sound]. Alpha1 empirics 0.52-1.16, no trend.

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

ROUND-20 (each in its own lane, per the mandate rule):
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

# merge-law-h2-test - can the merge law compute h_2 incrementally, matching Ziller-Morack?

Standalone computational test, 2026-08-23. Code: `rust3/src/bin/h2ladder.rs` (heavy exact
runs, asserts every matched value, 24 pre-existing gearsuite tests untouched and green) and
`research/merge_h2_ladder.py` (frame mapping, scanner battery, small ladders, op-count
closed forms, asserts). Reference values: Ziller-Morack arXiv:1706.03668 Table 1 (exact
h_2 for all p_n <= 73, transcribed in `docs/novel/paired-jacobsthal-values.md` section 6)
and the project's own fixed-twin ladder F(2,y) (`docs/gear-recursion.md` section 2).

**Answer in one line: the merge law is exact everywhere it was run - it replicates
Ziller-Morack's h_2 at 19 and 23 (the first in-project exact values there) and the twin
ladder through F(2,41) = 273 without ever constructing a new period, at 18x to 960x
fewer elementary operations than construction - but the h_2 ladder cannot be ridden to
p_n = 73, because h_2's maximiser wanders through the middle of the difference family, so
exactness needs every class, and the family count grows by a factor q' per rung.**

## 1. The frame mapping (extracted from the scripts that reproduced 18..192)

The task premise "h_2's object is pairs (m, m+2)" is **wrong**, and the correction is the
first result of the test. From `research/jacobsthal_family.py` / `jacobsthal_h2_17.py`
(the scripts whose output 18, 30, 66, 150, 192 matches ZM Table 1 verbatim):

- **ZM frame**: integers, modulus p_n# (gears 2 and 3 included), paired progressions at
  EVERY even difference 2e'; h_2(n) = j_2(p_n#).
- **Halved frame** (the project's computation frame): m = 2n+1 absorbs gear 2, factor 2 on
  all lengths. For reduced difference e, gear q in {3, 5, ..., y} blocks position n = 0
  and n = -e (mod q). F_e(y) = max cyclic gap of the survivors, and

      h_2 = 2 * max over ALL e of F_e(y).

- **Twin slot frame** (this project's machine): the e = 1 class only. Gear 3 confines the
  e = 1 survivors to one class mod 3, so F_1 = F_adjacent = 3 * F_slot, where slot
  k = (6k-1, 6k+1), gears 5..y, gear q blocks k = +-u (mod q), 6u = 1 (mod q). The
  corpus's F(2,y) is F_adjacent.

So a twin-only merge ladder computes F(2,y), **not** h_2: at y = 13, 2*F_1 = 66 while
h_2 = 150 (the twin class sits at the 13th-21st percentile of its family - harvester r17).
Verified by exhaustive scans at y = 5, 7, 11, 13 (h_2 = 18, 30, 66, 150, all ZM MATCH) in
`merge_h2_ladder.py` part 1. The one coincidence: at y = 7 the twin class is a maximiser
(2*F_1 = 30 = h_2). Computing h_2 by merge therefore means merging **every** difference
class of the old level, then maximising - which is what was implemented.

One scanner serves both frames. Adding a coprime gear q' deletes, per lap, the openings
whose positions lie in a two-element residue set {c, c+s} mod q'; the deleted pair shifts
by -P per lap and gcd(P, q') = 1 makes every c occur, so a run of consecutive openings is
deleted together in some lap iff its positions all lie in one such set (the chain
condition, exact). s is the tooth separation: **e mod q'** in the halved frame (teeth
{0, -e}; s = 0 when q' | e, the collapsed one-tooth case), **2u mod q'** in the slot frame
(teeth {u, -u}). Scanners s and q'-s accept exactly the same two-element sets, so
s = 0..floor(q'/2) covers every residue class of e mod q'. F(M+q') = max over surviving
old gaps and merged-run spans, read off the old word alone.

## 2. Validation of the implementation (before any claims)

- 668 (machine, gear, class) cases, both frames, including collapsed teeth (q | e, q' | e)
  and e = 0 mod P: merge == direct construction of the new period, **exact in all 668**
  (`merge_h2_ladder.py` part 2). The slot-frame cases reproduce the `gear-recursion.md`
  section 4a table (11+13 -> 11, 11+17 -> 16, 13+17 -> 18, 17+19 -> 25, 17+29 -> 26,
  19+23 -> 34, 19+31 -> 37).
- 200 random level-19 h_2 classes: merge (from the 17-word) == from-scratch sieve of
  P19 = 4,849,845, **200/200 exact** (`h2ladder sample19 200`).
- The h_2(19) and h_2(23) argmax classes were rebuilt by full direct construction
  (periods 4,849,845 and 111,546,435): F = 129 and 183 exactly as the merge predicted.
- Every streamed twin rung re-derives the previous level's F as the scanned stream's own
  max gap (a construction-equivalent check): 58 and 88 reconfirmed in-stream, and the
  stream opening counts matched the exact products A(31) = 6,226,553,025 and
  A(37) = 217,929,355,875.

## 2a. Metrics protocol

The primary cost metric is the **elementary-operation count**, machine-independent and
deterministic for both code paths (`h2ladder ops` prints the closed forms and verifies
them against instrumented counters; `merge_h2_ladder.py` part 5 recomputes them
independently - the two agree to the digit):

- **merge path** = generation visits (lap walks producing the old word, deletions
  included) + scanner pushes (chain-condition checks: one letter fed to one scanner) +
  base-word sieve strikes and cells where the path sieves a base word;
- **construction path** = sieve strikes (teeth per gear x P/q, summed over gears) +
  P cells scanned to read off the gaps.

Instrumented verification (ops mode, section C): slot-sieve survivor count 22,275 ==
A(17); real 17->19 scan pushes 22,531 == A(17) + 256 wrap margin; summed h_2 word
lengths 4,246,778,880 == the closed form. Wall times are a **secondary sanity check
only**: every wall time below is from a run executed **alone** (strictly sequential
protocol); earlier concurrent-run timings were discarded as contaminated.

## 3. The h_2 family ladder (ZM's object; ops primary, solo wall secondary)

| step | what ran | h_2 (merge) | ZM | match? | chain condition | merge ops | construction ops | ratio | solo wall (merge) |
|---|---|---|---|---|---|---|---|---|---|
| base 17 | exhaustive, 127,628 classes x P = 255,255 | 192 | 192 | MATCH | - (direct scan) | 8.52e10 (sieve, all classes) | same - this is the base scan | 1 | 4.4 s |
| 17 -> 19 | merge: 127,628 17-words x 10 scanners (s ~ 19-s) = all 2,424,932 classes; 19-period never built | **258** | 258 | **MATCH** | exact, held everywhere; no fallback (audited: 200 random classes + argmax vs construction) | 1.277e11 | 3.198e13 (2,424,922 classes x P19 sieve) | **250x** | 34.5 s |
| 19 -> 23 | merge: 2,424,932 19-words generated lap-wise from their 17-parents (never sieved at P19) x 12 scanners, every word fully scanned, no pruning | **366** | 366 | **MATCH** | exact, held everywhere; no fallback (argmax rebuilt by construction over P = 111,546,435: 183) | 1.813e13 | 1.745e16 (55,773,217 classes x P23 sieve) | **962x** | 7032 s (effectively solo; all 2,424,932 words, 0 pruned) |
| 23 -> 29 | NOT RUN | - | 450 | - | - | ~1.7e16 (estimate) - a dedicated multi-day run; not executed | ~1e19 | ~10^3x | - |
| 29 -> 31 .. 71 -> 73 | NOT RUN - infeasible | - | 570 .. 2622 | - | - | work multiplies by ~q'(q'-2) ~ 500-4000 per rung (class count x q', word length x (q'-2)); the 31 rung alone is ~10^18 ops | worse still | - | - |

h_2(19) = 258 and h_2(23) = 366 are the **first in-project exact values at those levels**
(the round-17/20 elite lift reached only h_2(19) >= 222; exhaustive scans were priced out
at 1.2e13 ops). They independently replicate Ziller-Morack by a different algorithm.

**Why the ladder stops, honestly.** Not because the chain condition fails - it never
failed anywhere it could be checked - but because h_2's argmax refuses to stay near the
top of the family between rungs:

- 19-argmax: e = 1,532,627 (= 1097 mod 255255, +-8 mod 19), F jumps 54 -> 129 at gear 19
  (+3.95 q'). Its 17-level value 54 is the **twin's own value** - 35,848 classes rank
  strictly above it at 17, so no elite lift could see it (the r20 lift's best,
  e = 1,335,364 from a 17-champion with F = 96, reached only 111).
- 23-argmax: e = 107,207,699 (= 599 mod 255255, +-9 mod 19, +-7 mod 23), trajectory
  F(17) = 45 -> F(19) = 81 -> F(23) = 183 (+4.43 q' at the last step), starting **below**
  the twin class.

So the "maximiser persistence" observed at 13 -> 17 (champions stay at the 99.3-99.8th
percentile) does **not** persist as a ladder principle: exact h_2 needs every class
carried, and the class count is P(y)/2. ZM's own route to 73 is a direct extremal-sequence
search in the condensed (two-free-residues) formulation, whose cost scales with h_2, not
with the period - a different algorithm class, not an incremental one. The merge law's
exact reach for h_2 is "the last exhaustively-scanned level plus one or two rungs" -
delivered here at 250x and 962x below construction cost.

## 4. The fixed-twin ladder F(2,y) (slot frame; adjacent = 3x; this is NOT h_2)

Merge ops = generation visits + scanner pushes; construction ops = strikes + P cells
(closed forms, verified instrumented; the streamed rungs share a fixed prologue of
3.17e8 ops to build the RAM word at 29, excluded below as it is common to all).

| step | merge F_slot (adj) | corpus | match? | chain condition | merge ops | construction ops | ratio | merge solo wall | construction solo wall |
|---|---|---|---|---|---|---|---|---|---|
| 17 -> 19 | 25 (75) | 75 | MATCH | exact | 2.04e5 | 3.63e6 | 17.8x | 0.0001 s | 0.00 s |
| 19 -> 23 | 34 (102) | 102 | MATCH | exact | 8.02e5 | 8.67e7 | 108x | 0.0011 s | 0.02 s |
| 23 -> 29 | 43 (129) | 129 | MATCH | exact | 1.67e7 | 2.59e9 | 155x | 0.023 s | 0.50 s |
| 29 -> 31 | 58 (174) | 174 | MATCH | exact | 4.45e8 | 8.24e10 | 185x | 0.56 s | 8.4 s |
| 31 -> 37 | 88 (264) | 264 | MATCH | exact | 1.29e10 | 3.12e12 | 242x | 5.4 s | 377.8 s |
| 37 -> 41 | 91 (273) | 273 | MATCH | exact | 4.64e11 | 1.30e14 | 281x | 230.3 s | not run (formula value; ~4 h at measured sieve rate) |
| 41 -> 43 | not obtained - run terminated for the round-21 machine handover (needs ~2-4 h exclusive) | 309 | - | exact (run not completed) | 1.86e13 | 5.70e15 | 306x | not measured - machine never idle | **infeasible** (P = 2.18e15 slots); the independent cross-check is the project's covering-search value 309 itself |
| 43 -> 47 | NOT RUN | (unknown - would be a first computation) | - | - | ~8e14 | ~2.6e17 | ~330x | est. ~3 h at measured rate - out of scope here; the pruned covering search (rust2) is the better tool past 43 | infeasible |

Every merge value through 41 agrees with the project's independently computed ladder -
including F(2,37) = 264 and F(2,41) = 273, computed from streamed words whose periods
(1.2e12, 5.1e13 slots) were never materialised. The 41 -> 43 rung would have been the
strongest cross-check available (merge law vs the rust2/maxgap covering search, two
wholly independent methods); its run was terminated for the round-21 machine handover
before completing, so F(2,43) = 309 stands on the covering search alone and the merge
cross-check at 43 remains open (the run is one `h2ladder twin43` invocation, ~2-4 h on
an idle machine).

Frame reconciliation for the table: the corpus F(2,y) column is the adjacent frame; the
slot values are F/3 exactly (F_adjacent = 3 F_slot, verified for every rung here and in
`gear-recursion.md` section 1).

## 5. Honesty ledger

- **A prune was claimed exact-safe, found unsound in review, and retracted.** The first
  h_2(23) run skipped words whose max(F2, best window of gaps >= 22 plus flanks) could
  not beat the running best, on the argument that 23-chains need interior gaps >= 22.
  That argument borrows the deletion-spacing lemma, whose premises (old gaps >= 3,
  adjacent teeth) do NOT transfer to the halved frame: classes with 3 | e have gaps of 1,
  and an interior gap can equal the tooth separation s. The run was therefore redone with
  no pruning at all; both runs give h_2(23) = 366 (the pruned mode is kept in the code,
  flagged as a heuristic, not used for any reported value). The h_2(19) run and all twin
  runs never pruned anything.
- **Chain-condition fallbacks needed: zero.** The condition is proven exact
  (`gear-recursion.md` 3-4a) and every value it produced that could be cross-checked by
  construction (668 battery cases, 200 sampled 19-classes, both h_2 argmaxes, twin rungs
  through 37, stream-internal F re-derivations) agreed exactly. No value was patched.
- **Saturation-theorem shortcut: never applicable** on these rungs. It needs
  q' - 1 > F(M) in matching units; on the twin ladder 3*F_slot(M) >= 54 always exceeds
  q' - 1 (<= 42). No run used the shortcut anywhere: every value came from the full chain
  scan, so nothing rests on the saturation regime.
- **Benchmark protocol corrected mid-test** (human feedback): early comparison runs were
  executed concurrently and their wall times were contaminated; all such timings were
  discarded, every reported wall time is from a solo sequential re-run, and the primary
  metric was switched to deterministic operation counts (section 2a), closed-form and
  instrumented, identical between the Rust and Python computations.
- **Not computed**: exact h_2 at 29 (est. ~1.7e16 merge ops - a dedicated multi-day run;
  estimated, not run) and beyond (infeasible, factor ~q'(q'-2) per rung); F(2,47) and
  beyond (est. ~8e14 ops); nothing past p_n = 73 - so despite ZM stopping at 73, **no
  first h_2 computation past 73 is claimed, and none is in reach by this route**; the
  "compute 79 and 83" branch of the task does not activate because the method does not
  hold to 73 in the first place.
- The task's framing "h_2 = the twin pattern" was corrected by the mapping (section 1)
  rather than silently worked around; the twin ladder is reported as what it is, F(2,y).

## 6. Verdict

The merge law passes the test it can pass, exactly, and fails to be a road to p_n = 73 for
a structural reason it itself exposes. As an incremental computer of paired-Jacobsthal
values it is exact and genuinely cheaper: it replicated Ziller-Morack's h_2(19) = 258 and
h_2(23) = 366 to the digit - the project's first exact values at those levels, previously
priced out (the round-20 elite lift could not close the gap, >= 222 vs 258) - at 250x and
962x fewer elementary operations than from-scratch construction of the new periods; on the
fixed-twin ladder it reproduced every corpus value F(2,19) = 75 through F(2,41) = 273,
the deep rungs computed over streamed words whose periods (up to 5.1e13 slots) were never
built, at operation ratios growing from 18x to 281x, where construction is out of reach
on this machine (the 41 -> 43 rung, ratio 306x, was terminated unfinished for the
round-21 machine handover). But it does not extend h_2 past 23-29: exactness
requires carrying the entire difference family (the 19-maximiser ranks 35,849th at level
17 and the 23-maximiser starts below the twin class - maximiser persistence is false at
these rungs, so no elite shortcut is exact), and the family's total word volume grows by
~q'(q'-2) per rung. Ziller-Morack's own table to 73 rests on a period-free extremal
search, a different algorithm class; the merge law's honest niche is per-difference
ladders (where it owns values the literature lacks) and a one-to-two-rung exact extension
of any exhaustively known family level - both delivered here with every number exact and
every check passed.

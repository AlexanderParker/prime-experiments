# COV(M): exact gap spectra of unscannable sieve machines by CRT + SAT

## 1. WHAT IT IS

Plain language: every question of the form "does the 6k+-1 sieve with gears
5..M ever produce a gap of exactly v" (or: a window of j consecutive gaps
summing to S, or: two adjacent gaps (u, v)) is decided WITHOUT scanning the
machine's period, by noticing that each gear blocks two residue classes at a
fixed separation with one free phase, and that by CRT every choice of phases
is realised by an actual position. The question becomes a tiny CNF
(~300 variables), a SAT solver answers it in seconds to minutes, every
positive answer is CRT'd back to an explicit slot address and re-verified
against the machine directly, and every negative answer is an exhaustive
refutation. Machines whose periods (5x10^13 to 5x10^18) no scan can cover
become exactly computable.

Precise form. Slot k is blocked by gear q iff k = +-u_q (mod q),
u_q = 6^{-1} mod q. Anchoring a window at position 0 and writing
a_q = (u_q - k) mod q, gear q blocks exactly {a_q, a_q + s_q} (mod q) with
s_q = -2u_q and a_q in Z_q free. A gap of exactly v occurs at machine M iff
there exist phases (a_q) with positions 1..v-1 covered and 0, v uncovered;
a j-gap window of span S with interior openings W likewise. Encoding: one
boolean per (gear, endpoint-valid phase), exactly-one per gear, one coverage
clause per interior position; spared-position selectors plus an
exactly-(j-1) cardinality constraint for F_j; distance clauses for the
qualifying spectrum Q_j; a forced spared position for adjacent pairs.

New exact values produced (2026-08-23), all previously unreachable:

    F(41) = 91 with COMPLETE hole list {84, 87, 89} (tail 92..100 all
        refuted; F(41) independently equals the merge-law record measured
        by the full-period machine-37 padding census - two disjoint
        methods agreeing)
    fuel caps decided at FULL period with no scan (fuel_sat.py):
        N_4(37->41) = 0 and N_4(41->43) = 0 - k_max = 3 at both steps;
        validated at 31->37 where SAT finds exactly the two known words
    THE FIRST DOUBLE-PADDED RUN: word (43, 43) at step 41->43 realized
        at witness k = 116,431,845,582 (three openings k, k+43, k+86
        sharing one residue mod 43) - a prediction standing since round
        16, for a period (5.07e13) no scan reaches; also gap 86 = 2q'
        occurs at machine 41 (single-gap double padding), previously
        never observed at any censused step
    F_2(37) = 90 exact (period 1.24e12; witness has a gap of 2 adjacent
        to the maximal gap 88); F_3(37) bounded to [97, 163] with the
        descent checkpointed (each refutation ~15 min)
    partial rows, checkpointed and resumable: F(43) >= 103 with 104
        refuted (a 5,770 s refutation), tail [105,118] open;
        F(47) >= 118; F(53) in [136, 145], the top pinned by the
        independent pruned covering search F(2,53) = 435
    adjacency of two maximal gaps refuted at machines 31, 37, 41
        (previous best: y <= 23 by class arithmetic + period scan)

## 2. WHY IT MIGHT BE NOVEL

Computing Jacobsthal-type quantities by exhaustive covering search is
classical (Jacobsthal function computations; the project's own pruned
F(2,y) searcher). What appears new here:

- the ENDPOINT-SPARED spectrum (which gap values occur, i.e. the hole
  lists), not just the maximal coverable run - holes are invisible to
  plain covering searches;
- window spectra F_j (max sum of j consecutive gaps) and qualifying
  spectra Q_j as SAT instances - to our knowledge these window statistics
  of sieve machines have no published exact values at any modulus;
- the two-classes-at-fixed-separation-one-free-phase reduction that makes
  the instance small (the corridor/tooth structure of the 6k+-1 frame);
- witness round-tripping: every SAT model is converted to an explicit
  slot address by CRT and checked against the machine definition, so
  correctness does not rest on the encoding.

## 3. PROOF / STATUS

SCRIPT-VERIFIED (finite): research/cov_sat.py.

- Validation: all 8 full-period-scanned machines (11..37) - F and the
  COMPLETE hole lists reproduced exactly (machine 37: 13 holes found by an
  11,829 s full-period scan of 1.24e12 slots, reproduced in 123 s);
  machine 23 F_2..F_6 = 39, 50, 58, 65, 77 and machine 29
  F_2..F_6 = 55, 65, 70, 85, 90 all exact, with witness addresses matching
  the round-17 census maximisers (k = 2,082,580; 29,098,935; 407,599,253;
  725,859,998).
- Positive answers carry machine-verified witnesses (assert in code).
- Negative answers are solver refutations (UNSAT); the solver is trusted
  for these. A kernel-checkable route exists (Formalist: a finite CRT
  enumeration per refuted v), not yet taken.
- Engineering note recorded for reproducibility: pysat's C cardinality
  encoder and Minisat22 both corrupted the heap over many instantiations
  (segfaults); a pure-Python sequential counter + Cadical153 is stable.

## 4. IMPLICATIONS

Inside the project: the prefix rows of every spectrum table (machines 37+)
get exact values instead of lower bounds; (D)-type criteria at steps beyond
scan reach become decidable (F_3(37) <= F + q' = 129 is the live one); the
first double-padded run question at 41->43 becomes attackable structurally;
upper bounds F_j that no prefix can give now exist. Outside: exact values
of a Jacobsthal-variant spectrum (with endpoint sparing and window depth)
at moduli beyond enumeration; the method transfers to any sieve with
finitely many residue classes per prime.

## 5. UNSOLVED QUESTIONS IT TOUCHES

Jacobsthal-function computation (known values stop where exhaustive search
stops; this route pushes specific structured cases much further); the
project's Reduction A (fuel caps and spectra at arbitrary steps); Polignac
per-difference analogues (the same encoding with the difference's gcd
classes).

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"Jacobsthal function SAT", "covering system SAT solver", "maximal gap
coprime residues exact computation", "sieve gap spectrum". Nearest known
work: Hagedorn's Jacobsthal computations; Holt-Rudd cycle methods (see
merge-law entry); SAT for covering systems (Cummings et al.?).

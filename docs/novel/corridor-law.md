# corridor-law - adjacent equal padded links forbidden for exactly 12 of 24 classes

## 1. WHAT IT IS

Plain language: a "padded link" in the merge step M -> M + q' is a killed run whose two kills
sit a multiple of q' apart (same tooth of the new gear), which costs a gap of at least q' in
the old machine. The question "can two equal padded links sit ADJACENT in one run?" turns out
to be decided by q' mod 35 alone - and the answer is NO for exactly half of the possible
classes: 12 of the 24 residues mod 35 coprime to 35 forbid the configuration outright, by the
(5,7) corridor alone, with no input from the machine's gap spectrum. Moreover the feasible and
infeasible classes interlock in a perfect dichotomy with the unequal shapes.

Definitions. Slot k = the pair (6k-1, 6k+1); gear q >= 5 blocks k iff k = +-(6^{-1}) (mod q)
(teeth +-u', u' = round(q/6)). The corridor mod 35 is the 15-residue exposed set
E = {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33} left twin-eligible by gears 5 and 7. For a chain
with step list `steps`, `carrier steps` is the set of base residues r mod 35 such that r and
every partial sum r + offset lie in E; if the carrier is empty the chain of openings cannot
exist anywhere, at any scale (`no_chain_of_carrier_empty`).

Precise form (kernel-checked): two adjacent equal padded links of gear q' occupy openings at
r, r + g, r + 2g with g = q' mod 35, i.e. a three-term arithmetic progression inside E. Then:

    equal_padding_forbidden_classes :
      ((Finset.range 35).filter fun g => Nat.gcd g 35 = 1 ∧ carrier [g, g] = ∅)
        = {1, 4, 6, 9, 11, 16, 19, 24, 26, 29, 31, 34}
    equal_padding_forbidden_card : (that set).card = 12

So the configuration is impossible for q' = 29, 31, 41, 59, 61, 71, 79, 89, ... (classes on
the left) and corridor-feasible for q' = 23, 37, 43, 47, 53, 67, 73, 83, 97, ... . Instance
theorem: `no_adjacent_padded_41 : carrier [41, 41] = ∅` - at 37->41 (g = 6), r, r+6, r+12
all in E has ZERO solutions. And the dichotomy:

    padding_shape_dichotomy : ∀ g < 35, gcd g 35 = 1 →
      (carrier [g, g] = ∅ ↔ carrier [g, (2g) % 35] ≠ ∅ ∧ carrier [(2g) % 35, g] ≠ ∅)

wherever the equal shape (1,1) is infeasible, both unequal shapes (1,2)/(2,1) are feasible,
and vice versa - a perfect complementary split of the 24 classes.

Completeness (lateral r17, script-verified): a shape with n openings can be blocked by gear q
only if q <= 2n, so for n <= 3 openings the mod-35 test IS the entire corridor - these
verdicts are complete over ALL moduli, not a mod-35 shadow. (Feasible means not
corridor-obstructed; it does not mean the configuration occurs.)

## 2. WHY IT MIGHT BE NOVEL

- It is a FORBIDDEN-CONFIGURATION law with an exact classification: not "rare", not "density
  0", but impossible - and the impossible set is computed exactly (12 named classes), with a
  sharp structural complement (the dichotomy with the unequal shapes). The corpus knows of no
  standard result classifying which common differences g admit a 3-term AP inside the
  twin-admissible residue set mod 35.
- The density of E is 15/35 = 3/7 - far above any Roth/Varnavides-type threshold, so E
  certainly contains many 3-APs in aggregate. The content is per-difference: for HALF the
  invertible differences there is no 3-AP at all. Aggregate AP-counting machinery does not see
  this.
- It replaced a dying quantitative bound with a permanent residue criterion: the r14 padding
  lemma's spectrum threshold (2q' > F(M)) expires at 37->41; the corridor law never expires
  (lateral 17-18).
- The delta from randomness is checked: a smooth supply^2/gaps model predicted ~5
  double-padded runs at 37->41 where the corridor forbids the adjacent shape outright
  (lateral, failed approach 12) - arithmetic selection beats the smooth law.

Honest deflation: the proof is a finite check over 35 residues, CRT-factorisable over gears 5
and 7, and entirely elementary. What might be novel is the statement and classification, not
any proof technique.

## 3. PROOF

Status: KERNEL-CHECKED.

- Lean: `proofs/TierA.lean` (round 16 block; ledger green, zero sorries, zero warnings):
  - `no_adjacent_equal_padded` - carrier [q, q] = ∅ implies no two adjacent equal padded
    links of gear q exist anywhere (via `no_chain_of_carrier_empty`).
  - `no_adjacent_padded_41 : carrier [41, 41] = ∅`.
  - `equal_padding_forbidden_classes` = {1,4,6,9,11,16,19,24,26,29,31,34} and
    `equal_padding_forbidden_card` = 12 (kernel `decide`; no `native_decide` in the ledger).
  - `padding_shape_dichotomy` as displayed above.
  - Axiom footprint: at most the standard three `[propext, Classical.choice, Quot.sound]`
    (ledger convention; the carrier machinery is finite-set arithmetic).
- Cross-checks: carriers checked against `research/flank_tierA_fix.py`; the round-16 theorems
  checked against lateral.md's r15 statement (feasibility a function of q' mod 35; the 12/24
  split; the dichotomy). The related shape law - consecutive padded links separated by j in
  {0,1} only, feasibility a function of q' mod 210 - is SCRIPT-VERIFIED for every prime to
  4000 (lateral r16) but is a separate, weaker-status statement.
- Related kernel-checked corridor context (same ledger, `proofs/Corridor.lean`):
  `exposed_iff_mem`, `endpoint_law`, `adjacency_law`, `forbidden_pairs_count` (294 of the
  1225 gap pairs mod 35 are jointly infeasible), `no_chain_of_forbidden`.

## 4. IMPLICATIONS

Inside the project:
- Padded structure is the sole escape from the literal cap (docs/novel/literal-cap.md), so
  laws capping padded shapes are what keeps merge runs short. This law kills the adjacent
  equal shape for half of all gears - including the knife-edge step 37->41 - with no
  spectrum input, and it seeded the r16 AP lemma + shape law (no 4 openings in AP with
  difference q'; padded separations j in {0,1} only).
- It marks 41->43 as the first step with NO obstruction of any kind (with F monotone), a
  banked prediction (first double-padded run at 41->43).
- Method template: state a shape as a step list, compute `carrier`, get a machine-free
  permanent verdict; completeness for n <= 5 openings means mod 35 suffices.

Outside: a worked example that admissible-residue wheels carry exact per-difference AP
prohibitions invisible to density arguments; potentially relevant to anyone counting
constellations in sieve cycles (Holt-Rudd) where certain patterns have population zero for
arithmetic reasons.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Twin prime conjecture via the project route: part of the padded-span control (C)/(E)
  infrastructure around the open flank bound (D).
- Polignac-type generality: whether the 12/24 classification transfers to general even gap d
  (the analogous corridor for the pair {0, d}) is open in the corpus; the shape law's
  q' mod 210 criterion is verified to 4000 but not kernel-checked.
- Hardy-Littlewood constellation counts: the law identifies constellations whose local count
  is exactly zero at half the classes - a boundary case of admissibility the HL heuristic
  handles only implicitly (singular series factor zero).

## 6. PRIOR-ART CHECK

Date: 2026-08-23. Engine: Claude WebSearch (web). Searches actually run:

1. ""three term arithmetic progression" OR "3-AP" in reduced residues mod 35 OR mod 105
   admissible twin residues classification common difference" - nearest: minimal counts of
   3-APs mod a prime (arXiv:math/0501004), 3-AP-free sets in cyclic groups
   (arXiv:2606.30186), standard admissible-residue definitions. Nothing on per-difference
   AP-freeness of the twin-admissible set mod 35, and no classification of differences.
2. "Erdos covering systems congruences forbidden configurations residue classes small primes
   consecutive positions" - covering systems (Hough 2015; Balister-Bollobas-Morris-Sahasrabudhe-
   Tiba) concern covering ALL integers by congruences; the corridor law is the complementary
   exposure question but no published analogue of the 12/24 classification appears in that
   literature.
3. ""twin prime" wheel modulo 35 "admissible residues" runs consecutive candidates blocked
   prime sieve pattern" - standard nu(q) = q - 2 counts (Tao 254A Notes 4); Mathar's "Table
   of prime gap constellations" (OEIS A022004 attachment) lists forbidden CONSTELLATIONS
   (patterns divisible somewhere) but classifies tuples, not common differences of padded
   links, and has no dichotomy statement.
4. "Holt Rudd "Eratosthenes sieve" cycle of gaps primorial constellations populations twin gap
   patterns" - Holt & Rudd (arXiv:1408.6002, arXiv:1510.00743): population dynamics of gap
   constellations in G(p#); the frame could express this law but their papers do not state
   per-class impossibility results for padded/adjacent shapes.
5. ""mod 35" 15 residues twin prime candidates "5 and 7" wheel forbidden gap classes
   classification kernel Lean formalized" - no hit combining these; no formalised (Lean or
   otherwise) forbidden-configuration classification for twin wheels found.

Nearest published results overall: forbidden prime-gap constellations (Mathar's tables; the
classical fact that inadmissible tuples have finitely many prime realisations); covering-
system literature (Erdos; Hough; Balister et al.); 3-AP counting in Z_p (Fitzpatrick et al.);
Holt-Rudd sieve-cycle constellation dynamics.

VERDICT: NOVEL AS FAR AS SEARCHED - for the exact classification (12 of 24 invertible classes
mod 35 forbid adjacent equal padded links; perfect dichotomy with the unequal shapes;
completeness of the mod-35 test for n <= 5 openings). The underlying METHOD (CRT + finite
residue check for tuple admissibility) is KNOWN and standard - this document claims the
classification and its role, not the technique. If a referee reads the statement as "certain
3-term patterns are inadmissible mod 35", that bare fact is a standard admissibility
computation; the classification-with-dichotomy as a law about padded merge structure was not
found anywhere.

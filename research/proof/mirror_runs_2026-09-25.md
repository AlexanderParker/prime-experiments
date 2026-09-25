PART 1: KERNEL (three modules, all in namespace RangeLine, none committed)

All three files are untracked. lakefile.toml is modified: each module has a `[[lean_lib]]` block appended at the end, in the order RangeRegion, RangeCentre, RangeReach, and none is in defaultTargets. I did not rebuild anything. I did re-check four things: the files exist, the declaration lists, a forbidden-token grep (sorry|admit|axiom|native_decide gives 0 hits in all three files), and the lakefile blocks.

1. RangeRegion: C:/dev/primes/proofs/RangeRegion.lean
- Imports: Mathlib.Data.Nat.Prime.Basic, Mathlib.NumberTheory.Primorial, Mathlib.Tactic, RangeCopies.
- Declarations (6 plus the def):
  - `def P (j) := Nat.minFac (900*j^2 - 1)`
  - `P_eq_minFac_legs`: P j = minFac((30j-1)(30j+1))
  - `minFac_sq_le_of_composite_dvd`
  - (a) `revealed_iff_minFac (hj : 1 ≤ j)`: ((30j-1).Prime ∧ (30j+1).Prime) ↔ 30j+1 < (P j)^2
  - `blamed_iff_minFac`: the negated form of (a)
  - (b) `total_blame_iff (Nmax)`: (¬∃ j, 1 ≤ j < Nmax with both legs prime) ↔ ∀ j with 1 ≤ j < Nmax, (P j)^2 ≤ 30j+1
  - `total_blame_iff_primorial (q)`: (b) at Nmax = primorial q / 30
- Proof of (a):
  - Forward: minFac of the product equals the prime leg it divides, so P j ≥ 29 and (30j-1)^2 > 30j+1.
  - Backward: a composite leg L has minFac m with m^2 ≤ L, and P j ≤ m.
- Build: `lake build RangeRegion` succeeded (3019 jobs). The reviewer's fresh `lake env lean` exited 0 with no output. 0 errors, 0 warnings.
- Sorry: none. Axioms: all 6 use only [propext, Classical.choice, Quot.sound].
- Reviewer: faithful to the request, with one flag against the request itself.
  - `total_blame_iff_primorial` is true but says nothing. For q ≥ 7, j = 1 (29, 31) is inside the bound and P 1 = 29 with 29² > 31, so both sides are false. For q < 7 both sides are vacuous.
  - The range-window form, with bounds q < 30j-1 and 30j+1 ≤ q#, follows in one line from `blamed_iff_minFac` but is not in the file.
  - Minor: the one-letter global `RangeLine.P` can capture an auto-bound `P` in later RangeLine files.
- Corrections by the lane: none.

2. RangeCentre: C:/dev/primes/proofs/RangeCentre.lean
- Imports: Mathlib.Data.ZMod.Basic, Mathlib.Tactic.
- Declarations:
  - `def StrikesCopy g k := g ∣ (30k-1)(30k+1)`
  - `strikesCopy_iff_sq`: StrikesCopy g k ↔ g ∣ 900k^2 - 1, for every k
  - `strikesCopy_iff_zmod (1 ≤ k)`
  - `centre_law_field`: over any field with 30 ≠ 0 and 4 ≠ 0, both strike ↔ (b-a = 0 ∧ 900a^2-1 = 0) ∨ (a+b = 0 ∧ 225(b-a)^2-1 = 0)
  - `centre_law (g.Prime, 7 ≤ g, 1 ≤ x, x < y)`: (StrikesCopy g x ∧ StrikesCopy g y) ↔ (g ∣ y-x ∧ g ∣ 900x^2-1) ∨ (g ∣ x+y ∧ g ∣ 225(y-x)^2-1)
- Proof: work in ZMod g and split into the four sign cases.
  - The converse uses 900y^2-1 = (900x^2-1) + 900(y+x)(y-x).
  - It also uses 900x^2-1 = (225(y-x)^2-1) + 225(3x-y)(x+y), and the same with x and y swapped.
- Build: success (3010 jobs), 0 errors, 0 warnings. One linter warning (haveI) was fixed.
- Sorry: none. Axioms: all 4 use only [propext, Classical.choice, Quot.sound].
- Reviewer: exactly the requested statement.
  - Every hypothesis is used and none is vacuous.
  - Every ℕ subtraction is a true subtraction under the hypotheses.
  - No g ∣ 900x²-1 is needed in the second disjunct, because y ≡ -x gives 225(y-x)² ≡ 900x².
  - It sharpens RangeCopies.copy_leg_rule / copy_pair_iff rather than duplicating it.
  - It re-proves the product identity locally, which is harmless.
- Corrections: none.

3. RangeReach: C:/dev/primes/proofs/RangeReach.lean
- Imports: RangeHandoff, RangeChain.
- Source: the scratch file adj_g5_0925/G5RangeAdj.lean, which is byte-identical to g5_dichotomy/G5Range.lean.
- Definitions:
  - TwinNode p := p.Prime ∧ (p+2).Prime
  - NextNode s t := TwinNode t ∧ s < t ∧ no TwinNode strictly between
  - Rung s r := TwinNode r ∧ s < r ∧ r+2 ≤ s#
  - Good s := ∃ t, NextNode s t ∧ t+2 ≤ s#
  - RangeAll := ∀ q ≥ 7, RangeStatement q
  - GoodReach: an inductive with root 29 that steps s → s⁺ only when s⁺+2 ≤ s#
  - NS := every good node s ≥ 29 has s⁺ and s⁺⁺ with s⁺⁺+2 ≤ s#
  - NSAll := the same for every node s ≥ 29, with no goodness condition
- 38 theorems (I counted 38 theorem/lemma lines), 0 sorries. Main ones:
  - range_iff_all_good: RangeAll ↔ ∀ TwinNode s ≥ 29, Good s
  - rangePrime_iff_all_good: the same with q restricted to primes
  - good_iff_rangeStatement: Good s ↔ RangeStatement s
  - good_iff_exists_rung
  - goodReach_iff and GoodReach.initial: the reachable nodes are exactly an initial run of nodes from 29
  - reach_unbounded_iff and range_iff_reach_unbounded: RangeAll ↔ GoodReach is unbounded
  - ns_of_nsAll, all_good_of_ns, range_of_ns, rangePrime_of_ns, twins_unbounded_of_ns
  - ns_iff_two_rungs: NS ↔ every node ≥ 29 has two distinct rungs
  - ns_iff_no_single_rung
  - drop_at_most_one, first_bad, bad_iff_terminal
  - not_parent_unique: 149 is a rung of both 29 and 59
  - Successor and rung lemmas: nextNode_unique, nextNode_le, exists_nextNode, exists_prevNode, rung_transport, overlap_agree, rungs_initial, rung_ge_next
- Build: success (3020 jobs), 0 errors, 0 warnings. One unused-variable warning was fixed.
- Sorry: none. Axioms:
  - 36 theorems use [propext, Classical.choice, Quot.sound]
  - nextNode_le uses [propext, Quot.sound]
  - nextNode_unique uses [propext]
- Reviewer: faithful to the spec.
  - NS is stated conditionally, which is a weaker hypothesis, so the results are stronger than asked. The spec-literal chain goes through ns_of_nsAll and is not a separately named theorem.
  - The reviewer checked the proofs of rangeAll_of_all_good, exists_prevNode, goodReach_iff and bad_iff_terminal.
  - There are no RangeLine name clashes.
  - Scope difference: the scratch file's copy-node results (nodes with 30 ∣ p+1, copy-node RNS → RangeCopy) are not in the repo. Twin-node NS and copy-node RNS are different hypotheses.
- Corrections by the lane:
  - The node set is twin nodes, not copy nodes.
  - Good is defined through the successor, and proved equivalent to "has a rung".
  - NS is conditional, with NSAll as the unconditional form.
  - Reach steps only to s⁺.
  - RANGE is proved for all q ≥ 7 and for prime q.
  - Dropped: nc_of_ns (it follows from all_good_of_ns), and no_node_region_17, which is false for twin nodes because of (311, 313) between 17² and 19².
  - leaf was renamed to first_bad / bad_iff_terminal.

PART 2: MATHS

Notation: M = q#/30, a = q#/2, ρ = a+1. The pair is P_d = {s, M-s} with s = (M-d)/2 and d odd in [1, M-2]. Low legs are L1 = a-15d-1 and L2 = a-15d+1; high legs are H1 = a+15d-1 and H2 = a+15d+1. t_g = 15⁻¹ mod g and w_g = (ρ-g²)/15. U⁻ = {g ∈ U_q : g² ≤ ρ}. Q = the largest prime ≤ √ρ.

A. MIRROR PAIRS UNDER ACTING

Standing:

A1 (Class form, PROVED)
- A prime g > q strikes the low member iff d ≡ M ± t_g (mod g), and the high member iff d ≡ -M ± t_g.
- Both members are survivors or neither is. They are survivors iff gcd(225d²-1, M) = 1.
- Checks:
  - d-form against j-form: all gears and all odd d at q = 7..19; at q = 23, 169 gears against all 3,718,214 odd d. 0 mismatches.
  - Survivor formula: q = 7..23, 0 mismatches.

A2 (Acting windows, PROVED)
- The low member is deleted iff some upper g in a low class has 15d ≤ ρ - g².
- The high member is deleted iff some upper h in a high class has 15d ≥ h² - ρ.
- Joint acting forces g² + h² < q#+2, hence gh < ρ. Equality is impossible because g²+h² ≡ 2 (mod 8) and q#+2 ≡ 0 (mod 4).
- g = h happens only when g ∣ q#∓2, d ≡ 0 (mod g), both struck legs are of the same type, and g ≤ √ρ.
- Checks: class-form deletion against an Eratosthenes sieve on every survivor pair at q = 7..23 gives 0 low and 0 high mismatches. The g = h cases have 0 leg-type violations.

A3 (Involution, stands)
- DD_q = Del ∩ (-Del) and D(h,g) = -D(g,h).
- Measurement only, not a result, at q = 23:
  - 1,325,362 survivor pairs
  - 1,043,159 doubly deleted
  - 145,157 with only the low member revealed
  - 120,841 with only the high member revealed
  - 16,205 with both revealed
- Instance: q = 23, d = 515, s = 3717957. All four legs 111538709, 111538711, 111554159, 111554161 are prime.

A4 (Leg classes, PROVED)
- In 15d-space the leg classes are a-1, a+1, 1-a, -1-a.
- Two legs are struck together only if g ∣ q#∓2, only at d ≡ 0 (mod g), and on the same leg type: L1 and H1 when g ∣ q#-2, L2 and H2 when g ∣ q#+2.
- When g ∣ q#∓1, the four classes form the progression -3/2, -1/2, 1/2, 3/2, labelled H,L,H,L or L,H,L,H. d = M is a high class iff g ∣ q#∓1.
- q = 23 classification (1745 gears in total):
  - 6 machine gears
  - 587 (divides q#+2)
  - 4801 and 11617 (divide q#-2)
  - 37 and 131 (divide q#-1)
  - 317 (divides q#+1)
  - 1733 generic

A5 (5', PROVED; the literal "exactly onto" is REFUTED)
- R is an order-reversing involution of [0, √(2ρ)]. It maps (√(ρ-15d), √(2ρ)] onto [0, √(ρ+15d)) and [0, √(ρ-15d)] onto [√(ρ+15d), √(2ρ)].
- g acts on the high member iff g² < ρ+15d, or g² = ρ+15d = 30(M-s)+1. In the second case the high upper leg is g², so g strikes and deletes it.
- The sandwich G(low) ⊆ U⁻ ⊆ G(high) holds.
- Refutation of "exactly onto": the image misses its endpoint y = √(ρ+15d). Instance q = 13, d = 143, s = 429:
  - Low legs: 12869 = 17·757, 12871 = 61·211
  - High legs: 17159 (prime), 17161 = 131²
- Further cases with H2 = g²:
  - q = 17, s = 261, g = 709
  - q = 23, s = 7261, g = 14929
- Measurement: 0, 1 and 3 disagreements at q = 7, 11, 13, all of them H2 = g² cases.

A6 (Mean-height identity, PROVED; definitions corrected)
- Revealed_q = Loss_q ∪ (S_Q ∩ [1, M-1] \ Gain_q), where:
  - Loss_q = revealed low survivors with lower leg ≤ √ρ (twins with legs in (q, √ρ])
  - Gain_q = high survivors in S_Q with a leg h·h', √ρ < h ≤ h' prime
- Low rule: a low survivor is revealed iff it is in S_Q or in Loss_q.
- High rule: a high survivor is revealed iff it is in S_Q and not in Gain_q.
- Composite legs of high S_Q copies are h·h' with √ρ < h < √(2ρ) and h' < 2√ρ.
- Consequence: not-Range(q) ⟺ Loss_q = ∅ and S_q \ Strike(U⁻) ⊆ Gain_q.
- Correction 1: the preamble's Loss without the survivor condition is wrong from q = 29 on (29 is a machine gear at q = 29).
- Correction 2: the reported Gain series was |Gain ∩ S_Q|, not |Gain|.
- Series at q = 7..23 (measurement only):

| Quantity | q=7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|
| \|Loss\| | 0 | 1 | 2 | 7 | 19 | 75 |
| \|Gain ∩ S_Q\| | 0 | 2 | 14 | 73 | 882 | 11133 |
| \|Gain\| | 0 | 3 | 18 | 178 | 2338 | 33745 |

- |Rev| at q = 23 is 298,408.

A7 (REFUTED: the high member's deletion always comes from a gear ≤ √ρ)
- Instance q = 13, d = 143: A(low) = {17, 61}, A(high) = {131}, and √ρ = 122.54.
- Instance q = 23, d = 38699:
  - Low legs: 2551·43499 and a prime, so A(low) = {2551}
  - High legs: 112126919 (prime) and 10589², so A(high) = {10589}
  - 10589 is above √ρ = 10561.55
- Such pairs exist at q = 13, 17, 19, 23 and not at q = 7, 11.
- Measurement: 0, 0, 8, 56, 724, 9731 at q = 7..23. My re-run shows this series counts doubly deleted pairs only. Without the "low also deleted" condition the count is 0, 2, 14, 73, 882, 11133, which equals |Gain ∩ S_Q|.

A8 (8', PROVED; two statements REFUTED)
- For a survivor pair with gap d and a prime g > q: g strikes both members iff g ∣ (q#-2)(q#+2) and g ∣ d.
  - If g ∣ q#-2, it strikes L1 and H1 (L1+H1 = q#-2, H1-L1 = 30d).
  - If g ∣ q#+2, it strikes L2 and H2 (L2+H2 = q#+2, H2-L2 = 30d).
  - A cross strike forces g ∣ q#.
- g acts on both members iff 15d ≤ ρ - g², which forces g ∈ U⁻.
- So the acting-and-striking-both set is exactly the primes of (q#-2)(q#+2) in U⁻ that have an odd multiple d ≤ w_g with gcd(225d²-1, M) = 1.
- Empty band (PROVED): [1, w_g] contains no multiple of g iff q#/2 ∓ 1 = g(g+e) with e odd and 1 ≤ e ≤ 13, equivalently iff 2q# ∓ 4 + e² is a square.
- Measured:
  - For q ≤ 3000, the only square is at q = 7 (q#-2, e = 5, g = 8), and 8 is not prime.
  - The existence clause (every such prime in U⁻ has a survivor multiple in its band) holds at q = 11..79.
- Acting-both sets at q = 11..23: {17}; {}; {23, 31, 179}; {29, 109}; {587, 4801}, each equal to the primes of (q#∓2) in U⁻, with every such d ≡ 0 (mod g).
- Sole-striker set ⊆ primes of (q#∓2) in U⁻ (PROVED). Equality is measured at q = 13..23. It is strict at q = 11 (17):
  - d = 17: A(low) = {17, 29}, A(high) = {17}
  - d = 51: A(low) = {17}, A(high) = {17, 19}
- REFUTED: "exactly" as a proved claim (the converse was not argued).
- REFUTED: "no pair has A(low) = A(high) = {g}". Instances:
  - q = 17, d = 3059: 23 on 209369 and 301139
  - q = 19, d = 4321: 29 on 4785029 and 4914659
  - q = 23, d = 68679: 587 on 110516251 and 112576621

A9 (Core, stands)
- A machine prime p strikes pair d iff 15d ≡ ±1 (mod p). d = 1 lies in a tooth iff p ∣ 224, i.e. p = 7.
- On the core d ≤ R, with R = min((ρ - g₋²)/15, the largest d with 15d < g₊² - ρ), G(low) = G(high) = U⁻.
- g₋ / g₊ / R at q = 11..23: 31/37/13, 113/127/74, 503/509/149, 2179/2203/224, 10559/10567/3597.
- d* = 3, 5, 5, 5, 11, 11 at q = 7..23. d* lies in the core at q = 11..23.
- At q = 7, d = 1: G(low) = {}, G(high) = {11}, since 121 = ρ+15.

A10 (REFUTED: P_{d*} is doubly deleted)
- Not doubly deleted:
  - q = 7: 59, 61 | 149, 151
  - q = 11: 13·83, 23·47 | 1229, 1231
  - q = 17: 255179, 255181 | 255329, 311·821
  - q = 19: 307·15797, 191·25391 | 4850009, 4850011
- Doubly deleted:
  - q = 13: A = {67} | {79}
  - q = 23: A = {7789} | {37, 43, 419}

A11 (11', PROVED; "4-class" REFUTED for primes of q#∓2)
- F(d) = L1·L2·H1·H2 = ((a-15d)²-1)((a+15d)²-1) = (a²-(15d+1)²)(a²-(15d-1)²).
- The root classes are ±t(a-1) and ±t(a+1). They are palindromic under d → -d, which swaps low and high.
- There are 4 distinct classes iff g ∤ (q#-2)(q#+2). Otherwise the classes are {0, ±2t}, with 0 a double root.
- 3-class gears at q = 11..23:
  - q = 11: 17 {0, 1, 16}
  - q = 17: 23 {0, 6, 17}, 31 {0, 4, 27}, 179 {0, 24, 155}
  - q = 19: 29 {0, 4, 25}, 109 {0, 51, 58}
  - q = 23: 587 {0, 39, 548}, 4801 {0, 640, 4161}, 11617 {0, 3098, 8519} (11617 lies above √ρ)
- At q = 7, 13 {0, 1, 12} is also a 3-class gear (it lies above √ρ).

Acting in this section:
- No statement uses acting beyond pointwise height inequalities.
- A2, A5, A8 and A9 use acting jointly on both members of a pair, but still per member.
- A1, A4 and A11 are pure residue statements.
- A7 and A10 use acting only through trial division.
- Locating: Loss_q in A6 is a height window. The identity itself sets no height bound.
- Counting: every tally above is descriptive measurement.

Unfinished:
1. No coupling between the two members beyond the sandwich, the same-gear transfer through primes of q#∓2, and the Gain/Loss exceptions. The two-sided covering statement (U⁻ covers the low half with Loss empty, and U⁻ with Gain covers the high half) was reduced to, not attempted.
2. Whether d* lies in the core for every q ≥ 11 was measured only at q = 11..23.
3. The residue law for which products of two band primes land on high survivor legs (the structure of Gain_q) was not worked out.
4. q ≥ 29 was not run, because the sieve to 6.5e9 exceeded the memory budget.
5. No Lean.
6. The (g, h) distribution over doubly deleted pairs is described only structurally.

B. RUN LAWS FOR UPPER GEARS ON SURVIVORS

Standing:

B1 (Distance law, PROVED from the centre law)
- If a prime g ≥ 7 strikes x and x+D, then g ∣ D(15D-1)(15D+1).
- D is blocked at q when D(225D²-1) is q-smooth. D = 1 is always blocked.
- Blocked D ≤ 120:

| q | Blocked D |
|---|---|
| 23 | {1, 3, 8} |
| 29 | {1, 3, 8, 85} |
| 31 | {1, 2, 3, 8, 33, 85} |
| 37, 41 | {1, 2, 3, 5, 8, 33, 85, 111} |
| 43 | adds 20 and 43 |
| 47 | adds 25 and 69 |

B2 (PROVED)
- (a) Triple law: if one gear strikes three copies within a span below 2g, two of them are exactly g apart and the third sits at ±t_g (mod g) from them.
- (b) c_g(L) = 2⌊L/g⌋ + min(L mod g, 1) + [L mod g > σ_g] is the largest number of g-teeth in L consecutive copies. It gives n ≤ Σ c_{g_j}(W+1) ≤ 2k⌈(W+1)/q⁺⌉, with W ≤ Γ_q(n).
- (c) If Γ_q(2k+1) < q⁺, no k upper gears strike 2k+1 consecutive survivors.
- (d) τ_g(N) = ⌊(N-1)/2⌋·g + [N even]·σ_g is the smallest span of N teeth.
- (a), c_g and τ_g are exact residue laws. The bounds in (b) and (c) are budget inequalities.

B3 (Exact values)
- Γ_q(5) = 10, 13, 19, 28, 36, 41, 47 at q = 11..31. The inner and cyclic values agree.
- Γ_37(2..5) = 39, 44, 52, 60 (DFS).
- Γ_41(2) = 50.
- Γ_29(9..12) = 54, 59, 63, 67.
- Γ_31(9..12) = 67, 69, 74, 78.

B4 (Measured, exhaustive)
- At q = 11..31, no two upper gears strike 5 consecutive survivors. The mechanism at each q:
  - q = 11, 13: Γ < q⁺.
  - q = 17, 19: no g-triple shape occurs in any 5-run.
  - q = 23: 14 g = 29 placements, none aligned.
  - q = 29: 4 aligned 5-runs, all with g = 31. Two leave D = 1, which is blocked. In the other two, no upper h completes the run.
  - q = 31: 14 aligned 5-runs. One leaves D = 3, which is blocked. In the other 13, no h completes the run.
- Γ_q(5) < 2q⁺ at every q in 11..31, so the triple law applies.
- Wording correction: the uniform chain "Γ_q(5) ≤ 47 < 2q⁺" is false at q = 11..19.
- The check line Γ_31(5) ≤ Γ_29(11) = 63 bounds a machine by a smaller one and is to be dropped. The direct scan gives Γ_31(5) = 47.

B5 (REFUTED at q = 37: two upper gears never strike 5 consecutive survivors, i.e. the 2k bound at k = 2)
- Five instances, all strikes acting:

| Start | Offsets | Neighbours | Strikers |
|---|---|---|---|
| 2749631283 | {0, 9, 29, 30, 41} | -1, +42 | 41 on 0−, 30+, 41−; 43 on 9−, 29+ |
| 92596865755 | {0, 4, 11, 24, 41} | -3, +46 | 41 on {0, 11, 41}; 43 on {4, 24} |
| 25579147518 | {0, 18, 20, 29, 43} | | 43 on {0, 20, 43}; 83 on {18, 29} |
| 44581037899 | {0, 6, 10, 23, 43} | | 43 on {0, 23, 43}; 61 on {6, 10} |
| 190395494309 | {0, 10, 21, 23, 43} | | 43 on {0, 23, 43}; 41 on {10, 21} |

- In the first instance, gear 23 has teeth at 5, 8, 28, 31.
- The figure "18 instances with g = 41" is an unflagged count and not the total, since g = 43 was not enumerated.
- The totals 21 and B_2(37) = 5 are not recomputed.

B6 (one gear on consecutive survivors)
- No one-gear 3-run exists at q = 11..31.
- At q = 37, exactly one copy in [1, M-1] starts a one-gear 3-run: 93761988719 + {0, 30, 41}, gear 41, legs −, +, −.
  - Γ_37(3) = 44 forces g ∈ {41, 43} and the shapes (0,11,41), (0,30,41), (0,20,43), (0,23,43).
  - These have 8, 8, 1 and 1 covering classes, and exactly one class is aligned.
  - The two g = 43 classes are a mirror pair: 224657416074 + 22700521710 + 43 = M.
- No one-gear 4-run exists at q = 37: Γ_37(4) = 52 = τ_41(4) leaves only (0,11,41,52), which has 0 coverings.
- REFUTED at q = 47 (the 2k bound at k = 1): gear 53 strikes the 4 consecutive survivors 247507649335490 + {0, 7, 53, 60}, legs −, +, −, +, neighbours -4 and +63, σ_53 = 7.
- The figure "428 coverings" is exact but is an unflagged count.

B7 (REFUTED: 2k bound at k = 3; three gears strike 7 consecutive survivors at q = 37)
- 185244748079 + {0, 3, 14, 16, 27, 44, 49}: 41 on {3, 14, 44}, 241 on {0, 16}, 331 on {27, 49}.
- 238876916838 + {0, 1, 12, 20, 29, 42, 43}: 43 on {0, 20, 43}, 41 on {1, 12, 42}, 1487 on {29}.

B8 (Alignment lemma, PROVED by a counting argument)
- With the other free gears fixed, at least ⌊R/G⌋ - 2n(k+1) - 2 values of v give copies whose survivors in the window are exactly s+K, each struck by its assigned g_j.
- Two conditions are implicit and hold in every use: every free gear is ≤ q < every upper gear, and the window contains offset 0.
- Flagged test: 69 cases at q = 29, 31 with k = 1, 2, 0 violations, minimum slack 3.

B9 (Acting pair law, exact, PROVED)
- For prime g ≥ 7, j ≥ 1 and 0 < D < g: g strikes and acts on both j and j+D iff D ∈ {σ_g, g-σ_g}, g ∣ 2j+D, and 30j+1 ≥ g².
- Leg form: equivalently 30j+1 ≥ g² and either [g ∣ 30j-1 and D = g - t_g] or [g ∣ 30j+1 and D = t_g]. j+D is struck on the opposite leg.
- The literal ⇐ direction is REFUTED: g = 41, j = 2749631283, D = 11 (j+11 is not struck).
- Defence instances (g, j, D): (13, 16, 6), (17, 13, 9), (19, 83, 14), (23, 56, 20), (29, 86, 27).

B10 (PROVED)
- (a) Acting is monotone: from h_g = ⌈(g²-1)/30⌉ upward.
- (c) n ≤ Σ c_{g_j}(s_n - max(s_1, h_{g_j}) + 1). This is budget-type.
- (d) Every instance above has every strike acting. The lowest instance is at 2749631283.
- Alignment copies with v ≥ 1 lie at height ≥ M / ∏r_i.

Acting in this section:
- No statement uses acting essentially. Acting enters only through the pointwise threshold h_g or as a pointwise check on instances.
- Counting flags:
  - B8, and the bounds in B2(b), B2(c) and B10(c), are counting- or budget-type.
  - "18 instances", "428 coverings" and "3 of 18" are unflagged counts and not load-bearing.
- Locating: none load-bearing. The Γ_29(11) check line is to be dropped.

Not established or unfinished:
- Whether B_1(q) is bounded, and whether any bound in k alone exists. The only all-N route found is a density argument and is not claimed.
- N = 5 for one gear: the DFS is too slow beyond q = 47.
- Two gears on 6 consecutive survivors at q = 37: partial, the rest timed out.
- Γ_37(5) (not recomputed), and Γ_41(k) for k ≥ 3.
- The first machine with a one-gear 4-run: 41, 43 or 47. q = 41 and 43 were only partly checked.
- The q = 37 two-gear 5-run enumeration is partial: stopped at e = 52, and g = 43, 47 were not run.

C. MY RE-CHECKS (bounded, foreground, all finished; nothing under C:/dev/primes modified)

Scripts are in C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/record0925/.

- mirror_check.py (q = 7..23, sieve to q#+1, 72 s at q = 23). Everything matched:
  - A1, A2 and the A6 identity: 0 mismatches.
  - The A3 tallies, the A6 series, the A8 acting-both sets and pair counts, the A9 core values and d*, and the A11 class sets with 0 violations.
  - The factorisations for A5, A7, A8 and A10.
- mirror_check2.py: established what the 0, 0, 8, 56, 724, 9731 series in A7 counts (the doubly deleted restriction).
- band_check.py: the q ≤ 3000 square test gives only (7, q#-2, e = 5, g = 8). The existence clause holds at q = 11..79, and matches the stated q = 29 and q = 53 examples.
- runs_check.py. Everything matched:
  - Distance law: 1290 struck pairs, 0 violations.
  - Blocked D lists at q = 23..47.
  - Tooth law σ = (κg±1)/15 for primes up to 20000: 0 violations.
  - c_g and τ_g formulas: 0 mismatches.
  - Acting pair law: 6,264,960 triples, 0 mismatches, and 0 in the leg form.
  - All nine q = 37 / q = 47 instances: the hull survivors are exactly the stated offsets, the neighbours match, and the stated strikers act.
  - Gear 23 teeth, and the M-sum of the mirror pair.
  - Γ_q(5) = 10, 13, 19, 28, 36, 41 at q = 11..29.
- Not re-checked: q = 31 scans, the covering DFS counts (8, 8, 1, 1; 428; Γ_37; Γ_41), and the alignment-lemma test.
- Processes: I left none running. There are 8 older python processes, started 22 to 25 Sep with working sets under 1.3 MB. They are not from these runs and I left them untouched.
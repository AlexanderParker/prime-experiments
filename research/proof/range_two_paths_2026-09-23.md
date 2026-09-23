# Record: Path A and Path B, seven angles (assembled 2026-09-23)

Each check is marked with where it came from:
- **[R]** I reproduced it with my own foreground scripts. They are in `C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/rec2309/`: v1_escape.py, v1b_r12_r14.py, v2_inert.py, v3_silent.py, v4_neigh.py, v5_translate.py, v6_lowersets.py, v6b_lap0.py, v7_stab.py, v8_misc.py, v9_inst.py.
- **[C]** Carried from the adjudicator's check. I did not re-run it.

The proofs are the adjudicated ones. I re-ran computations, not proofs.

**Notation**
- Case 1: g ≡ ±1, ±11 mod 30, with (A, B) = (28, 30) and r_1 = 29.
- Case 19: g ≡ ±7, ±13 mod 30, with (A, B) = (10, 12) and r_19 = 11.
- c_g = copy J_g = ceil((g² + 1)/30), with legs g² + A and g² + B.
- t_h = 15⁻¹ mod h; chi(x) = [x² ≡ 19 mod 30]; g' is the next gear and d = g' − g.

**Script locations**, all under the scratchpad folder:
- Escape angle: adjudicator adjX_pathA_esc/ (a1_tables, a2_join_bands, a3_r11, a4_r12_r14, a5_r14); worker escA/ (t1_tables, t2_escape, t3_join).
- Inert angle: adjudicator adjA/ (a1_inert, a2_actual, a3_dual, a4_nth, a5_period, a6_ell, a7_pell, a8_classes); worker top level (inert1, actual2, run3, motion4, copies5, fill6).
- Stretch angle: adjudicator adjA/ (a1_lengths, a2_direct, a3_nth, a4_classes, a5_cover_empty, a6_iv_wording); worker top level (pathA_1..pathA_6).
- Neighbours angle: adjudicator adj/ (s1_rawkills, s2_r1_r6, s2b_poly, s3_r7, s4_r8, s5_runs).
- Translation angle: adjudicator adj/ (s1_acting, s2_translate, s3_acting_breaks, s4_range_reach, s5_revealed_counts, s6_s1s2s3_orbit, s7_transfer, s8_orbit67); worker pathb/ (a_acting, a2_inequalities, b_instances, c_target_modulus, d_window_sources, e_shift, f_orbit_detail).
- Rungs angle: adjudicator adjB/ (a1_struct, a2_laps, a3_xfit, a4_mean, a5_acting, a6_gaps, a7_g29, a8_fits, a9_top, a10_gap); worker pathb/ (a_structure, b_acting, c_scaling).
- Stabiliser angle: adjudicator adjB/ (a1_stab, a2_force, a3_pairs, a4_window, a5_tie); worker top level (pb_offsets, pb_transfer, pb_bounds, pb_fibre).

---

## 1. PATH A: statements that stand

### A1. Escape systems of the upper gears

**A1.1 Strike criterion**
- 30·J_g = g² + r_c.
- A gear h < g strikes c_g iff g mod h lies in E_h^(c):
  - E_h^(1) = the unit roots of x² ≡ −28 or −30 (mod h).
  - E_h^(19) = the unit roots of x² ≡ −10 or −12 (mod h).
- The four targets are distinct mod every h ≥ 7: their pairwise differences are 2, 16, 18, 20.
- No h divides both legs of one copy. E_7^(1) = {}.
- **[R]** Gears 7..30000; 10029 strike pairs with h < g ≤ 30000; 0 mismatches; 0 both-leg strikes.

**A1.2 Sizes and residue rules**
- e_h^(1) = 2 + (h/7) + chi₋₃₀(h) for h ≥ 11, and e_7^(1) = 0.
- e_h^(19) = 2 + chi₋₁₀(h) + (h/3).
- Residue rules:
  - (−28/h) = (h/7).
  - (−12/h) = +1 iff h ≡ 1 mod 3.
  - (−10/h) = +1 iff h mod 40 ∈ S40 = {1,7,9,11,13,19,23,37}.
  - (−30/h) = +1 iff h mod 120 ∈ S120 = {1,11,13,17,23,29,31,37,43,47,49,59,67,79,101,113}.
- e19 = 4 iff h mod 120 ∈ {1,7,13,19,37,49,91,103}; e19 = 0 iff h mod 120 ∈ {17,29,71,83,101,107,113,119}; e19 = 2 on the other 16 unit classes.
- **[R]** Formulas checked against brute force for h ≤ 30000. The rules match Euler's criterion on all 78494 primes in [11, 10⁶], with 0 mismatches.

Exclusion table by h (e1, e19) **[R]**:

| h | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 | 59 | 61 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 0 | 4 | 2 | 2 | 0 | 4 | 4 | 2 | 4 | 0 | 4 | 2 | 2 | 2 | 0 |
| e19 | 4 | 2 | 4 | 0 | 4 | 2 | 0 | 2 | 4 | 2 | 2 | 2 | 2 | 2 | 2 |

Sample sets **[R]**:
- E_7^(19) = {2,3,4,5}; E_11^(1) = {4,5,6,7}; E_11^(19) = {1,10}; E_13^(1) = {3,10}; E_13^(19) = {1,4,9,12}.
- Classes free in both cases: ±1 at 7; ±2, ±3 at 11; ±2, ±5, ±6 at 13.

**A1.3 Independence of the four characters**
- chi₋₇, chi₋₃₀, chi₋₁₀, chi₋₃ are independent. The 15 nonempty products have square classes −7, −30, −10, −3, 210, 70, 21, 3, 10, 30, −21, −70, −210, −1, 7; none of them is 1.
- Each (chi7, chi30) pattern covers 48 of the 192 units mod 840. Each (chi10, chi3) pattern covers 8 of the 32 units mod 120. All 16 joint patterns cover 12 classes mod 840.
- Inert in both cases: {83,227,311,467,479,551,563,587,671,719,803,839} mod 840.
- Inert gears found by brute force:
  - Case 1, up to 200: 7,19,41,61,73,83,89,97,103,139,173,181.
  - Case 19, up to 200: 17,29,71,83,101,107,113,137,149,191.
  - Both, up to 1000: 83,227,311,467,479,563,587,719,839.
- **[R]**

**A1.4 Disjointness and free count**
- E^(1) ∩ E^(19) = ∅.
- The free count h − 1 − e1 − e19 is at least h − 9. Equality holds first at h = 37, then 277, 373, 613, 757, 877.
- The free count is at least 2 at every gear (it is 2 at h = 7).
- **[R]** h ≤ 30000.

**A1.5 Image of J_g mod h**
- The image is {j : 30j − r_c is a nonzero square mod h}. It has (h−1)/2 elements, and each fibre is {±g}.
- a_h is in the image iff (−A/h) = 1; −a_h is in the image iff (−B/h) = 1.
- e_h = 2·#({a_h, −a_h} ∩ image).
- **[C]** All h ≤ 400.

**A1.6 Dual rule**
- ±x (with x < h/2) lies in E_h^(c) iff h is a prime factor of x² + A_c or x² + B_c and h > 2x.
- Lists for x = 1..8:

| x | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| case 1 | {29,31} | {17} | {13,37} | {11,23} | {11,53} | {} | {79} | {23,47} |
| case 19 | {11,13} | {7} | {7,19} | {13} | {37} | {23} | {59,61} | {19,37} |

- **[R]** x ≤ 60, h < 5000.

**A1.7 Join at residue level**
- At level X there are 4·∏_{7≤h≤X}(h − 1 − e_h) escape classes per case, mod 30·∏h. They are spread equally over the four case residues mod 30.
- Passing row h refines each class (h − 1 − e_h)-to-1. No class is lost.
- Class counts at M = 210, 2310, 30030, 510510:
  - Case 1: 24, 144, 1440, 20160.
  - Case 19: 8, 64, 512, 8192.
- Children per parent: 6, 10, 14 in case 1; 8, 8, 16 in case 19.
- **[R]**

**A1.8 Join at gear level**
- Write g = kh + ρ with |ρ| < h/2. If h strikes leg L, then ρ² + L = mh with m ≥ 1. For odd k, ρ is even and m ≡ L mod 4.
- Row h is silent if ρ² < h − B (any k). For odd k it is also silent if ρ² < 2h − B', where B'_1 = 30 and B'_19 = 10.
- Minimal m: 1 on every leg for even k; for odd k, 2 on legs 30 and 10, and 4 on legs 28 and 12.
- Both inequalities are strict and both are attained: 652 pairs have ρ² = h − B, and 1623 odd-k pairs have ρ² = 2h − B'.
- First instance (g, h, ρ, k, m) among pairs attaining the minimal m:

| | leg 10 | leg 12 | leg 28 | leg 30 |
|---|---|---|---|---|
| even k | (23,11,1,2,1) | (53,13,1,4,1) | (59,29,1,2,1) | (61,31,−1,2,1) |
| odd k | (17,13,4,1,2) | (83,67,16,1,4) | (29,11,−4,3,4) | (19,17,2,1,2) |

- These are first among pairs at the minimal m. The first strike pairs overall on even-k legs 12 and 30 are (17,7,3,2,3) and (29,13,3,2,3).
- **[R]** All 10029 strike pairs.

**A1.9 Dormancy (reading "within" strictly)**
- Row h is silent on every g with |g − kh| < √(h − B), and with |g − kh| < √(2h − B') when k is odd. In particular it is silent on the open band (h, h + √(2h − B')).
- In each block [kh, (k+1)h) the odd candidates are kh + r with r ∈ E_h and r ≡ k+1 mod 2. There are exactly e_h/2 of them, one per pair ±r.
- For h to strike c_{h'} (h' the next gear) needs (h' − h)² ≥ 2h − B'.
- Consecutive gears up to 10⁷ (664575 pairs, maximum gap 154) give exactly two strikes:
  - 13 strikes c_17 (case 19, leg 299 = 13·23).
  - 17 strikes c_19 (case 1, leg 391 = 17·23).
  - 0 violations of the next-gear bound.
- 1049 distinct rows strike some c_g with g ≤ 30000; that is 1349 (row, case) pairs.
- **[R]** Consecutive pairs and row counts. **[C]** Block/parity rule for h ≤ 300, k < 40.

**A1.10 Escape classes are nonempty**
- For each case and finite row set H there are 4·∏(h − 1 − e_h) escape classes mod 30·∏H. Each is a unit class, so by Dirichlet it holds infinitely many gears.
- Every row of H acts on each such gear g > max H, and none of them strikes it.
- **[C]** Nonempty up to level 17.

**A1.11 R11**
- (a) At residue level, row exclusions are complete classes r mod h with r ∈ E_h. At gear level a row acts only on g > h. Example: 7 mod 59 in case 19, where 59 = 7² + 10.
- (b) Every unit class c mod M holds infinitely many gears whose c_g is struck by an acting row h < g. With the case fixed, the class must be case-compatible.
- (c′) Fix a case and a class c mod M that holds infinitely many gears of that case. Equivalently, gcd(c, M) = 1 and c mod gcd(M, 30) reduces a case residue. Then: every case gear of the class above some point escapes every row h ≤ X iff every non-inert row h ≤ X divides M and c mod h ∉ E_h.
- Level-13 measurement:
  - 1952 escape classes mod 30030 (1440 + 512).
  - In 1821 classes the first gear with c_g struck is the class's first gear; 121 classes have one revealed gear before it; 10 have two.
  - The largest first struck gear is 544501 (class 3961, prime).
- **[R]** Level-13 figures. **[C]** Each of the 5760 unit classes mod 30030 holds at least 4 struck gears ≤ 2·10⁶. The (c′) mechanism check covered 3904 (class, row ∈ {17,19,23}) pairs, all resolved below 6·10⁷; the largest is 4191739 when any r ∈ E_h is allowed.

**A1.12 R12′**
- Definitions:
  - F_q = {g ≥ 7 : g² + A_g > q, g² + B_g ≤ q#}.
  - T_q = {g ∈ F_q : g² + A_g > isqrt(q#)}.
  - L_q = F_q \ T_q.
- (i) With all primes ≥ 7 as rows and no cutoff, every c_g is covered. The blame is b(g) = lpf(g² + A_g), which is never g and never below 7.
- (ii) Acting on c_g ⟺ h ≤ g ⟺ h < g.
- (iii) Rows 7..isqrt(q#) with no cutoff leave uncovered exactly {g ∈ T_q : both legs prime}. This is the same set the actual system (rows h < g) leaves uncovered on T_q. These rows cover all of L_q through leg A.
- (iv) On T_q the cutoff h < g and the ceiling isqrt(q#) are interchangeable; on L_q a non-cover rests on the cutoff. The ceiling is the acting bound at the top copy (30j + 1 = q# − 29); no gear square lies in (q# − 29, q#].
- Figures for q = 7, 11, 13, 17, 19, 23, 29, 31:

| q | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| \|F_q\| | 3 | 12 | 37 | 124 | 440 | 1745 | 7870 | 37542 |
| \|L_q\| | 0 | 0 | 2 | 6 | 13 | 27 | 58 | 118 |
| ceiling-uncovered | 3 | 3 | 6 | 10 | 19 | 62 | 168 | 585 |
| cutoff-uncovered (both legs prime) | 3 | 3 | 8 | 13 | 22 | 68 | 179 | 598 |

- Ceiling-uncovered sets: at q = 13 they are 13, 79, 89, 97, 127, 131; at q = 29 they start 409, 541, 739, 1427, 1669, 1879; at q = 31 they start 739.
- The cutoff-uncovered part of L_q grows from [7, 11] at q = 13 to [7,11,13,79,89,97,127,131,223,241,251,409,541] at q = 31.
- **[R]** Raw divisibility of both legs by every prime 7..isqrt(q#), up to 447839 at q = 31.

**A1.13 Range membership of c_g**
- c_g lies in machine q's range iff q − A_g < g² ≤ q# − B_g.
- The family is every gear in (q, √q#] together with {g ≤ q : g² + A_g > q}.
- q# − 11 is ≡ 3 mod 4 and q# − 29 is a non-residue mod 7, so neither is a square for q ≥ 7.
- For a family gear g ≤ q, every row acting on c_g is a machine gear.
- **[R]** q ≤ 31. **[C]** q ≤ 3000.

**A1.14 R14′**
- Target: for every prime q ≥ 7, the rows h < g leave some g ∈ F_q with both legs of c_g prime.
- (i) Measured coverage:
  - The target holds at q = 7..31 over the whole family.
  - It holds at every prime q ∈ [7, 39377242978] through the 328 revealed gears below 200000 (233 in case 1, 95 in case 19). The largest is 198437, with legs 39377242979 and 39377242981.
  - Witness counts at q = 7..29 are 3, 3, 8, 13, 22, 68, 179; the least witness is always 7.
- (ii) Hand-off equivalence:
  - The target holds for all q iff the revealed gears 7 = g_0 < g_1 < … form an infinite sequence with g_{i+1}² + B ≤ (g_i² + A)# for every i.
  - The first interval [7, 59) is served by g_0, since 61 ≤ 210.
  - All 327 hand-offs below 200000 hold: every lower leg is ≥ 59, and 59# = 1922760350154212639070 exceeds every upper leg.
- (iii) The target implies the range statement and infinitely many revealed c_g.
- (iv) The domain is q ≥ 7.
- **[R]**

### A2. Inert gears and the threat set

**A2.1 Reach at c_g**
- Case 1:
  - h reaches leg A iff h ≡ 1, 2, 4 mod 7; h = 7 never does.
  - h reaches leg B iff h mod 120 ∈ S120.
  - h is inert iff h = 7, or h ≡ 3, 5, 6 mod 7 with h mod 120 ∈ N120 = {7,19,41,53,61,71,73,77,83,89,91,97,103,107,109,119}.
- Case 19:
  - Leg A iff h mod 40 ∈ S40; leg B iff h ≡ 1 mod 3.
- Each state is exactly a quarter of the unit classes.

Case 1 state lists mod 840 (48 classes each):
- **Inert:** [19,41,61,73,83,89,97,103,139,173,181,209,223,227,229,293,311,313,349,367,433,437,451,467,479,521,551,563,577,587,607,619,661,671,677,689,691,703,719,727,761,773,797,803,811,817,829,839]
- **A-only:** [53,71,107,109,127,191,193,197,211,239,247,281,317,323,331,337,347,359,379,401,421,431,443,449,457,463,487,499,533,541,557,569,571,583,589,599,641,653,673,683,697,709,739,781,793,809,823,827]
- **B-only:** [13,17,31,47,59,101,131,143,157,167,187,199,241,251,257,269,271,283,299,307,341,353,377,383,391,397,409,419,439,461,481,493,503,509,517,523,559,593,601,629,643,647,649,713,731,733,769,787]
- **Both:** [1,11,23,29,37,43,67,79,113,121,137,149,151,163,169,179,221,233,253,263,277,289,319,361,373,389,403,407,473,491,527,529,547,611,613,617,631,659,667,701,737,743,751,757,767,779,799,821]

Case 19 state lists mod 120:
- Inert {17,29,71,83,101,107,113,119}.
- A-only {11,23,41,47,53,59,77,89}.
- B-only {31,43,61,67,73,79,97,109}.
- Both {1,7,13,19,37,49,91,103}.

Removed classes of g mod h (leg A | leg B):
- Case 1: 11 → {4,7}|{5,6}; 13 → {}|{3,10}; 23 → {8,15}|{4,19}.
- Case 19: 7 → {2,5}|{3,4}; 11 → {1,10}|{}; 13 → {4,9}|{1,12}.

Checks:
- **[R]** I compared the literal lists with the character rules and they are identical. That closes the adjudicator's caveat that they lacked the worker's literal lists. Raw states for all primes 11..60000 match the class rule.
- **[C]** No actual strike falls on an unreachable leg for g < 200000. Six predicted triples are unrealised below 200000; they are realised beyond it (fill6.py).

**A2.2 Factor pools and the repaired general clause**
- Factor pools: prime factors of g² + 28 are ≡ 1, 2, 4 mod 7; of g² + 30 lie in S120 mod 120; of g² + 10 lie in S40 mod 40; of g² + 12 are ≡ 1 mod 3.
- Repaired general clause: on the n-th copy, a prime factor p of a leg g² + m has (−m/p) = 1 iff p ≠ g. p = g occurs exactly on the legs g strikes, so the clause holds throughout the silent run n ≤ p1(g).
- **[R]** 330 failures for g < 1000 and n ≤ g+1; all have p = g and none has n ≤ p1(g). **[C]** Factor pools for g < 10000.

**A2.3 Joint structure**
- (−30/h) = (−1/h)(−10/h)(−3/h).
- The span of −7, −10, −3 has 8 elements and does not contain −30.
- The jointly inert classes are exactly: h ≡ 3 mod 4, h ≡ 3, 5, 6 mod 7, h ≡ 2 mod 3, and (−10/h) = −1.
- 7 is the only gear that divides one of 28, 30, 10, 12.
- **[R]**

**A2.4 The n-th copy**
- Copy J_g + n − 1 has legs g² + 2K_n and g² + 2K_n + 2, with K_n = 15n − 1 (case 1) or 15n − 10 (case 19).
- h reaches leg A iff (−2K_n/h) = 1, and leg B iff (−(2K_n+2)/h) = 1. The state has period h in n.
- The case-19 state at n equals the case-1 state at n − 3·5⁻¹ mod h.
- Inert_c(n) = {h ∤ K_n(K_n+1) : both symbols −1} ∪ {h | K_n : (−(2K_n+2)/h) = −1} ∪ {h | K_n+1 : (−2K_n/h) = −1}.
- Square classes for n = 1..8:
  - Case 1: (−7,−30), (−58,−15), (−22,−10), (−118,−30), (−37,−6), (−178,−5), (−13,−210), (−238,−15).
  - Case 19: (−10,−3), (−10,−42), (−70,−2), (−1,−102), (−130,−33), (−10,−2), (−190,−3), (−55,−222).
- **[R]** Leg identities. **[C]** Stretch strikes for g < 6000; the shift; classes n ≤ 8.

**A2.5 Period counts**
- Over n = 1..h, with a = (2/h) and b = (−2/h):
  - inert (h+1−a−b)/4;
  - A-only (h+1+a+b)/4;
  - B-only (h+1+a+b)/4;
  - both (h−3−a−b)/4.
- All four states occur for every h ≥ 7.
- **[R]** Primes 7..5000, both cases.

**A2.6 Length of the inert run, ell**
- No gear is inert at every copy, and ell + 1 ≤ h.
- Measured over the 17981 gears h ≤ 200000:
  - Maximum ell_1 = 8, at 14951 and 53231.
  - Maximum ell_19 = 13, at 38639 only.
  - Tallies, case 1: {0:13467, 1:3419, 2:829, 3:135, 4:106, 5:14, 6:7, 7:2, 8:2}.
  - Tallies, case 19: {0:13507, 1:2252, 2:1664, 3:274, 4:217, 6:41, 7:18, 9:4, 10:3, 13:1}.
- **[R]**

**A2.7 Recurrence families of the inert set**
- Case 1:
  - n = s², with 15s² − 14r² = 1; the legs are g² + 28r² and g² + 30s².
  - Members n = 1, 3249, 10923025, 36723206689, from (s, r) = (1,1), (57,59), (3305,3421), (191633,198359).
  - Step: (s, r) → (29s + 28r, 30s + 29r).
- Case 19:
  - n = (s² + 2)/3, with 5s² − 6y² = −1; the legs are g² + 10s² and g² + 12y².
  - Members n = 1, 177, 85009, 40973857, from (s, y) = (1,1), (23,21), (505,461), (11087,10121).
  - Step: (s, y) → (11s + 12y, 10s + 11y).
- Repaired gain rule: Inert(n) = Inert(1) ∪ {h | r : h A-only at n = 1} ∪ {h | s : h B-only at n = 1}. In case 19, s sits on leg A and y on leg B.
- Observed changes:
  - n = 3249: none.
  - n = 10923025: 11 goes from both to B-only.
  - n = 36723206689: 43 and 659 go from both to B-only; 13 and 14741 go from B-only to inert.
  - n = 177: 7 goes from both to A-only; 23 goes from A-only to inert.
  - n = 85009: none.
  - n = 40973857: 349 and 11087 join.
- First hosts: n = 177 at g = 293 (p1 = 195, Prot = 280); n = 3249 at g = 4111 (p1 = 3836, Prot = 4393).
- **[R]** Recurrences. A direct search n ≤ 2·10⁶ finds only 3249 (case 1) and 177, 85009 (case 19). Hosts. **[C]** Observed changes.

**A2.8 Density law**
- Among non-divisor gears, those inert on copies 1..L have density 2^(−r_L), where r_L is the F2-rank of the square classes of −2K_n and −(2K_n+2) for n ≤ L. Sign lemma: every class inside the span is forced negative.
- Ranks r_1..r_13:
  - Case 1: 2,4,6,7,9,10,11,12,13,14,15,16,17.
  - Case 19: 2,3,5,6,8,8,9,10,10,11,12,12,12.
- Free copies n ≤ 60:
  - Case 1: [15,25,31,33,36,42,43,47,52,59,60].
  - Case 19: [6,9,12,13,19,23,24,30,31,33,38,39,42,43,44,45,48,49,50,57,58,59,60].
- Possible ell values are {L : r_{L+1} > r_L}. For L ≤ 13 that is every L in case 1, and {0,1,2,3,4,6,7,9,10,13} in case 19.
- Flagged count check, predicted vs observed:
  - Case 1: 13485.8/13467, 3371.4/3419, 842.9/829, 140.5/135, 105.4/106.
  - Case 19: 13485.8/13507, 2247.6/2252, 1685.7/1664, 281.0/274, 210.7/217, 35.1/41.
- **[R]** Ranks, free copies, tallies. **[C]** Predictions.

**A2.9 Least non-residue bounds**
- ell_1(h) < n(h), where n(h) is the least non-residue mod h.
- If ell_19 ≥ 4, then h ≡ 7 mod 8, 2, 3, 5, 7 are residues, and ell_19 < (2n(h) + 2)/3.
- n(h) < √h + 1.
- Record holders (h, ell, n(h)):
  - Case 1: (7,2,3), (73,4,5), (1151,5,13), (9719,6,13), (14951,8,19).
  - Case 19: (17,1,3), (101,2,2), (311,3,11), (1511,4,11), (5711,6,19), (10559,7,23), (38639,13,29).
- **[R]** 0 violations; 284 gears have ell_19 ≥ 4.

**A2.10 p1 formulas**
- Case 1: p1 = (kg − 14)/15 = t_g − 1, with k = 14, 4, 11, 1 for g ≡ 1, 11, 19, 29.
- Case 19: p1 = min(−1/3 mod g, −2/5 mod g), which is (g−2)/5 for g ≡ 7, 17; (g−1)/3 for g ≡ 13; (2g−1)/3 for g ≡ 23.
- **[R]** g < 30000.

**A2.11 Whole-run inert pairs**
- The pairs (g, h) with h inert on the whole silent run of c_g are exactly (11,7), (29,7), (29,19), with run lengths 2, 1, 1. There are none in case 19.
- Using min(p1, Prot) gives the same pairs, with lengths 1, 1, 1.
- Proof bound: min(p1, Prot) ≥ (g−14)/15 in case 1 and ≥ (2g−13)/15 in case 19.
- **[R]** g ≤ 200000, plus raw strikes on the three runs.

### A3. The silent run above each square, protected from above

**Formulas** (delta(g) = [g ≡ 7, 17 mod 30]; "acts" means square-eligible):
- Protected length: L = j_{g'} − J_g = (d(2g+d) − 30 + 18chi(g) + 12chi(g'))/30, with j_{g'} = (g'² − 1 + 12chi(g'))/30.
- In case 19 the region is shelf(g). In case 1 it is shelf(g) minus the head copy j_g, whose upper leg is g².
- First cofactor: h1 = g + 2s, with s = 14, 3, 4, 5, 3, 11, 10, 1 for g ≡ 1, 7, 11, 13, 17, 19, 23, 29 mod 30. The leg g·h1 is the upper leg iff g ≡ 7, 17.
- Silent run: p1 = (sg − c)/15, with c = 14 (case 1), 6 (g ≡ 7, 17), 5 (g ≡ 13, 23). By class:

| g mod 30 | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|
| p1 | (14g−14)/15 | (g−2)/5 | (4g−14)/15 | (g−1)/3 | (g−2)/5 | (11g−14)/15 | (2g−1)/3 | (g−14)/15 |

- Difference: L − p1 = (g'² − g·h1 − 2 + 12chi(g') + 2delta)/30 = (2g(d−s) + d² − 2 + 12chi(g') + 2delta)/30. The term −2 + 12chi(g') + 2delta takes exactly the values {−2, 0, 10, 12}.

**Gap conditions**
- g + s mod 30 takes the values 15, 10, 15, 18, 20, 0, 3, 0 by class, never a unit, so d = s never occurs.
- L > p1 ⟺ d > s ⟺ g'² > g·h1 ⟺ g' > (g + h1)/2.
- L < p1 ⟺ d < s, with one exception: g = 13 (d = 4, s = 5), where L = p1 = 4.
- The (class, d) pairs with d < s: class 1: d = 6, 10, 12; class 11: 2; class 13: 4; class 17: 2; class 19: 4, 10; class 23: 6, 8. None for classes 7 or 29.
- Case 1: g strikes nothing on shelf(g) except the head copy iff d < s.
- Case 19: g strikes nothing on shelf(g) iff d < s, i.e. iff (g, g+s−1] holds a prime.

**Stretch l = min(p1, L)**
- l ≤ 2 exactly at g = 7, 11, 17, 29, with (p1, L, l) = (1,2,1), (2,1,1), (3,2,2), (1,3,1).
- l ≥ 3 for every g ≥ 31; l = 3 first at g = 59.
- **[R]** All 664576 gears 7..10⁷, plus a direct p1 scan for g ≤ 10⁴.

**A3.1 n-th law on the protected region**
- g strikes copy J_g + n iff n ≡ x_g + t_g (lower leg) or n ≡ x_g (upper leg) mod g, with x_g = −1 in case 1 and −2·5⁻¹ in case 19.
- The eligible gears on every copy of the region are exactly the primes in [7, g].
- A gear h < g strikes copy J_g + n iff g² ≡ −(A+30n) or −(B+30n) mod h. The number of unit classes of g excluded at height n is 2[(−(A+30n)/h) = 1] + 2[(−(B+30n)/h) = 1].
- Linear forms:
  - (−(28+30n)/h) = (−2/h)((15n+14)/h)
  - (−(30+30n)/h) = (−30/h)((n+1)/h)
  - (−(10+30n)/h) = (−10/h)((3n+1)/h)
  - (−(12+30n)/h) = (−6/h)((5n+2)/h)
- A copy of the region is revealed iff both legs are prime.
- **[C]** Gears ≤ 10⁴: 3,337,391 copies, 146,694 revealed.

**A3.2 Parabolas, meets and inert heights**
- With λ_h(g) = −(g² + A)·30⁻¹ mod h: h strikes the lower leg of J_g + n iff n ≡ λ_h(g), and the upper leg iff n ≡ λ_h(g) − t_h (mod h).
- Meets: E_n and E_{n+D} share exactly the nonzero roots of −(B+30n) when 15D ≡ 1, those of −(A+30n) when 15D ≡ −1, and nothing otherwise.
- Inert heights per period: (h−1)/4, (h+1)/4, (h+3)/4, (h+1)/4 for h ≡ 1, 3, 5, 7 mod 8, the same in both cases.
- E^19_n = E^1_{n − 3·5⁻¹}.
- Gear-7 table, case 1 (lower / upper classes):

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| lower | – | – | – | ±1 | – | ±2 | ±3 |
| upper | – | – | ±1 | – | ±2 | ±3 | – |

- Case 19 is this table shifted by 2 (E19_0 = E1_5, E19_2 = E1_0).
- **[R]** Parabola rule for all h ≤ g ≤ 1500; inert heights for 7..2000; the tables. **[C]** Meets for h ≤ 600.

**A3.3 Leg rule and single-gear covers**
- h strikes two copies D apart iff h | D or h | 15D ± 1.
- So consecutive copies force h = 7, and copies two apart force h ∈ {29, 31}. No single gear covers a stretch with l ≥ 3.
- Covers occur exactly at:
  - g = 17 by h = 7 (301 = 7·43, 329 = 7·47);
  - g = 29 by h = 11 and h = 13 (869 = 11·79, 871 = 13·67).
- **[R]** Raw check, g ≤ 2·10⁵.

**A3.4 Empty stretches (measurement, g ≤ 10⁶)**
- Head stretches with no revealed copy, as (g, d, p1, L, l): (17,2,3,2,2), (29,2,1,3,1), (37,4,7,10,7), (41,2,10,5,5), (149,2,9,19,9).
- Protected regions with no revealed copy: g = 17, 29, 41.
- g = 37: copy 54 = (1619, 1621) is revealed; copy 53 has 1591 = 37·43. For g = 149 the first revealed copy in the region is 754.
- **[R]**

### A4. Neighbouring gears and each other's missed copies

Here E(h) is the set of first-lap gaps. It is a different object from the escape classes E_h of A1.

**A4.1 (R1)** No prime ≥ 7 divides two of d² + 10, d² + 12, d² + 28, d² + 30, for any integer d. **[C]** d ≤ 200000.

**A4.2 (R2) Kill rule and K(d)**
- h kills c_{h+d} iff h | d² + A', where A' comes from the case of g = h + d: {28, 30} when g ≡ ±1 mod 5, {10, 12} when g ≡ ±2 mod 5.
- K(d) lies in [7, (d² + 30)/2].

| d | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K(d) | {17} | {13} | {} | {23} | {7} | {11,29} | {} | {7,13,67} | {11} | {} | {31} | {293} | {353} | {199} | {7,13,29,31} |

- Kills: 295,748 with 7 ≤ h < g ≤ 10⁶; 6,788 with g ≤ 20000.
- **[R]** K(d) both from raw kills and from the definition. The 295,748 figure is my class count; the 6,788 figure is raw. **[C]** The adjudicator's 3.08·10⁹-pair raw brute force.

**A4.3 (R3) First-lap gaps**
- Let e = e_{A'}(h) be the even root of x² ≡ −A' in [0, h).
- h kills c_g iff d ≡ e or 2h − e mod 2h. For d < h this means d = e exactly.
- E(h) = {e_{A'}(h) > 0 : h + e in the case of A'}, and |E(h)| ≤ 4. The gears in (h, 2h) killed by h are exactly h + E(h).
- Over 7 ≤ h ≤ 400000: |E(h)| distribution {0:13883, 1:13965, 2:5082, 3:863, 4:64}; first-lap kills per h {0:28574, 1:4961, 2:313, 3:9}.
- **[C]**

**A4.4 (R4) The cofactor kappa**
- g² + A' = h(h + 2d) + (d² + A'), so kappa = (d² + A')/h = h + m − 2g.
- 2-adic class:
  - v2(kappa) = 1 for A' = 10 or 30.
  - A' = 12: v2 = 2 if 4 | d, and 4 if d ≡ 2 mod 4.
  - A' = 28: v2 = 2 if 4 | d, and ≥ 5 if d ≡ 2 mod 4.
- v3 = 1 iff A' ∈ {12, 30} and 3 | d; v5 = 1 iff A' ∈ {10, 30} and 5 | d.
- **[C]**

**A4.5 (R5) Distance law**
- Kappa floors: 2 (F10, F30), 4 (F12e, F28e), 16 (F12o), 32 (F28o).
- Hence d² ≥ 2h − 30. On the g side, (d+1)² ≥ 2g − 29 in case 1 and ≥ 2g − 9 in case 19.
- For copies above c_g: (d+1)² ≥ 2g + 1 − A'', with A'' = A' + 30M.
- Floor kills up to 10⁶ by family: F10 11, F30 6, F12e 12, F28e 5, F12o 30, F28o 42.
- F10 equalities: 13→17, 293→317, 3533→3617, 4423→4517, 43223→43517, 85703→86117, 98573→99017, 117133→117617, 164743→165317, 494023→495017, 576743→577817.
- F30 equalities: 17→19, 353→379, 65537→65899, 76847→77239, 282767→283519, 679793→680959.
- **[R]** F10 and F30 lists. **[C]** Other families; the copies-above law for g ≤ 4000.

**A4.6 (R6) Equality**
- g − h ≥ √(2h − 30), with equality iff kappa = 2 and A' = 30 (family F30).
- The case-19 width is sharp exactly on F10; the case-1 width exactly on F30.
- F12e, F28e, F12o, F28o are sharp only for their own kappa floors.

**A4.7 Consecutive gears**
- Bertrand gives d < p, so e² + A' ≥ 2p and e ≥ √(2p − 30).
- The previous gear kills c_next iff the prime gap lies in E(p). E(43) and E(47) are empty.

**A4.8 (R7) Complete lists (h, g) for g ≤ 10⁸**, with h the r-th lower neighbour:
- r=1: (13,17), (17,19)
- r=2: (23,31)
- r=3: (7,17), (29,41), (199,227)
- r=4: (11,23), (13,29), (67,83), (293,317), (353,379)
- r=5: (7,23), (11,29), (31,53)
- r=6: none
- r=7: (29,59), (31,61), (73,107)
- r=8: (7,37), (13,43), (37,71)
- r=9: (11,43), (23,61), (89,137)
- r=10: (13,53), (53,101), (1163,1231)
- r=11: (7,47), (4423,4517)
- r=12: (7,53), (11,59), (149,211), (1447,1523), (3533,3617)

Related figures:
- The largest killed g is 4517.
- Maximum r-step gaps below 10⁸ for r = 1..12: 220, 248, 300, 342, 390, 396, 414, 462, 488, 504, 546, 564.
- Least nearest-killer rank (searched to r = 300):
  - 39 in [10⁵, 10⁶), at 117133→117617 and 164743→165317;
  - 103 in [10⁶, 10⁷), at 1086343→1087817;
  - 275 in [10⁷, 10⁸), at 10783373→10788017.
- Conditional extension to 4·10¹⁸ holds for r ≤ 9: 9·1476 = 13284 < 14141.19. It fails at r = 10: 14760 > 14142.13.
- **[R]** All 5,761,452 gears to 10⁸.

**A4.9 (R8) Every range holds a c_g not killed by its r nearest lower gears (r ≤ 12, every q ≥ 7)**
- q < 59: c_7 = (59, 61). For 11 ≤ q ≤ 139: c_11 = (149, 151). Both are revealed.
- 149 ≤ q < 10¹⁵: any gear in (max(4517, √q), 10⁸]. The first is 4519, with legs 20421389 and 20421391.
- q ≥ 10¹⁵ (repaired argument):
  - Set x = max(q, 10¹⁶) and blocks of length √(2x − 30).
  - Rosser–Schoenfeld gives more than 0.7x/ln 2x gears in (x, 2x] for x ≥ 2.553·10⁸.
  - r_max(x) = (0.7x/ln 2x − 1)/(x/L + 1) is increasing, and r_max(10¹⁶) ≈ 2.637·10⁶.
  - The legs lie in [x² + 10, 4x² + 30] ⊂ (q, q#].
  - With the full RS form, x = max(q, 9.108·10¹⁵) suffices. This part covers r ≤ 2.6·10⁶.
- First silenced runs, h-side reading: r=1 29..31 (machine 11); r=2 37..43 (11); r=3 97..107 (13); r=12 2333..2399 (19).
- g-side reading: the first r=1 run is 11..13 (machine 7, with c_13 = (179, 181) revealed). For r = 2..12 the two readings give the same runs.
- **[R]** c_4519. r_max(10¹⁵) = 888,539; r_max(10¹⁶) = 2,637,438 in my float recomputation (the angle quotes 2,637,440). RS threshold 2.5527·10⁸. **[C]** The runs.

---

## 2. PATH B: statements that stand

### B1. The translation law with acting built in

**Formulas**
- M(j) = ∏{g ≥ 7 : g² ≤ 30j + 1}.
- c(j, e) = j + e·5⁻¹ mod M(j).
- For j ≥ 4: r(j, e) = j + (e + tM)/5 with t = −e·M⁻¹ mod 5, t ∈ {1, 2, 3, 4}.
- legs(r + kM) = legs(j, e) + 6(t + 5k)M.
- A gear g with g² > 30j + 1 strikes r iff 5j + e ≡ ±6⁻¹ − tM (mod g).

**B1.1 Lap-level vs column-exact acting**
- For j ≥ 1 the two acting sets agree at offsets 0 and 2.
- At offset 3 they differ exactly at laps j = (g² − 19)/30 with g a prime gear (g ≡ ±7, ±13 mod 30) and 30j + 17 prime.
- Laps 1..399999: disagreements 0 / 0 / 59 at offsets 0 / 2 / 3. The first ten are laps 1, 5, 45, 61, 73, 353, 381, 537, 997, 1657.
- There are 243 prime-square laps and 219 composite-square laps.
- **[R]**

**B1.2 Translation at the source**
- For g ∈ A(j) and e ∈ {2, 3}: g strikes column 5j + e iff g strikes copy c(j, e).
- **[C]** Laps < 20000.

**B1.3 Least class copy and inequalities (j ≥ 4)**
- r < M(j); r − j ≥ (M + e)/5; 30r + 1 > 6M ≥ G'².
- M(j) > 5j + 3.
- 6M(G) ≥ G'² on every shelf except G = 7.
- M(G) ≥ G'⁴ for G ≥ 19 (base case M(19) = 323323 ≥ 23⁴ = 279841).
- On laps j ≥ 2, the least copy lies below the source only at (2,2) (copy 1) and (3,2) (copy 2). At laps 0 and 1, M = 1 and the least copy is 0.
- **[R]** The inequalities for j < 6000 and shelves ≤ 30000. **[C]** The rest.

**B1.4 What an exposure constrains at r** (exposure (j, e), j ≥ 4, n = 5j + e, both legs L1 = 6n − 1 and L2 = 6n + 1 prime; g with g² > 30j + 1)
- (i) g strikes r iff n ≡ ±6⁻¹ − tM (mod g).
- (ii) For g ∉ {L1, L2}: n is open at g, and the exposure removes at most one of the two strike classes, exactly when 3tM ≡ ±1 (mod g).
- (iii) The source legs:
  - L1 never strikes the lower leg of r, and strikes the upper leg iff L1 | 3tM + 1.
  - L2 never strikes the upper leg, and strikes the lower leg iff L2 | 3tM − 1.
- (iv) Along r + kM, each such g strikes exactly 2 of every g consecutive copies, one per leg.
- (v) Both source legs act at r for every exposure with j ≥ 12. For 4 ≤ j ≤ 11 the exposures are (4,3), (6,2), (6,3), (7,3), (9,2), (10,2), (11,3), and both legs act only at (10,2).
- Instance (10,2): M = 17017, t = 4, r = 13624; legs 408719/408721 ≡ 65/67 mod 311 and ≡ 254/256 mod 313.
- Measured, not proved: for all 187,237 exposures with 4 ≤ j < 2·10⁶ (25,343 with j < 200000), neither source leg strikes r.
- **[R]** (v), the instance and the measurement (cross-checked against big-integer M for j < 3000). **[C]** (i)–(iv) for j < 3000.

**B1.5 Partners of a column**
- With the column-exact M(n): M(n) > n² exactly for n = 28..31 and n ≥ 48.
- For n ≥ 48, every partner n' ≠ n has n'² ≥ n² + M(n); it lies above n and faces a gear outside A(n).
- **[R]** n ≤ 399.

**B1.6 Class copies in the range** (N = q#/30, jmin = ⌊(q+1)/30⌋ + 1)
- For a window lap, the class holds exactly N/M(j) − D copies in the range, with D = ⌈(jmin − c0)/M⌉ when c0 < jmin and 0 otherwise. The least copy is c0 + D·M.
- The count is ≥ 1 on every window lap. Above the window it is 0.
- For j ≥ max(4, jmin): exactly N/M copies, the least being r(j, e).
- D = 1 exactly at:
  - (1,2) and (1,3) for q ≤ 23;
  - (2,2) for 29 ≤ q ≤ 53 (least copy 8);
  - (3,2) for 59 ≤ q ≤ 83 (least copy 9).
- First D > 0 for j ≥ 4: (4,3) at q = 599, (5,3) at 631, (4,2) at 1979, (5,2) at 2011. D = 2 first at (4,3), q = 2909.
- Instances:
  - q = 29, (2,2): count 30,808,062 (N/M = 30,808,063), least copy 8.
  - q = 59, (3,2): count 9,156,001,667,401,012,566, least copy 9.
- **[R]** Instances. Brute force over every range copy at q = 7, 11, 13 found 0 mismatches. The first-D values follow from r(4,3) = 20, r(5,3) = 21, r(4,2) = 66, r(5,2) = 67. **[C]** q ≤ 101.
- The partner condition M(j) ≤ 25(N−1)² − (5j+e)² was carried and never recomputed.

**B1.7 Exact downward transfer**
- From an exposed offset-2/3 column n to a copy j' with 5j' < n and (5j')² ≡ n² mod M(j'): j' is revealed, and M(j') ≤ n² − 25j'².
- The upward pairs (n, 5j') are exactly (2,5), (3,5), (3,10), (3,25), (32,45). The copies j' ≤ 4000 with M(j') ≤ 25j'² are 1, 2, 3, 4, 5, 7, 8, 9.
- For n ≤ 5·10⁶ there are 102,175 exposed columns and 157,062 pairs. Pairs per target copy: {1: 102173, 2: 40822, 5: 9021, 6: 1693, 8: 1657, 9: 1671, 14: 17, 19: 2, 20: 3, 27: 3}. Every target is revealed.
- **[R]**

**B1.8 The range statement at small q**
- Revealed copies in the range at q = 7, 11, 13, 17, 19, 23: 4, 19, 152, 1517, 19017, 298408.
- q = 29..53 hold through copy 2 = (59, 61).
- **[R]**

**B1.9 (13) Orbit measurement, not proved**
- For q = 7..67, every window offset-2/3 exposure has a revealed stabiliser-orbit copy in the range.
- Window exposures per q: 6, 7, 11, 13, 16, 20, 22, 28, 34, 37, 46, 54, 60, 64, 78, 86.
- **[C]**

### B2. General lower sets and the unit twist

**B2.1 Open offsets and their stabiliser**
- |O_y| = ∏(p − 2): 3, 15, 135, 1485 at y = 5, 7, 11, 13.
- 0 ∈ O_y and O_y = −O_y. The residues 0, 12, 18 mod 30 each hold a third.
- The affine stabiliser is exactly {n → un : u ≡ ±1 mod every p in [5, y]}, with no translation.
- **[R]** y = 5, 7, 11: 2, 4, 8 maps.

**B2.2 Blocked laps**
- A gear g > y strikes (j, e) iff j ≡ (±1 − e)·Y⁻¹ mod g.
- **[C]**

**B2.3 Across rungs**
- The twist is conjugation by Q = y'#/y#: Q·z' = z identically, i.e. a change of index under CRT.
- **[C]**

**B2.4 Exposure map**
- The map is c' ≡ ε·c mod M with ε² ≡ 1. A(c') ⊂ A(c) is sufficient for exposure to transfer, and is exactly what the map alone guarantees.
- **[C]**

**B2.5 Transfer cost**
- The minimum over nonzero open offsets is 6k*·M, where k* = min{k : 6kM mod y# ∈ O_y \ {0}}.
- Floor law: every translate spans at least ⌊6k*M/y#⌋ laps, attained by the offset e_down = 6k*M mod y#. At y = 5 the bound is ≥ (M−3)/5, with equality iff M ≡ 3 mod 5.
- The half-lap form holds in centred coordinates and for real sources on their own shelf. At y = 5 the only downward translates to positive centres are 72→30 and 102→60.
- **[R]** All 54 pairs y ∈ {5, 7, 11, 13}, X ≤ 61. **[C]** The real-source scan.

**B2.6 X_fit^floor**
- X_fit^floor(q, y) is the largest X with 6k*·X#/y# < q# − q.
- Rows are q; columns are y = 5, 7, 11, 13, 17, 19. Each cell shows **floor / realised / exposed**. Floor is X_fit^floor. Realised is the largest shelf actually reached by open centres. Exposed is the same over exposed centres only.

| q | y = 5 | y = 7 | y = 11 | y = 13 | y = 17 | y = 19 |
|---|---|---|---|---|---|---|
| 19 | 19 / 19 / 19 | 19 / 19 / 19 | 23 / 23 / 23 | 29 / 23 / 23 | 29 / 19 / 19 | 31 / 23 / 23 |
| 23 | 23 / 23 / 23 | 23 / 23 / 23 | 29 / 29 / 23 | 31 / 23 / 23 | 31 / 23 / 23 | 37 / 29 / 23 |
| 29 | 29 / 29 / 29 | 31 / 31 / 31 | 31 / 29 / 29 | 31 / 29 / 29 | 37 / 31 / 29 | 41 / 29 / 29 |
| 31 | 31 / 31 / 31 | 31 / 31 / 31 | 37 / 37 / 31 | 41 / 37 / 31 | 41 / 37 / 37 | 43 / 37 / 31 |

- Translates shorter than the range at y = 13:

| shelf | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|
| q = 19 | 76 | 16 | 4 | 0 | 0 |
| q = 23 | 1748 | 368 | 28 | 0 | 0 |

- **[R]** The floor column and the y = 13 counts. **[C]** The realised and exposed columns.

**B2.7 Mean and maximum cost**
- Every cost is at most X#/2 − 3M.
- At y = 5: mean = max = k*·X#/5.
- Measured over y < X ≤ 61 (as fractions of X#):

| y | mean | max |
|---|---|---|
| 7 | [0.2286, 0.2980] | [0.4286, 0.4857] |
| 11 | [0.2433, 0.2555] | [0.4883, 0.4987] |
| 13 | [0.2495, 0.2508] | [0.4991, 0.4999] |

- At y = 7, X = 37 and X = 41 give a mean of 8/35.
- **[R]**

**B2.8 log10(C_min/(q# − q)) at X_top**

| (q, y) | (7,5) | (7,7) | (13,13) | (19,13) | (23,13) |
|---|---|---|---|---|---|
| log10 | 1.77 | 0.63 | 60.65 | 1308.42 | 6409.91 |
| X_top | 13 | 13 | 173 | 3109 | 14929 |

- Known-opening copies in the range at q = 23, for y = 5, 7, 11, 13: 7436428, 1062346, 96576, 7428.
- **[R]**

**B2.9 Acting sets within a lap**
- acting(0) = acting(12) in every lap j ≥ 1. Offset 18 differs exactly when 30j + 19 = g².
- A lap splits into exactly 1 + #{primes g > y : Yj + 1 < g² ≤ Yj + y#} acting sets.
- This is proved for j ≥ 1 at 5 ≤ y ≤ 29. Lap 0 was checked directly for y ≤ 23: 1, 3, 11, 35, 121, 436, 1740 sets.
- Exact largest open-offset gaps G_y:

| y | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|
| G_y | 12 | 30 | 42 | 66 | 108 | 150 | 204 | 258 |

- **[R]** Direct counts over laps to 399999 / 59999 / 4999 / 399: 0 failures in the half-open convention; 239 / 81 / 14 / 2 failures in the [Yj, Yj + Y) convention. G_y for y ≤ 29 by lifting.

**B2.10 Fits**
- Translation fits: exactly 42→30, 48→90, 72→30, 102→60, all at y = 5.
- Signed fits add 78→90, 108→60 and 192→270.
- **[R]** Sources ≤ 3000 at y = 5..17. **[C]** Extension to 10⁶.

**B2.11 Wall**
- X'# > nextprime(X')² for X' ≥ 7.
- The single exception is c' = 30 at y = X' = 5.
- **[C]**

**B2.12 y = 13, shelf 17**
- The centres are 312 and 348. Their nearest known-opening translates are 150150 and 30030, both at or above 19² = 361.
- **[C]**

### B3. The stabiliser moves offsets, and acting

**B3.1 Affine maps preserving S**
- They are exactly c = 0, U² ≡ 1 mod q#/6, with U ≡ ±1 mod 5. No shift by r·(q#/30), r = 1..4, preserves S.
- **[R]** q = 7..17: 4, 8, 16, 32 maps.

**B3.2 Survivor classes**
- Class 0 is a survivor but has no copy in the range. Each of the ∏(g−2) − 1 nonzero survivor classes has exactly one copy in the range.
- **[R]** q = 7..19.

**B3.3 Orbits**
- Orbit = fibre {j' : j'² ≡ j²}. uj ≡ j iff D_u | j.
- **[C]**

**B3.4 Ties**
- Tied gears (g | j'² − j²) have equal status on j and j'. Untied gears strike at most one of them.
- The tied set is {g ≤ q} ∪ {gear factors of t}.
- **[C]**

**B3.5 Fixed copies**
- u ≠ ±1 fixes exactly M/D_u − 1 copies.
  - q = 11, u = 34: revealed fixed copies 14 and 35.
  - q = 11, u = 43: revealed fixed copies 22 and 44.
  - q = 13, u = 573: 142 fixed, 24 revealed.
- For u ≠ ±1: √(M+1) ≤ u ≤ M − √(M+1). Moved pairs have max² ≥ M + min².
- **[R]** Fixed sets. **[C]** Bounds.

**B3.6 Window below √M**
- p⁴ < 30q# is sufficient. It agrees with the direct test at every prime q in [7, 20000]; both fail at q = 7.
- **[R]**

**B3.7 Forced pairs preserve exposure**
- For a forced pair, REV(j) ⇒ REV(j').
- Forced pairs at q = 7..23: 3, 16, 65, 166, 539, 1709, with 0 counterexamples.
- **[R]**

**B3.8 Forced criterion above the window**
- A pair is forced iff every gear in (q, sh(j')] divides t.
- Above-window forced pairs at q = 7..23: 0, 1, 3, 3, 17, 18.
- Revealed sources among them:
  - q = 11: 71 → 6 (u = −1, t = −65).
  - q = 19: 189638 → 20 and 223679 → 27 (u = 235808).
- A forced pair never goes upward.
- **[R]**

**B3.9 Repaired (9)**
- (a) Copy-to-copy ties hold only for (0, 1), j' ≤ 3000.
- (b) The stabiliser keeps offsets {2, 3} within {2, 3}.
- (c) The class of an offset-2/3 column at its own acting modulus has a forced member above n only for n = 2 and n = 3. At the column-exact modulus, n = 8 also has one (copy 3 at modulus 7), but that source is struck.
- (d) Forced ties from columns: exactly 2→1, 3→1, 3→2, 8→3, 13→3, 13→4, 3→5, 32→9.
  - Exposure is carried in exactly 2→1, 3→1, 3→2, 3→5, 32→9.
  - Bounds: G(j') ≥ 25j'² for j' ≥ 10, and G(j') > j'² for j' ≥ 4.
- **[R]** j' ≤ 3000.

**B3.10 Invariant revealed sets**
- Only q = 7, u = 6. **[C]**

**B3.11 (10), demoted**
- "Every forced target lies in the window of its own shelf machine" is vacuous: it holds by definition.

---

## 3. Refuted statements, with instances

**0. The brief's survivor-class line** ("each survivor class has exactly one representative copy in the range").
- It fails for class 0, which has no copy in the range. **[R]** q = 7..19.

**Path A**

1. **R11(c) as originally stated.**
   - 7 is the only gear ≡ 7 mod 35 (and mod 210) up to 10⁶. It is a case-19 gear, E_11^(19) = {1, 10}, and 11 divides neither 35 nor 210.
   - For M = 5, 35, 55, every unit class incompatible with case 19 holds 0 case-19 gears. Example: 1 mod 5, where row 7 is non-inert for case 19 but does not divide 5.
2. **R12 under the brief's upper set.** The ceiling system is not a total cover at any q = 7..31 (uncovered 3, 3, 6, 10, 19, 62, 168, 585). At q = 7 the uncovered gears are 7, 11, 13.
3. **Dormancy with non-strict "within".** Both boundaries are attained: (31, 61) with ρ² = h − B, and (17, 19) with ρ² = 2h − B'.
4. **The general clause and its reason "2K_n < 30g".**
   - It fails at every copy g strikes: 330 failures for g < 1000, all with p = g.
   - Instances: g = 11, n = 3 gives 209 = 11·19; g = 7, n = 2 gives 91 = 7·13; g = 7, n = 3 gives 119 = 7·17.
5. **"Divisor gears are always inert".**
   - Case 1, n = 3, h = 11: 11 | K_3 = 44 closes leg A, but −90 ≡ 9 mod 11 is a square, so leg B is reachable and 11 is not inert.
   - For n ≤ 30 and h < 200 there are 45 such divisor gears, e.g. (n = 4, h = 59) and (n = 7, h = 7).
6. **"A recurrence of the inert set gains exactly the divisor gears".**
   - At n = 3249: 59 | r but 59 stays B-only; 19 | s but 19 was already inert.
   - At n = 10923025: 11 goes to B-only, not inert.
   - At n = 36723206689: 43 and 659 go to B-only.
   - At n = 177: 7 goes to A-only.
7. **Stretch (1) "exactly", under the strike-and-eligible reading.**
   - g = 13, copy 10: legs 299 = 13·23 and 301 = 7·43. Gear 17 strikes neither, and 19² > 301.
   - For every gear 7..3000 there is some copy at or above j_{g'} with no strictly acting gear above g.
8. **Stretch (iv): the n = 0 character times the un-normalised symbol.**
   - It is off by (14/h) or (2/h).
   - First failures: h = 17 for the 15n + 14 form, and h = 11 for the 5n + 2 form.
9. **"Every head stretch holds a revealed copy".**
   - g = 37, copies 46..52: 1379 = 7·197, 1411 = 17·83, 1441 = 11·131, 1469 = 13·113, 1501 = 19·79, 1529 = 11·139, 1561 = 7·223.
   - g = 17, 29, 41 and 149 also fail.
10. **The R6 header (sharp on all six families).** F12e, F28e, F12o and F28o are sharp only for their own floors. Examples: 13→17 has d² = 16 while 2h − 30 = −4; 11→29 has kappa = 32.
11. **R5 wording "minimum attained only at 17→19".** It is attained at all six F30 kills.
12. **The R8(c) argument at q = 10¹⁵.**
    - At that point the bound reaches only r ≤ 888,539 (913,612 with the full form). The ratios are 0.91361 for r = 10⁶ and 0.35139 for r = 2.6·10⁶.
    - The worker's 2.718 is the value at 10¹⁶, not 10¹⁵.
    - The refuter's constant 1.443 should be 1/0.7 = 1.4286.
13. **K(d) without the mod-5 case filter.** It would wrongly add 7 to K(4), {11, 23} to K(6), and 113 to K(14).

**Path B**

14. **S1 (the least translate is revealed).**
    - (6,2) → 607: 18209 = 131·139.
    - (9,2) → 610: 18299 has spf 29.
    - (6,3) → 407: 12209 has spf 29.
    - (10,2) → 13624: 408721 has spf 113.
    - (14,2) → 64679: 1940369 has spf 191.
    - Failure counts at q = 13..47: 3, 5, 7, 11, 13, 18, 24, 27, 36, 44.
15. **S2 (the class has a revealed copy in the range).**
    - q = 13, (6,2): the only class copy is 607, which is struck.
    - q = 41, (47,3): M = 247357937827, and all 41 class copies are struck.
    - Counts at q = 13..47: 3, 2, 2, 4, 2, 5, 6, 4, 9, 13.
16. **S3 (the full orbit has a revealed copy).** Both counterexamples are sources above the window.
    - q = 17, lap 14, offset 2: the only orbit copy is 7622 (spf 107 and 251).
    - q = 29, lap 35, offset 2: 12 orbit copies, all struck; the first is 32815693.
17. **S4 (every exposure has an exact transfer into the range).** False for q ≥ 29.
    - (2,2), n = 12: 144 − 100 = 44 ≡ 2 mod 7.
    - 87 sources in laps 2..399 have no transfer. The first ten are (2,2), (4,3), (6,3), (9,2), (11,3), (14,2), (15,2), (21,2), (27,3), (29,2).
18. **"Exposure never passes upward".** The upward pairs are (2,5), (3,5), (3,10), (3,25), (32,45).
19. **(6) literal "count N/M, least copy r" at laps 0 and 1.** The count is one short: q = 7, lap 1 gives 7 against 6.
20. **(3) side clause at lap 1.** The least copy, 0, lies below the source.
21. **Rungs (4) read as a necessary condition.**
    - 42 → 60: A(42) = {}, A(60) = {7}.
    - 72 → 240: A(72) = {7}, A(240) = {7, 11, 13}.
    - 16 instances in total.
22. **Rungs (5b) half-lap bound in floor coordinates.**
    - Fails at 22 of 54 pairs: (5,19), (5,23), (5,59), (5,61), (7,11), (7,17), (7,31), (7,47), (7,53), (7,61), (11,17), (11,19), (11,23), (11,29), (11,37), (11,41), (11,43), (13,23), (13,43), (13,47), (13,53), (13,61).
    - Instances: 378 → −1939560 (64664 laps against 64664.1); 1158 → −5730298560; 558 → −89236590.
    - (7,11) gives 0 laps against 0.443.
    - **[R]** The 22 pairs.
23. **Rungs (6a) as a statement about realised shelves.** At y = 13, shelves 29 and 31 give no translate shorter than the range for q = 19 or 23.
24. **Rungs (6b) y = 7 mean floor 0.2458.** The true minimum is 0.2286; 0.2458 is the (11, 37) value.
25. **Rungs (7b) counting lower-set squares.**
    - Lap 0 at y = 5: 1 vs 2 (gears ≥ 5) and 2 vs 4 (all primes).
    - Lap 0 at y = 7, 11, 13 with all primes: 6, 15, 40 vs 7, 16, 41.
26. **Rungs (11), a fit at y ≥ 7.** There is none; the corrected instance is B2.12.
27. **Rungs (12), offset 18 shares the acting set.** Lap 1: (47, 49) faces 7. This happens in 243 laps.
28. **Stabiliser "only u = 1 keeps height", copy by copy.** q = 11: u = 34 keeps 14 and 35; u = 43 keeps 22 and 44.
29. **Stabiliser (9) "ties never carry exposure upward".**
    - 3→2: 91 = 7·13. 3→5: 616 = 8·77. 32→9: 1001.
    - Also n = 8 → copy 3 at modulus 7, beyond the defence's (c).
30. **Stabiliser (11) header, "exposure transfers only under tie-containment", read pairwise.**
    - q = 11: 41→8 (u = 34, t = −21), 27→6 (u = 43, t = −9), 43→34 (u = −1, t = −9).
    - Each has tied set {7, 11}, gear 13 eligible but untied, and all four legs prime.
    - **[R]**
31. **Stabiliser (12), a uniform status for non-forced lower images.**
    - Per-u minima: revealed 1, 12, 125, 1117, 12987; struck 1, 33, 417, 5878, 97309.
    - 44 → 33: 989 = 23·43, with 23 untied.
    - q = 23, u ∈ {163437, 2288131}: 4576236 → 28, target (839, 841 = 29²), 29 untied.
    - **[R]** Both instances.

**Figure corrections (not refutations of statements)**
- Refuter's prime count 78495 → 78494.
- "1349 rows" means row-case pairs; there are 1049 rows.
- R11 largest resolution: 12326371 with r fixed at min E_h, but 4191739 with any r.
- The rungs adjudicator's first float count of 20 → the exact count is 22.

---

## 4. Not established (open questions)

**Path A**
- **R14 target.** For every q, does some g in F_q have both legs of c_g prime? Equivalently, is the revealed hand-off sequence infinite? No argument yet shows the union of row exclusions leaves a family gear uncovered for every q.
- **Converse.** Does the range statement imply the R14′ target?
- **Is ell_c(h) bounded?** Only the O(√h) least-non-residue bound is proved; the maxima 8 and 13 are measurements.
- **Density law 2^(−r_L).** It rests on Dirichlet. It was checked only for h ≤ 200000 and L ≤ 13, and is weaker for case 1 at L ≥ 5 (14, 7, 2 observed against 17.6, 8.8, 4.4 predicted). Ranks and free copies are computed only for L ≤ 60.
- **Gears above g** acting on copies beyond the protected stretch are not analysed; the threat set covers h < g only.
- **The second reading of "h threatens every copy".** It fails when l ≥ h. For l < h, where the inert heights fall relative to n = 0 is not classified.
- **Later g-silent runs** inside the protected region when L > p1: lengths not tabulated.
- **Empty head stretches and regions.** Are there finitely many? Measured only to g = 10⁶ (5 stretches, 3 regions).
- **Equality families.** Is any of F10, F30, F12e, F28e, F12o, F28o infinite?
- **Finiteness of the previous-gear and r-th-gear kill lists** unconditionally beyond 4·10¹⁸. The 10⁸ → 4·10¹⁸ extension rests on the published maximal-gap table, not re-verified.
- **R8 for r > 12** when 149 ≤ q < 10¹⁵.
- **Whether every c_g in a range can be killed by some gear.** This is the same question as the primality of the legs across the range.

**Path B**
- **S3 for window sources.** No proof; measured to q = 67. The refuter's q = 71..113 extension was not recomputed.
- **S4-type existence** for sources near n ≈ q#/6.
- **Target-modulus transfer** from sources above the range into the range.
- **"No source leg strikes r"** is a measurement only (j < 2·10⁶).
- **The partner condition** in B1.6 was never recomputed.
- **Mean cost ≈ X#/4.** Unproved.
- **A closed bound on k*.** None; the measured maximum is 9.
- **Upward landings** (c' + 1 ≥ nextprime(X)²) were not studied.
- **The destination-gear version (rungs R3)** has no measurement.
- **(7b) half (ii) for y ≥ 31.** It needs either exact G_31 (3.3·10¹⁰ columns) or a non-counting proof of G_y ≤ 4√(y#).
- **Stabiliser:** no structural rule for when the gears in (q, sh(j')] divide t (measured only for q ≤ 23).
- **Necessity of p⁴ < 30q#** in B3.6 is unproved.
- **The q = 23 tie check** on non-machine gears covered only 2 units and gears ≤ 60.

---

## 5. Computations that did not finish, and runs not made

- **Angle runs.** All seven angles report every run finished in the foreground (the longest took 75 s). Nothing under C:/dev/primes was modified.
- **Not attempted, beyond budget:** exact G_31 by enumerating 3.3·10¹⁰ columns.
- **Not recomputed:**
  - the refuter's q = 71..113 orbit extension (MR plus BPSW);
  - the Path B partner condition;
  - the published maximal-gap table;
  - the Rosser–Schoenfeld constants.
- **Path B translation worker's note: "machines q ≥ 53 not computed".** This refers to the S1/S2 counts. The adjudicator later extended the range statement to q = 53 (B1.8) and the orbit measurement to q = 67 (B1.9).
- **My runs:**
  - The first full run of v7_stab.py passed 300 s. The harness moved it to the background; I stopped it with no output. My own copy-tie loop caused this (it recomputed G(j') inside a double loop). I fixed it and reran it in four foreground parts: 2.3 s, 2.6 s, 30.0 s and 29.5 s. All finished.
  - v6 first under-counted lap-0 acting sets at y = 23 because its gear list stopped at 4000. The rerun (v6b) with gears to 20000 gives 1740.
  - All other runs finished in the foreground: v1b 11.8 s, v2 8.7 s, v3 29.3 s, v4 26.0 s, v6 47.8 s, v8 5.5 s; v1, v5 and v9 were short.

---

## 6. Locating and counting residue in what stands

**Path A escape**
- R14′(ii) phrases the target through successive witness positions: g_{i+1}² + B ≤ (g_i² + A)#. Its bound is the range's own edge.
- The measured coverage in R14′(i) is carried by one gear: 198437 alone serves every prime q in [31, 39377242978].
- The remaining counts are measurements over exact ranges and class structure: the R3 pattern sizes, the R7/R10 products, the R11 figures 1821/121/10, the R12 sizes, and the 328 revealed gears.

**Path A inert**
- Everything is anchored per gear, at c_g or on its stretch.
- The recurrence hosts (n = 177 at g = 293, n = 3249 at g = 4111) locate recurrences of the inert set, not revealed copies.
- Counts of gear states: the period counts, and the 2^(−r_L) density law with its flagged tallies.

**Path A stretch**
- The A3.4 measurement records whether a revealed copy lies in [J_g, j_{g'}). It is locating-shaped data. Any claim that every region beyond some g holds a revealed copy would be locating.
- Every result sits on shelf(g), where the acting set is machine g's gears. Stacking shelves to reach the range would split the range into bands; no standing statement does this.
- (1) depends on reading "acts" as square-eligible.

**Path A neighbours**
- Counting: the q ≥ 10¹⁵ part of R8 rests on a pigeonhole comparing gears in (x, 2x] with blocks of length √(2x − 30).
- Locating: R8 places the witness missed copy at a bounded height (g ≤ 10⁸, or g ∈ [x, 2x]). That copy is not claimed revealed.
- External inputs: the maximal gap 1476 below 4·10¹⁸ and the Rosser–Schoenfeld constants.

**Path B translation**
- (10) bounds the revealed target by its source (5j' < n, M(j') ≤ n² − 25j'²). For n ≤ 5·10⁶ the targets reach at most copy 27, and 102,173 of the 157,062 pairs land on copy 1, where the transfer is vacuous.
- (6) and (13) split sources at the window cut. (13) is the only positive range-level content, and it is a measurement over window sources.
- (3) and (5) bound the heights of translates and partners, which need not be revealed.
- (12) uses the located witness copy 2 as a check.

**Path B rungs**
- Located:
  - (5b)(ii), restricted to real sources on their own shelf.
  - (6a), worked shelf by shelf with period X#, i.e. a stack of smaller machines.
  - (6c), which uses X_top.
  - (8) and (9), which define a fit by a height bound on the destination.
  - (10), a window form at machine X'.
  - (11).
- Counted:
  - (6a), a translate length against the range length.
  - (6c), known-opening copies compared with C_min.
  - (6b) means, which are measurements.
  - (7b) for y ≥ 31, which rests on an inclusion–exclusion bound and is unresolved.

**Path B stabiliser**
- The counts are measurements.
- Locating:
  - (6), (10) and the always / sporadically / never taxonomy split machine q's range into the window and above the window.
  - The repaired (10) form reads a target through the smaller machine sh(j').
  - The height caps G(j') > j'², j' ≤ 9 and c0 > M/5 apply to forced-tie members, not to revealed copies.
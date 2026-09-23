# Round record: acting, missed copies, shelves, survivor classes and gear pairs

Every statement below comes from the six adjudicated angles. I re-ran the load-bearing ones as bounded foreground scripts, rebuilt from raw divisibility, and state the ranges I re-checked. Where no re-check range is given, the range is the angle's own.

## 1. The statements that stand

### A. Acting and revealing
A1. Gear g acts on copy j iff g² ≤ 30j+1. Writing the struck leg as g·h, this is exactly h ≥ g. It is a per-integer identity.

A2. For j ≥ 1: no acting gear strikes j iff both legs are prime. Proof: the least prime factor p of a composite leg has p² ≤ leg, so p acts. Copy 0 is the single exception: nothing acts on it and nothing strikes it.

A3. Every strike that does not act is shadowed. The least prime factor of that leg acts on the same copy.

A4. Gears 7 and 11 are the only gears with no strike below their own square. The number of below-square strikes of gear g is exactly 2⌊g/30⌋ + c(r), where r = g mod 30 and:

| r | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|
| c(r) | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 1 |

- Re-checked: all gears ≤ 2000, raw.
- Gears below 500 make 1043 below-square strikes on copies below 3000 (re-checked).

### B. The missed copy c_g
B1. c_g is the lowest copy at or above height g² that g does not strike.
- If g² ≡ 19 mod 30 (g ≡ 7, 13, 17, 23 mod 30), its legs are (g²+10, g²+12).
- If g² ≡ 1 mod 30 (g ≡ 1, 11, 19, 29 mod 30), its legs are (g²+28, g²+30).
- In both cases c_g is copy J_g = ⌈(g²+1)/30⌉.
- Reading the height off the lower leg or the upper leg gives the same copy. In case 1 the upper-leg reading first reaches (g²−2, g²), which g strikes because it contains g². This step is needed in the proof.
- Re-checked: all gears < 50000, both readings, raw.

B2. g divides neither leg of c_g. Otherwise g would divide 10 or 12 (case 19), or 28 or 30 (case 1). The only prime ≥ 7 dividing any of these is 7, which divides 28, and gear 7 is a case-19 gear.

B3. No gear larger than g acts on c_g. The next gear satisfies g'² ≥ (g+2)² > g²+30.

B4. A gear h with 7 ≤ h < g strikes c_g iff g² ≡ −A or −B (mod h), with (A,B) = (28,30) in case 1 and (10,12) in case 19. Since −A and −B differ by 2, each h excludes 0, 2 or 4 classes of g mod h.

B5. Inert gears. A gear h is inert when it can never strike c_g for any prime g > h of the given case.
- Case 1: h is inert iff h = 7, or [h ≡ 3, 5, 6 mod 7 and h mod 120 ∉ {1,11,13,17,23,29,31,37,43,47,49,59,67,79,101,113}].
- Case 19: h is inert iff h ≡ 2 mod 3 and h mod 40 ∉ {1,7,9,11,13,19,23,37}.
- Re-checked: residue form for all h ≤ 5000; raw with prime g < 10⁵ for the first 60 gears.

B6. Gear 7 never strikes c_g in case 1. In case 19 it strikes c_g iff g ≡ ±2, ±3 mod 7. Re-checked for g < 200000.

B7. c_g survives every smaller gear iff both legs of c_g are prime.
- Gears 7..199999: 233 case-1 gears and 95 case-19 gears.
- The congruence count and the both-legs-prime count agree. Re-checked.

B8. Only for each fixed finite H: among primes of a given case, those avoiding the excluded classes for all 7 ≤ h ≤ H have relative density ∏ w_h, with w_h = 1 − |E_h|/(h−1). This follows from CRT and Dirichlet.

B9. Strike law above the square. Put J = J_g and t_g = 15⁻¹ mod g. Then g strikes copy J+M iff M ≡ x_g or x_g + t_g (mod g), where:
- case 1: x_g ≡ −1 (mod g);
- case 19: x_g ≡ −2·5⁻¹ (mod g).

So exactly g−2 of every g consecutive copies above the square are missed. Re-checked for gears < 30000; raw for gears < 3000.

B10. p1 and p2 (closed forms in 3.B) are exact.
- p1 = 1, 2, 1 at g = 7, 11, 29, and p1 ≥ 3 for every other gear.
- So for every other gear, the 2nd and 3rd missed copies are J+1 and J+2.
- Proof: p1 ≤ 2 forces g ≤ 45.
- Re-checked for gears < 30000.

B11. Protection from above for copy J+M. No gear larger than g acts on J+M iff M < (g'² − g² − B)/30, with B = 30 in case 1 and B = 12 in case 19. Re-checked for case 19, g < 20000.

B12. If the upper leg of the n-th missed copy is below g'², then "no h < g strikes it" iff "both legs prime". Counts over gears 7..199999, computed from the n-th missed copy itself (re-checked):

| n | congruence (case 1 / case 19) | both legs prime (case 1 / case 19) |
|---|---|---|
| 1 | 233 / 95 | 233 / 95 |
| 2 | 513 / 157 | 513 / 157 |
| 3 | 167 / 327 | 167 / 326 |

The single n=3 mismatch is at g = 17, with legs (359, 361) and 361 = 19².

### C. Shelves
C1. shelf(g) = [j_g, j_g'), where

j_g = ⌈(g²−1)/30⌉ = ⌊(g²+11)/30⌋ = (g²−1+12χ(g))/30, with χ(g) = [g² ≡ 19 mod 30].

- The shelves tile the copies from j = 2 onward.
- Copy 1 = (29, 31) is shelf(5). Gear 5 acts there but strikes nothing, so copy 1 is revealed.
- Re-checked for gears < 2·10⁶.

C2. Shelf size: N(g) = (G(2g+G) + 12(χ(g')−χ(g)))/30, where G = g'−g. The numerator is always divisible by 30 and N(g) ≥ 1.
- Among gears whose next gear is ≤ 2·10⁶, the only shelves of size ≤ 2 are those of 7, 11 and 17, each of size 2.
- The class forms in 3.D were re-checked on the same range.

C3. The acting set on every copy of shelf(g) is exactly the primes in [7, g]. Re-checked raw on copies 0..19999.

C4. Own-shelf silence. Let off(r) be as in the table in 3.C. Then g strikes no copy of its own shelf iff g(g+off(r)) ≥ g'². Equivalently:
- G(2g+G) ≤ off·g (+2 for r ∈ {13, 23}), or
- (g, g + off/2 − 1] contains a prime.

This gives the complete classification:
- never silent for r = 1, 7, 11, 19, 29;
- r = 17: silent iff g+2 is prime;
- r = 13: silent iff g+4 is prime;
- r = 23: silent iff g+6 or g+8 is prime.

It is a theorem. The lower-leg edge case g(g+off)+2 = g'² is proved impossible: it forces g ∈ {7, 17, 31}, and none of those has a matching gap.

Re-checked for gears ≤ 300000; below 30000 there are 3242 gears, 576 of them silent.

C5. c_g = j_g in case 19 and c_g = j_g + 1 in case 1. c_g lies in shelf(g), and g strikes every copy from j_g up to c_g.

C6. If g is silent, every composite leg on its shelf has least prime factor < g.

C7. Twins at the gear.
- If g+2 is prime, the own-shelf strike count of g is 1 for r = 11, 2 for r = 29, and 0 for r = 17.
- For r = 11 the single strike is the head (leg g²).
- For r = 29 the strikes are the head and the copy with lower leg g(g+2).
- Re-checked for gears ≤ 300000.

C8. Deferral law. For a silent gear, the first acting strike lands on the shelf of the largest prime in (g, g + off/2 − 1]. Re-checked for gears ≤ 300000.
- Argument: the host is the largest prime ≤ isqrt(H), where H = g² + off·g + e and e ∈ {0, 2}.
- isqrt(H) = g + off/2 − 1 fails only at g = 23 (off = 20). Even there the largest prime is the same.

C9. Step law. For a strike on leg g(g+δ), with H = g(g+δ)+e, isqrt(H) = g + δ/2 − 1 whenever (δ/2−1)² − e ≤ 2g and δ²/4 > e. The second condition fails exactly for δ = 0, and for δ = 2 on the lower leg.

C10. Runs of consecutive silent gears have length at most 2. The runs of length 2 are exactly those starting at g ≡ 13 mod 30 with g+4 and g+6 both prime: 282 such runs below 300000 (re-checked).

C11. Every acting strike on shelf(p_i) comes from a gear ≤ p_i. So R_i = S_i ∩ V(p_i), where V(p) is the survivor set of machine p.

C12. No shelf contains a mirror pair (j, P−j) with P = p_i#/30, since every shelf element is below P/2.

C13. M_i is the least machine that reveals shelf i. It equals the maximum, over the shelf's copies with a composite leg, of the least prime factor among those composite legs.

### D. Cofactors
D1. Strikes of gear g are in bijection with cofactors h ≡ ±u mod 30, where u = g⁻¹ mod 30. The upper leg is struck iff h ≡ u, the lower leg iff h ≡ −u.

D2. The admissible cofactors are exactly the positive integers ≡ ±k0 mod 30, with k0 = min(u, 30−u) ∈ {1, 7, 11, 13}. (Z/30)*/{±1} is cyclic of order 4, generated by [7].
- Over the first 2000 gears (7..17417), k0 = 1, 7, 11, 13 occur 488, 507, 497, 508 times (re-checked).

D3. The struck copies are j = m·g ± j0, with j0 = (g·k0 − ε)/30 and ε = +1 if r < 15, −1 otherwise.
- The gaps alternate g − 2j0 and 2j0.
- a_g = −ε·j0 mod g.
- The first acting cofactor is g + off(r).
- Re-checked raw for gears ≤ 2000.

D4. First acting cofactor on each leg. With v = g mod 30 and u = v⁻¹ mod 30, set h_up − g = (u−v) mod 30 and h_lo − g = (−u−v) mod 30. The values are in 3.C and were re-checked on the first 3000 gears.
- h_lo ≠ g in every row. This alone does not show that c_g is missed; that is proved by B1–B3.

D5. Separation. 15·t_g = k·g + 1, with k by class:

| r | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|
| k | 14 | 2 | 4 | 8 | 7 | 11 | 13 | 1 |

- Re-checked for gears ≤ 30000.
- With σ_g = min(t_g, g − t_g): 15σ_g = k'g ± 1, with k' ∈ {1, 2, 4, 7}. The (k', sign) table is in 3.C.
- (p−1)/15 ≤ σ_p ≤ (7p+1)/15.

D6. What T implies. Let T be "a composite has a prime factor at or below its square root".
- T implies: acting ⇔ h ≥ g, revealed ⇔ both legs prime, and below-square strikes are redundant.
- T does not imply the following; each is refuted by a variant machine in which T holds:
  - the spoke set {1, 7, 11, 13} and k0 ≤ 13 (M210 has 24 values of k0, max 103; M2310 has 240, max 1153);
  - the exception set {7, 11} of the first-strike law (M210 has 13 exceptions: 11, 13, 17, 23, 29, 31, 41, 43, 47, 53, 59, 71, 79);
  - the mirror (SKEW has no mirror-symmetric gear ≥ 7);
  - the location of c_g (M210 has six offsets: 169→40, 151→58, 121→88, 109→100, 79→130, 1→208, where the key is g² mod 210; gear 11 deviates, to 298);
  - the constants of the shelf-silence criterion.

### E. Survivor classes
E1. S_q = ∏_{7≤g≤q} (Z/g ∖ {±a_g}) mod M_q = q#/30. Equivalently S_q = {j : gcd(900j²−1, M_q) = 1}, and |S_q| = ∏(g−2).
- For q = 7, 11, 13, 17: 5, 45, 495, 7425 (re-checked).

E2. The affine stabiliser of S_q is exactly {j ↦ uj : u² ≡ 1 mod M_q} ≅ (Z/2)^r. Translations are excluded.
- #{u : u² ≡ 1} = 2, 4, 8, 16 for q = 7..17 (re-checked).
- The orbit of j has size 2^#{g : j ≢ 0 mod g}.
- ∏(g−3) survivors lie in free orbits.

E3. Adding the next gear p: the reduction S_{q'} → S_q is onto and exactly (p−2)-to-1.
- For j ≥ 1, j ∈ S_q iff q < P(j), the least prime factor of (30j−1)(30j+1).
- The integer intersection of the whole tower is {0}.

E4. Thresholds. A_up = ⌈(g²−1)/30⌉ and A_lo = ⌈(g²+1)/30⌉. They are equal iff g² ≡ 19 mod 30 and differ by exactly 1 iff g² ≡ 1 mod 30.

E5. Each gear g > q strikes exactly two of the g subclasses mod M_q·g of each class, one per leg.
- It removes nothing below A_up.
- At or above A_lo it removes exactly one copy per class per leg in each block of M_q·g.
- The blind count blind_x = max(0, ⌈(A_x − r_x)/(M_q·g)⌉) takes at most two consecutive values.

E6. Gears above q delete no class; they only refine. Among copies j ≥ 0 of S_q, those acted on by no gear above q are exactly {0} and the copies with both legs prime.
- For 1 ≤ j < A_up(p): j ∈ S_q iff both legs are prime and both exceed q.

E7. Check, not a proof: all 495 classes of machine 13 hold a revealed copy with j < 10⁶. The latest first revealed copy is in class 548, at copy 35583 (re-checked).

E8. Machine-gears-only theorem: some copy of the range is struck by no gear ≤ q.

### F. Gear pairs (p < q = p + d, d even, d < p)
F1. Both strike copy j ⇔ (30j)² ≡ 1 mod pq.
- The four classes are 30j ≡ ±1 (same leg: pq divides one leg) and 30j ≡ ±w (split: p and q divide different legs), with d·w ≡ p+q (mod pq).
- For twins, w = p+1 exactly.

F2. Per period pq: 4 copies struck by both, 2(q−2) by p only, 2(p−2) by q only, (p−2)(q−2) by neither (CRT).

F3. Least split copy, with c defined by d·h_p = c·q ± 2:
- both gears act iff c > d;
- c is constant on each class of p mod 15d (the table in 3.E was re-checked for p < 6000).

F4. Twin specialisation. Twins occur only for p ≡ 11, 17, 29 mod 30. All the a, δ, least-split-copy and least-same-leg-copy formulas in 3.E hold (re-checked for all twin pairs with p < 200000).

F5. First-strike separation J_q − J_p. With J_g = min{j ≥ 1 : g | 30j ± 1} = min(a_g, g − a_g), the formulas for twins, cousins and sexy pairs in 3.E hold with 0 mismatches over:
- 2158 twin pairs,
- 2135 cousin pairs (including (7, 11)),
- 4294 sexy pairs,
- all with p in 7..199999, using exact rational comparison.

The brief listed this as UNRESOLVED; this assembly settled it.

F6. Sexy pairs, two constructions:
- (i) p ≡ 11 mod 90: j = (p(2p+11)/3 − 1)/30 has 30j+1 = p·(2p+11)/3 and 30j−1 = q·(2p−1)/3, with 30j+1 ≤ q².
- (ii) p ≡ 73 mod 90: j = (p(2p+13)/3 + 1)/30 has 30j−1 = p·(2p+13)/3 and 30j+1 = q·(2p+1)/3.
- Neither gear acts for p > 11 in (i) or for p ≡ 73 mod 90 in (ii). At p = 11, gear 11 acts and gear 17 does not.
- Re-checked: 32 pairs with p < 6000.

F7. Type-C twins (p ≡ 29 mod 30) have exactly two joint copies below about q² that are not acted on by both gears:
- j* = (p+1)/30, where neither acts;
- the copy with leg pq, where p acts and q does not.

Twins at 11 or 17 mod 30, and all cousins, have neither kind.

F8. Partner strikes on the other gear's missed copy. p strikes c_q iff p | d²+10 or d²+12 (q² ≡ 19 mod 30), or p | d²+28 or d²+30 (q² ≡ 1 mod 30). Symmetrically for q and c_p. For d ≤ 20, d < p, p ≤ 20000 (re-checked):
- p strikes c_q exactly for (13,17), (17,19), (23,31), (29,41), (67,83);
- q strikes c_p exactly for (17,23), (19,23), (31,43), (41,59), (83,103);
- allowing d ≥ p adds only p-strikes-c_q hits: (7,17), (7,23), (11,23), (11,29), (13,29);
- among twins, (17,19) is the only pair with a hit.

### G. Earlier laws carried into this round
- Strike law: j ≡ ±a_g mod g.
- Squaring law: j² ≡ a_g² mod g.
- Mirror: j → −j.
- Leg rule: two copies D apart are both struck by g iff g | D, 15D−1 or 15D+1.
- Locator closure: a class protected against all gears up to X has modulus divisible by X#/30.
- The barrier: without acting there is a total blame map, b(j) = lpf(30j−1).

## 2. The statements refuted

| Statement | Refuting instance |
|---|---|
| Earlier brief: "gears do not strike below their square" | Gear 13 strikes copy 3 (91 = 7·13). There are 1043 such strikes (gears < 500, copies < 3000). |
| Case-1 inert rule written purely with characters | False at h = 7 alone: 7 divides 28, so gear 7 is inert in case 1 (B5). |
| n = 3: "0 mismatches" between congruence survival and both legs prime | g = 17, legs (359, 361), 361 = 19². The refuter had counted copy J+n−1, not the n-th missed copy. |
| Protection bound M < (g'² − g² − 30)/30 in both cases | Fails in case 19. g = 13: copy J+3 = 9, legs (269, 271), is protected (17² = 289), but the B = 30 bound excludes M = 3. Also g = 43 (true max M 11, rule gives 10) and g = 47 (19 vs 18). |
| Silence propagates to the next gear | g = 17 is silent; gear 19 strikes leg 361 at copy 12 = j_19, on its own shelf. |
| "Gears above p_i never strike shelf copies" (original wording of C11) | Gear 13 strikes copy 3, which lies in shelf(7) = [2, 4). The strike does not act. |
| Every shelf contains a revealed copy | Shelf 17 = {10, 11} and shelf 29 = {28, 29, 30, 31} have none. |
| Mirror argument using only "shelf ⊂ (0, P)" | Insufficient: shelf 7 has top/P = 4/7. Needs the P/2 bound. |
| Silence criterion printed as (off − 2δ)g ≤ δ² | Sign slip; it is the exact negation of the true criterion. |
| Separation with t_g itself giving the four slopes | With t_g, k = 14, 2, 4, 8, 7, 11, 13, 1. The four slopes need min(t_g, g − t_g). |
| Cofactor window [g²/f, g'/f)² | Correct form is [g²/f, g'²/f). |
| d1/d2 value set without 2 and 8 | d2 = 8 for class 11 and d2 = 2 for class 29. The criterion is unaffected, since both have d1 = 0. |
| The first-strike exception set follows from T | M210: exceptions 11, 13, …, 79 (13 gears). Argument (i) withdrawn. |
| Spoke set, k0 ≤ 13, mirror, c_g location, silence constants follow from T | M210, M2310, SKEW (D6). |
| SKEW: "no gear ≥ 11 is mirror-symmetric" | Too weak. No gear ≥ 7 is, since symmetry needs g \| 2 or g \| 10. |
| M210 "seven offsets" | Six offsets, one per value of g² mod 210, plus a deviation at gear 11. |
| "h_lo ≠ g ⇒ c_g missed" | The clause alone does not prove it. |
| Type-C twins "non-acting joint copies = 2" | Correct only as "not both acting". With "neither acts" the count is 1 (j*). |
| isqrt(H) = g + off/2 − 1 for every silent gear | Fails at g = 23 (off = 20). The deferral conclusion is unaffected. |
| Cousin pairs, p in 7..199999: 2134 | The count is 2135, including (7, 11). |
| Sexy split copies: least-c constancy on p mod 30 | Fails. Constancy is on p mod 15d (the earlier s6 run's "FAIL" was a modulus error). |

## 3. Formulas and classifications

### 3.A Missed-copy law
- Case 1 (g ≡ ±1, ±11 mod 30): (A,B) = (28, 30). Case 19 (g ≡ ±7, ±13 mod 30): (A,B) = (10, 12).
- c_g = copy ⌈(g²+1)/30⌉, legs (g²+A, g²+B).
- c_g survives every smaller gear iff:
  - case 1: g ≢ ±√(−28), ±√(−30) mod h for all gears 7 ≤ h < g;
  - case 19: g ≢ ±√(−10), ±√(−12) mod h for all gears 7 ≤ h < g.
- Reachability of each leg:
  - (−28/h) = (h/7), so the g²+28 leg is reachable iff h ≡ 1, 2, 4 mod 7;
  - (−12/h) = (−3/h), so the g²+12 leg is reachable iff h ≡ 1 mod 3;
  - (−10/h) = 1 iff h mod 40 ∈ {1,7,9,11,13,19,23,37};
  - (−30/h) = 1 iff h mod 120 ∈ {1,11,13,17,23,29,31,37,43,47,49,59,67,79,101,113}.
- Local factor: w_h = (h − 1 − |E_h|)/(h − 1), with |E_h| = (1 + (−A/h)) + (1 + (−B/h)), dropping a term when h | A or h | B.
  - First factors, case 1: 1, 6/10, 10/12, 14/16, 1, 18/22, 24/28 at h = 7, 11, 13, 17, 19, 23, 29.
  - Case 19: 2/6, 8/10, 8/12, 1, 14/18, 20/22, 1.
  - Products up to h < 200000 (numerical): 0.026292 (case 1) and 0.010835 (case 19).
  - Summed predictions: 298.94 and 125.08, against raw survivors 233 and 95.
  - Inert gears: 4514/17981 (case 1) and 4474/17981 (case 19).
- Above the square: g strikes J+M iff M ≡ x_g or x_g + t_g (mod g).
- p1, the length of the silent run headed by c_g, by g mod 30:

| g mod 30 | p1 |
|---|---|
| 1 | 14(g−1)/15 |
| 7 | (g−2)/5 |
| 11 | (4g−14)/15 |
| 13 | (g−1)/3 |
| 17 | (g−2)/5 |
| 19 | (11g−14)/15 |
| 23 | (2g−1)/3 |
| 29 | (g−14)/15 |

- p2 = g−1 in case 1. In case 19: (g−1)/3, (4g−2)/5, (2g−1)/3, (4g−2)/5 for g ≡ 7, 13, 17, 23.
- For n ≤ p1 the n-th missed copy has legs (g² + A + 30(n−1), g² + B + 30(n−1)), and the same classification applies with those offsets. Gear 7's reach is periodic in n with period 7.

### 3.B Cofactor ordering

| r | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|
| u = r⁻¹ mod 30 | 1 | 13 | 11 | 7 | 23 | 19 | 17 | 29 |
| k0 | 1 | 13 | 11 | 7 | 7 | 11 | 13 | 1 |
| ε | + | + | + | + | − | − | − | − |
| c(r) | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 1 |
| off(r) | 0 | 6 | 0 | 10 | 6 | 0 | 20 | 0 |

- Cofactors: h_{2m} = 30m + k0, h_{2m+1} = 30(m+1) − k0.
- Copies: j_{2m} = mg + j0, j_{2m+1} = (m+1)g − j0, with j0 = (g·k0 − ε)/30.
- Legs alternate, starting with the upper leg iff r < 15.
- Gaps alternate (g(15 − k0) + ε)/15 and (g·k0 − ε)/15.
- a_g = −ε·j0 mod g.
- g is itself a cofactor iff r ∈ {1, 11, 19, 29}; that strike is the copy with upper leg g².
- Crossover gaps: (g−20, g+6) for r = 7, (g−6, g+10) for r = 13, (g−10, g+6) for r = 17, (g−6, g+20) for r = 23.

### 3.C Offset tables
First acting cofactor offsets (h_up − g, h_lo − g), rows (v; up, lo):

(1; 0, 28), (7; 6, 10), (11; 0, 8), (13; 24, 10), (17; 6, 20), (19; 0, 22), (23; 24, 20), (29; 0, 2).

Top gear on its own shelf:
- strikes at A(g) + (d1·g − c)/30 + m·g on the upper leg, and at A(g) + (d2·g + 2 − c)/30 + m·g on the lower leg;
- (u, d1, d2, c) by class: 1→(1,0,28,0), 7→(13,6,10,12), 11→(11,0,8,0), 13→(7,24,10,12), 17→(23,6,20,12), 19→(19,0,22,0), 23→(17,24,20,12), 29→(29,0,2,0).

Separation (k', sign) with σ = min(t, g − t): 1:(1,−), 7:(2,+), 11:(4,+), 13:(7,−), 17:(7,+), 19:(4,−), 23:(2,−), 29:(1,+).

### 3.D Shelf decomposition
- Floor: A(g) = j_g = (g² − 1 + c(g))/30, with c = 0 or 12 by g² mod 30.
- Size: |S(g)| = (d(2g+d) + c' − c)/30 = g·d/15 + d²/30 + e, with |e| ≤ 0.4.
- Size by class, (r, G) → N: (17,2) → (2g−4)/15; (11,2) → (2g+8)/15; (29,2) → (2g+2)/15; (13,4) → (4g+8)/15; (23,6) → (2g+4)/5; (23,8) → (8g+26)/15.
- Revealed copies: R(g) = S(g) ∩ V(g), with |V(g)| = ∏(f − 2).
- Silence: g(g + off) ≥ g'², equivalently a prime in (g, g + off/2 − 1].
- Deferral onto the shelf of the largest such prime; runs have length ≤ 2.

### 3.E Survivor class structure and gear-pair laws
Survivor classes: see E1–E6 for S_q, the stabiliser (Z/2)^r, (p−2)-to-1 splitting, blind_x, F_x = r_x + M_q·g·blind_x, and removal tails of period M_q·g.

Twin specialisation (d = 2):
- Class A, p ≡ 11: a_p = (19p+1)/30, a_q = (23q+1)/30, δ_p = (4p+1)/15, δ_q = (7q−1)/15.
- Class B, p ≡ 17: a_p = (7p+1)/30, a_q = (11q+1)/30, δ_p = (7p+1)/15, δ_q = (4q−1)/15.
- Class C, p ≡ 29: a_p = (p+1)/30 = q − a_q, δ_p = δ_q = (p+1)/15.
- J_q − J_p: (7−2p)/15 (A), (2p+11)/15 (B), 0 (C).
- Least split copy: (6pq+p+1)/30 (A), (6pq−p−1)/30 (B), (p+1)/30 (C).
- Least same-leg copy: (13pq+1)/30 (A, B), (pq+1)/30 (C).

Cousins (d = 4): J_q − J_p = (22−p)/15, 1, (p+26)/15 for p ≡ 7, 13, 19 mod 30.

Sexy pairs (d = 6): J_q − J_p = (2p+13)/5, (7−p)/5, (22−2p)/15, (2p+34)/15, (p+13)/5, (1−2p)/5 for p ≡ 1, 7, 11, 13, 17, 23 mod 30.

Least-c tables:
- d = 2: {11: 12, 17: 12, 29: 0} mod 30.
- d = 4: {7: 6, 13: 30, 19: 54, 37: 54, 43: 30, 49: 6} mod 60. Cousins never have c ≤ 4.
- d = 6: {1: 76, 7: 20, 11: 4, 13: 64, 17: 80, 23: 16, 31: 44, 37: 40, 41: 56, 43: 56, 47: 40, 53: 44, 61: 16, 67: 80, 71: 64, 73: 4, 77: 20, 83: 76} mod 90.

Cofactor form of the split copy: h_p = (qc ∓ 2)/d, h_q = (pc ∓ 2)/d.

## 4. What was not established
1. Whether infinitely many gears g have c_g revealed. Equivalently, whether g²+28, g²+30 (resp. g²+10, g²+12) are simultaneously prime for infinitely many prime g.
2. Whether survivors of the full sieve over all h < g have density equal to ∏_{h<g} w_h. Only the fixed-H statement is proved. The raw/product ratios at 200000 are 1.283 and 1.317.
3. Closed forms, in terms of L-values, for the two singular-series constants.
4. Survivor censuses of the n-th missed copy for n ≥ 4. The classification is stated for general n but checked only for n ≤ 3.
5. Why the least-c table is constant on p mod 15d. The argument gives only p mod 30d. There is also no general-d rule for which classes give c ≤ d.
6. A proof that, for sexy pairs, each pair in classes 11 and 73 mod 90 has exactly one split copy and the other classes have none. This is measured only for p < 6000.
7. An M210 crossover law and silence criterion. That they differ from the real machine is argued, not measured.
8. The SKEW boundary effect: its raw least-cofactor set has an extra value 19 from the j ≥ 1 boundary, not traced gear by gear.
9. Enumerated-class checks of the shelf–survivor identity beyond machine 23. Only the equivalent predicate form was checked beyond it.
10. Which acting-based statement forbids a total blame assignment. The barrier shows any such statement must involve acting; none was produced.

## 5. Computations that did not finish
- shelf_struct.py (shelf angle), part 1: per-copy acting sets up to g = 3000. It exceeded 290 s with no output and was stopped. It was replaced by shelf_act.py (gears ≤ 1500, 76102 copies) and shelf_big.py (to 300000), both of which completed.
- sc2_tower.py, first version (survivor angle): it exceeded its 420 s budget with no output and was stopped with TaskStop. The rewrite completed.
- sc1_box.py: the exhaustive affine-stabiliser search was run only up to M = 1001. The run at M = 17017 (about 1.7·10⁸ pairs) was not started.
- asm_1_missed.py (this assembly), first version: my c_g search stepped from copy 0 for every gear. That broke the brief's rule: the run passed the 300 s foreground limit, the harness moved it to the background, and I stopped it with TaskStop, with no output. After fixing the search it ran in 13.5 s in the foreground, and all figures above come from that run.
- Ranges the angles did not push further:
  - the exhaustive raw strike enumeration covers gears 7..2000; gears 2000–9000 have only windowed spot checks;
  - the raw crossover check stops at gear 5000;
  - the raw joint-class scans cover p ≤ 260;
  - the least-c search covers p < 6000;
  - the acting-escape scan covers p < 20000;
  - revealed copies were scanned to j ≤ 300000;
  - the cut and class-coverage sweeps stop at 10⁶ copies.

Nothing under C:/dev/primes was modified.

Files are in C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad:
- asm_lib.py
- asm_1_missed.py
- asm_2_shelf.py
- asm_3_pairs.py
- asm_4_protect.py
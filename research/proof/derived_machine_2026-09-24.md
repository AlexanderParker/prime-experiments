# Record of results for six angles, assembled 2026-09-24

The six angles are:
- **A.** Derived machine on the gears
- **B.** Escape set against its cutoff
- **C.** Hand-off chain
- **D.** Protected stretch in two coordinates
- **E.** Neighbour kills and gaps
- **F.** Path B: forced pairs and the orbit statement

**Assembly checks.** I wrote six scripts and ran each one in the foreground; all finished. The scripts are in `C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/assemble_0924/`:

| Script | Time |
|---|---|
| v1_derived.py | 6.1 s |
| v2_esc.py | 9.8 s |
| v3_handoff.py | 2.9 s |
| v4_stretch.py | 7.0 s |
| v5_nk.py | 9.2 s |
| v6_pathB.py | 288.8 s |

None of them disagreed with the adjudicated record. `[Asm: ...]` marks a statement these scripts reproduced, with the range used. Nothing under `C:/dev/primes` was modified.

**Notation.**
- Case κ ∈ {1, 19}.
- (A, B, c, B') = (28, 30, 29, 30) for κ = 1, and (10, 12, 11, 10) for κ = 19.
- L_1 = {±1, ±11} mod 30 and L_19 = {±7, ±13} mod 30.
- "Lattice" means all residues mod h; "units" means the nonzero residues.

---

## 1. Statements that stand

### A. Derived machine on the gears

- **A1.** J(x) = (x² + c)/30 is an integer on L_κ, and its legs are 30J − 1 = x² + A and 30J + 1 = x² + B.
  - J is strictly increasing for x ≥ 1.
  - The images of the two cases are disjoint, because (y − x)(y + x) = 18 has no factorisation into two factors of equal parity. PROVED.
  - [Asm: x ≤ 2·10⁵, 0 repeated J.]
- **A2.** The lattice class count is |E_h| = 2 + (−A/h) + (−B/h). It equals the unit count e_h except at (7,1), where the lattice count is 1 (class {0}) and the unit count is 0.
  - [Checked h = 7..3000.]
  - [Asm: h ≤ 2000, 0 failures. The e-table for 7..61 matches the brief. E^(1) ∩ E^(19) = ∅ on units for h ≤ 2000.]
- **A3. Separation law.** For classes u, v struck by h, (v − u)(v + u) ≡ τ_v − τ_u mod h, with τ = x² = ε − c.
  - The difference lies in {0, ±2} for a same-case pair and in {±16, ±18, ±20} for a cross-case pair.
  - Exactly, 30(J(v) − J(u)) = v² − u² + (c_v − c_u).
  - Mod h, 30ΔJ ≡ ε_v − ε_u ∈ {0, ±2}. This is the original leg rule (h | ΔJ or h | 15ΔJ ± 1) for both same-case and cross-case pairs.
  - The identity 30ΔJ = v² − u² holds only within one case.
  - [Checked lattice h ≤ 3000, and gear pairs ≤ 5000 (54,622 same-case, 19,664 cross-case). The cross-case offset is exactly ±18.]
- **A4. Leg rule.** On units, N(D) = e_h[h|D] + [h | D² + 4A] + [h | D² + 4B] + 2[h | D⁴ + 2(A+B)D² + 4], with e_h the unit count.
  - It is exact except at (7,1) with D = 0, where N = 0 and the formula gives 1.
  - On the lattice it is exact everywhere.
  - Proof: ordered pairs split into identity, fold and cross pairs.
  - [Checked 2,375,252 cases, h ≤ 3000.]
  - [Asm: h ≤ 400. The only mismatch is (7,1,D=0,N=0,formula 1).]
- **A5. Type coincidences.** At residue level the type conditions coincide only for h | 4(4A − 1)(4B + 1):
  - fold-B meets cross at 11 (D = ±1) in case 1 and at 7 (D = ±1) in case 19;
  - fold-A meets cross at 37 (D = ±6) in case 1 and at 13 (D = ±5) in case 19;
  - there is one lattice-only coincidence at (7,1), D = 0, on the zero class, which holds no gear.

  At gear level (repaired), the strikers with an acting gear pair meeting two type conditions are exactly {11, 37} in case 1 and {7, 13} in case 19, at any D.
  - [Checked residues h ≤ 3000 and gears ≤ 20000.]
  - [Asm: (4A−1)(4B+1) = 3·11²·37 and 3·7²·13; (4A+1)(4B+1) = 11²·113 and 7²·41.]
- **A6. Bridging (on primes).** Twin gears share leg strikers only from {11, 29, 31} (same case, g ≡ 29 mod 30) or from {13, 23, 37} (cross case, g ≡ 11 or 17 mod 30). At residue level, 7 appears only through the (7,1) zero class.
  - [Asm: all twin gears ≤ 10⁶; the shared prime factors ≥ 7 are exactly these sets.]
- **A7. Acting in the coordinate x.** For prime h ≥ 7 and lattice x ≥ 7, h acts on x iff h ≤ x, and h = x never strikes. This is exactly h² ≤ 30J(x) + 1.
  - Any statement about acting and striking on columns x is a statement about copies J(x). "Only in the derived machine" is withdrawn.
  - [Checked 266,665 columns up to 10⁶.]
  - [Asm: x ≤ 20000.]
- **A8. Symmetry.** On units, the stabiliser of E_h is {x → ±x} for every non-inert striker except (29,1).
  - At (29,1) it is {x → ux : u ∈ {±1, ±12}}, with 12² ≡ −1 mod 29. This swaps the legs: E = {1, 28} on leg A and {12, 17} on leg B.
  - On the lattice there is one more exception, (7,1).
  - No stabiliser contains a translation.
  - J(x') ≡ −J(x) is solvable for every x only at (29,1).
  - [Checked h ≤ 3000.]
  - [Asm: units, h ≤ 1000, multiplications and translations.]
- **A9.** ±1 is a struck class exactly at {29, 31} in case 1 and {11, 13} in case 19. [Asm: h ≤ 3000.]
- **A10. Joint classes (repaired).** For strikers h ≠ h' on one case:
  - jointly struck classes mod hh' number (n_A(h) + n_B(h))(n_A(h') + n_B(h'));
  - split-leg classes number n_A(h)n_B(h') + n_B(h)n_A(h');
  - n_T(h) = 1 + χ_{−T}(h), except n_A(7) = 0 on units in case 1.
  - On units the joint count is e_h·e_h' ∈ {0, 4, 8, 16}. The lattice gives the same, except (7,1) paired with h' gives e_h' ∈ {0, 2, 4}.
  - PROVED by CRT. [Checked 420 residue cases and 112 raw full-period cases.]
- **A11. Derived period and range.** The derived period is q#, and the derived range is x ∈ (√(q − A), √(q# − B)].
  - Each nonzero survivor class occurs once.
  - As an interval the copy range is shorter: q = 7 gives copies 1..6.
- **A12. Lattice hand-off (restatement of the brief).** The chain condition is x_{i+1}² + B ≤ (x_i² + A)#. Only the sufficient direction is proved.
  - Instance: c_7 = copy 2, with legs 59 and 61, is revealed and serves q = 7..53.
  - First revealed composite columns: 221, 451, 781, 989, 1111, 1189 (case 1) and 427, 517, 1477, 1513, 1687, 1757 (case 19).
  - Case-19 distances mod 30: {0, 4, 6, 10, 14, 16, 20, 24, 26}.
- **A13.** One acting striker can strike the missed copies of three consecutive same-case gears. First incidences:
  - case 1: (11: 59, 61, 71) and (11: 269, 271, 281);
  - case 19: (7: 17, 23, 37) and (7: 47, 53, 67).
  - [Asm: gears < 20000.]
- **A14. Items standing "as stated".** Items (4), (7), (10) and (11) stand as stated. Their numbered texts are not in the assembly input. The formula-block content they carry is below.
  - **Cross-separation parametrisation:** E_h = {±(s²+2)/(2s)} ∪ {±(s²−2)/(2s)}, with s a root of y⁴ + 2(A+B)y² + 4. [Asm: algebraic check. With x = (s²+2)/(2s), x² + A = (s⁴ + (4+4A)s² + 4)/(4s²), and 4 + 4A = 2(A+B) because B = A + 2. Leg B is the same.]
  - **Classification by state:** inert, A-only, B-only or both, by ((−A/h), (−B/h)). (7,1) is inert on primes and a 1-class striker {0} on the lattice.
  - **Window thresholds:** in any set of same-case lattice columns spanning ≤ W, a striker h > W²/4 + B strikes at most 2 columns, and h > W⁴/4 + (A+B)W²/2 + 1 strikes at most 1.
    - [Refuter's rebuild to 3·10⁵.]
    - [Asm: W = 2..8, h ≤ 4000, full periods 30h, 0 violations.]
    - Argument: D is even; a fold needs h ≤ (D² + 4B)/4; a cross needs h ≤ quartic/4; three struck columns contain a same-leg pair.

### B. Escape set against the cutoff

- **B1.** D_h ∩ E_h = ∅, by a parity argument.
  - D_h is empty exactly at {7, 11, 13, 17} in case 1 and {7} in case 19.
  - D_19^(1) = {2, 17} and D_11^(19) = {2, 9}.
  - [Checked rows 7..3000; raw silence test h ≤ 400 (82,920 tests).]
- **B2.** Every striker of c_g is live, so no striker lies above λ(g). Hence c_g is revealed iff g mod 30∏_{7≤h≤λ(g)} h lies in Esc({7..λ(g)}).
  - λ(g) = h* whenever h* > 2g/3; with B5(d) this covers every gear g ≥ 41.
  - Exceptions: λ(7) = λ(11) = 0, λ(13) = 7, λ(23) = 13, λ(37) = 23.
  - [Checked full-definition λ for all 664,576 gears ≤ 10⁷.]
  - [Asm: gears ≤ 2·10⁵, 0 mismatches, same exception list, revealed criterion 0 failures.]
  - Measured: min λ/g on [1000, 10⁷] is 0.9398 (at 1129); max (g − λ − d0) is 208 (at 1,469,893).
- **B3.** λ ≥ 7 for case-19 gears > 7, and λ ≥ 11 for case-1 gears > 11. λ = 0 only at 7 and 11. PROVED.
- **B4.** For every level-{7} escape class and every later non-inert row h, some prime gear g > h in the class is struck by h. Infinitely many such gears exist by Dirichlet, inherited from the brief.
  - [Checked 360 pairs (classes mod 210 × rows 11..61). The largest first struck gear is 54011 (case 1, class 41, row 47).]
- **B5 (repaired).**
  - **(a), (c):** for g ≥ 19, λ(g) ≤ g − d0(g) and d0 − 1 < g/3.
  - **(b):** holds for g ≥ 11.
  - **(d):** λ(g) > 2g/3 for every gear g ≥ 41. Hence Cap(P) ⊂ (P, max(37, 3P/2)] for every level P ≥ 7, with 37 attained at P = 23 (Cap(23) = {29, 31, 37}).
    - Proof pieces: the k = 1 lemma; full-definition λ for 41 ≤ g ≤ 10⁷; certified chains of non-inert primes (≡ 1 mod 7 in case 1, ≡ 1 mod 3 in case 19) on [6.6·10⁶, 1.26·10¹²]; and BMOR 2018 for g ≥ 1.2·10¹⁰ (margins 0.0551g in case 1, 0.1662g in case 19).
    - [Asm: P ≤ 132,661. 0 violations; equality only at P = 23. Cap(7) = {11, 13}, Cap(13) = {17, 23}. λ(g) > 2g/3 holds for all gears 41..2·10⁵.]
  - **(e):** there is at most one capped gear per class mod 30∏(7..P).
  - **(f):** Esc(q) = Pull(q) ∩ units.
    - Pull ∖ Esc is the non-unit classes with J ∈ S_q; these hold no gear > q.
    - |Pull| = 4∏(h − e_h − [κ = 1 and h = 7]).
    - For g > q: g ∈ Esc(q) iff J_g ∈ S_q.
    - For g ≤ q: J_g ∈ S_q iff c_g is revealed and g² + A > q.
    - Range clause: g ∈ Cap(q) implies q < g² + A and g² + B ≤ q#.
    - [Asm: q = 7, case 19 gives Esc 8, Pull 12, extras 7, 77, 133, 203.]
  - **Proved span:** P+ ≤ max Cap(P) ≤ max(37, 3P/2). The span ≈ √(2P)(1 + o(1)) is a measurement only. Decade maxima of (maxCap − P)/√(2P) are 2.387, 2.018, 1.709, 1.378, 1.223, 1.122, for P ≤ 6.25·10⁶.
- **B6.** For g ∈ Cap(P) with g ≠ P+, P+ is silent on g, and g mod P+ = g − P+.
  - [Checked P ≤ 10⁵.]
  - [Asm: P ≤ 132,661, 0 strikes.]
- **B7. Entering law.** Row λ(g) strikes c_g iff g − λ is an even root of x² ≡ −A' mod λ.
  - κ = 2 iff (d + 1)² = 2g + 1 − B', and then d = d0.
  - In that case g² + B' = h·((d+2)² + B')/2, from 4(g² + B') = (d² + B')((d+2)² + B').
  - Converse PROVED: if h = (d² + B')/2 is prime ≥ 7 and g = h + d is a gear of the matching case, then h = λ(g).
  - [Checked ≤ 10⁷: 42 kills, 39 with κ = 2 (21 on leg 10, 18 on leg 30) and 3 with κ = 4: 23→31 (leg 28), 67→83 (leg 12), 199→227 (leg 12).]
  - [Asm: ≤ 2·10⁵, 13 with κ = 2 and the same three with κ = 4.]
- **B8. Doomed class.** C ∈ Esc(P) is doomed iff no escaping descendant at any level P' ≥ P meets Cap(P'). This iff stands.
- **B9.** Every escape class at levels 7, 11, 13, 17 and 19, in both cases, holds a gear g > level whose c_g is revealed.
  - Class counts: 24+8, 144+64, 1440+512, 20160+8192, 362880+114688.
  - The largest witness is 42,561,936,551.
  - [Asm: level 7 reproduced; level-13 counts reproduced by direct enumeration.]

### C. Hand-off chain

- **C1 (H1).** With P = (g² + A_g)#, the following are equivalent (PROVED):

  > g'² + B' ≤ P ⇔ g'² ≤ P − 41 ⇔ J_{g'} < P/30 ⇔ σ(g'² + B') ≤ g² + A_g.

  - X(g) = isqrt(P − 41), and X(7) = 43,849,291,330.
  - X(11), X(13) and X(79) have 29, 36 and 1341 digits.
  - X is non-decreasing along the gears.
  - X increases strictly from g to the next gear iff a prime lies in (g² + A, g'² + A']. This is PROVED when g is revealed; it is measured for g'² + A' ≤ 3·10⁸.
  - [Asm: X(7), the digit counts, and the four forms within ±3000 of X(7).]
- **C2 (H2).** A copy with legs (p, p+2) serves exactly the q in [σ(p+2), p).
  - chain(S) ⇔ S is infinite, p_0 + 2 ≤ 210, and p_{k+1} + 2 ≤ p_k#.
  - RANGE ⇔ chain(T). PROVED.
  - Measured for j ≤ 10⁷: |T| = 388,397, chain(T) holds, and the tightest ratio is 29/7 (at 29 → 59).
  - [Asm: j ≤ 10⁶, |T| = 50,715, chain holds, tightest ratio 29/7.]
- **C3 (H3).** chain is upward-closed. The path form "every next revealed gear lies in (g, X(g)]" is the hand-off equivalence restated.
- **C4 (H4).** If a hand-off is strong (g_{i+2} ≤ p_{i+1}), then Δ_{i+1} > Δ_i, i.e. p_{i+1}#·u_{i+1} > p_i#·u_{i+2}. PROVED.
  - Measured below 10⁷: 6,548 revealed gears and 6,547 hand-offs; all hold and all are strong.
  - Δ_0 = 43.9908.
  - First five budgets: 10.238%, 3.949%, 5.388%, 0.146%, 0.117%.
  - Maximum ratio 79/13. Maximum gap 14,160 (8,541,301 → 8,555,461).
  - The largest revealed gear ≤ X(7) is 43,849,288,261 = X(7) − 3069. [Asm, deterministic Miller-Rabin.]
  - No revealed gear lies in (g*, X(13)], where g* = X(13) − 107,292. The rejections are exact; g* itself is BPSW-probable only.
- **C5 (H5).** For every integer n ≥ 2, n ∈ G iff n lies in a level-(n−1) escape class. PROVED.
  - H* values: H*(7) = 29, H*(11) = 71, H*(13) = 89, H*(79) = 3169. [Asm]
- **C6 (H6).** The regions [J_g, sh(g')) do not overlap. They tile every j ≥ 2 except the case-1 head copies (h² − 1)/30, whose upper leg is h².
  - Every revealed j ≥ 2 lies in the region of its top acting gear. Copy 1 is revealed and lies below every region.
  - chain(S_region) ⇔ RANGE.
  - Empty stretches through gear 17,299: 17, 29, 37, 41, 149. Empty regions: 17, 29, 41.
  - l ≥ 3 for g ∈ [31, 20000].
  - Corrected tallies for j ≤ 10⁷: 980 copies lie in no region (copy 1 plus 979 head copies). Of the revealed copies, exactly 1 (copy 1) lies in no region.
  - [Asm: j ≤ 10⁶, 354 such copies, all copy 1 or head copies; copy 1 is the only revealed one.]
- **C7 (H7).** The gears acting on c_m = copy ⌈(m² + 1)/30⌉ are exactly the primes 7..m, for every m ≥ 0 except m = 6 and m = 10. PROVED.
  - Forward arrows PROVED, with family bound m ≥ 7: chain(G) ⇒ chain(SQ_unit) ⇒ chain(SQ_all) ⇒ RANGE. See note N3.
  - Square-family class counts: 24/12, 168/108, 1848/972, 27720/16524.
  - [Asm: m ≤ 3000, exceptions {6, 10}.]
- **C8 (H8 i). Witnesses** (least revealed copy with lower leg > q):
  - copy 1 for q = 7..23;
  - c_7 for 29..53;
  - c_11 for 59..139;
  - c_13 for 149..173;
  - copy 8 = (239, 241) at q = 179 (gear 13, position 3).

  Of the 72 binding machines, the witness is a missed copy only at q = 59 and q = 149. [Asm: q < 200.]
- **C9 (H8 ii), measured for j ≤ 10⁷.**
  - chain(T − S_1) and chain(T − stretch copies) hold.
  - With SQ taken over m ≥ m0 ≥ 6, chain(T − SQ) and chain(T − (S_1 ∪ stretch ∪ SQ)) hold.
  - The revealed copies with upper leg ≤ 210 are 1, 2, 5 and 6.
- **C10 (H8 iii), reduction (PARTIAL).** RANGE ⇒ chain(G) is exactly chain(S_region) ⇒ chain(S_1).
  - Position lemma PROVED: a copy at position n ≥ 2 of gear g has legs strictly between g² + 30 and g'².

### D. Protected stretch in two coordinates

- **D1. Field.**
  - |C_h| = 2(h − 1).
  - Each column has 2 points, 15⁻¹ apart, and h − 2 free cells.
  - Rows with 4, 2 and 0 points number (h−3−a−b)/4, (h+1+a+b)/2 and (h+1−a−b)/4, with a = (2/h) and b = (−2/h).
  - There are (h − 1)/2 phases.
  - C^(19) = C^(1) + (0, 3·5⁻¹).
  - Each column has ∏(h − 2) free n-classes mod ∏H.
  - [Checked h ≤ 211 raw.]
  - [Asm: |C_h| and the row distribution, h ≤ 400.]
- **D2. Comb.**
  - Gaps alternate m_h and h − m_h.
  - Three consecutive teeth span exactly h.
  - Two teeth are D apart iff h | D(225D² − 1).
  - There is at most 1 tooth per m_h consecutive cells and at most 2 per h cells.
  - [Checked gears ≤ 12000: 7,558,134 strikes.]
- **D3.** Per striker, the tooth count lies between 2⌊l/h⌋ and 2⌈l/h⌉.
  - The first stretch where a striker takes 3 cells is g = 23 (striker 7, cells 1, 7, 8).
  - The first with a striker outside {7, 29, 31} is g = 43 (striker 11, cells 1, 9, 12).
  - [Asm]
- **D4.** Two strikers cover three consecutive cells only if 7 takes an adjacent pair or 29/31 takes the end cells; otherwise three distinct strikers are needed. No striker takes all three cells. PROVED.
- **D5. Depth caps.** W = ρ² + L + 30(n−1) = mh.
  - m ≥ 1, and m ≡ (ρ² + L)h⁻¹ mod 30.
  - m is odd when k is even. When k is odd, m is even and m ≡ L + 2(n−1) mod 4.
  - Silence floors: h − B for every k, and 2h − B'_n when k is odd.
  - The first-tooth formula holds (m0 = 30 only when L = 30 and 30 | ρ).
  - The top band h = g − d is silent.
- **D6. Fold.**
  - Stretches are disjoint and ordered.
  - A stretch is covered iff it is comb-covered iff it contains no twin copy.
  - Stretches abut (l = L) iff d < s(g mod 30), or g = 13.
  - 30·sh(g') + 1 = g'² for case-1 g', and J_{g'} = sh(g') + [g' case 1].
  - L = p1 only at 13.
  - The p1 class formula holds.
  - [Checked all 5,761,452 gears ≤ 99,999,989.]
  - [Asm: gears ≤ 10⁶.]
- **D7.** For every N and K, the statement "every K consecutive gears include one with a revealed cell among its first N cells" is false (CRT + Shiu).
  - For g ≥ 31, l(g) ≥ (g − 14)/15, with equality only for g ≡ 29 mod 30.
  - [Asm: g ≤ 10⁶.]
- **D8. Fixed finite striker sets (PROVED).** No fixed finite set S of strikers forces a revealed cell at fixed depth.
  - For any unit class a mod P_S = 30∏S, and any N and K, there are K consecutive gears ≡ a with l ≥ N and every cell n ≤ N struck by a prime below the gear.
  - The construction ran to N = 10 over all 5760 classes. The K gears being consecutive rests on Shiu.
- **D9.** The fully covered stretches among gears 7..99,999,989 are exactly 17, 29, 37, 41 and 149 (l = 2, 1, 7, 5, 9).
  - The only consecutive covered pair is {37, 41}, and there is no run of three.
  - [Asm: gears ≤ 10⁶.]
- **D10. Covered-run tables for H = 7..y**, y = 13, 17, 19, 23, 29:

  | y | D_all | D_sq (case 1, case 19) |
  |---|---|---|
  | 13 | 5 | (1, 4) |
  | 17 | 7 | (6, 6) |
  | 19 | 12 | (8, 10) |
  | 23 | 18 | (14, 13) |
  | 29 | 25 | (22, 17) |

  - D_sq uses the forward-prefix reading.
  - In each case D_sq for the next gear's case is ≥ that gear's l: 4 ≥ 2, 6 ≥ 5, 10 ≥ 10, 14 ≥ 1, 22 ≥ 13.

### E. Neighbour kills and gaps

- **E1.** The parts of K(d) are disjoint, and h kills c_{h+d} iff h ∈ K(d).
  - Table A4.2 equals K*(d) for even d ≤ 30.
  - The members of K(d) not in K* for d ≤ 30 all have 3 | h + d: d = 8: 19; d = 14: 7, 13, 103; d = 20: 103; d = 22: 257; d = 26: 7; d = 28: 11.
  - K(6) is empty.
  - This restates A4.2.
  - [Asm: 32,230 kills with g ≤ 10⁵, all in K(g − h).]
- **E2.** max K(d) ≤ (d² + 30)/2.
  - The top value is attained iff (d² + 30)/2 is prime and d ≡ 1 or 2 mod 5.
  - If the top member also has h + d prime, then d ≡ 2 or 26 mod 30.
  - This restates A4.2.
- **E3. Distance law.** (d+1)² ≥ 2g − 29 in case 1 and ≥ 2g − 9 in case 19.
  - Equality holds exactly on F30 and F10.
  - No killer lies in (g + 1 − √(2g − 29), g).
  - F30 up to 10⁶: d = 2, 26, 362, 392, 752, 1166.
  - F10 up to 10⁶: 11 instances, all with d ≡ 4 or 24 mod 30.
  - This restates A4.5/A4.6.
  - [Asm: kills with g ≤ 10⁵, 0 violations.]
- **E4.** κ = (d² + A')/h = h + m − 2g, with the 2-, 3- and 5-adic classes of A4.4. This restates A4.4.
- **E5.** ALL-KILLED(W) iff every g ∈ W has a composite leg. The proof step is repaired: √(g² + 30) < g + 2.
- **E6 ((6) repaired).** Let K°(d) = {h ∈ K(d) : h ∤ d}. For even d and D ≥ 2, K°(d) ∩ K°(d + D) ⊆ S(D).
  - Exact form with parts: branch (i) is a = b, h | D; branch (ii) is h | R_ab(D) with d ≡ x0.
  - K*(d) ∩ K*(d + D) ⊆ S(D).
  - K(d) ∩ K(d + D) ⊆ S(D) ∪ {7 : 7 | d(d + D)}.
  - [Checked 7,701,120 cases.]
- **E7 (summary piece 5, repaired).** For a prime p ≥ 7, p | g² + a and p | (g + D)² + b iff one of:
  - (i) a = b, p | D and p | g² + a; or
  - (ii) p ∤ D, p | R_ab(D) = (D² + a + b)² − 4ab, and g ≡ x0 = (a − b − D²)(2D)⁻¹ mod p.

  [Checked 14,923,808 residue cases.]
- **E8. Sharing law.** Every shared prime factor ≥ 7 of the legs of two gears D apart lies in S(D).
  - Members of S(D) are < (D² + 58)²/4.
  - Privacy: a prime above H(w) divides a leg of at most one gear in any window of width w.
  - Residue sentence (repaired): 'no gear ≤ H(w) kills g' ⇔ g mod M ∈ Esc(H(w)) ∪ {g' mod M : g' ≤ H(w) revealed}.
  - [Checked consecutive gear pairs ≤ 10⁷: 206,715 shared primes, 0 outside S(D).]
- **E9.** Twin gears share killers only from {11, 29, 31} (g ≡ 29 mod 30) or {13, 23, 37} (g ≡ 11, 17 mod 30). 7 never occurs.
  - [Asm: twin gears ≤ 10⁶.]
- **E10. H values.**

  | w | H(w) |
  |---|---|
  | 2 | 37 |
  | 4..6 | 449 |
  | 8..12 | 601 |
  | 14 | 619 |
  | 16..24 | 21,841 |
  | 26 | 128,521 |
  | 28 | 176,401 |
  | 30..32 | 228,601 |
  | 154 | 140,873,041 (< 11887² = 141,300,769) |

  [Asm: reproduced from realizable S(D) with explicit roots. R_28,10(4) = 1796 = 4·449.]
- **E11.** Gears ≡ 301 mod 330 are case 1 with 11 | g² + 28. By Shiu there are arbitrarily long runs of consecutive gears whose missed copies are all killed.
- **E12.** The hand-off equivalence is restated in the notation of (5), (6) and (8), with intervals (g_i, G_i]. This is LOCATING.
- **E13.** Table A4.2 is K*(d), not K(d).

### F. Path B: forced pairs and the orbit statement

- **F1. Tie identities.**
  - (30x−1)(30x+1) − (30y−1)(30y+1) = 900(x² − y²).
  - (30x−1)(30x+1) − (6n−1)(6n+1) = 36(25x² − n²).
  - These tie strikes for every prime g ≥ 5; at g = 5 the tie holds only through 5 | n.
  - [Asm: identities checked.]
- **F2.** The G(x) bounds of (2) hold with the reported failure sets. M_X ≥ X'⁴ for X ≥ 19 (M(19) = 323,323 ≥ 23⁴ = 279,841), and this fails at X = 17.
- **F3.** Forced pairs never go upward: forced ⇒ j' < j. PROVED.
  - [Asm: forced (u, j) totals 3, 16, 65, 166, 539 and distinct (j, j') totals 3, 16, 62, 152, 462 for q = 7..19; 0 upward.]
- **F4. Preservation.** On a forced pair, REV(j) ⇒ REV(j'). [Checked q = 7..23: 0 violations.]
- **F5. Criterion.** A pair is forced iff P(q, sh(j')] | ab, with the given parametrisation. For u = −1: forced iff j' < M_q/2 and P(q, sh(j')] | M_q − 2j'.
- **F6. Reach.** P(q, sh(j')] ≤ ab < M_q.
  - A source with j² < M_q has no forced target.
  - There is at most one forced target per shelf and one in the window.
- **F7. Orbit (repaired).** F = {x : 25x² ≡ n² mod M_X}, with n = 5j + e and X = sh(j).
  - |F| = 2^(k − w), with k = #primes in [7, X] and w = ω(gcd(n, M_X)).
  - The column orbit has size 2^(k+1−w) and maps 2-to-1 onto F.
  - Over machine q, F splits into ∏_{X<g≤q}(g+1)/2 orbits, of which ∏_{X<g≤q}(g−1)/2 are survivor orbits.
  - The machine orbit is one of these orbits, and is all of F iff X = q.
  - Top shelf: every F class lies in [jmin(X), M_X − 1] for X ≥ 19.
  - Item (v) stands.
- **F8. Measurement.** Every window exposure has a revealed orbit copy in the range for q ∈ [7, 61]; this is exact. For q = 67..113 the witnesses are BPSW-probable.
- **F9.** Exposures (0,2), (0,3) and (1,2) have M = 1. Exposure (0,2) gives exactly the range statement.
- **F10.** The orbit statement is equivalent to the fibre chain from q_e = max(7, sh(j)), with 30y⁺ + 1 ≤ (30y − 1)#. This is an equivalence, not a claim.
- **F11. Items (12)(b)–(d).**
  - The exposure-to-own-fibre forced ties are exactly (n, x) = (2,1), (3,1), (7,1), (12,1), (3,2), (17,2), (18,2), (3,5), (32,9). The list is complete for j ≤ 300, and every target is revealed.
  - Tied pairs lie only on copies {1, 2, 5, 6, 8, 9, 14, 27}.
  - Transfers land on {1, 2, 5, 6, 8, 9}; the last is at q = 263.
- **F12. Item (13).**
  - (a) For j ≥ 12, every fibre copy lies above its column: 5x > n.
  - (b) An untied acting gear lies in (X, sh(x)].
  - (c) Stands.

---

## 2. Formulas and classifications

### 2.1 The derived machine beside the original

| Feature | Original machine | Derived machine (case κ) |
|---|---|---|
| Points | Columns n with pair (6n−1, 6n+1); copies j with legs 30j∓1 | Columns x ∈ L_κ (x² ≡ 1 or 19 mod 30); gear version: primes in L_κ |
| Lower set | {2,3,5}; lap of 5 columns, open offsets 0,2,3; known opening (−1,1) at each lap head | Mod-30 condition is L_κ, in two cases. No derived lap-machine centre was built |
| Copy carried | Copy j | J(x) = (x² + c)/30. Injective and increasing; the case images are disjoint. For prime x, J(x) = J_x (the missed copy c_x) |
| Strike | g strikes j iff j ≡ ±a_g mod g (a_g = 30⁻¹): 2 classes | h \| (x²+A)(x²+B) = 900J² − 1 iff x mod h ∈ E_h = {±r_A, ±r_B} = J⁻¹{±30⁻¹} |
| Class count | 2 per gear | \|E_h\| = 2 + (−A/h) + (−B/h) ∈ {0, 2, 4}; lattice count 1 at (7,1) |
| Striker states | none | Inert / A-only / B-only / both, by ((−A/h), (−B/h)) |
| Acting | g² ≤ 30j + 1 | h ≤ x (the same relation); h = x never strikes |
| Revealed | No acting striker; for j ≥ 1, both legs prime | The same, on J(x). Composite x can be revealed (221, 451, ...; 427, 517, ...) |
| Period | q# | 30∏H in x for striker set H; derived period q# |
| Range | 30j − 1 > q, 30j + 1 ≤ q# | x ∈ (√(q−A), √(q#−B)]; each nonzero survivor class once |
| Pair rule | h \| D or h \| 15D ± 1 (linear forms) | N(D) = e_h[h\|D] + [h\|D²+4A] + [h\|D²+4B] + 2[h\|D⁴+2(A+B)D²+4] |
| Pair polynomials | D, 15D ± 1 | Case 1: D²+112, D²+120, D⁴+116D²+4. Case 19: D²+40, D²+48, D⁴+44D²+4. Cross-case: D⁴+76D²+324, D⁴+80D²+256, D⁴+80D²+400, D⁴+84D²+324. General: R = D⁴ − 2(τ+τ')D² + (τ−τ')² |
| Separation | 30ΔJ ≡ ε_v − ε_u ∈ {0, ±2} | (v−u)(v+u) ≡ τ_v − τ_u: {0, ±2} same case, {±16, ±18, ±20} cross case; 30ΔJ = v² − u² + (c_v − c_u) |
| Parametrisation | none | s a root of y⁴ + 2(A+B)y² + 4; E_h = {±(s²+2)/(2s)} ∪ {±(s²−2)/(2s)} |
| Symmetry | j → uj, u² = 1 mod M_q | Stabiliser {±x}, except (29,1): {±1, ±12}, which swaps legs; no translations |
| Special strikers | none | AP strikers h \| (4A−1)(4B+1): {11, 37} / {7, 13}. Adjacent-class h \| (4A+1)(4B+1): {11, 113} / {7, 41}. ±1 class: {29, 31} / {11, 13} |
| Twin bridging | none | Same case {11, 29, 31} (g ≡ 29); cross case {13, 23, 37} (g ≡ 11, 17) |
| Joint classes | none | (n_A+n_B)(n_A'+n_B') joint; n_A n_B' + n_B n_A' split-leg |
| Window thresholds | none | ≤ 2 struck columns per W-span for h > W²/4 + B; ≤ 1 for h > W⁴/4 + (A+B)W²/2 + 1 |
| Missed copy | c_g = copy ⌈(g²+1)/30⌉ | Column x = g |
| Chain | Hand-off (brief) | x_{i+1}² + B ≤ (x_i² + A)#; sufficient direction only |

### 2.2 Escape set against the cutoff

- **Case c(g):** (A, B, B') = (28, 30, 30) for g ≡ ±1, ±11 mod 30, and (10, 12, 10) for g ≡ ±7, ±13 mod 30.
- **Dormancy set:** D_h^c = {r unit : ρ² < h − B, or (ρ even and ρ² < 2h − B')}.
  - |D_h^c| = 2#{1 ≤ x < h/2 : x² < h − B} + 2#{even x : h − B ≤ x² < 2h − B'}.
- **Row split:** units = E_h ⊔ D_h ⊔ F_h. The 0-child holds only the gear h.
- **Live row:** h is live on g iff 7 ≤ h < g, e_h^{c(g)} > 0, and g mod h ∉ D_h. λ(g) is the maximum live row.
- **Formula for λ:** d0(g) is the least even d ≥ 2 with (d+1)² ≥ 2g + 1 − B'. h* = max{h prime ≤ g − d0 : h non-inert}. λ = h* when h* > 2g/3.
- **Revealed criterion:** g mod 30∏_{7≤h≤λ} h ∈ Esc({7..λ}).
- **Capped set:** Cap(P) = {g > P : λ(g) ≤ P}.
- **Entering identity:** 4(g² + B') = (d² + B')((d+2)² + B').
- **Pull:** |Pull(q)| = 4∏(h − e_h − [κ = 1, h = 7]).
- **Doomed-class criterion:** see B8.

### 2.3 Hand-off chain

- g → g' holds iff g' ≤ X(g) = isqrt((g² + A)# − 41), iff J_{g'} < (g² + A)#/30, iff σ(g'² + B') ≤ g² + A. Here σ(y) = min{q prime : q# ≥ y}.
- Serving interval: [σ(p+2), p).
- RANGE ⇔ p_0 + 2 ≤ 210 and p_{k+1} + 2 ≤ p_k# on all revealed copies.
- chain(G) ⇔ G meets W(g) = (g, X(g)] for every g ∈ G.
- Δ_i = θ(g_i² + A_i) − ln(g_{i+1}² + B_{i+1}), and Δ_0 = ln(59#/151).
- H*(g) = max{H : H# ≤ X(g) − g}.
- G = {n : n lies in a level-(n−1) escape class}.
- Square-family classes: 4∏(h − |R_h|), with R_h = E_h except R_7^(1) = {0}.
- **Implication lattice (forward arrows only):**
  - chain(S_1) = chain(G) ⇒ chain(S_{≤N}), chain(S_stretch) ⇒ chain(S_region) ⇔ RANGE;
  - chain(S_1) ⇒ chain(SQ_unit) ⇒ chain(SQ_all) ⇒ RANGE.

### 2.4 Protected stretch

- **Cell:** (g, n) = copy J_g + n − 1.
- **Field:** C_h = {(x, y) : x² + 30(y−1) ≡ −A or −B}. These are the level sets y = λ_h(x) and y = λ_h(x) − t_h, with λ_h(x) = 1 − (x² + A)·30⁻¹ and t_h = 15⁻¹ (note N4).
- **Comb:** teeth {λ_h(g), λ_h(g) − t_h} + hZ, with m_h = min(t_h, h − t_h) ≥ (h − 1)/15.
  - m-lists [Asm, h < 2·10⁵]:

  | m | primes h |
  |---|---|
  | 1 | 7 |
  | 2 | 29, 31 |
  | 3 | 11, 23 |
  | 4 | 59, 61 |
  | 5 | 19, 37 |
  | 6 | 13, 89 |
  | 7 | 53 |
  | 8 | 17 |

- **Caps:** see D5. B'_n = 30 or 28 in case 1 (n odd, n even), and 10 or 12 in case 19. Arcs: n = 1 + (mh − L − (g − kh)²)/30.
- **Abut table s** by g mod 30:

  | g mod 30 | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
  |---|---|---|---|---|---|---|---|---|
  | s | 14 | 3 | 4 | 5 | 3 | 11 | 10 | 1 |

- **p1:** p1 = (k·g − leg)/30, with (k, leg) by the same classes:

  | g mod 30 | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
  |---|---|---|---|---|---|---|---|---|
  | (k, leg) | (28,28) | (6,12) | (8,28) | (10,10) | (6,12) | (22,28) | (20,10) | (2,28) |

- **Stretch length:** l = min(p1, L), with L = sh(g') − J_g and sh(g') = ⌈(g'² − 1)/30⌉.

### 2.5 Neighbour kills

- **Killers at distance d:** K(d) = ⊔_{A'} {h ≥ 7 prime : h | d² + A', (h + d)² ≡ s_{A'} mod 5}. Here s_28 = s_30 = 1 and s_10 = s_12 = 4. Every element lies in [7, (d² + 30)/2].
- **K°(d)** = {h ∈ K(d) : h ∤ d}. **K*(d):** h + d prime.
- **Mirror:** leg = (g − d)(g + d + κ), with κ(g − d) = d² + A'.
- **ALL-KILLED** with r0(k).
- **S(D):** branches (i)/(ii) of E7, realizable, with explicit roots.
- **H(w)** = max_{D≤w} S(D) < (w² + 58)²/4.

### 2.6 Path B

- **Forced:** j'² ≡ j² mod M_{max(q, sh(j'))}.
  - Parameters: a = (j − j')/gcd(u − 1, M_q), b = (j + j')/gcd(u + 1, M_q), t = −ab.
  - j = (a·d₊ + b·d₋)/2 and j' = (b·d₋ − a·d₊)/2, with a ≡ b mod 2.
- **r_Y(j):** the unique r < √M_Y with r² ≡ j² mod M_Y.
- **Orbit:** as in F7, with k counted from 7.

---

## 3. Statements refuted, with instances

**A. Derived machine**
- **Cross-case identity.** 30ΔJ = v² − u² across cases is refuted. At h = 37, u = 3 (case 1), v = 5 (case 19): 30ΔJ ≡ 35 but v² − u² = 16. It fails on all 3200 cross-case ordered pairs.
- **Gear-level sentence of (6).** The worker's set {7} for case 19 and the explanation "not a case-19 distance" are refuted.
  - Instance (17, 103), D = 86: 13 | 299, 13 | 10621, and 13 divides both D² + 40 and the quartic.
  - D ≡ 26 mod 30 is an allowed case-19 distance.
  - The set {7} came from the cut D < 2h.
- **Original clauses of (15).**
  - The product clause fails on the lattice at (7,1) pairs, e.g. (7,13) has 2 joint classes (11 failures).
  - The split clause fails on units at (7,1) pairs with h' ∈ {11, 13, 17, 23, 29, 31, 37, 43, 47, 59}: the formula gives 2, the actual count is 0.
- **N(D) with e_h read as the lattice count.** Fails at (7,1), D = 0 (N = 1, formula 2).
- **(17).** The statement text is not in the input. It is refuted by the incidences in A13.
- **Struck and withdrawn.** (16)'s last sentence (complete periods of the smaller striker sets 7..y) is struck. (9)'s "only in the derived machine" is withdrawn.

**B. Escape set against the cutoff**
- **(10) conclusion refuted.** The statement was: "every revealed missed copy in machine q's range has λ(g) > q". At q = 13, c_7 = (59, 61) has λ = 0, c_11 = (149, 151) has λ = 0, and c_13 = (179, 181) has λ = 7.
- **(10) core (capped-strip reduction) refuted.** Cap(13) = {17, 23} and neither is revealed: 299 = 13·23, 301 = 7·43, 539 = 7²·11.
  - It fails at 50,139 of the 148,930 prime levels in [7, 2·10⁶]. The first failures are 13, 17, 19, 23, 29, …; the last is 1,999,993.
  - [Asm: Cap(13) and the legs reproduced.]
- **(4) refuted** by 360 struck pairs (see B4).
- **(5)(b) at g = 7:** pred 5 is not a level.
- **Original (5)(f) class identity.** At q = 7, case 19: Pull has 12 classes, Esc has 8.
- **(7) wording.** "d = d0 forces κ = 2" is corrected.
- **Remark struck.** (8)'s closing remark "cannot be decided at any finite level" is unproved and struck.
- **Count erratum.** 664,575 gears is corrected to 664,576.

**C. Hand-off chain**
- **H5 literal at n = 1.** 1 is a case-1 residue, and c_1 = copy 1 is revealed, but 1 is not a gear.
- **Heredity of chain.** {c_7} ∪ {c_g : g ≥ 43,849,311,221} leaves machine 59 unserved.
  - 43,849,311,221 is the least revealed gear above X(7).
  - Its upper leg is greater than 59# and σ = 61.
  - [Asm: deterministic Miller-Rabin.]
- **H6 tallies.** "0 of 388397" should be 1. "980" should be 979 for j ≤ 10⁷; the derivation's 980 ran to j = 10,000,568. The refuter's "980 = 979 + copy 1" is wrong.
- **H7 with family bound m ≥ 16.** chain(SQ) fails at q = 7, because the smallest upper leg is 271 > 210.
- **H8(i) "first time at q = 179".** Fails at q = 7, where the witness is copy 1, not a missed copy.
- **H8(ii) with SQ over all m ≥ 0 or all units m ≥ 1.** q = 7 is left unserved.

**D. Protected stretch**
- **Premise of (3).** "At most two teeth per stretch unless 7, 29, 31" is refuted at g = 43: striker 11 takes n = 1, 9, 12 (legs 1859, 2101, 2189 all divisible by 11).
- **(7) last sentence** as originally worded is false; it is replaced by D8.
- **Fixed-depth statement.** "Every K consecutive gears include one with a revealed cell among the first N" is false for every N and K.

**E. Neighbour kills**
- **Literal (6)** K(d) ∩ K(d + D) ⊆ S(D) is refuted by (d, D) = (14, 2).
  - 7 ∈ K(14) (224 = 2⁵·7) and 7 ∈ K(16) (266 = 2·7·19), but 7 ∉ S(2).
  - Every failure is h = 7 with shift 28, at d or d + D ≡ 14 or 42 mod 70.
  - [Asm]
- **Literal summary piece 5** is refuted by 11 and 151 with D = 140, a = b = 30, p = 7. Without p ≥ 7 it is also refuted by g = 3, D = 6: 39 = 3·13 and 93 = 3·31.
- **Proof step** "a composite leg is < (g+1)²" is false at g = 11 (151 > 144).
- **Row-by-row escape-class reading of the (8) residue sentence** fails exactly on the revealed gears ≤ H(w), e.g. {7, 11} at w = 4. The strike-only reading fails on the revealed gears with g² + A ≤ H(w).
- **K(8) versus table A4.2.** The brief's K(8) is {19, 23}, while table A4.2 has {23}. The table is K*.

**F. Path B**
- **V2.** Refuted at q = 19, n = 33 (exposure (6,3), legs 197 and 199).
  - The orbit of 33/5 mod 323,323 is 20614, 30437, 64658, 69047, 108130, 115709, 120098, 159181, 164142, 203225, 207614, 215193, 254276, 258665, 292886, 302709.
  - All 16 copies are in the range and none is revealed. For example, 108130 has legs 3243899 = 199·16301 and 3243901 = 37·73·1201.
  - There is also a degenerate failure at q = 7, n = 7: the orbit is {0}.
  - [Asm: the orbit, the range and non-revelation reproduced.]
- **(7)(ii) literal** (k counted over gears ≥ 5) is off by a factor of 2. Instance (2,2): legs 71 and 73, X = 7, F = {1, 6}; the literal formula gives 4.

---

## 4. Not established (open questions)

**A. Derived machine**
- Does each predicted bridging striker occur on some gear pair at distance D? For D ≥ 4, the observed sets up to 2·10⁵ are proper subsets.
- Which rational residues u/v are open under every striker? This is an S-unit search on (u² + Av²)(u² + Bv²), not run.
- The converse direction of the lattice chain. Do revealed composite columns shorten any hand-off? Not measured.
- A derived counterpart of the lap-machine centre.
- The distribution of fold half-separations.
- Consequences (1)–(3) of item (9): their text is not on disk.

**B. Escape set against the cutoff**
- Are the κ = 4 entering kills finite? Only g = 31, 83 and 227 occur up to 10⁷.
- Is Cap(q) ∩ Esc(q) = ∅ at infinitely many q?
- Is any escape class doomed? No structural statement decides it.

**C. Hand-off chain**
- RANGE ⇒ chain(G), equivalently chain(S_region) ⇒ chain(S_1).
- The reverse arrows chain(S_region) ⇒ chain(S_stretch) ⇒ chain(S_1), and chain(SQ) ⇒ chain(G).
- A property weaker than the hand-off that still implies it.
- Strict increase of X at every consecutive gear pair.
- A primality certificate for g* and its 71-digit legs.

**D. Protected stretch**
- Full covering of three or more consecutive gears beyond 10⁸.
- Is D_sq(primes < g) ≥ l(g) for all g > 31?
- Are only finitely many stretches fully covered?
- A version of D8 with N growing with g.
- The whole-stretch S-free variant.

**E. Neighbour kills**
- ALL-KILLED on the intervals (g_i, G_i].
- Is K(d) empty for infinitely many d?
- Are F30 and F10 infinite?
- Is each realizable member of S(D) realized infinitely often? For example, 21841 (D = 16) and 128521 (D = 26) are not realized below 10⁶.

**F. Path B**
- V1 (the orbit statement): not proved.
- V2 beyond q = 113.
- Control of the untied gears (X, sh(x)].
- Fibre chains other than via the range statement.
- A structural reason for the q = 19 V2 failure.

---

## 5. Computations that did not finish or were not run

- **F:** the first run of f1_forced.py for q = 19, 23 exceeded 300 s and was moved to the background by the harness. It was stopped after the q = 19 line (91.4 s), rewritten, and rerun in the foreground: q = 7..19 in 0.9 s, q = 23 in 65.9 s.
- **F:** all-pairs forced enumeration for q ≥ 29 was not run; M_29 is too large for the array method. Above-window pairs were checked only for q = 11..53. The V1 witnesses for q ≥ 67 are BPSW-probable and were not rerun.
- **B:** level-23 witnesses were not run. That is 8,825,600 classes, estimated at about 60 CPU-minutes in about 14 chunks (note N1).
- **A:** the S-unit search and the gear-level bridging converse were not run.
- **D:** D_sq at y = 31 was not computed (the period exceeds 6·10⁹). Consecutiveness in D8 cannot be computed at those moduli.
- Every other angle run finished in the foreground.
- The six assembly runs all finished. The longest was v6_pathB.py at 288.8 s, spent on the q = 19 forced enumeration.

---

## 6. Locating and counting residue in what stands

**A. Derived machine**
- **Locating:** A12 restates the brief's hand-off chain on the lattice, including its height condition. "x₀ = 7 serves q < 59" is a measured instance.
- **Counting:** counts appear only as flagged checks, in items (7), (8), (14) and (17). The class counts are root counts of quadratics mod h.
- **Scope:** the smaller-machine-period sentence of (16) is struck.

**B. Escape set against the cutoff**
- **Counting:** B5(d) rests on BMOR 2018, an explicit prime-counting estimate in arithmetic progressions, for gears above 1.26·10¹². This feeds B5(d), B5(e), the B5(f) range clause and B6's residue identification. Dirichlet in B4 and the escape-class products are inherited from the brief.
- **Locating:** these are height bounds on the capped set, not on a revealed copy:
  - B5(d): Cap(P) ⊂ (P, max(37, 3P/2)];
  - the B5(f) range clause;
  - B6;
  - the span measurement.

  The one reduction to the capped strip, (10) core, is refuted.

**C. Hand-off chain**
- **Locating form:** H1 and H3(iii) restate the target as "the next revealed gear lies in (g, X(g)]".
- **Locating measurements:** H3/H4 (the extreme revealed gears near X(7) and X(13), the maximum ratio and the maximum gap), and H8(i) (witnesses).
- **Size measures:** Δ and the budgets (θ); the monotone-Δ theorem, which is exact but a size-of-slack statement; and H*(g).
- **Tallies used as checks:** 388,397; 979/980; 6,548/6,547/328/233/95; 73/72; 207; the class products.

**D. Protected stretch**
- **Counts:** all are flagged checks.
- **Near-locating:** D10 (D_all/D_sq, Jacobsthal-type, class level); the S-free-cell remark in D8; the l bound; the per-gear first-revealed tables, which are measurements.
- **Imported:** D7 and D8 import Shiu.

**E. Neighbour kills**
- **Locating:** E12 places the next revealed gear in (g_i, G_i]. E8's privacy and H(w) are located only through E12.
- **Counting:** none is used as a result. E11 rests on Shiu.
- **Re-derivation:** E1–E5 restate A4.2, A4.4, A4.5/A4.6 and "revealed iff both legs prime".

**F. Path B**
- **Locating:**
  - F6: the reach bound and one target per shelf;
  - F10: the chain condition, which has the same shape as the Path A hand-off;
  - F11: finite lists (copies ≤ 9, transfers only for q ≤ 263);
  - F12(a): 5x > n;
  - F7: the top shelf locates classes.
- **Counted:** the orbit sizes 2^(k−w) and 2^(k+1−w), and the products ∏(g±1)/2, are stated as results. They are structural, not densities, and are not used for existence.
- **Checks:** the forced and exposure tallies.
- **Scope:** there is no range-to-window drift.

---

## Notes on the input found during assembly

- **N1.** Angle B's unfinished list quotes "8,798,208 classes" for level 23. The sum of the per-case counts is 6,531,840 + 2,293,760 = 8,825,600, from 4∏(h − 1 − e_h). Recomputed in v2_esc.py.
- **N2.** Angle B's unfinished list says "Not done: explicit g1 … measured only 41 ≤ g ≤ 10⁷". The standing repaired (5)(d) gives g1 = 41 with the four-piece proof. This record carries (5)(d).
- **N3.** H7's adjudicated text closes a cycle, "… ⇒ RANGE ⇒ chain(G)", citing the hand-off equivalence. The following all record RANGE ⇒ chain(G) and chain(SQ) ⇒ chain(G) as open:
  - H8(iii);
  - the formula block ("only the forward arrows are proved");
  - the not-established list.

  This record counts only the forward arrows as proved.
- **N4.** Two symbols are overloaded between angles:

  | Symbol | Angle D meaning | Other meaning |
  |---|---|---|
  | λ | λ_h(x), the field level-set function | Angle B: λ(g), the top live row |
  | sh | sh(g'), a gear's shelf start (also angle C) | Angle F: sh(j), the top acting gear of a copy |

- **N5.** Angle F's formula block writes "#gears ≤ X". It must be read as the number of primes in [7, X]; see F7.
- **N6.** Derived-machine items (4), (7), (10) and (11) stood "as stated", but their numbered texts are not in the input. A14 gives the formula-block content and the assembly checks on it.
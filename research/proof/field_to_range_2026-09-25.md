# Record: field rules carried to the range object (machine q, one period), 2026-09-25

This is a record of results only. It does not assess prospects. Nothing under C:/dev/primes was modified.

**Notation used throughout:**
- Copy j = (30j−1, 30j+1), and M = M_q = q#/30.
- S_q = {j : gcd(900j²−1, M) = 1}.
- U_q = the primes in (q, isqrt(q#)], and g_max = max U_q.
- Gear g acts on copy j iff g² ≤ 30j+1.
- a_g = 30⁻¹ mod g; t_g = 15⁻¹ = 2a_g mod g; σ_g = m_g = min(t_g, g−t_g).
- P(j) = lpf(900j²−1).
- T = the lower legs of revealed copies, i.e. p ≡ 29 (mod 30) with p and p+2 prime.
- U = the barrier's untruncated machine (same gears and classes, acting dropped).
- "T-rule" = a composite has a prime factor at or below its square root. The range map (3.8) lists these as its consequences: acting ⇔ cofactor ≥ gear; revealed ⇔ both legs prime; below-square strikes are redundant.

## 0. Verification run for this record

**Scripts.** All ran in the foreground from raw divisibility, via `uv run --directory C:/dev/primes`, each in under 6 s. They are in C:/Users/Alex/AppData/Local/Temp/claude/C--dev-primes/a1c3a0ad-3acd-4fbe-b091-1e36256528ec/scratchpad/record_0925/:
- verify_lean.py
- verify_a.py
- verify_b.py
- verify_c.py
- verify_d.py

Every check below had 0 failures unless a result says otherwise.

**Lean sources**
- All 85 named theorems and defs were found in the cited files. exists_unstruck also appears in ArcFloor.lean, and chain_law also in TopMachine.lean.
- There are 0 `sorry` tokens in code in the 16 Ladder*/Mirror/Periodic/TwoTeeth/MergeLaw/AnchorChain files and the 7 Range*.lean files.
- The Lean was not rebuilt.

**Whole range [1, M−1]**
- Survivors: 4, 44, 494, 7424, 126224 at q = 7..19, and 2650724 at q = 23. Each equals prod(g−2) − 1.
- Revealed copies: 4, 19, 152, 1517, 19017 at q = 7..19, and 298408 at q = 23.
- L_q = 0, 1, 3, 10, 26, 93 at q = 7..23.
- The survivor palindrome j ↔ M−j holds at q = 7..23.

**Acting and revealed status (q = 7..19)**
- Revealed ⇔ P(j)² > 30j+1 holds for every j in [1, M). TB(q) is false at every q.
- The acting and acting-free patterns of all gears 7..isqrt(q#) differ exactly on the twin copies with 30j−1 ≤ isqrt(q#): 0, 1, 3, 10, 26 copies. No copy is acting-struck but acting-free unstruck.
- Revealed under acting = both legs prime. The acting-dropped spared set = W_q.
- The acting-deleted survivors are exactly the survivors with a composite leg.

**Consecutive survivors (q = 7..19)**
- E1 analogue: an upper gear that strikes consecutive survivors at gap d divides d(225d²−1).
- Run records over the whole range, identical with and without acting:

| Gears | q = 7 | 11 | 13 | 17 | 19 |
|---|---|---|---|---|---|
| One upper gear | 0 | 2 | 2 | 2 | 2 |
| Two upper gears | 0 | 3 | 4 | 4 | 4 |
| Three upper gears | 0 | 4 | 5 | 5 | not run |

Two-gear witnesses: survivors 28..33 at q = 11, 401..407 at q = 13, and 1688..1694 at q = 17 and 19.

**Mirror and V1 (q = 7..19)**
- Central survivor pairs: (2,5), (36,41), (498,503), (8506,8511), (161656,161667). Each central gap equals d* = 3, 5, 5, 5, 11.
- The three-term shared-striker law holds for every s in [1, M/2).
- V1: every class has a lap k with q' | 900j²−1 and q'² ≤ 30j+1. The maximum k is 6, 6, 12, 16, 16.

**Γ_2 law and lift law (q = 11..19)**
- Γ_2 law, parts (a) and (b), holds on every consecutive survivor pair. Doubled incidences: 2, 14, 214, 3282.
- Lift law, tested on 200 classes × all k mod g for three U gears: 0 failures in 3388, 41800, 150200 and 632200 tests.

**Acting-dropped overlap, first differences**
- 29 (7→11, 7→13, 7→19)
- 59 (11→13, 11→19)
- 179 (13→17, 13→19)
- 809 (17→19)

**Tooth and pair laws**
- Tooth and κ law: 2259 primes 7..20000.
- 15σ_g = g−1 holds at 275 gears, all ≡ 1 (mod 30). The first are 31, 61, 151, 181, 211, 241.
- Pair law: primes 7..400.
- Copy E2 formula R(g,h): 351 pairs of gears 7..113. Values: 2 in 276 pairs, 3 in 72, 4 in 3.
- Alternating covers mod 899: 29-first at 463, 31-first at 433.
- F_g law: 78495 primes 7..10⁶.

**Covers and the q = 107 instance**
- C_7^free: the best cover over all 143 phases is 3 of 4.
- q = 11 phase c: the uncovered set is {2, 27}, and −kM ≡ c_g (mod g) for k = 22370436767022.
- (1197, 1213; 241):
  - 1197 and 1213 are consecutive survivors at q = 107, 109, 113, 127, 131, 137 and 139.
  - At q = 103 the next survivor after 1197 is 1202. At q = 149, 1197 is struck.
  - The legs are 35909 = 149·241, 35911 (prime), 36389 (prime) and 36391 = 151·241, with 241² = 58081.

**Nodes at 29#**
- 6469692419, 6469692809 and 6469693079 are twin lower legs.
- There is no twin copy in (6469693079, 6469694039) and none in (6469694039, 6469694969). Both endpoints are twin.
- 6469694041 − 29# = 811.

**Antipodes and c***
- The copy antipode (M+1)/2 and its partner (M−1)/2 have exactly one machine striker, 7, at every q = 7..113. The lower leg of (M+1)/2 is q#/2 + 14.
- c* = 10⁻¹ mod M:
  - revealed: 5 (q = 7) and 54 (q = 11);
  - deleted: 901 (27029 = 151·179), 11912 (357361 = 191·1871), 96997 (2909911 = 47·101·613) and 743643 (22309289 = 41·544129).

**Other instances**
- Copy M−1 is deleted by an acting upper gear at q = 11..23. Its legs are 2279 = 43·53; 29999 = 131·229 and 30001 = 19·1579; 510479 = 631·809; 9699659 = 29·59·5669; 223092839 = 13619·16381 and 223092841 = 2819·79139.
- There are 20 {59,61}-smooth numbers in (1, 29#]. The only pair at distance 2 is (59, 61).
- (30·1688−1) mod 89 = 87 and (30·1694+1) mod 89 = 2.

**Environment note.** Process PID 3952 has been running since 10:14 today (scratchpad/tower_limit/b2_twoadic.py) and holds 31 GB of private memory. Free virtual memory was about 1.1 GB during these runs. I did not touch it.

## 1. Inventory

**Field-to-range dictionary**
- Copy j is column 5j.
- Gear g ≥ 7 strikes copy j iff j ≡ ±a_g (mod g).
- A column distance m = 5D becomes a copy distance D. The field condition 3m ≡ ±1 (mod p) becomes 15D ≡ ±1 (mod g), which is RangeCopies.copy_pair_iff.
- One gear's two teeth sit t_g = 15⁻¹ apart.
- Gear 5 is in the lower set and never strikes a copy.

**Status caveats, as the sources give them**
- E2 exact values: an elementary argument plus exhaustion over gears to 101. The kernel holds only five_consecutive and four_consecutive.
- Three-gear records: exhaustion over the first eight gears.
- E3′ exact recursion: the draft table marks it PROVED, but its text says "verified 7..23".
- E5: elementary, with no kernel entry.
- Residue collapse: exact by CRT; the census was verified at 11→13 and 13→17.
- AlmostAll and ShortInterval contain twin counts (flagged).
- Excluded as single-machine certificates: mirror_exposed11/29, antipode_exposed11/29, periods_odd, LadderCertificate and LadderPratt.

### G1. Widening and persistence (the square-root rules)

Definitions: StrikesBy p n := p | 6n−1 ∨ p | 6n+1. TwinCentre s := 6 | s ∧ (s−1) and (s+1) are prime. Struck q n := ∃ p prime with 5 ≤ p ≤ q striking n. Gear q p := p is prime ∧ 5 ≤ p ≤ q.

- **widening_member** (LadderWidening): if Coprime m 6, 5 ≤ m, m < g² and g | m, then m = g or some prime h with 5 ≤ h < g divides m.
- **widening, Lemma B** (LadderWidening): if g is prime, g ≥ 5, n ≥ 1, 6n+1 < g² and StrikesBy g n, then (6n−1 = g ∨ 6n+1 = g) or some prime h with 5 ≤ h < g has StrikesBy h n.
- **twin_column_strikers** (LadderWidening): if TwinCentre(6n), p is prime and StrikesBy p n, then p = 6n−1 or p = 6n+1.
- **twin_slot_persists** (LadderWidening): if TwinCentre(6n), q < 6n−1 and q′ < 6n−1, then ¬Struck q′ n.
- **prime_of_unstruck_member, Lemma A** (LadderCovering): if q is prime, q ≥ 5, Coprime m 6, q < m ≤ q², and no Gear q p divides m, then m is prime.
- **leaf_is_sieve_data** (LadderDepth): if TwinCentre s, m < (s+1)², m is not prime and m ≥ 2, then some prime p ≤ s−1 divides m.
- **E5 origin square lemma** (field_proof_draft §5e; elementary; no kernel entry; verified for 75 consecutive pairs to q′ = 400).
  - Let q < q′ be consecutive primes. A member m ≤ q′² of a column of the window of q′ with no prime factor ≤ q is prime or equals q′².
  - Hence the columns of that window left unstruck by machine q are exactly its twin columns, plus the column (q′²−1)/6 when q′²−2 is prime.
  - So q′ fills at most one hole of machine q in its window, and that hole was never a twin slot.
- **Window conservation** (§5e): T(q′) = T(q) − [q′+2 prime] + N(q², q′²]. Structurally, the only slot a machine loses is (q′, q′+2), and every slot gained is a new twin at the top.
- **C2** (§3): row g marks the column of g² once, at n = (g²−1)/6, and only when g² ≤ q².
- **C3** (§3): a member with two prime factors g ≤ r is marked at column (gr ∓ 1)/6. A member with j ≥ 3 prime factors adds no new paint.
- **C4** (§3): higher:g and lower:g each union to the whole painted set.
- **Self-similarity** (§5a): higher:g = g × (the numbers in [g, q²/g] coprime to 6 and to every gear below g).

### G2. Covered runs, the record sandwich and alignment

Lean statements:
- **struck_periodic** (LadderWidening): if n ≥ 1 and every Gear q p divides P, then Struck q (n + kP) ↔ Struck q n.
- **tooth_of_class** (LadderWidening): if q′ is prime, q′ ≥ 5 and n ≡ invSix q′ (mod q′), then StrikesBy q′ n.
- **align_single_hole** (LadderWidening): suppose q ≤ q′, P is a period of machine q and coprime to q′, and a window [a, a+L) has exactly one hole h₀ of machine q. Then some a′ ≥ 1 has [a′, a′+L) fully struck by q′.
- **not_maxGapBelow_of_single_hole** (LadderWidening): under the same hypotheses, ¬MaxGapBelow q′ L.
- **holes_in_covered_run** (LadderFields): if q′ is the next prime after q and [a, a+L) is fully Struck q′, then #holes of machine q in it ≤ 2⌈L/q′⌉.
- **thin_band, E4e** (LadderFields): #holes of machine B in a run fully struck by machine q ≤ Σ_{B<p≤q prime} 2⌈L/p⌉.
- **copy_phase** (AnchorChain): OnTeeth u (x + jP) ↔ DeletedAt (2u) (−u − jP) x.
- **phase_bijective** (AnchorChain): if P is a unit, j ↦ −u − jP is bijective on ZMod g.
- **hop_iter** (AnchorChain): the nested hop equals (nextM)^[k+1].
- **newgap_le, newgap_le_step** (MergeLaw): merged-window sum ≤ B, under SpectrumBound, QualBound and teeth hypotheses with 4u ≤ q.
- **exists_unstruck** (LadderCovering): FreeUncoverable q a L ⇒ some n in [a, a+L) is unstruck.

Draft statements:
- **E3** (§5): a record run of q′ is old runs joined at q′-only columns, so F(q′) ≤ (h+1)F(q) + h, with h ≤ 2⌈F(q′)/q′⌉.
- **E3′** (§7): G_1(q) ≤ F(q′) ≤ G_k(q), with k = 2⌈F(q′)/q′⌉ (kernel halves as above). The exact recursion is over maximal chains of holes in q′'s two tooth classes.
- **E4c** (§7): the tooth distance 3⁻¹ mod q′ gives alternating hole gaps. Interior gaps are ≥ (q′−1)/3, and holes ≤ 3L/(q′−1) + 1.
- **E4d** (§7): F(q′) ≤ S_{t*}(q) with t* = ⌈(q′−1)/3⌉.
- **Residue collapse** (§5b): translates of W painted by q′ number q′, 2, 1 or 0 by the hole residues. So F(q′) = max(G_1(q), the longest alternating window).
- **E4f, twisted translates** (§5d): F(q′) ≤ q′(F_T(q; q′) + 1).

### G3. Two-gear freedom on consecutive columns

Lean statements (LadderFields unless marked):
- **no_adjacent, E1**: p prime ≥ 5 cannot strike both n and n+1 (n ≥ 1).
- **strike_two**: p strikes n and n+2 ⇒ p ∈ {5, 7}.
- **alternating_case**: g on n and n+2 together with h on n+1 and n+3 is impossible unless {g,h} = {5,7}.
- **five_consecutive, E2a**: any two gears leave one of n..n+4 unstruck.
- **four_consecutive, E2b**: the same for n..n+3 unless {g,h} = {5,7}.
- **neighbour_of_hit** (AnchorChain): if 6u = 1 and x is on the teeth, then x+1 is not.
- **class_count**: #{n ∈ [a, a+L) : n ≡ r} ≤ ⌈L/p⌉.
- **gear_count**: #{n ∈ [a, a+L) struck by p} ≤ 2⌈L/p⌉.

Draft statements:
- **E2 exact** (§5): the record is 4 for {5,7}, 3 if exactly one of g, h is 5 or 7, and 2 otherwise.
- **Three-gear records** (§5a, §7): 6, 5, 4 or 3 by the tooth-distance multiset; F({5,7,g}) = 6.

### G4. One gear along a chain: spacing, alternation, residue necessity

- **C1 and C5** (§3; kernel struck_classes and invSix_spec in LadderCovering): row g marks exactly 2 columns per g (classes ±6⁻¹, at distance 3⁻¹), never adjacent.
- **strike_distance** (LadderFields): p | m−n, or 3(m−n) ≡ ±1 (mod p).
- **strike_distance_ge** (LadderFields): if p ∤ m−n, then 3(m−n)+1 ≥ p.
- **TwoTeeth**, with Kill q u x := x ≡ ±u (mod q) and 4u < q:
  - kill_spacing: consecutive kills are 2u or q−2u apart;
  - next_kill_of_lo / next_kill_of_hi: the next kill is determined by the class;
  - kill_period: three consecutive kills span exactly q;
  - spacing_from_lo / spacing_from_hi (T2, T3): the stay or move classes;
  - kills_gap_ge: 2u ≤ y−x;
  - fuel_span_cap and fuel_le (T4, T5): k ≤ 1 + (x_{k−1}−x_0)/(2u).
- **AnchorChain**:
  - teeth_eq_phase: OnTeeth u x ↔ DeletedAt (2u) (−u) x;
  - chain_law: a common phase exists ↔ y−x ∈ {0, ±d};
  - no_two_up / no_two_down: two equal steps force 2d = 0.
- **MergeLaw**:
  - interior_gap_mod: interior gaps ≡ 0 or ±2u (mod q), with 2u < q;
  - floor_of_mod: ≥ 2u, with 4u ≤ q.

### G5. The dichotomy and the chain/ladder forms

Lean statements:
- **LadderDichotomy**: Good, NoConsecutiveLeaves, NoSingleRung, good_step, chainHyp_of_dichotomy, depthHyp_of_dichotomy, twins_unbounded_of_dichotomy, good_six. Here Rung s s′ := TwinCentre s′ ∧ (s−1)² < s′−1 ∧ s′+1 < (s+1)².
- **LadderDepth**: Chain, ChainHyp, Chain.ge, twins_unbounded_of_chains, chainHyp_of_depthHyp, twins_unbounded_of_chainsPow, parent_unique, and Depth / Depth.ge (Depth.ge: 6 + 2n ≤ s).
- **LadderInfinite**: Depth.bound (s+1 ≤ 8^(2^k)) and infinite_iff_depthHyp.
- **LadderAlmostAll**: rungs_disjoint, and chainHyp_of_almostAll (a count hypothesis, flagged).
- **LadderShortInterval**: rungPow_of_shortInterval.
- **LadderProduct**: ProductRung, productRung_gt, productHyp_of_sixHyp, productHyp_of_ladderHyp, twins_unbounded_of_product.
- **LadderRegion**: RegionHyp, consecutive_of_twinCentre, ladderHyp_of_regionHyp, WindowHyp, windowHyp_of_ladderHyp.

### G6. Period, mirror, universal clearance and no blocking cycle

- **D1** (§4, CRT): each joint period of a finite gear set contains exactly prod(g−2) ≥ 1 unstruck columns. "Where D1 stops": those columns may all lie outside the window.
- **Periodic**:
  - next_shift: next(k+P) = next k + P;
  - op_shift;
  - index_reduce: any gap word reduces to one period;
  - gap_mod.
- **Mirror**:
  - mirror_gear: q | P ⇒ q | lo(P−k) ↔ q | hi(k), and conversely;
  - antipode_open: 2s = P+1 ⇒ q ∤ lo s and q ∤ hi s;
  - self_mirror_unique (N odd);
  - even_card_involution;
  - window_count_even;
  - none_of_at_most_one.
- **LadderEuclid**:
  - lower_member_rough: if TwinCentre s, p | j and p ∉ {s±1}, then p ∤ s² + 6j − 1;
  - lower_member_rough_upto;
  - upper_member_iff: if p | j, then p | s² + 6j + 1 ⇔ p | s² + 1.

## 2. Per group: standing range forms, refuted forms, and upper-set joint pattern

### G1

**Standing**

1. **TB formula (PROVED; 0 mismatches at q = 7..23; TB(q) false at q = 7..23).**
   - TB(q) ⇔ every j ∈ [1, M_q) has P(j) ≤ q or P(j)² ≤ 30j+1.
   - Set form 1: S_q ∩ [1, M) ⊆ ⋃_{g∈U_q} ({j ≥ ⌈(g²+1)/30⌉ : 30j ≡ 1 (mod g)} ∪ {j ≥ ⌈(g²−1)/30⌉ : 30j ≡ −1 (mod g)}).
   - Set form 2: S_q ∩ [1, M) = ⊔_{g∈U_q} {j : P(j) = g, g² ≤ 30j+1}.
   - Negation: some j ∈ [1, M) has P(j) > max(q, √(30j+1)). This holds exactly at the revealed range copies, so the negation is the range statement at q.
2. **Row 3 (PROVED; q = 7..23).**
   - A survivor j ∈ [1, M) is revealed ⇔ Str_{U_q}(j) ⊆ {30j±1}.
   - For every j ∈ [1, M), with strikers taken over all primes 7..isqrt(q#): j is revealed ⇔ every striker is a leg.
3. **Row 1, shadowing (W1) (PROVED).**
   - Take a survivor j, a leg L and g ∈ U_q with g | L and L < g². Then L = g, or h = lpf(L) satisfies q < h < g and h² ≤ 30j+1.
   - Check tallies at q = 7..23: home strikes 0, 2, 6, 21, 65, 266; shadowed strikes 0, 7, 92, 1173, 16238, 288945.
4. **Row 2 (PROVED).** For every copy j ≥ 1, leg L and prime h | L: h² ≤ 30j+1 ⇔ h² ≤ L. At most one prime factor of L fails to act.
5. **Row 4, persistence (W2), lower-gear/acting form (PROVED; 15 pairs in {7..23}).** A revealed copy (p, p+2) with p > q is a revealed survivor of every q″ with q ≤ q″ < p and p+2 ≤ q″#. These machines form an interval.
6. **Row 6.** Survivors with 30j+1 ≤ q² are revealed (q = 7..23).
7. **Row 7, region law (W3) (PROVED).**
   - For every copy j ≥ 1: revealed ⇔ P(j)² > 30j+1.
   - In the stratum 30j+1 ≤ q′², the only non-revealed survivor is j = (q′²−1)/30, when q′² ≡ 1 (mod 30) and q′²−2 is prime. Instances: j = 12 at q = 17 (361 = 19²) and j = 28 at q = 23 (841 = 29²).
8. **Row 9, conservation on the range (PROVED).**
   - Inside the old range, q′ strikes survivors of q only by acting, except at its home copy.
   - The set identity was checked at 7→11 through 23→29. At 23→29, R(29) ∩ [1, M_23) = R(23) \ {1}.
9. **Row 11 (PROVED).** Non-revealed survivors = ⊔_{g∈U_q} F_g, with F_g = {j : P(j) = g, j not a home copy of g}. g acts on every member of F_g.
10. **Row 10.**
    - Per-leg acting thresholds: ⌈(g²+1)/30⌉ on the lower leg and ⌈(g²−1)/30⌉ on the upper leg.
    - A two-factor leg g·r with g ≤ r is acted on only by g.
11. **Row 12 (iii), strict form (PROVED; q = 7..23).**
    - For j > H_q = ⌊(⌊√q#⌋+1)/30⌋: revealed ⇔ no U_q strike.
    - At j = H_q the equivalence fails at q = 11 (j = 1) and q = 13 (j = 5).
12. **Row 13, V1 (PROVED; exists for every q ≥ 7).**
    - The model: legs 30j±1, gears U_q, cutoff H_q.
    - Representative heights: h(r) = min{j ≡ r (mod M) : q′ | 900j²−1, q′² ≤ 30j+1} ≤ r + q′M.
    - The blame b ≡ q′ is total and satisfies F1–F3.
    - Maximum laps: 6, 6, 12, 16, 16, 23 at q = 7..23.
13. **Row 14, repaired V2 (PROVED).**
    - B = 30 + q#·t, with t ≡ t₀ (mod Π), B·j_i ≡ 1 (mod p_i), B/30 prime and B/30 > q#.
    - The upper set is the primes in (q, √(B·M)].
    - F1–F3 hold with b(j_i) = p_i. Existence for every q ≥ 7 follows by Dirichlet.
14. **Row 15, repaired (PROVED).**
    - TB(q) ⇔ no j ∈ [1, M) with 30j−1 > q has both legs prime.
    - For B ≡ 30 (mod q#) with B > 30, every U_q gear acts on every copy j ≥ 1.
15. **Row 16 (MEASURED, flagged).** Every leg shape (30j+a, 30j+b) with d ≤ 60 has a revealed survivor at q = 7, 11, 13.

**Refuted**
- **Row 3 literal header** (all copies, U_q gears only):
  - q = 11, j = 17: legs 509 and 511 = 7·73; Str_U = ∅, yet j is not revealed.
  - q = 13, j = 3: legs 89 and 91 = 7·13; Str_U = {89}.
- **Row 5 read with U_q″ gears** ("a twin slot stays unstruck"): at q″ = 11, copy 1 (29, 31) is struck by 29 and 31, both in U_11.
- **Row 12, "all three fail in U"**: false for (ii).
- **Row 14 literal ("F3 holds for the built B")**:
  - q = 7: B = 7653900 = 2²·3·5²·31·823, and gears 31 and 823 strike 0 subclasses;
  - q = 11: 3797 | B;
  - q = 13: B/30 is composite and coprime to 30030;
  - q = 17: 16711463 | B.
- **V2 read over U_q and H_q**: survivors above H_q that no U_q gear strikes and that are not revealed number 2, 16, 151 and 1591 at q = 7, 11, 13, 17.
- **Row 15 literal right side**: fails for every q ≥ 29, because copy 1 lies in [1, M) and is outside the range.

**Joint pattern (upper set on survivors)**
- **Joint redundancy (PROVED; holds in U).** For the gears g_1 < g_2 < … of U_q and every k:

  S_q ∩ ⋃_{i≤k} Str(g_i) = S_q ∩ (⋃_{i≤k} ActStr(g_i) ∪ Home(g_1..g_k)).

  Checked literally at q ≤ 19. In U it was checked at 43, 339, 3242 and 42324 segments (q = 7..17), and pointwise at q = 7..23.
- At most one non-acting striker per leg.

### G2

**Standing**

1. **Formula (1) and acting localisation (3) (PROVED; q = 7..23).**
   - Range(q) ⇔ 0 ∉ C_q^act.
   - Range(q) ⇔ [some s ∈ S_q with 30s−1 ≤ g_max has both legs prime] or [0 ∉ C_q^free].
   - Both sides of (1) and (3) are equivalent to "some range survivor has both legs prime".
2. **Formula (2) (PROVED).**
   - For k ≥ 1, the translate kM + [1, M−1] is fully deleted by acting gears of L_q + U_q ⇔ (−kM mod g)_g ∈ C_q^free.
   - The map k ↦ (−kM mod g)_g is bijective mod prod(U_q).
   - Checks, 0 mismatches: k = 1..429 at q = 7; 3000 random k at q = 11; 150 random k < 10³⁰ at q = 13.
3. **Item (19), joint form (PROVED; q = 7..23).**
   - On the copies j ≥ 1, the acting deletion pattern of L_q + U_q differs from the acting-free pattern exactly on the twin copies with a leg in L_q + U_q. All of these lie at j ≤ (g_max+1)/30.
   - Per gear, g strikes without acting exactly on its tooth copies with j < ⌈(g²−1)/30⌉.
4. **Free covers.**
   - C_7^free = ∅, by exhaustion of 143 phases (best cover 3 of 4).
   - C_11^free = ∅, by exhaustive branch-and-bound and by HiGHS.
     - The largest cover is 42 of 44, at c = {13:6, 17:9, 19:7, 23:10, 29:6, 31:9, 37:33, 41:32, 43:11, 47:39}, which leaves {2, 27} uncovered.
     - That phase is realised by the translate k = 22370436767022.
   - Emptiness of C^free is sufficient for Range(q), not necessary.
5. **Persistence and periodicity.**
   - Acting deletion persists upward by multiples of P.
   - g_max² ≤ 30M at every q.
6. **Definitional and CRT items.**
   - (4) is definitional.
   - (11): the g / 2 / 1 / 0 translate classification.
   - (13) is CRT.
   - (12): the class distance is (15g)⁻¹ mod p (8184 tests at q = 13).
7. **Covered-run items.**
   - (6) holes ≤ 2⌈L/g_1⌉ on a covered run: PROVED; counting, flagged.
   - (7) R(q; g_1) = max(R0, the spans of maximal chains of consecutive survivors in g_1's acting teeth): PROVED.
   - (8), upper half, R ≤ G_{2⌈R/g_1⌉}: PROVED by the tooth count (counting, flagged). R ≤ G_2 is measured at q = 7..23.
   - (10) S_σ ≥ R: PROVED.
   - Measured at q = 7..23:

| q | 7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|
| R0 | 2 | 4 | 5 | 7 | 12 | 18 |
| R | 2 | 5 | 6 | 12 | 18 | 22 |
| Longest run deleted by the whole acting upper set | 2 | 10 | 37 | 126 | 144 | 320 |
| G1 | 3 | 5 | 7 | 10 | 18 | 22 |
| G2 | 4 | 6 | 10 | 13 | 19 | 28 |

8. **E4c origin form (PROVED where D < g).**
   - For consecutive survivors s < s′ struck by one gear g on opposite legs: g | s+s′, D ≡ ±t_g (mod g), D ≥ σ_g and g ≤ 30D+2.
   - A same-leg pair forces g | D.
   - There were 0 same-leg pairs at q = 11..23.
9. **Alternating tooth-class instances (MEASURED).**
   - q = 17, survivors 1735, 1736, 1737, 1741: gear 31 strikes 1735 (lower leg) and 1737 (upper); gear 19 strikes 1736 (lower) and 1741 (upper).
   - q = 23, survivors 27322, 27326, 27327, 27330: gears 37 and 61.
   - All of these strikes act.
10. **Item (17) (PROVED; primes to 10⁶).**
    - F_g = min(a_g, g−a_g) = (m·g ± 1)/30, with m = min(n, 30−n) and n = −g⁻¹ mod 30.
    - 4F_g ≤ g ⇔ m ∈ {1, 7}.
11. **Item (16).** hop_iter applies on the range. newgap_le's hteeth survives restricting kap to acting strikes.
12. **Aligning instance at q = 7.**
    - Run [3, 5] has the single survivor 5 = (149, 151).
    - The aligning translates are copies 26, 40, 68 and 75.

**Refuted**
- **(14) Phase to height.**
  - q = 11, H = {13, 17, 19, 23}, L = 8: a direct scan finds 144 realisers in more than one class.
  - Acting instance: K = [[], [29]] is realised at the origin only at x″ = 27, since copy 27 = (809, 811) and 841 = 29². The CRT class has no member in [27, 75].
- **(8) lower half, R ≥ G_1**: fails at q = 7 (2 < 3) and q = 13 (6 < 7).
- **(5) range half**: at q = 7, run [3, 5] holds the single survivor 5, whose legs are both prime, so nothing deletes it at k = 0.
- **P-periodicity of revealed status**: at P = 6685349671, 29 and 31 act on copy 1+P but not on copy 1.

**Joint pattern.** Formulas (1)–(3), item (19), and the C^free results above.

### G3

**Standing**

1. **Tooth law (PROVED; 2259 gears to 20000).**
   - σ_g = (κ_g·g ± 1)/15, with κ = 1, 2, 4, 7 for g ≡ ±1, ±7, ±11, ±13 (mod 30).
   - min 15σ_g/(g−1) is exactly 1, attained exactly at g ≡ 1 (mod 30).
2. **Pair law and cofactor law (PROVED).**
   - For 0 < D < g: g strikes j and j+D ⇔ D ∈ {σ_g, g−σ_g}.
   - The class of j is then unique mod g, and the struck legs are opposite: g·h and g·(h + 2κ_g), or g·(h + 2(15−κ_g)).
   - Both strikes act ⇔ h ≥ g.
   - The gears at copy distance D ≤ 4 are {7, 11, 23, 29, 31, 59, 61}.
3. **Comb count (PROVED).** c_g(W) = 2⌊W/g⌋ + min(W mod g, 1) + [W mod g > σ_g].
4. **Copy E2 record (PROVED).**
   - R(g,h) = 2 + [{g,h} ∩ {7, 29, 31} ≠ ∅] + [{g,h} ∈ {{7,11}, {7,23}, {29,31}}].
   - Checked on 946 pairs of gears 7..211, and rerun on 351 pairs here.
   - Triples: 560 triples of gears 7..67 give record 3 in 165 triples, 4 in 218, 5 in 155 and 6 in 22.
5. **Alternating classes (PROVED).**
   - 29 on j, j+2 with 31 on j+1, j+3 gives class 463 (mod 899). 31 on j, j+2 with 29 on j+1, j+3 gives class 433.
   - Among gear pairs 7..211, only (29, 31) gives an alternating cover.
6. **Four-runs (PROVED).**
   - Case 1, span S < min(g,h): the pairing is 12|34, 13|24 or 14|23, and both gears lie in E_q = {g ∈ (q, √q#] : σ_g ≤ Γ_q(4)}.
   - Case 2: any other two-gear four-run has min(g,h) ≤ S ≤ Γ_q(4).
7. **Repaired (13) (PROVED).** Let Σ be a stretch of consecutive survivors with span S and lowest height j ≥ j*(S) = ((15S+1)²−1)/30.
   - (a) If Σ holds no survivor with both legs prime and a leg ≤ √q#, the least number of upper gears covering Σ is the same with and without acting.
   - (b) The two differ exactly when Σ holds such a survivor and no twin survivor with both legs > √q#.
8. **Revealed with and without acting (PROVED).**
   - With acting: revealed ⇔ both legs prime.
   - Without acting: revealed ⇔ both legs prime and both > √q#.
   - The differences are 1, 3, 10, 26, 93, 332 at q = 11..29.
9. **Partial status.**
   - The survivor form of five_consecutive is proved only where Γ_q(5) < q′ (q = 11, 13).
   - The one-gear bound "no 3 consecutive survivors" is proved only where Γ_q(3) < q′ (q = 11..23).
10. **Capacity (PROVED; counting, flagged).** L ≤ Σ c_{g_i}(S+1) ≤ 2k⌈(S+1)/g_min⌉.
11. **Measured, identical with and without acting.**
    - Two gears: 3, 4, 4, 4, 4, 4 at q = 11..29.
    - One gear: 2 at q = 11..29.
    - Three gears: 4, 5, 5, 6 at q = 11..19 (exhaustive); 6 at q = 23 (acting only).
    - Γ_q(3,4,5) at q = 11..29: (6,7,10), (8,11,13), (11,14,19), (19,20,28), (23,29,36), (32,36,41).
    - No two-gear 5-run at q = 11..29.

**Refuted**
- **Original (13)**: its acting content is not "cofactor ≥ gear, failing only at cofactor 1".
  - q = 17, copy 33: 989 = 23·43, and 43 does not act.
  - Copy 50: 1501 = 19·79.
  - q = 19, copy 57: 1711 = 29·59.
  - q = 17, copy 12 has cofactor 1 and is not revealed.
- **(4) "any pair of E_q can do it"**: at q = 29, windows with both 211 and 59 occur 0 times, and with 211 and 61 once. Of 1116 eligible ordered combinations, 277 are realised (a count testing the prediction, flagged).
- **Count labels in (2), (4), (6), (7).**
- **The 89 check line**: (30·1688−1) mod 89 = 87 and (30·1694+1) mod 89 = 2. The witness itself stands: 89 strikes 1688 on 50641 = 89·569.

**Joint pattern.** Items 6, 7 and 11 above.

### G4

**Standing**

1. **Centre law (PROVED).** For a prime g ≥ 7 and copies x < y, g strikes both ⇔ [g | y−x and g | 900x²−1] or [g | x+y and g | 225(y−x)²−1].
2. **Tooth sets (PROVED).**
   - {j : g | 30j±1} = {a_g, g−a_g}.
   - On columns 5j+e (e = 0, 2, 3) the teeth are ±a_g − e·5⁻¹.
   - Teeth 1 apart occur only at g = 7, and 2 apart only at g = 29, 31.
   - Survivors: 71569574 at q = 29 and 2075517674 at q = 31.
3. **Spacing, corrected (PROVED).**
   - Consecutive g-struck survivors differ by ≡ 0 (same leg) or ±t_g (a switch, +t_g iff upper→lower), and switch signs alternate.
   - Across two switches the span is ≡ 0 (mod g).
   - Any three g-struck copies span ≥ g. On the copy line, three consecutive teeth span exactly g.
4. **Chain laws on the range.** teeth_eq_phase and chain_law apply verbatim with u = a_g and d = 2a_g.
5. **Interior gaps (PROVED by the direct residue argument).**
   - Interior gaps are ≡ 0 or ±t_g, hence ≥ m_g.
   - When all are < g they alternate t_g and g−t_g.
6. **Γ_k formula (PROVED).**
   - D_g(i) = max{k : g | Γ_k(i)}, with Γ_k(i) = gcd(30s_i+1, Δ_k⁺)·gcd(30s_i−1, Δ_k⁻) and Δ_k^± = gcd over m < k of P_m(15P_m ∓ 1).
   - Γ_2 incidences: 0, 2, 14, 214, 3282, 49273 at q = 7..23; 954764 at q = 29; 17261126 at q = 31.
   - Γ_3 = 0 at q = 7..31. Realised depth is 0 at q = 7 and 2 at q = 11..31.
7. **Bounds (PROVED).**
   - D_g ≥ 2 ⇒ g ≤ 15F_1 + 1.
   - D_g ≥ 3 ⇒ g ≤ min(F_2, (15F_1+1)/8).
   - m_g ≤ (7g+1)/15.
8. **(16′) (a)–(d) PROVED; (e) MEASURED.**
   - (a) T(s_i) ∩ T(s_{i+1}) = the upper primes of Γ_2(i).
   - (b) A(s_i) ∩ A(s_{i+1}) = {g upper prime of Γ_2(i) : g² ≤ 30s_i+1}.
   - (c) Every prime of Γ_2 is ≤ 15G_i + 1.
   - (d) A dropped g ∤ G_i has legs g·h and g·(h+δ) with h < g.
   - (e) The unrestricted equality T∩T = A∩A holds at every i for q = 7..31.
9. **Onset (PROVED).**
   - Acting on the first member of a chain gives acting on every later member.
   - Coincidence holds above (15F_1+1)².
   - There are 0 non-acting doubled pairs at q = 7..31 (measured).
10. **Mirror law and "the only common centre is copy 0": PROVED.**
11. **(14) lift law, for the new gear q′ on the range of q′ (PROVED).**
    - The centre sums of a window's q′ lifts run over every residue mod q′.
    - q′ doubles exactly one lift when 225G² ≡ 1 (mod q′), and two lifts when q′ | G.
    - Tower flag, left to the owner: this reads the range of q′ as q′ periods of machine q.
12. **(15) doubled-pair decomposition (PROVED).**
    - s + s′ = g·m, the legs are g(15m ∓ δ/2), and acting ⇔ 15m − δ/2 ≥ g.
    - At δ = 2 the pair is g × copy k, with acting ⇔ 30k−1 ≥ g.
13. **Per-gear minima (MEASURED count).** With raw strikes, every upper gear strikes a survivor at q = 11..23; the minima are 1, 5, 21, 76, 350.

**Refuted**
- **"Every upper gear strikes some survivor"**: at q = 7, gears 11 and 13 strike none of 1, 2, 5, 6.
- **"Consecutive g-struck survivors are spaced t_g or g−t_g"**: 23→36 (q = 11, g = 13); 30→47; 33→56; 28→57.
- **"Three consecutive g-struck survivors span exactly g"**: q = 11, g = 13, triple 16, 23, 36 with span 20.
- **"The census decides realised depth"**: at q = 19, gap 10 gives 58 distinct centre residues mod 149 and 50 mod 151, none of them 0.
- **Unrestricted (16′) equality**: at q = 107..139, gear 241 strikes consecutive survivors 1197 and 1213 but acts on neither. A∩A = ∅ while T∩T = {241}.

**Joint pattern.** Items 1, 6, 8 and 12.

### G5

**Standing**

1. **Node identity (PROVED; elementwise at q = 7..29).** V_q = T ∩ (q, q#−2]. The survivor classes number prod(g−2) − 1.
2. **Transport (PROVED / MEASURED).**
   - V_t = (V_a ∩ (t, ∞)) ⊔ (T ∩ (max(t, a#−2), t#−2]).
   - With acting, overlap agreement holds at all 15 pairs a < t in {7..23}, and at 29→59 (6,155,196 = V_29 \ {59}).
   - With acting dropped it fails at all 15 pairs. First differences: 29, 59, 179, 809, 3119.
3. **Lean, scratch G5RangeAdj.lean / G5Range.lean (0 sorries; propext, Classical.choice, Quot.sound):**
   - Good(s) ⇔ s⁺+2 ≤ s#, and NS ⇔ s⁺⁺+2 ≤ s#;
   - NS ⇒ all good ⇒ RANGE ⇒ twins unbounded;
   - RANGE ⇔ all good ⇔ Reach from 29 is unbounded;
   - NS ⇒ NC;
   - leaf_iff_terminal;
   - Reach is an initial segment;
   - not_parent_unique;
   - no_node_region_17.
4. **Reductions.**
   - RANGE ⇔ V_s ≠ ∅ for every node s ⇔ there are no consecutive nodes a < t with V_a = {t} and V_t = ∅.
   - NS ⇔ |V_s| ≥ 2 for every node s.
   - The converses fail at the order level, by two hand instances.
5. **Row 8, corrected (MEASURED).**
   - Machine 29 settles |V_s| ≥ 2 at every node s < 6,469,692,809, and Good(s) below 6,469,693,079.
   - With the two nodes 6,469,694,039 and 6,469,694,969 above 29#−2, NS holds at all 6,155,198 nodes ≤ 29#−2.
6. **M*_29 (PROVED).** With gears 7..29 plus every prime in (29, 29#) except 59 and 61, and acting dropped, the machine spares exactly copy 2.
7. **Split (PROVED).**
   - V_q = L_q ⊔ W_q, so Leaf(q) ⇔ L_q = W_q = ∅.
   - At 149→179 the transported set meets L_179 (at 239) and W_179 (at isqrt(179#) + 64,708, both legs Pocklington-certified).
8. **Product reading (PROVED).** It follows from RANGE at t. The nextprime route needs t+2 ≤ s#; for s = 29 it first fails at t = 6,469,694,039.
9. **Region (Lean).**
   - The copy form of RegionHyp fails between 17² and 19².
   - A survivor of machine s with upper leg < (s+2)² is revealed. Flagged as a floor-window statement.
10. **Counts table (MEASURED).**

| q | \|V\| | \|L\| | \|W\| |
|---|---|---|---|
| 7 | 4 | 0 | 4 |
| 11 | 19 | 1 | 18 |
| 13 | 152 | 3 | 149 |
| 17 | 1517 | 10 | 1507 |
| 19 | 19017 | 26 | 18991 |
| 23 | 298408 | 93 | 298315 |
| 29 | 6155197 | 332 | 6154865 |

|V_29| is odd.

**Refuted**
- **Row 8 as printed** ("machine 29 only; 59 out of reach"): see the corrected form above.
- **Range form of parent_unique**: 149 is a rung of both 29 and 59.
- **"Must" as a route constraint (row 13)**: the 149→179 instance above.
- **Mirror-closure of V_29**: only 509,954 of 6,155,197 revealed copies have a revealed mirror, and the best unit u maps only 1,647,379.

**Joint pattern.** The only acting-bearing content spans two machines: the overlap identity and the first-leaf theorem.

### G6

**Standing**

1. **Repaired (4)(c) (PROVED; q = 7..23).** Let E(k) = "no acting gear of G = 7..isqrt(q#) strikes k", and P = the product of G.
   - (a) E(k+P) ⇔ E(k) for k ≥ M.
   - (b) E(k+P) ⇒ E(k) for k ≥ 1.
   - (c) E(k) ∧ ¬E(k+P) exactly when both legs are prime and 30k−1 ≤ isqrt(q#).
   - So hper holds at q = 7 and fails at every q ≥ 11, first at copy 1.
   - Revealed has no period: copy j + pP has lower leg p(1+30P).
2. **Three-term mirror striker law (PROVED; q = 7..19 here).**
   - For 1 ≤ s < M/2 and x = 30s−1, the shared strikers are primes(gcd(x, q#−2)) ∪ primes(gcd(x+2, q#+2)) ∪ primes(gcd(x(x+2), q#)). The third set is empty iff s is a survivor.
   - A gear acts on both members iff g² ≤ 30s+1.
   - Height split: T(s)² + T(M−s)² = q# + 2. A gear with g² ≤ (q#+2)/2 acts on at least one member; a gear with g² > (q#+2)/2 acts on at most one.
3. **Range copies (PROVED).**
   - The range copies are exactly j ∈ [⌊(q+1)/30⌋+1, M−1]. The omitted copies are never survivors.
   - Range(q) ⇔ some survivor in [1, M−1] is revealed.
4. **Lift law (PROVED).**
   - g ∈ U strikes s + kM ⇔ k ≡ −s·M⁻¹ ± (q#)⁻¹ (mod g).
   - g acts on s + kM ⇔ k ≥ 1 or g² ≤ 30s+1.
5. **Pointwise discrepancy over the joint period.** It is the same set as 1(c), all in lift 0. The sub-machine form holds as well.
6. **Coincidence law (PROVED).** Clearance meets a tooth ⇔ g | q# ∓ 1, and tooth meets tooth ⇔ g | q# ∓ 2.
7. **Palindrome (PROVED).** Central gap = d* = min{d odd : gcd(225d²−1, M) = 1} = 3, 5, 5, 5, 11, 11 at q = 7..23.
8. **Mirror pairs (PROVED).**
   - Single-gear deletion needs g | q# ∓ 2 with g² ≤ 30s+1.
   - Range(q) ⇔ some mirror pair of survivors is not doubly deleted.
9. **Antipodes (PROVED).**
   - c* has legs ≡ (2, 4) mod every machine gear.
   - The copy antipode has legs q#/2 + 14 and q#/2 + 16, and 7 is its only machine striker.
10. **Involution lemma (PROVED).** Mirror non-invariance of "revealed" is shown by instance at q = 11..23. At q = 7 all four survivors are revealed.

**Refuted**
- **(4)(c) as printed.**
- **The q = 23 shared-striker line**: 11617 is not a single-gear deleter, and 47507 was omitted. The corrected single-gear deleters are 587 (2257 pairs) and 4801 (219 pairs).
- **The two-term header (3)**: q = 7, s = 3, where gear 7 strikes 91 and 119 and both gcds are 1.
- **Body (1) for q ≥ 29.**
- **Mirror pairing with no doubly-deleted pair**:
  - q = 11, pair (13, 64): 391 = 17·23 and 1921 = 17·113, with 17 acting on both;
  - pair (16, 61).

**Unresolved.** Body (12). Across all s in [1, M/2) at q = 7..19, b(s) = b(M−s) occurs 0, 0, 0, 255 and 1982 times.

## 3. Statements that use acting essentially

These are the standing statements that fail in U or are defined through acting.

**G1**
- The TB formula, its two set forms, and the equivalence of its negation with the range statement.
- Row 1: the shadowing gear h acts.
- Row 2: h² ≤ 30j+1 ⇔ h² ≤ L.
- Row 3 (the converse half).
- Row 7 general form: revealed ⇔ P(j)² > 30j+1.
- Row 8: the bottom-stratum exception.
- Row 11: g acts on F_g.
- Row 12 (i).
- Row 12 (iii): j > H_q ⇒ (revealed ⇔ no U_q strike).
- Repaired row 15.
- The F1 condition of V1 and V2.

**G2**
- Formula (1), C_q^act.
- Formula (3), acting localisation.
- Formula (2): acting is complete at k ≥ 1, since g_max² < 30kM.
- Item (19): the discrepancy set is exactly the twin copies with a leg in L_q + U_q.
- (14), acting form (refuted).
- (5), range half (refuted, because an acting gear cannot delete a prime leg).
- Non-periodicity of revealed.

**G3**
- Repaired (13): acting enters below j*(S), and above it only at twin survivors with a leg ≤ √q#.
- Revealed with and without acting differ exactly on the twin survivors with a leg ≤ √q#.

**G4.** Acting enters only as the per-gear onset g² ≤ 30s_i+1, applied on top of tooth-set facts that hold with acting dropped:
- (15d): acting ⇔ 15m − δ/2 ≥ g;
- (16′)(b)–(d);
- the onset rule and the coincidence above (15F_1+1)²;
- (10), where the kill predicate is an acting strike.

**G5**
- V_q = T ∩ (q, q#−2].
- Overlap agreement and transport, which fail at all 15 pairs with acting dropped.
- L_q = V_q \ W_q.
- The Lean chain theorems use acting only through the node identity.

**G6**
- Repaired (4)(c) and the pointwise discrepancy set.
- The sub-machine form.
- The acting clause of the lift law.
- The height split T(s)² + T(M−s)² = q# + 2.
- The single-gear deletion condition.
- Body (11)(A).
- Range(q) ⇔ some mirror pair is not doubly deleted.

**Status recorded in the adjudications**
- Every item above uses acting only through the T-rule or its listed consequences (acting ⇔ cofactor ≥ gear; revealed ⇔ both legs prime), or as a pointwise height inequality.
- G1 records all its items as consequences of T, and so excluded by clause (b).
- No standing statement in any group forbids a total blame map.
- G1 row 12 (ii) and rows 4 and 9 do not use acting essentially.

## 4. Not established (open questions)

1. **The barrier question.** Which acting-based statement forbids a total blame map on the range?
   - G1: with F1 literal and least-residue heights, the three facts of the question amount to the range statement itself.
2. **G1.**
   - Does a variant with legs of size about 30j, real thresholds, cutoff √q# and least-residue heights admit a total blame for some q? None was found for shapes with d ≤ 60 at q ≤ 13.
   - Not pursued: base-30 legs with a different finite lower set.
3. **G2.**
   - Is C_13^free empty?
   - Is C_q^free empty for every q?
   - Is there an origin-phase many-gear form of the covered-run laws? None was produced.
   - What rule gives the positions of the two-gear alternations?
4. **G3.**
   - Can two upper gears strike 5 or more consecutive survivors for some q? This is open from q = 17 on.
   - Is there a k-gear bound in k alone?
   - Does Case 2 of the four-runs need a gear striking 3 of the 4? UNRESOLVED.
5. **G4.**
   - Does the census admit depth 3 at larger q, and is such a window realised?
   - What law governs the realised centre residues (s_i + s_{i+1}) mod g?
   - Is there a survivor form of "no blind gears" for any q?
6. **G5.**
   - NS and NC.
   - Is there an involution preserving the revealed copies? None was found, and parity is closed at node 29.
   - A structural link between L_q and W_q was not attempted.
7. **G6.**
   - Body (12).
   - Is there a non-affine map pairing the survivors so that no pair is doubly deleted? Arbitrary pairings are excluded only by R < D, a count (flagged, q = 7..23).
   - What law decides the status of c*?
   - How are the deleting gears distributed on doubly-deleted mirror pairs?

## 5. Unfinished computations

**G1**
- Row 12 (ii) was checked literally only to q = 19. q = 29 (about 2·10⁸ copies) was not run.
- The shape search was not run for larger q or d.
- No Lean entries exist for rows 2, 3, 9, 11–15 or E5.

**G2**
- q = 13 free cover: HiGHS reached its 230 s limit, and the DFS prefix [1, 300] was not decided within 150,000 nodes.
- q ≥ 17 was not attempted.
- The offset-2 and offset-3 column forms were not rerun.
- No Lean for acting-complete, the E4c origin form, the phase bijection, phase to height, or acting localisation.

**G3**
- The 5-run search at q ≥ 31 (6.7·10⁹ copies) was not run, and the designed g-triple search was not run either.
- Three-gear records at q = 29 were not computed; at q = 23 the search ran with acting only. This record's rerun stopped at q = 17.
- The Jacobsthal-type existence route was not completed. It is of counting type.

**G4**
- Depth for q ≥ 37 was not measured. The bottom-of-range scan covered only copies < 3·10⁵ at q = 37..199.
- The Γ formula was checked against direct striking only at q ≤ 23 (and at q = 11..19 in this record).
- No Lean for the centre law, Γ_k, the depth bounds or the transition identity.

**G5**
- Node 59 (59# ≈ 1.9·10²¹) is out of reach.
- The G5 Lean file is scratch only and not registered in the lakefile.
- Argued in prose only: the Depth.bound tower, the product readings, the node-window claim, the order non-implications, and the models U′_q and M*_29.

**G6**
- The variant-machine test of the acting clauses at large q was not run.
- R ≥ D was evaluated only at q = 7..23.
- Joint identity (2) was enumerated only on six sub-machines.
- No Lean for the pair law, height split, lift-0 concentration, coincidence law, d*, or non-periodicity of revealed.
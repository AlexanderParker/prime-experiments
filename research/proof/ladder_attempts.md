# Proof attempts for LadderHyp / ChainHyp / SixHyp - every step marked

Prover lane, 2026-09-20. Written from BRIEF.md only. Measurements quoted below come from
`measure.py` and `measure2.py` in this folder (sieve to 10^8; each run about 1-5 s).

## Notation (fixed for all three attempts)

- s = 6c a twin centre (s-1, s+1 prime). N = 4c-1. J = {j : |j| <= 2c-1}, J* = J \ {0}.
- Members of column j: L_j = s^2 + 6j - 1, U_j = s^2 + 6j + 1.
- G(s) = primes in [5, s-5]. For p in G: S_p = {j in J* : p | L_j U_j} (p "strikes" j).
- a_p = -(s^2-1)/6 mod p, b_p = -(s^2+1)/6 mod p (the two struck classes, F1).
- r(s) = #{j in J* : no p in G strikes j} = number of rungs. Leaf <=> r(s) = 0.
- For a bound B: R_B(s) = {j in J* : no p in G with p <= B strikes j} (the B-rough offsets);
  omega_B(j) = #{p in G, p > B : j in S_p}; T_B = sum_{j in R_B} omega_B(j);
  X_B = sum_{j in R_B} (omega_B(j) - 1)^+ (the excess strikes).
- Measured anchor window: s = 10008 (10007, 10009 prime), c = 1668, N = 6671, |G| = 1227,
  r(s) = 160.

---

## Attempt A - covering argument for LadderHyp

A1. PROVED. Leaf <=> J* is covered by {S_p : p in G}; j in J* is a rung <=> no p in G strikes j.
Argument. A composite member m < (s+1)^2 has a prime factor q <= sqrt(m) < s+1, q >= 5 (m is
coprime to 6). q is not s, s-2, s-3, s-4 (s = 0 mod 6: those are even, even, = 3 mod 6, even).
q = s-1 strikes only j = 0 (F2). So q <= s-5, q in G. Conversely if no p in G strikes j != 0
then both members are prime (a composite member would give a q in G by the same argument), so
s^2 + 6j is a twin centre inside ((s-1)^2, (s+1)^2): a rung. j = 0 is never a rung
(L_0 = (s-1)(s+1)).

A2. PROVED. S_p = {j in J* : j = a_p or b_p (mod p)}, a_p - b_p = 3^{-1} (mod p), and for EVERY
p in G: a_p != 0 and b_p != -3^{-1} (mod p). (F7 holds for all gears, not just p <= 61: s-1 and
s+1 are primes larger than p, so p does not divide s^2 - 1, so s != +-1 mod p.) Equivalent
integer form: if p | j then p does not divide L_j; and then p | U_j iff p | s^2 + 1.
Measured at s = 10008: no gear strikes a lower member at an offset it divides (0 of 1227 gears).

A3. PROVED. |S_p| <= 2 ceil(N/p) and >= 2 floor(N/p) (F6). For p > 2s/3: |S_p| in {1, 2}
(the window holds 4s-1 integers, so 4 or 5 consecutive multiples of p; a multiple p*m is a
member iff m is coprime to 6; among 4 or 5 consecutive m there are 1 or 2 such). For
p in (s/3, 2s/3]: |S_p| <= 4. Measured at s = 10008: the 370 top-band gears strike 600 columns
(min 1, max 2 each).

A4. PROVED (exact count for small gears). Let P_B = prod_{5 <= p <= B} p. Then
floor(N/P_B) * prod_{5<=p<=B}(p-2) - 1 <= |R_B| <= ceil(N/P_B) * prod_{5<=p<=B}(p-2)
(the offsets unstruck by all p <= B form prod(p-2) classes mod P_B; the -1 removes j = 0).
Table: B = 13: P = 5005, prod(p-2) = 1485 (ratio 0.2967); B = 17: 85085, 22275 (0.2618);
B = 19: 1616615, 378675 (0.2342). Measured at s = 10008: |R_13| = 1979, |R_19| = 1573,
|R_61| = 911 (N * prod(1-2/p) = 1979.3, 1562.6, 917.8).

A5. PROVED (the ledger identity). For every B: r(s) = |R_B| - T_B + X_B.
(|R_B| = #{omega = 0} + #{omega >= 1}; T_B = #{omega >= 1} + X_B; r(s) = #{omega = 0}.)
Hence leaf <=> X_B = T_B - |R_B| exactly. Checked at all 199 twin centres 60 <= s <= 10^4.

A6. PROVED. omega_B(j) <= floor(ln(L_j U_j)/ln B) <= floor(4 ln(s+1)/ln B) =: Omega_B (the
gears > B striking j are distinct primes dividing L_j U_j < (s+1)^4). At s = 10008, B = 61:
Omega = 8, measured max omega = 6; distribution on R_61: {0:160, 1:376, 2:228, 3:80, 4:61,
5:2, 6:4}.

A7. FALSE as a closing step ("T_B < |R_B| - 1 by A3, so a leaf is impossible").
Counterexample s = 10008, B = 61: T_61 = 1350 > 910 = |R_61| - 1. By A3 the sum of the
per-gear counts over (61, s-5] exceeds |R_61| as soon as 2 * sum_{61<p<=s} 1/p > 1, and the
measured T_61/|R_61| is 1.48 at s = 10008. The crude count does close for small s (measured:
T_61 < |R_61| at s = 60, 72, 102, 108, 138, 150), i.e. it proves LadderHyp only where the
gears > 61 are too few - roughly s < 900.

A8. FALSE as a closing step ("X_B >= Y_B / Omega_B >= T_B - |R_B| + 1", with
Y_B = sum_j omega(omega - 1) = sum over ordered pairs p != p' > B of |S_p ∩ S_p' ∩ R_B|).
The first inequality is PROVED ((omega-1)^+ >= omega(omega-1)/Omega for omega <= Omega), the
second fails: at s = 10008, Y_61 = 1828, Omega_61 = 6 (measured max), Y/Omega = 305 < 440 =
T_61 - |R_61| + 1 (actual X_61 = 599). Any bound on X through pair intersections and a
maximum multiplicity loses a factor Omega and does not close.

A9. PROVED (top band and free columns are a small part of the ledger). The columns whose
lower member is s^2 - e^2 = (s-e)(s+e), e coprime to 6, 5 <= e < sqrt(2s-1), sit at
j = -(e^2-1)/6 and are composite with no prime needed ("free", k = 0 family; 46 of them at
s = 10008). Likewise the upper members (s+1)^2 - (6u)^2 = (s+1-6u)(s+1+6u) at j = 2c - 6u^2
(33 at s = 10008) and the upper members (s+2)^2 - (6v+3)^2 at j = 4c-1-6v(v+1). All free
families for bounded k have O(sqrt(s)) columns. By F3 a top-band gear p = s-e strikes exactly
(s+k)^2 - (e+k)^2, and for e < sqrt(2s) these ARE the free columns (k = 0 gives the e-th
member of the first family; k = 1, e = 5 mod 6 gives u = (e+1)/6; k = 2, e = 1 mod 6 gives
v = (e-1)/6). Ledger at s = 10008: strikes on R_61 by gears in (61,141]: 302; (141, s/3]: 819;
(s/3, s/2]: 73; (s/2, 2s/3]: 72; (2s/3, s-5]: 84. So the top band supplies 6% of T_61 and the
free columns are 79 of 6671; the covering work of a leaf is done by the gears in (61, s/3].

A10. PROVED (split the ledger at s/3). Put R_mid = R_{floor(s/3)}, T_big =
sum_{s/3 < p <= s-5} |S_p ∩ R_mid|, X_big = sum_{j in R_mid} (omega_big(j) - 1)^+. Then
r(s) = |R_mid| - T_big + X_big, so |R_mid| > T_big is SUFFICIENT for a rung.
Measured: it holds at every twin centre 60 <= s <= 10^4 (minimum margin 3 at s = 72); at
s = 10008: |R_mid| = 256, T_big = 100, X_big = 4, r = 160. Each p > s/3 strikes at most 4
columns (A3), so T_big <= 4 (pi(s-5) - pi(s/3)); this is 3036 at s = 10008 against
|R_mid| = 256, so the per-gear counts cannot give A10's inequality: what is needed is that the
big gears' strikes mostly miss R_mid (measured: 1706 strikes in total, 100 on R_mid).

A11. GAP (the point where the argument stops). Two forms, one exact and one sufficient:

GAP A-1 (exactly LadderHyp for s >= s_0): for every twin centre s >= s_0,
  X_61(s) >= T_61(s) - |R_61(s)| + 1.
GAP A-2 (sufficient): for every twin centre s >= s_0,
  |R_{floor(s/3)}(s)| > sum_{s/3 < p <= s-5} |S_p ∩ R_{floor(s/3)}(s)|.
GAP A-2 splits into a lower bound on |R_{s/3}| (GAP A-2a: for every s >= s_0,
|R_{floor(s/3)}(s)| >= N / ln^2 s) and an upper bound on the big gears' strikes on it
(GAP A-2b: for every s >= s_0, ln(s) * sum_{s/3<p<=s-5} |S_p ∩ R_{floor(s/3)}(s)| <=
5 |R_{floor(s/3)}(s)|). Measured at s = 10008: |R_mid| = 256 vs N/ln^2 s = 78.6 (A-2a holds);
ln(s) * T_big = 921 vs 5|R_mid| = 1280 (A-2b holds). A-2a and A-2b together give A-2 for
ln s > 5.

---

## Attempt B - chains from something weaker than LadderHyp

B1. PROVED (the forest). Every twin centre t has at most one parent (F5): the open interval
(sqrt(t) - 1, sqrt(t) + 1) has length 2 and contains at most one multiple of 6, and t is a
rung of s iff that multiple is s and s is a twin centre. Measured to 10^8: 440311 twin
centres; 146796 have a multiple of 6 in that interval; 15588 (3.5%) have a twin-centre
parent; the longest run of consecutive twin centres with no twin parent is 14201.
Consequence: any hypothesis of the form "among K consecutive twin centres one has a twin
parent" is FALSE for K <= 14201; chains cannot be built upward from an arbitrary twin centre,
only downward through rungs.

B2. PROVED (what ChainHyp is). Let A_0 = all twin centres, A_{n+1} = {s : some rung of s is
in A_n}. ChainHyp <=> A_n != {} for every n <=> for every n there is a twin centre t whose n
successive parents exist (n successive roundings of the square root to the nearest multiple
of 6 are all twin centres). An infinite chain implies ChainHyp; the converse is not needed.

B3. PROVED (K-Hyp alone). K-Hyp: among any K consecutive twin centres >= s_0 at least one has
a rung. The rungs of s are consecutive twin centres, so if r(s) >= K then some rung of s has
a rung: depth(s) >= 2. K-Hyp gives nothing about depth 3 unless the chosen rung again has
>= K rungs. So the step "K-Hyp => ChainHyp" is a GAP: the missing statement is
GAP B-0: for every n there is a twin centre in A_n with at least K rungs in A_n.
(Circular as stated; replaced by B4.)

B4. PROVED (dichotomy lemma). Suppose for some K >= 2 and s_0:
 (i)  NC_K: among any K consecutive twin centres >= s_0, at least one has a rung;
 (ii) Dich_K: no twin centre s >= s_0 has 1 <= r(s) <= K-1.
Then every twin centre s >= s_0 with a rung is the root of an infinite chain, hence ChainHyp.
Proof. r(s) >= 1 gives r(s) >= K by (ii); the r(s) rungs are >= K consecutive twin centres
>= s_0, so by (i) one of them, s_1, has a rung; s_1 >= s_0, repeat. Both (i) and (ii) are
weaker than LadderHyp in the sense that neither implies it: (i) with K = 2 is "leaves are
never consecutive"; (ii) allows leaves and is not implied by LadderHyp either (it is a
statement about the positive values of r). Measured: r(s) >= 2 for all twin centres
s <= 10^4 (r = 2 at s = 6, 12, 18, 30 only); r(s) >= 21 for 1000 <= s <= 8000 (minimum at
s = 1152). So Dich_2 holds to 10^4 with s_0 = 6 and Dich_21 holds on [1000, 8000].

B5. PROVED (two consecutive twin centres t < t' = t + 6d as leaves). (a) The stretches are
disjoint: (t'-1)^2 >= (t+5)^2 > (t+1)^2; the gap between them holds 4(3d-1)(t+3d) integers.
(b) Both windows lie on one column line; gear p strikes offset j' of t' iff it strikes offset
j' + D of t with D = (t'^2 - t^2)/6 = 2d(t + 3d), the same integer for every gear. So the
residues of t'^2 and t^2 are related by the single shift D and nothing else; the two
coverings are two independent coverings of two windows of the same line by the same gears.
(c) The gear sets differ by the primes in (t-5, t'-5]. For d = 1 (t' = t+6, t >= 66) these
are exactly t-1 and t+1, and in the stretch of t' they strike exactly three columns:
(t+1)(t+11) = (t'-5)(t'+5) at j' = -4, (t-1)(t+13) = (t'-7)(t'+7) at j' = -8 (lower members),
(t+1)(t+13) = (t'-5)(t'+7) at j' = 2c'-6 (upper member) - the cofactors m of t+1 lie in
{t+10,...,t+13} and of t-1 in {t+12,...,t+15}, and only t+11, t+13 are coprime to 6. All
three are free columns (A9). Hence: if t and t+6 are both leaves, both stretches are covered
by the SAME gear set [5, t-5].
(d) No further constraint between the two coverings follows from F1-F7: each is a covering of
its own window by the fixed periodic strike sets, and F1-F7 say nothing joint about two
windows 8(t+3) apart. So the statement that they cannot both be leaves is a GAP.

B6. GAP statements from B.
GAP B-1 (NC_2): for all consecutive twin centres t < t' with t >= s_0: r(t) + r(t') >= 1.
GAP B-2 (Dich_2): for every twin centre s >= s_0: r(s) != 1.
B-1 and B-2 together give (B4) an infinite chain from every twin centre >= s_0 that has a rung,
hence ChainHyp, hence twins unbounded. Neither alone gives ChainHyp: B-1 alone allows a forest
where non-leaves have exactly one rung which is a leaf (depth <= 1 everywhere); B-2 alone
allows all twin centres beyond s_0 to be leaves.

---

## Attempt C - the long window (SixHyp) and the per-gear discrepancy that would close it

C1. PROVED (kernel). SixHyp: for every twin centre t there is a twin centre in
(5(t-1), 7(t+1)). SixHyp => twins unbounded (t' > 5(t-1) > t for t >= 6).

C2. PROVED (column form; the classes do not depend on t). t' = 6k with k in
K_t = Z ∩ (5(t-1)/6, 7(t+1)/6), |K_t| in {floor(t/3)+2, floor(t/3)+3}. Gear p strikes column k
iff 6k = +-1 (mod p) iff k = +-6^{-1} (mod p): the two classes are +-k_p where k_p = (p -+ 1)/6
is the column that contains p itself; they are the same for every t. A composite member
< 7(t+1) has a prime factor <= sqrt(7t+7), so k in K_t is a twin centre iff no gear
p <= sqrt(7t+7) strikes it; the gears t-1, t+1 play no role (t-1 > sqrt(7t+7) for t >= 12).
Contrast with A: the Ladder window has N = 2s/3 columns and gears up to s (window length =
1.5 x gear bound); the Six window has t/3 columns and gears up to 2.65 sqrt(t) (window
length = t^{1/2}/8 x gear bound). In the Six window every gear p has
|S_p ∩ K_t| >= 2 floor(|K_t|/p) >= 2 floor(sqrt(t)/8): no gear is in the 1-or-2-strikes
regime of F3.

C3. PROVED (Euclid columns). With P = prod_{5<=p<=B} p, every column k = 0 (mod P) has
6k-1 = -1 and 6k+1 = +1 (mod p) for all p <= B: both members B-rough, unconditionally.
In the Ladder window the same device only half-works (A2): j = 0 (mod P) makes L_j B-rough
but U_j is struck by exactly the p <= B dividing s^2+1, so one must take
P' = prod{p <= B : p does not divide s^2+1} and avoid two classes of j mod each p <= B dividing
s^2+1. In the Six window: SixHyp <= Six_P: for every twin centre t >= t_0(P) there is m with
5(t-1) < 6mP < 7(t+1) and 6mP-1, 6mP+1 both prime. Measured to t <= 10^7: P = 1 (SixHyp): no
failure (minimum 2 twin centres in every window); P = 5: last failure t = 18; P = 35: last
failure t = 828 (16 failures); P = 385: last failure t = 11172 (70); P = 5005: last failure
t = 175782 (497). For the Euclid columns the remaining gears p in (B, sqrt(7t+7)] strike
m = -+(6P)^{-1} (mod p): the same machine on the reduced line of m, window length |K_t|/P.
So C3 removes the gears <= B exactly and changes nothing else; it is not a route by itself.

C4. PROVED (sequential ledger in the long window). Order the gears 5 = p_1 < p_2 < ... <= 
sqrt(7t+7); R^{(i)} = columns of K_t unstruck by p_1..p_i; E_i := |S_{p_{i+1}} ∩ R^{(i)}| -
(2/p_{i+1}) |R^{(i)}| (the discrepancy of the (i)-rough set in the two classes of the next
gear). Then |R^{(i+1)}| = |R^{(i)}| (1 - 2/p_{i+1}) - E_i, and unrolling,
  #(twin centres in (5(t-1), 7(t+1))) = |K_t| Pi_all - sum_i E_i Pi_{>i},
where Pi_all = prod_{5<=p<=sqrt(7t+7)} (1 - 2/p) and Pi_{>i} = prod over the gears after p_{i+1}.
This is an identity; every term is an integer or a rational computable from the window.

C5. PROVED (what discrepancy would suffice - and that it suffices only in the long window).
Suppose E_i <= 2 sqrt(|R^{(i)}|/p_{i+1}) + 2 for all i (random-size discrepancy). Then, with
|R^{(i)}| <= |K_t| <= t/3 + 3 and Pi_{>i} <= 1 and y = sqrt(7t+7):
  sum_i E_i Pi_{>i} <= 2 sqrt(t/3+3) * sum_{p<=y} p^{-1/2} + 2 pi(y)
                    <= 4 sqrt(t/3+3) * y^{1/2} + 2y  <  3.8 t^{3/4} + 6 t^{1/2}   (t >= 10^4).
If also Pi_all >= 1/ln^2 y (GAP C-4 below), the main term is >= (t/3) * 4/ln^2(7t+7), and
main > error as soon as t^{1/4} > 2.9 ln^2(7t+7) + 5 t^{-1/4} ln^2(7t+7), i.e. t >= 10^15.
So: random-size per-gear discrepancy + the product bound => SixHyp for t >= 10^15 (and SixHyp
is measured to 10^7; the range 10^7 < t < 10^15 would remain).
In the Ladder window the same computation FAILS even with E_i <= 2: the error sum is
>= 2 * #(gears in (sqrt(2s), s-5]) * (ln p/ln s)^2 ~ 2s/ln s, while the main term is
N Pi_all ~ (2s/3) * C/ln^2 s; the gears above sqrt(2s) are too many for any per-gear
discrepancy bound to be summed. This is the exact reason A stops at A11 and C does not:
the number of gears is sqrt(t)/ln t against a window of t/3 columns.

C6. GAP statements from C.
GAP C-1 (SixHyp as an integer sentence): for every twin centre t >= 6 there is k with
  5(t-1) < 6k < 7(t+1) and gcd((6k-1)(6k+1), prod_{5<=p<=sqrt(7t+7)} p) = 1.
GAP C-2 (Euclid sub-family, P = 35): for every twin centre t >= 830 there is m with
  5(t-1) < 210m < 7(t+1) and 210m-1, 210m+1 both prime.
GAP C-3 (per-gear discrepancy in the long window): for every twin centre t >= t_0, every gear
  p <= sqrt(7t+7), with R = {k in K_t : no gear q < p strikes k}:
  #{k in R : k = +-6^{-1} (mod p)} <= 2|R|/p + 2 sqrt(|R|/p) + 2.
GAP C-4 (product bound, rational inequality): for every x >= 5,
  prod_{5<=p<=x} (1 - 2/p) * ln^2 x >= 1.   (At x = 61 the left side is 2.33.)
C-3 and C-4 together give C-1 for t >= 10^15 (C5). C-3 is PROVED by exact CRT for the first
few gears only: when prod_{5<=q<=p} q <= |K_t| the rough set R is periodic inside the window
and the count in the two classes is 2 |R|/p up to at most 2 prod_{5<=q<p}(q-2) (one partial
period); this is within the allowance 2 sqrt(|R|/p) + 2 iff p * prod_{q<p}(q-2)^2 <= |R|,
i.e. p = 7 for t >= 10^3, p <= 11 for t >= 1.5*10^4, p <= 13 for t >= 2.4*10^6, p <= 17 for
t >= 4*10^8. For every gear beyond that C-3 is the gap.

---

## GAP table (ranked by how much of the target each would prove)

| rank | GAP | statement (one quantified sentence) | proves | test a script can run | refuted by |
|---|---|---|---|---|---|
| 1 | A-1 | For every twin centre s >= s_0: X_61(s) >= T_61(s) - \|R_61(s)\| + 1. | LadderHyp for s >= s_0 (exact equivalent), hence ChainHyp, SixHyp, twins unbounded | for each twin centre s <= 3*10^4 compute X_61, T_61, \|R_61\| (measure2.py `window_stats`); margin = X - (T - \|R\| + 1) = r(s) - 1; measured min margin 2 at s = 72 | a margin <= 0, i.e. r(s) = 0 |
| 2 | A-2 | For every twin centre s >= s_0: \|R_{floor(s/3)}(s)\| > sum_{s/3<p<=s-5} \|S_p ∩ R_{floor(s/3)}(s)\|. | LadderHyp (sufficient, not equivalent) | same script, margin = \|R_mid\| - T_big; measured min 3 at s = 72, 156 at s = 10008 | margin <= 0 at a twin centre s where r(s) >= 1 would show A-2 is too strong; margin <= 0 with r(s) = 0 refutes Ladder |
| 2a | A-2a | For every twin centre s >= s_0: \|R_{floor(s/3)}(s)\| >= N/ln^2 s. | half of A-2 | compute \|R_mid\| ln^2 s / N per s; measured 3.26 at s = 10008 | a value < 1 |
| 2b | A-2b | For every twin centre s >= s_0: ln(s) * sum_{s/3<p<=s-5} \|S_p ∩ R_{floor(s/3)}(s)\| <= 5 \|R_{floor(s/3)}(s)\|. | other half of A-2 (with 2a gives A-2 for ln s > 5) | compute ln(s) T_big / \|R_mid\| per s; measured 3.6 at s = 10008 | a value > 5 |
| 3 | B-1 + B-2 | (B-1) for all consecutive twin centres t < t', t >= s_0: r(t) + r(t') >= 1; (B-2) for every twin centre s >= s_0: r(s) != 1. | together: an infinite chain from every non-leaf >= s_0, hence ChainHyp and twins unbounded; NOT LadderHyp | count twin centres in ((s-1)^2,(s+1)^2) for consecutive s (measure.py part 1, extend to s <= 10^5 with a sieve to 10^10 in segments); measured: no leaf to s <= 10^4, min positive r = 2 (s <= 30), r >= 21 on [1000, 8000] | B-1: two consecutive s with r = 0; B-2: any s >= s_0 with r(s) = 1 |
| 4 | C-1 | For every twin centre t >= 6 there is k with 5(t-1) < 6k < 7(t+1) and gcd((6k-1)(6k+1), prod_{5<=p<=sqrt(7t+7)} p) = 1. | SixHyp, hence twins unbounded; not Ladder, not Chain | count twin centres in (5(t-1), 7(t+1)) for every twin centre t <= 10^8 (measure2.py part c, P = 1); measured to 10^7: minimum count 2 | a count 0 |
| 5 | C-2 | For every twin centre t >= 830 there is m with 5(t-1) < 210m < 7(t+1) and 210m +- 1 both prime. | SixHyp (stronger than C-1: a sub-family) | same with P = 35; measured: last failure t = 828, none in (828, 10^7] | a failure t > 828 |
| 6 | C-3 + C-4 | (C-3) for every twin centre t >= t_0, every gear p <= sqrt(7t+7), R = k in K_t unstruck by gears < p: #{k in R : k = +-6^{-1} mod p} <= 2\|R\|/p + 2 sqrt(\|R\|/p) + 2; (C-4) for every x >= 5: prod_{5<=p<=x}(1-2/p) ln^2 x >= 1. | SixHyp for t >= 10^15 only (C5); with the measured range leaves 10^7 < t < 10^15 open | C-3: at t = 10^6 and 10^7 run the sequential sieve on K_t and record max over p of (count - 2\|R\|/p - 2 sqrt(\|R\|/p) - 2); C-4: product * ln^2 x for x <= 10^7 | C-3: a positive value (then re-run with 3 sqrt instead of 2: C5 tolerates any constant); C-4: a value < 1 |
| - | B-0 | For every n there is a twin centre in A_n with >= K rungs in A_n. | ChainHyp (with K-Hyp) but circular | - | - |

FALSE steps recorded: A7 (crude count closes: fails from s ~ 900, e.g. T_61 = 1350 > 910 at
s = 10008); A8 (pair-intersection bound closes: 305 < 440 at s = 10008); B1 (parent-side
K-Hyp: orphan run of 14201 below 10^8); C5-in-the-Ladder-window (even zero-error per-gear
counts cannot be summed over the s/ln s gears above sqrt(2s)).

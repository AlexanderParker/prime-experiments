# Constructor: cumulative findings (compacted)

Compacted 2026-08-23; full verbatim rounds 1-19 log at
archive/constructor-full-r1-19.md (byte-identical to the pre-compaction file;
it contains 19 rounds: 1-6 and 8-20, there was no round 7).

**Mandate:** proof of twin-prime infinitude by construction/contradiction from
the proven mechanical laws; Condition X (a twin-free window) is the
contradiction target; every claim reproduced by a named script with asserted
identities.

---

## 1. Definitions (each stated once)

* **Slot** k = the pair (6k-1, 6k+1). **Gear** q (prime >= 5) **blocks** k
  iff k = +-c_q (mod q), c_q = 6^{-1} mod q, i.e. iff q divides a member.
* **Window** W(y) = {k : y < 6k-1 and 6k+1 < y^2}; N = |W(y)|; P = prime
  members among the 2N members; C = 2N - P composites.
* **Root kill**: composite m attributed to the unique gear
  lpf(m) <= sqrt(m) < y. **Horizon theorem**: the top gear contributes
  nothing interior (R(top) = 0); primality above y = non-divisibility below.
* **Census**: n0/n1/n2 = slots with 0/1/2 composite members
  (**twin/fragile/double**); D(t) = doubles count to prefix t.
* **Condition X**: some window W(y) contains no twin (n0 = 0).
* **Machine** M = gear set {5..y}; **openings** = exposed (unblocked) slots;
  **gap word** = gaps between consecutive openings. F(M) (= F_k, k-frame) =
  max gap (2-dim Jacobsthal object); F2(M) = max adjacent-pair sum;
  **spectrum** F_j(M) = max sum of j consecutive gaps. Adjacent/halved frame
  = 3 x k-frame; F(2,y) = the adjacent-frame chain value.
* **Consecutive step** M -> M+q' (q' = next prime): incr = F(M+q') - F(M);
  excess = F(M+q') - F2(M). **k-chain** = k consecutive M-openings all
  deleted by q'; its merge spans k+1 consecutive gaps of M.
* u' = round(q'/6); qualifying interior values V(q') = {v = 0 or +-2c mod q'}
  (teeth {c, q'-c}); literal letters a = 2u', b = q'-2u'. **Literal chain** =
  interiors exactly the alternating {a, b}. **Padded link** = interior gap
  = 0 mod q', hence >= q'.
* **Word** w = a chain's interior-gap sequence; span(w) = its sum; FS(w) =
  sum of the two flanking gaps at an occurrence; FS_max(w; M) = max over
  occurrences; occ(w) = occurrences per period; **compatible** = has a valid
  tooth start (from q' alone; firing is binary).
* E = the 15-residue exposed set mod 35 (gears 5, 7); carrier
  S_m(w) = {r in E_m : every partial sum r + w_1..w_j in E_m}.
* m_k = omega_L(k)*omega_R(k) (distinct-gear divisor counts of the members);
  S1(t) = sum m_k, M2(t) = sum m_k^2; R(t) = (S1^2/M2)/(t - P(t)).
* p_1 = density of qualifying-size gaps; p_j = qualifying rate of j-windows;
  L = ln(1/p_1); lambda = exponential scale of the window-sum tail;
  rho = machine density prod(1 - 2/q).
* litcap(q' mod 210) = the literal-chain member cap (2, 3, 4, or 6).

---

## 2. Established results

**R1 (slot cap, L1).** No gear blocks both members of a slot (q | 6k-1 and
q | 6k+1 force q | 2). Hence a gear root-kills at most one member per slot, a
double's two kills come from distinct gears, slot multiplicity is 0/1/2 only.

**R2 (partition of supply, L2).** Root (lpf) attribution partitions the
composites exactly: sum_q R(q) = C = 2N - P, zero double-counting.
R(q) = [y < q^2 < y^2] + #{r prime > q : y < qr < y^2} + H(q); the corpus
formula pi(y^2/q) - pi(q) + 1 is the first two terms (H large only for gear 5).

**R3 (census identities + zero-slack theorem).** Always: n0 + n1 + n2 = N,
n1 + 2 n2 = C, P = n1 + 2 n0, N - P = n2 - n0. Under X the census is forced in
every prefix: n1(t) = P(t), n2(t) = N(t) - P(t). Consequences: C1 (P <= N
globally), C2 (every run I of slots has P(I) <= N(I); prefix margin
n2(t) - n0(t) >= 0 everywhere), C3 (pseudo-twin ledger sum_q PT(q) = n1 =
P - 2 n0), C4 (prefix demands must be met by the pi(sqrt(t)) - 2 gears active
at depth t).

**R4 (run-condition equivalence).** Some run has P(I) > N(I) iff the window
has a twin (a twin slot is a run of excess +1). So statement CUM ("every
window has a run with P(I) > N(I)") is EXACTLY equivalent to Reduction A - a
lossless reparametrisation with the gears dropped out; no strictly weaker
ingredient exists. The excess-run pattern (46 primes in [53,283]) is
constellation-admissible - no congruence obstruction forbids recurrence.

**R5 (computed censuses).** y=13: N=25, P=32, C=18, n0=9 - C1 fails (P-N=+7).
y=23: N=83, P=90 - C1 fails (+7). C1 bites only while prime density among
+-1 mod 6 members exceeds 1/2, i.e. y < ~e^6 ~ 403. y=47: N=359, P=313, C1
passes (C-N=+46) but C2 fails by 7 (run of 39 slots, members 53..283, 46
primes). E(y) = max run excess: 7, 4, 3, 3, 3, 3, 3 at y = 47, 101, 199, 503,
1009, 2003, 5003 - flat 3, realising runs within ~700 of y (283, 1277-1303,
2657-2713, 5639-5659). First double-composite slot in all of N is k = 20
(pair (119,121)); every slot k <= 19 has a prime member.

**R6 (roots-of-unity law).** Slot k is hit by distinct gears q, q' iff
36 k^2 = 1 (mod qq'). Trivial roots +-1 = same-member (semiprime) hits;
nontrivial roots +-r, r = CRT(+1 mod q, -1 mod q'), = cross-member hits. A
slot is double iff 6k lands on a nontrivial root of unity mod qq' for some
active pair. The double set D is one fixed pinned subset of the integers -
computable by semiprime arithmetic, no primality tests. For twin-pair gears
(p, p+2) the nontrivial root is p+1.

**R7 (onset law + absolute cap).** L0(y) = lag from window start to first
double slot; n2 = 0 there unconditionally, so under X the onset prefix is
perfectly fragile (one prime + one composite per slot). Via
Montgomery-Vaughan pi(x+H) - pi(x) < 2H/ln H: **L0(y) <= L* = 27129 for
every y** (6 L* + 2 = 162776 > e^12 = 162755) - unconditional theorem of the
programme. Measured (442 windows, 13 <= y <= 3163): max L0 = 17 (y = 13);
L0 = 0 in 153/442; a twin precedes the first double in 132/442.

**R8 (descent / layer bands, scoped).** X at y forces every layer band
(y'^2, y''^2) above y (y'' = nextprime(y')) twin-free; bands have length
x^(1/2+o(1)) at height x (thinnest, at twin endpoints: exactly 4*sqrt(x) + 4).
Input tower: T1 a prime in every band - OPEN (implied by Legendre/Cramer, NOT
by RH); T2 a pair with gap <= d in every band - proven localisation floor is
exponent 0.525 (Alweiss-Luo 2018, inherited from Baker-Harman-Pintz), need
1/2; T3 the parity step 246 -> 2 (no partial result). Bounded-gap pair
density is ample (Maynard-Tao, surplus x^(1/2)/polylog); localisation and
parity fail. Route blocked at T1.

**R9 (X-consistency equation).** Arithmetic census theorem (unconditional):
P(t) = t - D(t) + n0(t) at every prefix; X iff P(t) = t - D(t) for all t.
Verdict: satisfiable - the forced value is the unconditional pointwise floor
of P, and every unconditional ceiling sits a parity factor above it:
rho(t) = (t - D)/(2H/ln H) is provably <= 1, measured max
0.4687/0.4785/0.4828 at y = 101/211/503, drifting to 1/2. Any theorem
separating P(t) from its floor at one t IS a twin-existence theorem
(separation = n0(t)). Supply decomposition: the only unconditionally
guaranteed doubles supply is the g = 2 classes pinned at u' <= (y+1)/6 by the
twins below y - 5-9% of split incidences at all scales; the other 91-95% is
alignment-conditional (~51% landing rate). Improving the MV/B-T constant 2
toward 1 is itself parity-class (Motohashi: Siegel-zero consequences).

**R10 (moment inventory + inversion zone).** X <=> n2(t) = t - P(t) <=> the
hit schedule compresses at mean M_X = S1/(t-P). Cauchy-Schwarz ceiling
C_CS = M2/S1 is unconditional; measured C_CS/M_X = 1.26, 1.41, 1.53, 1.58 at
y = 211, 503, 2003, 5003 - GROWING (lnln-divergent m-dispersion) while the
needed window M_X/M_real narrows (1.22 -> 1.05). **Inversion zone**: wherever
R(t) > 1, moments + prime count alone force n0 > 0. Zone nonempty at every y
tested to 10007; worked instance y = 503, t = 4: S1 = 3, M2 = 5 give
n2 >= 9/5 > 1 = t - P, forcing the twin (521, 523). Closing the zone as a
theorem needs a short-prefix prime lower bound at 0.07-0.13/integer
(superdense class - nothing published below the 0.525 exponent).

**R11 (fate of the zone).** R = eff * boost, eff = (S1^2/M2)/n2,
boost = 1 + n0/(t-P). Ladder to 10^7: sup R -> 1+ by trend, first EMPTY
windows at y = 5000011 and 10000019 (band T = 200000). Windows opening with a
twin within <= 4 slots revive R to 1.923 at any y (verified y = 5000087,
5000101, 5000539) - but "the zone revives for infinitely many y" is
EQUIVALENT to the twin prime conjecture. Detector of bottom twins (certifies
from moments + P without exhibiting the pair), never a generator; gives the
conjecture an exact floor-arithmetic address in each window's first slots.

**R12 (mirror theorem).** The involution k -> -k swaps omega_L and omega_R and
fixes m_k; every mirror-augmented moment doubles, so every ratio (R, eff,
boost, M_X, all ceilings) is invariant. Mirror-awareness is vacuous at the
moment level; any edge must use positions jointly with signs.

**R13 (LP moment ceilings).** Sharp n2 lower bound given S1, M2, M3 by LP:
integer order-2 beats continuous CS by 0.3-0.5% (extends the zone slightly:
at y = 10007, t = 17204 it still refutes, n2 >= 7744 > 7702). Order 3 adds
nothing without the arithmetic cap m <= (log_5 y^2)^2; with it, 0.6-2.8% more.
Window-scale chasm (~48%) untouched. The X-gap lives entirely in the zeroth
moment (twin mass), which no power moment sees - corroborates Mechanic
independently. Forcing must come from placement/pin/alternation structure.

**R14 (tolerance theorem - the route's sufficiency statement).** If the
increment law F(M+q') - F(M) <= alpha*q' holds at every consecutive step with
q' > 47, for ANY fixed alpha within alpha*(y)-scale - in particular alpha =
2.5 or 3 - then F(2,y) <= 354 + alpha*(S(y) - 328) < (y^2 - y)/2 for every
prime y >= 53 (S = prime sum). Checked exactly at every prime in [53, 10^6]:
zero failures, worst ratio 0.6557 at y = 113 (alpha = 3); beyond 10^6 by
Rosser-Schoenfeld (S(y) < 1.25506 y^2/ln y, sufficient once ln y > 7.54).
With y <= 47 known directly this gives a survivor in every window - twins
infinite. alpha*(y) grows like ln y: 5.64 at y = 101, 8.71 at 10^4, 13.3 at
10^6. Base chain (adjacent frame): F(2,y) = 6, 15, 21, 33, 54, 75, 102, 129,
174, 264, 273, 309, 354 at y = 5..47; F(2,53) >= 420 (search unfinished;
alpha = 2.5 demands F(2,53) <= 486, alpha = 3 <= 513). State the route at
alpha = 3 (R26).

**R15 (saturation regime).** If q - 1 > F(M), no two consecutive openings can
both be deleted, so F(M+q) = F2(M) and incr = F2 - F <= F < q (alpha = 1
automatic). But along the consecutive chain q' < F(M) always - the compliant
and needed regimes are disjoint; the theorem covers far-gear additions only.

**R16 (corridor laws, mod 35).** Endpoint law: a gap of length G has left
endpoint a mod 35 in A(G) = {r in E : r + G mod 35 in E}, |A| = 3..15
(G = 34 mod 35 forces a in {3, 18, 33}); concentration exceeds forcing
(gears <= 19: all 20 record gaps at residue 5; gears <= 23: records at
{3, 33}). Adjacency law: 294 of the 1225 length-pairs mod 35 have A3 empty.
Both kernel-checkable; prune the F(2,53) search by factor 2-5x.

**R17 (record scarcity, measured).** Full periods have 4-20 maximal gaps
(mirror-paired); minimum separation between record gaps is 0.45-2.29% of the
entire primorial period (851,695 slots at gears <= 23) - anti-clustering five
orders beyond what lemma 1 needs. Adjacent (F2-F)/q along the chain: 0.92,
0.88, 1.10, 0.78, 0.52 (gears <= 11..23), max 1.16, min 0.15 further out -
never above 1.2, no growth.

**R18 (top-stratum adjacency; per-machine alpha1 closure).** At machines
y = 13, 17, 19, 23 the top stratum occupies 4-6 classes mod 385 and no two
top-stratum classes can be adjacent (class-level check EMPTY at all four).
Dangerous-pair alpha1 certificates close at all four machines by three tiers
(A3-empty / mod-385 disjoint / direct); tier C grows with machine (4 -> 96).

**R19 (merge census + spectrum reduction).** Full-period censuses, steps
11->13 .. 23->29: chains 264 / 2,897 / 43,462 / 745,480 / 15,660,527;
excess_k = 0, 2, 0, 3, 4; max k = 2, 2, 3, 2 (k = 3: 62 chains at 19->23).
Argmax interiors are literal {2u', q'-2u'}; identity
excess = interior_sum - (F2 - g_L - g_R) verified. Rigorous:
F(M+q') <= F_{k_max+1}(M), excess <= F_{k_max+1} - F2. Spectrum increments
are q/3-scale, not F-scale (isolation generalised, measured).

**R20 (fuel caps).** Tail-run cap (residue-free): k_max <= T(M, 2u') + 1,
T = longest run of consecutive gaps >= 2u' (measured T = 3,2,4,3,4,5).
**Literal cap theorem**: the literal walk must stay in E mod 35; the maximal
run is a function of q' mod 210 only. Over all 48 invertible classes: cap 2
at 24 classes, cap 3 at 4, cap 4 at 14, cap 6 at 6 (q' = 37, 53, 83, 127,
157, 173 mod 210). **Literal chains have at most 6 members, for every gear,
forever** (verified against every prime to 5000). Explains realized k_max
2,2,3,2,4 (caps 2,2,4,3,4; saturated at 17, 19, 31); k = 5 at q' = 31
forbidden; the k = 4 event (29->31, word (10,21,10), 4 per period) sits at a
cap-4 gear. Extension beyond the cap requires a padded link.

**R21 (word-indexed identity).** With W(q') = the alternating words of length
<= litcap - 1 plus padded words:

    F(M+q') = max( F2(M), max over COMPATIBLE w of [span(w) + FS_max(w; M)] )

- an identity, not a ceiling (every occurrence of a compatible word fires in
|valid starts| of the q' CRT copies; incompatible words never fire). Word
list and compatibility depend on q' mod 210 alone; only occurrences and
flanks come from M. Verified exactly at all six steps 11->13 .. 29->31;
consistent with the padded winner at 31->37. Fuel length and record growth
are separate channels: long words have small flanks.

**R22 (tier A: both-maximal exclusion).** A word occurrence with flanks is an
(l+3)-point chain in E mod 35. "Both flanks maximal (= F)" is machine-free
FORBIDDEN at 14 of 16 word-step pairs (exceptions: w = (8), (15) at 19->23).
Tier B (moduli 385 ... 1616615, gears through 19) adds EXACTLY ZERO new
exclusions - feasibility lifts to every multiple modulus. Hierarchy: A
(machine-free, scalable) vs C (full period scan, unscalable - 3.3e10 slots at
31->37). Compatibility is CRT-independent of the carrier, so firing and
tier A never interact.

**R23 (padding arithmetic).** A padded link costs a full q' of budget while
the whole budget grants (alpha/3)q'; count bound p <= (F + (alpha/3)q')/q'
(step-dependent, ~F/q', grows like y/log y - no structural cap); onset gate
F(M) >= q' (Mechanic; the first three steps have none by impossibility).
Padded gaps are rare - 0 / 0 / 0 / 86 of 378,675 (0.023%) / 6 of 7,952,175 /
2,090 of 214,708,725 at steps 11->13 .. 29->31; only the exact value q' ever
occurs; no k >= 3 padded window exists. **The corpus's gear-37 anomaly is the
onset of padding**: at 31->37 the winner is [pad 37][literal 12], span 49,
FS 39, merged 88 = F_k(37) - a tier switching on, not a fluctuation. Min
opening-distance from a maximal gap to a padded gap: 710 / 558,331 / 47,729
(steps 19->23 / 23->29 / 29->31). At 41->43: p <= (91+43)/43 = 3.1, so p = 3
there is consistent.

**R24 (padded-flank requirement shape).** With p padded and ell - p literal
interiors: FS <= F - (p - alpha/3)q' - (ell-p)q'/3; at alpha = 2.5, p = 1
forces FS < F - q'/6, p = 2 forces FS < F - (7/6)q'. REQUIREMENTS given
tolerance, not derived facts (X15); padding provably limits only its own span
contribution (count + onset gate), never its flanks.

**R25 (per-step constants, k-frame).** Winner word, span, FS, incr/q':

    11->13 (4)  lit  span 4  FS 4   0.308 | 13->17 (6)   lit  6  12  0.412
    17->19 (13) lit  13  12  0.368        | 19->23 (8,15) lit 23  11  0.391
    23->29 (10) lit  10  33  0.310        | 29->31 (10)  lit 10  48  0.484
    31->37 (37,12) PADDED span 49 FS 39   0.811

Budget incr/q': 0.833 (alpha 2.5), 1.000 (alpha 3). Corpus next steps: 0.220q
and 0.837q adjacent (37->41, 41->43) - padding is intermittent. FS can exceed
F (1.09F at 13->17, 1.12F at 29->31), so "FS <= F" is false; measured
(FS-F)/q' <= +0.161.

**R26 (route form; alpha = 3 restatement).** The tolerance hypothesis factors
exactly into: (A) word list - finite, from q' mod 210 alone, PROVEN; (B)
literal span - <= 5 letters, span < (10/3)q', PROVEN; (C) padded span -
count-capped and onset-gated, PROVEN; (D) flank bound
FS_max(w) <= F + (alpha/3)q' - span(w) for every compatible w - OPEN, the
sole gap; (E) partial: R22. At alpha = 3 the binding step 31->37 has margin
+7 (19% of q') vs +0.83 (2.7%) at alpha = 2.5 - the 2.7% was an artifact of
the tighter admissible constant; nothing else depends on the choice. (D) at
alpha = 3 is incr_k <= q' localised to <= 6 pinned words per step.

**R27 (binding flanks are mid-size).** The flank pairs attaining FS_max are
never maximal: largest single flank across all 15 word-steps runs 0.16F to
0.81F (e.g. 29->31: FS_max = 48 at (gL,gR) = (18,30), F = 43) - so the
both-maximal exclusion excludes a configuration that never binds (X16). (D)
is a mid-tail x mid-tail pair-sum bound; margins at alpha = 3: >= 0.52q' at
every literal step, 0.19q' at the padded step.

**R28 (spectrum bridge + ordering).** span(w) + FS(w) = a sum of exactly k+1
consecutive gaps <= F_{k+1}(M) (definitional; the one-line kernel bridge).
Strict ordering: Wall-V clustering (F2 - F = O(q'), extreme x anything) ==>
spectrum flatness (F_{k_max+1} - F <= q', all windows) ==> (D) (qualifying
windows only, relative density ~(3/q')^{k-1}). Spectrum flatness is FALSE
(X17), so (D) cannot be weakened by dropping position information.

**R29 (measured envelope).** Across all 15 word-steps:
span(w)/F + maxflank(w)/F in [1.00, 1.45]. Ratio form of the requirement:
merged/F <= 1 + q'/F, gap +1.286 down to +0.121 at 31->37 - thinning
F-relative, stable q'-relative (incr/q' mean ~0.44, no upward trend).
Superseded as a machine law by the occurrence form (R33).

**R30 (suppression decomposed; par trading; shallow flatness).**
Compatibility suppresses via (i) size threshold (interiors >= 2u' ~ q'/3,
often zero effect at depth 3) and (ii) residue coincidence (3 of q' residues,
~10% - carries the whole suppression at binding depth). **Par trading**: each
added link buys ~q'/2 of span and costs about the same in flank sum, so
merged max is nearly depth-independent (spreads 0-14% at machines 13-29;
band ~25% after machine 31's 22.7%, machine 41's 9.3%); k_win <= 3 at all
seven measured steps, winners SHALLOWER as machines grow (k_win = 3 at
machine 31, 1 at 41). **Shallow flatness** F_4(M) - F(M) <= q' holds at all
six machines 11-29 (ratios 0.85, 0.88, 0.79, 0.57, 0.83, 0.87) where depth-5
fails at three. Intermediate target (D-a) k_win <= 3 + (D-b) F_4 - F <= q' -
subsumed by R31.

**R31 (suppression law + corrected flatness - the current criterion).**
Window composition profile: the extremal j-window migrates to several medium
gaps (max element/sum 0.35-0.64); deep extremal windows never contain the
record gap - why the isolation law cannot control them. Luck test: the
qualifying maximum sits where a random p-sample's max would (luck 10^-0.1 to
10^-1.3 - plausible); the structure lives in p_j: qualifying interiors are
strongly negatively correlated (p_j vs p_1^(j-2): x26 at machine 23 j=4, x6.7
and x1400 at machine 29 j=4, j=5). **Suppression law**:
F_j - qualmax_j ~ lambda*(j-2)*L. **Corrected flatness**:
(D) <== F_j(M) - F(M) <= q' + lambda*(j-2)*L for every j >= 2 - holds 15/15
(corrected margins 4.7 to 20.8, bounded, non-growing in j) where raw flatness
fails 5/15. The j = 2 case IS lemma 1 (F2 - F <= q'); deeper cases are the
EASIER ones - the reverse of what rounds 8-17 assumed. Status: lambda fitted,
p_1 measured, order-statistics step heuristic.

**R32 (rigorous exposure bound + how much anti-correlation (D) needs).**
"Gap = v" = (both endpoints exposed) AND (no opening between); dropping the
second gives the CRT-exact bound

    p_j <= (1/rho) * sum over qualifying tuples of prod_q c_q(...)/q

(multi-lag c_q(g); the 1/rho converts per-slot to per-opening - X20).
Corrected, the bound clears machine 19 j=6 and machine 23 j=5 but is SHORT by
x28.8 (machine 23 j=6) and x2.0 (machine 29 j=5, j=6) - the missing factor is
exactly the dropped no-opening condition (a local renewal factor). And (D)
needs almost no anti-correlation: where a constraint exists at all,
INDEPENDENCE (p_1^(j-2)) clears it by x170 to x201,381 - (D) only needs p_j
not positively correlated by more than ~170x.

**R33 (anti-correlation law + occurrence form of (D)).** Measured
R(lag) = P(both qualifying)/p_1^2: an adjacency effect and nothing more -
lag-1 deficit (exact ZERO at machines 11-17: qualifying gaps cannot be
adjacent there; 0.039-0.638 at machines 19-29), rebound above independence at
lag 2 (up to 1.897), independence by lag 4-5. Higher orders
super-multiplicative (machine 29: p_5/p_1^3 = 7.1e-4 vs pairwise 2.2e-2, a
further 30x). **Flank order-statistic law**: maxflank(w) ~ 2.05*ln(occ(w))
(sd 0.27), FS_max(w) ~ 2.77*ln(occ(w)) (sd 0.24); 2.77 matches the
independently fitted lambda = 2.73 - the envelope follows occurrence count,
not span (outlier: the 4-occurrence word (10,21,10)). **(D) in occurrence
form**: span(w) + lambda*ln(occ(w)) <= F + q' for every compatible w - the
first form where every term is a counting quantity with a closed-form upper
bound (occ <= N x exposure product). qspec41: max_j Q_j = 110 vs F + q' = 133
(margin +23), Q_j = 0 for j > 8 - the word-free criterion holds at the bigger
machine; the flagged margin collapse (0.45q' -> 0.10-0.11q') is ratio-only.

**R34 (wall verdict, multiplicative route).** Evades Wall I (no capacity
comparison), Wall II (no prime lower bound anywhere), Wall IV (hypothesis
strictly stronger than the conjecture, honestly lossy), and is not Wall III
(dimension-1 test: the analogous increment statement for ordinary Jacobsthal
would sharpen Iwaniec - unproven even without parity). The obstruction is a
FOURTH wall: **extreme-value control of sieve patterns (Wall V)** - now at
bounded complexity. The only route whose missing lemma is about the
machine's own gap word rather than about primes.

---

## 3. Refuted claims (kept as refuted - do not re-derive)

**X1. Pair-coincidence doubles bound.** n2 <= L*s(z) + g(g-1) needs s(z) < 1,
i.e. z <= 137 (s(127) = 0.959, s(139) = 1.005), band top < 19321 - but so
short a band can hold 12/ln(6L) >= 1.22 primes/slot (Brun-Titchmarsh), above
the < 1/slot X must supply. Empty intersection.

**X2. Onset-scale contradiction.** Killing X in the onset prefix needs
pi(y+H) - pi(y) >= H/6 + 1 (superdense, Hensley-Richards strength) - and as
a universal statement it is FALSE: 310/442 real windows have NO twin in the
onset prefix; the forced alternation is realised 70% of the time.

**X3. CUM as leverage.** Exactly equivalent to Reduction A (lossless
pigeonhole both ways); E(y) collapses to 3, content degenerates toward the
twin statement. Diagnostic value only.

**X4. Naive descent.** As first sketched, re-derives Reduction A at constant
~1/2; the band-weakened form remains bounded-gap strength, blocked at T1
(a prime between consecutive prime squares) before twins even enter.

**X5. Overdetermination of the X-consistency equation.** Zero degrees of
freedom: the census theorem makes both sides one arithmetic; the system
collapses to n0(t) = 0 with no residual structure. The forced value is the
unconditional floor; every ceiling sits a parity factor 2 above.

**X6. "CS ceiling lands at 2x the need".** Opposite measured: C_CS/M_X grows
1.26 -> 1.58 while the target window narrows 1.22 -> 1.05 - they move apart
on both ends. Bonferroni-2 is vacuous once mean m > 3. Selberg Lambda^2
bounds n0 from ABOVE - wrong direction against n0 = 0.

**X7. The inversion zone as generator.** "Zone revives for infinitely many y"
is equivalent to the twin prime conjecture; no certificate short of the
conjecture exists. Detector only (R11).

**X8. Mirror/third-moment tightening.** Mirror-awareness is vacuous at every
moment order (R12); LP order 3 moves < 3% against a 48% gap; the X-gap is
zeroth-moment only (R13).

**X9. Multiplicative per-step bounds of budget shape.** r <= (q'/q)^2 is
false at 6 of 12 chain steps; uniform ratio caps r <= c cannot close (pi(y)
steps vs a y^2 budget force geometric mean -> 1). Only the additive shape
incr <= alpha*q closes (R14). Corpus 6a's negative closure of per-step
increment bounds holds only for the elementary odd-sum threshold 1.8.

**X10. A-priori chain-condition cap in the needed regime.** Saturation
theorem applies only when q - 1 > F(M) - disjoint from the consecutive chain.
In-range, the raw cap (k_max + 2) is exponentially over budget; corpus 5.5:
gap structure alone cannot bound k.

**X11. Bounded-modulus residue laws capping sizes.** Every (G1, G2) pair is
within L1 distance 1 of a corridor-allowed pair (escape distance 1, any
bounded modulus) - corridors constrain where, never how big. Verbatim for
flank pairs (408/1225 forbidden at w = (10), slide 1 escapes) and for
spectrum increments.

**X12. Local capacity corridors.** F2_k(11) <= 12 is tight (actual 11), but
the margin rho - 2 sum 1/q dies two-three gears above ANY base (vacuous at
y = 17 for base {5,7}, at y = 31 for base {5..17}) - Wall I in local form.

**X13. Tier B.** Lifting 35 -> 385 -> ... -> 1616615 adds exactly zero
exclusions anywhere tier A did not (feasibility lifts structurally). B is not
a tier; the hierarchy is A vs C only.

**X14. "Padded links need a common object".** Padded gaps are RARE
(0.001-0.023% of gaps, mid-tail: q'/mean ~ Cy/log^2 y -> inf), not common;
and the literal cap does NOT cover padded chains (Lateral withdrawal) - no
structural cap on padded count, only budget arithmetic + onset gate.

**X15. "FS < F - q'/6 at padded occurrences" as structure.** It was a
requirement, not a derived fact; measured padded FS/F roughly doubles
(0.32, 0.32, 0.42, then 0.67 at 31->37). The anomaly does not bound itself.

**X16. Both-maximal exclusion closes steps for (D).** The binding flank pairs
are mid-size, never maximal (R27) - the exclusion excludes a configuration
that never binds. Correct, kernel-worthy, off-target; do not extend further
for (D).

**X17. Spectrum flatness (at fuel depth).** F_{k_max+1} - F <= q' is FALSE at
29->31 (F_5 - F = 42 vs q' = 31; true incr 15; lossiness x1.4-2.8) - the
qualifying restriction is load-bearing. Raw flatness fails 5 of 15
machine-depth pairs (j = 5, 6 at machines 23, 29; j = 6 at 19), exactly where
the suppression term is largest (repaired by R31).

**X18. "FS <= F".** False: FS/F = 1.09 (13->17) and 1.12 (29->31).

**X19. Span-monotone envelope as a machine law.** The variable was wrong:
maxflank follows ln(occ(w)), not span (R33). Correct observation, wrong
variable; Mechanic's refutation accepted.

**X20. The uncorrected exposure bound (conditioning error).** Omitting the
1/rho (per-window vs per-slot) made the bound appear to clear (D) word-free;
corrected it falls short x2-x29 at the constrained cases. The gap is exactly
the dropped "no opening strictly between" renewal factor.

**X21. The anti-correlation law as the needed input.** Over-specified:
independence alone clears every constrained case by x170-x201,381 (R32). The
needed fact is only "no > ~170x positive correlation"; the measured law
(adjacency-only deficit) is far stronger than required.

**X22. Round-13 tier-A flank test / round-12 count.** First tier-A pass
conflated left flank with gR = 1 (manufactured false exclusions); "0 of 17
word-step pairs" should be 16 pairs. Corrected figures stand in R22/R27.

**X23. "Arithmetic luck" as an endpoint.** The max-window's failure IS
plausibly luck (R31 test 1), but the rate p_j is structural
(anti-correlated); building the window-profile object turned the luck reading
into the suppression law. Do not stop at "luck" again.

---

## 4. Open questions and the live target

**LIVE TARGET - part (D), the sole open input of the tolerance route.**
Sufficient for twin infinitude via R14 (alpha = 3) + R21 + R20 + R23. Working
forms:

1. Flank form: FS_max(w; M) <= F(M) + q' - span(w) for every compatible
   qualifying word w (<= 6 words per step, pinned addresses).
2. Suppression-corrected flatness (heuristic lambda):
   F_j(M) - F(M) <= q' + lambda*(j-2)*L for every j >= 2 (j = 2 is lemma 1;
   holds 15/15 measured; deeper j are easier).
3. Occurrence form: span(w) + lambda*ln(occ(w)) <= F + q'.

**Named next construct: THE RENEWAL FACTOR** - a closed-form lower bound on
P(no opening strictly between | both endpoints exposed) at separation v: the
entire remaining gap between the rigorous exposure bound (R32, kernel-ready)
and sufficiency - worth exactly x2-x29 where needed, a one-lag object that
never meets the disjunction obstruction. Needs Mechanic's conditional
opening-density profile at machines 23, 29, 31 for v in the qualifying set;
the multi-lag exposure bound (c_q product + 1/rho) goes to Formalist.

Other open items:

* O1. F(2,53): search running, >= 420; alpha = 2.5 demands <= 486, alpha = 3
  demands <= 513. Decisive measurement (review 7a).
* O2. Padding intermittency: confirm 37->41 and 41->43 winners are literal
  (corpus increments 0.220q, 0.837q suggest yes); k_win vs k_max at machines
  31, 37, 41; F_4 - F vs q' there (direct (D-b) test).
* O3. Uniformity in y of pinned addresses (drift of strata classes mod 385;
  Lateral's question) - the class tier needs mod 5005 to stay sharp at scale.
* O4. Census falsification asserts: literal chain longer than
  litcap(q' mod 210); any chain with k > T(M, 2u') + 1; literal k = 5,6 at a
  non-cap-6 gear; any realized padded interior >= q' (flag); any step where
  max over compatible words of span + FS_max != F(M+q'); k = 7+ anywhere.
* O5. Machine lead from descent scoping: thinnest layer bands occur exactly
  at twin endpoints - self-reference at the binding case, uninterrogated.
* O6. The review's multiplicative tail bound (N(L) <= P exp(-cL/y), c > 6):
  the genuinely open sieve-side middle ground - CUM neither implies nor
  needs it.

**Standing rules / lessons earned:**

* Record every route's exact limiting event; name unproven inputs, never
  assume them (Hensley-Richards, Reduction A, 0.525 floor: named-not-used).
* Quote the route at alpha = 3, not 2.5 (the 2.7% margin was an artifact).
* Distinguish requirements from derived facts (X15); check conditioning
  per-window vs per-slot (X20) - both caught only by adversarial re-testing;
  corrections recorded up front in the round found.
* Flag trend observations as such, never promote to laws; "arithmetic luck"
  is a prompt to build the object, not a stopping point (X23).
* Tier A is the only corridor tier worth formalising (X13); do not extend the
  both-maximal exclusion for (D) (X16).
* The two attack surfaces (sieve/prime) transfer at zero cost and zero gain
  (R4); separating P(t) from its floor is already a twin theorem (R9).

---

## 5. Reproduction pointers

Scripts (research/), by result:

* constructor_ledger.py - R1-R3, R5 censuses (y = 13, 23, 47).
* double_onset.py - R7 onset law, L* = 27129, 442-window census.
* cumulative_margin.py - R4/R5 full-window margins E(y), y = 47..5003.
* x_consistency.py - R9 (uses Lateral's split_gap_law.py closed forms).
* compression_bound.py, compression_zone.py - R10 moments + inversion zone.
* zone_fate.py - R11 ladder to 10^7, LP ceilings R13.
* multiplicative_route.py - R14 tolerance theorem, [53, 10^6] verification.
* topgap_endpoint_law.py - R16 corridor laws, R17 record censuses (X11, X12).
* strata_adjacency.py - R18; merge_census.py - R19 (23->29 streamed, 1.078e9).
* fuel_bound.py - R20 literal cap (48-class check).
* word_ceiling.py, flank_bound.py - R21 identity, 6/6 verification.
* flank_tierA.py, flank_tierA_fix.py - R22 (the _fix corrects X22).
* padded_bound.py - R23/R24 padding arithmetic, gear-37 anomaly.
* flank_pairs.py - R25/R27 per-word margins at alpha = 3.
* window_profile.py, suppression_law.py - R31.
* anticorr_law.py - R32/R33 (exposure bound, R(lag), occurrence form).

Data (research/data/): prefix_census.csv (Mechanic, R5), fuel_census.csv
(F2(29) = 55, F2(31) = 68, N_k, k_max by step), gap_pair_joint.csv +
gap_pair_hist.csv (the p_j object, R33), multiplicity_summary.csv (R13),
qspec_table.csv (qspec41, R33). Lateral's suites: split_gap_law.py,
topgap_corridor.py (Lateral owns that name; Constructor's is
topgap_endpoint_law.py).

Anchors used throughout: requirement F(2,y) < (y^2 - y)/2; F_k(M+q') chain
11/18/25/34/43; F2_k(2,y) = 33, 48, 75, 93, 117.

---

## Round 20 append (transfer-matrix directive): R35-R39, the exact criterion, and the renewal ladder

**R35 (operator frame + nilpotency identities).** On C^{Z_P}: S = slot shift,
D = exposure projector = tensor product over gears of D_q (CRT), B = I - D.
Gap operator G_v = D(SB)^{v-1}SD; every census quantity is a matrix element
(N(v) = 1'G_v 1; joint N_j(u,v) = 1'G_u R^{j-1} G_v 1 with R = sum_v G_v = the
successor permutation on openings, one |E|-cycle). Exact identities, verified
by direct operator iteration at machines 11-19 (tm_nilpotency.py):
F(M) = nilpotency index of BS (largest gap = longest blocked run + 1); the
qualifying-gap partial map A_V (V = residue-qualifying values for q') is
NILPOTENT with index = deepest qualifying run + 1 - indices 2,2,2,3,3,4,4 at
machines 11..31, so the fuel cap is a nilpotency statement (k_max <=
index(A_V); machine 29's k=4 event sits exactly at index 4). HONEST BOUNDARY
on the directive's spectral hope: the exact frame has NO spectral gap - R is a
permutation, eigenvalues roots of unity; decorrelation is an AGGREGATION
phenomenon, not mixing of the exact dynamics.

**R36 (aggregated transfer matrix: Markov closure FAILS, with structure).**
T[u,v] = P(next gap = v | gap = u) built from Mechanic's exact full-period
pair census (machines 11-31; marginals match ghist to the 1-gap seam). The
one-step chain OVER-predicts deep qualifying runs by growing factors:
residue set V - predicted/exact = 391/8 = x49 at machine 29 depth 3,
2242/508 = x4.4 at machine 31 depth 3; size floors at machine 29 -
x4.4 / x12.6 / x40 at depths 3/4/5 (machine 31: x2.5 / x4.3 / x27).
Equivalently per-link conditionals fall geometrically (machine 29:
P(next qual | 1 prev) = 5.5e-3, P(| 2 prev) = 1.8e-4 - each link ~30x more
suppressed than the last). NO fixed-order transfer matrix on gap values can
be the proof object - the memory is longer than any fixed lag. Spectral
constants measured anyway: rho(T_VV)/p_1V = 0.65 / 0.039 / 0.20 / 0.24 at
machines 19/23/29/31 (the Markov FLOOR of the deficit; reality is deeper);
full-chain |lambda_2| = 0.55-0.66, stable across machines 11-31. The R33
lag-2 rebound is PARTLY Markov (chain predicts 1.27-1.41 vs measured
1.53-1.90 at 23/29) and is GONE at machine 31 (obs R(2) = 0.71 < 1 vs
predicted 1.36) - a regime change at padding onset. Lag decay: measured
qualifying autocorrelation dips at lag 3 (0.31-0.51) and recovers by lag 5;
the chain predicts recovery by lag 2 - more memory evidence.

**R37 (tropical/max-plus side).** F_j <= longest j-node path in the
pair-support graph (edges = value pairs occurring adjacently): exact at j=2
by construction, lossy from j=3 (x1.17-1.54, worsening with j) - the pair
table does not pin the deep spectra. The V-interior subgraph is ACYCLIC at
machines 11-17 (there the pair table alone PROVES the qualifying depth cap)
but has cycles from machine 19 on while the realized depth still caps at
2-3 - THE DEPTH CAP IS A >= 3-POINT PHENOMENON from machine 19 onward: no
2-point census (pair table, corridor law, c_q(g)) can certify it. Max cycle
means (full / V-graph): 5.5/- , 8/-, 12.5/-, 15.5/11.5, 19.5/10, 27.5/15.5,
34/31 at machines 11..31.

**R38 (THE RENEWAL LADDER - rigorous rate bounds; docs/novel/renewal-ladder.md).**
For a qualifying tuple (v_1..v_m) with opening offsets X and ANY chosen set Y
of interior offsets: run event subset of {X exposed, Y blocked}, and
#W'(X,Y) = sum_{T subset Y} (-1)^|T| prod_q c_q(X u T) - exact CRT closed
form, no period scan. run_m <= sum over tuples of #W'; nested Y (bisection
order, s points per gap) gives a monotone ladder from R32's exposure bound
(s=0) toward exact. Asserted >= exact censuses everywhere; every #W' >= 0.
RESULT: the ladder CLEARS the (D) requirement at every constrained case,
including both R32 failures - machine 23 j=6: requirement 1.5e-4, exposure
bound was short x28.8, ladder s=5 gives 5.0e-7 (clears x300); machine 29
j=5: requirement 1.8e-2, was short x2.0, ladder 2.0e-4 (clears x91); machine
29 j=6: 1.4e-6 vs 2.8e-3 (x2000). First joint-gap bounds beyond scan reach:
machine 37 (period 1.24e12): p_5 <= 3.4e-2, p_6 <= 9.8e-3. HONEST LIMITS:
tightness above exact degrades with machine size (x40 at 29 m=2 up to
x1.8e5 at 31 m=3 - fixed points per gap cover a shrinking fraction of
growing gaps); no zero certificate reached (smallest surviving total 4, at
machine 23 m=4 where truth is 0) - the 2^|Y| IE cost bars Y = all interiors.

**R39 (the EXACT QUALMAX CRITERION; new machine-31 data).** New full-period
censuses with cyclic seam exact (tm_resid_runs.py; machine 31 = 3.34e10
slots): machine 31 F_j = 58, 68, 85, 90, 92, 97 (j=1..6 - the spectra the
envelope job never delivered), p_1V = 0.018445, qualifying runs
502,708 / 508 / 0 at depths 2/3/4 (deficits x4.2 / x77 / exact zero). The 8
depth-3 runs at machine 29 enumerated: all permutations of {10,10,21}, span
41, window sums 47-55 - the machine's complete k=4 fuel inventory
(tm_deepruns.py). THE CRITERION: by the merge law every new gap is a window
sum with residue-qualifying interiors, so EXACTLY

    F(M+q') <= max( F2(M), max_{j>=3} qualmax_j(M; q') ),   and
    (D) at alpha = 3  follows from  max(F2, max_j qualmax_j) <= F + q'.

Measured at all seven steps 11->13 .. 31->37 (tm_qualmax_check.py): the
criterion HOLDS 7/7 with margins 0.52-0.69 q' at the six literal steps and
0.19 q' at the padded step 31->37; and the criterion value EQUALS F(M+q')
at 6 of 7 steps (slack 2 at 23->29 only). This is R31's suppression-
corrected flatness with the heuristic stripped: no lambda, no L, no order
statistics - three exact census quantities. It subsumes forms 1-3 of the
live target at measured steps; the suppression law remains the asymptotic
reading. (Mechanic's Q_j is the same object at the weaker size threshold;
its margin collapse to 0.10-0.11 q' vs 0.19 q' here says the residue
condition retains real margin the size floor loses.)

**Round-20 negatives, recorded:** (i) Markov/spectral closure of p_j fails
at every order tested - the directive's "deficit as spectral gap" is FALSE
in the aggregated chain and VACUOUS in the exact frame (R35/R36); what
survives is the nilpotency reading and the ladder. (ii) The pair-support
tropical bound cannot see depth caps from machine 19 on (R37). (iii) The
ladder cannot zero-certify at depth (R38) - rate bounds only.

**Named next constructs (and why not built this round):**
* THE EXACT PATTERN COUNTER - #(X exposed, ALL interiors blocked) cheaper
  than 2^|Y| IE: per-gear transfer DP or Mechanic's COV(M) CRT machinery.
  Needed for zero certificates (qualmax_j = 0 / Q_j = 0 without scan). Not
  built: the 2^|Y| barrier was only quantified late in the round.
* THE FLANKED LADDER - extend the tuple by (g_L, g_R) and certify
  span + g_L + g_R > F + q' combinations to zero: the DIRECT rigorous route
  to (D) per step. Blocked behind the exact counter (same certification
  hardness); the enumeration is small since only large flank pairs matter.
* lambda-free requirement at scale: R39's criterion needs qualmax bounds at
  unscannable machines; the flanked ladder is exactly that.

**Reproduction (round 20):** tm_resid_runs.py (exact residue-qualifying run
censuses + qualmax + spectra; data/tm_resid_runs.csv), tm_deepruns.py (deep
fuel inventories), tm_transfer.py (R36: Markov closure, Perron, lag tables),
tm_tropical.py (R37), tm_renewal_bound.py (R38 ladder + assertions),
tm_nilpotency.py (R35), tm_qualmax_check.py (R39). All checks asserted; all
censuses full-period with the cyclic seam stitched.

---

## Round 21 append (one algebra, not infinite rules): R40-R44

**R40 (THE TWO-TEETH KILL SPACING LAW, reproduced and proved;
docs/novel/two-teeth-kill-spacing.md).** The live 2026-08-24 finding, now
theorems T1-T5 asserted on EVERY window of every full joint period P*q',
steps 11->13 .. 29->31 (research/kill_spacing.py; joint period = q' copies
of the old opening sequence, boundary and cyclic seam handled exactly):
T1 {2c, -2c} mod q' = {2u', q'-2u'} - the tooth-difference residues ARE the
literal letters; T2 interior spacings = 0 or +-2c mod q'; T3 nonzero-class
signs STRICTLY ALTERNATE (padded spacings transparent; |#a - #b| <= 1 per
window); T4 minimum 2u'; T5 FUEL-SPAN LAW k <= 1 + span/(2u')
<= 1 + 3 span/(q'-1) - the fuel cap as closed-form span arithmetic, no
census. MEASURED M1: every realized spacing VALUE is exactly 2u', q'-2u',
or q' - never 2u'+q', 2q', ... which the residue classes admit - at all six
steps; at 29->31 (joint period 3.34e10, 237 s vectorised): spacings
10: 7,815,766 / 21: 205,068 / 31: 4,180 and nothing else; windows
421,392,436 with k = 1/2/3/4 : 413,380,422 / 7,999,018 / 12,992 / 4, and
T3 + max_span 41 FORCE all four k=4 windows to be word (10,21,10) -
mechanic's four addresses, re-derived. Window counts reproduce R19's chain
censuses exactly PLUS one seam window at 13->17 (2,898 vs 2,897) and
23->29 (15,660,528 vs 15,660,527): R19's linear scans did not stitch the
cyclic seam chain.

**R41 (NILPOTENCY ADDITIVITY = THE SUM SPLITTING; the counting boundary;
research/nilpotency_additivity.py, log data/nilpotency_additivity.log).**
New operator form (one line of algebra, dense-exact at {5}+7 and {5,7}+11,
operational by vector iteration at all four steps 11->13 .. 19->23):

    B_new S_new = (B_M S_M) (x) S'  +  (E_M S_M) (x) (B' S')

- adding gear q' is an exact Kronecker RECURSION: new blocked walk = old
blocked walk (x) shift + old renewal step (x) q'-kill; F(M+q') = nilpotency
index of the SUM (old factor index F(M), kill factor index 2). The operator
is a masked permutation (entries 0/1), so the m-th power expands over
binary kill-words with NO cancellation, and CRT separates the tensor
factors: (BS_new)^m != 0 iff SOME kill pattern is (left) an old-machine
pattern event AND (right) mod-q' realizable - and right-realizability is
EXACTLY the spacing law T2/T3 (verified: right-factor 0/1 matrices nonzero
on all spacing-law kill patterns, zero on all violating ones, q' = 13, 23).
The merge law, the word grammar (A), the padding count (C), and the
fuel-span cap are corollaries of this one identity plus T1-T5.
THE COUNTING BOUNDARY (honest negative, sharp): NO function of the marginal
data (F, kill index 2, spacing law, litcap) bounds the index of the sum.
The 2-point relaxation - every adjacent kill pair individually realizable -
is satisfied by the INFINITE alternating word from 19->23 on (adjacent
pairs (8,15) x31 and (15,8) x31 at machine 19) while the true chain stops
(consecutive triples (8,15,8), (15,8,15): ZERO). And the truncation arity
GROWS: 3-point at machines 19/23 (run3 = 0), 4-point at 29 (run3 = 8,
run4 = 0) - no fixed-arity joint law suffices (R36's growing memory,
operator side; matches R37's tropical boundary). delta <= q' is decided by
which spacing-compatible kill patterns the old machine realizes - the
anti-correlation clause (D), nothing else. FUEL-AS-BRIDGES, measured: the
per-k record's largest bridged old gap FALLS with k (19->23: 0.84F at k=1,
0.80F at k=2, 0.60F at k=3; the k=3 record window is the live session's
[4,8,15,7] = 34) - par trading in bridge form.

**R42 (THE CORRIDOR-PHASE TRANSFER CHAIN - the state-space answer;
research/tm_corridor_phase.py, logs data/tm_corrphase_*.log).** Rebuilt the
transfer chain with state = left-endpoint corridor phase and tested three
nested models against full-period exact censuses taken in the same pass
(machines 13, 19, 23, 29; --mod 35 and 385; every model built from exact
counts; V-run counts cross-checked EXACT vs tm_resid_runs.csv). The answer
at m29 depth-3 V-runs (the x1400 independence deficit; R36's baseline
rebuilt in-pass gives x48.8):

    state:        indep   value   ph35   (ph35,val)   ph385   (ph385,val)
    pred/exact:   x1400   x48.8   x3.6     x1.9        x2.3     x0.86

THE ANTI-CORRELATION'S CARRIER IS SMALL-GEAR PHASE: (phase mod 385, last
value) predicts the (D)-relevant deep runs within 15%. Size-floor runs
likewise collapse (m29 depths 3/4/5: value x4.4/x12.6/x40 -> hybrid-385
x1.12/x0.97/x2.2). THE LAG WAVE: the phase chain reproduces the
deficit-at-1-3/excess-at-4-7 oscillation with correct peak/trough lags at
every machine (amplitude damped x2-4 at mod 35, near-exact at mod 385:
m29 V-lags 2..7 exact 1.53/0.81/1.05/1.03/1.13/1.28 vs
1.56/0.83/0.86/0.99/1.16/1.26); the value chain is flat 1.00 from lag 3.
NEW SPECTRAL OBJECT: the phase chain's lambda_2 is COMPLEX -
|l2| = 0.963/0.912/0.886 at m13/19/23 with arg 34/43/46 deg, so period
360/arg = 7.8-8.4 lags and damping |l2| per lag: THE CORRIDOR RESONANCE IS
THIS EIGENVALUE (arg ~ 2 pi mean_gap/35); distinct from the value chain's
real lambda_2 = -0.55 (Mechanic's phi/3 object). HONEST RESIDUALS: every
phase chain over-predicts lag-1 adjacency (the exclusion there is
teeth-value, not phase); size-floor depths 5-6 keep memory beyond
(mod 385, value) (x2.2-3.0); machine 31 unmeasured (memory-sweep casualty
at 31%, deliberately not relaunched). corridor-resonance.md carries the
round-21 addendum.

**R43 (THE EXACT PATTERN COUNTER, from Lateral's cross-lane offer;
research/qualrun_zerocert.py, log data/qualrun_zerocert.log).** Lateral's
hereditary-zero pruned IE (psd_bite.bonferroni_runs) adapted to pattern
events with required-open seeding: #(X exposed, Y blocked) with Y = ALL
interiors - R38's named blocker (the 2^|Y| cost) DELIVERED for spans up to
~75. Key structural fact: nonzero subsets are downward-closed (N monotone),
so the DFS cost = |{T : N(T) > 0}| independent of order. VALIDATED exact
against every census row: m19 run2 = 234, run3 = run4 = 0 (zero
certificates); m23, m29 all rows - run3(29) = 8 reproduced by pure CRT
arithmetic in 14 s (3.0e6 nodes) - the 8 needles of a 1.08e9-slot period,
no scan; m31 (period 3.34e10) run2 = 502,708 EXACT MATCH (3.4e8 nodes,
1611 s). Partial run3(31): the six nonzero tuples found -
(12,12,25) 139, (12,25,12) 188, (12,25,25) 7 + mirrors 139/28/7 - sum 508
= the full census value; the heavy padded tuples ((25,25,49) etc.) exceed
the 3e8-node budget - cost grows ~exponentially in span (98M nodes at span
74, budget dead at span 99). HONEST NEGATIVE: the memoized alternating
recursion f(i, masks) = f(i+1, masks) - f(i+1, masks & rot_i) does NOT
beat the DFS - MORE states than DFS nodes on the same pattern (1.58M vs
1.37M at m19 (15,23,23)) and the span-74 m31 pattern unfinished at 10 min
where the DFS took 445 s (research/test_memo2.py + log) - mask states
barely coincide, so the sharing the memo bets on is absent. Machine 37
values not reached this round; the per-span cost curve is measured, and
Mechanic's COV-SAT (605 s/instance probes) is the named supplier there.

**R44 (R39 AT 37->41 - the criterion's first beyond-scan step, DECIDED
in-round).** Mechanic posted early per the brief: F_3(37) = 97 EXACT
(SAT witness k = 990,209,189,833, gaps [37, 23, 37] - a palindrome flanked
by the top gear's own value; UNSAT at every S in [98, 178];
178 = F2 + F is the a-priori cap). Budget F(37) + 41 = 129: the j=3 clause
qualmax_3 <= F_3 = 97 holds with margin 32 = 0.78 q' - the criterion's
margin is RESTORED at the litcap-2 gear (the Q_j collapse stays a litcap-6
phenomenon; next real test q' = 53). My j >= 4 concern (qualmax_4 had no
upper bound on the F_3-only route) is discharged by Mechanic's independent
confirmation from the r20 padding37 full-period census: max_j qualmax_j
(37;41) = 91 = F(41) EXACTLY, so the criterion value is max(90, 91) = 91
<= 129 - the EIGHTH measured step, EQUALITY criterion = F(M+q') at 7 of 8,
margin 38 = 0.93 q'. (D) at alpha = 3 at 37->41: DECIDED, both routes.

**Round-21 negatives, recorded:** (i) the counting boundary - additivity
has no marginal-arithmetic bound, and the truncation arity grows (R41);
(ii) phase chains cannot carry the lag-1 teeth exclusion, and size-floor
deep runs keep super-phase memory (R42); (iii) the memoized counter is not
faster (R43); (iv) the memory-pressure process sweep killed three of the
round's jobs (m31 phase census at 31%, the first kill_spacing 23/29 run,
the qualrun campaign mid-m31-run3) - the first was not relaunched, the
second was re-run vectorised (299 s -> 4 s at 23->29) and completed, the
third was closed deliberately with per-tuple data intact; nothing reported
rests on a partial scan.

**Reproduction (round 21):** kill_spacing.py (T1-T5 + M1; logs
data/kill_spacing_23.log, kill_spacing_23_29.log), nilpotency_additivity.py
(P1-P4; log data/nilpotency_additivity.log), tm_corridor_phase.py
(--mod 35/385; logs data/tm_corrphase_19_23.log, tm_corrphase_29_31.log
(m29 + dead m31 tail), tm_corrphase_23_mod385.log,
tm_corrphase_29_mod385.log), qualrun_zerocert.py (+ test_memo2.py for the
memo negative; logs data/qualrun_zerocert.log, test_memo2.log). All
censuses full-period seam-exact; model predictions labeled float.

---

## Round 22 append (the arity verdict + the arity-free generator): R45-R48

**R45 (THE ARITY LADDER - three arities separated, and R41's "growing arity"
CORRECTED; research/arity_ladder.py, research/arity_probe41.py, log
data/arity_probe41.log).** Round 21 reported "the truncation arity GROWS -
3-point at 19/23, 4-point at 29". That number was the RESIDUE arity, and a
residue-qualifying run is NOT a kill chain: the kernel-checked T3 alternation
(two-teeth-kill-spacing.md) forbids two consecutive letters of the same
nonzero class. Separating the three arities (all exact, full period):

    A_res  (nilpotency index of the residue-qualifying successor map)
    A_kill (residue + T3 alternation) = k_max, the fuel chain length
    A_relax(smallest m at which the m-point relaxation refutes the INFINITE
            alternating word ...a b a b... - R41's own boundary object)

    machine        11  13  17  19  23  29  31  37  41
    A_res           2   2   2   3   3   4   4   4   -
    A_kill = k_max  2   2   2   3   2   4   4   3  >=3
    A_relax         1   2   2   3   2   3   4   3   2
    litcap(q'%210)  2   2   2   4   3   4   6   2   4

A_res is monotone through m37; A_KILL AND A_RELAX ARE NOT. Both fall at m23
and again at m37, and A_relax falls to 2 at m41 - the earliest refutation
since m17. THE VERDICT: THE OPERATOR-RELEVANT ARITY NEITHER GROWS WITHOUT
BOUND NOR STABILISES - it is gear-arithmetic-valued, moving up and down with
the added prime rather than with the machine's size. Its literal part is
permanently capped: A_kill <= litcap(q' mod 210) at 7 of the 8 measured steps
and litcap <= 6 for every gear forever (R20 theorem). The one exception,
m37 (litcap 2, A_kill 3), is forced to use a PADDED link - and indeed all
1,579 killable 2-words at 37->41 are padded, since a 3-member literal chain
would break litcap 2. Only the padded component has no structural cap (only
R23's budget count ~F/q'). Litcap is an upper envelope, not a predictor:
at m41 litcap = 4 while the literal 2-word count is exactly 0. So R41's
"no fixed-arity rule exists" survives, but for a different and better reason:
not because the arity diverges, but because it is an arithmetic function of
q' with an uncapped padded component - which is precisely why the vehicle has
to be arity-free rather than a rule for each layer.

Supporting exact results:
* T3 CROSS-CHECK (new, five machines): applying the alternation filter to the
  residue census reproduces the FUEL census exactly - killable run_j == N_{j+1}
  at m19 (62 = 31+31, the (8,15)/(15,8) pairs; the 172 (8,8) pairs are
  T3-dead), m23 (0 of 288 - all (10,10)), m29 (4 = the (10,21,10) windows of
  8 residue runs), m31 (216 = 188 (12,25,12) + 28 (25,12,25) of 508), m37
  (0 of 8 - the only realized depth-3 residue word is (14,41,14), two class-a
  letters with a transparent padded link between them, so T3-dead). Two
  independently-produced censuses agree through one residue law.
* OVERLAP LEMMA (factor closure), proved and applied: if every realized
  depth-m word is known, run_{m+1} = 0 unless some pair w, w' of realized
  m-words has w[1:] == w'[:-1]. At m37 the realized depth-3 set is the single
  word (14,41,14), which does not overlap itself, so run_4^res(37) = 0 with NO
  further computation - A_res(37) = 4 established from Mechanic's exhaustive
  word census alone. (Inconclusive at m31, where the six realized words do
  overlap; there run_4 = 0 is the direct full-period census.)
* SPAN CEILING (proved, asserted at all eight machines): every qualifying gap
  is >= 2u' (T4) and j consecutive gaps sum to <= F_j, so
  A_res <= min{ j : F_j(M) < 2u' j }. True but loose by ~2x - the arity is
  NOT span-limited at any measured machine; it is limited by joint
  realizability, which is (D)'s content.
* MACHINE 41 (q' = 43), first arity data beyond any census, by exact CRT
  pattern counts (period 5.07e13, no scan; log data/arity_probe41.log):
  the two LITERAL 2-words (14,29) and (29,14) are BOTH ZERO, so
  A_relax(41) = 2 - the 2-point relaxation already refutes the infinite
  alternating word, the earliest refutation at any machine past m17. The
  nonzero killable 2-words are all PADDED: (14,43) = (43,14) = 170,203,
  (29,43) = (43,29) = 228 (total 340,862 over 11 words of span <= 90, one
  word (43,43) undecided at the 3e8-node budget), so A_kill(41) >= 3. All
  eight killable 3-words of span <= 90 are ZERO, including (14,29,14) and
  (29,14,29). HONEST LIMIT: the 3-word enumeration is cut at span 90, not at
  F_3(41) (which nobody has), so A_kill(41) is recorded as ">= 3 and no
  depth-3 chain of span <= 90", not as an exact value. Note litcap(43) = 4
  while the literal 2-word count is 0: LITCAP IS A PROVED CAP ON THE LITERAL
  PART, NOT A PREDICTOR of the realized arity.
* Conditional decay of the residue ladder (exact): per-link conditionals
  m19 0.0311/0.0199; m23 0.0307/0.0012; m29 0.0374/0.0055/0.00018;
  m31 0.0184/0.0044/0.0010 - falling, and at m31 stably by x4.3 per link.

**R46 (THE ARITY-FREE GENERATOR: F(M+q') IS A KLEENE STAR;
research/kleene_generator.py, research/kleene_stream.py,
docs/novel/kleene-generator.md).** The answer to the round's decisive
question - IS NILPOTENCY ADDITIVITY ARITY-FREE? - is YES, and here is the
form. On states (opening i, tooth s in {+,-}) define the max-plus matrix and
flank vectors

    K[(i,s),(i+1,s')] = d_i  if d_i mod q' in {0, a, b} and s -> s' is the T3
                             transition of d_i's class; else -inf
    L(i) = d_{i-1},   R(i,s) = d_i

THEOREM (identity, not a bound): F(M + q') = L^T (x) K* (x) R, where
K* = (+)_{m>=0} K^m is the Kleene star. Proof both ways from the merge law +
T2/T3 (<=) and CRT choice of the killing copy (>=); written out in the novel
doc. In R41's recursion the second summand (E_M S_M) (x) (B' S') is nilpotent
of INDEX 2 and the first is nilpotent of index F(M); the index of the sum is
not a function of those two (round 21's counting boundary) but IS exactly this
star. K is nilpotent with index = A_kill, so the star is a finite sum - but
the statement never names a depth. Its m-th layer is exactly qualmax_{m+2}:
one algebra generates every layer of R39's ladder.

COROLLARY (tropical dual certificate - (D) with the depth quantifier removed):
(D) at alpha = 3 holds iff there exists h on states with
    (C1) h(i,s) >= d_i
    (C2) h(i,s) >= d_i + h(i+1,s') for every legal qualifying transition
    (C3) d_{i-1} + h(i,s) <= F(M) + q'
Necessity h = K* (x) R; sufficiency because any super-solution dominates the
star. Every clause is a ONE-STEP, ONE-OPENING inequality - the first form of
(D) that is not an infinite family. This is max-plus LP duality for the
longest-path problem F(M+q') actually is (and it is the tropical face of the
covering-LP-duality thread the manager flagged as untested).

VERIFIED EXACT, full period, at every scannable consecutive step (dense and
streamed implementations agree digit for digit at m19 and m23):

    step        index(K)  L(x)K*(x)R = F(M+q')  F+q'   margin      layers
    11 -> 13        2            11              20   +9  0.69q'  [11, 8]
    13 -> 17        2            18              28  +10  0.59q'  [16, 18]
    17 -> 19        2            25              37  +12  0.63q'  [25, 25]
    19 -> 23        3            34              48  +14  0.61q'  [31, 33, 34]
    23 -> 29        2            43              63  +20  0.69q'  [39, 43]
    29 -> 31        4            58              74  +16  0.52q'  [55, 58, 55, 55]

index(K) == A_kill at every step, and h is always the LEAST super-solution
(every state tight) - the certificate is exactly saturated, so nothing is
being given away by using it.  The 29 -> 31 layer vector is the informative
one: the winner sits at ONE link (58) and the deeper layers fall back to 55,
so the deepest chain is NOT the maximiser - par trading, in Kleene form.

**R47 (THE FINITE-STATE CERTIFICATE - how big must the state be? Same
scripts).** The certificate becomes machine-free only if h can be replaced by
a function of a BOUNDED local state. Replacing the opening by a class and
taking edge weights = max realised gap gives a SOUND class-level max-plus
system, so its closure is a genuine upper bound on F(M+q'). Measured:

    step        value only        (ph 35, val)  (ph 385, val)  (ph 5005, val)  budget
    11 -> 13    11 certifies      11            11             11               20
    13 -> 17    21 certifies      21            20             18               28
    17 -> 19    30 certifies      28            28             25               37
    19 -> 23    CYCLIC (vacuous)  45 certifies  42             34               48
    23 -> 29    60 certifies      60            45             43               63
    29 -> 31    CYCLIC (vacuous)  99 FAILS +25  99 FAILS +25   91 FAILS +17     74

TWO FINDINGS, one positive and one negative, and the negative is the bigger.
POSITIVE: THE VALUE-ONLY ABSTRACTION IS CYCLIC EXACTLY WHERE A_relax >= 3
(machines 19 and 29 here, and by the same criterion 31 and 37) - R41's
counting boundary is precisely "the abstract operator loses nilpotency", and
a non-nilpotent tropical operator bounds nothing at all; adding the corridor
phase mod 35 restores nilpotency at both machines, and at 19 -> 23 it also
CERTIFIES (D) (45 <= 48).  R42's carrier does proof work there, not
statistics.
NEGATIVE (and it is decisive for the tactic): AT 29 -> 31 NO BOUNDED STATE
TESTED CERTIFIES - mod 35, 385 and 5005 all restore nilpotency but overshoot
the budget by +25, +25, +17 (bounds 99, 99, 91 against 74; exact 58).  So the
corridor-phase abstraction does NOT scale as it stands: the certificate is
arity-free but not yet MACHINE-free.  The named next construct is a tighter
abstraction - candidates, in the order I would try them: (a) edge weights
conditioned on the destination class rather than max-over-source (the current
weight max{d_i} is the crudest sound choice), (b) two gaps of history rather
than one, (c) the flank vector L abstracted separately from the chain state,
since the overshoot at m29 is 99 = a long chain paired with a large flank
that never co-occur.  Also honest: the bound is loose even where it
certifies (45 vs exact 34 at m19), and the state needed is not monotone in
the machine (value-only suffices at m23, where A_kill = 2, but is vacuous at
m19 and m29).

**R48 (CLOSED-FORM lambda_2 OF THE CORRIDOR-PHASE CHAIN, and Lateral's
formula adjudicated; research/lambda2_closed.py).** The phase chain adds the
next gap mod M, so under phase-value independence its transition matrix is the
CIRCULANT of the gap distribution and its eigenvalues are exactly that
distribution's Fourier coefficients:

    lambda_2 = phat(1) = sum_g P(gap = g) e(g / M)

- the gap distribution's characteristic function at the corridor frequency.
Exact full-period histograms give (machines 11/13/17/19/23):
|phat(1)| = 0.9612/0.9380/0.9128/0.8847/0.8576, arg = 29.1/34.2/38.3/42.1/45.4
deg. The true chain (empirical transition matrix restricted to its 15-state
support) gives 0.9849/0.9634/0.9396/0.9125/0.8859 at 29.3/34.4/38.7/42.8/46.3
deg - which REPRODUCES R42's measured 0.963/0.912/0.886 at 34/43/46 deg
exactly (asserted). So: the closed form nails the ARGUMENT (error 0.13-0.89
deg at every machine), hence the resonance period 360/arg, and understates the
MODULUS by a strikingly stable 0.0237/0.0253/0.0268/0.0278/0.0282 - a deficit
that is itself converging (~0.029), and that IS the phase-value correlation,
i.e. the corridor pinning. Cumulant form exp(i.theta.gbar - theta^2 var/2),
theta = 2 pi/35, reproduces phat(1) to 0.1-1.5%: LAMBDA_2 IS DETERMINED BY THE
MEAN GAP AND THE GAP VARIANCE ALONE, both closed-form CRT quantities (mean =
1/rho; variance via the blocked-run counts B(t) that Lateral's pruned IE
computes exactly - a clean cross-lane closure).
CROSS-LANE ADJUDICATION: Lateral's round-22 form lambda_j = rho w_j /
(1 - (1-rho) w_j), w_j = e(j/e), IS THE SAME OBJECT - it is exactly phat for a
GEOMETRIC gap distribution of density rho, evaluated at an e-th root of unity.
Against my exact chain values it errs by 0.0146/0.0225/0.0245 in modulus and
0.52/1.29/1.71 deg in argument at m13/19/23; my exact-histogram form errs by
0.025-0.028 in modulus and 0.22-0.89 deg in argument. Neither dominates: the
renewal instance is better in modulus, the exact-histogram instance better in
argument, and both are the same formula with a different gap law substituted.
LATERAL'S PRE-REGISTERED PREDICTION SETTLED IN-ROUND: they pre-registered
machine 29 mod 35 -> |lambda_2| = 0.862 +- 0.004, arg +49.2 +- 0.4 deg. My
exact full-period chain at m29 (2.147e8 gaps, streamed 35x35 transition
counts, cyclic seam stitched) gives |lambda_2| = 0.8617, arg +49.15 deg -
BOTH INSIDE THE PRE-REGISTERED BAND (log data/lambda2_29.log). Their own
closed formula evaluated at m29 gives 0.8366 / +47.09 deg (errors 0.025 /
2.06 deg) and my exact-histogram phat(1) gives 0.8335 / +48.09 deg (errors
0.028 / 1.06 deg), so the prediction they registered is sharper than either
raw closed form - the refinement behind it is worth extracting.

**Round-22 negatives and corrections, recorded:**
(i) SELF-CORRECTION of R41: "the truncation arity grows (3, 3, 4)" was
measured on the residue arity; on the operator-relevant (killable) arity the
sequence is 2,2,2,3,2,4,4,3 and it goes DOWN as well as up. The conclusion
R41 drew (no fixed-arity rule) survives; the reason changes.
(ii) The class-level (bounded-state) certificate is sound but LOSSY, and at
29 -> 31 EVERY tested bounded state (value; phase mod 35, 385, 5005 with
value) FAILS to certify - overshoot +25/+25/+17 against a budget of 74.  The
generator is arity-free; it is not yet machine-free, and corridor phase alone
does not make it so.
(iii) The span ceiling A_res <= min{j : F_j < 2u' j} is proved but loose by
~2x everywhere - span arithmetic alone does not explain the arity.
(iv) The dense machine-29 Kleene run was killed by memory starvation (2.5 GB
free of 15.6 GB, the rest held by Mechanic's jobs) and re-run as a segmented
streaming pass (kleene_stream.py, ~300 MB), which reproduces m19 and m23
digit for digit. Nothing filed rests on a partial pass.
(v) Machine 41's heavy padded 2-words and the depth-3 words of span > 86 were
not decided within the round's budget - A_kill(41) is therefore recorded as
">= 3", not as an exact value; the exact value needs F_3(41), which nobody
has. A_relax(41) = 2 IS exact (it only needs the two literal 2-words).

**Reproduction (round 22):** arity_ladder.py (three arities, T3 cross-check,
overlap lemma, span ceiling, litcap comparison - all asserted);
arity_probe41.py (log data/arity_probe41.log); kleene_generator.py (dense
identity + certificate + abstraction ladder, machines 11-23; log
data/kleene23.log); kleene_stream.py (segmented, machines 23 and 29; log
data/kleene_stream_23_29.log); lambda2_closed.py (closed form + Lateral's
form + R42 assertions; log data/lambda2_29.log). Novel doc:
docs/novel/kleene-generator.md.

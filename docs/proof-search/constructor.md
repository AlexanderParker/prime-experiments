# Constructor: cumulative findings (compacted, rounds 1-24)

Compacted 2026-08-29. Verbatim rounds 1-19 in `archive/constructor-full-r1-19.md` (19
rounds: 1-6 and 8-20; no round 7); verbatim rounds 20-24 in
`archive/constructor-full-r20-24.md`. Single cumulative statement: where a later round
superseded an earlier one only the final state is kept, with a one-line supersession note.

**Mandate:** proof of twin-prime infinitude by construction/contradiction from the proven
mechanical laws; Condition X (a twin-free window) is the contradiction target; every claim
reproduced by a named script with asserted identities.

---

## 1. Definitions (stated once)

* **Slot** k = pair (6k-1, 6k+1). **Gear** q (prime >= 5) **blocks** k iff k = +-c_q mod q,
  c_q = 6^{-1} mod q. **Window** W(y) = {k : y < 6k-1, 6k+1 < y^2}; N = |W|; P = prime
  members of the 2N members; C = 2N - P. **Root kill**: composite m attributed to
  lpf(m) <= sqrt(m) < y. **Horizon theorem**: top gear contributes nothing interior
  (R(top) = 0); primality above y = non-divisibility below.
* **Census** n0/n1/n2 = slots with 0/1/2 composite members (twin / fragile / double);
  D(t) = doubles to prefix t. **Condition X**: some W(y) has n0 = 0.
* **Machine** M = gears {5..y}; **openings** = unblocked slots; **gap word** = gaps between
  consecutive openings. F(M) = max gap (2-dim Jacobsthal object); F2(M) = max adjacent-pair
  sum; **spectrum** F_j(M) = max sum of j consecutive gaps. Adjacent/halved frame = 3 x
  k-frame; F(2,y) = adjacent-frame chain value.
* **Step** M -> M+q' (q' = next prime): incr = F(M+q') - F(M); excess = F(M+q') - F2(M).
  **k-chain** = k consecutive M-openings all deleted by q'; its merge spans k+1 consecutive
  gaps of M.
* u' = round(q'/6); qualifying interiors V(q') = {0 or +-2c mod q'} (teeth {c, q'-c});
  literal letters a = 2u', b = q'-2u'. **Literal chain** = interiors exactly alternating
  {a,b}. **Padded link** = interior = 0 mod q', so >= q'. litcap(q' mod 210) = literal-chain
  member cap (2, 3, 4 or 6).
* **Word** w = a chain's interior-gap sequence; span(w) = sum; FS(w) = sum of the two
  flanking gaps at an occurrence; FS_max(w;M) = max over occurrences; occ(w) = occurrences
  per period; **compatible** = has a valid tooth start (from q' alone; firing is binary).
* E = the 15-residue exposed set mod 35 (gears 5,7); carrier S_m(w) = {r in E_m : every
  partial sum r + w_1..w_j in E_m}. **Corridor phase** = left-endpoint residue mod 35 /
  385 / 5005.
* m_k = omega_L(k)*omega_R(k); S1 = sum m_k; M2 = sum m_k^2; R(t) = (S1^2/M2)/(t-P(t)).
  p_1 = density of qualifying-size gaps; p_j = qualifying rate of j-windows; L = ln(1/p_1);
  lambda = exponential scale of the window-sum tail; rho = machine density prod(1 - 2/q).
* **qualmax_j(M;q')** = max sum of j consecutive gaps whose j-2 interiors are all
  residue-qualifying and T3-legal; **Q_j(M;v)** = same at the weaker size floor (interiors
  >= v); **Q^[J]** = the marked relaxation of Q_J one gear down. **Layer k** of a chain =
  k links = k+1 killed openings = a window of k+2 gaps (layer 3 = J=5).
* **A_res / A_kill / A_relax** (R45): nilpotency index of the residue-qualifying successor
  map; same with T3 alternation imposed (= k_max); smallest m at which the m-point
  relaxation refutes the infinite alternating word ...abab....
* **A_m(mod)** (R49): max-plus abstraction whose state is the last m-1 gap VALUES ending at
  the current opening, optionally corridor phase and tooth; an edge exists iff the encoded
  m-tuple of consecutive gaps is REALISED in the period and the middle gap's T3 transition
  is legal. **MF_m** (R52): same with "realised" weakened to "corridor-admissible, values
  in 1..F" - machine-free by construction.

---

## 2. Established results

### 2.1 Census / arithmetic layer

**R1-R3 (slot cap, supply partition, census identities).** No gear blocks both members of a
slot (else q | 2): <= 1 root-kill per slot per gear, a double's kills come from distinct
gears, multiplicity 0/1/2 only. Root attribution partitions composites exactly,
sum_q R(q) = C = 2N - P; R(q) = [y < q^2 < y^2] + #{r prime > q : y < qr < y^2} + H(q) (the
corpus formula pi(y^2/q) - pi(q) + 1 is the first two terms; H large only for gear 5).
Always n0+n1+n2 = N, n1+2n2 = C, P = n1+2n0, N-P = n2-n0. Under X every prefix is forced:
n1(t) = P(t), n2(t) = N(t)-P(t). Hence C1 (P <= N globally), C2 (every run I has
P(I) <= N(I); prefix margin n2-n0 >= 0), C3 (sum_q PT(q) = n1 = P - 2n0), C4 (prefix demands
met by the pi(sqrt(t)) - 2 gears active at depth t).

**R4 (run-condition equivalence).** Some run has P(I) > N(I) iff the window has a twin (a
twin slot is a run of excess +1). CUM ("every window has a run with P(I) > N(I)") is EXACTLY
Reduction A - lossless, gears dropped out, no strictly weaker ingredient exists. The
excess-run pattern (46 primes in [53,283]) is constellation-admissible: no congruence
obstruction to recurrence.

**R5 (computed censuses).** y=13: N=25, P=32, C=18, n0=9, C1 fails (+7). y=23: N=83, P=90,
C1 fails (+7). C1 bites only while prime density among +-1 mod 6 members exceeds 1/2, i.e.
y < ~e^6 ~ 403. y=47: N=359, P=313, C1 passes (C-N=+46) but C2 fails by 7 (run of 39 slots,
members 53..283, 46 primes). E(y) = max run excess 7,4,3,3,3,3,3 at y = 47,101,199,503,1009,
2003,5003 - flat 3, realised within ~700 of y (283, 1277-1303, 2657-2713, 5639-5659). First
double-composite slot in all of N is k = 20 ((119,121)); every slot k <= 19 has a prime.

**R6 (roots-of-unity law).** Slot k is hit by distinct gears q,q' iff 36k^2 = 1 mod qq'.
Trivial roots +-1 = same-member (semiprime) hits; nontrivial roots +-r,
r = CRT(+1 mod q, -1 mod q'), = cross-member hits. A slot is double iff 6k lands on a
nontrivial root of unity mod qq' for some active pair - so D is one fixed pinned subset of
the integers, computable by semiprime arithmetic with no primality tests. For twin-pair
gears (p, p+2) the nontrivial root is p+1.

**R7 (onset law + absolute cap).** L0(y) = lag to first double slot; n2 = 0 there
unconditionally, so under X the onset prefix is perfectly fragile. Via Montgomery-Vaughan
pi(x+H) - pi(x) < 2H/ln H: **L0(y) <= L* = 27129 for every y** (6L*+2 = 162776 > e^12 =
162755) - unconditional theorem of the programme. Measured (442 windows, 13 <= y <= 3163):
max L0 = 17 (y=13); L0 = 0 in 153/442; a twin precedes the first double in 132/442.

**R8 (descent / layer bands, scoped).** X at y forces every layer band (y'^2, y''^2) above y
twin-free; bands have length x^(1/2+o(1)) at height x (thinnest, at twin endpoints, exactly
4sqrt(x)+4). Input tower: T1 a prime in every band - OPEN (implied by Legendre/Cramer, NOT
by RH); T2 a pair with gap <= d in every band - proven localisation floor is exponent 0.525
(Alweiss-Luo 2018, from Baker-Harman-Pintz), need 1/2; T3 the parity step 246 -> 2 (no
partial result). Pair density is ample (Maynard-Tao, surplus x^(1/2)/polylog); localisation
and parity fail. Blocked at T1.

**R9 (X-consistency equation).** Unconditional: P(t) = t - D(t) + n0(t) at every prefix; X
iff P(t) = t - D(t) for all t. Satisfiable - the forced value is the unconditional pointwise
FLOOR of P, and every unconditional ceiling sits a parity factor above:
rho(t) = (t-D)/(2H/ln H) <= 1 provably, measured max 0.4687/0.4785/0.4828 at y = 101/211/503,
drifting to 1/2. **Any theorem separating P(t) from its floor at one t IS a twin-existence
theorem** (separation = n0(t)). Supply: the only unconditionally guaranteed doubles supply
is the g=2 classes pinned at u' <= (y+1)/6 by twins below y - 5-9% of split incidences at
all scales; the other 91-95% is alignment-conditional (~51% landing rate). Improving the
MV/B-T constant 2 toward 1 is itself parity-class (Motohashi: Siegel-zero).

**R10-R13 (moments, inversion zone, mirror, LP ceilings).** X <=> n2(t) = t - P(t) <=> the
hit schedule compresses at mean M_X = S1/(t-P). CS ceiling C_CS = M2/S1 is unconditional;
C_CS/M_X = 1.26, 1.41, 1.53, 1.58 at y = 211, 503, 2003, 5003 - GROWING, while the needed
window M_X/M_real narrows 1.22 -> 1.05. **Inversion zone**: where R(t) > 1, moments + prime
count alone force n0 > 0; nonempty at every y tested to 10007; worked instance y = 503,
t = 4: S1 = 3, M2 = 5 give n2 >= 9/5 > 1 = t-P, forcing the twin (521,523). Closing it as a
theorem needs a short-prefix prime lower bound at 0.07-0.13/integer (superdense; nothing
published below the 0.525 exponent). (R11) R = eff*boost, eff = (S1^2/M2)/n2,
boost = 1 + n0/(t-P); ladder to 10^7 has sup R -> 1+, first EMPTY windows at y = 5000011 and
10000019 (band T = 200000); windows opening with a twin within <= 4 slots revive R to 1.923
at any y (y = 5000087, 5000101, 5000539). (R12 mirror theorem) k -> -k swaps omega_L,
omega_R and fixes m_k, so every mirror-augmented moment doubles and every ratio (R, eff,
boost, M_X, all ceilings) is invariant: mirror-awareness is vacuous at the moment level; any
edge must use positions jointly with signs. (R13) Sharp LP n2 bound from S1, M2, M3: integer
order-2 beats continuous CS by 0.3-0.5% (at y = 10007, t = 17204 it still refutes,
n2 >= 7744 > 7702); order 3 adds nothing without the arithmetic cap m <= (log_5 y^2)^2, with
it 0.6-2.8% more. The window-scale chasm (~48%) is untouched: **the X-gap lives entirely in
the zeroth moment (twin mass), which no power moment sees.** Forcing must come from
placement/pin/alternation structure.

### 2.2 The tolerance route and its parts

**R14 (tolerance theorem - the route's sufficiency statement).** If
F(M+q') - F(M) <= alpha*q' at every consecutive step with q' > 47, for ANY fixed alpha within
alpha*(y)-scale - in particular 2.5 or 3 - then F(2,y) <= 354 + alpha*(S(y) - 328) <
(y^2-y)/2 for every prime y >= 53 (S = prime sum). Checked exactly at every prime in
[53, 10^6]: zero failures, worst ratio 0.6557 at y = 113 (alpha = 3); beyond 10^6 by
Rosser-Schoenfeld (S(y) < 1.25506 y^2/ln y, sufficient once ln y > 7.54). With y <= 47 known
directly this gives a survivor in every window - twins infinite. alpha*(y) grows like ln y:
5.64 at y = 101, 8.71 at 10^4, 13.3 at 10^6. Base chain F(2,y) = 6,15,21,33,54,75,102,129,
174,264,273,309,354 at y = 5..47; F(2,53) >= 420 (alpha = 2.5 demands <= 486, alpha = 3
<= 513). **Route quoted at alpha = 3** (R26).

**R15 (saturation regime).** If q-1 > F(M) no two consecutive openings can both be deleted,
so F(M+q) = F2(M) and incr = F2 - F <= F < q (alpha = 1 automatic). But along the consecutive
chain q' < F(M) always: compliant and needed regimes are disjoint - the theorem covers
far-gear additions only.

**R16-R18 (corridor laws, record scarcity, top-stratum adjacency).** A gap of length G has
left endpoint a mod 35 in A(G) = {r in E : r+G mod 35 in E}, |A| = 3..15 (G = 34 mod 35
forces a in {3,18,33}); concentration exceeds forcing (gears <= 19: all 20 record gaps at
residue 5; gears <= 23: records at {3,33}); 294 of 1225 length-pairs mod 35 have A3 empty.
Both kernel-checkable; prune the F(2,53) search 2-5x. (R17) Full periods have 4-20 maximal
gaps (mirror-paired); minimum separation between record gaps is 0.45-2.29% of the entire
primorial period (851,695 slots at gears <= 23) - anti-clustering five orders beyond what
lemma 1 needs; adjacent (F2-F)/q along the chain 0.92, 0.88, 1.10, 0.78, 0.52 (gears <=
11..23), max 1.16, min 0.15 further out, never above 1.2, no growth. (R18) At y = 13,17,19,
23 the top stratum occupies 4-6 classes mod 385 and no two top-stratum classes can be
adjacent (class check EMPTY at all four); dangerous-pair alpha1 certificates close at all
four by three tiers (A3-empty / mod-385 disjoint / direct), tier C growing 4 -> 96.

**R19 (merge census + spectrum reduction).** Steps 11->13 .. 23->29: chains 264 / 2,897 /
43,462 / 745,480 / 15,660,527; excess_k = 0,2,0,3,4; max k = 2,2,3,2 (k = 3: 62 chains at
19->23). Argmax interiors are literal {2u', q'-2u'}; identity
excess = interior_sum - (F2 - g_L - g_R) verified. Rigorous: F(M+q') <= F_{k_max+1}(M),
excess <= F_{k_max+1} - F2. Spectrum increments are q/3-scale, not F-scale. CORRECTED by
R40: these linear scans missed the cyclic-seam chain - 2,898 at 13->17, 15,660,528 at 23->29.

**R20 (fuel caps).** Tail-run cap (residue-free): k_max <= T(M,2u') + 1, T = longest run of
consecutive gaps >= 2u' (measured T = 3,2,4,3,4,5). **Literal cap theorem**: the literal walk
must stay in E mod 35, and the maximal run is a function of q' mod 210 only. Over all 48
invertible classes: cap 2 at 24 classes, cap 3 at 4, cap 4 at 14, cap 6 at 6 (q' = 37,53,83,
127,157,173 mod 210). **Literal chains have at most 6 members, for every gear, forever**
(verified against every prime to 5000). Explains realized k_max 2,2,3,2,4 (caps 2,2,4,3,4;
saturated at 17,19,31); k = 5 at q' = 31 forbidden; the k = 4 event (29->31, word (10,21,10),
4 per period) sits at a cap-4 gear. Extension beyond the cap requires a padded link. Litcap
caps only the LITERAL part and never predicts realized arity (X29).

**R21 (word-indexed identity).** With W(q') = alternating words of length <= litcap-1 plus
padded words:

    F(M+q') = max( F2(M), max over COMPATIBLE w of [span(w) + FS_max(w; M)] )

An identity, not a ceiling (every occurrence of a compatible word fires in |valid starts| of
the q' CRT copies; incompatible words never fire). Word list and compatibility depend on
q' mod 210 alone; only occurrences and flanks come from M. Verified exactly at all six steps
11->13 .. 29->31; consistent with the padded winner at 31->37. Fuel length and record growth
are separate channels: long words have small flanks. Generalised to an arity-free algebra by
R46.

**R22 (tier A: both-maximal exclusion).** A word occurrence with flanks is an (l+3)-point
chain in E mod 35. "Both flanks maximal (= F)" is machine-free FORBIDDEN at 14 of 16
word-step pairs (exceptions w = (8), (15) at 19->23). Tier B (moduli 385 ... 1616615, gears
through 19) adds EXACTLY ZERO new exclusions. Hierarchy: A (machine-free, scalable) vs C
(full-period scan, unscalable - 3.3e10 slots at 31->37). Compatibility is CRT-independent of
the carrier, so firing and tier A never interact. Off-target for (D): X16.

**R23-R24 (padding arithmetic; the padded-flank requirement).** A padded link costs a full q'
of budget while the whole budget grants (alpha/3)q'; count bound p <= (F + (alpha/3)q')/q'
(~F/q', grows like y/log y - no structural cap); onset gate F(M) >= q' (first three steps
have none by impossibility). Padded gaps are rare: 0 / 0 / 0 / 86 of 378,675 (0.023%) / 6 of
7,952,175 / 2,090 of 214,708,725 at steps 11->13 .. 29->31; only the exact value q' ever
occurs; no k >= 3 padded window exists. **The corpus's gear-37 anomaly is the onset of
padding**: at 31->37 the winner is [pad 37][literal 12], span 49, FS 39, merged 88 = F_k(37).
Min opening-distance from a maximal gap to a padded gap: 710 / 558,331 / 47,729 (19->23 /
23->29 / 29->31). At 41->43, p <= (91+43)/43 = 3.1, so p = 3 is consistent. (R24) With p
padded and ell-p literal interiors, FS <= F - (p - alpha/3)q' - (ell-p)q'/3; at alpha = 2.5,
p = 1 forces FS < F - q'/6, p = 2 forces FS < F - (7/6)q' - REQUIREMENTS given tolerance, not
derived facts (X15). Padding provably limits only its own span contribution, never its flanks.

**R25 (per-step constants, k-frame).** Winner word, span, FS, incr/q':

    11->13 (4)  lit  span 4  FS 4   0.308 | 13->17 (6)   lit  6  12  0.412
    17->19 (13) lit  13  12  0.368        | 19->23 (8,15) lit 23  11  0.391
    23->29 (10) lit  10  33  0.310        | 29->31 (10)  lit 10  48  0.484
    31->37 (37,12) PADDED span 49 FS 39   0.811

Budget incr/q': 0.833 (alpha 2.5), 1.000 (alpha 3). Corpus next steps 0.220q and 0.837q
adjacent (37->41, 41->43) - padding is intermittent. FS can exceed F (1.09F at 13->17, 1.12F
at 29->31); measured (FS-F)/q' <= +0.161.

**R26 (route form at alpha = 3).** The tolerance hypothesis factors exactly into: (A) word
list - finite, from q' mod 210 alone, PROVEN; (B) literal span - <= 5 letters,
span < (10/3)q', PROVEN; (C) padded span - count-capped and onset-gated, PROVEN; (D) flank
bound FS_max(w) <= F + (alpha/3)q' - span(w) for every compatible w - the sole gap;
(E) partial: R22. At alpha = 3 the binding step 31->37 has margin +7 (19% of q') vs +0.83
(2.7%) at alpha = 2.5. (D) at alpha = 3 is incr_k <= q' localised to <= 6 pinned words/step.

**R27-R29 (flank size, spectrum bridge, envelope).** The flank pairs attaining FS_max are
never maximal: largest single flank across all 15 word-steps runs 0.16F to 0.81F (29->31:
FS_max = 48 at (gL,gR) = (18,30), F = 43); (D) is a mid-tail x mid-tail pair-sum bound, with
margins at alpha = 3 of >= 0.52q' at every literal step and 0.19q' at the padded step.
(R28) span(w) + FS(w) = a sum of exactly k+1 consecutive gaps <= F_{k+1}(M) (definitional;
the one-line kernel bridge); strict ordering Wall-V clustering (F2 - F = O(q')) ==> spectrum
flatness (F_{k_max+1} - F <= q') ==> (D) (qualifying windows only, relative density
~(3/q')^{k-1}) - flatness is FALSE (X17), so (D) cannot be weakened by dropping position
information. (R29) Across all 15 word-steps span(w)/F + maxflank(w)/F in [1.00, 1.45];
merged/F <= 1 + q'/F, gap +1.286 down to +0.121 at 31->37; incr/q' mean ~0.44, no upward
trend. SUPERSEDED as a machine law by R33: the variable is ln(occ), not span.

**R30 (suppression decomposed; par trading).** Compatibility suppresses via (i) a size
threshold (interiors >= 2u' ~ q'/3, often zero effect at depth 3) and (ii) residue
coincidence (3 of q' residues, ~10% - carries the whole suppression at binding depth).
**Par trading**: each added link buys ~q'/2 of span and costs about the same in flank sum, so
merged max is nearly depth-independent (spreads 0-14% at machines 13-29; band ~25% after
machine 31's 22.7%, machine 41's 9.3%); k_win <= 3 at all seven measured steps, winners
SHALLOWER as machines grow (k_win = 3 at machine 31, 1 at 41). (D-a)/(D-b) subsumed by R31,
then R39.

**R31 (suppression law + corrected flatness).** The extremal j-window migrates to several
medium gaps (max element/sum 0.35-0.64) and deep extremal windows never contain the record
gap. Luck test: the qualifying maximum sits where a random p-sample's max would (luck 10^-0.1
to 10^-1.3); the structure lives in p_j - qualifying interiors are strongly negatively
correlated (p_j vs p_1^(j-2): x26 at machine 23 j=4; x6.7 and x1400 at machine 29 j=4, j=5).
**Suppression law** F_j - qualmax_j ~ lambda(j-2)L. **Corrected flatness**: (D) <==
F_j(M) - F(M) <= q' + lambda(j-2)L for every j >= 2 - holds 15/15 (corrected margins 4.7 to
20.8, bounded, non-growing in j) where raw flatness fails 5/15. The j = 2 case IS lemma 1
(F2 - F <= q'); **deeper cases are the EASIER ones** - the reverse of what rounds 8-17
assumed, confirmed in certificate form by R52/R55. Superseded as the operative criterion by
R39 (same content, heuristic stripped); retained as the asymptotic reading.

**R32 (rigorous exposure bound; how much anti-correlation (D) needs).** "Gap = v" = (both
endpoints exposed) AND (no opening between); dropping the second gives the CRT-exact bound
p_j <= (1/rho) * sum over qualifying tuples of prod_q c_q(...)/q (multi-lag c_q(g); the 1/rho
converts per-slot to per-opening - X20). Corrected, it clears machine 19 j=6 and machine 23
j=5 but is SHORT by x28.8 (machine 23 j=6) and x2.0 (machine 29 j=5, j=6) - the missing
factor is exactly the dropped no-opening condition (closed by R38). And (D) needs almost no
anti-correlation: where a constraint exists at all, INDEPENDENCE (p_1^(j-2)) clears it by
x170 to x201,381.

**R33 (anti-correlation law + occurrence form of (D)).** R(lag) = P(both qualifying)/p_1^2 is
an adjacency effect and nothing more: lag-1 deficit (exact ZERO at machines 11-17 -
qualifying gaps cannot be adjacent there; 0.039-0.638 at 19-29), rebound above independence
at lag 2 (up to 1.897), independence by lag 4-5. Higher orders super-multiplicative (machine
29: p_5/p_1^3 = 7.1e-4 vs pairwise 2.2e-2, a further 30x). **Flank order-statistic law**
maxflank(w) ~ 2.05 ln(occ(w)) (sd 0.27), FS_max(w) ~ 2.77 ln(occ(w)) (sd 0.24); 2.77 matches
the independently fitted lambda = 2.73. **(D) in occurrence form**:
span(w) + lambda ln(occ(w)) <= F + q' for every compatible w (occ <= N x exposure product).
qspec41: max_j Q_j = 110 vs F + q' = 133 (margin +23), Q_j = 0 for j > 8. The lag-2 rebound
is partly Markov and is GONE at machine 31 (obs R(2) = 0.71 vs predicted 1.36) - a regime
change at padding onset (R36).

**R34 (wall verdict).** The multiplicative route evades Wall I (no capacity comparison),
Wall II (no prime lower bound anywhere), Wall IV (hypothesis strictly stronger than the
conjecture, honestly lossy), and is not Wall III (dimension-1 test: the analogous increment
statement for ordinary Jacobsthal would sharpen Iwaniec). The obstruction is a FOURTH wall:
**extreme-value control of sieve patterns (Wall V)** - now at bounded complexity. The only
route whose missing lemma is about the machine's own gap word rather than about primes.

### 2.3 Operator frame, spacing laws, the exact criterion

**R35 (operator frame + nilpotency identities).** On C^{Z_P}: S = slot shift, D = exposure
projector = tensor over gears of D_q (CRT), B = I - D. Gap operator G_v = D(SB)^{v-1}SD;
every census quantity is a matrix element (N(v) = 1'G_v 1; N_j(u,v) = 1'G_u R^{j-1} G_v 1
with R = sum_v G_v the successor permutation, one |E|-cycle). Verified by operator iteration
at machines 11-19: **F(M) = nilpotency index of BS**; the qualifying-gap partial map A_V is
NILPOTENT with index = deepest qualifying run + 1 - indices 2,2,2,3,3,4,4 at machines 11..31,
so the fuel cap is a nilpotency statement (k_max <= index(A_V)). HONEST BOUNDARY: the exact
frame has NO spectral gap - R is a permutation, eigenvalues roots of unity; decorrelation is
an AGGREGATION phenomenon, not mixing of the exact dynamics.

**R36 (aggregated transfer matrix: Markov closure FAILS, with structure).** T[u,v] =
P(next gap = v | gap = u) from exact full-period pair censuses (machines 11-31). The one-step
chain OVER-predicts deep qualifying runs by growing factors: residue set V - x49 at machine
29 depth 3 (391/8), x4.4 at machine 31 depth 3 (2242/508); size floors at machine 29 - x4.4 /
x12.6 / x40 at depths 3/4/5 (machine 31: x2.5 / x4.3 / x27). Per-link conditionals fall
geometrically (machine 29: 5.5e-3 with 1 previous, 1.8e-4 with 2). **NO fixed-order transfer
matrix on gap values can be the proof object: the memory is longer than any fixed lag.**
Constants measured anyway: rho(T_VV)/p_1V = 0.65 / 0.039 / 0.20 / 0.24 at machines 19/23/29/31
(the Markov FLOOR of the deficit); full-chain |lambda_2| = 0.55-0.66, stable across 11-31.
Qualifying autocorrelation dips at lag 3 (0.31-0.51), recovers by lag 5; the chain predicts
recovery by lag 2 - more memory evidence.

**R37 (pair-support tropical side).** F_j <= longest j-node path in the pair-support graph:
exact at j=2 by construction, lossy from j=3 (x1.17-1.54, worsening with j). The V-interior
subgraph is ACYCLIC at machines 11-17 (there the pair table alone PROVES the qualifying depth
cap) but has cycles from machine 19 on while realized depth still caps at 2-3: **the depth
cap is a >= 3-point phenomenon from machine 19 onward - no 2-point census (pair table,
corridor law, c_q(g)) can certify it.** Max cycle means (full / V-graph): 5.5/-, 8/-, 12.5/-,
15.5/11.5, 19.5/10, 27.5/15.5, 34/31 at machines 11..31.

**R38 (THE RENEWAL LADDER; docs/novel/renewal-ladder.md).** For a qualifying tuple (v_1..v_m)
with opening offsets X and ANY set Y of interior offsets, the run event is a subset of
{X exposed, Y blocked} and #W'(X,Y) = sum_{T subset Y} (-1)^|T| prod_q c_q(X u T) - exact CRT
closed form, no period scan. run_m <= sum over tuples of #W'; nested Y (bisection order, s
points per gap) gives a monotone ladder from R32's exposure bound (s=0) toward exact. RESULT:
the ladder CLEARS the (D) requirement at every constrained case, including both R32 failures
- machine 23 j=6: requirement 1.5e-4, exposure bound short x28.8, ladder s=5 gives 5.0e-7
(x300); machine 29 j=5: requirement 1.8e-2, was short x2.0, ladder 2.0e-4 (x91); machine 29
j=6: 1.4e-6 vs 2.8e-3 (x2000). First joint-gap bounds beyond scan reach: machine 37 (period
1.24e12), p_5 <= 3.4e-2, p_6 <= 9.8e-3. HONEST LIMITS: tightness above exact degrades with
machine size (x40 at 29 m=2 up to x1.8e5 at 31 m=3); no zero certificate reached (smallest
surviving total 4, at machine 23 m=4 where truth is 0) - the 2^|Y| IE cost bars Y = all
interiors (delivered by R43).

**R39 (THE EXACT QUALMAX CRITERION - the operative form of (D)).** Full-period censuses with
cyclic seam exact (machine 31 = 3.34e10 slots): machine 31 F_j = 58, 68, 85, 90, 92, 97
(j=1..6); p_1V = 0.018445; qualifying runs 502,708 / 508 / 0 at depths 2/3/4 (deficits x4.2 /
x77 / exact zero). The 8 depth-3 runs at machine 29 are all permutations of {10,10,21}, span
41, window sums 47-55 - the machine's complete k=4 fuel inventory. THE CRITERION: by the
merge law every new gap is a window sum with residue-qualifying interiors, so EXACTLY

    F(M+q') <= max( F2(M), max_{j>=3} qualmax_j(M; q') ),  and
    (D) at alpha = 3  follows from  max(F2, max_j qualmax_j) <= F + q'.

Measured at all seven steps 11->13 .. 31->37: HOLDS 7/7, margins 0.52-0.69q' at the six
literal steps and 0.19q' at the padded step; the criterion value EQUALS F(M+q') at 6 of 7
(slack 2 at 23->29 only). This is R31 with the heuristic stripped - no lambda, no L, no order
statistics, three exact census quantities. (Q_j is the same object at the weaker size
threshold; its margin collapse to 0.10-0.11q' vs 0.19q' says the residue condition retains
margin the size floor loses.)

**R44 (R39 at 37->41 - first beyond-scan step, DECIDED).** F_3(37) = 97 EXACT (SAT witness
k = 990,209,189,833, gaps [37,23,37] - a palindrome flanked by the top gear's own value;
UNSAT at every S in [98,178]; 178 = F2+F is the a-priori cap). Budget F(37)+41 = 129: the j=3
clause qualmax_3 <= F_3 = 97 holds with margin 32 = 0.78q' - margin RESTORED at the litcap-2
gear (the Q_j collapse is a litcap-6 phenomenon; next real test q' = 53). The j >= 4 concern
is discharged independently from the padding37 full-period census: max_j qualmax_j(37;41) =
91 = F(41) EXACTLY, so the criterion value is max(90,91) = 91 <= 129 - the EIGHTH measured
step, equality with F(M+q') at 7 of 8, margin 38 = 0.93q'. **(D) at alpha = 3 at 37->41:
DECIDED, both routes.**

**R40 (THE TWO-TEETH KILL SPACING LAW; docs/novel/two-teeth-kill-spacing.md).** T1-T5,
asserted on EVERY window of every full joint period P*q', steps 11->13 .. 29->31 (joint period
= q' copies of the old opening sequence, boundary and cyclic seam exact):

    T1  {2c, -2c} mod q' = {2u', q'-2u'} - tooth-difference residues ARE the literal letters.
    T2  interior spacings = 0 or +-2c mod q'.
    T3  nonzero-class signs STRICTLY ALTERNATE (padded spacings transparent; |#a - #b| <= 1
        per window).
    T4  minimum spacing 2u'.
    T5  FUEL-SPAN LAW  k <= 1 + span/(2u') <= 1 + 3 span/(q'-1) - the fuel cap as closed-form
        span arithmetic, no census.

MEASURED M1: every realized spacing VALUE is exactly 2u', q'-2u' or q' - never 2u'+q', 2q',
... which the residue classes admit - at all six steps. At 29->31 (joint period 3.34e10,
237 s vectorised): spacings 10: 7,815,766 / 21: 205,068 / 31: 4,180 and nothing else; windows
421,392,436 with k = 1/2/3/4: 413,380,422 / 7,999,018 / 12,992 / 4, and T3 + max_span 41
FORCE all four k=4 windows to be word (10,21,10). Reproduces R19's chain censuses exactly PLUS
the seam windows R19 missed.

**R41 (NILPOTENCY ADDITIVITY = THE SUM SPLITTING; the counting boundary).** Exact Kronecker
recursion (dense-exact at {5}+7 and {5,7}+11, operational by vector iteration at all four
steps 11->13 .. 19->23):

    B_new S_new = (B_M S_M) (x) S'  +  (E_M S_M) (x) (B' S')

New blocked walk = old blocked walk (x) shift + old renewal step (x) q'-kill; F(M+q') =
nilpotency index of the SUM (old factor index F(M), kill factor index 2). The operator is a
masked permutation (entries 0/1), so the m-th power expands over binary kill-words with NO
cancellation and CRT separates the tensor factors: (BS_new)^m != 0 iff SOME kill pattern is
(left) an old-machine pattern event AND (right) mod-q' realizable - and right-realizability
is EXACTLY T2/T3 (verified: right-factor matrices nonzero on all spacing-law kill patterns,
zero on all violating ones, q' = 13, 23). The merge law, the word grammar (A), the padding
count (C) and the fuel-span cap are corollaries of this identity plus T1-T5.
**THE COUNTING BOUNDARY (honest negative, sharp): NO function of the marginal data (F, kill
index 2, spacing law, litcap) bounds the index of the sum.** The 2-point relaxation - every
adjacent kill pair individually realizable - is satisfied by the INFINITE alternating word
from 19->23 on (pairs (8,15) x31 and (15,8) x31 at machine 19) while the true chain stops
(triples (8,15,8), (15,8,15): ZERO). delta <= q' is decided by which spacing-compatible kill
patterns the old machine realizes - clause (D), nothing else. FUEL-AS-BRIDGES: the per-k
record's largest bridged old gap FALLS with k (19->23: 0.84F at k=1, 0.80F at k=2, 0.60F at
k=3; the k=3 record window is [4,8,15,7] = 34). SUPERSEDED IN ONE PART: R41's "the truncation
arity GROWS (3-point at 19/23, 4-point at 29)" was the RESIDUE arity; on the
operator-relevant arity it is not monotone (R45, X28). R41's conclusion (no fixed-arity rule)
survives with a better reason.

**R42, R48 (corridor phase: statistical carrier and its closed-form eigenvalue).** Transfer
chain with state = left-endpoint corridor phase, three nested models against full-period
exact censuses in the same pass (machines 13, 19, 23, 29; mod 35 and 385). At m29 depth-3
V-runs (the x1400 independence deficit; value-only baseline rebuilt in-pass gives x48.8):

    state:        indep   value   ph35   (ph35,val)   ph385   (ph385,val)
    pred/exact:   x1400   x48.8   x3.6     x1.9        x2.3     x0.86

**The anti-correlation's statistical carrier is small-gear phase**: (phase mod 385, last
value) predicts the (D)-relevant deep runs within 15%; size-floor runs likewise collapse (m29
depths 3/4/5: x4.4/x12.6/x40 -> x1.12/x0.97/x2.2). LAG WAVE: the phase chain reproduces the
deficit-at-1-3 / excess-at-4-7 oscillation with correct peak/trough lags at every machine,
near-exact at mod 385 (m29 V-lags 2..7 exact 1.53/0.81/1.05/1.03/1.13/1.28 vs
1.56/0.83/0.86/0.99/1.16/1.26). The phase chain's lambda_2 is COMPLEX - |l2| =
0.963/0.912/0.886 at m13/19/23, arg 34/43/46 deg, period 360/arg = 7.8-8.4 lags: **the
corridor resonance IS this eigenvalue** (arg ~ 2 pi mean_gap/35), distinct from the value
chain's real lambda_2 = -0.55. HONEST RESIDUALS: every phase chain over-predicts lag-1
adjacency (that exclusion is teeth-value, not phase); size-floor depths 5-6 keep memory
beyond (mod 385, value) (x2.2-3.0).
(R48) Under phase-value independence the transition matrix is the CIRCULANT of the gap
distribution, so **lambda_2 = phat(1) = sum_g P(gap = g) e(g/M)** - the gap law's
characteristic function at the corridor frequency. Exact histograms (m11/13/17/19/23):
|phat(1)| = 0.9612/0.9380/0.9128/0.8847/0.8576, arg = 29.1/34.2/38.3/42.1/45.4 deg; the true
chain gives 0.9849/0.9634/0.9396/0.9125/0.8859 at 29.3/34.4/38.7/42.8/46.3 deg, reproducing
R42's measured values exactly. The closed form nails the ARGUMENT (error 0.13-0.89 deg) hence
the resonance period, and understates the MODULUS by a stable 0.0237-0.0282. Cumulant form
exp(i.theta.gbar - theta^2 var/2), theta = 2 pi/35, reproduces phat(1) to 0.1-1.5%:
**lambda_2 is determined by the mean gap and the gap variance alone**, both closed-form CRT
quantities. CROSS-LANE: Lateral's lambda_j = rho w_j / (1 - (1-rho)w_j), w_j = e(j/e) IS THE
SAME OBJECT - phat for a GEOMETRIC gap law of density rho; neither dominates (renewal
instance better in modulus, exact-histogram better in argument). Their pre-registered m29
mod-35 band |lambda_2| = 0.862 +- 0.004, arg +49.2 +- 0.4 deg was CONFIRMED by the exact
full-period chain (2.147e8 gaps): 0.8617 / +49.15 deg; the residual ~0.029 modulus deficit is
CLOSED by Lateral (non-geometricity of the exposed-step law, lambda_2 = q-hat(1/e)).
SCOPE (R54): corridor phase is a good STATISTICAL carrier and a DEAD certificate axis (X32);
its proof-side successes always combine it with gap history, never alone.

**R45 (THE ARITY LADDER - three arities separated).** All exact, full period:

    machine        11  13  17  19  23  29  31  37  41
    A_res           2   2   2   3   3   4   4   4   -
    A_kill = k_max  2   2   2   3   2   4   4   3  >=3
    A_relax         1   2   2   3   2   3   4   3   2
    litcap(q'%210)  2   2   2   4   3   4   6   2   4

A_res is monotone through m37; **A_kill and A_relax are NOT** - both fall at m23 and again at
m37, and A_relax falls to 2 at m41. VERDICT: the operator-relevant arity neither grows without
bound nor stabilises - it is gear-arithmetic-valued. Its literal part is permanently capped
(A_kill <= litcap at 7 of 8 measured steps; litcap <= 6 forever, R20); the one exception m37
(litcap 2, A_kill 3) is forced to use a PADDED link - all 1,579 killable 2-words at 37->41 are
padded. Only the padded component has no structural cap (only R23's ~F/q'). Supporting exact
results:
* T3 CROSS-CHECK (five machines): applying the alternation filter to the residue census
  reproduces the FUEL census exactly - killable run_j == N_{j+1} at m19 (62 = 31+31, the
  (8,15)/(15,8) pairs; the 172 (8,8) pairs are T3-dead), m23 (0 of 288 - all (10,10)), m29
  (4 = the (10,21,10) windows of 8 residue runs), m31 (216 = 188 (12,25,12) + 28 (25,12,25) of
  508), m37 (0 of 8 - the only realized depth-3 residue word is (14,41,14), two class-a letters
  with a transparent padded link, T3-dead).
* OVERLAP LEMMA (proved): given every realized depth-m word, run_{m+1} = 0 unless some pair
  w, w' of realized m-words has w[1:] == w'[:-1]. At m37 the realized depth-3 set is the single
  word (14,41,14), which does not overlap itself, so run_4^res(37) = 0 with no further
  computation - A_res(37) = 4 from the word census alone. (Inconclusive at m31, where the six
  realized words do overlap; there run_4 = 0 is the direct census.)
* SPAN CEILING (proved, asserted at all eight machines) A_res <= min{j : F_j < 2u'j} - true but
  loose by ~2x (X30).
* MACHINE 41 (q' = 43), first arity data beyond any census, exact CRT pattern counts (period
  5.07e13, no scan): the two LITERAL 2-words (14,29) and (29,14) are BOTH ZERO, so
  A_relax(41) = 2 - earliest refutation past m17. Nonzero killable 2-words are all PADDED:
  (14,43) = (43,14) = 170,203, (29,43) = (43,29) = 228 (total 340,862 over 11 words of span
  <= 90; one word (43,43) undecided at the 3e8-node budget), so A_kill(41) >= 3. All eight
  killable 3-words of span <= 90 are ZERO, including (14,29,14) and (29,14,29). HONEST LIMIT:
  the 3-word enumeration is cut at span 90, not at F_3(41) (which nobody has), so A_kill(41) is
  recorded as ">= 3 and no depth-3 chain of span <= 90".
* Conditional decay of the residue ladder (exact): m19 0.0311/0.0199; m23 0.0307/0.0012; m29
  0.0374/0.0055/0.00018; m31 0.0184/0.0044/0.0010 - falling, at m31 stably x4.3 per link.

### 2.4 The generator, the certificate ladder, the two-gap verdict

**R46 (THE ARITY-FREE GENERATOR: F(M+q') IS A KLEENE STAR;
docs/novel/kleene-generator.md).** On states (opening i, tooth s in {+,-}):

    K[(i,s),(i+1,s')] = d_i  if d_i mod q' in {0, a, b} and s -> s' is the T3
                             transition of d_i's class; else -inf
    L(i) = d_{i-1},   R(i,s) = d_i

**THEOREM (identity, not a bound): F(M + q') = L^T (x) K* (x) R**, K* = (+)_{m>=0} K^m. Proof
both ways from the merge law + T2/T3 (<=) and CRT choice of the killing copy (>=). K is
nilpotent with index = A_kill, so the star is a finite sum - but the statement never names a
depth. **Its m-th layer is exactly qualmax_{m+2}: one algebra generates every layer of R39's
ladder.** This resolves R41's counting boundary: the index of the sum is not a function of the
summands' indices, but IS this star.
COROLLARY (tropical dual certificate - (D) with the depth quantifier removed): (D) at
alpha = 3 holds iff there exists h on states with

    (C1) h(i,s) >= d_i
    (C2) h(i,s) >= d_i + h(i+1,s')   for every legal qualifying transition
    (C3) d_{i-1} + h(i,s) <= F(M) + q'

Necessity h = K* (x) R; sufficiency because any super-solution dominates the star. Every clause
is a ONE-STEP, ONE-OPENING inequality - the first form of (D) that is not an infinite family;
max-plus LP duality for the longest-path problem F(M+q') actually is. VERIFIED EXACT, full
period, at every scannable step (dense and streamed implementations agree digit for digit at
m19 and m23):

    step        index(K)  L(x)K*(x)R = F(M+q')  F+q'   margin      layers
    11 -> 13        2            11              20   +9  0.69q'  [11, 8]
    13 -> 17        2            18              28  +10  0.59q'  [16, 18]
    17 -> 19        2            25              37  +12  0.63q'  [25, 25]
    19 -> 23        3            34              48  +14  0.61q'  [31, 33, 34]
    23 -> 29        2            43              63  +20  0.69q'  [39, 43]
    29 -> 31        4            58              74  +16  0.52q'  [55, 58, 55, 55]

index(K) == A_kill at every step, and h is always the LEAST super-solution (every state tight)
- the certificate is exactly saturated. The 29->31 layer vector is informative: the winner sits
at ONE link (58) and deeper layers fall back to 55, so **the deepest chain is NOT the
maximiser** - par trading, in Kleene form.

**R47 (the class-level certificate - sound, lossy).** Replacing the opening by a class and
taking edge weights = max realised gap gives a SOUND class-level max-plus system:

    step        value only        (ph 35, val)  (ph 385, val)  (ph 5005, val)  budget
    11 -> 13    11 certifies      11            11             11               20
    13 -> 17    21 certifies      21            20             18               28
    17 -> 19    30 certifies      28            28             25               37
    19 -> 23    CYCLIC (vacuous)  45 certifies  42             34               48
    23 -> 29    60 certifies      60            45             43               63
    29 -> 31    CYCLIC (vacuous)  99 FAILS +25  99 FAILS +25   91 FAILS +17     74

POSITIVE, and it stands: **the value-only abstraction is CYCLIC exactly where A_relax >= 3** -
R41's counting boundary is precisely "the abstract operator loses nilpotency", and a
non-nilpotent tropical operator bounds nothing; corridor phase mod 35 restores nilpotency at
both machines and at 19->23 also CERTIFIES (45 <= 48). NEGATIVE, SUPERSEDED: R47's "at 29->31
no bounded state certifies" is withdrawn by R49 (X31).

**R49 (THE HISTORY LADDER - bounded state DOES certify (D) at 29->31; three gaps of history
are EXACT at every scannable step).** With A_m(mod) as in section 1: because the gap value is
IN the state, R47's three losses vanish at once - the edge weight is exactly d_i (not a max
over sources), the base R is exactly d_i, and for m >= 3 the LEFT FLANK L is exactly d_{i-1}.
Every real chain maps to an abstract walk of the same weight, so the closure is a SOUND upper
bound at every m, non-increasing in m. A_2 reproduces R47's "value only" column digit for digit
at all six steps.

    step        exact  budget   A_2  A_2+35 A_2+385   A_3  A_3+35 A_3+385   A_4  A_4+35
    11 -> 13      11      20     11      11      11    11      11      11    11     11
    13 -> 17      18      28     21      21      20    18      18      18    18     18
    17 -> 19      25      37     30      28      28    25      25      25    25     25
    19 -> 23      34      48   CYCL      45      42    35      35      34    34     34
    23 -> 29      43      63     60      60      45    43      43      43    43     43
    29 -> 31      58      74   CYCL      99      99    85      85      72    58     58
    31 -> 37      88      95   CYCL       -       -  CYCL       -     115    88   (A_5 88)

(i) AT THE FAILING STEP TWO GAPS OF HISTORY BEAT ANY AMOUNT OF CORRIDOR PHASE: at 29->31 R47's
ladder went 99, 99, 91 as the modulus climbed 35 -> 385 -> 5005, while A_3 with NO phase gives
85 from 1,460 states and A_3+phase385 CERTIFIES (72 <= 74, margin +2 = 0.06q'). Not uniform: at
19->23 R47's mod-5005 state gives 34 = exact where A_3 alone gives 35 - the two axes carry
different information and the combination dominates both.
(ii) **A_4 - three gap values, PHASE-FREE, 14,368 states and 3,513 edges at machine 29 - is
EXACT at ALL SEVEN scannable steps (11, 18, 25, 34, 43, 58, 88), machine 31 included.** A
fixed-order four-point local rule reproduces F(M+q') with no error at every scannable step.
(iii) **A_m is nilpotent exactly when m > A_relax(M)**: A_relax = 1,2,2,3,2,3,4 at machines
11..31 against smallest acyclic order 2,2,2,3,2,3,4 - 7 of 7.
Machine 31 (step 31->37, period 33,426,748,355; 6.23e9 gaps, 4,924 s) also gave, all
cross-validating existing rows: qualmax_{k+2} = 68, 85, 88, 68 (winner at layer 2, a 4-gap
window); Q_J(31;12) = 68, 85, 90, 91, 90, 88, 0 for J = 2..8, so max_J Q_J = 91 <= 95 (margin
+4); depth-3 chain inventory 216 windows, 188 of (12,25,12) and 28 of (25,12,25), reproducing
R45's T3 cross-check; maximising window [11, 12, 37, 28] = 88, i.e. R25's known PADDED winner
(37,12) with flanks (28,11), re-derived from the generator. **The generator is arity-free AND,
at every measured step, bounded-state certifiable. It is still not MACHINE-free:** A_m's state
space is O(F^{m-1}) and its edge set is the machine's own dictionary of realised gap m-tuples.

**R50 (THE J=5 OBJECT AT 29->31, EXACT).** Full period at machine 29 (2.147e8 openings, 109 s),
by layer k (window of k+2 gaps):

    layer k          0     1     2     3          (window gaps: 2, 3, 4, 5)
    qualmax_{k+2}   55    58    55    55          exact, residue + T3
    Q_{k+2}(29;10)  55    65    68    71          exact, size floor 2u' = 10
    A_2 + ph 35     55    80    92    99          FAILS  (+25)
    A_3 value only  55    58    67    85          FAILS  (+11)
    A_3 + ph 385    55    58    63    72          certifies
    A_4 value only  55    58    55    55          EXACT

THE EXACT J=5 INVENTORY IS FOUR WINDOWS, all with interior word (10,21,10) (span 41) and flank
pairs (7,7), (7,7), (7,4), (4,7), window sums 55, 55, 52, 52; addresses 858111062, 220171102,
672200337, 406081827. (Reproduces R40's four k=4 addresses and R45's T3-filtered count 4 of 8;
the depth-2 layer is 13,000 windows, 6,500 (10,21) and 6,500 (21,10), exactly mirror-paired.)
**The failure is entirely in the flanks and entirely at layer 3** - every failing bound is the
SAME interior word with flanks that do not occur:

    A_2 + phase 35   [29, 10, 21, 10, 29]  = 99     (realised PAIRS only)
    A_3 value only   [22, 10, 21, 10, 22]  = 85     (realised TRIPLES)
    A_3 + phase 385  [22, 10, 21, 10,  9]  = 72
    truth            [ 7, 10, 21, 10,  7]  = 55     (realised 5-tuples)

The machine's real depth-5 maximum is 55, NINETEEN under budget, and the true maximiser sits at
layer 1 (58, a 3-gap window) - par trading (R30) at its sharpest. The "J=5 failure" was a
property of the ABSTRACTION; the flank envelope of (10,21,10) collapses 29 -> 22 -> 7 as
required context deepens from pairs to triples to 5-tuples. That collapse is exactly what (D)
asserts, and it is invisible to any 2- or 3-point census (R37's boundary, re-measured).

**R51 (THE MARKED QUALIFYING SPECTRUM IS EXACT, NOT A RELAXATION).** Independently and
concurrently found by Mechanic (same bug, mechanism, corrected numbers). SANDWICH LEMMA
(proved, then verified): fix a phase phi; a relaxed window x_0 < ... < x_m with marked set M
(|M| = J-1, every unmarked interior killed, consecutive marked distances >= a) has
surviving-interior set S contained in M, so consecutive members of S are also >= a apart; with
s^- the largest survivor <= x_0 and s^+ the smallest survivor >= x_m, the survivors in
[s^-, s^+] are exactly {s^-} u S u {s^+}, a NEW-machine window of |S|+1 gaps clearing the
floor, of span >= x_m - x_0. Hence

    Q_J(new) <= Q^[J](old) <= max_{1 <= j <= J} Q_j(new),

so wherever Q_j(new) is non-decreasing up to J the relaxation LOSES NOTHING, and in every case
max_J Q^[J](old) = max_J Q_J(new) exactly - the only quantity the criterion uses. VERIFIED FOUR
WAYS. (a) BRUTE FORCE at 11->13 (machine 11 is 135 openings; every window, phase and marked
subset enumerated, no pruning, no DP): Q^[J](11) = [16,18,23,0] = the exact Q_J(13;6), where the
buggy marked_qspec.py reports [16,23,23,0]. (b) The CORRECTED implementation at all four
checkable steps returns the exact Q_J(new) in 22 of 22 entries: [16,18,23,0], [25,28,31,32],
[31,35,37,38], [39,43,50,55,60,0] - against [16,23,23,0], [25,28,32,33], [31,35,38,38],
[39,50,50,55,60,0]. (c) At the disputed step the corrected scan of machine 23 seeded at 70 finds
NOTHING above 71 at any J <= 7 in 79 s: max_J Q^[J](23) = 71 <= 74, **the 29->31 rung is not
lost**. (d) Q_J(29;10) recomputed from machine 23 by PHASE DECOMPOSITION equals the direct
machine-29 full-period scan exactly: 55, 65, 68, 71, 71, 71. MECHANISM: the buggy feasible()
returns True as soon as the marked quota J-1 is filled, without checking that surviving
interiors still to come are marked (an unmarked interior must be KILLED). Witness at 11->13:
window (252, 275), interiors [257, 263, 268, 270]; at phase 8 survivors are {263, 268}, distance
5 < a = 6, inadmissible at J = 3, but the DP takes M = {257, 263} (distance 6), hits cnt = 2 and
returns True, leaving 268 unmarked. The reported 85 is exactly the SURVIVOR-COUNT BOUND at J = 5
(max span of a machine-23 window carrying at most 4 surviving interiors, floor and marking
ignored): 55, 65, 70, 85, 90, 92 at J = 2..7. NET: the marked spectrum is a BETTER tool than
advertised (exact at every step, not "buys exactly one rung"); the 85 on the certificate side
(R50, A_3) is real, the 85 on the census side is an artifact, two different objects coinciding.

**R52 (MACHINE-FREE IS SATURATED AT THE CORRIDOR).** MF_m sound by inclusion:

    step        budget  exact   MF_3 mod 35  MF_3 mod 385  MF_4 mod 35  layer 0
    11 -> 13       20      11      15  OK       15  OK       15  OK        14
    13 -> 17       28      18      31 +3        31 +3        31 +3         21
    17 -> 19       37      25      47 +10       47 +10       47 +10        36
    19 -> 23       48      34     111 +63      111 +63      111 +63        50
    23 -> 29       63      43     105 +42      105 +42      105 +42        67
    29 -> 31       74      58     125 +51      125 +51      125 +51        86
    31 -> 37       95      88     211 +116     211 +116     211 +116      116

**The three columns are IDENTICAL at every step.** Neither a finer corridor modulus nor more
history buys a single unit once "realised" is weakened to "corridor-admissible": the
realizability information A_m uses is NOT corridor information - X11 and X13 in their sharpest
form, as numeric saturation rather than an exclusion count. And LAYER 0 alone - lemma 1,
F_2 <= F + q', with no chain in it - is already 2F or 2F-2 at every step and fails machine-free
from 19->23 on. **The machine-free wall is not in the deep layers; it is in the two-gap
statement** - R31's "deeper cases are the EASIER ones" in certificate form.

**R53 (CEGAR, first measurement of the obligation).** MF_4 is machine-free and gives 125; A_4
uses the machine's realised 4-tuples and gives 58 (exact). The difference is a set of yes/no
facts "is this 4-tuple of consecutive gaps realised by M?": MF_4 has 140,471 candidate edges
(68,578 distinct value 4-tuples) against A_4's 3,513 realised ones. Counterexample-guided
refinement: close the system; if the bound clears F+q' stop; else read off a maximising abstract
walk, ask the oracle about each of its 4-tuples, and delete every unrealised tuple (deleting a
VALUE tuple removes it at every corridor phase at once - sound). The bound is an upper bound on
F(M+q') at every stage. RUN 1, machine-free start at 29->31: 125 -> 86 in 12,781 refinements
(13,460 queries), then STOPS - the maximising walk is the EMPTY one and 86 = 43+43 is layer 0,
which uses no edge (R52's wall as a termination condition). RUN 2, with F_2(29) = 55 given
(deleting every state whose flank+base pair exceeds it; 131,804 of 186,732 states survive,
83,195 edges): 125 -> 74 and **(D) IS CERTIFIED** at iteration 5,863 after **6,395 ORACLE
QUERIES**, 55 s - against a 1,078,282,205-slot period scan. HONEST STATUS: the oracle is the
dumped realised-tuple set, which came from the scan, so this MEASURES the obligation and does
not yet avoid it; the refinement is greedy, so 6,395 is an upper bound for this strategy.
SUPERSEDED IN ONE PART: the "one extra integer" hypothesis is discharged by R58 - refining pairs
as well as 4-tuples certifies with NO given integer, in 955 queries.

**R55 (BOTH MACHINE-FREE SUPPLIERS OF THE TWO-GAP FACT SATURATE AT 2F).** The two-gap statement
F_2(M) <= F(M) + q' is measured exact at all seven full-period machines (slacks +9, +12, +12,
+17, +24, +19, +27 at m11..31; seam-stitched pair census - the lag-1 joint csv is a LINEAR scan
short one cyclic-seam pair, recovered exactly from the marginal defect against ghist and
asserted). Two candidate machine-free suppliers: (i) THE HISTOGRAM - the tight bound over all
cyclic rearrangements of the gap multiset is F + G_2, G_2 = largest value pairable with F;
maximal gaps are mirror-paired (W_1(F) >= 2 at 7/7, values 4, 12, 20, 20, 4, 2, 4), so G_2 = F
and the histogram bound IS 2F, which EXCEEDS the budget from 19->23 on (margins -2, -5, -12, -21
at m19/23/29/31). By Lateral's Jordan-=-histogram theorem every unitary invariant of N = BS is a
function of the histogram, so **no operator invariant can supply the two-gap fact - the wall is
a theorem, not a search failure.** (ii) THE CORRIDOR - R52's machine-free layer-0 column is 2F
or 2F-1 at every step (14/21/36/50/67/86/116 vs 2F = 14/22/36/50/68/86/116): the corridor knows
exactly as much as the histogram here and one unit more at two steps. Same wall. CONTROL:
arranging the SAME multiset uniformly at random on the cycle, the typical max adjacent-pair sum
R_2 = min{B : E[#pairs > B] < 1} is 11, 17, 28, 39, 50, 61, 76 - which CLEARS the budget at all
seven steps, and the real F_2 sits AT OR BELOW even that typical value (F_2 - R_2 = +0, -1, -3,
-8, -11, -6, -8). So the machine is anti-correlated beyond random at the pair level, and the
two-gap statement needs only "not worse than a typical arrangement" - but no
rearrangement-invariant fact can say that, which is why the invariant route dies at 2F. Also:
the adjacency law A(M) = max over adjacent pairs of min(g1,g2) <= q' holds 7/7 but A/q' climbs
0.38 -> 0.89, and m37 partial coverage gives A(37) >= 40 vs q'' = 41 (P5: dies at 37, undecided).

**R56 (THE SURVIVOR-EXTENDED KLEENE GENERATOR; docs/novel/survivor-generator.md).** NEW
IDENTITY, proved from the merge law + T2/T3 both directions: a window of two consecutive NEW
gaps is a window of old openings all killed at one q'-phase EXCEPT ONE SURVIVOR, and the spacing
straddling the survivor is d_i + d_{i+1}; the survivor lives iff cls(d_i) is ILLEGAL out of the
current tooth. Adding that single SKIP transition SIGMA:

    F_2(M+q') = L (x) K* (x) SIGMA (x) K* (x) R
    F_j(M+q') = L (x) K* (x) (SIGMA (x) K*)^(j-1) (x) R   (proved; script-checked at j = 2 only)

VERIFIED EXACT, full period, seam stitched, at every scannable step: F_2(M+q') = 16, 25, 31, 39,
55, 68, 90 - against the INDEPENDENT pair census, and at 31->37 against Mechanic's CRT+SAT
F_2(37) = 90. CONSEQUENCE: **the two-gap statement at machine M+q' is LAYER 0 OF THE SAME ALGEBRA
one gear down**, so R53's "one extra integer" is not an extra hypothesis - it is a PROJECTION of
the very dictionary the certificate queries (the realised-pair sub-dictionary of A_m IS F_2(M)).

**R57 (THE HISTORY LADDER ON THE SURVIVOR SYSTEM).** A_m built ONLY from realised m-tuples of M
(skip transitions COMPOSED from two realised m-tuples, so no (m+1)-fact is used):

    step        exact F_2(M+q')   A_4 bound   A_5 bound   next budget F(M+q')+q''
    11 -> 13        16               16 EXACT      -            28
    13 -> 17        25               25 EXACT      -            37
    17 -> 19        31               31 EXACT      -            48
    19 -> 23        39               42            39 EXACT     63
    23 -> 29        55               57            55 EXACT     74
    29 -> 31        68               93            68 EXACT     95

**A_4 clears the next step's two-gap budget at EVERY step** (margins +12, +12, +17, +21, +17, +2
- the +2 at 29->31 is thin but positive), and A_5 restores EXACTNESS exactly where A_4 goes
loose: the survivor system needs ONE MORE order of history than the plain system (plain A_4
exact 7/7, R49; survivor A_5 exact at all three steps tested, 91,708 states at m29). In
particular **A_5(23) delivers the literal R53 integer: F_2(29) = 55, from machine 23's 5-tuple
dictionary, with no machine-29 scan.**

**R58 (CEGAR NEEDS NO INTEGER AT ALL).** R53's loop refined only EDGES (4-tuples), so layer 0 -
which uses no edge - was invisible and it stalled at 86 = 2F; the given F_2(29) = 55 was the
patch. But a state's layer-0 content is its (flank, base) PAIR - a realisability fact of arity 2,
the same kind as the edge facts. Refining BOTH (pair oracle = the full-period lag-1 census,
seam-stitched; edge oracle = the A_4 dump):

    step      queries (arity-4 + arity-2)   result           edges-only control
    19 -> 23      106 + 75  = 181           CERTIFIED 48      stalls at 50 = 2F
    23 -> 29       28 + 62  =  90           CERTIFIED 63      stalls at 67 = 2F-1
    29 -> 31      761 + 194 = 955           CERTIFIED 74      stalls at 86 = 2F

**(D) at 29->31 is certified from the machine-free start by 955 realisability queries and nothing
else** - down from R53's 6,395 + one given integer. The control stalls exactly at R55's wall,
closing the loop between the two measurements. THE SLACK SWEEP (feed a claimed bound U on
F_2(29), edges-only refinement): every U <= 74 certifies, every U in [75,85] stalls at exactly U,
every U >= 86 stalls at 86. **So the obligation is EXACTLY the two-gap statement itself -
F_2(M) <= F(M) + q', the OLD machine against the NEW budget - with zero further slack demanded**,
and R56/R57 supply it from one gear further down (55 exact via A_5, 57 via A_4; both <= 74 with
room). THE DELETION LADDER (Mechanic, proved): F_2(M) <= F(M + 1 gear) gives F_2(29) <= F(31) =
58, and U = 58 certifies - NUMERICALLY sufficient but LOGICALLY CIRCULAR for the induction (it
prices F_2(29) by the very F(31) the step is certifying). The survivor generator is its
non-circular replacement: same shape, one gear DOWN, and sharper (55/57 vs 58). Where they
separate: deletion-ladder slack F(M+q') - F_2(M) is 3 at 29/31 but 1 at 37/41 (F_2(37) = 90 vs
F(41) = 91), while the budget slack grows - the ladder route thins as the survivor route does not.

**R59 (THE TWO-GAP VERDICT).** The weakest machine-independent fact that substitutes for R53's
integer: (1) **No machine-FREE fact exists at the invariant or congruence level** - the histogram
(hence every unitary invariant, by theorem) and the corridor (any modulus) both force only 2F,
over budget from 19->23 on (R55); proved for the two named families, not conjectured. (2) **The
correct substitute is not a fact ABOUT M at all but a PROJECTION OF THE INDUCTION**: the survivor
identity (R56) makes F_2(M) an output of the step below, so the certificate chain needs exactly
ONE kind of input per step - the realised-tuple dictionary - queried finitely (90-955 times per
step, R58). The two-gap statement is not an extra obligation; it descends. (3) What remains for
all machines: (i) answer dictionary queries WITHOUT a scan - R43's pruned-IE pattern counter is
the named supplier, and every query here is arity <= 5 and span <= F_2, the cheap end of its
measured cost curve; (ii) make the descent uniform in y (the survivor identity is already
uniform; the query count is not yet bounded by anything proven).

**R43 (THE EXACT PATTERN COUNTER - the dictionary supplier).** Lateral's hereditary-zero pruned
IE (psd_bite.bonferroni_runs) adapted to pattern events with required-open seeding: #(X exposed,
Y blocked) with Y = ALL interiors - R38's named blocker (the 2^|Y| cost) DELIVERED for spans up
to ~75. Key structural fact: nonzero subsets are downward-closed (N monotone), so the DFS cost =
|{T : N(T) > 0}| independent of order. VALIDATED exact against every census row: m19 run2 = 234,
run3 = run4 = 0 (zero certificates); m23 and m29 all rows - run3(29) = 8 reproduced by pure CRT
arithmetic in 14 s (3.0e6 nodes), the 8 needles of a 1.08e9-slot period, no scan; m31 (period
3.34e10) run2 = 502,708 EXACT MATCH (3.4e8 nodes, 1611 s). Partial run3(31): the six nonzero
tuples found - (12,12,25) 139, (12,25,12) 188, (12,25,25) 7 + mirrors 139/28/7 - sum 508 = the
full census value; heavy padded tuples exceed the 3e8-node budget. COST CURVE: grows
~exponentially in span (98M nodes at span 74; budget dead at span 99).

---

## 3. Refuted claims (kept as refuted - do not re-derive)

**X1. Pair-coincidence doubles bound.** n2 <= L*s(z) + g(g-1) needs s(z) < 1, i.e. z <= 137
(s(127) = 0.959, s(139) = 1.005), band top < 19321 - but so short a band can hold
12/ln(6L) >= 1.22 primes/slot (Brun-Titchmarsh), above the < 1/slot X must supply.

**X2. Onset-scale contradiction.** Needs pi(y+H) - pi(y) >= H/6 + 1 (Hensley-Richards strength)
and as a universal statement is FALSE: 310/442 real windows have NO twin in the onset prefix.

**X3. CUM as leverage.** Exactly equivalent to Reduction A (lossless both ways); E(y) collapses
to 3. Diagnostic value only.

**X4. Naive descent.** Re-derives Reduction A at constant ~1/2; the band-weakened form remains
bounded-gap strength, blocked at T1 (a prime between consecutive prime squares).

**X5. Overdetermination of the X-consistency equation.** Zero degrees of freedom: the census
theorem makes both sides one arithmetic; the system collapses to n0(t) = 0. The forced value is
the unconditional floor; every ceiling sits a parity factor 2 above.

**X6. "CS ceiling lands at 2x the need".** Opposite measured: C_CS/M_X grows 1.26 -> 1.58 while
the target window narrows 1.22 -> 1.05. Bonferroni-2 vacuous once mean m > 3; Selberg Lambda^2
bounds n0 from ABOVE - wrong direction against n0 = 0.

**X7. The inversion zone as generator.** "Zone revives for infinitely many y" is equivalent to
the twin prime conjecture; no certificate short of the conjecture exists. Detector only.

**X8. Mirror / third-moment tightening.** Mirror-awareness vacuous at every moment order; LP
order 3 moves < 3% against a 48% gap; the X-gap is zeroth-moment only.

**X9. Multiplicative per-step bounds of budget shape.** r <= (q'/q)^2 false at 6 of 12 chain
steps; uniform ratio caps cannot close (pi(y) steps vs a y^2 budget force geometric mean -> 1).
Only the additive shape incr <= alpha*q closes (R14).

**X10. A-priori chain-condition cap in the needed regime.** The saturation theorem applies only
when q-1 > F(M), disjoint from the consecutive chain; in-range the raw cap (k_max + 2) is
exponentially over budget. Gap structure alone cannot bound k.

**X11. Bounded-modulus residue laws capping sizes.** Every (G1,G2) pair is within L1 distance 1
of a corridor-allowed pair (any bounded modulus) - corridors constrain where, never how big.
Verbatim for flank pairs (408/1225 forbidden at w = (10), slide 1 escapes) and spectrum
increments. Sharpened to numeric saturation by R52/R55.

**X12. Local capacity corridors.** F2_k(11) <= 12 is tight (actual 11), but the margin
rho - 2 sum 1/q dies two-three gears above ANY base (vacuous at y = 17 for base {5,7}, y = 31
for {5..17}) - Wall I in local form.

**X13. Tier B.** Lifting 35 -> 385 -> ... -> 1616615 adds exactly zero exclusions anywhere tier A
did not. B is not a tier; the hierarchy is A vs C only. Re-confirmed numerically by R52.

**X14. "Padded links need a common object".** Padded gaps are RARE (0.001-0.023% of gaps,
mid-tail: q'/mean ~ Cy/log^2 y -> inf), and the literal cap does NOT cover padded chains - no
structural cap on padded count, only budget arithmetic + onset gate.

**X15. "FS < F - q'/6 at padded occurrences" as structure.** A requirement, not a derived fact;
measured padded FS/F roughly doubles (0.32, 0.32, 0.42, then 0.67 at 31->37).

**X16. Both-maximal exclusion closes steps for (D).** The binding flank pairs are mid-size, never
maximal (R27) - it excludes a configuration that never binds. Correct, kernel-worthy, off-target.

**X17. Spectrum flatness (at fuel depth).** F_{k_max+1} - F <= q' is FALSE at 29->31 (F_5 - F = 42
vs q' = 31; true incr 15; lossiness x1.4-2.8). Raw flatness fails 5 of 15 machine-depth pairs
(j = 5,6 at machines 23, 29; j = 6 at 19), exactly where suppression is largest (repaired R31/R39).

**X18. "FS <= F".** False: FS/F = 1.09 (13->17) and 1.12 (29->31).

**X19. Span-monotone envelope as a machine law.** Wrong variable: maxflank follows ln(occ(w)),
not span (R33).

**X20. The uncorrected exposure bound (conditioning error).** Omitting the 1/rho (per-window vs
per-slot) made the bound appear to clear (D) word-free; corrected it falls short x2-x29. The gap
is exactly the dropped "no opening strictly between" renewal factor (closed by R38).

**X21. The anti-correlation law as the needed input.** Over-specified: independence alone clears
every constrained case by x170-x201,381. The needed fact is only "no > ~170x positive correlation".

**X22. Round-13 tier-A flank test / round-12 count.** The first pass conflated left flank with
gR = 1 (manufactured false exclusions); "0 of 17 word-step pairs" should be 16. R22/R27 stand.

**X23. "Arithmetic luck" as an endpoint.** The max-window's failure IS plausibly luck, but the
rate p_j is structural; building the window-profile object turned luck into the suppression law.

**X24. Markov / spectral closure of p_j ("the deficit is a spectral gap").** FALSE in the
aggregated chain (over-predicts deep runs x4.4 to x49) and VACUOUS in the exact frame (R is a
permutation, no spectral gap). No fixed-order transfer matrix on gap values can be the proof
object (R35/R36).

**X25. Pair-support tropical bound as a depth-cap certificate.** Exact at j=2, lossy from j=3
(x1.17-1.54); the V-subgraph has cycles from machine 19 on while realized depth caps at 2-3 - the
depth cap is a >= 3-point phenomenon, invisible to any 2-point census (R37).

**X26. The renewal ladder as a zero-certifier.** Rate bounds only: smallest surviving total 4 at
machine 23 m=4 where truth is 0; tightness degrades x40 -> x1.8e5; the 2^|Y| IE cost bars Y = all
interiors (R38). Delivered instead by R43's pruned-IE DFS.

**X27. The memoized alternating recursion as a faster counter.** f(i,masks) = f(i+1,masks) -
f(i+1, masks & rot_i) has MORE states than DFS nodes on the same pattern (1.58M vs 1.37M at m19
(15,23,23)), and a span-74 m31 pattern was unfinished at 10 min against the DFS's 445 s (R43).

**X28. "The truncation arity grows" (R41's own reading).** SELF-CORRECTED in R45: the growing
sequence 3, 3, 4 was the RESIDUE arity, and a residue-qualifying run is not a kill chain (T3
forbids two consecutive same-class letters). Operator-relevant: A_kill = 2,2,2,3,2,4,4,3 and
A_relax = 1,2,2,3,2,3,4,3,2 - both go DOWN as well as up. R41's conclusion (no fixed-arity rule)
survives for a better reason: the arity is an arithmetic function of q' with an uncapped padded
component.

**X29. Litcap as a predictor of realized arity.** A proved cap on the LITERAL part only: at m41
litcap = 4 while the literal 2-word count is exactly 0, and at m37 litcap = 2 while A_kill = 3
(forced padded) (R45).

**X30. The span ceiling as an explanation of the arity.** A_res <= min{j : F_j < 2u'j} is proved
at all eight machines but loose by ~2x everywhere - the arity is limited by joint realizability,
which is (D)'s content (R45).

**X31. "At 29->31 NO bounded state certifies" (R47's negative).** WITHDRAWN by R49: true of every
state R47 tested (all corridor-phase refinements of a ONE-GAP state), false in general -
A_3 + phase 385 certifies (72 <= 74) and A_4 is exact (58). R54 gives the reason the tested family
was doomed: congruence states on proper divisors of the period, which Lateral's T1 proves vacuous
(X32). The lesson R41 stated and R47 did not apply: the missing information is JOINT
REALIZABILITY of consecutive gaps - refine the history, not the congruence. R47's ladder 99/99/91
was never converging; it was three samples of a family now proved vacuous.

**X32. Congruence-class potentials (any modulus) as certificates.** Lateral's T1: a potential
that is a function of k mod m for a PROPER DIVISOR m of the period certifies nothing - every
class mod m contains a blocked slot, and h(k) >= h(k-1) + 1 forces h up around the whole m-cycle
(0 >= m). Kills mod 35, 385, 5005 structurally (R54). A_m escapes: its state is a tuple of gap
VALUES, not a function of k mod anything, and its edges carry machine-specific realizability.
(Lateral's related negative - every unitary invariant is a function of the gap histogram - costs
the plan nothing: no branch routes through an invariant.)

**X33. Q^[J](23) = 85 as an independent second failure at 29->31.** An implementation artifact of
marked_qspec.feasible (quota-filled early return, survivors left unmarked); the number is exactly
the survivor-count bound at J = 5, a different object coinciding numerically. The corrected marked
spectrum is EXACT at 22 of 22 entries and max_J Q^[J](23) = 71 <= 74 (R51). The round-23 brief's
premise of "two independent methods failing at one step" is half withdrawn.

**X34. The machine-free corridor certificate.** Dead and quantified: MF_3 mod 35, MF_3 mod 385
and MF_4 mod 35 are IDENTICAL at every step, and layer 0 alone is 2F or 2F-2 and fails from
19->23 on (R52).

**X35. The histogram / any unitary operator invariant as a supplier of the two-gap fact.** The
tight rearrangement bound is F + G_2 = 2F (maximal gaps mirror-paired, W_1(F) >= 2 at 7/7),
exceeding the budget from 19->23 on by -2, -5, -12, -21. Since every unitary invariant of N = BS
is a function of the gap histogram (Lateral's Jordan theorem), this is a THEOREM that the
invariant route dies, not a search failure (R55).

**X36. The deletion ladder as the induction's supplier of F_2.** F_2(M) <= F(M + 1 gear) is proved
and numerically sufficient (F_2(29) <= 58, U = 58 certifies) but LOGICALLY CIRCULAR - it prices
F_2(29) by the very F(31) the step is certifying; its slack also thins (3 at 29/31, 1 at 37/41)
while the budget slack grows. Replaced non-circularly by R56/R57 (R58).

**X37. "F_2 needs slack below the budget".** Registered prediction P4 guessed the threshold at
60-68. REFUTED by the slack sweep: every U <= 74 certifies, U in [75,85] stalls at U, U >= 86
stalls at 86 - the obligation is EXACTLY the two-gap statement itself, zero further slack (R58).

---

## 4. Live target, open questions, named next constructs

### 4.1 The live target

**(D), the sole open input of the tolerance route**, sufficient for twin infinitude via R14
(alpha = 3) + R26(A,B,C). Strongest available statements of the obligation:

* **Operative criterion (R39):** max(F2(M), max_{j>=3} qualmax_j(M;q')) <= F(M) + q'. Verified
  8/8 measured steps (11->13 .. 37->41), equality with F(M+q') at 7 of 8.
* **Generator form (R46):** F(M+q') = L (x) K* (x) R, with the depth-free dual certificate
  (C1)-(C3) - every clause a one-step, one-opening inequality.
* **Certificate form (R49/R58):** run the A_4 (plain) / A_5 (survivor) max-plus closure; the ONLY
  machine input is the dictionary of realised gap m-tuples, and CEGAR needs 90-955 realisability
  queries per step (arity 2 and 4) to certify from a machine-free start with no given integer.
* **Two-gap descent (R56/R59):** F_2(M) is layer 0 of the same algebra one gear down - not a
  separate hypothesis; it descends.

**THE LIVE TARGET IS THE TWO-GAP LAW, SUPPLIED WITHOUT A SCAN**: answer the 90-955 per-step
dictionary queries by R43's pruned-IE CRT pattern counter instead of by a dumped realised-tuple
set, and bound the query count uniformly in y.

### 4.2 Named next constructs

* **THE GAP-TUPLE DICTIONARY WITHOUT A SCAN (the chain construct, top priority).** (D) at a step
  is now exactly: "given the set of realised 4-tuples (plus lag-1 pairs) of consecutive gaps of M,
  run a ~14k-state max-plus closure". The closure is finite, small and kernel-checkable as it
  stands; the dictionary is the only thing that still needs the period. R52 kills the corridor
  route to it. The live route is R43's counter, which decides ONE tuple by CRT arithmetic with no
  scan (run_3(29) = 8 reproduced in 14 s from a 1.08e9-slot period). Deciding EVERY candidate
  would be 68,578 zero-certificates at 1e3-1e6 nodes each (1e8-1e11 nodes, too many); CEGAR needs
  only 955 at 29->31 - and every query is arity <= 5 and span <= F_2, the cheap end of R43's cost
  curve. NOT YET BUILT: the CRT side answering those queries; needs R43's cost curve measured on
  span-4 and span-5 patterns, a small well-defined job.
* **UNIFORMITY OF THE DESCENT IN y.** The survivor identity (R56) is already uniform; the CEGAR
  query count is not bounded by anything proven. Needed for an all-machines statement.
* **THE FLANKED LADDER.** Extend R38's tuple by (g_L, g_R) and certify span + g_L + g_R > F + q'
  combinations to zero - the direct rigorous route to (D) per step, now unblocked by R43's exact
  counter. The enumeration is small since only large flank pairs matter (R27).
* **A TIGHTER ABSTRACTION AT SCALE.** A_m's state space is O(F^{m-1}). Open: whether the needed
  order m is bounded (m > A_relax(M), and A_relax is non-monotone - 1,2,2,3,2,3,4,3,2 at m11..41),
  and whether the survivor system's +1 order penalty is permanent.

### 4.3 Other open items

* **O1. F(2,53):** >= 420; alpha = 2.5 demands <= 486, alpha = 3 demands <= 513.
* **O2.** Padding intermittency beyond m41; k_win vs k_max and F_4 - F vs q' at machines 37,41,43.
* **O3.** Uniformity in y of pinned addresses (drift of strata classes mod 385); after X32 this
  matters for statistics and pruning only, never for certificates.
* **O4. Census falsification asserts (standing):** literal chain longer than litcap(q' mod 210);
  any chain with k > T(M,2u') + 1; literal k = 5,6 at a non-cap-6 gear; any realized padded
  interior > q'; any step where max over compatible words of span + FS_max != F(M+q'); k = 7+
  anywhere; any A_m closure below the exact value.
* **O5.** Thinnest layer bands occur exactly at twin endpoints - self-reference at the binding
  case, uninterrogated.
* **O6.** The multiplicative tail bound N(L) <= P exp(-cL/y), c > 6: the genuinely open sieve-side
  middle ground - CUM neither implies nor needs it.
* **O7. A_kill(41)** exact value - now ">= 3 and no depth-3 chain of span <= 90"; needs F_3(41).
* **O8. A(37) vs 41** (adjacency law A/q' climbing 0.38 -> 0.89): needs Mechanic's m37 pair census.
* **O9. Maslov dequantisation** (Lateral): the Kleene star, the Boolean filtration and the
  analytic resolvent as one computation in three semirings. A lead, never used.

### 4.4 Prediction scorecard (pre-registered; verdicts as filed)

Machine-31 history ladder (data/kleene_history_31_prediction.txt):

    P1  A_3 value-only CYCLIC                     CONFIRMED
    P2  A_4 value-only nilpotent                  CONFIRMED
    P3  A_4 NOT exact (it is the boundary order)  REFUTED - A_4 = 88 = exact
    P4  A_4 expected to FAIL the budget           REFUTED - 88 <= 95

P3/P4 extrapolated the boundary order's looseness at the five earlier steps (21, 30, 35, 60, 85
against exacts 18, 25, 34, 43, 58). Wrong: at machine 31 the boundary order is exact (A_5 also
gives 88, as soundness requires). The standing rule "never extrapolate a per-step share - look it
up" was violated in a lane that had just quoted it.

Two-gap round (data/twogap_prediction.txt):

    P1  survivor identity exact              CONFIRMED 7/7 (with m31)
    P2  A_4 certifies                        CONFIRMED 6/6
        "...and is exact"                    REFUTED - loose +3/+2/+25 from 19->23 on; A_5 is
                                             the exact order (R57)
    P3  nilpotency at equal orders           NOT DECIDED (only m = 4, 5 run; all non-cyclic,
                                             untested at m = 3)
    P4  slack threshold 60-68                REFUTED - threshold = budget = 74 (R58)
    P5  A(37) > 41                           UNDECIDED (partial data >= 40)
    P6  F_2(37) maximiser avoids the max gap  REFUTED by Mechanic's witness [2, 88]: F_2(37) =
                                             90 = 2 + 88 CONTAINS the maximal gap with the
                                             minimal partner (m31/37 differs from m29/31, whose
                                             maximisers were (20,35) and (33,35), mid-size pairs)

Cross-lane: Lateral's pre-registered m29 mod-35 band |lambda_2| = 0.862 +- 0.004, arg +49.2 +- 0.4
deg - CONFIRMED (exact 0.8617 / +49.15 deg), sharper than either side's raw closed form (R48).

### 4.5 Standing rules / lessons earned

* Record every route's exact limiting event; name unproven inputs, never assume them
  (Hensley-Richards, Reduction A, the 0.525 floor: named-not-used).
* Quote the route at alpha = 3, not 2.5 (the 2.7% margin was an artifact).
* Distinguish requirements from derived facts (X15); check conditioning per-window vs per-slot
  (X20) - both caught only by adversarial re-testing.
* Never extrapolate a per-step ratio across steps - look it up (P3/P4).
* Flag trend observations as such, never promote to laws; "arithmetic luck" is a prompt to build
  the object, not a stopping point (X23).
* When an abstraction fails, ask WHICH AXIS is starved before refining the one in hand: R47
  refined congruence for a whole round when the starved axis was history (X31).
* Tier A is the only corridor tier worth formalising (X13); do not extend the both-maximal
  exclusion for (D) (X16).
* The two attack surfaces (sieve/prime) transfer at zero cost and zero gain (R4); separating P(t)
  from its floor is already a twin theorem (R9).
* Memory starvation killed jobs in three separate rounds (r22, r23, r24). Scanners now retry
  refused allocations for up to 20 min; a refused 8 MiB allocation is not a finding. Nothing filed
  rests on a partial scan. Shared scratch paths are not safe - draft into the repo.
* Every census is full-period with the cyclic seam stitched; linear scans silently lose the seam
  window (R19 vs R40).

---

## 5. Reproduction pointers

Scripts (research/), by result: constructor_ledger.py (R1-R3, R5 censuses y = 13,23,47);
double_onset.py (R7, L* = 27129, 442-window census); cumulative_margin.py (R4/R5 margins E(y),
y = 47..5003); x_consistency.py (R9, uses Lateral's split_gap_law.py); compression_bound.py +
compression_zone.py (R10 moments, inversion zone); zone_fate.py (R11 ladder to 10^7, LP ceilings
R13); multiplicative_route.py (R14, [53,10^6]); topgap_endpoint_law.py (R16 corridor laws, R17
record censuses, X11, X12); strata_adjacency.py (R18); merge_census.py (R19, 23->29 streamed
1.078e9); fuel_bound.py (R20 literal cap, 48-class check); word_ceiling.py + flank_bound.py (R21,
6/6); flank_tierA.py + flank_tierA_fix.py (R22; the _fix corrects X22); padded_bound.py (R23/R24);
flank_pairs.py (R25/R27); window_profile.py + suppression_law.py (R31); anticorr_law.py (R32/R33).

Rounds 20-24: tm_nilpotency.py (R35); tm_transfer.py (R36); tm_tropical.py (R37);
tm_renewal_bound.py (R38 ladder + assertions); tm_resid_runs.py (R39 residue-run censuses, qualmax,
spectra; data/tm_resid_runs.csv); tm_deepruns.py (deep fuel inventories); tm_qualmax_check.py (R39
criterion, seven steps); kill_spacing.py (R40 T1-T5 + M1; logs data/kill_spacing_23.log,
kill_spacing_23_29.log); nilpotency_additivity.py (R41; log data/nilpotency_additivity.log);
tm_corridor_phase.py (R42, --mod 35/385; logs data/tm_corrphase_19_23.log, tm_corrphase_29_31.log,
tm_corrphase_23_mod385.log, tm_corrphase_29_mod385.log); qualrun_zerocert.py (R43; log
data/qualrun_zerocert.log) and test_memo2.py (X27; log data/test_memo2.log); lambda2_closed.py
(R48; log data/lambda2_29.log); arity_ladder.py (R45 three arities, T3 cross-check, overlap lemma,
span ceiling, litcap - all asserted) and arity_probe41.py (log data/arity_probe41.log);
kleene_generator.py (R46 dense identity + certificate + R47 ladder, machines 11-23; log
data/kleene23.log) and kleene_stream.py (segmented, machines 23, 29; log
data/kleene_stream_23_29.log); kleene_history.py (R49/R50 ladder, exact layers, deep-chain
inventory, size-floor spectrum, all asserted; logs data/kleene_history_11_23.log,
kleene_history_29.log, kleene_history_31.log; prediction kleene_history_31_prediction.txt; the
realised-tuple dump data/tuples4_29.txt comes from `KH_DUMP4=... kleene_history.py 29 --specs
4:0`); marked_bruteforce.py (R51 exhaustive 11->13 audit) and marked_survival.py (phase
decomposition, survivor-count bound, corrected marked spectrum; logs data/marked_survival_23.log,
marked_survival_2329.log); machinefree_cert.py (R52); cegar_cert.py (R53; logs data/cegar_29.log,
cegar_29_f2.log); twogap_table.py (R55; asserts seam stitching, marginals == ghist, both saturation
identities, the corridor cross-check, the random-arrangement control; log data/twogap_table.log);
survivor_generator.py (R56/R57; asserts F(M+q') AND F_2(M+q') against the known ladders at every
step, soundness of every abstraction closure, and branch B0 == F_2(M); logs
data/survivor_11_23.log, survivor_29.log, survivor_31.log, survivor_29_m5.log); cegar_pairs.py
(R58; pair oracle asserted to reproduce F_2(M) exactly before use; --states and --sweep modes; log
data/cegar_sweep.log).

Data (research/data/): prefix_census.csv (R5), fuel_census.csv (F2(29) = 55, F2(31) = 68, N_k,
k_max by step), gap_pair_joint.csv + gap_pair_hist.csv (the p_j object, R33 - the lag-1 joint csv
is a LINEAR scan short one cyclic-seam pair, recovered and asserted in twogap_table.py),
multiplicity_summary.csv (R13), qspec_table.csv (qspec41, R33), tm_resid_runs.csv (R39),
tuples4_29.txt (R53 oracle dump). Predictions: kleene_history_31_prediction.txt,
twogap_prediction.txt (both written before their runs). Round-24 draft: data/r24/r24_draft.md.

Novel docs (docs/novel/): renewal-ladder.md (R38), two-teeth-kill-spacing.md (R40),
corridor-resonance.md (R42/R48), kleene-generator.md (R46), survivor-generator.md (R56).

Lateral's suites: split_gap_law.py, topgap_corridor.py (Lateral owns that name; Constructor's is
topgap_endpoint_law.py), psd_bite.bonferroni_runs (the pruned IE R43 adapts).

Anchors: requirement F(2,y) < (y^2-y)/2; F_k(M+q') chain 11/18/25/34/43/58/88; F2_k(2,y) = 33, 48,
75, 93, 117; F_2(M+q') chain 16/25/31/39/55/68/90; budgets F+q' = 20/28/37/48/63/74/95/129.

All censuses reported here are full-period with the cyclic seam stitched; model predictions are
labeled float; nothing filed rests on a partial scan.


---

## Constructor round 25 - THE SCAN-FREE DICTIONARY AND THE CHAIN

Brief: (a) answer the 90-955 CEGAR realisability queries per step by CRT
arithmetic instead of a dumped realised-tuple set; (b) chain the steps so each
consumes only the previous machine's word; (c) report exactly where the chain
first needs a fact it cannot generate.

**R60 (THE REALISABILITY CSP - the dictionary supplier, built and gated).**
Slot k is blocked by gear q iff k = +-u_q mod q, u_q = 6^{-1} mod q; by CRT a
slot IS a phase vector (a_q). So for a gap tuple with prefix-sum points X and
interior points Y = (0, span) \ X:

  THEOREM. (v_1..v_m) occurs as m consecutive gaps of M = {5..y} iff the system
    (open)  a_q not in {+-u_q - x : x in X}  for every gear q
    (cover) for every t in Y some gear q has a_q = +-u_q - t
  is feasible. The period never appears - pi(y)-2 variables, domains <= q.

`research/crt_dict.py` decides it exactly: bitmask coverage, branch on the
uncovered point with fewest live options, plus a CAPACITY BOUND (if the
unassigned gears cannot cover the remaining points even at their individually
best phases, the node is dead). The capacity bound is what makes REFUTATIONS -
the only answers a certificate may act on - affordable; without it machine 31
one-gap refutations exceeded 2e6 nodes, with it they cost 0.4-1.5 s.
GATES (all in the script, `validate`): decision == (R43 pruned-IE count > 0) on
2,013 tuples of arity 1, 2, 3 at machines 11, 13, 17; nine published anchors
including (10,21,10) at m29 True and (21,10,21) False, the m19/m23 holes at 24,
and (14,41,14) at m37; and the corpus ladder recovered with no scan -
F = 7, 11, 18, 25, 34, 43, 58, 88 at machines 11..37 and
F_2 = 11, 16, 25, 31, 39, 55, 68 at machines 11..31 (F_2(29) attained at the
pair (20,35)), plus the witness pair (2,88) giving F_2(37) >= 90 = the corpus
value.  The gate stops short of the EXACT F_2(37) deliberately: pinning it is
~3,800 over-budget pair refutations at a measured 5.8 s each, and that cost is
itself one of the round's findings (R66).

**R61 (THE DICTIONARY, and it EQUALS the scan).** `research/scanfree_dict.py`
builds D_1, D_2, ... level by level with the overlap lemma (R45: every
contiguous sub-tuple of a realised tuple is realised), so F_j(M) = max span in
D_j and D_3/D_4 are exactly what A_4 needs:

    machine   D_1   D_2     D_3     D_4     F_1 F_2 F_3 F_4   build (1 core)
    19         23   221   1,216   4,489     25  31  35  38       2 s
    23         33   429   3,135  15,696     34  39  50  58      79 s
    29         41   730   7,184  45,854     43  55  65  70    ~30 min
    31         55 1,254  15,020       -     58  68  85   -      24 min

Machine 31's F_1, F_2, F_3 = 58, 68, 85 are EQUAL to R49's full-period values
(58, 68, 85, 90, 92, 97) obtained from a 3.34e10-slot scan; its level 4 was
deliberately deferred (a ~2 h job) when the box went to 947 MB free.

**At machine 23 the scan-free D_4 is SET-EQUAL to Mechanic's full-period scan
census `gap_tuples_23_4.csv` - 15,696 tuples, identical**, and at machines 19,
23, 29 it contains every tuple of the round-24 A_4 dumps with ZERO missing.
A CRT computation and a period scan agreeing tuple for tuple.

POST-FIX RE-GATE, and a confirmation of Mechanic's mid-round repair: with the
bug fixed, the scan-free level-2 dictionary is SET-EQUAL BOTH WAYS to the
REPAIRED full-period lag-1 pair census - 221 pairs at m19, 429 at m23, nothing
in either that is not in the other. Lateral caught the defect (every ghist row
dropped the cyclic wrap gap), Mechanic repaired it at source at 04:55, and the
CRT dictionary independently confirms the repaired file is now exact. My own
loader (chain_cegar.load_pairs) handles both states and asserts cyclic
consistency rather than patching a seam.

**R62 (THE CEGAR LOOP ON THE CRT ORACLE - R58 REPRODUCED QUERY FOR QUERY, AND
A NEW RUNG).** `research/chain_cegar.py` puts the CSP oracle where round 24 had
the dump. `--shadow y` runs both side by side and asserts agreement on every
query:

    step        queries (arity 4 + arity 2)  bound  budget  CRT vs dump
    19 -> 23        106 + 75  = 181           48      48    181/181 agree
    23 -> 29         28 + 62  =  90           63      63     90/90  agree
    29 -> 31        761 + 194 = 955           74      74    955/955 agree
    31 -> 37       (CRT only)    3,399        95      95    no dump exists
    37 -> 41       (CRT only)       94       235     129    NOT CERTIFIED

- exactly R58's counts, no disagreement in either direction, and then **31->37
CERTIFIED with no scan and no dump anywhere in the loop** (356 s wall, 267 s of
it oracle; MF_4 = 442,294 states / 558,182 edges). Every deletion is licensed by
an exhaustive refutation and an UNDECIDED query deletes nothing, so the running
value is an upper bound on F(M+q') at every stage. Oracle cost per query: 0.4 ms
mean at m19, 2.3 ms at m23, 16.3 ms at m29, 24.4 ms at m31 (worst single
query 0.19 s and 1.07 s). Query spans run well
past 2F (median 45 / 57 / 84, max 90 / 105 / 127 against 2F = 50 / 68 / 86;
at 31->37 median 62, max 167 against 2F = 116) -
R59's "span <= F_2" was optimistic and is corrected here.

**R62b (37 -> 41: NOT CERTIFIED, and the cost is the finding).** The next rung
was run to a cancellation, and the numbers are the round's answer to "where does
the chain stop in practice": MF_4 at (F, q') = (88, 41) is 1,566,576 states /
2,605,925 edges and starts machine-free at 1,310 against a budget of 129. Fed
the two-gap output of the rung below - F_2(37) <= 90, from R56's survivor
identity at machine 31 and independently from Mechanic's CRT+SAT, i.e. EXACTLY
the brief's chain shape - it reduces to 799,184 states / 564,661 edges and a
start bound of 258. Two refinement iterations took it 258 -> 235 in 94 queries
and 132 s, and it was cancelled there. THE COST IS ENTIRELY ARITY 2: at machine
37 an over-budget PAIR refutation costs 5.8 s mean / 23.7 s worst (40 sampled)
against 43 ms for a 4-tuple, because a pair leaves only three open points and
therefore much larger gear domains. Nothing about the rung is unsound or stuck -
it is a multi-hour job that was mis-sized for the round, and the mis-sizing is
recorded as such. Log: research/data/chain_37.log (closing record at the end).

**R63 (THE A_4 ENGINE - the exact value, not just the budget).**
`research/chain_a4.py` builds R49's history abstraction over the scan-free
D_3/D_4 and closes it:

    machine   states  edges   A_4 -> F(M+q')   corpus   budget  verdict
    19         2,432    380         34           34       48    EXACT
    23         6,270    968         43           43       63    EXACT
    29        14,368  3,513         58           58       74    EXACT

The m29 system is R49's system exactly (3,513 edges). So the exact F needed to
size the NEXT rung is an OUTPUT of this one: the ladder is self-propelling and
consumes only the gear list.
GATE PROVENANCE, stated exactly: the m19 and m23 rows are from the post-bug-fix
rerun (research/data/chain_a4_regate.log); the m29 row is from the pre-fix run
(research/data/chain_a4_19_29.log), which passed its own assertion (bound >=
corpus, printed EXACT) and reported 14,370 states - two MORE than the 14,368
above, being the two tooth-copies of the phantom state (1,1,1) the fix removes.
Those two states carry no edge (cls(1) = 9 at q' = 31, so no T3-legal
transition), the edge count 3,513 is identical pre and post fix at m29 exactly
as it is at m19 (380) and m23 (968), and the closure is therefore unchanged at
58. The post-fix m29 rerun was still in its level-4 dictionary build (a ~14 min
job running ~6x slow against six other lanes' processes) when the round closed;
it is a redundant confirmation, not a missing gate.

**R64 (THE TWO-GAP LAW IN COVERING FORM, AND THE THREE INSTRUMENTS).** R58's
sweep left the obligation as exactly F_2(M) <= F(M) + q'. R60 restates it with
no algebra in it at all:

  For every pair (g1,g2) with g1+g2 > F(M)+q', the system "0, g1, g1+g2 open,
  all g1+g2-2 interior points covered by the gears <= y" is INFEASIBLE.

`research/twogap_threshold.py` measures the three machine-free instruments that
act on that form.
* CAPACITY: kills 5 of 15 over-budget pairs at m23 and 0 of 3 / 78 / 231 /
  1,128 / 1,176 at m19 / m29 / m31 / m37 / m41; worst-case cap/need climbs
  1.33, 0.98, 1.04, 1.11, 1.15, 1.20. It kills only pairs with BOTH gaps near
  F, never the asymmetric ones. NOT a supplier - but NOT vacuous either, which
  the local form X12 did not distinguish.
* FIRST MOMENT (independence, rho = prod(1-2/q)): the model gets the two-gap
  law RIGHT at every machine - model increment F_2-F = 5,6,9,11,12,14,16,18,19
  against q' = 13..43 (ratio a flat 0.35-0.48), true increment
  4,5,7,6,5,12,10,2,12 (ratio 0.05-0.39). Unlike the histogram (X35) and the
  corridor (X34), which saturate at 2F, INDEPENDENCE DOES NOT SATURATE. It is
  the first machine-free instrument that gets layer 0 right.
* ASYMPTOTICS (closed form, model only): solving E_1 = E_2 = 1 gives model
  increment ~ log(F)/log(1/(1-rho)) = O(log^3 y) against a budget q' ~ y.
  Measured incr/q': 0.385 (y=11), 0.103 (1,487), 0.047 (5,261), 0.019 (20,509),
  0.0145 (29,917) - DECAYING. In the model the two-gap statement gets easier
  without bound: R31's "deeper cases are the easier ones", now for LAYER 0 and
  in closed form.

**R65 (CROSS-LANE: LATERAL'S MIRROR PARITY LAW VERIFIED SCAN-FREE, AND THE
F_2 = 2F ENDPOINT DECIDED).** Lateral's round-25 law says every adjacent EQUAL
pair (g,g) occurs an EVEN number of times except at one self-mirror value k_1.
`research/mirror_parity_lever.py` counts every equal pair exactly by CRT
enumeration - no period - and confirms it at six machines: the odd-count value
is unique and equals g = 3, 3, 5, 5, 5, 7 at m11..29 (counts 1, 13, 649,
10,965, 219,553, 1,739,485), always strictly below F. Two further results:
* (F,F) has count ZERO at all six machines - the F_2 = 2F endpoint that the
  histogram and corridor bounds both point at is NOT ATTAINABLE;
* the LARGEST equal adjacent pair is far under budget: max g with (g,g)
  realised = 5, 7, 8, 8, 15, 20, so 2g = 10, 14, 16, 16, 30, 40 against budgets
  20, 28, 37, 48, 63, 74 - slack 10, 14, 21, 32, 33, 34, GROWING. The equal-pair
  channel is not where F_2 lives (F_2's maximisers are asymmetric: (20,35) at
  m29, (5,34) at m23, (2,88) at m37).
Lateral's lever ("cap the count of adjacent (F,F) at one and parity gives
zero") is therefore live and, at these machines, already discharged by direct
count; `crt_dict.count_solutions` supplies the capped count at any machine.

**R66 (THE COST CURVE, and where the chain stops).** Pinning F(M) exactly from
the previous rung's certified bound F-bar = F(M_prev)+q (i.e. refuting every
one-gap value from F-bar down to F+1):

    machine    F-bar   F   refutations   worst single   total
    19           37   25       12           0.0 s        0.0 s
    23           48   34       14           0.0 s        0.2 s
    29           63   43       20           0.1 s        1.8 s
    31           74   58       16          13.2 s       87.5 s
    37           95   88        7          47.6 s      257.6 s

Single-gap refutation cost grows roughly x15 per added gear (m29 < 0.1 s,
m31 ~1 s, m37 10-20 s, m41 > 250 s undecided at a 6e7-node budget). Arity 2 and
arity 4 queries are far cheaper than arity 1 at the same machine (more open
points means tighter gear domains): 3 ms at m29 against 100 ms for a one-gap
refutation there.

**R67 (THE RESIDUE - the round's primary deliverable).** The chain does NOT
break on an arithmetic fact. Every rung M -> M+q' is a finite, decidable
computation whose only input is the list of primes up to y: the dictionary is
generated, F(M) and F_2(M) are max spans in it, A_4 gives F(M+q') exactly, and
the CEGAR certificate proves the budget. There is no step at which the chain
must reach outside itself for an integer. What it cannot generate is not a fact
about any one machine but the three quantities that decide whether the finite
computation SUCCEEDS, none of which is an output of any rung:

  (i)  THE ORDER. Nothing says which m makes A_m nilpotent and tight at machine
       M. It is m > A_relax(M), and A_relax is non-monotone (1,2,2,3,2,3,4,3,2
       at m11..41, R45); the plain system needs m = 4 and the survivor system
       m = 5 at every step measured, but that is a measurement, not a theorem.
  (ii) THE QUERY COUNT. 181, 90, 955, 3,399 at the four certified rungs -
       growing, and bounded by nothing proven. R59 named this and it is still
       open.
  (iii) THE DECISION COST. One realisability refutation is an exact CSP
       refutation whose cost grows ~x15 per gear at arity 1.

So the two-gap law's irreducible core, stated as sharply as this round can:
**every INSTANCE of the two-gap statement is a finite covering refutation the
chain generates itself; the STATEMENT - one bound valid at every y - is not,
because the chain has no y-uniform control of its own order, query count or
refutation cost.** R64 locates the missing argument precisely: the first-moment
model already proves the statement with a polylog-versus-linear margin, so what
is missing is not a sharper inequality but an unconditional TRANSFER of that
first moment - a large-deviation bound on the machine's own covering system -
which no rearrangement invariant (X35) and no congruence potential (X32) can
perform. That is Wall V with its scale named.

**NEGATIVES AND SELF-CORRECTIONS (round 25).**
* A REAL BUG, caught by a cross-lane cross-check, not by my own gates:
  `decide_cover` returned True for a pattern with EMPTY interior before testing
  the open-point domains, so the all-ones tuples (1,1), (1,1,1), (1,1,1,1) were
  wrongly called realised. Found because the scan-free D_4(23) had 15,697
  tuples against Mechanic's 15,696 and the single extra was (1,1,1,1). Fixed;
  post-fix the two dictionaries are SET-EQUAL. Nothing reported ever depended
  on it (those tuples carry no T3-legal edge and never attain a maximum; the
  only visible trace was A_4(29) reporting 14,370 states where R49 has 14,368,
  the two tooth-copies of the phantom state (1,1,1)). LESSON: the gate that
  caught it was an independent lane's census, not any assertion I wrote - a
  scan-free method needs a scanned counterpart to check against for exactly as
  long as one exists.
* A SECOND SELF-CORRECTION IN THE SAME SCRIPT: the first version of the closure
  wrote `np.maximum(new[usrc], seg, out=new[usrc])`, which writes into a
  fancy-index COPY, so the relaxation silently did nothing and 19->23 "certified"
  at 48 in 3 queries from an initial bound of 50 instead of 111. Caught within
  one run because the shadow gate compares against R58's known 181.
* R59's "every query is arity <= 5 and span <= F_2" was written without
  measuring the spans. MEASURED: max query span 90, 105, 127 at the three
  steps, against F_2 = 31, 39, 55 and 2F = 50, 68, 86. The arity claim holds;
  the span claim was wrong by a factor of 2-3. It does not change the
  conclusion (the queries are cheap because they have MANY open points, not
  because they are short) but the stated reason was wrong.
* The brief's chain shape - A_5(23) -> F_2(29) -> certify 29->31 -> A_5(29) ...
  - is superseded by its own success: once the dictionary is scan-free, the
  two-gap integer does not have to DESCEND from the step below, it is computed
  at the machine itself (F_2(29) = 55 in 75 s at m29 directly). R56/R59's
  descent remains the right structure for a UNIFORM argument, where no
  computation is available; it is no longer needed for a computed rung.
* Compute: the first machine-37 CEGAR run (top-k = 1, sequential oracle) was
  cancelled after 2 iterations and relaunched with top-k batching and a 5-way
  parallel oracle. Reason: measured 10 s per refinement iteration against an
  estimated thousands of iterations. Recorded as a cancellation, not a result.

**PREDICTION SCORECARD (research/data/r25_prediction.txt, written before any
round-25 script ran).**
* P1 ORACLE AGREEMENT - CONFIRMED. 181 + 90 + 955 queries, 100% agreement with
  the round-24 dumps, no disagreement in either direction.
* P2 QUERY SPAN (max < 2F, median < 1.2F) - REFUTED. Max span 90, 105, 127
  against 2F = 50, 68, 86 (167 against 116 at 31->37); medians 45, 57, 84
  against 1.2F = 30, 41, 52. Both halves wrong, in the same direction.
* P3 SCAN-FREE CERTIFICATION within 2x of 181/90/955 - CONFIRMED, and stronger
  than predicted: the counts are IDENTICAL, not merely close (the loop is
  deterministic and the oracle agrees, so it must be - which I should have
  predicted).
* P4 ORACLE COST (median < 10 ms, worst < 5 s at m19..37) - HALF CONFIRMED.
  True for the chain's own arity-2/arity-4 queries at every machine through 37.
  REFUTED for arity 1: 13.2 s worst at m31, 10-20 s at m37, > 250 s undecided
  at m41. The distinction between arities was not in the prediction.
* P5 CHAIN REACH (at least 31->37) - CONFIRMED for 31->37.  The 37->41 half
  ("better than even odds") is UNRESOLVED: cancelled on cost at bound 235
  against budget 129, and the cost is measured (R62b), not guessed.
* P6 THE RESIDUE - PARTLY CONFIRMED. (i)-(iii) of the guess were right: F, F_2
  and the certificate are all generable, so the residue is not an integer. The
  guess "the residue is the termination of the refinement loop" is right as far
  as it goes but incomplete - R67 names three quantities, not one, and the
  round adds the sharper statement that the missing object is a first-moment
  transfer, which the prediction did not anticipate.
* P7 (an engineering wall at m41 from state-space size) - REFUTED as stated:
  MF_4 at m41 is 1,723,138 states / 2,424,036 edges and builds in 15 s. The
  wall at m41 is the ORACLE (arity-1 refutations), not the state space.

**NEEDS / NEXT CONSTRUCTS.**
* THE UNIFORM ORDER. Prove A_relax(M) <= 4 (or any bound) for all M, or find
  the machine where A_4 first fails to certify. This is (i) of R67 and it is
  now a decidable question at any single machine.
* THE QUERY COUNT. Bound the number of refinement queries in terms of y. The
  measured 181, 90, 955, 3,399 is not monotone in y and the 23->29 dip is
  unexplained.
* THE FIRST-MOMENT TRANSFER. R64 says the two-gap law holds in the
  independence model by a polylog-vs-linear margin. The construct that would
  close (D) is a second-moment or large-deviation bound on the covering system
  making that margin unconditional. NOT ATTEMPTED THIS ROUND - it is
  derivation-grade and the escalation valve applies.
* FORMALIST HANDOFF. Each CEGAR deletion is a finite CRT refutation with an
  explicit certificate (empty gear domain, or an uncoverable interior point, or
  an exhausted search). The 19->23 rung is 181 such refutations plus a
  33,038-state closure - a kernel-checkable object with no scan behind it.


---

## Constructor round 26 - THE EXACT RECORD LAW, THE COST OF A RUNG, AND THE EIGHTH RUNG SCAN-FREE

Brief: (a) prove or refute Mechanic's Q* conjecture at the scannable steps;
(b) bound the CEGAR query count as a function of y (R67 item ii);
(c) extend the chain 37->41 with the oracle-cost fix.
Pre-registration: `research/data/r26_prediction.txt`, written before any
round-26 script ran; scored at the end of this append.

**R68 (MECHANIC'S Q* CONJECTURE IS A THEOREM - AND THE PROJECT ALREADY HAD
IT. Verified exactly at EIGHT steps.)** Mechanic's word-legal maximum

    Q*_J(M; legal for q') = max span of a J-gap window of M whose J-2 middle
    gaps are each 0 or +-s mod q' and induce a letter word of prefix-sum
    range <= 1

is, definitionally, R46's qualmax_J = LAYER J-2 OF THE KLEENE STAR K*:
condition (i) is K's edge condition `d mod q' in {0, a, b}`, condition (ii) is
K's T3 tooth transition ("stays inside the two teeth"), and the two free
flanks are K*'s L and R. R46's identity F(M+q') = L (x) K* (x) R - proved BOTH
ways in round 22 - therefore says exactly max_J Q*_J = F(M+q'), and the
direction Mechanic left open is R46's (>=) half, the CRT choice of the killing
copy. Stated without max-plus:

  ATTAINMENT THEOREM. If consecutive openings x_0 < ... < x_J of M have a
  legal middle-gap word then x_J - x_0 <= F(M+q').
  PROOF. Legality is exactly the existence of a tooth assignment t_1..t_{J-1}
  with x_{i+1} - x_i = (t_{i+1} - t_i)c (mod q'), c = 6^{-1} mod q'. Fix one,
  set r = t_1 c - x_1 (mod q'); the joint period is P(M) q' with
  gcd(P(M), q') = 1, so some translate x + jP(M) with jP(M) = r (mod q') is a
  window of M with the same gaps in which q' blocks EVERY interior. No opening
  of M+q' is then strictly inside, so the containing gap of M+q' is at least
  the span (longer still if q' also kills an endpoint). []
  J = 2 IS MECHANIC'S OWN DELETION LADDER F_2(M) <= F(M + one gear); the
  theorem is its extension to every depth.

COMPUTED INDEPENDENTLY at eight steps (`research/qstar.py`, log
`research/data/r26_qstar.log`, all assertions green). The "for every J"
quantifier is closed WITHOUT a depth cap or a span cap: every real legal
window maps to an A_4 walk of the same weight (R49 soundness), so the A_4
closure bounds max_J Q*_J over ALL J at once, and A_4's layer vector
terminates, which rigorously caps the depth. Per-depth values are then exact
by descending-span search seeded at the layer bound.

    M    q'   F(M)  F(M+q')  Q*_2 Q*_3 Q*_4 Q*_5  max_J Q*_J  J*    verdict
    11   13     7      11      11    8    -    -         11    2     EXACT
    13   17    11      18      16   18    -    -         18    3     EXACT
    17   19    18      25      25   25    -    -         25    2,3   EXACT
    19   23    25      34      31   33   34    -         34    4     EXACT
    23   29    34      43      39   43    -    -         43    3     EXACT
    29   31    43      58      55   58   55   55         58    3     EXACT
    31   37    58      88      68   85   88   68         88    4     EXACT
    37   41    88      91      90   90   91    -         91    4     EXACT

The eighth row is NEW - Mechanic's round-25 m37 census makes 37->41 decidable
and no Q* computation had ever reached it. J* = k_win + 1 at every step, and
the attaining windows are the project's own extremal objects re-derived:
(4,8,15,7) at 19->23 is R41's k=3 record window, (7,10,21,10,7) at 29->31 is
R50's exact J=5 inventory, (11,12,37,28) at 31->37 is R25's padded winner,
(2,88) at 37->41 is Mechanic's F_2(37) maximiser. NEW DATUM: F(41) = 91 is
attained at the window (21,14,41,15).

TWO CONSEQUENCES, one for each side of the ledger.
* FOR MECHANIC'S TWO SEEDED ANCHORS: their 43->47 and 47->53 runs were seeded
  at budget-1 and reported <= 149 and <= 170. The theorem gives the true
  values outright - max_J Q*_J(43; legal for 47) = F(47) = 118 and
  max_J Q*_J(47; legal for 53) = F(53) = 145 - so both certify with margins
  32 and 26, not +1, and no span cap is involved.
* A NEGATIVE THAT MUST BE SAID: because Q*_max EQUALS F(M+q'), "the word-legal
  criterion certifies (D)" is NOT a weaker statement than "(D) holds" - it is
  the same statement in another representation. The criterion's reported
  margins (0.52-0.69 q') are the TRUE margins of (D), not slack in a
  relaxation, and no proof can come from exploiting looseness in the
  criterion, because there is none. Q*'s value is entirely that it is computed
  on the OLD machine and never builds M + q'.
* A CORRECTION TO THE RECORD: R39 reported the criterion value equal to
  F(M+q') "at 6 of 7, slack 2 at 23->29". The exact computation gives
  max_J Q*_J(23; legal for 29) = 43 = F(29), attained at J = 3 by the window
  (10,10,23). The slack was an artifact; equality holds 7 of 7 - as the
  theorem requires, at every step, for ever.
Docs: `docs/novel/kleene-generator.md` section 4c (the theorem + the table),
`docs/novel/old-machine-spectrum.md` section 9 (the resolution, in Mechanic's
own vocabulary).

**R69 (THE QUERY COUNT IS BOUNDED - R67 item (ii) was too pessimistic, and the
question was aimed at the wrong object).** Two results.

(a) A PROVED CAP, machine-free. The loop only ever asks whether a value
4-tuple carried by a live MF_4 EDGE is realised, or whether the (flank, base)
pair of a live MF_4 STATE is; every answer is memoised, so no tuple is asked
twice. Hence FOR ANY STRATEGY WHATEVER

    queries  <=  T_4(F, q') + T_2(F, q')  <=  F^4 + F^2,

T_4, T_2 being the distinct value-tuple counts of MF_4, computable in advance
from the two integers (F, q') with no machine in them.  Measured
(`research/query_cap.py`, all asserted; T_4 = 68,578 at 29->31 reproduces
R53's independently measured number):

    M    q'   F   queries    T_4      T_2      cap      used %
    11   13    7        0        43       36       79    0.00
    13   17   11        6       342       88      430    1.40
    17   19   18       18     2,353      243    2,596    0.69
    19   23   25      181    14,730      475   15,205    1.19
    23   29   34       90    26,306      862   27,168    0.33
    29   31   43      955    68,578    1,398   69,976    1.36
    31   37   58    3,399   232,374    2,547  234,921    1.45
    37   41   88    2,879 1,128,323    5,871 1,134,194   0.25

So R67's "bounded by nothing proven" is superseded: the count IS bounded, by a
machine-free function of the step's own parameters, and the measured usage is
0.25-1.45% of it - a 5.8x band, the TIGHTEST correlate of the eight tried
(q/F^2 spans 20x, q/MF_4-edges spans 10x). T_4 ~ 0.02 F^4 at the four largest
machines, so queries ~ (5e-5 to 5e-4) F^4 measured.

(b) THE QUERY COUNT IS A PROPERTY OF THE STRATEGY, NOT OF THE STEP - so it was
the wrong object to ask for a law about. At 37->41 THE SAME RUNG costs 2,879
queries (topk = 1, no given integer), 12,695 (topk = 256, no given integer) or
5,771 (topk = 256, given F_2(37) <= 90). ORACLE-independence, by contrast, is
exact and was checked rather than assumed: run with Mechanic's full-period
censuses in place of the CRT decider, the loop reproduces round 25's counts
181 / 90 / 955 / 3,399 DIGIT FOR DIGIT at 19->23, 23->29, 29->31, 31->37
(`research/query_law.py`, asserted). Three new small steps: 11->13 costs ZERO
queries (MF_4 machine-free already certifies, 15 <= 20), 13->17 costs 6,
17->19 costs 18. The ladder is 0, 6, 18, 181, 90, 955, 3,399, 2,879 - NOT
monotone, and now non-monotone TWICE (the 23->29 dip of round 25, and a new
dip at 37->41).

**R70 (THE STRATEGY-FREE OBJECT: THE IRREDUNDANT CERTIFICATE).** What a kernel
proof must carry is not the queries but the DELETIONS - the set of
realisability refutations that brings MF_4's closure to the budget. That set
can be minimised: restore one member, re-close, and drop it permanently if the
bound still clears (restoring can only RAISE a max-plus closure, so the test
is exact; the result is irredundant, hence an upper bound on the true
minimum). `research/cert_minimise.py`:

    step        queries  greedy deletions  IRREDUNDANT  ratio
    11 -> 13          0             0            0        -
    13 -> 17          6             6            2      3.0x
    17 -> 19         18            18           11      1.6x
    19 -> 23        181           163           76      2.1x
    23 -> 29         90            90           58      1.6x
    29 -> 31        955           905          712      1.3x
    31 -> 37      3,399         3,235        2,189      1.5x
    37 -> 41      2,879         2,769        2,077      1.3x

THE DIP SURVIVES MINIMISATION: 23->29 needs 58 refutations against 19->23's 76
even though its machine is bigger, and 37->41 needs 2,077 against 31->37's
2,189. So the non-monotonicity is a fact about the STEPS, not an artifact of
the greedy search - the certificate cost tracks the added gear's arithmetic,
exactly as A_kill, A_relax and litcap do (R45). FOR FORMALIST: the eighth rung
is 2,077 finite CRT refutations plus one closure.

**R71 (THE EIGHTH RUNG: 37 -> 41 CERTIFIED, SCAN-FREE, WITH NO GIVEN
INTEGER).** R62b left this rung uncertified and named the blocker correctly -
oracle cost, not state space. The fix is to separate the two things the oracle
does. `research/chain_dict_oracle.py`:

  PHASE 1 - run the loop with Mechanic's EXACT full-period m37 4-tuple census
  as the oracle. Its induced level-2 projection is the exact realised-PAIR set
  (every consecutive pair sits inside some 4-window), and its induced F = 88,
  F_2 = 90 are asserted against the corpus before anything is deleted on its
  say-so. CERTIFIED: bound 129 <= budget 129, 12,695 queries, 204 iterations,
  63 s - with NO given integer at all.
  PHASE 2 - re-prove every one of the 12,587 deletions with the scan-free CRT
  decider, in parallel, now that the list is known in advance instead of being
  discovered one at a time inside the loop.

    RESULT: 12,587 of 12,587 deletions re-proved unrealised by CRT.
            CONTRADICTIONS with the census: 0.   UNDECIDED: 0.
            10,859 s of CPU, 2,174 s wall on 5 workers; mean 0.86 s,
            median 0.30 s, worst 25.1 s.

So the rung is a SCAN-FREE certificate: the census was used only to CHOOSE
which refutations to attempt, and every one of them is independently
established by CRT arithmetic from the gear list. Log
`research/data/r26_chain41_phase2_nof2.log`. The cost split confirms R62b's
diagnosis quantitatively - solving 1191x + 11396y = 10,859 with the measured
median gives x ~ 5.8 s for a pair and y ~ 0.35 s for a 4-tuple, i.e. Mechanic-
free reproduction of round 25's measured 5.8 s arity-2 figure (inference from
the aggregate, labelled as such).
CROSS-CHECK, independent vehicle: A_4 over the same census returns
F(41) = 91 EXACTLY (60,650 states, 12,843 edges) - so the ninth rung's input
integer is an output of the eighth, as the self-propelling ladder requires.

**R72 (THE NINTH RUNG 41 -> 43: NOT CERTIFIED, AND THE FAILURE SHAPE IS THE
FINDING - PLUS F_2(41) EXACT AS A BY-PRODUCT).**
* A SUPERSET IS A SOUND ORACLE, and this is worth stating because it is not
  obvious: the loop only ever ACTS on a NO, and "t is absent from a superset
  of the realised set" implies "t is unrealised". So Mechanic's m41 transfer
  dictionary (4,239,676 tuples, exact at depth 1) licenses genuine deletions.
  Run alone it STALLS at bound 222 against budget 134 after 2,452 queries and
  2,143 deletions in 38 s: the transfer dictionary is too inflated at arity 4
  to finish the job. HONEST NEGATIVE, gated by the run.
* HOW INFLATED, measured: of the 249 arity-4 tuples the stalled run's oracle
  called REALISED, the CRT decider refutes 12 of 12 sampled - mean 3.6 s,
  worst 6.0 s. So a HYBRID oracle (superset first, free; CRT on every YES) is
  exact and removes the cheap refutations from the CRT's workload.
* THE HYBRID RUN WAS MIS-SIZED AND WAS CANCELLED, not completed. Three
  attempts, all recorded: at topk = 16 it accumulated only 1,723 s of CPU in
  55 minutes with 5 workers (poor utilisation - `pool.map` blocks on the
  slowest member of a batch and m41 PAIR refutations cost 30-140 s each) and
  was cancelled; the properly-sized rerun (topk = 256, F_2 = 103, 6 workers)
  was KILLED BY A MEMORY EVENT at 12:35 along with the qual_spectrum job
  (three Application fault records; commit 41.5 of 65 GB, 3.6 GB physical
  free, other lanes running) - cause found and fixed at source, the superset
  being a 4.2M-entry Python int set (~500 MB), now a sorted numpy int64 array
  queried by searchsorted (34 MB); the lean third run went 95 minutes clean
  and was CANCELLED AT ROUND CLOSE with 5,575 s of worker CPU (~1,550 m41
  refutations) against 32 s in the main process. THE RUNG IS ENTIRELY
  ORACLE-BOUND, not closure-bound - R62b's diagnosis one gear earlier,
  reproduced. Closing record in `research/data/r26_chain43_hyb3.log`.
* THE BY-PRODUCT: F_2(41) = 103, INDEPENDENTLY AND NON-CIRCULARLY. CORRECTION
  TO MY OWN FIRST DRAFT OF THIS PARAGRAPH, caught by reading Mechanic's index
  entry rather than assuming: this is NOT a first computation. Mechanic's
  lap-phase transfer already pinned it (docs/novel/README.md: "it PINS
  F_2(41) = 103 with no descent (cap F(43) = 103 free, witness at 103)"). What
  is new is that their pin uses the DELETION-LADDER cap F_2(41) <= F(43) =
  103 - legitimate as a computation, circular as an induction step (X36) -
  whereas the sweep here never mentions F(43): it refutes every candidate pair
  ABOVE 103 outright. Sweeping Mechanic's m41 superset dictionary downwards by
  pair SUM and deciding each by CRT (`research/f2_41.py`; only 36 superset
  pairs have sum > 103, so the sweep is short even at machine 41's cost):
    F_2(41) = 103 EXACT, scan-free and non-circular, witness (75,28);
    refutations at sums 115, 113, 110, 108, 107, 106, 105, 104 (36 pairs,
    0 undecided), log `research/data/r26_f2_41.log`.
  Two methods, two vehicles, same integer. It also SHARPENS X36: F(M+q') -
  F_2(M) is 3 at 29->31, 1 at 37->41 and 0 at 41->43 - the deletion ladder's
  slack is exhausted exactly where the budget slack is growing.

**R73 (CROSS-LANE, FOR FORMALIST'S EIGHTH RUNG: THE m37 QUALIFYING SPECTRUM AT
FLOOR 14, WITHOUT A SCAN).** Formalist's stratified-dictionary vehicle is
blocked on Q_J(37; 14) - the size-floor spectrum at the floor 2u'(41) = 14 -
and a 1.24e12-slot scan is out. `research/qual_spectrum.py` computes it from
the realised-tuple dictionary by the same two-layer method as R68: an A_4
closure with the SIZE FLOOR as the edge predicate gives sound upper bounds at
every depth, and branch-and-bound over A_4 walks with those layer potentials
as an admissible heuristic gives the exact values.

    Q_2 = 88   witness (2,88)              [upper bound  90]
    Q_3 = 90   witness (2,62,28)           [upper bound  97]
    Q_4 = 97   witness (2,37,23,37)        [upper bound 103]
    Q_5 = 103  witness (4,26,14,35,28)     [upper bound 127]
    Q_6 = 110  witness (2,30,32,15,20,13)  [upper bound 146]

Q_2..Q_6 are EXACT, all against the budget F(37) + 41 = 129. Q_7 IS NOT
DELIVERED: the branch-and-bound was still running when the memory event of
12:35 killed the job, and the follow-up probe (does machine 37 realise a run
of FIVE consecutive gaps all >= 14? - which would settle Q_7 = 0 outright)
was still enumerating level 5 at round close and was stopped. What stands for
depth 7 is the sound layer bound Q_7 <= 174. Known: m37 realises 2,839
distinct 4-runs of gaps >= 14, so Q_6 > 0 is not marginal. NOTE FOR THAT LANE, and it
is the interesting part: unlike the WORD-LEGAL family, the size-floor family
does NOT terminate in the abstraction - the floor-14 A_4 layer bounds keep
growing (90, 97, 103, 127, 146, 174, ... , 380 at layer 16), so no depth cap
comes free and every depth needs its own exact computation. That is the
concrete price of using the size floor rather than the kill-word predicate,
and it is the same distinction R68 draws.

**NEGATIVES AND SELF-CORRECTIONS (round 26).**
* MY OWN R61 TABLE IS WRONG AT MACHINE 31. It records the scan-free
  D_2(31) = 1,254 and D_3(31) = 15,020. The correct values are 1,253 and
  15,019: that run predates round 25's own `decide_cover` empty-interior fix,
  so it counted the phantom tuples (1,1) and (1,1,1). Both are now refuted by
  the fixed decider and both are absent from Mechanic's independent
  full-period census (which the round-25 append had no chance to compare
  against, since D_4(31) was deferred). Nothing downstream used them - F_1,
  F_2, F_3 = 58, 68, 85 are unchanged - and MECHANIC'S m31 CENSUS IS CLEAN;
  the defect was mine.
* R67's "(ii) THE QUERY COUNT ... bounded by nothing proven" was too strong a
  negative and is corrected by R69(a): a machine-free cap exists and is
  trivial to prove. What is genuinely unbounded is the RATIO to that cap, and
  the count is anyway strategy-dependent, so the object worth bounding is
  R70's certificate, not the query count. I asked for a law about a quantity
  that has no reason to have one.
* THE 41->43 HYBRID RUN was launched at topk = 16 and cancelled on cost after
  55 minutes; the right shape (large topk, so each parallel batch is big
  enough to hide a 140 s outlier) was identified from the measurement, not
  before it. Recorded as a cancellation.
* I PIPED TWO LONG BACKGROUND RUNS THROUGH `tail`, which buffers, so they
  produced no visible progress for 25 and 10 minutes respectively and one had
  to be killed and relaunched. Trivial, and it cost real round time.

**PREDICTION SCORECARD (`research/data/r26_prediction.txt`, pre-registered).**
* P1 Q* HOLDS at all seven scannable steps - CONFIRMED, and at an eighth.
* P2 attaining depths 2 / 3 / 2&3 / 4 / 3 / 3 / 4 - CONFIRMED exactly, all
  seven, with 37->41 coming in at J* = 4 as the pattern predicts.
* P3 R39's slack-2 at 23->29 is wrong - CONFIRMED (the value is 43 = F(29)).
* P4 Q*_max(43;47) = 118 and Q*_max(47;53) = 145 - NOT CHECKABLE IN THIS LANE
  (it needs Mechanic's j5_multi at those steps); it is a theorem-implied
  prediction and stands as a falsifiable ask for their lane.
* P5 (i) the three small steps under 100 queries - CONFIRMED (0, 6, 18);
  (ii) non-monotone - CONFIRMED, and twice; (iii) report no fit - CONFIRMED.
* P6 the MF_4 SIZE is the best correlate - PARTLY CONFIRMED: the best of the
  eight correlates tried is the MF_4 distinct-tuple count T_4 + T_2 (5.8x
  band), which is an MF_4 size; but the edge count itself is worse (10x), so
  the prediction was right in kind and wrong in the specific statistic.
* P7 37->41 needs MORE queries than 31->37's 3,399 - REFUTED at the canonical
  strategy: 2,879 < 3,399. (It is true at topk = 256, which is exactly why
  P7 was a badly-posed prediction - see R69(b).)
* P8 37->41 CERTIFIES - CONFIRMED (129 <= 129).
* P9 the exact m37 census is the oracle that reaches it, with < 500 pair
  deletions to re-verify - HALF CONFIRMED: the census does reach it, but the
  no-given-integer run leaves 1,191 pair deletions, not < 500 (the
  F_2-seeded run leaves 79). All 1,191 were verified anyway.
* P10 arity 2 remains the cost - CONFIRMED (~64% of phase 2's CPU on 9% of the
  deletions). The D_1 pre-filter half was NOT TESTED (the dictionary oracle
  subsumes it).
* P11 at least one of P1-P10 refuted, and named - CONFIRMED (P7; P6 and P9
  partial).

**NEEDS / NEXT CONSTRUCTS.**
* THE NINTH RUNG, PROPERLY SIZED. 41->43 needs an exact m41 oracle. Two routes
  are now costed: (i) the hybrid superset+CRT loop at large topk with >= 6
  workers - the arity-2 refutations at m41 (30-140 s) are the whole cost and
  they parallelise; (ii) ask Mechanic for an EXACT m41 4-tuple census by
  transfer-plus-refutation (their transfer dictionary is exact at depth 1
  already; the inflation is at arity >= 2, and R72 measures it at 12/12).
  Route (ii) is the cheaper one and it is their vehicle, not mine.
* THE UNIFORM ORDER (R67 item i) is untouched and remains the sharpest single
  open question the chain has: prove A_relax(M) <= 4 for all M, or find the
  machine where A_4 first fails to certify.
* THE FIRST-MOMENT TRANSFER (R64/R67) remains the derivation-grade item and
  the escalation valve still applies. R68 sharpens what it must deliver: since
  Q*_max = F(M+q') exactly, there is no slack anywhere in the criterion family
  to trade against, and the transfer has to prove the record itself.
* THE INCREMENT-LAW CANDIDATE routed in by the manager -
  F(M+q') - F_2(M) <= min(2u', q'-2u') = 2u' at every LITERAL step, failing at
  the padded 31->37 - is reproduced from R68's witness table: the differences
  are 0, 2, 0, 3, 4, 3, 20, 1, 0 against 4, 6, 6, 8, 10, 10, 12, 14, 14 at
  11->13 .. 41->43, so it holds 8 of 9 and fails exactly at the padded step by
  +8. READING, labelled HYPOTHESIS: 2u' is the smallest positive legal letter,
  so the candidate says "one more link buys at most one small letter over the
  old two-gap maximum, unless the link is padded (worth a full q')". R68's
  witnesses are the record-window decompositions that would test it further.


---

## Constructor round 27 - THE ORDER IS BOUNDED (AND IT IS THE WRONG ORDER), AND THE INCREMENT LAW AT THE DEEP STEPS

Brief: (a) the uniform-order question (R67 item i) - prove A_relax(M) <= 4 for
all M or exhibit the first failure, using Mechanic's phase-saturation ceilings
as an arithmetic function the way R20 handled litcap; (b) the increment law
tested at 41->43, 43->47, 47->53, plus (routed in mid-round by the manager)
their depth-3 reduction, THE TRIPLE INEQUALITY, at 29->31, 31->37, 37->41 and
deeper if the tooling reaches; (c) the ninth rung 41->43 if an exact m41 census
lands from Mechanic.
Pre-registration: `research/data/r27_prediction.txt`, written before any
round-27 script ran; scored at the end of this append.
Gates, all re-run clean at round close and all green:
  research/uniform_order.py   -> all assertions passed  (r27_uniform_order.log)
  research/increment_law.py   -> all assertions passed  (r27_increment_law.log)
  research/triple_41.py --y 37 --q 41 --floor 89
      -> reproduces the exact m37 answer (MAX = 90, witness (28,14,48))
  research/chain_ps.py        -> 0 false kills on all 291,675 realised m37
                                 4-tuples

**R74 (THE UNIFORM ORDER: A_relax(M) <= 5 AT EVERY MACHINE - a theorem, in
closed form, with the six exceptional classes identified as an object the
project already had).** R67 named this "(i) THE ORDER" and both R25 and R26
closed asking for "a bound, any bound, valid at every machine". There is one.

  THEOREM. For every machine M = {5..y} with y >= 7 and q' = nextprime(y),
      A_relax(M) <= 5,
  and A_relax(M) <= 4 unless q' = 37, 53, 83, 127, 157 or 173 (mod 210).
  (For the one smaller machine M = {5}, A_relax = 1 directly: the letter
  b = 5 exceeds F({5}) = 2.)

PROOF, and the whole of it is arithmetic. By Mechanic's round-26 phase
saturation, a word with prefix-sum offsets X occurs nowhere if for some gear g
of M no translate of X fits inside E_g = Z_g minus {+-6^{-1}} - which uses only
the EXPOSED half of the realisability CSP and is therefore a fortiori a
refutation of "m consecutive gaps". For the m-letter alternation
X = {0, a, q', q'+a, 2q', ...}, so X mod g is determined by (a mod g, q' mod g),
and 3a = q' -+ 1 with the sign fixed by q' mod 6, so everything at gears 5 and 7
is a function of q' mod 210. Enumerating all 48 invertible classes:

    PS-order 2 : 24 classes      PS-order 4 :  2 classes  (23, 187)
    PS-order 3 : 16 classes      PS-order 5 :  6 classes  (37, 53, 83, 127,
                                                           157, 173)

Maximum 5 - the theorem. Adding gears 11 and 13 (moduli 2310, 30030) refutes
NOTHING further: 60 of 60 and 720 of 720 refinements of the six classes stay at
order 5. Independently, a direct sweep of every prime q' < 20000 with all gears
of M up to 100 reproduces the same distribution and the same exceptional
residues (the only extra is q' = 7, where gear 7 is not yet in M).

THE SIX CLASSES ARE EXACTLY R20's LITCAP-6 CLASSES, and that is not a
coincidence: by CRT a translate of a point set fits inside the exposed sets of
gears 5 and 7 separately iff it fits inside the corridor E mod 35, so phase
saturation at {5,7} and the literal cap are the same arithmetic seen from two
sides. The only difference is the quantifier over the two starting letters -
litcap MAXIMISES over them (it is about chain existence), the order MINIMISES
(one broken window kills the cycle). Two invariants the project found
independently, five rounds apart, are one object.

GATES. The refuter reproduces phase-saturation-arity.md's alternation-ceiling
column exactly at all eight steps 31->37 .. 61->67 in Mechanic's own convention
(the phase starting with s = 2*6^{-1} mod q'; the OTHER phase differs at three
of the eight steps, which is why the two numbers must not be conflated), and
refutes none of the seven words on the project's realised record.

A SELF-CORRECTION TO MY OWN R45 TABLE, and it is what made the question
answerable. A_relax recomputed EXACTLY at nine machines from full-period data
(direct cyclic scans m11..m23, Mechanic's exact 4-tuple censuses m29/31/37,
R45's exact CRT count at m41):

    machine   11 13 17 19 23 29 31 37 41
    A_relax    1  2  2  3  2  3  4  2  2     round 27, every entry from data
    R45 table  1  2  2  3  2  3  4  3  2     m37 entry was an ASSUMPTION

`arity_ladder.py` hardcodes "the 1- and 2-letter alternations are realised" at
machines 29, 31 and 37 instead of looking them up. At m37 that is false: gear 5
refutes (14, 27) by phase saturation, and Mechanic's exact m37 dictionary
confirms it is absent. Doc: `docs/novel/uniform-order-bound.md`.

**R75 (THE COMPANION NEGATIVE, AND IT IS THE SHARPER HALF: A_relax IS NOT THE
ORDER THE CHAIN NEEDS, AND THE ORDER THE CHAIN NEEDS CANNOT BE CAPPED THIS
WAY).** A_relax tests ONE candidate cycle. A_m is nilpotent - hence bounds
anything at all - only when EVERY legal cycle is broken, and padded letters
(= 0 mod q') are transparent to T3, so mixed padded/literal cycles are legal
too. Define N(M) = smallest m at which A_m is acyclic; then
max(2, A_relax) <= N <= A_res. Measured exactly this round from the same
dictionaries (small explicit graphs, cycle detection):

    machine   11 13 17 19 23 29 31 37
    A_relax    1  2  2  3  2  3  4  2
    N          2  2  2  3  2  3  4  3

R49's identity N = max(2, A_relax), which held 7 of 7, is REFUTED at the eighth
machine: A_relax(37) = 2 but N(37) = 3. The extra order is bought by a PADDED
cycle - machine 37's legal value set is {14, 27, 41, 55, 68, 82} and 41, 82 are
T3-transparent, so a cycle 14 -> 41 -> 27 -> 41 -> 14 alternates legally where
the pure alternation (14,27) does not even occur. So the corrected A_relax table
changes a PROXY, and the proxy is now known to be lossy.

AND THE METHOD DIES AT A LOCATABLE STEP. A cycle in A_m needs arbitrarily long
runs of legal gaps, so N(M) <= 1 + (longest realisable T3-legal word). Gears 5
and 7 refute a word iff its prefix-sum walk leaves the corridor E mod 35, so
    CORRCAP(q', F) = longest T3-legal word, values <= F, staying in E mod 35
is the strongest cap those gears can EVER give. Exactly (30 states = corridor
residue x tooth; INFINITE means the graph has a cycle):

    step     F(M)  F/q'  CORRCAP        step     F(M)  F/q'  CORRCAP
    19->23     25   1.1      4          37->41     88   2.1     25
    23->29     34   1.2      2          41->43     91   2.1     25
    29->31     43   1.4      3          43->47    103   2.2     11
    31->37     58   1.6      5          47->53    118   2.2      5
                                        53->59    145   2.5   INFINITE

and INFINITE at every larger ratio tested (F/q' = 3.3, 4.5, 7.0, 13.7). The
mechanism is explicit: the padded letters contribute steps j*q' mod 35, and
since gcd(q',35) = 1 those fill Z_35 as soon as F/q' is large enough, so the
corridor stops constraining. F/q' grows without bound along the chain (F is
Jacobsthal-scale, q' ~ y), so NO FIXED SET OF SMALL GEARS CAN EVER CAP THE
ORDER AGAIN. The first step where it fails is 53 -> 59.

VERDICT ON THE BRIEF'S QUESTION, stated plainly: phase saturation alone DOES cap
A_relax uniformly (at 5, not 4), and does NOT cap N. The uniform order, correctly
posed, is a question about the COVER half of the realisability CSP - "every
interior point blocked" - and no bounded gear set can supply that half
uniformly, because larger machines make covering EASIER, not harder. That is
where R67(i) actually lives, and it is not where I was looking.

**R76 (THE INCREMENT LAW AT THE THREE DEEP STEPS: HOLDS, 3 OF 3 - AND ONE OF
THEM NEEDS NO COMPUTATION AT ALL).** The manager's hypothesis is
F(M+q') - F_2(M) <= s_min(q') = min(2u', q'-2u') = 2u' (a = 2u' is always the
smaller letter, since a ~ q'/3 and b ~ 2q'/3).

    step        F(M+q')   F_2(M)     diff   s_min   verdict
    41->43         103       103        0      14   HOLDS
    43->47         118   [103,118]   <= 15     16   HOLDS - by monotonicity
    47->53         145       134       11      18   HOLDS

The middle row is the interesting one. F_2(43) is not on record (the corpus has
103 <= F_2(43) <= 118) and computing it at machine 43 is expensive - but it is
not needed. F_2 IS NON-DECREASING IN THE MACHINE (adding a gear deletes openings
and can only merge gaps, so every adjacent pair of M is a sub-sum of an adjacent
pair of M + q'), hence F_2(43) >= F_2(41) = 103, and 118 - 103 = 15 <= 16.
Together with R68's witness table (0, 2, 0, 3, 4, 3, 20, 1, 0 at 11->13 ..
41->43) the increment law now holds at ELEVEN of the twelve steps the project
can test, failing only at the padded 31->37, by +8.

**R77 (THE MANAGER'S TRIPLE INEQUALITY, EXACT AT EIGHT STEPS - AND THE PADDED
STEP'S FAILURE IS ENTIRELY IN THE PADDED MIDDLE).** Their reduced form: for
every adjacent gap triple (g_L, w, g_R) of M with LEGAL middle,
g_L + w + g_R <= F_2(M) + s_min(q'). Computed exactly (scans at m11..m23,
Mechanic's exact 4-tuple censuses at m29/31/37, projected to triples;
`research/increment_law.py`), with the LITERAL middles (w = +-s mod q') and the
PADDED middles (w = 0 mod q') separated - a separation the manager's five-step
probe never had to make, since no padded middle exists below 19->23:

     M    q'  s_min   F_2   rhs |  literal max  slack  witness   | padded max
    11    13     4     11    15 |      8         +7   (1,4,3)    |  none
    13    17     6     16    22 |     18         +4   (5,11,2)   |  none
    17    19     6     25    31 |     25         +6   (5,13,7)   |  none
    19    23     8     31    39 |     33         +6   (5,8,20)   |   31  +8
    23    29    10     39    49 |     43         +6   (10,10,23) |   40  +9
    29    31    10     55    65 |     58         +7   (18,10,30) |   49 +16
    31    37    12     68    80 |     70        +10   (5,25,40)  |   85  -5
    37    41    14     90   104 |     90        +14   (28,14,48) |   83 +21

  * THE LITERAL TRIPLE INEQUALITY HOLDS AT ALL EIGHT STEPS, the padded
    31->37 INCLUDED (70 <= 80, slack +10). The failure at 31->37 is carried
    entirely by the padded middle 37, exactly as the manager's reading of the
    increment law says it should be.
  * My round-27 P6 was therefore REFUTED AS WRITTEN (I predicted the whole
    inequality holds at 31->37) and CONFIRMED in the form that matters (the
    literal half does).
  * CROSS-VALIDATION: my literal column reproduces the manager's independently
    implemented probe digit for digit at the five steps they ran - 8, 18, 25,
    33, 43. Two implementations, same numbers.
  * THE SLACK SEQUENCE +7, +4, +6, +6, +6, +7, +10, +14 is bounded and GROWING
    at the last two steps, but the sharper reading is on the other side of the
    ledger (R78).

PER-DEPTH (the generalisation the manager asked about), against R68's exact Q*
table: Q*_2 = F_2(M) identically, so the law is a statement about J >= 3, and
Q*_J - F_2 <= s_min holds at 7 of the 8 steps, failing at 31->37 at BOTH J = 3
(+5) and J = 4 (+8). Every failing cell contains the padded letter 37.

**R78 (A FREE REDUCTION OF THE DEPTH-3 OBLIGATION, FOR THE MANAGER'S
DERIVATION).** For any triple of consecutive gaps, (g_L + w) and (w + g_R) are
2-gap windows of M, so both are <= F_2(M). Hence WITH NO HYPOTHESIS AT ALL

    g_L + w + g_R  <=  F_2(M) + min(g_L, g_R),

so the triple inequality is AUTOMATIC at any triple whose smaller flank is
<= s_min, and the entire depth-3 obligation reduces to triples with both flanks
above s_min. Measured, with Delta_3 = max over legal triples of (span - F_2) -
the quantity the law must cap - and the min-flank at the argmax:

     M    q'  s_min  Delta_3  minflank@argmax  free?  #legal  #both flanks>s_min
    11    13     4      -3           1          yes        6        0
    13    17     6       2           2          yes       72        0
    17    19     6       0           5          yes     1088       16
    19    23     8       2           5          yes    11698        4
    23    29    10       4          10          yes   243810       24
    29    31    10       3          18          no       544      131
    31    37    12       2           5          yes      991      205
    37    41    14       0          28          no      1327      317

THE SHAPE, and it is the useful part: Delta_3 is BOUNDED BY A CONSTANT (max 4
over all eight steps, no trend) while s_min grows linearly in q'. The free bound
already discharges 6 of the 8 steps outright. So the depth-3 half of the
increment law is not a tight inequality at all - it is a constant against a
linear function, with the free reduction covering most cases. If the manager's
induction needs a derivable statement, "Delta_3 = O(1)" is the shape to aim at,
not "Delta_3 <= s_min".

**R79 (THE NINTH RUNG: STILL NOT CERTIFIED, AND THE FAILURE IS NOW PINNED TO
THE ORACLE'S INFORMATION CONTENT, NOT TO COST OR STRATEGY).** No exact m41
4-tuple census appeared on disk this round, so brief item (c)'s precondition
never arrived. What was done instead is cheap and it settles a question.

  * PHASE SATURATION AS A FREE ORACLE TIER (`research/chain_ps.py`): superset
    ABSENT -> NO (free); else phase saturation -> NO (free, exact, and it comes
    from no scan); else CRT. Soundness gate: 0 false kills on all 291,675
    realised m37 4-tuples (an independent gate from Mechanic's, which used
    468,418 realised tuples).
  * MEASURED, AND IT IS A NEGATIVE: phase saturation answers 0 of the 27,197
    superset-YES queries the 41->43 loop actually asks. ZERO, not few. The
    reason is structural and worth recording: MF_4 is built MOD 35, so every
    edge it carries is already corridor-admissible, and phase saturation at
    gears 5 and 7 IS the corridor condition (R74's CRT remark). Gears 11 and
    13 fire only at |X| large, which arity-4 queries never reach. So Mechanic's
    "screens 4,239,676 -> 2,814,574 of the dictionary" is real as a screen of
    the DICTIONARY and worth exactly nothing on the loop's own query stream -
    two different numbers, and only the second one is the oracle's cost.
  * THE STALL IS AT 222 UNDER THREE SETTINGS: superset alone (R72), superset +
    phase saturation, and superset + phase saturation + the exact F_2(41) = 103
    (which cuts the state space to 1,058,228 states / 819,629 edges and the
    queries to 3,209, and still stalls at 222). So 222 is the transfer
    superset's INFORMATION LIMIT, not a search artifact and not a budget
    artifact. Only the CRT tier moves it, which is R72's costing unchanged.
    Route (ii) - Mechanic's exact m41 census - remains the cheap route and
    remains their vehicle.

**NEGATIVES AND SELF-CORRECTIONS (round 27).**
* R45's A_relax(37) = 3 IS WRONG (it is 2) and the defect is mine: my own
  `arity_ladder.py` hardcodes the m = 1 and m = 2 entries at machines 29, 31 and
  37 as "realised" instead of looking them up in the census. Nothing downstream
  used the entry, but it is the second time in two rounds that a hardcoded
  convenience in my own script has entered the record as a measurement (R61's
  m31 counts were the first). LESSON, and it is now a standing rule for this
  lane: a table cell that a script FILLS IN rather than LOOKS UP must be printed
  as such, or not printed.
* MY PRE-REGISTERED P1 IS REFUTED: phase saturation does NOT cap A_relax at 4.
  It caps it at 5, and the six classes where it stops at 5 are stable under
  gears 11 and 13. My hand computation P2 (gear 5 alone fails exactly on
  q' = 7, 23 mod 30) was CONFIRMED, which is what let me see that the gap is a
  class phenomenon rather than a machine phenomenon.
* P6 REFUTED AS WRITTEN (the triple inequality does not hold at 31->37 with a
  padded middle) - I predicted the wrong half of a distinction the manager's own
  message had already drawn ("w = +-s (mod q')"). I should have read the
  quantifier before predicting against it.
* I PIPED A LONG BACKGROUND RUN THROUGH `tail` AGAIN - the exact mistake
  recorded in the round-26 negatives, in the same lane, one round later. It cost
  about four minutes of blind waiting on the m37 gate. Background runs write to
  a log file, never to a pipe.
* THE SCRIPT'S OWN VERDICT LINE FOR THE m41 SUPERSET ROW SAYS "FAILS" and that
  is a presentation defect, not a result: 144 is a SUPERSET maximum and R72 had
  already measured the superset as heavily inflated at arity >= 2. The corrected
  reading is the CRT sweep in R80. Flagged here rather than silently patched.
* The two superset runs of `chain_ps.py` first wrote to the same JSON path
  (the F_2 seed was not in the tag), so the unseeded artifact was overwritten
  once before the tag was fixed. Both runs' numbers survive in their logs.

**R80 (THE TRIPLE INEQUALITY AT THE NINTH STEP, SCAN-FREE - and the sweep is a
reusable instrument).** `research/triple_41.py`: sweep Mechanic's m41 transfer
SUPERSET downward by span (a superset is a sound source of candidates, since
every realised triple is in it), halve by Lateral's reversal theorem, apply
phase saturation as a free prefilter, and decide the survivors with
`crt_dict.decide_cover`. Descending order means the first realised triple found
IS the maximum, with every larger candidate refuted.
GATE, at machine 37 where the answer is known from the exact census: MAX = 90
with witness (28,14,48) - identical to the exact-dictionary computation, value
and witness. Two vehicles, one number.
RESULT AT MACHINE 41 (q' = 43, budget F_2(41) + s_min(43) = 103 + 14 = 117),
every candidate decided, ZERO UNDECIDED at either half:

    LITERAL middles: every candidate of span 144 down to 108 REFUTED
                     (37 span levels, 0 realised, 0 undecided) => max <= 107
    PADDED  middles: every candidate of span 130 down to 117 REFUTED
                     (11 span levels, 0 realised, 0 undecided) => max <= 116

    so Q*_3(41; legal for 43) <= 116 < 117 - THE TRIPLE INEQUALITY IS
    CERTIFIED AT 41 -> 43, the NINTH step and the first certified with no
    census of any kind behind it.

Two remarks that are part of the result. (i) The padded half is the tight one
(+1 against the literal half's +10) - the same asymmetry that makes 31->37 the
single failing step in the whole family, now visible one gear later without a
failure. (ii) THE EXACT MAXIMUM IS NOT DELIVERED: the literal descent was
stopped at the span-108 level boundary after one arity-3 instance at span 107
ran past 20 minutes (against a 30-70 s norm at the levels above it). Every
completed level is a self-contained refutation set, so "max <= 107" is exact
and gate-clean; "= 107" is not claimed. The cost of an m41 arity-3 CRT
refutation is NOT uniform in span, and that is a new measurement for the
oracle-cost curve (R66/R72): mean ~40 s, but with a heavy tail that a node
budget does not bound in practice.

MY OWN OPERATIONAL ERROR, recorded because it cost round time: I killed the
first sweep at the span-108 boundary on the strength of a POLL THAT WAS STALE -
the level had in fact completed 156 s earlier and the job was healthy and
working on 107. Nothing was lost that the resumed run could not redo, but the
decision was made on a read, not on the log, and the log was right.

**PREDICTION SCORECARD (`research/data/r27_prediction.txt`, pre-registered).**
* P1 PS-order <= 4 for every q' with gears 5,7,11,13 - REFUTED. It is <= 5, with
  six exceptional classes mod 210 that gears 11 and 13 do not touch.
* P2 gear 5 alone fails exactly on q' = 7, 23 (mod 30) - CONFIRMED exactly.
* P3 the worst case is attained and the bound cannot be improved - CONFIRMED in
  the corrected form (order 4 is attained at classes 23 and 187, order 5 at the
  six litcap-6 classes; neither can be lowered by gears 11 or 13).
* P4 A_relax(37) = 2, not 3, and my R45 table is wrong - CONFIRMED, and the
  corrected ladder is 1,2,2,3,2,3,4,2,2 exactly as predicted.
* P5 the increment law holds at all three deep steps, with 43->47 needing no new
  computation - CONFIRMED, both halves.
* P6 the triple inequality holds at 29->31, 31->37, 37->41 - REFUTED as written
  (31->37 fails with a padded middle) and CONFIRMED for literal middles. The
  predicted slack band [3,15] was right (+7, +10, +14).
* P7 the maximiser is a relaxed-superset object, not a realised record window,
  from 29->31 on - REFUTED: every maximiser in the table is a realised triple of
  the exact census. At depth 3 the relaxation costs nothing.
* P8 rung nine not certified absent an exact m41 census - CONFIRMED.
* P9 at least one of P1-P8 refuted and named - CONFIRMED (P1, P6, P7).

**NEEDS / NEXT CONSTRUCTS.**
* THE ORDER QUESTION, CORRECTLY POSED. Is N(M) bounded? R74 settles the
  alternation; R75 shows the corridor cannot settle the rest and locates the
  step where it stops (53 -> 59). The object to attack is the longest realisable
  T3-LEGAL WORD (= A_res - 1), which is capped by the COVER half of the CSP, and
  the only machine-free handle on the cover half the project has is R43's pruned
  inclusion-exclusion counter. NOT ATTEMPTED THIS ROUND, and it is the natural
  round-28 item for this lane.
* Delta_3 = O(1). The depth-3 obligation's true shape (R78): the largest legal
  triple exceeds F_2(M) by at most 4 at every step measured, with no trend, while
  the budget s_min grows linearly. A derivation of the O(1) statement would give
  the manager's induction its depth-3 base with room to spare, and it is a
  statement about the old machine alone.
* THE NINTH RUNG needs the exact m41 4-tuple census (Mechanic's vehicle, route
  (ii) of R72's costing). The superset's information limit is now measured at
  222 under three settings and is not going to move.
* THE FIRST-MOMENT TRANSFER (R64/R67) remains the derivation-grade item and the
  escalation valve still applies.

# The second moment over arithmetic blocks

Prover lane, round 52. Parent: `research/proof/distortion_method.md` (node R2.c), whose closing
section names one crack in the collapse lemma: *"The collapse is forced only because the partition
is by classes mod `Q_{i-1}`. Any partition of the interval into blocks of `>= g_i` columns on which
the surviving set is near-uniform would restore the second moment ... That is a much weaker
equidistribution statement than a level of distribution, and it is not obviously a sieve. It is the
one crack in the collapse and it should be the next branch."*

Branch node: **R2.c.i**. Scripts in `research/anchor235/r52/` (prefix `bm_`); result outputs,
untracked, in `research/anchor235/r52/results/`. Every number this document relies on is written
here.

**Verdict in one paragraph.** Outcome **(C) with a real positive**: the crack opens where the
survivors are counted and closes where they must be bounded. The block inequality is a valid
theorem for ANY partition of the interval - the whole per-part algebra of BBMST Theorem 3.1 is
partition-agnostic, and the partition may even be re-chosen at every gear - so the collapse lemma
is genuinely avoidable, and on the real machine the gain is large: taking the whole window as one
block, the exact budget is `eta_B = 0.358, 0.359, 0.365, 0.366, 0.365` at `q = 59, 97, 199, 499,
997`, against r51's exact fibre budget `1.074, 1.224, 1.469, 1.707, 1.905` on the same window - a
factor 3.0 to 4.8 - and below 1 at all 107 machines `q <= 599` with no exception (the fibre budget
crosses 1 at `q = 43`). The exact block budget stays below 1 for blocks as short as
`beta_c = 1, 5, 12, 20, 26` columns, thousands of times shorter than the window. But every one of
those numbers is computed from the machine's own survivor counts, and the moment a phase-adversarial
bound is put in their place the budget busts at the SIXTH GEAR, before any tail exists: with the
adversary free to align each gear's two classes with the survivors, the term is
`min(1, (2/g)/Pi_{<g})^2` and the cumulative sum runs `0.160, 0.387, 0.567, 0.759, 0.917, 1.078` at
`g = 5, 7, 11, 13, 17, 19` - independently of the block length, the interval length and `q`. The
one relaxation that would rescue it - assume each block holds its fair share `beta Pi` of survivors
- is FALSE, and its refuting instance is the real machine `{5,7}`, which covers 4 consecutive
columns (`F(7) = 5`) while the local-density block budget at `beta = 4` is `0.944 < 1`. So the block
partition trades r51's tail collapse for a head that costs 3.5 times more: gear by gear the block
term beats the collapsed fibre term only at `g = 5, 7, 11` and is worse from `g = 13` on (ratios
1.25, 1.34, 1.54, 1.59, ...), and the block threshold `beta*` is `3.7e7, 5.1e13, 9.1e25, 4.7e43,
9.4e62` against r51's fibre `L* = 1.06e4, 3.89e6, 1.23e14, 1.90e30, 8.95e52` - worse by 3 to 13
orders of magnitude. The unconditional part of the block budget is three to five gears (those with
`Q_{<g} g <= L`, top gear 11, 11, 13, 13, 17), worth `0.275 .. 0.312`, and the adversary charges
1.77 to 3.19 times the room left over for everything above them, while the real machine uses about
a tenth of it. **The residual is the same level of distribution at dimension 2 that r51 named, but
in a weaker and exactly quantified form**: not "the survivors equidistribute in the classes of the
next gear" but "the weighted `L2` mean of the strike-rate excess `rho_g` stays below
`1/sqrt(sum 4/g^2) = 1.663`", measured `max rho_g = 1.35` at `q = 199` and `1.52` at `q = 997`.

---

## 1. Pre-registered (written before any computation of this round)

### 1.0 The object

The machine `M = {5..q}`: for each prime `5 <= g <= q` the two column classes `+-6^{-1} (mod g)`.
`I` = an interval of `L` consecutive columns (for the real machine, the window
`W(q) = (q'^2-1)/6`). The distortion method's engine (BBMST Invent. math. 228 (2022) Thm 3.1)
processes the gears one at a time, carrying a reweighted probability measure `P_i` on `I`, and its
hypothesis is

    eta = sum_i  min{ M_i^(1),  M_i^(2) / (4 delta_i (1 - delta_i)) }  <  1 ,
    M_i^(1) = E_{i-1}[alpha_i],  M_i^(2) = E_{i-1}[alpha_i^2] ,

where `alpha_i(x)` is the proportion of the PART containing `x` that gear `i` strikes. In BBMST the
parts are the congruence fibres (classes mod `Q_{i-1} = g_1...g_{i-1}`), and r51's collapse lemma
says that once `Q_{i-1} >= L` each part is a single column, `alpha in {0,1}`, `M^(2) = M^(1)`, and
the method degrades to the union bound `sum 2/g`, which busts 1 at four gears.

**This branch replaces the parts by ARITHMETIC BLOCKS**: `I` is cut into `n = ceil(L/beta)`
consecutive runs of `beta` columns, and `alpha_i(B)` is the fraction of block `B` struck by gear
`i` among the columns of `B` still uncovered before gear `i`. Three choices of `beta` are
pre-registered: fixed `beta`; `beta_i = c g_i` (the current gear); `beta_i = (prod_{j<i} g_j)^{1/(i-1)}`
(the geometric mean of the gears used).

### 1.1 The theory

**T (the crack opens).** A block of length `beta >> g_i` keeps `~2 beta/g_i` strikes spread over
`~beta Pi_{i-1}` survivors, so `alpha_i(B) ~ 2/g_i` stays fractional at EVERY gear, however large;
the second moment therefore never collapses and `eta_B` stays near `sum 4/g^2 = 0.365`. If the
price of the change of partition (blocks are not unions of residue classes, so the next gear's
strike count on a block is `2 ceil(beta/g)` up to `+-1` rather than exactly `2 beta/g`, and the
survivors inside a block need not be equidistributed in the classes of the next gear) is smaller
than the gain, the localised budget stays below 1 on an interval far shorter than r51's
`L*(q) = exp(theta(q^{0.73}))`.

### 1.2 Predictions, with numbers, and what refutes each

**E1 (validity: the inequality survives the change of partition).** Predicted: the per-part algebra
of Theorem 3.1 - mass preservation of the reweighting, and the loss bound
`P_i(B_i) <= min{M^(1), M^(2)/(4 delta(1-delta))}` - is PARTITION-AGNOSTIC once `alpha_i(B)` is
defined as the `P_{i-1}`-weighted fraction `P_{i-1}(S_i ∩ B)/P_{i-1}(B)`, so the block inequality is
a valid theorem for ANY partition, and even for a partition re-chosen at every stage. What is lost
is not validity but COMPUTABILITY: in BBMST `alpha_i = 2/g_i` on every fibre by CRT, with no input;
on blocks `alpha_i(B)` is an unknown of the problem. REFUTED if any step of Thm 3.1 needs the parts
to be residue classes for a reason other than computing `alpha`.

**E2 (the exact block budget on the real window is below 1, and is a tautology).** Predicted: with
the real teeth and the exact survivor counts, `eta_B(beta) < 1` for every `beta` at every `q`,
because at `beta = 1` the budget equals the covered fraction of the window and at `beta = L` it
equals `sum_i (fraction of survivors struck by gear i)^2 ~ sum 4/g^2`. Pre-registered numbers:
`eta_B(L) in [0.30, 0.45]` at every `q`; `eta_B(1) = 1 - (openings in the window)/W(q)`, i.e.
`0.84 .. 0.94`. If both hold, the exact block budget carries no information beyond "the window is
not covered", which it assumes. REFUTED if `eta_B(beta) > 1` for some `beta` at some `q` (the
window is not covered, so the theorem forbids it - a value above 1 would mean the budget is not the
loss but a strict over-estimate, which is the interesting case).

**E3 (the a priori block budget busts at the HEAD, not the tail).** This is the deciding
prediction. A theorem needs an upper bound on `alpha_i(B)` valid for every phase vector. The
adversary aligns gear `i`'s two classes with the survivors, so the honest envelope is
`alpha_i(B) <= min(1, 2 ceil(beta/g_i) / s_{i-1}(B))` with `s_{i-1}(B)` the survivor count.
Granting the adversary NOTHING and the method EVERYTHING (`s = beta Pi_{i-1}`, perfect local
density, `beta -> infinity` so the ceiling is free), the term is `((2/g)/Pi_{<g})^2`. Predicted:
that sum passes 1 within the first six gears at every machine, so the a priori block budget is
vacuous for every `beta` and every `q >= 19`. Pre-registered arithmetic:
`0.160, 0.227, 0.180, 0.193, 0.157, 0.162` at `g = 5, 7, 11, 13, 17, 19`, cumulative
`0.160, 0.387, 0.567, 0.759, 0.917, 1.078`. REFUTED if the exact computation puts the crossing
above the sixth gear, or if any `beta` keeps the a priori sum below 1.

**E4 (blocks are worse than fibres at every gear that matters).** The per-gear comparison is
`4/g^2` (both partitions, while the gear is uncollapsed), `2/g` (fibre, collapsed) against
`min(1, (2/g)/Pi_{<g})^2` (block, adversarial). Predicted: the block term is BELOW the fibre term
only at `g = 5, 7` and (marginally) `11`, and above it from `g = 13` on, at every machine; so
taking the better of the two partitions gear by gear gives back exactly r51's fibre budget and no
improvement in `L*`. REFUTED if the block term beats the fibre term at any gear `>= 13`.

**E5 (the phase-adversarial gate).** r51's refuting instance: localising with AVERAGE first moments
gives `A(7) <= 9.3` against the certified `A(7) = 37`. Predicted: the block envelope, being an
upper bound on the loss, passes the gate at every `K <= 12` (`L*_B(K) >= A(K)`) - trivially, since
it is predicted vacuous - and the exact block budget computed on a covered stretch (the real
machine's own record, `F(M)` columns) is `>= 1` at every `beta` and every `m11..m31`, with equality
the interesting case. REFUTED by any `(K, beta)` with a block bound below `A(K)`, or any record
stretch with `eta_B < 1`.

**E6 (gear ordering changes a constant, not an exponent).** Predicted: processing the gears
largest-first changes `eta_B` and `L*_B` by a bounded factor, because the block partition has no
collapse point to move; whereas for FIBRES the ordering moves the collapse point and therefore
`L*` by an exponent (`L*` is the primorial of the cut gear). REFUTED if the block `L*_B` moves by
more than a factor 10 under reordering, or if the fibre `L*` moves by less than a factor 10.

**E7 (what the block inequality proves unconditionally).** Predicted: nothing about an interval
shorter than the capacity bound allows, i.e. the unconditional reach is `sum_{g<=q} 2/g < 1`, four
gears, `L < 20`. The residual is again an equidistribution input - but a LOCAL one (the survivors
of a block of length `beta` are not concentrated in the two classes of the next gear), which is
formally a level of distribution at dimension 2 restricted to a single short interval. Predicted:
quantify how much of the window it covers unconditionally: 0%.

### 1.3 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| E1 | block inequality valid for any partition; only computability lost | **CONFIRMED** | section 2: mass preservation and the loss bound are per-part algebra; the fibre structure is used in BBMST only to EVALUATE `alpha_i = 2/g_i` by CRT. The partition may be re-chosen at every gear (used in R2's variable rules) |
| E2 | exact `eta_B < 1` at every beta; `eta_B(1)` = covered fraction; `eta_B(L) ~ 0.36` | **HALF CONFIRMED, half REFUTED** | `eta_B(1)` = covered fraction EXACTLY (identity, section 2.4), `0.849, 0.880, 0.915, 0.939, 0.951`; `eta_B(L) = 0.358 .. 0.366`, inside the predicted band. But `eta_B > 1` at short `beta` (max `1.311` at `beta = 7`, `q = 997`): the second moment is a STRICT over-estimate of the loss, and the crossing is at `beta_c = 1, 5, 12, 20, 26` (R1) |
| E3 | a priori block budget passes 1 at the sixth gear, every beta, every q | **CONFIRMED to the fourth decimal** | cumulative `0.16000, 0.38676, 0.56674, 0.75924, 0.91646, 1.07813` at `g = 5..19` (R3); the pre-registered guess was `1.078`. No dependence on `beta`, `L` or `q` |
| E4 | block term worse than fibre term from `g = 13` on | **CONFIRMED** | R4: block/fibre `= 0.400, 0.794, 0.990` at `g = 5, 7, 11` then `1.251, 1.336, 1.536, 1.585, ...` from `g = 13`; the block threshold `beta*` is `3.5e3` to `2.5e13` times r51's fibre `L*` |
| E5 | gate passed (vacuously); record stretches give `eta_B >= 1` | **CONFIRMED for the exact budget, REFUTED for the local-density envelope** | exact `eta_B >= 1` at every `beta` on every covered run found at `m11..m31` (60 evaluations, R6); the unconditional CAP envelope passes at every `K <= 12`; the LD envelope is REFUTED, with the real machine `{5,7}` as the instance (`0.944 < 1` at `beta = 4`, 4 columns covered) |
| E6 | ordering: constant for blocks, exponent for fibres | **CONFIRMED** | R7: block `eta_B(L)` moves `0.358 -> 0.344` (`q = 59`) and `0.366 -> 0.409` (`q = 499`), a factor `<= 1.15`; fibre `L*` moves from `1.07e4` to `2.49e17` at `q = 59` and from `1.93e30` to `1.90e202` at `q = 499` |
| E7 | unconditional reach = capacity bound, 0% of the window | **CONFIRMED** | the only unconditional envelope (CAP) is non-vacuous at `K <= 3` only (`beta* = 3, 9, 72`); for every machine from `m13` on it says nothing. The unconditional HEAD of the budget is 3-5 gears (`0.275 .. 0.312` of the budget) and the adversary is `1.77x` to `3.19x` over the room left (R8) |

Two predictions were wrong in an informative direction. E2's "below 1 at every `beta`" is false:
the second moment over short blocks over-estimates the loss so badly that the budget busts on an
interval that plainly contains 627 openings - the gap between `eta_B` and the true loss is the
object the method actually pays for. E5's expectation that the gate would be passed vacuously was
wrong for the LOCAL-DENSITY envelope, which is not vacuous and is false; that refutation is the
branch's reusable filter.

### 1.4 Outcomes, named in advance

* **(A) The crack opens.** Some block rule gives an a priori budget below 1 on an interval
  polynomial in `q`.
* **(B) The crack opens conditionally**, under a named local-density hypothesis weaker than the
  root.
* **(C) The crack does not open, and the mechanism says why.**

**Outcome: (C)**, with the qualification in the verdict paragraph: the partition change is valid
and the exact gain is large (factor 3.0 to 4.8 on the real window, below 1 where the fibre budget
is above 1), but no valid a priori bound survives, and the hypothesis that would supply one is
false at the scale where it is needed.

---

## 2. Setup: the block inequality, what is proved and what is added

### 2.1 The inequality

Let `I` be an interval of `L` columns, `g_1, ..., g_n` the gears in some order, `S_i` the set of
columns gear `i` strikes, `C_i = S_1 ∪ ... ∪ S_i`, `U_i = I \ C_i` the survivors, and
`B_i = S_i \ C_{i-1}` the columns NEWLY covered at stage `i` (BBMST's `B_i`). Fix `delta_i in
[0, 1/2]`. At stage `i` choose ANY partition `Pi_i` of `I` (it may differ from stage to stage);
`P_0` is uniform on `I` and, on each part `B in Pi_i`,

    a := alpha_i(B) = P_{i-1}(B_i ∩ B) / P_{i-1}(B) ,
    P_i = max{0, (a - delta_i)/(a (1 - delta_i))} P_{i-1}   on  B_i ∩ B ,
    P_i = min{1/(1 - a), 1/(1 - delta_i)}         P_{i-1}   on  B \ B_i .

> **Block inequality (BI).** If
> `eta_B = sum_i min{ M_i^(1), M_i^(2)/(4 delta_i(1-delta_i)) } < 1` with
> `M_i^(k) = E_{i-1}[alpha_i^k]` (expectation over the parts, weighted by `P_{i-1}`), then
> `U_n != ∅`: the gears do not cover `I`.

### 2.2 What survives the change of partition (proved)

Three steps, each pure per-part algebra, with no reference to what the part is:

1. **Mass preservation.** On a part of mass `m`,
   `m a max{0,(a-d)/(a(1-d))} + m(1-a) min{1/(1-a), 1/(1-d)} = m` for every `a in [0,1]`,
   `d in [0,1/2]` (both cases `a >= d` and `a < d` give exactly `m`). So `P_i` is a probability
   measure whatever the partition is.
2. **The loss bound (BBMST Lemma 3.3).** `P_i(B_i ∩ B) = m max{0, (a-d)/(1-d)}` and
   `max{0,(a-d)/(1-d)} <= min{ a, a^2/(4d(1-d)) }`, the second because `(a-2d)^2 >= 0`. Per part,
   partition-free.
3. **The distortion (BBMST Lemma 3.5).** `Delta_i = max{0, log(P_i/P_0)}` is bounded stage by
   stage by `log` of the same two factors; in the general-partition form the bound becomes
   `E_i[Delta_i] <= 2 sum_{j<=i} M_j^(1)/(1-delta_j)`, which for the machine is
   `<= 4 sum 2/g = O(log log q)`, finite. On an INTERVAL this constant is irrelevant: `P_0` is
   uniform with full support, so any `P_n(U_n) > 0` already gives `U_n != ∅`.

**The one place BBMST uses the fibre structure is not a step of the proof but an EVALUATION:** a
fibre is a full class mod `Q_{i-1}`, so by CRT it meets each class of `g_i` equally often and
`alpha_i = 2/g_i` on every fibre, for free. That is the whole reason the fibres are chosen. E1
confirmed.

### 2.3 What needs an added hypothesis

The trouble is entirely in evaluating `alpha_i(B)`. Two error terms, named as the brief asks:

* **The ceiling.** A block of `beta` consecutive columns meets each class mod `g` either
  `floor(beta/g)` or `ceil(beta/g)` times, so `|S_i ∩ B| = 2 beta/g_i + theta`, `|theta| <= 2`, and
  the block's struck FRACTION is `2/g_i + theta/beta`. Measured (R8): summing
  `(2 ceil(L/g)/L)^2` instead of `4/g^2` over all gears moves the budget from `0.36404` to
  `0.36407` at `q = 997` - the fifth decimal. **The ceiling is free.**
* **The conditioning.** What the inequality needs is not the struck fraction of the block but the
  struck fraction of its SURVIVORS,
  `alpha_i(B) = |U_{i-1} ∩ S_i ∩ B| / |U_{i-1} ∩ B| = (2/g_i) rho_i(B)`, and `rho_i(B) = 1` only
  if the survivors of the lower gears are equidistributed in the classes of `g_i` INSIDE the block.
  A block is a union of full classes mod `Q_{i-1} g_i` exactly when `Q_{i-1} g_i | beta`, and then
  `rho = 1` is a theorem; otherwise it is an assumption. **`rho` is the added hypothesis, and it is
  a level of distribution at dimension 2 localised to one block.**

Note what that says about the crack: *the blocks on which the second moment is provably fractional
are precisely the blocks that are unions of congruence fibres.* The block partition buys nothing
that the fibre partition does not already have, unless `rho` is supplied.

### 2.4 The measure is uniform on the survivors (proved), and two identities

> **Lemma (support).** With `delta_i = 1/2`, if `alpha_j(B) <= 1/2` at every earlier stage then
> `P_{i-1}` is uniform on `U_{i-1} ∩ B` and zero on `B \ U_{i-1}`, and the block's mass is exactly
> its `P_0`-mass. *Proof.* At `delta = 1/2` the factor on `B_j` is `max{0, 2 - 1/a} = 0` for
> `a <= 1/2`, and the factor off `B_j` is a constant on the part; mass per part is preserved. QED.

So under that condition `alpha_i(B)` is the plain counting fraction the brief defines, and the
computations below are exact integer ratios. Two identities follow and are confirmed exactly:

* **`beta = 1`:** each block is one column, `alpha in {0,1}`, and `eta_B` = the covered fraction of
  the interval. Measured `0.84918, 0.88005, 0.91512, 0.93858, 0.95117` at `q = 59..997`, equal to
  `1 - (openings)/W` to every printed digit. The inequality is then a tautology.
* **`beta = L`:** one block, `alpha_i = |U_{i-1} ∩ S_i|/|U_{i-1}|`, and `|U_n| = L prod(1-alpha_i)`,
  so the conclusion is again immediate. `eta_B = sum_i (2/g_i)^2 rho_i^2` with
  `rho_i = alpha_i g_i/2`.

The method's content lies strictly between the two, and it is the size of the `rho_i` that decides
it.

---

## 3. Results

### R1. The exact block budget on the real window (`bm_exact.py` part A)

Real teeth `u_g = 6^{-1} mod g`, the real window in columns, exact survivor counts, `delta = 1/2`,
gears in increasing order. Full tables in `results/bm_exact.txt`; `q = 199` in full:

    q = 199, window columns 34..7420, L = 7387, 44 gears

     beta   blocks    eta_B   true loss        beta   blocks    eta_B   true loss
        1     7387  0.91512    0.91512          40      185  0.57775    0.02707
        2     3694  1.08197    0.83254          50      148  0.53436    0.01354
        3     2463  1.02256    0.71003          64      116  0.48658    0.00000
        4     1847  1.11904    0.68120         100       74  0.44027    0.00000
        6     1232  1.11529    0.56017         200       37  0.40048    0.00000
        8      924  1.03991    0.43846         512       15  0.37879    0.00000
       10      739  1.03258    0.38085        1024        8  0.37310    0.00000
       13      569  0.94420    0.27571        2048        4  0.36960    0.00000
       16      462  0.87547    0.20649        4096        2  0.36741    0.00000
       25      296  0.71132    0.08190        7387        1  0.36523    0.00000
       32      231  0.62973    0.03899

Summary over the five machines (`beta_c` = least block length above which `eta_B` stays below 1,
by an integer scan; `sum 4/g^2` is the ideal):

    q      L=W(q)   gears   eta_B(1)   max eta_B   at beta   beta_c   eta_B(L)   sum 4/g^2
    59        610      15    0.84918      0.9607         2        1    0.35816     0.35161
    97       1684      23    0.88005      1.0145         2        5    0.35855     0.35727
   199       7387      44    0.91512      1.1190         4       12    0.36523     0.36155
   499      42085      93    0.93858      1.2490         7       20    0.36586     0.36345
   997     169514     166    0.95117      1.3108         7       26    0.36545     0.36404

Three readings.

* **The best block is the whole interval**, at every machine, and the budget there is within
  `0.0013 .. 0.0066` of `sum 4/g^2`. So the second moment does NOT collapse: r51's collapse is an
  artefact of the partition, exactly as the crack claimed.
* **The budget exceeds 1 at short blocks** (`beta = 2 .. 25`) even though the window contains
  hundreds of openings. `eta_B` is an upper bound on the true loss, and the gap between them is
  where the method spends: at `beta = 16`, `q = 199`, `eta_B = 0.875` against a true loss of
  `0.206`.
* **`beta_c` is tiny**: 12 columns at `q = 199`, 26 at `q = 997`, against windows of 7387 and
  169514 and against `F(59) = 161`. The exact budget is below 1 for blocks far shorter than the
  record.

Variable-`beta` rules (the partition re-chosen at each gear), `eta_B` / true loss:

    rule                                  q=59        q=97       q=199       q=499       q=997
    beta_i = g_i                   0.5703/.049 0.5380/.012 0.6129/.008 0.6268/.010 0.6159/.009
    beta_i = 4 g_i                 0.3960/0    0.3916/0    0.4028/0    0.4037/0    0.4039/0
    beta_i = 16 g_i                0.3591/0    0.3652/0    0.3737/0    0.3740/0    0.3731/0
    beta_i = geomean(gears used)   0.6833/.183 0.7204/.169 0.7466/.156 0.7684/.132     -
    beta_i = Q_{<i} (fibre length) 0.4810/.400 0.4784/.400 0.4788/.400 0.4672/.400 0.4671/.400

`beta_i = 16 g_i` is within `0.008` of the whole-interval optimum at every machine, so a block of
sixteen periods of the current gear already recovers essentially all of the second moment. The
"fibre length" rule (blocks as long as the fibre SPACING) sits at `0.47` with a true loss of `0.40`
- it is the arithmetic-block image of the fibre partition and it loses a fifth of the budget.

### R2. The sweep: 107 machines, no exception (`bm_sweep.py`)

`eta_B` at `beta = L = W(q)`, every prime `q` from 5 to 599, against r51's exact FIBRE budget on
the same window:

     q    L=W(q) gears    eta_B    ideal   eta_B-ideal   eta_I fibre   eta_I/eta_B
     7        19     2  0.24685  0.24163      +0.00521       0.25854         1.047
    13        46     4  0.25710  0.29836      -0.04126       0.49950         1.943
    19        85     6  0.31268  0.32328      -0.01060       0.67499         2.159
    29       155     8  0.35529  0.33560      +0.01969       0.83093         2.339
    43       361    12  0.34999  0.34723      +0.00276       1.01181         2.891
    59       610    15  0.35816  0.35161      +0.00655       1.07398         2.999
    97      1684    23  0.35855  0.35727      +0.00128       1.22407         3.414
   199      7387    44  0.36523  0.36155      +0.00368       1.46929         4.023
   307     16069    61  0.36628  0.36262      +0.00366       1.56096         4.262
   401     27813    77  0.36545  0.36314      +0.00232       1.64132         4.491
   499     42085    93  0.36586  0.36345      +0.00241       1.70684         4.665
   599     60100   107  0.36548  0.36364      +0.00184       1.75267         4.796

**107 machines, `eta_B >= 1` at none of them; the largest excess over `sum 4/g^2` is `0.01969` at
`q = 29`.** The fibre budget crosses 1 at `q = 43` and diverges; the block budget is flat at
`0.365`. That is the crack, measured: a factor `1.05 -> 4.80` and growing like `2 log log q`.

### R3. The a priori (phase-adversarial) block budget - the deciding computation (`bm_envelope.py` part A)

A theorem cannot use the machine's own survivor counts. The honest envelope grants the adversary
the freedom to put every strike of gear `i` on a survivor:
`alpha_i(B) <= min(1, 2 ceil(beta/g_i)/s_{i-1}(B))`. Granting the METHOD the most generous survivor
count possible (`s = beta Pi_{<i}`, every block at its fair density) and letting `beta -> infinity`
so the ceiling is free:

      g     Pi_<g   (2/g)/Pi     term   cumulative   fibre 2/g     4/g^2
      5   1.00000    0.40000  0.16000      0.16000     0.40000   0.16000
      7   0.60000    0.47619  0.22676      0.38676     0.28571   0.08163
     11   0.42857    0.42424  0.17998      0.56674     0.18182   0.03306
     13   0.35065    0.43875  0.19250      0.75924     0.15385   0.02367
     17   0.29670    0.39651  0.15722      0.91646     0.11765   0.01384
     19   0.26180    0.40208  0.16167      1.07813     0.10526   0.01108
     23   0.23424    0.37123  0.13781      1.21594     0.08696   0.00756
     29   0.21387    0.32246  0.10398      1.31992     0.06897   0.00476

**The a priori block budget passes 1 at the SIXTH gear, `g = 19`, with no dependence on `beta`, on
`L` or on `q`.** (E3's pre-registered arithmetic was `1.078`; measured `1.07813`.) The mechanism is
one line: `alpha` is a CONDITIONAL rate, so its numerator is a share of the block and its
denominator is a share of the survivors, and the survivors thin out like `Pi_{<g} ~ c/(log g)^2`
while the strikes do not. Where the fibre method has `alpha = 2/g` for free by CRT, the block
method has `alpha <= (2/g)/Pi_{<g}`, and `(2/g)/Pi_{<g}` sits at `0.40 +- 0.04` for the whole head.

Totals at finite `beta` for the real machines (LD = the generous local-density reading, CAP = the
only unconditional reading `s = beta - sum_{j<i} 2 ceil(beta/g_j)`, CRT+LD = LD with the terms
`4/g^2` wherever `Q_{<g} g <= beta` is provable):

    q    beta = W(q)   LD      CAP       CRT+LD          q      LD      CAP    CRT+LD
    59          610  1.913   12.739       1.613        499   2.943   90.722     2.482
    97         1684  2.229   20.728       1.934        997   3.113  163.722     2.509
   199         7387  2.621   41.723       2.158

Every one is above 1, at every `beta` tested (`W`, `10W`, `10^6`, `10^12`).

### R4. Blocks against fibres, gear by gear (`bm_envelope.py` part B)

    g       4/g^2   fibre 2/g   block adv   block/fibre   winner
    5     0.16000     0.40000     0.16000         0.400    block
    7     0.08163     0.28571     0.22676         0.794    block
   11     0.03306     0.18182     0.17998         0.990    block
   13     0.02367     0.15385     0.19250         1.251    fibre
   17     0.01384     0.11765     0.15722         1.336    fibre
   19     0.01108     0.10526     0.16167         1.536    fibre
   23     0.00756     0.08696     0.13781         1.585    fibre
   31     0.00416     0.06452     0.10498         1.627    fibre
   47     0.00181     0.04255     0.07090         1.666    fibre
   59     0.00115     0.03390     0.05300         1.564    fibre
   97     0.00043     0.02062     0.03089         1.498    fibre
  199     0.00010     0.01005     0.01352         1.345    fibre
  499     0.00002     0.00401     0.00394         0.982    block
  997     0.00000     0.00201     0.00149         0.741    block

**The block term first exceeds the collapsed fibre term at `g = 13`** and stays above it through
`g = 467` (86 consecutive primes), falling below again only from `g = 479`, where the survival
density has fallen far enough that `(2/g)/Pi^2 < 1`. Since the
budget is dominated by the head, the better-of-both-partitions bound is r51's fibre bound and the
change of partition buys nothing a priori. E4 confirmed.

Thresholds, scanned (the a priori budget is NOT monotone in `beta` - the `ceil` jumps - so every
threshold here is a scan over all integers to 4000 plus the transition points `Q_{<i} g_i` plus a
geometric grid; `bm_gate.py` part C):

    q      W(q)    beta* LD   beta* CAP   beta* CRT+LD    fibre L* (r51)   CRT+LD / L*
    59       610        none        none     3.7182e+07        1.0614e+04     3.503e+03
    97      1684        none        none     5.0708e+13        3.8882e+06     1.304e+07
   199      7387        none        none     9.0525e+25        1.2324e+14     7.345e+11
   499     42085        none        none     4.6622e+43        1.9028e+30     2.450e+13
   997    169514        none        none     9.4292e+62        8.9524e+52     1.053e+10

The only finite a priori block threshold is the hybrid CRT+LD one, it is a primorial like r51's,
and it is `3.5e3` to `2.5e13` times LONGER. Against `exp(theta(q^{0.73}))` (r51's law) and against
`W ~ q^2/6`: the block threshold is super-polynomial too, and strictly worse.

### R5. The adversarial gate, and the refutation of local density (`bm_gate.py` parts A, B)

`A(K)` (arc_multiset.md R1) is the longest opening-free STRETCH, i.e. the gap between consecutive
openings, so `K` primes actually cover `A(K) - 1` consecutive columns (verified against the exact
period scan: the longest blocked RUN is `1, 4, 6, 10, 17, 24` at `m5..m19`, one less than
`F = 2, 5, 7, 11, 18, 25`). An envelope is REFUTED if it says an interval of `A(K) - 1` columns
cannot be covered.

    K    A(K)  covered           LD                CAP           CRT+LD
    1       2        1     ok, b*=3           ok, b*=3         ok, b*=3
    2       5        4  REFUTED @4           ok, b*=9      REFUTED @4
    3       7        6    ok, b*=13          ok, b*=72       ok, b*=13
    4      16       15    ok, b*=37          ok, b*=-      REFUTED @11
    5      22       21   ok, b*=172          ok, b*=-        ok, b*=42
    6..12   -        -     ok, b*=-          ok, b*=-   ok, b*=223..1.6e6

> **The refuting instance.** `K = 2`, gears `{5, 7}` - the REAL machine at `q = 7`, not an
> adversarial set. Those two gears cover 4 consecutive columns (`F(7) = 5`, confirmed by a full
> period scan). The local-density block budget at `beta = 4` is
> `min(1, 2/4)^2 + min(1, 2/2.4)^2 = 0.25 + 0.694 = 0.944 < 1`, so it asserts that 4 columns cannot
> be covered. **The hypothesis "every block holds `beta Pi` survivors" is false, and it is false at
> the first place it is used.** The same envelope with the CRT terms added is refuted twice, at
> `K = 2` (`beta = 4`) and `K = 4` (`beta = 11`, gears `{5,7,11,13}` which cover 15).

This is the block analogue of r51's refutation of average first moments (`A(7) <= 9.3` against
`37`), and it is sharper: the instance is the real machine's own record at the second gear.

The only envelope that passes the gate everywhere is CAP, the pure counting bound - and CAP is
vacuous from `K = 4` and from `m13` onwards (`b* = -`). Its whole reach is
`beta* = 3, 9, 72` at `K = 1, 2, 3`.

### R6. The exact budget on covered stretches (`bm_exact.py` part B)

Validity check of the theorem and of the implementation: on a fully covered stretch the block
inequality forbids `eta_B < 1`. Longest covered run found by scanning columns `1..2x10^7`:

    m     run found   F(M)     eta_B over beta = 1..run          verdict
    11            6      7     1.000 .. 1.567  (6 values)        all >= 1
    13           10     11     1.000 .. 1.567  (7 values)        all >= 1
    17           17     18     1.000 .. 1.691  (9 values)        all >= 1
    19           24     25     1.000 .. 1.829  (9 values)        all >= 1
    23           33     34     1.000 .. 1.737 (11 values)        all >= 1
    29           37     43     1.000 .. 1.920 (11 values)        all >= 1
    31           44     58     1.000 .. 1.710 (11 values)        all >= 1

63 evaluations, no failure (the `beta = 1` entries are exactly 1, the covered-fraction identity;
they are reported as `0.99999999` in float and are equalities). The bound is never tight above
`beta = 1`: the smallest value above `beta = 1` is `1.183` (`m13`, `beta = 3`).

### R7. Gear ordering (`bm_envelope.py` part D)

    q     fibre L* increasing   fibre L* decreasing        ratio   block eta inc   block eta dec
    59            1.0671e+04            2.4869e+17   4.29e-14         0.35816         0.34434
    97            4.0753e+06            3.2116e+32   1.27e-26         0.35855         0.36248
   199            1.2542e+14            1.1537e+78   1.09e-64         0.36523         0.39012
   499            1.9338e+30           1.8986e+202  1.02e-172         0.36586         0.40887
   997            9.3348e+52                  none          -         0.36545         0.42082

**Ordering changes the FIBRE threshold by an exponent (13 to 172 orders of magnitude, and at
`q = 997` the decreasing order never reaches a finite threshold at all) and the BLOCK budget by a
constant (at most a factor 1.15).** E6 confirmed on both halves. Increasing order is optimal for
fibres, because the collapse count at length `L` is `#{i : g_1...g_{i-1} >= L}` and the small gears
delay it. For blocks the ordering is nearly irrelevant, which is the same fact from the other side:
there is no collapse point to move. Across `beta` at `q = 199`:

    beta        4       16       64      256     1024     7387
    increasing  1.11904  0.87547  0.48658  0.39153  0.37310  0.36523
    decreasing  1.10290  0.68061  0.46068  0.41576  0.40368  0.39012
    random      1.05637  0.75401  0.47780  0.39914  0.38137  0.37200

### R8. The three costs, and how much is unconditional (`bm_mech.py`)

Summed over gears at `beta = L = W(q)`:

    q          L     ideal   ceiling   condition   measured    fibre
    59       610   0.35161   0.35629     1.84748    0.35816   1.1748
    97      1684   0.35727   0.35972     2.18144    0.35855   1.2857
   199      7387   0.36155   0.36227     2.60077    0.36523   1.5192
   499     42085   0.36345   0.36355     2.93745    0.36586   1.7183
   997    169514   0.36404   0.36407     3.11115    0.36545   1.9046

**The ceiling costs the fifth decimal; the conditioning costs a factor 7 to 8.5.** That answers the
brief's question 4 with numbers: at `q = 199` the gain over fibres is `1.5192 - 0.36523 = 1.154`
(measured) and the loss to the conditioning is `2.60077 - 0.36155 = 2.239` (adversarial), so the
crack is worth `1.15` and the hole it opens is worth `2.24`. At `q = 997` the same pair is `1.55`
and `2.75`.

How much of the budget needs no hypothesis at all - the gears with `Q_{<g} g <= L`, for which the
block contains every class mod `Q_{<g} g` and `alpha = 2/g` up to an edge error:

    q     #head   top head g     head     room   measured tail   adv tail   adv/room
    59        3           11  0.27469  0.72531         0.08020    1.28074      1.766
    97        3           11  0.27469  0.72531         0.08420    1.61470      2.226
   199        4           13  0.29836  0.70164         0.06668    1.84153      2.625
   499        4           13  0.29836  0.70164         0.06749    2.17821      3.104
   997        5           17  0.31220  0.68780         0.05327    2.19469      3.191

**The unconditional head never reaches beyond the fifth gear.** The room it leaves is `0.688` to
`0.725`; the real machine's gears above it use `0.053` to `0.084` of that room (a thirteenth to an
eighth); the adversary is charged `1.77` to `3.19` times the room. That ratio is the exact size of
the missing input.

Where the equality `alpha_g = 2/g` is proved and where it is merely measured, on the real window:

    q          L   last g with Q_<g g <= L   first g with |rho-1| > 1% / 5% / 20%   max rho
    59       610                        11                          7 / 17 / 37     1.4901
    97      1684                        11                         17 / 29 / 31     1.4823
   199      7387                        13                         23 / 31 / 47     1.3489
   499     42085                        13                         31 / 67 / 113    1.4587
   997    169514                        17                         53 / 79 / 197    1.5203

The `rho` profile at `q = 199` (all 44 gears), `rho_g = alpha_g g/2`:

    5:1.000   7:1.000  11:0.999  13:1.006  17:0.994  19:1.002  23:1.017  29:0.984
   31:0.938  37:0.965  41:0.955  43:1.103  47:1.211  53:1.180  59:1.182  61:1.155
   67:1.251  71:1.265  73:1.349  79:1.115  83:1.301  89:0.960  97:1.010 101:1.327
  103:1.192 107:0.986 109:1.242 113:1.008 127:1.064 131:0.744 137:0.886 139:0.809
  149:0.768 151:1.124 157:1.067 163:0.874 167:0.388 173:0.538 179:0.420 181:0.569
  191:0.453 193:0.153 197:0.157 199:0.000

`eta_B = sum (4/g^2) rho_g^2`, and the budget is below 1 while the weighted `L2` mean of `rho`
stays below `1/sqrt(sum 4/g^2) = 1.6631`. Measured maxima `1.349` (`q = 199`) and `1.520`
(`q = 997`): the machine has a factor 1.09 to 1.23 of headroom in the sup norm and 1.66 in `L2`.
The top gears have `rho` well BELOW 1 (the last four at `q = 199` are `0.45, 0.15, 0.16, 0.00`):
by the time the top gears act, most of what they would strike is already struck.

---

## 4. Mechanism

The distortion engine's gain over the union bound is that it charges `E[alpha^2]` instead of
`E[alpha]`, and `alpha` is a proportion of a PART. Everything turns on how the part is chosen.

**Where blocks gain.** A congruence fibre of the interval has `L/Q_{<g}` columns and shrinks by a
factor `g_j` at every gear, so it is a single column after four or five gears and `alpha` is
`{0,1}`-valued: the second moment IS the first moment (r51's collapse). An arithmetic block never
shrinks - it has `beta` columns for ever - so `alpha_i(B)` stays a genuine fraction at every gear,
and if the survivors are struck at their fair rate it stays at `2/g_i`. That is measured, exactly,
on the real machine: `eta_B(beta = L) = 0.358 .. 0.366` against a fibre budget of `1.07 .. 1.91`,
below 1 at all 107 machines to `q = 599` where the fibre budget crosses 1 at `q = 43`. The gain is
real and it is a factor 3.0 to 4.8 on the real window.

**Where blocks lose.** A fibre is a union of residue classes, so the next gear's strikes on it are
uniform BY CRT, with no input. A block is not, so the strike rate on the block's SURVIVORS is
`(2/g) rho`, and `rho` is unknown. The only bound available a priori is that all `2 ceil(beta/g)`
strikes could land on survivors, i.e. `rho <= 1/Pi_{<g}`, and `Pi_{<g} ~ c/(log g)^2` decays: the
term becomes `((2/g)/Pi_{<g})^2`, which is `0.16 .. 0.19` for every one of the first six gears, and
their sum is `1.078` before the seventh gear is reached. So:

> **The block partition trades r51's tail collapse for a head that costs 3.5 times more.** In the
> fibre budget the head (gears 5 to 17) costs `0.30` and the divergence comes from the collapsed
> tail; in the block budget the head alone costs `1.08` and the tail is negligible. Gear by gear
> the block term is better only at `g = 5, 7, 11`, and worse from `g = 13`.

The two error terms of the change of partition are separated exactly (R8): the CEILING (a block is
not a union of classes, so a gear strikes `2beta/g + O(1)` of it) moves the budget in the fifth
decimal and is free; the CONDITIONING (the survivors need not be evenly spread in the next gear's
classes inside the block) costs a factor 7 to 8.5 and is the whole story. And the CONDITIONING is
free exactly when `Q_{<g} g | beta`, i.e. exactly when the block is a union of congruence fibres -
so the crack, followed to its provable end, closes back onto the fibre partition it was meant to
escape.

The last piece is why no cheap substitute works. Replacing the unknown survivor count by its fair
value `beta Pi` is the natural move and it is FALSE: `{5, 7}` cover 4 consecutive columns while the
substituted budget at `beta = 4` is `0.944`. The reason is the one the wall keeps naming - on a
short interval the phases are chosen, and a chosen phase beats its average - but here it appears in
a new place: not in the strike rate but in the SURVIVOR COUNT, which the adversary depresses below
`beta Pi` in the block it is attacking.

---

## 5. What is new

Checked against `docs/novel/README.md` (the whole index read this round) and the tree. The nearest
entries are `consistency-over-degree` and `moment-degree-ceiling`, where "block" means a block of
LP variables (one gear's indicator block) in a Boole-Bonferroni covering LP, not a run of columns;
their finding that "consistency is a statement ACROSS BLOCKS that no moment inequality can see" is a
different object with a suggestively parallel conclusion, and is noted, not re-derived. Nothing in
the index or in r51 carries a second moment over runs of columns.

1. **The block inequality is a theorem for any partition, re-chosen at every stage.** BBMST
   Theorem 3.1's proof uses the fibre structure only to EVALUATE `alpha_i = 2/g_i` by CRT; mass
   preservation, the loss bound and the distortion bound are per-part algebra. The freedom to
   re-choose the partition at every gear is used in R1 (`beta_i = 16 g_i` recovers the optimum) and
   is not available in BBMST.
2. **The collapse is an artefact of the partition, and the exact gain is measured.** `eta_B` at
   `beta = L` is `0.358, 0.359, 0.365, 0.366, 0.365` at `q = 59..997` against the fibre `1.074,
   1.224, 1.469, 1.707, 1.905`; below 1 at all 107 machines `q <= 599`, no exception; the fibre
   budget crosses 1 at `q = 43`. The block budget stays flat where the fibre budget grows like
   `2 log log q`.
3. **The head/tail exchange, exact.** The a priori block budget passes 1 at the sixth gear with
   cumulative `0.16000, 0.38676, 0.56674, 0.75924, 0.91646, 1.07813`, independently of `beta`, `L`
   and `q`; the per-gear crossover against the collapsed fibre term is at `g = 13`. This is a
   machine-free statement about two-class covering budgets on an interval.
4. **The refutation of local density, with the real machine as the instance.** Any argument that
   replaces a block's survivor count by `beta prod(1-2/g)` is false: `{5, 7}` cover 4 consecutive
   columns and the substituted budget there is `0.944 < 1`. This joins r51's average-first-moment
   refutation as a filter on covering-side attempts, and it is sharper (real teeth, second gear).
5. **The budget in `rho` coordinates.** `eta_B = sum_g (4/g^2) rho_g^2` with
   `rho_g = alpha_g g/2`, so the root's covering half is implied by "the weighted `L2` mean of the
   survivor strike-rate excess stays below `1.663`". Measured `max rho = 1.349` at `q = 199`,
   `1.520` at `q = 997`, with the top gears well below 1. This is the level of distribution the
   covering side needs, written as ONE number with a stated threshold rather than an asymptotic.
6. **`eta_B(beta = 1) = the covered fraction`, exactly**, at every interval: the block inequality
   at unit blocks is the tautology "not every column is covered". The family of block lengths
   therefore interpolates between two tautologies (`beta = 1` and `beta = L`), and all of the
   method's content is the a priori bound on `rho`, not the partition.

---

## 6. Verdict, toward the root

**Outcome (C).** The second moment over arithmetic blocks avoids the collapse lemma and is a valid
theorem for any partition, but it has no unconditional content on the window: its head busts the
budget at the sixth gear under any phase-adversarial bound, and the hypothesis that would rescue
the head is false at the second gear.

What the block inequality proves, stated exactly:

> **(BI-unconditional).** Let `I` be an interval of `beta` columns and `g_1 < ... < g_n` primes
> `>= 5` with two classes each at any phases. If
> `sum_i min(1, 2 ceil(beta/g_i)/(beta - sum_{j<i} 2 ceil(beta/g_j)))^2 < 1` then `I` is not
> covered. This is unconditional (no equidistribution input) and its reach is
> `beta <= 72` at three gears and nothing at four or more. It is the capacity bound with a square
> on it, and it covers **0%** of the window at every machine from `m13` on.

> **(BI-conditional).** With `rho_g := (fraction of the window's survivors of `{5..g^-}` struck by
> `g`) / (2/g)`, the machine `{5..q}` leaves an opening in its window whenever
> `sum_{5<=g<=q} (4/g^2) rho_g^2 < 1`. Unconditional for the gears with `Q_{<g} g <= W(q)` (three
> to five gears, worth `0.275 .. 0.312`); for the rest it is a level of distribution at dimension 2
> for the sifted set inside one window. The room left for those gears is `0.688 .. 0.725`, the real
> machine uses `0.053 .. 0.084` of it, and the phase adversary is charged `1.77` to `3.19` times it.

So the residual IS a level of distribution at dimension 2, as r51 said - but this branch localises
it to a single quantity with a threshold, and prices the gap: the machine is a factor 8 to 13
inside the room, the adversary a factor 1.8 to 3.2 outside it. Nothing between the two is decided
by any counting argument on the tree; deciding it is deciding the root.

**Toward the root, mechanism first.** The one thing the branch adds to the shape of the wall is
that face A's covering face now has a coordinate. r51 left "the surviving set must be
equidistributed in the classes of the next gear" as a qualitative need. Here it is a number,
`rho_g`, with an exact budget `sum 4/g^2 rho_g^2 < 1`, and the machine's own profile is measured at
44 and 166 gears. That profile has structure worth a child branch: `rho_g = 1.000` to three decimals
for every gear up to `23` at `q = 997` (far beyond the provable range, which stops at 17), rises to
`1.2 - 1.5` in the middle band, and falls below `0.5` for the top gears - the same three-band shape
(head, middle band striking at a constant rate, top gears) the flank decomposition found in the wall's
section 5b. Whether `rho_g` is a function of `g/q` alone is the obvious first question and is not
answered here.

---

## 7. Exceptionless, with the count

| statement | range | count | status |
|---|---|---|---|
| the block inequality holds for any partition, re-chosen at any stage | all | proof (section 2.2) | proved, per-part algebra |
| `P_{i-1}` uniform on the block's survivors when every earlier `alpha <= 1/2` | all | proof (section 2.4) | proved, one line |
| `eta_B(beta = 1)` = the covered fraction of the interval | all | proof + 12 checks | exact identity |
| `eta_B(beta = L) < 1` on the window with real teeth | `q = 5..599`, all 107 primes | 107 | exact, 0 exceptions, max `0.36673` |
| `eta_B(beta = L) - sum 4/g^2 <= 0.0197` | `q = 5..599` | 107 | exact, worst at `q = 29` |
| `eta_B(beta = L) < eta_I(fibre)` on the same window | `q = 7..599` | 106 | exact, ratio `1.047 -> 4.796` |
| `eta_B >= 1` on a fully covered stretch (the validity gate) | `m11..m31`, all tested `beta` | 63 | exact, no failure |
| the a priori block budget `>= 1.078` after six gears | every machine `{5..q}`, `q >= 19`, every `beta`, every `L` | proof (terms positive, cumulative `1.07813` at `g = 19`) | exact |
| the block term exceeds the collapsed fibre term | `13 <= g <= 467` | all 86 primes in range (contiguous; below at `g = 5, 7, 11` and again from `g = 479`) | exact |
| the CAP (unconditional) envelope passes the gate `L*_B >= A(K) - 1` | `K = 1..12` | 12 | exact |
| the LD envelope FAILS the gate | `K = 2` (`{5,7}`), `beta = 4` | 1 | exact, refuting instance `0.944 < 1` against 4 columns covered |
| gear ordering moves the block budget by `<= 1.15x` and the fibre `L*` by `>= 10^13` | `q = 59..997` | 5 | exact |

---

## 8. Dead ends, with the refuting instance

* **"Blocks restore the second moment, so the localised distortion method works."** DEAD as an
  unconditional statement: the restoration is real (factor 3.0 to 4.8, R2) but it is computed from
  the machine's own survivor counts. Every phase-adversarial version busts at the sixth gear
  (cumulative `1.07813` at `g = 19`, R3).
* **"Assume each block holds its fair share `beta prod(1-2/g)` of survivors."** DEAD, refuted by
  the real machine `{5,7}`: 4 consecutive columns are covered (`F(7) = 5`) while the hypothesis
  gives a budget of `0.944 < 1` at `beta = 4`. Also refuted at `K = 4` (`beta = 11`) in the hybrid
  form.
* **"Choose `beta` well."** DEAD: the best fixed `beta` is the whole interval at every machine, and
  the a priori budget's failure (`1.078` at six gears) has no `beta` in it at all. The best variable
  rule (`beta_i = 16 g_i`) is within `0.008` of the whole-interval optimum and equally
  unconditional-free.
* **"Reorder the gears to keep more of them uncollapsed."** DEAD for blocks (there is no collapse
  to avoid; ordering moves `eta_B` by at most 1.15x, R7). For FIBRES the ordering matters enormously
  and increasing order - the one r51 used - is already the best (decreasing order costs 13 to 172
  orders of magnitude).
* **"The block bound is a cheap certificate."** DEAD as stated: at `beta = 1` the inequality is the
  identity "the covered fraction is below 1" and at `beta = L` it is "a product of `(1 - alpha_i)`
  is positive". Both are tautologies given the counts. What is not a tautology is the `rho`
  formulation (section 6), and that is a conditional statement, not a certificate.

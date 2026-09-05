# Fibres of a sub-machine

Prover lane, round 54. Branch node **R2.c.ii**, child of `research/proof/distortion_method.md`
(R2.c), opened by the unstick file's reading-as-a-whole item 3 and by R2.c Idea 1 of
`research/proof/dead_branches_reopened_2.md`: *"Collapse happens when the product of gears used
exceeds the interval. Use fibres modulo the product of the small gears only (which fits the
window) and treat the remaining gears as a perturbation inside each fibre."*

Sibling: `research/proof/block_moment.md` (R2.c.i), which replaced the congruence fibres by
arithmetic blocks and died on the conditioning. This branch sits between the two: the partition is
still by congruence, but it stops refining at a chosen modulus.

Scripts in `research/anchor235/r54/` (prefix `sf_`); result outputs, untracked, in
`research/anchor235/r54/results/`. Every number this document relies on is written here.

---

## 1. Pre-registered (written before any computation of this round)

### 1.0 The object and the engine

Column `k` is the pair `(6k-1, 6k+1)`. Gear `g >= 5` strikes `k` iff `k = +-u_g (mod g)`,
`u_g = 6^{-1} mod g`. The machine `M = {5..q}`; the window is the columns
`k_lo = floor((q+1)/6) + 1 .. k_hi = (q'^2-1)/6`, `L = W(q) = k_hi - k_lo + 1` (the r51/r52
convention, so the budgets are directly comparable: `W = 610, 1684, 7387, 42085, 169514, 668335`
at `q = 59, 97, 199, 499, 997, 1999`).

The engine is BBMST Invent. math. 228 (2022) Theorem 3.1 in the form r51 reduced it to and r52
proved partition-agnostic: process the gears in some order carrying a probability measure `P_i`
on the interval; at stage `i` choose ANY partition of the interval into parts; on each part `B`,

    a := alpha_i(B) = P_{i-1}(B_i ∩ B) / P_{i-1}(B)        (B_i = the columns g_i strikes fresh)
    P_i = max{0, (a - d)/(a(1-d))} P_{i-1}  on B_i ∩ B ,
    P_i = min{1/(1-a), 1/(1-d)} P_{i-1}     on B \ B_i ,       d = delta_i in [0, 1/2],

and if `eta = sum_i min{M_i^(1), M_i^(2)/(4 d(1-d))} < 1`, with `M^(k) = E_{i-1}[alpha_i^k]`
(expectation over the parts under `P_{i-1}`), then the gears do not cover the interval. At the
optimal `d = 1/2` the term is `min{M^(1), M^(2)} = M^(2)`.

**The sub-machine fibre partition.** Fix a cut `a` and let `S = {5..a}` be the SUB-MACHINE,
`Q_s = prod_{g in S} g` its period (a gear-primorial: `1, 5, 35, 385, 5005, 85085, 1616615, ...`),
with `Q_s <= L`. The partition is chosen stage by stage:

* for a gear `g_i in S` (a small gear): the parts are the classes mod `Q_{<i} = g_1...g_{i-1}`
  (the refining partition, exactly as in BBMST/r51);
* for a gear `g_i > a` (a big gear): the parts are FROZEN at the classes mod `Q_s`.

A part of the frozen partition is a **fibre**: an arithmetic progression of step `Q_s` inside the
window, with `m = L / Q_s` columns (`+- 1`).

### 1.1 The theory

**T (the head is exact, the freeze keeps the fibre whole, the tail is a perturbation).** Three
claims, each to be proved or refuted in section 2:

1. While the partition refines with the gears, every part is a full class mod `Q_{<i}`, hence
   **entirely surviving or entirely dead**, and by CRT `alpha_i = 2/g_i` on every live part,
   exactly, provided the part holds whole classes mod `g_i`, i.e. `Q_{<i} g_i <= L`. So each
   small gear costs exactly `4/g^2`, and the sub-machine's whole cost is
   `sum_{g in S} 4/g^2 < 0.36455`, never the union bound.
2. At the freeze the fibres are still whole: a live fibre is a class mod `Q_s` that no small gear
   strikes, so **all `m` of its columns survive the sub-machine**. The first big gear therefore
   strikes exactly `2/g` of a live fibre (up to the ceiling `+-2/m`), again with no hypothesis.
   This is the one thing an arithmetic block cannot do (r52: a block loses `1 - prod(1-2/g)` of
   its columns to the small gears before the first big gear acts, and their positions are unknown)
   and it is the whole content of the word "sub-machine".
3. From the second big gear on, a fibre is only partly alive, and `alpha_i(f)` is the struck
   fraction of the fibre's SURVIVORS. Its mean over the fibres is the block quantity; its spread
   across the fibres is new. Because the frozen partition preserves each fibre's mass, the fibres
   carry EQUAL weight whatever their survivor counts, so the engine automatically renormalises a
   fibre the tail has over-struck. The budget is then
   `M^(2)_g = (mean_f alpha)^2 + Var_f(alpha)`, and the branch is the question of `Var_f`.

**The exchange, stated in advance.** A coarser partition has a smaller budget (a single block is
the smallest of all, r52's `0.365`) but nothing to evaluate `alpha` with; a finer partition
evaluates `alpha` by CRT but pays more, and in the limit collapses to the union bound (r51). The
sub-machine cut `a` is the exact interpolation. The branch asks whether the interpolation has a
sweet spot or is monotone.

### 1.2 Predictions, with numbers, and what refutes each

**P1 (N1: where the per-fibre first moment is exact).** The constraint N1 of the unstick file says
a covering argument must keep exact first moments. Predicted: the per-fibre first moment of gear
`g` is exact up to the ceiling `2/m` iff the fibre holds whole classes mod `g`, i.e.
`Q_s g <= L`; so with the brief's default `Q_s` = the largest gear-primorial below `W` the fibres
hold `m = W/Q_s` columns and at `q = 997` that is `169514/85085 = 1.99` columns - **the first
moment is exact for NO big gear at the default cut**. Requiring `m >= q` (every big gear whole)
forces `Q_s <= W/q ~ q/6`, i.e. the sub-machine is `{5,7}` (`Q_s = 35`) for `210 <= q < 2310`,
`{5,7,11}` (`385`) for `2310 <= q < 30030`, `{5,7,11,13}` (`5005`) from `q >= 30030`. So the
number of small gears the exactness allows grows like `ln W / ln ln W` but reaches 4 only at
`q = 30030`. REFUTED if the exactness range is different, or if `m >= g` is not the criterion.

**P2 (the exact budget is monotone in the cut).** Predicted: `eta_SF(q, Q_s)` is INCREASING in
`Q_s` at every `q`, running from r52's block budget `~0.365` at `Q_s = 1` to r51's fibre budget
`1.07 .. 1.91` at the largest primorial below `W`. Pre-registered values at `q = 997`:
`0.365 (Q_s=1)`, `0.37 (35)`, `0.40 (385)`, `0.6 (5005)`, `1.5 (85085)`, with the crossing of 1 at
`Q_s = 5005` or `85085`. REFUTED if `eta_SF` is not monotone in `Q_s`, or if it stays below 1 at
the largest primorial below `W` (which would mean the collapse is avoidable inside the congruence
partitions themselves).

**P3 (the deciding prediction: the unconditional envelope).** A theorem may not use the machine's
own survivor counts. In a live fibre the survivor count starts at `m` (claim 2 above) and each big
gear removes at most `2 ceil(m/g)` columns, so

    alpha_g(f) <= min(1, 2 ceil(m/g) / (m - sum_{a < h < g} 2 ceil(m/h)))      (SF-CAP)

is UNCONDITIONAL. Predicted: this is strictly better than r52's block CAP (whose denominator is
depleted by `2/5 + 2/7 = 0.686` before the first big gear, making it vacuous from four gears) and
strictly better than r51's collapsed fibre budget (which crosses 1 at `q = 43`), because the
depletion inside a fibre begins at zero. Its non-vacuity needs `sum_{a < g <= q} 2/g < 1`, so the
cut must satisfy `ln a / ln q > e^{-1/2} = 0.6065`, against r51's measured `0.728`; predicted
threshold `L*_SF(q) = exp((1+o(1)) theta(q^{0.6065}))`, with the measured `ln(cut)/ln q` rising
towards `0.61` and `L*_SF < L*_r51` at every `q >= 97`. REFUTED if the measured exponent is not
below r51's, or if SF-CAP is vacuous at every `q`, or if it is not below r51's `L*`.

**P4 (the obstruction, named and priced).** The cross-fibre term is
`V_g = Var_f(alpha_g)` under the equal fibre weights, i.e. the L2 discrepancy of gear `g`'s
strikes across the residue classes of `Q_s`: with `D_f(g) = (fresh strikes in f) - 2 s_f/g`,
`V_g = mean_f (D_f/s_f)^2 - (mean_f D_f/s_f)^2`. Over a FULL PERIOD every `D_f = 0` exactly (CRT),
so `V_g = 0` and `eta_SF = sum 4/g^2 < 0.36455`: the object is purely an interval discrepancy.
Predicted: on the window `sum_g V_g` is small for the real machine (below `0.05` at every `q` with
`Q_s = 35`) but a uniform bound `|D_f| <= C` needs `C` below 5 at `q = 997` to keep
`sum_g V_g < 1 - 0.365 = 0.635`, so **7b's proved anchor rigidity (interval discrepancy below 30
in every window) is not enough by an order of magnitude**; and in any case 7b bounds a SIGNED
first moment summed over the window, which cannot bound an L2 spread. REFUTED if 7b's bound
suffices, or if `sum_g V_g > 0.05` for the real machine at `Q_s = 35`.

**P5 (the adversarial gate).** Validity: on every MILP witness of `arc_milp_K*.txt` (an interval of
`A(K) - 1` columns actually covered by `K` gears, `K <= 12`) the exact `eta_SF` must be `>= 1`.
Predicted: passed at all 12, at every cut. Reach: predicted `L*_SF(K) < W_{K+1}` for `K <= 10` and
NOT for `K = 11, 12`, because at `L ~ W_{K+1} ~ 300` the fibre size `m = L/Q_s` is a few tens and
the ceiling `2/m` swamps `2/g` for the top gears - the sub-machine's advantage needs a long
interval and the ladder does not have one. REFUTED by any witness with `eta_SF < 1` (which would
refute the engine or the implementation), or by `L*_SF(11) < 308`.

**P6 (growth, and what it proves toward the root).** Predicted: with the cut forced to
`a > q^{0.6065}`, `Q_s = exp(theta(q^{0.6065}))` and the shortest interval SF-CAP can address is
`Q_s q` times a constant, i.e. super-polynomial in `q` where the target is `q^2/6`. Predicted: the
sub-machine fibre engine proves NOTHING unconditional about the window of any machine with
`q >= 37`, and its residual is a level of distribution for the sifted set `U_{<g}` at modulus
`Q_s g <= W^{1/2 + o(1)}` - level 1/2, every modulus, not on average - which is stronger than
r52's (modulus `g` alone) and weaker than r51's (modulus the primorial). REFUTED if the residual
is at a different level, or if any unconditional statement about a window survives.

### 1.3 Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | exactness iff `Q_s g <= L`; default cut exact for no big gear; sub-machine `{5,7}` for `210 <= q < 2310` | **CONFIRMED** | R1: the brief's default cut gives `m = 1.48 .. 8.41` at `q = 59..4999`, so no big gear's first moment is exact there; the exactness cut is `Q_s = 5` from `q = 23`, `35` from `q = 199`, `385` from `q = 2297`, `5005` from `q = 29989` (predicted 210, 2310, 30030) |
| P2 | `eta_SF` increasing in `Q_s`, `0.365 -> 1.9`, crossing at `Q_s = 5005` or `85085` | **CONFIRMED** | R2: at `q = 997` the budget runs `0.3655, 0.3655, 0.3661, 0.3739, 0.4943, 1.6863` at `Q_s = 1, 5, 35, 385, 5005, 85085`, monotone at all six `q`; the crossing of 1 is between `5005` and `85085` |
| P3 | SF-CAP unconditional, better than both r51 and r52; exponent `0.6065` against r51's `0.728` | **CONFIRMED** | R4: `L*_SF(q) = 4.20e4, 1.69e6, 1.70e10, 1.14e17, 1e25` at `q = 59..997` at cuts `a = 11, 13, 23, 41, 61`, against `none` (never finite) at `t = 0` and `1e20, 1e35, 1e81, 1e205, 1e414` at `t = all` in the SAME envelope; measured `ln a / ln q = 0.52 .. 0.66`, mean `0.59` |
| P4 | the obstruction is the cross-fibre L2 discrepancy; `C < 5` needed; 7b's 30 not enough | **HALF CONFIRMED, half REFUTED** | R5: the object is right (`V_g = 0` exactly on a full period, checked at three gears; `sum V_g = 0.0041 .. 0.0002` on the window). But `C*` GROWS: `4.8, 15.3, 38.1, 94.8` at `q = 199, 499, 997, 1999`, so a bound of 7b's size (30) IS enough from `q` between 701 and 997 on. Predicted `C < 5`: REFUTED |
| P5 | gate passed at every `K <= 12`; reach still `K <= 10` | **CONFIRMED, and it costs r51 its positive** | R6: the exact `eta_SF` is `1.21 .. 1.72` on all nine MILP witnesses at every cut (0 failures); SF-CAP is consistent (`L* >= A(K)-1`) at all 12 and proves the lemma only at `K = 1`. R7: r51's reach to `K = 10` used `M^(1) <= 2 ceil(L/g)/L`, which the witnesses REFUTE - the engine's own `M^(1)` is `1.80x` to `3.63x` that value on covered intervals |
| P6 | nothing unconditional about any window with `q >= 37`; residual at level 1/2, modulus `Q_s g` | **CONFIRMED** | R4: SF-CAP at `L = W(q)` is `1.00 .. 1.72` (loose) and infinite (strict) at every `q` from 17 to 1999 - vacuous at every window. The residual is the equidistribution of `U_{<g}` in the classes of `g` inside each progression of step `Q_s`, i.e. modulus `Q_s g <= 35 q ~ W^{1/2}` |

Two predictions were wrong in an informative direction. P4's guess that the needed discrepancy
bound shrinks with `q` is backwards: the fibres get longer faster than the gears multiply, so the
needed bound GROWS like `q^{1.3}` while the measured discrepancy grows like `q^{0.66}`, and the
machine's headroom widens from 1.38 to 5.98 between `q = 199` and `q = 1999`. And P5's routine
validity check turned into the branch's sharpest negative result: the covered witnesses refute the
first-moment step r51's adversarial positive rests on.

### 1.4 Outcomes, named in advance

* **(A)** The sub-machine cut has a sweet spot: an unconditional budget below 1 on an interval
  polynomial in `q`.
* **(B)** The cut improves the exponent of the addressable length but stays super-polynomial.
* **(C)** No gain over r51/r52; the freeze buys one gear and nothing else.

**Outcome: (B)**, with (C) true of the BUDGET and (B) true of the unconditional envelope: the
freeze does buy exactly one gear of exact evaluation and the budget is monotone in the cut with no
sweet spot, but the unconditional envelope gains 3 to 801 orders of magnitude in addressable
length over both extremes of its own family, and the cut exponent falls from r51's `0.728` to
`0.6065`.

---

## 2. Setup: the sub-machine fibre engine, proved and hypothetical parts

### 2.1 The step inequality, with its exact terms

Fix the cut `a`, `S = {5..a}`, `Q_s = prod S <= L`, and process the gears in increasing order.
Write `U_{<g}` for the columns of `I` no gear below `g` strikes, `s_f` for `|U_{<g} ∩ f|` in a
fibre `f`, and `S_g` for `g`'s strike set. The engine's step at gear `g` is

    M_g^(1) = E_P[alpha_g] ,   M_g^(2) = E_P[alpha_g^2] ,   term_g = min{M^(1), M^(2)} ,
    eta_SF  = sum_g term_g  <  1    ==>   the gears do not cover I,

with `alpha_g(B) = P(S_g ∩ B)/P(B)` on the part `B` of the current partition. Four facts, each
either proved here or marked as the place a hypothesis enters.

**(E1) The measure is uniform on each part's survivors, and every live part keeps its `P_0` mass.**
Proved by induction with `delta = 1/2`, provided every earlier `alpha <= 1/2`. On a part, the
factor on `S_g` is `max{0, 2 - 1/alpha} = 0` for `alpha <= 1/2` and the factor off `S_g` is the
constant `1/(1-alpha)`; mass per part is preserved (r52 section 2.2, step 1). For the refining
head the parts of the NEXT stage are the classes mod `Q_{<i} g_i`, each of which lies inside one
part of the current stage and is entirely struck or entirely unstruck by `g_i`, so uniformity is
inherited. At the freeze the same holds for the classes mod `Q_s`. **Consequence: `alpha_g(B)` is
the plain counting ratio (struck survivors)/(survivors), and the live fibres carry EQUAL weight
`1/N_live`, not weight proportional to their survivor counts.** That last point is what separates
the frozen partition from one block: an over-struck fibre is renormalised, not averaged away.

**(E2) The head is exact.** For `g_i in S` the part is a full class mod `Q_{<i}`, so every column
of a live part survives; the part is an arithmetic progression of `m_i = L/Q_{<i}` columns of step
`Q_{<i}`, coprime to `g_i`, so it meets each class mod `g_i` either `floor(m_i/g_i)` or
`ceil(m_i/g_i)` times and

    alpha_{g_i} = 2/g_i + theta,   |theta| <= 2/m_i ,      exactly, no hypothesis,

whenever `Q_{<i} g_i <= L`. So `term = 4/g^2` up to the ceiling. PROVED.

**(E3) The freeze buys exactly one more gear, and no more.** After the head, `P` is uniform on the
live fibres and every column of a live fibre survives, so the FIRST big gear also has
`alpha = 2/g + O(2/m)` exactly, with `m = L/Q_s`. PROVED. From the SECOND big gear on, the fibre is
only partly alive; `P` is uniform on the fibre's survivors, but those survivors are not a union of
classes mod `g`, and

    alpha_g(f) = (2/g) rho_g(f) ,   rho_g(f) = 1 iff U_{<g} ∩ f is equidistributed mod g.

`rho_g(f) = 1` is a THEOREM exactly when the fibre holds whole classes mod `Q_{<g} g / Q_s`, i.e.
when `Q_{<g} g <= L` - which is r52's head condition verbatim, three to five gears. **THE
HYPOTHESIS ENTERS HERE**, and nowhere else.

**(E4) The budget in two pieces.** With the equal fibre weights,

    M_g^(2) = (mean_f alpha_g(f))^2 + Var_f(alpha_g) ,
    D_f(g)  = (strikes of g on U_{<g} ∩ f) - 2 s_f/g ,      alpha_g(f) = 2/g + D_f/s_f .

Over a full period of `Q_s` times the gears below `g` times `g`, every `D_f = 0` by CRT (verified
exactly, R5), so `eta_SF = sum 4/g^2 < 0.36455` there. **On the window the entire content of the
branch is `sum_g Var_f(alpha_g)`, and it is an interval discrepancy: the L2 discrepancy of gear
`g`'s strikes across the residue classes of `Q_s`.**

### 2.2 Where N1 is satisfied (the exactness range)

N1 says a covering argument must keep exact first moments. Per fibre the first moment is exact up
to the ceiling `2/m` iff the fibre holds whole classes mod `g`, which needs `Q_s g <= L`. Two
readings of "the sub-machine":

* **the brief's default** (`Q_s` = the largest gear-primorial below `W`) gives fibres of
  `m = 1.58, 4.37, 1.48, 8.41, 1.99, 7.85, 2.58` columns at `q = 59, 97, 199, 499, 997, 1999,
  4999`. At `q = 997` that is `169514/85085 = 1.99` columns per fibre: **the first moment is exact
  for no big gear at all**, and the engine there is a relabelled collapse.
* **the exactness cut** (`Q_s` the largest gear-primorial with `Q_s q <= W`, so EVERY gear has
  whole classes in every fibre) is `Q_s = 1` for `q < 23`, `5` from `q = 23`, `35` from `q = 199`,
  `385` from `q = 2297`, `5005` from `q = 29989`. The sub-machine grows like the inverse primorial
  of `q/6`, i.e. `~ ln q / ln ln q` gears, and it reaches four gears only at `q ~ 3e4`.

Everything below is computed at both cuts; the exactness cut is the one N1 admits.

---

## 3. Results

### R1. The exactness range (`sf_engine.py`, `sf_disc.py`)

    the exactness cut Q_s(q) = largest gear-primorial with Q_s q <= W(q)
      from q = 5      Q_s = 1
      from q = 23     Q_s = 5       {5}
      from q = 199    Q_s = 35      {5,7}
      from q = 2297   Q_s = 385     {5,7,11}
      from q = 29989  Q_s = 5005    {5,7,11,13}

    the brief's default cut (largest gear-primorial below W) and its fibre length
      q      59      97     199     499     997    1999    4999
      Q_s   385     385    5005    5005   85085   85085  1616615
      m    1.58    4.37    1.48    8.41    1.99    7.85    2.58

P1 confirmed. The default cut violates N1 at every machine.

### R2. The exact budget by cut (`sf_engine.py` part A)

Real teeth, real window, gears increasing, `delta = 1/2`, the engine's own measure. `Q_s = 1` is
r52's one-block engine; the last column refines at every gear and is r51's.

    q = 997   L = 169514   166 gears   sum 4/g^2 = 0.36404
      Q_s        m     #fibres   eta_SF    sum V_g   max alpha
         1  169514.0         1   0.3655    0.0000     0.4000
         5   33902.8         5   0.3655    0.0001     0.4000
        35    4843.3        35   0.3661    0.0007     0.4000
       385     440.3       135   0.3739    0.0084     0.4000
      5005      33.9      1485   0.4943    0.1289     1.0000
     85085       2.0     22275   1.6863    1.3221     1.0000
      refining (r51)             1.8484

    the same at every machine (eta_SF by Q_s):

      q      Q_s=1     5        35       385      5005     85085    refining
      59    0.3582  0.3622   0.4029   0.9476       -        -       1.1142
      97    0.3585  0.3609   0.3839   0.7204       -        -       1.2159
     199    0.3652  0.3660   0.3730   0.4792   1.3624       -       1.4739
     499    0.3659  0.3661   0.3679   0.3932   0.8494       -       1.7008
     997    0.3655  0.3655   0.3661   0.3739   0.4943   1.6863      1.8484
    1999    0.3651  0.3651   0.3653   0.3676   0.4017   1.1569      2.0071

**The budget is monotone increasing in `Q_s` at every machine, with no exception in 27 cells.**
The two ends reproduce the siblings: `Q_s = 1` against r52's `0.35816, 0.35855, 0.36523, 0.36586,
0.36545`, and the refining column against r51's `1.074, 1.224, 1.469, 1.707, 1.905` - a
cross-validation of the implementation against both.

So there is no sweet spot: the finer the partition, the larger the budget, and CRT evaluates
`alpha` only at the fine end. P2 confirmed.

### R3. The cross-fibre term at the exactness cut (`sf_disc.py` part A)

`eta_real` is the machine's own budget; `eta_maxphase` replaces EACH gear's tooth pair by the
worst of its `(g-1)/2` pairs on the same survivors (a single-phase adversary, one gear at a time);
`eta_apriori` is the only bound available with no equidistribution input, `((2/g)/Pi_{<g})^2`.

      q     Q_s  fibres   sum V_g   eta_real  eta_maxphase  eta_apriori  max|D_f| @g   mean rank
     59       5       5   0.00410    0.36220      0.41142       1.9151    2.2 @13        0.540
     97       5       5   0.00231    0.36085      0.39212       2.2029    3.5 @17        0.477
    199      35      35   0.00778    0.37298      0.39241       2.6851    3.5 @43        0.481
    499      35      35   0.00209    0.36794      0.37400       3.0281    6.7 @59        0.543
    997      35      35   0.00065    0.36611      0.36865       3.1886   11.2 @59        0.597
   1999      35      35   0.00020    0.36528      0.36659       3.2919   15.9 @47        0.568

Three readings.

* **The cross-fibre term is tiny and shrinking**: `sum V_g = 0.0041 -> 0.0002` against room
  `0.636`. The whole budget is `sum 4/g^2` plus a thousandth.
* **A single adversarial phase buys almost nothing, and less and less.** `eta_maxphase` falls
  `0.411, 0.392, 0.392, 0.374, 0.369, 0.367` towards `sum 4/g^2 = 0.364`. Even with every gear
  re-phased to its worst pair against the real survivors, the budget is a third of the room. The
  reason is structural (section 4): the gear has ONE phase and the `Q_s` fibres see `Q_s`
  different induced phases, so no phase can be bad in all of them.
* **The gap to a theorem is one number**: `eta_apriori / eta_maxphase = 4.65, 5.62, 6.84, 8.10,
  8.65, 8.98`. That is the exact size of the missing input, and it is GROWING.

The real teeth's rank among all `(g-1)/2` pairs averages `0.48 .. 0.60` - undistinguished, as 7b
found on the section.

### R4. The unconditional envelope SF-CAP (`sf_envelope.py`) - the deciding computation

    alpha_g(f) <= min(1, 2 ceil(m^+/g) / (m^- - sum_{a < h < g} 2 ceil(m^+/h))) ,   m = L/Q_s,

accepted only while that bound itself stays `<= 1/2`, which makes E1's `alpha <= 1/2` condition
self-verifying, so the envelope assumes nothing at all.

**On the real window it is vacuous at every machine.** At the best cut, `eta` at `L = W(q)` is

      q      17     19     23     29     31     37     41     43     59     97    199    499    997   1999
    eta    1.08   1.19   1.70   1.34   1.63   1.00   1.05   1.72   1.20   1.08   1.21   1.33   1.20   1.17

(each with the `<= 1/2` condition already broken, so the honest value is infinite). The closest any
window comes is `q = 37` at `1.0033`. P6 confirmed: **the sub-machine fibre engine proves nothing
unconditional about any window.**

**The threshold, and the comparison inside one envelope.** `L*_SF(q)` is the shortest interval
SF-CAP can rule out, minimised over the cut; the last two columns are the SAME envelope at the two
extremes of the cut (`t = 0` is the one block, `t = all` is the full refinement).

      q  gears   best cut a   Q_s(a)      ln a/ln q  limit eta   L*_SF(q)   L*/W(q)   t=0     t=all
     17      5        5       5             0.568     0.486      1.30e2     2.28e0    none    8.51e4
     43     12        7       35            0.517     0.806      1.26e4     3.49e1    none    1e15
     59     15       11       385           0.588     0.554      4.20e4     6.88e1    none    1e20
     97     23       13       5005          0.561     0.599      1.69e6     1.00e3    none    1e35
    199     44       23       3.72e7        0.592     0.469      1.70e10    2.30e6    none    1e81
    499     93       41       5.07e13       0.598     0.479      1e17       2.72e12   none    1e205
    997    166       61       1.96e22       0.595     0.442      1e25       5.89e20   none    1e414
   1999    301       97       3.84e35       0.602     0.424      1e39       8.29e33   none    1e841

**The sub-machine cut beats both extremes of its own family: the one block never reaches a finite
threshold at any `q` (r52's CAP, vacuous), the full refinement needs `10^5` to `10^841` columns,
and the intermediate cut needs `10^2` to `10^39`** - better by 3 to 801 orders of magnitude. The
cut sits at `ln a / ln q = 0.517 .. 0.660`, mean `0.583`, against the predicted `e^{-1/2} = 0.6065`
(the cut must make `sum_{a<g<=q} 2/g < 1`, and `2(lnln q - lnln a) < 1` gives exactly that).

So `L*_SF(q) = exp((1+o(1)) theta(q^{0.60}))`: still super-polynomial where the target is `q^2/6`,
with a smaller exponent than the `0.728` r51 reported. P3 confirmed.

### R5. The obstruction, measured and priced (`sf_disc.py` parts B, C, D, E)

**The full-period control.** On a full period of `{5..g}` with `Q_s = 35`, every per-fibre
discrepancy is exactly zero:

    gears below 11 on the period of {5..11} =    385, fibres of   11 columns: max|D_f| = 0.000e+00
    gears below 13 on the period of {5..13} =   5005, fibres of  143 columns: max|D_f| = 0.000e+00
    gears below 17 on the period of {5..17} =  85085, fibres of 2431 columns: max|D_f| = 0.000e+00

So the object is purely an interval effect, as E4 says.

**Pricing a uniform bound `|D_f(g)| <= C`.** With `alpha_f = 2/g + D_f/s_f`,
`eta <= sum_g (2/g + C/s_min(g))^2`; `C*` is the largest `C` keeping that below 1.

      q     Q_s   room     C*      max|D_f|  rms|D_f|  C*/max   eta at C = 30
     59       5  0.6484    6.11       2.2      1.62     2.74      9.58
     97       5  0.6427   11.35       3.5      2.16     3.22      3.68
    199      35  0.6385    4.79       3.5      1.87     1.38     18.92
    499      35  0.6366   15.28       6.7      2.89     2.27      2.53
    997      35  0.6360   38.11      11.2      5.62     3.40      0.78
   1999      35  0.6357   94.80      15.9      7.95     5.98      0.45

and the growth, at the exactness cut:

      q       43     59     97    149   |  199    307    499    701    997   1499   1999
    C*      4.34   6.11  11.35  18.24   | 4.79   8.14  15.28  24.10  38.11  65.03  94.80
    max|D|   2.0    2.2    3.5    3.9   |  3.5    4.9    6.7    8.8   11.2   14.4   15.9
    C*/max  2.17   2.74   3.22   4.72   | 1.38   1.67   2.27   2.75   3.40   4.50   5.98

(the break is where the cut changes from `Q_s = 5` to `Q_s = 35`). `C*` grows like `q^{1.30}`
between `q = 199` and `q = 1999`; the measured discrepancy grows like `q^{0.66}`, the square-root
rate; **so the machine's headroom widens with `q`, from 1.38 to 5.98.**

**Against 7b.** 7b proved/measured the WINDOW-aggregate discrepancy of a gear on the anchor's
openings: `|D_fresh| <= 6.4` at the first gears above the anchor at every level to 5000, and
`max |D_raw| = 26.3` over all 663 levels and all anchors. Two things follow, pulling opposite
ways:

* a bound of that SIZE is enough from `q` between 701 (`C* = 24.1`) and 997 (`C* = 38.1`) on:
  `eta` at `C = 30` is `0.78` at `q = 997` and `0.45` at `q = 1999`. The branch does not need a
  saving in the discrepancy - a fixed constant would do, for all large `q`;
* but 7b's object is not this one. 7b bounds a SIGNED first moment summed over the whole window;
  `eta` needs `mean_f (D_f/s_f)^2`, a mean SQUARE resolved per residue class of `Q_s`. A signed
  aggregate bound implies nothing about an L2 spread - cancellation across fibres is exactly what
  the square removes. **7b's rigidity does not bound the cross-fibre term, and the reason is not
  the size of the constant but the norm.**

**Where the variance sits** (`q = 997`, `Q_s = 35`, 35 fibres): `V_g` is `0.000000` through
`g = 19`, rises to `~5e-6` in the middle band, falls back at the top; the per-fibre `max |D_f|`
peaks at `11.2` at `g = 59` and settles at `1.2 .. 4.8` for the top gears, while `min surv/fibre`
falls from 4843 to 527. The worst single pair beats the real pair by at most a factor 3 in
`M^(2)` at any one gear, and the real pair's rank is at the top (1.000) for the middle band
`g = 137 .. 419` and at the bottom (0.002) for the top gears - 7b's band law in the fibre
coordinate.

### R6. The adversarial gate, validity (`sf_gate.py` part A)

r50's MILP witnesses give, for each `K`, an interval of exactly `A(K) - 1` columns that `K` primes
DO cover, with the explicit strike sets. The engine forbids `eta_SF < 1` there.

      K   L = A(K)-1   moduli                              min over all cuts of eta_SF  verdict
      4       15       5,7,11,17                                 1.2063 (t=1)             ok
      5       21       5,7,11,23,29                              1.5749 (t=1)             ok
      6       27       5,7,11,17,37,53                           1.7208 (t=1)             ok
      7       36       5,7,11,13,17,19,31                        1.2552 (t=2)             ok
      8       44       5,7,11,13,19,29,31,83                     1.3172 (t=2)             ok
      9       67       5,7,11,13,17,23,31,37,47                  1.4845 (t=2)             ok
     10       87       5,7,11,13,17,19,23,29,37,79               1.6099 (t=2)             ok
     11      100       5,7,11,13,17,19,29,31,37,47,71            1.5575 (t=1)             ok
     12      114       5,7,11,13,17,19,29,31,37,43,53,79         1.5318 (t=2)             ok

**Nine witnesses, every cut, no failure; the minimum is 1.21.** The engine and the implementation
pass the gate.

### R7. The adversarial gate, reach - and a correction to r51's positive (`sf_gate.py` parts B, C)

      K   gears     A(K)   W_{K+1}   L*_SF(K)  cut   L*/W    proves lemma?   L* >= A(K)-1?
      1   ..5          2        8          4     0   0.500       YES              ok
      2   ..7          5       20         20     1   1.000       no               ok
      3   ..11         7       28         30     1   1.071       no               ok
      4   ..13        16       48         60     1   1.250       no               ok
      5   ..17        22       60        150     1   2.500       no               ok
      6   ..19        28       88        350     2   3.977       no               ok
      7   ..23        37      140        560     2   4.000       no               ok
      8   ..29        45      160       1120     2   7.000       no               ok
      9   ..31        68      228       1680     2   7.368       no               ok
     10   ..37        88      280       2555     2   9.125       no               ok
     11   ..41       101      308       4900     2  15.909       no               ok
     12   ..43       115      368      15680     2  42.609       no               ok

Consistency holds at all 12 (`L*_SF >= A(K) - 1` everywhere), and the unconditional envelope proves
the open lemma only at `K = 1`. The ladder is exactly where the sub-machine's advantage cannot
appear: `W_{K+1} ~ 300` columns leave `m = L/Q_s` in the tens, and the ceiling `2/m` swamps `2/g`
for every gear above 13.

**Why this is not simply weaker than r51.** r51's `L*max(K)` (`2.0, 5.0, 10.3, 16.1, 25.0, 40.3,
76.6, 106, 159, 255, 633, 1025`) uses the step `M^(1) <= 2 ceil(L/g)/L`, the value the first moment
would have under the UNIFORM measure. Under the engine's own measure `M^(1)` is `P_{i-1}(B_i)`, and
`P_{i-1}` is uniform on the LIVE parts only, so that step assumes the gear's strikes are shared
fairly between live and dead parts. On the covered witnesses it is false:

      K     L    worst gear    M^(1) (engine)   2 ceil(L/g)/L   ratio
      4    15        17           0.38889          0.13333       2.917
      5    21        23           0.33333          0.09524       3.500
      6    27        37           0.20833          0.07407       2.813
      7    36        31           0.20000          0.11111       1.800
      8    44        83           0.10073          0.04545       2.216
      9    67        37           0.21667          0.05970       3.629
     10    87        79           0.11269          0.04598       2.451
     11   100        37           0.16667          0.06000       2.778
     12   114        83           0.09782          0.03509       2.788

**Nine refuting instances, ratios 1.80 to 3.63.** So R2.c's "the localised budget PROVES
`A(K) < (p_{K+1}^2-1)/6` for every `K <= 10`" is conditional on an equidistribution step of exactly
the kind the wall keeps naming, not unconditional. SF-CAP is what remains when that step is
removed, and it reaches `K = 1`.

---

## 4. Mechanism

**What the freeze buys, exactly.** In BBMST the fibre at stage `i` is a full class mod `Q_{<i}`,
and the reason the method works is not that the fibre is long but that the MEASURE is constant on
it: `P_{i-1}` is a measure on the base `Z_{Q_{i-1}}` lifted uniformly, so `alpha_i` is a counting
proportion and the CRT evaluates it with no input. The non-uniformity the reweighting creates
inside a fibre at stage `i` is exactly resolved by the finer base at stage `i+1`. That is the whole
design, and it is why the partition must refine - and why, on an interval, it collapses.

Freezing the partition at `Q_s` keeps the fibres long for ever, and the head still kills WHOLE
fibres, so at the freeze the measure is still constant on each live fibre and the first big gear is
still evaluated by CRT. But the second big gear meets a measure that is uniform on the fibre's
SURVIVORS, not on the fibre, and the CRT has nothing to say about that. **The freeze buys exactly
one gear of exact evaluation and then hands back the same conditioning r52 met with blocks.**

**The three regimes in one line.** Let the partition be the classes mod `Q`, `Q | Q_s`:

    Q = 1        one block     budget 0.365, minimal, nothing evaluable  (r52: the trivial criterion)
    Q = Q_s      fibres        budget 0.365 + sum Var_f, one gear exact  (this branch)
    Q refining   full fibres   budget -> the union bound, all exact      (r51: the collapse)

and the budget is monotone increasing along that line (27 cells, no exception). Evaluability and
budget are the same axis, pulling opposite ways. There is no sweet spot because there is nothing to
trade: what a finer partition buys in CRT it pays in second moment, exactly.

**Where the sub-machine does win, and why.** Not in the budget - in the DENOMINATOR of the
unconditional bound. In a block, gears 5 and 7 have already removed `2/5 + 2/7 = 0.686` of the
columns before the first big gear acts, and a theorem cannot say which, so the survivor count in
the denominator of `alpha` starts at `0.314` and the first six terms sum to `1.078` (r52's E3). In
a live fibre the head has removed NOTHING - the fibres it kills are other fibres - so the
denominator starts at `1` and the tail terms are `((2/g)/(1 - sum_{a<h<g} 2/h))^2`, each of order
`4/g^2` until the running kill approaches 1. The unconditional budget is then limited by the
capacity of the TAIL alone, and the cut must satisfy `sum_{a<g<=q} 2/g < 1`, i.e.
`ln a / ln q > e^{-1/2}`. **The sub-machine converts the head's kills from a loss of density into a
loss of fibres, and that is worth 3 to 801 orders of magnitude in the addressable length.**

**Why it is still super-polynomial.** The condition `sum_{a<g<=q} 2/g < 1` forces the sub-machine
to contain every gear up to `q^{0.6065}`, so `Q_s = exp(theta(q^{0.6065}))`, and the interval must
hold enough fibres for the ceilings to be free: `L >= Q_s q` times a constant. The capacity sum
`2 log log q` diverges, so no fixed sub-machine ever suffices, and the primorial of a growing cut is
exponential. That is r51's shape with a better exponent, and the exponent comes from a clean place:
`e^{-1/2}` where r51 has `e^{-0.3175}`, because the collapsed tail pays `2/g` per gear where the
frozen tail pays `(2/g)^2/(1-c)^2`.

**The obstruction, named.** `sum_g Var_f(alpha_g)`, the L2 discrepancy of each gear's strikes
across the `Q_s` residue classes inside the window. It is exactly zero over a full period (checked)
and `0.0002 .. 0.008` on the window. The machine's own value is a two-thousandth of the room; the
single-phase adversary reaches a hundredth of it; the no-input bound is 5 to 9 times over it. The
missing statement is not about size but about norm: 7b proved a signed aggregate, the budget needs
a mean square per class.

---

## 5. What is new

Checked against `docs/novel/README.md` (index read this round; nothing on congruence fibres, on
sub-machine partitions, or on discrepancy across the residue classes of a sub-machine), r51 and
r52.

1. **The partition monotonicity, measured.** The distortion budget increases as the partition
   refines, from the one-block minimum `0.365` to the union bound, monotone at all 27 (machine,
   cut) cells; and CRT evaluability increases along the same axis. The collapse (r51) and the
   trivial criterion (r52, wall 5g) are the two ends of one line, and this branch is the line.
2. **The freeze buys exactly one gear.** The engine's measure is constant on a part only while the
   partition refines with the gears; freezing preserves it for the first gear above the sub-machine
   and no further. That is the precise content of "fibres of a sub-machine", and it is a theorem,
   not a measurement.
3. **SF-CAP: the first fully unconditional localised distortion bound on the tree.** Head gears
   kill whole fibres, so the survivor count inside a live fibre starts at `m`, not `m prod(1-2/g)`;
   the envelope is `sum_{g<=a} 4/g^2 + sum_{g>a} ((2/g)/(1 - sum_{a<h<g} 2/h))^2`, it assumes
   nothing (the `alpha <= 1/2` condition is self-verifying), and its threshold is
   `exp(theta(q^{0.60}))` where the one block never reaches a finite threshold and the full
   refinement needs `10^{841}` at `q = 1999`.
4. **The single-phase cross-fibre budget.** Replacing every gear's tooth pair by its worst pair on
   the same survivors leaves the budget at `0.411, 0.392, 0.392, 0.374, 0.369, 0.367` at
   `q = 59 .. 1999` - falling towards `sum 4/g^2`. One phase cannot be bad in `Q_s` fibres at once,
   and the freedom is worth less as `q` grows. This is the first measurement on the tree of what
   the phase adversary can actually buy against a CRT-structured partition, against what a counting
   bound must concede (4.7 to 9.0 times more).
5. **The needed discrepancy bound GROWS.** `C*(q)`, the largest uniform per-fibre discrepancy bound
   that closes the budget, is `4.8, 15.3, 38.1, 94.8` at `q = 199, 499, 997, 1999` (`~ q^{1.3}`),
   while the measured discrepancy grows like `q^{0.66}`. The covering side's residual therefore
   needs NO saving in `q` - a fixed constant of 7b's size suffices from `q ~ 10^3` on - and the
   difficulty is entirely at small `q` and entirely in the NORM: signed aggregate (7b, proved)
   against mean square per residue class (needed).
6. **Nine refuting instances for the first-moment step of r51's adversarial positive.** On r50's
   covered witnesses the engine's own `M^(1)` exceeds the uniform value `2 ceil(L/g)/L` by factors
   `1.80` to `3.63`. R2.c's `A(K) < W_{K+1}` for `K <= 10` is conditional on an equidistribution
   step; unconditionally the localised budget reaches `K = 1`.

---

## 6. Verdict

**Outcome (B)**: the sub-machine cut improves the exponent of the addressable length and yields the
first unconditional localised bound, and it stays super-polynomial. DEAD as a route to
`F(y) < y^2/6`; it leaves a theorem and a sharply located residual.

**What the sub-machine fibre engine proves unconditionally.**

> **(SF-CAP).** Let `I` be an interval of `L` columns and `g_1 < ... < g_n` primes `>= 5` with two
> classes each at any phases. Fix `t`, put `Q_{<i} = g_1...g_{i-1}`, `Q_s = Q_{<t+1} <= L`,
> `m^- = floor(L/Q_s)`, `m^+ = ceil(L/Q_s)`, and
> `alpha_i = min(1, 2 ceil(ceil(L/Q_{<i})/g_i) / floor(L/Q_{<i}))` for `i <= t`,
> `alpha_i = min(1, 2 ceil(m^+/g_i) / (m^- - sum_{t<j<i} 2 ceil(m^+/g_j)))` for `i > t`.
> If every `alpha_i <= 1/2` and `sum_i alpha_i^2 < 1`, the gears do not cover `I`.

Its reach: `K = 1` on the adversarial ladder; `L*_SF(q) = 1.3e2, 1.3e4, 4.2e4, 1.7e6, 1.7e10,
1e17, 1e25, 1e39` at `q = 17, 43, 59, 97, 199, 499, 997, 1999`, against windows `57 .. 668335`.
**No window of any machine from `q = 17` up.** Growth `exp((1+o(1)) theta(q^{0.6065}))`,
`0.6065 = e^{-1/2}`, from the cut condition `sum_{a<g<=q} 2/g < 1`.

**What it needs.** For each big gear `g` and each residue class `f` mod `Q_s` inside the window,
the survivors `U_{<g}` of the lower gears must meet `g`'s two classes near their fair share:

> **(the cross-fibre input).** `mean_f (D_f(g)/s_f)^2 <= (C/s_min)^2` with
> `D_f(g) = |U_{<g} ∩ S_g ∩ f| - 2 |U_{<g} ∩ f| / g`, uniformly over the gears, with `C <= C*(q)`,
> `C*(q) = 4.8, 15.3, 38.1, 94.8` at `q = 199, 499, 997, 1999`, growing like `q^{1.3}`.

**Is it again a level of distribution?** Precisely: it is the distribution of the sifted set
`U_{<g}` in the residue classes of the modulus `Q_s g`, inside a window of length `W ~ q^2/6`. With
the exactness cut `Q_s = 35`, `Q_s g <= 35 q ~ 35 sqrt(6W)`, so the **modulus is `W^{1/2+o(1)}`:
level 1/2** - the Bombieri-Vinogradov range in exponent, but demanded for every modulus and every
class, not on average, and for a sifted set rather than the primes. It is weaker in level than what
r51's collapsed gears need (moduli up to `exp(theta(g))`, the primorial) and stronger in uniformity
than what r52's one block needs (modulus `g` alone, one class pair). The sifting dimension is 2, so
an average-form input at this level is face A from the covering side, exactly as r51 said.

**Is it a discrepancy of a periodic set inside a short interval (7b's kind)?** Only for the gears
whose lower period fits the window: `Q_{<g} g <= W` holds for `g <= 11, 11, 13, 13, 17, 17` at
`q = 59 .. 1999`, and for those gears `rho_g(f) = 1` is a theorem by CRT - the same three-to-five
gear head r52 had, worth `0.275 .. 0.312`. From the sixth gear the period `Q_{<g}` exceeds the
window and `U_{<g}` is no longer periodic on the scale of the interval, so 7b's
interval-discrepancy argument (which does not use equidistribution of primes) does not extend
there. The answer is: **periodic-set-in-a-short-interval for the head (proved, five gears),
level-of-distribution for everything above it (open).**

**The exact residual, in one line.** `room = 1 - sum_{5<=g<=q} 4/g^2 = 0.6360` at `q = 997`; the
real machine spends `0.0007` of it, a worst-phase adversary `0.005`, and the only input-free bound
`2.82` - a factor `4.4` over. Nothing between the two is decided by any counting argument on the
tree.

---

## 7. Exceptionless, with the count

| statement | range | count | status |
|---|---|---|---|
| the engine's measure is uniform on each part's survivors, each live part keeping its `P_0` mass, while every `alpha <= 1/2` | all | proof (section 2.1 E1) | proved by induction |
| the head gears and the FIRST gear above the sub-machine have `alpha = 2/g` up to `2/m`, with no hypothesis | all, when `Q_s g <= L` | proof (E2, E3) | proved |
| `eta_SF` is increasing in the cut `Q_s` | `q = 59, 97, 199, 499, 997, 1999`, every admissible cut | 27 cells | exact, 0 exceptions |
| `eta_SF(Q_s=1)` equals r52's block budget, and the refining limit matches r51's fibre budget | `q = 59..997` | 10 | exact, cross-validated |
| every per-fibre discrepancy `D_f(g) = 0` on a full period of `{5..g}` | `g = 11, 13, 17`, `Q_s = 35` | 3 | exact |
| `sum_g Var_f(alpha_g) <= 0.0078` at the exactness cut | `q = 59..1999` | 6 | exact, worst `0.00778` at `q = 199` |
| the single-phase worst-pair budget stays below `0.412` and falls with `q` | `q = 59..1999`, every gear re-phased one at a time | 6 machines, 502 gear sweeps | exact |
| SF-CAP is vacuous at `L = W(q)` | every prime `q` tested from 17 to 1999 | 14 | exact, closest `1.0033` at `q = 37` |
| the SF-CAP threshold beats both extremes of its own cut family | `q = 17..1999` | 21 | exact, 3 to 801 orders |
| the exact `eta_SF >= 1` on every covered MILP witness, at every cut | `K = 4..12` | 9 witnesses x 5-13 cuts | exact, minimum `1.2063` |
| `L*_SF(K) >= A(K) - 1` (consistency with the exact ladder) | `K = 1..12` | 12 | exact |
| the engine's `M^(1)` exceeds `2 ceil(L/g)/L` on covered intervals | `K = 4..12` | 9 | exact, ratios `1.80 .. 3.63` |

---

## 8. Dead ends, with the refuting instance

* **"Fibres modulo the largest gear-primorial below the window keep the engine's content."** DEAD:
  those fibres hold `1.48` to `8.41` columns at `q = 59..4999`, so no big gear's first moment is
  exact and N1 is violated at the first gear above the sub-machine. The cut that satisfies N1 is
  the largest primorial with `Q_s q <= W`, which is `{5,7}` for `199 <= q < 2297`.
* **"A coarser congruence partition keeps the second moment fractional and beats the collapse."**
  DEAD as an improvement: it does keep it fractional (`0.366` against `1.85` at `q = 997`), but the
  budget is monotone in the cut and the coarse end is r52's one block, the trivial criterion (wall
  5g). Refuting instance: the whole monotone table, 27 cells.
* **"The freeze keeps CRT evaluability above the sub-machine."** DEAD: it keeps it for exactly one
  gear. From the second big gear the measure is uniform on the fibre's survivors, not on the fibre,
  and `rho_g(f) = 1` is provable only when `Q_{<g} g <= L` - r52's head condition, unchanged.
* **"7b's proved rigidity bounds the cross-fibre term."** DEAD: 7b bounds a signed aggregate over
  the window; the budget needs a mean square per residue class of `Q_s`, and no signed bound
  implies an L2 one. (The SIZE would be enough: `C = 30` gives `eta = 0.78` at `q = 997`.)
* **"The localised distortion budget proves `A(K) < W_{K+1}` for `K <= 10` unconditionally."** DEAD
  as stated, refuted by r50's own covered witnesses: the step `M^(1) <= 2 ceil(L/g)/L` fails under
  the engine's measure by factors `1.80` to `3.63` (nine instances, `K = 4..12`; worst `K = 9`,
  `g = 37`, `0.21667` against `0.05970`). Unconditionally the localised budget reaches `K = 1`.
* **"Choose the cut better."** DEAD: the cut is optimised over every admissible value at every `q`
  in both tables; the capacity sum forces `ln a / ln q > e^{-1/2}`, so `Q_s` is the primorial of a
  growing gear and `L*` is super-polynomial for every choice.

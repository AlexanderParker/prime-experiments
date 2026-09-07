# Branch: THE POSITION-LENGTH FRONTIER

Prover, round 65 (2026-09-06). Parent: node **7d** (research/proof/anchor_runs_zero.md) --
"exclusive kills start at g^2, so the effective machine at column k is {5..sqrt(6k+1)}, and the
region past zero is thinner than the period mean". The observation that spawned this branch is
7d's mechanism read as a *location* statement rather than a *density* one: if the effective
machine at column k is only {5..sqrt(6k+1)}, then a blocked stretch of length L cannot begin
before the effective machine is big enough to have a record of length L, and the record law
F(y) ~ y^2/24 turns that into a linear frontier x >= 3L.

Scripts in `research/anchor235/r65/` (prefix `pf_`), results in
`research/anchor235/r65/results/` (gitignored; every number this document relies on is written
into the document). Run with `uv run python research/anchor235/r65/<script>.py` from the
repository root; each script is self-contained (numpy only) and stays inside 3 cores / 3 GB.
Nothing is committed.

## Setup: the exact definitions

Vocabulary as docs/proof-search/alignment-rules.md section 0. Column `k` is the pair
`(6k-1, 6k+1)`; gear `g` strikes `k` iff `k = +-u_g (mod g)` with `u_g = 6^{-1} mod g`; the
machine is `M = {5..q}`, its period `P = prod g`; an **opening** is a column no gear strikes;
`F(M)` is the longest gap between consecutive openings (max-gap convention, wrap included), so
the longest **blocked run** has `F(M) - 1` columns. Window at rung `q` = the columns
`(q/6, W]`, `W = (q'^2-1)/6`, `q'` the next prime. Column 0 is open for every machine (`6*0+-1 =
+-1` is divisible by no gear), so every run starts at column `>= 1`.

**The frontier.** For a machine `M` and a column range `R` (the whole period, or the prefix
`[1, W]`):

- `R_min^>=(L)` = the least `x` in `R` such that columns `x, x+1, ..., x+L-1` are all blocked.
- `R_min^=(L)`  = the least `x` in `R` such that `x .. x+L-1` are blocked and `x-1`, `x+L` are
  open (a **maximal** run of length exactly `L` starting at `x`).

Both are read off the list of maximal runs: `R_min^>=(L) = min{start : length >= L}` and
`R_min^=(L) = min{start : length = L}`. `R_min^>=` is monotone by construction; `R_min^=` need
not be. `R_max` is the corresponding largest start.

**The effective machine.** `E(N) = {5..y}` with `y` the largest prime `<= sqrt(N)`. The claim
inherited from 7d and the layer/shadow law (docs/proofs/15) is that a column `k` is blocked
under `M` iff it is blocked under `E(6k+1)` -- exclusive strikes of a gear `g` open at `g^2`.
This branch tests that claim as stated, because the frontier is only as good as it is.

## Pre-registered (written before any script of this branch was run)

### The theory

A blocked stretch of length `L` beginning at column `x` is a blocked stretch of the effective
machine `E(6(x+L)+1)` plus the strikes of gears above `sqrt(6(x+L)+1)`, and those strikes are
never exclusive (layer law). So `L + 1 <= F(E)` and therefore

  `R_min(L) >= y_L^2/6 - L`,  `y_L` = the least prime with `F({5..y_L}) >= L + 1`.

With `F(y)` about `y^2/24` this is `R_min(L) >= 3L`: the position-length frontier. If a frontier
`R_min(L) >= c L` with `c > 0` held for **every** `L` and **every** machine, a stretch of length
`W` could not begin at column 0 and the window statement would follow in location form.

### Predictions, with numbers, and what refutes each

- **P1 (monotonicity and the mirror).** `R_min^>=` is monotone (gate, not a prediction).
  `R_min^=` is NOT monotone: at least one inversion at every machine m13..m23. The mirror
  `k -> -k (mod P)` maps the maximal-run list to itself, so `R_max^=(L) = P - R_min^=(L) - L`
  exactly, 0 exceptions at m7..m23. REFUTED by one mismatch.
- **P2 (the initial run).** `R_min^>=(L) = 1` for every `L <= d_0(M) - 1`, `d_0` the first
  opening past column 0 (`d_0 = 2, 3, 3, 5, 5, 5` at m7..m23). So `R_min(L)/L = 1/L` there, and
  the frontier statement is false at small `L` at every machine.
- **P3 (the minimum ratio).** `min_{L >= 6} R_min^>=(L)/L < 3` at every machine m11..m23, and
  the minimum is attained at `L = 6` or `L = 7`. REFUTED by a machine whose minimum is at
  `L >= 10` or above 3.
- **P4 (super-linear growth).** The frontier grows faster than linearly and roughly
  geometrically in `L`: `R_min^>=(F-1) > 10^3` at m23, and `R_min^>=(L+1)/R_min^>=(L) > 1.3` on
  average over the top half of the range at m19 and m23.
- **P5 (the window frontier).** In the prefix `[1, W]` at rung `q`, `R_min^>=(L) = 1` for every
  `L <= d_0 - 1` with `d_0 - 1` about `q/6` (169 at `q = 997`), so **`R_min(L) >= 3L` is FALSE
  in the window at every rung**, and no `L_0` below `d_0` exists. Past the initial run the
  frontier is a statement about first occurrences of twin gaps. Prediction: the initial run is
  never the longest run of the prefix (`d_0 - 1 < F_W` at every rung 23..997); REFUTED by one
  rung where the longest run of `[1, W]` starts at column 1.
- **P6 (position of the window's longest stretch).** Pre-registered YES as briefed: the longest
  stretch of the window sits in the upper half, i.e. its start is `> W/2` at a majority of rungs
  and the median fraction `x/W` exceeds 0.5.
- **P7 (mechanism: no exclusive big-gear strike).** At every frontier-attaining stretch with
  `x > (q+1)/6`, the number of columns struck ONLY by gears above `sqrt(6(x+L)+1)` is 0 --
  0 exceptions. Pre-registered doubt, stated before computing: at `x = 1` this must FAIL,
  because gear `g`'s own column `(g -+ 1)/6` carries the member `g` itself, a prime that no
  smaller gear divides; so the layer law's "first exclusive strike at `g^2`" has an exception
  set, and it is exactly the columns of twin gear pairs, all of them below `(q+1)/6`.
- **P8 (the induction).** The effective-machine bound is correct (0 violations of
  `R_min(L) >= y_L^2/6 - L` over `x > (q+1)/6`) and weak: weaker than the truth by a factor
  above 10 at `L = F - 1` at m23. It does NOT close the window statement, and the exact step
  that fails is `x = 1`, where the effective machine is `{5..sqrt(q)}` while the number of
  exclusive big-gear strikes `S` is of order `pi(q)`, so `L <= F_{S+1}(E)` is vacuous.
- **P9 (the family).** The `x = 1` anomaly is REAL-TEETH, not teeth-free: on 20 tooth-
  counterfactual members at m13..m19 the run starting at column 1 has median length 1 or 2
  against the real machine's `d_0 - 1`, and the family's minimum ratio over `L >= 6` is larger
  than the real machine's at a majority of members. REFUTED if the family reproduces the real
  initial run within a factor 2.

### Scorecard

| # | Prediction | Verdict |
|---|---|---|
| P1 | `R_min^=` non-monotone; mirror exact | |
| P2 | `R_min = 1` up to `d_0 - 1` | |
| P3 | min ratio `< 3` over `L >= 6`, attained at `L = 6` or 7 | |
| P4 | geometric growth; `R_min(F-1) > 10^3` at m23 | |
| P5 | window: no `L_0`; initial run never longest | |
| P6 | window's longest stretch in the upper half | |
| P7 | 0 exclusive big-gear strikes above `(q+1)/6`; exception set at `x = 1` | |
| P8 | bound correct, weak by `> 10x`, does not close | |
| P9 | the anomaly is real-teeth | |

(Results, mechanism, induction attempt, verdict and dead ends are appended below as the branch
runs.)

---

# Results

Every number below is exact (full periods, or exact sieves of the prefix `[1, W]`); nothing is
sampled except the 20-member counterfactual families, which are labelled as such.

## 0. The reduction, and the gate

Two facts run through everything, both proved here in a line and both gated numerically.

**(R) The prefix is the twin pairs.** For `1 <= k <= W = (q'^2-1)/6`, a member `6k+-1` is `1`, a
prime `<= q` (a gear, struck), a prime `> q` (not struck), or composite -- and a composite below
`q'^2` has least prime factor `<= q` unless it is `q'^2` itself. So the openings of `{5..q}` in
`[1, W]` are exactly the twin-prime columns with `6k-1 > q`, plus the column `W` when `q'^2 - 2`
is prime. GATE: direct sieving of `{5..q}` over `[0, W]` reproduces the reduced opening set
exactly at `q = 23, 59, 97, 211` (4 of 4).

**(E) The effective-machine theorem.** *For every column `k` with `6k - 1 > q`, `k` is blocked
under `{5..q}` if and only if it is blocked under `{5..floor(sqrt(6k+1))}`.*
Proof: let `n = 6k+-1` have a prime factor `p` in `[5, q]`. If `p <= sqrt(n)` we are done.
Otherwise `m = n/p < sqrt(n)`; `m > 1` because `n > q >= p`; `m` is coprime to 6, so its least
prime factor `r` satisfies `5 <= r <= m < sqrt(n) < q'`, hence `r <= q` and `r` is a gear below
`sqrt(n)`. []

The hypothesis `6k - 1 > q`, i.e. `k > (q+1)/6`, is not decoration: section 3 below shows the
exception set is non-empty and identifies it exactly.

## 1. The frontier on full periods, m7 .. m29

`pf_period.py` (m7..m23, complete run lists) and `pf_period29.py` (m29, the period of
1,078,282,205 columns streamed as 29 re-phased copies of m23's, Pareto staircase only).
The `F` ladder is reproduced exactly at every machine (5, 7, 11, 18, 25, 34, 43) as a gate.

**The frontier is bimodal, and the two modes are separated by `d_0`.**

| machine | `P` | `F-1` | `d_0` | `R_min^>=(L)` for `L < d_0` | `min_{L >= d_0} R_min^>=(L)/L` | at `L` |
|---|---|---|---|---|---|---|
| m7  | 35 | 4 | 2 | 1 | **3.250** | 4 |
| m11 | 385 | 6 | 3 | 1 | **3.250** | 4 |
| m13 | 5,005 | 10 | 3 | 1 | **3.250** | 4 |
| m17 | 85,085 | 17 | 5 | 1 | 6.778 | 9 |
| m19 | 1,616,615 | 24 | 5 | 1 | 4.625 | 24 |
| m23 | 37,182,145 | 33 | 5 | 1 | 4.625 | 24 |
| m29 | 1,078,282,205 | 42 | 7 | 1 | 4.625 | 24 |

`R_min^>=(L) = 1` for every `L <= d_0 - 1` and `R_min^>=(L) >= 3.25 L` for every `L >= d_0`:
**0 exceptions in 113 (machine, L) cells at m7..m29**. There is nothing in between -- the ratio
jumps from `1/(d_0-1)` to at least 3.25 in one step.

**The staircase is tiny, and it is the same staircase.** The Pareto points (a run longer than
every run starting earlier) determine `R_min^>=` completely. The low ones do not move as gears
are added:

| machine | Pareto points `(x, L)` |
|---|---|
| m7  | (1,1) (8,2) (13,4) |
| m11 | (1,2) (13,4) (53,5) (151,6) |
| m13 | (1,2) (13,4) (53,5) (89,6) (123,10) |
| m17 | (1,4) (53,5) (61,9) (118,17) |
| m19 | (1,4) (53,5) (59,11) (111,24) |
| m23 | (1,4) (53,5) (59,11) (111,24) (40148,25) (170034,26) (190056,27) (396199,29) (1479278,30) (2553844,31) (5606403,32) (12694429,33) |
| m29 | (1,6) (59,11) (111,24) (5643,25) (23278,27) (35564,29) (102273,32) (2278076,34) (2900801,36) (6603768,37) (144154491,39) (200906186,42) |

Columns 13, 53, 59, 111 carry the whole low frontier of every machine from m11 on; adding a gear
lengthens the run already sitting there rather than creating an earlier one (`R_min^>=(24) = 111`
at m19, m23 **and** m29). The run at column 111 is the twin gap `661 -> 809` -- columns 111..134,
members 665..805 -- and it is the tight instance of the frontier at m19, m23, m29 and (section 2)
at every rung `q = 23..131`. The minimum ratio 3.25 is attained by the run at column 13
(columns 13..16, members 77..97), which is blocked by `{5, 7}` alone: `L = 4 = F({5,7}) - 1`, so
the effective-machine bound is **exactly tight** there.

**Above the staircase's knee the frontier explodes.** At m23 `R_min^>=` is flat at 111 up to
`L = 24` and then runs 40,148 / 170,034 / 190,056 / 396,199 / 1,479,278 / 2,553,844 / 5,606,403 /
12,694,429 for `L = 25..33` -- growth ratios 4.2, 1.1, 2.1, 1.0, 3.7, 1.7, 2.2, 2.3 -- ending at
0.341 of the period. At m29 the record stretch first occurs at `x = 200,906,186 = 0.186 P`.
So `R_min^>=(F-1) = 12,694,429` at m23: prediction P4 (`> 10^3`) held by four orders of magnitude.

**Mirror.** `R_max^=(L) = P - R_min^=(L) - L + 1` at **88 of 88 realised lengths** over
m7..m23, 0 mismatches. `R_min^>=` is monotone by construction; `R_min^=` is not, with 0, 2, 3,
5, 9, 10 inversions at m7..m23 (the first inversions are at `(3,4)` and `(6,7)`).

## 2. The frontier in the prefix `[1, W]`, rungs q = 23 .. 19,997

`pf_window.py`, 2,254 rungs, exact via reduction (R).

**(a) The frontier law survives, with the same exception set and the same tight instance.**
`R_min^>=(L) >= 3L` for every `L` in `[d_0, F_pre]` at **all 211 rungs where that range is
non-empty, 8,375 (rung, L) cells, 0 exceptions**; the minimum over all of them is **4.625**, at
`q = 23`, `L = 24` -- the same stretch at column 111. `L_0 = d_0` exactly: below `d_0` the ratio
is `1/L` at every rung, and no smaller `L_0` exists.

**(b) THE INITIAL RUN TAKES OVER.** The run starting at column 1 has length `d_0 - 1`, and
`(d_0-1)/(q/6)` lies in `[0.973, 1.352]` (median 1.005) over all 2,254 rungs -- it is `q/6` to
within a third. The window's longest stretch grows like `(log q)^2`. They cross, and the
crossing is permanent:

| band of `q` | rungs | mean init run | mean `F_win` | init/`F_win` | initial run is the longest |
|---|---|---|---|---|---|
| 23-100 | 17 | 11.0 | 28.0 | 0.382 | 0 % |
| 100-300 | 37 | 34.3 | 77.6 | 0.451 | 0 % |
| 300-1000 | 106 | 113.3 | 154.8 | 0.725 | 4.7 % |
| 1000-3000 | 262 | 335.5 | 248.5 | 1.342 | 78.6 % |
| 3000-10000 | 799 | 1080.4 | 415.4 | 2.548 | 100 % |
| 10000-20000 | 1033 | 2501.2 | 530.0 | 4.683 | 100 % |

The last rung whose longest prefix run is not the initial one is **q = 1423**; after it the
initial run is the longest at **2,038 of 2,038 rungs**, 0 exceptions, and the Pareto staircase of
the whole prefix collapses to the single step `(1, d_0-1)` (median staircase size over all rungs:
1; maximum 5). So from `q = 1427` on the position-length frontier of the prefix has **exactly one
point**, and it sits at column 1.

**(c) Position of the window's longest stretch (P6).** Restricted to starts above `q/6`,
`x/W` has median **0.755**, is above 0.5 at 86.3 % and above 0.75 at 51.1 % of rungs. Pre-
registered YES: HELD. Examples: `q = 997`, `x = 141,726` of `W = 169,680` (0.835), `L = 241`;
`q = 19,997`, `x = 65,136,289` of `W = 66,740,020` (0.976), `L = 633`.

**(d) The window statement in this coordinate.** `(d_0-1)/W` runs from 0.0375 at `q = 29` down to
0.000050 at `q = 19,997`: the initial run occupies a shrinking `~ 1/q` fraction of the prefix.

## 3. Mechanism at the frontier

`pf_mech.py`, factorising both members of every column of each stretch.

**(a) Above column 1 the big gears contribute nothing.** At every frontier-attaining stretch with
`x > 1` -- the 11 non-initial m23 Pareto stretches, the 8 min-ratio/record stretches at
m7..m29, and the window's longest stretch at `q = 23, 97, 211, 997` (23 stretches in all) --
the number of columns with **no** effective striker is **0**, and the effective machine leaves
the stretch in **one piece**: the big gears fuse nothing. Their strikes are few and always
shared: `q = 997`, `x = 141,726`, `L = 241`: effective machine `{5..919}`, 11 big gears, 7
strikes on 7 of 241 columns, all shared. `q = 211`, `L = 82`: 10 big gears, 9 strikes on 8 of 82
columns. So the ends-or-middles reading at the frontier is: **there are no ends and no middles;
the frontier stretch is a single brick of the effective machine.**

**(b) At column 1 the layer law has an exception set, and it is exactly the twin gear pairs.**
A column has no effective striker iff **both** members are prime, and it is blocked iff a member
is a gear; so the exception set is the set of twin columns whose smaller member is `<= q`.
Measured over the whole prefix `[1, W]`:

| `q` | `W` | exception columns found | twin pairs `(p, p+2)`, `5 <= p <= q` | all below `(q+1)/6` | largest exception column |
|---|---|---|---|---|---|
| 23 | 140 | 3 | 3 | yes | 3 |
| 29 | 160 | 4 | 4 | yes | 5 |
| 47 | 468 | 5 | 5 | yes | 7 |
| 97 | 1,700 | 7 | 7 | yes | 12 |
| 211 | 8,288 | 14 | 14 | yes | 33 |
| 401 | 27,880 | 20 | 20 | yes | 58 |
| 997 | 169,680 | 34 | 34 | yes | 147 |

7 of 7 exact matches, 0 discrepancies. This sharpens node 7d and the shadow law: a gear's
exclusive strikes open at `g^2` **except at its own column**, `(g -+ 1)/6`, and that column is
exclusive precisely when `g` and `g +- 2` are both prime. Every such column lies below `(q+1)/6`.

**(c) The initial run is a fusion, and only there.** At `q = 997` the run `[1, 169]` is
**28 pieces** of the effective machine `{5..31}` (sizes 6, 2, 1, 4, 4, 1, 4, 1, 4, 1, 4, 1, 4, 5,
11, 1, 4, 9, 7, 4, 2, 3, 2, 24, 1, 4, 3, 22) fused at 27 junctions, and the fusing gears are
`[41,43], [59,61], [71,73], [101,103], ... , [881,883]` -- twin gear pairs, one per junction, in
order. 157 big gears make 253 strikes on 154 of the 169 columns. At `q = 211`: 11 pieces, 10
junctions, all twin gear pairs. At `q = 23`: 2 pieces fused by `[11,13]`.

The two regimes are structurally different objects, not two ends of one distribution: above
column `(q+1)/6` a frontier stretch is one effective-machine brick; at column 1 it is a chain of
effective-machine bricks mortared by the machine's own twin gears.

## 4. The induction attempt

`pf_induction.py`.

**The theorem the effective machine gives.** Let the stretch be `x .. x+L-1`, all blocked, with
`6x - 1 > q`; let `N = 6(x+L-1)+1` and `y` the largest prime `<= sqrt(N)`. By (E) the whole
stretch is blocked by `E = {5..y}`, so `L + 1 <= F(E)`, hence

>  **R_min(L) >= bound(L) := ceil((y_L^2 - 1)/6) - L + 1**, where `y_L` is the least prime with
>  `F({5..y_L}) >= L + 1`.

Equivalently: if `F(y) <= y^2/(6(c+1))` for every prime `y < q`, then `x >= c L + c + 1 + 5/6`
for every blocked stretch of `{5..q}` starting above column `(q+1)/6`.

This is an **induction on the machine**, not a circularity: it uses `F` only at machines strictly
below `sqrt(6(x+L))`, which the ladder certifies to `y = 59` and the closure instrument to 41.
For any machine `{5..q}` whatsoever and any stretch whose top member is below `59^2 = 3481`, the
frontier `x >= 1.25 L` is therefore **unconditional**.

**What constant it delivers.** `F(y)/(y^2/6) = 0.480, 0.612, 0.348, 0.391, 0.374, 0.416, 0.386,
0.307, 0.362` at `y = 5..31`. The crude uniform form gives only `c = 0.634` (the worst `y` is 7).
The `L`-by-`L` form is stronger:

| `L` | 4 | 6 | 10 | 17 | 24 | 33 | 42 | 57 |
|---|---|---|---|---|---|---|---|---|
| `y_L` | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
| `bound(L)` | 5 | 15 | 19 | 32 | 37 | 56 | 99 | 104 |
| `bound(L)/L` | **1.250** | 2.500 | 1.900 | 1.882 | 1.542 | 1.697 | 2.357 | 1.825 |

So the certified ladder proves the frontier with **c = 1.25** for all `L`, and **c = 1.54** once
`L >= 6`. `c = 3` would need `F(y) <= y^2/24` at every `y`, which the ladder refutes at
`y = 5, 7, 11` (`F/(y^2/24) = 1.92, 2.45, 1.39`). The brief's guess `c ~ 3` is right about the
truth (measured 3.25) and wrong about what the record law buys (1.25).

**Slack.** measured / bound at the minimising `L`: 2.60, 2.60, 2.60, 3.60, 3.00, 3.00, 3.00 at
m7..m29. The bound is never violated for `L >= d_0` (113 of 113 cells) and is violated at every
`L < d_0` by exactly the factor `bound(L)` (4, 7, 6, 5 at `L = 1, 2, 3, 4`), because
`R_min = 1` there.

**Where the bound has content at all.** `y = q` when `6x = q^2`, so the effective machine is a
proper sub-machine only for `x < (q^2-1)/6`: the whole prefix except the new section
`(q^2, q'^2]`. At the top of the window the bound reads `L + 1 <= F(M)`, the root itself, which
is 5d.ii's stop line in location coordinates. Deeper in the period it is worse than vacuous --
the m29 record stretch sits at `x = 200,906,186`, where `E = {5..34,703}` is 1,197 times the
machine.

**Does the induction close the window statement? No, and the failing step is exact.**
The window statement is `d_0 <= W`, i.e. *the run starting at column 1 stops before `W`*. The
frontier's hypothesis is `6x - 1 > q`, which excludes column 1 and only column 1's run. So the
frontier's exception set **is** the object the window statement is about. Quantitatively, at
`x = 1`:

| `q` | init run `L` | effective machine | `S_excl` = columns with no effective striker | `L / S_excl` |
|---|---|---|---|---|
| 97 | 16 | `{5..7}` | 7 | 2.29 |
| 211 | 37 | `{5..13}` | 14 | 2.64 |
| 401 | 69 | `{5..19}` | 20 | 3.45 |
| 997 | 169 | `{5..31}` | 34 | 4.97 |
| 4999 | 834 | `{5..67}` | 125 | 6.67 |

`S_excl` is by definition the number of openings `E` leaves inside the run, so the spectrum-plus-
depth bound `L + 1 <= F_{S+1}(E)` with `S = S_excl` **is true by the definition of `F_J`** and
carries no information: at `x = 1` it degenerates to an identity. The same degeneracy in the
fusion language: the run at column 1 ends at the first `E`-opening that is not a twin gear pair,
and *that column is `d_0` by definition*. There is no mechanism there to exploit -- the
statement is its own restatement.

**So the honest form of the induction step.** `R_min(L) >= cL` at machine `q` **does** follow
from `F(y) < y^2/(6(c+1))` for `y < q` plus theorem (E) -- but only on `x > (q+1)/6`, and (E) is
unconditional, so the induction step closes cleanly with `c = 1.25`. What does not close is the
transfer to the root, and the reason is not the big-gear fusions at the top of the window (the
brief's guess): those are **zero** at every frontier stretch (section 3a). It is the bottom: the
machine's own gears, striking their own columns, block `[1, d_0-1]` with a structure the
effective machine cannot see.

## 5. The family: is `c` real-teeth?

`pf_family.py`, 20 members per machine, teeth `+-d_g` with `d_g` uniform on `[1, (g-1)/2]`,
seed 20260906; the real member `d_g = (g -+ 1)/6` computed in the same run as a control.

**(a) The constant.** `c = min_{L > init} R_min^>=(L)/L` on the full period:

| machine | REAL `c` | family min | family median | family max | members with `c` below REAL |
|---|---|---|---|---|---|
| m13 | **3.250** | 0.286 | 1.167 | 18.000 | 18 of 20 |
| m17 | **6.778** | 0.154 | 1.333 | 37.250 | 18 of 20 |
| m19 | **4.625** | 0.250 | 1.292 | 16.500 | 17 of 20 |

The real machine's frontier constant is 2.8 to 5.1 times the family median and sits at the
85th-90th percentile of its own family at all three machines. **`c` is real-teeth**, in the
direction of the real machine being a *worse* early blocker than a typical re-toothing.

**(b) The anomaly at column 1 is emphatically real-teeth, and it grows.** On the prefix `[1, W]`:

| rung | REAL init run | family median init | family max | REAL/family median | family members whose longest prefix run is the initial one |
|---|---|---|---|---|---|
| 211 | 37 | 8.5 | 38 | 4.35 | 0 of 20 |
| 401 | 69 | 6.5 | 17 | 10.62 | 0 of 20 |
| 997 | 169 | 12.0 | 69 | 14.08 | 0 of 20 |

0 of 60 counterfactual members has the initial run as its longest prefix run, at rungs where the
real ratio is already 0.45, 0.66, 0.70 and heading to 1. A random re-toothing has no notion of
"gear `g` divides the member `g`", and it does not block column 1. The real machine's initial run
is not a phase accident; it is the arithmetic identity that every gear is a member of the very
column set it sieves.

## 6. Toward the root: what would have to be proved

The window statement in location form is `R_min^>=(W) > 1`, i.e. *the frontier's exception set
stops below `W`*. The exception set is exactly `{L : L < d_0}` (0 exceptions, m7..m29 and 2,254
rungs). So:

- A frontier law `R_min(L) >= cL` valid for **all** `L >= L_0` gives the window statement iff
  `L_0 <= W`. Measured `L_0 = d_0`. "`L_0 <= W`" is `d_0 <= W`: the window statement itself.
  **The constant `c` is irrelevant to the root; the content is entirely `L_0`, and `L_0 = d_0`.**
- The frontier is nevertheless not empty toward the root: it bounds every stretch of the window
  that does not start at column 1, by `L <= x/c` with `c = 1.25` proved and 3.25 measured.
  At `q = 997` that reads `L <= 113,381` against the truth `L = 241`: 470 times too weak, because
  the effective machine at `x = 141,726` is `{5..919}`, 0.92 of the machine. The frontier's
  useful range in `x` is `[q/6, q^2/6]` and its strength decays to nothing across it.
- What a proof would need instead: a bound on the length of the run at `x = 1` in terms of the
  gear set, i.e. on how far `pi_2(q)` twin gear pairs can chain the effective machine's pieces.
  Section 4 shows that as posed this is `d_0` restated. Any non-circular version must forbid a
  *configuration* of twin gear pairs, not count them -- which is R2.a.i.a.1.a's cover-number
  obstruction on the other side of the same wall.

## 7. Exceptionless statements, with counts

1. `R_max^=(L) = P - R_min^=(L) - L + 1` (the mirror on the frontier): **0 mismatches in 88
   realised lengths**, m7..m23.
2. `R_min^>=(L) = 1` for `L < d_0` and `R_min^>=(L) >= 3.25 L` for `L >= d_0`: **0 exceptions in
   113 (machine, L) cells**, m7..m29 full periods; minimum 3.25 at `(x, L) = (13, 4)`.
3. The same law in the prefix `[1, W]` with constant 3: **0 exceptions in 8,375 (rung, L) cells**
   over the 211 rungs of `q = 23..19,997` where `[d_0, F_pre]` is non-empty; minimum 4.625.
4. Theorem (E) (blocked above `(q+1)/6` iff blocked by the effective machine): proved; and the
   exception set below `(q+1)/6` is exactly the twin gear pairs -- **7 of 7 exact count matches**
   at `q = 23..997`, every exception column below `(q+1)/6`.
5. **0 columns with no effective striker** at 23 frontier-attaining stretches with `x > 1`, and
   the effective machine leaves each of them in exactly **one piece**.
6. The longest blocked run of `[1, W]` is the initial run at **2,038 of 2,038 rungs** from
   `q = 1427` to `q = 19,997`.
7. `(d_0 - 1)/(q/6)` in `[0.973, 1.352]`, median 1.005, over **2,254 rungs**.
8. The bound `R_min(L) >= bound(L)` holds at every `L >= d_0` and fails at every `L < d_0`:
   **0 mixed cases in 136 cells**, m7..m29.

## Scorecard, filled

| # | Prediction | Verdict |
|---|---|---|
| P1 | `R_min^=` non-monotone; mirror exact | **HELD** both halves (inversions 2,3,5,9,10 at m11..m23; 88 of 88 mirror) |
| P2 | `R_min = 1` up to `d_0 - 1` | **HELD**, 0 exceptions, m7..m29 and 2,254 rungs |
| P3 | min ratio `< 3` over `L >= 6`, at `L = 6` or 7 | **REFUTED in both halves.** Over `L >= d_0` the minimum is 3.25 (m7..m13) / 4.625 (m19..m29), never below 3; and it is attained at `L = 4`, 9 or 24, never 6 or 7. The `< 3` reading came from including `L < d_0`. |
| P4 | geometric growth; `R_min(F-1) > 10^3` at m23 | **HELD**: 12,694,429 = 0.341 P; growth ratios 1.0-4.2 above the knee |
| P5 | window: no `L_0`; initial run never longest | first half **HELD** (`L_0 = d_0` exactly); second half **REFUTED, and it is the branch's main finding**: the initial run is the longest at 2,038 of 2,038 rungs from `q = 1427` |
| P6 | window's longest stretch in the upper half | **HELD**: median `x/W = 0.755`, above 0.5 at 86.3 % |
| P7 | 0 exclusive big-gear strikes above `(q+1)/6`; exception set at `x = 1` | **HELD exactly as pre-registered, including the stated doubt**: 0 in 23 stretches; the exception set is the twin gear pairs, 7 of 7 count matches |
| P8 | bound correct, weak by `> 10x`, does not close | **PART REFUTED, part held.** Correct: yes. Weak by `>10x` at the minimising `L`: **refuted** -- the factor is only 2.6-3.6, and the bound's own constant is 1.25 against a truth of 3.25. Does not close: **held**, with the failing step named exactly (`x = 1`, not the top of the window). |
| P9 | the anomaly is real-teeth | **HELD**: 0 of 60 family members reproduce it; real/family median init 4.35, 10.6, 14.1 and growing; `c` is real-teeth too (85th-90th percentile) |

# What is new

1. **The position-length frontier as an object**, with the exact staircase per machine. Not on
   the tree, and not in docs/novel/README.md (checked: "frontier" there is the covering-LP
   frontier of node 4.i.a, a different object; nothing indexes first occurrence by position).
2. **Theorem (E)** in its sharp form, with the hypothesis `6k - 1 > q` and the exact exception
   set. Node 7d's "exclusive kills start at `g^2`" is true in the window and false in the prefix;
   the exceptions are exactly the twin gear pairs, one per pair, all below `(q+1)/6`. The
   shadow law (docs/proofs/15e) is the `m >= q^2` half; the `m = q` half is what the initial run
   is made of.
3. **The frontier law**: `R_min^>=(L) >= 3.25 L` for `L >= d_0`, exceptionless in 113 period
   cells and 8,375 window cells, with the tight instance identified (columns 13..16 at `L = 4`,
   blocked by `{5,7}` alone, where the effective-machine bound is an equality; and columns
   111..134, the twin gap `661 -> 809`, at `L = 24`).
4. **The frontier theorem and its constant**: the effective machine converts a record-law bound
   `F(y) <= y^2/(6(c+1))` into `x >= cL`, unconditional for any machine on stretches whose top
   member is below 3481, and delivering `c = 1.25` from the certified ladder. The frontier
   constant and the record-law constant are the same number, read in two coordinates.
5. **The prefix's frontier collapses to one point.** From `q = 1427` to 19,997, 0 exceptions in
   2,038 rungs, the longest blocked run of `[1, W]` starts at column 1, and the Pareto staircase
   of the whole prefix has size 1. The hardest position in the window is, eventually and always,
   the very bottom -- and its difficulty is `d_0`, growing like `q/6` against a window record
   growing like `(log q)^2`.
6. **The mortar at column 1 is the twin gear pairs.** The initial run is a 28-piece fusion at
   `q = 997`, and every junction is a twin gear pair, in order. Nothing else in the prefix is a
   fusion at all: at 23 frontier stretches above column 1 the effective machine leaves one piece.
7. **Real-teeth verdicts**: both the constant `c` (85th-90th percentile of its own family) and,
   overwhelmingly, the column-1 anomaly (0 of 60 members).

# Verdict

**FACT, exact, not a route** -- with one proved theorem and one sharp negative.

The frontier exists, is exceptionless above `d_0`, has a proved constant `c = 1.25` from the
certified ladder and a measured constant `c = 3.25`, and it is an honest induction on the
machine (theorem (E) is unconditional; only `F` at smaller machines is used). But it cannot
reach the root, and the reason is now exact and is *not* the reason the brief guessed: the
big-gear fusions at the top of the window are zero, and the whole obstruction sits at column 1,
where the machine's own gears strike their own columns and the frontier's hypothesis fails by
construction. In that region the spectrum-plus-depth bound degenerates to an identity, and the
statement "the initial run stops before `W`" is `d_0 <= W`, the window statement itself.

The branch's contribution toward the root is a **reduction with a sign**: it removes the whole of
`[q/6, W]` from suspicion (every stretch there obeys `x >= 1.25 L` provably and `x >= 3.25 L`
measurably) and concentrates the root on a single run, the one at column 1, whose length is
`d_0`. That is node 1e.i's quantity, now shown to be not merely *a* hard case but, from
`q = 1427` on, **the only Pareto point of the entire prefix**.

# Dead ends

- **`R_min(L) >= cL` as a route to the window statement.** DEAD by construction, with the
  refuting instance being the definition: the law's exception set is `L < d_0` and the window
  statement is `d_0 <= W`. Any `c` works and none helps.
- **The brief's diagnosis "probably the big-gear fusions near the top of the window".** REFUTED:
  0 columns with no effective striker and 1 piece of the effective machine at all 23 frontier
  stretches above column 1, including `q = 997`'s 241-column window record, where the 11 big
  gears make 7 shared strikes.
- **`min_{L >= 6} R_min/L < 3`, attained at `L = 6` or 7** (P3). REFUTED: the minimum over the
  law's domain is 3.25 and it is attained at `L = 4`, 9 or 24.
- **Deep-period use of the effective machine.** Vacuous: at m29's record the effective machine is
  `{5..34,703}`, 1,197 times the machine. The frontier has content only for `x < (q^2-1)/6`, and
  its strength decays to the root's own statement at the top of that range.
- **The fusion structure at column 1 as a mechanism.** It is an identity: the run ends at the
  first effective-machine opening that is not a twin gear pair, which is `d_0` by definition.

> Kernel correction (2026-09-07, round 39, `proofs/OneStepE.lean`): theorem (E) as phrased here
> ("for every column with 6k - 1 > q, blocked under {5..q} iff blocked under {5..sqrt(6k + 1)}")
> is FALSE outside the prefix: the kernel holds the refuting instance q = 5, k = 8 (members 47 and
> 49 = 7^2; blocked under {5..7}, open under {5}; sqrt(49) = 7 > q). The true statement needs the
> column inside the next prime's square, 6k + 1 < q'^2, which is exactly where this document used
> it (`OneStepE.blocked_iff_sqrt`), or the weaker `blocked_iff_of_sqrt_le` with sqrt(6k + 1) <= r
> <= q. At one step the exact exception set is the home column d_0(M) iff (q', q' + 2) is a twin
> pair, and the square column W iff q'^2 - 2 is prime (`OneStepE.new_iff`), and every maximal run
> strictly inside [1, W] is inherited (`maxRun_succ`).

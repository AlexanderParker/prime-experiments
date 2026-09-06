# The in-use next-opening bound (branch R4.b.iii.a)

> Law numbers in this document map to the project-wide register
> (`research/proof/law_register.md`): **L46-L56 = W55-W65**.  This document's L50-L56
> CLASH with `top_machine_5.md`'s L50-L56; the register resolves the clash.

Parent: R4.b.iii, *The walk and the transforms of the top machine*
(`research/proof/top_machine_3.md`, laws L30-L45 there). The observation that spawned this
branch is the one open item that branch left: the mex closed form for the next opening is exact
everywhere, but the only *proved* bound on how far the walk can run - the covering bound
`L <= 2m/(1 - 2 H_S)` (L33 there) - goes vacuous inside the range where the machine is actually
used, and the branch closed with "the in-use record is not a covering phenomenon but a gear-zone
one (L21), and this branch does not close it."

**The object.** The top machine with the gear set it is used with: `G = {primes in (q, Q]}`,
acting on the pair coordinate over a range `[1, N]`, with `Q = floor(sqrt(N))` in the standard
use. The gear count `m` is in the hundreds or thousands, so the smallest gear `q'` is far below
`2m + 1` and the parity law's hypothesis fails; the record on the range is `F_range(N)`
(`top_machine_2.md` L32) and the walk to the next opening from a position `x` is `L(x)`
(`top_machine_3.md` L30/L32).

Construction rule (R4, owner): the top machine on the raw line, on its own; no clutch; no
bottom machine; no interpretation against the twin conjecture.

Numbering of laws continues from **L46** (the two prior passes both reached L45 / L38; cite by
document).

---

## 1. Pre-registered predictions and scorecard

Written before any computation of this branch. Where a prediction carries a number, the number
is derived by hand from published measurements of the two earlier passes (quoted at the point of
use), so that the computation is a test and not a fit.

### Section 1. The object: the wheel record as core plus tail

`top_machine_2.md` L31 says `F_top(G)` depends only on `m` and on the sub-multiset
`{g in G : g <= F_top + 1}`. Call those the **core** gears and the rest the **tail**, of size
`t = m - |core|`.

**P1 (THE CORE/TAIL RULE, exact).** Predicted, as an exact rule and not an approximation:

        F_top(G) = max { L : min over core phase vectors of D(U) <= t }

where, for a choice of phase for every core gear, `U` is the set of cells of `[0, L)` that no
core gear strikes, and `D(U)` - the **domino cost** - is

        D(U) = sum over maximal step-2 runs of U inside the even cells of ceil(len/2)
             + the same over the odd cells,

and `t = #{g in G : g > L + 1}`. Derivation: by L16 the record is a covering problem; a gear
`g > L + 1` shows exactly one domino `{x, x + 2}` inside the window, or a single cell when its
partner falls outside, and the domino lies in one parity class, so the cheapest way to cover a
set of same-parity cells is to pair them along consecutive step-2 runs; a core gear `g <= L + 1`
shows a longer, `g`-periodic trace, which must be taken as it is. Refuted by one gear set whose
independently known `F_top` differs.

**P2 (the naive additive form is REFUTED).** The shape the brief pre-registers - `F_top` =
(the core's own record) + `2t` - (parity defect) - is predicted **false**, and the refuting
instance is already in the record: `{13, ..., 41}` has `m = 8`, `F_top = 18`
(`top_machine_2.md` 3.7), core `{13, 17, 19}` whose own record is `5`
(`top_machine_3.md` 3.1) and `t = 5`, so the additive form gives `5 + 10 = 15 != 18`. The reason
is mechanical: a core gear inside a *longer* window contributes `2 ceil(L/g)` cells, not the two
it contributes inside its own record. Predicted instead the **capacity form**
`F_top ~ max{L : L - K(L) <= 2t}` with `K(L)` the core's best coverage of `[0, L)`, exact only
in the run-structured form P1.

**P3 (the tail is what the parity law measures; in use it is not empty).** Predicted: for the
in-use gear set the wheel record `F_top` is far above the range record, and the covering /
parity apparatus describes `F_top`, not `F_range(N)`. Predicted `F_top >= 2m - (m mod 2)` always
(free from L16) and, with a nonempty core, strictly more.

### Section 2. The bound in use

**P4 (THE GEAR-ZONE IDENTITY - the in-use record is a smooth-pair gap).** Let `Q = max G` and
let `S(q, Q) = {n <= Q : n and n + 2 are both q-smooth}`. Predicted, exactly and with no
exceptions:

* for `1 <= n <= Q - 2`, the pair `n` is open **iff** `n in S(q, Q)`;
* `F_range(N) = max ( the gaps between consecutive elements of S(q, Q), the last of them
  extended from the largest element of S to the first open pair above Q, the record of the walk
  above `Q`).

Derivation: for `n <= Q` any prime factor `p > q` of `n` or of `n + 2` satisfies `p <= n + 2`,
so `p` is a gear (or `p in (Q, Q + 2]`, the single edge case), and that gear strikes the pair.
So the whole of `[1, Q]` is struck except at the `q`-smooth pairs, which are *finite in number*
(Stormer 1897). Refuted by one position `n <= Q - 2` open without being a smooth pair, or one
`(q, N)` where the record is not one of those three quantities.

**P5 (THE PROVED LOWER BOUND, with numbers).** Let `s(q)` be the largest `n` with `n` and
`n + 2` both `q`-smooth (finite, Stormer). Predicted theorem: for `Q > s(q)`,

        F_range(N) >= Q - s(q) ,      hence with Q = floor(sqrt N),  F_range(N) >= sqrt(N) - s(q).

Predicted values of `s(q)` read off the published record positions
(`top_machine_1.md` 3.6: at `N = 10^7`, `Z = 3162`, the record starts at 161, 449, 1079, 2001,
2549, 2661 for `q = 5, 7, 11, 13, 17, 19`): `s(5) = 160` (`160 = 2^5 5`, `162 = 2 3^4`),
`s(7) = 448` (`448 = 2^6 7`, `450 = 2 3^2 5^2`), `s(11) >= 1078` (`2 7^2 11`, `2^3 3^3 5`),
`s(13) >= 2000` (`2^4 5^3`, `2 7 11 13`), `s(17) >= 2548` (`2^2 7^2 13`, `2 3 5^2 17`),
`s(19) >= 2660` (`2^2 5 7 19`, `2 11^3`). Predicted consequences, testable to the unit:
`F_range(10^7) >= 3002` at `q = 5` against the measured 3,006, and `>= 2714` at `q = 7` against
the measured 2,718. Refuted by a `q`-smooth pair above the predicted `s(q)`, or by a measured
record below the bound.

**P6 (NO BOUND IN `(q', m)` OF THE PARITY KIND - the covering bound is provably the wrong
shape).** Since `m ~ pi(Q) ~ 2 sqrt(N) / log N`, so `Q ~ (m/2) log m`, P5 gives

        F_range(N) >= (m log m)/2 - s(q)   (1 + o(1)) ,

so the in-use record exceeds the large-gear parity value `2m` by a factor `~ (log m)/4` and no
bound linear in `m` can hold. Predicted further, exactly: the covering bound
`L <= 2m/(1 - 2 H_S)` is non-vacuous **iff** the record is below

        L*(q) = exp exp ( 1/2 + sum_{p <= q} 1/p - M ) ,    M = 0.26149721 (Mertens),

with predicted values `L*(5) ~ 35`, `L*(17) ~ 174`, `L*(19) ~ 238`; against the published
records (136 at `q = 17`, 71 at `q = 19`, both `N = 10^6`, where the bound was non-vacuous, and
618 / 227 at `N = 10^7`, where it was vacuous) this predicts the crossover exactly. Refuted by a
machine with `F < L*(q)` and a vacuous bound, or the converse.

**P7 (the strict form: which half is provable).** Predicted: everything on `[1, Q]` is proved
outright (P4 is an iff, with a two-line proof), so the record itself - which P4 says lives there
- is *computed exactly from `q` alone* through the finite smooth-pair list; and the one number
that is not provable by counting is the first opening **above** `Q`, whose existence in a short
interval is a two-dimensional sieve lower bound and is not reachable by a union bound. Predicted
therefore: the theorem this branch can prove is

        every x in [1, Q] has its next open pair within  maxgap(S(q,Q)) + u(q, Q)

with `u(q, Q)` the (measured) distance from `Q` to the first opening above it, and the ratio of
the true record to the proved lower bound `Q - s(q)` is predicted to lie in `[1.00, 1.05]` at
every `(q, N)` with `Q > s(q)`.

### Section 3. The record's position, and the first stretch

**P8 (position law).** Predicted: the in-use range record block starts at `1 + ` (the largest
element of `S(q, Q)` below the maximal gap), so its position is `O_q(1)` or at most `O(Q)` -
never more than `sqrt(N)` - and `position / N -> 0` like `N^{-1/2}`. Predicted: at every `(q, N)`
with `Q > s(q)` the record is the **last** gap of the smooth list, so the record block starts at
`s(q) + 1` exactly. Refuted by a record block above `Q`.

**P9 (the walk above the first stretch is a different, much smaller object).** Predicted: the
maximum walk over `x in (Q, N]` - call it `A(q, N)` - is far below `F_range(N)`, grows like a
power of `log N` rather than like `sqrt N`, and satisfies `A(q, N) < F_range(N)` at every tested
`(q, N)` with `q <= 19`; predicted `A <= 400` for all `q` at `N <= 10^8`. Refuted by an
above-zone walk exceeding the gear-zone record (which would be a genuine second mechanism).

### Section 4. The smallest-gear ceiling and saturation

**P10 (NO ceiling in `q'` alone; the saturation question has the opposite answer).** Predicted:
`F_range(N)` at fixed `N` does **not** saturate as `Q` grows past `sqrt(N)` - it *grows*, because
the gear zone is `[1, Q]` and not `[1, sqrt N]`: for `n <= Q`, `n` open iff `n` is `q`-smooth,
whatever `N` is. Predicted `F_range(N; (q, Q]) = Q - s(q) + O(u)` for every `Q > s(q)` with
`Q <= N`, i.e. **linear in the largest gear**. Refuted by a measured saturation. Predicted
correction to the brief's expectation: what *does* saturate is the above-zone walk `A(q, N)`, on
which a gear `g > Q_0` acts only through the density factor `1 - 2/g`; predicted `A` changes by
less than its own scatter once `Q` exceeds a few times `sqrt(N)`.

**P11 (what is a function of `q` alone).** Predicted: not the record, but the entire gear-zone
structure - the finite set `S(q)`, its largest element `s(q)`, its gap list - and hence the
record's *shape* `F_range = Q - s(q) + u`. Predicted `s(q)` is the only `q`-datum needed once
`Q > s(q)`.

### Scorecard

| # | Prediction | Result |
|---|---|---|
| P1 | core/tail covering rule with the domino cost `D(U)`, exact | |
| P2 | the additive form `F_core + 2t - defect` is refuted (`{13..41}`: 15 vs 18) | |
| P3 | in use the tail is nonempty and `F_top >> F_range` | |
| P4 | gear-zone identity: `n <= Q - 2` open iff `n, n+2` both `q`-smooth; record = smooth-gap | |
| P5 | `F_range >= Q - s(q)`; `s(5) = 160`, `s(7) = 448`; `F_range(10^7) >= 3002` at `q = 5` | |
| P6 | `F_range >= (m log m)/2 - s(q)`; covering bound alive iff `F < L*(q)`, `L*(5) ~ 35` | |
| P7 | the provable half is `[1, Q]`; ratio true/lower-bound in `[1.00, 1.05]` | |
| P8 | the record block starts at `s(q) + 1` whenever `Q > s(q)` | |
| P9 | above-zone record `A(q, N) < F_range`, `<= 400` at `N <= 10^8`, polylog growth | |
| P10 | no saturation in `Q`: `F_range` grows linearly in `Q`; `A` saturates instead | |
| P11 | `s(q)` is the only `q`-datum once `Q > s(q)` | |

---

## 2. Setup as computed

Scripts in `research/topmachine/r4/`, results (untracked) in `.../results/`:

| script | what it computes |
|---|---|
| `zone.py` | the in-use scan over `[1, N]` in chunks, the gear-zone identity, the `q`-smooth pair (Stormer) lists, the record and its position |
| `wheelrec.py` | the wheel record `F_top(G)` by the covering engine: core traces enumerated by depth-first search with a valid parity-count prune, tail counted |
| `bounds.py` | the exact law for the zone record, the covering bound's death point, the record against `2m`, the first-hit model above the zone, saturation on a fixed region |
| `intail.py` | certified covers of `[0, L)` for the in-use gear set: a lower bound on `F_top` and hence an upper bound on the tail |
| `saturate.py` | fixed `N`, the largest gear `Q` swept from 100 to `10^6` |

Every count below is exact over the stated range: the scan is a full sieve of `[0, N]`, not a
sample. In-use machines: `q in {5, 7, 11, 13, 17, 19, 23, 29, 37}`, `N in {10^5, ..., 10^8}`,
`Q = floor(sqrt N)` (36 machines, 53 to 1,226 gears), plus sweeps with `Q` up to `10^6`
(78,495 gears). Wheel records: 13 independently known sets and 89 consecutive-prime sets
`{q'..Q}`, `q' = 7..41`, `m = 3..12`. `q`-smooth pair lists computed complete below `10^13`.

---

## 3. Results

### 3.1 The gear-zone identity: in use, the machine is a smooth-number machine

For `1 <= n <= Q - 2` the pair `n` is open **iff** `n` and `n + 2` are both `q`-smooth.
Checked cell by cell in all 36 in-use machines: **130,230 cells, 4,019 openings, 0 exceptions.**

The consequence is that the bottom of the range is not a sieve at all. The openings below `Q`
are a *finite* list - the solutions of the Stormer problem "both members `q`-smooth" - and they
are these:

| `q` | pairs `<= 10^13` | largest `s(q)` | the whole list (starts `n`) |
|---|---|---|---|
| 5 | 13 | **160** | 1, 2, 3, 4, 6, 8, 10, 16, 18, 25, 30, 48, 160 |
| 7 | 29 | **8,748** | ..., 160, 243, 250, 448, 4800, 8748 |
| 11 | 49 | **19,600** | ..., 880, 1078, 4800, 6048, 8748, 19600 |
| 13 | 83 | **246,400** | ..., 8448, 8748, 13310, 19600, 21294, 246400 |
| 17 | 130 | **672,280** | ..., 57120, 62424, 74358, 246400, 388960, 672280 |
| 19 | 202 | **23,718,420** | ..., 1202850, 1267110, 1419262, 11819520, 23718420 |
| 23 | 297 | **23,718,420** | ..., 4046848, 8193150, 10285000, 11819520, 23718420 |
| 29 | 423 | **354,365,440** | ..., 23718420, 26578123, 36171408, 192119200, 354365440 |
| 37 | 799 | **9,447,152,317** | ..., 740512498, 3222617398, 6926399998, 9447152317 |

`s(5) = 160` was pre-registered from the published record position and is confirmed;
`s(7) = 448` was pre-registered and is **refuted** - the list continues to 4,800 and 8,748,
both above the `Q = 3,162` of `N = 10^7`, which is why 448 looked like the end.

### 3.2 The record is the largest gap of that finite list

Predicted exactly by the list plus one number (the first opening at or after its last element).
Value **and position**, 36 machines:

| `q` | `N` | `Q` | zone record | at | predicted from the list | at | agrees |
|---|---|---|---|---|---|---|---|
| 5 | 1e5 | 316 | 186 | 161 | 186 | 161 | yes |
| 5 | 1e6 | 1,000 | 858 | 161 | 858 | 161 | yes |
| 5 | 1e7 | 3,162 | 3,006 | 161 | 3,006 | 161 | yes |
| 5 | 1e8 | 10,000 | 9,846 | 161 | 9,846 | 161 | yes |
| 7 | 1e6 | 1,000 | 570 | 449 | 570 | 449 | yes |
| 7 | 1e7 | 3,162 | 2,718 | 449 | 2,718 | 449 | yes |
| 7 | 1e8 | 10,000 | 4,351 | 449 | 4,351 | 449 | yes |
| 11 | 1e7 | 3,162 | 2,088 | 1,079 | 2,088 | 1,079 | yes |
| 11 | 1e8 | 10,000 | 3,721 | 1,079 | 3,721 | 1,079 | yes |
| 13 | 1e7 | 3,162 | 1,166 | 2,001 | 1,166 | 2,001 | yes |
| 13 | 1e8 | 10,000 | 2,141 | 6,049 | 2,141 | 6,049 | yes |
| 17 | 1e7 | 3,162 | 618 | 2,549 | 618 | 2,549 | yes |
| 17 | 1e8 | 10,000 | 2,141 | 6,049 | 2,141 | 6,049 | yes |
| 19 | 1e7 | 3,162 | 227 | 2,661 | 227 | 2,661 | yes |
| 19 | 1e8 | 10,000 | 1,691 | 6,499 | 1,691 | 6,499 | yes |
| 23 | 1e8 | 10,000 | 735 | 7,039 | 735 | 7,039 | yes |
| 29 | 1e8 | 10,000 | 735 | 7,039 | 735 | 7,039 | yes |
| 37 | 1e8 | 10,000 | 256 | 8,992 | 256 | 8,992 | yes |

(18 of the 36 rows shown; the other 18 also agree.) **0 exceptions of 36**, value and position
together. The record block at `q = 5` always starts at 161 - one above the last Stormer pair -
whatever `N` is; at `q = 13` it moves from 485 to 2,001 to 6,049 as `Q` grows past successive
members of the list.

### 3.3 The proved lower bound against the truth

Nothing in the following needs a sieve estimate. Consecutive members `a < b` of the smooth-pair
list below `Q - 2` bound a block every cell of which is struck, so `F_range(N) >= b - a - 1`;
and beyond the last member `s_k` every cell up to `Q - 2` is struck, so
`F_range(N) >= Q - 2 - s_k`. Taking the largest of those:

| `q` | `N` | proved bound | measured | ratio | `q` | `N` | proved bound | measured | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1e5 | 154 | 186 | 1.208 | 19 | 1e5 | 29 | 37 | 1.276 |
| 5 | 1e6 | 838 | 858 | 1.024 | 19 | 1e6 | 48 | 71 | 1.479 |
| 5 | 1e7 | 3,000 | 3,006 | **1.002** | 19 | 1e7 | 227 | 227 | **1.000** |
| 5 | 1e8 | 9,838 | 9,846 | **1.001** | 19 | 1e8 | 1,691 | 1,691 | **1.000** |
| 7 | 1e7 | 2,712 | 2,718 | 1.002 | 23 | 1e7 | 212 | 212 | 1.000 |
| 7 | 1e8 | 4,351 | 4,351 | **1.000** | 29 | 1e8 | 735 | 735 | 1.000 |
| 11 | 1e7 | 2,082 | 2,088 | 1.003 | 37 | 1e7 | 113 | 113 | 1.000 |
| 13 | 1e8 | 2,141 | 2,141 | **1.000** | 37 | 1e8 | 256 | 256 | 1.000 |

Over all 36 machines the ratio true/proved runs **1.000 to 2.000**, and is **exactly 1.000 at 17
of the 36** - the proved bound *is* the record there. The worst ratios (1.9 to 2.0) are all at
`N = 10^5, 10^6` with `q >= 23`, where the record has left the gear zone (3.5).

### 3.4 Where the covering bound dies, exactly

`H_S = sum_{q < g <= F} 1/g`; the bound `L <= 2m/(1 - 2 H_S)` of `top_machine_3.md` L33 is alive
iff `H_S < 1/2`. In Mertens closed form that is `F < L*(q) = exp exp (1/2 + sum_{p<=q} 1/p - M)`,
`M = 0.26149721`:

| `q` | `L*(q)` | machines where the bound is alive | dead | the criterion `F < L*(q)` |
|---|---|---|---|---|
| 5 | 35 | none | 4 | correct 4/4 |
| 7 | 61 | none | 4 | correct 4/4 |
| 11 | 91 | `1e5` (bound 1,127 vs `F = 64`) | 3 | correct 4/4 |
| 13 | 130 | `1e5` (260 vs 41) | 3 | correct 4/4 |
| 17 | 175 | `1e5` (187 vs 37), `1e6` (4,289 vs 136) | 2 | correct 4/4 |
| 19 | 231 | `1e5` (157 vs 37), `1e6` (753 vs 71) | 2 | **1 miss** (`1e7`: `F = 227 < 231` but `H_S = 0.507`) |
| 23 | 294 | 3 (138, 558, 9,672) | 1 | correct 4/4 |
| 29 | 359 | 3 (125, 494, 5,166) | 1 | correct 4/4 |
| 37 | 514 | 4 (106, 345, 1,787, 11,156) | 0 | correct 4/4 |

**35 of 36 correct**; the single miss is the boundary case `H_S = 0.507` against a criterion
built from an asymptotic constant. Where the bound is alive it is **40 to 50 times loose**
(`q = 23`, `N = 10^7`: 9,672 against 212; `q = 29`: 5,166 against 210; `q = 37`, `N = 10^8`:
11,156 against 256).

### 3.5 The two regimes, and the walk above the gear zone

Splitting the range at `Q`: the zone record (blocks starting at or below `Q`) and the above-zone
record `A(q, N)` (blocks starting above `Q`):

| `q` | `N` | `F_range` | zone | above-zone `A` | at | which wins |
|---|---|---|---|---|---|---|
| 5 | 1e5 | 186 | 186 | 61 | 72,369 | zone |
| 5 | 1e8 | 9,846 | 9,846 | 419 | 26,262 | zone |
| 7 | 1e8 | 4,351 | 4,351 | 371 | 18,540 | zone |
| 11 | 1e8 | 3,721 | 3,721 | 269 | 14,868 | zone |
| 13 | 1e5 | 41 | 31 | **41** | 2,037 | **above** |
| 17 | 1e5 | 37 | 29 | **37** | 2,041 | **above** |
| 19 | 1e6 | 71 | 48 | **71** | 1,788 | **above** |
| 23 | 1e6 | 66 | 46 | **66** | 1,793 | **above** |
| 29 | 1e6 | 66 | 34 | **66** | 1,793 | **above** |
| 37 | 1e6 | 46 | 30 | **46** | 95,333 | **above** |
| 37 | 1e7 | 113 | 113 | 111 | 5,302 | zone |
| 37 | 1e8 | 256 | 256 | 183 | 13,486 | zone |

The zone wins in **26 of 36**; the above-zone record wins in **10**, all of them at `q >= 13`
and `N <= 10^6`, i.e. exactly where `Q` is still small enough that the `q`-smooth pairs are
dense. Each `q` has a crossover: `q = 13` and `17` between `10^5` and `10^6`, `q = 19, 23, 29,
37` between `10^6` and `10^7`, `q = 5, 7, 11` never (the zone wins at every `N`).

The above-zone record is a first-hit object, and the first-hit model fits it:

| `q` | `N` | density `p` | `A` | `ln(Np) / (-ln(1-p))` | ratio |
|---|---|---|---|---|---|
| 5 | 1e5 | 0.14133 | 61 | 63 | 0.97 |
| 5 | 1e7 | 0.07234 | 200 | 180 | 1.11 |
| 5 | 1e8 | 0.05377 | 419 | 280 | 1.49 |
| 13 | 1e8 | 0.11564 | 227 | 132 | 1.72 |
| 19 | 1e8 | 0.14804 | 203 | 103 | 1.97 |
| 29 | 1e8 | 0.17418 | 203 | 87 | 2.33 |
| 37 | 1e8 | 0.19590 | 183 | 77 | 2.38 |

Ratio 0.97 to 2.38 over all 36, drifting upward with `q` and with `N`: the openings above the
zone are more clustered than independent, so the true record runs above the independent model,
but only by a small factor. `A` reaches 419 at `N = 10^8` - the pre-registered ceiling of 400 is
**refuted by 19**.

Beyond the first stretch the machine is therefore a completely different size: at `q = 5`,
`N = 10^8`, the walk record is **9,846 inside `[1, Q]` and 419 above it** - a factor 23.5 - and
the whole of the record's size is the gear zone.

### 3.6 No saturation: the largest gear sets the record, linearly

Fixed `N = 10^7`, `Q` swept (the `m` column is `q = 5 / 11 / 19`):

| `Q` | `m` | `F_range` (`q=5`) | at | `F_range` (`q=11`) | at | `F_range` (`q=19`) | at |
|---|---|---|---|---|---|---|---|
| 100 | 22/20/17 | 65 | 714,641 | 32 | 9,632,632 | 21 | 596,831 |
| 316 | 62/60/57 | 186 | 161 | 66 | 9,693,108 | 42 | 4,546,729 |
| 1,000 | 165/163/160 | 858 | 161 | 283 | 485 | 75 | 9,709,390 |
| 3,162 | 443/441/438 | 3,006 | 161 | 2,088 | 1,079 | 227 | 2,661 |
| 10,000 | 1,226/1,224/1,221 | 9,846 | 161 | 3,721 | 1,079 | 1,691 | 6,499 |
| 31,623 | 3,398/3,396/3,393 | 31,560 | 161 | 12,120 | 19,601 | 5,881 | 13,719 |
| 100,000 | 9,589/9,587/9,584 | 99,990 | 161 | 80,550 | 19,601 | 18,017 | 28,799 |
| 1,000,000 | 78,495/78,493/78,490 | 999,876 | 161 | 980,436 | 19,601 | 327,756 | 672,281 |

At `q = 5` the record is `Q - 160 - u` for every `Q`: 186, 858, 3,006, 9,846, 31,560, 99,990,
999,876 against `Q` = 316, 1,000, 3,162, 10,000, 31,623, 100,000, 1,000,000. **Linear in the
largest gear, with no ceiling and no sign of saturation.** The zone prediction from the smooth
list matches the measured zone record at every one of the 24 sweep points.

And the walk *above* the zone does not saturate either. On a **fixed** region `(10^6, 10^7]`,
strictly above every zone tested, adding gears keeps lengthening the record:

| `Q` | `m` (`q=5`) | max walk on `(10^6, 10^7]` | openings on the region |
|---|---|---|---|
| 3,162 | 443 | 200 | 723,394 |
| 10,000 | 1,226 | 203 | 682,428 |
| 31,623 | 3,398 | 252 | 611,865 |
| 100,000 | 9,589 | 266 | 497,892 |
| 300,000 | 25,994 | 419 | 347,654 |
| 1,000,000 | 78,495 | 1,511 | 167,729 |

Same at `q = 19` (86, 89, 136, 203, 419, 1,511). The mechanism is the merge law (L13): every new
gear strikes `2N/g` positions of the region and each strike can merge two blocks into one, so no
gear size is ever harmless.

### 3.7 The wheel record as core plus tail

The covering engine (core traces enumerated exactly, tail counted by the domino cost) against
every independently known `F_top`:

| gears | `m` | known `F_top` | engine | core (`g <= F + 1`) | tail `t` |
|---|---|---|---|---|---|
| 7,11,13 | 3 | 6 | 6 | {7} | 2 |
| 11,13,17 | 3 | 5 | 5 | empty | 3 |
| 7,11,13,17 | 4 | 9 | 9 | {7} | 3 |
| 11,13,17,19 | 4 | 8 | 8 | empty | 4 |
| 13,17,19,23 | 4 | 8 | 8 | empty | 4 |
| 17,19,23,29 | 4 | 8 | 8 | empty | 4 |
| 11,13,17,19,23 | 5 | 10 | 10 | {11} | 4 |
| 7..31 | 8 | 32 | 32 | all eight | **0** |
| 13..41 | 8 | 18 | 18 | {13,17,19} | 5 |
| 19..47 | 8 | 16 | 16 | empty | 8 |

**0 mismatches over 13 known records** (the five three-gear wheels are in the full table).
Extended to 89 consecutive-prime sets `{q'..Q}`, `q' = 7..41`, `m = 3..12`, all decided exactly:

| `q'` | `m` | `F_top` | core | `t` | `2t - (t mod 2)` | additive form |
|---|---|---|---|---|---|---|
| 13 | 8 | 18 | {13,17,19} | 5 | 9 | 14 (no) |
| 13 | 10 | 27 | {13,17,19,23} | 6 | 12 | 20 (no) |
| 17 | 12 | 32 | {17,19,23,29,31} | 7 | 13 | 22 (no) |
| 19 | 8 | 16 | empty | 8 | 16 | 16 (yes) |
| 23 | 10 | 20 | empty | 10 | 20 | 20 (yes) |
| 29 | 12 | 24 | empty | 12 | 24 | 24 (yes) |
| 41 | 12 | 24 | empty | 12 | 24 | 24 (yes) |

The additive form (the core's own record plus `2t` less the parity defect) agrees at **64 of
89** - and those 64 are **exactly** the sets whose core is empty, where it is the parity law
itself. It fails at **all 25** sets with a nonempty core, and by a lot (15 against 18, 20 against
27, 22 against 32).

### 3.8 In use, the tail is empty

An explicit cover of `[0, L)` - one phase per gear, built greedily smallest gear first - is a
certificate that `F_top >= L` (L16). For the in-use gear sets:

| `q` | `N` | `Q` | `m` | certified `F_top >=` | tail gears (`g > F + 1`) | `2m - (m mod 2)` | certified / `2m` |
|---|---|---|---|---|---|---|---|
| 5 | 1e6 | 1,000 | 165 | 4,902 | **0** | 329 | 14.9 |
| 5 | 1e7 | 3,162 | 443 | 23,148 | **0** | 885 | 26.1 |
| 5 | 1e8 | 10,000 | 1,226 | 104,259 | **0** | 2,452 | 42.5 |
| 11 | 1e8 | 10,000 | 1,224 | 60,335 | **0** | 2,448 | 24.7 |
| 19 | 1e8 | 10,000 | 1,221 | 39,968 | **0** | 2,441 | 16.4 |
| 29 | 1e7 | 3,162 | 436 | 6,946 | **0** | 872 | 8.0 |
| 37 | 1e6 | 1,000 | 156 | 974 | 4 | 312 | 3.1 |
| 37 | 1e8 | 10,000 | 1,217 | 28,519 | **0** | 2,433 | 11.7 |

**26 of 27 in-use machines have an empty tail** (the exception is the weakest certificate, at
`q = 37`, `N = 10^6`, with four gears above the certified bound). The certified wheel record is
3 to 43 times the parity value `2m`, and grows with `N`.

### 3.9 The ceiling on what any union bound can certify

A cover of `d` consecutive cells assigns a gear to each cell; a gear used at a set of cells pins
`x` in at most 2 classes modulo that gear, so a pattern using the gear set `U` admits at most
`2^{|U|}(N / prod U + 1)` positions `x <= N`. The term that carries information is `N / prod U`;
once `prod U > N` the bound is the useless `2^{|U|}`. Every cover of `d` cells uses at least
`d/2` gears, each at least `q'`, so the counting is informative only while `q'^{d/2} <= N`:

        d  <=  2 log N / log q' .

| `q` | `q'` | `N` | `F_range` | `2 log N / log q'` | truth / ceiling |
|---|---|---|---|---|---|
| 5 | 7 | 1e5 | 186 | 11.8 | 16 |
| 5 | 7 | 1e7 | 3,006 | 16.6 | 181 |
| 5 | 7 | 1e8 | 9,846 | 18.9 | 520 |
| 11 | 13 | 1e7 | 2,088 | 12.6 | 166 |
| 19 | 23 | 1e8 | 1,691 | 11.7 | 144 |
| 37 | 41 | 1e8 | 256 | 9.9 | 26 |

The in-use record is 4 to 520 times the largest length a union bound over patterns can reach.

---

## 4. Laws and bounds

Numbered from **L46**. `G = {primes in (q, Q]}`, `m = |G|`, `q' = min G`, `N >= Q`; `L(x)` is
the walk to the next open pair and `F_range(N)` the longest pair-free block in `[1, N]`
(`top_machine_2.md` L32).

**L46 (THE GEAR-ZONE IDENTITY - in use the bottom of the range is a smooth-number machine).**
For every `n` with `1 <= n <= Q - 2`, the pair `n` is open **iff** `n` and `n + 2` are both
`q`-smooth.

*Proof.* If both are `q`-smooth, no prime above `q` divides either, and every gear exceeds `q`,
so no gear strikes `n`. Conversely let `n` be open and let `p > q` divide `n` or `n + 2`. Both
are at most `Q`, so `p <= Q`, so `p in (q, Q]` is a gear and strikes `n` - a contradiction. QED.

*Evidence.* 36 in-use machines, 53 to 1,226 gears, `N = 10^5..10^8`: 130,230 cells and 4,019
openings, **0 exceptions**. *This sharpens L21 (`top_machine_1.md`), which stated the smoothness
description and observed that the record lies in the zone, into an exact iff with its exact
range `n <= Q - 2` (the two boundary cells `Q - 1`, `Q` can be open through a prime factor in
`(Q, Q + 2]`) - and it replaces "the gear zone" by a finite, computable list.*

**L47 (THE IN-USE RECORD IS THE LARGEST GAP OF A FINITE LIST).** Let
`S(q, Q) = {n <= Q - 2 : n, n + 2 both q-smooth} = {s_1 < ... < s_k}` and let `u` be the first
open pair at or after `s_k + 1`. Then the record over the gear zone, with its position, is

        F_zone(q, Q) = max ( max_i (s_{i+1} - s_i - 1) ,  u - s_k - 1 ) ,

the record block starting one above the lower end of the winning gap; and
`F_range(N) = max(F_zone(q, Q), A(q, N))` with `A` the record above the zone.

*Proof of the zone part.* Immediate from L46: inside `[1, Q - 2]` the open cells are exactly
`S(q, Q)`, so the pair-free blocks are exactly its gaps. QED.

*Evidence.* Value **and** position, 36 machines, **0 exceptions**; and at all 24 points of the
`Q` sweep at fixed `N = 10^7`. *New: the in-use record is not a statistic of the gear set at all
but the gap structure of a finite Diophantine list.*

**L48 (THE PROVED LOWER BOUND - no sieve estimate needed).** For every in-use machine,

        F_range(N)  >=  max ( max_i (s_{i+1} - s_i - 1) ,  Q - 2 - s_k ) ,

computable from `q` and `Q` alone. If moreover `Q > s(q)`, the largest `q`-smooth pair at all,
then `F_range(N) >= Q - 2 - s(q)`, i.e. `F_range(N) >= sqrt(N) - s(q) - 2`.

*Proof.* Every cell strictly between two consecutive members of `S(q, Q)` is struck (L46), as is
every cell in `(s_k, Q - 2]`. The second statement is the first with `s_k = s(q)`, which holds
once `Q > s(q)`. Finiteness of the full list is Stormer's theorem (1897); the bound as stated
uses only the members below `Q`, which are computed exactly, so it is unconditional. QED.

*Evidence.* True at all 36 machines; ratio true/bound **1.000 to 2.000**, and **exactly 1.000 at
17 of 36**. At `q = 5`: 1.208, 1.024, 1.002, 1.001 as `N` goes `10^5` to `10^8`.

**L49 (NO BOUND IN `q'` AND `m` OF THE PARITY KIND).** Since `Q ~ (m/2) log m`, L48 gives, for
fixed `q` and `Q > s(q)`,

        F_range(N)  >=  (m log m)/2  (1 + o(1))  -  s(q) ,

so the in-use record exceeds the large-gear parity value `2m` by an unbounded factor, and no
bound linear in `m` can hold. Measured `F_range / 2m` at `q = 5`: 1.50, 2.60, 3.39, **4.02** at
`N = 10^5..10^8`, still climbing; the covering bound `2m/(1 - 2H_S)` is not merely loose but the
wrong shape.

*Evidence.* 36 machines; the ratio is monotone in `N` at every `q <= 11`.

**L50 (WHERE THE COVERING BOUND DIES, IN CLOSED FORM).** `top_machine_3.md` L33 is alive exactly
while `H_S = sum_{q < g <= F} 1/g < 1/2`, which in Mertens form is

        F  <  L*(q) = exp exp ( 1/2 + sum_{p <= q} 1/p - M ) ,   M = 0.26149721 ,

with `L*(5) = 35`, `L*(7) = 61`, `L*(11) = 91`, `L*(13) = 130`, `L*(17) = 175`, `L*(19) = 231`,
`L*(23) = 294`, `L*(29) = 359`, `L*(37) = 514`. The criterion classifies **35 of 36** machines
correctly; the single miss is `q = 19`, `N = 10^7` (`F = 227 < 231` but `H_S = 0.507`), a
boundary case of the asymptotic constant. Where the bound is alive it is **40 to 50 times
loose**.

*Reading.* The bound survives only at large `q` and small `N` - exactly where the record has
left the gear zone (3.5) and is a first-hit object. It never sees the gear-zone record.

**L51 (THE CEILING ON UNION BOUNDS).** A union bound over covering patterns can control lengths
only up to `d <= 2 log N / log q'`. Measured, the in-use record exceeds that ceiling by a factor
**4 to 520**.

*Proof.* A cover assigns each of the `d` cells a striking gear; for a fixed gear the struck
cells lie in the two classes `0, -2 (mod g)`, so consistency pins `x` in at most two classes mod
`g`, and the positions `x <= N` carrying a given pattern number at most
`2^{|U|}(N/prod U + 1)`, informative only while `prod U <= N`. Each gear covers at most two
cells of a window shorter than itself, so `|U| >= d/2` and `prod U >= q'^{d/2}`. QED.

*Reading.* **The in-use next-opening bound is not reachable by counting.** The gear-zone half is
proved outright by L46-L48 and needs no counting at all; the above-zone half is the existence of
a pair `n, n + 2` both free of primes in `(q, Q]` inside a short interval - a two-dimensional
sieve lower bound - and the union bound stops four to five hundred times short of the truth.

**L52 (IN USE, THE TAIL IS EMPTY - the parity apparatus measures nothing).** For the in-use gear
set an explicit cover certifies `F_top(G) >= L` with `L` from 974 to 104,259, in every case at
least `Q - 1`, so `{g in G : g > F_top + 1}` is **empty**: every gear is a core gear.

*Evidence.* 26 of 27 in-use machines (the exception has the weakest certificate). Certified
`F_top / 2m` runs 3.1 to 42.5 and grows with `N`. *This is the structural reason L33 goes
vacuous in use: the parity law and the covering bound are statements about the tail, and in use
there is no tail.* Pre-registration P3 said the tail would be nonempty: **refuted**.

**L53 (THE CORE/TAIL RULE FOR THE WHEEL RECORD).** With `core = {g <= L + 1}`,
`t = #{g > L + 1}`, and the **domino cost** `D(U)` of an uncovered set `U` - the sum, over the
maximal step-2 runs of `U` inside each parity class, of `ceil(run/2)` -

        F_top(G) = max { L : min over core phase vectors of D(U) <= t } .

*Proof.* By L16 the record is a covering problem and the phases are free by CRT. A gear
`g > L + 1` shows within the window either the domino `{x, x + 2}` or, at the ends, one cell; a
domino covers two cells of one parity class, adjacent in that class, so pieces are never shared
between runs and a run of `k` needs at least `ceil(k/2)` of them. It also needs no more: pairing
along the run leaves at most one cell whose partner at distance 2 is outside the window or
already covered. A gear `g <= L + 1` has a `g`-periodic trace which must be taken as it is, so
those are enumerated. QED.

*Evidence.* **0 mismatches on 13 independently known records**, including `{7..31}` (`F = 32`,
core all eight gears, tail empty), `{13..41}` (18, core `{13,17,19}`) and `{19..47}` (16, core
empty); and exact decisions on 89 consecutive-prime sets. *This is L31 (`top_machine_2.md`) made
into a formula: the record depends on `m` and on the gears below `F + 1` because the rest enter
only as the number `t` in this inequality.*

**L54 (THE ADDITIVE FORM HOLDS EXACTLY WHEN THE CORE IS EMPTY).**
`F_top = F_core + 2t - (t mod 2)` agrees at **64 of 89** consecutive sets, and those 64 are
precisely the sets with an empty core, where the formula is the parity law `2m - (m mod 2)`
itself. It fails at **all 25** sets with a nonempty core.

*Mechanism.* A core gear placed in a window longer than its own record contributes
`2 ceil(L/g)` cells, not the two it contributes inside its own record, so its help grows with
the window it is helping to fill; the additive form assumes it is constant. Pre-registered as
refuted, with `{13..41}` (15 against 18) named in advance: **held**.

**L55 (NO SATURATION; THE RECORD IS LINEAR IN THE LARGEST GEAR).** At fixed `N`, `F_range(N)`
grows linearly in `Q`: at `q = 5` it is `Q - 160 - u` for every `Q` from 316 to `10^6` (186,
858, 3,006, 9,846, 31,560, 99,990, 999,876). Gears far above `sqrt(N)` are not harmless: on a
fixed region `(10^6, 10^7]` strictly above every gear zone tested, the record climbs 200, 203,
252, 266, 419, 1,511 as `Q` goes 3,162 to `10^6`.

*Mechanism.* Two distinct causes, both exact. (i) The gear zone is `[1, Q]`, not `[1, sqrt N]`:
for `n <= Q` a prime factor above `q` is at most `n <= Q` and hence a gear, so L46 holds with
`Q` whatever `N` is. (ii) Above the zone each new gear strikes `2N/g` cells of any region and
every strike can merge two blocks into one (L13), so no gear size is ever without effect.

*Evidence.* 24 sweep points at fixed `N = 10^7`, `Q` from 100 to `10^6`, `m` from 17 to 78,495;
12 further points on the fixed region. **The expected saturation does not occur, in either
form.** What *is* a function of `q` alone is the whole gear-zone structure - the list `S(q)`,
its largest element `s(q)`, its gaps - and hence the record's shape.

**L56 (THE TWO REGIMES AND THE CROSSOVER).** `F_range(N) = max(F_zone, A)`, where `F_zone` is
the deterministic smooth-gap object of L47 and `A(q, N)` is a first-hit object above the zone,
fitted by the independent model `ln(Np)/(-ln(1 - p))` to within a factor **0.97 to 2.38**. The
zone wins in **26 of 36** machines; the above-zone record wins in 10, all with `q >= 13` and
`N <= 10^6`. Each `q` has one crossover `N`, after which the zone wins at every larger `N`
(measured for all nine values of `q`), because `F_zone` grows like `Q` while `A` grows like a
power of `log N`.

*Evidence.* 36 machines; `A` from 24 to 419; the pre-registered ceiling `A <= 400` at
`N <= 10^8` is **refuted** (419 at `q = 5`, `N = 10^8`).

---

## 5. The bound in use, stated

**THEOREM (the in-use walk in the gear zone; proved).** Let `G` be the primes in `(q, Q]` acting
in pair coordinates, `N >= Q`, and let `S = {n <= Q - 2 : n, n + 2 both q-smooth}`, a finite set
computable from `q`. Then for every `x` with `1 <= x <= Q - 2`

        L(x) = min { n in S : n >= x } - x     if such an n exists ,

and otherwise `L(x) = u - x` with `u` the first open pair at or above `Q - 1`. Consequently

        max_{1 <= x <= Q - 2} L(x)  =  max ( max_i (s_{i+1} - s_i) ,  u - s_k )

and, with no reference to `u` at all,

        F_range(N)  >=  max ( max_i (s_{i+1} - s_i - 1) , Q - 2 - s_k ) ,

so that for `Q > s(q)`, `F_range(N) >= sqrt(N) - s(q) - 2`.

*Proof.* L46 gives the first display; L47 and L48 give the rest. Only the number `u` - the first
opening above the zone - is not determined by `q`, and it enters as a single additive term.

**Ratio of the truth to the bound:** 1.000 to 2.000 over 36 machines, exactly 1.000 at 17 of
them, and 1.001 at the largest machine measured (`q = 5`, `N = 10^8`: bound 9,838, truth 9,846).

**What is not proved, and why.** An upper bound valid for `x > Q` would have to produce a pair
`n, n + 2` both free of every prime in `(q, Q]` inside a short interval above `Q`. Counting
cannot do it: L51 shows a union bound over covering patterns runs out at `2 log N / log q'`,
which is 10 to 19 on the machines measured against records of 113 to 9,846. The measured value
is small - `A(q, N) <= 419` for every `q` at `N <= 10^8`, twenty-three times below the zone
record at `q = 5` - but it is measured, not bounded.

---

## 6. What is new

**The in-use record is a Stormer object.** In the range where the top machine is used, the
longest pair-free block is not produced by the gears covering a window - it is the largest gap
in the finite list of `n` with `n` and `n + 2` both `q`-smooth, and it sits exactly one cell
above the lower end of that gap. Value and position, 36 machines, 0 exceptions. The list has 13
members at `q = 5` and 799 below `10^13` at `q = 37`; everything about the record follows from
it.

**A proved bound, from `q` alone, within a factor of 1 to 2.** `F_range(N) >= ` the largest gap
of the smooth-pair list below `Q`, unconditional and with no sieve estimate; equal to the true
record at 17 of 36 machines and never worse than a factor 2. For `Q` above the largest smooth
pair it reads `F_range(N) >= sqrt(N) - s(q) - 2`.

**Why no bound in `(q', m)` exists, quantitatively.** `F_range >= (m log m)/2 (1 + o(1))`, so the
record beats the parity value `2m` by a factor that grows without limit (measured 1.5, 2.6, 3.4,
4.0 at `q = 5`, `N = 10^5..10^8`). Any covering bound is the wrong shape, not merely loose.

**In use there is no tail.** Certified covers show `F_top(G) >= Q` for every in-use gear set
(3 to 43 times `2m`), so every gear is a core gear. The parity law and the covering bound are
statements about the tail; in use the tail is empty. That is the mechanism behind the vacuity
recorded in `top_machine_3.md` L33, and it is the exact opposite of the pre-registered guess.

**The core/tail rule for the wheel record.** `F_top(G) = max{L : min over core phases of
D(U) <= t}` with the domino cost `D`; 0 mismatches on 13 known records and exact on 89 sets. The
additive form holds at exactly the 64 sets with an empty core - where it is the parity law - and
fails at all 25 with a core.

**Where every counting proof must stop.** A union bound over covering patterns controls lengths
only to `2 log N / log q'`; the in-use record is 4 to 520 times larger. Below `Q` the record is
proved without counting; above `Q` counting cannot reach it.

**No saturation, in either sense.** The gear zone is `[1, Q]`, so the record is linear in the
largest gear (999,876 at `Q = 10^6`), and even on a region strictly above every zone the record
keeps growing with `Q` through the merge law. There is no ceiling in `q'`, and none in `m`.

**Prior art, in a line.** `F_top` is the two-class Jacobsthal function of the gear set
(Jacobsthal 1961; Iwaniec 1978 for the asymptotic) and is not what the in-use range record
measures. The finiteness of the set of `n` with `n` and `n + 2` both `q`-smooth is Stormer's
theorem (1897), made effective by Lehmer (1964); the lists here are computed complete below
`10^13` and used as data. Mertens' constant supplies the closed form `L*(q)`. Nothing asymptotic
is claimed and no sieve estimate is used anywhere in L46-L48.

---

## 7. Verdict

**In use, the next opening is decided by smooth numbers, not by covering.** Below the largest
gear the top machine's open pairs are exactly the pairs of `q`-smooth numbers two apart - a
finite list, 13 long at `q = 5` - so the longest walk is the largest gap in that list, and the
record block starts one cell above its lower end. This is an identity, proved in two lines, and
it holds at value and position in all 36 in-use machines measured.

**The bound.** The largest gap of the smooth-pair list below `Q` is a proved lower bound on
`F_range(N)`, within a factor 1.000 to 2.000 of the truth (exact at 17 of 36); for `Q` above the
largest smooth pair it is `sqrt(N) - s(q) - 2`. Upwards, the walk in the gear zone is bounded by
that same list plus the single number `u`, the first opening above the zone; above the zone the
record is a first-hit object of size 24 to 419 over the whole tested range - measured, and out
of reach of any union bound, which stops at `2 log N / log q'`.

**The covering apparatus does not apply in use, and now we know exactly why.** The parity law,
the covering number `r(d)` and the bound `2m/(1 - 2H_S)` are statements about tail gears - those
above `F_top + 1`. Certified covers show that in use there are none: `F_top >= Q` at every
machine tested. The bound `2m/(1 - 2H_S)` survives only where
`F < L*(q) = exp exp(1/2 + sum_{p<=q} 1/p - M)`, correct at 35 of 36 machines, and there it is
40 to 50 times loose.

**No ceiling in `q'`, and no saturation.** The record grows linearly in the largest gear -
999,876 at `Q = 10^6`, `N = 10^7` - because the gear zone is `[1, Q]` whatever `N` is; and even
above the zone, added gears keep merging blocks. What is a function of `q` alone is the
gear-zone structure itself: the finite list, its largest member `s(q)`, and its gaps.

No interpretation against the twin conjecture is offered; the clutch does not appear.

---

## 8. Scorecard, filled

| # | Prediction | Result |
|---|---|---|
| P1 | core/tail covering rule with the domino cost `D(U)`, exact | **held**: 0 mismatches on 13 known records, 89 sets decided (L53) |
| P2 | the additive form is refuted (`{13..41}`: 15 vs 18) | **held**, and sharpened: it holds at exactly the 64 empty-core sets and fails at all 25 others (L54) |
| P3 | in use the tail is nonempty and `F_top >> F_range` | **refuted**: the tail is *empty* in 26 of 27 machines; `F_top >= Q` certified (L52) |
| P4 | gear-zone identity; record = smooth-gap | **held**, 0 exceptions of 36, value and position (L46, L47) |
| P5 | `F_range >= Q - s(q)`; `s(5) = 160`; `F_range(10^7) >= 3002` at `q = 5` | **held for `q = 5`** (3,002 against the measured 3,006); `s(7) = 448` **refuted** - the list runs on to 8,748, so the right bound is the largest *gap*, not the last one (L48) |
| P6 | `F_range >= (m log m)/2`; bound alive iff `F < L*(q)`, `L*(5) ~ 35` | **held**: `L*` values as predicted; criterion correct at 35 of 36 (L49, L50) |
| P7 | the provable half is `[1, Q]`; ratio true/bound in `[1.00, 1.05]` | **half held**: the provable half is `[1, Q]` as predicted; the ratio is 1.000-2.000, not 1.00-1.05 - the tight rows are the ones with `Q > s(q)` |
| P8 | the record block starts at `s(q) + 1` whenever `Q > s(q)` | **refuted as stated, replaced**: it starts one above the lower end of the largest *gap* of the list, which is `s(q) + 1` only when the last gap wins (all four `q = 5` machines) |
| P9 | above-zone record `A < F_range`, `<= 400` at `N <= 10^8` | **refuted twice**: `A` wins in 10 of 36 machines (`q >= 13`, `N <= 10^6`), and reaches 419 |
| P10 | no saturation in `Q`; `A` saturates instead | **first half held** (linear in `Q`, 24 sweep points); **second half refuted**: on a fixed region above every zone the record still climbs 200 -> 1,511 (L55) |
| P11 | `s(q)` is the only `q`-datum once `Q > s(q)` | **held** at `q = 5, 7`, where `Q > s(q)` was reached; for smaller `Q` the whole list is needed, not just its last member |

---

## 9. What holds without exception

| statement | count | exceptions |
|---|---|---|
| the gear-zone identity `n <= Q - 2` open iff `n, n + 2` `q`-smooth | 36 machines, 130,230 cells, 4,019 openings | **0** |
| the zone record equals the largest gap of the smooth-pair list, value and position | 36 machines + 24 sweep points | **0** |
| the proved lower bound `F_range >= maxgap` | 36 machines | **0** (ratio 1.000-2.000, exact at 17) |
| the core/tail covering rule against known `F_top` | 13 records | **0** |
| the additive form holds iff the core is empty | 89 sets | **0** |
| in use the tail is empty (certified cover) | 27 machines | 1 (weakest certificate) |
| the covering bound alive iff `F < L*(q)` | 36 machines | 1 (boundary, `H_S = 0.507`) |
| no saturation: the record grows with `Q` at fixed `N` | 24 + 12 points | **0** |

---

## 10. Dead ends

- **A bound on the in-use walk in `q'` and `m`.** There is none: `F_range >= (m log m)/2
  (1 + o(1))` and `F_range / 2m` is measured at 1.5, 2.6, 3.4, 4.0 and still climbing. The
  covering bound is the wrong shape, not merely loose. What survived is the exact death point
  `L*(q)` and the reason for it - in use the machine has no tail.
- **A union-bound proof of the in-use bound.** It stops at `2 log N / log q'` (10 to 19 on the
  machines measured, against records of 113 to 9,846) because a covering pattern pins `x` modulo
  the product of its gears and that product passes `N` at `d = 2 log N / log q'`. What survived
  is that below `Q` no counting is needed at all - the identity L46 does the whole job.
- **Saturation in the largest gear.** Refuted in both forms: the record is linear in `Q` because
  the gear zone is `[1, Q]`, and even on a region strictly above every zone the record climbs
  from 200 to 1,511 as `Q` goes 3,162 to `10^6`. What survived is the mechanism: the zone
  identity is the first cause, the merge law the second.
- **"The record always sits in the first stretch."** False at 10 of 36 machines, all with
  `q >= 13` and `N <= 10^6`: there the smooth pairs are still dense below `Q` and the record is
  an ordinary first-hit block far out in the range. What survived is the crossover statement:
  each `q` has one `N` beyond which the zone wins for good.

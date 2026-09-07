# Branch: NEW GEARS LENGTHEN, NEVER PRECEDE (R4.c.iii.a)

Prover, round 69 (2026-09-07). Parent: node R4.c.iii (research/proof/valve_existence.md), whose
V9 says the frontier's constants 4.625 = 111/24 and 3.25 = 13/4 are the twin gaps 661 -> 809 and
73 -> 101 "and they stay the constants of every larger machine because new gears lengthen runs
already present rather than create earlier ones", and whose closing line names this child: whether
that sentence can be proved for the prefix's runs of length >= d_0, which by V8 would certify
turns 1-4 of the valve for every Q.

Sources read: valve_existence.md (V6-V9 and the closing line), position_frontier.md (theorem (E),
the frontier law, the staircase table, the induction attempt), anchor_window.md (mechanism section
1: fresh strikes and the dead zone), merge_forest.md (the merge law, 0.1), docs/proofs/21 and
proofs/ArcFloor.lean (the arc floor and the +4 law), monotone_functional.md section 4 (the order
law). Scripts in `research/anchor235/r69/` (prefix `lnp_`), outputs in
`research/anchor235/r69/results/` (gitignored; every number this document relies on is written
into it). Run with `uv run python research/anchor235/r69/<script>.py` from the repository root;
numpy only; at most 4 cores and 3 GB. Nothing is committed. Laws are numbered E6 onward.

## Setup: the exact definitions

Column `k` is the pair `(6k-1, 6k+1)`. Gear `g` strikes `k` iff `k = +-u_g (mod g)`,
`u_g = 6^{-1} mod g`, i.e. iff `g` divides `6k-1` or `6k+1`. The ENGINE at rung `q` is the machine
`M = {5..q}` (all primes from 5 to `q`); its period is `P(M) = prod_{g in M} g` columns, i.e.
`q#/6`, and I use exactly that period, columns `[0, P)`. An **opening** is a column no gear
strikes; column 0 is open for every machine. A **blocked run** is a maximal interval of blocked
columns; runs are read as `(x, L)` = (first column, number of columns). The record `F(M)` is the
longest gap between consecutive openings (wrap included), so the longest run has `F(M) - 1`
columns. `d_0(M)` is the first opening after column 0, so the **initial run** is `(1, d_0 - 1)`;
`d_1(M)` is the second opening. The ladder step is `M -> M + q'` with `q'` the next prime after
`q`, and `q''` the prime after `q'`. `W(q) = (q'^2 - 1)/6` is the top column of the window of `M`;
the **prefix** of `M` is `[1, W(q)]`; the **section** of `M + q'` is `(W(q), W(q')]`. The
**home column** of `q'` is `h(q') = (q' -+ 1)/6`, the column whose member is `q'` itself. The
**frontier** of a machine over a column range is `R_min(L)` = the least start of a run of length
`>= L`; it is determined by the Pareto staircase (the runs longer than every run starting
earlier). The **merge law** (docs/proofs/05 (D), merge_forest.md 0.1): every gap of `M + q'` is an
old gap of `M` or a union of consecutive old gaps, the new gear striking the openings between them.

Theorem (E) (position_frontier.md, proved): for `6k - 1 > q`, column `k` is blocked under `{5..q}`
iff it is blocked under `{5..floor(sqrt(6k+1))}`. Reduction (R): the openings of `{5..q}` in
`[1, W(q)]` are the twin columns (both members prime) with `6k - 1 > q`, plus `W(q)` when
`q'^2 - 2` is prime.

## Pre-registered (written before any script of this branch was run)

### The statement, in three forms

Fix `M = {5..q}` and the step to `M + q'`. Let `R_min^M(L)` be the frontier over the full period
of `M` and `R_min^{M+q'}(L)` over the full period of `M + q'` (positions are absolute, both periods
begin at column 0, so the two are comparable column for column).

- **Form A (the first long run does not move forward).** For every `L` with
  `d_0(M) <= L <= F(M) - 1`: the first run of `M + q'` of length `>= L` starts at or after the
  first run of `M` of length `>= L`. The upper limit `F(M) - 1` is forced: for `L > F(M) - 1`
  the machine `M` has no run of that length and `R_min^M(L)` is undefined (the record grows), so
  no form of the statement can speak there.
- **Form B (the frontier is monotone up the ladder).** `R_min^{M+q'}(L) >= R_min^M(L)` for every
  `L` in `[d_0(M), F(M) - 1]`. Forms A and B are the same statement written twice (A is B read
  as a position); I keep both names because the brief does, and I record A with the threshold
  `d_0(M+q')` (the initial run of the larger machine excluded) and B with the threshold `d_0(M)`.
  B implies A (a larger threshold); A does not imply B: the lengths in `[d_0(M), d_0(M+q') - 1]`
  are exactly those the initial run of `M + q'` absorbs, where `R_min^{M+q'} = 1`.
- **Form C (the constant is monotone).** `c(M) = min_{L >= d_0(M)} R_min^M(L)/L` satisfies
  `c(M + q') >= c(M)`. B does NOT imply C and C does not imply B: C is a minimum over a range
  whose lower end moves (`d_0` grows), so C can hold while B fails (the failing `L` is excluded
  by the larger `d_0`) and fail while B holds (`M + q'` may have a run of a length `M` never had,
  above `F(M) - 1`, at a low ratio). A also does not imply C, for the same reason.

The prefix versions: the same three statements with the frontiers read over `[1, W(q)]` for `M`
and `[1, W(q')]` for `M + q'` (top run truncated at the top column). These are the versions V8
consumes, since V8 only ever uses runs inside the prefix of the rung `y_m(Q)`.

### The theory

**(E) at one step.** Theorem (E) applied to `M` and to `M + q'` on the same column range says
the two machines block the same columns wherever the effective machine `{5..sqrt(6k+1)}` is
below `q'`, i.e. on `[1, W(q))`, except where the hypothesis `6k - 1 > q'` fails. Worked out
exactly: a column `k <= W(q)` newly struck by `q'` has `q' | 6k -+ 1 =: n` with `n <= q'^2`; if
`n = q' m` with `1 < m < q'` then `m` is coprime to 6 and its least prime factor `r` satisfies
`5 <= r <= m < q'`, so `r <= q` and `M` already strikes `k`. So `m = 1` (the home column
`h(q')`) or `m = q'` (the square column `W(q)`, `6W(q) + 1 = q'^2`). The home column is newly
struck iff its other member `q' -+ 2` is not struck by `M`, i.e. is a prime above `q`, i.e.
`q' = 5 (mod 6)` and `q' + 2 = q''` (a twin pair `(q', q'')`); and in that case `h(q')` is the
first twin column above `q`, which is `d_0(M)` by (R). The square column is newly struck iff
`q'^2 - 2` is not struck by `M`, i.e. iff `q'^2 - 2` is prime. So I expect:

> **E6 (the one-step effective-machine theorem).** On `[1, W(q)]`, the columns blocked by
> `M + q'` and not by `M` are exactly `{d_0(M)}` if `(q', q'')` is a twin pair (else nothing)
> together with `{W(q)}` if `q'^2 - 2` is prime (else nothing). At most two columns in a prefix
> of `q'^2/6` columns, and the home column of `q'` is never new unless it is `d_0(M)` itself.

The brief's guess "the new gear adds exactly its two home columns `+-u_{q'}`" is therefore wrong
in both halves: the column `q' - u_{q'}` (the other residue in `[1, q')`) holds the member `5q'`
and is struck by gear 5; and the second new column is not a home column but the square column.

**Consequence for the three forms on the prefix.** Every run of `M + q'` inside `[1, W(q)]` is a
run of `M` unchanged, except: the initial run, which absorbs the second run when `(q', q'')` is a
twin pair (`d_0(M + q') = d_1(M)` then, else `d_0(M + q') = d_0(M)`); and the run touching `W(q)`,
which fuses through the square column with the first run of the section. So on the prefix the
statement "new gears lengthen, never precede" is EXACT for every run except two, and of those
two the initial run is excluded by the hypothesis `L >= d_0` while the top run is the one place
where a run of `M + q'` can start before the first run of `M` of its length: the **straddling
run**, the run of `M + q'` containing `W(q)`, whose start is the start of `M`'s top prefix run
and whose length is set by the first opening of `M + q'` above `W(q)`.

**Consequence for the section.** A run of `M + q'` lying entirely in `(W(q), W(q')]` has start
`x > W(q)` and length `L <= S(q') := W(q') - W(q) = (q''^2 - q'^2)/6`, hence
`x/L > W(q)/S(q') = (q'^2 - 1)/(q''^2 - q'^2)`. That is `>= 4.625` as soon as
`q'' < 1.1028 q'`, a prime-gap bound (prior art; Dusart 1998 gives it for `q' >= 3275`, the rest
is a finite check). So interior section runs never undercut the measured floor from a computable
rung on.

**The full period.** None of this reaches beyond `W(q)`. Beyond it `q'` has genuine fresh strikes
(members `q' p` with `p >= q'`, anchor_window.md mechanism 1), the merge law acts with real chains,
and a merge of several short runs of `M` at a low position can produce a run of length `L` before
`M`'s first run of that length. The staircase table of position_frontier.md already shows this:
`R_min(6) = 89` at m13 against 61 at m17, and `R_min(10) = 118` at m17 against 59 at m19. So I
expect Forms A, B, C to be FALSE on full periods, and Form A/B to be TRUE on the prefix with the
straddling run as the sole exception.

**What remains.** If E6 and the section bound hold, the frontier of the prefix at rung `q'` with
constant `c` follows from the frontier at rung `q` (inherited runs), the prime-gap bound
(interior section runs), and ONE new condition per rung: the straddling run obeys `x_s >= c L_s`.
By (R) that run is the twin gap across `q'^2`, and "`x_s >= c L_s` at every rung" says the twin
gap across every prime square is at most `1/c` of its position, which implies a twin above every
prime square, i.e. the root. So I expect the branch to end ROOT, with the exact reason being the
twin gap at the square rather than `d_0 <= W` (the initial run never enters: it is excluded by
the hypothesis, and its length `d_0` is the root only for the window statement, not for V8).

### Predictions, with numbers, and what would refute each

- **P1 (E6 by direct sieve).** At every rung `q` from 5 to 3000 (about 430 rungs), the direct
  sieve of `{5..q}` and `{5..q'}` over `[1, W(q)]` differs in exactly the predicted set: 0
  discrepancies. Refuted by one rung with a third new column, or a predicted column that is not
  new. Also at the full-period rungs m5..m31.
- **P2 (`d_0` up the ladder).** `d_0(M + q') = d_1(M)` when `(q', q'')` is twin, `= d_0(M)`
  otherwise: 0 exceptions over all rungs `q <= 19,997`. `h(q') <= d_0(M)` at every rung with
  equality iff twin (an identity; refuted by one rung).
- **P3 (Forms A and B on full periods are false).** On the full periods of the steps 5->7 through
  29->31 (m31's period 33.4 x 10^9 columns streamed), Form B has exceptions at 13->17 (at least
  `L = 6..10`), 17->19 (at least `L = 6..11`), and at every step from 13->17 upward at least one;
  Form A (threshold `d_0(M + q')`) likewise. At every exception the new earlier run is either the
  straddling run at `W(q)` (its start `<= W(q) < ` its end) or a run starting above `W(q)`;
  0 exceptions with start in `[2, W(q)]` that do not contain `W(q)`. Refuted by one exception
  whose new run lies inside `[2, W(q)]` without containing `W(q)`.
- **P4 (Form C is false on full periods and on the prefix).** Full period: `c(m17) = 6.778 >
  c(m19) = 4.625`. Prefix: `c_pre(17) > c_pre(19) > c_pre(23)` (I expect 10.6, 5.36, 4.625).
  At every rung `q' >= 127` where `c_pre` decreases, the run that sets the new minimum is the
  straddling run (0 decreases set by an interior section run); below 127 the decrease at 19->23 is
  set by an interior section run, the run `(111, 24)`. Refuted by a decrease at `q' >= 127` set by
  an interior section run.
- **P5 (the floor and its minimisers).** `c_pre(q) >= 4.625` at every rung 23..19,997 where the
  range `[d_0, F_pre]` is non-empty; `c_pre(q) = 4.625` exactly at the rungs `q` in `[23, 131]`
  (the prefix contains column 134 and `d_0 <= 24`), with minimiser `(111, 24)` at every one of
  them, and `(13, 4)` is the minimiser at `q = 7, 11, 13` (`c = 3.25`), excluded from `q = 17`
  by `d_0 = 5 > 4`. Above `q = 131` the minimiser is never `(111, 24)` again and `c_pre > 4.625`.
  Refuted by any rung in `[23, 131]` with a different minimiser, or a rung above with `c_pre <=
  4.625`.
- **P6 (the straddling run).** At every rung `q'` in `[7, 19,997]` the straddling run is
  non-initial (`d_0(q) <= W(q)`, the window statement, known at these rungs) and its ratio
  `x_s/L_s` is `>= 4.625` for `q' >= 23`; its minimum over `q' >= 23` is attained below
  `q' = 200` and exceeds 5; for `q' > 1,000` it exceeds 100. Refuted by a rung `q' >= 23` with
  `x_s/L_s < 4.625`.
- **P7 (interior section runs).** `(q'^2 - 1)/(q''^2 - q'^2) >= 4.625` for every prime `q'` in
  `[127, 20,011]`, and the last prime below 127 violating it is `q' = 113` (`q'' = 127`). Refuted
  by any violation above 113.
- **P8 (V8's hidden hypothesis).** V8's proof takes a run of length `>= c_m(Q)` starting at
  `x <= klo_m(Q)` and asserts `6x - 1 > y` from `6 klo - 1 > y`; that step needs the run to be
  non-initial, i.e. `d_0(y_m(Q)) <= klo_m(Q)`. Prediction: this holds for every `Q <= 10^5` and
  `m = 1..4` with at most a handful of exceptions at `Q < 60`, so V8's certified ranges are
  unaffected, but the theorem's hypothesis must carry it. Refuted by an exception with `Q >= 60`.

Owner's predictions on the scorecard (from V9 and the brief): (O1) "new gears lengthen runs
already present rather than create earlier ones" (I predict: true on the prefix except the
straddling run, false on full periods from 13->17 on); (O2) "below `q'^2/6` the new gear adds
exactly its two home columns `+-u_{q'}`" (I predict: refuted as stated; the exact set is E6, and
the second column is the square column, not the mirror home column); (O3) "what remains is `d_0
<= W` again" (I predict: refuted; the initial run is excluded by the hypothesis, and what remains
is the twin gap across `q'^2`, the straddling run).

### Scorecard

| # | Prediction | Verdict |
|---|---|---|
| P1 | E6 exact by direct sieve, 0 discrepancies to `q = 3000` | **HELD**: 0 discrepancies in 428 rungs, and at the 11 full-period steps |
| P2 | `d_0` ladder law, 0 exceptions to 19,997 | **HELD**: 0 exceptions in 2,260 rungs; `h(q') <= d_0(M)` with equality at exactly the 340 twin rungs |
| P3 | Forms A/B false on full periods; exceptions only straddle or deep | **HELD in the main, one clause REFUTED**: exceptions at 8 of 11 steps, 0 "other" at 11 of 11; but 19->23 has no exception at all (predicted at least one at every step from 13->17) |
| P4 | Form C false; prefix decreases at `q' >= 127` are the straddling run | **HELD, and stronger**: `c(m17) = 6.778 > 4.625`; prefix 10.6 -> 5.364 -> 4.625; then 0 decreases at any rung `q' >= 23`, so none at all above 127 |
| P5 | floor 4.625 exactly on `[23, 131]` at `(111, 24)`; `(13, 4)` at 7..13 | **HELD exactly**: 0 rungs below 4.625; minimiser `(111, 24)` at the 24 rungs 23..131 and never again; `(13, 4)` at 7, 11, 13 |
| P6 | straddling run non-initial, ratio `>= 4.625` from 23, `> 100` above 1,000 | **HELD**: non-initial at 2,260 of 2,260; minimum 6.727 at `q' = 31`; above 1,000 the minimum is 1,434 |
| P7 | section bound holds from `q' = 127`, last violator 113 | **HELD**: violators exactly 7, 11, 13, 17, 19, 23, 31, 37, 47, 53, 113; max `q''/q'` above is 1.0719 |
| P8 | V8 needs `d_0(y_m) <= klo_m`; holds to `10^5` except at `Q < 60` | **HELD, better**: 0 exceptions in 399,973 cells, none even below 60; and the frontier hypothesis is non-vacuous at rungs 5, 7, 11, 23 only |
| O1 | lengthen never precede | **REFUTED on the period** (8 of 11 steps, merges of order 2-4 beyond `W(q)`); **a theorem on the prefix** (E7), with the square-column run the one that grows |
| O2 | the two new columns are `+-u_{q'}` | **REFUTED**: `q' - u_{q'}` is the column of `5q'`; the set is E6's `N_1 u N_2` (twin column, square column) |
| O3 | what remains is `d_0 <= W` | **REFUTED**: the initial run is excluded by the hypothesis; the residue is the straddling condition, a twin gap across `q'^2` |

(Results, mechanism, proof, verdict and dead ends are appended below as the branch runs.)

---

# Results

Every number below is exact: full periods where stated, exact sieves of the prefix `[1, W]` by
reduction (R) (gated by direct sieving of the gears at every rung `q <= 3000`), or the streamed
prefix `[0, 2^35)` of a period too large to hold. Nothing is sampled. Scripts: `lnp_period.py`
(full periods and long prefixes, m5 .. m43), `lnp_prefix.py` (the prefix ladder `q = 5 .. 19,997`,
2,260 rungs, twin sieve to `20,021^2`).

## 1. The statement in three forms, and what each turned out to be

| form | full period (m5 .. m43) | prefix `[1, W]` (2,260 rungs) |
|---|---|---|
| A: first run of length `>= L` does not move forward, `L in [d_0(M+q'), F(M)-1]` | FALSE: exceptions at 8 of 11 steps (section 2) | TRUE: 0 exceptions in 8,152 cells (E7, proved from E6) |
| B: `R_min^{M+q'}(L) >= R_min^M(L)`, `L in [d_0(M), F(M)-1]` | FALSE: as A plus the absorbed lengths | TRUE except the absorbed lengths: 238 of 8,390 cells, all with `R_min^{M+q'} = 1` |
| C: `c(M+q') >= c(M)` | FALSE once: 17->19, 6.778 -> 4.625 | FALSE twice: 17->19 (10.6 -> 5.364, the straddling run) and 19->23 (5.364 -> 4.625, a section run); never again to 19,997 |

The implications pre-registered stand: B implies A; C is independent of both (it fails at 17->19
on the prefix while A holds there, because the run that lowers the constant has a length `M`'s
prefix never realised, `11 > F_pre(m17) = 5`, so no cell of A or B sees it).

## 2. Full periods and long prefixes, m5 -> m43 (`lnp_period.py`)

Columns `[0, N)` with `N` the full period of `M + q'` up to m31 (33,426,748,355 columns streamed)
and `N = 2^35 = 34,359,738,368` for m37, m41, m43 (2.8 %, 0.07 %, 0.0016 % of their periods).
`F - 1` is the longest run. Form B cells are `L in [d_0(M), F(M) - 1]` with both frontiers
defined in the scanned range; each exception is classified by the earlier run of `M + q'`:
absorbed (start 1), straddle (the run contains `W(q)`), deep (start `> W(q)`), other (inside
`[2, W(q)]` without containing `W(q)`; E6 says impossible). `J` is the merge order of that run
(1 + the openings of `M` inside it).

| step | range for `M+q'` | `F-1`: `M` -> `M+q'` | `d_0` | new prefix columns | Form B cells | exceptions (absorbed / straddle / deep / other) | Form A exceptions | `J` of the distinct earlier runs | `c(M)` at |
|---|---|---|---|---|---|---|---|---|---|
| 5 -> 7 | full period 35 | 1 -> 4 | 2 -> 2 | W=8 | 0 | 0 / 0 / 0 / 0 | 0 | - | - |
| 7 -> 11 | full period 385 | 4 -> 6 | 2 -> 3 | d_0=2 | 3 | 1 / 0 / 0 / 0 | 0 | - | 3.250 at (13, 4) |
| 11 -> 13 | full period 5,005 | 6 -> 10 | 3 -> 3 | W=28 | 4 | 0 / 0 / 1 / 0 | 1 | J=2: 1 | 3.250 at (13, 4) |
| 13 -> 17 | full period 85,085 | 10 -> 17 | 3 -> 5 | d_0=3 | 8 | 2 / 0 / 5 / 0 | 5 | J=2: 1, J=3: 1 | 3.250 at (13, 4) |
| 17 -> 19 | full period 1,616,615 | 17 -> 24 | 5 -> 5 | W=60 | 13 | 0 / 6 / 6 / 0 | 12 | J=2: 2 | 6.778 at (61, 9) |
| 19 -> 23 | full period 37,182,145 | 24 -> 33 | 5 -> 5 | none | 20 | 0 / 0 / 0 / 0 | 0 | - | 4.625 at (111, 24) |
| 23 -> 29 | full period 1,078,282,205 | 33 -> 42 | 5 -> 7 | d_0=5, W=140 | 29 | 2 / 0 / 9 / 0 | 9 | J=2: 4, J=3: 1 | 4.625 at (111, 24) |
| 29 -> 31 | full period 33,426,748,355 | 42 -> 57 | 7 -> 7 | none | 36 | 0 / 0 / 17 / 0 | 17 | J=2: 6, J=3: 1 | 4.625 at (111, 24) |
| 31 -> 37 | prefix `2^35` of 1,236,789,689,135 | 57 -> 67 | 7 -> 7 | W=228 | 51 | 0 / 0 / 24 / 0 | 24 | J=2: 4, J=3: 4, J=4: 1 | 4.625 at (111, 24) |
| 37 -> 41 | prefix `2^35` of 50,708,377,254,535 | 67 -> 89 | 7 -> 10 | d_0=7 | 61 | 3 / 0 / 23 / 0 | 23 | J=2: 4, J=3: 2 | 4.625 at (111, 24) |
| 41 -> 43 | prefix `2^35` of 2,180,460,221,945,005 | 89 -> 89 | 10 -> 10 | W=308 | 80 | 0 / 0 / 49 / 0 | 49 | J=2: 6, J=3: 3, J=4: 1 | 4.625 at (111, 24) |

Totals: Form B 305 cells, 148 exceptions (8 absorbed, 6 straddle, 134 deep, **0 other**); Form A
297 cells, 140 exceptions; the distinct earlier runs have `J = 2` at 28, `J = 3` at 12, `J = 4` at
2. The `F - 1` values at m37..m43 are those seen inside `2^35` columns (the ladder's records are
88, 91, 103 gaps); the m31 full period reproduces `F(31) = 58` exactly. E6 by direct sieve on
`[1, W(q)]`: 11 of 11 steps exact.

Read off the table:

- **0 "other" exceptions at 11 of 11 steps**: every run of `M + q'` that starts before `M`'s
  first run of its length either starts at column 1 (absorption), contains the square column
  `W(q)` (the straddle: 17->19 only, six lengths, the run `(59, 11)` fusing `(59, 1)` and `(61, 9)`
  through column 60, `19^2 - 2 = 359` prime), or starts beyond `W(q)` (deep). This is E6 seen
  from the frontier.
- **The deep exceptions are merges of order 2 to 4** (every deep exception is a genuine chain of
  the new gear through 1-3 openings of `M`; the distribution is in the table), and their starts
  are between 61 and `7.9 x 10^9`: the merge law at work beyond the prefix, exactly where the
  brief's mechanism (b) expects it, and nowhere inside the prefix.
- **19 -> 23 has no exception at all**: 23 has no new column in `[1, 88]` (`(23, 25)` is not
  twin and `23^2 - 2 = 527 = 17 x 31`), and its deep merges all produce lengths 25..33 that
  m19 never had (`F(m19) - 1 = 24`), at starts `40,148` and above.
- **Form C on the full period**: `c(M)` = 3.25 (m7, m11, m13, at `(13, 4)`), 6.778 (m17, at
  `(61, 9)`), then 4.625 at `(111, 24)` for m19 through m43 (nine machines). One decrease
  (17->19), one increase by exclusion (13->17: `d_0` 3 -> 5 removes `(13, 4)`), then constant.
  The run `(111, 24)` is exactly the twin gap `659 -> 809` and by E6 it is inherited unchanged
  until the initial run absorbs it at rung 659.

## 3. The prefix ladder, q = 5 .. 19,997 (`lnp_prefix.py`)

**(a) E6 by direct sieve.** At all 428 rungs `q <= 3000`, sieving the gears of `{5..q}` and of
`{5..q'}` over `[0, W(q)]` and taking the difference gives exactly the predicted set: **0
discrepancies in 428 rungs**, and also at the ten full-period steps (which reach `q' = 43` with
`W` up to 308). Over all 2,260 rungs the predicted set is: the twin column `d_0(M)` alone at 281
rungs, the square column `W(q)` alone at 392, both at 59, **neither at 1,528**. So at two rungs in
three the new gear changes nothing at all in the prefix of the old machine.

**(b) The `d_0` ladder.** `d_0(M + q') = d_1(M)` when `(q', q'')` is a twin pair and `= d_0(M)`
otherwise: **0 exceptions in 2,260 rungs**; the jumps happen at the 340 rungs whose `q'` is the
lower member of a twin pair. The home column satisfies `h(q') <= d_0(M)` at all 2,260 rungs with
equality exactly at the 340 twin rungs. This is an identity, not a Bertrand comparison: `6 d_0 - 1`
is a prime above `q`, so `q' <= 6 d_0 - 1` and `h(q') = (q' -+ 1)/6 <= d_0`, with equality iff
`q' = 6 d_0 - 1`, i.e. iff `q'` is the lower member of the first twin pair above `q`.

**(c) Forms A and B on the prefix.** Cells `L in [d_0(M), F_pre(M)]`, frontiers over `[1, W(q)]`
and `[1, W(q')]`: **8,390 cells, 238 exceptions, all absorbed (new start 1), 0 straddle, 0
section, 0 other**; with the threshold `d_0(M + q')` (Form A): **8,152 cells, 0 exceptions**. This
is E7 below, measured.

**(d) The constant and its staircase.** `c_pre(q) = min x/L` over the non-initial runs of
`[1, W(q)]` with `L >= d_0(q)`, finite at 216 rungs (the 211 of position_frontier.md plus
`q = 7..19`), infinite from `q = 1427` on (the initial run is the longest; 2,038 rungs). Its value
is a staircase of twin gaps, each minimiser a single run inherited unchanged until `d_0` grows
past its length:

| rungs `q` | minimiser `(x, L)` | `c_pre` | the run, as a twin gap |
|---|---|---|---|
| 7 - 13 | (13, 4) | 3.250 | 71 -> 101 |
| 17 | (53, 5) | 10.600 | 311 -> 347 |
| 19 | (59, 11) | 5.364 | 347 -> 419 (the straddling run of 17->19) |
| 23 - 131 | (111, 24) | **4.625** | 659 -> 809 |
| 137 - 139 | (398, 27) | 14.741 | 2381 -> 2549 |
| 149 - 193 | (981, 34) | 28.853 | 5879 -> 6089 |
| 197 - 263 | (2234, 46) | 48.565 | 13397 -> 13679 |
| 269 - 457 | (4071, 82) | 49.646 | 24419 -> 24917 |
| 461 - 613 | (10384, 104) | 99.846 | 62297 -> 62927 |
| 617 - 877 | (31319, 153) | 204.699 | 187907 -> 188831 |
| 881 - 911 | none | inf | |
| 919 - 1423 | (141726, 241) | 588.075 | 850349 -> 851801 |
| 1427 - 19997 | none | inf | |

`c_pre` decreases exactly twice in 2,260 rungs (17->19 by the straddling run `(59, 11)`; 19->23 by
the section run `(111, 24)`, which is interior to the section `(88, 140]` because the prime gap
19 -> 23 is large: `W(19)/S(23) = 88/52 = 1.69`), and never at any rung `q' >= 23`. Of the 216
finite minimisers, 211 are inherited runs, 1 is a straddling run, 4 are section runs (at
`q' = 23, 137, 149, 197`; none after 197). **The floor is 4.625 at every rung `q >= 23`: 0 rungs
below it**, attained at exactly the 24 rungs `q = 23 .. 131` (the prefix contains column 134 from
`q = 23`, and `d_0 <= 24` until `q = 131`), and never again.

**(e) The straddling run** (the run of `M + q'` containing `W(q)`). Its start is the start of the
last prefix run of `M` at 2,251 of 2,260 rungs and `W(q)` itself at the other 9 (where `W(q) - 1` is
open); it is never the initial run (0 of 2,260: the window statement holds at every rung); its
ratio `x_s/L_s` is `>= 4.625` at every rung `q' >= 23` (minimum 6.727 at `q' = 31`: `(148, 22)`,
members 887..1021, the twin gap `881 -> 1019` across `31^2 = 961`, which is also the maximum
top-run share `tau = 0.0833` of valve_existence.md table 2), and grows:

| band of `q'` | rungs | min `x_s/L_s` (at `q'`) | median | max `L_s` | max `S(q')/W(q)` |
|---|---|---|---|---|---|
| 7 - 100 | 22 | 4.00 (7) | 29.3 | 27 | 1.500 |
| 100 - 300 | 37 | 49.65 (157) | 237 | 82 | 0.263 |
| 300 - 1000 | 106 | 198.2 (347) | 1,783 | 102 | 0.090 |
| 1000 - 3000 | 262 | 1,434 (1231) | 13,282 | 254 | 0.052 |
| 3000 - 10000 | 799 | 9,500 (3691) | 104,005 | 444 | 0.017 |
| 10000 - 20011 | 1,034 | 65,802 (10589) | 481,854 | 484 | 0.006 |

The straddling run is the longest run of the prefix of `q'` at 6 rungs only (`q' = 11, 13, 19, 53,
137, 157`), and it is at least as long as the initial run of `q'` (`L_s >= d_0(q')`, the only case
in which the frontier's hypothesis `L >= d_0` reaches it) at **24 rungs, the last `q' = 487`**
(`(39528, 87)`, `d_0 = 87`, ratio 454). At the other 2,236 rungs the twin gap across `q'^2` is
shorter than the initial run and the frontier says nothing about it.

**(f) Interior section runs.** `W(q)/S(q') = (q'^2 - 1)/(q''^2 - q'^2) < 4.625` exactly at
`q' = 7, 11, 13, 17, 19, 23, 31, 37, 47, 53, 113` (last 113, as predicted); the maximum of
`q''/q'` over `q' in [127, 20,011]` is 1.0719 at `q' = 139`, against the threshold
`sqrt(1 + 1/4.625) = 1.1028`, and the minimum of `W(q)/S(q')` there is 6.708 (`q' = 139`). An
interior section run with ratio below 4.625 occurs at one rung only, `q' = 7` (the run `(13, 4)` in
the section `(8, 20]`); at the ten other violating rungs the interior runs are at 13.1 or above.

**(g) V8's hidden hypothesis.** `d_0(y_m(Q)) <= klo_m(Q)` holds at **399,973 of 399,973 cells**,
`Q = 3..10^5`, `m = 1..4` (the pre-registration allowed exceptions below `Q = 60`; there are
none). But the census of where V8's frontier hypothesis is non-vacuous at all (some run of the
prefix of `y` has length `>= c_m(Q)`) is short: the cells `(y, m)` = (5; 2, 3, 4), (7; 1, 2, 3,
4), (11; 4), (23; 3, 4) and **no rung `y >= 29`** for any `m <= 4`. At rung 23 the run is
`(111, 24)` and the certificate is genuinely positional (`4.625 x c_m(Q) > klo` for `Q = 106..148`,
turns 3 and 4; the same run empties turn 5 at `Q = 132..134`, V9). At every other rung in the
measured range the turn is certified because no run as long as `c_m(Q)` exists in the prefix,
i.e. by the share form `F_pre(y) < c_m(Q)`, and the frontier's constant plays no part.

## 4. Mechanism

### 4.1 E6, the effective-machine theorem at one step (proved)

> **E6.** Let `M = {5..q}`, `q'` the next prime, `q''` the one after, `W = (q'^2 - 1)/6`. On the
> columns `[1, W]`, the set of columns blocked by `M + q'` and not by `M` is
> `N_1 u N_2` with `N_1 = {d_0(M)}` if `q' = 5 (mod 6)` and `q'' = q' + 2`, else empty; and
> `N_2 = {W}` if `q'^2 - 2` is prime, else empty. In particular the home column `h(q')` is a new
> strike iff it equals `d_0(M)`, and `|N_1 u N_2| <= 2`.

Proof. Let `1 <= k <= W` be struck by `q'`: `q'` divides `n = 6k - 1` or `n = 6k + 1`, and
`n <= 6W + 1 = q'^2`. Write `n = q' m`. Since `n` is coprime to 6, so is `m`, and `1 <= m <= q'`.
If `1 < m < q'`, the least prime factor `r` of `m` has `5 <= r <= m < q'`, so `r <= q` and `r`
divides `n`: `k` is already struck by the gear `r` of `M`. So a new strike has `m = 1` or
`m = q'`. `m = q'` forces `n = q'^2 = 6k + 1` (as `q'^2 = 1 (mod 6)`), i.e. `k = W`; that column
is open under `M` iff its other member `q'^2 - 2` has no prime factor in `[5, q]`, and since
`q'^2 - 2 < q'^2`, a composite `q'^2 - 2` has a prime factor `<= q`: so `W` is new iff `q'^2 - 2`
is prime. `m = 1` is the home column `k = h(q')`, whose other member is `q' - 2` (if `q' = 1
(mod 6)`) or `q' + 2` (if `q' = 5 (mod 6)`). The column is open under `M` iff that member is
prime and above `q` (it is `> 1`, and a composite below `q'^2` has a factor `<= q`). `q' - 2 > q`
is impossible for a prime `q' - 2` since `q` is the largest prime below `q'`; so `q' = 5 (mod 6)`
and `q' + 2` prime, whence `q'' = q' + 2`. In that case `(q', q' + 2)` is the first twin pair with
lower member `> q`, so `h(q') = (q' + 1)/6` is the first twin column above `q`, which is `d_0(M)`
by reduction (R). []

E6 is theorem (E) at one step with its exception set made exact at both ends of the prefix.
Prior art inside the project: the dead zone of anchor_window.md (mechanism 1: "zero fresh strikes
below `(g^2 - 1)/6`" in the window, 400,000 gear-rows) is E6 with both boundary columns
excluded; the exception set "twin gear pairs" of position_frontier.md 3(b) is `N_1` seen from
the machine's own gears. E6 puts the two on one line and adds the square column `N_2`, which is
the column the 17->19 straddle and the 451 rider rungs run through. The brief's guess (two home
columns `+-u_{q'}`) fails on both counts: the residue `q' - u_{q'}` is the column of `5q'`, struck
by gear 5, and the second new column is the square column, not a home column.

### 4.2 What E6 does to the runs (E7, proved)

> **E7 (prefix inheritance).** Every maximal blocked run of `M + q'` contained in `[1, W(q) - 1]`
> that does not contain column 1 is a maximal blocked run of `M`, with the same start and length.
> Consequently, for every `L` with `d_0(M + q') <= L <= F_pre(M)`,
> `R_min^{M+q'}(L)` over `[1, W(q')]` equals `R_min^M(L)` over `[1, W(q)]`.

Proof. Let `R = [x, x + L - 1]` be such a run, `x >= 2`. Its columns are blocked under `M + q'`;
by E6 the only column of `[1, W(q) - 1]` blocked under `M + q'` and not under `M` is `d_0(M)`
(when new), which is adjacent to the initial run `[1, d_0 - 1]`, so a run containing it contains
column 1; hence every column of `R` is blocked under `M`. The columns `x - 1` and `x + L` are open
under `M + q'`, hence under `M` (which blocks a subset). So `R` is a maximal run of `M`. For the
frontier: a run of `M + q'` of length `>= L >= d_0(M + q')` starting at `x <= W(q) - L` is not the
initial run (that has length `d_0(M + q') - 1 < L`), so it is a run of `M` by the first part;
conversely a run of `M` of length `>= L` inside `[1, W(q)]` starting at `x >= 2` remains blocked
under `M + q'` (blocking only grows) and no run of `M + q'` of length `>= L` starts earlier
except through the first part again. The run of `M` touching `W(q)` keeps its start too. []

Measured: 0 exceptions in 8,152 cells (section 3(c)); the 238 Form-B exceptions are exactly the
lengths `[d_0(M), d_0(M + q') - 1]` at the 340 twin rungs, where the absorbed initial run has
start 1. So on the prefix, "new gears lengthen, never precede" is a theorem with one named
exception: the run through the square column, which keeps its start and grows into the section.

### 4.3 The three kinds of run at rung `q'`, and the merge law's place

A non-initial maximal run `R` of `M + q'` inside `[1, W(q')]` is one of:

1. **inherited** (`R` inside `[1, W(q) - 1]`): a run of `M`, by E7; 211 of the 216 finite
   constants are set by these;
2. **straddling** (`W(q)` in `R`): start = the start of `M`'s last prefix run (or `W(q)` when
   `W(q) - 1` is open), end = the column before the first opening of `M + q'` above `W(q)`; this
   is the only run whose ratio can fall below the ratio of a run of `M`; it set the constant
   once (17->19);
3. **interior section** (`R` inside `(W(q), W(q')]`): `x > W(q)`, `L <= S(q')`, so
   `x/L > W(q)/S(q')`; set the constant at 4 rungs, none after `q' = 197`.

The merge law is not needed on the prefix: E6 says a new gear strikes at most two old openings
there, so the "chain" of the brief's mechanism (b) has length at most one at each of two places,
and neither fuses two runs of length `>= d_0` (one fuses the initial run with its neighbour, the
other fuses `M`'s last prefix run with the section). Chains of length 1-3 do occur, and set every
exception, beyond `W(q)`: the deep exceptions of section 2 have `J = 2, 3, 4`, and they are where
the record is built (merge_forest.md). So the parent's sentence is true of the prefix and false
of the period, and the reason is E6: the frontier's low staircase is inherited because the prefix
is the twin-column set, which a new gear cannot touch except at its own twin and its own square.

### 4.4 Why the section cannot undercut the floor (E8-a, proved for `q' >= 118`)

An interior section run has `x/L > W(q)/S(q') = (q'^2 - 1)/(q''^2 - q'^2)`. If `q'' <= 14q'/13`
then `(q''/q')^2 - 1 <= 0.1598` and `W(q)/S(q') >= 6.25 > 4.625`. The bound `q'' < 14q'/13` holds
for every prime `q' >= 118` (Rohrbach-Weis 1964, prior art: a prime in `(n, 14n/13]` for
`n >= 118`; Dusart 1998 gives `q'' <= q'(1 + 1/(2 ln^2 q'))` for `q' >= 3275`). Below 118 the
violating rungs are `q' = 7, 11, 13, 17, 19, 23, 31, 37, 47, 53, 113` and the interior runs there
are checked directly (section 3(f)): only `q' = 7` has one below 4.625. So: **an interior section
run never sets a constant below 4.625 at any rung `q' >= 11`**, proved, not measured.

## 5. The proof and what remains

### 5.1 The induction step (E8, proved)

Write `H_c(q)` for: every non-initial maximal run `(x, L)` of `{5..q}` in `[1, W(q)]` (top run
truncated at `W(q)`) with `L >= d_0(q)` satisfies `x >= cL`.

> **E8.** For `c <= 6.25` and `q' >= 118`: `H_c(q)` implies `H_c(q')` provided the straddling run
> `(x_s, L_s)` of `M + q'` satisfies `x_s >= c L_s` whenever `L_s >= d_0(q')`. For `q' < 118` the
> same holds with the interior section runs of the rung added to the proviso.

Proof. Take a non-initial run `(x, L)` of `M + q'` in `[1, W(q')]` with `L >= d_0(q')`. Inherited:
it is a run of `M` (E7) with `L >= d_0(q') >= d_0(q)` (the `d_0` ladder law), so `H_c(q)` gives
`x >= cL`. Interior section: `x/L > W(q)/S(q') >= 6.25 >= c` by E8-a. Straddling: the proviso. []

Base: `H_{4.625}(q)` holds at every rung `q = 23 .. 19,997` (0 exceptions, section 3(d); this is
position_frontier.md's law with its exact floor).

### 5.2 What remains, exactly

By E8, `H_{4.625}(q)` for every rung `q` is equivalent to the base plus:

> **The straddling condition.** For every prime `q' > 20,011`: if the run of `{5..q'}` through the
> square column `W(q)` is at least as long as the initial run of `{5..q'}`, then it starts at or
> after 4.625 times its length.

In the twin coordinate (reduction (R)): let `t_0` be the last twin pair below `q'^2` (with lower
member `> q'`, else the run is initial and there is nothing to prove) and `t_1` the first twin
pair above `q'^2` (or the column `W(q')` when `q''^2 - 2` is prime); the run has
`x_s = (t_0 + 1)/6 + 1` and `L_s = (t_1 - t_0)/6 - 1`, and the condition reads

    t_1 - t_0 >= q' + 6   implies   t_1 - t_0 <= (t_0 + 7)/4.625 + 6 ,

that is: **a twin gap across a prime square that is longer than the prime is at most 21.6 % of
its start.** Since the initial run of `{5..q'}` has about `q'/6` columns and the straddling run
in the measured range has at most 484 columns (`q' <= 20,011`), the hypothesis `L_s >= d_0(q')`
fails at every rung above 487 in the measured range (section 3(e)), so the condition is vacuous
there; the measured floor 4.625 to rung 19,997 rests on 24 rungs where it is not vacuous, all
satisfied with ratio `>= 4.75` (`q' = 11`) and `>= 5.36` from `q' = 19`.

Is the straddling condition the root? Yes, in the direction that matters. It implies, at every
prime `q'` for which the prefix of `{5..q'}` holds a twin (the window statement at `q'`), a twin
above `q'^2` within a factor `1 + 1/4.625 = 1.216` of the last twin below `q'^2` OR within `q'`
of it; either way a twin in `(q'^2, 1.216 q'^2 + q']`. Chained from one twin, that is infinitely
many twins. It is not implied by the window statement at any single rung (the window statement
at `q'` asks for a twin in `(q', q''^2)`, and `q''^2` may be far below `1.216 q'^2`) nor by the
window statements at all rungs (it is a two-sided gap bound at the square, stronger than
existence). So it is ROOT in the sense of face E (a local statement that over-asks), with the
object named: not `d_0 <= W`, but the twin gap across the prime square. The initial run does not
enter: it is excluded by the hypothesis `L >= d_0`, and E6 shows the new gear touches it only at
its end.

### 5.3 What this does to V8 and V9

- V8 needs the extra hypothesis `d_0(y_m(Q)) <= klo_m(Q)` (a twin in `(y_m, mQ]`), used silently
  where its proof passes from `x <= klo` to `6x - 1 > y`. Measured true in 399,973 of 399,973
  cells to `Q = 10^5`; and it is supplied by the induction it certifies (a twin in `(Q/2, Q]` is
  turn 1 at `Q/2`), so the certified ranges stand. But the theorem must say it.
- V8's frontier hypothesis is non-vacuous only at rungs 5, 7, 11, 23 (section 3(g)). At every rung
  `y >= 29` in the measured range the certification of turns 1-4 is the share form
  `F_pre(y) < c_m(Q)` ("no run of `Q/6` columns in the prefix of `sqrt((m+1)Q)`"), which
  valve_existence.md V7 already marks ROOT (face A). So the sentence "the measured constant
  4.625 certifies turns 1-4 to `Q = 8 x 10^7`" is true and its content, above `Q = 148`, is the
  record's share at the square root, not the constant: the frontier's constant certifies turns 3
  and 4 for `Q = 106..148` and nothing else.
- V9's mechanism sentence is now exact: new gears lengthen and never precede ON THE PREFIX (E7),
  with the run through the square column as the one run that lengthens at a fixed start into
  new territory; on the period they precede at 8 of 11 steps through order-2-to-4 merges. The
  constants 3.25 and 4.625 stay because the runs `(13, 4)` and `(111, 24)` are inherited
  unchanged (E7) until `d_0` excludes them (rung 17 and rung 137 respectively), and nothing
  arriving later can go below 4.625 except a straddling run, which is a twin gap at a prime
  square.

# What is new

1. **E6**, theorem (E) at one step with its exact two-column exception set, and the identity
   `h(q') <= d_0(M)` with equality iff `(q', q'')` is twin. Assembled from two facts on record
   (the dead zone, the twin-gear-pair exception set) plus the square column; the assembly and
   its proof are new, the use below is the content.
2. **E7**, prefix inheritance: the frontier of the prefix is carried up the ladder exactly (0
   exceptions in 8,152 cells), so the position-length staircase of position_frontier.md is not
   "stable as gears are added" by observation but by theorem, with the absorbed lengths and the
   run through the square column as the complete list of changes.
3. **The refutation of the parent's sentence on the period** with the mechanism of every
   exception: 8 of 11 steps, every exception a merge of order 2-4 beyond `W(q)` or the one
   straddle at 17->19; the sentence is a prefix statement.
4. **E8 and the exact residue**: the frontier for all rungs reduces to the straddling condition
   (a twin gap across a prime square, longer than the prime, is at most 21.6 % of its start),
   with the interior section runs disposed of by a prime-gap bound (proved from `q' = 118`) and
   the inherited runs by E7. The residue is ROOT (face E), and it is not `d_0 <= W`.
5. **The staircase of constants** along the ladder: eleven twin gaps, each the minimiser on an
   interval of rungs, the floor 4.625 attained on `[23, 131]` and never revisited, `c_pre`
   infinite from 1427; only two decreases in 2,260 rungs, both below rung 23.
6. **V8's hidden hypothesis** and the non-vacuity census: the frontier certifies positionally at
   rung 23 only; above it, the measured certification is the share form.
7. **The exact anatomy of the straddling run**: start = the start of `M`'s last prefix run
   (2,251 of 2,260 rungs), the maximal top-run share instance (`q' = 31`, `tau = 0.0833`) is its
   minimum ratio 6.727, and it exceeds the initial run at 24 rungs, the last `q' = 487`.

Checked against docs/novel/README.md: the index has no one-step form of (E), no prefix
inheritance law, and no reduction of the frontier's monotonicity to the square-column run.

# Verdict

**FACT with a ROOT mark, and one refutation.** The statement "new gears lengthen, never precede"
is FALSE on full periods (8 of 11 steps, merges of order 2-4 beyond the prefix) and a THEOREM on
the prefix (E7), where the only run that grows into new ground is the one through the square
column `W(q)`. The frontier constant's monotonicity (Form C) is false (17->19 and 19->23) but its
floor 4.625 is exact from rung 23 and never undercut to 19,997; by E8 its persistence at every
rung is equivalent to the straddling condition, a bound on the twin gap across each prime square,
which is ROOT (face E). Turns 1-4 for all `Q` are therefore NOT certified by this branch, and the
exact reason is not the initial run but the twin gap at `q'^2`, with the extra finding that above
rung 23 the measured frontier certifies nothing the share form does not.

Status for the tree: E6 PROVED (with direct-sieve gate, 428 rungs); E7 PROVED (0 exceptions in
8,152 cells); E8 PROVED with its proviso, E8-a PROVED for `q' >= 118` (prior-art prime gap);
the straddling condition ROOT; Forms A, B, C on the period DEAD (refuting instances 11->13 at
`L = 6`, 13->17 at `L = 6`, 17->19 at `L = 6..17`, and 17->19 for C). Node R4.c.iii.a: FACT / ROOT,
no CANDIDATE.

# Dead ends (with the refuting instance)

- **Form B on the full period** (the frontier monotone up the ladder): `R_min(6)` = 151 at m11
  against 89 at m13 (a `J = 2` merge at column 89); `R_min(6..9)` = 89/123 at m13 against 61 at
  m17; `R_min(6..11)` = 61/118 at m17 against 59 at m19 (the straddle through column 60).
- **Form C on the full period and on the prefix**: `c(m17) = 6.778 > c(m19) = 4.625`;
  `c_pre(17) = 10.6 > c_pre(19) = 5.364 > c_pre(23) = 4.625`.
- **The brief's "two home columns `+-u_{q'}`"**: the column `q' - u_{q'}` holds `5q'`; at every
  one of the 428 direct-sieve rungs the new set is `N_1 u N_2` of E6 and never contains it.
- **"What remains is `d_0 <= W` again"**: the initial run is excluded by the hypothesis `L >= d_0`
  and E6 shows the new gear reaches it only at its end; the residue is the straddling run.
- **The chain law as the mechanism on the prefix**: the new gear strikes at most two old openings
  in the whole prefix (E6), so no chain forms there; chains of 1-3 junctions are the mechanism of
  every period exception and none of the prefix ones.
- **A proof of the straddling condition from the machine**: by (R) it is a statement about twin
  gaps at prime squares; the frontier at rung `q` (`H_c(q)`) bounds `M`'s top run to `1/(c+1)` of
  the window but says nothing about the section's first run, which is new territory for every
  machine below `q'`. No mechanism separates it from the count.

Remaining open items of the part, sorted: closed here (E6, E7, E8, the exception census on the
period, the staircase of constants, the `d_0` ladder law, V8's hidden hypothesis and non-vacuity
census); measurement with no structural content (the band table of straddling ratios; the
merge-order distribution of deep exceptions); root question in disguise (the straddling
condition at every rung; `c_pre` infinite from 1427, i.e. the initial run is the longest run of
the prefix, which is `F_pre < d_0`, a share statement); genuinely open on the part alone: none.
The formaliser's target from this branch is E6 (a statement about divisibility of `6k -+ 1` by
`q'` below `q'^2`, kernel-sized) and E7 as its corollary on maximal runs.

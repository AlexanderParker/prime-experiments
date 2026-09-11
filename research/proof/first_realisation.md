# The first realisation: where the engine's long runs sit (proof skeleton step 8 as a position)

Theorist lane (Fable), 2026-09-11. Parent: `research/proof/proof_skeleton.md` section 13, which
states the one unproved statement (step 8) and says it is a statement about WHERE the engine's
long runs sit, not how long they are, and names the covering-problem decision of
`order_law_37_41.md` 2.2 as the one instrument that speaks about positions. Scripts in
`research/anchor235/r74/` (prefix `xm_`); outputs in `research/anchor235/r74/results/` (untracked).
Every number this document relies on is written into it. Nothing is committed.

Vocabulary used here, by construction (section 0.1): column, run, section, stretch. Not used:
window (except when quoting a document that uses it), rung, ladder, descent.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The objects, by construction

- **Column** `k` is the pair of numbers `(6k - 1, 6k + 1)`. Column 0 is `(-1, 1)`.
- **Gear** `g` (a prime `>= 5`) strikes column `k` iff `k = +-u_g (mod g)`, `u_g = 6^{-1} mod g`;
  equivalently iff `g` divides a member of the column. The gear's two **teeth** are the residue
  classes `u_g` and `-u_g`; their distance is `d_g = 2 u_g mod g`.
- **Engine** `{5..p}`: the gears that are the primes from 5 to `p`. Its **period** is
  `P_p = prod_{5 <= g <= p} g`; the struck/open pattern repeats with period `P_p` and is
  symmetric under `k -> -k` (the mirror), since each gear's teeth are `+-u_g`.
- A column is **open** under the engine if no gear strikes it; column 0 is open under every
  engine (`0 = +-u_g` is impossible).
- A **run** of length `l` at column `x` is: columns `x, x+1, ..., x+l-1` all struck. A run's
  length is the number of struck columns. The project's record `F(p)` is the largest DISTANCE
  between consecutive openings, so the longest run has length `F(p) - 1`.
- **`x_min(p, l)`** := the least `x >= 1` such that the engine `{5..p}` has a run of length `l`
  at `x`. It is non-decreasing in `l`, and `x_min(p, l) <= P_p / 2` for every realised `l` (the
  mirror sends a run at `[x, x+l-1]` to one at `[P_p - x - l + 1, P_p - x]`, and one of the two
  starts at or below `P_p / 2`).
- **The cut** `p` and the next prime `p'`. `a := (p^2 - 1)/6` is the column whose upper member
  is `p^2` (struck by `p`); `b := (p'^2 - 1)/6` the column whose upper member is `p'^2` (not
  struck by `p'`, which is not a gear). The **section** at the cut `p` is the columns whose
  members lie strictly between `p^2` and `p'^2`: columns `a+1 .. b-1`, of which there are
  `l_p - 1` with `l_p := (p'^2 - p^2)/6 = b - a`. (The proof skeleton and the brief write the
  section's length as `l_p`; the number of columns strictly inside is `l_p - 1`.)
- **Step 8 in position form.** Below `b` a column is open under `{5..p}` iff both its members
  are `p`-rough, and a number below `p'^2` that is `p`-rough is 1 or a prime (skeleton 3), so
  the section's open columns are exactly its twin prime pairs. Step 8 at the cut `p` says the
  section holds an open column, i.e. the columns `a+1 .. b-1` are not all struck. Since `a` is
  struck, that is: **no run of length `l_p` starts at column `a`.** A sufficient condition is
  `x_min(p, l_p) > a`, and the brief's form `x_min(p, l_p) > p'^2 / 6 = b + 1/6` is stronger
  still (it says the first run of the section's length anywhere is beyond the section).
- **The covering instrument** (`order_law_37_41.md` 2.2, `r70/ol_pattern.py`). Read the columns
  `x + o`, `o = 0..S`, in the offset coordinate. Gear `g` strikes offset `o` iff
  `o = +-u_g - x (mod g)`: two classes `{t_g, t_g + d_g}` mod `g` whose position
  `t_g = -u_g - x mod g` is a free phase, and by CRT the phase vector `(t_g)_g` runs over every
  combination exactly once as `x` runs over one period. A pattern (a set OPEN of offsets that
  must be open, its complement CLOSED in `[0, S]` that must be struck) is realised iff some
  phase vector strikes no offset of OPEN and every offset of CLOSED. That is the instrument;
  it decides existence over the whole period.
- **A cover of a run.** A run of length `l` at `x` is the pattern OPEN = {} , CLOSED = `[0, l-1]`.
  A **cover** is an assignment of each offset `o` in `[0, l-1]` to a gear `g(o)` and a tooth
  `s(o) in {+1, -1}` with `x + o = s(o) u_{g(o)} (mod g(o))`. Given the cover, every gear that
  is used has its residue fixed: `x = s u_g - o (mod g)` for any offset `o` assigned to it (the
  cover is consistent iff all offsets assigned to `g` agree on this residue), and every gear
  that is not used is free. So the columns realising a given cover form ONE residue class mod
  `M_C = prod_{g used} g`, and
  `x_min(p, l) = min over consistent covers C of [the least positive element of the class of C]`.
  This is what "the covering instrument computes" turned into a position: the instrument's YES
  is the non-emptiness of a union of residue classes mod `P_p`; `x_min` is the least element of
  that union.
- **Forced gears.** In a realised run, an offset struck by exactly one gear is a **sole strike**;
  the gear is **forced** at that run (its residue is fixed by that offset in every cover of the
  run). `U(p, l)` := the set of forced gears of the run at `x_min(p, l)`, `M_U := prod U`,
  `r_U := x_min mod M_U`. Every column `y = r_U (mod M_U)` has the forced gears striking the same
  offsets as at `x_min`; if `y < x_min` the run fails at `y`, and the reason is an offset that
  at `x_min` was struck only by non-forced gears and at `y` is struck by none.
- **Twin-gear coincidence columns** (`docs/novel/tooth-sharing-pinning.md`, cited not
  re-derived): for twin gears `(g, g+2)` the columns struck by both are the four classes mod
  `g(g+2)`: `(g+1)/6` (the column holding `g` and `g+2` themselves, `g = 5 mod 6`),
  `(g^2 + g - 1)/6`, `(g(g+2) + 1)/6` (the twin-product column) and `(5 g(g+2) - 1)/6`.

### 0.2 What the construction gives about positions, and what it does not (stated before measuring)

- Gives: `x_min(p, l)` is an exact, finite object: the least element of a union of residue
  classes mod `P_p`, one class per consistent cover. It is computable by a scan of the period
  in column order (the first run of length `l` met), or, for a pattern with few realisations,
  by enumerating the covers and taking the least element of each class.
- Gives: monotonicity in the gear set. If `M subset M'` then every run of `M` is contained in a
  run of `M'`, so `x_min(M', l) <= x_min(M, l)`. Adding gears moves every first realisation
  EARLIER or leaves it. Hence a LOWER bound on `x_min({5..p}, l)` cannot come from a smaller
  engine; it must come from the engine itself or from a larger one. The largest engine is all
  the primes, whose first run of length `l` below `p'^2 / 6` is the first twin gap of `l`
  columns: the root object.
- Gives: the note in the brief. For `x >= a = (p^2 - 1)/6` every gear `g <= p` has
  `x / g >= (p^2 - 1)/(6p) > p/6 - 1`, so every gear has struck at least `2 floor(x/g) - 1 >= p/3 - 3`
  columns below `x`: inside the section no gear is in the state "not yet engaged" that governs
  the initial run of the prefix (laws E6, E7 of `lengthen_never_precede.md`: the new gear's
  first strikes are its home columns at `(g -+ 1)/6`, and on the prefix new gears lengthen and
  never precede). So E6/E7-type arguments, which locate the FIRST strikes of each gear, say
  nothing about the section: at the section every gear's strikes are ordinary members of its
  two classes, with `x mod g` running through every residue as `x` runs through `l_p >= (2p+2)/3`
  consecutive columns whenever `g <= l_p`, and through at most `2` strikes for the gears
  `g > l_p` (at most one column per class). That last fact is the one structural constraint the
  section's position imposes: the gears above `l_p` are sparse in it, at most 2 strikes each.
- Does not give: any lower bound on the least element of a residue class from the class alone.
  A class mod `M_C` has its least element anywhere in `[0, M_C)`, and which residue the cover
  fixes is the arithmetic of `+-u_g - o`, which nothing in the covering condition constrains
  toward large values. So a lower bound on `x_min` must come from WHICH covers are consistent
  (the teeth), not from the cover count or the class count. The tooth families of
  `fusion_lemma.md` 3.4 (same gears, other teeth) are the counter-construction to test any
  claimed mechanism against.
- Stop rules. Capacity bounds on the cover (each gear strikes at most `2 ceil(l/g)` offsets,
  at most one per class when `g > l`) bound LENGTH and are the Jacobsthal-type argument: noted,
  not re-derived. The least element of a single CRT class is the classical CRT minimum: not
  re-derived. The first-hit count model (W32, `engine_laws_m37.md` U5) is a count and is used
  only as a comparison column.

### 0.3 Theory

**T.** `x_min(p, l)` carries no structural floor from the gears' arithmetic: it behaves as the
least element of a union of `S_p(l)` residue classes spread over the period, where
`S_p(l) = sum_{d >= l+1} (d - l) c_p(d)` is the number of columns per period at which a run of
length `>= l` starts (`c_p(d)` the gap census). Its size is set by `P_p / S_p(l)` up to a
first-hit factor of order one, and the section-form of step 8 holds because `S_p(l_p)` is small
relative to `P_p / b`, which is a fact about the census and the teeth, not about positions.
The first long runs are built on the forced gears' coincidences in the sense that the forced
set is nearly the whole engine at the record and shrinks as `l` falls; the least solution sits
where it does because earlier members of the forced class are blocked by the non-forced gears.

### 0.4 Predictions, each with the number that refutes it

- **P1 (gates).** The scan reproduces the certified corpus: `F(p) = 7, 11, 18, 25, 34, 43, 58, 88`
  at `p = 11, 13, 17, 19, 23, 29, 31, 37`; `x_min(23, 33) = 12,694,429` (the first record stretch of
  m23, `r34/results/q2_record_position.txt`: opening at 12,694,428, struck 12,694,429..12,694,461);
  the window records of `engine_laws_m37.md` 3.1: `x_min(23, 24) = 111` (gap 25 at column 110) and
  `x_min(47, 27) = 398` (gap 28 at column 397); the r71 gate scans' widest gaps in the first
  `1.5 x 10^8` columns: 63, 65, 65, 72 at m41, m43, m47, m53. REFUTED by any mismatch.
- **P2 (the pre-registered law, form X1: a first-hit floor).**
  `x_min(p, l) >= P_p / (4 S_p(l))` for every `p` in 11..37 and every `l` in `1..F(p)-1`.
  If it held with `S_p(l_p) <= P_p / (4 b)`, step 8 would follow; the second inequality is a
  census statement. Refuted by one cell with `x_min(p, l) S_p(l) / P_p < 1/4`. Honest
  expectation: REFUTED at a few cells (a first hit among `S` spread positions can be early;
  minimum of the ratio over ~250 cells expected between 0.02 and 0.1), median of the ratio near
  `ln 2 = 0.69`.
- **P3 (the law in `p` and `l` only, form X2: quadratic floor).** `x_min(p, l) >= (3/8) l^2`
  for all `l`. This is the weakest uniform-in-`l` floor that gives the brief's form of step 8 at
  a twin cut (`p' = p + 2`: `p'^2/6 = l_p p'/(2(p'-p)) . p'/(p'+p) < (3/8) l_p^2` because
  `l_p = (2p+2)/3 > 2p/3` gives `p'^2/6 < (p+2)^2/6 <= (3/8)(2p/3)^2 (1 + O(1/p))`... exact check in
  section 3). Pre-registered as REFUTED by the cell `l = 24`: the run of 24 struck columns at
  columns 111..134 (the twin gap `(659, 661)` to `(809, 811)`), `111 < (3/8) 576 = 216`, present in
  every engine with `p >= 23` (it lies below `29^2/6`). Kept to show why no floor uniform in `l`
  can be the route: the twin gaps of the prefix are the engine's own early runs.
- **P4 (step 8 in position form, the brief's target).** `x_min(p, l_p) > p'^2/6` at every cut
  `p = 11..53`. Prediction on the size: the ratio `x_min(p, l_p) / (p'^2/6)` is NOT the frontier
  ratio 4.6 of `frontier_floor_1e7.md` (that is `x_s / L_s`, the straddling run's start over its
  length, a different object); at cuts where `l_p` is short (`p = 11, 17, 29, 41`) the first run
  of the section's length sits within a few multiples of `b`; at the long-section cuts
  (`p = 47, 53`, `l_p = 100, 112`) the first such run sits beyond `10^9` columns. Refuted by one
  cut with `x_min(p, l_p) <= p'^2/6`, which would be a section with no twin.
- **P5 (mechanism: the forced set and the class).** At the record (`l = F(p) - 1`) every gear is
  forced (`U = {5..p}`) at m11..m37 (compare law L4 of `pair_statement.md`, stated there for
  above-record stretches); at `l = l_p` the forced set is a proper subset at every `p >= 19`.
  And the least solution is NOT the least element of its forced class in most cells:
  `x_min(p, l) > M_U` (so earlier members of the class were blocked) at more than half the cells
  with `l >= 10`. Refuted by `U != {5..p}` at a record, or by `x_min < M_U` at more than half of
  those cells.
- **P6 (the brief's Q3: twin-gear coincidences).** The first run of length `l >= l_p` is not
  built on the twin-gear coincidence columns: the fraction of cells `(p, l >= 10)` whose run
  contains a coincidence column of some twin-gear pair `(g, g+2)` with `g >= 11` is below 0.5
  (the `(5, 7)` pair's four classes mod 35 are met by every run of `>= 24` columns, so `(5, 7)`
  is excluded from the count as trivial). Refuted by a fraction at or above 0.5.
- **P7 (the counter-construction).** On the tooth families of `fusion_lemma.md` 3.4 (gears
  `{5..p}` with teeth `+-v_g`, `1 <= v_g <= (g-1)/2`), the section at the cut `p` is fully struck
  for some member at `p = 13` already (`l_p = 20`, columns 29..47), and the share of members
  killing the section is between 1% and 20% at `p = 13, 17, 19`. Refuted by 0 killers at all
  three cuts (then the two-tooth construction alone forbids a struck section at small `p`, which
  would be a mechanism to hunt).

### 0.5 Scorecard

Filled in section 6.

---

## 1. The construction, and what it decides (PROVED, from the definitions)

Everything in this section is built from 0.1 and the proof skeleton's steps 2, 3, 5; no
measurement is used.

**C1 (the first realisation is a least element).** Fix the engine `{5..p}` and a pattern
(OPEN, CLOSED on `[0, S]`). Let `V` be the set of phase vectors `(x mod g)_g` that realise it (the
instrument's object). By CRT each vector is one residue class mod `P_p`, so the set of realising
columns is a union of `|V|` classes, and the first realisation is the least positive element of
that union. For a run of length `l` (OPEN empty) every cover `C` fixes the residues of its used
gears and leaves the others free, so the realising columns of `C` are one class mod
`M_C = prod(used)`, and `x_min(p, l) = min_C (least element of C's class)`. Two ways to compute
it, both exact: read the columns in order and stop at the first run of length `l` (the scan);
or enumerate the covers and walk each class upward to its least element (the CRT-minimum
instrument of `xm_words.py`). They agree at every gate (section 2).

**C2 (monotone in the gear set).** `M subset M'` implies every run of `M` is a run of `M'`, so
`x_min(M', l) <= x_min(M, l)` for every `l`. A lower bound for `{5..p}` therefore never comes
from a smaller engine.

**C3 (below `b` the engine's runs are twin gaps).** For `x + l - 1 < b = (p'^2 - 1)/6` a run of
length `l` at `x` under `{5..p}` is `l` consecutive columns with no twin prime pair (skeleton 3
and 5: below `p'^2` a column both of whose members are `p`-rough has both members prime). So
the part of the staircase `x_min(p, .)` that lies below `b` is the first-occurrence sequence of
twin-free stretches, the same for every engine whose `b` is above it. Measured (section 3.1):
`x_min = 1, 53, 59, 111` for `l` up to `4, 5, 11, 24` at every engine from m19 to m53.

**C4 (step 8 in position form, exactly).** Let `L_a(p)` be the length of the run at the square
column `a` measured from `a` (the columns `a, a+1, ...` up to the first open column). Then

    step 8 at p  <=>  L_a(p) < l_p  <=>  the first run of length l_p at or after a does not start at a.

Proof: `a` is struck by `p`; the section is struck entirely iff `a .. b-1` are all struck iff
`L_a(p) >= b - a = l_p`. And `L_a(p)` is the column distance from `a` to the first open column
above `a`, i.e. to the first twin pair above `p^2` (by C3 the first open column above `a` and
below `b` is a twin; if there were none below `b` then `L_a >= l_p`). So `L_a(p) = L_1(p)`, the
first-twin offset above `p^2` in columns of `frontier_floor_1e7.md` section 1, and step 8 in
position form is the statement `L_1(p) < l_p`. **Stop line:** that object is measured at every
prime cut to `10^7` with 0 exceptions and the stronger bound `L_1 < (2p+1)/3 <= l_p - 1/3`
(`frontier_floor_1e7.md` section 5, the r3 first-twin scan, node R4.d.i.a's arc bound); the
position form of step 8 is R4.d.i.a's measured object in other words, and it is not re-derived
here. What this branch adds is the quantifier on the other side: the brief's
`x_min(p, l_p) > b` is stronger than step 8 and is FALSE at `p = 29` (section 3.2), so the
first-anywhere quantifier is the wrong one; the run at the square column is the object.

**C5 (the least element of a forced class).** At any realised run, the residues of the forced
gears are fixed, so the run's column is at least the least element of its forced class
`r_U mod M_U`. This is the only inequality on positions the construction offers, and it is
empty whenever `M_U > x`: measured, `x_min(p, l)` IS the least element of its forced class at
82 of 82 rows with at least two forced gears (section 4), which says only that `M_U` exceeds
`x_min` there.

## 2. Setup, instruments, gates

| script | what | gate |
|---|---|---|
| `xm_scan.py p` | segmented sieve of `{5..p}` in column order (tiled pattern of the gears `<= 13`, strided writes for the rest; 7-8 workers; chunks of `2^25`, each seen `1,024` columns past its end so a run is read whole by the chunk it starts in); returns the running-record staircase `(x_i, r_i)` (which is `x_min(p, .)`) and the histogram of run lengths; stops at `P/2` (the mirror) or a column limit; `--from a` starts at a column with the column before it treated as open, so the run through `a` is measured from `a` | `F = 7, 11, 18, 25, 34, 43, 58, 88` at m11..m37 (8 of 8); `x_min(23, 33) = 12,694,429` (r34's first record stretch, exact); `x_min(23, 24) = 111` and `x_min(47, 27) = 398` (the window records of `engine_laws_m37.md` 3.1, exact); m41/m43/m47/m53 in the first `1.5 x 10^8` columns: widest runs 62, 64, 64, 71, i.e. gaps 63, 65, 65, 72 as r71's gate scans (4 of 4) |
| `xm_words.py p --open 0,S` or `--word g1,g2,..` | the CRT-minimum instrument: exact-cover search over (gear, residue) with the r70 capacity bound, every leaf walked upward to the least column that also satisfies the free gears, bounded by the best so far; every answer re-verified by re-sieving the pattern at the returned column against every gear | the openings before the first record runs: m23 `12,694,428`, m29 `200,906,185`, m31 `1,468,940,242` (3 of 3, against the scan); m37 `90,816,580,902` against the scan's `90,816,580,903` (second measurement of a new number) |
| `xm_mech.py` | per staircase row: kill map, used and forced gears, `M_U`, `r_U`, `k*`, the first failing earlier class members, twin-gear coincidence columns | consistency with the scan's runs (every kill map has no open column) |
| `xm_family.py`, `xm_family_check.py`, `xm_family_cover.py` | the tooth family at the cut `p`: exhaustive (bitmasks, to `p = 31`), an independent plain-loop recount, and an exact-cover search over the teeth for the cuts too large to enumerate (37, 41) | the first two agree: 15 of 1,440 at `p = 17`, 6,030 of 1,995,840 at `p = 29` (second measurement); the search finds killers exactly at the enumerated cuts that have them (17, 29) and none at the enumerated cuts that do not, and every exhibited killer is re-verified column by column |
| `xm_table.py` | the tables below, `S_p(l)` from the scan's own histogram on the scanned range | `S(33)/X` at m23 gives 2 record runs per half period, i.e. r34's 4 per period |

Cost: m11..m31 full half periods in 45 s (m31: `1.67 x 10^10` columns in 42 s on 8 workers);
m37 to its record 234 s (7 workers, `9.08 x 10^10` columns); m41 and m43 prefixes of `2 x 10^11`
columns 572 s each; m47, m53 prefixes of `4 x 10^11`; the record words by the CRT instrument
21 s (m37), 455 s (m41), incomplete at m43 (1,800 s), m47 and m53 (see 3.3a); the m41
confirmation scan 1,608 s. Peak memory under 1 GB; at most 8 processes at any time.

## 3. Results (exact)

### 3.1 The staircase `x_min(p, l)` at every engine with a period on record, and the prefixes beyond

One row per record-breaking run; `l..` is the range of run lengths whose first realisation is
that row's `x`. `x/P` is the position as a fraction of the period. (Second convention: the
project's gap is the run length plus one.)

| engine | `P` | rows: `l -> x_min` |
|---|---|---|
| m11 | 385 | 1-2 -> 1; 3-4 -> 13; 5 -> 53; 6 -> 151 (0.392 P) |
| m13 | 5,005 | 1-2 -> 1; 3-4 -> 13; 5 -> 53; 6 -> 89; 7-10 -> 123 (0.0246 P) |
| m17 | 85,085 | 1-4 -> 1; 5 -> 53; 6-9 -> 61; 10-17 -> 118 (0.00139 P) |
| m19 | 1,616,615 | 1-4 -> 1; 5 -> 53; 6-11 -> 59; 12-24 -> 111 (0.0000687 P) |
| m23 | 37,182,145 | 1-4 -> 1; 5 -> 53; 6-11 -> 59; 12-24 -> 111; 25 -> 40,148; 26 -> 170,034; 27 -> 190,056; 28-29 -> 396,199; 30 -> 1,479,278; 31 -> 2,553,844; 32 -> 5,606,403; 33 -> 12,694,429 (0.3414 P) |
| m29 | 1,078,282,205 | 1-6 -> 1; 7-11 -> 59; 12-24 -> 111; 25 -> 5,643; 26-27 -> 23,278; 28-29 -> 35,564; 30-32 -> 102,273; 33-34 -> 2,278,076; 35-36 -> 2,900,801; 37 -> 6,603,768; 38-39 -> 144,154,491; 40-42 -> 200,906,186 (0.1863 P) |
| m31 | 33,426,748,355 | 1-6 -> 1; 7-11 -> 59; 12-24 -> 111; 25-29 -> 5,643; 30-31 -> 6,026; 32 -> 16,351; 33 -> 35,564; 34-37 -> 102,273; 38-41 -> 254,736; 42-44 -> 14,995,459; 45-49 -> 113,629,016; 50 -> 558,590,208; 51 -> 672,200,331; 52-54 -> 944,791,621; 55-57 -> 1,468,940,243 (0.0439 P) |
| m37 | 1,236,789,689,135 | 1-6 -> 1; 7-11 -> 59; 12-24 -> 111; 25 -> 3,648; 26-29 -> 5,643; 30-33 -> 6,024; 34-36 -> 35,561; 37 -> 102,273; 38-41 -> 254,736; 42 -> 1,840,038; 43-44 -> 2,009,176; 45 -> 9,728,088; 46-48 -> 15,526,414; 49 -> 26,759,969; 50-62 -> 27,819,088; 63-67 -> 13,134,580,831; 68-69 -> 54,795,018,338; 70-87 -> 90,816,580,903 (0.0734 P) |
| m41 (prefix `2 x 10^11` = 0.0039 P; the record by the CRT instrument) | 50,708,377,254,535 | 1-9 -> 1; 10-11 -> 59; 12-24 -> 111; 25 -> 398; 26-29 -> 5,643; 30-33 -> 6,024; 34-44 -> 24,174; 45-50 -> 582,938; 51-62 -> 27,819,088; 63-64 -> 976,255,606; 65-66 -> 2,082,536,749; 67 -> 6,401,563,001; 68-69 -> 8,912,208,171; 70-71 -> 11,268,699,504; 72 -> 13,931,061,513; 73-89 -> 16,365,163,681 (0.000323 P); 90 -> 630,700,131,373 (0.01244 P, CRT instrument; scan confirmation in 3.3) |
| m43 (prefix `2 x 10^11` = 0.0001 P) | 2,180,460,221,945,005 | 1-9 -> 1; 10-11 -> 59; 12-24 -> 111; 25-27 -> 398; 28-29 -> 5,643; 30-33 -> 6,024; 34-49 -> 24,169; 50-52 -> 42,818; 53-54 -> 10,405,231; 55-60 -> 26,759,958; 61-62 -> 27,819,088; 63-64 -> 53,194,054; 65-69 -> 222,056,286; 70-71 -> 3,445,492,376; 72-76 -> 6,046,981,091; 77-82 -> 7,931,283,226; 83-89 -> 16,365,163,681 (the m41 run of 89, unchanged by 43); 90-102: beyond `2 x 10^11` (the record 102 by the CRT instrument, 3.3a) |
| m47 (prefix `4 x 10^11`) | 102,481,630,431,415,235 | 12-24 -> 111; 25-27 -> 398; 28-29 -> 1,863; 30-33 -> 6,024; 34-49 -> 24,169; 50-52 -> 42,818; 53-56 -> 8,392,991; 57-60 -> 26,759,958; 61-62 -> 27,819,088; 63-64 -> 53,194,054; 65-69 -> 222,056,286; ...; 82 -> 7,931,283,226; 83-84 -> 9,059,823,188; 85 -> 11,980,733,818; 86-94 -> 16,365,163,681 (the same column as m41's 89 and m43's 89, lengthened by 47 to 94); 95-117: beyond `4 x 10^11` |
| m53 (prefix `4 x 10^11`) | 5,431,526,412,865,007,455 | 12-24 -> 111; 25-27 -> 398; 28-32 -> 981; 33 -> 6,024; 34-49 -> 24,169; 50-52 -> 42,818; 53-54 -> 6,056,971; 55 -> 8,083,123; 56 -> 8,392,991; 57-60 -> 9,605,858; 61-67 -> 10,580,013; ...; 82-84 -> 6,046,981,083; 85 -> 11,980,733,818; 86-94 -> 16,365,163,681; 95-99 -> 64,872,198,296; 100-144: beyond `4 x 10^11` |

Three readings of the table, each a fact of the rows:

1. **The prefix rows are shared.** `x_min = 1` up to the initial run of home strikes (length
   `2, 2, 4, 4, 4, 6, 6, 6, 9` at m11..m41: the columns `1..` holding `5, 7, 11, 13, ...` themselves,
   E7's initial run), then `53` (run 5), `59` (run 6..11), `111` (run 12..24) at every engine from
   m19 on: these are the twin gaps `(311,313)..(347,349)`, `(347,349)..(419,421)`,
   `(659,661)..(809,811)`, by C3. The engine's own runs begin only above `b`.
2. **The rows are nested across engines.** `5,643` (run 25 at m29, 29 at m31 and m37),
   `6,024/6,026`, `35,561/35,564`, `102,273` (32, 37, 37), `254,736` (41, 41), `27,819,088` (62 at
   m37 and m41), `16,365,163,681` (89 at m41 and m43): a first realisation of the smaller engine
   is lengthened in place by the added gears (C2 and E7's "lengthen"), and the added gear also
   creates earlier first realisations at some levels (`x_min(31, 30) = 6,026 < x_min(29, 30) =
   102,273`), which is the full-period failure of "never precede" that r69 recorded.
3. **The steps are erratic.** m37: run 49 at `26,759,969`, run 62 at `27,819,088` (thirteen levels
   in one step, `1.04 x` the previous position), then run 63 only at `13,134,580,831` (`472 x`
   later). m41: run 72 at `1.39 x 10^10`, run 89 at `1.64 x 10^10` (seventeen levels), run 90 at
   `6.31 x 10^11` (`38.5 x` later). The staircase is a sequence of outliers, not a smooth curve.

### 3.2 The section rows: the run at the square column, and the first run of the section's length

`L_a` = the run at `a` measured from `a` (the from-`a` scan; confirmed independently by
primality: the first twin above `p^2` sits at column `a + L_a` at all 12 cuts, 12 of 12).
`x^{>=a}` = the first run of length `l_p` starting at or after `a`. `x_min` = the first anywhere.

| `p` | `p'` | `a` | `b` | `l_p` | `L_a` | `L_a / l_p` | twins in the section | `x^{>=a}(p, l_p)` | `x_min(p, l_p)` | `F(p) - 1` |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 13 | 20 | 28 | 8 | 3 | 0.375 | 2 | none (max run 6) | none | 6 |
| 13 | 17 | 28 | 48 | 20 | 2 | 0.100 | 7 | none (max run 10) | none | 10 |
| 17 | 19 | 48 | 60 | 12 | 4 | 0.333 | 2 | **118** (`= b + 58`, `5.8 l_p` past `a`) | 118 | 17 |
| 19 | 23 | 60 | 88 | 28 | 10 | 0.357 | 4 | none (max run 24) | none | 24 |
| 23 | 29 | 88 | 140 | 52 | 7 | 0.135 | 8 | none (max run 33) | none | 33 |
| 29 | 31 | 140 | 160 | 20 | 3 | 0.150 | 2 | **148**, INSIDE the section (`a + 8`, run 22 to column 169) | **111 < a** | 42 |
| 31 | 37 | 160 | 228 | 68 | 10 | 0.147 | 11 | none (max run 57) | none | 57 |
| 37 | 41 | 228 | 280 | 52 | 10 | 0.192 | 7 | 27,819,088 (`99,354 b`) | 27,819,088 | 87 |
| 41 | 43 | 280 | 308 | 28 | 3 | 0.107 | 3 | **5,643** (`18.3 b`) | 5,643 | 90 |
| 43 | 47 | 308 | 368 | 60 | 4 | 0.067 | 11 | 26,759,958 (`72,717 b`) | 26,759,958 | 102 |
| 47 | 53 | 368 | 468 | 100 | 5 | 0.050 | 13 | `> 4 x 10^11` (`> 8.5 x 10^8 b`; run 94 at `1.64 x 10^10` is the largest below) | same | 117 |
| 53 | 59 | 468 | 580 | 112 | 27 | 0.241 | 13 | `> 4 x 10^11` (`> 6.9 x 10^8 b`; run 99 at `6.49 x 10^10` the largest below) | same | 144 |

- **P4, the brief's form `x_min(p, l_p) > p'^2/6`: REFUTED at `p = 29`** and vacuous (no run of the
  section's length exists) at `p = 11, 13, 19, 23, 31`. At `p = 29` the engine `{5..29}` has its
  first run of length 20 at column 111 (it is 24 long: the twin gap 659..811), below the square
  column 140; and its first run of length 20 AT OR AFTER the square column starts at 148, eight
  columns into the twenty-column section, and runs to 169 past the section's end at 160. Step 8
  holds there because the run at `a = 140` is 3 long (the twin `(857, 859)` at column 143). So
  the first-anywhere quantifier says nothing at this cut, and the at-or-after quantifier is
  tight to 8 columns; what carries step 8 is `L_a = 3 < 20` (C4).
- Where the brief's form holds it holds by large factors that are not the frontier's 4.6: `118 / 60
  = 2.0` at 17, `5,643 / 308 = 18.3` at 41, `9.9 x 10^4` at 37, `7.3 x 10^4` at 43, above `8.5 x 10^8`
  at 47 and `6.9 x 10^8` at 53 (P4's size prediction "beyond `10^9` columns" at 47 and 53 is not
  decided: the scans stop at `4 x 10^11` with runs of 94 and 99 as the largest seen). The quantity
  `x_min(p, l_p) / b` is not a ratio with a floor; it is whichever outlier of the staircase first
  reaches `l_p`.
- `L_a / l_p` (the measured content of step 8 in position form) is at most 0.375 (`p = 11`) on
  these twelve cuts, 0.241 at `p = 53` (the run 468..494, 27 columns, ending at the twin
  `(2969, 2971)`), and it is `L_1 / l_p` of the frontier scan, whose maximum to `10^7` is
  `0.7714 x (2p+1)/(3 l_p) <= 0.7714` at `p = 53`.

### 3.3 The records and the words: first realisations

| engine | `F` | run | first realisation (run start) | fraction of `P` | methods | runs of this length per period |
|---|---|---|---|---|---|---|
| m11 | 7 | 6 | 151 | 0.392 | scan; r34 (0.3922) | 2 |
| m13 | 11 | 10 | 123 | 0.0246 | scan; r34 (0.0246) | 2 (mirror pair 123 / 4,873) |
| m17 | 18 | 17 | 118 | 0.00139 | scan | 20 (r34) |
| m19 | 25 | 24 | 111 | 0.0000687 | scan | 20 (r34) |
| m23 | 34 | 33 | 12,694,429 | 0.3414 | scan; CRT instrument; r34 (0.3414) | 4 |
| m29 | 43 | 42 | 200,906,186 | 0.1863 | scan; CRT instrument | 2 |
| m31 | 58 | 57 | 1,468,940,243 | 0.0439 | scan; CRT instrument | 4 |
| m37 | 88 | 87 | **90,816,580,903** | 0.0734 | scan; CRT instrument (agree) | 1 in the first 0.0734 P |
| m41 | 91 | 90 | **630,700,131,373** | 0.01244 | CRT instrument (455 s, 17 leaves, verified); scan resumed from `2 x 10^11` finds the 90-run first at the same column (1,608 s): two methods agree | 1 in the first 0.0124 P |
| m43 | 103 | 102 | `<= 233,885,349,904,191` (a verified realisation; the search was stopped at its 1,800 s limit before certifying it as the first) | 0.107 | CRT instrument, incomplete | |
| m47 | 118 | 117 | not decided within 2,400 s | | CRT instrument, running when this was written (results append to `results/words_records.log`) | |
| m53 | 145 | 144 | not decided within 3,000 s | | same | |

The r34 certificates stop at m23; at m29 and m31 the scan and the CRT instrument are each
other's second measurement (exact agreement at both). The record positions at m37 and m41 are
new numbers.

**The maximal words of m37 with respect to 41** (`fusion_lemma.md` 3.2) and the record fusion,
first realisations in m37 (the opening at the word's first gap; CRT instrument, each verified):

| word | span | first realisation | fraction of `P(37)` | leaves | note |
|---|---|---|---|---|---|
| `(14, 41)` | 55 | 109,580,398 | 0.0000886 | 4,542 | gear 23 free at the least solution |
| `(41, 14)` | 55 | 320,904,217 | 0.000259 | 5,052 | |
| `(27, 41)` | 68 | 816,663,702,755 | 0.6603 | 2 | its mirror is the row below: `x + x' = P - 68` |
| `(41, 27)` | 68 | 420,125,986,312 | 0.3397 | 1 | |
| `(21, 14, 41, 15)` (the record fusion of 41) | 91 | 175,901,839,712 | 0.1422 | 2 | |
| `(21, 14, 41, 22)` (the relaxed word) | 98 | not realised | | 0 | as r70 found |

So the record run of m41 (first at `630,700,131,373`) is not the first realisation of the
fusing 4-word in m37 with the gear 41 added at the right phase (that would be at or after
`175,901,839,712`; the gear 41 has one admissible residue in 41, and the first admissible
realisation among the word's copies is `3.6 x` later).

### 3.3a The m41 confirmation and the reach of the CRT instrument

The scan of m41 resumed from column `2 x 10^11` (7 workers, 1,608 s, `4.3 x 10^11` further
columns) meets its first run of 90 at `630,700,131,373`, the column the CRT instrument returned
455 s earlier from 17 leaves: the new record position is measured twice by independent
methods. At m43 the instrument's exact-cover search over twelve gears did not complete in
1,800 s; its best leaf, `233,885,349,904,190` (the opening; run at `+1`), is a verified
realisation of the 102-run and an upper bound on `x_min(43, 102)`, not its value. The
instrument's reach for record patterns is therefore m41 in minutes and m43 in more than half
an hour; for the section rows it is not needed (3.2 uses the scans).

### 3.4 The two pre-registered laws

**X1 (first-hit floor `x_min . rho >= 1/4`, `rho = S_p(l) / X` the density of run starts of length
`>= l` on the scanned range `X`): REFUTED.**

| engine | cells | min `x_min . rho` (at `l`, `x`) | median | cells below 1/4 |
|---|---|---|---|---|
| m11 | 6 | 0.353 (2, 1) | 1.45 | 0 |
| m13 | 10 | 0.279 (10, 123) | 1.87 | 0 |
| m17 | 17 | **0.027** (17, 118) | 0.52 | 5 |
| m19 | 24 | **0.0014** (24, 111) | 0.48 | 8 |
| m23 | 33 | 0.014 (24, 111) | 1.51 | 6 |
| m29 | 42 | 0.044 (24, 111) | 1.03 | 5 |
| m31 | 57 | 0.083 (41, 254,736) | 0.86 | 10 |
| m37 | 87 | 0.050 (62, 27,819,088) | 2.40 | 7 |
| m41 (prefix) | 89 | 0.077 (44, 24,174) | 1.16 | 10 |
| m43 (prefix) | 89 | 0.043 (52, 42,818) | 1.43 | 14 |
| m47 (prefix) | 94 | 0.041 (94, 16,365,163,681) | 1.65 | 12 |
| m53 (prefix) | 99 | 0.162 (99, 64,872,198,296) | 2.11 | 3 |

The record of `{5..19}` (run 24) is realised at column 111, `700 x` before its mean spacing over
the period (20 copies in `1,616,615` columns); the record of `{5..17}` at 118, `37 x` early. The
same column 111 is the earliest cell at m23 and m29. So the first-hit floor is not a floor: the
prefix's twin gap `(659..811)` is the engine's own early run, at every engine that contains it.
The medians (0.5 to 2.4) are where a first hit among spread positions sits; the minima are the
prefix's outliers.

**X2 (quadratic floor `x_min >= (3/8) l^2`): REFUTED** as pre-registered, at `l = 24` (`111 / 216 =
0.514`) for `p = 19..31`, and at `l = 4..9` for every engine (the home-strike run at column 1:
`x / (3/8 l^2) = 0.167` at `l = 4`, `0.074` at `l = 6`, `0.033` at `l = 9`). Restricted to
`l >= (2p+2)/3`: 3.28 (m13), 1.09 (m17), 0.514 (m19, m23, m29, m31, all at `l = 24`), 14.75
(m37 and m41, at `l = 33`, `x = 6,024`). The exact form of the check promised in P3: at a twin cut
`p' = p + 2`, `b = (p^2 + 4p + 3)/6` and `l_p = (4p + 4)/6`, so `(3/8) l_p^2 = (p + 1)^2 / 6 >= b - 1/3`;
hence `x_min(p, l_p) >= (3/8) l_p^2` would give `x_min(p, l_p) > b - 1` at every twin cut, and the
law fails at `l = 24`.

### 3.5 The counter-construction: the tooth family

Every gear `g` of `{5..p}` with teeth `+-v_g`, `1 <= v_g <= (g-1)/2` (the real engine is
`v_g = min(u_g, g - u_g)`); does the member strike every column of the section `a+1 .. b-1`?

| cut `p -> p'` | section columns | family | killers | share | first killer's teeth `(v_5, v_7, ...)` | real teeth |
|---|---|---|---|---|---|---|
| 7 -> 11 | 11 (9..19) | 6 | 0 | 0 | | (1, 1) |
| 11 -> 13 | 7 (21..27) | 30 | 0 | 0 | | (1, 1, 2) |
| 13 -> 17 | 19 (29..47) | 180 | 0 | 0 | | (1, 1, 2, 2) |
| 17 -> 19 | 11 (49..59) | 1,440 | **15** | 1.042 % | **(1, 1, 2, 6, 1)** | (1, 1, 2, 2, 3) |
| 19 -> 23 | 27 (61..87) | 12,960 | 0 | 0 | | (1, 1, 2, 2, 3, 3) |
| 23 -> 29 | 51 (89..139) | 142,560 | 0 | 0 | | (1, 1, 2, 2, 3, 3, 4) |
| 29 -> 31 | 19 (141..159) | 1,995,840 | **6,030** | 0.302 % | **(1, 1, 1, 2, 1, 2, 4, 2)** | (1, 1, 2, 2, 3, 3, 4, 5) |
| 31 -> 37 | 67 (161..227) | 29,937,600 | 0 | 0 | | (1, 1, 2, 2, 3, 3, 4, 5, 5) |
| 37 -> 41 | 51 (229..279) | 538,876,800 (search, not enumerated) | **>= 1** | | **(2, 1, 2, 2, 4, 3, 3, 9, 13, 5)**, verified by direct sieve | (1, 1, 2, 2, 3, 3, 4, 5, 5, 6) |
| 41 -> 43 | 27 (281..307) | 1.6 x 10^10 (search) | **>= 1** | | **(1, 2, 1, 4, 6, 3, 6, 7, 8, 1, 1)**, verified by direct sieve | (1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7) |
| 43 -> 47 | 59 (309..367) | 3.4 x 10^11 (search) | **>= 1** | | **(1, 2, 4, 3, 7, 5, 3, 5, 9, 12, 12, 16)**, verified | (1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7) |
| 47 -> 53 | 99 (369..467) | 7.7 x 10^12 (search, 17 s) | **>= 1** | | **(1, 3, 4, 5, 6, 3, 2, 7, 2, 17, 8, 20, 3)**, verified | (1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8) |
| 53 -> 59 | 111 (469..579) | 2.0 x 10^14 (search, 13 s) | **>= 1** | | **(2, 1, 4, 5, 7, 4, 2, 8, 2, 7, 4, 10, 22, 14)**, verified | (1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9) |

Second measurement: a plain per-member loop (no bitmasks) gives 15 and 6,030 at 17 and 29; the
exact-cover search over the teeth (`xm_family_cover.py`) finds killers at 17, 29, 37, 41 in
under a second each and its example members are re-verified column by column. The first
killer at 17 strikes the section as `5 | 7+17 | 5 | 17 | 11 | 5 | 7 | 5 | 7+11 | 13 | 5+13`
(columns 49..59, gear per column), and its run at the square column 48 (struck by 7) is at
least 12 = `l_p`: in this member the least solution for the section's length sits AT the
section. The killer at 37 strikes the 51 columns 229..279 as
`11 | 7+31 | 19 | 5+7+13 | 5+11+23 | 17 | 31 | 13 | 5+7 | 5 | 7 | 11 | 29 | 5+17 | 5 | 7+11+19 | 13 | 7 | 5 | 5 | 13 | 19+23 | 7+11+17 | 5+29 | 5+7 | 37 | 11 | 23 | 5 | 5+7+13 | 17 | 7 | 31 | 5+11+13 | 5+19 | 37 | 7 | 11+31 | 5+7 | 5+17 | 19 | 29 | 13 | 5+7 | 5+11+23 | 7 | 13 | 17 | 5+11 | 5 | 7+23`.
**P7 is refuted on the cut (no killer at 13; the smallest cut with one is 17) and confirmed on
the point: the two-tooth covering structure with the real gears kills the section at
`p = 17, 29` and at every cut from 37 to 53, and at none of `p = 7, 11, 13, 19, 23, 31`.** The
pattern is not "short sections only": 111 columns are killed with fourteen gears at 53 while 19
columns are not killed with four gears at 13 and 67 columns are not with nine at 31; what
decides is the gears' capacity against the section (a member exists iff the exact-cover
problem over the teeth is satisfiable, decided in under 20 s at every cut here), and from 37
on it is always satisfiable: past 31 the section is never protected by the two-tooth structure
alone.

## 4. Mechanism

### 4.1 What a first realisation looks like

At every row with run length `>= 11` every gear of the engine is used (m23..m41, 78 rows, 0
exceptions), the columns are struck `1.29` to `1.74` times on average (the run is nearly a
tiling), and the forced set is:

| engine | rows `r >= 10` | forced = all gears | forced = all but one | forced at the record |
|---|---|---|---|---|
| m17 | 1 | 1 | 0 | all 5 |
| m19 | 2 | 2 | 0 | all 6 |
| m23 | 10 | 8 | 1 (`r = 24`: 6 of 7) | all 7 |
| m29 | 11 | 7 | 2 | all 8 |
| m31 | 14 | 9 | 2 | all 9 |
| m37 | 17 | 8 | 6 | all 10 |
| m41 | 15 | 8 | 5 | all 11 |

**P5, first half: CONFIRMED, 7 of 7 engines have every gear forced at the record** (m11 and
m13 too: 3 of 3, 4 of 4). Second half: **REFUTED in the opposite direction** — at 82 of 82 rows
with at least two forced gears `x_min` is the least element of its forced class (`k* = 0`); the
only rows with earlier class members are the initial home-strike runs (`r = 2..9` at `x = 1`,
zero or one forced gear). By C5 that is `M_U > x_min` at every such row (`M_U / x_min >= 4.5 x 10^2`
at m31's `r = 32`, the tightest), so the forced class contributes no information about the
position beyond the residue vector itself.

Three kill maps (gear per column, `+` = struck by both):

- m29, the section run at 148 (22 columns, forced `{5, 7, 11, 13, 19, 23}`, 17 and 29 not forced):
  `7 | 5+19 | 17+29 | 5 | 11 | 7 | 5+13 | 7+19 | 5+11+17 | 23 | 13 | 5 | 7 | 5 | 7 | 11 | 5 | 23 | 5 | 7+11+13+17 | 19 | 5+7+29`.
- m19, the record at 111 (24 columns, all forced):
  `5+7+19 | 11 | 7 | 5 | 13 | 5+17 | 19 | 7 | 5+11+13 | 7 | 5 | 17 | 11 | 5 | 7 | 5 | 7 | 13 | 5 | 11+19 | 5 | 7+13 | 17 | 5+7+11`.
- m37, the outlier run of 62 at 27,819,088 (all ten forced, mean multiplicity 1.516):
  `13 | 5 | 11 | 5+7 | 13+17 | 7 | 5+19+23 | 31 | 5+29 | 11 | 7 | 5 | 7+19 | 5+11+13 | 23 | 17 | 5 | 7+13 | 5 | 7 | 11 | 5+17 | 37 | 5 | 7+11 | 19 | 5+7+13 | 29 | 5+31 | 23 | 13 | 5+7+11+19 | 17 | 5+7 | 37 | 11 | 5 | 23+29 | 5+7+17+31 | 13 | 7 | 5 | 11 | 5+13 | 19 | 7 | 5+11 | 7 | 5 | 17 | 19 | 5 | 7+13+23 | 5+11 | 7 | 17 | 5+13+29 | 11 | 5 | 7+31+37 | 23 | 5+7`.

Nothing in these maps is positional: a run is a residue vector, the vector's CRT image is the
column, and the same vector's copies sit at every `x + t P`. Which vector is realised first is
the order of the CRT images of the realising vectors, and that order has no structure the maps
show (r34's q2b looked at the record vectors of m7..m23 for agreements with the gears' first
strikes and multiples of `5g`, and found chance rates; the same holds here).

### 4.2 The brief's Q3: are the first long runs built on twin-gear coincidences?

Count, over the 70 rows with `r >= 10` at m17..m41, of runs containing a column struck by both
members of a twin-gear pair `(g, g+2)` with `g >= 11` (`(11,13)`: classes `2, 24, 119, 141` mod 143;
`(17,19)`: `3, 54, 269, 320` mod 323; `(29,31)`: `5, 150, 749, 894` mod 899; `(41,43)` at m43 and up),
against the number expected if the run's position were uniform mod each `g(g+2)` (exact
per-row probabilities, independent across pairs by CRT):

| engine | rows | observed | chance |
|---|---|---|---|
| m17 | 1 | 1 | 0.38 |
| m19 | 2 | 1 | 0.97 |
| m23 | 10 | 7 | 6.35 |
| m29 | 11 | 7 | 7.36 |
| m31 | 14 | 6 | 10.74 |
| m37 | 17 | 10 | 13.46 |
| m41 | 15 | 12 | 12.36 |
| all | 70 | **44** | **51.6** |

**No.** The coincidence columns appear in the first runs at the chance rate or below it (44
against 51.6). The pre-registered threshold "fraction below 0.5" was the wrong statistic (long
runs meet a class among four mod 143 with probability 0.55 at `r = 30` and 0.94 at `r = 87`
by chance); P6's reading stands and its number is replaced by the chance comparison. The
arc-floor object (two twin gears strike the same column early, `ArcFloor.lean`) is a fact about
the prefix and plays no part in where the long runs sit.

### 4.3 Why the least solution is where it is

Three facts, each measured on the tables above and each exact:

1. Below `b` the answer is the primes'. `x_min(p, l) = 1, 53, 59, 111` for `l <= 4, 5, 11, 24` is the
   first-occurrence sequence of twin gaps (C3), the same at every engine that contains those
   columns, and the twin gap `659..811` (24 columns) is realised `700 x` and `37 x` before the mean
   spacing of the engine's own runs of that length at m19 and m17. The engine's runs below `b`
   are not the engine's; they are the twins'.
2. Above `b` the first realisation of each length is an outlier of a stationary process, nested
   across engines (3.1 reading 2), with steps of one column to a factor 472 between consecutive
   levels (3.1 reading 3), median first-hit ratio 0.5 to 2.4 and minima down to 0.03 (3.4). No
   floor is visible in the ratio and none is expected of a minimum of spread positions.
3. The position of the section is `a = (p^2 - 1)/6`, below every engine's first run of length
   `>= 25` (`5,643` at m29 and up, `3,648` at m37) and above the prefix's twin gaps. Step 8 asks
   about the run at exactly `a`, and that run's length is `L_1(p)`, the first-twin offset (C4).

## 5. Proof attempt, and the exact obstruction

**What is proved (no hypothesis):** C1 (the first realisation is the least element of a finite
union of residue classes, one per cover; computable two ways), C2 (monotone in the gear set),
C3 (below `b` the runs are twin gaps), C4 (step 8 at the cut `p` is `L_a(p) < l_p`, and `L_a` is
the first-twin offset `L_1`), C5 (the only position inequality the construction gives, and its
emptiness at 82 of 82 rows).

**The attempted lower bound.** A lower bound on `x_min(p, l)` from the construction would have to
bound the least element of a residue class `r_C mod M_C` from below. There is no such bound: `r_C`
is `s u_g - o (mod g)` per used gear, any residue can occur, and the least element of a class is
its residue. The one inequality available, C5, is `x >= r_U`, and at every measured row `r_U = x`.
By C2 the bound cannot come from a smaller engine; the larger engines converge to the primes,
whose first run of length `l_p` at or after `a` is the twin gap at `p^2`, i.e. the root.

**The exact obstruction, as a construction.** Take the gears `5, 7, 11, 13, 17` with teeth
`+-1, +-1, +-2, +-6, +-1` (the family member of 3.5; the real engine has `+-1, +-1, +-2, +-2, +-3`).
It has two classes per gear at a fixed distance and one free phase per gear, so the covering
instrument, the CRT description of `x_min`, C1, C2 and C5 hold for it verbatim, and it strikes
every column of the section of the cut 17 (columns 49..59): its run at the square column 48 has
length `>= 12 = l_p`. So no argument that uses only the gears' sizes and the two-classes-per-gear
structure can prove `L_a(p) < l_p` even at `p = 17`; the argument must use `u_g = 6^{-1} mod g`,
i.e. that gear `g`'s two classes are exactly the multiples of `g` among `6k -+ 1`. The same
brick as skeleton 10 (V12: shifted gears cover a section) and `fusion_lemma.md` 3.4 (the
remainder fails on the family), now at the section's own position with five gears and two
teeth changed. **Smallest instance:** `p = 17`, section 49..59, member `(1, 1, 2, 6, 1)`, 15
members of 1,440. **Also at `p = 29`** (19 columns, 6,030 of 1,995,840) **and at every cut from
37 to 53** (members exhibited in 3.5, each verified; 111 columns killed with fourteen gears at
53). **None** at `p = 7, 11, 13, 19, 23, 31` (sections of 11, 7, 19, 27, 51, 67 columns,
exhaustive): at those six cuts step 8 is a fact of every two-tooth engine with those gears (the
cover problem over the teeth is unsatisfiable), so no real-teeth mechanism is needed there; at
17, 29 and from 37 on the real teeth are the whole content. For an engine of primes with the
real teeth the least solution never sits at the section on the twelve cuts
(`L_a / l_p <= 0.375`).

**Are only counts available?** No: positions are available exactly (C1, the two instruments,
every number in section 3). What is not available is a lower bound on a position, because the
construction fixes residues and a residue is not bounded below. Counts (the census, W32's
first-hit model) predict positions to within a factor of order one at the medians and fail
by `700 x` at the prefix's outliers; a lower bound on `x_min` by counts would be a lower bound on
the minimum of spread positions, which does not exist. The step is therefore ROOT in position
form as it is in count form: `L_1(p) < l_p` at every cut, measured to `10^7`, no mechanism.

## 6. Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| P1 | gates | **CONFIRMED** 8 of 8 `F`; `x_min(23, 33) = 12,694,429`; window records 111 and 398; r71's four widest gaps | section 2 |
| P2 (X1) | first-hit floor `>= 1/4` | **REFUTED** as expected: min 0.0014 (m19, `l = 24`), 80 of 647 cells below 1/4 over m11..m53; medians 0.48-2.40 | 3.4 |
| P3 (X2) | quadratic floor `(3/8) l^2` | **REFUTED** as pre-registered at `l = 24` (0.514) and at the home-strike runs (0.033-0.167) | 3.4 |
| P4 | `x_min(p, l_p) > p'^2/6` at every cut | **REFUTED at `p = 29`** (`111 < 140 = a`); vacuous at 11, 13, 19, 23, 31; holds at 17, 37, 41, 43 (2.0, 9.9e4, 18.3, 7.3e4 times `b`); the ratio has no floor and is not 4.6 | 3.2 |
| P5 | all gears forced at the record; `x_min > M_U` at more than half the cells | **CONFIRMED** 9 of 9 engines at the record; second half **REFUTED**: `x_min = r_U` at 82 of 82 rows with `>= 2` forced gears | 4.1 |
| P6 | first runs not built on twin-gear coincidences (fraction below 0.5) | reading **CONFIRMED**, threshold wrong: 44 observed against 51.6 by chance over 70 rows | 4.2 |
| P7 | a family killer at `p = 13`; 1-20 % at 13, 17, 19 | **REFUTED on the cut** (0 at 13 and 19; the smallest is 17 at 1.04 %, then 29 at 0.30 %, then killers by search at 37 and 41; none at 7, 11, 13, 19, 23, 31); the obstruction exhibited | 3.5, 5 |
| brief | step 8 is `x_min(p, l_p) > p'^2 / 6` | **the quantifier is wrong**: step 8 is `L_a(p) < l_p`, the run at the square column (C4), which is `L_1(p)` of the frontier scan; `x_min` is below the square column at `p = 29` | 1, 3.2 |

## 7. What is new

1. **The position form of step 8, exactly (C4):** step 8 at the cut `p` is `L_a(p) < l_p`, the run
   through the square column measured from it against the section's length, and `L_a = L_1`,
   the first-twin offset; the brief's `x_min(p, l_p) > b` is strictly stronger and false at
   `p = 29`. This closes the "first realisation" formulation: the first run of the section's
   length ANYWHERE is the wrong object (it can sit in the prefix, as the twin gap 659..811 does
   for `{5..29}`); the run AT the square column is the object, and it is R4.d.i.a's.
2. **The first-realisation staircases** `x_min(p, l)` for every `l` at m11..m37 (complete) and the
   prefixes of m41..m53, with the record positions at m37 (`90,816,580,903`, 0.0734 P, two
   methods) and m41 (`630,700,131,373`, 0.0124 P, two methods), and the first realisations of the maximal
   words of m37 w.r.t. 41 (3.3). New exact numbers; the staircases are nested across engines
   and erratic in their steps.
3. **The CRT-minimum instrument** (`xm_words.py`): the first realisation column of a local
   pattern with no scan, by enumerating covers and walking each class to its least element;
   exact, self-verifying, 21 s at m37 and 455 s at m41 for the record pattern. It extends
   `ol_pattern.py` from existence to position.
4. **The forced-class fact:** at 82 of 82 rows with two or more forced gears the first
   realisation is the least element of its forced class, and every gear is forced at the record
   (9 of 9 engines) — the position carries no information beyond the residue vector.
5. **The family counter-construction at the section's own position:** two-tooth engines with the
   real gears kill the section at `p = 17` (15 of 1,440, teeth `(1,1,2,6,1)`), `p = 29`
   (6,030 of 1,995,840) and every cut from 37 to 53 (killers found by exact-cover search over
   the teeth and verified), and at none of `p = 7, 11, 13, 19, 23, 31` (exhaustive); the
   smallest engine at which the least solution sits at the section, the proof that any argument
   for step 8 at those cuts must use `u_g = 6^{-1} mod g`, and the six cuts at which it need not.
6. **Two laws refuted with their instances** (X1 at m19 `l = 24`, 0.0014; X2 at `l = 24`, 0.514),
   showing that no position floor uniform in `l` or of first-hit type exists for the engine: the
   prefix's twin gaps are the engine's own early runs.

Prior art, one line each: the CRT description of a window's realisation is r70's instrument,
used; the first-twin offset `L_1` and its arc bound are `frontier_floor_1e7.md` / R4.d.i.a,
rediscovered here in position language and stopped; the first-hit model is W32 (Kourbatov-type),
used only as a comparison; the coincidence classes are `tooth-sharing-pinning`, cited. Nothing
outside the repository is claimed.

## 8. Verdict

**The first realisation `x_min(p, l)` is an exact object with two instruments and complete
tables to m37, and it is not the object step 8 is about.** Step 8 in position form is the run
through the square column, `L_a(p) < l_p` (C4), which is the first-twin offset above `p^2`
(R4.d.i.a, measured to `10^7`, 0 exceptions): ROOT. The first-anywhere quantifier is refuted as
a target at `p = 29`. The construction gives no lower bound on any position (a residue is not
bounded below; the least element of the forced class is the position itself at 82 of 82 rows),
lower bounds cannot come from smaller engines (C2), and the family member `(1,1,2,6,1)` at
`p = 17` shows that the two-tooth covering structure with the real gears cannot give step 8
without the real teeth. Where the difficulty moved: nowhere new; it is confirmed to sit in
`u_g = 6^{-1} mod g` at the square column, the same brick as V12, V17 and the fusion remainder,
now with the smallest instance (five gears, two teeth changed, an eleven-column section).

- PROVED (in writing): C1-C5. Kernel: none.
- MEASURED: the staircases (3.1), the section rows (3.2, with the primality second measurement),
  the records (3.3), X1/X2 (3.4), the family (3.5, two methods), the mechanism counts (4).
- ROOT: `L_a(p) < l_p`; equivalently the arc bound `L_1 < (2p+1)/3` with `l_p >= (2p+2)/3`.
- What survived and where it goes: the CRT-minimum instrument (positions of any word at any
  engine to m53 in minutes), the nested staircases, the family's section killers as the
  obstruction's smallest instance. Node R4.c.iii.a / R4.d.i.a: the position form is theirs.

## 9. Dead ends (bricks), each with its refuting instance

| idea | dies at | instance | why it cannot be revived |
|---|---|---|---|
| step 8 as `x_min(p, l_p) > p'^2/6` (first run of the section's length anywhere) | `p = 29` | `x_min(29, 20) = 111 < a = 140`, the twin gap 659..811 | the prefix's twin gaps are the engine's runs (C3); the section is above them but the first run need not be |
| a first-hit floor `x_min >= P/(kappa S(l))` | m19, `l = 24` | `x_min . rho = 0.0014` (`700 x` early) | a minimum of spread positions has no floor; the record of `{5..19}` sits in the prefix |
| a floor uniform in `l`, `x_min >= c l^2` with the `c` that gives step 8 at twin cuts | `l = 24`, every `p >= 19` | `111 < 216` | the same gap; and the home-strike runs at `x = 1` |
| a floor from the forced class `x >= r_U` (C5) | every row | `r_U = x` at 82 of 82 | the forced modulus exceeds the position; the inequality is an identity |
| the first long runs as twin-gear coincidence structures (arc floor) | 70 rows | 44 observed against 51.6 by chance | coincidence columns are met at the chance rate; the arc floor is a prefix fact |
| step 8 from the two-class covering structure with the real gears | `p = 17` | teeth `(1,1,2,6,1)`: section 49..59 struck | the real teeth are the whole content |
| a lower bound from a smaller engine | C2 | `x_min(31, 30) = 6,026 < x_min(29, 30) = 102,273` | adding a gear only lowers first realisations |

## 10. Open items on the part alone, sorted

- **Closed here.** The definition and computation of `x_min`; the identity of the position form
  of step 8 with `L_1 < l_p`; the record positions to m41 (two methods each from m23); the two
  laws; the family at the section to `p = 53` (exhaustive to 31, by cover search from 37).
- **Not finished, measurement only.** The record positions at m43 (upper bound
  `233,885,349,904,191`), m47, m53: the CRT instrument needs more than its time limits there; a
  resumable search with a memo, as r73's `fl_next.py` has, would finish m43 in a few hours.
- **Measurement with no structural content.** The fractions of the period at which records first
  appear (0.39, 0.025, 0.0014, 0.00007, 0.34, 0.19, 0.044, 0.073, 0.012); the coincidence counts.
- **Root question in disguise.** `L_a(p) < l_p`; `x^{>=a}(p, l_p) > a`; any floor on `x_min` at
  `l = l_p` valid for the primes.
- **Genuinely open on the part alone, with the attack.** (i) [answered in the same round: the
  cover search over the teeth kills the sections at 37 and 41; the family's behaviour at the
  cuts 43..59 is one millisecond search each and is measurement only]. (ii) The order of the CRT images: is
  the first realisation of the record at a fraction of the period that falls with `p` (0.34,
  0.19, 0.044, 0.073, 0.012 at m23..m41)? A count of record runs per period by the census
  instrument (`u45_census.py`) at m37 and m41 would say whether the first is early relative to
  its copies or whether the copies are many; measurement only. (iii) The nested staircase: at
  which levels does the added gear PRECEDE (create an earlier first realisation) rather than
  lengthen in place, as a function of the gear's residue at the old run; r69's full-period
  exceptions are the same object.

## 11. Files

- `research/anchor235/r74/xm_scan.py` - the segmented scan; `results/xm_scan_m{p}.json`
  (staircase, histogram, checkpoint), `results/xm_scan_m{p}_from{a}.json` (from the square
  column), `results/scan_m{p}.log`
- `research/anchor235/r74/xm_words.py` - the CRT-minimum instrument; `results/xm_words.jsonl`
  (every run, with the residue vector and the verification flag), `results/words_records.log`
- `research/anchor235/r74/xm_mech.py` - kill maps, forced sets, coincidences; `results/xm_mech.json`,
  `results/mech_all.txt`
- `research/anchor235/r74/xm_family.py`, `xm_family_check.py`, `xm_family_cover.py` - the tooth
  family at the section (enumeration, recount, cover search); `results/xm_family.json`,
  `results/family.txt`, `results/family_check.txt`
- `research/anchor235/r74/xm_table.py` - the tables; `results/xm_table.json`
- `results/` is untracked; nothing here is committed.

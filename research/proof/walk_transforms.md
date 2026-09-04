# W.t - Transformations of the path

Branch of R2.a (the machine feeds on itself). Parent observation: the walk from `q^2` under
`{5..q}` starts on a tooth of the top gear, is struck by it once, and lands on a twin before the
top gear's next tooth (W1, one exception at `q = 53`). R2.a described the walk's *frame*. This
branch takes the **path itself** - the run of blocked columns the walk crosses - in six
representations and transforms each one, to find what part of the machine gives the path its
shape.

Scripts: `research/anchor235/r38/pt_path.py` (the sweep), `pt_qr.py` (the offset transform),
`pt_spectrum.py` (spectra and autocorrelation), `pt_levels.py` (across levels). Result outputs
(untracked): `research/anchor235/r38/results/pt_*.txt`. Every number this document relies on is
printed here.

Overlap note: branch W.a (`research/proof/walk_path.md`, the arithmetic decomposition of the same
path, run in parallel without contact) independently reached the offset-progression form of P5,
the gear-5 forced phase and `L >= 2` of P2, and the quadratic-residue admissibility of P6. Those
three are corroborated twice, by different code, and belong to both branches. The rest of what is
below is only here (see `docs/novel/walk-path-transforms.md` for the split).

Owner's organising question for this branch, verbatim: *"How does the machine build the path,
what parts of the machine contribute to the shape of the path, are those parts of the machine
measured individually and understood and proven, and if they are, then work on figuring out how
to prove their interactions' involvement in the path-shape."* Every feature below is therefore
tagged with the **order of interaction** needed to explain it:

* **order 0** - the anchor `2,3,5` alone (the cycle-30 frame, gear 5's teeth `+-1 mod 5`);
* **order 1** - one gear, through its proven per-gear pattern: gear `g` strikes the two
  arithmetic progressions `k = +-u_g (mod g)` (tooth rule `6 u_g = +-1`), phase fixed by
  `q^2 mod g`, the two arcs between the teeth `d_g` and `g - d_g` (two-teeth kill-spacing law);
* **order 2** - a proven pairwise law: the chain law, the merge law, tooth-sharing pinning, the
  gear-5 lock, neighbour-of-hit;
* **order 3+** - needs the joint behaviour of three or more gears; **no proof on record**.

---

## 0. Pre-registered (written before computing)

### 0.1 Objects

For a prime `q >= 5` write `p = prevprime(q)`, `q' = nextprime(q)`, and

* `k_0 = (q^2 - 1)/6` - the column holding `q^2` (the walk's first column);
* the **machine** is `{5..q}` throughout; `dep(k)` = the number of gears of `{5..q}` striking
  column `k`; `ms(k)` = the smallest such gear (0 if the column is open);
* the **path** `P(q)` = columns `k_0 .. k_0 + L - 1`, all blocked, with `k_0 + L` the landing
  (the first opening at or above `k_0`); `L = L(q)` is the walk length;
* the **section** `S(q)` = columns `(p^2, q^2]`, i.e. `k_p + 1 .. k_0` with `k_p = (p^2-1)/6` -
  the new part of the machine's window, in the project's vocabulary;
* the **run-out** `[k_0, k_1]`, `k_1 = (q'^2-1)/6` - where the path lives (above the window).
  On `[k_0, k_1)` an opening of `{5..q}` is exactly a twin column, because a number
  `n <= q'^2` with no prime factor `<= q` is prime or `q'^2`, and no prime lies in `(q, q')`.
* `c = 6^{-1} mod q`, `u_q = min(c, q-c)`, `d = 2c mod q` (the forward tooth arc), `q - d` (the
  backward tooth arc).

### 0.2 The six representations

(a) **blocked string** `B(k) = [dep(k) >= 1]` on `S(q)` and on `[k_0, k_1]`;
(b) **depth string** `dep(k)` on the same;
(c) **smallest-striker word** `ms(k)` along the path;
(d) **hop sequence per layer**: for each gear `g`, the offsets `i` with `ms(k_0+i) = g`;
(e) **anchor-30 cycle coordinate**: `k mod 5`, i.e. which of the three twin slots (`11|13` at
    `k = 2`, `17|19` at `k = 3`, `29|31` at `k = 0 mod 5`) the path crosses;
(f) **mirror path**: the backward walk from `k_0` to the previous opening, length `L^-`, its
    word and depth string, against the forward path.

### 0.3 Predictions, with what would refute each

**T1 (predicted TRUE, order 0, elementary).** The path starts on the `29|31` slot when
`q = +-1, +-11 (mod 30)` and on the `17|19` slot when `q = +-7, +-13 (mod 30)`; it *never*
starts on the `11|13` slot. Refuted by one `q` with `k_0 = 2 (mod 5)`.

**T2 (predicted TRUE, order 1).** The top gear strikes the column `k_0` and no other column
within `d` above it or `q - d` below it, with `d = (q+1)/3` for `q = 5 (mod 6)` and
`d = (2q+1)/3` for `q = 1 (mod 6)`; so `k_0` is the unique `q`-strike in an interval of length
`q` around it. Refuted by one `q` with a second `q`-strike inside `(k_0 - (q-d), k_0 + d)`.

**T3 (predicted TRUE, NEW).** Because the forward arc is `~q/3` when `q = 5 (mod 6)` and
`~2q/3` when `q = 1 (mod 6)`, the one-strike rule has twice the room in the second class. The
prediction: every tight walk - `L/d` above 0.4, and the single failure `q = 53` - lies in the
class `q = 5 (mod 6)`. Refuted by one `q = 1 (mod 6)` with `L/d > 0.4`.

**T4 (predicted TRUE, order 1, NEW as far as the record goes).** Among all teeth of gear `q`
inside its own window `(q, q^2]`, the column `k_0` is the *only* one at which `q` is the sole
striker of its own member: any other tooth carries the member `q m` with `1 < m < q`, and every
such `m` has a prime factor `<= q`. Hence the path starts at the shallowest tooth of the top
gear. Refuted by one tooth column `k < k_0` whose `q`-member has no second striker.

**T5 (predicted TRUE for the first clause, FALSE as a rule for the second).** `dep(k_0) = 1`
iff `q^2 - 2` is prime (the square gate, N2 read as a depth); but the path's minimum depth is
*not* always at offset 0 - other depth-1 columns occur inside the path. Report the fraction.

**T6 (predicted TRUE, order 2).** In the smallest-striker word the only forbidden transitions
are the diagonal ones `g -> g` at distance 1 (neighbour-of-hit: `d_g = 3^{-1} mod g` is never
`+-1`). Refuted by an off-diagonal pair `(g, h)`, both `<= 43`, never adjacent in the corpus
while residue-admissible.

**T7 (predicted TRUE, order 1).** A gear `g` strikes the path twice only if
`min(d_g, g - d_g) < L`, i.e. only if `g <= 3L + 1`; every gear above that strikes at most once.
Refuted by one double-striking gear with `g > 3L + 1`.

**T8 (predicted TRUE, order 1).** Consecutive hop offsets of the same gear differ by exactly
`d_g` or `g - d_g` (two-teeth kill-spacing), never anything else. Refuted by one other
difference.

**T9 (predicted FALSE - nothing expected).** The blocked string's discrete Fourier spectrum over
the section is the sum of the per-gear terms (spikes at frequencies `m/g`); the walk's position
is *not* spectrally distinguished from other positions in the section. What would surprise me: a
spectral statistic (local phase, energy in a band) that separates `k_0` from random columns of
the section at every `q`.

**T10 (predicted FALSE - nothing expected).** The path length `L(q)` is a typical run length of
the section's blocked string and of walks from other teeth of `q`. What would surprise me: the
`q^2` walk being systematically the longest, or the shortest, among the teeth.

**T11 (predicted FALSE - nothing expected, from N3).** No representation of level `n` predicts a
feature of level `n+1`. The only thing that repeats across levels is the *frame* (start on a
tooth, one strike, arc `d`), which is W1/W4 and already on record. What would surprise me: the
word, the depth profile or the hop-layer set at level `n` determining anything at `n+1`.

**T12 (shape, not a fit).** The growth of `L`, of the number of hopping layers, of the maximum
depth and of the total strike count with `q`: report the shape by decade, name the mechanism,
and stop the moment it is a density statement.

### 0.4 What counts as a rule here

(i) a statement about positions or residues, not a rate; (ii) an exact exception count over a
stated range; (iii) uniform in `q`. A density, a fitted exponent or an average is not a rule. A
restatement of the tooth rule, the two-teeth kill-spacing law, neighbour-of-hit, the chain law,
the gear-5 lock, the horizon (least-prime-factor) lemma, the merge law or the Hardy-Littlewood
count is **not** a finding: it is noted in one line, tagged with its order, and the sub-question
stops.

### 0.5 Scorecard

| # | Prediction | Order | Verdict |
|---|---|---|---|
| T1 | start slot by `q mod 30`, never `11\|13` | 0 | **CONFIRMED**, 0 exceptions / 2,260 (`q = 5` degenerate) |
| T2 | one `q`-strike in the whole tooth interval | 1 | **CONFIRMED**, 0 exceptions / 2,260 |
| T3 | tight walks only in `q = 5 (mod 6)` | 1 | **REFUTED as stated** (3 of 12 tight walks are `1 mod 6`); the sharpened two-sided form is P3 |
| T4 | `k_0` the unique sole-striker tooth of `q` | 1 | **CONFIRMED**, 0 of 337,011 teeth |
| T5 | `dep(k_0) = 1` iff square gate; min not always at 0 | 1 | **BOTH CONFIRMED** (0 / 2,260; min at offset 0 in 666 of 2,212) |
| T6 | only `g -> g` forbidden | 2 | **CONFIRMED**: 12 zero diagonal cells, 132 of 132 off-diagonal cells realised |
| T7 | double strike only if `g <= 3L + 1` | 1 | **CONFIRMED**, 0 exceptions |
| T8 | same-gear hop spacing is `d_g` or `g - d_g` | 1 | **CONFIRMED**, 0 exceptions |
| T9 | no spectral distinction | - | **CONFIRMED** (null); the spectrum is the gear lines, `k_0` sits at percentile 0.37-0.50 |
| T10 | `L` typical among teeth and random columns | - | **CONFIRMED**: percentile 0.5270; 47.3% of tooth walks longer |
| T11 | nothing crosses levels | - | **CONFIRMED** (null); only the frame repeats |
| T12 | shapes | - | reported; `L` tracks the twin-gap null to within 2% from `q = 200` |

Not pre-registered, found while testing (and the branch's main results): **P5-P7**, the offset
transform - two arithmetic progressions per gear in the offset coordinate, the quadratic-residue
bar on which gear may reach which offset, and the square phase vector.

---

## 1. Setup

| object | range computed | script |
|---|---|---|
| the path from `q^2` under `{5..q}`, all six representations | every prime `q = 5..19,997` (2,260 walks), segmented sieve on the columns `(p^2, q'^2]` | `pt_path.py` |
| the section's blocked string, run lengths, 1,000 random-column walks and every tooth of `q` per section | the same 2,260 sections, 11,756 other teeth | `pt_path.py` |
| the offset transform: strikers of column `k_0 + i` against the admissible set | 2,260 walks x 193 offsets `i = -96..96` x every gear: **493,101,490** (gear, offset) checks | `pt_qr.py` |
| section spectra and autocorrelation | 20 sampled `q` from 101 to 19,997, section lengths 132 to 76,380 columns | `pt_spectrum.py` |
| the path's own correlation in the offset coordinate | all 2,260 walks x 257 offsets | `pt_spectrum.py` |
| chains across levels | 20 chains from `q = 5..79`, 49 levels, `g` to 734,471 (prime table to 2 x 10^6) | `pt_levels.py` |

Openings on the run-out are certified twins by the window lemma (no prime in `(q, q')`, so a
number `<= q'^2` with no factor `<= q` is prime, except `q'^2` itself at the endpoint).

## 2. Results

### 2.1 Representation (e), the anchor-30 coordinate: the start slot is pinned

`q^2 = 1` or `19 (mod 30)` for every prime `q > 5`, so `k_0 = (q^2-1)/6` is `0` or `3 (mod 5)`:

| `q mod 30` | 1 | 7 | 11 | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|---|---|---|
| `k_0 mod 5` | 0 | 3 | 0 | 3 | 3 | 0 | 3 | 0 |
| walks | 275 | 287 | 288 | 285 | 282 | 277 | 283 | 282 |

Slot 11-13 (`k_0 = 2 mod 5`): **0 of 2,260**. The one walk outside the table is `q = 5`
(`k_0 = 4`, the degenerate machine whose top gear is the anchor's own gear 5).

Consequences, all exact and all order 0:

* gear 5 **never** strikes the walk's first column (`q > 5`);
* `k_0 + 1` is `1` or `4 (mod 5)` in **both** classes, so **gear 5 strikes offset 1 at every
  prime `q > 5`** and therefore `L >= 2` always. Measured minimum `L` over `q > 5`: **2**.
  Opening density at offset `+1` over all 2,260 walks: 0.00044, i.e. the single `q = 5`.
* gear 5's whole contribution to the path is fixed in advance: it strikes the offsets
  `i = 1, 4 (mod 5)` in class A (`q = +-1, +-11 mod 30`, 1,122 walks) and `i = 1, 3 (mod 5)` in
  class B (`q = +-7, +-13 mod 30`, 1,137 walks). Offsets `i = 1 (mod 5)` are struck at **every**
  `q`. Gear-5 share of the 88,677 path columns: **0.4025**, i.e. `2/5` exactly.
* backward the same computation is *asymmetric*: `k_0 - 1` is `4 (mod 5)` in class A (struck)
  and `2 (mod 5)` in class B (an anchor-open slot). Opening density at offset `-1`: 0.03230
  against 0.00044 at `+1`.

### 2.2 Representation (f), the mirror path, and the two tooth arcs

`6 k_0 = -1 (mod q)` puts the walk on the tooth `-c`; the other tooth is `+c`, so the forward
arc is `d = 2c mod q` and the backward arc `q - d`. By `q mod 6`:

| class | walks | forward arc `d` | backward arc `q-d` | median `L/d` | median `L^-/(q-d)` | max `L/d` | max `L^-/(q-d)` |
|---|---|---|---|---|---|---|---|
| `q = 1 mod 6` | 1,124 | `(2q+1)/3` | `(q-1)/3` | 0.0056 | 0.0111 | 0.7692 (`q=19`) | **1.2000** (`q=31`) |
| `q = 5 mod 6` | 1,136 | `(q+1)/3` | `(2q-1)/3` | 0.0112 | 0.0052 | **1.5000** (`q=53`) | 0.4066 (`q=137`) |

So the short arc is *forward* for `q = 5 (mod 6)` and *backward* for `q = 1 (mod 6)`, and both
failures of the one-strike property sit in the short direction of their own class: `q = 53`
forward and `q = 31` backward. Over all 2,260 walks the maximum of (path length)/(arc) is
**1.5000 in the short direction and 0.7692 in the long one**. Exceptions to `k_0 = -c`, next
strike at `+d`, previous at `-(q-d)`: **0 of 2,260**.

The forward and backward paths are otherwise unrelated: correlation of `L` and `L^-` **0.0589**;
`L^- > L` at 1,074 of 2,260; the forward word equals the reversed backward word at **0 of 44**
walks with `q <= 200`. The local mirror is not the period mirror `k -> -k`.

Maximal run through `k_0` (`L^- + L`): median 64, max 484 at `q = 16,987`. `L^- = 0` (the
column of `q^2` starts its own run) at 73 of 2,260.

### 2.3 Representation (d), the offset transform: two progressions per gear

The members of column `k_0 + i` are `q^2 + 6i - 2` and `q^2 + 6i`, so

```
   g strikes offset i   iff   i = i_lo  or  i = i_hi   (mod g),
   i_lo = (2 - q^2) 6^{-1} mod g,   i_hi = -q^2 6^{-1} mod g,   i_lo - i_hi = d_g.
```

Checked against divisibility at every (gear, offset) pair: **0 disagreements in 493,101,490
checks**; walks where the separation is not `d_g`: **0 of 2,260**. So the path is exactly the
covering of `[0, L)` by two arithmetic progressions per gear, difference `g`, separation `d_g`,
phase set by `q^2 mod g` alone.

Two consequences follow from the phase being a function of `q^2`:

**The quadratic-residue bar.** Gear `g` can strike offset `i` for *some* `q` only if `2 - 6i` or
`-6i` is a nonzero quadratic residue mod `g`. Strikes by a barred gear: **0 in 493,101,490
checks**. Size of the admissible set, as a fraction of the 2,260 gears:

| offset `i` | -54 | -24 | -6 | -3 | -1 | **0** | 1 | 3 | 12 | 96 | median over the 192 nonzero offsets |
|---|---|---|---|---|---|---|---|---|---|---|---|
| admissible | 1.0000 | 1.0000 | 1.0000 | 0.7482 | 0.7473 | **0.4960** | 0.7478 | 0.7500 | 0.7447 | 0.7482 | 0.7473 (min 0.7323) |

At `i = 0` the admissible set is **exactly** `{g = +-1 (mod 8)}` (the QR test and the mod-8 test
agree on all 2,260 gears), and of the **3,212** strikers of a first column other than the top
gear, **0** fall outside that class. At `i = -6t^2` the constant `-6i = (6t)^2` is a perfect
square, so *every* gear is admissible.

**The square phase vector.** Both phases are functions of `q^2 mod g`, and `q^2` is a square
modulo every gear, so the walk's phase vector lies in the image of the squaring map - one part in
`2^{pi(q)}` of the phase space (2,260 coordinates at `q = 20,000`). Exact, for every `q`.

**Hop spacings (T7, T8).** Over all paths: gears striking a path twice with
`min(d_g, g - d_g, g) >= L`: **0**. Same-gear strike spacings outside `{d_g, g - d_g, g}`: **0**.
Distinct striking gears per path: min 1, median 53, max 458; distinct smallest-striker gears
(hop layers): min 1, median 11, max 61; total strikes on the path: median 84, max 1,358.

### 2.4 Representation (b), the depth string: dip, plateau, spike

Mean depth by normalised position along the path (bin 0 = the first column, bin 20 = the last
blocked column before the landing), over all 2,260 paths:

```
 3.05 3.32 3.31 3.30 3.29 3.29 3.27 3.28 3.28 3.30 3.30 3.31 3.29 3.39 3.24 3.28 3.32 3.32 3.33 3.33 3.75
```

| quantity | measured | independent-gear value |
|---|---|---|
| first column `k_0` | **2.4212** | - (see below) |
| random blocked column of the section | 3.2692 | `sum 2/g = 3.1805` |
| last blocked column before the landing | **3.7668** | `sum 2/(g-2) = 3.7007` |
| column just above the landing | 3.6934 | `sum 2/(g-2) = 3.7007` |

`dep(k_0) = 1` exactly when `q^2 - 2` is prime: **0 exceptions of 2,260** (451 walks with the
gate open). Depth-1 share at `k_0`: 0.1996, against 0.1381 for a random blocked column of the
same section. The path's minimum depth sits at offset 0 in **666 of 2,212** paths (30%, against
4.8% for a uniform position); the maximum sits in the last bin 152 times and the first 81 times.
Max depth along a path: min 1, median 7, max 10.

**Depth is a function of the offset.** Averaged over the 1,033 walks with `q >= 10,000` (one
machine-size band), the mean depth at offset `i` runs from **2.0465 at `i = 87`** to **5.8209 at
`i = -54`**, and the root-count prediction (the number of solutions of `x^2 = -6i` and
`x^2 = 2-6i` mod `g`, summed over the gears with weight `1/(g-1)`) reproduces it with
**correlation 0.9694** (mean 3.3905 measured against 3.5962 predicted, max deviation 0.5888 over
the nonzero offsets). The opening density at offset `i` runs from **0.00000** (at `i = -96`) to
**0.10649** (at `i = 87`), mean 0.02128. The second moment is not reproduced: measured variance
2.7224 against 2.9171 predicted for independent gears, correlation 0.5621 across offsets.

**Forced columns behind `q^2`.** Column `k_0 - 6t^2` carries the member `q^2 - 36t^2 =
(q-6t)(q+6t)`, blocked at every `q > 6t+1`: **9,021 checked, 0 open**. Forward, `q^2 + 6i` is a
square only for `i >= (2q+2)/3` (`q = 5 mod 6`) or `(4q+8)/3` (`q = 1 mod 6`) - beyond the tooth
arc in both classes. So the forced-composite square columns lie only behind the walk.

### 2.5 Representation (c), the smallest-striker word

88,677 path columns. Letter shares: 5: 0.403, 7: 0.172, 11: 0.079, 13: 0.053, 17: 0.037,
19: 0.028, 23: 0.022, 29: 0.014, 31: 0.014, 37: 0.012; letters above 100: 0.1022.

Transition counts for gears `<= 43`: all **12 diagonal cells are 0** and all **132 off-diagonal
cells are non-zero** (smallest cell 2). Against an independent-letter model, row 5 and column 5
run at 1.5-1.9 and big-to-big cells at 0.2-1.0. Both are order 0, not new structure: gear 5
occupies the offsets `1, 4 (mod 5)` (class A) or `1, 3 (mod 5)` (class B), so two of the three
non-5 residues are followed by a 5-residue and

```
   P(next letter = 5 | this letter is not 5) = 2/3 exactly,   against the share 2/5.
```

Measured inside paths: **35,693 of 52,212 = 0.6836**. The excess over `2/3` is a selection
effect - a blocked run cannot end at a column whose successor is a 5-tooth.

### 2.6 Representation (a) transformed: run lengths, spectrum, autocorrelation (T9, T10)

*Run lengths.* Section length median 18,744 columns, max 340,340; blocked runs per section
median 443; longest run in the section median 251, max 633. The path `L` sits at percentile
**0.505** (median over `q`) among its own section's blocked runs, and is longer than every run of
its own section at 16 of 2,260.

*Spectrum.* The section's blocked string, mean removed. The largest peak is the gear-5 line
`2/5` at every one of the 20 sampled `q`; the second is `3/7`; the third is `1/7`, `3/11`, `5/11`
or `6/13`. The energy share carried by the lines of gears `<= 43` falls with `q`:
0.9909 (`q = 101`), 0.9934 (131), 0.6670 (307), 0.3903 (541), 0.2228 (2,161), 0.1280 (6,569),
0.0597 (11,467), 0.0799 (19,997) - median 0.3171 over the sample. Peak amplitudes `|A|/N` fall
from 0.062 to 0.011.

*Autocorrelation* of the section string at lags 1..12 (`q = 8,677`): -0.014, -0.001, -0.007,
-0.017, **+0.016**, -0.021, **+0.022**, +0.008, -0.010, -0.003, -0.013, +0.011 - the lag-5 and
lag-7 positives are the two smallest gears' periods, everything else is at the noise level.

*Is `k_0` distinguished?* Local opening count in a window centred on `k_0`, as a percentile of
the same window slid over the section: median **0.502** (`W = 50`), **0.371** (`W = 200`),
**0.431** (`W = 1000`). And the walk length itself: percentile **0.5270** (median over `q`)
among 1,000 random blocked-column walks of the same section; **47.34%** of the 11,756 walks from
the other teeth of `q` are strictly longer than `L`; `L` is below its own section's random-blocked
mean at 1,359 of 2,260. **The position of `q^2` is not distinguished by any length or spectral
statistic** - only by its depth (2.42 against 3.27).

*The path's own correlation in the offset coordinate*, over all 2,260 walks and offsets
`-128..128` (blocked fraction 0.9716): lags 1..15 give -0.009, +0.022, +0.011, -0.003, **+0.050**,
-0.006, **+0.040**, +0.014, -0.002, **+0.034**, -0.002, +0.026, +0.020, +0.005, **+0.042** - the
peaks at 5, 7, 10, 15 are the two smallest gears' periods.

### 2.7 Scaling (T12)

| `q` range | walks | median `L` | max `L` | median `L^-` | median hop layers | median max depth | median strikes | median `d` |
|---|---|---|---|---|---|---|---|---|
| 5-20 | 6 | 2 | 10 | 0 | 2 | 2 | 3 | 5 |
| 20-50 | 7 | 5 | 10 | 7 | 3 | 3 | 9 | 16 |
| 50-100 | 10 | 8 | 27 | 4 | 4 | 4 | 16 | 35 |
| 100-200 | 21 | 10 | 45 | 9 | 6 | 5 | 25 | 66 |
| 200-500 | 49 | 12 | 87 | 11 | 8 | 5 | 32 | 154 |
| 500-1,000 | 73 | 17 | 83 | 17 | 8 | 5 | 39 | 326 |
| 1,000-2,000 | 135 | 19 | 139 | 16 | 10 | 6 | 56 | 673 |
| 2,000-5,000 | 366 | 24 | 265 | 22 | 10 | 6 | 74 | 1,547 |
| 5,000-10,000 | 560 | 25 | 402 | 29 | 11 | 7 | 86 | 3,345 |
| 10,000-20,000 | 1,033 | 32 | 383 | 34 | 12 | 7 | 110 | 6,658 |

Overall `L`: min 1, median 25, mean 39.24, max **402 at `q = 8,699`**; `L^-`: min 0, median 26,
mean 38.31, max 421 at `q = 15,121`. Against the twin-gap null `(ln q^2)^2 / (12 C_2)` the mean
`L` gives ratios 1.176, 0.911, 1.029, 1.071, 1.001, 0.917, 0.988, 1.008, 1.006, 0.982 by decade -
**the path length is the twin-gap null and nothing else**. That is a rate; stopped there.
Max depth grows like `log log`, hop layers like `L` over the mean hop, strikes like `L sum 2/g`;
all rates, all stopped.

### 2.8 Across levels (T11)

20 chains from `q = 5..79`, 49 levels, deepest `g = 734,471`. 29 consecutive level pairs.

*Nothing crosses:* `L_n` vs `L_{n+1}` correlation **0.0940** (`L` increases at 25 of 29); hop
layers correlation 0.1105; the class `g mod 30` has a single successor at only 3 of 9 source
classes; the square gate is not inherited (open to shut 11 times, shut to open 4, shut to shut
13, open to open 1); the first three letters of the word agree at 8 of 29. The max-depth
correlation 0.5597 is the machine growing along a chain, not a relation.

*The frame repeats exactly:* levels with `L < d`: **48 of 49** (the exception is `q = 53`);
levels starting on slot 11-13: **0 of 49**; levels whose offset 1 is blocked by gear 5: **48 of
49** (the exception is `q = 5`); levels where `dep(k_0) = 1` exactly when `g^2 - 2` is prime:
**49 of 49**; gear-5 share of path columns across all levels **0.4031**.

## 3. Candidate rules, with exception counts

**P1 (pinned start slot).** For every prime `q > 5` the walk starts on the 29-31 slot if
`q = +-1, +-11 (mod 30)` and on the 17-19 slot if `q = +-7, +-13 (mod 30)`; never on 11-13.
**0 exceptions in 2,260 walks and 49 chain levels.** Order 0.

**P2 (the anchor's fixed contribution to the path).** Gear 5 never strikes offset 0, strikes
offset 1 at every `q > 5` (hence `L >= 2`), and strikes exactly the offsets `1, 4 (mod 5)`
(class A) or `1, 3 (mod 5)` (class B). Offsets `i = 1 (mod 5)` are struck at every `q`.
Consequence in the word: `P(next letter = 5 | this letter is not 5) = 2/3` by residues.
**0 exceptions in 2,260 walks** (opening density at offset `+1`: 1 walk in 2,260, the degenerate
`q = 5`). Order 0.

**P3 (one tooth per run).** The maximal blocked run containing the column of `q^2` holds exactly
one strike of the top gear; equivalently `L < d` and `L^- < q - d`. **2 exceptions in 2,260
walks**: `q = 31` (backward, `L^- = 12` against `q - d = 10`) and `q = 53` (forward, `L = 27`
against `d = 18`); none above `q = 53`. Each exception is in the short-arc direction of its own
class, and the maximum of (path)/(arc) over all walks is 1.5000 short against 0.7692 long. Order
1 for the frame; the inequality itself is order `pi(q)`.

**P4 (the unique sole-striker tooth).** Among all teeth of gear `q` in its own window
`(q, q^2]`, the column of `q^2` is the only one at which `q` is the sole striker of its member.
**0 exceptions over 337,011 teeth** (`q <= 4000`); one-line proof for all `q`. Order 1.

**P5 (the offset transform).** Gear `g` strikes offset `i` iff `i = (2-q^2)6^{-1}` or
`-q^2 6^{-1} (mod g)`, two progressions of difference `g` separated by `d_g`. **0 disagreements
in 493,101,490 (gear, offset) checks; 0 walks with a wrong separation.** Order 1.

**P6 (the quadratic-residue bar).** Gear `g` can strike offset `i` for some `q` only if `2 - 6i`
or `-6i` is a nonzero quadratic residue mod `g`; the barred quarter of the machine is decided by
the offset alone. At `i = 0` the admissible set is exactly `{g = +-1 (mod 8)}`; at `i = -6t^2`
it is everything. **0 strikes by a barred gear in 493,101,490 checks; 0 of 3,212 first-column
strikers outside `+-1 (mod 8)`.** Order 1.

**P7 (the square phase vector).** The phase of every gear's pair of progressions is a function of
`q^2 mod g`, hence a square in every coordinate: the walk's phase vector lies in the image of the
squaring map, one part in `2^{pi(q)}` of the phase space. Exact for every `q`. **And it makes no
difference to the length**: percentile of `L` among random blocked columns 0.5270. Order 1 per
gear, order `pi(q)` as a joint constraint.

**P8 (the depth profile).** Dip, plateau, spike: 2.4212 at the first column, 3.24-3.39 across
the interior (independent-gear value `sum 2/g = 3.1805`, measured 3.2692 on a random blocked
column), 3.7668 at the last blocked column (independent-gear value `sum 2/(g-2) = 3.7007`).
`dep(k_0) = 1` iff `q^2 - 2` is prime: **0 exceptions in 2,260**. Orders 1 (dip, plateau) and 2
(spike).

**P9 (the per-offset depth law).** The mean depth at offset `i` is a function of `i` alone -
the root counts of `-6i` and `2-6i` over the gears - with correlation **0.9694** across 193
offsets, range 2.0465 to 5.8209. The variance is not reproduced (correlation 0.5621, measured
2.7224 against 2.9171). Order 1 for the mean.

**P10 (the forced square columns).** Column `k_0 - 6t^2` is blocked at every `q > 6t + 1` (its
member is `(q-6t)(q+6t)`): **9,021 checked, 0 open**. Forward the nearest such column is at
`i >= (2q+2)/3`, beyond the tooth arc: the family lies entirely behind the walk. Order 1.

**P11 (the word's transitions).** The only forbidden transition is `g -> g` (12 zero diagonal
cells for gears `<= 43`); every one of the 132 off-diagonal cells occurs. Order 2
(neighbour-of-hit), and everything else about the transition matrix is P2.

**P12 (double strikes).** A gear strikes the path twice only if `min(d_g, g-d_g, g) < L`, and
consecutive same-gear strikes are `d_g`, `g - d_g` or `g` apart. **0 exceptions each.** Order 1
(two-teeth kill-spacing).

## 4. Mechanism, and the order of interaction

**Order 0 - the anchor alone, and it owns 40% of the path.** `q` is coprime to 30, so
`q^2 = 1` or `19 (mod 30)` and `k_0 = 0` or `3 (mod 5)`: the walk's first column is pinned to one
of two anchor-open slots. Since gear 5's teeth are `+-1 (mod 5)`, that single pinned residue
fixes gear 5's entire pattern along the path in advance - `i = 1, 4` or `i = 1, 3 (mod 5)` - so
the largest contributor to the path (0.4025 of its columns, exactly `2/5`) is deterministic up to
one bit of `q mod 30`. It also forces offset 1 to be blocked at every `q > 5`, and it makes the
forward and backward directions inequivalent at distance 1.

**Order 1 - each gear, two progressions in the offset.** `6(k_0+i) - 1 = q^2 + 6i - 2` and
`6(k_0+i)+1 = q^2+6i`, so `g` strikes offset `i` iff `q^2 = 2-6i` or `-6i (mod g)`, i.e.
`i = (2-q^2)6^{-1}` or `-q^2 6^{-1}`, two progressions of difference `g` separated by
`d_g = 2 x 6^{-1}` - the two-teeth kill-spacing law read in the offset coordinate. Three things
follow with no further input:

1. *the bar*: `q^2` is a square, so `g` can reach offset `i` only if `2-6i` or `-6i` is a residue
   mod `g` - a quarter of the machine is barred from each offset, and at `i = 0` a half is
   (`g | q^2 - 2` needs `2` a residue, i.e. `g = +-1 mod 8`);
2. *the shallow start*: `q^2 = q x q` and every other tooth of `q` in the window carries
   `q m` with `1 < m < q`, whose factors are all gears - so `k_0` is the only column in the whole
   window where `q` is a sole striker, `dep(k_0) = 1 +` (the gears dividing `q^2-2`), and the
   square gate is exactly the event that the half-machine `{g = +-1 mod 8}` misses;
3. *the arcs*: `k_0 = -6^{-1} (mod q)` sits on a tooth with `d` ahead and `q-d` behind, one of
   them `~q/3` and the other `~2q/3` by `q mod 6`.

The per-offset depth is then the sum of the root counts, which is why the depth profile in the
raw offset coordinate is a fixed function of `i` (correlation 0.967) with peaks exactly at the
offsets `-6t^2` where the polynomial `q^2+6i` factors as a difference of squares.

**Order 2 - one proven pairwise law is needed, and only one.** The spike at the far end of the
path (3.77 against a plateau of 3.27) is the neighbour-of-an-opening effect: for each gear,
conditioning on `g` missing column `x+1` leaves both teeth available at `x`, so
`P(g strikes x | x+1 open) = 2/(g-2)` and the expected depth becomes `sum 2/(g-2) = 3.7007`
against `sum 2/g = 3.1805`. The same law (`d_g` is never `+-1`, kernel-checked as
`AnchorChain.neighbour_of_hit`) is the zero diagonal of the word's transition matrix. No other
pairwise law was needed anywhere in the branch.

**Order 3+ - what is left.** After orders 0-2 the path's *local* shape is fully accounted for:
which columns gear 5 takes, where the deep and shallow columns sit, which gear can reach which
offset, how far apart one gear's strikes are, what the two ends look like. What is not accounted
for is **the first offset that every progression misses** - the length `L` - and its two-sided
form P3. This is the covering question: `2 pi(q)` progressions with pinned separations `d_g` and
square phases, and the question is where they leave a hole.

**The lowest-order unexplained feature of the path's shape is therefore the length itself, and
its sharpest small-scale form is P3 (one tooth per run, `L + L^- < q`).** Below that there is
nothing unexplained in this branch's measurements. The one measured quantity whose order-1 model
is only half right - the *variance* of the depth at a fixed offset (correlation 0.5621, measured
6.7% below the independent-gear value) - is not a finding: the model assumes `q` uniform modulo
each gear, which fails for the gears near `q`, so it is untested rather than unexplained, and it
is named here as the next thing to measure properly rather than as a result.

A second thing worth naming: P7 says the phase vector of the walk is a *global square* - an
exact, severe, joint restriction on all `pi(q)` gears at once - and the length statistic does not
notice it (percentile 0.5270 against 0.5). Why an order-`pi(q)` restriction of density
`2^{-pi(q)}` leaves the covering length untouched is the branch's sharpest open question about
interactions, and it is testable: it predicts that a counterfactual family with free phases has
the same `L` distribution as the real square-phase one.

## 5. What is new

Screened against `docs/novel/README.md`, `docs/proof-search/anchor-235.md` 9c-9g and
`research/proof/self_feeding.md`:

* **On record, restated here, not claimed:** the walk's tooth frame (W1), the square gate (W2),
  the level-free transfer congruence (W3 / N3 - P5 is that same congruence written in the walk's
  own offset coordinate, so the congruence is prior art), the hit and chain laws and
  neighbour-of-hit (`anchor-235-layer-laws` L1-L3, kernel-checked), the two-teeth kill-spacing law
  (P12), the gear-5 lock (5g), the identity "hops = walk length" (9c), and the machine's Fourier
  structure (`golden-spectral-gap`, `walk-transform-pole-identity`) - section 2.6 is that
  structure seen in a finite section, and adds nothing but the numbers.
* **New as far as the record goes:** P1 and P2 (the start slot pinned by `q mod 30`, gear 5's
  whole contribution to the path fixed in advance, `L >= 2` always, the `+1`/`-1` asymmetry, the
  `2/3` successor law); P3 (the two-sided one-tooth-per-run rule with its two exceptions and the
  short-arc explanation of both); P4 (the `q^2` tooth is the unique sole-striker tooth of the top
  gear in its own window, hence the shallow start); P6 (the quadratic-residue bar on which gear
  may reach which offset, with the exact `+-1 mod 8` identification at offset 0); P7 (the square
  phase vector); P9 (the per-offset depth law); P10 (the forced square columns behind `q^2` and
  their absence ahead); and the depth profile's dip-plateau-spike shape with its two exact
  independent-gear values.
* **Prior art named in one line each, after the mechanism:** that the prime divisors of `n^2 - 2`
  are `+-1 (mod 8)` is Gauss's second supplement - what is new is that this is the machine's own
  description of the walk's first column, halving the gear set that can reach it; the per-offset
  opening density is a Hardy-Littlewood singular series for the pair `(x^2+6i-2, x^2+6i)` - a
  density, stopped at the stop line; `L` tracking `(ln q^2)^2/(12 C_2)` is the Hardy-Littlewood
  twin-gap null - a rate, stopped.

## 6. Verdict

**FACT, not a route.** Twelve exact rules (P1-P12) with the exception counts above; of the twelve
pre-registered predictions one is refuted as stated (T3) and one confirmed in the negative
direction it predicted (T5's second clause), the rest confirmed; and three nulls (T9, T10, T11)
confirmed as pre-registered.

The branch answers the owner's question in the machine's terms. The machine builds the path out
of two ingredients that are separately understood and proven: the anchor, which lays down two of
every five columns at positions fixed by `q mod 30` alone, and each gear above it, which lays
down two arithmetic progressions in the offset with difference `g`, separation `d_g` and a phase
that is a square modulo `g`. Every feature of the path's shape this branch could measure - the
start slot, the shallow first column, the depth profile, the word's letters and transitions, the
hop spacings, the two ends, the forced columns behind - falls out of those two, with exactly one
proven pairwise law (neighbour-of-hit) needed for the far end. Nothing needs a third gear.

What needs a third gear is only the length, and the length is the root question. So this branch,
like every other under R2/R3, produces position and shape facts and stops at the same statement.
Its one lever-shaped item is P7 with the P3/T10 measurement beside it: the phase vector of the
walk is confined to a set of density `2^{-pi(q)}` and the covering length does not notice. That
is either the reason the walk is short (and then it is a route) or a coincidence to be measured
out on the counterfactual family. The measurement is stated above as the next test.

## 7. Dead ends (do not re-enter)

* The section-level Fourier spectrum and autocorrelation of the blocked string: the peaks are the
  gear lines `j/g`, the largest is gear 5's `2/5` at every `q`, and the position of `q^2` is not
  distinguished by any local density or spectral statistic (percentiles 0.37-0.50). This is
  `walk-transform-pole-identity` and `golden-spectral-gap` seen in a finite window.
* Any relation between consecutive levels of the chain other than the frame: refuted again
  (correlations 0.09-0.11, the class map not a function, the gate not inherited) - the same null
  as R2.a's P2.3-P2.5.
* The path length as a distinguishing statistic: `L` is the twin-gap null to within 2% from
  `q = 200` and sits at percentile 0.51-0.53 among the section's runs, the other teeth's walks and
  random blocked columns. A rate.
* The forward and backward words as reflections of one another: 0 of 44, correlation of the two
  lengths 0.0589. The local mirror is not the period mirror.
* The per-offset opening density as a finding: it is the Hardy-Littlewood singular series of the
  offset, stopped at the stop line once the admissible-set mechanism was named.

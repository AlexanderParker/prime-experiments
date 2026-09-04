# Branch W.a - THE PATH TAKEN APART

Parent: node R2.a (the machine feeds on itself, research/proof/self_feeding.md), whose rules
W1-W4 describe the walk from `q^2` from the top gear's side only. The observation that spawned
this branch: W1 says the top gear strikes the walk once and then is inert for a whole tooth arc,
so **the walk is made entirely by the old gears** - and nothing on the record says what those
gears do along it. The lead being followed (the owner's): the walk from `q^2` under `{5..q}`
starts on a tooth of the top gear, is struck by it exactly once, and lands on a twin before that
gear's next tooth at every prime from 59 to 4999 (one exception, 53).

The owner's organising question for this branch, verbatim:

> "How does the machine build the path, what parts of the machine contribute to the shape of the
> path, are those parts of the machine measured individually and understood and proven, and if
> they are, then work on figuring out how to prove their interactions' involvement in the
> path-shape."

Scripts: `research/anchor235/r38/pa_*.py`. Result outputs (untracked):
`research/anchor235/r38/results/pa_*.txt`. Every number this document relies on is written into
the document.

---

## 0. Pre-registered (written before any computation of this branch)

### 0.1 The object, stated exactly

Fix a prime `q >= 5`. The machine is `M = {5..q}` (gears = the primes from 5 to `q`). Column `k`
carries the pair `(6k-1, 6k+1)`. Gear `g` **strikes** `k` iff `g | 6k-1` or `g | 6k+1`, i.e. iff
`k = +-u_g (mod g)` with `u_g = 6^{-1} mod g` (the tooth rule, docs/proofs/02, kernel).

`k_0 = (q^2-1)/6` is the column carrying `q^2`. The **path** is the sequence of columns
`k_0, k_0+1, ..., k_0+L`, where `k_0+L` is the first opening at or after `k_0` (the **landing**);
`L` is the **walk length**. Columns are indexed along the path by the **offset** `i`, so column
`k_0 + i` carries the members `q^2 + 6i - 2` (lower) and `q^2 + 6i` (upper).

**The identity this branch starts from.** Gear `g` strikes the column at offset `i` iff

```
    g | q^2 + 6i - 2      (lower member)      <=>   i = (2 - q^2) u_g   (mod g)
    g | q^2 + 6i          (upper member)      <=>   i = (- q^2) u_g     (mod g)
```

so each gear contributes exactly **two arithmetic progressions in `i`, of common difference `g`,
whose two phases are `d_g = 2 u_g` apart and are fixed by `q^2 mod g`, i.e. by `q mod g`**. The
path is therefore the least-prime-factor structure of the two linear-in-`i` sequences
`q^2 + 6i - 2` and `q^2 + 6i`, and the whole path is a function of the residue vector
`(q mod g)_{g <= q}` alone.

### 0.2 Part inventory (the owner's first question), with status

The parts of the machine that can bear on the path, each with what is proved about it:

| part | statement | status |
|---|---|---|
| **A. the tooth rule** | gear `g`'s action on the path is the two progressions above, phases from `q mod g` | **PROVED**, kernel (`TwoTeeth.kill_period`, docs/proofs/02) |
| **B. the two arcs** | the two phases are `d_g = 2u_g mod g` apart; arcs `d_g` and `g - d_g`; teeth never adjacent | **PROVED**, kernel (`TwoTeeth.kill_spacing`, `AnchorChain.neighbour_of_hit`) |
| **C. the start phase (W1)** | `k_0 = -u_q (mod q)`: the path starts ON a tooth of the top gear, whose next strike is `d = 2u_q mod q` offsets on | **PROVED** (elementary, self_feeding N1); the consequence *`L < d`* is **MEASURED** (1 exception in 667, `q = 53`) |
| **D. the square gate (W2)** | the deepest hopping layer is `q` iff `q^2 - 2` is prime | **PROVED given `L < d`** (self_feeding N2); rests on C's measured half |
| **E. the anchor's fixed pattern** | gear 5's phases on the path, computed from `q^2 mod 5 in {1,4}` | **NEW here** (0.4 P2/P3); the gear-5 lock it specialises is PROVED (node 5g) |
| **F. the transfer gears** | at offset 0 and birth-offset `+-1` the admissible carriers are exactly `{7,17,31}` | **PROVED** (self_feeding N3, elementary congruence) |
| **G. the layer/shadow law** | a gear exposes nothing below its own square; the effective machine at column `k` is `{5..sqrt(6k+1)}` | **PROVED**, kernel (`Gear.R_eq_zero_of_below_sq`, docs/proofs/15) |
| **H. the gear-5 lock** | every maximal blocked stretch has gear 5 at its coverage-maximal phase | **PROVED** (node 5g, five-case proof) |
| **I. the merge law** | a gap of `M+g` is a fusion of lower gaps whose interior openings `g` strikes | **PROVED**, kernel (`MergeLaw.interior_gap_mod`, docs/proofs/05) |
| **J. the chain law** | `g` strikes two consecutive lower openings iff their gap is `0` or `+-d_g (mod g)` | **PROVED**, kernel (`AnchorChain.chain_law`) |
| **K. the character constraint** | `q^2` is a quadratic residue mod every gear `g < q`, so a gear's phase pair on the path is not free | **NEW here** (0.4 P1/P4); the underlying fact (prime divisors of a quadratic lie in half the classes) is classical |

What is **not** on the record for any of these: what they do *jointly* along one path. That is
this branch.

### 0.3 What would count as a rule

As node R2.a: (i) a statement about positions or residues, not a rate; (ii) an exact exception
count over a stated range; (iii) uniform in `q`. A density, a fitted curve or an average is not a
rule. Restating the tooth rule, the hit law, the chain law, the merge law, the gear-5 lock,
W1-W4, the layer law or the Hardy-Littlewood count is **not** a finding: it is noted in one line
and the sub-question stops. In particular, if a sub-question reduces to "the prime divisors of
`x^2 - 2` are `+-1 mod 8`" or any other quadratic-reciprocity class statement, it is named in one
line as classical and only its *machine* consequence is pursued.

### 0.4 Predictions, with numbers, and what refutes each

**The parts.**

- **P1 (the offset character law).** Whether gear `g` can strike offset `i` *at all* is
  independent of `q`: it needs `-6i` or `2 - 6i` to be a quadratic residue mod `g` (or `= 0`).
  Prediction: over every prime `q` in `5..20000`, no gear ever strikes an offset for which both
  are quadratic non-residues. REFUTED by one such strike.
- **P2 (the anchor's two patterns).** Gear 5 strikes the path at offsets `= 1, 4 (mod 5)` when
  `q = +-1 (mod 5)` and at offsets `= 1, 3 (mod 5)` when `q = +-2 (mod 5)`; in particular gear 5
  strikes offset 1 of **every** path and never an offset `= 0` or `2 (mod 5)`. REFUTED by one
  exception; predicted 0 exceptions and `L >= 2` for every `q >= 7`.
- **P3 (the length residue law).** Hence `L mod 5 in {0,2,3}` when `q = +-1 (mod 5)` and
  `L mod 5 in {0,2,4}` when `q = +-2 (mod 5)`. REFUTED by one `q` outside its class set.
- **P4 (the first column's strikers).** Every gear other than `q` striking offset 0 is
  `= +-1 (mod 8)`. Classical (divisors of `x^2-2`); recorded as a gate on the code, one line.

**The path's shape.**

- **P5 (depth is an arithmetic function of the offset, not of `q`).** Define
  `lambda(i) = sum_g 2 chi_g(i)/(g-1)` with `chi_g(i)` = the number of the two targets
  `-6i, 2-6i` that are nonzero quadratic residues mod `g` (`chi = 2` if both). Prediction: the
  mean strike depth at offset `i`, averaged over all `q` whose path reaches offset `i`, is
  ordered by `lambda(i)`: Spearman correlation `> 0.9` over offsets `1..60`, and the offsets that
  are most often landings are those of smallest `lambda`. REFUTED by correlation `< 0.5`.
- **P6 (the path is a tail, not a whole stretch).** The maximal blocked stretch containing the
  path extends to the left of `k_0` at more than 90% of `q`; predicted median left extension
  `>= 2`. REFUTED if `k_0 - 1` is open at more than 10% of `q`.
- **P7 (the top gear's rank in the bucket vector).** At the landing, the top gear's distance to
  its next strike (`d - L`) is in the top decile of all gears' distances at more than 90% of `q`.
  REFUTED below 50%.
- **P8 (sensitivity = sole strikers).** Re-phasing one gear at a time: `L` can be **shortened**
  only by re-phasing a gear that is the sole striker of some column of the path, and every gear
  of the machine (including gears that strike nothing on the path) has a phase that **lengthens**
  `L`. Predicted 0 exceptions to the first half. REFUTED by a gear that is not a sole striker
  whose re-phasing shortens `L`.
- **P9 (the top gear never fuses inside its own walk).** In the layer nest of the path, the
  largest gear that removes an interior survivor is `< q` at every `q > 53`; equivalently the top
  gear's only column on the path is offset 0. This is W1 read in the layer language; recorded
  with its exception count and not counted as new.
- **P10 (the path is a fusion of ordinary lower gaps).** For every layer `g <= 31` whose record
  `F_g` is known, the largest gap of the layer-`g` survivor set inside the path is `< F_g`:
  no lower machine's record sits inside the walk. REFUTED by one layer at one `q` reaching `F_g`.
- **P11 (the walk from a square is not a typical tooth-start walk).** Comparing `L(q^2)` with the
  walk lengths from the other teeth of gear `q` in the same window ("tooth-start record"):
  predicted the square start is systematically **longer** (median percentile `> 0.5`), because
  offset 1 is forced blocked by gear 5 at a square start and is free at a general tooth start.
  REFUTED by median percentile `<= 0.5`.

**The interactions (the owner's second question).**

- **P12 (order-one is not enough).** No function of the individual gears' phases taken one at a
  time (i.e. of the multiset of first-strike offsets `b_g`) determines `L`: there exist two
  primes `q1 != q2` whose paths agree in the whole first-strike multiset but differ in `L`.
  Predicted TRUE (that is what "interaction" means); refuted by finding the multiset determines
  `L` over the sweep.
- **P13 (which interactions the path uses).** Count, on every path, how many columns are struck
  by exactly one gear (order-1 columns, decided by A alone), by two, by three; and how many
  *consecutive* pairs of columns are struck by the same gear at chain-law spacing (order-2
  interactions of the proved kind). Prediction: more than half of all path columns have depth 1,
  and the landing is decided by a coincidence of order = the number of distinct sole strikers.
  This is a measurement, not a rule; it is what tells us which interaction law is needed.
- **P14 (the first unproven interaction).** Pre-registered guess, to be confirmed or corrected by
  the data: the first unproven interaction on the way from the parts to `L` is the **joint
  covering statement at order `>= 2` between gear 5's forced pattern (part E) and the free gears'
  phases** - i.e. "the offsets left open by gear 5 (three classes mod 5) are all struck by the
  remaining gears for `L-1` steps". Everything below that (each gear's own progressions, the two
  arcs, the gear-5 pattern, the character constraint) is proved or provable in a line; everything
  above it (`L < d`, the square gate's second half, the landing being a twin) needs it.

### 0.5 Scorecard

| # | prediction | verdict and evidence |
|---|---|---|
| P1 | offset character law, 0 exceptions | **CONFIRMED**, 0 hits at a non-residue offset (q = 5..499, all gears, offsets < 80) |
| P2 | gear 5's two patterns, 0 exceptions | **CONFIRMED**, 0 exceptions in 2,259 paths; grew into N-W1 |
| P3 | `L mod 5` law, 0 exceptions | **CONFIRMED**, 0 exceptions in 2,259 paths; sharpened to N-W3 (mod 35, 0 in 2,258) |
| P4 | first column's strikers `+-1 mod 8` | **CONFIRMED**, 0 exceptions in 2,260 paths; classical, filed as a gate |
| P5 | depth ordered by `lambda(i)` | **CONFIRMED and sharpened**: Spearman 0.9985, values within 2%; 0 landings on the 8 highest-lambda offsets, 500 on the 8 lowest |
| P6 | the path is a tail | **CONFIRMED**: k_0 - 1 open at only 73 of 2,260 (3.2%); median left extension 26; k_0 at median 0.487 of its stretch |
| P7 | top gear's bucket rank | **CONFIRMED**: farthest decile at 91.0%, single farthest at 49.4% |
| P8 | sensitivity = sole strikers | **HALF CONFIRMED, half REFUTED**: shortening implies sole striker, 0 of 13,861 exceptions; but 515 cells have NO lengthening phase - all of them sole strikers (0 of 155 not sole at the recount). Became N-W8 |
| P9 | top gear does not fuse inside | **CONFIRMED and strengthened to N-W5**: 0 of 2,260, including q = 53 |
| P10 | no lower record inside | **REFUTED as stated, CONFIRMED restricted**: gears 5 and 7 reach their own record inside the path; no layer above 7 does. Same restriction R3.h reports for records and window stretches |
| P11 | square start longer than tooth start | **CONFIRMED**: median percentile 0.600, mean ratio 1.243, 0 of 412 square starts of length 1 against 4.80% of tooth starts. Became N-W6 |
| P12 | order-one insufficient | **BADLY POSED, replaced**: the first-strike multiset is injective across q, so the test is vacuous; the well-posed version is the minimum blocking set (median 9, max 43), reported in 2.5 and section 5 |
| P13 | interaction census | **MEASURED, prediction wrong**: only 13.0% of path columns have depth 1, not more than half (mean depth 3.299); gears striking two or more columns are a median 2.9% of the machine |
| P14 | the first unproven interaction | **CORRECTED**: the first unproven interaction is not the gear-5/free-gears order-2 statement (that is proved, N-W1 plus CRT) but the unbounded-order joint covering statement L < d - see section 5 |

---

*(Sections 1 onwards - Setup, Results, Candidate rules, Mechanism, What is new, Verdict, Dead
ends - are written after the computation.)*

---

## 1. Setup (exact ranges)

No sampling anywhere except where stated. Scripts in `research/anchor235/r38/`.

| object | range | script |
|---|---|---|
| the path (every striker of every column), word, depth, layer nest, containing stretch, bucket vector at the landing, the character gates | **every prime `q` = 5..20000 (2,260 paths, 88,677 path columns)** | `pa_path.py` |
| depth at offsets `0..79` with **no path selection** (every `q`, whether or not the path reaches the offset) | same 2,260 primes | `pa_path.py` |
| the offset character gate (every gear, every offset `< 80`) | `q` = 5..499 | `pa_path.py` |
| bucket vector at **every step**, the landing's flanks, the interaction census, the exact minimum blocking set of the path | `q` = 5..4999 (667 paths, 19,410 steps) | `pa_sens.py` |
| one-gear re-phasing, FREE and REAL | `q` = 5..997 (166 paths, **13,861 gear cells**) | `pa_sens.py` |
| tooth-start comparison (up to 150 tooth columns of gear `q` in its own window) | `q` = 5..2999, 412 machines with `>= 20` starts, **57,125 tooth-start walks** | `pa_torus.py` |
| the six `{5,7}` skeletons | exact, all six classes | `pa_skel.py` |
| `L` by residue class; the `L mod 35` gate | every prime 5..20000 | `pa_class.py` |
| which size of gear the path needs | every prime 5..20000 | `pa_bound.py` |

Landings: all 2,260 verified twin pairs by deterministic Miller-Rabin (`q^2 + 6L` up to
`4.0e8`). Walk length over the sweep: min 1 (only `q = 5`), median 25, max **402 at `q = 8699`**
(then 383 at 15,053 and 265 at 4,637).

## 2. Results

### 2.1 The parts, one at a time (the owner's first question)

**A. Each gear is two arithmetic progressions in the offset.** Proved, kernel (docs/proofs/02).
Gear `g` strikes offset `i` iff `i = (2-q^2) u_g` or `i = (-q^2) u_g (mod g)`, the two classes
`d_g = 2u_g` apart. Every number below is computed from this and nothing else.

**K. The character constraint (new).** Because `q^2` is a quadratic residue mod every gear
`g < q`, gear `g`'s phase pair is drawn from only `(g-1)/2` of the `g` possibilities. Equivalently
gear `g` can strike offset `i` at all only if `-6i` or `2 - 6i` is a quadratic residue mod `g`.
The gate over `q = 5..499`, every gear, every offset below 80: **0 hits at an offset whose two
targets are both non-residues.** Two consequences measured over the whole sweep:

* every gear other than `q` striking offset 0 is `+-1 (mod 8)` (divisors of `q^2 - 2`) -
  **0 exceptions in 2,260 paths** (classical; recorded as a gate, not a finding);
* the **depth at an offset is an arithmetic function of the offset alone**. With
  `lambda(i) = sum_g 2 chi_g(i)/(g-1)`, `chi_g(i)` = how many of `-6i, 2-6i` are nonzero residues
  mod `g`, the measured mean depth at offset `i` over all 2,260 paths matches `lambda(i)` to
  Spearman **0.9985** over `i = 1..79` and to 1-2% in value:

| offset `i` | 1 | 2 | 5 | 6 | 10 | 11 | 16 | 17 | 21 |
|---|---|---|---|---|---|---|---|---|---|
| `lambda(i)` | 3.649 | 3.233 | 2.730 | 3.694 | 2.066 | 3.753 | 4.196 | 2.208 | 3.404 |
| measured mean depth | 3.650 | 3.277 | 2.751 | 3.731 | 2.115 | 3.774 | 4.214 | 2.268 | 3.452 |
| times the landing | 1 | 47 | 89 | **0** | **183** | **0** | **0** | **107** | **0** |

The eight offsets of lowest `lambda` (10, 17, 77, 32, 55, 52, 75, 47) take **500 of the 2,260
landings (22.1%)**; the eight of highest `lambda` (16, 46, 71, 51, 41, 56, 79, 31) take **1**.

**E. The anchor's fixed pattern (new).** `q^2 = 1` or `4 (mod 5)`, so gear 5's two offset classes
are `{1,4}` or `{1,3}` mod 5 - never anything else, and **offset 1 is in both**. Over the 2,259
paths with `q >= 7`: gear 5 misses offset 1 **0 times**; its class set differs from the predicted
one **0 times**; `L mod 5` falls outside the predicted three classes **0 times**.

| `q^2 mod 5` | paths | `L mod 5` histogram |
|---|---|---|
| 1 (`q = +-1 mod 5`) | 1,122 | 0: 380, 2: 370, 3: 372 - **never 1, never 4** |
| 4 (`q = +-2 mod 5`) | 1,137 | 0: 399, 2: 361, 4: 377 - **never 1, never 3** |

Adding gear 7 (`q^2 = 1, 2` or `4 mod 7`) gives exactly **six** possible `{5,7}` skeletons, one
per quadratic residue class mod 35. Each leaves the same 15 of every 35 offsets open (`3/5 x 5/7`);
what differs is where they sit:

| `q^2 mod 35` | 5's offsets | 7's offsets | open offsets in `1..35` |
|---|---|---|---|
| 1 | 1, 4 | 1, 6 | 2 3 5 7 10 12 17 18 23 25 28 30 32 33 35 |
| 4 | 1, 3 | 2, 4 | 5 7 10 12 14 15 17 19 20 22 24 27 29 34 35 |
| 9 | 1, 3 | 0, 2 | 4 5 10 12 15 17 19 20 22 24 25 27 29 32 34 |
| 11 | 1, 4 | 2, 4 | 3 5 7 8 10 12 13 15 17 20 22 27 28 33 35 |
| 16 | 1, 4 | 0, 2 | 3 5 8 10 12 13 15 17 18 20 22 25 27 32 33 |
| 29 | 1, 3 | 1, 6 | 2 4 5 7 9 10 12 14 17 19 24 25 30 32 35 |

`L mod 35` lies in its class's 15-element set at **all 2,258 primes `q >= 11`, 0 exceptions**.

**C/D. The top gear (W1, W2) - and a strengthening.** The top gear is the smallest striker of
offset 0 at 451 of 2,260 paths, exactly the square-gate cases (`q^2-2` prime, 20.0%). New: the top
gear is the smallest striker of **no other path column, at any `q` in the sweep - 0 exceptions in
2,260 paths, including `q = 53`**, where it does strike a second column (offset 18) but is not
that column's smallest striker. So in the layer nest the top gear removes no interior survivor
anywhere: the strengthening survives W1's only exception.

**G. Which size of gear the path needs.** 9,690 of the 88,677 path columns (10.93%) have their
smallest striker above `sqrt(q)`; **1,995 of 2,260 paths (88.3%) contain at least one such
column** (median 3, max 43). The walk under gears `<= sqrt(q)` alone equals the true walk at only
463 of 2,258 primes (median 7 against 25); under gears `<= q/2` at 1,664 of 2,258. **No truncation
of the machine reproduces the path.**

### 2.2 The blocker sequence (item 1)

Over 88,677 path columns of 2,260 paths:

* total strikes 292,540, i.e. **3.299 strikes per path column** against `L`;
* depth histogram 1: 11,559 (13.0%), 2: 20,970, 3: 19,566, 4: 16,061, 5: 11,627, 6: 5,505,
  7: 2,350, 8: 876, 9: 154, 10: 9 - depth 1 is the exception, not the rule;
* smallest-striker letters: 5 (35,694 = 40.3%), 7 (15,256), 11 (7,007), 13 (4,723), 17 (3,291),
  19 (2,455), 23 (1,980), 29 (1,268), 31 (1,241), 37 (1,051);
* distinct letters per path: min 1, median 11, max 61; distinct strikers of any depth: median 53,
  max 458; sole strikers: median 3, max 36;
* which member is struck: lower only 12,657 (14.3%), upper only 13,038 (14.7%), **both members
  struck by different gears 62,982 (71.0%)**.

### 2.3 The bucket vector (item 2)

State: `b_g(i)` = offsets from the current column to gear `g`'s next strike. Over 19,410 steps of
667 paths (`q <= 4999`):

* the **nearest-tooth gear** is 5 at 42.5% of steps, 7 at 16.5%, 5 or 7 at 59.0%; then 11 (8.3%),
  13 (5.6%), 17 (3.5%), 19 (2.9%), 23 (2.2%);
* the distance to the nearest tooth is **1 at 18,734 steps, 2 at 639, 3 at 33, 4 at 3** - the
  bucket vector's smallest positive entry never exceeds 4 anywhere on any path;
* the top gear's normalised rank by bucket distance, averaged along its own path, has median
  0.1099 (0 = farthest); it is the single farthest gear at every step of only **1 of 667** paths;
* **at the landing** the vector has no zero by definition; its smallest entry is 1 at 2,239 of
  2,260 paths and 2 at the remaining 21, never more, and every distance-2 landing is `= 2 (mod 5)`
  (gear 5 owns a flank, N6/5g; distance `>= 3` is forbidden by the alignment law, docs/proofs/04 -
  noted in one line, not pursued);
* the stopper gear at the landing is 5 at 1,534 of 2,260, 7 at 258, 11 at 101, 13 at 82;
* the top gear's own entry at the landing (`d - L` whenever `L < d`) has median 4,034 and
  minimum 1; the
  top gear is the farthest of all gears at the landing at 49.4% of paths and in the farthest
  decile at 91.0%.

### 2.4 Determinism and sensitivity (item 3)

Re-phasing one gear at a time over 13,861 gear cells (`q <= 997`, 166 paths):

| statement | count | exceptions |
|---|---|---|
| every gear of the machine has **some** phase that changes `L` | 13,861 | 0 (sensitive fraction of the machine: median 1.0000) |
| re-phasing gear `g` can **shorten** `L` only if `g` is a sole striker of a path column | 13,861 | **0** |
| gears that can shorten `L`: median 3.4% of the machine | 166 paths | - |
| some phase of `g` **lengthens** `L` | 13,346 of 13,861 | 515 cells fail |
| every failing cell is a sole striker of the path (recount at `q <= 300`) | 155 cells | **0 not sole** |

The 515 exceptions have a clean mechanism: a gear that is the sole striker of three or more path
columns in three distinct classes mod `g` cannot both keep them blocked and block the landing with
only two teeth, so its maximum over phases is `L` itself.

**FREE against REAL re-phasing** (the character constraint priced): moving the phase to any of the
`g` classes versus moving `q mod g` to any of the `g-1` nonzero classes.

| quantity | FREE | REAL | agree |
|---|---|---|---|
| min `L` over phases, mean over cells | 18.96 | 18.99 | 13,815 of 13,861 |
| max `L` over phases, mean over cells | **39.18** | **33.93** | 10,333 of 13,861 |

So the character constraint costs the machine nothing at the bottom and **13.4% of the reachable
maximum walk length at the top**: a real prime cannot put its gears in the phases that would make
the longest walks.

### 2.5 The layer nest (item 4)

The path's blocked run decomposed as in `ends_or_middles.md`:

* the **fusion count** (pieces joined by the largest gear that removes an interior survivor) is
  **2 at 2,234 paths and 3 at 25**, never 4 or more - the same shape the window's longest stretches
  have (R3.h), reached here from the walk side;
* the last fusion is never done by the top gear (2.1 above, 0 of 2,260);
* the layers at which the survivor set's largest gap reaches the lower machine's own record `F_g`
  are gears 5 and 7 only, exactly as R3.h reports for records and window stretches - noted, not
  re-derived;
* the **minimum blocking set** of the path (exact branch and bound for `L <= 40`, greedy above,
  667 paths): min 1, median **9**, max **43**; `mincov / L` median 0.45; it equals the set of sole
  strikers at only 7 of 667 paths (median 3 sole strikers, median excess 5).

### 2.6 The path inside its stretch (item 4, P6)

`k_0` is always blocked (by `q`), so the path is the **right tail** of a maximal blocked stretch:

* `k_0 - 1` is open - i.e. the path IS the whole stretch - at **73 of 2,260 paths (3.2%)**;
* left extension `e`: median 26, and `e >= L` at 49.2% of paths;
* `k_0`'s position in its own stretch, `e/(e+L)`: median **0.487**. The column carrying `q^2` sits
  at the middle of the blocked run it falls in, like an ordinary blocked column;
* containing stretch length: median 65, max 444 (78 paths hit the probe wall at `e = 128`).

### 2.7 The landing (item 5)

Over 667 paths (`q <= 4999`): the last blocked column has median depth 3 and smallest striker 5 at
438 of 667; the column past the landing has median depth 3 and smallest striker 5 at 447 of 667.
The landing's residue vector is the bucket vector of 2.3: no zero entry, smallest entry 1 (or 2 at
a prime quadruplet), the top gear near the far end.

### 2.8 The path on the torus (item 6)

**The square start against the other teeth of the same gear.** For each machine, walks under
`{5..q}` started from up to 150 tooth columns of gear `q` inside its own window, 57,125 walks over
412 machines:

| quantity | value |
|---|---|
| percentile of `L(q^2)` among the tooth-start walks | median **0.600**, mean 0.588, min 0.048, max 1.000 |
| `L(q^2)` above the tooth-start median | 241 of 412 machines (58.5%) |
| mean `L(q^2)` against mean tooth-start `L` | 24.83 against 19.98, **ratio 1.243** |
| walks of length 1 | 2,744 of 57,125 tooth starts (4.80%); **0 of 412 square starts** |

**The residue tabulation.** Over all 2,260 primes, by `q^2 mod 35` (the six skeletons):

| class | n | median `L` | mean `L` | max |
|---|---|---|---|---|
| 1 | 373 | 30 | 39.33 | 383 |
| 4 | 375 | 27 | 41.34 | 300 |
| 9 | 378 | 27 | 42.75 | 279 |
| 11 | 372 | 27 | 40.01 | 402 |
| 16 | 373 | 22 | 35.89 | 290 |
| 29 | 380 | 24 | 36.93 | 240 |

A spread of 1.36x in the median and 1.19x in the mean, with **no rule**: the ordering does not
follow the first open offset of the skeleton (class 1 has it at 2 and the longest median; class 16
has it at 3 and the shortest), nor the count of early open offsets. Recorded as a measured
tendency, not a rule, per the project's standing direction.

### 2.9 Six full paths

Printed in full (offset, every striker with the member it strikes, smallest striker, depth) in
`results/pa_path.txt`: `q = 53, 59, 137, 2593, 4637, 8699`. Summary lines:

| `q` | `k_0` | `L` | landing twin | left ext. `e` | stretch | position of `q^2` | sole strikers | stopper |
|---|---|---|---|---|---|---|---|---|
| 53 | 468 | 27 | 2969 \| 2971 | 0 | 27 | 0.000 | 5,7,11,13,17,19,29,41,43 | gear 5 at 1 |
| 59 | 580 | 8 | 3527 \| 3529 | 1 | 9 | 0.111 | 7, 13 | gear 5 at 1 |
| 137 | 3128 | 24 | 18911 \| 18913 | 37 | 61 | 0.607 | 5,11,41,83,109,113 | gear 5 at 2 |
| 2593 | 1120608 | 187 | 6724769 \| 6724771 | 0 | 187 | 0.000 | 23 gears, largest 1721 | gear 5 at 1 |
| 4637 | 3583628 | 265 | 21503357 \| 21503359 | 65 | 330 | 0.197 | 27 gears, largest 4451 | gear 5 at 1 |
| 8699 | 12612100 | 402 | 75675011 \| 75675013 | 42 | 444 | 0.095 | (longest walk in range) | - |

The first eight columns of `q = 59` read: `7L 59U | 5L 11U 17L 41L | 7U | 13L | 5U 31L | 11L 29L |
5L 19L 37L | 7L 13U`, word `7 5 7 13 5 11 5 7`. The word of `q = 137` is
`7 5 7 5 19 11 5 7 5 7 19 5 83 5 7 109 5 113 5 23 11 5 41 5` - gear 5 every second or third
column throughout, gear 7 filling, and four single big gears (83, 109, 113, 41) each holding one
column alone.

## 3. Candidate rules, with exception counts

**N-W1 (the anchor's two path patterns).** For every prime `q >= 7`, gear 5 strikes the path at
the offsets `= 1` and `= 4 (mod 5)` when `q = +-1 (mod 5)`, and at `= 1` and `= 3 (mod 5)` when
`q = +-2 (mod 5)`; in particular it strikes **offset 1 of every path** and never an offset `= 0`
or `= 2 (mod 5)`. Exceptions over 2,259 paths: **0**.

**N-W2 (the length residue law).** Hence `L != 1 (mod 5)` always, and `L mod 5 in {0,2,3}` when
`q = +-1 (mod 5)`, `L mod 5 in {0,2,4}` when `q = +-2 (mod 5)`; and `L >= 2` for every `q >= 7`.
Exceptions over 2,259 paths: **0**.

**N-W3 (the fifteen-class law).** `L mod 35` lies in the 15-element set fixed by `q^2 mod 35`, one
of the six sets tabulated in 2.1. Exceptions over 2,258 primes `q >= 11`: **0**.

**N-W4 (the offset character law).** Whether gear `g` can strike offset `i` of the path is
independent of `q`: it requires `-6i` or `2 - 6i` to be a quadratic residue mod `g`. Exceptions
over `q = 5..499`, all gears, offsets `< 80`: **0**. Consequence: the mean depth at offset `i` is
the arithmetic function `lambda(i)` (Spearman 0.9985, values within 2%), and the landing histogram
follows it - **0 landings** on the eight highest-`lambda` offsets against 500 on the eight lowest.

**N-W5 (the top gear does not shape its own path).** The top gear is the smallest striker of no
column of its own path except offset 0; equivalently it removes no interior survivor in the layer
nest. Exceptions over 2,260 paths: **0**, *including* `q = 53`, the one exception to W1's `L < d`.
This is strictly stronger than N1/W1 and is not on the record.

**N-W6 (the square start is a long start).** The walk from `q^2` is longer than the walks from the
top gear's other teeth in the same window: median percentile 0.600 over 412 machines, mean ratio
1.243, and length-1 walks occur at 4.80% of tooth starts and at **none** of the square starts.

**N-W7 (the character constraint priced).** Over 13,861 gear cells the maximum walk length
reachable by re-phasing one gear freely is 39.18 on average; the maximum reachable by an actual
residue of `q` is 33.93. The teeth a real prime can present are half of those a counterfactual
machine can, and the half it cannot reach is the half that makes long walks.

**N-W8 (sensitivity is exactly the sole strikers).** Re-phasing one gear shortens `L` only if that
gear is a sole striker of a path column (0 of 13,861 cells otherwise), and a gear has no
lengthening phase only if it is a sole striker (0 of 155 cells otherwise). The first half is
prover A's L4 corollary (docs/proofs/19) read on the path - a gate, not a finding; the second half
and its two-teeth mechanism are new.

Restatements filed, not counted: stopper distance `<= 2` is the alignment law (docs/proofs/04);
"every gear other than `q` striking offset 0 is `+-1 mod 8`" is the classical class condition on
divisors of `x^2-2`; `T_all = q` iff the square gate is W2; the four-or-more fusion never occurring
is R3.h's window result met from the walk side; the smallest gear needed to reproduce the walk is
W2's top hop layer `T`.

## 4. Mechanism - how the machine builds the path

**Order one.** Each gear lays down two arithmetic progressions of common difference `g` in the
offset, phases `(2-q^2)u_g` and `-q^2 u_g`. That is the whole of each part, and it is proved.
Because `q^2` enters only as a square, each gear's phase is confined to `(g-1)/2` of `g` classes -
the sub-torus of density `prod (g-1)/2g` (0.008 over the first six gears alone). The path from a
square is therefore **not a generic walk**: N-W1 and the six skeletons are the visible bottom of
that constraint, N-W7 is its price at the top.

**The anchor.** Gear 5 is the only gear whose behaviour on the path is fully decided by one bit of
`q`. `q^2 = 1` or `4 (mod 5)` and the two targets at offset 1 are `-6 = 4` and `-4 = 1`, so
whichever of the two the square is, one of them is hit: **offset 1 is blocked for every `q`**. That
one line is N-W1, N-W2 and half of N-W6.

**The top gear.** `6k_0 = q^2 - 1 = -1 (mod q)`, so the path starts on a tooth and the next tooth
is `d = 2u_q` on (W1). What 2.1 adds is that even when the path does reach the second tooth
(`q = 53`), the top gear is not that column's smallest striker: for `q` to be the smallest striker
of a later column, that column's member must be `q m` with `m` free of primes below `q`, hence
`m >= q + 2`, which is `d` further on again. So the top gear cannot shape its own path at all -
its only structural act is to make offset 0 blocked, which is what puts the walk on a tooth in the
first place.

**The middle.** 71% of path columns are struck on both members by different gears, mean depth
3.30, only 13% of columns have a single striker. The nearest tooth in the bucket vector is never
further than 4 and is gear 5 or 7 at 59% of steps. The path is therefore not a thin chain of
coincidences but a thick cover: the machine has three strikers per column to spare and still stops
after 25 columns.

**Why it stops.** The landing is the first offset that every one of the `2 pi(q)` progressions
misses. The certificate that the columns before it are covered needs a median of **9** gears and up
to 43 (2.5), and 11% of the covered columns are covered only by a gear above `sqrt(q)`. So the
covering is genuinely high order and genuinely uses the large gears.

## 5. The interactions - the owner's second question

**Already proved, order two.** (i) the two-arc law: which pairs of columns one gear can strike
(`TwoTeeth.kill_spacing`); (ii) neighbour-of-hit: no gear strikes two adjacent columns
(`AnchorChain.neighbour_of_hit`); (iii) the chain law: `g` strikes two consecutive lower openings
iff their gap is `0` or `+-d_g (mod g)` (`AnchorChain.chain_law`); (iv) the merge law: a gap of
`M+g` is a fusion of lower gaps whose interior openings `g` strikes (`MergeLaw.interior_gap_mod`);
(v) the gear-5 lock (node 5g); (vi) tooth-sharing pinning; (vii) CRT independence of any two gears'
phases.

**Which of them the path actually uses, counted.** Over 667 paths: gears striking two or more path
columns - the only gears at which the two-arc and chain laws have anything to say - number median
**8**, max 79, i.e. a median of **2.9% of the machine**; every other gear contributes a single
column and is pure order one. The merge law is used exactly once per path: the layer nest is a
**two-piece fusion at 2,234 of 2,259 paths** and a three-piece fusion at 25, so one gear (never the
top gear) does the single junction. The gear-5 lock is used at every column (N-W1) and at the
landing (the stopper is at distance 1 unless the landing is `= 2 mod 5`).

**What an interaction law would have to say.** `L` is the first offset missed by all `2 pi(q)`
progressions simultaneously. An order-`k` law - a statement about `k` gears at a time - can certify
that a *given* set of `k` gears covers a run, but the run the path actually needs covered requires
a minimum of median 9 and up to 43 gears, and that number grows with `L`. Two measurements make
this sharp:

* the minimum blocking set of the path has median size 9 against median `L` 25 (`mincov/L = 0.45`)
  and is essentially never the sole-striker set (7 of 667);
* 88.3% of paths contain a column that only a gear above `sqrt(q)` blocks, so the covering cannot
  be reduced to a bounded initial segment of the machine.

**The first unproven interaction, precisely.** Everything of order one and order two on the list
above is proved. The first statement on the way from the parts to `L` that is not proved is the
**joint covering statement of unbounded order**:

> for every prime `q`, the `2 pi(q)` progressions do **not** cover a run of `d = 2u_q` consecutive
> offsets starting at offset 1.

That is exactly W1's measured half `L < d`, and it is the first step in the chain that no
composition of the proved pairwise laws reaches - the tree's own verdict on the counter ladder
(`docs/novel/cover-half-counter-ladder.md`: "no fixed-depth truncation of the inclusion-exclusion
counter and no exposure-only argument bounds `L` uniformly") is the same wall, met here from the
walk. Below it, the statements that ARE reachable from the parts are the ones this branch proves:
N-W1 to N-W3 (one gear's forced phase, hence forbidden lengths), N-W5 (the top gear is inert on its
own path, from the layer law), N-W4 (which gears can act where, from the character), N-W8 (which
gears can move the landing, from the two-teeth count). None of them bounds a length.

## 6. What is new

Checked against `docs/novel/README.md` (walk-tooth-frame, anchor-235-layer-laws,
cover-half-counter-ladder, corridor-law, golden-spectral-gap), `docs/proofs/`, and
`docs/proof-search/anchor-235.md` 9c-9g.

* **N-W1, N-W2, N-W3** - no located prior art. Section 9d gives the hit law and chain law for a
  general slot; nothing on the record says that the walk from a *square* has gear 5's phase decided
  by one bit of `q`, that offset 1 is blocked for every `q`, or that the walk *length* is confined
  to three classes mod 5 and fifteen mod 35.
* **N-W4** - the underlying fact (the prime divisors of a quadratic polynomial's values lie in half
  the classes) is classical and is named here in one line. What is new is its use as a *shape*
  statement: the depth profile along the path is a fixed arithmetic function of the offset, the
  same for every `q`, and the landing distribution follows it.
* **N-W5** - a strengthening of the recorded W1/N1 that survives W1's only exception; not on the
  record in any form.
* **N-W6, N-W7** - new. The comparison of the square start against the top gear's other teeth, and
  the price of the character constraint in reachable walk length, are not on the record.
* **N-W8** - first half is prover A's L4 corollary read on the path (a gate); the second half (no
  lengthening phase iff sole striker, with the two-teeth mechanism) is new.
* Not new, filed: stopper distance `<= 2` (alignment law); `+-1 mod 8` at offset 0 (classical);
  `T_all = q` iff square gate (W2); `T` = smallest machine reproducing the walk (W2); four-piece
  fusion never (R3.h); gears 5 and 7 reaching their own record inside (R3.h); the hop-rate figures
  (9c, rates).

## 7. Verdict

**FACT, not a route.** Eight exact statements about the path, four of them (N-W1, N-W2, N-W3,
N-W5) with zero exceptions over every prime to 20,000, and a complete answer to the owner's two
questions: the parts are order-one and proved (each gear = two progressions, phases from `q mod g`,
restricted to the quadratic-residue sub-torus), the interactions the path uses are counted (a
median of 8 gears strike twice, one gear does the single junction, gear 5 acts at every column),
and the first unproven interaction is named exactly - the unbounded-order joint covering statement
`L < d`, which is W1's measured half and is twin-Bertrand at scale `q/3`.

Toward the root: the branch adds no length lever, and it explains why not. The machine's parts are
individually complete and provable; what decides `L` is the simultaneous miss of `2 pi(q)`
progressions, whose certificate has median size 9 and unbounded growth, and 88% of paths need a
gear above `sqrt(q)` to be covered at all. Every statement this branch could prove from the parts
is a **position or forbidden-class statement** (which offsets a gear can act on, which lengths are
impossible, which gear is inert) - the same family as the gear-5 lock, the corridor law and the
slot rule. The one new thing of a different kind is N-W7: the character constraint is the first
measured way in which the *real* machine is provably weaker than its counterfactual family, and it
weakens it in the right direction (shorter maximal walks). That is a lead for the family work, not
a bound.

## 8. Dead ends (do not re-enter)

* **`L` from the residue class of `q` at any fixed modulus.** The six `{5,7}` skeletons give a
  1.36x spread in median `L` with no rule: the ordering follows neither the first open offset nor
  the count of early open offsets (2.8). This is escape-distance 1 again.
* **Truncating the machine.** The walk under gears `<= sqrt(q)` is the true walk at 463 of 2,258
  primes and under gears `<= q/2` at 1,664 of 2,258; 88.3% of paths have a column only a gear above
  `sqrt(q)` blocks. No bounded initial segment of the machine determines the path.
* **The top gear as a shaper.** N-W5 closes it: the top gear's only act is offset 0.
* **`k_0` as a distinguished position inside its stretch.** `k_0` sits at median 0.487 of the
  blocked run it falls in and the run reaches back a median of 26 columns; the walk from `q^2` is
  the ordinary right tail of an ordinary stretch, and nothing about the stretch is special.
* **Order-one prediction of `L`.** The minimum blocking set has median 9 and max 43 and is almost
  never the sole-striker set; no statistic of the individual gears' first strikes can carry `L`.

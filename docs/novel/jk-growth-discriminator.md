# Exact values of `j_k` for `k = 3, 4, 5`, and the k-axis as a substitute for the z-axis

**Status: SCRIPT-VERIFIED (exact values, each with a machine-verified witness and
an exhaustive infeasibility proof) + MEASURED (the growth statistics).**
Harvester lane, round 28, 2026-08-29.
Gate: `research/jk_growth.py` → **ALL ASSERTIONS GREEN**;
reference engine `research/jk_cover.py`; fast engine `rust2/src/bin/jkcov6.rs`;
driver `research/jk_run.py`.
Prior-art check: §6, dated 2026-08-29, first-hand.

---

## 1. What it is

Two things, one of which is data and one of which is a method.

**(i) THE FIRST EXACT VALUES OF `j_k` FOR `k >= 3`.** `j_k(P(z))` is the k-class
Jacobsthal function of `docs/novel/jk-family.md`: the largest gap between
consecutive `n` for which every `n + E_i` is coprime to the primorial `P(z)`,
maximised over admissible `k`-tuples `E`. `k = 1` is the ordinary Jacobsthal
function (OEIS A048670), `k = 2` is Ziller–Morack's `h_2` (OEIS A288815).
Round 27 evaluated `j_3` at `z = 3, 5, 7` by exhaustion and recorded that
`z = 11` "needs a real algorithm". This round built it.

| `z` | `j_1` (A048670) | `j_2` (A288815) | `j_3` | `j_4` | `j_5` |
|---|---|---|---|---|---|
| 3 | 4 | 6 | 6 | – | – |
| 5 | 6 | 18 | 24 | 30 | – |
| 7 | 10 | 30 | 78 | 150 | 180 |
| 11 | 14 | 66 | **180** | **420** | **930** |
| 13 | 22 | 150 | **306** | **1230** | **2070** |
| 17 | 26 | 192 | **612** | **2340** (r29) | **5490** |
| 19 | 34 | 258 | **972** | **3810** (r29) | – |
| 23 | 40 | 366 | **1398** (r29) | – | – |

*(The `z = 23` and the two `j_4` entries were added in round 29; see §9, which
also records that round 28's `j_3(23)` split run was invalid as an upper bound
and how it was reproved — twice, by two independent engines.)*

Bold entries are computed here and are, as far as the prior-art check reaches,
**the first exact values of `j_k` for `k >= 3` anywhere**. Each is exact in both
directions: a *witness* (an explicit set of `k` residue classes per prime,
re-verified by independent code inside the binary and again in Python) and an
*exhaustive proof* that one more position cannot be covered.

**(ii) THE METHOD: TRADE THE z-AXIS FOR THE k-AXIS.** Since round 24 this lane
has carried one falsification target: *one exact `h_2` beyond `p_n = 73`*,
because the two live growth readings

* **(A)** the parameter-free random-choice heuristic `j_k ~ z (log z)^k`, and
* **(B)** the layered Erdős–Rankin construction `(P2')`, a **theorem**,
  `j_k >> z (log z)^{2k-1}` up to `loglog` powers,

differ by `(log z)^{k-1}` — at `k = 2`, one log — so they only separate by
2.6–3.6× at `z = 151..251`. That number has not been bought: **A072753 and
A288815 have both carried exactly 21 terms since June 2017** (records read
first-hand 2026-08-29), and §5 prices the computation.

**The separation is `(log z)^{k-1}` and `k` is free.** At `k = 3` the models are
two logs apart, at `k = 5` four. So the family answers the same question with
numbers that cost seconds instead of a number nobody has computed in nine years.

---

## 2. Why it might be novel

* The exact values of `j_k` for `k >= 3` are new because the object is new
  (`jk-family.md` §2, prior-art-checked round 27 and again here).
* The *method* — using the sifting dimension as the experimental axis when the
  modulus axis is out of reach — is, as far as the check reaches, not something
  anyone has done for Jacobsthal-type functions, for the simple reason that the
  family had not been written down. It is a general move: any two models that
  differ by a power of `log` **in a parameter that indexes a family** can be
  separated along the family instead of along the size.
* The engine is an independent reimplementation: it reproduces **fourteen**
  published `A048670` values (`z = 3..47`) and **nine** published `A288815`
  values (`z = 2..31`, §5) by a *different algorithm* from the published ones
  (a bound-and-branch DFS on the covering restatement, against Ziller–Morack's
  portioned ILP and Hagedorn's slack-counting search). To the prior-art check's
  knowledge these are the **first independent verifications of the paired
  Jacobsthal values** since they were deposited.

---

## 3. The engine (proof status: exact, both directions)

**REDUCTION (proved, and the reason the search is affordable).** In the covering
restatement `j_k(P(z)) - 1 = ` the longest interval coverable with
`|S_p| <= min(k, p-1)` classes mod each `p <= z`, every prime `p <= k+1` has
`cap = p-1`: it kills all but one class, the survivors lie in a single class mod
`p`, and the problem rescales. With `D = prod_{p <= k+1} p`,

    j_k(P(z)) = D * (m + 1),
    m = the longest run [1, m] coverable by k NON-ZERO classes mod p
        for each prime  k+1 < p <= z.

`D = 2` at `k = 1` (this is Hagedorn's `h(n+1) = 2w(n) + 2`), `D = 6` at
`k = 2, 3` (this is Ziller–Morack's `h_2 = 6 ω_2 + 6`, i.e. A288815 = 6·A072753 + 6),
`D = 30` at `k = 4, 5`. **Class 0 is excluded** because a *maximal* covered run
has an uncovered position on each side; translating one of them to 0 forbids
class 0 at every prime. This is exactly the `a_i, b_i ∈ {1,...,p_i-1}`
normalisation of Ziller–Morack's own formulation, here derived rather than
assumed, and it generalises to every `k`.

**SEARCH.** Branch on *which prime covers the leftmost uncovered position*.
Every uncovered position has the same option set (a prime whose committed class
contained it would already have covered it), so the leftmost position is as good
a branching variable as any and gives the tree a prefix structure; the depth is
bounded by `sum_p min(k, p-1)`, since each branch commits one class.

**CANONICAL FORM (the symmetry break, and it is worth two orders of magnitude).**
Committing prime `p` at position `j` is rejected when an earlier commit
`(j', p')` has `j' ≡ j (mod p)` and `p' > p`: `p` was free at that time and the
same class covers `j'`, so the identical class set is reachable by committing
`p` first. Among the orderings that produce a given class set, exactly the
"always take the smallest available prime" ordering survives. This is
Ziller–Morack's RPA2 rule ("select the smallest prime if there are any at
choice") transported to a different search. Measured effect at `k = 1, z = 29`:
476,683 nodes → 3,801.

**BOUND (`v3`, the prefix-window capacity criterion).** For *every* prefix
`[j, x]` of the residual window, the number of still-uncovered positions must
not exceed the capacity of the free classes *restricted to that prefix*: for
each free prime `p` with `f_p` classes left, the `f_p` largest values of
`#{uncovered ≡ r (mod p)}, r ≠ 0`. Short prefixes are where large primes are
weakest (one position per class), so this is the *sliding* form of Hagedorn's
`m_i` criterion and it uses exact residual counts where his uses an a-priori
worst case. Computed in one pass with incremental residues, `O(span × #free)`.

**VALIDATION, four independent ways.**

1. The covering restatement is checked **against the definition** by brute force
   at `k = 1, 2, 3 × z = 2, 3, 5, 7` (`research/jk_cover.py` §A) — twelve cases,
   all equal.
2. Fourteen published `A048670` values reproduced exactly (`z = 3..47`).
3. Nine published `A288815`/`A072753` values reproduced exactly (`z = 2..31`).
4. Every witness is re-verified by code that shares nothing with the search, and
   a third engine (a SAT encoding decided by CaDiCaL, `jk_cover.sat_cover`)
   agrees with the DFS wherever both reach.

The canonical-form rule was the round's **named risk** (pre-registration PR6):
if unsound it would make values come out *too small*. Twenty-three published
values reproduced exactly is the check that would have caught it.

---

## 4. The measurement

Write `delta_k(z) = prod_{p<=z}(1 - min(k,p-1)/p)` (exact, no asymptotics),
`P(z)` the primorial, `N = delta_k P`. The **parameter-free** random-choice
model is the expected largest gap among `N` points thrown on a cycle of length
`P`:

    model_k(z) = (P/N) log N = log(N) / delta_k .

Put `R_k(z) = j_k(P(z)) / model_k(z)`. **Model (A) says `R_k` is constant in `z`;
model (B) forces it to grow like `(log z)^{k-1}`.**

**THE CALIBRATION.** `R_1` runs
`0.590, 0.471, 0.487, 0.411, 0.406, 0.376` over `z = 7..23` and then
`0.350, 0.367, 0.354, 0.342, ..., 0.361` all the way to `z = 113` — a small-`z`
transient and then **flat to within 4% over eighteen further values**. At `k = 1`
the two models coincide and the truth is known (Rankin/FGKT attain `z log z` up
to `loglog` powers). So `k = 1` measures the method's own bias, and it is ≈ 0.

**THE k = 2 SIGNAL.** `R_2` runs `0.791` at `z = 7` to `0.889` at `z = 73`
(`0.821 → 0.889` on the clean window `z = 23..73` where `R_1` is flat) — a real
**+8% drift** where model (A) needs 0% and model (B) needs **+37%**.

**THE CALIBRATED FRACTION.** With `Q_k = R_k/R_1` (which removes the transient)
and, on a window `[z_0, z_1]`,

    f_k = log(Q_k(z_1)/Q_k(z_0)) / ((k-1) · log(log z_1 / log z_0)),

`f_k = 0` under model (A) and `f_k = 1` under model (B) — **and under model (B)
`f_k` is the same at every `k`**. That equality is the `(k-1)` scaling, and it is
precisely what the family can test and the `k = 2` ladder alone cannot.

| window | `f_2` | `f_3` | `f_4` | `f_5` |
|---|---|---|---|---|
| 7..13 | 1.599 | −0.282 | −0.104 | −0.310 |
| 7..17 | 1.116 | 0.229 | – | 0.014 |
| 7..19 | 0.882 | 0.251 | – | – |
| 23..73 (clean, `k=2` only) | **0.257** | – | – | – |
| 7..73 (full published range) | 0.720 | – | – | – |

**READING.** The entries are neither 0 nor 1, and — the point — **they are not
equal across `k`**: on every matched window `f` falls steeply as `k` rises. The
extra logs that model (B)'s shape requires are not appearing at the rate `(k-1)`
demands.

**THE SECOND, INDEPENDENT FORM.** Fit `j_k ~ z (log z)^{a_k}` by least squares of
`log(j_k/z)` on `log log z` and put `e_k = a_k - k`. Model (A) needs `e_k = 0`;
model (B) needs `e_k >= k-1`.

| `k` | `a_k` | `e_k` | model (B) needs | verdict |
|---|---|---|---|---|
| 1 | 0.921 | −0.079 | 0 | **calibration** |
| 2 | 2.614 | 0.614 | 1 | far below (B) |
| 3 | 3.556 | 0.556 | 2 | far below (B) |
| 4 | 4.757 | 0.757 | 3 | far below (B) |
| 5 | 6.724 | 1.724 | 4 | far below (B) |

**The excess over model (A) is REAL — `e_2..e_4` sit at 0.5–0.8 against a
calibration bias of −0.08 — and it DOES NOT GROW WITH `k`.** Model (B)'s shape
requires it to grow by one per unit `k`.

**THE HONEST CAVEAT, and it is load-bearing.** `(P2')` carries a `C^k/B^{2k}`
factor which is about `0.03` at `z = 73, k = 2`; the construction "does not exist
below `log z ~ 300`" (harvester r26 §10f). So **none of this refutes the
theorem** — (B) remains a proved lower bound whose regime starts far above any
`z` reached here. What is measured is the shape of the *truth* on the range where
exact values exist, and on that range the truth looks like model (A) plus a
constant excess, uniformly in `k`. The lane's own standing lesson applies to this
document too: **model claims expire like citations.**

---

## 5. The price of the z-axis, and what it buys instead

Exhaustive node counts of `jkcov6` at `k = 2` (exact, single process):

| `z` | 13 | 17 | 19 | 23 | 29 |
|---|---|---|---|---|---|
| nodes | 150 | 2,577 | 53,560 | 1,491,366 | 55,917,112 |
| ratio | – | 17.2 | 20.8 | 27.8 | 37.5 |

The ratio itself grows about 1.30× per step. Extrapolating one step at a time at
a measured 2.0e5 nodes/s/core on 16 cores: `z = 31` ≈ 850 s (**done this round**),
`z = 37` ≈ 15 h (**the next purchasable rung for this vehicle**), `z = 41` ≈ 51 days,
and everything beyond is an extrapolation of an extrapolation and is printed only
to show the shape of the wall. **This vehicle is not the state of the art** —
Ziller–Morack reached `z = 73` in 2017 with a portioned ILP (Giovanni Resta's
binary-ILP formulation, recorded in A072753's own OEIS comments), which is a far
better machine. The measured fact about the target, as opposed to about the
vehicle, is that **the frontier has not moved in nine years** while both authors
remained active on the sequence.

`z = 151..251` is five to nine further primes past a frontier that has not moved
once. That is why this round substituted the `k`-axis.

---

## 6. Prior-art check (2026-08-29, first-hand)

* **OEIS A072753** (record #79, Aug 02 2017) read in full: 19 terms
  `2,4,10,24,31,42,60,74,94,117,148,173,213,236,275,316,364,409,436` at
  `n = 3..21`; formula `a(n) = (A288815(n) - 6)/6`; comments record Giovanni
  Resta's binary-ILP approach and John F. Morack's GLPK runs, and record that
  `a(19)` was first published as 355 and **corrected to 364** by Ziller.
  Keywords `hard,more`.
* **OEIS A288815** (record #19, Apr 12 2026) read in full: 21 terms, ending
  `2622` at `n = 21` (`p_21 = 73`). Keywords `hard,more`. **Still 21 terms.**
* **OEIS A048670** (record #164, Jul 11 2026) read in full: b-file to `n = 64`
  (Bozek, Google Cloud, 2021). The ordinary problem is at `n = 64`; the paired
  one is at `n = 21`.
* `j_k` for `k >= 3` appears nowhere: round 27's sweep (OEIS `seq:6,24,78`, 19
  hits, none number-theoretic in this sense; arXiv metadata over the complete
  math.NT Jacobsthal set) is re-affirmed, and the new values `180, 306, 612, 972`
  give a fresh search key.
* **VERDICT: NOVEL AS FAR AS SEARCHED** for the `k >= 3` values and for the
  k-axis method. **NOT NOVEL** — and stated as such — are: the covering
  restatement (`jk-family.md` §2, and it is ZM Prop. 1.5(2) at `k = 2`), the
  `D`-reduction (Hagedorn at `k = 1`, Ziller–Morack at `k = 2`), and the
  canonical-form symmetry break (Ziller–Morack RPA2).

---

## 7. Unsolved questions it touches

* **Ziller–Morack Conjecture 6** (`h_2(p_n#) < p_n^2 - p_n`): untouched, but the
  measurement says the truth is `z (log z)^{2.6}`-shaped on the computed range,
  i.e. the conjecture asks far less than the truth — the r24 reading (P4),
  re-confirmed from a second statistic.
* **(P3)**, "is the polylog exponent exactly 3?": the data says the exponent on
  the computed range is `2.6`, and that the `(k-1)` growth the exponent-`2k-1`
  reading needs is absent up to `k = 5`. That is *evidence about the range*, not
  about the asymptotic, and it is the first evidence of any kind.
* **The conjecture `j_k(P(x)) = x (log x)^{2k-1+o(1)}`** (`jk-family.md` §3d):
  this round supplies the first data against its *finite-range* shape and
  nothing against its asymptotic form. The conjecture should be restated with
  that distinction visible.

---

## 8. Reproduction

* `research/jk_cover.py` — reference engine (Python DFS + SAT) and the
  definition-vs-restatement brute force. Gate.
* `rust2/src/bin/jkcov6.rs` — the fast engine (reduced lattice, canonical form,
  `v3` bound). `cargo build --release --bin jkcov6`.
* `rust2/src/bin/jkcover.rs` — the unreduced engine, kept as an independent
  cross-check (no reduction, no canonical form).
* `research/jk_run.py` — two-phase parallel driver (witness, then seeded split
  infeasibility proof).
* `research/jk_growth.py` → `research/data/jk_growth.out` — all tables above.
  Gate.
* `research/data/r28_harvester_prereg.txt` — pre-registration, written before the
  runs it scores.

---

## 9. ROUND-29 ADDENDUM (2026-09-03): three new values, a protocol defect in
## our own round-28 run, and the model separation decided at two clean steps

Harvester lane, round 29. Gates: `research/jk_axis29.py` → **ALL ASSERTIONS
GREEN**; `research/jk_sat29.py check` (the SAT cross-engine); the round-28 gates
`jk_cover.py`, `jk_growth.py`, `j2_referee.py`, `j2_citesweep.py` re-run clean.
Pre-registration: `research/data/r29_harvester_prereg.txt`.

### 9a. A PROTOCOL DEFECT IN THE ROUND-28 `j_3(P(23))` RUN — found here, in our
### own work, before anyone quoted the number

`jk_run.py`'s phase-2 split is sound **only if no worker ever improves on the
shared incumbent**. Read out of `jkcov6.rs`: a node is pruned when
`feasible_to(cov, j, best + 1)` fails, so a worker whose `best` has risen prunes
*more* above the split depth, visits *fewer* split-depth nodes, and its global
`leafctr` counter diverges from the other workers'. The parts
`leafctr % nparts == part` then need not cover the tree. The driver's own
docstring says so and prints "rerun needed".

Round 28 launched the `j_3(P(23))` phase-2 run at seed 219 with 14 workers. All
fourteen finished EXACT with verified witnesses — **and two of them beat the
seed**, reaching `m = 227` and `m = 232`. The round ended before the parts were
harvested, so the violation was never acted on. Round 29 harvested them and
split the verdict:

* **VALID**: a machine-verified witness of length `m = 232`, hence
  `j_3(P(23)) >= 6 × 233 = 1398`.
* **INVALID as an upper bound**: rerun required at seed 232.

The rerun (`research/jk_run29.py`, explicit seed, fatal protocol assertion, five
workers) confirms it. **`j_3(P(23)) = 1398`, `m = 232`, EXACT.**

The lesson is general and belongs beside the engine: *a branch-and-bound split
is a proof only when the incumbent is a fixed point of the run.* If any worker
improves it, the parts stop partitioning and the answer is a lower bound.

### 9b. THREE NEW EXACT VALUES

| `z` | 3 | 5 | 7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|---|---|
| `j_3` | 6 | 24 | 78 | 180 | 306 | 612 | 972 | **1398** |
| `j_4` | – | 30 | 150 | 420 | 1230 | **2340** | **3810** | – |
| `j_5` | – | – | 180 | 930 | 2070 | 5490 | – | – |

* `j_3(P(23)) = 1398` — `m = 232`; 7.38e9 nodes at seed 219 (13.6 core-hours,
  round 28) plus the round-29 confirmation at seed 232.
* `j_4(P(17)) = 2340` — `m = 77`; 351,958 nodes, 0.345 s.
* `j_4(P(19)) = 3810` — `m = 126`; 99,408,318 nodes, 448.8 s.

Every value carries a witness re-verified by code that shares nothing with the
search, and an exhaustive infeasibility proof at `m + 1`.

**PROCESS MISS, recorded as one:** `j_4(P(17))` was computed as a cost probe
*before* the round's pre-registration was written. It is a new value and it
should have been pre-registered. It is used below as an input, never scored as
a hit.

### 9c. THE MODEL SEPARATION, DECIDED AT TWO POST-TRANSIENT STEPS

Round 28's weakness was named in the doc: the `k >= 3` data lay entirely inside
the small-`z` transient. Both new steps are outside it, and **both were
pre-registered with a numerical prediction from each model before the answer
existed** (`research/data/r28_harvester_prereg.txt` addendum, written
2026-08-30; `research/data/r29_harvester_prereg.txt` H3).

| step | `R_k` before | `R_k` after | measured move | (A) needs | (B) needs |
|---|---|---|---|---|---|
| `k=3`, 19 → 23 | 1.2100 | 1.2084 | **−0.13 %** | 0 % | +13.4 % |
| `k=4`, 17 → 19 | 1.4426 | 1.3768 | **−4.56 %** | 0 % | +12.2 % |

The `k = 3` step is the cleanest single measurement this lane has ever taken on
the question: the pre-registered model-(A) prediction was **1398** and the
answer is **1398, exactly, to the unit**, against model (B)'s 1590. The `k = 4`
step moves the *other way* from (B).

### 9d. AND A CORRECTION TO §4, AGAINST OURSELVES

Round 28 recorded that the measured excess `e_k = a_k − k` "does not grow with
`k`" (−0.08, 0.61, 0.56, 0.76, 1.72 at `k = 1..5`). With `j_4` now carrying five
points instead of three, the ladder is

    e_k = -0.08, 0.61, 0.73, 1.45, 1.72   at k = 1, 2, 3, 4, 5

and it **does** grow monotonically. The round-28 sentence is withdrawn. What
replaces it is sharper, not weaker: the excess is a **consistent fraction** of
what (B) demands —

    e_k / (k-1) = 0.61, 0.37, 0.48, 0.43   at k = 2, 3, 4, 5,

so on the computed range the truth looks like `z (log z)^{k + c(k-1)}` with
`c ≈ 0.45`: **strictly between (A) (`c = 0`) and (B) (`c = 1`), and at the same
place at every `k`**. That is a stronger statement than "the extra logs are
absent" and it is the shape a future model has to reproduce.

**THE STANDING CAVEAT IS UNCHANGED AND STILL LOAD-BEARING:** `(P2')` carries a
`C^k/B^{2k}` factor worth ≈0.03 at `z = 73, k = 2` and the construction does not
exist below `log x ≈ 300`. **None of this refutes the theorem.** It measures the
shape of the truth where exact values exist.

### 9e. THE PRICE, RE-MEASURED — AND §5's PRICES WERE WRONG

§5 priced the `k`-axis rungs by extrapolating the `k = 2` node curve. Measured
`k = 3` node counts (11,740 → 556,927 → 50,867,900 → 7.38e9 at `z = 13, 17, 19,
23`) give per-prime ratios 47.4×, 91.3×, 145.1×, themselves growing ≈1.75× per
step — steeper than `k = 2`'s, because each prime carries three classes and the
branching factor at every node is larger.

| rung | round-28 price | measured / projected | error |
|---|---|---|---|
| `j_3(23)` | ~1–2 core-hours | **13.6 core-hours** (actual) | 9× low |
| `j_3(29)` | ~10 core-hours | ~3,500 core-hours (1.9e12 nodes) | ~350× low |
| `j_3(31)` | ~100 core-hours | ~1.5e6 core-hours (8e14 nodes) | ~15,000× low |

`j_3(29)` and `j_3(31)` are **NOT ATTEMPTED** and are not buyable *on this
vehicle*. `j_4(P(23))` projects at ~5e11 nodes (~800 core-hours) and is also out
of reach here. **§9f re-prices `j_3(29)` at ~20 core-hours on a different
engine — read that section before quoting this table.**

### 9f. THE ILP HOLE OF §5, CLOSED — AND THE ANSWER IS NEGATIVE

§5 recorded an honest hole: "Ziller–Morack reached `z = 73` with a portioned ILP
… which is a far better machine", and round 28 added "I did not build an ILP,
and I do not know how much it would buy."

Two corrections and a measurement.

1. **The ILP is not an OEIS comment — it is printed mathematics we had never
   read.** It is **equation (2.2) of Ziller & Morack, arXiv:1611.03310, §2**:
   binary `x_{i,j}` for each prime `p_i` and class `j ∈ {1..p_i−1}`, with
   `Σ_j x_{i,j} = 1` per prime, a covering constraint per position, and an
   objective `Σ 2^{m_2−k} y_k` that finds the maximum `m` in one program. Seven
   rounds of this lane cited that paper without reading §2. **Standing lesson 10
   ("a hypothesis cited by number is an unread hypothesis") extends: a METHOD
   cited by reputation is an unread method.**
2. **Generalising it to `k` classes is one character**: `Σ_j x_{i,j} = 1` becomes
   `≤ k`. So the two-class ILP has existed, in print, since 2016.
3. **Measured, and it does not rescue the frontier.** `research/jk_sat29.py`
   encodes exactly that program as SAT (cardinality-constrained CNF, class 0
   excluded, reduced lattice) and decides both directions with CaDiCaL. It
   reproduces every known value. Its cost, in the solver's own operation counts
   (conflicts on the UNSAT side, which is the expensive direction):

   | `k=2`, `z` | 13 | 17 | 19 | 23 |
   |---|---|---|---|---|
   | SAT conflicts (UNSAT direction) | 131 | 1,570 | 14,503 | 178,618 |
   | ratio | – | 12.0× | 9.2× | 12.3× |
   | `jkcov6` DFS nodes | 150 | 2,577 | 53,560 | 1,491,366 |
   | ratio | – | 17.2× | 20.8× | 27.8× |

   (`z = 29`: SAT 2,952,407 conflicts, ratio 16.5×; DFS 55,917,112 nodes, ratio
   37.5×.) The solver's growth ratio is **flatter** than the DFS's at every
   step. **At `k = 2` that is not enough**: at `z = 31`, the DFS's 4.9
   core-hours, the solver did not decide even the *satisfiable* direction in
   570 s, and 12–16× per prime still needs `12^14` more work to reach
   `p_n = 73` from `p_n = 31`. **So the ILP route does not move `h_2`**, and
   the round-28 hole is now a measurement rather than an unknown.

4. **AT `k = 3` IT IS A DIFFERENT STORY, AND IT RE-PRICES §9e.**

   | `k=3`, `z` | 17 | 19 | 23 |
   |---|---|---|---|
   | SAT conflicts (UNSAT direction) | 8,889 | 201,771 | **8,710,802** |
   | ratio | – | 22.7× | 43.2× |
   | `jkcov6` DFS nodes | 556,927 | 50,867,900 | 7.38e9 |
   | ratio | – | 91.3× | 145.1× |

   **CaDiCaL proved `j_3(P(23)) = 1398` outright — both directions, one
   process, no split; it produced its own witness at m = 232 and proved
   m = 233 impossible — in 831 s on one core**, against the DFS's 13.6
   core-hours over fourteen workers. Two consequences:

   * **An independent two-sided proof of the value with no protocol risk of any
     kind.** §9a's defect is a property of *splitting* a branch-and-bound; a
     single-process UNSAT proof cannot have it. `j_3(P(23)) = 1398` no longer
     rests on the split.
   * **The next rung is purchasable again.** Carrying the measured SAT ratio
     forward at the same ≈1.9×-per-step growth: `j_3(P(29))` projects at ~7e8
     conflicts ≈ **20 core-hours**, against ~3,500 on the DFS. `j_3(P(31))`
     projects at ~8.7e10 conflicts ≈ 2,300 core-hours and stays out of reach.
     `j_3(29)` was **not launched** — a ~17-hour single-threaded job cannot
     finish inside a round.

   **THE GENERAL LESSON, and §9e is the second time this doc has had to learn
   it: a price is a property of a VEHICLE, not of a target.** §5 priced the
   `k`-axis off the `k = 2` node curve and was wrong by up to 15,000×; §9e then
   priced `j_3(29)` off the `k = 3` node curve and called it "not buyable",
   and a different engine does it in twenty core-hours.

   What none of this settles: a tuned portioned ILP with branch-and-cut,
   symmetry breaking and warm starts — ZM's actual vehicle — is a different
   program from this one and was not tested.

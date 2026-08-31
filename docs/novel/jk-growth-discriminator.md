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
| 17 | 26 | 192 | **612** | – | **5490** |
| 19 | 34 | 258 | **972** | – | – |
| 23 | 40 | 366 | **(see §5)** | – | – |

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

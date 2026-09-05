# The distortion method on the machine

LITERATURE-AND-CONSTRUCTION lane, round 51. Parent: `research/proof/the_wall.md` section 5e,
whose closing sentence is that every route from the machine's structure ends at a Jacobsthal-type
covering bound at exponent 2 "which no sieve reaches and no covering method has been tried on",
and that the one untried technique named in the record is the distortion method of covering
systems.

Brief: read the distortion-method literature (Balister-Bollobas-Morris-Sahasrabudhe-Tiba,
Hough, Filaseta-Ford-Konyagin-Pomerance-Yu), state the machine's problem in each paper's terms,
take the method's core inequality and try to run it on an interval, and give a verdict:
(a) a localised distortion inequality below `q^2/6`, (b) above `q^2/6` but polynomial, or
(c) density only, no localisation without a sieve-type input.

Scripts in `research/anchor235/r51/` (prefix `dm_`); result outputs, untracked, in
`research/anchor235/r51/results/`. Every number this document relies on is written here.

**Verdict in one paragraph.** Outcome **(c)**, and the deciding step is exact and elementary.
The distortion method's core inequality is `eta := sum_i E[alpha_i^2] < 1`, where `alpha_i(x)` is
the proportion of the fibre over `x` (the class of `x` mod the product of the gears already
processed) that gear `i` strikes. Over the machine's full period the CRT makes `alpha_i = 2/g_i`
on every fibre, so `eta = sum 4/g^2 < 0.36455` for EVERY set of primes above 3: the method applies
to the machine as it stands, its hypothesis holds with room, and its conclusion - the machine does
not cover `Z` - is one the CRT already gives with a better constant (9,367x to 786,128x better at
`q = 59..499`). Localised to an interval of length `L`, the fibres are the classes mod
`Q_{i-1} = g_1...g_{i-1}` intersected with the interval, and as soon as `Q_{i-1} > L` each fibre
holds ONE column, `alpha_i` is `0/1`-valued, `alpha^2 = alpha`, and **the second-moment term
equals the first-moment term identically** (the collapse lemma, section 4.1: one line, phase-free,
measure-free). From that gear on the method IS the union bound `sum 2/g`, which is face A's dead
capacity budget A2 and passes 1 at four gears. On the real machine's own window the collapse
begins at the fifth or sixth gear (17 or 19) and the exact budget is `eta_I = 1.074, 1.224, 1.469, 1.707`
at `q = 59, 97, 199, 499` - above 1 at every machine, and rising like `2 log log q`. The shortest
interval the localised budget can speak about at all is `L*(q) = 1.07e4, 4.08e6, 1.25e14, 1.93e30,
9.33e52` at `q = 59, 97, 199, 499, 997`, against windows `610, 1684, 7387, 42085, 169514`; `L*`
grows like `exp(theta(q^{0.71}))`, so the method's reach is super-polynomial in `q` where the
target is `q^2/6` - worse even than the sieve's `q^{4.27}`. What the collapsed gears would need is
a statement about how the surviving set distributes in the classes of the next gear inside a short
interval, which is a level-of-distribution input, i.e. a sieve remainder; at dimension 2 that is
face A and the parity barrier. One thing survives and is worth keeping: on the ADVERSARIAL ladder
the localised budget is not vacuous - it proves `A(K) < (p_{K+1}^2 - 1)/6`, the open lemma, for
every `K <= 10`, and fails from `K = 11` (threshold `632.9` against `W = 308`). The crossing at
`K = 11` is the first place any general method has been seen to run out on that lemma, and it is
the collapse that does it.

---

## 1. Pre-registered expectations

Written before any paper was opened this round, and before any computation.

**E1 (the hypotheses will not fit).** Every theorem of the distortion-method literature will be
stated for systems with DISTINCT MODULI whose MINIMUM MODULUS is LARGE (`min d >= N`, `N` large),
with ONE class per modulus, covering ALL of `Z`. The machine has fifteen to a hundred moduli, all
of them small (`5 <= g <= q`), two classes each, and asks about an interval. Predicted: no theorem
applies as stated, and the failing hypothesis is the minimum modulus, which for the machine is 5
and can never be raised. REFUTED if any theorem in the corpus is stated with no lower bound on the
moduli, or with `k` classes per modulus and small moduli allowed.

**E2 (density is free for us, so a density theorem is worth nothing).** The machine's moduli are
distinct primes, so over a full period the CRT makes the gears independent and the uncovered
density is EXACTLY `prod_{5<=g<=q} (1 - 2/g) ~ c/(log q)^2 > 0`. A theorem whose conclusion is
"the uncovered set has positive density" therefore has a conclusion we already have by one line of
CRT, with the exact constant. REFUTED if any paper's density conclusion is stronger than the CRT
product for distinct prime moduli (it cannot be; the product is exact).

**E3 (the difficulty the method solves is not our difficulty).** Predicted mechanism: the moduli of
a covering system are NOT coprime, the events are correlated, `prod (1 - 1/d)` is not a valid lower
bound, and the whole apparatus exists to control those correlations. The machine's moduli are
pairwise coprime, so its correlations are exactly zero. Predicted: the method addresses a
difficulty the machine does not have, and the machine has a difficulty (LOCALISATION) the method
never meets. REFUTED if the distortion apparatus is doing something other than decorrelation, or if
any of the papers proves an interval statement.

**E4 (localisation costs the sieve).** The uniform measure on `Z/PZ` is the only measure for which
every class of every gear has mass exactly `1/g`. Replacing it by the normalised indicator of an
interval of length `L` breaks that; the honest naive bound is `L prod(1 - 2/g) - 3^{pi(q)}`, vacuous
at every machine, and the only known repairs are Brun/Rosser truncation, i.e. a sieve, at dimension
2 with sifting limit `beta_2 = 4.2664`, giving `L >= q^{4.27}` and not `q^2/6`. Predicted outcome:
**(b) or (c)**, with (c) more likely, the lossy step being the passage from the uniform measure on
the period to the normalised indicator of the interval.

**E5 (the adversarial side is not in print).** No covering-systems paper bounds the longest interval
coverable by `K` arbitrary primes with 2 classes each. REFUTED by any printed bound on the length of
an interval coverable by a bounded number of prime moduli with bounded classes each.

**E6 (a number, so the lane can be scored).** If a localised distortion inequality exists, its
tolerated length will be `L ~ q^{2+delta}`, never below `q^2/6`; the pre-registered guess is that
the naive "+1 per gear" reading tolerates `L ~ q log q` (which would look like a proof of the
conjecture) and that this is an illusion. The lane must report BOTH numbers. A lane that reports
only the naive one has failed.

### Scorecard

| # | expectation | verdict | evidence |
|---|---|---|---|
| E1 | hypotheses do not fit; min modulus is the failing one | **HALF REFUTED** | true of the headline theorems (BBMST Thm 1.1 needs `d_i >= M`; FFKPY needs `n > N`), FALSE of the engine: BBMST **Theorem 3.1 has no lower bound on the moduli at all**, and its hypothesis `eta < 1` HOLDS for the machine at every `q` with `eta_Z < 0.36455` (R1). Multiplicity (k classes per modulus) is also already in the corpus: FFKPY Thm 2 (multiplicity `s`), and the distortion method has been run at multiplicity `s` in the function-field papers |
| E2 | density conclusion is free for the machine (CRT) | **CONFIRMED, and priced** | BBMST Thm 3.1's density bound is `1.52e-5, 3.69e-6, 5.42e-7, 8.09e-8` at `q = 59, 97, 199, 499` against the exact CRT densities `1.42e-1, 1.15e-1, 8.56e-2, 6.36e-2`: weaker by 9,367x to 786,128x (R1) |
| E3 | the method decorrelates; the machine is already decorrelated | **CONFIRMED** | the apparatus is `nu(d) = prod_{p|d} 1/(1-delta_p)` and `beta(C) = sum_{gcd(n_i,n_j)>1} 1/(n_i n_j)`; both vanish or are trivial for pairwise-coprime prime moduli. FFKPY Lemma 2.1 (`delta >= alpha - beta`) reduces on the machine to `delta >= prod(1-2/g)`, which is an EQUALITY, not a bound (R2) |
| E4 | localisation costs a sieve; naive error `3^{pi(q)}` | **CONFIRMED, and made exact** | the naive Legendre localisation needs `L >= 1.01e8, 8.19e11, 1.15e22, 3.70e45` at the four machines (R6); the method's own localisation needs `L*(q) = 1.07e4, 4.08e6, 1.25e14, 1.93e30` (R5). The step is named exactly: the collapse lemma (R4.1) |
| E5 | nothing in print on the adversarial interval question | **CONFIRMED** | the covering corpus's conclusions are densities over `Z`; the only printed interval bound of this shape is Stevens 1977 `H(r) <= 2 r^{2+2e log r}`, ONE class per prime, and it is `2.2e1` to `9.5e14` times the two-class truth `A(K)` (R7). No two-class analogue exists (confirmed against `literature_increment.md` 2a/3d) |
| E6 | naive `L ~ q log q`, honest `L` not below `q^2/6` | **CONFIRMED both halves** | naive "+1 per gear": `L = 211, 400, 1028, 2923`, i.e. BELOW the window at every machine - the illusion, exactly as pre-registered; honest: `L*(q)` above, `exp(theta(q^{0.71}))` (R5, R6) |

Two expectations were wrong in an informative direction. E1 was wrong about the ENGINE: the
technical theorem carries no minimum-modulus hypothesis and the machine satisfies it comfortably,
so the method is not blocked by hypotheses - it is blocked by having nothing to say. E6's
prediction that the localised bound would be polynomial (`q^{2+delta}`) was too generous: it is
super-polynomial.

---

## 2. The theorems, quoted

Verification key: `[READ]` = read the paper's own text (arXiv HTML / ar5iv / publisher page) this
round; `[SECONDARY]` = reported by a source I read; `[PROJECT]` = already on the project record,
cited not re-derived.

### 2.1 Balister, Bollobas, Morris, Sahasrabudhe, Tiba, *On the Erdos covering problem: the density of the uncovered set*, Inventiones math. 228 (2022) 377-414; arXiv:1811.03547

**Theorem 1.1.** *Let `eps > 0` and let `mu` be the multiplicative function defined by
`mu(p^i) = 1 + (log p)^{3+eps}/p` for all primes `p` and integers `i >= 1`. There exists `M > 0`
such that if `A_1, ..., A_k` are arithmetic progressions with distinct moduli
`d_1, ..., d_k >= M`, and `C = sum_i mu(d_i)/d_i`, then the density of the uncovered set
`R := Z \ union A_i` is at least `e^{-4C}/2`.* `[READ]`

**Theorem 1.2 (Schinzel's conjecture).** *If `A` is a finite collection of arithmetic progressions
that covers the integers, then at least one of the moduli divides another.* `[READ]`

**Theorem 1.4.** *Let `A = {A_d : d in D}` be a finite collection of arithmetic progressions with
distinct moduli that covers `Z`, and `Q = lcm(D)`. Then either `2 | Q`, or `9 | Q`, or `15 | Q`.*
`[READ]`

**Theorem 3.1 (the engine - the only statement of the paper with no lower bound on the moduli).**
*Let `A = {A_d : d in D}` be a finite collection of arithmetic progressions and let
`delta_1, ..., delta_n in [0, 1/2]`. If*

    eta := sum_{i=1}^n min{ M_i^(1),  M_i^(2) / (4 delta_i (1 - delta_i)) }  <  1,

*then `A` does not cover `Z`, and `P_0(R) >= (1 - eta) exp( -(2/(1-eta)) sum_{d in D} nu(d)/d )`.*
`[READ]`

with, in the paper's notation:

* the probability space is `Z_Q`, `Q = lcm(D)`, decomposed prime by prime: `Q_i = prod_{j<=i}
  p_j^{gamma_j}`, and `Z_{Q_i} = Z_{Q_{i-1}} x Z_{p_i^{gamma_i}}` by CRT;
* the **fibre** `F(x) = {(x, y) : y in Z_{p_i^{gamma_i}}}` over a point `x in Z_{Q_{i-1}}`;
* `B_i` = the points newly covered at stage `i` (by progressions whose modulus divides `Q_i` but
  not `Q_{i-1}`), and (eq. 4) `alpha_i(x) := |{y : (x,y) in B_i}| / p_i^{gamma_i}`, the
  **proportion of the fibre struck at stage `i`**;
* the **moments** `M_i^(1) := E_{i-1}[alpha_i(x)]`, `M_i^(2) := E_{i-1}[alpha_i(x)^2]`;
* the **reweighted measures** `P_i`, defined fibre by fibre: on `B_i`,
  `P_i = max{0, (alpha_i - delta_i)/(alpha_i (1 - delta_i))} P_{i-1}`, off `B_i`,
  `P_i = min{1/(1 - alpha_i), 1/(1 - delta_i)} P_{i-1}`;
* the **distortion** (eq. 12) `Delta_i(x) := max{0, log(P_i(x)/P_0(x))}`, with (Lemma 3.5)
  `E_i[Delta_i] <= 2 sum_{d in D_i} nu(d)/d`, `nu(d) = prod_{p_j | d} 1/(1 - delta_j)` (eq. 8);
* the loss at each stage (Lemma 3.3) `P_i(B_i) <= min{M_i^(1), M_i^(2)/(4 delta_i (1-delta_i))}`.

The minimum-modulus bound of the covering literature (`616,000`, improving Hough's `10^16`) is the
headline consequence of this machinery. `[SECONDARY]` (the survey arXiv:2211.01417 and the
publisher pages; I did not read the numerical section first-hand).

### 2.2 Hough, *Solution of the minimum modulus problem for covering systems*, Annals of Math. 181 (2015) 361-382; arXiv:1307.0874

**Theorem 1.** *The least modulus of a distinct covering system is at most `10^16`.* (The arXiv
abstract as fetched reads `10^18`; the published version and the BBMST survey both say `10^16`.)
`[READ abstract]` + `[SECONDARY]`

Method: reweighted measures `mu_i` (uniform on good fibres), "bias statistics"
`beta_k(i) = sum_{m | Q_i} l_k(m) max_b mu_i(... n (b mod m)) / mu_i(...)` tracking how irregularly
the surviving set distributes modulo divisors, a growth inequality for them (Proposition 3), a
second-moment bound `M_k(i,n) <= beta_k(i)` (Lemma 4), and Markov plus convexity to certify enough
good fibres (Lemma 5). The minimum-modulus hypothesis enters at the initial condition (C0),
`sum_{m > M, p | m => p <= P_0} 1/m < delta`, which needs `M` large. `[READ]`

### 2.3 Filaseta, Ford, Konyagin, Pomerance, Yu, *Sieving by large integers and covering systems of congruences*, J. Amer. Math. Soc. 20 (2007) 495-517; arXiv:math/0507374

Definitions `[READ]`: a **residue system** `C` is a finite MULTISET of pairs `(n, r)`, i.e.
residue classes `r mod n`; `S(C)` is the multiset of moduli; the **multiplicity** of `n` is how
often it occurs; `delta(C)` is the density of the uncovered set `R(C)`; `alpha(C) = prod_j
(1 - 1/n_j)`; `beta(C) = sum_{i<j, gcd(n_i,n_j) > 1} 1/(n_i n_j)`; `L(N,s) = exp(log N *
log log(s log N)/log(s log N))`.

**Lemma 2.1.** *For any residue system `C`, `delta(C) >= alpha(C) - beta(C)`.* `[READ]`

**Theorem 2.** *Let `0 < b < 1/2`, `0 < c < (1/3)(1 - 4b^2)`, and `N` sufficiently large. If `C`
is a residue system with `S(C)` consisting of integers `n > N` with multiplicity at most
`s <= exp(b sqrt(log N log log N))`, and `sum_{n in S(C)} 1/n <= c log L(N,s)`, then
`delta(C) > 0`.* `[READ]`

**Theorem 3.** *... if `C` has distinct moduli from `(N, KN]` with `K = L(N,s)^{(1-log 2)^{-1}
- eps}/s` ... then `delta(C) > 0`.* **Theorem 4** gives `delta(C) >= (1 + O((log N)^{-lambda}))
alpha(C)` in a narrower range. `[READ]`

**Theorem 7 (the variance).** *For a set `T` of distinct integers with minimum `N >= 3`,
`(1/W(T)) sum_{C in curly-C(T)} |delta(C) - alpha|^2 << alpha^2 (log N)/N^2`.* `[READ]`

Note for the record: FFKPY is the paper of this corpus that DOES allow several classes per
modulus (multiplicity `s`), and Theorem 2's hypothesis is a bound on `sum 1/n`, i.e. exactly the
machine's capacity sum - but with `n > N` and `N` sufficiently large.

### 2.4 What is NOT in any of them

* **No interval statement.** Every conclusion in BBMST, Hough and FFKPY is about the natural
  DENSITY of the uncovered subset of `Z`. The survey (arXiv:2211.01417, "Erdos covering systems",
  the authors' own exposition of the distortion method) states the method as a tool for covering
  `Z` and names its later uses as: reciprocal-sum problems, covering systems of multiplicity `s`
  over `F_q[x]`, and global function fields. No paper in the corpus bounds the length of an
  interval that a system can cover. `[READ]` (abstract and survey scope) + `[READ]` (the three
  papers' theorem lists above).
* **No fixed-separation structure.** Nowhere in the covering-systems corpus does a system in which
  each modulus contributes two classes at a FIXED offset appear. The one place in the wider
  literature where the machine's own system is named is Ford-Konyagin-Maynard-Pomerance-Tao,
  *Long gaps in sieved sets*, Remark 7 - "a two-dimensional system in which `I_p = {0, 2} (mod p)`"
  - and that remark is about LOWER bounds (`>> log X log log X`, "the trivial bound coming from
  these methods"), not upper. `[PROJECT]` (`docs/novel/README.md`, `layered-erdos-rankin`;
  `research/proof/iwaniec_two_class.md` section 1).

---

## 3. The translation: the machine in each paper's terms (R2)

The machine `M = {5..q}`: for each prime `5 <= g <= q`, the two classes `+-6^{-1} (mod g)` of the
column index. In covering-system language it is a residue system `C_M` with `pi(q) - 2` distinct
PRIME moduli, each of multiplicity **2**, the two classes at the fixed separation
`d_g = 2 * 6^{-1} = 3^{-1} (mod g)`. The target is an INTERVAL of `W(q) = (q'^2 - 1)/6` columns,
not `Z`.

| paper | does it apply? | why, and what it says |
|---|---|---|
| BBMST **Thm 1.1** | **no** | needs `d_i >= M` with `M` absolute and large; the machine's smallest modulus is 5, permanently. The failing hypothesis is exactly the one E1 named |
| BBMST **Thm 1.2, 1.4** | vacuously | they are structure theorems about systems that DO cover `Z`; the machine does not, so they say nothing |
| BBMST **Thm 3.1** | **YES** | no lower bound on the moduli; the moduli need not be distinct in any essential way (the theorem is written in terms of the fibre proportions `alpha_i`, so two classes per prime simply makes `alpha_i = 2/g_i` instead of `1/g_i`). Its hypothesis `eta < 1` HOLDS: `eta_Z = sum 4/g^2 < 0.36455`. Conclusion: the machine does not cover `Z`, with a density bound |
| Hough | **no** | condition (C0) needs the minimum modulus above `10^16` |
| FFKPY **Lemma 2.1** | **YES, and it is an identity** | `beta(C_M) = 0` (the moduli are pairwise coprime), so the lemma reduces to `delta >= alpha`, and for prime moduli with two classes each the CRT makes `delta = prod(1 - 2/g)` exactly. The lemma is the CRT |
| FFKPY **Thms 2, 3, 4** | **no** | all need `n > N` with `N` sufficiently large; Theorem 2 is the one that allows multiplicity, and the machine's multiplicity 2 is harmless there - it is the small moduli that are fatal |
| FFKPY **Thm 7** | **no** | a variance over random class choices with `min N >= 3` and an error `<< alpha^2 log N / N^2`; at `N = 5` the error term is larger than the main term |
| FKMPT Remark 7 | names the system | the only appearance of `I_p = {0,2} (mod p)` in print; a LOWER-bound remark |

**Does any of it bound the length of an interval two classes per prime up to `q` can cover?**
**No.** Not one theorem in the corpus has an interval in its conclusion.

**Why density does not give an interval bound, stated exactly.** For the machine, density is not
merely insufficient - it is FREE and it is EXACT. By the CRT the openings of `{5..q}` are a
periodic set of period `P = prod_{5<=g<=q} g` and density `prod (1 - 2/g)`, which at `q = 499` is
`6.36e-2`: about one column in sixteen is open, and the expected number of openings in the window
is `W * prod(1-2/g) = 42085 * 0.0636 = 2678`. The root asks whether ONE of them is there. Density
says nothing about that, because the window is shorter than the period by a factor of order
`e^q / q^2`, and a periodic set of positive density can have gaps of any size up to its period.
The whole content of the root is a LARGE-DEVIATION statement about a periodic set inside a window
`10^{-10^{200}}` of the period long; density is the mean, and the mean is not in dispute.

**Does the method's engine give local information with a modification?** That is section 4. The
answer is that it does - the budget localises formally - but that the budget then collapses to the
capacity sum before it reaches the window.

---

## 4. The construction attempt: the core inequality on an interval

### 4.0 The core inequality, reduced

In Theorem 3.1 the free parameters are `delta_i in [0, 1/2]`. Since `4 delta(1-delta) <= 1` with
equality at `delta = 1/2`, and since `alpha in [0,1]` gives `alpha^2 <= alpha` and hence
`M^(2) <= M^(1)`, each term is minimised at `delta_i = 1/2`, where it equals `M_i^(2)`. So for the
purpose of the hypothesis `eta < 1` the core inequality is

> **(CI)**   `eta = sum_i E_{i-1}[ alpha_i(x)^2 ]  <  1`,
> `alpha_i(x)` = the proportion of the fibre over `x` (its class mod `Q_{i-1}`) struck by gear `i`.

(The `delta_i` only trade against the density conclusion, through `nu(d) = prod 1/(1-delta_j)`;
they do not help the hypothesis.) **This is the whole method in one line: it replaces the union
bound `sum M^(1)` by the second moment `sum M^(2)`, and pays for the replacement in the density
constant.** For the machine over its period the CRT gives `alpha_i = 2/g_i` on every fibre and

    eta_Z(q) = sum_{5 <= g <= q} 4/g^2   ->   4 * sum_{p >= 5} 1/p^2 = 0.364545,

so **the machine's capacity budget `sum 2/g` (which passes 1 at four gears and is face A's dead
A2) is replaced by a budget that converges and never reaches 0.37.** That is exactly the kind of
non-saturating budget the root needs. Everything now depends on whether it survives localisation.

### R4.1 / section 4.1 - The collapse lemma (the answer, in one line)

> **Collapse lemma.** Let `I` be an interval of `L` columns and let the gears be processed in
> increasing order, `Q_{i-1} = g_1 ... g_{i-1}`. If `Q_{i-1} >= L` then every fibre of `I` (a class
> mod `Q_{i-1}`) contains at most one column, so `alpha_i` is `{0,1}`-valued, so
> `alpha_i^2 = alpha_i`, so `M_i^(2) = M_i^(1)` and
> `min{M_i^(1), M_i^(2)/(4 delta(1-delta))} = M_i^(1)` for EVERY `delta in [0,1/2]`.
>
> *Proof.* Two columns of `I` congruent mod `Q_{i-1}` differ by at least `Q_{i-1} >= L > `
> the diameter of `I`. And `4 delta(1-delta) <= 1`. QED.

The lemma is phase-free, measure-free and machine-free: it holds for the real teeth, for any
adversarial phases, and under any measure the method chooses. **From the first gear whose
predecessors' product exceeds the interval, the distortion method is exactly the union bound.**
And the union bound on the machine is `sum 2/g`, the capacity budget, face A's A2, which passes 1
at `{5,7,11,13}`.

The intermediate regime is the same statement quantitatively. With `m_i = L/Q_{i-1}` the fibre
size, `alpha_i <= min(1, 2/m_i)` pointwise (a gear puts at most two teeth in a fibre) and
`E[alpha_i] <= 2 ceil(L/g_i)/L <= min(1, 2/g_i + 2/L)`, while `E[alpha_i^2] >= (E alpha_i)^2 =
4/g_i^2` by Cauchy-Schwarz. So each term of (CI) lies in

    [ 4/g^2 ,  min(1, 2/m) * min(1, 2/g + 2/L) ] ,      m = max(1, L/Q_{<g}) ,

which is `4/g^2` when `m >= g` (the fibre carries whole classes: no localisation loss) and
`2/g + 2/L` when `m = 1` (full collapse). The gain of the second moment over the first is the
factor `min(1, 2/m)`: **the method's entire advantage is proportional to the number of columns of
the interval that lie in one class of everything below the current gear.**

### R1 / section 4.2 - The budget over the period

`research/anchor235/r51/dm_budget.py`, part A. Exact.

    q   gears   sum 2/g   eta_Z    exact density   BBMST Thm 3.1 bound   truth/bound
    59     15    1.7283  0.3516     1.422485e-01          1.518654e-05       9,367x
    97     23    1.9390  0.3573     1.148911e-01          3.692822e-06      31,112x
   199     44    2.2314  0.3616     8.557450e-02          5.415003e-07     158,032x
   499     93    2.5268  0.3635     6.362986e-02          8.094080e-08     786,128x

`eta_Z < 1` at every `q`, with limit `0.364545`, so **Theorem 3.1 applies to the machine as it
stands and proves that the machine does not cover `Z`** - by an argument that is not the CRT. Its
density conclusion is four to six orders of magnitude weaker than the CRT product. That is E2,
priced.

### R3 / section 4.3 - The budget on the real window, exact, real teeth

`dm_budget.py`, part B. For each gear, `M^(1)` and `M^(2)` are computed exactly by partitioning the
window's columns into their classes mod `Q_{<g}` and averaging the fibre proportions; `M2_surv` is
the same under the uniform measure on the columns still open after the smaller gears (the method's
`P_{i-1}` lies between the two). Real teeth, `u_g = 6^{-1} mod g`, real window
`(q, q'^2]` in columns.

    q = 59, window columns 11..620, L = 610, 15 gears

        g     Q_<g   fibre m   M1        M2        M2/M1
        5        1    610.00   0.40000   0.16000   0.400
        7        5    122.00   0.28525   0.08140   0.285
       11       35     17.43   0.18197   0.03504   0.193
       13      385      1.58   0.15410   0.09754   0.633   <- fibre shorter than the gear
       17     5005      1.00   0.11803   0.11803   1.000   <- COLLAPSE
       19    85085      1.00   0.10492   0.10492   1.000
       23  1616615      1.00   0.08689   0.08689   1.000
       ... every gear from 17 up: M2 = M1 exactly

    eta_I(uniform)   = 1.0740      eta_I(survivors) = 1.0147      [need < 1]
    sum 2/g          = 1.7283      eta_Z            = 0.3516

The four machines:

    q    L = W(q)   eta_I uniform   eta_I survivors   sum 2/g   eta_Z   first collapsed gear
    59        610          1.0740            1.0147    1.7283  0.3516        17 (5th gear)
    97       1684          1.2241            1.1839    1.9390  0.3573        17 (5th gear)
   199       7387          1.4693            1.4773    2.2314  0.3616        19 (6th gear)
   499      42085          1.7068            1.7414    2.5268  0.3635        19 (6th gear)

**The budget is above 1 at every machine, at the window's own length, with the real teeth.** The
head (four or five gears) contributes about `0.30`; everything else contributes its full capacity
`2/g`, and `sum_{g >= 17} 2/g` alone is `0.707, 0.918, 1.210, 1.505`. The excess over 1 grows like
`2 log log q`: `1.07, 1.22, 1.47, 1.71`. There is no version of this that improves with `q` - the
head is bounded by `0.36455` forever and the tail diverges.

### R5 / section 4.4 - The length the localised budget tolerates

`dm_budget.py`, part C, and `dm_adversary.py`. `L*(q)` is the least `L` with the rigorous form of
(CI) below 1 (section 4.1's upper envelope, which grants the method every benefit short of
falsehood). `eta` is decreasing in `L`, so `L*` is the SHORTEST interval about which the localised
method can say anything at all.

    q     W(q)      F        eta@W    L*(q)         L*/W        cut gear   ln(cut)/ln(q)
    59     610    161       1.1748    1.0671e+04    1.75e+01          13          0.629
    97    1684      -       1.2857    4.0753e+06    2.42e+03          19          0.644
   199    7387      -       1.5192    1.2542e+14    1.70e+10          41          0.702
   499   42085      -       1.7183    1.9338e+30    4.59e+25          79          0.703
   997  169514      -            -    9.3348e+52    5.51e+47         137          0.713

(`F({5..59}) = 161` is the certified record ladder; the higher `F` are past the scan wall.)

`L*(q)` is the primorial of the cut gear: the method needs the interval to be as long as the
product of all the gears BELOW the point where the tail capacity `sum_{g > cut} 2/g` drops under
`1 - head`. Since `sum_{cut < g <= q} 2/g ~ 2 log(log q / log cut)` and the head is capped at
`0.36455`, the cut sits at `cut ~ q^{0.728}` and

    L*(q)  =  exp( (1 + o(1)) * theta(q^{0.728}) )  =  exp( q^{0.71 ... 0.73} ),

measured `ln(cut)/ln(q) = 0.629, 0.644, 0.702, 0.703, 0.713` at `q = 59..997`, rising towards the
predicted `0.728`. The target is `W ~ q^2/6`. **The localised distortion method's reach is
super-polynomial in `q` where the target is quadratic** - and worse than the sieve, whose
dimension-2 sifting limit gives `q^{4.2664}` (`iwaniec_two_class.md`).

### R6 / section 4.5 - The naive localisation, both readings (the pre-registered E6 test)

`dm_budget.py`, part D.

    q    n=gears   W(q)    prod(1-2/g)    L for naive-1    L for naive-2
    59        15    610       0.142249            210.9       1.0087e+08
    97        23   1684       0.114891            400.4       8.1941e+11
   199        44   7387       0.085574           1028.3       1.1508e+22
   499        93  42085       0.063630           2923.2       3.7035e+45

* **naive-1** is the reading the brief asked to be tested: "an interval of length `L` meets each
  class `L/g` times up to 1, so the error is the `+1`s". Sum them: `openings >= L prod(1-2/g)
  - 2 * (#gears)`, positive as soon as `L > 2 n / prod(1-2/g)`, i.e. `L = 211, 400, 1028, 2923`.
  These are BELOW the window at every machine (`610, 1684, 7387, 42085`) and grow like
  `q log q / (log q)^{-2} ~ q (log q)^3`, so naive-1 "proves" the twin prime conjecture. It is of
  course false: `prod(1-2/g)` is the density over the PERIOD, and the step that puts it in front of
  `L` is the multiplicative independence of the gears, which on an interval is not free. The `+1`s
  are the error in ONE gear's count; the product of `n` such counts has `2^n` correction terms, not
  `n`.
* **naive-2** is the same with the correction terms paid for honestly: Legendre's inclusion-exclusion
  over the `3^n` pairs `(d, class-choice)` with `|rho_d| < 1`. It needs `L >= (3^n - 1)/prod(1-2/g)`:
  `1.0e8, 8.2e11, 1.2e22, 3.7e45`. Above the window by 5, 8, 18 and 41 orders of magnitude.
* The distortion method sits strictly between the two (`1.07e4, 4.08e6, 1.25e14, 1.93e30`): it is a
  genuine improvement on Legendre by 4 to 15 orders of magnitude, and still above the window by 1
  to 25.

### 4.6 What exactly would have to be supplied

At a collapsed gear the method needs to know that the surviving columns of the interval do not sit
disproportionately in the two classes of the next gear - i.e. that `P_{i-1}(B_i)` is close to
`2/g_i`, with an error small enough that the `n - t` collapsed terms sum to less than `1 - head`.
That is a statement about the distribution of a sifted set in the progressions of a new modulus
inside a short interval: **a level-of-distribution statement, which is precisely the remainder term
of a sieve.** The method's own device for it - the second moment over fibres - is unavailable by
the collapse lemma. So the localised distortion method needs a sieve input, the sieve is at
dimension 2 because there are two classes per prime, and dimension 2 is face A: the lower-bound
function `f_2(s)` vanishes for `s <= beta_2 = 4.2664` while the window sits at `s = 2`. That is the
parity barrier again, reached from the covering side rather than the sieve side.

---

## 5. The adversarial side (R7)

The project's adversarial object is `A(K)` = the longest run of consecutive columns that `K`
primes `>= 5`, each with two classes at its own fixed separation and any phase, can block; exact to
`K = 12` (`arc_multiset.md` R1) and the open lemma is `A(K) < (p_{K+1}^2 - 1)/6`.

**Is there a printed bound?** No. The covering corpus has no interval conclusion at all (section
2.4). The nearest printed object of this shape is the ONE-class function `H(r) = max_{omega(n)=r}
j(n)` and Stevens' 1977 bound `H(r) <= 2 r^{2 + 2e log r}` (`dm_adversary.py`):

    r      2 r^(2+2e ln r)      A(r), the 2-class truth      ratio
    2           1.0901e+02                            5    2.18e+01
    4           1.1033e+06                           16    6.90e+04
    8           2.0733e+12                           45    4.61e+10
   12           1.0925e+17                          115    9.50e+14

so even in one class, and even for the same `r`, the only printed bound of this shape is fifteen
orders of magnitude above the truth at `K = 12`. There is no two-class analogue in print
(`literature_increment.md` 2a/3d: `h_2` has two papers, no upper bound of any kind). E5 confirmed.

**What the localised distortion budget gives on `A(K)`** (`dm_adversary.py`). Take the `K` smallest
gears (the worst `K`-set for the budget, since both `4/g^2` and `2/g` fall with `g`), and compute
`L*max(K)`, the shortest interval the localised (CI) can rule out. If a localised Theorem 3.1 is
valid, then `A(K) <= L*max(K)`.

    K   gears                 A(K)     W    L*max       L*/W    L*avg   avg <= A?  collapsed@W
    1   5                        2     8    2.000e0    0.250    1.00e0    REFUTED       0
    2   5,7                      5    20    5.033e0    0.252    1.00e0    REFUTED       0
    3   5,7,11                   7    28    1.032e1    0.369    1.00e0    REFUTED       1
    4   5,7,11,13               16    48    1.606e1    0.335    1.34e0    REFUTED       1
    5   ..17                    22    60    2.498e1    0.416    2.21e0    REFUTED       2
    6   ..19                    28    88    4.034e1    0.458    5.94e0    REFUTED       3
    7   ..23                    37   140    7.655e1    0.547    9.30e0    REFUTED       4
    8   ..29                    45   160    1.065e2    0.666    1.69e1    REFUTED       5
    9   ..31                    68   228    1.589e2    0.697    4.40e1    REFUTED       6
   10   ..37                    88   280    2.546e2    0.909    7.02e1    REFUTED       7
   11   ..41                   101   308    6.329e2    2.05     1.52e2         ok       8
   12   ..43                   115   368    1.025e3    2.79     4.90e2         ok       9
   14   ..53                     -   580    2.657e3    4.58
   16   ..59                     -   748    1.508e4   20.2
   18   ..67                     -   888    5.574e4   62.8
   21   ..79                     -  1320    4.955e5  375
   24   ..97                     -  1768    5.716e6 3.23e3
   28   ..109                    -  2688    1.407e8 5.24e4
   32   ..137                    -  3700    3.843e9 1.04e6

Three readings, and the middle one is the finding.

* **`L*max >= A(K)` at every `K <= 12`** (`2 >= 2`, `5.03 >= 5`, `10.3 >= 7`, `16.06 >= 16`,
  `25 >= 22`, `40 >= 28`, `76.6 >= 37`, `106 >= 45`, `159 >= 68`, `255 >= 88`, `633 >= 101`,
  `1025 >= 115`), so the rigorous envelope of section 4.1 is consistent with the exact ladder at
  every point where the ladder is known. That is the validity gate.
* **The localised budget proves the open lemma `A(K) < (p_{K+1}^2 - 1)/6` for every `K <= 10`, and
  fails from `K = 11`.** `L*/W` runs `0.25, 0.25, 0.37, 0.34, 0.42, 0.46, 0.55, 0.67, 0.70, 0.91`,
  then `2.05, 2.79, 4.58, 20.2, 62.8, 375, 3.2e3, 5.2e4, 1.0e6`. The crossing is at `K = 11`, and
  after it the failure is total: `L*` grows like a primorial and `W` like `(K log K)^2`. This is
  the first time on the tree that a general method has been seen to REACH the open lemma at all,
  and the exact place where it stops. What stops it is the collapse count: at `K = 10` seven of the
  ten gears are collapsed at `L = W` and their capacity `sum 2/g = 0.71` still leaves room; at
  `K = 11` it is eight gears and `0.83`, and the head `0.30` puts the total over 1.
* **The average reading is not merely weak, it is FALSE.** If one localises with the AVERAGE first
  moments (every gear strikes its average share `2/g`, teeth independent within a fibre) the
  threshold `L*avg` drops to `1.34, 2.21, 5.94, 9.30, 16.9, 44.0, 70.2, 152` at `K = 4..11` - at or
  below the exact `A(K)` at `K = 1..11`, so it would assert that `K` gears cannot block runs they
  demonstrably do block (`A(7) = 37` against a claimed threshold `9.3`). **The exact `A(K)` ladder
  refutes the average localisation outright.** The reason is the one the wall keeps naming: on a
  short interval the phases are CHOSEN, and a chosen phase strikes more than its average share.
  Any localisation of the distortion method must therefore carry the phase-adversarial first
  moment, which is what section 4.1's envelope does, and that envelope is what crosses at `K = 11`.

---

## 6. Verdict

**Outcome (c): the method gives density only, and localisation is impossible without a sieve-type
input.** With the qualification that the localisation is not vacuous - it exists, it is
computable, and it reaches to `K = 10` on the adversarial ladder before it stops.

The deciding step, stated once:

> **The collapse.** The distortion method's whole gain over the union bound is the replacement of
> `M^(1) = E[alpha]` by `M^(2) = E[alpha^2]`, valid because `sum 1/g^2` converges where `sum 1/g`
> does not. On an interval of `L` columns, the fibre over a point at stage `i` is its class mod
> `Q_{i-1} = g_1...g_{i-1}` intersected with the interval; once `Q_{i-1} >= L` that fibre is a
> single column, `alpha_i` is `{0,1}`-valued, `alpha^2 = alpha`, and `M^(2) = M^(1)` for every
> choice of the method's parameters. From that gear on, the distortion method IS the capacity
> bound `sum 2/g` - face A's A2, dead since round 1, and over budget at four gears.

On the real machine's own window the collapse begins at the fifth or sixth gear, and the exact budget at
`L = W(q)` is `1.074, 1.224, 1.469, 1.707` at `q = 59, 97, 199, 499`: over 1 at every machine and
diverging like `2 log log q`. The shortest interval the method can address is
`L*(q) = exp((1+o(1)) theta(q^{0.728}))`, measured `1.07e4, 4.08e6, 1.25e14, 1.93e30, 9.33e52` at
`q = 59, 97, 199, 499, 997` against windows `610, 1684, 7387, 42085, 169514`. So the answer is not
(a) and not even (b): the tolerated length is not polynomial in `q` at all.

The step that needs the sieve is named: to keep a collapsed gear's term below its capacity `2/g`
one needs the surviving set of the interval to be equidistributed in the classes of the next gear,
which is a level of distribution, which is a sieve remainder; and the sieve here is two classes per
prime, dimension 2, sifting limit `beta_2 = 4.2664` against the window's `s = 2`. **So yes, the
step is the parity barrier again, reached from the covering side.** That is not a new wall; it is
the same wall, and this lane's contribution is to show that the one named untried technique arrives
at it too, and where.

What is worth keeping from the round, in order of value:

1. **The reduced core inequality `eta = sum_i E[alpha_i^2] < 1`** with `delta_i = 1/2`, and the
   fact that over the period the machine's `eta_Z = sum 4/g^2 < 0.36455` for EVERY set of primes
   above 3. This is a non-CRT, non-counting, never-saturating budget for the covering problem. It
   is the first budget on the tree that does not cross 1. Everything about the root is now the
   question of how much of it survives on a short interval, and that question has a name: the
   collapse.
2. **The collapse lemma**, one line, phase-free and measure-free, which says exactly how much
   survives: the gain factor is `min(1, 2 Q_{<g}/L)` per gear.
3. **The adversarial crossing at `K = 11`.** The localised budget PROVES `A(K) < (p_{K+1}^2-1)/6`
   for `K <= 10` and fails at 11. That is a genuine, if small, positive result about the open
   lemma, and it locates the failure in one number (the tail capacity of the collapsed gears).
4. **The refutation of the average localisation by the exact `A(K)` ladder.** Any argument that
   localises with average first moments is false, and `A(7) = 37 > 9.3` is the refuting instance.
   This is a reusable filter on future covering-side attempts.

What is dead: the distortion method as a route to `F(y) < y^2/6`. Not for want of hypotheses (the
engine applies to the machine unchanged) but for want of content on an interval.

### Children this branch opens

* **The half-collapsed regime.** The gain factor `min(1, 2Q_{<g}/L)` is between 0 and 1 for exactly
  one or two gears at each `(q, L)`. Is there an ordering of the gears other than increasing that
  keeps more of them uncollapsed? The fibres are classes mod the product of the gears already
  processed, so the collapse count at a given `L` is `#{i : prod of the first i-1 chosen gears
  >= L}` and is minimised by processing the LARGEST gears first (their product reaches `L` in fewer
  steps, but those steps are the cheap ones). Cheap to compute; it changes which gears pay `2/g`
  and which pay `4/g^2`, and the head is where all the mass is.
* **A second moment over a coarser partition.** The collapse is forced only because the partition
  is by classes mod `Q_{i-1}`. Any partition of the interval into blocks of `>= g_i` columns on
  which the surviving set is near-uniform would restore the second moment. The natural candidate is
  a partition into blocks of length `g_i` (arithmetic blocks, not congruence fibres), for which the
  proportion struck is exactly `2/g_i` for the gear itself; what has to be shown is that the
  SURVIVING measure is not concentrated in the blocks. That is a much weaker equidistribution
  statement than a level of distribution, and it is not obviously a sieve. It is the one crack in
  the collapse and it should be the next branch.
* **`eta_Z < 0.36455` for every prime set as a machine-free fact.** It says: two classes per prime,
  any primes above 3, any phases, never cover more than a bounded fraction of the period in the
  second-moment sense. The project has no other budget with this property. Where else does
  `sum 4/g^2` appear on the tree, and does the same square-summability survive any of the tree's
  other reductions (islands, arcs, the tiler function `h_S(L)`)?

---

## 7. Prior art recorded

Every reference, its exact statement, and whether the machine's fixed separation (one third) or its
two-classes-per-prime structure appears in it.

| source | exact statement used | fixed separation? | two classes per modulus? |
|---|---|---|---|
| Balister, Bollobas, Morris, Sahasrabudhe, Tiba, *On the Erdos covering problem: the density of the uncovered set*, Invent. math. 228 (2022) 377-414, arXiv:1811.03547 | Thm 1.1 (density `>= e^{-4C}/2` for distinct moduli `>= M`); Thm 1.2 (Schinzel); Thm 1.4 (`2|Q` or `9|Q` or `15|Q`); **Thm 3.1** (`eta < 1` implies not covering, with the density bound), Lemmas 3.3, 3.5, eq. 4, 8, 12 | **no** | **not named, but the engine is indifferent**: Thm 3.1 is written in fibre proportions `alpha_i`, so `k` classes per modulus costs only `alpha_i = k/g` |
| Balister, Bollobas, Morris, Sahasrabudhe, Tiba, *Erdos covering systems* (survey/exposition of the distortion method), arXiv:2211.01417 | "a simpler and stronger variant of Hough's method ... which we call the distortion method"; minimum modulus reduced from `10^16` to `616,000`; later uses listed as reciprocal-sum problems, `F_q[x]`, global function fields | no | multiplicity `s` systems appear in the function-field applications |
| Hough, *Solution of the minimum modulus problem for covering systems*, Ann. of Math. 181 (2015) 361-382, arXiv:1307.0874 | Thm 1 (least modulus `<= 10^16`); the bias statistics `beta_k(i)`, Prop. 3, Lemmas 4, 5; condition (C0) `sum_{m>M, P_0-smooth} 1/m < delta` | no | no (one class per modulus, distinct moduli) |
| Filaseta, Ford, Konyagin, Pomerance, Yu, *Sieving by large integers and covering systems of congruences*, J. Amer. Math. Soc. 20 (2007) 495-517, arXiv:math/0507374 | Lemma 2.1 (`delta >= alpha - beta`); Thms 1-7 as quoted in section 2.3; definitions of `alpha, beta, delta, L(N,s)`, multiplicity | no | **YES**, as multiplicity `s` in Thm 2 - the only place in this corpus where several classes share a modulus by hypothesis; but the moduli must exceed a large `N` |
| Ford, Konyagin, Maynard, Pomerance, Tao, *Long gaps in sieved sets*, Remark 7 | "our methods only seem to give good results in the one-dimensional case ... a two-dimensional system in which `I_p = {0,2} (mod p)`"; `>> log X log log X` as "the trivial bound coming from these methods" | **the machine's own system, named** (`{0,2}` is the fixed separation in integer coordinates) | **yes, exactly ours** - and the remark is a LOWER bound, no upper bound is offered |
| Stevens, Math. Ann. 226 (1977) 95-97, via Hajdu-Saradha (1.1) | `H(r) <= 2 r^{2 + 2e log r}`, `H(r) = max_{omega(n)=r} j(n)`: the longest interval `r` arbitrary primes can block with ONE class each | no | no |
| Ziller & Morack, arXiv:1706.00317, Conjecture 6 and Thm 4.1 | `h_2(n) < p_n^2 - p_n` implies Goldbach and the infinitude of prime pairs | no (max over class assignments) | **yes** (`omega(2)=1, omega(p)=2`) | 
| Iwaniec 1978 / DHR dimension 2 | `f_2(s) = 0` for `s <= beta_2 = 4.2664`; `F <= C y^{4.27+eps}` | - | the two-class transfer, `research/proof/iwaniec_two_class.md` |

**One-line prior-art conclusion.** The fixed separation appears in the whole literature exactly
once, in FKMPT's Remark 7, and there only as an example of a system their (lower-bound) methods
handle badly. Two classes per modulus appears as "multiplicity" in FFKPY and in the function-field
distortion papers, always with the moduli large. Nowhere does any paper bound the length of an
interval such a system can cover. The machine's problem, stated in the covering literature's own
language, is a question that literature has not asked.

---

## 8. Dead ends, with the refuting instance

* **"Apply BBMST Theorem 1.1 (or Hough, or FFKPY Thms 2-4) to the machine."** DEAD by hypothesis:
  all of them require the moduli to exceed a large absolute `M` or `N`; the machine's smallest
  modulus is 5, permanently.
* **"The distortion method gives a positive density, which is what we need."** DEAD: for
  pairwise-coprime prime moduli the CRT gives the density exactly, `prod(1-2/g)`, and Theorem 3.1's
  bound is 9,367x to 786,128x weaker (R1). FFKPY's Lemma 2.1 reduces to the same identity since
  `beta(C_M) = 0`.
* **"Localise the product formula by paying one unit of error per gear."** DEAD, refuted by its own
  conclusion: it gives `L = 211, 400, 1028, 2923` at `q = 59..499`, below the window, i.e. it
  "proves" the twin prime conjecture. The `n` errors are the errors of `n` separate counts, not of
  their product; the product has `3^n` correction terms and needs `L >= 1.0e8 ... 3.7e45` (R6).
* **"Localise the distortion inequality with average first moments."** DEAD, refuted by the exact
  adversarial ladder: it would give `A(7) <= 9.3` against the certified `A(7) = 37`, and fails the
  same way at `K = 1..11` (R7). On a short interval the phases are chosen and beat their average.
* **"Choose the `delta_i` better."** DEAD: `4 delta(1-delta) <= 1` on `[0, 1/2]`, so
  `min{M^(1), M^(2)/(4 delta(1-delta))}` is minimised at `delta = 1/2` where it equals `M^(2)`,
  and at a collapsed gear `M^(2) = M^(1)` for every `delta`. The parameters trade only against the
  density constant, never against the hypothesis.

## 9. Exceptionless, with the count

| statement | range | count | status |
|---|---|---|---|
| `Q_{i-1} >= L` implies `M_i^(2) = M_i^(1)` (the collapse lemma) | all | proof, not a check | proved, one line |
| `eta_Z = sum_{5<=g<=q} 4/g^2 < 0.36455` | every set of primes `>= 5` | proof (`sum_{p>=5} p^{-2} = 0.091136`) | exact |
| `eta_I(W(q)) > 1` with the real teeth, uniform and survivor measures | `q = 59, 97, 199, 499` | 8 evaluations | exact |
| `L*max(K) >= A(K)` (the localised envelope is consistent with the exact ladder) | `K = 1..12` | 12 | exact |
| `L*avg(K) <= A(K)` (the average localisation is false) | `K = 1..11` | 11 | exact, refuting instance `A(7) = 37 > 9.30` |
| `L*max(K) < W(p_{K+1})` (the localised budget proves the open lemma) | `K = 1..10` | 10 | exact; fails at `K = 11` (`632.9 > 308`) |
| `ln(cut gear)/ln q` rising to `0.728` | `q = 59, 97, 199, 499, 997` | 5 | measured `0.629, 0.644, 0.702, 0.703, 0.713` |

# The k-class Jacobsthal family `j_k`

**Status: NOTE DRAFT (round 27, 2026-08-29).** Standalone, publishable as a
short note or as the closing section of Unit 1. Every numerical and finite
claim below is asserted in `research/jk_family.py` (**ALL ASSERTIONS GREEN**,
2026-08-29); the two hypotheses the upper rung needs from Opera de Cribro are
asserted in `research/j2_odcpages.py` (**ALL ASSERTIONS GREEN**, same date,
from page images read first-hand this round).

Item **(P6)** of the harvester ranking, opened at round-26 close: *`j_k` is
defined, has a proved lower bound at every `k`, has a stated upper conjecture,
and appears nowhere in the literature.*

---

## 1. What it is

Two functions in the literature are the `k = 1` and `k = 2` cases of one
family, and nobody has written the family down.

* `k = 1` is the **ordinary Jacobsthal function** `j(m)` - the largest gap
  between consecutive integers coprime to `m` (Jacobsthal 1960; at primorials,
  OEIS A048669).
* `k = 2` is **Ziller-Morack's paired Jacobsthal function** `j_2 = h_2`
  (arXiv:1706.00317) - the largest gap between consecutive `n` with `n` and
  `n+E` both coprime to `m`, maximised over even `E`.

**DEFINITION.** Let `k >= 1` and let `E = (0 = E_0 <= E_1 <= ... <= E_{k-1})`
be an **admissible** `k`-tuple: for every prime `p`, `E` does not meet every
residue class mod `p`. For squarefree `m` put

    j_E(m) = the largest gap between consecutive integers n with
             gcd( (n+E_0)(n+E_1)...(n+E_{k-1}), m ) = 1,

the gap taken cyclically over one period `m`, and

    j_k(m) = max over admissible k-tuples E of j_E(m).

(Repeats among the `E_i` are permitted and are sometimes optimal at small `m`;
they simply mean the tuple spends fewer than `k` classes at some primes. The
`k = 2` maximum is over even `E` precisely because an odd `E` gives an
inadmissible tuple at `p = 2`.)

**PROPOSITION 1 (the covering restatement).** For every `z >= 2`,

    j_k(P(z)) - 1  =  the length of the longest interval coverable by
                      choosing, at each prime p <= z, a set S_p of residue
                      classes mod p with

                            |S_p|  <=  min(k, p-1).

*Proof.* Both directions are the Chinese Remainder Theorem. (⇒) The classes
killed by `E` at `p` are `-t + {E_i mod p}` for the interval's translation
`t`; that set has size at most `k`, and at most `p-1` by admissibility.
(⇐) Given a family `(S_p)`, choose for each `p <= z` a surjection
`{0,...,k-1} -> S_p` and let CRT define the `E_i`; then `{E_i mod p} = S_p` at
every `p <= z`, and the tuple is admissible because `|S_p| <= p-1`. ∎

**The single expression `min(k, p-1)` is the whole content of the family.** It
reproduces the two known cases exactly - `1` everywhere at `k = 1` (the
ordinary covering problem), and `1` at `p = 2` with `2` at every odd `p` at
`k = 2`, which is Ziller-Morack's `omega(2) = 1`, `omega(p) = 2` and our
density `g(2) = 1/2`, `g(p) = 2/p`. And it is what makes **the sifting
dimension of the `k`-th member equal to `k`**, which is where every bound
below comes from.

Both forms were brute-forced independently and agree at every case computable
by exhaustion (`jk_family.py` section A):

| `k` | `z` | `j_k(P(z))` shift form | witness `E` | covering form |
|---|---|---|---|---|
| 1 | 3 | 4 | (0) | 4 |
| 1 | 5 | 6 | (0) | 6 |
| 1 | 7 | 10 | (0) | 10 |
| 2 | 3 | 6 | (0,2) | 6 |
| 2 | 5 | 18 | (0,4) | 18 |
| 2 | 7 | 30 | (0,2) | 30 |
| **3** | **3** | **6** | (0,0,2) | 6 |
| **3** | **5** | **24** | (0,2,6) | 24 |
| **3** | **7** | **78** | (0,2,18) | 78 |

The `k = 1` row is A048669 (`4, 6, 10`); the `k = 2` row is Ziller-Morack's
`h_2 = 6, 18, 30`. **The three `k = 3` values are, as far as this lane can
establish, the first evaluation of the function** - and their smallness is the
point: the object is elementary, computable by hand, and unnamed.

---

## 2. Why it might be novel

The bounds below are **standard sieve theory applied to a new object**, and
the note must say so in those words (this is Unit 1's not-claim 2, inherited).
What is not standard is that nobody has applied it, because nobody has written
the object down.

* **`j_k` appears nowhere under any name.** Prior-art sweeps in rounds 25 and
  26 (citation graph, not keywords) and the re-check of section 7 below.
* The one paper that names our `k = 2` sieving system in print -
  Ford-Konyagin-Maynard-Pomerance-Tao, arXiv:1802.07604, **Remark 7** - names
  it in order to say their methods do not reach it, and treats it as a
  two-dimensional instance of a one-dimensional theory. They do not define a
  family.
* Kalmynin-Konyagin (arXiv:2302.00459) generalise the *progression* `x+i` to
  polynomial values and stay **one-dimensional**. Different axis.
* Ziller-Morack define `j_2` and conjecture an upper bound; they prove no
  lower bound of any kind and state no general `k`.

**The honest one-sentence description of the contribution:** *the ordinary and
the paired Jacobsthal functions are the first two members of a family indexed
by the sifting dimension; the classical ladder is uniform in `k`; and the one
genuinely new theorem, the layered lower bound, holds at every `k`.*

---

## 3. The two ladders, at every `k`

### 3a. Upper: the Legendre rung

`j_k(P(z)) <= prod_{p<=z} (1 + omega_p) / V_k(z) + 1`, with
`omega_p = min(k, p-1)` and `V_k(z) = prod_{p<=z}(1 - omega_p/p)`. At `k = 2`
the numerator is `1*2 * 3^{n-1} = 2*3^{n-1}`, which is Unit 1's Theorem 1
verbatim (asserted at `z = 3, 5, 7`). Checked against the exact values of
section 1: the bound holds at every computed case, with ratios
`0.24 - 0.007`.

### 3b. Upper: the polynomial rung, explicit

Everything in Unit 1's Theorem 2G is uniform in the dimension. Opera de Cribro
Corollary 6.13 gives, at dimension `kappa`,

    beta_kappa = 1 + 2 ( e^{1/(2 kappa)} - 1 )^{-1} ,

so

    **THEOREM (the family's polynomial rung).**  For every k >= 1,

        j_k(P(z))  <<_{k,eps}  z^{beta_k + eps} ,
        beta_k = 1 + 2(e^{1/(2k)} - 1)^{-1} ,     4k - 1 < beta_k < 4k + 1 ,

    and with pre-sieving in place of ODC's own preliminary sifting, every
    constant is computable, exactly as in Theorem 2G.

| `k` | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| `beta_k` | 4.082988 | **8.041623** | 12.027765 | 16.020828 | 20.016664 | 40.008333 | 80.004167 |

`beta_2 = 8.041623` is Theorem 2G's exponent, so the family rung *contains*
Unit 1's best explicit upper bound as its `k = 2` case.

**The hypotheses were checked first-hand this round, and they hold at every
`k` at once.** ODC Proposition 6.7 requires `g` to satisfy **(5.38)** with
`kappa` bounded by **(6.69)**; rounds 24-26 had read neither equation. Both
were fetched and read on 2026-08-29 (`research/data/odc6_scans/PA42.png`,
`PA67.png`) and are asserted in `research/j2_odcpages.py`:

* **(5.38)** (p. 42) is `prod_{w<=p<z}(1-g(p))^{-1} <= K (log z/log w)^kappa`
  for `z > w >= 2`, `K > 1` - the form we had used, from the book. It also
  prints the consequence `g(p) <= 1 - 1/K`, whose converse
  `K >= (1-g(p))^{-1}` explains our measured `K`-ladder in one line
  (`K = 3` at `p_0 = 3` because `g(3) = 2/3`; `5/3` at `p_0 = 5`; `7/5` at
  `p_0 = 7`).
* **(6.69)** (p. 67) is `kappa < (2/c)(log((beta+1)/(beta-1)))^{-1}` with `c`
  the root of `c(log c - 1) = 1` - i.e. **exactly `alpha < 1/c = 0.2784645`**,
  the convergence condition `a = alpha e^{1+alpha} < 1` of (6.67). And
  Corollary 6.13's own `beta_kappa` gives `(beta+1)/(beta-1) = e^{1/(2kappa)}`,
  hence `alpha = 1/4` **identically in `kappa`**. So (6.69) holds for every
  `kappa > 0`, which is why the book states the corollary "for `kappa > 0`".

The one arithmetic change with `k` is the remainder: `|r_d| <= k^{nu(d)}`, so
`sum_{d<D} k^{nu(d)} << D (log D)^{k-1}` in place of `D(log D)`. The level is
still `D = m^{1-o(1)}` and the exponent is still exactly `beta_k`.

### 3c. Lower: the layered Erdős-Rankin rung

**THEOREM (P2', round 26; `docs/novel/layered-erdos-rankin.md` §4).** Let
`pi_E(t) <= c_1^{(k)} t/(log t)^k` for the admissible `k`-tuples in play.
With `A = log x`, `B = log A`, `C = log B`,

    j_k(P(x))  >=  ( K_k + o(1) ) x A^{2k-1} C^k / B^{2k} ,
    K_k = k / ( (k(2k-1))^k c_1^{(k)} ) .

At `k = 1` this is `(1/c_1^{(1)} + o(1)) x A C/B^2 = (1+o(1)) x A C/B^2`, a
factor `e^gamma = 1.781` below Rankin's proved constant - the correct side. At
`k = 2` it is `1/(18 c_1) = 0.0127524` with Lichtman's record twin constant.

### 3d. The sandwich, and the conjecture

    x A^{2k-1} C^k / B^{2k}   <<   j_k(P(x))   <<   x^{beta_k + eps},
                                                     beta_k ~ 4k

**CONJECTURE.** `j_k(P(x)) = x (log x)^{2k-1 + o(1)}` for every `k >= 1`.

At `k = 1` this is the standard expectation for the Jacobsthal function; at
`k = 2` it is round 26's sharpened form of (P3); at `k >= 3` it is new because
the object is. **Any claimed upper bound `j_k << x A^{f(k)}` with
`f(k) < 2k-1` is contradicted outright by (P2') at that `k`** - the family
supplies free consistency checks on a whole sequence of claims, which is the
practical reason to publish it even though each individual rung is standard.

---

## 4. The `k >= 4` shift-set question - ANSWERED

This was the family's one named piece of real work
(`layered-erdos-rankin.md` §6 item 3, round 26): the layered construction puts
class `-E_i mod p` in layer `i`, and for `k >= 4` the shifts `0,2,...,2(k-1)`
are not pairwise distinct modulo every odd prime (`3 | 6` already at `k = 4`).
Round 26 wrote "the finitely many offending primes can be set aside, but a
clean statement wants the optimal shift set."

**It costs nothing, and here is why.**

1. `0, 2, ..., 2(k-1)` is the wrong tuple: from `k = 3` it is not even
   admissible (`0,2,4` covers all of `Z/3`, so no `n` survives). The
   construction needs an admissible tuple, and admissible `k`-tuples exist for
   every `k` - e.g. `E = {q_1,...,q_k} - q_1` for the `k` least primes
   `q_i > k`, admissible because no `q_i` is divisible by any `p <= k` (so the
   tuple misses class `-q_1` there) and because it has only `k <= p-1`
   elements for `p > k`.
2. With any admissible tuple, a collision `E_i ≡ E_j (mod p)` can occur only
   at a prime dividing a pairwise difference, hence only at
   `p <= M_k := max_{i<j}(E_j - E_i)`, **a constant depending on `k` alone**.
   The construction's greedy layer runs over `[P, z1]` with `P = A^{2k-1} → ∞`,
   so for large `x` **every colliding prime lies below `P`**, i.e. inside the
   Eratosthenes layers, never inside the greedy layer. And a collision inside
   the Eratosthenes layers costs nothing: layers `i` and `j` coincide at that
   `p`, which uses *fewer* than the `k` available classes, while the survivor
   structure ("no `n+E_i` has a prime factor in `[3,P) ∪ (z1, x/L]`") is
   untouched. So `Sigma = prod_{P<=p<=z1}(1 - k/p)` needs no correction and
   `K_k` stands as printed.
3. **The threshold is trivial:** it is enough that `A^{2k-1} > M_k`, i.e.
   `x > exp(M_k^{1/(2k-1)})`, which is under `e^4` for every `k <= 12`
   (tabulated in the gate) against the construction's own threshold of
   `log x ~ 300`.

What remains of the question is a genuinely finite optimisation and **not a
gap in the theorem**: which admissible `k`-tuple minimises `c_1^{(k)}`
(equivalently the singular series `S(E)`)? That moves the constant `K_k` and
nothing else.

**A simplification of our own argument, recorded while here.** The greedy
lemma at general `k` - *some `k` distinct classes mod `p` contain together at
least `kN/p` elements of any `N`-element set* - has a one-line proof: the `p`
class counts average `N/p`, so the `k` largest of them average at least `N/p`
and sum to at least `kN/p`. This subsumes round 26's `k = 2` lemma, whose
proof (via `n_(1) >= N/p`, `n_(2) >= (N-n_(1))/(p-1)` and monotonicity) was
correct but longer than it needed to be. The `k = 2` statement `2N/p` was and
is exact.

---

## 5. Implications

1. **Unit 1 gains a closing section with a genuinely new object in it.** Every
   rung of Unit 1 becomes an instance, and the paper's weakest structural
   point - "these are standard sieves applied to one function" - becomes
   "these are standard sieves applied to a family, and the family is the
   contribution."
2. **The family is a referee tool.** `2k-1` distinct instances of the same
   question, each with a proved lower bound, each free to check any future
   upper-bound claim against.
3. **It locates Ziller-Morack Conjecture 6 in a family.** Their conjecture is
   `j_2 < p^2 - p`, i.e. exponent 2 at dimension 2. The family's proved
   exponent is `beta_k ~ 4k` and its conjectured truth is `1 + o(1)`, so the
   conjecture asks for exponent `k` at dimension `k` - the level at which a
   survivor in `(y, y^2]` is a prime `k`-tuple. **The parity obstruction of
   Unit 1's "ceiling" section is therefore uniform in `k`**, not special to
   twins.
4. **It sharpens what the `j` vs `j_2` separation is.** Each class spent as a
   split-range Eratosthenes layer converts an `O(1)` Mertens entitlement into
   a full log of thinning; the `k`-th layer's survivors are prime `k`-tuples,
   `k` logs thinner. The family says the one-log separation between `j` and
   `j_2` is the first step of an arithmetic progression in `k`.

---

## 6. Unsolved questions it touches

1. **Erdős problems #687 (with a $1000 prize) and #970** - both open, both the
   `k = 1` case of the family's upper question (re-checked 2026-08-29 via
   round 26's sweep). Iwaniec 1978's `j(P(z)) << z^2` is still the record
   after 48 years, and **it is better than the family rung at `k = 1`**
   (`beta_1 = 4.083 > 2`). Stated plainly: the family rung is the only bound
   in existence for `k >= 2` and is *not* the best bound at `k = 1`.
2. **(P3), the paired-Iwaniec problem**, and its family form: is
   `j_k << x A^{a}` for some `a`? Priced NOT REACHABLE in round 26 -
   `j_2 >= j` by the collapse transfer, so a polylog bound at `k = 2` gives
   one at `k = 1`, which is Erdős #687.
3. **Exact values.** `j_3(P(z))` for `z >= 11` is a finite computation nobody
   has run. `h_2` beyond `p_n = 73` remains the single most decisive
   purchasable number for the `k = 2` growth law (it now separates
   `z(log z)^2` from `z(log z)^3`).
4. **The `loglog` exponent `2k`** in (P2'). It comes from `sigma ~ B^2` and is
   not optimised; sharpening `rho` and the greedy moves the constant, not the
   exponent.

---

## 7. Prior-art check (round 27, dated **2026-08-29**)

Re-run because checks expire (harvester 7d clause 1). This lane's round-26
sweep - citation graph rather than keywords - is 0 days old and its verdict
"`j_k` appears nowhere under any name" is carried forward. Run again this
round, first-hand:

* **Web search**, `"Jacobsthal function" + generalisation + k-tuple / prime
  constellation / residue classes per prime`, 2026-08-29: returns the ordinary
  function (Hagedorn's computation of `h(n)`; Ziller arXiv:1611.03310;
  Kalmynin-Konyagin arXiv:2302.00459), the long-gaps literature
  (FGKT arXiv:1408.4505, FKMPT arXiv:1802.07604, Maynard), and OEIS A048669 /
  A048670. **No `k`-class family, at any `k >= 3`, under any name.** The one
  adjacency worth naming in the note: computations of the ordinary `h(n)` use
  admissible-set techniques descended from Gordon-Rodemich's work on the prime
  `k`-tuples conjecture, so the *technique* has met `k`-tuples - the
  *function* has not.
* **Web search**, `"paired Jacobsthal" / j_k / k classes per prime / covering
  interval`, 2026-08-29: only the two Ziller-Morack records
  (arXiv:1706.00317, arXiv:1706.03668) and the long-gaps literature.
* **OEIS**, 2026-08-29 (the search endpoint 403s to the fetch tool; queried
  directly instead, `oeis.org/search?...&fmt=text`):
  * `seq:6,24,78` returns **19 sequences, none of them number-theoretic in our
    sense** - permutations within a bounded displacement (A002526, A306209,
    A263703...), acyclic orientations of Turán graphs, billiard words, a
    linear recurrence (A276179). **Zero mentions of Jacobsthal, primorial,
    coprimality or residue-class covering across all 19.**
  * `jacobsthal function primorial` returns **6 sequences**: A048669 (`j`),
    A048670 (`j` at primorials), A128759, A058989 and A049300 (the
    one-class covering form of the same object), A319148. **No `k`-class
    analogue at any `k >= 3`, and A288815 - the paired case - is the only
    other member of the family in OEIS at all.**

**VERDICT: NOVEL AS FAR AS SEARCHED**, with two explicit qualifications
carried from round 26 - the `k = 2` order `z log z` for this system is in
print as "the trivial bound" (FKMPT Remark 7), and the upper rungs are
standard sieve theory applied to a new object, not new sieve theory.

---

## 8. Reproduction

* `research/jk_family.py` → `research/data/jk_family.out`. Sections A (the
  definition and the covering restatement, both forms brute-forced at
  `k = 1,2,3` and `z = 3,5,7`), B (`beta_k` at every `k`, and the honest
  `k = 1` comparison with Iwaniec), C (the Legendre rung against the exact
  values), D (the lower constant `K_k`), E (the `k >= 4` shift-set question
  answered, with the admissible tuples, `M_k` and the thresholds tabulated to
  `k = 12`), F (the general-`k` greedy lemma, 40,000 random distributions).
  **ALL ASSERTIONS GREEN, 2026-08-29.**
* `research/j2_odcpages.py` → `research/data/j2_odcpages.out`. Sections A
  ((5.38)), B ((6.69)), C (p. 74), D (what moved in Unit 1). **ALL ASSERTIONS
  GREEN, 2026-08-29.**
* Page images: `research/data/odc6_scans/PA42.png`, `PA43.png`, `PA44.png`,
  `PA45.png`, `PA67.png`, `PA74.png` (fetched 2026-08-29), alongside round
  25's `PA65`, `PA68`-`PA73`, `PA112`.
* Companions: `docs/novel/layered-erdos-rankin.md` (the `k = 2` proof in
  full), `docs/novel/j2-upper-bound.md` §11 (Unit 1 as assembled),
  `docs/novel/j2-lower-ladder.md`.

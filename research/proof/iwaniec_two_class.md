# Iwaniec's Jacobsthal bound, made explicit, and carried to two classes

PROVER D, branch 3a of `research/proof/theory_tree.md`.  Brief: obtain Iwaniec's 1978 argument
with every constant, redo it for the two-class sieve, and decide whether it can give
`F(y) <= C_2 y^2` with `C_2 < 1/6` (the kernel route `BlockedSlots.twins_infinite_iff_survivor_in_window`).

**Verdict in one paragraph.**  `C_2` does not exist by this route.  Iwaniec's argument is
*shifted sieve* (1978, new) on top of *Rosser's linear sieve with sharp remainder* (1971, the
engine).  Every constant in it is traceable and two of them (`c_0`, `c_1` of Iwaniec 1971
Theorem 1 / Corollary) have never been made explicit by anyone; but the constant is not what
breaks in two classes.  The shifted-sieve step and the sieve-to-covering step lose nothing.
All the loss is in the sieve lower bound, and in two classes it is not a constant but an
exponent: the two-class sieve has dimension 2, whose lower-bound function `f_2(s)` is
identically zero for `s <= beta_2 = 4.2664` (Diamond-Halberstam-Richert), while the window
statement sits at `s = log(y^2)/log y = 2`.  The transfer yields `F <= C y^{4.27 + eps}`, not
`C y^2`.  A rigorous finite-size version of the same certificate (pre-registered, below) gives
two-class bounds that exceed `(z^2 - z)/6` at every `z >= 8` tested and grow like `z^{3.5..3.7}`
on `z <= 72`.  The method uses only the NUMBER of classes, so it bounds Ziller-Morack's `h_2`;
and any class-count-only bound below `y^2/6` would prove the twin prime conjecture outright
(ZM 2017 Theorem 4.1 = the kernel route), so no "improvement of the constant" can get there.
Branch 3a is DEAD as a sieve route; branch 3 survives only with the real teeth, i.e. as the
twin prime problem itself.

Verification key: `[READ]` I read the text (OCR of the open-access scan, or the PDF's text
layer, or the scanned page image); `[SECONDARY]` reported by a paper I read; `[COMPUTED]` exact
integer/float computation this round, script inline below; `[MINE]` my derivation.

---

## 1. Sources

| item | what | how obtained | status |
|---|---|---|---|
| Iwaniec, *On the problem of Jacobsthal*, Demonstratio Math. 11 (1978) 225-231, DOI 10.1515/dema-1978-0121 | the theorem, the shifted sieve (Lemma 1, with proof), Lemma 2 quoted from [1971], the assembly | De Gruyter open-access page, full OCR text in the HTML (the PDF endpoint refuses non-browser clients) | `[READ]` |
| Iwaniec, *On the error term in the linear sieve*, Acta Arith. 19 (1971) 1-30 | Theorem 1 (1.1)-(1.2), Corollary (1.3)-(1.4), Theorem 2 `C_0(r) << r^2 log^2 r`, Lemma 1 (Rosser scheme) | matwbn.icm.edu.pl scan, fetched through the browser session, page images 1-5 read | `[READ]` pp. 1-5 |
| Hagedorn, Math. Comp. 78 (2009) 1073-1087 | "(0.1) j(n) << log^2 n ... due to Iwaniec"; "(1.1) h(n) << n^2 log^2 n ... proved using sieve theory"; Kanold `2^n`; Stevens | PDF text layer | `[READ]` |
| Costello-Watts, arXiv:1306.1064 (2013) | "Iwaniec's proof that g(n) <= X (k log k)^2 for some unknown constant X" | PDF text layer | `[READ]` |
| Diamond-Halberstam-Richert sieve, dimension 2: `alpha_2 = 5.3577`, `beta_2 = 4.2664` | the sifting limit `f_2(s) = 0` for `s <= beta_2` | Kao, arXiv:1606.03505 (2016), text: "For kappa = 2, it is known that alpha_2 = 5.3577... and beta_2 = 4.2664..." | `[SECONDARY]` via Kao; DHR's book *A Higher-Dimensional Sieve Method* (Cambridge Tracts 177, 2008) not obtained |
| Franze, arXiv:1012.3809 (2010), *Sifting limits for the Lambda^2 Lambda^- sieve* | Selberg's lower-bound sieve beats DHR only for `kappa >= 3` | abstract | `[READ abstract]` |
| OEIS A288815 (`h_2`, 21 terms), A048670 (`h`, 64 terms) | the published values used in section 6 | b-files via the browser | `[READ]` |
| Ford-Konyagin-Maynard-Pomerance-Tao, *Long gaps in sieved sets* | Remark 7: "our methods only seem to give good results in the one-dimensional case ... a two-dimensional system in which `I_p = {0, 2} (mod p)`" | PDF text layer | `[READ]`, corroborates the dimension diagnosis |

Not obtained: DHR 2008 (book), Jurkat-Richert 1965/1969, Vaughan 1976-77 (`C(r) << r^2 log^4 r`,
cited by Iwaniec in the added-in-proof note), Erdos problem #687 page (403 this round).

---

## 2. Iwaniec 1978, the argument in full

### 2.1 Objects

`A` = a set of `X` consecutive integers.  `Q = q_1 ... q_r` a product of `r` arbitrary distinct
primes.  `S(A, Q) = #{a in A : (a, Q) = 1}`.  For `d | Q`, `|A_d| = #{a in A : d | a} = X/d + rho_d`
with `|rho_d| < 1` (an interval of `X` integers contains `floor` or `ceil` of `X/d` multiples of `d`).
Iwaniec writes this as his condition (R) with `A = 1, B = 1, f(d) = d`.  `[READ]`

**Theorem (Iwaniec 1978).**  There is an absolute constant `c > 0` such that for arbitrary primes
`q_1 < ... < q_r`, `r > 1`, every interval of length `c (r log r)^2` contains at least `r`
integers coprime to `q_1 ... q_r`.  **Corollary.**  `C(r) << r^2 log^2 r`.
(OCR garbles the exponent typography; the corollary line and the final line of the proof --
"the right hand side of (7) is `> y/log^2 z > r`" -- fix the reading.  Added in proof: Vaughan
[Proc. Edinburgh Math. Soc. 20 (1976-77) 329-331] had derived `C(r) << r^2 log^4 r` from the same
1971 paper.)  `[READ]`

Introduction, verbatim in substance: the sieving limit of the linear sieve "being occured here"
equals 2 (Jurkat-Richert), "so it easily follows `C(r) < c(eps) r^{2+eps}` for all `eps > 0`.  Of
course by the sieve method the exponent 2 cannot be reduced, however some small improvements are
still possible if the error term in the sieve problem is allowed."  For the first `r` primes,
`C_0(r)`, Jurkat-Richert give `C_0(r) << r^2 exp(log r)^{13/14}` and [1971] gives
`C_0(r) << r^2 log^2 r`.  The 1978 paper's only new content is the passage from the first `r`
primes to arbitrary `r` primes.  `[READ]`

### 2.2 Lemma 1, the shifted sieve (new in 1978; "inspired by Halberstam-Richert, Acta Arith. 18 (1971)")

Setting.  `Q` squarefree with `|A_d| = X/f(d) + rho_d` for `d | Q`, `f` multiplicative, `f > 1`.
`P` another squarefree number with the same number of prime factors, `g` multiplicative with
`g(n) > 1` for `n | P`, and `l` a multiplicative bijection from the divisors of `P` to those of `Q`
(prime to prime).  Hypothesis (2): `g(n) <= f(l(n))` for every `n | P`.

Sieve weights `(lambda_n)_{n | P}`, real, with `lambda_1 = 1`, satisfying either
`(+)  sum_{n | m} lambda_n >= 0` for all `m | P`, or `(-)  sum_{n | m} lambda_n <= 0` for all `m | P, m > 1`.

**Lemma 1 (lower half, (4)).**  Under `(-)`,

    S(A, Q)  >=  X * prod_{q | Q} (1 - 1/f(q)) * [prod_{p | P} (1 - 1/g(p))]^{-1} * sum_{n | P} lambda_n / g(n)
                 -  sum_{n | P} |lambda_n| |rho_{l(n)}|.

Proof (Iwaniec gives the `(-)` case in eight lines; this is that proof, written out).  `[READ]`+`[MINE]`
for the two identities.

(i)  For each `a in A` let `m(a) = gcd(a, Q)`.  By `(-)`, `[m(a) = 1] >= sum_{d | m(a)} lambda_{l^{-1}(d)}`.
Summing over `a`: `S(A,Q) >= sum_{d | Q} lambda_{l^{-1}(d)} |A_d| = X sum_{d|Q} lambda_{l^{-1}(d)}/f(d) + sum_d lambda_{l^{-1}(d)} rho_d`.
The second sum is `>= - sum_{n|P} |lambda_n| |rho_{l(n)}|`.

(ii)  Identity, for squarefree `Q` and any multiplicative `f` with `f(q) != 1`:

    1/f(d)  =  prod_{q | Q} (1 - 1/f(q))  *  sum_{k : d | k | Q}  prod_{q | k} 1/(f(q) - 1).

Check: the right side is `prod_{q|Q} (f(q)-1)/f(q) * prod_{q|d} 1/(f(q)-1) * prod_{q | Q/d} (1 + 1/(f(q)-1))`
`= prod_{q|Q} (f(q)-1)/f(q) * prod_{q|d} 1/(f(q)-1) * prod_{q|Q/d} f(q)/(f(q)-1) = prod_{q|d} 1/f(q)`.
Hence

    sum_{d|Q} lambda_{l^{-1}(d)}/f(d)  =  prod_{q|Q}(1 - 1/f(q))  *  sum_{k | Q} w_Q(k) T(k),
    w_Q(k) = prod_{q|k} 1/(f(q)-1),   T(k) = sum_{d | k} lambda_{l^{-1}(d)} = sum_{n | l^{-1}(k)} lambda_n.

(iii)  By `(-)`, `T(1) = 1` and `T(k) <= 0` for `k > 1`.  By hypothesis (2), `f(q) - 1 >= g(p) - 1`
for `p = l^{-1}(q)`, so `w_Q(k) <= w_P(l^{-1}(k))` where `w_P(m) = prod_{p|m} 1/(g(p)-1)`.  A non-positive
term multiplied by a LARGER non-negative weight gets smaller, so
`sum_{k|Q} w_Q(k) T(k) >= sum_{m|P} w_P(m) sum_{n|m} lambda_n`.

(iv)  The same identity read backwards for `P` and `g`:
`sum_{m|P} w_P(m) sum_{n|m} lambda_n = [prod_{p|P}(1 - 1/g(p))]^{-1} sum_{n|P} lambda_n/g(n)`.
Chain (i)-(iv).  QED.

**What the lemma does.**  It transports the sieve WEIGHTS built for the modulus `P` to the modulus
`Q` and says the main term for `Q` is at least the main term for `P` times the ratio of the two
singular products.  Only `sum_{n|m} lambda_n <= 0` (the lower-bound sieve axiom) and
`g(p) <= f(q)` are used.  Nothing about which residue class is removed enters, and nothing about
the number of classes enters except through `f` and `g`.

### 2.3 Lemma 2, the engine (quoted by Iwaniec 1978 from Iwaniec 1971)

Let `z > 2`, `y > z`, `P = prod_{p < z} p`, and `lambda_n` the Rosser lower-bound weights of level `y`:
`lambda_n = mu(n)` if `n = p_1 ... p_u`, `p_1 > ... > p_u`, with `p_1 ... p_{2l-1} p_{2l}^3 < y` for
every `2l <= u`; `lambda_n = 0` otherwise.  These satisfy `(-)` (truncating the Buchstab iteration
at even steps drops non-negative terms), and `lambda_n = 0` for `n >= y`.  The trivial remainder
bound is `sum_{n|P} |lambda_n| <= y`, "too weak to prove (1)".  Iwaniec then quotes two estimates:

    (5)  sum_{n | P} |lambda_n|  <  y (log y)^{-2}          [as OCR'd; the constant in front, if any, is unreadable]
    (6)  sum_{n | P} lambda_n / n  =  prod_{p < z} (1 - 1/p) * { 2 e^gamma log(s-1)/s  +  O(1/log y) },   s = log y / log z,

"for `4 < z^2 < y < z^4`" (last exponent unreadable; `y = C z^2` is all that is used).  Verbatim:
"The proof is very complicated and utilizes some differential equations with shifted arguments.
These results seem to be best possible.  Note that in the Selberg upper bound method the remainder
term can also be estimated by the value of order `y/log y` (see [5])."  `[READ]`

What Iwaniec 1971 actually states (scan pp. 1-3, `[READ]`).  Theorem 1: for `M` any set with
`||M_d| - y/d| <= 1` for `(d,k) = 1` (condition (*)), `y > 1`, `s < log y/(log log 3y)^{11/5}`,
there is an absolute constant `c_0` with

    (1.2)  A_k(M; y^{1/s})  >  y R_k(y^{1/s}) { 1 - f(s)/s - c_0 (1 + s^2 log^5 s / log^2 y)^{5s} MM(s) log log 3k / log y },   s >= 2,

`f(s) = s - 2 e^gamma log(s-1)` for `2 <= s <= 4` (so `1 - f(s)/s = 2 e^gamma log(s-1)/s`, the
linear-sieve lower function), `MM(s) = exp{-s(log s + log log s - 1 + log log s/log s) + O(s/log s)}`.
Corollary (1.4): `A_k(M; z) > y R_k(z) { 1 - f(s)/s - c_1 log log 3k / log y }`.  Theorem 2:
`C_0(r) << r^2 log^2 r`, "Corollary implies easily".  **`c_0` and `c_1` are "absolute constants"
and are not given numerically anywhere in the paper's statement; `MM(s)` itself carries an `O`.**
This is the origin of the "unknown constant" that Hagedorn, Mercer and Costello-Watts all report,
and I confirm it at the source.

Check of (5)'s order `[COMPUTED]`: the number of Rosser lower-bound supports `n < y`, `n | P(z)`,
at `y = 4 z^2` is `1.19, 1.11, 1.13, 1.15, 1.14` times `y/log^2 y` at `z = 50, 100, 200, 300, 400`
(support 140, 394, 1260, 2528, 4070); at `y = 16 z^2` the ratio is 0.80-0.86, at `y = 64 z^2`
0.55-0.62.  So (5) is the right order with a constant near 1 at `s ~ 2.25`, and the constant
depends on `s`.  Check of (6): the exact `sum lambda_n/n` is `1.74 .. 1.97` times
`V(z) * 2 e^gamma log(s-1)/s` at `s = 2.23..2.35`, `1.33..1.38` at `s ~ 2.5`, `1.17..1.18` at `s ~ 2.85`
-- the `O(1/log y)` term is positive and, at these sizes, comparable to the main term.

### 2.4 The assembly (section 3 of the 1978 paper), with the constants tracked

Take `z = p_r` (the `r`-th prime), `P = prod_{p <= p_r} p` -- the first `r` primes -- and
`l(p_i) = q_i`.  With `f(d) = d`, `g(n) = n`, hypothesis (2) is `p_i <= q_i`, true because the
`i`-th smallest of any `r` primes is at least the `i`-th prime.  Lemmas 1 and 2 give

    (7)  S(A, Q)  >=  X prod_{q|Q}(1 - 1/q) { 2 e^gamma log(s-1)/s + O(1/log y) }  -  sum_{n|P} |lambda_n|
                  >=  X prod_{q|Q}(1 - 1/q) { 2 e^gamma log(s-1)/s - c_1'/log y }  -  c_2 y / log^2 y.

Put `y = C z^2`, so `s = 2 + log C/log z` and `2 e^gamma log(s-1)/s = e^gamma log C/log z * (1 + O(log C/log z))`.
Put `X = [prod_{q|Q}(1 - 1/q)]^{-1} y / log z`.  Then the main term is
`y (e^gamma log C - c_1'/2 + o(1)) / log^2 z` and the remainder is `c_2 y/(4 log^2 z)(1 + o(1))`, so

    S(A, Q)  >=  y (e^gamma log C - c_1'/2 - c_2/4 + o(1)) / log^2 z   >   y / log^2 z   >=  r

once `C` exceeds `exp((c_1'/2 + c_2/4 + 1) e^{-gamma})` and `z` is large (`C z^2/log^2 z >= r` is
`C r^2 (1+o(1)) >= r`).  Finally `prod_{q|Q}(1 - 1/q)^{-1} <= prod_{p <= p_r}(1 - 1/p)^{-1} = e^gamma log p_r (1 + o(1))`
(Mertens; the product over arbitrary `r` primes is at least the product over the first `r`), so

    X  <=  e^gamma (1 + o(1)) y  =  e^gamma C (1 + o(1)) p_r^2  =  e^gamma C (1 + o(1)) (r log r)^2.

**The constant of Iwaniec's theorem is `c = e^gamma C`, with `C = exp((c_1'/2 + c_2/4 + 1) e^{-gamma})`,
where `c_1'` is Iwaniec 1971's `c_1` (never numerical) and `c_2` the constant of (5) (measured ~1.1-1.2
at `s ~ 2.25`, not stated numerically in print).**  Where each factor comes from:

| step | factor | loss? | source |
|---|---|---|---|
| shifted sieve (Lemma 1) | `prod(1-1/q)/prod(1-1/p) >= 1` | none -- the first `r` primes are the worst case | Iwaniec 1978 |
| sieve lower function at `s = 2 + delta` | `2 e^gamma log(1+delta)/(2+delta) ~ e^gamma delta` | THE binding term: it vanishes at `s = 2`, which is why the exponent 2 is the sieve limit and why `C` must be "large" | Jurkat-Richert 1965/1969, Iwaniec 1971 (1.2) |
| remainder `sum |lambda_n| <= c_2 y/log^2 y` | `c_2 y/log^2 y` against a main term `~ y log C/log^2 z` | this is what Selberg's `y/log y` cannot do; Iwaniec 1971 is the ONLY reason the exponent is exactly 2 | Iwaniec 1971 |
| `|rho_d| < 1` | exact for intervals | none | arithmetic |
| passage sieve -> covering | `S > 0` is the covering statement | none | definition |

**Which sieve inequality:** Rosser's combinatorial (Brun-type, beta = 2) sieve in Iwaniec's 1971
sharp-remainder form.  Not the large sieve (which bounds survivors from ABOVE and says nothing about
a covering), not Selberg's upper bound (Iwaniec explicitly notes its remainder is `y/log y`, too big
at `s -> 2`), not Montgomery's.  `[READ]`

### 2.5 A fully explicit finite version of the one-class argument `[COMPUTED]`

Drop Lemma 2 and use only what is exact: the Rosser lower-bound weights (verified to satisfy `(-)`
on all 32,767 proper divisors of `P(50)`) and `|rho_d| < 1`.  Then for the first `r` primes
(`Q = P`, the Lemma-1 worst case) every interval of `X` integers has
`S >= X M - R`, `M = sum lambda_n/n`, `R = sum |lambda_n| = #support`, so `j(P(z)) <= floor(R/M) + 1 =: X0_1(z)`,
minimised over the level `y = C z^2`, `C in {0.6, ..., 3}`:

    k   p_k    X0_1   best C   support   M        X0_1/p_k^2   h(k) [A048670]   X0_1/h(k)
     5   11      81    1.0        12   0.1498     0.669          14              5.8
    10   29     361    0.7        34   0.0942     0.429          46              7.8
    15   47     873    0.8        68   0.0779     0.395         100              8.7
    20   71    1635    0.6       100   0.0612     0.324         174              9.4
    25   97    2703    0.6       152   0.0562     0.287         258             10.5
    30  113    4066    0.6       200   0.0492     0.318         330             12.3
    40  173    7694    0.6       362   0.0471     0.257         538             14.3
    50  229   12616    0.6       558   0.0442     0.241         762             16.6
    60  281   18871    0.6       760   0.0403     0.239        1002             18.8
    80  409   36007    0.6      1328   0.0369     0.215           -               -
   100  541   59741    0.6      2064   0.0346     0.204           -               -
   130  733  105906    0.6      3282   0.0310     0.197           -               -
   160  941  168510    0.6      4896   0.0291     0.190           -               -

Rigorous, explicit, and already `0.19-0.67 p_k^2` -- i.e. the finite sieve certifies
`j(P(z)) < p^2` from the first row on, with the optimal level BELOW `z^2` (`C = 0.6` is the grid
floor; the finite sum is positive where the asymptotic `f(s)` is zero).  It is 6-19x above the true
`h(k)` and the ratio grows.  This is the honest explicit content of Iwaniec's route in one class:
an effective constant well under 1 at every computable size, and no proof that it stays bounded as
`s -> 2` without the 1971 error term.

---

## 3. Two classes: pre-registration (filed before the two-class run)

Verbatim from `scratchpad/prereg_proverD.txt`, written after the one-class control (`C >= 1.5` grid,
`z <= 258`) and before any two-class computation:

> P1. No finite `C_2`.  Carrying Iwaniec's 1978 argument to two classes per prime turns the sieve
> into a dimension-2 sieve, whose lower-bound main-term function `f_2(s)` is identically 0 for
> `s <= beta_2 = 4.2664` (DHR).  The window statement sits at `s = 2 < beta_2`.  Expected output
> of the transfer: `F <= C y^{beta_2 + eps}` (constant not explicit), i.e. the loss is an EXPONENT
> (`y^{2.27}`), not a constant.
>
> P2. Finite-size rigorous certificate (Rosser-type lower-bound weights, trivial remainder
> `|rho_d| <= 2^{omega(d)}`): the certified two-class interval length `X0_2(z)` will EXCEED
> `(z^2 - z)/6` at every `z >= 13` tested, and `X0_2(z)/z^2` will INCREASE with `z` (fitted local
> exponent between 3 and 4.5 for `z in [12, 60]`).
>
> P3. One-class control: `X0_1(z)/z^2 = 0.67 -> 0.29` at `z = 12..258`; expected to keep
> decreasing slowly at `z <= ~1000` with the optimal level `C` below 1.5.
>
> P4. The method uses ONLY the number of classes, so it bounds `h_2`.  Expected: `A072753 < (p^2-p)/6`
> at every published value, closest at `p = 13` (24 vs 26).

---

## 4. Two classes: the transfer

### 4.1 The objects in column units (no integer conversion needed)

Columns `k in Z`; gear `g` (prime, `5 <= g <= y`) strikes `k` iff `k = +-u_g (mod g)`.  This is a
sieve on the integers `k` removing `omega(g) = 2` classes per prime `g >= 5` (gears 2, 3 are the
columns themselves).  For `d | Q = prod gears`, `A_d = {k in A : every prime of d strikes k}`,
`|A_d| = 2^{omega(d)} X/d + rho_d`, `|rho_d| < 2^{omega(d)}`.  Working in columns, "`F(y) < y^2/6`"
is "every run of `(y^2 - y)/6` columns has an open one"; there is no factor 6 to convert because
the sieve is on columns.  (The literature's `h_2` is in integers: `h_2 = 6 A072753 + 6`.)

### 4.2 Lemma 1 transfers verbatim

With `f(d) = d/2^{omega(d)}`, `g(n) = n/2^{omega(n)}` (both multiplicative, `> 1` for primes `>= 3`),
`P` = the first `r` primes `>= 5`, `l(p_i) = q_i`: hypothesis (2) is `p_i/2 <= q_i/2`, true.  The
proof in 2.2 goes through unchanged (it never used the value of `f` beyond `f > 1`).  So

    S(A, Q) >= X prod_{q|Q}(1 - 2/q) [prod_{p|P}(1 - 2/p)]^{-1} sum_{n|P} lambda_n 2^{omega(n)}/n  -  sum_{n|P} |lambda_n| 2^{omega(n)}.

Again the first `r` primes `>= 5` are the worst case (`prod(1-2/q) >= prod(1-2/p)`), and again the
class POSITIONS never enter.  `[MINE]`, checked line by line against 2.2.

### 4.3 Lemma 2 does not transfer: the dimension

The sum `sum_{n|P} lambda_n 2^{omega(n)}/n` is the main term of a sieve with density
`omega(p)/p = 2/p`, i.e. `prod_{w <= p < z}(1 - 2/p)^{-1} <= K (log z/log w)^2`: **dimension `kappa = 2`**.
The analogue of (6) is the DHR lower bound `sum lambda_n 2^{omega(n)}/n = V_2(z) { f_2(s) + o(1) }` with
`f_2(s) = 0` for `s <= beta_2 = 4.2664...` (Diamond-Halberstam-Richert; the value `[SECONDARY]` via
Kao 2016, who cites it as known).  The linear sieve's `2 e^gamma log(s-1)/s`, positive for every
`s > 2`, is replaced by a function that is ZERO on the whole range `s <= 4.27`.

The window needs `s = 2`: an opening in `(y, y^2]` is an interval of `~y^2/6` columns sieved by
primes up to `z = y`, level `D <= X ~ y^2`, `s = log D/log z <= 2`.  So the transferred main term
is zero there, for every `C`, and no choice of `X = C y^2` makes `(7)` positive.  What the transfer
DOES give: with `y = z^{beta_2 + eps}` and `X ~ y`, `S > 0`, i.e.

    F(y)  <=  h_2-type bound  <=  C(eps) y^{beta_2 + eps}  =  C(eps) y^{4.2664 + eps}      (`[MINE]`; constant not explicit)

which in the literature's normalisation is `h_2(n) << p_n^{4.27+eps}`.  (I found no published upper
bound on `h_2` of any kind -- literature lane, section 3d -- so even this weak line is, as far as
I can find, unwritten.  It is a corollary, not a result.)  The exponent `beta_2` is where the
two-class argument dies; `C_2 y^2` is never reached, so **`C_2` is not a number -- the answer to
"is `C_2 < 1/6`" is that the method's `C_2` is infinite.**

### 4.4 The finite-size two-class certificate `[COMPUTED]`, scoring P2

Exactly as in 2.5 but with `2^{omega}` weights, primes `5 <= p < z`, level `y = z^s`
(`s in {1.5, 2, ..., 6}`, capped at `y <= 3e7`), Rosser truncation parameter `beta in {1..6}`
(`p_1 ... p_{2l-1} p_{2l}^{beta+1} < y`); every such weight system satisfies `(-)`, so each row is a
theorem: every run of `X0_2(z)` consecutive columns contains a column open under ANY two-class
assignment to the primes in `[5, z)`.

    z    X0_2    best s  beta   R_2    M_2       X0_2/z^2   (z^2-z)/6   X0_2 / ((z^2-z)/6)
     8      16     1.5    1       5   0.3143      0.250        9.3        1.7
    12      45     3.5    3      11   0.2468      0.312       22.0        2.0
    14      93     3.0    2      21   0.2272      0.474       30.3        3.1
    18     173     3.5    3      27   0.1566      0.534       51.0        3.4
    20     354     4.0    4      33   0.0934      0.885       63.3        5.6
    24     635     4.0    4      87   0.1370      1.102       92.0        6.9
    30     957     3.5    3     117   0.1224      1.063      145.0        6.6
    32    1426     3.5    3     135   0.0947      1.393      165.3        8.6
    38    2139     4.0    4     153   0.0715      1.481      234.3        9.1
    42    3201     4.0    4     231   0.0722      1.815      287.0       11.2
    44    4719     4.0    4     285   0.0604      2.438      315.3       15.0
    48    6891     4.0    4     375   0.0544      2.991      376.0       18.3
    54    9900     4.0    4     641   0.0648      3.395      477.0       20.8
    60   13628     4.0    4     943   0.0692      3.786      590.0       23.1
    62   17538     4.0    4    1073   0.0612      4.562      630.3       27.8
    68   22965     4.0    4    1367   0.0595      4.966      759.3       30.2
    72   29519     4.0    4    1629   0.0552      5.694      852.0       34.6

Least-squares exponent of `X0_2` in `z` over `z in [24, 72]`: **3.68** (end-to-end 24 -> 72: 3.49).
The optimal level sits at `s = 3.5-4` (the `s = 4.5` rows are cut by the `3e7` cap from `z = 48`
on, so the true optimum may be higher; that only raises the exponent toward `beta_2`).  Against
the one-class control, where the same certificate gives `X0_1/z^2` FALLING from 0.67 to 0.19, the
two-class certificate's `X0_2/z^2` RISES from 0.25 to 5.7 and its ratio to the window budget
`(z^2 - z)/6` goes from 1.7 to 35.  The real machine has `F(y)/((y^2-y)/6) = 0.28-0.44` at
`y = 11..59`.

**P2: CONFIRMED** (exceeds the budget at every `z`, including `z = 8, 12` below the pre-registered
threshold 13; ratio increasing; fitted exponent 3.68 inside the pre-registered band 3-4.5).
**P3: CONFIRMED** (decreasing to 0.190 at `p = 941`; optimal `C` at the grid floor 0.6 < 1.5).
**P1: CONFIRMED** by the DHR statement plus P2's growth; the exponent has not yet reached 4.27 at
`z = 72` and I do not claim the finite data pins the limit -- it pins the sign of the trend.
**P4: CONFIRMED**, section 6.

---

## 5. Where the loss is, and whether anything known can repair it

Accounting for the two-class `(7)`:

| step | one class | two classes |
|---|---|---|
| shifted sieve | factor `>= 1`, no loss | factor `>= 1`, no loss |
| `|rho_d|` | `< 1` | `< 2^{omega(d)}`, the standard dimension-2 remainder; not the obstruction (the finite certificate carries it exactly) |
| sieve -> covering | none | none |
| **sieve lower function at the window's `s = 2`** | `2 e^gamma log(s-1)/s -> 0^+`, positive for `s > 2`; explicit constant needs Iwaniec 1971's `c_1` | **`f_2(2) = 0` identically; positive only for `s > 4.2664`** |

**The lossiest step is the sieve inequality, and its loss is not its constant.**  The DHR function
is zero at `s = 2` whatever the remainder does.  Three named improvements:

- *Montgomery-Vaughan large-sieve constants.*  The large sieve bounds the number of survivors from
  ABOVE (`<= (N + Q^2)/L`); a covering statement needs survivors from BELOW.  The constants are
  irrelevant to the direction.  Applying it to the covering sets themselves reduces to capacity
  counting, which alignment-rules 6.3 records as near-achievable and slack-free.
- *Selberg's sieve with explicit weights.*  Selberg's `Lambda^2` is an upper-bound sieve; its
  lower-bound form `Lambda^2 Lambda^-` has, in dimension 2, a sifting limit that is NOT better than
  DHR's (Franze 2010: superior only for `kappa >= 3`).  So `~4.27` stands.
- *Second-moment / variance arguments.*  These bound how many `d` have large `rho_d`; the remainder
  is already exact here (`|rho_d| < 2^{omega(d)}`, carried without loss in the finite certificate).
  Nothing to gain.

And the structural reason none of them can work, stated as a theorem-level implication `[MINE]`:
every one of these tools uses only `|A_d| = 2^{omega(d)} X/d + rho_d`, hence bounds the maximum over
all two-class assignments, `h_2`.  If any of them gave `h_2(pi(y)) <= 6 C_2 y^2` with `C_2 < 1/6`
for all large `y`, then (the real teeth being one assignment) `F(y) < (y^2 - y)/6`, and the kernel
route (`twins_infinite_iff_survivor_in_window`; in print, Ziller-Morack 2017 Theorem 4.1) gives
infinitely many twin primes.  So a class-count-only constant below `1/6` is not a sieve-theoretic
improvement away; it is the twin prime conjecture.  In dimension 1 the analogous statement is that
the sieving limit 2 is sharp (Selberg's parity examples); in dimension 2 I have no proof that
`beta_2` is optimal for all conceivable sieves and do not claim one -- the implication above is
what is proved.

Whether the actual teeth could be used: Lemma 1 discards them at step (i).  Any route through the
teeth has to certify an opening in `(y, y^2]` at every `y`, which is branch 3 proper, not 3a.

---

## 6. What the method bounds: the two-class maximum `h_2`, against the window `[READ]`+`[COMPUTED]`

`A288815(n) = h_2(n)` (Ziller-Morack / Resta), `A072753 = (h_2 - 6)/6` in column units, against
`(p^2 - p)/6`:

    p     h_2   A072753   (p^2-p)/6   ratio       p     h_2   A072753   (p^2-p)/6   ratio
     5     18       2        3.3     0.600       41    894     148      273.3     0.541
     7     30       4        7.0     0.571       43   1044     173      301.0     0.575
    11     66      10       18.3     0.545       47   1284     213      360.3     0.591
    13    150      24       26.0     0.923       53   1422     236      459.3     0.514
    17    192      31       45.3     0.684       59   1656     275      570.3     0.482
    19    258      42       57.0     0.737       61   1902     316      610.0     0.518
    23    366      60       84.3     0.711       67   2190     364      737.0     0.494
    29    450      74      135.3     0.547       71   2460     409      828.3     0.494
    31    570      94      155.0     0.606       73   2622     436      876.0     0.498
    37    708     117      222.0     0.527

The maximum over assignments sits at 0.48-0.60 of the window budget from `p = 29` on, after the
`p = 13` near-miss (0.923; ZM's Conjecture 6 `h_2 < p^2 - p` holds at all 21 values, tightest there).
The real machine: `F(y)/((y^2-y)/6) = 0.382, 0.423, 0.397, 0.439, 0.403, 0.318, 0.374, 0.396, 0.333,
0.342, 0.327, 0.316, 0.282` at `y = 11..59` (`F = 7, 11, 18, 25, 34, 43, 58, 88, 91, 103, 118, 145, 161`,
max-gap units).  The counterfactual family's maxima are 1.6-1.9x the real `F` (alignment-rules
5.4), i.e. up to ~0.75 of the budget -- still under it, consistent with `A072753` being the
family's ceiling.  The sieve certificate of 4.4 is 1.7-35x ABOVE the budget on the same range.

---

## 7. Scores and verdict

- P1 CONFIRMED, P2 CONFIRMED, P3 CONFIRMED, P4 CONFIRMED.  (Nothing refuted; the pre-registration
  was conservative -- P2's threshold `z >= 13` was met from `z = 8`.)
- **`C_2`: does not exist by Iwaniec's route.**  The two-class transfer produces
  `F <= C y^{beta_2 + eps}`, `beta_2 = 4.2664`, constant not explicit; at finite sizes the rigorous
  version certifies `X0_2 = 1.7x .. 35x` the window budget on `z = 8..72`, growing like `z^{3.7}`.
- **Does it beat `1/6`?  No, and not by a constant: by an exponent of `2.27`.**
- **Lossiest step:** the sieve lower bound -- specifically the sieving limit of dimension 2, not
  the constant, not the shifted-sieve step (lossless), not the covering passage (lossless).
- **Teeth:** not used; the bound is on `h_2`.  Any class-count-only bound below `y^2/6` would prove
  the twin prime conjecture (kernel route / ZM Thm 4.1), so no known constant improvement is a
  candidate.
- **One class, for the record:** Iwaniec's constant is `e^gamma exp((c_1/2 + c_2/4 + 1) e^{-gamma})`
  with `c_1` from Iwaniec 1971 Cor. (1.4) (never numerical in print) and `c_2 ~ 1.1-1.2` measured
  for (5); the fully explicit finite certificate gives `j(P(z)) <= 0.19..0.67 p^2` on `p <= 941`.
- Branch 3a: **DEAD** (as a sieve route, with the reason recorded).  Branch 3 stays OPEN only as
  "use the teeth", which is the twin prime problem.

## 8. Files and reproduction

- This document.  Append to `docs/proof-search/agents-shared.md` under "Prover D".
- Pre-registration: `scratchpad/prereg_proverD.txt` (session scratchpad; text reproduced in section 3).
- Computations: inline `uv run python` scripts (Rosser weights enumerated by DFS; the `(-)`
  condition checked exhaustively at `z = 50`; one-class table `C in {0.6..3}`, `z = p_k + 1`;
  two-class table `s in {1.5..6}`, `beta in {1..6}`, `y <= 3e7`).  All single-core, seconds to a few
  minutes each.  Outputs: `scratchpad/twoclass_run.txt`.
- Sources saved locally: Iwaniec 1971 scan (`scratchpad/iwaniec1971.pdf`, page images
  `iw71_p1_*.png`), Hagedorn 2009 (`scratchpad/hagedorn.pdf`), Costello-Watts 2013, FKMPT, Kao 2016
  (tool-results PDFs).

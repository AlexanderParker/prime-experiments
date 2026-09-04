# Literature: is the increment statement in print?

LITERATURE lane, round 33.  Brief: find whether the statement the project needs -- the budget
inequality `F(M + q') <= F(M) + q'` (alignment-rules 0, 3.10, 4.2, 7) -- is known, conjectured or
provably hard in the published Jacobsthal literature, in either class count.  This file goes past
the adjacency table in Harvester round 29 (`agents-shared.md`); nothing from that table is
repeated except where a new statement inside one of those papers is being reported.

**Verification key.**  `[READ]` = I extracted and read the text of the paper itself (ar5iv HTML or
the PDF's own text layer, locally).  `[ABSTRACT]` = abstract / publisher metadata only.
`[SECONDARY]` = the statement is reported by another paper I read, and I could not obtain the
original.  `[OEIS]` = read from the OEIS raw record (`?fmt=text`), which is a primary-ish record
carrying its own attributions.  Anything unverified is marked **unverified**.

**Frames, stated once.**  The project's `F` is in *column* units (one column = the pair
`6k-1, 6k+1`, i.e. six integers) and is the *realised* record of the twin sieve, whose teeth are
pinned at `+-6^{-1} mod q`.  The published two-class object `h_2` is in *integer* units and is a
*maximum over all admissible class assignments*.  So `F_bc(y) <= A072753(pi(y))` pointwise and
`6 F ~ h_2` in scale.  The two are not the same function and the difference is load-bearing below
(item 2b).

---

## (1) ONE-CLASS INCREMENT: `j(P_{k+1}) <= j(P_k) + p_{k+1}`

### 1a. The exact statement, in print?  **NONE FOUND.**

No paper, preprint, OEIS comment, Erdos-problem entry or note that I could find states, tests,
proves or disproves any *additive* bound on `j(P_{k+1}) - j(P_k)`, on `j(mn) - j(m)`, or on
`j(np)` against `j(n) + p`.  Searched: the whole arXiv abstract index for "Jacobsthal" (80 most
recent hits, `export.arxiv.org` API -- after 2020 the only number-theoretic Jacobsthal-function
papers at all are Ziller 2020 and Kalmynin-Konyagin 2023); OEIS A048669, A048670, A058989,
A072752, A072753, A288815 in full; erdosproblems.com #687, #688, #689, #970; and the full text of
Hagedorn 2009, Hajdu-Saradha 2012, Ziller-Morack 2016/2017, Ziller 2019/2020, Mercer 2018.
**Verdict: NONE FOUND.**  The increment question, in the additive form the project needs, is not
in the literature at all -- not as a theorem, not as a conjecture, not as a computation.

### 1b. The nearest published increment statement: a MULTIPLICATIVE one, and it is a conjecture

- **Hajdu & Saradha 2012, Lemma 2.3 (theorem).**  `H(r) = max(H*(r), 2 H*(r-1))` for `r >= 2`,
  where `H(r) = max_{omega(n)=r} j(n)` and `H*(r) = max_{omega(n)=r, 2 not | n} j(n)`.  Followed by
  the remark, verbatim: *"It is important to note that for all the r values occurring in the
  present paper we have `H'(r) >= H*(r)`, that is `H(r) = 2 H*(r-1)`.  It is very much likely that
  this equality is valid for all `r > 1`."*
  Source: Math. Comp. 81 (2012) 2461-2471, DOI 10.1090/S0025-5718-2012-02581-6; text read from
  the author copy at `math.unideb.hu/sites/default/files/inline-files/jacobsrevsaradha.pdf`.
  `[READ]`  **Verdict: KNOWN theorem (the identity); the extension to all `r` is CONJECTURED by
  Hajdu-Saradha.**
- **Ziller 2019, Conjecture 3.2**, verbatim: *"`H(k) < 2 * H(k-1)` for all `k >= 3`.  The specified
  assertion is equivalent to `Omega(k) <= 2 * Omega(k-1) + 1` for all `k >= 3`.  Its validity would
  simplify the calculation of `H(k)` ... The assumption was already made in [3]."* ([3] =
  Hajdu-Saradha.)  Source: arXiv:1903.11973, "New computational results on a conjecture of
  Jacobsthal".  `[READ]`  **Verdict: CONJECTURED, by Ziller (2019), crediting Hajdu-Saradha
  (2012).  Verified for `k <= 43`.**

  This is the closest thing in print to an increment law, and it is *far weaker* than what the
  project needs: `H(k) < 2 H(k-1)` permits an increment of size `H(k-1)`, i.e. `~k^2`, where the
  project asks for `p_k ~ k log k`.  Nobody has conjectured the strong (additive) form.

### 1c. The related exact relations for adding one specific prime

- **Hajdu & Saradha 2012, Lemma 2.2 (theorem):** `j(2m) = 2 j(m)` for odd `m`.  `[READ]`  This is
  the only exact "add one prime" identity in the literature, and it is special to `p = 2`.
- **Hagedorn 2009, Proposition 2.8 (theorem):** `h(n+1) = 2 w(n) + 2`, where `w(n)` is the maximal
  length of an `S_n`-killing sieve and `S_n` = the first `n` **odd** primes.  Equivalently (Hajdu-
  Saradha's restatement) `h(r) = 2 h*(r-1)` for `r >= 2` with `h*(r) = j(p_2 p_3 ... p_{r+1})`.
  Source: Math. Comp. 78 (2009) 1073-1087, DOI 10.1090/S0025-5718-08-02166-2; full text read from
  `hagedorn.pages.tcnj.edu/files/2022/08/Jacobsthal.pdf`.  `[READ]`  In OEIS terms this is
  `A048670(n) = 2 A072752(n) + 2` and `A058989(n) = 2 A072752(n) + 1`.  `[OEIS]`
  **Verdict: KNOWN theorem.**  Note what it is: the increment from `h(n)` to `h(n+1)` is entirely
  absorbed into a *change of object* (halving by the prime 2), not bounded.
- **Ziller 2020, Proposition 1.6 (theorem):** `h(k) >= 2 p_{k-1}` for `k > 1`, by CRT.
  arXiv:2007.01808.  `[READ]`  Hagedorn states the same as an unnumbered Proposition 1.1.
  **Verdict: KNOWN theorem** -- a *lower* bound of exactly the shape the project's increment
  question lives in, and the only elementary one.

### 1d. What the tables say (arithmetic on published tables only)

Read from the b-file of OEIS A048670 (Bozek 2021, 64 terms; terms 1-49 are Hagedorn 2009, 50-54
Ziller-Morack 2016, 55-57 Gerbicz 2017, 58-64 Bozek 2021).  `[OEIS]`

**`h(k+1) - h(k) <= p_{k+1}` holds at all 63 computable steps, `k = 1..63`, with no exception.**
The ratio `(h(k+1)-h(k)) / p_{k+1}` is `0.667` at the very first step (`k=1`), `0.615` at `k=5`,
and then falls away: `0.13, 0.039` at the last two steps (`k = 62, 63`, primes 307 and 311).
The largest increment anywhere in the table is `40` (`h(63)-h(62) = 1098-1058`, against
`p_63 = 307`).  So the one-class increment statement is **TESTED to `k = 64` with a factor 8-25 of
room at the top**, and it has never been written down by anyone.  It is not a hard statement at
the computed range; it is an unasked question.

### 1e. The ONE-HOLE function (manager's addendum)

The manager's identity -- *the longest stretch containing at most one integer coprime to `P_k`
equals `j(P_{k+1})`, and the two-hole stretch equals `j(P_{k+2})`, by parity, valid while
`j(P_k) < 2 p_{k+1}` (through `k = 18`)* -- **has its mechanism in print twice, and its statement
in print zero times.**

- **Hagedorn 2009, Definition 2.2 and Proposition 2.5.**  Definition 2.2 defines an *`(S,k)`-killing
  sieve of length `z`*: a set of classes for `r - k` of the `r` primes of `S` such that all of
  `[1,z]` except a set `I` of `k` distinct integers is covered.  **That is exactly the `k`-hole
  object.**  Proposition 2.5, verbatim: *"Let `S` be a set of `r` distinct primes.  There is an
  `S`-killing sieve of length `z` if and only if there is an `(S,k)`-killing sieve of length `z`
  for some `k` in `[0,r]`."*  The proof of `(<=)` is one line: the `k` unused primes are assigned
  one hole each.  So *`k` holes bought with `k` fewer primes = no holes with all primes*, exactly
  the trade the manager's identity makes.  Hagedorn credits the `k = 1` case to J. Haugland,
  private correspondence, July 2005 (his reference [4] -- **not a publication**).  `[READ]`
- **Hajdu & Saradha 2012, section 2.3(b.1)**, verbatim: *"If we find that a subset `S'` of `S` with
  `|S'| = r'` covers a subset `A'` of `A` with `|A \ A'| <= r - r'`, then the `S'`-covering of `A'`
  can be extended to an `S`-covering of `A`."*  Attributed there to "the following ideas of
  Hagedorn".  `[READ]`  Same statement, general `k`.
- **The parity half is also in print**, on both sides: Hagedorn's Proposition 2.8 proof is exactly
  the parity argument (a maximal run has both endpoints coprime, so `h` is even, so the run halves
  to an odd-prime covering); and Ziller 2020 closes with the complementary crossover, verbatim:
  *"For all known values of Jacobsthal's function [10], we have `2 p_{k-1} < h(k-1)` for `k > 18`,
  i.e. `h(k) > 2 p_k` for `k > 17`."*  `[READ]`  That is the same threshold as the manager's
  validity range `j(P_k) < 2 p_{k+1}` (holds through `k = 18`: `h(18) = 132 < 134 = 2 p_19`;
  fails at `k = 19`: `152 > 142`).
- **Not found anywhere:** the function "longest stretch with at most `h` integers coprime to `P_k`"
  as a *named function with computed values*; no OEIS sequence (OEIS full-text search for
  "killing sieve" returns nothing, and A072752/A072753/A048670/A058989/A288815 carry no
  hole-variant); no "prime-free intervals with holes" in Ziller-Morack 2016 (I checked the paper --
  no holes generalisation there; their pruning is a capacity bound, Lemma 2.2 / Corollary 2.3, not
  a hole count).
  **Verdict: the MECHANISM is a KNOWN theorem (Hagedorn 2009 Prop. 2.5, idea credited to Haugland
  2005, restated by Hajdu-Saradha 2012).  The IDENTITY as the manager states it -- one-hole
  Jacobsthal of `P_k` = `j(P_{k+1})`, with the parity reason and the validity range -- is NOT in
  print in that form.  It is a corollary of Prop. 2.5 plus the parity argument, and worth
  attributing that way rather than claiming as new.**

---

## (2) TWO-CLASS: is `F(M + q') <= F(M) + q'` stated for `h_2` or for `k`-class Jacobsthal?

### 2a.  **NONE FOUND -- and there is essentially no two-class literature after 2017 to have said it.**

The entire published two-class corpus is two 2017 preprints by the same two authors:

- **Ziller & Morack 2017a**, arXiv:1706.00317, *"Divisibility in paired progressions, Goldbach's
  conjecture, and the infinitude of prime pairs"*.  `[READ]`  Definition 2.1 (paired Jacobsthal
  function `j_2(n)`), Definition 2.2 (`h_2(n) = j_2(p_n#)`), Conjectures 4, 5, 6, Theorem 4.1.
- **Ziller & Morack 2017b**, arXiv:1706.03668, *"A short note on the computation of the generalised
  Jacobsthal function for paired progressions"*.  Computed values `h_2(n)` for `n <= 21`
  (`p_n <= 73`).  `[ABSTRACT + WebFetch summary]`; the value table is independently confirmed by
  OEIS A288815 `[OEIS]`.

Neither states any increment, growth-rate, difference or "add one prime" statement of any kind.  I
asked for exactly that in a targeted read of 1706.03668 and the answer was that the paper contains
**no** remarks on `h_2(n+1) - h_2(n)`, on `h_2(n)/p_n^2`, or on behaviour under adding a prime.
No paper on `h_2` or on any `k >= 2` class-count Jacobsthal function has appeared since (arXiv
abstract index swept for "Jacobsthal", 2017-2026: nothing).  Ziller 2020's Proposition 2.7
(propagation of coverings) is one-class.  **Verdict: NONE FOUND, and the field is empty, not
merely silent.**

### 2b.  BUT: the increment statement is FALSE for the published two-class MAXIMUM, at one step

Arithmetic on the published table (OEIS A072753 = the column-unit two-class record,
`A288815(n) = 6 A072753(n) + 6 = h_2(n)`; values by Ziller, Morack and Resta, 2002-2017):

    primes           {5}  ..7  ..11  ..13  ..17  ..19  ..23  ..29  ..31  ..37 ...
    A072753            2    4    10    24    31    42    60    74    94   117 ...
    increment           -    2     6    14     7    11    18    14    20    23 ...
    added prime q'      -    7    11    13    17    19    23    29    31    37 ...

`24 - 10 = 14 > 13`.  **The step `{5,7,11} -> {5,7,11,13}` violates `F(M+q') <= F(M) + q'` by
exactly 1.**  Every one of the other 17 computable steps (through `p = 73`) satisfies it.  In
integer units the same violation reads `h_2(6) - h_2(5) = 150 - 66 = 84 > 78 = 6 p_6`.

This does **not** contradict anything the project has measured: `h_2` is the *maximum over class
assignments* and the project's `F` is the *realised* twin-sieve record, with `F_bc <= A072753`
pointwise (`6, 10, 17, 24, 33, 42, 57, 87` against `10, 24, 31, 42, 60, 74, 94, 117` at
`y = 11..37`).  What it does say is sharp and worth carrying: **the budget inequality is FALSE for
the tooth-free (maximum) version of the object at a small step, so no proof of it can go through
"any two classes per prime" -- it must use the actual teeth.**  That is the same conclusion
alignment-rules 3.10 and section 5 already reach from the counterfactual family (13-22% violate),
now with an independent published witness.  `[OEIS]`, arithmetic on published values only.

### 2c.  The adjacent Erdos problem, for the record

**Erdos problem #689** (erdosproblems.com/689, `[READ]`): *"Let `n` be sufficiently large.  Is
there some choice of congruence class `a_p` for all primes `2 <= p <= n` such that every integer in
`[1,n]` satisfies at least two of the congruences `= a_p (mod p)`?"*  OPEN; Erdos 1979/1980; also
Problem 45 on Green's open problems list.  **This is one class per prime with multiplicity two, not
two classes per prime** -- a different object, and it is the *closest* thing in the Erdos corpus to
a two-class Jacobsthal question.  Recorded so it is not mistaken for ours.  Erdos also asks (in
[Er80, p.106], carried on #687) about the weakened covering in which *all except `o(y/log y)`* of
the integers in `[1,y]` are covered -- which is the `h`-hole family of item (1e) in asymptotic
dress, and is likewise open with no bound.

---

## (3) THE ROUTE: `F(y) < y^2/6 => infinitely many twin primes`

### 3a.  The two-class route IS in print, as a theorem with a named conjecture -- and it is the
### project's own statement

- **Ziller & Morack 2017a, Conjecture 6**, verbatim: *"Upper bound of the primorial paired
  Jacobsthal function.  Let `n in N >= 3`.  Then `h_2(n) < p_n^2 - p_n`."*
- **Ziller & Morack 2017a, Theorem 4.1**, verbatim: *"The conjectured upper bound of the primorial
  paired Jacobsthal function is sufficient for the truth of the Goldbach conjecture and of the
  infinitude of prime pairs for every even difference."*  (Proof: from Propositions 3.2, 3.5 and
  Corollaries 3.1, 3.4.)
- Their Conjecture 5 is the window statement in the project's own shape: *"for every `n` and every
  prime `p > 2n` there exists a pair of primes `q_1, q_2` with `p < q_1 < p^2` and
  `q_2 - q_1 = 2n`"* -- the same `(y, y^2]` window as the kernel route.
- Source arXiv:1706.00317.  `[READ]`  Also stated in OEIS A288815: *"If `a(n) < p_n^2 - p_n` holds
  for `n >= 3` then Goldbach's conjecture and the twin prime conjecture hold as well."*  `[OEIS]`

  **In the project's units** `h_2 ~ 6F`, so `h_2(n) < p_n^2 - p_n` is `F < (p_n^2 - p_n)/6`, i.e.
  **exactly `F(y) < y^2/6` up to the linear term.**  So the project's route is the published
  ZM 2017 Theorem 4.1, and the project's target inequality is *weaker* than ZM Conjecture 6 (which
  is stated for the maximum over class assignments, `F_bc <= A072753`).
  **Verdict: the implication is a KNOWN theorem (ZM 2017 Thm 4.1); the bound it needs is
  CONJECTURED by Ziller & Morack (2017), verified at 21 values to `p = 73`.**
  Action for the project: this must be cited wherever the route is stated.  It is the single most
  adjacent published item to the whole programme and it is one line stronger than what we need.

### 3b.  The one-class route, published form

- **Mercer 2018, Theorem 1**, verbatim: *"Let `d` be a positive integer.  If there exists a
  positive integer `k` such that `(p_{k+1}^2 - 2)/(h(k)+1) >= d`, then every eligible arithmetic
  progression `a + dZ` contains at least one prime."*  Its Lemma 2 is the project's kernel route
  verbatim: *"Suppose `n` is an integer such that `1 <= n < p_{k+1}^2` and `n` is coprime to
  `p_k#`.  Then `n` is prime."*
  **Corollary 1:** every eligible AP with `d <= 76` contains a prime (from `h(54) = 742`, so
  `(257^2 - 2)/743 > 76`).  **Corollary 2:** if `h(n) = o(p_{n+1}^2)` -- e.g.
  `h(n) << p_{n+1}^2/log p_{n+1}` -- then *every* eligible AP contains a prime, i.e. **an
  elementary proof of Dirichlet's theorem.**
  **Remark:** *"Known values of `h(k)` appear to give support to the conjecture that
  `(p_{k+1}^2 - 2)/(h(k)+1)` grows without bound"*, with the table `11.13, 20.40, 27.79, 30.44,
  39.38, 48.72, 52.65, 59.44, 61.59, 71.15` at `k = 5, 10, ..., 50`.
  Source: Idris Mercer, *Dirichlet's theorem and Jacobsthal's function*, INTEGERS 18 (2018) #A26,
  `math.colgate.edu/~integers/s26/s26.pdf`; preprint arXiv:1708.05415.  `[READ]`
  **Verdict: KNOWN theorem (Mercer 2018).**
- Mercer credits the stronger consequence to **Kanold**: *"It was shown in [4] that a bound of the
  form `h(n) <= C p_n^{2-eps}` that holds for all `n` would lead to a short proof of Linnik's
  theorem and Dirichlet's theorem."*  [4] = H.-J. Kanold, *Uber Primzahlen in arithmetischen
  Folgen*.  `[SECONDARY]` -- **unverified**, I did not obtain Kanold's paper.
- **Volfson 2022**, arXiv:2211.13255, *"Maximum distance between consecutive primes and other
  related questions"*, defines `d(p_r^2 - 1)` = the maximum gap between consecutive members of the
  reduced residue system mod `p_r#` **restricted to the interval `[p_{r+1}, p_r^2 - 1]`** -- i.e.
  the *windowed* Jacobsthal function, which is the one-class analogue of the project's
  "record below the window".  His Conjecture 2 is `d(p_r^2-1) <= 2 p_r + 1` and his Assertion 4:
  *"If `d(p_r^2-1) <= 2 p_r + 1` then Legendre's conjecture holds."*  `[READ]`
  **Verdict: CONJECTURED by Volfson (2022), single-author unrefereed preprint, no journal
  reference; treat as low-weight.**  It is nevertheless the only place I found that computes the
  *windowed* record rather than the full-period one, and he reports values to `4561#`.
  Worth a look by whoever owns the window measurements; **I did not verify his values.**

### 3c.  Explicit constants: what exists

| bound | constant explicit? | source | verification |
|---|---|---|---|
| `h(n) <= 2^n` | YES, fully explicit | Kanold 1967, Math. Ann. 170, 314-326 | `[SECONDARY]` (Hagedorn s.1, OEIS A048669) |
| `h(n) <= 2^sqrt(n)` for `n >= e^50` | YES | Kanold 1967 | `[SECONDARY]` (Hagedorn (1.1) region) |
| `h(n) <= 2 n^{2 + 2e log n}` (`n >= 15`) | YES | Stevens 1977, Math. Ann. 226, 95-97 | `[SECONDARY]` (Hagedorn (1.2), Hajdu-Saradha (1.1)) |
| `H(r) <= 2 r^{2 + 2e log r}` | YES | Stevens 1977 | `[READ]` in Hajdu-Saradha (1.1); they note it gives `H(10) <= 2 x 10^{14.6}` |
| `h(n) << n^2 log^2 n`, i.e. `j(n) << log^2 n` | **NO** | Iwaniec 1978, *On the problem of Jacobsthal*, Demonstratio Math. 11, 225-231/232, DOI 10.1515/dema-1978-0121 | `[ABSTRACT]` + three independent `[SECONDARY]` readings (Hagedorn: "for an unknown constant C"; Mercer: "for an unknown constant C"; Costello-Watts: same) |
| `h(n) <= 0.27749612254 n^2 log n`, **for `50 <= n <= 10,000` only** | YES | Costello & Watts, arXiv:1208.5342 -> Math. Comp. 84 (2015) 1389-1400 | `[READ]` via Mercer's independent quotation of the same constant, cross-checked against a direct read of the preprint |
| `H(k) <= k^2` for `2 <= k <= 12` | YES (range-restricted) | Kanold | `[SECONDARY]` via Ziller 2019 discussion of Conjecture 3.3 |
| `Y(x) << x^2` (Erdos #687 form) | **NO** | Iwaniec 1978 | `[READ]` erdosproblems.com/687 |

So: **the only explicit constant in the whole one-class upper-bound literature that is anywhere
near `p^2` is Costello-Watts's `0.2775 n^2 log n`, and it is proved only on `50 <= n <= 10^4`.**
Iwaniec's implied constant has never been made explicit by anyone, in fifty years.  Note the
frame: `0.2775 n^2 log n` with `p_n ~ n log n` is `~ 0.2775 p_n^2 / log p_n`, which is `o(p_n^2)`
-- so *if* Costello-Watts held for all `n` it would already discharge Mercer's Corollary 2 and
give an elementary Dirichlet.  It does not; the range restriction is the whole difficulty.

### 3d.  Is an explicit constant below `1/6` in the TWO-CLASS setting known to be out of reach?

**No -- and nobody says so, either way.  I found no source that asserts it is out of reach, and no
source that asserts it is reachable, because there is no published upper bound on `h_2` of any
kind -- not explicit, not ineffective, not asymptotic.**  Stated plainly, so it is not
over-claimed:

- **There is no published upper bound on `h_2(n)` at all.**  ZM 2017/2017b give a definition, 21
  computed values and Conjecture 6.  Nothing else exists.
- The reason no bound follows from the existing machinery is structural and *is* on record, though
  not stated about `h_2` by name: Iwaniec's route is the **linear (dimension-one) sieve**
  (his 1971 *On the error term in the linear sieve* is the engine of the 1978 bound), and
  two classes per prime is a **dimension-two** sieve, `prod (1 - 2/p) ~ C/(log x)^2`.  The project
  already carries the same observation as the decisive negative on FKMPT (Harvester r29 item 8:
  hypothesis (1.2) fails at dimension two).  **The inference "therefore Iwaniec does not transfer"
  is MINE, from the definition of the linear sieve; I found no author who states it.**  Flagging it
  as such: **unverified as an attributed claim.**
- The honest summary for the manager: *"an explicit constant below `1/6` in the two-class setting"*
  is not known to be out of reach; it is simply **unattempted**.  The two-class object has one
  definition paper, one computation paper, no bounds paper, and no follower in nine years.

---

## (4) SUBADDITIVITY / SUBMULTIPLICATIVITY IN THE MODULUS

- `j(mn) <= j(m) j(n)`: **NOT FOUND in any source.**  Searched OEIS A048669 (which lists the
  standard property set: `g(n) = g(rad n)`, Kanold's `2^w`, Iwaniec's `X (w log w)^2` -- and no
  multiplicative inequality), the OeisWiki Jacobsthal page (HTTP 403, **not obtained**), and the
  full texts above.  If the project has seen it stated somewhere, that source should be produced;
  I could not find it, and I would treat it as folklore until it is.  **Verdict: NONE FOUND.**
- What *is* in print, and is the honest weaker relative:
  - **Ziller & Morack 2016, Remark 1.1:** `j(n_1 n_2) >= j(n_1)` and `j(n_1 n_2) >= j(n_2)` for all
    `n_1, n_2`; and strictly, `j(n_1 n_2) > j(n_1)` and `> j(n_2)` when `n_1, n_2 > 1` are coprime.
    arXiv:1611.03310.  `[READ]` (via targeted extraction).  Monotone, **not** submultiplicative.
  - **`j(n) = j(rad n)`** -- everywhere, e.g. Hagedorn s.0.1, OEIS A048669.  `[READ]`
  - **`j(2m) = 2 j(m)` for odd `m`** -- Hajdu-Saradha Lemma 2.2 (item 1c).  This is the *only*
    exact multiplicative relation for adding a prime, and it is exact, not an inequality, and only
    for `p = 2`.  It is also the *sharp* instance of an additive-increment failure: adding the
    prime 2 to `m` increases `j` by `j(m)`, unboundedly more than `2`.  Any additive increment
    conjecture must therefore be stated for the *primorial ladder from below* (`p_{k+1}` large
    relative to the structure it enters), never for an arbitrary added prime.  That caveat is not
    stated anywhere in the literature and the project should state it when it writes the conjecture
    down.
- **"Adding one prime to the modulus increases the maximal gap by at most a bounded multiple of
  that prime":** **NONE FOUND**, in any class count, in any form -- additive, multiplicative or
  asymptotic.  The only bounded-ratio statements in print are Hajdu-Saradha/Ziller's factor-2
  doubling conjecture on `H` (item 1b), which is a multiple of the *function*, not of the prime.

---

## Answer to the manager's question

**No.  In neither class count is the increment statement in print as a theorem, and in neither
class count is it named as a conjecture.**  For the ordinary Jacobsthal function the additive form
`j(P_{k+1}) <= j(P_k) + p_{k+1}` appears nowhere -- not in Jacobsthal's own problem, not in
Erdos's problem list (#687, #689, #970 are all about the *size* of `h`, never its increments), not
in Hagedorn 2009, Hajdu-Saradha 2012, Ziller-Morack 2016, Mercer 2018, Ziller 2019/2020, and not
in any OEIS comment on A048669/A048670/A058989/A072752/A072753/A288815; it is nevertheless true at
all 63 computable steps of the published table with a factor 8-25 of room at the top.  For the
two-class function the situation is starker: `h_2` has exactly two papers (Ziller & Morack, both
2017), 21 computed values, no growth remarks of any kind, and no successor in nine years -- so
there is nothing to have stated it.  The nearest published statements are of a different shape:
Hajdu-Saradha's identity `H(r) = max(H*(r), 2H*(r-1))` with their "very much likely" extension,
promoted by Ziller (2019, Conjecture 3.2) to `H(k) < 2 H(k-1)` -- a *doubling* bound, weaker than
the project's by a factor `~k^2/(k log k)`; and Hagedorn's exact `h(n+1) = 2 w(n) + 2`, which
converts the increment away rather than bounding it.  Two things the project should nevertheless
take from this sweep and act on.  First, **the route is already published**: Ziller & Morack 2017
Theorem 4.1 proves that `h_2(n) < p_n^2 - p_n` (their Conjecture 6) implies Goldbach and the
infinitude of prime pairs -- which in column units is `F < y^2/6`, the project's own statement,
and their conjecture is the *stronger* maximum-over-assignments form; the one-class analogue is
Mercer 2018 Theorem 1 / Corollary 2 (`h(n) = o(p_{n+1}^2)` gives an elementary Dirichlet), with
Lemma 2 being the kernel route verbatim.  Second, **the increment statement is false for the
published two-class maximum**: A072753 jumps `10 -> 24` when 13 is added, `14 > 13`, the unique
violation among the 18 computable steps -- independent published confirmation that any proof of
the budget inequality must use the real teeth and cannot be a statement about "two classes per
prime".  On explicit constants: the only one in print anywhere near `p^2` is Costello-Watts's
`h(n) <= 0.27749612254 n^2 log n` restricted to `50 <= n <= 10^4`; Iwaniec's 1978 constant has
never been made explicit by anyone; and in the two-class setting there is no upper bound of any
kind, so a constant below `1/6` there is **not** known to be out of reach -- it is unattempted, and
no author says otherwise.

---

### Sources (all URLs checked this round)

- Hagedorn, *Computation of Jacobsthal's function h(n) for n < 50*, Math. Comp. 78 (2009) 1073-1087.
  https://doi.org/10.1090/S0025-5718-08-02166-2 ; text: https://hagedorn.pages.tcnj.edu/files/2022/08/Jacobsthal.pdf
- Hajdu & Saradha, *Disproof of a conjecture of Jacobsthal*, Math. Comp. 81 (2012) 2461-2471.
  https://doi.org/10.1090/S0025-5718-2012-02581-6 ; text: https://math.unideb.hu/sites/default/files/inline-files/jacobsrevsaradha.pdf
- Ziller & Morack, *Algorithmic concepts for the computation of Jacobsthal's function*, arXiv:1611.03310
- Ziller & Morack, *Divisibility in paired progressions, Goldbach's conjecture, and the infinitude of prime pairs*, arXiv:1706.00317
- Ziller & Morack, *A short note on the computation of the generalised Jacobsthal function for paired progressions*, arXiv:1706.03668
- Ziller, *New computational results on a conjecture of Jacobsthal*, arXiv:1903.11973
- Ziller, *On differences between consecutive numbers coprime to primorials*, arXiv:2007.01808
- Mercer, *Dirichlet's theorem and Jacobsthal's function*, INTEGERS 18 (2018) #A26, https://math.colgate.edu/~integers/s26/s26.pdf ; arXiv:1708.05415
- Costello & Watts, *A computational upper bound on Jacobsthal's function*, arXiv:1208.5342 ; Math. Comp. 84 (2015) 1389-1400
- Iwaniec, *On the problem of Jacobsthal*, Demonstratio Math. 11 (1978) 225-231, https://doi.org/10.1515/dema-1978-0121
- Volfson, *Maximum distance between consecutive primes and other related questions*, arXiv:2211.13255
- Erdos problems #687 (Y(x), $1000), #688, #689, #970, https://www.erdosproblems.com/
- OEIS A048669, A048670 (+ b-file, 64 terms), A058989, A072752, A072753, A288815

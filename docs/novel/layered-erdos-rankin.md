# layered-erdos-rankin - the Erdos-Rankin construction run k times, and the k-class Jacobsthal function

Round 25 (harvester): the construction and its bookkeeping. **Round 26: WRITTEN
OUT WITH CONSTANTS - it is a proof.** Gates: `research/j2_rankin_layer.py`
(round 25, the finite ingredients) and `research/j2_layer_proof.py` (round 26,
the constant-tracked assembly), both ALL ASSERTIONS GREEN.

Status line, round 26: **the asymptotic statement is proved and the leading
constant is explicit.** What is NOT delivered is a numerically written-down
threshold `x_0` (section 6) - the ingredients are all effective, but the `o(1)`
decays like `(log C + 1)/C` with `C = logloglog log x`, so an honest `x_0` is
not writeable. Round 25's blanket status line ("asymptotic bookkeeping, NOT a
written-out proof") is superseded by section 3 below.

## 0. Definitions

`j_k(m)` - the **k-class Jacobsthal function**: the longest run of consecutive
integers that can be covered by choosing at most `k` residue classes modulo each
prime `p | m` (one class only at p = 2). `j_1` is the ordinary Jacobsthal
function of `m`. For `m = p_n#` the paired Jacobsthal function of Ziller-Morack
satisfies `h_2(p_n#) = j_2(p_n#)`, by the round-23 restatement re-verified
exactly at z = 3, 5, 7 against the known values 6, 18, 30 (section A of the
round-25 gate). Write `x = p_n`, `A = log x`, `B = log A`, `C = log B`.

## 1. What it is

    THEOREM (P2', round 26).  Let c_1 be any constant for which the twin count
    satisfies pi_2(t) <= c_1 t/(log t)^2 for all t >= t_1.  Then

        j_2(P(x))  >=  ( 1/(18 c_1) + o(1) ) * x A^3 C^2 / B^4 .

    Admissible c_1, all normalised against the Hardy-Littlewood singular series
    Pi(t) = 2 C_2 t/(log t)^2 = 1.3203236 t/(log t)^2 :

        c_1                      source                          1/(18 c_1)
        3.29956 x 2C_2 = 4.35649 Lichtman 2024 (record, asympt.)  0.0127524
        3.39951 x 2C_2 = 4.48845 Wu 2004                          0.0123774
        8       x 2C_2 = 10.5626 Selberg 1947; explicit form in   0.0052597
                                 Riesel-Vaughan 1983 Lemma 5,
                                 effective for t >= e^42

    The statement carries an o(1), so the best ASYMPTOTIC constant is
    admissible: **the headline constant is 0.0127524**, and the fully
    effective-in-principle alternative is 0.0052597.
    Consequences:
      h_2(P(z))  >>  z (log z)^{3-o(1)} , so any bound h_2 = O(z (log z)^a)
      forces  a >= 3;  and the round-24 extreme-value model ~2.56 z (log z)^2
      is NOT a ceiling (it is exceeded by a full log).

    GENERAL k, same proof:
        j_k(P(x)) >= ( k/((k(2k-1))^k c_1^(k)) + o(1) ) * x A^(2k-1) C^k/B^(2k),
    c_1^(k) an admissible Brun upper-bound constant for the k-tuple count, and
    for k >= 4 with the O(1) primes dividing a shift difference set aside.

**A CONSTANT ERROR OF OUR OWN, SELF-CAUGHT.** The first draft of the gate set
`c_1 = 8 C_2 = 5.2813`, reading Selberg's classical 8 as multiplying `C_2`. It
multiplies the FULL singular series `2 C_2`. Verified first-hand 2026-08-29
against Lichtman arXiv:2109.02851 (Algebra & Number Theory 19 (2025) no. 1),
whose Theorem 1.2 is `pi_2(x) <~ 3.29956 Pi(x)` with
`Pi(x) = 2x/(log x)^2 prod_{p>2}(1-2/p)/(1-1/p)^2` and whose history table gives
Selberg 8, Bombieri-Davenport 4, BFI 3.5, Wu 2004 3.39951. **The first draft was
a factor of two too good** - and, by coincidence, its wrong value 5.2813 is
exactly Bombieri-Davenport's 1966 constant, which is why nothing looked odd.
Another instance of harvester 7d clause 2.

Round 25 stated the shape `x A^(2k-1) C^k/((5k)^k B^(2k))`. The written-out
proof **improves the denominator to `(k(2k-1))^k`** - a factor 2.778 at k = 2 -
and shows the round-25 form is **inadmissible for k >= 4** (section 5, item iv).

## 2. Why it might be novel - AND THE ONE PIECE OF PRIOR ART THAT NAMES OUR SYSTEM

**READ THIS FIRST.** Round 26's prior-art sweep found, and I then read
first-hand (ar5iv rendering of arXiv:1802.07604, 2026-08-29),
**Ford-Konyagin-Maynard-Pomerance-Tao, "Long gaps in sieved sets", REMARK 7**:

> "Unfortunately our methods only seem to give good results in the
> one-dimensional case. Consider for instance the set {n in P : n+2 in P} of
> (the lower) twin primes. This corresponds to a two-dimensional system in which
> I_p = {0 (mod p), 2 (mod p)} for all primes p. The 'trivial' bound coming from
> these methods would give a bound of >> log X log log X for the largest gap
> between lower twin primes up to X (or between the largest such twin prime and
> X), and one could possibly hope to improve this bound by a small power of
> log log X using a variant of the methods in this paper. However, a sieve upper
> bound (e.g., [7, Cor. 2.4.1]) combined with the pigeonhole principle already
> gives a bound of >> log^2 X in this case."

**That is our sieving system, named in print, by these five authors.** So the
sentence "nobody appears to have asked what happens with two classes per prime"
is NOT safe and is withdrawn. What survives, precisely, and it is stronger for
being precise (all three asserted arithmetically in the gate, section F0):

1. **Their "trivial" bound is the order of our (P1), not of this theorem.** In
   covering coordinates (`log X ~ x`, `log log X ~ A`), `>> log X log log X` is
   `>> z log z`. Round 24's (P1) proves `(1.349+o(1)) z log z`. **NOVELTY
   QUALIFICATION on (P1), self-found and recorded: the ORDER `z log z` for this
   exact system is in print (2018-2022) as trivially available.** (P1) remains
   the first PROVED bound with an explicit constant, and the first stated for
   Ziller-Morack's `h_2`; it is not the first appearance of the order.
2. **They hoped for "a small power of log log X". This theorem gives TWO FULL
   POWERS.** `x A^3 C^2/B^4` over `x A` is `A^{2-o(1)}` - asserted numerically.
   The route is different from theirs: a layered Erdos-Rankin covering, not
   their sieved-set machinery. **FKMPT flagging the two-dimensional case as out
   of reach for their methods, and hoping for `(log log X)^eps`, is the sharpest
   available statement of what this construction contributes.**
3. **Their `>> log^2 X` pigeonhole bound is no obstruction here - but it kills
   any twin-prime-gap corollary.** Two different quantities: gaps between ACTUAL
   twin primes near `X` have pigeonhole floor `(log X)^2 = x^2`, which beats
   `x A^3`; but the SIFTED SET of the covering problem has density `~1/A^2`
   inside its period, so the same pigeonhole gives only `>> A^2 = (log x)^2`,
   which this theorem beats by a full power of `x`. So the theorem is a genuine
   statement about `j_2 = h_2`, and **no twin-prime-gap corollary may be
   claimed** - one would be weaker than an argument FKMPT call trivial. This is
   now item 5 of the not-claims list.

## 2a. What is new, given section 2

The ordinary Erdos-Rankin construction spends its one class per prime on two
jobs at once, and the whole art of it is that class 0 on a SPLIT range
`[2,P) u (z1, x/L]` delivers survivor density ~1/log y where its Mertens
entitlement is only O(1) - a full log of "non-independence" gain - while leaving
the middle range `[P, z1]` free for a greedy sparsification. **No one appears to
have CARRIED OUT that construction with TWO classes per prime** (FKMPT name the
system and decline it - section 2). The answer is that the second layer is a
SHIFTED
Eratosthenes (class -2 instead of class 0), the joint survivor set is the set of
TWIN primes rather than the primes, and the gain is another full log - because
Brun's upper bound for twins is a log stronger than Chebyshev's for primes.

Prior art, checked first-hand 2026-08-29 (round 25) and re-checked 2026-08-29
(round 26, see section 7):

* **Ford-Konyagin-Maynard-Pomerance-Tao, "Long gaps in sieved sets"**
  (arXiv:1802.07604, JEMS 23 (2021) 667-700 + corrigendum JEMS 25 (2023)
  2483-2485) bound gaps in `{n : n mod p not in I_p for all p <= x}` for
  **GIVEN** `I_p`. Their Definition 1 requires the system to be non-degenerate,
  `B`-bounded, **ONE-DIMENSIONAL** (`prod_{p<=x}(1-|I_p|/p) ~ C_1/log x`) and
  `rho`-supported; Theorem 1 then gives a gap `>= x(log x)^{C(rho)-o(1)}`. Our
  system is TWO-dimensional, so their theorem does not apply to it - which is
  what their own Remark 7 says. Ours also CHOOSES the classes rather than
  receiving them. **It must be cited, and Remark 7 must be quoted and
  addressed** (section 2).
* Ford-Green-Konyagin-Tao (arXiv:1408.4505) and Maynard: the k = 1 case, and
  the calibration target. Their `c` may be taken arbitrarily large; we make no
  such claim.
* Ziller-Morack (arXiv:1706.00317) define `j_2` and conjecture only an upper
  bound; they prove no lower bound of any kind.
* The k-class Jacobsthal function `j_k` does not appear in the literature under
  any name we can find.

## 3. STATUS - what is proved and what is not

**PROVED (round 26).** The statement of section 1, in the `o(1)` form. Every
step is section 4's; every ingredient is standard, unconditional and cited; the
assembly is asserted numerically in `research/j2_layer_proof.py` sections A-E.

**NOT DELIVERED, and one thing that must never be claimed.**
* **NO TWIN-PRIME-GAP COROLLARY.** Section 2 item 3: a pigeonhole argument that
  FKMPT call trivial already beats anything this theorem says about gaps between
  actual twin primes. The theorem is about the covering quantity `j_2 = h_2`.
* A written-down `x_0`. Effective, astronomical - section 6 item 1.
* Optimality of `(loglog)^4` or of `1/(18 c_1)`. Both are what THIS parameter
  choice gives (section 6 item 3).
* Any kernel check. The statement is asymptotic; there is nothing finite in it.
* Any finite-z content. Round 25 measured that the parameterisation admits no
  choice at all below `log x ~ 300`. **(P1)'s `h_2 >= (1.349+o(1)) z log z`
  remains the bound to quote at any z a human will ever see.**

**THE CALIBRATION, and it is now at CONSTANT level, not shape level.** Run the
identical write-up at k = 1 and it returns `j(P(x)) >= (1+o(1)) x A C/B^2`.
Rankin's proved theorem in these coordinates is `(e^gamma + o(1)) x A C/B^2`,
`e^gamma = 1.781072`. **So the write-up lands a factor 1.781 BELOW the
classical constant for the classical case** - the right side of it, and by a
small factor. An accounting that came out ABOVE Rankin would have been a bug;
this one does not. (The shortfall is the crude greedy and the elementary
`rho <= 1/Gamma` bound, both known to be improvable.) Round 25 could only check
the SHAPE (residual spread 0.072 over eight decades); round 26 checks the
CONSTANT.

## 4. THE PROOF

Throughout, `x` is large, `A = log x`, `B = log A`, `C = log B`, and `P(x)` is
the primorial over primes `<= x`.

### 4.0 The restatement (round 23, brute-forced at z = 3,5,7)

Killing `n` requires `p | n` or `p | n+E` for some `p | m`, so the killed
classes mod `p` are `{-t, -t-E}` where `t` is the (single, global) translation
of the interval and `E` the (single, global) even shift. By CRT `t` and `E` may
be prescribed independently at every prime, so `{-t, -t-E} mod p` realises **any
ordered pair of classes mod each odd p**, and at `p = 2` (where `E` is even) a
single class. Hence

    j_2(P(x)) - 1 = the longest interval coverable by <= 2 arbitrary classes
                    mod each odd p <= x, and one class mod 2.

### 4.1 Parameters

    L  := B                       (the split point is x/L)
    P  := A^3                     (small-prime cut;  log P = 3B)
    u  := theta B / C,  theta = 2 + 4(log C + 1)/C   ->  2
    z1 := x^(1/u)                 (medium-prime cut; log z1 = A/u = AC/(theta B))
    y  := K x A^3 C^2 / B^4       (the interval length; K fixed below)
    sigma := log P/log z1 = 3 theta B^2/(A C)

For large `x`: `P < z1 < x/L`, `u > 1`, `y > x`, and `L y/x = K A^3 C^2/B^3 < P`
- all four asserted in the gate.

### 4.2 The four layers

    LAYER 1   class  0  mod p   for p = 2 and for p in [3,P) u (z1, x/L]
    LAYER 2   class -2  mod p   for p in [3,P) u (z1, x/L]
    LAYER 3   two classes, chosen greedily, at each p in [P, z1]
    LAYER 4   two classes, used for matching, at each p in (x/L, x]

Every prime `<= x` is used in exactly one layer, and no prime is given more than
two classes. Layers 1 and 2 collide only where `p | 2`, i.e. only at `p = 2`,
where the paired problem has one class anyway - so **the shift c = 2 costs
nothing** (round-25 gate section B).

### 4.3 Layers 1-2: the survivors are twins-or-smooth

An `n <= y` surviving layers 1 and 2 has neither `n` nor `n+2` divisible by any
prime in `{2} u [3,P) u (z1, x/L]`; i.e. all prime factors of `n` and of `n+2`
lie in `[P, z1] u (x/L, oo)`. If some `q | n` with `q > x/L` then
`n/q <= yL/x < P`, and `n/q` is `P`-rough, so `n/q = 1` and `n` is prime.
Otherwise `n` is `z1`-smooth. The same for `n+2 <= y+2`. Hence

    survivors  subset  T u S_0 u S_2,
    T   = {n <= y : n and n+2 both prime},
    S_0 = {n <= y : n z1-smooth},   S_2 = {n <= y : n+2 z1-smooth}.

Checked exactly, in the two-sided form actually used, at five parameter sets in
`j2_layer_proof.py` section B: **0 violations**.

### 4.4 The two counts

    |T|        <= pi_2(y) <= c_1 y/(log y)^2 <= c_1 y/A^2         (Brun/Selberg)
    |S_0|+|S_2| <= 2 Psi(y+2, z1) <= 3 y rho(u_y),  u_y = log y/log z1

`rho` is Dickman's function; `Psi(y,z) = y rho(u)(1 + O(log(u+1)/log z))` is
Hildebrand's theorem, and our `log z1 = AC/(theta B)` makes the error term
`O(log B / (AC/B))`, i.e. utterly negligible - we are extremely deep in its
range. We then use the elementary `rho(v) <= 1/Gamma(v+1) <= e^{-v(log v - 1)}`.

**Only an UPPER bound for twins is used.** That is the whole reason the
construction is unconditional and parity-free.

### 4.5 Layer 3: the greedy, and it is exactly 2/p

**LEMMA.** Let `R` be a finite set of integers, `p >= 3` prime, `N = |R|`. Then
some two DISTINCT classes mod `p` contain together at least `2N/p` elements.

*Proof.* Let `n_(1) >= n_(2)` be the two largest class counts. `n_(1) >= N/p`,
and `n_(2) >= (N-n_(1))/(p-1)` because the other `p-1` classes hold `N-n_(1)`.
So `n_(1)+n_(2) >= n_(1)(p-2)/(p-1) + N/(p-1)`, which is increasing in `n_(1)`
for `p >= 3`; at `n_(1) = N/p` it equals `N(2p-2)/(p(p-1)) = 2N/p`. QED

This was the step named in advance as the risk (pre-registration PR1). It does
not merely survive - it is **exact**, with no `O(N/p^2)` loss. Applying it at
every `p in [P,z1]`,

    Sigma := prod_{P<=p<=z1} (1 - 2/p)
           = ( prod (1-1/p) )^2 prod (1 - 1/(p-1)^2)
           <= ( prod (1-1/p) )^2  =  sigma^2 (1 + O(1/log^2 P))

by Mertens with Rosser-Schoenfeld's explicit error; `log P = 3B -> oo`, so the
correction tends to 1.

### 4.6 Layer 4: capacity

    2(pi(x) - pi(x/L)) >= (2x/A)(1 - 1.1/B)     (Dusart)

Each of the `2(pi(x)-pi(x/L))` classes is set to `n mod q` for a distinct
remaining survivor `n`; unused classes are set arbitrarily. The construction
covers `[1,y]` provided

    Sigma ( |T| + |S_0| + |S_2| )  <=  2(pi(x) - pi(x/L)).                 (*)

### 4.7 Solving (*)

Divide (*) by `y Sigma` and substitute `y = K x A^3 C^2/B^4`,
`sigma^2 = 9 theta^2 B^4/(A^2 C^2)`. Writing `R := A^2 rho(u_y)`, (*) becomes

    K  <=  2 (1 - 1.1/B) / ( 9 theta^2 M ( c_1 + 3 R ) ),   M -> 1.

Everything hinges on `R`. With `u_y = theta B/C` and `log u_y = log theta + C -
log C`,

    log R = B [ k - (theta/C)(log theta + C - log C - 1) ]   (k = 2 here),

so `R -> 0` **iff `theta > 2`**, and `theta = 2` exactly fails: the bracket is
then `+2(log C + 1 - log 2)/C > 0` and the smooth term swamps the twin term.
Taking `theta = 2 + 4(log C+1)/C` makes the bracket `~ -2(log C+1)/C` and
`log R ~ -2B(log C+1)/C -> -oo`, while `theta -> 2`, so

    K  ->  2/(9 * 4 * c_1)  =  1/(18 c_1).                                 QED

The gate tabulates `K` against `C` up to `C = 10^6` and asserts monotone
convergence to within 0.006% of `1/(18 c_1)`.

## 5. Implications

**(i) THE ROUND-24 GROWTH MODEL IS NOT AN UPPER BOUND** (unchanged from round
25, now on a proved footing). The `~2.56 z (log z)^2` extreme-value model is the
largest gap in a RANDOM set of density `prod(1-k/p) ~ 1/A^k`; `j_k` is a MAXIMUM
over choices, and at `k = 2` this construction exceeds the model by a log. The
sandwich must read "a lower bound the data cannot separate from the models",
never "a truth".

**(ii) THE ONE-LOG SEPARATION BETWEEN j AND j_2 IS STRUCTURAL.** Each class
spent as a split-range Eratosthenes layer converts an `O(1)` Mertens entitlement
into a full log of thinning. At `k = 1` the survivors to be matched are PRIMES
(`~ y/log y`); at `k = 2` they are TWINS (`~ c_1 y/(log y)^2`) - one log fewer,
and that log is exactly Brun's.

**(iii) IT PRICES (P3), THE PAIRED-IWANIEC PROBLEM.** See section 6a.

**(iv) A SELF-CORRECTION OF ROUND 25's GENERAL-k CONSTANT.** Round 25 fixed
`P = A^5` for all `k`. But `P` must exceed `L y/x`, which is of order
`A^(2k-1)`. So `P = A^5` is admissible only for `k <= 3` (coinciding with the
correct cut exactly at `k = 3`), is a factor `(5/(2k-1))^k` too large at
`k = 1, 2`, and is **INADMISSIBLE for k >= 4** - round 25's closed form is too
optimistic there. Round 25's PR3 (the POWER is `2k-1`) stands; its printed
CONSTANT `(5k)^k` does not, for `k >= 4`.

**(v) THE CONSTRUCTION IS PARITY-FREE *BECAUSE* IT STOPS AT RANKIN LEVEL.** The
FGKT/Maynard improvement of the ordinary construction - which removes the
`logloglog` and makes the constant arbitrary - works by producing MANY PRIMES in
a single residue class via a multidimensional sieve. Its `k = 2` analogue would
need many TWINS in a single residue class, i.e. a LOWER bound for twin primes:
precisely the parity barrier. So the layered construction can be unconditional
or it can be FGKT-strength, not both, and the reason is structural rather than
technical. This is the round's sharpest new statement about the method.

## 6. Unsolved questions it touches

1. **The threshold.** Every ingredient is effective, so `x_0` is computable; the
   `o(1)` decays like `(log C + 1)/C`, `C = loglogloglog x`, so writing it down
   honestly is not possible. Whether a DIFFERENT parameterisation gives a
   finite-`z` statement is open and would be genuinely useful (round 25 measured
   that this one admits no choice at all below `log x ~ 300`).
2. **The exponent 4 of `loglog`.** `4 = 2k`; it comes from `sigma ~ B^2` and is
   not optimised. Sharpening `rho` (the `log log u` term) and the greedy moves
   the constant, not the exponent.
3. ~~**The general-`k` shift set.**~~ **ANSWERED IN ROUND 27 - see
   `docs/novel/jk-family.md` §4; gate `research/jk_family.py` section E.** The
   question was: for `k >= 4` the shifts `0,2,...,2(k-1)` are not pairwise
   distinct modulo every odd prime (`3 | 6` already at `k = 4`), so what does
   the optimal shift set look like? **The answer is that it costs nothing.**
   `0,2,...,2(k-1)` is the wrong tuple - from `k = 3` it is not even admissible
   (`0,2,4` covers `Z/3`). Take any admissible `k`-tuple (e.g.
   `{q_1,...,q_k} - q_1` for the `k` least primes `q_i > k`). A collision
   `E_i ≡ E_j (mod p)` then needs `p | E_j - E_i`, hence
   `p <= M_k := max_{i<j}(E_j - E_i)`, a constant in `k`; and the greedy layer
   runs over `[P, z1]` with `P = A^{2k-1} → ∞`, so for large `x` every
   colliding prime lies **below** `P`, inside the Eratosthenes layers, where a
   collision merely means two layers coincide and uses *fewer* than the `k`
   available classes - the survivor structure of §4.3 is untouched. Hence
   `Sigma = prod_{P<=p<=z1}(1-k/p)` with no correction, `K_k` stands as
   printed, and the threshold `x > exp(M_k^{1/(2k-1)})` is under `e^4` for
   every `k <= 12` against this construction's own `log x ~ 300`. **What is
   left is a finite optimisation, not a gap:** which admissible tuple minimises
   `c_1^{(k)}` (equivalently the singular series `S(E)`)? That moves the
   constant only.
4. **The upper side of the family.** Is `j_k << x A^{2k-1+eps}`? At `k = 1` this
   is Iwaniec's territory and open; see 6a.

### 6a. (P3), THE PAIRED-IWANIEC PROBLEM - PRICED

**Statement.** `(P3)`: is `h_2(P(z)) = O(z (log z)^a)` for some `a`?

**What round 26 changes.** Before this theorem, `(P3)` was open with no
constraint on `a`. Now:

* **`a >= 3` is forced** by the theorem of section 1, and `a >= 2k-1` for the
  general `j_k`. So `(P3)` is no longer "is it polylog?" but "**is the polylog
  exponent 3?**".
* **The matching conjecture is now sharp and falsifiable**: `h_2(P(z)) =
  z (log z)^{3 + o(1)}`, i.e. the construction of section 4 is essentially
  optimal. That is a statement someone can attack from either side.

**Price.** NOT REACHABLE, and the reason is structural, not effort:

1. `(P3)` at `k = 1` - "is `j(P(z)) = O(z (log z)^a)`?" - is a **known open
   problem**. The record upper bound is Iwaniec 1978, `j(n) << (omega(n) log
   omega(n))^2`, i.e. `j(P(z)) << z^2` - a full power of `z`, not a polylog -
   and it has stood for 48 years (re-verified against the Erdos problems
   database in round 24; nothing in 2025-2026 touches it).
2. Our `k = 2` version cannot be easier: `j_2 >= j` (the collapse transfer,
   `b - a = p#`), so a polylog bound for `j_2` implies one for `j`.
3. Our own upper ladder reaches `z^{8.04}` explicitly and `z^{4.266+eps}` by
   citation, both far above any polylog, and both sit AT a sifting limit -
   section 2e of `j2-upper-bound.md` shows the exponent IS the sifting limit and
   no level refinement moves it.

**Therefore, honestly labelled: (P3) is strictly harder than an open problem of
Erdos's that has been unmoved since 1978, and this lane will not attempt it.**
What the lane CAN contribute, and now has, is the lower constraint `a >= 3` and
the sharpened conjecture. The residual reachable item is item 4 of section 6:
the family `j_k` gives `2k-1` distinct instances of the same question, and any
UPPER bound of the form `j_k << x A^{f(k)}` with `f(k) < 2k-1` would be an
outright contradiction - i.e. the family supplies free consistency checks on any
future upper-bound claim. That is a referee tool, not a theorem.

## 7. Prior-art check (ROUND 26, dated 2026-08-29)

Re-run because checks EXPIRE (harvester 7d clause 1). What was searched and what
came back:

* **THE HIT: FKMPT Remark 7** - section 2. Read first-hand. It names our
  sieving system. Round 25's blanket novelty sentence is withdrawn and replaced
  by the three precise statements of section 2. **This is the round's most
  important citation finding and it must appear in the paper's related-work
  section, quoted.**
* **No theorem on large gaps between consecutive twin primes or prime k-tuples
  by an Erdos-Rankin covering.** This was named in round 25 as the largest
  novelty risk; searched again (arXiv metadata: "gaps between twin primes",
  "large gaps" + "twin primes", "gaps between prime pairs", "covering system" +
  "twin primes") and no such paper exists. FKMPT Remark 7 is why: the pigeonhole
  bound already beats what a covering gives for actual twin primes, so nobody
  had a reason to build the construction. **That is also why our object must be
  `j_2`, not twin-prime gaps.**
* **Erdos-Rankin with >= 2 classes per prime: none found.** Maynard's survey
  (arXiv:1910.13450) Lemma 5 and FGKT sec. 1 both state the framework as one
  class per prime. The nearest-looking precedent is Maier-Pomerance's idea of
  using one class at a large prime to remove TWO survivors - a different thing,
  and worth naming in the paper so a referee does not confuse them.
* **Erdos problems database (accessed 2026-08-29):** #689 asks for one class per
  prime covering every integer at least TWICE (multiplicity of covering, not of
  classes); #1205, #1200 likewise one class each. **#687 and #970 confirm
  Iwaniec 1978 is STILL the record upper bound for the ordinary Jacobsthal
  function, both open**, which is what prices (P3) in section 6a.
* **Kalmynin-Konyagin arXiv:2302.00459** replaces the progression `x+i` by
  `x+f(i)` and stays ONE-dimensional. Not our object. Verdict unchanged.
* **`j_k` under any name: not found.**

RELAY-SOURCING NOTE, per harvester 7d clause 2: the sweep was run by a
sub-search; the two LOAD-BEARING items (FKMPT Remark 7, and Lichtman's twin
constant) were then re-read first-hand by me before being used. Items I did not
re-read myself are labelled as such in `docs/proof-search/harvester.md`
section 10.

## 8. Reproduction

* `research/j2_layer_proof.py` -> `research/data/j2_layer_proof.out`. Sections
  A (the greedy lemma, exact, 40,000 random distributions), B (the two-sided
  survivor structure, 0 violations), C (the assembly, `K` to `C = 10^6`),
  D (`P = A^(2k-1)` forced and optimal), E (k = 1 constant-level calibration
  against Rankin's `e^gamma`, and the general-k table), F (the honest
  boundary), G (pre-registration scored: PR1-PR5 all confirmed).
* `research/j2_rankin_layer.py` -> `research/data/j2_rankin_layer.out`
  (round 25: the restatement brute-forced, `c = 2` costs nothing, the survivor
  structure, the shape calibration, the finite-z negative).

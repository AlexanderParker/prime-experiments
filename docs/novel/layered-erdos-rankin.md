# layered-erdos-rankin - the Erdos-Rankin construction run k times, and the k-class Jacobsthal function

Round 25 (harvester). Gate: `research/j2_rankin_layer.py`, ALL ASSERTIONS GREEN.
Status is stated bluntly in section 3: **asymptotic bookkeeping, script-verified
and calibrated against a published theorem; NOT a written-out proof.**

## 0. Definitions

`j_k(m)` - the **k-class Jacobsthal function**: the longest run of consecutive
integers that can be covered by choosing at most `k` residue classes modulo each
prime `p | m` (one class only at p = 2). `j_1` is the ordinary Jacobsthal
function of `m`. For `m = p_n#` the paired Jacobsthal function of Ziller-Morack
satisfies `h_2(p_n#) = j_2(p_n#)`, by the round-23 restatement re-verified
exactly here at z = 3, 5, 7 against the known values 6, 18, 30 (section A of the
gate). Write `x = p_n`, `A = log x`, `B = log A`, `C = log B`.

## 1. What it is

    CLAIM (bookkeeping).   j_k(P(x))  >>  x A^(2k-1) C^k / ((5k)^k B^(2k)) .

    k = 1  is the Erdos-Rankin / Ford-Green-Konyagin-Tao interval length
           x log x logloglog x/(loglog x)^2 - a THEOREM, and the calibration.
    k = 2  gives   h_2(P(z))  >>  z (log z)^3 (logloglog z)^2/(100 (loglog z)^4),
           TWO logs above the round-24 proved bound (P1) = (1.349+o(1)) z log z,
           and ONE log above what round 24's open problem (P2) asked for.

## 2. Why it might be novel

The ordinary Erdos-Rankin construction spends its one class per prime on two
jobs at once, and the whole art of it is that class 0 on a SPLIT range
`[2,P) u (z1, x/4]` delivers survivor density ~1/log y where its Mertens
entitlement is only O(1) - a full log of "non-independence" gain - while leaving
the middle range `[P, z1]` free for a greedy/random sparsification. **Nobody
appears to have asked what happens when you have TWO classes per prime and can
therefore run that trick twice.** The answer is that the second layer is a
SHIFTED Eratosthenes (class -2 instead of class 0), the joint survivor set is the
set of twin primes rather than the primes, and the gain is another full log.

Prior art, checked first-hand 2026-08-29:

* **Ford-Konyagin-Maynard-Pomerance-Tao, "Long gaps in sieved sets"**
  (arXiv:1802.07604, JEMS 2021). They bound gaps in
  `{n : n mod p not in I_p for all p <= x}` for **GIVEN** `I_p` with `|I_p| <= C_0`
  bounded and equal to 1 on average, obtaining `x(log x)^(1/exp(C C_0))`. That is
  the ADVERSARIAL problem - the classes are handed to them. Ours CHOOSES the
  classes and asks for the maximum. Neither result contains the other, and the
  shapes are utterly different (a tiny power of log against `A^(2k-1)`). **It must
  be cited**, and round 24's "RELAY-SOURCED, re-verify before citing" flag on this
  paper is hereby discharged.
* Ford-Green-Konyagin-Tao (arXiv:1408.4505) and Maynard: the k = 1 case, and the
  calibration target. Their `c` may be taken arbitrarily large; we make no such
  claim - our constants are unoptimised.
* Ziller-Morack (arXiv:1706.00317) define `j_2` and conjecture only an upper
  bound (`h_2 < p^2 - p`); they prove no lower bound of any kind.
* The k-class Jacobsthal function `j_k` does not appear in the literature under
  any name we can find (the round-24 citation-graph sweep found "paired
  Jacobsthal" returning NO RECORDS in zbMATH at all).

## 3. STATUS - read this before quoting anything above

* Sections A-C of the gate verify the FINITE, checkable ingredients **exactly**:
  the restatement (brute-forced against h_2 at z = 3,5,7), that the shift c = 2
  collides with no odd prime's class 0, and the survivor-structure claim
  (survivors of the two Eratosthenes layers are twins-or-smooth) at four
  parameter sets by direct sieving.
* Section D verifies the ASYMPTOTIC BOOKKEEPING against a published theorem: the
  same optimiser, run at k = 1, tracks the FGKT closed form with a residual whose
  spread is **0.072 over eight decades of log x** (and 0.271 at k = 2). That is
  the strongest test available - it is calibrated against someone else's theorem
  rather than against itself.
* **NOT DELIVERED**: a written-out proof with the constants tracked. The three
  analytic ingredients (Selberg/Brun upper bound for twins, Rankin's smooth-number
  bound, the pigeonhole greedy) are all standard and unconditional, but they have
  not been assembled on paper. Until they are, this is a claim about a
  bookkeeping, not a theorem.
* The `(loglog z)^O(1)` exponent is **not optimised** - 4 is what this parameter
  choice gives, not what the method gives.
* No kernel check: the statement is asymptotic, so there is nothing finite to
  check.

## 4. The construction

By the restatement, `j_2(P(x)) - 1` is the longest `[1,L]` coverable using one
class mod 2 and two arbitrary classes mod p for every odd `p <= x`. Fix `c = 2`
(and note `0 != -2 mod p` for every odd p, so the two layers never collide; at
p = 2 the paired problem has only one class anyway, and n odd => n+2 odd, so
layer 2 never wants the modulus 2). Put `P = A^5` and `z1 = x^(1/u)`.

    LAYER 1   class 0   mod p  for p = 2 and for p in [3,P) u (z1, x/4]
    LAYER 2   class -2  mod p  for p in [3,P) u (z1, x/4]
    LAYER 3   the two free classes at each p in [P, z1], used GREEDILY
    LAYER 4   the two free classes at each p in (x/4, x], used for MATCHING

After layers 1-2 a survivor n has every prime factor of BOTH n and n+2 inside
`[P,z1] u (x/4, oo)`. For `y < xP/4` the cofactor argument forces each of n, n+2
to be a prime in `(x/4, y]` or a P-rough z1-smooth number, so

    survivors  in  {n <= y : n, n+2 both prime}  u  {n : n z1-smooth}
                                                 u  {n : n+2 z1-smooth},

of size at most `8 S(2) y/(log y)^2 + 2 Psi(y,z1)` - **both bounds
unconditional** (that is the whole point: we need an UPPER bound on twin primes,
which Brun/Selberg supply, and never a lower one, so the construction is
parity-free). Layer 3 shrinks that by at most
`prod_{P<=p<=z1}(1-1/p)(1-1/(p-1)) ~ (log P/log z1)^2` by pigeonhole alone.
Layer 4 finishes if what remains is at most `2(pi(x) - pi(x/4))`.

Balancing the smooth term against the twin term forces `u ~ kB/C`, whence
`S = log P/log z1 = 5Bu/A` and the closed form of section 1.

## 5. Implications

**(i) THE ROUND-24 GROWTH MODEL IS NOT AN UPPER BOUND, and must be relabelled.**
Round 24 recorded "TRUTH z^(1+o(1)), best model ~2.56 z (log z)^2" from a
parameter-free extreme-value model. That model is the largest gap in a RANDOM set
of density `prod(1-k/p) ~ 1/A^k`, namely `x/density = x A^k`; it is a
random-choice heuristic, and `j_k` is a MAXIMUM over choices. At k = 1 the
heuristic is right (Rankin attains `x A` up to loglog powers). At k = 2 this
construction exceeds it by a log. **So the round-24 sandwich paragraph must say
"a lower bound the data cannot separate from the models" rather than "a truth",
and the model's status is heuristic-not-ceiling.** This is a genuine correction to
harvester 3c, found by the construction that the same round asked for.

**(ii) The one-log separation between j and j_2 is structural, not incidental.**
Each class spent as a split-range Eratosthenes layer converts an O(1) Mertens
entitlement into a full log of thinning; k classes therefore buy `2k` logs of
density against a matching capacity of `x/log x`, giving `x A^(2k-1)`. The paired
problem's "one log thinner" (harvester 3a) is exactly the k = 1 -> k = 2 step.

**(iii) It sharpens what ZM Conjecture 6 is asking.** Conjecture 6 says
`h_2 < p^2 - p`. Against a lower bound of `z A^3` and an explicit upper bound of
`z^8.04` (Theorem 2G, j2-upper-bound.md sec. 10), the conjecture sits far above
the construction and far below the proved ceiling - it remains, as round 24 said,
easy as a statement about the truth and hard as a statement provable by sieve.

**(iv) NO FINITE-z CONTENT, and the honest form of that.** The pre-registration
predicted the layering would be a LOSS at reachable z. **That prediction was
wrong in its mechanism**: the layering is not a loss, it simply DOES NOT EXIST
below `log x ~ 300` (`z ~ e^300`), because `[P, z1]` is empty there. Same
practical conclusion - **(P1)'s `h_2 >= (1.349+o(1)) z log z` remains the bound to
quote at any z a human will ever see** - but arrived at differently, and the
prediction as worded is scored WRONG.

## 6. Unsolved questions it touches

* Round-24 problem **(P2)** (`h_2 >> z(log z)^2/(loglog z)^O(1)`) is superseded:
  the construction asks for and gets one more log. (P2) should be replaced by
  **(P2') write the k = 2 construction out as a theorem with constants**, which
  is ordinary work, not research.
* **(P3)**, the paired-Iwaniec UPPER question `h_2 = O(z(log z)^A)`, is untouched
  and is now the more interesting side: with a lower bound of `z(log z)^3` and an
  upper bound of `z^7.94`, the gap is the whole problem.
* The general-k statement `j_k >> x A^(2k-1)` is a family nobody has stated. Its
  UPPER side is untouched: is `j_k << x A^(2k-1+eps)`? At k = 1 that is Iwaniec's
  `j(P(x)) << x^2` territory and wide open.
* The falsification target of harvester 3c stands and gains sharpness: **one exact
  h_2 beyond p_n = 73**. The competing readings are now `z(log z)^2` (heuristic)
  and `z(log z)^3` (this construction, asymptotically), which differ by a further
  factor `log z`.

## 7. Prior-art check

Run by me, dated **2026-08-29**, first-hand: arXiv:1802.07604 (FKMPT abstract and
main-theorem statement), arXiv:1408.4505 (FGKT abstract and bound), and Tao's own
exposition of the Erdos-Rankin construction (terrytao.wordpress.com, 2014-08-21)
for the four-stage structure and the parameter `z = exp(log x logloglog x/(4
loglog x))`. Nothing found that runs the construction with more than one class
per prime, and nothing that names `j_k`. Round 24's flag on FKMPT ("relay-sourced,
re-verify before citing") is discharged: the paper is real, its theorem is the
adversarial one, and it does not contain this.

## 8. Reproduction

`research/j2_rankin_layer.py` - sections A (restatement brute-forced), B (c = 2
costs nothing), C (survivor structure, exact), D (bookkeeping, calibrated at
k = 1 against FGKT and run at k = 1..5), E (finite-z, honest negative), F
(pre-registration scored: PR1 confirmed, PR2 confirmed, PR3 confirmed, PR4
confirmed but its mechanism refuted). Output: `research/data/j2_rankin_layer.out`.

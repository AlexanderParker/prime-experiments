# j2-lower-ladder - the first lower bound on h_2 from the paired structure, and the growth law reread

Round 24 (Harvester), 2026-08-28. Status per claim below; script
research/j2_lower2.py (all assertions green), predecessor research/j2_lower.py
(round 23). This doc SUPERSEDES the round-23 "THE LOWER LADDER" subsection of
j2-upper-bound.md section 1 in three places, each marked there.

## 1. What it is

### 1a. The restatement (proved, one line, script-checked exactly at z = 3..13)

j_2(P(z)) - 1 is the longest interval [1, L] such that for some even E every
n in [1, L] has n or n + E divisible by a prime <= z. The n that are themselves
divisible are covered for free, so the condition is exactly:

    every z-ROUGH n in [1, L] has n + E divisible by some prime <= z.

The ordinary (one-class) Jacobsthal problem is the same statement with "every
z-rough n" replaced by "every n". Rough numbers have density ~ e^{-gamma}/log z,
so the paired problem covers a set THINNER BY A FACTOR ~ log z - one logarithm,
not a power. Brute-forced against Ziller-Morack's h_2 at z = 3, 5, 7, 11, 13:
exact agreement (research/j2_lower2.py L1).

### 1b. THEOREM (P1) - the construction (proved; certificates verified by independent sieve at z = 13..10^5)

    h_2(P(z)) >= (1/(2 e^{-gamma} C_2) + o(1)) z log z = (1.3490 + o(1)) z log z,

C_2 the twin prime constant. Proof in section 3. This is the first lower bound
on h_2 = j_2 (Ziller-Morack's paired Jacobsthal at primorials) that uses the
paired structure at all; the only previous rung was the collapse transfer
j_2 >= j (round 21), and the best proved ordinary bound (FGKMT) is
z log z * logloglog z/loglog z = o(z log z). So the construction strictly beats
everything the transfer can ever give, by a factor tending to infinity.

Certificates as run do better: L/(z (log z)^2) sits near 0.7 for z = 10^3..10^5
(varying < 1.25x) while L/(z log z) climbs - the construction as executed
already tracks the one-extra-logarithm law; only its worst-case analysis loses
the second log.

### 1c. The growth law, reread (measured; corrects round 23)

Round 23 tabulated "TRUTH: h_2 ~ (p_n^2 - p_n)/2 (measured)". That reading is
UNSUPPORTED BY ITS OWN DATA. On Ziller-Morack's 21 exact values (p_n = 5..73):

- c z^2 and c z (log z)^2 fit EQUALLY WELL (implied-constant spread 1.87x each);
  the two laws differ by z/(log z)^2, which itself only moves 2.1x over the
  table, so 21 points at these sizes cannot separate them.
- The residual drifts run in OPPOSITE directions: h_2/(z^2-z) FALLS 0.962 ->
  0.499 over z = 13..73; h_2/(z log^2 z) RISES 1.754 -> 1.951. Neither settles.
- The parameter-free extreme-value model (survivor density V(z), largest gap
  ~ log(period x density)/density) gives h_2 ~ (theta(z) + log V)/V ~
  2.56 z (log z)^2 and j ~ 1.78 z log z; measured/model is 0.78-0.92 for h_2
  (z >= 7, excluding the z = 5 outlier 1.64) and 0.34-0.47 for j. The model
  says the paired answer carries exactly ONE MORE LOGARITHM than the ordinary
  one - the same factor as 1a - and both are z^{1+o(1)}.
- The discriminating measurement (research/j2_lower2.py L4): the paired-minus-
  ordinary LOCAL EXPONENT gap, measured on ranges 11..29 / 23..47 / 43..73 /
  11..73, is 0.753 / 0.474 / 0.328 / 0.568 - always in (0.2, 0.8), never near
  the +1.0 that quadratic-vs-linear separation requires. Logarithmic separation
  predicts +1/log z to +2/log z ~ 0.25-0.50 at these sizes. ASSERTED in-script.
- The ratio test: h_2/j runs 1.33x..2.51x the density ratio W(z)/V(z)
  (~1.44 log z), drifting UP; in exponent terms h_2/j = (W/V)^t with
  t = 1.22..1.51 over the table. Between one and one-and-a-half powers of the
  density ratio - consistent with logarithmic separation, not decisive alone.
  (A first draft claimed "tracks W/V within 1.3x"; the assertion gate caught
  the overstatement and the honest numbers are these.)

VERDICT (measured, honestly bounded): the data cannot decide between z^2 and
z (log z)^2; the model plus the exponent-gap measurement both point at
h_2 = z^{1+o(1)} (specifically ~ 2.56 z (log z)^2), NOT at z^2/2. Round 23's
"truth ~ p^2/2" is downgraded to "one of two readings, and the less supported
one".

### 1d. Retraction (round 23's capacity argument)

Round 23 argued the paired problem is quadratic because its covering capacity
sum_p omega(p)/p (2.19/2.41/3.01 at z = 13/19/73) exceeds the ordinary one's
(1.34/1.46/1.76): "the ordinary covering is counting-constrained at every
computable size, the paired one is not". RETRACTED: capacity is not scale-free.
The ordinary capacity reaches those same paired values at z ~ 4e3 / 5e4 / 3e5 /
6e6 (computed exactly in-script), where the ordinary answer is still
z^{1+o(1)}. Capacity above 1 never implied a quadratic answer for either
problem. What actually separates them is 1a's factor log z, once.

### 1e. The open problems, restated so they are the right problems

Round 23's named open problem "prove h_2 >> p_n^{1+delta}" is, on 1c's model,
asking for something FALSE, and its justification was 1d. Replaced by:

- (P1) proved here: h_2(P(z)) >= (1.349 + o(1)) z log z.
- (P2) open, the real lower target: h_2(P(z)) >> z (log z)^2 / (loglog z)^{O(1)}
  - carry Rankin/FGKMT smooth-number machinery through the paired construction.
  Still a construction; still nothing parity-shaped in the way. Evidence the
  machinery lands on 2-class sieves: Kalmynin-Konyagin (section 6) extract two
  extra Rankin-type log factors from quadratic polynomial sieves.
- (P3) open, upper: is h_2(P(z)) = O(z (log z)^A) for some A? The paired
  analogue of Iwaniec's j(n) << (k log k)^2. Would refute the quadratic reading
  outright. Our own explicit ladder gives only z^15 (j2-upper-bound.md round-24
  update); best exponent 4.266 by citation.
- (P4) reframe of ZM Conjecture 6 (h_2 < p^2 - p): on 1c's model it is TRUE
  WITH ROOM - z (log z)^2 against z^2 - i.e. it asks for far less than the
  truth. This does not touch round 22's separate point (exponent 2 on a
  kappa = 2 problem sits below the conjectured sifting floor as a SIEVE
  statement); it says the conjectured extremal behaviour is probably not
  extremal.

FALSIFICATION TARGET (decidable): one exact h_2(p_n#) beyond p_n = 73 separates
the models - at z = 151/199/251 the extreme-value model and the fitted
z log^2 z law sit a factor ~2.5-3.6 below the quadratic. ZM's own algorithm
reached 73 in 2017; a dedicated run at 151+ decides.

## 2. Why it might be novel

- Ziller-Morack (both papers + ancillary files, corpus full-text reads) prove
  NO lower bound on j_2 beyond elementary monotonicity; the collapse transfer
  (round 21) was this project's, and it is o(z log z).
- No published work states any lower bound for the two-free-classes-per-prime
  covering problem (rounds 21-23 sweeps plus this round's polynomial-Jacobsthal
  search; section 6).
- The z log z threshold here is NOT the ordinary problem's: for j, reaching
  c z log z with explicit c is Rankin-hard (the smooth numbers must be
  covered); for h_2 the free 0-classes eat every smooth number, and c z log z
  becomes elementary. That asymmetry is the content of 1a and appears nowhere
  in the literature found.

## 3. Proof of Theorem (P1)

Fix eps > 0, put L = c z log z with c = (1 - eps)/(2 e^{-gamma} C_2). E will be
even, so the p = 2 classes {0 mod 2} for n and n + E coincide and every even n
is covered. For odd p <= z the killed classes are {0, -E mod p}. Every
n in [1, L] with a prime factor <= z is covered by the 0-classes; what remains
is T_0 = {1} cup {primes in (z, L]} (using L < z^2), |T_0| = pi(L) - pi(z) + 1.

GREEDY PHASE: for each odd prime p <= w := z^{1-eps'} in increasing order:
every element of the current survivor set T is coprime to p, so T meets only
the p - 1 nonzero classes mod p, and the largest holds at least |T|/(p-1)
elements; set E = -c_p mod p for that class c_p (then n = c_p mod p implies
p | n + E). Each step multiplies |T| by at most (p-2)/(p-1), so afterwards
|T| <= |T_0| A(w), A(w) = prod_{3<=p<=w} (p-2)/(p-1) =
(2 e^{-gamma} C_2 + o(1))/log w (standard; in-script check: ratio 1.0000 at
w = 10^6).

MATCHING PHASE: give each surviving q in T its own unused prime p in (w, z]
and set E = -q mod p (q is 1 or a prime > z >= p, so the class is nonzero).
Feasible iff |T_0| A(w) <= pi(z) - pi(w); with the above choices the sides are
(c + o(1)) z (2 e^{-gamma} C_2)/((1 - eps') log z) and (1 + o(1)) z/log z, so
every c < (1 - eps')/(2 e^{-gamma} C_2) closes. E exists by CRT over 2 and the
distinct odd primes, is even, and is nonzero mod every greedy prime. Every
n in [1, L] is covered, so j_2(P(z)) - 1 >= L; let eps, eps' -> 0. The constant
is 1/(2 e^{-gamma} C_2) = 1.34904... QED.

Certificate check (research/j2_lower2.py L2): the construction was executed at
z = 13, 19, 43, 73, 200, 10^3, 10^4, 10^5; each output E (as residues) was
re-verified by an INDEPENDENT sieve of the whole interval (code path disjoint
from the builder), and at every z with known h_2 the reached L is <= h_2. All
asserted.

## 4. Implications

- The j_2 ladder now has real rungs on BOTH sides: 1.349 z log z below,
  z^15 explicit / z^{4.266} by citation above (round-24 j2-upper-bound.md).
- Unit 1's referee narrative changes: the sandwich is no longer
  "p^{1+o(1)} .. p^{4.266} around a truth of p^2/2" but "c z log z .. z^{4.266}
  around a truth the data cannot separate from ~2.6 z (log z)^2, with the
  quadratic reading measurably losing ground".
- The percentile/extremal work (twin-percentile.md) is untouched - it compares
  differences within a fixed z.
- ZM Conjecture 6 gains support and loses glamour simultaneously (P4).

## 5. Unsolved questions it touches

(P2), (P3) above; the h_2(151+) computation as a decidable model test; whether
greedy's measured extra logarithm (1b) can be proved directly (the size of the
largest residue class of the primes in (z, L] mod p is a mean-value /
large-sieve statement - named, not attempted this round).

## 6. Prior-art check (run by me, dated 2026-08-28)

- Kalmynin-Konyagin, "A polynomial analogue of Jacobsthal function",
  arXiv:2302.00459 (full text on disk this session, read): they bound
  j_f(P(y)) = max shift x making x + f(i) non-coprime to P(y) for all i <= m -
  shifted polynomial VALUES. For quadratic f the per-prime killed set is the
  <= 2 square roots of a global shift: a one-parameter family of 2-class sieves
  DIFFERENT from ours ({0, -E} mod every p, one global E) - neither family
  contains the other, and their covered object is a polynomial sequence, not an
  interval. "Jacobsthal" in the paired/two-residue sense: absent. NOT prior art
  for (P1); STRONG evidence for (P2) (their M(f) = 2 factor is two Rankin-type
  logs on a 2-class quadratic sieve).
- Ford-Green-Konyagin-Maynard-Tao, "Long gaps between primes" (J. AMS 31,
  2018): ordinary problem only; enters here via the transfer comparison.
  (The round-23 doc cited its bound with Rankin's (loglog)^2 denominator; the
  FGKMT form has a single loglog - corrected in j2-upper-bound.md this round.)
- Ford-Konyagin-Maynard-Pomerance-Tao, "Long gaps in sieved sets" (located by
  this round's literature search): sieving ONE residue class per prime in
  general position - the relayed reading is that it does not treat two free
  classes per prime. Flagged RELAY-SOURCED; re-verify verbatim before citing.
- Rounds 21-23 sweeps (Semantic Scholar citation graph, zbMATH, OpenAlex, OEIS
  A288815, arXiv full metadata): no paired-Jacobsthal lower-bound literature
  exists; re-confirmed unchanged this round for the lower side.

## 7. ROUND 25 - (P2) SUPERSEDED, AND THE GROWTH MODEL DEMOTED
## (research/j2_rankin_layer.py, all assertions green; full write-up in
## docs/novel/layered-erdos-rankin.md)

Round 24's named problem (P2) asked for `h_2 >> z (log z)^2/(loglog z)^O(1)` by
Rankin-style layering. Round 25 built the layering. It gives **one log more than
(P2) asked for**, and it generalises to a family:

    j_k(P(x))  >>  x A^(2k-1) C^k/((5k)^k B^(2k)),   A = log x, B = log A, C = log B

with `j_k` the k-classes-per-prime Jacobsthal function. `k = 1` IS the published
Erdos-Rankin / Ford-Green-Konyagin-Tao length `x log x logloglog x/(loglog x)^2`;
`k = 2` is ours:

    h_2(P(z))  >>  z (log z)^3 (logloglog z)^2 / (100 (loglog z)^4).

MECHANISM, in one sentence: class 0 on a SPLIT range `[2,P) u (z1,x/4]` buys a
full log of thinning where its Mertens entitlement is only O(1) - that is Rankin's
gain - and the paired problem's SECOND class buys the same gain a second time by
running the identical trick on `n+2` instead of `n`, so the joint survivor set is
the twin primes rather than the primes. Only an UPPER bound on twin primes is
needed (Brun/Selberg), so the construction is parity-free.

**STATUS: asymptotic bookkeeping, script-verified and calibrated against a
published theorem, NOT a written-out proof.** The finite ingredients are exact
(the restatement of sec. 1a re-brute-forced at z = 3,5,7; the shift c = 2 costing
no class; the twins-or-smooth survivor structure by direct sieving); the
bookkeeping is validated by running the SAME optimiser at k = 1 and checking it
tracks the FGKT closed form with residual spread 0.072 over eight decades of
log x. See layered-erdos-rankin.md sec. 3 for the full list of what is not
delivered.

### 7a. CORRECTION TO SECTION 1c - the extreme-value model is NOT a ceiling

Section 1c above records "TRUTH z^(1+o(1)), best model ~2.56 z (log z)^2". That
model is the largest gap in a RANDOM set of density `prod(1-k/p) ~ 1/(log z)^k`,
i.e. `z (log z)^k` - a **random-choice heuristic**, while `j_k` is a MAXIMUM over
choices. At k = 1 the heuristic happens to be right (Rankin attains it up to
loglog powers). At k = 2 the layered construction **exceeds it by a full log**.

So the sandwich of 1c must be re-labelled:

    proved lower  h_2 >= (1.349+o(1)) z log z                    [(P1), finite-z]
    bookkeeping   h_2 >> z (log z)^3 (lll z)^2/(ll z)^4          [round 25]
    HEURISTIC     ~2.56 z (log z)^2   -- NOT a ceiling; a random-choice model
    proved upper  p_n^8.04162 explicit / p_n^(4.266+eps) by citation

and the word "TRUTH" must not be attached to the model. This is the round-24
corollary "MODEL CLAIMS EXPIRE LIKE CITATIONS" firing on the round-24 model.

### 7b. The problems, re-stated

* **(P2) is superseded.** Replace it by **(P2') write the k = 2 construction out
  as a theorem with every constant tracked.** That is ordinary work, not research:
  the three ingredients (Selberg/Brun twin upper bound, Rankin's Psi bound,
  pigeonhole greedy) are standard and unconditional.
* **(P3)** (the paired-Iwaniec upper question) is now the interesting side: the
  gap runs from `z (log z)^3` to `z^7.94`.
* **NEW (P5): the k-class family.** Is `j_k(P(x)) << x A^(2k-1+eps)`? Nobody has
  stated `j_k`, let alone bounded it above.
* The falsification target is unchanged and sharpened: **one exact h_2 beyond
  p_n = 73**. The competing readings are now `z(log z)^2` (heuristic) and
  `z(log z)^3` (construction), a further factor of `log z` apart.

### 7c. Honest negative

There is **no finite-z content**. The pre-registration predicted the layering
would be a LOSS at reachable z; that was wrong in mechanism - the construction
does not exist at all below `log z ~ 300`, because the greedy range `[P, z1]` is
empty there. Practical conclusion unchanged: **(P1) is the bound to quote at any
z a human will ever see.**

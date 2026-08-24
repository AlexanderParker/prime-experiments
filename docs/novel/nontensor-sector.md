# The non-tensor sector of the sieve machine: exactly rank 2 at depth 1, unbounded at window depth

Status: PROVED (depth-1 rank; the merge-cut structure theorem; the 2^n bound)
+ SCRIPT-VERIFIED with exact integer/mod-p ranks
(`research/nontensor.py`, machines 11/13/17/19/23). Established round 22
(Lateral, the round's spine question). Prior-art check: NOT YET CHECKED
(section 6).

## 1. WHAT IT IS

Plain language. Everything the machine can compute gear by gear is a tensor
(Kronecker) product over the gears: the exposure projector, the shift, the
circulant, the open count, the autocorrelation. The one object that is NOT a
product is BLOCKING, B = I - (x)_q E_q - the complement of a product - and
that single failure to factor is where F(M), requirement (D), and Wall V all
live (matrix-formulation piece 4; eigenvalue-statistics' localisation of the
GUE question). The round-22 spine question was: HOW BIG is that non-factoring
part, and DOES IT GROW WITH THE MACHINE? - because a bounded answer means a
fixed-arity rule exists, and an unbounded one means only an arity-free
generator (nilpotency) can work. This entry answers it, exactly, with three
results that pull in different directions and together locate the difficulty
precisely.

Definitions. For a bipartition of the gears G1 | G2 (d1 = prod G1,
d2 = prod G2, d1 d2 = P), CRT makes any function on Z_P a d1 x d2 matrix.
Its rank is the SCHMIDT RANK across that cut: rank 1 means "a product of a
G1-thing and a G2-thing", rank r means "a sum of r products and no fewer".
Two standard facts make this the right measure: max over cuts of the Schmidt
rank is a certified LOWER BOUND on the tensor rank over the full m-fold gear
partition, and rank over GF(p) is a certified lower bound on rank over Q - so
measured growth cannot be an artifact.

RESULT 1 (DEPTH 1 - a theorem; bounded). Reshaping the blocking vector
b = 1 - (x)_q e_q across ANY cut gives J - x y^T with J = 1 1^T,
x = (x)_{G1} e_q, y = (x)_{G2} e_q. Both x and y are non-constant 0/1
vectors (every gear has teeth and exposure), so

    SCHMIDT RANK OF B = 2 EXACTLY, at every cut, at every machine,

while the exposure E_M has rank 1 at every cut. Likewise
BS = (x)S_q - (x)(E_q S_q) has operator Schmidt rank exactly 2. THE ENTIRE
NON-TENSOR SECTOR AT DEPTH 1 IS ONE RANK-ONE CORRECTION. It does not grow.
The difficulty of F is therefore not dimensional at depth 1.

RESULT 2 (THE MERGE CUT - a theorem; linear, machine-independent in form).
Cut off the TOP gear q'. Then

    V[r, k] = prod over i with (k+i) OPEN in the old machine of [ r+i in T_q' ]

- the column depends ONLY on the old machine's opening pattern O inside the
window, which is the merge law's own content. Since |T_q'| = 2, for n <= q'
the column VANISHES unless |O| <= 2, and the surviving columns are: the
all-ones vector (present iff n < F_old), the vectors 1_{T - i} (one per
realised singleton class), and single basis vectors e_r (one per realised
literal pair, i.e. per offset difference +-2u' mod q'). Hence

    rank_n(merge cut) = [n < F_old] + #singleton classes + #literal pairs
                      <= 2n + 1.

VERIFIED EXACTLY (measured rank == predicted rank at every row) at machines
11/13/17/19/23 for all n <= min(14, q'). The MERGE DIRECTION IS FIXED-ARITY,
and this is a structural reason why the merge law is a theorem at all.

RESULT 3 (WINDOW DEPTH, INTERNAL CUTS - measured; UNBOUNDED). What F actually
asks about is the window indicator

    v_n(k) = prod_{i<n} b(k+i),   F(M) = min{ n : v_n == 0 },

since (BS)^n = diag(v_n) S^n. Inclusion-exclusion gives
v_n = sum_{I subset [n]} (-1)^{|I|} (x)_q (prod_{i in I} e_q(.+i)), so
rank_n <= min(2^n, d1, d2), with rank_1 = 2 and rank_F = 0. The profile in
between is the size of the sector at depth n. Measured (exact mod-p ranks at
two primes, agreeing):

  * at the FIXED corridor cut {5,7} (d1 = 35 for every machine) the peak
    rank is 15, 26, 33, 35 at machines 13, 17, 19, 23 (at depths 6, 8, 10,
    11) - it SATURATES the cut;
  * AT EVERY FIXED CUT the peak rank rises from machine 19 to machine 23,
    and five cuts become FULL (peak = d1, the largest rank the cut can
    carry):

        cut       d1     peak at m19   peak at m23
        {5,7}     35        33            35  FULL
        {5,11}    55        48            55  FULL
        {7,11}    77        69            77  FULL
        {7,13}    91        73            86
        {7,17}   119        81           104
        {7,19}   133        83           103
        {11,13}  143       126           143  FULL
        {11,17}  187       140           187  FULL
        {11,19}  209       161           199
        {13,17}  221       138           220
        {13,19}  247       119           244
        {17,19}  323       109           286
        {5,7,11} 385       119           201

  * every SINGLE-GEAR cut is already FULL from machine 17 on (peak rank
    5/5, 7/7, 11/11, 13/13, 17/17 at m17; and 19/19 as well at m19), while at
    machine 11 only the gear-5 cut is full - saturation arrives cut by cut as
    the machine grows;
  * the certified tensor-rank lower bound (max over measured cuts) is

        machine    11    13    17     19     23
        TR_low      6    17    54    161    326

VERDICT. The sector fills whatever dimension a cut makes available - at
machine 23 it attains the FULL rank of five different cuts - so the tensor
rank of the deep window indicator grows at least like the largest available
cut, ~ sqrt(P) = exp(Theta(y)). THE NON-TENSOR SECTOR GROWS WITHOUT BOUND AT
WINDOW DEPTH, while being exactly 2-dimensional at depth 1 and linear across
the merge cut.

WHERE THE GROWTH LIVES, AND WHY IT IS INVISIBLE. The growth is carried by
(BS)^n = diag(v_n) S^n - which is NILPOTENT, spectrum {0} at every depth
(see farey-chebyshev-spectrum.md). So the direction in which the sector grows
is exactly the direction that has no spectrum, no eigenvalues, and no
bounded-order correlation signature. That is the same wall three independent
frames already hit: the tropical 2-point boundary (r20 R37), the operator
counting boundary (Constructor R41: no function of the marginal data bounds
the index of the sum), and the bounded-moment LP that never bites (Lateral
r21, margins 1e1..1e10 and growing).

## 2. WHY IT MIGHT BE NOVEL

Kronecker rank / operator Schmidt rank is standard numerical linear algebra
(Van Loan), and "blocking is the complement of a product" is one line. What
appears unrecorded:

- the exact statement that a two-teeth sieve's blocking operator has Schmidt
  rank exactly 2 across EVERY gear bipartition, so the non-tensor sector at
  depth 1 is one-dimensional and machine-independent;
- the DEPTH-GRADED measurement: rank_n <= min(2^n, d1, d2) with a measured
  profile that saturates the cut, giving a certified lower bound on tensor
  rank growing like sqrt(P) - a quantitative form of "the joint space is
  irreducible";
- the merge-cut structure theorem, which derives the merge law's
  old-machine-only character from a rank computation and prices it at
  <= 2n+1;
- the resulting dichotomy: the sieve's non-factorisation is BOUNDED in every
  direction that has a spectrum and UNBOUNDED exactly in the nilpotent
  direction. That is a precise statement of why bounded-order spectral,
  moment, and transfer-matrix methods cannot see the Jacobsthal-type maximal
  gap.

## 3. PROOF / STATUS

PROVED: Result 1 (two lines, above); Result 2's structure (|T| = 2 forces
|O| <= 2 for n <= q'); the bound rank_n <= min(2^n, d1, d2).
SCRIPT-VERIFIED (`research/nontensor.py`, assertion-gated):
* Result 1 asserted at machines 11/13/17 over ALL bipartitions (3, 7, 15
  cuts), with rank(exposure) = 1 and rank(blocking) = 2 at every one;
* Result 2 asserted as measured == predicted at every (machine, n) row, plus
  rank <= 2n+1;
* Result 3's profiles computed from exact integer Gram matrices V V^T with
  rank taken over GF(p) at p = 2147483647 and 2147483629 (the two agree at
  every row, and each is a certified lower bound on the rational rank);
* F(M) recovered as the depth where v_n vanishes: 7, 11, 18, 25, 34 at
  machines 11/13/17/19/23, asserted against the known ladder.
MEASURED (not proved): that the peak rank keeps saturating the cut, hence the
sqrt(P) growth statement. Five machines; the cut cap d1 <= 400 is a compute
limit, not a mathematical one. Machine-23 run:
`research/data/nontensor_big.log` (527 s for the depth profile; depths 1-6
are marked "." there because the support exceeded the 6e6 build cap - the
binding cap at those depths is 2^n <= 64 anyway).

## 4. IMPLICATIONS

Inside the project - this is the round-22 spine, answered:
- NO FIXED-ARITY RULE can exist for the window/realizability content of (D):
  the joint information that must cross a gear cut at window depth grows with
  the machine. Constructor's independently measured truncation arity (3-point
  at 19/23, 4-point at 29) is the same phenomenon counted from the other
  side, and both say "growing".
- An ARITY-FREE generator is therefore the only surviving vehicle, and
  NILPOTENCY is exactly that: one equation about all orders at once. This
  entry says more - the growth is IN the nilpotent direction, so nilpotency
  is not merely a convenient formulation, it is where the content is.
- The merge law's fixed arity (Result 2) explains why the per-step ladder is
  mechanical while the within-machine bound is not: adding a gear is cheap,
  reading a machine's own joint structure is not.
- Practical: the rank profile is a new cheap diagnostic. rank_n = 0 is
  exactly F <= n, and the profile's peak location and height are computable
  from position laws alone, with no period scan beyond the sieve.

Outside: a sieve-theoretic instance where the failure of a local-to-global
(product) structure is quantified exactly as a tensor rank, with a proved
bounded case (depth 1) and a measured unbounded case (window depth).

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Jacobsthal / Ziller-Morack h_2 (F is the vanishing depth of the profile);
requirement (D) and the twin-prime route (the growth statement is what makes
fixed-arity relaxations provably insufficient - it does not by itself bound
anything); Wall V (this is its dimensional form). Open here: is
rank_n = min(2^n, d1, d2) exactly in a range of n, i.e. is the sector
GENERICALLY full? Measured peaks reach 35/35 at {5,7} but only 326/391 at
{17,23}; the deficit's law is unknown.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). Terms for the manager:
"Kronecker rank / operator Schmidt rank of a complement of a tensor product
projector"; "tensor rank lower bound via matricization sieve residues";
"entanglement of CRT product states arithmetic"; "nilpotency index of a
difference of Kronecker products"; "wheel sieve blocked pattern tensor
decomposition"; "communication complexity of coprimality across prime
factors" (the Schmidt rank across a cut is exactly the nondeterministic
communication rank of the window predicate, which may be the literature's
name for this object). Expected nearest art: Van Loan's Kronecker-product
survey for the frame; the delta to check is the sieve statements (rank 2 at
depth 1, the merge-cut 2n+1 law, the measured saturation).

# walk-path-parts - the walk from `q^2` decomposed: forced anchor phase, forbidden walk lengths, the character sub-torus, and the top gear's inertness

Branch W.a (round 38), research/proof/walk_path.md. Scripts research/anchor235/r38/pa_*.py.

## 1. WHAT IT IS

Plain words. Take a prime `q` and start at the number `q^2`, walking upward in columns
(a column `k` is the pair `6k-1, 6k+1`) until both members of a column are prime. Which
column blocks you is decided by which gears (primes `5..q`) divide `q^2 + 6i - 2` or
`q^2 + 6i`. Because `q^2` is a *square*, the gears cannot be in arbitrary phase: each gear
sees a quadratic residue, so it can act only on some of the offsets, and the smallest gear
of all is pinned completely. The consequences are exact: the second column of the walk is
always blocked by gear 5; the walk's *length* can never be `1 (mod 5)`; the length lies in
one of only six explicit 15-element sets modulo 35; and the top gear, which creates the
window, is never the smallest blocker of any column of its own walk except the first.

Precise. Let `q >= 5` be prime, `u_g = 6^{-1} mod g`, `k_0 = (q^2-1)/6`, and let `L` be the
least `i >= 1` with column `k_0 + i` unstruck by every gear `5..q`. Gear `g` strikes offset
`i` iff `i = (2-q^2) u_g` or `i = (-q^2) u_g (mod g)`.

* **(W.a-1) Forced anchor phase.** For `q >= 7`, gear 5's strike offsets are `{1, 4} mod 5`
  if `q = +-1 (mod 5)` and `{1, 3} mod 5` if `q = +-2 (mod 5)`. Offset 1 is always struck.
* **(W.a-2) Forbidden walk lengths.** Hence `L != 1 (mod 5)` for every `q >= 7`;
  `L mod 5 in {0,2,3}` or `{0,2,4}` by the class of `q mod 5`; and `L >= 2`.
* **(W.a-3) The fifteen-class law.** `L mod 35` lies in the 15-element set of offsets left
  open by gears 5 and 7, which is one of exactly six sets, fixed by `q^2 mod 35`.
* **(W.a-4) The offset character law.** Whether gear `g` can strike offset `i` **at all** is
  independent of `q`: it requires `-6i` or `2-6i` to be a quadratic residue mod `g`. The mean
  strike depth at offset `i` is therefore a fixed arithmetic function
  `lambda(i) = sum_g 2 chi_g(i)/(g-1)` of the offset alone.
* **(W.a-5) The top gear is inert on its own walk.** `q` is the smallest striker of no column
  of the walk except offset 0 - including at `q = 53`, where `q` does strike a second column.

## 2. WHY IT MIGHT BE NOVEL

The classical shadow of (W.a-4) is the elementary fact that the prime divisors of the values
of a quadratic polynomial lie in prescribed classes; that is named here and not claimed. What
is not classical is the use of it as a *shape* statement about the first-passage problem: the
depth profile along the walk from a square is a fixed arithmetic function of the offset, the
same for every `q`, and the landing distribution follows it (0 landings on the eight
highest-`lambda` offsets of `1..79` against 500 of 2,260 on the eight lowest). (W.a-1) to
(W.a-3) are statements about the *length* of a first-passage walk being confined to explicit
residue classes; the project's record (anchor-235 sections 9c-9d, docs/novel/walk-tooth-frame)
gives the hit law and chain law for a general start and says nothing about a square start.
(W.a-5) strictly strengthens the recorded rule W1/N1 ("the top gear strikes its own walk once")
and, unlike it, has no exception.

## 3. PROOF

* (W.a-1) PROVED, one line: `q^2 = 1` or `4 (mod 5)`; the two targets at offset 1 are `-6 = 4`
  and `-4 = 1 (mod 5)`, so one of them is `q^2` whichever square it is. The class sets follow
  from `i = (2-q^2)u_5` and `i = -q^2 u_5` with `u_5 = 1`. SCRIPT-VERIFIED at every prime
  `q = 7..19997` (2,259 paths, 0 exceptions), `research/anchor235/r38/pa_path.py`.
* (W.a-2) PROVED, immediate from (W.a-1). SCRIPT-VERIFIED, 0 exceptions in 2,259 paths.
* (W.a-3) PROVED by the same computation at gears 5 and 7 (the six residue classes of `q^2`
  mod 35 are the quadratic residues). SCRIPT-VERIFIED at every prime `q = 11..19997`
  (2,258 primes, 0 exceptions), `pa_class.py`.
* (W.a-4) PROVED (the necessity half is Euler's criterion). The quantitative half - measured
  mean depth against `lambda(i)` - is MEASURED: Spearman 0.9985 over offsets `1..79`, values
  within 2%, 2,260 primes, `pa_path.py`.
* (W.a-5) MEASURED, 0 exceptions in 2,260 paths (`pa_path.py`). Mechanism (a proof sketch, not
  written out): for `q` to be the smallest striker of a later column, that column's member is
  `q m` with `m` free of primes below `q`, so `m >= q+2`, which is another tooth arc away.

Two further measured items from the same branch: the walk from `q^2` is longer than walks from
the top gear's other teeth in the same window (median percentile 0.600 over 412 machines, mean
ratio 1.243, and 0 of 412 square starts have `L = 1` against 4.80% of 57,125 tooth starts); and
the quadratic-residue restriction costs the machine 13.4% of the maximum walk length reachable
by re-phasing a single gear (39.18 free against 33.93 by a real residue of `q`, over 13,861
gear cells).

## 4. IMPLICATIONS

Inside the project: the walk-frame rules W1-W4 (docs/novel/walk-tooth-frame) describe the walk
from the top gear's side; these describe it from the old gears' side and give the first
*forbidden values* for the walk length. (W.a-5) removes the top gear from the shaping question
entirely, so the walk from `q^2` is a statement about the machine `{5..q-}` only. The character
sub-torus is the first measured respect in which the real machine is provably weaker than its
counterfactual tooth family, and it is weaker in the direction the conjecture needs (shorter
maximal walks) - a lead for the counterfactual-family work, not a bound.

Outside: nothing here bounds `L`. The branch also measures why: the minimum set of gears needed
to certify the blocked run has median 9 and maximum 43 over 667 paths, and 88.3% of paths
contain a column blocked only by a gear above `sqrt(q)`, so no bounded-order or bounded-machine
argument reproduces the walk.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

The statement `L < d = 2u_q` (the top gear strikes its own walk exactly once) is a
twin-Bertrand-strength statement at scale `q/3` and stays open; these rules constrain `L`'s
residue but not its size. Nothing here bears on Ziller-Morack Conjecture 6 or the paired
Jacobsthal ladder.

## 6. PRIOR-ART CHECK

NOT YET CHECKED (no web access this lane). Screened against docs/novel/README.md (all 60+
index entries; nearest are walk-tooth-frame and anchor-235-layer-laws), docs/proofs/01-19, and
docs/proof-search/anchor-235.md sections 9c-9g: no entry states any of (W.a-1) to (W.a-5).
Terms a checker should run: "first prime gap after a square", "least offset i with n^2+6i+-1
both prime", "quadratic residue restriction on prime divisors of x^2+c along a walk",
"Jacobsthal function at a square", "twin prime after p^2".

# literal-cap - literal chains have at most 6 members, every gear, forever

## 1. WHAT IT IS

Plain language: when a new prime ("gear") q' >= 11 is merged into the twin-sieve machine, it
kills some of the surviving slots. A "literal chain" is a run of consecutive kills built purely
from the gear's own two teeth - each step of the run moves by the exact tooth spacing, with no
padding. The finding is that such a run can never have more than 6 members, no matter how large
q' is, and that the exact maximum is decided by q' mod 210 alone: of the 48 invertible residue
classes mod 210, 24 cap at 2 members, 4 cap at 3, 14 cap at 4, and exactly 6 classes
(q' = 37, 53, 83, 127, 157, 173 mod 210) attain the ceiling 6.

Definitions. Slot k = the pair (6k-1, 6k+1). Gear (prime) q >= 5 blocks slot k iff
k = +-u' (mod q) with u' = round(q/6) (equivalently k = +-(6^{-1}) mod q) - the gear's two
"teeth". The machine M_y runs gears 5..y; the corridor mod 35 is the 15-element exposed set
E = {0,2,3,5,7,10,12,17,18,23,25,28,30,32,33} of residues mod 35 left twin-eligible by gears 5
and 7 (15 = (5-2)(7-2)). In the merge step M -> M + q', a literal link connects two kills of
gear q' at spacing s = 2u or q'-s (mod q') - opposite teeth - with the letters alternating; a
literal chain is a maximal run of such links all of whose members lie in E mod 35 (they must:
every opening is exposed).

Precise form (as kernel-checked). Define the tooth walk

    wpos t s r ph i = (r + ((i + ph) / 2) * t + (if (i + ph) % 2 = 1 then s else 0)) % 35

with t = q' mod 35, s = sOf(q' mod 210) = (2u) mod 35, base residue r, phase ph in {0,1}.
Then for every c < 210 with gcd(c, 210) = 1, every r < 35 and both phases, the walk has NO run
of 7 consecutive members inside E (`no_run_seven`). Consequently any literal chain of walk
members that are all Exposed has length L <= 6 (`literal_chain_le_six`), with NO upper bound on
q'. The cap 6 is sharp: the classes admitting a 6-run are exactly
{37, 53, 83, 127, 157, 173} mod 210 (`cap_six_classes_sharp`).

Full cap spectrum over the 48 classes (computed exactly, script-verified):

    cap   2    3    4    6
    #cls  24   4    14   6

## 2. WHY IT MIGHT BE NOVEL

- It is a UNIFORM, PERMANENT bound: one finite computation (48 classes x 35 bases x 2 phases x
  7 steps) settles the behaviour of every prime gear forever. Most run-length statements in the
  literature (runs of quadratic residues, gaps between reduced residues) grow with the modulus;
  this one is constant.
- The bound is NOT the trivial pigeonhole/density fact it superficially resembles. The exposed
  set has density 15/35 = 3/7, and a generic (t, s) walk mod 35 is NOT capped at 6: over all
  1225 (t, s) pairs mod 35 the run spectrum is {2, 3, 4, 5, 6, 8, 10, 140} (formalist verdict
  5 - "cap <= 6 for ALL (t,s) pairs mod 35" is FALSE). The restriction to invertible classes
  mod 210 - i.e. the arithmetic linking t and s through u' = round(q'/6) - does real work. So
  the cap is a property of the coupled arithmetic of q', not of the exposed set alone.
- The classification is exact and sharp, not an O(-) bound: the cap value as a function of
  q' mod 210, with the attaining classes listed.
- Consequence inside the theory: it explains all realized fuel censuses (measured k_max
  2, 2, 3, 2, 4 against caps 2, 2, 4, 3, 4 - saturated at gears 17, 19, 31), forbids literal
  k = 5 at q' = 31, and forces any longer killed run to buy a padded link (a gap >= q' in M).

Classical shadow, stated honestly: that an AP with difference coprime to 35 cannot stay in a
proper subset of Z_35 forever is trivial equidistribution; that twin-admissible residues mod 35
number (5-2)(7-2) = 15 is the standard Hardy-Littlewood local count. Neither gives a uniform
finite cap for the two-tooth alternating walk, which is the content here.

## 3. PROOF

Status: KERNEL-CHECKED (the cap <= 6, the class reduction, and the sharpness set);
SCRIPT-VERIFIED (the full 48-class cap spectrum {2:24, 3:4, 4:14, 6:6} and the confirmation
against every prime q' <= 5000, 0 mismatches).

- Lean: `proofs/LiteralCap.lean` (round 13, 998 jobs, zero sorries):
  - `no_run_seven` - the finite check: for all c < 210 with gcd(c,210)=1, all r < 35, both
    phases, `run7 (c % 35) (sOf c) r ph = false`. Kernel `decide`; no `native_decide`
    anywhere in the ledger.
  - `s_eq` - the class reduction: `(2 * u) % 35 = sOf (q % 210)` under
    `6*u + 1 = q or 6*u = q + 1`, so the walk data depends on q' mod 210 only.
  - `literal_chain_le_six` - the cap: exposure of all L members forces L <= 6.
  - `cap_six_classes_sharp` - the six attaining classes are exactly {37,53,83,127,157,173}.
  - Axiom footprint: the standard three `[propext, Classical.choice, Quot.sound]` at most
    (per the formalist ledger convention; the period-scan style lemmas in this ledger run on
    `[propext]`/`[propext, Quot.sound]`).
- Scripts: `research/literal_cap_gap_d.py` (48-class table the Lean file was verified
  against before formalising) and `research/fuel_bound.py` (constructor R20's 48-class check;
  verification against every prime to 5000).
- Statement provenance: constructor R20 ("Literal chains have at most 6 members, for every
  gear, forever"), lateral definitions block (literal link = opposite-teeth spacing,
  alternation).

Scope caveat (recorded in the corpus, X14 and lateral withdrawal 9): the cap covers LITERAL
chains only. Padded runs (spacing 0 mod q', cost >= q' in M) escape it; their count is bounded
only by budget arithmetic + the onset gate. Any statement "killed runs are bounded by 6" would
be FALSE.

## 4. IMPLICATIONS

Inside the project:
- Part (B) of the five-part factorisation of the route - literal span <= 5 letters, span
  < (10/3) q' - is fully kernel-checked and universal; the sole open part (D) inherits a
  finite, pinned word list per step (<= 6 words, from q' mod 210 alone).
- The fuel bound k_max <= cap(q' mod 210) turns the empirical fuel census into arithmetic;
  the census falsification asserts (constructor O4) are sharp because the cap is sharp.
- Generalises: the same statement over every even gap d is the Polignac cap
  (docs/novel/polignac-cap.md), ceiling 12.

Outside: a clean, machine-verified example of a "wheel" statement that is genuinely about the
arithmetic of the incoming prime, not about the wheel's density - a counterexample generator
(the 1225-pair spectrum reaching 140) is part of the record. Relevant to anyone modelling
Eratosthenes-sieve merge dynamics (Holt-Rudd style cycle-of-gaps recursions) since it caps a
natural class of long-range events in the recursion.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- Twin prime conjecture, via this project's route: the cap is proven infrastructure for the
  tolerance/route factorisation ((A)/(B) closed, (D) open). It does not itself touch
  infinitude.
- Hardy-Littlewood k-tuple admissibility: the cap refines the local admissibility count
  (which only says |E| = 15) into a dynamic run-length statement the HL framework does not
  ask for.
- The word-list enumeration as a function of q' mod 210 (part (A)) is computed but not yet
  kernel-checked - the named open formalisation target 1 in the formalist ledger.

## 6. PRIOR-ART CHECK

Date: 2026-08-23. Engine: Claude WebSearch (web). Searches actually run:

1. "maximal run consecutive integers coprime residue classes admissible tuples mod 210
   classification" - nearest: OEIS A023193 (max size of admissible tuple in [0,n)),
   Ford-Green-Konyagin-Maynard-Tao large-gaps machinery, arXiv:2007.01808 (differences between
   consecutive numbers coprime to n). All about admissible-tuple SIZE or coprime GAPS, not
   run caps for a two-tooth walk classified by the prime's residue class.
2. "Jacobsthal function maximal gap reduced residues primorial g(210)" - Jacobsthal g(n)
   (Hagedorn's computations; Integers 18 (2018) #A26) is the closest classical function: max
   gap between consecutive totatives of a primorial. It bounds gaps in the WHEEL, not runs of
   a gear's kill-walk inside the twin corridor; and it grows with the primorial where this cap
   is constant.
3. ""twin prime" wheel modulo 35 "admissible residues" runs consecutive candidates blocked
   prime sieve pattern" - standard admissible-residue counts (nu(q) = q-2 for q >= 5, e.g.
   Tao's 254A Notes 4); no run-length classification found.
4. "longest run arithmetic progression terms coprime to n avoiding residue classes pigeonhole
   bound" - nearest: runs of consecutive quadratic residues/non-residues (Burgess
   O(p^{1/4} log^{3/2} p)) - a different set, and a growing bound.
5. ""37, 53, 83, 127, 157, 173" ... "mod 210" prime pattern" - the six classes surface only in
   unrelated contexts (Goldbach partitions of 210; the AP 7+210k). No published list of these
   classes as run-cap maximisers.
6. "Holt Rudd "Eratosthenes sieve" cycle of gaps primorial constellations populations twin gap
   patterns" - Holt & Rudd, "Eratosthenes sieve and the gaps between primes"
   (arXiv:1408.6002) and "Combinatorics of the gaps between primes" (arXiv:1510.00743): the
   closest published FRAME (recursion on the cycle of gaps G(p#), populations of
   constellations). Their results are population/enumeration dynamics of constellations; no
   run-length cap for a single incoming prime's kill pattern, and no classification by
   q' mod 210, appears there.
7. "Gallagher runs of sieve survivors wheel factorization longest run ..." - Gallagher's
   larger sieve and wheel-factorization implementations; nothing on capped kill-runs.

Nearest published results overall: Jacobsthal's function (gap structure of the primorial
wheel); Montgomery & Vaughan, "On the distribution of reduced residues" (Annals 1986 -
moments of totative gaps); Burgess runs of quadratic residues; Holt-Rudd cycle-of-gaps
constellation dynamics; Hardy-Littlewood local admissibility counts.

VERDICT: NOVEL AS FAR AS SEARCHED - for the statement "the maximal literal (two-tooth
alternating) chain is <= 6 uniformly in q', with the exact cap a function of q' mod 210,
computed over all 48 classes and sharp at six named classes". The AMBIENT facts used (15
admissible residues mod 35 = (5-2)(7-2); equidistribution of coprime-difference APs mod 35)
are KNOWN and claimed by nobody here as new. No published classification of residue classes
mod 210 controlling maximal runs was found in any search.

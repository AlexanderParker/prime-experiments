# tooth-sharing-pinning - twin gear pairs pin their four double-kill CRT classes in closed form, including the twin-product slot

Status: PROVED (elementary proof below; script-verified 60/60 twin pairs to
2000; not kernel-checked). Established round 1 (lateral, "twin pin"). Prior-art
check: 2026-08-23 (section 6).

## 1. WHAT IT IS

Plain language. In the project's slot frame (slot k = the pair (6k-1, 6k+1);
gear q blocks the two residue classes k = +-(6^{-1} mod q) mod q), a twin prime
pair (p, p+2) viewed as two gears is special: both gears have the SAME numeric
tooth offset u' = (p+1)/6. Because of that, the four residue classes mod
P = p(p+2) where BOTH gears kill in the same slot are not scattered - they are
pinned in closed form: {+u', -u', +u'(p+1), -u'(p+1)} mod P. And the mixed
class is not just any slot: at k = u'(p+1) the slot's lower member is
6u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2) EXACTLY - the product of the twin pair
itself. Consequence: every twin gear pair wastes at least two kills per window
on slots that are already dead (its own slot and its product slot),
deterministically, at every level.

Precise form. Let (p, p+2) be a twin prime pair, p >= 5, and set
u' = (p+1)/6 (an integer since p = 5 mod 6) and P = p(p+2).

(a) SHARED TOOTH. Gear p blocks k = +-u' (mod p) and gear p+2 blocks
    k = +-u' (mod p+2): the two gears' teeth carry the same numeric value u'
    (u' = 6^{-1} mod p and u' = -6^{-1} mod (p+2)).

(b) PINNED DOUBLE-KILL CLASSES. The slots where both gears block are exactly
    the four CRT classes

        k = +u', -u', +u'(p+1), -u'(p+1)   (mod P).

    The equal-sign classes +-u' are SPLIT kills: at k = u' (mod P) gear p
    kills the lower member and gear p+2 kills the upper member (at k = u'
    itself these are p and p+2 - the pair kills its own slot); the mirror
    class -u' is the reflection.

(c) TWIN-PRODUCT SLOT. The mixed classes +-u'(p+1) are SAME-MEMBER kills, and
    the base point is exact, not just a congruence:

        6 u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2).

    At k = u'(p+1) both gears kill the SAME member, the semiprime p(p+2),
    while the other member p(p+2) + 2 is untouched by the pair - two kills
    spent where one would do.

(d) WASTE COROLLARY. Since u' <= (y+1)/6 and p(p+2) < y^2 for every twin pair
    with p+2 <= y, the classes +-u' and the product slot all appear inside the
    working window at every level: every twin gear pair donates >= 2
    deterministic in-window wasted kills.

Generalisation (constructor R6, roots-of-unity law): for ANY gear pair (q, q'),
slot k is hit by both iff 36 k^2 = 1 (mod qq'); the nontrivial root pins the
cross-member coincidences at the semiprime slots. The twin case is the one
where the root has the closed form p+1 and the tooth values coincide.

## 2. WHY IT MIGHT BE NOVEL

* The statement is uniform and closed-form in the pair: one formula
  {+-u', +-u'(p+1)} for every twin pair forever, in the slot frame where the
  twin-prime machine actually operates, with the mixed class identified as a
  concrete integer landmark (the twin-product slot), not merely a residue.
* The self-referential content - a twin pair's own sieve action deterministically
  wastes kills on its own slot and its own product - is a "self-blocking"
  statement about the twin constellation we found nowhere in the literature as
  a published identity.
* Honest classical shadow, stated up front: the algebraic core is textbook.
  The square roots of 1 mod a semiprime qq' are {+-1, +-r} with r built by CRT,
  and for the modulus p(p+2) the nontrivial root is p+1 because
  (p+1)^2 - 1 = p(p+2) - a one-line identity. Twin primes as the modulus
  p(p+2) also has a classical pedigree (Clement's 1949 congruence
  characterises twin primes mod p(p+2)). What we did NOT find published: the
  shared-tooth fact in any wheel/slot frame, the four double-kill classes as a
  closed form, or the wasted-kill accounting drawn from it. The finding is the
  packaging of the classical identity into an exact sieve-waste law - the delta
  is real but modest, and this document says so.

## 3. PROOF

Elementary, self-contained:

1. (Shared tooth.) 6u' = p+1 = 1 (mod p), so u' = 6^{-1} = c_p (mod p) and the
   teeth of gear p are +-u' (mod p). Also 6u' = p+1 = -1 (mod p+2), so
   u' = -6^{-1} = -c_{p+2} (mod p+2) and the teeth of gear p+2 are likewise
   +-u' (mod p+2). This proves (a). (Member sides: 6u' - 1 = p and
   6u' + 1 = p+2, so at k = +u' gear p kills the lower member and gear p+2 the
   upper; signs flip at -u'.)
2. (CRT.) "Both gears block k" means k = eps*u' (mod p) and k = del*u'
   (mod p+2) with eps, del in {+1, -1}: four sign patterns, hence exactly four
   classes mod P. Equal signs give k = +-u' (mod P). For mixed signs note
   p+1 = +1 (mod p) and p+1 = -1 (mod p+2), so u'(p+1) = +u' (mod p) and
   = -u' (mod p+2): the mixed classes are +-u'(p+1) (mod P). This proves (b).
3. (Product slot.) 6u' = p+1 gives 6u'(p+1) - 1 = (p+1)^2 - 1 = p(p+2). Both
   p and p+2 divide the lower member of slot k = u'(p+1); the upper member
   6k+1 = (p+1)^2 + 1 = p(p+2) + 2 is coprime to both (it is 2 mod p and
   2 mod p+2). This proves (c), and (d) follows by the size bounds stated.

Verification: research/tooth_sharing.py - all four classes checked against
brute-force double-kill enumeration for all 60 twin pairs with p < 2000, 60/60
exact. Not kernel-checked (listed as a proposed Lean handoff in
docs/proof-search/lateral.md, status untracked).

Sources: docs/proof-search/lateral.md, Established result 1 ("TWIN PIN", r1);
generalisation docs/proof-search/constructor.md R6.

## 4. IMPLICATIONS

Inside the project:

* Feeds the SHARING LAW (lateral r2): survivors per full period = prod(q-2)
  regardless of phases - sharing moves WHERE the waste lands, never HOW MANY
  survive; the sub-period expectation E[waste_shared - waste_indep] = 1 - 2R/P
  was confirmed pair-by-pair against this pinning.
* Honest negative already recorded (lateral refuted angle 2): the two
  guaranteed wasted kills land on already-decided slots (the self-block slot
  and the semiprime slot), so tooth-sharing COUNT alone gains only O(T(y)) per
  window against a needed ~K/log^2 - this identity does not close the
  recursion, and the project does not claim it does.
* The self-block census U(t) built on class +-u' is a term of the flagship
  identity P(t) = t + T(t) - B(t) + U(t) (lateral r5), which is exact per-slot
  at every tested scale.
* The g = 2 case is the unique gap class whose split-double supply is
  unconditionally in-window at every scale (lateral r3): b0 = 1 only at g = 2,
  the split representative x = u' <= K. The pinning is why.

Outside the project: an exact local description of how the twin constellation
interacts with its own sieve - the local (mod p, mod p+2) coincidences that the
Hardy-Littlewood singular series counts in aggregate, here pinned to named
integer addresses (in particular the twin-product landmark (p+1)^2 - 1). Mostly
of expository/structural value unless combined with a counting mechanism.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

* The twin prime conjecture, self-referentially: the machine's hostility to
  twins includes each existing twin pair wasting exactly the kills that would
  otherwise have hit fresh slots - a small deterministic pro-survivor bias,
  quantified by the sharing law as ~ +1.3 survivors per 10 twin pairs.
* Hardy-Littlewood local factors: the (p-1)/(p-2)-type factors aggregate
  exactly these double-hit coincidences; the pinning gives their positions,
  not just their density.
* Abandoned-but-not-refuted thread (lateral): the joint-necessity census -
  twin pairs jointly own a pseudo-twin at the product slot when p(p+2) + 2 is
  prime (p = 5, 149, 179, 239, ...) - never censused against generic pairs.
* T1 twin-dead-centre law (mechanic, round 10): every twin dead-centres the
  thinnest layer band above it - its product slot k = 6m^2 sits at offset
  T/2 exactly; the same product-slot landmark appearing in the descent
  frame.

## 6. PRIOR-ART CHECK (2026-08-23)

Searches actually run (engine: Claude WebSearch, 2026-08-23):

1. "\"p(p+2)\" twin prime modulus square root of unity sieve residue classes
   semiprime"
2. "twin primes sieve wheel \"6k-1\" \"6k+1\" both blocked residues overlap CRT
   twin pair product slot"
3. "Clement's theorem twin primes congruence modulo p(p+2) characterization"
4. "prime constellation sieve \"self-blocking\" OR \"wasted\" overlap twin prime
   pair removes own product residue class Hardy-Littlewood singular series
   local factor"

Nearest published results found:

* P. A. Clement, "Congruences for sets of primes", Amer. Math. Monthly 56
  (1949): n, n+2 are twin primes iff 4((n-1)! + 1) = -n (mod n(n+2)) - the
  classical use of the twin-product modulus p(p+2). DELTA: a Wilson-type
  primality criterion; it uses the modulus but states nothing about sieve
  residue coincidences, shared tooth offsets, or double-kill classes.
* Textbook CRT: the four square roots of 1 modulo a semiprime qq' are
  {+-1, +-r}, r = CRT(+1, -1); for qq' = p(p+2) the nontrivial root is p+1
  ((p+1)^2 - 1 = p(p+2)). Standard in any number theory text (and the basis of
  Fermat-style factoring). DELTA: this IS the algebraic core of parts (b)-(c);
  what is not in the literature is its transport to the 6k+-1 slot frame (the
  shared tooth value u' = round(p/6), the four DOUBLE-KILL classes as
  +-u', +-u'(p+1), the split-vs-same-member classification) and the wasted-kill
  corollary (d).
* Wheel/segmented twin-prime sieves (e.g. "Twin Primes Segmented Sieve of
  Zakiya", 2022; assorted 6k+-1 sieve papers on arXiv): use the 6k+-1 frame and
  blocked residues per prime, some noting the mutual exclusivity of p | 6x-1
  and p | 6x+1. DELTA: none states the twin-pair shared-tooth fact, the pinned
  four-class closed form, or the product-slot identity; these are algorithmic
  papers about enumeration, not structural identities.
* Hardy-Littlewood k-tuple literature (singular series local factors;
  admissibility of residue classes for twin pairs, e.g. "On twin prime
  distribution and associated biases", arXiv:2111.09053; Kevin Ford's sieve
  notes): the local factor at p counts exactly the residues killed by the pair
  constraint, and admissible-class analyses note that a class a mod q needs
  (a, q) = (a+2, q) = 1. DELTA: aggregate densities and admissibility counts,
  never the closed-form positions of the coincidence classes for the pair
  (p, p+2) as moduli, and no self-blocking/waste statement.
* Search 4 ("self-blocking" prime constellations) surfaced only a 2025
  arXiv heuristic preprint (2512.03288) using a "blocking prime" notion for
  p = 3 anti-correlations - unrelated in content (and not peer-reviewed
  prior art for this identity). No published closed-form CRT identity for
  twin-pair double kills was found in any search.

VERDICT: PARTIAL OVERLAP. The algebraic core (four square roots of unity mod
p(p+2), with (p+1)^2 = 1 mod p(p+2)) is classical/textbook, and twin primes as
the modulus p(p+2) is classical (Clement 1949). The delta - what was not found
anywhere - is the slot-frame formulation: both gears of a twin pair sharing the
tooth value u' = round(p/6), the four double-kill classes pinned as
{+-u', +-u'(p+1)} mod p(p+2) with the mixed class at the twin-product slot, and
the resulting ">= 2 deterministic in-window wasted kills per twin gear pair"
accounting. As a named identity in sieve-waste form it is NOVEL AS FAR AS
SEARCHED, but its proof content is elementary and any expert would derive it
on demand; it should be presented as a structural observation, not a deep
result.

# The class tree: umbrellas and shields as a branching structure

Session note, following `docs/umbrellas-and-shields.md` and `docs/pair-anatomy.md`. The tree the
umbrella picture suggests is a real object (the turn-law tree of `twin-prime-program.md` section
17e; the nested classes of section 1h), with these mechanics:

* **A node at level i** = a residue class mod P_i = a spot description under every umbrella so far.
* **Adding gear q** splits each node into q children - the q phases the new gear can present, one
  per lap of the old machine. The turn law kills exactly 2 of them (the children landing on the new
  gear's teeth, positions closed-form from the slip); q-2 survive, the shield-child (= 0 mod q)
  among them.
* **The tree is never extinct**: every node has >= q-2 >= 3 children (prod(q-2) >= 1, proven).
  The fully-shielded branch is k = 0 mod everything (slot 0, primorial multiples); generic twins
  are all-umbrella branches.

Demo - the diagonal branches k=3 and k=2:

    level 5:   class 3 mod 5     open        class 2 mod 5     open
    level 7:   class 3 mod 35    open        class 2 mod 35    open
    level 11:  class 3 mod 385   open        class 2 mod 385   DEAD - tooth of 11
    level 13:  class 3 mod 5005  open -> twin (17,19)
                                             ((11,13) killed by its own gears - self-blocking
                                              visible as tree pruning)

**Sound prune (proven, section 17e):** the level-i ancestor of slot k is k mod P_i <= k, so
discarding branches whose smallest representative exceeds the search bound never loses an answer
within the bound. Smallest-representative-first search of the tree is correct and complete - it is
the constructor, tree-shaped, and finds every twin.

**The obstruction, stated in tree terms:** following open branches controls openness, not position.
When a branch dies and the search steps to a sibling, the sibling class's smallest representative
can jump by primorial-scale amounts (the CRT-dial lesson: changing one level's residue moves the
representative by that level's idempotent). The tree provably always has open branches, and one
within F_k(y) of any point - but bounding the sideways distance to the nearest open branch inside
the window is Reduction A itself. Section 1h's sharp form: the tree's infinite paths are profinite
integers; only the paths that stay small are twins.

Every route in the programme is an attempt to bound the sideways step.

## Sufficient sub-sets: which gears the window actually needs

Script: `research/sufficient_subset.py`. Two one-line facts frame it:

1. **No subset covers the whole window.** Every gear's square is an in-window root kill on a
   candidate (q^2 = 1 mod 6 always - squares of gears are always right members), so dropping gear q
   falsely opens slot (q^2-1)/6 whenever q^2-2 is prime. Drop 13 from the y=13 set and slot
   28 = (167, 169) reports as a twin.
2. **But the window is graded** - the square-root tower localised: gears <= z are exact on slots
   whose members stay below nextprime(z)^2. The subset needed depends on where in the window you
   look, not on the window top.

Measured consequence for finding the FIRST twin above y (depth = isqrt(twin member + 2)):

     y    first twin   depth needed   gears kept/total
     41   (59,61)          7              2/11
    109   (137,139)       11              3/27
    197   (227,229)       15              4/43
    389   (419,421)       20              6/75

Needed depth averages 0.42*sqrt(6y); the kept fraction collapses toward zero. At the bottom edge
(members < 25) the 6-cycle alone certifies - (17,19) needs zero gears >= 5. Caveat, honestly: "the
first twin sits close above y" is an empirical input (corpus 12a: within 169 of y for all
y <= 3163); proving that closeness would be stronger than Reduction A.

Half-winding: the mirror fixes a subset's behaviour in half its primorial - true, but the mismatch
under attack is e^y against y^2, so the factor 2 is conceptual rather than asymptotic.

## The event horizon and the layer law (scripts: research/event_horizon.py, research/layer_ledger.py)

**Event-horizon theorem** (two lines, verified y = 13..79): any composite member strictly inside
(y, y^2) has a prime factor <= sqrt(M) < y, so the top gear is never the root cause of an interior
kill - gears STRICTLY BELOW y decide the open interior exactly. The top gear's whole unique
contribution is the boundary: its self-pair at the bottom edge and its square at the horizon,
which false-positives precisely when y^2 - 2 is prime (167, 359, 839, 1367, 1847, 2207, 3719,
5039 across the tested range). The exclusion works exactly once per window: the second gear's
square lies strictly inside.

Side fact: the primorial-scale unwind does NOT always yield twins - nudge home 595 of the
{5,7,11,13} machine is (3569, 3571) with 3569 = 43*83. Openness beyond the horizon is not twinhood.

**Layer law** (verified for the nine layers 13->17 .. 43->47): one layer = one prime retiring into
the working set, horizon advancing y^2 -> y'^2. The newly activated gear's entire novel workload is

    1. retro-closing the old horizon square y^2 (owed iff y^2-2 prime), and
    2. the slots y*c for primes c in (y, y'^2/y) - one to three explicit numbers per layer
       (Bertrand: y'^2/y < 4y) - each owed iff its partner member is prime.

Everything else in the fresh band is closed by the old gears. Seven of nine tested layers owe
nothing in-band at all; the exceptions are 221 = 13*17 (beside prime 223) and 437 = 19*23 (beside
prime 439). A layer's new content is a short explicit list of semiprime slots, enumerable in
advance - the tower's complexity lives in the number of layers, never inside one.

## The exact minimal subset (script: research/minimal_subset.py)

Necessity law: gear q is needed iff one of its root kills pairs with a PRIME partner in the window -
a pseudo-twin like (209,211) = (11*19, prime) that only q can unmask. Verified exact for y = 13..59:

    y   minimal set               dropped
    13  {5,7,13}                  11
    17  {5,7,11,13}               17
    23  {5,...,19}                23
    31  {5,...,29}                31
    41  {5,...,37}                41
    47  {5,...,37,43,47}          41
    59  {5,...,47}                53, 59

The minimal set is all gears minus the newest one or two, and droppability is transient (11 returns
at y=17 when the window reaches (209,211); 41 survives dropped through y=47). Unification: "q is
necessary" = "q owns a lone-killer fragile slot in the window" - the one-away census of
docs/band-attribution.md and the minimal-subset question are the same object.

## Downward exclusion in a fixed window (the 13-set worked example)

Fixed window of the 13-set (members to 169), excluding downward: after the top gear (horizon
theorem), gear 11 also drops - {5,7} alone reproduce every twin in the window, recovering (11,13)
itself, with only the conceded horizon slot (167,169) false. 11's five window kills, accounted:
11 = real twin; 55, 77 caught on the member side (by 5, by 7); 121, 143 caught on the partner side
(119 = 7*17, 145 = 5*29).

The mechanism is two-tier: STRUCTURAL below 11^2 = 121 (the horizon theorem at 11's own scale) and
PARTNER-LUCK in the band (121, 169] - two root kills there, both happening to sit beside
composites. The luck ends at the 17-window when (209,211) arrives.

The floor: gear 7 is indispensable - (47,49) and (77,79) each pair a 7-root-kill with a prime. The
fixed-13-window minimal chain bottoms out at {5,7}.

## Downward exclusion across windows: the square gate

Hypothesis tested: exclusion iterates down until the first gear with a coprime in the window.
Refuted as stated - the stopper is virtually always the gear's own SQUARE: descending exclusion
halts at the first q with q^2 - 2 prime, because a gear's square is its first root kill
(q^2 < q*r), met before any coprime. Results (horizon slot conceded):

    window y   dropped      stop gear   blocking pseudo-twin
    13         13, 11          7        (47, 49)
    17/19      17 (,19)       13        (167, 169)
    23/29      23 (,29)       19        (359, 361)
    31/37      31 (,37)       29        (839, 841)
    41/43      41 (,43)       37        (1367, 1369)
    47..59     top 1-2        43/47     (1847,1849)/(2207,2209)

The governing sequence is the primality of q^2 - 2 down the gear list (prime at 5, 7, 13, 19, 29,
37, 43, 47; composite at 11, 17, 23, 31, 41, 53, 59 in range). Exclusion depth = the run of
consecutive composites from the top - one or two gears except the 13-window's run to {5,7}.
Gears with q^2 - 2 prime are permanent floor-setters, each owning an eternal square pseudo-twin.

Where the coprime instinct still lives: a coprime stopper is possible in principle (11's
209 = 11*19 beside prime 211), but 13's square gate (167,169) blocks the descent before 11 is
reached in every window from 17 up. Coprime stoppers need a run of square-luck directly above
them, and the q^2-2 primes are too dense for that in this range.

## The coprime census (script: research/coprime_census.py)

The second killer category after squares, catalogued for the 23-window:

    gear   coprimes  pseudo-twins  in-set/out-set
    23     0 (horizon)   -             -
    19     1             1           1/0     <- first coprime = top-pair product 19*23 = 437
    17     4             3           2/2
    13     6             4           3/3
    11     10            6           6/4
    7      17            10          5/12
    5      24            18          6/18

Laws: coprimes appear exactly one step below the top (horizon theorem in coprime language); counts
fan downward like pi(y^2/q) - pi(q); in-set coprimes are the crossed teeth of the pair machines
(143, 221, 437); the pseudo-twin fraction rises as q falls - gear 5's first six coprimes
(35,55,65,85,95,115) all sit beside primes (37,53,67,83,97,113). The low gears' coprimes are the
machine's densest source of fragile slots.

## Territory map of the 13-set (window < 169)

Per gear: shadow zone (< q^2, always covered by smaller gears), then the root-kill territory
[q^2, 169). PT = pseudo-twin (partner prime, the deciding kills).

    gear 5:  shadow < 25,  territory 9 kills:
      25(s4,23 PT) 35(s6,37 PT) 55(s9,53 PT) 65(s11,67 PT) 85(s14,83 PT)
      95(s16,97 PT) 115(s19,113 PT) 145(s24,143 comp) 155(s26,157 PT)
    gear 7:  shadow < 49,  territory 6 kills:
      49(s8,47 PT) 77(s13,79 PT) 91(s15,89 PT) 119(s20,121 comp) 133(s22,131 PT) 161(s27,163 PT)
    gear 11: shadow < 121, territory 2 kills:
      121(s20,119 comp) 143(s24,145 comp)
    gear 13: shadow < 169, territory 0 interior kills; square 169(s28,167 PT) at the horizon.

Readings: territory sizes collapse upward (9, 6, 2, 0) - the low gears own the window and the top
gear's territory is wholly beyond the horizon. Gear 11's droppability is visible as a map fact:
both its kills land on slots whose partner is another gear's kill (slot 20 = 7's coprime + 11's
square; slot 24 = 5's coprime + 11's coprime) - the only territory overlaps in the map, and
exactly the crossed double-kills of the pair anatomy. Pseudo-twin density at the bottom is brutal:
gear 5 has 8 of 9 kills beside primes, gear 7 has 5 of 6 - the three composite partners in the
whole map (119, 121, 145) are precisely the mutual-coverage slots.

## The overlap map and the composite root law (13-window)

Full degree census of slots 2..28: 9 twins (degree 0), 10 fragile (degree 1 - the deciding
pseudo-twin slots), 7 double (5 pair products + 2 crossed: slot 2 = the twin gears' own pair
11:L+13:R, slot 20 = the 7x11 cross 119|121), 1 hub (slot 24 = (143,145): the 11x13 product plus
gear 5 on the partner - a pair-lattice act landing inside a lower gear's lattice).

**Composite root law (verified):** every squarefree product of set gears acts unshadowed exactly
once per window - at its own value - if it fits at all. Same-member joint hits of a pair are
qr*j with j = +-1 mod 6; j = 5 already overflows the window (5*35 = 175 > 169), so each pair
scores exactly one product coincidence (35, 55, 65, 77, 91, 143 - six pairs, six products, no
more), and no triple product fits (385 > 169), so zero same-member triples. Beyond its product, a
coprime's multiples carry a smaller cofactor and fall into that gear's lattice - the recursion
"lower lattice is just primes and coprimes" made exact.

Near-balance observed: 9 twins vs 10 fragile slots - the window's thin margin between real twins
and pseudo-twins, with the forced overlap slots as the remainder.

## Pinpointing twins in the umbrella stack (research/umbrella_tools.py)

Inside a window, a twin IS a slot whose joint umbrella exists over the certifying set (graded
depth: gears <= sqrt(member)). Umbrella-jumping - next joint umbrella, read its interval in closed
form, hop past - pinpointed all 55 twins of the 47-window, every one verified prime, with the six
prime quadruplets of the range appearing automatically as width-2 umbrellas ((101..109),
(191..199), (821..829), (1481..1489), (1871..1879), (2081..2089)) - the points-and-dominoes law
operating live.

Each twin carries a stack certificate: per-gear rooms, minima = the binding gears. Slot 23 =
(137,139): gear 5 room right 0, gear 7 room left 0, gear 11 room right 0 - the twin pinched to
width 1 from three directions. Twins sit in needle's eyes, and the certificate names the needles.

## Umbrella-stack pinpointing at scale (windows to y = 2003)

    y      window   twins   quads  max stride  stride/window
    101     1700     202     10       35         2.1%
    199     6600     574     20       83         1.3%
    499    41500    2557     56      154         0.37%
    997   165668    8087    161      242         0.15%
    2003  668668   26870    460      252         0.04%

All twins verified prime; 26,870 twins to ~4e6 generated from umbrella arithmetic alone in 3.2s.
The stride/window ratio collapses by two orders of magnitude across the range - Reduction A's
slack measured live. Quadruplet share (width-2 umbrellas) holds near 1.7% throughout.

**Bug caught and fixed en route** (recorded per the build-and-test discipline): computing a joint
umbrella with the certifying set of its first slot and extending rightward claims slots where the
tower has activated a NEW gear inside the interval (a square crossing mid-umbrella) - one false
twin per large window, exposed by full verification. The fix judges every slot at its own graded
depth; the failure mode is the horizon law in miniature.

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

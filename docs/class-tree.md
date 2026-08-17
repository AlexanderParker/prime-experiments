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

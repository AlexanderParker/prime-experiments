# The network of walks (owner, 2026-09-13)

Built as described: breadth-first from a start node (a gear pair), one child per anchor option
the rule under test allows, children unique to their path (a tree), a child may not be the node
just left, a node stops expanding when it reaches the destination. Script
research/stack/r8/walk_network.py (`uv run python walk_network.py q depth [nodecap]`).

Setup (as in anchors_walk.md): a step from column n to anchor a flips about (n + a + 2)/2 and
carries the gears dividing n + a + 2; the certified set after the step is (C intersect carried)
union K(a), what the walk kept plus what the landing place is known open to. Anchor families:
gear pairs and home, squares (K = {g}), blind classes (K = {5, 7}), caustic runs (K = {h});
when several families give the same column their K sets merge. Destination: a node in the
window (q, q^2] certified for every gear. Rules compared: keep (the flip must carry every gear
currently certified), carry1 (the flip must carry at least one gear), free (any anchor).

## Results

Machine 13 (gears 5, 7, 11, 13; 38 anchors; 9 window twins; depth 4):

| rule | anchors | nodes expanded | destination paths | distinct columns | twins reached | first depth |
|---|---|---|---|---|---|---|
| keep | any set | 2-3 | 0 | 0 | 0 of 9 | - |
| carry1 | pairs only | 2 | 0 | 0 | 0 of 9 | - |
| carry1 | + squares | 83,768 | 8,817 | 5 | 5 of 9 | 1 |
| carry1 | + blind | 2 | 1 | 1 | 1 of 9 | 1 |
| carry1 | + caustic | 533 | 175 | 3 | 3 of 9 | 1 |
| carry1 | all | 97,107 | 9,885 | 5 | 5 of 9 | 1 |
| free | + squares | 69,618 (depth 3) | 6,616 | 5 | 5 of 9 | 1 |

Machine 31 (9 gears; 188 anchors; 30 window twins; depth 3, node cap 300,000): keep 0 under
every anchor set; carry1 reaches 5 of 30 with squares (3,163 nodes), 3 of 30 with blind
(120,336 nodes), 5 of 30 with caustic (209,692 nodes), 5 of 30 with all (3,000 nodes); free
explodes at the cap without reaching more.

Shortest paths found at 13 (carry1, all anchors): 29: (11, 13) -> 29; 41: (5, 7) -> 23 -> 41;
59: (5, 7) -> 83 -> home -> 59; 137: (11, 13) -> 17 -> 71 -> 137; 149: (5, 7) -> 149.
At 31: 179: (5, 7) -> 179; 191: (5, 7) -> 191; 59: (5, 7) -> 149 -> 59; 197: (5, 7) -> 131
-> 197; 71: (5, 7) -> 149 -> 317 -> 71.

## Which rules worked, and why

- keep never works beyond machine 7. The start pair is already certified for all gears but two,
  and a flip that carries all of them needs an axis divisible by their product; no anchor sits
  there. Knowledge cannot be carried whole; it is dropped and re-acquired at landing places.
- carry1 is the productive rule: every destination found is reached by dropping most of the
  certified set at each step and picking it up again from the anchors' known sets.
- free finds nothing carry1 does not and expands ten to twenty times as many nodes.
- Caustic anchors are the cheapest per destination (533 nodes for 3 twins at 13 against 83,768
  for 5 with squares) because one caustic column is known open to many gears at once.
- Every destination's final step lands on an anchor whose own merged known set already covers
  every gear, or all but the one or two gears the flip carries. The walk supplies at most the
  last one or two gears; the landing place supplies the rest.

## What the network located

Columns in the window whose merged known set covers every gear by the families alone, with no
walk at all:

| machine | window twins | such columns | which |
|---|---|---|---|
| 7 | 4 | 4 | 11, 17, 29, 41 |
| 11 | 7 | 2 | 29, 41 |
| 13 | 9 | 3 | 29, 41, 149 |
| 17 | 15 | 4 | 29, 41, 149, 179 |
| 19, 23 | 17, 21 | 3 | 29, 41, 179 |
| 29 to 97 | 28 to 187 | 1 | 179 (after 13^2, offset +2 columns) |
| 101 to 113 | 201 to 234 | 2 | 179; 9419 (after 97^2, offset +2) |
| 211 | 626 | 1 | 9419 |
| 401 | 1789 | 1 | 143651 (after 379^2, offset +2) |

The recurring shape is the column two after a square, the numbers (g^2 + 10, g^2 + 12): 179 =
13^2 + 10, 9419 = 97^2 + 10, 143651 = 379^2 + 10. It collects gear g from the square family,
5 and 7 from a blind class of some other square, and every other gear from the caustic runs
after g^2. Honest reading: for a gear h above g, "before h's first strike after g^2" means
"no multiple of h in the column", so the caustic knowledge for those gears is the divisibility
check in other words, and the column is known open to every gear exactly when its two members
are checked prime gear by gear. The families package the checks; they do not replace them. The
one closed-form part is the offset: the candidate sits at +2 columns after the square, where 5
and 7 are excluded by class and g by distance.

## Verdict

The network answers the efficiency question: carry1 with caustic anchors is the efficient rule,
keep is impossible, free is waste. It also shows what a completed walk is: the last step lands
on a place already known open to everything, so the walk certifies rather than locates, and
the located places are the (g^2 + 10, g^2 + 12) columns when they happen to be twins (g = 7,
13, 97, 379 in range). No rule was found that produces a destination the anchors did not
already contain.

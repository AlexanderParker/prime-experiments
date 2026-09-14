# Loop: algorithms that carry residues by mirrors to an open flip in the window

Owner, 2026-09-15: the mirror action is the only provable way to bring residues along the number
line; construct an algorithm that picks up all the required residues along a walk to guarantee
an open flip in the window; define algorithms around these principles until one finds a twin in
the window without secret knowledge about the opening.

Rules of the loop: every algorithm is stated in the machine's terms (origin, mirrors, periods,
directions); what information it uses is stated; it is run at every machine 11 to 2000 at least;
the result is the count of machines where it lands on a twin inside the window, with where it
fails; no counting arguments; one entry per algorithm, kept whether it works or not.

Standing facts the algorithms must respect (proved):
- A walk from home on mirrors M_1, M_2, ... lands at -1 + 2 (integer combination of the M_i), so
  the landing is -1 modulo the gcd of the mirrors: residue -1 is carried exactly for the gears
  dividing every mirror. To land in the window the gcd must be at most q^2 / 2.
- A gear's phase at a column is the column mod the gear; it strikes the slot when the phase is 0
  or gear - 2; a step about a mirror holding the gear moves its phase by 0, any other step by
  the mirror size mod the gear, with the step's sign.

## Entries

### 1. The pick-up walk, greedy (2026-09-15)

Visit the gears above the base in order; at gear g the mirror is {base, g}; choose the first
(periods k in 1..K, direction) such that every gear visited so far, g itself and the base are
off their teeth at the new column, and the column stays in [0, q^2]. Uses the walk's own phases,
no primality. research/stack/r8/pickup_walk.py, results_pickup_walk.txt, machines 11 to 2000.
- Descending from q: K = 1 stuck at all 299; K = 2 twin at 2, stuck 297; K = 3 twin 3; K = 5 twin
  10, stuck 289 (q = 31 reaches 227). Stuck typically at a gear in the middle (101: at 37; 499:
  at 281; 1999: at 281 or 1373).
- Ascending from the first gear above the base: stuck at all 299 for every K, at 7, 11, 13, 17
  or 23: the small gears leave few open phases and each step offers only 2K columns.
Verdict: greedy with 2K choices per step cannot hold every visited gear off its teeth; the
constraint tightens by one gear per step while the choices stay at 2K.

### 2. The pick-up walk, with backtracking (2026-09-15)

Same invariant, depth-first search over (periods 1..3, direction), node budget 200000, machines
11 to 600 (research/stack/r8/pickup_walk_search.py). Not a rule: a feasibility check.
- Descending: a walk exists at 68 of 105 machines (q = 31: 227 by up, up, down, up, up, down 2
  periods, ...; q = 101: 5099; q = 199: 20639). None found at 11, 13, 17, 23, 61 to 89, 421, 431,
  ... within the budget; the search dies fast there (24 to 144 nodes): no branch survives the
  small gears once they are all visited.
- Ascending: none at any machine. Visiting the small gears first leaves too few open phases
  for the later steps' 2K columns.
Verdict: the invariant "every visited gear off its teeth" can be kept along a walk at two thirds
of the machines with a search, and at none by a fixed rule so far. The block is the same joint
condition as the final flip, spread over the steps instead of concentrated in one.

### 3. Pick-up walks with fixed period rules from slip inverses (2026-09-15)

Rule C: descending; at gear g's step the periods k are chosen so the previous gear's phase lands
at the centre of its open arc (k = (centre - phase) times the slip inverse mod the previous
gear). Rule D: the previous two gears centred at once (k from the CRT of the two). Up only, or
alternating. No search, no primality (research/stack/r8/pickup_walk_rules.py).
- Landing inside the window at 1 of 299 for C up, C alt, D up; 0 for D alt. Twin 0.
- Why: centring one gear costs k up to that gear, so the move is up to 2 P g g' where g' is the
  previous gear; with both gears large that exceeds q^2 (q = 499: landing 9037979, ten times
  past the window, though open to every gear of the machine; q = 1999: 116353859).
- Exact limit: a step about {base, g} can place another gear g' at a chosen phase only if
  2 P g g' fits in the window, i.e. g g' at most q^2 / (2P), which is at least q. So targeted
  phase control is affordable only between gears whose product is at most q^2 / (2P); the
  large gears can be held (zero slip inside their own mirror) but not steered.
Verdict: dead as a full walk; the affordability limit is a fact to build with.

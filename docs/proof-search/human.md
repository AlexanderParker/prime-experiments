# human.md - the state of the hunt, in plain language

(Manager-rewritten 2026-09-06 after the proof-hunt rounds 32-55. Current-state snapshot; the
tree of every branch and verdict is research/proof/theory_tree.md, the map of every blocker is
research/proof/the_wall.md, the method is the theory-tree skill.)

## The five-minute version

We model twin primes as a machine: one gear per prime, each blocking positions on a fixed
schedule; a twin prime pair is a position every gear misses. The proof we want has one shape:
a known object we can point at and say "this is always in the window, because the machine
works this way, and nothing the machine does can prevent it". The window is the range where an
unblocked position is certified to be a twin (below the square of the next prime).

WHAT THE LAST TWO WEEKS FOUND, IN ORDER OF STRENGTH:

1. TWO NEW WRITTEN PROOFS. File 20: no ten or fewer primes, each blocking two positions at its
   own fixed spacing, can ever block the next prime's whole window (certified for every count
   up to ten), and the exact longest run any K such primes can block is 2, 5, 7, 16, 22, 28 for
   K up to six, by reasoning. File 21: two gears whose blocking spacings are equal (twin primes
   always are) must waste a strike on any run longer than that spacing plus one, so twin gears
   collide at (g + 4)/3, the earliest any pair can; the collision deficit of any pair grows by
   exactly four per common period; and a bound built from these collisions proves the small
   cases of the covering lemma without a solver. These are the first proofs on the adversarial
   side (any gears, any phases), which is stronger than what the conjecture needs.

2. A NAMED CANDIDATE OBJECT, MEASURED TO 200,000. Just past the square of any number q that has
   no factor 2, 3 or 5, look at the positions q^2 + 6i with i twelve more than a multiple of
   35. One of them, within a small fraction of the top gear's own tooth arc, is always
   unblocked, so it is a twin prime pair. Zero exceptions for every such q from 2849 to
   200,000, primes and composites alike, and the count of such open positions keeps growing.
   The reason it holds is measured too: covering all of those positions needs a number of
   cooperating gears that grows with the arc, while any single cover pins q as an integer.

3. THE MACHINE'S PARTS ARE NOW ALL PROVEN AND ITS PATH IS UNDERSTOOD. The walk from any square
   to the next twin starts on the top gear's own tooth and is struck by it once; which gears
   can reach a given offset at all does not depend on q (a quadratic-residue rule); the gears
   that can close both ends of a gap of size v are the prime factors of 3v - 1 and 3v + 1,
   i.e. the two numbers living in column v/2; the fixed points of halving are the twin columns;
   every maximal blocked run has gear 5 at its most-covering phase (the gear-5 lock); a record
   run is ordinary gaps of the machine below, fused at the ends by three top gears; junctions
   are ordinary openings; the neighbours of any gap of size six or more sum to at most the
   one-hole record.

4. WHAT THE WALL IS, EXACTLY. Every route from the machine's structure ends at one of three
   statements, and each is now precise. (a) Counting: a proof that uses only how many
   positions each prime blocks is a dimension-two sieve, which cannot reach the window (limit
   4.27, window at 2); the covering-systems method (distortion) applies to the machine but
   collapses on an interval for the same reason. (b) Transfer: the candidate object is rare
   among ALL possible phase settings (the real ones are typical in every symmetry, spacing and
   squareness measure), and turning "rare" into "never for real q" needs equidistribution far
   beyond any theorem; a bound below one on the failing fraction is already the conjecture.
   (c) Order: the interaction between gears needed to cut coverage below the window grows with
   the number of gears (block size about K - 3), so no fixed-order law reaches all machines.
   Position facts (corridors, locks, pinning) never see length.

## Honest ledger

- The conjecture is accepted as true and every measurement agrees with slack: the record
  stretch is a quarter of the window at every computed machine; the walk from the square lands
  within 265 columns to 5,000; the candidate object has growing room.
- Proved and new: files 20 and 21; the gear-5 lock; the junction theorem; the shadow and move
  lemmas; the span lemma and the head collision; the fibre and fixed-point theorems of the
  half-column map; the coupling-gear divisor laws; the type lemma; the exact spectrum recursion.
- Refuted this fortnight (and worth knowing): the real teeth are typical (spacing, coherence,
  squareness, cover number); coherence of spacings is a liability; twin gears are the cheapest
  small gears, so de-twinning LOWERS the record; the flank brick is the pair statement itself;
  no bounded-order interaction law exists; the sum rule cannot force depletion; the one-block
  covering inequality is trivial.
- Open: the one statement, in any of its three faces. Nothing measured says which face gives
  way first.

## The map

- research/proof/theory_tree.md: every branch, nested by what spawned it, with verdicts.
- research/proof/the_wall.md: every blocker stated precisely, the shape they make, the weak
  points tested (all six of the first map, and the second map's four), and the corrections.
- research/proof/dead_branches_reopened.md and _2.md: the unstick protocol run twice.
- docs/proofs/: 21 written proofs with ELI5 intros, status, prior art and their relationship
  to the conjecture.
- docs/novel/README.md: the register of results with prior-art status.

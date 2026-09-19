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

### 4. Blind choices of the landing offset D (2026-09-15)

Every walk from home lands at -1 + 2D, open by construction exactly to the gears dividing D.
Two flips about axes a1 and a2 give D = a2 - a1: the owner's slip, the difference of two mirror
sizes (times periods); the gears dividing the difference keep the origin's open phase without
sitting in either mirror. A blind walk is therefore a blind choice of D at most q^2 / 2.
research/stack/r8/blind_D_rules.py, results_blind_D_rules.txt, machines 11 to 5000 (665):
- D1 the largest primorial at most q^2/2: twin 15 of 665.
- D2 the largest lcm(1..m) at most q^2/2: 95 of 665.
- D3 the largest factorial: 6. D6 the largest 2^a 3^b (D a multiple of 3): 13.
- D4 the slip of the top two spiral steps, P (q - p) with p the prime below q (the difference
  of the mirrors {base, q} and {base, p}): landing in the window at 426 of 665 (below q when the
  gap q - p is small), twin at 119 of those 426. The best blind rule so far, at 28 percent of
  the machines where it lands inside: q = 101 lands (239, 241), q = 499 lands (3359, 3361).
- D5 the primorial spiral: 73 of 665.
Verdict: no blind D reaches every machine; the slip of the top two steps is the strongest
single rule. Every rule is open to the gears dividing D and blind to the rest.

### 5. Landing offsets from slips of step pairs (2026-09-15)

research/stack/r8/blind_slip_rules.py, results_blind_slip_rules.txt, machines 11 to 5000 (665).
D = P times: the slip of consecutive pair j (S(j), j = 1..6): twin 116 to 119 of about 420 in
the window; the first pair whose slip clears the window's start (W): 193 of 665; the spiral over
the top m gears (A(m)): m = 2: 119 of 426, m = 3: 75 of 665, m = 4: 213 of 622, m = 5: 67, m = 6:
167 of 655; the top pair with 2 and 3 periods: 107 of 576, 145 of 618; the slip across a gap of
two gears (q - g3): 198 of 631; of three: 168 of 663.
Verdict: every blind offset lands on a twin at a quarter to a third of the machines where it is
in the window, which is the share of twins among the columns open to the base low in the
window. No offset rule rises above that share. The odd spirals (m = 3, 5) sit lower because
their landings are farther up the window.

### 6. Hand the residues forward from machine to machine (2026-09-15)

Origin for machine q = the previous machine's landing (open to every gear up to p when it is a
twin); one flip up about {2,3,q}, {2,3,5,q} or {base, q}, the fewest periods that put the
landing above q. research/stack/r8/chain_machines.py.
- The chain leaves the twins at the first machine: from (11, 13) the flip about {2,3,7} lands
  at 95 = 5 * 19; from (17, 19) at 101, 103 twin... then breaks at the next. Continuing from
  every landing regardless, the origin climbs by 2M per machine and leaves the windows: twin
  at 22 of 562 landings in the window with {2,3,q}, none in the window with the larger mirrors.
- Why: the flip keeps only the phases of the gears in its mirror; the origin's openness to the
  other gears is not carried, and the accumulated moves outrun the windows.
Verdict: dead. Handing forward carries the mirror's gears and nothing else, as the gcd fact says.

### Interim reading after six algorithms

Every blind construction from home lands at -1 + 2D and is open exactly to the gears dividing
D, with D at most q^2 / 2; every blind rule tried lands on a twin at about the share of twins
among the base-open columns near its landing (a quarter to a third low in the window, less
higher up). A rule that always lands on a twin would have to choose D so that 2D - 1 and 2D + 1
avoid every gear up to q at every machine, which is the theorem itself in the machine's terms.
The loop continues with different principles for the choice of D, recorded here as they are
tried.

### 7. Landing offsets from the gaps between the top gears (2026-09-15)

research/stack/r8/blind_gap_rules.py, machines 11 to 5000 (665). D = P times: the product of
the two top half-gaps 125 of 488; the sum of the top three gaps 168 of 663; the product of the
three half-gaps 134 of 626; the largest of the three gaps 215 of 596; their lcm 95 of 550; the
top gap times the gear count 63 of 618; a mixed form 142 of 601; q times the top half-gap 33 of
432. Verdict: the same quarter-to-a-third band, highest for the smallest D (largest gap: the
landing sits lowest). Nothing above the base-open share.

### 8. Offsets from column-twin gears and from q's residues (2026-09-15)

research/stack/r8/blind_rules8.py, machines 11 to 5000. D = P times the slip of the two largest
gears whose own column is a twin: 146 of 641; of the two largest twin-member gears: 64 of 221;
q minus the largest column-twin gear: 106 of 639. D = P_s times q mod g_min: 91 of 664; times
(q-1)/2 mod g_min: 104 of 665; times the gear count: 64 of 554; times 1 or 2 by q mod 6: 306 of
662. The last is the descent at t = 1 or 2 chosen by q's class mod 6; it scores because t = 2
is a twin at P_s = 2310 and 30030's ranges hold t = 3 (the machines 4620 to 5000 with q = 1 mod
6 then fail); it is the first-gap table of the descent seen through q's class, not a mechanism.
Verdict: same band, with one coincidental high mark.

### 9. The sub-machine's landing plus one blind flip with the top gear (2026-09-15)

research/stack/r8/blind_rules9.py, machines 121 to 5000 (639). From the level-1 landing E_1
(at most q), one flip up with the fewest periods that enter the window: {3, q} 55 of 639;
{2,3,q} 61; {base, q} 31; {3, p} 54; {3, q} then {3, p} down (E_1 + 6 (q - p)) 76; {2,3,q} then
{2,3,p} down 76. Verdict: below the band (9 to 12 percent): these landings carry only 2, 3 (and
q or p), not even the machine's base, so they are open to fewer gears than the D rules.

### 10. What a blind algorithm would have to know (2026-09-15)

Axis differences that are multiples of the sub-machine's primorial cannot fit: the primorial of
the gears up to sqrt q passes q^2 once sqrt q >= 11 (recorded 2026-09-14), so a pair of flips can
carry at most the largest primorial P_max at most q^2 / 2, and the landing is 2 t P_max - 1 with
t free. Every rule for t tried in entries 4 to 8 is a guess and lands in the band.
The descent's first-gap table (research/stack/r8/results_descent_first_gap.txt) shows what the
guess would have to be: one integer per primorial, the first t with 2 t P_max +- 1 both prime:
  5#: 1, 7#: 1, 11#: 2, 13#: 3, 17#: 4, 19#: 12, 23#: 2, 29#: 8, 31#: 11, 37#: 2,
and at every primorial to 37# that t is inside the reach of the smallest machine of the range.
So an algorithm "flip on q#, flip back to P_max, take t from the table" lands on a twin at every
machine to about 10^14 with ten stored integers; the integers are the secret knowledge, one per
primorial, and the conjecture in this form is: for every primorial P, some t at most about
2 P_prev / p (the reach at the smallest machine) has 2 t P - 1 and 2 t P + 1 both prime.
Verdict: the loop's target reduces to a rule for one integer per primorial; no rule found.

### 11. Rules for the one integer per primorial (2026-09-15)

First gaps extended to 61#: 1, 1, 2, 3, 4, 12, 2, 8, 11, 2, 37, 12, 72, 14, 7, 130 for 5# to 61#
(the gap grows slowly; the reach at the smallest machine of each range, about 2 P_prev / p,
grows as the primorial). Eleven rules for t from the primorial alone tested at the 16
primorials: t = 1 (2 twins), 2 (3), the next gear (1), the last base gear (0), their difference
(1), P mod the next gear (2), the inverse and minus inverse of 2P mod the next gear (0, 0),
(next gear + 1)/2 (3), (last gear + 1)/2 (2), the number of base gears (2). No rule gives a
twin at more than 3 of 16. Verdict: the one integer per primorial has no rule among the natural
residues of the primorial; it is the position of the first pair of primes in the progression
2 t P +- 1, i.e. the opening itself.

### 12. Products of gear pairs and sums of primorials as the offset (2026-09-15)

Machines 11 to 5000 (665). D = P times two gears: the two smallest above the base 10 of 660;
the two largest that fit 82 of 655; the smallest with the largest fitting 64 of 663; g_min
squared 0 of 663 (measured; no reason established, checked: q = 101 gives 2939 with 2941 = 17 * 173).
D = sums of primorials at most q^2/2: P_max + P_prev 4 of 644; P_max - P_prev 116 of 665;
P_max + P_prev + P_prev2 148 of 641. Verdict: the band again (the sums of primorials carry
P_prev and sit low). The zero for g_min squared is no phenomenon: D depends on the base alone, so the rule has one
landing per primorial range (299, 2939, 50819, ... six values to q = 5000), and those six
happen to be struck (299 = 13 * 23; 2941 = 17 * 173; 50819 = 89 * 571). The same caveat holds for
every rule whose D depends only on the base: its count over machines is a count of ranges.

### 13. The pick-up walk with a deterministic scoring rule (2026-09-15)

research/stack/r8/pickup_walk_score.py, results_pickup_walk_score.txt, machines 11 to 2000.
Descending over the gears above the base; at gear g (mirror {base, g}) the move among periods
1..K and both directions is the one that keeps every visited gear off its teeth and maximises
the smallest distance of any visited phase from its nearer tooth; if none keeps the invariant,
the move with the fewest gears on a tooth. Deterministic; uses the walk's own phases, never
primality.
- In the window at all 299 machines. Landing with no visited gear on a tooth, hence a twin:
  K = 3: 116; K = 5: 175; K = 9: 197; K = 15, 25, 60: 220 of 299.
- The 79 failures: the gear left on a tooth is 11 at 50 machines, 7 at 21, 5 at 6, 13 at 1, 3 at
  1: the smallest gears above the base, visited last, where the joint constraint is tightest.
  The first defect appears at every stage of the walk, most often in the last fifth.
Verdict: the strongest residue-carrying rule so far: three quarters of the machines with no
primality used. Not a guarantee; the failures are the small gears at the walk's end.

### 14. The pick-up walk with a flexibility score and two repair flips (2026-09-15)

research/stack/r8/pickup_walk_v2.py, results_pickup_walk_v2.txt, machines 11 to 2000 (299).
Base with 2 and 3 forced in; gears visited from q downward (or small gears first, or large then
small); at gear g the mirror is {base, g}, periods 1 to 15, both directions; the chosen move
keeps every visited gear off its two teeth and, among those, leaves the next gear's step the
most such moves ('flex'), ties by the largest distance of any phase from a tooth then the
smaller move; after the last gear up to two repair flips about {base, s}, s a gear at most
sqrt q, by the same rule over all gears. Uses the walk's own phases only.
- Descending, distance score: 220 without repair, 291 with two repair flips.
- Descending, flexibility score: 296 without repair (fails 11, 13, 17), 299 of 299 with two.
- Small gears first, flexibility, two repairs: 299 of 299. Large-then-small: 299 of 299.
Every landing is open to every gear by the walk's own check, hence a twin by the landing lemma;
each was verified against the sieve. The walk carries residues by mirrors only and never
consults primality or the window; it is not a closed formula (it reads its own phases) and it is
not proved to always keep the invariant. Entry 15 runs it further.

### 15. The flex pick-up walk with two repair flips, further (2026-09-15)

research/stack/r8/pickup_walk_v3.py, results_pickup_walk_v3.txt. Entry 14's walk (descending,
flexibility score, periods 1 to 15, two repair flips about {base, s} with s at most sqrt q; 2 and
3 forced into the base) on every machine 11 to 4000 and every 60th prime to 11549: 561 machines,
a twin in the window at all 561, no failure. q = 499 lands 107099; q = 1999 lands 3821579.

## Standing after fifteen entries

The loop's target is met as a measurement: entry 14's walk carries residues by mirrors only,
consults no primality and no lookahead at the window, reads only its own phases (column mod
gear, computed from the mirror sizes), and lands on a twin at every machine tested (860
machines over entries 14 and 15). What is not met: a proof that the invariant "every visited
gear off its teeth" can always be kept, or that two repair flips always close a defect. That
proof is the twin prime statement for this walk; the loop does not produce it. The blind
constructions (closed formulas for the landing) all sit at the base-open twin share and cannot
reach every machine; the walk that does reach every machine is the one that chooses by its own
phases, which is the machine's own arithmetic, not a hidden fact about the opening.

Addendum to entry 15: the first run (thought lost) finished: every machine 11 to 5000 and every
20th prime to 19891, 745 machines, a twin in the window at all 745, no failure. q = 4999 lands
6842219; q = 15149 lands 138900299; q = 15881 lands 108722459.

### 16. The owner's chain: origin twin, base flip, machine grown by a member (2026-09-15)

Convention (owner): a flip "on P" has its axis at P/2 from the origin centre, landing O + P;
the pair map n -> 2a - n - 2 carries the gears dividing 2a = P. Chain: from the origin twin,
base = the lowest gears with product at most q^2 that do not strike the origin (a gear striking
the origin cannot be carried: from (5, 7) the gear 5 carries its own zero); landing O + P; next
machine = the larger (or smaller) member of the landing.
- (-1, 1) at machine 3, base {2, 3}: lands (5, 7). Machine 7 (or 5), base {2, 3} (5 strikes the
  origin), gears above the base 5 and 7: lands (11, 13). Machine 13, base {2, 3, 5}, above 7,
  11, 13: lands (41, 43). Machine 43, base {2, 3, 5, 7}, above 11 to 43: lands (251, 253),
  253 = 11 * 23. Three twins, then the chain breaks; the same by the smaller member.
- The family O + kP over the window at each machine: machine 7: 7 landings, twins at k = 1, 2,
  4, 6; machine 13: 5 landings, twins at 1, 2, 3; machine 43: 8 landings, twins at 2, 4, 5, 6,
  8 (k = 1 is the struck one). With the base limited to q^2 / 2 the chain breaks at once
  (base {2} at machine 3 lands (1, 3)).
- "One potential kill" holds only when exactly one gear lies above the base; from machine 13 on
  there are three or more, and the landing at k = 1 is decided by their teeth jointly.
Verdict: the chain is the descent with t = 1 and a moving origin; it carries the base and no
more, as the gcd fact says, and breaks at the fourth machine.

### 17. The twins' version of the single-prime closed form (owner's suggestion, 2026-09-15)

The single prime's next gap is the mex (the first residue missing) over the union of every
gear's progression (-y) mod g + j g on the line (R4.d.ii, kernel mex_form). The twins have the
same property, and it is already in the kernel: TopMachineWalk.lean defines Res G x as the union
over gears of the TWO offsets, of the left member and of the right, and mexS G x as the first j
outside it; open_mexS and mex_form prove that x + mexS is open to every gear of G, and is the
least such, in the free regime (every gear larger than twice the gear count); mexS_le bounds it
by 2 |G|. Measured here on the column line above q with every gear 5..q: the twin mex gives
the first twin above q exactly at every machine 11 to 5000 (the 163 apparent misses are the
machines where q itself heads a twin, which the mex, starting at j = 1, skips); mex at most 28,
median 5. It is the stack locator of 2026-09-15 and the repair walk of 2026-09-13 in closed
form: the same object, and its standing is the same as the nth prime's node, a FACT and not a
route, because the mex's value is the record of the gears' teeth on the stretch, not a formula.
The free regime, where the mex is provably below the bound, needs every gear above the base to
exceed twice their count, which never holds inside a window for the full gear set (at q = 2000
the smallest gear above the base is 13 against a count near 300).

### 18. Settle the next gear, then grow (owner's rule, 2026-09-15)

Rule: base = the carried gears (product P); the next gear g = the first prime not in the base;
flip on P (landing O + kP) taking the first k whose landing is off g's two teeth (provable:
within three flips, the kernel's settle lemma; measured k = 1 or 2 at every step); g joins the
base; the machine grows.
- Machine = the newest gear: settle 5 from (-1, 1) on 6: k = 2, (11, 13), in (5, 25]. Settle 7
  on 30: k = 1, (41, 43), in (7, 49]. Settle 11 on 210: the smallest landing 251 is past 121:
  no flip on the base stays in the window. Stops at the third gear: the base 210 exceeds the
  window (11, 121], the gcd wall.
- Machine = the twin member found: the window (L, L^2] always holds the next base, but the gears
  between the newest settled gear and L are not settled and strike: settle 13 on 2310 from
  (461, 463): k = 1, (2771, 2773), off 13's teeth but 2773 = 47 * 59 and 2771 = 17 * 163; settle
  23 on 9699690: (10243001, 10243003) struck by 29; settle 29: struck by 47. Twins at the steps
  for 5, 7, 11, 17, 19 (by the unsettled gears' chance), struck at 13, 23, 29.
Verdict: each settle step is provable and cheap; what cannot be done is to keep every settled
gear settled without holding it in the base, and the base is bounded by the window (newest-gear
machine) or the unsettled gears outnumber the settled (twin-member machine). Same wall.

### 19. The family of settle-and-grow walks (owner: iterate walks of this kind; 2026-09-15)

research/stack/r8/settle_walks.py, results_settle_walks.txt. 72 variants: settle order (smallest
unsettled gear, largest, current striker) x mirror (base; base times the settled gears that
fit) x k rule (first k off the new gear's teeth; first k off the teeth of the new gear AND every
previously settled gear; k = 1) x base rule (all settled gears that fit q^2/2; the largest
primorial at most q^2/2) x machine rule (newest gear; twin member). Chain from (-1, 1).
- Best: 7 twins, every variant with the 'keep' k rule and the newest-gear machine: (-1,1) ->
  (11,13) on 6 -> (17,19) on 6 -> (29,31) on 6 -> (59,61) on 30 -> (149,151) on 30 -> (179,181)
  on 30 -> (239,241) on 30, settling 5, 7, 11, 13, 17, 19, 23 in turn; then at machine 23
  (window (23, 529], base 210) settling 29 from 239 on 210: the one landing inside, 449, has
  451 = 11 * 41, the settled gear 11 back on its tooth, and the next landing 659 is past the
  window. The settled gears outside the base (11 to 23) are not carried by the flip on 210 and
  the window allows one flip: the gcd wall.
- Twin-member machine: 4 twins at best (base the largest primorial, 'first' or 'keep'), then a
  gear not yet settled strikes (59 at the fifth step); 'all' base: 3 twins.
- k = 1 never leaves the origin's window at machine 3 (the first flip on 6 lands at 5, below
  the window's start... (5,7) with 5 = q's window start): 0 by construction of the window test.
Verdict: the 'keep' rule is the pick-up walk restricted to the newest-gear machine; it reaches
seven twins and stops where the base can no longer fit the window. No variant of this family
carries the unsettled gears; the two walls are the same as entry 18.

### 20. The two variants of entry 19 (2026-09-15, loop)

Repair flip (a second flip on the base when no single flip keeps every settled gear) and the
base allowed up to q^2, each with the window of the old machine or of the new gear.
- Window of the old machine: 0 twins (the first flip from (-1, 1) on 6 lands at 5, at the
  window's start, so the settle of 5 has no room; same as before).
- Window of the new gear, base at most q^2/2: 8 twins, with or without the repair flip: the
  chain of entry 19 plus (659, 661) at machine 29; then settling 31 on 210 from 659 has no
  landing to 959 keeping 11, 13, 17, 19, 23, 29 off their teeth (the two flips available inside
  the window cannot satisfy six gears' two teeth each).
- Base at most q^2: 5 twins; the larger base (210 from machine 17) leaves fewer flips inside.
Verdict: the repair flip adds nothing (the window holds at most two landings once the base is
210); the base bound trades flips for carried gears one for one. The family is exhausted at
the gcd wall.

### 21. The reach theorem: what a mirror walk carries, exactly (2026-09-15)

Kernel proofs/MirrorWalkReach.lean (round 55, built, 0 sorries, standard axioms):
- gcdL_dvd_combo, gcdL_is_combo, multiple_reachable: a walk from home on mirrors M_1..M_n, any
  periods, directions, repeats and order, lands at -1 + 2D with D an integer combination of
  the mirrors; D is a multiple of their gcd G, and every multiple of G is such a D (Bezout on
  a list). The reachable columns are exactly -1 + 2 G Z.
- landing_open_of_dvd_combo: the landing is open to every gear dividing D (residue -1 carried).
- struck_reachable: for a gear g coprime to 2G, some reachable column is struck by g: the walk
  guarantees nothing about g.
- gcd_le_of_in_window: a landing inside (q, q^2] forces 2G <= q^2 + 1.
Closed statement: a mirror walk guarantees openness by construction to exactly the gears of
its gcd, and inside the window that gcd is at most (q^2 + 1)/2, one primorial's worth. Chains
of windows change nothing: every landing of every chain lies in -1 + 2 G Z for the gcd of all
mirrors used, and each window's bound applies to the landing inside it.
Consequence for the loop's target: no walk built from mirror actions alone can be proved to
land on a twin in the window without reading the teeth of the gears outside its gcd; the
guarantee it can give ends at the base. Any provable walk needs a mechanism other than
mirrors to control the residues above the base, or a proof of the window statement itself.

### 22. Every one- and two-flip rule from a grammar (owner: not all combinations tried; 2026-09-15)

research/stack/r8/rule_grammar_search.py, results_rule_grammar_search.txt, machines 11 to
2000. Origins: home, the twin gear pairs (5,7) to (41,43), the top twin gear pair, the top one
at most sqrt q. Mirrors: the spiral base B, the largest primorial at most q^2/2 and the one
below, B*q, B*p, B*g (g the first gear above the base), 6q, 6p, 30q, B*q*p. Periods 1 to 3,
both directions. 480 one-flip and 28800 two-flip fixed rules, each scored by the machines
where it lands on a twin in the window.
- Top scores: origin (17,19), flip B*g three periods up: 296 of 299; (41,43) with B*g twice
  up: 286; the top twin gears at most sqrt q with the primorial below the largest, once up:
  265; home with B*g twice up: 222. Two flips add nothing above 296.
- What the top rules are: B*g = P_s, the first primorial above q/2, which takes four values
  across the 299 machines (30, 210, 2310, 30030); the landing 17 + 6 P_s is therefore one of
  four numbers, 197, 1277, 13877, 180197, the first three twins, and the score counts the
  machines sharing those ranges. Followed through the primorials: 180197 at 13# is not a twin,
  and from 17# on the landing 17 + 6 P_s is divisible by 17, the origin gear, at every
  primorial. (41,43) + 4 P_s: twins at 7# and 11# only, divisible by 41 from 41#. Every
  high-scoring rule is a base-only offset, one landing per primorial range, its score a count
  of ranges, and each dies at a fixed primorial by the reach theorem's own arithmetic.
Verdict: the grammar's best rules are coincidences over the first four primorial ranges,
already dead beyond them. Consistent with entry 21.

Addendum to entry 22: the grammar restricted to machine-varying mirrors (B*q, B*p, 6q, 6p, 30q,
B*q*p), one and two flips, every origin: best 128 of 299, origin (29,31), B*q up then B*p down
(the slip of the top pair from a twin origin: 29 + 2B(q - p)); then 105, 103, 96, 87. All in
the base-open band. No machine-varying fixed rule leaves it.

### 23. The larger grammar, up to three flips (2026-09-15)

research/stack/r8/rule_grammar_search2.py, results_rule_grammar_search2.txt: 1152 one-flip,
110592 two-flip and 46875 restricted three-flip rules over origins to (101,103), machine-varying
mirrors B*g1..g3, B*q, B*p, B*p2, 6q, 6p, 6g1, B*q*g1, 2q, 2p, periods 1..3 and 'first period
entering the window' and 'last inside', both directions. Best 297 of 299: home, B*g2 twice down
then B*g3 three times up; and (11,13), B*g2 up then B*g3 up. Both depend on the base alone
(landing -1 + 2B(3g3 - 2g2) and 11 + 2B(g2 + g3)): one landing per primorial range. Followed
through the ranges: twins at the bases 6, 30, 210; not twins at 2310 (106259, 166331), so both
fail at every machine with q/2 in [2310, 30030), i.e. from q = 4621. Three flips: 286 at best,
the same shape. Verdict: as entry 22.

### 24. Evolutionary search over walk algorithms (owner, 2026-09-15)

research/stack/r8/evolve_walk.py. Genome = origin rule + steps of (mirror rule, period rule,
direction rule) from a grammar of machine quantities (base, primorials fitting the window,
gears above the base, gears below q, twin gear pairs, the gear at sqrt q, residues of q, the
step index, the column's position against the window); no residue or primality test of a
candidate column anywhere. Fitness: the streak of consecutive machines from q = 31 landing on a
twin in the window, then the streak from 11, then the total, then fewer steps. Elite kept,
tournament pool, mutation, crossover, random immigrants; the elite is written to
results_evolve_walk.json each generation. First run (fitness = streak from 11, 80 generations,
population 200): stalled at a streak of 6 (to q = 29), the elite converged to one genome; the
machines 11..29 with base {2,3} and windows to 841 decide the streak by luck. Second run with
the reworked fitness and diversity: running.

Entry 24, second run (fitness: streak from 31, then from 11, total, steps; 120 generations of
300, machines 11 to 2999, 426): by generation 60 the streak reached 420, every machine from 31
to 2999, total 424 of 426, with the genome: origin the third twin gear pair (11, 13); flip on
B * g4 (the fourth gear above the base) once up; then flip on B * g1 with the first period that
enters the window, up. Decoded: the first landing 11 + 2 B g4 already exceeds q throughout each
base range, so the entering period is always 1 and the landing is 11 + 2B (g4 + g1): one number
per base range, 227 (base 6), 1451 (base 30), 14291 (base 210), all twins, and 166331 at base
2310, not a twin (the same number as entry 23's second rule, since g4 + g1 = g2 + g3 = 36
there). The evolution rediscovered the per-range constant; with machines to 3000 only four
ranges exist to be fitted. Third form, evolve_walk2.py: machines sampled eight per base range
across the ranges of 6 to 43# (q to about 6 * 10^15, twin test by isprime), so a genome must
work across eighteen ranges. Running.

Entry 24, third form (cross-range machines, 98 sampled over the ranges of the bases 6 to 43#,
100 generations of 300): best streak 21 sampled machines from q = 31, to q = 4363, total 24 of
98; the genome is the same per-range constant as the second run (origin (11,13), flip on
B * g4 up, then B * g1 with the entering period), which dies at the first machine of the base-
2310 range, as decoded. No genome in 100 generations lands beyond the third base range in a
streak; the totals sit at the base-open share (24 of 98). Elite kept in results_evolve_walk2.json
for the next run.

Entry 24, fourth run (150 generations of 400 from the kept elite): no change; best streak 21,
total 24 of 98, the same genome. The grammar is exhausted for this fitness.

## Standing after twenty-four entries

Fixed rules (grammar searches over 160,000 rules, evolution over about 200,000 genomes) score
at the base-open share across ranges and reach long streaks only as per-range constants over
the first three primorial ranges. The reach theorem (entry 21, kernel round 55) says why: a
mirror walk's landing family is -1 + 2 G Z for the gcd G of its mirrors, open by construction
to the gears of G only, G bounded by the window. Algorithms that reach every machine (entries
13 to 15) do so by reading the landing's phases. No walk built from mirror rules alone lands
on a twin provably; widening the search over mirror rules cannot change that, since every rule
lands inside the same family. What could: a mechanism other than mirrors that constrains the
residues of the gears above the base, or a proof of the window statement by other means.

### 25. Evolution seeded with the hand-built walks (owner, 2026-09-15)

research/stack/r8/evolve_walk3.py. The genome gains macro steps: 'spiral' over a gear set
(all above the base, above sqrt q, at most sqrt q, column-twin gears, twin-member gears, solo
gears, the top a gears; order; first direction; base = spiral base, {2,3} or {2,3,5}; one or two
periods), 'descent' (landing 2 t P_s - 1 with a t rule), 'levels' (the recursive spiral), and
flips with mirrors 3h, 6h, 30h, B*h for h the a-th gear above sqrt q. Seeds: the six spirals,
the levels walk, the descent at t = 1, 2, 3, each spiral followed by one flip about {3, h} or
{2,3,h} for h the a-th gear above sqrt q (a = 0..3, up and down), levels then a flip, and the
grammar's best rules. Machines: every prime to 200 then every third to 6000 (288). Seed
ranking at generation 0 (streak from 31 / total of 288): descent t = 3: 7 / 64; sqout spiral
then {2,3,h_1} up: 7 / 24; descent t = 1: 6 / 52; sqout spiral: 5 / 72; sqout then {3, h_0}: 5 / 38.
The hand-built walks without a residue-chosen final gear sit at the base-open share, as
measured before (spiral landings 39 to 135 of 299). Run of 60 generations of 300: running.

Entry 25, result (60 generations of 300, 288 machines to 6000): by generation 30 the best is
origin (17,19), one flip on B * g1 = P_s (the first primorial above q/2) three periods up:
landing 17 + 6 P_s, streak 229 machines from 31 to q = 4603, total 232 of 288. This is entry
22's top rule, the per-range constant: 197, 1277, 13877 (twins) for the bases 6, 30, 210,
180197 (not a twin) from q = 4621, and divisible by 17 from 17# on. The seeded spirals, levels
and descent were outcompeted by it from generation 10 (their streaks 5 to 7, totals 24 to 72).
Verdict: with the hand-built walks as seeds the evolution converges to the same coincidence;
the macro steps add no fixed rule above the base-open share.

### 26. Islands: the hand-built families evolved independently (owner, 2026-09-15)

research/stack/r8/evolve_walk4.py. Five populations, no migration, each with a membership rule
enforced in fitness: spiral (a spiral step, no descent or levels), spiral+flip (a spiral then
flips), levels, descent, flips (flips only and not base-only: some mirror from the gears below
q, above sqrt q or the twin gears, or a period rule reading the window or q's residues). Each
seeded with its family's hand-built walks plus random members of the family; per-island elites
kept in results_evolve_walk4_<island>.json. Machines as entry 25. Smoke test at generation 0:
spiral and spiral+flip 7 / 24 (sqout spiral then {2,3,h1} up), levels 2 / 28, descent 7 / 64
(t = 3), flips 2 / 10. Run of 50 generations of 200 per island: running.

Entry 26, short run (50 generations of 200 per island, 288 machines to 6000), final bests:
- spiral and spiral+flip: streak 20 (to q = 113), total 84: sqout spiral on base {2,3} with two
  periods, then a flip about {3, h_2} (the third gear above sqrt q) with the entering period.
- levels: streak 8, total 65 (levels then a descent step, i.e. the descent alone).
- descent: streak 282 (every sampled machine 31 to 5981), total 287 of 288: descent with
  t = q mod 2 = 1, then a flip on the base B four periods up: landing 2 P_s - 1 + 8 B =
  2 B (g_1 + 4) - 1, one number per base range: 107, 659, 6299, 78539, 1261259, twins at the
  five bases 6 to 30030 (t = g_1 + 4 = 9, 11, 15, 17, 21 are gaps of those primorials'
  stripes), then 23483459 at 510510 not a twin: dies from q = 1021021. Five ranges by chance,
  one more than any earlier constant.
- flips: streak 229 (to q = 4603), total 233: the constant 11 + 2B(g_4 + g_1) again.
Verdict: the islands find their family's best per-range constant where the family allows one
(descent, flips) and sit at the base-open share where it does not (spiral, levels). The
descent island's winner is the descent at t = g_1 + 4: a fixed t rule that happens to hit the
first gap's neighbourhood for five primorials. Long run (400 generations of 300) continuing.

Entry 26, long run (400 generations of 300 per island, separate elites, results_evolve_walk4_long_*):
- spiral (no descent or levels step allowed): streak 29 (to q = 167), total 73 of 288: sqout
  spiral on the base, then flips on B * h_2 and 6 * g_5. The base-open share, a coincidence run
  on the small machines.
- spiral+flip, levels, descent: all three converge to the same landing, streak 282 (every
  sampled machine from 31 to 5981), total 287 of 288: a descent step (which replaces the walk's
  column by 2 t P_s - 1 with t = 1 under each island's t rule) followed by a flip on the base
  four periods up, i.e. 2 B (g_1 + 4) - 1, the five-range constant of the short run (dies at
  base 510510, q from 1021021). The spiral+flip and levels membership rules allowed a descent
  step inside, and the descent step, which discards the walk before it, invaded both.
- flips: streak 229, the constant 11 + 2B(g_4 + g_1).
Verdict: with 400 generations the islands settle where the short run left them. The strongest
object any island finds is "discard the walk, land at a fixed multiple of P_s, shift by a fixed
multiple of B": one integer per primorial range, chosen by chance across five ranges. Nothing
machine-varying rises above the base-open share.

### 27. Fifth form: residue rules for gear and period selection allowed (owner, 2026-09-15)

Owner's clarification: the bar was on hunting landings by division; residue checks that
choose a gear are allowed as long as the step does not test which landings are twins; a
fitness term should favour genomes that reach the same output without residue checks.
research/stack/r8/evolve_walk5.py adds gear-selection mirrors read at the current column
before the flip: hi_free (a gear above sqrt q whose phase is off its two teeth, so the flip
about {3, h} cannot be struck by h: a provable step), hi_far (the gear whose phase is farthest
from a tooth), hi_mod6, hi_adj (2 P_s = 1 mod h), hi_res (q's residue in the middle third); and
period rules steering one named gear (k_avoid: the smallest k off that gear's teeth; k_center:
the k centring its phase). Fitness: streak from 31, streak from 11, total, purity (1 without
residue rules, as a tie-break), fewer steps. Seeds: the hand-built walks and the residue-
selected final flips. Seed ranking at generation 0: descent t = 3: 7 / 64 (pure); sqout spiral
then {2,3,h_1}: 7 / 24 (pure); sqout then 6 * hi_free[1]: 7 / 18; descent then k_avoid: 7 / 12.
The residue-selected flips do not beat the pure ones at the seeds: choosing the flip's own gear
free settles that gear only, and the other gears take the landing at the usual share. Run of
200 generations of 300: running.

Entry 27, result (200 generations of 300, 288 machines to 6000): by generation 20 the best is
the constant 11 + 2B(g_4 + g_1) again (streak 229 to q = 4603, total 233), first found with
k_avoid periods (which return k = 1 there) and then replaced by its pure equivalent through
the purity tie-break; unchanged to generation 199. The residue-selection rules (hi_free,
hi_far, hi_adj, hi_res, k_avoid, k_center) appear in no winner: choosing the flip's gear or one
period by a residue settles one gear, and the landing meets the rest at the base-open share.
Verdict: under the owner's line (gear and period selection by residues allowed, landing tests
barred) the search finds nothing beyond the per-range constants and the base-open share.

### 28. Sixth form: multi-gear mirrors of settled gears, and a residue term (loop, 2026-09-15)

research/stack/r8/evolve_walk6.py. New mirrors: at the current column the gears whose phase is
off their two teeth are settled; a flip may take the mirror B times the first (a+1) settled
gears above the base (set_free), above sqrt q (set_free_hi), or all settled gears above the
base whose product keeps the mirror at most q^2/4 (set_free_fit); such a flip carries every
settled gear in it. Fitness adds, after the streaks and the total, the mean number of the
machine's gears on a tooth at the landing (fewer better), then purity, then steps. Seed ranking:
one flip from home on set_free_fit with the entering period: streak 14 (to q = 89), total 20,
mean gears on a tooth 1.59; the descent at t = 3: 7 / 64 / 1.55; descent then k_avoid: 1.29 on
a tooth but 12 twins. Run of 150 generations of 300: running.

Entry 28, first run: stopped by the system at generation 45 (low memory); at that point the
best was the constant 11 + 2B(g_1 + g_4) again (streak 229, total 233, mean gears on a tooth
0.59, pure). The settled-set flips did not hold the lead past the first generations. Resumed
from the kept elite for 100 more generations.

Entry 28, resumed run (100 more generations from the elite): unchanged; the constant
11 + 2B(g_1 + g_4) at streak 229, mean gears on a tooth 0.59. The multi-gear settled mirrors
appear in no winner. Verdict: same as entries 24 to 27.

### 29. Seventh form: per-range constants barred; residues as the target (loop, 2026-09-15)

research/stack/r8/evolve_walk7.py. Mode 'varying' (a genome whose landings coincide within
every base range scores zero), 120 generations of 300: best streak 54 (to q = 521), total 55 of
288: origin the top twin gear pair at most sqrt q, one flip on 6h two periods up, h the first
gear above sqrt q with h = 5 mod 6. Decoded: the landing t + 24h changes only when sqrt q
crosses a prime, so it is a table keyed on sqrt q: 269 (machines 31..113), 419 (127..283), 569
(293..523), three twins; then 713 = 23 * 31 and the landing falls below q from 541 on. A
coincidence class the base check does not catch (a table keyed on a slowly varying quantity).
The genuinely varying rules (landing changing at every machine) stay at the base-open share.
Mode 'teeth' (mean gears on a tooth first): running.

Entry 29, mode 'teeth' (100 generations of 300): the winner from generation 0 to 99 is the
descent at t = 2, landing 4 P_s - 1: mean gears on a tooth 0.635, total 181 of 288 (the base-
2310 range holds most of the sampled machines and 9239 is a twin), streak 0 (23 and 119 at the
bases 6 and 30 are not twins). A per-range constant again; asking for residues rather than
twins changes nothing, because a landing with few gears on a tooth is a landing that is a
twin at most machines of its range, and the base-only landings are the only ones that hold a
low count across a whole range.

## Standing after twenty-nine entries

Seven evolutionary forms (plain, cross-range, seeded, islands, residue selection, settled-set
mirrors, barred constants and residue target) and three grammar searches: every winner is a
lookup table keyed on a slowly varying quantity (the base, or sqrt q), holding for three to
five ranges by chance, or a machine-varying walk at the base-open share. The reach theorem
(entry 21) accounts for both. The evolutionary line is closed: no fitness or grammar over
mirror rules has produced a walk that lands on a twin beyond chance without reading the
landing, and none can, since every such walk's landing family is fixed by its gcd.

### 30. Eighth form: fitness by distance to the nearest twin (owner, 2026-09-15)

research/stack/r8/evolve_walk8.py. Each landing scored by its distance in slots to the
nearest twin inside the window (0 at a twin; outside the window, the window's width plus the
overshoot); primary fitness the mean of log2(1 + distance), averaged per base range and then
over the ranges (so the range holding most machines cannot dominate); then the streak, the
total, purity, steps. Machines balanced: up to 25 per base range for the bases 6, 30, 210,
2310 below 6000. Seed ranking under the unbalanced set: descent t = 2 (distance 0.85, twins
181, a constant twin in the 2310 range), descent t = 3 (1.56), descent then B * hi_free (1.64,
1.72), one settled-set flip (1.74). Run of 150 generations of 300 with the balanced set: running.

Entry 30, result (150 generations of 300, 88 machines balanced over four base ranges): best
mean log2(1 + distance) 0.4155, twins 55 of 88, streak 0: origin the top twin gear pair at most
sqrt q, one flip on the primorial below the largest fitting the window, entering period, up.
Decoded: the landing t + 2 P' takes fifteen values over the machines to 6000 (a table keyed on
sqrt q and on the primorial range): 431 (machines 127..241), 4637 (293..839), 4649, 60089,
60101 (1693..3469), 1021091 twins; 65, 425, 4631, 60119, 1021079 not. Six twin entries of
fifteen; the distance fitness favours it because its misses sit a few slots from a twin in the
dense low window. A table with two keys, not a mechanism. Next run adds a distinct-landings
requirement (at least 80 percent of the machines must land on different columns).

Entry 30, distinct-landings run (at least 80 percent of the machines landing on different
columns; 150 generations of 300): best mean log2(1 + distance) 0.8912, twins 32 of 88, streak 4:
home; flip on 3h with h the first gear above sqrt q with h = 5 mod 6, period the top gap mod 5,
alternating; then flip on twice the fifth prime below q with the smallest period off the teeth
of the third gear above the base. Machine-varying (its landing moves with q's neighbouring
primes); 32 of 88 is the base-open share. The fitness improved from 1.72 (generation 0) to
0.89 (generation 50) and then held. Longer run from the elite: launched.

Entry 30, long distinct-landings run (300 generations of 400 from the elite): the fitness
moved from 0.8912 to 0.8896 with the twin count falling from 32 to 27 of 88 (a genome with
nearer misses and fewer hits): a plateau. No machine-varying walk under the distance fitness
rises above the base-open share.

## Standing after thirty entries

With the smooth fitness (distance to the nearest twin, balanced across base ranges) and tables
barred, the evolution climbs for about fifty generations and then holds at the base-open
share, about one machine in three, with landings a slot or two from a twin on average. With
tables allowed it finds tables: two-key tables of up to fifteen landings, six of them twins.
The walks that reach every machine (entries 13 to 15) remain the only ones at 100 percent, and
they read the landing's phases. The evolutionary line, under every fitness tried, settles
where the reach theorem says it must.

### 32. Single flips on whole field sets and their combinations (owner, 2026-09-15)

research/stack/r8/field_set_flips.py, results_field_set_flips.txt: 15 field sets (base, sqout,
sqin, coltwin, colopen, col5, col7, twinL, twinR, solo, 1 and 5 mod 6, divisors of q - 1 and of
q + 1, top three gears), the set as ONE mirror (the product of all its gears), plus every union
and intersection of two sets and each set joined with the base; origins home and the top twin
gear pair; up and down; one period and the first period entering the window; 1600 rules over
105 machines 11 to 2000 (every prime to 173, then every fourth).
- Every set larger than a few gears overshoots the window at one period (sqin, sqout above
  q = 121, coltwin, twinL, twinR, solo, m1, m5, colopen, col5, col7 each have products far past
  q^2): their single flips never land inside. Only the base, the divisor sets of q +- 1, the
  top three gears, and small intersections fit.
- Best by twins: the base as the mirror from the top twin gear pair, entering period, up: 28 of
  105 (mean log2(1 + distance) 1.30); base with the divisors of q + 1 from home: 28 of 105; the
  base from home: 21 of 105; base with sqout (fits at 37 machines): 21 of 37; base with col7
  (fits at 50): 21 of 50. Intersections such as twinL with twinR (the gear 5) or divisor sets
  land inside at most machines but on a twin at 5 to 7 of 105.
- Downward flips and the top-three-gears mirror never land inside the window.
Verdict: whole-set mirrors either overshoot or are the base; the ones that fit land at the
base-open share (a quarter to a third), the same as every single flip carrying the base.

### 31. Ninth form: the settle walk with bounded residue memory (loop, 2026-09-15)

research/stack/r8/evolve_walk9.py. The pick-up walk of entry 14 as a macro step with a memory
w: at each gear's step the period and direction are chosen so that the gear and the last w
visited gears are off their teeth (w = 0: the flip's own gear only, a provable step; w = all:
entry 14). Fitness: streak from 31, streak from 11, total, then the smaller memory, then steps.
Seed ranking at generation 0 (288 machines to 6000):
  memory all, descending, K = 15, two repairs: streak 282 (every sampled machine 31 to 5981),
    total 287 of 288 -- the successful run under the fitness, and the entry-14 walk;
  memory 13: streak 31, total 89;  memory 5: streak 28, total 79;  memory 8: total 68;
  memory 0 (only the flip's own gear checked): the best is a plain flip constant, streak 48.
  small-first order with memory all: streak 7, total 237 to 253 (the small gears first leave
  the large ones unsettled at the end).
So the streak needs the whole memory: a walk checking only the last thirteen gears reaches a
third of the machines. The residue checking that makes the walk succeed is the full check of
every visited gear, which at the last step is the openness of the landing to every gear: the
sieve on the candidates. The run continues to see whether evolution finds a smaller memory
with the same streak.

Entry 31, measurement (research/stack/r8/settle_memory.py, results_settle_memory.txt; the
evolution itself was killed twice by the system's memory watchdog): the settle walk
(descending, K = 15, two repairs, flex rule) with memory w, machines every prime to 200 then
every third to 4000 (210):
  w = 0: streak 0, total 22;  1: 2 / 25;  2: 1 / 58;  3: 3 / 73;  5: 28 / 67;  8: 4 / 125;
  13: 31 / 83;  21: 35 / 73;  34: 47 (to q = 389) / 160;  55: 53 (to 499) / 185;
  89: 88 (to 1223) / 199;  144: 204 (every machine from 31 to 3947) / 209;  all: the same.
The streak grows with the memory and reaches the full run only when the memory holds every
gear visited, or at least the last 144 (the smallest gears, in descending order; for the
machines below 830 that is every gear). Read plainly: the walk succeeds exactly when the
landing is checked against the teeth of all the gears that can strike it, which is the sieve on
the candidate; a bounded memory succeeds only as far as the unchecked gears happen to miss.
The successful run under the fitness is entry 14's walk; it is a locator that reads the
landing, not a walk that carries the residues.

Entry 31, carrying variant (research/stack/r8/settle_memory_carry.py): the mirror at each step
is the base times the remembered settled gears that fit (product times sixty at most q^2), so
those gears keep their phase without being checked. Worse at every memory: memory all 203 of
210 (against 209 without carrying), streak 1 (fails at 37); memory 13: 84; 55: 130; 89: 147.
Carrying enlarges the mirror and leaves fewer periods inside the window, which costs more than
the checks it saves: the affordability limit (entry 3) seen from the memory side.

## Standing after thirty-two entries

A successful run under the fitness exists: the settle walk with full memory (entry 14), landing
on a twin at every sampled machine from 31 to 5981, and at 204 of 204 to 3947 in the direct
measurement. It succeeds because at each step it checks the landing against the teeth of
every gear visited, which by the last step is every gear of the machine: the sieve on the
candidate. Every reduction of that checking (bounded memory, carrying in the mirror, residue
rules for one gear or one period) falls back to the base-open share or below. Every rule that
reads nothing lands at the base-open share or is a lookup table. The evolutionary search has
converged, under nine forms and seven fitness definitions, on the same two outcomes the reach
theorem predicts. Loop stopped.

### 33. Keeping paths with lookahead (loop, 2026-09-16)

research/stack/r8/settle_lookahead.py, results_settle_lookahead.txt, 210 machines to 3947,
full-memory settle walk, descending, K = 15, taking the first keeping move (no flex scoring)
and demanding a keeping path of depth d ahead. Machines with a step lacking a d-keeping move:
depth 1: 195 of 210 (landings twins at 149); depth 2: 46 (twins 200); depth 3: 28 (twins 207;
q = 431 with 14 such steps, 461 with 21). Lookahead lowers the count and does not remove it;
a depth that never fails would be a search over the whole remaining walk, the sieve.
The flex rule of entry 14 (scoring by the next step's options) is depth 2 with a preference,
which is why it lacks a keeping move at 27 machines against 46 here. Verdict: no d-step local
lemma to depth 3.

### 34. More periods, and the previous gear carried in the mirror (loop, 2026-09-16)

research/stack/r8/settle_periods.py, results_settle_periods.txt (machines to 3947 for K = 15,
to 3000 for the rest; the memory watchdog stopped the first run). Steps lacking a keeping move:
K = 15, flex rule: 27 of 210 (twins 207). K = 30, flex: 27 of 170 (twins 167). K = 60 and 120
without the flex scoring (its lookahead scan was gated off above 30 for cost): 152 of 170
(twins 130, 132): without the one-step lookahead the walk runs into dead ends whatever the
number of periods. Previous gear carried in the mirror {base, g, g_prev}, K = 15: 148 of 170
(twins 133): the mirror grows by a factor g_prev and few periods stay inside the range.
Verdict: the count of missing keeping moves is set by the lookahead, not by the periods; a
larger mirror is worse (the affordability limit again). Next: the flex scoring at K = 60.

Entry 34, addendum: with the lookahead enabled, K = 60 and K = 120 give exactly the same 27
machines and the same failing steps as K = 15 (q = 11, 13, 17, 23, 61, 67, 73, 79, 83, 89,
127, 431 with 7, ...): the extra periods never supply a keeping move where the first fifteen
did not. So the missing moves are not for want of periods; at those steps few candidates fit
the range at all, or every candidate in range is struck. Measured next: the number of in-range
candidates at each failing step and which gears strike them.

### 35. Anatomy of the failing steps (loop, 2026-09-16)

research/stack/r8/settle_failures.py, results_settle_failures.txt, machines to 3000, K = 15.
Of the 80 steps lacking a keeping move, 77 have at most three of the thirty candidates inside
the range [0, q^2 - 2] (most have one or two: q = 61 step 7, gear 31, column 2819: one
candidate, struck by 31; q = 79 step 7, gear 47: one, struck by 47; q = 67 step 10: two,
struck by 37 and 53), and only 3 have more than three candidates all struck (q = 11's step
with 9 candidates all struck by 3, 5, 7, 11; q = 17's with 4; q = 73's with 4). So the failures
are the affordability wall, not the residues: at the large gears the mirror 2Pg is a sizeable
fraction of q^2 and one or two candidates fit; when those happen to be struck there is no
keeping move. Where several candidates fit, one keeps at nearly every step.
Next change: shrink the mirror at every step to {2, 3, g} (P = 6) and visit 5 and 7 as gears,
so that q/(12 g) or more candidates fit at every step, and measure the missing moves again.

### 36. The settle walk with the smallest base {2, 3} (loop, 2026-09-16)

research/stack/r8/settle_base6.py, results_settle_base6.txt (machines to 3000; the K = 40
line was cut by the memory watchdog and reruns). Mirror {2, 3, g} at every step, 5 and 7
visited as gears, full memory, K = 15: machines with a step lacking a keeping move fall from
27 to 8 of 170, and of those only the toy machines 13 and 23 fail for want of candidates in
range; at 1471, 1861, 2819, 2843 one step has more than three candidates in range and all are
struck (a residue failure). Landings in the window 169 of 170, twins 163.
So the small mirror removes the affordability failures as expected; what is left is a handful
of true residue failures, and the landing is a twin at 163 of 170 (the walk recovers from most
defects, not all). The candidate count per step now runs from about q/12 at the top gear to
q^2 / 60 at gear 5.

Entry 36, K = 40 (93 machines, every prime to 200 then every fifth to 2000, foreground run):
machines with a step lacking a keeping move: 4 of 93, all toy machines (11, 13, 17, 23); at
every sampled machine from 29 to 2000 a keeping move existed at EVERY step. Landings in the
window 92, twins 90 (the misses are among the toy machines). So with the mirror {2, 3, g} and
forty periods each way, the one-step statement held at every step of every machine sampled
above 23. Verification on every machine to 1500 follows.

Entry 36, every machine 29 to 700 (116 machines, K = 40, mirror {2, 3, g}, full memory): a
keeping move at every step of every machine; 116 of 116 landings are twins.
Every machine 701 to 1100 (59) and 1101 to 1500 (55): the same, no missing keeping move, every
landing a twin. So on every machine from 29 to 1500 (230 machines) the settle walk with the
mirror {2, 3, g}, full memory and forty periods each way never lacks a keeping move, and its
landing is a twin every time.
Every machine 1501 to 1750 (33) and 1751 to 2000 (31): the same. In all, every machine from
29 to 2000 (294 machines): the settle walk with mirror {2, 3, g}, full memory and forty
periods each way never lacks a keeping move at any step, and its landing is a twin every time.
The method that got here (owner): read each failure on its own; the failures were the
affordability wall, and shrinking the mirror removed them.

### 37. Piece 2, the free regime, in the kernel (2026-09-16)

proofs/MirrorWalkSettleFree.lean (round 57): strikes_iff_offA, resA_card_le,
keeping_move_free: with the visited gears all larger than twice their number, a keeping move
among 2n + 1 consecutive candidates exists, exactly. Covers the descending walk's early steps;
the tail with the small gears is uncovered (recursive_walk.md, piece 2).

### 38. The softer target (q, q#] (owner, 2026-09-16)

Quick measurement, machines 11 to 400 (74): primorial spiral with the base bounded by q^e / 2,
landing in (q, q#] and a twin by primality test: e = 1: 13 twins, all inside the window; e = 2:
16, none inside the window; e = 3: 10; e = 4: 6. Beyond q^2 a column open to every gear up to
q need not be prime (two factors above q), so the machine certifies nothing there: the softer
range gives up the landing lemma, and the rates do not rise. Not pursued further.

### 39. The tail as the machine one level down (loop, 2026-09-16)

research/stack/r8/settle_tail.py, machines 401 to 997 (90), mirror {2, 3, g}, K = 40. The
prefix = the free-regime steps; the tail = the gears below about q / 4.5 down to 5; the big
gears stay in memory. Per tail step (3327 steps), of 79.5 candidates in range: open to the
small gears visited so far 55.2; open to the big gears 55.1; open to both, the keeping moves,
40.9; open to the small gears but struck by a big one 14.3. Tail steps with no keeping move: 0.
So in the tail the keeping moves are plentiful on average (forty of eighty), the big gears
remove about a quarter of what the small gears leave, and the tail's own machine (the gears
below q / 4.5) leaves two thirds of the candidates open on average. The binding quantity is
the minimum over steps, measured next.
Machines 1009 to 1249 (36, 1938 tail steps): per tail step 42.7 keeping moves of 80 on
average, never zero; but the MINIMUM over the tail steps is always at the last step, gear 5,
where every gear is visited: 3 keeping moves of 80 at q = 1019, 1039, 1061, 1087; 4 at 1171; 5
to 8 elsewhere. So the walk's difficulty sits at one place: the last flip, about {2, 3, 5},
with all gears in memory, which is the final step by residues of 2026-09-14 with eighty
candidates. Everything before it has margin; the last step's margin is a handful and will
shrink as q grows unless the periods grow with it. The tail is not the machine one level down;
it is the prefix plus one final flip whose existence is the twin statement on eighty spaced
columns.

### 40. The last step's margin against q (loop, 2026-09-16)

research/stack/r8/last_step_margin.py, results_last_step_margin.txt. The last flip is about
{2, 3, 5} from a column L open to 2, 3, 5 (L = 11, 17 or 29 mod 30); candidates L +- 60 k
inside the window. Proxy: 200 columns per q drawn from the window with the right residue.
Twins among the eighty candidates at K = 40, and the K at which every sampled column has a twin
within K periods:
  q = 500: mean 8.36, min 3, none 0 of 200; K needed 29
  1000: 6.42, 1, 0; 31        2000: 5.28, 0, 1; 47        5000: 4.29, 0, 1; 43
  10000: 3.71, 0, 7; 91       20000: 3.09, 0, 3; 54       50000: 2.60, 0, 11; 103
  100000: 2.12, 0, 11; 71     300000: 1.98, 0, 33; 155     1000000: 1.55, 0, 40; 133
The margin at a fixed K falls like the twin density at the window's scale (about 1 / (ln q)^2
per column, so about 80 / (ln q)^2 twins among the candidates: 8.4 at q = 500 against the
measured 8.36, 1.6 at 10^6 against 1.55). The K needed grows slowly (29 to 155 over three
decades) and stays far below what the window affords (K up to q^2 / 120). So the last step is
never short of room; it is short of a proof that the progression L + 60k holds a twin below
q^2, which is the twin statement on a progression mod 30 - the descent's statement of
2026-09-15 with P_s = 30.
Verdict: the walk's difficulty is a single arithmetic progression statement, not a shortage of
candidates. K must grow with q (no fixed K works: at K = 40 a sample has no twin from
q = 2000 on), and any K growing like (ln q)^2 suffices in the measurement.

### 41. The walk's own last step (loop, 2026-09-16)

research/stack/r8/last_step_real.py: the settle walk run to the step before gear 5, then the
last flip about {2, 3, 5} from the column L it actually reached. Machines 907 to 1297 (57):
the first k with L +- 60k a twin has max 12 and mean 3.7; twins among the first eighty
candidates: min 3, mean 8.56; the column before the last flip sits at 28.6% of the window on
average (q = 997: L = 925079, 93% up the window, first twin at k = 7; q = 1009: L = 98867,
9.7% up, first twin at k = 1). Better than the proxy of entry 40 (mean 6.42 at q = 1000)
because the walk's column tends to sit low in the window, where twins are denser.
So the whole construction is: a prefix that settles every gear but 5 (three quarters of it
proved, the rest measured clean and with wide margins), then one flip about {2, 3, 5} whose
candidates are the progression L + 60k inside the window. The only unproved requirement, at
every machine tested, is met within twelve periods.

### 42. The last flip is redundant: the walk lands on the twin one step earlier (loop, 2026-09-16)

The last flip's mirror {2, 3, 5} has move 60k, and 5 divides 60, so the flip PRESERVES the
phases of 2, 3 and 5: it cannot settle gear 5. Gear 5 must already be open at the column before
it. Measured (machines 900 to 1300, 57): that column is open to gear 5 at all 57, and is
already a twin at all 57. So the walk has finished at the step before.
The mechanism is the one-step lookahead: at the gear-7 step the score prefers candidates whose
next step (gear 5) has keeping moves, and since the gear-5 move preserves 5, a candidate with
any such option must itself be open to 5; being open to every gear from 7 up as well, it is a
twin. The lookahead is not a heuristic here, it is the carrier of gear 5's condition.
Construction as it now stands: gears q down to 7, mirror {2, 3, g}, K = 40 periods each way,
memory every visited gear, score = keep all visited settled, then maximise the next gear's
options. The landing at the gear-7 step is a twin. The tight requirement is that step's: one of
its eighty candidates L +- 84k must be open to every gear from 5 to q. Measured margin: 3 to 14
such candidates at machines near 1000, mean 8.6; first success within 12 periods at every
machine 907 to 1297 and within 7 at the sampled machines 2003 to 4889.

### 43. Stopping the walk early (loop, 2026-09-16)

research/stack/r8/walk_stop.py, machines 300 to 900 (92). Stop at gear 7, lookahead over 5:
twin landings 92 of 92, no failure; candidates at the last step open to every gear from 5 up:
min 5, mean 9.37. Stop at gear 11, lookahead over 7 and 5: twin landings 33 of 92, though the
margin is still min 4, mean 7.40 - the scoring only looks one gear ahead, so it does not
enforce 7 and 5 together; the candidates exist, the rule does not pick them.
So the construction that works is: gears q down to 7, mirror {2, 3, g}, K = 40, memory every
visited gear, score = keep all settled then maximise the next gear's options (which at the last
step is gear 5 and carries its condition). Landing: a twin, at every machine tested.
The final step's requirement, stated cleanly: among the eighty columns L +- 84k, one is open to
every gear from 5 to q. Measured margin 4 to 14. A count cannot prove it: the small gears strike
about 2/h of the candidates each, and the sum over the gears diverges, so no counting bound
leaves a survivor (and counting is not allowed here in any case). The requirement is the twin
statement on a progression of modulus 84 inside the window.

### 44. Enlarging the proved prefix (loop, 2026-09-16)

research/stack/r8/prefix_extend.py.
(a) The spacing route (a gear h strikes at most 2K/h + 2 of the 2K candidates, so the prefix
survives while the sum of 1/h over the visited gears stays below 1) reaches LESS far than the
free regime: at q = 503 it ends at step 36 of 94 against the free regime's 65; at q = 937,
step 38 of 157 against 111. It is also a density bound, not an exact statement, so it would not
be admissible here in any case. Dropped.
(b) The free regime's own reach, with the period count taken as the number of visited gears
(K = n, the lemma's own hypothesis): steps covered, against ln q / (ln q + 2):
   q = 1000: 117 of 166 (70.5%), last free gear 239; law 77.5%
   10000: 940 of 1227 (76.6%), gear 1889; 82.2%
   100000: 7778 of 9590 (81.1%), gear 15559; 85.2%
   1000000: 66148 of 78496 (84.3%), gear 132299; 87.4%
The covered fraction rises towards 1 like ln q / (ln q + 2): the unproved tail is a vanishing
fraction of the walk, but it always ends at the gears 5, 7, 11, ..., and those are exactly the
gears whose teeth are dense on any candidate line. A single flip carrying the whole tail would
need the mirror to hold the primorial of about q/4, astronomically past q^2 (affordability),
so the tail cannot be collapsed into one step either.
Standing: the walk is proved step by step for a fraction of the steps tending to 1, and the
residual is the twin statement on one progression, as entry 43 states.

### 45. Settling the small gears first, carried in the base (loop, 2026-09-16)

research/stack/r8/small_first_carry.py, machines 300 to 1200 (134). A flip about {B, g} keeps
the phases of every gear dividing B, so a carried primorial base would make the later steps
immune to the small gears. But the base must satisfy 4 B q <= q^2, i.e. B <= q / 4, so the
largest carried primorial is 30 at q = 503 and 210 at q = 997: the first three or four gears,
exactly what the walk already carries. The gears above the base still include 11, 13, 17, ...,
which are small against their own count, so "every step in the free regime" holds at 0 of 134
machines. Twin landings 98 of 134 (worse than the {2,3,g} walk with lookahead, which lands on
a twin everywhere: the larger mirror costs candidates and this version has no lookahead).
Verdict: closed by the affordability limit, the same wall as the reach theorem, now step-wise:
a mirror can carry only the gears of a primorial at most q / 4.

### 46. Choosing the last stride (loop, 2026-09-16)

research/stack/r8/last_stride_choice.py, machines 400 to 1000 (90). The walk is run down to the
last two gears, then the last stride is tried over the first eight gears above sqrt q. Some
choice of stride gives StepOpen at all 90 machines; each fixed stride works at 87 to 90 of 90;
and the smallest working period over the choices is at most 4 at every machine (q = 503: gear
23 works at k = 1 with seven open candidates of eighty; q = 997: gear 47 at k = 3 with five).
So StepOpen may be stated over a union of eight progressions with K = 4, that is sixty-four
candidates, rather than one progression with eighty. The union has more candidates and the
strides are different, so the small gears' forbidden classes fall differently on each; the
hypothesis is weaker in form. Its content is unchanged: some column of a finite explicit set
inside the window must be open to every gear, which is the window statement on that set.

### 47. Is the candidate family a complete residue system? (loop, 2026-09-16)

research/stack/r8/family_cover.py, machines 400 to 900 (76), family = 8 strides x 4 periods x
2 directions = 64 candidates. The family misses residues modulo 5 at every machine, and the
missed class is always L's own: the stride 12 g k is 2 g k mod 5, so k = 1..4 gives the four
nonzero multiples and never returns to 0 mod 5; the family covers four classes of five and
misses the class of L itself. The same for any h > K (missing L's class mod h), so a family
with K periods is a complete system exactly for the gears at most K. Since L is open to the
small gears (it comes from the walk), missing its own class is a loss, not a gain.
Candidates surviving the gears 5, 7, 11, 13, 17: min 11, mean 15.8 of 64. So the five smallest
gears cut the family to about a quarter, and every remaining gear is above 17 and strikes about
2/h of what is left. The sum of 2/h over the gears 19 to q is near 1 at these sizes, so no
count leaves a survivor; the measured survivors (three to fourteen open to every gear) come from
the strikes overlapping, which is the twin statement's own content.

### 48. The returning mirror, and the trade stated exactly (loop, 2026-09-16)

research/stack/r8/returning_mirror.py, machines 400 to 2000 (225). Ending the walk on a column
congruent to home modulo the tail primorial T inherits the tail gears' openness from the origin
instead of re-earning it. But the landing must lie in the window, so 2T <= q^2: the inherited
tail reaches only gear 13 (T = 30030) up to q = 1300 and gear 17 (T = 510510) beyond, leaving
73 to 296 gears to be handled, while the columns in reach fall to 16, 4, 3, 2 as T grows. A
twin among the inherited columns at 87 of 225 machines: q = 997 has 16 columns and 6 twins,
q = 1999 has 3 columns and none. Worse than the walk, and it is the descent of 2026-09-15
re-derived as an endgame.
The trade behind every one of these attempts, now in the kernel (MirrorWalkInWindow,
mirror_times_candidates): candidates spaced 2M apart inside a window of length q^2 - q number
at most (q^2 - q) / (2M). Carried gears multiply M; candidates need room; the two divide the
same window. Every variant of the last eight entries has spent the window on one side or the
other, and the product is fixed.

### 49. StepOpen's margin at larger machines, and the period law (loop, 2026-09-16)

research/stack/r8/stepopen_large.py, 60 draws per q of a column low in the window with the
right residue mod 30, candidates over the first eight strides above sqrt q.
- Fixed family (8 strides, 4 periods, 64 candidates): open candidates mean 2.97 at q = 1000,
  2.02 at 3000, 1.37 at 10^4, 1.30 at 3 x 10^4, 0.98 at 10^5, 0.92 at 3 x 10^5; draws with none
  1, 9, 10, 13, 23, 22 of 60. A fixed family thins out, as the twin density says it must
  (64 candidates times about 2.2 / (ln q)^2).
- Family scaled as (ln q)^2 / 4 periods (176 candidates at q = 1000 up to 624 at 3 x 10^5):
  open candidates min 2 to 3 and mean 9.3 to 10.4 at every q, no draw with none.
So the construction must let the period count grow, K about (ln q)^2 / 4, and then the margin
is flat in q (about ten open candidates of a few hundred). The window affords far more: at the
top gears K may run to about q / 24 before the candidates leave the window, so the growth costs
nothing. Recorded as the walk's period law; the hypothesis StepOpen is stated with this K.

### 50. The whole walk under the period law (loop, 2026-09-17)

research/stack/r8/walk_scaled.py. Entry 49's family had eight strides; the walk's own step has
one stride and 2K candidates, so the law must be read per step: K = (ln q)^2 / 4 gives only
eleven periods at q = 1000 and the walk then fails at 15 of 145 machines 29 to 900 (failures
101, 157, 191, 211, 223, 281, ...), with one step lacking a keeping move. With K = (ln q)^2
(48 periods at q = 1000, 96 candidates per step): twin landings 145 of 145, no step lacking a
keeping move. So the walk's period law is K = (ln q)^2, not a fixed forty and not a quarter of
that; at q = 1000 it is 48 periods, at 10^6 it is 191, always far below the window's own limit
of about q / 24.
The free-regime prefix is 67.5% of the steps at these sizes (it does not depend on K: the lemma
needs the visited gears above twice their number).

### 51. The period count the lemma needs, against the one the walk uses (loop, 2026-09-17)

Two numbers had been conflated. The free-regime lemma proves a keeping move exists among
2n + 1 candidates when the n visited gears all exceed 2n: it needs the period count to reach n
at step n. The walk, measured, needs only K = (ln q)^2 uniformly (entry 50: clean at 145 of 145
machines 29 to 900 and 85 of 85 machines 901 to 1500, free prefix 71%).
- With a uniform K = (ln q)^2 the lemma covers only the first K steps: 47 of 166 at q = 1000
  (28%), 190 of 78496 at q = 10^6 (0.2%).
- With the period count taken as the lemma asks, K_n = 2n + 1 at step n, the lemma covers the
  whole first cut (every visited gear above twice their number): 70.5% of the steps at
  q = 1000, 76.6% at 10^4, 84.3% at 10^6.
- And that period count fits the window throughout the cut: the candidates' span
  12 g_n (2n + 1) as a fraction of q^2 is at worst 0.90 at q = 1000, 0.68 at 10^4, 0.44 at 10^6
  (falling, since the cut ends where g is about q / 4.5 while n is about q / ln q).
So the construction is stated with the period count K_n = max((ln q)^2, 2n + 1) at step n: the
walk uses the first term, the proof of the prefix uses the second, and both fit the window.
The proved prefix is then the first cut, rising to 1 like ln q / (ln q + 2).

### 52. The tail's requirement is uniform (loop, 2026-09-17)

research/stack/r8/tail_uniform.py, machines 200 to 700 (79), K = (ln q)^2. At EVERY tail step
(the steps after the first cut) some candidate is open to every gear of the machine, at all 79
machines; the minimum over a machine's tail steps is 1 at worst and 2.14 on average; and the
first tail step already has such a candidate at every machine.
So StepOpen need not be reserved for the last step: it holds at the tail's first step, and the
walk may stop there. The construction becomes: run the proved prefix (the first cut), then take
the first candidate open to every gear; that column is a twin by the landing lemma. The open
statement is StepOpen at ONE step, the first of the tail, with the column that the proved
prefix hands over.

### 53. The margin at the tail's first step, against q (loop, 2026-09-17)

research/stack/r8/first_tail_margin.py, 40 draws per q of a handover column (a column of the
lower window open to every gear the prefix has settled, and to 2, 3, 5), stride 12 g with g the
first gear below the cut, K = (ln q)^2:
   q = 1000: cut at gear 233 (step 117 of 166), K = 47, 94 candidates: open min 2, mean 4.65,
             draws with none 0 of 40
   2000: gear 433, K = 57, 114 candidates: min 1, mean 4.78, none 0
   5000: gear 997, K = 72, 144 candidates: min 0, mean 4.55, none 2 of 40
   10000: gear 1879, K = 84, 168 candidates: min 2, mean 4.88, none 0
   20000: gear 3533, K = 98, 196 candidates: min 2, mean 5.62, none 0
The margin is flat in q at about five open candidates, and a drawn handover column occasionally
has none at the first tail step (two of forty at q = 5000). The walk's own column does better
(its lookahead steers it), and entry 52 showed every tail step carries an open candidate, so
the statement to name is StepOpen at SOME tail step, not at the first. Recorded that way.

### 54. Extra strides cannot move the cut (loop, 2026-09-17)

With A strides of K periods each (2AK candidates), a visited gear h strikes at most 2*ceil(K/h)
candidates per stride, so at most 2A*ceil(K/h) in all. The lemma needs the total struck below
2AK, i.e. the sum over the visited gears of ceil(K/h) below K. The factor A cancels: it
multiplies candidates and strikes alike. With every visited gear above K each term is 1 and the
condition is n < K; with gears below K the terms grow like K/h and the sum passes K exactly
where the free regime ends. So the cut stands at "the visited gears exceed twice their number",
whatever the number of strides: 70.5% of the steps at q = 1000, 76.6% at 10^4, 84.3% at 10^6.
The cut is where exact pigeonhole stops. Below it a gear strikes more than one candidate per
stride, the strikes must overlap for a survivor to remain, and that overlap is the twin
statement. This is the boundary of what this route proves, stated cleanly.

### 55. No visiting order moves the cut (loop, 2026-09-17)

The cut depends on the visited SET, not on the order: after n steps the memory holds n of the
machine's gears, so its smallest is at most the n-th largest gear of the machine. Descending
order attains that bound, so it is optimal, and the best cut over all orders is the largest n
with the n-th largest gear above 2n: 117 of 166 steps at q = 1000 (gear 239, about q/4.2), 940
of 1227 at 10^4 (1889, q/5.3), 7778 of 9590 at 10^5 (15559, q/6.4), 66148 of 78496 at 10^6
(132299, q/7.6). Interleaving cannot help: a gear left unvisited is not in the memory, so it
does not raise the minimum; it only delays its own settling, and every gear must be settled
before the landing.
With entries 54 and 55 the boundary is fixed from both sides: neither more strides nor a
different order moves it. The proved prefix is exactly the steps where the memory's smallest
gear exceeds twice the memory's size, and it is the largest prefix any pigeonhole argument can
reach on this construction.

### 56. Settle by exclusion (loop, 2026-09-17)

Keeping an unvisited gear's phase across a move 12 g k needs h | k (for h not 2, 3 or g), so
keeping a whole unvisited set U needs prod(U) | k and the move is at least 12 g prod(U), which
must stay inside the window: 12 g prod(U) <= q^2. Measured: the unvisited set that can be kept
while the top gear moves is {5} at q = 101, {5, 7} at 503, 997 and 1999 - three or four gears
of the machine's 24, 94, 166, 301. The rest cannot be kept at all.
So settling by exclusion keeps exactly the carried base of the primorial spiral and leaves the
remaining gears unconstrained, as before: the trade lemma again, reached from the other side.
Every way of spending the window has now been tried - carry the gears in the mirror, keep them
by the period, inherit them from the origin, spread them over strides, reorder the visits - and
each buys the same amount and no more.

### 57. The tail's struck classes have a fixed shape (loop, 2026-09-17)

On the candidate line c + s k the gear h strikes at k = -c a and k = -(c + 2) a, with a the
inverse of s modulo h, so the two teeth sit a fixed distance -2a apart: fixed by h and the
stride alone, the same for every column, and only the PAIR'S POSITION moves with the column.
Measured for the strides 12g at g = 11, 13, 101: gear 5's gap is 4, 3, 4; gear 7's is 2, 6, 5;
gear 11's is 10, 10; gear 13's is 12, 8; and so on, each an exact residue with no pattern
across h.
So the tail's question is a covering question with explicit shapes: fixed-width pairs, one per
gear, placed on the candidate line by the column, and a survivor exists unless the pairs cover
every candidate. That is the fields on the line again, with the shapes named; it is not a count,
but it is also not a mechanism that forces a gap, because the placements are exactly the column's
residues, which the walk does not control.

### 58. Can the prefix choose the handover column's small residues? (loop, 2026-09-17)

research/stack/r8/handover_choice.py, machines 300 to 800 (77), K = (ln q)^2. At the prefix's
last step the candidates that keep every visited gear settled cover, on average, 11.7% of the
180 admissible residue triples mod 5, 7, 11 (min 0%, max 17.8%); no machine reaches all 180.
The reason is the arithmetic of one step: the candidates are c + 12 g k for k = 1..K, so their
residues mod 5, 7, 11 run through at most K values of a single progression in each modulus, and
K = (ln q)^2 is far below 385; the keeping condition then removes most of those. The prefix
does not choose the handover residues; it offers a small, structured subset.
To choose freely the walk would need K of the order of 385 (the product of the small moduli)
at a single step, and in general the product of the tail's gears, which is the primorial of
about q / 4.5 - past q^2 by the trade lemma. So the placements stay given, not chosen, and the
tail's covering question is entered where the prefix leaves it.

### 59. Choosing among the handover columns (loop, 2026-09-17)

research/stack/r8/handover_set.py, machines 300 to 700 (63), K = (ln q)^2, prefix run without
the lookahead term so the whole keeping set H at its last step can be enumerated (32 columns at
q = 691). Of those columns, 97.4% lead to at least one candidate open to every gear at the FIRST
tail step (minimum 92.5% over the machines), the best column offers 8.6 open candidates on
average, and at 17 machines every column of H works. At one machine of the 63, q = 307, no
column of H works at the first tail step: the walk must use a later tail step there, which
entry 52 shows it can (that measurement followed the walk with its lookahead, and the lookahead
is what steers it to a column with a continuation).
So the freedom to choose the handover column removes almost all of the first step's risk but
not all of it, and the statement stays as named in entry 53: StepOpen at SOME tail step.

### 60. How many tail steps does the walk need? (loop, 2026-09-17)

research/stack/r8/tail_steps_needed.py, machines 200 to 900 (108), K = (ln q)^2, the walk run
with its lookahead. The first tail step at which some candidate is open to every gear of the
machine is step 0 at all 108 machines - the walk can finish at the tail's very first step, every
time, when it arrives there by its own path (entry 59's 97.4% measured the handover columns of a
prefix run WITHOUT the lookahead; the lookahead removes the remaining risk).
So the open statement can be stated at one named step, not "some step": StepOpen at the tail's
first step, applied to the column the lookahead prefix hands over. The construction is then a
fixed finite procedure with one hypothesis:
   prefix (proved, the first cut) -> handover column -> one step of 2K candidates -> twin.
Extension to machines 901 to 1600 running.

Entry 60, extension: machines 901 to 1200 (42): the first tail step at which the walk can finish
is step 0 at all 42, as at 200 to 900. 150 machines in all, no exception.

### 61. The construction as one theorem (loop, 2026-09-17)

proofs/MirrorWalkTheorem.lean (round 59, built, 0 sorries, standard axioms):
- `Handover`: the walk's data at the handover (column, stride, period bound, top gear).
- `construction_twin`: with G holding every prime from 5 below the top gear, StepOpen at the
  handover gives a twin prime pair inside the window.
- `window_statement_of_stepOpen`: if every machine's walk meets its step hypothesis, every
  machine's window holds a twin prime pair - the window statement, from the construction.
So the chain is complete as an implication, with exactly one hypothesis left, and that
hypothesis is measured at the tail's first step at every machine from 200 to 1200.

### 62. Who strikes the handover's candidates? (loop, 2026-09-17)

research/stack/r8/handover_strikers.py, machines 300 to 800 (77), K = (ln q)^2. Per handover
step: 75.8 candidates; struck by a gear ABOVE the cut 22.5; by a tail gear only 48.4; open 4.92;
distinct gears striking 54.9, out of 68 above the cut and 31 in the tail.
Two things this settles. First, the gears above the cut are not spent: the handover column is
open to them, but the candidates are new columns and those gears strike 22.5 of 75.8 - the
prefix's work does not carry to the next line, as the trade lemma says it cannot. Second, the
tail gears do most of the striking (48.4), and 55 distinct gears strike at all, so the covering
is spread thinly: no small set of gears is responsible, and the survivors are the columns that
every one of the 55 misses.
A proof of StepOpen must therefore handle about 55 gears striking about 71 of 76 candidates
with no gear dominating - the twin statement's own shape at this scale, with the counts made
explicit.

### 63. Does the prefix earn its place?  No (loop, 2026-09-17)

research/stack/r8/no_prefix.py, machines 300 to 800 (77), K = (ln q)^2, open candidates at the
final step:
  (a) the full walk, prefix to the cut then the handover step: 4.92 on average, none at 0
      machines;
  (b) no prefix at all, one step from home with the same stride: 2.73, none at 5 machines;
  (c) no prefix, one step from home, best stride of the first eight gears above sqrt q: 7.31,
      none at 0 machines.
So (c) beats the full walk: choosing among eight strides from home gives half again as many
open candidates as walking the whole prefix first, and works at every machine. The prefix is
not decorative, it is worse than the freedom it consumes.
This is the trade lemma once more: the prefix spends the window's room on settling gears whose
settling does not survive the next line (entry 62: the gears above the cut still strike 22.5 of
75.8 candidates at the handover), while the eight strides spend the same room on candidates,
which do survive.
The construction therefore reduces to the one-flip locator of Part III: from home, among the
columns -1 + 12 g k d for eight gears g, k = 1..K and both directions, one is open to every gear
of the machine. That is the statement to carry; the settle walk's proved prefix proves that the
walk can continue, not that the ending is easier.

### 64. The one-flip locator's margin and scaling (loop, 2026-09-17)

research/stack/r8/oneflip_margin.py: from home, the columns -1 + 12 g k d, open to every gear
(hence twins in the window). Open candidates:
   q = 1000: A = 8, K = (ln q)^2 (752 candidates) 31; A = 8, K = 40 (640) 23; A = ln q (564) 25;
             A = 1, K = (ln q)^3 (658) 23
   2000: 33 / 20 / 27 / 31        5000: 44 / 21 / 44 / 31
   10000: 40 / 20 / 46 / 48       20000: 40 / 15 / 42 / 41
So with any scaling that grows the candidate count like (ln q)^2 or better, the margin holds at
thirty to fifty open columns; with a fixed K = 40 it thins (23 to 15 over the range). The shape
of the family hardly matters: one stride with K = (ln q)^3 periods does as well as eight strides
with (ln q)^2, at the same candidate count - the margin follows the number of candidates, not
their arrangement.
The first open candidate is always at a small period (k = 10 to 19 at these sizes) and the
smallest stride tried (the first gear above sqrt q), so the locator finds its twin at once.
The statement to carry is: among the columns -1 + 12 g k d with g the first gear above sqrt q
and k up to (ln q)^3, one is open to every gear of the machine.

### 65. The one-flip locator stated in the kernel as the final form (loop, 2026-09-17)

proofs/OneFlipLocator.lean (round 60), built, 0 sorries, axioms propext / Classical.choice /
Quot.sound:

    def oneFlip (g : ℤ) (k : ℕ) (d : ℤ) : ℤ := -1 + 12 * g * k * d

    def OneFlipOpen (G : Finset ℕ) (g : ℤ) (K P : ℕ) : Prop :=
      ∃ (k : ℕ) (d : ℤ) (m : ℕ), 1 ≤ k ∧ k ≤ K ∧ (d = 1 ∨ d = -1) ∧
        (m : ℤ) = oneFlip g k d ∧ 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧
        ∀ h ∈ G, ¬ (h ∣ 6 * m - 1) ∧ ¬ (h ∣ 6 * m + 1)

    theorem oneflip_twin : (G full below P) → OneFlipOpen G g K P →
      ∃ m, 1 ≤ m ∧ 6 * m + 1 < P ^ 2 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime

    theorem window_statement_of_oneflip : (OneFlipOpen at every machine) → the window statement

So the construction now has one definition and one hypothesis, and the hypothesis names an
explicit finite family: from home, one flip about {2, 3, g} with g the first gear above sqrt q
and period k up to (ln q)^3, either direction.

What this replaces: MirrorWalkTheorem's `Handover` (round 59) carried the settle walk's column
and stride as data the prefix had to produce. Entry 63 showed the prefix is worse than no
prefix (4.92 open candidates against 7.31), so the handover is not something to build - the
column is home, -1, and the stride is 12 g. `Handover` and `construction_twin` stay in the tree
as the general step form; `oneflip_twin` is their instance at c = -1, s = 12 g d, and is now
the statement the document carries.

Proof document: research/proof/proof_skeleton.md Part IV b rewritten to match. Section 15 is
the one-flip locator (stated, one hypothesis, measured to 20000); section 16 keeps the settle
walk as a recorded result with its two proved parts, `in_range_move`/`window_move`/`base_fits`/
`stays_in_range` (round 56) and `keeping_move_free` (round 57), and records that part (b)
applies to the one flip at n = 1 - it is why a period exists that keeps g itself off its teeth;
section 17 names the one open hypothesis. Part V standing updated.

Where the loop stands after 65 entries: everything around the flip is proved, and the one thing
left is that some column of the family is open. The trade lemma says why no walk removes it -
room in the window buys either carried gears or candidates, never both - so the next line of
attack is not another walk shape but the gears that strike the family: which gears can strike
how many of the (ln q)^3 columns, and whether their striking classes can be shown to miss one.

### 66. The teeth law on the one-flip family, and which mirror to use (loop, 2026-09-17)

The family's members, written out: the column -1 + 12 g k has members 72 g k - 7 and 72 g k - 5.
So a gear h coprime to the stride strikes the candidate k exactly when

    k == 7 u  (mod h)    or    k == 5 u  (mod h),      u = (72 g)^{-1} mod h.

Every gear's two teeth on the family are the SAME shape - the fixed pair (7, 5) - scaled by that
gear's own unit. The family fixes the shape, the gear fixes the scale, nothing else enters.
Checked against division at every gear and every k = 1..60 for q = 500 and q = 1000: 0
mismatches. PROVED in the kernel: `strike_iff_scaled` (h divides s k - c iff k = c u, for any c),
`oneflip_teeth` (both members at once), `oneflip_members` (the members are that form)
[proofs/OneFlipLocator.lean round 61, 0 sorries].

Which mirror. research/stack/r8/oneflip_classes.py surveyed every admissible mirror gear g:

   q      best g (open)    g = first > sqrt q (open)    worst g (open)    mean over all mirrors
   500    7  (30)          23  (8)                      97  (0)           2.66 over 93
   1000   5  (37)          37  (17)                     271 (0)           3.61 over 166
   2000   5  (47)          47  (19)                     719 (0)           5.41 over 301
   5000   5  (55)          71  (24)                     2843 (0)          8.29 over 667

So the mirror the previous entry proposed (first gear above sqrt q) is a poor choice, not a good
one: it leaves 0 open columns at q = 101 and about half the margin of g = 5 everywhere larger.
The best mirror is the smallest that fits the window. research/stack/r8/oneflip_small_mirror.py,
open columns at K = (ln q)^3:

   q       {2,3}   {2,3,5}   {2,3,7}   {2,3,11}   first > sqrt q
   251     17      25        16        9          2
   1009    28      37        34        28         17
   5003    40      55        49        42         24
   20011   44      58        59        58         36

{2, 3, 5} is best or tied at every size, and the margin grows with q instead of thinning.

Why a bigger mirror is worse, mechanically: it carries more gear phases (the gears dividing the
mirror never strike) but its stride spaces the candidates further apart, so the same period count
runs out to much larger numbers - at q = 20011 the sqrt mirror's stride 10728 puts its 971
candidates out past 10^7 while stride 360 keeps them under 3.5 x 10^5. That is the trade lemma
read on the mirror rather than on the walk: room in the window buys carried gears or candidates,
never both.

Failure modes, looked at individually. The three machines with no open column at g = 5 are
q = 11, 13, 17, and the cause is not covering: their windows end at 121, 169, 289, all below the
first candidate 353, so the family is EMPTY there - the mirror does not fit. This is exactly the
`base_fits` condition of round 56 (4 P g <= q^2 - q - 2) failing. The mirror {2, 3} (stride 72)
has a candidate in every window from q = 13 and an open one at every machine 13 to 20011 except
q = 11. So the statement carries two mirrors: {2, 3} below q = 19 and {2, 3, 5} above, or simply
q >= 19 with the small machines exhibited outright.

Next: the teeth law says the striking classes are 7u and 5u. Two gears strike the same candidate
when 7u_h = 5u_h' ... modulo different moduli, which is where the covering lives. The question
that decides the open statement is whether the map h -> u_h can be shown to leave a k uncovered,
and the units u_h are the inverses of 360 modulo each gear - a fixed object, the same for every
machine, growing only by adding gears.

### 67. Correction: the family's members, and the teeth law that follows (loop, 2026-09-17)

Entry 66 is wrong and this entry replaces it. It read the landing -1 + 12 g k d as a COLUMN and
so wrote the members as 72 g k - 7 and 72 g k - 5. The landing is a MEMBER, not a column: the
flip about the mirror {2, 3, ...g} sends the home pair (-1, +1) to

    (12 g k d - 1,  12 g k d + 1),     column m = 2 g k d.

(Entry 64's script used exactly this - x = -1 + 12 g k d with the pair (x, x + 2) - so entry 64
stands. It was entry 66's two new scripts that took the wrong form.) The wrong form is not just
mislabelled, it is degenerate: with g = 5 its member 72 g k - 5 is always divisible by 5, and
with g = 7 its member 72 g k - 7 is always divisible by 7, so the mirrors {2,3,5} and {2,3,7}
could never have produced a twin. Entry 66's scripts hid this by skipping every gear dividing
the stride, which is right for the true family and wrong for that one. All of entry 66's mirror
comparison numbers are therefore void.

The teeth law on the true family, proved and measured (0 mismatches against division at
q = 1000 and 5000, all gears, k = 1..80):

  * a gear dividing the stride 12 g never strikes at all - it would have to divide 1;
  * every other gear h strikes at exactly two periods, k = v and k = -v modulo h, where v is the
    inverse of 12 g modulo h.

So the teeth are a SYMMETRIC pair about k = 0, and k = 0 is home. That is the mirror's carrying
property read on the candidate line: the mirror's own gears are open along the whole family, and
every other gear places its two teeth as mirror images about home. Kernel: `oneFlip` corrected
to 2 g k d, `oneFlip_members`, `strike_iff_scaled`, `oneflip_teeth`, `mirror_gear_never_strikes`
[proofs/OneFlipLocator.lean, round 62, 0 sorries, axioms propext / Classical.choice / Quot.sound].

The mirror comparison redone (research/stack/r8/oneflip_symmetric_teeth.py), open columns within
K = (ln q)^3 periods:

     q      {2,3}   {2,3,5}   {2,3,7}   {2,3,11}   first gear > sqrt q
    251      21       32        22         21          16
   1009      35       48        38         31          24
   5003      22       69        55         47          31
  20011       0       72        67         59          41

{2, 3, 5} wins at every machine. {2, 3} collapses at q = 10007 and beyond for a different
reason - its window starts at k = q / 12, already past K, so the family has no candidate in
range at all. Every machine from 11 to 20011 has an open column with {2, 3, 5}; the fewest is 1
(at q = 11), and the first open period sits 0 to 27 past the window's start, mean 4.7.

How far the periods must run, measured to two million (mirror {2, 3, 5}, stride 60):

        q      window starts at k     first open period past it     open of the first 400
     1009              17                        0                          58
    20011             334                       10                          43
   100003            1667                       13                          43
   500009            8334                       34                          33
  2000003           33334                        8                          23

So the period bound is not polylog: it is the window's own start, k = q / 60, plus an offset
measured at 0 to 34 over that whole range. The candidates before the start are below q and are
not in the window; the walk must simply begin where the window begins.

What this changes in the statement: the family is now the columns 2 g k d with the mirror
{2, 3, 5}, periods from the window's start, and the open count grows with q instead of thinning.
What it does not change: the hypothesis is still that one of them is open, and that is still the
window statement on one progression.

### 68. The killers by field type, the primorial mirrors, and the chain (loop, 2026-09-17)

Three measurements and one new kernel piece.

(a) Killer census by the fields explorer's ids (research/stack/r8/oneflip_killer_fields.py).
Mirror {2,3,5}, the first 60 candidates past the window's start, every composite member named by
its field:

    q = 1009   struck 49 of 60, both members 16;  higher1:7 x15, higher1:11 x10, higher1:13 x7,
               higher1:23 x5, higher1:17 x5, higher1:19 x4;  products:2 x56, products:3 x8,
               products:4 x1
    q = 100003 struck 54 of 60, both members 28;  higher1:7 x14, higher1:13 x7, higher1:11 x7,
               higher1:29 x4, higher1:17 x4, higher1:19 x4;  products:2 x54, products:3 x23,
               products:4 x5

So the kills are almost all products:2 and products:3, and the smallest gear is nearly always a
single power (higher1, not higher) of a gear just above the mirror. The largest-gear ids are
scattered one or two each - lower:g carries no structure here. Carrying more gears clears only
part: gears up to 37 clear 39 of 49 kills at q = 1009 but only 29 of 54 at q = 100003, so the
small gears matter less as the machine grows, exactly as their teeth thin.

(b) Primorial mirrors (research/stack/r8/oneflip_primorial_mirror.py). Mirror = every gear up to
B; first open period past the window's start, 200 periods tried:

      q      B=5  B=7  B=11  B=13  B=17  B=19  B=23  B=29
    1009      0    5     1     2     -     -     -     -
    20011    10    4     7     2     3    11     -     -
   100003    13    1     2     1     3    11     1     -
  1000003    10    2     3     4     3    11     1     7

The constant columns are the tell: once 2 B# exceeds q, the window starts at k = 1 and the first
open period no longer depends on the machine at all. The landing is then a FIXED number serving
many machines at once.

(c) The chain (research/stack/r8/primorial_chain.py, kernel proofs/MirrorWalkChain.lean round 63,
0 sorries). A landing t = 2 B# k that is a twin pair lies in the window of every machine with
sqrt(t) <= q < t - 1 - a band from its square root up to itself. So the machines are covered by a
chain of landings whose bands overlap, and the overlap condition is t(next) + 1 < (t - 1)^2.
PROVED: `chain_covers`, `window_statement_of_chain`.

The first twin landing at each primorial mirror, B = 5 to 127:

    k = 1, 1, 2, 3, 4, 12, 2, 8, 11, 2, 37, 12, 72, 14, 7, 130, 121, 32, 103, 10, 56, 62, 36,
        40, 24, 63, 113, 1, 6

All 28 consecutive pairs satisfy the chain condition, so these 29 landings cover every machine
from 8 to 4.8 x 10^49. The slack is enormous and grows: at B = 11 the period could have been 38
instead of 2, at B = 17 31796 instead of 4, at B = 23 1.2 x 10^8 instead of 2, at B = 113
6.3 x 10^46 instead of 1.

What this changes. The open statement no longer has to be met once per machine. It has to be met
once per primorial mirror: some period k up to about the previous landing squared over 2 B# has
2 B# k +- 1 a twin. The mirror switches off every gear up to B by construction (the teeth law's
first clause), so only the gears above B can strike, and the allowance per link grows like the
square of the link before.

What it does not change: this is still an existence statement about twins, so it is not a proof.
It is a restructuring of the target from one statement per machine to one per primorial, with the
freedom in each link growing without bound.

### 69. The carry wall, proved and measured (loop, 2026-09-17)

The mirror is the only thing that switches a gear off, and the window says how many it can
switch off. A landing at 2 M k inside (q, q^2] forces 2 M <= q^2, and every gear is at least 2,
so the carried gears number at most log2(q^2). PROVED: `two_pow_card_le`, `carried_le_log`,
`mirror_in_window_carries_le`, `uncarried_card`
[proofs/MirrorWalkCarry.lean, round 64, 0 sorries, axioms propext / Classical.choice /
Quot.sound]. `uncarried_card` is the other half: of the machine's gears all but at most
log2(q^2) stay live, and those are exactly the ones the period has to dodge.

Measured on the real gear sets (research/stack/r8/carry_wall.py), largest primorial mirror that
fits the window:

         q      gears   mirror to B   carried   periods afforded   uncarried   free regime needs
      1009        169        13           6                 16         163     gears above 326
      5003        670        19           8                  1         662     gears above 1324
     20011       2263        19           8                 20        2255     gears above 4510
    100003       9593        23           9                 22        9584     gears above 19168
   1000003      78499        31          11                  2       78488     gears above 156976
  10000019     664580        37          12                  6      664568     gears above 1329136

The smallest uncarried gear in each row is the next prime after B: 17, 23, 23, 29, 37, 41. The
free regime of `keeping_move_free` needs every uncarried gear above twice their number, so at
q = 1000003 it needs them above 156976 and they start at 37 - a factor of four thousand, and
growing.

The trade in one machine, q = 1000003, every primorial mirror:

      B    carried   uncarried   periods the window affords
      2        1       78498        250,001,250,001
      7        4       78495          2,380,964,285
     13        6       78493             16,650,099
     19        8       78491                 51,548
     23        9       78490                  2,241
     29       10       78489                     77
     31       11       78488                      2

Each gear carried divides the periods by that gear and removes exactly one gear from the dodge
list. That is why no mirror choice can reach the free regime: the mirror buys gears
logarithmically and the window pays for them geometrically. The sixty-odd rounds of walks were
all paying that same price in different arrangements, and the wall is now a kernel statement
rather than a series of measurements.

What survives: the chain of entry 68 does not fight this wall, it sidesteps the per-machine form
of it - one landing serves every machine from its square root up to itself, so the construction
needs a landing per primorial, each with an allowance growing like the square of the one before.
The wall says the landing cannot be forced by carrying; the chain says it does not have to be
forced often.

### 70. The certificate: the window statement proved below 1.29 x 10^8 (loop, 2026-09-17)

The chain of entry 68 has a consequence worth taking: each link roughly squares the range
covered, so the number of landings needed to settle every machine below X grows like log log X.
Taking each link as large as the chain condition allows - the largest twin centre below the
square of the one before (research/stack/r8/chain_certificate.py):

     n   landing                    steps of 6 below the square   covers machines up to
     1   108                                  2                   107
     2   11352                               16                   11351
     3   128845110                           15                   1.28845e8
     4   1.66011e16                          33                   1.66011e16
     5   2.75595e32                         165                   2.75595e32
     6   7.59527e64                        2248                   7.59527e64
     7   5.76882e129                        462                   5.76882e129
     8   3.32793e259                      49508                   3.32793e259

Nine landings (with the start 12) settle every machine from 11 to 10^259. Each was found within
a few hundred columns of the square - the deepest search was 49508 columns at the last link.

The first four are now in the kernel. `window_statement_below`
[proofs/MirrorWalkCertificate.lean, round 65, 0 sorries, axioms propext / Classical.choice /
Quot.sound] proves with no hypotheses:

    every machine q with 11 <= q <= 128845108 has a twin prime pair inside its window (q, q^2].

Supporting kernel pieces added this round: `chain_covers_upto` (a finite chain covers a bounded
range of machines) and `window_statement_upto` (the same with the twin property attached)
[proofs/MirrorWalkChain.lean].

Each link is a one-flip landing: t = 2 M k is the flip from home about the mirror of product M at
period k, so 108 = 2 x 6 x 9 is the mirror {2,3} at period 9, and 11352 = 2 x 2838 x 2 the mirror
of product 2838 at period 2.

What this is and is not. It is the first unconditional window-statement result in the kernel: a
range of machines settled outright, by construction rather than by checking each machine. It is
not progress on the open statement, which is about all machines; the certificate's length grows
like log log X, so no finite certificate closes it. What it does close is the verification
question - "does the construction actually work" - for every machine below 1.29 x 10^8 in the
kernel and below 10^259 on paper.

### 71. The multiplicative chain: the next landing is a multiple of this one (loop, 2026-09-17)

A landing t is a twin centre, and every gear dividing t / 2 is open at it. Flip about the mirror
whose product IS t / 2: the move is 2 (t/2) j = t j, so the candidates from that landing are the
centres

    t * j,   j = 2, 3, 4, ...

and by the teeth law no gear dividing t / 2 can strike any of them. A landing t * j with
j + 3 <= t automatically satisfies the chain condition, since t j + 1 < (t - 1)^2 whenever
j <= t - 3. So the bands overlap by construction and no separate chain check is needed.

PROVED: `mult_chain_window` [proofs/MirrorWalkChain.lean, round 66, 0 sorries, axioms propext /
Classical.choice / Quot.sound]. Given a sequence with t(n+1) = t n * j n, 2 <= j n, j n + 3 <= t n,
6 | t 0, 12 <= t 0, and each t n a twin centre, every machine from t 0 - 1 onward has a twin pair
inside its window.

So the open statement now reads with no mention of machines or windows at all:

    for every twin centre t, some j <= t - 3 has t * j a twin centre.

MEASURED (research/stack/r8/multiplicative_chain.py). Over every twin centre from 12 to 20000
(340 landings) the smallest such j is 2 to 96, mean 15.2, and 30 or less in 89% of cases. The
largest j used is 0.66% of the allowance. By size:

    t near      smallest j: min   max    mean      allowance
    1e3                       4    46     11.8        1,017
    1e4                       2    90     17.2       10,005
    1e5                       6    50     19.5      100,149
    1e6                       6    92     27.9    1,000,035
    1e7                       4    70     23.4   10,000,137
    1e8                       7   529     92.2  100,000,035

The multiplier grows slowly (mean 12 to 92 across five decades) while the allowance grows like t,
so the margin widens by a factor of ten per decade.

Why this is the cleanest form so far: the mirror is not chosen, it is the landing's own gear set;
the chain condition is not checked, it follows from j <= t - 3; the machines never appear. The
construction is a single self-similar rule - from a twin centre, step to a small multiple of it
that is again a twin centre - and everything else in the proof document is the machinery that
turns that rule into the window statement.

What is unchanged: the rule still asserts twins exist, so it is a restatement of the open content
in the machine's own terms, not a proof of it. It is the smallest such restatement the search has
produced: one sentence, no parameters, no window.

### 72. The enriching chain: each landing carries one more gear (loop, 2026-09-17)

Two kernel facts make the multiplicative chain self-improving:

  * a gear dividing the landing can never strike its multiples - it would have to divide 1
    (`landing_gear_never_strikes`);
  * along a multiplicative chain every gear of a landing divides every later landing, so the
    carried set only ever grows (`mult_carried_monotone`).
[proofs/MirrorWalkChain.lean, round 67, 0 sorries.]

The mechanism shows up directly in the multipliers. 400 landings near 10^6, smallest j with t j a
twin centre, grouped by which of the gears 5, 7, 11 divide the landing
(research/stack/r8/enriching_chain.py):

    gears in the landing     n     mean j   median
    (none)                 191     37.7       25
    (5)                     92     20.6       15
    (7)                     50     33.9       19
    (11)                    24     37.1       29
    (5, 7)                  19     15.1        8
    (5, 11)                 15     18.9       14
    (7, 11)                  7     15.1       16
    (5, 7, 11)               2     11.5       20

A landing that carries gear 5 reaches the next landing with roughly half the multiplier of one
that does not, and carrying 5 and 7 halves it again. The gear that matters most is the smallest
one missing, exactly as the teeth law says: a gear h that does not divide t forbids the two
classes j = +-t^{-1} mod h, and the smaller h is, the larger a share of the multipliers it takes.

So the chain has a greedy form: at each step multiply by the smallest gear the landing does not
yet carry, times whatever small factor is needed to land on a twin centre again. Run from 12:

  step   new gear   multiplier j   m = j / p   landing digits   gears carried   share of allowance
     1          5              5           1        2                3          0.42
     2          7              7           1        3                4          0.12
     3         11             22           2        4                5          0.052
     4         13             91           7        6                6          0.0098
     8         29            203           7       16               10          4.8e-12
    12         43           4945         115       29               14          5.3e-22
    16         61             61           1       41               18          2.3e-37
    21         83           4399          53       62               24          5.7e-55

Twenty-one steps reach a 62-digit landing carrying 24 gears, and the multiplier is a vanishing
share of what the chain condition allows - by step 21 it is spending 5.7 x 10^-55 of the
allowance. The extra factor m stayed under 700 at every step.

This is the primorial spiral in its working form: the landings are multiples of a growing
primorial, each reached from the one before by a single mirror flip about the landing's own gear
set, and the covering theorem behind it is proved (`mult_chain_window`). What each step still
asserts is that such a multiplier exists, which is the open content; what the measurement shows
is that the room needed collapses to nothing as the chain proceeds while the room available grows
like the landing.

### 73. The multiplicative step's margin, and the certificate's ceiling (loop, 2026-09-17)

The open content is one sentence: for every twin centre t, some j <= t - 3 has t * j a twin
centre. This measures how much room that has - every working j, not just the first
(research/stack/r8/multiplier_margin.py, all 80 twin centres from 12 to 3000):

     t in            landings   working multipliers   mean   share of the allowance
     [12, 300)          17          3 to 31           15.1        16.5%
     [300, 1000)        16         19 to 72           47.5         8.0%
     [1000, 2000)       26         50 to 161          84.6         5.9%
     [2000, 3000)       20         87 to 284         136.5         5.6%

No landing has none. The smallest margin anywhere is 3, at the chain's own start t = 12 (the
working multipliers there are 5, 6 and 9, giving the landings 60, 72 and 108). The count of working
multipliers grows roughly with the landing while the share of the allowance settles near 5 or 6
percent, so the statement gets easier to meet as the chain climbs, never harder.

Between 58 and 100 percent of the working multipliers share a gear with the landing, which is
what the carrying rule predicts: gears dividing the landing cannot strike its multiples, so a
multiplier built from them starts with fewer forbidden classes than one built from new gears.

The certificate's ceiling. I tried to extend `window_statement_below` past its fourth link by
adding the fifth, 16601062113221682, which would settle every machine below 1.66 x 10^16. The
primality of a seventeen-digit member is beyond what `norm_num` will do in the kernel in
reasonable time (over ten minutes for one of the two members, against seconds for the nine-digit
link). Extending the certificate therefore needs Lucas or Pratt certificates for the two members
of each new link, not trial division. Recorded as the next kernel task if the range matters; the
proved range stays at machines below 128845109.

### 74. Lucas certificates, and the chain certificate extended (loop, 2026-09-17)

Entry 73 recorded the ceiling: `norm_num` proves a prime by trial division, so the kernel
certificate could not pass its fourth link. Measured here: a thirteen-digit prime
(1000000000039) took six and a half minutes and then failed on the recursion limit; the
seventeen-digit members of the fifth link would cost about 1.3 x 10^8 divisions each.

The fix is a Lucas certificate - a witness a of order p - 1 modulo p, checked by one
square-and-multiply chain and one check per prime factor of p - 1. The cost is the logarithm
rather than the square root: about fifty squarings for a seventeen-digit prime, each a single
multiplication that `norm_num` does instantly.

Kernel tools (proofs/PrattTools.lean, round 69, 0 sorries):
  * `cast_pow_eq_one_iff` moves `lucas_primality`'s ZMod statement to arithmetic on N;
  * `sq_of` and `sq_mul_of` are the two chain steps in literal form, so a generated proof never
    has to rewrite inside a goal;
  * `ne_one_of_mod` discharges the order conditions.

Generator: research/stack/r8/gen_pratt.py emits the Lean proof for a given prime, recursing into
the factors of p - 1 whenever they are too large for `norm_num` themselves (the threshold is
10^8). Checked on 1000000000039: certified in 34 seconds, against six and a half minutes of
failure by trial division.

proofs/PrattCertificates.lean holds the certificates for the fifth link's two members,
16601062113221681 and 16601062113221683, with the two recursive certificates they need
(2842647622127 and 144205681). 1671 lines, built in 2 minutes 39 seconds.

So the chain certificate now runs to five links:

    12,  108,  11352,  128845110,  16601062113221682

and `window_statement_below` [proofs/MirrorWalkCertificate.lean] proves, with no hypotheses:

    every machine q with 11 <= q <= 16601062113221680 has a twin prime pair inside (q, q^2].

That is a range of machines 1.3 x 10^8 times wider than round 65's, from five twin pairs and the
covering theorem. The next link (2.76 x 10^32) needs the same treatment one level deeper, since
the factors of its members minus one are themselves large.

### 75. The sixth link, certified here (loop, 2026-09-18)

Answer to the question of whether the sixth link can be run on this machine: yes, and it is done.

  * The recursion is shallow. The sixth link is 275595263287044304869593048464770; its two
    members need 8 certificates in all, at depth 3: the two 33-digit members, then
    733915969930370835373, 4680807713 and 150968500433427556811, then 118261309739333 and
    541474482383801, then 2463487181. Every factorisation of p - 1 along the way is within
    sympy's reach in seconds.
  * proofs/PrattCertificates.lean is now 7390 generated lines holding 12 certificates (links five
    and six and their recursions); it checks in 1 minute 48 seconds and builds in 3 minutes 42.
  * proofs/MirrorWalkCertificate.lean carries six links:

        12, 108, 11352, 128845110, 16601062113221682, 275595263287044304869593048464770

    and `window_statement_below` now proves, with no hypotheses and no sorries:

        every machine q with 11 <= q <= 275595263287044304869593048464768 has a twin prime pair
        inside its window (q, q^2].

    That is every machine below 2.76 x 10^32, from six twin pairs and the covering theorem.

What blocks the seventh link (5.77 x 10^129): a Lucas certificate for a prime P needs the
factorisation of P - 1, and for a 130-digit member that is a hard factorisation, not a slow one.
The chain has a partial way round it. In the multiplicative chain the landing t is a product of
the previous landing and a small multiplier, so t is factored by construction - which certifies
the UPPER member t + 1 for free, since its P - 1 is exactly t. The lower member t - 1 needs
t - 2 factored, and that is the random one. So a certificate-friendly chain would search among
the many admissible landings for one whose t - 2 breaks up under trial division into small
factors and a certifiable cofactor. That is a search, not a new idea, and it is the way to keep
the certificate climbing.

### 76. The step: what the machine cannot do, proved (loop, 2026-09-18)

Taking the recommendation of the last exchange - leave the certificate at six links, spend the
effort on the step - this round proves the negative half, which was until now argued from
measurements (entries 54, 55) rather than stated.

(a) The free regime's cut is sharp. `keeping_move_free` needs every visited gear above twice
their number. With three gears {5, 7, 11}, one of them (5) below 2 x 3, and the start 370, every
candidate of the run k = 0..6 is struck: 370 and 375 by gear 5, 371 and 378 by gear 7, 374 by
gear 11. So no keeping move exists and the conclusion genuinely fails. PROVED by decision
procedure: `keeping_move_free_sharp` [proofs/MirrorWalkSettleFree.lean, round 71, 0 sorries].

(b) Carrying cannot reach the free regime. A mirror that fits the window carries at most
log2(q^2) gears (`carried_le_log`, round 64), so if the machine has one more gear below some
bound b than the mirror can hold, a gear at most b is always left live
(`small_gear_uncarried`); and once the live gears number at least b/2 the free-regime hypothesis
fails outright, for every choice of mirror (`free_regime_unreachable`)
[proofs/MirrorWalkCarry.lean, round 71, 0 sorries].

Together with the measurements already on record - at q = 1000003 the machine has 78499 gears,
the largest mirror that fits carries 11, and the free regime would need the live gears above
156976 when they start at 37 - the pigeonhole route is closed from both sides: its hypothesis is
unreachable by any mirror, and where the hypothesis fails its conclusion can fail too.

What this leaves. The step - for every twin centre t, some j <= t - 3 has t j a twin centre - has
no forcing available from the mechanisms the machine has: mirror carrying is logarithmic
(`carried_le_log`), composition carries only the gcd (`gcdL_dvd_combo`, round 55), the window
trades carried gears against candidates exactly (`mirror_times_candidates`, round 56), and
pigeonhole stops exactly at the free-regime cut, which no mirror reaches. Every construction of
rounds 40 to 70 was a different arrangement of those four facts.

### 77. The analytic route, and the bridge it would need (loop, 2026-09-18)

Following the decision to aim at the weakest statement of the window's shape that known methods
can support: a Chen pair in every window, meaning a prime p in (q, q^2] whose partner p + 2 has
at most two prime factors.

The literature position, checked this round:
  * The best explicit constant for the twin half of Chen's theorem is
    pi_{1,2}(x) >= 1.205 C_2 x / (log x)^2 (Bordignon and Starichkova, arXiv:2405.05727), where
    pi_{1,2} counts primes p <= x with p + 2 a product of at most two primes and C_2 is the twin
    prime constant. It is stated for sufficiently large x with NO computable threshold.
  * The fully explicit work is on the Goldbach half: every even number above exp(exp(32.7)) is a
    prime plus a product of at most two primes (Bordignon, Johnston and Starichkova,
    arXiv:2207.09452), reduced to exp(exp(15.85)) under the generalised Riemann hypothesis.
  * So for the twin half there is no effective threshold to pair with a finite check, and where
    thresholds do exist they start at exp(exp(32.7)), which is beyond any certificate: ours
    reaches 2.76 x 10^32 (round 70).

The consequence is worth stating plainly. Loosening the target from "by construction" to
"analytic" does not produce a proof for every machine even of the weakened statement, because the
sieve results are asymptotic and the gap between our certificate and their thresholds cannot be
closed by computation.

What is now in the kernel is the bridge that would turn any such bound into the window statement,
with no analysis in it (proofs/WindowFromCount.lean, round 72, 0 sorries):

    `exists_in_window_of_count`: if fewer good numbers lie below q than below q^2, then a good
    number lies inside (q, q^2] - for any decidable property.
    `chen_window_of_count`: the same for Chen pairs.
    `chenPair_of_twin`: every twin pair is a Chen pair, so the analytic target is genuinely
    weaker than the machine's.

MEASURED (research/stack/r8/chen_window.py). Chen pairs are dense near the window's start, so the
counting hypothesis is not close to tight:

        q     first Chen pair past q   first twin past q   Chen pairs in (q, 20q]
       11              +2                    +6                      32
      101              +6                    +6                     166
     1009             +10                   +10                    1005
    10007             +30                   +30                   6895
   100003             +16                  +148                  50562
  1000003             +34                   +34                 386936

and the stretch (q, 20q] where those all sit is 1.9% of the window at q = 1009 and 0.019% at
q = 100003. The hypothesis of `chen_window_of_count` therefore has room to spare at every size
measured; what is missing is not room but an effective bound.

### 78. The anatomy of the failures, and the one lever (loop, 2026-09-18)

Asked to look at the reasons for failure, name the machine features behind them, and say how to
use that. Written up in full as research/proof/failure_anatomy.md; the substance:

The four stops and their causes.
  1. Silence costs the primorial. A mirror silences exactly the gears dividing its product, so to
     leave no live gear at or below X the product must be divisible by X#. PROVED this round:
     `silence_costs_primorial`, `primorial_le_of_silence` [proofs/MirrorWalkCarry.lean, round 73,
     0 sorries]. With the window's 2 M k <= q^2 this prices the carry wall exactly: linear gain,
     geometric cost.
  2. Composition intersects. Several mirrors land on the multiples of their gcd and carry only
     the gcd's gears (`gcdL_dvd_combo`, round 55) - carried sets can never be added.
  3. The trade. The mirror's product is both the price of carrying and the spacing of candidates
     (`mirror_times_candidates`, round 56), so one object does both jobs.
  4. Pigeonhole ends at the free-regime cut, which no mirror reaches (`free_regime_unreachable`)
     and where it fails the conclusion can fail (`keeping_move_free_sharp`), round 71. The cause
     is that each gear has exactly two teeth per candidate line - the dimension-2 sieve, which is
     also where parity stops the analytic route (entry 77).

The single lever. All four share one root: the machine's only lever on a gear is divisibility. A
gear either divides the landing and is silent forever, or is live and strikes two classes. I
tested whether the landings, being constructed, carry anything beyond their factorisation
(research/stack/r8/landing_structure.py): landings of the form 2^a 3^b - maximal algebraic
structure, two gears carried - against the next twin centre above each:

    mean smallest multiplier to the next landing: structured 38.4, control 15.2
    mean gears carried:                           structured 2.00, control 4.00

The structured landings are two and a half times worse, and the control's only advantage is
carrying twice as many gears. Shape does not help, divisibility does. So the lever is exactly
one, and `silence_costs_primorial` prices it.

What a proof must do, derived from the four: (1) not silence its way out, since silencing below X
costs X# against a window of q^2, capping the silenced gears at about 2 log2 q of pi(q); (2) not
count strikes, since the live gears have divergent reciprocal sum and counting can only win
through cancellation, which is the sieve, which is parity; (3) name a candidate rather than a
population - none of the four stops touches a reason why ONE named candidate is open, and such a
reason cannot come from the gear set; (4) face the pair as one object, since the pair at N is the
factorisation N^2 - 1 = (N-1)(N+1) and every mechanism the machine has treats the members
separately as two teeth.

Requirement (3) is the one no route has met, and (1), (2), (4) say why the obvious substitutes
fail. That is the shape of the hole, drawn tightly enough to recognise a genuine idea - or a
re-run of rounds 40 to 72 - immediately.

### 79. Requirement three attempted: what arithmetic can name (loop, 2026-09-18)

Entry 78 left one requirement unmet: a proof must name a candidate rather than a population, with
the reason coming from the candidate's own arithmetic. This round establishes what arithmetic can
and cannot name here.

Arithmetic names candidates CLOSED, freely. The sharpest instance, now in the kernel
[proofs/LandingForms.lean, round 74, 0 sorries]:

  `perfect_power_landing`: the only landing that is a perfect power is 4, the pair (3, 5).

The proof is the shape of every argument of this kind. If the landing is x^k with k >= 2 then
x - 1 divides the lower member, so it is composite unless x = 2; then the lower member 2^k - 1
being prime forces k prime, and for odd k the upper member 2^k + 1 is divisible by 3
(`three_dvd_two_pow_add_one`). Only k = 2 survives. Supporting: `not_prime_of_pow` removes the
other obvious shape, a member that is a proper power.

Arithmetic names candidates OPEN only up to the wall, and it has exactly two ways of doing it -
which turn out to cost the same.

  * By divisibility: a candidate 2 M k +- 1 is coprime to every gear dividing M, by construction
    (`mirror_gear_never_strikes`). Silencing every gear up to X this way costs X#
    (`silence_costs_primorial`, round 73).
  * By congruence: instead of making gears divide the landing, choose the period so the candidate
    misses their teeth. PROVED this round that this is the same kind of object:
    `openness_periodic` [proofs/MirrorWalkCarry.lean] - whether a column's members escape a gear
    set depends only on the column modulo the product of that set. So a congruence choice over the
    gears up to X names a class of period X#, and to meet the window it needs X# <= q^2 - q.

Same price, different mechanism. That matters because the carry wall could have looked like an
artifact of insisting on divisibility; it is not. Both levers are bounded by the primorial against
a quadratic window, so both reach only the gears up to about 2 log2 q.

So the position on requirement three is exact: arithmetic can name any number of candidates as
closed, and can name a candidate open only to the gears below a primorial that fits the window.
The gears from there to sqrt(N) - which is all but logarithmically many - have no naming
mechanism at all in this machine. That is the hole, and it is now described by what is in it
rather than by what has failed.

### 80. The run bound: the window statement from j2, and the exponent that is missing (loop, 2026-09-18)

Requirement 3 asks for a mechanism naming a candidate open against the gears between the
primorial's reach and sqrt(N). There is exactly one classical object that does that, and this
project has already worked on it: the paired Jacobsthal function j2, the longest run of columns in
which none escapes every gear of a primorial. A bound on j2 names an open column in EVERY run of
that length - no counting, no sieve cancellation, no choice of mirror.

PROVED this round [proofs/JacobsthalWindow.lean, round 75, 0 sorries]:

  `window_of_column_gap`: if every run of J consecutive columns holds one that no gear of the
  machine strikes, and a run of J columns starting at the machine's top gear still ends below
  q^2, then the window holds a twin prime pair.
  `window_statement_of_gap_law`: the same for every machine, from a uniform run law.

So the window statement follows from a bound of the shape j2(q#) < q^2 - q. The project's own
ladder (docs/novel, j2-upper-bound) proves j2(p_n#) << p_n^(4.266+eps) by the fundamental lemma,
after the elementary 3^(n+1) log^2 p_n rung and a quasi-polynomial one. The window needs exponent
2. The same ladder carries the ceiling: exponent 2 sits below Selberg's conjectural floor
2 kappa = 4 for dimension-2 sifting, so the missing factor is parity, not technique - the same
wall as the analytic route (entry 77).

MEASURED (research/stack/r8/run_gap_window.py) - the truth inside the windows themselves:

       q     window columns    longest struck run    run / (ln q)^2    open columns
     101          1,684                34               1.60              202
     251         10,459               104               3.41              818
     503         42,085               153               3.95            2,585
    1009        169,512               241               5.04            8,278
    2003        668,335               251               4.34           26,870

The longest run actually occurring is a few hundred columns against windows of hundreds of
thousands - four orders of magnitude of room, growing - and tracks (ln q)^2 rather than any power
of q. So the hypothesis of `window_of_column_gap` is not merely true but enormously slack at every
size measured. What is missing is a proof of a bound at exponent 2, and the sifting limit says no
sieve gives one.

This is the sharpest statement of the whole position: the implication from a run bound is proved
and cheap; the run bound needed is exponent 2; the best proved is 4.266; the floor that any sieve
can reach is 4; and the truth is polylogarithmic.

### 81. What residues can never decide: the machine's own parity statement (loop, 2026-09-18)

Every mechanism the machine has names a column by a residue class. Carrying names the class of the
mirror's multiples; the period rule names a class modulo the product of the gears it dodges
(`openness_periodic`, round 74); a walk of flips names the class of the gcd's multiples
(`gcdL_dvd_combo`, round 55). That is the whole toolkit, and this round proves it cannot finish -
with no measurement and no appeal to the literature.

PROVED [proofs/ClassIndistinguishable.lean, round 76, 0 sorries]:

  `class_has_closed_column`: for any modulus M, any residue r, any gear p dividing neither 6 nor
  M, and any bound N, there is a column m >= N with m congruent to r modulo M whose lower member
  6m - 1 is divisible by p and larger than p.
  `no_class_of_twins`: hence that column is not a twin column.

So no statement of the form "the columns of this class are twin columns" is ever true, for any
class of any modulus, however large. The construction that produces the witness is explicit: solve
6m congruent to 1 modulo p alongside m congruent to r modulo M, which is possible because 6M is
invertible modulo p, then push the solution past any bound by adding multiples of M p.

What this settles. Requirement 3 of the anatomy asked for a mechanism naming a candidate open
against the gears the primorial cannot reach. This says the machine's mechanisms cannot name a
candidate at all - only a class - and every class contains closed columns. The twin columns inside
a class are picked out by something the residues do not see. That is the machine's own form of the
parity obstruction, and it is now a theorem rather than an observation about sieve technique.

Taken with entries 78 to 80 the position is complete and closed on every side:
  * the four stops and their causes, with the single lever priced (entry 78);
  * the two levers - divisibility and congruence choice - shown to be the same object with the
    same primorial price (entry 79);
  * the run-bound route, whose implication is proved and whose input needs exponent 2 against a
    sifting floor of 4 (entry 80);
  * and now the impossibility itself: residues name classes, classes never separate twins
    (entry 81).

### 82. The alignment at infinity: which two of the three steps are true (loop, 2026-09-18)

The owner's observation: home is open because at 0 every gear sits at residue 0, and the two
neighbours are the one place no gear can reach - a gear striking them would divide 1. Carried to
the limit: the primorial of all gears would put every gear at 0 at once, so its neighbours would
be open to everything, hence a twin.

Two of the three steps are true, and both are now in the kernel
[proofs/AlignmentLimit.lean, round 77, 0 sorries]:

  `open_columns_for_any_gears`: for EVERY finite set of gears and every bound, there are columns
  beyond that bound open to all of them - take the column to be a multiple of their product, so
  each gear divides 6m and therefore misses 6m +- 1. This is the alignment at every finite level,
  and it is exactly the primorial construction.

  `no_column_open_to_all_gears`: NO column is open to every gear - the lower member is above 1,
  so it has a prime factor, and that factor is a gear striking it.

So the alignment exists at every finite level and nowhere in the limit: the quantifiers do not
commute. "For every gear set there is an open column" is true; "there is a column open to every
gear" is false. The window statement lives between them and is stated in the file as
`WindowStatement`: for every machine q, some column in (q, q^2] is open to the gears up to q. The
gear set is fixed by the column's own size, and that is the pattern neither of the two facts
supplies.

The measured form of the same thing (entry from the previous exchange): the pair either side of
the primorial is open to every gear up to P by construction, but its members are of size about
e^P, whose primality is decided by the gears up to e^(P/2) - a set that is not aligned. It is a
genuine twin only at P = 3, 5 and 11 up to P = 53; at P = 7 the lower member is 209 = 11 x 19, and
at P = 13 the upper is 30031 = 59 x 509.

So the intuition is sound and it is the reason the construction works at all; what it cannot do is
close the gap, because closing it would need the alignment to hold for a gear set that grows with
the number it is aligning.

### 83. Treating infinity as a number: what each system actually gives (loop, 2026-09-18)

The follow-up question: if infinity were treated as a number, would the alignment argument be
true? The answer depends on which system, and none of them makes it true for free.

  * Extended reals, or naive infinity. There, infinity minus one and infinity plus one are the
    same object as infinity, so the pair collapses to a single element and primality is not
    defined on it. The statement becomes meaningless rather than true.
  * Profinite integers. Zero does have every gear at residue zero and the neighbours are units,
    exactly as the observation says. But that is the local picture, and it is the sieve's own
    picture: it says there is no congruence obstruction to twins at any modulus - which is why the
    conjecture is believed and why the heuristic densities are positive. It says nothing about
    integers, because a profinite element is a coherent system of residues, not a number with a
    size or a factorisation.
  * Nonstandard integers (hyperintegers). This is the system where infinity IS a number, and the
    transfer principle applies: a first-order statement holds of the standard integers exactly
    when it holds of the nonstandard ones. The twin prime conjecture is equivalent to the
    existence of an infinite hyperinteger H with H - 1 and H + 1 both hyperprime. So the
    nonstandard version is not easier, it is the same statement.
  * And the alignment fails there for the same reason it fails here. By transfer the primorial
    function extends, so there is a hyperprimorial of an infinite hyperprime, and its neighbours
    are coprime to every standard prime and to every hyperprime below it - but they still have a
    hyperprime factor above it. The finite shadow of that is proved this round:
    `aligned_neighbour_factor` [proofs/AlignmentLimit.lean, round 77, 0 sorries] - if no gear up
    to B divides n and n > 1, then n has a prime factor above B. Alignment never removes the
    factor, it only pushes it above the aligned set.

What converts "the factor is above B" into "there is no factor" is the square-root rule, and it
applies only below B squared. That is the window, and it is why the whole problem is the window
rather than the alignment.

### 84. A top number, and its mirror side: why the exercise defeats itself (loop, 2026-09-18)

The owner's thought experiment: zero was accepted as a number and the negatives are the concept on
its far side, so by the same standard admit a top number and a mirror side beyond it; then the
neighbours of that top number would be open to every gear, with no new gears above it, and a twin
would exist there. And the owner's own observation: that would also stop the positive integers, so
the mirror-side member would not be a prime at all, and the conjecture would come out false.

Both halves are right, and the reason is worth recording because it applies to every attempt of
this shape.

**Adjoining zero and the negatives keeps the arithmetic; adjoining a top number does not.** The
extension from the counting numbers to the integers adds inverses and keeps every law - addition,
multiplication, divisibility, primality all survive unchanged. Every extension that adds a top
element loses one of the three things the machine needs:

  * the one-point extension (the projective line) makes the two neighbours of the top the same
    object, so the pair collapses;
  * the ordinals give a top-like element with a successor but no predecessor - there is no "omega
    minus one" - so the lower member does not exist;
  * the surreals and other fields give both neighbours but make every nonzero element invertible,
    so no element is prime and primality is vacuous;
  * the nonstandard integers keep everything, and by transfer also keep Euclid: there are
    hyperprimes above every infinite element, so "no new gears above it" is false there too.

**"No new gears above the top" is exactly what cannot be arranged.** PROVED, trivially and for the
record: `gears_above_every_bound` [proofs/AlignmentLimit.lean, round 78] - above every bound there
is another gear. The successor that builds the gear set forbids a last gear, and that is Euclid.

**And the self-defeat the owner spotted is the general shape.** If the positives stop at the top
number, the successor axiom fails, and the successor axiom is what generates the gears whose
alignment the argument was using. A framework with a largest number cannot also be the framework
that has the primes. So the exercise decides the answer by choosing the axioms, in either
direction, which is why it cannot be a route to a proof or a disproof.

The salvageable content is the part already proved in entry 82: the alignment is real at every
finite level, and the quantifiers are what fail in the limit.

### 85. Infinity as an adopted number: the systems that already do it, and what each costs
(loop, 2026-09-18)

The owner's point: zero also needed allowances - division by zero is undefined - and was adopted
anyway; so infinity could be adopted as a number, with signed elements beyond it, and n/0 would
become defined rather than undefined.

The instinct matches mathematics that exists. Three systems already treat infinity as a number in
close to the way described, and all three are rigorous:

  * **The projective line (Riemann sphere).** Exactly the proposed sharpening: n/0 is a genuine
    element, infinity. The cost is that the far side collapses - infinity plus twelve is infinity,
    so there is no distinct element beyond it, and the pair either side of infinity is one point.
  * **The surreal numbers.** Here omega is a number, and omega + 5, omega - 12, omega / 2 are all
    distinct - precisely the signed far side proposed. Arithmetic is kept in full: the surreals are
    an ordered field. The cost is that every nonzero element is invertible, so no element is prime
    and primality becomes vacuous. Also, omega being invertible does NOT make 1/0 defined.
  * **The hyperreals and hyperintegers.** Infinite elements, full arithmetic, and in the integer
    version primality still means something. The cost is the transfer principle: every first-order
    truth carries over, including Euclid, so there are primes above every infinite element, and the
    twin question there is equivalent to the twin question here (entry 83).

What none of them can do is make n/0 a number while keeping the ring laws, and that is a theorem
rather than a tradition. PROVED for the record: `zero_not_invertible`
[proofs/AlignmentLimit.lean, round 79] - in any ring with 1 different from 0, no x has 0 * x = 1,
since 0 * x = 0. Defining n/0 to be a new element therefore does not remove the obstruction; it
moves it, and the system must give up one of the laws. Wheel algebras do exactly that: they define
division by zero and pay by weakening subtraction, so x - x = 0 no longer holds everywhere.

So the analogy with zero holds, with one asymmetry worth stating. Adopting zero cost one operation
at one point, and everything else survived. Adopting a top element costs either the far side
(projective), or primality (any field, including the surreals), or nothing at all except that the
question transfers unchanged (hyperintegers). In no case does the twin statement become easier,
and in the only case where primality survives it becomes literally the same statement.

### 86. Single primes versus pairs: nothing blocks the first (loop, 2026-09-18)

Asked what blocks proving that the machine generates primes infinitely. The answer is that nothing
does - the machine proves it, and the proof is the alignment argument itself.

  * Euclid IS the alignment argument. The neighbours of a primorial are open to every gear up to
    that bound, so any prime factor they have lies above it (`aligned_neighbour_factor`, round 77).
    That is exactly Euclid's step, in the machine's vocabulary, and `gears_above_every_bound`
    (round 78) records the conclusion: above every bound there is another gear.
  * The window version for single primes is also a theorem, and a wasteful one. PROVED this round:
    `window_has_prime` [proofs/AlignmentLimit.lean, round 80, 0 sorries] - every machine's window
    (q, q^2] holds a prime, by Bertrand, which already places one inside (q, 2q].

So the one-member statement is settled with room to spare, and the two-member statement is the
conjecture. The whole difference is one number:

    a gear strikes ONE class of a single number, and TWO classes of a pair.

Everything downstream follows from that 1 against 2:
  * the free-regime cut sits at gears above the number of gears in the single case (one residue
    per gear, so n gears exclude at most n positions - the same pigeonhole as `mexS_le` with one
    residue instead of two; no kernel lemma carries that exact single-tooth form, and `mexT_le` is
    the THREE-residue twin-candidate walk, not the single one) and at twice that in the pair case
    (`keeping_move_free`);
  * the reciprocal-sum deficit is a factor of log in the single case, which survives, and a factor
    of log squared in the pair case, which does not;
  * the sifting dimension is 1 against 2, and the parity obstruction bites only at dimension 2.

The machine is therefore not silent on infinitude - it proves the single-prime form of every
statement it can express. What it cannot do is carry two members at once, and that is the same
1-against-2 that stops Chen at almost-primes, stops the Maynard-Tao line at 6 under the strongest
hypotheses, and stops our run bound at exponent 4 instead of 2 (entry 80).

### 87. The counterexample hunt, asked structurally (loop, 2026-09-18)

New exercise: instead of proving the machine works, look for what could stop it - a machine whose
window has every twin slot killed. Asked structurally first, because that question has an answer.

**A total kill is impossible, and cheaply so.** Each gear takes two classes of columns, so over a
full period the columns no gear touches have density the product of (1 - 2/h) over the gears, which
is positive. The uncovered set is never empty; it is a union of classes modulo the primorial. So
any counterexample must be LOCAL - the uncovered columns all pushed outside one window - which is
exactly the run-length question of entry 80.

**So the sharp question is the adversarial one:** if the two classes of every gear could be CHOSEN
rather than being fixed by arithmetic, could they be arranged to cover a whole window? Measured by
greedy adversary (research/stack/r8/adversarial_window.py), gears in increasing order, each taking
its two best classes:

       q    columns in window    uncovered with chosen teeth    twins actually present
      29           136                     12  (8.8%)                 29  (21%)
      37           222                     19  (8.6%)                 41  (18%)
      47           361                     22  (6.1%)                 61  (17%)
      59           571                     39  (6.8%)                 87  (15%)
      71           829                     55  (6.6%)                121  (15%)
     101          1684                     93  (5.5%)                202  (12%)
     149          3676                    196  (5.3%)                365  (10%)

Two readings. The adversary never manages a complete cover at these sizes, so the window is not
merely lucky - even chosen teeth leave 5 to 9 percent of it open. And the machine's own teeth leave
about twice as much room as the best arrangement greedy finds, so the configuration the residues
actually take is noticeably WORSE at killing twins than a deliberate one would be. Both fractions
fall slowly with q, which is what a proof would have to control.

(Correction of framing, on the owner's note: the comparison is not structure against arithmetic.
The machine's structure IS the arithmetic of the actual residues, looked at a different way - the
same object, manipulated differently. What the experiment compares is two CONFIGURATIONS of the
same residues: the one the integers actually take, where each gear's two classes are fixed by the
same consecutive line every other gear reads, and an arbitrary one, where each gear's classes are
chosen independently of the rest. The difference measured is rigidity against independence, and
the rigid configuration is the weaker killer.)

**What the adversarial question is, named.** Covering a run of consecutive columns with two classes
per gear is the extremal problem behind the paired Jacobsthal function - the same object as entry
80. Its known bounds straddle the window. For the ONE-class function (Jacobsthal proper)
Rankin-type constructions give runs much longer than q and Iwaniec's upper bound is about q^2, so
that function sits at the window's own exponent; for the TWO-class paired function the best proved
upper exponent is this project's 4.266 (docs/novel, j2-upper-bound) and the lower constructions are
at least as long as the one-class ones. Either way the window at q^2 lies between what is known
from below and from above, so the question sits exactly at the exponent-2 knife edge that entry 80
identified, and no amount of rearranging the gears settles it either way.

**What this exercise rules out.** It rules out the hope that some conspiracy of gears could be
exhibited - a total cover is impossible by density, and a local cover is the Jacobsthal problem, so
a counterexample is not something one constructs but something one would have to find. And it rules
in a measured fact worth keeping: the configuration the residues actually take is about half as
efficient at killing twins as an independently chosen one, at every size tested. That gap between
the rigid configuration and the free one is where a proof would have to live.

### 88. Review of rounds 60 to 81, and the hunt with prejudice (loop, 2026-09-18)

**Review.** Three inaccuracies found in the record and corrected in place:
  1. Entry 86 cited `mexT_le` as the single-tooth walk. It is the three-residue twin-candidate walk
     (`Res3`). No kernel lemma carries the exact one-residue form; the single-tooth cut at n is the
     same pigeonhole as `mexS_le` with one residue per gear instead of two. Corrected.
  2. Entry 87 conflated the one-class and two-class Jacobsthal exponents. The one-class function
     has Iwaniec's bound near q^2; the two-class (paired) function's best proved upper exponent is
     this project's 4.266. Corrected.
  3. The window's first column is off by one in three measurement scripts (run_gap_window.py,
     adversarial_window.py, chen_window.py use lo = q // 6 + 1, which admits the column whose lower
     member EQUALS q when q = 5 mod 6). Effect: one extra column and, at such q, one extra "twin"
     counted - e.g. q = 29 reports 136 columns and 29 twins where the window (29, 841] has 135 and
     28. Every conclusion drawn survives the correction; the new scripts this round use
     lo = q // 6 + 2. Not rewritten retroactively; recorded here.
Everything else checked holds: the certificate's stated range (11 to 275595263287044304869593048464768)
matches the theorem, the machines 5 and 7 below it have twins in their windows ((11,13) and
(17,19) both lie in (5, 25]), the chain condition numbers, the carry-wall table, the sharpness
witness at 370, and the kernel names cited in entries 78 to 87.

**Part A - a real adversary.** Round 81's greedy adversary was weak. Simulated annealing with six
restarts over the free configuration - each gear choosing two classes independently - to minimise
the uncovered columns of the window (research/stack/r8/adversary_anneal.py):

       q   columns   gears   best uncovered (anneal)   greedy   sum of 2/h
      29      135       8               8                 12       1.400
      37      221      10              13                 19       1.519
      47      360      13              18                 22       1.657
      59      570      15              31                 39       1.728
      71      828      18              42                 55       1.819
     101     1683      24              86                 93       1.959

The adversary improves on greedy by a third and still never reaches zero, at any size, even
though the sum of 2/h exceeds 1 everywhere - counting permits a cover and the free configuration
still cannot find one. So at these sizes the window is safe against every configuration the
annealer can reach, not only the rigid one the integers take. Whether an exact search would find a
cover at q = 29 (search space about 4.6 x 10^16, an ILP or SAT question) is open and would be the
next step if the free configuration is to be understood.

**Part B - the closest calls.** Every machine to 10^7 (664,575 of them), the distance from q to
the first twin above it (research/stack/r8/closest_call.py):

    largest distance:                1722 at q = 9923987   (6.6 x (ln q)^2)
    largest against (ln q)^2:        7.79 at q = 850349    (distance 1452)
    largest share of the window:     0.0545, at q = 11 (the twin 17 in a window of 110)

Above q = 11 no machine ever needs more than a vanishing share of its window - at 10^7 the worst
case uses 1.75 x 10^-11 of it. The worst distances grow like (ln q)^2 with a constant below 8,
which is Cramer's scale for gaps, and nothing in 664,575 machines comes within eleven orders of
magnitude of the window's edge.

**What the hunt with prejudice says so far.** No configuration reachable by search kills a window,
rigid or free; the closest real call is the smallest machine; the worst real distances sit on the
(ln q)^2 scale with the window at q^2. A counterexample would have to be a machine whose first
twin is more than q^2 away, against a measured record of 1722 at ten million. The hunt has not
found a crack, and it has found where the crack would have to be: in the free configuration's
extremal runs, which is the paired Jacobsthal question, at exponent 2.

Addendum to entry 88 - the growth of the record distance. The running records of D(q), the
distance from q to the first twin above it, over all machines to 10^7:

        q         D     D/(ln q)^2    edge q^2-q-D     D/edge
      347        72        2.10          1.2e5        6.0e-4
     2381       168        2.78          5.7e6        3.0e-5
    24419       498        4.88          6.0e8        8.4e-7
   187907       924        6.27          3.5e10       2.6e-8
   850349      1452        7.79          7.2e11       2.0e-9
  9923987      1722        6.63          9.8e13       1.8e-11

A power fit on the records above q = 1000 gives D about 0.17 (ln q)^3.4; against (ln q)^2 the
constant drifts up from 2 to 8 across the range, so the records sit between (ln q)^2 and
(ln q)^3, consistent with the Cramer-style expectation of (ln q)^3 for twins (density 1/(ln q)^2).
The edge grows as q^2, so the ratio D/edge falls like (ln q)^3 / q^2: by a factor of about 10^7
across the four decades measured, from 6 x 10^-4 at q = 347 to 1.8 x 10^-11 at q = 9923987.

### 89. The hunt with prejudice, continued: exact covers, poison machines, worst stretches
(loop, 2026-09-18)

Three attacks, each aimed where a counterexample would have to be.

**Attack 1 - exact search over the free configuration.** Annealing (entry 88) never covered a
window. This decides it exactly for small machines: can two classes per gear, chosen freely, cover
every column of (q, q^2]? Branch on the smallest uncovered column, capacity bound, bitmask columns
(research/stack/r8/exact_cover_search.py):

       q   gears   columns   exact answer                       fewest uncovered met   nodes
      11      3        18    no free configuration covers            3                   68
      13      4        25    no free configuration covers            1                2,115
      17      5        45    no free configuration covers            4              100,061
      19      6        56    no free configuration covers            4            7,634,683
      23      7        87    no free configuration covers   5   634,932,505 (708 s)
      29      8       135    still running in the background; the search grows about a hundredfold per gear, so this one is near a day in Python and is left to finish on its own

So for every machine up to 19 the statement "some column of the window escapes every gear" is
TRUE FOR EVERY CONFIGURATION OF THE TEETH, not only the one the integers take - it is a fact about
two classes per prime and the window's length, verified exhaustively. That is the strongest form of
the window statement, and it holds at the bottom of the range with no arithmetic in it at all. The
search grows about a hundredfold per gear, which is the frontier of this method.

**Attack 2 - poison machines.** The adversary's natural candidates: q just below a multiple of a
primorial, where the numbers k P# - j for small j carry small factors
(research/stack/r8/poison_machines.py). Every k from 1 to 7 for P# up to 23#, 49 machines:

    worst: D = 684 at q = 223092827 (just below 6 x 23#), which is 1.85 (ln q)^2

against the all-machine record 1722 at q = 9923987 (6.6 (ln q)^2). The poison machines are
ordinary - below the records, and mostly below 1 (ln q)^2. The reason is the alignment: the
neighbourhood of a primorial multiple is where small gears are AT zero, which is the opposite of
dangerous.

**Attack 3 - the machine's own worst stretches, exactly.** For the gears 5..P with their real
teeth the struck pattern has period the product of the gears, and within it a longest run of
struck columns - the paired Jacobsthal length of that gear set, in the machine's units
(research/stack/r8/extremal_runs.py):

     P      period      longest run (columns)    in numbers
     7          35             4                     24
    11         385             6                     36
    13        5005            10                     60
    17       85085            17                    102
    19     1616615            24                    144
    23    37182145            33                    198

A machine placed at the start of each stretch, at three recurrences each, reaches its first twin
at D between 76 and 538, never above 2.42 (ln q)^2 - ordinary again. And the rigid runs themselves
are tiny: 33 columns for the gears to 23, against a free configuration that (by annealing at
q = 29) can leave only 8 of 135 columns open. The rigid configuration is far weaker at building
long runs than a free one, which is the same fact as entry 87's "half as efficient", seen from the
run's side.

**Standing after three attacks.** No configuration reachable by search, rigid or free, kills a
window; the free configuration provably cannot for q up to 19; the adversary's natural candidates
are ordinary; the machine's own worst stretches are polylogarithmic. The crack, if there is one,
is above the exact frontier and inside the exponent-2 question, exactly where every other route
put it.

### 90. What would have to emerge for a window to fail, and whether it can (loop, 2026-09-18)

Asked to stop searching and instead say what properties a failure would need, then test whether
those properties can exist in any state of the machine.

**The failure condition, split at a cut.** Take any cut P below q. The gears up to P leave a set
of survivors in the window - the columns none of them strikes - and a failure needs the gears in
(P, q] to strike EVERY survivor. If each large gear's two classes fell on survivors in the same
proportion as on all columns (independence), the share of survivors they strike would be

    1 - product over P < h <= q of (1 - 2/h),

which is far below 1 whenever the cut is a power of q near 1: about 2 ln 2 / ln q at P = q/2.
So a failure REQUIRES the large gears' teeth to land on the small gears' survivors far more often
than their share - by a factor 1 / (that expression), which grows without bound as the cut rises.
That is the property that would have to emerge: a strong positive correlation between where the
big gears strike and where the small gears leave gaps, across the whole window at once.

**Measured on real machines** (research/stack/r8/failure_conditions.py), the share of survivors
the large gears actually strike, against independence, and the ratio a failure would need:

       q     cut     survivors    struck share   independence   actual ratio   ratio failure needs
    5003     q/2      149,392        0.126          0.155           0.81              6.4
    5003   q^0.75     252,016        0.482          0.435           1.11              2.3
    5003   q^0.5      556,473        0.765          0.743           1.03              1.35
    5003   q^0.25   1,787,502        0.927          0.920           1.01              1.09

(the same pattern at q = 1009 and 2003: ratios 0.81, 1.07, 1.02, 1.005 and 0.80, 1.08, 1.02,
1.005). The property is absent at every cut, and at the top cut the machine runs the OTHER way:
the gears in (q/2, q] are less efficient on survivors than their share, ratio 0.8, where a failure
would need 6.4.

**Why the top cut runs the other way - a proved constraint.** A gear h in (q/2, q] strikes a
survivor of the gears up to q/2 only through a member n = h k with k free of primes up to q/2 and
k < 2q; so k is 1 or a single prime in (q/2, 2q). PROVED: `top_gear_cofactor`
[proofs/FailureConditions.lean, round 84, 0 sorries] - if a gear in (q/2, q] divides a member at
most q^2 with no prime factor up to q/2, the member is the gear itself or the gear times one prime
in (q/2, 2q). The top gears cannot strike survivors freely; each strike on a survivor is a product
of two large primes, which is a rigid feature of the residues and not a count. (The count it
implies is not tight enough to forbid failure on its own, and counting is not the route; the point
is the mechanism: the top gears' teeth on the survivor set are pinned to the primes of (q/2, 2q).)

**So, can the required property exist in any state of the machine?** Two answers.
  * In the rigid configuration, the state is the residues of one integer, and the mechanism above
    pins the top gears' strikes on survivors to two-prime products; the measured efficiency is
    below independence at the top cut and within 11 percent of it elsewhere, with the requirement
    at 6.4 and 2.3. Nothing measured moves toward the requirement as q grows; the top-cut ratio is
    flat at 0.8 from 1009 to 5003 while the requirement grows like ln q.
  * In the free configuration - teeth chosen at will - the required correlation can be built for
    ONE cut by choosing each large gear's classes to hit survivors, but exact search (entry 89)
    shows that even then no cover exists for q up to 23, and annealing leaves 5 percent open
    beyond that. The reason is the same in both worlds: a large gear h has about q^2 / (6h) strikes
    to place and the survivor set is spread evenly over its h classes, so its best class beats its
    average class by a vanishing margin as q grows.

**Standing.** The failure property is now named - super-proportional efficiency of the large gears
on the small gears' survivors, growing without bound with the cut - and it is absent in every
measured state, reversed at the top cut by a proved mechanism, and not constructible by a free
adversary at any size reached. What is not on record is a proof that it is impossible at every q,
and that proof would be the window statement.

### 91. The share logic applied field by field (loop, 2026-09-18)

Asked whether any field, by the fields explorer's ids, can gain a member that enters a blocking
state. Apply the share logic of entry 90 to each.

**The reduction.** A window is blocked only if every column is struck. Order the gears: a column
is first struck by the smallest gear dividing one of its members, so a block is exactly the union
over g of the kills of higher:g (composites whose smallest gear is g) covering every column. So
the question is about higher:g, and the other fields are answered by how they feed it.

**multiples (row g at every multiple of g).** A rigid lattice: two classes of columns modulo g,
positions fixed by g alone (plus and minus the inverse of 6). On any union of R classes modulo a
modulus coprime to g, consecutive members step by a unit modulo g and cycle through all of g's
classes evenly, so the lattice takes its share 2/g of that set up to at most 2R strays. It cannot
concentrate on anyone's survivors. Verdict: cannot enter a blocking state; the share is exact up
to a bounded stray.

**squares (row g at g^2).** PROVED: `square_is_upper_member` [proofs/FieldBlocking.lean, round 85]
- a gear's square is 1 mod 6, so it is the upper member of exactly one column and never a lower
member. One column per gear in (sqrt q, q]: measured 158 columns at q = 1009, 651 at q = 5003,
against windows of 169,512 and 4.17 million. Verdict: cannot enter a blocking state.

**lower:g and lower1:g (composites whose largest gear is g).** PROVED:
`lower_on_survivor_is_power` - a member of lower:g that no gear below g strikes is a power of g.
So lower:g reaches the survivors of the smaller gears only at g^2, g^3, ... - at most log q
columns per gear - and everything else it strikes is already struck by a smaller gear. Verdict:
cannot enter a blocking state; on survivors it is the squares field and its higher powers.

**products:j (exactly j prime factors).** Every member is in exactly one higher:g, by its smallest
factor, so products:j adds no coverage beyond the higher fields; in the window j runs only to about
2 ln q / ln 5. Verdict: answered by higher:g.

**higher:g and higher1:g (composites whose smallest gear is g).** This is the field that does the
blocking, if anything does. It acts only on the survivors of the gears below g, and its kills
there against the lattice share 2/g of those survivors (research/stack/r8/field_blocking.py):

    q = 5003        gears   mean ratio   min     max     total kills
    g <= q^0.25         2     1.000      1.000   1.000    2,383,332
    q^0.25..q^0.5      15     1.001      1.000   1.004    1,231,029
    q^0.5..q^0.75      89     1.029      0.944   1.199      304,457
    q^0.75..q         562     1.001      0.000   1.532      121,473

(q = 1009: 1.000, 1.000, 1.035, 0.956.) Below sqrt q the field takes its share to three decimal
places; above it the per-gear ratio fluctuates only because the counts are small, and the band
mean sits at 1.00. For g above q^(2/3) every kill's cofactor is 1 or a prime (sampled, no
exception), which is `top_gear_cofactor` (round 84) in the field's own terms: higher1:g up there
is exactly g times a prime. Verdict: higher:g is the lattice of g restricted to rough cofactors;
it takes its share of the survivors and no more; for large g its members are pinned to g^2 and
g p with p prime.

**So no field, taken alone, has a member that can enter a blocking state.** Each is either a rigid
lattice at share 2/g (multiples, and higher:g on survivors), a single column per gear (squares),
the powers of one gear (lower:g on survivors), or a relabelling of the higher fields (products:j).
What a block would need is not a field with a special member but the fields' shares failing to
overlap - the many-body avoidance of entry 90 - and that is not a property any single field can
carry.

### 92. Pairs of fields: can any two combine into a blocking state? (loop, 2026-09-18)

The owner's framing: the fields are the many-body property split into distinct parts, so the next
part is pairs. For two gears g1 < g2, take the survivors of the gears below g1; each field kills
its own share of them (entry 91); a pair combines toward a block only if their kills AVOID each
other - overlap less than independence, so the union is larger than the shares alone give.

**The pair interaction, isolated** (research/stack/r8/field_pair_interaction.py). For 6000 random
pairs per machine, the columns of the survivors below g1 that both gears strike, against the
product of each gear's own measured kill rate on those survivors:

    q = 1009                  actual both-struck   independent    ratio
      both gears <= sqrt q          59,778           59,780.1     1.000
      one below, one above          79,836           79,693.5     1.002
      both gears > sqrt q            7,444            8,439.7     0.882
    q = 2003
      both gears <= sqrt q         160,247          160,246.2     1.000
      one below, one above         105,062          104,908.3     1.001
      both gears > sqrt q            7,819            8,795.7     0.889

Pairs with at least one gear below sqrt q overlap exactly as independence says, to four decimal
places over tens of thousands of coincidences. Pairs of two large gears overlap 11 percent LESS
than independence - they do avoid each other, and that is the direction a block needs.

**The mechanism of the large-pair avoidance, from the fields.** For g1, g2 above sqrt q, a column
both strike has (apart from the single column of g1 g2 itself) one member g1 p and the other
g2 p' with p, p' prime (`top_gear_cofactor` in each field), so the coincidence is a solution of
g2 p' - g1 p = 2 in primes. Two products of large primes two apart is a rigid pattern; its count
runs 11 percent below the independent product, flat from q = 1009 to 2003. So the avoidance is a
feature of the pair (higher1:g1, higher1:g2) with both cofactors prime, and it is bounded: it acts
on the overlap term, which for two large gears is about 4/(g1 g2) of the survivors - negligible
against the survivors themselves.

**The one exact pair.** PROVED: `square_lone_killer_iff` [proofs/FieldBlocking.lean, round 86] -
the square g^2 is the upper member of its column and the lower member is g^2 - 2, so the pair
(squares, everything else) combines on that column exactly when g^2 - 2 is composite. Measured:
45 of 158 squares at q = 1009 and 75 of 290 at q = 2003 are lone killers, the rest are joined by
another field on the lower member. Squares never combine with anything to block more than their
one column.

**What the first baseline showed, for the record** (research/stack/r8/field_pairs.py). Against the
pure lattice share the union ratios run 0.07 to 1.49, but that spread is the SINGLE-field
deviation of the large gears (entries 90 and 91: below share at the top, prime-cofactor structure
in the middle), not a pair effect; isolating the interaction removes it entirely for pairs below
sqrt q and leaves the 11 percent for pairs above.

**Verdict on pairs.** No pair of fields can combine into a blocking state. Pairs involving any
gear below sqrt q are independent to four decimals; pairs of large gears avoid each other by 11
percent, on an overlap term that is itself negligible against the survivors, through a rigid
pattern (two large-prime products two apart) that is the pair's own version of the cofactor
constraint. The many-body property, split into pairs, shows one bounded interaction and nothing
that grows.

### 93. Triples of fields (loop, 2026-09-18)

The next part of the many-body property: three fields at once. For gears g1 < g2 < g3, on the
survivors of the gears below g1, the columns all three strike are compared with two baselines
(research/stack/r8/field_triples.py, 20,000 triples per machine, stratified by how many of the
three gears lie above sqrt q):
  * independence of the three fields - the product of each gear's own measured kill rate;
  * the pairwise-consistent baseline, built from the three measured pair overlaps (Kirkwood's
    superposition), which removes what the pairs already explain and leaves the pure three-body
    term.

    q = 1009   gears above sqrt q   triple-struck   vs independence   vs pairs   (pure three-body)
                     0                2,466,244          1.000           1.000
                     1                  118,326          0.999           0.997
                     2                    5,478          1.032           1.032
                     3                       42          0.718           0.903
    q = 2003
                     0                5,255,834          1.000           1.000
                     1                  163,966          0.998           0.997
                     2                    5,373          1.026           1.023
                     3                       31          0.845           0.971

Reading it.
  * Triples with all three gears below sqrt q: no interaction at any level - independence and the
    pairwise baseline both at 1.000 over millions of coincidences.
  * One large gear: 0.998 to 0.999; two large gears: 1.02 to 1.03, a slight EXCESS of overlap,
    which is the wrong direction for a block.
  * Three large gears: against independence 0.72 and 0.85, but against the pairs 0.90 and 0.97.
    So most of the three-large avoidance is the pair avoidance of entry 92 counted three times,
    and the pure three-body term is small, shrinks from q = 1009 to 2003, and sits on counts of
    42 and 31 where the statistical noise is itself about 15 percent.

**Verdict on triples.** No triple of fields can combine into a blocking state. Everything a
triple does is explained by its pairs, to a tenth of a percent when any gear is small, and to
within noise when all three are large. The pure three-body term goes toward 1 with q, not away
from it. Nothing at the third level grows.

**The pattern across the levels.** Singles take their share (entry 91). Pairs are independent
except for an 11 percent avoidance between two large gears, on a negligible term, through the
rigid pattern of two large-prime products two apart (entry 92). Triples add nothing beyond their
pairs. The many-body property, split into its first three parts, shows one bounded interaction at
the second level and none at the third.

### 94. Quadruples of fields (loop, 2026-09-18)

Four fields at once, same construction (research/stack/r8/field_quads.py): for g1 < g2 < g3 < g4,
on the survivors of the gears below g1, the columns all four strike, against independence of the
four and against the triple-consistent baseline (fourth-order superposition from the four measured
triple overlaps, six pairs and four singles), which leaves the pure four-body term. 4000
quadruples per stratum, stratified by how many gears lie above sqrt q.

    q = 1009   gears above sqrt q   quad-struck   vs independence   vs triples
                     0                301,150         1.001           1.002
                     1                 15,925         0.997           0.999
                     2                    798         0.991           0.969
                     3                     27         0.879           0.349 *
                     4                      1         2.339 *         0.097 *
    q = 2003
                     0                533,471         1.000           0.999
                     1                 18,218         1.005           1.008
                     2                    600         0.955           0.960
                     3                     13         0.815           0.944
                     4                      0         0.000 *         0.000 *

(* below measurability: with three or four large gears the quadruple coincidences number 0 to 27,
and the triple-consistent baseline, being a ratio of products of near-zero triple counts, is
unstable there - at q = 1009 it predicts 77 against an independence value of 31 and an actual of
27, which is the baseline failing, not the fields interacting.)

Reading it.
  * Zero, one or two large gears: 1.00 to within 0.05 against both baselines, over hundreds of
    thousands of coincidences. No four-body interaction.
  * Three large gears: 0.88 and 0.82 against independence. Three large gears carry three large
    pairs, each avoiding at 0.89 (entry 92), so the pair avoidance alone predicts about 0.70; the
    quadruples avoid LESS than their pairs would, the same direction as the triples.
  * Four large gears: one coincidence at q = 1009 and none at q = 2003. Below measurability, and
    that is itself the finding: four large fields strike a common survivor column about once per
    machine.

**Verdict on quadruples.** No quadruple of fields can combine into a blocking state. Where the
counts are measurable the four-body term is absent; where the fields are all large the
coincidences vanish faster than any baseline can be estimated, because four large-prime cofactors
on two members two apart is a pattern that occurs about once per window.

**The pattern at four levels.** Singles at their share. Pairs independent, except 11 percent
avoidance between two large gears on a negligible term. Triples explained by pairs. Quadruples
explained by pairs and triples where measurable, and vanishing where all are large. Each level
above the second either adds nothing or makes the coincidences rarer. A block needs the opposite:
coincidences that thin out FASTER than independence at every level so the union grows, and the
fields do that only among the large gears, by a bounded amount, on terms that are already
negligible.

### 95. The many-body interactions, by logic (loop, 2026-09-18)

Asked, with the reminder to use logic rather than counting: what many-body interactions exist
that can kill the machine's ability to generate twins?

A body is a gear; an interaction among gears is a relation between where their teeth fall - a
relation among their phases at a column N = 6m with members N - 1 and N + 1. Going through the
mechanics, the machine has exactly five, and each is now a kernel statement or a named mechanism
[proofs/ManyBody.lean, round 89, 0 sorries]:

  1. **Member coprimality.** No gear strikes both members of one column, since it would divide
     their difference 2 (`no_gear_both_members`). A column's killers split into two disjoint sets,
     one per member. Two-body, bounded to one sentence, and it works FOR a block (no strike is
     wasted on a column already struck by the same gear) - but it caps at one member per gear.
  2. **Stacking is bounded by size.** Three gears whose product exceeds q^2 cannot all divide one
     member at most q^2 (`no_three_large_on_member`); with `top_gear_cofactor` (round 84), above
     q/2 at most one can. Large gears cannot pile onto a member. Works AGAINST a block, and it is
     a statement about size, not about any count.
  3. **Joint strikes are one class.** Two coprime gears striking chosen members of the same column
     do so on exactly one residue class modulo their product (`joint_strike_class`, the Chinese
     remainder theorem in the machine's words). So the lattices carry no interaction at all: over
     a full joint period, coincidences are exactly the product of the shares. This is the level at
     which the pair and triple measurements (entries 92, 93) came out at 1.000 for the small
     gears - not approximately, but because there is nothing there.
  4. **Truncation.** The window is shorter than a joint period, so a joint class appears in it a
     whole number of times, one more or one fewer than its share. This is the ONLY source of
     deviation between coincidence and share, and it is fixed, tuple by tuple, by the window's
     endpoints modulo the tuple's period - by q's residues - not by anything the gears do to each
     other. The 11 percent pair avoidance among large gears (entry 92) lives here, on tuples whose
     period exceeds the window.
  5. **The cofactor recursion.** A gear's strike lands on a survivor of the smaller gears exactly
     when its cofactor is free of them, so the kills of a field on the survivors are indexed by
     the rough cofactors, which are the survivors of a smaller machine. Self-similar; it is the
     descent of round 54 seen from the killer's side. It relates a gear to the set of smaller
     gears through the cofactor, and to no other gear directly.

**Which of these can kill?** None can coordinate gears across a window. Interactions 1, 2 and 4
are bounded per column or per tuple; 3 is exactly zero; 5 is a recursion into the same structure
one level down. A block would need a sixth kind of interaction: a relation between the residues
of ONE integer - the window's start q - modulo different gears, holding across enough gears to
arrange every tuple's truncation stray coherently. There is no such relation. The residues of an
integer below the joint period are free (CRT surjectivity, `class_has_closed_column` and
`open_columns_for_any_gears` are both instances); the only thing that distinguishes q's residue
vector from an arbitrary one is that q is small - below the joint period of the gears it must
coordinate.

So the whole many-body question reduces, by logic, to one statement: **can a small integer have
an adversarial residue vector?** Small means below the window's own scale, adversarial means its
truncation strays over the gears cover a run as long as the window. That is precisely the free-
configuration question of entries 88 and 89 - decided exactly and negatively for every machine up
to 23, unreachable by annealing beyond, and open in general as the exponent-2 question of entry
80. It is not a counting statement; it is a statement about which residue vectors small integers
can carry, and the machine's five interactions are silent on it because none of them looks at q
itself.

### 96. What the adversarial residue vector must contain (loop, 2026-09-18)

Asked what the window's residue vector would need to contain, or how it would need to behave, to
kill the machine.

**What the vector is.** The strike pattern of the gears on the column line is fixed by
arithmetic; the residue vector of the window's start only says where the window sits in that
pattern. For each gear h it fixes ONE number: the offset of the window's start from the gear's
first tooth. The gear's second tooth is not free - the two teeth are the classes m = +inv(6) and
m = -inv(6) modulo h, and their separation is fixed by h alone: three times it is -1 modulo h.
PROVED: `teeth_separation` [proofs/ManyBody.lean, round 90, 0 sorries]. So the vector has one
free entry per gear, the shift, with the tooth pair rigid; the window's start can slide the pair
but never open or close it.

**What it would have to do.** With the shifts (c_h) the killing condition reads: every offset t
in the window's length is congruent to c_h or to c_h + inv(3) modulo some gear h. Three things
follow by logic, without counting:
  1. The small gears' pattern is periodic and leaves holes - a union of classes modulo their
     product - so the large gears must pass through every hole. A large gear passes through a
     hole class evenly (interaction 3, `joint_strike_class`), so the vector must arrange the
     large gears' shifts so that, hole by hole, some large gear's tooth lands on each individual
     hole position: a matching of holes to (gear, tooth, occurrence), with every hole matched.
  2. By the exact identity open = main + signed strays (Legendre's identity over the tuples,
     exact, not an estimate), where main = W times the product of (1 - 2/h) is positive and each
     tuple's stray is fixed by the vector modulo that tuple's product, an adversarial vector is one
     whose signed strays over ALL tuples sum to exactly minus the main term. That is the behaviour:
     the truncation strays of every tuple, each bounded and each determined by the vector on that
     tuple alone, would have to cancel a positive number that none of them sees.
  3. The vector's entries are the residues of one integer, and CRT leaves them free; the only
     property that distinguishes a real window's vector from an arbitrary one is that the integer
     is smaller than the joint period. So "adversarial" is a property of the shifts alone, and the
     question is whether ANY shift vector - realised by any integer, small or large - covers.

**The exact search over the vectors that can exist.** The searches of entries 88 and 89 let both
teeth of each gear move freely and were therefore generous to the adversary. This one moves only
the shift, with the pair rigid (research/stack/r8/exact_shift_search.py, branching on the lowest
uncovered column, which some unfixed gear must cover through one of its two teeth):

       q   gears   columns   exact answer                       fewest uncovered   nodes
      11      3        18    no residue vector kills                  6                 19
      13      4        25    no residue vector kills                  3                213
      17      5        45    no residue vector kills                  7              3,009
      19      6        56    no residue vector kills                  6             35,883
      23      7        84    no residue vector kills                  8            525,427
      29      8       135    no residue vector kills                 14          7,958,761
      31      9       154    no residue vector kills                 14        138,157,831
      37     10       221    no residue vector kills                 21      2,598,322,871

The rigid search is a thousand times cheaper than the free one (q = 23: 0.5 s against 708 s), so
the exact frontier moves from 23 to 31 and beyond. The fewest uncovered columns any vector reaches
GROWS with the machine - 6, 3, 7, 6, 8, 14, 14, 21 - so the best adversary is falling further
behind the window, not catching up. Machine 37 took 2.6 billion nodes and 57 minutes; machine 41
would need about a day in Python and was stopped, so the exact frontier of the rigid search stands
at 37 until the search is compiled.

**So, in one sentence.** The vector would have to be a shift vector whose rigid tooth pairs cover
the whole window, equivalently one whose truncation strays cancel the positive main term exactly;
no such vector exists for any gear set up to 37, and the shortfall of the best one grows with q.

### 97. Pure powers in the field analysis - what was counted and what was not named (loop, 2026-09-18)

Asked whether the field analysis (entries 91 to 94) forgot the perfect powers.

**Counted, yes; named, only the squares.** The fields explorer's ids have no field for powers
beyond `squares` (row g at g^2). A higher power g^k, k >= 3, sits in three of the explorer's
fields at once - `higher:g` (smallest gear g, dividing more than once, so not `higher1:g`),
`lower:g` (largest gear g, not `lower1:g`) and `products:k` - and every kill measurement of
entries 91 to 94 tested only "some gear divides a member", so the powers were inside every count.
Two of the proved statements cover them without saying so: `lower_on_survivor_is_power` (a member
of lower:g on a survivor of the smaller gears is a power of g - squares AND higher powers) and
`top_gear_cofactor` (above q^(2/3) the only power that can appear is the square, since g^3
exceeds q^2). What was not stated is what the higher powers look like as members. Now it is.

**Which member a power occupies.** PROVED: `power_member_side` [proofs/FieldBlocking.lean, round
91, 0 sorries] - for a gear g >= 5, an even power is 1 mod 6 and is the UPPER member of its
column; an odd power is g mod 6, so it is the upper member for gears 1 mod 6 (7, 13, 19, ...) and
the LOWER member for gears 5 mod 6 (5, 11, 17, ...). Squares are always upper members
(`square_is_upper_member`); cubes and fifth powers of 5, 11, 17 are lower members. That is the one
way a pure power behaves unlike the square.

**Measured in the window** (all pure powers of gears with q < g^k <= q^2):

    q = 1009:   158 squares,  38 higher powers  (14 on the lower member, 24 on the upper);
                14 of the 38 higher powers are lone killers of their column
    q = 5003:   651 squares,  88 higher powers  (35 lower, 53 upper);  28 of 88 lone killers

Against windows of 169,512 and 4.17 million columns. The higher powers are 5^5 to 5^10, 7^4 to
7^8, 11^3 to 11^7, 13^3 to 13^5 and so on - one column each, and each only its own gear's column.

**Verdict.** The powers were in every count and in two of the proofs; the omission was the
label, and one fact about them - the member side of odd powers - which is now proved. Nothing
about them changes any verdict: like the squares they are one column per power, and a gear's
powers in the window number at most log base g of q^2, so the field of all pure powers together
holds fewer columns than the squares field it extends.

### 98. The owner's argument that no gear set kills forever, checked step by step (loop, 2026-09-19)

The owner's argument, in the machine's terms, and the standing of each step.

  1. A new prime gear's multiples alternate odd, even, odd, even, so only every second strike
     can land on a member (the members are odd). TRUE. On the column line this is built in: the
     gear's two teeth per cycle are exactly its odd multiples at the positions one either side of
     a multiple of six.
  2. No single gear kills forever, since it strikes only every second turning - on the column
     line, two classes of columns out of g. TRUE: a gear's share is 2/g (`oneflip_teeth`), and no
     gear strikes both members of one column (`no_gear_both_members`).
  3. The longest open run of a gear set is one less than its smallest gear. TRUE and now PROVED
     in both directions [proofs/AlignmentLimit.lean, round 92, 0 sorries]:
     `open_run_lt_gear` - any g consecutive integers contain a multiple of g, so no open run
     reaches g; `open_run_after_alignment` - right after any common multiple of the set, the
     next g - 1 integers are open to every gear, since a gear dividing one of them would divide
     an offset below itself. So the longest open run is exactly the smallest gear less one, and it
     recurs at every common multiple.
  4. Only 3 could shorten those runs to below a twin's width, and 3 is in the base, cannot kill
     the fold it created, and has no later twin among the primes (no later gear shares a factor
     with 3). TRUE: 2 and 3 are the only gears that could strike a member at every column, and
     the fold removes them from the members. Every gear above them strikes 2 of g columns.
  5. Therefore no finite set of gears above 3 kills every twin slot. TRUE and PROVED:
     `open_columns_for_any_gears` (round 77) - for every finite gear set there are columns open
     to all of them, as far out as one likes; they sit at the common multiples, which is step 3
     seen on the column line.

So the argument proves the infinite statement: no gear set, however large, can kill the twin
slots forever, and the reason is exactly the one given - after every common multiple the machine
returns to the home configuration and the next smallest-gear-less-one steps are open.

What it does not reach is the window. The open run guaranteed by step 3 sits at the common
multiple of the set, and for the machine q that multiple is the product of all its gears, far
beyond q^2 (the carry wall, `silence_costs_primorial`). Inside one window the set may or may not
leave a column open; that is the question the exact searches answer machine by machine (no
placement kills, to q = 37) and that no argument yet answers for every q. The distinction is the
quantifier of entry 82: for every gear set there is an open column (proved, and this is the
owner's argument), against for every machine there is an open column inside its own window.

### 99. What a gear to come can do, exactly (loop, 2026-09-19)

Asked to name the properties a future gear would need to kill a twin slot, specifically, so the
requirement can be checked against the machine's rules.

**The rule.** Let p < q be consecutive gears. The machine q extends the window from p^2 to q^2,
and the new stretch (p^2, q^2] is the only place the new gear could matter for twins that were
open before it arrived. PROVED: `new_gear_only_square` [proofs/StretchRule.lean, round 93, 0
sorries] - a member n in (p^2, q^2] divisible by q is either q^2 itself or already divisible by a
gear at most p. The reason is size: n = q k with k <= q; k = q is the square; k = 1 is q itself,
below p^2; and 2 <= k < q gives k a prime factor below q, hence at most p, which already strikes n.

**So a gear to come can do exactly one thing in the stretch it opens: close the column of its own
square.** Every other column the gears up to p left open in (p^2, q^2] is a twin prime pair of the
machine q, by the square-root rule. The "gears to come" cannot kill a slot near N; the slot is
decided by the gears up to sqrt N, all of which are present when the window reaches N. Nothing
later touches it.

**What a kill would therefore require, stated in the machine's rules.** For the window of machine
q to be twin-free, every consecutive-gear stretch (p_i^2, p_{i+1}^2] inside it must be twin-free,
and in each one the gears up to p_i would have to leave open nothing but the next square's
column - a stretch of about p_i x gap / 3 columns, covered by the gears below it with a single
exception. That is not a nebulous future gear; it is a specific configuration of the present
gears: their rigid tooth pairs covering a short stretch just above p_i^2 completely, except at one
column that the next square then closes. For a permanent kill this would have to happen at every
consecutive pair of gears beyond some point, without exception.

**Does the machine forbid it?** Not outright by the rules on record: the open-run law
(`open_run_after_alignment`) guarantees an open run only at common multiples, which lie far
beyond q^2, and the exact searches (no placement covers a window, to q = 37) speak about whole
windows, not single stretches. What the machine's rules do fix is the shape: the killers are the
present gears' teeth, nothing new is admitted, no property beyond size and residue is available,
and the one contribution of the arriving gear is its square.

**Measured** (research/stack/r8/stretch_twins.py): all 666 consecutive-gear stretches below
q = 5000. None is twin-free. The fewest twins in a stretch is 2, at the narrowest stretches
(p, q) = (5, 7), (11, 13), (17, 19), (29, 31) with 4 to 20 columns; the smallest twins-per-column
ratio is 0.0216 at p = 3539. The configuration a kill needs - a stretch covered completely by the
gears below it - has not occurred once.

**The question, sharpened one more step.** A twin-free window needs a twin-free stretch. So the
whole conjecture, in the machine's terms, is: for every consecutive pair of gears p < q, the gears
up to p leave an open column in (p^2, q^2] other than the square's. That is one stretch of about
p x gap / 3 columns against the gears up to p, with nothing else in play.

### 100. The hunt at one stretch: killers exist in the residue space, the machine never visits them
(loop, 2026-09-19)

The target is one stretch wide (entry 99): for consecutive gears p < q, can the gears up to p
cover (p^2, q^2] except the square's column? Three results.

**1. Under the strike law's own constraint, killer configurations exist.** The stretch starts at
p^2, so gear h's phase at its start is fixed by r = p mod h: the start column is (r^2 + 5) inv(6)
modulo h - the strike law of proof_skeleton section 12. The adversary at a stretch therefore
chooses a SQUARE residue per gear, one of (h-1)/2 values, not one of h. Exact search over those
(research/stack/r8/stretch_kill_search.py):

      p    q   gears   columns   some residue vector covers the stretch?
      5    7      1        4     no
     11   13      3        8     no
     13   17      4       20     no
     17   19      5       12     YES
     19   23      6       28     no
     23   29      7       52     no
     29   31      8       20     YES
     31   37      9       68     no
     37   41     10       52     YES
     41   43     11       28     YES
     43   47     12       60     YES
     47   53     13      100     YES
     53   59     14      112     YES
     59   61     15       40     YES

From p = 37 on, every stretch tested has a killer in the residue space; the narrow twin-gear
stretches (17-19, 29-31, 41-43, 59-61) have them earliest. So the protection of a stretch is NOT
in the shape of the constraint. Something in the residue model can kill.

**2. But each stretch has exactly one realisable vector, and it is p's own.** The gear set of a
stretch is the gears up to p with next prime q, and p is the only prime with that gear set. So the
residue vector at the stretch is not chosen: it is (p mod h) for h up to p, one point of the
space, fixed by p. Counting the killers against the whole space
(research/stack/r8/killer_fraction.py, exact enumeration with pruning):

      p    q   columns   residue vectors      killers    fraction     p's own vector kills?
     17   19      12            85,085           376    4.4 x 10^-3       no
     19   23      28         1,616,615             0    0                 no
     23   29      52        37,182,145             0    0                 no
     29   31      20     1,078,282,205     1,708,372    1.6 x 10^-3       no
     31   37      68    33,426,748,355             0    0                 no

The killers are a few tenths of a percent of the space where they exist at all, and p's own
vector is never among them.

**3. Measured over every stretch to 20000** (research/stack/r8/stretch_twins_sieve.py, 2260
consecutive-gear stretches by segmented sieve): none is twin-free. The fewest twins in a stretch
is 2, at the narrowest stretches (5-7, 11-13, 17-19, 29-31); among the stretches with p above
10000 the fewest is 147, at p = 10427; the smallest twins-per-column ratio anywhere is 0.0185, at
p = 19139. The number the machine has to beat is 1 twin per stretch, and it never comes within a
factor of a hundred of the edge once p is past 10000.

**What this says about the killers.** The hunt has found them: they are residue vectors of the
gears up to p that cover the stretch above p^2, they exist from p = 17 on, they are rare, and they
are not visited. The machine visits one point per stretch, p's own residues, and the question
"can a killer appear" is exactly "is p ever congruent, modulo every gear up to itself at once, to
one of the few killer vectors of its own stretch". That is a statement about where the primes sit
in the residue space of their own gears - the same object as the exponent-2 question, now with
the killers named and counted at small p.

### 101. Exploring the question: is a prime ever a killer of its own stretch? (loop, 2026-09-19)

The question from entry 100, taken apart.

**1. What "a prime's own vector" can and cannot be.** A prime p carries r_h = p mod h, which is
never 0 for a gear h below p and is 0 at the gear p itself. So a killer vector that uses a zero
residue below p can never be a prime's. Recounting the killers under that constraint
(research/stack/r8/killer_prime_compatible.py):

    stretch 17..19:  376 killers in the residue space, prime-compatible killers: 0
    stretch 29..31:  1,708,372 killers, prime-compatible killers: 48,896

At 17..19 every killer needs some gear to divide p, so no prime could ever have killed that
stretch - a clean structural exclusion at that one size. At 29..31 prime-compatible killers exist;
29's own vector (4, 1, 7, 3, 12, 10, 6, 0) is not one, and the nearest killer differs from it at
only 2 of the 8 gears. So nothing structural forbade a kill at 29..31: the prime that could have
done it would have had to be 29 with two residues changed, and there is no such prime, because 29
is the only prime with that gear set.

**2. The realisability constraint, stated.** A stretch's residue vector is realised by exactly one
integer: p itself. The killer set K_p is a set of classes modulo the product of the gears up to p,
and p is a killer exactly when its own class lies in K_p. There is one trial per prime, and the
trials at different primes live in different spaces. Nothing links them except size: p is smaller
than every product of two of its gears, so its residues are the residues of a small number - and
that is the only property the machine's rules give.

**3. How rare killers become** (research/stack/r8/killer_fraction_sampled.py, 200,000 random
prime-compatible vectors at each twin-gear stretch, the adversary's easiest case):

      p   columns   fraction of vectors that kill    -ln(fraction) / (p / ln^2 p)
     29       20          2.7 x 10^-3                       2.31
     41       28          6.2 x 10^-4                       2.48
     59       40          3.1 x 10^-4                       2.28
     71       48          2.5 x 10^-4                       2.13
    101       68          2.5 x 10^-5                       2.24
    107       72          5.0 x 10^-6                       2.49
    137 and up            none in 200,000                   above 1.7

The fraction falls like exp(-c p / ln^2 p) with c between 2.1 and 2.5, which is what covering
about 2p/3 columns at an open density near 1/ln^2 p predicts. The number of stretches grows only
like p / ln p. So, as a HEURISTIC and nothing more: the expected number of dead stretches over all
primes is a rapidly convergent sum dominated by p below a few hundred, where the exact and sieved
measurements have already found none. This is the Cramer-style reading of the machine, recorded
as such; it is not a proof and the owner's standing rule against counting proofs applies to it.

**4. What a proof would need, restated on this object.** Not that killers are rare - they are, and
that is a count - but that p's class never falls in K_p. The classes in K_p are arbitrary points of
the residue space; p's class is the point whose lift is p. A proof would need a property that
separates "the class of a prime of the gear set's own size" from the killer classes, for every p.
The machine's rules supply exactly one property of that class - its lift is below every product
of two gears - and nothing on record turns that into an exclusion. The 17..19 exclusion shows the
shape such an argument would have (every killer there needs a gear to divide p); it does not
persist at 29..31.

### 102. Structure, location, relative kill positions: the rules that govern a kill, and whether they forbid it (loop, 2026-09-19)

The owner's redirection: not counts, structure - what set of rules could kill the machine's
ability to continue, and are those rules achievable. Here is the rule set as it now stands, all of
it proved, and the honest answer to whether it forbids a kill.

**The rules a kill must obey, in the machine.**
  1. The rigid pair: each gear's two teeth are one welded part, one third of a turn apart
     (`teeth_separation`). The window's start slides the pair, never opens it.
  2. Coprimality: no gear strikes both members of a column (`no_gear_both_members`); a column
     dies by one strike on one member.
  3. Size: three gears whose product exceeds q^2 cannot share a member (`no_three_large_on_member`);
     above q/2 a gear strikes a survivor only through one large prime cofactor (`top_gear_cofactor`).
  4. Joint strikes are one class modulo the product (`joint_strike_class`): the lattices carry no
     interaction; only truncation by the window's edges deviates from the shares.
  5. The stretch rule: an arriving gear closes only its own square's column
     (`new_gear_only_square`); every slot is decided by the gears below its square root.
  6. The location law, NEW this round [proofs/KillPositions.lean, round 96, 0 sorries]: a gear h
     strikes the member p^2 + a only if -a is a square modulo h
     (`strike_after_square_isSquare`). So each position after a square admits killers only from
     the gears in the residue classes where -a is a square:
       - the square's own neighbour p^2 - 2: only gears congruent to 1 or 7 modulo 8
         (`square_neighbour_killers`);
       - the next column's lower member p^2 + 4: only gears congruent to 1 modulo 4
         (`next_lower_killers`);
       - the next upper member p^2 + 6: only gears with -6 a square; and so on down the stretch,
         one class condition per position, decided by the offset and not by the gear's size.

**The location law, read off real machines** (research/stack/r8/kill_positions_after_square.py):
at every position in the first eight columns above 29^2, 101^2 and 1009^2, about half the gears
are eligible (4 of 8, 11 to 15 of 24, 77 to 87 of 167), and every actual killer lies in its
eligible class, as it must. The square's neighbours: 839 is prime; 10199 = 7 x 31 x 47, all three
congruent to 7 modulo 8; 1018079 = 17 x 59887, with 17 congruent to 1 modulo 8. The law holds
exactly and it is a statement about position.

**Do these rules forbid a dead stretch?** No - and this is the point to be clear about. The
killer configurations found in entry 100 were built from square-residue shifts, which is
precisely the location law; they obey rules 1 to 6 in full, and they cover the stretch. From
p = 17 the residue space contains configurations that satisfy every rule on record and kill;
from p = 29 some of those are prime-compatible (no zero residue below p). So the mechanics as
identified do NOT make a dead stretch impossible. What keeps the machine alive at each stretch is
that its actual configuration is p's own residue vector, and no rule on record ties p's vector
away from the killers of its own stretch. The rules constrain WHICH gears can act at each
location; they do not constrain whether the one realisable configuration lands on a killer.

**What an impossibility proof would therefore have to contain.** A seventh rule: a relation
among the residues of one prime, p, against its own gears, holding for every p, strong enough to
exclude the killer classes of its stretch. Entry 95 showed the machine's interactions carry no
such relation - the residues of one integer are free apart from size - and the exact and sampled
searches confirm the rules on record permit killers. If the owner's claim that a permanent kill
is impossible given the mechanics is to be a theorem, that seventh rule is the theorem, and it is
not among the six.

**Standing after the location law.** The rule set governing a kill is complete as far as the
machine's mechanics are known, every rule is in the kernel, and the set is consistent with a dead
stretch. The machine's survival at every stretch measured is therefore not yet a consequence of
its rules; it is a fact about which point of its residue space each prime occupies.

### 103. How long is a dead stretch, and can it be infinite (loop, 2026-09-19)

**A dead stretch is finite by definition.** It is (p^2, q^2] for consecutive gears p < q: q^2 - p^2
numbers, about 2 p x gap, which is about p x gap / 3 columns. For twin gears (gap 2) that is
4p + 4 numbers, (2p + 2)/3 columns: 12 columns at 17..19, 20 at 29..31, 40 at 59..61.

**A dead run of a fixed gear set is finite, and proved so.** Every column that is a multiple of
the gears' product is open to all of them - the gear would have to divide 1
(`open_at_multiple_of_product`, proofs/StretchRule.lean, round 97, 0 sorries) - so no run of
struck columns can span a full period; every dead run of the gears up to p is shorter than their
product. The actual bound is far smaller: the longest run of the rigid teeth of the gears to P,
measured over a full period (entry 89), is 4, 6, 10, 17, 24, 33 columns for P = 7, 11, 13, 17,
19, 23 - the paired Jacobsthal length of that gear set.

**What a dead stretch would need, in those terms.** By the stretch rule the gears up to p must
do all the killing in (p^2, q^2] except at the square, so the rigid pattern of the gears up to p
must contain a run at least p x gap / 3 columns long, AND that run must begin exactly at p^2.
Comparing the two:

      P     longest rigid run (columns)    twin-gear stretch would need
      13             10                            9
      17             17                           12
      19             24                           13
      23             33                           16

From P = 17 the rigid pattern's longest runs are long enough to cover a twin-gear stretch. So
length is not what protects the stretch; POSITION is. The long runs sit at the pattern's extremal
phases, spaced a full period apart, and a dead stretch needs one of them to start at p^2 - a point
that is a square modulo every gear at once (the location law, entry 102). The killer search of
entry 100 shows such starts exist in the residue space; the measurements show p^2 has never been
one.

**Can a dead run be infinite?** Not for any fixed gear set - that is the theorem above. An
infinite dead run in the growing machine would have to be handed from stretch to stretch: the
gears up to p_1 kill (p_1^2, p_2^2], then the gears up to p_2 kill (p_2^2, p_3^2], and so on
forever, each handover being one of the coincidences of position just described, at every
consecutive pair of gears without exception. Each link is a finite event decided by finitely many
gears; the infinite chain of them is exactly the negation of the twin prime conjecture. So the
answer is: a dead run has a proved finite bound at every fixed level, an infinite one is not a run
of any gear set but an unending sequence of handovers, and the machine's rules bound each link
without forbidding the sequence.

### 104. How the later gears kill adjacent stretches: only in pairs (loop, 2026-09-19)

Asked how the gears up to subsequent primes would kill adjacent stretches, given that their kill
zones are predictable and interleave with the lower gears' slip and congruence.

**The mechanism, proved.** Fix a base p. A member above p^2 and below p^3 that no gear up to p
strikes is a prime, the square of a prime above p, or the product of two primes above p
(`rough_member_form`, proofs/StretchRule.lean, round 98, 0 sorries). The reason is size: its
least prime factor a exceeds p, so it is a, or a times a prime, or at least a^3 > p^3. So across
the stretches above p^2 the gears above p never kill singly. A later gear kills a column the base
could not only as one factor of a two-prime product, or as its own square. The stretch rule
(entry 99) is the first case of this - in (p^2, q^2] the only product available is q^2 - and this
is the general law up to p^3.

**What a dead run across adjacent stretches therefore is.** Take the base p_1 and the run from
p_1^2 through the stretches of p_2, p_3, ... up to p_1^3. The gears up to p_1 - a FIXED set - do
all the killing except at the columns whose member is a square p_i^2 or a product p_i p_j of two
primes above p_1. Those exception columns are the interleave points the owner describes: their
positions are set by the residues of the later primes against the base pattern, and for a product
the residue is multiplicative - p_i p_j modulo the base product is the product of the two
residues. So a dead run across k stretches requires:

    every column that the base pattern leaves open in the whole span carries a member that is a
    square or a two-prime product of primes above p_1.

The base pattern's open columns are a fixed periodic set (a union of classes modulo the product of
the gears up to p_1), recurring at every common multiple (`open_at_multiple_of_product`). The
requirement is that the two-prime products of the later primes land on every one of them, in
every period, using only primes that exist in the right ranges - a covering of a periodic set by
the products a x b of later primes, phase by phase.

**Why the base cannot do it alone, and the pairs are the whole story.** The base's own longest
run is the paired Jacobsthal length, finite and proved shorter than the period; so any run longer
than that must have its holes plugged, and by the theorem the only plugs are squares and pairs of
later primes. Squares are one column per prime and always upper members (`square_is_upper_member`,
`power_member_side`). So the plugs are, to within one column per prime, the products of two primes
above the base: the field higher1:g restricted to prime cofactors, the same object as
`top_gear_cofactor` (round 84) and the large-pair avoidance of entry 92.

**The interleave, made specific.** For the base pattern's open column at m and a later product
a b to plug it, a b must equal 6m - 1 or 6m + 1. Modulo the base product P, that reads
a b = 6m -+ 1 (mod P): the product of the two later primes' residues must hit the residue of the
open column's member. The later primes' residues are free (CRT), their products are free, but the
primes themselves must exist at the right sizes - a b must lie in the span - and there are only
about (span / ln) of them per size. A dead run is therefore a statement that the multiplicative
combinations of the residues of the later primes, restricted to the primes that actually exist in
each size range, cover every open residue of the base in every period of the span. That is the
structural content of "the later gears kill adjacent stretches", and it is exactly what no rule on
record forces or forbids.

### 105. The plan of attack on the four remaining killers, and its first two results (loop, 2026-09-19)

The four concepts not ruled out (killer residue vectors, two-prime products, multiplicative
interleave, truncation strays) now have a written plan: research/proof/killer_attack_plan.md -
for each, the statement, what is proved, the structural experiment paired with the lemma it aims
at, the proof route, and the stop criterion. Order of work 3, 2, 1, 4. Two results already.

**Concept 2, the plugs: the plug law, proved.** Let p_1 < p_2 < p_3 be consecutive gears. In the
stretch (p_2^2, p_3^2] a member that the gears up to p_1 leave open and that p_2 strikes is
p_2 k with k a prime at least p_3 (`plug_law`, proofs/StretchRule.lean, round 99, 0 sorries; the
size condition p_3^2 < p_2^3 holds for every consecutive pair from 5 on). So the newly established
gear plugs the base's holes only at products of itself with the next primes - p_2 p_3, and p_2 p_4
when that fits - a sparse set fixed in advance by the primes themselves. Across a run of stretches
the plugs are exactly the squares and the products of near-consecutive primes. Concept 2 reduces
to concept 3: whether those products land on the base's open columns.

**Concept 1, killer vectors: small lifts do kill short runs.** Scanning integers x rather than free
residue vectors (research/stack/r8/killer_lift_scan.py), so that every vector tested is the vector
of an actual small number: the squares x^2 that start a run of L columns fully struck by the gears
up to P, with L the twin-gear stretch length for P, x up to 2,000,000:

      P    L     hits    first hits
     17   12    4701    55, 158, 488, 715, 1423 (prime), 1467, ...
     29   20    2229    1638, 2117, 2947, 3272, ...
     41   28     971    1052, 1366, 6756, 10074, ...
     59   40     323    8899, 12561, 24922, 26987 (prime), ...
     71   48     281    24922, 30798, 33969, 41682, ..., 52951 (prime)

So a real prime's square does start covered runs of fixed length - 26987^2 begins 40 columns
struck by the gears up to 59 alone. The realisability barrier is not absolute: position permits.
What 26987 does not do is kill its own stretch, which is (26987^2, 26993^2], 53,980 columns long,
against the 40 it has covered. That is the whole of concept 1 in one line: a square can start a
covered run of any fixed length, and the length a kill needs grows like p x gap / 3 while the runs
a square can start grow only polylogarithmically (entry 88: the record first-twin distance to
10^7 is 1722, on the (ln q)^2 to (ln q)^3 scale). Reduction lemma to add next: a kill needs the run
above p^2 to reach the stretch's end, `kill_needs_run`.

**Standing on the plan.** Concept 2 is closed into 3 by the plug law. Concept 1 is position
permitted, length forbidden, and the length is the exponent gap - the same one as the run bound
of entry 80, now stated at the stretch: need exponent 1 in p, achieved polylog. Concepts 3 and 4
are next: 3 asks whether products of near-consecutive primes carry any relation to the base's
residues, and 4 is the accounting of the others.

### 106. Concept 3, the multiplicative interleave: it carries a relation, and the relation is not a machine rule (loop, 2026-09-19)

**The plug's position is fixed by the gaps.** By the plug law the plug of the stretch above p_2^2
is p_2 p_3, the product of consecutive primes. Writing p_2 = p_1 + g_1 and p_3 = p_1 + g_1 + g_2,
its offset from the base's square is p_2 p_3 - p_1^2 = p_1 (2 g_1 + g_2) + g_1 (g_1 + g_2), and its
offset from its own stretch's start is p_2 g_2. So the plug is not free: it sits where the two
gaps put it. Its column is open to the base exactly when its partner member p_2 p_3 -+ 2 has no
prime factor up to p_1 - modulo each gear h that is the quadratic (r + g_1)(r + g_1 + g_2) -+ 2 in
the base's residue r = p_1 mod h, and it has no algebraic factorisation for any gap pattern.

**Measured** (research/stack/r8/plug_interleave.py, plug_interleave_control.py; all 2260
consecutive triples with p_1 below 20000), the share of plugs whose column is open to the base:

    plug p_2 p_3, consecutive primes                    0.124
    p_2 times the fifth prime after it                  0.166
    p_2 times a random prime in (p_2, 2 p_2)            0.155
    free residues would give                            0.197

Two effects, and they are different. Against free residues every product falls short, because
the partner is about p_1^2 in size and must be prime or a product of two primes above p_1 to be
base-open (`rough_member_form`), which is rarer than avoiding one class per gear. That is the
size structure already on record. But the CONSECUTIVE product falls short of the non-consecutive
ones by a further quarter: 0.124 against 0.155 to 0.166. The residues of consecutive primes
against the base are correlated - a consecutive pair avoids the relation p_3 = -+ 2 / p_2 modulo
small gears less often than an arbitrary pair does.

**What that relation is.** It is the bias in the residues of consecutive primes (Lemke Oliver and
Soundararajan, 2016), whose explanation rests on the Hardy-Littlewood prime-tuple conjectures.
So the interleave does carry a relation between later primes and the base, and it is exactly of
the kind the plan asked about - but its origin is not a rule of the machine: it is a conjectural
distribution law about primes, the same family of statements as the twin conjecture itself. The
machine's rules give the plug's position from the gaps and nothing about the partner's residues
beyond CRT.

**Concept 3 closed, per the plan's stop criterion.** The multiplicative interleave is
Dirichlet-type distribution with a consecutive-prime correlation on top; no machine rule forces or
forbids the covering it would need for a kill. Concept 2 was reduced to it by the plug law, so
both now rest on the same outside object. Next: concept 1's reduction lemma.

### 107. Concept 1 closed: position permits, length forbids, and the length is the conjecture (loop, 2026-09-19)

**The reduction, proved.** `kill_needs_run` [proofs/StretchRule.lean, round 101, 0 sorries]: if
every column of the stretch (p^2, q^2] is struck, then the least open column above p^2 lies beyond
q^2. A dead stretch is a struck run starting at the square at least as long as the stretch, and
nothing else. With the stretch rule (the arriving gear adds only its square) the run is the base's
own run above its square.

**The run above the square, measured** (research/stack/r8/run_after_square.py, every prime to
200,000): R(p) = columns above p^2 to the first twin, against the stretch (q^2 - p^2)/6 a kill
needs.

    record runs:      p        R(p)    stretch needed    R / stretch    R / (ln p)^2
                    487          86           652          0.132           2.25
                   1523         138          4072          0.034           2.57
                   4637         264          3092          0.085           3.70
                   8699         401         23208          0.017           4.87
                  39953         411        239772          0.0017          3.66
                  80363         521        160732          0.0032          4.08
                 171233         703       1027452          0.0007          4.84

The largest share of a stretch ever covered from its square is 0.32, at p = 19 (9 of 28 columns).
Above p = 1000 the share never reaches a tenth; at 171233 it is seven parts in ten thousand. The
records R(p) sit between 2 and 5 times (ln p)^2 across the whole range, the stretch grows like
p x gap / 3, and the ratio between them falls by a further factor of ten with every factor of
about thirty in p.

**Verdict on concept 1.** A square can start a covered run of any fixed length - the vector of a
real prime does land on killer sets of fixed length (entry 105). What no prime's square has done
is start a run as long as its own stretch, and the reason is not position but growth: the runs a
square starts grow like (ln p)^2, the stretch a kill needs grows like p. A bound R(p) < stretch(p)
is the window statement for that stretch; proving it is the conjecture. Concept 1 is closed per
the plan: position permits, length forbids, and the length is the exponent gap - exponent 1 in p
against polylog, the stretch-level form of entry 80's exponent 2 against 4.266.

**Concept 4** was the accounting of the other three and is closed with them.

**The plan, closed.** Of the four killer concepts not ruled out, two (the plugs and the
interleave) rest on a conjectural distribution law about primes outside the machine's rules - the
consecutive-prime residue correlation and the placement of two-prime products; one (killer
vectors) rests on the growth exponent of the run a square can start; and one (strays) is the
identity that accounts for the others. No machine rule forces a kill, no machine rule forbids one,
and the machine's own survival at every stretch measured is a fact about the primes' residue
vectors and the run-length exponent, not a consequence of any rule on record. That is the same
conclusion as the anatomy of round 73, now reached from the killers' side with each killer named,
its mechanism proved, and its residual stated.

### 108. The owner's argument as the spine of the proof, with its one lemma named (loop, 2026-09-19)

The argument, four lines:
  1. the machine always generates twin gaps;
  2. sometimes a gap is blocked;
  3. no mechanic of the machine blocks the gaps permanently;
  4. therefore the machine generates twins without end.

Standing of each line in the kernel [proofs/OwnerArgument.lean, round 102, 0 sorries]:
  1. PROVED: open columns recur at every level (`open_columns_for_any_gears`,
     `open_run_after_alignment`, `open_at_multiple_of_product`).
  2. PROVED: a gap is blocked by an ordinary gear striking on one of its two cycles in six
     (`no_gear_both_members`, `top_gear_cofactor`, `rough_member_form`, `plug_law`,
     `strike_after_square_isSquare`).
  4. PROVED FROM 3: `twins_unbounded_of_survival` - if line 3 holds in the form below, then above
     every bound there is a twin prime pair, by the square-root rule.
  3. THE SURVIVAL LEMMA, the one object left:

        Survival: for every gear p with next gear q, some column strictly inside (p^2, q^2)
        escapes every gear up to p.

     Everything the search has produced bears on this and nothing else. It is line 3 in the exact
     form line 4 needs; every stretch measured satisfies it (entries 99, 100, 107); no rule on
     record proves it for every p.

From here the work is one thing: prove Survival from the machine's mechanics. The attack works
in the machine's own terms - the base pattern's open runs, the top gears' strikes as products of
primes straddling p, and the location law - and its first law is in the kernel:

  `product_kill_square_law`: a product of two primes straddling p, at distance a below and b
  above, is (p - a)(p + b) = p^2 + (b - a) p - a b, and (a + b)^2 = (b - a)^2 + 4 a b. So it lands
  on the offset o = (b - a) p - a b above the square, and (b - a)^2 - 4 o is the square (a + b)^2
  modulo p. A straddling product can reach an offset o only if (b - a)^2 - 4 o is a quadratic
  residue modulo p, with b - a at most about twice the gap for the product to fall inside the
  stretch. The top gears' kills in the stretch are therefore confined to the offsets whose
  discriminant against a small set of differences d = b - a is a square modulo p - a location
  law for the stretch's own top layer, alongside the one for its base (`strike_after_square_isSquare`).

Program for the next rounds, on Survival directly:
  (a) the base layer: the open runs of the gears up to B inside the stretch, their positions
      relative to p^2 by the location law, and which of them the location law leaves eligible to
      the top layer;
  (b) the top layer: the straddling products' reach, by the square law above, and the residues
      d^2 - 4 o that occur for the differences d the stretch admits;
  (c) the join: whether the two layers' eligibility sets can together cover the base's open runs,
      as a structural question about square residues modulo p and modulo the base gears - the
      same square-residue structure at both ends of the stretch.

### 109. The survival lemma on the primorial family: one fixed universal pattern, and the square is an ordinary position in it (loop, 2026-09-19)

Steps (a) to (c) of the program of entry 108, run on the primorial family, and what they gave.

**The reduction.** Take the family 30 t +- 1, the mirror {2, 3, 5} from home (t = 0 is home). A
gear h >= 7 strikes the family at t = +-inv(30) modulo h - positions fixed by h alone, the same
for every machine. So the gears 7..p lay ONE fixed pattern on the t-line, and the machine p only
selects the range [p^2/30, q^2/30] of it. PROVED: `survival_of_family`
[proofs/OwnerArgument.lean, round 103, 0 sorries] - if at every gear that fixed pattern leaves
some t of the range open to the gears up to p, the survival lemma holds, hence twins without end.
In the machine's terms the whole conjecture is now one property of one universal object:

    the struck run of the fixed pattern of the gears 7..p on the t-line, starting at the
    square's position t0 = p^2/30, is shorter than the stretch's span (q^2 - p^2)/30.

Base layer, top layer and their join all live inside that pattern: the base gears are its small
periods, the top gears its sparse teeth, the location law its phase at t0.

**Is the square a special position of the pattern?** Measured
(research/stack/r8/survival_family.py): the struck run from t0 = p^2/30 against runs from twelve
random positions in the same range with the same gears -

       p     stretch span in t   run at the square   mean random run   max random run
    1009           269                 18                  6.2               21
    2003          1070                  2                 10.4               31
    4001           533                 14                 15.5               45
    6007          1602                 17                 12.6               46
    8009          1068                 20                 18.0               69
   10007          1334                  2                 21.1               63
   15013          4004                101                 20.2               59
   19997         18670                  1                 24.9               54

The run at the square is an ordinary run of the pattern: inside the spread of the random ones
at every size, once above it (101 against a maximum of 59 at 15013) and often far below. The
square's residue structure does not shorten the runs, and it does not lengthen them. So the
location law, which fixes the phase at t0 by square residues, is not what protects the stretch;
the pattern's runs are short EVERYWHERE, and the square is one more place.

**What the pattern's runs are.** The runs of the fixed pattern of the gears up to p are the
paired Jacobsthal runs of that gear set in the family's units: measured here at 20 to 70 over
ranges of thousands, and at 33 columns over a whole period for the gears to 23 (entry 89). The
span a kill needs is (q^2 - p^2)/30, about p x gap / 15 - exponent 1 in p. So the survival lemma
on the family reads: the paired Jacobsthal length of the gears up to p, in family units, is less
than p x gap / 15. The truth measured is polylogarithmic; the strongest proved upper bound is this
project's exponent 4.266 (entry 80); the window version needed exponent 2 and the stretch version
needs exponent 1.

**Where the attack stands, plainly.** The owner's argument is now formalised end to end with one
lemma left, and that lemma is the statement that a universal, explicitly constructed pattern - the
teeth of the primes on the line 30 t +- 1 - never has a struck run as long as p x gap / 15 at the
position p^2 / 30. Every mechanism the machine has is a feature of that pattern; the pattern is
fixed; the square is an ordinary position. A proof is a bound on the runs of that pattern at
exponent 1, and the attack from here is on the pattern itself: its runs, their positions, and why
they stay polylogarithmic - which is the paired Jacobsthal question, in the machine's own units,
with the machine's own name on it.

### 110. The survival lemma placed in the work already on record (loop, 2026-09-19)

The owner is right that most of the ground under the survival lemma was covered earlier. Pulling
the record together against the lemma, with what it settles and the one new consequence.

**Already on record, and not to be re-derived.**
  * The reduction of the window statement to a run bound is proved (`window_of_column_gap`, entry
    80), and the exponent map is exact (research/proof/length_face.md): a bound F(q) < C q^2 with
    C < 1/6 IS the twin prime conjecture and more; C >= 1/6 or any exponent in (2, 4.27) implies
    nothing parity-sensitive.
  * The upper ladder for the two-class run record j_2 (docs/novel/j2-upper-bound.md, section 11a):
    elementary 3^(n+1) log^2 p_n; Brun quasi-polynomial p_n^(9.30 loglog p_n); explicit
    fundamental-lemma rungs at exponents 19, 17, 15 and the beta-sieve rung at 8.04162 (2G-inf
    7.93727); the non-explicit sifting-limit rung at 4.266. Fifteen is the least integer FI 7.7 can
    deliver at dimension 2; ODC's beta-sieve lower bound has constant B = 0 for kappa >= 1/2.
  * The ceiling, corrected in round 23: exponent 2 is the PARITY barrier, not an arithmetic fact
    about the sifting limit; 4.266 is what the best constructed dimension-2 sieve reaches, 4 is
    Selberg's conjectural optimum, about 1.47 is the best proved floor, and whether beta_2 < 4 is
    open and independent of parity.
  * The rigid record is strictly below the free one: F(M) - 1 <= A072753, equal at {5,7} only and
    strictly below from {5,7,11} on (87 against 117 at {5..37}) - research/proof/law_register.md.
  * Exact rigid tools: the record rule (F(M') is a maximum over g phases of gaps on one period of
    the lower machine, kernel-checked at 17), the loaded record rule (F_top(G) = max{L : minimum
    domino cost over core phase vectors <= tail count}, an iff, kernel `loaded_record_rule`), the
    position frontier (R_min >= 3L at 8375 cells, the mirror law at 88 of 88), and the record frame
    (DEAD as a route: one corridor, one word, size-two frame set and any nameable break decider all
    refuted). No location rule for the gap exists on the record (research/proof/location_rules.md).
  * Lower bounds, parity-free: h_2(P(z)) >= (1.349 + o(1)) z log z by greedy matching, and the
    layered Erdos-Rankin theorem j_2(P(x)) >= (1/(18 c_1) + o(1)) x A^3 C^2 / B^4, so any
    h_2 = O(z (log z)^a) forces a >= 3 (docs/novel/layered-erdos-rankin.md).
  * Measured laws: the window's longest struck run tracks (ln q)^2; the record distance to the
    first twin above q tracks between (ln q)^2 and (ln q)^3 to 10^7; the rigid worst stretches
    are 4, 6, 10, 17, 24, 33 columns for the gears to 7..23.

**The one consequence that is new, and it is decisive about method.** The survival lemma at the
stretch (entry 108) needs a run bound of exponent 1: no struck run of the gears up to p as long as
the twin-gear stretch, about 4p in integers, at the position p^2. The layered Erdos-Rankin theorem
on record says the FREE two-class pattern has runs of length x (log x)^(3 - o(1)) - longer than 4x
for large x. So in the free model the stretch-level survival is FALSE asymptotically: there exist
free residue vectors whose runs exceed a twin-gear stretch. Hence:

    no argument that treats the gears' phases as free can prove the survival lemma at the
    stretch; any proof at that level must use the rigid configuration - the actual teeth at
    +-inv(6), the actual residues of p - which is precisely the machine's own mechanism and
    nothing else.

The window-level statement (exponent 2, C < 1/6) is the one the free model does not refute
(Ziller-Morack Conjecture 6, open), and it is where sieve methods could in principle act and
where parity stops them at 4.266 against 2. So the two forms split cleanly:

    window form   - exponent 2 - free-model-compatible - blocked by parity for sieves;
    stretch form  - exponent 1 - free model FALSE      - needs the rigid mechanism.

The owner's thesis that the mechanism is everything therefore has an exact statement: the
survival lemma is provable, if at all, only from the rigid configuration, and the tools on record
for the rigid configuration are the record rule, the loaded record rule, and the position
frontier. What they give today are exact characterisations of F(M) and no bound of exponent
below 2; the record's own falsification target stands: one exact rigid record beyond q = 59
(F(59) is pinned to [161, 178]; the certified ladder ends at F(47) = 118), and one exact
h_2(p_n#) beyond p_n = 73.

**Program from here, using only rigid tools.** The loaded record rule is an exact iff: a run of
length L is coverable iff the minimum domino cost over core phase vectors is at most the tail
count. Its closed-form corollary F_top(G) <= Lcap(G) is the counting bound and too weak. The
rigid question is whether the ACTUAL core phase vector at p^2 - fixed by the location law, square
residues at every core gear - has domino cost above the tail count for every L reaching the
stretch's end. That is a statement about square residues against the domino cost, computable
exactly at each p, and it is the form of the survival lemma that lives entirely inside the kernel's
own rigid machinery.

### 111. The survival lemma weighed exactly: the weak form IS the conjecture, the strong form is more; entry 110's program withdrawn (loop, 2026-09-19)

**Two corrections to the record, both proved.**

1. `Survival` (entry 108) asks EVERY stretch (p^2, q^2) to survive. Step 4 of the owner's
   argument uses only that stretches above every bound survive. That weak form is now defined
   (`SurvivalInf`) and PROVED equivalent to twins unbounded, both directions
   [`survivalInf_iff_twins_unbounded`, proofs/OwnerArgument.lean, round 104, 0 sorries]:
     * forward: a surviving column of a stretch is a twin pair, by the square-root rule
       (`twin_of_surviving_stretch`);
     * converse: a twin pair (6m-1, 6m+1) far above N lies in the stretch of the largest prime
       p with p^2 < 6m-1 - the next prime q has q^2 > 6m+1 because q^2 is odd and composite and
       at least 6m-1; the pair escapes every gear up to p because its members are primes above
       p^2; and p > N by Bertrand's postulate.
   So line 3 of the argument, in the exact form line 4 needs, is not a reduction of the
   conjecture: it IS the conjecture, restated as "not every stretch beyond some point is dead".
   The strong form `Survival` - a twin pair between every two consecutive prime squares - is a
   strictly stronger statement, of the same kind as Legendre's conjecture on primes between
   consecutive squares (open), and is what entries 107 and 109 measured: true at every p to
   200,000, with the record run above a square at 2 to 5 (ln p)^2 against a stretch of order
   p x gap.

2. Entry 110's program (the loaded record rule's domino cost at the square-residue core phase
   vector) is WITHDRAWN. The rule's core/tail split needs a window shorter than the tail gears:
   core = gears <= L + 1, tail = gears > L + 1. The stretch has length q^2 - p^2 >= 2p + 1 in
   integers, so every gear up to p is core and the tail is empty; the rule reduces to "the core's
   uncovered set is nonempty", which is the survival statement itself. Taking a sub-window of
   length L < p inside the stretch makes the gears in (L + 1, p] the tail, but the tail count is
   then about pi(p), far above L/2, so the rule's necessary condition (cost <= tail count) holds
   trivially and decides nothing; only the tail gears' ACTUAL positions matter, and those are the
   question. The rigid record tools decide records of windows shorter than the gears; the stretch
   is longer than all of them. No route there.

**Where the target sits now, in one line.** The owner's argument is formalised end to end; its
line 3 is the twin prime conjecture exactly (weak form) or Legendre-type for twins (strong form).
The tactic changes: no further reformulation of the conjecture into a survival, run, record, or
covering statement can lower its weight - the record now has five such reformulations, each proved
equivalent or stronger (window <-> run bound of exponent 2; stretch <-> exponent 1; survival weak
<-> conjecture; survival strong -> conjecture; chain covering <- twin-gap growth t_{i+1} < t_i^2).
The remaining question is which mechanism of the machine yields a lower bound on open columns
that is not a sieve estimate, and the record's own answer is the one-flip / multiplicative chain:
a twin at column c gives the family of columns c x j whose gear residues are the residues of c
rotated by j - open iff j avoids two classes per gear - which is the two-class sieve on j over a
range of length c. The chain reproduces the problem at every twin; it does not lower its weight.

### 112. The owner's line 3 against the record's counter-machine: the mechanics do not force survival, and what a proof must therefore use (loop, 2026-09-19)

Before any new hypothesis, the record's own test of the owner's claim, made on 2026-09-11 and not
cited in the rounds since (research/proof/fold_mechanic.md, sections 2.1, 4.2, 6; scripts
research/anchor235/r78/fm_*.py).

**The counter-machine.** Take the gear set G = P_+ u (P_- \ W): every prime = +1 mod 6, and every
prime = -1 mod 6 EXCEPT the lower members of twin prime pairs. Let M_G be the monoid it generates.
On M_G every mechanic the owner's argument uses holds verbatim:
  * strikes are multiples (dilation), gears strike at their own teeth and nowhere else;
  * a prime slips every gear below it (the hand-up: the gears of a section are exactly the
    elements no smaller gear strikes);
  * the square-root rule: an element below the next gear's square with no gear factor below it
    is a gear;
  * a product kills once: each composite element has one least gear factor, and the strike is
    that factorisation;
  * both classes of gears at every scale, in every residue class of every modulus, with the
    primes' density.
And on M_G EVERY section is fully struck, the smallest being [49, 2809): 85 columns, 0 twin gear
pairs. Proved (the Lemma of 2.1): step 8 holds on a section of M_G iff the removed set keeps a twin
lower inside it; 0 mismatches over 7,309 sections to 10^7.

**What that decides for the argument of entry 108.** Lines 1 and 2 are theorems for every gear
set. Line 3 - no mechanic blocks the gaps permanently - is TRUE of the mechanics and FALSE as a
conclusion on M_G: the mechanics are all present there and the gaps are blocked at every section.
So line 3 in the form line 4 needs (SurvivalInf, entry 111) does not follow from the mechanics of
striking; if it did, it would hold on M_G. The argument's gap is not a missing mechanic of the
gears - the record has now checked every named one - but the one property M_G lacks: its line is
not all of S = {6j +- 1}. The column (t, t + 2) of a twin lower t is not a column of M_G at all.

**The admissibility criterion this gives, for every future round.** A proof of survival must
use that every j is a column of the real machine (the line's completeness, the strike-class law's
first half), and must use it in a way that is neither a free-phase cover nor a count. Test for any
proposed argument: delete the open columns from the line and rerun the argument; if it still goes
through, it is wrong, because its conclusion fails on M_G. The run-length form (entries 80, 107,
109) passes the test - a struck run shorter than the interval uses that the interval's columns are
consecutive - and its weight is the conjecture (entry 111). The pin at p = 29
(origin_mechanic.md section 3) passes it too - it is a fact about which multiples exist - and the
record found no general form of it.

**Consequence for the loop.** Rounds that rediscover a mechanic of striking cannot progress; the
record has closed that direction twice (fold_mechanic.md, entries 107-111). Next: state the
counter-machine in the kernel as an independence theorem - an abstract machine satisfying the
striking axioms, the real machine and M_G both models, survival failing on M_G - so the owner's
argument carries, as a theorem, the exact statement of what its line 3 must add.

### 113. The counter-machine in the kernel: survival is independent of the mechanics of striking (loop, 2026-09-19)

[proofs/CounterMachine.lean, round 106, 0 sorries]

**The abstract machine.** A LINE is a multiplicative set of fold survivors (numbers = +-1 mod 6)
containing 1. Its GEARS are its irreducibles; a gear strikes what it divides; its COLUMNS are the
pairs (6j-1, 6j+1) with both members on the line. `Line.SurvivalInf` is entry 111's weak survival
statement written for a line: above every bound, some stretch (p^2, q^2) between consecutive gears
holds a column of the line that no gear up to p strikes.

**The mechanics are theorems about every line.**
  * `Line.exists_gear_dvd`: every element above 1 has a gear factor;
  * `Line.gear_of_rough`: the square-root rule - an element below q^2 that no gear up to p
    divides (every gear below q being at most p) is a gear;
  * `Line.twin_of_surviving`: a surviving column of a stretch is a twin gear pair.

**The real machine is a line.** `real` = all of S = {6j +- 1}. `real_gear_iff`: its gears are
exactly the primes >= 5. `real_column`: every j >= 1 is a column. `real_survivalInf_iff`: its
survival statement is the kernel's `MirrorWalk.SurvivalInf`, i.e. the twin prime conjecture
(entry 111).

**The counter-machine is a line.** `counter` = the monoid generated by every prime = 1 mod 6 and
every prime = 5 mod 6 whose +2 is not prime (the twin lowers removed, fold_mechanic.md 2.1).
`counter_gear_iff`: its gears are exactly those generators. `counter_never_survives`: NO stretch
of it ever survives - a surviving column would be a twin gear pair, twin primes whose lower member
is a generator, and the lower member of a twin pair is never a generator.
`counter_not_survivalInf`: its survival statement is false.

**What the theorem says about the owner's argument.** Lines 1 and 2 (open columns recur; a gap
is blocked by an ordinary gear striking in the ordinary way) and the square-root rule hold on
every line, the counter line included. The counter line has every gear above 1 irreducible, every
prime slipping every lower gear, every composite killed by its least gear once, both classes of
gears at every scale. And it never survives. So line 3 - no mechanic blocks the gaps permanently -
does not follow from the mechanics of striking; on the counter line the same mechanics block every
stretch forever. The difference between the two lines is a single fact, `real_column`: on the real
line every j is a column. Any proof of line 3 must use that fact, and must use it in a way that
does not survive deleting the open columns (entry 112's admissibility test), which rules out every
free-phase cover and every count.

**Standing of the target.** In the kernel now: the owner's argument end to end, with its one
lemma proved equivalent to the conjecture (entry 111) and proved independent of the mechanics
(this entry). The proof, if one exists in the machine's terms, is a use of the completeness of the
line - every 6j +- 1 is on it - that is neither cover nor count. The record's only exhibited
non-count use is the pin at p = 29 (which multiples exist), with no general form found.

### 114. The wall map updated: seven angles, each closed on record, and the form a proof must take (loop, 2026-09-19)

research/proof/the_wall.md gains the section "The survival lemma, weighed and placed": entries
111-113 as the wall's new edges (line 3 weak form is the conjecture; line 3 independent of the
mechanics; the admissibility test), and a table of the seven angles with the pointer to where each
is closed - free-phase cover / run bound, rigid record tools, mechanics alone, analytic, chain,
the pins' arithmetic, bilinear switching. The bilinear angle was already on record
(period_scale.md 3.10: the g-m switching is an exact identity E = 2D + Q, not an inequality).
What a proof must be, in one line: a use of the line's completeness that dies when the open
columns are deleted and is neither a cover of free phases nor a count; nothing on record is such
a use.

### 115. The two layers of a stretch and the plug-run law (loop, 2026-09-19)

Branch opened from the owner's interleave claim (the kill zones of the primes are predictable and
interleave with the lower gears) and entry 108's program (a), (b), (c), now measured exactly on
the stretch's own two layers. Scripts research/stack/r8/two_layer_census.py, plug_run_gears.py.

**The two layers, exact.** In the stretch (p^2, q^2) every composite member is g x m with g its
least prime factor, g <= p < m. With B = q^(2/3): a member whose least factor exceeds B has a
PRIME cofactor (rough_member_form; m < q^2 / g < g^2). So the stretch splits into
  * the BASE layer, gears 5..B: a periodic two-tooth pattern, leaving the base-open columns;
  * the TOP layer, gears in (B, p]: each strike is a pinned product g x r with r prime in
    [q, q^2 / g) - the straddling products of entry 108 - and a base-open column it lands on is
    a plug.
The survivors of both layers are the twins. A dead stretch is a plug run - a run of CONSECUTIVE
base-open columns all plugged - as long as the whole base-open sequence. K(p) = the longest plug
run.

**Pre-registered and measured (every prime 7 <= p <= 20,000; 2,259 stretches).**
  * Predicted killed fraction 5/9 = 1 - (log B / log q)^2. REFUTED: the top layer plugs 0.645 of
    the base-open columns from p ~ 1,000 on (0.629 at 10^2, 0.643 at 10^3, 0.647 at 10^4), stable.
    The twin share 0.355 is the sieve's value (log B / log q^2)^2 e^(2 gamma) = 0.35, not the
    naive product - the base pattern's open columns are Mertens-thin, the twins are sieve-thin.
  * Predicted K(p) <= 2 ln(#base-open) + 4 (independent kills at 0.556). REFUTED at p = 433
    (21 against 13.8) and at every record after: K = 24 at 2477 (nb 1995), 25 at 3137, 32 at 5717
    (nb 2830, bound 19.9). The plug runs are LONGER than independent plugging at the measured rate
    would give: K(p) sits at 3 to 4 ln(nb), i.e. as if the per-column plug rate inside a run were
    0.75, not 0.65 - the plugs cluster.
  * No dead stretch (as at entry 107): K(p) / nb = 0.011 at p = 5717 and falling.

**Inside the record plug runs (plug_run_gears.py).** p = 5717: 32 plugs by 32 DISTINCT top gears
spanning 337..5639, 11 of 32 by gears below 2B; p = 2477: 22 distinct gears of 24, 7 below 2B;
p = 433: 16 of 21, 13 below 2B. So the clustering is NOT the smallest top gears (whose pins are
densest) - the whole top layer takes part, each gear once. That is the owner's "each kills once"
made exact for a plug run: the run at 5717 spans 436 columns and every top gear above 1309 has its
two teeth at least (g - 1)/3 > 436 columns apart (teeth_separation), so it can plug at most once
inside the run; the gears below 1309 could plug twice and do not. A plug run of length K inside a
window shorter than (B - 1)/3 columns needs K distinct top gears, each landing its one pinned
product g x r on a base-open column.

**What the law says.** The top layer's action on a stretch is a set of K(p) ~ 3.5 ln(nb) - length
plug runs, made of distinct gears' single pinned products, against a base-open sequence of length
nb ~ p x gap / (log p)^2. The clustering (3.5 against 2.3) is real and unexplained: it is the one
new fact of the branch, and the next probe is its mechanism - whether consecutive plugs share the
prime cofactor r (the same top prime r plugs g x r and g' x r for consecutive top gears g, g' - the
interleave of consecutive primes the owner named), which the pin structure permits exactly when
g' - g < (q^2 - p^2) / r.

**Standing against the target.** FACT. The plug-run law is entry 107's run law read on the
base-open subsequence: K(p) polylog against nb of exponent 1. Its clustering is a mechanism
question about the top layer, open.

### 116. Correction to 115, and the independence law of the two layers (loop, 2026-09-19)

**Correction.** Entry 115 read the record plug run K = 32 at p = 5717 against 2 ln(nb) + 4 for
its own stretch and called the plugs clustered. Wrong comparison: the record is the extreme over
the WHOLE sample, 4,266,465 base-open columns across 2,259 stretches, and the independent
expectation for the longest run of a 0.6463-rate process over that many trials is
ln(N) / ln(1 / 0.6463) = 35.0. The measured 32 is BELOW it. There is no clustering; the claim
is withdrawn. (The double plugs by one gear at both its teeth - 509 x 12239 and 509 x 12241,
263 x 23687 and 263 x 23689 at p = 2477 - are the cofactor column (r, r + 2) of the lower level,
the self-feeding correspondence of research/proof/self_feeding.md, known.)

**The independence law (research/stack/r8/plug_rate_by_neighbour.py, every prime 7..20,000).**
The top layer's plug rate on a base-open column, bucketed by the base pattern around it:
  * by distance to the previous base-open column, d = 1..11 and >= 12 (n from 96,194 to
    2,159,721 per bucket): every bucket within 1.5 sigma of the global 0.6463;
  * by the base strikes on the two neighbouring columns, all 13 realised patterns of
    (left/right struck at c-1, c+1) (n from 24,569 to 1,685,828): every bucket within 1.0 sigma.
So the top layer's action on a base-open column is independent of the base layer's local
configuration, to within sampling error over four million columns. The two layers of a stretch do
not interact: the top layer is a Bernoulli plugging of the base-open sequence at a rate fixed by
the sizes alone (0.646 = one minus the sieve's twin share, entry 115).

**What this decides.** The owner's interleave - the top gears' kill zones woven into the base
gears' congruences - has no visible structure at the stretch: the weave is exactly as
independent as two unrelated periodic patterns. Survival of the stretch is therefore not carried
by any correlation between the layers; it is carried by the independence itself (an independent
0.646 plugging of nb ~ p gap/(log p)^2 slots never plugs all of them). That is the sieve's
picture of the stretch, now measured on the machine's own layers with the rigid teeth in place,
and it is the picture the parity barrier says cannot be turned into a proof by counting. A proof
from the machine must therefore find structure the two-layer census does not see - at the pins
of single gears (entry 113's last open item), not in the layers' statistics.

### 117. The pins under the rigid pair: single-gear killers end at p = 41, and the kill distance of a stretch (loop, 2026-09-19)

Two probes on entry 113's last open item - structure at the pins of single gears.

**Single-gear killers, rigid (research/stack/r8/single_gear_killers.py, every p <= 3,000).** A
gear g is a single-gear killer of the stretch if re-phasing its rigid tooth pair {s + u, s - u}
(u = 6^-1 mod g, the pair's shape kept - `teeth_separation`) lands a tooth on every twin of the
stretch and abandons no column that only g struck. Result: exactly two stretches have one -
p = 17 (gear 17, shift 4; two twins) and p = 41 (gear 11, shift 10; three twins) - and none
from 43 to 3,000. The record's p = 29 killer (origin_mechanic.md section 3, teeth at +-2 against
the real +-5) is a FREE two-class tooth, not a rigid pair: with the pair's separation 2u = 10
fixed, no shift of gear 29 reaches both twins 143 and 147 (separation 4). So under the machine's
own tooth law the pin protection of a stretch is never one gear's from p = 43 on, for the plain
reason that the twins of a stretch (7 at 37, 13 at 47, 984 at 5,717) do not fit in two residue
classes of any gear.

**The kill distance (research/stack/r8/kill_distance.py, exact by iterative deepening to
depth 5, p <= 71 in 8 minutes).** d(p) = the least number of gears whose rigid pairs must be
re-phased away from the real phase to strike every column of the stretch, a re-phased gear's
abandoned lone kills becoming targets too.

       p   q   cols  twins  d(p)          p   q   cols  twins  d(p)
       7  11    11     4    > 5          41  43    27     3     1
      11  13     7     2     2           43  47    59    11     5
      13  17    19     7    > 5          47  53    99    13    > 5
      17  19    11     2     1           53  59   111    13    > 5
      19  23    27     4    > 5          59  61    39     5     3
      23  29    51     8    > 5          61  67   127    19    > 5
      29  31    19     2     2           67  71    91    11     5
      31  37    67    11    > 5          71  73    47     3     2
      37  41    51     7     5

The distance tracks the twin count: 1 or 2 gears when the stretch holds 2 or 3 twins, 3 at 5
twins, 5 at 7 to 11, beyond 5 at 13 and more. Mechanism: a re-phased gear g covers the twins in
two residue classes modulo g - many for a small g - but abandons its lone kills, about
(2 L / g) c / (ln p)^2 columns for a small g, which the other re-phased gears must then cover at
two columns per period each; so the cheap gears are the expensive ones, and the distance grows
with the twins. The capacity bound (re-phased gears strike at most sum 2 ceil(L / g) new
columns) gives only d >= 1 here and is not the mechanism.

**Standing.** Both FACT. d(p) is a new object - the Hamming distance from the real configuration
to the nearest killer - and its measured law is d ~ twins / 2.5, growing without bound. Its base
case d(p) >= 1 is the survival of the stretch itself, so a lower bound on d is not a route to
the conjecture but a statement of how far the machine sits from failure: at p = 5,717, with 984
twins, of the order of 400 gears would have to move at once.

### 118. The kill distance, exact by integer programming: unkillable stretches, and d = half the twins (loop, 2026-09-19)

research/stack/r8/kill_distance_ilp.py (scipy 1.16 milp / HiGHS): one binary per (gear, shift),
one shift per gear, every column of the stretch struck, minimise the gears moved off the real
phase. Every stretch 7 <= p <= 109, exact (the search of entry 117 confirmed where it finished).

      p    q  cols twins   d   d/twins  gears moved
      7   11    11    4   none        - the stretch cannot be killed by any rigid re-phasing
     11   13     7    2   none        -
     13   17    19    7   none        -
     17   19    11    2    1   0.50   17
     19   23    27    4   none        -
     23   29    51    8   none        -
     29   31    19    2    2   1.00   19, 29
     31   37    67   11   none        -
     37   41    51    7    4   0.57   13, 23, 29, 37
     41   43    27    3    1   0.33   11
     43   47    59   11    5   0.45   5, 11, 17, 37, 43
     47   53    99   13    8   0.62   5, 11, 13, 17, 23, 31, 37, 47
     53   59   111   13    8   0.62   5, 7, 11, 19, 23, 31, 37, 47
     59   61    39    5    3   0.60   5, 13, 53
     61   67   127   19    7   0.37   7, 11, 17, 37, 41, 59, 61
     67   71    91   11    5   0.45   5, 7, 13, 29, 41
     71   73    47    3    2   0.67   11, 71
     73   79   151   15    8   0.53   11, 13, 19, 31, 37, 43, 67, 73
     79   83   107   14    6   0.43   7, 31, 37, 43, 53, 71
     83   89   171   14    8   0.57   5, 13, 29, 37, 43, 53, 61, 83
     89   97   247   21   10   0.48   13, 17, 29, 31, 43, 61, 67, 73, 79, 89
     97  101   131   15    6   0.40   13, 23, 29, 47, 73, 83
    101  103    67    7    3   0.43   11, 13, 31
    103  107   139   10    5   0.50   13, 17, 53, 89, 103
    107  109    71    6    3   0.50   17, 19, 43
    109  113   147   11    5   0.45   17, 37, 67, 89, 109

**Two facts.**
  1. UNKILLABLE STRETCHES. At p = 7, 11, 13, 19, 23 and 31 no assignment of shifts to the rigid
     pairs of the gears 5..p strikes every column of the stretch: those stretches are open under
     EVERY phase vector, the real one included, by the pair shape and the lengths alone. From
     p = 37 on every stretch is killable in residue space (entry 105's threshold, now at the
     stretch and exact), and the machine's own vector is never a killer.
  2. HALF THE TWINS. Where a killer exists, the distance from the real configuration to the
     nearest one is d = 0.33 to 0.67 of the twin count, 0.50 on average over the 20 killable
     stretches, with no drift from 37 to 109. The gears moved are spread over the whole range
     (5 to p), always including the top gear or one near it in about half the cases.

**Mechanism of the ratio.** A moved gear covers the twins in two classes modulo g and must have
its abandoned lone kills re-covered; the optimum moves a mix of small gears (each takes 2 to 4
twins, costs many lone kills) and large gears (each takes 1 or 2 twins at its pinned products,
costs 0 to 2 lone kills) - about two twins per moved gear.

**Standing.** FACT, and a sharpening of the rigid-vs-free record. In the free two-class model the
stretch is killable from p = 17 (entry 105); with the rigid pair it is unkillable to 31, and the
real machine sits at Hamming distance half its twin count from the nearest killer at every p
measured. The conjecture is d(p) >= 1 for infinitely many p; the law measured is d(p) ~ twins/2,
i.e. the survival margin of a stretch is not one column but half its twins' worth of gears.

### 119. The shift-rigid record IS the machine's record, and the ILP computes it without the period (loop, 2026-09-19)

**The object.** F_shift(p): the longest run [0, L) that SOME assignment of shifts to the rigid
tooth pairs of the gears 5..p strikes completely (research/stack/r8/shift_rigid_record.py, ILP
feasibility, L increasing).

**The identity, and why.** F_shift(p) = F({5..p}), the machine's own record with its real
phases. Reason: by the Chinese remainder theorem every shift vector (s_g) is realised by one
window position x of the real pattern (x = -s_g mod g for every g), so the real pattern's runs
over one period are exactly the runs of all shift vectors, and the worst of them is the record.
The re-phasing adversary of entries 117-118 therefore never leaves the machine: every rigid
killer it finds is a real position of the real pattern, at some other window. (The free
two-class adversary, h_2 / A072753, does leave it.)

**Values (ILP), against the certified ladder:**

      p    F_shift   known F(M)          time
      5       1          1
      7       4          4
     11       6          6
     13      10         10
     17      17         17
     19      24         24
     23      33         33 (entry 89)     1.7 s
     29      42         42                7.9 s
     31      57         57               20 s
     37      87         87 (law_register)  50 s
     41      90         90              313 s
     47     bisecting 110..130 (118 certified) - in progress
     59     bisecting 161..179 (pinned [161, 178]) - in progress

Every value agrees with the record where the record has one. The ILP reaches in seconds what
the period scan cannot: the period of {5..31} is 2 x 10^11 columns and the exact record 57 came
from a 9-gear ILP in 20 seconds. The record's own falsification target - one exact rigid record
beyond q = 59 (entry 110) - is now a computation, not a scan.

**Consequence for the kill-distance law (entry 118).** d(p) is the Hamming distance, in gears,
from the window at p^2 to the nearest window of the same pattern that is fully struck. That the
nearest such window is half the twins' worth of gears away says the pattern's dead windows (they
exist from p = 37, being the record's own runs of length >= the stretch) are nowhere near the
squares in residue space. The survival lemma in this language: the square's window is never one
of the pattern's dead windows - and the dead windows of the pattern of gears <= p have length
F(p) = 57 at 31, 87 at 37, 118 at 47, against stretches of length 2 p gap / 6 which exceed
F(p) from p = 37 on (stretch 51 at 37 < 87 - the square's window is short enough to be killable
in principle; and it is not killed).

### 120. Kernel: every shift vector is a window of the real pattern (loop, 2026-09-19)

[proofs/RigidShift.lean, round 112, 0 sorries]
  * `window_realises_shift`: for distinct primes G and any shifts s, there is a window position x
    such that for every gear g in G, every tooth offset t and every relative column i, the real
    tooth t strikes x + i iff the shifted tooth s g + t strikes i (Chinese remainder theorem,
    x = (g - 1) s g mod g for every g).
  * `shifted_pattern_is_window`: the shifted rigid pair of every gear, read at window x, is the
    real pair - the shifted machine IS the real machine at another window.
So entry 119's identity F_shift = F(M) is kernel-backed at its mechanism, and entries 117-118's
re-phasing adversary is a walk along the real pattern's period: the kill distance d(p) is the
number of gears in which the square's window differs, in residue, from the nearest fully struck
window of the same pattern. F(47) bisection: 120 uncoverable, 117 coverable (118 certified on
record; 118, 119 pending); F(59) queued.

### 121. The exact rigid ladder to 53, a new record certified, and the record against the stretch (loop, 2026-09-19)

By the CRT identity (entries 119-120) the ILP over shift vectors computes the machine's record
F(M) exactly; the coverable half is CERTIFIED by turning the ILP's shift vector into a window
position x by CRT and checking the real pattern's run at x directly
(research/stack/r8/rigid_record_certificate.py); the uncoverable half is HiGHS's proof of
infeasibility. Values are the struck run (the record's certified ladder counts run + 1: 118 at
47 is this 117, 88 at 37 is this 87).

      p    run F   p ln p   F / p ln p   stretch (q^2-p^2)/6   stretch / F
      5      1
      7      4       14      0.29           11                  2.8
     11      6       26      0.23            7                  1.2
     13     10       33      0.30           19                  1.9
     17     17       48      0.35           11                  0.65
     19     24       56      0.43           27                  1.1
     23     33       72      0.46           51                  1.5
     29     42       98      0.43           19                  0.45
     31     57      106      0.54           67                  1.2
     37     87      134      0.65           51                  0.59
     41     90      152      0.59           27                  0.30
     43    102      162      0.63           59                  0.58
     47    117      181      0.65           99                  0.85
     53    144      210      0.69          112                  0.78   NEW, certified
     59  [161,164]  241      ~0.67          39                  0.24   bisecting
     61   <= 209    251                    143                          bisecting

**The new record.** F({5..53}) = 144: window x = 1249461754311661376 in the period
5431526412865007455, members 7496770525869968255 and ...257 at the start, the real pattern struck
for exactly 144 consecutive columns from x (certificate run = 144). Shift vector
5:4, 7:4, 11:9, 13:8, 17:4, 19:12, 23:15, 29:22, 31:9, 37:36, 41:30, 43:9, 47:26, 53:30.
The record's own target (entry 110: one exact rigid record beyond q = 59) is within reach of
the same computation; F(59) is pinned to [161, 164] against the record's [160, 177].

**The growth.** F / (p ln p) rises from 0.3 at p = 13 to 0.69 at 53 - the rigid record grows
faster than p ln p, as the free one must (Erdos-Rankin type: the layered lower bound gives
p (ln p)^(3 - o(1)) eventually). The stretch length (q^2 - p^2)/6 = p gap/3 + gap^2/6 is
p ln p / 3 on average and at most about p (ln p)^2 / 3 (Cramer). So from p = 37 the record
exceeds most stretches (stretch / F below 1 at 37, 41, 43, 47, 53, 59), i.e. the pattern of the
gears up to p has fully struck windows longer than the stretch - the killers of entry 118 - and
the ratio falls: the killers grow relative to the stretch. The survival of the stretch is never
a matter of the pattern lacking a run long enough; from 37 on it has them, and increasingly so.

**Standing.** FACT (new exact values, a certified record, and the comparison that places the
killers' length against the stretch's). The survival lemma is exactly: the square's window is
not one of those runs. The runs exist; their number per period is small (a run of length F needs
about pi(p) gears in specific phases) and their positions are the extremal alignments of the
pattern; the squares' positions are square residues. No relation between the two sets is on
record, and entries 105 and 109 measured none.

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

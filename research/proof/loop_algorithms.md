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

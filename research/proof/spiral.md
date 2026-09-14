# The spiral and the primorial spiral (owner's find, 2026-09-13; formalised the same day; the primorial spiral named 2026-09-14)

## Definition

Machine q, gears the primes up to q. Odd gears descending: g_1 = q, g_2 = p', ..., g_r = 3.
The spiral is the walk from home (-1, 1) with one flip per odd gear, in that order, each about
the axis one period of the mirror {2, g_i} from the current column (axis = current centre +
d_i · 2 g_i), the direction d_i alternating: up, down, up, ... (d_i = (-1)^(i+1)).

- One step moves the column by 4 d_i g_i. PROVED [MirrorWalkSpiral.spiral_step]
- Closed form: the spiral ends at

      E(q) = -1 + 4 A(q),   A(q) = g_1 - g_2 + g_3 - ... ± g_r,

  four times the alternating sum of the odd gears. PROVED [spiral_eq, spiral_home]
- What it carries: the endpoint is open to every gear dividing A(q) (it sees such a gear as
  home does). PROVED [spiralEnd_open_of_dvd]
- It never overshoots: A(q) ≤ q, so E(q) < 4q ≤ q² for q ≥ 5. PROVED [altSum_le_head,
  spiralEnd_lt]
- It lands above q at every machine measured: E(q) between 1.40 q and 2.71 q, mean 2.003 q,
  machines 5 to 20000. This is A(q) > (q + 1)/4, the alternating sum of the odd gears exceeding a
  quarter of q, a statement about the gaps between the gears. MEASURED, not proved.

## What the endpoint is

- Position: near 2q, inside the window by construction (measured above q, proved below q²).
- Column class: E mod 6 is 1, 3 or 5 in equal shares (759, 753, 748 of 2260 machines), because
  the mirrors {2, g} do not carry 3. So the endpoint is a left member of a twin slot at a third
  of the machines, a right member at a third, off the slots at a third. The closing flip of 2 or
  3 the owner used moves it onto a slot.
- Carried gears: the gears dividing A(q), on average 1.6 per machine (q = 7: 5; q = 13: 7;
  q = 17: 5; q = 5, 11, 19: none). Everything else at the endpoint is undecided by the walk.
- Distance to the nearest twin in the window: within 6 at every machine to 29, then 7 of 15 to
  100, 47 of 143 to 1000, 113 of 501 to 5000; median 14, largest 100 to 5000. Not a function of
  whether q is itself a twin member (research/proof/walk_parts.md).

Endpoints: q = 5: 7; 7: 19; 11: 23; 13: 27; 17: 39; 19: 35; 23: 55; 29: 59 (a twin); 31: 63;
37: 83.

## The spiral as the launch point

The spiral is a deterministic, blind, mirror-only walk that uses every gear once and lands
inside the window near 2q at every machine. It is the first construction on the tree that
enters the window from the machine's structure alone. Explorations that start from its
endpoint (the closing move, other pairings, other orders, k varying with the gear) can be
compared on one footing: where they land relative to E(q), which gears they carry, and how far
the nearest twin is. Scripts: the spiral and its measurements are in research/stack/r8/
(spiral tests in walk_parts.md); kernel proofs/MirrorWalkSpiral.lean (round 49).

## Residues at the endpoint, and the {2, 3, g} spiral (owner, 2026-09-13)

Owner: stepping through every gear may pick up and rule out residues, placing the endpoint in
a specific position relative to all the residues; and try the spiral with 2 and 3 in every
step. Machines 7 to 5000 (666).

Residues at the {2, g} endpoint E. The strike rate of each gear h at E against 2/h, the rate at
a random column: 3: 0.667 vs 0.667; 5: 0.380 vs 0.400; 7: 0.284 vs 0.286; 11: 0.155 vs 0.182;
13: 0.188 vs 0.154; 17: 0.115 vs 0.118; 19: 0.103 vs 0.105; 23: 0.089 vs 0.087; 29: 0.067 vs
0.069; 31: 0.068 vs 0.065. Every gear strikes the endpoint at its ordinary rate. E is itself a
twin at 22 of 666 machines. The spiral rules out nothing by residue: E = -1 + 4A and gear h
strikes it iff h divides 4A - 1 or 4A + 1, and A modulo h is the alternating sum of the other
gears, which h sees as a random residue.

The {2, 3, g} spiral: gears 5 .. q descending, each step with the mirror {2, 3, g}, moving
12 d g; endpoint E3 = -1 + 12 A5, A5 the alternating sum of the gears from 5 up. E3 lies
between 3.3 q and 9.7 q (mean 6.0 q), inside the window; E3 = 5 mod 6 always (2 and 3 carried,
so the endpoint is always a left slot member). E3 is a twin inside the window at 60 of 666
machines (0.09, the twin rate of a slot column near 6q); distance to the nearest twin median
18, largest 228. Strike rates per gear at E3 again match 2/h (5: 0.426 vs 0.400; 7: 0.300 vs
0.286; 11: 0.174 vs 0.182; 13: 0.152 vs 0.154; 17: 0.106 vs 0.118; 23: 0.094 vs 0.087).

Reading: the spiral's endpoint carries exactly the divisors of the alternating sum (2 with the
{2, g} pairing; 2 and 3 with the {2, 3, g} pairing) and nothing else; every other gear meets the
endpoint at its own rate. Passing through every gear leaves no residue trace beyond the gears
that divide the sum.

## Orders, gear sets, bases and stacking (owner's ideas, tested separately; 2026-09-13)

Scripts research/stack/r8/spiral_ideas.py (machines 11 to 1500, 235 machines) and
spiral_stack.py (11 to 2000). Endpoint E = -1 + 2 P_base A, A the alternating sum of the
spiral's gears in the chosen order.

| variant | E/q mean (min..max) | inside the window | E a twin | nearest twin median distance | gears carried |
|---|---|---|---|---|---|
| (1) order descending (the spiral) | 2.02 (1.70..2.39) | 235 of 235 | 8 | 12 | 1.24 |
| (1) order ascending | -0.12 (-2.48..2.24) | 117 | 5 | 74 | 1.24 |
| (1) order by the tooth 6^-1 mod g | -0.83 (-14.5..10.4) | 81 | 9 | 424 | 1.20 |
| (1) order by tooth + gear | 0.50 (-11.1..11.3) | 109 | 8 | 70 | 1.16 |
| (1) striking order (first strike above q) | 0.39 (-7.1..9.2) | 95 | 1 | 220 | 1.20 |
| (1) random orders (5 per machine) | 0.90 (-49..54) | 560 of 1175 | 21 | 110 | 1.23 |
| (2) gears below sqrt q only, descending | 0.11 (0.05..1.00) | 0 | 0 | 754 | 0.56 |
| (2) gears below sqrt q, striking order | 0.02 (-0.34..1.00) | 0 | 0 | 772 | 0.48 |
| (3) base {2,3,g}, all gears, descending | 6.07 (3.6..9.7) | 235 of 235 | 32 | 12 | 1.26 |
| (3) base {2,3,g}, gears below sqrt q | 0.28 (-0.09..2.03) | 6 | 6 | 606 | 0.51 |
| (3) base {2,3,g}, striking order | -1.03 (-27..21) | 99 | 11 | 678 | 1.13 |

Stacking (spiral_stack.py): stages with bases {2}, {2,3}, {2,3,5}, ... while the base's product
stays below q; the endpoint is the sum of the stage moves, so the stage order does not matter;
E/q mean 219 (5.6..306), a twin at 3 of 299 machines, carried gears 1.34 (no accumulation: the
carried set is the divisors of E + 1, and the stages' sums do not share divisors).

Read off:

- The order is what keeps the spiral in the window. Descending is the only order whose
  alternating sum is pinned (0 <= A <= q, proved) and it lands near 2q every time; every other
  order, the striking order included, lets the sum wander in sign and size and lands below q or
  outside the window at half the machines or more.
- The gear set below sqrt q is too small: its alternating sum is about sqrt(q)/2, so the
  endpoint sits near 2 sqrt q, below the window at every machine. The higher gears do not blow
  the landing out; they are what lift it to 2q, and the proved bound A <= q keeps it inside.
- The base {2,3,g} lands at 6q, on a slot member always, a twin at 14 percent (the slot rate
  there); the base {2} lands at 2q, on a slot member a third of the time, a twin at 3 percent.
- Carried gears stay near one per machine in every variant: no order, set or base makes the
  alternating sum collect gears as divisors. Stacking does not accumulate them either.

So of the ideas, the spiral as first found (descending, base {2} or {2,3}, all gears) is the
one with a proved landing zone; the others lose the landing before they could gain anything.

Base = all gears below sqrt q (owner's correction, 2026-09-13): spiral over the gears above sqrt q with the mirror {all gears <= sqrt q, g}; endpoint E = -1 + 2 P_B A_high, P_B the product of the base. The base's product outgrows the window: q = 11 .. 23 (base 2, 3): E at 0.2 to 0.9 q^2, inside; from q = 29 (base 2, 3, 5) E passes q^2 at most machines; machines 11 to 2000: inside the window 9 of 299, twins among those 3. The base below sqrt q is the carry-cap wall in spiral form: its product must stay below q^2 / 2, and the primorial of sqrt q does not.

Stacking with the initial direction alternating per layer (owner, 2026-09-14): layers with bases {2}, {2,3}, {2,3,5}, ... (product below q), the first layer starting up, the second down, the third up, ...; the endpoint is -1 + the signed sum of the layer moves. Machines 11 to 2000: E/q mean -156 (-230 .. 34), inside the window at 36 of 299 (the same-start stack: 289 of 299 at E/q about 219), a twin at 2, carried gears 1.32. The alternation cancels the layers' moves against each other and pushes the endpoint below zero at most machines; with layers allowed up to product q^2/2 both versions leave the window at every machine (E/q of order 10^5). No carry accumulation in any version.

Base {2,3,5} (owner, 2026-09-14): spiral over the gears from 7 up (descending) with the mirror {2,3,5,g}, each step moving 60 d g; endpoint E = -1 + 60 A7. Machines 11 to 5000: E between 21.7 q and 41.5 q (mean 30.1 q); inside the window at 658 of 665 (the misses are q below 30, where 30q exceeds q^2); E a twin at 65 of the 658, rate 0.099, against 0.102 for a random slot column open to 5 (the base is carried, nothing else is); nearest twin median 30 away; carried gears beyond the base 1.21 per machine. For comparison base {2}: 2q, twin rate 0.033; base {2,3}: 6q, rate 0.090. Each base buys exactly its own gears' factor in the twin rate and moves the landing up by the base's product.

Base {2,3,5,7} (owner, 2026-09-14): spiral over the gears from 11 up, step 420 d g, endpoint E = -1 + 420 A11. Machines 11 to 5000: E between 64 q and 420 q (mean 211 q); inside the window at 622 of 665, the first at q = 181 (below that 210 q exceeds q^2); E a twin at 67 of the 622, rate 0.108, against 0.116 for a random slot column open to 5 and 7; nearest twin median 30; carried gears beyond the base 1.06. The base is carried and nothing more, as at every smaller base; the next base {2,3,5,7,11} would land near 2310 q and enter the window only from q = 2311.

## The primorial spiral (owner's rule, 2026-09-14; named 2026-09-14)

Rule: the base is every lower gear whose running product stays at most q/2 (2, then 2 3, then
2 3 5, ... as long as the product is at most q/2); the spiral runs over the remaining gears,
descending, each step with the mirror {base, g}, directions alternating; endpoint
E = -1 + 2 P A, P the base's product, A the alternating sum of the gears above the base.

- Inside the window at every machine 11 to 20000 (2258 of 2258). The upper end is proved:
  A <= q (altSum_le_head) and P <= q/2 give E < q^2. The lower end (E > q) is measured; the
  landing sits at 0.20 q^2 on average, at most 0.56 q^2.
- Base sizes: 1 gear at q = 11; 2 from 13; 3 from 61; 4 from 421; 5 from 4621. The base changes
  exactly where the next primorial passes q/2.
- E a twin at 205 of 2258 (0.091), the slot rate adjusted for the carried base; nearest twin
  median 42 away, largest 768; gears carried beyond the base 1.17 per machine.
- Examples: q = 61, base 2 3 5, E = 32.4 q; q = 421, base 2 3 5 7, E = 181.6 q; q = 4621, base
  2 3 5 7 11, E = 2322.5 q, nearest twin 228 away.

So the primorial spiral is a blind mirror walk with a proved ceiling, landing inside the window
at every machine tried, carrying the largest base the window admits (about log q gears) and
about one chance gear beyond it. It is the spiral in its strongest exact form; the gears above
the base still meet its endpoint at their own rate.

## Named: the primorial spiral. The closing step by base flips (owner, 2026-09-14)

The max-base spiral is named the primorial spiral: its base is the largest primorial below
q/2, and a gear joins the base exactly when its primorial passes q/2.

Owner's suspicion: the jump from the endpoint to a twin is a flip about the base's own mirror
(2, 3, ... whatever the base is), up or down. Tested, machines 11 to 20000 (2258): from the
endpoint E, flips about the base mirror land at E + 2kP and E - 2kP, all open to the base by
construction.

- A twin at E itself or one base flip away: 546 of 2258. Within three base flips: 1062. Every
  machine reaches a twin within 400 base flips (largest needed 34).
- Smallest number of base flips to a twin: median 4, mean 5.5. First hit up 1077, down 976, at
  E 205: no preferred direction.

So one base flip is not the rule. The base-mirror landings are slot columns open to the base
and nothing more, and a twin appears among them at the base-adjusted slot rate (about 0.09 to
0.12 per landing), which gives the geometric pattern seen: a quarter at the first flip, half
within three, a tail to 34. The base mirror keeps the base carried exactly; it does not steer
the other gears.

Stacking the spiral and the primorial spiral (owner, 2026-09-14), machines 11 to 20000: the spiral alone lands at 0.001 q^2 on average (near 2q), inside at all, a twin at 68, carrying 1.58 gears (divisors of E + 1); the primorial spiral alone at 0.200 q^2, inside at all, a twin at 205, carrying 3.85 gears (its base plus about one); the spiral then the primorial spiral (both starting up) at 0.201 q^2, inside at all, a twin at 29, carrying 1.54 gears, the endpoint spread over the classes 1, 3, 5 mod 6 like the plain spiral; with the second starting down the sum goes negative at every machine; the primorial spiral then the spiral (down) lands inside at 2257 with 24 twins. Stacking loses the base: E + 1 = 4 A1 + 2 P A2 is divisible by the base only when the base divides A1, so the carried set falls back to the plain spiral's and the endpoint leaves the slot grid. The primorial spiral alone is strictly better than any stack containing it.

## Phase 2 as residue avoidance (owner, 2026-09-14)

Owner: the high gear's period must be an open residue and get lucky with the composite
killers; if we know the killers we pick an upper gear that is not one. Made exact:

- The final step {3, h} from the spiral's landing E lands at L = E + 6h (up) or E - 6h (down).
  It is open to 2 and 3 by construction, and to h itself iff E is open to h (the mirror
  carries h). For every other gear g, by the landing law (MirrorWalk.struck_flip_iff): g
  strikes L iff 6h = -E or -(E + 2) (mod g) for the up flip, 6h = E or E + 2 (mod g) for the
  down flip. So each gear forbids exactly two residue classes of h, both computed from E,
  which phase 1 gives in closed form.
- Checked at every machine 11 to 199, both directions, every high gear in reach: the high
  gears that avoid every forbidden class are exactly the hits of the table (0 mismatches once
  the check is written as a boolean). Examples of the forbidden classes for the down flip:
  q = 47 (E = 323): 5 forbids h = 0, 3; 7 forbids 4, 6; 11 forbids 1, 8; 13 forbids 0, 4; the
  passing h are 7, 29, 31, 37 down and 23, 41 up. q = 197 (E = 6719): passing 173, 193 up and
  163 down.

So the algorithm for the final step is: from E, list the two forbidden classes per gear; take
the first gear above sqrt q (either direction) that sits in no forbidden class; flip about
{3, h}. The landing is a twin by the landing law and the square-root rule (every gear up to q
consulted, L below q^2). No primality is tested; residues are.

What this is and is not: it is the direct construction again on the gear line, with the
classes now fixed by the spiral's E instead of by g^2; the killers per gear are known, the
choice of h is a sieve on the high gears against two classes per gear; whether some high gear
in reach always passes is the second-sieve statement (measured: 98 of 100 machines to 569 have
a passing h with the mirror {3, h}; the two exceptions have passing subsets of other shapes).

## The forbidden classes as fields (owner, 2026-09-14)

Owner: the forbidden values have the known, proved shape of each field type (squares, products
of j gears, ...); link the residue test to those findings. Done as a killer map
(research/stack/r8/results_phase2_killers.txt): for every high gear h in reach at every machine
11 to 2000, the landing L = E +- 6h either passes (a twin) or is charged to the one field that
strikes it: higher:g with g the smallest gear factor of the struck member, the member (left L or
right L + 2) and the order j of the composite.

- Candidates 86,396; pass 3,639; kills 82,757.
- Kills by field: higher:5 42,041; higher:7 14,584; higher:11 5,325; higher:13 3,764; higher:17
  2,456; higher:19 1,951; higher:23 1,469; higher:29 1,048; higher:31 884; higher:37 691. The
  small gears' fields take the kills in the order of the fields' own shares of the line (the
  higher:g field is g times the survivors of the gears below g, kernel FieldsC).
- Kills by order of the struck composite: order 2: 34,924; 3: 32,439; 4: 12,483; 5: 2,554; 6:
  324; 7: 33. Orders 2 and 3 take 81 percent, orders above 7 never appear, as the order
  ceiling says (5^j <= q^2).
- Left and right members struck equally (41,509 and 41,248). The square field never strikes a
  landing (0 of 82,757): the landings E +- 6h are not squares.

The link, exact: the two forbidden classes of h modulo g are the trace of the multiples field
of g on the landing family {E +- 6h}. Partitioned by smallest factor, the failures are the
traces of the higher:g fields, each of which is proved to be g times the survivors of the gears
below g and periodic; so the residue test is the statement "L and L + 2 lie in no higher:g field
for g <= q", and a high gear passes exactly when its landing escapes every field's trace. What
the fields give beyond the two classes per gear is the accounting: which field takes each
failure, in the proved proportions. What they do not give is a reason the traces of all the
fields on the high-gear line never cover it, which is the open statement.

## Phase 2 field by field (2026-09-14)

The high-gear line in reach from E, the gears consulted in the machine's order (script
research/stack/r8/phase2_fields.py, table results_phase2_fields.txt, machines 11 to 2000).

- Stages, totals: in reach 86,396; left by higher:5 44,355; left by higher:5 and higher:7
  29,771; left by every gear of the sub-machine (gears up to sqrt q) 13,334; left by every
  gear up to q (twins) 3,639.
- The sub-machine leaves a survivor at every machine except 11 (where the sub-machine is
  empty and the three high gears are all taken). A passing gear exists at every machine
  except 11.
- What takes a sub-machine survivor is always a high gear g above sqrt q dividing L or L + 2,
  and the struck member is then g times a number free of the gears below sqrt q: order 2
  (g times a prime of the window) in 9,139 cases, order 3 (g times two primes above sqrt q)
  in 556, never higher. So after the sub-machine the only killers left are the higher:g fields
  of the high gears themselves, and each strikes the line in its two classes exactly as the
  landing law says.
- Sample rows (q, E, reach, after 5, after 5 and 7, after sub-machine, pass, first pass):
  31, 227: 16, 8, 4, 4, 3, h = 7 up. 101, 3179: 44, 22, 14, 14, 3, h = 13 up.
  499, 119699: 174, 88, 62, 29, 11, h = 37 up. 1999, 409499: 578, 296, 196, 69, 20, h = 103 up.

So the open item splits into two proved-shape parts: (a) the sub-machine's traces on the
high-gear line leave survivors (the sub-machine is the machine of sqrt q acting on a line of
length about q, the same object as the window statement one level down); (b) among those
survivors the high gears' own fields, each striking two classes, do not take every one. Part
(b) is where the landings that fail are g times a window prime: the walk's landing is a column
of the higher:g field for one high gear g.

## The landing is column h modulo the base (2026-09-14)

Exact, kernel proofs/MirrorWalkColumn.lean (round 51, built, 0 sorries, standard axioms):
base_strikes_up_iff, base_strikes_down_iff, landing_open_base_iff, primorialEnd_modEq.

- The primorial spiral lands at E = -1 + 2 P A, so E = -1 modulo P, the product of the base.
- Up step {3, h}: L = E + 6h = 6h - 1 and L + 2 = 6h + 1 modulo every base gear. The landing
  IS column h. Down step: L = -(6h + 1), L + 2 = -(6h - 1): the reflection of column h.
- So a base gear strikes the landing iff it strikes column h = (6h - 1, 6h + 1). The base
  gears' forbidden classes of h are +-6^{-1} modulo g: fixed by the gear, independent of E and
  of the direction. Checked at every machine 11 to 2000, both directions, every high gear in
  reach: 0 mismatches (research/stack/r8/phase2_column.py, results_phase2_column.txt).
- Gear roles on the high-gear line, per machine: base (2, 3, and the gears with primorial at
  most q/2; classes fixed, = column h), sub non-base (from the next prime to sqrt q; classes
  depend on E and direction), high (above sqrt q; strike only as g | L or g | L + 2, order 2
  or 3). Example q = 499: base {2, 3, 5, 7}, sub non-base {11, 13, 17, 19}, high from 23.
- Totals to 2000: in reach 86,396; column h open to the base 30,710; left by the whole
  sub-machine 13,382; pass 3,639. High gears whose own column is a twin prime pair: 8,825, of
  which 1,076 pass (their landings are open to the base for free, then meet the E-dependent
  gears).
- What this fixes: the base part of the residue test is not a second sieve at all; it is
  the fields' own statement about column h. The E-dependence lives only in the sub non-base
  gears (between the primorial bound and sqrt q). Widening the base pushes E out of the window
  (base product must stay at most q/2), so the E-dependent band is exactly the price of the
  landing ceiling.

## The band gears' classes (2026-09-14)

Script research/stack/r8/band_classes.py, table results_band_classes.txt, machines 11 to 2000.

- A band gear g (non-base, up to sqrt q) forbids h in the classes -E (6d)^{-1} and
  -(E + 2)(6d)^{-1} modulo g, fixed by E mod g = -1 + 2 P A mod g, that is by A mod g, the
  alternating sum of the non-base gears (q down to the first prime above the base) modulo g.
- Measured: E mod g takes every residue, with no shape. g = 11 over 273 machines: residues
  0..10 occur 20, 30, 19, 24, 20, 20, 32, 22, 23, 33, 30 times. g = 13 over 264 machines: 18,
  16, 16, 24, 23, 21, 22, 23, 14, 25, 22, 22, 18. g = 17 over 242: between 8 and 17 each.
  E itself is struck by g in about 2 of g cases (11: 53 of 273; 13: 40 of 264; 17: 23 of 242),
  the two-class share and no more.
- Examples: q = 499, E = 119699: E mod 19, 17, 13, 11 = 18, 2, 8, 8; up classes of h: 19
  forbids 16, 3; 17 forbids 11, 5; 13 forbids 3, 7; 11 forbids 6, 2. q = 1999: ten band gears
  11 to 43, E mod g = 2, 12, 3, 11, 7, 19, 20, 20, 32, 10.
- No closed form: A mod g is the alternating sum of the primes above the base modulo g, which
  the machine does not fix.
- The band is forced by the window, not by the spiral: a landing E in the window with E = -1
  modulo every gear up to sqrt q would need the primorial of sqrt q to divide E + 1, and that
  primorial exceeds q^2 from q = 121 on (2*3*5*7*11 = 2310 > 121... exactly: the primorial of
  sqrt q passes q^2 once sqrt q >= 11). So any phase 1 that lands in the window leaves a band
  of gears between the primorial bound and sqrt q whose classes on the landing are not chosen.
- Standing: the open item is exactly the band and the high gears together: from a column E
  in the window, some high gear h has E +- 6h open to the band and to the high gears' fields.
  The base is settled (column h); the band's classes are E's residues; the high gears strike
  only as g | L or g | L + 2. That is the whole remaining statement, in parts.

## Each field type of the band as its own object (owner, 2026-09-14)

Owner: the band is an aggregate with conditional members; solve each field type in it, not
the band as one object. Script research/stack/r8/band_field_traces.py, table
results_band_field_traces_499.txt.

Exact shape of one field type on the landing line (step up, L = E + 6h, h a high gear):
- Band gear g, left member: the class h = a_g (mod g), a_g = -E 6^{-1}; right member: the
  class b_g = -(E + 2) 6^{-1}. Inside a class h = a + g t and the struck member is g c with
  c = c_0 + 6 t: the cofactors run along a column line one level down (c = +-1 mod 6).
- Field type (g, member, order j) = the t of the class at which c is a product of j - 1 primes
  all at least g. A cofactor with a prime below g is the smaller gear's kill, not g's.
- So each field type is: one class of h, one cofactor line, one factor shape of c. Three
  objects, each already proved in the fields: the class (landing law), the line (column
  structure), the factor shape (higher:g = g times survivors of the lower gears).
- q = 499, band gear 11, left member: class h = 6 (mod 11), c = 10885 + 6t. Hits: h = 61 (c =
  5*37*59, gear 5's kill), 83 (7^2*223, gear 7's), 127 (47*233, order 3), 149 (19*577, order 3),
  193 (10987 prime, order 2), 281 (5*2207, gear 5's), 347 (prime, order 2), 457 (prime, order 2),
  479 (11*1013, order 3). Right member: class 2 (mod 11), c = 10883 + 6t, order 2 at h = 101,
  167, 431; order 3 at 233; order 4 at 277 (11*17*59); gears 5, 7 take 79, 211, 409.
- Band gear 19 at 499: left class 16 (mod 19) has one own kill (h = 73, order 2), the other
  three t are gears 11, 13, 5's kills; right class 3 (mod 19): order 2 at 193, 307, 383, order
  3 at 41, gears 5 and 7 take 79, 269, 421.
- Order 2 of g on the h-line is two coupled prime conditions in t: h = a + g t prime (h is a
  gear) and c = c_0 + 6 t prime. Order j: c a product of j - 1 primes at least g.

## One field type, alone: (g, left member, order 2), kernel round 52 (2026-09-14)

Owner: I group fields and get stuck on the group. So one type at a time. The first, in the
kernel (proofs/MirrorWalkFieldType.lean, built, 0 sorries, standard axioms), three parts:
- the class (left_class_iff): g | E + 6h iff h = a (mod g), where 6a = -E (mod g); g coprime
  to 6, so every gear from 5.
- the line (left_member_on_line): h = a + g t gives E + 6h = g (c_0 + 6t), g c_0 = E + 6a. The
  cofactor runs on the column line c_0 + 6t.
- the shape (left_order_two_iff): the type takes h iff c_0 + 6t is prime. A smaller gear p takes
  the cofactor exactly on 6t = -c_0 (mod p) (cofactor_taken_by_iff), and a prime cofactor at
  least g has no such divisor (order_two_excludes_smaller): the two never overlap.
The same three lemmas serve any gear (base, band or high), the right member (a replaced by the
class of E + 2) and the down step (sign of 6). Next types: (g, left, order 3) with the cofactor
a product of two primes at least g; then the right member; then the down step.

## Next type alone: (g, left member, order 3), kernel round 52 continued (2026-09-14)

- Shape (left_order_three_iff): the left member is g times two primes at least g iff some prime
  p_1 >= g divides the cofactor c_0 + 6t with a prime quotient at least g.
- Line one level down (quotient_on_line): inside p_1's class t = t_1 + p_1 s the quotient is
  d_0 + 6s with p_1 d_0 = c_0 + 6 t_1. A column line again. So order 3 of g = for some p_1 >= g,
  the order-2 shape on the quotient line: the type is the last type, one level down.
- Table (type_order3.py). q = 499, g = 11, class h = 6 (mod 11), c = 10885 + 6t: order-3 kills
  at t = 11 (c = 47 * 233, p_1's class t = 11 mod 47, s = 0, quotient line 233 + 6s), t = 13
  (19 * 577; class 13 mod 19; 577 + 6s), t = 43 (11 * 1013; class 10 mod 11, s = 3; 995 + 6s).
  q = 1999, g = 11: ten order-3 kills, p_1 from 13 to 191, each with its class of t and quotient
  line; e.g. t = 150: c = 17 * 2243, class 14 mod 17, s = 8, line 2195 + 6s.
- Order j in general is the same descent j - 2 times: each factor taken in turn on its own
  class of the current line, the last quotient prime. Nothing new appears past order 3.

## Right member, down step, and one high gear's type (2026-09-14, kernel round 52 continued)

- Right member, step up: class h = b (mod g) with 6b = -(E + 2) (right_class_iff); line
  E + 6h + 2 = g (c_0 + 6t) with g c_0 = E + 2 + 6b (right_member_on_line). Same shapes as the
  left member with the class moved.
- Step down: class h = a' with 6a' = E (down_class_iff); line E - 6h = g (c_0 - 6t) with
  g c_0 = E - 6a' (down_member_on_line). The cofactor line runs downward.
- A high gear's type alone: with g^2 > q and the member at most q^2, the cofactor cannot be
  three primes at least g (high_gear_no_order_four: g^4 <= g c <= q^2 < g^4). So a high gear
  has types of order 2 and 3 only, exactly as measured (9,139 and 556, none higher).
- Table (type_right_down.py), q = 499, g = 11: right member up, class h = 2 (mod 11),
  c = 10883 + 6t: order 2 at t = 9, 15, 39; order 3 at 21 (101 * 109); order 4 at 25
  (11 * 17 * 59); gears 5, 7 at 7, 37 and 19. Left member down, class 5 (mod 11), c = 10879 - 6t:
  order 2 at t = 28, 36, 42; order 3 at 12, 16; gears 5, 7 at 24, 34 and 6. Right member down,
  class 9 (mod 11), c = 10877 - 6t: order 2 at t = 4, 28; order 3 at 14, 20, 34, 40; gears 5, 7
  at 2, 22 and 8.
All built, 0 sorries, standard axioms.

## First pair of types: base gear 5 with band gear 11 on the landing line (2026-09-14)

Script research/stack/r8/pair_base_band.py, table results_pair_base_band_499_11.txt. Two anchors,
not one: gear 5 (base) sees column h itself, teeth at h = 1 (left member) and 4 (right) mod 5 at
every machine with 5 in the base; gear 11 (band) sees the landing, teeth at h = 6 (left) and 2
(right) mod 11 at q = 499, fixed by E mod 11. One row per high gear: where column h stands
against gear 5's teeth, where the landing stands against gear 11's teeth. Pair survivors miss
both; what then takes a survivor is another single type (gear 7 the most often, then the high
gears 13, 17, 31, 79, 83, 223, 257, 293 at order 2). The joint pattern of the pair is the
alignment of the two anchors: column h against E.

## Second pair, same anchor: base gears 5 and 7 on column h (2026-09-14)

Kernel MirrorWalkFieldType (teeth_five, teeth_seven, pair_five_seven; built, 0 sorries):
- gear 5's teeth on the h-line: h = 1 (left member), 4 (right) mod 5; gear 7's: h = 6 (left),
  1 (right) mod 7; column h open to both iff h avoids 1, 4 mod 5 and 1, 6 mod 7: fifteen
  classes mod 35, the same at every machine with 5 and 7 in the base (E = -1 mod 35).
- Measured (pair_same_anchor.py) at q = 499, 997, 1999: the high gears the pair leaves occupy
  exactly the classes 2, 3, 12, 17, 18, 23, 32, 33 mod 35 at all three machines; the other
  seven open classes (0, 5, 7, 10, 25, 28, 30) are multiples of 5 or 7 and hold no gear.
  So for gears the pair leaves eight classes of 35, fixed.

## Third pair, cross anchor: band gear 11 with high gear 13 at q = 499

- Both anchored on E: gear 11's teeth h = 6 (left), 2 (right) mod 11; gear 13's teeth h = 3
  (left), 7 (right) mod 13; both pairs of classes fixed by E = 119699 mod 11 and mod 13.
- The pair leaves 58 of the high gears in reach, spread over 51 classes mod 143.
- Same shape as the (5, 7) pair, with the teeth placed by E instead of by the gear alone.
  Gear 13 is itself a high gear: its landing (h = 13) is struck by 13 only if 13 | E or E + 2,
  which is the class test again.

## Fourth pair, cross anchor across machines: base gear 5 with gear 13; and the own gear (2026-09-14)

Script research/stack/r8/pair_base_high.py, table results_pair_base_high.txt; kernel
own_landing_iff (built, 0 sorries).
- Gear 5's teeth fixed at h = 1, 4 mod 5 at every machine from 61 (5 in the base). Gear 13's
  teeth move with E: q = 61, E mod 13 = 3, teeth 6, 10; q = 67, E mod 13 = 11, teeth 9, 0;
  q = 73 and 89, E mod 13 = 1, teeth 2, 6; q = 101, 127, 167, E mod 13 = 7, teeth 1, 5; q = 499,
  E mod 13 = 8, teeth 3, 7. Same E mod 13 gives the same teeth, whatever the machine.
- The pair leaves, at q = 61 to 167, between 6 and 16 of the high gears in reach; the gears
  17, 37, 43, 47 (h = 2 or 3 mod 5) recur as survivors whenever 13's teeth miss their class.
- The own gear: h strikes its own landing E + 6h iff h | E (left) or h | E + 2 (right)
  (own_landing_iff), so the step's gear tests only E. Measured: at q = 61 to 167 and 499, 997
  the own strikes are (67, 13), (73, 17), (79, 29), (89, 17), (97, 43), (97, 67), (101, 11),
  (101, 17), (107, 11), (107, 13), (127, 17), (127, 37), (127, 107), (131, 47), (131, 83),
  (137, 29), (149, 19), (149, 37), (151, 41), (151, 101), (997, 227), (997, 877); each h divides
  E or E + 2. Gear 13 in the pair above is exempt from its own teeth for that reason.

## A triple: the fixed pair (5, 7) on column h with one E-anchored gear (2026-09-14)

Kernel triple_five_seven_iff (built, 0 sorries): with E = -1 mod 35 and gear g's teeth a, b
placed by E, the landing E + 6h is open to 5, 7 and g iff h avoids 1, 4 mod 5; 6, 1 mod 7; a, b
mod g. Composition of the pair lemma with the two class lemmas; nothing new enters.
Table triple_5_7_g.py with g = 11: q = 499 (teeth 6, 2) leaves 26 high gears, 6 twins among
them (37, 47, 173, 227, 313, 467), the rest each taken by one next single type (gear 13 six
times, 17 four times, 31 twice, then 19, 43, 61, 79, 83, 223, 257, 293 once); q = 997 (teeth 9,
5) leaves 41; q = 1999 (teeth 7, 3) leaves 76. The survivors' classes mod 385 differ by
machine exactly as gear 11's teeth move; the base part of each class is fixed.

## One machine, one gear at a time (2026-09-14)

Script research/stack/r8/one_gear_at_a_time.py, tables results_one_gear_at_a_time_499.txt and
_997.txt. q = 499, E = 119699, 87 high gears in reach, step up. Each gear in the machine's
order, its teeth on the h-line, what it takes from what remains:
- gear 5 (base, teeth 1, 4): takes 43, 44 remain. gear 7 (base, teeth 6, 1): takes 13, 31 remain.
- gear 11 (band, teeth 6, 2): takes 5, 26 remain. gear 13 (teeth 3, 7): takes 6, 20 remain.
  gear 17 (teeth 11, 5): takes 4, 16 remain. gear 19 (teeth 16, 3): takes 1, 15 remain.
- high gears, each taking at most one: 31 takes 257 and 443 (cofactors 3911, 3947, both prime);
  43 takes 353; 61 takes 373 (cofactor 1999); 79 takes 103; 83 takes 67; 223 takes 157
  (cofactor 541); 257 takes 53 (cofactor 467); 293 takes 23 (cofactor 409). Every other high
  gear takes nothing. Every high-gear kill is order 2 here.
- passing gears: 37, 47, 173, 227, 313, 467.
A high gear g's teeth are two positions a, b mod g; with h < g the tooth is the single gear
h = a or h = b, so a high gear can take only the gears standing exactly on its two teeth. Gear
293's teeth at 23 and 218: 23 is a gear and was taken; 218 is not a gear.

## A high gear's teeth as positions (2026-09-14)

Kernel (MirrorWalkFieldType, built, 0 sorries): tooth_unique (two points of one class closer
than g are one point), tooth_at_most_two (a gear above q/2 holds at most two points per tooth
in (0, q]). Table research/stack/r8/high_teeth_positions.py, results_high_teeth_positions_499.txt:
every high gear's tooth positions in reach, marked gear / not gear / standing when its turn came.
- q = 499: gear 31 (smallest high) has 16 positions per tooth, takes 257 and 443 (left tooth
  9 + 31t at t = 8 and 14). Gear 43: right tooth 9 + 43t holds 353 at t = 8. Gear 61: right
  tooth 7 + 61t holds 373 at t = 6. Gear 79: left tooth 24 + 79t holds 103 at t = 1. Gear 83:
  left tooth 67 at t = 0. Gear 223: right tooth 157 at t = 0. Gear 257 (above q/2): right tooth
  53 at t = 0, left tooth 139 (already taken by the base). Gear 293: left tooth 23 at t = 0;
  right tooth 218 is not a gear.
- Above q/2 the teeth are one or two positions each, most of them not gears (172, 423, 88, 339
  for gear 251; 225 and 40, 317 for 277). The gears among them were mostly taken already by the
  base or band (139, 257, 449, 59, 149, 317, 379, 167, 23), so the high gear's own kill is the
  exception: only 257 and 293 above q/2 take a standing gear at q = 499.
- So a high gear's field on the landing line is finite and explicit from E: the positions
  a_g + g t and b_g + g t up to q, at most two each above q/2, and it kills only where such a
  position is a gear still standing.

## The band gears' teeth as positions; the whole line written from E (2026-09-14)

Script research/stack/r8/band_teeth_positions.py, table results_band_teeth_positions_499.txt.
q = 499, E = 119699, every tooth of every gear up to sqrt q written as positions on the h-line,
each position marked not a gear / gear taken earlier by which gear / gear taken now.
- gear 5, left tooth 1 + 5t: the gears 31, 41, 61, 71, 101, 131, 151, 181, 191, 211, 241, 251,
  ...; right tooth 4 + 5t: 29, 59, 79, 89, 109, 139, 149, 179, 199, 229, 239, ... Fixed teeth,
  every fifth position, gears wherever the position is prime.
- gear 7, left 6 + 7t: takes 83, 97, 167, 223, 293, 307, ...; passes 41, 139, 181, 251, 349
  (taken by 5 already). Right 1 + 7t: takes 43, 113, 127, 197, 337; passes 29, 71, 211, 239, 281.
- gear 11 (band), left 6 + 11t: positions 28, 39, 50, 61, 72, 83, ..., 490; gears on them 61,
  83, 127, 149, 281, 479 (taken by 5 or 7 already), 193, 347, 457 (taken now). Right 2 + 11t:
  233, 277 taken now; 79, 101, 167, 211, 409, 431 already gone.
- gear 13, left 3 + 13t: takes 107, 263, 367; right 7 + 13t: 137, 163, 397. gear 17, left
  11 + 17t: 283, 317, 487; right 5 + 17t: 73. gear 19, left 16 + 19t: nothing new (73 by 17,
  149 by 5, 263 by 13, 491 by 5); right 3 + 19t: 383.
- Standing after the sub-machine: 23, 37, 47, 53, 67, 103, 157, 173, 227, 257, 313, 353, 373,
  443, 467 (15). Then the high gears' positions (previous section) take 23, 53, 67, 103, 157,
  257, 353, 373, 443, leaving 37, 47, 173, 227, 313, 467.
So at one machine the whole test is a list of positions from E: base teeth fixed, band teeth at
a_g + g t and b_g + g t across the line, high teeth at a few positions each. A gear passes iff
it stands on none of the positions. Nothing implicit remains at a given machine.

## Approaches A and B by field type (owner, 2026-09-14)

Owner: the crux is the residues carried across the flip; know which field types can bite at
the mirrored position from the mirror size and the carried residues; (A) exclude the dangerous
types at the final step and pick a safe axis; (B) build the spiral from gears of one type.

A (research/stack/r8/final_step_types.py, results_final_step_types.txt, machines 11 to 2000):
- Exclusion 1, no E residue used: base gears (E = -1 mod g) bite the landing iff they bite
  column h itself, so drop every h whose own column is painted by a base gear. Candidates
  86,396 become 30,710.
- Exclusion 2, no E residue on the gear: the axis gear bites iff h | E or h | E + 2. Drops 320
  more: 30,390.
- The rest is charged to the E-placed types: a gear at most sqrt q outside the base takes
  17,158 (products:2 and up), a gear above sqrt q takes 9,593 (products:2 or 3), 3,639 pass.
  A passing h exists at every machine but 11. The type rules alone do not pick the axis; the
  E-placed gears still decide. Among survivors whose own column is a twin (8,703), 1,076 pass.

B (research/stack/r8/spiral_types.py, results_spiral_types.txt, 299 machines 11 to 2000; base
kept, spiral over one type of gear above the base, E' = -1 + 2 P A'):
- all (the primorial spiral): E' in the window 299/299, E' itself a twin 39, passing final
  step 298 (not 11).
- coltwin (gears whose own column is a twin): 299 / 62 / 298.
- sqin (only gears above sqrt q): 299 / 42 / 299: a passing final step at EVERY machine,
  11 included (q = 11: base {2}, E' = 35, h = 11 up lands (101, 103)).
- sqout (only gears at most sqrt q above the base): in window 292 of 295, E' itself a twin
  135 of 295, passing 295 of 295. The landing is low (E' = 1679 at 499, 7559 at 1999, 8819 at
  997) because A' is an alternating sum of gears below sqrt q.
- twinmem 299 / 32 / 298; solo (295 machines) 295 / 43 / 293.
- Sample q = 499: all 119699 (twin, pass 11); coltwin 114659 (twin, 11); sqin 121379 (13 paints
  the right member, 12); sqout 1679 = 23 * 73 with 1681 = 41^2 (15 pass, first 53 up); twinmem
  18479 (17 | left, 12); solo 126419 (167 | left, 9).
Two leads from B: the sqin spiral (gears above sqrt q only) has a passing final step at every
machine tested including 11; the sqout spiral (gears at most sqrt q) lands on a twin itself at
135 of 295 machines and low in the window.

## More spiral constructions by type (owner: keep trying; 2026-09-14)

Scripts research/stack/r8/spiral_variants2.py, spiral_stacks2.py, spiral_sides.py; tables
results_*.txt; machines 11 to 2000 (299; 295 where the type has gears). Columns: landing in the
window / landing itself a twin / machines with a passing final step.
- Shape: only descending order with the first flip up keeps the landing in the window, for
  every gear set. Ascending, first flip down, twin pairs as one mirror, and two periods all
  leave the window at many machines (e.g. sqin desc down: 0 in window; pairs: 0).
- sqout desc up: 292 / 135 / 295 of 295. sqin desc up: 299 / 42 / 299, no failure, 11 included.
- Stacks: sqout then sqin (both up) = sqin then sqout = E of the full spiral shifted, 295 / 39 /
  295. sqin then sqout with the second flip down returns exactly the primorial spiral's E.
- The sub-machine's own primorial spiral (base_s with product at most sqrt(q)/2, gears up to
  sqrt q): lands in the sub-window (sqrt q, q] at 297 of 299 and ON A TWIN at 112 of 299
  (q = 499: E_s = 71; q = 1999: E_s = 239; both twins). Its final step passes at 276 only (the
  landing is low, so E_s + 6h with h > sqrt q sits low in the window). sub then sqin up:
  298 / 37 / 276.
- By the side of the gear's own column: left gears (5 mod 6) 299 / 52 / 297 (fails 11, 71);
  right gears (1 mod 6) 299 / 36 / 298; leftsq 270 / 39 / 281 of 281 (no failure); rightsq
  287 / 79 / 288 of 288 (no failure); lefthi and righthi like the full sets; twinL, twinR like
  the full spiral.
Standing: every set of gears at most sqrt q (sqout, leftsq, rightsq) gives a passing final
step at every machine where the set is nonempty; sqin does the same over the whole range. The
landings of the small-gear spirals sit low in the window and are twins themselves far more
often (sqout 135, rightsq 79 of 288, sub-machine 112 of 299).

## Groups by the base gear striking the column; the sub-machine spiral iterated (2026-09-14)

Script research/stack/r8/spiral_groups_levels.py, table results_spiral_groups_levels.txt.
- Groups (gears above the base by which base gear paints their own column): open (no base
  gear), by5, by7, by57. Spirals over each, descending first up: open 286 / 46 / 286, by5 286 /
  42 / 286, by7 222 / 22 / 222, by57 222 / 40 / 222 (in window / landing a twin / passing final
  step, over the machines where the group is nonempty). Every group passes wherever it exists.
  q = 499: open (32 gears) E = 83159, by5 (29) 82739, by7 (14) 91559, by57 (16) 120959.
- Levels (sub of sub): b_0 = q, b_1 = floor sqrt q, b_2 = floor sqrt b_1 while at least 5;
  level k has base_k = the lowest gears with product at most b_k/2, always holding 2 and 3 (so
  every landing is a left member, E = 5 mod 6), and gears in (b_{k+1}, b_k] outside base_k.
  From the deepest level up, each level's spiral runs from the previous landing, first flip up.
  Level 2 (gears 5 or 5, 7 with base {2, 3}) lands at 59 or 23, twins. Level 1 added (the
  sub-machine's spiral from there): in the sub-window (sqrt q, q] at 288 of 294, itself a twin
  at 99 of 294, passing final step at all 294. Level 0 added (gears above sqrt q with the
  machine's base): in the window 299 of 299, twin 25, passing final step at ALL 299, 11
  included. q = 499: 71 (twin) then 121451 (pass 9, first 83 up); q = 1999: 59, 359, 417419
  (pass 22, first 197 up).
Without 3 forced into the deepest base the levels lose the column alignment (E = 1 mod 6) and
the final step never passes: the base must carry 2 and 3 at every level.

## The final step one level down; sqout at level 0 (owner: both; 2026-09-14)

Script research/stack/r8/spiral_levels2.py, table results_spiral_levels2.txt, 294 machines
with a sub-machine (sqrt q at least 5).
- A. From the level-1 landing E_1 (in the sub-window (sqrt q, q]), the final step with a
  sub-machine gear h at most sqrt q, landing E_1 +- 6h inside the sub-machine's window: passes
  at 292 of 294. The two failures, 29 and 31, have b_1 = 5 and the level-1 spiral over the single
  gear 5 overshoots the sub-window (E_1 = 59 > 25). Examples: q = 499, E_1 = 71, h = 5 up lands
  (101, 103); q = 997, E_1 = 227, h = 7 up lands (269, 271); q = 1999, E_1 = 359, h = 17 up lands
  (461, 463). So the same walk, run inside the sub-machine, finds the sub-machine's twin.
- B. Level 0 run with the sqout set (gears at most sqrt q outside the machine's base) instead of
  sqin: landing in the window 292 of 294, itself a twin 9, passing final step at all 294.
  q = 499: 1751, 14 pass, first 41 up; q = 1999: 7919, 35 pass, first 103 up.
- C. From E_1 directly, the final step with h above sqrt q: E_1 is below the window (6 in), yet
  the step E_1 + 6h lands inside it and passes at all 294 (q = 499: first 83 up; q = 1999: first
  313 up). The level-0 spiral is not needed for the final step to pass: the sub-machine's landing
  plus one flip with a high gear reaches a twin of the machine's window at every machine tested.

## The primorial descent (owner, 2026-09-14)

Owner: a walk that carries all residues from an open pair (the primorial does) but converges
in the window: from the first pair flip on the machine's primorial, then flip back on partial
primorials (without q, without q - 1, ...) one or more times.

Built (research/stack/r8/primorial_descent.py, results_primorial_descent.txt): from home flip up
on q# (one period), then down on q#/q with k_1 periods, on q#/(q q') with k_2, ..., stopping at
the first primorial P_s above q/2. Every flip is a real primorial axis. The landing is
2 t P_s - 1 with t chosen by the periods; it carries residue -1 at every gear of P_s (open to
the base by construction) and lies in the window iff (q + 1)/(2 P_s) <= t <= (q^2 - 1)/(2 P_s).
- A twin among the landings at all 299 machines 11 to 2000.
- q = 11: P_s = 6, t in 1..10, twins at t = 1, 5, 6, 9 (first landing (11, 13)); descent: q#
  = 2310, then down on 210 with 10 periods, on 30 with 6, on 6 with 4: 2310 - 2100 - 180 - 24
  = 6 = 1 * P_s.
- q = 13, 31: P_s = 30, first twin t = 1, landing (59, 61). q = 101: P_s = 210, t = 1, (419, 421).
- q = 499, 997, 1999: P_s = 2310, twins at t = 2, 12, 17, 20, 24, 29, 33, 39, 49, 50, 53, ...
  first landing (9239, 9241); the failing t and their striking gears are the same list at all
  three machines: t = 1 by 31, 3 by 83, 4 by 17, 5 by 13, 6 by 19, 7 by 73, 8 by 13, ...
- Why the list is the same: a gear g outside the base strikes 2 t P_s - 1 iff t = (2 P_s)^{-1}
  (mod g) and 2 t P_s + 1 iff t = -(2 P_s)^{-1} (mod g): two classes of t fixed by g and P_s
  alone, no landing-dependent residue anywhere. The E-dependence of the spiral's classes is
  gone; the classes are the gear's own against the primorial.
- What remains is the same statement in its cleanest dress: among t from about q/(2 P_s) to
  q^2/(2 P_s), one t outside the two fixed classes of every gear g above the base with g <= q
  (gears above q are not in the machine). The family 2 t P_s - 1 is the set of columns open to
  the base; the descent reaches every one of them in the window.

## The descent in the kernel, and its teeth on the t-line (2026-09-14)

Kernel proofs/MirrorWalkDescent.lean (round 54, built, 0 sorries, standard axioms):
- descentEnd_eq: the landing of up on Q then down on (P_i, k_i) is -1 + 2 (Q - sum k_i P_i).
- descentEnd_form: with P_s dividing Q and every P_i, the landing is 2 t P_s - 1 for an integer t.
- descent_open_base: the landing is open to every gear dividing P_s (residue -1 carried).
- descent_strikes_iff: gear g strikes the left member iff 2 P_s t = 1 (mod g), the right iff
  2 P_s t = -1 (mod g): two classes of t fixed by g and P_s.
- descent_in_window: in the window iff q + 1 < 2 t P_s and 2 t P_s + 1 <= q^2.

Teeth on the t-line (research/stack/r8/descent_classes.py, results_descent_classes_210.txt and
_2310.txt): with the gears taken in order, each gear's two teeth and the t it takes.
- P_s = 210 (base 2, 3, 5, 7), t = 1..60: gear 11 teeth 6, 5 takes 5, 6, 16, 17, 27, 28, 38, 39,
  49, 50, 60; gear 13 teeth 10, 3 takes 3, 10, 23, 29, 36, 42, 55; 17 (10, 7) takes 7, 24, 41,
  44, 58; 19 (10, 9) takes 9, 47, 48; 23 (4, 19) takes 4, 19; 29 (27, 2) takes 2, 31, 56; ...;
  standing t = 1, 8, 14, 15, 18, 21, 22, 25, 43, 52, landings (419, 421), (3359, 3361),
  (5879, 5881), (6299, 6301), (7559, 7561), (8819, 8821), (9239, 9241), (10499, 10501), ...
- P_s = 2310 (base to 11), t = 1..100: gear 13 teeth 8, 5 takes 16 of the 100; 17 (4, 13)
  takes 10; 19 (13, 6) takes 6; 23 (15, 8) takes 6; 29 (13, 16) takes 5; 31 (1, 30) takes 1, 92,
  94; gears from 37 take 1 to 3 each; standing t = 2, 12, 17, 20, 24, 29, 33, 39, 49, 50, 53, 58,
  65, 67, 68, 88, 90, 91, 93, 97, landings (9239, 9241), (55439, 55441), (78539, 78541), ...
- The teeth are pairs of residues symmetric about 0: right tooth = -(left tooth) mod g (gear 13
  at 2310: 8 and 5 = -8; gear 31: 1 and 30). So each gear's two teeth on the t-line are t and
  -t: the descent's landing family is symmetric under t -> -t modulo every gear, the mirror
  property of the primorial P_s seen on the t-line.

## The t-line as the fields construction (2026-09-14)

Kernel: descent_teeth_symmetric (g strikes the left member at t iff the right member at -t) and
descent_reflect (the landing at -t is the reflection of the landing at t about home, member for
member), MirrorWalkDescent, built, 0 sorries.
Grid (research/stack/r8/descent_fields.py, results_descent_fields_210.txt and _2310.txt): rows
= gears above the base, columns = t, L/R where the gear divides the left/right member; the twin
columns are the unpainted ones (checked against primality at every t).
P_s = 210, t = 1..60, rows 11 to 157:
    11 ....RL.........RL.........RL.........RL.........RL.........R
    13 ..R......L.....R......L.....R......L.....R......L.....R.....
    17 ......R..L.............R..L.............R..L.............R..
    19 ........RL.................RL.................RL............
    23 ...L..............R.......L..............R.......L..........
    29 .R........................L...R........................L...R
  twin T      T     TT  T  TT  T                 T        T
- Each row is a stripe of period g with two marks per period, L at t_0 and R at -t_0.
- The gap from the L tooth to the R tooth inside a period is P_s^{-1} mod g (2 t_0 = P_s^{-1}):
  gears with P_s = 1 (mod g) have the two teeth adjacent (11 and 19 at P_s = 210: 210 = 1 mod 11
  and mod 19, rows show RL side by side); gear 13 has gap 2^{-1} = 7 (teeth 3 and 10); 17 has gap
  3 (7 and 10); 23 gap 15 (4 and 19).
- So on the t-line every row is the multiples field of g folded to period g with its two teeth
  placed by P_s mod g alone; the twins are the columns no row reaches. The whole picture for a
  given P_s serves every machine q with q/2 < P_s <= q^2/2, the window fixing only which columns
  t are in reach.

## The gap rule in the kernel; the t-line at P_s = 30030 (2026-09-14)

- descent_gap: with t_0 the left tooth (2 P_s t_0 = 1 mod g), P_s (t_0 - (-t_0)) = 1 mod g: the
  gap between a gear's two teeth is the inverse of P_s modulo g. descent_gap_one: when P_s = 1
  mod g the teeth are adjacent (2 t_0 = 1 mod g). Built, 0 sorries.
- Grid at P_s = 30030 (base 2 to 13), t = 1..150, 145 rows from 17 to 3001
  (results_descent_fields_30030.txt): row 17 "R..............L.R.............." (teeth 1 and 16,
  gap 15 = 30030^{-1} mod 17); row 19 "L................R.L" (teeth 1 and 18, adjacent across
  the period: 30030 = 1 mod 19); row 23 teeth 10 and 13; 29 teeth 1 and 28 (30030 = 1 mod 29);
  31 teeth 12 and 19; 37 teeth 4 and 33; ... Twin columns in t = 1..150: 3, 5, 7, 9, 11, 14, 21,
  25, 27, 31, 32, 46, 61, 62, 65, 68, 71, 91, 98, 106, 109, 114, 139, 140, 141.
- Gears g with P_s = 1 mod g (19, 29 at 30030; 11, 19 at 210) have their teeth at t = 1 and
  t = -1: they strike the landing at t = 1 (2 P_s - 1 and 2 P_s + 1 are the two members with
  P_s = 1 mod g giving 2 P_s + 1 = 3 mod g... exactly: the L tooth at t_0 = 2^{-1} P_s^{-1} = 2^{-1}
  when P_s = 1). The picture is the same object at every primorial: stripes of period g, two
  teeth each at t_0 and -t_0, t_0 = (2 P_s)^{-1} mod g.

## The first gap of each primorial against the smallest machine in its range (2026-09-14)

research/stack/r8/descent_first_gap.py, results_descent_first_gap.txt. P_s serves the machines
2 P_prev <= q < 2 P_s; at the smallest, q_min, the reach on the t-line is (q_min^2 - 1)/(2 P_s),
about 2 P_prev / p_s, and it only grows with q.
  P_s = 30 (q_min 13): reach 2, first gap t = 1, landing (59, 61).
  210 (61): reach 8, first gap 1, (419, 421); gaps in reach 1, 8.
  2310 (421): reach 38, first gap 2, (9239, 9241); gaps 2, 12, 17, 20, 24, 29, 33.
  30030 (4621): reach 355, first gap 3, (180179, 180181); gaps 3, 5, 7, 9, 11, 14, 21, 25, ...
  510510 (60077): reach 3534, first gap 4, (4084079, 4084081).
  9699690 (1021043): reach 53740, first gap 12.
  223092870 (19399411): reach 843454, first gap 2.
  6469693230 (446185769): reach 15385717, first gap 8.
  200560490130 (12939386503): reach 417399566, first gap 11.
  7420738134810 (401120980261): reach 10841107574, first gap 2.
The first gap sits at t <= 12 at every primorial to 37#, while the reach at the smallest machine
grows as 2 P_prev / p_s. So the descent's landing on a twin at every machine reduces to: the
stripes of P_s have a gap at some t at most (q_min^2 - 1)/(2 P_s), which the first ten primorials
meet at t <= 12. Exact statement, no landing in it; the gap's existence is the open item.

## (b) The columns t = 1, 2, ... of each primorial, read gear by gear (2026-09-14)

research/stack/r8/descent_small_t.py, results_descent_small_t.txt. Column t is the pair
(2 t P_s - 1, 2 t P_s + 1). Below the first gap every painting gear has its tooth exactly at t
(g > t, so t is on a tooth iff the tooth is t): no periodic small gear reaches these columns,
because every gear above the base is larger than the first gap's t (at most 12).
- P_s = 30: t = 1 is the gap (59, 61). 210: t = 1 (419, 421).
- 2310: t = 1 is 31 * 149 on the left (teeth of 31 and 149 both at 1); t = 2 gap (9239, 9241).
- 30030: t = 1 painted by 17 R, 19 L, 29 L, 109 L, 3533 R; t = 2 by 113 L, 1063 L; t = 3 gap.
- 510510: t = 1 by 181, 5641 (R); t = 2 by 1429 R (1429^2); t = 3 by 1451, 2111 (R); t = 4 gap.
- 9699690: eleven columns painted, each by 2 to 6 gears with their tooth there (t = 7: 29, 37,
  151, 271, 467, 899309); t = 12 gap. 223092870: t = 1 by 41, 97, 191, 24083, 10882579; t = 2 gap.
  6469693230: t = 8 gap. 200560490130: t = 11 gap.
- So at small t the picture is not stripes at all: it is a list of gears whose tooth happens to
  sit at that t, i.e. the prime factors of 2 t P_s +- 1 above the base. The first gap is the
  first t at which both 2 t P_s - 1 and 2 t P_s + 1 have no factor at all; the stripes (periodic
  teeth) only begin to matter once t passes the smallest gear above the base (13 at 2310, 17 at
  30030), which is beyond the first gap at every primorial to 31#.

## The one-mirror stack across machine sizes and base sets (owner, 2026-09-15)

research/stack/r8/stack_one_mirror.py (pictures, span tables), stack_bases.py
(results_stack_bases.txt; 75 machines 31 to 1999; bases null, {2}, {2,3}, {2,3,5}, {2,3,5,7},
spiral). Layer g = the sieve of base + {g} on the slot line from the origin to its landing
2 P g - 1 (mirrored once about P g; the sieve is symmetric about P g, so the mirror image is the
sieve itself). Stack = layers over each other; hole = slot no layer marks; true hole = twin.
- Exact: a hole is a twin as long as every layer is still active; false holes come only from
  layers that have ended. Span 0 (window start to the first landing) has holes = twins at every
  base and machine (spiral: 4567 of 4567; {2,3,5,7}: 6205 of 6205).
- Null base and {2}: the landings 2g - 1 and 4g - 1 sit at or below the window start for almost
  every layer, so the stack marks almost nothing inside the window (null: 11787 holes of 11817
  slots; {2}: 35380 of 35408). The stack needs 2 and 3 in the base to say anything about slots.
- {2,3}: first landings 12 g - 1 are below q for most layers: span 0 holds a twin at 2 of 75.
  {2,3,5}: 18 of 75 (span 0 = (q, 419] is empty once q > 419). {2,3,5,7}: 75 of 75 in this sample,
  but P = 210 is above q/2 below q = 421, where span 0 is the whole window.
- Spiral base (product at most q/2): span 0 = (q, 2 P g_min - 1] = (q, 2 P_s - 1] with P_s the
  first primorial above q/2, and it holds a twin at 75 of 75 sampled machines; checked at every
  machine 11 to 5000 it fails at 27: 11; 41 to 59 (span 0 ends at 59); 347 to 419 (ends at 419);
  4547 to 4603 (ends at 4619): the machines just below 2 P_s, where span 0 is short or empty.
- True-hole share falls span by span as layers end (spiral: span 1 928 of 936, span 3 784 of
  868, span 5 2321 of 2786, span 8 1487 of 2641, spans 12 and beyond 237921 of 1533172).
- The twin-gear spans (ending at 13, 31, 43, 61, ...) show no different twin share in aggregate
  (spiral: 18309 of 314116 slots against 239792 of 4421249): the q = 101 reading was noise.
Standing: the part of the stack most consistent in twin appearance is span 0, the stretch before
the first layer ends, where holes are twins by construction; it holds a twin at every machine
except those just below 2 P_s, where it is too short.

## The stack locator (owner, 2026-09-15: can we build a locator)

research/stack/r8/stack_locator.py, results_stack_locator.txt. Base = the largest primorial P
with 2 P g_min - 1 <= q^2; span 0 = (q, 2 P g_min - 1]; the locator walks the slots upward from
q and stops at the first slot no layer marks. Every layer is active on span 0, so the first hole
is a twin: this is section_twin_of_unstruck in the kernel (a slot below q^2 that no gear up to
q strikes is a twin prime pair), nothing new to prove for correctness.
- Every machine 11 to 5000 (665): a twin located at all 665. Slots walked: 5 at the median,
  29 at most. q = 101: base to 7, span 0 = (101, 4619], located (107, 109) after 2 slots.
  q = 499: base to 11, (499, 60059], located (521, 523) after 4. q = 1999: base to 13,
  (1999, 1021019], (2027, 2029) after 5. q = 4603: base to 17, (4603, 19399379], (4637, 4639)
  after 6.
- What it is: the first twin above q, found by testing slots against the gears in order. Its
  correctness is proved; it lands in the window by construction; that it finds anything is the
  statement that span 0 holds a hole, i.e. a twin in (q, 2 P g_min - 1], which covers most of
  the window: the window statement itself. It is also a scan, slot by slot, which is what the
  owner ruled out as a mechanism on 2026-09-13 (the repair walk).

## The spiral with the slip (owner's idea; run 2026-09-15)

research/stack/r8/spiral_slip.py, results_spiral_slip.txt, 299 machines 11 to 2000. Slip of a
step = the signed overshoot of its move d_i = +-2 P g_i past whole cycles of another gear (the
residue in (-g/2, g/2]); summed over the spiral against the previous gear (S_A), against every
earlier gear (S_B), against every gear outside the base (S_C). Candidates E +- S (nearest slot)
and E +- 6S, against the plain landing E.
- E itself: twin at 39 of 299. E + S_A 17, E - S_A 18, E + 6 S_A 22, E - 6 S_A 26; S_B variants
  23, 26, 20, 21; S_C variants 18, 29, 18, 13 (in the window 252 to 299).
- No slip sum does better than the plain landing; most do worse. q = 1999: E = 409499 (not a
  twin), E + S_A = 411821 twin, E + 6 S_A = 423461 twin, E + 6 S_C = 266897 twin, the rest not.
Verdict: the summed slip does not point to a twin. Closed.

## Slip as the difference of mirror sizes (owner's definition, 2026-09-15)

Slip = the difference between the products of two gear sets. In the spiral the mirrors are
base times gear, so at q = 31 (base 2, 3) the mirrors are 186, 174, 138, 114, 102, 78, 66, 42,
30 and the slips between consecutive steps 12, 36, 24, 12, 24, 12, 24, 12: sum 156, sum with the
flip directions' signs -12, landing 227. Over the 299 machines 11 to 2000, twins in the window:
the landing itself 39; landing + sum 29; landing - sum 48 of 182 in the window; landing + signed
sum 39; landing - signed sum 35; the column at the sum 31; at twice the sum 31; at the signed sum
26 of 117; at twice the signed sum 20 of 129. The plain sum of consecutive slips is the first
mirror minus the last (186 - 30 = 156 at q = 31), so "landing - sum" is the landing minus that
difference. No candidate stands out from the landing's own rate. Closed as run; open to another
pairing of the sets if the owner has one.

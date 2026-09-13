# The spiral (owner's find, 2026-09-13; formalised the same day)

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

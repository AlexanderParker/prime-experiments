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

# The recursive walk, as parts

2026-09-14. The walk found on 2026-09-14 (research/proof/spiral.md, sections "sub of sub" and
"the final step one level down"), written as parts with what each part has: proved (kernel
name), exact by construction, measured, or open. Nothing here is the walk as one object.

## The parts

### 1. Origin

- The home pair (-1, 1): column 0, open to every gear (kernel `home_open`, MirrorWalk.lean).

### 2. Levels

- Bounds b_0 = q, b_1 = floor sqrt q, b_2 = floor sqrt b_1, ... while b_k >= 5.
- Level k has base_k = the lowest gears with product at most b_k / 2, always holding 2 and 3
  (product P_k, 6 | P_k), and gears_k = the primes in (b_{k+1}, b_k] outside base_k, taken
  descending.
- Exact: base_{k+1} is contained in base_k, so P_{k+1} | P_k.

### 3. One level's spiral (a step of the walk)

- From the previous landing n, one flip per gear g of the level about the mirror {base_k, g},
  first flip up, alternating. Each flip moves the column by 2 P_k g in its direction.
- Proved: the landing is n + 2 P_k A_k with A_k the alternating sum of the level's gears
  (`spiralP_eq`, MirrorWalkLevels.lean, round 53).
- Proved: the landing is congruent to n modulo P_k (`spiralP_modEq`).
- Proved: a left member lands on a left member, since 6 | P_k (`spiralP_left_member`). This is
  why 2 and 3 must sit in every level's base: without 3 in the deepest base the landings came
  out as right members and no final flip passed (measured, spiral_groups_levels.py).
- Proved: the landing is at most n + 2 P_k g_1 with g_1 the level's largest gear
  (`spiralP_le`, from `altSum_le_head`).

### 4. The landings

- Deepest level (base {2, 3}, gears {5} or {5, 7}): landing 59 or 23. Exact, both twins.
- Level 1 landing E_1: in the sub-window (sqrt q, q] at 288 of 294 machines to 2000; itself a
  twin at 99 of 294 (measured). E_1 = -1 + sum over levels j >= 1 of 2 P_j A_j.
- Level 0 landing E_0 = E_1 + 2 P_0 A_0 (gears above sqrt q): in the window at 299 of 299,
  itself a twin at 25 (measured). E_0 = E_1 (mod P_0): the machine's base gears 5, 7 do NOT
  see column h at the final flip here (E_0 is not -1 mod P_0); the walk passes without that.

### 5. The final flip

- From a landing E, one flip about {3, h}, h a gear, landing (E + 6h, E + 6h + 2) (or E - 6h).
- Proved: gear g strikes the landing iff 6h = -E or -(E + 2) mod g (`strikes_landing_iff`,
  MirrorWalkFinal.lean); a landing every gear up to q misses, below q^2, is a twin prime pair
  (`final_step_twin`); the flip's own gear strikes only if h | E or h | E + 2 (`own_landing_iff`).
- Proved: from E <= q with h <= q the landing stays below q^2 once q >= 8
  (`final_flip_below_square`); it enters the window iff 6h > q - E (`final_flip_above_q`).
- Measured, from E_1 directly with h above sqrt q: a passing h at all 294 machines with a
  sub-machine (spiral_levels2.py, C). From E_0 with h above sqrt q: all 299 machines, 11
  included (spiral_groups_levels.py). From E_1 with a sub-machine gear h at most sqrt q, landing
  in the sub-machine's window: 292 of 294 (the two failures have b_1 = 5, where the level-1
  spiral over the single gear 5 overshoots the sub-window).

### 6. The target zone

- The machine's window (q, q^2]; one level down, the sub-machine's window (b_1, b_1^2]. A twin
  there is the window statement, which alone gives step 8 and infinitude
  (research/proof/proof_skeleton.md, Part IV).

## What is open

- Part 5, existence: that from the level landing some gear h of the level above gives a landing
  no gear strikes. The same statement one level down is the sub-machine's window statement,
  reached by the same walk, which is where the recursion would close if the final flip's
  existence were proved from the parts. Not proved. No counting.

## What the recursion offers that the single spiral did not

- Every step is the same object at every level: base with 2 and 3, descending spiral, one flip
  with a gear of the level above. The parts to prove are the same at every level.
- The landings are low: E_1 <= q, so the final flip's landing E_1 + 6h sits in the first sixth
  of the window's length and the flip enters the window iff 6h > q - E_1 (proved).
- The base gears of the machine no longer need E = -1 mod P_0; the alignment needed is only
  E = 5 mod 6, carried by 2 and 3 at every level (proved).

## The level rule, and the final flip in the fields (2026-09-14, later)

- Level rule: levels continue while b_k >= 11 (a level needs a window long enough for a flip of
  6h with h a gear of the level above: at b = 7 the window (7, 49] is shorter than 6 * 5 from
  the landing 23, at b = 5 the level over the single gear 5 overshoots). With this rule, machines
  11 to 2000 (research/stack/r8/levels_rule_and_final.py): top-level final flip passes at all
  299; of the 273 machines with a sub-machine, E_1 lies in the sub-window at 273, is itself a
  twin at 113, the final step one level down (a sub-machine gear, landing in the sub-machine's
  window) passes at 273, and the direct final flip from E_1 with h above sqrt q passes at 273.
  No failure of any part at any machine tested.
- The final flip from E_1 in the fields, q = 499 (E_1 = 71, the flip enters the window iff
  h > 71.3): h = 73 painted in row 7 (member 511 = 7 * 73, products:2); 79 row 5 (545 = 5 * 109);
  83 PASS (569, 571); 89 row 5 (605 = 5 * 11^2, products:3); 97 row 5; 101 row 7; 103 row 13
  (689 = 13 * 53); 107 row 5 (715 = 5 * 11 * 13); 109 row 5 (725 = 5^2 * 29); 113 row 7; then
  PASS at 131, 163, 193, 271, 313, 373, 431, 443 among the gears to 499 (9 of 87 pass). q = 1999
  (E_1 = 239, h > 293.3): PASS at 307, 317, 503, 557, 613, 653, 683, 733, 797, 863, 877, 1087,
  1433, 1783, 1913, 1987 (16 pass).
- The landings E_1 + 6h are at most 7q, in the first stretch of the window, where the products:j
  orders are 2 and 3 only and the painting rows are the small gears 5, 7, 11, 13: the fields
  that paint here are the multiples rows of the small gears, products:2 and products:3.

## The stretch (q, 7q] against the columns E_1 + 6h, by rows (2026-09-14, later)

Script research/stack/r8/stretch_rows.py, table results_stretch_rows.txt.
- A composite member of a landing at most 7q has its smallest gear at most sqrt(7q) (59 at
  q = 499, 118 at 1999). The rows that decide a column are therefore the gears up to sqrt(7q):
  the sub-machine's gears and the next few. Larger rows appear only as the cofactor of a
  smaller row's kill (the lower:g reading), never as the first painter.
- Row g paints the columns of h in two classes, a_g = -E_1 6^{-1} and b_g = -(E_1 + 2) 6^{-1}
  modulo g. q = 499, E_1 = 71: row 5 paints h = 4, 2 (mod 5), 39 columns of 75; row 7 paints
  1, 3 (mod 7), 25 columns; row 11 paints 1, 8 (mod 11), 13; row 13 paints 12, 3 (mod 13), 12;
  row 17 paints 8, 2 (mod 17), 10; rows 19 to 59 paint 1 to 8 columns each. Pass columns: h = 83,
  131, 163, 193, 271, 313, 373, 431, 443. q = 1999, E_1 = 239: row 5 paints h = 1, 4 (mod 5);
  row 7 paints 1, 3 (mod 7); row 11 paints 6, 2 (mod 11); ... rows up to 113 paint 3 to 7
  columns each; pass at 307, 317, 503, 557, 613, 653, 683, 733, 797, 863, 877, 1087, 1433, 1783,
  1913, 1987.
- So the final flip from E_1 is exactly: a gear h in ((q - E_1)/6, q] standing in none of the
  two classes of any row up to sqrt(7q). Three conditions on h per row g of the sub-machine
  (h itself not 0 mod g, since h is a gear; h not a_g; h not b_g), two per row between sqrt q
  and sqrt(7q). The stretch is the machine sqrt(7q)'s territory: the final flip is the window
  statement of the machine of size about 2.6 sqrt q, with the pattern (h, E_1 + 6h, E_1 + 6h + 2)
  in place of (n, n + 2).

## The pattern one level down, by the same walk: tested, does not close (2026-09-14, later)

- pattern_one_down.py: the machine m = largest prime at most sqrt(7q) run by its own walk lands
  at E_m in its window (m, m^2] which runs to about 7q, so its flips E_m +- 6h' reach the h-range
  ((q - E_1)/6, q] only when E_m happens to be low: a valid h for q located at 25 of 273
  machines (q = 997: E_m = 2399 > q, no candidate at all).
- pattern_one_down2.py: the sub-machine as locator, flips from its landing E_1 with its own gears
  h'': n = E_1 +- 6h'' located a valid h (n a gear of q in range with E_1 + 6n a twin) at 135 of
  273; with the mirror {2, 3, h''} (n = E_1 +- 12h'') at 232 of 273 (q = 499: h'' = 5 up, n = 131,
  landing (857, 859); q = 997: h'' = 11 up, n = 359, (2381, 2383); q = 1999: h'' = 37 up, n = 683,
  (4337, 4339)); fails cluster at the small machines 127 to 233 where the sub-machine has few
  gears. Not a closing rule.
- Standing: the final flip's h exists at every machine tested (273 of 273) but no walk of the
  level below locates it by a fixed rule yet. The open item is unchanged in content: existence
  of a gear h in ((q - E_1)/6, q] outside the two classes of every row up to sqrt(7q).

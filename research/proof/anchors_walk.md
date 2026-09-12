# Anchors for the mirror walk: squares, blind classes, caustics (2026-09-13)

Owner's direction: the walk needs steps that offset from a known location and land in a known
place with known properties; try three anchor families from the field work, one-off lanes with
explicit specs, manager does the inference. Lanes on Opus; scripts and result tables in
research/stack/r8/anchors_squares.py, anchors_blind.py, anchors_caustic.py and
results_anchors_*.md. Earlier steps: research/stack/r8/mirror_walk.py, backward_walk.py,
backward_walk_killed.py; the view docs/mirror_walk_view.html.

## The rules of the walk, as built

- A flip about an axis A/2 sends column n to A - n - 2 and carries exactly the gears dividing A
  (openness to those gears agrees at the two ends). Two flips compose to a translation.
- An anchor is a column a with a set K(a) of gears to which it is known open by construction.
  Base anchors: home (-1, 1), open to every gear; the gear pairs (g, g+2), open to every gear
  but their own two members.
- Gear h is certified at n by anchor a iff h is in K(a) and n = a or n = -a - 2 (mod h). A
  column is fully certified iff every gear is certified by some anchor. Certification never
  certifies a struck column (0 cases at every machine, every family).
- Backward walk of a killed column: the killer is carried exactly onto the gear pair containing
  it, on the matching side (0 violations at 13, 31, 101). Backward walk of a twin with the base
  anchors only: certifies gears up to about 2q / ln^2 q, never the large ones (0 twins fully
  certified at 31 .. 401): a gear h needs about h/2 anchors known open to it, the pairs number
  about q / ln^2 q.

## The three families, machines 31, 101, 211, 401, 1009

| family | anchors at 1009 | verification | gears fully covered | twins fully certified | walk length (anchors per twin) at 1009 | anchors that are twins |
|---|---|---|---|---|---|---|
| A squares: the g - 1 columns either side of g^2 not struck by g, K = {g} | 153,593 | 0 failures | all, every machine | all (8278 of 8278), by A alone | 132 / 144 / 159 for 167 gears | 7,719 of 153,593 |
| B blind classes: offsets 5, 10, 12, 17 mod 35 from every square g >= 11, K = {5, 7} | 72,701 | 0 failures | 5 and 7 only, complete from machine 31 | all for gears 5 and 7 by B alone; 0 overall | n/a | 8,299 of 72,701 |
| C caustics: the run from g^2 to gear h's first strike, K = {h}, merged over h | 73,759 | 0 failures | every gear >= 23; 5 never (0 of 3), 7 (3 of 5), 11, 13, 17, 19 one short | 0 by C alone (gear 5); all with the base anchors | 3 / 68 / 79 | 3,913 of 73,759 |

Mean number of gears one caustic anchor is known open to: 3.4, 7.2, 12.5, 20.5, 41.7 at q =
31, 101, 211, 401, 1009. Gear 5 has no caustic anchor because g^2 is 1 or 4 mod 5 for every g
other than 5, so 5 strikes the column right after any square (first strike at column offset 1).
Side check of the caustic law: the lane's transcription asked for the offset to be 0 mod 6,
which is wrong (the law's e is the offset of the struck member, g^2 + e in a twin slot, not a
column offset); read in the law's own form the recorded first strikes agree, e.g. (g, h) =
(11, 5): r = 1, e = 4, 121 + 4 = 125 = 5^3, column offset 1 as found.

## Inference

1. The square anchors make the walk complete for every gear at every size, with one anchor per
   gear, and the reason is plain: the g - 1 columns either side of g^2 run through every
   residue class mod g, so for any column n there is an anchor in n's class or its mirror
   class. Certification by square anchors is the residue rule with a witness attached: n is
   open to g iff n is not 0 or -2 mod g, and the anchor is the column near g^2 in the same
   class.
2. The blind classes supply the first non-twin known places that carry two gears at once
   (open to 5 and 7 by construction, wherever they are); the caustic runs supply columns known
   open to dozens of gears at once (the columns just after a square, before the first strikes),
   and they halve the walk length. Cross-gear knowledge is real and comes from the square
   origin; it is blind on gear 5 and thin on 7 to 19.
3. What the walk is and is not. Given a twin, a walk certifying it from known places exists,
   with about as many steps as gears (fewer with caustic anchors). The walk is a checker. To be
   a locator it would have to run the other way: choose, for every gear h, an anchor class
   (a_h or its mirror), and produce the column n with those residues. That column is fixed mod
   q# by the choices, and it lies in the window only for the choices whose combined residue
   is below q^2. Which combinations are small is the open count in another coordinate: the
   same wall, now stated as "which anchor choices compose to a column below q^2".
4. What is new and usable: (i) the kill side of the walk is exact (killer onto its own pair,
   matching side); (ii) the three anchor families are exact, verified sources of known-open
   columns per gear, with the caustic family carrying many gears per column; (iii) the walk
   length bound: a window twin is certified by at most one anchor per gear, and by about
   0.4 anchors per gear with caustic anchors at q = 1009.

## Corrected assessment (owner, 2026-09-13)

The verdict "the walk is a checker, not a locator" above was wrong in emphasis. The accurate
statement: the walk is a working locator algorithm. Its steps are proved (axis rule, kill
rule, certification rule). Started at (5, 7) with the square, blind-class and caustic anchors,
it lands on a window twin on every machine tried, and in its reduced form (the landing is the
column two after the square of a gear between sqrt q and q) on every machine to 20000. What
is missing is a termination proof: that a landing place exists in (sqrt q, q] for every q.
That termination proof is the proof of step 8; nothing else about the walk is open.

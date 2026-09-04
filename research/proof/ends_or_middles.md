# Branch R3.h - ENDS OR MIDDLES: what the longest open stretch is made from

Parent: theory_tree.md node 5 ("made at the top"), refined by 5d (the record set is pinned and
nearly unique at m29 and m31) and 5g (the gear-5 lock and the allocation law).  The branch is
opened by the human's question of 2026-09-04, restated: *what is the longest open stretch made
from - is it alignment of the ends, or the middle parts of the meta sieves?*  The meta sieves are
the lower machines {5..7}, {5..11}, ... seen as periodic patterns inside the record stretch.

Scripts in research/anchor235/r37/: h_ends_middles.py (period records and runner-ups, full
periods), h_window_decomp.py (the window's longest stretch at rungs 23..997),
h_window_fusion.py (every stretch of every window), h_summary.py (the tables).  Results in
research/anchor235/r37/results/ (untracked); every number this document relies on is written into
the document.

## Pre-registered (written before any computation of this branch)

### Vocabulary fixed for this document

A record stretch of machine M = {5..q} is the blocked run between two consecutive openings
x < y = x + F.  The **record interval** is the closed interval [x, y]; its **ends** are the two
openings x and y; its **interior** is x+1 .. y-1.  The **kill layer** of an interior column k is
the smallest gear of M that strikes k (every interior column has one; the two ends have none).
The **layer decomposition** of the interval is the word of kill layers.  For a gear g,
S_g := { k in [x, y] : kill layer of k is > g } - the **survivors** at layer g, i.e. the openings
of the lower machine {5..g} inside the record interval, always including the two ends.  The
**gap word at layer g** is the sequence of differences of consecutive elements of S_g; it sums to
F.  k_g := |S_g| - 1 is the number of lower gaps the gears above g **fuse** into the record.
maxgap_g is the largest letter of the gap word at layer g.  F_g := F({5..g}) is the lower
machine's own record (2, 5, 7, 11, 18, 25, 34, 43, 58 at g = 5, 7, 11, 13, 17, 19, 23, 29, 31).

Two readings of the human's question, made exact:

- **MIDDLES.**  The record is long because some lower machine already has a long gap there and
  the top gears extend it a little.  Signature: maxgap_g is close to F at some layer g well below
  the top, and maxgap_g is at or near that layer's own record F_g.
- **ENDS.**  The record is long because several *ordinary* lower gaps sit end to end and the top
  gears strike exactly the openings between them, fusing them.  Signature: maxgap_g stays a small
  fraction of F up to the top three or four layers, no maxgap_g reaches F_g, and the top gears
  each remove interior survivors.

### The theory

The record is an ENDS object: it is a fusion of ordinary lower gaps, and the alignment that makes
it is the alignment of the *junctions* between those gaps with the teeth of the top gears - not
the alignment of a long lower middle with anything.  Consequence for the root: what the record
needs is a coincidence in the top gears' phases, and the window is exactly where the top gears'
phases each occur a bounded number of times.

### Predictions, with numbers, and what refutes each

- **E1 (no lower record inside).**  For every record stretch of m13..m31 and every layer g < q,
  maxgap_g < F_g strictly: the record interval never contains a lower machine's own record gap.
  REFUTED by one layer with maxgap_g >= F_g.
- **E2 (the fraction curve).**  frac_g := maxgap_g / F is increasing in g and small at the bottom:
  frac_7 <= 0.25 at m29 and m31, and frac at the second-highest gear (29 at m31, 23 at m29)
  <= 0.75.  REFUTED by frac_7 > 0.25 or frac_{q-} > 0.75.
- **E3 (the top gear fuses at least three).**  At m23, m29 and m31 the top gear removes at least
  two interior survivors, i.e. k_{q-} >= 3.  REFUTED by a record where the top gear removes 0 or
  1.  (Known and not re-derived: 9e already reports 1, 1, 2, 1, 3 top-layer survivors at
  m11..m23, so the prediction is only about m29 and m31 being like m23 and not like m19.)
- **E4 (removals are middles, not ends).**  At the top three layers of m29 and m31, more than half
  of the removed interior survivors sit in the middle three-fifths of the interval, offsets
  j/F in [0.2, 0.8].  REFUTED if half or fewer do.
- **E5 (corridor extremality at layer 7).**  For the m29 and m31 records, the start residue
  x mod 35 minimises |E_35 intersect [r, r+F]| over the 35 rotations r; i.e. the record's layer-7
  interior is a corridor-extremal configuration.  REFUTED if the count is above the minimum;
  a secondary, weaker mark is "in the lowest three of the 35 rotations".
- **E6 (the window is not made the same way).**  At the rungs q = 23..997, the window's longest
  stretch has k_g reaching 1 at a gear far below q, so the top gears do no fusing: gear q strikes
  no column of the window's longest stretch at more than half of the rungs.  REFUTED if gear q
  strikes a column at more than half the rungs.
- **E7 (the candidate exact statement).**  *At every rung q = 23..997, the window's longest
  stretch contains at most one column struck by the top gear q; the period records at m23, m29
  and m31 contain at least two.*  REFUTED by one window stretch with two q-struck columns.
- **E9 (fusion depth in the window; written before part C was run).**  Call the **fusion count**
  of a stretch 1 + the number of interior survivors removed by the largest gear that removes any
  (the number of lower pieces that gear joins).  Deep fusions do occur in a window - any gear
  striking two adjacent survivors gives one - so the prediction is not that they are absent but
  that they are carried by SHORT stretches: at every rung 23..997 the longest stretch carrying a
  fusion of three or more is shorter than F_W, and the window's own longest stretch has fusion
  count 2.  REFUTED if the window's longest stretch has fusion count >= 3 at more than a few
  rungs, or if the longest deeply-fused stretch is the longest stretch.
- **E8 (runner-ups share the mechanism).**  The runner-up stretches at m23..m31 have the same
  shape of decomposition as the records: same sign of the frac curve, top gear removing at least
  one interior survivor at m29 and m31.  REFUTED if the runner-ups are made in a visibly
  different way (e.g. a long middle at a low layer).

### Scorecard

| # | prediction | verdict | evidence |
|---|---|---|---|
| E1 | maxgap_g < F_g at every layer | REFUTED as stated, CONFIRMED restricted | gears 5 and 7 always reach their own record inside; 13 and 17 do inside the m31 record, 17 inside the m19 record. But at the top three depths of m23, m29, m31 the ratio is 0.600, 0.676, 0.581-0.698 and never 1 (results 2) |
| E2 | frac_7 <= 0.25 and frac_{q-} <= 0.75 at m29, m31 | CONFIRMED | frac_7 = 0.116 (m29), 0.086 (m31); frac at the second gear 0.535 (m29), 0.431-0.517 (m31) |
| E3 | top gear removes >= 2 interior survivors at m23, m29, m31 | CONFIRMED | 3, 2, 2 removals, so fusions of 4, 3, 3 pieces |
| E4 | > half of top-three-layer removals in the middle three-fifths | CONFIRMED | 48 of 48 decomposed stretches |
| E5 | corridor-extremal layer-7 interior at m29, m31 | REFUTED, and inconsistent in direction | 0 of 44 non-degenerate stretches at the minimum; m31 near-minimal (3 of 35 below), m29 rich (24 of 35 below) |
| E6 | gear q strikes nothing in the window stretch at > half the rungs | REFUTED in the letter, CONFIRMED in the mechanism | q strikes a column at 98 of 160 rungs, but REMOVES a survivor at 0 of 160 |
| E7 | window stretch has <= 1 top-gear column at every rung | REFUTED | two q-struck columns at 8 rungs (23, 29, 31, 41, 59, 83, 113, 179); replaced by W1-W4 |
| E8 | runner-ups share the shape | CONFIRMED with one exception | runner-ups fuse 3 pieces at m23 and m29 and 2-4 at m31; three m31 runner-up classes and none of the records are two-piece fusions, and one m29 runner-up class uses a same-tooth strike distance q rather than a letter |
| E9 | deep fusions in the window sit on short stretches | CONFIRMED and sharpened | fusion >= 3 at all 160 rungs but never on the longest stretch (median 0.34 F_W); fusion >= 4 at 0 of 160 rungs, 0 of 1.3 million stretches |

### What this branch could find that is not already known

Known and not to be re-derived: the merge law and the record law (docs/proofs/05, 09) - that a
gap of the bigger machine is a fusion of lower gaps whose interior openings the new gear strikes
is a theorem, not a finding; the hit law and chain law and the survivor COUNTS at m11..m23
(anchor-235 section 9d); the two-teeth kill-spacing law; the gear-5 lock (5g); the corridor law
(docs/proofs/14); the count of frame columns in a window (5d.i).  What is not on the record is
the decomposition of the EXACT records at m29 and m31 - positions, gap words, which gear removes
which survivor and on which tooth - and the ends-versus-middles measurement on them, plus the
same measurement on the window's longest stretch.  The question is not *that* a record is a
merge; it is *what the merged pieces are*, and whether the pieces are ordinary or extremal.

## Setup (exact ranges)

Full-period residue sieves, no sampling anywhere.

| machine | gears | period P | F | runner-up | record stretches | runner-up stretches |
|---|---|---|---|---|---|---|
| {5..7} | 5, 7 | 35 | 5 | 3 | 2 | - |
| {5..11} | .. 11 | 385 | 7 | 6 | 4 | - |
| {5..13} | .. 13 | 5,005 | 11 | 10 | 12 | - |
| {5..17} | .. 17 | 85,085 | 18 | 16 | 20 | - |
| {5..19} | .. 19 | 1,616,615 | 25 | 23 | 20 | - |
| {5..23} | .. 23 | 37,182,145 | 34 | 33 | 4 | 2 |
| {5..29} | .. 29 | 1,078,282,205 | 43 | 40 | 2 | 8 |
| {5..31} | .. 31 | 33,426,748,355 | 58 | 55 | 4 | 34 |

The m29 and m31 scans are fresh (3.1 s and 155.8 s, chunked); both reproduce the corpus F and the
record starts of research/anchor235/r35.  The gap spectra near the top confirm the isolation
already on record (no 41 or 42 below 43; no 56 or 57 below 58).  Six stretches per set are
decomposed in full; the summary tables cover all 48 decomposed stretches.

The window part runs at every prime rung q = 23..997 (160 rungs) over the columns
lo = q//6 + 1 .. hi = (q'^2-1)//6, where an opening of {5..q} is a twin pair.  Part B decomposes
the window's longest stretch; part C computes the fusion depth of **every** blocked stretch of
every window (1.3 million stretches in all).

Scripts: h_ends_middles.py (part A), h_window_decomp.py (part B), h_window_fusion.py (part C),
h_summary.py (the tables).  Logs h_A.log, h_B.log, h_C.log, h_S.log in the results directory.

## Results

### 1. The layer decomposition of the exact records

The record interval read as a nest of lower-machine gap words.  k_g = pieces at layer g,
mx = the largest piece, F_g = the lower machine's own record.

**m29, F = 43, x = 200906185** (the second record is its mirror at x = 877375977):

| layer g | k_g | gap word | mx | F_g | mx/F_g | mx/F | removed by g (offset : residue, tooth) |
|---|---|---|---|---|---|---|---|
| 5 | 26 | 2 1 2 2 1 2 2 1 2 2 1 2 2 1 2 2 1 2 2 1 2 2 1 2 2 1 | 2 | 2 | 1.000 | 0.047 | 17 columns, alternating teeth 1+ / 4- |
| 7 | 19 | 3 2 2 1 2 2 1 2 2 3 2 5 1 5 2 3 2 2 1 | 5 | 5 | 1.000 | 0.116 | 2:6+, 18:1-, 23:6+, 25:1-, 30:6+, 32:1-, 37:6+ (two chains of 2) |
| 11 | 15 | 3 2 2 1 2 2 3 5 2 5 6 5 2 2 1 | 6 | 7 | 0.857 | 0.140 | 13:9-, 17:2+, 28:2+, 35:9- |
| 13 | 11 | 5 2 1 2 5 5 2 5 6 7 3 | 7 | 11 | 0.636 | 0.163 | 3:2-, 12:11+, 38:11+, 42:2- |
| 17 | 8 | 7 1 2 5 5 7 13 3 | 13 | 18 | 0.722 | 0.302 | 5:3+, 22:3+, 33:14- |
| 19 | 5 | 7 3 5 5 23 | 23 | 25 | 0.920 | 0.535 | 8:3-, 27:3-, 40:16+ (one chain of 2) |
| 23 | 3 | 10 10 23 | 23 | 34 | 0.676 | 0.535 | 7:19-, 15:4+ |
| 29 | 1 | 43 | 43 | 43 | 1.000 | 1.000 | 10:24-, 20:5+ (one chain of 2) |

**m31, F = 58, x = 1468940242** (four record stretches, two mirror pairs; the second class at
x = 21844264615 is the reverse word through layer 23 and differs only at layer 29, where its word
is 18 10 30 rather than the reverse 25 10 23 -- the two classes hand different junctions to gears
29 and 31):

| layer g | k_g | gap word | mx | F_g | mx/F_g | mx/F | removed by g |
|---|---|---|---|---|---|---|---|
| 5 | 35 | 1 2 2 1 2 2 ... (the 5-word) | 2 | 2 | 1.000 | 0.034 | 23 columns |
| 7 | 23 | 3 2 5 1 5 2 3 2 2 1 2 2 1 2 2 3 2 5 1 5 2 3 2 | 5 | 5 | 1.000 | 0.086 | 12 columns (four chains of 2) |
| 11 | 19 | 3 2 5 1 5 2 3 2 2 1 2 2 3 5 2 5 6 5 2 | 6 | 7 | 0.857 | 0.103 | 31:9-, 35:2+, 46:2+, 53:9- |
| 13 | 15 | 3 2 5 1 5 2 5 3 2 2 3 7 5 11 2 | 11 | 11 | 1.000 | 0.190 | 21:11+, 25:2-, 38:2-, 51:2- |
| 17 | 10 | 3 7 6 2 5 3 4 3 7 18 | 18 | 18 | 1.000 | 0.310 | 5:14-, 11:3+, 28:3+, 45:3+, 56:14- |
| 19 | 8 | 3 15 5 3 4 3 7 18 | 18 | 25 | 0.720 | 0.310 | 10:16+, 16:3- (one chain of 2) |
| 23 | 5 | 23 7 3 7 18 | 23 | 34 | 0.676 | 0.397 | 3:4+, 18:19-, 26:4+ (one chain of 2) |
| 29 | 3 | 23 10 25 | 25 | 43 | 0.581 | 0.431 | 30:24-, 40:5+ |
| 31 | 1 | 58 | 58 | 58 | 1.000 | 1.000 | 23:26+, 33:5- (one chain of 2) |

**m23, F = 34, x = 12694428**: layer words 
`5: [2 2 1 ...]`, `7: [2 2 1 2 2 3 2 5 1 5 2 3 2 2]`, `11: [2 2 1 2 2 3 7 1 7 3 4]`,
`13: [2 2 3 5 7 1 7 3 4]`, `17: [4 3 5 8 7 7]`, `19: [4 8 15 7]`, `23: [34]`;
gear 23 removes three survivors at offsets 4 (19-), 12 (4+) and 27 (19-) - one chain of three,
teeth alternating.

Survivors removed per layer, first record of each machine (the same information as the counts in
anchor-235 section 9d, now with positions and teeth; the counts agree):

| machine | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|---|
| m17 | 7 | 5 | 2 | 1 | 2 | | | | |
| m19 | 10 | 6 | 3 | 2 | 2 | 1 | | | |
| m23 | 14 | 6 | 3 | 2 | 3 | 2 | 3 | | |
| m29 | 17 | 7 | 4 | 4 | 3 | 3 | 2 | 2 | |
| m31 | 23 | 12 | 4 | 4 | 5 | 2 | 3 | 2 | 2 |

### 2. Ends versus middles

**The largest piece as a fraction of the record (frac_g = mx/F), first record of each machine:**

| machine | F | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|---|---|
| m17 | 18 | 0.111 | 0.278 | 0.333 | 0.611 | 1.000 | | | | |
| m19 | 25 | 0.080 | 0.200 | 0.240 | 0.440 | 0.720 | 1.000 | | | |
| m23 | 34 | 0.059 | 0.147 | 0.206 | 0.206 | 0.235 | 0.441 | 1.000 | | |
| m29 | 43 | 0.047 | 0.116 | 0.140 | 0.163 | 0.302 | 0.535 | 0.535 | 1.000 | |
| m31 | 58 | 0.034 | 0.086 | 0.103 | 0.190 | 0.310 | 0.310 | 0.397 | 0.431 | 1.000 |

The curve is flat and low through {5..17} and then climbs in three steps.  At m29 and m31 no
single gap of any machine up to {5..17} is more than 31% of the record; the whole length is made
by the last three gears.

**Does the record contain a lower machine's own record?**  mx/F_g by depth below the top gear,
over every record and runner-up stretch (1.000 = a lower record sits inside):

| machine | kind | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|
| m17 | record | 13: 0.636-1.000 | 11: 0.857 | 7: 1.000 | 5: 1.000 | |
| m19 | record | 17: 0.722-1.000 | 13: 0.636-1.000 | 11: 0.857 | 7: 1.000 | 5: 1.000 |
| m23 | record | 19: 0.600 | 17: 0.444-0.833 | 13: 0.636-0.727 | 11: 1.000 | 7: 1.000 |
| m23 | runner | 19: 0.800 | 17: 0.444 | 13: 0.636 | 11: 0.857 | 7: 1.000 |
| m29 | record | 23: 0.676 | 19: 0.920 | 17: 0.722 | 13: 0.636 | 11: 0.857 |
| m29 | runner | 23: 0.588-0.853 | 19: 0.600-0.800 | 17: 0.389-0.722 | 13: 0.636 | 11: 0.857-1.000 |
| m31 | record | 29: 0.581-0.698 | 23: 0.676 | 19: 0.720 | 17: 1.000 | 13: 1.000 |
| m31 | runner | 29: 0.512-0.744 | 23: 0.500-0.618 | 19: 0.480-0.720 | 17: 0.667-1.000 | 13: 0.727-1.000 |

At the three machines whose record set has collapsed (m23, m29, m31) no lower record sits inside
at any of the top three depths; the closest approach is 0.920 (the m29 record contains a 23-gap of
{5..19}, whose record is 25).  Lower records do sit inside at the bottom -- gears 5 and 7 always,
gears 13 and 17 inside the m31 record -- but those are short gaps that occur everywhere.

**The top gear's own fusion.**  F = left flank + interior pieces + right flank, the interior
pieces being the distances between the top gear's strikes on the survivors:

| machine | kind | F | pieces fused | interior pieces | letters {a, b} of q | flanks | flank sum |
|---|---|---|---|---|---|---|---|
| m17 | record | 18 | 3 | 11 or 6 | {6, 11} | (5, 2) or (5, 7) or (7, 5) | 7 or 12 |
| m19 | record | 25 | 2 or 3 | none, or 13 | {6, 13} | (7, 18), (18, 7), (7, 5), (5, 7) | 25 or 12 |
| m23 | record | 34 | 4 | 8, 15 (or 15, 8) | {8, 15} | (4, 7) or (7, 4) | 11 |
| m23 | runner | 33 | 3 | 8 | {8, 15} | (5, 20) or (20, 5) | 25 |
| m29 | record | 43 | 3 | 10 | {10, 19} | (10, 23) or (23, 10) | 33 |
| m29 | runner | 40 | 3 | 10, or 29 | {10, 19} | (10, 20), (20, 10), (8, 3), (3, 8) | 30 or 11 |
| m31 | record | 58 | 3 | 10 | {10, 21} | (23, 25), (30, 18), (18, 30), (25, 23) | 48 |
| m31 | runner | 55 | 2, 3 or 4 | none, 10, or 21, 10 | {10, 21} | (25, 30), (30, 15), (13, 32), (2, 22) | 55, 45 or 24 |

Every interior piece of every record is one of the top gear's two letters, and where a gear takes
two junctions its two strikes are on opposite teeth.  No record uses a full period q as a strike
distance; one runner-up class at m29 does (its interior piece is 29).

**The ends.**  The column immediately outside the record -- the one that would have to be struck
for the run to reach further -- is struck by gear 5 at both ends at every record of m13, m17, m19,
m29 and m31, and at every runner-up class of m23, m29 and m31.  At the four m23 records it is
struck by 11, 13 or 23 instead, and in two of them it is not struck at all, so the flanking gap
there is 1.  The ends of a record are made at the bottom of the machine; the interior joints are
made at the top.

**Where the removals sit.**  Over all 48 decomposed stretches, more than half of the removals made
by the top three gears fall in the middle three-fifths of the interval: 48 of 48.  The largest
piece is an interior letter of the gap word at the bottom layers and an end letter at the top: at
the m29 record the argmax position moves 1/26, 12/19, 11/15, 10/11, 7/8, 5/5, 3/3, 1/1 as the
layer rises.

### 3. The corridor inside

The layer-7 interior of a stretch is fixed by x mod 35 and the length.  Counting the exposed
columns |E_35 cap [x, x+F]| against the 35 rotations of the same length:

| machine | kind | F | count | min | max | rotations strictly below |
|---|---|---|---|---|---|---|
| m13 | record | 11 | 4 or 5 | 3 | 7 | 2 or 15 |
| m17 | record | 18 | 7 | 6 | 10 | 6 |
| m19 | record | 25 | 10 | 9 | 13 | 2 |
| m23 | record | 34 | 15 | 15 | 15 | degenerate (F+1 = 35) |
| m23 | runner | 33 | 15 | 14 | 15 | 15 |
| m29 | record | 43 | 20 | 17 | 21 | 24 |
| m29 | runner | 40 | 18 or 19 | 16 | 19 | 19 or 29 |
| m31 | record | 58 | 24 | 23 | 27 | 3 |
| m31 | runner | 55 | 23 | 22 | 26 | 6 |

**No record is corridor-extremal**: 0 of 44 non-degenerate stretches sit at the minimum, and the
direction is not even consistent.  m13, m19 and m31 sit at the second-lowest attainable count
(2, 2 and 3 of the 35 rotations strictly below), m17 in the lower fifth, while m29 sits high --
24 of 35 rotations carry fewer exposed columns than the record's phase does.  The m23 record is
exactly 35 columns long, so gears 5 and 7 leave 15 survivors whatever the phase and the corridor
has no say there at all.  The record does not economise at the corridor: it takes an ordinary or
survivor-rich phase mod 35 and pays for the extra survivors with the higher gears.  The one
corridor-level regularity that does hold is gear 5's, and that is the proved gear-5 lock of node
5g, not re-derived here.

### 4. The window at rungs 23..997

The window's longest stretch, decomposed the same way.  g* is the largest gear that removes an
interior survivor (the gear that finishes the stretch); nq the columns of the stretch struck by
the top gear q; npart the gears that remove at least one survivor.

| q | F_W | F_W/q | g* | g*/q | nq | npart | ngears | pieces below g* | mx/F_W there |
|---|---|---|---|---|---|---|---|---|---|
| 23 | 25 | 1.087 | 19 | 0.826 | 3 | 6 | 7 | 2 (7, 18) | 0.720 |
| 47 | 28 | 0.596 | 43 | 0.915 | 1 | 11 | 13 | 2 (26, 2) | 0.929 |
| 113 | 47 | 0.416 | 107 | 0.947 | 2 | 15 | 28 | 2 (32, 15) | 0.681 |
| 157 | 83 | 0.529 | 151 | 0.962 | 1 | 20 | 35 | 2 (32, 51) | 0.614 |
| 257 | 105 | 0.409 | 179 | 0.696 | 0 | 23 | 53 | 2 (29, 76) | 0.724 |
| 433 | 154 | 0.356 | 311 | 0.718 | 0 | 27 | 82 | 2 (145, 9) | 0.942 |
| 829 | 168 | 0.203 | 701 | 0.846 | 0 | 35 | 143 | 2 (105, 63) | 0.625 |
| 997 | 242 | 0.243 | 877 | 0.880 | 1 | 34 | 166 | 2 (203, 39) | 0.839 |

Over all 160 rungs the top gear q **never** removes an interior survivor (g* = q at 0 of 160);
g*/q has median 0.618, F_W/q median 0.292, npart/ngears median 0.299.  The top gear does strike a
column of the stretch at 98 of 160 rungs, and two columns at 8 rungs (23, 29, 31, 41, 59, 83, 113,
179) -- but always a column some smaller gear had already taken.

**Fusion depth over every stretch of every window.**  The fusion count of a stretch is one more
than the number of survivors its closing gear removes, i.e. the number of lower pieces that gear
joins.  Over the 1.3 million stretches of the 160 windows:

- the window's **longest** stretch has fusion count exactly 2 at **160 of 160** rungs;
- a fusion of 3 occurs somewhere in every window (3 to 132 stretches per window), but the longest
  stretch carrying one is 0.200 to 0.920 of F_W, median 0.338;
- a fusion of **4 or more occurs nowhere**: 0 of 160 windows, 0 of 1.3 million stretches;
- the deepest fusion of a window never sits on its longest stretch: 0 of 160 rungs.

**How the two are assembled.**  Counting the gears that take the piece count from 8 down to 1:

| | machine or rung | gears used | their removals | pieces they fuse (Y) |
|---|---|---|---|---|
| period record | m23 | 17, 19, 23 | 3, 2, 3 | 9 |
| period record | m29 | 19, 23, 29 | 3, 2, 2 | 8 |
| period record | m31 | 23, 29, 31 | 3, 2, 2 | 8 |
| window stretch | q = 113 | 79, 89, 107 | 1, 1, 1 | 4 |
| window stretch | q = 433 | 293, 307, 311 | 1, 1, 1 | 4 |
| window stretch | q = 997 | 853, 859, 877 | 1, 1, 1 | 4 |

Over all 160 rungs the three top fusing gears of the window's longest stretch join Y = 4, 5 or 6
pieces (4 at most rungs) against 8, 8, 9 for the period records; the number of gears spanned in
taking the piece count from 8 to 1 rises from 4 at q = 23 to 86 at q = 997, of which only 8
actually fuse; and the largest gear that removes two or more survivors from the window's longest
stretch sits at 0.047 to 0.872 of q, median 0.216, above q/2 at only 19 of 160 rungs.

## Mechanism

A record interval is a nest.  At the bottom, gears 5 and 7 cut it into 19 to 35 tiny pieces of
sizes 1, 2, 3 and 5; that part is ordinary, and both bottom records (gear 5's 2, gear 7's 5) occur
inside it several times over.  Nothing there is extreme: at m29 and m31 no piece of any machine up
to {5..17} exceeds 31% of the record.  What makes the record is what the last three gears do.

At m29 the layer-17 word is 7 1 2 5 5 7 13 3 -- eight ordinary pieces with seven junctions between
them.  Gear 19 strikes three of those junctions, gear 23 two, gear 29 two: seven junctions, three
gears, every junction closed.  The same shape at m31 (layer-19 word 3 15 5 3 4 3 7 18, seven
junctions, gears 23, 29, 31 taking 3, 2, 2) and at m23 (layer-13 word 2 2 3 5 7 1 7 3 4, eight
junctions, gears 17, 19, 23 taking 3, 2, 3).  The junctions each gear takes sit at the distances
its two teeth allow, 0 or +-d_g modulo g: at the TOP gear of every record the strikes are on
opposite teeth at exactly a letter's distance -- m29's two 29-strikes are 10 apart, 10 = 2u_29;
m23's three 23-strikes are 8 and 15 apart, the two letters alternating; m31's two 31-strikes are
10 apart, 10 = 31 - 2u_31 -- while a middle gear may use the same tooth a full period apart
instead (m29's gear 19 takes offsets 8 and 27, both on tooth 3, 19 apart, then 40 at 13 = d_19).
Where two
junctions taken by one gear were adjacent survivors it is the chain law's double hop that does it
(one such chain at m29, one at m31, one of length three at m23).  That the spacings are the
letters, and that adjacency needs the chain classes, are the kill-spacing and chain laws, already
kernel-checked; what is new is that they are the entire top of the record and that the pieces they
join are ordinary.

So the record's length reads

    F = left flank + (interior pieces, each a letter of the top gear) + right flank,

the interior sum being decided by the gear alone: m23's 34 = 4 + (8 + 15) + 7 with 8 + 15 = 23 = q
exactly; m29's 43 = 10 + 10 + 23 with the middle piece the letter 10; m31's 58 = 23 + 10 + 25 with
the middle piece the letter 10.  Only the two flanks are free, and they are ordinary gaps of the
lower machine: 4 and 7 against F(19) = 25; 10 and 23 against F(23) = 34; 23 and 25 against
F(29) = 43.

The window's longest stretch is assembled the opposite way.  It is a chain of two-piece fusions,
one gear at a time, spread over dozens of gears -- 34 of the 166 gears remove anything at q = 997,
and above the eighth-from-last of them every fusing gear removes exactly one survivor -- and its
own top gear does nothing: at every one of
the 160 rungs the largest gear that removes a survivor is strictly below q, at a median of 0.62 q.
Where a record's top three gears close seven or eight junctions, a window stretch's top three
close three to five.  A four-piece fusion, the m23 record's shape, occurs nowhere in any window at
any of the 160 rungs, and a three-piece fusion occurs only on stretches at a median of a third of
the window's own longest.

## The answer

**It is the ends.**  The longest open stretch is not a long middle of any meta sieve dressed up;
it is a row of ordinary lower gaps whose junctions the top three gears strike.  At m29 the record
43 is 10 + 10 + 23 as gaps of {5..23}, none of them near that machine's own record 34, and one
layer down it is the eight-piece word 7 1 2 5 5 7 13 3 of {5..17} whose largest piece is 13, 30%
of the record; the seven junctions of that word are closed by gears 19, 23 and 29 taking three,
two and two of them, each gear's strikes separated by one of the distances its two teeth allow and
gear 29's two by exactly its letter 10, on opposite teeth.  At m31 the record 58 is 23 + 10 + 25
as gaps of {5..29} against that
machine's record 43, and below it the eight-piece word 3 15 5 3 4 3 7 18 of {5..19} whose seven
junctions gears 23, 29 and 31 close as 3 + 2 + 2, gear 31's two strikes 10 apart on opposite teeth.
Through {5..17} the record is an ordinary stretch: the largest piece is 31% of it at m31 and 30%
at m29, its layer-5 and layer-7 sub-records occur inside it repeatedly, and its phase mod 35 is not
even corridor-economical (0 of 44 stretches sit at the corridor minimum; m29's phase carries more
exposed columns than 24 of the 35 rotations).  The middles contribute nothing extremal; the ends
of ordinary pieces, aligned with the top gears' teeth, contribute everything.

**Against the root.**  Read this way the budget inequality is a statement about the flanks alone.
If the top gear fuses k pieces, the k - 2 interior pieces alternate strictly between the two
letters a and b (the merge law's grammar), so their sum is fixed by the gear: nothing at k = 2, one
letter at k = 3, exactly q' at k = 4, q' plus a letter at k = 5.  Then F(M) <= F(M^-) + q' is
exactly

    (left flank) + (right flank)  <=  F(M^-) + q' - (interior sum),

which is F(M^-) + q' at k = 2 -- that is the pair statement, node 1 -- then F(M^-) + (the other
letter) at k = 3, F(M^-) at k = 4, and F(M^-) - (a letter) at k = 5.  Measured: 33 <= 34 + 19 at
m29 (slack 20), 48 <= 43 + 21 at m31 (slack 16), 11 <= 25 at m23 (slack 14), 25 <= 18 + 19 at m19
(slack 12).  Each extra pair of letters costs a whole q' of interior while the budget grants only
q', so a deeper fusion demands strictly shorter flanks; that is why the fusion depth does not grow
with the machine, and it puts the surviving question exactly on node 2g's three-gap repulsion --
with the middle gap now known to be a letter of the new gear rather than free.  This is a
reformulation of two known laws, not a new theorem, and is recorded as such.

**What the window does to the mechanism.**  Everything the record needs from its top three gears,
the window denies -- not by forbidding any one gear's strike but by never assembling the joint
configuration.  Measured at 160 rungs over 1.3 million stretches:

- **(W1)** the top gear q never removes a survivor from the window's longest stretch (0 of 160),
  whereas in a period record it necessarily does, since F(M) > F(M^-);
- **(W2)** the window's longest stretch is a two-piece fusion at every rung, while the records at
  m23, m29 and m31 are four-, three- and three-piece fusions;
- **(W3)** a fusion of four or more pieces by one gear occurs nowhere in any window at any of the
  160 rungs, and a fusion of three occurs only on stretches of median 0.34 F_W (max 0.92 F_W,
  never on the longest);
- **(W4)** the three top fusing gears of a window stretch join 4 to 6 pieces; of a record, 8 or 9.

So the statement asked for is true as measured, in the form *a fusion of four ordinary lower pieces
by one gear never occurs inside a window*, exact at 160 rungs.  What it does **not** do is bound
the length: a two-piece fusion of two long pieces is long, and that is exactly how the window's
longest stretches are built (203 + 39 at q = 997).  The counting version of the same observation --
the top three gears' joint phase has period about q^3 against a window of q^2/6 columns, so a
record-shaped configuration is expected about once in 6q windows -- is node 5d.i's frame count one
level up and inherits its verdict: it says where a record can sit, not that no long stretch sits in
the window.  FACT, not a route.

## What is new

Not new, met on the way and not re-derived: the merge law and the record law (a gap of the bigger
machine is a fusion of lower gaps whose interior openings the new gear strikes); the two-teeth
kill-spacing law (strike distances are the two letters, alternating); the hit and chain laws and
the survivor counts at m11..m23 (anchor-235 section 9d); the gear-5 lock; the corridor law; the
frame count in a window (5d.i).

New here: (i) the exact layer decompositions of the m29 and m31 records -- positions, gap words,
teeth, chains -- which 9d stops short of, and with them the fact that the top three gears close
every junction of an eight- or nine-piece word; (ii) that no lower machine's own record sits inside
a record at any of the top three depths at m23, m29 and m31, the closest approach being 0.920;
(iii) the ends-versus-middles reading F = flank + letters + flank and the flank form of the budget
inequality that follows from it; (iv) that a record is never corridor-extremal, and that the
direction is not even consistent (m29 corridor-rich, m31 corridor-poor); (v) the window contrast
W1-W4, in particular that no gear anywhere in any window at 160 rungs fuses four pieces while the
m23 record does, and that the window's longest stretch is a two-piece fusion at every rung.

## Verdict

**FACT (exact, kept, not a route).**  The record is an ENDS object: ordinary lower pieces joined at
their ends by the top three gears, the joins landing at the gears' own letters, the two flanks the
only free lengths.  The window's longest stretch is built the opposite way, one two-piece fusion at
a time over dozens of gears, and the record's assembly pattern -- a four-piece fusion, or three
gears closing eight junctions -- occurs in no window at any rung to 997.  This answers the human's
question exactly and localises the remaining freedom in the two flanks, which is node 2g.  It does
not bound F_W, because a two-piece fusion of two long pieces is long.

## Dead ends

- **E1 as pre-registered is refuted.**  A record does contain lower records: gear 5's 2 and gear
  7's 5 always, gear 13's 11 and gear 17's 18 inside the m31 record, gear 17's 18 inside the m19
  record.  Only the restricted form -- top three depths, at m23, m29 and m31 -- survives.
- **E5 refuted.**  The record is not a corridor-extremal configuration: 0 of 44 non-degenerate
  stretches sit at the minimum, and m29 sits above 24 of the 35 rotations.  Corridor economy is
  not what a record does.  At m23 the record is exactly 35 columns long, so the corridor cannot
  discriminate there at all.
- **E6 and E7 as pre-registered are refuted in the letter.**  The top gear does strike a column of
  the window's longest stretch at 98 of 160 rungs, and two columns at 8 rungs.  The right quantity
  is not the strike but the removal; on that the statement holds at every rung.
- **The corridor rank as a locator.**  The ranks disagree in direction between m29 and m31, so no
  rule of the shape "records sit at a distinguished phase mod 35" can exist.  This is the
  escape-distance-1 ceiling met again and is not re-entered.

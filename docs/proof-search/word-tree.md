# The word tree inside the window

Manager, round 29 (2026-09-02). Human's construction. Pre-registration
research/data/r29/word_tree_prereg.md (T1-T5, scorecard appended), script
research/word_tree_r29.py, log research/data/r29/word_tree.log (667 machines, q <= 4999,
window to slot 4,171,667; exact integers, three gates green after two pre-registered
statements were refuted and one was mis-normalised - see the scorecard).

## 1. The construction

Root: the machine's word restricted to the reduction window (q/6, W], W = (q_next^2 - 2)/6
(the largest k with 6k+1 < q_next^2). Every blocked slot in the window is labelled by its
DEATH RUNG r(k) = the smallest prime p >= 5 dividing 6k-1 or 6k+1; inside the window every
composite 6k+-1 has such a factor <= q, so the label is always a gear of the machine. A
blocked run [a, b] at level p splits into the runs of the sub-machine m_p inside [a, b]
(its n-tuple); those split at the next gear down; the leaves are single kills labelled by
their gear. Openings inside the window are exactly the twin prime pairs.

Two facts are forced before any computation and shape everything below:

  (i) the window word is the twin-prime indicator, so a node "allows twins" iff it is an
      opening, and a node "kills" iff it is a composite with a small factor - there is no
      third kind of node, and no node kills beyond its own length;
  (ii) adding a gear never changes a letter inside the old window (the new gear's first
      in-window kill is at 6k+-1 = q'^2, i.e. the old window's end). So across machines the
      tree is ONE fixed infinite tree seen through a growing window; gear p's branch is its
      comb of kills at the odd multiples 5p, 7p, 11p, ... of p that are not already dead.

## 2. What the trees look like (measured)

Per-machine table (from the log; G_W = largest gap between consecutive openings inside the
window, #seal = number of distinct death rungs inside the maximal run, last = the largest):

    q      W       #open   G_W   G_W/W    F(M)  #seal  last  last/q  G_W/ln^2(6W)
    13     47      16      5     0.106    11    2      7     0.54    0.16
    19     87      21      12    0.138    25    6      19    1.00    0.31
    29     159     31      25    0.157    43    6      19    0.66    0.53
    53     579     87      28    0.048    145   11     43    0.81    0.42
    97     1699    202     35    0.021    -     12     59    0.61    0.41
    199    7419    626     83    0.011    -     20     151   0.76    0.72
    499    42167   2585    154   0.004    -     27     311   0.62    0.99
    997    169679  8278    242   0.001    -     34     877   0.88    1.26
    1999   668667  26870   252   0.0004   -     40     1637  0.82    1.09
    4999   4171667 130543  365   0.0001   -     54     3733  0.75    1.26

The fusion tree of the maximal run at m4999 (364 blocked slots, 54 sealing gears), read
top-down, is in the log. Its shape is the same at every machine drawn (53, 199, 997, 4999):

  BOTTOM (gears 5, 7). Gear 5 kills 40% of the run (146 of 364 at m4999), every kill
  isolated (the teeth +-1 mod 5 are never adjacent): the level-5 word is a 1-tuple comb.
  Gear 7 kills 2/7 of what is left (61 of 218). After gears 5 and 7 the sub-runs are still
  tiny (lengths 1, 2, 4 only): the two smallest gears make the fine grain of every word and
  never a run.

  MIDDLE (gears 11 to about q/10). Each gear kills several slots (11: 29, 13: 20, 17: 14,
  ..., 97: 1 at m4999), Mertens-weighted, and each kill fuses two neighbouring sub-runs. The
  tuple lengths grow from 1-4 to 10-40 across these levels.

  TOP (gears above about q/10). EVERY gear kills EXACTLY ONE slot in the run (36 of the 37
  gears >= 97 at m4999; 22 of 22 gears >= 137 at m997; 9 of 9 gears >= 53 at m199). Each
  such kill is a binary fusion of two adjacent sub-runs, so the top of the tree is a chain of
  ~30 binary merges, one gear each, and the run is sealed only when the last of them lands.

Why the top is single-kill: a slot with death rung p > q/10 has both 6k-1 and 6k+1 free of
factors below p, so it is a near-twin (a prime beside a product of two primes >= p, cofactor
< 6W/p < 60 q). Such slots have density ~ (twin density) x (ln q / ln p)^2 - (twin density),
a few per mille of slots at q ~ 5000 - so a run of length ~400 holds about two of them, and
each big gear can own at most one. That is also why T4 was refuted: the maximal run contains
a slot with death rung > q/2 at 623 of 667 machines (93%). The big gears are needed, but
each contributes one letter; the run's LENGTH is made by the small gears.

## 3. What grows and what does not

  G_W is log-scale, the window is q^2-scale. G_W/W: 0.138 (m19) -> 0.048 (m53) -> 0.011
  (m199) -> 0.0001 (m4999); G_W / ln^2(6W) sits at 0.4-1.3. The in-window maximal gap is
  the maximal twin-prime gap below q_next^2, nothing else.

  The record never lives in the window (T2', q >= 11): G_W = 28 against F = 145 at m53, and
  by m4999 G_W = 365 against a window of 4.2 million slots and a record that, if F/W stays
  at 1/4 (killer-spec.md section 3.1), is around a million. The record gap and the
  in-window gap are different objects: the record is a period-scale phenomenon (the machine
  as a covering system), the window gap is a log-scale phenomenon (the integers' small
  factors).

  Branch weights are Mertens exactly (T3'): the fraction of window slots whose death rung is
  <= 7 is 4/7 and <= 13 is 0.7033, to within 0.01 at every q >= 100 (my pre-registered
  statement normalised by blocked slots instead of all slots and was refuted at 259
  machines; the corrected form holds at all 667).

  Depth grows (T5 refuted): the number of sealing gears of the maximal run is 11, 20, 34, 54
  at q = 53, 199, 997, 4999, roughly (run length) x (the near-twin density above plus the
  Mertens tail). There is no bounded-depth structure; depth is a count of distinct small
  factors in a stretch of ~ln^2 integers.

## 4. Killers and allowers, answered

Inside the window there are no killer NODES beyond ln^2 scale: every run is sealed within
G_W slots, every gear above q/10 adds one letter to the run it touches, and the question
"which branch kills twins for ever" has the answer "none can, inside the window, unless
G_W reaches W" - which is the condition K1 of killer-spec.md restated. The tree makes
visible WHY K1 is hard to reach: a window-sized run would need every one of its ~W slots
sealed, the small gears seal a Mertens fraction ~ 1 - c/ln^2 q of them, and the remaining
c W/ln^2 q near-twin slots would each need their own big gear (one kill per gear at the
top), against ~q/ln q big gears each of which kills ~q/ln^2 q slots spread over the whole
window and not concentrated in one run. The counting is the usual sieve heuristic and not
a proof; what the tree adds is the shape of the obstruction - single-kill top, comb bottom -
and the certainty (fact (ii)) that no future rung can rewrite it.

## 5. What this changes

  (a) The in-window tree is fully determined by the integers' factorisation and is
      invariant under growth; its statistics are Mertens. Nothing in it needs Opus time.
  (b) The proof-relevant object is the RECORD gap outside the window (period scale), where
      the machine is a covering system and the record law / (D) / F/W live. The tree says
      the window gap will never be the record for any machine with q >= 11.
  (c) The single-kill-at-the-top pattern is the in-window face of the uniform-order theorem
      (A_kill <= 5 dead interior openings per rung): here it is <= 2 for every gear above
      q/20 in the maximal run, at every machine drawn.

## 6. The path to each twin (follow-up, same day)

Human's follow-up: look at the whole path from the root to each twin, not only the root
and the leaves. Pre-registration research/data/r29/twin_path_prereg.md (U1-U5), script
research/twin_path_r29.py, log research/data/r29/twin_path.log; 19 of 20 gates green, the
one failure (U4 at m53) missed its pre-registered band by 0.003 on 85 twins.

### 6.1 What the path is

A twin is an opening at every level, so it is never inside a run: at level p it is the
boundary between two runs of the sub-machine m_p, and its path is the pair of flank chains
L_p(k), R_p(k) (distance to the nearest opening of m_p on each side), non-decreasing in p.
Walking away from the twin, the death rungs of the flank slots form a sequence, and L_p is
the position of the first slot whose rung exceeds p. So the LEFT PATH IS THE PREFIX-MAXIMUM
STRUCTURE of that sequence: the gears that touch the flank are its running records, and the
number of old openings a gear fuses is the number of slots equal to it before the next
record. This is exact and needs no computation; what follows is what it looks like in bulk
(85 / 624 / 8,276 / 130,541 twins with both flanks inside the window at q = 53 / 199 / 997
/ 4999).

### 6.2 Measured

    q      mean flank  events/twin  iid-records  max events  arity max (5 / 7 / >=11)  flank max gear > q/2
    53     6.67        2.02         1.99         7           1 / 2 / 2                 0.247
    199    11.8        2.61         2.50         7           1 / 2 / 2                 0.199
    997    20.5        3.15         2.98         11          1 / 2 / 2                 0.141
    4999   32.0        3.60         3.38         14          1 / 2 / 3                 0.117

  Twins are generic openings at every lower level (U1). At m997 the distribution of L_p over
  the 8,276 twins matches the exact cyclic gap distribution of m_p (period scan) to total
  variation 0.003-0.009 for p = 5..19, and the means agree to three figures (1.664 vs 1.667,
  2.316 vs 2.333, 2.844 vs 2.852, 3.376 vs 3.370, 3.842 vs 3.820, 4.299 vs 4.269). A twin's
  residues modulo the small gears are CRT-independent of its survival at the large ones, so
  its low-level flank is an unbiased sample of the generic word. Nothing about a twin is
  visible from below.

  The immediate neighbour is killed by 5 two-thirds of the time (0.706, 0.668, 0.664, 0.666
  at the four machines) against 0.41-0.47 for a generic blocked slot: the residue argument
  (a twin sits at k = 0, 2 or 3 mod 5, and two of those three put a tooth next door)
  predicts exactly 2/3. Adjacent twins (prime quadruplets) are 8%, 3%, 2%, 1.3%.

  Events per twin grow like the records of a Mertens-distributed sequence: the iid-records
  model (draw death rungs independently with the observed frequencies, count strict prefix
  maxima over the observed flank length) predicts the mean to 2-6%, the excess growing
  slowly with q (adjacent slots cannot both be killed by 5, which makes the true sequence
  slightly more varied than iid). Mean events 2.0 -> 3.6 from m53 to m4999 while the mean
  flank goes 6.7 -> 32: about one event per doubling of the flank.

  Every event is a binary merge, and the median merge DOUBLES the flank. For gears >= 11 the
  fused count is 1 in 99.4% of events (arity 2 in 0.6%, arity 3 four times in 346,031
  events at m4999; the theorem allows 5), so each event joins the current flank run to the
  next run across one dead opening. The median ratio new/old flank is 2.00 at every machine
  (mean 2.4-3.1), and the absorbed run is at least as long as the flank it joins in 49-55%
  of events. The twin bounding the maximal m997 run (851801, 851803) reaches its 242-slot
  flank in five events - gears 5, 19, 37, 137, 877 - with the tuples below [1], [1,1],
  [3,7], [11,26], [38,202]: the big run was assembled elsewhere and joined to the twin by
  one large gear at the very end.

  Mirror (U5): left and right flank statistics are identical (TV 0.0001 at m997, 0.0000 at
  m4999; the fraction with a big last gear agrees to three figures on both sides at every
  machine). The window sample sees the palindrome symmetry of the periodic word.

  Largest gear on the flank exceeds q/2 for 25%, 20%, 14%, 12% of twins (U4: band [0.25,
  0.55] missed by 0.003 at m53; [0.05, 0.20] held at m4999). Class of the largest flank gear
  at m4999: <= 7 for 6%, <= q/10 for 44%, <= q/2 for 38%, > q/2 for 12%.

### 6.3 What the paths say

  (a) The path to a twin is a chain of binary merges read upward from the twin: the flank
      absorbs the neighbouring run across one dead opening, roughly doubling each time, and
      the gears doing it are the running maxima of the death rungs outward from the twin.
      There is no branching structure specific to twins; a twin's path is what any opening
      of m_p sees at level p, continued to q because its residues at the remaining gears
      happened to miss the teeth.
  (b) The number of merges is logarithmic in the flank (records of a sequence), the flank is
      logarithmic-squared in the window, and every merge is width-1 in 99.4% of cases - so
      the in-window version of the increment law is very tame: a twin's gap grows at rung q'
      by (one dead opening + one neighbouring run), and only when q' is a new prefix maximum
      of its flank.
  (c) Nothing here is proof-relevant beyond confirming that in-window words are generic
      sieve words. The period-scale record (killer-spec.md, the ladder) remains the object.

## 7. The new section each machine adds (follow-up, same day)

Human's framing: machine 5 has window 25, machine 7 has window 49; look only at the new
section 26..48 that machine 7 adds, and likewise for every machine. Section of the rung
p -> q: the slots k with p^2 < 6k+1 < q^2. The sections partition the slots from k = 5 on.
Pre-registration research/data/r29/section_prereg.md (S1-S5), script
research/section_probe_r29.py, log research/data/r29/section_probe.log; 667 sections up to
q = 5003; 8 of 9 gates green, the failure (S4) is a bookkeeping error explained below.

### 7.1 What is forced

Every composite below q^2 has a prime factor <= p. So inside the section p -> q the gears
5..p are exact - the periodic word of m_p restricted to the section IS the twin-prime
indicator there - and the new gear q does nothing in its own section (its first kill is q^2,
the far edge). In this numbering the section attributed to machine q is the last stretch
where the PREVIOUS machine is still telling the truth, just before it starts lying at q^2.
The previous gear p enters only through p*m with m a prime in (p, q^2/p): the slot p^2 is
the section's near edge and excluded, so the candidates are p*q and at most two more.

The first sections as words (T = twin, number = death rung):

     5 ->  7  (25, 49)    slots 5..7     T 5 T
     7 -> 11  (49, 121)   slots 9..19    5 T 5 T 7 5 7 5 T T 5
    11 -> 13  (121, 169)  slots 21..27   5 7 T 5 T 5 7
    13 -> 17  (169, 289)  slots 29..47   5 T 5 T T 5 11 5 13 T 5 T 5 11 7 5 T 5 T
    29 -> 31  (841, 961)  slots 141..159 5 23 T 5 11 5 T 7 5 17 5 11 7 5 7 5 23 13 5

### 7.2 Measured

    q range     sections  min twins (at q)  max G_S/|S| (at q)  twins/H-L  gear-p kills = 0  last sealer > p/2
    5-100       22        2  (7)            0.684 (31)          0.987      0.73              0.95
    100-300     37        6  (109)          0.352 (109)         0.980      0.62              0.89
    300-1000    106       10 (463)          0.221 (601)         0.993      0.74              0.87
    1000-3000   262       21 (1153)         0.177 (1291)        1.007      0.79              0.89
    3000-5003   240       51 (3541)         0.092 (3253)        1.000      0.74              0.88

  No dead section (S1): every one of the 667 sections holds a twin; the minimum count rises
  2, 6, 10, 21, 51 across the bands, and the minimum is always at a gap-2 rung (7, 109,
  1153, 3541 are the upper members of twin-prime pairs), whose section is the shortest:
  |S| = (4q - 4)/6 slots.

  The section gap shrinks against the section (S2): G_S/|S| (largest gap between twins in
  the section or from an edge to the nearest twin) is below 1 everywhere, maximum 0.684 at
  29 -> 31 (19 slots, 2 twins, gap 13), then 0.352, 0.221, 0.177, 0.092 by band. At the gap-2
  rungs G_S is 0.5-2.9 times ln^2 q while |S| is 2q/3, so the ratio falls like ln^2 q / q.

  Twin counts are Hardy-Littlewood (S3): summed over the sections with 1000 <= q <= 5003
  the observed/predicted ratio is 1.0028 (band 3%), and 0.98-1.01 in every band; single
  sections scatter 0.6-1.4 as Poisson counts do.

  The old gear is almost invisible in the new section (S4): gear p is the death rung of at
  most 3 slots in its section, and of none at 77% of sections with q >= 500. My
  pre-registered band [0.35, 0.70] counted p^2 as a candidate; p^2 is the excluded near edge,
  so the candidates are 1-3 slots p*q, p*q_2, ... each credited with probability about
  1.7 / ln p, which gives 75-80% zero. Refuted as stated, explained by the boundary.

  Section Mertens: fraction of slots with death rung <= 7 is 0.571-0.577 (4/7 = 0.5714) in
  every band. Last sealer of the maximal run > p/2 at 88% of sections (S5 held): the same
  near-twin mechanism as the window (products of two primes from (p/2, 2p)).

### 7.3 What the sections say

  (a) The new territory a machine claims is sieved entirely by the OLD machine; the newest
      gear is silent there by construction and the previous gear touches at most three
      slots. Sections differ from one another only by scale, and at every scale they are
      Mertens words with Hardy-Littlewood twin counts.
  (b) A dead section p -> q would be a twin-free stretch of 2q(q - p) integers ending at
      q^2, i.e. a twin gap >= 4 sqrt(x) at x = q^2 for a gap-2 rung. Observed twin gaps in the
      sections are ln^2-scale; the margin G_S/|S| is 0.09 at q ~ 5000 and falls like
      ln^2 q / q. Killing twins for ever means every section dead from some rung on, and
      the section framing shows the first obstacle is already a sqrt(x)-size twin gap.
  (c) Nothing here is provable by the machine: twin gaps are unbounded in principle and
      the ln^2 scale is heuristic. The section decomposition is the clean way to state
      what the kernel iff demands rung by rung: survivor in the window = a twin in at
      least one section of the window, and the sections are independent samples of the
      same Mertens process at growing scale.

### 7.4 The trees inside the sections (follow-up; exploratory, not pre-registered)

The first pass over the sections computed aggregates only (twin counts, gaps, and the
sealing-gear count of the maximal run). This pass builds the fusion tree of the maximal
blocked run in every section, prints it for the sections ending at q = 31, 53, 199, 997,
1999, 4999, and measures the shape of all 667. Two examples from the log:

    29 -> 31 (19 slots, run of 12, 6 sealing gears): 23, 17, 13, 11 each kill one slot
    (fusing 2-, 3-, 3-, 4-tuples); 7 kills 3, leaving the 5-tuple [1,1,1,1,1]; 5 kills 5.
    991 -> 997 (1987 slots, run of 116, 28 sealing gears): every gear from 47 up to 607
    (15 levels) kills exactly one slot; 43, 41, 31 kill two; 23: 3, 19: 4, 13: 7, 11: 11,
    7: 19, 5: 47. Top merge: gear 607 fuses [62, 53].

Shape of the maximal-run tree by band (depth = number of distinct death rungs in the run;
single/depth = fraction of levels that kill exactly one slot; top chain = number of
consecutive single-kill levels from the top; top balance = shorter/longer piece at the
final merge):

    q range     sections  run len  depth  single/depth  top chain  chain/depth  top balance
    5-100         22       15.6     6.6     0.587         2.8       0.471        0.36
    100-300       37       39.7    12.6     0.579         5.6       0.463        0.36
    300-1000     106       85.9    20.8     0.578         9.8       0.461        0.45
    1000-3000    262      140.3    29.6     0.627        14.3       0.483        0.42
    3000-5003    240      197.8    36.8     0.626        17.7       0.477        0.40

Pooled over the 502 sections with q >= 1000, the top five levels are single-kill in 100% of
the trees, level 6-8 in 98-99%, level 12 in 91%.

What this says. The section trees are the window trees of section 2 in miniature and the
shape is scale-free: the top single-kill chain is 46-48% of the depth and the single-kill
levels are 58-63% of the depth in every band from q = 5 to q = 5003, while the run length
grows 13x. The top of every tree is a chain of one-slot binary merges (the near-twin
mechanism of section 2, now visible in runs of length 15 as well as 200), the bottom is the
5,7 comb, and the final merge is not balanced (the last gear joins pieces of ratio about
2:5 on average). Nothing new for a proof: it confirms that the run structure inside a
section is the generic Mertens tree at that run length, with no section-specific feature
from the gear p that owns the section (it kills at most three slots, section 7.1).

### 7.5 The tuple side (follow-up; exploratory, not pre-registered)

Sections 2 and 7.4 index the tree by gears. research/tuple_tree_r29.py forgets the gears
and reads the same trees as merge events on the pieces: a kill joins the piece of length a
on its left and b on its right into a + 1 + b (an "extension" when a or b is 0, a "join"
when both are positive). Maximal run per section, 667 sections, log
research/data/r29/tuple_tree.log.

    q range     merges   join frac   median join ratio   top quarter median   last-3 median
    5-100          343   0.359       0.500               0.444                0.392
    100-300       1469   0.385       0.500               0.400                0.368
    300-1000      9107   0.393       0.500               0.432                0.375
    1000-3000    36765   0.396       0.500               0.429                0.375
    3000-5003    47478   0.397       0.500               0.412                0.348

    join ratio min/max by stage (tenths of the merge sequence, q >= 1000):
    stage   4      5      6      7      8      9
    median  0.500  0.500  0.500  0.500  0.444  0.360
    mean    0.749  0.697  0.571  0.524  0.470  0.411
    (stages 0-3 are the gear-5 kills: no piece exists yet, every kill is an extension)

    most common join pairs (min, max), q >= 1000: (1,2) 21%, (1,1) 15%, (1,4) 11%,
    (2,3) 3%, (2,6) 2%, (4,5) 2%, (4,4) 2% - and 13% with both pieces >= 8.

    top 3-tuple as fractions of the run: 149 distinct patterns in 640 sections; the most
    common, (0.2, 0.3, 0.5), occurs 18 times (2.8%).

What this says. (a) The tuple trees are all different objects: no top 3-tuple pattern
reaches 3% of sections, so nothing repeats from section to section at the tuple level -
only the statistics of the merges are universal. (b) 60% of all merges are extensions
(the kill lands beside an already blocked slot), 40% are joins, in every band. (c) The
median join ratio is exactly 1/2 through the whole middle of the tree, in every band and
at every stage up to the 8th tenth. This is the 5,7 comb: after gears 5 and 7 the pieces
have lengths 1, 2, 4 only (section 2), and the middle gears join them in the pairs (1,2),
(1,1), (1,4), (2,4), ... so the typical join is "a piece plus one of half its length". The
doubling seen along twin paths (section 6.2, median flank ratio 2.00 per event) is the same
fact seen from the twin's side. (d) The top of the tree is unbalanced: the last merges join
pieces in ratio about 1:3 (median 0.35-0.44), not halves. The run is sealed when a
near-twin slot (one single-kill gear) closes the gap between a large piece and a piece a
third its size; the last gear does not "meet in the middle".

## 8. Provenance of the new twins (the object the human asked for; same day)

Sections 7.4 and 7.5 traced runs and merges; the human's question was the opposite object:
take each NEW twin (a twin in the section p -> q, the part of the window that machine q
adds), and trace the words it lives in - at level r it is an opening of the sub-machine m_r,
sitting at position k mod r of gear r's own word and inside a local word of m_r; going up,
that word is absorbed into a larger word whenever a gear kills the opening bounding it. The
old window is ignored (already checked by the smaller machine). Pre-registration
research/data/r29/provenance_prereg.md (V1-V4, scorecard), script
research/twin_provenance_r29.py, log research/data/r29/twin_provenance.log; 130,664 new
twins across the 667 sections up to q = 5003; 8/10 gates.

### 8.1 What a provenance looks like

Word at level r = (L_r, R_r), the gaps to the nearest openings of m_r; the letter string is
m_r on k-8..k+8. From the log, section 29 -> 31 (19 slots, two new twins):

    twin at slot 143 (857, 859); residues 5:3 7:3 11:0 13:0 17:7 19:10 23:5
      level  5 (k mod 5 = 3):   word (1, 2)  oxooxoxoTxoxooxox
      level 11 (k mod 11 = 0):  word (1, 4)  oxooxoxoTxxxoxxox
      level 23 (k mod 23 = 5):  word (3, 4)  oxooxoxxTxxxoxxxx
      level 29 (k mod 29 = 27): word (5, 4)  oxooxxxxTxxxoxxxx   final
      interacting gears: left [23, 29], right [5, 11]; framing pair (5, 5)
    twin at slot 147 (881, 883); residues 5:2 7:0 11:4 13:4 17:11 19:14 23:9
      level  5 (k mod 5 = 2):   word (2, 1)  xoxooxoxToxoxooxo
      level  7 (k mod 7 = 0):   word (2, 3)  xoxooxoxTxxoxoxxx
      level 11 (k mod 11 = 4):  word (4, 3)  xoxooxxxTxxoxxxxx
      level 17 (k mod 17 = 11): word (4, 10) xoxooxxxTxxxxxxxx
      level 23 (k mod 23 = 9):  word (4, 23) xoxxoxxxTxxxxxxxx
      level 29 (k mod 29 = 2):  word (4, 23) xxxxoxxxTxxxxxxxx   final
      interacting gears: left [5, 11], right [7, 17, 23]; framing pair (5, 5)

Every new twin of the sections 29 -> 31 and 47 -> 53 and two of 991 -> 997 are printed the
same way in the log. Gear 31 (resp. 53, 997) appears in no provenance: the section's own
gear kills nothing in its section (7.1), so a new twin's provenance ends at gear p.

### 8.2 Measured (all 667 sections)

  V1 - which residue combinations enable the new twins. Over the 122,546 new twins with
  q >= 1000, the residue classes are uniform over the tooth-avoiding classes to total
  variation 0.0026 (mod 5: 0.331 / 0.335 / 0.334 at k = 0, 2, 3), 0.0033 (mod 35, 15 open
  classes, least 0.0658 most 0.0675 against 0.0667) and 0.0097 (mod 385, 135 classes).
  V2 - framing pairs (death rungs of the two slots bounding the twin's final word). Joint
  within TV 0.024 of the product of its marginals; left marginal 5: 0.665, 7: 0.134,
  11: 0.045, 13: 0.028, 17: 0.017, right the same to three figures; (5,5) alone 44%.
  V3 - interacting gears per new twin (levels where the word changes, one side):

      q range     new twins   mean left   mean right   iid-records model
      5-100          191       2.20        2.14         2.17
      100-300        880       2.81        2.85         2.70
      300-1000      7047       3.21        3.19         3.03
      1000-3000    45710       3.52        3.51         3.31
      3000-5003    76836       3.69        3.69         3.47

  V4 - REFUTED. The largest gear interacting with a new twin exceeds p/2 for 46-48% of new
  twins at q >= 1000 (pre-registered 5-25%, carried over from the twin-path pass, whose
  11.7% was a window average). Checked by k-decile at m4999: 0.000 in the lowest two
  deciles of the window, 0.427 in the top decile (left flank alone 0.243). The section IS
  the top of the window, so the provenance of a new twin is framed by a gear above p/2
  about half the time - a fact the window average hid. Also refuted: the fraction of new
  twins whose word is final by level 13 is 0.7-2%, not 5-25% (a guess without derivation).

### 8.3 What the provenance says

  (a) The combination of gear interactions that enables a new twin is any tooth-avoiding
      residue vector, and the new section samples those vectors uniformly (V1 at three
      moduli, to a few parts per thousand). There is no preferred combination, no gear
      whose position in its own word makes a new twin more or less likely: the enabling
      pattern is the CRT product, nothing finer. This is the same statement as U1 for the
      window, now restricted to the new part where the human wanted it tested, and it is
      sharper there.
  (b) The two sides of a new twin are independent (V2): the left word and the right word
      are drawn separately. A new twin is framed by gear 5 on each side two thirds of the
      time (residue argument of 6.2), and by (5,5) 44% of the time.
  (c) The number of gears that touch a new twin's word grows like the records of an iid
      Mertens sequence (V3, model within 6%): 2.2 at q < 100, 3.7 at q ~ 5000, about one
      more gear per factor 10 in q. Provenance depth is ln ln-slow.
  (d) The one new thing: the top of a new twin's provenance is a big gear (> p/2) about half
      the time, because a new twin lives at numbers ~ p^2 where slots with no factor below
      p/2 in either number - near-twins - have density of the same order as twins, so each
      flank of ~ ln^2 p slots holds about one. The window average (11.7%) was dominated by
      the low part of the window where such slots cannot exist (a number below (p/2)^2 with
      no factor <= p/2 is prime). This is the section view earning its keep: the new twins
      are the ones framed by the newest gears, and the old window is not representative of
      them.
  (e) For the proof: the provenance is the twin's residue vector plus the records of its
      flank rungs; the first is uniform by CRT and the second is a Mertens records process.
      Killing twins for ever would need a rung from which no tooth-avoiding vector lands in
      the section, and (a) says the vectors that land are the generic ones with no
      preference - the kill would have to remove every class at once, not a pattern.

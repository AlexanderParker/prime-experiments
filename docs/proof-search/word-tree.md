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

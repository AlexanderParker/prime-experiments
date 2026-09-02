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

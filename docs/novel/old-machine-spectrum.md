# old-machine-spectrum - the qualifying ladder of a machine, computed on a machine below it

Round 23 (mechanic). Companion to `merge-law.md`: where the merge law computes the single
number F(M+q') from the old machine's gap word, this computes the whole QUALIFYING LADDER
Q_J of a machine r gears ahead, exactly, on the old machine's period - and the same
one-line mechanism yields a free upper bound on every F_j.

## 1. WHAT IT IS

Plain language. Adding a gear q' to the machine multiplies the period by q', and the new
period consists of q' "laps" of the old one, each lap being the old opening set with two
residue classes mod q' deleted; which two shifts from lap to lap and every phase occurs in
exactly one lap. So a window of the NEW machine is nothing but a window of the OLD machine
plus a lap number - and if you add r gears at once, a window of the old machine plus a
TUPLE of r phases, one per new gear, all r free and independent. Every extremal question
about the new machine that is local to a window therefore becomes a question about the old
machine's window list, at 1/(q_1...q_r) of the cost. The two consequences taken here are
(A) the qualifying spectrum Q_J of the new machine, exactly, and (B) a bound tying the
old machine's F_j ladder to the new machine's single F.

Setting. Machine M = gears 5..y acting on slot space; gear q blocks the two residues
+-6^{-1} mod q; openings = unblocked slots, period P = product of the gears; F(M) = largest
gap, F_j(M) = largest sum of j consecutive gaps, and

    Q_J(M; a) = max sum of J consecutive gaps whose J-2 MIDDLE gaps are all >= a

(the qualifying spectrum; the interior floor a = 2u'' = 2*round(q''/6) is the residue
necessity threshold of the gear after next). Let q_1 < ... < q_r be primes not in M and
M' = M + q_1 + ... + q_r, with period P' = P * q_1 * ... * q_r.

THEOREM (lap-phase transfer). The map k |-> (k mod P, (k mod q_1, ..., k mod q_r)) is a
bijection from Z_{P'} onto Z_P x Z_{q_1} x ... x Z_{q_r}, and k is an opening of M' iff
k mod P is an opening of M and, for each i, k mod q_i is not a tooth of q_i. Consequently a
maximal window of consecutive M'-openings is exactly a pair (window of consecutive
M-openings, phase tuple) such that: the two ENDPOINTS and the chosen SURVIVORS avoid every
gear's two teeth, and every other M-opening strictly inside the window is hit by at least
one of the r gears.

COROLLARY A (exact qualifying spectrum from below). For every J,

    Q_J(M'; a) = max over (window W of M, phase tuple c) of span(W),
                 subject to: exactly J-1 interior M-openings of W survive all r gears,
                 their consecutive distances are >= a, every other interior is killed,
                 and both endpoints survive.

Both sides are computed on M's period. Cost ratio to a direct scan of M': q_1...q_r.

COROLLARY B (deletion-ladder bound). F_{r+1}(M) <= F(M + q_1 + ... + q_r).
Proof: take the window realising F_{r+1}(M); it has exactly r interior M-openings; choose
the (unique) phase tuple that puts interior i on a tooth of gear q_i. All r interiors die.
If the endpoints survive the new gap is exactly F_{r+1}(M); if an endpoint also dies the
new gap containing the window is longer. Either way F(M') >= F_{r+1}(M). []
(r = 1 is merge-law.md's "F(M+q') >= F2(M) unconditionally"; the content here is that r
new gears buy r rungs of the F_j ladder, one designated kill each, because the r phases are
independent.)

THE RELAXATION IS EMPIRICALLY FREE. Formalist's round-22 construct Q^[J](M) drops the
survival requirement on the marked openings and on the endpoints, keeping only "every
unmarked interior is killed", which makes it a certified UPPER bound Q_J(M') <= Q^[J](M)
statable without any survival predicate. Measured over five steps (11->13, 13->17, 17->19,
19->23, 23->29) and every depth J = 2..7 - 30 of 30 entries - the relaxation is not merely
tight, it is EQUAL:

    step      J:          2     3     4     5     6     7    max   budget F(new)+q''
    11->13  Q_J(13)      16    18    23     0     -     -     23      28
            Q^[J](11)    16    18    23     0     -     -
    13->17  Q_J(17)      25    28    31    32    34     0     34      37
            Q^[J](13)    25    28    31    32    34     0
    17->19  Q_J(19)      31    35    37    38     0     -     38      48
            Q^[J](17)    31    35    37    38     0     -
    19->23  Q_J(23)      39    43    50    55    60     0     60      63
            Q^[J](19)    39    43    50    55    60     0
    23->29  Q_J(29)      55    65    68    71    71    71     71      74
            Q^[J](23)    55    65    68    71    71    71

## 2. WHY IT MIGHT BE NOVEL

The lap structure itself is CRT and is not new (it is the merge law's section 1(a), and
Holt-Rudd's cycle-of-gaps recursion is the one-residue-class analogue). What is not
standard is the use: an EXTREMAL statistic of a large modulus - and a whole ladder of
constrained extremal statistics, not just the maximum - is computed without ever
constructing that modulus, by enumerating windows of a small modulus against a free phase
tuple. Published computations of Jacobsthal-type values (Hagedorn 2009, Costello-Watts
2015, Ziller-Morack 2016/17) search each modulus from scratch; Holt-Rudd materialise the
whole new cycle. Corollary B is elementary enough that it is probably folklore for
Jacobsthal's function (h_k of a modulus vs h of a larger one) and is recorded here for the
free numbers it produces, not as a claim.

## 3. PROOF

Status: PROVED (elementary, above) + SCRIPT-VERIFIED at every step where an independent
exact value exists.

Scripts (all assertion-gated):
- `research/j5_census.py` - the r = 1 scan, three survival regimes R0 (no survival = the
  round-22 relaxation), R1 (endpoints survive), R2 (endpoints and marks survive = Corollary
  A). ANCHOR: R2 must reproduce the known exact Q_J(new); asserted at 11->13, 13->17,
  17->19, 19->23, 23->29 - the machine-23 ladder (39/43/50/55/60/0, r21 full-period scan)
  recovered from machine 19's period, and the machine-29 ladder (55/65/68/71/71/71, r17
  full-period scan) recovered from machine 23's period.
- `research/j5_deep.py` - segmented version for machines whose period does not fit in
  memory (machine 29: 1.078e9 slots, 214,708,725 openings).
- `research/j5_multi.py` - the r >= 2 version (free phase TUPLE). Validated at r = 1
  against j5_census, then run up the ladder from machine 23's period (7,952,175
  openings). Every reported witness is CRT'd to a real address of the TARGET machine and
  asserted there by `research/multi_witness_verify.py` (openings where claimed, every
  other interior slot blocked, every middle gap at or above the floor) - e.g. the r = 2
  J = 5 witness lands at machine-31 address k = 4,665,550,937 with gaps [5,18,37,30,1],
  which is the r17 census's own F_5 maximiser region, and the r = 4 J = 7 witness at
  machine-41 address k = 44,081,564,139,100 with gaps [2,16,17,35,28,20,14].

      r  new gears         target        period ratio    time   max_J Q_J   budget
      1  29                Q_J(29;10)             29    209 s        71       74
      2  29,31             Q_J(31;12)            899    338 s        91       95
      3  29,31,37          Q_J(37;14)         33,263    585 s       114      129
      4  29,31,37,41       Q_J(41;14)      1,363,783    601 s       132      134

  ANCHORS AT r >= 2: the r = 2 row reproduces machine 31's full-period ladder
  68/85/90/91/90/88 entrywise; the r = 3 row reproduces the two independently known
  machine-37 numbers F_2(37) = 90 and F_3(37) = 97 (the latter cost 55 SAT refutations in
  round 21) from three gears below, and its Q_4/Q_5/Q_6 = 103/110/112 sit under the exact
  F_4/F_5/F_6 = 105/113/120.
  A SOUNDNESS TRAP, recorded because it was hit: the walk may be stopped when a LOWER
  BOUND on the survivor count exceeds J-1, but with r >= 2 gears that bound is not
  monotone in the window length (it can lose r per step and gain 1), so stopping on it
  directly MISSES windows. What is monotone is the true minimum survivor count S(m) - the
  optimal phase tuple for a long window leaves at most as many survivors in any prefix -
  so the correct stop is on the RUNNING MAXIMUM of the lower bound. All r >= 2 values here
  are from the corrected version.
  A REPORTED "no window above X" is relative to the scan's span cap: windows longer than
  the cap are not examined, so a certification is conditional on "no admissible window of
  span above the cap". Failures (a window found above budget) carry no such condition.
- `research/j5_verify.py` - independent literal-enumeration controls: 295,763
  (window, phase, J) triples at machines 19 and 23 with the admissibility predicate
  decided by itertools.combinations, asserted equal to the scan's; plus the whole spectrum
  by brute force at the two smallest steps.
- `research/deletion_ladder.py` - Corollary B asserted at all 32 (M, j) pairs where both
  sides are known exactly (machines 13..37 against F(17)..F(53)). All pass; one attained
  with equality (F_2(17) = 25 = F(19)); tightest non-equality F_2(37) = 90 vs F(41) = 91.

## 4. IMPLICATIONS

INSIDE THE PROJECT.
1. It certifies (D)-route rungs from machines that are already scanned and, for the small
   ones, already kernel-checked. The 29->31 rung's input max_J Q_J(29;10) = 71 <= 74 comes
   from MACHINE 23's period, not machine 29's.
2. It corrects round 22. My round-22 tool reported max_J Q^[J](23) = 85 > 74 ("RUNG LOST",
   "the construct buys exactly one rung, not a ladder"). That was a bug in the mark-choice
   predicate (it returned success as soon as J-1 marks were placed, without checking that
   the interiors BEYOND the last mark were killed), and both conclusions are withdrawn.
3. Corollary B turns the corpus F-ladder into free exact caps for machines past the scan
   wall: F_2(41) <= F(43) = 103, F_3(41) <= F(47) <= F(53) = 145, F_2(47) <= 145. The
   first of these, combined with one SAT witness at exactly 103, PINS F_2(41) = 103 with
   no descent at all.
4. IT DECIDED THE PROJECT'S NAMED OPEN COMPUTATION, NEGATIVELY. Run at r = 5 and r = 6
   from machine 23 it evaluates max_J Q_J(43; 16) and max_J Q_J(47; 18) - the word-free
   criterion at the steps 43->47 and 47->53 - which no other method could reach (13-gear
   SAT refutations there run hours per instance and round 21 recorded them as
   undecidable). Both FAIL against their exact budgets, and the failure is confined to
   depths 6 and 7, so it is a statement about the criterion's depth allowance rather than
   about (D), which holds at both steps by the corpus ladder:
       43->47:  max_J Q_J(43;16) = 152 vs budget F(43)+47 = 150, witness Q_7 >= 152 at
                machine-43 address k = 110,350,776,715,218, gaps [35,20,20,17,20,17,23];
       47->53:  max_J Q_J(47;18) = 177 vs budget F(47)+53 = 171, witness Q_7 >= 177 at
                machine-47 address k = 41,120,916,229,562,503, gaps [14,20,36,19,20,45,23].
   Both witnesses are asserted at the target machine (every interior slot blocked, every
   middle gap at or above the floor). The period ratio bought at r = 6 is
   29*31*37*41*43*47 = 2,756,205,443, and the whole computation is a scan of 7,952,175
   machine-23 openings. The failures sit at depths 6 and 7 only, so they bound the
   criterion's DEPTH ALLOWANCE rather than (D).

OUTSIDE. Any extremal question about a primorial modulus that is local to a bounded window
transfers downward the same way. The limit is that the window must stay short relative to
the added gears, which is exactly the regime the Jacobsthal-type questions live in.

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

- (D) at alpha = 3 (the project's live route): supplies the qualifying inputs of a rung
  from one machine lower, which is where the kernel-checked censuses already are.
- Ziller-Morack style computation of paired Jacobsthal values: Corollary B is a free
  interlacing between the order-j values of one modulus and the order-1 value of a larger
  one, and every check it passes is a consistency check on the published ladder.
- Polignac / gap population: nothing directly.

## 6. PRIOR-ART CHECK

NOT YET CHECKED (no web access in this lane). Terms a checker should try: "Jacobsthal
function order k", "g(n,k) maximal gap k coprime residues", "cycle of gaps recursion",
"Holt Rudd cycle of gaps", "prime gap primorial lap structure CRT", "Hagedorn Jacobsthal
computation", "Ziller Morack Jacobsthal", and specifically for Corollary B
"h_k(n) <= h(n p_1 ... p_{k-1})". Expect PARTIAL OVERLAP at least: the lap structure is
standard, and Corollary B may well be classical.

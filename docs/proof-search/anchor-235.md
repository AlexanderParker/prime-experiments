# The anchor 2,3,5 line (the human's line of enquiry)

Manager, round 29 (2026-09-02/03). Scripts in research/anchor235/ (each self-contained, run
with the repository interpreter; the ones importing spf_sieve locate word_tree_r29 through a
relative path). Everything here is computed; nothing is pre-registered beyond the stated
expectations, and the mis-steps are recorded in section 9.

Conventions fixed on 2026-09-03 after a boundary slip (section 9):

  cycle j  = the numbers 30j+11, 30j+13, 30j+17, 30j+19, 30j+29, 30j+31 - three whole twin
             slots 11|13, 17|19, 29|31 - i.e. slots k = 5j+2, 5j+3, 5j+5 (numbers 6k +- 1);
  anchor-open slots = k mod 5 in {0, 2, 3}; anchor-open numbers = 1, 11, 13, 17, 19, 29 mod 30;
  gear q's hits = the anchor-open multiples q x m, m in 1..29 with q x m open, repeated every 30q;
  the cycle index of a hit n is (n - 11) div 30.

## 1. Why the anchor is 2,3,5

The anchor 30 has 6 open numbers per cycle, so a single gear q >= 7 hits 6 numbers per run of
30q and leaves q - 6 cycles untouched (2 for q = 7, two hits share a cycle). Anchor 210 has 48
open numbers per cycle: a gear hits 48 cycles per run and nothing is untouched until q > 48.
Anchor 6 has 2 opens per cycle but the cycles are single slots, no structure to see. So 30 is
the smallest anchor with untouched-cycle structure at every gear and the largest where that
structure starts at the very next gear (the stride result of lower-sieve.md section 5: untouched
copies only at {5} + 7).

## 2. One gear over the anchor (research/anchor235/anchor30b.py, recheck_cycles.py)

Four classes by q mod 30, the six m-values whose products q x m are anchor-open:

  q = +-1  : m = 1, 11, 13, 17, 19, 29        q = +-7  : m = 7, 11, 13, 17, 19, 23
  q = +-11 : m = 1, 7, 11, 19, 23, 29         q = +-13 : m = 1, 7, 13, 17, 23, 29

The six hits sit at fractions m/30 of the run; the untouched runs between them are q x (gap in
m) numbers long, linear in q, one shape per class. Checked for every prime 11 <= q <= 5000: six
distinct hit cycles, q - 6 untouched. Clean ends (first and last cycle of the run untouched)
from q = 37 on, no exception above; exceptions below: 11, 13, 17, 19 both ends hit; 29, 31 first
cycle hit (the gear's own prime q x 1); 7 and 23 clean at both ends.

Forced, no favourite slot type: q x m runs over the six residues 1, 11, 13, 17, 19, 29 mod 30
exactly once, so every gear hits each slot type (11|13, 17|19, 29|31) exactly twice per run -
once on the lower number, once on the upper. The class only fixes WHERE in the run.

## 3. Clean end zones and their alignment (ends.py, align.py)

Gear q's clean end zone is the window +-h_q around every multiple of 30q, h_q = q for the
classes +-1, +-11, +-13 and 7q for the class +-7 (the nearest anchor-open multiple is q x m_min,
m_min = 1 or 7). Two zones drift by 30(q' - q) per period and realign exactly at 30qq'; a set
of zones stacks exactly at 30 x prod q; the joint zone density is prod (2h_q / 30q).

Alignment of the end zones of ALL gears 37 <= q <= sqrt(n): exact search over n in
[1369, 10^7] - ZERO solutions. Closest calls: missing one gear's zone never past n = 1680 (one
gear required), missing two never past 2478, missing three never past 3550. Expected fraction
3.1e-2 with 2 gears, 3.8e-14 with 15. The end zones alone are too narrow to ever line up; the
ratio is the explanation, the exact search is the evidence.

## 4. Middle runs (middle.py, middle_align.py, section_trend.py)

Hit points of two gears q, q' coincide only at multiples of q x q' (first 1517 = 37 x 41). Nine
gears 37..71 below 300000: untouched numbers 50141 against prod (1 - 1/q) x anchor-open = 50165.
At slot level (slot_interact.py) the sharing is exact: gear q hits k = +-u_q mod q, so two gears
share a slot in exactly 4 residue classes mod q x q' - two where both sit on the same number
(multiples of q x q'), two where each takes one number of the slot (double kill); 12/5 of these
per q x q' are anchor-open; three gears share a slot only mod q x q' x q''. Tooth density
sum 2/q over 7..47 is 1.257 against a blocked fraction 0.745: 0.512 of the teeth land on
already-blocked slots, and that waste is entirely this rigid sharing.

Per section (q^2, q'^2), nothing from lower windows, to q = 5000: every section holds an
aligned slot (a twin); minimum per section rises 2, 3, 6, 7, 19, 21, 42, 51, 68 across the bins
5-50 .. 4000-5000; the longest blocked run of anchor-open slots inside a section grows like
q^0.51 (fit q >= 100) while the section grows like 2q ln q, so run/section falls: median 0.235
-> 0.020, worst 0.544 (29 -> 31) -> 0.085. Aligned count = anchor-open x prod_{7<=g<=q}(1-2/g)
x 0.66-1.0.

## 5. Cycles: the whole anchor surviving (cycle_survive.py, cycle_map.py, cycle_rule.py, next_cycle.py, cycle_sections.py)

Cycle j is untouched by gear q iff j mod q avoids six residues, ((q x m - 11) div 30) mod q
over q's six m (five residues for q = 7):

  q = 7: avoid {1,2,3,4,5}; 11: {0,2,3,6,8,10}; 13: {0,2,5,7,9,12}; 17: {0,3,7,9,12,16};
  19: {0,4,6,11,14,18}; 23: {5,8,9,12,14,17}; 29: {0,10,12,16,18,27}; 31: {0,11,13,17,19,29}.

Under gears 7..Q the open cycles are a fixed pattern of period prod q with prod (q - 6) open
cycles per period (2 for gear 7): gear 7 alone leaves j = 0, 6 mod 7; gears 7, 11 leave 10 per
77 (j = 7, 20, 27, 34, 42, 48, 49, 56, 62, 70); gears 7, 11, 13 leave 70 per 1001. A cycle open
under every gear q <= sqrt(30j + 31) has all six numbers prime. Below 10^8: 156 such cycles
(j = 0, 601, 3261, 5523, 13075, 22119, 33411, ...), all on the rule and none off it; 15350
cycles with two of the three twin slots; 409142 with one; 2908685 with none. Survivors split
78/78 between j = 0 and j = 6 mod 7. Rate: 0.56-0.63 x the sieve density sum over 10^6..10^8,
i.e. the prime-sextuplet rate ~ 1/ln^6.

Walk to the next open cycle from any q: start at the cycle holding q^2, step j, test j mod q
against the residue sets of every prime q <= sqrt(30j + 31), smallest first; no primality test
anywhere. From q = 37, 97, 499, 997, 4999, 10007, 100003 the walk lands at j = 601, 601, 13075,
33411, 833056, 3390372, 333444712 (all six numbers prime, checked afterwards), after 556, 288,
4776, 278, 57, 52371, 91379 cycles. Gear 7 makes ~71% of the rejections, 7 + 11 + 13 ~93%.

Against the window sections to 10^8: 1226 sections, 1088 with no open cycle, 121 with one, 16
with two, 1 with three; the share holding one rises 0% (q < 100) -> 13% (3000-10000); longest
dry stretch 50 sections (q = 7079..7549); position inside a section uniform (quartiles 0.24,
0.48, 0.74). The section is not the natural unit for whole cycles.

Existence: for any finite gear set open cycles exist for ever (prod (q - 6) per period, CRT).
For the growing gear set the question is the Hardy-Littlewood sextuplet conjecture for the
pattern p, p+2, p+6, p+8, p+18, p+20 - open, stronger than twin primes.

## 6. Slots: the object that is in every window (slot_walk.py)

Same construction, two forbidden residues per gear instead of six (the teeth +-u_q). Open
slots under 7..Q: period prod q, prod (q - 2) x 3/5 per period, never zero. The walk from q^2
by residue checks only: q = 37 -> 1427|1429 after 10 slots; 97 -> 9419|9421 after 2;
499 -> 249131|249133 after 22; 997 -> 994067|994069 after 10; 4999 -> 24990239|24990241 after
40; 10007 -> 100140119|100140121 after 12; 100003 -> 10000600481|10000600483 after 79 slots
(section 533392 slots). All twins.

## 7. The window question in this language (period_vs_window.py, where_worst.py, start_of_period.py)

At gear q the gears in play are 7..q, their untouched slots are a known periodic pattern, and
the kernel's window (proofs/BlockedSlots.lean:327, numbers (y, y^2], i.e. slots (q/6, W],
W = (q'^2 - 1)/6) is the opening stretch of that pattern. It includes every lower section; the
section-only statement "a twin in every (q^2, q'^2)" is STRONGER than the twin conjecture (a
dead section is a twin gap >= 4 sqrt(x), word-tree.md 268), and the kernel needs only one
survivor anywhere in (q, q'^2).

Full-period words (blocked-slot counts; the corpus F is these plus one, matching the ladder
7, 11, 18, 25, 34 exactly):

  q     period      worst run   window W   open in window   run entering the window
  7          35         4          12            4               2
  11        385         6           8            2               4
  13       5005        10          20            7               4
  17      85085        17          12            2               4
  19    1616615        24          28            4              11
  23   37182145        33          52            8               7

The worst run of the pattern is longer than the current section already at q = 17 (17 against
12; 144 runs >= W covering 2.3% of the period), and F(59) = 161 against a section of 40. So
existence in the section is positional: the run the pattern has AT q^2/6 is short (to q = 5000:
at most 0.663 of the section, q = 137; first twin a median 18 slots past q^2, max 264 at
q = 4637). The worst runs sit deep in the period in mirror pairs (positions k and P - k,
fractions 0.3-0.7 or at the period's ends), never at the window - word-tree.md 78 (T2').

Against the WHOLE window the position drops out: F(q) < W(q) - q/6 forces a survivor in the
window whatever the pattern does at q^2. Measured F/W = 0.25 flat from q = 5 to 53
(killer-spec.md 40), F(59) = 161 against W = 620. The record law F(M + q') = max_J Q*_J =
L (x) K* (x) R (constructor.md 498, attainment theorem) computes the new machine's worst run
from the current machine's windows; (D) F(M + q') <= F(M) + q' holds at every computable rung
through 59 (203 against budget 204 at 53 -> 59). If (D) holds at every rung then
F(y) <= sum_{q<=y} q ~ y^2 / (2 ln y) against W ~ y^2 / 6, so F/W <= 3 / ln y < 1 for y > 20,
a survivor in every window, twins infinite. The increment law is not generic (lateral.md 1699:
violated in 5-40% of counterfactual tooth placements) - a property of the real teeth
+-6^{-1} mod q, so a proof must use the machine.

## 8. What a new gear must do to lengthen the worst run (extension.py)

The new record of machine M + q' is an old stretch of consecutive openings x_0 < ... < x_J with
every interior opening killed by q' and both end openings surviving; kills = J - 1. Consecutive
kills differ by 0 (same tooth) or +-2u' (opposite teeth) mod q' - the record law's legal word.

  machine       + q'   F old -> new   old gaps inside   kills   kill residues   differences mod q'
  {5,7,11}       13      7 -> 11       [6, 5]            1       [11]            -
  {..13}         17     11 -> 18       [5, 11, 2]        2       [3, 14]         [11]   (teeth 3, 14; allowed 0, 6, 11)
  {..17}         19     18 -> 25       [7, 18]           1       [3]             -
  {..19}         23     25 -> 34       [4, 8, 15, 7]     3       [19, 4, 19]     [8, 15] (teeth 4, 19; allowed 0, 8, 15)
  53 -> 59 (corpus)     145 -> 161     [10, 118, 33]     2       same tooth      118 = 2 x 59 = 0 mod 59

"Both sides of a stretch" is the two-kill case: the two openings bordering a big old gap both
sit on teeth - same tooth when the gap is a multiple of q' (118 = 2 x 59), opposite teeth when
it is +-2u' mod q'. Parity is not a constraint: old gaps are mixed parity (odd slightly ahead:
69/66, 793/692, 12193/10082, 210051/168624), and since q' is odd the two opposite-tooth
differences 2u' mod q' and q' - 2u' have opposite parity, so any gap parity has a tooth
arrangement. The obstacle is arithmetic mod q', never parity. Extensions realised: +4, +7, +7,
+9 against allowances 13, 17, 19, 23; +16 of 59 at 53 -> 59.

## 9. (D) as a statement about the old word alone (record_decomp.py, both.py, mincount.py, three_in_q.py, gsparse.py)

Conventions in this section: F, F_2, F_3 are blocked-slot counts (corpus F minus 1); F_m is
the largest sum of m consecutive gaps of the old word {5..q} minus 1; F' = F(M + q'). All
figures are exact over the full period, rungs {5}+7 through {5..23}+29 (P = 37182145).

Decomposition of the new record at every rung (record_decomp.py):

  rung   F   F_2  F'   F'-F_2  s_min  kills  old gaps in the record (interior mod q')
  +7     1    3    4   1       2      2      [2, 2, 1]
  +11    4    6    6   0       4      1      [2, 5]
  +13    6   10   10   0       4      1      [6, 5]
  +17   10   15   17   2       6      2      [5, 11, 2]     11 = -2u'
  +19   17   24   24   0       6      1      [7, 18]
  +23   24   30   33   3       8      3      [4, 8, 15, 7]  8 = 2u', 15 = -2u'
  +29   33   38   42   4      10      2      [10, 10, 23]   10 = 2u'

Three facts.

  - Lower side, forced with no computation: F' >= F_2(M). The middle opening of any two
    consecutive old gaps dies in exactly 2 of its q' lifts; the ends survive (run = both
    gaps) or die too (run longer). The best one-kill run equals F_2 at every rung.
  - Upper side: F' - F_2 = 1, 0, 0, 2, 0, 3, 4 against s_min = 2, 4, 4, 6, 6, 8, 10 - the
    increment law, over every kill chain, not only the record (both.py: rung 23 has 733670
    one-kill chains max 30, 11746 two-kill max 32, 62 three-kill max 33; rung 29 has 15.4M
    one-kill max 38, 243822 two-kill max 42, no three-kill chain).
  - Interior gaps of the record chain are exactly +-2u' mod q', never a multiple of q':
    kills alternate teeth at the minimum stride. Since 3 x 2u' = 1 mod q', s_min = (q' +- 1)/3,
    so a chain of m kills spends at least (m - 1)(q' - 1)/3 slots on its interior.

Counterfactual teeth (both.py). Teeth {a, a + delta}: only delta matters, because the lifts
jP run over every residue mod q', so shifting both teeth relabels the lifts. Over every
delta = 1 .. (q' - 1)/2 at all eight rungs: (D) F' <= F + q' never fails (max over delta
4, 10, 15, 22, 27, 38, 49 against F + q' = 8, 15, 19, 27, 36, 47, 62; the real delta = 2u' is
never the worst). The increment law with budget s_min(delta) = min(delta, q' - delta) fails
only at small delta (rung 11: 1, 2; 13: 1; 17: 1, 5; 19: 1; 23: 2; 29: 2, 3, 4). So the
"not generic" verdict of section 7 is about the increment law; (D) itself is teeth-free on
every computed rung.

Teeth-free proof of (D) at a rung. Two teeth are two step-q' progressions, so within any q'
consecutive slots they kill at most 2 openings. Hence a run of the new word is a stretch of
the old word in which every q'-window holds <= 2 openings ("3-sparse"). Let G_t(M, q') be
the longest stretch in which every q'-window holds <= t openings (G_0 = F once F >= q').
Then

    F_2 <= F'(real teeth) <= max over delta F'(delta) <= G_2,   and   F_3 <= max over delta F'(delta)

(the last: three consecutive gaps d_0, d_1, d_2 - take delta = d_1 mod q' and kill both
middle openings). Measured (gsparse.py, three_in_q.py):

  rung    F   F_2  F_3   G_1  G_2   F + q'   margin F + q' - G_2
  +13     6   10   15     0   15     19       4
  +17    10   15   22     0   22     27       5
  +19    17   24   27    24   32     36       4
  +23    24   30   34    31   38     47       9
  +29    33   38   49    39   49     62      13

At rung 29 the cap is exact: G_2 = F_3 = 49, worst stretch gaps 23 | 4 | 23, and delta = 4
was the worst counterfactual (49). Counting alone (mincount.py: min openings in an
F + q' + 1 stretch against the two-progression capacity 2 ceil((F + q' + 1)/q')) proves
(D) at rungs 7, 11, 17, 19 and fails at 13, 23, 29; the q'-window criterion (some q'-window
of every F + q' + 1 stretch holds >= 3 openings) holds at all eight rungs, exactly 3 at
13, 17, 19, 29.

What must be proved for every rung, both about the CRT word {5..q} only:

  - sufficient:  G_2(M, q') <= F(M) + q'   (longest 3-sparse stretch <= record + one stride);
  - necessary for the delta-uniform form:  F_3(M) - F(M) <= q'   (three consecutive gaps
    never sum past record + q' + 1); measured 3, 6, 9, 12, 10, 10, 16 against q' = 7 .. 29,
    about half of q' with no clear trend (0.43, 0.55, 0.69, 0.71, 0.53, 0.43, 0.55).

Record gaps are isolated (both.py, part (a)): the neighbours of every record gap are
(1,2) at {5,7}; (1,3) at {5..11}; (2,2), (2,5) at {5..13}; up to 7 at {5..17}; <= 5 at
{5..19}; <= 7 next to any gap >= 0.8 F at {5..23}, <= 7 at {5..29}. F_2 - F = 2, 2, 4, 5, 7,
6, 5 against q' - s_min = 5, 7, 9, 11, 13, 15, 19. This is the part with no teeth in it at
all: why a record-size gap of a CRT word has only small neighbours.

Caveat on record: if F_3 - F ever crosses q' the delta-uniform route dies and only the real
delta = 2u' remains; extending F_3 and G_2 past rung 29 needs the ladder's stratified
dictionary (the full period is too large).

### 9a. Past rung 29 by search (ea_cover.py, ils_cover.py, tail.py; results/ils_run.txt, results/delta_run.txt)

The covering form makes every quantity above a search over offset vectors: F + 1 is the
longest interval [0, L) that offsets (c_g) can cover with every i congruent to c_g or
c_g + d_g (mod g) for some g; F_2 and F_3 allow one and two holes; G_2 allows at most two
holes per q'-window. ea_cover.py is a genetic algorithm on the offset genome (tournament
selection, 1-3 gear mutation, 20% uniform crossover, replace-worst, restarts); it recovers
F, F_2, F_3 exactly at {5..23} but stalls on larger machines (F 69 against 87 at {5..37}).
ils_cover.py is best-response coordinate ascent with sideways moves and random kicks; it
recovers F exactly at {5..29}, {5..31}, {5..37}, {5..41}, {5..53} and is short by 3 at
{5..43} and 1 at {5..47}. Everything below is therefore a lower bound (exact corpus F in
brackets).

  machine  q'    F      F_2   F_3   G_2   F+q'
  {5..29}  31   42[42]   54    64    69    73
  {5..31}  37   57[57]   67    84    92    94
  {5..37}  41   87[87]   89    96   119   128
  {5..41}  43   90[90]  102   109   132   133
  {5..43}  47   99[102] 115   122   152   149
  {5..47}  53  116[117] 131   144   174   170
  {5..53}  59  144[144] 144   160   194   203

The sufficient criterion G_2 <= F + q' fails from rung 47 (152 > 149, 174 > 170). The
stretches that beat it are not two-progression patterns (holes at 10, 12, 64, 69, 125, 129
in the 152-stretch), so (D) itself is untouched; the 3-sparse relaxation is simply too
loose past 43. The necessary condition F_3 - F <= q' holds everywhere, ratios 0.71, 0.73,
0.22, 0.44, 0.43, 0.51, 0.27.

Tooth spacing tested directly (results/delta_run.txt): add gear q' with teeth {a, a + delta} for
every delta and search the new record. Rung 41 -> 43: F + q' = 133, worst delta = 7 gives
111, the real delta = 14 gives 102 = exact F(43). Rung 43 -> 47: 149, worst delta = 19 gives
121. Rung 47 -> 53: 170, worst delta = 9 gives 139, real delta = 18 gives 129 (exact 144;
the search is weak here). No delta approaches F + q' at any of the three rungs; the real
spacing is never the worst one. Lower bounds can confirm survival, not refute it.

Why holes are cheap and the stride is not (tail.py): the number of gaps >= L in one period
falls like exp(-lambda L) with lambda between 0.46 and 0.69 at rungs 13..29; each allowed
hole multiplies the number of admissible configurations by about the number of hole
positions, so it buys about ln(F)/lambda slots. Predicted F_2 - F = 2.6, 3.9, 6.2, 6.6, 6.2
against measured 4, 5, 7, 6, 5; F_3 - F predicted 5, 8, 12, 13, 12.5 against 9, 12, 10, 10,
16. Holes buy logarithmic length; q' grows linearly. Heuristic, not a proof.

### 9b. Closed form for the walk: genetic programming (gp_walk.py, hopdepth.py; results/gp_*.txt)

The human asked for a genetic algorithm hunting a closed form for the walk, in the
machine's own parts. gp_walk.py evolves expression trees over the components a_g = (u_g -
s) mod g, b_g = (-u_g - s) mod g (distance from start s to gear g's next tooth), gear
sizes and small constants, with +, -, *, min, max, mod, floor-div, <, if-then-else;
fitness is the exact-match rate on all 5005 starts of {5,7,11,13} with a parsimony term;
every reported tree is also scored on the unseen machine {5..17}.

Direct target W(s) (distance to the next opening): population 1000, 400 generations, the
best tree plateaus at 55.6% (baseline "always 0" is 29.7%) with

    W = [a5 b5 a7 b7 a11 b11 min(a13, b13) ... == 0]

i.e. it rediscovers "W = 1 exactly when some gear hits s and the next slot is open" and
never learns any W >= 2. Nothing in the function set expresses the chain of hits.

Layered target H_g(s) = W_g(s) - W_{g-}(s), the hop the top gear adds at the lower walk's
landing x: the best tree is

    H = 0 if a_g(x) b_g(x) != 0 else 2

at 90.8% (92.2% on the unseen rung). That is the layered closure of section 8 found by the
search itself: the hop is zero unless the top gear hits the landing, and then it is the
lower word's walk again from x + 1. hopdepth.py measures the recursion: the fraction of
starts with no hit is exactly 1 - 2/g (0.8182, 0.8462, 0.8824, 0.8947, 0.9130 at g = 11,
13, 17, 19, 23), one hit 18%, 15%, 11%, 10%, 8.6%, two hits 0.2-0.3%, three hits essentially
never (the uniform-order theorem's <= 5 is the cap). Given one hit the hop is the lower
word's gap from the landing: 1 (11-26%), 2 (27-54%), 3, 5, up to the lower record.

So the walk closes layer by layer as

    W_g(s) = W_{g-}(s) + sum over hits of (1 + W_{g-}(landing + 1)),

with at most a handful of hits, and the residual the search cannot express is the lower
walk itself. A closed form for W_g would need a closed form for W_{g-} at the shifted
landings; none has been found. The genetic programming result is that the only structure
visible in the components at this depth is the hit indicator, and the searched function
set, which contains every arithmetic and comparison operation used elsewhere in this
document, does not close it.

## 10. Corrections on record

  - First anchor pass appended "the real machine grows, untouched happens exactly once" - the
    human said this overstepped the line of enquiry; withdrawn, memory saved.
  - Cycle boundary: the first cycle-survival table used blocks 30j..30j+29 for the residue
    rule while counting survivors on 30j+11..30j+31; the prime check flagged 155 "violations"
    that were real primes. Fixed to the convention above; zero exceptions; the cycle-indexed
    claims of section 2 rechecked under it (the sharing pair at q = 7 is 77/91, the 29/31
    exception is "first cycle hit"; substance unchanged).
  - "Every section holds an aligned slot is the twin prime conjecture" - overstated: it is
    stronger (section 7).

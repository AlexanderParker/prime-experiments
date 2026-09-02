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

## 9. Corrections on record

  - First anchor pass appended "the real machine grows, untouched happens exactly once" - the
    human said this overstepped the line of enquiry; withdrawn, memory saved.
  - Cycle boundary: the first cycle-survival table used blocks 30j..30j+29 for the residue
    rule while counting survivors on 30j+11..30j+31; the prime check flagged 155 "violations"
    that were real primes. Fixed to the convention above; zero exceptions; the cycle-indexed
    claims of section 2 rechecked under it (the sharing pair at q = 7 is 77/91, the 29/31
    exception is "first cycle hit"; substance unchanged).
  - "Every section holds an aligned slot is the twin prime conjecture" - overstated: it is
    stronger (section 7).

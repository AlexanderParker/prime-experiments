# Evidence for the step, generated locally while the lanes are paused (2026-09-08)

Scripts: research/stack/r3/section_start.py, research/stack/r3/blind_offsets.py,
research/stack/r4/step_object.py. Outputs in their results/ folders (gitignored); the numbers
used are here. Vocabulary: the stack by squares (cuts c_1 = base, c_{k+1} = p_k^2 with p_k the
first prime at or above c_k; section k = [c_k, c_{k+1}); machine k = the primes of section k,
machine 1 taking every prime from 5; a slot is (n, n + 2) with n = 5 mod 6).

## 1. Every computed link of every chain holds twins, and the first twin sits within a few cycles of the cut

Chain from base 3: cuts 3, 9, 121, 16129, 260,467,321 (first gears 3, 11, 127, 16139; the earlier
note "16141" was wrong, 16139 is prime). Twin-gear pairs per link: 1 ((5, 7) in [5, 9)), 8, 276,
1,027,948 (link 4 = [16129, 260,467,321), 14,218,065 gears). First twin above each cut: 11, 137,
16139, at offsets 2, 16, 10 numbers (0.1, 0.5, 0.3 cycles of 30). Chains from bases 5, 7, 11, 13,
17, 19, 23 (cuts to 7,946,761 and 260,467,321): every link holds twins; the first twin above a cut
is at most 160 numbers above it (5.3 cycles, base 7 at cut 2809), against arcs of 500 to 64,554
numbers. The first-twin scan of section_start.py: for every prime q to 200,000 (17,981 primes)
the first twin above q^2 lies below the long arc of q, 0 exceptions, median 0.09% of the arc,
worst 77% at q = 53. The scan to q = 10^6 (78,495 primes): 0 exceptions, median 0.02% of the arc, 90% 0.13%, 99% 1.0%, worst still 77% at q = 53; 28.9% of first twins in the 5/7-blind classes. The scan to q = 10^7 (664,576 primes): 0 exceptions; median offset below 0.005% of the arc, 99% at 0.15%, worst still q = 53 at 77% (53^2 = 2809, first twin 2969, 160 numbers above the cut); blind-class share 28.1%.

## 2. The band structure, checked cycle by cycle

At the start of every section k >= 2, per cycle of 30 for 200 cycles: twins = slots open under
machines 1 .. k-1, cycle by cycle, at every base (base 23, section 2: totals 125 = 125). The newest
machine engages only from the NEXT cut (gear g of machine k has its first genuine strike at
g x p_k >= c_{k+1}), so the "start-of-section excess" of stacked_squares.md S6 is the gradual
engagement of the newest machine: a count, not a structure.

## 3. Twin gears on the next section: they strike most because they are small; their collisions are a small waste

On section 2 of each chain (machine 1 striking): slots struck only by machine 1's twin gears
23,064, only by its other gears 2,782, by both 19,928, open 2,918 (base 23, 48,692 slots). The twin
gears dominate because the smallest gears (5, 7, 11, 13, 17, 19) are twins. Collisions of a twin
pair (slots struck by both members) against the pair's strikes: 8.6 to 10.8% where (5, 7)
dominates (bases 5 to 23, section 2), 0.6 to 4.8% on later sections (base 7, section 3: 5,090 of
797,084 = 0.64%). By arithmetic a twin pair collides on 4 of every p(p + 2) numbers against 4 of
every p it strikes, a waste near 1/(p + 2); measured on the anchored slots it is 2 to 5 times that
for the small pairs. Cheapness is real, small, and concentrated in the smallest pairs: it cannot
force a slot open by itself. Candidate 11a of the skeleton is a density correction, not a forcing.

## 4. Blind offsets

Relative to a prime square, gear g strikes offset i iff -6i or 2 - 6i is a nonzero square mod g;
the blind classes per gear number 1 + (pairs of nonzero squares differing by 2), verified against
brute force at 76 primes to 400 with 0 mismatches. Gears 5 and 7 are jointly blind at i = 5, 10,
12, 17 mod 35. Among the first twins above prime squares to 50,000, the four most frequent offsets
are 10, 17, 12, 5 (332, 219, 188, 158 of 5,130): the blind classes. The census of ALL twins below
the arc against a control start (section_start.py census, q to 20,000) shows the square is NOT
richer: 322,186 twins above squares against 321,052 above controls (ratio 1.0035); the square
only fixes the classes by exact availability fractions (1, 2/3, 1/2, 1/3, 1/6 of the primes q per
class mod 35). A score "how many small gears are blind at offset i" does not separate from the
offset's position in the arc in the first-twin data; not claimed.

## Verdict

The step is measured true at every computed link, with the first twin within a few cycles of the
cut. The square gives structure to WHERE (the blind classes, exact) and nothing to HOW MANY (the
control census). Twin-gear cheapness is a small density correction. What remains for the lanes: a
property of a section's survivors, not a count, that the squaring preserves.

## Review of the local runs (2026-09-08)

1. The arc bound is a count with a growing margin, not a structure. The expected number of twins
   in (q^2, q^2 + 4q] by the twin density near q^2 is about 1.3 q / ln^2 q: 4 at q = 53, 5 x 10^4
   at q = 10^7. The measured first-twin offsets sit where the density puts them (median a few
   hundred numbers above the cut at 10^7, i.e. 3 ln^2 q). The only place the bound is tested is
   small q, and the worst case, q = 53 at 77% of the arc, has not moved since 10^4. As evidence
   for the step it is strong; as a mechanism it is ROOT (existence by count).
2. The blind-class preference fades with q: the share of first twins in the four 5/7-blind
   classes is 30.0% at 2 x 10^5, 28.9% at 10^6, 28.1% at 10^7, against 26.7% uniform. The class
   census matches the availability fractions exactly (17: 51,698 against 7: 33,700 is 1.53,
   availability 1 against 2/3). The square fixes classes and adds no preference at large q.
3. Twin-gear cheapness is a density correction of a few per cent, concentrated in (5, 7).
4. The band structure is exact cycle by cycle; the newest machine engages only from the next cut.

Net: every local measurement of the square's structure (blind classes, arc bound, cheapness) is
either exact-but-a-count or fading. The one input no counter-machine reproduces remains the
recursion of the construction: a machine's gears are the survivors of the machines below. That
is where the Friday lanes go; the square-start lead is closed as FACT (blind classes exact) and
ROOT (existence in the arc).

## 5. The composite record per section (2026-09-09, research/stack/r4/composite_record.py)

Inside section k+1 the slots struck by machines 1..k are all slots but the twins (band
structure), so the composite record of machines 1..k on the section is the longest twin-free run
of slots inside it, and the step at link k is exactly "that record is shorter than the section".
Measured (sieve to 3 x 10^8):

| chain | section | length (slots) | twins | composite record (slots) | record / length | first twin above the cut |
|---|---|---|---|---|---|---|
| base 3 | [9, 121) | 19 | 8 | 4 | 0.21 | +2 |
| base 3 | [121, 16129) | 2,668 | 276 | 46 | 0.017 | +16 |
| base 3 | [16129, 260,467,321) | 43,408,532 | 1,027,948 | 579 | 1.3 x 10^-5 | +10 |
| base 5 | [25, 841) | 136 | 29 | 24 | 0.18 | +4 |
| base 5 | [841, 727,609) | 121,128 | 6,224 | 167 | 1.4 x 10^-3 | +16 |
| base 7 | [49, 2809) | 460 | 74 | 27 | 0.059 | +10 |
| base 7 | [2809, 7,946,761) | 1,323,992 | 48,249 | 254 | 1.9 x 10^-4 | +160 |
| base 13 | [169, 29,929) | 4,960 | 455 | 82 | 0.017 | +10 |
| base 17 | [289, 85,849) | 14,260 | 1,056 | 104 | 7.3 x 10^-3 | +22 |
| base 23 | [529, 292,681) | 48,692 | 2,917 | 153 | 3.1 x 10^-3 | +40 |
| base 31 | [961, 935,089) | 155,688 | 7,688 | 241 | 1.6 x 10^-3 | +58 |

The record inside a section is the largest twin gap inside it (the record at 187,913 in the
base-23 section is the gap 187,907 -> 188,831 of W103; at 850,355 the gap 850,349 -> 851,801).
Along the base-3 chain the record grows 4, 46, 579 slots while the section grows 19, 2,668,
43 million: the ratio falls from 0.21 to 1.3 x 10^-5. The step holds at every computed link with
a margin that widens like the square of the cut against the log-squared growth of twin gaps.
That widening is the count again (the density of twins), stated as a record; what the recursion
would have to supply is a reason the record of machines 1..k, which are the survivors of the
sections below, cannot reach the length of the section above them.

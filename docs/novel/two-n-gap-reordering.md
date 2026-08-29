# The 2n-gap reordering (human's sort-step idea, manager probe)

## 1. WHAT IT IS

Sort the machine's openings not by position but by PHASE VECTOR (CRT-lex: key =
(k mod q1, k mod q2, ...) lexicographic). In that order the adjacent differences
(mod P) take EXACTLY 2n DISTINCT VALUES at n gears - verified exactly at machines
[5,7,11] through [5,7,11,13,17,19,23]: 6, 8, 10, 12, 14. Natural order shows
7, 10, 17, ... (irregular growth).

Origin: the human's suggestion (2026-08-31) that the sliding windows, viewed in a
suitably SORTED order (a "sort step" per turn), should have an obvious gap pattern.
First probe confirmed it on contact.

## 2. WHY IT MIGHT BE NOVEL

The flavour is the three-distance theorem (a single rotation's sorted gaps take <= 3
values) generalised: the opening set is a PRODUCT SET in phase coordinates, and in
lex order its successor operation is an odometer - each gear plausibly contributes
exactly 2 step-values (ordinary increment + wrap), giving 2n. The lex-order result
may be an afternoon theorem; the FRAMING is the point: THE MACHINE = A TRIVIAL
PRODUCT ORDER x AN ARITHMETIC SHUFFLE, and every hard question (the record gap) is a
property of the shuffle alone.

## 3. PROOF

STATUS: MEASURED (exact, 5 machines; research/reorder_probe outputs in the session
log). The 2n law and the odometer mechanism are unproved. Candidate proof:
lex-successor analysis of product sets (each carry level contributes one step value
per direction).

## 4. IMPLICATIONS

If proved: a canonical coordinate system in which the opening set is trivial;
natural-order quantities (gaps, records) become properties of the permutation
between the two orders - a new object with exact structure. Possible connection to
Lateral's exposed-step law and gear-cell decomposition (both are partial versions of
"order by phase").

## 5. UNSOLVED QUESTIONS IT TOUCHES

Three-distance-type theorems for unions/products; the record gap as a shuffle
statistic.

## 6. PRIOR-ART CHECK

Not yet checked. Terms: three-distance theorem generalisations (Steinhaus), gaps of
Kronecker/Beatty sequences, lex-order successor of product sets / mixed-radix
odometers, "union of arithmetic progressions gap structure". NOTE: the lex-order 2n
fact may well be folklore; the delta to check is its application to sieve opening
sets and the shuffle framing.

# 17. The increment law at the six literal steps: what is kernel-checked, and what is not

## In plain words

At six specific steps, adding 13, 17, 19, 23, 29 and 31, the kernel checks two things: the old
machine really has a pair of neighbouring gaps of a stated total size, and no gap of the new
machine exceeds that size plus the new gear's smallest letter. Together they say that at those
steps the new record is at most the old best pair plus one small letter. This is true only at
those six steps: at the next step, adding 37, the record exceeds that amount by eight, and the
statement is not a general theorem.

## Vocabulary

Column `k` is the pair `(6k-1, 6k+1)`; machine `mN` is `{5..N}`; `F(M)` is the largest gap
between consecutive openings and `F_2(M)` the largest sum of two consecutive gaps (max-gap
convention).  For an incoming gear `q'`, `u' = round(q'/6)` and
`s_min(q') := min(2u', q' - 2u') = 2u'`, the smallest positive legal letter (file 05).

The **increment law** is the statement `F(M + q') - F_2(M) <= s_min(q')`.  Reading: one more
aligned strike buys at most one small letter over the old two-gap maximum -- unless the strike
is padded (the same tooth one lap later), when it is worth a full `q'`.  A step `M -> M + q'`
is **literal** if the record of `M + q'` is carried by a run whose middles are all literal
letters (no padded middle); the six literal steps are 11->13, 13->17, 17->19, 19->23, 23->29,
29->31.

Classical translation: `F_2(M)` is the longest stretch of columns containing exactly one
`M`-rough twin candidate strictly inside; the law compares the next machine's record with it.

## Statement

**Theorem (kernel-checked, six instances).**  At each of the six literal steps there are three
consecutive openings `a < b < c` of the old machine with `c - a = v`, and every gap of the new
machine is at most `v + s_min(q')`:

    step       s_min    v = the exhibited c - a    every gap of the new machine is
    11 -> 13     4      11  (252, 257, 263)        <= 15
    13 -> 17     6      16  (117, 122, 133)        <= 22
    17 -> 19     6      25  (110, 117, 135)        <= 31
    19 -> 23     8      31  (1118917, 1118927, 1118948)     <= 39
    23 -> 29    10      39  (19016898, 19016903, 19016937)  <= 49
    29 -> 31    10      55  (858386140, 858386160, 858386195)  <= 65

Consequently `F_2(M) >= v` and `F(M + q') <= v + s_min <= F_2(M) + s_min(q')` at each of the
six steps: the increment law holds there.  Moreover the exhibited `v` is sharp at 19, 23, 29
(`F_2(19) = 31`, `F_2(23) = 39`, `F_2(29) = 55` exactly).

**What is NOT a theorem.**  The increment law is not proved as a general statement; it is not
kernel-checked at any step beyond these six; and it is not true at the padded step 31->37,
where `F(37) = 88` against `F_2(31) + s_min(37) = 68 + 12 = 80` (measured, +8).  It is
measured to hold at 11 of 12 testable corpus steps and out of sample at 53->59.  It also fails
on 13-22% of the tooth-counterfactual family (0-6.5% with the incoming tooth pinned), so no
proof from "same gears, same density, symmetric teeth" can exist.  It is not called a law here
beyond the six steps.

## Proof

The theorem is six conjunctions, each with two halves.

1. **Lower half (realisability).**  For each step, the three listed columns are openings of the
   old machine and no opening lies strictly between consecutive ones; this is a finite check
   on the old machine's opening test (the tooth rule at each gear, file 02).  At 11, 13, 17 it
   is decided column by column; at 19, 23, 29 the listed column is a single column of a machine
   of period up to `1.08e9`, and the check is `exposed_iff` at that column and its gap
   interiors (`interval_cases` over at most 35 columns).  The columns themselves were found by
   the LP thread as phase vectors (exact-cover backtracking, no period scan) and turned into
   one column by CRT.  Hence `F_2(M) >= v`.
2. **Upper half (covering).**  At 11->13, 13->17, 17->19 the corpus already carries a kernel
   bound on the new record strictly tighter than `v + s_min`: `F(13) <= 11 < 15`,
   `F(17) <= 18 < 22`, `F(19) <= 25 < 31` (`Machine13.spectrum_one`, `Machine17.spectrum_one`,
   `Machine19.spectrum_one`, period scans in `proofs/Machine13Q.lean`, `Machine17Q.lean`, `Machine19Q.lean`).  At 19->23, 23->29, 29->31 the bound
   `F(M + q') <= v + s_min` (`39`, `49`, `65`) is a case-split LP-duality certificate at the
   increment width `W_inc = v + s_min`, strictly smaller than the budget width `F(M) + q'`
   (`48`, `63`, `74`): the two smallest gears' phases are held (35 cases), and in every case an
   exact rational dual certificate shows that no stretch of width `W_inc` is fully blocked
   (`IncCert23.F_le`, `IncCert29.F_le`, `IncCert31.F_le`; the mechanism is the one written out
   in file 18 for 31->37).
3. **Assembly.**  From `F_2(M) >= v` and `F(M+q') <= v + s_min`,
   `F(M+q') <= F_2(M) + s_min`.
4. **Sharpness at 19, 23, 29.**  The kernel also shows no bound `F_2 <= v - 1` holds
   (`Increment.f2_19_sharp`, `f2_23_sharp`, `f2_29_sharp`, from the realisers as indices of the
   gap sequence via `Increment.pair_attained` and the enumeration's completeness), so the
   ledger's `F_2` hypotheses at those machines cannot be lowered.

The base cases are kernel facts; there is no induction step, and the LP vehicle cannot supply
one (its cost is a primorial in the number of held gears).  The quantity that decides
certifiability at a step is `W_inc - F(M+q')`, negative at exactly one corpus step, the padded
31->37, where the increment width asks for something false.

## Status

Kernel: `Increment.increment_law_literal_steps` (the six-fold conjunction),
`Increment.increment_11_13`, `increment_13_17`, `increment_17_19`, `increment_19_23`,
`increment_23_29`, `increment_29_31`; realisers `Increment.f2_11`, `f2_13` (from
`Machine13.pair16_realized`), `f2_17`, `f2_19`, `f2_23`, `f2_29`; upper halves
`Increment.g13_le_11`, `g17_le_18`, `g19_le_25`, `IncCert23.F_le`, `IncCert29.F_le`,
`IncCert31.F_le`; sharpness `Increment.f2_19_index`, `f2_23_index`, `f2_29_index`,
`Increment.f2_19_sharp`, `f2_23_sharp`, `f2_29_sharp`, `Increment.pair_attained`
(`proofs/Increment.lean`, `proofs/IncCert23.lean`, `IncCert29.lean`, `IncCert31.lean`;
1,749 kernel jobs).

Verified computationally (beyond the kernel): the law measured at 11 of 12 testable corpus
steps, failing only at 31->37 by +8; confirmed out of sample at 53->59 (`F(59) = 161` against
`F_2(53) + 20 = 179`); differences `F(M+q') - F_2(M)` against `s_min`:
`0, 2, 0, 3, 4, 3, 20, 1, 0` at 11->13 .. 41->43; the 37->41 step certified by the LP thread
at the increment width (round 29, mixed split) but not kernelised.

## Prior art, and what is new

**Leverages.**  The letters of file 05, the frame of file 08, and the exact rational LP-duality
certificates whose mechanism is written out in file 18.  The literature sweep
(`research/proof/literature_increment.md`) fixes the status of the statement being certified:
the additive increment is not in print in either class count -- not as theorem, conjecture or
computation -- and the nearest published items are of a different shape, the multiplicative
Hajdu & Saradha 2012 Lemma 2.3 with its "very much likely" extension, promoted by Ziller 2019
Conjecture 3.2 to `H(k) < 2 H(k-1)`, and Hagedorn 2009 Proposition 2.8 (`h(n+1) = 2 w(n) + 2`),
which converts the increment away rather than bounding it.

**New.**  The six certified instances, and the width they are certified at: `v + s_min(q')`, the
increment width, strictly tighter than the budget width `F(M) + q'`, with the realiser and the
covering bound both kernel facts and sharpness of `F_2` at three of the machines.  Equally new
and load-bearing is the negative: the statement fails at 31 -> 37 by +8, so it is not offered as
a general theorem here.

**Not new.**  None of the six instances restates a published result.  The register does supply an
independent published witness for the negative: the increment fails for the published two-class
maximum at the step `{5,7,11} -> {5,7,11,13}` (A072753 jumps `10 -> 24`, i.e. `14 > 13`), so no
proof can pass through "any two classes per prime" and must use the real teeth -- the same
conclusion the tooth-counterfactual family reaches from inside the project.

## Relationship to the conjecture

Kernel base cases of a statement that is not a theorem and is false at 31->37.  They certify
six rungs of the budget inequality at the stronger increment width; no induction step exists,
and the general law is measured only and refuted on the tooth-counterfactual family.

## Where it is used

As base cases only.  The label "increment law" on the record refers to these six instances;
the target the search certifies rung by rung is the budget inequality `F(M+q') <= F(M) + q'`.
The increment law at a step implies the budget inequality there exactly when
`F_2(M) - F(M) <= q' - s_min(q')`, which is measured true at every computed step
(`F_2 - F = 2, 2, 4, 5, 7, 6, 5, ...` against `q' - s_min = 5, 7, 9, 11, 13, 15, 19, ...`)
but is not itself a theorem.

## Source

Manager round 26 (the statement); LP-duality thread rounds 27-28 (the certificates and the
witnesses); Formalist round 28 (`proofs/Increment.lean`);
`docs/proof-search/alignment-rules.md` 3.10.

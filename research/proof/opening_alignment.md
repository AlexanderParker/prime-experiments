# Openings aligning with the engine's twin slots (R4.c.v, prover, 2026-09-07)

Parent: R4.c (the valves). Spawned by the owner's mechanism of 2026-09-07: the manifold has
periods of alignment, runs of gaps and openings, evidenced by the longest gap being the period
of its first gear (all teeth aligned) and the longest opening being the arc of its first gear;
if the longest opening aligns over the engine's twin slots, which occur at a definite frequency
that phase-aligns one or more slots, that is the proof, within or outside the certified range.

Deliverable in the owner's words: whether the manifold's openings (maximal runs of consecutive
manifold-open pairs on the raw line) align with the engine's twin slots (pairs (n, n + 2) with
both members coprime to q#) so that some opening below Q^2 must hold a slot; an engine-and-
manifold-open pair below Q^2 is a twin (exhaust cap), so an aligned opening below Q^2 is a twin
prime pair sitting inside a manifold opening.

Vocabulary (the owner's, canonical): ENGINE (primes <= q; twin slots = engine-open pairs, a
fixed residue pattern mod q#), MANIFOLD (primes in (q, Q], teeth 0 and -2 on pairs, dominoes
{x, x + 2}), OPENING (a maximal run of L consecutive manifold-open pairs n, n + 1, ..., n + L - 1),
the run ceiling q' - 3 (L10, kernel `run_lt`, `run_attained`), smooth zone [1, Q], quiet zone
(Q, Q^2], TURN m = (mQ, (m + 1)Q] (the scratch lane's word), charge s x P (air s, fuel P).
Laws here are numbered V20 onward (V13-V19 reserved for the parallel lane).

Sources read: valves_scratch.md, valves_reconcile.md, top_machine_1.md section 4 (L2, L6, L10,
L11, L21), top_machine_lean.md (the L10 kernel rows), manifold_census_large.md, turn_ledger.md
V2, valve_existence.md V12 and the adversary's construction (research/valves/r2/invariants.py,
whose saved phases `used` are reused). Scripts in research/valves/r4/, outputs in
research/valves/r4/results/ (gitignored; every number used is in this file).

## Pre-registered (written before any script of this branch was run)

### The objects, exactly

An opening of length L >= 2 at n is the same thing as a maximal block of L + 2 consecutive
manifold-open INTEGERS n, n + 1, ..., n + L + 1 (proof: pairs n and n + 1 open give the integers
n, n + 1, n + 2, n + 3 open, and inductively the block; maximality of the pair run is
maximality of the integer block because n + 1 and n + L are open, so the struck member of the
bounding pairs n - 1 and n + L is n - 1, respectively n + L + 2). An opening of length 1 is an
open pair (n, n + 2) whose middle n + 1 may be struck, with (n - 1 or n + 1) struck and
(n + 1 or n + 3) struck. The run ceiling L <= q' - 3 is the integer statement "no gear q'
misses q' consecutive integers" read on blocks (q' - 1 open integers at most).

Per-gear factor. A gear g misses k consecutive integers in exactly g - k of its g phases
(for k <= g), so the density of "L + 2 consecutive open integers starting at n" over a full
period W = prod g is f(L + 2) with f(k) = prod_g (g - k)/g, and the count of openings of
exactly L >= 2 per period is the second difference W [f(L + 2) - 2 f(L + 3) + f(L + 4)] (L11);
for L = 1 it is prod(g - 2) - 2 prod(g - 4) + prod(g - 5).

Alignment over a full period is EXACT and GENERIC. The engine's slot set is q#-periodic, the
manifold's opening set is W-periodic, gcd(q#, W) = 1. In one period q# W the starts n of the
openings of length L hit every residue class mod q# exactly count(L) times. Hence the fraction
of openings of length L holding at least one slot is exactly |S_L| / q# where
S_L = {r mod q# : some r + i, 0 <= i < L, is a slot}. The slots mod 30 are 11, 17, 29 (gaps 6,
12, 12), so at q = 5: |S_L| = 3L for L <= 4, i.e. exactly 1/10, 1/5, 3/10, 2/5 of the openings
of length 1, 2, 3, 4 hold a slot, and 3/5 of the LONGEST openings hold none. At q = 7 (15 slots
mod 210, minimum slot gap 6) |S_L| = 15 L for L <= 6, less for L = 7, 8; at q = 11 (135 slots
mod 2310) |S_L| = 135 L for L <= 6, less for L = 7..10. The same argument gives, for any set of
gear phases (the free-phase copy), exactly the same numbers: over full periods every alignment
statistic is phase-blind. So whatever "structured alignment" exists can only live on a RANGE,
and in the quiet zone it lives in the turn structure: an open integer m in turn t is s x P with
q-smooth s <= t (or an ember), so a block of consecutive open integers carries a SMOOTH VECTOR
(s_0, ..., s_{L+1}) and no block with a given vector exists before turn max s_i (the scratch
lane's onset law, read on blocks). A block holds a slot iff s_i = s_{i+2} = 1 for some
i <= L - 1 (both members coprime to q#), i.e. iff it holds a twin prime pair (below Q^2).

### Predictions, with numbers, and what would refute each

- A-P1 (the spectrum below Q^2). For every (q, Q) run the count of openings of length L in
  (Q, Q^2] is BELOW the CRT count (Q^2 - Q) x [f(L+2) - 2f(L+3) + f(L+4)] for every L >= 2, by
  a factor that falls with L (the quiet zone's open density is 1/log Q at turn 1, rising toward
  the CRT value only at the top); the ratio measured / CRT for L = 1 is between 0.3 and 0.7 and
  for L = q' - 3 below 0.3. Refuted by a ratio above 1 at any L, or a ratio not falling in L.
- A-P2 (the longest opening present). The ceiling q' - 3 is attained below Q^2 at q = 5 (L = 4)
  for Q = 10^3, 10^4, 10^5, with counts in the hundreds at Q = 10^3 and Q = 10^4 (CRT
  estimate: prod_{7 <= g <= Q} (1 - 6/g) ~ 3 x 10^-5 at Q = 10^4, times 10^8, times a quiet-zone
  factor below 0.3); attained at q = 7 (L = 8) for Q = 10^4 with a count below 100 and possibly 0
  at Q = 10^3; NOT attained at q = 11 (L = 10) at Q = 10^3 and probably not at 10^4 (CRT count of
  order 20 before the quiet-zone thinning). So the manager's size fact (ii) "may not occur below
  Q^2 at all for large q" is predicted to hold from q = 11 on and to fail at q = 5. Refuted by
  the ceiling attained at q = 11, Q = 10^3, or missing at q = 5.
- A-P3 (the first opening of each length). The first opening of length L above Q sits in a turn
  >= the smallest possible max s_i over blocks of L + 2 consecutive integers (turn 1 for L = 1,
  since a twin pair is a block-free opening; for L >= 2 the block contains an even integer, so
  turn >= 2; for L >= 4 the block has three evens, one a multiple of 4 and one a multiple of 6,
  so turn >= 6 at q = 5 unless an ember lands in it). Exact prediction: no opening of length
  >= 4 below 6Q at q = 5 except ones holding a q-smooth number above Q. Refuted by one such.
- A-P4 (the engine's gap at each opening; the owner's prediction on the scorecard). Owner: long
  openings always contain a twin slot, and there is a threshold L* below the ceiling above
  which every opening holds one. Prover: no such L* exists at the period level (3/5 of the
  length-4 openings at q = 5 hold no slot, exactly), and on the quiet zone the fraction of
  openings of length L holding a slot is at most about |S_L| / q# and LOWER at the top length
  in the early turns (the earliest long blocks start at residues whose forced smooth parts are
  smallest, and those blocks, e.g. 1..6 mod 30 with smooth parts (1, 2, 3, 4, 5, 6), hold no
  slot; a slot-holding block of 6 needs its middle multiple of 6 fuelled at s >= 6 and its
  other evens at s >= 2, 4, so it enters later). Numbers: at (5, 10^4) the fraction of
  length-4 openings holding a slot is between 0.2 and 0.5; the number of slots per opening of
  length L is 0 or 1 for L <= 5 (slot gaps >= 6) and at most ceil(L/6). Refuted by every
  opening of some length L <= q' - 3 holding a slot, with at least 20 openings of that length.
- A-P5 (the composite with the first gear). The engine plus q' has period q# q' and its open
  pairs are the slots at the q' - 2 non-tooth residues mod q'; inside each arc of q' (residues
  1..q' - 3 mod q') every arc position receives every residue mod q# exactly once per period:
  GENERIC, exactly, and the same for every phase of q'. Adding gears one at a time over full
  periods thins the aligned openings by exactly the CRT factor (g - L - 2)/g per length-L block
  (exact identity, not a measurement). The deviation from CRT is confined to the range: in the
  quiet zone the real manifold's aligned openings are below CRT by the turn factor; a
  random-phase copy of the same gears on the same range is within sampling error of CRT (no turn
  structure, because its open integers are not s x P); the V12 greedy adversary has ZERO aligned
  openings in turns 1..60 by construction while its opening spectrum is within a factor 2 of
  the real manifold's. Refuted by a full-period count off CRT (would be an arithmetic error), or
  by a random-phase copy showing the real manifold's turn deficit.
- A-P6 (forced alignment). The first position n > Q at which an opening of length >= L holds a
  slot is, for L = 1, the first twin prime above Q (t_1 - Q = 19, 7, 151 at Q = 10^3, 10^4,
  10^5), which is not bounded by q# q' or by any smooth-zone constant (valves_scratch.md
  section 4: t_1 - Q ~ log^2 Q on average); for larger L it is later still and grows with Q.
  Predicted: for L = q' - 3 at q = 5 the first aligned opening is above 6Q at every Q; the
  quantity is unbounded in Q at every L. Refuted by a bound in terms of q# q' holding at all
  three Q.
- A-P7 (what alignment means for the adversary). For the adversary, an engine-and-adversary-open
  pair below Q^2 is not a twin (its fuel is not the primes); "an opening holds a slot" means a
  pure-imprint pair survives the adversary, which the adversary was built to prevent in turns
  1..60. Prediction: the real manifold and the adversary have the same full-period alignment
  fractions (exact), comparable opening spectra on the range, and differ only in where the
  openings' starts fall mod q# on the range; the adversary steers them off S_L with 774 nonzero
  phases, and the real manifold cannot steer because every phase is zero. So the alignment
  fact is "phase zero plus a count", the root, not a new invariant.

### Scorecard (filled after the runs)

| item | prediction | verdict | evidence |
|---|---|---|---|
| Manager (i): the longest opening q' - 3 is far shorter than the engine's longest blocked pair run | holds | HOLDS, exact: engine blocked pair runs 11, 29, 41 at q = 5, 7, 11 against ceilings 4, 8, 10; even the mean slot spacing q#/slots = 10, 14, 17.1 exceeds the ceiling; an opening holds 0 or 1 slot up to length 6 and two slots only at lengths 7-9 (1 case at (7, 10^4); 4 + 3 + 2 at (11, 10^4)) | Setup |
| Manager (ii): the longest opening is rare, density about exp(-(q' - 1) sum 1/g), may be absent below Q^2 for large q | present at q = 5, marginal at q = 7, absent at q = 11 | HALF: the density is exactly prod (1 - (q' - 1)/g) = 1.5e-4, 2.6e-5, 7.0e-6 (q = 5; Q = 10^3, 10^4, 10^5), 1.3e-6, 7.5e-8 (q = 7), 4.4e-7, 1.4e-8 (q = 11); the ceiling length is PRESENT in 6 of 7 runs (247, 3,634, 78,660; 6, 14; 0, 6 openings) and absent only at (11, 10^3) where CRT expects 0.4; it is present at (11, 10^4) with 6 openings where CRT expects 1.4 | Spectrum |
| Owner: long openings always contain a twin slot; a threshold L* below the ceiling | no L* | REFUTED with counts: openings of the ceiling length holding a slot 1,387 of 3,634 (q = 5, 10^4), 31,091 of 78,660 (10^5), 97 of 247 (10^3); 11 of 14 (q = 7, 10^4), 4 of 6 (10^3); 5 of 6 (q = 11, 10^4); one length below the ceiling 60 of 114 (q = 7), 12 of 20 (q = 11); no length with 20 or more openings is always aligned | Engine's gap |
| Owner: the slot frequency phase-aligns with the opening pattern so an opening must hold a slot below Q^2 | generic, not forced | GENERIC: over a full period the alignment is exactly uniform and phase-blind (V21, 0 mismatches in 15 stages x 3 phase sets x 3 engines); on the quiet zone the aligned fraction is 0.8 to 1.0 times the uniform value at every length (V24); the first aligned opening grows linearly with Q; the identity "aligned opening = twin" makes the forcing question the root (ROOT) | First gear, Forced alignment |
| A-P1 spectrum below CRT, ratio falling in L | below, falling | REFUTED: the ratio is ABOVE 1 and RISES with L in every real run (1.09, 1.19, 1.25, 1.38 at (5, 10^4); up to 4.2 at L = 10, q = 11); the random-phase copy is 1.00 at every L; mechanism found (V23) | Spectrum |
| A-P2 longest present: 4 at q = 5 all Q; 8 at q = 7 marginal; < 10 at q = 11 | as stated | HALF: q = 5 and q = 7 as predicted (8 present at both Q = 10^3 and 10^4); q = 11 attains 10 at Q = 10^4 (6 openings), refuting the "probably not" | Spectrum |
| A-P3 onset of long openings (turn >= 6 for L >= 4 at q = 5 unless an ember) | as stated | HOLDS and sharpened: V22 (0 exceptions in 65,115 ember-free openings); exact thresholds: a slot-free opening of length L can enter at turn L + 2, a slot-holding one not before turn 6 (L = 2), 12 (3 <= L <= 8), 16 (L = 9), 18 (L = 10) | Spectrum, Engine's gap |
| A-P4 fraction holding a slot 0.2-0.5 at the top length; slots per opening 0/1 for L <= 5 | as stated | HOLDS: 0.382 at (5, 10^4); 0 or 1 slot up to L = 6 | Engine's gap |
| A-P5 composite exact = CRT, phase-blind; deviation only on the range; random phase = CRT | as stated | HOLDS: every stage count equals the CRT integer exactly, random phases identical; on the range the random-phase copy is 1.000 +- 0.03 at every L and every truncation; the real manifold deviates, below CRT with gears to 1,000-3,000 (0.97 .. 0.87) and above CRT with all gears to Q (1.09 .. 1.38) | First gear |
| A-P6 first aligned opening unbounded in Q | as stated | HOLDS: first aligned opening of length 4 at 20,476 / 185,529 / 1,885,304 for Q = 10^3, 10^4, 10^5 (turn 20, 18, 18): linear in Q, not bounded by q# q' = 210; the turn is pinned just above the onset threshold 12 | Forced alignment |
| A-P7 adversary: same period statistics, zero aligned openings on the range | as stated | HOLDS with a correction: the adversary's aligned fraction is uniform (0.097, 0.201, 0.304, 0.408 at (5, 10^4)) and its first aligned opening of every length <= 4 is in turn 61 (610,049 .. 615,346), the first turn it did not cover; but the adversary is not a manifold on the integers: it breaks the run ceiling (1,809 openings over the ceiling at (5, 10^4), longest 9; 11 at (7, 10^4), longest 10) because a gear at nonzero phase lets its own multiples through | Forced alignment |

## Setup

Scripts (research/valves/r4/): `opening_spectrum.py q Q [model] [Gmax]` (segmented sieve of
(Q, Q^2] by the gears in (q, Gmax]; openings as maximal runs of the pair indicator; per opening
its length, its start residue mod q#, its turn, the number of twin slots inside it; models
`real` = phase zero, `rand:<seed>` = every gear strikes one random class of integers, `adv` =
the V12 one-tooth adversary on the fuel with the phases saved by research/valves/r2);
`composite_exact.py q` (the engine with the first 1..5 manifold gears over full periods, phase
zero and two random phase vectors per stage); `opening_vectors.py q Q` (the smooth vectors of
the stored long openings, the onset check, the slot position); `onset_threshold.py q` (the
smallest possible max smooth part of a block of L + 2 consecutive integers, slot-holding and
slot-free, searched to n = 2 x 10^7). Outputs in research/valves/r4/results/.

Runs: (q, Q) = (5, 7, 11) x (10^3, 10^4) and (5, 10^5) real; rand:1 at all six (q, Q) and
rand:2 at (5, 10^4); adv at (5, 7) x (10^3, 10^4); the truncation series Gmax = 7, 11, 13, 17,
19, 23, 29, 50, 100, 300, 1000, 3000 at (5, 10^4) for real and rand:1. Wall time 0-4 s per run
at Q <= 10^4, 225 s at Q = 10^5, one core, under 600 MB.

Validation. Open pairs in (Q, Q^2 - 2]: 100,058 / 141,276 / 171,943 at Q = 10^3 and
5,376,501 / 7,765,397 / 9,665,288 at Q = 10^4 for q = 5 / 7 / 11, the scratch lane's charge
counts exactly; 323,712,275 at (5, 10^5), the census's 323,712,289 less the 14 smooth-zone
pairs. Openings holding a slot, summed over lengths and counting two-slot openings twice:
8,134 (Q = 10^3), 440,107 (Q = 10^4, all three q), 27,411,455 (Q = 10^5): the twin counts. No
opening exceeds the ceiling q' - 3 in any real or random-phase run (0 of 5.4 million at
(5, 10^4), 0 of 316 million at (5, 10^5)).

The engine's slots and its blocked runs (exact, mod q#): 3 slots mod 30 (11, 17, 29), longest
blocked run of pairs 11; 15 slots mod 210, longest blocked run 29; 135 slots mod 2310, longest
blocked run 41. |S_L| (residues r mod q# such that one of r, ..., r + L - 1 is a slot): 3L for
L <= 4 at q = 5; 15L for L <= 6 then 102, 114 at q = 7; 135L for L <= 6 then 924, 1038,
1152, 1266 at q = 11.

## The opening spectrum on the quiet zone

**V20 (openings are blocks; exact).** An opening of length L >= 2 at n is a maximal block of
L + 2 consecutive manifold-open integers n, ..., n + L + 1; an opening of length 1 is an open
pair {n, n + 2} whose middle may be struck. Proof in the pre-registration. Consequences: the per
gear factor for a run of L >= 2 open pairs is (g - L - 2)/g (the gear must miss L + 2 consecutive
integers), the exact-length count per period is the second difference of prod (g - k) at
k = L + 2 (L11), and the ceiling q' - 3 is "q' cannot miss q' consecutive integers".

**The spectrum, measured against CRT** (count of openings of exactly L in (Q, Q^2], then the CRT
count (Q^2 - Q)[f(L+2) - 2f(L+3) + f(L+4)], then the ratio):

| (q, Q) | L = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| (5, 10^3) | 87,260 / 76,834 / 1.14 | 4,282 / 3,381 / 1.27 | 1,082 / 765 / 1.41 | 247 / 146 / 1.69 | | | | | | |
| (5, 10^4) | 4,995,397 / 4,577,112 / 1.09 | 145,592 / 122,906 / 1.19 | 25,128 / 20,143 / 1.25 | 3,634 / 2,644 / 1.38 | | | | | | |
| (5, 10^5) | 309,595,110 / 300,584,063 / 1.03 | 5,761,714 / 5,430,829 / 1.06 | 759,699 / 697,793 / 1.09 | 78,660 / 69,685 / 1.13 | | | | | | |
| (7, 10^3) | 112,828 / 99,801 / 1.13 | 7,698 / 6,104 / 1.26 | 2,580 / 1,913 / 1.35 | 851 / 564 / 1.51 | 262 / 154 / 1.70 | 73 / 38 / 1.93 | 16 / 8.0 / 2.0 | 6 / 1.3 / 4.6 | | |
| (7, 10^4) | 6,862,857 / 6,141,930 / 1.12 | 295,394 / 239,781 / 1.23 | 73,331 / 55,510 / 1.32 | 17,192 / 12,064 / 1.43 | 3,527 / 2,418 / 1.46 | 741 / 434 / 1.71 | 114 / 66 / 1.72 | 14 / 7.5 / 1.86 | | |
| (11, 10^3) | 128,853 / 115,523 / 1.12 | 10,187 / 8,296 / 1.23 | 4,006 / 2,993 / 1.34 | 1,529 / 1,041 / 1.47 | 557 / 347 / 1.61 | 204 / 110 / 1.86 | 50 / 32 / 1.54 | 20 / 8.8 / 2.27 | 7 / 2.1 / 3.3 | 0 / 0.4 / - |
| (11, 10^4) | 8,230,284 / 7,282,329 / 1.13 | 429,700 / 341,809 / 1.26 | 124,260 / 91,420 / 1.36 | 34,642 / 23,537 / 1.47 | 8,931 / 5,794 / 1.54 | 2,382 / 1,350 / 1.76 | 571 / 294 / 1.94 | 134 / 58 / 2.30 | 20 / 10.1 / 1.98 | 6 / 1.4 / 4.2 |

The random-phase copy of the same gears on the same range: ratios 1.000, 1.002, 0.993, 1.028
(seed 1) and 1.000, 1.001, 1.003, 0.998 (seed 2) at (5, 10^4); 1.000, 0.999, 0.999, 1.006,
0.987, 0.978, 1.09, 0.93 at (7, 10^4); 1.00, 1.003, 0.998, 0.996, 0.993, 1.03, 1.04, 0.93,
1.09 at (11, 10^4); 0.998, 1.014, 0.997, 1.12 at (5, 10^3). So CRT is exactly the free-phase
expectation, and the real manifold's spectrum is ABOVE it by a factor that rises with the
length: A-P1 refuted in sign and in trend.

**Where the excess sits (per turn band, real manifold, ratio to CRT).** (5, 10^4): L = 1: 0.30
in turns 1-2, 1.01 in 3-10, 1.49 in 11-100, 1.35 in 101-1000, 1.06 in 1001-9999; L = 4: 0, 0.47,
3.87, 2.81, 1.21. (5, 10^5): L = 1: 0.30, 1.04, 1.64, 1.54, 1.02 (bands to 99,999); L = 4: 0,
0.36, 4.74, 3.97, 1.10. (7, 10^4): L = 6: 0, 0, 3.58, 4.09, 1.45. The truncation series at
(5, 10^4), real / random, L = 1..4: identical to CRT to three decimals for gears up to 300
(ratios 1.000; the truncated period 7 x 11 x ... x 29 divides the range many times); with gears
to 1,000 the real manifold is BELOW CRT, 0.971, 0.944, 0.922, 0.906, and with gears to 3,000
0.953, 0.911, 0.884, 0.867, both falling in L; with all gears to 10^4 it is above, 1.091,
1.185, 1.247, 1.375, rising in L; the random-phase copy stays at 1.000 +- 0.02 throughout.

**V23 (the quiet-zone excess; MEASURED, mechanism stated).** The count of openings of length L in
(Q, Q^2] exceeds the CRT count by a factor rising with L, at every (q, Q) run; the excess is
absent in the free-phase copy and vanishes in the bottom two turns (ratio 0.30 for L = 1,
0 for L >= 3). Mechanism: with every phase zero the fuel is the primes above Q, whose density
at height y is 1/log y, i.e. at y = Q^u it is e^gamma / u times the density prod (1 - 1/g) that
the free-phase copy has at every height (u < e^gamma = 1.78 is most of the zone in log scale);
and an open integer with smooth part s has its fuel at height y / s, where primes are denser
still. A block of L + 2 consecutive open integers carries L + 2 such factors, its evens and
multiples of 3 and 5 the largest, so the excess compounds with L. In the bottom turns the block
cannot exist at all (V22 below), whence the deficit there; with the gears truncated at
1,000-3,000 the fuel is not the primes but the numbers free of primes to 1,000 on a range
reaching 10^8, which are thinner than CRT (the sieve at a third or a quarter of the range's
exponent), whence the deficit there. Prior art, one line: the factor e^gamma is Mertens'
constant, the truncated deficit is Buchstab's function below 1; neither is new, and the branch
uses them only to place the excess.

**The longest opening present below Q^2 (law of the longest, with counts).** The density of a
block of q' - 1 consecutive open integers is exactly prod_{g in (q, Q]} (1 - (q' - 1)/g)
(the manager's exp(-(q' - 1) sum 1/g) is its leading term): 1.465e-4, 2.644e-5, 6.969e-6 at
q = 5 (Q = 10^3, 10^4, 10^5); 1.308e-6, 7.521e-8 at q = 7; 4.371e-7, 1.418e-8 at q = 11. The
CRT count of openings of exactly the ceiling length in the zone, and the measured count: 146
vs 247, 2,644 vs 3,634, 69,685 vs 78,660 (q = 5); 1.3 vs 6, 7.5 vs 14 (q = 7); 0.4 vs 0,
1.4 vs 6 (q = 11). Law (measured, 7 of 7): the ceiling is attained below Q^2 whenever the CRT
count of the ceiling length is at least 1, and the measured count is 1.1 to 4.6 times the CRT
count (the excess of V23 at its largest, since the ceiling length is the longest block); the
one absence is (11, 10^3), where the longest present is 9 (7 openings, CRT 2.1). The longest
openings sit high in the zone: first opening of the ceiling length at 18,362 (turn 18),
109,012 (turn 10), 894,664 (turn 8) for q = 5; 356,137 (turn 356) and 703,121 (turn 70, holding
the ember 703,125 = 3^2 5^7; the first ember-free one 1,627,055, turn 162) for q = 7; 3,822,027
(turn 382) for q = 11 at Q = 10^4. Their smooth vectors at q = 7 and 11 carry entries like 42,
64, 90, 180, 210, 363: a block of 10 or 12 consecutive open integers must fuel every member,
and the multiples of 8, 9, 12, 7 x 6, ... among 12 consecutive integers have large air.

**V22 (onset of openings; EXACT, 0 exceptions in 65,115).** An ember-free opening of length L in
turn t has every member s_i x P_i with P_i a prime above Q, so t >= max_i s_i (the scratch lane's
onset law read on the block). Checked on every stored opening of length >= q' - 4 in the seven
real runs: 0 exceptions in 1,318 + 23,625 + 39,992 (q = 5) + 20 + 127 (q = 7) + 7 + 26 (q = 11).
Exact thresholds (onset_threshold.py, the minimum of max_i s_i over all blocks, searched to
n = 2 x 10^7; the extremal patterns are the blocks 1..L+2 and 8..13 and repeat with a small
period): a slot-free opening of length L >= 2 can enter at turn L + 2 (vector (1, 2, ..., L + 2)),
and a slot-free opening of length 1 at turn 3 (the pair (1, 3): the family (1, 3) or (3, 1) of
the scratch lane); a slot-holding opening of length L cannot enter before turn 1 (L = 1, the
twin itself, vector (1, 1)), turn 6 (L = 2, vector (1, 6, 1, 4)),
turn 12 (3 <= L <= 8, vectors (9, 10, 1, 12, 1), (8, 9, 10, 1, 12, 1), ..., (4, 5, 6, 7, 8, 9,
10, 1, 12, 1)), turn 16 (L = 9, q = 11) and turn 18 (L = 10, q = 11, vector (8, 9, 10, 11, 12, 1,
14, 15, 16, 1, 18, 1)). The mechanism: the member between the twin is a multiple of 6 with air
6 only if it is 2 mod 4 and not 0 mod 9, and then the even two steps away is 0 mod 4 and
carries air 4 x 5 = 20 when it is also 0 mod 5, or the block's other multiple of 6 is 0 mod 4
and carries 12; every slot-holding block of 5 or more consecutive integers pays air 12
somewhere. So the SLOT-HOLDING long openings enter the quiet zone TWICE as late (turn 12) as
the slot-free ones (turn 6 at the ceiling length of q = 5), unless an ember (a q-smooth number
above Q) supplies a member with no fuel.

Observed against the thresholds: the first opening of length 4 at q = 5 is slot-free in all
three runs, at turn 18 (vector (2, 3, 4, 5, 6, 1)), 10 ((4, 1, 6, 5, 8, 9)), 8 ((8, 5, 6, 1, 4,
3)); the first slot-holding one is at turn 20 (10^3, an ember 20,480 = 2^12 x 5 inside; the
first ember-free at 20,714, vector (2, 15, 4, 1, 18, 1), turn 20), 18 (10^4, vector (3, 10, 1,
12, 1, 2)), 18 (10^5, vector (8, 15, 2, 1, 12, 1)); the first slot-holding opening of length 3
at (5, 10^4) is at 129,586, turn 12, vector (2, 1, 12, 1, 10), exactly on the threshold. The
first slot-holding openings of length 2 are ember-seeded at every Q: 2,591 (with 2,592 = 2^5
3^4), 21,598 (with 21,600 = 2^5 3^3 5^2), the ember law of the scratch lane in the opening
picture: below turn 6 a twin can sit in an opening of length 2 only next to an ember.

## The engine's gap at each opening

Slots per opening: 0 or 1 for every length up to 6 in every run; two slots in 1 opening of
length 7 at (7, 10^4) and in 4, 3, 2 openings of lengths 7, 8, 9 at (11, 10^4); never three.
The manager's size fact (i) in its exact form: an opening of length L holds at most
ceil(L / 6) slots because consecutive slots are at least 6 apart, and the ceiling q' - 3 is
below the mean slot spacing 10, 14, 17.1.

**The fraction of openings of length L holding a slot, against the uniform value |S_L| / q#:**

| (q, Q) | L = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform, q = 5 | 0.100 | 0.200 | 0.300 | 0.400 | | | | | | |
| (5, 10^3) | 0.080 | 0.167 | 0.289 | 0.393 (97 / 247) | | | | | | |
| (5, 10^4) | 0.081 | 0.184 | 0.290 | 0.382 (1,387 / 3,634) | | | | | | |
| (5, 10^5) | 0.084 | 0.191 | 0.295 | 0.395 (31,091 / 78,660) | | | | | | |
| uniform, q = 7 | 0.071 | 0.143 | 0.214 | 0.286 | 0.357 | 0.429 | 0.486 | 0.543 | | |
| (7, 10^3) | 0.058 | 0.108 | 0.180 | 0.257 | 0.286 | 0.562 (41 / 73) | 0.312 (5 / 16) | 0.667 (4 / 6) | | |
| (7, 10^4) | 0.056 | 0.125 | 0.195 | 0.270 | 0.333 | 0.387 | 0.526 (60 / 114) | 0.786 (11 / 14) | | |
| uniform, q = 11 | 0.058 | 0.117 | 0.175 | 0.234 | 0.292 | 0.351 | 0.400 | 0.449 | 0.499 | 0.548 |
| (11, 10^3) | 0.048 | 0.087 | 0.140 | 0.188 | 0.230 | 0.382 | 0.340 (17 / 50) | 0.400 (8 / 20) | 0.571 (4 / 7) | - |
| (11, 10^4) | 0.045 | 0.098 | 0.154 | 0.214 | 0.258 | 0.317 | 0.382 (218 / 571) | 0.463 (62 / 134) | 0.600 (12 / 20) | 0.833 (5 / 6) |

Random phase at (5, 10^4): 0.100, 0.201, 0.304, 0.406 (seed 1), 0.100, 0.201, 0.297, 0.400
(seed 2); at (7, 10^4): 0.072, 0.141, 0.214, 0.286, 0.363, 0.475, 0.47, 0.57; at (11, 10^4):
0.058, 0.116, 0.173, 0.234, 0.293, 0.359, 0.43, 0.37, 0.36.

**V24 (alignment on the quiet zone is generic; MEASURED).** The fraction of openings of length L
holding a slot is 0.78 to 1.0 times the uniform value |S_L| / q# at every length with 100 or
more openings, in every run; the deficit is largest at L = 1 (0.80, 0.81, 0.84 at q = 5;
0.82, 0.79 at q = 7; 0.83, 0.78 at q = 11) and shrinks with L (0.98, 0.955, 0.99 at the ceiling
of q = 5). The random-phase copy sits at 1.00. Mechanism of the deficit: a slot-holding block
has two members with air 1 (the twin), which get no height boost, while a slot-free block of
the same length can put air on every member; the same V23 mechanism, read on residues. By
turn band the aligned fraction rises toward uniform: (5, 10^4), L = 4: 0.326 in turns 11-100,
0.363 in 101-1000, 0.388 in 1001-9999 (uniform 0.400); L = 1: 0.952 in turns 1-2 (the only
length-1 openings there are twins and ember pairs), 0.228 in 3-10, 0.104 in 11-100, 0.083,
0.080. The slot's position inside an aligned block is uniform: L = 4 at (5, 10^4): 341, 363,
359, 324 at positions 0..3; L = 3: 1,937, 1,872, 1,958. The member between the twin carries air
6 most often (307 of the 1,387 aligned length-4 openings at (5, 10^4), then 12: 162, 18: 131,
30: 136, 24: 74), i.e. the twin's midpoint over 6 is a prime above Q or a smooth number.

**No threshold L*.** At the ceiling length the slot-free openings are 2,247 of 3,634 (q = 5,
10^4), 47,569 of 78,660 (10^5), 150 of 247 (10^3); 3 of 14 (q = 7, 10^4), 2 of 6 (10^3);
1 of 6 (q = 11, 10^4). One below the ceiling: 54 of 114 (q = 7), 8 of 20 (q = 11). The owner's
"long openings always contain a twin slot" fails at every length with 20 or more openings and
at the ceiling in every run; the largest lengths at q = 7, 11 hold a slot in 11 of 14 and 5 of 6
cases, which is 0.79 and 0.83 against uniform 0.54 and 0.55 (small counts; the random-phase
copy gives 4 of 7 and 4 of 11 at the same lengths).

**Twins by the length of the opening they sit in.** (5, 10^4): 404,635 of 440,107 (91.9 %) in
openings of length 1 (both neighbouring pairs struck), 26,804 (6.1 %) in length 2, 7,281
(1.65 %) in 3, 1,387 (0.32 %) in 4. (5, 10^5): 26,056,906 (95.1 %), 1,098,974 (4.0 %), 224,484
(0.82 %), 31,091 (0.11 %). (7, 10^4): 382,799; 36,866; 14,272; 4,635; 1,176; 287; 60; 11 (+ 1 in
a two-slot opening). (11, 10^4): 368,289; 41,953; 19,075; 7,428; 2,300; 756; 218; 62; 12; 5
(+ 9 in two-slot openings). Per pair of an opening the slot rate is flat: 0.081, 0.092, 0.097,
0.095 at (5, 10^4) for L = 1..4 (uniform 0.100); twins do not prefer long openings beyond
the number of pairs a long opening offers.

## The first gear and the engine, exactly

**V21 (period alignment is uniform and phase-blind; EXACT).** Over one period q# W of the
engine with any set of manifold gears at any phases, the openings of length L start in every
residue class mod q# exactly count(L) times, so exactly |S_L| count(L) / q# of them hold a slot,
the slot positions inside the arcs are equidistributed, and adding a gear g multiplies the
number of blocks of length k, aligned or not, by exactly g - k. Proof: gcd(q#, W) = 1 (CRT).
Verified by full enumeration (composite_exact.py): q = 5, stages {7}, {7, 11}, {7, 11, 13},
{7, 11, 13, 17}, {7, 11, 13, 17, 19} (periods 210 to 9,699,690): openings of length 4 number
30, 150, 1,050, 11,550, 150,150 (= CRT, each stage x5, x7, x11, x13 = g - 6) and 12, 60, 420,
4,620, 60,060 of them hold a slot (= count x 12/30); every length's count equals its CRT
integer and the starts mod 30 are exactly uniform (min = max in every class); two random
phase vectors per stage give identical counts, 0 mismatches. Same at q = 7 (stages to
{11, 13, 17, 19}: length 8 counts 210, 630, ... with 114/210 aligned) and q = 11 (stages to
{13, 17, 19}: length 10 with 1,266/2,310 aligned). The arc table of q' alone: the long arc
(residues 1..q'-3 mod q') seen from its start residue r mod q# has a slot at each of its
positions for exactly 3, 15, 135 of the q# starts (uniform in the position); arcs holding
0 / 1 / 2 slots: 18 / 12 / 0 (q = 5), 96 / 108 / 6 (q = 7), 1,044 / 1,182 / 84 (q = 11). The
alignment of the engine's pattern with the first gear's arc is GENERIC in the strongest sense:
it is the same for every phase of every gear, and the same for the random copy.

Consequently every non-generic feature of alignment lives on the range, and on the range it is
the difference between the real manifold and its free-phase copy: the truncation series shows
the two agree to three decimals while the gears' product divides the range many times (gears
to 300 at (5, 10^4)) and separate only when the gears reach the range's scale, the real one
first below CRT (gears to 1,000-3,000) then above (all gears); the aligned FRACTION, however,
stays within 0.8-1.0 of uniform in every real run and truncation (0.103, 0.206, 0.306, 0.405
at gears to 1,000; 0.081, 0.184, 0.290, 0.382 with all gears). Phase zero moves the spectrum
(V23); it does not move the alignment beyond the twin's own deficit (V24).

## Forced alignment

**The first aligned opening of length >= L above Q** (position, turn; E = holds an ember):
(5, 10^3): L = 1: 1,019 (turn 1); 2: 2,591 (2, E); 3: 10,935 (10, E); 4: 20,476 (20, E).
(5, 10^4): 10,007 (1); 21,598 (2, E); 129,586 (12); 185,529 (18). (5, 10^5): 100,151 (1);
652,241 (6); 345,599 (3, E, the ember 345,600 = 2^9 3^3 5^2); 1,885,304 (18). (7, 10^4):
10,007; 21,598; 129,586; 175,961 (17); 320,609 (32); 421,708 (42); 2,486,189 (248); 703,121
(70, E). (11, 10^4): ...; 3,734,836 (373, L = 9); 3,822,027 (382, L = 10). In units of Q the
first aligned opening of length 4 sits at 20.5, 18.6, 18.9: it grows linearly with Q and is
bounded by no structural constant (q# q' = 210, 2,310, 30,030 are all passed at Q = 10^4); what
is bounded is its TURN, pinned just above the onset threshold 12 of V22 by the density of
slot-holding blocks there. The first aligned opening of length 1 is the first twin above Q
(valves_scratch.md section 4: t_1 - Q ~ log^2 Q on average, unbounded).

**The adversary.** For the V12 one-tooth free-phase adversary (each gear removes one class of
the FUEL, phases chosen greedily to empty the pure charge in turns 1..60) an "aligned opening"
is an engine-open pair both of whose members are adversary-open, i.e. a pure-imprint pair the
adversary failed to cover; it is not a twin, since the adversary's fuel is not the primes. Its
alignment is uniform where it is not steered: aligned fractions 0.097, 0.201, 0.304, 0.408
(L = 1..4) at (5, 10^4) and 0.072, 0.131, 0.197, 0.260, 0.317, 0.416 at (7, 10^4), and its
first aligned opening of every length <= 4 is in turn 61 (610,049; 612,850; 615,346; 611,409 at
q = 5; 610,187; 610,996; 623,379; 610,689 at q = 7), the first turn it did not cover; at
(5, 10^3), where the adversary ran out of gears (V12), its first aligned opening is 1,199 in
turn 1. Its opening spectrum is not the manifold's: it breaks the run ceiling (1,809 openings
over the ceiling at (5, 10^4), longest 9; 114 at (5, 10^3), longest 8; 11 at (7, 10^4),
longest 10; 3 at (7, 10^3), longest 11), because a gear g at nonzero phase kills the fuel class
c_g and lets the multiples of g through, so the adversary is a machine on the fuel, not a
manifold on the integers; and it shares the real manifold's excess over CRT at long lengths
(2.6 at L = 4, (5, 10^4)) by a different mechanism (the rough parts (m + i)/s_i of a block are
not consecutive, so one class per gear covers fewer of them than g - k phases would).

What distinguishes the real case: over a full period nothing (V21 is phase-blind); on the
range, the real manifold's aligned openings are twins and appear at the uniform rate from
turn 1 (deficit 0.8), while the adversary's are steered to zero for as many turns as it has
gears to spend; the real manifold cannot steer because every phase is zero, and "the
zero-phase manifold's openings hold a slot in every zone" is the conjecture (V12's verdict,
unchanged). The counting content is the twin count; the structural content is the localisation
of V22: the slot-holding openings of length >= 3 enter at turn 12, the slot-free ones at
turn L + 2, and the first twins in long openings are ember-seeded.

## Laws

- **V20** (openings are blocks; EXACT, proof above). Opening of length L >= 2 = maximal block of
  L + 2 consecutive manifold-open integers; per-gear factor (g - L - 2)/g.
- **V21** (period alignment uniform and phase-blind; EXACT, CRT). Over a period q# W, openings
  of length L start in every class mod q# exactly count(L) times; |S_L| / q# of them hold a
  slot; the same at every phase. 0 mismatches in 15 stages x 3 phase sets x 3 engines.
- **V22** (onset of openings; EXACT, 0 exceptions in 65,115). An ember-free opening with smooth
  vector (s_i) lies in turn >= max s_i. Thresholds: slot-free openings from turn L + 2;
  slot-holding from turn 6 (L = 2), 12 (3 <= L <= 8), 16 (L = 9), 18 (L = 10).
- **V23** (the quiet-zone excess; MEASURED with mechanism). The opening count in (Q, Q^2]
  exceeds CRT by a factor rising with L (1.09 to 1.38 at (5, 10^4), to 4.2 at L = 10,
  (11, 10^4)); the free-phase copy is at 1.00; the excess is the primes' falling density
  (e^gamma / u at height Q^u) compounded over the block's members at heights y / s_i.
- **V24** (alignment on the zone generic; MEASURED). The fraction of openings of length L
  holding a slot is 0.78 to 1.0 times |S_L| / q#, every run, every length with >= 100 openings;
  no length is always aligned; slots per opening at most ceil(L / 6).

## What is new

The block form V20 and the onset thresholds of V22 (slot-holding long openings enter the quiet
zone at turn 12, slot-free ones at turn L + 2, the extremal vectors named) have no located
prior art in the project; V21 is the CRT read on openings and is recorded so that the "phase
alignment" question has an exact answer at the period level; V23's mechanism is Mertens and
Buchstab in the machine's words (prior art, one line, not new); V24 and the twin-by-length
table are the localisation the brief asked for. Toward the root: nothing here forces a slot
into an opening below Q^2; the branch's exact content is where the aligned openings CANNOT be
(below turn 12 without an ember, for lengths 3 to 8) and that the manifold's own pattern is
blind to the engine's pattern over its period.

## Verdict

ROOT. An opening below Q^2 holding an engine slot is a twin prime pair, exactly (exhaust cap),
so "some opening holds a slot below Q^2" is "there is a twin in (Q, Q^2]", the conjecture with
the window cut to the quiet zone, restated. The owner's mechanism - the longest opening
aligning over the slots - is refuted as a forcing: the alignment of the manifold's openings
with the engine's slots is exactly uniform and phase-blind over a period (V21), generic to
within the twin's own deficit on the quiet zone (V24: 0.8 to 1.0 of uniform), and the longest
openings hold no slot in 62 % (q = 5), 21 % (q = 7, 3 of 14) and 17 % (q = 11, 1 of 6) of
cases. What the opening structure adds is localisation, all of it exact or measured with
mechanism: twins sit in openings of length 1 in 92-95 % of cases, in the ceiling length in
0.1-0.3 %; the slot-holding long openings enter the zone at turn 12 and the slot-free ones at
turn L + 2 (V22), so the earliest long openings are slot-free by arithmetic, and the earliest
twins inside long openings are seeded by embers; the manifold's opening spectrum on the quiet
zone exceeds its CRT count by a factor rising with the length (V23), which is phase zero's
one visible act on the spectrum and does not touch the alignment. The two size facts: (i)
holds exactly (blocked runs 11, 29, 41 against ceilings 4, 8, 10; at most ceil(L/6) slots per
opening); (ii) holds as a density (prod (1 - (q' - 1)/g), 1.5e-4 down to 1.4e-8) but the
ceiling length is present below Q^2 in 6 of 7 runs, absent only where CRT expects 0.4.

## Dead ends (with the refuting instance)

- A-P1, the quiet zone thins the spectrum below CRT: refuted by every real run (ratio 1.14 at
  L = 1, 1.69 at L = 4, (5, 10^3)); the zone ENRICHES it, and the random-phase copy shows the
  CRT value is the free-phase expectation, not a ceiling.
- "The longest opening is absent below Q^2 for q >= 11": refuted at (11, 10^4), 6 openings of
  length 10, the first at 3,822,027 holding the twin (3,822,029, 3,822,031).
- "Long openings always hold a slot" (owner): refuted at every length with 20 or more openings
  (2,247 slot-free of 3,634 at the ceiling of (5, 10^4)); at the ceiling of q = 7 and 11 by
  3 of 14 and 1 of 6.
- A structural bound on the first aligned opening: refuted by the linear growth 20,476 /
  185,529 / 1,885,304 with Q; only the turn is pinned (20, 18, 18), by V22 plus density.
- Alignment as an invariant separating the real manifold from the adversary: over the period
  both are uniform (V21); on the range the adversary is steered to zero exactly where it chose
  to be and uniform elsewhere (turn 61 on); the separation is V12's phase zero, not alignment.
- The mean smooth part of aligned versus slot-free long openings as a discriminator: dominated
  by embers (means 244 vs 37,855 at (5, 10^4), L = 4); dropped.

## Open items of the part, sorted

- Closed here: the block form (V20); period alignment (V21); the onset thresholds (V22); the
  spectrum's sign against CRT (V23); the alignment fraction on the zone (V24).
- Measurement with no structural content: the exact excess factors of V23 per (q, Q, L); the
  twin-by-length percentages; the 0.8 deficit at L = 1.
- Root question in disguise: any opening holding a slot below Q^2; the first aligned opening's
  position; the aligned count in any turn.
- Genuinely open on the part alone: a closed form for the onset threshold of a slot-holding
  block of k consecutive integers as a function of k and the engine (the data says 12 for
  5 <= k <= 10 and 16, 18 for k = 11, 12; the attack is the exact minimisation over residue
  patterns mod 2^a 3^b 5^c, a finite computation per engine, and its formalisation); and
  whether V23's excess factor at the ceiling length has a closed form in u = log x / log Q and
  the block's smooth vector (an integral of prod 1/log(x/s_i) over the zone against the CRT
  product), a heuristic, not a law.

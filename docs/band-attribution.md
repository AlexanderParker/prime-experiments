# The band-by-band attribution matrix

Script: `research/band_attribution.py`. Computed over all 10^6 candidate pairs to midpoint `6*10^6`
(168 bands). Bands are `B_h = (p_h^2, p_{h+1}^2]` over consecutive primes; a candidate pair
`(6k-1, 6k+1)` lives in the band containing its midpoint; each composite member is attributed to its
root gear `q = lpf(m)` (unique by law L2), and `q` to the band containing it. Motivation: the
fresh-block recursion - in any band, gear `q`'s fresh blocks sit at `q*r` with `r` running over the
*primes* in `band/q`, so the machine's blockers in each band are its own output from lower bands,
re-entered as structure (the concrete form of the feedback loop of `twin-prime-program.md` section 17d).

## The matrix (shown to band <=53^2; full matrix from the script)

    band(mid)      cand twin dead |  <=3^2  <=5^2  <=7^2  <=11^2
       <=11^2     12    4    8 |      8      1      0      0
       <=13^2      8    2    6 |      5      2      0      0
       <=17^2     20    7   13 |     12      6      0      0
       <=23^2     28    4   24 |     17     12      0      0
       <=29^2     52    8   44 |     33     23      1      0
       <=37^2     68   11   57 |     42     29      8      0
       <=47^2     60   11   49 |     37     25     12      0
       <=53^2    100   13   87 |     63     37     19      1

The matrix is the square-root tower made visible: deaths in band `h` draw root gears only from bands
up to the tower-half (the band containing `sqrt` of band `h`'s numbers) - a hard lower-triangular
cutoff, exact by law L3.

## Findings at 6e6

* **Volume is carried by the lowest gear bands, with a long tail.** Gears `{5,7}` account for 39.60%
  of all 1,587,153 member-kills; `{11..23}` for 24.57%; band `<=7^2` (gears 29..47) for 9.41%; and the
  shares decay slowly - every band contributes, matching the exact `2/q` ledger per band. No band is
  negligible (the low-energy-beats lesson in attribution form).
* **Half of all blocking is direct re-entry of the machine's own prime output.** 50.5% of composite
  members are fresh semiprimes `q*r` with `r` prime; the rest decompose recursively. The fresh-kill
  census pairs gear bands with cofactor bands far above them (e.g. gear band `<=17^2` x cofactor band
  `<=127^2`), quantifying how far down the machine reaches for its own output.
* **The twin-deciding margin is semiprime re-entry.** Of 962,085 dead pairs, 337,017 were "one-away" -
  exactly one composite member. The lone killer's band distribution: `{5,7}` 45.9%, `{11..23}` 23.2%,
  then a fat tail with ~15% from bands `<=11^2` upward - and from band `<=13^2` on, the lone-killing
  member is a semiprime in essentially every case (from `<=19^2` on: every case without exception).
  So at the margin that decides twinhood, high-band killers act *exclusively* through fresh semiprime
  re-entry - the "coprime blocking" mechanism, and exactly the fragile, invisible-from-below cases of
  section 23b.

## Reading, honestly

The matrix quantifies the cascade; it does not bound anything. The counts are consistent with (and
implied by) the exact `E_q = 0` ledger. Its value is locational: volume pressure on twin candidates
comes from the small gears, whose lattices any window sees in full; the *deciding* pressure at the
margin comes from semiprime re-entry of primes from far-away bands, which is precisely the
bounded-depth-invisible mechanism. Any by-construction argument for the persistence of twins must
therefore control the placement of the products `q*r` against the 1,5 slots - a bilinear statement
about pairs of machine outputs, not a per-gear statement. This sharpens where the open bound lives;
it does not move it.

# Location rules for the gap: the candidates and their failures (manager, 2026-09-11)

Owner's direction: "we need location rules ... the composites are located offset to their
prime / square factors." The composites' location rule is exact and on record
(origin_mechanic.md D3: every composite n = g . m = g^2 + g (m - g), g its least prime factor,
m open under the gears below g; each gear's new strikes begin at its square, kernel
OneStepE.new_iff). This note tests location rules for the GAP: a single column of a section,
named from the primes at or below the cut, that is a twin in every section.

Script: research/stack/r8/location_rules.py (validated: the first-twin offsets L_1(p) at
p = 11, 13, 17 reproduce first_realisation.md's ratios 0.375, 0.100, 0.333 exactly).

## Setup

Column k = the slot (6k - 1, 6k + 1). For consecutive primes p < p' the finer section is the
columns k with p^2 <= 6k - 1 and 6k + 1 < p'^2 (a rule for the finer sections is a rule for
the construction's sections, which are unions of them). A candidate names one column of the
section from p and p' alone; it succeeds when that column is a twin. Candidates: the first
column after the square (sq1); the first column after the square at an offset blind to the
small gears 5, 7, 11, 13 below p (sqblind; blind at offset i iff -6i and 2 - 6i are both
non-squares mod g); the section's middle column (mid) and the column of the midpoint of the
squares (midsq); the columns holding p p' - 2 and p p' + 2 (prod_lo, prod_hi: the mirror
axis of the two gears at radius 1); their neighbours (mirror_lo, mirror_hi); and the previous
section's first-twin offset reused (twinoff).

## Results, N = 10^6 (165 sections; N = 10^5 with 62 sections gives the same first failures)

| candidate | successes | fraction | first failure (p, p', column) |
|---|---|---|---|
| sq1 | 1 | 0.006 | (7, 11, 9) |
| sqblind | 1 | 0.006 | (7, 11, none blind) |
| mid | 4 | 0.024 | (5, 7, 6) |
| midsq | 4 | 0.024 | (5, 7, 6) |
| prod_lo | 8 | 0.049 | (13, 17, 36) |
| prod_hi | 5 | 0.030 | (5, 7, 6) |
| mirror_lo | 9 | 0.055 | (5, 7, 4) |
| mirror_hi | 10 | 0.061 | (7, 11, 14) |
| twinoff | 17 | 0.103 | (7, 11, 9) |

The first-twin offset above the square, L_1(p), is blind to the small gears in 1 of 165
sections. L_1 at p = 5 .. 53: 1, 2, 3, 2, 4, 10, 7, 3, 10, 10, 3, 4, 5, 27.

## Reading

No candidate is a location rule: every one fails at a section with p <= 13, and the success
fractions (0.6% to 10%) are the twin density at these scales, i.e. what a column chosen with
no information about the section would give. The best, reusing the previous section's offset,
is a guess informed by one section back and fails at the second section tested. The
containment idea (the first twin sits at a small-gear-blind offset) is refuted at 164 of 165.

The obstruction, stated once: a column named from the primes at or below p is a fixed
position in the section, and the section's strikers include every gear up to p, each of
whose dilates lands at positions set by residues of the named column modulo that gear; a
name that does not depend on those residues cannot avoid them, and a name that does is the
CRT computation, i.e. the sieve. This is Part IV.4's demand read as position: no location
rule for the gap exists on the record, and the composites' exact location rule (every
dilate explicit) does not yield one for the complement.

## Verdict

FACT, not a route: the composites are located exactly (D3); the gap is not located by any
rule built from the lower primes' positions that was tested; the fractions are chance.

# The fields construction: every striking part in its own field, proved one at a time (2026-09-11)

Owner's instruction: break each striking part out into a field, prove each field's behaviour
systematically, then deal with their relationships bit by bit. This document is that
programme as a statement list. Every statement carries one of: PROVED (kernel name in
proofs/), PROVABLE (a known method gives it; not yet formalised; the method named),
MEASURED (checked on a stated range, no proof), OPEN (no method on record). The order is the
order of proof: fields first, relations second, the union last.

Definitions (construction). S = the numbers = +-1 mod 6, the fold's survivors; column k = the
slot (6k - 1, 6k + 1); a gear g >= 5 strikes column k iff g divides 6k - 1 or 6k + 1. Field 1 =
the primes >= 5. Field j = the members of S with exactly j prime factors counted with
multiplicity (all >= 5). The square field = the squares of primes >= 5, a sub-field of field 2.
A field HITS column k if one of the column's members lies in it. The overlay = the union of
the fields j >= 2. A section is [c, c') with c' = nextprime(c)^2 (the construction) or
[p^2, p'^2) for consecutive primes (the finer statement).

---

## A. Field 1 (the primes; each gear's home strike)

| # | statement | status |
|---|---|---|
| A1 | field 1 = the primes >= 5 | PROVED `Fields.field_one_iff_prime_ge_five` |
| A2 | field 1 never kills: column k is a twin iff both members are in field 1 | PROVED `Fields.prime_field_never_kills` |
| A3 | the gears of machine k are exactly field 1 in section k (the hand-up) | PROVED `MachineStack.stack_eq_primesLE` |
| A4 | field 1 has members in both classes +-1 mod 6 in every section [x, 2x), x >= 7 | PROVABLE (Breusch 1932; Bertrand for the progressions mod 6; mathlib has Bertrand, not the mod-6 form) |
| A5 | inside a section [p^2, p'^2) a member of S is in field 1 iff no gear g <= p strikes it (the square-root rule) | PROVED `OneStepE.blocked_iff_sqrt`, `CoreLeftover.twin_of_rough` |

## B. The square field

| # | statement | status |
|---|---|---|
| B1 | squares are right members only (a square is = 1 mod 6) | PROVED `Fields.square_right_only` |
| B2 | each prime p >= 5 hits exactly one column, k = (p^2 - 1)/6 | PROVED `Fields.square_column_unique`, `square_column_eq_W` |
| B3 | the square field cannot cover a stretch: the number of columns it hits in [a, a + l) is at most the number of primes whose squares land there | PROVED `Fields.squares_in_section_le`, `squares_cannot_cover` |
| B4 | in the construction's section [p_k^2, p_{k+1}^2) the squares present are those of the primes in [p_k, p_{k+1}), fewer than the section's columns | PROVED `Fields.squares_cannot_cover_W` |
| B5 | the square field is empty in every finer section and equals the cuts of the construction | PROVED `Fields.squareField_empty_between`, `squareField_empty_between'`, `squareField_cuts` (FieldsB) |
| B6 | a gear's first new strike is its square: below p^2 every strike of gear p is an echo of a smaller gear; at p^2 it is new | PROVED `OneStepE.new_iff` |

## C. Field 2 (products of two primes)

| # | statement | status |
|---|---|---|
| C1 | field 2 = the union over primes p >= 5 of p . field 1 | PROVED `Fields.field_succ_eq_union` (j = 1) |
| C2 | class rule: p q = 1 mod 6 iff p, q in the same class; field 2 hits right members by same-class pairs and left members by cross-class pairs | PROVED `Fields.class_rule_two`, `class_rule_field` |
| C3 | mirror law: g (6m - 1) and g (6m + 1) are symmetric about 6 g m; in columns k_1 + k_2 = 2 g m up to the sign cases | PROVED `Fields.mirror_law`, `mirror_law_columns`, `mirror_dilate` |
| C4 | the range rule: gear g's field-2 strikes in [x, y) are g times the primes in [x/g, y/g); an interval is free of field 2 iff for every prime g <= sqrt(y) the interval divided by g is free of field 1 | PROVED `Fields.field_dilate` (membership); the interval form is its restatement; validated 0 mismatches on [121, 16129) (research/stack/r8/range_rule.py) |
| C5 | in a section [p^2, p'^2) (or below t^3 for the core t), every member of field 2 is g . m with g <= sqrt(n) a gear and m prime (the two-prime lemma's second case) | PROVED `CoreLeftover.primeOrSemiprime_of_rough_lt_cube` for the t-rough members |
| C6 | field 2 alone cannot cover a section: some column has neither member in field 2 | PROVABLE by count (Landau: semiprimes have density (log log x)/log x among integers; among S the share is below 1/2 from a computable x on); no elementary proof on record; not formalised |
| C7 | the symmetric pairs of field 2 about 6 g m are exactly the pairs {6m - i, 6m + i} both in field 1, the twin being radius 1 | PROVED (fields.md E4, the mirror law read backwards); MEASURED 0 mismatches over 171,243 axes |

## D. Fields 3, 4, 5, 6, each on its own (census research/stack/r8/field_census.py, section [121, 16129): 5,336 members of S, 2,667 columns, gears to 113)

The general statements shared by every field j >= 3 are in D-all below; first each field
separately, with its own measured facts on the section (second measurement on the base-5
section [841, 727609) in research/stack/r8/results_field_census_b5s3.txt).

### Field 3 (products of three primes)

| fact | value on [121, 16129) |
|---|---|
| size, share of S | 919, 17.2% |
| class split left / right | 469 / 450 (nearly even; the class rule: left iff an odd count of class -1 factors) |
| first and last member | 125 = 5^3; 16115 |
| least factor: largest present | 23 (confinement: 23^3 = 12167 < 16129 < 29^3) |
| least-factor profile | 5: 473 (51.5%), 7: 245, 11: 105, 13: 58, 17: 25; gears <= 13 carry 95.9% |
| longest field-3-free range of S | 29 members, 1333 .. 1417 |
| columns hit: left only / right only / both | 408 / 389 / 61 |
| columns where field 3 is the only striking field | 417 of 2,667 |
| mirror pairs about 6 . 5 . m and 6 . 7 . m | 93 and 63 |

Field 3's own shape: it is carried by the gear 5 for half its members and by the gears up to
13 for 96%, so on a section below p'^2 it is the dilate 5 . field 2 with a thin remainder;
its free ranges are long (29 members against field 2's 12) and its exclusive kills are 417
columns, 16% of the section: without field 3 those 417 columns would be twins.

### Field 4 (products of four primes)

| fact | value on [121, 16129) |
|---|---|
| size, share of S | 161, 3.0% |
| class split left / right | 77 / 84 |
| first and last member | 625 = 5^4; 16121 |
| least factor: largest present | 11 (11^4 = 14641 < 16129 < 13^4) |
| least-factor profile | 5: 130 (80.7%), 7: 30, 11: 1; gears <= 13 carry 100% |
| longest field-4-free range of S | 168 members, 121 .. 623 (the field is empty below 5^4) |
| columns hit: left only / right only / both | 77 / 84 / 0 (no column has both members in field 4) |
| columns where field 4 is the only striking field | 81 |
| mirror pairs about 6 . 5 . m and 6 . 7 . m | 2 and 1 |

Field 4's own shape: four-fifths of it is 5 . field 3; it never hits both members of a
column on this section; it kills 81 columns on its own.

### Field 5 (products of five primes)

| fact | value on [121, 16129) |
|---|---|
| size, share of S | 16, 0.3% |
| class split | 9 / 7 |
| first and last member | 3125 = 5^5; 15925 = 5^2 . 7^2 . 13 |
| least factor | 5 for all 16 (5^5 = 3125 < 16129 < 7^5 = 16807) |
| longest field-5-free range of S | 1,001 members, 121 .. 3121 |
| columns hit | 9 left, 7 right, 0 both; 8 exclusive kills |

Field 5's own shape: entirely 5 . field 4 on this section; 8 exclusive kills.

### Field 6

One member, 15625 = 5^6, a right member; no exclusive kill (its column's left member 15623
is composite). Field 7 and above are empty below 5^7 = 78125 > 16129.

### Second measurement: each field on the base-5 section [841, 727609) (242,256 members of S, 121,127 columns, gears to 839)

| field | size (share) | left / right | first member | largest least factor | gear-5 share | longest free range | both members | exclusive kills |
|---|---|---|---|---|---|---|---|---|
| prime | 58,462 (24.1%) | 29,275 / 29,187 | 853 | - | - | 37 | 6,224 twins | 6,224 |
| semiprime | 102,377 (42.3%) | 51,106 / 51,271 | 841 = 29^2 | 839 | 13.1% | 19 | 21,359 | 45,172 |
| three-factor | 60,188 (24.8%) | 30,168 / 30,020 | 845 = 5 . 13^2 | 89 (89^3 < 727609 < 97^3) | 35.2% | 29 (1333 .. 1417 again) | 6,933 | 22,746 |
| four-factor | 17,560 (7.2%) | 8,731 / 8,829 | 875 = 5^3 . 7 | 29 (29^4 < 727609 < 31^4) | 61.2% | 130 | 335 | 5,544 |
| five-factor | 3,201 (1.3%) | 1,619 / 1,582 | 3125 = 5^5 | 13 | 82.1% | 761 | 5 | 1,019 |
| six-factor | 426 (0.2%) | 207 / 219 | 15625 = 5^6 | 7 | 93.9% | 4,928 | 0 | 148 |
| seven-factor | 40 | 21 / 19 | 78125 = 5^7 | 5 | 100% | 25,761 | 0 | 17 |

Read against the base-3 section: every field's largest least factor is exactly the largest
prime g with g^j below the section's top (D4 holds with equality in the sense that the bound
is attained: 89 for three factors, 29 for four, 13 for five, 7 for six, 5 for seven); the
class split of every field is even to within 1%; the gear-5 share falls with the section's
size within a field and rises with the field index; the three-factor field's longest free
range is the same interval 1333 .. 1417 on both sections (it lies in both); exclusive kills
in order 45,172 / 22,746 / 5,544 / 1,019 / 148 / 17 (37%, 19%, 4.6%, 0.8%, 0.1%, 0.01% of
columns), and 6,224 twins.

### D-all. What every field j >= 3 satisfies

| # | statement | status |
|---|---|---|
| D1 | field j = the union over primes p >= 5 of p . field j-1; field j ∩ 5S = 5 . field j-1 | PROVED `Fields.field_succ_eq_union`, `field_inter_five` |
| D2 | class rule: a member is = 1 mod 6 iff its count of class -1 factors is even | PROVED `Fields.class_rule` |
| D3 | field j is empty below 5^j, and 5^j is its first member | PROVED `Fields.field_empty_below`, `five_pow_mem_field` (FieldsB) |
| D4 | confinement: n in field j has minFac(n)^j <= n, so below P^2 its least factor g satisfies g^j < P^2 (field j hits only the teeth of the gears below P^{2/j}) | PROVED `Fields.field_least_factor_le`, `field_confined` (FieldsB); MEASURED 0 exceptions in 20 sections |
| D5 | the deepest field present below P^2 has index j with 5^j < P^2 | PROVED `Fields.field_index_le` (FieldsB); MEASURED exact (equality) at p = 11..53 |
| D6 | field j alone cannot cover a section | PROVABLE by count (Landau: density (log log x)^{j-1}/((j-1)! log x)); not formalised |

## D-alone. Each factor-count field in isolation (research/stack/r8/count_field_alone.py, section [121, 16129), gaps in S-steps)

| field | members | mean gap | commonest gaps (count) | largest gap, after | same-class : opposite-class consecutive | longest one-class run | members per tenth of the section | divisible by 5 | by 7 |
|---|---|---|---|---|---|---|---|---|---|
| prime | 1,847 | 2.89 | 1 (566), 2 (431), 3 (341), 4 (164), 5 (145) | 15 after 15683 | 703 : 1,143 (0.62) | 6 | 238, 199, 190, 186, 179, 177, 170, 171, 160, 177 | 0 | 0 |
| semiprime | 2,392 | 2.23 | 1 (1,052), 2 (595), 3 (339), 4 (185), 5 (102) | 13 after 15253 | 857 : 1,534 (0.56) | 8 | 232, 246, 241, 240, 243, 234, 244, 236, 254, 222 | 18.7% | 14.0% |
| three-factor | 919 | 5.81 | 3 (135), 1 (114), 2 (108), 7 (97), 4 (82) | 30 after 1331 | 384 : 534 (0.72) | 7 | 58, 77, 89, 91, 92, 105, 95, 106, 95, 111 | 51.5% | 36.0% |
| four-factor | 161 | 32.3 | 33 (13), 17 (11), 7 (10), 50 (9), 20 (9) | 117 after 875 | 63 : 97 (0.65) | 5 | 6, 10, 13, 16, 18, 15, 23, 18, 23, 19 | 80.7% | 54.0% |
| five-factor | 16 | 284 | 417, 250, 167 (2 each) | 583 after 4375 | 6 : 9 | 3 | 0, 1, 1, 1, 1, 3, 1, 3, 2, 3 | 100% | 56.2% |

What each field alone shows: the prime field thins along the section (238 to 160-177 per
tenth) while the semiprime field is flat (222-254) and the three- and four-factor fields
thicken (58 to 111; 6 to 19), the fields trading density along the section exactly as the
count of prime factors of a number grows with its size; the semiprime field's gaps are the
shortest (mean 2.23 S-steps, 44% of consecutive pairs adjacent); every field prefers
opposite-class consecutive members (ratios 0.56-0.72 against 1.0 for no preference) and no
field runs longer than 8 members in one class; the three-factor field's commonest gap is 3
S-steps, not 1, a spacing the semiprime field does not have; the share divisible by 5 rises
with the field index (0, 18.7, 51.5, 80.7, 100%), which is the nesting field j ⊇ 5 . field
j-1 growing to fill the field. None of these is a periodic structure (E6); the gear fields
(G-alone) are where the periodic structure lives.

## G. The second kind of field: one per gear, starting at its square (research/stack/r8/gear_fields.py)

Owner's instruction: a field type for each gear and its composites, starting with its own
square. Gear field of g = the members of S whose least prime factor is g: n = g . m with m >= g
a survivor of the gears below g (the gear's own strikes from its square on; multiples that a
smaller gear already struck belong to that smaller gear's field). The square g^2 is the first
member of every gear field; the square field is the set of these first members. The gear
fields partition the composites of S (one least factor each), as the factor-count fields do.

Section [121, 16129), 5,336 members of S, 2,667 columns, 29 gears from 5 to 127. Square field:
26 members present (11^2 = 121 to 127^2 = 16129 excluded, so 121 .. 113^2), all right members,
11 exclusive kills (the columns whose left member g^2 - 2 is prime; kernel OneStepE.new_iff).

| gear field | size | share of composites | square in section | left / right | m prime | longest free run of S | exclusive kills |
|---|---|---|---|---|---|---|---|
| 5 | 1,067 | 30.6% | no (25 below) | 534 / 533 | 447 | 6 | 465 |
| 7 | 610 | 17.5% | no (49 below) | 305 / 305 | 335 | 13 | 232 |
| 11 | 334 | 9.6% | yes | 167 / 167 | 228 | 36 | 116 |
| 13 | 256 | 7.3% | yes | 128 / 128 | 198 | 60 | 87 |
| 17 | 180 | 5.2% | yes | 89 / 91 | 155 | 78 | 57 |
| 19 | 150 | 4.3% | yes | 75 / 75 | 139 | 88 | 51 |
| 23 | 120 | 3.4% | yes | 58 / 62 | 118 | 136 | 38 |
| 29 | 92 | 2.6% | yes | 45 / 47 | 92 | 240 | 31 |
| 31 | 87 | 2.5% | yes | 45 / 42 | 87 | 280 | 29 |
| 37 | 73 | 2.1% | yes | 37 / 36 | 73 | 416 | 22 |
| 41 | 65 | 1.9% | yes | 31 / 34 | 65 | 520 | 21 |
| 43 | 61 | 1.7% | yes | 31 / 30 | 61 | 576 | 21 |
| gears 47 .. 113 | 394 | 11.3% | yes | | all m prime | | |

Facts per gear field, exact: (G1) the class split is even to within one member because the
member g . m has the class of g times the class of m and m runs over the survivors, which
alternate; (G2) from gear 29 on every member has m prime (m >= g and m < 16129 / g < g^2 means
m has no factor below g and is below g^2, so m is prime: the two-prime lemma at the gear's own
scale); (G3) the gear field's free runs on S are g times the free runs of R_g, the open set of
the gears below g, so they grow with g (6, 13, 36, 60, 78, 88, 136, 240, ...); (G4) exclusive
kills fall roughly as the field's size (465, 232, 116, 87, 57, ...), gear 5 alone accounting
for 17% of the section's columns (a relation between fields, belongs to part E). The gear fields are the dilation form D3 read as fields:
gear g's field is g . R_g exactly.

### G-alone. Each gear field in isolation, on the whole line (research/stack/r8/gear_field_alone.py)

Gear field of g on the line: F_g = { g m : m in S, m >= g, no prime below g divides m }, the
gear's dilate of the lower machine's survivors. Measured over two full periods for g = 5..23,
each claim checked exactly:

| gear field | period in columns | hits per period | mirror | largest gap (columns) | the three main gaps and their counts |
|---|---|---|---|---|---|
| 5 | 5 | 2 | r hit iff -r hit | 3 | residues 1, 4 mod 5 |
| 7 | 35 | 8 | yes | 7 | 2 (3), 5 (2), 7 (2); residues 1, 8, 13, 15, 20, 22, 27, 34 mod 35 |
| 11 | 385 | 48 | yes | 18 | 4 (14), 7 (15), 11 (14); then 15 (2), 18 (2) |
| 13 | 5,005 | 480 | yes | 30 | 4 (135), 9 (134), 13 (142); then 17, 22, 26, 30 |
| 17 | 85,085 | 5,760 | yes | 62 | 6 (1,484), 11 (1,485), 17 (1,690); then 23 .. 62 |
| 19 | 1,616,615 | 92,160 | yes | 82 | 6, 13, 19 (22,275 / 22,274 / 26,630); then 25 .. 82 |
| 23 | 37,182,145 | 1,658,880 | yes | 130 | 8, 15, 23 (378,675 / 378,675 / 470,629); then 31 .. 130 |

Exact facts of a gear field alone, each verified above and PROVED in the kernel (proofs/FieldsC.lean,
round 44: Survivor g m, gearField g = { g m : g <= m, Survivor g m }; sq_mem_gearField G0;
gearField_minFac and mem_gearField_of_minFac G1, the least-factor partition; gearField_disjoint
G2; survivor_periodic, gearField_periodic G3 = (G5) below; survivor_neg G4 = (G7); card_survivors_period
= (G6), via Survivor g m <-> Coprime (6P) m and the totient; gearField_gap_scaling = (G8), the gap is
g times the survivor gap):
(G5) F_g is periodic in columns with period the product of the gears 5 .. g (the dilate by g
of the lower machine's period); (G6) hits per period = 2 . product over 5 <= h < g of (h - 1)
(m is a single number, so each lower gear removes one residue in h; two classes of the fold);
(G7) mirror symmetry about column 0: r is hit iff -r is hit (the survivors below g are
symmetric under m -> -m); (G8) the gap spectrum is the lower survivors' gap spectrum in
numbers (2, 4, 6 and their sums) scaled by g / 6: the three main gaps are the images of 2, 4,
6, with the 6-image most frequent, and the largest gap is g / 6 times the lower machine's
largest gap between survivors (3, 7, 18, 30, 62, 82, 130 columns for g = 5 .. 23). So a gear
field in isolation is completely described: it is the single-tooth wheel of the gears below g
(the manifold's objects, closed forms on record: period, census, symmetry) carried to the
line by multiplication by g, and it holds no information beyond that wheel. Its first member
is g^2, and every other member is g times a survivor of the lower machine that is at least g.

## H. The third kind of field: the composites of a gear and the gears below it (research/stack/r8/smooth_fields.py)

Owner's instruction (2026-09-12): a field containing just the composites of the gear and the
lower gears. Two readings, both built: the smooth field C_g = the composites of S whose prime
factors all lie in {5, ..., g} (cumulative: the machine {5..g} composing only with itself), and
its layer L_g = the members of C_g whose largest prime factor is g (new at g): L_g = g times
the members of S that are g-smooth and at least 5, so C_g is the disjoint union of the layers
L_5, L_7, ..., L_g. Section [121, 16129):

| gear g | smooth field C_g | layer L_g | first of L_g | last | L_g below g^2 / at or above | left / right | largest gap (S-steps) | g^a alone / g x prime / g x composite |
|---|---|---|---|---|---|---|---|---|
| 5 | 4 | 4 | 125 = 5^3 | 15625 | 0 / 4 | 2 / 2 | 4,167 | 4 / 0 / 0 |
| 7 | 15 | 11 | 175 = 5^2 . 7 | 12005 | 0 / 11 | 5 / 6 | 1,143 | 2 / 0 / 9 |
| 11 | 36 | 21 | 121 = 11^2 | 15125 | 0 / 21 | 11 / 10 | 1,283 | 3 / 0 / 18 |
| 13 | 67 | 31 | 143 = 11 . 13 | 15925 | 1 / 30 | 14 / 17 | 789 | 2 / 1 / 28 |
| 17 | 105 | 38 | 187 = 11 . 17 | 15895 | 2 / 36 | 19 / 19 | 669 | 2 / 2 / 34 |
| 19 | 151 | 46 | 133 = 7 . 19 | 16093 | 4 / 42 | 21 / 25 | 747 | 2 / 4 / 40 |
| 23 | 202 | 51 | 161 = 7 . 23 | 15295 | 5 / 46 | 27 / 24 | 414 | 2 / 5 / 44 |
| 29 | 255 | 53 | 145 = 5 . 29 | 15979 | 8 / 45 | 27 / 26 | 348 | 1 / 7 / 45 |
| 31 | 310 | 55 | 155 = 5 . 31 | 15283 | 9 / 46 | 26 / 29 | 248 | 1 / 8 / 46 |
| 37 | 365 | 55 | 185 = 5 . 37 | 15725 | 11 / 44 | 26 / 29 | 296 | 1 / 9 / 45 |

What this field is in isolation, exact: the smooth field C_g is the image of the exponent
lattice {(a_5, a_7, ..., a_g) : sum >= 2} under (a) -> 5^{a_5} 7^{a_7} ... g^{a_g}, restricted
to S (every such product is in S automatically, since each factor is +-1 mod 6); its class is
the parity of the count of class -1 factors (D2); it is closed under multiplication and is the
multiplicative semigroup of the machine {5..g} minus its generators; its members thin as a
power of log x (the count of g-smooth numbers up to x is a polynomial in log x of degree
pi(g) - 2 for fixed g: Ennola / Dickman, register, KNOWN); its gaps grow without bound, so it
never covers a section on its own and never comes close (4 to 365 members of 5,336 here). The
layer L_g in isolation: its first member is the smallest product g . m with m a g-smooth
survivor (5g when 5g >= the section's start, else g^2 or a larger smooth multiple), members
below g^2 are echoes of the lower gears (0 at g = 5, 7, 11; 11 of 55 at g = 37), the kinds
split into g^a alone (1-4), g x prime (0-9) and g x smooth composite (the majority), and
the class split is even to within 3. Relation to the other kinds, for part E: L_g is the
part of gear g's field (part G) whose cofactor m is g-smooth, and C_g is the struck set of
the machine {5..g} restricted to the numbers with no prime factor above g, so the overlay
minus the union of the C_g is exactly the strikes whose cofactor carries a prime above g.

## I. The fourth kind: each gear's composites of itself and only higher gears

Owner's instruction (2026-09-12). Two readings, both exact: "itself and higher" (the cofactor
m has every prime factor >= g, g itself allowed) is exactly the gear field of part G (least
factor g); "strictly higher" (every prime factor of m above g, so g divides n exactly once) is
the gear field minus the multiples of g^2. Both are closed: the strict field is g times the
survivors of the wheel {5..g} (all gears up to and including g), periodic in columns with
period g times the product of the primes in [5, g], with 2 . prod_{5 <= h <= g} (h - 1) hits per
period (checked: gear 7, period 245, 48 per period; gear 11, period 4,235, 480 per period; both
periodic over two periods), mirror-symmetric, gaps g times the wheel's.

This kind completes a three-way split of a gear's multiples on S, exact and verified on
[121, 16129) (the columns add up at every gear):

| gear g | multiples of g | itself + higher (gear field, part G) | of which strictly higher (g exactly once) | of which pure powers g^a | of which g^2 x higher | itself + lower only (smooth layer, part H) | mixed, lower and higher (echoes) |
|---|---|---|---|---|---|---|---|
| 5 | 1,067 | 1,067 | 853 | 4 | 210 | 4 | 0 |
| 7 | 762 | 610 | 523 | 2 | 85 | 11 | 143 |
| 11 | 486 | 334 | 304 | 3 | 27 | 21 | 134 |
| 13 | 410 | 256 | 236 | 2 | 18 | 31 | 125 |
| 17 | 313 | 180 | 169 | 2 | 9 | 38 | 97 |
| 19 | 281 | 150 | 142 | 2 | 6 | 46 | 87 |
| 23 | 232 | 120 | 117 | 2 | 1 | 51 | 63 |
| 29 | 184 | 92 | 91 | 1 | 0 | 53 | 40 |
| 31 | 172 | 87 | 86 | 1 | 0 | 55 | 31 |
| 37 | 144 | 73 | 72 | 1 | 0 | 55 | 17 |

(The pure powers g^a are counted in both "itself + higher" and "itself + lower", so the
columns sum to the multiples once the powers are counted once.) Reading per gear: the new
strikes of gear g (part G) are almost all g times a single higher prime or a higher survivor;
the echoes (a lower factor and a higher factor) are the multiples of g that a smaller gear
already struck and grow as a share with g (0% at 5, 19% at 7, 28% at 11, 31% at 17, 12% at 37
on this section); the lower-only layer (part H) is the sparse smooth part. In the machine's
terms: what gear g adds to the struck set when it is added to the wheel is exactly its part-G
field (its strict part plus its powers times higher survivors), and everything else it strikes
was already struck.

## J. A machine below its next square, across all the fields (research/stack/r8/machine_map.py)

Owner's instruction (2026-09-12): start with the machine {5}, look below 25 across all the
fields, then bigger machines, and see which fields locate the twins. Below the next prime's
square g'^2 every composite member has least factor <= g, so every kill is by the gear field of
a gear of the machine; the square field kills one column per gear (its square column) and the
factor-count fields split the kills by depth.

Machine {5}, columns below 49 (7 columns): twins at k = 1, 2, 3, 5, 7 (the pairs (5,7),
(11,13), (17,19), (29,31), (41,43)); column 4 killed by 25 = 5^2 (gear field 5, square field,
field 2); column 6 killed by 35 = 5 . 7 (gear field 5, field 2).

Machine {5, 7}, columns below 121 (19 columns): twins at k = 1, 2, 3, 5, 7, 10, 12, 17, 18;
kills: gear field 5 at columns 4, 6, 9, 11, 14, 16, 19 (25, 35, 55, 65, 85, 95, 115), gear field
7 at 8, 13, 15 (49, 77, 91); the squares 25, 49; every kill in field 2.

| machine {5..g} | columns below g'^2 | twins | member-kills by gear field (least factor) | by factor-count field | by the square field |
|---|---|---|---|---|---|
| {5} | 7 | 5 | 5: 2 | F2: 2 | 1 |
| {5, 7} | 19 | 9 | 5: 7, 7: 3 | F2: 10 | 2 |
| {5..11} | 27 | 11 | 5: 10, 7: 6, 11: 2 | F2: 17, F3: 1 (125) | 3 |
| {5..13} | 47 | 18 | 5: 18, 7: 9, 11: 5, 13: 3 | F2: 31, F3: 4 | 4 |
| {5..17} | 59 | 20 | 5: 23, 7: 13, 11: 7, 13: 4, 17: 2 | F2: 43, F3: 6 | 5 |
| {5..19} | 87 | 24 | 5: 34, 7: 19, 11: 11, 13: 7, 17: 4, 19: 2 | F2: 67, F3: 10 | 6 |
| {5..23} | 139 | 32 | 5: 55, 7: 31, 11: 17, 13: 13, 17: 9, 19: 7, 23: 3 | F2: 113, F3: 21, F4: 1 (625) | 7 |

What locates the twins below g'^2, exactly: the twins are the columns avoiding every gear
field's residues (k not = +-6^-1 mod h for every h <= g: the wheel's open columns, a CRT set)
TOGETHER WITH the home columns of the machine's own twin gears (k = 1 for (5, 7), k = 2 for
(11, 13), k = 3 for (17, 19): a gear strikes its own column only at itself, and it is prime).
Checked at every machine 5..23: the wheel's open columns plus the home columns = the twins,
with no exception. So below the next square the twin locator is CLOSED: the CRT complement of
the gear fields' residues within the range, plus the twin gears themselves. The square field
kills exactly one column per gear of the machine (its square column, always a right member);
field 2 carries nearly every kill (113 of 135 at {5..23}); field 3 enters at 125 = 5^3 (machine
11), field 4 at 625 = 5^4 (machine 23), each first at a power of 5 (D3).

The kills by gear field fall with the gear (55, 31, 17, 13, 9, 7, 3 at {5..23}): the field of
gear 5 does 41% of the killing, the five smallest gears 92%. This is the location rule for
the twins in every machine's range below its next square, and it is the sieve: what step 8
needs is the same rule on the section [p_k^2, p_{k+1}^2), i.e. the CRT complement's members in
a stretch far shorter than the wheel's period, which the CRT does not place.

## K. Unwinding one gear at a time: how each field advances into the new range (research/stack/r8/unwind_steps.py)

Owner's instruction (2026-09-12): not the imprint (CRT, sieve) but the mechanism: pull the
machine apart one step at a time, see how each field advances, how the fields' elements shift
relative to each other, as numbers, as offsets from an origin. Step g -> g' (consecutive
gears): the new range is (g^2, g'^2); the ORIGIN is the square g^2, column a = (g^2 - 1)/6;
offsets are counted in columns from a.

Step 7 -> 11, origin 49 (column 8), 11 columns:
+1 (53, 55): 55 = 5 x 11 | +2 (59, 61) TWIN | +3 (65, 67): 65 = 5 x 13 | +4 (71, 73) TWIN |
+5 (77, 79): 77 = 7 x 11 | +6 (83, 85): 85 = 5 x 17 | +7 (89, 91): 91 = 7 x 13 | +8 (95, 97):
95 = 5 x 19 | +9 (101, 103) TWIN | +10 (107, 109) TWIN | +11 (113, 115): 115 = 5 x 23.
Every kill is a lower prime (11, 13, 17, 19, 23, all from the range [3^2, 5^2)) carried up
by 5 or 7. Twins at offsets 2, 4, 9, 10.

Step 13 -> 17, origin 169 (column 28), 19 columns: kills 175 = 5 x 35, 185 = 5 x 37, 187 =
11 x 17, 203 = 7 x 29, 205 = 5 x 41, 209 = 11 x 19, 215 = 5 x 43, 217 = 7 x 31, 221 = 13 x 17,
235 = 5 x 47, 245 = 5 x 49, 247 = 13 x 19, 253 = 11 x 23, 259 = 7 x 37, 265 = 5 x 53, 275 =
5 x 55; twins at offsets 2, 4, 5, 10, 12, 17, 19. The cofactors come from three earlier ranges
([3^2, 5^2): 5 of them, [5^2, 7^2): 8, [7^2, 11^2): 3), and 14 of the 16 are primes.

| step | new range | columns | twins at offsets from the origin | kills per gear | cofactors' source ranges | first strike of each gear field (offset, member) |
|---|---|---|---|---|---|---|
| 5 -> 7 | (25, 49) | 3 | 1, 3 | 5: 1 | [2^2, 3^2): 1 | 5: (2, 35) |
| 7 -> 11 | (49, 121) | 11 | 2, 4, 9, 10 | 5: 5, 7: 2 | [3^2, 5^2): 7 | 5: (1, 55), 7: (5, 77) |
| 11 -> 13 | (121, 169) | 7 | 3, 5 | 5: 3, 7: 2, 11: 1 | [3^2, 5^2): 3, [5^2, 7^2): 3 | 5: (1, 125), 7: (2, 133), 11: (4, 143) |
| 13 -> 17 | (169, 289) | 19 | 2, 4, 5, 10, 12, 17, 19 | 5: 8, 7: 3, 11: 3, 13: 2 | 5 / 8 / 3 from the three ranges | 5: (1, 175), 7: (6, 203), 11: (3, 187), 13: (9, 221) |
| 17 -> 19 | (289, 361) | 11 | 4, 10 | 5: 5, 7: 3, 11: 2, 13: 1, 17: 1 | 2 / 4 / 6 | 5: (1, 295), 7: (2, 301), 11: (5, 319), 13: (2, 299), 17: (6, 323) |
| 19 -> 23 | (361, 529) | 27 | 10, 12, 17, 27 | 5: 11, 7: 6, 11: 4, 13: 3, 17: 2, 19: 1 | 2 / 8 / 17 | 5: (1, 365), 7: (2, 371), 11: (8, 407), 13: (3, 377), 17: (5, 391), 19: (13, 437) |

THE MECHANISM, as numbers. (K1) Every kill in the new range is an element of an earlier range
carried up by one gear: n = h . m with m in (g^2 / h, g'^2 / h), an earlier range; below h^3
the cofactor is a lower prime (14 of 16 at the step 13 -> 17), so the new range's kill pattern
is the superposition, over the gears h <= g, of the earlier prime patterns scaled by h and
shifted to start at g^2 / h. (K2) The newest gear enters its own new range at g . g' (35, 77,
143, 221, 323, 437), at offset (g g' - g^2)/6 = g (g' - g)/6 from the origin: its gap, times
itself, in sixths. Gear 5 enters at offset 1 at every step from 11 on (5 times the survivor
just above g^2 / 5). (K3) The offsets of every gear's strikes from the origin are fixed by the
origin's residues: gear h strikes offset i iff g^2 + 6i = 0 or 2 (mod h) [kernel
SquareColumn.offset_strike], so gear h's copy in the new range is its two arithmetic
progressions in i with phases set by g^2 mod h. The origin is a square, so the phase of every
lower gear's copy is the SQUARE of g's own residue: phase_h = (g mod h)^2 mod h. In machine
words: adding gear g places every lower gear's field in the new range at the squared phase of
where g sits in that gear's wheel; the new range's twins are the lower wheel's openings read
at the phase vector ((g mod h)^2)_h over a stretch of (g'^2 - g^2)/6 columns. (K4) The twins'
offsets from the origin (1, 3 | 2, 4, 9, 10 | 3, 5 | 2, 4, 5, 10, 12, 17, 19 | 4, 10 | 10, 12,
17, 27) are the offsets i at which no gear's progression lands on either side, i.e. -6i and
2 - 6i are both outside the residue set {(g mod h)^2 - ...}: the pointer to the openings is
the vector of squared residues of the newest gear, and the openings are its blind offsets.

What the unwinding shows that the imprint does not: the new range is not sifted by arbitrary
phases; its phases are the squares of the newest gear's residues, and g's residues are where g
itself sat as an opening of the lower wheels. So the machine's next section is the lower
wheel read at the squared position of the gear the lower wheel just produced: the hand-up in
phase form. The relationship between the fields at a step is therefore: (prime field below g)
-> (g, a member of it) -> (its residue vector, its position in every lower gear field) ->
(squared) -> (the phases of every gear field in the new range) -> (the twins as the blind
offsets of that squared vector). Every arrow is exact and numerical. What is not closed is the
last: which squared vectors leave a blind offset within (g'^2 - g^2)/6 columns; that is step
8 in phase form, and it now reads: the squared residue vector of a prime is never a covering
vector for a stretch as long as its own square gap.

## L. Which squared vectors leave a blind offset inside the new range (research/stack/r8/squared_vectors.py and the exact scans)

Owner's instruction (2026-09-12): find which squared vectors leave a blind offset inside a
stretch as long as g's own square gap. Setting: step g -> g', gears h in [5, g], L = (g'^2 -
g^2)/6; a phase vector phi = (phi_h)_h places the origin; offset i is struck by h iff (phi_h +
6i) mod h in {0, 2}; the vector FAILS if offsets 1..L are all struck (no twin in the new range).

Three kinds of vector, 400 samples each per g, against the real one (g = 5 .. 109):

| kind | leaves a blind offset in 1..L | mean number of blind offsets |
|---|---|---|
| the real vector phi_h = g^2 mod h | always (27 of 27 steps); its count sits inside the random spread at every step (3, 4, 3, 7, 3, 4, 9, 2, 12, 7, 4, 12, 13, 13, 6, 19, 12, 3, 15, 14, 15, 21, 15, 8, 11, 6, 11) | as the random means |
| a random SQUARED vector phi_h = r_h^2, r_h in 1..h-1 | 1.000 at every g except 0.993 at g = 17 and 0.998 at g = 71 | 2.0 .. 28.8 |
| a random free vector phi_h in 0..h-1 | 1.000 except 0.993 at g = 17 and 0.998 at g = 29 | 2.4 .. 29.2 |

Exact, by scanning the whole wheel period (every phase vector once), the FAILING SET = the
origins x such that the columns x+1 .. x+L are all struck, i.e. the origins sitting just
before a struck run of the wheel at least L long:

| g | wheel period | L | wheel record F | failing vectors | of the period | among square origins (6x+1 a square mod every gear) | among square origins with 6x+1 a NONZERO square mod every gear | the real origin fails |
|---|---|---|---|---|---|---|---|---|
| 7 | 35 | 12 | 5 | 0 | 0 | 0 of 12 | 0 | no |
| 11 | 385 | 8 | 7 | 0 | 0 | 0 of 72 | 0 | no |
| 13 | 5,005 | 20 | 11 | 0 | 0 | 0 of 504 | 0 | no |
| 17 | 85,085 | 12 | 18 | 370 | 4.3 x 10^-3 | 14 of 4,536 | 3 | no |
| 19 | 1,616,615 | 28 | 25 | 0 | 0 | 0 of 45,360 | 0 | no |

The answer, exact: a vector fails iff it is the phase of a column just before a struck run of
the wheel of length at least L. When L exceeds the wheel's record (g = 7, 11, 13, 19 here: L =
12 > 5, 8 > 7, 20 > 11, 28 > 25) NO vector fails, squared or not: the record route. When L is
below the record (g = 17: 12 < 18) the failing set is the starts of the wheel's runs of length
>= L, 370 of 85,085 phases, 14 of them square origins, and 3 of those with the origin value a
nonzero square mod every gear, i.e. squares of numbers coprime to the wheel (e.g. 6x + 1 =
280219 with roots n = 158, 1467, 3252, 5163, ... mod 85,085, all coprime to 85,085). So nothing
structural excludes the square of a prime from the failing set: the failing origins include
squares of numbers coprime to every gear, and a prime is one residue class among them. At g =
17 the real origin 17^2 (column 48) is not one of the 3; the one trial the wheel {5..17}
gets is the prime 17 itself, and it passes.

What this settles for the two statements. For the finer statement (a twin between
consecutive prime squares) the failing set is empty whenever L exceeds the record and is a
small set of residue classes otherwise, with the prime one draw per wheel and no structural
exclusion: the property is measured (0 exceptions to 10^7) and its only visible reason is the
size of the failing set, (1 - delta)^L in order, summable over the primes. For step 8 (the
construction's sections, L about q^2/6 with q the top gear) the failing set is empty at every
certified q because F(q) < q^2/6 there (the record route, III.4), and step 8 for all q is
exactly "the failing set stays empty", i.e. F(q) < q^2/6 for every q (IV.1). The unwinding
therefore ends where the length face ended, with one addition: the origin of every section
is a square, and a square origin's phases are the squared residues of its root, but the
failing set contains square origins with roots coprime to every gear, so the squaring is not
what keeps the origin out of it.

## M. Keep looking: what the square origin gives, exactly (research/stack/r8/blind_pointer.py, root_to_square.py)

Two exact structures of the strikes after a square origin, one for the small gears and one
for the top gears, and the measured consequence.

M1. The small gears' gift: blind classes. From a square origin, gear h can strike offset i only
if -6i or 2 - 6i is a square mod h. Offsets where neither is a square are blind to h for EVERY
square origin: for the gears 5 and 7 the blind offsets are i = 5, 10, 12, 17 (mod 35), for
{5, 7, 11} a class of density 3.12%, for {5, 7, 11, 13} 0.96%. The twins of the new ranges sit on
them far above their share (424 steps, primes g in [17, 3000], 53,867 twins):

| blind to | share of offsets | share of the twins | enrichment | twins among the blind offsets below L | steps where no blind offset is a twin |
|---|---|---|---|---|---|
| 5, 7 | 11.4% | 26.9% (14,474) | 2.4 x | 8.4% (14,474 of 171,895) | yes, some (min 0) |
| 5, 7, 11 | 3.1% | 9.0% (4,867) | 2.9 x | 10.4% | yes (min 0) |
| 5, 7, 11, 13 | 0.96% | 3.5% (1,872) | 3.6 x | 12.6% | yes (min 0) |

The first twin of a range is blind to 5 and 7 in 156 of 424 steps (37%, against 11.4% by
share). So the square origin points: the blind classes are where the twins concentrate. But
the pointer is not a guarantee: in some steps no blind-class offset below L is a twin (the
minimum count is 0 at every depth), and 268 of 424 first twins are not blind to 5 and 7. The
twins on non-blind offsets are where the particular phase g^2 mod h happens to miss.

M2. The top gears' comb, exact. For a gear h = g - t (t even, small), g^2 = t^2 mod h, so h's
multiples near the square are h m with m in S just above g^2 / h = g + t + t^2/h: the strikes
above the square are at e = s (g - t) - t^2 for the s = 1, 2, ... with g + t + s in S (two of
every six values of s), i.e. at offsets (s (g - t) - t^2) / 6 from the origin. Checked on 28
(g, gear) pairs at g = 101 .. 2003: the first strike above the square is at the least positive
such e in every case (e = 2g - 2t - t^2 when g + t + 2 is in S, e = 4 (g - t) - t^2 when it is
g + t + 4, and so on). Below the square the identity g^2 - t^2 = (g - t)(g + t) strikes the
offsets -t^2/6 for t = 6, 12, 18, ... (the ones landing in S). So the top gears' strikes near
the square form an explicit comb in t, with first strikes clustered near offset g/3 (s = 2)
and 2g/3 (s = 4); they are sparse (about sqrt(g) of them across the first g/3 offsets) and
carry no density; the density is the small and middle gears'.

M3. The root and its square on one wheel. For the wheel {5..g}, every root r (a number coprime
to the wheel) has a single-tooth gap d(r) to the next root and a two-tooth first blind offset
B(r) after r^2; the finer statement at the prime g is B(g) <= L(g) = d (2g + d)/6. Over full
periods of roots on the wheels 7, 11, 13, 17 (47; 479; 5,759; 92,159 roots) B(r) <= L(r) at
every root, but only because L(r) grows with r (L >= 2r/3) while B is bounded by the wheel's
record (max B = 10, 15, 20 on the wheels 13, 17, 19); the content is confined to the roots
below 1.5 x the record, which are the primes just above g, and B does not depend on d (mean B
by d: 3.7 .. 5.4 across d = 2 .. 28 on the wheel 19, no trend): the single-tooth gap after r
carries no information about the two-tooth gap after r^2.

Where this leaves the why. The strikes after a square are: the small gears at
square-restricted phases (M1, a structured comb with blind classes, 2.4-3.6 x enrichment of
the twins on them), the top gears at polynomial offsets (M2, sparse, explicit), and the
middle gears at phases g^2 mod h with no visible structure; the twin is an offset all three
miss. M1 and M2 are the mechanism the owner asked for in the numbers: offsets from the origin
that point to where the openings concentrate. What they do not give is a guarantee at every
step, and the steps without a blind twin are decided by the middle gears' phases, which are
the residues of one prime modulo the primes between 13 and g.

## E. Relations between fields

| # | statement | status |
|---|---|---|
| E1 | the fields partition the composites of S: every composite lies in exactly one field | PROVED `Fields.field_unique`, `mem_field_length` |
| E2 | the overlay of the fields j >= 2 is exactly the struck set of S (given the square-root rule inside a section) | PROVED `Fields.overlay_iff_composite`, `blocked_iff_hits_overlay` |
| E3 | column k is a twin iff no field j >= 2 hits it | PROVED `Fields.twin_iff_not_hits_overlay` |
| E4 | the square field is the diagonal of field 2 (p . p); field 2 minus the square field is the off-diagonal | PROVED by definition; `hits_squareField_iff` |
| E5 | in a section below t^3 only fields 1 and 2 occur among the t-rough members (fields >= 3 are struck by the core) | PROVED `CoreLeftover.primeOrSemiprime_of_rough_lt_cube` |
| E6 | no field is periodic (for every M >= 1 some n in field j has n + M outside it), while the overlay of any finite gear set is periodic with period the product of the gears (for k >= 1) | PROVED `Fields.field_not_periodic`, `field_not_closed_add`, `overlay_periodic` (FieldsB) |
| E7 | the class bias alternates with j (field 2 right-heavy, field 3 left-heavy, field 4 right-heavy) | MEASURED 9 of 9 signs (fields.md); known in print (Meng 2018) |
| E8 | a column's two members lie in fields (i, j); the column is hit iff (i, j) != (1, 1) | PROVED by E1-E3 |

## F. The union, and where the difficulty lands

| # | statement | status |
|---|---|---|
| F1 | the union of the fields 2 .. J covers every column of a section iff the section holds no twin | PROVED (E3) |
| F2 | the number of columns of a section hit by the union = sum over j of (columns hit by field j) - sum over i < j of (columns hit by both) + ...: inclusion-exclusion over the fields, where "hit by both i and j" means one member in field i and the other in field j, or one member in both (impossible, E1) | PROVED as an identity (finite inclusion-exclusion) |
| F3 | each single term of F2 is a count of one field (C6, D6): provable, and their sum exceeds the section's column count | PROVABLE by count |
| F4 | the correction terms of F2 are the joint counts of the pairs (Omega(6k - 1), Omega(6k + 1)) over the section: the joint distribution of the two members' field indices | this is where step 8 lives: the twin count is the (1, 1) cell of that joint distribution |
| F5 | the (1, 1) cell is positive in every section | = step 8. MEASURED to c_5 and to 10^7 on the finer sections. OPEN |

So the programme, carried out, lands the whole difficulty in one object: the joint
distribution of the field indices of a column's two members. Every single-field statement is
a count of one field or an exact identity (A-D, all PROVED or PROVABLE); every relation is an
identity (E); the union's coverage is inclusion-exclusion whose correction terms are the joint
distribution (F2-F4), and the twin is its (1, 1) cell (F5). What is on record about that joint
object: the pairs (P, P1 P2) and (P1 P2, P1 P2) among the core-open columns follow the
independent prediction (core_leftover.md, dead_branches_reopened_4.md: binomial at K = 8..16,
0 structure); the sign of the sum Omega(L) + Omega(R) is the sieve-invisible parity
(length_face.md LF3). A proof of F5 by this programme is a proof that the (1, 1) cell of the
joint distribution is never empty on a section, which the counts of the single fields cannot
give (they bound the marginals, not the cell) and which the sieve cannot give (the parity
twin has the same marginals and an empty cell below q'^2).

## The next proof steps in order (each an execution spec, no lane needed for the first three)

1. D4 confinement, D3, D5 into the kernel (elementary; one Formalist round with the
   statements above inline).
2. B5 both halves and E6 into the kernel (elementary).
3. C6 / D6 as counts: state the exact inequality the programme needs (the share of field j
   among the survivors of a section below 1) and cite Landau; formalisation deferred (mathlib
   lacks the Landau estimates).
4. F2 as a kernel identity over a finite section (inclusion-exclusion over the field-index
   pairs), giving F5 <=> the (1, 1) cell is positive, with every other cell a count.
5. The joint object itself: the (i, j) table per section, its marginals (the single-field
   counts) and its correction terms, measured on every section to 10^7 and compared with the
   product of the marginals (independence) cell by cell; the (1, 1) cell against its
   independent prediction. This is the one measurement the programme asks for, and it decides
   whether the fields' interaction is anything but independence at every scale.

## Step 5, done locally (research/stack/r8/joint_table.py, validated: 8 twins in [9, 121), 276 in [121, 16129))

The joint field-index table T(i, j) of a column's members (i left, j right), each cell against
the product of its marginals (independence), on five sections:

| section | columns | twin cell (1, 1) observed | independent | ratio | z |
|---|---|---|---|---|---|
| [9, 121) | 18 | 8 | 9.3 | 0.86 | -0.4 |
| [121, 16129) | 2,667 | 276 | 319.4 | 0.86 | -2.4 |
| [997^2, 1009^2) | 4,011 | 191 | 193.5 | 0.99 | -0.2 |
| [1999^2, 2003^2) | 2,667 | 93 | 101.0 | 0.92 | -0.8 |
| [3163^2, 3167^2) | 4,219 | 135 | 145.1 | 0.93 | -0.8 |

The other cells sit at 0.5 to 1.6 of independence with small counts (the (4, 4) cell is 0 of
2.4 at section 3). The twin cell's ratio against independence has a known value: the
Hardy-Littlewood singular series with the fold's primes removed, prod over p >= 5 of
(1 - 1/(p - 1)^2) = 0.8802, which is what the five sections scatter about. So the interaction
of the fields at the twin cell is the singular series: each gear g >= 5 removes 2 of its g
classes from the column while independence would remove 1 of g twice; that is the whole
correction, and it is KNOWN (Hardy-Littlewood 1923; register). The programme's F4 therefore
reads: the joint distribution is the product of the marginals times the singular series at
the (1, 1) cell, measured; a proof that the cell is non-empty on every section is a proof of
the Hardy-Littlewood count's positivity on a section, which no method in print gives. The
fields programme has now located step 8 exactly, and it is the same place.

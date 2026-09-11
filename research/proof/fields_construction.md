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

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

### D-all. What every field j >= 3 satisfies

| # | statement | status |
|---|---|---|
| D1 | field j = the union over primes p >= 5 of p . field j-1; field j ∩ 5S = 5 . field j-1 | PROVED `Fields.field_succ_eq_union`, `field_inter_five` |
| D2 | class rule: a member is = 1 mod 6 iff its count of class -1 factors is even | PROVED `Fields.class_rule` |
| D3 | field j is empty below 5^j, and 5^j is its first member | PROVED `Fields.field_empty_below`, `five_pow_mem_field` (FieldsB) |
| D4 | confinement: n in field j has minFac(n)^j <= n, so below P^2 its least factor g satisfies g^j < P^2 (field j hits only the teeth of the gears below P^{2/j}) | PROVED `Fields.field_least_factor_le`, `field_confined` (FieldsB); MEASURED 0 exceptions in 20 sections |
| D5 | the deepest field present below P^2 has index j with 5^j < P^2 | PROVED `Fields.field_index_le` (FieldsB); MEASURED exact (equality) at p = 11..53 |
| D6 | field j alone cannot cover a section | PROVABLE by count (Landau: density (log log x)^{j-1}/((j-1)! log x)); not formalised |

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

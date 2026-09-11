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
| B5 | the square field is empty in every finer section and equals the cuts of the construction | PROVED for the cuts by B2 and the definition of the cuts; the emptiness in finer sections is B2 read with p < q < p' impossible |
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

## D. Field j >= 3

| # | statement | status |
|---|---|---|
| D1 | field j = the union over primes p >= 5 of p . field j-1; field j ∩ 5S = 5 . field j-1 | PROVED `Fields.field_succ_eq_union`, `field_inter_five` |
| D2 | class rule: a member is = 1 mod 6 iff its count of class -1 factors is even | PROVED `Fields.class_rule` |
| D3 | field j is empty below 5^j | PROVABLE, elementary (each factor >= 5); not formalised |
| D4 | confinement: in a section below p'^2, every member of field j has a prime factor <= p'^{2/j}, so field j hits only the teeth of the gears <= p'^{2/j} | PROVABLE, elementary (the least of j factors is at most the j-th root); MEASURED 0 exceptions in 20 sections (fields.md E5); not formalised |
| D5 | the deepest field present in section k+1 has index floor(2 log_5 p_{k+1}) | PROVABLE from D3 and the section's top; MEASURED exact at p = 11..53 |
| D6 | field j alone cannot cover a section | PROVABLE by count (Landau: density (log log x)^{j-1}/((j-1)! log x)); not formalised |

## E. Relations between fields

| # | statement | status |
|---|---|---|
| E1 | the fields partition the composites of S: every composite lies in exactly one field | PROVED `Fields.field_unique`, `mem_field_length` |
| E2 | the overlay of the fields j >= 2 is exactly the struck set of S (given the square-root rule inside a section) | PROVED `Fields.overlay_iff_composite`, `blocked_iff_hits_overlay` |
| E3 | column k is a twin iff no field j >= 2 hits it | PROVED `Fields.twin_iff_not_hits_overlay` |
| E4 | the square field is the diagonal of field 2 (p . p); field 2 minus the square field is the off-diagonal | PROVED by definition; `hits_squareField_iff` |
| E5 | in a section below t^3 only fields 1 and 2 occur among the t-rough members (fields >= 3 are struck by the core) | PROVED `CoreLeftover.primeOrSemiprime_of_rough_lt_cube` |
| E6 | no field is periodic, though the overlay restricted to the gears <= q is periodic with period the product of the gears | PROVED (fields.md E2, two lines); not formalised |
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

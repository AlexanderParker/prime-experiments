# Closed forms for the fields: locating hits and locating opens (2026-09-12)

For every field type built in research/proof/fields_construction.md, the closed form for where
it hits (its members, in columns) and for where it is open (the columns it leaves), with the
honest status of each: CLOSED (an explicit formula or a finite computation with no search
beyond a stated bound), CLOSED RELATIVE TO the primes (explicit once the primes below a bound
are known), or NOT CLOSED (only the walk). Column k = (6k - 1, 6k + 1); a number n in S sits in
column (n + 1)/6 if n = 5 mod 6 (left member) and (n - 1)/6 if n = 1 mod 6 (right member).

## 1. The square field: CLOSED

Hits: the right members p^2 for primes p >= 5, at columns k = (p^2 - 1)/6, one per prime,
never a left member. [kernel Fields.square_column_unique, square_right_only]
Opens: every column except those; between consecutive primes p < p' the columns
(p^2 - 1)/6 + 1 .. (p'^2 - 1)/6 - 1 are all open to the square field, and the square field has
no member strictly between p^2 and p'^2. [kernel Fields.squareField_empty_between]
Next hit after column x: the column of q^2 for q = the least prime with q^2 > 6x + 1, i.e. the
next prime above sqrt(6x + 1). Next open: x + 1 unless x + 1 is a square column.

## 2. The prime field: NOT CLOSED in n; CLOSED as a certified walk

Hits: the primes >= 5 themselves, one per gear (the home strike). No formula in n on the
record; the next prime after p is the self-certifying mex form of research/proof/
next_gap_closed_form.md section 2 (core progressions truncated at B, tail residues, exact when
the mex is below B). Opens: the composites of S, i.e. the overlay (section 7 below).

## 3. The gear field of g (least factor g): CLOSED as a periodic residue set

Hits: n = g m with m >= g a survivor of the gears below g, i.e. m coprime to 6 P where P = the
product of the primes in [5, g). In columns: the set is periodic with period g P columns, has
exactly 2 . prod_{5 <= h < g} (h - 1) members per period, is mirror-symmetric (r hit iff -r
hit), and its members are the columns of g m for the residues m mod 6P coprime to 6P, listed
once per period. [kernel FieldsC: gearField_periodic, card_survivors_period, survivor_neg,
gearField_gap_scaling] For the first gears the residue lists are short: gear 5 hits the
columns = 1, 4 mod 5; gear 7 the columns = 1, 8, 13, 15, 20, 22, 27, 34 mod 35; gear 11 has
48 residues mod 385.
Next hit after column x: the next survivor m above (6x + 1)/g in the wheel {5..g-1}, which is
the wheel's certified walk: m + mex over h < g of the progression of (-m) mod h truncated at a
bound B, exact when the mex is below B (the single-tooth form of next_gap_closed_form.md; for
gear 5 the wheel is empty and the next hit is a lookup in the period 5). Then n = g m and its
column.
Opens: the complement of the residue set within the period; the gaps of the gear field are g
times the gaps of the wheel's survivors, so the longest open run of columns is g/6 times the
wheel's largest survivor gap (3, 7, 18, 30, 62, 82, 130 columns for g = 5..23, measured).
CLOSED: everything about a gear field is a statement about the wheel {5..g-1}, whose period,
count, symmetry and census are on record, scaled by g.

## 4. The smooth field of g and its layer: CLOSED

Hits of the smooth field: the products prod_{5 <= h <= g} h^{a_h} with exponent sum at least 2,
each in S automatically, at the column of that product; an explicit enumeration by exponent
vectors, finite below any bound (at most (log x / log 5)^{pi(g)-2} members below x, the count a
polynomial in log x: Ennola, KNOWN). Layer of g: the same with a_g >= 1, equal to g times the
g-smooth members of S that are at least 5.
Next hit after x: the least lattice point above x, a finite minimisation over exponent vectors
with h^{a_h} <= 2x, no search on the line. Opens: everything else; the field's gaps grow
without bound, so from any column the next open column is x + 1 or x + 2 except at the
finitely many smooth members below any bound.

## 5. The factor-count field j: CLOSED RELATIVE TO the primes, NOT CLOSED in n

Hits of field 2 in [x, y): the union over primes g <= sqrt(y) of g times the primes in
[x/g, y/g) (the range rule; kernel Fields.field_dilate, field_succ_eq_union); field j: g
times field j-1 in [x/g, y/g), recursively down to the primes. Given the primes below y/5,
every member of every field below y is listed with no search. In n alone there is no formula
(the fields are not periodic: kernel FieldsB.field_not_periodic).
Opens of field j: the complement; an interval is free of field j iff for every prime g <=
sqrt(y) the interval divided by g is free of field j-1 (the range rule read for opens); the
longest field-j-free ranges are alignments of scaled prime gaps (research/stack/r8/
range_rule.py: the longest field-2-free range of [121, 16129), 15257..15289, is the
intersection of the prime gaps (3049, 3061) x 5, (2179, 2203) x 7, (1381, 1399) x 11,
(1171, 1181) x 13, trimmed by the larger gears).
Confinement below P^2: field j hits only the teeth of the gears g with g^j < P^2 [kernel
FieldsB.field_confined], so inside a section the deep fields are located by a few small gears:
at P = 59, field 3 by {5, 7, 11, 13}, field 4 by {5, 7}, field 5 by {5} alone.

## 6. The gear fields together, the smooth fields together

The gear fields partition the composites of S by least factor [kernel FieldsC.gearField_minFac,
gearField_disjoint], so the overlay is their disjoint union: the struck set on S is
  union over gears g of g . (survivors of the gears below g, at least g),
each term CLOSED (section 3). The smooth fields are nested (C_5 in C_7 in ...), each CLOSED
(section 4); their union over all g is the whole overlay, since every composite is smooth for
its largest factor, and the layer of g is the part of gear g's field with a g-smooth cofactor.

## 7. The overlay and its opens (the twins): CLOSED only as the certified walk

Hits: the composites of S = the union of the gear fields, each closed, but the union's
membership at a column needs the least factor of the column's members, i.e. a divisibility
test against every gear up to the square root: the sieve. Opens: the twins, located by the
two-tooth certified mex form (next_gap_closed_form.md section 5: the next twin after p is
p + mex of the core progressions and the tail residues, exact when below the bound). No
formula in n; a bound on the walk is the record of the machine on a stretch, the object of
step 8. This is the one open locator that is not closed, and it is exactly the conjecture.

## Summary

| field | hits | opens | closed? |
|---|---|---|---|
| square | columns (p^2 - 1)/6 | all other columns; none between consecutive squares | CLOSED |
| prime | the primes | the composites | walk only |
| gear g | g x (survivors below g), periodic, 2 prod (h - 1) per period | the complement mod the period; gaps g x the wheel's | CLOSED (as the wheel scaled) |
| smooth of g, layer of g | the exponent lattice of {5..g} | everything else, gaps unbounded | CLOSED |
| factor-count j | g x field j-1 over the gears, recursively to the primes | alignments of scaled prime gaps | relative to the primes |
| overlay | the union of the gear fields | the twins | walk only: step 8 |

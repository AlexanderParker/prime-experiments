# The walk through the fields (owner, 2026-09-13)

Owner's direction: do not prove that the teeth cannot cover the landing zones; prove that
nothing stops the walk from landing. Identify which fields each step of the walk passes
through, and for each field prove why it steers toward the landing zone. Script
research/stack/r8/walk_fields.py; machines 31, 101, 401 in full, 11 to 20000 for the run at the
zone start.

## The walk and its zone

One flip from home about the axis k M, M = 6: landing (12k - 1, 12k + 1), the column 2k. The
landing zone of machine q is the run of k with q < 12k - 1 and 12k + 1 <= q^2 (78 columns at
31, 842 at 101, 13,367 at 401). The rule takes the first k of the zone whose landing no gear up
to q strikes.

## What each field does on the zone (exact)

- multiples of h (the rows of the machine): row h paints k = -+12^-1 (mod h), two teeth per
  period h; on the zone the painted share is 2/h to the third decimal at every machine (5:
  0.400, 7: 0.286, 11: 0.182, ... 401: 0.005). Steering: the rule steps past a painted k; row h
  alone can force at most two consecutive skips per period... exactly, row h leaves h - 2 of
  every h consecutive k, and never paints two adjacent k unless h = 5 (teeth 2, 3 mod 5).
- squares: 12k + 1 = g^2 (the left member 12k - 1 is never a square); one k per gear g, and
  each lies in the row of its g. Contained in the multiples field.
- higher:h (the landing member is h times a number with no smaller gear factor): contained in
  row h; the union over h is exactly the painted set. So the higher fields are the multiples
  rows read by smallest gear; they add no paint.
- products:j: the painted set split by the number of prime factors; the union is exactly the
  painted set (31: j = 2: 57, 3: 10, 4: 2; 401: 9125, 5355, 1490, 215, 28). No paint of its own.
- blind gears: none. On the m-line every gear h >= 5 has its two teeth (12 is invertible mod
  h), unlike the square-offset zones where a quarter of the gears are blind. Every gear of
  the machine acts on the zone.

So the only fields that act on the landing zone are the multiples rows, one per gear; the
squares, the higher fields and the product fields are relabellings of their paint. A step of
the walk passes through exactly the rows whose teeth fall on the k it steps over.

## The joint fact (exact, not statistical)

The rows are independent modulo the product P of the gears: in every full period of P
consecutive k there are exactly prod (h - 2) unpainted k. The zone is one phase of that period
(78 of 3.3 x 10^10 at 31; 13,367 of 10^162 at 401). The unpainted found in the zone against the
period share: 14 against 14.5 (31), 100 against 94.8 (101), 906 against 913.2 (401).

## What stops the walk, exactly

The rule lands on the first unpainted k at or after the zone start k_0 = the first k with
12k - 1 > q. It is stopped only if every k from k_0 to the zone's end is painted: a painted run
anchored at the zone start spanning the whole zone. Nothing else can stop it. Let L(q) be the
painted run at the zone start (the k the rule steps past). Machines 11 to 20000: mean 8.7,
median 6, largest 55 (q = 13007, zone length 14,097,420); the first column above q already
lands at 208 machines. L(q) tracks the twin-gap scale near q (ln^2 q / 12: 1.8, 4.0, 7.1, 8.2 at
q = 101, 1009, 10007, 19997 against L = 0, 0, 4, 12).

The rows that paint the run at the zone start are the small composites just above q: every
number in (q, 2q) is prime or has a factor below q, so the run at the zone start is decided by
the primes just above q, and L(q) is the distance from q to the first twin above q with
midpoint a multiple of 12, in columns of the m-line.

## The statement, sharpened by the fields

The walk lands for machine q iff L(q) is below the zone length (q^2 - q)/12. This is weaker
than the record statement (the record bounds every run; only the run anchored at the zone
start matters), and it is a statement about the columns just above q: the first aligned twin
above q lies below q^2. Kernel: MirrorWalk.walk_lands_of_record already gives the landing from
any anchored stretch (take x = k_0); the sharpened hypothesis is the run at k_0 alone.

## Per-field steering, summarised

| field | on the zone | steering | status |
|---|---|---|---|
| multiples of h | two teeth per period h | the rule steps past them; h - 2 free k per h consecutive | exact |
| squares | one k per gear, right member | inside row g | exact |
| higher:h | smallest-gear relabelling of row h | no paint of its own | exact |
| products:j | order relabelling of the painted set | no paint of its own | exact |
| blind gears | none on the m-line | every gear acts | exact |
| mirror-carried gears (S) | their rows are removed from the zone | passage by mirror, exact, capped by the primorial below q^2/2 | exact |
| all rows jointly | one phase of the period; prod (h-2) free per period | the zone holds the period share to within a few percent | measured |
| the run at the zone start | L(q) = the twin gap above q in m-line columns | the only thing that can stop the walk | measured to 20000 |

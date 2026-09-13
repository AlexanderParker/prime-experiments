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

## The paint just above q (owner: let's do that; 2026-09-13)

Field docs/zone_start_field.html; script research/stack/r8/zone_start.py. Origin q, columns the
numbers q + d, row h painted where h divides q + d (one class of d per row, d = -q mod h: the
top gear fixes every row's phase, as the square fixed it at the square origin).

Exact facts of the zone start:

- A gear h paints an offset d only through a multiple q + d = h m with m >= 5 coprime to 6
  (h itself is below q), so only the gears h <= (q + d)/5 can reach offset d. Gears above
  2q/5 paint nothing below 2q.
- A painted member q + d below 2q is a composite below 2q, so it has a prime factor at most
  sqrt(2q): as long as the run at the zone start stays below q offsets, it is painted by the
  gears up to sqrt(2q) as smallest factors, each with its cofactor as the other row. The
  machine of size sqrt(2q) lays the paint at the start of the zone of the machine of size q.
- Machines 11 to 20000: the run at the zone start spans at most 0.36 q (q = 431) for every
  q >= 29, and every painted member's smallest gear is at most 0.78 sqrt(2q); the five machines
  11 to 23 are the exceptions (their zones start within a few columns of q and the run
  overshoots q).

Read at the worst machine in range, q = 13007 (run of 55 columns, landing at offset 672): the
first painted columns are (13019, 13021) by rows 29, 47, 277, 449; (13031, 13033) by 83, 157;
(13043, 13045) by 5, 2609; (13055, 13057) by 5, 7, 11, 373, 1187; each painted column shows a
small gear and its cofactor. Rows able to reach the landing offset: 555 of 2260 gears at
q = 19997, 44 of 167 at q = 1009.

So the open statement, read at the zone start: the gears up to sqrt(2q), phased by q, cannot
paint every aligned column from q up to q^2. Whatever they paint in (q, 2q) they paint as the
smallest factors of the composites there, with the cofactors following; above 2q the larger
gears join with their own small multiples. The question is the length of the painted run at
the phase -q of the small gears, which is the twin gap above q.

## The sub-machine's wheel at phase -q (owner: go; 2026-09-13)

Script research/stack/r8/submachine_phase.py.

Exact chain. Below 2q every painted member of the landing family is a composite below 2q, so
it has a prime factor at most sqrt(2q): below 2q, painted by the machine = painted by the
sub-machine B(q) = the gears up to sqrt(2q). So the run at the zone start, L(q), is the painted
run of the B(q)-wheel on the m-line at the phase k_0, as long as it stays below 2q; and
L(q) <= R_B, the m-line record of the sub-machine (its longest painted run over a full period),
whenever R_B < q/12. More generally, with B the gears up to sqrt(q + l): if R_B < l/12 the run
cannot reach l.

The sub-machine records, exact over full periods (opens = the CRT survivors, prod (h - 2) per
period; the record = the largest gap between consecutive opens):

| y | period on the m-line | opens per period | record R_y | R_y / (y^2/24) |
|---|---|---|---|---|
| 5 | 5 | 3 | 2 | 1.92 |
| 7 | 35 | 15 | 4 | 1.96 |
| 11 | 385 | 135 | 7 | 1.39 |
| 13 | 5,005 | 1,485 | 9 | 1.28 |
| 17 | 85,085 | 22,275 | 17 | 1.41 |
| 19 | 1,616,615 | 378,675 | 19 | 1.26 |
| 23 | 37,182,145 | 7,952,175 | 34 | 1.54 |
| 29 | 1,078,282,205 | 214,708,725 | 43 | 1.23 |

R_y grows like y^2 (between y^2/20 and y^2/12 here), the same growth as the certified records
F(q) of the full machine (6F/q^2 between 0.28 and 0.42). L(q) <= R_{B(q)} at 73 of the 77
machines with sqrt(2q) <= 29 (the four misses are q = 11 .. 23, where the run overshoots 2q);
L(q) is far below R_B at every one (at q = 419: L = 0, R = 34).

What the sufficient condition becomes. With B the gears up to sqrt((1 + c) q) the condition is
R_B < c q / 12; since R_y is about y^2/20, this needs c > 1.5 and in the limit of large c it is
R_y < y^2/12, the record route (8c) on the m-line for the sub-machine. Every coordinate we
have used (the full window, the square zone, the zone start) leads to the same statement: the
record of a wheel of gears up to y stays below y^2/12 (the constant depending on the
coordinate). The record route is the invariant form.

Is the phase -q special? No. For the wheel of the gears 5..y at every phase of one period,
against the phases k_0(q) for the primes q to 200,000:

| y | all phases: mean run, max | prime phases -q: mean run, max | share with run >= 5: all, prime |
|---|---|---|---|
| 13 | 1.94, 9 | 2.09, 9 | 0.100, 0.110 |
| 17 | 2.32, 17 | 2.50, 14 | 0.153, 0.168 |
| 19 | 2.70, 19 | 2.89, 19 | 0.204, 0.219 |

The prime phases behave like all phases. The walk's short runs (L mean 8.7 to 20000) against
the sub-machine records (34, 43) are what any phase gets: the record is the worst phase, and
the zone start is an ordinary one.

Standing after this step: the zone-start reading is exact and it lands on the record route at
the sub-machine, with the record's known y^2 growth and known margins. It gives a location
(the first aligned twin above q) and a mechanism (the sub-machine lays the paint), and it does
not give a reason the record stays below y^2/12.

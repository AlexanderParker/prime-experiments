# Manager notes toward the proof (2026-09-04)

Exact computations by the manager while the prover lanes run. Vocabulary as in
docs/proof-search/alignment-rules.md section 0.

## 1. The budget inequality per step, decomposed on the top gear's two arcs

With a = 2 round(q'/6) (the short arc) and b = q' - a (the long arc), a + b = q':

    F(M+q') - F(M) = [F_2(M) - F(M)]  +  [F(M+q') - F_2(M)]
                     join cost           extra-kill cost
                     (measured 2..16)    (<= a at literal steps, kernel at six; 20 vs 12 once, padded)

so the budget slack per step is b - join cost, measured 5, 6, 6, 9, 14, 9, 15, 25, 17, 18, 19, 25
at 11->13 .. 53->59 and widening. The window grows by (q'^2 - p^2)/6 = 8, 20, 12, 28, 52, 20, 68,
52, 28, 60, 100, 112 at the same steps. Typical gap (period / openings) 2.85 .. 6.8.

## 2. One-class world: the one-hole stretch IS the next record

Ordinary Jacobsthal on primorials P_k (one killed residue per prime), full periods to P_9:

    k   P_k        j   one-hole  two-hole   next prime
    1   2          2    4         6          3
    2   6          4    6        10          5
    3   30         6   10        14          7
    4   210       10   14        22         11
    5   2310      14   22        26         13
    6   30030     22   26        34         17
    7   510510    26   34        40         19
    8   9699690   34   40        46         23
    9   223092870 40   46        58         29

one-hole(P_k) = j(P_{k+1}) and two-hole(P_k) = j(P_{k+2}) at every k shown. Reason: gaps between
coprimes are even, so two kills by one new prime p in a stretch need a gap that is a positive
multiple of p, hence >= 2p, impossible while j(P_k) < 2 p_{k+1}; Hagedorn's table has that through
k = 18 (first failure k = 19: 152 >= 2 x 71). So in one-class, through k = 18, the pair statement
is exactly the increment statement j(P_{k+1}) <= j(P_k) + p_{k+1}, which Hagedorn's values satisfy
with increments 2,2,4,4,8,4,8,6,6,12,8,8,16,10,6,12,14,20,22,... against primes 3,5,7,...

The two-class machine has chains from the start (letters a ~ q'/3 < F), which is why its ladder is
harder than the one-class ladder at the same size.

## 3. Sole-coverer profile of record stretches (exact, full periods)

In every record stretch and every one-hole record stretch, one-class P_5..P_8 and two-class
m11..m19, EVERY gear is the sole coverer of at least one column (no idle gear; a phase shift of an
idle gear would cover the hole and lengthen the record), and the top gear is the sole coverer of
exactly 1 or 2 columns. The overlap is small: columns struck by two or more gears are 3/13, 5/21,
7/25, 12/33 (one-class P_5..P_8) and 1/6, 1/10, 3/17, 6/24 (two-class m11..m19) - a record stretch
is a near-perfect tiling, and the top gear's whole contribution is one or two columns. This is the
"made at the top" picture in one number.

Consequence tried, not closed: removing the top gear turns the record stretch of M into a one-hole
stretch of M-minus-top (when the top gear is sole at one column), so F(M) <= F_2(M^-); and the
one-hole record of M into a two- or three-hole stretch of M^-, so F_2(M) <= F_3(M^-) or F_4(M^-).
That relates the join cost at M to hole costs one level down, but the hole costs F_{J+1} - F_J at
a fixed machine are not monotone in J (m29: 12, 10, 5, 15, 5), so no descent closes from this alone.

## 4. Phase-shift lemma (one-class, elementary, on record here for the lanes)

Let [x - g1, x + g2] be a one-hole stretch of P (hole x coprime to P). For a prime q0 | P choose
t = 0 mod P/q0, t = -x mod q0. The translate by t covers x and uncovers exactly the columns that
q0 alone covered. Hence in a one-hole RECORD stretch every prime is the sole coverer of at least
one column (else the translate is a full cover longer than j, contradiction). The q0-only columns
are multiples of q0, so at most (g1 + g2)/q0 + 1 of them, spaced >= q0 apart. This bounds nothing
by itself (after the shift the pieces between new holes are each <= j, which gives only
(holes + 1)(j + 1)), but it is the only mechanism found so far that moves a hole, and it says the
hole cost is governed by how few columns the top primes cover alone.

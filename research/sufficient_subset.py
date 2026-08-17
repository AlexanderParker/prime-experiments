"""Sufficient sub-sets of gears for finding twins inside the window.

Two framing facts, each one line:

1. NO subset is sufficient for the WHOLE window (y, y^2]: every gear's square
   is an in-window root kill on a candidate (q^2 = 1 mod 6 always, so q^2 is
   always a right member), and dropping gear q falsely opens slot (q^2-1)/6
   whenever q^2 - 2 is prime. Drop 13 from the y=13 set: slot 28 = (167,169)
   reports as a twin.

2. But the window is GRADED: the pattern of gears <= z is exact on slots whose
   members stay below (nextprime(z))^2 - the square-root tower, localised. So
   the subset needed depends on where in the window you look, and finding the
   FIRST twin above y needs only gears <= sqrt(that twin's upper member).

This script measures the required depth z*(y) = isqrt(m+2) for the first twin
(m, m+2) above y. Result (y up to 400): the needed depth averages
0.42 * sqrt(6y), and the kept-gear count collapses - at y = 389 only 6 of 75
gears are needed. Caveat recorded honestly: "the first twin sits close above
y" is empirical (corpus 12a: within 169 of y for all y <= 3163); proving it
would be stronger than Reduction A.

Half-winding note: the mirror symmetry means a subset's behaviour is fixed by
half its primorial, but the mismatch under attack is e^y vs y^2 - the factor
of 2 is conceptual, not asymptotic.
"""
import math
from sympy import isprime, primerange

def first_twin_above(y):
    m = y + 1
    while True:
        if m % 6 == 5 and isprime(m) and isprime(m + 2):
            return m
        m += 1

def required_depth(y):
    m = first_twin_above(y)
    return m, math.isqrt(m + 2)

if __name__ == "__main__":
    print(" y    first twin    slot   depth  gears kept/total")
    for y in primerange(11, 400):
        m, d = required_depth(y)
        gears = [q for q in primerange(5, y + 1)]
        kept = [q for q in gears if q <= d]
        print(f"{y:4d}  ({m},{m+2})  {(m+1)//6:6d}  {d:5d}   {len(kept)}/{len(gears)}")

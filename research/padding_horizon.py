"""Round 14 lateral, part 3: what the p <= 1 bound buys, and where it lapses.

With at most one padded link per run AND padded links of size exactly q'
(both hold while 2q' > F(M) and F_2(M) < 2q'), a legal killed run is
  [literal chain] --q'-- [literal chain],
each literal chain capped at 6 kills (cap-6 theorem), so

    k <= 12      and      span <= 2*(2q' + s) + q' = 5q' + 2s <= 6.33 q'

- a CONSTANT multiple of q', restoring the ceiling I withdrew in round 13
(at 6.33q' instead of the 2.67q' that literal-only reasoning gave).

The two enabling conditions are threshold conditions on the spectrum, and both
fail at the SAME step. Tabulated here.
"""
from math import prod
from split_gap_law import primes

SPEC = {13: [11, 16, 23, 26, 28, 31], 17: [18, 25, 28, 33, 35, 40],
        19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
        29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97]}
SPEC_LB = {37: [88, 90]}
STEPS = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31), (31, 37), (37, 41)]

print(f"{'step':>9} {'F(M)':>6} {'F2(M)':>6} {'q':>3} {'2q':>4} "
      f"{'pad size = q?':>14} {'p<=1?':>7} {'span ceiling':>13}")
for y, qp in STEPS:
    spec = SPEC.get(y) or SPEC_LB.get(y)
    lb = y in SPEC_LB
    F, F2 = spec[0], spec[1]
    u = pow(6, -1, qp); s = (2 * u) % qp
    single = 2 * qp > F          # no 2q' gap can exist -> padded links are exactly q'
    ponly = F2 < 2 * qp          # no two adjacent padded links
    mark = ">=" if lb else " "
    ceil = f"{(5*qp + 2*s)/qp:.2f} q'" if (single and ponly) else "NONE"
    print(f"  {y:>4}->{qp:<3} {mark}{F:>5} {mark}{F2:>5} {qp:>3} {2*qp:>4} "
          f"{('yes' if single else 'NO'):>14} {('yes' if ponly else 'NO'):>7} "
          f"{ceil:>13}")
print()
print("Both conditions are F(M) or F2(M) vs 2q'. Ratios (want < 1):")
for y, qp in STEPS:
    spec = SPEC.get(y) or SPEC_LB.get(y)
    lb = y in SPEC_LB
    print(f"  {y:>2}->{qp:<2}: F(M)/2q' = {spec[0]/(2*qp):.2f}, "
          f"F2(M)/2q' = {spec[1]/(2*qp):.2f}" + ("  (lower bounds)" if lb else ""))
print()
print("F and F2 grow superlinearly against the next prime, so both ratios climb")
print("monotonically and, once past 1, stay past. The ceiling is therefore a")
print("SMALL-MACHINE phenomenon that ends exactly at 37->41 - it does not")
print("restore an asymptotic bound.")

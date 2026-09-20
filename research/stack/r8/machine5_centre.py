"""Mirror about lap 0 (node c.iv, Q6 on laps): the lap machine 7..q, period P_L = prod(7..q), is
symmetric under k -> -k, so also about the half period. The two centre laps (P_L -+ 1)/2 have
members 15 P_L +- 14 and 15 P_L +- 16; since every gear divides P_L, gear g strikes the lap at
offset j/2 from the centre (j odd) iff g | 15j - 1 or g | 15j + 1 - the leg rule read from the
centre, as 30k -+ 1 is read from the origin. PRE-REGISTERED: (1) the centre pair is always struck
by 7 (14 = 2 x 7); (2) offsets j = 3: 11 or 23; 5: 19 or 37; 7: 13 or 53; 9: 17 or 67; 11: 41 or
83; 13: 7 or 97; (3) the central struck run has 2m laps, m = number of consecutive odd j from 1
whose 15j -+ 1 has a prime factor in [7, q]: 4 laps for q = 11, 13, 17; 10 for q = 19, 23, 29,
31, 37; 16 for q = 41 .. 113; (4) at q = 11 the record run IS the central run (laps 37..40).
"""
from sympy import primerange, factorint
from math import prod
def small_factor(n, q):  # least prime factor of n in [7, q], or None
    return next((p for p in sorted(factorint(n)) if 7 <= p <= q), None)
print("offset j: gears striking the centre laps (prime factors >= 7 of 15j-1, 15j+1):")
for j in range(1, 26, 2):
    print(f"  j={j:2d}: {sorted(p for m in (15*j-1, 15*j+1) for p in factorint(m) if p >= 7)}")
for q in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 113, 127]:
    m = 0
    while small_factor(15*(2*m+1)-1, q) or small_factor(15*(2*m+1)+1, q): m += 1
    print(f"q={q:3d}: central struck run {2*m} laps (offsets j = +-1..+-{2*m-1})")
# direct check at q = 11..23: laps about P_L/2 struck as predicted
for q in [11, 13, 17, 19, 23]:
    gears = list(primerange(7, q + 1)); P = prod(gears); c = (P - 1) // 2
    word = []
    for k in range(c - 6, c + 8):
        word.append([g for g in gears if (30*k-1) % g == 0 or (30*k+1) % g == 0])
    print(f"q={q}: P_L={P}, laps {c-6}..{c+7} struck by {word}")

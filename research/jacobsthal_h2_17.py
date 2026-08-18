"""h_2 at the next primorial (gears <= 17) - the decisive test of Ziller-Morack
Conjecture 6, whose margin collapsed to 3.8% at gears <= 13."""
import numpy as np
from math import prod, gcd
gears = [3, 5, 7, 11, 13, 17]
P = prod(gears)
best, arg, twin = -1, [], None
buf = np.empty(P, bool)
for e in range(1, P // 2 + 1):
    buf[:] = True
    for q in gears:
        buf[0::q] = False
        buf[(-e) % q::q] = False
    idx = np.flatnonzero(buf)
    g = np.diff(np.append(idx, idx[0] + P))
    F = int(g.max())
    if e == 1:
        twin = F
    if F > best:
        best, arg = F, [e]
    elif F == best and len(arg) < 6:
        arg.append(e)
    if e % 20000 == 0:
        print(f"  ... e = {e}/{P//2}, running max F = {best}", flush=True)
y = 17
print(f"gears {gears}  P = {P}  scanned e = 1..{P//2}")
print(f"h_2 = 2*maxF = {2*best}   (maxF = {best} at e = {arg})")
print(f"twin case F_2 = {twin} (h_2 twin = {2*twin})")
print(f"bound y^2-y = {y*y-y}   Conjecture 6: "
      f"{'HOLDS' if 2*best < y*y-y else 'FAILS'}   margin = {y*y-y - 2*best}")

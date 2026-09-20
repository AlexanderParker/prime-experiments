"""The residue-collapse census (draft section 5b): the number C_{q'}(L) of covered windows of
length L of machine q' (over its period P_q q') equals, by CRT,
   sum over windows W of machine q of length L (over P_q) of
      q'   if W has no hole,
      2    if all holes of W are congruent mod q',
      1    if the holes of W occupy exactly two residues r1, r2 mod q' with r2 - r1 = +-inv3 mod q',
      0    otherwise.
Verification at q = 11 -> 13 and q = 13 -> 17 for several L (direct count in the joint period).
"""
from math import prod

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})
def paint(gears, P):
    painted = bytearray(P)
    for g in gears:
        for t in teeth(g):
            painted[t::g] = b"\x01" * len(range(t, P, g))
    return painted

for base, q2 in [([5, 7, 11], 13), ([5, 7, 11, 13], 17)]:
    P = prod(base); old = paint(base, P); d = inv(3, q2)
    Pj = P * q2; new = paint(base + [q2], Pj)
    for L in [6, 8, 10, 12, 14, 17, 20]:
        # census over machine q windows
        census = 0
        holes_pos = [i for i in range(P) if not old[i]]
        # sliding: holes in window [s, s+L) cyclic
        for s in range(P):
            hs = [(s + i) % P for i in range(L) if not old[(s + i) % P]]
            if not hs: census += q2; continue
            res = sorted({h % q2 for h in hs})
            # careful: residues of actual positions in the joint period differ by multiples of P;
            # the census must use positions h in [s, s+L) as integers (not reduced mod P) - the
            # window at translation s + mP has holes at h + mP with residue (h + mP) mod q'. All
            # holes shift by the same mP, so their residue DIFFERENCES are what matter:
            base_h = hs[0]
            diffs = sorted({(h - base_h) % q2 for h in hs})
            if diffs == [0]: census += 2
            elif len(diffs) == 2 and (diffs[1] % q2 in (d % q2, (-d) % q2)): census += 1
        direct = sum(1 for s in range(Pj) if all(new[(s + i) % Pj] for i in range(L)))
        print(f"q = {base[-1]} -> {q2}, L = {L}: census {census}, direct {direct}, match {census == direct}")

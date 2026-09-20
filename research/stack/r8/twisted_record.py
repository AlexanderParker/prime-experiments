"""The twisted machine T(q; q'): gears 5..q, gear g strikes k iff 6(c + k q') = +-1 mod g for a fixed
class c, i.e. k in two classes mod g at distance (3q')^{-1} mod g. F_T = its record (longest painted
run; by RigidShift the record over phases equals the record of the pattern for c = 0 over one
period). PROVED: F(q') <= q' (F_T(q; q') + 1) since every non-tooth class of a covered window is a
painted run of T of length >= floor(L/q').
Pre-registered: F_T(q; q') is of the same order as F(q) (within a factor 2), so the bound is q' times
too weak against the window q'^2/6; verify F(q') <= q'(F_T + 1) at each step.
"""
from math import prod

def inv(a, m): return pow(a, -1, m)

def twisted_record(gears, q2):
    P = prod(gears)
    painted = bytearray(P)
    for g in gears:
        w = inv(6 * q2, g)          # k = +-w mod g  (class c = 0)
        for t in {w % g, (-w) % g}:
            painted[t::g] = b"\x01" * len(range(t, P, g))
    z = painted.find(b"\x00"); rot = painted[z:] + painted[:z]; run = mx = 0
    for x in rot:
        if x: run += 1; mx = max(mx, run)
        else: run = 0
    return mx

def record(gears):
    P = prod(gears); painted = bytearray(P)
    for g in gears:
        c = inv(6, g)
        for t in {c % g, (-c) % g}:
            painted[t::g] = b"\x01" * len(range(t, P, g))
    z = painted.find(b"\x00"); rot = painted[z:] + painted[:z]; run = mx = 0
    for x in rot:
        if x: run += 1; mx = max(mx, run)
        else: run = 0
    return mx

primes = [5, 7, 11, 13, 17, 19, 23]
for k in range(2, 7):
    gears = primes[:k]; q2 = primes[k]
    F = record(gears); FT = twisted_record(gears, q2); Fn = record(gears + [q2])
    win = (q2 * q2 - 1) // 6 - (q2 + 7) // 6 + 1
    print(f"q = {gears[-1]} -> q' = {q2}: F(q) = {F}, F_T(q;q') = {FT}, distances (3q')^-1 mod g = "
          f"{[inv(3*q2, g) for g in gears]}; bound q'(F_T+1) = {q2*(FT+1)} vs F(q') = {Fn}, window {win}")

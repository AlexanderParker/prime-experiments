"""The owner's construction (2026-09-21): drop q from the machine, let the lower machine 5..q_-
cycle inside q's range, and follow the known opening (-1, 1) at its copies k * P_L, k = 1..q-1
(P_L = product of the gears 5..q_-). Members 6 k P_L -+ 1 are coprime to every gear below q by
construction; each overlay gear g >= q strikes at most two copies. Question: does some copy have
both members prime, for every q? Direct check to q = 113.
"""
from sympy import isprime, primerange
from math import prod
ps = list(primerange(5, 120))
for i in range(1, len(ps)):
    q = ps[i]; lower = ps[:i]; PL = prod(lower)
    tw = [k for k in range(1, q) if isprime(6 * k * PL - 1) and isprime(6 * k * PL + 1)]
    pr = sum(1 for k in range(1, q) for m in (6 * k * PL - 1, 6 * k * PL + 1) if isprime(m))
    print(f"q={q:3d} lower=5..{lower[-1]:3d} copies={q-1:3d} prime members={pr:3d} twin copies={len(tw)} k={tw}")

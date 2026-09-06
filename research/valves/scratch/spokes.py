"""Spokes: the columns of mQ (numbers mQ - 1, mQ + 1), m = 1 .. Q - 1, for Q = q#.
Always engine-open (coprime to q#); twin iff both prime. Counts for 5#, 7#, 11#, 13#, 17#
(17# sampled at m <= 200000).

usage: uv run python research/valves/scratch/spokes.py
"""
import math
from sympy import isprime, primerange

Q = 1
for q in [5, 7, 11, 13, 17]:
    Q *= q
    if q == 5:
        Q = 30
    M = min(Q - 1, 200000)
    tw = 0; first = None; last = None; opens = 0
    for m in range(1, M + 1):
        a = isprime(m * Q - 1); b = isprime(m * Q + 1)
        opens += a + b
        if a and b:
            tw += 1
            if first is None:
                first = m
            last = m
    # heuristic: prod_{p<=q} p/(p-2) (p=2 -> 2) * 2C_2 / log^2(mQ) summed
    C2 = 1.0
    for p in primerange(3, 10 ** 6):
        C2 *= 1 - 1 / (p - 1) ** 2
    boost = 2.0
    for p in primerange(3, q + 1):
        boost *= p / (p - 2)
    # conditional on both members coprime to q#: 4 prod_{3<=p<=q} p^2/(p-1)^2 prod_{p>q} p(p-2)/(p-1)^2 = boost * 2C_2
    local = 1.0
    for p in primerange(3, q + 1):
        local *= 1 - 1 / (p - 1) ** 2
    heur = sum(boost * 2 * C2 / (math.log(m * Q) ** 2) for m in range(1, M + 1))
    print(f"{q}# = {Q}: m <= {M}: spoke twins {tw} (heuristic {heur:.1f}), first m {first}, last m {last}, prime members {opens} of {2 * M}")

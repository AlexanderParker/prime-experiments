"""Stacking the hole words (node c.ix follow-up). A chain of m filled holes starting at hole h0
puts gear g at phase h0 mod g of its word; by CRT every phase tuple occurs, so the longest chain
over a period is a puzzle on the words alone. The densest m-window of a word is the most laps the
gear can put into m consecutive holes. PRE-REGISTERED (forced fills): q = 19 - a filled hole needs
5 laps, the words' densest 1-windows are 11: 2, 13: 1, 17: 1, 19: 1 (sum 5), so EVERY filled hole
at q = 19 has 11 doubling and 13, 17, 19 once each, with no coincidences; q = 23 - a 2-chain needs
10 laps, densest 2-windows 11: 3, 13: 2, 17: 2, 19: 2, 23: 2 (sum 11), so every 2-chain has 11 in
a 3-window ("14|5" or "1|25") or every other gear at its maximum. Also the word bound: longest
chain m(q) <= largest m with sum of densest m-windows >= 5m; compare with the observed chains.
"""
from sympy import primerange
from math import prod
import numpy as np
def word(g):
    a = pow(30, -1, g)
    return [[p for p in range(1, 6) if (7*h+4+p) % g in (a, (-a) % g)] for h in range(g)]
W = {g: word(g) for g in primerange(11, 62)}
def dense(g, m):
    w = W[g]; return max(sum(len(w[(h + i) % g]) for i in range(m)) for h in range(g))
print("word bound on the chain length (largest m with sum of densest m-windows >= 5m):")
for q in [19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
    gears = [g for g in W if g <= q]
    m = 1
    while m < 40 and sum(dense(g, m + 1) for g in gears) >= 5 * (m + 1): m += 1
    print(f"  q={q}: bound m <= {m}" + ("  (NO BOUND: the densest windows sum to at least 5 per hole)" if m >= 40 else f"  (densest windows at m={m}: {[dense(g, m) for g in gears]})"), flush=True)
# q = 19: every filled hole has 11 doubling, 13/17/19 once, no coincidence
gears = [11, 13, 17, 19]; P = prod(gears)
def strikers(h, gears):
    return [[g for g in gears if (30*(7*h+4+p)-1) % g == 0 or (30*(7*h+4+p)+1) % g == 0] for p in range(1, 6)]
filled = [h for h in range(P) if all(strikers(h, gears))]
ok = all(sum(11 in s for s in S) == 2 and all(len(s) == 1 for s in S) for S in (strikers(h, gears) for h in filled))
print(f"q=19: {len(filled)} filled holes (check), forced fill 11 double + 13, 17, 19 single, no coincidences: {ok}")
# q = 23: 2-chains, 11's contribution
gears = [11, 13, 17, 19, 23]; P = prod(gears)
struck = np.zeros(7 * P + 20, dtype=bool)
for g in [7] + gears:
    a = pow(30, -1, g); struck[a::g] = True; struck[(-a) % g::g] = True
H = struck[5:5 + 7 * P].reshape(P, 7)[:, :5]; f = H.all(axis=1)
starts = [h for h in range(P - 1) if f[h] and f[h + 1]]
from collections import Counter
c11 = Counter(); cmax = Counter()
for h in starts:
    S = strikers(h, gears) + strikers(h + 1, gears)
    n11 = sum(11 in s for s in S); c11[n11] += 1
    cmax[tuple(sum(g in s for s in S) for g in gears)] += 1
print(f"q=23: {len(starts)} two-hole chains (check); laps from 11 in the chain: {dict(c11)}; contribution patterns (11,13,17,19,23): {dict(cmax)}")

"""Inside 13's staircase (node c.vii): hole h = 1..10 mod 13 carries 13 at position p(h) =
5,5,4,4,3,3,2,2,1,1. The other four positions are filled by the gears 11, 17, 19, 23, 29, ... A
gear fills two positions of one hole only at its lap distance d (11, 23: 3; 29, 31: 2; 59, 61: 4)
and the two laps lie in its two classes (30k = -1, 30(k+d) = +1 mod g). RULES: with 13 at p, the
admissible doubler pairs are the position pairs (a, a+d) avoiding p; a hole is filled 2+2 (two
doublers + 13), 2+1+1, or 1+1+1+1. Print the table per p, then read every observed chain
(q = 19, 23, 29, and the q = 31, 37 partial-period chains) against it: position of 13 as
predicted, doublers at admissible pairs.
"""
import numpy as np
from math import prod
dist = {11: 3, 23: 3, 29: 2, 31: 2, 59: 4, 61: 4, 37: 5, 19: 5}
for p in [5, 4, 3, 2, 1]:
    rest = [x for x in range(1, 6) if x != p]
    pairs = {d: [(a, a + d) for a in rest if a + d in rest] for d in (2, 3, 4)}
    two_two = [(x, y) for d in pairs for x in pairs[d] for e in pairs for y in pairs[e] if x < y and not set(x) & set(y)]
    print(f"13 at {p}: rest {rest}; d=2 pairs {pairs[2]} (29,31); d=3 {pairs[3]} (11,23); d=4 {pairs[4]} (59,61); 2+2 fills {two_two}")
def read_chain(q, h0, m):
    gears = [g for g in [11, 13, 17, 19, 23, 29, 31, 37] if g <= q]
    for h in range(h0, h0 + m):
        strikers = [[g for g in gears if (30*(7*h+4+p)-1) % g == 0 or (30*(7*h+4+p)+1) % g == 0] for p in range(1, 6)]
        p13 = [p+1 for p in range(5) if 13 in strikers[p]]
        pred = {1:5,2:5,3:4,4:4,5:3,6:3,7:2,8:2,9:1,10:1}.get(h % 13)
        pos = {}
        for i, ss in enumerate(strikers):
            for g in ss: pos.setdefault(g, []).append(i + 1)
        doubles = {g: v for g, v in pos.items() if len(v) == 2}
        for g, (a, b) in doubles.items(): assert g in dist and b - a == dist[g], (g, a, b)
        kind = "2+2" if len(doubles) == 2 else ("2+1+1" if len(doubles) == 1 else "1+1+1+1")
        print(f"  q={q} hole {h} (={h%13} mod 13): 13 at {p13} predicted {pred}; fill {[ss for ss in strikers]}; doublers {doubles}; type {kind}")
read_chain(19, 6, 1); read_chain(23, 4370, 2); read_chain(29, 8857125, 3); read_chain(31, 19728987, 4); read_chain(37, 3180257, 4)

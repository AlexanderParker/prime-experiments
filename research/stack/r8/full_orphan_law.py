"""Spot-check of the two load-bearing facts under the workflow's reduction (2026-09-22).

DEAD PAIR. Gear 7's block word over b = 0..6 is  - | 3 | 3 | 1 | 12 | 2 | - , so gear 7 is silent
exactly at b = 6 and b = 0 mod 7, which are ADJACENT blocks: a dead pair. Its six cells (two blocks
by three rows) must all be taken by gears >= 11.

ORPHAN LAW (claimed): no gear >= 11 has two cells one block apart whose RIGHT cell lies in row 1.
Consequence: in a dead pair the right block's row-1 cell is taken by a gear that contributes
nothing else to that pair, so N(1) - the least number of distinct gears >= 11 covering a dead
pair - is at least 4 (one orphan gear, then five cells at no more than two cells a gear).

Checked here directly from the COLUMN definition (gear g strikes column n iff g | 6n - 1 or
6n + 1, block b row p being column 5b + e_p), not from the cell rule, for every gear to 2000; and
N(1) computed exhaustively over the gears that can reach a dead pair at all.
"""
from sympy import primerange

E = (0, 2, 3)
LIMIT = 2000


def strikes(g, b, p):
    n = 5 * b + E[p - 1]
    return (6 * n - 1) % g == 0 or (6 * n + 1) % g == 0


print("ORPHAN LAW: adjacent-block cell pairs (row of left cell, row of right cell) per gear")
seen = {}
for g in primerange(11, LIMIT + 1):
    for b in range(g):
        for pl in (1, 2, 3):
            if not strikes(g, b, pl):
                continue
            for pr in (1, 2, 3):
                if strikes(g, b + 1, pr):
                    seen.setdefault((pl, pr), []).append(g)
for k in sorted(seen):
    gs = sorted(set(seen[k]))
    print(f"  left row {k[0]} -> right row {k[1]}: gears {gs[:8]}{' ...' if len(gs) > 8 else ''}")
orphans = [k for k in seen if k[1] == 1]
print(f"  pairs whose RIGHT cell is row 1: {orphans}  -> ORPHAN LAW {'HOLDS' if not orphans else 'FAILS'}")

# N(1): least number of distinct gears >= 11 covering the six cells of a dead pair.
# A dead pair is blocks (b, b+1) with b = 6 mod 7. Gear phases are free and independent (CRT), so
# a gear may be placed at any of its g phases; what matters is which subsets of the six cells one
# gear can supply. Enumerate those subsets over all gears, then find the least cover.
from itertools import combinations

subsets = set()
for g in primerange(11, LIMIT + 1):
    for r in range(g):
        s = frozenset((i, p) for i in (0, 1) for p in (1, 2, 3) if strikes(g, r + i, p))
        if s:
            subsets.add(s)
target = {(i, p) for i in (0, 1) for p in (1, 2, 3)}
subs = sorted(subsets, key=lambda x: -len(x))
best = None
for k in range(1, 7):
    for combo in combinations(subs, k):
        u = set()
        for c in combo:
            u |= c
        if u >= target:
            best = k
            break
    if best:
        break
print(f"\nN(1) = {best} (least distinct gears >= 11 covering a dead pair; claimed 4)")
print("largest cell set one gear can supply in a dead pair:", max(len(s) for s in subsets))

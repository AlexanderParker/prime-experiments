"""Deeper search of the walk network: shortest path to EVERY window twin under the carry1 rule
with all anchors, searching over states (column, certified set, previous column) so the depth
is not capped by the tree's growth; the shortest paths are the same as in the path-unique tree.
Usage: uv run python walk_network_deep.py q maxdepth
"""
import sys
from collections import deque
sys.path.insert(0, 'research/stack/r8')
from walk_network import anchors_for
from sympy import isprime


def main():
    q, maxd = int(sys.argv[1]), int(sys.argv[2])
    gears, A = anchors_for(q); G = frozenset(gears)
    opts = list(A.items())
    starts = [g for g in gears if g + 2 in G]
    twins = [n for n in range(q + 1, q * q - 1) if n % 6 == 5 and isprime(n) and isprime(n + 2)]
    best = {}
    for s in starts:
        seen = {(s, G - {s, s + 2}, None)}; dq = deque([(s, G - {s, s + 2}, None, [s])])
        while dq:
            n, C, prev, path = dq.popleft()
            if len(path) - 1 >= maxd: continue
            for a, (K, fl) in opts:
                if a == n or a == prev: continue
                Ax = n + a + 2; carried = frozenset(h for h in gears if Ax % h == 0)
                if not carried: continue
                C2 = (C & carried) | K
                if q < a and a + 2 <= q * q and C2 == G:
                    if a not in best or len(path) < len(best[a][1]): best[a] = (s, path + [a])
                    continue
                key = (a, C2, n)
                if key not in seen: seen.add(key); dq.append((a, C2, n, path + [a]))
    fam = lambda x: '/'.join(A[x][1]) if x in A else '?'
    print(f"machine {q}: window twins {len(twins)}, reached {sum(1 for t in twins if t in best)} within depth {maxd}")
    for t in twins:
        if t in best: print(f"  {t}: depth {len(best[t][1]) - 1}: " + " -> ".join(f"{x}({fam(x)})" for x in best[t][1]))
        else: print(f"  {t}: not reached")
    from collections import Counter
    print("  depth histogram:", sorted(Counter(len(v[1]) - 1 for v in best.values()).items()))


if __name__ == "__main__":
    main()

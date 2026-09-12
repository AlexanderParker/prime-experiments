"""The network of walks (owner, 2026-09-13): breadth-first from a start node (a gear pair), one
child per anchor option allowed by the rule under test, children unique to their path (a tree:
two paths landing on the same column are two nodes), a child may not be the node just left,
stop expanding a node when it reaches the destination. Then compare which rules reached
destinations most efficiently.

Machine q: gears the primes 5..q. Column n = 5 mod 6 is (n, n+2); open to gear h iff h divides
neither member. Anchors (known places) with their known-open gear sets K:
  pairs   the gear pairs (g, g+2), K = gears minus {g, g+2}; home (-1, 1), K = all gears
  squares the g-1 columns either side of g^2 not struck by g, K = {g}
  blind   column offsets 5, 10, 12, 17 mod 35 from every square g >= 11 up to q^2, K = {5, 7}
  caustic the columns after g^2 before gear h's first strike, K = {h}, merged over h
A step from column n to anchor a flips about the axis (n + a + 2)/2 and carries the gears
dividing n + a + 2. Certified set after the step: C' = (C intersect carried) union K(a): what
the walk brought and kept, plus what the landing place is known to be open to.
Destination: a node inside the window (q < a, a + 2 <= q^2) with C' = all gears (a column the
walk has certified open to every gear, hence a twin).
Rules under test (which anchor options are allowed as the next node):
  keep    the flip must carry every gear currently certified (knowledge never lost)
  carry1  the flip must carry at least one gear of the machine
  free    any anchor
plus the fixed rules: next != current, next != previous; depth cap; node cap.
Usage: uv run python walk_network.py q depth [nodecap]
"""
import sys
from collections import deque, Counter
from sympy import primerange, isprime


def anchors_for(q):
    gears = list(primerange(5, q + 1)); G = frozenset(gears); A = {}
    def add(n, K, fam):
        if n < -1 or n % 6 != 5: return
        if n in A: A[n] = (A[n][0] | K, A[n][1] + [fam] if fam not in A[n][1] else A[n][1])
        else: A[n] = (frozenset(K), [fam])
    add(-1, G, 'home')
    for g in gears:
        if g + 2 in G: add(g, G - {g, g + 2}, 'pair')
    for g in gears:
        for j in range(1, g):
            for n in (g * g - 2 - 6 * j, g * g - 2 + 6 * j):
                if n % g and (n + 2) % g: add(n, {g}, 'square')
    for g in gears:
        if g < 11: continue
        c0 = g * g - 2; i = 1
        while c0 + 6 * i + 2 <= q * q:
            if i % 35 in (5, 10, 12, 17): add(c0 + 6 * i, {5, 7}, 'blind')
            i += 1
    for g in gears:
        c0 = g * g - 2
        for h in gears:
            if h == g: continue
            i = 1
            while True:
                n = c0 + 6 * i
                if n % h == 0 or (n + 2) % h == 0: break
                add(n, {h}, 'caustic'); i += 1
    return gears, A


def run(q, depth, cap):
    gears, A = anchors_for(q); G = frozenset(gears)
    fams = {'pairs': {'home', 'pair'}, '+squares': {'home', 'pair', 'square'}, '+blind': {'home', 'pair', 'blind'},
            '+caustic': {'home', 'pair', 'caustic'}, 'all': {'home', 'pair', 'square', 'blind', 'caustic'}}
    starts = [g for g in gears if g + 2 in G]
    twins = [n for n in range(q + 1, q * q - 1) if n % 6 == 5 and isprime(n) and isprime(n + 2)]
    print(f"\n== machine {q}: gears {gears}, anchors {len(A)} ({Counter(f for v in A.values() for f in v[1])}), window twins {len(twins)}, starts {[(g, g+2) for g in starts]}, depth cap {depth}, node cap {cap}")
    print("rule | anchor set | nodes expanded | destinations (paths) | distinct destination columns | window twins reached | first depth | mean depth | example shortest path")
    for rule in ('keep', 'carry1', 'free'):
        for fname, fset in fams.items():
            opts = [(n, K) for n, (K, fl) in A.items() if set(fl) & fset]
            dests = []; expanded = 0; nodes = 0; example = None
            for s in starts:
                root = (s, G - {s, s + 2}, None, 0, [s]); dq = deque([root])
                while dq and nodes < cap:
                    n, C, parent, d, path = dq.popleft(); expanded += 1
                    if d >= depth: continue
                    for a, K in opts:
                        if a == n or a == parent: continue
                        Ax = n + a + 2; carried = frozenset(h for h in gears if Ax % h == 0)
                        if rule == 'keep' and not C <= carried: continue
                        if rule == 'carry1' and not carried: continue
                        C2 = (C & carried) | K; nodes += 1
                        if q < a and a + 2 <= q * q and C2 == G:
                            dests.append((a, d + 1, path + [a]))
                            if example is None or d + 1 < example[1]: example = (a, d + 1, path + [a])
                        else:
                            dq.append((a, C2, n, d + 1, path + [a]))
            cols = set(a for a, _, _ in dests); reached = sum(1 for t in twins if t in cols)
            fd = min((d for _, d, _ in dests), default=None); md = (sum(d for _, d, _ in dests) / len(dests)) if dests else None
            print(f"{rule} | {fname} | {expanded} | {len(dests)} | {len(cols)} | {reached}/{len(twins)} | {fd} | {md if md is None else round(md, 2)} | {example[2] if example else '-'}")
            if rule == 'carry1' and fname in ('+squares', '+caustic', 'all') and dests:
                best = {}
                for a, d, path in dests:
                    if a not in best or d < best[a][0]: best[a] = (d, path)
                fam = lambda x: '/'.join(A[x][1]) if x in A else '?'
                print("    shortest path per destination: " + "; ".join(f"{a}: " + " -> ".join(f"{x}({fam(x)})" for x in path) for a, (d, path) in sorted(best.items())))


if __name__ == "__main__":
    q = int(sys.argv[1]); depth = int(sys.argv[2]); cap = int(sys.argv[3]) if len(sys.argv) > 3 else 300000
    run(q, depth, cap)

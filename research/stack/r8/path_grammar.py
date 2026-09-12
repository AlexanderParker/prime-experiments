"""Read all the walks of the network, not the shortest ones, across machine sizes, and look
for a simple stepwise rule they share (owner, 2026-09-13).

Network as in walk_network.py (carry1 rule, all anchor families, path-unique nodes, no return
to the node just left, stop at a destination = a window column certified for every gear).
Every destination path is rewritten as a sequence of step tokens that name only structure:
  landing family (pair / home / square / blind / caustic, merged),
  the square the landing sits next to and the offset from it in columns (signed, if within
  the square's own anchor reach, else 'far'),
  the direction (up / down) and the number of gears carried.
Then, per machine: the most common token sequences among ALL destination paths, the token
sequences that reach the most distinct twins, and the sequences common to every machine.
Usage: uv run python path_grammar.py depth q1 q2 ...
"""
import sys
from collections import deque, Counter, defaultdict
from sympy import primerange, isprime
sys.path.insert(0, 'research/stack/r8')
from walk_network import anchors_for


def nearest_square(n, gears):
    best = None
    for g in gears:
        c = g * g - 2
        if abs(n - c) < 6 * g and (best is None or abs(n - c) < abs(n - best[1])): best = (g, c)
    return best


def token(prev, a, A, gears):
    K, fams = A[a]
    ns = nearest_square(a, gears)
    place = f"sq{ns[0]}{'+' if a >= ns[1] else '-'}{abs(a - ns[1]) // 6}" if ns else 'far'
    carried = sum(1 for h in gears if (prev + a + 2) % h == 0)
    return f"{'/'.join(fams)}@{place}:{'up' if a > prev else 'down'}:c{carried}"


def run(q, depth, cap=400000):
    gears, A = anchors_for(q); G = frozenset(gears)
    opts = list(A.items()); starts = [g for g in gears if g + 2 in G]
    twins = set(n for n in range(q + 1, q * q - 1) if n % 6 == 5 and isprime(n) and isprime(n + 2))
    paths = []; nodes = 0
    for s in starts:
        dq = deque([(s, G - {s, s + 2}, None, [s])])
        while dq and nodes < cap:
            n, C, prev, path = dq.popleft()
            if len(path) - 1 >= depth: continue
            for a, (K, fl) in opts:
                if a == n or a == prev: continue
                Ax = n + a + 2; carried = frozenset(h for h in gears if Ax % h == 0)
                if not carried: continue
                C2 = (C & carried) | K; nodes += 1
                if q < a and a + 2 <= q * q and C2 == G: paths.append(path + [a])
                else: dq.append((a, C2, n, path + [a]))
    seqs = Counter(); reach = defaultdict(set)
    for p in paths:
        toks = tuple(token(p[k - 1], p[k], A, gears) for k in range(1, len(p)))
        # abstract the square index away: keep only offset/family/direction/carried
        abs_toks = tuple(t.replace(t[t.index('@') + 1:t.index(':')], t[t.index('@') + 1:t.index(':')].lstrip('sq0123456789') if t[t.index('@') + 1:t.index(':')] != 'far' else 'far') for t in toks)
        seqs[abs_toks] += 1; reach[abs_toks].add(p[-1])
    print(f"\n== machine {q}: depth {depth}, destination paths {len(paths)}, distinct twins reached {len(set(p[-1] for p in paths))} of {len(twins)}, nodes {nodes}")
    print("most common step sequences (family@offset-from-the-nearest-square:direction:gears carried), with the twins they reach:")
    for s, c in seqs.most_common(8): print(f"   {c:5d} paths, {len(reach[s])} twins: " + " -> ".join(s))
    print("sequences reaching the most distinct twins:")
    for s, r in sorted(reach.items(), key=lambda kv: -len(kv[1]))[:5]: print(f"   {len(r)} twins ({seqs[s]} paths): " + " -> ".join(s))
    return set(seqs)


if __name__ == "__main__":
    depth = int(sys.argv[1]); common = None
    for q in map(int, sys.argv[2:]):
        s = run(q, depth); common = s if common is None else (common & s)
    print(f"\nstep sequences present at every machine listed: {len(common)}")
    for s in list(common)[:12]: print("   " + " -> ".join(s))

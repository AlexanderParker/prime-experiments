"""The field view of one layer: rows = the old gears (below g) and the new row g, columns = the
natural numbers from g^2 to g'^2 (g, g' consecutive primes), a mark where the gear divides the
number. Below the rows: the twin members (T), the single primes (p). Offsets are from g^2.
Then, over every layer up to gmax, location probes read off the same grids:
  - which row closes each twin-slot member of the layer (the smallest gear dividing it), as a
    map offset -> row, and how many columns each row closes alone (no other old row on it);
  - the twins' offsets from g^2 and from g'^2 (first and last twin of the layer);
  - symmetry of the twin set about the layer's midpoint and about g g' (mirrored pairs found
    against the number expected if the twins were placed at the wheel's rate);
  - row g's strikes: offsets g (m - g), m listed.
Usage: uv run python layer_fields.py show g        (print the grid for the layer g^2 .. g'^2)
       uv run python layer_fields.py probe gmax    (the probes over all layers with g <= gmax)
"""
import sys
from collections import Counter
from sympy import primerange, nextprime, isprime, factorint


def layer(g):
    gp = nextprime(g); return gp, g * g, gp * gp


def show(g):
    gp, lo, hi = layer(g)
    old = list(primerange(5, g)); rows = old + [g]
    ns = list(range(lo, hi + 1))
    print(f"layer {g}^2 = {lo} .. {gp}^2 = {hi}, {len(ns)} numbers; rows {rows}; offsets from {lo}")
    head = ''.join('|' if (n - lo) % 10 == 0 else ' ' for n in ns)
    print(f"{'offset':>6} {head}")
    for h in rows:
        line = ''.join('x' if n % h == 0 and n % 6 in (1, 5) else ('-' if n % h == 0 else '.') for n in ns)
        print(f"{h:>6} {line}")
    tw = ''.join('T' if isprime(n) and (isprime(n - 2) or isprime(n + 2)) else ('p' if isprime(n) else ' ') for n in ns)
    print(f"{'twins':>6} {tw}")
    print("x = the gear divides a twin-slot member (n = 1 or 5 mod 6); - = the gear divides an even or 3-multiple (kills nothing)")


def probe(gmax):
    first_off, last_off, mid_sym, gg_sym, alone = [], [], [], [], Counter()
    closers = Counter(); rowg = []
    for g in primerange(5, gmax + 1):
        gp, lo, hi = layer(g); old = list(primerange(5, g))
        twins = [n for n in range(lo + 1, hi - 1) if n % 6 == 5 and isprime(n) and isprime(n + 2)]  # left members
        if not twins: print(f"NO TWIN in layer {g} -> {gp}"); continue
        first_off.append((g, twins[0] - lo)); last_off.append((g, hi - twins[-1]))
        L = hi - lo
        # symmetry: pairs (t, t') with t + t' + 2 = lo + hi (columns mirrored about the midpoint) and about 2 g g'
        S = set(twins)
        # a column mirrors to a column only about a multiple of 6 (the fold has parity): centre 6c maps (t, t+2) to (12c - t - 2, 12c - t)
        best = (0, None); tot = 0
        for c in range(lo // 6 + 1, hi // 6 + 1):
            k = sum(1 for t in twins if (12 * c - t - 2) in S) // 2
            tot += k
            if k > best[0]: best = (k, 6 * c)
        ncent = hi // 6 - lo // 6
        mid_sym.append((g, best[0], best[1], tot / max(ncent, 1), len(twins), (lo + hi) // 2, g * gp))
        gg_sym.append(0)
        # closers: for each twin-slot member in the layer that some old row divides, the smallest such row; alone = only one old row on it
        for n in range(lo + 1, hi):
            if n % 6 not in (1, 5): continue
            rs = [h for h in old if n % h == 0]
            if rs:
                closers[min(rs)] += 1
                if len(rs) == 1 and n // rs[0] > 1 and all(n // rs[0] % h for h in old): alone[rs[0]] += 1
        ms = [n // g for n in range(lo + g, hi, g) if n % 6 in (1, 5) and all(n % h for h in old)]
        rowg.append((g, [m - g for m in ms]))
    n = len(first_off)
    print(f"layers {n}")
    print("first twin offset from g^2 (left member - g^2): " + ', '.join(f"{g}:{o}" for g, o in first_off[:20]))
    print(f"  max first offset {max(o for _, o in first_off)} at g = {max(first_off, key=lambda x: x[1])[0]}; layers where the first twin is within 6 g of g^2: {sum(1 for g, o in first_off if o <= 6 * g)} of {n}")
    print("last twin distance below g'^2: " + ', '.join(f"{g}:{o}" for g, o in last_off[:20]))
    print(f"  max last distance {max(o for _, o in last_off)} at g = {max(last_off, key=lambda x: x[1])[0]}")
    print("mirror centres (multiples of 6): per layer the best centre, its mirrored twin pairs, the mean pairs over all centres, twins; midpoint; g g'")
    for g, k, c, mean, nt, mid, gg in mid_sym[:24]: print(f"  {g}: best centre {c} with {k} pairs (mean {mean:.2f}, twins {nt}); midpoint {mid}, g g' {gg}, best - midpoint {c - mid if c else None}")
    print(f"  layers where the best centre's pair count is at least 3x the mean and at least 4: {sum(1 for _, k, c, mean, nt, mid, gg in mid_sym if k >= 4 and k >= 3 * mean)} of {n}")
    print(f"closing row (smallest old gear on a struck twin-slot member), share: " + ', '.join(f"{h}:{c/sum(closers.values()):.3f}" for h, c in sorted(closers.items())[:8]))
    print(f"members closed by exactly one old row (that row times a number no old row divides): " + ', '.join(f"{h}:{c}" for h, c in sorted(alone.items())[:8]))
    print("row g's strikes in its own layer, as offsets/g (= m - g, m a survivor): " + '; '.join(f"{g}:{d}" for g, d in rowg[:16]))


if __name__ == "__main__":
    if sys.argv[1] == 'show': show(int(sys.argv[2]))
    else: probe(int(sys.argv[2]))

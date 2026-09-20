"""E3: how the record run grows when a gear is added (initial segments 5..p, p <= 23, exact).

For each machine {5..p}: find every record run (longest painted run) in one period; inside it,
count the columns painted ONLY by the new gear p (the holes of the old machine that p fills), the
longest run of the old machine inside the new record, and the positions of p's teeth in the run.
Pre-registered (tree node R5.f.xxxiv.b): the new record contains at most 2 ceil(F/p) columns
painted only by p, in practice 1 or 2 for p <= 23, i.e. the new record is old runs joined at one or
two holes; the old runs inside it are at or near the old record.
"""
from math import prod

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})

def paint(gears, P):
    painted = bytearray(P)
    for g in gears:
        for t in teeth(g):
            painted[t::g] = b"\x01" * len(range(t, P, g))
    return painted

def runs(painted):
    """maximal runs of 1s over one period, treated cyclically only if the whole period is not painted"""
    P = len(painted); out = []; i = 0
    # rotate so that position 0 is unpainted, if any
    z = painted.find(b"\x00")
    if z < 0: return [(0, P)]
    rot = painted[z:] + painted[:z]
    while i < P:
        if rot[i]:
            j = i
            while j < P and rot[j]: j += 1
            out.append(((z + i) % P, j - i)); i = j
        else: i += 1
    return out

primes = [5, 7, 11, 13, 17, 19, 23]
prev = None
for k in range(1, len(primes) + 1):
    gears = primes[:k]; p = gears[-1]; P = prod(gears)
    new = paint(gears, P)
    F = max(L for _, L in runs(new))
    recs = [(s, L) for s, L in runs(new) if L == F]
    line = f"p = {p}: F = {F}, record runs in the period: {len(recs)}"
    if prev is not None:
        old_gears = gears[:-1]; Pold = prod(old_gears)
        old = paint(old_gears, Pold)
        details = []
        for s, L in recs[:6]:
            cols = [(s + i) % P for i in range(L)]
            only_p = [i for i, c in enumerate(cols) if not old[c % Pold]]
            # longest old run inside the new record
            best = cur = 0
            for c in cols:
                if old[c % Pold]: cur += 1; best = max(best, cur)
                else: cur = 0
            details.append((s, only_p, best))
        line += f"; old F = {prev}; per record run (start, positions painted only by {p}, longest old run inside): {details}"
    print(line)
    prev = F

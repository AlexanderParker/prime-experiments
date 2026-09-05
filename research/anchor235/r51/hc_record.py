"""hc_record.py -- the record's gap word under the half-column map (items 2 and 4).

For each machine m11..m31, take the record stretch(es), decompose them layer by layer,
and read every piece, every letter and every chain distance in column coordinates.
Then build the halving tree (piece -> half-column -> gears there -> their home columns
-> ...) and report its closure.

Writes results/hc_record.txt.
"""
import os, json
import numpy as np
from sympy import factorint, isprime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]

# published record starts (ends_or_middles section 1, arc_multiset R8); verified below
DEEP = {29: [200906185, 877375977], 31: [1468940242, 21844264615]}


def u_of(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def teeth(g):
    u = u_of(g)
    return (u % g, (-u) % g)


def strikes(g, k):
    return k % g in teeth(g)


def open_in(gears, k):
    return not any(strikes(g, k) for g in gears)


def records(top):
    """all record stretches (x, F) of {5..top} in one period, by full sieve."""
    gears = [g for g in GEARS if g <= top]
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        for t in teeth(g):
            blocked[t % g::g] = True
    idx = np.flatnonzero(~blocked)
    d = np.diff(idx)
    F = int(d.max())
    pos = [int(idx[i]) for i in np.flatnonzero(d == F)]
    return F, pos, P


def col_reading(v):
    """(kind, column index, the two members / halves)."""
    if v % 2 == 0:
        c = v // 2
        return "column", c, (6 * c - 1, 6 * c + 1)
    h1, h2 = (3 * v - 1) // 2, (3 * v + 1) // 2
    c = (v - 1) // 4 if v % 4 == 1 else (v + 1) // 4
    return "half", c, (h1, h2)


def gears_of(v):
    """the coupling gears of a distance v: primes >= 5 dividing 3v-1 or 3v+1."""
    s = set()
    for m in (3 * v - 1, 3 * v + 1):
        for p in factorint(m):
            if p >= 5:
                s.add(p)
    return s


def word(gears, x, F):
    """the gap word of {5..g} inside [x, x+F], as (openings, gaps)."""
    ops = [k for k in range(x, x + F + 1) if open_in(gears, k)]
    return ops, [ops[i + 1] - ops[i] for i in range(len(ops) - 1)]


def descend(cols, depth=12):
    """closure of a set of columns under c -> home columns of the prime factors
    (>= 5) of the members of c.  Returns (closure, edges, terminals)."""
    seen = set()
    edges = []
    frontier = set(cols)
    while frontier and depth:
        depth -= 1
        nxt = set()
        for c in sorted(frontier):
            if c in seen or c < 1:
                continue
            seen.add(c)
            for m in (6 * c - 1, 6 * c + 1):
                for p in factorint(m):
                    if p < 5:
                        continue
                    up = u_of(p)
                    edges.append((c, m, p, up))
                    if up not in seen:
                        nxt.add(up)
        frontier = nxt
    terminals = sorted(c for c in seen if all(
        u_of(p) == c for m in (6 * c - 1, 6 * c + 1) for p in factorint(m) if p >= 5))
    return sorted(seen), edges, terminals


def main():
    lines = []
    tot_letters = ok_letters = 0
    tot_flanks = ok_flanks = 0
    tot_chain = ok_chain = 0
    all_cols = set()

    for top in [11, 13, 17, 19, 23, 29, 31]:
        gears_all = [g for g in GEARS if g <= top]
        if top in DEEP:
            F = {29: 43, 31: 58}[top]
            pos = DEEP[top]
            # verify: x and x+F open, interior blocked
            for x in pos:
                assert open_in(gears_all, x) and open_in(gears_all, x + F)
                assert not any(open_in(gears_all, k) for k in range(x + 1, x + F))
        else:
            F, pos, _ = records(top)
            pos = pos[:2]
        lines.append("=" * 78)
        lines.append("m%d  F = %d  record starts %s" % (top, F, pos))
        for x in pos:
            lines.append("  record at x = %d" % x)
            for i, g in enumerate(gears_all):
                low = gears_all[:i + 1]
                ops, gaps = word(low, x, F)
                nxt = gears_all[i + 1] if i + 1 < len(gears_all) else None
                interior = ops[1:-1]
                struck = [k for k in interior if nxt and strikes(nxt, k)]
                # letters = differences of consecutive struck interior openings
                letters = [struck[j + 1] - struck[j] for j in range(len(struck) - 1)]
                flanks = [w for w in gaps]
                lines.append("   layer %-3d word %s" % (g, gaps))
                # every piece in column coordinates
                for w in gaps:
                    kind, c, mem = col_reading(w)
                    cg = gears_of(w) | {p for p in low if w % p == 0}
                    coupled = sorted(cg & set(low))
                    all_cols.add(c)
                    tot_flanks += 1
                    if coupled:
                        ok_flanks += 1
                    lines.append("     piece %-3d -> %-6s column %-5d members %-18s gears %-22s in M: %s"
                                 % (w, kind, c, str(mem), str(sorted(cg)), str(coupled)))
                if nxt:
                    for L in letters:
                        tot_letters += 1
                        kind, c, mem = col_reading(L)
                        # the letter must half-column (or quarter-column) to u_nxt
                        hit = (c == u_of(nxt))
                        if hit:
                            ok_letters += 1
                        lines.append("     LETTER %-3d of gear %-3d -> %-6s column %-5d (home column of %d is %d) %s"
                                     % (L, nxt, kind, c, nxt, u_of(nxt), "OK" if hit else "MISS"))
                    # chain distances: all pairs of struck interior openings
                    for a in range(len(struck)):
                        for b in range(a + 1, len(struck)):
                            dl = struck[b] - struck[a]
                            tot_chain += 1
                            good = (dl % nxt == 0) or (nxt in gears_of(dl))
                            if good:
                                ok_chain += 1
                            else:
                                lines.append("     CHAIN MISS gear %d distance %d" % (nxt, dl))
        lines.append("")

    lines.append("=" * 78)
    lines.append("letters landing on the new gear's home column: %d of %d" % (ok_letters, tot_letters))
    lines.append("pieces coupled in the machine below:          %d of %d" % (ok_flanks, tot_flanks))
    lines.append("chain distances with g | delta or g in Leg:   %d of %d" % (ok_chain, tot_chain))

    # ---- the halving tree ------------------------------------------------
    lines.append("")
    lines.append("HALVING TREE of the m29 and m31 records")
    for top, pieces in ((29, None), (31, None)):
        gears_all = [g for g in GEARS if g <= top]
        F = {29: 43, 31: 58}[top]
        x = DEEP[top][0]
        seedvals = set()
        for i, g in enumerate(gears_all):
            _, gaps = word(gears_all[:i + 1], x, F)
            seedvals.update(gaps)
        seedvals.add(F)
        seedcols = {col_reading(v)[1] for v in seedvals}
        clos, edges, terms = descend(seedcols)
        lines.append("  m%d: %d distinct piece sizes over all layers -> %d seed columns %s"
                     % (top, len(seedvals), len(seedcols), sorted(seedcols)))
        lines.append("       closure: %d columns %s" % (len(clos), clos))
        lines.append("       terminal (fixed) columns: %s" % terms)
        twin = [c for c in clos if isprime(6 * c - 1) and isprime(6 * c + 1)]
        lines.append("       twin columns in the closure: %s" % twin)
        lines.append("       terminals == twin columns: %s" % (terms == sorted(twin)))
        lines.append("       edges c -> u_p (a sample): %s" % edges[:12])

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "hc_record.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n".join(lines[-40:]))


if __name__ == "__main__":
    main()

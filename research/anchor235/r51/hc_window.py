"""hc_window.py -- the window's longest stretch under the half-column map (item 5, P11).

For each prime rung q, the window of M = {5..q} is the columns lo = q//6+1 .. (q^2-1)//6,
where an opening of M is a twin pair.  Take the longest opening-free stretch of the window,
read its length and its layer decomposition in column coordinates, and test:
  (a) is the stretch length V coupled in M?
  (b) is column V/2 itself an opening of M (a twin column) or blocked?
  (c) do the letters of the decomposition land on the home column of the gear that closes them?
Writes results/hc_window.txt.
"""
import os
import numpy as np
from sympy import primerange, factorint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def u_of(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def teeth(g):
    u = u_of(g)
    return (u % g, (-u) % g)


def blocked_array(gears, lo, hi):
    n = hi - lo
    b = np.zeros(n, dtype=bool)
    for g in gears:
        for t in teeth(g):
            b[(t - lo) % g::g] = True
    return b


def col_reading(v):
    if v % 2 == 0:
        c = v // 2
        return "column", c, (6 * c - 1, 6 * c + 1)
    h1, h2 = (3 * v - 1) // 2, (3 * v + 1) // 2
    c = (v - 1) // 4 if v % 4 == 1 else (v + 1) // 4
    return "half", c, (h1, h2)


def coupling(v, gears):
    s = set()
    for m in (3 * v - 1, 3 * v + 1):
        for p in factorint(m):
            if p >= 5:
                s.add(p)
    return sorted((s | {g for g in gears if v % g == 0}) & set(gears))


def main():
    lines = []
    rungs = [q for q in primerange(23, 1000)]
    nc_ok = nc_tot = 0
    open_half = blocked_half = 0
    let_ok = let_tot = 0
    piece_open = piece_tot = 0
    rows = []
    for q in rungs:
        gears = [g for g in primerange(5, q + 1)]
        lo, hi = q // 6 + 1, (q * q - 1) // 6 + 1
        b = blocked_array(gears, lo, hi)
        idx = np.flatnonzero(~b) + lo
        if idx.size < 2:
            continue
        d = np.diff(idx)
        V = int(d.max())
        x = int(idx[int(np.argmax(d))])
        cg = coupling(V, gears)
        nc_tot += 1
        if cg:
            nc_ok += 1
        kind, c, mem = col_reading(V)
        # is the half-column itself an opening of M?
        colopen = (c >= 1) and not any(c % g in teeth(g) for g in gears)
        if colopen:
            open_half += 1
        else:
            blocked_half += 1
        # layer decomposition of the stretch
        letters_here = []
        pieces_here = []
        for i, g in enumerate(gears):
            low = gears[:i + 1]
            ops = [k for k in range(x, x + V + 1)
                   if not any(k % gg in teeth(gg) for gg in low)]
            if len(ops) < 2:
                continue
            gaps = [ops[j + 1] - ops[j] for j in range(len(ops) - 1)]
            nxt = gears[i + 1] if i + 1 < len(gears) else None
            if nxt is None:
                continue
            interior = ops[1:-1]
            struck = [k for k in interior if k % nxt in teeth(nxt)]
            for j in range(len(struck) - 1):
                dl = struck[j + 1] - struck[j]
                a, bb = 2 * u_of(nxt), nxt - 2 * u_of(nxt)
                if dl in (a, bb):          # a genuine letter, not a pad or a wrap
                    let_tot += 1
                    kk, cc, _ = col_reading(dl)
                    if cc == u_of(nxt):
                        let_ok += 1
                    letters_here.append((nxt, dl, cc))
            if i + 1 == len(gears) - 0:
                pass
        # top-layer word: pieces of the last fusion (openings of {5..q-} inside)
        low = gears[:-1]
        ops = [k for k in range(x, x + V + 1)
               if not any(k % gg in teeth(gg) for gg in low)]
        topword = [ops[j + 1] - ops[j] for j in range(len(ops) - 1)]
        for w in topword:
            piece_tot += 1
            _, cw, _ = col_reading(w)
            if cw >= 1 and not any(cw % g in teeth(g) for g in gears):
                piece_open += 1
        rows.append((q, V, x, cg, kind, c, mem, colopen, topword))
        if q <= 61 or q in (101, 199, 401, 601, 997):
            lines.append("q=%-4d V=%-3d x=%-8d coupling gears in M %-24s  V/2 -> %-6s column %-6d %-18s %s   top word %s"
                         % (q, V, x, str(cg), kind, c, str(mem),
                            "OPEN (twin column)" if colopen else "blocked", topword))
    lines.append("")
    lines.append("rungs q = 23..997 (%d rungs)" % nc_tot)
    lines.append("  (a) the window's longest stretch length V is coupled in M: %d of %d"
                 % (nc_ok, nc_tot))
    lines.append("  (b) column V/2 is itself an opening of M (a twin column): %d of %d; blocked: %d"
                 % (open_half, nc_tot, blocked_half))
    lines.append("  (c) letters of the decomposition landing on the closing gear's home column: %d of %d"
                 % (let_ok, let_tot))
    lines.append("  (d) top-layer pieces whose half-column is an opening of M: %d of %d"
                 % (piece_open, piece_tot))
    # the uncoupled exceptions in (a)
    exc = [(q, V) for (q, V, x, cg, *_ ) in rows if not cg]
    lines.append("  uncoupled window stretches: %s" % exc)
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "hc_window.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()

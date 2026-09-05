"""hc_spectrum.py -- exact gap spectra of the machines m5..m31 over full periods,
and the half-column classification of every size up to F (P8, P9).

m5..m23 sieved whole; m29 and m31 chunked over the full period with 4 processes.
Writes results/hc_spectrum.txt and results/hc_spectrum.json.
"""
import os, json, sys
import numpy as np
from multiprocessing import Pool
from sympy import factorint

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]
CHUNK = 100_000_000


def u_of(g):
    return (g + 1) // 6 if g % 6 == 5 else (g - 1) // 6


def teeth(g):
    u = u_of(g)
    return (u % g, (-u) % g)


def _seg(args):
    """openings-gap histogram over columns [lo, hi); returns (counts dict, first, last)."""
    gears, lo, hi = args
    maxgap = 0
    counts = {}
    first = None
    last = None
    x = lo
    while x < hi:
        y = min(x + CHUNK, hi)
        n = y - x
        blocked = np.zeros(n, dtype=bool)
        for g in gears:
            for t in teeth(g):
                start = (t - x) % g
                blocked[start::g] = True
        idx = np.flatnonzero(~blocked)
        if idx.size:
            idx = idx + x
            if first is None:
                first = int(idx[0])
            if last is not None and idx.size:
                d = int(idx[0]) - last
                counts[d] = counts.get(d, 0) + 1
            if idx.size > 1:
                d = np.diff(idx)
                bc = np.bincount(d)
                nz = np.flatnonzero(bc)
                for v in nz:
                    counts[int(v)] = counts.get(int(v), 0) + int(bc[v])
            last = int(idx[-1])
        del blocked, idx
        x = y
    return counts, first, last


def spectrum(gears, nproc=4):
    period = 1
    for g in gears:
        period *= g
    bounds = [period * i // nproc for i in range(nproc + 1)]
    tasks = [(gears, bounds[i], bounds[i + 1]) for i in range(nproc)]
    if nproc == 1:
        res = [_seg(t) for t in tasks]
    else:
        with Pool(nproc) as p:
            res = p.map(_seg, tasks)
    counts = {}
    firsts, lasts = [], []
    for c, f, l in res:
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
        firsts.append(f)
        lasts.append(l)
    # stitch segment boundaries (and the wrap)
    for i in range(nproc):
        j = (i + 1) % nproc
        if lasts[i] is None or firsts[j] is None:
            continue
        d = firsts[j] - lasts[i] + (period if j == 0 else 0)
        counts[d] = counts.get(d, 0) + 1
    return counts, period


def leg(v, gears):
    return {g for g in gears if (3 * v - 1) % g == 0 or (3 * v + 1) % g == 0}


def pad(v, gears):
    return {g for g in gears if v % g == 0}


def col_reading(v):
    """the half-column of v: (kind, column, members/odd parts)."""
    if v % 2 == 0:
        c = v // 2
        return ("column", c, (6 * c - 1, 6 * c + 1))
    h1, h2 = (3 * v - 1) // 2, (3 * v + 1) // 2
    c = (v - 1) // 4 if v % 4 == 1 else (v + 1) // 4
    return ("half", c, (h1, h2))


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ["7", "11", "13", "17", "19", "23", "29", "31"]
    out = {}
    lines = []
    for top in [int(w) for w in which]:
        gears = [g for g in GEARS if g <= top]
        counts, period = spectrum(gears, nproc=4 if top >= 23 else 1)
        F = max(counts)
        tot = sum(counts.values())
        present = sorted(counts)
        absent = [v for v in range(1, F + 1) if v not in counts]
        out["m%d" % top] = {"period": period, "F": F, "n_gaps": tot,
                            "spectrum": present, "absent": absent,
                            "counts": {str(k): v for k, v in sorted(counts.items())}}
        lines.append("m%-3d gears %-30s period %-14d F = %-3d gaps %-12d"
                     % (top, str(gears), period, F, tot))
        lines.append("     spectrum: %s" % present)
        lines.append("     absent below F: %s" % absent)
        # the half-column classification of every v in [2, F]
        unc = []
        for v in range(2, F + 1):
            L = leg(v, gears)
            P = pad(v, gears)
            if not L and not P:
                unc.append(v)
        out["m%d" % top]["uncoupled"] = unc
        lines.append("     uncoupled in M (v >= 2, v <= F): %s" % unc)
        bad_fwd = [v for v in unc if v in counts]
        bad_rev = [v for v in absent if v >= 2 and v not in unc]
        lines.append("     P8 forward (uncoupled => absent): %d of %d hold, exceptions %s"
                     % (len(unc) - len(bad_fwd), len(unc), bad_fwd))
        lines.append("     P9 converse (absent => uncoupled): exceptions %s" % bad_rev)
        for v in sorted(set(unc) | set(absent)):
            if v < 2:
                continue
            kind, c, mem = col_reading(v)
            fac = [sorted(factorint(m)) for m in mem]
            lines.append("       v=%-3d %-6s column %-4d members %-16s factors %s  %s"
                         % (v, kind, c, str(mem), str(fac),
                            "ABSENT" if v not in counts else "present(%d)" % counts[v]))
        lines.append("")
        print("\n".join(lines[-(6 + len(set(unc) | set(absent))):]))
        sys.stdout.flush()
    with open(os.path.join(OUT, "hc_spectrum.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "hc_spectrum.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()

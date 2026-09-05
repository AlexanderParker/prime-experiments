"""hc_rarity2.py -- controlled version of the depletion test.

For every machine and every size v in [2,F] compute
    r(v) = count(v) / median{count(w) : |w-v| <= 4, w != v, w coupled}
and report where the UNCOUPLED sizes sit in the distribution of r over all sizes,
plus the median r by number of coupling gears.  Writes results/hc_rarity2.txt.
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def coup(v, gears):
    L = {g for g in gears if (3 * v - 1) % g == 0 or (3 * v + 1) % g == 0}
    P = {g for g in gears if v % g == 0}
    return sorted((L | P) & set(gears))


def main():
    data = json.load(open(os.path.join(OUT, "hc_spectrum.json")))
    lines = []
    cells = []
    for key in ["m7", "m11", "m13", "m17", "m19", "m23", "m29", "m31"]:
        if key not in data:
            continue
        d = data[key]
        top = int(key[1:])
        gears = [g for g in GEARS if g <= top]
        counts = {int(k): v for k, v in d["counts"].items()}
        F = d["F"]
        sizes = list(range(2, F + 1))
        cg = {v: coup(v, gears) for v in sizes}
        cnt = {v: counts.get(v, 0) for v in sizes}
        ratio = {}
        for v in sizes:
            nb = [cnt[w] for w in sizes if abs(w - v) <= 4 and w != v and cg[w]]
            if not nb:
                continue
            nb.sort()
            med = nb[len(nb) // 2]
            ratio[v] = (cnt[v] / med) if med else None
        vals = sorted(r for r in ratio.values() if r is not None)
        lines.append("%s  F=%d  sizes tested %d" % (key, F, len(vals)))
        # distribution of r for coupled sizes
        cvals = sorted(ratio[v] for v in sizes if cg[v] and ratio.get(v) is not None)
        if cvals:
            q = lambda p: cvals[min(len(cvals) - 1, int(p * len(cvals)))]
            lines.append("   coupled sizes: r quartiles %.3f / %.3f / %.3f  (n=%d), fraction with r < 0.2: %.3f"
                         % (q(.25), q(.5), q(.75), len(cvals),
                            sum(1 for x in cvals if x < 0.2) / len(cvals)))
        for v in sizes:
            if cg[v] or ratio.get(v) is None:
                continue
            r = ratio[v]
            below = sum(1 for x in cvals if x <= r)
            pct = 100.0 * below / len(cvals) if cvals else float('nan')
            lines.append("   UNCOUPLED v=%-3d count %-12d r = %.4f   percentile among coupled sizes: %.1f"
                         % (v, cnt[v], r, pct))
            cells.append((key, v, cnt[v], r, pct))
        # median r by number of coupling gears
        by = {}
        for v in sizes:
            if ratio.get(v) is None:
                continue
            by.setdefault(len(cg[v]), []).append(ratio[v])
        lines.append("   median r by number of coupling gears: " +
                     ", ".join("%d:%.3f(n=%d)" % (k, sorted(vv)[len(vv) // 2], len(vv))
                               for k, vv in sorted(by.items())))
        lines.append("")
    lines.append("SUMMARY: %d uncoupled (machine, size) cells" % len(cells))
    lines.append("  cells with r < 1 (depleted):        %d of %d"
                 % (sum(1 for c in cells if c[3] < 1), len(cells)))
    lines.append("  cells with r < 0.2:                 %d of %d"
                 % (sum(1 for c in cells if c[3] < 0.2), len(cells)))
    lines.append("  cells at or below the 10th percentile of coupled sizes: %d of %d"
                 % (sum(1 for c in cells if c[4] <= 10.0), len(cells)))
    lines.append("  cells: " + ", ".join("%s v=%d r=%.4f pct=%.1f" % (c[0], c[1], c[3], c[4]) for c in cells))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "hc_rarity2.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()

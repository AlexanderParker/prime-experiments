"""hc_rarity.py -- the graded form of the spectrum question: are uncoupled sizes
depleted rather than absent?  Uses results/hc_spectrum.json.

For every machine, for every v in [2, F], compare count(v) with the counts of its
coupled neighbours of comparable size, and rank the uncoupled sizes.
Writes results/hc_rarity.txt.
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def leg_pad(v, gears):
    L = {g for g in gears if (3 * v - 1) % g == 0 or (3 * v + 1) % g == 0}
    P = {g for g in gears if v % g == 0}
    return L, P


GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def main():
    data = json.load(open(os.path.join(OUT, "hc_spectrum.json")))
    lines = []
    for key in ["m7", "m11", "m13", "m17", "m19", "m23", "m29", "m31"]:
        if key not in data:
            continue
        d = data[key]
        top = int(key[1:])
        gears = [g for g in GEARS if g <= top]
        counts = {int(k): v for k, v in d["counts"].items()}
        F = d["F"]
        tot = d["n_gaps"]
        lines.append("%s  F=%d  gaps=%d" % (key, F, tot))
        # weight of a size: sum 2/g over its coupling gears in M (capacity to chain)
        rows = []
        for v in range(2, F + 1):
            L, P = leg_pad(v, gears)
            cg = sorted((L | P) & set(gears))
            c = counts.get(v, 0)
            rows.append((v, c, cg))
        # rank by count among all v in [2,F]
        order = sorted(rows, key=lambda r: r[1])
        rank = {r[0]: i for i, r in enumerate(order)}
        n = len(rows)
        unc = [r for r in rows if not r[2]]
        lines.append("  uncoupled sizes (no gear of M divides v or a member of column v/2):")
        for v, c, cg in unc:
            # local comparison: coupled sizes within +-4 of the same parity class
            nb = [cc for (vv, cc, gg) in rows if gg and abs(vv - v) <= 4 and vv != v]
            med = sorted(nb)[len(nb) // 2] if nb else 0
            lines.append("    v=%-3d count %-10d rank %d of %d (0 = rarest)   median count of coupled sizes within +-4: %d   ratio %.4f"
                         % (v, c, rank[v], n, med, (c / med) if med else float('nan')))
        # the five rarest realised sizes and their coupling gears
        rare = [r for r in order if r[1] > 0][:6]
        lines.append("  six rarest REALISED sizes: " +
                     ", ".join("%d:%d%s" % (v, c, cg) for v, c, cg in rare))
        absent = [r[0] for r in rows if r[1] == 0]
        lines.append("  absent: " + ", ".join("%d%s" % (v, [g for g in rows if g[0] == v][0][2])
                                              for v in absent))
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "hc_rarity.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()

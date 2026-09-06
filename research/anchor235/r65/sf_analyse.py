"""sf_analyse.py -- tables for the structured-families branch.

Reads results/families_window.tsv, families_section.tsv, islands.tsv and prints:
  (1) excess by family and q-band, window and section, with the normalised excess E_F/E_win;
  (2) the pooled excess over q >= 500 with its sampling sigma, and the q-trend;
  (3) witness thresholds (last rung with no survivor);
  (4) the survivor-density check (P8).
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

BANDS = [(23, 100), (100, 300), (300, 1000), (1000, 2500), (2500, 5000)]
ORDER = ["win", "o7", "o11", "o13", "o17", "o19",
         "b7", "b11", "b13", "b17", "b19",
         "a7", "a11", "a13", "a17", "a19", "isl"]


def load(fn):
    rows = []
    with open(os.path.join(OUT, fn)) as f:
        hdr = f.readline().split()
        for ln in f:
            p = ln.split("\t")
            rows.append(dict(fam=p[0], y=int(p[1]), q=int(p[2]), span=int(p[3]),
                             nclass=int(p[4]), mod=int(p[5]), n=int(p[6]),
                             surv=int(p[7]), rate=float(p[8])))
    return rows


def pooled(rows):
    """pooled excess = sum surv / sum (n*rate), with Poisson sigma"""
    num = sum(r["surv"] for r in rows)
    den = sum(r["n"] * r["rate"] for r in rows)
    if den <= 0:
        return float("nan"), float("nan"), 0
    e = num / den
    sig = math.sqrt(num) / den if num > 0 else 1.0 / den
    return e, sig, num


def table(rows, title):
    byfam = {}
    for r in rows:
        byfam.setdefault(r["fam"], []).append(r)
    print("\n=== %s ===" % title)
    print("%-5s %10s %8s  %s" % ("fam", "dens", "pool>=500", "  ".join("%9s" % ("q<%d" % b[1]) for b in BANDS)))
    winpool = {}
    for b in BANDS:
        winpool[b] = pooled([r for r in byfam["win"] if b[0] <= r["q"] < b[1]])[0]
    wp500 = pooled([r for r in byfam["win"] if r["q"] >= 500])[0]
    out = {}
    for fam in ORDER:
        if fam not in byfam:
            continue
        rs = byfam[fam]
        dens = rs[0]["nclass"] / rs[0]["mod"]
        e5, s5, n5 = pooled([r for r in rs if r["q"] >= 500])
        cells = []
        for b in BANDS:
            e, s, nn = pooled([r for r in rs if b[0] <= r["q"] < b[1]])
            cells.append("%9s" % ("%.3f" % e if nn > 0 else "  -"))
        print("%-5s %10.3e %8s  %s   norm=%.4f+-%.4f  twins=%d"
              % (fam, dens, "%.4f" % e5, "  ".join(cells), e5 / wp500, s5 / wp500, n5))
        out[fam] = (e5, s5, n5, e5 / wp500, s5 / wp500)
    return out, wp500


def trend(rows, fam):
    """least squares of excess against 1/ln q, per-rung pooled in 10 q-bins"""
    rs = sorted([r for r in rows if r["fam"] == fam], key=lambda r: r["q"])
    rs = [r for r in rs if r["q"] >= 300]
    if not rs:
        return None
    nb = 8
    per = max(1, len(rs) // nb)
    pts = []
    for i in range(0, len(rs), per):
        ch = rs[i:i + per]
        e, s, nn = pooled(ch)
        if nn > 0:
            qm = sum(r["q"] for r in ch) / len(ch)
            pts.append((1.0 / math.log(qm), e, s))
    if len(pts) < 3:
        return None
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    m = len(pts)
    den = m * sxx - sx * sx
    slope = (m * sxy - sx * sy) / den
    inter = (sy - slope * sx) / m
    return inter, slope, pts


def main():
    w = load("families_window.tsv")
    s = load("families_section.tsv")
    isl = load("islands.tsv")
    ow, wpw = table(w, "WINDOW: excess = survivors / (n * prod_{y<g<=q}(1-2/g))")
    os_, wps = table(s + isl, "SECTION: same, islands included")

    print("\n=== trend in 1/ln q (window, q >= 300): excess = a + b/ln q ===")
    for fam in ORDER:
        t = trend(w, fam)
        if t:
            a, b, pts = t
            print("%-5s  a=%.4f  b=%.3f   first=%.3f last=%.3f" % (fam, a, b, pts[0][1], pts[-1][1]))
    print("  (a is the extrapolated excess at q = infinity; e^{2gamma}/4 = %.5f)"
          % (math.exp(2 * 0.5772156649) / 4))

    print("\n=== witness thresholds: last rung with NO survivor ===")
    print("%-5s %14s %14s %10s %10s" % ("fam", "window last-0", "section last-0", "win n@4999", "sec n@4999"))
    for fam in ORDER:
        rw = [r for r in w if r["fam"] == fam] or [r for r in isl if r["fam"] == fam]
        rs = [r for r in s if r["fam"] == fam] or [r for r in isl if r["fam"] == fam]
        z_w = max([r["q"] for r in rw if r["surv"] == 0], default=None)
        z_s = max([r["q"] for r in rs if r["surv"] == 0], default=None)
        cw = sum(1 for r in rw if r["surv"] == 0)
        cs = sum(1 for r in rs if r["surv"] == 0)
        nw = [r["n"] for r in rw if r["q"] == 4999]
        ns = [r["n"] for r in rs if r["q"] == 4999]
        print("%-5s %14s %14s %10s %10s   (zeros: %d win, %d sec of %d rungs)"
              % (fam, "none" if z_w is None else z_w, "none" if z_s is None else z_s,
                 nw[0] if nw else "-", ns[0] if ns else "-", cw, cs, len(rw)))

    print("\n=== P8: survivors per member against the small machine's own set o_y ===")
    print("%-5s %8s %8s %8s" % ("fam", "surv/n", "o_y surv/n", "ratio"))
    oy = {}
    for r in s + w:
        pass
    for fam in ORDER:
        rs = [r for r in w if r["fam"] == fam and r["q"] >= 500]
        if not rs:
            continue
        d = sum(r["surv"] for r in rs) / sum(r["n"] for r in rs)
        yy = rs[0]["y"]
        base = [r for r in w if r["fam"] == ("o%d" % yy) and r["q"] >= 500]
        if not base:
            base = [r for r in w if r["fam"] == "win" and r["q"] >= 500]
        db = sum(r["surv"] for r in base) / sum(r["n"] for r in base)
        print("%-5s %8.5f %8.5f %8.4f" % (fam, d, db, d / db))


if __name__ == "__main__":
    main()

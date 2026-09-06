"""sf_corr.py -- correlations between located families, on the SECTIONS (which are disjoint,
so the counts are an honest independent sample: the sections at q = 23..4999 tile (23^2, 5003^2)).

For each pair of families, per rung: |F|, |G|, |F n G| in the section, the CRT-predicted
|F n G| (density product * section length), and the survivors on F n G against |F n G| * rate.
Also the 2x2 contingency of survival against membership.

Writes results/corr.txt.
"""
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
QMAX = 5000
ISLAND = (5, 10, 12, 17)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def teeth(g):
    u = pow(6, -1, g)
    return (u % g, (-u) % g)


def crt_sets(gears, sets):
    M, cur = 1, [0]
    for g, S in zip(gears, sets):
        nxt, Mg = [], M * g
        for r in cur:
            for s in S:
                x = r
                while x % g != s:
                    x += M
                nxt.append(x % Mg)
        cur, M = nxt, Mg
    return M, sorted(cur)


def main():
    sieve = primes_upto(QMAX + 10)
    rungs = [int(q) for q in np.nonzero(sieve[:QMAX + 1])[0] if q >= 23]
    nx = {}
    for q in rungs:
        p = q + 1
        while not sieve[p]:
            p += 1
        nx[q] = p
    QTOP = nx[rungs[-1]]
    isp = primes_upto(QTOP * QTOP)
    gears = [int(g) for g in np.nonzero(sieve[:QMAX + 1])[0] if g >= 5]
    cp, v = {3: 1.0}, 1.0
    for g in gears:
        v *= 1 - 2.0 / g
        cp[g] = v

    fam = {}
    for y in (7, 11, 13):
        gs = [g for g in gears if g <= y]
        P = 1
        for g in gs:
            P *= g
        fam["a%d" % y] = (y, P, [0])
        _, T = crt_sets(gs, [teeth(g) for g in gs])
        fam["b%d" % y] = (y, P, sorted({(c + 1) % P for c in T} | {(c - 1) % P for c in T}))
        _, O = crt_sets(gs, [[r for r in range(g) if r not in teeth(g)] for g in gs])
        fam["o%d" % y] = (y, P, O)

    pairs = [("a7", "isl"), ("a11", "isl"), ("a13", "isl"),
             ("b7", "isl"), ("b11", "isl"), ("b13", "isl"),
             ("o7", "o13"), ("o13", "isl"), ("a7", "b7"), ("a13", "b13")]

    acc = {p: dict(nF=0, nG=0, nI=0, sI=0, eI=0.0, sF=0, eF=0.0, sG=0, eG=0.0,
                   rungs_nonempty=0, rungs=0, all_or_none=0) for p in pairs}
    tot = dict(cells=0, twins=0)

    for q in rungs:
        qp = nx[q]
        k_s = (q * q - 1) // 6 + 1
        k_hi = (qp * qp - 1) // 6 - 1
        if k_hi < k_s:
            continue
        seg = np.arange(k_s, k_hi + 1, dtype=np.int64)
        tw = isp[6 * seg - 1] & isp[6 * seg + 1]
        k0 = (q * q - 1) // 6
        islm = np.isin((seg - k0) % 35, ISLAND)
        masks = {"isl": islm}
        for nm, (y, P, R) in fam.items():
            lut = np.zeros(P, dtype=bool)
            lut[np.array(R, dtype=np.int64)] = True
            masks[nm] = lut[seg % P]
        tot["cells"] += len(seg)
        tot["twins"] += int(tw.sum())
        for (f, g) in pairs:
            yf = fam[f][0] if f in fam else 7
            yg = fam[g][0] if g in fam else 7
            y = max(yf, yg)
            rate = cp[q] / cp[y]
            mf, mg = masks[f], masks[g]
            mi = mf & mg
            a = acc[(f, g)]
            a["rungs"] += 1
            nF, nG, nI = int(mf.sum()), int(mg.sum()), int(mi.sum())
            a["nF"] += nF; a["nG"] += nG; a["nI"] += nI
            a["sF"] += int((tw & mf).sum()); a["eF"] += nF * (cp[q] / cp[yf])
            a["sG"] += int((tw & mg).sum()); a["eG"] += nG * (cp[q] / cp[yg])
            a["sI"] += int((tw & mi).sum()); a["eI"] += nI * rate
            if nI > 0:
                a["rungs_nonempty"] += 1
            if nI == 0 or nI == min(nF, nG):
                a["all_or_none"] += 1

    with open(os.path.join(OUT, "corr.txt"), "w") as out:
        def w(s=""):
            print(s)
            out.write(s + "\n")
        w("sections tile (23^2, %d^2): %d columns, %d twins" % (QTOP, tot["cells"], tot["twins"]))
        w()
        w("%-9s %-9s %10s %10s %10s %9s %9s %9s %9s %8s" %
          ("F", "G", "|F|", "|G|", "|F n G|", "predCRT", "exc F", "exc G", "exc FnG", "sig"))
        for (f, g) in pairs:
            a = acc[(f, g)]
            pred = a["nF"] * a["nG"] / max(tot["cells"], 1)
            eF = a["sF"] / a["eF"] if a["eF"] else float("nan")
            eG = a["sG"] / a["eG"] if a["eG"] else float("nan")
            eI = a["sI"] / a["eI"] if a["eI"] else float("nan")
            sg = (math.sqrt(a["sI"]) / a["eI"]) if a["sI"] > 0 and a["eI"] else float("nan")
            w("%-9s %-9s %10d %10d %10d %9.0f %9.4f %9.4f %9.4f %8.4f" %
              (f, g, a["nF"], a["nG"], a["nI"], pred, eF, eG, eI, sg))
        w()
        w("all-or-nothing test (|F n G| is 0 or min(|F|,|G|) at every rung):")
        for (f, g) in pairs:
            a = acc[(f, g)]
            w("  %-9s %-9s  %d of %d rungs all-or-nothing; non-empty at %d rungs (%.3f)"
              % (f, g, a["all_or_none"], a["rungs"], a["rungs_nonempty"],
                 a["rungs_nonempty"] / a["rungs"]))


if __name__ == "__main__":
    main()

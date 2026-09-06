"""sf_thick.py -- the thickness table, the named-rung excesses with honest per-rung sigma,
and the exact island law for the primorial family.

(1) thickness: for each family, n against W and against the sifting range q; s = ln n / ln q;
    the dimension-2 sieve needs s > 4.27 (node 3a / face A1) for a lower bound.
(2) named rungs: per-rung excess with its own Poisson sigma (the pooled window numbers in
    sf_analyse are over OVERLAPPING windows and their sigma is not honest; sections are disjoint).
(3) the exact law: when is the primorial family a family of islands?
    A_y (k = 0 mod P_y) is an island class iff gear 5 and gear 7 are both barred at the offset
    i = -k_0, i.e. iff q^2 - 1 and q^2 + 1 are both non-(nonzero-QR) mod 5 and mod 7.
    Checked against the measured overlap.
"""
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
QMAX = 5000
ISLAND = (5, 10, 12, 17)
S_SIEVE = 4.27


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def main():
    sieve = primes_upto(QMAX + 10)
    rungs = [int(q) for q in np.nonzero(sieve[:QMAX + 1])[0] if q >= 23]
    nx = {}
    for q in rungs:
        p = q + 1
        while not sieve[p]:
            p += 1
        nx[q] = p
    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    rows = {}
    with open(os.path.join(OUT, "families_window.tsv")) as f:
        f.readline()
        for ln in f:
            p = ln.split("\t")
            rows.setdefault(p[0], {})[int(p[2])] = (int(p[6]), int(p[7]), float(p[8]))
    isl = {}
    with open(os.path.join(OUT, "islands.tsv")) as f:
        f.readline()
        for ln in f:
            p = ln.split("\t")
            isl[int(p[2])] = (int(p[6]), int(p[7]), float(p[8]))
    srow = {}
    with open(os.path.join(OUT, "families_section.tsv")) as f:
        f.readline()
        for ln in f:
            p = ln.split("\t")
            srow.setdefault(p[0], {})[int(p[2])] = (int(p[6]), int(p[7]), float(p[8]))

    NAMED = (101, 499, 997, 2503, 4999)
    order = ["win", "o7", "o11", "o13", "o17", "o19", "b7", "b11", "b13", "b17", "b19",
             "a7", "a11", "a13", "a17", "a19", "isl"]

    w("=== (1) thickness: n = family size in the window, s = ln n / ln q ===")
    w("    dimension-2 sieve gives a lower bound only for s > %.2f; needed n at q = 4999 is %.3g"
      % (S_SIEVE, 4999.0 ** S_SIEVE))
    w("%-5s %12s  %s" % ("fam", "density", "  ".join("%17s" % ("q=%d" % q) for q in NAMED)))
    for nm in order:
        src = rows.get(nm) or isl
        cells = []
        for q in NAMED:
            if nm == "isl":
                n = isl[q][0]
            else:
                n = rows[nm][q][0]
            s = math.log(n) / math.log(q) if n > 0 else float("-inf")
            cells.append("%17s" % ("n=%d s=%.2f" % (n, s) if n > 0 else "n=0 s=-inf"))
        d = (rows[nm][4999][0] / rows["win"][4999][0]) if nm != "isl" else (4.0 / 35.0)
        w("%-5s %12.4e  %s" % (nm, d, "  ".join(cells)))
    w("  (isl is measured on the SECTION, the only place it is defined)")

    w()
    w("=== (2) per-rung excess with its own sigma (window | section) ===")
    w("%-5s  %s" % ("fam", "  ".join("%21s" % ("q=%d" % q) for q in NAMED)))
    for nm in order:
        cells = []
        for q in NAMED:
            if nm == "isl":
                n, sv, r = isl[q]
                nw, svw, rw = 0, 0, 0.0
            else:
                nw, svw, rw = rows[nm][q]
                n, sv, r = srow[nm][q]
            ew = svw / (nw * rw) if nw * rw > 0 else float("nan")
            sw = math.sqrt(svw) / (nw * rw) if svw > 0 else float("nan")
            es = sv / (n * r) if n * r > 0 else float("nan")
            cells.append("%21s" % ("%.2f+-%.2f|%.2f+-%.2f" % (ew, sw, es, math.sqrt(sv) / (n * r) if sv > 0 else 9.99)
                                   if nm != "isl" else " -   |%.2f+-%.2f" % (es, math.sqrt(sv) / (n * r) if sv > 0 else 9.99)))
        w("%-5s  %s" % (nm, "  ".join(cells)))

    w()
    w("=== (3) exact law: when are the primorial multiples an island class? ===")
    qr5 = {(x * x) % 5 for x in range(1, 5)}
    qr7 = {(x * x) % 7 for x in range(1, 7)}

    def barred(gq, x):
        # gear g barred at offset i iff neither -6i nor 2-6i is a NONZERO quadratic residue
        s = qr5 if gq == 5 else qr7
        a, b = x % gq, (x + 2) % gq
        return (a == 0 or a not in s) and (b == 0 or b not in s)

    ok5 = ok7 = ok = 0
    meas = 0
    pred_by_class = {}
    for q in rungs:
        k0 = (q * q - 1) // 6
        i = (-k0) % 35
        x = (-6 * i)  # = q^2 - 1 mod 35
        b5, b7 = barred(5, x), barred(7, x)
        ok5 += b5
        ok7 += b7
        pr = b5 and b7
        ok += pr
        actual = (i % 35) in ISLAND
        assert pr == actual, (q, i, b5, b7, actual)
        meas += actual
        pred_by_class.setdefault(q % 7, [0, 0])
        pred_by_class[q % 7][0] += 1
        pred_by_class[q % 7][1] += actual
    w("  gear 5 barred at the column-0 offset: %d of %d rungs" % (ok5, len(rungs)))
    w("  gear 7 barred at the column-0 offset: %d of %d rungs" % (ok7, len(rungs)))
    w("  both (the family is an island family): %d of %d = %.4f  [predicted 1/3 = 0.3333]"
      % (ok, len(rungs), ok / len(rungs)))
    w("  by q mod 7: " + ", ".join("%d:%d/%d" % (c, v[1], v[0]) for c, v in sorted(pred_by_class.items())))
    w("  0 disagreements between the QR rule and the residue test, %d rungs" % len(rungs))

    w()
    w("=== (4) how many of B_7's seven classes mod 35 are island classes, by rung ===")
    B7 = [0, 2, 5, 7, 28, 30, 33]
    hist = {}
    for q in rungs:
        k0 = (q * q - 1) // 6
        c = sum(1 for r in B7 if ((r - k0) % 35) in ISLAND)
        hist[c] = hist.get(c, 0) + 1
    w("  overlap size: " + ", ".join("%d classes at %d rungs" % (k, v) for k, v in sorted(hist.items())))
    w("  mean %.4f against the independent prediction 7 * 4/35 = 0.800"
      % (sum(k * v for k, v in hist.items()) / len(rungs)))

    with open(os.path.join(OUT, "thickness.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

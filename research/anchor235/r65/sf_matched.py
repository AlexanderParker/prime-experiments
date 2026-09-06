"""sf_matched.py -- the matched-rate two-sample tests of section 5.1.

Each located family is compared with the REST of the small machine's own open set O_y, inside the
same disjoint sections. Every class of O_y carries the identical fair rate prod_{y<g<=q}(1-2/g),
so the comparison is a plain ratio of twins per member with no model in it.

Also the islands against the corridor's non-island classes.

Writes results/matched.txt.
"""
import math
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
QMAX = 5000
ISL = (5, 10, 12, 17)


def sv(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** .5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def teeth(g):
    u = pow(6, -1, g)
    return (u % g, (-u) % g)


def crt(gs, sets):
    M, cur = 1, [0]
    for g, S in zip(gs, sets):
        nx2, Mg = [], M * g
        for r in cur:
            for s in S:
                x = r
                while x % g != s:
                    x += M
                nx2.append(x % Mg)
        cur, M = nx2, Mg
    return M, sorted(cur)


def main():
    sp = sv(QMAX + 10)
    rungs = [int(q) for q in np.nonzero(sp[:QMAX + 1])[0] if q >= 23]
    nx = {}
    for q in rungs:
        p = q + 1
        while not sp[p]:
            p += 1
        nx[q] = p
    isp = sv(nx[rungs[-1]] ** 2)
    gears = [5, 7, 11, 13, 17, 19]
    fams = {}
    for y in (7, 11, 13, 17, 19):
        gs = [g for g in gears if g <= y]
        P = 1
        for g in gs:
            P *= g
        _, T = crt(gs, [teeth(g) for g in gs])
        _, O = crt(gs, [[r for r in range(g) if r not in teeth(g)] for g in gs])
        fams["a%d" % y] = (y, P, [0], O)
        fams["b%d" % y] = (y, P, sorted({(c + 1) % P for c in T} | {(c - 1) % P for c in T}), O)
    lines = []

    def w(s):
        print(s)
        lines.append(s)

    E35 = sorted(r for r in range(35) if r % 5 not in (1, 4) and r % 7 not in (1, 6))
    ni = si = nn = sn = 0
    for q in rungs:
        qp = nx[q]
        ks, kh = (q * q - 1) // 6 + 1, (qp * qp - 1) // 6 - 1
        if kh < ks:
            continue
        seg = np.arange(ks, kh + 1, dtype=np.int64)
        k0 = (q * q - 1) // 6
        tw = isp[6 * seg - 1] & isp[6 * seg + 1]
        inE = np.isin(seg % 35, E35)
        isl = np.isin((seg - k0) % 35, ISL)
        a, b = inE & isl, inE & ~isl
        ni += int(a.sum()); si += int((tw & a).sum())
        nn += int(b.sum()); sn += int((tw & b).sum())
    ri, rn = si / ni, sn / nn
    se = math.sqrt(si / ni ** 2 + sn / nn ** 2) / rn
    w("islands vs corridor non-islands (E_35 has 15 classes, 4 of them islands)")
    w("  island n=%d surv=%d rate=%.6f ; rest n=%d surv=%d rate=%.6f" % (ni, si, ri, nn, sn, rn))
    w("  ratio %.5f +- %.5f  (%.2f sigma from 1)" % (ri / rn, se, (ri / rn - 1) / se))
    w("")
    w("%-5s %10s %10s %12s %12s %s" % ("fam", "n_F", "n_Oy-F", "rate F", "rate rest", "ratio"))
    for nm in ("a7", "b7", "a11", "b11", "a13", "b13", "a17", "b17", "a19", "b19"):
        y, P, R, O = fams[nm]
        lutF = np.zeros(P, bool); lutF[np.array(R, dtype=np.int64)] = True
        lutO = np.zeros(P, bool); lutO[np.array(O, dtype=np.int64)] = True
        nF = sF = nR = sR = 0
        for q in rungs:
            qp = nx[q]
            ks, kh = (q * q - 1) // 6 + 1, (qp * qp - 1) // 6 - 1
            if kh < ks:
                continue
            seg = np.arange(ks, kh + 1, dtype=np.int64)
            tw = isp[6 * seg - 1] & isp[6 * seg + 1]
            m = seg % P
            f = lutF[m]; o = lutO[m] & ~f
            nF += int(f.sum()); sF += int((tw & f).sum())
            nR += int(o.sum()); sR += int((tw & o).sum())
        rF, rR = sF / nF, sR / nR
        se = math.sqrt(sF / nF ** 2 + sR / nR ** 2) / rR
        w("%-5s %10d %10d %12.6f %12.6f  %.5f +- %.5f (%.2f sig)"
          % (nm, nF, nR, rF, rR, rF / rR, se, (rF / rR - 1) / se))
    with open(os.path.join(OUT, "matched.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

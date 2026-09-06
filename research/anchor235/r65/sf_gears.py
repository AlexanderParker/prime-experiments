"""sf_gears.py -- the take of each gear on each located family, against 2/g (branch item 4).

(A) named rungs, WINDOW: for families win, o7, o13, b7, b13, a7, a13 sieve the family's window
    members gear by gear ascending and record fresh_g / N_cur(g) against 2/g, binned by
    t = ln g / ln q' (7b's coordinate).
(B) SECTIONS pooled over every rung q >= 101: the same for the islands and for o13, b7, a7 (the
    islands live only past q^2, so this is the only pooling that includes them).

Writes results/gears.txt.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
QMAX = 5000
NAMED = (997, 2503, 4999)
ISLAND = (5, 10, 12, 17)
BINS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0001]


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


def sieve_family(ks, gearlist):
    """returns list of (g, N_cur, fresh) sieving ks by the gears in order"""
    cur = ks
    out = []
    for g in gearlist:
        n = len(cur)
        if n == 0:
            out.append((g, 0, 0))
            continue
        r = cur % g
        u = pow(6, -1, g)
        hit = (r == u % g) | (r == (-u) % g)
        f = int(hit.sum())
        out.append((g, n, f))
        cur = cur[~hit]
    return out, cur


def main():
    sieve = primes_upto(QMAX + 10)
    gears = [int(g) for g in np.nonzero(sieve[:QMAX + 1])[0] if g >= 5]
    rungs = [int(q) for q in np.nonzero(sieve[:QMAX + 1])[0] if q >= 23]
    nx = {}
    for q in rungs:
        p = q + 1
        while not sieve[p]:
            p += 1
        nx[q] = p

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
    fam["win"] = (3, 1, [0])

    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    w("=== (A) window at named rungs: take = fresh_g/N_cur(g) divided by 2/g, binned by ln g/ln q' ===")
    w("%-5s %6s  %s" % ("fam", "q", "  ".join("%11s" % ("t<%.2f" % b) for b in BINS[1:])))
    for q in NAMED:
        qp = nx[q]
        k_lo = (q + 1) // 6 + 1
        k_hi = (qp * qp - 1) // 6 - 1
        gl = [g for g in gears if g <= q]
        lq = np.log(qp)
        for nm in ("win", "o7", "o13", "b7", "b13", "a7", "a13"):
            y, P, R = fam[nm]
            if P == 1:
                ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
            else:
                lut = np.zeros(P, dtype=bool)
                lut[np.array(R, dtype=np.int64)] = True
                seg = np.arange(k_lo, k_hi + 1, dtype=np.int64)
                ks = seg[lut[seg % P]]
            rows, surv = sieve_family(ks, [g for g in gl if g > y])
            acc = [[0.0, 0.0] for _ in BINS[1:]]
            for g, n, f in rows:
                t = np.log(g) / lq
                for i in range(len(BINS) - 1):
                    if BINS[i] <= t < BINS[i + 1]:
                        acc[i][0] += f
                        acc[i][1] += n * 2.0 / g
                        break
            cells = ["%11s" % ("%.4f" % (a[0] / a[1]) if a[1] > 0 else "-") for a in acc]
            w("%-5s %6d  %s   n0=%d surv=%d" % (nm, q, "  ".join(cells), len(ks), len(surv)))
        w()

    w("=== (B) sections pooled over rungs q >= 101 (islands included) ===")
    fams_b = ("isl", "o13", "b7", "a7")
    acc = {nm: [[0.0, 0.0] for _ in BINS[1:]] for nm in fams_b}
    n0 = {nm: 0 for nm in fams_b}
    sv = {nm: 0 for nm in fams_b}
    for q in rungs:
        if q < 101:
            continue
        qp = nx[q]
        k_s = (q * q - 1) // 6 + 1
        k_hi = (qp * qp - 1) // 6 - 1
        if k_hi < k_s:
            continue
        seg = np.arange(k_s, k_hi + 1, dtype=np.int64)
        k0 = (q * q - 1) // 6
        lq = np.log(qp)
        gl = [g for g in gears if g <= q]
        for nm in fams_b:
            if nm == "isl":
                y = 7
                ks = seg[np.isin((seg - k0) % 35, ISLAND)]
            else:
                y, P, R = fam[nm]
                lut = np.zeros(P, dtype=bool)
                lut[np.array(R, dtype=np.int64)] = True
                ks = seg[lut[seg % P]]
            rows, surv = sieve_family(ks, [g for g in gl if g > y])
            n0[nm] += len(ks)
            sv[nm] += len(surv)
            a = acc[nm]
            for g, n, f in rows:
                t = np.log(g) / lq
                for i in range(len(BINS) - 1):
                    if BINS[i] <= t < BINS[i + 1]:
                        a[i][0] += f
                        a[i][1] += n * 2.0 / g
                        break
    w("%-5s  %s" % ("fam", "  ".join("%11s" % ("t<%.2f" % b) for b in BINS[1:])))
    for nm in fams_b:
        cells = ["%11s" % ("%.4f" % (a[0] / a[1]) if a[1] > 0 else "-") for a in acc[nm]]
        w("%-5s  %s   n0=%d surv=%d" % (nm, "  ".join(cells), n0[nm], sv[nm]))

    with open(os.path.join(OUT, "gears.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

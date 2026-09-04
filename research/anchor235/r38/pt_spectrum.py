"""W.t - spectra and correlations (T9).

A. the discrete Fourier spectrum of the section's blocked string, machine {5..q}: the top
   peaks and the gear line j/g each belongs to; the energy in the gear lines.
B. the autocorrelation of the same string at lags 1..40.
C. is the position of q^2 distinguished?  The local opening count in windows of 50, 200 and
   1000 columns centred on k_0, as a percentile of the same statistic over the section.
D. the path's own correlation structure: over all q, the correlation of blocked(k_0+i) with
   blocked(k_0+i+h) - the transform of the path in the offset coordinate.

Writes results/pt_spectrum.txt.
Usage: uv run python research/anchor235/r38/pt_spectrum.py [--Q 20000] [--nsample 24]
"""
import argparse
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def sieve_flags(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def build(q, p, qn, gears):
    k0 = (q * q - 1) // 6
    lo = (p * p - 1) // 6 + 1
    hi = (qn * qn - 1) // 6
    n = hi - lo + 1
    om = np.zeros(n, dtype=np.uint8)
    for g in gears:
        c = pow(6, -1, g)
        for r in (c % g, (-c) % g):
            st = (r - lo) % g
            om[st:: g] += 1
    return lo, k0, om > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=20000)
    ap.add_argument("--nsample", type=int, default=24)
    ap.add_argument("--I", type=int, default=128)
    a = ap.parse_args()
    Q, I = a.Q, a.I
    log = open(os.path.join(OUT, "pt_spectrum.txt"), "w")

    def say(*xs):
        s = " ".join(str(x) for x in xs)
        print(s)
        log.write(s + "\n")

    fl = sieve_flags(4 * Q + 200)
    plist = [int(x) for x in np.flatnonzero(fl)]
    gears_all = [x for x in plist if x >= 5]
    qs = [x for x in gears_all if x <= Q]
    say("gears 5..%d: %d" % (Q, len(qs)))

    # sample of q, geometrically spread
    tgt = np.unique(np.round(np.geomspace(101, Q, a.nsample)).astype(int))
    sample = sorted({min(qs, key=lambda x: abs(x - t)) for t in tgt})

    say("")
    say("=== A. the section's Fourier spectrum ===")
    say("machine {5..q}, blocked string on the section (p^2, q^2], mean removed.")
    say("Top peaks |A(m)|/N with the nearest gear line j/g (|m/N - j/g| in units of 1/N):")
    say("")
    tot_gear_energy = []
    for q in sample:
        ip = plist.index(q)
        p, qn = plist[ip - 1], plist[ip + 1]
        gears = [g for g in gears_all if g <= q]
        lo, k0, blk = build(q, p, qn, gears)
        sec = blk[: k0 - lo + 1].astype(np.float64)
        N = sec.size
        x = sec - sec.mean()
        A = np.fft.rfft(x)
        mag = np.abs(A) / N
        order = np.argsort(-mag)[:6]
        lines = []
        for m in order:
            f = m / N
            best = None
            for g in gears[:12]:
                for j in range(1, g):
                    dd = abs(f - j / g)
                    if best is None or dd < best[0]:
                        best = (dd, j, g)
            lines.append("m=%d |A|/N=%.5f ~ %d/%d (off %.2f/N)"
                         % (m, mag[m], best[1], best[2], best[0] * N))
        # energy in the small-gear lines (within 2/N of some j/g, g <= 43)
        fs = np.arange(mag.size) / N
        near = np.zeros(mag.size, dtype=bool)
        for g in [g for g in gears if g <= 43]:
            for j in range(1, g):
                near |= np.abs(fs - j / g) < 2.0 / N
        e_all = float((mag[1:] ** 2).sum())
        e_gear = float((mag[1:][near[1:]] ** 2).sum())
        tot_gear_energy.append(e_gear / e_all)
        say(" q=%-6d N=%-7d  gear-line energy share %.4f" % (q, N, e_gear / e_all))
        for s in lines[:3]:
            say("      " + s)
    say("gear-line (g <= 43) energy share over the sample: min %.4f, median %.4f, max %.4f"
        % (min(tot_gear_energy), float(np.median(tot_gear_energy)), max(tot_gear_energy)))

    say("")
    say("=== B. autocorrelation of the section's blocked string ===")
    say("lag:      " + " ".join("%6d" % h for h in range(1, 13)))
    for q in sample[::4]:
        ip = plist.index(q)
        p, qn = plist[ip - 1], plist[ip + 1]
        gears = [g for g in gears_all if g <= q]
        lo, k0, blk = build(q, p, qn, gears)
        sec = blk[: k0 - lo + 1].astype(np.float64)
        x = sec - sec.mean()
        v = float((x * x).mean())
        ac = [float((x[:-h] * x[h:]).mean()) / v for h in range(1, 13)]
        say(" q=%-6d " % q + " ".join("%6.3f" % t for t in ac))

    say("")
    say("=== C. is the position of q^2 distinguished? local opening count ===")
    say("window W centred on k_0, opening count, as a percentile of the same window statistic")
    say("slid over the whole section:")
    for W in (50, 200, 1000):
        pcts = []
        for q in sample:
            ip = plist.index(q)
            p, qn = plist[ip - 1], plist[ip + 1]
            gears = [g for g in gears_all if g <= q]
            lo, k0, blk = build(q, p, qn, gears)
            op = (~blk).astype(np.float64)
            if op.size < 3 * W:
                continue
            cs = np.concatenate([[0.0], np.cumsum(op)])
            i0 = k0 - lo
            if i0 - W // 2 < 0 or i0 + W - W // 2 >= op.size:
                continue
            here = cs[i0 + W - W // 2] - cs[i0 - W // 2]
            starts = np.arange(0, (k0 - lo) - W)
            vals = cs[starts + W] - cs[starts]
            pcts.append(float((vals < here).mean()))
        say("   W=%-5d percentile of the window at k_0: median %.3f, min %.3f, max %.3f"
            % (W, float(np.median(pcts)), min(pcts), max(pcts)))

    say("")
    say("=== D. the path's own correlation in the offset coordinate ===")
    say("over every prime q in 5..%d, B(i) = [column k_0 + i blocked]; correlation of B(i)"
        % Q)
    say("with B(i+h), averaged over i in [-%d, %d] and over q:" % (I, I))
    G = np.array(gears_all, dtype=np.int64)
    M = np.zeros((len(qs), 2 * I + 1), dtype=np.int8)
    for t, q in enumerate(qs):
        ng = int(np.searchsorted(G, q, side="right"))
        gg = G[:ng]
        r = (q * q) % gg
        iv = np.array([pow(6, -1, int(g)) for g in gg], dtype=np.int64)
        i_lo = ((2 - r) * iv) % gg
        i_hi = ((-r) * iv) % gg
        row = np.zeros(2 * I + 1, dtype=np.int8)
        for base in (i_lo, i_hi):
            first = base - gg * ((base + I) // gg)
            m = 0
            while True:
                v = first + m * gg
                sel = (v >= -I) & (v <= I)
                if not sel.any():
                    break
                row[(v[sel] + I)] = 1
                m += 1
        M[t] = row
    X = M.astype(np.float64)
    X = X - X.mean()
    v = float((X * X).mean())
    say("lag h:    " + " ".join("%6d" % h for h in range(1, 16)))
    ac = []
    for h in range(1, 16):
        ac.append(float((X[:, :-h] * X[:, h:]).mean()) / v)
    say("corr:     " + " ".join("%6.3f" % t for t in ac))
    say("blocked fraction of the offsets: %.4f" % M.mean())
    say("(the lag-5 and lag-10 peaks are gear 5's two-and-one pattern; the alternating sign at")
    say(" small lags is the anchor, not the higher gears.)")
    log.close()


if __name__ == "__main__":
    main()

"""W.t - the offset transform of the path: which gears CAN strike each column.

The members of column k_0 + i (k_0 = (q^2-1)/6) are  q^2 + 6i - 2  and  q^2 + 6i.
So gear g strikes that column iff  q^2 = 2 - 6i  or  q^2 = -6i   (mod g),
i.e. iff one of the two offset constants is a quadratic residue mod g AND q sits on one of
its square roots.  The admissible SET depends on the offset alone - q enters only as a phase.

Measured here, for every prime q in [5, Q] and offsets i in [-I, I]:
  A. the striker set of column k_0 + i against the predicted admissible set (exception count);
  B. the size of the admissible set by offset (3/4 of the machine generically, 1/2 at i = 0);
  C. i = 0 sharpened: every striker of the walk's first column except q itself is g = +-1 mod 8;
  D. realisation: is every admissible (g, i) pair actually used by some q (no further law);
  E. the depth mean and variance by offset, and the opening density by offset, against the
     machine's own generic values - what the filter does to the shape of the path.
  F. the forced-composite columns k_0 - 6t^2 (member (q-6t)(q+6t)) and their forward absence.

Writes results/pt_qr.txt.
Usage: uv run python research/anchor235/r38/pt_qr.py [--Q 20000] [--I 128]
"""
import argparse
import os
from collections import Counter

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=20000)
    ap.add_argument("--I", type=int, default=128)
    a = ap.parse_args()
    Q, I = a.Q, a.I
    log = open(os.path.join(OUT, "pt_qr.txt"), "w")

    def say(*xs):
        s = " ".join(str(x) for x in xs)
        print(s)
        log.write(s + "\n")

    fl = sieve_flags(Q + 100)
    gears = [int(x) for x in np.flatnonzero(fl) if x >= 5 and x <= Q]
    G = np.array(gears, dtype=np.int64)
    say("gears 5..%d: %d;  offsets i in [%d, %d]" % (Q, len(gears), -I, I))

    # ---------------- admissibility table adm[i][g]: can g EVER strike offset i (q != g)?
    offs = list(range(-I, I + 1))
    adm = {}
    adm_lo = {}     # via the lower member q^2 + 6i - 2
    adm_hi = {}     # via the upper member q^2 + 6i
    for i in offs:
        alo = np.zeros(len(gears), dtype=bool)
        ahi = np.zeros(len(gears), dtype=bool)
        for j, g in enumerate(gears):
            e = (g - 1) // 2
            v1 = (2 - 6 * i) % g
            v2 = (-6 * i) % g
            alo[j] = v1 != 0 and pow(v1, e, g) == 1
            ahi[j] = v2 != 0 and pow(v2, e, g) == 1
        adm_lo[i] = alo
        adm_hi[i] = ahi
        adm[i] = alo | ahi

    say("")
    say("=== B. size of the admissible set by offset (fraction of the %d gears) ===" % len(gears))
    say(" offset i   via lower member   via upper member   union      (union expected 3/4)")
    for i in [-24, -6, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 12, 24, 96]:
        if i in adm:
            say("   %5d      %.4f             %.4f          %.4f"
                % (i, adm_lo[i].mean(), adm_hi[i].mean(), adm[i].mean()))
    fr = np.array([adm[i].mean() for i in offs if i != 0])
    say("over all %d nonzero offsets: union fraction min %.4f, median %.4f, max %.4f"
        % (len(fr), fr.min(), np.median(fr), fr.max()))
    say("at i = 0: lower member (q^2 - 2) admissible fraction %.4f (gears g = +-1 mod 8),"
        % adm_lo[0].mean())
    say("          upper member (q^2) admissible only for g = q, fraction %.4f"
        % adm_hi[0].mean())
    m8 = np.array([g % 8 in (1, 7) for g in gears])
    say("          gears with g = +-1 mod 8: %.4f;  agreement with the QR test: %s"
        % (m8.mean(), bool((m8 == adm_lo[0]).all())))

    # root-count weights: expected depth at offset i if q were uniform mod each gear
    rho = {}
    for i in offs:
        w = np.zeros(len(gears))
        for j, g in enumerate(gears):
            v1 = (2 - 6 * i) % g
            v2 = (-6 * i) % g
            r1 = 1 if v1 == 0 else (2 if adm_lo[i][j] else 0)
            r2 = 1 if v2 == 0 else (2 if adm_hi[i][j] else 0)
            w[j] = (r1 + r2) / (g - 1)
        rho[i] = np.cumsum(w)
    rhov = {i: np.cumsum(np.array([
        (((1 if (2 - 6 * i) % g == 0 else (2 if adm_lo[i][j] else 0))
          + (1 if (-6 * i) % g == 0 else (2 if adm_hi[i][j] else 0))) / (g - 1))
        * (1 - ((1 if (2 - 6 * i) % g == 0 else (2 if adm_lo[i][j] else 0))
                + (1 if (-6 * i) % g == 0 else (2 if adm_hi[i][j] else 0))) / (g - 1))
        for j, g in enumerate(gears)])) for i in offs}

    # ---------------- A/C/D/E: sweep over q
    qs = [g for g in gears if g >= 5]
    INV6 = np.array([pow(6, -1, int(g)) for g in G], dtype=np.int64)
    ap_bad = 0
    ap_checks = 0
    sep_bad = 0
    pred_sum = np.zeros(len(offs))
    predv_sum = np.zeros(len(offs))
    five_hit = np.zeros(len(offs))
    five_hit_A = np.zeros(len(offs))
    five_hit_B = np.zeros(len(offs))
    nA = nB = 0
    exc_A = 0
    exc_C = 0
    checks_A = 0
    strikers_C = 0
    used = {i: np.zeros(len(gears), dtype=bool) for i in offs}
    dep_sum = np.zeros(len(offs))
    dep_sq = np.zeros(len(offs))
    dep_n = np.zeros(len(offs))
    open_n = np.zeros(len(offs))
    dep_sumR = np.zeros(len(offs)); dep_sqR = np.zeros(len(offs))
    dep_nR = np.zeros(len(offs)); open_nR = np.zeros(len(offs))
    predv_sumR = np.zeros(len(offs)); pred_sumR = np.zeros(len(offs))
    QMIN = Q // 2
    idx = {i: t for t, i in enumerate(offs)}
    for qi, q in enumerate(qs):
        ng = int(np.searchsorted(G, q, side="right"))
        gg = G[:ng]
        r = (q * q) % gg
        # the two arithmetic progressions in the OFFSET coordinate, one per member:
        #   g strikes offset i  iff  i = i_lo or i_hi (mod g),
        #   i_lo = (2 - q^2) 6^-1,  i_hi = -q^2 6^-1,  i_lo - i_hi = 2*6^-1 = d_g (mod g)
        iv = INV6[:ng]
        i_lo = ((2 - r) * iv) % gg
        i_hi = ((-r) * iv) % gg
        dg = (2 * iv) % gg
        if not (((i_lo - i_hi) % gg == dg).all()):
            sep_bad += 1
        for t, i in enumerate(offs):
            pred_sum[t] += rho[i][ng - 1]
            predv_sum[t] += rhov[i][ng - 1]
        k0m5 = ((q * q - 1) // 6) % 5
        if q > 5:
            clsA = (k0m5 == 0)
            nA += clsA
            nB += (not clsA)
            for t, i in enumerate(offs):
                h5 = ((k0m5 + i) % 5) in (1, 4)
                five_hit[t] += h5
                if clsA:
                    five_hit_A[t] += h5
                else:
                    five_hit_B[t] += h5
        for i in offs:
            lo = (r + 6 * i - 2) % gg == 0
            hi = (r + 6 * i) % gg == 0
            hit = lo | hi
            # the offset-AP form of the same statement
            ap = (((i - i_lo) % gg) == 0) | (((i - i_hi) % gg) == 0)
            ap_checks += ng
            if not (ap == hit).all():
                ap_bad += int((ap != hit).sum())
            t = idx[i]
            dpt = int(hit.sum())
            dep_sum[t] += dpt
            dep_sq[t] += dpt * dpt
            dep_n[t] += 1
            if dpt == 0:
                open_n[t] += 1
            if q >= QMIN:
                dep_sumR[t] += dpt; dep_sqR[t] += dpt * dpt; dep_nR[t] += 1
                pred_sumR[t] += rho[i][ng - 1]; predv_sumR[t] += rhov[i][ng - 1]
                if dpt == 0:
                    open_nR[t] += 1
            # A: predicted admissibility (q itself is the exception at i = 0)
            pred = adm[i][:ng].copy()
            pred |= (gg == q)
            checks_A += ng
            bad = hit & ~pred
            if bad.any():
                exc_A += int(bad.sum())
            used[i][:ng] |= hit
            if i == 0:
                s = gg[hit]
                for v in s:
                    v = int(v)
                    if v == q:
                        continue
                    strikers_C += 1
                    if v % 8 not in (1, 7):
                        exc_C += 1
        if qi % 200 == 0:
            print("  ... q=%d (%d/%d)" % (q, qi, len(qs)), flush=True)

    say("")
    say("=== A. the striker set against the predicted admissible set ===")
    say("(gear, offset) pairs checked: %d over %d walks and %d offsets"
        % (checks_A, len(qs), len(offs)))
    say("strikes by a gear the offset bars: %d" % exc_A)

    say("")
    say("=== A2. the path in the OFFSET coordinate: two progressions per gear ===")
    say("gear g strikes offset i iff i = i_lo or i_hi (mod g), with")
    say("   i_lo = (2 - q^2) 6^-1 mod g,  i_hi = -q^2 6^-1 mod g,  i_lo - i_hi = d_g,")
    say("so the whole path is the covering of [0, L) by two APs per gear, difference g,")
    say("separation d_g, phase set by q^2 mod g alone.")
    say("(gear, offset) pairs where the AP form disagrees with divisibility: %d of %d"
        % (ap_bad, ap_checks))
    say("walks where i_lo - i_hi != d_g for some gear: %d of %d" % (sep_bad, len(qs)))
    say("PHASE RESTRICTION: the phase of both progressions is a function of q^2 mod g, and")
    say("q^2 is a square mod every gear, so the phase vector of the walk lies in the image of")
    say("the squaring map - one of 2^-pi(q) of the phase space (pi(q) = %d at q = %d)."
        % (len(gears), Q))

    say("")
    say("=== C. the walk's first column ===")
    say("strikers of k_0 other than the top gear, over all %d walks: %d" % (len(qs), strikers_C))
    say("those not congruent to +-1 mod 8: %d" % exc_C)
    say("so the first column of every walk is struck by the top gear and by NOTHING outside")
    say("the half of the machine with 2 a quadratic residue.")

    say("")
    say("=== D. realisation: is the admissible set exactly the striking set? ===")
    for i in [0, 1, 2, 3, -1, -2]:
        sm = [j for j, g in enumerate(gears) if g <= 200]
        na = int(adm[i][sm].sum()) + (0 if i else 0)
        nu = int(used[i][sm].sum())
        say("   offset %3d, gears <= 200: admissible %d, actually used %d, unused %s"
            % (i, na, nu, [gears[j] for j in sm if adm[i][j] and not used[i][j]]))

    say("")
    say("=== E. what the filter does to the path: depth and openness by offset ===")
    mean = dep_sum / dep_n
    var = dep_sq / dep_n - mean ** 2
    opd = open_n / dep_n
    pred = pred_sum / dep_n
    say("offset   mean depth   predicted   variance   opening density")
    for i in [-96, -54, -24, -6, -3, -2, -1, 0, 1, 2, 3, 4, 6, 12, 24, 96]:
        if i in idx:
            t = idx[i]
            say("  %5d     %.4f      %.4f     %.4f      %.5f"
                % (i, mean[t], pred[t], var[t], opd[t]))
    predv = predv_sum / dep_n
    nz0 = [t for t in range(len(offs)) if offs[t] != 0]
    say("measured vs predicted VARIANCE (independent gears): mean measured %.4f, mean predicted"
        " %.4f, max |diff| %.4f, correlation %.4f"
        % (var[nz0].mean(), predv[nz0].mean(), np.abs(var[nz0] - predv[nz0]).max(),
           float(np.corrcoef(var[nz0], predv[nz0])[0, 1])))
    say("mean depth measured vs predicted excluding i = 0: max |diff| %.4f"
        % np.abs(mean[nz0] - pred[nz0]).max())
    meanR = dep_sumR / dep_nR; varR = dep_sqR / dep_nR - meanR ** 2
    predR = pred_sumR / dep_nR; predvR = predv_sumR / dep_nR; opdR = open_nR / dep_nR
    say("")
    say("restricted to the %d walks with q >= %d (one machine size band, so the between-q"
        " spread does not enter the variance):" % (int(dep_nR[0]), QMIN))
    say("   mean depth: measured %.4f, predicted %.4f, max |diff| over nonzero offsets %.4f,"
        " correlation %.4f" % (meanR[nz0].mean(), predR[nz0].mean(),
                               np.abs(meanR[nz0] - predR[nz0]).max(),
                               float(np.corrcoef(meanR[nz0], predR[nz0])[0, 1])))
    say("   variance:   measured %.4f, predicted (independent gears) %.4f, correlation %.4f"
        % (varR[nz0].mean(), predvR[nz0].mean(),
           float(np.corrcoef(varR[nz0], predvR[nz0])[0, 1])))
    say("   opening density by offset: min %.5f at i=%d, max %.5f at i=%d, mean %.5f"
        % (opdR[nz0].min(), offs[nz0[int(np.argmin(opdR[nz0]))]],
           opdR[nz0].max(), offs[nz0[int(np.argmax(opdR[nz0]))]], opdR[nz0].mean()))
    say("   mean depth by offset: min %.4f at i=%d, max %.4f at i=%d"
        % (meanR[nz0].min(), offs[nz0[int(np.argmin(meanR[nz0]))]],
           meanR[nz0].max(), offs[nz0[int(np.argmax(meanR[nz0]))]]))
    say("   at i = 0: mean depth %.4f, variance %.4f" % (meanR[idx[0]], varR[idx[0]]))
    say("")
    say("=== G. the anchor along the path (order 0) ===")
    say("k_0 = 0 mod 5 (class A, q = +-1, +-11 mod 30): %d walks;  k_0 = 3 mod 5 (class B,"
        " q = +-7, +-13 mod 30): %d walks" % (nA, nB))
    say("gear 5 strikes offset i iff k_0 + i = 1 or 4 mod 5, i.e.")
    say("   class A: i = 1 or 4 mod 5;   class B: i = 1 or 3 mod 5.")
    say("offsets struck by gear 5 at EVERY q: i = 1 mod 5 (both classes).")
    fa = np.array([five_hit_A[t] / max(nA, 1) for t in range(len(offs))])
    fb = np.array([five_hit_B[t] / max(nB, 1) for t in range(len(offs))])
    say("class A: offsets with gear-5 strike fraction 1.0: %s"
        % sorted(offs[t] for t in range(len(offs)) if fa[t] == 1.0)[:14])
    say("class B: offsets with gear-5 strike fraction 1.0: %s"
        % sorted(offs[t] for t in range(len(offs)) if fb[t] == 1.0)[:14])
    say("offsets struck by gear 5 in BOTH classes (fraction 1.0 overall): %s"
        % sorted(offs[t] for t in range(len(offs)) if fa[t] == 1.0 and fb[t] == 1.0)[:14])
    say("gear-5 share of all path offsets: %.4f (exactly 2/5 by the pinned start slot)"
        % (five_hit[nz0].sum() / (max(nA + nB, 1) * len(nz0))))
    say("opening density at i = +1: %.5f;  at i = -1: %.5f  (forward always 5-struck,"
        " backward only in class A)" % (opd[idx[1]], opd[idx[-1]]))
    say("")
    say("measured vs predicted mean depth over all %d offsets: max |diff| %.4f,"
        " correlation %.6f" % (len(offs), np.abs(mean - pred).max(),
                               float(np.corrcoef(mean, pred)[0, 1])))
    say("per-offset mean depth: min %.4f at i=%d, max %.4f at i=%d"
        % (mean.min(), offs[int(np.argmin(mean))], mean.max(), offs[int(np.argmax(mean))]))
    nz = [idx[i] for i in offs if i != 0]
    say("over the %d nonzero offsets: mean depth %.4f (sd of the per-offset means %.4f),"
        % (len(nz), mean[nz].mean(), mean[nz].std()))
    say("   mean variance %.4f, mean opening density %.5f" % (var[nz].mean(), opd[nz].mean()))
    say("at i = 0: mean depth %.4f, variance %.4f, opening density %.5f"
        % (mean[idx[0]], var[idx[0]], opd[idx[0]]))
    fwd = [idx[i] for i in offs if 1 <= i <= I]
    bwd = [idx[i] for i in offs if -I <= i <= -1]
    say("forward offsets 1..%d : mean depth %.4f, opening density %.5f"
        % (I, mean[fwd].mean(), opd[fwd].mean()))
    say("backward offsets -1..-%d: mean depth %.4f, opening density %.5f"
        % (I, mean[bwd].mean(), opd[bwd].mean()))

    # ---------------- F. the difference-of-squares columns
    say("")
    say("=== F. the forced-composite columns behind q^2 ===")
    say("column k_0 - 6t^2 has upper member q^2 - 36t^2 = (q - 6t)(q + 6t): composite for")
    say("every q > 6t + 1, hence blocked whenever a factor is <= q, i.e. always (q - 6t <= q).")
    ex = 0
    tot = 0
    for q in qs:
        ng = int(np.searchsorted(G, q, side="right"))
        gg = G[:ng]
        for t in range(1, 5):
            j = 6 * t * t
            if q - 6 * t <= 1:
                continue
            tot += 1
            m = q * q - 36 * t * t
            if not ((m % gg == 0).any()):
                ex += 1
    say("columns k_0 - 6t^2, t = 1..4, over all walks: %d checked, %d not blocked" % (tot, ex))
    say("forward: q^2 + 6i is a perfect square only for i >= ((q+2)^2 - q^2)/6 = (2q+2)/3")
    say("(q = 5 mod 6) or (4q+8)/3 (q = 1 mod 6) - beyond the tooth arc d in both classes,")
    say("so no forced-composite square column lies within the forward arc, while the backward")
    say("direction carries them at distances 6, 24, 54, 96, ...")

    log.close()


if __name__ == "__main__":
    main()

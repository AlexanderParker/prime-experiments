"""Full-window cumulative margin trajectories (mechanic round 3).

M(t) = N(t) - P(t) = t - P(t) over the ENTIRE window of y (slots t = 1..W
from k_lo = ceil((y-1)/6)), where P(t) = prime members among the first t
slots (actual primality; the boundary members y-2 / y count as prime when
prime, as in rounds 1-2 - open-interval users adjust slot 1).

Structural note stated up front: M(t), n0, n1, n2 depend ONLY on the
primality of members - the margin trajectory is gear-blind. Layer bands
(squares p^2, fresh-gear activation) can therefore only show up as a
statistical prime-density effect, i.e. not at all; this script MEASURES
that (slope of M in windows before/after each band boundary, plus matched
mid-band controls) rather than assuming it.

Per window reported:
- min M(t), its t_min (absolute, as member value, as fraction of W)
- last t with M(t) < T for thresholds T in {0, 1, 10, 100, 1e3, 1e4, 1e5}
- M at ~8 log-spaced checkpoints per decade, against the PNT model
    Mhat(t) = t - [li(6t + m0) - li(m0)],  m0 = 6*k_lo - 1
  (growth is "t minus li", i.e. slope 1 - 6/ln(member): asymptotically
  linear in t, NOT t/log t; the checkpoint table lets anyone verify).
- band-boundary slope test: slopes (M(t_b)-M(t_b-h))/h vs after, h adapted
  to boundary spacing, aggregated with matched controls.

Outputs: research/data/margin_summary.csv, margin_checkpoints.csv,
margin_bands.csv (APPEND mode - delete the files to regenerate from scratch;
rerunning the same y duplicates its rows).
Usage: uv run python research/margin_trajectory.py [y...]
Default ladder: 101 149 211 307 401 419 503 1009 2003 5003 10007 20011 50021.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime

THRESH = (0, 1, 10, 100, 1000, 10000, 100000)


def li_diff(a, b, pts=4000):
    """integral of dt/ln t from a to b (a >= 5)."""
    if b <= a:
        return 0.0
    x = np.geomspace(a, b, pts)
    return float(np.trapezoid(1.0 / np.log(x), x))


def build_probes(y, k_lo, k_hi):
    """checkpoint t's + band-boundary probe t's."""
    W = k_hi - k_lo + 1
    cps = sorted({int(round(10 ** (i / 8))) for i in range(0, 200)
                  if 1 <= round(10 ** (i / 8)) <= W} | {W})
    import math
    r = math.isqrt(y)
    bps = [p for p in primes_upto(y) if p > r]
    bounds = []  # (p, t_b)
    for p in bps:
        t_b = (p * p - 1) // 6 - k_lo + 1
        if 1 <= t_b <= W:
            bounds.append((p, t_b))
    probes = set(cps)
    band_rows = []  # (p, t_b, h)
    for i, (p, t_b) in enumerate(bounds):
        prev_t = bounds[i - 1][1] if i > 0 else 0
        next_t = bounds[i + 1][1] if i + 1 < len(bounds) else W + 1
        h = min(1000, (t_b - prev_t) // 2, (next_t - t_b) // 2)
        if h < 50 or t_b - h < 1 or t_b + h > W:
            continue
        band_rows.append((p, t_b, h))
        probes |= {t_b - h, t_b, t_b + h}
    ctrl_rows = []  # (t_c, h) mid-band controls matched in h
    for i in range(len(band_rows) - 1):
        _, t1, h1 = band_rows[i]
        _, t2, h2 = band_rows[i + 1]
        t_c = (t1 + t2) // 2
        h = min(h1, h2, (t2 - t1) // 4)
        if h >= 50 and t_c - h > t1 and t_c + h < t2:
            ctrl_rows.append((t_c, h))
            probes |= {t_c - h, t_c, t_c + h}
    return W, cps, band_rows, ctrl_rows, sorted(probes)


def trajectory(y, seg=8_000_000):
    gears = [q for q in primes_upto(y) if q >= 5]
    k_lo = -((-(y - 1)) // 6)
    k_hi = (y * y + 1) // 6
    W, cps, band_rows, ctrl_rows, probes = build_probes(y, k_lo, k_hi)
    probes_arr = np.array(probes, dtype=np.int64)
    Mprobe = {}
    uvals = [pow(6, -1, q) for q in gears]
    Poff = 0
    minM, argmin = None, None
    last_below = {T: 0 for T in THRESH}
    for a in range(k_lo, k_hi + 1, seg):
        b = min(k_hi + 1, a + seg)
        n = b - a
        exL = np.zeros(n, bool)
        exR = np.zeros(n, bool)
        for q, u in zip(gears, uvals):
            exL[(u - a) % q::q] = True
            exR[(-u - a) % q::q] = True
        if a == k_lo:  # boundary slot: members may be <= y (gear = prime)
            for arr, m in ((exL, 6 * k_lo - 1), (exR, 6 * k_lo + 1)):
                if m <= y and is_prime(m):
                    arr[0] = False
        pcnt = (~exL).astype(np.int64) + (~exR).astype(np.int64)
        cum = np.cumsum(pcnt)
        tloc = np.arange(a - k_lo + 1, b - k_lo + 1, dtype=np.int64)
        M = tloc - (cum + Poff)
        i = int(np.argmin(M))
        if minM is None or M[i] < minM:
            minM, argmin = int(M[i]), int(tloc[i])
        for T in THRESH:
            idx = np.flatnonzero(M < T)
            if len(idx):
                last_below[T] = max(last_below[T], int(tloc[idx[-1]]))
        lo = np.searchsorted(probes_arr, tloc[0])
        hi = np.searchsorted(probes_arr, tloc[-1], side="right")
        for t in probes_arr[lo:hi]:
            Mprobe[int(t)] = int(t - (Poff + cum[t - (a - k_lo + 1)]))
        Poff += int(cum[-1])
    m0 = 6 * k_lo - 1
    checkpoints = []
    for t in cps:
        mh = t - li_diff(m0, 6 * (k_lo + t - 1) + 1)
        checkpoints.append((t, Mprobe[t], mh))
    bands = [(p, t, h, (Mprobe[t] - Mprobe[t - h]) / h,
              (Mprobe[t + h] - Mprobe[t]) / h) for p, t, h in band_rows]
    ctrls = [(t, h, (Mprobe[t] - Mprobe[t - h]) / h,
              (Mprobe[t + h] - Mprobe[t]) / h) for t, h in ctrl_rows]
    return dict(y=y, W=W, k_lo=k_lo, minM=minM, argmin=argmin,
                last_below=last_below, P=Poff, checkpoints=checkpoints,
                bands=bands, ctrls=ctrls)


def main():
    ys = [int(a) for a in sys.argv[1:]] or [
        101, 149, 211, 307, 401, 419, 503, 1009, 2003, 5003, 10007,
        20011, 50021]
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)

    def opencsv(name, header):
        path = os.path.join(ddir, name)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        f = open(path, "a")
        if new:
            f.write(header + "\n")
        return f
    fsum = opencsv("margin_summary.csv",
                   "y,W,minM,t_min,member_min,frac_min,"
                   + ",".join(f"last_below_{T}" for T in THRESH)
                   + ",M_end,P_end")
    fcp = opencsv("margin_checkpoints.csv", "y,t,member,M,M_li_model")
    fbd = opencsv("margin_bands.csv", "y,p,t_b,h,slope_before,slope_after")
    print(f"{'y':>7} {'W':>10} {'minM':>7} {'t_min':>7} {'frac_min':>9} "
          f"{'last<0':>7} {'last<100':>9} {'M(W)':>10} {'band_dslope':>12} "
          f"{'ctrl_dslope':>12} {'sec':>6}")
    for y in ys:
        t0 = time.time()
        r = trajectory(y)
        dt = time.time() - t0
        W, lb = r["W"], r["last_below"]
        mem = 6 * (r["k_lo"] + r["argmin"] - 1)
        fsum.write(f"{y},{W},{r['minM']},{r['argmin']},{mem},"
                   f"{r['argmin']/W:.3e},"
                   + ",".join(str(lb[T]) for T in THRESH)
                   + f",{W - r['P']},{r['P']}\n")
        for t, M, mh in r["checkpoints"]:
            fcp.write(f"{y},{t},{6*(r['k_lo']+t-1)},{M},{mh:.1f}\n")
        for p, t_b, h, sb, sa in r["bands"]:
            fbd.write(f"{y},{p},{t_b},{h},{sb:.5f},{sa:.5f}\n")
        db = [sa - sb for _, _, _, sb, sa in r["bands"]]
        dc = [sa - sb for _, _, sb, sa in r["ctrls"]]
        def ms(v):
            if not v:
                return "-"
            m = sum(v) / len(v)
            se = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5 \
                / len(v) ** 0.5
            return f"{m:+.4f}~{se:.4f}"
        print(f"{y:>7} {W:>10} {r['minM']:>7} {r['argmin']:>7} "
              f"{r['argmin']/W:>9.2e} {lb[0]:>7} {lb[100]:>9} "
              f"{W - r['P']:>10} {ms(db):>12} {ms(dc):>12} {dt:>6.1f}")
        sys.stdout.flush()
    for f in (fsum, fcp, fbd):
        f.close()
    print(f"\nwrote margin_summary.csv, margin_checkpoints.csv, "
          f"margin_bands.csv in {ddir}")


if __name__ == "__main__":
    main()

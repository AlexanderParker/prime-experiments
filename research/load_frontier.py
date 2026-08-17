"""Round 6 lateral: the LOAD-LENGTH FRONTIER - max prime load reality achieves
on twin-free runs of each length L, against the X-ceiling (load 1 per slot).

Setting: open interior only (slots with both members > y, the X-domain). A
twin-free slot carries 0 or 1 prime members (2 primes = twin, by definition),
so under X every L-run carries at most L primes - the C2 pigeonhole ceiling,
load 1. Reality's frontier:

    maxload(L) = max over twin-free L-windows of (prime members)/L

The frontier equals 1 up to L* (the longest SATURATED run: every slot exactly
one prime, no twins), then decays. gap(L) = 1 - maxload(L) is zero for L <= L*
and first opens just above it - that is the length scale where reality still
touches the X-ceiling, i.e. where a compression bound must fight hardest.
Also computed: maxload_any(L) (twins allowed - what X must forgo), the
bottom-band branch (windows starting within y slots of the interior start -
the starved band of the inversion zone), and the anatomy of record-holders.

Run: uv run python research/load_frontier.py   (from repo root; numpy)
"""
import numpy as np

from derivative_scan import sieve
from split_gap_law import primes

def frontier(y, anatomy_Ls=(25, 100)):
    print(f"--- y = {y} ---")
    K, gears, oml, omr, gvl, gvr = sieve(y)
    s0 = (y + 1) // 6 + 1                     # first slot with 6k-1 > y
    prate = ((oml == 0).astype(np.int8) + (omr == 0))
    prate[:s0] = 0
    Tm = (prate == 2)
    ts = np.flatnonzero(Tm)
    strides = np.diff(ts)
    maxstride = int(strides.max())
    # saturated runs: prate == 1
    sat = (prate == 1).astype(np.int8)
    # longest run of 1s
    d = np.diff(np.concatenate(([0], sat, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    lens = ends - starts
    Lstar = int(lens.max())
    istar = int(starts[np.argmax(lens)])
    print(f"  interior [{s0},{K}]; twins {len(ts)}; max stride {maxstride}; "
          f"L* (longest saturated run, load=1) = {Lstar} at slot {istar} "
          f"(depth {istar/K:.4f}, members ~{6*istar})")
    Pcs = np.concatenate(([0], np.cumsum(prate, dtype=np.int64)))
    Tcs = np.concatenate(([0], np.cumsum(Tm.astype(np.int64))))
    Ls = sorted({1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 63, 80,
                 100, 126, 160, 200, 252, 320, 400, maxstride}
                & set(range(1, maxstride + 1)) | {Lstar, Lstar + 1, maxstride})
    band_hi = s0 + y                          # bottom band: first y interior slots
    print(f"  {'L':>5} {'maxload':>8} {'gap':>7} {'any':>7} {'bottom':>7} "
          f"{'depth':>7} {'parent':>7}")
    rows = {}
    top1 = np.sort(strides)[-max(len(strides) // 100, 1):][0]
    for L in Ls:
        Pw = Pcs[s0 + L:K + 2] - Pcs[s0:K + 2 - L]
        Tw = Tcs[s0 + L:K + 2] - Tcs[s0:K + 2 - L]
        valid = Tw == 0
        if not valid.any():
            continue
        Pv = np.where(valid, Pw, -1)
        i = int(Pv.argmax())
        ml = Pv[i] / L
        ml_any = Pw.max() / L
        # bottom band branch
        nb = max(0, min(len(Pv), band_hi - s0))
        mb = Pv[:nb].max() / L if nb and Pv[:nb].max() >= 0 else float('nan')
        start = s0 + i
        # parent stride
        j = np.searchsorted(ts, start)
        lo = ts[j - 1] if j > 0 else s0 - 1
        hi = ts[j] if j < len(ts) else K + 1
        parent = int(hi - lo)
        rows[L] = (ml, start, parent)
        print(f"  {L:>5} {ml:>8.4f} {1-ml:>7.4f} {ml_any:>7.4f} {mb:>7.4f} "
              f"{start/K:>7.4f} {parent:>7}{' top1%' if parent >= top1 else ''}")
    # anatomy of record-holders
    print(f"  anatomy of record windows (composite-member lpf shares):")
    for L in sorted(set(anatomy_Ls) | {Lstar, maxstride}):
        if L not in rows:
            continue
        ml, start, parent = rows[L]
        small = tot = n2in = 0
        for k in range(start, start + L):
            for om, v in ((oml[k], 6 * k - 1), (omr[k], 6 * k + 1)):
                if om >= 1:
                    tot += 1
                    lpf = next(g for g in gears if v % g == 0)
                    small += lpf <= 13
            n2in += oml[k] >= 1 and omr[k] >= 1
        print(f"    L={L:>4}: load {ml:.4f} depth {start/K:.4f} parent {parent} "
              f"n2-inside {n2in} ({n2in/L:.2f}/slot)  lpf<=13 share "
              f"{small}/{tot} = {small/tot:.2f}")
    return Lstar, maxstride, rows

if __name__ == "__main__":
    print("=" * 72)
    print("LOAD-LENGTH FRONTIER: max twin-free prime load vs run length")
    print("(X-ceiling = 1 per slot; gap(L) = 1 - maxload(L))")
    for y in (1009, 3163, 10007):
        frontier(y)

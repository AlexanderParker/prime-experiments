"""Round 20 (mechanic): THE JOINT GAP-PAIR CENSUS at separations 1..5.

Constructor's suppression law rests on p_j, the joint distribution of
QUALIFYING-SIZE gaps at separations 1..j, and on the deficit of the measured
joint against the independent product.  This tool measures the whole joint
object exactly, at full period where reachable, in ONE stream pass:

  ghist[v]            gap-value histogram
  pair[j][u][v]       # of i with d_i = u and d_{i+j} = v,  j = 1..LAGS
  minhist[m][x]       # of i with min(d_i .. d_{i+m-1}) = x,  m = 2..RUNS

Every threshold question is then a summation over these tables, so a single
run answers "both gaps >= a at lag j" and "all m consecutive gaps >= a" for
EVERY a at once - no re-scan per step, and no per-step share extrapolation.

Usage: uv run python research/gap_pair_census.py y [--limit N] [--seg N]
Writes research/data/gap_pair_{pair,min,hist}.csv (append) and prints the
deficit tables against independence at the machine's own qualifying floors.
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
from flank_envelope import primes_upto

V = 128          # gap values are < 128 at every machine reached here
LAGS = 5
RUNS = 6


def run(y, limit=None, seg=64_000_000, verbose=True, start=0):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, start + limit)
    uvals = [pow(6, -1, g) for g in gears]
    ghist = np.zeros(V, np.int64)
    pair = np.zeros((LAGS + 1, V, V), np.int64)
    minh = np.zeros((RUNS + 1, V), np.int64)
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    ngap = 0
    ctx = LAGS + RUNS + 4
    for lo in range(start, K, seg):
        hi = min(K, lo + seg)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tail, op])
        if len(ops) > ctx:
            d = np.minimum(np.diff(ops), V - 1).astype(np.int64)
            new = ops[1:] >= lo          # the gap's right end is new
            ghist += np.bincount(d[new], minlength=V)
            ngap += int(new.sum())
            n = len(d)
            for j in range(1, LAGS + 1):
                if n <= j:
                    break
                m = new[j:]
                if m.any():
                    a, b = d[:-j][m], d[j:][m]
                    pair[j] += np.bincount(a * V + b,
                                           minlength=V * V).reshape(V, V)
            for mlen in range(2, RUNS + 1):
                if n < mlen:
                    break
                run_min = d[:n - mlen + 1].copy()
                for t in range(1, mlen):
                    np.minimum(run_min, d[t:n - mlen + 1 + t], out=run_min)
                sel = new[mlen - 1:]
                if sel.any():
                    minh[mlen] += np.bincount(run_min[sel], minlength=V)
        tail = ops[-ctx:].copy() if len(ops) >= ctx else ops.copy()
        if verbose:
            print(f"  seg to {hi:.4g} ({100*hi/K:.1f}%) "
                  f"{time.time()-t0:.0f}s", flush=True)
    # ROUND-25 DEFECT FIX (found by Lateral's parity law, routed to this lane).
    # A PERIOD IS A CIRCLE: N openings carry N gaps, the last running from the
    # final opening round to the first.  np.diff over a linear pass drops it,
    # so every full-period table used to be short by its seam structures -
    # 1 gap, j+1 pairs at lag j, m minima at run length m.  Harmless for
    # densities, fatal for exact identities.  The seam is computed in closed
    # form from the first and last few openings (research/cyclic_close_r25.py),
    # so the close costs nothing and needs no rescan.
    if start == 0 and K == P:
        sys.path.insert(0, HERE)
        from cyclic_close_r25 import seam_structures, LAGS as SL, RUNS as SR
        assert (SL, SR) == (LAGS, RUNS), "seam module lag/run mismatch"
        wrap, spairs, sruns, _, _, _ = seam_structures(y)
        c = lambda v: min(int(v), V - 1)          # noqa: E731
        ghist[c(wrap)] += 1
        ngap += 1
        for j, ps in spairs.items():
            for a, b in ps:
                pair[j][c(a), c(b)] += 1
        for m, vals in sruns.items():
            for v in vals:
                minh[m][c(v)] += 1
        if verbose:
            print(f"  cyclic close: wrap gap {wrap} (= the first gap, by the "
                  f"mirror law) + seam pairs/runs", flush=True)
    return dict(y=y, P=P, K=K, start=start, gears=gears, ghist=ghist, pair=pair,
                minh=minh, ngap=ngap, secs=time.time() - t0)


def report(r, floors=None):
    y, gh, ngap = r["y"], r["ghist"], r["ngap"]
    F = int(np.flatnonzero(gh)[-1])
    tail = np.cumsum(gh[::-1])[::-1]        # tail[a] = #gaps >= a
    print(f"\n=== machine {y}: period {r['P']:.4g}, scanned {r['K']:.4g} "
          f"({100*r['K']/r['P']:.3f}%), {ngap:,} gaps, F = {F}, "
          f"{r['secs']:.0f}s")
    if floors is None:
        floors = sorted({2 * round(q / 6) for q in
                         (13, 17, 19, 23, 29, 31, 37, 41, 43)
                         if 2 * round(q / 6) <= F})
    for a in floors:
        p1 = tail[a] / ngap
        print(f"  --- qualifying floor a = {a}: P(g >= a) = {p1:.6g} "
              f"({int(tail[a]):,} of {ngap:,} gaps)")
        print("      lag j   both >= a      independent      obs/indep")
        for j in range(1, LAGS + 1):
            both = int(r["pair"][j][a:, a:].sum())
            npairs = int(r["pair"][j].sum())
            obs = both / npairs if npairs else 0.0
            ind = p1 * p1
            print(f"        {j}     {both:>12,}   {ind*npairs:>14,.1f}   "
                  f"{obs/ind if ind else 0:8.4f}"
                  f"{'   DEFICIT x%.1f' % (ind/obs) if obs > 0 and obs < ind else ('   EXCESS x%.1f' % (obs/ind) if obs > ind else '   ZERO')}")
        print("      run m   all >= a       independent      obs/indep")
        for m in range(2, RUNS + 1):
            allm = int(r["minh"][m][a:].sum())
            nwin = int(r["minh"][m].sum())
            obs = allm / nwin if nwin else 0.0
            ind = p1 ** m
            print(f"        {m}     {allm:>12,}   {ind*nwin:>14,.1f}   "
                  f"{obs/ind if ind else 0:8.4f}"
                  f"{'   DEFICIT x%.1f' % (ind/obs) if obs > 0 and obs < ind else ('   EXCESS x%.1f' % (obs/ind) if obs > ind else '   ZERO (obs = 0)')}")


def write_csv(r):
    os.makedirs(DDIR, exist_ok=True)
    y = r["y"]
    cov = (r["K"] - r.get("start", 0)) / r["P"]
    p = os.path.join(DDIR, "gap_pair_hist.csv")
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    with open(p, "a") as f:
        if new:
            f.write("y,coverage,kind,index,value,count\n")
        for v in np.flatnonzero(r["ghist"]):
            f.write(f"{y},{cov:.6f},ghist,0,{int(v)},{int(r['ghist'][v])}\n")
        for m in range(2, RUNS + 1):
            for v in np.flatnonzero(r["minh"][m]):
                f.write(f"{y},{cov:.6f},minhist,{m},{int(v)},"
                        f"{int(r['minh'][m][v])}\n")
    p2 = os.path.join(DDIR, "gap_pair_joint.csv")
    new = not os.path.exists(p2) or os.path.getsize(p2) == 0
    with open(p2, "a") as f:
        if new:
            f.write("y,coverage,lag,gu,gv,count\n")
        for j in range(1, LAGS + 1):
            nz = np.flatnonzero(r["pair"][j].ravel())
            for z in nz:
                u, v = divmod(int(z), V)
                f.write(f"{y},{cov:.6f},{j},{u},{v},"
                        f"{int(r['pair'][j][u, v])}\n")
    print(f"  wrote {p}, {p2}")


def main():
    args = sys.argv[1:]
    limit, seg = None, 64_000_000
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(float(args[i + 1]))
        del args[i:i + 2]
    if "--seg" in args:
        i = args.index("--seg")
        seg = int(float(args[i + 1]))
        del args[i:i + 2]
    quiet = "--quiet" in args
    if quiet:
        args.remove("--quiet")
    start = 0
    if "--start" in args:
        i = args.index("--start")
        start = int(float(args[i + 1]))
        del args[i:i + 2]
    y = int(args[0])
    r = run(y, limit=limit, seg=seg, verbose=not quiet, start=start)
    report(r)
    write_csv(r)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

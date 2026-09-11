"""xm_scan.py -- x_min(p, l) for the engine {5..p} by a segmented scan in column order.

Objects (research/proof/first_realisation.md section 0.1):
  column k = (6k-1, 6k+1); gear g strikes k iff k = +-u_g (mod g), u_g = 6^{-1} mod g;
  a RUN of length l at x = the columns x..x+l-1 all struck;
  x_min(p, l) = the least x >= 1 with a run of length l at x.

x_min(., l) is a staircase: it is the start of the first run whose length is >= l, i.e. the
first RECORD-BREAKING run at level l when the runs are read in column order.  So the scan
returns the running-record sequence (x_i, r_i): r_1 < r_2 < ... and x_min(p, l) = x_i for the
least i with r_i >= l.

Method: the columns are sieved in chunks (a tiled pattern of the gears <= 13, then strided
writes for the larger gears), each chunk yields its own record staircase plus the histogram of
run lengths starting inside it; chunks are merged in order.  Runs that cross a chunk boundary
are seen whole by the chunk they start in (each chunk is sieved OVER columns past its end;
OVER exceeds every F(p) for p <= 59).  Column 0 is open under every engine, so the scan
starts at column 1 with column 0 as the opening before it.

The scan stops at P/2 + OVER (the mirror k -> -k puts the first realisation of every run
length at or below P/2) or at --max-columns, and checkpoints to results/xm_scan_m{p}.json;
--resume continues from the checkpoint.

Validation targets (P1 of the document): F(23) = 34 with x_min(23, 33) = 12,694,429 (the
certified first record stretch of m23, r34); the corpus F at m11..m37; the window records.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
OVER = 1024
HMAX = 512          # histogram of run lengths 0..HMAX-1

_G = {}


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def u_of(g):
    return pow(6, -1, g)


def period(p):
    P = 1
    for g in gears_of(p):
        P *= g
    return P


def base_pattern(base):
    Pb = 1
    for g in base:
        Pb *= g
    pat = np.zeros(Pb, dtype=bool)
    for g in base:
        u = u_of(g)
        pat[u % g::g] = True
        pat[(-u) % g::g] = True
    return Pb, pat


def init(p, force_open=None):
    gears = gears_of(p)
    base = [g for g in gears if g <= 13]
    big = [g for g in gears if g > 13]
    Pb, pat = base_pattern(base)
    _G["Pb"] = Pb
    _G["pat"] = pat
    _G["big"] = big
    _G["force_open"] = force_open   # a column treated as open (the --from mode: runs are
    #                                 measured from the start column, whatever precedes it)


def sieve_segment(start, n):
    """bool array: struck?, for the columns start .. start+n-1."""
    Pb, pat, big = _G["Pb"], _G["pat"], _G["big"]
    off = start % Pb
    reps = (off + n + Pb - 1) // Pb + 1
    arr = np.tile(pat, reps)[off:off + n].copy()
    for g in big:
        u = u_of(g)
        for t in (u % g, (-u) % g):
            i0 = (t - start) % g
            arr[i0::g] = True
    return arr


def runs_in(cs, ce):
    """All runs starting in [cs, ce): (starts, lens) in column order."""
    s0 = cs - 1
    n = ce - s0 + OVER
    arr = sieve_segment(s0, n)
    fo = _G.get("force_open")
    if fo is not None and s0 <= fo < s0 + n:
        arr[fo - s0] = False
    op = np.flatnonzero(~arr).astype(np.int64) + s0
    if op.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = op[:-1] + 1
    lens = op[1:] - op[:-1] - 1
    keep = (starts >= cs) & (starts < ce) & (lens > 0)
    return starts[keep], lens[keep]


def work(args):
    cs, ce = args
    starts, lens = runs_in(cs, ce)
    hist = np.bincount(np.minimum(lens, HMAX - 1), minlength=HMAX)
    if lens.size == 0:
        return [], hist
    cm = np.maximum.accumulate(lens)
    prev = np.concatenate([[0], cm[:-1]])
    idx = np.flatnonzero(lens > prev)
    return [(int(starts[i]), int(lens[i])) for i in idx], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1 << 24)
    ap.add_argument("--max-columns", type=int, default=None)
    ap.add_argument("--target", type=int, default=None,
                    help="stop once a run of this length is found (F(p) - 1)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=64, help="tasks between checkpoints")
    ap.add_argument("--out", default=None)
    ap.add_argument("--from", dest="from_col", type=int, default=None,
                    help="start the scan at this column, with the column before it treated as "
                         "open (so the run through the start column is measured from it)")
    a = ap.parse_args()
    p = a.p
    here = os.path.dirname(os.path.abspath(__file__))
    suffix = "" if a.from_col is None else f"_from{a.from_col}"
    out = a.out or os.path.join(here, "results", f"xm_scan_m{p}{suffix}.json")
    P = period(p)
    limit = P // 2 + OVER
    if a.max_columns is not None:
        limit = min(limit, a.max_columns)
    stair = []
    gmax = 0
    hist = np.zeros(HMAX, dtype=np.int64)
    scanned_to = 1 if a.from_col is None else a.from_col
    force_open = None if a.from_col is None else a.from_col - 1
    t_prev = 0.0
    if a.resume and os.path.exists(out):
        with open(out) as f:
            ck = json.load(f)
        stair = [tuple(e) for e in ck["stair"]]
        gmax = ck["gmax"]
        hist = np.array(ck["hist"], dtype=np.int64)
        scanned_to = ck["scanned_to"]
        t_prev = ck.get("secs", 0.0)
        print(f"resume m{p} from column {scanned_to}, gmax {gmax}", flush=True)
    t0 = time.time()

    def save(done):
        rec = {"p": p, "gears": gears_of(p), "P": P, "limit": limit, "scanned_to": scanned_to,
               "done": done, "gmax": gmax, "stair": stair, "hist": hist.tolist(),
               "secs": t_prev + time.time() - t0, "over": OVER, "chunk": a.chunk,
               "from": a.from_col}
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, out)

    start0 = scanned_to

    def tasks():
        cs = start0
        while cs < limit:
            ce = min(cs + a.chunk, limit)
            yield (cs, ce)
            cs = ce

    done = False
    ntask = 0
    with Pool(a.workers, initializer=init, initargs=(p, force_open)) as pool:
        it = pool.imap(work, tasks(), chunksize=2)
        for (cs, ce), (res, h) in zip(tasks(), it):
            hist += h
            for (x, L) in res:
                if L > gmax:
                    gmax = L
                    stair.append((x, L))
                    print(f"m{p}: run {L:4d} first at x = {x:>16,d}   ({x / P:.6f} P)   "
                          f"t = {t_prev + time.time() - t0:8.1f}s", flush=True)
            scanned_to = ce
            ntask += 1
            if a.target is not None and gmax >= a.target:
                done = True
                break
            if ntask % a.checkpoint_every == 0:
                save(False)
        pool.terminate()
    if scanned_to >= limit:
        done = True
    save(done)
    print(f"m{p}: scanned to {scanned_to:,d} of limit {limit:,d} (P = {P:,d}); gmax {gmax}; "
          f"done {done}; {t_prev + time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

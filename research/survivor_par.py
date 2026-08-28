"""Round 24 (constructor): SHARDED DRIVER for the survivor generator.

The human's compute-policy update (2026-08-28) allows multi-core; memory is
the binding constraint (pool <= 3 for memory-heavy children).  The sequential
pass in research/survivor_generator.py already interacts between segments only
through carried overlap margins, so it shards cleanly:

  * the slot range [0, P) is split into W contiguous shards;
  * each worker scans [lo0 - PAD, hi0) (PAD slots of warm-up so the first
    OVL-skipped openings of its range are owned by the previous shard's
    overlap - every gap index is INTERIOR to some worker's owned region);
  * the LAST worker wraps the cyclic seam: after P it continues into
    [0, PAD) shifted by +P, exactly what the sequential pass's final
    eat(concat[tail, head + P]) does;
  * results merge by max (F, F_2, witnesses) and union (realised tuples) -
    both idempotent, so shard overlap is harmless; only UNDER-coverage could
    bias, and PAD = 40,000 slots ~ 7,400 openings >> OVL + MAXK + 8 ~ 48.

VERIFICATION GATE: the driver asserts digit-for-digit agreement with the
sequential pass's verified outputs (KNOWN_F, KNOWN_F2) and is required to
reproduce machines 19/23/29 - all already double-sourced - before its
machine-31 output is trusted.  The eat() kernel is a replication of the
sequential one with range/wrap parameters; the gate is what checks the
replication.

Usage: uv run python research/survivor_par.py y [y ...] [--seg N] [--m 4]
                                                  [--workers 3]
"""
import os
import sys
import time
from math import prod

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from survivor_generator import (classify, legal, land, primes, next_prime,   # noqa: E402
                                abstract, KNOWN_F, KNOWN_F2, OVL, MAXK, BASE)

PAD = 40_000


def shard(args):
    y, m, seg, lo0, hi0, wrap = args
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    uvals = [pow(6, -1, g) for g in gears]
    F_old = F_new = F2_new = F2_old = 0
    wit = {}
    tuples = (np.zeros(BASE ** m, bool) if m == 4 else set())
    ngaps = 0
    tail = None
    start = max(0, lo0 - PAD)

    def eat(ops, first_new):
        nonlocal F_old, F_new, F2_new, F2_old, ngaps
        d = np.diff(ops)
        n = len(d)
        if n <= 2 * (MAXK + OVL + 8):
            return
        assert int(d.max()) < BASE, "gap value exceeds the packing base"
        d = d.astype(np.int32)
        cls = classify(d, q1, a, b)
        d2 = np.zeros(n, np.int32)
        d2[:-1] = d[:-1] + d[1:]
        cls2 = classify(d2, q1, a, b)
        NEG = -(1 << 28)
        h = np.stack([d.copy(), d.copy()])
        nxt = np.arange(1, n)
        used = 0
        for _ in range(MAXK):
            hn = h.copy()
            for s in (0, 1):
                lg = legal(s, cls[:-1])
                ld = land(s, cls[:-1])
                hn[s][:-1] = np.where(lg, np.maximum(h[s][:-1],
                                                     d[:-1] + h[ld, nxt]),
                                      h[s][:-1])
            if np.array_equal(hn, h):
                break
            h = hn
            used += 1
        assert used < MAXK, "chain longer than MAXK"
        G = np.full((2, n), NEG, np.int32)
        for s in (0, 1):
            lg = legal(s, cls[:-2])
            stop = d[:-2] + d[1:-1]
            lg2 = legal(s, cls2[:-2])
            ld2 = land(s, cls2[:-2])
            cont = np.where(lg2, stop + h[ld2, np.arange(2, n)],
                            NEG).astype(np.int32)
            G[s][:-2] = np.where(lg, NEG,
                                 np.maximum(stop.astype(np.int32), cont))
        for _ in range(MAXK):
            Gn = G.copy()
            for s in (0, 1):
                lg = legal(s, cls[:-1])
                ld = land(s, cls[:-1])
                Gn[s][:-1] = np.where(lg, np.maximum(G[s][:-1],
                                                     d[:-1] + G[ld, nxt]),
                                      G[s][:-1])
            if np.array_equal(Gn, G):
                break
            G = Gn
        i0 = max(OVL, first_new)
        i1 = n - MAXK - OVL
        if i1 <= i0:
            return
        sl = slice(i0, i1)
        ngaps += i1 - i0
        F_old = max(F_old, int(d[sl].max()))
        L = d[i0 - 1:i1 - 1]
        F_new = max(F_new, int(max((L + h[0][sl]).max(),
                                   (L + h[1][sl]).max())))
        F2_old = max(F2_old, int((d[i0 - 1:i1 - 1] + d[sl]).max()))
        for s in (0, 1):
            v = L + G[s][sl]
            j = int(np.argmax(v))
            if int(v[j]) > F2_new:
                F2_new = int(v[j])
                i = i0 + j
                wit["A"] = ("branchA", s, [int(x) for x in d[i - 1:i + 6]],
                            int(ops[i]))
        vb0 = d[i0 - 1:i1 - 1] + d[sl]
        if int(vb0.max()) > F2_new:
            F2_new = int(vb0.max())
            wit["B0"] = ("branchB0", int(vb0.max()))
        for t in (0, 1):
            v = (d[i0 - 2:i1 - 2] + d[i0 - 1:i1 - 1] + h[t][i0:i1])
            v = np.where(~legal(1 - t, cls[i0 - 1:i1 - 1]), v, NEG)
            j = int(np.argmax(v))
            if int(v[j]) > F2_new:
                F2_new = int(v[j])
                i = i0 + j
                wit["B1"] = ("branchB1", t, [int(x) for x in d[i - 2:i + 4]],
                             int(ops[i]))
        lo, hi = i0 - 1, min(i1 + m, n)
        if hi - lo > m:
            k = np.zeros(hi - lo - m + 1, np.int64)
            for j in range(m):
                k = k * BASE + d[lo + j:hi - m + 1 + j].astype(np.int64)
            if m == 4:
                tuples[k] = True
            else:
                tuples.update(np.unique(k).tolist())

    def openings(lo, hi, shift=0):
        for attempt in range(60):
            try:
                ex = np.zeros(hi - lo, bool)
                for g, u in zip(gears, uvals):
                    ex[(u - lo) % g::g] = True
                    ex[(-u - lo) % g::g] = True
                return (np.flatnonzero(~ex) + lo + shift).astype(np.int64)
            except MemoryError:
                time.sleep(20)
        raise MemoryError("segment %d unallocatable after 60 retries" % lo)

    t0 = time.time()
    ranges = [(lo, min(hi0, lo + seg), 0) for lo in range(start, hi0, seg)]
    if wrap:
        ranges.append((0, PAD, P))          # cyclic seam continuation
    for idx, (lo, hi, shift) in enumerate(ranges):
        op = openings(lo, hi, shift)
        ops = op if tail is None else np.concatenate([tail, op])
        for attempt in range(60):
            try:
                eat(ops, OVL)
                break
            except MemoryError:
                time.sleep(20)
        else:
            raise MemoryError("eat at %d unallocatable" % lo)
        tail = ops[-(2 * (MAXK + OVL + 8) + 2):].copy()
        del op, ops
        if idx % 32 == 0 and P > 5e8:
            print("    shard@%d: %d/%d segs %.0fs"
                  % (lo0, idx, len(ranges), time.time() - t0), flush=True)
    tup_out = (np.flatnonzero(tuples) if m == 4
               else np.array(sorted(tuples), np.int64))
    return dict(F_old=F_old, F_new=F_new, F2_old=F2_old, F2_new=F2_new,
                wit=wit, tup=tup_out, ngaps=ngaps,
                secs=time.time() - t0)


def run_par(y, seg=12_000_000, m=4, workers=3):
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    q2 = next_prime(q1)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    t0 = time.time()
    W = workers
    cuts = [P * i // W for i in range(W + 1)]
    jobs = [(y, m, seg, cuts[i], cuts[i + 1], i == W - 1) for i in range(W)]
    print("=== machine %d -> %d  (period %d, %d workers, seg %d)"
          % (y, q1, P, W, seg), flush=True)
    if W == 1:
        parts = [shard(jobs[0])]
    else:
        with Pool(W) as pool:
            parts = pool.map(shard, jobs)
    F_old = max(p["F_old"] for p in parts)
    F_new = max(p["F_new"] for p in parts)
    F2_old = max(p["F2_old"] for p in parts)
    F2_new = max(p["F2_new"] for p in parts)
    wit = {}
    for p in parts:                       # keep the first witness per branch
        for kk, w in p["wit"].items():
            if kk not in wit:
                wit[kk] = w
    tuples = np.zeros(BASE ** m, bool)
    for p in parts:
        tuples[p["tup"]] = True
    ngaps = sum(p["ngaps"] for p in parts)
    print("  F(M) = %d   F_2(M) = %d   letters a=%d b=%d"
          % (F_old, F2_old, a, b))
    print("  EXACT  F(M+q')   = %d   (known %s)" % (F_new, KNOWN_F.get(q1)))
    print("  EXACT  F_2(M+q') = %d   (known %s)" % (F2_new, KNOWN_F2.get(q1)))
    if q1 in KNOWN_F:
        assert F_new == KNOWN_F[q1], (y, F_new, KNOWN_F[q1])
    if y in KNOWN_F2:
        assert F2_old == KNOWN_F2[y], (y, F2_old, KNOWN_F2[y])
    if q1 in KNOWN_F2:
        assert F2_new == KNOWN_F2[q1], (y, F2_new, KNOWN_F2[q1])
        print("     SURVIVOR IDENTITY VERIFIED (exact, full period, sharded)")
    for kk in sorted(wit):
        print("     witness %s: %s" % (kk, wit[kk]))
    assert F2_new >= F_new, (F2_new, F_new)
    print("  the NEXT step's two-gap budget: F(M+q') + q'' = %d + %d = %d"
          "   -> margin %+d" % (F_new, q2, F_new + q2, F_new + q2 - F2_new))
    print("  realised gap %d-tuples of M: %d  (scanned %d gaps incl. shard"
          " overlap)" % (m, int(tuples.sum()), ngaps))
    print("  (%.0f s wall; worker secs %s)"
          % (time.time() - t0, [int(p["secs"]) for p in parts]))
    return dict(y=y, q1=q1, q2=q2, F_old=F_old, F2_old=F2_old, F_new=F_new,
                F2_new=F2_new, tuples=tuples, P=P, ngaps=ngaps, a=a, b=b,
                m=m)


def main():
    args = sys.argv[1:]
    seg = 12_000_000
    m = 4
    workers = 3
    for flag in ("--seg", "--m", "--workers"):
        if flag in args:
            i = args.index(flag)
            v = args[i + 1]
            del args[i:i + 2]
            if flag == "--seg":
                seg = int(float(v))
            elif flag == "--m":
                m = int(v)
            else:
                workers = int(v)
    for y in args:
        res = run_par(int(y), seg=seg, m=m, workers=workers)
        t0 = time.time()
        ab = abstract(res)
        if ab.get("cyclic"):
            print("  A_%d survivor closure: CYCLIC (vacuous)" % m)
        else:
            nb = res["F_new"] + res["q2"]
            print("  A_%d over M: %d states, %d ordinary + %d skip edges"
                  % (m, ab["states"], ab["oedges"], ab["sedges"]))
            print("     plain closure  -> F(M+q')   <= %d   (exact %d) %s"
                  % (ab["plain"], res["F_new"],
                     "EXACT" if ab["plain"] == res["F_new"] else ""))
            print("     survivor closure -> F_2(M+q') <= %d   (known %s) %s"
                  % (ab["F2bound"], KNOWN_F2.get(res["q1"], "?"),
                     "EXACT" if ab["F2bound"] == KNOWN_F2.get(res["q1"])
                     else ""))
            print("       branch A %d, branch B0 (= F_2(M)) %d, branch B1 %d"
                  % (ab["bA"], ab["b0"], ab["bB1"]))
            print("     two-gap statement at M+q':  %d <= F(M+q') + q'' = %d"
                  "  %s (margin %+d)"
                  % (ab["F2bound"], nb,
                     "CERTIFIES" if ab["F2bound"] <= nb else "FAILS",
                     nb - ab["F2bound"]))
            assert ab["plain"] >= res["F_new"]
            assert ab["F2bound"] >= res["F2_new"]
            assert ab["b0"] == res["F2_old"], (ab["b0"], res["F2_old"])
        print("  (abstraction %.0f s)" % (time.time() - t0))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()

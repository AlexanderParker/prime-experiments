"""Round 28 (mechanic): EXTEND THE EXACT m41 4-TUPLE SHARD - by theorem first,
then by solver.

ROUND-27 STATE (C33/C33b).  The exact m41 arity-4 dictionary is COMPLETE at
every span <= 77 (169,981 reverse classes decided, 338,855 tuples realised);
the remaining ~1.23M reverse classes were priced at ~4.0e6 core-seconds, i.e.
a multi-round object.

WHAT CHANGED THIS ROUND - THE DEPTH-0 LEMMA (research/onset_anatomy_r28.py).

    D_4(M)  SUBSET  D_4(M + q')   for every q' >= 11.

Proof: a realised M 4-tuple has 5 exposed points, so at most 10 residues are
forbidden for the new gear's phase; for q' >= 11 an admissible phase exists,
and y_0 mod q' runs over every residue across the q' laps (CRT).  Checked at
all six exact pairs 13->17 ... 31->37 and against the round-27 m41 shard.

CONSEQUENCE: every candidate that is ALREADY in the exact m37 dictionary is
YES at machine 41 with no solver and no scan, at EVERY span - including the
expensive bands the round-27 pricing priced at 3-4 seconds a decision.  This
splits the census into a free half and a paid half and re-prices the rest.

Usage:
  <venv>/python research/m41_shard_r28.py price
  <venv>/python research/m41_shard_r28.py work W w SPAN_MAX HOURS
  <venv>/python research/m41_shard_r28.py merge W SPAN_MAX
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "r28")
R27 = os.path.join(DATA, "r27")
# the tightest certified superset available (walk-screened if it exists)
CANDS = [os.path.join(OUT, "gap_tuples_41_4_walkscreened.csv"),
         os.path.join(R27, "gap_tuples_41_4_screened_spancap.csv")]
M37 = os.path.join(DATA, "gap_tuples_37_4.csv")
SHARD77 = os.path.join(R27, "gap_tuples_41_4_exact_le77.csv")
B = 100


def load_arr(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int16)
    assert a.ndim == 2 and a.shape[1] == 4, a.shape
    return a


def key(a):
    a = a.astype(np.int64)
    return ((a[:, 0] * B + a[:, 1]) * B + a[:, 2]) * B + a[:, 3]


def candidate_file():
    for p in CANDS:
        if os.path.exists(p):
            return p
    raise SystemExit("no candidate superset on disk")


def classes(span_max=10 ** 9):
    """Reverse-class representatives of the superset, ascending span, with a
    flag for 'already in D_4(37)' (YES BY THEOREM)."""
    p = candidate_file()
    a = load_arr(p)
    k, kr = key(a), key(a[:, ::-1])
    assert np.isin(kr, k).all(), "candidate set is not reverse-closed"
    keep = (k <= kr) & (a.sum(axis=1) <= span_max)
    a = a[keep]
    order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0], a.sum(axis=1)))
    a = a[order]
    old = set(key(load_arr(M37)).tolist())
    free = np.array([int(x) in old for x in key(a)], bool)
    return p, a, free


def price():
    p, a, free = classes()
    span = a.sum(axis=1)
    print("candidate superset: %s" % p)
    print("reverse classes: %d   of which YES BY THEOREM (in D_4(37)): %d "
          "(%.1f%%)   remaining: %d"
          % (len(a), int(free.sum()), 100.0 * free.mean(),
             int((~free).sum())))
    done = set()
    if os.path.exists(SHARD77):
        done = set(key(load_arr(SHARD77)).tolist())
    print("\n  band        classes    free (D_4(37))   PAID    of the paid, "
          "already decided <= span 77")
    bands = [(1, 60), (61, 77), (78, 80), (81, 90), (91, 100), (101, 110),
             (111, 118)]
    for lo, hi in bands:
        m = (span >= lo) & (span <= hi)
        f = int(free[m].sum())
        paid = int((~free[m]).sum())
        dec = sum(1 for i in np.flatnonzero(m & ~free)
                  if int(key(a[i:i + 1])[0]) in done)
        print("  %3d-%3d %10d %14d %8d %14d" % (lo, hi, int(m.sum()), f, paid,
                                                dec))
    print("\n  round-27 measured decision cost: 0.032 s/class at span <= 60, "
          "0.189 at 61-80, 3.495 at 81-100, ~4.2 at 101-120, ~2.8 at 121-140")
    for lo, hi, rate in ((78, 80, 0.189), (81, 90, 3.495), (91, 100, 3.495)):
        m = (span >= lo) & (span <= hi) & ~free
        print("  band %3d-%3d : %d PAID classes x %.3f s = %.0f core-seconds"
              % (lo, hi, int(m.sum()), rate, int(m.sum()) * rate))


# ------------------------------------------------------------------ worker
def work(W, w, span_max, hours, node_budget=2_000_000):
    from crt_dict import decide_cover, gears_of, Budget
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "m41sh_w%d_%d.log" % (w, W))
    p, a, free = classes(span_max)
    # skip the free half entirely and everything already decided in round 27
    done = set()
    if os.path.exists(SHARD77):
        done |= set(key(load_arr(SHARD77)).tolist())
    prev = os.path.join(R27, "m41cen_w%d_5.log")
    for i in range(5):
        q = prev % i
        if os.path.exists(q):
            for line in open(q):
                pr = line.split()
                if len(pr) == 2 and pr[1] in "YNU":
                    t = tuple(int(x) for x in pr[0].split(","))
                    done.add(((t[0] * B + t[1]) * B + t[2]) * B + t[3])
    ks = key(a)
    mine = [i for i in range(len(a))
            if not free[i] and int(ks[i]) not in done and i % W == w]
    have = set()
    if os.path.exists(path):
        for line in open(path):
            pr = line.split()
            if len(pr) == 2 and pr[1] in "YNU":
                have.add(pr[0])
    qs = gears_of(41)
    fh = open(path, "a", buffering=1)
    stride = max(1, len(mine) // 20)
    deadline = time.time() + hours * 3600.0
    print("worker %d/%d: PAID share %d (free half and r27 decisions skipped), "
          "already logged %d, span <= %d, stride %d"
          % (w, W, len(mine), len(have), span_max, stride), flush=True)
    t0 = time.time()
    n = ny = nn = nu = 0
    for i in mine:
        t = tuple(int(x) for x in a[i])
        s = ",".join(str(x) for x in t)
        if s in have:
            continue
        if time.time() > deadline:
            print("  DEADLINE after %d decisions" % n, flush=True)
            break
        X = [0]
        for g in t:
            X.append(X[-1] + g)
        xs = set(X)
        Y = [z for z in range(1, X[-1]) if z not in xs]
        try:
            ok, _, _ = decide_cover(qs, X, Y, node_budget=node_budget)
            v = "Y" if ok else "N"
        except Budget:
            v = "U"
        fh.write("%s %s\n" % (s, v))
        n += 1
        ny += v == "Y"
        nn += v == "N"
        nu += v == "U"
        if n % stride == 0:
            print("  %d/%d span=%d Y=%d N=%d U=%d t=%.0fs"
                  % (n, len(mine), sum(t), ny, nn, nu, time.time() - t0),
                  flush=True)
    fh.close()
    print("worker %d/%d done: %d decided (Y=%d N=%d U=%d) in %.0fs"
          % (w, W, n, ny, nn, nu, time.time() - t0), flush=True)


# ------------------------------------------------------------------- merge
def merge(W, span_max):
    p, a, free = classes(span_max)
    ks = key(a)
    span = a.sum(axis=1)
    verdict = {}
    if os.path.exists(SHARD77):
        for k in key(load_arr(SHARD77)).tolist():
            verdict[int(k)] = "Y"
    for pat, nw in ((os.path.join(R27, "m41cen_w%d_5.log"), 5),
                    (os.path.join(OUT, "m41sh_w%d_%d.log"), W)):
        for i in range(nw):
            q = pat % ((i, W) if "m41sh" in pat else i)
            if not os.path.exists(q):
                continue
            for line in open(q):
                pr = line.split()
                if len(pr) == 2 and pr[1] in "YNU":
                    t = tuple(int(x) for x in pr[0].split(","))
                    kk = ((t[0] * B + t[1]) * B + t[2]) * B + t[3]
                    old = verdict.get(kk)
                    assert old is None or old == pr[1] or "U" in (old, pr[1]),\
                        ("verdict clash", t, old, pr[1])
                    if old is None or old == "U":
                        verdict[kk] = pr[1]
    # free half: YES by the depth-0 lemma
    nfree = 0
    for i in np.flatnonzero(free):
        kk = int(ks[i])
        if verdict.get(kk) not in ("Y",):
            assert verdict.get(kk) != "N", ("DEPTH-0 LEMMA CONTRADICTED", i)
        verdict[kk] = "Y"
        nfree += 1
    frontier = 0
    for s in range(1, int(span.max()) + 1):
        idx = np.flatnonzero(span == s)
        if len(idx) == 0:
            frontier = s
            continue
        if all(verdict.get(int(ks[i])) in ("Y", "N") for i in idx):
            frontier = s
        else:
            break
    rows = set()
    for i in range(len(a)):
        if span[i] <= frontier and verdict.get(int(ks[i])) == "Y":
            t = tuple(int(x) for x in a[i])
            rows.add(t)
            rows.add(t[::-1])
    out = os.path.join(OUT, "gap_tuples_41_4_exact_le%d.csv" % frontier)
    with open(out, "w") as fh:
        fh.write("g1,g2,g3,g4\n")
        for t in sorted(rows):
            fh.write("%d,%d,%d,%d\n" % t)
    print("verdicts %d (of which %d free by the depth-0 lemma)"
          % (len(verdict), nfree))
    print("COMPLETE-BY-SPAN FRONTIER: every 4-tuple of span <= %d is decided"
          % frontier)
    print("EXACT dictionary at span <= %d: %d tuples -> %s"
          % (frontier, len(rows), out))
    nsup = int((span <= frontier).sum()) * 2
    print("inflation of the superset over that region: %.4fx"
          % (nsup / max(1, len(rows))))
    return frontier


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "price":
        price()
    elif c == "work":
        work(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
             float(sys.argv[5]))
    elif c == "merge":
        merge(int(sys.argv[2]), int(sys.argv[3]))

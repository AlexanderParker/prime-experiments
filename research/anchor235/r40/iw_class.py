"""R2.a.i.a.1 - can the witness set be made smaller?  (items 4, 6, 7)

Two ways of squeezing the B = 7 island witness:

  (a) BY CLASS.  The witness set is the four classes 5, 10, 12, 17 (mod 35), density 4/35.  Does
      a single class already carry a free island for every prime q above some point?  That would
      shrink the witness set to one arithmetic progression of density 1/35.
  (b) BY ARC.  The witness asks for a free island in [1, d).  How short may the arc be made?
      Report max over q of (first free island)/d by band: an arc [1, theta d) that still always
      contains a free island bounds the walk by theta d, not d.

Writes results/iw_class.txt.
Usage: uv run python research/anchor235/r40/iw_class.py [--QMAX 200000] [--WORKERS 3]
"""
import argparse
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CLASSES = (5, 10, 12, 17)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


G = {}


def init(qmax):
    fl = sieve(qmax + 10)
    gears = np.array([p for p in range(5, qmax + 1) if fl[p]], dtype=np.int64)
    G["fl"] = fl
    G["gears"] = gears
    G["u"] = np.array([pow(6, -1, int(g)) for g in gears], dtype=np.int64)


def one_q(q):
    gears = G["gears"]
    u = G["u"]
    nq = int(np.searchsorted(gears, q, side="right"))
    qq = q * q
    d = (2 * pow(6, -1, q)) % q
    if d < 2:
        return None
    gl = gears[:nq]
    ul = u[:nq]
    r = qq % gl
    a = ((2 - r) * ul) % gl
    b = ((-r) * ul) % gl
    struck = np.zeros(d, dtype=bool)
    T = max(d // 8, 40)
    small = gl <= T
    for j in np.flatnonzero(small):
        g = int(gl[j])
        aa = int(a[j])
        bb = int(b[j])
        if aa < d:
            struck[aa::g] = True
        if bb < d:
            struck[bb::g] = True
    big = ~small
    if big.any():
        gb = gl[big]
        J = int(d // T) + 2
        for base in (a[big], b[big]):
            for jj in range(J):
                pos = base + jj * gb
                pos = pos[pos < d]
                if pos.size:
                    struck[pos] = True
    row = [q, d]
    firsts = []
    for c in CLASSES:
        cm = np.zeros(d, dtype=bool)
        cm[c::35] = True
        cm[0] = False
        fr = np.flatnonzero(cm & ~struck)
        row.append(len(fr))
        firsts.append(int(fr[0]) if len(fr) else -1)
    row += firsts
    return row


def run_chunk(args):
    qlist, qmax = args
    if "gears" not in G:
        init(qmax)
    out = []
    for q in qlist:
        r = one_q(q)
        if r is not None:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--QMAX", type=int, default=200000)
    ap.add_argument("--WORKERS", type=int, default=3)
    args = ap.parse_args()
    qmax = args.QMAX
    fl = sieve(qmax + 10)
    qs = [p for p in range(5, qmax + 1) if fl[p]]
    log = open(os.path.join(OUT, "iw_class.txt"), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")
        log.flush()

    W = args.WORKERS
    chunks = [(qs[w::W], qmax) for w in range(W)]
    if W > 1:
        import multiprocessing as mp
        with mp.Pool(W) as pool:
            parts = pool.map(run_chunk, chunks)
    else:
        parts = [run_chunk(c) for c in chunks]
    rows = []
    for p in parts:
        rows.extend(p)
    rows.sort()
    A = np.array(rows, dtype=np.int64)
    np.save(os.path.join(OUT, "iw_class.npy"), A)
    q = A[:, 0]
    d = A[:, 1]
    say("primes 5..%d: %d" % (qmax, len(A)))

    say("")
    say("=== (a) the witness restricted to ONE island class mod 35 ===")
    say(" class   primes with no free island of that class   largest such q   above 1487")
    for k, c in enumerate(CLASSES):
        f = A[:, 2 + k]
        bad = f == 0
        say("   %-6d %-42d %-16s %d"
            % (c, int(bad.sum()), int(q[bad].max()) if bad.any() else "-",
               int((bad & (q > 1487)).sum())))
    say("")
    say(" class   largest q with no free island of that class, by band")
    for k, c in enumerate(CLASSES):
        f = A[:, 2 + k]
        bad = f == 0
        line = []
        for lo, hi in ((1487, 5000), (5000, 20000), (20000, 50000), (50000, 200001)):
            sel = bad & (q > lo) & (q <= hi)
            line.append("%d-%d: %d" % (lo, hi, int(sel.sum())))
        say("   %-6d %s" % (c, "; ".join(line)))
        if bad.any():
            say("          largest: %s" % list(int(v) for v in q[bad][-8:]))

    say("")
    say("=== pairs of classes: which two classes together are exception-free above 1487? ===")
    for i in range(4):
        for j in range(i + 1, 4):
            f = A[:, 2 + i] + A[:, 2 + j]
            bad = (f == 0) & (q > 1487)
            say("  classes %2d and %2d: failures above 1487: %d %s"
                % (CLASSES[i], CLASSES[j], int(bad.sum()),
                   list(int(v) for v in q[bad][:10])))

    say("")
    say("=== (b) how short an arc still carries a free island ===")
    ff = A[:, 6:10]
    ff = np.where(ff < 0, 1 << 40, ff)
    first = ff.min(axis=1)
    ok = first < (1 << 39)
    frac = first[ok] / d[ok]
    say(" band            primes   max (first free)/d    at q      median   max absolute offset")
    for lo, hi in ((1487, 5000), (5000, 20000), (20000, 50000), (50000, 100000),
                   (100000, 200001)):
        sel = ok & (q > lo) & (q <= hi)
        if not sel.any():
            continue
        fr = first[sel] / d[sel]
        say(" %6d-%-8d %7d  %.4f              %-9d %.4f   %d"
            % (lo, hi, int(sel.sum()), fr.max(), int(q[sel][fr.argmax()]),
               float(np.median(fr)), int(first[sel].max())))
    sel = ok & (q > 1487)
    fr = first[sel] / d[sel]
    say("")
    say("over all primes q > 1487: max (first free island)/d = %.4f at q = %d"
        % (fr.max(), int(q[sel][fr.argmax()])))
    for thr in (0.1, 0.2, 0.3, 0.5):
        say("   free island inside [1, %.1f d): %d of %d primes (failures %s)"
            % (thr, int((fr < thr).sum()), int(sel.sum()),
               list(int(v) for v in q[sel][fr >= thr][:12])))
    log.close()


if __name__ == "__main__":
    main()

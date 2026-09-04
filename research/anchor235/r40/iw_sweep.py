"""R2.a.i.a.1 - the island witness under pressure, sweep 1.

For EVERY integer q with gcd(q, 6) = 1 in [5, QMAX] (primes and composites alike):

  d      = 2 * 6^-1 (mod q)                    the top gear's forward tooth arc
  gear g = every prime 5 <= g <= q             the machine
  g strikes offset i  iff  i = (2 - q^2) u_g  or  i = -q^2 u_g   (mod g),  u_g = 6^-1 mod g
  island for bound B  = an offset no gear <= B can strike at any q   (the quadratic-residue bar)

and reports, for B = 7, 11, 13: how many islands lie in [1, d), how many of them are struck by no
gear at all (FREE islands - each one is an opening, so free >= 1 witnesses L < d), and the first
free one.  Composites are included so that the witness can be tested as a statement about the two
quadratics q^2 + 6i - 2 and q^2 + 6i with no primality hypothesis on q.

Writes results/iw_sweep.txt and results/iw_sweep.npz.
Usage: uv run python research/anchor235/r40/iw_sweep.py [--QMAX 100000] [--WORKERS 4]
"""
import argparse
import os
from math import isqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R39 = os.path.join(os.path.dirname(HERE), "r39", "results")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

BS = (7, 11, 13)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def island_masks(dmax):
    """boolean mask over offsets 0..dmax-1 of the island set for each bound B."""
    m = {}
    for B in BS:
        res = np.load(os.path.join(R39, "rl_isl_%d.npy" % B))
        mod = int(open(os.path.join(R39, "rl_isl_%d_mod.txt" % B)).read())
        mask = np.zeros(dmax, dtype=bool)
        rs = np.array(sorted(int(v) for v in res), dtype=np.int64)
        for base in range(0, dmax, mod):
            idx = rs + base
            mask[idx[idx < dmax]] = True
        m[B] = mask
    return m


G = {}


def init(qmax, dmax):
    fl = sieve(qmax + 10)
    gears = np.array([p for p in range(5, qmax + 1) if fl[p]], dtype=np.int64)
    G["fl"] = fl
    G["gears"] = gears
    G["u"] = np.array([pow(6, -1, int(g)) for g in gears], dtype=np.int64)
    G["mask"] = island_masks(dmax)


def one_q(q):
    """returns (q, isprime, d, [n_B, free_B, firstfree_B for B in BS], free positions)."""
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
    out = [q, 1 if G["fl"][q] else 0, d]
    freepos7 = None
    for B in BS:
        isl = G["mask"][B][:d].copy()
        isl[0] = False
        n = int(isl.sum())
        fr = np.flatnonzero(isl & ~struck)
        out += [n, len(fr), int(fr[0]) if len(fr) else -1]
        if B == 7:
            freepos7 = fr
    return out, freepos7


def run_chunk(args):
    qlist, qmax, dmax = args
    if "gears" not in G:
        init(qmax, dmax)
    rows = []
    enr = np.zeros(3, dtype=np.int64)  # free7 total, of which B=11 islands, of which B=13
    for q in qlist:
        res = one_q(q)
        if res is None:
            continue
        row, fr = res
        rows.append(row)
        if len(fr):
            enr[0] += len(fr)
            enr[1] += int(G["mask"][11][fr].sum())
            enr[2] += int(G["mask"][13][fr].sum())
    return rows, enr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--QMAX", type=int, default=100000)
    ap.add_argument("--WORKERS", type=int, default=4)
    args = ap.parse_args()
    qmax = args.QMAX
    dmax = (2 * qmax + 1) // 3 + 4
    qs = [q for q in range(5, qmax + 1) if q % 2 and q % 3]

    log = open(os.path.join(OUT, "iw_sweep.txt"), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")

    say("q coprime to 6 in [5, %d]: %d values;  d_max = %d" % (qmax, len(qs), dmax))

    chunks = []
    W = args.WORKERS
    # interleave so every worker gets a spread of q (cost grows with q)
    for w in range(W):
        chunks.append((qs[w::W], qmax, dmax))

    if W > 1:
        import multiprocessing as mp
        with mp.Pool(W) as pool:
            parts = pool.map(run_chunk, chunks)
    else:
        parts = [run_chunk(c) for c in chunks]

    rows = []
    enr = np.zeros(3, dtype=np.int64)
    for r, e in parts:
        rows.extend(r)
        enr += e
    rows.sort()
    A = np.array(rows, dtype=np.int64)
    np.savez_compressed(os.path.join(OUT, "iw_sweep.npz"), rows=A, enr=enr)
    say("walks measured: %d" % len(A))

    col = {"q": 0, "prime": 1, "d": 2}
    for k, B in enumerate(BS):
        col["n%d" % B] = 3 + 3 * k
        col["f%d" % B] = 4 + 3 * k
        col["ff%d" % B] = 5 + 3 * k

    q = A[:, col["q"]]
    isp = A[:, col["prime"]].astype(bool)
    say("")
    say("=== 1. the witness over ALL integers coprime to 6 (B = 7) ===")
    f7 = A[:, col["f7"]]
    n7 = A[:, col["n7"]]
    fail = f7 == 0
    say("integers with at least one island in [1,d): %d;  with none: %d"
        % (int((n7 > 0).sum()), int((n7 == 0).sum())))
    say("integers whose islands in [1,d) are ALL struck (witness fails): %d of %d"
        % (int(fail.sum()), len(A)))
    say("largest failing integer: %d" % int(q[fail].max()))
    say("")
    say("split by gcd(q, 35):")
    for gv, name in ((1, "coprime to 35"), (5, "5 | q, 7 !| q"), (7, "7 | q, 5 !| q"),
                     (35, "35 | q")):
        sel = np.gcd(q, 35) == gv
        if not sel.any():
            continue
        fl = fail & sel
        say("  gcd = %2d (%-14s): %6d integers, %6d fail (%.4f), largest failure %s"
            % (gv, name, int(sel.sum()), int(fl.sum()), fl.sum() / sel.sum(),
               int(q[fl].max()) if fl.any() else "none"))
    say("")
    say("failures coprime to 35, all of them:")
    sel = (np.gcd(q, 35) == 1) & fail
    say("  %s" % list(int(v) for v in q[sel]))
    say("  of which prime: %s" % list(int(v) for v in q[sel & isp]))
    say("  of which composite: %s" % list(int(v) for v in q[sel & ~isp]))

    say("")
    say("=== 2. primes only, B = 7 (N-R4 extended) ===")
    fp = fail & isp
    say("primes in [5, %d]: %d;  failures: %d;  list: %s"
        % (qmax, int(isp.sum()), int(fp.sum()), list(int(v) for v in q[fp])))
    above = fp & (q > 1487)
    say("prime failures above q = 1487: %d" % int(above.sum()))

    say("")
    say("=== 3. composites vs primes coprime to 30, by band ===")
    say(" band            primes  fail   rate      composites  fail   rate")
    bands = [(5, 100), (100, 1000), (1000, 5000), (5000, 20000), (20000, 50000),
             (50000, 100001)]
    cop = np.gcd(q, 30) == 1
    for lo, hi in bands:
        sel = (q >= lo) & (q < hi) & cop
        p = sel & isp
        c = sel & ~isp
        say(" %6d-%-6d %8d %5d  %.5f %10d %6d  %.5f"
            % (lo, hi, int(p.sum()), int((p & fail).sum()),
               (p & fail).sum() / max(1, p.sum()),
               int(c.sum()), int((c & fail).sum()),
               (c & fail).sum() / max(1, c.sum())))

    say("")
    say("=== 4. the slack law: free islands per prime q, by band (B = 7) ===")
    say(" band              primes   min free   median   mean     max    min islands")
    for lo, hi in [(1487, 5000), (5000, 10000), (10000, 20000), (20000, 50000),
                   (50000, 100001)]:
        sel = isp & (q > lo) & (q <= hi)
        if not sel.any():
            continue
        v = f7[sel]
        say(" %6d-%-6d %10d %10d %8d %8.2f %7d %10d"
            % (lo, hi, int(sel.sum()), int(v.min()), int(np.median(v)), float(v.mean()),
               int(v.max()), int(n7[sel].min())))
    for lo, hi in [(1487, 5000), (5000, 10000), (10000, 20000), (20000, 50000),
                   (50000, 100001)]:
        sel = isp & (q > lo) & (q <= hi)
        if not sel.any():
            continue
        v = f7[sel]
        say("   band %6d-%-6d: argmin q = %d (free %d, islands %d, d = %d)"
            % (lo, hi, int(q[sel][v.argmin()]), int(v.min()),
               int(n7[sel][v.argmin()]), int(A[sel, col["d"]][v.argmin()])))

    say("")
    say("=== 5. B = 11 and B = 13 (primes only) ===")
    for B in (7, 11, 13):
        fB = A[:, col["f%d" % B]]
        nB = A[:, col["n%d" % B]]
        fl = isp & (fB == 0) & (nB > 0)
        none = isp & (nB == 0)
        qq = q[fl]
        say("B = %2d: primes with no island in [1,d): %d (largest %s);  failures: %d; largest %s"
            % (B, int(none.sum()), int(q[none].max()) if none.any() else "-",
               int(fl.sum()), int(qq.max()) if fl.any() else "-"))
        if fl.sum() <= 40:
            say("        failures: %s" % list(int(v) for v in qq))
        else:
            say("        first 20: %s" % list(int(v) for v in qq[:20]))
            say("        last  20: %s" % list(int(v) for v in qq[-20:]))
    # nesting
    f7m = (A[:, col["f7"]] == 0) & isp
    f11m = (A[:, col["f11"]] == 0) & isp
    f13m = (A[:, col["f13"]] == 0) & isp
    say("")
    say("nesting Fail_7 subset Fail_11: exceptions %d" % int((f7m & ~f11m).sum()))
    say("nesting Fail_11 subset Fail_13: exceptions %d" % int((f11m & ~f13m).sum()))
    say("brief's direction Fail_13 subset Fail_7: exceptions %d (counterexamples)"
        % int((f13m & ~f7m).sum()))
    say("brief's direction Fail_13 subset Fail_11: exceptions %d" % int((f13m & ~f11m).sum()))

    say("")
    say("=== 6. enrichment of the free islands on higher-B islands (W12) ===")
    say("free B=7 islands counted: %d;  of which B=11 islands: %d (%.4f, base 12/44 = %.4f)"
        % (enr[0], enr[1], enr[1] / enr[0], 12 / 44))
    say("                          of which B=13 islands: %d (%.4f, base 48/(4*143) = %.4f)"
        % (enr[2], enr[2] / enr[0], 48 / (4 * 143)))
    log.close()


if __name__ == "__main__":
    main()

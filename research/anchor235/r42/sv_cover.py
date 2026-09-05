"""R2.a.i.a.1.b item S10 - the global-square test on a cover.

A cover C = {g_1..g_K} with phases r_j forces q^2 = r_j (mod g_j), so it pins one residue R
modulo P = prod g_j; when P > q^2 that residue determines q^2 as an integer (N-C7).  Two parts:

  (A) CONTROL - every REAL failure q (integers coprime to 30 up to QMAX): take the exact minimum
      cover of the islands of [1,d) at the real phases and check that its CRT lift R is exactly
      q^2, i.e. a perfect square, and hence a quadratic residue modulo every gear outside C.

  (B) THE TEST - locally-square phase vectors at a q0 past the last real failure.  For each
      FAILING vector take the exact minimum cover, its lift R, and ask: is R a perfect square?
      is R a nonzero quadratic residue modulo the next t gears outside the cover?

Usage: uv run python research/anchor235/r42/sv_cover.py [--Q0 2861] [--N 2000000] [--BATCH 100000]
"""
import argparse
import json
import os
from math import isqrt, prod

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def incidence(isl, gl, us, rvec):
    """rows = gears, entries = island indices struck."""
    inc = []
    for gi, g in enumerate(gl):
        u = us[g]
        r = int(rvec[gi]) % g
        b = (-r * u) % g
        a = (b + 2 * u) % g
        s = [j for j, i in enumerate(isl) if i % g == a or i % g == b]
        inc.append(s)
    return inc


def min_cover(inc, m, ng):
    rowsi, colsi = [], []
    for gi, s in enumerate(inc):
        for j in s:
            rowsi.append(j)
            colsi.append(gi)
    if len(set(rowsi)) < m:
        return None
    A = csr_matrix((np.ones(len(rowsi)), (rowsi, colsi)), shape=(m, ng))
    res = milp(c=np.ones(ng), constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(ng), bounds=Bounds(0, 1))
    if not res.success:
        return None
    return [i for i in range(ng) if res.x[i] > 0.5]


def crt(rs, ms):
    R, M = 0, 1
    for r, mm in zip(rs, ms):
        g = pow(M % mm, -1, mm)
        t = ((r - R) % mm) * g % mm
        R += M * t
        M *= mm
    return R, M


def is_sq(n):
    r = isqrt(n)
    return r * r == n


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q0", type=int, default=2861)
    ap.add_argument("--N", type=int, default=2000000)
    ap.add_argument("--BATCH", type=int, default=100000)
    ap.add_argument("--QMAX", type=int, default=3000)
    ap.add_argument("--TMAX", type=int, default=24)
    ap.add_argument("--MAXCOV", type=int, default=120)
    a = ap.parse_args()
    fl = sieve_np(max(a.Q0, a.QMAX) + 10)
    allp = np.flatnonzero(fl).astype(np.int64)
    lines = []

    # ---------------- (A) control: the real failures ----------------
    lines.append("(A) CONTROL - every real failure q coprime to 30 up to %d" % a.QMAX)
    lines.append("   q     d    m   K   log10 P  log10 P/q^2  R == q^2 ?")
    nA = okA = 0
    for q in range(11, a.QMAX + 1):
        if q % 2 == 0 or q % 3 == 0 or q % 5 == 0:
            continue
        d = (2 * pow(6, -1, q)) % q
        if d < 2:
            continue
        isl = islands(d)
        m = len(isl)
        if m == 0:
            continue
        gl = [int(g) for g in allp if 7 < g <= q]
        us = {g: pow(6, -1, g) for g in gl}
        rvec = [(q * q) % g for g in gl]
        inc = incidence(isl, gl, us, rvec)
        cov = min_cover(inc, m, len(gl))
        if cov is None:
            continue                      # some island open: not a failure
        gs = [gl[i] for i in cov]
        rs = [rvec[i] for i in cov]
        R, P = crt(rs, gs)
        ok = (R == q * q)
        nA += 1
        okA += ok
        lines.append("  %5d %5d %4d %3d  %8.3f  %10.3f    %s"
                     % (q, d, m, len(cov), np.log10(float(P)), np.log10(float(P)) - 2 * np.log10(q), ok))
    lines.append("   real failures with R == q^2 exactly: %d of %d" % (okA, nA))
    print("\n".join(lines), flush=True)

    # ---------------- (B) failing locally-square vectors ----------------
    q0 = a.Q0
    d = (2 * pow(6, -1, q0)) % q0
    isl = islands(d)
    m = len(isl)
    posl = np.full(d, -1, dtype=np.int64)
    posl[np.array(isl)] = np.arange(m)
    gl = [int(g) for g in allp if 7 < g < q0]      # top gear inert, as in the real walk
    us = {g: pow(6, -1, g) for g in gl}
    ng = len(gl)
    lines.append("")
    lines.append("(B) locally-square vectors at q0 = %d : d = %d, m = %d islands, %d gears (11..%d)"
                 % (q0, d, m, ng, gl[-1]))
    rng = np.random.default_rng(31337)
    fails = []
    done = 0
    while done < a.N and len(fails) < a.MAXCOV:
        n = min(a.BATCH, a.N - done)
        R_ = np.zeros((n, ng), dtype=np.int32)
        struck = np.zeros((n, m), dtype=bool)
        rws = np.arange(n)
        for gi, g in enumerate(gl):
            u = us[g]
            s = rng.integers(1, g, n)
            r = (s * s) % g
            R_[:, gi] = r
            b = (-r * u) % g
            aa = (b + 2 * u) % g
            for base in (aa, b):
                k = 0
                while True:
                    vals = base + k * g
                    sel = vals < d
                    if not sel.any():
                        break
                    v = vals[sel]
                    j = posl[v]
                    ok = j >= 0
                    if ok.any():
                        struck[rws[sel][ok], j[ok]] = True
                    k += 1
        nopen = m - struck.sum(axis=1)
        bad = np.flatnonzero(nopen == 0)
        for bi in bad:
            fails.append(R_[bi].copy())
        done += n
        print("   batch done %d/%d, failing vectors so far %d" % (done, a.N, len(fails)), flush=True)
    lines.append("   %d failing vectors in %d draws (rate %.3g)" % (len(fails), done, len(fails) / done))

    surv = np.zeros(a.TMAX + 1, dtype=np.int64)
    nsq = 0
    Ks, PQ = [], []
    for rvec in fails:
        inc = incidence(isl, gl, us, rvec)
        cov = min_cover(inc, m, ng)
        if cov is None:
            continue
        gs = [gl[i] for i in cov]
        rs = [int(rvec[i]) for i in cov]
        R, P = crt(rs, gs)
        Ks.append(len(cov))
        PQ.append(np.log10(float(P)) - 2 * np.log10(q0))
        if is_sq(R):
            nsq += 1
        outside = [g for g in gl if g not in set(gs)]
        t = 0
        for g in outside:
            if t >= a.TMAX:
                break
            if legendre(R, g) == 1:
                t += 1
            else:
                break
        for tt in range(0, min(t, a.TMAX) + 1):
            surv[tt] += 1
    nc = len(Ks)
    lines.append("   exact minimum covers computed: %d ; K mean %.2f min %d max %d ; log10 P/q0^2 mean %.2f min %.2f"
                 % (nc, float(np.mean(Ks)), min(Ks), max(Ks), float(np.mean(PQ)), float(np.min(PQ))))
    lines.append("   covers whose CRT lift R is a PERFECT SQUARE: %d of %d" % (nsq, nc))
    lines.append("   QR screen - R a nonzero QR modulo the first t gears outside the cover:")
    lines.append("      t   survivors   fraction   2^-t")
    for t in range(0, a.TMAX + 1):
        lines.append("     %3d %10d   %8.5f  %8.5f" % (t, surv[t], surv[t] / nc, 2.0 ** -t))
    txt = "\n".join(lines)
    print(txt[-3000:], flush=True)
    open(os.path.join(OUT, "sv_cover.txt"), "w").write(txt + "\n")
    json.dump(dict(nA=nA, okA=okA, nfail=len(fails), draws=done, K=Ks, nsq=nsq,
                   surv=surv.tolist()), open(os.path.join(OUT, "sv_cover.json"), "w"))


if __name__ == "__main__":
    main()

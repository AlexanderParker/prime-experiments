"""R2.a.i.a.1.b - the deciding experiment: real vs locally-square vs random phase vectors.

Fix a prime q0.  That fixes the arc d = 2*6^-1 mod q0, the gear set G = {primes 7 < g <= q0} and
the island set I = {i in [1,d) : i mod 35 in {5,10,12,17}}.  Gears 5 and 7 keep square phases in
every kind (they can strike no island, by the definition of the island set); they are included
only for the walk length L.  Five kinds of phase vector on the SAME d, G, I:

  REAL       r_g = q^2 mod g for a prime q > 6000       (one integer's square at every gear)
  REALNEAR   the same for primes q0 < q <= 20*q0
  LS         r_g uniform on the (g-1)/2 nonzero squares mod g, independent across gears
  RND        r_g uniform on the g-1 nonzero residues mod g, independent across gears
  LSI        LS, but the top gear q0 is inert (r_{q0} = 0), as in the real walk

Per vector: number of open islands, failure (0 open), walk length L (first open offset >= 1,
computed on the first NL vectors only).

Usage: uv run python research/anchor235/r42/sv_main.py [--NL 5000] [--W 256]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

BANDS = {
    200: ([179, 191, 197, 181, 193, 199], 300000),
    500: ([491, 503, 509, 463, 487, 499], 300000),
    1000: ([971, 983, 1013, 991, 997, 1009], 300000),
    2000: ([1997, 2003, 2027, 1999, 2011, 2017], 100000),
    5000: ([5003, 5009, 5021, 4993, 4999, 5011], 50000),
}
KINDS = ("REAL", "REALNEAR", "LS", "RND", "LSI")


def sieve_np(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def arc(q):
    return (2 * pow(6, -1, q)) % q


def islands(d):
    return np.array([i for i in range(1, d) if i % 35 in (5, 10, 12, 17)], dtype=np.int64)


def mark_islands(struck, posl, base, g, d, rows):
    k = 0
    while True:
        vals = base + k * g
        sel = vals < d
        if not sel.any():
            return
        v = vals[sel]
        j = posl[v]
        ok = j >= 0
        if ok.any():
            struck[rows[sel][ok], j[ok]] = True
        k += 1


def mark_offsets(soff, base, g, W, rows):
    k = 0
    while True:
        vals = base + k * g
        sel = vals < W
        if not sel.any():
            return
        soff[rows[sel], vals[sel]] = True
        k += 1


def phases(kind, g, u, dg, n, rng, qsq, q0):
    if kind in ("REAL", "REALNEAR"):
        r = qsq % g
    elif kind == "RND":
        r = rng.integers(1, g, n)
    else:
        if kind == "LSI" and g == q0:
            return None
        s = rng.integers(1, g, n)
        r = (s * s) % g
    b = (-r * u) % g
    a = (b + dg) % g
    return a, b


def run_one(q0, gears5, uall, N, NL, W, rng, qfar, qnear):
    d = arc(q0)
    isl = islands(d)
    m = len(isl)
    posl = np.full(d, -1, dtype=np.int64)
    posl[isl] = np.arange(m)
    idx = int(np.searchsorted(gears5, q0, side="right"))
    gl, ul = gears5[:idx], uall[:idx]
    res = {}
    for kind in KINDS:
        qs = qfar if kind == "REAL" else (qnear if kind == "REALNEAR" else None)
        n = len(qs) if qs is not None else N
        if n == 0:
            continue
        qsq = qs.astype(np.int64) ** 2 if qs is not None else None
        nl = min(NL, n)
        rows = np.arange(n)
        rowsL = rows[:nl]
        struck = np.zeros((n, m), dtype=bool)
        soff = np.zeros((nl, W), dtype=bool)
        for j in range(len(gl)):
            g, u = int(gl[j]), int(ul[j])
            dg = (2 * u) % g
            ab = phases(kind, g, u, dg, n, rng, qsq, q0)
            if ab is None:
                continue
            a, b = ab
            if g > 7:
                mark_islands(struck, posl, a, g, d, rows)
                mark_islands(struck, posl, b, g, d, rows)
            mark_offsets(soff, a[:nl], g, W, rowsL)
            mark_offsets(soff, b[:nl], g, W, rowsL)
        nopen = m - struck.sum(axis=1)
        free = ~soff[:, 1:W]
        has = free.any(axis=1)
        L = np.where(has, 1 + free.argmax(axis=1), W)
        res[kind] = dict(
            n=int(n), nL=int(nl), m=int(m), d=int(d),
            fail=int((nopen == 0).sum()),
            open_mean=float(nopen.mean()), open_sd=float(nopen.std()), open_min=int(nopen.min()),
            open_p01=float(np.percentile(nopen, 1)),
            L_mean=float(L.mean()), L_med=float(np.median(L)), L_max=int(L.max()),
            L_cens=int((L == W).sum()),
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NL", type=int, default=5000)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--QP", type=int, default=5000000)
    a = ap.parse_args()
    fl = sieve_np(a.QP)
    allp = np.flatnonzero(fl).astype(np.int64)
    gears5 = allp[(allp >= 5) & (allp < 6000)]
    uall = np.array([pow(6, -1, int(g)) for g in gears5], dtype=np.int64)
    rng = np.random.default_rng(20260906)
    lines, rows_out = [], []
    for Q, (qlist, N) in BANDS.items():
        for q0 in qlist:
            far = allp[allp > 6000]
            if len(far) > N:
                far = far[np.linspace(0, len(far) - 1, N).astype(np.int64)]
            near = allp[(allp > q0) & (allp <= 20 * q0)]
            if len(near) > N:
                near = near[np.linspace(0, len(near) - 1, N).astype(np.int64)]
            res = run_one(q0, gears5, uall, N, a.NL, a.W, rng, far, near)
            for kind, r in res.items():
                r.update(band=Q, q0=int(q0), kind=kind, arc=("short" if q0 % 6 == 5 else "long"))
                rows_out.append(r)
                lines.append(
                    "band %5d q0 %5d %-5s d %5d m %4d | %-8s n %7d fail %7d (%.6g) open mean %8.3f sd %6.3f min %3d | L mean %6.2f med %5.1f max %4d cens %d"
                    % (Q, q0, r["arc"], r["d"], r["m"], kind, r["n"], r["fail"], r["fail"] / r["n"],
                       r["open_mean"], r["open_sd"], r["open_min"],
                       r["L_mean"], r["L_med"], r["L_max"], r["L_cens"]))
                print(lines[-1], flush=True)
    open(os.path.join(OUT, "sv_main.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(rows_out, open(os.path.join(OUT, "sv_main.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

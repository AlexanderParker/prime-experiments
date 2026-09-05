"""sr_rung31.py -- the recursion at the top rung 29 -> 31, streamed over the m29 period.

The m29 period is 1,078,282,205 columns and 214,708,725 openings, too many to hold as one
array of positions, so the period is sieved in chunks and the run enumeration is carried
across chunk boundaries with an overlap of JCAP openings.  Produces the exact m31 spectrum
with the survival / merge split, gated against the corpus values
(F=58, |Spec|=55, m(4)=398,923,200, m(6)=299,202,120, m(24)=174,704, m(36)=3,152, m(41)=134).

Writes results/sr_rung31.txt
"""
import os, sys, json, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
GEARS = [5, 7, 11, 13, 17, 19, 23, 29]
QP = 31
JCAP = 40
CHUNK = 40_000_000


def u_of(g):
    return pow(6, -1, g)


def c_local(q, v):
    u = u_of(q)
    s = {u % q, (-u) % q, (u - v) % q, ((-u) - v) % q}
    return q - len(s)


def main():
    t0 = time.time()
    P = 1
    for g in GEARS:
        P *= g
    d = (2 * u_of(QP)) % QP
    surv_acc = {}
    merge_acc = {}
    mergeJ = {}
    cvals = [c_local(QP, v) for v in range(200)]

    buf = np.empty(0, dtype=np.int64)
    lo = 0
    while lo < P:
        hi = min(lo + CHUNK, P)
        n = hi - lo
        blocked = np.zeros(n, dtype=bool)
        for g in GEARS:
            u = u_of(g)
            for t in (u % g, (-u) % g):
                blocked[(t - lo) % g:: g] = True
        op = np.flatnonzero(~blocked).astype(np.int64) + lo
        del blocked
        buf = np.concatenate([buf, op])
        del op
        nstart = buf.size - JCAP
        if nstart > 0:
            process(buf, nstart, d, cvals, surv_acc, merge_acc, mergeJ)
            buf = buf[nstart:]
        lo = hi
        print(f"  ..{hi/P*100:5.1f}%  {time.time()-t0:6.1f}s", flush=True)

    # wrap: the last JCAP openings of the period start runs continuing into the next copy
    _, head = sieve_head(GEARS, JCAP + 5)
    tail = np.concatenate([buf, head + P])
    process(tail, buf.size, d, cvals, surv_acc, merge_acc, mergeJ)

    m = {}
    for v, c in surv_acc.items():
        m[v] = m.get(v, 0) + c
    for v, c in merge_acc.items():
        m[v] = m.get(v, 0) + c
    Nn = sum(m.values())
    Ln = sum(v * c for v, c in m.items())
    Nex, Pex = 1, 1
    for g in GEARS + [QP]:
        Nex *= g - 2
        Pex *= g
    lines = []
    W = lines.append
    W(f"=== rung 29 -> 31   q'=31 d={d} letters {{{min(d,31-d)}, {max(d,31-d)}}}")
    W(f"  F(m31) = {max(m)}   |Spec| = {len(m)}")
    W(f"  sum m   = {Nn}  vs prod(q-2) = {Nex}   {'OK' if Nn==Nex else 'FAIL'}")
    W(f"  sum v m = {Ln}  vs P' = {Pex}   {'OK' if Ln==Pex else 'FAIL'}")
    W(f"  absent below F: {[v for v in range(1, max(m)) if v not in m]}")
    for v in (4, 6, 24, 36, 41, 23, 25, 35, 37):
        W(f"  m(31;{v}) = {m.get(v,0)}  = survival {surv_acc.get(v,0)} + merge {merge_acc.get(v,0)}")
    W("  spectrum: " + " ".join(f"{v}:{m[v]}" for v in sorted(m)))
    W("  survival: " + " ".join(f"{v}:{surv_acc.get(v,0)}" for v in sorted(m)))
    W(f"  merge mass by J: {{{', '.join(f'{J}: {sum(a.values())}' for J, a in sorted(mergeJ.items()))}}}")
    for J, a in sorted(mergeJ.items()):
        W(f"  merge J={J} by size: " + " ".join(f"{v}:{a[v]}" for v in sorted(a)))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_rung31.txt"), "w").write(txt)
    json.dump({"m": {str(k): v for k, v in m.items()},
               "surv": {str(k): v for k, v in surv_acc.items()},
               "merge_by_J": {str(J): {str(k): c for k, c in a.items()} for J, a in mergeJ.items()}},
              open(os.path.join(OUT, "spec_rec_m31.json"), "w"))
    print(txt)
    print(f"total {time.time()-t0:.1f}s")


def sieve_head(gears, k):
    P = 1
    for g in gears:
        P *= g
    n = 4000
    blocked = np.zeros(n, dtype=bool)
    for g in gears:
        u = u_of(g)
        for t in (u % g, (-u) % g):
            blocked[t % g:: g] = True
    op = np.flatnonzero(~blocked).astype(np.int64)
    return P, op[:k]


def process(buf, nstart, d, cvals, surv_acc, merge_acc, mergeJ):
    qp = QP
    res = (buf % qp).astype(np.int64)
    idx = np.arange(nstart, dtype=np.int64)
    g1 = buf[1:nstart + 1] - buf[:nstart]
    uq, inv = np.unique(g1, return_inverse=True)
    wt = np.array([cvals[int(v)] for v in uq], dtype=np.int64)[inv]
    bc = np.bincount(g1, weights=wt)
    for v in np.flatnonzero(bc):
        surv_acc[int(v)] = surv_acc.get(int(v), 0) + int(bc[v])
    y = res[idx + 1]
    ca = y.copy()
    cb = (y - d) % qp
    va = np.ones(nstart, dtype=bool)
    vb = np.ones(nstart, dtype=bool)
    for J in range(2, JCAP):
        e0 = res[idx]
        eJ = res[idx + J]
        cad = (ca + d) % qp
        cbd = (cb + d) % qp
        wa = va & (e0 != ca) & (e0 != cad) & (eJ != ca) & (eJ != cad)
        wb = vb & (e0 != cb) & (e0 != cbd) & (eJ != cb) & (eJ != cbd)
        w = wa.astype(np.int64) + wb.astype(np.int64)
        nz = w > 0
        if nz.any():
            span = buf[idx[nz] + J] - buf[idx[nz]]
            bc = np.bincount(span, weights=w[nz])
            tot = 0
            for v in np.flatnonzero(bc):
                merge_acc[int(v)] = merge_acc.get(int(v), 0) + int(bc[v])
                tot += int(bc[v])
                dj = mergeJ.setdefault(J, {})
                dj[int(v)] = dj.get(int(v), 0) + int(bc[v])
        va = va & ((eJ == ca) | (eJ == cad))
        vb = vb & ((eJ == cb) | (eJ == cbd))
        alive = va | vb
        if not alive.any():
            return
        idx = idx[alive]
        ca, cb, va, vb = ca[alive], cb[alive], va[alive], vb[alive]
    raise RuntimeError("JCAP reached")


if __name__ == "__main__":
    main()

"""sr_recursion.py -- the exact recursion for the gap-multiplicity function under adding a gear.

    m_{M+q'}(v) = c_{q'}(v) . m_M(v)  +  Merge_{q'}(v)

with c_{q'}(v) = q'-2 if q'|v, q'-3 if v = +-d (mod q'), q'-4 otherwise (the survival coefficient,
which is exactly the local factor of the autocorrelation count), and Merge the weighted count of
J >= 2 runs of M of span v, weight(R) = #{phases r : every interior of R lies in {r, r+d} and
neither end does}.

Verified rung by rung against the directly sieved spectrum of M'.  Reports the survival/merge
split by size and the merge contribution by J.

Writes results/sr_recursion.txt and results/spec_rec_m*.json
"""
import os, sys, json
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
PR = [5, 7, 11, 13, 17, 19, 23, 29]
JCAP = 60


def u_of(g):
    return pow(6, -1, g)


def c_local(q, v):
    u = u_of(q)
    s = {u % q, (-u) % q, (u - v) % q, ((-u) - v) % q}
    return q - len(s)


def sieve_opens(gears):
    P = 1
    for g in gears:
        P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        u = u_of(g)
        blocked[u % g:: g] = True
        blocked[(-u) % g:: g] = True
    return P, np.flatnonzero(~blocked).astype(np.int64)


def recursion(opens, P, qp, verbose=False):
    """Return (m_new, survival, merge_by_J) as dicts size -> count."""
    N = opens.size
    d = (2 * u_of(qp)) % qp
    ext = np.concatenate([opens, opens[:JCAP] + P])
    res = (ext % qp).astype(np.int64)

    # J = 1 : survival of old gaps
    g1 = (ext[1:N + 1] - ext[:N]).astype(np.int64)
    Fold = int(g1.max())
    cvals = np.array([c_local(qp, v) for v in range(Fold + 1)], dtype=np.int64)
    surv_w = cvals[g1]
    surv = np.bincount(g1, weights=surv_w).astype(object)

    merge_by_J = {}
    total = surv.copy()

    # J >= 2
    idx = np.arange(N, dtype=np.int64)
    # candidates after the first interior (opening idx+1)
    y = res[idx + 1]
    ca = y.copy()
    cb = (y - d) % qp
    va = np.ones(N, dtype=bool)
    vb = np.ones(N, dtype=bool)
    for J in range(2, JCAP):
        e0 = res[idx]
        eJ = res[idx + J]
        # a candidate r is usable if neither end lies in {r, r+d}
        def ok(c):
            return (e0 != c) & (e0 != (c + d) % qp) & (eJ != c) & (eJ != (c + d) % qp)
        wa = va & ok(ca)
        wb = vb & ok(cb)
        w = wa.astype(np.int64) + wb.astype(np.int64)
        span = (ext[idx + J] - ext[idx]).astype(np.int64)
        nz = w > 0
        if nz.any():
            mb = np.bincount(span[nz], weights=w[nz]).astype(object)
            merge_by_J[J] = mb
            if mb.size > total.size:
                total = np.concatenate([total, np.zeros(mb.size - total.size, dtype=object)])
            total[:mb.size] += mb
        # extend: opening idx+J becomes an interior
        yJ = res[idx + J]
        va = va & ((yJ == ca) | (yJ == (ca + d) % qp))
        vb = vb & ((yJ == cb) | (yJ == (cb + d) % qp))
        alive = va | vb
        if not alive.any():
            break
        idx = idx[alive]
        ca, cb, va, vb = ca[alive], cb[alive], va[alive], vb[alive]
    else:
        raise RuntimeError("JCAP reached")
    Jtop = J
    return total, surv, merge_by_J, Jtop


def as_dict(arr):
    return {int(i): int(arr[i]) for i in range(len(arr)) if arr[i]}


def main():
    lines = []
    W = lines.append
    for i in range(1, len(PR)):
        gears = PR[:i]
        qp = PR[i]
        P, opens = sieve_opens(gears)
        total, surv, mbj, Jtop = recursion(opens, P, qp)
        mnew = as_dict(total)
        sdict = as_dict(surv)
        Pn = P * qp
        Nn = sum(mnew.values())
        Ln = sum(v * c for v, c in mnew.items())
        Nex = 1
        for g in gears + [qp]:
            Nex *= g - 2
        W(f"=== rung {gears[-1]} -> {qp}   (M = {{5..{gears[-1]}}}, q'={qp}, d={(2*u_of(qp))%qp}, "
          f"letters {{{min((2*u_of(qp))%qp, qp-(2*u_of(qp))%qp)}, {max((2*u_of(qp))%qp, qp-(2*u_of(qp))%qp)}}})")
        W(f"  recursion gives F(M')={max(mnew)}  |Spec|={len(mnew)}  sum m={Nn} (prod(q-2)={Nex}, "
          f"{'OK' if Nn == Nex else 'FAIL'})  sum v m={Ln} (P'={Pn}, {'OK' if Ln == Pn else 'FAIL'})")
        W(f"  longest surviving J-run: J = {Jtop}")
        # direct check against sieve when affordable
        if qp <= 23:
            Pn2, op2 = sieve_opens(gears + [qp])
            gg = np.diff(np.concatenate([op2, [op2[0] + Pn2]]))
            bc = np.bincount(gg)
            direct = {int(v): int(bc[v]) for v in np.flatnonzero(bc)}
            err = {v: (mnew.get(v, 0), direct.get(v, 0))
                   for v in set(mnew) | set(direct) if mnew.get(v, 0) != direct.get(v, 0)}
            W(f"  direct sieve of M': {'IDENTICAL, 0 error on all ' + str(len(direct)) + ' sizes' if not err else 'MISMATCH ' + str(err)}")
        json.dump({"m": {str(k): v for k, v in mnew.items()},
                   "surv": {str(k): v for k, v in sdict.items()},
                   "merge_by_J": {str(J): as_dict(a) for J, a in mbj.items()}},
                  open(os.path.join(OUT, f"spec_rec_m{qp}.json"), "w"))
        # survival / merge split
        rows = []
        for v in sorted(mnew):
            s = sdict.get(v, 0)
            m = mnew[v] - s
            rows.append(f"{v}:{mnew[v]}={s}+{m}")
        W("  size: total = survival + merge --  " + "  ".join(rows))
        # merge mass by J
        byJ = {J: int(sum(int(x) for x in a)) for J, a in mbj.items()}
        W(f"  merge mass by J: {byJ}")
        sys.stdout.flush()
        print("\n".join(lines[-6:]))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_recursion.txt"), "w").write(txt)


if __name__ == "__main__":
    main()

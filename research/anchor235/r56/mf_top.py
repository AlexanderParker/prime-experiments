"""mf_top.py -- the merge forest at the top two rungs, 23 -> 29 and 29 -> 31.

Pass 1 streams the m29 period as 29 copies of the m23 period (openings of m23 whose residue mod 29
avoids the two teeth in that copy), giving the exact m29 spectrum, the order distribution of the
whole rung, and the full lineage of every m29 gap of size >= 22 = F/2.

Pass 2 streams the same sequence again and enumerates every run of m29 openings of span >= 29
whose interiors are struck by 31 in some copy and whose ends are not: those runs, with their
phases, ARE the m31 gaps of size >= 29 = F/2 (the merge law).  The full m31 spectrum is
accumulated at the same time as a gate.

Nothing of size 29 x N_23 or 31 x N_29 is ever materialised: ancestor counts use the exact tiled
prefix sum  S_tiled[t] = (t // N) * Total + S[t % N].
"""
import os, sys, time, json
import numpy as np
from mf_core import build_levels, u_of, PRIMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
JCAP = 40


def c_local(q, v):
    u = u_of(q)
    s = {u % q, (-u) % q, (u - v) % q, ((-u) - v) % q}
    return q - len(s)


class Base:
    """the m23 level, with the prefix sums needed for ancestor counts at any layer"""

    def __init__(self, L):
        self.L = L
        lv = L[6]
        self.lv = lv
        self.N = lv.N
        self.P = lv.P
        self.pref = {}
        self.tot = {}
        for k in range(7):
            base = lv.anc[k] if k < 6 else np.ones(lv.N, dtype=np.int64)
            s = np.zeros(lv.N + 1, dtype=np.int64)
            np.cumsum(base, out=s[1:])
            self.pref[k] = s
            self.tot[k] = int(s[-1])
        s = np.zeros(lv.N + 1, dtype=np.int64)
        np.cumsum(lv.size, out=s[1:])
        self.spref = s
        self.stot = int(s[-1])

    def anc_count(self, k, t0, t1):
        N = self.N
        return (t1 // N - t0 // N) * self.tot[k] + self.pref[k][t1 % N] - self.pref[k][t0 % N]


def copy_openings(base, j, q):
    """openings of {5..q} inside copy j of the m23 period: positions and tiled index t."""
    u = u_of(q)
    r = (base.lv.O + j * base.P) % q
    m = (r != u % q) & (r != (-u) % q)
    idx = np.flatnonzero(m).astype(np.int64)
    return base.lv.O[idx] + j * base.P, j * base.N + idx


def pass1(base, thr=22):
    """m29: spectrum, orders, and every gap of size >= thr with its lineage."""
    q = 29
    N23, P23 = base.N, base.P
    P29 = P23 * q
    spec = np.zeros(64, dtype=np.int64)
    orders = np.zeros(16, dtype=np.int64)
    big = []           # (t0, order, size)
    carry_pos = carry_t = None
    first_pos = first_t = None
    for j in range(q):
        pos, t = copy_openings(base, j, q)
        if carry_pos is not None:
            pos = np.concatenate([carry_pos, pos])
            t = np.concatenate([carry_t, t])
        else:
            first_pos, first_t = pos[0], t[0]
        sz = pos[1:] - pos[:-1]
        od = t[1:] - t[:-1]
        spec += np.bincount(sz, minlength=64)
        orders += np.bincount(od, minlength=16)
        sel = np.flatnonzero(sz >= thr)
        if sel.size:
            big.append(np.stack([t[sel], od[sel], sz[sel]], axis=1))
        carry_pos, carry_t = pos[-1:], t[-1:]
    # the wrap gap: last opening of the period to the first opening of the next period
    sz = int(first_pos + P29 - carry_pos[0])
    od = int(first_t + q * N23 - carry_t[0])
    spec[sz] += 1
    orders[od] += 1
    if sz >= thr:
        big.append(np.array([[carry_t[0], od, sz]], dtype=np.int64))
    return spec, orders, np.concatenate(big) if big else np.zeros((0, 3), np.int64), P29


def pass2(base, thr=29, keep_top=45):
    """m31: full spectrum via the recursion, plus statistics on every gap of size >= thr.
    Returns spectrum arrays and accumulators."""
    q29, q31 = 29, 31
    N23, P23 = base.N, base.P
    P29 = P23 * q29
    d = (2 * u_of(q31)) % q31
    cvals = np.array([c_local(q31, v) for v in range(200)], dtype=np.int64)
    surv = np.zeros(80, dtype=np.int64)
    merge = np.zeros(80, dtype=np.int64)
    mergeJ = {}
    acc = dict(order=np.zeros(48, dtype=np.int64),          # weighted order histogram, big gaps
               piecesz=np.zeros(64, dtype=np.int64),         # weighted piece-size histogram
               maxsz=np.zeros(64, dtype=np.int64),
               minsz=np.zeros(64, dtype=np.int64),
               rest=np.zeros(64, dtype=np.int64),
               joint=np.zeros((64, 64), dtype=np.int64),
               closed=0, low=0, tot=0, maxdepth0=0, runs=0)
    profiles = set()
    top = []            # (size, t0, J, phases, piece t-range) for size >= keep_top
    buf_pos = np.empty(0, dtype=np.int64)
    buf_t = np.empty(0, dtype=np.int64)
    first_pos = first_t = None
    t0w = time.time()
    for j in range(q29 + 1):
        if j < q29:
            pos, t = copy_openings(base, j, q29)
            if first_pos is None:
                first_pos, first_t = pos.copy(), t.copy()
        else:
            pos = first_pos[:JCAP + 5] + P29
            t = first_t[:JCAP + 5] + q29 * N23
        buf_pos = np.concatenate([buf_pos, pos])
        buf_t = np.concatenate([buf_t, t])
        nstart = buf_pos.size - (JCAP + 5)
        if nstart > 0:
            process(base, buf_pos, buf_t, nstart, d, cvals, surv, merge, mergeJ,
                    acc, profiles, top, thr, keep_top)
            buf_pos = buf_pos[nstart:]
            buf_t = buf_t[nstart:]
        print(f"    copy {j}/{q29}  {time.time()-t0w:6.1f}s  runs={acc['runs']}", flush=True)
    return surv, merge, mergeJ, acc, profiles, top


def process(base, bpos, bt, nstart, d, cvals, surv, merge, mergeJ, acc, profiles, top,
            thr, keep_top):
    qp = 31
    N23 = base.N
    res = (bpos % qp).astype(np.int64)
    idx = np.arange(nstart, dtype=np.int64)
    g1 = bpos[1:nstart + 1] - bpos[:nstart]
    surv += np.bincount(g1, weights=cvals[g1], minlength=80).astype(np.int64)
    # J = 1 big gaps (survivals of m29 gaps of size >= thr)
    sel = np.flatnonzero(g1 >= thr)
    if sel.size:
        w = cvals[g1[sel]]
        record_big(base, bpos, bt, sel, np.ones(sel.size, dtype=np.int64), w, g1[sel],
                   acc, profiles, top, keep_top)
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
        nz = np.flatnonzero(w > 0)
        if nz.size:
            span = bpos[idx[nz] + J] - bpos[idx[nz]]
            bc = np.bincount(span, weights=w[nz], minlength=80).astype(np.int64)
            merge += bc
            dj = mergeJ.setdefault(J, np.zeros(80, dtype=np.int64))
            dj += bc
            sel = nz[span >= thr]
            if sel.size:
                record_big(base, bpos, bt, idx[sel], np.full(sel.size, J, dtype=np.int64),
                           w[sel], bpos[idx[sel] + J] - bpos[idx[sel]],
                           acc, profiles, top, keep_top)
        va = va & ((eJ == ca) | (eJ == cad))
        vb = vb & ((eJ == cb) | (eJ == cbd))
        alive = va | vb
        if not alive.any():
            return
        idx = idx[alive]
        ca, cb, va, vb = ca[alive], cb[alive], va[alive], vb[alive]
    raise RuntimeError("JCAP reached")


F29 = 43


def record_big(base, bpos, bt, starts, Js, ws, spans, acc, profiles, top, keep_top):
    """accumulate statistics for m31 gaps of size >= thr; starts index into the buffer."""
    acc['runs'] += starts.size
    acc['tot'] += int(ws.sum())
    acc['order'] += np.bincount(Js, weights=ws, minlength=48).astype(np.int64)
    Jmax = int(Js.max())
    mx = np.zeros(starts.size, dtype=np.int64)
    mn = np.full(starts.size, 63, dtype=np.int64)
    allbig = np.ones(starts.size, dtype=bool)
    anylow = np.zeros(starts.size, dtype=bool)
    mxdep0 = np.zeros(starts.size, dtype=bool)
    for k in range(Jmax):
        act = Js > k
        s = starts[act]
        psz = bpos[s + k + 1] - bpos[s + k]
        od = bt[s + k + 1] - bt[s + k]
        acc['piecesz'] += np.bincount(psz, weights=ws[act], minlength=64).astype(np.int64)
        cur, curd = mx[act], mxdep0[act]
        upd = psz > cur
        mx[act] = np.where(upd, psz, cur)
        mxdep0[act] = np.where(upd, od > 1, curd)
        mn[act] = np.minimum(mn[act], psz)
        a = allbig[act]; allbig[act] = a & (psz >= F29 / 3)
        b = anylow[act]; anylow[act] = b | (psz < F29 / 4)
    acc['maxsz'] += np.bincount(mx, weights=ws, minlength=64).astype(np.int64)
    acc['rest'] += np.bincount(spans - mx, weights=ws, minlength=64).astype(np.int64)
    acc['joint'] += np.bincount(mx * 64 + (spans - mx), weights=ws, minlength=4096).astype(np.int64).reshape(64, 64)
    acc['minsz'] += np.bincount(mn, weights=ws, minlength=64).astype(np.int64)
    acc['closed'] += int(ws[allbig].sum())
    acc['low'] += int(ws[anylow].sum())
    acc['maxdepth0'] += int(ws[mxdep0].sum())
    # order profiles (deduped on runs, not weighted)
    t0 = bt[starts]
    t1 = bt[starts + Js]
    prof = np.stack([base.anc_count(k, t0, t1) for k in range(7)] + [Js], axis=1)
    for row in np.unique(prof, axis=0):
        profiles.add(tuple(int(x) for x in row))
    sel = np.flatnonzero(spans >= keep_top)
    for i in sel.tolist():
        s, J = int(starts[i]), int(Js[i])
        top.append((int(spans[i]), int(bt[s]), J, int(ws[i]),
                    [int(bpos[s + k + 1] - bpos[s + k]) for k in range(J)],
                    [int(bt[s + k]) for k in range(J + 1)], int(bpos[s])))


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    L = build_levels()
    base = Base(L)
    W("=== PART B/C: the merge forest at rungs 23->29 and 29->31 ===")
    W("")
    spec29, ord29, big29, P29 = pass1(base)
    N29 = int(spec29.sum())
    W(f"m29: N = {N29} (prod(q-2) = {27*base.N})  "
      f"sum v m = {int((np.arange(64)*spec29).sum())} vs P29 = {P29}")
    W(f"  F(m29) = {int(np.flatnonzero(spec29).max())}, |Spec| = {int((spec29>0).sum())}, "
      f"absent below F: {[v for v in range(1, int(np.flatnonzero(spec29).max())) if spec29[v]==0]}")
    W(f"  gate: m(4) = {spec29[4]}, m(6) = {spec29[6]}, m(24) = {spec29[24]}, m(36) = {spec29[36]}")
    W(f"  order histogram at rung 29: {[int(x) for x in ord29[:8]]}  "
      f"sum order = {int((np.arange(16)*ord29).sum())} vs 29 N23 = {29*base.N}  "
      f"sum(order-1) = {int((np.arange(16)*ord29).sum()) - N29} vs 2 N23 = {2*base.N}")
    W(f"  max order = {int(np.flatnonzero(ord29).max())}, mean = "
      f"{(np.arange(16)*ord29).sum()/N29:.6f} vs 29/27 = {29/27:.6f}")
    W(f"  gaps of size >= 22: {big29.shape[0]} (= sum_(v>=22) m29(v) = {int(spec29[22:].sum())})")
    np.save(os.path.join(OUT, "big29.npy"), big29)
    np.save(os.path.join(OUT, "spec29.npy"), spec29)
    W("")
    W(f"[pass 1 {time.time()-t0:.1f}s]")
    print("\n".join(lines), flush=True)
    open(os.path.join(OUT, "mf_top.txt"), "w").write("\n".join(lines))

    surv, merge, mergeJ, acc, profiles, top = pass2(base)
    m31 = surv + merge
    F31 = int(np.flatnonzero(m31).max())
    N31 = int(m31.sum())
    W("")
    W(f"m31: F = {F31}, |Spec| = {int((m31>0).sum())}, N = {N31} vs prod(q-2) = {29*N29}, "
      f"sum v m = {int((np.arange(80)*m31).sum())} vs P31 = {31*P29}")
    W(f"  gate: m(4) = {m31[4]}, m(6) = {m31[6]}, m(24) = {m31[24]}, m(36) = {m31[36]}, "
      f"m(41) = {m31[41]}; absent below F "
      f"{[v for v in range(1, F31) if m31[v]==0]}")
    W(f"  merge mass by J: {{{', '.join(f'{J}: {int(a.sum())}' for J, a in sorted(mergeJ.items()))}}}")
    W(f"  order histogram at rung 31 (all gaps): 1: {int(surv.sum())}, "
      + ", ".join(f"{J}: {int(a.sum())}" for J, a in sorted(mergeJ.items())))
    so = int(surv.sum()) + sum(J * int(a.sum()) for J, a in mergeJ.items())
    W(f"  sum order = {so} vs 31 N29 = {31*N29}; sum(order-1) = {so-N31} vs 2 N29 = {2*N29}; "
      f"mean order = {so/N31:.6f} vs 31/29 = {31/29:.6f}")
    W("")
    W(f"  m31 gaps of size >= 29: {acc['tot']} (= sum_(v>=29) m31(v) = {int(m31[29:].sum())}), "
      f"from {acc['runs']} distinct runs of m29")
    W(f"  weighted order histogram of those: {[int(x) for x in acc['order'][:8]]}")
    W(f"  distinct order profiles (k_5..k_23, J_29): {len(profiles)}")
    W(f"  all pieces >= F29/3: {acc['closed']}/{acc['tot']} = {acc['closed']/acc['tot']:.4f}; "
      f"some piece < F29/4: {acc['low']/acc['tot']:.4f}; "
      f"largest piece born at rung 29 (depth 0): {acc['maxdepth0']/acc['tot']:.4f}")
    ps = acc['piecesz']
    W(f"  piece size histogram (v: count) " +
      " ".join(f"{v}:{int(ps[v])}" for v in np.flatnonzero(ps)))
    W(f"  max-piece histogram " + " ".join(f"{v}:{int(acc['maxsz'][v])}" for v in np.flatnonzero(acc['maxsz'])))
    fr = [(int(a), int(np.flatnonzero(acc["joint"][a]).max())) for a in range(64) if acc["joint"][a].any()]
    W("  frontier (largest piece -> largest rest, and their sum): " + " ".join(f"{a}->{b}({a+b})" for a, b in fr))
    W("  REST histogram (size - largest piece) " + " ".join(f"{v}:{int(acc['rest'][v])}" for v in np.flatnonzero(acc['rest'])))
    W(f"  min-piece histogram " + " ".join(f"{v}:{int(acc['minsz'][v])}" for v in np.flatnonzero(acc['minsz'])))
    W("")
    W("  the largest m31 gaps (size >= 45), one line per run:")
    top.sort(key=lambda r: -r[0])
    for r in top[:40]:
        W(f"    size {r[0]} J={r[2]} w={r[3]} pieces {r[4]} tstart={r[1]} pos_in_m29={r[6]}")
    json.dump({"top": [[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in top]},
              open(os.path.join(OUT, "m31_top.json"), "w"))
    np.save(os.path.join(OUT, "spec31.npy"), m31)
    W("")
    W(f"[total {time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "mf_top.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

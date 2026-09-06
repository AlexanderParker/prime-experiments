"""pl_glue.py -- THE ONE-GEAR GLUE: close the middle opening of an attaining 2-run by re-phasing
exactly one gear, and read off the certificate F(M) >= S - loss.

For a 2-run x_0 < x_1 < x_2 of M (gaps a = x_1 - x_0, v = x_2 - x_1, span S = a + v) hold every
gear at its observed phase except one gear h, whose teeth are moved to {u_h + s, -u_h + s}.  The
resulting window pattern occurs somewhere in the period (CRT: the moduli are coprime), so any
blocked run it exhibits is a genuine gap of M and is <= F(M).

    Glue1(run) = max over (h, s) with x_1 BLOCKED of the gap of the re-phased machine that
                 contains the column x_1
    loss       = S - Glue1        so   E(v) = S - F <= loss   whenever loss >= 0

Reports, per machine and per realised size v (using the attaining pair a = r(v)):
  * the best Glue1 over a sample of occurrences, the loss, and whether loss <= 3;
  * the closing gear h and its shift, and whether h | a or h | v (the pad disqualification);
  * tightness: Glue1 == F(M) (the glue reaches the record exactly);
  * a two-gear glue (Glue2) for the sizes where Glue1 fails to reach F, at the letter only.

Outputs results/pl_glue_<tag>.txt / .json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of                    # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
E = 96            # window margin each side (> F at every machine tested)


def letters(q):
    a = (2 * u_of(q)) % q
    return (min(a, q - a), max(a, q - a))


def masks_for(gears, us, x0, W):
    """blocked masks of every gear on the window [x0 - E, x0 - E + W), as bool arrays."""
    base = x0 - E
    out = []
    for g, u in zip(gears, us):
        m = np.zeros(W, dtype=bool)
        for t in (u % g, (-u) % g):
            start = (t - base) % g
            m[start::g] = True
        out.append(m)
    return out


def gap_at(mask, j):
    """size of the gap of `mask` containing the BLOCKED column j: distance between the nearest
    open column left of j and the nearest open column right of j.  None if the window is short."""
    if mask[j]:
        li = j - 1
        while li >= 0 and mask[li]:
            li -= 1
        ri = j + 1
        n = mask.size
        while ri < n and mask[ri]:
            ri += 1
        if li < 0 or ri >= n:
            return None
        return ri - li
    return None


def glue1(gears, us, x0, a, v, F):
    """best one-gear re-phasing.  Returns (best, h, s, detail)."""
    S = a + v
    W = S + 2 * E
    ms = masks_for(gears, us, x0, W)
    j0, j1, j2 = E, E + a, E + S
    allm = np.zeros(W, dtype=bool)
    for m in ms:
        allm |= m
    assert not allm[j0] and not allm[j1] and not allm[j2], "the run is not a run"
    best, bh, bs = 0, None, None
    base = x0 - E
    for k, (g, u) in enumerate(zip(gears, us)):
        others = np.zeros(W, dtype=bool)
        for kk, m in enumerate(ms):
            if kk != k:
                others |= m
        if others[j0] or others[j2]:
            continue                     # an end is blocked without h: not a valid target
        for s in range(g):
            t1, t2 = (u + s) % g, (-u + s) % g
            m = others.copy()
            m[(t1 - base) % g::g] = True
            m[(t2 - base) % g::g] = True
            if not m[j1] or m[j0] or m[j2]:
                continue
            gp = gap_at(m, j1)
            if gp is not None and gp > best:
                best, bh, bs = gp, g, s
    return best, bh, bs


def glue2(gears, us, x0, a, v):
    """best two-gear re-phasing (both gears free), for the sizes the one-gear glue cannot close."""
    S = a + v
    W = S + 2 * E
    ms = masks_for(gears, us, x0, W)
    j0, j1, j2 = E, E + a, E + S
    base = x0 - E
    best, bp = 0, None
    n = len(gears)
    for k1 in range(n):
        for k2 in range(k1 + 1, n):
            others = np.zeros(W, dtype=bool)
            for kk, m in enumerate(ms):
                if kk not in (k1, k2):
                    others |= m
            if others[j0] or others[j2]:
                continue
            g1, u1 = gears[k1], us[k1]
            g2, u2 = gears[k2], us[k2]
            for s1 in range(g1):
                m1 = others.copy()
                for t in ((u1 + s1) % g1, (-u1 + s1) % g1):
                    m1[(t - base) % g1::g1] = True
                if m1[j0] or m1[j2]:
                    continue
                for s2 in range(g2):
                    m = m1.copy()
                    for t in ((u2 + s2) % g2, (-u2 + s2) % g2):
                        m[(t - base) % g2::g2] = True
                    if not m[j1] or m[j0] or m[j2]:
                        continue
                    gp = gap_at(m, j1)
                    if gp is not None and gp > best:
                        best, bp = gp, (g1, s1, g2, s2)
    return best, bp


def occurrences(size, a, v, cap):
    """indices i with (size[i], size[i+1]) = (a, v) or (v, a), spread over the period."""
    n = size.size
    left = size[:-1]
    right = size[1:]
    hit = ((left == a) & (right == v)) | ((left == v) & (right == a))
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return np.array([], dtype=np.int64)
    if idx.size <= cap:
        return idx
    step = idx.size / cap
    return idx[(np.arange(cap) * step).astype(np.int64)]


def run_machine(gears, us, O, size, qn, W, cap=24, sizes=None, do_glue2=False):
    t0 = time.time()
    F = int(size.max())
    mult = np.bincount(size.astype(np.int64))
    realised = np.flatnonzero(mult).tolist()
    F2 = 0
    r = {}
    lft, rgt = size[:-1], size[1:]
    for v in realised:
        s1 = set(np.unique(rgt[lft == v]).tolist()) | set(np.unique(lft[rgt == v]).tolist())
        r[v] = max(s1) if s1 else 0
        F2 = max(F2, v + r[v])
    aL, bL = letters(qn)
    W(f"\n=== M = {gears}  q' = {qn}  F = {F}  F_2 = {F2}  a_L = {aL} ===")
    W("  v | a=r(v) | S | E=S-F | Glue1 | loss | <=3? | tight? | h | s | h|a | h|v | occ")
    rows = []
    todo = sizes if sizes is not None else realised
    for v in todo:
        if v not in r or r[v] == 0:
            continue
        a = r[v]
        S = a + v
        occ = occurrences(size, a, v, cap)
        best, bh, bs, bi = 0, None, None, None
        for i in occ:
            i = int(i)
            aa, vv = int(size[i]), int(size[i + 1])
            x0 = int(O[i])
            g, h, s = glue1(gears, us, x0, aa, vv, F)
            if g > best:
                best, bh, bs, bi = g, h, s, (i, aa, vv)
        loss = S - best if best else None
        ok = (best >= S - 3)
        tight = (best == F)
        rows.append(dict(v=v, a=a, S=S, E=S - F, glue=best, loss=loss, ok=bool(ok),
                         tight=bool(tight), h=bh, s=bs, occ=int(occ.size),
                         h_div_a=bool(bh and a % bh == 0), h_div_v=bool(bh and v % bh == 0)))
        W(f"  {v:>3} | {a:>3} | {S:>3} | {S-F:>+3} | {best:>3} | "
          f"{'-' if loss is None else loss:>3} | {'yes' if ok else 'NO ':>3} | "
          f"{'yes' if tight else 'no ':>3} | {str(bh):>3} | {str(bs):>3} | "
          f"{'Y' if bh and a % bh == 0 else '.'} | {'Y' if bh and v % bh == 0 else '.'} | "
          f"{occ.size}")
        assert best <= F, f"SOUNDNESS FAILURE: glue {best} > F {F} at v={v}"
    nok = sum(1 for x in rows if x["ok"])
    ntight = sum(1 for x in rows if x["tight"])
    ndiv = sum(1 for x in rows if x["h_div_a"] or x["h_div_v"])
    W(f"  one-gear glue reaches S-3 at {nok} of {len(rows)} sizes; reaches F exactly (tight) at "
      f"{ntight}; closing gear divides a or v at {ndiv}")
    aLrow = next((x for x in rows if x["v"] == aL), None)
    if aLrow:
        W(f"  THE LETTER a_L = {aL}: S = {aLrow['S']}, E = {aLrow['E']:+d}, Glue1 = "
          f"{aLrow['glue']}, loss = {aLrow['loss']}, closing gear h = {aLrow['h']} "
          f"(shift {aLrow['s']}), certificate: F >= {aLrow['glue']} so E(a_L) <= {aLrow['loss']}")
    if do_glue2:
        for x in rows:
            if not x["ok"]:
                occ = occurrences(size, x["a"], x["v"], 4)
                b2, bp = 0, None
                for i in occ[:4]:
                    i = int(i)
                    g2, p2 = glue2(gears, us, int(O[i]), int(size[i]), int(size[i + 1]))
                    if g2 > b2:
                        b2, bp = g2, p2
                x["glue2"], x["glue2_pair"] = int(b2), bp
                W(f"  two-gear glue at v = {x['v']}: {b2} (loss {x['S']-b2}) with {bp}")
    W(f"  [{time.time()-t0:.1f}s]")
    return dict(gears=list(gears), qn=qn, F=F, F2=F2, aL=aL, bL=bL, rows=rows,
                nok=nok, ntight=ntight, ndiv=ndiv)


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE ONE-GEAR GLUE: F(M) >= S - loss for the attaining 2-run of every size ===")
    W("Glue1 = the gap containing the closed middle opening, over all (gear, shift) re-phasings")
    W("of ONE gear; every configuration counted occurs in the period by CRT, so Glue1 <= F(M).")
    L = build_levels()
    res = {}
    for n, qn in ((2, 13), (3, 17), (4, 19), (5, 23), (6, 29)):
        lv = L[n]
        us = [u_of(g) for g in lv.gears]
        res[str(qn)] = run_machine(list(lv.gears), us, lv.O, lv.size, qn, W,
                                   cap=24, do_glue2=True)
        W(f"[cumulative {time.time()-t0:.1f}s]")
    json.dump(res, open(os.path.join(OUT, "pl_glue.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "pl_glue.txt"), "w").write(txt)
    print(f"wrote {OUT}/pl_glue.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    for k, e in res.items():
        aLrow = next((x for x in e["rows"] if x["v"] == e["aL"]), None)
        print(f"q'={k:>2} F={e['F']:>2} aL={e['aL']:>2} "
              f"glue(aL)={aLrow['glue'] if aLrow else '-'} loss={aLrow['loss'] if aLrow else '-'} "
              f"h={aLrow['h'] if aLrow else '-'} | S-3 reached at {e['nok']}/{len(e['rows'])}, "
              f"tight at {e['ntight']}")


if __name__ == "__main__":
    main()

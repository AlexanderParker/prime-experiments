"""fc_top31.py -- the FULL frontier a -> Rest(a) at rung 29 -> 31, over every gap of the m31
period (not only the big ones), with the attaining fusion word and position for every a, and the
piece census (occurrences / fused / interior) per m29 gap size.

Built on r56's mf_top machinery: the m29 period is streamed as 29 copies of the m23 period, and
every word-legal run of m29 openings whose interiors are struck by 31 in some copy and whose ends
are not is an m31 gap (merge law), enumerated with its multiplicity w in {1, 2}.

Writes results/fc_top31.txt and results/fc_top31.json.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r56"))
from mf_core import build_levels, u_of  # noqa: E402
from mf_top import Base, copy_openings, c_local  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
JCAP = 40
F29 = 43
Q29, Q31 = 29, 31


class Acc:
    def __init__(self, A):
        self.Rest = -np.ones(A + 1, dtype=np.int64)
        self.RestJ = {j: -np.ones(A + 1, dtype=np.int64) for j in (1, 2, 3, 4, 5)}
        self.wit = {}
        self.occ = np.zeros(A + 1, dtype=np.int64)
        self.fus = np.zeros(A + 1, dtype=np.int64)
        self.inte = np.zeros(A + 1, dtype=np.int64)
        self.spec = np.zeros(96, dtype=np.int64)
        self.orders = np.zeros(48, dtype=np.int64)

    def push(self, bpos, starts, Js, ws, spans, mx):
        rest = spans - mx
        loc = -np.ones(self.Rest.size, dtype=np.int64)
        np.maximum.at(loc, mx, rest)
        for jj in self.RestJ:
            sj = Js == jj
            if sj.any():
                np.maximum.at(self.RestJ[jj], mx[sj], rest[sj])
        hits = np.flatnonzero((loc >= 0) & (loc > self.Rest))
        for a in hits.tolist():
            cand = np.flatnonzero((mx == a) & (rest == loc[a]))
            i = int(cand[int(np.argmin(Js[cand]))])
            s, J = int(starts[i]), int(Js[i])
            word = [int(bpos[s + k + 1] - bpos[s + k]) for k in range(J)]
            self.Rest[a] = loc[a]
            self.wit[a] = (int(loc[a]), word, int(bpos[s]), int(ws[i]))
        self.spec += np.bincount(spans, weights=ws, minlength=96).astype(np.int64)
        self.orders += np.bincount(Js, weights=ws, minlength=48).astype(np.int64)


def process(bpos, bt, nstart, d, cvals, acc):
    res = (bpos % Q31).astype(np.int64)
    idx = np.arange(nstart, dtype=np.int64)
    g1 = bpos[1:nstart + 1] - bpos[:nstart]
    w1 = cvals[g1]
    sel = np.flatnonzero(w1 > 0)
    acc.occ += np.bincount(g1, weights=np.full(g1.size, Q31, dtype=np.int64),
                           minlength=acc.occ.size).astype(np.int64)
    if sel.size:
        acc.push(bpos, idx[sel], np.ones(sel.size, dtype=np.int64), w1[sel], g1[sel], g1[sel])
    y = res[idx + 1]
    ca = y.copy()
    cb = (y - d) % Q31
    va = np.ones(nstart, dtype=bool)
    vb = np.ones(nstart, dtype=bool)
    for J in range(2, JCAP):
        e0 = res[idx]
        eJ = res[idx + J]
        cad = (ca + d) % Q31
        cbd = (cb + d) % Q31
        wa = va & (e0 != ca) & (e0 != cad) & (eJ != ca) & (eJ != cad)
        wb = vb & (e0 != cb) & (e0 != cbd) & (eJ != cb) & (eJ != cbd)
        w = wa.astype(np.int64) + wb.astype(np.int64)
        nz = np.flatnonzero(w > 0)
        if nz.size:
            st = idx[nz]
            ww = w[nz]
            span = bpos[st + J] - bpos[st]
            mx = np.zeros(st.size, dtype=np.int64)
            for k in range(J):
                psz = bpos[st + k + 1] - bpos[st + k]
                np.maximum(mx, psz, out=mx)
                acc.fus += np.bincount(psz, weights=ww, minlength=acc.fus.size).astype(np.int64)
                if 0 < k < J - 1:
                    acc.inte += np.bincount(psz, weights=ww,
                                            minlength=acc.inte.size).astype(np.int64)
            acc.push(bpos, st, np.full(st.size, J, dtype=np.int64), ww, span, mx)
        va = va & ((eJ == ca) | (eJ == cad))
        vb = vb & ((eJ == cb) | (eJ == cbd))
        alive = va | vb
        if not alive.any():
            return
        idx = idx[alive]
        ca, cb, va, vb = ca[alive], cb[alive], va[alive], vb[alive]
    raise RuntimeError("JCAP reached")


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    L = build_levels()
    base = Base(L)
    d = (2 * u_of(Q31)) % Q31
    cvals = np.array([c_local(Q31, v) for v in range(200)], dtype=np.int64)
    acc = Acc(F29)
    P29 = base.P * Q29
    buf_pos = np.empty(0, dtype=np.int64)
    buf_t = np.empty(0, dtype=np.int64)
    first_pos = first_t = None
    for j in range(Q29 + 1):
        if j < Q29:
            pos, t = copy_openings(base, j, Q29)
            if first_pos is None:
                first_pos, first_t = pos.copy(), t.copy()
        else:
            pos = first_pos[:JCAP + 5] + P29
            t = first_t[:JCAP + 5] + Q29 * base.N
        buf_pos = np.concatenate([buf_pos, pos])
        buf_t = np.concatenate([buf_t, t])
        nstart = buf_pos.size - (JCAP + 5)
        if nstart > 0:
            process(buf_pos, buf_t, nstart, d, cvals, acc)
            buf_pos = buf_pos[nstart:]
            buf_t = buf_t[nstart:]
        print(f"   copy {j}/{Q29}  {time.time()-t0:6.1f}s", flush=True)

    F31 = int(np.flatnonzero(acc.spec).max())
    N31 = int(acc.spec.sum())
    W("=== rung 29 -> 31: the FULL frontier, every gap of the m31 period ===")
    W(f"gate: F(m31) = {F31} (58), N = {N31} (6,226,553,025), "
      f"sum v m = {int((np.arange(96)*acc.spec).sum())} (33,426,748,355), "
      f"orders 1..5 = {[int(x) for x in acc.orders[1:6]]} "
      f"(5,805,160,589 / 413,380,422 / 7,999,018 / 12,992 / 4)")
    Fold, q = F29, Q31
    aL = min(d, q - d); bL = q - aL
    B = lambda a: Fold + q - a
    realised = [a for a in range(acc.Rest.size) if acc.Rest[a] >= 0]
    s = {a: int(B(a) - acc.Rest[a]) for a in realised}
    amin = min(realised, key=lambda a: (s[a], -a))
    peaks = [a for a in realised if a + acc.Rest[a] == F31]
    W(f"F_old = {Fold}, q' = {q}, letters ({aL}, {bL}), F_old mod q' = {Fold % q}, "
      f"interior-legal? {'YES' if (Fold % q) in (0, aL, bL) else 'no'}")
    W(f"budget slack min s = {s[amin]} at a = {amin} (a/F_old = {amin/Fold:.3f}); "
      f"record attained at a = {peaks}; top slack s(F_old) = {s[Fold]} = "
      f"{q} - Rest({Fold}) = {q} - {int(acc.Rest[Fold])}")
    W("  a | a/F_old | Rest(a) | a+Rest | B(a) | s(a) | Rest_2 | Rest_3 | Rest_4+ | "
      "a_legal | J | word | max at | left | right | class | w | x")
    for a in realised:
        r, word, x, w = acc.wit[a]
        i = int(np.argmax(word)); l = sum(word[:i]); rr = sum(word[i + 1:])
        side = "left-heavy" if l > rr else ("right-heavy" if rr > l else "balanced")
        pos = "end" if (i == 0 or i == len(word) - 1) else "interior"
        r2 = int(acc.RestJ[2][a]); r3 = int(acc.RestJ[3][a])
        r4 = max(int(acc.RestJ[4][a]), int(acc.RestJ[5][a]))
        lg = 'YES' if (a % q in (0, aL, bL) and a >= aL) else 'no'
        W(f"  {a} | {a/Fold:.3f} | {int(acc.Rest[a])} | {a+int(acc.Rest[a])} | {B(a)} | {s[a]} | "
          f"{r2} | {r3} | {r4} | {lg} | "
          f"{len(word)} | {' '.join(map(str, word))} | {i} | {l} | {rr} | {side}/{pos} | {w} | {x}")
    W("")
    W("  piece census per m29 gap size: occ = m29(v) * 31, fused = inside a J>=2 m31 gap, "
      "interior = an interior piece; predicted fused/occ = 4/31 (generic), 3/31 (v = +-d), "
      "2/31 (v = 0 mod 31); interior/occ = 0, 1/31, 2/31 respectively")
    W("  v | occ | fused | fused/occ * 31 | interior | interior/occ * 31 | v mod 31 | class")
    for v in np.flatnonzero(acc.occ).tolist():
        cls = "0" if v % q == 0 else ("+-d" if v % q in (aL, bL) else "generic")
        W(f"  {v} | {int(acc.occ[v])} | {int(acc.fus[v])} | "
          f"{acc.fus[v]*q/acc.occ[v]:.4f} | {int(acc.inte[v])} | "
          f"{acc.inte[v]*q/acc.occ[v]:.4f} | {v % q} | {cls}")
    json.dump(dict(Rest={a: int(acc.Rest[a]) for a in realised},
                   R2={a: int(acc.RestJ[2][a]) for a in realised},
                   R3={a: int(acc.RestJ[3][a]) for a in realised},
                   R4={a: max(int(acc.RestJ[4][a]), int(acc.RestJ[5][a])) for a in realised},
                   s=s, wit={a: list(acc.wit[a]) for a in realised},
                   occ=[int(x) for x in acc.occ], fus=[int(x) for x in acc.fus],
                   inte=[int(x) for x in acc.inte],
                   spec=[int(x) for x in acc.spec]),
              open(os.path.join(OUT, "fc_top31.json"), "w"))
    W(f"\n[total {time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "fc_top31.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

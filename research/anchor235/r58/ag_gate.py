"""ag_gate.py -- THE AVAILABILITY GATE.

For every rung (old machine M, incoming gear q') the gate asks: for which old sizes `a` does some
occurrence of `a` have a neighbour that is a LEGAL interior piece, i.e. of size
v = 0 or +-d (mod q') and v >= a_L (file 05 T2)?  Once the gate is closed at `a`, no J >= 3 fusion
has `a` as its largest piece, so Rest(a) = Rest_2(a) and the budget at `a` is the pair statement.

Computed exactly, on full periods, for the old machines {5}, {5,7}, ..., {5..23} (built by
mf_core.build_levels) and {5..29} (streamed as 29 copies of the m23 period).

Outputs results/ag_gate.txt and results/ag_gate.json.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import build_levels, u_of  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def letters(q, v=None):
    """the two letters of gear q with tooth value v (default: the real tooth 6^{-1})."""
    if v is None:
        v = u_of(q)
    a = (2 * v) % q
    return (min(a, q - a), max(a, q - a))


def legal_lut(mx, q, aL, bL):
    """boolean table over sizes 0..mx: v is a legal interior piece for gear q'."""
    v = np.arange(mx + 1)
    return ((v % q == 0) | (v % q == aL) | (v % q == bL)) & (v >= aL)


def gate_stats(size, q, aL, bL, chunk=8_000_000):
    """All gate quantities of the old machine whose cyclic gap sizes are `size`.

    Returns a dict of arrays indexed by old size a (0..mx):
      m       multiplicity
      hasL/hasR/has/has2/hasM   the gate counts (occurrences, not (gap, side) pairs)
      maxlegal / minlegal       largest / smallest legal neighbour of an a-gap (0 if none)
      maxlegalM                 largest legal neighbour of size <= a (0 if none)
      D       the level-2 dictionary: D[a][v] = # of (a, v) ordered adjacent pairs
    """
    n = size.size
    mx = int(size.max())
    W = mx + 1
    left = np.empty_like(size)
    left[1:] = size[:-1]
    left[0] = size[-1]
    right = np.empty_like(size)
    right[:-1] = size[1:]
    right[-1] = size[0]
    LUT = legal_lut(mx, q, aL, bL)

    m = np.zeros(W, dtype=np.int64)
    hasL = np.zeros(W, dtype=np.int64)
    hasR = np.zeros(W, dtype=np.int64)
    has = np.zeros(W, dtype=np.int64)
    has2 = np.zeros(W, dtype=np.int64)
    hasM = np.zeros(W, dtype=np.int64)
    maxlegal = np.zeros(W, dtype=np.int64)
    minlegal = np.full(W, 10**9, dtype=np.int64)
    maxlegalM = np.zeros(W, dtype=np.int64)
    D = np.zeros((W, W), dtype=np.int64)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c = size[s:e].astype(np.int64)
        l = left[s:e].astype(np.int64)
        r = right[s:e].astype(np.int64)
        m += np.bincount(c, minlength=W)
        D += np.bincount(c * W + r, minlength=W * W).reshape(W, W)
        ll = LUT[l]
        rl = LUT[r]
        hasL += np.bincount(c[ll], minlength=W)
        hasR += np.bincount(c[rl], minlength=W)
        both = ll & rl
        either = ll | rl
        has2 += np.bincount(c[both], minlength=W)
        has += np.bincount(c[either], minlength=W)
        lv = np.where(ll, l, 0)
        rv = np.where(rl, r, 0)
        np.maximum.at(maxlegal, c, np.maximum(lv, rv))
        # smallest legal neighbour
        big = np.iinfo(np.int64).max
        lv2 = np.where(ll, l, big)
        rv2 = np.where(rl, r, big)
        np.minimum.at(minlegal, c, np.minimum(lv2, rv2))
        # legal neighbour that keeps `a` the largest piece
        lvm = np.where(ll & (l <= c), l, 0)
        rvm = np.where(rl & (r <= c), r, 0)
        mm = np.maximum(lvm, rvm)
        hasM += np.bincount(c[mm > 0], minlength=W)
        np.maximum.at(maxlegalM, c, mm)
    minlegal[minlegal >= 10**9] = 0
    return dict(m=m, hasL=hasL, hasR=hasR, has=has, has2=has2, hasM=hasM,
                maxlegal=maxlegal, minlegal=minlegal, maxlegalM=maxlegalM, D=D, mx=mx)


def build_m29_gaps(lv23):
    """the cyclic gap sizes of {5..29}, exactly, as int8 (214,708,725 gaps)."""
    q = 29
    u = u_of(q)
    r23 = lv23.O % q
    P23 = lv23.P
    total = (q - 2) * lv23.N
    out = np.empty(total, dtype=np.int8)
    w = 0
    prev_last = None
    first = None
    for j in range(q):
        rj = (r23 + j * P23) % q
        keep = (rj != u % q) & (rj != (-u) % q)
        pos = lv23.O[keep] + j * P23
        if first is None:
            first = int(pos[0])
        if prev_last is not None:
            pos = np.concatenate([[prev_last], pos])
        dif = np.diff(pos)
        out[w:w + dif.size] = dif.astype(np.int8)
        w += dif.size
        prev_last = int(pos[-1])
    # closing gap of the cyclic period
    out[w] = first + q * P23 - prev_last
    w += 1
    assert w == total, (w, total)
    return out


def analyse(name, q, old_size, aL, bL, Fold, W, res):
    st = gate_stats(old_size, q, aL, bL)
    m, D, mx = st["m"], st["D"], st["mx"]
    realised = np.flatnonzero(m).tolist()
    F2 = 0
    for a in realised:
        vs = np.flatnonzero(D[a])
        if vs.size:
            F2 = max(F2, a + int(vs.max()))
    legal_a = lambda a: (a % q in (0, aL, bL)) and a >= aL
    gate_open = {a: (legal_a(a) or st["hasM"][a] > 0) for a in realised}
    a_gate = max([a for a in realised if gate_open[a]], default=0)
    a_has = max([a for a in realised if st["has"][a] > 0], default=0)
    a_hasM = max([a for a in realised if st["hasM"][a] > 0], default=0)
    a_gate2 = max([a for a in realised if st["has2"][a] > 0], default=0)
    holes = [a for a in realised if a < a_gate and not gate_open[a]]
    holes_has = [a for a in realised if a < a_has and st["has"][a] == 0]
    # which legal v occur adjacent to a_gate (either side)
    def legal_nbrs(a):
        vs = set(np.flatnonzero(D[a]).tolist()) | set(np.flatnonzero(D[:, a]).tolist())
        return sorted(v for v in vs if legal_a(v))
    W(f"\n=== rung {name}: q' = {q}, F_old = {Fold}, F_2(M) = {F2}, "
      f"letters (a_L, b_L) = ({aL}, {bL}) ===")
    W(f"  legal set L_q' below F_old: {[v for v in range(1, Fold + 1) if legal_a(v)]}")
    W(f"  a_gate = {a_gate} ({a_gate/Fold:.3f} F_old)   a_has = {a_has}   a_hasM = {a_hasM}   "
      f"a_gate2 = {a_gate2}   F_2 - a_L = {F2 - aL}   cap slack = {F2 - aL - a_gate}   "
      f"cap vacuous (>= F_old)? {'YES' if F2 - aL >= Fold else 'no'}")
    W(f"  holes below a_gate (realised, gate closed): {holes}")
    W(f"  holes below a_has (realised, has = 0):      {holes_has}")
    W(f"  legal neighbours occurring at a = a_gate ({a_gate}): {legal_nbrs(a_gate)}")
    W("  a | m(a) | a legal | hasL | hasR | has | has2 | hasM | min legal nbr | max legal nbr | "
      "max legal nbr <= a | gate")
    for a in realised:
        W(f"  {a} | {int(m[a])} | {'YES' if legal_a(a) else 'no'} | {int(st['hasL'][a])} | "
          f"{int(st['hasR'][a])} | {int(st['has'][a])} | {int(st['has2'][a])} | "
          f"{int(st['hasM'][a])} | {int(st['minlegal'][a])} | {int(st['maxlegal'][a])} | "
          f"{int(st['maxlegalM'][a])} | {'OPEN' if gate_open[a] else 'closed'}")
    res[str(q)] = dict(
        q=q, Fold=Fold, F2=F2, aL=aL, bL=bL, a_gate=a_gate, a_has=a_has, a_hasM=a_hasM,
        a_gate2=a_gate2, holes=holes, holes_has=holes_has,
        legal_nbrs_at_gate=legal_nbrs(a_gate),
        realised=realised,
        m={a: int(m[a]) for a in realised},
        has={a: int(st["has"][a]) for a in realised},
        hasL={a: int(st["hasL"][a]) for a in realised},
        hasR={a: int(st["hasR"][a]) for a in realised},
        has2={a: int(st["has2"][a]) for a in realised},
        hasM={a: int(st["hasM"][a]) for a in realised},
        minlegal={a: int(st["minlegal"][a]) for a in realised},
        maxlegal={a: int(st["maxlegal"][a]) for a in realised},
        maxlegalM={a: int(st["maxlegalM"][a]) for a in realised},
        legal_a={a: bool(legal_a(a)) for a in realised},
        gate_open={a: bool(gate_open[a]) for a in realised},
    )
    return st


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE AVAILABILITY GATE: has(a), has2(a), a_gate, at every rung 5->7 .. 29->31 ===")
    W("legal set L_q' = { v >= a_L : v = 0, +-d (mod q') } (file 05 T2).")
    W("gate OPEN at a iff a is itself legal, or some occurrence of a has a legal neighbour v <= a.")
    L = build_levels()          # {5} .. {5..23}
    res = {}
    for n in range(1, 8):
        q = PRIMES[n] if n < 7 else 29
        old = L[n - 1] if n < 7 else L[6]
        aL, bL = letters(q)
        analyse(f"{'{5..' + str(old.gears[-1]) + '}'} -> +{q}", q, old.size, aL, bL, old.F, W, res)
    W(f"\n[full-period rungs done {time.time() - t0:.1f}s]")
    # rung 29 -> 31: the old machine is m29, streamed
    g29 = build_m29_gaps(L[6])
    F29 = int(g29.max())
    W(f"\n[m29 gap array built: {g29.size:,} gaps, F(m29) = {F29}, "
      f"sum = {int(g29.astype(np.int64).sum()):,} (period 1,078,282,205), {time.time()-t0:.1f}s]")
    aL, bL = letters(31)
    analyse("{5..29} -> +31", 31, g29.astype(np.int16), aL, bL, F29, W, res)
    W(f"\n[total {time.time() - t0:.1f}s]")
    json.dump(res, open(os.path.join(OUT, "ag_gate.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "ag_gate.txt"), "w").write(txt)
    print(f"wrote {OUT}/ag_gate.txt  ({len(txt)} chars)")
    for k in res:
        r = res[k]
        print(f"q'={k:>2} F_old={r['Fold']:>2} F_2={r['F2']:>2} aL={r['aL']:>2} "
              f"a_gate={r['a_gate']:>2} ({r['a_gate']/r['Fold']:.3f}) a_has={r['a_has']:>2} "
              f"a_hasM={r['a_hasM']:>2} a_gate2={r['a_gate2']:>2} F2-aL={r['F2']-r['aL']:>2} "
              f"holes={r['holes']}")


if __name__ == "__main__":
    main()

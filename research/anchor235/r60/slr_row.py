"""slr_row.py -- THE SHORT-LETTER ROW, exact, at every rung 5->7 .. 29->31.

For each old machine M = {5}, {5,7}, ..., {5..29} on its FULL period:
  * the adjacent-pair dictionary D[a][v] (a gap of size a immediately followed by one of size v);
  * the row maxima r(v) = max { a : (a,v) in Dict_2 } for every realised size v;
  * the deficit d(v) = min(F, F_2 - v) - r(v) against the two free caps;
  * the endpoint cost c_p(v) = |T_p u (T_p - v)| for every gear p, and Leg(v);
  * the short-letter row R(a_L), the long-letter row R(b_L) and the padded row R(q'), with counts
    and with the holes (realised sizes below the row maximum that are absent from the row).

Outputs results/slr_row.txt and results/slr_row.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
sys.path.insert(0, os.path.join(HERE, "..", "r58"))
from mf_core import build_levels, u_of                    # noqa: E402
from ag_gate import build_m29_gaps                        # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def letters(q):
    a = (2 * u_of(q)) % q
    return (min(a, q - a), max(a, q - a))


def teeth(p):
    u = u_of(p)
    return {u % p, (-u) % p}


def cost(p, v):
    """number of residues mod p forbidden to the LEFT END of a gap of size v."""
    T = teeth(p)
    return len(T | {(t - v) % p for t in T})


def cost_closed(p, v):
    """the closed form: 2 if p | v, 3 if p | 3v-1 or p | 3v+1, else 4."""
    if v % p == 0:
        return 2
    if (3 * v - 1) % p == 0 or (3 * v + 1) % p == 0:
        return 3
    return 4


def chain_legal(p, v):
    """file 05 (C): both ends of a v-gap can be struck by p in one copy iff v = 0, +-d_p mod p."""
    d = (2 * u_of(p)) % p
    return v % p in (0, d % p, (-d) % p)


def dict_of(size, chunk=20_000_000):
    """D[a][v] over the cyclic gap array `size` (any integer dtype)."""
    n = size.size
    mx = int(size.max())
    W = mx + 1
    D = np.zeros((W, W), dtype=np.int64)
    m = np.zeros(W, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c = size[s:e].astype(np.int64)
        if e < n:
            r = size[s + 1:e + 1].astype(np.int64)
        else:
            r = np.empty(e - s, dtype=np.int64)
            r[:-1] = size[s + 1:e]
            r[-1] = size[0]
        m += np.bincount(c, minlength=W)
        D += np.bincount(c * W + r, minlength=W * W).reshape(W, W)
        del c, r
    return D, m, mx


def analyse(name, gears, size, qn, W, res):
    t0 = time.time()
    D, m, mx = dict_of(size)
    sym = bool(np.array_equal(D, D.T))
    realised = np.flatnonzero(m).tolist()
    F = mx
    F2 = max(a + v for a in realised for v in np.flatnonzero(D[a]).tolist())
    aL, bL = letters(qn)
    # row maxima
    row = {}          # v -> sorted list of a in R(v)
    r = {}
    for v in realised:
        s = set(np.flatnonzero(D[:, v]).tolist()) | set(np.flatnonzero(D[v]).tolist())
        row[v] = sorted(s)
        r[v] = max(s) if s else 0
    d = {v: min(F, F2 - v) - r[v] for v in realised}
    c5 = {v: cost_closed(5, v) for v in realised}
    c7 = {v: cost_closed(7, v) for v in realised}
    leg = {v: [p for p in gears if cost_closed(p, v) <= 3] for v in realised}

    def rowinfo(v):
        if v not in row:
            return None
        holes = [a for a in realised if a < r[v] and a not in set(row[v])]
        cnt = {int(a): int(D[a][v] + D[v][a]) for a in row[v]}
        return dict(v=v, r=r[v], d=d[v], holes=holes, counts=cnt,
                    c5=c5[v], leg=leg[v], mult=int(m[v]))

    W(f"\n=== M = {{5..{gears[-1]}}}  q' = {qn}  F = {F}  F_2 = {F2}  "
      f"letters (a_L, b_L) = ({aL}, {bL})  [{time.time()-t0:.1f}s] ===")
    W(f"  dictionary symmetric (D = D^T): {sym}   realised sizes: {len(realised)} of 1..{F}")
    W(f"  Leg(a_L={aL}) = {{p : p | {aL}({3*aL-1})({3*aL+1})}} ; 3a_L = {3*aL} = q' "
      f"{'-' if 3*aL == qn - 1 else '+'} 1 ; the two neighbours of 3a_L are "
      f"{3*aL-1} and {3*aL+1}")
    for tag, v in (("short letter a_L", aL), ("long letter b_L", bL), ("padded q'", qn)):
        ri = rowinfo(v)
        if ri is None:
            W(f"  row {tag} = {v}: NOT REALISED as a gap size of M")
            continue
        W(f"  row {tag} = {v}: r = {ri['r']} ({ri['r']/F:.3f} F, {ri['r']/F2:.3f} F_2), "
          f"cap min(F, F_2-v) = {min(F, F2-v)}, deficit d = {ri['d']}, c_5 = {ri['c5']}, "
          f"Leg cap M = {ri['leg']}, mult(v) = {ri['mult']}, holes = {ri['holes']}")
        W(f"      R({v}) with counts: " + " ".join(f"{a}:{ri['counts'][a]}" for a in ri['counts']))
    W("  profile  v : r(v) [cap] {d} c5 c7 mult :")
    W("    " + "  ".join(f"{v}:{r[v]}[{min(F,F2-v)}]{{{d[v]}}}c{c5[v]}/{c7[v]}"
                         for v in realised))
    # stratification of the deficit by gear-5 endpoint cost
    strat = {}
    for cc in (2, 3, 4):
        vs = [v for v in realised if c5[v] == cc]
        if vs:
            ds = sorted(d[v] for v in vs)
            rs = [r[v] / F for v in vs]
            strat[cc] = dict(n=len(vs), med=float(np.median(ds)), mean=float(np.mean(ds)),
                             mn=ds[0], mx=ds[-1], medr=float(np.median(rs)), sizes=vs)
            W(f"  c_5 = {cc}: {len(vs):3d} sizes, deficit d min/median/max = "
              f"{ds[0]}/{np.median(ds):.1f}/{ds[-1]}, median r/F = {np.median(rs):.3f}")
    nzero = sum(1 for v in realised if d[v] == 0)
    W(f"  d(v) = 0 (row at its free cap) at {nzero} of {len(realised)} realised sizes")
    # monotonicity of r above 7
    up = [(v, r[v]) for i, v in enumerate(realised[:-1])
          if v >= 7 and r[realised[i + 1]] > r[v]]
    W(f"  r(v) increases somewhere above v=7: {'YES at ' + str(up) if up else 'no'}")
    res[str(qn)] = dict(
        gears=list(gears), qn=qn, F=F, F2=F2, aL=aL, bL=bL, sym=sym, realised=realised,
        r={int(v): int(r[v]) for v in realised}, d={int(v): int(d[v]) for v in realised},
        c5={int(v): int(c5[v]) for v in realised}, c7={int(v): int(c7[v]) for v in realised},
        mult={int(v): int(m[v]) for v in realised},
        leg={int(v): leg[v] for v in realised},
        rows={t: rowinfo(v) for t, v in (("aL", aL), ("bL", bL), ("pad", qn))},
        strat=strat, nzero=nzero, monotone_break=up,
    )
    del D
    return res[str(qn)]


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE SHORT-LETTER ROW: r(v) = max { a : (a,v) is an adjacent gap pair of M } ===")
    W("free caps: r(v) <= F(M) and r(v) <= F_2(M) - v ; deficit d(v) = min(F, F_2-v) - r(v).")
    W("c_p(v) = |T_p u (T_p - v)| = 2 if p|v, 3 if p|3v-1 or p|3v+1, else 4.")
    # closed form of the endpoint cost, and the chain-law identification, checked exhaustively
    bad = [(p, v) for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
           for v in range(1, 121)
           if cost(p, v) != cost_closed(p, v) or (cost_closed(p, v) <= 3) != chain_legal(p, v)]
    W(f"\n[instrument] c_p(v) closed form and the chain-law identification "
      f"Leg(v) = {{p : c_p(v) <= 3}}: {len(bad)} mismatches over 15 gears x 120 sizes")
    assert not bad, bad[:5]

    L = build_levels()          # {5} .. {5..23}
    res = {}
    for n in range(0, 7):
        qn = PRIMES[n + 1]
        analyse("", L[n].gears, L[n].size, qn, W, res)
    W(f"\n[full-period rungs done {time.time()-t0:.1f}s]")
    g29 = build_m29_gaps(L[6])
    W(f"[m29 gap array: {g29.size:,} gaps, F = {int(g29.max())}, "
      f"sum = {int(g29.astype(np.int64).sum()):,}  {time.time()-t0:.1f}s]")
    analyse("", [5, 7, 11, 13, 17, 19, 23, 29], g29, 31, W, res)
    W(f"\n[total {time.time()-t0:.1f}s]")

    # cross-rung table
    W("\n=== the short-letter row across the rungs ===")
    W("rung q' | F | F_2 | a_L | c_5(a_L) | Leg(a_L) cap M | r(a_L) | r/F | r/F_2 | "
      "F_2-a_L | d(a_L) | r(b_L) | b_L | c_5(b_L)")
    for k in res:
        e = res[k]
        ra = e["rows"]["aL"]
        rb = e["rows"]["bL"]
        W(f"{k} | {e['F']} | {e['F2']} | {e['aL']} | {ra['c5'] if ra else '-'} | "
          f"{ra['leg'] if ra else '-'} | {ra['r'] if ra else 0} | "
          f"{(ra['r']/e['F']) if ra else 0:.3f} | {(ra['r']/e['F2']) if ra else 0:.3f} | "
          f"{e['F2']-e['aL']} | {ra['d'] if ra else '-'} | {rb['r'] if rb else 0} | "
          f"{e['bL']} | {rb['c5'] if rb else '-'}")

    json.dump(res, open(os.path.join(OUT, "slr_row.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(OUT, "slr_row.txt"), "w").write(txt)
    print(f"wrote {OUT}/slr_row.txt ({len(txt)} chars)")
    for k in res:
        e = res[k]
        ra = e["rows"]["aL"]
        print(f"q'={k:>2} F={e['F']:>2} F2={e['F2']:>2} aL={e['aL']:>2} "
              f"c5={ra['c5'] if ra else '-'} r(aL)={ra['r'] if ra else 0:>2} "
              f"r/F={(ra['r']/e['F']) if ra else 0:.3f} d={ra['d'] if ra else '-'} "
              f"Leg={ra['leg'] if ra else []} holes={ra['holes'] if ra else []}")


if __name__ == "__main__":
    main()

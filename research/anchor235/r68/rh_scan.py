"""rh_scan.py -- positions of the record stretches of M + q' with the openings of M inside them,
by the copy law (docs/proofs/05 (A)): the period of B + {extra gears} is prod(extra) copies of B's
period, and column X = x + j P_B is struck by gear g iff X = +-u_g (mod g).

    uv run python research/anchor235/r68/rh_scan.py small        rungs 13->17, 17->19, 19->23, 23->29
                                                                 (base = M itself, q' copies)
    uv run python research/anchor235/r68/rh_scan.py 29           29 -> 31 (base m23; 899 copies)
    uv run python research/anchor235/r68/rh_scan.py 31           31 -> 37: the 4-window (11,12,37,28)
                                                                 located in m31 (899 copies), lifted to m37
    uv run python research/anchor235/r68/rh_scan.py 37 [nproc]   37 -> 41: the 4-window (15,41,14,21)
                                                                 located in m37 (33,263 copies), lifted to m41

Output: results/positions_<rung>.json with, per record occurrence, the left open column c of the
stretch, the fusion, x_0 = c + f_L and x_0 mod every gear of M and mod q'.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import CORPUS_F, OUT, RECORD_FUSIONS, gears_of, next_gear, sieve_machine, u_of, strikes

_G = {}
TAIL = 4000


def _init(base_top):
    key = ("B", base_top)
    if key not in _G:
        P, O = sieve_machine(gears_of(base_top))
        _G[key] = (P, O)
    return _G[key]


def copy_survivors(base_top, j, new_gears, ncopies):
    P, O = _init(base_top)
    jj = j % ncopies
    keep = np.ones(O.size, dtype=bool)
    for g in new_gears:
        k = ("R", base_top, g)
        if k not in _G:
            _G[k] = (O % g).astype(np.int16)
        R = _G[k]
        u = u_of(g)
        t1, t2 = u % g, (-u) % g
        sh = (jj * P) % g
        r = R + np.int16(sh)
        r[r >= g] -= g
        keep &= (r != t1) & (r != t2)
    return O[keep] + np.int64(j) * np.int64(P)


def find_gaps(base_top, new_gears, ncopies, j0, j1, target_gap=None, patterns=()):
    """Scan copies [j0, j1) of the machine B + new_gears.  Returns the left openings X of every
    gap == target_gap, and of every run of consecutive gaps equal to one of `patterns`."""
    P, _ = _init(base_top)
    hits_gap, hits_pat = [], []
    prev = copy_survivors(base_top, j0 - 1, new_gears, ncopies)[-TAIL:]
    for j in range(j0, j1):
        S = np.concatenate([prev, copy_survivors(base_top, j, new_gears, ncopies)])
        gaps = np.diff(S)
        lo = np.int64(j) * np.int64(P) - TAIL // 2
        hi = np.int64(j + 1) * np.int64(P) - TAIL // 2
        starts = S[:-1]
        valid = (starts >= lo) & (starts < hi)
        if target_gap is not None:
            idx = np.flatnonzero(valid & (gaps == target_gap))
            hits_gap.extend(int(S[i]) for i in idx)
        for pat in patterns:
            k = len(pat)
            if gaps.size < k:
                continue
            m = valid[: gaps.size - k + 1].copy()
            for t, v in enumerate(pat):
                m &= gaps[t: gaps.size - k + 1 + t] == v
            idx = np.flatnonzero(m)
            hits_pat.extend((int(S[i]), list(pat)) for i in idx)
        prev = S[-TAIL:]
    return hits_gap, hits_pat


def _worker(args):
    return find_gaps(*args)


def phase_vector(x0, gears, q):
    return {str(g): int(x0 % g) for g in gears + [q]}


def fusion_at(c, G, O_old_sorted_set_fn):
    """The openings of the old machine strictly inside (c, c + G) -> the fusion gaps."""
    inside = O_old_sorted_set_fn(c, c + G)
    pts = [c] + inside + [c + G]
    return [int(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]


def small_rungs():
    out = {}
    for y in (13, 17, 19, 23):
        q = next_gear(y)
        G = CORPUS_F[q]
        P, O = _init(y)
        gears = gears_of(y)
        t0 = time.time()
        hits, _ = find_gaps(y, [q], q, 0, q, target_gap=G)
        recs = []
        for c in hits:
            # openings of M inside (c, c+G): O is M's period; reduce mod P
            lo, hi = c % P, c % P + G
            ins = O[(O > lo) & (O < hi)].tolist() + [o + P for o in O[(O + P > lo) & (O + P < hi)].tolist()]
            ins = sorted(set(ins))
            pts = [lo] + ins + [hi]
            fus = [int(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
            x0 = c + fus[0]
            recs.append({"c": int(c), "fusion": fus, "x0": int(x0), "phase": phase_vector(x0, gears, q)})
        out[f"{y}->{q}"] = {"F": G, "count_per_period": len(recs), "records": recs,
                            "secs": round(time.time() - t0, 1)}
        print(f"{y}->{q}: F={G}, {len(recs)} record stretches per period; fusions "
              f"{sorted(set(tuple(r['fusion']) for r in recs))} [{time.time()-t0:.1f}s]", flush=True)
    with open(os.path.join(OUT, "positions_small.json"), "w") as f:
        json.dump(out, f, indent=1)


def rung_29():
    """29 -> 31: m31 = 899 copies of m23 filtered by 29, 31; record gaps 58; m29 openings inside."""
    y, q = 29, 31
    G = CORPUS_F[q]
    gears = gears_of(y)
    t0 = time.time()
    hits, _ = find_gaps(23, [29, 31], 899, 0, 899, target_gap=G)
    P23, O23 = _init(23)
    recs = []
    for c in hits:
        # m29 openings inside (c, c+G): m23 openings not struck by 29
        j = c // P23
        cand = np.concatenate([O23 + (j - 1) * P23, O23 + j * P23, O23 + (j + 1) * P23])
        cand = cand[(cand > c) & (cand < c + G)]
        cand = cand[~strikes(29, cand)]
        pts = [c] + cand.tolist() + [c + G]
        fus = [int(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        x0 = c + fus[0]
        recs.append({"c": int(c), "fusion": fus, "x0": int(x0), "phase": phase_vector(x0, gears, q)})
    out = {"29->31": {"F": G, "count_per_period": len(recs), "records": recs, "secs": round(time.time() - t0, 1)}}
    with open(os.path.join(OUT, "positions_29.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"29->31: F={G}, {len(recs)} record stretches per period; fusions "
          f"{sorted(set(tuple(r['fusion']) for r in recs))} [{time.time()-t0:.1f}s]", flush=True)


def lift(y, q, hits_pat, ncopies_old, P_old, secs, tag, nproc_note=""):
    """Positions X of a fusion window in M (period P_old, copies of m23) -> the column of the
    record stretch in M + q': X + k P_old with the interior openings on the teeth of q'."""
    gears = gears_of(y)
    u = u_of(q)
    teeth = {u % q, (-u) % q}
    recs = []
    for X, pat in hits_pat:
        offs = [0]
        for v in pat:
            offs.append(offs[-1] + v)
        for k in range(q):
            c = X + k * P_old
            inner = [(c + o) % q in teeth for o in offs[1:-1]]
            ends = [(c + o) % q in teeth for o in (offs[0], offs[-1])]
            if all(inner) and not any(ends):
                x0 = c + pat[0]
                recs.append({"c": int(c), "X_in_M": int(X), "k": k, "fusion": list(pat), "x0": int(x0),
                             "phase": phase_vector(x0, gears, q)})
    out = {f"{y}->{q}": {"F": CORPUS_F[q], "count_per_period": len(recs), "records": recs,
                         "window_hits_in_M": len(hits_pat), "secs": round(secs, 1), "note": nproc_note}}
    with open(os.path.join(OUT, f"positions_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{y}->{q}: {len(hits_pat)} windows in M, {len(recs)} record stretches per period of M+q' "
          f"[{secs:.1f}s]", flush=True)
    for r in recs[:6]:
        print("   ", r["fusion"], "c =", r["c"], "phase", r["phase"])


def rung_31():
    y, q = 31, 37
    pats = [(11, 12, 37, 28), (28, 37, 12, 11)]
    t0 = time.time()
    _, hits = find_gaps(23, [29, 31], 899, 0, 899, patterns=pats)
    P23, _ = _init(23)
    lift(y, q, hits, 899, 899 * P23, time.time() - t0, "31")


def rung_37(nproc):
    y, q = 37, 41
    pats = [(15, 41, 14, 21), (21, 14, 41, 15)]
    ncop = 29 * 31 * 37
    t0 = time.time()
    nchunks = nproc * 6
    edges = np.linspace(0, ncop, nchunks + 1).astype(int)
    tasks = [(23, [29, 31, 37], ncop, int(edges[i]), int(edges[i + 1]), None, pats)
             for i in range(nchunks) if edges[i + 1] > edges[i]]
    hits = []
    with Pool(nproc) as pool:
        for _, hp in pool.imap_unordered(_worker, tasks):
            hits.extend(hp)
            print(f"  chunk done, windows so far {len(hits)} [{time.time()-t0:.0f}s]", flush=True)
    P23, _ = _init(23)
    lift(y, q, hits, ncop, ncop * P23, time.time() - t0, "37", f"{nproc} processes")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "small":
        small_rungs()
    elif mode == "29":
        rung_29()
    elif mode == "31":
        rung_31()
    elif mode == "37":
        rung_37(int(sys.argv[2]) if len(sys.argv) > 2 else 3)

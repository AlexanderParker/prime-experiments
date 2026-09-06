"""mo_order.py -- candidate (e), the branch's own: THE DE BRUIJN RELAXATION OF THE CLOSURE, and
the order of interaction the budget needs.

The record law says F(M + q') = max_{J <= J_max} Q*_J(M; q'), where a J-fusion is a run of J
consecutive gaps of M whose J - 1 interior openings are struck in one phase and whose two ends are
not.  The constraint "this run is REALISED in M" is a statement about the whole window; relax it
to a statement of bounded order:

    a J-word is LEVEL-k ADMISSIBLE  iff  every k consecutive entries of it lie in D_k(M)
                                        (and, for J <= k, iff the word itself lies in D_J(M)).

    B_k(M; q') := max { span of a level-k admissible J-word that fuses at some phase, J <= J_max }.

B_1 >= B_2 >= ... >= B_{J_max} = F(M + q') exactly, and every B_k is a functional of D_k(M) alone
- a bounded-order functional of the merge closure that BOUNDS the record.  The question this
script answers exactly is: at which order k does B_k(M; q') first fall to or below the budget
F(M) + q'?  That is the order of interaction the budget needs, asked of the merge closure.

The fusion condition is tested by brute force over the q' phases (no word theory), so the answer
is exact.  State of the dynamic programme: (the last k-1 gaps, the residue (o + z) mod q').

Usage: uv run python research/anchor235/r66/mo_order.py [kmax] [K0]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import OUT, PRIMES, base_gaps, closure_step, dict_from_gaps, next_gear, u_of

NEG = -(1 << 40)

# J_max = L + 2 of the RUNG y -> (next gear): branching_identity.md 3.1 / ladder_closure.md 3.1,
# values 3, 2, 3, 3, 3, 4, 3, 5, 5, 4 at the rungs 5->7 .. 37->41.
JMAX_RUNG = {5: 3, 7: 2, 11: 3, 13: 3, 17: 3, 19: 4, 23: 3, 29: 5, 31: 5, 37: 4}


def dicts_from_windows(win, kmax):
    """D_1(M) .. D_kmax(M) as sorted uint8 arrays of distinct k-tuples, from a window dictionary
    (rows terminated by a zero are short and contribute only their complete prefixes)."""
    out = []
    for k in range(1, kmax + 1):
        if k > win.shape[1]:
            out.append(None)
            continue
        good = np.all(win[:, :k] != 0, axis=1)
        w = np.ascontiguousarray(win[good, :k])
        v = np.unique(w.view([("", np.uint8)] * k))
        out.append(v.view(np.uint8).reshape(-1, k))
    return out


def windows_from_period(gaps, K):
    n = gaps.size
    gp = np.concatenate([gaps, gaps[:K]])
    w = np.empty((n, K), dtype=np.uint8)
    for i in range(K):
        w[:, i] = gp[i:i + n]
    return w


def _index_of(rows, keys):
    """Row indices of `keys` inside sorted distinct `rows` (both uint8, same width)."""
    k = rows.shape[1]
    rv = rows.view([("", np.uint8)] * k).ravel()
    kv = np.ascontiguousarray(keys).view([("", np.uint8)] * k).ravel()
    return np.searchsorted(rv, kv)


def exact_J(Dj, q, J):
    """max span of a realised J-word that fuses at some phase (exact, no relaxation)."""
    if Dj is None or Dj.shape[0] == 0:
        return None, None
    d = (2 * u_of(q)) % q
    n = Dj.shape[0]
    off = np.zeros((J + 1, n), dtype=np.int64)
    for i in range(J):
        off[i + 1] = off[i] + Dj[:, i]
    best, arg = None, None
    for z in range(q):
        if z % q in (0, d):
            continue
        ok = np.ones(n, dtype=bool)
        for i in range(1, J):
            ok &= np.isin((off[i] + z) % q, [0, d])
        ok &= ~np.isin((off[J] + z) % q, [0, d])
        if ok.any():
            sp = off[J][ok]
            i = int(sp.argmax())
            if best is None or int(sp[i]) > best:
                best = int(sp[i])
                arg = [int(x) for x in Dj[np.flatnonzero(ok)[i]]]
    return best, arg


def relaxed_J(Dk, Dkm1, q, J, k):
    """max span of a level-k admissible J-word that fuses at some phase, J > k >= 1."""
    d = (2 * u_of(q)) % q
    struck = np.zeros(q, dtype=bool)
    struck[0] = True
    struck[d] = True
    if k == 1:
        sizes = Dk[:, 0].astype(np.int64)
        nS = 1
        frm = np.zeros(sizes.size, dtype=np.int64)
        to = np.zeros(sizes.size, dtype=np.int64)
        gap = sizes
    else:
        nS = Dkm1.shape[0]
        frm = _index_of(Dkm1, Dk[:, :k - 1])
        to = _index_of(Dkm1, Dk[:, 1:])
        gap = Dk[:, k - 1].astype(np.int64)
    V = np.full((nS, q), NEG, dtype=np.int64)
    # seed: the first k-1 gaps, all of whose openings o_1..o_{k-1} must be struck
    if k == 1:
        for z in range(q):
            if not struck[z]:
                V[0, z] = 0
    else:
        n = Dkm1.shape[0]
        cum = np.zeros((k, n), dtype=np.int64)
        for i in range(k - 1):
            cum[i + 1] = cum[i] + Dkm1[:, i]
        for z in range(q):
            if struck[z]:
                continue
            ok = np.ones(n, dtype=bool)
            for i in range(1, k):
                ok &= struck[(cum[i] + z) % q]
            if not ok.any():
                continue
            r = (cum[k - 1][ok] + z) % q
            idx = np.flatnonzero(ok)
            np.maximum.at(V, (idx, r), cum[k - 1][ok])
    placed = k - 1
    # relax while the newly made opening must be struck: gaps g_placed .. g_{J-2}
    for _ in range(placed, J - 1):
        W = np.full((nS, q), NEG, dtype=np.int64)
        for r in range(q):
            cur = V[frm, r]
            live = cur > NEG
            if not live.any():
                continue
            r2 = (r + gap) % q
            good = live & struck[r2]
            if good.any():
                np.maximum.at(W, (to[good], r2[good]), cur[good] + gap[good])
        V = W
        if (V > NEG).sum() == 0:
            return None
    # last gap: the far endpoint must be UNstruck
    best = None
    for r in range(q):
        cur = V[frm, r]
        live = cur > NEG
        if not live.any():
            continue
        r2 = (r + gap) % q
        good = live & ~struck[r2]
        if good.any():
            v = int((cur[good] + gap[good]).max())
            best = v if best is None else max(best, v)
    return best


def order_table(Ds, q, Jmax, kmax):
    """B_k for k = 1..kmax, and the per-J detail."""
    rows = {}
    for k in range(1, kmax + 1):
        if Ds[k - 1] is None:
            continue
        per = {}
        for J in range(1, Jmax + 1):
            if J <= k:
                if Ds[J - 1] is None:
                    per[J] = None
                    continue
                v, _ = exact_J(Ds[J - 1], q, J)
                per[J] = v
            else:
                per[J] = relaxed_J(Ds[k - 1], Ds[k - 2] if k > 1 else None, q, J, k)
        vals = [v for v in per.values() if v is not None]
        rows[k] = {"B": max(vals) if vals else None,
                   "per_J": {str(J): per[J] for J in per},
                   "exact_up_to_J": k}
    return rows


def run_machine(name, win, q, Jmax, kmax, F_old, out):
    Ds = dicts_from_windows(win, kmax)
    sizes = [None if D is None else int(D.shape[0]) for D in Ds]
    t0 = time.time()
    tab = order_table(Ds, q, Jmax, kmax)
    budget = F_old + q
    least = None
    for k in sorted(tab):
        if tab[k]["B"] is not None and tab[k]["B"] <= budget and least is None:
            least = k
    out[name] = {"machine": name, "gear": q, "J_max": Jmax, "F_old": F_old, "budget": budget,
                 "dict_sizes": sizes, "B": {str(k): tab[k]["B"] for k in tab},
                 "per_J": {str(k): tab[k]["per_J"] for k in tab},
                 "least_order_within_budget": least, "secs": round(time.time() - t0, 1)}
    print(f"{name} -> {q}: |D_k| = {sizes} | B_k = "
          f"{ {k: tab[k]['B'] for k in tab} } | budget {budget} | least order "
          f"{least} [{time.time()-t0:.1f}s]", flush=True)
    return out


def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    K0 = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    JMAX = JMAX_RUNG
    out = {}
    for y in [5, 7, 11, 13, 17, 19, 23]:
        P, g = base_gaps(y)
        K = min(g.size, max(kmax, JMAX[next_gear(y)] if next_gear(y) in JMAX else kmax) + 1)
        Kw = min(g.size, kmax + 2)
        win = windows_from_period(g, Kw)
        run_machine(f"m{y}", win, next_gear(y), JMAX[y], kmax, int(g.max()), out)
        with open(os.path.join(OUT, "order.json"), "w") as f:
            json.dump(out, f, indent=1)
    # the top of the ladder: m29, m31 (and m37 if the depth survives)
    P, g = base_gaps(23)
    win, mult = dict_from_gaps(g, K0)
    print(f"base m23: |D_{K0}| = {win.shape[0]:,}", flush=True)
    Fprev = int(g.max())
    for q in (29, 31, 37):
        t0 = time.time()
        _, _, s1 = closure_step(win, mult, q, m=win.shape[1], mode="fixed", tag_order=True,
                                collect=False)
        m = max(1, s1["mmin"])
        win, mult, st = closure_step(win, mult, q, m=m, mode="fixed")
        spec = s1["specJ"].sum(axis=0)
        F = int(np.flatnonzero(spec).max())
        print(f"m{q}: depth {m}, F = {F}, |D| = {win.shape[0]:,} [{time.time()-t0:.1f}s]",
              flush=True)
        nq = next_gear(q)
        if q in JMAX:
            run_machine(f"m{q}", win, nq, JMAX[q], min(kmax, m), F, out)
        Fprev = F
        with open(os.path.join(OUT, "order.json"), "w") as f:
            json.dump(out, f, indent=1)
        if m <= 1:
            break


if __name__ == "__main__":
    main()

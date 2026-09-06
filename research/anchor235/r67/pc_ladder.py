"""pc_ladder.py -- the same items on the top of the ladder, m29, m31, m37, off the r61/r66
closure operator (D_15^#(m23) -> m29 at depth 10 -> m31 at depth 6 -> m37 at depth 2), where no
period can be built:

  * spectrum (exact, with multiplicities) -> legal alphabet, PAD alphabet, holes;
  * every realised legal k-word for k <= depth, with its multiplicity (the k-windows of the
    dictionary are complete: loss = 0), hence L, L_bare, L_small, L_pad, L_pad0, L_skip within
    the depth, and k_L within the depth;
  * P3 at m37: the realised legal 2-words, in particular the size-feasible skip 2-words
    (27,55), (55,27), (14,68), (68,14);
  * the (order, value) mass of the rung's padded letters (shadow test (a)) and the record
    witnesses (shadow test (b));
  * E1 checked within depth.

Usage: uv run python research/anchor235/r67/pc_ladder.py [K0]
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_core import (CORPUS_F, CORPUS_L, CORPUS_LPAD, OUT, base_gaps, closure_step, corrcap3,
                     dict_from_gaps, k_L_table, letter_data, letters_of, next_gear, skeleton,
                     word_kind)
from lc_core import witnesses  # noqa: E402
from pc_words import alphabet  # noqa: E402


def legal_kwords(win, mult, q, kmax):
    """Counter of realised legal k-words (k <= kmax) with multiplicities, off the dictionary."""
    n, K = win.shape
    kmax = min(kmax, K)
    lt = letters_of(win.astype(np.int64), q)
    out = {}
    ok = np.ones(n, dtype=bool)
    prev = np.zeros(n, dtype=np.uint8)
    for k in range(kmax):
        c = lt[:, k]
        ok &= (win[:, k] != 0) & (c != 3) & ~((c != 0) & (c == prev))
        prev = np.where(c == 0, prev, c)
        idx = np.flatnonzero(ok)
        cnt = Counter()
        for i in idx:
            cnt[tuple(int(v) for v in win[i, :k + 1])] += int(mult[i])
        out[k + 1] = cnt
        if not idx.size:
            break
    return out


def analyse(name, win, mult, q, F, spec, specJ_in, log):
    t0 = time.time()
    depth = win.shape[1]
    alph = alphabet(spec, q, F)
    kw = legal_kwords(win, mult, q, depth)
    L = max((k for k, c in kw.items() if c), default=0)
    by_kind = {"bare": 0, "small": 0, "pad": 0, "pad0": 0, "skip": 0}
    words = {}
    for k, c in kw.items():
        for w, m in c.items():
            kind = word_kind(w, q)
            by_kind[kind] = max(by_kind[kind], k)
            if kind != "bare":
                by_kind["pad"] = max(by_kind["pad"], k)
            if any(v % q == 0 for v in w):
                by_kind["pad0"] = max(by_kind["pad0"], k)
            s = skeleton(w, q)
            words[" ".join(map(str, w))] = {"kind": kind, "mult": m, "S0": s["S0"], "S1": s["S1"],
                                            "runs0": s["runs0"], "runs1": s["runs1"],
                                            "skips": s["skips0"] + s["skips1"], "jumps": s["jumps"]}
    # k_L within depth: the maximal-word Counter is replaced by the top-level k-words
    top = Counter({w: m for w, m in kw.get(L, {}).items()}) if L else Counter()
    # subwords of the realised L-words give all realised k-words for k <= L (every realised
    # legal k-word with k < L is a prefix of ... not necessarily; use the full sets instead)
    from pc_core import longest_admissible
    krows, kL = {}, None
    for k in range(1, min(L + 1, depth) + 1):
        rk = set(kw.get(k, {}).keys())
        val, cyc = longest_admissible(rk, q, k)
        if val is not None and val < L:
            val = L
        krows[k] = "inf" if cyc else val
        if kL is None and not cyc and val == L:
            kL = k
    cc3, _ = corrcap3(q % 210)
    qq = next_gear(q)
    row = {"machine": name, "q": q, "F": F, "depth": depth, "rows": int(win.shape[0]),
           "N": int(mult.sum()), "alphabet": alph, "L_within_depth": L,
           "L_bare": by_kind["bare"], "L_small": by_kind["small"], "L_pad": by_kind["pad"],
           "L_pad0": by_kind["pad0"], "L_skip": by_kind["skip"],
           "corpus_L": CORPUS_L.get(name), "corpus_Lpad": CORPUS_LPAD.get(name),
           "words": words, "k_L": kL, "L_k": krows, "corrcap3": cc3,
           "E1_ok": max(by_kind["bare"], by_kind["small"]) <= cc3,
           "W_k": {k: int(sum(c.values())) for k, c in kw.items()},
           "secs": round(time.time() - t0, 1)}
    if specJ_in is not None:
        # padded letters of THIS machine by the order of the fusion that made them (test a)
        row["pad_orders"] = {v: {int(J): int(specJ_in[J][v]) for J in range(1, specJ_in.shape[0])
                                 if specJ_in[J][v] > 0} for v in alph["pad"]}
    log[name] = row
    print(f"m{name} q'={q} F={F} depth={depth} rows={win.shape[0]:,} | legal={sorted(alph['realised'])} "
          f"holes={alph['holes']} pad={alph['pad']} | L<= depth: L={L} bare={by_kind['bare']} "
          f"small={by_kind['small']} pad={by_kind['pad']} pad0={by_kind['pad0']} skip={by_kind['skip']} "
          f"(corpus L={CORPUS_L.get(name)} Lpad={CORPUS_LPAD.get(name)}) | k_L={kL} L^(k)={krows} | "
          f"CORRCAP_3={cc3} E1={'OK' if row['E1_ok'] else 'FAIL'} | W_k={row['W_k']} "
          f"| pad_orders={row.get('pad_orders')} [{time.time()-t0:.1f}s]", flush=True)
    for w in sorted(words, key=lambda w: (-len(w.split()), w)):
        if len(w.split()) >= 2 or words[w]["kind"] != "bare":
            print(f"    ({w}) x{words[w]['mult']} {words[w]['kind']} S0={words[w]['S0']} S1={words[w]['S1']}")
    return row


def main():
    K0 = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    log = {}
    t0 = time.time()
    P, g = base_gaps(23)
    win, mult = dict_from_gaps(g, K0)
    print(f"base m23: |D_{K0}| = {win.shape[0]:,} [{time.time()-t0:.1f}s]", flush=True)
    Fprev = int(g.max())
    specJ_prev = None
    for q in (29, 31, 37):
        t1 = time.time()
        K = win.shape[1]
        _, _, s1 = closure_step(win, mult, q, m=K, mode="fixed", tag_order=True, collect=False)
        m = max(1, s1["mmin"])
        w2, m2, st = closure_step(win, mult, q, m=m, mode="fixed")
        specJ = s1["specJ"]
        spec = specJ.sum(axis=0).astype(np.int64)
        F = int(np.flatnonzero(spec).max())
        print(f"m{q}: depth {m}, F = {F} (corpus {CORPUS_F[q]}), |D| = {w2.shape[0]:,}, loss={st['loss']} "
              f"over0={s1['over0']} [{time.time()-t1:.1f}s]", flush=True)
        # record witnesses of this rung (test b): middles of the record-attaining fusions
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in range(1, specJ.shape[0])
              if specJ[j].sum()}
        wit = witnesses(win, mult, q, {J: v for J, v in Qs.items() if v == F}, maxout=6)
        rec = [{"J": J, "gaps": h["gaps"], "middles": h["gaps"][1:-1],
                "kind": word_kind(h["gaps"][1:-1], q) if len(h["gaps"]) > 2 else "none", "z": h["z"]}
               for J, lst in wit.items() for h in lst]
        print(f"   record {F}: {[(r['J'], r['gaps'], r['kind']) for r in rec]}", flush=True)
        win, mult = w2, m2
        nq = next_gear(q)
        row = analyse(q, win, mult, nq, F, spec, specJ, log)
        row["record_witnesses"] = rec
        row["Qstar"] = Qs
        row["budget_slack"] = Fprev + q - F
        Fprev = F
        with open(os.path.join(OUT, f"ladder_K{K0}.json"), "w") as f:
            json.dump(log, f, indent=1, default=int)
        if m <= 1:
            break
    print(f"total [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()

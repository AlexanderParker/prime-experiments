"""pc_words.py -- items 1-3 on the full periods m5..m23 (direct sieve in the anchored column
coordinate), with respect to the next gear q':

  * the spectrum, the legal alphabet, the PAD alphabet (padded letters realised, with
    multiplicities), the size-feasible legal values that are spectral holes;
  * every maximal realised legal word with its occurrence count; L, L_bare, L_pad (brief:
    at least one letter >= q'), L_pad0 (at least one PAD-class letter), L_small, L_skip;
  * the skeleton (S_0, S_1), runs, skips, jumps of every distinct realised legal word (P6);
  * the order k_L at which L is decided (P7);
  * for the longest padded words: position, gears at the junctions, needed gears per letter,
    shared needed gears, the counting cap numbers (P5);
  * the record of M + q' and whether its middles contain a padded letter (shadow test (b));
  * the order (fusion depth) distribution of the padded letters of M + q' (shadow test (a)),
    read off the closure step's (order, value) mass table;
  * E1 checked: L_small <= CORRCAP_3(q' mod 210).

Usage: uv run python research/anchor235/r67/pc_words.py
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_core import (CORPUS_F, CORPUS_L, CORPUS_LPAD, OUT, closure_step, corrcap3, dict_from_gaps,
                     gears_of, junction_analysis, k_L_table, letter_data, letters_of,
                     maximal_legal_words, next_gear, sieve_machine, skeleton, word_kind)
from lc_core import witnesses  # noqa: E402

BASES = [5, 7, 11, 13, 17, 19, 23]
LOOK = 40


def alphabet(spec, q, F):
    L = letter_data(q)
    d = L["d"]
    legal = [v for v in range(1, F + 1) if v % q in (0, d, (-d) % q)]
    realised = {v: int(spec[v]) for v in legal if v < spec.size and spec[v] > 0}
    holes = [v for v in legal if v not in realised]
    pad = {v: m for v, m in realised.items() if v >= q}
    return {"a": L["a"], "b": L["b"], "d": d, "legal_values": legal, "realised": realised,
            "holes": holes, "pad": pad, "n_legal": len(realised), "n_pad": len(pad),
            "bare_mult": {L["a"]: int(spec[L["a"]]) if L["a"] < spec.size else 0,
                          L["b"]: int(spec[L["b"]]) if L["b"] < spec.size else 0},
            "skip_letters": [v for v in pad if v != q]}


def main():
    log = {}
    for y in BASES:
        t0 = time.time()
        q = next_gear(y)
        gears = gears_of(y)
        P, O = sieve_machine(gears)
        N = O.size
        g = np.empty(N, dtype=np.int64)
        g[:-1] = O[1:] - O[:-1]
        g[-1] = O[0] + P - O[-1]
        F = int(g.max())
        spec = np.bincount(g)
        alph = alphabet(spec, q, F)
        gext = np.concatenate([g, g[:LOOK]])
        words = maximal_legal_words(gext, q, N)
        cnt = Counter(w for _, w in words)
        L = max((len(w) for w in cnt), default=0)
        by_kind = {"bare": 0, "small": 0, "pad": 0, "pad0": 0, "skip": 0}
        kinds = {}
        for w in cnt:
            k = word_kind(w, q)
            kinds[w] = k
            by_kind[k] = max(by_kind[k], len(w))
            if k != "bare":
                by_kind["pad"] = max(by_kind["pad"], len(w))
            if any(v % q == 0 for v in w):
                by_kind["pad0"] = max(by_kind["pad0"], len(w))
        # skeletons of every distinct word
        skel = {}
        max_run, max_jump_small = 0, 0
        for w in cnt:
            s = skeleton(w, q)
            skel[" ".join(map(str, w))] = {"kind": kinds[w], "count": cnt[w], "S0": s["S0"],
                                          "S1": s["S1"], "runs0": s["runs0"], "runs1": s["runs1"],
                                          "skips": s["skips0"] + s["skips1"], "jumps": s["jumps"]}
            max_run = max(max_run, s["max_run"])
            if kinds[w] in ("bare", "small"):
                max_jump_small = max(max_jump_small, max(s["jumps"] or [0]))
        krows, kL = k_L_table(cnt, q, L)
        # longest padded words with positions, junction analysis
        pad_words = [(i, w) for i, w in words if kinds[w] != "bare"]
        Lp = by_kind["pad"]
        longest_pad = [(i, w) for i, w in pad_words if len(w) == Lp]
        junc = []
        seen = set()
        for i, w in longest_pad:
            if w in seen and len(junc) >= 6:
                continue
            seen.add(w)
            if len(junc) < 12:
                junc.append(junction_analysis(int(O[i]), w, gears))
        # the longest word overall, if bare, for the counting cap too
        longest_all = [(i, w) for i, w in words if len(w) == L]
        junc_all = [junction_analysis(int(O[i]), w, gears) for i, w in longest_all[:4]]
        # the record of M + q' and its middles (shadow test b); the pad-letter orders (test a)
        Kb = min(N, 8)
        win, mult = dict_from_gaps(g.astype(np.uint8), Kb)
        _, _, st = closure_step(win, mult, q, m=Kb, mode="fixed", tag_order=True, collect=False)
        specJ = st["specJ"]
        newspec = specJ.sum(axis=0)
        Fnew = int(np.flatnonzero(newspec).max())
        Qs = {int(j): int(np.flatnonzero(specJ[j]).max()) for j in range(1, specJ.shape[0])
              if specJ[j].sum()}
        wit = witnesses(win, mult, q, {J: v for J, v in Qs.items() if v == Fnew}, maxout=6)
        rec = []
        for J, lst in wit.items():
            for h in lst:
                mids = h["gaps"][1:-1]
                rec.append({"J": J, "gaps": h["gaps"], "middles": mids,
                            "kind": word_kind(mids, q) if mids else "none", "z": h["z"]})
        qq = next_gear(q)
        alph_next = alphabet(newspec.astype(np.int64), qq, Fnew)
        pad_orders = {}
        for v in alph_next["pad"]:
            pad_orders[v] = {int(J): int(specJ[J][v]) for J in range(1, specJ.shape[0])
                             if specJ[J][v] > 0}
        cc3, _ = corrcap3(q % 210)
        row = {"machine": y, "q": q, "P": P, "N": N, "F": F, "corpus_F": CORPUS_F[y],
               "alphabet": alph, "L": L, "L_bare": by_kind["bare"], "L_small": by_kind["small"],
               "L_pad": by_kind["pad"], "L_pad0": by_kind["pad0"], "L_skip": by_kind["skip"],
               "corpus_L": CORPUS_L.get(y), "corpus_Lpad": CORPUS_LPAD.get(y),
               "n_distinct_words": len(cnt), "n_word_occurrences": len(words),
               "words": skel, "k_L": kL, "L_k": krows, "max_run": max_run,
               "max_jump_small": max_jump_small, "corrcap3": cc3,
               "E1_ok": max(by_kind["bare"], by_kind["small"]) <= cc3,
               "longest_pad_words": Counter(w for _, w in longest_pad),
               "n_longest_pad_positions": len(longest_pad),
               "junctions_pad": junc, "junctions_longest": junc_all,
               "F_next": Fnew, "Qstar": Qs, "record_witnesses": rec,
               "pad_alphabet_next": alph_next["pad"], "pad_orders_next": pad_orders,
               "secs": round(time.time() - t0, 1)}
        row["longest_pad_words"] = {" ".join(map(str, w)): c for w, c in row["longest_pad_words"].items()}
        log[y] = row
        print(f"m{y} q'={q} F={F} a,b={alph['a']},{alph['b']} legal={sorted(alph['realised'])} "
              f"holes={alph['holes']} pad={alph['pad']} | L={L} bare={by_kind['bare']} "
              f"small={by_kind['small']} pad={by_kind['pad']} pad0={by_kind['pad0']} "
              f"skip={by_kind['skip']} (corpus L={CORPUS_L.get(y)} Lpad={CORPUS_LPAD.get(y)}) | "
              f"k_L={kL} L^(k)={krows} | max_run={max_run} max_jump_small={max_jump_small} | "
              f"CORRCAP_3={cc3} E1={'OK' if row['E1_ok'] else 'FAIL'} | words={len(cnt)} "
              f"| F_next={Fnew} rec={[(r['J'], r['gaps'], r['kind']) for r in rec[:3]]} "
              f"| pad_orders_next={pad_orders} [{time.time()-t0:.1f}s]", flush=True)
        for w in sorted(cnt, key=lambda w: (-len(w), w)):
            print(f"    {w} x{cnt[w]} {kinds[w]} S0={skel[' '.join(map(str, w))]['S0']} "
                  f"S1={skel[' '.join(map(str, w))]['S1']}")
        for jn in junc[:2]:
            print(f"    pad word {jn['word']} at X0={jn['X0']}: needed={[gp['needed'] for gp in jn['gaps']]} "
                  f"shared={jn['shared_needed']} counting={jn['counting']}")
            print(f"      openings: {[(o['X'] % q, o['left'], o['right'], o['mod35']) for o in jn['openings']]}")
        with open(os.path.join(OUT, "words_full.json"), "w") as f:
            json.dump(log, f, indent=1, default=int)


if __name__ == "__main__":
    main()

"""ol2_step.py -- one rung M -> q' of the order law, entirely on the covering instrument.

For the machine M (whose D_1, D_2 come from ol2_dict.py, i.e. from the instrument, not a ladder):

  1. L(M) with respect to q'  -- the longest REALISED legal letter word, found by enumerating
     legal words over the letters of q' that are realised gap values of M, filtering adjacent
     pairs through D_2(M), and putting each candidate to the instrument.  J_max = L + 2.
  2. B_k(M; q') for k = L and k = L + 1 -- the widest span of a level-k admissible J-word,
     J <= J_max, that fuses at some phase of q'.  A J-word is level-k admissible iff every k
     consecutive entries lie in D_k(M); for J <= k that is the word itself.  Membership in D_k(M)
     is one covering problem, so no dictionary deeper than 2 is ever built.
  3. The exact row Q*_J (J = 1 .. J_max) and with it F(M + q') = max_J Q*_J, recomputed from the
     instrument alone -- the gate against the certified record.

Usage: uv run python research/anchor235/r71/ol2_step.py <y> [procs]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import (F2_CAP_ONLY, F2_RECORD, F_RECORD, OUT, d_of, decide_many,  # noqa
                      enumerate_fusing, fj_cap, gears_upto, letter_class, load_memo, memo,
                      next_gear, save_memo, u_of)

NAMES = {0: "PAD", 1: "UP", 2: "DOWN", 3: "BAD"}
CAPS = {}
MEMOPATH = ""


def seed_memo_super(cap1, cap2):
    """Neither D_1 nor D_2 tabulated.  The ONLY facts written into the memo are the ones the
    certified record table gives for free: no 2-window of span above F_2 <= cap2 is realised.
    Nothing is asserted about any window the instrument has not decided."""
    m = memo()
    for a in range(1, cap1 + 1):
        for b in range(1, cap1 + 1):
            if a + b > cap2:
                m[(a, b)] = False
    return len(m)


def seed_memo_lazy(D1, cap1, cap2):
    """D_2 was not tabulated for this engine.  Only the two facts the record table certifies are
    written into the memo: which 1-windows are realised (ol2_dict.py decided every one), and that
    no 2-window of span above F_2 <= cap2 is realised.  Every other pair is left for the
    descending scans to decide on demand, which is strictly fewer covering problems: only the
    pairs that occur inside a word wide enough to matter are ever put to the instrument."""
    m = memo()
    s1 = set(D1)
    for v in range(1, cap1 + 1):
        m[(v,)] = v in s1
    for a in range(1, cap1 + 1):
        for b in range(1, cap1 + 1):
            if a + b > cap2 or a not in s1 or b not in s1:
                m[(a, b)] = False
    return len(m)


def seed_memo(D1, D2, cap1, cap2):
    """ol2_dict.py has already decided EVERY 1-window of span <= F(M) and every 2-window of span
    <= F_2 cap; write those verdicts (both signs) into the memo so no covering problem is solved
    twice.  Windows outside the certified caps are False by the record table itself."""
    m = memo()
    s1 = set(D1)
    for v in range(1, cap1 + 1):
        m[(v,)] = v in s1
    for a in range(1, cap1 + 1):
        for b in range(1, cap1 + 1):
            if a + b <= cap2:
                m[(a, b)] = (a, b) in D2
            else:
                m[(a, b)] = False          # F_2(M) <= cap2, certified deletion-ladder cap
    return len(m)


def subwindows(word, k):
    if len(word) <= k:
        return [tuple(word)]
    return [tuple(word[i:i + k]) for i in range(len(word) - k + 1)]


def descending_scan(words, k, gears, procs, label, floor=None, log=True, cap_k=None,
                    soft_budget=400_000, resolve_hard=True):
    """`words` is {word: phase}.  Return the widest word all of whose k-subwindows are realised,
    scanning strictly in descending span so the first hit is the maximum.  Batches one span level
    at a time so the instrument runs in parallel."""
    by_span = {}
    for w, z in words.items():
        s = sum(w)
        if floor is not None and s < floor:
            continue
        by_span.setdefault(s, []).append((w, z))
    tested = 0
    calls0 = {"calls": 0, "nodes": 0, "fallback": 0, "undecided": 0, "wall": 0.0, "hard": 0}
    hardlist = []
    deferred = 0

    def over(t):
        c = (cap_k or {}).get(len(t))
        return c is not None and sum(t) > c

    def run(need, **kw):
        v, st = decide_many(need, gears, procs=procs, log=False, **kw)
        for key in ("calls", "nodes", "fallback", "undecided"):
            calls0[key] += st[key]
        calls0["wall"] += st["wall"]
        return v

    for s in sorted(by_span, reverse=True):
        batch = by_span[s]
        allsub = {t for w, _ in batch for t in subwindows(w, k)}
        for t in allsub:                       # certified deletion-ladder cap, no solver call
            if over(t):
                memo()[t] = False
        verd = {t: False for t in allsub if over(t)}
        # PASS 1, soft: cheap verdicts only.  A word with one cheap NO is rejected here and its
        # expensive subwindows are never proved.
        verd.update(run(sorted(t for t in allsub if not over(t)), soft=True, budget=soft_budget))
        pend = [(w, z) for w, z in batch
                if not any(verd.get(t) is False for t in subwindows(w, k))]
        # PASS 2, full: only the subwindows of words still standing
        need2 = sorted({t for w, _ in pend for t in subwindows(w, k) if verd.get(t) is None})
        if need2 and not resolve_hard:
            # the caller asked for a certified LOWER bound only: a word with a deferred subwindow
            # is skipped, never assumed, and the count is reported with the answer
            deferred += len({w for w, _ in pend if any(verd.get(t) is None
                                                       for t in subwindows(w, k))})
            need2 = []
        if need2:
            calls0["hard"] += len(need2)
            hardlist += [list(t) for t in need2][:50]
            if log:
                print(f"  [{label}] span {s}: {len(need2)} deferred window(s) resolved with the "
                      f"full solver", flush=True)
            verd.update(run(need2))
        tested += len(batch)
        hits = [(w, z) for w, z in batch
                if all(verd.get(t) is True for t in subwindows(w, k))]
        if hits:
            if log:
                print(f"  [{label}] span {s}: {len(hits)} of {len(batch)} words admissible "
                      f"-- STOP.  {tested:,} words scanned", flush=True)
            return {"span": s, "word": list(hits[0][0]), "phase": hits[0][1],
                    "ties": [list(w) for w, _ in hits][:20], "words_scanned": tested,
                    "hard_windows": hardlist, "deferred_above": deferred,
                    "value_exact": deferred == 0, "stats": calls0}
        if log and tested % 500 < len(batch):
            print(f"  [{label}] ... {tested:,} words scanned, at span {s}, "
                  f"{calls0['calls']:,} solver calls, {calls0['wall']:.0f}s", flush=True)
    return {"span": None, "word": None, "phase": None, "ties": [], "words_scanned": tested,
            "hard_windows": hardlist, "deferred_above": deferred, "value_exact": deferred == 0,
            "stats": calls0}


def legal_letter_words(D1, D2, q, maxlen=8):
    """Every legal letter word over the realised gap values of M, by length.
    Struck classes: start at c in {0, d}; PAD keeps c, UP needs c = 0 -> d, DOWN needs c = d -> 0.
    Legality is exactly `no two consecutive equal nonzero letters, pads transparent`."""
    d = d_of(q)
    lets = [v for v in D1 if letter_class(v, q) != 3]
    out = {}
    # state: (last gap, current struck class c)
    cur = []
    for v in lets:
        lc = letter_class(v, q)
        for c0 in (0, d):
            c1 = c0 if lc == 0 else (d if (lc == 1 and c0 == 0) else (0 if (lc == 2 and c0 == d)
                                                                     else None))
            if c1 is None:
                continue
            cur.append(((v,), c1))
    out[1] = sorted({w for w, _ in cur})
    for n in range(2, maxlen + 1):
        nxt = []
        for w, c in cur:
            for v in lets:
                if (w[-1], v) not in D2:
                    continue
                lc = letter_class(v, q)
                c1 = c if lc == 0 else (d if (lc == 1 and c == 0) else (0 if (lc == 2 and c == d)
                                                                       else None))
                if c1 is None:
                    continue
                nxt.append((w + (v,), c1))
        if not nxt:
            break
        cur = nxt
        out[n] = sorted({w for w, _ in cur})
    return out


def term_bound(J, cap, why):
    """A term that provably cannot be the maximum is recorded at its certified upper bound
    instead of being computed.  Used only when cap < F(M + q') and cap <= budget, so the term is
    below both the record and the budget and can bind neither B_k nor max_J Q*_J."""
    return {"span": cap, "word": None, "phase": None, "ties": [], "words_scanned": 0,
            "bound_only": True, "why": why, "value_exact": True, "hard_windows": [],
            "deferred_above": 0,
            "stats": {"calls": 0, "nodes": 0, "fallback": 0, "undecided": 0, "wall": 0.0}}


def term_J1(FM, q, dd_):
    """Q*_1 = the widest realised gap of M that fuses (both flanks unstruck).  F(M) IS a realised
    gap (the certified record) and no gap is wider, so as soon as F(M) fuses at some phase the
    term is F(M) exactly -- with no covering problem, which matters because a wide 1-window is the
    most expensive pattern the instrument can be asked about."""
    for z in range(q):
        if z % q in (0, dd_):
            continue
        if (z + FM) % q not in (0, dd_):
            return {"span": FM, "word": [FM], "phase": z, "ties": [[FM]], "words_scanned": 0,
                    "undecided_above": [], "certified": True,
                    "stats": {"calls": 0, "nodes": 0, "fallback": 0, "undecided": 0, "wall": 0.0}}
    return None


def main():
    y = int(sys.argv[1])
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gears = gears_upto(y)
    q = next_gear(y)
    d = d_of(q)
    FM = F_RECORD[y]
    Fnext = F_RECORD[q]
    budget = FM + q
    t00 = time.time()

    path = os.path.join(OUT, f"dict_m{y}.json")
    dd = json.load(open(path)) if os.path.exists(path) else {}
    cap2 = F2_RECORD.get(y, F2_CAP_ONLY.get(y, F_RECORD[q]))
    # D_1 as a table where ol2_dict.py produced one; otherwise the superset 1 .. F(M).  Either way
    # every k-window (k >= 2) below is decided by the instrument, so every B_k is exact: a superset
    # here only adds candidate words, which the exact test then rejects.
    D1 = [int(v) for v in dd["D1"]] if dd.get("D1") else list(range(1, FM + 1))
    D1_tabulated = bool(dd.get("D1"))
    lazy = not dd.get("D2")
    print(f"=== rung m{y} -> {q} ===", flush=True)
    print(f"gears {gears}; u = {u_of(q)}, d = {d}, letter floor a_L = {min(d, q - d)}", flush=True)
    print(f"F(m{y}) = {FM}, budget F + q' = {budget}; certified F(m{q}) = {Fnext}", flush=True)
    if lazy:
        D2 = set((a, b) for a in D1 for b in D1 if a + b <= cap2)
        print(f"|D_1| = {len(D1)} ({'tabulated' if D1_tabulated else 'superset 1..F'}, "
              f"max {max(D1)}); D_2 NOT tabulated -- the level-2 filter is the superset of "
              f"{len(D2):,} pairs of span <= {cap2}, and every k-window is decided exactly by the "
              f"instrument, so every B_k below is exact", flush=True)
        nseed = (seed_memo_lazy(D1, FM, cap2) if D1_tabulated
                 else seed_memo_super(FM, cap2))
    else:
        D2 = set(tuple(int(x) for x in w) for w in dd["D2"])
        print(f"|D_1| = {len(D1)} (max {max(D1)}), |D_2| = {len(D2):,}, F_2 = {dd['F_2']}",
              flush=True)
        nseed = seed_memo(D1, D2, FM, cap2)
    print(f"memo seeded with {nseed:,} decided 1- and 2-windows", flush=True)
    global CAPS, MEMOPATH
    MEMOPATH = os.path.join(OUT, f"memo_m{y}.json")
    n0 = load_memo(MEMOPATH)
    print(f"memo file: {n0:,} verdicts carried over from earlier runs", flush=True)
    CAPS = {j: fj_cap(y, j) for j in range(1, 8)}
    print(f"certified deletion-ladder span caps F_j(m{y}) <= {CAPS}", flush=True)
    lets = {v: NAMES[letter_class(v, q)] for v in D1 if letter_class(v, q) != 3}
    print(f"letters of {q} among the realised gaps: {lets}", flush=True)

    rep = {"y": y, "q": q, "gears": gears, "u": u_of(q), "d": d, "a_L": min(d, q - d),
           "F_M": FM, "budget": budget, "F_next_certified": Fnext,
           "D1_size": len(D1), "D2_size": len(D2), "F_2": dd.get("F_2"), "lazy_D2": lazy,
           "letters": lets}

    # ---------------------------------------------------------------- 1. L(M)
    cands = legal_letter_words(D1, D2, q)
    print(f"legal letter words by length: "
          f"{ {n: len(v) for n, v in cands.items()} }", flush=True)
    L = 0
    Lwit = None
    Ldetail = {}
    for n in sorted(cands):
        if n == 1 and not D1_tabulated:
            # only EXISTENCE is needed at length 1 (a longer realised word settles L anyway):
            # take the letters in increasing span and stop at the first realised one.  The wide
            # 1-windows are the most expensive covering problems there are (only two open offsets
            # constrain the phases) and none of them is needed.
            good = []
            for w in sorted(cands[1], key=sum):
                vv, _ = decide_many([w], gears, procs=1)
                if vv[w]:
                    good = [list(w)]
                    break
            st = {"wall": 0.0}
        else:
            # only EXISTENCE matters at each length, so try for a cheap witness first (narrowest
            # words, small node budget); the full solver is called only when no cheap witness
            # exists, which is exactly the length where the answer is "none", i.e. where L is
            # decided and an exhaustive refutation is genuinely required.
            order = sorted(cands[n], key=sum)
            v, st = decide_many(order, gears, procs=procs, soft=True, budget=400_000)
            good = [list(w) for w in order if v[w] is True]
            if not good:
                pend = [w for w in order if v[w] is None]
                if pend:
                    print(f"  length {n}: no cheap witness; {len(pend)} deferred word(s) go to "
                          f"the full solver", flush=True)
                    v2, st2 = decide_many(pend, gears, procs=procs)
                    st = {"wall": st["wall"] + st2["wall"]}
                    good = [list(w) for w in pend if v2[w] is True]
        Ldetail[n] = {"candidates": len(cands[n]), "realised": len(good),
                      "example": good[0] if good else None}
        print(f"  length {n}: {len(cands[n]):,} legal candidates, {len(good):,} realised "
              f"({st['wall']:.0f}s)", flush=True)
        if good:
            L = n
            Lwit = good[0]
        else:
            break
    Jmax = L + 2
    rep["L"] = L
    rep["L_witness"] = Lwit
    rep["L_detail"] = Ldetail
    rep["J_max"] = Jmax
    print(f"L(m{y}) = {L}  (witness {Lwit}),  J_max = L + 2 = {Jmax}", flush=True)
    if len(sys.argv) > 3 and sys.argv[3] == "L":
        with open(os.path.join(OUT, f"L_m{y}.json"), "w") as f:
            json.dump({"y": y, "q": q, "L": L, "witness": Lwit, "J_max": Jmax,
                       "detail": Ldetail, "secs": round(time.time() - t00, 1)}, f, indent=1)
        save_memo(MEMOPATH)
        return

    # ---------------------------------------------------------------- words per J
    words = {}
    for J in range(1, Jmax + 1):
        t0 = time.time()
        words[J] = enumerate_fusing(J, D1, D2, q)
        print(f"J = {J}: {len(words[J]):,} level-2 admissible fusing words, spans "
              f"{max(map(sum, words[J])) if words[J] else '-'} down to "
              f"{min(map(sum, words[J])) if words[J] else '-'}  ({time.time()-t0:.0f}s)",
              flush=True)

    # ---------------------------------------------------------------- 2. B_k for k = L, L+1
    for k in (L, L + 1):
        per = {}
        for J in range(1, Jmax + 1):
            if not words[J]:
                per[J] = {"span": None}
                continue
            if J <= k:
                # For J <= k the term is the EXACT Q*_J, and Q*_J <= max_J Q*_J = F(M + q') by
                # the record law, so the whole block of exact terms is subsumed by the single
                # certified number F(M + q').  And B_k >= F(M + q') always (the true extremal
                # fusion is realised, hence level-k admissible at every k).  Therefore
                #     B_k = max( F(M + q') , the relaxed terms J > k )
                # exactly, with no exact-term computation at all.  That is the whole of the cost
                # saving of this branch: the expensive wide-thin patterns live in the J <= k
                # terms, and they never bind.
                per[J] = {"span": None, "subsumed": True, "exact": True,
                          "note": f"exact Q*_{J} <= F(m{q}) = {Fnext} (record law), subsumed by "
                                  f"the certified record term"}
                print(f"  k = {k}, J = {J}: exact term, subsumed by F(m{q}) = {Fnext}", flush=True)
                continue
            # k = L + 1 is the decisive scan and needs every refutation (it is an UPPER bound on
            # B_{L+1}).  k = L only has to beat the budget, and a witness does that, so its
            # expensive refutations are skipped and its value is reported as a lower bound.
            r = descending_scan(words[J], k, gears, procs, f"k={k},J={J}", cap_k=CAPS,
                                resolve_hard=(k == L + 1))
            r["exact"] = False
            per[J] = r
            print(f"  k = {k}, J = {J}: relaxed = {r['span']}"
                  f"{'' if r.get('value_exact', True) else ' (LOWER BOUND, '
                     + str(r['deferred_above']) + ' words above it deferred)'} "
                  f"word {r['word']} at phase {r['phase']}  "
                  f"[memo {save_memo(MEMOPATH):,}]", flush=True)
        vals = [p["span"] for p in per.values() if p.get("span") is not None]
        B = max(vals + [Fnext])
        arg = [J for J in per if per[J].get("span") == B] + (["record"] if B == Fnext else [])
        exact_all = all(p.get("value_exact", True) for p in per.values()
                        if not p.get("subsumed"))
        rep[f"B_{k}_exact"] = exact_all
        rep[f"B_{k}_upper"] = max(max(map(sum, words[J])) for J in words if words[J])
        rep[f"B_{k}"] = B
        rep[f"per_J_{k}"] = {str(J): per[J] for J in per}
        rep[f"argmax_J_{k}"] = arg
        print(f"B_{k}(m{y}; {q}) {'=' if exact_all else '>='} {B}   budget {budget}: "
              f"{'WITHIN' if B <= budget else 'ABOVE'} (margin {budget - B}); argmax J = {arg}; "
              f"level-2 superset upper bound {rep[f'B_{k}_upper']}", flush=True)
        with open(os.path.join(OUT, f"step_{y}_{q}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"  memo saved: {save_memo(MEMOPATH):,} verdicts", flush=True)

    # ------------------------------------------------- 3a. the record WITNESS (always, cheap)
    # The record law gives max_J Q*_J = F(M + q') <= the certified record, so the check the
    # instrument can add is the other direction: exhibit a realised J-fusion of span exactly
    # F(M + q'), found with no refutation anywhere (YES is the cheap direction).
    wit = None
    for J in range(Jmax, 0, -1):
        cand = [w for w in words.get(J, {}) if sum(w) == Fnext]
        if not cand:
            continue
        v, _ = decide_many(sorted(cand), gears, procs=procs, soft=True, budget=2_000_000)
        hit = [w for w in cand if v[w] is True]
        if hit:
            wit = {"J": J, "word": list(hit[0]), "phase": words[J][hit[0]], "span": Fnext,
                   "candidates_at_span": len(cand), "realised_at_span": len(hit)}
            break
    rep["record_witness"] = wit
    print(f"record witness at span {Fnext} = F(m{q}): {wit}", flush=True)
    save_memo(MEMOPATH)
    with open(os.path.join(OUT, f"step_{y}_{q}.json"), "w") as f:
        json.dump(rep, f, indent=1)

    # ---------------------------------------------------------------- 3b. the exact record row
    if len(sys.argv) > 3 and sys.argv[3] == "norec":
        print("full record scan skipped (norec)", flush=True)
        rep["kstar"] = L + 1 if (rep[f"B_{L+1}"] <= budget and rep[f"B_{L}"] > budget) else None
        kk = L + 1
        bw = rep[f"per_J_{kk}"][str([J for J in rep[f'argmax_J_{kk}'] if J != "record"][0])
                                ]["word"] if [J for J in rep[f'argmax_J_{kk}']
                                              if J != "record"] else None
        if bw:
            v, _ = decide_many([tuple(bw)], gears, procs=1)
            rep["binding_word"] = bw
            rep["binding_realised"] = v[tuple(bw)]
            rep["binding_letters"] = [NAMES[letter_class(g, q)] for g in bw]
            print(f"binding word {bw} letters {rep['binding_letters']}: "
                  f"{'REALISED' if rep['binding_realised'] else 'NOT realised'}", flush=True)
        rep["secs"] = round(time.time() - t00, 1)
        save_memo(MEMOPATH)
        with open(os.path.join(OUT, f"step_{y}_{q}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\nk* = {rep['kstar']} (L + 1 = {L+1});  total {rep['secs']}s", flush=True)
        return
    exact = {}
    for J in range(1, Jmax + 1):
        if not words[J]:
            exact[J] = None
            continue
        c = CAPS.get(J)
        if J <= 2 and c is not None and c < Fnext:
            r = term_bound(J, c, f"Q*_{J} <= F_{J}(m{y}) <= {c} < F(m{q}) = {Fnext}: cannot "
                                 f"attain the record")
        else:
            r = (term_J1(FM, q, d) if J == 1 else None) or \
                descending_scan(words[J], Jmax, gears, procs, f"exact J={J}", cap_k=CAPS)
        exact[J] = r
        print(f"  Q*_{J} = {r['span']}  word {r['word']} at phase {r['phase']}  "
              f"[memo {save_memo(MEMOPATH):,}]", flush=True)
    Qs = [exact[J]["span"] if exact[J] else None for J in range(1, Jmax + 1)]
    Fcalc = max(v for v in Qs if v is not None)
    rep["Qstar"] = Qs
    rep["F_next_computed"] = Fcalc
    rep["record_gate"] = (Fcalc == Fnext)
    rep["exact_detail"] = {str(J): exact[J] for J in exact}
    print(f"Q*_J = {Qs};  F(m{q}) from the instrument = {Fcalc}, certified {Fnext}: "
          f"{'MATCH' if Fcalc == Fnext else 'DISAGREE'}", flush=True)

    # binding word realised?
    kk = L + 1
    am = [J for J in rep[f'argmax_J_{kk}'] if J != "record"]
    bw = rep[f"per_J_{kk}"][str(am[0])]["word"] if am else None
    if bw is None:
        rep["binding_word"] = None
        rep["binding_realised"] = None
        rep["kstar"] = L + 1 if (rep[f"B_{L+1}"] <= budget and rep[f"B_{L}"] > budget) else None
        rep["secs"] = round(time.time() - t00, 1)
        with open(os.path.join(OUT, f"step_{y}_{q}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\nk* = {rep['kstar']} (L + 1 = {L+1});  total {rep['secs']}s", flush=True)
        return
    v, _ = decide_many([tuple(bw)], gears, procs=1)
    rep["binding_word"] = bw
    rep["binding_realised"] = bool(v[tuple(bw)])
    rep["binding_letters"] = [NAMES[letter_class(g, q)] for g in bw]
    print(f"binding word {bw} letters {rep['binding_letters']}: "
          f"{'REALISED' if rep['binding_realised'] else 'NOT realised'}", flush=True)
    rep["kstar"] = L + 1 if (rep[f"B_{L+1}"] <= budget and rep[f"B_{L}"] > budget) else None
    rep["secs"] = round(time.time() - t00, 1)
    with open(os.path.join(OUT, f"step_{y}_{q}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nk* = {rep['kstar']} (L + 1 = {L+1});  total {rep['secs']}s", flush=True)


if __name__ == "__main__":
    main()

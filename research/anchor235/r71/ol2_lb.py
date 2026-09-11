"""ol2_lb.py -- a certified LOWER bound on B_k(M; q') above a threshold, cheaply.

The order law's second half only needs `B_L > F(M) + q'`, and a witness proves that.  The
descending scan of ol2_step.py proves much more than that -- it refutes every wider word on the
way down -- and those refutations are the expensive half of a rung.  This script does the cheap
half alone: it scans the fusing J-words in ASCENDING span starting just above the threshold and
stops at the first word all of whose k-subwindows are realised.  The answer is a lower bound with
a witness, never an exact value, which is exactly what the law's second half asks for.

Usage: uv run python research/anchor235/r71/ol2_lb.py <y> <k> <threshold> [procs] [maxspan]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import (F2_CAP_ONLY, F2_RECORD, F_RECORD, OUT, decide_many,  # noqa: E402
                      enumerate_fusing, fj_cap, gears_upto, letter_class, load_memo, memo,
                      next_gear, save_memo)

NAMES = {0: "PAD", 1: "UP", 2: "DOWN", 3: "BAD"}


def subwindows(word, k):
    if len(word) <= k:
        return [tuple(word)]
    return [tuple(word[i:i + k]) for i in range(len(word) - k + 1)]


def main():
    y = int(sys.argv[1])
    k = int(sys.argv[2])
    thr = int(sys.argv[3])
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    maxspan = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
    gears = gears_upto(y)
    q = next_gear(y)
    FM = F_RECORD[y]
    Fnext = F_RECORD[q]
    cap2 = F2_RECORD.get(y, F2_CAP_ONLY.get(y, Fnext))
    CAPS = {j: fj_cap(y, j) for j in range(1, 9)}
    print(f"=== lower bound on B_{k}(m{y}; {q}) above {thr} ===", flush=True)
    print(f"F(m{y}) = {FM}, budget = {FM + q}, certified F(m{q}) = {Fnext}, caps {CAPS}",
          flush=True)
    # the step run for the same engine may still be writing memo_m{y}.json, so read it and write
    # this run's new verdicts to a file of its own -- never overwrite another process's memo
    load_memo(os.path.join(OUT, f"memo_m{y}.json"))
    path = os.path.join(OUT, f"memo_m{y}_lb{k}.json")
    print(f"memo: {load_memo(path):,} verdicts on file", flush=True)
    dd_path = os.path.join(OUT, f"dict_m{y}.json")
    dd = json.load(open(dd_path)) if os.path.exists(dd_path) else {}
    D1 = [int(v) for v in dd["D1"]] if dd.get("D1") else list(range(1, FM + 1))
    D2 = set((a, b) for a in D1 for b in D1 if a + b <= cap2)
    Jmax = F_RECORD and json.load(open(os.path.join(OUT, f"step_{y}_{q}.json")))["J_max"] \
        if os.path.exists(os.path.join(OUT, f"step_{y}_{q}.json")) else None
    if Jmax is None:
        Jmax = k + 1
    M = memo()
    best = None
    t00 = time.time()
    for J in range(k + 1, Jmax + 1):
        words = enumerate_fusing(J, D1, D2, q)
        cand = {w: z for w, z in words.items() if thr < sum(w) <= maxspan}
        by_span = {}
        for w, z in cand.items():
            by_span.setdefault(sum(w), []).append((w, z))
        print(f"J = {J}: {len(words):,} fusing words, {len(cand):,} with span in "
              f"({thr}, {maxspan}]", flush=True)
        for s in sorted(by_span):                      # ASCENDING: the cheap end first
            batch = by_span[s]
            subs = {t for w, _ in batch for t in subwindows(w, k)}
            for t in subs:
                if sum(t) > (CAPS.get(k) or 10 ** 9):
                    M[t] = False
            need = sorted(t for t in subs if M.get(t) is None)
            if need:
                decide_many(need, gears, procs=procs, soft=True, budget=400_000)
            hits = [(w, z) for w, z in batch
                    if all(M.get(t) is True for t in subwindows(w, k))]
            if hits:
                w, z = hits[0]
                best = {"span": s, "word": list(w), "phase": z, "J": J,
                        "letters": [NAMES[letter_class(g, q)] for g in w],
                        "ties": [list(a) for a, _ in hits][:10],
                        "subwindows": {",".join(map(str, t)): True
                                       for t in subwindows(w, k)}}
                print(f"  J = {J}: WITNESS at span {s}: {list(w)} at phase {z} "
                      f"({time.time()-t00:.0f}s)", flush=True)
                break
            print(f"  J = {J}: span {s}: none of {len(batch)} admissible "
                  f"({time.time()-t00:.0f}s)", flush=True)
        save_memo(path)
        if best:
            break
    out = {"y": y, "q": q, "k": k, "threshold": thr, "budget": FM + q, "witness": best,
           "secs": round(time.time() - t00, 1)}
    with open(os.path.join(OUT, f"lb_{y}_{q}_k{k}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"B_{k}(m{y}; {q}) >= {best['span'] if best else 'none found'}   "
          f"budget {FM + q}", flush=True)


if __name__ == "__main__":
    main()

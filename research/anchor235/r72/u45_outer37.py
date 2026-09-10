"""u45_outer37.py -- the J-run outer law at m37, with D_3(m37) as the exact prune.

J = 3 is then a maximum over the rows of D_3(m37) and needs no solver at all.  J = 4 and J = 5 are
descending scans over the words whose every 3-subwindow lies in D_3 (a complete superset of the
realised J-windows, and a very tight one), each word decided by the free-phase covering instrument
r70/ol_pattern.py; the first realised word in descending outer sum is the exact maximum.

Usage: uv run python research/anchor235/r72/u45_outer37.py [jmax]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r70")))

from ol_pattern import realised_word, realised_word_search  # noqa: E402

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
MEMO = {}
CALLS = {"n": 0, "secs": 0.0, "nodes": 0, "fallback": 0}


def is_realised(word):
    key = tuple(int(x) for x in word)
    key = min(key, key[::-1])
    if key not in MEMO:
        t0 = time.time()
        c = {}
        try:
            v = bool(realised_word_search(key, GEARS, counter=c))
        except RuntimeError:
            CALLS["fallback"] += 1
            v = bool(realised_word(key, GEARS))
        MEMO[key] = v
        CALLS["n"] += 1
        CALLS["nodes"] += c.get("nodes", 0)
        CALLS["secs"] += time.time() - t0
    return MEMO[key]


def main():
    jmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    z = np.load(os.path.join(OUT, "m37_D3.npz"))
    win, mult = z["win"], z["mult"]
    rows = [tuple(int(x) for x in r) for r in win]
    full = [r for r in rows if all(r)]
    D3 = set(full)
    D2 = set((a, b) for a, b, c in full) | set((b, c) for a, b, c in full)
    D1 = sorted({x for r in full for x in r})
    F2 = max(a + b for a, b in D2)
    F3 = max(sum(r) for r in full)
    rep = {"D3_rows": len(rows), "D3_full": len(full), "D2": len(D2), "D1": len(D1),
           "F1": max(D1), "F2": F2, "F3": F3, "mass": int(mult.sum())}
    print(f"D_3(m37): {len(rows):,} rows ({len(full):,} of full depth), mass {int(mult.sum()):,}; "
          f"F = {max(D1)}, F_2 = {F2}, F_3 = {F3}, |D_1| = {len(D1)}, |D_2| = {len(D2)}",
          flush=True)

    # ---- J = 3: straight off the table
    best3 = max(((a + c, (a, b, c)) for a, b, c in full if b >= 6))
    hits3 = sorted({r for r in full if r[1] >= 6 and r[0] + r[2] == best3[0]})
    rep["J3"] = {"outer": best3[0], "witness": list(best3[1]), "all_maximisers": [list(h) for h in hits3]}
    print(f"  J = 3: max(g_1 + g_3) over middles >= 6 = {best3[0]}  witness {best3[1]}  "
          f"({len(hits3)} maximisers)", flush=True)

    nxt3 = {}
    prv3 = {}
    for a, b, c in full:
        nxt3.setdefault((a, b), []).append(c)
        prv3.setdefault((b, c), []).append(a)

    def scan(J, floor):
        words = []
        if J == 4:
            for (a, u, v) in full:
                if u < 6 or v < 6:
                    continue
                for b in nxt3.get((u, v), ()):
                    if a + b > floor:
                        words.append((a, u, v, b))
        elif J == 5:
            for (a, u, v) in full:
                if u < 6 or v < 6:
                    continue
                for w in nxt3.get((u, v), ()):
                    if w < 6:
                        continue
                    for b in nxt3.get((v, w), ()):
                        if a + b > floor and (a, u, v) in D3 and (u, v, w) in D3:
                            words.append((a, u, v, w, b))
        words = sorted(set(words), key=lambda t: -(t[0] + t[-1]))
        tested = 0
        for w in words:
            ok = True
            for i in range(J - 3):
                if w[i:i + 4] and not is_realised(w[i:i + 4]):
                    ok = False
                    break
            if not ok:
                continue
            tested += 1
            if is_realised(w):
                return {"outer": w[0] + w[-1], "word": list(w), "span": sum(w),
                        "candidates": len(words), "solver_words": tested}
            if tested % 100 == 0:
                print(f"    J={J}: {tested:,} solver words, at outer {w[0]+w[-1]}, "
                      f"{CALLS['secs']:.0f}s", flush=True)
        return {"outer": None, "candidates": len(words), "solver_words": tested,
                "note": f"nothing realised above {floor}"}

    for J in range(4, jmax + 1):
        r = None
        for fl in (F2, F2 - 5, F2 - 10, F2 - 16, F2 - 24, F2 - 34, F2 - 48, 0):
            r = scan(J, max(fl, 0))
            r["floor"] = max(fl, 0)
            print(f"    J = {J}, floor {max(fl,0)}: outer {r['outer']} "
                  f"(candidates {r['candidates']:,}, solver words {r['solver_words']:,}, "
                  f"{CALLS['secs']:.0f}s)", flush=True)
            if r["outer"] is not None:
                break
        rep[f"J{J}"] = r
        print(f"  J = {J}: outer max = {r['outer']}  word {r.get('word')}", flush=True)

    rep["calls"] = dict(CALLS)
    with open(os.path.join(OUT, "outer37.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "calls"}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()

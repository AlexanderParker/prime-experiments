"""u45_outer.py -- N(v) <= F_2 and the J-run outer law at a machine with no period, by the
free-phase covering instrument of research/anchor235/r70/ol_pattern.py.

The J-run outer law (research/proof/glue_covering.md 2.8(b)): for J consecutive gaps g_1..g_J of
the machine M with every one of the J-2 middles >= 6,  g_1 + g_J <= F_2(M).  J = 3 is the law
N(v) <= F_2(M) for v >= 6 of research/proof/neighbour_profile.md 2.2.

The test here is a DESCENDING SCAN: every candidate word whose adjacent pairs all lie in D_2(M)
is enumerated (that is a complete superset of the realised J-windows), sorted by descending outer
sum g_1 + g_J, and put to the covering instrument, which decides membership in D_J(M) exactly.
The first realised word is therefore the exact maximum -- no period is ever built.

Pruning that is exact, not heuristic: a realised J-window has every one of its (J-2) 3-subwindows
and (J-3) 4-subwindows realised, so those (memoised) verdicts kill a candidate before the full
word is put to the solver.  The machine is symmetric under k -> -k, so a word and its reversal are
realised together and share one memo entry.

Usage: uv run python research/anchor235/r72/u45_outer.py <y> [jmax] [floor]
   y = 23, 29 are the gates (the direct sieve of u45_sieve.py has the answers);
   y = 37 is the out-of-sample rung, D_1/D_2 read from r70/results/m37_dict.npz.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "r70")))

from ol_pattern import realised_word, realised_word_search  # noqa: E402

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

MEMO = {}
CALLS = {"n": 0, "secs": 0.0, "nodes": 0, "fallback": 0}


def is_realised(word, gears):
    key = tuple(int(x) for x in word)
    key = min(key, key[::-1])
    if key not in MEMO:
        t0 = time.time()
        c = {}
        try:
            v = bool(realised_word_search(key, gears, counter=c))
        except RuntimeError:
            CALLS["fallback"] += 1
            v = bool(realised_word(key, gears))
        MEMO[key] = v
        CALLS["n"] += 1
        CALLS["nodes"] += c.get("nodes", 0)
        CALLS["secs"] += time.time() - t0
    return MEMO[key]


def load_tables(y):
    """(D_1, D_2) for the machine {5..y}.  m37 from the closure dictionary of r70; the gate
    machines from the direct sieve of u45_sieve.py (their D_2 is then rebuilt by the instrument
    and checked against the sieve's own adjacent pairs where available)."""
    gears = [p for p in PRIMES if p <= y]
    if y == 37:
        z = np.load(os.path.join(os.path.dirname(HERE), "r70", "results", "m37_dict.npz"))
        win = z["win"]
        d1 = sorted(int(v) for v in np.unique(win[:, 0]))
        d2 = set(map(tuple, win[:, :2].astype(int)))
        src = "closure dictionary r70/results/m37_dict.npz"
    else:
        with open(os.path.join(OUT, f"sieve_m{y}.json")) as f:
            s = json.load(f)
        d1 = sorted(int(v) for v in s["spectrum"])
        d2 = None
        src = f"direct sieve results/sieve_m{y}.json"
    return gears, d1, d2, src


def build_d2(d1, gears, log=None):
    """D_2 decided by the instrument alone: every ordered pair of realised values."""
    out = set()
    n = 0
    for a in d1:
        for b in d1:
            n += 1
            if (b, a) in out or is_realised((a, b), gears):
                out.add((a, b))
        if log:
            print(f"    D_2 build: {n:,}/{len(d1)**2:,} pairs, {len(out):,} realised, "
                  f"{CALLS['secs']:.0f}s", flush=True)
    return out


def gate_d2(d1, d2, gears, nneg=200, seed=7):
    """Bounded gate of the instrument against a dictionary-supplied D_2: every dictionary pair
    must come back realised, and a random sample of non-dictionary pairs must come back not."""
    import random
    rng = random.Random(seed)
    pos_bad = [list(p) for p in sorted(d2) if not is_realised(p, gears)]
    allp = [(a, b) for a in d1 for b in d1 if (a, b) not in d2]
    rng.shuffle(allp)
    sample = allp[:nneg]
    neg_bad = [list(p) for p in sample if is_realised(p, gears)]
    return {"positives": len(d2), "positive_failures": pos_bad,
            "negatives_sampled": len(sample), "negative_failures": neg_bad}


def scan(J, d1, d2, gears, floor, cap_words=12_000_000, log=None):
    """Descending scan for max(g_1 + g_J) over realised J-windows with all middles >= 6."""
    nxt = {a: [b for b in d1 if (a, b) in d2] for a in d1}
    prv = {b: [a for a in d1 if (a, b) in d2] for b in d1}
    mids = [v for v in d1 if v >= 6]
    packed = []                      # (outer << 7J) | the word, one int64 per candidate

    def emit(mid):
        base = 0
        for x in mid:
            base = (base << 7) | x
        base <<= 7                   # room for the closing gap
        pa, nb = prv[mid[0]], nxt[mid[-1]]
        for a in pa:
            ha = (a << (7 * (J - 1))) | base
            for b in nb:
                s = a + b
                if s > floor:
                    packed.append((s << (7 * J)) | ha | b)

    if J == 3:
        for v in mids:
            emit((v,))
    elif J == 4:
        for u in mids:
            for v in nxt[u]:
                if v >= 6:
                    emit((u, v))
    elif J == 5:
        for u in mids:
            for v in nxt[u]:
                if v < 6:
                    continue
                for w in nxt[v]:
                    if w >= 6:
                        emit((u, v, w))
    else:
        raise ValueError(J)
    if len(packed) > cap_words:
        raise MemoryError(f"{len(packed):,} candidate words above the floor {floor}")
    arr = np.array(packed, dtype=np.int64)
    del packed
    arr = np.sort(arr)[::-1]
    nwords = arr.size
    tested = 0
    for key in arr:
        key = int(key)
        w = tuple((key >> (7 * (J - 1 - i))) & 127 for i in range(J))
        # exact sub-window pruning
        ok = True
        for L in (3, 4):
            if L >= J:
                continue
            for i in range(J - L + 1):
                if not is_realised(w[i:i + L], gears):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        tested += 1
        if is_realised(w, gears):
            return {"J": J, "outer": w[0] + w[-1], "word": list(w), "span": sum(w),
                    "candidates": nwords, "words_tested": tested}
        if log and tested % 200 == 0:
            print(f"    J={J} ... {tested:,} solver words, at outer {w[0]+w[-1]}, "
                  f"{CALLS['secs']:.0f}s", flush=True)
    return {"J": J, "outer": None, "word": None, "candidates": nwords,
            "words_tested": tested, "note": f"nothing realised above the floor {floor}"}


def main():
    y = int(sys.argv[1])
    jmax = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    floor = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    gears, d1, d2, src = load_tables(y)
    t0 = time.time()
    rep = {"y": y, "gears": gears, "source": src, "D1": len(d1), "d1": d1}
    if d2 is None:
        d2 = build_d2(d1, gears, log=True)
        rep["D2_from"] = "instrument"
    else:
        rep["D2_from"] = "closure dictionary"
        rep["D2_gate"] = gate_d2(d1, d2, gears)
        print(f"  gate D_2: {rep['D2_gate']['positives']} dictionary pairs, "
              f"{len(rep['D2_gate']['positive_failures'])} instrument failures; "
              f"{rep['D2_gate']['negatives_sampled']} sampled non-pairs, "
              f"{len(rep['D2_gate']['negative_failures'])} instrument false positives "
              f"({CALLS['secs']:.0f}s)", flush=True)
    rep["D2"] = len(d2)
    rep["F2_from_D2"] = max(a + b for a, b in d2)
    rep["F1"] = max(d1)
    print(f"m{y}: |D_1| = {len(d1)}, |D_2| = {len(d2)}, F = {rep['F1']}, "
          f"F_2 = {rep['F2_from_D2']}  ({src})", flush=True)
    rep["scan"] = {}
    F2 = rep["F2_from_D2"]
    for J in range(3, jmax + 1):
        # the law's own test first (is anything realised ABOVE F_2?), then descend to the exact
        # maximum; the solver memo is shared, so a lower floor only re-generates candidates.
        r = None
        for fl in ([floor] if floor else [F2, F2 - 5, F2 - 10, F2 - 16, F2 - 24, F2 - 34,
                                          F2 - 48, 0]):
            r = scan(J, d1, d2, gears, max(fl, 0), log=True)
            r["floor"] = max(fl, 0)
            print(f"    J = {J}, floor {max(fl,0)}: outer {r['outer']} "
                  f"(candidates {r['candidates']:,}, solver words {r['words_tested']:,})",
                  flush=True)
            if r["outer"] is not None:
                break
        rep["scan"][str(J)] = r
        print(f"  J = {J}: outer max = {r['outer']}  word {r['word']}  "
              f"(candidates {r['candidates']:,}, solver words {r['words_tested']:,}, "
              f"{time.time()-t0:.0f}s)", flush=True)
    rep["calls"] = dict(CALLS)
    rep["secs"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, f"outer_m{y}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("y", "F1", "F2_from_D2", "scan", "calls", "secs")},
                     indent=1)[:4000], flush=True)


if __name__ == "__main__":
    main()

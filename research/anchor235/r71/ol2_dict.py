"""ol2_dict.py -- D_1(M) and D_2(M) from the covering instrument alone, no ladder, no scan.

D_1(M) = { v : the pattern "open at 0 and v, struck at every column between" is realisable }.
D_2(M) = { (a, b) : "open at 0, a, a+b, struck at every column between" is realisable }.

Each is ONE covering problem over the gears of M (r70/ol_pattern.py), so the depth-2 dictionary
of a machine that has never been scanned -- m41, m43, m47, m53 -- is available directly.

The search range is closed by the certified record table:
  * a realised gap value is at most F(M);
  * a realised 2-window spans at most F_2(M) <= F(M + q') (the deletion-ladder cap: at some phase
    of q' the single interior opening of the stretch is deleted, so the stretch is opening-free in
    M + q').  Both caps are quoted from the corpus, never guessed.

Usage: uv run python research/anchor235/r71/ol2_dict.py <y> [procs]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import F_RECORD, OUT, decide_many, gears_upto, next_gear  # noqa: E402


def main():
    y = int(sys.argv[1])
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gears = gears_upto(y)
    q = next_gear(y)
    cap1 = F_RECORD[y]
    cap2 = F_RECORD[q]          # F_2(M) <= F(M + q')
    print(f"m{y}: gears {gears}; F(m{y}) = {cap1}; next gear {q}, F = {cap2} (cap on a 2-window)",
          flush=True)

    t0 = time.time()
    v1, s1 = decide_many([(v,) for v in range(1, cap1 + 1)], gears, procs=procs, log=True)
    D1 = sorted(v[0] for v, ok in v1.items() if ok is True)
    und1 = sorted(v for v, ok in v1.items() if ok is None)
    if und1:
        print(f"  !! UNDECIDED D_1 words: {und1}", flush=True)
    print(f"|D_1(m{y})| = {len(D1)}, max = {max(D1)} (F = {cap1}); "
          f"{s1['calls']:,} covering problems, {s1['wall']:.0f}s", flush=True)
    missing = [v for v in range(1, cap1 + 1) if v not in set(D1)]
    print(f"  absent values: {missing}", flush=True)

    # The strike set is symmetric under k -> -k (gear g strikes k iff k = +-u_g), so a window is
    # realised iff its reversal is: only the pairs a <= b are put to the instrument and the rest
    # are reflected.  Verified exactly at m37: 0 of 2,053 rows of D_2(m37) is mirror-asymmetric.
    if len(sys.argv) > 3 and sys.argv[3] == "d1":
        rep = {"y": y, "gears": gears, "q_next": q, "F": cap1, "cap2": cap2,
               "D1": D1, "D1_absent": missing, "undecided_D1": [list(w) for w in und1],
               "stats": {"D1": s1, "total_secs": round(time.time() - t0, 1)}}
        with open(os.path.join(OUT, f"dict_m{y}.json"), "w") as f:
            json.dump(rep, f)
        print(f"written results/dict_m{y}.json (D_1 only)  [{time.time()-t0:.0f}s]", flush=True)
        return

    cand = [(a, b) for a in D1 for b in D1 if a <= b and a + b <= cap2]
    print(f"D_2: {len(cand):,} candidate pairs a <= b (both entries realised, span <= {cap2}); "
          f"the reversals come for free by the mirror", flush=True)
    v2, s2 = decide_many(cand, gears, procs=procs, log=True)
    D2 = sorted(set([w for w, ok in v2.items() if ok is True]
                    + [(w[1], w[0]) for w, ok in v2.items() if ok is True]))
    und2 = sorted(w for w, ok in v2.items() if ok is None)
    if und2:
        print(f"  !! UNDECIDED D_2 words: {len(und2)}  {und2[:20]}", flush=True)
    span2 = max(a + b for a, b in D2)
    print(f"|D_2(m{y})| = {len(D2):,}, widest span {span2} = F_2(m{y}); "
          f"{s2['calls']:,} covering problems, {s2['wall']:.0f}s", flush=True)

    rep = {"y": y, "gears": gears, "q_next": q, "F": cap1, "cap2": cap2,
           "D1": D1, "D1_absent": missing, "F_2": span2,
           "undecided_D1": [list(w) for w in und1], "undecided_D2": [list(w) for w in und2],
           "D2": [list(w) for w in D2],
           "stats": {"D1": s1, "D2": s2, "total_secs": round(time.time() - t0, 1)}}
    with open(os.path.join(OUT, f"dict_m{y}.json"), "w") as f:
        json.dump(rep, f)
    print(f"written results/dict_m{y}.json  [{time.time()-t0:.0f}s total]", flush=True)


if __name__ == "__main__":
    main()

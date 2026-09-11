"""ol2_flank.py -- the flank profile of a middle word.

The binding term of B_{L+1} sits at J = L + 2, so it has exactly two (L+1)-subwindows and they
overlap in the L letters of the middle.  Its span is therefore A + C - m with m the middle's span,
A the span of the left (L+1)-window and C that of the right one.  The identity is arithmetic; the
CONTENT is whether A and C are the widest the engine allows, i.e.

    A(mid) = m + max { f <= F(M) : (f, mid...) is a realised (L+1)-window }
    C(mid) = m + max { f <= F(M) : (mid..., f) is a realised (L+1)-window }

This script measures both profiles exhaustively: every flank value from F(M) downwards is put to
the instrument until one is realised, and the whole profile is recorded so the shape can be read
(how many flank values work, not only the largest).

Usage: uv run python research/anchor235/r71/ol2_flank.py <y> <mid,as,commas> [procs] [full]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import (F_RECORD, OUT, decide_many, fj_cap, gears_upto,  # noqa: E402
                      letter_class, load_memo, memo, next_gear)

NAMES = {0: "PAD", 1: "UP", 2: "DOWN", 3: "BAD"}


def main():
    y = int(sys.argv[1])
    mid = tuple(int(x) for x in sys.argv[2].split(","))
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    full = len(sys.argv) > 4 and sys.argv[4] == "full"
    gears = gears_upto(y)
    q = next_gear(y)
    FM = F_RECORD[y]
    k = len(mid) + 1
    cap = fj_cap(y, k)
    load_memo(os.path.join(OUT, f"memo_m{y}.json"))
    M = memo()
    print(f"=== flank profile at m{y} (-> {q}), middle {list(mid)} span {sum(mid)}, "
          f"letters {[NAMES[letter_class(g, q)] for g in mid]} ===", flush=True)
    print(f"F(m{y}) = {FM}, {k}-window cap F_{k}(m{y}) <= {cap}", flush=True)
    out = {"y": y, "q": q, "middle": list(mid), "m": sum(mid), "cap": cap, "F_M": FM}
    t0 = time.time()
    for side in ("left", "right"):
        vals = [f for f in range(1, FM + 1) if cap is None or f + sum(mid) <= cap]
        words = [((f,) + mid) if side == "left" else (mid + (f,)) for f in vals]
        order = sorted(words, key=sum, reverse=True)
        if full:
            v, st = decide_many(order, gears, procs=procs)
            good = sorted(sum(w) for w in order if v[w] is True)
            top = max(good) if good else None
            out[side] = {"tested": len(order), "realised_spans": good, "max_span": top,
                         "n_realised": len(good), "secs": round(st["wall"], 1)}
            print(f"  {side}: {len(good)} of {len(order)} flank values realised, widest span "
                  f"{top} (flank {top - sum(mid) if top else None}), {st['wall']:.0f}s",
                  flush=True)
        else:
            top = None
            n = 0
            for w in order:
                v, _ = decide_many([w], gears, procs=1)
                n += 1
                if v[w] is True:
                    top = sum(w)
                    break
            out[side] = {"tested": n, "max_span": top, "exhaustive_above": True,
                         "secs": round(time.time() - t0, 1)}
            print(f"  {side}: widest realised span {top} (flank "
                  f"{top - sum(mid) if top else None}); {n} values refuted above it", flush=True)
    a, c = out["left"]["max_span"], out["right"]["max_span"]
    out["A_plus_C_minus_m"] = (a + c - sum(mid)) if (a and c) else None
    print(f"  A = {a}, C = {c}, m = {sum(mid)};  A + C - m = {out['A_plus_C_minus_m']}", flush=True)
    with open(os.path.join(OUT, f"flank_m{y}_{'_'.join(map(str, mid))}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

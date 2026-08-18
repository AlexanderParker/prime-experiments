"""Round 17 (mechanic): WHAT THE UNRESTRICTED MAXIMISER LOOKS LIKE.

Constructor r17: span + FS <= F_{k+1} is free, but the resulting
spectrum-flatness statement is FALSE at 29->31 (the unrestricted 5-window
maximum sits 42 above F while only 31 is allowed, and the true increment is
15).  So the qualifying restriction carries the whole difference.  This tool
exhibits the difference: it finds every window of j consecutive gaps whose sum
is within `slack` of F_j, prints its ADDRESS and its gap word, and classifies
the j-2 interior gaps that would have to be the word:

    literal    : every interior gap is exactly a or b (a = 2*round(q'/6))
    alternating: literal and the letters alternate a,b,a,... (a legal word)
    qualifying : every interior gap is 0 or +-s mod q' (padded letters allowed)

Usage: uv run python research/unrestricted_max.py y q' [j ...] [--slack N]
       [--limit SLOTS] [--top N]
"""
import os
import sys
import time
import numpy as np
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flank_envelope import primes_upto


def main():
    args = sys.argv[1:]
    slack, limit, top = 0, None, 12
    for flag, conv in (("--slack", int), ("--limit", lambda x: int(float(x))),
                       ("--top", int)):
        if flag in args:
            i = args.index(flag)
            v = conv(args[i + 1])
            del args[i:i + 2]
            if flag == "--slack":
                slack = v
            elif flag == "--limit":
                limit = v
            else:
                top = v
    y, q1 = int(args[0]), int(args[1])
    js = [int(x) for x in args[2:]] or [3, 4, 5, 6]
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uvals = [pow(6, -1, g) for g in gears]
    a = 2 * round(q1 / 6)
    b = q1 - a
    s = (2 * pow(6, -1, q1)) % q1
    print(f"machine {y} -> q'={q1}: letters a={a} b={b}, padded step s={s} "
          f"mod {q1}; period {P:.4g}, scanning {K:.4g}", flush=True)

    best = {j: (0, []) for j in js}
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for lo in range(0, K, 64_000_000):
        hi = min(K, lo + 64_000_000)
        ex = np.zeros(hi - lo, bool)
        for g, u in zip(gears, uvals):
            ex[(u - lo) % g::g] = True
            ex[(-u - lo) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        ops = np.concatenate([tail, op])
        if len(ops) > max(js) + 2:
            d = np.diff(ops)
            c = np.concatenate([[0], np.cumsum(d)])
            for j in js:
                if len(d) < j:
                    continue
                tot = c[j:] - c[:-j]
                new = ops[j:-1] >= lo if len(ops) > j + 1 else None
                cap = int(tot.max())
                if cap > best[j][0]:
                    best[j] = (cap, [])
                thr = max(best[j][0] - slack, 1)
                idx = np.flatnonzero(tot >= thr)
                idx = idx[ops[idx + j] >= lo]
                for i in idx[:5000]:
                    w = tuple(int(x) for x in d[i:i + j])
                    best[j][1].append((int(tot[i]), int(ops[i]), w))
                best[j] = (best[j][0],
                           sorted(set(best[j][1]), reverse=True)[:400])
        tail = ops[-(max(js) + 3):].copy()
    print(f"  scanned in {time.time()-t0:.0f}s", flush=True)

    for j in js:
        cap, lst = best[j]
        lst = [t for t in lst if t[0] >= cap - slack]
        print(f"\n=== F_{j} = {cap}   ({len(lst)} windows within {slack} of it)")
        for tot, addr, w in lst[:top]:
            inner = w[1:-1]
            lit = all(x in (a, b) for x in inner)
            alt = lit and all(inner[i] != inner[i + 1]
                              for i in range(len(inner) - 1))
            qual = all(x % q1 in (0, s, (q1 - s) % q1) for x in inner)
            print(f"   sum={tot:4d} k={addr:>12,}  flanks=({w[0]},{w[-1]}) "
                  f"interior={inner}  span={sum(inner)}  "
                  f"literal={'Y' if lit else 'n'} alternating="
                  f"{'Y' if alt else 'n'} qualifying={'Y' if qual else 'n'}")
        nlit = sum(1 for t, _, w in lst
                   if all(x in (a, b) for x in w[1:-1]))
        nq = sum(1 for t, _, w in lst
                 if all(x % q1 in (0, s, (q1 - s) % q1) for x in w[1:-1]))
        print(f"   of the {len(lst)} maximisers: {nlit} literal, "
              f"{nq} qualifying")


if __name__ == "__main__":
    main()

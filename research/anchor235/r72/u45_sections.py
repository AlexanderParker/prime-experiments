"""u45_sections.py -- W32's first-hit law on the engine's SECTION records, every prime cut.

The section of the cut p is the window's new part, the columns k with 6k-1 > p^2 and 6k+1 < q^2
(q = nextprime(p)); the machine acting there is the engine {5..p} and its openings are exactly the
twin pairs (research/proof/frontier_floor_1e7.md 1, reduction (R)).  The section record F_sec(p)
is the longest twin gap in it, in columns.

For each cut this compares F_sec(p) with W32's first-hit prediction

    F_range(N) = max { d : P(p) / c(d) <= N } - 1,     N = the section's column count,

with c(d) the EXACT full-period census of {5..p} from u45_census.py (a covering count -- P(p) is
astronomical and is never built), and, where affordable, with the record distribution over random
translates of the same gear set and the same length.

Usage: uv run python research/anchor235/r72/u45_sections.py [pmax] [samples]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from u45_census import c_of_gears, gears_upto, primes_upto  # noqa: E402


def openings(gears, klo, khi):
    n = khi - klo + 1
    blocked = np.zeros(n, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        blocked[(u - klo) % g::g] = True
        blocked[((-u) - klo) % g::g] = True
    return np.flatnonzero(~blocked).astype(np.int64) + klo


def translate_records(gears, N, samples, rng, batch=2048):
    ar = np.arange(N)
    tabs = []
    for g in gears:
        u = pow(6, -1, g)
        t = (np.arange(g)[:, None] + ar[None, :]) % g
        tabs.append((g, (t == u % g) | (t == (-u) % g)))
    out = np.empty(samples, dtype=np.int32)
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        blocked = np.zeros((b, N), dtype=bool)
        for g, T in tabs:
            blocked |= T[rng.integers(0, g, size=b)]
        op = ~blocked
        prev = np.maximum.accumulate(np.where(op, ar[None, :], -1), axis=1)
        pv = np.concatenate([np.full((b, 1), -1, dtype=prev.dtype), prev[:, :-1]], axis=1)
        out[done:done + b] = np.where(op & (pv >= 0), ar[None, :] - pv, 0).max(axis=1)
        done += b
    return out


def main():
    pmax = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    ctrl_max = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    ps = [p for p in primes_upto(pmax + 200) if p >= 7]
    rng = np.random.default_rng(20260910)
    rows = []
    t0 = time.time()
    for i, p in enumerate(ps[:-1]):
        if p > pmax:
            break
        q = ps[i + 1]
        gears = gears_upto(p)
        P = 1
        for g in gears:
            P *= g
        lo = 1
        while 6 * lo - 1 <= p * p:
            lo += 1
        hi = (q * q - 2) // 6
        while 6 * hi + 1 >= q * q:
            hi -= 1
        N = hi - lo + 1
        op = openings(gears, lo, hi)
        if op.size < 2:
            rows.append({"p": p, "q": q, "N": N, "twins": int(op.size), "F": None})
            continue
        dif = np.diff(op)
        F = int(dif.max())
        at = int(op[int(np.argmax(dif))])
        d, last, peak = 1, 0, 0
        try:
            while d <= 400:
                c, pk = c_of_gears(gears, d, cap=3_000_000)
                peak = max(peak, pk)
                if c == 0 or c * N < P:
                    break
                last, d = d, d + 1
            pred = last - 1
        except MemoryError:
            pred = None
        row = {"p": p, "q": q, "N": N, "twins": int(op.size), "F": F, "at": at,
               "pred": pred, "diff": None if pred is None else F - pred, "peak_states": peak}
        if pred is not None and p <= ctrl_max and N <= 4000:
            rec = translate_records(gears, N, samples, rng)
            row["median"] = int(np.median(rec))
            row["p99"] = int(np.percentile(rec, 99))
            row["P_ge_F"] = round(float((rec >= F).mean()), 5)
        rows.append(row)
        if pred is None:
            print(f"p={p}: census out of reach at N={N}", flush=True)
            break
        print(f"p={p:5d} q={q:5d} N={N:5d} twins={row['twins']:4d} F={F:4d} pred={pred:4d} "
              f"diff={row['diff']:+3d} median={row.get('median')} "
              f"P(rec>=F)={row.get('P_ge_F')} [{peak} states]", flush=True)
    good = [r for r in rows if r.get("diff") is not None]
    diffs = [r["diff"] for r in good]
    summ = {"cuts": len(good), "pmax_reached": good[-1]["p"] if good else None,
            "within_1": sum(1 for d in diffs if abs(d) <= 1),
            "within_2": sum(1 for d in diffs if abs(d) <= 2),
            "mean_diff": round(sum(diffs) / len(diffs), 3) if diffs else None,
            "min_diff": min(diffs) if diffs else None, "max_diff": max(diffs) if diffs else None,
            "secs": round(time.time() - t0, 1)}
    print(json.dumps(summ, indent=1), flush=True)
    with open(os.path.join(OUT, "sections.json"), "w") as f:
        json.dump({"rows": rows, "summary": summ}, f, indent=1)


if __name__ == "__main__":
    main()

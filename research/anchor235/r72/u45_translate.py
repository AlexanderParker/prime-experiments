"""u45_translate.py -- the control for U5: is the engine's WINDOW an ordinary translate?

W32's first-hit law is a statement about a range of N columns of a fixed gear set.  u45_window.py
measures it on the one translate the engine cares about -- the phase-zero one, the window
(y, y'^2).  This script measures the same record on RANDOM translates of the same gear set and the
same length, so the window's record can be placed in the machine's own distribution.

By CRT a uniform random column of the period is a uniform independent residue in each Z_g, so the
translates are drawn exactly, with no period.

Usage: uv run python research/anchor235/r72/u45_translate.py [samples]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

from u45_census import gears_upto  # noqa: E402

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def lut(g, N):
    """(g, N) bool: LUT[r, o] = True iff the gear strikes offset o of a window starting at a
    column with x = r (mod g)."""
    u = pow(6, -1, g)
    o = np.arange(N)
    r = np.arange(g)[:, None]
    t = (r + o[None, :]) % g
    return (t == u % g) | (t == (-u) % g)


def records(y, N, samples, batch=4096, rng=None):
    """The longest gap between consecutive openings in `samples` random translates of length N."""
    gears = gears_upto(y)
    rng = rng or np.random.default_rng(12345)
    tabs = [(g, lut(g, N)) for g in gears]
    ar = np.arange(N)
    out = np.empty(samples, dtype=np.int32)
    nop = np.empty(samples, dtype=np.int32)
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        blocked = np.zeros((b, N), dtype=bool)
        for g, T in tabs:
            blocked |= T[rng.integers(0, g, size=b)]
        op = ~blocked
        prev = np.maximum.accumulate(np.where(op, ar[None, :], -1), axis=1)
        pv = np.concatenate([np.full((b, 1), -1, dtype=prev.dtype), prev[:, :-1]], axis=1)
        gap = np.where(op & (pv >= 0), ar[None, :] - pv, 0)
        out[done:done + b] = gap.max(axis=1)
        nop[done:done + b] = op.sum(axis=1)
        done += b
    return out, nop


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    with open(os.path.join(OUT, "window.json")) as f:
        rows = json.load(f)["rows"]
    rep = []
    t0 = time.time()
    for row in rows:
        y = row["y"]
        r = {"y": y}
        for tag in ("window", "section"):
            w = row[tag]
            N, F, pred = w["N"], w["F"], w["pred"]
            if F is None:
                continue
            rec, nop = records(y, N, samples)
            r[tag] = {
                "N": N, "F": F, "pred": pred,
                "median": int(np.median(rec)), "mean": round(float(rec.mean()), 2),
                "p90": int(np.percentile(rec, 90)), "p99": int(np.percentile(rec, 99)),
                "max": int(rec.max()),
                "P_ge_F": round(float((rec >= F).mean()), 6),
                "percentile_of_F": round(float((rec < F).mean()) * 100, 3),
                "openings_mean": round(float(nop.mean()), 2),
                "openings_in_window": w["openings"],
            }
        rep.append(r)
        w, s = r.get("window"), r.get("section")
        print(f"y={y:3d} window N={w['N']:4d}: F_W={w['F']} pred={w['pred']} "
              f"median={w['median']} p99={w['p99']} max={w['max']} "
              f"P(rec>=F_W)={w['P_ge_F']}  twins {w['openings_in_window']} vs mean "
              f"{w['openings_mean']}", flush=True)
        if s:
            print(f"        section N={s['N']:4d}: F={s['F']} pred={s['pred']} "
                  f"median={s['median']} p99={s['p99']} P(rec>=F)={s['P_ge_F']}", flush=True)
    with open(os.path.join(OUT, "translate.json"), "w") as f:
        json.dump({"samples": samples, "rows": rep, "secs": round(time.time() - t0, 1)}, f,
                  indent=1)
    print(f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

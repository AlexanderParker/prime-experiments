"""W3 mechanism - why the tail gears (g > d) behave differently under different separations.

A gear g > d can strike at most two islands of [1, d), and it strikes two exactly when some
reachable phase puts its two classes on two islands.  Its two classes are s_g apart, and both
islands lie in [1, d) < g, so the two islands must differ by exactly

        delta(g)  in  {s_g, g - s_g}  intersected with  (0, d).

So a tail gear is usable at all only if one of its two "tooth distances" lands inside the arc, and
it is usable for a PARTICULAR pair only if that distance is exactly the pair's difference.  Two
filters follow, and they are the whole story for the tail:

  F1 (arithmetic): delta must be an island difference mod 35, i.e. delta mod 35 in
     {0, 2, 5, 7, 12, 23, 28, 30, 33} (the differences of {5, 10, 12, 17}).  9 of 35 residues.
  F2 (length): the number of island pairs at difference delta falls linearly in delta, roughly
     (d - delta) * c(delta mod 35) / 35, so a LARGE delta leaves few pairs.

The real separation has  3 s_g = 1 (mod g),  so  s_g = (g+1)/3 or (2g+1)/3  and
min(s_g, g - s_g) = (g -+ 1)/3  EXACTLY, for every gear.  A random separation has
min(s_g, g - s_g) uniform on (0, g/2).  This script measures both filters.

Usage: uv run python research/anchor235/r43/sep_tail.py --d 560 --nrand 30 --tag t560
"""
import argparse
import json
import os
import random
from math import isqrt


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def pair_count(isl_set, d, delta):
    if delta <= 0 or delta >= d:
        return 0
    return sum(1 for i in isl_set if (i + delta) in isl_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=560)
    ap.add_argument("--nrand", type=int, default=30)
    ap.add_argument("--tag", type=str, default="t")
    args = ap.parse_args()
    d = args.d
    isl = islands(d)
    S = set(isl)
    FL = sieve(3 * d + 10)
    TAIL = [p for p in range(d + 1, 3 * d + 3) if FL[p]]
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(OUT, exist_ok=True)

    D35 = sorted({(a - b) % 35 for a in (5, 10, 12, 17) for b in (5, 10, 12, 17)})
    print("d=%d  m=%d  tail gears in (d, 3d]: %d   island differences mod 35: %s"
          % (d, len(isl), len(TAIL), D35), flush=True)

    def stats(usable_deltas):
        """usable_deltas: list of the set of tooth distances inside (0,d) for each tail gear."""
        nd = [len(x) for x in usable_deltas]
        f1 = 0
        pc = []
        for x in usable_deltas:
            best = max([pair_count(S, d, dd) for dd in x], default=0)
            pc.append(best)
            if best > 0:
                f1 += 1
        alld = [dd for x in usable_deltas for dd in x]
        return dict(mean_ndelta=sum(nd) / len(nd),
                    mean_delta_over_d=(sum(alld) / len(alld) / d) if alld else 0.0,
                    frac_delta_is_island_diff=f1 / len(usable_deltas),
                    mean_pairs_at_best_delta=sum(pc) / len(pc),
                    max_pairs=max(pc))

    res = {"d": d, "m": len(isl), "ntail": len(TAIL), "D35": D35}

    real = []
    for g in TAIL:
        s = pow(3, -1, g)
        real.append([x for x in (s, g - s) if 0 < x < d])
    res["real"] = stats(real)
    print("real      ndelta %.3f  mean delta/d %.3f  frac usable %.3f  mean pairs %.2f  max %d"
          % tuple(res["real"][k] for k in ("mean_ndelta", "mean_delta_over_d",
                                           "frac_delta_is_island_diff",
                                           "mean_pairs_at_best_delta", "max_pairs")), flush=True)

    for cr in ["1/2", "1/5", "2/5", "2/7", "2/11", "2/13"]:
        c, r = (int(v) for v in cr.split("/"))
        rows = []
        for g in TAIL:
            if g == r:
                continue
            s = (c * pow(r, -1, g)) % g
            rows.append([x for x in (s, g - s) if 0 < x < d])
        res["coh:" + cr] = stats(rows)
        print("coh %-5s ndelta %.3f  mean delta/d %.3f  frac usable %.3f  mean pairs %.2f  max %d"
              % ((cr,) + tuple(res["coh:" + cr][k] for k in
                               ("mean_ndelta", "mean_delta_over_d",
                                "frac_delta_is_island_diff", "mean_pairs_at_best_delta",
                                "max_pairs"))), flush=True)

    acc = []
    for k in range(args.nrand):
        rng = random.Random(31000 + k)
        rows = []
        for g in TAIL:
            s = rng.randrange(1, g)
            rows.append([x for x in (s, g - s) if 0 < x < d])
        acc.append(stats(rows))
    keys = ("mean_ndelta", "mean_delta_over_d", "frac_delta_is_island_diff",
            "mean_pairs_at_best_delta", "max_pairs")
    res["rand"] = {k: dict(mean=sum(a[k] for a in acc) / len(acc),
                           lo=min(a[k] for a in acc), hi=max(a[k] for a in acc)) for k in keys}
    print("rand n=%d" % args.nrand, flush=True)
    for k in keys:
        v = res["rand"][k]
        print("   %-28s mean %.3f  [%.3f, %.3f]   real %.3f"
              % (k, v["mean"], v["lo"], v["hi"], res["real"][k]), flush=True)

    p = os.path.join(OUT, "sep_tail_%s.json" % args.tag)
    json.dump(res, open(p, "w"), indent=1)
    print("written", p)


main()

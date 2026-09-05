"""Branch 2f.i - the converse sweeps at 23 -> 29.

Sub-families swept (M = {5..23}, q' = 29, one lower period of 37,182,145 columns per member,
the machinery of research/proof/chain_family_r32.py reused unchanged):

  coh      every admissible rational c/r: the OLD gears all coherent, incoming tooth PINNED
           to v' = 5 (so the letters are a = 10, b = 19, 3a = q' + 1)
  cohfull  every admissible rational: ALL gears coherent, incoming gear included (its tooth is
           then the rational's own value, not necessarily the pinned one)
  oneoff   coherent in all but one gear: one old gear takes every other tooth value, the rest
           stay on the rational; incoming tooth pinned  (= "compatible in all but one pair",
           the pairs through the odd gear being the incompatible ones)
  ctrl     a size-matched random pinned control (fixed seed) - the base rate, measured

Usage: uv run python research/anchor235/r46/cp_sweep.py coh
       uv run python research/anchor235/r46/cp_sweep.py oneoff --procs 3
"""
import argparse
import json
import os
import random
import sys
import time
from math import prod
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "proof"))
from cp_compat import OUT, admissible, best_core, coherent_member, incompat  # noqa: E402
from chain_family_r32 import (gaps_of, gears_of, letter_a, next_prime, open_mask,  # noqa: E402
                              qstar_table, real_tooth, summarize)

Y = 23
GEARS = gears_of(Y)
Q1 = next_prime(Y)
VPIN = real_tooth(Q1)
B = 30


def evaluate(arg):
    teeth, v1 = arg
    g = gaps_of(open_mask(GEARS, list(teeth), prod(GEARS)))
    a = letter_a(Q1, v1)
    F, F2, tab = qstar_table(g, Q1, a)
    s = summarize(F, F2, tab, Q1)
    return dict(teeth=list(teeth), v1=v1, a=a, F=F, F2=F2, chain=s["chain"],
                L=s["L"], viol={str(k): v for k, v in s["viol"].items()},
                pair_ok=s["pair_ok"],
                argmax={str(k): v for k, v in s["argmax"].items()})


def tmin_ok(teeth):
    """(T): no gear has adjacent teeth."""
    return all(min((2 * v) % q, q - (2 * v) % q) >= 2 for q, v in zip(GEARS, teeth))


def build(mode, seed=46):
    rats_old = admissible(GEARS, B)
    rats_all = admissible(GEARS + [Q1], B)
    if mode == "coh":
        seen = {}
        for (r, c) in rats_old:
            t = coherent_member(GEARS, r, c)
            if t is not None:
                seen.setdefault(tuple(t), []).append((r, c))
        return [(t, VPIN) for t in seen], seen
    if mode == "cohfull":
        seen = {}
        for (r, c) in rats_all:
            t = coherent_member(GEARS + [Q1], r, c)
            if t is not None:
                seen.setdefault(tuple(t), []).append((r, c))
        return [(t[:-1], t[-1]) for t in seen], seen
    if mode == "oneoff":
        base = {}
        for (r, c) in rats_old:
            t = coherent_member(GEARS, r, c)
            if t is not None:
                base.setdefault(tuple(t), []).append((r, c))
        out = {}
        for t in base:
            for i, q in enumerate(GEARS):
                for v in range(1, (q - 1) // 2 + 1):
                    if v == t[i]:
                        continue
                    tt = list(t)
                    tt[i] = v
                    out[tuple(tt)] = (q, v, t)
        return [(t, VPIN) for t in out], out
    if mode == "ctrl":
        rng = random.Random(seed)
        out = set()
        n = int(os.environ.get("CTRL_N", "2600"))
        while len(out) < n:
            out.add(tuple(rng.randrange(1, (q - 1) // 2 + 1) for q in GEARS))
        return [(t, VPIN) for t in sorted(out)], None
    raise SystemExit("mode?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()
    members, aux = build(args.mode)
    print("mode %s: %d members, machine {5..%d} + q'=%d, pinned v'=%d, period %d"
          % (args.mode, len(members), Y, Q1, VPIN, prod(GEARS)))
    t0 = time.time()
    rows = []
    with Pool(args.procs) as p:
        for i, r in enumerate(p.imap(evaluate, members, chunksize=4)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(members), time.time() - t0), flush=True)
    rats_all = admissible(GEARS + [Q1], B)
    nb = nc = np_ = 0
    nT = 0
    vT = 0
    for r in rows:
        chain_v = bool(r["viol"])
        pair_v = not r["pair_ok"]
        r["budget_viol"] = chain_v or pair_v
        r["T"] = tmin_ok(r["teeth"])
        k, rc, core = best_core(GEARS + [Q1], r["teeth"] + [r["v1"]], rats_all)
        r["k"] = k
        r["I"] = incompat(len(GEARS) + 1, k)
        r["rc"] = rc
        nc += chain_v
        np_ += pair_v
        nb += r["budget_viol"]
        nT += r["T"]
        vT += r["T"] and r["budget_viol"]
    print("== %s: %d members, %d budget violators (%d chain, %d pair), rate %.4f%%"
          % (args.mode, len(rows), nb, nc, np_, 100.0 * nb / len(rows)))
    print("   with (T): %d members, %d budget violators, rate %.4f%%"
          % (nT, vT, 100.0 * vT / nT if nT else 0.0))
    for r in rows:
        if r["budget_viol"]:
            extra = ""
            if args.mode == "oneoff" and aux:
                q, v, base = aux[tuple(r["teeth"])]
                extra = " | odd gear %d at v=%d, base rational member %s" % (q, v, list(base))
            print("   VIOL teeth=%s v'=%d F=%d F2=%d chain=%d budget=%d viol=%s pair_ok=%s "
                  "k=%d I=%d (T)=%s%s"
                  % (r["teeth"], r["v1"], r["F"], r["F2"], r["chain"], r["F"] + Q1,
                     r["viol"], r["pair_ok"], r["k"], r["I"], r["T"], extra))
    with open(os.path.join(OUT, "sweep_%s_m23.json" % args.mode), "w") as f:
        json.dump(rows, f)
    print("   %.0fs total" % (time.time() - t0))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()

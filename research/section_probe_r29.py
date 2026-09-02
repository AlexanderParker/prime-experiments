"""New window sections per machine - manager, round 29.

Pre-registration: research/data/r29/section_prereg.md (S1..S5).

Section of the rung p -> q (consecutive primes >= 5): slots k with p^2 < 6k+1 < q^2. Inside
it gears 5..p are exact (every composite below q^2 has a factor <= p) and gear q is silent.

Usage: python section_probe_r29.py [--qmax 5003] [--words 12]
"""
import argparse
import math
from collections import Counter

import numpy as np

from word_tree_r29 import spf_sieve, death_rungs, runs_of

NGATE, NFAIL = 0, 0
TWO_C2 = 1.3203236


def gate(cond, msg):
    global NGATE, NFAIL
    NGATE += 1
    if not cond:
        NFAIL += 1
        print("  GATE FAIL: " + msg)
    else:
        print("  ASSERT ok: " + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=5003)
    ap.add_argument("--words", type=int, default=12)
    a = ap.parse_args()
    primes = [int(p) for p in np.flatnonzero(spf_sieve(a.qmax + 100) == np.arange(a.qmax + 101)) if p >= 5]
    ps = [p for p in primes if p <= a.qmax]
    Wmax = (ps[-1] ** 2 - 2) // 6
    spf = spf_sieve(6 * Wmax + 1)
    r_all = death_rungs(Wmax, spf)

    rows = []
    print("=== sections p -> q: slots with p^2 < 6k+1 < q^2 ===")
    for i in range(len(ps) - 1):
        p, q = ps[i], ps[i + 1]
        k_lo = p * p // 6 + 1
        k_hi = (q * q - 2) // 6
        r = r_all[k_lo - 1:k_hi]
        S = k_hi - k_lo + 1
        tw = np.flatnonzero(r == 0)
        n_tw = int(tw.size)
        if n_tw >= 2:
            G_in = int(np.diff(tw).max())
        else:
            G_in = 0
        if n_tw:
            G_edge = int(max(tw[0] + 1, S - tw[-1]))
        else:
            G_edge = S
        G = max(G_in, G_edge)
        ks = np.arange(k_lo, k_hi + 1)
        hl = float((6 * TWO_C2 / np.log(6 * ks) ** 2).sum())
        kills_p = int((r == p).sum())
        cand = [m for m in primes if p <= m and p * m < q * q]
        bw = r[r > 0]
        f7 = float((bw <= 7).mean()) if bw.size else 0.0
        dens = 1 - n_tw / S
        rs = runs_of(r > 0)
        if rs:
            s, e = max(rs, key=lambda t: t[1] - t[0])
            sub = r[s:e + 1]
            last = int(sub.max()); nseal = len(set(sub.tolist())); runlen = e - s + 1
        else:
            last = 0; nseal = 0; runlen = 0
        rows.append(dict(p=p, q=q, k_lo=k_lo, k_hi=k_hi, S=S, n=n_tw, G=G, G_in=G_in, G_edge=G_edge,
                         hl=hl, kills_p=kills_p, ncand=len(cand), f7=f7, dens=dens,
                         last=last, nseal=nseal, runlen=runlen))
        if i < a.words:
            word = " ".join("T" if x == 0 else str(int(x)) for x in r)
            print(f"  {p:>4} -> {q:<4} numbers ({p * p}, {q * q}) slots {k_lo}..{k_hi} ({S}): {word}")
            print(f"        twins {n_tw}, H-L {hl:.2f}, gear {p} kills {kills_p} of {len(cand)} candidates "
                  f"{[p * m for m in cand]}, max run {runlen} sealed by {nseal} gears (last {last})")

    print("\n=== selected sections ===")
    print(f"{'p':>5} {'q':>5} {'|S|':>6} {'twins':>5} {'H-L':>7} {'obs/HL':>6} {'G_S':>4} {'G/|S|':>6} "
          f"{'p-kills':>7} {'cand':>4} {'<=7 x dens':>10} {'run':>4} {'seal':>4} {'last/p':>6}")
    show = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 97, 101, 199, 211, 499, 503, 997, 1009,
            1999, 2003, 4999, 5003}
    for d in rows:
        if d["q"] in show:
            print(f"{d['p']:>5} {d['q']:>5} {d['S']:>6} {d['n']:>5} {d['hl']:>7.1f} {d['n'] / d['hl']:>6.2f} "
                  f"{d['G']:>4} {d['G'] / d['S']:>6.3f} {d['kills_p']:>7} {d['ncand']:>4} "
                  f"{d['f7'] * d['dens']:>10.3f} {d['runlen']:>4} {d['nseal']:>4} {d['last'] / d['p']:>6.2f}")

    print("\n=== aggregates by q range ===")
    bands = [(5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 5003)]
    print(f"{'q range':>12} {'sections':>8} {'min twins':>9} {'at q':>6} {'max G/|S|':>9} {'at q':>6} "
          f"{'twins/HL':>8} {'p-kills=0':>9} {'mean p-kills':>12} {'last>p/2':>8} {'<=7 x dens':>10}")
    for lo, hi in bands:
        sel = [d for d in rows if lo <= d["q"] <= hi]
        if not sel:
            continue
        mn = min(sel, key=lambda d: d["n"]); mg = max(sel, key=lambda d: d["G"] / d["S"])
        tot = sum(d["n"] for d in sel); hl = sum(d["hl"] for d in sel)
        z = sum(1 for d in sel if d["kills_p"] == 0) / len(sel)
        mk = np.mean([d["kills_p"] for d in sel])
        big = sum(1 for d in sel if d["last"] > d["p"] / 2) / len(sel)
        f7 = np.mean([d["f7"] * d["dens"] for d in sel])
        print(f"{lo:>5}-{hi:<6} {len(sel):>8} {mn['n']:>9} {mn['q']:>6} {mg['G'] / mg['S']:>9.3f} {mg['q']:>6} "
              f"{tot / hl:>8.3f} {z:>9.3f} {mk:>12.3f} {big:>8.3f} {f7:>10.3f}")

    print("\n=== gates ===")
    z_all = [d for d in rows if d["n"] == 0]
    gate(not z_all, f"S1 no dead section up to q={a.qmax} ({len(z_all)} dead)")
    m100 = min(d["n"] for d in rows if d["q"] >= 100); m1000 = min(d["n"] for d in rows if d["q"] >= 1000)
    gate(m100 >= 3, f"S1 min twins q >= 100 is {m100} >= 3")
    gate(m1000 >= 20, f"S1 min twins q >= 1000 is {m1000} >= 20")
    gate(all(d["G"] < d["S"] for d in rows), "S2 G_S < |S| at every section")
    g100 = max(d["G"] / d["S"] for d in rows if d["q"] >= 100)
    gate(g100 < 0.5, f"S2 max G_S/|S| at q >= 100 is {g100:.3f} < 0.5")
    sel = [d for d in rows if 1000 <= d["q"] <= 5003]
    ratio = sum(d["n"] for d in sel) / sum(d["hl"] for d in sel)
    gate(abs(ratio - 1) <= 0.03, f"S3 twins / Hardy-Littlewood over 1000 <= q <= 5003 = {ratio:.4f} (band 3%)")
    gate(all(d["kills_p"] <= 3 for d in rows), f"S4 gear-p kills in its section <= 3 (max {max(d['kills_p'] for d in rows)})")
    sel = [d for d in rows if d["q"] >= 500]
    z = sum(1 for d in sel if d["kills_p"] == 0) / len(sel)
    gate(0.35 <= z <= 0.70, f"S4 fraction of sections (q >= 500) with zero gear-p kills = {z:.3f} in [0.35, 0.70]")
    big = sum(1 for d in sel if d["last"] > d["p"] / 2) / len(sel)
    gate(0.30 <= big <= 0.90, f"S5 last sealer > p/2 at {big:.3f} of sections q >= 500 (band [0.30, 0.90])")
    print(f"\nGATES: {NGATE - NFAIL} passed, {NFAIL} failed of {NGATE}")


if __name__ == "__main__":
    main()

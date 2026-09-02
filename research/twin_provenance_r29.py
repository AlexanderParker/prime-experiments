"""Provenance of the new twins in each section - manager, round 29.

Pre-registration: research/data/r29/provenance_prereg.md (V1..V4).

For every twin in the section p -> q (slots with p^2 < 6k+1 < q^2, the part of the window
that machine q adds) the twin is traced through the sub-machines m_5, m_7, ..., m_p: at level
r it sits at position k mod r of gear r's word and inside the local word of m_r, the gap pair
(L_r, R_r). The word changes at the gears that kill the opening bounding it; those gears are
the twin's interacting gears. Gear q is never among them (it kills nothing in its section).

Usage: python twin_provenance_r29.py [--qmax 5003] [--print 31,53] [--print2 997]
"""
import argparse
from collections import Counter

import numpy as np

from twin_path_r29 import flank_events, iid_records_mean, tv
from word_tree_r29 import spf_sieve, death_rungs

NGATE, NFAIL = 0, 0


def gate(cond, msg):
    global NGATE, NFAIL
    NGATE += 1
    if not cond:
        NFAIL += 1
        print("  GATE FAIL: " + msg)
    else:
        print("  ASSERT ok: " + msg)


def open_classes(mod_gears):
    """residue classes mod prod(mod_gears) that avoid every tooth."""
    P = 1
    for g in mod_gears:
        P *= g
    ok = np.ones(P, dtype=bool)
    for g in mod_gears:
        u = pow(6, -1, g)
        ok[u % g::g] = False
        ok[(-u) % g::g] = False
    return P, np.flatnonzero(ok)


def flanks(r_all, k):
    """death rungs walking left and right from the twin at slot k (1-based) until the next
    opening; None if the walk leaves the sieved range."""
    n = r_all.size
    i = k - 2  # index of slot k-1
    seq_l = []
    while i >= 0 and r_all[i] > 0:
        seq_l.append(int(r_all[i])); i -= 1
    if i < 0:
        return None, None
    i = k  # index of slot k+1
    seq_r = []
    while i < n and r_all[i] > 0:
        seq_r.append(int(r_all[i])); i += 1
    if i >= n:
        return None, None
    return seq_l, seq_r


def word_levels(seq_l, seq_r, gears):
    """(L_r, R_r) at every level r in gears, from the flank sequences."""
    out = []
    for g in gears:
        L = next((d + 1 for d, v in enumerate(seq_l) if v > g), len(seq_l) + 1)
        R = next((d + 1 for d, v in enumerate(seq_r) if v > g), len(seq_r) + 1)
        out.append((g, L, R))
    return out


def letters(r_all, k, level, rad=8):
    s = []
    for j in range(k - rad, k + rad + 1):
        if j < 1 or j > r_all.size:
            s.append(" ")
        else:
            v = r_all[j - 1]
            s.append("o" if v == 0 or v > level else ("T" if v == 0 else "x"))
    s[rad] = "T"
    return "".join(s)


def print_provenance(r_all, k, seq_l, seq_r, gears, p):
    ev_l = flank_events(seq_l)
    ev_r = flank_events(seq_r)
    change = {g for g, _, _ in ev_l} | {g for g, _, _ in ev_r}
    print(f"    twin at slot {k} ({6 * k - 1}, {6 * k + 1}); residues " +
          " ".join(f"{g}:{k % g}" for g in gears if g <= 23) + (" ..." if p > 23 else ""))
    wl = word_levels(seq_l, seq_r, gears)
    last = None
    for g, L, R in wl:
        if (L, R) != last or g == gears[0] or g == gears[-1]:
            tag = "changes" if g in change else ("start" if g == gears[0] else "final")
            print(f"      level {g:>5} (k mod {g} = {k % g:>4}): word ({L}, {R})  {letters(r_all, k, g)}  {tag}")
            last = (L, R)
    fl = seq_l[-1] if seq_l else 0
    fr = seq_r[-1] if seq_r else 0
    print(f"      interacting gears: left {[g for g, _, _ in ev_l]}, right {[g for g, _, _ in ev_r]}; "
          f"framing pair ({fl}, {fr})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=5003)
    ap.add_argument("--print", dest="pr", type=str, default="31,53")
    ap.add_argument("--print2", type=str, default="997")
    a = ap.parse_args()
    primes = [int(p) for p in np.flatnonzero(spf_sieve(a.qmax + 100) == np.arange(a.qmax + 101)) if p >= 5]
    ps = [p for p in primes if p <= a.qmax]
    Wmax = (ps[-1] ** 2 - 2) // 6
    spf = spf_sieve(6 * Wmax + 1)
    r_all = death_rungs(Wmax, spf)
    pr_full = {int(x) for x in a.pr.split(",") if x}
    pr_two = {int(x) for x in a.print2.split(",") if x}

    bands = [(5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 5003)]
    mods = {5: open_classes([5]), 35: open_classes([5, 7]), 385: open_classes([5, 7, 11])}
    res_cnt = {m: Counter() for m in mods}
    frame = Counter()
    per_band = {b: dict(n=0, ev_l=[], ev_r=[], Ll=[], big=0, final13=0, rungs=Counter(), nb=0) for b in bands}
    print("=== provenance of the new twins per section ===")
    skipped = 0
    for i in range(len(ps) - 1):
        p, q = ps[i], ps[i + 1]
        gears = ps[:i + 1]
        k_lo = p * p // 6 + 1
        k_hi = (q * q - 2) // 6
        band = next(b for b in bands if b[0] <= q <= b[1])
        d = per_band[band]
        tws = [int(k) for k in np.flatnonzero(r_all[k_lo - 1:k_hi] == 0) + k_lo]
        if q in pr_full or q in pr_two:
            print(f"  --- section {p} -> {q}: slots {k_lo}..{k_hi}, {len(tws)} new twins ---")
        shown = 0
        for k in tws:
            seq_l, seq_r = flanks(r_all, k)
            if seq_l is None:
                skipped += 1
                continue
            ev_l = flank_events(seq_l); ev_r = flank_events(seq_r)
            d["n"] += 1
            d["ev_l"].append(len(ev_l)); d["ev_r"].append(len(ev_r))
            d["Ll"].append(len(seq_l) + 1)
            top = max([g for g, _, _ in ev_l] + [g for g, _, _ in ev_r])
            if top > p / 2:
                d["big"] += 1
            if top <= 13:
                d["final13"] += 1
            for v in seq_l + seq_r:
                d["rungs"][v] += 1; d["nb"] += 1
            if q >= 1000:
                for m in mods:
                    res_cnt[m][k % m] += 1
                fl = seq_l[-1] if seq_l else 0
                fr = seq_r[-1] if seq_r else 0
                frame[(fl, fr)] += 1
            if q in pr_full or (q in pr_two and shown < 2):
                print_provenance(r_all, k, seq_l, seq_r, gears, p)
                shown += 1
    print(f"  (twins whose flank leaves the sieved range, skipped: {skipped})")

    print("\n=== V1: residue combinations of the new twins, sections q >= 1000 pooled ===")
    tvs = {}
    for m, (P, cls) in mods.items():
        n = sum(res_cnt[m].values())
        obs = {int(c): res_cnt[m].get(int(c), 0) / n for c in cls}
        uni = {int(c): 1 / len(cls) for c in cls}
        tvs[m] = tv(obs, uni)
        stray = sum(v for c, v in res_cnt[m].items() if c not in set(int(x) for x in cls))
        print(f"  mod {m:>3}: {len(cls)} open classes, {n} twins, TV from uniform {tvs[m]:.4f}, "
              f"twins in tooth classes {stray}")
        if m == 5:
            print("    " + " ".join(f"k={int(c)}:{obs[int(c)]:.3f}" for c in cls))
        if m == 35:
            lo = min(obs, key=obs.get); hi = max(obs, key=obs.get)
            print(f"    least class {lo} ({obs[lo]:.4f}), most {hi} ({obs[hi]:.4f}), uniform {1 / len(cls):.4f}")

    print("\n=== V2: framing pairs (left rung, right rung), q >= 1000 pooled ===")
    n = sum(frame.values())
    joint = {k: v / n for k, v in frame.items()}
    ml = Counter(); mr = Counter()
    for (l, r), v in frame.items():
        ml[l] += v; mr[r] += v
    prod = {(l, r): ml[l] * mr[r] / n / n for l in ml for r in mr}
    tv_f = tv(joint, prod)
    print(f"  {n} twins; TV(joint, product of marginals) = {tv_f:.4f}")
    print("  left marginal:  " + " ".join(f"{g}:{ml[g] / n:.3f}" for g in [5, 7, 11, 13, 17, 19, 23]))
    print("  right marginal: " + " ".join(f"{g}:{mr[g] / n:.3f}" for g in [5, 7, 11, 13, 17, 19, 23]))
    print("  most common pairs: " + ", ".join(f"({l},{r}) {v / n:.3f}" for (l, r), v in frame.most_common(8)))

    print("\n=== V3/V4: interacting gears per new twin, by q range ===")
    print(f"{'q range':>12} {'twins':>7} {'mean events L':>13} {'mean events R':>13} {'iid model':>9} "
          f"{'top gear > p/2':>14} {'final at 13':>11}")
    stats = {}
    for b in bands:
        d = per_band[b]
        if not d["n"]:
            continue
        probs = {k: v / d["nb"] for k, v in d["rungs"].items()}
        iid = iid_records_mean(probs, d["Ll"])
        ml_ = float(np.mean(d["ev_l"])); mr_ = float(np.mean(d["ev_r"]))
        stats[b] = (ml_, mr_, iid, d["big"] / d["n"], d["final13"] / d["n"])
        print(f"{b[0]:>5}-{b[1]:<6} {d['n']:>7} {ml_:>13.3f} {mr_:>13.3f} {iid:>9.3f} "
              f"{d['big'] / d['n']:>14.3f} {d['final13'] / d['n']:>11.3f}")

    print("\n=== gates ===")
    gate(tvs[5] < 0.02, f"V1 TV mod 5 = {tvs[5]:.4f} < 0.02")
    gate(tvs[35] < 0.03, f"V1 TV mod 35 = {tvs[35]:.4f} < 0.03")
    gate(tvs[385] < 0.06, f"V1 TV mod 385 = {tvs[385]:.4f} < 0.06")
    gate(tv_f < 0.03, f"V2 framing pair TV(joint, product) = {tv_f:.4f} < 0.03")
    gate(0.60 <= ml[5] / n <= 0.72, f"V2 left framing rung 5 at {ml[5] / n:.3f} in [0.60, 0.72]")
    hi = [stats[b] for b in bands if b[0] >= 300 and b in stats]
    gate(all(2.5 <= s[0] <= 4.5 for s in hi), "V3 mean left events in [2.5, 4.5] at every band q >= 300")
    gate(all(hi[j][0] < hi[j + 1][0] for j in range(len(hi) - 1)), "V3 mean left events increase with q")
    gate(all(abs(s[0] - s[2]) / s[2] <= 0.15 for s in hi), "V3 iid-records model within 15% at every band q >= 300")
    big1000 = [stats[b][3] for b in bands if b[0] >= 1000 and b in stats]
    gate(all(0.05 <= x <= 0.25 for x in big1000), f"V4 top interacting gear > p/2 in [0.05, 0.25] at q >= 1000: {[round(x, 3) for x in big1000]}")
    gate(all(0.05 <= s[4] <= 0.25 for s in hi), f"V4 word final at level 13 in [0.05, 0.25] at q >= 300: {[round(s[4], 3) for s in hi]}")
    print(f"\nGATES: {NGATE - NFAIL} passed, {NFAIL} failed of {NGATE}")


if __name__ == "__main__":
    main()

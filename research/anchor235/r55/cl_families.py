"""r55 cl_families - item 3: the real separation against random and coherent families.

For every pair of gears g < h <= 61, the collision onset L0 (least L >= 2 with c > 0) and the
permanent onset L1 (least L beyond which c > 0 for every larger L; exact, because
c(L + gh) = c(L) + 4 forces c >= 4 above the period, so every zero lies in [1, gh]) are
computed for

  * the REAL separation      s_g = 3^{-1} (mod g)                       (the machine)
  * 20 RANDOM draws          s_g uniform in [1, g-1], independently per gear
  * the COHERENT families    s_g = c r^{-1} (mod g)  for c/r in the W3 list

and the real value's percentile among the random draws is reported, twin pairs apart from
non-twin.  This is the pairwise version of face C's question.
"""
import os
import random

import numpy as np

from cl_core import (RESULTS, PRIMES, arc, coherent_sep, dump, pair_profile, real_sep,
                     say_factory)

LINES = []
say = say_factory(LINES)

GEARS = [p for p in PRIMES if 5 <= p <= 61]
COHERENT = [(1, 3), (1, 5), (2, 5), (2, 7), (4, 7), (2, 11), (3, 11), (2, 13)]
NDRAW = 20
SEED = 20260905


def onsets(g, sg, h, sh):
    P = g * h
    c = pair_profile(g, sg, h, sh, P)["c"]
    nz = np.nonzero(c[2:] > 0)[0]
    L0 = int(nz[0]) + 2 if len(nz) else -1
    zer = np.nonzero(c[1:P + 1] == 0)[0]
    L1 = int(zer[-1]) + 2 if len(zer) else 1
    return L0, L1


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng = random.Random(SEED)
    say("=" * 100)
    say("ITEM 3 - the real one-third separation against random and coherent separations")
    say("=" * 100)
    say(f"  gears {GEARS[0]}..{GEARS[-1]}, {len(GEARS) * (len(GEARS) - 1) // 2} pairs; "
        f"{NDRAW} random draws (seed {SEED}); coherent families "
        f"{['%d/%d' % cr for cr in COHERENT]}")
    say("  L0 = collision onset (least L >= 2 with c > 0); L1 = permanent onset")
    say()
    rows = []
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            sg, sh = real_sep(g), real_sep(h)
            L0r, L1r = onsets(g, sg, h, sh)
            rnd = [onsets(g, rng.randrange(1, g), h, rng.randrange(1, h))
                   for _ in range(NDRAW)]
            r0 = sorted(x[0] for x in rnd)
            r1 = sorted(x[1] for x in rnd)
            coh = {}
            for (cc, rr) in COHERENT:
                if g % rr == 0 or h % rr == 0 or cc % g == 0 or cc % h == 0:
                    continue
                coh[(cc, rr)] = onsets(g, coherent_sep(g, cc, rr), h,
                                       coherent_sep(h, cc, rr))
            pct0 = sum(1 for v in r0 if v < L0r) / NDRAW
            pct1 = sum(1 for v in r1 if v < L1r) / NDRAW
            rows.append(dict(g=g, h=h, P=g * h, twin=(h == g + 2),
                             ag=arc(g, sg), ah=arc(h, sh),
                             L0=L0r, L1=L1r, r0=r0, r1=r1, coh=coh,
                             pct0=pct0, pct1=pct1,
                             sharedarc=(arc(g, sg) == arc(h, sh))))
    # --- twin pairs, in full
    say("  TWIN PAIRS - real onset against the random draws")
    say(f"  {'g':>4} {'h':>4} {'a':>4} {'gh':>6} {'real L0':>8} {'rand L0 min':>12} "
        f"{'median':>8} {'max':>6} {'pctile':>7} | {'real L1':>8} {'rand L1 med':>12} "
        f"{'pctile':>7}")
    for r in rows:
        if r["twin"]:
            say(f"  {r['g']:>4} {r['h']:>4} {r['ag']:>4} {r['P']:>6} {r['L0']:>8} "
                f"{r['r0'][0]:>12} {int(np.median(r['r0'])):>8} {r['r0'][-1]:>6} "
                f"{r['pct0']:>7.2f} | {r['L1']:>8} {int(np.median(r['r1'])):>12} "
                f"{r['pct1']:>7.2f}")
    tw = [r for r in rows if r["twin"]]
    say(f"  twin pairs where the real onset is STRICTLY BELOW every random draw: "
        f"{sum(1 for r in tw if r['L0'] < r['r0'][0])} of {len(tw)}")
    say(f"  twin real onset / random median: "
        f"{[round(r['L0'] / np.median(r['r0']), 3) for r in tw]}")
    say()
    # --- non-twin
    nt = [r for r in rows if not r["twin"]]
    p0 = np.array([r["pct0"] for r in nt])
    p1 = np.array([r["pct1"] for r in nt])
    say(f"  NON-TWIN PAIRS ({len(nt)}): percentile of the real onset among {NDRAW} random draws")
    say(f"    L0 percentile: min {p0.min():.2f} median {np.median(p0):.2f} "
        f"max {p0.max():.2f} mean {p0.mean():.3f}")
    say(f"    L1 percentile: min {p1.min():.2f} median {np.median(p1):.2f} "
        f"max {p1.max():.2f} mean {p1.mean():.3f}")
    say(f"    real strictly below every draw (L0): {sum(1 for r in nt if r['L0'] < r['r0'][0])}"
        f"; strictly above every draw: {sum(1 for r in nt if r['L0'] > r['r0'][-1])}")
    say(f"    ratio real L0 / random-median L0: min "
        f"{min(r['L0'] / np.median(r['r0']) for r in nt):.3f} median "
        f"{np.median([r['L0'] / np.median(r['r0']) for r in nt]):.3f} max "
        f"{max(r['L0'] / np.median(r['r0']) for r in nt):.3f}")
    say(f"    non-twin pairs whose real L1 is below the random median: "
        f"{sum(1 for r in nt if r['L1'] < np.median(r['r1']))} of {len(nt)}; "
        f"above: {sum(1 for r in nt if r['L1'] > np.median(r['r1']))}")
    say(f"    non-twin real L1 / gh: median "
        f"{np.median([r['L1'] / r['P'] for r in nt]):.4f};  random L1 / gh: median "
        f"{np.median([np.median(r['r1']) / r['P'] for r in nt]):.4f}")
    say()
    say("  NORMALISED ONSET  L0 / (max(a_g,a_h) + 1)   (1.000 = the earliest an onset can be;")
    say("  the raw onset is dominated by the arcs, and a random separation may have arc 1,")
    say("  which the real separation never has - a_g is even, file 20 Lemma 1)")
    nr_tw = [r["L0"] / (max(r["ag"], r["ah"]) + 1) for r in tw]
    nr_nt = [r["L0"] / (max(r["ag"], r["ah"]) + 1) for r in nt]
    say(f"    real, twin pairs      : {[round(x, 3) for x in nr_tw]}")
    say(f"    real, non-twin pairs  : median {np.median(nr_nt):.3f} "
        f"min {min(nr_nt):.3f} max {max(nr_nt):.3f}")
    rng3 = random.Random(SEED + 2)
    nr_rand = []
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            for _ in range(5):
                sg, sh = rng3.randrange(1, g), rng3.randrange(1, h)
                L0, _ = onsets(g, sg, h, sh)
                nr_rand.append(L0 / (max(arc(g, sg), arc(h, sh)) + 1))
    nr_rand = np.array(nr_rand)
    say(f"    random ({len(nr_rand)} draws) : median {np.median(nr_rand):.3f} "
        f"mean {nr_rand.mean():.3f} max {nr_rand.max():.3f}; "
        f"fraction at 1.000 = {np.mean(nr_rand == 1):.3f}")
    say()
    # --- shared arc is the mechanism: random draws that happen to share an arc
    say("  IS IT THE SHARED ARC?  onset = max(a_g,a_h)+1 (the earliest possible) against "
        "whether the two separations share a short arc, over ALL pairs and ALL draws:")
    shared_hit = shared_tot = other_hit = other_tot = 0
    rng2 = random.Random(SEED + 1)
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            for _ in range(NDRAW):
                sg, sh = rng2.randrange(1, g), rng2.randrange(1, h)
                a, b = arc(g, sg), arc(h, sh)
                L0, _ = onsets(g, sg, h, sh)
                hit = (L0 == max(a, b) + 1)
                if a == b:
                    shared_tot += 1
                    shared_hit += hit
                else:
                    other_tot += 1
                    other_hit += hit
    say(f"    random draws with a_g = a_h: {shared_hit}/{shared_tot} have the earliest "
        f"possible onset;  with a_g != a_h: {other_hit}/{other_tot}")
    rr = [r for r in rows if r["sharedarc"]]
    say(f"    REAL separations with a_g = a_h: {len(rr)} pairs "
        f"{[(r['g'], r['h']) for r in rr]} - all of them twin pairs: "
        f"{all(r['twin'] for r in rr)}")
    say()
    # --- coherent families
    say("  COHERENT FAMILIES c/r (s_g = c r^{-1} mod g at every gear): onset ratio to real")
    say(f"  {'family':>8} {'pairs':>6} {'L0 med ratio to real':>22} {'L1 med ratio':>14} "
        f"{'shared-arc pairs':>18}")
    for (cc, rrn) in COHERENT:
        v0, v1, sh_ = [], [], 0
        for r in rows:
            if (cc, rrn) not in r["coh"]:
                continue
            a0, a1 = r["coh"][(cc, rrn)]
            v0.append(a0 / r["L0"])
            v1.append(a1 / r["L1"])
            g, h = r["g"], r["h"]
            if arc(g, coherent_sep(g, cc, rrn)) == arc(h, coherent_sep(h, cc, rrn)):
                sh_ += 1
        say(f"  {('%d/%d' % (cc, rrn)):>8} {len(v0):>6} {np.median(v0):>22.3f} "
            f"{np.median(v1):>14.3f} {sh_:>18}")
    say()
    say("  (family 1/3 is the real machine, so its ratios are 1.000 by construction)")
    dump(LINES, "cl_families.txt")


if __name__ == "__main__":
    main()

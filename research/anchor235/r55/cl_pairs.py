"""r55 cl_pairs - items 1 and 2: reproduce the head collision, then every pair to 97.

Item 1: c(5,7;L), c_max, c_dis at the head-collision lengths and around them.
Item 2: for every pair of gears g < h <= 97 and every L from 2 up to min(2gh, 5000):
        the collision onset (least L with c > 0), the exact growth law, the slope.
"""
import os

import numpy as np

from cl_core import (RESULTS, PRIMES, arc, dump, is_prime, maxstrike, pair_profile,
                     real_sep, say_factory)

LINES = []
say = say_factory(LINES)

GEARS = [p for p in PRIMES if 5 <= p <= 97]


def item1():
    say("=" * 96)
    say("ITEM 1 - the head collision reproduced, and the three deficits kept apart")
    say("=" * 96)
    g, h = 5, 7
    sg, sh = real_sep(g), real_sep(h)
    say(f"  gear 5: s = {sg}, arc = {arc(g, sg)};  gear 7: s = {sh}, arc = {arc(h, sh)}"
        f"  (twin pair, shared arc 2);  period gh = 35")
    pr = pair_profile(g, sg, h, sh, 80, want_split=True)
    say()
    say(f"  {'L':>4} {'max5':>5} {'max7':>5} {'sum':>5} {'joint':>6} {'c':>4} "
        f"{'c_max':>6} {'c_dis':>6}")
    for L in list(range(2, 41)) + [44, 45, 50, 58, 63, 70, 80]:
        say(f"  {L:>4} {pr['maxg'][L]:>5} {pr['maxh'][L]:>5} "
            f"{pr['maxg'][L] + pr['maxh'][L]:>5} {pr['joint'][L]:>6} {pr['c'][L]:>4} "
            f"{pr['cmax'][L]:>6} {pr['cdis'][L]:>6}   (-1 = no such phase pair)")
    say()
    hc = {L: int(pr['c'][L]) for L in (16, 22, 28)}
    say(f"  HEAD COLLISION check: c(5,7;16),c(5,7;22),c(5,7;28) = "
        f"{hc[16]}, {hc[22]}, {hc[28]}  (file 20 says 1, 1, 2) -> "
        f"{'REPRODUCED' if (hc[16], hc[22], hc[28]) == (1, 1, 2) else 'MISMATCH'}")
    say(f"  at those L, c_max = {int(pr['cmax'][16])}, {int(pr['cmax'][22])}, "
        f"{int(pr['cmax'][28])} and c_dis = {int(pr['cdis'][16])}, {int(pr['cdis'][22])}, "
        f"{int(pr['cdis'][28])}")
    onset = next(L for L in range(2, 81) if pr['c'][L] > 0)
    say(f"  ONSET L0(5,7) = {onset} (least L >= 2 with c > 0); c is "
        f"{[int(pr['c'][L]) for L in range(2, 20)]} at L = 2..19")
    # periodicity
    bad = [L for L in range(1, 46) if pr['c'][L + 35] != pr['c'][L] + 4]
    say(f"  growth law c(L+35) = c(L)+4 over L = 1..45: {len(bad)} exceptions {bad}")
    return pr


def item2():
    say()
    say("=" * 96)
    say("ITEM 2 - every pair of gears g < h <= 97: onset, growth law, slope")
    say("=" * 96)
    rows = []
    for i, g in enumerate(GEARS):
        for h in GEARS[i + 1:]:
            P = g * h
            Lmax = min(2 * P + 4, 5000)
            sg, sh = real_sep(g), real_sep(h)
            c = pair_profile(g, sg, h, sh, Lmax)["c"]
            nz = np.nonzero(c[2:] > 0)[0]
            onset = int(nz[0]) + 2 if len(nz) else -1
            # the PERMANENT onset: c(L+gh) = c(L)+4 makes c > 0 automatic for L > gh,
            # so every zero of c lies in [1, gh] and L1 = 1 + (last zero) is exact.
            zer = np.nonzero(c[1:P + 1] == 0)[0]
            L1 = int(zer[-1]) + 2 if len(zer) else 1
            # exact growth law, checked wherever two periods fit
            if Lmax >= P + 1:
                hi = Lmax - P
                exc = int(np.count_nonzero(c[1:hi + 1] + 4 != c[1 + P:hi + P + 1]))
                checks = hi
            else:
                exc, checks = -1, 0
            cP = int(c[P]) if P <= Lmax else -1
            rows.append(dict(g=g, h=h, P=P, onset=onset, L1=L1, exc=exc, checks=checks,
                             cP=cP, twin=(h == g + 2), Lmax=Lmax,
                             ag=arc(g, sg), ah=arc(h, sh),
                             c100=int(c[100]) if Lmax >= 100 else -1,
                             c1000=int(c[1000]) if Lmax >= 1000 else -1))
    tot_checks = sum(r["checks"] for r in rows if r["exc"] >= 0)
    tot_exc = sum(r["exc"] for r in rows if r["exc"] >= 0)
    npairs_checked = sum(1 for r in rows if r["exc"] >= 0)
    say(f"  pairs: {len(rows)};  growth law c(L+gh) = c(L)+4 checked on {npairs_checked} pairs, "
        f"{tot_checks} instances, EXCEPTIONS = {tot_exc}")
    say(f"  and c(gh) = 4 exactly on every pair where the period fits: "
        f"{sum(1 for r in rows if r['cP'] == 4)} of "
        f"{sum(1 for r in rows if r['cP'] >= 0)}")
    say()
    say("  TWIN PAIRS (shared short arc a = (g+1)/3):")
    say(f"  {'g':>4} {'h':>4} {'a':>4} {'gh':>7} {'onset':>6} {'a+1':>5} {'L1':>6} "
        f"{'onset/g':>8} {'onset/gh':>9} {'c(100)':>7} {'c(1000)':>8}")
    for r in rows:
        if r["twin"]:
            say(f"  {r['g']:>4} {r['h']:>4} {r['ag']:>4} {r['P']:>7} {r['onset']:>6} "
                f"{r['ag'] + 1:>5} {r['L1']:>6} "
                f"{r['onset'] / r['g']:>8.3f} {r['onset'] / r['P']:>9.4f} "
                f"{r['c100']:>7} {r['c1000']:>8}")
    say(f"  ONSET = a+1 on twin pairs: "
        f"{sum(1 for r in rows if r['twin'] and r['onset'] == r['ag'] + 1)} of "
        f"{sum(1 for r in rows if r['twin'])}")
    say(f"  ONSET = max(a_g,a_h)+1 over ALL pairs: "
        f"{sum(1 for r in rows if r['onset'] == max(r['ag'], r['ah']) + 1)} of {len(rows)};"
        f"  onset >= max(a_g,a_h)+1: "
        f"{sum(1 for r in rows if r['onset'] >= max(r['ag'], r['ah']) + 1)} of {len(rows)}")
    say(f"  PERMANENT onset L1 (c > 0 for every L >= L1; exact, since c(L+gh)=c(L)+4 makes "
        f"c >= 4 above gh): L1/gh min {min(r['L1'] / r['P'] for r in rows):.4f} "
        f"median {np.median([r['L1'] / r['P'] for r in rows]):.4f} "
        f"max {max(r['L1'] / r['P'] for r in rows):.4f}")
    say(f"  twins: L1 = {[r['L1'] for r in rows if r['twin']]}, L1/gh = "
        f"{[round(r['L1'] / r['P'], 4) for r in rows if r['twin']]}")
    say()
    tw = [r for r in rows if r["twin"]]
    nt = [r for r in rows if not r["twin"]]
    say(f"  twin onsets: {[r['onset'] for r in tw]}")
    say(f"  twin onset/g: min {min(r['onset'] / r['g'] for r in tw):.3f} "
        f"max {max(r['onset'] / r['g'] for r in tw):.3f}")
    say(f"  twin onset/gh: min {min(r['onset'] / r['P'] for r in tw):.4f} "
        f"max {max(r['onset'] / r['P'] for r in tw):.4f}")
    on_nt = np.array([r["onset"] / r["P"] for r in nt])
    say(f"  non-twin onset/gh over {len(nt)} pairs: min {on_nt.min():.4f} "
        f"median {np.median(on_nt):.4f} max {on_nt.max():.4f}")
    on_nt_g = np.array([r["onset"] / r["g"] for r in nt])
    say(f"  non-twin onset/g:  min {on_nt_g.min():.3f} median {np.median(on_nt_g):.3f} "
        f"max {on_nt_g.max():.3f}")
    say(f"  twin pairs with onset < g: {sum(1 for r in tw if r['onset'] < r['g'])} of {len(tw)}")
    say(f"  non-twin pairs with onset < g: "
        f"{sum(1 for r in nt if r['onset'] < r['g'])} of {len(nt)}")
    say()
    say("  the smallest onsets in the whole family (top 25 by onset):")
    rows_s = sorted(rows, key=lambda r: r["onset"])
    say(f"  {'g':>4} {'h':>4} {'a_g':>5} {'a_h':>5} {'gh':>7} {'onset':>6} {'twin':>5}")
    for r in rows_s[:25]:
        say(f"  {r['g']:>4} {r['h']:>4} {r['ag']:>5} {r['ah']:>5} {r['P']:>7} "
            f"{r['onset']:>6} {str(r['twin']):>5}")
    say()
    say("  the largest onsets (bottom 10):")
    for r in rows_s[-10:]:
        say(f"  {r['g']:>4} {r['h']:>4} {r['ag']:>5} {r['ah']:>5} {r['P']:>7} "
            f"{r['onset']:>6} {str(r['twin']):>5}")
    say()
    # arc-difference view: does |a_g - a_h| decide the onset?
    say("  onset against the arc difference |a_g - a_h| (non-twin pairs pooled):")
    from collections import defaultdict
    byd = defaultdict(list)
    for r in rows:
        byd[abs(r["ag"] - r["ah"])].append(r["onset"] / r["P"])
    say(f"  {'|a_g-a_h|':>10} {'pairs':>6} {'min onset/gh':>13} {'median':>9} {'max':>9}")
    for d in sorted(byd)[:14]:
        v = np.array(byd[d])
        say(f"  {d:>10} {len(v):>6} {v.min():>13.4f} {np.median(v):>9.4f} {v.max():>9.4f}")
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    item1()
    rows = item2()
    dump(LINES, "cl_pairs.txt")
    np.save(os.path.join(RESULTS, "pairs_onset.npy"),
            np.array([(r["g"], r["h"], r["onset"]) for r in rows]))


if __name__ == "__main__":
    main()

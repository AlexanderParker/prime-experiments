"""Second battery of cross-part measurements on the new ranges.

(e) twin density in the range (twins per column) against the prime gap d = g' - g that sets
    the range's length: mean density per gap class d = 2, 4, 6, 8..12, 14..20, 22+;
(f) per-gear strike counts in the range against the expectation 2L/h, pooled over ranges, for
    the gears 5..47 and for the top gear g itself: ratio observed/expected;
(g) autocorrelation of the twin counts of consecutive ranges (density residuals);
(h) what kills the neighbours of a twin: for each twin at offset i, the least factor of the
    killing member at i - 1 and i + 1, tallied by gear, against the same tally for all killed
    columns: does a twin's neighbour die to a different gear profile?
Usage: uv run python cross_measures2.py gmin gmax
"""
import sys
from collections import defaultdict, Counter
import numpy as np
from sympy import primerange, nextprime, isprime, factorint


def main():
    gmin, gmax = int(sys.argv[1]), int(sys.argv[2])
    dens_by_gap = defaultdict(list); resid = []
    gear_obs = Counter(); gear_exp = defaultdict(float); top_obs = 0; top_exp = 0.0
    nb = Counter(); allk = Counter()
    for g in primerange(gmin, gmax + 1):
        gp = nextprime(g); a = (g * g - 1) // 6; L = (gp * gp - g * g) // 6; d = gp - g
        killer = {}
        tw = []
        for i in range(1, L + 1):
            l, r = 6 * (a + i) - 1, 6 * (a + i) + 1
            pl, pr = isprime(l), isprime(r)
            if pl and pr: tw.append(i); continue
            hs = []
            for n, pp in ((l, pl), (r, pr)):
                if not pp:
                    h = min(factorint(n)); hs.append(h); allk[h] += 1
                    if h <= 47: gear_obs[h] += 1
                    if h == g: top_obs += 1
            killer[i] = min(hs)
        for h in primerange(5, min(47, g) + 1): gear_exp[h] += 2 * L / h
        top_exp += 2 * L / g
        dens = len(tw) / L
        gc = d if d <= 6 else (8 if d <= 12 else (14 if d <= 20 else 22))
        dens_by_gap[gc].append(dens)
        # expected density from the section's size: 6 * 1.3203 * 0.88.. use the empirical overall later
        resid.append((g, dens, L))
        for i in tw:
            for j in (i - 1, i + 1):
                if j in killer: nb[killer[j]] += 1
    print(f"primes g in [{gmin}, {gmax}]: {len(resid)} ranges")
    print("(e) twin density by the gap d = g' - g:", {k: f"n={len(v)}, mean {100*np.mean(v):.2f}% (se {100*np.std(v)/np.sqrt(len(v)):.2f})" for k, v in sorted(dens_by_gap.items())})
    print("(f) per-gear strikes in the range, observed / 2L/h:", {h: f"{gear_obs[h]/gear_exp[h]:.3f}" for h in sorted(gear_exp) if gear_exp[h] > 0})
    print(f"    the top gear g itself: observed {top_obs} against 2L/g summed {top_exp:.1f}, ratio {top_obs/top_exp:.3f}")
    # (g) autocorrelation of density residuals (against the running mean of neighbours)
    dens = np.array([x[1] for x in resid]); Ls = np.array([x[2] for x in resid])
    r = dens - dens.mean(); ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]; ac2 = np.corrcoef(r[:-2], r[2:])[0, 1]
    print(f"(g) autocorrelation of the twin density across consecutive ranges: lag 1 {ac1:+.3f}, lag 2 {ac2:+.3f} (n = {len(dens)})")
    tot_nb = sum(nb.values()); tot_all = sum(allk.values())
    print("(h) least factor of the member killing a twin's neighbour column (share) against all killed members (share):",
          {h: f"{100*nb[h]/tot_nb:.1f}% vs {100*allk[h]/tot_all:.1f}%" for h in (5, 7, 11, 13, 17, 19, 23)})


if __name__ == "__main__":
    main()

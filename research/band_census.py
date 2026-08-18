"""Layer-band twin census: the T1 reopening interrogation (mechanic r10).

EVENT DEFINITION (deliverable 1). Bands B_i = (p_i^2, p_{i+1}^2) between
squares of consecutive primes >= 5 tile the number line; in slot space,
band i = slots k with kb_i < k <= kb_{i+1}, kb = (p^2-1)/6 (exact: p^2 = 1
mod 6). Exact thickness T_i = (p'^2 - p^2)/6 = g(2p+g)/6, g = the prime
gap. So "thinnest band at a height <=> gap-2 (twin) endpoints" is EXACT but
TRIVIAL: T is monotone in g. The non-trivial machine event: for a twin
(p, p+2) = (6m-1, 6m+1), T = 4m and the twin's PRODUCT SLOT k = 6m^2 sits
at offset exactly 2m = T/2 - the band's center slot - with member
36m^2 - 1 = p(p+2) composite BY the twin itself. Every twin pre-blocks the
center of the thinnest band above it (algebra, not measurement; verified
here anyway per band). The census decides: are twin-endpoint bands
twin-poor beyond that one guaranteed dead slot (exact law vs density)?

Counted per band (exact, every slot): T, twin slots (both members prime),
prime members. Aggregates: per gap class g (bands/slots/twins/density/
empty bands/first empty), per height decade the g=2 density vs all-band
density raw AND center-excluded, min primes per band (T1 side), and for
every g=2 band the center-slot check + whether the center's partner
36m^2+1 is prime (fragile center).

Output: research/data/band_census_<P>.csv (p,g,T,twins,primes) + printed
aggregates. Usage: uv run python research/band_census.py [P]  (default 100003)
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime


def census(P, seg=16_000_000):
    ps_all = primes_upto(P + 300)
    ps = [p for p in ps_all if p >= 5]
    while ps[-1] > P and ps[-2] > P:
        ps.pop()
    # ps: primes 5..first prime > P (band boundaries)
    nb = len(ps) - 1
    kb = np.array([(p * p - 1) // 6 for p in ps], dtype=np.int64)
    K = int(kb[-1])
    sieve_ps = ps_all  # primality: all primes <= sqrt(6K+1) <= ps[-1]
    sieve_ps = [q for q in sieve_ps if q >= 5]
    uvals = [pow(6, -1, q) for q in sieve_ps]
    twins = np.zeros(nb, dtype=np.int64)
    primes_in = np.zeros(nb, dtype=np.int64)
    t0 = time.time()
    for a in range(4, K + 1, seg):
        b = min(K + 1, a + seg)
        n = b - a
        exL = np.zeros(n, bool)
        exR = np.zeros(n, bool)
        for q, u in zip(sieve_ps, uvals):
            exL[(u - a) % q::q] = True
            exR[(-u - a) % q::q] = True
        if a == 4:  # own-value fix: member == q is prime
            for q in sieve_ps:
                if (q + 1) % 6 == 0 and (q + 1) // 6 >= a:
                    exL[(q + 1) // 6 - a] = False
                if (q - 1) % 6 == 0 and (q - 1) // 6 >= a:
                    exR[(q - 1) // 6 - a] = False
        pl, pr = ~exL, ~exR
        kk = np.arange(a, b, dtype=np.int64)
        idx = np.searchsorted(kb, kk, side="left") - 1
        ok = (idx >= 0) & (idx < nb)
        tw = pl & pr & ok
        twins += np.bincount(idx[tw], minlength=nb)[:nb]
        pcnt = pl.astype(np.int64) + pr
        primes_in += np.bincount(idx[ok], weights=pcnt[ok],
                                 minlength=nb)[:nb].astype(np.int64)
    print(f"scan K={K} ({nb} bands, P~{P}): {time.time()-t0:.0f}s")
    return ps, kb, twins, primes_in


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 100003
    ps, kb, twins, primes_in = census(P)
    nb = len(ps) - 1
    parr = np.array(ps[:-1], dtype=np.int64)
    gaps = np.diff(np.array(ps, dtype=np.int64))
    T = np.diff(kb)
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, f"band_census_{P}.csv")
    with open(path, "w") as f:
        f.write("p,g,T,twins,primes\n")
        for i in range(nb):
            f.write(f"{parr[i]},{gaps[i]},{T[i]},{twins[i]},{primes_in[i]}\n")
    print(f"wrote {path}")

    print("\nper gap class (exact):")
    print(f"{'g':>4} {'bands':>7} {'slots':>12} {'twins':>9} {'tw/slot':>8} "
          f"{'empty':>6} {'first empty p':>13} {'minPrimes':>9}")
    for g in sorted(set(gaps.tolist())):
        m = gaps == g
        sl = int(T[m].sum())
        tw = int(twins[m].sum())
        emp = m & (twins == 0)
        fe = int(parr[emp][0]) if emp.any() else 0
        print(f"{g:>4} {int(m.sum()):>7} {sl:>12} {tw:>9} "
              f"{tw/sl if sl else 0:>8.4f} {int(emp.sum()):>6} "
              f"{fe if fe else '-':>13} {int(primes_in[m].min()):>9}")

    print("\ng=2 bands vs all bands, per height decade "
          "(density per slot; g2x = center slot excluded):")
    dec = np.floor(np.log10(parr.astype(float) ** 2)).astype(int)
    print(f"{'dec':>4} {'g2 bands':>8} {'g2 tw/slot':>10} {'g2x tw/slot':>11} "
          f"{'all tw/slot':>11} {'ratio':>6} {'ratio_x':>7}")
    for d in sorted(set(dec.tolist())):
        md = dec == d
        m2 = md & (gaps == 2)
        if not m2.any():
            continue
        sl2, tw2 = int(T[m2].sum()), int(twins[m2].sum())
        sla, twa = int(T[md].sum()), int(twins[md].sum())
        d2 = tw2 / sl2
        d2x = tw2 / (sl2 - int(m2.sum()))  # center slots are dead by law
        da = twa / sla
        print(f"{d:>4} {int(m2.sum()):>8} {d2:>10.4f} {d2x:>11.4f} "
              f"{da:>11.4f} {d2/da:>6.3f} {d2x/da:>7.3f}")

    # center-slot law verification + fragile-center stat (g=2 bands)
    m2 = gaps == 2
    checked = frag = 0
    for p in parr[m2].tolist():
        mm = (p + 1) // 6
        c = 36 * mm * mm
        assert (c - 1) == p * (p + 2)          # product = center L member
        assert not is_prime(c - 1)             # composite by the twin
        checked += 1
        if is_prime(c + 1):
            frag += 1
    print(f"\ncenter-slot law verified on {checked}/{int(m2.sum())} "
          f"g=2 bands (product = L member of k=6m^2, composite). "
          f"Fragile centers (36m^2+1 prime): {frag} "
          f"({100*frag/checked:.1f}%)")
    tw2_tot = int(twins[m2].sum())
    emp2 = int((m2 & (twins == 0)).sum())
    print(f"g=2 bands: {int(m2.sum())}, twins inside {tw2_tot}, "
          f"twin-empty {emp2}, min twins {int(twins[m2].min())}, "
          f"min primes {int(primes_in[m2].min())}")


if __name__ == "__main__":
    main()

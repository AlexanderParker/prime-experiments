"""Recount the reflection n -> M - 2 - n on two chain sections, independently of generated.py.

generated.py reports "mirror_hits / mirror_chance" with chance = |set|^2 / (numbers coprime to 30 in
the section).  Two things that model leaves out are checked here:
  (i) only 3 of the 8 classes mod 30 can have an image coprime to 30 (n = 11, 17, 29 mod 30, since
      M = 0 mod 30 forces the image into -2 - n), so the ceiling on ordered hits is 3/8 of chance;
  (ii) generated.py counts a same-segment pair twice and a cross-segment pair once, so its ratio
      halves on the multi-segment sections.

Usage: uv run python research/stack/r6/mirror_check.py
"""
import numpy as np


def primes_in(lo, hi):
    s = np.ones(hi, dtype=bool)
    s[:2] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    k = np.flatnonzero(s).astype(np.int64)
    return k[(k >= lo) & (k < hi)]


def main():
    for lo, hi in ((2809, 7946761), (16129, 260467321)):
        m = primes_in(lo, hi)
        M = 30 * round((lo + hi) / 30)
        img = M - 2 - m
        inside = (img >= lo) & (img < hi)
        member = np.zeros(hi - lo, dtype=bool)
        member[m - lo] = True
        hits = int(member[img[inside] - lo].sum())
        cop30 = int(((np.arange(lo, hi) % 2 == 1) & (np.arange(lo, hi) % 3 != 0) & (np.arange(lo, hi) % 5 != 0)).sum()) \
            if hi - lo < 3 * 10 ** 7 else None
        if cop30 is None:                       # count coprime-to-30 without a full index array
            cop30 = 0
            step = 10 ** 7
            for a in range(lo, hi, step):
                b = min(a + step, hi)
                r = np.arange(a, b) % 30
                cop30 += int(((r % 2 == 1) & (r % 3 != 0) & (r % 5 != 0)).sum())
        chance = len(m) ** 2 / cop30
        elig = np.isin(m % 30, [11, 17, 29])
        print(f"section [{lo:,}, {hi:,}): M = {M:,}")
        print(f"  members {len(m):,}; coprime to 30 {cop30:,}; chance (ordered) {chance:,.1f}")
        print(f"  members in the 3 classes that can have a coprime image: {int(elig.sum()):,} "
              f"= {elig.mean():.4f} of members (3/8 = 0.375)")
        print(f"  ordered mirror hits, recounted: {hits:,} = {hits / chance:.4f} of chance "
              f"= {hits / (0.375 * chance):.4f} of the 3/8 ceiling")
        # the Hardy-Littlewood Goldbach prediction for the ordered count, as a ratio to chance:
        #   2 C_2 x prod_{p | M-2, p odd} (p-1)/(p-2) x (8/30)
        n = M - 2
        fac = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                fac.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            fac.append(n)
        prod = 1.0
        for p_ in fac:
            if p_ > 2:
                prod *= (p_ - 1) / (p_ - 2)
        pred = 1.3203236 * (8 / 30) * prod
        print(f"  M - 2 = {M - 2:,} = {' x '.join(str(f) for f in fac)}; "
              f"Hardy-Littlewood ratio 2C_2 (8/30) prod (p-1)/(p-2) = {pred:.4f} against measured {hits / chance:.4f}")


if __name__ == "__main__":
    main()

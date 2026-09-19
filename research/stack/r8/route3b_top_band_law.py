"""ROUTE 3, second pass (post hoc, after the first claim failed): the exact top-band law.

Route 3's pre-registered claim ('one strike per near gear, at the k = 0 column') FAILED: gears
p = s - e with e = 5 mod 6 always strike a second column.  The corrected exact statement, tested
here on the same range, is:

  A top gear p = s - e (2s/3 < p <= s+1) strikes the member (s+k)^2 - (e+k)^2 = p*(s + e + 2k),
  k >= 0, exactly when that member lies in ((s-1)^2, (s+1)^2) and
        k = 0 mod 3  ->  lower member,      k = -e mod 3  ->  upper member
  (k = e mod 3 never).  Every top gear strikes 1 or 2 columns, never 0 (the m-interval
  ((s-1)^2/p, (s+1)^2/p) has >= 4 integers and 4 consecutive residues mod 6 always contain 1 or 5).
  NEAR BAND (|e|+2)^2 <= 2s + 3: only k = 0 (the difference-of-squares column, lower member) and k = 1
  (the second family j = 2c - (e+1)^2/6, upper member, present iff e = 5 mod 6) - both columns are
  dead at every twin centre by identity, so the near band is idle on every live column.

CLAIM: 0 exceptions to the k-rule over all top-band strikes, all twin centres s <= 10^5; near-band
strikes have k in {0, 1} only; strike count per top gear is 1 or 2 with 0 zeros.
"""
import sys, time, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from lib_twins import twin_centres_upto, primes_upto

S_MAX = 100_000
t0 = time.time()
tw = twin_centres_upto(S_MAX)
ps = primes_upto(S_MAX + 2)
k_exc = 0; near_k_exc = 0; zero_gears = 0; top_gears = 0
near_one = near_two = 0
kmax_seen = 0
k_hist = {}
for s in tw.tolist():
    c = s // 6; W = 2 * c - 1
    top = ps[(ps > (2 * s) // 3) & (ps <= s + 1)]
    lo2, hi2 = (s - 1) ** 2, (s + 1) ** 2
    for p in top.tolist():
        inv6 = pow(6, -1, p); e = s - p
        strikes = []
        for sign in (-1, +1):
            a = (-(s * s + sign) * inv6) % p
            for r in (a, a - p):
                if -W <= r <= W:
                    strikes.append((r, sign))
        top_gears += 1
        if not strikes:
            zero_gears += 1
        # predicted strike set from the k-rule
        pred = []
        base = s * s - e * e                        # x_k = base + 2*k*p, increasing in k
        k_lo = max(0, -((base - lo2) // (2 * p)))   # smallest k with x_k > lo2
        k_hi = (hi2 - 1 - base) // (2 * p)          # largest k with x_k < hi2
        for k in range(k_lo, k_hi + 1):
            x = base + 2 * k * p
            if not (lo2 < x < hi2):
                continue
            if k % 3 == 0:
                j = (x + 1 - s * s) // 6; sg = -1
            elif (k + e) % 3 == 0:
                j = (x - 1 - s * s) // 6; sg = +1
            else:
                continue
            if -W <= j <= W:                        # the window excludes the columns +-2c
                pred.append((j, sg))
        if sorted(pred) != sorted(strikes):
            k_exc += 1
            if k_exc <= 5:
                print(f"  k-rule exception: s={s} p={p} e={e} strikes={strikes} predicted={pred}")
        for (r, sign) in strikes:
            x = s * s + 6 * r + sign
            kk = (x // p - s - e) // 2
            k_hist[kk] = k_hist.get(kk, 0) + 1
            kmax_seen = max(kmax_seen, kk)
            if (abs(e) + 2) ** 2 <= 2 * s + 3 and kk not in (0, 1):
                near_k_exc += 1
        if e != 0 and (abs(e) + 2) ** 2 <= 2 * s + 3:
            if len(strikes) == 1: near_one += 1
            elif len(strikes) == 2: near_two += 1
print("\n=== RESULT TABLE: the top-band k-rule, all twin centres s <= 10^5 ===")
print(f"{'top-band gears':48s} {top_gears:>12,d}")
print(f"{'gears with zero strikes':48s} {zero_gears:>12,d}")
print(f"{'k-rule exceptions (strike set != predicted set)':48s} {k_exc:>12,d}")
print(f"{'near-band gears with 1 / 2 strikes':48s} {near_one:>12,d} / {near_two:,}")
print(f"{'near-band strikes with k outside {0,1}':48s} {near_k_exc:>12,d}")
print(f"{'largest k seen in the top band':48s} {kmax_seen:>12d}")
print("k histogram (first 8): " + ", ".join(f"k={k}:{k_hist.get(k,0):,}" for k in range(8)))
verdict = "HOLDS" if (k_exc == 0 and zero_gears == 0 and near_k_exc == 0) else "FAILS"
print(f"\nVERDICT (corrected law): {verdict}   ({time.time()-t0:.1f}s)")

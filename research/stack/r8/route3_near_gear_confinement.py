"""ROUTE 3 - near-gear confinement: the level's own primes are idle on the live columns.

In the stretch of a twin centre s the gears run to s+1.  A gear p = s - e near the top (|e| small)
has s = e mod p, so its two closed classes are j = -(e^2 - 1)/6 and j = -(e^2 + 1)/6 mod p.  Because
the window (|j| <= 2c-1, i.e. |6j| < 2s) is shorter than p, each class has at most one representative
inside, and the arithmetic of e (e = +-1 mod 6) decides which:
   e^2 < 2s - 5 :  exactly ONE strike, the lower member at j = (1 - e^2)/6, which is (s-e)(s+e)
                   - a difference-of-squares column, dead by identity.  (e = +-1 is the kernel's
                   twin_gear_strikes_centre_only; a twin centre q = s - 6D near s contributes
                   e = 6D +- 1, so siblings and neighbours in the forest are mutually idle.)
   larger |e|   :  every strike is the member p*(s + e + delta) with delta in (e^2/s - 2, e^2/s + 2)
                   and delta = 0 mod 6 (lower member) or delta = -2/e mod 6 (upper member): at most
                   two strikes, each an identity (s-e)(s+e+delta), nothing else.
So the top band's strikes are completely enumerated by e alone, and every prime within sqrt(2s) of
s covers nothing that is not already dead.  A leaf (T(s) = 0) must therefore be covered by gears
with |s - p| > sqrt(2s) only; and along the tree the prime members of the parent's stretch near
the rung become exactly these idle gears of the child.

CLAIM (with a number): for every twin centre s <= 10^5 and every prime p with 0 < |s-p| = |e| and
e^2 <= 2s - 5, gear p strikes exactly one column of the stretch, j = (1 - e^2)/6, lower member -
0 exceptions over all 1,223 twin centres; and every strike of every gear p > 2s/3 is of the form
p*(s + e + delta) with delta in {0, 2, 4, 6, 8, ...}, |delta - e^2/s| < 2, mod-6 rule as stated -
0 exceptions.  Refuted by one gear in the near band with 0 or 2 strikes or a strike off the
difference-of-squares column, or one top-band strike outside the delta rule.
"""
import sys, time, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from lib_twins import twin_centres_upto, primes_upto

S_MAX = 100_000
t0 = time.time()
tw = twin_centres_upto(S_MAX)
ps = primes_upto(S_MAX + 2)
print(f"twin centres to {S_MAX}: {len(tw)}; primes to {S_MAX+2}: {len(ps)}")

near_exc = 0; near_gears = 0; near_strikes_total = 0
delta_exc = 0; top_gears = 0; top_strikes = 0
hist_strikes = {0: 0, 1: 0, 2: 0}
delta_hist = {}
band_zero = 0   # gears in (sqrt(2s), 2 sqrt s) with zero strikes
band_cnt = 0
worst_e_ratio = 0.0
for s in tw.tolist():
    c = s // 6
    W = 2 * c - 1
    top = ps[(ps > (2 * s) // 3) & (ps <= s + 1)]
    for p in top.tolist():
        inv6 = pow(6, -1, p)
        e = s - p
        n = 0
        cols = []
        for sign, member in ((-1, "lower"), (+1, "upper")):   # member = s^2 + 6j + sign
            a = (-(s * s + sign) * inv6) % p        # closed class of j
            # representatives of a mod p inside [-W, W]
            for r in (a, a - p):
                if -W <= r <= W:
                    n += 1
                    cols.append((r, sign))
                    x = s * s + 6 * r + sign
                    assert x % p == 0
                    mth = x // p
                    delta = mth - (s + e)
                    delta_hist[(delta, sign, e % 6)] = delta_hist.get((delta, sign, e % 6), 0) + 1
                    ok = (delta >= 0 and delta % 2 == 0 and abs(delta - e * e / s) < 2 and
                          ((sign == -1 and delta % 6 == 0) or (sign == +1 and (delta * e + 2) % 6 == 0)))
                    if not ok:
                        delta_exc += 1
                        if delta_exc <= 5:
                            print(f"  delta exception: s={s} p={p} e={e} j={r} sign={sign} delta={delta}")
        top_gears += 1; top_strikes += n
        hist_strikes[n] += 1
        if e != 0 and e * e <= 2 * s - 5:
            near_gears += 1; near_strikes_total += n
            expect = ((1 - e * e) // 6, -1)
            if n != 1 or cols[0] != expect:
                near_exc += 1
                if near_exc <= 5:
                    print(f"  near exception: s={s} p={p} e={e} strikes={cols} expected={expect}")
        if 2 * s - 5 < e * e < 4 * s:
            band_cnt += 1
            if n == 0:
                band_zero += 1

print("\n=== RESULT TABLE: top band p in (2s/3, s+1], all twin centres s <= 10^5 ===")
print(f"{'top-band gears examined':50s} {top_gears:>12,d}")
print(f"{'strikes per top gear: 0 / 1 / 2':50s} {hist_strikes[0]:>12,d} / {hist_strikes[1]:,} / {hist_strikes[2]:,}")
print(f"{'near-band gears (e^2 <= 2s-5)':50s} {near_gears:>12,d}")
print(f"{'near-band exceptions (not exactly the DoS column)':50s} {near_exc:>12,d}")
print(f"{'band sqrt(2s) < |e| < 2 sqrt(s): gears / zero-strike':50s} {band_cnt:>12,d} / {band_zero:,}")
print(f"{'delta-rule exceptions over all top-band strikes':50s} {delta_exc:>12,d}")
tops = sorted(delta_hist.items(), key=lambda kv: -kv[1])[:12]
print("most common (delta, member sign, e mod 6) among top-band strikes:")
for (d, sg, em), cnt in tops:
    print(f"   delta={d:>3d} member={'upper' if sg > 0 else 'lower'} e mod 6={em}  count={cnt:,}")
# idle fraction: near-band gears as a share of the top band, and of all gears
s_last = int(tw[-1])
near_cnt = int(np.sum(np.abs(ps - s_last) ** 2 <= 2 * s_last - 5)) - 0
print(f"at s = {s_last}: near band holds {near_cnt} primes of {int(np.sum(ps <= s_last+1))} gears "
      f"(2 sqrt(2s)/ln s = {2*math.sqrt(2*s_last)/math.log(s_last):.1f})")
verdict = "HOLDS" if (near_exc == 0 and delta_exc == 0) else "FAILS"
print(f"\nVERDICT: {verdict}   ({time.time()-t0:.1f}s)")

"""ROUTE 1 - the consecutive-product window (brief hint (c)).

Two consecutive twin centres s < t, t = s + 6d.  Their square stretches are DISJOINT
((t-1)^2 - (s+1)^2 = (t-s-2)(t+s) > 0), so the literal 'overlapping stretches' reading is empty.
The window where their phases meet is the PRODUCT window
        W(s,t) = ((s-1)(t-1), (s+1)(t+1)),  members st + 6j -+ 1, |j| <= c_s + c_t - 1,
which sits strictly between the two square stretches, has length 2(s+t) ~ 4t (square strength,
theta = 1/2) and is NOT dead at its centre (st -+ 1 has no identity factorisation, unlike s^2 - 1).
Gear g strikes offset j iff st + 6j = +-1 mod g: the phase is the 'geometric mean' -(st -+ 1)/6 of
the two square phases.  A twin centre in W(s,t) is a product-rung of the pair; the chain
t -> rung of (pred(t), t) climbs like squaring.

CLAIM (falsifiable, with a number): for every consecutive pair of twin centres 30 <= s < t with
t <= 30,000, W(s,t) holds at least one twin centre; the count N(s,t) follows the generic law
2(s+t) * 1.3203 / ln(st)^2 with mean ratio 1.00 +- 0.03 and the centre column st -+ 1 is a twin
at the generic rate (no identity), so a ladder may step from t into EITHER (t,t) or (pred(t),t):
two square-strength windows per node with different phases.
Refuted by: an empty W(s,t) at s >= 30; a ratio drift beyond 5%; a centre-twin rate off by 3x.
"""
import sys, time, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from lib_twins import twin_centres_upto, count_in_windows

T_MAX = 30_000
S_MIN = 30
t0 = time.time()
N = (T_MAX + 1) ** 2 + 10
tw = twin_centres_upto(N)                     # exact twin centres to (t+1)^2
print(f"twin centres to {N:.3e}: {len(tw):,}   ({time.time()-t0:.1f}s)")
small = tw[tw <= T_MAX]
pairs_s = small[:-1]
pairs_t = small[1:]
mask = pairs_s >= S_MIN
s = pairs_s[mask].astype(np.int64)
t = pairs_t[mask].astype(np.int64)
print(f"consecutive pairs with {S_MIN} <= s < t <= {T_MAX}: {len(s):,}")

# product window and the two square stretches
Nst = count_in_windows(tw, (s - 1) * (t - 1), (s + 1) * (t + 1))
Nss = count_in_windows(tw, (s - 1) ** 2, (s + 1) ** 2)
Ntt = count_in_windows(tw, (t - 1) ** 2, (t + 1) ** 2)
law = 2 * (s + t) * 1.3203 / np.log(s.astype(float) * t) ** 2
ratio = Nst / law

# is the window centre st a twin centre?  (st-1, st+1 both prime)  exact via membership
st = s * t
centre_twin = np.isin(st, tw)
gen_rate = 1.3203 / np.log(st.astype(float)) ** 2   # generic twin-centre rate among multiples of 6 -> per column
# per-column rate among multiples of 6 is 6*1.3203/ln^2, but we compare centre hit count to expected count directly
exp_centre = np.sum(6 * 1.3203 / np.log(st.astype(float)) ** 2)

print("\n=== RESULT TABLE: consecutive-product window W(s,t) ===")
print(f"{'quantity':52s} {'value':>14s}")
print(f"{'empty product windows (N(s,t) = 0)':52s} {int(np.sum(Nst == 0)):>14d}")
i = int(np.argmin(Nst))
print(f"{'min N(s,t)':52s} {int(Nst[i]):>14d}   at (s,t) = ({s[i]},{t[i]})")
print(f"{'mean N(s,t)':52s} {Nst.mean():>14.2f}")
print(f"{'mean ratio N(s,t) / law':52s} {ratio.mean():>14.4f}")
print(f"{'ratio std / expected Poisson std':52s} {ratio.std()/np.mean(1/np.sqrt(law)):>14.3f}")
for lo_, hi_ in [(30, 1000), (1000, 3000), (3000, 10000), (10000, 30000)]:
    m = (t > lo_) & (t <= hi_)
    if m.any():
        print(f"{'  ratio by decade t in (%d, %d]' % (lo_, hi_):52s} {ratio[m].mean():>14.4f}   n={m.sum()}")
print(f"{'pairs with N(s,t) > T(s)':52s} {int(np.sum(Nst > Nss)):>14d}")
print(f"{'pairs with N(s,t) > T(t)':52s} {int(np.sum(Nst > Ntt)):>14d}")
print(f"{'pairs with N(s,t) > both':52s} {int(np.sum((Nst > Nss) & (Nst > Ntt))):>14d}")
print(f"{'window centre st is itself a twin centre (count)':52s} {int(centre_twin.sum()):>14d}   expected generic {exp_centre:.1f}")
print(f"{'min T(s) / min T(t) over these pairs':52s} {int(Nss.min()):>7d} / {int(Ntt.min())}")
# disjointness check, exact
gap = (t - 1) ** 2 - (s + 1) * (t + 1)
print(f"{'W(s,t) strictly between the stretches (all pairs)':52s} {str(bool(np.all(gap > 0) and np.all((s-1)*(t-1) > (s+1)**2))):>14s}")

verdict = "HOLDS" if (np.sum(Nst == 0) == 0 and abs(ratio.mean() - 1) < 0.05) else "FAILS"
print(f"\nVERDICT: {verdict}   ({time.time()-t0:.1f}s)")

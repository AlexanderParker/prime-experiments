"""ROUTE 2 - the product forest (every found twin becomes a multiplier).

For twin centres s <= t the product window W(s,t) = ((s-1)(t-1), (s+1)(t+1)) holds the members
st + 6j -+ 1, |j| <= c_s + c_t - 1; a twin centre inside is a PRODUCT-RUNG of (s,t).  s = t is the
square ladder (LadderHyp); s = 6 is the window (5t-5, 7t+7); s ~ t^a gives localisation exponent
theta = 1/(1+a) at height t^(1+a).  ProductHyp: every twin centre t has a product-rung with SOME
twin centre s <= t.  LadderHyp => ProductHyp => twins unbounded (a product-rung exceeds t), and the
supply of windows at t is the number of twins already found - the structure is self-improving in
the brief's sense (b).  Gears s+-1, t+-1 strike W(s,t) at the two cross-product columns
j = +-(c_t - c_s) only ((s+1)(t-1), (s-1)(t+1)); the centre st -+ 1 has no identity factor.

CLAIMS (each with a number):
 (A) every pair of twin centres 42 <= s <= t <= 10,000 has a product-rung; the minimum count is 3,
     attained only on the diagonal (the square ladder's minimum at s = t = 42);
 (B) the product forest on twin centres <= 10^6 has exactly three roots, 6, 12, 18 - every other
     twin centre u is a product-rung of some pair s <= t < u;
 (C) the multiplier-6 window (5t-5, 7t+7) alone holds a twin centre for every twin centre
     t <= 10^6 with at most 5 exceptions, all below t = 100;
 (D) the product CENTRE st is a twin centre at K x the generic rate, K = prod_p r_p where r_p is
     the exact probability that a product of two admissible residues mod p avoids +-1, divided by
     1 - 2/p (K < 1: the shared phases of the two factors are visible at the centre).
Refuted by: an empty window in (A); a fourth root in (B); > 5 exceptions or one above 100 in (C);
a centre rate outside K +- 25% in (D).
"""
import sys, time, math, bisect
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from lib_twins import twin_centres_upto, count_in_windows, primes_upto

t0 = time.time()
# ---------- (A) all pairs 42 <= s <= t <= 10^4 ----------
T_A = 10_000
tw8 = twin_centres_upto((T_A + 1) ** 2 + 10)
small = tw8[(tw8 >= 42) & (tw8 <= T_A)]
S, T = np.meshgrid(small, small, indexing="ij")
m = S <= T
S = S[m].astype(np.int64); T = T[m].astype(np.int64)
N = count_in_windows(tw8, (S - 1) * (T - 1), (S + 1) * (T + 1))
law = 2 * (S + T) * 1.3203 / np.log(S.astype(float) * T) ** 2
print(f"(A) pairs: {len(S):,}; twin centres to {(T_A+1)**2:.2e}: {len(tw8):,}  ({time.time()-t0:.1f}s)")
print(f"    empty windows: {int((N == 0).sum())};  min count {int(N.min())} at pairs "
      f"{[(int(a), int(b)) for a, b in zip(S[N == N.min()], T[N == N.min()])][:6]}")
print(f"    off-diagonal min count: {int(N[S < T].min())} at "
      f"{[(int(a), int(b)) for a, b in zip(S[(S<T)&(N==N[S<T].min())], T[(S<T)&(N==N[S<T].min())])][:6]}")
print(f"    mean N/law {float((N/law).mean()):.4f};  diagonal mean N/law {float((N/law)[S==T].mean()):.4f}")

# ---------- (B) roots of the product forest to 10^6 ----------
T_B = 1_000_000
tw6 = twin_centres_upto(T_B + 10)
twl = tw6.tolist()
roots = []
indeg = np.zeros(len(twl), dtype=np.int64)
for idx, u in enumerate(twl):
    deg = 0
    # s ranges over twin centres with s <= t and (s-1)(t-1) < u : s <= sqrt(u) + 2 suffices
    smax = int(math.isqrt(u)) + 2
    for s in twl:
        if s > smax:
            break
        # t in (u/(s+1) - 1, u/(s-1) + 1), t >= s, t < u
        tlo = u / (s + 1) - 1
        thi = u / (s - 1) + 1
        a = bisect.bisect_right(twl, tlo)
        b = bisect.bisect_left(twl, thi)
        for t in twl[a:b]:
            if t >= s and t < u and (s - 1) * (t - 1) < u < (s + 1) * (t + 1):
                deg += 1
    indeg[idx] = deg
    if deg == 0:
        roots.append(u)
print(f"(B) twin centres to 10^6: {len(twl):,}; product-forest roots: {roots}  ({time.time()-t0:.1f}s)")
print(f"    in-degree (number of parent pairs) min over u >= 30: {int(indeg[np.array(twl) >= 30].min())}, "
      f"median {int(np.median(indeg))}, at u ~ 10^6: {int(indeg[-1])}")

# ---------- (C) multiplier 6 only ----------
tw7 = twin_centres_upto(7 * T_B + 20)
Nc = count_in_windows(tw7, 5 * tw6 - 5, 7 * tw6 + 7)
exc = tw6[Nc == 0]
print(f"(C) multiplier-6 window (5t-5, 7t+7): exceptions (no twin centre) = {exc.tolist()};  "
      f"min count for t > 100: {int(Nc[tw6 > 100].min())} at t = {int(tw6[tw6 > 100][np.argmin(Nc[tw6 > 100])])}")

# ---------- (D) the product centre's own singular factor ----------
def r_p(p):
    adm = [a for a in range(p) if a not in (1, p - 1)]
    ok = sum(1 for a in adm for b in adm if (a * b) % p not in (1, p - 1))
    return (ok / len(adm) ** 2) / (1 - 2 / p)
K = 1.0
for p in primes_upto(200):
    if p >= 5:
        K *= r_p(int(p))
ST = S * T
centre = np.isin(ST, tw8)
expected_generic = float(np.sum(6 * 1.3203 / np.log(ST.astype(float)) ** 2))
obs = int(centre.sum())
print(f"(D) product centre st twin: observed {obs}, generic expectation {expected_generic:.1f}, "
      f"K = {K:.4f} (r_5 = {r_p(5):.4f}, r_7 = {r_p(7):.4f}, r_11 = {r_p(11):.4f}), "
      f"predicted {K*expected_generic:.1f}, observed/predicted = {obs/(K*expected_generic):.3f}")

vA = (N == 0).sum() == 0 and N.min() == 3 and N[S < T].min() > 3
vB = roots == [6, 12, 18]
vC = len(exc) <= 5 and (len(exc) == 0 or exc.max() < 100)
vD = abs(obs / (K * expected_generic) - 1) < 0.25
print(f"\nVERDICT: A {'HOLDS' if vA else 'FAILS'} | B {'HOLDS' if vB else 'FAILS'} | "
      f"C {'HOLDS' if vC else 'FAILS'} | D {'HOLDS' if vD else 'FAILS'}   ({time.time()-t0:.1f}s)")

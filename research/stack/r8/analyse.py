"""Claims 1, 2, 4 from rungs_0_3000.csv."""
import os, math, itertools
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.genfromtxt(os.path.join(HERE, "rungs_0_3000.csv"), delimiter=",", names=True)
s = D["s"].astype(np.int64); j = D["j"].astype(np.int64); sp = D["sp"].astype(np.int64)
T = D["T"].astype(np.int64); L = D["L"]; Lw = D["L_win"]
n = len(T)
print(f"rungs {n}, parents {len(np.unique(s))}, s' range [{sp.min()}, {sp.max()}], T range [{T.min()}, {T.max()}]")
print(f"L: min {L.min():.6f} max {L.max():.6f};  L_win: min {Lw.min():.8f} max {Lw.max():.8f}")

# parent check: T(s) for the parents themselves vs 1.3203 s/ln^2 s
ps, cnt = np.unique(s, return_counts=True)
print(f"parents: mean T(s)/(s/ln^2 s) = {np.mean(cnt/(ps/np.log(ps)**2)):.4f} over {len(ps)} parents (measured 1.3203 earlier)")


def omega_gear(x):
    x = abs(int(x)); k = 0; p = 5
    while p * p <= x:
        if x % p == 0:
            k += 1
            while x % p == 0: x //= p
        p += 2
    if x >= 5: k += 1
    return k


def regress(y, x, w=None, label=""):
    X = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])
    if w is None: w = np.ones_like(y)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = (y - X @ beta) * sw
    dof = len(y) - 2
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))
    se = math.sqrt(cov[1, 1])
    print(f"  {label}: slope {beta[1]:+.5f}  SE {se:.5f}  slope/SE {beta[1]/se:+.2f}  (intercept {beta[0]:.4f})")
    return beta[1], se


print("\n=== Claim 1: inherited local factor ===")
base = 4 * sp / (6 * np.log(sp) ** 2)
law0 = 1.3203 * L * base
print(f"mean T/law0 (brief's constant 1.3203) = {np.mean(T/law0):.4f}")
C = np.mean(T / (L * base))
law = C * L * base
ratio = T / law
print(f"fitted constant C = {C:.4f} in C*L*4s'/(6 ln^2 s'); equivalently {C*4/6:.4f} in C*s'/ln^2 s'; mean ratio {ratio.mean():.6f}")
print(f"var(ratio) = {ratio.var(ddof=1):.3e};  1/mean(T) = {1/T.mean():.3e};  mean(1/law) = {np.mean(1/law):.3e}")
print(f"var(ratio)*mean(T) = {ratio.var(ddof=1)*T.mean():.3f};  var(ratio)/mean(1/law) = {ratio.var(ddof=1)/np.mean(1/law):.3f}")
disp = np.mean((T - law) ** 2 / law)
print(f"dispersion index mean((T-law)^2/law) = {disp:.3f}  (Poisson 1, SE about {math.sqrt(2/n):.3f})")
# by height bands, to show where any non-Poisson spread comes from
print("  height bands (s' decile): n, mean ratio, dispersion index")
edges = np.quantile(sp, np.linspace(0, 1, 6))
for a, b in zip(edges[:-1], edges[1:]):
    m = (sp >= a) & (sp <= b)
    print(f"    s' in [{int(a)}, {int(b)}]: n {m.sum():4d}  mean ratio {ratio[m].mean():.4f}  disp {np.mean((T[m]-law[m])**2/law[m]):.3f}")
om = np.array([omega_gear(x) for x in j])
jm30 = j % 30
print("regressions of ratio (OLS and WLS with weight = law):")
regress(ratio, om, label="ratio ~ omega_gear(j) [= omega(6j)-2], OLS")
regress(ratio, om, w=law, label="ratio ~ omega_gear(j), WLS")
regress(ratio, jm30, label="ratio ~ (j mod 30) numeric, OLS")
regress(ratio, jm30, w=law, label="ratio ~ (j mod 30) numeric, WLS")
print("  omega_gear(j) classes: k, n, mean ratio, SE")
for k in sorted(set(om)):
    m = om == k
    print(f"    {k}: n {m.sum():4d}  mean {ratio[m].mean():.4f}  SE {ratio[m].std(ddof=1)/math.sqrt(m.sum()) if m.sum()>1 else float('nan'):.4f}")
groups = [ratio[jm30 == r] for r in range(30) if np.sum(jm30 == r) > 1]
F, p = stats.f_oneway(*groups)
print(f"  one-way ANOVA of ratio across j mod 30 classes: F {F:.3f} p {p:.3f} ({len(groups)} classes)")
print("  j mod 30 classes: r, n, mean ratio")
print("   ", ", ".join(f"{r}:{np.sum(jm30==r)}/{ratio[jm30==r].mean():.3f}" for r in range(30) if np.sum(jm30 == r) > 0))

print("\n=== Claim 2: smooth offsets ===")
def report(mask, name):
    a = ratio[mask]; b = ratio[~mask]
    sea = a.std(ddof=1) / math.sqrt(len(a)); seb = b.std(ddof=1) / math.sqrt(len(b))
    z = (a.mean() - b.mean()) / math.sqrt(sea ** 2 + seb ** 2)
    print(f"  {name:8s}: n {len(a):4d} mean {a.mean():.4f} SE {sea:.4f} | rest n {len(b):4d} mean {b.mean():.4f} SE {seb:.4f} | diff/SE {z:+.2f}")
report(j % 5 == 0, "5 | j")
report(j % 7 == 0, "7 | j")
report(j % 35 == 0, "35 | j")
report(om == 0, "om=0")

print("\n=== Claim 4: missing (s mod g, j mod g) pairs ===")
for g in (5, 7, 11, 13):
    occ = set(zip((s % g).tolist(), (j % g).tolist()))
    missing = [(a, b) for a in range(g) for b in range(g) if (a, b) not in occ]
    inv6 = pow(6, -1, g)
    expl_a = [(a, b) for (a, b) in missing if a in (1, g - 1)]
    expl_b = [(a, b) for (a, b) in missing if (a * a + 6 * b) % g in (1, g - 1)]
    unexpl = [(a, b) for (a, b) in missing if (a, b) not in set(expl_a) | set(expl_b)]
    # coverage: pairs allowed by both conditions that occur / allowed
    allowed = [(a, b) for a in range(g) for b in range(g) if a not in (1, g - 1) and (a * a + 6 * b) % g not in (1, g - 1)]
    occ_allowed = [p for p in allowed if p in occ]
    # pairs with a = +-1 that DO occur (parents 6, 12)
    occ_pm1 = sorted(p for p in occ if p[0] in (1, g - 1))
    print(f"g={g}: pairs {g*g}, occurring {len(occ)}, missing {len(missing)}; "
          f"explained by s=+-1 mod g: {len(expl_a)}; by s^2+6j=+-1 mod g: {len(expl_b)}; "
          f"both: {len(set(expl_a)&set(expl_b))}; unexplained: {len(unexpl)} {unexpl}")
    print(f"      allowed by both conditions: {len(allowed)}, of which occurring {len(occ_allowed)}; "
          f"occurring pairs with s=+-1 mod g: {occ_pm1} (from parents {sorted(set(s[(s%g==1)|(s%g==g-1)].tolist()))})")
    print(f"      missing pairs: {missing}")
    print(f"      min count over occurring allowed pairs: {min(sum(1 for a,b in zip(s%g, j%g) if (a,b)==p) for p in occ_allowed)}")

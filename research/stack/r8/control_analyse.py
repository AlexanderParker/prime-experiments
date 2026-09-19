"""Dispersion of rung counts vs same-height control windows, against the same law."""
import os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = np.genfromtxt(os.path.join(HERE, "rungs_0_3000.csv"), delimiter=",", names=True)
Cc = np.genfromtxt(os.path.join(HERE, "control_0_3000.csv"), delimiter=",", names=True)
ctrl = dict(zip(Cc["sp"].astype(np.int64), Cc["T_ctrl"].astype(np.int64)))
sp = R["sp"].astype(np.int64); T = R["T"].astype(np.int64)
Tc = np.array([ctrl[x] for x in sp])
base = 4 * sp / (6 * np.log(sp) ** 2)
n = len(T)
for name, X in (("rungs", T), ("control", Tc)):
    C = np.mean(X / base)
    law = C * base
    disp = np.mean((X - law) ** 2 / law)
    print(f"{name:8s}: n {n}  fitted C {C:.4f}  mean count {X.mean():.1f}  dispersion index {disp:.3f} (SE {math.sqrt(2/n):.3f})  var(ratio)/mean(1/law) {np.var(X/law, ddof=1)/np.mean(1/law):.3f}")
    edges = np.quantile(sp, np.linspace(0, 1, 6))
    print("   bands:", "  ".join(f"[{int(a)},{int(b)}] {np.mean((X[m]-law[m])**2/law[m]):.3f}" for a, b in zip(edges[:-1], edges[1:]) for m in [(sp >= a) & (sp <= b)]))
# paired comparison
d = T - Tc
print(f"paired: mean(T - T_ctrl) = {d.mean():+.2f}  SE {d.std(ddof=1)/math.sqrt(n):.2f};  corr(T, T_ctrl) = {np.corrcoef(T, Tc)[0,1]:.4f}")
# large-count subset (T >= 5000) where Poisson approx of the index is clean
m = T >= 5000
for name, X in (("rungs", T), ("control", Tc)):
    C = np.mean(X[m] / base[m]); law = C * base[m]
    print(f"  {name:8s} T>=5000 subset: n {m.sum()}  dispersion {np.mean((X[m]-law)**2/law):.3f} (SE {math.sqrt(2/m.sum()):.3f})")

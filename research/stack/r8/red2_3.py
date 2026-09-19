"""Red lane, script 3: final task-D numbers with the corrected K, and small-t bias for task E."""
import math
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
C2 = 0.6601618158468695739278121100145
LAW = 2 * C2
K = 0.8231032  # converged prod_{p>=5} r_p, from red2

twins = np.load(os.path.join(OUT, "twins.npy"))
n6 = np.load(os.path.join(OUT, "n6.npy")).astype(np.float64)
tarr = np.load(os.path.join(OUT, "t6.npy")).astype(np.int64)
T = np.load(os.path.join(OUT, "sqT.npy")).astype(np.float64)
ts = np.load(os.path.join(OUT, "sqt.npy")).astype(np.int64)

B = twins[(twins >= 42) & (twins <= 10**4)]
obs = 0
E = 0.0
Eoff = 0.0
obs_list = []
for i in range(len(B)):
    s = int(B[i])
    tt = B[i:]
    prod = (s * tt).astype(np.int64)
    pos = np.searchsorted(twins, prod)
    hit = twins[np.minimum(pos, len(twins) - 1)] == prod
    obs += int(hit.sum())
    for j in np.nonzero(hit)[0]:
        obs_list.append((s, int(tt[j])))
    e = (12 * C2 / np.log(prod.astype(np.float64)) ** 2)
    E += float(e.sum())
    Eoff += float(e[1:].sum())

npairs = len(B) * (len(B) + 1) // 2
print(f"centres in [42,10^4] = {len(B)}   pairs = {npairs}")
print(f"observed st a twin centre : {obs}")
print(f"E_generic                 : {E:.3f}")
print(f"obs/E_generic             : {obs/E:.5f}   (predicted K = {K:.5f})")
print(f"(obs/E)/K                 : {obs/E/K:.5f}")
print(f"K*E_generic               : {K*E:.2f}")
print(f"Poisson z vs K*E          : {(obs-K*E)/math.sqrt(K*E):+.3f}   sd={math.sqrt(K*E):.2f}")
print(f"two-sided p (normal)      : {math.erfc(abs(obs-K*E)/math.sqrt(2*K*E)):.4f}")
print(f"diagonal pairs s=t        : {len(B)}, E charged {E-Eoff:.2f}, observed 0 (s^2-1 factors)")
print(f"off-diagonal obs/E        : {obs/Eoff:.5f}   /K = {obs/Eoff/K:.5f}, "
      f"z = {(obs-K*Eoff)/math.sqrt(K*Eoff):+.3f}")
print(f"smallest 5 st hits        : {obs_list[:5]}")

print()
print("small-t bias of the two laws (task E control)")
for lo, hi in ((6, 200), (200, 1000), (1000, 10**4), (10**4, 10**5)):
    m6 = (tarr >= lo) & (tarr < hi)
    mS = (ts >= lo) & (ts < hi)
    r6 = n6[m6] / (2 * (tarr[m6] + 6) * LAW / np.log(6.0 * tarr[m6]) ** 2)
    rS = T[mS] / (LAW * ts[mS] / np.log(ts[mS].astype(np.float64)) ** 2)
    print(f"  t in [{lo},{hi}): n={int(m6.sum()):>5}  mean ratio W6={r6.mean():.4f}  "
          f"mean ratio square={rS.mean():.4f}")

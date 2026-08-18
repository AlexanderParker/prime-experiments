"""Round 18: how much of the gap-histogram erraticity does the autocorrelation
construct explain? Regress log(count) on g (smooth decay) with and without
log(c_5*c_7) (the arithmetic selection)."""
import numpy as np
from exposed_autocorr import c_q, gap_hist

for y in (19, 23, 29):
    cnt, gears = gap_hist(y)
    F = int(max(i for i in range(len(cnt)) if cnt[i]))
    lo = 15
    g = np.array([x for x in range(lo, F + 1) if cnt[x] > 0])
    n = np.array([cnt[x] for x in g], float)
    ph = np.array([c_q(5, x) * c_q(7, x) for x in g], float)
    ly, lg, lp = np.log(n), g.astype(float), np.log(ph)
    def fit(cols):
        A = np.column_stack([np.ones(len(g))] + cols)
        b, *_ = np.linalg.lstsq(A, ly, rcond=None)
        r = ly - A @ b
        return 1 - r.var() / ly.var(), r.std()
    r2a, sda = fit([lg])
    r2b, sdb = fit([lg, lp])
    print(f"machine {y} (F={F}, {len(g)} nonzero gap values in [{lo},{F}]):")
    print(f"  log(count) ~ g            : R^2 = {r2a:.3f}, residual sd = {sda:.3f}")
    print(f"  log(count) ~ g + log(c5c7): R^2 = {r2b:.3f}, residual sd = {sdb:.3f}")
    print(f"  -> the autocorrelation explains "
          f"{100*(1 - (sdb/sda)**2):.0f}% of the residual variance "
          f"left by the smooth decay")
    zeros = [x for x in range(lo, F + 1) if cnt[x] == 0]
    if zeros:
        print(f"  absent values below F: {zeros}; their c5*c7 = "
              f"{[c_q(5,x)*c_q(7,x) for x in zeros]} (minimum possible is 3)")

import numpy as np
from math import prod
ps = [7, 11, 13, 17, 19, 23]
nxt = {7: 11, 11: 13, 13: 17, 17: 19, 19: 23, 23: 29}
for i, q in enumerate(ps):
    gs = ps[:i + 1]; P = prod(gs) * 5
    k = np.arange(P); w = np.isin(k % 5, (0, 2, 3))
    for g in gs:
        u = pow(6, -1, g); w &= (k % g != u) & (k % g != g - u)
    opens = np.flatnonzero(w)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]])) - 1
    W = (nxt[q] ** 2 - q * q) // 6; k0 = q * q // 6
    order = np.argsort(-gaps)[:6]
    print(f"q={q}: period {P}, mirror centre {P/2:.0f}, window k={k0}..{k0+W-1} (W={W}); worst runs (length @ start slot, fraction of period, distance to centre):")
    print("   " + "; ".join(f"{int(gaps[o])} @ {int(opens[o])+1} ({(opens[o]+1)/P:.3f}, {abs(opens[o]+1-P/2):.0f})" for o in order))
    # how typical is a short run at the window start: fraction of period positions lying inside a blocked run of length >= W
    cover = np.zeros(P, bool)
    for o in np.flatnonzero(gaps >= W):
        a = int(opens[o]) + 1; cover[a:a + int(gaps[o])] = True
    print(f"   fraction of the period inside a blocked run >= W: {cover.mean():.4f}; runs >= W: {int((gaps >= W).sum())}; run at window start: {int(gaps[np.searchsorted(opens, k0) - 1])}")

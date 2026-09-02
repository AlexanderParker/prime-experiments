import numpy as np
from math import prod
ps = [7, 11, 13, 17, 19, 23, 29, 31]
nxt = {7: 11, 11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}
print("gears 7..q: full-period word of untouched anchor-open slots; longest blocked run anywhere (F) vs the current window W = (q'^2 - q^2)/6; and the run at the window")
for i, q in enumerate(ps):
    gs = ps[:i + 1]; P = prod(gs) * 5
    if P > 200_000_000: break
    k = np.arange(P)
    w = np.isin(k % 5, (0, 2, 3))
    for g in gs:
        u = pow(6, -1, g); w &= (k % g != u) & (k % g != g - u)
    opens = np.flatnonzero(w)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]]))   # cyclic
    F = int(gaps.max()) - 1
    where = int(opens[gaps.argmax()]) + 1
    W = (nxt[q] ** 2 - q * q) // 6
    k_lo, k_hi = q * q // 6 + 1, (nxt[q] ** 2 - 2) // 6
    inwin = opens[(opens >= k_lo) & (opens <= k_hi)]
    # blocked run that covers the window start
    prev = opens[opens < k_lo].max() if (opens < k_lo).any() else -1
    nxt_open = opens[opens >= k_lo].min()
    print(f"  q={q:>2}: period {P:>10} slots, open {len(opens):>8}; F = {F:>4} blocked slots at k={where} (numbers {6*where-1}..); window W = {W:>3} slots at k={k_lo}..{k_hi}; "
          f"open slots in window {len(inwin)}, first at k={int(nxt_open)} ({6*int(nxt_open)-1}|{6*int(nxt_open)+1}), blocked run entering the window {int(nxt_open - prev - 1)}")

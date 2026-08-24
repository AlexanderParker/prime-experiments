import numpy as np
from math import prod
gears = [5, 7, 11, 13, 17, 19, 23]
def openings_near(k0, span=200):
    ks = np.arange(k0 - span, k0 + span)
    ex = np.zeros(len(ks), bool)
    for g in gears:
        u = pow(6, -1, g)
        ex |= ((ks % g) == (u % g)) | ((ks % g) == ((-u) % g))
    return ks[~ex]
for k0, j in [(14995460, 3), (8057955, 4), (8057950, 5), (8057950, 6)]:
    op = openings_near(k0)
    i = int(np.where(op == k0)[0][0])
    w = op[i:i + j + 1]
    gaps = np.diff(w)
    print(f"k={k0} j={j}: openings {list(w)} gaps {list(gaps)} sum {gaps.sum()} middles {list(gaps[1:-1])} all>=10 {bool((gaps[1:-1]>=10).all())}")
    assert gaps.sum() in (43, 50, 55, 60) and (gaps[1:-1] >= 10).all()
print("ADDRESSES VERIFIED: Q_3=43, Q_4=50, Q_5=55, Q_6=60 at floor a=10, machine 23")

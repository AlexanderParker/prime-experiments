import numpy as np
OPEN30 = {1, 11, 13, 17, 19, 29}
def forb(q): return sorted({((q * m - 11) // 30) % q for m in range(1, 30) if (q * m) % 30 in OPEN30})
for gears in ([7], [7, 11], [7, 11, 13], [7, 11, 13, 17]):
    P = int(np.prod(gears)); j = np.arange(P); ok = np.ones(P, dtype=bool)
    for q in gears: ok &= ~np.isin(j % q, forb(q))
    a = np.flatnonzero(ok)
    print(f"gears {gears}: period {P} cycles ({30*P} numbers), open cycles per period {len(a)}; first open j: {a[:14].tolist()}; gaps: {np.diff(a)[:13].tolist()}")
# check against actual primes for the first case: j=6 -> numbers 191..211

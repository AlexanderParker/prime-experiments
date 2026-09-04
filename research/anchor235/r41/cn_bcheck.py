"""R2.a.i.a.1.a - N-C5 across a third bound: K_13(d) against K_7 at the same island count."""
import sys, time
sys.path.insert(0, "research/anchor235/r41")
from cn_real import sieve, island_pred, adversary_K

NMAX = 31000
FL = sieve(NMAX)
PR = [p for p in range(5, NMAX + 1) if FL[p]]
MOD, RES = island_pred(13, PR)
RS = set(RES)
print("B = 13 : %d island classes mod %d (density %.6f)" % (len(RES), MOD, len(RES) / MOD))
print("   d       m     K_13(d)   secs")
for d in [1260, 2100, 3360, 5460, 7560, 10010]:
    t0 = time.time()
    isl = [i for i in range(1, d) if i % MOD in RS]
    if not isl:
        continue
    K, lb, gears = adversary_K(d, isl, PR, 13, 600.0)
    print("  %-7d %-5d %-9d %.1f   %s" % (d, len(isl), K, time.time() - t0, gears[:14]))

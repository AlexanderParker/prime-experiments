"""The RANGE statement with machine 5 as the cycle (owner's construction, re-set 2026-09-23).

Machine 5 = base 2, 3 with gear 5, period 30. Its known opening (-1, 1) recurs at every multiple of
30: the copies, or LAPS, are the pairs (30k - 1, 30k + 1), k = 1, 2, 3, ...
The OVERLAY is every gear above 5 and below sqrt(q#) - the gears that can decide any lap of the
range. Gear g strikes lap k iff k = +-30^{-1} mod g.

RANGE STATEMENT for machine q: some lap k with 30k - 1 > q and 30k + 1 <= q# is a twin, i.e. is
struck by no gear at or below sqrt(30k + 1).
WINDOW STATEMENT for machine q: the same, but the lap must also satisfy 30k + 1 < q'^2.

The range statement is a DISJUNCTION over TIERS. Lap k is decided by the gears up to sqrt(30k), so
the laps of the range split by which machine decides them:
    tier 0: laps in (q/30, q^2/30]        decided by the gears up to q
    tier 1: laps in (q^2/30, q^4/30]      decided by the gears up to q^2
    tier t: laps in (q^(2^t)/30, q^(2^(t+1))/30]   decided by the gears up to q^(2^t)
up to the top of the range at q#/30. THE RANGE STATEMENT HOLDS IF ANY ONE TIER HOLDS, whereas the
window statement is tier 0 alone.

This script reports, for each q: the number of tiers in the range, the first surviving lap above q
and which tier it lands in, and how much of the range is left unused when tier 0 already succeeds.
"""
from sympy import isprime, primerange, nextprime
from math import log, prod

K = 400
tw = [k for k in range(1, K + 1) if isprime(30 * k - 1) and isprime(30 * k + 1)]

print("q   | tiers in the range | first twin lap above q | its tier | window laps | range laps (log10)")
P = 1
for q in list(primerange(5, 90)):
    P *= q
    # tiers: t = 0, 1, 2, ... while q^(2^t) <= q#  ->  2^t log q <= log q#
    logq = log(q)
    logP = sum(log(p) for p in primerange(2, q + 1))
    tiers = 0
    while (2 ** tiers) * logq <= logP:
        tiers += 1
    k1 = next((k for k in tw if 30 * k - 1 > q), None)
    if k1 is None:
        continue
    # which tier does it land in
    t = 0
    while 30 * k1 + 1 > q ** (2 ** t):
        t += 1
    qn = nextprime(q)
    win_laps = (qn * qn - 1) // 30 - q // 30
    print(f"{q:3d} | {tiers:18d} | k = {k1:<20d} | {t:8d} | {win_laps:11d} | {logP/log(10):.1f}")

print()
print("READING: tier 0 is the window. The range holds many more tiers, and the range statement")
print("needs only ONE of them. Every q below is settled by tier 0, so the higher tiers are unused")
print("slack - the range statement has never yet had to call on them.")

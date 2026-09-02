# anchor 2,3,5: for each gear q, the six hit cycles (1-based, cycle = slots 5j..5j+4), the m with q*m open, positions as fraction of q
from collections import defaultdict
OPEN = {1, 11, 13, 17, 19, 29}
def hits(q):
    out = []
    for m in range(1, 30):
        n = q * m
        if n % 30 in OPEN:
            k = (n + 1) // 6 if n % 6 == 5 else (n - 1) // 6
            out.append((k // 5 + 1, m, n))
    return sorted(out)
primes = [p for p in range(11, 400) if all(p % d for d in range(2, int(p**0.5) + 1))]
by_class = defaultdict(list)
for q in primes:
    h = hits(q)
    by_class[q % 30].append((q, h))
for r in sorted(by_class):
    print(f"=== q = {r} mod 30: m = {[m for _, m, _ in hits(by_class[r][0][0])]} ===")
    for q, h in by_class[r][:6]:
        cyc = [c for c, _, _ in h]
        gaps = [cyc[0] - 1] + [b - a - 1 for a, b in zip(cyc, cyc[1:])] + [q - cyc[-1]]
        print(f"  q={q:>3}: hit cycles {cyc}  fractions {[round(c / q, 3) for c in cyc]}  untouched runs {gaps}")

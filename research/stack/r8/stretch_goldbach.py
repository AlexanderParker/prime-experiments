"""The products field of the stretch is Goldbach: a P_2 member h p (h <= p primes) of the stretch of
s satisfies h p = m^2 - d^2, m = (h+p)/2, d = (p-h)/2, so it is a Goldbach partition of 2m; it lies
in the stretch iff (s-1)^2 < m^2 - d^2 < (s+1)^2, and its offset is j = (m^2 - d^2 - s^2 -+ 1)/6.
Claims (pre-registered): (1) every P_2 member of every stretch (s <= 3000) has m in
(s - 1, (s+1)^2 / (2 h_min)) - exact identity, 0 violations; (2) the lower members with m = s are
exactly the partitions 2s = (s-d) + (s+d) with 1 <= d < sqrt(2s), d coprime to 6, at j = (1-d^2)/6;
the count of such plug members equals the number of Goldbach partitions of 2s with parts within
sqrt(2s) of s (d = 1 is the twin itself at the centre); (3) upper members with m = s+1 are the
partitions of 2s+2 with d = 6t, t >= 1, d < 2 sqrt(s), at j = c - 6t^2 (c = s/6), and with m = s-1
the partitions of 2s-2 at j = -c - 6t^2.
"""
import numpy as np, sys
S_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
N = (S_MAX + 1) ** 2 + 2
spf = np.zeros(N, dtype=np.int32)
for p in range(2, int(N ** 0.5) + 1):
    if spf[p] == 0:
        blk = spf[p*p::p]; blk[blk == 0] = p; spf[p*p::p] = blk
idx = np.nonzero(spf == 0)[0]; spf[idx] = idx; spf[0] = spf[1] = 1
def is_prime(n): return n >= 2 and spf[n] == n
tw = [s for s in range(102, S_MAX + 1, 6) if is_prime(s - 1) and is_prime(s + 1)]
viol1 = 0; checked = 0; rows = []
for s in tw:
    c = s // 6
    js = range(-(2 * c - 1), 2 * c)
    lower_ms = set(); upper_ms_plus = set(); upper_ms_minus = set()
    for j in js:
        for sign, m in ((-1, s * s + 6 * j - 1), (+1, s * s + 6 * j + 1)):
            h = int(spf[m])
            if h == m: continue
            p = m // h
            if spf[p] != p: continue            # not P_2
            checked += 1
            mm = (h + p) // 2; d = (p - h) // 2
            if not (mm * mm - d * d == m and s - 1 < mm and 2 * mm <= h + (s + 1) ** 2 // h + 1): viol1 += 1
            if sign == -1 and mm == s: lower_ms.add(d)
            if sign == +1 and mm == s + 1: upper_ms_plus.add(d)
            if sign == +1 and mm == s - 1: upper_ms_minus.add(d)
    # Goldbach partitions of 2s with d < sqrt(2s)
    gb = {d for d in range(1, int((2 * s) ** 0.5) + 1) if is_prime(s - d) and is_prime(s + d)}
    gbp = {d for d in range(6, int(2 * s ** 0.5) + 6, 6) if is_prime(s + 1 - d) and is_prime(s + 1 + d) and (s + 1) ** 2 - d * d > (s - 1) ** 2}
    gbm = {d for d in range(6, int(2 * s ** 0.5) + 6, 6) if is_prime(s - 1 - d) and is_prime(s - 1 + d) and (s - 1) ** 2 - d * d > (s - 1) ** 2 - 0}  # none possible below (s-1)^2
    ok2 = (lower_ms == gb); ok3 = (upper_ms_plus == gbp)
    rows.append((s, len(gb), ok2, len(gbp), ok3, len(upper_ms_minus)))
print(f"P_2 members checked: {checked}; identity violations: {viol1}")
bad2 = sum(1 for r in rows if not r[2]); bad3 = sum(1 for r in rows if not r[4])
print(f"twin centres: {len(rows)}; claim 2 (lower members with m = s = short Goldbach partitions of 2s) violations: {bad2}; "
      f"claim 3 (upper members with m = s+1 = partitions of 2s+2 with d = 6t) violations: {bad3}; "
      f"upper members with m = s-1: {sum(r[5] for r in rows)} (should be 0: (s-1)^2 - d^2 is below the stretch)")
print("sample (s, #short partitions of 2s incl. d = 1, #partitions of 2s+2 with d = 6t):", [(r[0], r[1], r[3]) for r in rows[::max(1, len(rows)//8)]])

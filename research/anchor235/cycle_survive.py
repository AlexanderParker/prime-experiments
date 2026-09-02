import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import numpy as np
from word_tree_r29 import spf_sieve
X = 100_000_000
spf = spf_sieve(X + 40)
isp = spf == np.arange(len(spf)); isp[:2] = False
J = X // 30
j = np.arange(J)
base = 30 * j
opens = np.stack([isp[base + r] for r in (11, 13, 17, 19, 29, 31)], axis=1)   # six open numbers of cycle j
slots = np.stack([opens[:, 0] & opens[:, 1], opens[:, 2] & opens[:, 3], opens[:, 4] & opens[:, 5]], axis=1)  # 11|13, 17|19, 29|31
nn = opens.sum(1); ns = slots.sum(1)
print(f"cycles j < {J} (numbers to {X}); anchor-open numbers per cycle surviving all gears (prime):")
print("  numbers surviving 0..6: " + ", ".join(f"{k}: {int((nn == k).sum())}" for k in range(7)))
print("  twin slots surviving 0..3: " + ", ".join(f"{k}: {int((ns == k).sum())}" for k in range(4)))
full = np.flatnonzero(nn == 6)
print(f"  cycles with whole anchor pattern surviving (all six prime): {len(full)}; first: {[(int(x), 30*int(x)) for x in full[:12]]}")
three = np.flatnonzero(ns == 3)
print(f"  cycles with all three twin slots surviving: {len(three)} (same set: {np.array_equal(full, three)})")
two = np.flatnonzero(ns == 2)
print(f"  cycles with two twin slots surviving: {len(two)}; first: {[30*int(x) for x in two[:10]]}; last below {X}: {30*int(two[-1])}")
# per section: does the section contain a cycle with >=2, 3 surviving slots
primes = np.flatnonzero(isp[: int(X**0.5) + 60]); primes = [int(p) for p in primes if p >= 5]
no3 = no2 = 0; nsec = 0
for a, b in zip(primes, primes[1:]):
    if b * b > X: break
    lo, hi = a * a // 30 + 1, (b * b) // 30 - 1
    if hi < lo: continue
    nsec += 1
    no3 += int((ns[lo:hi + 1] == 3).sum() == 0)
    no2 += int((ns[lo:hi + 1] >= 2).sum() == 0)
print(f"  sections (q^2, q'^2) to {X}: {nsec}; without a fully surviving cycle: {no3}; without even a 2-slot cycle: {no2}")

"""Constructor round 8: the multiplicative route (corpus 7.1 / review sec 7 pt 3).

Data: the exact consecutive-machine chain (adjacent frame, corpus + covering-
bound-route sec 16):  gears-to-y : F(2,y)
  5:6, 7:15, 11:21, 13:33, 17:54, 19:75, 23:102, 29:129, 31:174, 37:264,
  41:273, 43:309, 47:354   (53: >= 420, search unfinished)
Requirement (Reduction A per window, adjacent frame): F(2,y) < (y^2 - y)/2.

Part 1 - ratio data. r(step) = F(next)/F(prev) vs the window budget
(q'/q)^2; running margin F / [(y^2-y)/2]; cumulative log-budget bookkeeping.

Part 2 - the threshold theorem, verified numerically. If the increment
law incr(step q) <= alpha*q holds at every consecutive step with q > 47, then
    F(2,y) <= 354 + alpha * (S(y) - S(47)),   S(y) = sum of primes <= y,
and the conclusion F(2,y) < (y^2-y)/2 holds for ALL y >= 47 provided the
inequality 354 + alpha*(S(y)-328) < (y^2-y)/2 holds - checked here exactly
for every prime y <= 10^6, and beyond by Rosser-Schoenfeld
(S(y) < 1.25506*y^2/ln y, so alpha = 3 needs ln y > ~7.53: y > 1862 -
covered by the numeric range with room). Alpha = 2.5 and 3 both reported
(observed max incr/q = 2.432 at gear 37; corpus refuted only alpha = 1.8,
the odd-sum elementary threshold).

Part 3 - critical alpha*(y): the largest per-step constant the window
tolerates at each scale = [(y^2-y)/2 - 354] / (S(y) - 328) -> ln y growth.
"""
import math

CHAIN = [(5, 6), (7, 15), (11, 21), (13, 33), (17, 54), (19, 75), (23, 102),
         (29, 129), (31, 174), (37, 264), (41, 273), (43, 309), (47, 354)]


def part1():
    print("step      r=F'/F   budget(q'/q)^2  verdict   incr  incr/q   F'/req(y')")
    logsum_r = logsum_b = 0.0
    for (q0, f0), (q1, f1) in zip(CHAIN, CHAIN[1:]):
        r = f1 / f0
        b = (q1 / q0) ** 2
        logsum_r += math.log(r)
        logsum_b += math.log(b)
        req = (q1 * q1 - q1) / 2
        print(f"{q0:>3}->{q1:<3}  {r:6.4f}   {b:6.4f}          "
              f"{'OVER ' if r > b else 'under'}   {f1-f0:>4}  {(f1-f0)/q1:6.3f}   "
              f"{f1/req:6.4f}")
    print(f"cumulative: sum ln r = {logsum_r:.4f} vs sum ln budget = "
          f"{logsum_b:.4f}  (under overall, over at half the steps)")
    # F(2,53) partial
    print(f"53 partial: r >= {420/354:.4f} vs budget {(53/47)**2:.4f} "
          f"(decided when the search finishes; r <= budget iff F(2,53) <= "
          f"{int(354*(53/47)**2)})")


def prime_sums(limit):
    bs = bytearray([1]) * (limit + 1)
    bs[0:2] = b"\x00\x00"
    for i in range(2, int(limit**0.5) + 1):
        if bs[i]:
            bs[i * i:: i] = bytearray(len(bs[i * i:: i]))
    return bs


def part2(limit=10**6):
    bs = prime_sums(limit)
    S = 0
    S47 = None
    worst = (0.0, None)
    fails = []
    for n in range(2, limit + 1):
        if bs[n]:
            S += n
            if n == 47:
                S47 = S
            if n >= 53 and S47 is not None:
                req = (n * n - n) / 2
                for alpha in (2.5, 3.0):
                    lhs = 354 + alpha * (S - S47)
                    if lhs >= req:
                        fails.append((alpha, n))
                ratio = (354 + 3.0 * (S - S47)) / req
                if ratio > worst[0]:
                    worst = (ratio, n)
    print(f"\nS(47) = {S47} (sum of primes <= 47)")
    print(f"alpha in {{2.5, 3.0}}: failures of 354 + alpha*(S(y)-S(47)) < "
          f"(y^2-y)/2 for prime y in [53, {limit}]: {fails if fails else 'NONE'}")
    print(f"worst ratio (alpha=3): {worst[0]:.4f} at y = {worst[1]} "
          f"(max over all prime y; < 1 everywhere)")
    print("tail beyond 1e6: S(y) < 1.25506 y^2/ln y (Rosser-Schoenfeld), so "
          "3*S(y) < (y^2-y)/2 iff ln y > 7.531*(y/(y-1)): y >= 1863. Covered.")


def part3():
    bs = prime_sums(10**6)
    S = 0
    S47 = None
    marks = {47: None, 101: None, 1009: None, 10007: None, 100003: None,
             999983: None}
    for n in range(2, 10**6 + 1):
        if bs[n]:
            S += n
            if n == 47:
                S47 = S
            if n in marks:
                marks[n] = ((n * n - n) / 2 - 354) / (S - S47) if n > 47 else None
    print("\ncritical alpha*(y) = [(y^2-y)/2 - 354]/[S(y) - S(47)]  (~ln y):")
    for n, a in marks.items():
        if a:
            print(f"  y = {n:>7}: alpha* = {a:7.3f}   (ln y = {math.log(n):.3f})")


if __name__ == "__main__":
    part1()
    part2()
    part3()

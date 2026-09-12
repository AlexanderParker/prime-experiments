"""Multi-step walks on real mirror axes (owner: do it; 2026-09-13).

Flips about real axes a_1, a_2, ... (each a multiple of the product of a gear set with 2, 3).
Composition: two flips slide by 2 (a_2 - a_1); three flips reflect about a_3 - a_2 + a_1; in
general the walk from n is n -> 2A - n - 2 (odd steps) or n -> n + 2A (even steps) with A the
alternating sum of the axes. A slide by 2A carries the gears dividing A; a reflection about A
carries the gears dividing A. So the certification of the END of a walk is the set of gears
dividing the alternating sum A, whatever the intermediate landings were: stepwise tracking
(intersecting the carried sets flip by flip) undercounts it.
From home (-1, 1) every walk therefore ends on the column (2A - 1, 2A + 1), A a multiple of 6,
certified for the gears dividing A. Multi-step walks reach exactly the landings one flip
reaches, with A now free to be any multiple of 6 (a difference of two real axes such as
42 k_2 - 30 k_1), so the gears carried are the divisors of A rather than a fixed S.
Checks here: (1) the composition law on random walks, exact; (2) the undercount of stepwise
tracking; (3) the two-step rule: A = the smallest multiple of 6 with the landing in the window
whose class avoids the teeth of every gear not dividing A (teeth: 2A = -+1 mod h), realised as
home -> flip about 30 k_1 -> flip about 42 k_2 with 42 k_2 - 30 k_1 = A; certify the landing;
report per machine how many gears A carries (certified by the mirrors) and how many the roots
had to handle.
Usage: uv run python multi_mirror_walk.py qmax
"""
import sys, random
from sympy import primerange, isprime


def open_to(n, h): return bool(n % h) and bool((n + 2) % h)


def main():
    qmax = int(sys.argv[1]); primes = list(primerange(2, qmax + 1)); gears = [p for p in primes if p >= 5]
    # (1) composition law and (2) undercount, random walks
    random.seed(1); viol = 0; under = 0; trials = 0
    sets = [[2, 3], [2, 3, 5], [2, 3, 7], [2, 3, 5, 7], [2, 3, 11], [2, 3, 5, 11]]
    for _ in range(3000):
        n = -1; axes = []; step_cert = set(gears)
        for j in range(random.randint(1, 5)):
            S = random.choice(sets); M = 1
            for g in S: M *= g
            k = random.randint(1, 50); a = k * M; axes.append(a); n = 2 * a - n - 2
            step_cert &= set(h for h in gears if (2 * a) % h == 0)
        A = sum(a if j % 2 == len(axes) % 2 - 1 or (len(axes) % 2 == 1 and j % 2 == 0) else -a for j, a in enumerate(axes))
        # recompute A directly from the definition: alternating sum with the last axis positive
        A = 0
        for j, a in enumerate(reversed(axes)): A += a if j % 2 == 0 else -a
        expect = 2 * A - (-1) - 2 if len(axes) % 2 == 1 else -1 + 2 * A
        if expect != n: viol += 1
        law_cert = set(h for h in gears if A % h == 0)
        trials += 1
        # every gear dividing A must be open at n (home is open to all)
        if any(not open_to(n, h) for h in law_cert): viol += 1
        if law_cert - step_cert: under += 1
    print(f"(1) composition law (end = 2A - n - 2 or n + 2A, A the alternating sum; gears dividing A open at the end): {trials} random walks, violations {viol}")
    print(f"(2) stepwise tracking undercounts the certified set in {under} of {trials} walks")
    # (3) the two-step rule
    qs = [q for q in primes if q >= 11]
    ok = 0; fails = []; carried = []; handled = []
    for q in qs:
        rest_all = [h for h in gears if h <= q]
        A = 6 * ((q + 1) // 12 + 1)
        while 2 * A + 1 <= q * q:
            n = 2 * A - 1
            if n > q and all(open_to(n, h) for h in rest_all if A % h):
                break
            A += 6
        else:
            fails.append((q, 'no A')); continue
        n = 2 * A - 1
        # realise A = 42 k2 - 30 k1 with k1, k2 >= 1: 7 k2 - 5 k1 = A / 6
        m = A // 6; k2 = next(k for k in range(1, 6) if (7 * k - m) % 5 == 0); k1 = (7 * k2 - m) // 5
        while k1 < 1: k2 += 5; k1 = (7 * k2 - m) // 5
        n1 = 2 * (30 * k1) - (-1) - 2; n2 = 2 * (42 * k2) - n1 - 2
        assert n2 == n, (q, A, k1, k2, n1, n2, n)
        if isprime(n) and isprime(n + 2): ok += 1
        else: fails.append((q, n))
        carried.append(sum(1 for h in rest_all if A % h == 0)); handled.append(sum(1 for h in rest_all if A % h))
    print(f"(3) two-step rule home -> 30 k1 -> 42 k2 with A = 42 k2 - 30 k1 the smallest multiple of 6 avoiding the teeth of the gears not dividing A: machines {len(qs)}, landings twins {ok}, failures {fails[:5]}; gears carried by the mirrors per machine mean {sum(carried)/len(carried):.2f}, gears handled by the roots mean {sum(handled)/len(handled):.1f}")


if __name__ == "__main__":
    main()

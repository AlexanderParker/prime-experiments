"""Round 82 / loop entry 88: the counterexample hunt with prejudice, part A - a real adversary.

Round 81's greedy adversary left 5 to 9 percent of each window uncovered.  Greedy is weak.  This
runs simulated annealing with restarts over the free configuration - each gear h choosing two
classes modulo h - to minimise the number of uncovered columns in the window (q, q^2].  If the
minimum reaches 0 for some machine, a twin-free window is achievable by SOME configuration of the
same residues, and the machine's protection rests entirely on the rigid configuration the integers
actually take.  If it never reaches 0, the free configuration has an obstruction of its own.
"""

import math
import random
import sys


def primes_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def anneal(q, gears, iters, seed):
    rng = random.Random(seed)
    lo = q // 6 + 2  # strictly above q: 6 lo - 1 > q
    hi = (q * q - 1) // 6
    n = hi - lo + 1
    # count[i] = number of (gear, class) pairs covering column i
    classes = {h: [rng.randrange(h), rng.randrange(h)] for h in gears}
    count = [0] * n
    for h in gears:
        for r in classes[h]:
            k = lo + ((r - lo) % h)
            while k <= hi:
                count[k - lo] += 1
                k += h
    uncovered = sum(1 for c in count if c == 0)
    best = uncovered
    T = 2.0
    for it in range(iters):
        h = gears[rng.randrange(len(gears))]
        slot = rng.randrange(2)
        old = classes[h][slot]
        new = rng.randrange(h)
        if new == old:
            continue
        # delta: remove old class, add new class
        delta = 0
        k = lo + ((old - lo) % h)
        removed = []
        while k <= hi:
            count[k - lo] -= 1
            if count[k - lo] == 0:
                delta += 1
            removed.append(k)
            k += h
        k = lo + ((new - lo) % h)
        added = []
        while k <= hi:
            if count[k - lo] == 0:
                delta -= 1
            count[k - lo] += 1
            added.append(k)
            k += h
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            classes[h][slot] = new
            uncovered += delta
            if uncovered < best:
                best = uncovered
        else:
            for k in added:
                count[k - lo] -= 1
            for k in removed:
                count[k - lo] += 1
        T = max(0.05, 2.0 * (1 - it / iters))
    return n, best


def main():
    print("simulated annealing adversary: minimum uncovered columns over free configurations")
    print("     q   columns   gears   best of 6 restarts   greedy (round 81)   sum of 2/h")
    greedy = {29: 12, 37: 19, 47: 22, 59: 39, 71: 55, 101: 93}
    for q in (29, 37, 47, 59, 71, 101):
        gears = [h for h in primes_to(q) if h >= 5]
        best = None
        n = None
        for seed in range(6):
            n, b = anneal(q, gears, 60000, seed)
            best = b if best is None else min(best, b)
        s = sum(2.0 / h for h in gears)
        print("%6d  %8d  %6d  %19d  %18d  %10.3f" % (q, n, len(gears), best, greedy[q], s))
    print()
    print("a 0 in the best column would mean some free configuration kills every twin slot of that window")


if __name__ == "__main__":
    sys.exit(main())

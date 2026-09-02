"""The layered walk from q^2: landing_g(t) = landing under gears {5..g} from slot t, built
by the closure W_g = W_{g-} + hits of g:  x = landing_{g-}(t); while g hits x: x = landing_{g-}(x+1).
Run for every prime q <= Q from the slot holding q^2 under gears 5..q; record which layers
hop, how many times, the walk length, and check the landing is a twin prime pair.

Usage: layered_walk.py [--Q 5000]
"""
import argparse
import sys
from collections import Counter

sys.setrecursionlimit(10000)


def primes_upto(n):
    s = bytearray([1]) * (n + 1)
    s[0] = s[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13):
        if n % p == 0:
            return n == p
    i = 17
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def make_walker(gears):
    teeth = [(pow(6, -1, g), g) for g in gears]
    hops = Counter()
    calls = [0]

    def landing(i, t):
        # landing under gears[:i+1] from slot t
        calls[0] += 1
        if i < 0:
            return t
        u, g = teeth[i]
        x = landing(i - 1, t)
        while x % g == u or x % g == g - u:
            hops[g] += 1
            x = landing(i - 1, x + 1)
        return x

    return landing, hops, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=5000)
    a = ap.parse_args()
    ps = [p for p in primes_upto(a.Q) if p >= 5]
    layers_dist = Counter(); total_hops = Counter(); walk_lens = []
    small_share = []
    rows = []
    for idx, q in enumerate(ps):
        gears = ps[:idx + 1]
        landing, hops, calls = make_walker(gears)
        s = (q * q - 1) // 6  # slot holding q^2 (number 6s+1 = q^2)
        x = landing(len(gears) - 1, s)
        n1, n2 = 6 * x - 1, 6 * x + 1
        ok = is_prime(n1) and is_prime(n2)
        L = x - s
        nl = len(hops); nh = sum(hops.values())
        layers_dist[nl] += 1; total_hops[nh] += 1; walk_lens.append(L)
        small = sum(v for g, v in hops.items() if g <= 13)
        small_share.append((small, nh))
        rows.append((q, L, nl, nh, max(hops.values()) if hops else 0, sorted(hops.items()), ok, calls[0]))
        if q in (7, 11, 13, 37, 97, 499, 997, 4999) or not ok:
            print(f"q={q}: walk {L} slots, landing {n1}|{n2} twin={ok}, hopping layers {nl}, hops {nh}, "
                  f"per layer {sorted(hops.items())}, calls {calls[0]}")
    n = len(ps)
    print(f"\n{n} primes 5..{a.Q}: all landings twin: {all(r[6] for r in rows)}")
    print("hopping layers per walk: " + " ".join(f"{k}:{v}" for k, v in sorted(layers_dist.items())))
    print("total hops per walk:     " + " ".join(f"{k}:{v}" for k, v in sorted(total_hops.items())))
    mx = max(rows, key=lambda r: r[4])
    print(f"max hops at one layer: {mx[4]} (q={mx[0]}, layer hops {mx[5]})")
    sm = sum(s for s, _ in small_share); tot = sum(t for _, t in small_share)
    print(f"share of hops made by gears <= 13: {sm}/{tot} = {sm / tot:.3f}")
    # which layer sizes hop: hops by gear rank bucket
    by_gear = Counter()
    for r in rows:
        for g, v in r[5]:
            by_gear[g] += v
    top = sorted(by_gear.items(), key=lambda kv: -kv[1])[:8]
    print("hops by gear (top): " + " ".join(f"{g}:{v}" for g, v in top))
    lw = sorted(walk_lens)
    print(f"walk length: median {lw[len(lw) // 2]}, max {lw[-1]} at q={max(rows, key=lambda r: r[1])[0]}")
    # walk length vs number of gears hopping: is the walk just a sum of small hops?
    print("walk length vs hops (q, L, hopping layers, hops) for the 5 longest walks: "
          + "; ".join(f"({r[0]}, {r[1]}, {r[2]}, {r[3]})" for r in sorted(rows, key=lambda r: -r[1])[:5]))


if __name__ == "__main__":
    main()

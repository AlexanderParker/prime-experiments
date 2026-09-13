"""Sequential (settle one gear per step) walks: origin types and step rules, which land inside
the window (owner, 2026-09-13: find a consistent origin type with a consistent per-step rule).

Column k = the pair (12k - 1, 12k + 1); gear h strikes k iff h divides a member. Sequential walk:
gears taken in an order; at gear h, if the current k is struck, move by the stride D (a multiple
of every gear settled so far, so they stay settled), trying j in the given choices; then D := D h.
The final k is open to every gear by construction; success = the final landing lies in the
window (q, q^2] (then it is a twin).
Origins: zone start (first k with 12k - 1 > q); primorial-adjacent (the first multiple of
5 7 11 13 at or above the zone start, open to those four gears with no move, and the stride
starts at 5005); the square column ((g^2 - 1)/12 for g the first prime above sqrt q; the column
whose right member is g^2); the middle of the window.
Orders: ascending, descending.
Choices: j in 0, 1, 2; j in 0, -1, 1.
Reported per variant, machines 11 .. qmax: landings inside the window, and the gears that forced
the exit (the first move made with a stride larger than the window).
Usage: uv run python sequential_variants.py qmax
"""
import sys
from collections import Counter
from sympy import primerange, nextprime
from math import isqrt


def struck(k, h): return (12 * k - 1) % h == 0 or (12 * k + 1) % h == 0


def walk(k, gears, D0, choices):
    D = D0; exit_gear = None; moves = 0
    for h in gears:
        if D % h == 0:
            # h already settled by the origin's construction (it divides the stride); the landing is open to h automatically? only if the origin was; check
            if struck(k, h): return None, h, moves
            continue
        if struck(k, h):
            for j in choices:
                if j and not struck(k + j * D, h):
                    k += j * D; moves += 1; break
            else:
                return None, h, moves
        D *= h
    return k, exit_gear, moves


def main():
    qmax = int(sys.argv[1]); primes = list(primerange(5, qmax + 1)); qs = [q for q in primes if q >= 11]
    variants = []
    for oname in ('zone start', 'primorial-adjacent', 'square column', 'window middle'):
        for order in ('ascending', 'descending'):
            for cname, choices in (('0,1,2', (0, 1, 2)), ('0,-1,1', (0, -1, 1))):
                variants.append((oname, order, cname, choices))
    print("origin | order | choices | inside the window / machines | first exit forced by (gear: count) | mean moves")
    for oname, order, cname, choices in variants:
        inside = 0; forced = Counter(); mv = 0
        for q in qs:
            gears = [h for h in primes if h <= q]
            if order == 'descending': gears = gears[::-1]
            klo, khi = (q + 1) // 12 + 1, (q * q - 1) // 12
            if oname == 'zone start': k0, D0 = klo, 1
            elif oname == 'primorial-adjacent': k0, D0 = ((klo + 5004) // 5005) * 5005, 5005
            elif oname == 'square column':
                g = nextprime(isqrt(q)); k0, D0 = (g * g - 1) // 12, 1
            else: k0, D0 = (klo + khi) // 2, 1
            if oname == 'primorial-adjacent' and k0 > khi: k0, D0 = klo, 1   # small machines: no room, fall back
            k, ex, moves = walk(k0, gears, D0, choices); mv += moves
            if k is not None and klo <= k <= khi: inside += 1
            else:
                # which gear forced the exit: recompute the first move whose stride exceeded the window length
                D = D0; f = ex
                if k is not None:
                    kk = k0; D = D0
                    for h in gears:
                        if D % h == 0: continue
                        if struck(kk, h):
                            for j in choices:
                                if j and not struck(kk + j * D, h): kk += j * D; break
                            if D > khi - klo: f = h; break
                        D *= h
                forced[f] += 1
        print(f"{oname} | {order} | {cname} | {inside} / {len(qs)} | {forced.most_common(5)} | {mv/len(qs):.2f}")


if __name__ == "__main__":
    main()

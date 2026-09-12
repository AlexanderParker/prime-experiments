"""The locator: a fixed column offset after every square, open to the small gears by class.

Column offset i after the square g^2 (g a gear, g^2 = 1 mod 6): the column (g^2 - 2 + 6i,
g^2 + 6i). Offsets open to 5 for every g: i = 0 or 2 mod 5 (g^2 is 1 or 4 mod 5); open to 7
for every g: i = 3 or 5 mod 7 (g^2 is 1, 2 or 4 mod 7); both: i = 5, 10, 12, 17 mod 35. The
column at offset i is open to g itself for g > 6i. For any other gear h the column is struck
iff g^2 = -(6i - 2) or g^2 = -6i (mod h): each gear strikes the offset-i column of at most
four residue classes of g mod h (the square roots), and a gear for which neither value is a
quadratic residue never strikes that offset from any square (a blind gear for the offset).
So "the offset-i column after g^2 is a twin" is a sieve on the gear line g, at most four
classes removed per gear, none for the blind gears.

Reported: (1) per offset i in {2, 5, 10, 12, 17}: the gears below 200 that are blind for it,
and the mean number of classes removed per gear; (2) per machine q (primes to qmax): whether
some gear g with sqrt(q) < g <= q has its offset-i column a twin (the located twin lies in the
window), how many such g, and the smallest; (3) the machines with none, per offset, and with
none over all five offsets.
Usage: uv run python locator.py qmax
"""
import sys
from sympy import primerange, isprime, sqrt_mod


def classes(h, i):
    """residue classes of g mod h whose offset-i column is struck by h"""
    out = set()
    for v in (-(6 * i - 2), -6 * i):
        r = sqrt_mod(v % h, h, all_roots=True)
        if r: out |= set(r)
    return out


def main():
    qmax = int(sys.argv[1]); offsets = [2, 5, 10, 12, 17]
    gears200 = list(primerange(5, 200))
    print("(1) offset i | blind gears below 200 (never strike the offset-i column of any square) | mean classes removed per gear")
    for i in offsets:
        blind = [h for h in gears200 if not classes(h, i)]
        mean = sum(len(classes(h, i)) for h in gears200) / len(gears200)
        print(f"   i = {i}: blind {blind}; mean classes {mean:.2f}")
    primes = list(primerange(5, qmax + 1))
    hit = {i: {g for g in primes if isprime(g * g - 2 + 6 * i) and isprime(g * g + 6 * i)} for i in offsets}
    print("(2) per offset: machines q (primes 7..qmax) with a located twin in the window; the first few machines with none; the smallest g at q = 1009 and 10007 if in range")
    none_all = []
    for q in primes[1:]:
        ok = False
        for i in offsets:
            if any(g * g > q and g <= q for g in hit[i]): ok = True
        if not ok: none_all.append(q)
    for i in offsets:
        none = [q for q in primes[1:] if not any(g * g > q and g <= q for g in hit[i])]
        ex = {}
        for Q in (1009, 10007):
            if Q <= qmax: ex[Q] = min((g for g in hit[i] if g * g > Q and g <= Q), default=None)
        print(f"   i = {i}: machines with a located twin {len(primes) - 1 - len(none)} of {len(primes) - 1}; none at {none[:12]}{' ...' if len(none) > 12 else ''}; smallest g {ex}")
    print(f"(3) machines with no located twin over all five offsets: {none_all[:20]}{' ...' if len(none_all) > 20 else ''} ({len(none_all)} of {len(primes) - 1})")
    print("    gears g whose offset-2 column is a twin, first 20:", sorted(hit[2])[:20])
    print("    gears g whose offset-5 column is a twin, first 20:", sorted(hit[5])[:20])


if __name__ == "__main__":
    main()

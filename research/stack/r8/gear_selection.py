"""Search for a gear-selection rule (owner, 2026-09-13): one flip from an origin pair about an
axis k M, M = 6 times a set of extra gears chosen from the machine by a rule (q, the gear before
q, 5, ...), k chosen by a rule, that lands on a twin inside the window (q, q^2] at every machine.
No test of the landing: the rule fixes the origin, the gears and the multiple; the landing is
certified afterwards.

Origin candidates: home (-1, 1); the gear pairs (5, 7), (11, 13), (17, 19), (29, 31), (41, 43),
(59, 61), (71, 73), (101, 103) when below q; the largest gear pair below q.
Extra gears E (the axis is k * 6 * prod E): none; {q}; {p-} (the gear before q); {q, p-};
{p-, p--}; {5}; {5, q}; {7}; {5, 7}; {5, 7, q}.
Multiple k: 1, 2, 3, 5, q, p-, and "the smallest k with the landing above q".
Landing from origin n about the axis a: (2a - n - 2, 2a - n). Success: both members prime and
q < 2a - n - 2, 2a - n <= q^2.
Reported: for every (origin, E, k) rule the machines succeeded of all machines 11 .. qmax, the
top rules, and whether any rule is perfect.
Usage: uv run python gear_selection.py qmax
"""
import sys
from sympy import primerange, prevprime, isprime


def main():
    qmax = int(sys.argv[1]); qs = list(primerange(11, qmax + 1))
    pairs = [5, 11, 17, 29, 41, 59, 71, 101]
    origins = [('home', lambda q: -1)] + [(f'({g}, {g+2})', (lambda g: (lambda q: g if g + 2 <= q else None))(g)) for g in pairs] + \
              [('largest pair below q', lambda q: max(g for g in primerange(5, q - 1) if isprime(g + 2) and g + 2 <= q))]
    extras = [('none', lambda q: []), ('{q}', lambda q: [q]), ('{p-}', lambda q: [prevprime(q)]), ('{q, p-}', lambda q: [q, prevprime(q)]),
              ('{p-, p--}', lambda q: [prevprime(q), prevprime(prevprime(q))]), ('{5}', lambda q: [5]), ('{5, q}', lambda q: [5, q]),
              ('{7}', lambda q: [7]), ('{5, 7}', lambda q: [5, 7]), ('{5, 7, q}', lambda q: [5, 7, q])]
    ks = [('k=1', lambda q, M, n: 1), ('k=2', lambda q, M, n: 2), ('k=3', lambda q, M, n: 3), ('k=5', lambda q, M, n: 5),
          ('k=q', lambda q, M, n: q), ('k=p-', lambda q, M, n: prevprime(q)),
          ('k=first above q', lambda q, M, n: (q + n + 2) // (2 * M) + 1)]
    results = []
    for oname, ofun in origins:
        for ename, efun in extras:
            for kname, kfun in ks:
                ok = 0; tried = 0
                for q in qs:
                    n = ofun(q)
                    if n is None: continue
                    M = 6
                    for e in efun(q): M *= e
                    k = kfun(q, M, n); a = k * M; L = 2 * a - n - 2
                    tried += 1
                    if q < L and L + 2 <= q * q and isprime(L) and isprime(L + 2): ok += 1
                if tried: results.append((ok / tried, ok, tried, oname, ename, kname))
    results.sort(reverse=True)
    print(f"machines 11..{qmax}: {len(qs)}; rules tried {len(results)}; perfect rules: {sum(1 for r in results if r[1] == r[2])}")
    print("top rules: success share | succeeded / tried | origin | extra gears | multiple")
    for r in results[:15]: print(f"   {r[0]:.3f} | {r[1]} / {r[2]} | {r[3]} | {r[4]} | {r[5]}")
    # the landing position of the best rule and the share by size
    best = results[0]
    print(f"best rule landing positions (2a / q^2) and success by decade for {best[3]}, {best[4]}, {best[5]}:")
    ofun = dict(origins)[best[3]]; efun = dict(extras)[best[4]]; kfun = dict(ks)[best[5]]
    for lo, hi in ((11, 100), (100, 300), (300, 1000), (1000, qmax + 1)):
        ok = tried = 0; pos = []
        for q in [x for x in qs if lo <= x < hi]:
            n = ofun(q)
            if n is None: continue
            M = 6
            for e in efun(q): M *= e
            a = kfun(q, M, n) * M; L = 2 * a - n - 2; tried += 1; pos.append(2 * a / (q * q))
            if q < L and L + 2 <= q * q and isprime(L) and isprime(L + 2): ok += 1
        if tried: print(f"   q in [{lo}, {hi}): {ok} / {tried}; landing at {sum(pos)/len(pos):.4f} of q^2 on average")


if __name__ == "__main__":
    main()

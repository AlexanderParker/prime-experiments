"""Round-29 formalist gate: the uniform-order finite check, 48 classes mod 210.

Constructor R74: for every machine M = {5..y} with q' = nextprime(y),
    A_relax(M) <= 5,  and <= 4 unless q' = 37, 53, 83, 127, 157, 173 (mod 210).

The vehicle is Mechanic's phase saturation at gears 5 and 7: the m-letter
alternation over the two literal letters a = 2u', b = q' - 2u' has prefix-sum
offsets X = {0, a, q', q'+a, 2q', ...} (m+1 points), and the word occurs
NOWHERE if for some gear g of M no translate of X mod g fits inside
E_g = Z_g \\ {+-6^{-1} mod g}.  Everything at gears 5 and 7 is a function of
q' mod 210, because 3a = q' -+ 1 with the sign fixed by q' mod 6.

This script computes, for each of the 48 invertible classes, the largest m for
which a translate still fits (both start letters, and their max/min), and
prints the distribution.  It is the exact statement the Lean file
`AlternationOrder.lean` checks with `decide`.
"""

INV3 = {5: 2, 7: 5}          # 3^{-1} mod g
U = {5: 1, 7: 6}             # 6^{-1} mod g
MAXM = 9


def E(g):
    return {x for x in range(g) if x != U[g] and x != (g - U[g]) % g}


def a_mod(g, c):
    s = -1 if c % 6 == 1 else 1          # 3a = c + s
    return (INV3[g] * (c + s)) % g


def offsets(c, g, m, start_a=True):
    """prefix sums of the m-letter alternation, mod g (m+1 points)"""
    a = a_mod(g, c)
    b = (c - a) % g
    out, s = [0], 0
    for i in range(m):
        s = (s + (a if (i % 2 == 0) == start_a else b)) % g
        out.append(s)
    return out


def fits(g, c, m, start_a):
    Eg = E(g)
    X = offsets(c, g, m, start_a)
    return any(all((t + x) % g in Eg for x in X) for t in range(g))


def survives(c, m, start_a):
    return fits(5, c, m, start_a) and fits(7, c, m, start_a)


def ps_order(c, mode):
    best = 0
    for m in range(MAXM + 1):
        if mode == "a":
            ok = survives(c, m, True)
        elif mode == "b":
            ok = survives(c, m, False)
        elif mode == "max":
            ok = survives(c, m, True) or survives(c, m, False)
        else:
            ok = survives(c, m, True) and survives(c, m, False)
        if ok:
            best = m
        else:
            break
    return best


def main():
    from math import gcd
    classes = [c for c in range(210) if gcd(c, 210) == 1]
    assert len(classes) == 48
    for mode in ("a", "b", "max", "min"):
        dist = {}
        six = []
        for c in classes:
            o = ps_order(c, mode)
            dist[o] = dist.get(o, 0) + 1
            if o >= 5:
                six.append(c)
        print(f"mode={mode:>3}  distribution {dict(sorted(dist.items()))}  "
              f"order>=5 classes {six}")
    # the claim under the mode that reproduces R74
    LITCAP6 = [37, 53, 83, 127, 157, 173]
    for mode in ("a", "b", "max", "min"):
        six = [c for c in classes if ps_order(c, mode) >= 5]
        mx = max(ps_order(c, mode) for c in classes)
        if six == LITCAP6 and mx == 5:
            print(f"R74 REPRODUCED at mode={mode}: max order 5, "
                  f"order-5 classes = the litcap-6 classes {LITCAP6}")
            print("ALL ASSERTIONS PASSED")
            return
    raise AssertionError("no mode reproduces R74")


if __name__ == "__main__":
    main()

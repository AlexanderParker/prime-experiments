"""Round 31 (Formalist) gate: the bare-alternation inadmissible set S.

Independent re-implementation of what `proofs/BareAlternation.lean` decides in
the kernel, asserted against the kernel's own emitted list.  The Lean vehicle
scans `List.range g` with a recursive prefix-sum `offsets`; this one builds the
exposed set `E_g = Z_g \\ {u, g-u}` as a Python set and the offsets with
`itertools.accumulate`.

  a = (c -+ 1)/3   (the bare letter of the class `+d'`, `d' = 2u'`)
  b = c - a        (the bare letter of `-d'`)
  the bare m-letter alternation is admissible iff SOME translate of its
  prefix-sum offsets avoids the teeth of gear 5 AND of gear 7, for at least
  one of the two start letters.

Usage:  uv run python research/bare_alt_r31.py
Exit 0 and "ALL ASSERTIONS PASSED" is the gate.
"""

from itertools import accumulate
from math import gcd

# gear -> u = 6^{-1} mod gear; teeth are {u, gear - u}
TEETH_U = {5: 1, 7: 6}

# The kernel's list, copied verbatim from `BareAlt.S` in
# proofs/BareAlternation.lean (theorem `bareAlt_inadmissible_iff`).
S_KERNEL = [11, 13, 17, 19, 29, 41, 43, 47, 59, 71, 73, 79, 101, 103, 107,
            109, 131, 137, 139, 151, 163, 167, 169, 181, 191, 193, 197, 199]

# Corpus L(M) (Constructor R100 / R81), M = 11 .. 53.
CORPUS = [
    # M, q', L(M)
    (11, 13, 1), (13, 17, 1), (17, 19, 1), (19, 23, 2), (23, 29, 1),
    (29, 31, 3), (31, 37, 3), (37, 41, 2), (41, 43, 2), (43, 47, 2),
    (47, 53, 4), (53, 59, 3),
]


def exposed(g):
    u = TEETH_U[g]
    return set(range(g)) - {u, g - u}


def a_of_class(c):
    return (c - 1) // 3 if c % 6 == 1 else (c + 1) // 3


def alt_offsets(first, second, m):
    """Prefix sums of the m-letter alternation (m+1 points)."""
    word = [first if i % 2 == 0 else second for i in range(m)]
    return [0] + list(accumulate(word))


def fits(g, offs):
    E = exposed(g)
    return any(all((t + o) % g in E for o in offs) for t in range(g))


def bare_fits(first, second, m):
    offs = alt_offsets(first, second, m)
    return fits(5, offs) and fits(7, offs)


def admissible(a, b, m):
    return bare_fits(a, b, m) or bare_fits(b, a, m)


def bare_cap(c):
    """Longest bare alternation length (in LETTERS) that is admissible."""
    m = 0
    while admissible(a_of_class(c), c - a_of_class(c), m + 1):
        m += 1
    return m


def main():
    classes = [c for c in range(210) if gcd(c, 210) == 1]
    assert len(classes) == 48, len(classes)

    S = [c for c in classes if not admissible(a_of_class(c),
                                              c - a_of_class(c), 3)]
    print(f"|S| = {len(S)}")
    print("S =", S)
    assert S == S_KERNEL, ("S disagrees with the kernel list", S, S_KERNEL)
    print("GATE 1  S == BareAlt.S (kernel)                       OK")

    # the two one-start-letter sets, for the record
    SA = [c for c in classes
          if not bare_fits(a_of_class(c), c - a_of_class(c), 3)]
    SB = [c for c in classes
          if not bare_fits(c - a_of_class(c), a_of_class(c), 3)]
    print(f"|S_A| = {len(SA)}  |S_B| = {len(SB)}  |S_A and S_B| = {len(S)}")
    assert len(SA) == 32 and len(SB) == 36
    print("GATE 2  |S_A| = 32, |S_B| = 36                        OK")

    # the mirror c -> 210 - c: a -> 70 - a and b -> 140 - b, so both offset
    # sets negate mod 35 and the teeth pairs are closed under negation - all
    # three sets are mirror-closed, with the SAME start letter.
    assert set(210 - c for c in S) == set(S)
    assert set(210 - c for c in SA) == set(SA)
    assert set(210 - c for c in SB) == set(SB)
    print("GATE 3  S, S_A, S_B all mirror-closed (c -> 210 - c)  OK")

    # cross-check against round 29's psMin distribution (AlternationOrder)
    def surv(c, m, both):
        a, b = a_of_class(c), c - a_of_class(c)
        f = [bare_fits(a, b, m - 1), bare_fits(b, a, m - 1)]
        return all(f) if both else any(f)

    ps_min = {c: sum(1 for m in range(1, 10) if surv(c, m, True))
              for c in classes}
    dist = {k: sum(1 for c in classes if ps_min[c] == k) for k in (2, 3, 4, 5)}
    assert dist == {2: 24, 3: 16, 4: 2, 5: 6}, dist
    assert [c for c in classes if ps_min[c] == 5] == [37, 53, 83, 127, 157, 173]
    print("GATE 4  psMin 24/16/2/6, order-5 = litcap six         OK")

    # the numerals BareAltInst decides
    assert a_of_class(29) == 10 and 29 - a_of_class(29) == 19
    assert a_of_class(41) == 14 and 41 - a_of_class(41) == 27
    assert not admissible(10, 19, 3)
    assert not admissible(14, 27, 2)
    assert not fits(5, [0, 14, 43])
    assert not fits(5, [0, 16, 47])
    print("GATE 5  the m23 / m37 / m41 / m43 numerals            OK")

    # Constructor's PSORD table (docs/novel/bare-word-uniform-cap.md 1.3),
    # kernel-checked in BareAlternation.psord_eq_one_iff / _two_iff / _five_iff
    psord = {c: bare_cap(c) for c in classes}
    by_val = {v: [c for c in classes if psord[c] == v] for v in (1, 2, 3, 4, 5)}
    assert len(by_val[1]) == 24 and len(by_val[2]) == 4
    assert len(by_val[3]) == 14 and by_val[4] == [] and len(by_val[5]) == 6
    assert by_val[2] == [29, 59, 151, 181]
    assert by_val[5] == [37, 53, 83, 127, 157, 173]
    assert sorted(by_val[1] + by_val[2]) == S
    print("GATE 6  PSORD 24/4/14/0/6, S = {PSORD <= 2}          OK")

    print()
    print("  M    q'   a   b   q' in S   bare cap   L(M)   L > bare cap?")
    for M, q, L in CORPUS:
        c = q % 210
        a, b = a_of_class(c), c - a_of_class(c)
        cap = bare_cap(c)
        print(f" {M:3}  {q:3}  {a:3} {b:3}   {str(c in S):>5}      "
              f"{cap:5}   {L:4}   {'YES' if L > cap else '-'}")
    over = [M for M, q, L in CORPUS if L > bare_cap(q % 210)]
    assert over == [37, 41, 43, 53], over
    print("GATE 7  L exceeds the bare cap at exactly M = 37, 41, 43, 53  OK")

    print()
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

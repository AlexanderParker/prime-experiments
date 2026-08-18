"""Constructor round 9: the endpoint/adjacency corridor laws vs lemma 1
(top-gap anti-clustering, F2 - F <= alpha1 * q).

(Formerly research/topgap_corridor.py; renamed - Lateral's round-9
neighbourhood analysis now owns that filename. Frames: k-frame throughout;
adjacent/halved = 3 x k-frame.)

Corridor laws PROVEN here (bounded-modulus, gears 5,7; E = 15 exposed
residues mod 35):
  ENDPOINT LAW: a machine gap of length G has both endpoints exposed, so its
  left endpoint residue lies in A(G) = {r in E : r + G mod 35 in E}.
  |A| ranges 3..15 by shift; G = 34 mod 35 forces {3, 18, 33}.
  ADJACENCY LAW: adjacent gaps (G1, G2) force the triple r, r+G1, r+G1+G2
  into E: allowed set A3(G1, G2); 294 of 1225 length-pairs mod 35 have
  A3 empty - forbidden adjacencies from gears 5,7 alone (e.g. (1,1)).
  ESCAPE FACT (the negative): every (G1, G2) is within L1 distance 1 of an
  allowed pair - residue laws cannot cap sizes.

Verified on full periods y = 11..23 (endpoint law + adjacency law hold at
every recorded gap; F2 values reproduce corpus F2(2,y) = 33, 48, 75, 93, 117
exactly); near-max census and separations measured.
"""
import numpy as np
from math import prod

GEARS = {11: [5, 7, 11], 13: [5, 7, 11, 13], 17: [5, 7, 11, 13, 17],
         19: [5, 7, 11, 13, 17, 19], 23: [5, 7, 11, 13, 17, 19, 23]}


def exposed_mod(gears):
    m = prod(gears)
    arr = np.ones(m, bool)
    for q in gears:
        c = pow(6, -1, q)
        arr[c::q] = False
        arr[(q - c) % q::q] = False
    return arr


E_SET = set(np.flatnonzero(exposed_mod([5, 7])).tolist())


def A(shift):
    return {r for r in E_SET if (r + shift) % 35 in E_SET}


def A3(s1, s2):
    return {r for r in E_SET if (r + s1) % 35 in E_SET
            and (r + s1 + s2) % 35 in E_SET}


def machine(y):
    gears = GEARS[y]
    P = prod(gears)
    idx = np.flatnonzero(exposed_mod(gears))
    gaps = np.diff(np.append(idx, idx[0] + P))
    F = int(gaps.max())
    pair = gaps + np.roll(gaps, -1)
    F2 = int(pair.max())
    q_next = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29}[y]
    print(f"\n=== gears<={y}  P={P:,}  F_k={F}  F2_k={F2}  F2-F={F2-F}  "
          f"adj (F2-F)*3/q_next = {3*(F2-F)/q_next:.3f}")
    for d in (0, 2, (q_next - 1) // 3):
        sel = np.flatnonzero(gaps >= F - d)
        lefts = idx[sel]
        ok = all((int(a) % 35) in A(int(g) % 35)
                 for a, g in zip(lefts, gaps[sel]))
        minsep = int(np.diff(np.sort(lefts)).min()) if len(sel) > 1 else P
        print(f"  gaps>=F-{d}: n={len(sel):>4} left-res mod35 "
              f"{sorted({int(a) % 35 for a in lefts})} law "
              f"{'OK' if ok else 'VIOLATED'} minsep {minsep:,} "
              f"({minsep/P:.2%} of period)")
    j = int(np.argmax(pair))
    g1, g2 = int(gaps[j]), int(gaps[(j + 1) % len(gaps)])
    a = int(idx[j]) % 35
    al = A3(g1 % 35, g2 % 35)
    print(f"  F2 pair ({g1},{g2}) left-res {a}; A3 = {sorted(al)} "
          f"{'OK' if a in al else 'VIOLATED'}")


def a3_zero_table():
    zeros = [(s1, s2) for s1 in range(35) for s2 in range(35)
             if not A3(s1, s2)]
    print(f"\nA3 zeros mod 35: {len(zeros)}/1225 forbidden; "
          f"first: {zeros[:8]}")
    worst = max(min(abs(d1) + abs(d2)
                    for d1 in range(-17, 18) for d2 in range(-17, 18)
                    if A3((s1 + d1) % 35, (s2 + d2) % 35))
                for s1 in range(35) for s2 in range(35))
    print(f"escape distance (L1) to nearest allowed pair: max = {worst}")


def local_capacity():
    """Two-scale local capacity on top of a corridor base: exact, and where
    it dies. Base gears B (density rho of exposed), killers K = rest: a span
    S of consecutive slots fully covered needs rho*S - 1 <= sum 2*ceil(S/q),
    so S <= (2*#K + 1)/(rho - 2*sum 1/q) when positive."""
    print("\nlocal two-scale capacity caps on F2_k (base | killers -> cap):")
    for base_y, y in ((7, 11), (7, 13), (7, 17), (17, 23), (17, 29)):
        base = [g for g in [5, 7, 11, 13, 17] if g <= base_y]
        kill = [g for g in [5, 7, 11, 13, 17, 19, 23, 29] if base_y < g <= y]
        rho = float(np.mean(exposed_mod(base)))
        s = sum(1 / q for q in kill)
        margin = rho - 2 * s
        cap = (2 * len(kill) + 1) / margin if margin > 0 else float("inf")
        print(f"  base {{5..{base_y}}} rho={rho:.4f} | kill {kill} "
              f"2*sum1/q={2*s:.4f} -> "
              f"{'VACUOUS' if margin <= 0 else f'F2_k({y}) <= {cap:.0f}'}")


if __name__ == "__main__":
    print(f"E mod 35: {sorted(E_SET)}")
    print(f"A(34)={sorted(A(34))} A(11)={sorted(A(11))} A(25)={sorted(A(25))}")
    for y in (11, 13, 17, 19, 23):
        machine(y)
    a3_zero_table()
    local_capacity()

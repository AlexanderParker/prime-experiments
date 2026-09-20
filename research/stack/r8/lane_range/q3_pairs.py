"""Q3: two gears g < g' of T.  Coincidences (columns struck by both) are exactly the four CRT
residues (e c_g mod g, e' c_g' mod g'), e,e' in {+-1}; same-sign pairs are the two classes of
the composite gear gg' (gg' | 6n-1 or gg' | 6n+1), crossed-sign pairs are g | 6n-e, g' | 6n+e.
In one period: at most 4 ceil(P/gg') coincidences, at most 4 when gg' > P.
Coincidences on openings of q: in the coordinate m (n = r + m gg') they are the openings of the
twisted machine with teeth (gg')^{-1}(+-c_h - r) mod h, tooth distance (3gg')^{-1} mod h.
"""
import numpy as np
from math import ceil
from itertools import combinations
from lane_common import gears, period, c, openings, T_primes


def crt(r1, m1, r2, m2):
    return (r1 + m1 * ((r2 - r1) * pow(m1, -1, m2) % m2)) % (m1 * m2)


def main(q=13):
    P = period(q)
    O = openings(q)
    Oset = set(O.tolist())
    T = T_primes(q)
    gs = gears(q)
    cols = np.arange(1, P + 1)
    print(f"q={q} P={P} T={T[0]}..{T[-1]} ({len(T)} gears, {len(T)*(len(T)-1)//2} pairs)")
    maxco = 0
    n_big = 0
    n_bigco = []
    example = None
    for g, g2 in combinations(T, 2):
        cg, cg2 = c(g), c(g2)
        sg = (cols % g == cg) | (cols % g == g - cg)
        sg2 = (cols % g2 == cg2) | (cols % g2 == g2 - cg2)
        co = cols[sg & sg2]
        M = g * g2
        res = {crt(e * cg % g, g, e2 * cg2 % g2, g2): (e, e2) for e in (1, -1) for e2 in (1, -1)}
        assert len(res) == 4
        # exact residue description
        assert set((co % M).tolist()) <= set(res)
        pred = np.concatenate([np.arange(r if r else M, P + 1, M) for r in res])
        assert set(pred.tolist()) == set(co.tolist()), f"residue description fails {g},{g2}"
        # sign meaning
        for n in co[:8]:
            e, e2 = res[int(n) % M]
            assert (6 * int(n) - e) % g == 0 and (6 * int(n) - e2) % g2 == 0
            if e == e2:
                assert (6 * int(n) - e) % M == 0
        # bound
        assert len(co) <= 4 * ceil(P / M)
        if M > P:
            n_big += 1
            assert len(co) <= 4
            n_bigco.append(len(co))
        maxco = max(maxco, len(co))
        # coincidences on openings: twisted machine in m
        for r, (e, e2) in res.items():
            ns = np.arange(r if r else M, P + 1, M)
            m_idx = (ns - r) // M
            on_open = np.array([int(n) in Oset for n in ns], dtype=bool)
            Minv = {h: pow(M, -1, h) for h in gs}
            pred_open = np.ones(len(ns), dtype=bool)
            for h in gs:
                t1 = (Minv[h] * (c(h) - r)) % h
                t2 = (Minv[h] * (-c(h) - r)) % h
                assert (t1 - t2) % h == pow(3 * M, -1, h)
                pred_open &= (m_idx % h != t1) & (m_idx % h != t2)
            assert np.array_equal(on_open, pred_open), f"twisted rule fails {g},{g2},{r}"
        if example is None and g == 17 and g2 == 19:
            example = (g, g2, res, co.tolist(), [int(n) for n in co if int(n) in Oset])
    g, g2, res, co, coo = example
    print(f"  example g,g'={g},{g2}: residues mod {g*g2}: {res}; coincidences in [1,P]: {co}; on openings: {coo}")
    print(f"  all pairs: residue description exact; bound 4ceil(P/gg') holds; max coincidences per period = {maxco} (check)")
    print(f"  pairs with gg' > P: {n_big}; their coincidence counts: {sorted(set(n_bigco))} (all <= 4) (check)")
    print(f"  twisted-machine rule for coincidences on openings PASS for all pairs and all four residues")


if __name__ == "__main__":
    main()

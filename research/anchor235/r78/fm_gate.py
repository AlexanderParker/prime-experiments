"""fm_gate.py -- P1, P2, P5: the gates. Machine 2's eight twin gears in [9, 121); the monoid M = <5, P_+> at
[25, 961) (20 columns, 74 irreducibles, 0 twin gear pairs) and [961, 935089) (9,985 columns, 36,812 irreducibles);
the class reading (every irreducible in the sections is = 1 mod 6, every column has 5 | L); and T3's residue
check 6^{-1} mod g = k_g iff g = 5 (mod 6). Output: results/gate.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fm_common import primes_upto, spf_table, in_monoid_mask, k_of, inv6, chain_from

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def main():
    out = {}
    # P1a: machine 2 = {5, 7} on [9, 121): columns 2..19, open ones
    cols = list(range(2, 20))
    open_cols = [j for j in cols if all((6 * j - 1) % g and (6 * j + 1) % g for g in (5, 7))]
    out["machine2_open_columns"] = open_cols
    out["machine2_twins"] = [(6 * j - 1, 6 * j + 1) for j in open_cols]
    print("machine 2 open columns:", open_cols, "expected [2, 3, 5, 7, 10, 12, 17, 18]")
    assert open_cols == [2, 3, 5, 7, 10, 12, 17, 18]

    # P1b, P2: the monoid M = <5, P_+>
    N = 1_000_000
    spf = spf_table(N)
    P = primes_upto(N)
    gen = np.zeros(N + 1, dtype=bool)
    gen[P[(P % 6 == 1)]] = True
    gen[5] = True
    inM = in_monoid_mask(spf, gen)
    G = np.flatnonzero(gen)
    chain = chain_from(G, 5, N)
    print("chain of M:", chain[:4])
    assert [c for c, p in chain[1:4]] == [25, 961, 935089], chain[:4]
    secs = []
    for (lo, p), (hi, _) in zip(chain[1:], chain[2:]):
        irr = [int(g) for g in G if lo < g < hi]
        js = np.arange(lo // 6 + 1, (hi - 1) // 6 + 1)
        L = 6 * js - 1
        R = 6 * js + 1
        inside = (L > lo) & (R < hi)
        both = inside & inM[L] & inM[R]
        cols = js[both]
        # twin gear pairs: both members in G (= both prime, since G is a set of primes)
        twin = [int(j) for j in cols if gen[6 * j - 1] and gen[6 * j + 1]]
        five_div_L = int(np.sum((6 * cols - 1) % 5 == 0))
        all_1mod6 = all(g % 6 == 1 for g in irr)
        rec = dict(lo=lo, hi=hi, columns=int(len(cols)), irreducibles=len(irr), twin_gear_pairs=len(twin),
                   first_columns=[(int(6 * j - 1), int(6 * j + 1)) for j in cols[:4]],
                   irreducibles_all_1mod6=all_1mod6, columns_with_5_dividing_L=five_div_L)
        secs.append(rec)
        print("M section", rec)
    out["M_sections"] = secs
    assert secs[0]["columns"] == 20 and secs[0]["irreducibles"] == 74 and secs[0]["twin_gear_pairs"] == 0
    assert secs[1]["columns"] == 9985 and secs[1]["irreducibles"] == 36812 and secs[1]["twin_gear_pairs"] == 0
    assert secs[0]["first_columns"] == [(35, 37), (65, 67), (95, 97), (125, 127)]

    # P5: 6^{-1} mod g against k_g, all primes 5 <= g <= 1e5
    bad = []
    for g in primes_upto(100_000):
        g = int(g)
        if g < 5:
            continue
        r = inv6(g)
        k = k_of(g)
        want = k if g % 6 == 5 else g - k
        if r != want:
            bad.append((g, r, k))
    out["P5_exceptions"] = bad
    print("P5 exceptions (6^{-1} mod g vs k_g / g - k_g):", len(bad))
    assert not bad
    # the residues in words, for the document: first gears
    tbl = []
    for g in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
        r = inv6(g)
        tbl.append(dict(g=g, cls=1 if g % 6 == 1 else -1, k=k_of(g), left_residue=r, right_residue=(-r) % g,
                        left_member_at_left_residue=6 * r - 1, right_member_at_right_residue=6 * ((-r) % g) + 1))
    out["tooth_table"] = tbl
    for t in tbl:
        print(t)
    with open(os.path.join(RES, "gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("GATE OK")


if __name__ == "__main__":
    main()

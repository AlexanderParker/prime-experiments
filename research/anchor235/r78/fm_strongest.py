r"""fm_strongest.py -- the strongest thinning that breaks 8 on the chain from 5 (correcting the pre-registered form,
which removed 5 itself and moved the chain to 7), and the exhibited strike maps of the smallest obstruction sections.

  T = P_- \ (W in (25, 961)): 31 twin lowers removed, 5 kept; chain 5, 31, 967; section [25, 961) predicted 0 twin gear pairs.
  T = P_- \ (W in (25, 961)) + {857}: one twin lower put back; predicted 1.
  T = P_- \ (W in (25, 841)): 29 removed; chain 5, 31 (29 removed), 967; [25, 961) keeps 857, 881: predicted 2.
Strike maps: G = P_+ u (P_- \ W) on [49, 2809) (first 12 columns of M_G, each member with its least generator);
             G = P_+ u {23 mod 30} on [49, 2809) (all 25 columns).
Output: results/strongest.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fm_common import primes_upto, spf_table, in_monoid_mask, chain_from
from fm_thinned import analyse, section_count

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")


def strike_map(G, N, spf, lo, hi, limit):
    gen = np.zeros(N + 1, dtype=bool)
    gen[G[G <= N]] = True
    inM = in_monoid_mask(spf, gen)
    js = np.arange(lo // 6 + 1, (hi - 1) // 6 + 1, dtype=np.int64)
    L, R = 6 * js - 1, 6 * js + 1
    ok = (L > lo) & (R < hi) & inM[L] & inM[R]
    rows = []
    for j in js[ok][:limit]:
        l, r = int(6 * j - 1), int(6 * j + 1)
        def fac(n):
            fs = []
            while n > 1:
                p = int(spf[n]); fs.append(p); n //= p
            return fs
        rows.append(dict(column=int(j), L=l, L_factors=fac(l), R=r, R_factors=fac(r)))
    return rows, int(ok.sum())


def main():
    N = 2_000_000
    spf = spf_table(N)
    P = primes_upto(N).astype(np.int64)
    P = P[P >= 5]
    Pm = P[P % 6 == 5]
    Pp = P[P % 6 == 1]
    isP = np.zeros(N + 3, dtype=bool)
    isP[P] = True
    W = Pm[isP[Pm + 2]]
    Wset = set(int(t) for t in W)
    log = open(os.path.join(RES, "strongest.log"), "w")
    out = {}
    variants = {
        "T = P_- \\ (W in (25, 961))": np.array([t for t in Pm if not (int(t) in Wset and 25 < t < 961)]),
        "T = P_- \\ (W in (25, 961)) + {857}": np.array([t for t in Pm if not (int(t) in Wset and 25 < t < 961) or t == 857]),
        "T = P_- \\ (W in (25, 841))": np.array([t for t in Pm if not (int(t) in Wset and 25 < t < 841)]),
    }
    removed = sorted(t for t in Wset if 25 < t < 961)
    out["removed_twin_lowers_25_961"] = removed
    print("twin lowers in (25, 961):", len(removed), removed)
    for name, T in variants.items():
        out[name] = analyse(name, T.astype(np.int64), Pp, spf, P, W, N, log)
    # strike maps
    G1 = np.union1d(Pp, np.array([t for t in Pm if int(t) not in Wset]))
    rows, ncol = strike_map(G1, N, spf, 49, 2809, 12)
    out["map_no_twin_lower_49_2809"] = dict(columns=ncol, first=rows)
    print("G = P_+ u (P_- \\ W), [49, 2809):", ncol, "columns; first 12:")
    for r in rows:
        print("  ", r)
    G2 = np.union1d(Pp, Pm[Pm % 30 == 23])
    rows2, ncol2 = strike_map(G2, N, spf, 49, 2809, 25)
    out["map_23mod30_49_2809"] = dict(columns=ncol2, all=rows2)
    print("G = P_+ u {23 mod 30}, [49, 2809):", ncol2, "columns:")
    for r in rows2:
        print("  ", r)
    with open(os.path.join(RES, "strongest.json"), "w") as f:
        json.dump(out, f, indent=1, default=int)


if __name__ == "__main__":
    main()

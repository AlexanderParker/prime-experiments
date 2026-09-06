"""bi_family.py -- what the moments of the order distribution say about the teeth.

The mean order is teeth-free (q'/(q'-2), the two-copies law).  The VARIANCE is the first moment
that sees the teeth, and by the closed form it sees them only through

    S = (1/N) sum_{r >= 2} C_r,      Var = 2 [ (q'-4) + S (q'-2) ] / (q'-2)^2 .

This scores the real machine against the counterfactual family (teeth at +-v_g, v_g uniform in
1..(g-1)/2 -- alignment-rules section 5, the family of r56/mf_family.py, same seed) at the two
rungs 11 -> 13 and 13 -> 17.

Usage:  uv run python research/anchor235/r57/bi_family.py
"""
import os
import random
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_core import (OUT, u_of, sieve_machine, gaps_of, letters, chain_counts,
                     orders_from_chains, add_gear_stream)


def score(gears, vs):
    q, v = gears[-1], vs[-1]
    P, O = sieve_machine(gears[:-1], vs[:-1])
    gaps = gaps_of(P, O).astype(np.uint8)
    N = gaps.size
    lt = letters(gaps, q, u=v)
    C = chain_counts(lt, q, N, rmax=10)
    nJ = orders_from_chains(C)
    _, _, _, hist, _ = add_gear_stream(P, O, q, want_gaps=False, u=v)
    direct = [int(hist[J]) for J in range(1, len(nJ) + 1)]
    ident = all(direct[i] == nJ[i] for i in range(len(nJ)))
    Nnew = sum(nJ)
    s1 = sum((i + 1) * n for i, n in enumerate(nJ))
    s2 = sum((i + 1) ** 2 * n for i, n in enumerate(nJ))
    pred2 = q * N + 4 * N + 2 * sum(C[2:])
    mean = s1 / Nnew
    var = s2 / Nnew - mean ** 2
    S = sum(C[2:]) / N
    varclosed = 2 * ((q - 4) + S * (q - 2)) / (q - 2) ** 2
    return dict(q=q, N=N, nJ=nJ, ident=ident, mean=mean, exact=q / (q - 2), var=var,
                varclosed=varclosed, S=S, C2=C[2], C3=C[3], n3=nJ[2] if len(nJ) > 2 else 0,
                maxorder=max(i + 1 for i, n in enumerate(nJ) if n > 0),
                mom2ok=(s2 == pred2))


def main():
    lines = []
    W = lines.append
    W("bi_family -- the variance of the order distribution against the counterfactual family")
    rng = random.Random(20260906)
    for gears in ([5, 7, 11, 13], [5, 7, 11, 13, 17]):
        real = [u_of(g) for g in gears]
        real = [min(v, g - v) for v, g in zip(real, gears)]
        W("")
        W(f"=== rung {gears[-2]} -> {gears[-1]} (real teeth v = {real}) ===")
        W("member | v | n_2 | n_3 | max order | mean | q'/(q'-2) | Var | closed form | S | "
          "C_2 | identity holds | 2nd moment holds")
        rows = []
        r = score(gears, real)
        rows.append(("REAL", list(real), r))
        seen = {tuple(real)}
        while len(rows) < 21:
            vs = [rng.randrange(1, (g - 1) // 2 + 1) for g in gears]
            if tuple(vs) in seen:
                continue
            seen.add(tuple(vs))
            rows.append((f"m{len(rows)}", vs, score(gears, vs)))
        for name, vs, r in rows:
            W(f"{name} | {vs} | {r['nJ'][1]} | {r['n3']} | {r['maxorder']} | {r['mean']:.9f} | "
              f"{r['exact']:.9f} | {r['var']:.9f} | {r['varclosed']:.9f} | {r['S']:.6f} | "
              f"{r['C2']} | {r['ident']} | {r['mom2ok']}")
        vars_ = [r["var"] for _, _, r in rows]
        n3s = [r["n3"] for _, _, r in rows]
        rv, rn = vars_[0], n3s[0]
        W(f"real Var = {rv:.9f}; family min {min(vars_[1:]):.9f}, "
          f"median {sorted(vars_[1:])[len(vars_[1:])//2]:.9f}, max {max(vars_[1:]):.9f}; "
          f"percentile of the real machine = "
          f"{sum(1 for v in vars_ if v < rv)/len(vars_):.3f}")
        W(f"real n_3 = {rn}; family min {min(n3s[1:])}, "
          f"median {sorted(n3s[1:])[len(n3s[1:])//2]}, max {max(n3s[1:])}; "
          f"percentile = {sum(1 for v in n3s if v < rn)/len(n3s):.3f}")
        W(f"mean order exact for {sum(1 for _,_,r in rows if abs(r['mean']-r['exact'])<1e-12)}"
          f" of {len(rows)} members; branching identity holds for "
          f"{sum(1 for _,_,r in rows if r['ident'])} of {len(rows)}; second-moment identity for "
          f"{sum(1 for _,_,r in rows if r['mom2ok'])} of {len(rows)}")
    open(os.path.join(OUT, "bi_family.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print("\nwrote results/bi_family.txt")


if __name__ == "__main__":
    main()

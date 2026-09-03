"""Depth of the nested formula per layer, and the record as a two-class residue chain.

Layer g on the machine M = {5..g-}: the full period of {5..g} is g copies of the lower
period, copy j shifted by j P_M; since P_M is invertible mod g the g copies realise every
deletion phase r in Z_g exactly once, deleting the lower openings with residue r or
r + d_g (mod g). So the layer needs only the lower opening residues mod g, once:

    D_g   = longest run of consecutive lower openings whose residues mod g all lie in one
            two-class set {r, r + d_g}  (chain depth = nesting depth of the formula)
    F_g+1 = max over such runs of (gap before) + (run span) + (gap after)   (record law)

Computes D_g and F_g on one lower period only (no full-period array), through g = 29, and
the unrolled term count prod (1 + D_h).
"""
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23, 29]


def on_teeth(g, x):
    u = pow(6, -1, g)
    return (x % g == u) | (x % g == g - u)


def main():
    terms = 1
    for n in range(1, len(PR)):
        low, g = PR[:n], PR[n]
        P = prod(low)
        k = np.arange(P, dtype=np.int64)
        w = np.ones(P, dtype=bool)
        for h in low:
            w &= ~on_teeth(h, k)
        X = np.flatnonzero(w)
        # doubled to see runs across the period boundary
        X2 = np.concatenate([X, X + P])
        gaps2 = np.diff(X2)
        u = pow(6, -1, g); d = (2 * u) % g
        best_D, best_F, best_r = 0, 0, None
        for r in range(g):
            hit = (X2 % g == r) | (X2 % g == (r + d) % g)
            h8 = hit.astype(np.int8)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], h8, [0]])) != 0)
            starts, ends = edges[0::2], edges[1::2]     # run = X2[starts:ends]
            keep = starts < len(X)
            starts, ends = starts[keep], ends[keep]
            if len(starts) == 0:
                continue
            runlen = ends - starts
            D = int(runlen.max())
            # merged gap = X2[end] - X2[start-1]  (survivor before to survivor after)
            ok = (starts >= 1) & (ends < len(X2))
            span = X2[ends[ok]] - X2[starts[ok] - 1]
            Fr = int(span.max()) - 1 if len(span) else 0
            if D > best_D:
                best_D = D
            if Fr > best_F:
                best_F, best_r = Fr, r
        terms *= (1 + best_D)
        print(f"layer {g:>2} on {'+'.join(map(str, low))}: lower openings {len(X)}, "
              f"chain depth D_{g} = {best_D}, F_{g} = {best_F} (phase r = {best_r}), "
              f"unrolled terms prod(1 + D) = {terms}")


if __name__ == "__main__":
    main()

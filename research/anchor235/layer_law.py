"""Hop law along a walk beyond density.

For the record stretch of the word {5..q} (exact period, q <= 29) the walk crosses L = F + 1
blocked slots. Layer by layer: S_g = slots of the stretch still open under {5..g}
(S_q = 0), hops_g = S_{g-} - S_g = slots whose smallest blocker is g. Density expectation
E_g = L prod_{h <= g} (1 - 2/h). Pre-registered expectations: (1) S_g < E_g at every layer of
the record stretch (the record is ahead of density everywhere); (2) hops at the top layers are
<= 5; (3) the deficit S_g - E_g is made at the small gears. Compared against random stretches
of the same length and against the runner-up gaps.

Also prints the bottom closed forms: the recursion depth (max chain of hits at one layer)
per gear, i.e. how far the unrolled expression must nest.
"""
import sys
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23]


def blocked_by(g, k):
    u = pow(6, -1, g)
    return (k % g == u) | (k % g == g - u)


def profile(gears, k):
    """k: slots of a stretch. returns S_g list and hops_g list."""
    alive = np.ones(len(k), dtype=bool)
    S, H = [], []
    for g in gears:
        hit = alive & blocked_by(g, k)
        H.append(int(hit.sum()))
        alive &= ~blocked_by(g, k)
        S.append(int(alive.sum()))
    return S, H


def main():
    for idx in range(1, len(PR)):
        gears = PR[:idx + 1]
        q = gears[-1]
        P = prod(gears)
        k = np.arange(P, dtype=np.int32)
        w = np.ones(P, dtype=bool)
        for g in gears:
            w &= ~blocked_by(g, k)
        X = np.flatnonzero(w)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        order = np.argsort(-gaps)
        L = int(gaps[order[0]]) - 1
        F = L
        E = [L * prod(1 - 2 / h for h in gears[:i + 1]) for i in range(len(gears))]
        print(f"\n{'+'.join(map(str, gears))}: F = {F}, record stretch L = {L} blocked slots; "
              f"density E_g = {', '.join(f'{e:.1f}' for e in E)}")
        rows = []
        for r in range(3):
            i = order[r]
            start = X[i] + 1
            ks = np.arange(start, start + int(gaps[i]) - 1)
            S, H = profile(gears, ks)
            rows.append((f"gap #{r + 1} L={len(ks)}", S, H))
        # random stretches of the record length: mean survivors per layer
        rng = np.random.default_rng(0)
        Sr = np.zeros(len(gears)); n = 2000
        for _ in range(n):
            st = int(rng.integers(P))
            ks = np.arange(st, st + L)
            S, _ = profile(gears, ks)
            Sr += S
        Sr /= n
        for name, S, H in rows:
            print(f"  {name:>14}: survivors S_g = {S}  hops = {H}")
        print(f"  random L-stretch: survivors mean = {[round(float(x), 1) for x in Sr]}")
        ratio = [S / e if e > 0 else 0 for S, e in zip(rows[0][1], E)]
        print(f"  record S_g / E_g = {[round(x, 2) for x in ratio]}")
        # defining law of a double hit at layer g: consecutive lower openings x < y both on
        # g's teeth  <=>  y - x = 0 or +-d_g (mod g), d_g = 2u_g mod g. List the realised gaps.
        for i in range(1, len(gears)):
            low = gears[:i]; g = gears[i]; d = (2 * pow(6, -1, g)) % g
            wl = np.ones(P, dtype=bool)
            for h in low:
                wl &= ~blocked_by(h, k)
            Xl = np.flatnonzero(wl)
            hit = blocked_by(g, Xl)
            both = hit[:-1] & hit[1:]
            dg = np.diff(Xl)[both]
            allowed = sorted(set(int(v) for v in dg))
            law_ok = all((v % g) in (0, d, g - d) for v in allowed)
            h8 = hit.astype(np.int8)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], h8, [0]])) != 0)
            depth = int((edges[1::2] - edges[0::2]).max()) if len(edges) else 0
            print(f"  layer {g} (d={d}, lower F={int(np.diff(Xl).max()) - 1}): chain depth {depth}, "
                  f"double-hit gaps {allowed}, law {law_ok}, doubles {int(both.sum())} of {int(hit.sum())} hits")
        print()


if __name__ == "__main__":
    main()

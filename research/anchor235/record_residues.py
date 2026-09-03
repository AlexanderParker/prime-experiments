"""The residue pattern that makes a record, layer by layer from the top.

For the record stretch of {5..q} (q <= 23, full period), at each of the top four layers g:
the survivors of the lower gears inside the stretch (positions relative to the stretch start),
which tooth of g each sits on (+ for u_g, - for -u_g), and the differences between consecutive
survivors as classes mod g (0 = same tooth, +d / -d = tooth switch, d_g = 2u_g mod g). Also:
does the neighbour of a g-hit tend to be open (the human's "either side of a hit" question):
P(x+1 open | x is a g-hit) against P(x+1 open | x blocked) and P(open).
"""
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23]


def blocked_by(g, k):
    u = pow(6, -1, g)
    return (k % g == u) | (k % g == g - u)


def main():
    for idx in range(2, len(PR)):
        gears = PR[:idx + 1]
        P = prod(gears)
        k = np.arange(P, dtype=np.int32)
        w = np.ones(P, dtype=bool)
        for g in gears:
            w &= ~blocked_by(g, k)
        X = np.flatnonzero(w)
        gaps = np.diff(np.concatenate([X, [X[0] + P]]))
        i = int(np.argmax(gaps))
        start = int(X[i]) + 1
        L = int(gaps[i]) - 1
        ks = np.arange(start, start + L, dtype=np.int64)
        print(f"\n{'+'.join(map(str, gears))}: record F = {L}, stretch starts at slot {start} (mod {P})")
        for j in range(len(gears) - 1, max(len(gears) - 5, 0), -1):
            g = gears[j]; u = pow(6, -1, g); d = (2 * u) % g
            alive = np.ones(L, dtype=bool)
            for h in gears[:j]:
                alive &= ~blocked_by(h, ks)
            surv = ks[alive]
            rel = [int(x - start) for x in surv]
            tooth = ["+" if x % g == u else ("-" if x % g == g - u else "?") for x in surv]
            diffs = np.diff(surv)
            cls = []
            for v in diffs:
                r = int(v % g)
                cls.append("0" if r == 0 else ("+d" if r == d else ("-d" if r == g - d else f"?{r}")))
            print(f"  layer {g:>2} (u={u}, d={d}): survivors at {rel} teeth {''.join(tooth)} "
                  f"diffs {[int(v) for v in diffs]} classes {cls}")
        # either side of a hit
        for g in (gears[-1], gears[-2], 7):
            hit = blocked_by(g, k)
            nxt = np.roll(w, -1)
            p_open = w.mean()
            p_next_given_hit = nxt[hit].mean()
            p_next_given_blocked = nxt[~w].mean()
            p_next_given_open = nxt[w].mean()
            print(f"  gear {g:>2}: P(open) = {p_open:.4f}; P(x+1 open | x g-hit) = {p_next_given_hit:.4f}; "
                  f"P(x+1 open | x blocked) = {p_next_given_blocked:.4f}; P(x+1 open | x open) = {p_next_given_open:.4f}")


if __name__ == "__main__":
    main()

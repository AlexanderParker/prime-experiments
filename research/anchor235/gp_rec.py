"""GP for the top-gear hop with the lower walk available as a primitive.

Target H_g(s) = W_g(s) - W_{g-}(s) on machine {5..g}, evaluated at the lower landing x.
Terminals: aTOP = (u_g - x) mod g, bTOP = (-u_g - x) mod g, hit0 = [top gear hits x],
L1 = W_{g-}(x + 1) (lower walk from the slot after x), y = x + 1 + L1, hit1 = [top hits y],
L2 = W_{g-}(y + 1), z = y + 1 + L2, hit2 = [top hits z], L3 = W_{g-}(z + 1), gear g, constants.
The question is whether the search closes the hop exactly once the lower walk is a primitive
(the chain hit0 ? 1 + L1 + (hit1 ? 1 + L2 + (hit2 ? 1 + L3 : 0) : 0) : 0 is expressible), and
what it finds with the chain terminals removed (--nochain: only aTOP, bTOP, L1).
"""
import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_walk import word, walk, PR, run_gp


def features(gears, s, chain=True):
    low = gears[:-1]; g = gears[-1]; u = pow(6, -1, g)
    w, P = word(gears); wl, Pl = word(low)
    W = walk(w, P); Wl = walk(wl, Pl)
    x = s + Wl[s % Pl]
    H = W - Wl[s % Pl]
    hit = lambda t: (((t % g) == u) | ((t % g) == g - u)).astype(np.int64)
    L1 = Wl[(x + 1) % Pl]; y = x + 1 + L1
    L2 = Wl[(y + 1) % Pl]; z = y + 1 + L2
    L3 = Wl[(z + 1) % Pl]
    cols = [(u - x) % g, (-u - x) % g, hit(x), L1]
    names = ["aTOP", "bTOP", "hit0", "L1"]
    if chain:
        cols += [hit(y), L2, hit(z), L3]
        names += ["hit1", "L2", "hit2", "L3"]
    cols.append(np.full(P, g)); names.append("g")
    return np.stack(cols).astype(np.int64), names, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=13)
    ap.add_argument("--pop", type=int, default=1000)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nochain", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gears = [g for g in PR if g <= a.q]
    gears_t = gears + [PR[len(gears)]]
    X, names, H = features(gears, np.arange(int(np.prod(gears))), not a.nochain)
    Xt, names_t, Ht = features(gears_t, np.arange(int(np.prod(gears_t))), not a.nochain)
    print(f"target hop of gear {gears[-1]} on {'+'.join(map(str, gears))}, terminals {names}; "
          f"unseen gear {gears_t[-1]}; H=0 at {float((H == 0).mean()):.4f}")
    run_gp(X, names, H, a.pop, a.gens, rng, Xt, Ht, names_t)


if __name__ == "__main__":
    main()

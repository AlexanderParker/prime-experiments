"""Genetic programming hunt for a closed form of the walk.

Machine {5..q}. For a start slot s the components are, per gear g, the distances to its next
tooth hits a_g = (u_g - s) mod g, b_g = (-u_g - s) mod g. The walk W(s) is the smallest
t >= 0 with t not in {a_g + g Z} u {b_g + g Z} for any g. We evolve expression trees over
the a_g, b_g, g and constants with +, -, *, min, max, mod, floordiv, lt, ite; fitness is the
exact-match rate on every s of one period (plus a parsimony term); the best tree is then
tested on the next machine (unseen), where the extra gear's inputs are simply absent.

Layered variant (--layer): target the hop H_g(s) = W_g(s) - W_{g-}(s) of the top gear g,
inputs the components at the lower walk's landing x = s + W_{g-}(s).

Usage: gp_walk.py [--q 17] [--pop 600] [--gens 150] [--seed 0] [--layer]
"""
import argparse
import random
from math import prod

import numpy as np

PR = [5, 7, 11, 13, 17, 19, 23]


def word(gears):
    P = prod(gears)
    k = np.arange(P, dtype=np.int64)
    w = np.ones(P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        w &= (k % g != u) & (k % g != g - u)
    return w, P


def walk(w, P):
    """W[s] = distance from s to the next open slot (0 if s open)."""
    ww = np.concatenate([w, w])
    nxt = np.zeros(2 * P, dtype=np.int64)
    last = -1
    for i in range(2 * P - 1, -1, -1):
        if ww[i]:
            last = i
        nxt[i] = last
    return (nxt[:P] - np.arange(P))


def components(gears, s):
    cols, names = [], []
    for g in gears:
        u = pow(6, -1, g)
        cols.append((u - s) % g); names.append(f"a{g}")
        cols.append((-u - s) % g); names.append(f"b{g}")
    return np.stack(cols).astype(np.int64), names


# ---- expression trees ----------------------------------------------------------------
FUNCS = {
    "add": (2, lambda x, y: x + y),
    "sub": (2, lambda x, y: x - y),
    "mul": (2, lambda x, y: np.clip(x * y, -10**6, 10**6)),
    "min": (2, np.minimum),
    "max": (2, np.maximum),
    "mod": (2, lambda x, y: np.where(y != 0, x % np.where(y != 0, y, 1), x)),
    "div": (2, lambda x, y: np.where(y != 0, x // np.where(y != 0, y, 1), x)),
    "lt": (2, lambda x, y: (x < y).astype(np.int64)),
    "ite": (3, lambda c, x, y: np.where(c != 0, x, y)),
}
FN = list(FUNCS)


class Node:
    __slots__ = ("op", "kids", "val")

    def __init__(self, op, kids=(), val=None):
        self.op, self.kids, self.val = op, list(kids), val

    def size(self):
        return 1 + sum(k.size() for k in self.kids)

    def __str__(self):
        if self.op == "T":
            return str(self.val)
        return f"{self.op}({', '.join(map(str, self.kids))})"


def rand_tree(depth, names, rng):
    if depth == 0 or (depth < 4 and rng.random() < 0.3):
        if rng.random() < 0.7:
            return Node("T", val=rng.choice(names))
        return Node("T", val=rng.choice([0, 1, 2, 3, 5, 7, 11, 13, 17, 19]))
    op = rng.choice(FN)
    ar = FUNCS[op][0]
    return Node(op, [rand_tree(depth - 1, names, rng) for _ in range(ar)])


def evaluate(node, X, idx):
    if node.op == "T":
        v = node.val
        if isinstance(v, str):
            if v in idx:
                return X[idx[v]]
            return np.zeros(X.shape[1], dtype=np.int64)  # missing input (unseen machine)
        return np.full(X.shape[1], v, dtype=np.int64)
    args = [evaluate(k, X, idx) for k in node.kids]
    return FUNCS[node.op][1](*args)


def copy(node):
    return Node(node.op, [copy(k) for k in node.kids], node.val)


def nodes_of(node, acc=None):
    acc = [] if acc is None else acc
    acc.append(node)
    for k in node.kids:
        nodes_of(k, acc)
    return acc


def crossover(a, b, rng):
    a, b = copy(a), copy(b)
    na, nb = nodes_of(a), nodes_of(b)
    x = rng.choice(na[1:]) if len(na) > 1 else None
    y = rng.choice(nb)
    if x is None:
        return copy(y)
    x.op, x.kids, x.val = y.op, [copy(k) for k in y.kids], y.val
    return a


def mutate(a, names, rng):
    a = copy(a)
    n = rng.choice(nodes_of(a))
    r = rand_tree(rng.randint(0, 3), names, rng)
    n.op, n.kids, n.val = r.op, r.kids, r.val
    return a


def fitness(tree, X, idx, y, size_pen=0.002):
    try:
        pred = evaluate(tree, X, idx)
    except Exception:
        return -1.0, 0.0
    exact = float((pred == y).mean())
    mae = float(np.abs(pred - y).mean())
    return exact - size_pen * tree.size() - 0.01 * min(mae, 20), exact


def run_gp(X, names, y, pop_n, gens, rng, Xtest=None, ytest=None, names_test=None):
    idx = {n: i for i, n in enumerate(names)}
    pop = [rand_tree(rng.randint(2, 5), names, rng) for _ in range(pop_n)]
    fits = [fitness(t, X, idx, y) for t in pop]
    best = max(range(pop_n), key=lambda i: fits[i][0])
    best_t, best_f = copy(pop[best]), fits[best]
    for gen in range(gens):
        new = [copy(best_t)]
        while len(new) < pop_n:
            def tourn():
                c = rng.sample(range(pop_n), 5)
                return pop[max(c, key=lambda i: fits[i][0])]
            r = rng.random()
            if r < 0.6:
                child = crossover(tourn(), tourn(), rng)
            elif r < 0.95:
                child = mutate(tourn(), names, rng)
            else:
                child = rand_tree(rng.randint(2, 5), names, rng)
            if child.size() > 60:
                child = rand_tree(3, names, rng)
            new.append(child)
        pop = new
        fits = [fitness(t, X, idx, y) for t in pop]
        b = max(range(pop_n), key=lambda i: fits[i][0])
        if fits[b][0] > best_f[0]:
            best_t, best_f = copy(pop[b]), fits[b]
        if gen % 25 == 0 or gen == gens - 1:
            line = f"  gen {gen:>4}: best exact-match {best_f[1]:.4f}  size {best_t.size()}  {best_t}"
            if Xtest is not None:
                idx_t = {n: i for i, n in enumerate(names_test)}
                line += f"   | unseen machine exact-match {fitness(best_t, Xtest, idx_t, ytest)[1]:.4f}"
            print(line, flush=True)
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=17)
    ap.add_argument("--pop", type=int, default=600)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gears = [g for g in PR if g <= a.q]
    w, P = word(gears)
    s = np.arange(P)
    W = walk(w, P)
    nxt_g = PR[len(gears)]
    gears_t = gears + [nxt_g]
    wt, Pt = word(gears_t)
    st = np.arange(Pt)
    Wt = walk(wt, Pt)
    if not a.layer:
        X, names = components(gears, s)
        Xt, names_t = components(gears_t, st)
        print(f"target: walk W(s) on {'+'.join(map(str, gears))} ({P} starts; W=0 at {float((W == 0).mean()):.3f}, "
              f"mean {W.mean():.2f}, max {W.max()}); baseline 'predict 0' exact {float((W == 0).mean()):.4f}; "
              f"unseen machine + {nxt_g}")
        run_gp(X, names, W, a.pop, a.gens, rng, Xt, Wt, names_t)
    else:
        # hop of the top gear: lower machine gears[:-1], landing x = s + W_lower(s)
        low = gears[:-1]
        wl, Pl = word(low)
        Wl = walk(wl, Pl)
        Wl_s = Wl[s % Pl]
        x = s + Wl_s
        H = W - Wl_s
        X, names = components(gears, x)
        print(f"target: hop H_g(s) of gear {gears[-1]} at the lower landing x on {'+'.join(map(str, gears))} "
              f"(H=0 at {float((H == 0).mean()):.3f}, mean {H.mean():.2f}, max {H.max()}); "
              f"inputs = components at x")
        # unseen: next machine, hop of its top gear at its lower landing
        lowt = gears_t[:-1]
        wlt, Plt = word(lowt)
        Wlt = walk(wlt, Plt)[st % Plt]
        xt = st + Wlt
        Ht = Wt - Wlt
        Xt, names_t = components(gears_t, xt)
        # rename the top gear's inputs to generic names so the tree transfers
        names = [n if n[1:] != str(gears[-1]) else n[0] + "TOP" for n in names]
        names_t = [n if n[1:] != str(gears_t[-1]) else n[0] + "TOP" for n in names_t]
        run_gp(X, names, H, a.pop, a.gens, rng, Xt, Ht, names_t)


if __name__ == "__main__":
    main()

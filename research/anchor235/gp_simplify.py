"""Evolve simpler forms of the closed (nested residue) form of the walk.

Two experiments, each a genetic programme over expression trees with lexicographic fitness
(number of slots exact, then fewer nodes), seeded with the nested form of section 9f and
with random trees, so that the search can only keep an expression if it is at least as
exact as the seed and prefers it if it is smaller.

  residues  target W_{5,7} (period 35) or W_{5..11} (period 385) as an expression in the
            residues r5 = s mod 5, r7 = s mod 7 (, r11) and small constants. Seed = the
            nested form written out in residues (about 40 nodes).
  layer     target the hop H_g = W_g - W_M of one layer g on M = {5..g-} as an expression
            in the chain primitives: hit_i = [i-th landing on g's teeth], L_i = i-th lower
            gap after the landing (one lower evaluation each), xr = landing mod g, and the
            constants g, d_g, u_g, 0..3. Cost = nodes + 10 x (distinct L_i used), so a form
            using fewer lower evaluations wins outright. Seed = hit0 (L1 + hit1 (L2 + hit2 L3)).
            Fitness on a sample (all chain slots + random slots); the winner is re-verified
            on the full period.

usage: gp_simplify.py residues 7|11 [seed] [gens]
       gp_simplify.py layer g [seed] [gens]
"""
import sys
from math import prod

import numpy as np

# ------------------------------------------------------------------ expression trees
# node = (op, child, child, ...) or ('c', value) or ('v', name)
ARITY = {'add': 2, 'sub': 2, 'mul': 2, 'mod': 2, 'min': 2, 'max': 2, 'eq': 2, 'lt': 2, 'ite': 3}
OPS = list(ARITY)


def ev(t, env):
    op = t[0]
    if op == 'c':
        return t[1]
    if op == 'v':
        return env[t[1]]
    a = ev(t[1], env)
    if op == 'ite':
        return np.where(a > 0, ev(t[2], env), ev(t[3], env))
    b = ev(t[2], env)
    if op == 'add': return a + b
    if op == 'sub': return a - b
    if op == 'mul': return a * b
    if op == 'mod': return np.where(b > 0, np.mod(a, np.where(b > 0, b, 1)), a)
    if op == 'min': return np.minimum(a, b)
    if op == 'max': return np.maximum(a, b)
    if op == 'eq': return (a == b).astype(np.int64)
    if op == 'lt': return (a < b).astype(np.int64)
    raise ValueError(op)


def size(t):
    return 1 if t[0] in ('c', 'v') else 1 + sum(size(c) for c in t[1:])


def show(t):
    op = t[0]
    if op == 'c': return str(t[1])
    if op == 'v': return t[1]
    return f"{op}({', '.join(show(c) for c in t[1:])})"


def used_vars(t, acc=None):
    acc = set() if acc is None else acc
    if t[0] == 'v': acc.add(t[1])
    elif t[0] != 'c':
        for c in t[1:]: used_vars(c, acc)
    return acc


def rand_tree(rng, vars_, consts, depth):
    if depth == 0 or rng.random() < 0.3:
        if rng.random() < 0.6:
            return ('v', vars_[rng.integers(len(vars_))])
        return ('c', int(consts[rng.integers(len(consts))]))
    op = OPS[rng.integers(len(OPS))]
    return (op,) + tuple(rand_tree(rng, vars_, consts, depth - 1) for _ in range(ARITY[op]))


def subtrees(t, path=()):
    yield path, t
    if t[0] not in ('c', 'v'):
        for i, c in enumerate(t[1:], 1):
            yield from subtrees(c, path + (i,))


def replace(t, path, new):
    if not path: return new
    i = path[0]
    return t[:i] + (replace(t[i], path[1:], new),) + t[i + 1:]


def crossover(rng, a, b):
    pa = list(subtrees(a)); pb = list(subtrees(b))
    p, _ = pa[rng.integers(len(pa))]
    _, s = pb[rng.integers(len(pb))]
    return replace(a, p, s)


def mutate(rng, t, vars_, consts):
    pa = list(subtrees(t))
    p, s = pa[rng.integers(len(pa))]
    r = rng.random()
    if r < 0.4:
        return replace(t, p, rand_tree(rng, vars_, consts, 2))
    if r < 0.7 and s[0] not in ('c', 'v'):   # hoist a child: the shrinking move
        return replace(t, p, s[1 + rng.integers(len(s) - 1)])
    if s[0] == 'c':
        return replace(t, p, ('c', int(consts[rng.integers(len(consts))])))
    if s[0] == 'v':
        return replace(t, p, ('v', vars_[rng.integers(len(vars_))]))
    ops = [o for o in OPS if ARITY[o] == ARITY[s[0]]]
    return replace(t, p, (ops[rng.integers(len(ops))],) + s[1:])


# ------------------------------------------------------------------ machines
def on_teeth(g, x):
    u = pow(6, -1, g)
    return (x % g == u) | (x % g == g - u)


def walk(gears):
    P = prod(gears)
    k = np.arange(P, dtype=np.int64)
    w = np.ones(P, dtype=bool)
    for g in gears:
        w &= ~on_teeth(g, k)
    idx = np.flatnonzero(np.concatenate([w, w]))
    nxt = idx[np.searchsorted(idx, k)]
    return nxt - k, w


# ------------------------------------------------------------------ experiments
def V(n): return ('v', n)
def C(n): return ('c', n)
def ADD(a, b): return ('add', a, b)
def MUL(a, b): return ('mul', a, b)
def MOD(a, b): return ('mod', a, b)
def EQ(a, b): return ('eq', a, b)


def setup_residues(top):
    gears = [5, 7] if top == 7 else [5, 7, 11]
    W, _ = walk(gears)
    s = np.arange(prod(gears), dtype=np.int64)
    env = {f"r{g}": s % g for g in gears}
    vars_ = list(env)
    consts = list(range(0, 12))

    # seed: nested form in residues. W5(t) = [t=1]+[t=4] on r5 of t; teeth of 7: 6, 1.
    def w5(r5): return ADD(EQ(r5, C(1)), EQ(r5, C(4)))
    def hit7(r7): return ADD(EQ(r7, C(6)), EQ(r7, C(1)))
    r5, r7 = V('r5'), V('r7')
    w0 = w5(r5)                                  # W5(s)
    x5 = MOD(ADD(r5, w0), C(5)); x7 = MOD(ADD(r7, w0), C(7))
    h1 = hit7(x7)
    st1 = ADD(C(1), w5(MOD(ADD(x5, C(1)), C(5))))  # 1 + W5(x+1)
    x15 = MOD(ADD(x5, st1), C(5)); x17 = MOD(ADD(x7, st1), C(7))
    h2 = hit7(x17)
    st2 = ADD(C(1), w5(MOD(ADD(x15, C(1)), C(5))))
    seed = ADD(w0, ADD(MUL(h1, st1), MUL(MUL(h1, h2), st2)))
    if top == 11:
        # one more layer, depth 1: teeth of 11: u = 2, 9. landing y = s + W_{5,7}(s)
        # (seed's W_{5,7} reused as subtree: heavy, that is the point)
        y5 = MOD(ADD(r5, seed), C(5)); y7 = MOD(ADD(r7, seed), C(7)); y11 = MOD(ADD(V('r11'), seed), C(11))
        h = ADD(EQ(y11, C(2)), EQ(y11, C(9)))
        # W_{5,7}(y+1): substitute residues of y+1 into a copy of seed
        def subst(t, m):
            if t[0] == 'v': return m[t[1]]
            if t[0] == 'c': return t
            return (t[0],) + tuple(subst(c, m) for c in t[1:])
        w57y = subst(seed, {'r5': MOD(ADD(y5, C(1)), C(5)), 'r7': MOD(ADD(y7, C(1)), C(7))})
        seed = ADD(seed, MUL(h, ADD(C(1), w57y)))
    return env, vars_, consts, W, seed, lambda t: size(t), None, None


def setup_layer(g):
    PR = [5, 7, 11, 13, 17, 19, 23]
    low = PR[:PR.index(g)]
    gears = low + [g]
    Wt, _ = walk(gears)
    WM, wl = walk(low)          # lower walk on its own period (periodic, index mod P_M)
    PM = prod(low)
    P = prod(gears)
    s = np.arange(P, dtype=np.int64)
    wm = WM[s % PM]
    x = s + wm
    H = Wt - wm
    hit = []; L = []
    cur = x
    for i in range(4):
        hit.append(on_teeth(g, cur).astype(np.int64))
        if i < 3:
            step = 1 + WM[(cur + 1) % PM]
            L.append(step)
            cur = cur + step
    u = pow(6, -1, g); d = (2 * u) % g
    env_full = {f"hit{i}": hit[i] for i in range(4)}
    env_full.update({f"L{i + 1}": L[i] for i in range(3)})
    env_full['xr'] = x % g
    # sample: every chain slot (hit0 and hit1) plus random slots
    rng = np.random.default_rng(1)
    chain = np.flatnonzero(hit[0] & hit[1])
    single = np.flatnonzero(hit[0])
    pick = np.unique(np.concatenate([chain, rng.choice(single, min(20000, len(single)), replace=False),
                                     rng.integers(P, size=50000)]))
    env = {k: v[pick] for k, v in env_full.items()}
    vars_ = list(env)
    consts = [0, 1, 2, 3, g, d, u, g - d]
    seed = MUL(V('hit0'), ADD(V('L1'), MUL(V('hit1'), ADD(V('L2'), MUL(V('hit2'), V('L3'))))))

    def cost(t):
        return size(t) + 10 * sum(1 for v in used_vars(t) if v.startswith('L'))
    return env, vars_, consts, H[pick], seed, cost, env_full, H


def run(kind, arg, use_seed, gens):
    if kind == 'residues':
        env, vars_, consts, target, seed, cost, env_full, target_full = setup_residues(arg)
    else:
        env, vars_, consts, target, seed, cost, env_full, target_full = setup_layer(arg)
    n = len(target)
    rng = np.random.default_rng(7)
    POP = 600

    def fit(t):
        try:
            v = ev(t, env)
        except Exception:
            return (-1, 10 ** 6)
        v = np.broadcast_to(v, target.shape)
        return (int((v == target).sum()), cost(t))

    pop = [rand_tree(rng, vars_, consts, 4) for _ in range(POP)]
    cap = max(80, size(seed) + 20) if use_seed else 80
    if use_seed:
        pop[0] = seed
        for i in range(1, 60):
            pop[i] = mutate(rng, seed, vars_, consts)
    scored = [(fit(t), t) for t in pop]
    print(f"{kind} {arg}: {n} fitness slots; seed exact {fit(seed)[0]}/{n}, cost {cost(seed)}, "
          f"seed used {'yes' if use_seed else 'no'}", flush=True)
    best = None
    for gen in range(gens):
        scored.sort(key=lambda z: (-z[0][0], z[0][1]))
        if best is None or (scored[0][0][0], -scored[0][0][1]) > (best[0][0], -best[0][1]):
            best = scored[0]
            print(f"  gen {gen:>4}: exact {best[0][0]}/{n} cost {best[0][1]} nodes {size(best[1])}: {show(best[1])}", flush=True)
        elite = [z[1] for z in scored[:20]]
        new = list(elite)
        while len(new) < POP:
            def tour():
                c = [scored[rng.integers(len(scored))] for _ in range(5)]
                c.sort(key=lambda z: (-z[0][0], z[0][1]))
                return c[0][1]
            child = crossover(rng, tour(), tour()) if rng.random() < 0.6 else tour()
            if rng.random() < 0.5:
                child = mutate(rng, child, vars_, consts)
            if size(child) <= cap:
                new.append(child)
        scored = [(fit(t), t) for t in new]
    scored.sort(key=lambda z: (-z[0][0], z[0][1]))
    b = scored[0]
    print(f"final: exact {b[0][0]}/{n} cost {b[0][1]} nodes {size(b[1])}: {show(b[1])}")
    if env_full is not None:
        v = np.broadcast_to(ev(b[1], env_full), target_full.shape)
        print(f"full-period check: {int((v == target_full).sum())}/{len(target_full)} exact; "
              f"lower gaps used {sorted(x for x in used_vars(b[1]) if x.startswith('L'))}")


if __name__ == "__main__":
    kind = sys.argv[1]; arg = int(sys.argv[2])
    use_seed = (sys.argv[3] == 'seed') if len(sys.argv) > 3 else True
    gens = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    run(kind, arg, use_seed, gens)

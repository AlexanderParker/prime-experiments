"""Evolutionary search over walk algorithms (owner, 2026-09-15).

A walk algorithm = an origin rule and a list of steps; each step names a mirror rule, a period
rule and a direction rule, all computed from the machine alone (q, its gears, the base, the
step index, the column's position).  Nothing in a genome may test a candidate column by
division: no residues of the column, no primality.  Allowed inputs: q, the gear list, the
spiral base, the largest primorials fitting the window, the twin gear pairs among the gears
(known primes), residues of q itself, the column's numeric position against the window.

Fitness: the number of consecutive machines from 11 upward at which the walk lands on a twin
inside the window (the first failing machine stops the count); ties broken by fewer steps.

Population evolves by mutation (change a rule, add or drop a step, change the origin) and
crossover; the elite survives; the best genomes are written to results_evolve_walk.json after
every generation so a later run can continue from them.

usage: uv run python research/stack/r8/evolve_walk.py [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path

QMAX = 3000
MIRROR_KINDS = ['B', 'Pmax', 'Pmax_prev', 'B_x_above', 'B_x_below', 'six_x_above', 'six_x_below', 'B_x_twin', 'B_x_root', 'two_x_below', 'B_x_above_pair']
PERIOD_KINDS = ['const', 'enter', 'top', 'q_mod', 'count_mod', 'gap_mod']
DIR_KINDS = ['up', 'down', 'alt', 'q_mod4', 'q_mod6', 'step_mod3']
ORIGIN_KINDS = ['home', 'twin_pair', 'top_twin', 'top_twin_root']

class Machine:
    def __init__(self, q, sv):
        self.q = q; self.sv = sv
        ps = list(primerange(2, q + 1)); self.ps = ps
        base = []; B = 1
        for p in ps:
            if B * p <= q // 2: B *= p; base.append(p)
            else: break
        self.base = base; self.B = B
        self.above = [x for x in ps if x not in base]
        self.below = ps[::-1]                       # q, p1, p2, ...
        prims = []; P = 1
        for p in ps:
            P *= p
            if P <= q * q // 2: prims.append(P)
            else: break
        self.prims = prims
        self.twins = [x for x in ps if x + 2 in ps]  # twin gear pairs (left members)
        self.root = max([x for x in ps if x * x <= q], default=None)
        self.gaps = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]

def mirror_value(m, rule):
    kind, a = rule['kind'], rule.get('a', 0)
    if kind == 'B': return m.B
    if kind == 'Pmax': return m.prims[-1]
    if kind == 'Pmax_prev': return m.prims[-2] if len(m.prims) > 1 else None
    if kind == 'B_x_above': return m.B * m.above[a] if a < len(m.above) else None
    if kind == 'B_x_below': return m.B * m.below[a] if a < len(m.below) else None
    if kind == 'six_x_above': return 6 * m.above[a] if a < len(m.above) else None
    if kind == 'six_x_below': return 6 * m.below[a] if a < len(m.below) else None
    if kind == 'B_x_twin': return m.B * m.twins[-1 - a] if a < len(m.twins) else None
    if kind == 'B_x_root': return m.B * m.root if m.root else None
    if kind == 'two_x_below': return 2 * m.below[a] if a < len(m.below) else None
    if kind == 'B_x_above_pair': return m.B * m.above[a] * m.above[a + 1] if a + 1 < len(m.above) else None
    return None

def origin_value(m, rule):
    kind, a = rule['kind'], rule.get('a', 0)
    if kind == 'home': return -1
    if kind == 'twin_pair': return m.twins[a] if a < len(m.twins) else None
    if kind == 'top_twin': return m.twins[-1] if m.twins else None
    if kind == 'top_twin_root': return max([x for x in m.twins if x * x <= m.q], default=None)
    return None

def direction(rule, step, m):
    kind = rule['kind']
    if kind == 'up': return 1
    if kind == 'down': return -1
    if kind == 'alt': return 1 if step % 2 == 0 else -1
    if kind == 'q_mod4': return 1 if m.q % 4 == 1 else -1
    if kind == 'q_mod6': return 1 if m.q % 6 == 1 else -1
    if kind == 'step_mod3': return 1 if step % 3 != 2 else -1
    return 1

def period(rule, m, O, M, step):
    kind, a = rule['kind'], rule.get('a', 1)
    q = m.q
    if kind == 'const': return max(1, a)
    if kind == 'enter':
        j = 1
        while O + 2 * j * M <= q and j < 10 ** 6: j += 1
        return j
    if kind == 'top':
        j = (q * q - 2 - O) // (2 * M)
        return j if j >= 1 else None
    if kind == 'q_mod': return (q % max(2, a)) or 1
    if kind == 'count_mod': return (len(m.above) % max(2, a)) or 1
    if kind == 'gap_mod': return ((m.gaps[-1] // 2) % max(2, a)) or 1
    return 1

def run_walk(genome, m):
    O = origin_value(m, genome['origin'])
    if O is None: return None
    L = O
    for i, st in enumerate(genome['steps']):
        M = mirror_value(m, st['mirror'])
        if M is None: return None
        k = period(st['period'], m, L, M, i)
        if k is None: return None
        d = direction(st['dir'], i, m)
        L = L + 2 * k * M * d
        if L < -1 or L > 4 * m.q * m.q: return None
    return L

def fitness(genome, machines):
    # primary: the streak of consecutive machines from the smallest that land on a twin in the window;
    # secondary: the total number of machines that do (selection pressure while the streak is short);
    # then fewer steps
    # streak31: the run of consecutive successes from q = 31 (the machines 11..29 have windows of
    # at most 841 and the base {2,3}; a rule's failure there says little); streak11 from q = 11
    oks = []
    for m in machines:
        L = run_walk(genome, m)
        oks.append(L is not None and m.q < L <= m.q * m.q - 2 and bool(m.sv[L] and m.sv[L + 2]))
    def streak(start):
        n = 0
        for m, ok in zip(machines, oks):
            if m.q < start: continue
            if ok: n += 1
            else: break
        return n
    return (streak(31), streak(11), sum(oks), -len(genome['steps']))

def random_rule(kinds, amax):
    return {'kind': random.choice(kinds), 'a': random.randint(0, amax)}

def random_step():
    return {'mirror': random_rule(MIRROR_KINDS, 4), 'period': random_rule(PERIOD_KINDS, 5), 'dir': random_rule(DIR_KINDS, 0)}

def random_genome():
    return {'origin': random_rule(ORIGIN_KINDS, 3), 'steps': [random_step() for _ in range(random.randint(1, 4))]}

def mutate(g):
    g = json.loads(json.dumps(g)); r = random.random()
    if r < 0.15 and len(g['steps']) < 8: g['steps'].insert(random.randint(0, len(g['steps'])), random_step())
    elif r < 0.25 and len(g['steps']) > 1: g['steps'].pop(random.randrange(len(g['steps'])))
    elif r < 0.35: g['origin'] = random_rule(ORIGIN_KINDS, 3)
    else:
        st = random.choice(g['steps']); which = random.choice(['mirror', 'period', 'dir'])
        if random.random() < 0.5: st[which] = random_rule({'mirror': MIRROR_KINDS, 'period': PERIOD_KINDS, 'dir': DIR_KINDS}[which], {'mirror': 4, 'period': 5, 'dir': 0}[which])
        else: st[which]['a'] = max(0, st[which].get('a', 0) + random.choice([-1, 1]))
    return g

def crossover(a, b):
    cut_a = random.randint(0, len(a['steps'])); cut_b = random.randint(0, len(b['steps']))
    steps = a['steps'][:cut_a] + b['steps'][cut_b:]
    return {'origin': random.choice([a['origin'], b['origin']]), 'steps': (steps or [random_step()])[:8]}

def base_only(g):
    fixed_m = {'B', 'Pmax', 'Pmax_prev', 'B_x_above', 'six_x_above', 'B_x_above_pair'}
    return all(st['mirror']['kind'] in fixed_m and st['period']['kind'] == 'const' and st['dir']['kind'] in ('up', 'down', 'alt', 'step_mod3') for st in g['steps']) and g['origin']['kind'] in ('home', 'twin_pair')

def describe(g):
    o = g['origin']; s = ", ".join(f"{st['mirror']['kind']}[{st['mirror'].get('a', 0)}] x{st['period']['kind']}[{st['period'].get('a', 1)}] {st['dir']['kind']}" for st in g['steps'])
    return f"origin {o['kind']}[{o.get('a', 0)}]; steps: {s}"

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    N = QMAX * QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    machines = [Machine(q, sv) for q in primerange(11, QMAX + 1)]
    out_path = Path("research/stack/r8/results_evolve_walk.json")
    pop = []
    if out_path.exists():
        try: pop = [g for g in json.loads(out_path.read_text(encoding="utf-8")).get('elite', [])]
        except Exception: pop = []
    seeds = [
        {'origin': {'kind': 'home', 'a': 0}, 'steps': [{'mirror': {'kind': 'B_x_above', 'a': 1}, 'period': {'kind': 'const', 'a': 2}, 'dir': {'kind': 'down'}}, {'mirror': {'kind': 'B_x_above', 'a': 2}, 'period': {'kind': 'const', 'a': 3}, 'dir': {'kind': 'up'}}]},
        {'origin': {'kind': 'twin_pair', 'a': 1}, 'steps': [{'mirror': {'kind': 'B_x_above', 'a': 1}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'up'}}, {'mirror': {'kind': 'B_x_above', 'a': 2}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'up'}}]},
        {'origin': {'kind': 'twin_pair', 'a': 3}, 'steps': [{'mirror': {'kind': 'B_x_below', 'a': 0}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'up'}}, {'mirror': {'kind': 'B_x_below', 'a': 1}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'down'}}]},
        {'origin': {'kind': 'home', 'a': 0}, 'steps': [{'mirror': {'kind': 'B_x_below', 'a': 0}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'up'}}, {'mirror': {'kind': 'B_x_below', 'a': 1}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'down'}}]},
    ]
    pop += seeds
    while len(pop) < popn: pop.append(random_genome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness(g, machines), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:20]]
        best_f, best_g = scored[0]
        start = next(i for i, m in enumerate(machines) if m.q >= 31)
        line = f"gen {gen}: streak from 31: {best_f[0]} machines (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); from 11: {best_f[1]}; total {best_f[2]} of {len(machines)}; {-best_f[3]} steps; base-only {base_only(best_g)}: {describe(best_g)}"
        log.append(line); print(line, flush=True)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': describe(best_g), 'log': log[-50:]}, indent=1), encoding="utf-8")
        pool = [g for f, g in scored[:60]]
        newpop = list(elite)
        while len(newpop) < popn:
            r = random.random()
            if r < 0.15: newpop.append(random_genome())
            elif r < 0.45: newpop.append(crossover(random.choice(pool), random.choice(pool)))
            else:
                g = random.choice(pool)
                for _ in range(random.randint(1, 3)): g = mutate(g)
                newpop.append(g)
        pop = newpop

if __name__ == "__main__":
    main()

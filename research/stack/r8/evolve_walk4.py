"""Evolutionary search, fourth form (owner, 2026-09-15): islands seeded with the hand-built
families, evolving independently, so the families compete within themselves and not against
the per-range constants.

Islands (each its own population, no migration), with a membership rule enforced in fitness
(a genome outside its island's family scores zero):
  spiral        must contain a spiral step and no descent or levels step
  spiral+flip   a spiral step followed by at least one flip
  levels        must contain the levels step
  descent       must contain a descent step
  flips         flips only, and not base-only (some mirror or period must vary with q beyond
                the base: a mirror from the gears below q, above sqrt q, the twin gears, or a
                period rule 'enter' / 'top' / residues of q)
Machinery, grammar, operators and fitness from evolve_walk3.py; machines: every prime to 200
then every third to 6000 (288).  Each island's elite is written to results_evolve_walk4_<island>.json.
usage: uv run python research/stack/r8/evolve_walk4.py [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

VARYING_M = {'B_x_below', 'six_x_below', 'B_x_twin', 'B_x_root', 'two_x_below', 'three_x_hi', 'six_x_hi', 'B_x_hi', 'B_x_toptwin', 'thirty_x_hi'}
VARYING_P = {'enter', 'top', 'q_mod', 'count_mod', 'gap_mod'}

def member(island, g):
    types = [st['type'] for st in g['steps']]
    if island == 'spiral': return 'spiral' in types and 'descent' not in types and 'levels' not in types
    if island == 'spiral+flip':
        if 'spiral' not in types or 'flip' not in types: return False
        return types.index('spiral') < len(types) - 1 and types[-1] == 'flip'
    if island == 'levels': return 'levels' in types
    if island == 'descent': return 'descent' in types
    if island == 'flips':
        if any(t != 'flip' for t in types): return False
        return any(st['mirror']['kind'] in VARYING_M or st['period']['kind'] in VARYING_P for st in g['steps'])
    return False

def island_seeds(island):
    S = [g for g in E.seeds() if member(island, g)]
    return S

def random_member(island):
    for _ in range(200):
        g = E.rgenome()
        if island == 'spiral': g['steps'] = [E.rspiral()] + [E.rflip() for _ in range(random.randint(0, 1))]
        if island == 'spiral+flip': g['steps'] = [E.rspiral()] + [E.rflip() for _ in range(random.randint(1, 2))]
        if island == 'levels': g['steps'] = [{'type': 'levels'}] + [E.rflip() for _ in range(random.randint(0, 2))]
        if island == 'descent': g['steps'] = [E.rdescent()] + [E.rflip() for _ in range(random.randint(0, 2))]
        if island == 'flips': g['steps'] = [E.rflip() for _ in range(random.randint(1, 3))]
        if member(island, g): return g
    return g

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    prefix = sys.argv[4] if len(sys.argv) > 4 else ''
    random.seed(seed)
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    start = next(i for i, m in enumerate(machines) if m.q >= 31)
    print(f"machines {len(machines)} (11 to {qs[-1]})", flush=True)
    summary = []
    for island in ('spiral', 'spiral+flip', 'levels', 'descent', 'flips'):
        out_path = Path(f"research/stack/r8/results_evolve_walk4_{prefix}{island}.json")
        pop = []
        if out_path.exists():
            try: pop = [g for g in json.loads(out_path.read_text(encoding="utf-8")).get('elite', []) if member(island, g)]
            except Exception: pop = []
        pop += island_seeds(island)
        while len(pop) < popn: pop.append(random_member(island))
        def fit(g): return E.fitness(g, machines) if member(island, g) else (0, 0, 0, -99)
        log = []; best_f = None; best_g = None
        for gen in range(gens):
            scored = sorted(((fit(g), g) for g in pop), key=lambda t: t[0], reverse=True)
            elite = [g for f, g in scored[:24]]; best_f, best_g = scored[0]
            line = f"[{island}] gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); total {best_f[2]} of {len(machines)}; {-best_f[3]} steps: {E.describe(best_g)}"
            log.append(line)
            if gen % 10 == 0 or gen == gens - 1: print(line, flush=True)
            out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': E.describe(best_g), 'log': log[-40:]}, indent=1), encoding="utf-8")
            pool = [g for f, g in scored[:60]]
            newpop = list(elite)
            while len(newpop) < popn:
                r = random.random()
                if r < 0.12: newpop.append(random_member(island))
                elif r < 0.42: newpop.append(E.crossover(random.choice(pool), random.choice(pool)))
                else:
                    g = random.choice(pool)
                    for _ in range(random.randint(1, 3)): g = E.mutate(g)
                    newpop.append(g)
            pop = newpop
        summary.append((island, best_f, E.describe(best_g)))
    print("\nislands, final best:")
    for island, f, d in summary: print(f"   {island:<12} streak {f[0]:>3} total {f[2]:>3} steps {-f[3]}: {d}")

if __name__ == "__main__":
    main()

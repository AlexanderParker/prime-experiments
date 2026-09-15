"""Evolutionary search, seventh form (loop, 2026-09-15): two changed targets.

mode 'varying': the fitness of the sixth form, but a genome whose landing depends on the base
    alone (every mirror from the base and the gears just above it, constant periods, fixed
    origin) scores zero: the per-range constants are barred, so the best machine-varying walk
    shows.  A genome counts as base-only when its landing is the same for every machine that
    shares a base (checked directly: the landings within each base range must all be equal).
mode 'teeth': the primary target is the residues, not the twin: the mean number of the machine's
    gears on a tooth at the landing (fewer better), then the streak, then the total; to see what
    structure evolves when the walk is asked to carry residues rather than to hit the opening.
usage: uv run python research/stack/r8/evolve_walk7.py mode [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E
import evolve_walk6 as G

def landing_map(genome, machines):
    return [G.run_walk(genome, m) for m in machines]

def base_only_by_landings(lands, machines):
    by_base = {}
    for L, m in zip(lands, machines):
        by_base.setdefault(m.B, set()).add(L)
    return all(len(s) <= 1 for s in by_base.values())

def fitness(mode, genome, machines):
    lands = landing_map(genome, machines)
    oks = []; teeth = []
    for L, m in zip(lands, machines):
        ok = L is not None and m.q < L <= m.q * m.q - 2 and bool(m.sv[L] and m.sv[L + 2])
        oks.append(ok)
        if L is not None and m.q < L <= m.q * m.q - 2: teeth.append(sum(1 for g in m.ps if g >= 5 and L % g in (0, g - 2)))
        else: teeth.append(len(m.ps))
    def streak(start):
        n = 0
        for m, ok in zip(machines, oks):
            if m.q < start: continue
            if ok: n += 1
            else: break
        return n
    mt = -round(sum(teeth) / len(teeth), 3); pure = 0 if G.uses_residues(genome) else 1
    if mode == 'varying':
        if base_only_by_landings(lands, machines): return (0, 0, 0, -99, 0, -9)
        return (streak(31), streak(11), sum(oks), mt, pure, -len(genome['steps']))
    return (mt, streak(31), sum(oks), pure, -len(genome['steps']))

def main():
    mode = sys.argv[1]
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    popn = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    random.seed(seed)
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    start = next(i for i, m in enumerate(machines) if m.q >= 31)
    out_path = Path(f"research/stack/r8/results_evolve_walk7_{mode}.json")
    pop = []
    if out_path.exists():
        try: pop = list(json.loads(out_path.read_text(encoding="utf-8")).get('elite', []))
        except Exception: pop = []
    pop += G.seeds()
    while len(pop) < popn: pop.append(G.rgenome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness(mode, g, machines), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:24]]; best_f, best_g = scored[0]
        if mode == 'varying':
            line = f"[{mode}] gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); total {best_f[2]} of {len(machines)}; mean on a tooth {-best_f[3]}; pure {best_f[4]}; {-best_f[5]} steps: {E.describe(best_g)}"
        else:
            line = f"[{mode}] gen {gen}: mean on a tooth {-best_f[0]}; streak from 31: {best_f[1]}; total {best_f[2]} of {len(machines)}; pure {best_f[3]}; {-best_f[4]} steps: {E.describe(best_g)}"
        log.append(line)
        if gen % 10 == 0 or gen == gens - 1: print(line, flush=True)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': E.describe(best_g), 'log': log[-60:]}, indent=1), encoding="utf-8")
        pool = [g for f, g in scored[:70]]
        newpop = list(elite)
        while len(newpop) < popn:
            r = random.random()
            if r < 0.12: newpop.append(G.rgenome())
            elif r < 0.42: newpop.append(E.crossover(random.choice(pool), random.choice(pool)))
            else:
                g = random.choice(pool)
                for _ in range(random.randint(1, 3)): g = G.mutate(g)
                newpop.append(g)
        pop = newpop

if __name__ == "__main__":
    main()

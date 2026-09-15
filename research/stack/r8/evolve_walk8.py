"""Evolutionary search, eighth form (owner, 2026-09-15): fitness by how close the landing is
to a twin.

For each machine the walk's landing L (a column, 5 mod 6) is scored by its distance in slots
to the nearest twin prime pair inside the window: 0 when the landing is a twin; a landing
outside the window scores the distance to the window's edge plus the width of the window.
Primary fitness = the mean of log2(1 + distance) over the machines (lower is better, so a walk
that lands on or beside twins everywhere beats one that hits some and misses others by far);
then the streak from 31, then the total, then purity, then fewer steps.  Grammar, operators
and seeds of the sixth form (mirror flips, residue-chosen gears and periods, settled-set
mirrors, spirals, descent, levels).
usage: uv run python research/stack/r8/evolve_walk8.py [generations] [population] [seed]
"""
import sys, json, random, bisect, math
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E
import evolve_walk6 as G

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    idx = np.arange(5, N - 2, 6); twins = idx[sv[idx] & sv[idx + 2]].tolist()
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    start = next(i for i, m in enumerate(machines) if m.q >= 31)
    def distance(m, L):
        lo, hi = m.q + 1, m.q * m.q - 2
        width = (hi - lo) // 6 + 1
        if L is None: return 4 * width
        if L < lo or L > hi: return width + (lo - L if L < lo else L - hi) // 6 + 1
        j = bisect.bisect_left(twins, L)
        best = None
        for c in (twins[j - 1] if j > 0 else None, twins[j] if j < len(twins) else None):
            if c is not None and lo <= c <= hi:
                d = abs(c - L) // 6
                best = d if best is None else min(best, d)
        return best if best is not None else width
    def fitness(genome):
        ds = []; oks = []
        for m in machines:
            L = G.run_walk(genome, m); d = distance(m, L)
            ds.append(math.log2(1 + d)); oks.append(d == 0)
        def streak(s):
            n = 0
            for m, ok in zip(machines, oks):
                if m.q < s: continue
                if ok: n += 1
                else: break
            return n
        return (-round(sum(ds) / len(ds), 4), streak(31), sum(oks), 0 if G.uses_residues(genome) else 1, -len(genome['steps']))
    out_path = Path("research/stack/r8/results_evolve_walk8.json")
    pop = []
    if out_path.exists():
        try: pop = list(json.loads(out_path.read_text(encoding="utf-8")).get('elite', []))
        except Exception: pop = []
    pop += G.seeds()
    while len(pop) < popn: pop.append(G.rgenome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness(g), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:24]]; best_f, best_g = scored[0]
        line = f"gen {gen}: mean log2(1 + distance) {-best_f[0]}; streak from 31: {best_f[1]} (to q = {machines[start + best_f[1] - 1].q if best_f[1] else '-'}); twins {best_f[2]} of {len(machines)}; pure {best_f[3]}; {-best_f[4]} steps: {E.describe(best_g)}"
        if gen == 0:
            for f, g in scored[:6]: print(f"   seed rank: distance {-f[0]}, streak {f[1]}, twins {f[2]}: {E.describe(g)}", flush=True)
        log.append(line)
        if gen % 5 == 0 or gen == gens - 1: print(line, flush=True)
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

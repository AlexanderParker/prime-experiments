"""Evolutionary search, second form (2026-09-15): machines sampled across many primorial ranges.

evolve_walk.py scored genomes on every machine to 3000, which spans four base ranges (bases 6,
30, 210, 2310); its winner was a per-range constant (one landing per range, three twins by
chance).  Here the machines are sampled across the ranges of the bases 6 to 43# (q from 11 to
about 6 * 10^15), eight machines per range spread over the range, so a rule must work across
eighteen ranges and cannot be a short list of constants.  Twin test by sympy.isprime on the
landing (no sieve).  Genome, grammar, operators as in evolve_walk.py.  Fitness: the streak of
consecutive sampled machines from the smallest with q >= 31, then from 11, then total, then
fewer steps.

usage: uv run python research/stack/r8/evolve_walk2.py [generations] [population] [seed]
"""
import sys, json, random
from sympy import primerange, isprime, nextprime
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk as ew

class MachineBig:
    def __init__(self, q):
        self.q = q
        ps = list(primerange(2, min(q, 2000) + 1)) if q > 2000 else list(primerange(2, q + 1))
        self.ps = ps
        base = []; B = 1
        for p in ps:
            if B * p <= q // 2: B *= p; base.append(p)
            else: break
        self.base = base; self.B = B
        self.above = [x for x in ps if x not in base][:8]
        # gears below q: q itself and the primes just below it
        below = [q]; x = q
        for _ in range(6):
            if x <= 2: break
            x = int(__import__('sympy').prevprime(x)); below.append(x)
        self.below = below
        prims = []; P = 1
        for p in primerange(2, 200):
            P *= p
            if P <= q * q // 2: prims.append(P)
            else: break
        self.prims = prims
        tw = [x for x in ps if x + 2 in ps]
        # top twin gear pairs near q for large q: search downward from q
        if q > 2000:
            x = q; found = []
            while len(found) < 2 and x > 3:
                x = int(__import__('sympy').prevprime(x))
                if isprime(x + 2): found.append(x)
            tw = tw + found[::-1]
        self.twins = tw
        r = int(q ** 0.5); self.root = int(__import__('sympy').prevprime(r + 1)) if r >= 3 else (2 if r == 2 else None)
        self.gaps = [q - below[1]]
        self.sv = None

def ok_landing(m, L):
    return L is not None and m.q < L <= m.q * m.q - 2 and isprime(L) and isprime(L + 2)

def fitness_big(genome, machines):
    oks = [ok_landing(m, ew.run_walk(genome, m)) for m in machines]
    def streak(start):
        n = 0
        for m, ok in zip(machines, oks):
            if m.q < start: continue
            if ok: n += 1
            else: break
        return n
    return (streak(31), streak(11), sum(oks), -len(genome['steps']))

def sample_machines():
    qs = []
    P = 1; prev = 1
    for p in primerange(2, 44):
        P *= p
        if P < 6: prev = P; continue
        lo, hi = 2 * prev, 2 * P            # machines with q/2 in [prev, P): base = prev's gears
        for i in range(8):
            x = lo + (hi - lo) * (2 * i + 1) // 16
            qs.append(int(nextprime(x)))
        prev = P
    qs = sorted(set(q for q in qs if q >= 11))
    return qs

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    qs = sample_machines()
    machines = [MachineBig(q) for q in qs]
    print(f"machines sampled: {len(machines)}, q from {qs[0]} to {qs[-1]}", flush=True)
    out_path = Path("research/stack/r8/results_evolve_walk2.json")
    pop = []
    if out_path.exists():
        try: pop = list(json.loads(out_path.read_text(encoding="utf-8")).get('elite', []))
        except Exception: pop = []
    while len(pop) < popn: pop.append(ew.random_genome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness_big(g, machines), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:20]]
        best_f, best_g = scored[0]
        start = next(i for i, m in enumerate(machines) if m.q >= 31)
        line = f"gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); from 11: {best_f[1]}; total {best_f[2]} of {len(machines)}; {-best_f[3]} steps; base-only {ew.base_only(best_g)}: {ew.describe(best_g)}"
        log.append(line); print(line, flush=True)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': ew.describe(best_g), 'machines': qs, 'log': log[-60:]}, indent=1), encoding="utf-8")
        pool = [g for f, g in scored[:60]]
        newpop = list(elite)
        while len(newpop) < popn:
            r = random.random()
            if r < 0.15: newpop.append(ew.random_genome())
            elif r < 0.45: newpop.append(ew.crossover(random.choice(pool), random.choice(pool)))
            else:
                g = random.choice(pool)
                for _ in range(random.randint(1, 3)): g = ew.mutate(g)
                newpop.append(g)
        pop = newpop

if __name__ == "__main__":
    main()

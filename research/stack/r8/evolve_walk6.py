"""Evolutionary search, sixth form (owner: keep experimenting, 2026-09-15).

Two additions to the fifth form:
  1. multi-gear mirrors chosen by residues: at the current column c, the gears whose phase is
     off their two teeth are 'settled'; a mirror may be B times the product of the first (a+1)
     settled gears above the base (set_free[a]), or of the first (a+1) settled gears above sqrt q
     (set_free_hi[a]), or B times ALL settled gears above the base whose product keeps the mirror
     at most q^2 / 4 (set_free_fit): a flip about such a mirror carries every settled gear in it.
  2. a fitness term for residues: after the streaks and the total, the mean number of gears of
     the machine on a tooth at the landing (fewer is better), then purity, then fewer steps.
usage: uv run python research/stack/r8/evolve_walk6.py [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E
import evolve_walk5 as F

SET_M = ['set_free', 'set_free_hi', 'set_free_fit']
FLIP_M = F.FLIP_M + SET_M

def settled(gs, c): return [g for g in gs if c % g not in (0, g - 2)]

def mirror_value(m, rule, c):
    k, a = rule['kind'], rule.get('a', 0)
    if k in SET_M:
        pool = m.above if k != 'set_free_hi' else m.hi
        s = settled(pool, c)
        M = m.B
        if k == 'set_free_fit':
            for g in s:
                if M * g <= m.q * m.q // 4: M *= g
                else: break
            return M
        for g in s[:a + 1]: M *= g
        return M if s[:a + 1] else None
    return F.mirror_value(m, rule, c)

def apply_step(st, m, L, i):
    if st['type'] == 'flip':
        M = mirror_value(m, st['mirror'], L)
        if M is None: return None
        k = F.period(st['period'], m, L, M, i)
        if k is None: return None
        return L + 2 * k * M * E.direction(st['dir'], i, m)
    return E.apply_step(st, m, L, i)

def run_walk(genome, m):
    L = E.origin_value(m, genome['origin'])
    if L is None: return None
    for i, st in enumerate(genome['steps']):
        L = apply_step(st, m, L, i)
        if L is None or L < -1 or L > 4 * m.q * m.q: return None
    return L

def uses_residues(g):
    return F.uses_residues(g) or any(st['type'] == 'flip' and st['mirror']['kind'] in SET_M for st in g['steps'])

def fitness(genome, machines):
    oks = []; teeth = []
    for m in machines:
        L = run_walk(genome, m)
        ok = L is not None and m.q < L <= m.q * m.q - 2 and bool(m.sv[L] and m.sv[L + 2])
        oks.append(ok)
        if L is not None and m.q < L <= m.q * m.q - 2:
            teeth.append(sum(1 for g in m.ps if g >= 5 and L % g in (0, g - 2)))
        else: teeth.append(len(m.ps))
    def streak(start):
        n = 0
        for m, ok in zip(machines, oks):
            if m.q < start: continue
            if ok: n += 1
            else: break
        return n
    return (streak(31), streak(11), sum(oks), -round(sum(teeth) / len(teeth), 3), 0 if uses_residues(genome) else 1, -len(genome['steps']))

def rflip(): return {'type': 'flip', 'mirror': E.rrule(FLIP_M, 4), 'period': E.rrule(F.PERIODS, 5), 'dir': E.rrule(E.DIRS, 0)}
def rstep():
    r = random.random()
    return rflip() if r < 0.7 else E.rspiral() if r < 0.88 else E.rdescent() if r < 0.96 else {'type': 'levels'}
def rgenome(): return {'origin': E.rrule(E.ORIGINS, 3), 'steps': [rstep() for _ in range(random.randint(1, 3))]}

def mutate(g):
    g = json.loads(json.dumps(g)); r = random.random()
    if r < 0.15 and len(g['steps']) < 6: g['steps'].insert(random.randint(0, len(g['steps'])), rstep())
    elif r < 0.25 and len(g['steps']) > 1: g['steps'].pop(random.randrange(len(g['steps'])))
    elif r < 0.33: g['origin'] = E.rrule(E.ORIGINS, 3)
    else:
        st = random.choice(g['steps'])
        if st['type'] == 'flip':
            w = random.choice(['mirror', 'period', 'dir'])
            if random.random() < 0.5: st[w] = E.rrule({'mirror': FLIP_M, 'period': F.PERIODS, 'dir': E.DIRS}[w], {'mirror': 4, 'period': 5, 'dir': 0}[w])
            else: st[w]['a'] = max(0, st[w].get('a', 0) + random.choice([-1, 1]))
        else:
            g2 = E.mutate({'origin': g['origin'], 'steps': [st]}); g['steps'][g['steps'].index(st)] = g2['steps'][0]
    return g

def seeds():
    S = F.seeds(); home = {'kind': 'home', 'a': 0}
    for first in ([], [{'type': 'spiral', 'set': 'sqout', 'order': 'desc', 'start': 'up', 'base': 'spiral', 'k': 1, 'a': 3}], [{'type': 'levels'}]):
        for mk in SET_M:
            for a in (0, 1, 2):
                for d in ('up', 'down'):
                    S.append({'origin': home, 'steps': first + [{'type': 'flip', 'mirror': {'kind': mk, 'a': a}, 'period': {'kind': 'enter', 'a': 1}, 'dir': {'kind': d}}]})
    # two set-free flips in a row
    S.append({'origin': home, 'steps': [{'type': 'flip', 'mirror': {'kind': 'set_free_fit', 'a': 0}, 'period': {'kind': 'enter', 'a': 1}, 'dir': {'kind': 'up'}}, {'type': 'flip', 'mirror': {'kind': 'set_free_fit', 'a': 0}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'up'}}]})
    return S

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    start = next(i for i, m in enumerate(machines) if m.q >= 31)
    print(f"machines {len(machines)} (11 to {qs[-1]})", flush=True)
    out_path = Path("research/stack/r8/results_evolve_walk6.json")
    pop = []
    if out_path.exists():
        try: pop = list(json.loads(out_path.read_text(encoding="utf-8")).get('elite', []))
        except Exception: pop = []
    pop += seeds()
    while len(pop) < popn: pop.append(rgenome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness(g, machines), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:24]]; best_f, best_g = scored[0]
        line = f"gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); from 11: {best_f[1]}; total {best_f[2]} of {len(machines)}; mean gears on a tooth {-best_f[3]}; pure {best_f[4]}; {-best_f[5]} steps: {E.describe(best_g)}"
        if gen == 0:
            for f, g in scored[:8]: print(f"   seed rank: streak31 {f[0]}, total {f[2]}, teeth {-f[3]}, pure {f[4]}: {E.describe(g)}", flush=True)
        log.append(line); print(line, flush=True)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': E.describe(best_g), 'log': log[-60:]}, indent=1), encoding="utf-8")
        pool = [g for f, g in scored[:70]]
        newpop = list(elite)
        while len(newpop) < popn:
            r = random.random()
            if r < 0.12: newpop.append(rgenome())
            elif r < 0.42: newpop.append(E.crossover(random.choice(pool), random.choice(pool)))
            else:
                g = random.choice(pool)
                for _ in range(random.randint(1, 3)): g = mutate(g)
                newpop.append(g)
        pop = newpop

if __name__ == "__main__":
    main()

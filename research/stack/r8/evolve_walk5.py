"""Evolutionary search, fifth form (owner, 2026-09-15): residue rules allowed for choosing
gears and periods, but no step may test candidate landings for openness or primality.

Added to the grammar of evolve_walk3.py:
  gear-selection mirrors (the flip's gear h is chosen among the gears above sqrt q by a residue
  rule read at the CURRENT column c, before the flip):
    hi_free[a]      the (a+1)-th gear h above sqrt q whose phase c mod h is off its two teeth
                    (0 and h - 2): the flip about {3, h} keeps h's phase, so h cannot strike the
                    landing -- a provable step
    hi_far[a]       the gear above sqrt q whose phase is farthest from its nearer tooth (a-th best)
    hi_mod6[a]      the first gear above sqrt q with h = 1 (a even) or 5 (a odd) mod 6
    hi_adj[a]       the a-th gear above sqrt q with 2 P_s = 1 mod h (teeth adjacent on the t-line)
    hi_res[a]       the first gear above sqrt q with q mod h in the middle third of (0, h)
  period rules reading one gear's phase (steering one gear, never testing the landing as a whole):
    k_avoid[a]      the smallest k with the landing off the teeth of the a-th gear above the base
    k_center[a]     the k (mod that gear) placing that gear's phase at the centre of its open arc
Fitness: streak from 31, streak from 11, total, purity (1 when no residue rule is used, so pure
genomes win ties), fewer steps.  Machines as evolve_walk3.  Seeds: the hand-built walks and the
residue-selected final flips (spiral / levels / descent then a flip on 3h or 6h with h = hi_free).
usage: uv run python research/stack/r8/evolve_walk5.py [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

RES_M = ['hi_free', 'hi_far', 'hi_mod6', 'hi_adj', 'hi_res']
RES_P = ['k_avoid', 'k_center']
FLIP_M = E.FLIP_M + ['three_x_' + r for r in RES_M] + ['six_x_' + r for r in RES_M] + ['B_x_' + r for r in RES_M]
PERIODS = E.PERIODS + RES_P

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def select_gear(m, rule, a, c):
    hs = m.hi
    if not hs: return None
    if rule == 'hi_free':
        ok = [h for h in hs if c % h not in (0, h - 2)]
        return ok[a] if a < len(ok) else None
    if rule == 'hi_far':
        order = sorted(hs, key=lambda h: -dist(c % h, h))
        return order[a] if a < len(order) else None
    if rule == 'hi_mod6':
        want = 1 if a % 2 == 0 else 5
        return next((h for h in hs if h % 6 == want), None)
    if rule == 'hi_adj':
        ok = [h for h in hs if (2 * m.Ps) % h == 1]
        return ok[a] if a < len(ok) else None
    if rule == 'hi_res':
        return next((h for h in hs if h // 3 < m.q % h < 2 * h // 3), None)
    return None

def mirror_value(m, rule, c):
    k, a = rule['kind'], rule.get('a', 0)
    for pre, mult in (('three_x_', 3), ('six_x_', 6), ('B_x_', None)):
        if k.startswith(pre) and k[len(pre):] in RES_M:
            h = select_gear(m, k[len(pre):], a, c)
            if h is None: return None
            return (mult or m.B) * h
    return E.mirror_value(m, rule)

def period(rule, m, O, M, step):
    k, a = rule['kind'], rule.get('a', 1)
    if k in RES_P:
        if a >= len(m.above): return None
        g = m.above[a]
        if k == 'k_avoid':
            for j in range(1, 4 * g):
                L = O + 2 * j * M
                if L % g not in (0, g - 2): return j
            return None
        if k == 'k_center':
            s = (2 * M) % g
            if s == 0: return 1
            target = (g - 2) // 2
            j = ((target - O % g) * pow(s, -1, g)) % g
            return j if j >= 1 else g
    return E.period(rule, m, O, M, step)

def apply_step(st, m, L, i):
    if st['type'] == 'flip':
        M = mirror_value(m, st['mirror'], L)
        if M is None: return None
        k = period(st['period'], m, L, M, i)
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
    for st in g['steps']:
        if st['type'] == 'flip':
            if any(st['mirror']['kind'].endswith(r) for r in RES_M) or st['period']['kind'] in RES_P: return True
    return False

def fitness(genome, machines):
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
    return (streak(31), streak(11), sum(oks), 0 if uses_residues(genome) else 1, -len(genome['steps']))

def rflip(): return {'type': 'flip', 'mirror': E.rrule(FLIP_M, 4), 'period': E.rrule(PERIODS, 5), 'dir': E.rrule(E.DIRS, 0)}
def rstep():
    r = random.random()
    return rflip() if r < 0.65 else E.rspiral() if r < 0.85 else E.rdescent() if r < 0.95 else {'type': 'levels'}
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
            if random.random() < 0.5: st[w] = E.rrule({'mirror': FLIP_M, 'period': PERIODS, 'dir': E.DIRS}[w], {'mirror': 4, 'period': 5, 'dir': 0}[w])
            else: st[w]['a'] = max(0, st[w].get('a', 0) + random.choice([-1, 1]))
        else:
            g2 = E.mutate({'origin': g['origin'], 'steps': [st]}); g['steps'][g['steps'].index(st)] = g2['steps'][0]
    return g

def seeds():
    S = E.seeds(); home = {'kind': 'home', 'a': 0}
    sp = lambda s: {'type': 'spiral', 'set': s, 'order': 'desc', 'start': 'up', 'base': 'spiral', 'k': 1, 'a': 3}
    for first in ([sp('all')], [sp('sqin')], [sp('sqout')], [{'type': 'levels'}], [{'type': 'descent', 't': {'kind': 'const', 'a': 1}}]):
        for mk in ('three_x_hi_free', 'six_x_hi_free', 'three_x_hi_far', 'B_x_hi_free'):
            for a in (0, 1):
                for d in ('up', 'down'):
                    S.append({'origin': home, 'steps': first + [{'type': 'flip', 'mirror': {'kind': mk, 'a': a}, 'period': {'kind': 'enter' if first[0]['type'] == 'levels' else 'const', 'a': 1}, 'dir': {'kind': d}}]})
    for a in range(3):
        S.append({'origin': home, 'steps': [{'type': 'descent', 't': {'kind': 'const', 'a': 1}}, {'type': 'flip', 'mirror': {'kind': 'B', 'a': 0}, 'period': {'kind': 'k_avoid', 'a': a}, 'dir': {'kind': 'up'}}]})
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
    out_path = Path("research/stack/r8/results_evolve_walk5.json")
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
        line = f"gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); from 11: {best_f[1]}; total {best_f[2]} of {len(machines)}; pure {best_f[3]}; {-best_f[4]} steps: {E.describe(best_g)}"
        if gen == 0:
            for f, g in scored[:8]: print(f"   seed rank: streak31 {f[0]}, total {f[2]}, pure {f[3]}: {E.describe(g)}", flush=True)
        log.append(line); print(line, flush=True)
        pure_best = next(((f, g) for f, g in scored if f[3] == 1), None)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': E.describe(best_g), 'best_pure': E.describe(pure_best[1]) if pure_best else None, 'best_pure_fitness': pure_best[0] if pure_best else None, 'log': log[-60:]}, indent=1), encoding="utf-8")
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

"""Evolutionary search, third form (owner, 2026-09-15): seeded with the best hand-built walks.

Genome = origin + a list of steps; a step is one of:
  flip     one mirror flip: mirror rule, period rule, direction rule (as evolve_walk.py), with
           extra mirrors 3 * g, 6 * g, B * g for g the a-th gear above sqrt q, and B * top twin gear
  spiral   the alternating spiral over a gear set: set in {all above the base, above sqrt q,
           at most sqrt q, column-twin gears, twin-member gears, solo gears, the top a gears};
           order desc or asc; first flip up or down; base in {spiral base, {2,3}, {2,3,5}};
           periods 1 or 2 per flip
  descent  flip up on the largest primorial P_s > q/2 ... the descent landing 2 t P_s - 1 with the
           t rule in {const, q_mod, count_mod, gap_mod}  (the walk's landing is replaced)
  levels   the recursive spiral (levels while b_k >= 11, base with 2 and 3 at every level)
No step tests a candidate column by division or primality; the gear sets above are properties
of the gears (known primes), not of the landing.
Fitness: streak of machines from q = 31, streak from 11, total, fewer steps.  Machines: every
prime to 200, then every third prime to 6000 (sieve).  Seeds: the primorial spiral, the sqin,
sqout, coltwin spirals, the levels walk, the descent at t = 1, 2, 3, the spiral followed by one
flip about {3, h} or {2,3,h} for h the a-th gear above sqrt q (a = 0..3, up and down), and the
grammar's best fixed rules.
usage: uv run python research/stack/r8/evolve_walk3.py [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path

QMAX = 6000
FLIP_M = ['B', 'Pmax', 'Pmax_prev', 'B_x_above', 'B_x_below', 'six_x_above', 'six_x_below', 'B_x_twin', 'B_x_root', 'two_x_below',
          'three_x_hi', 'six_x_hi', 'B_x_hi', 'B_x_toptwin', 'thirty_x_hi']
PERIODS = ['const', 'enter', 'top', 'q_mod', 'count_mod', 'gap_mod']
DIRS = ['up', 'down', 'alt', 'q_mod4', 'q_mod6', 'step_mod3']
ORIGINS = ['home', 'twin_pair', 'top_twin', 'top_twin_root']
SETS = ['all', 'sqin', 'sqout', 'coltwin', 'twinmem', 'solo', 'top_a']
BASES = ['spiral', '23', '235']

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
        self.below = ps[::-1]
        prims = []; P = 1
        for p in ps:
            P *= p
            if P <= q * q // 2: prims.append(P)
            else: break
        self.prims = prims
        self.twins = [x for x in ps if x + 2 in ps]
        r = int(q ** 0.5); self.r = r
        self.root = max([x for x in ps if x <= r], default=None)
        self.hi = [x for x in ps if x > r]
        self.gaps = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]
        self.sets = {'all': self.above, 'sqin': [x for x in self.above if x > r], 'sqout': [x for x in self.above if x <= r],
                     'coltwin': [x for x in self.above if sv[6 * x - 1] and sv[6 * x + 1]],
                     'twinmem': [x for x in self.above if sv[x - 2] or sv[x + 2]], 'solo': [x for x in self.above if not (sv[x - 2] or sv[x + 2])]}
        # first primorial above q/2 for the descent
        P = 1
        for p in ps:
            P *= p
            if P > q / 2: break
        self.Ps = P

def mirror_value(m, rule):
    k, a = rule['kind'], rule.get('a', 0)
    if k == 'B': return m.B
    if k == 'Pmax': return m.prims[-1]
    if k == 'Pmax_prev': return m.prims[-2] if len(m.prims) > 1 else None
    if k == 'B_x_above': return m.B * m.above[a] if a < len(m.above) else None
    if k == 'B_x_below': return m.B * m.below[a] if a < len(m.below) else None
    if k == 'six_x_above': return 6 * m.above[a] if a < len(m.above) else None
    if k == 'six_x_below': return 6 * m.below[a] if a < len(m.below) else None
    if k == 'B_x_twin': return m.B * m.twins[-1 - a] if a < len(m.twins) else None
    if k == 'B_x_root': return m.B * m.root if m.root else None
    if k == 'two_x_below': return 2 * m.below[a] if a < len(m.below) else None
    if k == 'three_x_hi': return 3 * m.hi[a] if a < len(m.hi) else None
    if k == 'six_x_hi': return 6 * m.hi[a] if a < len(m.hi) else None
    if k == 'B_x_hi': return m.B * m.hi[a] if a < len(m.hi) else None
    if k == 'thirty_x_hi': return 30 * m.hi[a] if a < len(m.hi) else None
    if k == 'B_x_toptwin': return m.B * m.twins[-1] if m.twins else None
    return None

def origin_value(m, rule):
    k, a = rule['kind'], rule.get('a', 0)
    if k == 'home': return -1
    if k == 'twin_pair': return m.twins[a] if a < len(m.twins) else None
    if k == 'top_twin': return m.twins[-1] if m.twins else None
    if k == 'top_twin_root': return max([x for x in m.twins if x * x <= m.q], default=None)
    return None

def direction(rule, step, m):
    k = rule['kind']
    return {'up': 1, 'down': -1, 'alt': 1 if step % 2 == 0 else -1, 'q_mod4': 1 if m.q % 4 == 1 else -1,
            'q_mod6': 1 if m.q % 6 == 1 else -1, 'step_mod3': 1 if step % 3 != 2 else -1}.get(k, 1)

def period(rule, m, O, M, step):
    k, a = rule['kind'], rule.get('a', 1); q = m.q
    if k == 'const': return max(1, a)
    if k == 'enter':
        j = 1
        while O + 2 * j * M <= q and j < 10 ** 6: j += 1
        return j
    if k == 'top':
        j = (q * q - 2 - O) // (2 * M); return j if j >= 1 else None
    if k == 'q_mod': return (q % max(2, a)) or 1
    if k == 'count_mod': return (len(m.above) % max(2, a)) or 1
    if k == 'gap_mod': return ((m.gaps[-1] // 2) % max(2, a)) or 1
    return 1

def base_value(m, b):
    if b == 'spiral': return m.B
    if b == '23': return 6
    return 30

def apply_step(st, m, L, i):
    t = st['type']
    if t == 'flip':
        M = mirror_value(m, st['mirror'])
        if M is None: return None
        k = period(st['period'], m, L, M, i)
        if k is None: return None
        return L + 2 * k * M * direction(st['dir'], i, m)
    if t == 'spiral':
        P = base_value(m, st['base'])
        gs = m.sets[st['set']] if st['set'] != 'top_a' else m.above[-(st.get('a', 3) + 1):]
        gs = [g for g in gs if P % g != 0]
        if not gs: return None
        gs = sorted(gs, reverse=(st['order'] == 'desc'))
        d = 1 if st['start'] == 'up' else -1; k = st.get('k', 1)
        for g in gs:
            L += 2 * k * P * g * d; d = -d
        return L
    if t == 'descent':
        tr = st['t']; k, a = tr['kind'], tr.get('a', 1)
        t = {'const': max(1, a), 'q_mod': (m.q % max(2, a)) or 1, 'count_mod': (len(m.above) % max(2, a)) or 1, 'gap_mod': ((m.gaps[-1] // 2) % max(2, a)) or 1}.get(k, 1)
        return 2 * t * m.Ps - 1
    if t == 'levels':
        bounds = [m.q]
        while int(bounds[-1] ** 0.5) >= 11: bounds.append(int(bounds[-1] ** 0.5))
        E = -1
        for kk in range(len(bounds) - 1, -1, -1):
            b = bounds[kk]; lo = bounds[kk + 1] if kk + 1 < len(bounds) else 1
            lps = [p for p in m.ps if p <= b]; bk = []; Pk = 1
            for p in lps:
                if Pk * p <= max(b // 2, 6): Pk *= p; bk.append(p)
                else: break
            gk = sorted([p for p in lps if p > lo and p not in bk], reverse=True)
            d = 1
            for g in gk: E += 2 * Pk * g * d; d = -d
        return E
    return None

def run_walk(genome, m):
    L = origin_value(m, genome['origin'])
    if L is None: return None
    for i, st in enumerate(genome['steps']):
        L = apply_step(st, m, L, i)
        if L is None or L < -1 or L > 4 * m.q * m.q: return None
    return L

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
    return (streak(31), streak(11), sum(oks), -len(genome['steps']))

def rrule(kinds, amax): return {'kind': random.choice(kinds), 'a': random.randint(0, amax)}
def rflip(): return {'type': 'flip', 'mirror': rrule(FLIP_M, 4), 'period': rrule(PERIODS, 5), 'dir': rrule(DIRS, 0)}
def rspiral(): return {'type': 'spiral', 'set': random.choice(SETS), 'order': random.choice(['desc', 'asc']), 'start': random.choice(['up', 'down']), 'base': random.choice(BASES), 'k': random.choice([1, 1, 2]), 'a': random.randint(1, 6)}
def rdescent(): return {'type': 'descent', 't': rrule(['const', 'q_mod', 'count_mod', 'gap_mod'], 6)}
def rstep():
    r = random.random()
    return rflip() if r < 0.6 else rspiral() if r < 0.85 else rdescent() if r < 0.95 else {'type': 'levels'}
def rgenome(): return {'origin': rrule(ORIGINS, 3), 'steps': [rstep() for _ in range(random.randint(1, 3))]}

def mutate(g):
    g = json.loads(json.dumps(g)); r = random.random()
    if r < 0.15 and len(g['steps']) < 6: g['steps'].insert(random.randint(0, len(g['steps'])), rstep())
    elif r < 0.25 and len(g['steps']) > 1: g['steps'].pop(random.randrange(len(g['steps'])))
    elif r < 0.33: g['origin'] = rrule(ORIGINS, 3)
    else:
        st = random.choice(g['steps'])
        if st['type'] == 'flip':
            w = random.choice(['mirror', 'period', 'dir'])
            if random.random() < 0.5: st[w] = rrule({'mirror': FLIP_M, 'period': PERIODS, 'dir': DIRS}[w], {'mirror': 4, 'period': 5, 'dir': 0}[w])
            else: st[w]['a'] = max(0, st[w].get('a', 0) + random.choice([-1, 1]))
        elif st['type'] == 'spiral':
            w = random.choice(['set', 'order', 'start', 'base', 'k', 'a'])
            st[w] = {'set': random.choice(SETS), 'order': random.choice(['desc', 'asc']), 'start': random.choice(['up', 'down']), 'base': random.choice(BASES), 'k': random.choice([1, 2]), 'a': random.randint(1, 6)}[w]
        elif st['type'] == 'descent': st['t'] = rrule(['const', 'q_mod', 'count_mod', 'gap_mod'], 6)
        else: g['steps'][g['steps'].index(st)] = rstep()
    return g

def crossover(a, b):
    ca = random.randint(0, len(a['steps'])); cb = random.randint(0, len(b['steps']))
    steps = (a['steps'][:ca] + b['steps'][cb:]) or [rstep()]
    return {'origin': random.choice([a['origin'], b['origin']]), 'steps': steps[:6]}

def describe(g):
    parts = []
    for st in g['steps']:
        if st['type'] == 'flip': parts.append(f"flip {st['mirror']['kind']}[{st['mirror'].get('a', 0)}] x{st['period']['kind']}[{st['period'].get('a', 1)}] {st['dir']['kind']}")
        elif st['type'] == 'spiral': parts.append(f"spiral {st['set']}{'[' + str(st.get('a', 3)) + ']' if st['set'] == 'top_a' else ''} {st['order']} {st['start']} base {st['base']} k{st.get('k', 1)}")
        elif st['type'] == 'descent': parts.append(f"descent t={st['t']['kind']}[{st['t'].get('a', 1)}]")
        else: parts.append("levels")
    o = g['origin']
    return f"origin {o['kind']}[{o.get('a', 0)}]; " + "; ".join(parts)

def seeds():
    S = []
    sp = lambda s, base='spiral', start='up': {'type': 'spiral', 'set': s, 'order': 'desc', 'start': start, 'base': base, 'k': 1, 'a': 3}
    home = {'kind': 'home', 'a': 0}
    for s in ('all', 'sqin', 'sqout', 'coltwin', 'twinmem', 'solo'):
        S.append({'origin': home, 'steps': [sp(s)]})
    S.append({'origin': home, 'steps': [{'type': 'levels'}]})
    for t in (1, 2, 3): S.append({'origin': home, 'steps': [{'type': 'descent', 't': {'kind': 'const', 'a': t}}]})
    for s in ('all', 'sqin', 'sqout'):
        for a in range(4):
            for d in ('up', 'down'):
                for mk in ('three_x_hi', 'six_x_hi'):
                    S.append({'origin': home, 'steps': [sp(s), {'type': 'flip', 'mirror': {'kind': mk, 'a': a}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': d}}]})
    S.append({'origin': home, 'steps': [{'type': 'levels'}, {'type': 'flip', 'mirror': {'kind': 'three_x_hi', 'a': 0}, 'period': {'kind': 'enter', 'a': 1}, 'dir': {'kind': 'up'}}]})
    S.append({'origin': {'kind': 'twin_pair', 'a': 3}, 'steps': [{'type': 'flip', 'mirror': {'kind': 'B_x_below', 'a': 0}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'up'}}, {'type': 'flip', 'mirror': {'kind': 'B_x_below', 'a': 1}, 'period': {'kind': 'const', 'a': 1}, 'dir': {'kind': 'down'}}]})
    S.append({'origin': home, 'steps': [{'type': 'flip', 'mirror': {'kind': 'B_x_above', 'a': 1}, 'period': {'kind': 'const', 'a': 2}, 'dir': {'kind': 'down'}}, {'type': 'flip', 'mirror': {'kind': 'B_x_above', 'a': 2}, 'period': {'kind': 'const', 'a': 3}, 'dir': {'kind': 'up'}}]})
    return S

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    N = QMAX * QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [Machine(q, sv) for q in qs]
    print(f"machines {len(machines)} (11 to {qs[-1]})", flush=True)
    out_path = Path("research/stack/r8/results_evolve_walk3.json")
    pop = []
    if out_path.exists():
        try: pop = list(json.loads(out_path.read_text(encoding="utf-8")).get('elite', []))
        except Exception: pop = []
    pop += seeds()
    while len(pop) < popn: pop.append(rgenome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness(g, machines), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:24]]
        best_f, best_g = scored[0]
        start = next(i for i, m in enumerate(machines) if m.q >= 31)
        line = f"gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); from 11: {best_f[1]}; total {best_f[2]} of {len(machines)}; {-best_f[3]} steps: {describe(best_g)}"
        if gen == 0:
            for f, g in scored[:6]: print(f"   seed rank: streak31 {f[0]}, total {f[2]}: {describe(g)}", flush=True)
        log.append(line); print(line, flush=True)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': describe(best_g), 'log': log[-60:]}, indent=1), encoding="utf-8")
        pool = [g for f, g in scored[:70]]
        newpop = list(elite)
        while len(newpop) < popn:
            r = random.random()
            if r < 0.12: newpop.append(rgenome())
            elif r < 0.42: newpop.append(crossover(random.choice(pool), random.choice(pool)))
            else:
                g = random.choice(pool)
                for _ in range(random.randint(1, 3)): g = mutate(g)
                newpop.append(g)
        pop = newpop

if __name__ == "__main__":
    main()

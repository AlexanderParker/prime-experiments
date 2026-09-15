"""Evolutionary search, ninth form (owner: loop, make changes until a successful run; 2026-09-15).

New macro step 'settle': the pick-up walk with bounded residue memory.  Over the gears above
the base in an order (desc / asc / small-first), the mirror at gear g is {base, g}; the period
and direction are chosen among k = 1..K, both directions, so that g and the last w visited
gears (the memory) are off their two teeth at the new column; among those, the 'flex' rule
prefers the move leaving the next gear the most such options, ties by the largest distance of
any remembered phase from a tooth.  If no move keeps the memory settled, the move with the
fewest remembered gears on a tooth.  After the walk, up to R repair flips about {base, s},
s a gear at most sqrt q, by the same rule over the memory.  The memory w is the parameter
that measures how much residue checking the walk needs: w = 0 checks only the flip's own gear
(a provable step each time), w = 'all' is entry 14's walk.
Fitness: streak from 31, streak from 11, total, then the SMALLER memory (so a walk that
succeeds with less residue checking wins ties), then fewer steps.
Machines: every prime to 200, then every third to 6000.
usage: uv run python research/stack/r8/evolve_walk9.py [generations] [population] [seed]
"""
import sys, json, random
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E
import evolve_walk6 as G

MEM = [0, 1, 2, 3, 5, 8, 13, 'all']
ORDERS = ['desc', 'asc', 'small-first']

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def settle_walk(m, L, st):
    q = m.q; P = m.B; K = st.get('K', 15); w = st.get('w', 3); order = st.get('order', 'desc'); R = st.get('R', 2); flex = st.get('flex', True)
    gears = list(m.above)
    if order == 'desc': seq = gears[::-1]
    elif order == 'asc': seq = gears
    else:
        r = m.r; seq = [g for g in gears if g <= r] + [g for g in gears if g > r][::-1]
    visited = []
    def choose(L, g, memory, nxt):
        best = None
        for k in range(1, K + 1):
            for d in (1, -1):
                n = L + 2 * k * P * g * d
                if n < 0 or n > q * q - 2: continue
                on = sum(1 for h in memory if n % h in (0, h - 2))
                md = min(dist(n % h, h) for h in memory) if memory else 0
                fl = 0
                if flex and on == 0 and nxt is not None:
                    fl = sum(1 for k2 in range(1, K + 1) for d2 in (1, -1)
                             if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in memory + [nxt]))
                sc = (-on, fl, md, -k)
                if best is None or sc > best[0]: best = (sc, n)
        return best[1] if best else None
    for i, g in enumerate(seq):
        visited.append(g)
        memory = visited if w == 'all' else visited[-(w + 1):]
        n = choose(L, g, memory, seq[i + 1] if i + 1 < len(seq) else None)
        if n is None: return None
        L = n
    small = [g for g in gears if g <= m.r] or gears[:3]
    allg = [x for x in m.ps if x >= 5]
    memory = visited if w == 'all' else visited[-(w + 1):]
    for _ in range(R):
        if all(L % h not in (0, h - 2) for h in memory): break
        best = None
        for s in small:
            for k in range(1, K + 1):
                for d in (1, -1):
                    n = L + 2 * k * P * s * d
                    if not (q < n <= q * q - 2): continue
                    on = sum(1 for h in memory if n % h in (0, h - 2))
                    sc = (-on, min(dist(n % h, h) for h in memory), -k)
                    if best is None or sc > best[0]: best = (sc, n)
        if best: L = best[1]
    return L

def apply_step(st, m, L, i):
    if st['type'] == 'settle': return settle_walk(m, L, st)
    return G.apply_step(st, m, L, i)

def run_walk(genome, m):
    L = E.origin_value(m, genome['origin'])
    if L is None: return None
    for i, st in enumerate(genome['steps']):
        L = apply_step(st, m, L, i)
        if L is None or L < -1 or L > 4 * m.q * m.q: return None
    return L

def memory_of(genome):
    ws = [st.get('w', 3) for st in genome['steps'] if st['type'] == 'settle']
    if not ws: return 0
    return 99 if 'all' in ws else max(ws)

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
    return (streak(31), streak(11), sum(oks), -memory_of(genome), -len(genome['steps']))

def rsettle(): return {'type': 'settle', 'order': random.choice(ORDERS), 'w': random.choice(MEM), 'K': random.choice([3, 5, 9, 15]), 'R': random.choice([0, 1, 2, 3]), 'flex': random.random() < 0.7}
def rstep():
    r = random.random()
    return rsettle() if r < 0.4 else G.rflip() if r < 0.8 else E.rspiral() if r < 0.92 else E.rdescent()
def rgenome(): return {'origin': E.rrule(E.ORIGINS, 3), 'steps': [rstep() for _ in range(random.randint(1, 3))]}

def mutate(g):
    g = json.loads(json.dumps(g)); r = random.random()
    if r < 0.15 and len(g['steps']) < 5: g['steps'].insert(random.randint(0, len(g['steps'])), rstep())
    elif r < 0.25 and len(g['steps']) > 1: g['steps'].pop(random.randrange(len(g['steps'])))
    elif r < 0.33: g['origin'] = E.rrule(E.ORIGINS, 3)
    else:
        st = random.choice(g['steps'])
        if st['type'] == 'settle':
            key = random.choice(['order', 'w', 'K', 'R', 'flex'])
            st[key] = {'order': random.choice(ORDERS), 'w': random.choice(MEM), 'K': random.choice([3, 5, 9, 15]), 'R': random.choice([0, 1, 2, 3]), 'flex': random.random() < 0.7}[key]
        else:
            g2 = G.mutate({'origin': g['origin'], 'steps': [st]}); g['steps'][g['steps'].index(st)] = g2['steps'][0]
    return g

def describe(g):
    parts = []
    for st in g['steps']:
        if st['type'] == 'settle': parts.append(f"settle {st.get('order')} memory {st.get('w')} K{st.get('K')} R{st.get('R')} {'flex' if st.get('flex') else 'dist'}")
        else: parts.append(E.describe({'origin': g['origin'], 'steps': [st]}).split('; ', 1)[1])
    o = g['origin']
    return f"origin {o['kind']}[{o.get('a', 0)}]; " + "; ".join(parts)

def seeds():
    S = []
    for w in MEM:
        for order in ORDERS:
            S.append({'origin': {'kind': 'home', 'a': 0}, 'steps': [{'type': 'settle', 'order': order, 'w': w, 'K': 15, 'R': 2, 'flex': True}]})
    return S

def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    popn = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    start = next(i for i, m in enumerate(machines) if m.q >= 31)
    print(f"machines {len(machines)} (11 to {qs[-1]})", flush=True)
    out_path = Path("research/stack/r8/results_evolve_walk9.json")
    pop = []
    if out_path.exists():
        try: pop = list(json.loads(out_path.read_text(encoding="utf-8")).get('elite', []))
        except Exception: pop = []
    pop += seeds()
    while len(pop) < popn: pop.append(rgenome())
    log = []
    for gen in range(gens):
        scored = sorted(((fitness(g, machines), g) for g in pop), key=lambda t: t[0], reverse=True)
        elite = [g for f, g in scored[:20]]; best_f, best_g = scored[0]
        line = f"gen {gen}: streak from 31: {best_f[0]} (to q = {machines[start + best_f[0] - 1].q if best_f[0] else '-'}); from 11: {best_f[1]}; total {best_f[2]} of {len(machines)}; memory {-best_f[3]}; {-best_f[4]} steps: {describe(best_g)}"
        if gen == 0:
            for f, g in scored[:12]: print(f"   seed rank: streak31 {f[0]}, total {f[2]}, memory {-f[3]}: {describe(g)}", flush=True)
        log.append(line); print(line, flush=True)
        out_path.write_text(json.dumps({'elite': elite, 'best_fitness': best_f, 'best': describe(best_g), 'log': log[-60:]}, indent=1), encoding="utf-8")
        pool = [g for f, g in scored[:50]]
        newpop = list(elite)
        while len(newpop) < popn:
            r = random.random()
            if r < 0.12: newpop.append(rgenome())
            elif r < 0.42: newpop.append(E.crossover(random.choice(pool), random.choice(pool)))
            else:
                g = random.choice(pool)
                for _ in range(random.randint(1, 2)): g = mutate(g)
                newpop.append(g)
        pop = newpop

if __name__ == "__main__":
    main()

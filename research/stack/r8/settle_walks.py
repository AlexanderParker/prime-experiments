"""Loop, iteration 19: the family of settle-and-grow walks (owner, 2026-09-15: the base grows
with the machine; iterate walks of this kind to find one that brings the right residues).

State: origin twin O (left member), base B (carried gears, product P), machine top q.
Step: choose the gear g to settle; flip on the mirror M (landing O + k M) with the k rule; then
grow the base and the machine.  Variants:
  settle order   'asc' the smallest unsettled gear <= q; 'desc' the largest unsettled gear <= q;
                 'striker' the smallest gear currently striking the landing (none: 'asc')
  mirror         'base' M = P; 'base+settled' M = P times the previously settled non-base gears
                 that still fit the window
  k rule         'first' the first k with the landing off g's teeth; 'keep' the first k with the
                 landing off the teeth of g AND of every previously settled gear (bounded search);
                 'one' k = 1
  base rule      'all' every settled gear joins the base if the product stays at most q^2 / 2;
                 'prim' base = the largest primorial at most q^2 / 2
  machine rule   'gear' q = the newest settled gear; 'twin' q = the larger member of the landing
Chain from (-1, 1), machine 3, base {2, 3}; a step fails when no landing in the window is found
(k up to 60) or when the landing is struck by a gear of the machine.  Reported per variant: the
number of twins landed before the first failure, and the failure.
usage: uv run python research/stack/r8/settle_walks.py
"""
import itertools
from sympy import primerange, isprime, nextprime
from pathlib import Path

def strikers(n, q):
    return [g for g in primerange(5, min(q, 10 ** 6) + 1) if n % g == 0 or (n + 2) % g == 0]

def run(order, mirror, krule, brule, mrule, maxsteps=25):
    O = -1; base = [2, 3]; P = 6; q = 3; settled = []; twins = 0; log = []
    for step in range(maxsteps):
        gears = [g for g in primerange(5, q + 1)]
        unsettled = [g for g in gears if g not in base and g not in settled]
        if not unsettled:
            g = nextprime(max(base + settled + [3]))
        elif order == 'asc': g = unsettled[0]
        elif order == 'desc': g = unsettled[-1]
        else:
            st = [h for h in strikers(O, q) if h not in base]
            g = st[0] if st else unsettled[0]
        M = P
        if mirror == 'base+settled':
            for h in settled:
                if h not in base and M * h <= q * q // 2: M *= h
        keep = [h for h in settled if h not in base] if krule == 'keep' else []
        found = None
        for k in ([1] if krule == 'one' else range(1, 61)):
            L = O + k * M
            if L > q * q - 2 and q > 3: break
            if all(L % h and (L + 2) % h for h in [g] + keep): found = L; break
        if found is None:
            log.append(f"step {step}: settling {g} on mirror {M} from {O}: no landing in the window ({q}, {q*q}]"); break
        L = found
        settled.append(g)
        newq = g if mrule == 'gear' else L + 2
        if mrule == 'gear': newq = max(newq, q)
        st = strikers(L, newq)
        tw = isprime(L) and isprime(L + 2)
        log.append(f"step {step}: settle {g} on {M} -> ({L}, {L + 2}) {'TWIN' if tw else 'struck by ' + str(st[:4])}; machine {newq}")
        if not tw: break
        twins += 1; O = L; q = newq
        if brule == 'all':
            for h in sorted(settled):
                if h not in base and P * h <= q * q // 2: P *= h; base.append(h)
        else:
            base = []; P = 1
            for p in primerange(2, q + 1):
                if P * p <= q * q // 2: P *= p; base.append(p)
                else: break
        if L > 10 ** 12: log.append("stopped: landing beyond 10^12"); break
    return twins, log

def main():
    out = [__doc__.strip(), ""]
    rows = []
    for order, mirror, krule, brule, mrule in itertools.product(('asc', 'desc', 'striker'), ('base', 'base+settled'), ('first', 'keep', 'one'), ('all', 'prim'), ('gear', 'twin')):
        twins, log = run(order, mirror, krule, brule, mrule)
        rows.append((twins, order, mirror, krule, brule, mrule, log))
    rows.sort(key=lambda r: -r[0])
    out.append(f"{'twins':>5}  order    mirror        k      base  machine   last line")
    for twins, order, mirror, krule, brule, mrule, log in rows:
        out.append(f"{twins:>5}  {order:<8} {mirror:<13} {krule:<6} {brule:<5} {mrule:<8} {log[-1][:110]}")
    out.append("")
    out.append("best chains in full:")
    for twins, order, mirror, krule, brule, mrule, log in rows[:3]:
        out.append(f"--- {order} {mirror} {krule} {brule} {mrule}: {twins} twins"); out += ["   " + l for l in log]
    Path("research/stack/r8/results_settle_walks.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

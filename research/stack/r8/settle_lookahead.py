"""Loop, entry 33: does a keeping move exist with lookahead?  At each step of the full-memory
settle walk (descending, K = 15), a move is 'd-keeping' if it keeps every visited gear off its
teeth and there is a d-keeping move at the next step (depth d), i.e. a keeping path of d steps
exists.  For depth d = 1, 2, 3: the machines (to 3947) with a step where no d-keeping move
exists.  If some depth never fails, the local lemma is a d-step statement.
usage: uv run python research/stack/r8/settle_lookahead.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def keeping_path(L, seq, i, visited, P, q, K, depth):
    """is there a path of `depth` keeping moves from column L at step i?"""
    if depth == 0 or i >= len(seq): return True
    g = seq[i]; vis = visited + [g]
    for k in range(1, K + 1):
        for d in (1, -1):
            n = L + 2 * k * P * g * d
            if n < 0 or n > q * q - 2: continue
            if all(n % h not in (0, h - 2) for h in vis):
                if keeping_path(n, seq, i + 1, vis, P, q, K, depth - 1): return True
    return False

def walk(m, depth, K=15):
    q = m.q; P = m.B; seq = list(m.above)[::-1]; visited = []; L = -1; fails = []
    for i, g in enumerate(seq):
        visited.append(g)
        chosen = None
        for k in range(1, K + 1):
            for d in (1, -1):
                n = L + 2 * k * P * g * d
                if n < 0 or n > q * q - 2: continue
                if all(n % h not in (0, h - 2) for h in visited) and keeping_path(n, seq, i + 1, visited, P, q, K, depth - 1):
                    chosen = n; break
            if chosen is not None: break
        if chosen is None:
            fails.append((i, g))
            # fall back: fewest on teeth
            best = None
            for k in range(1, K + 1):
                for d in (1, -1):
                    n = L + 2 * k * P * g * d
                    if n < 0 or n > q * q - 2: continue
                    on = sum(1 for h in visited if n % h in (0, h - 2))
                    if best is None or on < best[0]: best = (on, n)
            chosen = best[1]
        L = chosen
    return L, fails

def main():
    E.QMAX = 4000
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    out = [__doc__.strip(), ""]
    for depth in (1, 2, 3):
        bad = []; twins = 0
        for m in machines:
            L, fails = walk(m, depth)
            if fails: bad.append((m.q, len(fails)))
            if L is not None and m.q < L <= m.q * m.q - 2 and m.sv[L] and m.sv[L + 2]: twins += 1
        line = f"depth {depth}: machines with a step lacking a {depth}-keeping move: {len(bad)} of {len(machines)}: {bad[:14]}{'...' if len(bad) > 14 else ''}; landings that are twins {twins}"
        out.append(line); print(line, flush=True)
    Path("research/stack/r8/results_settle_lookahead.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

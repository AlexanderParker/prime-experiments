"""Direct measurement (the evolution was killed twice for system memory): the settle walk's
streak as a function of its residue memory w.  Machines every prime to 200 then every third to
4000.  usage: uv run python research/stack/r8/settle_memory.py
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E
import evolve_walk9 as W

def main():
    E.QMAX = 4000
    N = E.QMAX * E.QMAX * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ps = list(primerange(11, E.QMAX + 1)); qs = [p for p in ps if p <= 200] + [p for i, p in enumerate(ps) if p > 200 and i % 3 == 0]
    machines = [E.Machine(q, sv) for q in qs]
    out = [__doc__.strip(), "", f"machines {len(machines)} (11 to {qs[-1]}); settle walk descending, K = 15, two repairs, flex rule; memory w = number of previously visited gears kept off their teeth at each step"]
    for w in (0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 'all'):
        st = {'type': 'settle', 'order': 'desc', 'w': w, 'K': 15, 'R': 2, 'flex': True}
        oks = []; fails = []
        for m in machines:
            L = W.settle_walk(m, -1, st)
            ok = L is not None and m.q < L <= m.q * m.q - 2 and bool(m.sv[L] and m.sv[L + 2])
            oks.append(ok)
            if not ok: fails.append(m.q)
        streak = 0
        for m, ok in zip(machines, oks):
            if m.q < 31: continue
            if ok: streak += 1
            else: break
        line = f"   memory {str(w):>4}: streak from 31 {streak:>3} (to q = {[m.q for m in machines if m.q >= 31][streak - 1] if streak else '-'}), total {sum(oks):>3} of {len(machines)}; first failures {fails[:6]}"
        out.append(line); print(line, flush=True)
    Path("research/stack/r8/results_settle_memory.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()

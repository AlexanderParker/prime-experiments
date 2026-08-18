"""Round 12: firing law at step 31->37 (period 3.34e10) - site residues mod 37.
Lean: only the two literal words per k, positions and residues, no merges.
"""
from collections import Counter
from math import prod
import numpy as np
from split_gap_law import primes

y, qp = 31, 37
gears = primes(5, y)
P = prod(gears)
u = pow(6, -1, qp); s = (2 * u) % qp
W = {}
for k in (3, 4, 5):
    W[(k, 'start_s')] = (tuple((s if i % 2 == 0 else qp - s) for i in range(k - 1)), (qp - u) % qp)
    W[(k, 'start_sb')] = (tuple((qp - s if i % 2 == 0 else s) for i in range(k - 1)), u)
res = {key: Counter() for key in W}
tot = {key: 0 for key in W}
CH = 100_000_000
carry = None
a = 0
while a < P:
    S = min(CH, P - a)
    killed = np.zeros(S, bool)
    for q in gears:
        uq = pow(6, -1, q)
        for t in (uq, q - uq):
            killed[(t - a) % q::q] = True
    ext = np.flatnonzero(~killed).astype(np.int64) + a
    if carry is not None:
        ext = np.concatenate((carry, ext))
    d = np.diff(ext)
    for key, (w, fire) in W.items():
        n = len(w)
        if len(d) < n: continue
        m = d[:len(d) - n + 1] == w[0]
        for j in range(1, n):
            m &= d[j:len(d) - n + 1 + j] == w[j]
        idx = np.flatnonzero(m)
        if len(idx):
            r = ext[idx] % qp
            res[key].update(r.tolist())
            tot[key] += len(idx)
    carry = ext[-8:]
    a += S
    print(f"  progress {a/P:.1%}", flush=True)
print(f"STEP {y}->{qp}: period {P}, u={u}, s={s}, P mod {qp} = {P % qp}")
for key in sorted(W):
    k, tag = key
    w, fire = W[key]
    if not tot[key]: continue
    c = res[key]
    fired = c.get(fire, 0)
    occ = sorted(c.values())
    print(f"  k={k} word {w} fire-residue {fire}: sites {tot[key]}, "
          f"FIRED {fired} (expected {tot[key]/qp:.2f} at 1/q'); "
          f"residues occupied {len(c)}/{qp}, occupancy min {occ[0]} max {occ[-1]}")
    if tot[key] <= 300:
        print(f"     residue histogram: {dict(sorted(c.items()))}")

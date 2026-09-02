import numpy as np
from math import prod
from collections import Counter
ps = [5, 7, 11, 13, 17, 19, 23]
def word(gs, P):
    k = np.arange(P); w = np.ones(P, bool)
    for g in gs:
        u = pow(6, -1, g); w &= (k % g != u) & (k % g != g - u)
    return np.flatnonzero(w)
for i in range(2, len(ps) - 1):
    old, q2 = ps[:i + 1], ps[i + 1]
    Po, Pn = prod(old), prod(old) * q2
    oo = word(old, Po); on = word(old + [q2], Pn)
    go = np.diff(np.concatenate([oo, [oo[0] + Po]])); gn = np.diff(np.concatenate([on, [on[0] + Pn]]))
    Fo, Fn = int(go.max()), int(gn.max())
    par = Counter((go % 2).tolist())
    u = pow(6, -1, q2); s = sorted({(2 * u) % q2, (-2 * u) % q2})
    # new record window: its start opening and the old openings inside it (killed by q2)
    j = int(gn.argmax()); a, b = int(on[j]), int(on[j]) + Fn
    inside = [int(x) for x in np.arange(a + 1, b) if ((x % Po) in set(oo.tolist()))]   # old openings inside (all must be killed by q2)
    gaps_old = np.diff([a] + inside + [b]).tolist()
    res = [x % q2 for x in inside]
    print(f"machine {old} + {q2}: F {Fo} -> {Fn} (allowance F+q' = {Fo + q2}); old gap parity {dict(par)}; teeth +-u = {u},{q2-u}; consecutive kills differ by 0 or {s} mod {q2}")
    print(f"   new record window at k={a}..{b}: old gaps inside {gaps_old} (sum {sum(gaps_old)}), openings killed {len(inside)} at residues mod {q2}: {res}; differences of kills mod {q2}: {[(inside[t+1]-inside[t]) % q2 for t in range(len(inside)-1)]}")

"""Round 11 lateral, part 2: the k=4 event (step 29->31, word (10,21,10)) -
locate the 4 instances, verify the alternation-grammar prediction, and test
the pinning law on the resulting machine-31 gaps.

Grammar prediction (u'-free): a k-chain's interior spacing word alternates
sigma = 2u'_q and q - sigma, so for k = 4 it must be (s,q-s,s) or (q-s,s,q-s);
for q = 31, sigma = 10: (10,21,10) or (21,10,21). Sides must alternate.

Run: uv run python research/k4_pinning.py    (repo root; numpy)
"""
from math import prod
import numpy as np
from split_gap_law import primes
from topgap_corridor import chunk_openings
from topgap_nesting import local_openings
from address_drift import compatible_phases
from word_shapes import halves

q31, u31 = 31, pow(6, -1, 31)          # teeth: 26 (left-kill), 5 (right-kill)
gears29 = primes(5, 29)
P29 = prod(gears29)

hits = []
carry = None
a = 0
while a < P29:
    S = min(20_000_000, P29 - a)
    ops = chunk_openings(gears29, a, S)
    ext = ops if carry is None else np.concatenate((carry, ops))
    d = np.diff(ext)
    for pat in ((10, 21, 10), (21, 10, 21)):
        idx = np.flatnonzero((d[:-2] == pat[0]) & (d[1:-1] == pat[1]) & (d[2:] == pat[2]))
        for i in idx:
            four = ext[i:i+4]
            sides = ['L' if o % q31 == u31 else 'R' if o % q31 == (q31-u31) % q31
                     else '-' for o in four]
            if '-' not in sides:
                hits.append((int(four[0]), pat, ''.join(sides)))
    carry = ext[-6:]
    a += S
hits = sorted(set(hits))
print(f"k=4 chains of gear 31 in machine 29 (period {P29}): {len(hits)}")
for t0, pat, sides in hits:
    # machine-31 gap: from the 29-opening before t0 to the one after t0+41
    loc = local_openings(31, t0 - 60, t0 + 41 + 61)
    j = np.searchsorted(loc, t0)
    t, tend = int(loc[j-1]), int(loc[j])       # 31-machine gap boundaries
    G = tend - t
    Lw, Rw = halves(31, t, G)
    cp = compatible_phases(list(np.concatenate((
        [0], np.cumsum([-g for g in Lw]))) ) + [G + x for x in
        np.concatenate(([0], np.cumsum(Rw)))], (5, 7, 11))
    npin = len(cp[5]) * len(cp[7]) * len(cp[11])
    ok = t % 5 in cp[5] and t % 7 in cp[7] and t % 11 in cp[11]
    print(f"  chain at {t0} (word {pat}, sides {sides}): new 31-gap "
          f"[{t},{tend}] G={G}; mirror partner at {P29*31 - tend}; "
          f"pinning |phases mod 385| = {npin} (<=4: {npin<=4}), "
          f"observed in set: {ok}")
mir = {(t, t+G) for t, _, _ in [(h[0], 0, 0) for h in hits]}
print("word classes found:", sorted(set(h[1] for h in hits)),
      "| side patterns:", sorted(set(h[2] for h in hits)))

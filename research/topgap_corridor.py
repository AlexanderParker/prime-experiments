"""Round 9 lateral: corridor analysis of TOP-GAP neighbourhoods.

Frame: slot space k (pair (6k-1, 6k+1)); machine M_y = gears 5..y; openings =
slots unhit on both sides; period P = prod q. Conversions: m-space gap =
6 x slot gap; halved-coordinate gap (corpus F(2,y)) = 3 x slot gap.

Objects: F = max gap (slot units) between consecutive openings over the full
period; F2 = max ADJACENT-PAIR span (g_i + g_{i+1}); the tolerance lemma
alpha1 needs F2 - F <= alpha1 * q_next.

Corridor questions:
 1. Mirror law: openings are symmetric under k -> -k, so maximal gaps come in
    mirror pairs or are self-mirrored (centered at 0 or P/2, endpoints +-c).
 2. Addresses: where do top-gap endpoints sit mod 35 (the {5,7} lattice has
    exactly 15 open classes mod 35 - baseline uniform 1/15 by CRT)? Is the
    top of the gap spectrum corridor-forced the way the L*=13 landmark was?
 3. Flanks (isolation law) and near-top neighbourhood words (g-2,g-1,G,g+1,
    g+2): is the near-top language finite (32-cap style) or growing?
 4. F2 - F census across machines -> alpha1 evidence.

Machines: y = 13, 17, 19, 23 full period in memory; y = 29 streamed
(P = 1,078,282,205) with boundary carry.

Run: uv run python research/topgap_corridor.py    (repo root; numpy)
"""
from collections import Counter
from math import prod

import numpy as np

from split_gap_law import primes

OPEN35 = sorted(k for k in range(35)
                if k % 5 not in (1, 4) and k % 7 not in (1, 6))

def chunk_openings(gears, a, S):
    killed = np.zeros(S, bool)
    for q in gears:
        u = pow(6, -1, q)
        for t in (u, q - u):
            killed[(t - a) % q::q] = True
    return np.flatnonzero(~killed).astype(np.int64) + a

def analyze(y, chunk=20_000_000, topN=2000):
    gears = primes(5, y)
    P = prod(gears)
    qnext = next(p for p in primes(y + 1, 2 * y + 100))
    print(f"--- y = {y}: period {P}, q_next = {qnext} ---")
    carry = None          # last few openings of previous chunk
    F = 0
    F2 = 0
    F2loc = None
    records = []          # (G, leftpos, g-2, g-1, g+1, g+2)
    T = 4                 # adaptive threshold
    a = 0
    while a < P:
        S = min(chunk, P - a)
        ops = chunk_openings(gears, a, S)
        ext = ops if carry is None else np.concatenate((carry, ops))
        d = np.diff(ext)
        if len(d):
            F = max(F, int(d.max()))
            s2 = d[:-1] + d[1:]
            if len(s2):
                m = int(s2.max())
                if m > F2:
                    F2 = m
                    i = int(s2.argmax())
                    F2loc = (int(ext[i]), int(d[i]), int(d[i + 1]))
            hits = np.flatnonzero(d >= T)
            for i in hits:
                if i < 2 or i > len(d) - 3:
                    ctx = [int(d[j]) if 0 <= j < len(d) else -1
                           for j in (i - 2, i - 1, i + 1, i + 2)]
                else:
                    ctx = [int(d[i - 2]), int(d[i - 1]),
                           int(d[i + 1]), int(d[i + 2])]
                records.append((int(d[i]), int(ext[i]), *ctx))
            if len(records) > 40 * topN:
                records.sort(reverse=True)
                records = records[:topN]
                T = records[-1][0]
        carry = ext[-5:] if len(ext) >= 5 else ext
        a += S
    records.sort(reverse=True)
    records = records[:topN]
    top = [r for r in records if r[0] == F]
    # 1. mirror structure of maximal gaps: interval [l+?, ...] left open at pos,
    #    gap G: interval of killed slots (pos, pos+G); mirror maps it to
    #    (P - pos - G, P - pos)
    ivs = {(r[1], r[1] + r[0]) for r in top}
    mirrored = {((P - b) % P, (P - a2) % P) for a2, b in ivs}
    self_m = sum(1 for iv in ivs if iv in mirrored and
                 (2 * iv[0] + F) % P in (0,))
    print(f"  F = {F} slots (halved {3*F}), maximal gaps: {len(ivs)}; "
          f"mirror-closed: {ivs == mirrored}; self-mirrored (centered at 0): "
          f"{sum(1 for a2, b in ivs if (a2 + b) % P == 0)}")
    # 2. endpoint addresses mod 35
    n_ends = min(len(records), 200)
    le = Counter(r[1] % 35 for r in records[:n_ends])
    re_ = Counter((r[1] + r[0]) % 35 for r in records[:n_ends])
    tot = n_ends
    top3L = le.most_common(3)
    top3R = re_.most_common(3)
    print(f"  top-{tot} gaps, LEFT endpoint mod 35 top3: "
          f"{[(c, f'{v/tot:.2f}') for c, v in top3L]} (baseline 0.067/class)")
    print(f"  RIGHT endpoints mod 35 top3: "
          f"{[(c, f'{v/tot:.2f}') for c, v in top3R]}; "
          f"mirror check: left classes = -right classes: "
          f"{sorted(le) == sorted((-c) % 35 for c in re_)}")
    print(f"  maximal-gap endpoints: left {sorted(set(r[1] % 35 for r in top))} "
          f"right {sorted(set((r[1]+r[0]) % 35 for r in top))} mod 35")
    # 3. flanks + neighbourhood words at near-top (G >= 0.9 F)
    near = [r for r in records if r[0] >= 0.9 * F]
    fl = Counter((r[3], r[4]) for r in near)       # (g-1, g+1)
    words = {tuple(r[2:6]) for r in near} | {tuple(r[0:1]) and (r[2], r[3], r[0], r[4], r[5]) for r in near}
    words = {(r[2], r[3], r[0], r[4], r[5]) for r in near}
    print(f"  near-top (G >= 0.9F): {len(near)} gaps, flank pairs (g-1,g+1) "
          f"top5: {fl.most_common(5)}")
    print(f"  distinct 5-gap neighbourhood words at near-top: {len(words)}")
    # 4. F2
    print(f"  F2 = {F2} at leftpos {F2loc[0]} (components {F2loc[1]}+{F2loc[2]}), "
          f"F2 - F = {F2 - F} slots = {3*(F2-F)} halved; "
          f"(F2-F)/q_next = {(F2 - F)/qnext:.3f} slot-units "
          f"({3*(F2-F)/qnext:.3f} halved)")
    return dict(y=y, F=F, F2=F2, records=records, near=near, P=P)

if __name__ == "__main__":
    print("=" * 72)
    print("TOP-GAP CORRIDOR ANALYSIS (slot frame; halved = 3 x slot)")
    print(f"open classes mod 35 ({len(OPEN35)}): {OPEN35}")
    res = {}
    for y in (13, 17, 19, 23, 29):
        res[y] = analyze(y)
    print("=" * 72)
    print("alpha1 evidence table: y, F, F2, F2-F (slot), (F2-F)*3/q_next:")
    for y, r in res.items():
        qn = next(p for p in primes(y + 1, 2 * y + 100))
        print(f"  y={y:>3}: F {r['F']:>3}  F2 {r['F2']:>3}  "
              f"dF {r['F2']-r['F']:>3}  alpha1_halved {(3*(r['F2']-r['F']))/qn:.3f}")
    print("near-top language size trend (finite vs growing):")
    for y, r in res.items():
        w = {(a, b, g, c, d) for g, p, a, b, c, d in r['near']}
        print(f"  y={y:>3}: near-top gaps {len(r['near']):>5}, distinct words {len(w):>5}")

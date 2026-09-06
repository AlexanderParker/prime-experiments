"""pc_phase.py -- POSITIONS on the big machines by the copy law (docs/proofs/05 (A)):
the period of M + q' is q' copies of M's period, and column X = x + j P is struck by q' iff
X = +-u (mod q').  So machine m29 is 29 copies of m23's period, m31 is 29 * 31 = 899 copies and
m37 is 29 * 31 * 37 = 33,263 copies; each copy is one pass over m23's 7,952,175 openings, and a
realised legal word of the big machine is found with its global column X0.

Per copy j the buffer is [tail of copy j-1] + [copy j]; words are recorded iff they START at a
global column in [j P - D, (j+1) P - D) with D = 3000 > any word span, so every word of the
period is recorded exactly once (copy 0's tail is copy n-1 shifted below 0).

Records, per big machine and with respect to its next gear: every maximal realised legal word
with its occurrence count, the exact counts W_1, W_2, W_3 of positions whose first k gaps form a
legal word (gate: pc_ladder.py's multiplicities), and the global positions of the padded words
of maximal padded length, with the junction analysis of pc_core (the gears' teeth are arithmetic
on X; no period is needed).

Usage: uv run python research/anchor235/r67/pc_phase.py 29          (29 copies)
       uv run python research/anchor235/r67/pc_phase.py 31          (899 copies)
       uv run python research/anchor235/r67/pc_phase.py 37 [nproc]  (33,263 copies, background)
"""
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_core import (OUT, gears_of, junction_analysis, letters_of, maximal_legal_words, next_gear,
                     sieve_machine, teeth, word_kind)

D = 3000
TAIL = 2500
BASE = 23
_G = {}


def _init():
    if "O" in _G:
        return
    gears = gears_of(BASE)
    P, O = sieve_machine(gears)
    _G["P"] = P
    _G["O"] = O


def copy_survivors(j, new_gears, ncopies):
    """Openings of the big machine inside copy j (any integer j; the period wraps).  The
    residues O mod g are precomputed once; copy j shifts them by j P mod g."""
    _init()
    P, O = _G["P"], _G["O"]
    jj = j % ncopies
    keep = np.ones(O.size, dtype=bool)
    for g in new_gears:
        if ("R", g) not in _G:
            _G[("R", g)] = (O % g).astype(np.int16)
        R = _G[("R", g)]
        t1, t2 = teeth(g)
        sh = (jj * P) % g
        r = R + np.int16(sh)
        r[r >= g] -= g
        keep &= (r != t1) & (r != t2)
    return O[keep] + np.int64(j) * np.int64(P)


def legal_pair_ok(lt1, lt2):
    return (lt1 != 3) & (lt2 != 3) & ~((lt2 != 0) & (lt2 == lt1))


def scan_range(args):
    j0, j1, new_gears, ncopies, q_next, keep_len = args
    _init()
    P = _G["P"]
    cnt = Counter()
    W = np.zeros(4, dtype=np.int64)
    pad_pos = []
    prev = copy_survivors(j0 - 1, new_gears, ncopies)[-TAIL:]
    for j in range(j0, j1):
        S = np.concatenate([prev, copy_survivors(j, new_gears, ncopies)])
        gaps = np.diff(S)
        lo = np.int64(j) * np.int64(P) - D
        hi = np.int64(j + 1) * np.int64(P) - D
        starts = S[:-1]
        valid = (starts >= lo) & (starts < hi)
        vidx = np.flatnonzero(valid)
        first_valid, n_valid = int(vidx[0]), int(vidx[-1]) + 1
        # exact W_k gates: positions whose first k gaps form a legal word
        lt = letters_of(gaps, q_next)
        ok1 = (lt != 3)
        W[1] += int(ok1[valid].sum())
        if gaps.size > 1:
            ok2 = np.zeros(gaps.size, dtype=bool)
            ok2[:-1] = ok1[:-1] & legal_pair_ok(lt[:-1], lt[1:])
            W[2] += int(ok2[valid].sum())
            ok3 = np.zeros(gaps.size, dtype=bool)
            # last nonzero class before position i+2 within the word (i, i+1, i+2)
            lastnz = np.where(lt[1:-1] == 0, lt[:-2], lt[1:-1])
            ok3[:-2] = ok2[:-2] & (lt[2:] != 3) & ~((lt[2:] != 0) & (lt[2:] == lastnz))
            W[3] += int(ok3[valid].sum())
        words = maximal_legal_words(gaps, q_next, n_valid)
        for i, w in words:
            if i < first_valid:
                continue
            cnt[w] += 1
            if word_kind(w, q_next) != "bare":
                pad_pos.append((int(S[i]), w))
        prev = S[-TAIL:]
    if pad_pos:
        Lp = max(len(w) for _, w in pad_pos)
        pad_pos = [(x, w) for x, w in pad_pos if len(w) == Lp][:keep_len]
    return cnt, pad_pos, W


def main():
    q = int(sys.argv[1])
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    new_gears = [g for g in (29, 31, 37) if g <= q]
    ncopies = 1
    for g in new_gears:
        ncopies *= g
    q_next = next_gear(q)
    gears = gears_of(q)
    t0 = time.time()
    _init()
    # optional partial scan of copies [j0, j1): positions only, counts partial
    j0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    j1 = int(sys.argv[4]) if len(sys.argv) > 4 else ncopies
    partial = (j0, j1) != (0, ncopies)
    nchunks = nproc * 8 if nproc > 1 else 1
    edges = np.linspace(j0, j1, nchunks + 1).astype(int)
    tasks = [(int(edges[i]), int(edges[i + 1]), new_gears, ncopies, q_next, 400)
             for i in range(nchunks) if edges[i + 1] > edges[i]]
    cnt = Counter()
    pad_pos = []
    W = np.zeros(4, dtype=np.int64)
    if nproc > 1:
        with Pool(nproc) as pool:
            for c, pp, w in pool.imap_unordered(scan_range, tasks):
                cnt.update(c)
                pad_pos.extend(pp)
                W += w
                print(f"  chunk done, words so far {sum(cnt.values()):,} W={W[1:].tolist()} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    else:
        for t in tasks:
            c, pp, w = scan_range(t)
            cnt.update(c)
            pad_pos.extend(pp)
            W += w
    L = max((len(w) for w in cnt), default=0)
    Lp = max((len(w) for _, w in pad_pos), default=0)
    pad_pos = [(x, w) for x, w in pad_pos if len(w) == Lp]
    pad_pos.sort()
    kinds = {w: word_kind(w, q_next) for w in cnt}
    junc = []
    seen = Counter()
    for x, w in pad_pos:
        if seen[w] >= 3:
            continue
        seen[w] += 1
        junc.append(junction_analysis(x, w, gears))
    out = {"machine": q, "q_next": q_next, "copies": ncopies, "L": L, "L_pad": Lp,
           "W": W[1:].tolist(), "partial": [j0, j1] if partial else None,
           "words": {" ".join(map(str, w)): {"count": c, "kind": kinds[w]} for w, c in cnt.items()},
           "n_longest_pad_positions": len(pad_pos),
           "longest_pad_positions": [(x, list(w)) for x, w in pad_pos[:400]],
           "junctions": junc, "secs": round(time.time() - t0, 1)}
    tag = f"_{j0}_{j1}" if partial else ""
    with open(os.path.join(OUT, f"phase_m{q}{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=int)
    print(f"m{q} (w.r.t. {q_next}): copies {j0}..{j1} of {ncopies}, L={L}, L_pad={Lp}, W_1..3={W[1:].tolist()}, "
          f"{len(pad_pos)} positions of the longest padded words [{time.time()-t0:.1f}s]")
    for w in sorted(cnt, key=lambda w: (-len(w), w)):
        if len(w) >= 2 or kinds[w] != "bare":
            print(f"   {w} x{cnt[w]} {kinds[w]}")
    for jn in junc[:6]:
        print(f"   pad word {jn['word']} at X0={jn['X0']}: needed={[gp['needed'] for gp in jn['gaps']]} "
              f"shared={jn['shared_needed']} counting={jn['counting']}")
        print(f"     openings: {[(o['X'] % q_next, o['left'], o['right'], o['mod35']) for o in jn['openings']]}")


if __name__ == "__main__":
    main()

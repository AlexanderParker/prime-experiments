"""Round 13 correction: F(M+q') with PADDED links, not just literal words.

Gear q' kills openings at residues {u, -u} mod q'. A run of consecutive old
openings is jointly killed (for the phase fixed by its first element) iff every
consecutive spacing is = 0, +2u or -2u (mod q') - spacing = 0 means the same
tooth twice (a PADDED link, costing >= q'), +-2u means switching teeth (the
literal link, costing s or q'-s, or those plus multiples of q').

merge_decompose.py matched only the LITERAL values s / q'-s, so it computed a
lower bound. Correct algorithm: maximal runs under the mod-q' condition.

    F(M+q') = max over maximal killed runs of ( o[i+k] - o[i-1] )
"""
import sys
from math import prod

import numpy as np

from split_gap_law import primes

KNOWN_F = {17: 18, 19: 25, 23: 34, 29: 43, 31: 58}

def step(y, qp, chunk=100_000_000):
    gears = primes(5, y)
    P = prod(gears)
    u = pow(6, -1, qp)
    ok_res = {0, (2 * u) % qp, (-2 * u) % qp}
    best = 0
    bestinfo = None
    F2 = 0
    Fold = 0
    tail = None                      # (openings, linkflags) carried over
    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            uq = pow(6, -1, q)
            for t in (uq, q - uq):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if tail is not None:
            o = np.concatenate((tail, o))
        d = np.diff(o)
        if len(d):
            Fold = max(Fold, int(d.max()))
        if len(o) > 2:
            F2 = max(F2, int((o[2:] - o[:-2]).max()))
        link = np.isin(d % qp, list(ok_res))          # link[j]: o[j]~o[j+1]
        # maximal runs of consecutive True in link -> killed run of len r+1
        idx = np.flatnonzero(link)
        if len(idx):
            brk = np.flatnonzero(np.diff(idx) != 1)
            starts = np.concatenate(([idx[0]], idx[brk + 1]))
            ends = np.concatenate((idx[brk], [idx[-1]]))   # inclusive
            for st, en in zip(starts, ends):
                i = st                      # run = o[i .. en+1]
                k = en + 1 - i + 1
                if i - 1 < 0 or en + 2 >= len(o):
                    continue
                merged = int(o[en + 2] - o[i - 1])
                if merged > best:
                    best = merged
                    bestinfo = (int(o[i]), k, tuple(int(x) for x in d[i:en + 1]),
                                int(o[i] - o[i - 1]), int(o[en + 2] - o[en + 1]))
        tail = o[-200:]
        a += S
    Fnew = max(F2, best)
    chk = KNOWN_F.get(qp)
    v = "" if chk is None else f"  [known {chk}: {'OK' if chk == Fnew else 'MISMATCH'}]"
    print(f"STEP {y}->{qp} (u={u}, links = {sorted(ok_res)} mod {qp}): "
          f"F_old {Fold}, F2 {F2}, F_new {Fnew}{v}")
    if bestinfo and best >= F2:
        pos, k, word, fl, fr = bestinfo
        pad = [w for w in word if w % qp == 0 or w > qp]
        print(f"  winner: run of {k} kills at {pos}, spacings {word} "
              f"(span {sum(word)}), flanks {fl}+{fr}; padded links: "
              f"{pad if pad else 'none (literal)'}")
    print(f"  excess = {Fnew - F2} ({(Fnew-F2)/qp:+.3f} q')")
    return Fnew

if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    for y, qp in [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]:
        if y <= lim:
            step(y, qp)

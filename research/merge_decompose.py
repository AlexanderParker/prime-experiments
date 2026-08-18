"""Round 13 lateral: F(M+q') and its excess, computed from the OLD machine alone.

Consequence of the corrected firing law (round 12): over the FULL new period
every fuel site fires exactly once, so residues drop out entirely and

    F(M+q') = max over k>=1, over all k-sites, of  ( o[i+k] - o[i-1] )

where a k-site is k consecutive old openings whose spacing word is one of the
two alternating literal words of q' (k=1: any opening, no constraint), and
o[i-1], o[i+k] are the bracketing old openings. This is the Constructor's word
identity made computational: no new-period scan, no residue bookkeeping.
k=1 reproduces F2(M) exactly; k>=2 terms are the excess candidates.

Reports per step: F_new (checked against the known value), which k and which
word wins, its flank sums, and FS_max + occurrence count per word - the inputs
to the excess-growth question.
"""
import sys
from math import prod

import numpy as np

from split_gap_law import primes

KNOWN_F = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}

def openings(gears, a, S):
    killed = np.zeros(S, bool)
    for q in gears:
        u = pow(6, -1, q)
        for t in (u, q - u):
            killed[(t - a) % q::q] = True
    return np.flatnonzero(~killed).astype(np.int64) + a

def step(y, qp, chunk=100_000_000, kmax=6):
    gears = primes(5, y)
    P = prod(gears)
    u = pow(6, -1, qp)
    s = (2 * u) % qp
    words = {}
    for k in range(2, kmax + 1):
        words[(k, 'a')] = tuple((s if i % 2 == 0 else qp - s) for i in range(k - 1))
        words[(k, 'b')] = tuple((qp - s if i % 2 == 0 else s) for i in range(k - 1))
    best = {}                      # key -> (merged, pos, FL, FR)
    cnt = {key: 0 for key in words}
    F2 = 0
    Fold = 0
    carry = None
    a = 0
    while a < P:
        S = min(chunk, P - a)
        o = openings(gears, a, S)
        if carry is not None:
            o = np.concatenate((carry, o))
        d = np.diff(o)
        if len(d):
            Fold = max(Fold, int(d.max()))
        if len(o) > 2:             # k = 1: merged = o[i+1] - o[i-1]
            F2 = max(F2, int((o[2:] - o[:-2]).max()))
        for key, w in words.items():
            k = key[0]
            n = k - 1
            if len(d) < n + 2:
                continue
            m = d[1:len(d) - n] == w[0]         # site starts at index i>=1
            for j in range(1, n):
                m &= d[1 + j:len(d) - n + j] == w[j]
            idx = np.flatnonzero(m) + 1         # i
            if not len(idx):
                continue
            cnt[key] += len(idx)
            merged = o[idx + k] - o[idx - 1]
            b = int(merged.argmax())
            cand = (int(merged[b]), int(o[idx[b]]),
                    int(o[idx[b]] - o[idx[b] - 1]),
                    int(o[idx[b] + k] - o[idx[b] + k - 1]))
            if key not in best or cand[0] > best[key][0]:
                best[key] = cand
        carry = o[-(kmax + 2):]
        a += S
    Fnew = max([F2] + [v[0] for v in best.values()])
    tag = "F2 (k=1)"
    for key, v in best.items():
        if v[0] == Fnew:
            tag = f"k={key[0]} word {words[key]}"
    chk = KNOWN_F.get(qp)
    print(f"STEP {y}->{qp} (u={u}, s={s}): F_old {Fold}, F2 {F2}, "
          f"F_new {Fnew}" + (f"  [known {chk}: "
          f"{'OK' if chk == Fnew else 'MISMATCH'}]" if chk else ""))
    print(f"  winner: {tag}; excess = F_new - F2 = {Fnew - F2} "
          f"({(Fnew-F2)/qp:+.3f} q')")
    for key in sorted(best, key=lambda z: (z[0], z[1])):
        k = key[0]
        merged, pos, FL, FR = best[key]
        print(f"    k={k} word {words[key]}: occurrences {cnt[key]:>7}, "
              f"span {sum(words[key]):>3}, best merged {merged:>3} "
              f"(flanks {FL}+{FR} = FS {FL+FR}) at {pos}")
    return dict(y=y, qp=qp, Fold=Fold, F2=F2, Fnew=Fnew,
                cnt=dict(cnt), best=dict(best), words=words)

if __name__ == "__main__":
    ys = [(13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    for y, qp in ys:
        if y <= lim:
            step(y, qp)

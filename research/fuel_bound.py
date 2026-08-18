"""Constructor round 11: the fuel bound - what limits chain length.

THEOREM (one line, used throughout): every qualifying interior gap of a chain
is = 0 or +-2c mod q' and positive, hence >= 2u' (the smallest positive value
of +-3^{-1} mod q', 2u' = (q'-+1)/3). Therefore a k-chain's k-1 interior gaps
are CONSECUTIVE gaps all >= 2u', and

    k_max(M, q') <= T(M, 2u') + 1,

T(M, t) = the longest run of consecutive gaps all >= t in M's gap word.
Fuel is capped by TAIL-RUN length - rigorously, residues not needed.

SECOND THEOREM (residues dropped, upper bound): the merged value of any
k-chain is a sum of k+1 consecutive gaps whose k-1 middle gaps are all
>= 2u'. Define the QUALIFYING SPECTRUM
    Q_{k+1}(M; t) = max sum of k+1 consecutive gaps with middle k-1 all >= t.
Then increment(M -> q') <= max_k Q_{k+1}(M; 2u') - F(M)  (Q_2 = F2: lemma 1).
The whole tolerance hypothesis reduces to Q-FLATNESS at the realized depths -
fuel folds into flatness.

Computed per consecutive step (machines 11..23):
  T(M, 2u'), the Q-spectrum to depth T+2, Q - F vs the 2.5q' budget, and the
  corridor-walk caps for LITERAL alternating chains at modulus 35/385
  (part 1 of the mandate: exposure counting at bounded modulus).
"""
import numpy as np
from math import prod

STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29)]
GEARS = {y: [g for g in [5, 7, 11, 13, 17, 19, 23] if g <= y]
         for y in (11, 13, 17, 19, 23)}


def exposed(gears, m):
    a = np.ones(m, bool)
    for q in gears:
        c = pow(6, -1, q)
        a[c::q] = False
        a[(q - c) % q::q] = False
    return a


def gapword(y):
    P = prod(GEARS[y])
    idx = np.flatnonzero(exposed(GEARS[y], P))
    return np.diff(np.append(idx, idx[0] + P)).astype(np.int64)


def tail_runs(gaps, t):
    """max run length of consecutive gaps >= t (cyclic, run < len)."""
    g2 = np.concatenate([gaps, gaps])
    big = g2 >= t
    best = cur = 0
    for b in big:
        cur = cur + 1 if b else 0
        best = max(best, cur)
        if best >= len(gaps):
            break
    return min(best, len(gaps))


def q_spectrum(gaps, t, depth):
    """Q_{k+1} for k = 1..depth: max sum of k+1 consecutive with middle k-1
    all >= t. Vectorised via runs of big gaps."""
    n = len(gaps)
    g2 = np.concatenate([gaps, gaps])
    out = {}
    csum = np.concatenate([[0], np.cumsum(g2)])
    big = g2 >= t
    for k in range(1, depth + 1):
        if k == 1:
            s = g2[:n] + g2[1:n + 1]
            out[2] = int(s.max())
            continue
        # middle window big-run test: positions i..i+k (k+1 gaps), middles
        # i+1..i+k-1 all big
        ok = np.ones(n, bool)
        for j in range(1, k):
            ok &= big[j:j + n]
        if not ok.any():
            out[k + 1] = None
            continue
        sums = csum[np.arange(n) + k + 1] - csum[np.arange(n)]
        out[k + 1] = int(sums[ok].max())
    return out


def walk_cap(q1, m, gears):
    """Max literal side-alternating chain length inside E mod m: positions
    r + j*q1 and r + j*q1 + s1 all exposed; cap = max total members."""
    E = exposed(gears, m)
    u = round(q1 / 6)
    s1 = (2 * u) % m
    best = 0
    for r in range(m):
        # longest run over j (cyclic in j up to m) of both-exposed
        run = mx = 0
        for j in range(2 * m):
            a = (r + j * q1) % m
            if E[a] and E[(a + s1) % m]:
                run += 1
                mx = max(mx, run)
            else:
                run = 0
        best = max(best, mx)
    return 2 * best  # each good j contributes 2 chain members (L and R)


if __name__ == "__main__":
    print("literal-chain corridor caps (exposure counting, part 1):")
    print("  q'    cap mod 35   cap mod 385")
    for q1 in (13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        c35 = walk_cap(q1, 35, [5, 7])
        c385 = walk_cap(q1, 385, [5, 7, 11])
        print(f"  {q1:>3}   {c35:>5}        {c385:>5}")

    print("\nper-step fuel + Q-spectrum (t = 2u', budget = 2.5q'/3 k-frame):")
    for y, q1 in STEPS:
        gaps = gapword(y)
        t = 2 * round(q1 / 6)
        T = tail_runs(gaps, t)
        F = int(gaps.max())
        Q = q_spectrum(gaps, t, min(T + 2, 8))
        budget = 2.5 * q1 / 3
        qstr = "  ".join(f"Q_{j}={v if v is not None else '-'}"
                         f"({'' if v is None else v - F:+d})".replace("(+", "(+")
                         for j, v in Q.items())
        worst = max((v - F) for v in Q.values() if v is not None)
        print(f"  step {y}->{q1}: t={t}  T={T}  k_max <= {T+1}  F={F}")
        print(f"    {qstr}")
        print(f"    max Q - F = {worst} vs budget {budget:.1f} k-frame "
              f"({'WITHIN' if worst <= budget else 'EXCEEDS'}); "
              f"realized incr = {[11,18,25,34,43][STEPS.index((y,q1))]-F} ")

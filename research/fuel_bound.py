"""Constructor round 11: the fuel bound - what limits chain length.

THEOREM 1 (one line): every qualifying interior gap of a chain is = 0 or
+-2c mod q' and positive, hence >= 2u' (smallest positive value of
+-3^{-1} mod q'). So a k-chain's k-1 interior gaps are CONSECUTIVE gaps all
>= 2u', and k_max(M, q') <= T(M, 2u') + 1, T = longest run of consecutive
gaps >= t. Fuel is capped by tail-run length, rigorously.

THEOREM 2 (residues dropped): merged value of any k-chain <= Q_{k+1}(M; 2u'),
the QUALIFYING SPECTRUM = max sum of k+1 consecutive gaps whose k-1 middle
gaps are all >= 2u'. increment <= max_k Q_{k+1} - F. (Q_2 = F2: lemma 1.)

THEOREM 3 (corridor cap on LITERAL chains): a literal chain (spacings exactly
alternating {2u', q'-2u'}) has member positions r, r+2u', r+q', r+q'+2u', ...
all exposed; mod 35 this is an interleaved walk with period 70; the maximal
run is computable per (q' mod 35, 2u' mod 35) - i.e. per q' mod 210. Padded
links (any other qualifying spacing) cost a gap >= q'.

Computed: literal caps for all primes q' < 1000 (max over the table = the
absolute literal-fuel cap); per-step T, Q-spectrum, budget check (machines
11..23 full, machine 29 in one 231M-gap array); the k=5-at-31 verdict.
"""
import numpy as np
from math import prod

STEPS = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31)]
ALL = [5, 7, 11, 13, 17, 19, 23, 29]
GEARS = {y: [g for g in ALL if g <= y] for y in (11, 13, 17, 19, 23, 29)}
FNEW = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}


def exposed(gears, m):
    a = np.ones(m, bool)
    for q in gears:
        c = pow(6, -1, q)
        a[c::q] = False
        a[(q - c) % q::q] = False
    return a


E35 = exposed([5, 7], 35)


def literal_cap(q1):
    """Exact max member count of a literal alternating chain mod 35."""
    s1 = (2 * round(q1 / 6)) % 35
    best = 0
    for r in range(35):
        for phase in (0, 1):          # start on L or R tooth
            run = mx = 0
            for i in range(140):      # two full periods of the walk
                j, par = divmod(i + phase, 2)
                pos = (r + j * q1 + (s1 if par else 0)) % 35
                if E35[pos]:
                    run += 1
                    mx = max(mx, run)
                else:
                    run = 0
            best = max(best, mx)
    return best


def gapword(y):
    if y <= 23:
        P = prod(GEARS[y])
        idx = np.flatnonzero(exposed(GEARS[y], P))
        return np.diff(np.append(idx, idx[0] + P)).astype(np.int8)
    # y = 29: chunked
    P = prod(GEARS[y])
    chunks = []
    prev = None
    for lo in range(0, P, 50_000_000):
        hi = min(lo + 50_000_000, P)
        arr = np.ones(hi - lo, bool)
        for q in GEARS[y]:
            c = pow(6, -1, q)
            for a0 in (c, (q - c) % q):
                arr[(a0 - lo) % q::q] = False
        pos = np.flatnonzero(arr).astype(np.int64) + lo
        if len(pos) == 0:
            continue
        if prev is not None:
            chunks.append(np.diff(np.concatenate([[prev], pos])).astype(np.int8))
        else:
            first = pos[0]
            chunks.append(np.diff(pos).astype(np.int8))
        prev = pos[-1]
    chunks.append(np.array([first + P - prev], dtype=np.int8))
    return np.concatenate(chunks)


def tail_and_Q(gaps, t, depth=6):
    g = gaps.astype(np.int32)
    big = g >= t
    # T: longest run of big (linear scan via run-length encoding)
    d = np.diff(np.flatnonzero(np.diff(np.concatenate(
        [[False], big, [False]]).astype(np.int8)) != 0).reshape(-1, 2),
        axis=1)
    T = int(d.max()) if len(d) else 0
    Q = {2: int((g[:-1] + g[1:]).max())}
    n = len(g)
    for k in range(2, depth + 1):     # Q_{k+1}: middles k-1 all big
        ok = big[1:n - k + 1].copy()
        for j in range(2, k):
            ok &= big[j:n - k + j]
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            Q[k + 1] = None
            continue
        s = np.zeros(len(idx), np.int64)
        for j in range(k + 1):
            s += g[idx + j]
        Q[k + 1] = int(s.max())
    return T, Q


if __name__ == "__main__":
    print("literal corridor caps (exact, member count), all primes q' < 1000:")
    caps = {}
    for q1 in [p for p in range(11, 1000) if all(p % d for d in range(2, p))]:
        caps.setdefault(literal_cap(q1), []).append(q1)
    for c in sorted(caps):
        v = caps[c]
        print(f"  cap {c}: {len(v)} primes  (first: {v[:8]})")
    print(f"  ABSOLUTE LITERAL CAP over all q' < 1000: {max(caps)}")

    print("\nk=5 at q'=31: literal word (10,21,10,21) needs 5 walk members; "
          f"cap(31) = {literal_cap(31)} -> "
          f"{'FORBIDDEN mod 35' if literal_cap(31) < 5 else 'allowed'}")

    print("\nper-step fuel + Q-spectrum (t = 2u'; budget k-frame = 2.5q'/3):")
    for y, q1 in STEPS:
        gaps = gapword(y)
        t = 2 * round(q1 / 6)
        T, Q = tail_and_Q(gaps, t)
        F = int(gaps.max())
        parts = []
        worst = 0
        for j, v in Q.items():
            if v is None:
                parts.append(f"Q_{j}=-")
            else:
                parts.append(f"Q_{j}={v}({v - F:+d})")
                worst = max(worst, v - F)
        print(f"  {y}->{q1}: t={t} T={T} k_max<= {T+1} litcap={literal_cap(q1)}"
              f" F={F} incr={FNEW[q1]-F}")
        print(f"    {'  '.join(parts)}  maxQ-F={worst} "
              f"budget={2.5*q1/3:.1f} "
              f"[{'WITHIN' if worst <= 2.5*q1/3 else 'EXCEEDS'}]")

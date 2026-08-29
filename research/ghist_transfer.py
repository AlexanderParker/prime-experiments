"""Round 26 (mechanic): FULL-PERIOD GAP HISTOGRAM BY LAP-PHASE TRANSFER,
CYCLICALLY CLOSED - the first histogram at a machine no scan reaches.

WHY.  Lateral's U6/U9 need one full-period gap histogram beyond machine 31.
The round-20 machine-37 scan cost 11,829 s and threw its array away (rule 26),
and machine 41's period is 5.07e13 slots - 41x machine 37 - so a direct sieve
is ~350,000 core-seconds.  This tool does it from a SMALL machine's period.

THE CONSTRUCTION (the lap-phase transfer of K2, used for counting instead of
maximising).  Let machine OLD have period P and opening set O, and add gears
q_1..q_r, T = prod q_i, so the new period is P' = T * P.  Lap j is the slot
range [jP, (j+1)P).  For x in O, the slot x + jP survives gear q_i iff

    x + jP  !=  +-u_i  (mod q_i),   u_i = 6^{-1} mod q_i,

i.e. iff x is not in the two teeth {c_i - u_i, c_i + u_i} of PHASE
c_i = -jP (mod q_i).  P is invertible mod every q_i and the q_i are distinct,
so by CRT the map j -> (c_1(j), ..., c_r(j)) is a BIJECTION from laps onto all
T phase tuples.  Hence

    the new machine's openings, lap by lap, are exactly the phase-filtered
    copies S_c = {x in O : x not in either tooth of c_i, for every i},

and the new machine's gap multiset over its whole period is

    (union over all T phase tuples of the INTERNAL gaps of S_c)
  + (the T LAP-BOUNDARY gaps, in lap order j -> j+1 mod T).

The boundary gaps are what a linear close drops (rule 25, C26).  Here they are
computed exactly: gap(j -> j+1) = P - last(S_{c(j)}) + first(S_{c(j+1)}), and
the last one (j = T-1 -> 0) is the period's wrap gap.  Total gaps come out to
prod(q-2) and sum(g * count) to P', both ASSERTED.

COST.  Work is proportional to T * |O| ~ (new machine's opening count), done in
numpy over arrays of |O| entries, with the phase loops NESTED so the outer
gears' filtering is amortised.  Machine 37 from machine 23: T = 33,263,
~1,000 core-seconds.  Machine 41 from machine 23: T = 1,364,183, ~44,000
core-seconds, split over workers by the OUTERMOST gear's phase.

usage:
  worker : python research/ghist_transfer.py OLD q1,q2,... OUT.npz [C0 C1]
           (C0,C1 = half-open range of the FIRST gear's phase; default all)
  merge  : python research/ghist_transfer.py OLD q1,q2,... merge OUT.csv f1.npz ...
"""
import sys
import time
from math import prod

import numpy as np

FMAX = 512          # histogram length; every gap is asserted to fit
F_KNOWN = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145}


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def openings(y):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    # ROUND-26: int32 openings and int8 residues.  P(23) = 37,182,145 < 2^31
    # and every residue is < 53, so this holds a worker to ~250 MB instead of
    # ~460 MB - and MEMORY, not cores, is what limits the worker count here.
    return np.flatnonzero(~ex).astype(np.int32), P


OLD = int(sys.argv[1])
NEW = [int(x) for x in sys.argv[2].split(',')]
TARGET = NEW[-1]
T = prod(NEW)
U = [pow(6, -1, q) for q in NEW]
GEARS_ALL = [p for p in primes_upto(TARGET) if p >= 5]
NOPEN_NEW = prod(q - 2 for q in GEARS_ALL)


def flat_index(cs):
    idx = 0
    for c, q in zip(cs, NEW):
        idx = idx * q + c
    return idx


# ---------------------------------------------------------------- merge
if sys.argv[3] == 'merge':
    out = sys.argv[4]
    op, P = openings(OLD)
    PNEW = P * T
    hist = np.zeros(FMAX, np.int64)
    first = np.full(T, -1, np.int64)
    last = np.full(T, -1, np.int64)
    for f in sys.argv[5:]:
        d = np.load(f)
        hist += d['hist']
        c0, c1 = int(d['c0']), int(d['c1'])
        blk = T // NEW[0]
        sl = slice(c0 * blk, c1 * blk)
        assert (first[sl] == -1).all(), f"overlapping worker ranges in {f}"
        first[sl] = d['first']
        last[sl] = d['last']
    assert (first >= 0).all(), "worker ranges do not tile the phase space"
    # LAP-BOUNDARY GAPS, in lap order.  c_i(j) = -j*P mod q_i; CRT makes
    # j -> (c_i(j)) a bijection onto all T tuples, which is asserted here.
    lapidx = np.empty(T, np.int64)
    for j in range(T):
        lapidx[j] = flat_index([(-j * P) % q for q in NEW])
    assert len(np.unique(lapidx)) == T, "lap -> phase tuple is not a bijection"
    b = P - last[lapidx] + first[np.roll(lapidx, -1)]
    assert b.min() >= 1 and b.max() < FMAX, (b.min(), b.max())
    hist += np.bincount(b, minlength=FMAX)
    tot = int(hist.sum())
    wsum = int((hist * np.arange(FMAX)).sum())
    F = int(np.flatnonzero(hist)[-1])
    print(f"machine {TARGET} = {OLD} + {NEW}: period {PNEW:,}, laps {T:,}")
    assert tot == NOPEN_NEW, (tot, NOPEN_NEW, "gap total != prod(q-2)")
    assert wsum == PNEW, (wsum, PNEW, "sum(g*count) != period")
    print(f"  ASSERT gaps          = {tot:,}  = prod(q-2)   OK")
    print(f"  ASSERT sum(g*count)  = {wsum:,}  = period      OK")
    if TARGET in F_KNOWN:
        assert F == F_KNOWN[TARGET], (F, F_KNOWN[TARGET])
        print(f"  ASSERT max gap       = {F}  = F({TARGET})       OK")
    # the wrap gap is the LAST lap-boundary gap; the closed form (C26) says it
    # equals the first gap of the period.
    wrap = int(b[-1])
    firstgap = int(P - last[lapidx[T - 1]] + first[lapidx[0]])
    s0 = first[lapidx[0]]
    assert s0 == 0, ("slot 0 must be an opening of every machine", s0)
    print(f"  wrap gap = {wrap}   (closed form C26: wrap = first gap)")
    holes = [v for v in range(1, F) if hist[v] == 0]
    print(f"  holes below F: {holes}")
    print(f"  distinct gap values: {int((hist > 0).sum())}")
    with open(out, 'w') as fh:
        fh.write("y,gap,count\n")
        for v in range(1, F + 1):
            if hist[v]:
                fh.write(f"{TARGET},{v},{int(hist[v])}\n")
    print(f"  wrote {out}")
    for qp in (41, 43, 47, 53, 59, 61):
        if qp <= F:
            print(f"  padding supply hist[{qp}] = {int(hist[qp]):,}")
    np.savez(out + ".npz", hist=hist)
    sys.exit()

# ---------------------------------------------------------------- worker
DELTA = '--delta' in sys.argv
if DELTA:
    sys.argv.remove('--delta')
OUT = sys.argv[3]
C0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
C1 = int(sys.argv[5]) if len(sys.argv) > 5 else NEW[0]
t0 = time.time()
op, P = openings(OLD)
n = len(op)
print(f"machine {OLD}: P = {P:,}, {n:,} openings; adding {NEW} "
      f"(T = {T:,} laps); phase-0 range [{C0}, {C1})", flush=True)

hist = np.zeros(FMAX, np.int64)
blk = T // NEW[0]
first = np.full((C1 - C0) * blk, -1, np.int64)
last = np.full((C1 - C0) * blk, -1, np.int64)
R = len(NEW)
ndone = 0


def last_level_delta(arr, rl, cs):
    """THE DELTA FAST PATH for the innermost gear.

    The q children of one parent differ from the parent only by the removal of
    two residue classes - about 2/q of the elements - so the child's gap
    histogram is the PARENT's histogram with a local correction around each
    removed element, at cost proportional to the number REMOVED (2n/q) rather
    than to n.  At q = 41 that is a ~20x cut on the level that carries ~97% of
    the work.

    A maximal run of removed indices [S..E] deletes the parent gaps
    D[S-1 .. E] and creates the single merged gap arr[E+1] - arr[S-1];
    consecutive runs are separated by at least one kept element, so their
    D-index ranges [S-1, E] are disjoint and the corrections never overlap.
    A run touching an end of the array creates no merged gap and moves the
    child's first/last element instead.
    """
    global ndone
    q, u = NEW[R - 1], U[R - 1]
    n = len(arr)
    D = np.diff(arr)
    Hpar = np.bincount(D, minlength=FMAX)
    # ascending index lists per residue, from ONE radix argsort
    order = np.argsort(rl.astype(np.int16), kind='stable')
    cnt = np.bincount(rl, minlength=q)
    off = np.concatenate([[0], np.cumsum(cnt)])
    for c in range(C0, C1) if R == 1 else range(q):
        r1, r2 = (c - u) % q, (c + u) % q
        R1 = order[off[r1]:off[r1 + 1]]
        R2 = order[off[r2]:off[r2 + 1]]
        Rm = np.sort(np.concatenate([R1, R2])) if r1 != r2 else np.sort(R1)
        cs.append(c)
        h = Hpar
        if len(Rm):
            brk = np.flatnonzero(np.diff(Rm) != 1)
            S = Rm[np.concatenate([[0], brk + 1])]
            E = Rm[np.concatenate([brk, [len(Rm) - 1]])]
            lo = np.maximum(S - 1, 0)
            hi = np.minimum(E, n - 2)
            L = hi - lo + 1
            tot = int(L.sum())
            base = np.concatenate([[0], np.cumsum(L)[:-1]])
            gi = np.repeat(lo, L) + (np.arange(tot) - np.repeat(base, L))
            h = Hpar - np.bincount(D[gi], minlength=FMAX)
            mid = (S >= 1) & (E <= n - 2)
            if mid.any():
                g = arr[E[mid] + 1] - arr[S[mid] - 1]
                h = h + np.bincount(g, minlength=FMAX)
            f0 = int(arr[E[0] + 1]) if S[0] == 0 else int(arr[0])
            l0 = int(arr[S[-1] - 1]) if E[-1] == n - 1 else int(arr[-1])
        else:
            f0, l0 = int(arr[0]), int(arr[-1])
        hist[:] += h
        idx = flat_index(cs) - C0 * blk
        first[idx] = f0
        last[idx] = l0
        assert int(h.sum()) == n - len(Rm) - 1, (c, int(h.sum()), n, len(Rm))
        cs.pop()
        ndone += 1
        if ndone % 100000 == 0:
            print(f"  {ndone:,} laps t={time.time()-t0:.0f}s", flush=True)


def rec(level, arr, res, cs):
    """arr = surviving OLD-openings; res[t] = arr % NEW[t] for t >= level."""
    global ndone
    q, u = NEW[level], U[level]
    rl = res[level]
    lastlev = (level == R - 1)
    if lastlev and DELTA:
        last_level_delta(arr, rl, cs)
        return
    for c in range(C0, C1) if level == 0 else range(q):
        keep = (rl != (c - u) % q) & (rl != (c + u) % q)
        a2 = arr[keep]
        cs.append(c)
        if lastlev:
            assert len(a2) >= 2
            g = np.diff(a2)
            hist[:] += np.bincount(g, minlength=FMAX)
            assert int(g.max()) < FMAX
            idx = flat_index(cs) - C0 * blk
            first[idx] = a2[0]
            last[idx] = a2[-1]
            ndone += 1
            if ndone % 20000 == 0:
                print(f"  {ndone:,} laps t={time.time()-t0:.0f}s", flush=True)
        else:
            r2 = list(res)
            for t in range(level + 1, R):
                r2[t] = res[t][keep]
            rec(level + 1, a2, r2, cs)
        cs.pop()


res0 = [(op % q).astype(np.int8) for q in NEW]
rec(0, op, res0, [])
np.savez_compressed(OUT, hist=hist, first=first, last=last,
                    c0=np.int64(C0), c1=np.int64(C1))
print(f"worker c0 in [{C0},{C1}) done: {ndone:,} laps, "
      f"{int(hist.sum()):,} internal gaps, {time.time()-t0:.0f}s", flush=True)

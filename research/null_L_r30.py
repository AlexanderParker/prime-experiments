"""null_L_r30.py -- the null model for L(M), done exactly.

HARVESTER lane, round 30 follow-on (1).  Pre-registration:
research/data/r30/prereg_harvester_r30_null_L.txt (written first).

L(M) = longest realised legal word of machine M = {5..y} for its next prime q':
a run of consecutive gaps each = 0 or +-s (mod q'), s = 2u', u' = 6^{-1} mod q',
with the nonzero classes strictly alternating (T3), padded (= 0) letters
transparent.  Measured L = 1,1,1,2,1,3,3,2,2,2,4 at y = 11..47 (ledger).

NULL MODELS (gaps in one period, N = prod(q-2) of them):
  I-eq   i.i.d. gaps, each of the three classes with probability 1/q', NO
         alternation constraint (a run is legal iff every gap is in a class)
  I-eqA  same, WITH alternation
  I-act  i.i.d. gaps drawn from the machine's ACTUAL gap-value distribution
         (exact full-period histogram), no alternation
  I-actA same, with alternation                          <- the null N0
  M1-A   order-1 Markov chain on the gap CLASSES mod q' (transition matrix
         measured on the period, m11..m23 only), with alternation
For each the EXACT expectation of the longest legal run over N i.i.d. / Markov
steps is computed by a finite automaton and matrix powers (float64; the
longest-run law needs no precision beyond 1e-12 here).  Also the Erdos-Renyi
estimate log(N)/log(1/p) and the manager's q'/log(q'/3).
"""
from __future__ import annotations

import csv
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2, 43: 2, 47: 4}


def primes_upto(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(range(i * i, n + 1, i)))
    return [i for i in range(2, n + 1) if s[i]]


def next_prime(y):
    p = y + 1
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)):
        p += 1
    return p


def gears(y):
    return [q for q in primes_upto(y) if q >= 5]


def period_gaps(y):
    """cyclic gap sequence of one period, by direct sieve (y <= 23)."""
    gs = gears(y)
    P = 1
    for q in gs:
        P *= q
    k = np.arange(P, dtype=np.int64)
    op = np.ones(P, dtype=bool)
    for q in gs:
        u = pow(6, -1, q)
        r = k % q
        op &= (r != u) & (r != (q - u) % q)
    O = np.nonzero(op)[0]
    return np.diff(np.concatenate([O, [O[0] + P]]))


def hist_from_csv(y):
    f = os.path.join(HERE, "data", "r26", f"ghist_{y}.csv")
    h = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if int(row["y"]) == y:
                h[int(row["gap"])] = int(row["count"])
    return h


def classes(q):
    u = pow(6, -1, q)
    s = (2 * u) % q
    return u, s


def class_of(g, q, s):
    m = g % q
    if m == 0:
        return 0
    if m == s:
        return 1
    if m == (q - s) % q:
        return 2
    return 3


# --------------------------------------------------------------------------
# exact longest-legal-run expectation for an i.i.d. class sequence
# --------------------------------------------------------------------------
def prob_no_run_iid(pc, N, k, alternate):
    """P(no legal run of length >= k) over N i.i.d. class draws.
    pc = (p0, p+, p-, p_other).  States: (ell, c) with ell = current legal run
    length (0..k-1), c = last nonzero class in the run (0 none, 1 +, 2 -)."""
    if k == 1:
        return (1.0 - (pc[0] + pc[1] + pc[2])) ** N if N < 10_000 else math.exp(N * math.log(max(1e-300, 1.0 - (pc[0] + pc[1] + pc[2]))))
    idx = {}
    def st(ell, c):
        return idx.setdefault((ell, c), len(idx))
    st(0, 0)
    for ell in range(1, k):
        for c in (0, 1, 2):
            st(ell, c)
    n = len(idx)
    T = np.zeros((n, n))
    for (ell, c), i in idx.items():
        # other -> run breaks
        T[i, st(0, 0)] += pc[3]
        # padded
        if ell + 1 < k:
            T[i, st(ell + 1, c)] += pc[0]
        # literal +
        for cls, p in ((1, pc[1]), (2, pc[2])):
            if alternate and c == cls:
                # T3 violated: the run restarts at this gap (length 1, last = cls)
                if 1 < k:
                    T[i, st(1, cls)] += p
            else:
                if ell + 1 < k:
                    T[i, st(ell + 1, cls)] += p
    v = np.zeros(n); v[st(0, 0)] = 1.0
    # v * T^N  by repeated squaring
    R = np.eye(n)
    M = T.copy(); e = N
    while e:
        if e & 1:
            R = R @ M
        M = M @ M
        e >>= 1
    return float((v @ R).sum())


def expected_longest(pc, N, alternate, kmax=80):
    E = 0.0; tail = []
    for k in range(1, kmax + 1):
        pk = 1.0 - prob_no_run_iid(pc, N, k, alternate)      # P(L >= k)
        tail.append(pk)
        E += pk
        if pk < 1e-12:
            break
    return E, tail


# --------------------------------------------------------------------------
# order-1 Markov null on classes (needs the actual sequence)
# --------------------------------------------------------------------------
def prob_no_run_markov(Tm, pi0, N, k):
    """classes 0..3; Tm[a][b] = P(next = b | this = a).  States (ell, c, prev)."""
    if k == 1:
        # no legal gap at all: every class is 'other'
        return float(pi0[3] * (Tm[3][3] ** (N - 1))) if Tm[3][3] > 0 else 0.0
    idx = {}
    def st(ell, c, prev):
        return idx.setdefault((ell, c, prev), len(idx))
    for prev in range(4):
        st(0, 0, prev)
        for ell in range(1, k):
            for c in (0, 1, 2):
                st(ell, c, prev)
    n = len(idx)
    T = np.zeros((n, n))
    for (ell, c, prev), i in idx.items():
        for nxt in range(4):
            p = Tm[prev][nxt]
            if p == 0:
                continue
            if nxt == 3:
                T[i, st(0, 0, 3)] += p
            elif nxt == 0:
                if ell + 1 < k:
                    T[i, st(ell + 1, c, 0)] += p
            else:
                if c == nxt:
                    T[i, st(1, nxt, nxt)] += p
                elif ell + 1 < k:
                    T[i, st(ell + 1, nxt, nxt)] += p
    v = np.zeros(n)
    # start: first class drawn from pi0, run length 1 if legal (k >= 2 here)
    v[idx[(0, 0, 3)]] += pi0[3]
    v[idx[(1, 0, 0)]] += pi0[0]
    v[idx[(1, 1, 1)]] += pi0[1]
    v[idx[(1, 2, 2)]] += pi0[2]
    R = np.eye(n); M = T.copy(); e = N - 1
    while e:
        if e & 1:
            R = R @ M
        M = M @ M
        e >>= 1
    return float((v @ R).sum())


def expected_longest_markov(Tm, pi0, N, kmax=60):
    E = 0.0
    for k in range(1, kmax + 1):
        pk = 1.0 - prob_no_run_markov(Tm, pi0, N, k)
        E += pk
        if pk < 1e-12:
            break
    return E


def main():
    ys = [int(a) for a in sys.argv[1:]] or [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    print("y   q'   s   N            logN   3/q'    pL(act)  p0      p+      p-     "
          "ER(eq)  q'/ln(q'/3)  E[L] I-eq  I-eqA  I-act  I-actA  M1-A   L_meas  ratio N0/L")
    rows = []
    for y in ys:
        gs = gears(y); q = next_prime(y); u, s = classes(q)
        N = 1
        for g in gs:
            N *= (g - 2)
        logN = math.log(N)
        peq = (1 / q, 1 / q, 1 / q, 1 - 3 / q)
        E_eq, _ = expected_longest(peq, N, False)
        E_eqA, _ = expected_longest(peq, N, True)
        er_eq = logN / math.log(q / 3)
        mgr = q / math.log(q / 3)
        have_hist = y <= 37
        if have_hist:
            if y <= 23:
                gaps = period_gaps(y)
                vals, cnts = np.unique(gaps, return_counts=True)
                h = {int(v): int(c) for v, c in zip(vals, cnts)}
                if y >= 13:
                    hc = hist_from_csv(y)
                    assert hc == h, ("histogram mismatch vs r26 csv", y)
            else:
                h = hist_from_csv(y)
            assert sum(h.values()) == N, ("N mismatch", y, sum(h.values()), N)
            pc = [0.0, 0.0, 0.0, 0.0]
            for g, c in h.items():
                pc[class_of(g, q, s)] += c / N
            E_act, _ = expected_longest(tuple(pc), N, False)
            E_actA, tailA = expected_longest(tuple(pc), N, True)
            E_m1 = float("nan")
            if y <= 23:
                cl = np.array([class_of(int(g), q, s) for g in gaps], dtype=np.int64)
                Tm = np.zeros((4, 4))
                nxt = np.roll(cl, -1)
                for a in range(4):
                    for b in range(4):
                        Tm[a, b] = np.sum((cl == a) & (nxt == b))
                pi0 = Tm.sum(axis=1) / N
                rs = Tm.sum(axis=1, keepdims=True)
                Tm = np.divide(Tm, rs, out=np.zeros_like(Tm), where=rs > 0)
                E_m1 = expected_longest_markov(Tm, pi0, N)
            pL = pc[0] + pc[1] + pc[2]
        else:
            pc = [float("nan")] * 4; pL = float("nan")
            E_act = E_actA = E_m1 = float("nan")
        L = MEASURED_L[y]
        ratio = (E_actA / L) if have_hist else (E_eqA / L)
        rows.append((y, q, s, N, logN, pL, pc, E_eq, E_eqA, E_act, E_actA, E_m1, L, ratio, er_eq, mgr))
        print(f"{y:2d}  {q:2d}  {s:2d}  {N:12d}  {logN:5.2f}  {3/q:.4f}  {pL:.4f}  {pc[0]:.4f}  {pc[1]:.4f}  {pc[2]:.4f}  "
              f"{er_eq:6.2f}  {mgr:6.2f}       {E_eq:5.2f}  {E_eqA:5.2f}  {E_act:5.2f}  {E_actA:5.2f}  {E_m1:5.2f}   {L:d}      "
              f"{ratio:5.2f}{'' if have_hist else '  (proxy: eq-A / L)'}")
    # --- decomposition of the suppression at the machines with exact histograms
    print("\nDECOMPOSITION of E[L] null -> measured (multiplicative factors):")
    print("y    I-eq->I-eqA (alternation)   I-eqA->I-actA (class probs)   I-actA->L (dependence)   I-actA->M1-A (order-1 dep.)")
    for (y, q, s, N, logN, pL, pc, E_eq, E_eqA, E_act, E_actA, E_m1, L, ratio, er_eq, mgr) in rows:
        if y > 37:
            continue
        print(f"{y:2d}   {E_eqA/E_eq:6.3f}                      {E_actA/E_eqA:6.3f}                      "
              f"{L/E_actA:6.3f}                  {(E_m1/E_actA) if E_m1 == E_m1 else float('nan'):6.3f}")
    print("\nnull_L_r30: done")


if __name__ == "__main__":
    main()

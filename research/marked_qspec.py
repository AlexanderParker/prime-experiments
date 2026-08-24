"""Round 22 (mechanic, for Formalist): THE MARKED QUALIFYING SPECTRUM Q^[j].

Formalist's construct (agents-shared round-22, verdict 3). For a step
old -> new = old + q', with the NEXT prime q'' setting the floor a = 2u'':

  Q^[j](old) = max span x_J - x_0 over windows of OLD-machine openings
               x_0 < ... < x_J carrying J-1 MARKED interior openings
               x_1..x_{J-1} whose MIDDLE mutual distances are all >= a,
               such that every UNMARKED interior opening is KILLED by q'.

"Killed by q'" is phase-relative: gear q' kills the two residues
{c-u', c+u'} mod q' and every phase c occurs as the old period repeats q'
times inside the new period.  So a window is admissible iff SOME phase c
kills all unmarked interiors.  Dropping the requirement that the marked
openings SURVIVE that phase is what makes this a relaxation, hence
    Q_j(new) <= Q^[j](old)
which is the inequality under test.  Computed on the OLD machine's period
(q' times cheaper than the new machine's).

Usage: python marked_qspec.py            (runs the four checkable steps)
"""
import sys, time
from math import prod
import numpy as np

def primes_upto(n):
    s = np.ones(n+1, bool); s[:2] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]: s[i*i::i] = False
    return [int(p) for p in np.flatnonzero(s)]

def openings(y):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True; ex[(-u) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64), P

def feasible(interior, marked_needed, floor_a, x0, xJ):
    """Can we pick `marked_needed` marked interiors (the rest already forced)
    so that consecutive MIDDLE distances are all >= floor_a?"""
    # marked list must be a subset of `interior` of size marked_needed,
    # containing all of `forced`; middles are gaps between consecutive marked.
    n = len(interior)
    if marked_needed == 0:
        return n == 0
    if n < marked_needed:
        return False
    # DP over choices: pick indices increasing, track last picked position
    from functools import lru_cache
    iv = interior
    @lru_cache(maxsize=None)
    def rec(idx, cnt, last):
        if cnt == marked_needed:
            return True
        if idx >= n:
            return False
        # skip iv[idx] only if it is not forced
        res = False
        if not forced_flag[idx]:
            res = rec(idx+1, cnt, last)
        if res: return True
        # take iv[idx]
        if cnt == 0 or iv[idx] - last >= floor_a:
            if rec(idx+1, cnt+1, iv[idx]):
                return True
        return False
    return rec(0, 0, -10**18)

def marked_spectrum(old, qp, qpp, Jmax=5, span_cap=200, verbose=True):
    op, P = openings(old)
    n = len(op)
    up = pow(6, -1, qp)
    a = 2 * round(qpp / 6)
    res = (op % qp).astype(np.int64)
    ext = np.concatenate([op, op[:400] + P])
    rext = (ext % qp).astype(np.int64)
    best = {J: 0 for J in range(2, Jmax+1)}
    bestw = {J: None for J in range(2, Jmax+1)}
    t0 = time.time()
    global forced_flag
    for i in range(n):
        cov = np.zeros(qp, np.int64)
        n_int = 0
        for m in range(1, 400):
            span = int(ext[i+m] - ext[i])
            if span > span_cap: break
            if m >= 2:
                r = int(rext[i+m-1])
                cov[(r-up) % qp] += 1; cov[(r+up) % qp] += 1
                n_int = m-1
            best_cov = int(cov.max()) if n_int else 0
            if n_int - best_cov > Jmax-1: break
            for J in range(2, Jmax+1):
                if n_int < J-1: continue
                if span <= best[J]: continue
                if n_int - best_cov > J-1: continue
                interior = [int(x) for x in ext[i+1:i+m]]
                ok = False
                for c in range(qp):
                    unc = n_int - int(cov[c])
                    if unc > J-1: continue
                    kill = {(c-up) % qp, (c+up) % qp}
                    forced_flag = [ (int(rext[i+1+t]) not in kill) for t in range(n_int) ]
                    if sum(forced_flag) > J-1: continue
                    if feasible(tuple(interior), J-1, a, int(ext[i]), int(ext[i+m])):
                        ok = True; break
                if ok:
                    best[J] = span
                    bestw[J] = (int(ext[i]), span)
    if verbose:
        print(f"  computed in {time.time()-t0:.0f}s over {n:,} openings of machine {old}")
    return best, bestw, a

STEPS = [ (11, 13, 17), (13, 17, 19), (17, 19, 23), (19, 23, 29) ]
# known exact Q_j(new; 2u'') for j=2..5  (j=2 entry is F_2(new))
KNOWN = {13: [16, 18, 23, 0], 17: [25, 28, 31, 32], 19: [31, 35, 37, 38], 23: [39, 43, 50, 55]}
BUDGET = {13: 11+17, 17: 18+19, 19: 25+23, 23: 34+29}

for old, qp, qpp in STEPS:
    new = qp
    print(f"\nSTEP {old} -> {new} (q'={qp}, floor a = 2u''({qpp}))")
    best, bestw, a = marked_spectrum(old, qp, qpp)
    print(f"  floor a = {a}, budget F({new}) + {qpp} = {BUDGET[new]}")
    print(f"   J   Q_J({new}) exact   Q^[J]({old})   holds?   <= budget?")
    for J in range(2, 6):
        kn = KNOWN[new][J-2]
        mk = best[J]
        holds = "YES" if mk >= kn else "*** FAILS ***"
        within = "yes" if mk <= BUDGET[new] else "NO"
        print(f"  {J:2d}   {kn:8d}        {mk:8d}     {holds:14s} {within}")

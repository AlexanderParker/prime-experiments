"""Round 23 (mechanic): THE MULTI-GEAR MARKED SPECTRUM.

Round 23 established (research/j5_census.py) that the marked qualifying
spectrum computed on machine OLD with ONE free phase mod q' is EXACTLY
Q_J(OLD + q'), because requiring the endpoints and the marked openings to
survive phase c makes them precisely the consecutive new-machine openings,
and CRT makes every phase occur.

That argument does not care how many gears are added.  With r new gears
q_1..q_r and one free phase per gear, a window of OLD-machine openings plus a
phase tuple corresponds to exactly one address mod (P_old * q_1 * ... * q_r),
so scanning (window, phase tuple) pairs computes Q_J of the machine r gears
ahead - from the OLD machine's period, which is prod(q_i) times cheaper.

This is the construct that would price an upper bound on Q_j(47; 18): from
machine 29 it is 5 gears ahead (31,37,41,43,47) and the period ratio is
31*37*41*43*47 = 9.5e7.  Here it is BUILT and VALIDATED at r = 2:
    machine 23 + {29, 31}  ->  Q_J(31; 12), known exactly = 68/85/90/91/90/88
and its cost is measured so the r = 5 version can be priced honestly.

usage: uv run python research/j5_multi.py OLD q1[,q2,...] QPP
  e.g.  research/j5_multi.py 23 29,31 37
"""
import sys, time
from math import prod
import numpy as np

OLD = int(sys.argv[1])
NEW = [int(x) for x in sys.argv[2].split(',')]
QPP = int(sys.argv[3])
A_FLOOR = 2 * round(QPP / 6)
# ROUND-24: optional floor override (argv[7]).  Floor 1 = NO middle-gap
# constraint, so Q_J(target; 1) = F_J(target): the same lap-phase transfer
# decides the UNRESTRICTED spectrum of a machine r gears ahead.  Every other
# code path is untouched; feasible_marks with a = 1 accepts any mark set.
if len(sys.argv) > 7:
    A_FLOOR = int(sys.argv[7])
SPAN_CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 200
JMAX = int(sys.argv[6]) if len(sys.argv) > 6 else 7
# ROUND-25: optional argv[8] = 'legal'.  THE WORD-LEGAL CRITERION.
# The plain qualifying condition asks only that each of the J-2 middle gaps be
# >= a = 2u'.  What the MERGE LAW actually needs is stronger and is exactly
# a_kill.py's word legality: the J-1 interior openings are all deleted by ONE
# phase of gear QPP, so each middle gap must lie in V = {0, +s, -s} mod QPP
# (s = 2u' mod QPP) AND the resulting letter word must have prefix-sum range
# <= 1 (the two teeth are one step apart).  ">= a" is the shadow of that: the
# smallest positive legal value IS a, so 'legal' is a strict refinement and
# feasible_marks' a-spacing pre-filter stays sound.
WORDLEGAL = len(sys.argv) > 8 and sys.argv[8] == 'legal'
# ROUND-26: optional argv[9], argv[10] = I0, I1 - the half-open range of START
# OPENING INDICES this process walks.  The outer loop over start openings is
# embarrassingly parallel: every window is attributed to exactly one start
# index, so disjoint ranges tile the period and max over workers = the global
# maximum.  Each worker keeps its OWN running best (seeded identically), which
# only ever skips windows of span <= its own best >= the seed, so the union is
# still exact.  Purely additive: with fewer than 10 args nothing changes.
I0 = int(sys.argv[9]) if len(sys.argv) > 9 else 0
I1 = int(sys.argv[10]) if len(sys.argv) > 10 else None
_S_LET = (2 * pow(6, -1, QPP)) % QPP


def legal_word(gaps):
    """gaps (the J-2 middle gaps) form a kill word for one phase of QPP."""
    p = lo = hi = 0
    for v in gaps:
        r = v % QPP
        if r == 0:
            L = 0
        elif r == _S_LET:
            L = 1
        elif r == (-_S_LET) % QPP:
            L = -1
        else:
            return False
        p += L
        lo, hi = min(lo, p), max(hi, p)
    return hi - lo <= 1
F_EXACT = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
           43: 103, 47: 118, 53: 145}   # 47 pinned this round (F(2,47) = 354)
Q_EXACT = {(23, 10): {2: 39, 3: 43, 4: 50, 5: 55, 6: 60, 7: 0},
           (29, 10): {2: 55, 3: 65, 4: 68, 5: 71, 6: 71, 7: 71},
           (31, 12): {2: 68, 3: 85, 4: 90, 5: 91, 6: 90, 7: 88}}
TARGET = NEW[-1]
BUDGET = F_EXACT[TARGET] + QPP
KNOWN_Q = None if WORDLEGAL else Q_EXACT.get((TARGET, A_FLOOR))


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def openings(y):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    return np.flatnonzero(~ex).astype(np.int64), P


def feasible_marks(pos, forced, need, a):
    n = len(pos)
    if need == 0:
        return () if not forced else None
    if n < need:
        return None
    best = [None]

    def rec(idx, chosen, last):
        if best[0] is not None:
            return
        k = len(chosen)
        if k == need:
            if forced <= set(chosen):
                best[0] = tuple(chosen)
            return
        if idx >= n or n - idx < need - k:
            return
        if k == 0 or pos[idx] - last >= a:
            chosen.append(idx)
            rec(idx + 1, chosen, pos[idx])
            chosen.pop()
            if best[0] is not None:
                return
        if idx not in forced:
            rec(idx + 1, chosen, last)

    rec(0, [], -10 ** 18)
    return best[0]


def main():
    t0 = time.time()
    op, P = openings(OLD)
    n = len(op)
    us = [pow(6, -1, q) for q in NEW]
    print(f"machine {OLD} (P = {P:,}, {n:,} openings) + gears {NEW} "
          f"-> Q{'*' if WORDLEGAL else ''}_J({TARGET}; "
          f"{'word-legal for ' + str(QPP) if WORDLEGAL else A_FLOOR}); "
          f"budget F({TARGET})+{QPP} = {BUDGET}", flush=True)
    print(f"  period ratio bought: {prod(NEW):,}", flush=True)
    LOOK = 96
    # ROUND-26: build ext/res ONLY over the index window this process walks.
    # A start index i touches ext[i .. i+LOOK), so a worker on [I0, I1) needs
    # exactly [I0, I1 + LOOK) - and the local lists are then 1/nworkers of the
    # memory.  This matters: the full ext + res for six gears is ~700 MB of
    # Python objects per process, and MEMORY IS THE BINDING CONSTRAINT on this
    # box (compute policy).  With I0 = 0 and I1 = n this is the old object,
    # wrapped exactly as before (op + the first LOOK openings shifted by P).
    i_lo = I0
    i_hi = n if I1 is None else min(I1, n)
    hi_need = i_hi + LOOK
    if hi_need <= n:
        seg = op[i_lo:hi_need]
    else:
        seg = np.concatenate([op[i_lo:], op[:hi_need - n] + P])
    ext = [int(x) for x in seg]
    res = [[int(x) % q for x in seg] for q in NEW]
    NLOC = i_hi - i_lo
    arg = sys.argv[4] if len(sys.argv) > 4 else ''
    SEEDED = arg.startswith('seed')
    if SEEDED and arg != 'seed':
        v = int(arg[4:])          # explicit floor, e.g. seed94
        best = {J: v for J in range(2, JMAX + 1)}
        print(f"  SEEDED at {v} for every J - values at or below {v} are NOT "
              f"resolved; the reported max is max(true, {v}) and every window "
              f"of span > {v} is still examined", flush=True)
    elif SEEDED and KNOWN_Q:
        best = {J: (KNOWN_Q[J] - 1 if KNOWN_Q.get(J) else 0)
                for J in range(2, JMAX + 1)}
        print(f"  SEEDED at {best} (verification mode: every window of span "
              f"above the seed is still examined)", flush=True)
    else:
        best = {J: 0 for J in range(2, JMAX + 1)}
    bestw = {J: None for J in range(2, JMAX + 1)}
    nwin = ncand = 0
    # ROUND-26: print the walked range ALWAYS.  The range comes from trailing
    # POSITIONAL arguments, and a caller that omits an earlier optional slot
    # shifts them silently - which happened once this round (the F_2(53) launch
    # left out argv[8], so every worker read its own HI as its I0 and walked a
    # SUFFIX instead of a tile, leaving [0, HI_0) uncovered).  Printing the
    # range unconditionally makes that visible in the first line of the log.
    print(f"  WALKING start-opening indices [{i_lo:,}, {i_hi:,}) of {n:,}"
          + ("" if (i_lo, i_hi) == (0, n) else
             " - RANGE WORKER: the reported maxima are this range's; the "
             "global maximum is the max over a TILING set of workers"),
          flush=True)
    for i in range(NLOC):
        x0 = ext[i]
        covs = [[0] * q for q in NEW]
        mxs = [0] * len(NEW)
        n_int = 0
        lbmax = 0
        for m in range(1, LOOK):
            span = ext[i + m] - x0
            if span > SPAN_CAP:
                break
            if m >= 2:
                for gi, q in enumerate(NEW):
                    r = res[gi][i + m - 1]
                    for c in ((r - us[gi]) % q, (r + us[gi]) % q):
                        covs[gi][c] += 1
                        if covs[gi][c] > mxs[gi]:
                            mxs[gi] = covs[gi][c]
                n_int = m - 1
            # PRUNE.  S(m) = min over phase tuples of #survivors is NON-DECREASING
            # in m (the optimal tuple for the long window leaves at most as many
            # survivors in any prefix of it), and n_int - sum(mxs) is a lower
            # bound on S(m).  With r >= 2 gears that lower bound is NOT itself
            # monotone (sum(mxs) can gain r per step while n_int gains 1), so
            # breaking on it directly would be UNSOUND - windows could be missed.
            # Breaking on its RUNNING MAXIMUM is sound and monotone.
            lb = n_int - sum(mxs)
            if lb > lbmax:
                lbmax = lb
            if lbmax > JMAX - 1:
                break
            if n_int < 1:
                continue
            nwin += 1
            todo = [J for J in range(2, JMAX + 1)
                    if span > best[J] and n_int >= J - 1]
            if not todo:
                continue
            if lb > JMAX - 1:     # sound per-window SKIP (not a walk break):
                continue          # this window cannot leave <= JMAX-1 survivors
            ncand += 1
            pos = ext[i + 1:i + m]
            rr = [[res[gi][i + 1 + t] for t in range(n_int)]
                  for gi in range(len(NEW))]
            e0 = [res[gi][i] for gi in range(len(NEW))]
            e1 = [res[gi][i + m] for gi in range(len(NEW))]
            # enumerate phase tuples with a pruned product walk
            def walk(gi, alive, ends_ok):
                if not ends_ok:
                    return
                if gi == len(NEW):
                    forced = set(alive)
                    if len(forced) > JMAX - 1:
                        return
                    for J in list(todo):
                        need = J - 1
                        if len(forced) > need or n_int < need:
                            continue
                        if feasible_marks(pos, forced, need, A_FLOOR) is None:
                            continue
                        # marks must survive every new gear: marks == forced
                        if len(forced) != need:
                            continue
                        wf = tuple(sorted(forced))
                        mids = [pos[wf[t + 1]] - pos[wf[t]]
                                for t in range(need - 1)]
                        ok = (legal_word(mids) if WORDLEGAL
                              else all(g >= A_FLOOR for g in mids))
                        if ok:
                            if span > best[J]:
                                best[J] = span
                                bestw[J] = (x0, m, tuple(phases), wf)
                                todo.remove(J)
                    return
                q, uq = NEW[gi], us[gi]
                rem = sum(mxs[gi + 1:])
                # BRANCH ON DISTINCT KILL SETS, NOT ON PHASES.  Many phases of a
                # gear remove the same subset of the still-alive interiors (most
                # remove none at all), and exploring them separately made the
                # 6-gear tuple walk blow up: with weak pruning the tree is
                # 29*31*37*41*43*47 = 2.7e9 leaves.  One representative phase per
                # distinct kill set is exact - the survivors, hence the whole
                # admissibility question, depend on the phase only through it -
                # and collapses the branching to a handful.
                opts = {}
                for c in range(q):
                    k1, k2 = (c - uq) % q, (c + uq) % q
                    if e0[gi] in (k1, k2) or e1[gi] in (k1, k2):
                        continue        # endpoint must survive every gear
                    ks = frozenset(t for t in alive if rr[gi][t] in (k1, k2))
                    if ks not in opts:
                        opts[ks] = c
                for ks, c in opts.items():
                    na = [t for t in alive if t not in ks]
                    if len(na) > (JMAX - 1) + rem:
                        continue
                    phases.append(c)
                    walk(gi + 1, na, True)
                    phases.pop()
            phases = []
            walk(0, list(range(n_int)), True)
        # ROUND-26 rule 28: the progress stride must come from the WORKER's
        # own share, not the whole job - a stride larger than a worker's range
        # makes a healthy job indistinguishable from a stalled one.
        if i % max(1, NLOC // 20) == 0 and i:
            print(f"  i={i + i_lo:,}/{n:,} t={time.time()-t0:.0f}s best={best} "
                  f"(windows {nwin:,}, expanded {ncand:,})", flush=True)

    dt = time.time() - t0
    print(f"\nscan complete in {dt:.0f}s; windows walked {nwin:,}, "
          f"phase-expanded {ncand:,}", flush=True)
    print("   J    Q_J(%d; %d)   known" % (TARGET, A_FLOOR))
    for J in range(2, JMAX + 1):
        kn = KNOWN_Q.get(J) if KNOWN_Q else None
        print(f"  {J:2d}      {best[J]:6d}      {kn}")
    mx = max(best.values())
    print(f"\n  max over J = {mx}  vs budget {BUDGET}  -> "
          f"{'CERTIFIES' if mx <= BUDGET else 'FAILS by +%d' % (mx-BUDGET)}")
    if KNOWN_Q:
        bad = [(J, best[J], KNOWN_Q[J]) for J in range(2, JMAX + 1)
               if KNOWN_Q.get(J) is not None and best[J] != KNOWN_Q[J]]
        print("  ANCHOR:", "PASSED" if not bad else f"MISMATCH {bad}")
    for J in range(2, JMAX + 1):
        if bestw[J]:
            print(f"    J={J}: k={bestw[J][0]:,} span={best[J]} "
                  f"phases={bestw[J][2]} marks={bestw[J][3]}")


if __name__ == '__main__':
    main()

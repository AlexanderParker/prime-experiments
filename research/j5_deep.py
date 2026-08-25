"""Round 23 (mechanic): the marked qualifying spectrum at DEEP machines.

Same object as research/j5_census.py (R0 = round-22 relaxation, R1 = + endpoint
survival, R2 = + marked survival, which is EXACTLY Q_J(new)), but the old
machine's period is sieved in SEGMENTS so machines 29 and 31 fit in memory
(machine 29: P = 1.078e9 slots, 214,708,725 openings).

usage: uv run python research/j5_deep.py OLD QP QPP [seg_slots]
  e.g.  research/j5_deep.py 29 31 37     -> bounds Q_J(31; 12), budget F(31)+37
"""
import sys, time
from math import prod
import numpy as np

F_EXACT = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
Q_EXACT = {(23, 10): {2: 39, 3: 43, 4: 50, 5: 55, 6: 60, 7: 0},
           (29, 10): {2: 55, 3: 65, 4: 68, 5: 71, 6: 71, 7: 71},
           (31, 12): {2: 68, 3: 85, 4: 90, 5: 91, 6: 90, 7: 88},
           (37, 14): {2: 90, 3: 97, 4: 105, 5: 113, 6: 120, 7: None}}
OLD, QP, QPP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
SEG = int(sys.argv[4]) if len(sys.argv) > 4 else 50_000_000
SEEDED = (len(sys.argv) > 5 and sys.argv[5] == 'seed')
A_FLOOR = 2 * round(QPP / 6)
BUDGET = F_EXACT[QP] + QPP
KNOWN_Q = Q_EXACT.get((QP, A_FLOOR))
JMAX = 7
SPAN_CAP = 160
SPAN_MIN_REPORT = BUDGET + 1
LOOK = 80


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


GEARS = [p for p in primes_upto(OLD) if p >= 5]
P = prod(GEARS)


def seg_openings(lo, hi):
    ex = np.zeros(hi - lo, bool)
    for g in GEARS:
        u = pow(6, -1, g)
        for t in (u % g, (-u) % g):
            start = (t - lo) % g
            ex[start::g] = True
    return (np.flatnonzero(~ex) + lo)


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
    u = pow(6, -1, QP)
    print(f"machine {OLD}: gears {GEARS}, period {P:,}; q'={QP}, u'={u}, "
          f"floor a={A_FLOOR}, budget F({QP})+{QPP} = {BUDGET}", flush=True)
    # SEED: start each running maximum one below the known exact Q_J, so the
    # scan only expands windows that could MATCH OR BEAT it.  This does not
    # weaken the result - every window of span > seed is still examined, so a
    # reported max above the seed is the true max, and any excess would be
    # found - but it must be reported as a seeded verification, not a blind
    # re-derivation.  Without it the warm-up cost dominates (measured 1,544
    # openings/s unseeded at machine 29 vs the seeded rate printed below).
    seed = {J: (KNOWN_Q[J] - 1 if KNOWN_Q and KNOWN_Q.get(J) else 0)
            for J in range(2, JMAX + 1)}
    if SEEDED:
        print(f"  SEEDED at {seed} (verification mode)", flush=True)
    else:
        seed = {J: 0 for J in range(2, JMAX + 1)}
    best = {R: dict(seed) for R in (0, 1, 2)}
    bestw = {R: {J: None for J in range(2, JMAX + 1)} for R in (0, 1, 2)}
    hits = []
    prune_depth = JMAX - 1
    buf = []
    first = None
    done = 0

    def process(upto):
        nonlocal done
        rl = [x % QP for x in buf]
        for i in range(upto):
            cov = [0] * QP
            mx = 0
            n_int = 0
            x0 = buf[i]
            for m in range(1, LOOK):
                span = buf[i + m] - x0
                if span > SPAN_CAP:
                    break
                if m >= 2:
                    r = rl[i + m - 1]
                    c1 = (r - u) % QP; c2 = (r + u) % QP
                    cov[c1] += 1
                    if cov[c1] > mx: mx = cov[c1]
                    cov[c2] += 1
                    if cov[c2] > mx: mx = cov[c2]
                    n_int = m - 1
                if n_int - mx > prune_depth:
                    break
                if n_int < 1:
                    continue
                interesting = (span >= SPAN_MIN_REPORT)
                if not interesting:
                    for J in range(2, JMAX + 1):
                        if span > best[0][J] and n_int >= J - 1:
                            interesting = True
                            break
                if not interesting:
                    continue
                pos = buf[i + 1:i + m]
                rpos = rl[i + 1:i + m]
                e0_r, e1_r = rl[i], rl[i + m]
                lim = n_int - (JMAX - 1)
                for c in range(QP):
                    if cov[c] < lim:      # cannot leave <= JMAX-1 unkilled
                        continue
                    kill = {(c - u) % QP, (c + u) % QP}
                    forced = {t for t in range(n_int) if rpos[t] not in kill}
                    if len(forced) > JMAX - 1:
                        continue
                    ends_ok = (e0_r not in kill) and (e1_r not in kill)
                    for J in range(2, JMAX + 1):
                        need = J - 1
                        if len(forced) > need or n_int < need:
                            continue
                        w = feasible_marks(pos, forced, need, A_FLOOR)
                        if w is None:
                            continue
                        if span > best[0][J]:
                            best[0][J] = span; bestw[0][J] = (x0, m, c, w)
                        if ends_ok:
                            if span > best[1][J]:
                                best[1][J] = span; bestw[1][J] = (x0, m, c, w)
                            if len(forced) == need:
                                wf = tuple(sorted(forced))
                                if all(pos[wf[t + 1]] - pos[wf[t]] >= A_FLOOR
                                       for t in range(need - 1)):
                                    if span > best[2][J]:
                                        best[2][J] = span
                                        bestw[2][J] = (x0, m, c, wf)
                        if J == 5 and span >= SPAN_MIN_REPORT:
                            hits.append((x0, m, span, c, w, tuple(pos),
                                         tuple(rpos), e0_r, e1_r, ends_ok,
                                         len(forced)))
            done += 1

    lo = 0
    while lo < P:
        hi = min(lo + SEG, P)
        seg = seg_openings(lo, hi)
        if first is None:
            first = [int(x) for x in seg[:LOOK]]
        buf.extend(int(x) for x in seg)
        upto = len(buf) - LOOK
        if upto > 0:
            process(upto)
            buf = buf[upto:]
        print(f"  slots {hi:,}/{P:,}  openings done {done:,}  "
              f"t={time.time()-t0:.0f}s  R0={best[0]}  hits={len(hits)}",
              flush=True)
        lo = hi
    # cyclic tail
    buf.extend(x + P for x in first)
    process(len(buf) - LOOK)

    print(f"\nscan complete in {time.time()-t0:.0f}s, {done:,} openings",
          flush=True)
    print("   J    R0 (r22 relax)   R1 (+ends)   R2 (= Q_J exact)   known")
    for J in range(2, JMAX + 1):
        kn = KNOWN_Q.get(J) if KNOWN_Q else None
        print(f"  {J:2d}      {best[0][J]:6d}        {best[1][J]:6d}"
              f"        {best[2][J]:6d}          {kn}")
    m0 = max(best[0].values()); m1 = max(best[1].values()); m2 = max(best[2].values())
    print(f"\n  max over J:  R0 {m0}   R1 {m1}   R2 {m2}   budget {BUDGET}")
    for R, mm in ((0, m0), (1, m1), (2, m2)):
        print(f"  R{R}: {'CERTIFIES the rung' if mm <= BUDGET else 'FAILS by +%d' % (mm - BUDGET)}")
    if KNOWN_Q:
        bad = [(J, best[2][J], KNOWN_Q[J]) for J in range(2, JMAX + 1)
               if KNOWN_Q.get(J) is not None and best[2][J] != KNOWN_Q[J]]
        print("  ANCHOR:", "PASSED (R2 == known exact Q_J)" if not bad
              else f"MISMATCH {bad} - REPORT, do not overwrite")
    print("\n  witnesses (R2, = the exact qualifying windows):")
    for J in range(2, JMAX + 1):
        if bestw[2][J]:
            x0, m, c, w = bestw[2][J]
            print(f"    J={J}: k={x0:,} span={best[2][J]} phase c={c} marks={w}")
    print(f"\n  J=5 R0 windows of span >= {SPAN_MIN_REPORT}: {len(hits)}")
    for h in hits[:20]:
        print("   ", h)


if __name__ == '__main__':
    main()

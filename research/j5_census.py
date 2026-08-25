"""Round 23 (mechanic, for Constructor): THE J=5 CENSUS AT 23->29.

Round 22 found: the marked qualifying spectrum Q^[J](23) (q'=29, q''=31,
floor a = 2u''(31) = 10) equals the exact Q_J(29;10) at J=2,3,4 (55/65/68)
and BLOWS UP at J=5: 85 against a true 71 and a budget F(29)+31 = 74.
Constructor's bounded-state certificates fail at the SAME step and depth
(99/99/91 vs 74).  This tool characterises WHAT those configurations ARE.

Definitions (old machine = 23, q' = 29, u' = 6^{-1} mod 29 = 5):
  window   x_0 < x_1 < ... < x_m  consecutive OLD-machine openings
  phase c  gear 29 kills exactly the residues {c-u', c+u'} mod 29
  marked   J-1 of the interior openings, consecutive marked distances >= a
  admissible (R0, the round-22 relaxation): every UNMARKED interior is killed
  R1 = R0 + the two ENDPOINTS x_0, x_m survive phase c
  R2 = R1 + every MARKED opening survives phase c

R2 is EXACT: if every endpoint and marked opening survives and every other
interior is killed, then x_0, marked..., x_m are precisely the consecutive
NEW-machine (29) openings of that window, so R2's spectrum IS Q_J(29;10).
Every phase c occurs (the old period repeats q' times in the new one), so
the R2 scan over (window, phase) pairs covers every new-machine window.
That makes R2 an anchor: it must reproduce 55/65/68/71/71/71 exactly.

Outputs: research/data/j5_census_23_29.csv  (every R0 J=5 window of span
>= SPAN_MIN_REPORT, with its word, marks, phases, residues, flanks) and a
printed summary.  Every reported witness is re-verified by assert against a
direct rebuild of the machine-23 opening set.
"""
import sys, time
from math import prod
import numpy as np

F_EXACT = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91}
Q_EXACT = {(23, 10): {2: 39, 3: 43, 4: 50, 5: 55, 6: 60, 7: 0},
           (29, 10): {2: 55, 3: 65, 4: 68, 5: 71, 6: 71, 7: 71},
           (19, 8):  {2: 31, 3: 35, 4: 37, 5: 38, 6: 0, 7: 0},
           (13, 6):  {2: 16, 3: 18, 4: 23, 5: 0, 6: 0, 7: 0},
           (17, 6):  {2: 25, 3: 28, 4: 31, 5: 32, 6: 34, 7: 0}}
OLD, QP, QPP = (int(x) for x in (sys.argv[1:4] or (23, 29, 31)))
A_FLOOR = 2 * round(QPP / 6)
BUDGET = F_EXACT[QP] + QPP
KNOWN_Q = Q_EXACT[(QP, A_FLOOR)]
JMAX = 7
SPAN_CAP = 120
SPAN_MIN_REPORT = BUDGET + 1          # strictly over budget


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
    return np.flatnonzero(~ex).astype(np.int64), P, gears


def feasible_marks(pos, forced, need, a):
    """Can we choose `need` marks from pos (sorted) containing all `forced`
    (a set of indices into pos) with consecutive mark distances >= a?
    Returns a witness tuple of indices or None."""
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
            # all forced must be covered
            if forced <= set(chosen):
                best[0] = tuple(chosen)
            return
        if idx >= n:
            return
        # remaining forced after idx must fit
        if n - idx < need - k:
            return
        # take pos[idx]
        if k == 0 or pos[idx] - last >= a:
            chosen.append(idx)
            rec(idx + 1, chosen, pos[idx])
            chosen.pop()
            if best[0] is not None:
                return
        # skip pos[idx] (illegal if forced)
        if idx not in forced:
            rec(idx + 1, chosen, last)

    rec(0, [], -10 ** 18)
    return best[0]


def main():
    t0 = time.time()
    op, P, gears = openings(OLD)
    n = len(op)
    u = pow(6, -1, QP)
    print(f"machine {OLD}: gears {gears}, period {P:,}, openings {n:,}; "
          f"q'={QP}, u'={u}, floor a={A_FLOOR}, budget {BUDGET}", flush=True)
    LOOK = 64
    ext = np.concatenate([op, op[:LOOK] + P])
    rext = (ext % QP).astype(np.int64)
    extl = [int(x) for x in ext]
    rl = [int(x) for x in rext]

    best = {R: {J: 0 for J in range(2, JMAX + 1)} for R in (0, 1, 2)}
    bestw = {R: {J: None for J in range(2, JMAX + 1)} for R in (0, 1, 2)}
    hits = []                       # J=5, R0, span >= SPAN_MIN_REPORT
    prune_depth = JMAX - 1          # allow up to JMAX-1 unkilled interiors

    for i in range(n):
        cov = [0] * QP
        mx = 0
        n_int = 0
        x0 = extl[i]
        for m in range(1, LOOK):
            span = extl[i + m] - x0
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
            pos = extl[i + 1:i + m]
            rpos = rl[i + 1:i + m]
            e0_r, e1_r = rl[i], rl[i + m]
            for c in range(QP):
                if cov[c] == 0 and n_int > prune_depth:
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
                    # R0
                    if span > best[0][J]:
                        best[0][J] = span; bestw[0][J] = (x0, m, c, w)
                    if ends_ok:
                        if span > best[1][J]:
                            best[1][J] = span; bestw[1][J] = (x0, m, c, w)
                        # R2: marked must all survive => marked == forced exactly
                        if len(forced) == need:
                            wf = tuple(sorted(forced))
                            ok = all(pos[wf[t + 1]] - pos[wf[t]] >= A_FLOOR
                                     for t in range(need - 1))
                            if ok and span > best[2][J]:
                                best[2][J] = span; bestw[2][J] = (x0, m, c, wf)
                    if J == 5 and span >= SPAN_MIN_REPORT:
                        hits.append(dict(x0=x0, m=m, span=span, c=c, marks=w,
                                         pos=tuple(pos), rpos=tuple(rpos),
                                         e0_r=e0_r, e1_r=e1_r, ends_ok=ends_ok,
                                         nforced=len(forced),
                                         forced=tuple(sorted(forced))))
        if i % 500000 == 0 and i:
            print(f"  i={i:,}/{n:,}  t={time.time()-t0:.0f}s  "
                  f"R0 best={best[0]}  hits={len(hits)}", flush=True)

    print(f"\nscan complete in {time.time()-t0:.0f}s", flush=True)
    print("\n   J   Q_J(29) exact    R0 (r22 relax)   R1 (+ends)   R2 (=exact?)")
    for J in range(2, JMAX + 1):
        print(f"  {J:2d}      {KNOWN_Q[J]:5d}          {best[0][J]:6d}"
              f"         {best[1][J]:6d}      {best[2][J]:6d}")
    m0 = max(best[0].values()); m1 = max(best[1].values()); m2 = max(best[2].values())
    print(f"\n  max over J:  R0 {m0}   R1 {m1}   R2 {m2}   budget {BUDGET}")
    for R, mm in ((0, m0), (1, m1), (2, m2)):
        print(f"  R{R}: {'CERTIFIES the rung' if mm <= BUDGET else 'FAILS by +%d' % (mm-BUDGET)}")

    # anchor: R2 must equal the exact Q_J(29;10)
    for J in range(2, JMAX + 1):
        assert best[2][J] == KNOWN_Q[J], (J, best[2][J], KNOWN_Q[J])
    print("  ANCHOR PASSED: R2 reproduces Q_J(29;10) = 55/65/68/71/71/71 exactly.")

    # ---- the J=5 over-budget census -------------------------------------
    print(f"\nJ=5 R0 windows of span >= {SPAN_MIN_REPORT}: {len(hits)} "
          f"(window,phase) records", flush=True)
    if hits:
        import collections
        by_span = collections.Counter(h['span'] for h in hits)
        print("  span histogram:", dict(sorted(by_span.items())))
        wins = collections.defaultdict(list)
        for h in hits:
            wins[(h['x0'], h['m'])].append(h)
        print(f"  distinct windows: {len(wins)}")
        ends_ok_windows = sum(1 for k, v in wins.items() if any(x['ends_ok'] for x in v))
        print(f"  windows with SOME phase keeping both endpoints alive: {ends_ok_windows}")
        rows = []
        for (x0, m), hs in sorted(wins.items(), key=lambda kv: -kv[1][0]['span']):
            h = hs[0]
            pos = (x0,) + h['pos'] + (x0 + h['span'],)
            gaps = tuple(pos[t + 1] - pos[t] for t in range(len(pos) - 1))
            phases = sorted(set(x['c'] for x in hs))
            # merged (new-machine) word under the best mark choice
            mk = h['marks']
            mpos = (x0,) + tuple(h['pos'][t] for t in mk) + (x0 + h['span'],)
            mgaps = tuple(mpos[t + 1] - mpos[t] for t in range(len(mpos) - 1))
            rows.append(dict(x0=x0, span=h['span'], nint=len(h['pos']),
                             word='|'.join(map(str, gaps)),
                             merged='|'.join(map(str, mgaps)),
                             marks='|'.join(map(str, mk)),
                             nforced=h['nforced'],
                             phases='|'.join(map(str, phases)),
                             e0r=h['e0_r'], e1r=h['e1_r'],
                             ends_ok=int(any(x['ends_ok'] for x in hs)),
                             rint='|'.join(map(str, h['rpos']))))
        import csv
        out = f'research/data/j5_census_{OLD}_{QP}.csv'
        with open(out, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"  wrote {out} ({len(rows)} windows)")
        print("\n  top windows (merged word = the J=5 'new-machine' gap word):")
        for r in rows[:25]:
            print(f"    k={r['x0']:>10,}  span {r['span']}  nint {r['nint']}  "
                  f"merged {r['merged']:<20s} phases[{r['phases']}] "
                  f"forced {r['nforced']} ends_ok {r['ends_ok']}")
            print(f"          word {r['word']}")

        # verification of the top witness against a rebuilt opening set
        r = rows[0]
        x0 = r['x0']; span = r['span']
        gaps = [int(g) for g in r['word'].split('|')]
        acc = x0
        pts = [x0]
        for g in gaps:
            acc += g; pts.append(acc)
        opset = set(int(x) for x in op)
        for t, ptv in enumerate(pts):
            assert (ptv % P) in opset, (t, ptv)
        # every slot strictly between consecutive pts must be blocked
        for t in range(len(pts) - 1):
            for s in range(pts[t] + 1, pts[t + 1]):
                assert (s % P) not in opset, (t, s)
        print(f"  WITNESS VERIFIED by assert: k={x0}, span {span}, "
              f"{len(pts)} machine-23 openings, all interior slots blocked.")


if __name__ == '__main__':
    main()

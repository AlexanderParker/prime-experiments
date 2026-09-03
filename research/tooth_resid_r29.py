"""
LATERAL round 29, BLOCK A - RESIDUAL INCREMENT-LAW VIOLATORS AFTER TOOTH PINNING.

Round 28 measured, over the exhaustive counterfactual family

    V(y) = prod_{q<=y} {1..(q-1)/2}      (gear q's teeth at +-v_q)

that the INCREMENT LAW  inc := F(M+q') - F_2(M) <= s_min(q')  is violated by
13.3 / 13.9 / 14.5 / 21.7 percent of members at 7->11 .. 17->19, and that PINNING
the incoming gear's tooth to the twin value v_q' = round(q'/6) drops that to
0 / 0 / 1.1 / 6.5 percent.  This script characterises WHAT REMAINS.

Notation for a step M = {5..y} -> q':  u' = round(q'/6), a = 2u', b = q'-a,
s_min = min(a,b) = a.  The LEGAL LETTER CLASSES mod q' are {0, a, b}.

Objects computed per family member (old tooth vector v, new tooth v_q'):

  F, F_2                 old machine's record and two-gap record (cyclic)
  Q*_J   (J = 3..JCAP)   max span of a WORD-LEGAL J-window: J-2 middles each
                         0 or +-2v_q' mod q' (T2) with the NONZERO classes
                         strictly alternating and padded middles transparent (T3)
  F(M+q')                the new machine's record (direct sieve, or the block
                         decomposition at 19->23)
  Pcong                  F(M) mod q' in {0,a,b}          (Constructor's shape)
  Pbig                   the old machine realises a legal gap w > a
  A4                     Q*_3 > F_2 + a                  (the depth-3 predicate)

GATES.  The headline gate is the ATTAINMENT THEOREM tested FAMILY-WIDE:
max(F_2, max_{J>=3} Q*_J) must equal F(M+q') at EVERY member, not just at the
real machine.  (The theorem's proof is CRT + the two-tooth structure, both of
which every family member has, so this is a genuine prediction about 27k sieves.)

Usage:
  python tooth_resid_r29.py --steps small          # 7->11 .. 17->19, full family
  python tooth_resid_r29.py --steps 19_23 --workers 4
  python tooth_resid_r29.py --report
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

GEARS = [5, 7, 11, 13, 17, 19, 23]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "r29")

JCAP = 7          # windows of at most JCAP gaps; asserted never binding
NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def sieve_openings(gears, vs, P):
    blocked = np.zeros(P, dtype=bool)
    for q, v in zip(gears, vs):
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    return np.flatnonzero(~blocked).astype(np.int32)


def Fj_ladder(g, jmax=6):
    """F_j = max sum of j consecutive cyclic gaps, j = 1..jmax."""
    n = g.size
    ext = np.concatenate([g, g[:jmax + 1]]).astype(np.int64)
    pre = np.zeros(ext.size + 1, dtype=np.int64)
    np.cumsum(ext, out=pre[1:])
    out = []
    for j in range(1, jmax + 1):
        out.append(int((pre[j:n + j] - pre[:n]).max()))
    return out


def cyclic_gaps(op, P):
    """The N cyclic gaps, in opening order (gap t = o_{t+1} - o_t, last wraps)."""
    g = np.empty(op.size, dtype=np.int32)
    g[:-1] = op[1:] - op[:-1]
    g[-1] = np.int32(P) - op[-1] + op[0]
    return g


def lex_ok(cj, last):
    """T3: a nonzero class may not follow the same nonzero class."""
    return (cj == 0) | (last == 0) | (last != cj)


def qstar_family(g, qp, vqp, jcap=JCAP):
    """(Q3, Q3_middle, Qmax, Jmax_used, capped, minlegalmid_of_Q3) for one member.

    g is the cyclic gap array.  A word-legal J-window has J-2 middles; we sweep
    k = J-2 = 1..jcap-2 middles.  Legality: value mod q' in {0, a, b}; class 0
    for the padded letter, +1 for a, -1 for b; the nonzero classes must strictly
    alternate along the run (padded letters transparent).
    """
    a = (2 * vqp) % qp
    b = qp - a
    smin = min(a, b)
    n = g.size
    K = jcap                      # margin for cyclic wrap
    ext = np.concatenate([g, g[:K + 2]])
    r = ext % qp
    cls = np.zeros(ext.size, dtype=np.int8)
    legal = np.zeros(ext.size, dtype=bool)
    legal |= (r == 0)
    cls[r == a] = 1
    legal |= (r == a)
    cls[r == b] = -1
    legal |= (r == b)
    if a == b:                    # cannot happen for odd q' but be explicit
        raise AssertionError("a == b")
    pre = np.zeros(ext.size + 1, dtype=np.int64)
    np.cumsum(ext, out=pre[1:])

    # run of k middles starting at index i (i >= 1 so a left flank exists).
    # After the first level the valid set is ~3% of the array, so COMPACT it:
    # every deeper level then costs 3% of what a full-width pass would.
    idx = np.arange(1, n + 1)
    Q = np.full(jcap - 1, -1, dtype=np.int64)     # Q[k-1] = Q*_{k+2}
    Q3mid = -1
    Q3fl = -1
    Q3 = -1
    capped = False
    keep = legal[idx]
    idx = idx[keep]
    last = cls[idx]
    for k in range(1, jcap - 1):
        if k > 1:
            j = idx + (k - 1)
            cj = cls[j]
            ok = lex_ok(cj, last) & legal[j]
            idx = idx[ok]
            last = np.where(cj[ok] != 0, cj[ok], last[ok])
        if idx.size == 0:
            break
        span = (ext[idx - 1].astype(np.int64)
                + (pre[idx + k] - pre[idx])
                + ext[idx + k].astype(np.int64))
        am = int(np.argmax(span))
        m = int(span[am])
        Q[k - 1] = m
        if k == 1:
            Q3 = m
            Q3mid = int(ext[idx[am]])
            Q3fl = int(min(int(ext[idx[am] - 1]), int(ext[idx[am] + 1])))
        if k == jcap - 2:
            capped = True
    Qmax = int(Q.max())
    jmax = int(np.argmax(Q)) + 3 if Qmax >= 0 else -1
    # does the old machine realise a legal gap strictly above a?
    gl = legal[:n]
    big = bool(np.any(gl & (g > smin)))
    return Q3, Q3mid, Q3fl, Qmax, jmax, capped, big, Q.copy()


P19 = 5 * 7 * 11 * 13 * 17 * 19
P23 = P19 * 23


def m23_F(O19, rr, v23):
    """F of the m23 machine from the m19 opening set (block decomposition)."""
    c = P19 % 23
    t0, t1 = v23 % 23, (-v23) % 23
    best = 0
    firsts = np.empty(23, dtype=np.int64)
    lasts = np.empty(23, dtype=np.int64)
    for j in range(23):
        s0 = (t0 - j * c) % 23
        s1 = (t1 - j * c) % 23
        v = O19[(rr != s0) & (rr != s1)]
        d = v[1:] - v[:-1]
        m = int(d.max())
        if m > best:
            best = m
        off = j * P19
        firsts[j] = int(v[0]) + off
        lasts[j] = int(v[-1]) + off
    nxt = np.roll(firsts, -1).copy()
    nxt[-1] += P23
    best = max(best, int((nxt - lasts).max()))
    return best


def work_small(args):
    """One chunk of old-tooth vectors for a step with a direct new-machine sieve."""
    ogears, qp, lo, hi = args
    P = 1
    for q in ogears:
        P *= q
    Pn = P * qp
    space = [list(range(1, (q - 1) // 2 + 1)) for q in ogears]
    ovecs = list(itertools.product(*space))
    rows = []
    for oi in range(lo, hi):
        ov = list(ovecs[oi])
        op = sieve_openings(ogears, ov, P)
        g = cyclic_gaps(op, P)
        F = int(g.max())
        Fl = Fj_ladder(g, 5)
        F2 = Fl[1]
        F3, F4, F5 = Fl[2], Fl[3], Fl[4]
        for vqp in range(1, (qp - 1) // 2 + 1):
            Q3, Q3mid, Q3fl, Qmax, jmax, capped, big, Qall = qstar_family(g, qp, vqp)
            opn = sieve_openings(ogears + [qp], ov + [vqp], Pn)
            gn = cyclic_gaps(opn, Pn)
            Fn = int(gn.max())
            if opn.size != int(np.prod([q - 2 for q in ogears + [qp]])):
                raise AssertionError("new-machine opening count moved")
            rows.append((oi, vqp, F, F2, F3, F4, F5, Fn, Q3, Q3mid, Q3fl,
                         Qmax, jmax, int(capped), int(big), int(op.size)))
    return rows


def work_1923(args):
    """One chunk of m19 tooth vectors at the PINNED 19->23 step (v_23 = 4).

    `redo` = re-derive F(m23) here with the block decomposition (the expensive
    half, ~40 ms a member).  With redo=False the run takes F(m23) from round
    28's gated table `data/r28/tooth_m23_pinned.npy` and a random SAMPLE is
    re-derived separately as the double-source gate.
    """
    lo, hi, vqp, redo = args
    ogears = [5, 7, 11, 13, 17, 19]
    space = [list(range(1, (q - 1) // 2 + 1)) for q in ogears]
    ovecs = list(itertools.product(*space))
    rows = []
    for oi in range(lo, hi):
        ov = list(ovecs[oi])
        op = sieve_openings(ogears, ov, P19)
        g = cyclic_gaps(op, P19)
        F = int(g.max())
        Fl = Fj_ladder(g, 5)
        F2 = Fl[1]
        F3, F4, F5 = Fl[2], Fl[3], Fl[4]
        Q3, Q3mid, Q3fl, Qmax, jmax, capped, big, Qall = qstar_family(g, 23, vqp)
        if redo:
            rr = (op % 23).astype(np.int8)
            Fn = m23_F(op, rr, vqp)
        else:
            Fn = -1
        rows.append((oi, vqp, F, F2, F3, F4, F5, Fn, Q3, Q3mid, Q3fl,
                     Qmax, jmax, int(capped), int(big), int(op.size)))
    return rows


COLS = ("oi vqp F F2 F3 F4 F5 Fn Q3 Q3mid Q3fl Qmax jmax capped big "
        "nopen").split()


def analyse(tag, arr, ogears, qp, twin_ov_index, npin):
    """Score the predicates on one step's table."""
    d = {c: arr[:, i] for i, c in enumerate(COLS)}
    A = (2 * d["vqp"]) % qp
    B = qp - A
    smin = np.minimum(A, B)
    inc = d["Fn"] - d["F2"]
    viol = inc > smin
    print("")
    print("===== STEP {5..%d} -> %d   (%d members) ====="
          % (ogears[-1], qp, len(arr)))

    att = np.maximum(d["F2"], d["Qmax"])
    gate(bool(np.all(att == d["Fn"])),
         "ATTAINMENT max(F_2, max_J Q*_J) == F(M+q') at all %d members" % len(arr))
    gate(not bool(np.any(d["capped"])),
         "no legal window reached the J = %d cap" % JCAP)
    gate(bool(np.all(d["nopen"] == np.prod([q - 2 for q in ogears]))),
         "every member has prod(q-2) = %d openings" % np.prod([q - 2 for q in ogears]))
    # A0 CORRECTED (my pre-registered A0 was WRONG - see the round-29 scorecard).
    # The right necessary condition is the PEEL BOUND, not a statement about the
    # middle: F_2 >= g_L + w and F_2 >= w + g_R, so span <= F_2 + min(g_L, g_R).
    # Hence Q*_3 > F_2 + s_min forces min flank > s_min at the attaining window.
    hasQ3 = d["Q3"] >= 0
    gate(bool(np.all(d["Q3"][hasQ3] <= d["F2"][hasQ3] + d["Q3fl"][hasQ3])),
         "PEEL BOUND Q*_3 <= F_2 + (min flank at the argmax) at every member (%s)"
         % tag)
    hot = hasQ3 & (d["Q3"] > d["F2"] + smin)
    gate(bool(np.all(d["Q3fl"][hot] > smin[hot])),
         "A0-corrected: Q*_3 > F_2 + s_min forces min flank > s_min (%s)" % tag)

    pin = d["vqp"] == npin
    print("  full family:    violated by %d/%d = %.2f%%"
          % (int(viol.sum()), len(viol), 100.0 * viol.mean()))
    print("  PINNED v_q'=%d:  violated by %d/%d = %.2f%%"
          % (npin, int(viol[pin].sum()), int(pin.sum()), 100.0 * viol[pin].mean()))

    for name, sel in (("FULL", np.ones(len(arr), bool)), ("PINNED", pin)):
        v = viol[sel]
        n, nv = int(sel.sum()), int(v.sum())
        if nv == 0:
            print("  [%s] %d members, ZERO violators - nothing to characterise"
                  % (name, n))
            continue
        F = d["F"][sel]; Q3 = d["Q3"][sel]; F2 = d["F2"][sel]
        Aa = A[sel]; sm = smin[sel]; big = d["big"][sel].astype(bool)
        Q3mid = d["Q3mid"][sel]; jm = d["jmax"][sel]; Q3fl = d["Q3fl"][sel]
        rr = F % qp
        pcong = np.zeros(n, bool)
        for av in np.unique(Aa):
            m = Aa == av
            pcong[m] = np.isin(rr[m], [0, int(av), int(qp - av)])
        P3 = Q3 > F2 + sm
        print("  [%s] violators %d / %d = %.2f%%" % (name, nv, n, 100.0 * nv / n))
        print("     A1/A2 Pcong (F(M) mod q' in {0,A,B}):"
              "  sensitivity %.1f%%  PPV %.1f%%  specificity %.1f%%"
              "  (Pcong holds at %d/%d)"
              % (100.0 * (pcong & v).sum() / nv,
                 100.0 * (pcong & v).sum() / max(1, int(pcong.sum())),
                 100.0 * ((~pcong) & (~v)).sum() / max(1, n - nv),
                 int(pcong.sum()), n))
        print("     A4  P3 := Q*_3 > F_2 + s_min  vs violation:"
              "  agreement %.3f%%   (P3 & !viol %d ; viol & !P3 %d)"
              % (100.0 * (P3 == v).mean(), int((P3 & ~v).sum()),
                 int((v & ~P3).sum())))
        js, jc = np.unique(jm[v], return_counts=True)
        print("        attaining depth J among violators: "
              + " ".join("J=%d:%d" % (js[i], jc[i]) for i in range(len(js))))
        print("     A5  Pbig (a legal gap > s_min is realised):"
              "  sensitivity %.1f%%  PPV %.1f%%"
              % (100.0 * (big & v).sum() / nv,
                 100.0 * (big & v).sum() / max(1, int(big.sum()))))
        mv = Q3mid[v & P3]
        if mv.size:
            bb = qp - Aa[v & P3]
            smv = sm[v & P3]
            vals, cnts = np.unique(mv, return_counts=True)
            srt = np.argsort(-cnts)[:6]
            print("     A6  depth-3 attaining middles (value:count): "
                  + " ".join("%d:%d" % (vals[i], cnts[i]) for i in srt))
            big_letter = np.maximum(Aa[v & P3], bb)
            print("        share whose middle is the LARGER letter max(A,B) "
                  "or that + q': %.1f%%"
                  % (100.0 * np.isin(mv, np.unique(np.concatenate(
                      [big_letter, big_letter + qp]))).mean()))
            print("        share whose middle IS the old record F(M): %.1f%%"
                  % (100.0 * (mv == F[v & P3]).mean()))
            print("        share whose middle equals the minimal letter s_min: "
                  "%.1f%%" % (100.0 * (mv == smv).mean()))
        # SPECTRAL (congruence-free) predicates.  Constructor's spectrum-plus-
        # depth certificate says F(M+q') <= max_{2<=J<=Jmax} F_J(M), so
        #     Spec_J : max(F_2..F_J) <= F_2 + s_min
        # is a SUFFICIENT condition for no violation that uses no congruence at
        # all.  How much of the non-violating majority does it certify?
        F3 = d["F3"][sel]; F4 = d["F4"][sel]; F5 = d["F5"][sel]
        for jj, arrs in ((3, [F3]), (4, [F3, F4]), (5, [F3, F4, F5])):
            spec = np.maximum.reduce([F2] + arrs) <= F2 + sm
            bad = int((spec & v).sum())
            print("     SPEC_%d  max(F_2..F_%d) <= F_2 + s_min : holds at "
                  "%d/%d = %.1f%% of members, certifies %.1f%% of the "
                  "NON-violators, unsound at %d violators"
                  % (jj, jj, int(spec.sum()), n, 100.0 * spec.mean(),
                     100.0 * (spec & ~v).sum() / max(1, n - nv), bad))
        # A3: best predictor of the form "F(M) mod q' in S"
        cnt = {}
        for res in range(qp):
            m = rr == res
            if m.sum():
                cnt[res] = (int((m & v).sum()), int(m.sum()))
        order = sorted(cnt, key=lambda r_: -(cnt[r_][0] / cnt[r_][1]))
        chosen, bestba = [], 0.0
        for r_ in order:
            cand = chosen + [r_]
            pp = np.isin(rr, cand)
            ba = 0.5 * (100.0 * (pp & v).sum() / nv
                        + 100.0 * ((~pp) & (~v)).sum() / max(1, n - nv))
            if ba > bestba:
                bestba, chosen = ba, cand
        print("     A3  best 'F(M) mod q' in S' predictor: balanced accuracy "
              "%.1f%%  |S| = %d  S = %s" % (bestba, len(chosen), sorted(chosen)))
        carr = sorted(r_ for r_ in cnt if cnt[r_][0] > 0)
        print("        residues of F(M) mod q' carrying ANY violator: %s"
              % carr)
        print("        legal classes {0} u {A} u {B} over this selection: "
              "A in %s, B in %s"
              % (sorted(set(int(x) for x in np.unique(Aa))),
                 sorted(set(int(qp - x) for x in np.unique(Aa)))))
    return d, viol, pin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="small")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    twin = {q: min(pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in GEARS}
    gate([twin[q] for q in GEARS] == [1, 1, 2, 2, 3, 3, 4],
         "twin tooth vector is (1,1,2,2,3,3,4) = round(q/6)")

    import multiprocessing as mp
    if a.steps == "small":
        steps = [([5, 7], 11), ([5, 7, 11], 13), ([5, 7, 11, 13], 17),
                 ([5, 7, 11, 13, 17], 19)]
        for ogears, qp in steps:
            space = [list(range(1, (q - 1) // 2 + 1)) for q in ogears]
            nov = int(np.prod([len(s) for s in space]))
            chunks = []
            step = max(1, nov // (a.workers * 4))
            for lo in range(0, nov, step):
                chunks.append((ogears, qp, lo, min(nov, lo + step)))
            t0 = time.time()
            if a.workers > 1 and nov > 50:
                with mp.Pool(a.workers) as pool:
                    res = pool.map(work_small, chunks)
            else:
                res = [work_small(c) for c in chunks]
            rows = [r for part in res for r in part]
            arr = np.array(rows, dtype=np.int64)
            np.save(os.path.join(OUT, "resid_%d_%d.npy" % (ogears[-1], qp)), arr)
            print("\n[%d->%d] %d members in %.1f s" % (ogears[-1], qp, len(arr),
                                                       time.time() - t0))
            ovecs = list(itertools.product(*space))
            ti = ovecs.index(tuple(twin[q] for q in ogears))
            analyse("%d->%d" % (ogears[-1], qp), arr, ogears, qp, ti, twin[qp])
    elif a.steps == "19_23":
        ogears = [5, 7, 11, 13, 17, 19]
        nov = 12960
        step = max(1, nov // (a.workers * 8))
        chunks = [(lo, min(nov, lo + step), 4, False)
                  for lo in range(0, nov, step)]
        t0 = time.time()
        with mp.Pool(a.workers) as pool:
            res = pool.map(work_1923, chunks)
        rows = [r for part in res for r in part]
        arr = np.array(rows, dtype=np.int64)
        np.save(os.path.join(OUT, "resid_19_23.npy"), arr)
        print("\n[19->23 pinned] %d members in %.1f s" % (len(arr), time.time() - t0))
        # double-source gate against round 28's own table
        ref = np.load(os.path.join(os.path.dirname(OUT), "r28",
                                   "tooth_m23_pinned.npy"))
        ref = ref[np.argsort(ref[:, 0])]
        arr = arr[np.argsort(arr[:, 0])]
        ci = {c: i for i, c in enumerate(COLS)}
        gate(bool(np.array_equal(arr[:, ci["F"]], ref[:, 2])),
             "F(m19) agrees with round 28's tooth_m23_pinned.npy at all 12960 "
             "(independent re-sieve)")
        gate(bool(np.array_equal(arr[:, ci["F2"]], ref[:, 3])),
             "F_2(m19) agrees with round 28's table at all 12960")
        # F(m23) is TAKEN from round 28's gated table; a random sample is
        # re-derived here from scratch as the double-source check.
        arr[:, ci["Fn"]] = ref[:, 4]
        rng = np.random.default_rng(2929)
        samp = sorted(int(x) for x in rng.choice(12960, 400, replace=False))
        space0 = [list(range(1, (q - 1) // 2 + 1)) for q in ogears]
        ovecs0 = list(itertools.product(*space0))
        okn = 0
        for oi in samp:
            opx = sieve_openings(ogears, list(ovecs0[oi]), P19)
            rrx = (opx % 23).astype(np.int8)
            if m23_F(opx, rrx, 4) == int(ref[oi, 4]):
                okn += 1
        gate(okn == len(samp),
             "F(m23) re-derived from scratch agrees with round 28's table at "
             "%d/%d randomly sampled members" % (okn, len(samp)))
        space = [list(range(1, (q - 1) // 2 + 1)) for q in ogears]
        ovecs = list(itertools.product(*space))
        ti = ovecs.index(tuple(twin[q] for q in ogears))
        analyse("19->23", arr, ogears, 23, ti, 4)
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

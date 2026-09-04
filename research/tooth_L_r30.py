"""
LATERAL round 30, BLOCKS A and B - L(M) AND THE DEPTH-2 SLACK ON THE
TOOTH-COUNTERFACTUAL FAMILY.

Family V(y) = prod_{q<=y} {1..(q-1)/2}: gear q's teeth at +-v_q; gears, period P
and survivor count prod(q-2) fixed.  Step M -> q' has the FULL family
V(y) x {1..(q'-1)/2} (the new gear's tooth v_q' moves too) and the PINNED family
v_q' = round(q'/6), the twin value.

Per (member, v_q') this script computes, exactly:

  F, F_2, F_3          the old machine's spectrum (cyclic)
  L                    the length of the longest REALISED legal word: a run of
                       consecutive gaps of M, each = 0, a or b (mod q') with
                       a = 2v_q', b = q'-a, whose NONZERO classes strictly
                       alternate (padded letters transparent).  R89: J_max = L+2,
                       A_kill = L+1.
  the deepest word     its letters, and the co-deletability gate: the L+1
                       openings it spans lie in ONE two-class set mod q'
  n_a, n_b, n_0        how many gaps of M lie in each legal class
  Q*_J, J = 3..L+2     the word-legal spans (cross-checked against round 29)
  F(M+q')              at the small steps (direct sieve) and at the pinned
                       19->23 (round 28's gated table)

Block B needs only F, F_2 and q': slack(M; q') = F(M) + q' - F_2(M).

Usage:
  uv run python research/tooth_L_r30.py --steps small
  uv run python research/tooth_L_r30.py --steps 19_23 --workers 4 --max-chunks 16
  uv run python research/tooth_L_r30.py --steps 23_29 --workers 4 --sample 600 --max-chunks 8
  uv run python research/tooth_L_r30.py --report
Chunked runs write ONE FILE PER CHUNK FROM THE CHILD and resume from disk, so a
lost parent costs nothing; re-invoke until "pending 0".
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "r30")
R29 = os.path.join(HERE, "data", "r29")
R28 = os.path.join(HERE, "data", "r28")
TWIN = {5: 1, 7: 1, 11: 2, 13: 2, 17: 3, 19: 3, 23: 4, 29: 5}
WMAX = 7            # words up to length 7 (J up to 9); asserted never binding
NGATE = 0
COLS = ("oi vqp F F2 F3 L na nb n0 Qmax Q3 Fn capped "
        "w1 w2 w3 w4 w5 w6 w7").split()
CI = {c: i for i, c in enumerate(COLS)}


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def prodl(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def sieve_openings(gears, vs, P):
    blocked = np.zeros(P, dtype=bool)
    for q, v in zip(gears, vs):
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    return np.flatnonzero(~blocked).astype(np.int64)


def cyclic_gaps(op, P):
    g = np.empty(op.size, dtype=np.int64)
    g[:-1] = op[1:] - op[:-1]
    g[-1] = P - op[-1] + op[0]
    return g


def Fj_ladder(g, jmax=3):
    n = g.size
    ext = np.concatenate([g, g[:jmax + 1]])
    pre = np.zeros(ext.size + 1, dtype=np.int64)
    np.cumsum(ext, out=pre[1:])
    return [int((pre[j:n + j] - pre[:n]).max()) for j in range(1, jmax + 1)]


def legal_words(g, op, P, qp, vqp):
    """L, Q (Q[k-1] = Q*_{k+2}), deepest word, n_a, n_b, n_0, capped.

    Also GATES that the deepest word's L+1 openings are co-deletable: their
    residues mod q' lie in one two-class set {r, r +- a}.
    """
    a = (2 * vqp) % qp
    b = qp - a
    n = g.size
    ext = np.concatenate([g, g[:WMAX + 3]])
    r = ext % qp
    cls = np.zeros(ext.size, dtype=np.int8)
    cls[r == a] = 1
    cls[r == b] = -1
    legal = (r == 0) | (r == a) | (r == b)
    pre = np.zeros(ext.size + 1, dtype=np.int64)
    np.cumsum(ext, out=pre[1:])
    idx = np.arange(1, n + 1)
    idx = idx[legal[idx]]
    last = cls[idx]
    Q = np.full(WMAX, -1, dtype=np.int64)
    L = 0
    deep_t = -1
    for k in range(1, WMAX + 1):
        if k > 1:
            j = idx + (k - 1)
            cj = cls[j]
            ok = legal[j] & ((cj == 0) | (last == 0) | (last != cj))
            idx = idx[ok]
            last = np.where(cj[ok] != 0, cj[ok], last[ok])
        if idx.size == 0:
            break
        L = k
        span = ext[idx - 1] + (pre[idx + k] - pre[idx]) + ext[idx + k]
        am = int(np.argmax(span))
        Q[k - 1] = int(span[am])
        deep_t = int(idx[am])
    capped = (L == WMAX)
    word = [int(ext[deep_t + i]) for i in range(L)] if L > 0 else []
    n_a = int(np.count_nonzero(cls[:n] == 1))
    n_b = int(np.count_nonzero(cls[:n] == -1))
    n_0 = int(np.count_nonzero(r[:n] == 0))
    if L > 0:
        res = set()
        for i in range(L + 1):
            t = deep_t + i
            res.add(int((op[t % n] + P * (t // n)) % qp))
        if len(res) > 2:
            raise AssertionError("deepest word not co-deletable: %s" % res)
        if len(res) == 2:
            d = (max(res) - min(res)) % qp
            if d not in (a, b):
                raise AssertionError("two-class set is not {r, r+-a}: %s" % res)
        # every letter of the word is a legal class
        for w in word:
            if w % qp not in (0, a, b):
                raise AssertionError("illegal letter in deepest word")
    return L, Q, word, n_a, n_b, n_0, capped


def row(oi, vqp, F, F2, F3, Fn, res):
    L, Q, word, n_a, n_b, n_0, capped = res
    w = word + [-1] * (WMAX - len(word))
    return [oi, vqp, F, F2, F3, L, n_a, n_b, n_0, int(Q.max()), int(Q[0]),
            Fn, int(capped)] + w


def space_of(ogears):
    return [list(range(1, (q - 1) // 2 + 1)) for q in ogears]


def work_small(args):
    ogears, qp, lo, hi = args
    P = prodl(ogears)
    Pn = P * qp
    ovecs = list(itertools.product(*space_of(ogears)))
    rows = []
    for oi in range(lo, hi):
        ov = list(ovecs[oi])
        op = sieve_openings(ogears, ov, P)
        g = cyclic_gaps(op, P)
        F, F2, F3 = Fj_ladder(g, 3)
        for vqp in range(1, (qp - 1) // 2 + 1):
            res = legal_words(g, op, P, qp, vqp)
            opn = sieve_openings(ogears + [qp], ov + [vqp], Pn)
            if opn.size != prodl([q - 2 for q in ogears + [qp]]):
                raise AssertionError("new-machine opening count moved")
            Fn = int(cyclic_gaps(opn, Pn).max())
            rows.append(row(oi, vqp, F, F2, F3, Fn, res))
    return rows


def work_chunk(args):
    """FULL family at one step, chunk [lo, hi) of old tooth vectors, every
    v_q'; result written FROM THE CHILD to its own file."""
    ogears, qp, lo, hi, oi_list, fname = args
    if os.path.exists(fname):
        return fname
    P = prodl(ogears)
    ovecs = list(itertools.product(*space_of(ogears)))
    rows = []
    t0 = time.time()
    sel = oi_list if oi_list is not None else range(lo, hi)
    for oi in sel:
        ov = list(ovecs[oi])
        op = sieve_openings(ogears, ov, P)
        if op.size != prodl([q - 2 for q in ogears]):
            raise AssertionError("opening count moved")
        g = cyclic_gaps(op, P)
        F, F2, F3 = Fj_ladder(g, 3)
        for vqp in range(1, (qp - 1) // 2 + 1):
            res = legal_words(g, op, P, qp, vqp)
            rows.append(row(oi, vqp, F, F2, F3, -1, res))
        del op, g
    arr = np.array(rows, dtype=np.int64)
    np.save(fname + ".tmp.npy", arr)
    os.replace(fname + ".tmp.npy", fname)
    with open(fname + ".log", "w") as fh:
        fh.write("chunk %s done: %d rows in %.1f s\n"
                 % (os.path.basename(fname), len(rows), time.time() - t0))
    return fname


# ---------------------------------------------------------------- analysis

def rank_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rank_avg(x), rank_avg(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def eta2(vals, levels):
    vals = np.asarray(vals, dtype=np.float64)
    tot = ((vals - vals.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    between = 0.0
    for lv in np.unique(levels):
        m = levels == lv
        between += m.sum() * (vals[m].mean() - vals.mean()) ** 2
    return float(between / tot)


def pct(fam, x):
    fam = np.asarray(fam)
    return 100.0 * np.mean(fam < x), 100.0 * np.mean(fam <= x)


def analyse(tag, arr, ogears, qp, full, sample_note=""):
    d = {c: arr[:, i] for i, c in enumerate(COLS)}
    y = ogears[-1]
    space = space_of(ogears)
    ovecs = list(itertools.product(*space))
    twin_ov = tuple(TWIN[q] for q in ogears)
    ti = ovecs.index(twin_ov)
    npin = TWIN[qp]
    a_of = (2 * d["vqp"]) % qp
    b_of = qp - a_of
    print("")
    print("===== STEP {5..%d} -> %d : %d rows (%s family%s) ====="
          % (y, qp, len(arr), "FULL" if full else "PINNED", sample_note))
    gate(not bool(np.any(d["capped"])),
         "%s: no word reached the length cap %d" % (tag, WMAX))
    pin = d["vqp"] == npin
    real = (d["oi"] == ti) & pin
    if not sample_note:
        gate(int(real.sum()) == 1, "%s: the real machine is in the table once" % tag)
    have_real = int(real.sum()) == 1
    L = d["L"]
    # ---------------- BLOCK A: L ----------------
    print("  [A] L histogram, FULL family:   "
          + "  ".join("L=%d:%d" % (k, int((L == k).sum()))
                      for k in range(0, int(L.max()) + 1)))
    print("      L histogram, PINNED v_q'=%d: " % npin
          + "  ".join("L=%d:%d" % (k, int((L[pin] == k).sum()))
                      for k in range(0, int(L.max()) + 1)))
    Lmax = int(L.max())
    print("      max L over the %s family = %d  (attained by %d rows; pinned max %d)"
          % ("full" if full else "pinned", Lmax, int((L == Lmax).sum()),
             int(L[pin].max())))
    if have_real:
        Lr = int(L[real][0])
        lo_, hi_ = pct(L[pin], Lr)
        print("      REAL machine: L = %d, F = %d, F_2 = %d;  pinned-family "
              "percentile: %.1f%% strictly below, %.1f%% at or below"
              % (Lr, int(d["F"][real][0]), int(d["F2"][real][0]), lo_, hi_))
    print("      fraction with L >= 3: full %.2f%%   pinned %.2f%%"
          % (100.0 * np.mean(L >= 3), 100.0 * np.mean(L[pin] >= 3)))
    # eta^2 by gear (full family only makes sense with v_q' moving)
    ov_arr = np.array([ovecs[int(o)] for o in d["oi"]], dtype=np.int64)
    e2 = {}
    for gi, q in enumerate(ogears):
        e2[q] = eta2(L, ov_arr[:, gi])
    if full:
        e2[qp] = eta2(L, d["vqp"])
    print("      eta^2 of L by gear tooth: "
          + "  ".join("%d:%.3f" % (q, e2[q]) for q in sorted(e2)))
    if full:
        best_old = max(e2[q] for q in ogears)
        print("      -> new gear %d: %.3f  vs best old gear %.3f  (%s)"
              % (qp, e2[qp], best_old,
                 "NEW GEAR LARGER" if e2[qp] > best_old else "an old gear is larger"))
        # mean L by v_q'
        means = []
        for v in range(1, (qp - 1) // 2 + 1):
            m = d["vqp"] == v
            means.append((v, float(L[m].mean()), float(np.mean(L[m] >= 3)),
                          int(L[m].max())))
        print("      mean L by v_q' (v: a,b | mean L | P(L>=3) | max L):")
        for v, mL, p3, mx in means:
            print("         v=%2d: a=%2d b=%2d | %.3f | %.4f | %d%s"
                  % (v, (2 * v) % qp, qp - (2 * v) % qp, mL, p3, mx,
                     "   <- TWIN" if v == npin else ""))
        argmin_v = min(means, key=lambda t: t[1])[0]
        argmax_v = max(means, key=lambda t: t[1])[0]
        print("      argmin_v mean L = %d (%s), argmax_v = %d; twin v = %d"
              % (argmin_v,
                 "EXTREME" if argmin_v in (1, (qp - 1) // 2) else "interior",
                 argmax_v, npin))
    # mechanism: min(n_a, n_b)
    mn = np.minimum(d["na"], d["nb"])
    rho_full = spearman(L, mn)
    rho_pin = spearman(L[pin], mn[pin])
    rho_pin_nab = spearman(L[pin], d["na"][pin] * d["nb"][pin])
    print("      spearman(L, min(n_a,n_b)): full %.3f   pinned %.3f ;  "
          "spearman(L, n_a*n_b) pinned %.3f" % (rho_full, rho_pin, rho_pin_nab))
    if have_real:
        mr = int(mn[real][0])
        lo_, hi_ = pct(mn[pin], mr)
        print("      REAL min(n_a,n_b) = %d (n_a=%d n_b=%d n_0=%d); pinned "
              "percentile %.1f%% below, %.1f%% at or below"
              % (mr, int(d["na"][real][0]), int(d["nb"][real][0]),
                 int(d["n0"][real][0]), lo_, hi_))
    # deepest words at max L: literal or padded?
    wcols = [d["w%d" % i] for i in range(1, WMAX + 1)]
    for sel_name, sel in (("full", np.ones(len(arr), bool)), ("pinned", pin)):
        m = sel & (L == L[sel].max())
        Lm = int(L[sel].max())
        padded = np.zeros(len(arr), bool)
        for i in range(Lm):
            padded |= (wcols[i] % qp == 0) & m
        print("      [%s] at max L = %d: %d rows, %d with a padded letter, %d literal"
              % (sel_name, Lm, int(m.sum()), int((padded & m).sum()),
                 int((m & ~padded).sum())))
        shown = 0
        for ridx in np.flatnonzero(m):
            if shown >= 6:
                break
            wd = [int(wcols[i][ridx]) for i in range(Lm)]
            print("         oi=%5d teeth=%s v_q'=%d (a=%d,b=%d) F=%d F_2=%d "
                  "word=%s Q*_%d=%d"
                  % (int(d["oi"][ridx]), ovecs[int(d["oi"][ridx])],
                     int(d["vqp"][ridx]), int(a_of[ridx]), int(b_of[ridx]),
                     int(d["F"][ridx]), int(d["F2"][ridx]), wd, Lm + 2,
                     int(d["Qmax"][ridx])))
            shown += 1
    # ---------------- BLOCK B: depth-2 slack ----------------
    # slack depends on the OLD machine and q' only: one row per oi
    _, first = np.unique(d["oi"], return_index=True)
    F1 = d["F"][first]
    F21 = d["F2"][first]
    oi1 = d["oi"][first]
    slack = F1 + qp - F21
    exc = F21 - F1
    print("  [B] depth-2 slack F + q' - F_2 over V(%d) (%d members%s): min %d, "
          "median %.1f, max %d;  F_2 - F: max %d, median %.1f"
          % (y, len(slack), sample_note, int(slack.min()),
             float(np.median(slack)), int(slack.max()), int(exc.max()),
             float(np.median(exc))))
    nz = int((slack <= 0).sum())
    print("      members with slack <= 0 (depth-2 half FAILS): %d / %d = %.3f%%"
          % (nz, len(slack), 100.0 * nz / len(slack)))
    if have_real:
        sr = int(slack[oi1 == ti][0])
        lo_, hi_ = pct(slack, sr)
        print("      REAL slack = %d (F=%d, q'=%d, F_2=%d); percentile %.1f%% "
              "strictly below, %.1f%% at or below"
              % (sr, int(F1[oi1 == ti][0]), qp, int(F21[oi1 == ti][0]), lo_, hi_))
    order = np.argsort(slack)
    print("      the %d smallest-slack members:" % min(5, len(order)))
    for j in order[:5]:
        o = int(oi1[j])
        print("         oi=%5d teeth=%s F=%d F_2=%d slack=%d"
              % (o, ovecs[o], int(F1[j]), int(F21[j]), int(slack[j])))
    # B5: attainment at depth 2.  F(M+q') is DIRECT (sieve / round 28's gated
    # table) where Fn >= 0; elsewhere it is taken from the RECORD LAW
    # F(M+q') = max(F_2, max_J Q*_J) = max over two-class runs of
    # (before + span + after), structural (R68; gated at 27,570 members in
    # round 29 and at every direct row here).
    Fn = d["Fn"]
    known = Fn >= 0
    law = np.maximum(d["F2"], d["Qmax"])
    if known.any():
        gate(bool(np.all(law[known] == Fn[known])),
             "%s: ATTAINMENT / RECORD LAW max(F_2, max_J Q*_J) == F(M+q') at all "
             "%d rows with a DIRECT F(M+q')" % (tag, int(known.sum())))
        gate(bool(np.all(Fn[known] >= d["F2"][known])),
             "%s: F(M+q') >= F_2(M) at all direct rows (so slack <= 0 => (D) fails)"
             % tag)
    src = "direct" if known.all() else ("record-law at %d of %d rows"
                                        % (int((~known).sum()), len(Fn)))
    Fn = np.where(known, Fn, law)
    s_row = d["F"] + qp - d["F2"]
    m0 = s_row <= 0
    if m0.any():
        at2 = int((Fn[m0] == d["F2"][m0]).sum())
        print("      [B5] among %d (member, v_q') rows with slack <= 0: "
              "F(M+q') == F_2(M) at %d (%.1f%%)   [F(M+q') %s]"
              % (int(m0.sum()), at2, 100.0 * at2 / m0.sum(), src))
    dfail = Fn > d["F"] + qp
    d2fail = d["F2"] > d["F"] + qp
    print("      (D) fails at %d / %d rows (%.2f%%); of these the depth-2 half "
          "fails at %d (%.1f%% of the (D) failures)   [F(M+q') %s]"
          % (int(dfail.sum()), len(Fn), 100.0 * dfail.sum() / len(Fn),
             int(d2fail.sum()), 100.0 * d2fail.sum() / max(1, dfail.sum()), src))
    dfp = dfail[pin]
    print("      pinned v_q'=%d: (D) fails at %d / %d (%.2f%%), depth-2 half at %d"
          % (npin, int(dfp.sum()), int(pin.sum()), 100.0 * dfp.mean(),
             int(d2fail[pin].sum())))
    return d


def cross_check_r29(tag, arr, fname, pinned_only=False):
    """Q*_3 and max_J Q*_J against round 29's independent table."""
    ref = np.load(fname)
    # r29 COLS: oi vqp F F2 F3 F4 F5 Fn Q3 Q3mid Q3fl Qmax jmax capped big nopen
    key_ref = {(int(r[0]), int(r[1])): (int(r[2]), int(r[3]), int(r[8]), int(r[11]))
               for r in ref}
    n = 0
    bad = 0
    for r in arr:
        k = (int(r[CI["oi"]]), int(r[CI["vqp"]]))
        if k not in key_ref:
            continue
        n += 1
        F, F2, Q3, Qm = key_ref[k]
        if (F, F2, Q3, Qm) != (int(r[CI["F"]]), int(r[CI["F2"]]),
                               int(r[CI["Q3"]]), int(r[CI["Qmax"]])):
            bad += 1
    gate(n > 0 and bad == 0,
         "%s: F, F_2, Q*_3, max_J Q*_J agree with round 29's table at all %d "
         "shared rows (%d mismatches)" % (tag, n, bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="report")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=3030)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gate([min(pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in TWIN]
         == [TWIN[q] for q in TWIN],
         "twin tooth vector is round(q/6) = %s" % [TWIN[q] for q in TWIN])
    import multiprocessing as mp

    if a.steps == "small":
        steps = [([5, 7], 11), ([5, 7, 11], 13), ([5, 7, 11, 13], 17),
                 ([5, 7, 11, 13, 17], 19)]
        for ogears, qp in steps:
            nov = prodl([len(s) for s in space_of(ogears)])
            step = max(1, nov // (a.workers * 4))
            chunks = [(ogears, qp, lo, min(nov, lo + step))
                      for lo in range(0, nov, step)]
            t0 = time.time()
            if a.workers > 1 and nov > 50:
                with mp.Pool(a.workers) as pool:
                    res = pool.map(work_small, chunks)
            else:
                res = [work_small(c) for c in chunks]
            arr = np.array([r for part in res for r in part], dtype=np.int64)
            fn = os.path.join(OUT, "L_%d_%d.npy" % (ogears[-1], qp))
            np.save(fn, arr)
            print("\n[%d->%d] %d rows in %.1f s -> %s"
                  % (ogears[-1], qp, len(arr), time.time() - t0, fn))
        return 0

    if a.steps == "19_23":
        ogears, qp = [5, 7, 11, 13, 17, 19], 23
        nov = 12960
        step = 90
        pending = []
        for lo in range(0, nov, step):
            fn = os.path.join(OUT, "L_19_23_c%05d.npy" % lo)
            if not os.path.exists(fn):
                pending.append((ogears, qp, lo, min(nov, lo + step), None, fn))
        print("[19->23 FULL] chunks pending: %d (running up to %d now, %d workers)"
              % (len(pending), a.max_chunks, a.workers))
        todo = pending[:a.max_chunks]
        t0 = time.time()
        if todo:
            with mp.Pool(a.workers) as pool:
                for fn in pool.imap_unordered(work_chunk, todo):
                    print("   done %s  (%.0f s elapsed)"
                          % (os.path.basename(fn), time.time() - t0), flush=True)
        left = len(pending) - len(todo)
        print("[19->23 FULL] pending %d" % left)
        return 0

    if a.steps == "23_29":
        ogears, qp = [5, 7, 11, 13, 17, 19, 23], 29
        nov = 142560
        rng = np.random.default_rng(a.seed)
        ovecs_ti = list(itertools.product(*space_of(ogears))).index(
            tuple(TWIN[q] for q in ogears))
        samp = sorted(set(int(x) for x in rng.choice(nov, a.sample, replace=False))
                      | {ovecs_ti})
        with open(os.path.join(OUT, "sample_23_29.txt"), "w") as fh:
            fh.write("seed %d sample %d + twin oi %d\n" % (a.seed, a.sample, ovecs_ti))
            fh.write(" ".join(str(x) for x in samp) + "\n")
        per = 10
        pending = []
        for ci in range(0, len(samp), per):
            fn = os.path.join(OUT, "L_23_29_s%04d.npy" % ci)
            if not os.path.exists(fn):
                pending.append((ogears, qp, 0, 0, samp[ci:ci + per], fn))
        print("[23->29 SAMPLE %d] chunks pending: %d (running up to %d now, %d workers)"
              % (len(samp), len(pending), a.max_chunks, a.workers))
        todo = pending[:a.max_chunks]
        t0 = time.time()
        if todo:
            with mp.Pool(a.workers) as pool:
                for fn in pool.imap_unordered(work_chunk, todo):
                    print("   done %s  (%.0f s elapsed)"
                          % (os.path.basename(fn), time.time() - t0), flush=True)
        print("[23->29 SAMPLE] pending %d" % (len(pending) - len(todo)))
        return 0

    # ------------------------------------------------------------ REPORT
    steps = [([5, 7], 11), ([5, 7, 11], 13), ([5, 7, 11, 13], 17),
             ([5, 7, 11, 13, 17], 19)]
    realL = {}
    for ogears, qp in steps:
        arr = np.load(os.path.join(OUT, "L_%d_%d.npy" % (ogears[-1], qp)))
        tag = "%d->%d" % (ogears[-1], qp)
        cross_check_r29(tag, arr, os.path.join(R29, "resid_%d_%d.npy"
                                               % (ogears[-1], qp)))
        d = analyse(tag, arr, ogears, qp, full=True)
        ovecs = list(itertools.product(*space_of(ogears)))
        ti = ovecs.index(tuple(TWIN[q] for q in ogears))
        real = (d["oi"] == ti) & (d["vqp"] == TWIN[qp])
        realL[tag] = (int(d["L"][real][0]), int(d["F"][real][0]),
                      int(d["F2"][real][0]))
    # 19->23 FULL, from the chunk files
    files = sorted(f for f in os.listdir(OUT)
                   if f.startswith("L_19_23_c") and f.endswith(".npy"))
    if files:
        arr = np.concatenate([np.load(os.path.join(OUT, f)) for f in files])
        full = (len(np.unique(arr[:, 0])) == 12960)
        print("\n[19->23] %d chunk files, %d rows, %d distinct old members%s"
              % (len(files), len(arr), len(np.unique(arr[:, 0])),
                 "" if full else "  (PARTIAL)"))
        cross_check_r29("19->23", arr, os.path.join(R29, "resid_19_23.npy"))
        # F(m23) at the pinned rows from round 28's gated table
        ref = np.load(os.path.join(R28, "tooth_m23_pinned.npy"))
        fn_of = {int(r[0]): int(r[4]) for r in ref}
        f_of = {int(r[0]): (int(r[2]), int(r[3])) for r in ref}
        pinm = arr[:, CI["vqp"]] == 4
        ok = all(f_of[int(r[CI["oi"]])] == (int(r[CI["F"]]), int(r[CI["F2"]]))
                 for r in arr[pinm])
        gate(ok, "19->23: F(m19), F_2(m19) agree with round 28's table at all "
                 "%d pinned rows" % int(pinm.sum()))
        arr[pinm, CI["Fn"]] = [fn_of[int(o)] for o in arr[pinm, CI["oi"]]]
        d = analyse("19->23", arr, [5, 7, 11, 13, 17, 19], 23, full=True,
                    sample_note="" if full else ", PARTIAL")
        ovecs = list(itertools.product(*space_of([5, 7, 11, 13, 17, 19])))
        ti = ovecs.index((1, 1, 2, 2, 3, 3))
        real = (d["oi"] == ti) & (d["vqp"] == 4)
        if real.any():
            realL["19->23"] = (int(d["L"][real][0]), int(d["F"][real][0]),
                               int(d["F2"][real][0]))
    files = sorted(f for f in os.listdir(OUT)
                   if f.startswith("L_23_29_s") and f.endswith(".npy"))
    if files:
        arr = np.concatenate([np.load(os.path.join(OUT, f)) for f in files])
        print("\n[23->29] SAMPLE: %d chunk files, %d rows, %d distinct old members "
              "of 142,560" % (len(files), len(arr), len(np.unique(arr[:, 0]))))
        d = analyse("23->29", arr, [5, 7, 11, 13, 17, 19, 23], 29, full=True,
                    sample_note=", SAMPLE of V(23)")
        ovecs = list(itertools.product(*space_of([5, 7, 11, 13, 17, 19, 23])))
        ti = ovecs.index((1, 1, 2, 2, 3, 3, 4))
        real = (d["oi"] == ti) & (d["vqp"] == 5)
        if real.any():
            realL["23->29"] = (int(d["L"][real][0]), int(d["F"][real][0]),
                               int(d["F2"][real][0]))
    print("\nREAL MACHINE (L, F, F_2) by step: %s" % realL)
    exp = {"11->13": (1, 7, 11), "13->17": (1, 11, 16), "17->19": (1, 18, 25),
           "19->23": (2, 25, 31), "23->29": (1, 34, 39)}
    for k, v in exp.items():
        if k in realL:
            gate(realL[k] == v, "real machine at %s has (L, F, F_2) = %s "
                                "(corpus L row 1,1,1,2,1; F 7,11,18,25,34)"
                 % (k, v))
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

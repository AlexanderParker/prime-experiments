"""r69 / new gears lengthen, never precede -- part 2: the prefix ladder q = 5 .. 19997.

For every rung q -> q' (q' the next prime, q'' the one after):
  * E6 by DIRECT sieve (q <= QDIRECT): blocked({5..q'}) \\ blocked({5..q}) on [1, W(q)] against the
    predicted set {d_0(M) if (q', q'') twin} u {W(q) if q'^2 - 2 prime};
  * the d_0 ladder law: d_0(M + q') = d_1(M) iff (q', q'') twin, else d_0(M);
  * the prefix frontier of M on [1, W(q)] and of M + q' on [1, W(q')] (reduction (R): openings are
    the twin columns with 6k - 1 > rung, plus the top column when rung'^2 - 2 is prime);
  * c_pre(q) = min x/L over non-initial runs with L >= d_0, its minimiser; c_any(q) over all
    non-initial runs; Form B on the prefix with every exception classified (absorbed / straddle /
    section / other); the straddling run (the run of M + q' containing W(q)); the interior section
    runs' minimum ratio against the bound W(q)/S(q');
  * the transitions of c_pre and which run sets a decrease;
  * V8's hidden hypothesis d_0(y_m(Q)) <= klo_m(Q) for Q <= 10^5, m = 1..4.

Self-contained, numpy only.  Peak memory about 800 MB.
Run: uv run python research/anchor235/r69/lnp_prefix.py [QMAX] [QDIRECT]
"""
import json
import os
import sys
import time

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 19997
QDIRECT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
C_FLOOR = 4.625


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def direct_blocked(gears, top):
    """blocked columns of the machine over [0, top] by the gears' teeth."""
    b = np.zeros(top + 1, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        b[u::g] = True
        b[(g - u) % g::g] = True
    return b


def runs_from_openings(ops, top):
    """maximal blocked runs (start, length) of a machine whose openings in [1, top] are ops
    (sorted, may be empty); the run touching top is truncated at top."""
    ops = np.asarray(ops, dtype=np.int64)
    bounds = np.concatenate(([0], ops, [top + 1]))
    starts = bounds[:-1] + 1
    lens = bounds[1:] - bounds[:-1] - 1
    m = lens > 0
    return starts[m], lens[m]


def pareto(starts, lens):
    if starts.size == 0:
        return starts, lens
    rmv = np.maximum.accumulate(lens)
    keep = np.empty(lens.size, dtype=bool)
    keep[0] = True
    keep[1:] = rmv[1:] > rmv[:-1]
    return starts[keep], lens[keep]


def rmin_fn(px, pl):
    def f(L):
        i = int(np.searchsorted(pl, L, side="left"))
        return int(px[i]) if i < pl.size else None
    return f


def main():
    t0 = time.time()
    lim = QMAX + 500
    sm = sieve(lim)
    small_primes = np.flatnonzero(sm).tolist()
    rungs = [p for p in small_primes if 5 <= p <= QMAX]
    nxt = {}
    for i, p in enumerate(small_primes[:-2]):
        nxt[p] = (small_primes[i + 1], small_primes[i + 2])
    q_top = rungs[-1]
    qpp_top = nxt[nxt[q_top][0]][0]
    NMAX = qpp_top * qpp_top + 10
    print("sieving to %d" % NMAX, flush=True)
    isp = sieve(NMAX)
    KMAX = (qpp_top * qpp_top - 1) // 6
    CH = 1 << 22
    parts = []
    for a in range(1, KMAX + 1, CH):
        bnd = min(a + CH, KMAX + 1)
        k = np.arange(a, bnd, dtype=np.int64)
        t = isp[6 * k - 1] & isp[6 * k + 1]
        parts.append(k[t])
    twincols = np.concatenate(parts)
    del parts
    print("twin columns to %d: %d  (%.0f s)" % (KMAX, twincols.size, time.time() - t0), flush=True)

    def openings(rung, top, rung_next):
        """openings of {5..rung} in [1, top] by (R); top = W(rung) = (rung_next^2-1)/6."""
        lo = int(np.searchsorted(twincols, (rung + 2) // 6))
        while lo < twincols.size and 6 * int(twincols[lo]) - 1 <= rung:
            lo += 1
        hi = int(np.searchsorted(twincols, top, side="right"))
        o = twincols[lo:hi]
        if isp[rung_next * rung_next - 2] and (o.size == 0 or int(o[-1]) != top):
            o = np.concatenate((o, [top]))
        return o

    rows = []
    e6_direct = []          # (q, ok, found_set, predicted_set)
    d0_law_exc = 0
    formB_exc = {"absorbed": 0, "straddle": 0, "section": 0, "other": 0}
    formB_cells = 0
    formB_other_examples = []
    formA_exc = {"straddle": 0, "section": 0, "other": 0}
    formA_cells = 0
    lines = []
    Wr = lines.append
    prev = None
    trans = []              # transitions of c_pre
    strad_min = None
    sec_viol = []           # rungs q' where W(q)/S(q') < C_FLOOR
    sec_min_ratio_viol = [] # rungs where an interior section run has ratio < C_FLOOR
    floor_viol = []         # rungs with c_pre < C_FLOOR (q >= 23)
    minimiser_111 = []
    minimiser_13 = []
    n_finite = 0
    for q in rungs:
        qp, qpp = nxt[q]
        Wq = (qp * qp - 1) // 6
        Wqp = (qpp * qpp - 1) // 6
        twin_next = isp[qp + 2] and (qp % 6 == 5)
        rider = bool(isp[qp * qp - 2])
        ops_M = openings(q, Wq, qp)
        ops_N = openings(qp, Wqp, qpp)
        # d_0, d_1
        d0M = int(ops_M[0]) if ops_M.size else None
        d1M = int(ops_M[1]) if ops_M.size > 1 else None
        d0N = int(ops_N[0]) if ops_N.size else None
        h = (qp + 1) // 6 if qp % 6 == 5 else (qp - 1) // 6
        pred = set()
        if twin_next:
            pred.add(d0M)
        if rider:
            pred.add(Wq)
        # d_0 law
        expect_d0N = d1M if twin_next else d0M
        if d0N != expect_d0N:
            d0_law_exc += 1
        # E6 by direct sieve
        if q <= QDIRECT:
            gM = [p for p in small_primes if 5 <= p <= q]
            bM = direct_blocked(gM, Wq)
            bN = direct_blocked(gM + [qp], Wq)
            new = np.flatnonzero(bN & ~bM)
            new = set(int(x) for x in new if x >= 1)
            e6_direct.append((q, new == pred, sorted(new), sorted(pred)))
        # runs
        sM, lM = runs_from_openings(ops_M, Wq)
        sN, lN = runs_from_openings(ops_N, Wqp)
        FpreM = int(lM.max()) if lM.size else 0
        # non-initial runs of M with L >= d0
        niM = sM > 1
        selM = niM & (lM >= (d0M if d0M else 10 ** 12))
        if selM.any():
            r = sM[selM] / lM[selM]
            i = int(np.argmin(r))
            c_pre = float(r[i]); cmin = (int(sM[selM][i]), int(lM[selM][i]))
            n_finite += 1
        else:
            c_pre = None; cmin = None
        if niM.any():
            r = sM[niM] / lM[niM]
            i = int(np.argmin(r))
            c_any = float(r[i]); amin = (int(sM[niM][i]), int(lM[niM][i]))
        else:
            c_any = None; amin = None
        # top run of M (truncated at Wq)
        if sM.size and sM[-1] + lM[-1] - 1 == Wq:
            x_top, L_top = int(sM[-1]), int(lM[-1])
        else:
            x_top, L_top = None, 0
        # straddling run of N: the run containing Wq
        j = int(np.searchsorted(sN, Wq, side="right")) - 1
        assert j >= 0 and sN[j] <= Wq < sN[j] + lN[j], (q, Wq)
        x_s, L_s = int(sN[j]), int(lN[j])
        strad_initial = (x_s == 1)
        strad_ratio = x_s / L_s
        # interior section runs
        sec = sN > Wq
        S = Wqp - Wq
        bound = Wq / S
        if sec.any():
            rs = sN[sec] / lN[sec]
            i = int(np.argmin(rs))
            sec_ratio = float(rs[i]); secmin = (int(sN[sec][i]), int(lN[sec][i]))
        else:
            sec_ratio = None; secmin = None
        if bound < C_FLOOR:
            sec_viol.append(qp)
        if sec_ratio is not None and sec_ratio < C_FLOOR:
            sec_min_ratio_viol.append((qp, secmin))
        # Form B / A on the prefix
        pxM, plM = pareto(sM, lM)
        pxN, plN = pareto(sN, lN)
        RM, RN = rmin_fn(pxM, plM), rmin_fn(pxN, plN)
        if d0M is not None and FpreM >= d0M:
            for L in range(d0M, FpreM + 1):
                a, b = RM(L), RN(L)
                formB_cells += 1
                if b is not None and b < a:
                    if b == 1:
                        formB_exc["absorbed"] += 1
                    elif b <= Wq < b + lN[int(np.searchsorted(sN, b))] :
                        formB_exc["straddle"] += 1
                    elif b > Wq:
                        formB_exc["section"] += 1
                    else:
                        formB_exc["other"] += 1
                        if len(formB_other_examples) < 10:
                            formB_other_examples.append((q, L, a, b))
                if d0N is not None and L >= d0N:
                    formA_cells += 1
                    if b is not None and b < a:
                        if b <= Wq < b + lN[int(np.searchsorted(sN, b))]:
                            formA_exc["straddle"] += 1
                        elif b > Wq:
                            formA_exc["section"] += 1
                        else:
                            formA_exc["other"] += 1
        # c_pre of N (next rung) computed when we get there; record transition later
        row = dict(q=q, qp=qp, qpp=qpp, Wq=Wq, Wqp=Wqp, twin_next=bool(twin_next), rider=rider,
                   h=h, d0M=d0M, d1M=d1M, d0N=d0N, FpreM=FpreM, c_pre=c_pre, cmin=cmin,
                   c_any=c_any, amin=amin, x_top=x_top, L_top=L_top, x_s=x_s, L_s=L_s,
                   strad_initial=strad_initial, strad_ratio=strad_ratio, S=S, bound=bound,
                   sec_ratio=sec_ratio, secmin=secmin)
        rows.append(row)
        if q >= 23 and c_pre is not None and c_pre < C_FLOOR:
            floor_viol.append((q, c_pre, cmin))
        if cmin == (111, 24):
            minimiser_111.append(q)
        if cmin == (13, 4):
            minimiser_13.append(q)
        if qp >= 23 and not strad_initial:
            if strad_min is None or strad_ratio < strad_min[0]:
                strad_min = (strad_ratio, qp, x_s, L_s)
    # transitions of c_pre along the ladder
    dec = []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a["c_pre"] is not None and b["c_pre"] is not None and b["c_pre"] < a["c_pre"]:
            # which run sets b's minimum: b['cmin'] = (x, L) in the prefix of rung b['q'] = a['qp']
            x, L = b["cmin"]
            Wq = a["Wq"]
            if x <= Wq < x + L - 1:
                kind = "straddle"
            elif x > Wq:
                kind = "section"
            elif x + L - 1 < Wq:
                kind = "inherited"
            else:
                kind = "other"
            dec.append((a["q"], a["qp"], a["c_pre"], b["c_pre"], x, L, kind))
    # V8 hidden hypothesis
    v8_exc = []
    v8_cells = 0
    QV = 100000
    isp_small = sm  # primes to lim
    plist = np.flatnonzero(sieve(int((5 * QV + 2) ** 0.5) + 10))
    for m in (1, 2, 3, 4):
        for Q in range(3, QV + 1):
            top = (m + 1) * Q + 2
            y = int(plist[int(np.searchsorted(plist, int(top ** 0.5) + 1, side="right")) - 1])
            while y * y > top:
                y = int(plist[int(np.searchsorted(plist, y)) - 1])
            if y < 5:
                continue
            klo = -(-(m * Q + 2) // 6)
            lo = int(np.searchsorted(twincols, (y + 2) // 6))
            while 6 * int(twincols[lo]) - 1 <= y:
                lo += 1
            d0y = int(twincols[lo])
            v8_cells += 1
            if d0y > klo:
                v8_exc.append((Q, m, y, d0y, klo))

    # ---- report
    Wr("rungs q = %d..%d (%d rungs); twin sieve to %d; %.0f s" % (rungs[0], rungs[-1], len(rungs), NMAX, time.time() - t0))
    Wr("")
    Wr("E6 by direct sieve on [1, W(q)], q <= %d: %d rungs, %d discrepancies" % (QDIRECT, len(e6_direct), sum(1 for e in e6_direct if not e[1])))
    for e in e6_direct:
        if not e[1]:
            Wr("   DISCREPANCY q=%d found=%s predicted=%s" % (e[0], e[2], e[3]))
    n_tw = sum(1 for r in rows if r["twin_next"]); n_rd = sum(1 for r in rows if r["rider"])
    n_both = sum(1 for r in rows if r["twin_next"] and r["rider"]); n_none = sum(1 for r in rows if not r["twin_next"] and not r["rider"])
    Wr("   new columns per rung: twin (d_0 absorbed) at %d rungs, square (rider) at %d, both at %d, none at %d, of %d" % (n_tw, n_rd, n_both, n_none, len(rows)))
    Wr("d_0 ladder law d_0(M+q') = d_1(M) iff (q',q'') twin else d_0(M): %d exceptions in %d rungs" % (d0_law_exc, len(rows)))
    Wr("home column h(q') <= d_0(M) at every rung: %s; equality exactly at twin rungs: %s" % (
        all(r["h"] <= r["d0M"] for r in rows), all((r["h"] == r["d0M"]) == r["twin_next"] for r in rows)))
    Wr("")
    Wr("FORM B on the prefix (L in [d_0(M), F_pre(M)], R_min over [1,W(q)] vs [1,W(q')]): %d cells; exceptions %s" % (formB_cells, formB_exc))
    Wr("   'other' examples: %s" % formB_other_examples)
    Wr("FORM A on the prefix (L >= d_0(M+q') too): %d cells; exceptions %s" % (formA_cells, formA_exc))
    Wr("")
    Wr("c_pre finite (some non-initial run with L >= d_0) at %d rungs" % n_finite)
    Wr("floor: rungs q >= 23 with c_pre < %.3f: %d %s" % (C_FLOOR, len(floor_viol), floor_viol[:10]))
    Wr("minimiser (111,24) at rungs: %s" % minimiser_111)
    Wr("minimiser (13,4) at rungs: %s" % minimiser_13)
    Wr("c_pre decreases along the ladder: %d" % len(dec))
    for d in dec:
        Wr("   %5d -> %5d : %.4f -> %.4f  set by run (x=%d, L=%d) : %s" % d)
    Wr("   decreases at q' >= 127 set by a section run: %d; by the straddling run: %d; inherited: %d; other: %d" % (
        sum(1 for d in dec if d[1] >= 127 and d[6] == "section"), sum(1 for d in dec if d[1] >= 127 and d[6] == "straddle"),
        sum(1 for d in dec if d[1] >= 127 and d[6] == "inherited"), sum(1 for d in dec if d[1] >= 127 and d[6] == "other")))
    Wr("")
    Wr("STRADDLING RUN (run of M+q' containing W(q)): initial at %d rungs; min ratio over q' >= 23 non-initial: %s" % (
        sum(1 for r in rows if r["strad_initial"]), strad_min))
    Wr("   rungs q' >= 23 with straddling ratio < %.3f: %s" % (C_FLOOR, [(r["qp"], r["strad_ratio"]) for r in rows if r["qp"] >= 23 and not r["strad_initial"] and r["strad_ratio"] < C_FLOOR]))
    Wr("   rungs q' > 1000 with straddling ratio <= 100: %s" % [(r["qp"], round(r["strad_ratio"], 1)) for r in rows if r["qp"] > 1000 and r["strad_ratio"] <= 100])
    kinds = {"inherited": 0, "straddle": 0, "section": 0, "other": 0}
    for i in range(1, len(rows)):
        b = rows[i]
        if b["c_pre"] is None:
            continue
        x, L = b["cmin"]
        Wq = rows[i - 1]["Wq"]
        if x <= Wq < x + L - 1:
            kinds["straddle"] += 1
        elif x > Wq:
            kinds["section"] += 1
        elif x + L - 1 < Wq:
            kinds["inherited"] += 1
        else:
            kinds["other"] += 1
    Wr("   the c_pre minimiser at rung q' relative to W(q): %s" % kinds)
    Wr("   straddling run = the c_pre minimiser at %d rungs" % sum(1 for i in range(1, len(rows)) if rows[i]["c_pre"] is not None and rows[i]["cmin"] == (rows[i - 1]["x_s"], rows[i - 1]["L_s"])))
    Wr("SECTION bound W(q)/S(q') < %.3f at q' = %s (last %s)" % (C_FLOOR, sec_viol, sec_viol[-1] if sec_viol else None))
    Wr("   interior section runs with ratio < %.3f: %s" % (C_FLOOR, sec_min_ratio_viol))
    Wr("")
    Wr("V8 hidden hypothesis d_0(y_m(Q)) <= klo_m(Q), Q <= %d, m = 1..4: %d cells, %d exceptions: %s" % (QV, v8_cells, len(v8_exc), v8_exc[:40]))
    Wr("")
    Wr("per-rung table (selected):")
    Wr("    q    q'   q''       W(q)  twin rider    d0M   d1M   d0N   Fpre    c_pre  minimiser     c_any  anymin     top(x,L)    straddle(x,L,ratio)   sec_min_ratio  W/S")
    sel = set([5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1201, 1409, 1423, 1427, 1601, 2003, 3001, 4999, 7001, 9973, 14999, 19997])
    for r in rows:
        if r["q"] in sel:
            Wr("%5d %5d %5d %10d %5s %5s %6s %5s %5s %6d %8s %-12s %8s %-10s %-12s %-24s %-14s %6.1f" % (
                r["q"], r["qp"], r["qpp"], r["Wq"], "T" if r["twin_next"] else "-", "R" if r["rider"] else "-",
                r["d0M"], r["d1M"], r["d0N"], r["FpreM"],
                ("%.3f" % r["c_pre"]) if r["c_pre"] is not None else "inf", str(r["cmin"]),
                ("%.3f" % r["c_any"]) if r["c_any"] is not None else "inf", str(r["amin"]),
                str((r["x_top"], r["L_top"])), "(%d,%d,%.2f)%s" % (r["x_s"], r["L_s"], r["strad_ratio"], "i" if r["strad_initial"] else ""),
                ("%.2f %s" % (r["sec_ratio"], r["secmin"])) if r["sec_ratio"] is not None else "-", r["bound"]))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "lnp_prefix.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "lnp_prefix.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, dec=dec, formB=formB_exc, formA=formA_exc, v8_exc=v8_exc, e6=[(e[0], e[1]) for e in e6_direct]), f)
    print(txt[:20000])


if __name__ == "__main__":
    main()

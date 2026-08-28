"""Streamed fuel census: k_max across machine steps at scale (mechanic r11).

Machine M(y) = gears 5..y; openings = survivors mod P = prod(gears). Probe
gear q deletes openings at k = +-u_q mod q; a run of consecutive openings is
co-deletable in one lap iff their residues mod q fit in a window {a, a+s},
s = 2u_q = 3^(-1) mod q (chain_census.py frame, corpus-verified).

This implementation streams the period in numpy segments and counts
CO-DELETABLE k-TUPLES N_k (k consecutive openings, window condition) instead
of greedy maximal runs - convention-free, and equal to the maximal-run census
when k_max <= k (a maximal k-run contributes exactly one k-tuple and no
(k+1)-tuple). Method: consecutive-opening gap d classified mod q into offset
letters step +s -> +1, step -s -> -1, 0 mod q -> 0, else break; a (k-1)-word
of letters is window-valid iff its prefix-sum range <= 1. N_k = count of
window-valid (k-1)-words; instances k >= 3 recorded with address, interior
gap word, flanking gaps (up to a cap; count always exact). Also tracked:
F_k = max gap, F2 = max adjacent gap sum, span checks for the chain
prediction, and openings/N1 (openings in no valid pair).

Segment boundaries: tail of 8 openings carried; words counted at their
START index once. Full-period runs are exact censuses; partial runs (huge
periods) are exact on the scanned prefix, labeled.

Usage: uv run python research/fuel_census.py y [q1 q2 ...] [--kmax K]
       [--limit SLOTS]
Default probes: next 4 primes after y. Output: printed census + CSV append
research/data/fuel_census.csv.
"""
import os
import sys
import time
import numpy as np
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime

KWORD = 8   # count tuples up to k = KWORD (words of length KWORD-1)
CAP_LIST = 200


def fuel(y, probes, limit=None, seg=64_000_000, start=0):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, start + limit)
    uvals = [pow(6, -1, g) for g in gears]
    pr = {}
    for q in probes:
        u = pow(6, -1, q)
        s = (2 * u) % q
        pr[q] = dict(s=s, N=np.zeros(KWORD + 1, dtype=np.int64),
                     inst=[], Fk=0, F2=0, pred=0,
                     span=np.zeros(KWORD + 1, dtype=np.int64),
                     n1_pairs=0)
    tail = np.array([], dtype=np.int64)
    total_open = 0
    Fj = np.zeros(8, dtype=np.int64)  # spectrum: F_j = max sum of j gaps
    t0 = time.time()
    for a in range(start, K, seg):
        b = min(K, a + seg)
        ex = np.zeros(b - a, bool)
        for g, u in zip(gears, uvals):
            ex[(u - a) % g::g] = True
            ex[(-u - a) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + a
        total_open += len(op)
        ops = np.concatenate([tail, op])
        new0 = len(tail)  # count words starting at index >= new0-? see below
        # words starting index t counted when t >= start_ofs
        start_ofs = max(0, new0 - (KWORD + 1))
        # we counted words starting in the previous segment already;
        # exact rule: count word start t if ops[t] >= a (first element new)
        if len(ops) >= 2:
            d = np.diff(ops)
            c = np.concatenate([[0], np.cumsum(d)])
            for j in range(1, 7):
                if len(d) >= j:
                    Fj[j] = max(Fj[j], int((c[j:] - c[:-j]).max()))
            for q, st in pr.items():
                s = st["s"]
                dm = d % q
                letter = np.full(len(d), 9, dtype=np.int8)  # 9 = break
                letter[dm == 0] = 0
                letter[dm == s] = 1
                letter[dm == (q - s) % q] = -1
                valid = letter != 9
                st["Fk"] = max(st["Fk"], int(d.max()))
                if len(d) >= 2:
                    st["F2"] = max(st["F2"], int((d[:-1] + d[1:]).max()))
                # count a word iff its LAST opening is new (>= a): covers
                # boundary-straddling words once, never double-counts
                m2 = valid & (ops[1:] >= a)
                st["N"][2] += int(m2.sum())
                # flanked merged-gap spans: deleting word openings t..t+k-1
                # creates gap ops[t+k] - ops[t-1]; track max (chain pred)
                if len(ops) >= 4:
                    v2 = m2[1:-1] if len(m2) > 2 else m2[:0]
                    if v2.any():
                        sp2 = (ops[3:] - ops[:-3])[v2]
                        st["pred"] = max(st["pred"], int(sp2.max()))
                if m2.any():
                    st["span"][2] = max(st["span"][2], int(d[m2].max()))
                # openings covered by no valid pair (for N1): count new
                # openings t with neither valid[t-1] nor valid[t]
                covL = np.zeros(len(ops), bool)
                covL[:-1] |= valid
                covL[1:] |= valid
                st["n1_pairs"] += int(((~covL) & (ops >= a)
                                       & (ops < b)).sum())
                # k >= 3: window-valid words, cumulative
                ok = valid.copy()
                lo = np.minimum(0, letter)
                hi = np.maximum(0, letter)
                run = letter.astype(np.int16)
                for k in range(3, KWORD + 1):
                    L = len(d) - (k - 2)
                    if L <= 1:
                        break
                    ok = ok[:-1] & valid[k - 2:]
                    run = run[:-1] + letter[k - 2:]
                    # prefix-sum window: track min/max of partial sums
                    lo = np.minimum(lo[:-1], run)
                    hi = np.maximum(hi[:-1], run)
                    okk = ok & ((hi - lo) <= 1)
                    mk = okk & (ops[k - 1:k - 1 + L] >= a)
                    cnt = int(mk.sum())
                    st["N"][k] += cnt
                    if cnt:
                        idx = np.flatnonzero(mk)
                        sp = ops[idx + k - 1] - ops[idx]
                        st["span"][k] = max(st["span"][k], int(sp.max()))
                        take = idx if k >= 4 else                             idx[:max(0, CAP_LIST - len(st['inst']))]
                        inner = mk.copy()
                        inner[:1] = False
                        LL = len(mk)
                        if k + 1 <= len(ops) - 1:
                            iv = np.flatnonzero(inner)
                            iv = iv[iv + k < len(ops)]
                            if len(iv):
                                spf = ops[iv + k] - ops[iv - 1]
                                st["pred"] = max(st["pred"],
                                                 int(spf.max()))
                        if k >= 3:
                            for t in take:
                                word = tuple(int(x) for x in
                                             d[t:t + k - 1])
                                fl = (int(d[t - 1]) if t > 0 else -1,
                                      int(d[t + k - 1])
                                      if t + k - 1 < len(d) else -1)
                                st["inst"].append(
                                    (k, int(ops[t]), word, fl))
        tail = ops[-(KWORD + 1):] if len(ops) else tail
    dt = time.time() - t0
    return dict(y=y, P=P, K=K, start=start, gears=gears,
                openings=total_open, probes=pr, Fj=Fj, secs=dt)


def report(r, csvf):
    y, K, P = r["y"], r["K"], r["P"]
    lo = r.get("start", 0)
    # ROUND-24 FIX.  This line used to print K (the END slot) as "scanned"
    # and K/P as coverage, IGNORING --start.  A RESUMED run therefore
    # reported "100.0%" while having scanned only [start, K), and its
    # openings and every N_k were counts for THAT RANGE ALONE.  That is
    # exactly what produced the r21 machine-37 line "1.237e+12 slots
    # (100.0%), 112,205,953,878 openings" against the exact
    # prod_{5<=q<=37}(q-2) = 217,929,355,875: the three chained runs
    # [0, 1.2e11), [1.2e11, 6e11), [6e11, P) have opening counts
    # 21,144,680,389 + 84,578,721,608 + 112,205,953,878 summing EXACTLY to
    # prod(q-2).  Always report the RANGE, never the endpoint.
    scanned = K - lo
    frac = scanned / P
    print(f"machine y={y}: period {P:.3e}, range [{lo}, {K}) = "
          f"{scanned:.3e} slots ({100*frac:.1f}% of the period), "
          f"openings {r['openings']} (THIS RANGE ONLY), {r['secs']:.0f}s")
    if lo:
        print("  NOTE resumed run: openings and every N_k below are counts "
              "for this range only - sum the chained runs to get the "
              "period, and words straddling a junction are counted by "
              "neither run (the tail is empty at a resume).")
    print("  spectrum F_j (max sum of j consecutive gaps), j=1..6: "
          + " ".join(str(int(r['Fj'][j])) for j in range(1, 7)))
    for q, st in r["probes"].items():
        N = st["N"]
        kmax = max([k for k in range(2, KWORD + 1) if N[k] > 0],
                   default=1)
        print(f"  q={q} s={st['s']}: Fk={st['Fk']} F2={st['F2']} "
              f"pred={max(st['F2'], st['pred'])} "
              f"N1={st['n1_pairs']} "
              + " ".join(f"N{k}={N[k]}" for k in range(2, KWORD + 1)
                         if N[k] or k <= 4)
              + f"  k_max={kmax}")
        for k, addr, word, fl in st["inst"]:
            if k >= 4 or (k == 3 and N[3] <= 80):
                tag = "  <<<< k>=4" if k >= 4 else ""
                print(f"    k={k} at k-addr {addr}: word {word} "
                      f"flanks {fl}{tag}")
        csvf.write(f"{y},{q},{K},{P},{r['openings']},{st['Fk']},"
                   f"{st['F2']},{st['n1_pairs']},"
                   + ",".join(str(N[k]) for k in range(2, KWORD + 1))
                   + f",{kmax}\n")
        csvf.flush()


def main():
    args = sys.argv[1:]
    limit = None
    start = 0
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    if "--start" in args:
        i = args.index("--start")
        start = int(args[i + 1])
        del args[i:i + 2]
    y = int(args[0])
    probes = [int(a) for a in args[1:]]
    if not probes:
        p = y + 2
        while len(probes) < 4:
            if is_prime(p):
                probes.append(p)
            p += 2
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, "fuel_census.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a")
    if new:
        f.write("y,q,K_scanned,period,openings,Fk,F2,N1,"
                + ",".join(f"N{k}" for k in range(2, KWORD + 1))
                + ",k_max\n")
    r = fuel(y, probes, limit=limit, start=start)
    report(r, f)
    f.close()


if __name__ == "__main__":
    main()

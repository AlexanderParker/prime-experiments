"""Round 22 (constructor): THE ARITY LADDER - three distinct truncation
arities of the qualifying-run algebra, measured exactly at machines 11..41.

Round 21 (R41) reported "the truncation arity GROWS - 3-point at 19/23,
4-point at 29".  That number was the RESIDUE arity (nilpotency index of the
residue-qualifying successor map, tm_resid_runs.csv run_j = 0).  A residue
run is NOT the same thing as a kill chain: the two-teeth spacing law T3
(docs/novel/two-teeth-kill-spacing.md, kernel-checked) says the nonzero
letter classes must STRICTLY ALTERNATE, padded links transparent.  This
script separates the three arities and measures each exactly:

  A_res(M)   = min{ j : run_j^res  = 0 }   residue-qualifying runs only
  A_kill(M)  = min{ j : run_j^kill = 0 }   residue + T3 alternation
               ( = k_max(M -> q'), the fuel chain length: a k-kill chain has
                 k-1 killable interiors )
  A_relax(M) = min{ m : some m-window of the infinite alternating word
               ...a b a b... is unrealized }  - the arity at which R41's
               m-point relaxation refutes the infinite chain

Sources, all exact and full-period:
  * direct cyclic scan here (machines 11..23)
  * research/data/tm_resid_runs.csv   (residue runs, m11..m31, full period)
  * research/data/fuel_census.csv     (fuel chains N_k, full period rows)
  * research/data/run3_31.log, run3_37.log  (exact per-WORD depth-3 censuses)

Cross-check proved here: N_{k}(M -> q') == killable run_{k-1}(M) at every
machine where both censuses exist - an independent empirical confirmation of
T3 (the alternation filter alone converts one census into the other).

Also asserted: the SPAN CEILING  A_res <= min{ j : F_j(M) < 2u' * j }
(every qualifying gap is >= 2u' by T4, and j consecutive gaps sum to <= F_j).

Usage: uv run python research/arity_ladder.py
"""
import csv
import os
import re
import sys
from collections import Counter
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")

# F(M) chain (k-frame maximal gap), machines 11..41
KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91}


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def next_prime(y):
    p = y + 1
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)):
        p += 1
    return p


def letters(q1):
    """(a, b) = the two literal letters = the two nonzero tooth-difference
    classes; a = 2u' is the smaller."""
    u1 = round(q1 / 6)
    return 2 * u1, q1 - 2 * u1


def classify(v, q1, a, b):
    """T3 class of a gap value: +1 (class a), -1 (class b), 0 (padded),
    None (not qualifying)."""
    r = v % q1
    if r == 0:
        return 0
    if r == a:
        return 1
    if r == b:
        return -1
    return None


def killable(word, q1, a, b):
    last = 0
    for v in word:
        c = classify(v, q1, a, b)
        if c is None:
            return False
        if c == 0:
            continue
        if c == last:
            return False
        last = c
    return True


# ---------------------------------------------------------------- scans
_SCAN_CACHE = {}


def scan(y, maxj=4):
    """Full-period cyclic scan: word census of residue-qualifying j-runs."""
    if y in _SCAN_CACHE:
        return _SCAN_CACHE[y]
    gears = primes(5, y)
    P = prod(gears)
    q1 = next_prime(y)
    a, b = letters(q1)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    d = np.diff(np.concatenate([op, [op[0] + P]]))
    n = len(d)
    qual = np.array([classify(int(v), q1, a, b) is not None for v in d])
    words = {}
    for j in range(1, maxj + 1):
        ok = np.ones(n, bool)
        for t in range(j):
            ok &= np.roll(qual, -t)
        c = Counter()
        for i in np.flatnonzero(ok):
            c[tuple(int(d[(i + t) % n]) for t in range(j))] += 1
        words[j] = c
    Fj = []
    cum = np.zeros(n, np.int64)
    for j in range(1, 7):
        cum = cum + np.roll(d, -(j - 1)).astype(np.int64)
        Fj.append(int(cum.max()))
    r = dict(y=y, q1=q1, a=a, b=b, P=P, ngaps=n, F=int(d.max()), Fj=Fj,
             words=words)
    _SCAN_CACHE[y] = r
    return r


def parse_wordlog(path):
    """Parse a run3_*.log per-word census into {word: count}."""
    out = {}
    for line in open(path):
        m = re.match(r"\s+word \((.*?)\) span (\d+): (?:count = ([\d,]+)|0)",
                     line)
        if m:
            w = tuple(int(x) for x in m.group(1).split(","))
            out[w] = int(m.group(3).replace(",", "")) if m.group(3) else 0
    return out


def main():
    resid = {}
    with open(os.path.join(DDIR, "tm_resid_runs.csv")) as f:
        for row in csv.DictReader(f):
            resid[int(row["y"])] = row
    fuel = {}
    with open(os.path.join(DDIR, "fuel_census.csv")) as f:
        for row in csv.DictReader(f):
            y, q = int(row["y"]), int(row["q"])
            if q != next_prime(y):
                continue
            if int(row["K_scanned"]) != int(row["period"]):
                continue                       # full-period rows only
            fuel[y] = row

    table = {}
    print("=== part 1: direct full-period cyclic scans, machines 11..23")
    for y in (11, 13, 17, 19, 23):
        r = scan(y)
        q1, a, b = r["q1"], r["a"], r["b"]
        res = {j: sum(r["words"][j].values()) for j in r["words"]}
        kil = {j: sum(c for w, c in r["words"][j].items()
                      if killable(w, q1, a, b)) for j in r["words"]}
        assert r["F"] == KNOWN_F[y], (y, r["F"])
        row = resid[y]
        for j in (1, 2, 3, 4):
            assert res[j] == int(row["run%d" % j]), (y, j, res[j])
        assert int(row["ngaps"]) == r["ngaps"]
        for j in (1, 2, 3, 4, 5, 6):
            assert r["Fj"][j - 1] == int(row["F%d" % j]), (y, j)
        print("  y=%2d q'=%2d a=%2d b=%2d  ngaps %10d  F=%2d  res %s  kill %s"
              % (y, q1, a, b, r["ngaps"], r["F"],
                 [res[j] for j in (1, 2, 3, 4)],
                 [kil[j] for j in (1, 2, 3, 4)]))
        for j in (2, 3):
            if r["words"][j]:
                print("      depth-%d words: " % j + ", ".join(
                    "%s:%d%s" % (w, c, "*" if killable(w, q1, a, b) else "")
                    for w, c in sorted(r["words"][j].items())))
        table[y] = dict(q1=q1, a=a, b=b, res=res, kill=kil, Fj=r["Fj"],
                        src="scan")

    print("\n(* = killable: passes the T3 alternation filter)")
    print("\n=== part 2: machines 29, 31, 37 from full-period censuses")
    for y in (29, 31, 37):
        q1 = next_prime(y)
        a, b = letters(q1)
        row = resid.get(y)
        res = ({j: int(row["run%d" % j]) for j in (1, 2, 3, 4)} if row
               else {1: None, 2: None, 3: 8, 4: None})   # m37: run3 from log
        fr = fuel[y]
        Nk = [int(fr["N%d" % k]) for k in range(1, 9)]
        kil = {j: Nk[j] for j in range(0, 5)}      # killable run_j = N_{j+1}
        kmax = int(fr["k_max"])
        Fj = ([int(row["F%d" % j]) for j in range(1, 7)] if row else
              [KNOWN_F[y], int(fr["F2"]), 97, 0, 0, 0])
        print("  y=%2d q'=%2d a=%2d b=%2d  res %s  kill %s  k_max=%d"
              % (y, q1, a, b, [res.get(j) for j in (1, 2, 3, 4)],
                 [kil[j] for j in (1, 2, 3, 4)], kmax))
        table[y] = dict(q1=q1, a=a, b=b, res=res, kill=kil, Fj=Fj,
                        kmax=kmax, src="census")

    print("\n=== part 3: T3 CROSS-CHECK - the alternation filter turns the "
          "residue census into the fuel census")
    w31 = parse_wordlog(os.path.join(DDIR, "run3_31.log"))
    k31 = sum(c for w, c in w31.items() if killable(w, 37, 12, 25))
    print("  m31 depth-3: residue %d over %d nonzero words; killable (T3) %d;"
          " fuel census N4(31->37) = %d"
          % (sum(w31.values()), sum(1 for c in w31.values() if c), k31,
             table[31]["kill"][3]))
    assert k31 == table[31]["kill"][3], (k31, table[31]["kill"][3])
    print("      killable:     " + ", ".join(
        "%s:%d" % (w, c) for w, c in sorted(w31.items())
        if c and killable(w, 37, 12, 25)))
    print("      T3-violating: " + ", ".join(
        "%s:%d" % (w, c) for w, c in sorted(w31.items())
        if c and not killable(w, 37, 12, 25)))
    w37 = parse_wordlog(os.path.join(DDIR, "run3_37.log"))
    k37 = sum(c for w, c in w37.items() if killable(w, 41, 14, 27))
    print("  m37 depth-3: residue %d (words %s); killable (T3) %d; "
          "fuel census N4(37->41) = %d"
          % (sum(w37.values()), [w for w, c in w37.items() if c], k37,
             table[37]["kill"][3]))
    assert k37 == table[37]["kill"][3] == 0
    assert sum(w37.values()) == 8
    for y in (19, 23, 29):
        fr = fuel[y]
        for j in (1, 2, 3):
            if j in table[y]["kill"]:
                assert table[y]["kill"][j] == int(fr["N%d" % (j + 1)]), \
                    (y, j, table[y]["kill"][j], fr["N%d" % (j + 1)])
        print("  m%d: killable runs == fuel N_k  OK  %s"
              % (y, [table[y]["kill"][j] for j in (1, 2, 3)]))

    print("\n=== part 4: OVERLAP LEMMA (factor closure)")
    print("  If every realized depth-m word is known, run_{m+1} = 0 unless")
    print("  some pair w, w' of realized m-words has w[1:] == w'[:-1].")
    for y, wl in ((31, w31), (37, w37)):
        real = [w for w, c in wl.items() if c]
        ov = [(w, v) for w in real for v in real if w[1:] == v[:-1]]
        print("  m%d: realized depth-3 words %s -> %d overlapping pairs%s"
              % (y, real, len(ov),
                 "  => run_4^res = 0 PROVED by the lemma" if not ov else
                 "  => lemma inconclusive (census needed)"))
        if not ov:
            table[y]["res"][4] = 0

    print("\n=== part 5: THE THREE ARITIES")
    print("    y   q'   a   b  A_res  A_kill  A_relax   notes")
    arities = {}
    for y in sorted(table):
        t = table[y]
        q1, a, b = t["q1"], t["a"], t["b"]
        ares = next((j for j in (1, 2, 3, 4) if t["res"].get(j) == 0), None)
        akil = next((j for j in (1, 2, 3, 4) if t["kill"].get(j) == 0), None)
        arel = None
        for m in (1, 2, 3, 4):
            wins = [tuple((a if (i + s) % 2 == 0 else b) for i in range(m))
                    for s in (0, 1)]
            if t["src"] == "scan":
                cnt = min(scan(y)["words"][m].get(w, 0) for w in wins)
            elif y == 29:
                # depth<=3 inventory: all depth-3 runs are permutations of
                # {10,10,21} (R39 tm_deepruns) so (21,10,21) is absent
                cnt = {1: 1, 2: 1, 3: 0}.get(m)
            elif y == 31:
                cnt = {1: 1, 2: 1, 3: min(w31.get(w, 0) for w in wins),
                       4: 0}.get(m)
            elif y == 37:
                cnt = {1: 1, 2: 1, 3: min(w37.get(w, 0) for w in wins)}.get(m)
            else:
                cnt = None
            if cnt == 0:
                arel = m
                break
        arities[y] = (ares, akil, arel)
        if akil is not None and "kmax" in t:
            assert akil == t["kmax"], (y, akil, t["kmax"])
        print("  %3d %4d %3d %3d %6s %7s %8s   %s"
              % (y, q1, a, b, ares, akil, arel,
                 "k_max=%d" % t["kmax"] if "kmax" in t else ""))

    ys = sorted(table)
    print("\n  machines            : " + ", ".join(str(y) for y in ys))
    print("  A_res   (residue)   : "
          + ", ".join(str(arities[y][0]) for y in ys))
    print("  A_kill  (= k_max)   : "
          + ", ".join(str(arities[y][1]) for y in ys))
    print("  A_relax (relaxation): "
          + ", ".join(str(arities[y][2]) for y in ys))

    print("\n=== part 6: SPAN CEILING  A_res <= min{ j : F_j < 2u'*j }")
    for y in ys:
        t = table[y]
        Fj, a = t["Fj"], t["a"]
        ceil = next((j for j in range(1, 7)
                     if Fj[j - 1] and Fj[j - 1] < a * j), None)
        ares = arities[y][0]
        ok = ceil is None or ares is None or ares <= ceil
        print("  m%2d: 2u'=%2d  F_j=%s  ceiling=%s  A_res=%s  %s"
              % (y, a, Fj, ceil, ares, "OK" if ok else "VIOLATION"))
        assert ok, y

    print("\n=== part 7: A_kill against the LITERAL CAP (R20) - does the "
          "arity track the gear's arithmetic?")
    E = [r for r in range(35) if r % 5 not in (1, 4) and r % 7 not in (1, 6)]
    Eset = set(E)

    def litcap(q1):
        a, b = letters(q1)
        best = 1
        for r in E:
            for first in (a, b):
                k, cur, nxt_letter = 1, r, first
                while (cur + nxt_letter) % 35 in Eset:
                    cur = (cur + nxt_letter) % 35
                    k += 1
                    nxt_letter = a if nxt_letter == b else b
                best = max(best, k)
        return best

    for y in ys:
        t = table[y]
        lc = litcap(t["q1"])
        ak = arities[y][1]
        pad = "" if ak <= lc else "  (exceeds litcap - padded link required)"
        print("  m%2d -> q'=%2d (q' mod 210 = %3d): litcap %d   A_kill %s%s"
              % (y, t["q1"], t["q1"] % 210, lc, ak, pad))

    print("\n=== part 8: conditional decay of the residue run ladder")
    for y in ys:
        r = table[y]["res"]
        if not r.get(1):
            continue
        ng = int(resid[y]["ngaps"]) if y in resid else None
        row = ["p1=%.6g" % (r[1] / ng)] if ng else []
        for j in (2, 3, 4):
            if r.get(j - 1):
                row.append("r%d/r%d=%.4g" % (j, j - 1, r[j] / r[j - 1]))
        print("  m%2d: " % y + "  ".join(row))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()

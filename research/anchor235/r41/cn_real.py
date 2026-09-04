"""R2.a.i.a.1.a item 3 - the real machine against the adversary.

For a bound B in {7, 11, 13} the islands are the offsets no gear <= B can reach.  For an integer q
coprime to 6 with arc d, gear g strikes offset i iff q^2 = 2 - 6i or -6i (mod g).

  * at every FAILING q (no free island in [1, d)): the real minimum cover R(q) of all islands by
    gears of (B, q] with their REAL phases, exact ILP;
  * the adversarial cover number K_B(d) for that same d, exact ILP (free phases, any gears > B);
  * the ratio R(q) / K_B(d).  R(q) >= K_B(d) always, since the real phases are a legal
    adversarial play.
  * at NON-failing q: the real minimum cover of the STRUCK islands against K_B(d).

Usage: uv run python research/anchor235/r41/cn_real.py [--B 7] [--QMAX 3000] [--tl 300]
"""
import argparse
import os
import time
from math import isqrt

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def bar(g):
    """Offset classes mod g that gear g can never strike at any q coprime to g."""
    qr = set((x * x) % g for x in range(1, g))
    out = []
    for i in range(g):
        if ((2 - 6 * i) % g) not in qr and ((-6 * i) % g) not in qr:
            out.append(i)
    return out


def island_pred(B, PR):
    """Return (modulus, sorted residues) of the island classes for bound B."""
    gs = [p for p in PR if p <= B]
    mod = 1
    res = [0]
    for g in gs:
        bg = set(bar(g))
        newres = []
        for r in res:
            for t in range(g):
                x = r + mod * t
                if x % g in bg:
                    newres.append(x)
        mod *= g
        res = sorted(newres)
    return mod, res


def arc(q):
    return (q + 1) // 3 if q % 6 == 5 else (2 * q + 1) // 3


def ilp_min_cover(m, sets, tl, gear_clique=True):
    n = len(sets)
    if n == 0:
        return None, -1
    rows, cols = [], []
    for j, (g, s) in enumerate(sets):
        for e in s:
            rows.append(e)
            cols.append(j)
    cons = [LinearConstraint(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, n)),
                             lb=1, ub=np.inf)]
    if gear_clique:
        bygear = {}
        for j, (g, s) in enumerate(sets):
            if g > 0:
                bygear.setdefault(g, []).append(j)
        rr, cc, nr = [], [], 0
        for g, js in bygear.items():
            if len(js) > 1:
                for j in js:
                    rr.append(nr)
                    cc.append(j)
                nr += 1
        if nr:
            cons.append(LinearConstraint(coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nr, n)),
                                         lb=-np.inf, ub=1))
    out = milp(c=np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": float(tl), "mip_rel_gap": 0.0, "presolve": True})
    if out.x is None:
        return None, -1
    return [j for j in range(n) if out.x[j] > 0.5], out.mip_dual_bound


def maximal_only(S):
    L = sorted(S, key=lambda s: -len(s))
    keep = []
    for s in L:
        if not any(s < t for t in keep):
            keep.append(s)
    return keep


def adversary_K(d, isl, PR, B, tl):
    m = len(isl)
    idx = {i: k for k, i in enumerate(isl)}
    cand = []
    for g in PR:
        if g <= B:
            continue
        if g > 3 * d + 2:
            break
        buck = {}
        for i in isl:
            buck.setdefault(i % g, []).append(idx[i])
        u = pow(6, -1, g)
        # only phases whose strike classes actually contain an island can matter, and every such
        # phase is r = -6i or 2 - 6i for an island i: O(m) phases per gear instead of O(g).
        rs = set()
        h = (g - 1) // 2
        for i in isl:
            for r in (((-6 * i) % g), ((2 - 6 * i) % g)):
                if r and pow(r, h, g) == 1:
                    rs.add(r)
        S = set()
        for r in rs:
            s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
            if len(s) >= 2:
                S.add(frozenset(s))
        for s in maximal_only(S):
            cand.append((g, s))
    for k in range(m):
        cand.append((0, frozenset((k,))))
    pick, lb = ilp_min_cover(m, cand, tl)
    return (len(pick) if pick else -1), lb, sorted(cand[j][0] for j in pick if cand[j][0] > 0)


def real_sets(q, isl, PR, B):
    """Real strike sets of the machine (B, q] at q; returns (sets, free-island list)."""
    idx = {i: k for k, i in enumerate(isl)}
    m = len(isl)
    hit = [0] * m
    sets = []
    for g in PR:
        if g > q:
            break
        if g <= B and q % g:
            continue          # a gear <= B not dividing q cannot reach an island (the bar)
        r = (q * q) % g
        u = pow(6, -1, g)
        a = ((2 - r) * u) % g
        b = ((-r) * u) % g
        s = [idx[i] for i in isl if i % g == a or i % g == b]
        if s:
            for e in s:
                hit[e] += 1
            sets.append((g, frozenset(s)))
    free = [isl[e] for e in range(m) if hit[e] == 0]
    return sets, free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=7)
    ap.add_argument("--QMAX", type=int, default=3000)
    ap.add_argument("--tl", type=float, default=300.0)
    ap.add_argument("--dmax", type=int, default=4000)
    ap.add_argument("--sample", type=str, default="")
    ap.add_argument("--tag", type=str, default="real")
    args = ap.parse_args()
    LOG = open(os.path.join(OUT, "cn_%s.txt" % args.tag), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        LOG.write(s + "\n")
        LOG.flush()

    NMAX = max(3 * args.dmax + 10, args.QMAX + 10)
    FL = sieve(NMAX)
    PR = [p for p in range(5, NMAX + 1) if FL[p]]
    MOD, RES = island_pred(args.B, PR)
    RESSET = set(RES)
    say("# B = %d ; islands = %d classes mod %d ; density %.5f"
        % (args.B, len(RES), MOD, len(RES) / MOD))

    # --- find failures (fast: keep only the still-free islands, gear by gear)
    fails = []
    for q in range(5, args.QMAX + 1, 2):
        if q % 3 == 0 or q % 5 == 0:
            continue
        d = arc(q)
        isl = np.array([i for i in range(1, d) if i % MOD in RESSET], dtype=np.int64)
        if len(isl) == 0:
            fails.append((q, d, 0))
            continue
        m0 = len(isl)
        free = isl
        for g in PR:
            if g > q:
                break
            if g <= args.B and q % g:
                continue
            r = (q * q) % g
            u = pow(6, -1, g)
            a = ((2 - r) * u) % g
            b = ((-r) * u) % g
            rem = free % g
            free = free[(rem != a) & (rem != b)]
            if len(free) == 0:
                break
        if len(free) == 0:
            fails.append((q, d, m0))
    say("# failures found to q = %d : %d ; largest %s"
        % (args.QMAX, len(fails), fails[-1][0] if fails else "-"))
    say("")
    say("#  q      d      m     R(q)   K_B(d)  R/K    gears of a real optimal cover")
    rows = []
    for q, d, m in fails:
        if m == 0 or d > args.dmax:
            say("  %-6d %-6d %-5d   (no island in [1,d))" % (q, d, m) if m == 0
                else "  %-6d %-6d %-5d   (arc beyond dmax)" % (q, d, m))
            continue
        isl = [i for i in range(1, d) if i % MOD in RESSET]
        sets, free = real_sets(q, isl, PR, args.B)
        pick, _ = ilp_min_cover(len(isl), sets, args.tl)
        R = len(pick)
        K, klb, kg = adversary_K(d, isl, PR, args.B, args.tl)
        rows.append((q, d, m, R, K))
        say("  %-6d %-6d %-5d %-6d %-7d %-6.2f %s"
            % (q, d, m, R, K, R / K if K > 0 else -1,
               sorted(sets[j][0] for j in pick)[:12]))
    if rows:
        rr = [R / K for _, _, _, R, K in rows if K > 0]
        say("")
        say("# R/K over %d failures: min %.2f median %.2f max %.2f ; monotone in q? %s"
            % (len(rr), min(rr), sorted(rr)[len(rr) // 2], max(rr),
               all(rr[i] <= rr[i + 1] for i in range(len(rr) - 1))))

    # --- non-failing q: minimum cover of the STRUCK islands vs K(d)
    if args.sample:
        say("")
        say("# NON-FAILING q: minimum blocking set of the STRUCK islands vs K_B(d)")
        say("#  q      d      m     free  struck  MBS   K_B(d)  MBS/K")
        for q in [int(v) for v in args.sample.split(",")]:
            d = arc(q)
            if d > args.dmax:
                continue
            isl = [i for i in range(1, d) if i % MOD in RESSET]
            sets, free = real_sets(q, isl, PR, args.B)
            struck = [e for e in range(len(isl)) if any(e in s for _, s in sets)]
            ren = {e: k for k, e in enumerate(struck)}
            s2 = [(g, frozenset(ren[e] for e in s)) for g, s in sets if s]
            pick, _ = ilp_min_cover(len(struck), s2, args.tl)
            MBS = len(pick)
            K, klb, kg = adversary_K(d, isl, PR, args.B, args.tl)
            say("  %-6d %-6d %-5d %-5d %-7d %-5d %-7d %.2f"
                % (q, d, len(isl), len(free), len(struck), MBS, K, MBS / K if K > 0 else -1))
    LOG.close()


main()

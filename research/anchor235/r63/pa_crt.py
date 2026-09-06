"""pa_crt.py -- the exact CRT search: the largest neighbour a letter gap can have, decided
scan-free from the residues alone.

In tooth units lam_g = 6 x (mod g) every gear has its teeth at +-1, and

    gear g strikes the column at offset j   iff   lam_g + 6 j = +-1 (mod g).

A 2-run with the letter on the left is three prescribed openings at offsets 0, a_L, S = a_L + a
with every other offset in (0, S) struck.  Choosing one class lam_g per gear is a choice of
configuration; by CRT every configuration occurs somewhere in the period, so

    the cell (a_L, a) is realisable  <=>  some choice of classes covers every interior column.

This is an exact covering feasibility problem with |M| <= 9 gears and at most 60 columns; it is
solved here by depth-first search on the lowest uncovered column, with the reachability prune.
No period is scanned.

Also computed, for the minimal INFEASIBLE cell of each rung:
  * the counting (LP-shaped) certificate: a sub-interval W of the run with
        sum_g max_lam |{j in W : g strikes j}|  <  |W|,
    which is a local, machine-checkable proof of infeasibility -- and whether one exists at all;
  * which gears the search actually needs.

Outputs results/pa_crt.txt / .json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import u_of                                  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

NODECAP = 40_000_000


def unorm(g):
    w = u_of(g)
    return min(w, g - w)


def eps_of(q):
    return 1 if q % 6 == 5 else -1


def strike(g, lam, j):
    r = (lam + 6 * j) % g
    return r == 1 % g or r == (g - 1) % g


def options(g, opens, S):
    """admissible classes for gear g (missing every prescribed opening), as (mask, lam) pairs."""
    out = {}
    cover = [j for j in range(1, S) if j not in opens]
    for lam in range(g):
        if any(strike(g, lam, j) for j in opens):
            continue
        m = 0
        for i, j in enumerate(cover):
            if strike(g, lam, j):
                m |= 1 << i
        out.setdefault(m, lam)
    return [(m, lam) for m, lam in out.items()], cover


class Search:
    def __init__(self, gears, opens, S):
        self.gears = gears
        self.opts = {}
        self.union = {}
        for g in gears:
            o, self.cover = options(g, opens, S)
            self.opts[g] = o
            u = 0
            for m, _ in o:
                u |= m
            self.union[g] = u
        self.full = (1 << len(self.cover)) - 1
        self.nodes = 0
        self.sol = None

    def run(self):
        self.nodes = 0
        self.sol = None
        ok = self._dfs(0, tuple(self.gears), {})
        return ok, self.sol, self.nodes

    def _dfs(self, covered, rest, assign):
        self.nodes += 1
        if self.nodes > NODECAP:
            raise RuntimeError("node cap")
        if covered == self.full:
            self.sol = dict(assign)
            return True
        reach = 0
        for g in rest:
            reach |= self.union[g]
        if (self.full & ~covered) & ~reach:
            return False
        miss = (~covered) & self.full
        j = (miss & -miss).bit_length() - 1
        bit = 1 << j
        for i, g in enumerate(rest):
            if not (self.union[g] & bit):
                continue
            nrest = rest[:i] + rest[i + 1:]
            for m, lam in self.opts[g]:
                if not (m & bit):
                    continue
                assign[g] = lam
                if self._dfs(covered | m, nrest, assign):
                    return True
                del assign[g]
        return False


def feasible(gears, v, a):
    """is the cell (v, a) -- three openings at 0, v, v + a -- realisable?"""
    S = v + a
    opens = {0, v, S}
    # a gear with no class at all that misses the three prescribed openings kills the cell
    # outright (this is the gear-5 pair filter when it fires); the covering search below would
    # not see it, because such a gear contributes no options and is simply never chosen.
    for g in gears:
        if not any(not any(strike(g, lam, o) for o in opens) for lam in range(g)):
            return False, None, 0
    s = Search(gears, opens, S)
    return s.run()


def counting_certificate(gears, v, a, maxlen=None):
    """the smallest sub-interval W of (0, S) whose demand exceeds the gears' capacity."""
    S = v + a
    opens = {0, v, S}
    best = None
    for lo in range(1, S):
        for hi in range(lo, S):
            W = [j for j in range(lo, hi + 1) if j not in opens]
            if not W:
                continue
            cap = 0
            per = {}
            for g in gears:
                c = 0
                for lam in range(g):
                    if any(strike(g, lam, o) for o in opens):
                        continue
                    c = max(c, sum(1 for j in W if strike(g, lam, j)))
                per[g] = c
                cap += c
            if cap < len(W):
                if best is None or len(W) < best[0]:
                    best = (len(W), lo, hi, cap, dict(per))
    return best


def main():
    t0 = time.time()
    L = []
    W = L.append
    W("=== THE EXACT CRT SEARCH: the row of a letter, decided from residues alone ===")
    W("cell (v, a) = three prescribed openings at 0, v, v + a with every column between them")
    W("struck; a choice of one residue class per gear is a configuration, and by CRT every")
    W("configuration occurs in the period, so feasibility = realisability.  No period is scanned.")

    RUNGS = [(13, [5, 7, 11], 7, 3),
             (17, [5, 7, 11, 13], 11, 7),
             (19, [5, 7, 11, 13, 17], 18, 12),
             (23, [5, 7, 11, 13, 17, 19], 25, 20),
             (29, [5, 7, 11, 13, 17, 19, 23], 34, 25),
             (31, [5, 7, 11, 13, 17, 19, 23, 29], 43, 35)]
    res = {}
    for qn, gears, F, rknown in RUNGS:
        aL = 2 * unorm(qn)
        W(f"\n=== rung {qn}  M = {gears}  F = {F}  a_L = {aL}  "
          f"(scanned r(a_L) = {rknown}, pinned bound F + 3 - a_L = {F + 3 - aL}) ===")
        W("  a | span | feasible? | nodes | witness lam (tooth units)")
        rows = []
        top = 0
        for a in range(F, 0, -1):
            t1 = time.time()
            try:
                ok, sol, nodes = feasible(gears, aL, a)
            except RuntimeError:
                W(f"  {a:>3} | {aL+a:>4} | NODE CAP EXCEEDED | >{NODECAP} |")
                rows.append(dict(a=a, feasible=None, nodes=NODECAP))
                continue
            rows.append(dict(a=a, feasible=bool(ok), nodes=int(nodes),
                             sol={str(k): v for k, v in (sol or {}).items()}))
            W(f"  {a:>3} | {aL+a:>4} | {'YES' if ok else 'no ':<9} | {nodes:>9} | "
              f"{sol if ok else ''}  [{time.time()-t1:.1f}s]")
            if ok and top == 0:
                top = a
                break
        W(f"  --> largest feasible neighbour of an a_L-gap: {top}   "
          f"(scanned r(a_L) = {rknown}; {'MATCH' if top == rknown else 'MISMATCH'})")
        W(f"  --> a_L + top = {aL + top}, F = {F}, excess {aL + top - F:+d}; "
          f"pinned bound F + 3 - a_L = {F + 3 - aL}, slack {F + 3 - aL - top}")
        # the counting certificate at the minimal infeasible cell
        amin = top + 1
        t1 = time.time()
        cert = counting_certificate(gears, aL, amin)
        if cert:
            n, lo, hi, cap, per = cert
            W(f"  counting certificate at a = {amin}: window [{lo}, {hi}] has {n} columns to "
              f"cover and total capacity {cap} < {n}; per gear {per}  [{time.time()-t1:.1f}s]")
        else:
            W(f"  counting certificate at a = {amin}: NONE -- no sub-interval of the run has "
              f"demand above the gears' capacity, so the infeasibility is not a counting fact "
              f" [{time.time()-t1:.1f}s]")
        res[str(qn)] = dict(gears=gears, F=F, aL=aL, rknown=rknown, top=top, rows=rows,
                            cert=cert)
        json.dump(res, open(os.path.join(OUT, "pa_crt.json"), "w"))
        open(os.path.join(OUT, "pa_crt.txt"), "w").write("\n".join(L))

    txt = "\n".join(L)
    open(os.path.join(OUT, "pa_crt.txt"), "w").write(txt)
    print(f"wrote {OUT}/pa_crt.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    for k, e in res.items():
        print(f"rung {k}: top {e['top']} vs scanned {e['rknown']} "
              f"({'match' if e['top'] == e['rknown'] else 'MISMATCH'})")


if __name__ == "__main__":
    main()

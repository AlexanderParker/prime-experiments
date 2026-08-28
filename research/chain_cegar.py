"""Round 25 (constructor): THE CHAIN - (D) certified with NO period scan and
NO dumped dictionary anywhere in the loop.

R58 reduced (D) at one step to 90-955 realisability queries against a dumped
realised-tuple set (which came from a full-period scan).  research/crt_dict.py
answers exactly those queries by CRT arithmetic from the GEAR LIST alone.  This
script puts the two together and runs the ladder as far as it goes.

  MODE --shadow y   : run the loop with the CRT oracle AND the round-24 dump
                      side by side, asserting they agree on every query.
                      Available at y = 19, 23, 29 (the steps with dumps).
  MODE --step y     : run the loop with the CRT oracle ONLY.  y may be any
                      machine in STEPS, including 31 and 37 where no dump and
                      no scan exists.
  MODE --chain      : run every step in order and print the ladder.

Sound by construction: an edge/state is deleted only when the oracle PROVES
the tuple is unrealised (an exhaustive refutation of the CRT cover CSP).  An
UNDECIDED query (node budget exceeded) never deletes, so the bound stays an
upper bound on F(M + q') at every stage.

Usage:  python research/chain_cegar.py --shadow 29
        python research/chain_cegar.py --chain
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from machinefree_cert import build_mf_edges, NEG        # noqa: E402
import crt_dict                                          # noqa: E402

DDIR = os.path.join(HERE, "data")

# y : (F(M), q', exact F(M+q') where known else None)
STEPS = {19: (25, 23, 34), 23: (34, 29, 43), 29: (43, 31, 58),
         31: (58, 37, 88), 37: (88, 41, 91), 41: (91, 43, 103)}
MOD = 35
M = 4
ITCAP = 400000


# --------------------------------------------------------------- oracles
_PY = None
_PNB = None


def _pinit(y, nb):
    global _PY, _PNB
    _PY, _PNB = y, nb


def _pwork(t):
    try:
        return crt_dict.realised(_PY, t, _PNB)
    except crt_dict.Budget:
        return None


class CRTOracle:
    """Scan-free.  answers(tuple) -> True/False/None (None = undecided).

    `batch(list)` answers a whole refinement round's queries in parallel; the
    answers are memoised so a repeated query is free.
    """

    def __init__(self, y, node_budget=2_000_000, workers=1):
        self.y = y
        self.nb = node_budget
        self.n = 0
        self.spans = []
        self.secs = 0.0
        self.undecided = 0
        self.slowest = (0.0, None)
        self.memo = {}
        self.pool = None
        if workers > 1:
            from multiprocessing import Pool
            self.pool = Pool(workers, initializer=_pinit,
                             initargs=(y, node_budget))

    def batch(self, tups):
        need = [t for t in dict.fromkeys(tups) if t not in self.memo]
        if not need:
            return
        t0 = time.time()
        if self.pool is not None:
            ans = self.pool.map(_pwork, need, chunksize=1)
        else:
            ans = [_seq(self.y, t, self.nb) for t in need]
        dt = time.time() - t0
        self.secs += dt
        for t, a in zip(need, ans):
            self.memo[t] = a
            self.n += 1
            self.spans.append(sum(t))
            if a is None:
                self.undecided += 1
        if dt / max(1, len(need)) > self.slowest[0]:
            self.slowest = (dt / max(1, len(need)), need[0])

    def __call__(self, tup):
        if tup in self.memo:
            return self.memo[tup]
        t0 = time.time()
        r = _seq(self.y, tup, self.nb)
        dt = time.time() - t0
        self.secs += dt
        self.n += 1
        self.spans.append(sum(tup))
        self.memo[tup] = r
        if r is None:
            self.undecided += 1
        if dt > self.slowest[0]:
            self.slowest = (dt, tup)
        return r

    def close(self):
        if self.pool is not None:
            self.pool.terminate()


def _seq(y, tup, nb):
    try:
        return crt_dict.realised(y, tup, nb)
    except crt_dict.Budget:
        return None


class DumpOracle:
    """Round-24 oracle: the full-period realised-tuple dump + pair census."""

    def __init__(self, y):
        self.y = y
        self.o4 = set()
        for line in open(os.path.join(DDIR, "tuples4_%d.txt" % y)):
            self.o4.add(tuple(int(x) for x in line.split()))
        self.o2 = load_pairs(y)

    def batch(self, tups):
        return

    def __call__(self, tup):
        return tup in (self.o2 if len(tup) == 2 else self.o4)


def load_pairs(y):
    """Mechanic's full-period lag-1 joint census.

    ROUND-25 NOTE. Until 2026-08-29 both this file and gap_pair_hist.csv were
    LINEAR scans, each short the one cyclic wrap-around item (Lateral caught
    the ghist half; round 24 had already caught the joint half), and the seam
    pair was recovered from the marginal defect.  Mechanic repaired both at
    source mid-round: the joint marginals now equal ghist exactly and the totals
    agree (214,708,725 gaps at m29).  This loader handles BOTH states - it
    verifies cyclic consistency and only falls back to the seam recovery when
    the marginals are short.
    """
    import csv
    import collections
    o2 = set()
    mu, mv, hist = collections.Counter(), collections.Counter(), \
        collections.Counter()
    for r in csv.DictReader(open(os.path.join(DDIR, "gap_pair_joint.csv"))):
        if int(r["y"]) != y or int(r["lag"]) != 1 or \
                r["coverage"] != "1.000000":
            continue
        gu, gv, c = int(r["gu"]), int(r["gv"]), int(r["count"])
        o2.add((gu, gv))
        mu[gu] += c
        mv[gv] += c
    for r in csv.DictReader(open(os.path.join(DDIR, "gap_pair_hist.csv"))):
        if int(r["y"]) == y and r["kind"] == "ghist" and \
                r["coverage"] == "1.000000":
            hist[int(r["value"])] = int(r["count"])
    du = [g for g in hist if hist[g] - mu[g] == 1]
    dv = [g for g in hist if hist[g] - mv[g] == 1]
    if not du and not dv:
        assert mu == hist and mv == hist, "joint marginals != ghist"
        return o2                                   # repaired file: complete
    assert len(du) == 1 and len(dv) == 1, (du, dv)
    o2.add((du[0], dv[0]))                          # pre-repair: the seam pair
    return o2


# --------------------------------------------------------------- closure
# Edges are kept SORTED BY SOURCE once and for all, so the max-plus relaxation
# step is a segment maximum (np.maximum.reduceat) instead of the scattered
# np.maximum.at - the difference is ~50x at the 2.6M-edge sizes of machine 37.
def close_sorted(S, edst, ew, live, Rs, Ls, usrc, starts, cap=96):
    hh = Rs.copy()
    for _ in range(cap):
        vals = np.where(live, ew + hh[edst], NEG)
        seg = np.maximum.reduceat(vals, starts)
        new = Rs.copy()
        new[usrc] = np.maximum(Rs[usrc], seg)
        if np.array_equal(new, hh):
            break
        hh = new
    else:
        return None, None
    return hh, int((Ls + hh).max())


def walk_sorted(hh, Ls, edst, ew, live, usrc, starts, ends, where, st=None):
    """Greedy maximising walk from the arg-max state (or a given state)."""
    if st is None:
        st = int(np.argmax(Ls + hh))
    out = []
    for _ in range(40):
        gi = where.get(st)
        if gi is None:
            break
        lo, hi = starts[gi], ends[gi]
        sl = slice(lo, hi)
        gains = np.where(live[sl], ew[sl] + hh[edst[sl]], NEG)
        j = lo + int(np.argmax(gains))
        if int(gains[j - lo]) < int(hh[st]):
            break
        out.append(j)
        st = int(edst[j])
    return st, out


def run_step(y, oracle, shadow=None, verbose=True, itcap=ITCAP,
             topk=1, f2=0):
    F, Q1, EXACT = STEPS[y]
    budget = F + Q1
    t0 = time.time()
    S, esrc, edst, ew, Rs, Ls, tup = build_mf_edges(F, Q1, MOD, M)
    Rs, Ls = Rs.copy(), Ls.copy()
    build_secs = time.time() - t0
    if verbose:
        print("  MF_%d(mod %d) built: %d states, %d edges  (%.0fs)"
              % (M, MOD, S, len(esrc), build_secs), flush=True)
    if f2:
        # THE BRIEF'S CHAIN SHAPE (R53 run 2): consume the PREVIOUS rung's
        # two-gap output.  A state whose (flank, base) pair sums above a SOUND
        # upper bound on F_2(M) carries an unrealised pair, so deleting it is
        # sound - given that bound.  f2 is an ASSUMPTION, labelled as such in
        # the report; it is not derived inside this loop.
        okst = (Ls + Rs) <= f2
        keep = okst[esrc] & okst[edst]
        esrc, edst, ew, tup = esrc[keep], edst[keep], ew[keep], tup[keep]
        Ls = np.where(okst, Ls, NEG)
        Rs = np.where(okst, Rs, NEG)
        if verbose:
            print("  given F_2(M) <= %d: %d states survive, %d edges"
                  % (f2, int(okst.sum()), len(esrc)), flush=True)
    order = np.argsort(esrc, kind="stable")
    esrc, edst, ew, tup = esrc[order], edst[order], ew[order], tup[order]
    usrc, starts = np.unique(esrc, return_index=True)
    ends = np.append(starts[1:], len(esrc))
    where = {int(s): i for i, s in enumerate(usrc.tolist())}
    t64 = tup.astype(np.int64)
    key4 = ((t64[:, 0] * 128 + t64[:, 1]) * 128 + t64[:, 2]) * 128 + t64[:, 3]
    by4 = {}
    for i, k in enumerate(key4.tolist()):
        by4.setdefault(k, []).append(i)
    by4 = {k: np.array(v, np.int64) for k, v in by4.items()}
    pairkey = (Ls * 128 + Rs).astype(np.int64)
    bypair = {}
    for i, k in enumerate(pairkey.tolist()):
        if Ls[i] > NEG // 2:
            bypair.setdefault(k, []).append(i)
    bypair = {k: np.array(v, np.int64) for k, v in bypair.items()}
    live = np.ones(len(esrc), bool)
    stlive = np.ones(S, bool)
    dead = np.zeros(S, bool)
    asked4, asked2, killed4, killed2 = set(), set(), set(), set()
    disagree = []
    it = 0
    while True:
        it += 1
        hh, bnd = close_sorted(S, edst, ew, live, Rs, Ls, usrc, starts)
        if hh is None:
            return dict(status="CYCLIC", it=it, y=y)
        if bnd <= budget:
            break
        # TOP-K BATCHING: refine along the K best abstract objects at once,
        # not just the single arg-max.  Every deletion is still individually
        # justified by an oracle refutation, so soundness is unchanged; this
        # only cuts the number of closures (the expensive part at m37+).
        sc = Ls + hh
        if topk <= 1:
            tops = [int(np.argmax(sc))]
        else:
            over = np.flatnonzero(sc > budget)
            if len(over) > topk:
                over = over[np.argsort(sc[over])[::-1][:topk]]
            tops = over.tolist() or [int(np.argmax(sc))]
        wl = []
        for st0 in tops:
            _st, w = walk_sorted(hh, Ls, edst, ew, live, usrc, starts, ends,
                                 where, st0)
            wl.extend(w)
        idx = np.unique(np.array(wl, np.int64)) if wl \
            else np.zeros(0, np.int64)
        progress = False
        sts = list(tops)
        for e in idx.tolist():
            sts.append(int(esrc[e]))
            sts.append(int(edst[e]))
        # answer the whole round's queries at once (parallel when asked)
        pre = []
        for s in sts:
            if stlive[s] and Ls[s] > NEG // 2:
                p2 = (int(Ls[s]), int(Rs[s]))
                if p2 not in killed2:
                    pre.append(p2)
        for e in idx.tolist():
            t = tuple(int(x) for x in tup[e])
            if t not in killed4:
                pre.append(t)
        oracle.batch(pre)
        for s in sts:
            if not stlive[s] or Ls[s] <= NEG // 2:
                continue
            p2 = (int(Ls[s]), int(Rs[s]))
            if p2 in killed2:
                continue
            asked2.add(p2)
            ans = oracle(p2)
            if shadow is not None:
                ref = shadow(p2)
                if ans is not None and ans != ref:
                    disagree.append(("pair", p2, ans, ref))
            if ans is False:
                killed2.add(p2)
                js = bypair.get(p2[0] * 128 + p2[1])
                if js is not None and len(js):
                    stlive[js] = False
                    Ls[js] = NEG
                    Rs[js] = NEG
                    dead[js] = True
                    live &= ~(dead[esrc] | dead[edst])
                progress = True
        for e in idx.tolist():
            t = tuple(int(x) for x in tup[e])
            if t in killed4:
                continue
            asked4.add(t)
            ans = oracle(t)
            if shadow is not None:
                ref = shadow(t)
                if ans is not None and ans != ref:
                    disagree.append(("quad", t, ans, ref))
            if ans is False:
                killed4.add(t)
                k = ((t[0] * 128 + t[1]) * 128 + t[2]) * 128 + t[3]
                js = by4.get(k)
                if js is not None:
                    live[js] = False
                progress = True
        if not progress:
            return dict(status="STALLED", it=it, y=y, bound=bnd,
                        q4=len(asked4), q2=len(asked2), k4=len(killed4),
                        k2=len(killed2), secs=time.time() - t0,
                        disagree=disagree, edges=int(live.sum()))
        if verbose and (it % 500 == 0 or it < 3):
            print("    it %6d  bound %4d  q %5d+%5d  killed %5d+%5d  "
                  "edges %8d  %6.0fs"
                  % (it, bnd, len(asked4), len(asked2), len(killed4),
                     len(killed2), int(live.sum()), time.time() - t0),
                  flush=True)
        if it > itcap:
            return dict(status="ITCAP", it=it, y=y, bound=bnd,
                        q4=len(asked4), q2=len(asked2),
                        secs=time.time() - t0, disagree=disagree)
    return dict(status="CERTIFIED", it=it, y=y, bound=bnd, budget=budget,
                q4=len(asked4), q2=len(asked2), k4=len(killed4),
                k2=len(killed2), secs=time.time() - t0, disagree=disagree,
                states=S, edges0=len(esrc), edges=int(live.sum()))


def report(r, orc, F, Q1, EXACT, shadowed=False):
    print("\n  RESULT: %s  bound %s   budget %d   (exact F(M+q') = %s)"
          % (r["status"], r.get("bound"), F + Q1, EXACT))
    print("  queries: %d of arity 4, %d of arity 2  (total %d), %d iterations"
          % (r.get("q4", 0), r.get("q2", 0),
             r.get("q4", 0) + r.get("q2", 0), r["it"]))
    print("  deleted: %d unrealised 4-tuples, %d unrealised pairs"
          % (r.get("k4", 0), r.get("k2", 0)))
    if orc.spans:
        sp = sorted(orc.spans)
        print("  ORACLE COST: %d calls, %.1f s total, %.1f ms mean, "
              "worst %.2f s on %s"
              % (orc.n, orc.secs, 1000 * orc.secs / max(1, orc.n),
                 orc.slowest[0], orc.slowest[1]))
        print("  QUERY SPANS: min %d  median %d  p90 %d  max %d   "
              "(F = %d, 2F = %d)"
              % (sp[0], sp[len(sp) // 2], sp[int(0.9 * len(sp))], sp[-1],
                 F, 2 * F))
        if orc.undecided:
            print("  UNDECIDED queries (node budget): %d - NONE of these "
                  "deleted anything" % orc.undecided)
    if r.get("disagree"):
        print("  *** ORACLE DISAGREEMENT with the round-24 dump: %d ***"
              % len(r["disagree"]))
        for d in r["disagree"][:10]:
            print("      ", d)
    elif shadowed:
        print("  shadow: CRT oracle and the round-24 dump agree on ALL "
              "%d queries" % (r.get("q4", 0) + r.get("q2", 0)))
    else:
        print("  (no shadow at this step - no scan and no dump exists)")
    print("  wall %.0f s" % r.get("secs", 0))


def main():
    args = sys.argv[1:]
    nb = 2_000_000
    if "--nodes" in args:
        nb = int(args[args.index("--nodes") + 1])
    if "--probe" in args:
        for y in [int(x) for x in args[args.index("--probe") + 1].split(",")]:
            F, Q1, EXACT = STEPS[y]
            t0 = time.time()
            S, esrc, _, _, _, _, _ = build_mf_edges(F, Q1, MOD, M)
            print("m%-2d F=%2d q'=%2d : MF_4 has %9d states %9d edges "
                  "(%.0fs)" % (y, F, Q1, S, len(esrc), time.time() - t0),
                  flush=True)
        return
    if "--shadow" in args:
        y = int(args[args.index("--shadow") + 1])
        F, Q1, EXACT = STEPS[y]
        print("=== SHADOW step %d -> %d :  F = %d, q' = %d, budget %d"
              % (y, Q1, F, Q1, F + Q1))
        orc = CRTOracle(y, nb)
        r = run_step(y, orc, shadow=DumpOracle(y))
        report(r, orc, F, Q1, EXACT, shadowed=True)
        assert r["status"] == "CERTIFIED", r["status"]
        assert not r["disagree"], "CRT oracle disagrees with the scan dump"
        print("\nall assertions passed")
        return
    topk = int(args[args.index("--topk") + 1]) if "--topk" in args else 1
    wk = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    f2 = int(args[args.index("--f2") + 1]) if "--f2" in args else 0
    ys = ([int(x) for x in args[args.index("--step") + 1].split(",")]
          if "--step" in args else [19, 23, 29, 31, 37, 41])
    rows = []
    for y in ys:
        F, Q1, EXACT = STEPS[y]
        print("\n=== step %d -> %d :  F = %d, q' = %d, budget %d, "
              "CRT ORACLE ONLY (no dump, no scan)"
              % (y, Q1, F, Q1, F + Q1), flush=True)
        orc = CRTOracle(y, nb, workers=wk)
        r = run_step(y, orc, topk=topk, f2=f2)
        report(r, orc, F, Q1, EXACT)
        rows.append((y, Q1, F + Q1, r, orc))
    print("\n\nTHE CHAIN")
    print("  step        F    q'  budget  result      bound  queries  "
          "oracle s   wall s")
    for y, Q1, bud, r, orc in rows:
        print("  %2d -> %-2d  %4d  %4d  %6d  %-10s %5s  %7d  %8.1f %8.0f"
              % (y, Q1, STEPS[y][0], Q1, bud, r["status"], r.get("bound"),
                 r.get("q4", 0) + r.get("q2", 0), orc.secs,
                 r.get("secs", 0)))


if __name__ == "__main__":
    main()

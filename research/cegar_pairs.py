"""Round 24 (constructor): R53's "one extra integer" was an ARTEFACT OF
REFINING EDGES ONLY - plus the slack sweep the brief asks for.

R53 ran counterexample-guided refinement on the machine-free system MF_4 at
29 -> 31.  Starting machine-free it stalled at 86 = 2F, because the maximising
abstract object was the EMPTY walk: layer 0 uses no edge, and the loop only
ever deleted unrealised 4-tuple EDGES.  Given the single integer F_2(29) = 55
it then certified in 6,395 oracle queries.

But layer 0 is not edge data - it is STATE data.  A state of MF_m is a value
(m-1)-tuple of consecutive gaps (plus a corridor phase and a tooth), and its
layer-0 contribution is the sum of its last two digits.  So "is this state
realised?" is a query OF EXACTLY THE SAME KIND as the edge queries, one arity
lower.  This script runs the refinement with BOTH kinds of query and measures:

  MODE --states : refine states (realised (m-1)-tuples) as well as edges, and
                  give the loop NO integer at all.  If it certifies, R53's
                  extra hypothesis disappears and the whole obligation at a
                  step is one dictionary of realisability facts.

  MODE --sweep  : feed the loop a two-gap bound U (a sound upper bound on
                  F_2(M), not necessarily the exact value) for U = 55.. and
                  find the LARGEST U that still certifies.  That measures how
                  much slack the two-gap statement has - i.e. how weak the
                  substitute fact is allowed to be.  Round-24 item 2
                  (research/survivor_generator.py) derives U = 57 for machine
                  29 from machine 23's own A_4 dictionary, so the sweep says
                  whether that derived bound is strong enough.

Usage: uv run python research/cegar_pairs.py --states
       uv run python research/cegar_pairs.py --sweep 55 90
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from machinefree_cert import build_mf_edges, NEG          # noqa: E402

DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# y : (F(M), q', exact F(M+q'), exact F_2(M))
STEPS = {19: (25, 23, 34, 31), 23: (34, 29, 43, 39), 29: (43, 31, 58, 55)}
Y = 29
DUMP = os.path.join(DDIR, "tuples4_29.txt")
F, Q1, EXACT, F2TRUE = STEPS[Y]
BUDGET = F + Q1
MOD = 35
M = 4
CAP = 300000


def set_step(y):
    global Y, DUMP, F, Q1, EXACT, F2TRUE, BUDGET
    Y = y
    DUMP = os.path.join(DDIR, "tuples4_%d.txt" % y)
    F, Q1, EXACT, F2TRUE = STEPS[y]
    BUDGET = F + Q1


def close_with(esrc, edst, ew, Rs, Ls, S, cap=64):
    hh = Rs.copy()
    for _ in range(cap):
        new = hh.copy()
        if len(esrc):
            np.maximum.at(new, esrc, ew + hh[edst])
        if np.array_equal(new, hh):
            break
        hh = new
    else:
        return None, None
    return hh, int((Ls + hh).max())


def walk_from(hh, Ls, esrc, edst, ew):
    st = int(np.argmax(Ls + hh))
    out = []
    order = np.argsort(esrc, kind="stable")
    ss = esrc[order]
    for _ in range(40):
        lo = int(np.searchsorted(ss, st, "left"))
        hi = int(np.searchsorted(ss, st, "right"))
        if lo == hi:
            break
        cand = order[lo:hi]
        gains = ew[cand] + hh[edst[cand]]
        j = int(cand[int(np.argmax(gains))])
        if int(ew[j] + hh[edst[j]]) < int(hh[st]):
            break
        out.append(j)
        st = int(edst[j])
    return st, out


def load_oracle():
    """Two independent oracles.

    o4 - realised gap 4-tuples of machine 29, from the round-23 A_4 dump
         (T3-filtered, which is exactly right for EDGE queries: every MF_4
         edge is T3-legal by construction, so 'in the dump' == 'realised').
    o2 - realised ADJACENT PAIRS of machine 29, from Mechanic's full-period
         lag-1 joint census.  NOT derivable from the T3-filtered dump (that
         would under-report and make state deletion UNSOUND), so it is read
         from research/data/gap_pair_joint.csv with the cyclic seam stitched
         exactly as research/twogap_table.py does.
    """
    import csv
    import collections
    o4 = set()
    for line in open(DUMP):
        o4.add(tuple(int(x) for x in line.split()))
    d = os.path.join(os.path.dirname(DUMP), "gap_pair_joint.csv")
    o2 = set()
    mu, mv, hist = collections.Counter(), collections.Counter(), \
        collections.Counter()
    for r in csv.DictReader(open(d)):
        if int(r["y"]) != Y or int(r["lag"]) != 1 or r["coverage"] != "1.000000":
            continue
        gu, gv, c = int(r["gu"]), int(r["gv"]), int(r["count"])
        o2.add((gu, gv))
        mu[gu] += c
        mv[gv] += c
    h = os.path.join(os.path.dirname(DUMP), "gap_pair_hist.csv")
    for r in csv.DictReader(open(h)):
        if int(r["y"]) == Y and r["kind"] == "ghist" and r["coverage"] == "1.000000":
            hist[int(r["value"])] = int(r["count"])
    du = [g for g in hist if hist[g] - mu[g] == 1]
    dv = [g for g in hist if hist[g] - mv[g] == 1]
    assert len(du) == 1 and len(dv) == 1, (du, dv)
    o2.add((du[0], dv[0]))                    # the cyclic seam pair
    assert max(u + v for u, v in o2) == F2TRUE, "F_2(M) from the pair oracle"
    return o4, o2


def build(F2=0):
    S, esrc, edst, ew, Rs, Ls, tup = build_mf_edges(F, Q1, MOD, M)
    Rs = Rs.copy()
    Ls = Ls.copy()
    if F2:
        okst = (Ls + Rs) <= F2
        keep = okst[esrc] & okst[edst]
        esrc, edst, ew, tup = esrc[keep], edst[keep], ew[keep], tup[keep]
        Ls = np.where(okst, Ls, NEG)
        Rs = np.where(okst, Rs, NEG)
    return S, esrc, edst, ew, Rs, Ls, tup


def state_triples(S, tup, esrc, edst):
    """The value 3-tuple of every state, recovered from the incident edges."""
    tri = np.full((S, 3), -1, np.int16)
    tri[esrc] = tup[:, :3]
    tri[edst] = tup[:, 1:4]
    return tri


def run_refine(F2=0, refine_states=False, verbose=True):
    o4, o2 = load_oracle()
    t0 = time.time()
    S, esrc, edst, ew, Rs, Ls, tup = build(F2)
    t64 = tup.astype(np.int64)
    key4 = ((t64[:, 0] * 64 + t64[:, 1]) * 64 + t64[:, 2]) * 64 + t64[:, 3]
    by4 = {}
    for i, k in enumerate(key4.tolist()):
        by4.setdefault(k, []).append(i)
    # a STATE's layer-0 content is its last two digits (Ls, Rs) = a pair of
    # consecutive gaps.  Group the states by that pair.
    pairkey = (Ls * 64 + Rs).astype(np.int64)
    bypair = {}
    for i, k in enumerate(pairkey.tolist()):
        if Ls[i] > NEG // 2:
            bypair.setdefault(k, []).append(i)
    bypair = {k: np.array(v, np.int64) for k, v in bypair.items()}
    live = np.ones(len(esrc), bool)
    stlive = np.ones(S, bool)
    asked4, asked3, killed4, killed3 = set(), set(), set(), set()
    it = 0
    bnd = None
    while True:
        it += 1
        hh, bnd = close_with(esrc[live], edst[live], ew[live], Rs, Ls, S)
        if hh is None:
            return dict(status="CYCLIC", it=it)
        if bnd <= BUDGET:
            return dict(status="CERTIFIED", it=it, bound=bnd,
                        q4=len(asked4), q3=len(asked3),
                        k4=len(killed4), k3=len(killed3),
                        secs=time.time() - t0)
        st, w = walk_from(hh, Ls, esrc[live], edst[live], ew[live])
        idx = np.flatnonzero(live)[w] if w else np.zeros(0, np.int64)
        progress = False
        if refine_states:
            # the state that attains the maximum, and every state on the walk
            sts = [int(np.argmax(Ls + hh))]
            for e in idx.tolist():
                sts.append(int(esrc[e]))
                sts.append(int(edst[e]))
            for s in sts:
                if not stlive[s] or Ls[s] <= NEG // 2:
                    continue
                p2 = (int(Ls[s]), int(Rs[s]))
                asked3.add(p2)
                if p2 not in o2:
                    if p2 not in killed3:
                        killed3.add(p2)
                        js = bypair.get(p2[0] * 64 + p2[1])
                        if js is not None and len(js):
                            stlive[js] = False
                            Ls[js] = NEG
                            Rs[js] = NEG
                            bad = np.isin(esrc, js) | np.isin(edst, js)
                            live &= ~bad
                    progress = True
        for e in idx.tolist():
            t = tuple(int(x) for x in tup[e])
            asked4.add(t)
            if t not in o4:
                progress = True
                if t in killed4:
                    continue
                killed4.add(t)
                k = ((t[0] * 64 + t[1]) * 64 + t[2]) * 64 + t[3]
                live[by4.get(k, [])] = False
        if not progress:
            return dict(status="STALLED", it=it, bound=bnd,
                        q4=len(asked4), q3=len(asked3),
                        k4=len(killed4), k3=len(killed3),
                        secs=time.time() - t0)
        if verbose and (it % 2000 == 0 or it < 4):
            print("   it %5d bound %3d  q4 %5d q3 %5d  killed %5d/%5d  "
                  "%6d edges  %.0fs"
                  % (it, bnd, len(asked4), len(asked3), len(killed4),
                     len(killed3), int(live.sum()), time.time() - t0),
                  flush=True)
        if it > CAP:
            return dict(status="CAP", it=it, bound=bnd, q4=len(asked4),
                        q3=len(asked3), secs=time.time() - t0)


def main():
    if "--step" in sys.argv:
        set_step(int(sys.argv[sys.argv.index("--step") + 1]))
    print("STEP %d -> %d :  F = %d, q' = %d, budget %d, exact F(M+q') = %d,"
          " exact F_2(M) = %d" % (Y, Q1, F, Q1, BUDGET, EXACT, F2TRUE))
    if "--states" in sys.argv:
        print("MODE --states: refine STATES (realised 3-tuples) and EDGES "
              "(realised 4-tuples); NO integer given.")
        r = run_refine(F2=0, refine_states=True)
        print("\nRESULT: %s at bound %s after %d iterations"
              % (r["status"], r.get("bound"), r["it"]))
        print("  oracle queries: %d of arity 4, %d of arity 2  (total %d)"
              % (r.get("q4", 0), r.get("q3", 0),
                 r.get("q4", 0) + r.get("q3", 0)))
        print("  unrealised deleted: %d 4-tuples, %d PAIRS, %.0f s"
              % (r.get("k4", 0), r.get("k3", 0), r.get("secs", 0)))
        # control: the same run WITHOUT state refinement must stall at 2F
        print("\nCONTROL (edges only, no integer - reproduces R53 run 1):")
        r0 = run_refine(F2=0, refine_states=False, verbose=False)
        print("  %s at bound %s after %d iterations, %d queries"
              % (r0["status"], r0.get("bound"), r0["it"], r0.get("q4", 0)))
        assert r0["status"] in ("STALLED", "CERTIFIED"), r0
        print("  (2F = %d - the histogram/corridor wall of round-24 item 1)"
              % (2 * F))
    if "--sweep" in sys.argv:
        i = sys.argv.index("--sweep")
        lo, hi = int(sys.argv[i + 1]), int(sys.argv[i + 2])
        print("MODE --sweep: feed a SOUND upper bound U on F_2(29) and see "
              "whether (D) at 29->31 still certifies.")
        print("   U   status     bound  iters  queries   secs")
        best = None
        for U in range(lo, hi + 1):
            r = run_refine(F2=U, refine_states=False, verbose=False)
            print("  %3d  %-9s  %5s  %5d  %7d  %5.0f"
                  % (U, r["status"], r.get("bound"), r["it"],
                     r.get("q4", 0), r.get("secs", 0)))
            if r["status"] == "CERTIFIED":
                best = U
        print("\n  LARGEST U that still certifies: %s   (budget F+q' = %d, "
              "exact F_2(29) = 55)" % (best, BUDGET))


if __name__ == "__main__":
    main()

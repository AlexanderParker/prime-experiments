"""Round 26 (constructor): THE 37->41 RUNG, with the oracle cost paid up front.

R62b left 37 -> 41 uncertified.  The measured reason was NOT the state space
(P7 refuted: MF_4 at (88,41) builds in 15 s) but the ORACLE: an over-budget
arity-2 refutation at machine 37 costs 5.8 s mean / 23.7 s worst against 43 ms
for a 4-tuple, and the loop asks thousands of them one at a time.

THE FIX, in two phases, each sound on its own terms.

  PHASE 1 - run the CEGAR loop with an EXACT SCANNED dictionary as the oracle.
  Mechanic's research/data/gap_tuples_37_4.csv is the full-period realised
  4-tuple census of machine 37 (291,675 tuples, both assertions passed:
  openings = prod(q-2) and max gap = 88).  It is EXACT, so "not in the
  dictionary" is a valid refutation.  Its induced level-2 projection is the
  exact realised-PAIR set of machine 37 - every pair of consecutive gaps sits
  inside some 4-window - so it answers the arity-2 queries too, in O(1).
  This phase certifies (D) at the step MODULO THE SCAN.

  PHASE 2 - re-prove every DELETION the loop made with the scan-free CRT
  oracle (research/crt_dict.py).  The set of deletions is now known in
  advance, so they can be BATCHED and run in parallel instead of discovered
  one at a time inside the loop.  If every deletion is independently refuted
  by CRT, the certificate is scan-free: the scan was used only to CHOOSE
  which refutations to attempt.

Soundness note.  Phase 1 is a certificate relative to the scanned dictionary;
phase 2 removes that relativisation deletion by deletion.  A deletion that
phase 2 leaves UNDECIDED (node budget) is reported, not hidden - the phase-1
bound then stands only relative to the scan.

Usage:
  python research/chain_dict_oracle.py --step 37 --topk 64 --phase 1
  python research/chain_dict_oracle.py --verify 37 --workers 4
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_cegar                                       # noqa: E402
import crt_dict                                          # noqa: E402

DDIR = os.path.join(HERE, "data")
DICT_CSV = {23: "gap_tuples_23_4.csv", 29: "gap_tuples_29_4.csv",
            31: "gap_tuples_31_4.csv", 37: "gap_tuples_37_4.csv"}


def load_exact_dict(y):
    """The full-period realised 4-tuple census, plus its induced levels.

    ASSERTED: the induced level-1 set's max is F(y) and the induced level-2
    set's max sum is F_2(y) - both against the corpus - so the file really is
    the dictionary it claims to be before anything is deleted on its say-so.
    """
    path = os.path.join(DDIR, DICT_CSV[y])
    d4 = set()
    with open(path) as f:
        for r in csv.reader(f):
            if r[0] == "g1":
                continue
            d4.add(tuple(int(x) for x in r))
    d3, d2, d1 = set(), set(), set()
    for t in d4:
        d3.add(t[:3])
        d3.add(t[1:])
        d2.add(t[:2])
        d2.add(t[1:3])
        d2.add(t[2:])
        for v in t:
            d1.add((v,))
    F = max(v[0] for v in d1)
    F2 = max(sum(p) for p in d2)
    assert F == crt_dict.KNOWN_F[y], (y, F, crt_dict.KNOWN_F[y])
    assert F2 == crt_dict.KNOWN_F2[y], (y, F2, crt_dict.KNOWN_F2[y])
    return {1: d1, 2: d2, 3: d3, 4: d4}, F, F2


def _enc(t):
    k = 0
    for v in t:
        k = k * 128 + int(v)
    return k


class SupersetDictOracle:
    """Machine 41 (mechanic round 25): the TRANSFER dictionary
    gap_tuples_41_4_transfer.csv is a SUPERSET of machine 41's realised
    4-tuples - built from the exact m37 dictionary by the deletion transfer,
    exact at depth 1 (its induced 1-tuple set is {1..91} minus {84,87,89},
    reproducing F(41) = 91 and COV-SAT's complete m41 hole list).

    A SUPERSET IS ENOUGH FOR A CERTIFICATE, and that is the point: the loop
    only ever ACTS on a NO.  "t not in a superset of the realised set" implies
    "t is not realised", so every deletion licensed here is sound.  A YES is
    merely a refusal to delete, which never breaks soundness - it only makes
    the certificate weaker.  So the rung either certifies (a true certificate)
    or does not (no conclusion), and it can never certify wrongly.

    MEMORY (round 26, learned the hard way - a 4.2M-entry Python set of ints
    plus MF_4's index dictionaries put the first hybrid run over the box's
    commit limit and it was killed silently at 12:35, with two other lanes
    running).  Levels are held as SORTED numpy int64 ARRAYS of base-128 codes
    and queried by searchsorted: 4.2M tuples cost 34 MB instead of ~500 MB.
    """

    def __init__(self, y=41, path=None):
        path = path or os.path.join(DDIR, "gap_tuples_41_4_transfer.csv")
        self.y = y
        e4, e3, e2, d1 = [], [], [], set()
        with open(path) as f:
            for line in f:
                if line[0] == "g":
                    continue
                t = tuple(int(x) for x in line.split(","))
                e4.append(_enc(t))
                e3.append(_enc(t[:3]))
                e3.append(_enc(t[1:]))
                e2.append(_enc(t[:2]))
                e2.append(_enc(t[1:3]))
                e2.append(_enc(t[2:]))
                for v in t:
                    d1.add(v)
        self.D = {1: d1,
                  2: np.unique(np.array(e2, np.int64)),
                  3: np.unique(np.array(e3, np.int64)),
                  4: np.unique(np.array(e4, np.int64))}
        del e2, e3, e4
        self.F = max(d1)
        self.F2 = int(max((k // 128) + (k % 128) for k in self.D[2].tolist()))
        self.n = 0
        self.spans = []
        self.secs = 0.0
        self.undecided = 0
        self.slowest = (0.0, None)
        self.memo = {}

    def batch(self, tups):
        return

    def __call__(self, tup):
        self.n += 1
        self.spans.append(sum(tup))
        if len(tup) == 1:
            return tup[0] in self.D[1]
        arr = self.D.get(len(tup))
        if arr is None:
            return True                 # no information at this arity
        k = _enc(tup)
        i = int(np.searchsorted(arr, k))
        return i < len(arr) and int(arr[i]) == k

    def close(self):
        return


class HybridOracle:
    """SUPERSET first (free), scan-free CRT second (exact).

    Measured at machine 41 (round 26): of the tuples the superset dictionary
    calls REALISED, the CRT decision refutes 12 of 12 sampled 4-tuples - the
    transfer dictionary is heavily inflated at arity 4 - so the superset acts
    as a free pre-filter that removes the cheap refutations and leaves the CRT
    solver only the ones it has to work for.  A superset NO is already a
    proof, so it is returned immediately; a superset YES is passed to CRT,
    which is exact.  The composite oracle is therefore EXACT, and every
    deletion is a genuine refutation.
    """

    def __init__(self, y, node_budget=4_000_000, workers=1):
        self.sup = SupersetDictOracle(y)
        self.y = y
        self.nb = node_budget
        self.D = self.sup.D
        self.F, self.F2 = self.sup.F, self.sup.F2
        self.memo = {}
        self.n = 0
        self.ncrt = 0
        self.spans = []
        self.secs = 0.0
        self.undecided = 0
        self.slowest = (0.0, None)
        self.pool = None
        if workers > 1:
            from multiprocessing import Pool
            self.pool = Pool(workers, initializer=_vinit,
                             initargs=(y, node_budget))

    def _sup_no(self, t):
        return not self.sup.__call__(t)

    def batch(self, tups):
        need = []
        for t in dict.fromkeys(tups):
            if t in self.memo:
                continue
            if self._sup_no(t):
                self.memo[t] = False
            else:
                need.append(t)
        if not need:
            return
        t0 = time.time()
        if self.pool is not None:
            res = self.pool.map(_vwork, need, chunksize=1)
        else:
            _vinit(self.y, self.nb)
            res = [_vwork(t) for t in need]
        self.secs += time.time() - t0
        for t, r, d in res:
            self.memo[t] = r
            self.ncrt += 1
            if r is None:
                self.undecided += 1
            if d > self.slowest[0]:
                self.slowest = (d, t)

    def __call__(self, tup):
        self.n += 1
        self.spans.append(sum(tup))
        if tup in self.memo:
            return self.memo[tup]
        if self._sup_no(tup):
            self.memo[tup] = False
            return False
        _vinit(self.y, self.nb)
        t, r, d = _vwork(tup)
        self.secs += d
        self.ncrt += 1
        self.memo[tup] = r
        if r is None:
            self.undecided += 1
        if d > self.slowest[0]:
            self.slowest = (d, tup)
        return r

    def close(self):
        if self.pool is not None:
            self.pool.terminate()


class ExactDictOracle:
    """Answers any query of arity 1..4 from the exact full-period census."""

    def __init__(self, y):
        self.y = y
        self.D, self.F, self.F2 = load_exact_dict(y)
        self.n = 0
        self.spans = []
        self.secs = 0.0
        self.undecided = 0
        self.slowest = (0.0, None)
        self.memo = {}

    def batch(self, tups):
        return

    def __call__(self, tup):
        self.n += 1
        self.spans.append(sum(tup))
        v = tup in self.D[len(tup)]
        self.memo[tup] = v
        return v

    def close(self):
        return


# ------------------------------------------------------------- phase 2
_VY = None
_VNB = None


def _vinit(y, nb):
    global _VY, _VNB
    _VY, _VNB = y, nb


def _vwork(t):
    t0 = time.time()
    try:
        r = crt_dict.realised(_VY, t, _VNB)
    except crt_dict.Budget:
        r = None
    return t, r, time.time() - t0


def verify(y, workers=1, node_budget=4_000_000, path=None):
    """Phase 2: CRT-refute every deletion phase 1 made."""
    path = path or os.path.join(DDIR, "chain_dict_%d.json" % y)
    rec = json.load(open(path))
    dels = [tuple(t) for t in rec["killed2"]] + \
           [tuple(t) for t in rec["killed4"]]
    print("phase 2: %d deletions to re-prove by CRT (%d pairs, %d 4-tuples)"
          % (len(dels), len(rec["killed2"]), len(rec["killed4"])), flush=True)
    t0 = time.time()
    if workers <= 1:
        _vinit(y, node_budget)
        res = [_vwork(t) for t in dels]
    else:
        from multiprocessing import Pool
        with Pool(workers, initializer=_vinit,
                  initargs=(y, node_budget)) as pool:
            res = []
            for i, r in enumerate(pool.imap_unordered(_vwork, dels,
                                                      chunksize=1)):
                res.append(r)
                if (i + 1) % 25 == 0:
                    print("    %d/%d  %.0fs" % (i + 1, len(dels),
                                                time.time() - t0), flush=True)
    bad = [(t, r) for t, r, _ in res if r is True]
    und = [t for t, r, _ in res if r is None]
    costs = sorted(d for _, _, d in res)
    print("\nPHASE 2 RESULT for machine %d" % y)
    print("  deletions re-proved unrealised by CRT: %d of %d"
          % (len(dels) - len(bad) - len(und), len(dels)))
    print("  CONTRADICTIONS (scan said no, CRT says yes): %d %s"
          % (len(bad), bad[:5]))
    print("  UNDECIDED at %d nodes: %d %s" % (node_budget, len(und), und[:5]))
    if costs:
        print("  cost: total %.0f s, mean %.2f s, median %.2f s, worst %.1f s"
              % (sum(costs), sum(costs) / len(costs),
                 costs[len(costs) // 2], costs[-1]))
    assert not bad, "the scanned dictionary and the CRT oracle DISAGREE"
    rec["verified"] = len(dels) - len(und)
    rec["undecided_verify"] = [list(t) for t in und]
    json.dump(rec, open(path, "w"))
    print("  wall %.0f s" % (time.time() - t0))
    return not und


def main():
    args = sys.argv[1:]
    if "--verify" in args:
        y = int(args[args.index("--verify") + 1])
        wk = int(args[args.index("--workers") + 1]) if "--workers" in args \
            else 1
        nb = int(args[args.index("--nodes") + 1]) if "--nodes" in args \
            else 4_000_000
        pa = args[args.index("--path") + 1] if "--path" in args else None
        verify(y, wk, nb, pa)
        return
    y = int(args[args.index("--step") + 1]) if "--step" in args else 37
    topk = int(args[args.index("--topk") + 1]) if "--topk" in args else 64
    F, Q1, EXACT = chain_cegar.STEPS[y]
    wk = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    if "--hybrid" in args:
        orc = HybridOracle(y, workers=wk)
    elif "--superset" in args:
        orc = SupersetDictOracle(y)
    else:
        orc = ExactDictOracle(y)
    print("=== step %d -> %d  F = %d  q' = %d  budget %d" % (y, Q1, F, Q1,
                                                             F + Q1))
    print("  %s dictionary: |D_1| = %d, |D_2| = %d, |D_3| = %d, "
          "|D_4| = %d;  F = %d, F_2 <= %d"
          % ("SUPERSET" if "--superset" in args else "exact",
             len(orc.D[1]), len(orc.D[2]), len(orc.D[3]), len(orc.D[4]),
             orc.F, orc.F2), flush=True)
    t0 = time.time()
    # --nof2 : do NOT pre-filter states by the two-gap number.  Every pair
    # deletion is then discovered and justified individually by the oracle,
    # so the rung consumes NO given integer at all (R58's "CEGAR needs no
    # integer") - at the price of many more arity-2 refutations to re-prove
    # in phase 2.
    f2 = 0 if "--nof2" in args else orc.F2
    if "--f2" in args:
        f2 = int(args[args.index("--f2") + 1])
    r = chain_cegar.run_step(y, orc, topk=topk, f2=f2)
    chain_cegar.report(r, orc, F, Q1, EXACT)
    tag = ("_nof2" if "--nof2" in args else "") + \
        ("_sup" if "--superset" in args else "") + \
        ("_hyb" if "--hybrid" in args else "")
    out = os.path.join(DDIR, "chain_dict_%d%s.json" % (y, tag))
    json.dump(dict(y=y, status=r["status"], bound=r.get("bound"),
                   budget=F + Q1, it=r["it"], topk=topk,
                   q4=r.get("q4", 0), q2=r.get("q2", 0),
                   killed4=[list(t) for t in r.get("killed4", [])],
                   killed2=[list(t) for t in r.get("killed2", [])],
                   asked4=[list(t) for t in r.get("asked4", [])],
                   asked2=[list(t) for t in r.get("asked2", [])],
                   secs=time.time() - t0), open(out, "w"))
    print("\ndeletions written to %s" % out)


if __name__ == "__main__":
    main()

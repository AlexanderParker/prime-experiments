"""Round 27 (mechanic): THE EXACT m41 4-TUPLE CENSUS, BY ASCENDING SPAN.

Brief item (a).  The full exact census is PRICED AT ~4.0e6 core-seconds by the
CRT route and >= 2e5 core-seconds by the period route
(research/price_m41_census_r27.py) - both far past one round.  So this delivers
the largest SELF-CONTAINED SHARD: every candidate decided EXACTLY, in
ASCENDING SPAN ORDER, so that at any stopping point the deliverable is

    "the exact machine-41 4-tuple dictionary is COMPLETE at every span <= S",

with S read off the logs rather than asserted.  Above the frontier the
screened superset still stands as a certified superset (C31), and above
F_4(41) every candidate is zero by the deletion-ladder theorem.

CANDIDATES.  research/data/r26/gap_tuples_41_4_screened.csv - the round-26
phase-saturation screen of Constructor's dict_transfer superset.  A superset
of the truth (K4 + C31), so a NO here is a genuine NO and a YES here is a
genuine YES: the decision is exact either way, the superset only bounds WHICH
tuples must be asked about.

MIRROR HALVING (rule 27).  #occ(w) = #occ(reverse w) exactly, so only one
tuple per reverse class is decided and the verdict is copied.  The superset's
reverse-closure is ASSERTED, not assumed.

DECIDER.  crt_dict.decide_cover - Constructor's exact set-cover CSP over the
eleven gears.  A Budget exception is recorded as UNKNOWN and NEVER as a
deletion (rule: never delete on an unknown).

Each worker walks classes i == w (mod W) of the span-ascending order, appends
to its own log, and resumes from that log.  Every worker stops at a wall
deadline so the round can close with the job finished (job-completion rule).

Usage:
  worker : python research/m41_census_r27.py work W w SPAN_MAX HOURS
  merge  : python research/m41_census_r27.py merge W SPAN_MAX
  gate   : python research/m41_census_r27.py gate
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crt_dict import decide_cover, gears_of, Budget          # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "r27")
SCREENED = os.path.join(DATA, "r26", "gap_tuples_41_4_screened.csv")

B = 100          # gaps are < 100 at machine 41 (F(41) = 91), so base 100 packs


def load_csv(path, head="g1,g2,g3,g4"):
    out = []
    with open(path) as fh:
        h = fh.readline().strip()
        assert h == head, (path, h)
        for line in fh:
            line = line.strip()
            if line:
                out.append(tuple(int(x) for x in line.split(",")))
    return out


def load_arr(path):
    """(n,4) int16 array of the CSV - ~23 MB for the m41 superset, against
    ~600 MB for a python list of tuples (the round-26 memory lesson)."""
    a = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int16)
    assert a.ndim == 2 and a.shape[1] == 4, a.shape
    assert a.max() < B, a.max()
    return a


def _key(a):
    a = a.astype(np.int64)
    return ((a[:, 0] * B + a[:, 1]) * B + a[:, 2]) * B + a[:, 3]


def ordered_candidates(span_max):
    """Reverse-class representatives with span <= span_max, ascending span.

    Reverse-closure of the candidate set is ASSERTED (Lateral's mirror law is
    what licenses deciding one per class; if the superset were not closed the
    halving would lose tuples)."""
    a = load_arr(SCREENED)
    k = _key(a)
    kr = _key(a[:, ::-1])
    assert np.isin(kr, k, assume_unique=False).all(), \
        "candidate set is not reverse-closed"
    keep = (k <= kr) & (a.sum(axis=1) <= span_max)
    a = a[keep]
    order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0], a.sum(axis=1)))
    return a[order]


def as_tuples(a):
    return [tuple(int(x) for x in row) for row in a]


def make_XY(gaps):
    X = [0]
    for g in gaps:
        X.append(X[-1] + g)
    xs = set(X)
    return X, [t for t in range(1, X[-1]) if t not in xs]


def decide(qs, gaps, node_budget):
    X, Y = make_XY(gaps)
    try:
        ok, _, _ = decide_cover(qs, X, Y, node_budget=node_budget)
    except Budget:
        return "U"
    return "Y" if ok else "N"


# ------------------------------------------------------------------- worker
def work(W, w, span_max, hours, node_budget=2_000_000):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "m41cen_w%d_%d.log" % (w, W))
    cands = ordered_candidates(span_max)
    mine = as_tuples(cands[w::W])          # this worker's share only
    del cands
    done = set()
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 2 and p[1] in "YNU":
                done.add(tuple(int(x) for x in p[0].split(",")))
    qs = gears_of(41)
    todo = [t for t in mine if t not in done]
    # rule 28: progress stride from THIS worker's own share
    stride = max(1, len(mine) // 20)
    deadline = time.time() + hours * 3600.0
    print("worker %d/%d: share %d, already decided %d, todo %d, span <= %d, "
          "deadline %.2f h, stride %d"
          % (w, W, len(mine), len(done), len(todo), span_max, hours, stride),
          flush=True)
    fh = open(path, "a", buffering=1)
    t0 = time.time()
    n = 0
    ny = nn = nu = 0
    for t in todo:
        if time.time() > deadline:
            print("  DEADLINE reached after %d decisions" % n, flush=True)
            break
        v = decide(qs, t, node_budget)
        fh.write("%s %s\n" % (",".join(str(x) for x in t), v))
        n += 1
        if v == "Y":
            ny += 1
        elif v == "N":
            nn += 1
        else:
            nu += 1
        if n % stride == 0:
            print("  %d/%d  span=%d  Y=%d N=%d U=%d  t=%.0fs"
                  % (n, len(todo), sum(t), ny, nn, nu, time.time() - t0),
                  flush=True)
    fh.close()
    print("worker %d/%d done: %d decided this run (Y=%d N=%d U=%d) in %.0fs"
          % (w, W, n, ny, nn, nu, time.time() - t0), flush=True)


# -------------------------------------------------------------------- merge
def merge(W, span_max):
    cands = as_tuples(ordered_candidates(span_max))
    verdict = {}
    for w in range(W):
        path = os.path.join(OUT, "m41cen_w%d_%d.log" % (w, W))
        if not os.path.exists(path):
            continue
        for line in open(path):
            p = line.split()
            if len(p) == 2 and p[1] in "YNU":
                t = tuple(int(x) for x in p[0].split(","))
                prev = verdict.get(t)
                assert prev is None or prev == p[1], ("verdict clash", t)
                verdict[t] = p[1]

    # FRONTIER: the largest S with every class of span <= S decided.
    by_span = {}
    for t in cands:
        by_span.setdefault(sum(t), []).append(t)
    frontier = 0
    for s in sorted(by_span):
        if all(t in verdict and verdict[t] != "U" for t in by_span[s]):
            frontier = s
        else:
            break

    ny = sum(1 for v in verdict.values() if v == "Y")
    nn = sum(1 for v in verdict.values() if v == "N")
    nu = sum(1 for v in verdict.values() if v == "U")
    print("decided reverse classes: %d  (Y %d / N %d / UNKNOWN %d)"
          % (len(verdict), ny, nn, nu))
    print("COMPLETE-BY-SPAN FRONTIER: every 4-tuple of span <= %d is decided"
          % frontier)

    # emit the exact dictionary below the frontier, both orientations
    rows = set()
    for t, v in verdict.items():
        if v == "Y" and sum(t) <= frontier:
            rows.add(t)
            rows.add(t[::-1])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "gap_tuples_41_4_exact_le%d.csv" % frontier)
    with open(out, "w") as fh:
        fh.write("g1,g2,g3,g4\n")
        for t in sorted(rows):
            fh.write("%d,%d,%d,%d\n" % t)
    # the same region of the screened superset, for the inflation figure
    sa = load_arr(SCREENED)
    n_sup = int((sa.sum(axis=1) <= frontier).sum())
    print("EXACT dictionary at span <= %d : %d tuples   (wrote %s)"
          % (frontier, len(rows), out))
    print("screened SUPERSET over the same region: %d tuples -> inflation "
          "%.4fx" % (n_sup, n_sup / max(1, len(rows))))
    # per-span table
    print("\n  span   candidates   realised   refuted")
    tot_c = tot_y = tot_n = 0
    for s in sorted(by_span):
        if s > frontier:
            break
        c = len(by_span[s])
        y = sum(1 for t in by_span[s] if verdict[t] == "Y")
        tot_c += c
        tot_y += y
        tot_n += c - y
        if s % 5 == 0 or c > 200:
            print("  %4d %12d %10d %9d" % (s, c, y, c - y))
    print("  TOTAL (reverse classes) %d: realised %d, refuted %d"
          % (tot_c, tot_y, tot_n))
    return frontier


# --------------------------------------------------------------------- gate
def gate():
    """Two-sided soundness gate on the decider, at machines whose exact
    4-tuple dictionary this lane scanned in full (C21).

    POSITIVE control: every tuple of the exact dictionary must decide YES.
    NEGATIVE control: a tuple absent from the exact dictionary, but with every
    gap a realised gap value of the machine, must decide NO.
    (Formalist's round-26 lesson: an audit without positive controls is not an
    audit.)
    """
    import random
    t0 = time.time()
    rng = random.Random(4127)
    print("DECIDER GATE - two-sided, against this lane's own full-period "
          "4-tuple censuses (C21)\n")
    for y, fn, n_pos, n_neg in ((23, "gap_tuples_23_4.csv", 15696, 2000),
                                (29, "gap_tuples_29_4.csv", 3000, 3000),
                                (31, "gap_tuples_31_4.csv", 2000, 2000)):
        exact = load_csv(os.path.join(DATA, fn))
        S = set(exact)
        qs = gears_of(y)
        # positive
        pos = exact if len(exact) <= n_pos else rng.sample(exact, n_pos)
        bad = 0
        for t in pos:
            if decide(qs, t, 2_000_000) != "Y":
                bad += 1
        assert bad == 0, ("positive control failed", y, bad)
        # negative: random 4-tuples over the machine's realised gap VALUES
        vals = sorted({g for t in exact for g in t})
        neg = []
        while len(neg) < n_neg:
            t = tuple(rng.choice(vals) for _ in range(4))
            if t not in S:
                neg.append(t)
        badn = 0
        for t in neg:
            if decide(qs, t, 2_000_000) != "N":
                badn += 1
        assert badn == 0, ("negative control failed", y, badn)
        print("  m%-2d  positive %6d/%6d YES   negative %6d/%6d NO   (%d "
              "gears)" % (y, len(pos), len(pos), len(neg), len(neg), len(qs)))
    print("\nALL ASSERTIONS PASSED  (%.0fs)" % (time.time() - t0))


def main():
    cmd = sys.argv[1]
    if cmd == "work":
        work(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
             float(sys.argv[5]))
    elif cmd == "merge":
        merge(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "gate":
        gate()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

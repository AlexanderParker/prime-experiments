"""Round 28 (constructor): THE COVER-HALF ORDER N(M), SCAN-FREE.

R75 posed the order question correctly and then showed the corridor cannot
answer it.  N(M) is the smallest m at which the history abstraction A_m is
ACYCLIC - the smallest order at which a bounded-state certificate can bound
anything at all.  A cycle in A_m is an infinite periodic word of gaps, all of
them T3-legal kill letters, every m-window of which is REALISED by M.  So

    N(M)  <=  1 + L(M),        L(M) = longest realised fully-legal word,

and L(M) = A_res(M) - 1 is decided by the COVER half of the realisability CSP
("every interior slot blocked"), which no bounded gear set supplies (R75).
R75 measured N by hand from dictionaries at m11..m37 and stopped: "NOT
ATTEMPTED THIS ROUND, and it is the natural round-28 item for this lane."

THIS FILE ANSWERS IT WITH THE CRT COUNTER, so it reaches machines no census
does.  The construction is small because legality is severe: at values <= F(M)
there are only a handful of legal letters, T3 forbids most sequences of them,
and the spectrum F_2..F_4 kills most of what is left - so the whole graph is
built from a few hundred exact CRT decisions.

  * enumerate every T3-legal word of length m over the legal letters <= F(M),
    pruned by the spectrum (a sub-window of j consecutive gaps sums to <= F_j);
  * decide each by phase saturation (free) then crt_dict.decide_cover (exact);
  * A_m = the de Bruijn graph on realised legal m-tuples, nodes the
    (m-1)-prefixes; report the first m with no cycle.

GATE: N(M) must reproduce R75's hand-computed row 2,2,2,3,2,3,4,3 at
m11..m37 - a completely different vehicle (there: dictionaries from full-period
scans; here: CRT arithmetic from the gear list).

Usage:  .venv/Scripts/python.exe research/cover_order.py [--upto 47]
                                                         [--workers 4]
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                          # noqa: E402
from perj_scanfree import (next_prime, gears_of, exposed, ps_refuted,  # noqa
                           spectrum, spec_ok, job)

KNOWN_F = crt_dict.KNOWN_F
# R75's hand-computed row, the gate
N_R75 = {11: 2, 13: 2, 17: 2, 19: 3, 23: 2, 29: 3, 31: 4, 37: 3}
A_RES = {11: 2, 13: 2, 17: 2, 19: 3, 23: 3, 29: 4, 31: 4, 37: 4}


def legal_words(q1, a, b, F, vals, n, Fspec):
    """Every T3-legal word of length n, values <= F, pruned by the spectrum."""
    LV = []
    for v in range(1, F + 1):
        if vals is not None and v not in vals:
            continue
        r = v % q1
        if r == 0:
            LV.append((v, 0))
        elif r == a % q1:
            LV.append((v, 1))
        elif r == b % q1:
            LV.append((v, -1))
    words = [[]]
    for _ in range(n):
        nxt = []
        for w in words:
            last = next((c for _, c in reversed(w) if c), 0)
            for v, c in LV:
                if c and c == last:
                    continue
                cand = w + [(v, c)]
                if spec_ok([x for x, _ in cand], Fspec):
                    nxt.append(cand)
        words = nxt
    return [tuple(v for v, _ in w) for w in words]


def has_cycle(tuples, m):
    """Cycle in the de Bruijn graph on realised legal m-tuples."""
    if m == 1:
        # a state is the empty prefix; any realised legal letter loops on it
        return [tuples[0]] if tuples else None
    adj = {}
    for t in tuples:
        adj.setdefault(t[:-1], set()).add(t[1:])
    colour = {}
    stack = []

    def dfs(u):
        colour[u] = 1
        for v in adj.get(u, ()):  # noqa
            if colour.get(v) == 1:
                stack.append(v)
                return True
            if colour.get(v) is None and dfs(v):
                if len(stack) < 40:
                    stack.append(v)
                return True
        colour[u] = 2
        return False

    for u in list(adj):
        if colour.get(u) is None and dfs(u):
            return stack
    return None


def realised_legal(y, words, gears, E, workers, nodes):
    live, free_kill = [], 0
    for w in words:
        X, acc = [0], 0
        for v in w:
            acc += v
            X.append(acc)
        if ps_refuted(X, gears, E):
            free_kill += 1
        else:
            live.append(w)
    if not live:
        return [], free_kill, 0
    if workers > 1 and len(live) > 8:
        with Pool(workers) as pool:
            res = pool.map(job, [(y, w, nodes) for w in live], chunksize=1)
    else:
        res = [job((y, w, nodes)) for w in live]
    yes = [w for w, ok, dt in res if ok]
    und = sum(1 for w, ok, dt in res if ok is None)
    return yes, free_kill, und


def main():
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    upto = opt("--upto", 47)
    workers = opt("--workers", 4)
    nodes = opt("--nodes", 8_000_000)
    mcap = opt("--mcap", 7)
    print("=" * 78)
    print("THE COVER-HALF ORDER N(M) - smallest m with A_m ACYCLIC, scan-free")
    print("=" * 78)
    print("   M    q'   F | m : #legal words  free-killed  realised  cycle?")
    out = {}
    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        if y > upto:
            break
        q1 = next_prime(y)
        u1 = round(q1 / 6)
        a, b = 2 * u1, q1 - 2 * u1
        F = KNOWN_F[y]
        Fspec, vals, exact = spectrum(y)
        if Fspec is None:
            Fspec, vals, exact = {1: F}, None, False
        gears = gears_of(y)
        E = {g: exposed(g) for g in gears}
        print("  %3d %5d %3d | letters (%d,%d)  spectrum %s  [%s]"
              % (y, q1, F, a, b, [Fspec.get(j) for j in (1, 2, 3, 4)],
                 "exact" if exact else "upper bound"))
        N, L, cyc = None, 0, None
        t0 = time.time()
        for m in range(1, mcap + 1):
            words = legal_words(q1, a, b, F, vals, m, Fspec)
            if not words:
                print("        m=%d : 0 legal words at all  ->  ACYCLIC" % m)
                N = N or m
                break
            yes, fk, und = realised_legal(y, words, gears, E, workers, nodes)
            c = has_cycle(yes, m) if yes else None
            print("        m=%d : %5d words, %5d free-killed, %4d realised, "
                  "%s%s   [%.0f s]"
                  % (m, len(words), fk, len(yes),
                     "CYCLIC" if c else "acyclic",
                     " (%d undecided)" % und if und else "",
                     time.time() - t0))
            if yes:
                L = m
            if not c:
                N = m
                break
            cyc = c
        out[y] = (N, L, cyc)
        gate = N_R75.get(y)
        if gate is not None:
            assert N == gate, ("N(M) GATE against R75", y, N, gate)
        print("        => N(%d) = %s   (longest realised legal word L = %d, "
              "so N <= L+1 = %d)%s"
              % (y, N, L, L + 1,
                 "   GATE OK vs R75" if gate is not None else "   [NEW]"))
        if cyc:
            print("           last cycle seen at m = %d: %s"
                  % (N - 1, " -> ".join(str(list(s)) for s in cyc[:4])))
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("   M      11 13 17 19 23 29 31 37 41 43 47")
    for nm, d in (("N(M)", {y: out[y][0] for y in out}),
                  ("L(M)", {y: out[y][1] for y in out}),
                  ("A_res-1 (R45)", {y: A_RES[y] - 1 for y in A_RES
                                     if y in out})):
        print("  %-14s %s" % (nm, " ".join("%2s" % d.get(y, "-")
                                           for y in (11, 13, 17, 19, 23, 29,
                                                     31, 37, 41, 43, 47))))
    print("\nall assertions passed")


if __name__ == "__main__":
    main()

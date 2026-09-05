"""sr_holes.py -- why a size is ABSENT rather than merely depleted.

By the recursion a size v is realised at M+q' iff either m_M(v) > 0 (an old gap survives,
c_q'(v) >= q'-4 >= 1 copies) or some J >= 2 run of M of span v is struck legally with both ends
open in some copy.  So an absent size is one of two things:
  * a SPAN hole: M has no run of consecutive gaps of that span at all;
  * a PHASE hole: runs of that span exist, but every one of them has weight 0 -- the interiors
    cannot all be struck in one copy, or every copy that strikes them also strikes an end.
This enumerates both counts for every size absent at M', at the rungs 11->13 .. 23->29.

Writes results/sr_holes.txt
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PR = [5, 7, 11, 13, 17, 19, 23, 29]
JCAP = 60


def u_of(g):
    return pow(6, -1, g)


def sieve_opens(gears):
    P = 1
    for g in gears:
        P *= g
    b = np.zeros(P, dtype=bool)
    for g in gears:
        u = u_of(g)
        b[u % g:: g] = True
        b[(-u) % g:: g] = True
    return P, np.flatnonzero(~b).astype(np.int64)


def main():
    lines = []
    W = lines.append
    for i in range(2, len(PR)):
        M, qp = PR[i - 1], PR[i]
        gears = PR[:i]
        P, op = sieve_opens(gears)
        Pn, opn = sieve_opens(gears + [qp])
        gg = np.diff(np.concatenate([opn, [opn[0] + Pn]]))
        spec_new = set(int(v) for v in np.unique(gg))
        Fn = int(gg.max())
        absent = [v for v in range(1, Fn) if v not in spec_new]
        d = (2 * u_of(qp)) % qp
        N = op.size
        ext = np.concatenate([op, op[:JCAP] + P])
        res = (ext % qp).astype(np.int64)
        # census of runs by span, and total weight, for J >= 1
        span_count = {}
        weight = {}
        for J in range(1, JCAP):
            sp = ext[np.arange(N) + J] - ext[np.arange(N)]
            if sp.min() > Fn:
                break
            keep = sp <= Fn
            for v, c in zip(*np.unique(sp[keep], return_counts=True)):
                span_count[int(v)] = span_count.get(int(v), 0) + int(c)
        W(f"=== rung {M} -> {qp}   F(M')={Fn}   absent below F: {absent}")
        for v in absent:
            # legality: enumerate the runs of span v explicitly
            tot_runs, legal = 0, 0
            for J in range(1, JCAP):
                idx = np.arange(N)
                sp = ext[idx + J] - ext[idx]
                sel = np.flatnonzero(sp == v)
                if sel.size == 0:
                    if (ext[idx + J] - ext[idx]).min() > v:
                        break
                    continue
                tot_runs += sel.size
                for t in sel:
                    ok = 0
                    for r in range(qp):
                        S = {r, (r + d) % qp}
                        if all(int(res[t + k]) in S for k in range(1, J)) and \
                           int(res[t]) not in S and int(res[t + J]) not in S:
                            ok += 1
                    legal += ok
            kind = "SPAN hole (no run of that span)" if tot_runs == 0 else \
                   f"PHASE hole ({tot_runs} runs of that span, total weight {legal})"
            W(f"  v={v}: {kind}")
        W("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_holes.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()

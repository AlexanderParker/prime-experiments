"""r65 / position-length frontier, part 1b: the full period of m29, streamed.

P(m29) = 29 * P(m23) = 1,078,282,205 columns.  The m23 blocked pattern is built once and
re-used for each of the 29 copies; gear 29's two residue classes are re-phased per copy.
Only the PARETO STAIRCASE (the runs longer than every run starting earlier) is kept, which
determines R_min^>=(L) for every L.  Runs are stitched across copy boundaries by a carry.

Self-contained, numpy only.  Peak memory about 350 MB.
Run: uv run python research/anchor235/r65/pf_period29.py
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

G23 = [5, 7, 11, 13, 17, 19, 23]
P23 = 1
for g in G23:
    P23 *= g
Q = 29
P29 = P23 * Q


def main():
    base = np.zeros(P23, dtype=bool)
    for g in G23:
        u = pow(6, -1, g)
        base[u::g] = True
        base[(g - u) % g::g] = True

    u29 = pow(6, -1, Q)
    res = sorted({u29 % Q, (-u29) % Q})

    pareto = []           # (start, length) with length a strict running max
    best = 0
    carry_len = 0         # length of the blocked run ending at the end of the previous block
    carry_start = None

    scratch = np.empty(P23, dtype=bool)
    a = np.empty(P23 + 2, dtype=np.int8)
    a[0] = 0
    a[-1] = 0

    for j in range(Q):
        np.copyto(scratch, base)
        off = (j * P23) % Q
        for r in res:
            s = (r - off) % Q
            scratch[s::Q] = True
        a[1:-1] = scratch
        d = np.diff(a)
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        ln = en - st
        base_col = j * P23
        if st.size:
            # stitch the carry onto a run that starts at local column 0
            starts = st.astype(np.int64) + base_col
            lens = ln.astype(np.int64)
            if carry_len and st[0] == 0:
                starts[0] = carry_start
                lens[0] = lens[0] + carry_len
                carry_len = 0
            elif carry_len:
                # the carried run ended at the block boundary: close it
                if carry_len > best:
                    best = carry_len
                    pareto.append((carry_start, carry_len))
                carry_len = 0
            # if the last run touches the block end, carry it
            if en[-1] == P23:
                carry_start = int(starts[-1])
                carry_len = int(lens[-1])
                starts = starts[:-1]
                lens = lens[:-1]
            if lens.size:
                rm = np.maximum.accumulate(lens)
                keep = np.empty(lens.size, dtype=bool)
                keep[0] = lens[0] > best
                keep[1:] = (rm[1:] > rm[:-1]) & (rm[1:] > best)
                for x, L in zip(starts[keep].tolist(), lens[keep].tolist()):
                    if L > best:
                        best = L
                        pareto.append((x, L))
    if carry_len and carry_len > best:
        pareto.append((carry_start, carry_len))
        best = carry_len

    lines = []
    W = lines.append
    W("machine m29  gears=%s  P=%d  longest run=%d (F=%d)"
      % (G23 + [Q], P29, best, best + 1))
    W("")
    W("PARETO STAIRCASE (x, L, x/L):")
    for x, L in pareto:
        W("  x=%-12d L=%-3d  x/L=%12.3f   x/P=%.6f" % (x, L, x / L, x / P29))
    W("")
    # R_min^>=(L) from the staircase
    W("R_min^>=(L) and the ratio, every L:")
    W("   L   R_min>=(L)     ratio")
    for L in range(1, best + 1):
        x = min(xx for xx, ll in pareto if ll >= L)
        W("  %2d   %11d  %10.3f" % (L, x, x / L))
    d0 = min(x for x, L in pareto) if pareto else None
    W("")
    W("initial run length (L with R_min = 1): %d"
      % max([L for L in range(1, best + 1)
             if min(xx for xx, ll in pareto if ll >= L) == 1] or [0]))
    rr = [(min(xx for xx, ll in pareto if ll >= L) / L, L) for L in range(6, best + 1)]
    W("min ratio over L>=6: %.4f at L=%d" % min(rr))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "pf_period29.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

# Branch R3.h - ENDS OR MIDDLES, part C: the fusion depth of EVERY blocked stretch of the
# window, not only the longest, at rungs 23..997.
#
# For a stretch between consecutive openings x < y of {5..q} inside the window, the CLOSING GEAR
# g* is the largest gear that removes an interior survivor, and its FUSION COUNT is the number of
# pieces (gaps of {5..g*-}) it joins - i.e. one more than the number of survivors it removes.
# The period record at m23/m29/m31 has fusion count 4/3/3 at the top gear.  This asks whether a
# fusion of three or more ever happens inside a window, and if so on stretches of what length.
#
# Run: uv run python research/anchor235/r37/h_window_fusion.py
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "h_window_fusion.txt")
TSV = os.path.join(HERE, "results", "h_window_fusion.tsv")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_upto(1100)

say("Branch R3.h part C.  Fusion depth of every blocked stretch of the window, rungs 23..997.")
say("fusion count of a stretch = 1 + (survivors the largest participating gear removes).")
say("The period record's top gear fuses 4 (m23), 3 (m29), 3 (m31) pieces.")
say("")
say("   q   nstretch  F_W   fuse(F_W)  maxfuse  len at maxfuse  #stretches with fuse>=3   "
    "longest with fuse>=3   longest with fuse>=4")
tsv = ["q\tnstretch\tF_W\tfuse_FW\tmaxfuse\tlen_at_maxfuse\tn_fuse3\tlong3\tlong4\thist"]
rows = []

for qi, q in enumerate(PR):
    if q < 23 or q > 997:
        continue
    qp = PR[qi + 1]
    gears = [p for p in PR if 5 <= p <= q]
    lo = q // 6 + 1
    hi = (qp * qp - 1) // 6
    n = hi - lo + 1
    w = np.ones(n, bool)
    for g in gears:
        for t in teeth(g):
            w[(t - lo) % g::g] = False
    idx = np.flatnonzero(w).astype(np.int64) + lo
    if idx.size < 2:
        continue
    # kill layer of every column of the window, by sieving smallest-gear-first
    kill = np.zeros(n, np.int32)
    for g in reversed(gears):
        for t in teeth(g):
            kill[(t - lo) % g::g] = g
    FW = int(np.diff(idx).max())
    hist = {}
    best = (0, 0)          # (fusion count, length)
    long3 = long4 = 0
    n3 = 0
    fuseFW = None
    for i in range(idx.size - 1):
        x, y = int(idx[i]), int(idx[i + 1])
        G = y - x
        if G == 1:
            continue
        seg = kill[x - lo + 1: y - lo]      # kill layers of the interior columns
        gstar = int(seg.max())
        nrem = int((seg == gstar).sum())
        fuse = nrem + 1
        hist[fuse] = hist.get(fuse, 0) + 1
        if fuse > best[0] or (fuse == best[0] and G > best[1]):
            best = (fuse, G)
        if fuse >= 3:
            n3 += 1
            long3 = max(long3, G)
        if fuse >= 4:
            long4 = max(long4, G)
        if G == FW and fuseFW is None:
            fuseFW = fuse
    say(f"{q:>5} {idx.size - 1:>8} {FW:>5} {fuseFW:>9} {best[0]:>9} {best[1]:>13} {n3:>21}"
        f" {long3:>21} {long4:>21}")
    tsv.append(f"{q}\t{idx.size - 1}\t{FW}\t{fuseFW}\t{best[0]}\t{best[1]}\t{n3}\t{long3}\t"
               f"{long4}\t{sorted(hist.items())}")
    rows.append(dict(q=q, FW=FW, fuseFW=fuseFW, maxfuse=best[0], lenmax=best[1], n3=n3,
                     long3=long3, long4=long4, ns=idx.size - 1))

say("")
say("=" * 100)
tot = len(rows)
say(f"rungs {tot}")
say(f"  the window's LONGEST stretch has fusion count 2 at "
    f"{sum(1 for r in rows if r['fuseFW'] == 2)} of {tot}; "
    f">= 3 at {sum(1 for r in rows if r['fuseFW'] >= 3)} "
    f"(rungs {[r['q'] for r in rows if r['fuseFW'] >= 3]})")
say(f"  some stretch of the window has fusion count >= 3 at "
    f"{sum(1 for r in rows if r['n3'] > 0)} of {tot} rungs; >= 4 at "
    f"{sum(1 for r in rows if r['long4'] > 0)}")
say(f"  longest stretch carrying a fusion of >= 3, as a fraction of F_W: min "
    f"{min(r['long3'] / r['FW'] for r in rows):.3f}  median "
    f"{sorted(r['long3'] / r['FW'] for r in rows)[tot // 2]:.3f}  max "
    f"{max(r['long3'] / r['FW'] for r in rows):.3f}")
say(f"  longest stretch carrying a fusion of >= 4, as a fraction of F_W: median "
    f"{sorted(r['long4'] / r['FW'] for r in rows)[tot // 2]:.3f}  max "
    f"{max(r['long4'] / r['FW'] for r in rows):.3f}")
say("")
say("The joint statement: is the window's longest stretch ever ALSO its most deeply fused?")
say(f"  rungs where the maximum fusion count is attained at a stretch of length F_W: "
    f"{sum(1 for r in rows if r['lenmax'] == r['FW'])} of {tot}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(TSV, "w", encoding="utf-8") as f:
    f.write("\n".join(tsv) + "\n")
print("written", OUT, TSV)

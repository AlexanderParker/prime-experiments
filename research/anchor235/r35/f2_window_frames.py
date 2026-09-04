# Branch 5d.i, question 3: the window test.  For every column of the window
# (q/6, (q'^2-1)/6] whose residues match a record frame, walk forward through the would-be record
# stretch and record the BREAK - the first open column - its offset, and the break column's
# residues.  Part A uses the exact (5, 7, top) frames of research/anchor235/r35/results/
# f1_frames.tsv (q <= 31).  Part B drops the top gear and runs the (5, 7) frame, and the
# unrestricted longest blocked run, at every prime rung to q = 2000.  Part C is the P6 diagnosis:
# is the break offset decided by one middle gear, by the break column's residue mod 35, or by the
# distance to a gear square (g^2-1)/6?
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/f2_window_frames.py
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "f2_window_frames.txt")
TSV = os.path.join(HERE, "results", "f2_window.tsv")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def primes_to(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_to(2100)
FKNOWN = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
          43: 103, 47: 118, 53: 145}


def open_mask(gears, lo, hi):
    """Boolean array over columns lo..hi (inclusive): open under the machine `gears`."""
    n = hi - lo + 1
    w = np.ones(n, bool)
    for g in gears:
        for t in teeth(g):
            i0 = (t - lo) % g
            w[i0::g] = False
    return w


def run_lengths(w):
    """For each index i, the number of consecutive blocked entries starting at i (w False),
    i.e. the offset of the first open column at or after i.  Vectorised: the running minimum,
    from the right, of the indices of the open entries."""
    n = w.size
    nxt = np.where(w, np.arange(n), n).astype(np.int64)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    return nxt - np.arange(n)


# ---------------------------------------------------------------- Part A: the exact frames
say("Branch 5d.i / f2.  The window test.")
say("Window of {5..q} = columns (q/6, (q'^2-1)/6]; an open column there IS a twin pair.")
say("")
say("=== Part A.  Columns of the window carrying an exact record frame (5, 7, top).")
frames = {}
with open(os.path.join(HERE, "results", "f1_frames.tsv")) as f:
    next(f)
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        q = int(p[0][1:])
        frames.setdefault(q, []).append((int(p[3]), int(p[4]), int(p[5])))
say("  q    F   window cols   35q    expected frame cols   actual   break offsets (offset:count)")
tsv = ["part\tq\tF\tframe\twindow_cols\tn_frame_cols\tmax_break\tbreaks"]
for q in sorted(frames):
    gears = [p for p in PR if 5 <= p <= q]
    qp = PR[PR.index(q) + 1]
    lo = q // 6 + 1
    hi = (qp * qp - 1) // 6
    F = FKNOWN[q]
    w = open_mask(gears, lo, hi + F + 2)
    rl = run_lengths(w)
    ncols = hi - lo + 1
    k = np.arange(lo, hi + 1)
    tot = []
    for fr in sorted(set(frames[q])):
        sel = (k % 5 == fr[0]) & (k % 7 == fr[1]) & (k % q == fr[2])
        cols = k[sel]
        br = rl[cols - lo]
        tot.extend(int(b) for b in br)
        for c, b in zip(cols, br):
            tsv.append(f"A\t{q}\t{F}\t{fr}\t{ncols}\t{len(cols)}\t{int(b)}\t{int(c)}")
    say(f"  {q:<4} {F:<3} {ncols:<13} {35 * q:<6} {ncols / (35 * q):<21.3f} {len(tot):<8} "
        f"{sorted(Counter(tot).items()) if tot else '-'}")
say("  (a break offset b means: the b columns from the frame column are all blocked and column b")
say("   is open, so the window's attempt at a record on that frame reaches length b, against F-1.)")

# ---------------------------------------------------------------- Part B: (5,7) frames, all rungs
say("")
say("=== Part B.  The (5, 7) frame only, and the unrestricted longest blocked run, per rung.")
say("  For each rung: L* = the longest blocked run of {5..q} starting inside the window (the")
say("  window's best attempt at a record), its start column and that column's (5, 7) class; the")
say("  same restricted to the record's own (5, 7) frame class where it is known; and F(M).")
say("")
say("  q     window cols   F     L*     start of L*      (5,7) of start   L* on record frame   note")
rows = []
for q in PR:
    if q < 11 or q > 2000:
        continue
    gears = [p for p in PR if 5 <= p <= q]
    qp = PR[PR.index(q) + 1]
    lo = q // 6 + 1
    hi = (qp * qp - 1) // 6
    pad = 3000
    w = open_mask(gears, lo, hi + pad)
    rl = run_lengths(w)
    body = rl[:hi - lo + 1]
    i = int(np.argmax(body))
    Lstar = int(body[i])
    start = lo + i
    F = FKNOWN.get(q)
    fr57 = sorted({(a, b) for a, b, c in frames.get(q, [])})
    onframe = None
    if fr57:
        k = np.arange(lo, hi + 1)
        sel = np.zeros(k.size, bool)
        for a, b in fr57:
            sel |= (k % 5 == a) & (k % 7 == b)
        onframe = int(body[sel].max()) if sel.any() else 0
    rows.append((q, hi - lo + 1, F, Lstar, start, (start % 5, start % 7), onframe, fr57))
    if q <= 60 or q % 200 < 10 or q > 1900:
        say(f"  {q:<5} {hi - lo + 1:<13} {str(F):<5} {Lstar:<6} {start:<16} {(start % 5, start % 7)}"
            f"            {str(onframe):<20} {'F-1 = ' + str(F - 1) if F else ''}")
    tsv.append(f"B\t{q}\t{F}\t{fr57}\t{hi - lo + 1}\t-\t{Lstar}\t{start}")
say("")
mx = max(r[3] / (r[2] - 1) for r in rows if r[2])
say(f"  L*/(F-1) over the rungs where F is known: "
    + ", ".join(f"q={r[0]}: {r[3]}/{r[2] - 1} = {r[3] / (r[2] - 1):.2f}" for r in rows if r[2]))
say(f"  maximum of L*/(F-1) = {mx:.3f}")

# ---------------------------------------------------------------- Part C: what decides the break
say("")
say("=== Part C.  What decides the break (P6).")
say("  At each rung, the break column reached from every window column of the record's (5, 7)")
say("  frame class: its residue mod 35, the nearest gear square (g^2-1)/6, and the gears that")
say("  come within one column of striking it.")
for q in [23, 31, 101, 211, 503, 1009]:
    if q not in [p for p in PR]:
        continue
    gears = [p for p in PR if 5 <= p <= q]
    qp = PR[PR.index(q) + 1]
    lo = q // 6 + 1
    hi = (qp * qp - 1) // 6
    w = open_mask(gears, lo, hi + 3000)
    rl = run_lengths(w)
    k = np.arange(lo, hi + 1)
    fr57 = sorted({(a, b) for a, b, c in frames.get(q, [])}) or [(4, 4), (4, 6), (1, 6)]
    sel = np.zeros(k.size, bool)
    for a, b in fr57:
        sel |= (k % 5 == a) & (k % 7 == b)
    cols = k[sel]
    br = rl[cols - lo]
    brk = cols + br                      # the open column
    sq = np.array([(g * g - 1) // 6 for g in gears])
    d = np.abs(brk[:, None] - sq[None, :]).min(axis=1)
    say(f"  q = {q}: {len(cols)} frame columns, break offsets mean {br.mean():.2f} max {int(br.max())}"
        f" (F-1 = {FKNOWN.get(q, 0) - 1 if q in FKNOWN else '?'})")
    c35 = Counter(int(x) for x in brk % 35)
    top35 = c35.most_common(3)
    say(f"     break column mod 35: {len(c35)} classes, top {top35}, "
        f"fair share {len(cols) / max(len(c35), 1):.1f}")
    say(f"     distance from the break column to the nearest gear square: min {int(d.min())}, "
        f"median {int(np.median(d))}, share within 2: {(d <= 2).mean():.4f}")
    # which gear "nearly" struck the break column
    near = Counter()
    for g in gears:
        a, b = teeth(g)
        r = brk % g
        near[g] = int((((r - a) % g <= 1) | ((a - r) % g <= 1) | ((r - b) % g <= 1)
                       | ((b - r) % g <= 1)).sum())
    say(f"     gears within one column of striking the break, top 5: {near.most_common(5)}"
        f"  (out of {len(cols)})")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(TSV, "w", encoding="utf-8") as f:
    f.write("\n".join(tsv) + "\n")
print("written", OUT, TSV)

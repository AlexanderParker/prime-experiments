# Branch 5d.i, questions 1 and 2: the record stretch as a frame (gears 5, 7, top) plus a filling
# (the middle gears).  For every record stretch of {5..q}, q = 7..31, over the FULL period:
#   - the start column's residue under every gear, the full strike table, the top gear's strike
#     offsets and their corridor residues mod 35, gear 7's and gear 5's likewise;
#   - the set of distinct (5, 7, top) frames, and for each frame the exact number of COMPLETIONS
#     (choices of middle-gear residues that block every offset the frame leaves open) against the
#     product over middle gears of the number of residues that strike at least one such offset.
# Full periods: m7..m23 in one array, m29 (P = 1.08e9) and m31 (P = 3.34e10) by chunked scan.
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/f1_record_frames.py
import os
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "f1_record_frames.txt")
TSV = os.path.join(HERE, "results", "f1_frames.tsv")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def record_starts(gears, P, thresh, chunk=1 << 24):
    """All openings x with gap(x) >= thresh, over the full period, by chunked residue sieve.
    Column 0 is always open (0 is never +-u_g), so the cycle closes at 0."""
    ts = [teeth(g) for g in gears]
    last = None
    first = None
    big = []          # (x, gap)
    fmax = 0
    base = 0
    w = np.empty(chunk, bool)
    while base < P:
        n = min(chunk, P - base)
        w[:n] = True
        for g, (a, b) in zip(gears, ts):
            for t in (a, b):
                i0 = (t - base) % g
                w[i0:n:g] = False
        idx = np.flatnonzero(w[:n]).astype(np.int64) + base
        if idx.size:
            if first is None:
                first = int(idx[0])
            if last is not None:
                d = int(idx[0]) - last
                fmax = max(fmax, d)
                if d >= thresh:
                    big.append((last, d))
            df = np.diff(idx)
            if df.size:
                fmax = max(fmax, int(df.max()))
                for j in np.flatnonzero(df >= thresh):
                    big.append((int(idx[j]), int(df[j])))
            last = int(idx[-1])
        base += n
    d = first + P - last          # the wrap gap
    fmax = max(fmax, d)
    if d >= thresh:
        big.append((last, d))
    return fmax, sorted(big)


def cover_dist(R, gears):
    """Over the prod(gears) choices of middle-gear start residues (= the columns of the period
    carrying the frame), the exact distribution of the COVERED subset of R.  Returns
    (masks, counts, states) with masks a bitmask over R's positions.  A tuple whose mask is full
    is a COMPLETION: every offset of the record length is blocked, so the column starts a
    blocked run of length F-1, i.e. a record stretch."""
    pos = {o: i for i, o in enumerate(R)}
    masks_all = []
    for g in gears:
        a, b = teeth(g)
        ms = []
        for r in range(g):                     # r = start residue mod g
            m = 0
            for o in R:
                if (r + o) % g in (a, b):
                    m |= 1 << pos[o]
            ms.append(m)
        masks_all.append(np.array(ms, dtype=np.int64))
    cur_m = np.array([0], dtype=np.int64)
    cur_c = np.array([1], dtype=np.int64)
    peak = 1
    for ms in masks_all:
        m = (cur_m[:, None] | ms[None, :]).ravel()
        c = np.repeat(cur_c, ms.size)
        um, inv = np.unique(m, return_inverse=True)
        uc = np.zeros(um.size, dtype=np.int64)
        np.add.at(uc, inv, c)
        cur_m, cur_c = um, uc
        peak = max(peak, um.size)
    return cur_m, cur_c, peak


def break_hist(R, masks, counts):
    """From the covered-subset distribution, the histogram of the BREAK OFFSET: the first offset
    of the would-be record stretch left open by every gear.  Key -1 = no break (a completion)."""
    full = (1 << len(R)) - 1
    notm = (~masks) & full
    h = {}
    low = notm & (-notm)
    idx = np.where(notm == 0, -1, np.rint(np.log2(np.maximum(low, 1))).astype(np.int64))
    for i in np.unique(idx):
        tot = int(counts[idx == i].sum())
        h[-1 if i < 0 else R[int(i)]] = tot
    return h


LADDER = [7, 11, 13, 17, 19, 23, 29, 31]
FKNOWN = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}

say("Branch 5d.i / f1.  The record stretch as frame (5, 7, top) + filling (middle gears).")
say("Offsets j are relative to the start s = x+1 of the blocked run; the run is j = 0 .. F-2.")
tsv = ["machine\tF\tn_records\tframe_5\tframe_7\tframe_top\tn_with_frame\ttop_offsets\ttop_mod35\tword"]

for i, q in enumerate([5] + LADDER):
    if q == 5:
        continue
    gears = [5] + LADDER[:LADDER.index(q) + 1]
    P = prod(gears)
    t0 = time.time()
    fmax, big = record_starts(gears, P, FKNOWN[q])
    F = fmax
    starts = [x for x, d in big if d == F]
    assert F == FKNOWN[q], (q, F, FKNOWN[q])
    say("")
    say(f"=== M = {{5..{q}}}   P = {P}   F = {F}   {len(starts)} record stretches   "
        f"(scan {time.time() - t0:.1f} s)")
    top = q
    a_top, b_top = teeth(top)
    frames = {}
    for x in starts:
        s = x + 1
        km = {}
        for g in gears:
            aa, bb = teeth(g)
            km[g] = [j for j in range(F - 1) if (s + j) % g in (aa, bb)]
        fr = (s % 5, s % 7, s % top)
        frames.setdefault(fr, []).append(s)
        res = ",".join(f"{g}:{s % g}" for g in gears)
        strikes = " ".join(f"{g}:{km[g]}" for g in gears)
        c35 = [(s + j) % 35 for j in km[top]]
        word = [km[top][k + 1] - km[top][k] for k in range(len(km[top]) - 1)]
        say(f"  x={x:<12} s={s:<12} frac={s / P:.4f}  residues {res}")
        say(f"      strikes {strikes}")
        say(f"      top gear {top}: offsets {km[top]}  cols mod 35 {c35}  word {word}"
            f"   (letters a={2 * min(a_top, b_top)} b={top - 2 * min(a_top, b_top)})")
        say(f"      gear 7: offsets {km[7]} cols mod 35 {[(s + j) % 35 for j in km[7]]}"
            f" | gear 5: offsets {km[5]} cols mod 35 {[(s + j) % 35 for j in km[5]]}")
    say(f"  distinct (5, 7, top) frames: {len(frames)}   sizes {sorted(len(v) for v in frames.values())}")
    for fr, ss in sorted(frames.items()):
        say(f"    frame {fr}: {len(ss)} record(s), starts {ss[:6]}{' ...' if len(ss) > 6 else ''}")

    # ---- question 2: the filling count, per distinct frame
    mid = [g for g in gears if g not in (5, 7, top)]
    say(f"  middle gears {mid}, candidate columns per frame = prod(mid) = {prod(mid)}"
        f"  (= P / (35*{top}) = {P // (35 * top)})")
    say("    frame                 |R| (offsets the frame leaves open)   completions   prod n_g   ratio")
    for fr, ss in sorted(frames.items()):
        s0 = ss[0]
        blocked_by_frame = set()
        for g in (5, 7, top):
            aa, bb = teeth(g)
            blocked_by_frame |= {j for j in range(F - 1) if (s0 + j) % g in (aa, bb)}
        R = [j for j in range(F - 1) if j not in blocked_by_frame]
        masks, counts, peak = cover_dist(R, mid)
        C = int(counts[masks == (1 << len(R)) - 1].sum()) if len(R) else int(counts.sum())
        h = break_hist(R, masks, counts)
        ng = []
        for g in mid:
            aa, bb = teeth(g)
            ng.append(sum(1 for r in range(g) if any((r + o) % g in (aa, bb) for o in R)))
        pn = prod(ng) if ng else 1
        say(f"    {str(fr):<20}  |R| = {len(R):<3} R = {R}")
        say(f"        completions = {C}   n_g = {dict(zip(mid, ng))}   prod n_g = {pn}"
            f"   ratio = {C / pn:.3e}   completions/candidates = {C / prod(mid):.3e}"
            f"   (DP peak states {peak})")
        cand = prod(mid)
        tail = 0
        parts = []
        for b in sorted(k for k in h if k >= 0):
            parts.append(f"{b}:{h[b]}")
        say(f"        break offset histogram over the {cand} frame columns of the period: "
            + " ".join(parts))
        surv = cand
        surv_line = []
        for b in sorted(k for k in h if k >= 0):
            surv -= h[b]
            surv_line.append(f">{b}:{surv}")
        say(f"        survivors past each break: " + " ".join(surv_line[-8:]))
        tsv.append(f"m{q}\t{F}\t{len(starts)}\t{fr[0]}\t{fr[1]}\t{fr[2]}\t{len(ss)}\t"
                   f"{[j for j in range(F - 1) if (s0 + j) % top in teeth(top)]}\t"
                   f"{[(s0 + j) % 35 for j in range(F - 1) if (s0 + j) % top in teeth(top)]}\t"
                   f"{C}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(TSV, "w", encoding="utf-8") as f:
    f.write("\n".join(tsv) + "\n")
print("written", OUT, TSV)

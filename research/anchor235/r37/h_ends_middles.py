# Branch R3.h - ENDS OR MIDDLES.  The layer decomposition of the exact record stretches of
# {5..q}, q = 7..31, over the FULL period, plus the runner-up stretches at m23..m31.
#
# For a stretch between consecutive openings x < y = x + G:
#   - the KILL LAYER of every interior column (the smallest gear striking it) and its full
#     striker set;
#   - per layer g, the survivors S_g (openings of {5..g} in [x, y], ends included), the gap word
#     of S_g, its largest letter, the number of lower gaps k_g fused above g;
#   - which survivors gear g removes, their residues mod g and which tooth, and whether removed
#     survivors that are adjacent in S_{g-} form a chain (the chain law);
#   - the ends: the kill layer of the columns just outside (x-1, y+1) and the neighbouring gaps;
#   - the layer-7 corridor: |E_35 cap [x, x+G]| and its rank over the 35 rotations.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r37/h_ends_middles.py
import os
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
OUT = os.path.join(HERE, "results", "h_ends_middles.txt")
TSV = os.path.join(HERE, "results", "h_layers.tsv")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


LADDER = [7, 11, 13, 17, 19, 23, 29, 31]
FKNOWN = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
E35 = sorted({r for r in range(35) if r % 5 in (0, 2, 3) and r % 7 in (0, 2, 3, 4, 5)})


def big_gaps(gears, P, thresh, chunk=1 << 24):
    """All (x, gap) with gap >= thresh over the full period, by chunked residue sieve."""
    ts = [teeth(g) for g in gears]
    last = None
    first = None
    big = []
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
    d = first + P - last
    fmax = max(fmax, d)
    if d >= thresh:
        big.append((last, d))
    return fmax, big


def strikers(k, gears):
    return [g for g in gears if k % g in teeth(g)]


def decompose(x, G, gears):
    """Kill-layer word and per-layer survivor structure of the interval [x, x+G]."""
    kl = {}          # offset -> kill layer (None for the two ends)
    st = {}
    for j in range(0, G + 1):
        s = strikers(x + j, gears)
        st[j] = s
        kl[j] = min(s) if s else None
    assert kl[0] is None and kl[G] is None, (x, G)
    layers = []
    for g in gears:
        S = [j for j in range(G + 1) if kl[j] is None or kl[j] > g]
        gaps = [S[i + 1] - S[i] for i in range(len(S) - 1)]
        rem = [j for j in range(G + 1) if kl[j] == g]
        layers.append(dict(g=g, S=S, gaps=gaps, rem=rem))
    return kl, st, layers


def corridor_rank(x, G):
    """|E_35 cap [r, r+G]| for r = x mod 35, and the rank of that count among the 35 rotations."""
    cnt = [sum(1 for j in range(G + 1) if (r + j) % 35 in E35) for r in range(35)]
    r = x % 35
    c = cnt[r]
    srt = sorted(cnt)
    return c, srt[0], srt[-1], sum(1 for v in cnt if v < c), cnt


def chains(prevS, rem):
    """Runs of consecutive elements of the previous layer's survivor list that this gear removes."""
    remset = set(rem)
    out = []
    run = []
    for j in prevS:
        if j in remset:
            run.append(j)
        else:
            if run:
                out.append(run)
            run = []
    if run:
        out.append(run)
    return out


tsv = ["machine\tkind\tx\tG\tlayer\tnS\tk_g\tmaxgap\tF_g\tfrac\tfirstgap\tlastgap\tnrem\trem_offsets"]
summary = []

for q in LADDER:
    gears = [5] + LADDER[:LADDER.index(q) + 1]
    P = prod(gears)
    F = FKNOWN[q]
    thresh = max(3, F - 4)
    t0 = time.time()
    fmax, big = big_gaps(gears, P, thresh)
    assert fmax == F, (q, fmax, F)
    vals = sorted({d for _, d in big}, reverse=True)
    runner = vals[1] if len(vals) > 1 else None
    recs = [(x, d) for x, d in big if d == F]
    runs = [(x, d) for x, d in big if runner is not None and d == runner]
    say("")
    say("=" * 100)
    say(f"=== M = {{5..{q}}}   P = {P}   F = {F}   gap spectrum near the top: "
        f"{ {v: sum(1 for _, d in big if d == v) for v in vals} }   (scan {time.time() - t0:.1f} s)")
    say(f"    {len(recs)} record stretches; runner-up value {runner} with {len(runs)} stretches")

    for kind, group in (("record", recs), ("runner", runs)):
        if kind == "runner" and q < 23:
            continue
        show = group[:6]
        for (x, G) in show:
            kl, st, layers = decompose(x, G, gears)
            say("")
            say(f"  --- {kind}  x = {x}  G = {G}  y = {x + G}   residues "
                + ",".join(f"{g}:{x % g}" for g in gears))
            say(f"      kill-layer word (offset:layer, ends '.'): "
                + " ".join(f"{j}:{'.' if kl[j] is None else kl[j]}" for j in range(G + 1)))
            sole = {j: st[j][0] for j in range(1, G) if len(st[j]) == 1}
            say(f"      sole-struck columns (offset:gear): {sole}")
            # ends
            klm = strikers(x - 1, gears)
            klp = strikers(x + G + 1, gears)
            say(f"      OUTSIDE ends: column x-1 struck by {klm} (kill layer "
                f"{min(klm) if klm else 'OPEN'}); column y+1 struck by {klp} (kill layer "
                f"{min(klp) if klp else 'OPEN'})")
            c, cmin, cmax, rank, _ = corridor_rank(x, G)
            say(f"      layer-7 corridor: x mod 35 = {x % 35}; |E_35 cap [x, x+G]| = {c} "
                f"(min over 35 rotations {cmin}, max {cmax}, rotations strictly below {rank})")
            say("      layer  |S_g|  k_g  gap word (sums to G)                    maxgap  F_g   frac   "
                "removed at this layer (offset res_g tooth)")
            prevS = None
            for L in layers:
                g = L["g"]
                S, gaps, rem = L["S"], L["gaps"], L["rem"]
                mg = max(gaps) if gaps else 0
                Fg = FKNOWN[g]
                u = teeth(g)[0]
                remtxt = ", ".join(f"{j}:{(x + j) % g}{'+' if (x + j) % g == u else '-'}" for j in rem)
                ch = chains(prevS, rem) if prevS is not None else []
                chtxt = f"  chains {[len(c2) for c2 in ch if len(c2) > 1]}" if any(len(c2) > 1 for c2 in ch) else ""
                say(f"       {g:>4}   {len(S):>3}  {len(gaps):>3}  {str(gaps):<40} {mg:>5}  {Fg:>4}  "
                    f"{mg / G:.3f}   {remtxt}{chtxt}")
                tsv.append(f"m{q}\t{kind}\t{x}\t{G}\t{g}\t{len(S)}\t{len(gaps)}\t{mg}\t{Fg}\t"
                           f"{mg / G:.4f}\t{gaps[0] if gaps else 0}\t{gaps[-1] if gaps else 0}\t"
                           f"{len(rem)}\t{rem}")
                prevS = S
            # ends vs middles summary for this stretch
            say("      ends/middles: layer  first+last / G   maxgap position (index of the "
                "largest letter, out of k_g)   removals in middle 3/5")
            prevS = None
            for L in layers:
                g, S, gaps, rem = L["g"], L["S"], L["gaps"], L["rem"]
                if not gaps:
                    continue
                mg = max(gaps)
                ai = gaps.index(mg)
                ef = (gaps[0] + gaps[-1]) / G if len(gaps) > 1 else 1.0
                mid = sum(1 for j in rem if 0.2 * G <= j <= 0.8 * G)
                say(f"                     {g:>4}   {ef:.3f}            {ai + 1}/{len(gaps)}"
                    f"{'  (an END letter)' if ai in (0, len(gaps) - 1) else '  (an INTERIOR letter)'}"
                    f"          {mid}/{len(rem)}")
                prevS = S
            if kind == "record":
                summary.append((q, F, x, [(L["g"], len(L["S"]), max(L["gaps"]) if L["gaps"] else 0,
                                           len(L["rem"])) for L in layers]))

say("")
say("=" * 100)
say("SUMMARY: frac_g = maxgap_g / F for the first record stretch of each machine")
say("machine  F   " + "".join(f"{g:>8}" for g in [5] + LADDER))
for q, F, x, rows in summary:
    d = {g: mg for g, ns, mg, nr in rows}
    say(f"m{q:<6} {F:<3} " + "".join(f"{(d[g] / F):>8.3f}" if g in d else f"{'':>8}"
                                     for g in [5] + LADDER))
say("")
say("SUMMARY: k_g (number of lower gaps above layer g that the gears above g fuse)")
say("machine  F   " + "".join(f"{g:>8}" for g in [5] + LADDER))
for q, F, x, rows in summary:
    d = {g: ns - 1 for g, ns, mg, nr in rows}
    say(f"m{q:<6} {F:<3} " + "".join(f"{d[g]:>8}" if g in d else f"{'':>8}" for g in [5] + LADDER))
say("")
say("SUMMARY: survivors removed at each layer")
say("machine  F   " + "".join(f"{g:>8}" for g in [5] + LADDER))
for q, F, x, rows in summary:
    d = {g: nr for g, ns, mg, nr in rows}
    say(f"m{q:<6} {F:<3} " + "".join(f"{d[g]:>8}" if g in d else f"{'':>8}" for g in [5] + LADDER))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(TSV, "w", encoding="utf-8") as f:
    f.write("\n".join(tsv) + "\n")
print("written", OUT, TSV)

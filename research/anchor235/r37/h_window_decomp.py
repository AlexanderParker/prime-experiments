# Branch R3.h - ENDS OR MIDDLES, part B: the same layer decomposition applied to the WINDOW's
# longest stretch at every prime rung q = 23..997, against the period record's.
#
# Window at rung q (project vocabulary): columns lo = q//6 + 1 .. hi = (q'^2 - 1)//6, q' the next
# prime.  An opening of {5..q} there is a twin pair.  F_W = the longest gap between consecutive
# openings inside the window.  For that stretch we compute the kill-layer word, the survivors
# S_g at every layer, the number k_g of lower gaps fused above g, the largest sub-gap maxgap_g,
# the CLOSING GEAR g* (the largest gear that removes an interior survivor - the gear that makes
# the stretch), and the number of columns struck by the top gear q.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r37/h_window_decomp.py
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
OUT = os.path.join(HERE, "results", "h_window_decomp.txt")
TSV = os.path.join(HERE, "results", "h_window.tsv")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


FKNOWN = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
          41: 91, 43: 103, 47: 118, 53: 145}


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_upto(1100)

say("Branch R3.h part B.  The window's longest stretch, decomposed by layer, at rungs 23..997.")
say("Window at rung q: columns lo = q//6+1 .. hi = (q'^2-1)//6; openings there are twin pairs.")
say("g* = the largest gear that removes an interior survivor of the stretch (the gear that makes it).")
say("nq = columns of the stretch struck by the TOP gear q.  npart = gears that remove at least one.")
say("")
say("   q    q'   F_W   F_W/q     x       g*    g*/q   nq  npart  ngears   k_g at g = 5,7,11,13,...(first 8)"
    "   maxgap/F_W at those layers")

tsv = ["q\tqp\tlo\thi\tF_W\tx\tnstretch\tgstar\tnq\tnpart\tngears\tk_word\tfrac_word\tkillword"]
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
            i0 = (t - lo) % g
            w[i0::g] = False
    idx = np.flatnonzero(w).astype(np.int64) + lo
    if idx.size < 2:
        continue
    df = np.diff(idx)
    FW = int(df.max())
    where = [int(idx[j]) for j in np.flatnonzero(df == FW)]
    x = where[0]
    G = FW
    # kill layers of the interval [x, x+G]
    kl = []
    for j in range(G + 1):
        k = x + j
        s = [g for g in gears if k % g in teeth(g)]
        kl.append(min(s) if s else None)
    assert kl[0] is None and kl[-1] is None
    layers = []
    for g in gears:
        S = [j for j in range(G + 1) if kl[j] is None or kl[j] > g]
        gaps = [S[i + 1] - S[i] for i in range(len(S) - 1)]
        rem = [j for j in range(G + 1) if kl[j] == g]
        layers.append((g, S, gaps, rem))
    part = [g for g, S, gaps, rem in layers if rem]
    gstar = max(part)
    nq = sum(1 for j in range(G + 1) if (x + j) % q in teeth(q))
    kword = [len(gaps) for g, S, gaps, rem in layers][:8]
    fword = [round(max(gaps) / G, 3) if gaps else 0 for g, S, gaps, rem in layers][:8]
    say(f"{q:>5} {qp:>5} {FW:>5} {FW / q:>7.3f} {x:>8} {gstar:>7} {gstar / q:>6.3f} {nq:>4} "
        f"{len(part):>5} {len(gears):>6}   {kword}   {fword}")
    tsv.append(f"{q}\t{qp}\t{lo}\t{hi}\t{FW}\t{x}\t{len(where)}\t{gstar}\t{nq}\t{len(part)}\t"
               f"{len(gears)}\t{[len(g2) for _, _, g2, _ in layers]}\t"
               f"{[round(max(g2) / G, 4) if g2 else 0 for _, _, g2, _ in layers]}\t"
               f"{[('.' if v is None else v) for v in kl]}")
    rows.append(dict(q=q, FW=FW, x=x, gstar=gstar, nq=nq, npart=len(part), ngears=len(gears),
                     layers=layers, G=G, kl=kl, nstretch=len(where)))

say("")
say("=" * 100)
say("E6/E7 tally over the rungs 23..997")
tot = len(rows)
say(f"rungs: {tot}")
say(f"  top gear q strikes NO column of the window's longest stretch: "
    f"{sum(1 for r in rows if r['nq'] == 0)} of {tot}")
say(f"  top gear strikes exactly one: {sum(1 for r in rows if r['nq'] == 1)}")
say(f"  top gear strikes two or more: {sum(1 for r in rows if r['nq'] >= 2)}"
    f"  (rungs {[r['q'] for r in rows if r['nq'] >= 2]})")
say(f"  the top gear REMOVES an interior survivor (g* = q): "
    f"{sum(1 for r in rows if r['gstar'] == r['q'])} of {tot}"
    f"  (rungs {[r['q'] for r in rows if r['gstar'] == r['q']][:20]})")
say(f"  g*/q: min {min(r['gstar'] / r['q'] for r in rows):.3f}  median "
    f"{sorted(r['gstar'] / r['q'] for r in rows)[tot // 2]:.3f}  max "
    f"{max(r['gstar'] / r['q'] for r in rows):.3f}")
say(f"  F_W/q: min {min(r['FW'] / r['q'] for r in rows):.3f}  median "
    f"{sorted(r['FW'] / r['q'] for r in rows)[tot // 2]:.3f}  max "
    f"{max(r['FW'] / r['q'] for r in rows):.3f}")
say(f"  participating gears npart: min {min(r['npart'] for r in rows)} median "
    f"{sorted(r['npart'] for r in rows)[tot // 2]} max {max(r['npart'] for r in rows)}; "
    f"npart/ngears median {sorted(r['npart'] / r['ngears'] for r in rows)[tot // 2]:.3f}")

# how many gears are needed to reach k_g = 1 (the stretch closed), and the frac curve there
say("")
say("The closing profile: at which layer does k_g fall to 1, and what is maxgap/F_W just below it")
say("   q   F_W   g*   k at g*-   frac at g*-   the last three fusions (gap word at the layer "
    "below g*)")
for r in rows:
    layers = r["layers"]
    below = [L for L in layers if L[0] < r["gstar"]]
    if not below:
        continue
    g_, S_, gaps_, rem_ = below[-1]
    say(f"{r['q']:>5} {r['FW']:>5} {r['gstar']:>4}   {len(gaps_):>4}      "
        f"{max(gaps_) / r['G']:.3f}       {gaps_}")

# E1 analogue in the window: is any sub-gap a lower machine's record?
say("")
say("E1 in the window: layers g <= 53 where maxgap_g >= F({5..g}) (a lower record inside)")
bad = 0
for r in rows:
    for g, S, gaps, rem in r["layers"]:
        if g in FKNOWN and gaps and max(gaps) >= FKNOWN[g]:
            say(f"   q = {r['q']}  layer {g}: maxgap {max(gaps)} >= F({g}) = {FKNOWN[g]}")
            bad += 1
            break
say(f"   rungs with a lower record inside the window stretch: {bad} of {tot}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(TSV, "w", encoding="utf-8") as f:
    f.write("\n".join(tsv) + "\n")
print("written", OUT, TSV)

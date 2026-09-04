# Branch R3.h - ENDS OR MIDDLES: the summary tables.  Reads the (machine, kind, x, G) list from
# h_layers.tsv (written by h_ends_middles.py, full-period scans) and recomputes, for every record
# and runner-up stretch:
#   T1  the layer profile: k_g (pieces), maxgap_g, F_g, maxgap_g/F_g, maxgap_g/G;
#   T2  the top-gear decomposition F = leftflank + (letters of the top gear) + rightflank;
#   T3  the ends: kill layer of the columns just outside, and the flank gaps of the lower machine;
#   T4  the layer-7 corridor rank of the start residue mod 35;
#   T5  the E4 tally (removals in the middle three-fifths at the top three layers).
# Run: uv run python research/anchor235/r37/h_summary.py
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "results", "h_layers.tsv")
OUT = os.path.join(HERE, "results", "h_summary.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


LADDER = [5, 7, 11, 13, 17, 19, 23, 29, 31]
FK = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
E35 = sorted({r for r in range(35) if r % 5 in (0, 2, 3) and r % 7 in (0, 2, 3, 4, 5)})

seen = []
with open(TSV, encoding="utf-8") as f:
    next(f)
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        key = (p[0], p[1], int(p[2]), int(p[3]))
        if not seen or seen[-1] != key:
            if key not in seen:
                seen.append(key)


def analyse(mach, x, G):
    q = int(mach[1:])
    gears = [g for g in LADDER if g <= q]
    kl = []
    for j in range(G + 1):
        s = [g for g in gears if (x + j) % g in teeth(g)]
        kl.append(min(s) if s else None)
    layers = {}
    for g in gears:
        S = [j for j in range(G + 1) if kl[j] is None or kl[j] > g]
        gaps = [S[i + 1] - S[i] for i in range(len(S) - 1)]
        rem = [j for j in range(G + 1) if kl[j] == g]
        layers[g] = (S, gaps, rem)
    return q, gears, kl, layers


say("Branch R3.h - ENDS OR MIDDLES.  Summary over the exact record and runner-up stretches,")
say("full periods, machines {5..7}..{5..31}.  G = the stretch's length (gap between the two")
say("flanking openings); a 'piece' at layer g is a gap of the lower machine {5..g} inside it.")

# ---------- T1 ----------
say("")
say("T1.  Layer profile of the FIRST record stretch of each machine.")
say("     k_g = pieces at layer g;  mx = largest piece;  F_g = that machine's own record;")
say("     mx/F_g tells whether the piece is an ordinary gap or a lower record;  mx/G its share.")
firsts = {}
for mach, kind, x, G in seen:
    if kind == "record" and mach not in firsts:
        firsts[mach] = (x, G)
for mach in [f"m{q}" for q in LADDER[1:]]:
    if mach not in firsts:
        continue
    x, G = firsts[mach]
    q, gears, kl, layers = analyse(mach, x, G)
    say("")
    say(f"  {mach}  F = {G}  x = {x}")
    say("     g   k_g   largest piece   F_g    mx/F_g   mx/G    gap word")
    for g in gears:
        S, gaps, rem = layers[g]
        mx = max(gaps) if gaps else 0
        say(f"   {g:>3}   {len(gaps):>3}       {mx:>3}        {FK[g]:>3}    {mx / FK[g]:.3f}    "
            f"{mx / G:.3f}   {gaps if len(gaps) <= 12 else str(gaps[:12]) + '...'}")

# ---------- T2 ----------
say("")
say("=" * 100)
say("T2.  The top gear's fusion: F = left flank + interior pieces + right flank, where the")
say("     interior pieces are the distances between the top gear's strikes on the survivors.")
say("     Letters of gear q: a = 2u_q, b = q - 2u_q (u_q = 6^-1 mod q).")
say("")
say(" machine   F   pieces fused   interior pieces   letters {a,b}   left flank  right flank  "
    "flank sum  interior sum")
for mach, kind, x, G in seen:
    q, gears, kl, layers = analyse(mach, x, G)
    prev = gears[-2] if len(gears) > 1 else None
    if prev is None:
        continue
    S, gaps, rem = layers[prev]
    _, _, remtop = layers[q]
    u = teeth(q)[0]
    dq = (2 * u) % q
    a, b = min(dq, q - dq), max(dq, q - dq)
    inner = gaps[1:-1] if len(gaps) >= 2 else []
    lets = f"{{{a}, {b}}}"
    say(f" {mach:<7} {kind[:3]} {G:>4}   {len(gaps):>6}        {str(inner):<18} {lets:<12} "
        f"  {gaps[0]:>5}      {gaps[-1]:>5}      {gaps[0] + gaps[-1]:>5}      {sum(inner):>5}")

# ---------- T3 / T4 / T5 ----------
say("")
say("=" * 100)
say("T3-T5.  The ends, the corridor, and where the removals sit.")
say("  outL/outR = the smallest gear striking the column immediately outside the left/right end")
say("              ('OPEN' = that column is an opening too, so the flanking gap is 1)")
say("  corridor  = |E_35 cap [x, x+G]| / min over the 35 rotations / max / how many rotations")
say("              are strictly below it   (degenerate when G+1 is a multiple of 35)")
say("  mid3/5    = removals in the middle three-fifths of the interval, over the TOP THREE layers")
say("")
say(" machine kind    G     outL  outR   corridor (count/min/max/rank)   mid3/5 at top 3 layers")
e5ok = e5tot = 0
e4ok = e4tot = 0
for mach, kind, x, G in seen:
    q, gears, kl, layers = analyse(mach, x, G)
    lo = [g for g in gears if (x - 1) % g in teeth(g)]
    hi = [g for g in gears if (x + G + 1) % g in teeth(g)]
    cnt = [sum(1 for j in range(G + 1) if (r + j) % 35 in E35) for r in range(35)]
    c = cnt[x % 35]
    rank = sum(1 for v in cnt if v < c)
    top3 = gears[-3:]
    nm = nt = 0
    for g in top3:
        for j in layers[g][2]:
            nt += 1
            if 0.2 * G <= j <= 0.8 * G:
                nm += 1
    say(f" {mach:<7} {kind[:3]:<5} {G:>4}   {(min(lo) if lo else 'OPEN'):>5} "
        f"{(min(hi) if hi else 'OPEN'):>5}      {c:>3} / {min(cnt):>3} / {max(cnt):>3} / {rank:>3}"
        f"{'  (degenerate)' if min(cnt) == max(cnt) else '':<14}      {nm}/{nt}")
    if min(cnt) != max(cnt):
        e5tot += 1
        if c == min(cnt):
            e5ok += 1
    e4tot += 1
    if nt and nm > nt / 2:
        e4ok += 1
say("")
say(f"E5: corridor-extremal (count = minimum over the 35 rotations) at {e5ok} of {e5tot} "
    f"non-degenerate stretches")
say(f"E4: more than half of the top-three-layer removals in the middle three-fifths at "
    f"{e4ok} of {e4tot} stretches")

# ---------- T6: the ratio mx/F_g at the top three layers, all stretches ----------
say("")
say("=" * 100)
say("T6.  Does a record contain a LOWER RECORD?  mx/F_g by layer, over every record and")
say("     runner-up stretch of m13..m31 (min .. max over the stretches).")
byl = defaultdict(list)
for mach, kind, x, G in seen:
    q, gears, kl, layers = analyse(mach, x, G)
    if q < 13:
        continue
    for i, g in enumerate(gears):
        S, gaps, rem = layers[g]
        if not gaps:
            continue
        depth = len(gears) - 1 - i          # 0 = the top gear, 1 = one below, ...
        byl[(mach, kind, depth)].append(max(gaps) / FK[g])
say("  d1 = one gear below the top, d2 = two below, ...;  the entry is mx/F_g over the")
say("  stretches, min-max.  1.000 means that lower machine's own record sits inside.")
for mach in [f"m{q2}" for q2 in LADDER[1:]]:
    for kind in ("record", "runner"):
        ds = sorted(d for (m2, k2, d) in byl if m2 == mach and k2 == kind and d >= 1)
        if not ds:
            continue
        parts = []
        for d in ds:
            v = byl[(mach, kind, d)]
            parts.append(f"d{d} ({[g2 for g2 in LADDER if g2 <= int(mach[1:])][-1 - d]}): "
                         f"{min(v):.3f}-{max(v):.3f}")
        say(f"  {mach:<7} {kind[:3]}   " + "  ".join(parts))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT)

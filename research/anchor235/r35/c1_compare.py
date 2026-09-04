# Branch 5d.ii, question 3/4: the two profiles compared.
# (a) The DISTINCT window record stretches over the rungs 7..997 (F_W is inherited: the same
#     stretch is the window record at many consecutive rungs), each with its holder set, its
#     sole-column density, its (5,7) phase and its MINIMUM BLOCKING SET (exact set cover).
# (b) The same quantities for the PERIOD record at m7..m23 (where L4 forces every gear essential).
# (c) The square-gate joint deletion: every gear with g^2 > 6*top+1 removed AT ONCE -- provably
#     harmless, because a composite below q'^2 has a prime factor below its own square root.
# (d) Gear-by-gear drop(g) versus drop_W(g) at the rungs where both exist (q = 11..23).
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/c1_compare.py
import numpy as np, os, time
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "c1_compare.txt")
lines = []
def say(s=""):
    print(s); lines.append(s)

def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]
PR = primes_upto(1100)
def nextp(q): return PR[PR.index(q) + 1]

def strikes(k, g):
    u = pow(6, -1, g); return k % g in (u, g - u)

def mincover(cols, gears):
    """exact minimum number of gears covering every column of cols (all must be coverable)."""
    idx = {c: i for i, c in enumerate(cols)}
    full = (1 << len(cols)) - 1
    masks = {}
    for g in gears:
        m = 0
        for c in cols:
            if strikes(c, g): m |= 1 << idx[c]
        if m: masks[g] = m
    order = sorted(masks, key=lambda g: -bin(masks[g]).count("1"))
    best = [len(order) + 1, None]
    def rec(cov, chosen):
        if cov == full:
            if len(chosen) < best[0]: best[0] = len(chosen); best[1] = list(chosen)
            return
        if len(chosen) + 1 >= best[0]: return
        rem = full & ~cov
        # branch on the uncovered column with the fewest covering gears
        bestg = None
        cnt = 10 ** 9
        for j in range(len(cols)):
            if rem >> j & 1:
                gs = [g for g in order if masks[g] >> j & 1]
                if len(gs) < cnt:
                    cnt, bestg = len(gs), gs
                    if cnt == 1: break
        for g in bestg:
            chosen.append(g); rec(cov | masks[g], chosen); chosen.pop()
    rec(0, [])
    return best[0], sorted(best[1] or [])

def window_data(q):
    qq = nextp(q); lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    gears = [g for g in PR if 5 <= g <= q]
    cnt = np.zeros(n, np.int16); gsum = np.zeros(n, np.int32)
    for g in gears:
        u = pow(6, -1, g)
        for t in (u, g - u):
            s = (t - lo) % g; cnt[s::g] += 1; gsum[s::g] += g
    op = np.flatnonzero(cnt == 0) + lo
    d = np.diff(op); F = int(d.max()); x = int(op[int(np.argmax(d))])
    return dict(q=q, qq=qq, lo=lo, hi=hi, gears=gears, cnt=cnt, gsum=gsum, op=op, F=F, x=x,
                nrec=int((d == F).sum()))

say("Branch 5d.ii Q3/Q4: the window profile against the period profile.")
t0 = time.time()

# ---------- (a) distinct window record stretches
say("\n### (a) The distinct window record stretches, rungs 7..997")
say("F_W is INHERITED: the window at rung q is (q, q'^2], so the longest blocked stretch below q'^2")
say("stays the record until a longer one appears.  One row per distinct stretch.")
say("  x      F_W  rungs holding it   |gears|  |N|  sole  soleden  mincov  mincov set (<=12)      (ph5,ph7)")
seen = {}
order = []
for q in PR:
    if q < 7 or q > 997: continue
    w = window_data(q)
    key = (w['x'], w['F'])
    if key not in seen:
        seen[key] = dict(w=w, rungs=[q]); order.append(key)
    else:
        seen[key]['rungs'].append(q)
rowsA = []
for key in order:
    e = seen[key]; w = e['w']; x, F = key
    cols = list(range(x + 1, x + F))
    sole = {}
    for c in cols:
        st = [g for g in w['gears'] if strikes(c, g)]
        if len(st) == 1: sole.setdefault(st[0], []).append(c)
    mc, mcset = mincover(cols, w['gears'])
    top = x + F
    eff = [g for g in w['gears'] if g * g <= 6 * top + 1]
    rowsA.append(dict(x=x, F=F, rungs=e['rungs'], ngears=len(w['gears']), holders=sorted(sole),
                      nsole=sum(len(v) for v in sole.values()), mc=mc, mcset=mcset,
                      ph=( (x + 1) % 5, (x + 1) % 7 ), eff=len(eff), top=top,
                      maxsole=max(sole) if sole else 0))
    say(f"  {x:<7}{F:>4}  {e['rungs'][0]}..{e['rungs'][-1]:<12} {len(w['gears']):>5} {len(sole):>5} "
        f"{sum(len(v) for v in sole.values()):>5} {sum(len(v) for v in sole.values())/(F-1):>7.2f} "
        f"{mc:>6}  {str(mcset[:12]):<38} {((x+1)%5, (x+1)%7)}")
say(f"  distinct stretches: {len(rowsA)}; distinct (5,7) phases among them: "
    f"{len(set(r['ph'] for r in rowsA))} -> {sorted(set(r['ph'] for r in rowsA))}")
say(f"  sole-column density (sole columns / interior columns): "
    f"min {min(r['nsole']/(r['F']-1) for r in rowsA):.2f}, max {max(r['nsole']/(r['F']-1) for r in rowsA):.2f}, "
    f"mean {sum(r['nsole']/(r['F']-1) for r in rowsA)/len(rowsA):.2f}")
say(f"  minimum blocking set size: {[r['mc'] for r in rowsA]}")
say(f"  holders |N| vs gears: {[(r['F'], len(r['holders']), r['ngears']) for r in rowsA]}")

# ---------- (b) the period record, same quantities
say("\n### (b) The period record at m7..m23, same quantities")
say("  M        P          F   sole  soleden  mincov  mincov set        (ph5,ph7) of the first stretch")
ladder = [7, 11, 13, 17, 19, 23]
for i, q in enumerate(ladder):
    gears = [5] + ladder[:i + 1]; P = prod(gears)
    w = np.ones(P, bool)
    for g in gears:
        u = pow(6, -1, g); w[u::g] = False; w[g - u::g] = False
    op = np.flatnonzero(w); d = np.diff(np.concatenate([op, [op[0] + P]]))
    F = int(d.max()); starts = [int(op[j]) for j in np.flatnonzero(d == F)]
    phs = sorted(set(((x + 1) % 5, (x + 1) % 7) for x in starts))
    x = starts[0]; cols = list(range(x + 1, x + F))
    sole = {}
    for c in cols:
        st = [g for g in gears if strikes(c, g)]
        if len(st) == 1: sole.setdefault(st[0], []).append(c)
    mc, mcset = mincover(cols, gears)
    say(f"  {{5..{q:>2}}} {P:>10} {F:>5} {sum(len(v) for v in sole.values()):>5} "
        f"{sum(len(v) for v in sole.values())/(F-1):>7.2f} {mc:>6}  {str(mcset):<20} {phs}")
    say(f"        holders (gears with a sole column) = {sorted(sole)} of {gears}; "
        f"stretches {len(starts)}, all phases {phs}")

# ---------- (c) square-gate joint deletion
say("\n### (c) The square gate as a JOINT deletion (theorem, tested)")
say("Every blocked column k of the window has 6k-1 or 6k+1 composite below q'^2, hence a prime")
say("factor <= sqrt(6k+1): so the gears above sqrt(6*top+1) are JOINTLY droppable on the record")
say("stretch.  Test: remove them all at once and check the stretch is still blocked; then remove")
say("the individually-zero-drop set Z all at once and check the same.")
say("  q     F_W   gears  |above sqrt|  stretch survives?   |Z|  Z jointly droppable?")
for q in (59, 173, 499, 997):
    w = window_data(q); x, F = w['x'], w['F']
    cols = list(range(x + 1, x + F))
    top = x + F
    above = [g for g in w['gears'] if g * g > 6 * top + 1]
    keep = [g for g in w['gears'] if g not in above]
    ok = all(any(strikes(c, g) for g in keep) for c in cols)
    # zero-drop set (recomputed here, cheaply, from w)
    sole1 = (w['cnt'] == 1)
    idx = np.arange(x + 1 - w['lo'], x + F - w['lo'])
    holders = sorted(set(int(v) for v in w['gsum'][idx][sole1[idx]].tolist()))
    N = []
    for g in holders:
        opg = np.flatnonzero((w['cnt'] == 0) | (sole1 & (w['gsum'] == g))) + w['lo']
        if int(np.diff(opg).max()) < F: N.append(g)
    Z = [g for g in w['gears'] if g not in N]
    okZ = all(any(strikes(c, g) for g in N) for c in cols)
    say(f"  {q:<5} {F:>4}  {len(w['gears']):>5}  {len(above):>11}   {str(ok):<18} {len(Z):>4}  {okZ}")

# ---------- (d) drop vs drop_W gear by gear
say("\n### (d) The two profiles gear by gear, at the rungs where both exist")
say("drop(g) = F(M) - F(M minus g) over the full period; drop_W(g) the same inside the window.")
for q in (11, 13, 17, 19, 23):
    i = ladder.index(q); gears = [5] + ladder[:i + 1]; P = prod(gears)
    def rec(gs):
        if not gs: return 1
        p = prod(gs); a = np.ones(p, bool)
        for g in gs:
            u = pow(6, -1, g); a[u::g] = False; a[g - u::g] = False
        o = np.flatnonzero(a); return int(np.diff(np.concatenate([o, [o[0] + p]])).max())
    F = rec(gears)
    w = window_data(q); x, FW = w['x'], w['F']
    sole1 = (w['cnt'] == 1)
    prof = []
    for g in gears:
        Fg = rec([h for h in gears if h != g])
        opg = np.flatnonzero((w['cnt'] == 0) | (sole1 & (w['gsum'] == g))) + w['lo']
        FWg = int(np.diff(opg).max())
        prof.append((g, F - Fg, FW - FWg))
    say(f"  M = {{5..{q}}}: F = {F}, F_W = {FW} (window columns {w['lo']}..{w['hi']}, "
        f"record stretch at {x+1}..{x+FW-1})")
    say("    gear:      " + "  ".join(f"{g:>4}" for g, _, _ in prof))
    say("    drop:      " + "  ".join(f"{a:>4}" for _, a, _ in prof))
    say("    drop_W:    " + "  ".join(f"{b:>4}" for _, _, b in prof))

say(f"\n[{time.time() - t0:.1f}s]")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT)

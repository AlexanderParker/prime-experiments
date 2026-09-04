# Branch 5d.ii, question 2: the deletion profile of the WINDOW's own longest blocked stretch.
# At every rung q (prime, 7..997) with q' = next prime: window columns lo = q//6+1 .. hi = (q'^2-1)//6.
# F_W = the longest gap between consecutive openings BOTH of which lie in the window (max-gap
# convention, matching F); the record stretch is the blocked columns between them.
# For every gear g of {5..q}: the sole-strike columns of g inside every record stretch, the drop
# F_W(M) - F_W(M minus g), the zero set Z, and the JOINT deletion F_W(M minus Z).
# Cheap exactness: drop_W(g) > 0 iff g is the sole striker of an interior column of EVERY record
# stretch (removing g must split all of them); the value is then recomputed exactly.
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/w1_window_profile.py
import numpy as np, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "w1_window_profile.txt")
TSV = os.path.join(RES, "w1_rungs.tsv")
lines = []
def say(s=""):
    print(s); lines.append(s)

def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]

PR = primes_upto(1100)
def nextp(q):
    return PR[PR.index(q) + 1]

def maxgap(op):
    """max distance between consecutive entries; returns (F, list of left endpoints attaining it)"""
    if len(op) < 2: return None, []
    d = np.diff(op)
    F = int(d.max())
    return F, [int(op[j]) for j in np.flatnonzero(d == F)]

FOCUS = {59, 173, 499, 997}
# F ladder on record (research/proof/anchor_runs_zero.md Q3), for the F_W/F ratio
FLAD = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
        43: 103, 47: 118, 53: 145, 59: 161}

rows = []
say("Branch 5d.ii Q2: the window's own longest blocked stretch and what holds it up.")
say("Window at rung q: columns lo = q//6+1 .. hi = (q'^2-1)//6.  F_W = max gap between consecutive")
say("openings inside it.  drop_W(g) = F_W - F_W(M minus g) >= 0.  Z = the zero-drop gears.")
t0 = time.time()
for q in PR:
    if q < 7 or q > 997: continue
    qq = nextp(q)
    lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    gears = [g for g in PR if 5 <= g <= q]
    cnt = np.zeros(n, np.int16)
    gsum = np.zeros(n, np.int32)
    for g in gears:
        u = pow(6, -1, g)
        for t in (u, g - u):
            s = (t - lo) % g
            cnt[s::g] += 1
            gsum[s::g] += g
    op = np.flatnonzero(cnt == 0) + lo
    if len(op) < 2:
        say(f"rung {q}: fewer than two openings in the window -- skipped"); continue
    F_W, starts = maxgap(op)
    head, tail = int(op[0]) - lo, hi - int(op[-1])
    sole1 = (cnt == 1)
    # candidate holders: gears that are sole striker of an interior column of EVERY record stretch
    holders = None
    solecols = {}
    for x in starts:
        idx = np.arange(x + 1 - lo, x + F_W - lo)          # interior columns of the stretch
        m = sole1[idx]
        gs = gsum[idx][m]
        cols = idx[m] + lo
        here = {}
        for g, c in zip(gs.tolist(), cols.tolist()):
            here.setdefault(int(g), []).append(int(c))
        solecols[x] = here
        holders = set(here) if holders is None else (holders & set(here))
    drops = {}
    for g in sorted(holders):
        opg = np.flatnonzero((cnt == 0) | (sole1 & (gsum == g))) + lo
        Fg, _ = maxgap(opg)
        drops[g] = F_W - Fg
    N = sorted([g for g in drops if drops[g] > 0])
    Z = [g for g in gears if g not in N]
    # joint deletion of the whole zero set: keep only the gears of N
    if N:
        blocked = np.zeros(n, bool)
        for g in N:
            u = pow(6, -1, g)
            for t in (u, g - u):
                blocked[(t - lo) % g::g] = True
        opN = np.flatnonzero(~blocked) + lo
        F_WN, _ = maxgap(opN)
    else:
        F_WN = None
    x0 = starts[0]
    sq = int((6 * (x0 + F_W) + 1) ** 0.5)                  # square gate at the stretch's top
    above = [g for g in N if g * g > 6 * (x0 + F_W) + 1]
    rows.append(dict(q=q, qq=qq, lo=lo, hi=hi, W=hi, ngears=len(gears), nopen=len(op),
                     F_W=F_W, nrec=len(starts), x=x0, head=head, tail=tail,
                     frac=(x0 + 1 - lo) / max(1, hi - lo), distW=hi - (x0 + F_W),
                     nN=len(N), N=N, sqrt_top=sq, F_WN=F_WN, jointkept=(F_WN == F_W),
                     ph5=(x0 + 1) % 5, ph7=(x0 + 1) % 7, above_sq=above,
                     maxN=(max(N) if N else 0), F=FLAD.get(q)))
    if q in FOCUS:
        say(f"\n=== rung q = {q}, q' = {qq}: window columns {lo}..{hi} ({n} columns), "
            f"{len(gears)} gears, {len(op)} openings")
        say(f"  F_W = {F_W} at x = {x0} (stretch {x0+1}..{x0+F_W-1}, numbers {6*(x0+1)-1}..{6*(x0+F_W-1)+1}); "
            f"{len(starts)} stretch(es) attain it; head run {head}, tail run {tail}")
        say(f"  position: fraction {(x0 + 1 - lo) / max(1, hi - lo):.3f} of the window, "
            f"{hi - (x0 + F_W)} columns below W;  period record F(M) = {FLAD.get(q, '-')}")
        say(f"  square gate at the stretch's top: sqrt(6*hi_stretch+1) = {sq}, so every gear above "
            f"{sq} can have NO sole column there ({len([g for g in gears if g > sq])} of {len(gears)} gears)")
        say(f"  gears with a sole column in every record stretch (candidates): {sorted(holders)}")
        say("  gear   kills-in-stretch  sole cols (first 8)                     drop_W")
        for g in sorted(holders):
            cc = solecols[x0].get(g, [])
            u = pow(6, -1, g)
            kk = sum(1 for k in range(x0 + 1, x0 + F_W) if k % g in (u, g - u))
            say(f"  {g:>4}   {kk:>15}  {str([c - x0 for c in cc[:8]]):<38} {drops[g]:>4}")
        say(f"  NONZERO-drop set N = {N} (size {len(N)}); zero set Z = the other "
            f"{len(gears) - len(N)} gears")
        say(f"  joint deletion of the whole zero set Z: F_W(M minus Z) = {F_WN} "
            f"({'stretch survives' if F_WN == F_W else 'stretch does not survive'})")
        # what each gear of N owns, in numbers
        for g in N:
            cc = solecols[x0].get(g, [])
            say(f"    gear {g} sole columns in the record stretch: " +
                ", ".join(f"{c}=({6*c-1},{6*c+1})" for c in cc[:6]))

say(f"\n[{time.time() - t0:.1f}s]  rungs computed: {len(rows)}")

# ---- summary tables
say("\n### The window record and its holders, every rung")
say("  q    q'     W       F_W  F(M)  F_W/F  frac   distW  |N|  N (nonzero-drop gears)   sqrtTop  |Z|")
for r in rows:
    fr = f"{r['F_W'] / r['F']:.2f}" if r['F'] else "  - "
    say(f"  {r['q']:>4} {r['qq']:>4} {r['W']:>7}  {r['F_W']:>4} {str(r['F'] or '-'):>5} {fr:>6} "
        f"{r['frac']:.3f} {r['distW']:>7}  {r['nN']:>3}  {str(r['N'])[:26]:<26} {r['sqrt_top']:>5} "
        f"{r['ngears'] - r['nN']:>5}")

say("\n### Stability of the holder set")
allq = [r['q'] for r in rows]
for g in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]:
    elig = [r for r in rows if r['q'] >= g]
    hit = [r for r in elig if g in r['N']]
    say(f"  gear {g:>3}: nonzero drop at {len(hit):>3} of {len(elig):>3} rungs "
        f"({len(hit) / max(1, len(elig)):.3f}); last rung where it is needed: "
        f"{max([r['q'] for r in hit]) if hit else '-'}")
always = [g for g in [5, 7, 11, 13, 17, 19] if all(g in r['N'] for r in rows if r['q'] >= g)]
say(f"  ALWAYS set (nonzero drop at every eligible rung): {always}")
say(f"  rungs where N is exactly {{5}}: {[r['q'] for r in rows if r['N'] == [5]]}")
say(f"  |N| over the rungs: min {min(r['nN'] for r in rows)}, max {max(r['nN'] for r in rows)}, "
    f"mean {sum(r['nN'] for r in rows) / len(rows):.2f}")
say(f"  |N| by band: " + "  ".join(
    f"q<{b}: {sum(r['nN'] for r in rows if r['q'] < b) / max(1, len([r for r in rows if r['q'] < b])):.2f}"
    for b in (30, 100, 300, 1000)))
say(f"  rungs where max(N) > sqrt(6*top+1): {[r['q'] for r in rows if r['above_sq']]} (must be empty: square gate)")
say(f"  joint deletion of Z keeps the record stretch at "
    f"{sum(1 for r in rows if r['jointkept'])} of {len(rows)} rungs")
say(f"  consecutive-rung overlap of N: " +
    f"{sum(len(set(a['N']) & set(b['N'])) / max(1, len(set(a['N']) | set(b['N'])))for a, b in zip(rows, rows[1:])) / (len(rows) - 1):.3f} mean Jaccard")
ph = {}
for r in rows:
    ph[(r['ph5'], r['ph7'])] = ph.get((r['ph5'], r['ph7']), 0) + 1
say(f"  (5,7) phase of the window record ((x+1) mod 5, mod 7): {len(ph)} distinct values over "
    f"{len(rows)} rungs; counts {sorted(ph.items(), key=lambda t: -t[1])[:8]}")
say(f"  F_W < F(M) at {sum(1 for r in rows if r['F'] and r['F_W'] < r['F'])} of "
    f"{sum(1 for r in rows if r['F'])} rungs with F on record; violations: "
    f"{[(r['q'], r['F_W'], r['F']) for r in rows if r['F'] and r['F_W'] >= r['F']]}")
say(f"  position: upper half of the window at {sum(1 for r in rows if r['frac'] > 0.5)} of {len(rows)} rungs")

with open(TSV, "w") as f:
    f.write("q\tqp\tlo\thi\tngears\tnopen\tF_W\tnrec\tx\thead\ttail\tfrac\tdistW\tnN\tN\tsqrt_top\tF_WN\tph5\tph7\tF\n")
    for r in rows:
        f.write(f"{r['q']}\t{r['qq']}\t{r['lo']}\t{r['hi']}\t{r['ngears']}\t{r['nopen']}\t{r['F_W']}\t"
                f"{r['nrec']}\t{r['x']}\t{r['head']}\t{r['tail']}\t{r['frac']:.4f}\t{r['distW']}\t"
                f"{r['nN']}\t{','.join(map(str, r['N']))}\t{r['sqrt_top']}\t{r['F_WN']}\t{r['ph5']}\t"
                f"{r['ph7']}\t{r['F'] or ''}\n")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT, "and", TSV)

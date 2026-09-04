# Branch 5d.ii, question 4: is there a gear whose sole strikes are ALWAYS in the window's longest
# stretch, and one whose sole strikes are NEVER there?
# (1) per rung: the largest drop_W and which gear attains it (does removing gear 5 take the window
#     record all the way down to the window's runner-up stretch?);
# (2) the rungs where gear 5 or gear 7 is NOT needed;
# (3) over the 13 distinct window record stretches: the union of the holder sets, and the gears
#     eligible at the top rung that hold up no window record anywhere (the NEVER set);
# (4) F_W against F({5..z}), z = sqrt(6*top+1), the effective machine at the stretch's top -- the
#     bound "a window stretch ending at x is a blocked stretch of the effective machine".
# Self-contained; numpy only.  Run: uv run python research/anchor235/r35/q4_object.py
import numpy as np, os, time
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "q4_object.txt")
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

FLAD = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
        43: 103, 47: 118, 53: 145, 59: 161}

say("Branch 5d.ii Q4: always and never, on the window's longest stretch.")
t0 = time.time()
rows = []
for q in PR:
    if q < 7 or q > 997: continue
    qq = nextp(q); lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    gears = [g for g in PR if 5 <= g <= q]
    cnt = np.zeros(n, np.int16); gsum = np.zeros(n, np.int32)
    for g in gears:
        u = pow(6, -1, g)
        for t in (u, g - u):
            s = (t - lo) % g; cnt[s::g] += 1; gsum[s::g] += g
    op = np.flatnonzero(cnt == 0) + lo
    d = np.diff(op); F = int(d.max()); starts = [int(op[j]) for j in np.flatnonzero(d == F)]
    runner = int(np.sort(d)[-1 - (d == F).sum()]) if (d == F).sum() < len(d) else 0
    sole1 = (cnt == 1)
    hold = None; per = {}
    for x in starts:
        idx = np.arange(x + 1 - lo, x + F - lo)
        gs = set(int(v) for v in gsum[idx][sole1[idx]].tolist())
        per[x] = gs
        hold = gs if hold is None else (hold & gs)
    drops = {}
    for g in sorted(hold):
        opg = np.flatnonzero((cnt == 0) | (sole1 & (gsum == g))) + lo
        drops[g] = F - int(np.diff(opg).max())
    N = {g: v for g, v in drops.items() if v > 0}
    mx = max(N.values()) if N else 0
    arg = [g for g in N if N[g] == mx]
    rows.append(dict(q=q, F=F, x=starts[0], top=starts[0] + F, runner=runner, N=sorted(N),
                     maxdrop=mx, arg=arg, hold=sorted(hold), ceil=F - runner))
say("\n### (1) the largest single drop and who attains it")
say("PRE-REGISTERED IN PASSING AND REFUTED HERE: 'F_W minus the window's runner-up gap is a")
say("ceiling on every drop' is FALSE -- removing a gear opens columns everywhere in the window,")
say("so it shortens the runner-up stretch too.  Columns: rung, F_W, runner-up gap, the false")
say("ceiling, the largest drop, the gear(s) attaining it.")
say("  q     F_W  runner  (false)  maxdrop  argmax gears")
prev = None
for r in rows:
    if (r['x'], r['F']) != prev:
        say(f"  {r['q']:>4} {r['F']:>5} {r['runner']:>7} {r['ceil']:>8} {r['maxdrop']:>8}  {r['arg']}")
        prev = (r['x'], r['F'])
att = sum(1 for r in rows if r['maxdrop'] == r['ceil'])
say(f"  drops ABOVE the false ceiling at {sum(1 for r in rows if r['maxdrop'] > r['ceil'])} of "
    f"{len(rows)} rungs (the refutation); equal to it at {att}; gear 5 is an argmax at "
    f"{sum(1 for r in rows if 5 in r['arg'])} rungs, gear 7 at {sum(1 for r in rows if 7 in r['arg'])}")

say("\n### (2) where gear 5 and gear 7 are NOT needed")
for g in (5, 7, 11, 13):
    miss = [r['q'] for r in rows if g not in r['N']]
    say(f"  gear {g:>3}: zero drop at rungs {miss[:20]}{' ...' if len(miss) > 20 else ''} "
        f"({len(miss)} of {len(rows)})")
    for r in rows:
        if g not in r['N'] and g in r['hold']:
            say(f"      (rung {r['q']}: gear {g} HAS a sole column in the record stretch but "
                f"removing it leaves an equally long stretch elsewhere)")
            break

say("\n### (3) the distinct stretches, their holders, and the never set")
seen = {}
for r in rows:
    seen.setdefault((r['x'], r['F']), r)
uni = set()
for k, r in seen.items():
    uni |= set(r['hold'])
    say(f"  x = {r['x']:>7} F_W = {r['F']:>4} (first at rung {r['q']}): holders {r['hold']}")
top_gears = [g for g in PR if 5 <= g <= 997]
never = [g for g in top_gears if g not in uni]
say(f"  union of holders over the 13 stretches: {len(uni)} gears, max {max(uni)}")
say(f"  NEVER set (a gear <= 997 that holds up no window record anywhere in 7..997): "
    f"{len(never)} gears, smallest {never[:12]}")
say(f"  the never set as a share of the machine at rung 997: {len(never)}/{len(top_gears)} = "
    f"{len(never)/len(top_gears):.3f}")

say("\n### (3b) how many sole columns each holder owns (period versus window)")
say("In the period record every gear owns SEVERAL sole columns; in the window most holders own")
say("exactly one, so the window drop is set by WHERE that column sits, not by the gear's size.")
for k, r in sorted(seen.items()):
    gears2 = [g for g in PR if 5 <= g <= r['q']]
    cc = {}
    for c in range(r['x'] + 1, r['top']):
        st = [g for g in gears2 if c % g in (pow(6, -1, g), g - pow(6, -1, g))]
        if len(st) == 1: cc[st[0]] = cc.get(st[0], 0) + 1
    v = sorted(cc.values(), reverse=True)
    say(f"  window stretch x = {r['x']:>7} (F_W = {r['F']:>4}): sole counts {v}; "
        f"holders owning exactly one: {sum(1 for t in v if t == 1)} of {len(v)}")
ladder2 = [7, 11, 13, 17, 19, 23]
for i, q3 in enumerate(ladder2):
    gs = [5] + ladder2[:i + 1]; P3 = prod(gs)
    a = np.ones(P3, bool)
    for g in gs:
        u = pow(6, -1, g); a[u::g] = False; a[g - u::g] = False
    o = np.flatnonzero(a); dd = np.diff(np.concatenate([o, [o[0] + P3]]))
    Fp = int(dd.max()); x0 = int(o[int(np.argmax(dd))])
    cc = {}
    for c in range(x0 + 1, x0 + Fp):
        st = [g for g in gs if c % g in (pow(6, -1, g), g - pow(6, -1, g))]
        if len(st) == 1: cc[st[0]] = cc.get(st[0], 0) + 1
    v = sorted(cc.values(), reverse=True)
    say(f"  period record {{5..{q3}}} (F = {Fp}): sole counts {v}; "
        f"holders owning exactly one: {sum(1 for t in v if t == 1)} of {len(v)}; gears {len(gs)}")

say("\n### (4) F_W against the record of the effective machine at the stretch's top")
say("A blocked window column k has 6k-1 or 6k+1 composite below q'^2, hence a prime factor")
say("<= sqrt(6k+1): so a window stretch ending at column x is a blocked stretch of {5..z},")
say("z = sqrt(6x+1), and F_W <= F({5..z}).  Where F({5..z}) is on record:")
say("  first rung   x..top      F_W    z = sqrt(6*top+1)   effective machine   F({5..z})  F_W/F")
for k, r in sorted(seen.items()):
    z = int((6 * r['top'] + 1) ** 0.5)
    eff = [g for g in PR if 5 <= g <= z]
    y = eff[-1] if eff else 0
    Fz = FLAD.get(y)
    rat = f"{r['F'] / Fz:.3f}" if Fz else "-"
    say(f"  {r['q']:>10}   {r['x']}..{r['top']:<10} {r['F']:>4}   {z:>5}              "
        f"{{5..{y}}}         {str(Fz or '-'):>6}   {rat}")

say(f"\n[{time.time() - t0:.1f}s]")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT)

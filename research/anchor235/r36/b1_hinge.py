# Branch 5g, Theory B (the hinge) and Theory A's window half (A4).
#
# At every prime rung q = 23..1999 (q' = next prime, window columns lo = q//6+1 .. hi = (q'^2-1)/6):
#   * the window's longest blocked stretch (max gap between consecutive openings both inside the
#     window), and its second- and third-longest by DISTINCT length;
#   * per column of a stretch, the number of gears striking it; MIN-STRIKERS = the smallest;
#     a HINGE is a column with exactly one striker, its HINGE GEAR that striker;
#   * g_h = the largest hinge gear; its column, its relative position, its split value;
#   * the length rules  L <= g_h/3,  L <= 2 g_h/3,  L <= g_h;
#   * the coverage profile r_g = c_g/m_g of the stretch (Theory A's A4), with the share of
#     striking gears whose m_g is 1 (for which "at coverage maximum" is vacuous).
#
# m_g(L) = max over phases of c_g is computed in closed form: the maximum count of a periodic
# 2-point set in a window of length L is attained at a window starting on a point, so
# m_g(L) = max(c_g at phase u, c_g at phase g-u).  Asserted against brute force for g <= 200.
#
# Self-contained; numpy only.  Run: uv run python research/anchor235/r36/b1_hinge.py
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "b1_hinge.txt")
TSV = os.path.join(RES, "b1_rungs.tsv")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


PR = primes_upto(2100)
NXT = {PR[i]: PR[i + 1] for i in range(len(PR) - 1)}


def teeth(g):
    u = pow(6, -1, g)
    return (u, g - u)


def cov(s, L, g):
    n = 0
    for t in teeth(g):
        off = (t - s) % g
        if off < L:
            n += 1 + (L - 1 - off) // g
    return n


def maxcov(L, g):
    a, b = teeth(g)
    return max(cov(a, L, g), cov(b, L, g))


# gate the closed form
for g in PR:
    if g < 5 or g > 200:
        continue
    for L in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
        assert maxcov(L, g) == max(cov(r, L, g) for r in range(g)), (g, L)
say("Branch 5g / Theory B (the hinge) + Theory A's window half.")
say("Gate: the closed form for m_g(L) matches brute force at every gear 5..199 and 12 lengths.")
say("")

FOCUS = {59, 173, 499, 997, 1999}
rows = []
t0 = time.time()
for q in PR:
    if q < 23 or q > 1999:
        continue
    qq = NXT[q]
    lo, hi = q // 6 + 1, (qq * qq - 1) // 6
    n = hi - lo + 1
    gears = [g for g in PR if 5 <= g <= q]
    cnt = np.zeros(n, np.int16)
    gsum = np.zeros(n, np.int64)
    for g in gears:
        for t in teeth(g):
            s = (t - lo) % g
            cnt[s::g] += 1
            gsum[s::g] += g
    op = np.flatnonzero(cnt == 0) + lo
    if len(op) < 4:
        continue
    d = np.diff(op)
    lens = sorted(set(d.tolist()), reverse=True)[:3]
    rec = {}
    for rank, D in enumerate(lens):
        idx = np.flatnonzero(d == D)
        starts = [int(op[j]) + 1 for j in idx]           # blocked stretch starts
        L = D - 1
        s = starts[0]
        i0 = s - lo
        c = cnt[i0:i0 + L]
        gs = gsum[i0:i0 + L]
        mn = int(c.min())
        hin = np.flatnonzero(c == 1)
        hg = [(int(gs[j]), int(j)) for j in hin]         # (gear, offset in stretch)
        gh, oh = (max(hg) if hg else (0, -1))
        # split value of each hinge gear: longest surviving blocked run after removing it
        def split(gear):
            offs = [j for gg, j in hg if gg == gear]
            pts = [-1] + offs + [L]
            return max(pts[i + 1] - pts[i] - 1 for i in range(len(pts) - 1))
        best_drop, best_gear, best_off = -1, 0, -1
        for gg, j in hg:
            dr = L - split(gg)
            if dr > best_drop:
                best_drop, best_gear, best_off = dr, gg, j
        rec[rank] = dict(D=D, L=L, s=s, ntie=len(starts), minstr=mn,
                         nhinge=len(hg), gh=gh, oh=oh,
                         pos=(oh / (L - 1) if L > 1 else 0.5),
                         hinges=hg, bdrop=best_drop, bgear=best_gear,
                         bpos=(best_off / (L - 1) if L > 1 else 0.5),
                         maxstr=int(c.max()), meanstr=float(c.mean()))
    r0 = rec[0]
    L = r0["L"]
    # coverage profile of the longest stretch (A4)
    prof = []
    for g in gears:
        cg = cov(r0["s"], L, g)
        if cg == 0:
            continue
        mg = maxcov(L, g)
        prof.append((g, cg, mg, cg / mg))
    triv = sum(1 for g, cg, mg, r in prof if mg == 1)
    rows.append(dict(q=q, qq=qq, lo=lo, hi=hi, ngears=len(gears), nopen=len(op),
                     rec=rec, nstrike=len(prof), triv=triv, prof=prof))
    if q in FOCUS:
        say(f"=== rung q = {q}, q' = {qq}: window {lo}..{hi} ({n} columns), {len(gears)} gears, "
            f"{len(op)} openings")
        for rank in sorted(rec):
            r = rec[rank]
            say(f"  #{rank+1} longest: gap {r['D']}, L = {r['L']} blocked columns from "
                f"{r['s']} ({r['ntie']} stretch(es) of this length); strikers per column "
                f"min {r['minstr']} mean {r['meanstr']:.2f} max {r['maxstr']}; "
                f"{r['nhinge']} hinge column(s)")
            say(f"     hinge gears (gear @ offset): "
                + ", ".join(f"{g}@{j}" for g, j in sorted(r['hinges'])[:14])
                + ("..." if len(r['hinges']) > 14 else ""))
            say(f"     largest hinge gear g_h = {r['gh']} at offset {r['oh']} of {r['L']} "
                f"(position {r['pos']:.2f}); g_h/q = {r['gh']/q:.3f}; "
                f"3L/(2 g_h) = {3*r['L']/(2*r['gh']):.3f}; L/g_h = {r['L']/r['gh']:.3f}")
            say(f"     largest-split hinge: gear {r['bgear']} at offset {r['bpos']:.2f}, "
                f"drop {r['bdrop']} of {r['L']}")
        say(f"  A4: of the {len(prof)} gears that strike the longest stretch, {triv} have "
            f"m_g = 1 (share {triv/len(prof):.3f}) so 'at maximum' is vacuous for them")
        say(f"     r_g by band g/q: " + "  ".join(
            f"[{a:.1f},{b:.1f}) n={sum(1 for g,_,_,_ in prof if a<=g/q<b)} "
            f"mean={np.mean([r for g,_,_,r in prof if a<=g/q<b]):.2f}"
            for a, b in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
            if any(a <= g / q < b for g, _, _, _ in prof)))
        say(f"     the bottom gears: " + "  ".join(
            f"{g}:{cg}/{mg}={r:.2f}" for g, cg, mg, r in prof[:8]))
        say("")
    del cnt, gsum, op

say(f"[{time.time()-t0:.1f}s]  rungs computed: {len(rows)}")
say("")

# ------------------------------------------------------------------ scoring
say("=== B1.  min-strikers over the columns of the window's longest stretch")
bad = [r['q'] for r in rows if r['rec'][0]['minstr'] != 1]
say(f"  min-strikers = 1 at {len(rows)-len(bad)} of {len(rows)} rungs; exceptions {bad}")
for rank in (1, 2):
    b2 = [r['q'] for r in rows if rank in r['rec'] and r['rec'][rank]['minstr'] != 1]
    say(f"  #{rank+1} longest stretch: exceptions {b2} "
        f"(of {sum(1 for r in rows if rank in r['rec'])} rungs)")

say("")
say("=== B2.  the size of the largest hinge gear g_h")
gh = np.array([r['rec'][0]['gh'] / r['q'] for r in rows])
say(f"  g_h/q over {len(rows)} rungs: min {gh.min():.3f}, median {np.median(gh):.3f}, "
    f"mean {gh.mean():.3f}, max {gh.max():.3f}")
for thr in (0.25, 0.5, 0.75, 0.9):
    say(f"  share of rungs with g_h > {thr:.2f} q: {(gh > thr).mean():.3f}")
say(f"  rungs where g_h < q/4: {[r['q'] for r in rows if r['rec'][0]['gh'] < r['q']/4][:20]}")

say("")
say("=== B3.  where the hinge sits")
pos = np.array([r['rec'][0]['pos'] for r in rows])
bpos = np.array([r['rec'][0]['bpos'] for r in rows])
allpos = np.array([j / max(1, r['rec'][0]['L'] - 1)
                   for r in rows for g, j in r['rec'][0]['hinges']])
say(f"  position of the g_h hinge: central band [0.25,0.75] at {np.mean((pos>=.25)&(pos<=.75)):.3f} "
    f"of rungs; mean {pos.mean():.3f}")
say(f"  position of the largest-split hinge: central band at "
    f"{np.mean((bpos>=.25)&(bpos<=.75)):.3f} of rungs; mean {bpos.mean():.3f}")
say(f"  position of ALL hinges pooled ({len(allpos)} columns): central band at "
    f"{np.mean((allpos>=.25)&(allpos<=.75)):.3f}; mean {allpos.mean():.3f}")
say(f"  number of hinges per longest stretch: min {min(r['rec'][0]['nhinge'] for r in rows)}, "
    f"median {int(np.median([r['rec'][0]['nhinge'] for r in rows]))}, "
    f"max {max(r['rec'][0]['nhinge'] for r in rows)}")

say("")
say("=== B4 / B6.  the length rules")
for name, f in (("L <= g_h/3", lambda r: 3 * r['L'] <= r['gh']),
                ("L <= 2 g_h/3", lambda r: 3 * r['L'] <= 2 * r['gh']),
                ("L <= g_h", lambda r: r['L'] <= r['gh'])):
    fails = [(r['q'], r['rec'][0]['L'], r['rec'][0]['gh']) for r in rows if not f(r['rec'][0])]
    say(f"  {name:<14}: holds at {len(rows)-len(fails)} of {len(rows)} rungs; "
        f"first failures {fails[:6]}")
rat = np.array([2 * r['rec'][0]['gh'] / (3 * r['rec'][0]['L']) for r in rows])
say(f"  ratio 2 g_h/(3L): min {rat.min():.3f} (rung "
    f"{rows[int(rat.argmin())]['q']}), median {np.median(rat):.3f}, max {rat.max():.3f}")
say(f"  ratio g_h/L: min {min(r['rec'][0]['gh']/r['rec'][0]['L'] for r in rows):.3f}, "
    f"median {np.median([r['rec'][0]['gh']/r['rec'][0]['L'] for r in rows]):.3f}")
say(f"  worst ten rungs by 2 g_h/(3L): " + ", ".join(
    f"q={rows[i]['q']} L={rows[i]['rec'][0]['L']} g_h={rows[i]['rec'][0]['gh']} "
    f"({rat[i]:.2f})" for i in np.argsort(rat)[:10]))

say("")
say("=== B5.  the same rules on the 2nd and 3rd longest stretches")
for rank in (1, 2):
    sub = [r for r in rows if rank in r['rec']]
    f2 = [(r['q'], r['rec'][rank]['L'], r['rec'][rank]['gh']) for r in sub
          if 3 * r['rec'][rank]['L'] > 2 * r['rec'][rank]['gh']]
    g2 = np.array([r['rec'][rank]['gh'] / r['q'] for r in sub])
    say(f"  #{rank+1}: L <= 2 g_h/3 holds at {len(sub)-len(f2)} of {len(sub)}; failures {f2[:8]}")
    say(f"      g_h/q median {np.median(g2):.3f}; central-band position "
        f"{np.mean([(r['rec'][rank]['pos']>=.25)and(r['rec'][rank]['pos']<=.75) for r in sub]):.3f}")

say("")
say("=== A4.  the window profile is degenerate at the top")
say("  q     L    gears striking   with m_g = 1   share   mean r_g (all)   mean r_g (m_g>=2)")
for r in rows:
    if r['q'] not in FOCUS and r['q'] not in (23, 101, 307, 701, 1301):
        continue
    p = r['prof']
    nz = [x[3] for x in p if x[2] >= 2]
    say(f"  {r['q']:<5} {r['rec'][0]['L']:<4} {r['nstrike']:<16} {r['triv']:<14} "
        f"{r['triv']/r['nstrike']:<7.3f} {np.mean([x[3] for x in p]):<16.3f} "
        f"{(np.mean(nz) if nz else float('nan')):.3f}")
sh = np.array([r['triv'] / r['nstrike'] for r in rows])
say(f"  share with m_g = 1, over all {len(rows)} rungs: min {sh.min():.3f}, "
    f"median {np.median(sh):.3f}, max {sh.max():.3f}")

with open(TSV, "w", encoding="utf-8") as f:
    f.write("q\tqp\tL\tstart\tntie\tminstr\tnhinge\tgh\tgh_off\tgh_pos\tbgear\tbdrop\t"
            "nstrike\ttriv\tL2\tgh2\tL3\tgh3\n")
    for r in rows:
        a = r['rec'][0]
        b = r['rec'].get(1)
        c = r['rec'].get(2)
        f.write(f"{r['q']}\t{r['qq']}\t{a['L']}\t{a['s']}\t{a['ntie']}\t{a['minstr']}\t"
                f"{a['nhinge']}\t{a['gh']}\t{a['oh']}\t{a['pos']:.4f}\t{a['bgear']}\t"
                f"{a['bdrop']}\t{r['nstrike']}\t{r['triv']}\t"
                f"{b['L'] if b else ''}\t{b['gh'] if b else ''}\t"
                f"{c['L'] if c else ''}\t{c['gh'] if c else ''}\n")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", OUT, "and", TSV)

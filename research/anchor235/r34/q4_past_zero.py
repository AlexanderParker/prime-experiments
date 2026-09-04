# Branch 7d, question 4: openings in (0, W] against a typical stretch of the same length deep in the period.
# For every prime Q up to 997: W = (Q'^2-1)/6, openings of {5..Q} in (0, W] by direct residue sieving,
# checked against the twin pairs in (Q, Q'^2]; the period mean W * prod(1-2/g) (exact, CRT); the
# distribution of the count over ALL stretches of length W in the full period for Q <= 23 (sliding sum),
# and over 4000 random stretches (uniform start in [0,P) = independent uniform residues per gear) for Q > 23.
# Mechanism: the shield (columns 1..(Q-1)/6 all blocked), the per-gear first exclusive kill at g^2, and the
# band-by-band comparison against the EFFECTIVE machine {5..sqrt(6k+1)}.
# Self-contained; numpy only.  Run: uv run python research/anchor235/r34/q4_past_zero.py
import numpy as np, os, time
from math import prod, log, exp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "q4_past_zero.txt")
lines = []
def say(s=""):
    print(s); lines.append(s)

def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(x) for x in np.flatnonzero(s)]

rng = np.random.default_rng(7)
allp = primes_upto(1100)
levels = [q for q in allp if 7 <= q <= 997]
t0 = time.time()
say("Q4: openings in (0, W] against the period mean and against typical stretches of length W.")
say("cols: Q Q' W | open(0,W] twins(Q,Q'^2] | mean=W*prod(1-2/g) ratio | shield=(Q-1)//6 blocked cols | typical stretch: mean, min, max, percentile of the zero stretch | eff-machine prediction")
say("eff-machine prediction = sum over k in (0,W] of prod_{5<=g<=sqrt(6k+1)}(1-2/g): the density if the gears above sqrt(6k+1) are silent at k.")
res = []
for Q in levels:
    Qn = allp[allp.index(Q) + 1]; W = (Qn * Qn - 1) // 6
    gears = [g for g in allp if 5 <= g <= Q]
    w = np.ones(W + 1, bool); w[0] = False
    for g in gears:
        u = pow(6, -1, g); w[u::g] = False; w[g - u::g] = False
    n_open = int(w.sum())
    # twins in (Q, Q'^2]
    N = 6 * W + 2
    sv = np.ones(N + 1, bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    ks = np.arange(1, W + 1)
    # openings in (0, W] = twin pairs with Q < 6k-1, 6k+1 < Q'^2, plus the square gate at column W (Q'^2-2 prime)
    tw = int((sv[6 * ks - 1] & sv[6 * ks + 1] & (6 * ks - 1 > Q) & (6 * ks + 1 < Qn * Qn)).sum()) + int(sv[Qn * Qn - 2])
    dens = prod(1 - 2 / g for g in gears); mean = W * dens
    shield = (Q - 1) // 6
    # effective-machine prediction
    lg = np.array(gears); pref = np.cumprod(1 - 2 / lg)      # pref[i] = prod over gears[:i+1]
    sq = np.sqrt(6 * ks + 1)
    idx = np.searchsorted(lg, sq, side="right") - 1          # index of largest gear <= sqrt(6k+1)
    eff = np.where(idx >= 0, pref[np.clip(idx, 0, None)], 1.0)
    eff_pred = float(eff.sum())
    # typical stretches
    if Q <= 23:
        P = prod(gears); full = np.ones(P, bool)
        for g in gears:
            u = pow(6, -1, g); full[u::g] = False; full[g - u::g] = False
        cs = np.concatenate([[0], np.cumsum(full, dtype=np.int64)])
        # stretch starting at s covers columns s+1..s+W (so the zero stretch is s=0); cyclic via doubling
        full2 = np.concatenate([full, full[:W + 1]]); cs2 = np.concatenate([[0], np.cumsum(full2, dtype=np.int64)])
        counts = cs2[W + 1: W + 1 + P] - cs2[1: 1 + P]
        typ = (float(counts.mean()), int(counts.min()), int(counts.max()), float((counts < n_open).mean()), float((counts <= n_open).mean()))
        kind = "full"
    elif Q in (29, 31, 37, 41, 43, 47, 53, 59, 61, 71, 101, 173, 199, 251, 307, 401, 499, 601, 701, 797, 907, 997):
        S = 1000 if Q > 200 else 4000
        cnt = np.zeros(S, np.int64)
        for s in range(S):
            ww = np.ones(W + 1, bool); ww[0] = False
            for g in gears:
                u = pow(6, -1, g); r = int(rng.integers(g))       # r = start mod g
                for t in (u, g - u):
                    first = (t - r) % g
                    ww[first::g] = False
            cnt[s] = ww.sum()
        typ = (float(cnt.mean()), int(cnt.min()), int(cnt.max()), float((cnt < n_open).mean()), float((cnt <= n_open).mean()))
        kind = f"mc{S}"
    else:
        typ = (mean, -1, -1, float('nan'), float('nan')); kind = "mean only"
    res.append((Q, Qn, W, n_open, tw, mean, shield, typ, kind, eff_pred))
    if Q <= 60 or Q in (61, 71, 101, 173, 199, 251, 307, 401, 499, 601, 701, 797, 907, 997):
        say(f"Q={Q:>4} Q'={Qn:>4} W={W:>7} | open {n_open:>6} twins {tw:>6} {'OK' if tw == n_open else 'MISMATCH'} | mean {mean:>9.1f} ratio {n_open / mean:.3f} | shield {shield:>3} | typical {typ[0]:>8.1f} [{typ[1]}, {typ[2]}] pct< {typ[3]:.3f} pct<= {typ[4]:.3f} ({kind}) | eff {eff_pred:>8.1f} ratio {n_open / eff_pred:.3f}")
say(f"\nelapsed {time.time() - t0:.1f}s")
# the trend of the ratio
say("\nRatio open(0,W]/mean by band of Q (mean over primes in the band), and the limiting value 1/(2 e^-gamma)^2 = %.4f:" % (1 / (2 * exp(-0.5772156649)) ** 2))
Qs = np.array([r[0] for r in res]); rat = np.array([r[3] / r[5] for r in res]); rat_eff = np.array([r[3] / r[9] for r in res])
for a, b in ((7, 30), (30, 60), (60, 120), (120, 250), (250, 500), (500, 1000)):
    m = (Qs >= a) & (Qs < b)
    say(f"  Q in [{a},{b}): n={int(m.sum())} ratio mean {rat[m].mean():.3f} min {rat[m].min():.3f} max {rat[m].max():.3f} | vs eff-machine prediction mean {rat_eff[m].mean():.3f}")
say("Count of levels where the zero stretch has MORE openings than the period mean: %d of %d; fewer: %d" % (
    sum(1 for r in res if r[3] > r[5]), len(res), sum(1 for r in res if r[3] < r[5])))
sampled = [r for r in res if r[8] != "mean only"]
say("At the %d sampled levels, the zero stretch's count is below the typical-stretch MEDIAN (pct<= below 0.5) at %d, above at %d" % (
    len(sampled), sum(1 for r in sampled if r[7][4] < 0.5), sum(1 for r in sampled if r[7][3] > 0.5)))

# Mechanism at four levels: which gears have struck exclusively by column k; first exclusive kill of each gear
say("\nMechanism at Q = 59, 173, 499, 997: gears whose first exclusive kill lies at or above column (g^2-1)/6 (their square), share of each gear's strikes in (0,W] that are wasted (land on a column another gear also strikes),")
say("and the band-by-band local density of openings against the full-machine mean density and the effective-machine density.")
for Q in (59, 173, 499, 997):
    Qn = allp[allp.index(Q) + 1]; W = (Qn * Qn - 1) // 6
    gears = [g for g in allp if 5 <= g <= Q]
    ns = np.zeros(W + 1, np.int32)
    for g in gears:
        u = pow(6, -1, g); ns[u::g] += 1; ns[g - u::g] += 1
    say(f"\n  Q={Q}: W={W}, shield columns 1..{(Q-1)//6} blocked: {'yes' if (ns[1:(Q-1)//6+1] > 0).all() else 'NO'}; first opening d_0 = {int(np.flatnonzero(ns[1:] == 0)[0]) + 1}")
    firsts = []
    for g in gears:
        u = pow(6, -1, g)
        cols = np.sort(np.concatenate([np.arange(u, W + 1, g), np.arange(g - u, W + 1, g)]))
        excl = cols[ns[cols] == 1]
        sqcol = (g * g - 1) // 6
        fe = int(excl[0]) if len(excl) else None
        firsts.append((g, fe, sqcol, len(cols), len(excl)))
    below_sq = [(g, fe, sqcol) for g, fe, sqcol, _, _ in firsts if fe is not None and fe < sqcol]
    say(f"    gears whose first exclusive kill is BELOW their square column (g^2-1)/6: {below_sq}  (expected: only self-pairs at the bottom edge)")
    at_sq = sum(1 for g, fe, sqcol, _, _ in firsts if fe == sqcol)
    say(f"    gears whose first exclusive kill is exactly their square column (g^2-2, g^2) with g^2-2 prime: {at_sq} of {len(gears)}; gears with no exclusive kill in (0,W]: {[g for g, fe, _, _, _ in firsts if fe is None]}")
    say("    g, first excl col, square col, strikes in (0,W], exclusive: " + "; ".join(f"{g}:{fe}/{sq}/{n}/{e}" for g, fe, sq, n, e in firsts[:8]) + " ... " + "; ".join(f"{g}:{fe}/{sq}/{n}/{e}" for g, fe, sq, n, e in firsts[-6:]))
    # band-by-band local density
    dens_full = prod(1 - 2 / g for g in gears)
    lg = np.array(gears); pref = np.cumprod(1 - 2 / lg)
    edges = [0] + [int(W * f) for f in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)]
    say(f"    band (cols)            open   full-mean   ratio   eff-mean   ratio   gears silent (g^2 > 6k+1 at band top)")
    for a, b in zip(edges, edges[1:]):
        if b <= a: continue
        kk = np.arange(a + 1, b + 1)
        n_o = int((ns[a + 1:b + 1] == 0).sum())
        sq = np.sqrt(6 * kk + 1); idx = np.searchsorted(lg, sq, side="right") - 1
        eff = np.where(idx >= 0, pref[np.clip(idx, 0, None)], 1.0).sum()
        silent = sum(1 for g in gears if g * g > 6 * b + 1)
        say(f"    {a + 1:>7}..{b:<7}  {n_o:>7}  {len(kk) * dens_full:>9.1f}  {n_o / (len(kk) * dens_full):>6.3f}  {eff:>9.1f}  {n_o / eff if eff else float('nan'):>6.3f}   {silent}")
with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
print("written", OUT)

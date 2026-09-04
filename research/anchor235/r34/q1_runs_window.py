# Branch 7d, question 1: the window of level Q in run units, per gear.
# Window at level Q = numbers (Q, Q'^2], Q' the next prime; columns k in (Q/6, W], W = (Q'^2-1)/6.
# Column k = (6k-1, 6k+1). Gear g strikes k iff k = +-u_g mod g, 6 u_g = g -+ 1.
# A run of gear g over the anchor 2,3,5 = 30g numbers = 5g columns; g's six hits per run are
# its strikes on anchor-open columns (k mod 5 in {0,2,3}).
# For each gear g <= Q: runs covered by the window (W_len / 5g), hits landing inside it, and
# exclusive kills (hits where g is the ONLY gear of {5..Q} striking the column).
# Self-contained; numpy only.  Run: uv run python research/anchor235/r34/q1_runs_window.py
import numpy as np, os, sys, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "q1_runs_window.txt")

def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return np.flatnonzero(s)

def next_prime(q):
    p = q + 2
    while not all(p % d for d in range(2, int(p ** 0.5) + 1)): p += 2
    return p

lines = []
def say(s=""):
    print(s); lines.append(s)

t0 = time.time()
say("Q1: the window (Q, Q'^2] in run units, per gear.  Columns k in (Q/6, W], W=(Q'^2-1)/6; a run = 5g columns.")
say("hits = g's strikes on anchor-open columns in the window; excl = hits where g is the only gear of {5..Q} striking.")
summary = []
for Q in (59, 173, 499, 997):
    Qn = next_prime(Q); W = (Qn * Qn - 1) // 6; k_lo = Q // 6 + 1   # window columns k_lo..W
    gears = [int(g) for g in primes_upto(Q) if g >= 5]
    ks = np.arange(k_lo, W + 1)
    nstrike = np.zeros(W + 1, np.int32)          # number of gears striking column k (index = k)
    for g in gears:
        u = pow(6, -1, g)
        for t in (u, g - u):
            nstrike[t::g] += 1
    anchor_open = np.zeros(W + 1, bool); anchor_open[0::5] = True; anchor_open[2::5] = True; anchor_open[3::5] = True
    n_open = int((nstrike[k_lo:] == 0).sum())
    # check: openings in the window are exactly the twin pairs in (Q, Q'^2]
    N = 6 * W + 2
    sv = np.ones(N + 1, bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    # openings in (Q/6, W] = twin pairs (6k-1, 6k+1) with Q < 6k-1 and 6k+1 < Q'^2, plus the square gate: column W
    # holds (Q'^2-2, Q'^2) and is an opening iff Q'^2-2 is prime (alignment-rules 4.1 rider).
    tw = int((sv[6 * ks - 1] & sv[6 * ks + 1] & (6 * ks - 1 > Q) & (6 * ks + 1 < Qn * Qn)).sum())
    gate = int(sv[Qn * Qn - 2])
    say(f"\nQ = {Q}, Q' = {Qn}, W = {W}, window columns {k_lo}..{W} ({W - k_lo + 1} columns), openings {n_open}, twin pairs in (Q, Q'^2) = {tw}, square gate [{Qn}^2-2 prime] = {gate}: {'OK' if tw + gate == n_open else 'MISMATCH'}")
    say(f"  {'g':>4} {'g mod 30':>8} {'runs':>8} {'strikes':>8} {'hits':>7} {'excl':>6} {'first excl col':>14} {'first excl numbers':>22}")
    zero_excl = []; rows = []
    for g in gears:
        u = pow(6, -1, g)
        cols = np.concatenate([np.arange(u, W + 1, g), np.arange(g - u, W + 1, g)])
        cols = cols[cols >= k_lo]
        hits = cols[anchor_open[cols]] if g != 5 else cols
        excl = hits[nstrike[hits] == 1]
        runs = (W - k_lo + 1) / (5 * g)
        fe = int(excl.min()) if len(excl) else None
        fe_num = f"{6*fe-1},{6*fe+1}" if fe is not None else "-"
        rows.append((g, g % 30, runs, len(cols), len(hits), len(excl), fe))
        if len(excl) == 0: zero_excl.append(g)
    # print: all gears at Q=59; at larger Q the first 12, every 10th, and the top 12 plus every zero-exclusive gear
    if Q <= 60:
        show = rows
    else:
        idx = set(range(12)) | set(range(0, len(rows), 10)) | set(range(len(rows) - 12, len(rows)))
        idx |= {i for i, r in enumerate(rows) if r[5] == 0}
        show = [rows[i] for i in sorted(idx)]
    for g, gm, runs, ns, nh, ne, fe in show:
        say(f"  {g:>4} {gm:>8} {runs:>8.2f} {ns:>8} {nh:>7} {ne:>6} {str(fe) if fe is not None else '-':>14} {(str(6*fe-1)+','+str(6*fe+1)) if fe is not None else '-':>22}")
    n_gear = len(gears)
    ratio = lambda g: g / Q
    say(f"  gears {n_gear}; gears with ZERO exclusive kills: {len(zero_excl)} -> {zero_excl}")
    say(f"  as fractions of Q: {[round(g / Q, 3) for g in zero_excl]}; smallest zero-exclusive gear / Q = {min(zero_excl) / Q if zero_excl else None}")
    say(f"  gears with an exclusive kill: {n_gear - len(zero_excl)} of {n_gear}; the set is {'the whole machine' if not zero_excl else 'a PROPER subset'}")
    # exclusive kills per gear: total, share of hits, by band of g/Q
    tot_excl = sum(r[5] for r in rows); tot_hits = sum(r[4] for r in rows)
    say(f"  total hits {tot_hits}, total exclusive {tot_excl}; blocked columns {W - k_lo + 1 - n_open} (exclusive kills are {tot_excl / (W - k_lo + 1 - n_open):.3f} of the blocked columns)")
    for a, b in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0001)):
        band = [r for r in rows if a <= r[0] / Q < b]
        if band:
            say(f"    g/Q in [{a},{b}): {len(band)} gears, excl per gear min {min(r[5] for r in band)} median {np.median([r[5] for r in band]):.0f} max {max(r[5] for r in band)}, excl/hits {sum(r[5] for r in band) / max(1, sum(r[4] for r in band)):.3f}")
    # where the exclusive kills of the top gears sit: cofactor structure
    say("  top four gears, their exclusive kills (column: numbers, factorisation of the g-multiple):")
    for g in gears[-4:]:
        u = pow(6, -1, g)
        cols = np.concatenate([np.arange(u, W + 1, g), np.arange(g - u, W + 1, g)]); cols = cols[cols >= k_lo]
        excl = np.sort(cols[(nstrike[cols] == 1) & anchor_open[cols]])
        desc = []
        for k in excl[:6]:
            k = int(k); a, b = 6 * k - 1, 6 * k + 1
            m = a // g if a % g == 0 else b // g
            desc.append(f"k={k}: ({a},{b}) = {g}x{m} beside prime")
        say(f"    g={g}: {len(excl)} exclusive; " + ("; ".join(desc) if desc else "none"))
    summary.append((Q, Qn, W, n_gear, len(zero_excl), zero_excl))

say("\nSummary: Q, Q', W, gears, zero-exclusive gears")
for s in summary: say(f"  {s}")
say(f"\nelapsed {time.time() - t0:.1f}s")
with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
print("written", OUT)

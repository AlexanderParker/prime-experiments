# Branch 7d, question 2: where the record stretches sit relative to zero and to the gears' clean end zones.
# Full periods {5..7} .. {5..23} (P = 37,182,145 columns at 23), residue sieve over columns.
# Max-gap convention: F(M) = max distance between consecutive openings (wrap included); the record
# STRETCH is the F-1 blocked columns x+1 .. x+F-1 between openings x and x+F.
# Clean end zone of gear g (anchor frame): columns within +-h_g of every multiple of 5g, where h_g is
# the column distance from 5gj to g's first anchor-open strike (numbers: +-g m_min around 30gj).
# Self-contained; numpy only.  Run: uv run python research/anchor235/r34/q2_record_position.py
import numpy as np, os, time
from math import prod

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "q2_record_position.txt")
lines = []
def say(s=""):
    print(s); lines.append(s)

def openings(gears, P):
    w = np.ones(P, bool)
    for g in gears:
        u = pow(6, -1, g); w[u::g] = False; w[g - u::g] = False
    return np.flatnonzero(w)

def gaps_of(op, P):
    return np.diff(np.concatenate([op, [op[0] + P]]))   # gap i = op[i+1]-op[i], last is the wrap

def zone_halfwidth(g):
    # smallest c > 0 with c = +-u_g mod g and c anchor-open (c mod 5 in {0,2,3})
    u = pow(6, -1, g); c = 1
    while True:
        if (c % g in (u, g - u)) and (c % 5 in (0, 2, 3)): return c
        c += 1

def rec_max(gears):
    P = prod(gears); op = openings(gears, P); return int(gaps_of(op, P).max())

ladder = [7, 11, 13, 17, 19, 23]
t0 = time.time()
say("Q2: record stretches, their position as a fraction of the period, and their relation to every gear's clean end zone.")
say("Convention: F = max gap between consecutive openings; stretch = blocked columns x+1..x+F-1; fraction = (x+1)/P.")
say("zone(g) = columns within +-h_g of a multiple of 5g (h_g = distance to g's first anchor-open strike); 'in' = stretch inside zone, 'x' = intersects, '.' = disjoint.")
for i, q in enumerate(ladder):
    gears = [5] + ladder[:i + 1]; P = prod(gears)
    op = openings(gears, P); gaps = gaps_of(op, P)
    F = int(gaps.max()); N = len(op)
    d0 = int(op[1])
    hw = {g: zone_halfwidth(g) for g in gears if g >= 7}
    Fminus = {g: rec_max([h for h in gears if h != g]) for g in gears if g >= 7}
    say(f"\nM = {{5..{q}}}, P = {P}, openings {N}, F = {F}, d_0 = {d0}, F_2 = {int((gaps[:-1] + gaps[1:]).max())}")
    say(f"  zone half-widths h_g: {hw};  F(M minus g): {Fminus}  (F(M)={F}; a stretch inside g's zone is a stretch of M minus g, so it needs F(M-g) >= F)")
    for level, L in (("F", F), ("F-1", F - 1), ("F-2", F - 2)):
        idx = np.flatnonzero(gaps == L)
        if len(idx) == 0:
            say(f"  gap {L} ({level}): none realised"); continue
        starts = [int(op[j]) for j in idx]           # x: opening before the stretch
        say(f"  gap {L} ({level}): {len(idx)} stretches")
        # mirror pairing: stretch x .. x+L  <->  P - x - L .. P - x
        S = set(starts)
        pairs = sum(1 for x in starts if (P - x - L) % P in S)
        selfm = [x for x in starts if (P - x - L) % P == x]
        say(f"    mirror-closed: {pairs} of {len(starts)} have their partner P-x-L in the set; self-mirror: {selfm}")
        # any translation symmetry by a multiple of 5g?  (differences of starts divisible by 5g)
        for g in gears:
            if g < 7: continue
            diffs = [abs(a - b) for a in starts for b in starts if a < b]
            hit = [d for d in diffs if d % (5 * g) == 0]
            if hit: say(f"    NOTE: {len(hit)} start differences divisible by 5*{g}: {hit[:5]}")
        show = starts if len(starts) <= 24 else starts[:12] + starts[-12:]
        for x in show:
            lo, hi = x + 1, x + L - 1           # the blocked columns
            frac = lo / P
            zs = []
            dists = []
            for g in gears:
                if g < 7: continue
                m = 5 * g; h = hw[g]
                # nearest multiple of 5g to the stretch (distance 0 if it contains one)
                j = round(((lo + hi) / 2) / m); c = j * m
                dist = 0 if lo <= c <= hi else min(abs(lo - c), abs(hi - c))
                dists.append(dist)
                # zone relation: columns c' with c' mod m in (-h, h)
                r_lo = lo % m
                # stretch inside zone iff whole interval within (c-h, c+h) for the nearest c
                inside = (c - h < lo) and (hi < c + h)
                # intersects iff some column of the stretch is within h of a multiple of m
                inter = inside or any(abs(t - c2) < h for c2 in (c - m, c, c + m) for t in (lo, hi)) or (lo <= c <= hi)
                if not inter:
                    # careful check: intersects iff min over columns of distance to nearest multiple < h
                    cols = np.arange(lo, hi + 1); dd = np.minimum(cols % m, m - cols % m); inter = bool((dd < h).any())
                zs.append("in" if inside else ("x" if inter else "."))
            say(f"    x={x:>9} cols {lo}..{hi} frac {frac:.4f} | dist to 5g-multiple {dists} | zone {zs}")
        # collective statistics of the level
        fr = np.array([(x + 1) / P for x in starts])
        say(f"    fractions: min {fr.min():.4f} max {fr.max():.4f}; within 0.3..0.7: {int(((fr >= .3) & (fr <= .7)).sum())} of {len(fr)}; below 0.05 or above 0.95: {int(((fr < .05) | (fr > .95)).sum())}")
        # zone containment tally per gear
        for g in gears:
            if g < 7: continue
            m = 5 * g; h = hw[g]
            n_in = 0; n_x = 0
            for x in starts:
                lo, hi = x + 1, x + L - 1
                cols = np.arange(lo, hi + 1); dd = np.minimum(cols % m, m - cols % m)
                if (dd < h).all(): n_in += 1
                elif (dd < h).any(): n_x += 1
            say(f"    gear {g}: zone half-width {h}, stretches inside {n_in}, intersecting {n_x}, disjoint {len(starts) - n_in - n_x} of {len(starts)}" + (f"  [F(M-{g}) = {Fminus[g]} < {L}: inside impossible by theorem]" if Fminus[g] < L else f"  [F(M-{g}) = {Fminus[g]} >= {L}: inside not excluded by the theorem]"))
    # typical stretch of length F: fraction of period positions within any gear's zone (base rate)
    say(f"  base rate: fraction of columns inside zone(g) = (2h_g-1)/(5g): " + ", ".join(f"{g}: {(2 * hw[g] - 1) / (5 * g):.3f}" for g in hw))
    say(f"  elapsed {time.time() - t0:.1f}s")
with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
print("written", OUT)

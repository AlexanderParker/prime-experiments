"""Manager scan, 2026-09-06. Distance from every record stretch (and runner-ups) to the nearest
all-teeth column (a column struck by EVERY gear: k = +-u_g mod g for each g; 2^m per period).
Full periods m11..m23. Distances measured to the nearest column of the stretch, cyclically.
"""
import numpy as np, itertools, sys, time
PR = [5, 7, 11, 13, 17, 19, 23]

def crt(residues, moduli):
    x, M = 0, 1
    for r, m in zip(residues, moduli):
        t = ((r - x) * pow(M, -1, m)) % m
        x += M * t; M *= m
    return x % M

def run(gears):
    t0 = time.time()
    P = 1
    for g in gears: P *= g
    us = [pow(6, -1, g) for g in gears]
    blocked = np.zeros(P, dtype=bool)
    for g, u in zip(gears, us):
        blocked[u::g] = True; blocked[(g - u) % g::g] = True
    opens = np.flatnonzero(~blocked)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]]))
    F = int(gaps.max())
    # all-teeth columns
    allteeth = np.array(sorted(crt([s * u % g for s, u, g in zip(signs, us, gears)], gears)
                               for signs in itertools.product([1, -1], repeat=len(gears))))
    def dist_to_allteeth(lo, hi):
        # min cyclic distance from any column in [lo, hi] to an all-teeth column
        best = P
        for a in allteeth:
            for a2 in (a, a + P, a - P):
                if lo <= a2 <= hi: return 0
                best = min(best, lo - a2 if a2 < lo else a2 - hi)
        return best
    print(f"machine {{5..{gears[-1]}}} P={P} F={F} all-teeth columns={len(allteeth)} (spacing ~{P//len(allteeth)}) [{time.time()-t0:.1f}s]")
    for target in (F, F - 1, F - 2, F - 3):
        idx = np.flatnonzero(gaps == target)
        if len(idx) == 0:
            print(f"  gap {target}: none realised"); continue
        ds = []
        for i in idx:
            lo = int(opens[i]) + 1; hi = lo + target - 2  # blocked columns strictly between openings
            ds.append(dist_to_allteeth(lo, hi))
        ds = sorted(ds)
        print(f"  gap {target}: {len(idx)} stretches; distance to nearest all-teeth column: min {ds[0]} median {ds[len(ds)//2]} max {ds[-1]}  (F={F}, random expectation ~{P//len(allteeth)//4})")
    # all-teeth columns: what surrounds them? longest blocked run containing each
    runs = []
    for a in allteeth:
        # walk left/right from a while blocked
        l = a
        while blocked[(l - 1) % P]: l -= 1
        r = a
        while blocked[(r + 1) % P]: r += 1
        runs.append(r - l + 1)
    runs = sorted(runs)
    print(f"  blocked run through all-teeth columns: min {runs[0]} median {runs[len(runs)//2]} max {runs[-1]}  (record run length {F-1})")
    sys.stdout.flush()

if __name__ == "__main__":
    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    for i in range(2, len(PR)):
        if PR[i] > upto: break
        run(PR[:i + 1])

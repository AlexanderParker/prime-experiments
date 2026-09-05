"""Manager, 2026-09-06. Two spaces between bricks, computed on full periods m11..m23.

(a) Which gap sizes a machine realises at all (the spectrum as a set), with multiplicities near the top.
(b) The neighbour-sum profile N(v) = max over gaps of size v of (left gap + right gap), the 3-run
    length N(v) + v, and the smallest v* above which every 3-run fits the budget F + q'.
    Compare v* with the new gear's letters d = 2u and q' - d.

Gaps are in the max-gap convention (distance between consecutive openings); F = max gap.
"""
import numpy as np, sys, time
PR = [5, 7, 11, 13, 17, 19, 23, 29, 31]

def sieve_period(gears):
    P = 1
    for g in gears: P *= g
    blocked = np.zeros(P, dtype=bool)
    for g in gears:
        u = pow(6, -1, g)
        blocked[u::g] = True
        blocked[(g - u) % g::g] = True
    return P, blocked

def run(gears, qn):
    t0 = time.time()
    P, blocked = sieve_period(gears)
    opens = np.flatnonzero(~blocked)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]]))  # cyclic
    F = int(gaps.max())
    sizes, counts = np.unique(gaps, return_counts=True)
    realised = set(int(s) for s in sizes)
    missing = [v for v in range(1, F + 1) if v not in realised]
    top = [(int(s), int(c)) for s, c in zip(sizes, counts) if s >= F - 6]
    # neighbour sums: for gap i, left = gaps[i-1], right = gaps[i+1] (cyclic)
    left = np.roll(gaps, 1); right = np.roll(gaps, -1)
    nsum = left + right
    N = {}
    for v in sizes:
        m = gaps == v
        N[int(v)] = int(nsum[m].max())
    u = pow(6, -1, qn); d = 2 * u % qn; letters = sorted({d, qn - d})
    budget = F + qn
    over = [v for v in N if N[v] + v > budget]
    vstar = (max(over) + 1) if over else 1
    print(f"machine {{5..{gears[-1]}}} P={P} F={F} q'={qn} d={d} letters={letters} budget={budget}  [{time.time()-t0:.1f}s]")
    print(f"  realised gap sizes: {len(realised)} of 1..{F}; missing below F: {missing}")
    print(f"  top of spectrum (size,count): {top}")
    print(f"  v* (smallest v with N(w)+w <= budget for all w >= v) = {vstar};  v*/d = {vstar/d:.2f}")
    for v in letters + [d - 1, d + 1, 2, 3, 4]:
        if v in N:
            print(f"    v={v:3d}: N(v)={N[v]:3d}  N(v)+v={N[v]+v:3d}  budget-{'ok' if N[v]+v<=budget else 'OVER'} slack={budget-N[v]-v}")
    # profile by v: print N(v)+v for all v
    prof = "  profile N(v)+v: " + " ".join(f"{v}:{N[v]+v}" for v in sorted(N))
    print(prof)
    sys.stdout.flush()

if __name__ == "__main__":
    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    for i in range(1, len(PR)):
        if PR[i] > upto: break
        gears = PR[:i + 1]
        if gears[-1] < 11: continue
        run(gears, PR[i + 1])

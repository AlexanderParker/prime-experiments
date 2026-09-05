"""Item 4: the counterfactual family.  200 random symmetric-tooth members at m13, m17, m19
(teeth at +-v_g, v_g uniform in 1..(g-1)/2 -- the alignment-rules section 5 family), each on
its full period.  For each member: the attaining 3-runs with v >= 6, split into the trivial
ones (v >= min(L,R), discharged by the peel bound) and the HARD ones (v < min(L,R)), and the
C2 covering rate on each part.  The real machine is one member of the family and is scored the
same way, so the question "does the lemma need the real teeth" is answered on the hard part,
which is the only part with content.
"""
import sys, random
import numpy as np
from math import prod
from gl_glue import gears_of, us_of, cov_pair, solve_cover

PR = [5, 7, 11, 13, 17, 19]


def sieve_v(gears, vs):
    P = prod(gears)
    b = np.zeros(P, dtype=bool)
    for g, v in zip(gears, vs):
        b[v % g::g] = True
        b[(-v) % g::g] = True
    return P, b


def stats(P, b):
    opens = np.flatnonzero(~b)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]]))
    F = int(gaps.max())
    F2 = int((gaps + np.roll(gaps, -1)).max())
    left, right = np.roll(gaps, 1), np.roll(gaps, -1)
    ns = (left + right).astype(np.int64)
    NM = 2 * F + 2
    cnt = np.bincount(gaps.astype(np.int64) * NM + ns, minlength=(F + 1) * NM)
    N = {}
    for k in np.flatnonzero(cnt).tolist():
        v, s = k // NM, k % NM
        if s > N.get(v, -1):
            N[v] = s
    return opens, gaps, left, right, F, F2, N


def score(gears, vs):
    P, b = sieve_v(gears, vs)
    opens, gaps, left, right, F, F2, N = stats(P, b)
    ns = left + right
    tr = trok = hd = hdok = 0
    for v in sorted(N):
        if v < 6:
            continue
        idx = np.flatnonzero((gaps == v) & (ns == N[v]))
        idx = idx[(idx >= 1) & (idx + 1 < gaps.size)]
        for i in idx.tolist():
            x0 = int(opens[i - 1]); L = int(gaps[i - 1]); R = int(gaps[i + 1])
            T, h, cL, cR = cov_pair(gears, vs, x0, L, R, x0 + L + v)
            ok = solve_cover(cL, cR, T, h) is not None
            if v >= min(L, R):
                tr += 1; trok += ok
            else:
                hd += 1; hdok += ok
    exc = [v for v in N if v >= 6 and N[v] > F2]
    return F, F2, tr, trok, hd, hdok, exc


def main(out, nmem=200, seed=20260905):
    rng = random.Random(seed)
    for top in (13, 17, 19):
        gears = [p for p in PR if p <= top]
        realv = tuple(min(u, g - u) for g, u in zip(gears, us_of(gears)))
        fsize = prod((g - 1) // 2 for g in gears)
        target = min(nmem, fsize)
        rows = []
        seen = set()
        while len(rows) < target:
            vs = tuple(rng.randrange(1, (g + 1) // 2) for g in gears)
            if vs in seen:
                continue
            seen.add(vs)
            rows.append((vs,) + score(gears, vs))
        rr = score(gears, realv)
        out.write(f"\n=== family m{top}: {target} random symmetric members "
                  f"(family size {fsize})\n")
        out.write(f"  REAL member teeth={realv}: F={rr[0]} F_2={rr[1]}; attaining v>=6: "
                  f"trivial {rr[3]}/{rr[2]}, HARD {rr[5]}/{rr[4]}"
                  f" = {100*rr[5]/max(rr[4],1):.1f}%; law exceptions {rr[6]}\n")
        hardtot = sum(r[5] for r in rows)
        hardok = sum(r[6] for r in rows)
        trivtot = sum(r[3] for r in rows)
        trivok = sum(r[4] for r in rows)
        exc = sum(1 for r in rows if r[7])
        out.write(f"  family pooled: trivial {trivok}/{trivtot} "
                  f"({100*trivok/max(trivtot,1):.1f}%), HARD {hardok}/{hardtot} "
                  f"({100*hardok/max(hardtot,1):.1f}%)\n")
        permem = [100 * r[6] / r[5] for r in rows if r[5] > 0]
        permem.sort()
        out.write(f"  members with at least one HARD attaining run: {len(permem)}/{target}; "
                  f"per-member HARD rate min/med/max = "
                  f"{permem[0]:.0f}/{permem[len(permem)//2]:.0f}/{permem[-1]:.0f} %\n"
                  if permem else "  no member has a hard attaining run\n")
        out.write(f"  members with a law exception (some v>=6 with N(v) > F_2): {exc}/{target}\n")
        below = sum(1 for r in rows if r[5] > 0 and
                    100 * r[6] / r[5] < (100 * rr[5] / max(rr[4], 1)))
        out.write(f"  members whose HARD rate is BELOW the real machine's: {below}/"
                  f"{len(permem) if permem else 0}\n")
        out.flush()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, "w") if dest else sys.stdout
    main(o)
    if dest:
        o.close()

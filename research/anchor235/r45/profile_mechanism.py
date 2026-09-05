"""Branch 2g.i, parts 2-4: the counterfactual family, the exact mechanism of an attaining
3-run, and the concatenation (gluing) test.  Prover, 2026-09-05.

Part 2  family(): 200 random symmetric-tooth members at m13, m17, m19 (teeth at +-v_g,
        v_g in 1..(g-1)/2, the alignment-rules 5 family), each on its full period; report
        F, F_2, the threshold v0 for the cap F+1 and the threshold v0F2 for the cap F_2,
        and whether the cap holds from v = 6 up.

Part 3  mechanism(): for every 3-run of M that ATTAINS N(v) at a given v, list gear by gear
        which columns of the run it strikes and on which tooth, name the sole strikers,
        give the chain-law class of the middle gap's two ends (x2 - x1 mod g against
        {0, +d, -d}) and the residues x1, x2 mod every gear that strikes inside a flank.

Part 4  glue(): the concatenation test.  Two versions, both exact CRT over subsets of the
        gear set.  Write x0 < x1 < x2 < x3 for the run's openings, L = x1-x0, v = x2-x1,
        R = x3-x2.
          A (tight glue, shift v+1): find z with the left flank's L-1 blocked columns at
            z..z+L-2 and the right flank's R-1 blocked columns at z+L-1..z+L+R-3.  A gear
            asked to serve both sides needs g | v+1.  Success => F >= L+R-1 => N(v) <= F+1.
          B (holed glue, shift v): left flank at z..z+L-2, a hole at z+L-1, right flank at
            z+L..z+L+R-2.  A gear serving both sides needs g | v.  Success => the glued
            object is either one blocked run of length L+R-1 (=> N(v) <= F) or an adjacent
            gap pair of sum >= L+R (=> N(v) <= F_2).
        Each gear is assigned to the left phase or the right phase (2^k assignments); CRT
        gives z mod P; the run is then checked column by column against the real machine.
"""
import numpy as np, sys, random, itertools
from math import prod

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def teeth(gears, vs):
    return {g: (v % g, (-v) % g) for g, v in zip(gears, vs)}


def real_teeth(gears):
    return [pow(6, -1, g) for g in gears]


def sieve(gears, vs):
    P = prod(gears)
    blocked = np.zeros(P, dtype=bool)
    for g, v in zip(gears, vs):
        blocked[v % g::g] = True
        blocked[(-v) % g::g] = True
    return P, blocked


def gap_stats(P, blocked):
    opens = np.flatnonzero(~blocked)
    gaps = np.diff(np.concatenate([opens, [opens[0] + P]]))
    F = int(gaps.max())
    F2 = int((gaps + np.roll(gaps, -1)).max())
    left, right = np.roll(gaps, 1), np.roll(gaps, -1)
    ns = left + right
    NM = 2 * F + 2
    key = gaps.astype(np.int64) * NM + ns
    cnt = np.bincount(key, minlength=(F + 1) * NM)
    N = {}
    for k in np.flatnonzero(cnt):
        v, s = int(k) // NM, int(k) % NM
        if s > N.get(v, -1):
            N[v] = s
    return opens, gaps, F, F2, N


def thresholds(N, cap):
    over = [v for v in N if N[v] > cap]
    return (max(over) + 1) if over else 1, sorted(over)


# ---------------------------------------------------------------- part 2
def family(out, nmem=200, seed=20260905):
    rng = random.Random(seed)
    for top, qn in ((13, 17), (17, 19), (19, 23)):
        gears = [p for p in PRIMES if p <= top]
        uq = pow(6, -1, qn)
        a = min(2 * uq % qn, qn - 2 * uq % qn)
        rows = []
        seen = set()
        realv = tuple(min(u, g - u) for g, u in zip(gears, real_teeth(gears)))
        fsize = prod((g - 1) // 2 for g in gears)
        target = min(nmem, fsize)
        while len(rows) < target:
            vs = tuple(rng.randrange(1, (g + 1) // 2) for g in gears)
            if vs in seen:
                continue
            seen.add(vs)
            P, blocked = sieve(gears, vs)
            _, _, F, F2, N = gap_stats(P, blocked)
            v0, ov = thresholds(N, F + 1)
            v0f, ovf = thresholds(N, F2)
            rows.append((vs, F, F2, v0, v0f, max(ovf) if ovf else 0, vs == realv))
        okF1 = sum(1 for r in rows if r[3] <= a)
        okF2 = sum(1 for r in rows if r[4] <= 6)
        okF2a = sum(1 for r in rows if r[4] <= a)
        P, blocked = sieve(gears, realv)
        _, _, RF, RF2, RN = gap_stats(P, blocked)
        rv0, _ = thresholds(RN, RF + 1)
        rv0f, _ = thresholds(RN, RF2)
        out.write(f"\n--- family m{top} (q'={qn}, a={a}); {target} random symmetric members\n")
        out.write(f"    real member: F={RF} F_2={RF2} v0(cap F+1)={rv0} v0(cap F_2)={rv0f}\n")
        out.write(f"    members with v0(cap F+1) <= a={a}: {okF1}/{target}\n")
        out.write(f"    members with v0(cap F_2) <= 6:     {okF2}/{target}\n")
        out.write(f"    members with v0(cap F_2) <= a={a}: {okF2a}/{target}\n")
        v0s = sorted(r[3] for r in rows)
        v0fs = sorted(r[4] for r in rows)
        out.write(f"    v0(cap F+1) distribution: min={v0s[0]} med={v0s[target // 2]} "
                  f"max={v0s[-1]}\n")
        out.write(f"    v0(cap F_2) distribution: min={v0fs[0]} med={v0fs[target // 2]} "
                  f"max={v0fs[-1]}\n")
        worst = sorted(rows, key=lambda r: -r[3])[:5]
        for w in worst:
            out.write(f"      worst v0(F+1)={w[3]:3d} (F={w[1]}, F_2={w[2]}) teeth={w[0]}\n")
        worstf = sorted(rows, key=lambda r: -r[4])[:5]
        for w in worstf:
            out.write(f"      worst v0(F_2)={w[4]:3d} (F={w[1]}, F_2={w[2]}) teeth={w[0]}\n")
        out.flush()


# ---------------------------------------------------------------- part 3+4
def attaining_runs(opens, gaps, N, v, limit=None):
    """indices i with gaps[i] == v and gaps[i-1]+gaps[i+1] == N[v]"""
    n = gaps.size
    left, right = np.roll(gaps, 1), np.roll(gaps, -1)
    hit = np.flatnonzero((gaps == v) & (left + right == N[v]))
    hit = hit[(hit >= 1) & (hit + 1 < n)]
    return hit if limit is None else hit[:limit]


def strike_table(gears, us, x0, span):
    """for each column of [x0, x0+span] the gears striking it, with tooth sign."""
    tab = []
    for c in range(x0, x0 + span + 1):
        hits = []
        for g, u in zip(gears, us):
            if c % g == u % g:
                hits.append((g, '+'))
            elif c % g == (-u) % g:
                hits.append((g, '-'))
        tab.append((c, hits))
    return tab


def crt(mods, rems):
    z, Mm = 0, 1
    for m, r in zip(mods, rems):
        # solve z = z (mod Mm), z = r (mod m)
        t = ((r - z) * pow(Mm, -1, m)) % m
        z += Mm * t
        Mm *= m
    return z % Mm, Mm


def glue_test(gears, us, blocked, P, x0, L, v, R, version):
    """version 'A': tight (shift v+1); 'B': holed (shift v).  Returns (ok, best_desc).

    Left-assigned gears keep their left-flank phase (z = x0+1 mod g), right-assigned gears
    take the right flank's phase translated left by `shift` (z = x0+1+shift mod g)."""
    k = len(gears)
    if version == 'A':
        T = L + R - 2                       # blocked columns wanted
        shift = v + 1
        holes = set()
    else:
        T = L + R - 1
        shift = v
        holes = {L - 1}
    for mask in range(1 << k):
        rems = []
        for i, g in enumerate(gears):
            rems.append((x0 + 1) % g if (mask >> i) & 1 else (x0 + 1 + shift) % g)
        z, _ = crt(gears, rems)
        # target columns are z + j for j in 0..T-1 ; note the left flank starts at
        # (x0+1) so the offset baseline is absorbed: rems are relative to x0+1.
        good = True
        for j in range(T):
            if j in holes:
                continue
            if not blocked[(z + j) % P]:
                good = False
                break
        if good:
            hole_blocked = all(blocked[(z + j) % P] for j in holes)
            return True, (mask, z, hole_blocked)
    return False, None


def mechanism(out, top, qn, vlist, maxruns=3, glue_sample=40):
    gears = [p for p in PRIMES if p <= top]
    us = real_teeth(gears)
    vs = [min(u, g - u) for g, u in zip(gears, us)]
    P, blocked = sieve(gears, vs)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    v0, over = thresholds(N, F + 1)
    v0f, overf = thresholds(N, F2)
    uq = pow(6, -1, qn)
    a = min(2 * uq % qn, qn - 2 * uq % qn)
    b = qn - a
    out.write(f"\n===== m{top}: P={P} F={F} F_2={F2} q'={qn} a={a} b={b}\n")
    out.write(f"      v0(cap F+1)={v0} over={[(x, N[x]) for x in over]}\n")
    out.write(f"      v0(cap F_2)={v0f} over={[(x, N[x]) for x in overf]}\n")
    for v in vlist:
        if v not in N:
            continue
        hits = attaining_runs(opens, gaps, N, v)
        out.write(f"\n  -- v={v}: N(v)={N[v]} (F{N[v] - F:+d}, F_2{N[v] - F2:+d}), "
                  f"span={N[v] + v}, attaining 3-runs: {hits.size}\n")
        for i in hits[:maxruns]:
            x0 = int(opens[i - 1]); Lg = int(gaps[i - 1]); Rg = int(gaps[i + 1])
            x1 = x0 + Lg; x2 = x1 + v; x3 = x2 + Rg
            out.write(f"     run x0={x0} (L,v,R)=({Lg},{v},{Rg}) span={Lg + v + Rg}\n")
            tab = strike_table(gears, us, x0, Lg + v + Rg)
            sole = {}
            for c, hits2 in tab:
                if len(hits2) == 1:
                    sole.setdefault(hits2[0][0], []).append(c - x0)
            # closers of each junction
            for name, col in (("x0+1", x0 + 1), ("x1-1", x1 - 1), ("x1+1", x1 + 1),
                              ("x2-1", x2 - 1), ("x2+1", x2 + 1), ("x3-1", x3 - 1)):
                hs = [f"{g}{s}" for g, s in strike_table(gears, us, col, 0)[0][1]]
                out.write(f"       closer {name:5s} (offset {col - x0:3d}): {hs}\n")
            out.write(f"       sole strikers (gear: offsets): "
                      f"{ {g: v2 for g, v2 in sorted(sole.items())} }\n")
            cls = []
            for g, u in zip(gears, us):
                d = (2 * u) % g
                r = v % g
                c = 'pad' if r == 0 else ('+d' if r == d else ('-d' if r == (-d) % g
                                                               else 'illegal'))
                cls.append(f"{g}:{c}(x1={x1 % g},x2={x2 % g},u={u})")
            out.write(f"       middle class of v mod each gear: {' '.join(cls)}\n")
            out.write(f"       shared-gear divisibility: gears dividing v={v}: "
                      f"{[g for g in gears if v % g == 0]}; dividing v+1={v + 1}: "
                      f"{[g for g in gears if (v + 1) % g == 0]}\n")
        # glue test over a sample of attaining runs
        nA = nB = nBhole = 0
        samp = hits[:glue_sample]
        for i in samp:
            Lg = int(gaps[i - 1]); Rg = int(gaps[i + 1]); xx0 = int(opens[i - 1])
            okA, _ = glue_test(gears, us, blocked, P, xx0, Lg, v, Rg, 'A')
            okB, infoB = glue_test(gears, us, blocked, P, xx0, Lg, v, Rg, 'B')
            nA += okA; nB += okB
            if okB and infoB[2]:
                nBhole += 1
        out.write(f"     GLUE over {samp.size} attaining runs: A (tight, needs g|v+1 for "
                  f"shared gears) {nA}/{samp.size};  B (holed, needs g|v) {nB}/{samp.size}"
                  f" (of which middle also blocked: {nBhole})\n")
        out.flush()


def sweep(out, top, qn, vmin=2, cap=200):
    """glue test over EVERY realised gap size, all (capped) attaining 3-runs."""
    gears = [p for p in PRIMES if p <= top]
    us = real_teeth(gears)
    vs = [min(u, g - u) for g, u in zip(gears, us)]
    P, blocked = sieve(gears, vs)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    uq = pow(6, -1, qn)
    a = min(2 * uq % qn, qn - 2 * uq % qn)
    b = qn - a
    out.write(f"\n##### glue sweep m{top}: F={F} F_2={F2} q'={qn} a={a} b={b}\n")
    out.write("   v  N(v)  N-F  N-F2  #runs  A(tight)  B(holed)  5|v  gearsdiv(v)\n")
    totA = totB = tot = 0
    for v in sorted(N):
        if v < vmin:
            continue
        hits = attaining_runs(opens, gaps, N, v)[:cap]
        nA = nB = 0
        for i in hits:
            Lg = int(gaps[i - 1]); Rg = int(gaps[i + 1]); xx0 = int(opens[i - 1])
            nA += glue_test(gears, us, blocked, P, xx0, Lg, v, Rg, 'A')[0]
            nB += glue_test(gears, us, blocked, P, xx0, Lg, v, Rg, 'B')[0]
        if v >= 6:
            totA += nA; totB += nB; tot += hits.size
        mark = "  <-- letter" if v in (a, b) else ""
        out.write(f"  {v:3d} {N[v]:5d} {N[v] - F:+5d} {N[v] - F2:+5d} {hits.size:6d} "
                  f"{nA:8d} {nB:9d}  {'y' if v % 5 == 0 else 'n'}  "
                  f"{[g for g in gears if v % g == 0]}{mark}\n")
    out.write(f"  TOTAL over v >= 6: A {totA}/{tot}, B {totB}/{tot}\n")
    out.flush()


def sole_gears(gears, us, lo, hi):
    """gears that are the ONLY striker of some column in [lo, hi] (inclusive)."""
    s = set()
    for c in range(lo, hi + 1):
        hits = [g for g, u in zip(gears, us) if c % g in (u % g, (-u) % g)]
        if len(hits) == 1:
            s.add(hits[0])
    return s


def diag(out, top, qn, cap=400):
    """for every attaining 3-run with v >= 6 that the holed glue FAILS on, say why."""
    gears = [p for p in PRIMES if p <= top]
    us = real_teeth(gears)
    vs = [min(u, g - u) for g, u in zip(gears, us)]
    P, blocked = sieve(gears, vs)
    opens, gaps, F, F2, N = gap_stats(P, blocked)
    out.write(f"\n***** glue-B failure diagnosis m{top} (F={F}, F_2={F2})\n")
    nfail = ntot = 0
    shared_bad = 0
    nok_bad = 0
    for v in sorted(N):
        if v < 6:
            continue
        for i in attaining_runs(opens, gaps, N, v)[:cap]:
            Lg = int(gaps[i - 1]); Rg = int(gaps[i + 1]); x0 = int(opens[i - 1])
            ntot += 1
            ok, _ = glue_test(gears, us, blocked, P, x0, Lg, v, Rg, 'B')
            x1 = x0 + Lg; x2 = x1 + v
            SL = sole_gears(gears, us, x0 + 1, x1 - 1)
            SR = sole_gears(gears, us, x2 + 1, x2 + Rg - 1)
            both = sorted(SL & SR)
            bad = [g for g in both if v % g]
            if ok:
                nok_bad += bool(bad)
                continue
            nfail += 1
            shared_bad += bool(bad)
            out.write(f"   v={v:2d} (L,v,R)=({Lg},{v},{Rg}) x0={x0}  sole-left={sorted(SL)} "
                      f"sole-right={sorted(SR)}  shared={both}  shared-not-dividing-v={bad}\n")
    out.write(f"   failures {nfail} of {ntot} attaining runs with v >= 6; "
              f"{shared_bad} of {nfail} have a shared sole striker that does not divide v; "
              f"among the {ntot - nfail} SUCCESSES {nok_bad} have one too\n")
    out.flush()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    out = open(dest, "w") if dest else sys.stdout
    if what in ("all", "family"):
        family(out)
    if what in ("all", "mech"):
        for top, qn, vl in ((13, 17, [2, 5, 6, 7, 11]),
                            (17, 19, [2, 6, 7, 8, 13]),
                            (19, 23, [2, 7, 8, 10, 15])):
            mechanism(out, top, qn, vl)
    if what in ("all", "sweep"):
        for top, qn in ((13, 17), (17, 19), (19, 23)):
            sweep(out, top, qn)
    if what in ("all", "diag"):
        for top, qn in ((17, 19), (19, 23)):
            diag(out, top, qn)
    if what == "sweep23":
        sweep(out, 23, 29, vmin=2, cap=30)
    if dest:
        out.close()

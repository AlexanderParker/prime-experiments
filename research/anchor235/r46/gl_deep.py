"""Item 2 at the deep rungs: m29 (1.08e9 columns) and m31 (3.34e10 columns).

Chunked sieve in 4 processes, exactly the partition of r45/deep_profile.py (gaps attributed by
their LEFT endpoint, a 4096-column margin each side so every 3-run is complete).  Collected:
  - N(v) for every realised v (to identify the attaining 3-runs);
  - every 3-run with v >= 6 and L + R >= THRESH, capped, with its x0 (the glue test needs only
    x0 mod g, so no period is ever held);
  - the top runs per v so the attaining ones are recovered exactly.
Then the covering test C2 is run on the collected runs, split trivial (v >= min(L,R), the peel
bound) / hard (v < min(L,R)).
"""
import numpy as np, sys, time
from multiprocessing import Pool
from gl_glue import gears_of, us_of, glue
from gl_shadow import min_moves

MARGIN = 4096
CHUNK = 3 * 10 ** 7
VMAX = 128
NMAX = 256
KEEP = 400            # per v, top runs by L+R


def worker(args):
    gears, lo, hi, thresh = args
    u = [pow(6, -1, g) for g in gears]
    joint = np.zeros(VMAX * NMAX, dtype=np.int64)
    big = []                       # runs with L+R >= thresh, v >= 6
    f2 = 0
    perv = {}                      # v -> list of (L+R, x0, L, R)
    c0 = lo
    while c0 < hi:
        c1 = min(c0 + CHUNK, hi)
        s, e = c0 - MARGIN, c1 + MARGIN
        n = e - s
        blocked = np.zeros(n, dtype=bool)
        for g, ug in zip(gears, u):
            for t in (ug, g - ug):
                blocked[(t - s) % g::g] = True
        idx = np.flatnonzero(~blocked)
        del blocked
        opens = idx.astype(np.int64) + s
        del idx
        gaps = np.diff(opens).astype(np.int32)
        own = np.flatnonzero((opens[:-1] >= c0) & (opens[:-1] < c1))
        oi = own[(own >= 1) & (own + 1 < gaps.size)]
        if oi.size:
            v = gaps[oi].astype(np.int64)
            ns = (gaps[oi - 1] + gaps[oi + 1]).astype(np.int64)
            joint += np.bincount(v * NMAX + ns, minlength=VMAX * NMAX)[:VMAX * NMAX]
            f2 = max(f2, int((gaps[oi] + gaps[oi + 1]).max()))
            sel = (v >= 6) & (ns >= thresh)
            for j in np.flatnonzero(sel).tolist():
                i = int(oi[j])
                rec = (int(ns[j]), int(opens[i - 1]), int(gaps[i - 1]), int(gaps[i + 1]))
                big.append((int(v[j]),) + rec)
            # top KEEP per v (for attainment)
            for vv in np.unique(v[v >= 6]).tolist():
                m = np.flatnonzero(v == vv)
                order = m[np.argsort(-ns[m])][:KEEP]
                lst = perv.setdefault(vv, [])
                for j in order.tolist():
                    i = int(oi[j])
                    lst.append((int(ns[j]), int(opens[i - 1]), int(gaps[i - 1]),
                                int(gaps[i + 1])))
                lst.sort(key=lambda r: -r[0])
                del lst[KEEP:]
        c0 = c1
    return joint, big, perv, f2


def run(top, thresh_off, nproc=4):
    gears = gears_of(top)
    P = 1
    for g in gears:
        P *= g
    # first find F cheaply from a partial pass?  no: use the recorded ladder threshold offset
    bounds = [P * i // nproc for i in range(nproc + 1)]
    # threshold on L+R: pass an absolute number computed from the known F ladder
    FLAD = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
    thresh = FLAD[top] - thresh_off
    jobs = [(gears, bounds[i], bounds[i + 1], thresh) for i in range(nproc)]
    t0 = time.time()
    with Pool(nproc) as pool:
        parts = pool.map(worker, jobs)
    joint = sum(p[0] for p in parts)
    f2 = max(p[3] for p in parts)
    big = [r for p in parts for r in p[1]]
    perv = {}
    for p in parts:
        for vv, lst in p[2].items():
            perv.setdefault(vv, []).extend(lst)
    for vv in perv:
        perv[vv].sort(key=lambda r: -r[0])
        del perv[vv][KEEP:]
    return gears, P, joint, big, perv, thresh, f2, time.time() - t0


def main(out, top, thresh_off=8):
    gears, P, joint, big, perv, thresh, F2, dt = run(top, thresh_off)
    us = us_of(gears)
    J2 = joint.reshape(VMAX, NMAX)
    N = {}
    for v in range(1, VMAX):
        nz = np.flatnonzero(J2[v])
        if nz.size:
            N[v] = int(nz.max())
    F = max(v for v in N)
    out.write(f"\n===== m{top}  P={P}  F={F}  F_2={F2}  [{dt:.1f}s]  "
              f"collected {len(big)} 3-runs with v>=6 and L+R>={thresh}\n")
    over = [v for v in N if v >= 6 and N[v] > F2]
    out.write(f"  law N(v) <= F_2 for v>=6: exceptions {over}\n")
    out.write(f"  max_(v>=6) N(v) = {max(N[v] for v in N if v>=6)} at v="
              f"{max((v for v in N if v>=6), key=lambda v: N[v])}\n")
    # attaining runs
    att = []
    for vv, lst in sorted(perv.items()):
        best = N[vv]
        for (s, x0, L, R) in lst:
            if s == best:
                att.append((x0, L, vv, R))
    tr = trok = hd = hdok = 0
    mv = {}
    hard_fail = []
    for (x0, L, v, R) in att:
        good, bl, br = min_moves(gears, us, x0, L, v, R)
        if v >= min(L, R):
            tr += 1; trok += good
        else:
            hd += 1; hdok += good
            if not good:
                hard_fail.append((L, v, R, x0))
        if good:
            m = min(bl, br); mv[m] = mv.get(m, 0) + 1
    out.write(f"  ATTAINING 3-runs v>=6: {len(att)} (capped at {KEEP} per v); "
              f"C2 ok {trok+hdok} ({100*(trok+hdok)/max(len(att),1):.1f}%); "
              f"trivial (v>=min(L,R)) {trok}/{tr}; HARD (v<min(L,R)) {hdok}/{hd} "
              f"= {100*hdok/max(hd,1):.1f}%; moves {dict(sorted(mv.items()))}\n")
    for (L, v, R, x0) in hard_fail[:20]:
        out.write(f"     HARD FAIL (L,v,R)=({L},{v},{R}) sum={L+R} "
                  f"(F{L+R-F:+d}, F_2{L+R-F2:+d}) x0={x0}\n")
    # all collected runs
    tr = trok = hd = hdok = 0
    mv = {}
    seen = set()
    for (v, s, x0, L, R) in big:
        if (x0, L, v, R) in seen:
            continue
        seen.add((x0, L, v, R))
        good, bl, br = min_moves(gears, us, x0, L, v, R)
        if v >= min(L, R):
            tr += 1; trok += good
        else:
            hd += 1; hdok += good
        if good:
            m = min(bl, br); mv[m] = mv.get(m, 0) + 1
    out.write(f"  ALL 3-runs v>=6 with L+R>={thresh}: {tr+hd}; C2 ok {trok+hdok} "
              f"({100*(trok+hdok)/max(tr+hd,1):.1f}%); trivial {trok}/{tr}; "
              f"HARD {hdok}/{hd} = {100*hdok/max(hd,1):.1f}%; moves {dict(sorted(mv.items()))}\n")
    out.flush()


if __name__ == "__main__":
    top = int(sys.argv[1])
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    o = open(dest, "a") if dest else sys.stdout
    main(o, top, off)
    if dest:
        o.close()

"""Branch 2g.i (prover, 2026-09-05).  The neighbour-sum profile N(v) on full periods, to m31.

Extends the manager's spectrum_profile.py (m11..m23, single pass in RAM) to m29 (1.08e9 columns)
and m31 (3.34e10 columns) by chunked sieving in 4 processes.  Conventions exactly as there:
a gap is the distance between consecutive openings, cyclic over the period; F = max gap.

Computed per machine M (with q' = the next prime, letters a = 2u', b = q' - a):
  - the gap spectrum (count per size) and F;
  - the joint histogram of (v, left+right) over all gaps, hence N(v) for every realised v;
  - Q*_J(M; q') for J = 1..7 (largest span of a word-legal J-run), with a witness for each;
  - a witness 3-run at v = a and v = b.

Gaps are partitioned by their LEFT endpoint, so every gap of the period is counted exactly once.
Each worker sieves its range plus a margin on both sides so neighbours and J-runs are complete.
"""
import numpy as np, sys, time, os
from multiprocessing import Pool

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
MARGIN = 4096          # > max span of any 7-run we look at
NMAX = 256             # cap on left+right (2F); F <= 91 through m41
VMAX = 128             # cap on a gap size
JMAX = 7
CHUNK = 3 * 10 ** 7


def classify(gaps, qn, a, b):
    """0 = pad (=0 mod q'), 1 = class a, 2 = class b, 3 = illegal."""
    r = gaps % qn
    cls = np.full(gaps.shape, 3, dtype=np.int8)
    cls[r == 0] = 0
    cls[r == a % qn] = 1
    cls[r == b % qn] = 2
    return cls


def worker(args):
    gears, qn, lo, hi = args
    u = [pow(6, -1, g) for g in gears]
    uq = pow(6, -1, qn)
    a = min(2 * uq % qn, qn - 2 * uq % qn)
    b = qn - a
    spec = np.zeros(VMAX, dtype=np.int64)
    joint = np.zeros(VMAX * NMAX, dtype=np.int64)
    qstar = [(-1, None)] * (JMAX + 1)          # (span, witness) per J
    lw = {a: (-1, None), b: (-1, None)}        # letter witnesses for N(v)
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
        gaps = np.diff(opens).astype(np.int32)          # gap i = opens[i+1]-opens[i]
        # gaps owned by this chunk: left endpoint in [c0, c1)
        own = np.flatnonzero((opens[:-1] >= c0) & (opens[:-1] < c1))
        if own.size:
            spec += np.bincount(gaps[own], minlength=VMAX)[:VMAX]
            # neighbour sums: gap i has left gap i-1 and right gap i+1
            oi = own[(own >= 1) & (own + 1 < gaps.size)]
            v = gaps[oi].astype(np.int64)
            ns = (gaps[oi - 1] + gaps[oi + 1]).astype(np.int64)
            joint += np.bincount(v * NMAX + ns, minlength=VMAX * NMAX)[:VMAX * NMAX]
            for lv in (a, b):
                m = gaps[oi] == lv
                if m.any():
                    k = int(np.argmax(np.where(m, ns, -1)))
                    if ns[k] > lw[lv][0]:
                        j = oi[k]
                        lw[lv] = (int(ns[k]), (int(opens[j - 1]), int(gaps[j - 1]),
                                               int(gaps[j]), int(gaps[j + 1])))
            # Q*_J
            cls = classify(gaps, qn, a, b)
            base = own[own + JMAX < gaps.size]
            span = gaps[base].astype(np.int64)
            ok = np.ones(base.size, dtype=bool)
            prev = np.zeros(base.size, dtype=np.int8)
            for J in range(2, JMAX + 1):
                span = span + gaps[base + J - 1]
                if J >= 3:                                   # new middle is gaps[base+J-2]
                    c = cls[base + J - 2]
                    ok &= (c != 3)
                    ok &= ~((c != 0) & (c == prev))
                    prev = np.where(c != 0, c, prev).astype(np.int8)
                cand = np.where(ok, span, -1)
                k = int(np.argmax(cand))
                if cand[k] > qstar[J][0]:
                    j = int(base[k])
                    qstar[J] = (int(cand[k]), (int(opens[j]),
                                               [int(x) for x in gaps[j:j + J]]))
        c0 = c1
    return spec, joint, qstar, lw


def run(gears, qn, nproc=4):
    t0 = time.time()
    P = 1
    for g in gears:
        P *= g
    bounds = [P * i // nproc for i in range(nproc + 1)]
    jobs = [(gears, qn, bounds[i], bounds[i + 1]) for i in range(nproc)]
    if nproc == 1:
        parts = [worker(jobs[0])]
    else:
        with Pool(nproc) as pool:
            parts = pool.map(worker, jobs)
    spec = sum(p[0] for p in parts)
    joint = sum(p[1] for p in parts)
    qstar = [(-1, None)] * (JMAX + 1)
    lw = {}
    for p in parts:
        for J in range(2, JMAX + 1):
            if p[2][J][0] > qstar[J][0]:
                qstar[J] = p[2][J]
        for k, val in p[3].items():
            if val[0] > lw.get(k, (-1, None))[0]:
                lw[k] = val
    return P, spec, joint, qstar, lw, time.time() - t0


def report(gears, qn, out):
    P, spec, joint, qstar, lw, dt = run(gears, qn)
    F = int(np.max(np.flatnonzero(spec)))
    total = int(spec.sum())
    uq = pow(6, -1, qn)
    a = min(2 * uq % qn, qn - 2 * uq % qn)
    b = qn - a
    J2 = joint.reshape(VMAX, NMAX)
    N = {}
    for v in range(1, VMAX):
        nz = np.flatnonzero(J2[v])
        if nz.size:
            N[v] = int(nz.max())
    over = [v for v in N if N[v] > F + 1]
    v0 = (max(over) + 1) if over else 1
    qstar[1] = (F, None)
    best = max(range(1, JMAX + 1), key=lambda J: qstar[J][0])
    w = out.write
    w(f"\n=== machine {{5..{gears[-1]}}}  P={P}  openings={total}  F={F}  "
      f"q'={qn} letters a={a} b={b}  budget F+q'={F + qn}  [{dt:.1f}s]\n")
    missing = [v for v in range(1, F + 1) if v not in N]
    w(f"  spectrum: {len(N)} sizes realised of 1..{F}; missing below F: {missing}\n")
    w(f"  top of spectrum (size,count): "
      f"{[(v, int(spec[v])) for v in range(max(1, F - 6), F + 1) if spec[v]]}\n")
    w(f"  v_0 (least v with N(w) <= F+1 for all realised w >= v) = {v0}\n")
    w(f"  spikes (v with N(v) > F+1): {[(v, N[v], N[v] - F - 1) for v in sorted(over)]}\n")
    w(f"  N(a={a}) = {N.get(a)}  (F {'-' if N.get(a, 0) < F else '+'}"
      f"{abs(N.get(a, 0) - F)});  N(b={b}) = {N.get(b)}\n")
    w(f"  Q*_3 from letters = max({N.get(a, 0)}+{a}, {N.get(b, 0)}+{b}) = "
      f"{max(N.get(a, 0) + a, N.get(b, 0) + b)}\n")
    for lv in (a, b):
        if lv in lw and lw[lv][1]:
            x0, L, vv, R = lw[lv][1]
            w(f"  witness N({lv}) = {lw[lv][0]}: x0={x0} (L,v,R)=({L},{vv},{R}) span={L+vv+R}\n")
    for J in range(1, JMAX + 1):
        tag = " <-- ATTAINS" if J == best else ""
        wit = qstar[J][1]
        w(f"  Q*_{J} = {qstar[J][0]}{tag}" + (f"   witness x={wit[0]} gaps={wit[1]}\n"
                                              if wit else "\n"))
    w(f"  F(M+q') = max_J Q*_J = {qstar[best][0]} (attained at J={best})\n")
    w("  profile v: N(v) [count of gaps of size v] :\n")
    line = []
    for v in sorted(N):
        line.append(f"{v}:{N[v]}({int(spec[v])})")
    w("    " + " ".join(line) + "\n")
    w("  profile N(v)+v: " + " ".join(f"{v}:{N[v] + v}" for v in sorted(N)) + "\n")
    out.flush()


if __name__ == "__main__":
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    out = open(dest, "w") if dest else sys.stdout
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    for i in range(2, len(PRIMES)):
        if PRIMES[i] > top:
            break
        if PRIMES[i] < lo:
            continue
        report(PRIMES[:i + 1], PRIMES[i + 1], out)
    if dest:
        out.close()

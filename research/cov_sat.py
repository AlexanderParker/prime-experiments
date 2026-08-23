"""Round 20 (mechanic): COV(M) - the exact coverability spectrum by CRT + SAT.

Slot k is blocked by gear q  iff  k = +-u_q (mod q), u_q = 6^{-1} mod q.
Anchoring a window's left opening at position 0 and writing a_q = (u_q - k)
mod q, gear q blocks exactly the positions i with

    i = a_q  or  i = a_q + s_q  (mod q),    s_q = -2 u_q mod q,

with the phase a_q in Z_q FREE.  By CRT every phase vector (a_q)_q is
attained by an actual k in the period, so occurrence questions about gaps
are pure finite constraint problems - no period scan:

  GAP:  v occurs  <=>  exists phases: 0 and v blocked by NO gear,
                       1..v-1 each blocked by SOME gear.
  F_j:  a window of j consecutive gaps with total span S occurs
        <=>  exists phases + a set W of j-1 interior spared positions:
             0, S, W all unblocked, everything else in (0,S) blocked.
  Q_j:  same with the j-2 MIDDLE gaps all >= a (the qualifying floor):
        pairwise distance constraints on W plus flank-distance constraints.

Encoding: one boolean per (gear, valid phase); exactly-one per gear;
one coverage clause per interior position; spared-position selectors w_i
with w_i -> not x_{q,a} for every phase covering i, coverage clause
(w_i OR ...), and exactly-(j-1) cardinality over the w_i.

Every SAT witness is CRT'd back to an explicit slot k and VERIFIED against
the machine by direct +-u_q arithmetic (assert), so a positive answer never
rests on the encoding being right.  Refutations are validated against the
full-period measured hole lists at machines 11..37 (r17 + r19, exact).

Usage:
  uv run python research/cov_sat.py validate
  uv run python research/cov_sat.py predict 41 43 47 53 [--vmax N]
  uv run python research/cov_sat.py fj y jmax [--smax N]   # exact F_j
  uv run python research/cov_sat.py qj y jmax a [--smax N] # exact Q_j
Run with: uv run --with python-sat python research/cov_sat.py ...
"""
import sys
import time
from math import prod
from pysat.solvers import Cadical153 as SatSolver
from pysat.formula import IDPool


def exactly_k(xs, k, pool):
    """Two-sided sequential-counter CNF for sum(xs) == k (pure Python -
    pysat's C CardEnc corrupts the heap over many instantiations)."""
    n = len(xs)
    if k > n:
        return [[]]
    cl = []
    # r[i][t] <-> at least t of xs[:i] true; i in 0..n, t in 1..k+1
    r = [[None] * (k + 2) for _ in range(n + 1)]
    for i in range(n + 1):
        for t in range(1, k + 2):
            r[i][t] = pool.id(("r", id(xs), i, t))
    for t in range(1, k + 2):
        cl.append([-r[0][t]])                      # r_{0,t} = false
    for i in range(1, n + 1):
        x = xs[i - 1]
        for t in range(1, k + 2):
            prev_t = r[i - 1][t]
            cl.append([-prev_t, r[i][t]])
            if t == 1:
                cl.append([-x, r[i][t]])
                cl.append([-r[i][t], prev_t, x])
            else:
                prev_t1 = r[i - 1][t - 1]
                cl.append([-x, -prev_t1, r[i][t]])
                cl.append([-r[i][t], prev_t, x])
                cl.append([-r[i][t], prev_t, prev_t1])
    cl.append([r[n][k]] if k >= 1 else [])
    cl.append([-r[n][k + 1]])
    return [c for c in cl if c != []]

MEASURED_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
              41: 91}   # 41: merge-law full period (padding37) + COV r20
MEASURED_HOLES = {11: [], 13: [9], 17: [17], 19: [19, 24], 23: [24],
                  29: [41, 42], 31: [54, 56, 57],
                  37: [73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87]}


def primes_upto(n):
    s = list(range(n + 1))
    for i in range(2, int(n ** 0.5) + 1):
        if s[i] == i:
            for j in range(i * i, n + 1, i):
                if s[j] == j:
                    s[j] = i
    return [i for i in range(2, n + 1) if s[i] == i]


def gears_of(y):
    return [p for p in primes_upto(y) if p >= 5]


def crt(residues, moduli):
    x, m = 0, 1
    for r, q in zip(residues, moduli):
        x += m * ((r - x) * pow(m, -1, q) % q)
        m *= q
    return x % m


def blocked_by(k, q):
    u = pow(6, -1, q)
    return k % q == u % q or k % q == (-u) % q


def verify_window(k, S, spared, qs):
    """Direct machine check: positions k+i for i in 0..S; spared set must be
    open, everything else blocked.  Returns True/False."""
    sp = set(spared) | {0, S}
    for i in range(S + 1):
        b = any(blocked_by(k + i, q) for q in qs)
        if i in sp:
            if b:
                return False
        else:
            if not b:
                return False
    return True


def build_gap_instance(S, qs, spared_budget=0, pool=None):
    """CNF for: 0..S window, endpoints spared, exactly `spared_budget`
    spared interior positions (selectors returned), rest covered.
    Returns (clauses, phase_vars, spare_vars, pool) or None if some gear
    cannot spare an endpoint."""
    pool = pool or IDPool()
    clauses = []
    phase = {}                       # (q, a) -> var
    cover = {i: [] for i in range(1, S)}   # position -> covering phase vars
    for q in qs:
        u = pow(6, -1, q)
        s = (-2 * u) % q
        forb = {0 % q, S % q, (-s) % q, (S - s) % q}
        vs = []
        for a in range(q):
            if a in forb:
                continue
            var = pool.id(("x", q, a))
            phase[(q, a)] = var
            vs.append(var)
            b = (a + s) % q
            for i in range(1, S):
                r = i % q
                if r == a or r == b:
                    cover[i].append(var)
        if not vs:
            return None
        clauses.append(vs)                       # at least one phase
        for i in range(len(vs)):                 # at most one phase
            for j in range(i + 1, len(vs)):
                clauses.append([-vs[i], -vs[j]])
    spare = {}
    if spared_budget:
        for i in range(1, S):
            spare[i] = pool.id(("w", i))
        for i in range(1, S):
            clauses.append([spare[i]] + cover[i])         # covered or spared
            for var in cover[i]:
                clauses.append([-spare[i], -var])          # spared => open
        clauses.extend(exactly_k([spare[i] for i in range(1, S)],
                                 spared_budget, pool))
    else:
        for i in range(1, S):
            if not cover[i]:
                return None
            clauses.append(cover[i])
    return clauses, phase, spare, pool


def solve_gap(v, qs):
    """Exact: does a gap of exactly v occur?  Returns (bool, witness k)."""
    if v <= 1:
        return True, None
    inst = build_gap_instance(v, qs)
    if inst is None:
        return False, None
    clauses, phase, _, _ = inst
    with SatSolver(bootstrap_with=clauses) as m:
        if not m.solve():
            return False, None
        model = set(l for l in m.get_model() if l > 0)
        res, mod = [], []
        for (q, a), var in phase.items():
            if var in model:
                res.append((pow(6, -1, q) - a) % q)   # a = (u - k) mod q
                mod.append(q)
        k = crt(res, mod)
        assert verify_window(k, v, [], qs), (v, k, "witness fails machine")
        return True, k


def solve_window(S, j, qs, min_middle=0):
    """Exact: does a window of j consecutive gaps with total span S occur
    (j-1 interior openings)?  min_middle > 0 additionally requires the j-2
    MIDDLE gaps >= min_middle (the qualifying constraint; flanks free).
    Returns (bool, witness k, spared positions)."""
    inst = build_gap_instance(S, qs, spared_budget=j - 1)
    if inst is None:
        return False, None, None
    clauses, phase, spare, pool = inst
    if min_middle > 1:
        # middle gaps are the gaps between consecutive SPARED positions
        # (interior openings); with j-1 >= 2 interior openings, every gap
        # between two spared positions is a middle gap.
        for i in range(1, S):
            for i2 in range(i + 1, min(i + min_middle, S)):
                clauses.append([-spare[i], -spare[i2]])
    with SatSolver(bootstrap_with=clauses) as m:
        if not m.solve():
            return False, None, None
        model = set(l for l in m.get_model() if l > 0)
        res, mod = [], []
        for (q, a), var in phase.items():
            if var in model:
                res.append((pow(6, -1, q) - a) % q)
                mod.append(q)
        k = crt(res, mod)
        sp = sorted(i for i in spare if spare[i] in model)
        assert verify_window(k, S, sp, qs), (S, j, k, sp, "witness fails")
        if min_middle > 1:
            pts = [0] + sp + [S]
            mids = [pts[t + 1] - pts[t] for t in range(1, len(pts) - 2)]
            assert all(g >= min_middle for g in mids), (S, j, sp, mids)
        return True, k, sp


def gap_spectrum(y, vmax, verbose=False, vmin=1, log_all=False):
    qs = gears_of(y)
    out = {}
    for v in range(vmin, vmax + 1):
        t0 = time.time()
        r, k = solve_gap(v, qs)
        out[v] = r
        if log_all:
            print(f"    v={v}: {r} ({time.time()-t0:.0f}s)", flush=True)
        elif verbose and time.time() - t0 > 10:
            print(f"    v={v}: {r} ({time.time()-t0:.0f}s)", flush=True)
    return out


def validate():
    print("VALIDATION against full-period measured spectra (machines 11..37,"
          " exact hole lists + F).  Witnesses machine-verified by assert.")
    all_ok = True
    for y in sorted(MEASURED_F):
        F = MEASURED_F[y]
        t0 = time.time()
        cov = gap_spectrum(y, F + 3)
        pred_holes = [v for v in range(1, F) if not cov[v]]
        top_ok = cov[F]
        above = [cov[F + 1], cov[F + 2], cov[F + 3]]
        ok = (pred_holes == MEASURED_HOLES[y]) and top_ok and not any(above)
        all_ok &= ok
        print(f"  machine {y:2d}: F={F}  holes pred {pred_holes} "
              f"vs meas {MEASURED_HOLES[y]}  v=F ok:{top_ok} "
              f"v=F+1..F+3:{above}  "
              f"{'AGREES' if ok else 'MISMATCH'}  ({time.time()-t0:.1f}s)",
              flush=True)
    print(f"=> {'ALL 8 MACHINES AGREE - COV(M) exact' if all_ok else 'MISMATCH'}")
    return all_ok


def predict(ys, vmax=None, vmin=1):
    for y in ys:
        qs = gears_of(y)
        P = prod(qs)
        vm = vmax or 170
        t0 = time.time()
        print(f"\nPREDICT machine {y} (gears {qs}), period {P:.4g} "
              f"- beyond any scan; COV exact by CRT+SAT.  v in "
              f"[{vmin},{vm}], every v logged (kill-resumable)",
              flush=True)
        cov = gap_spectrum(y, vm, vmin=vmin, log_all=True)
        realized = [v for v in cov if cov[v]]
        F = max(realized)
        holes = [v for v in range(vmin, F) if not cov[v]]
        tail_clear = all(not cov[v] for v in range(F + 1, vm + 1))
        print(f"  F({y}) = {F}  over scanned range  (all v in ({F},{vm}] "
              f"non-realizable: {tail_clear})")
        print(f"  holes in [{vmin},F) = {holes}")
        print(f"  ({time.time()-t0:.0f}s)", flush=True)


def fj(y, jmax, smax=None):
    """Exact F_j by descending S per depth from the chained cap
    F_j <= F_{j-1} + F_1 (exact: every gap of the window is a machine gap,
    so <= F_1, and the first j-1 gaps form a (j-1)-window)."""
    qs = gears_of(y)
    F1 = MEASURED_F.get(y)
    if F1 is None:
        cov = gap_spectrum(y, smax or 170)
        F1 = max(v for v in cov if cov[v])
    print(f"EXACT F_j, machine {y} (gears {qs}) - SAT, no scan; F_1 = {F1}",
          flush=True)
    prev = F1
    for j in range(2, jmax + 1):
        hi = min(prev + F1, smax) if smax else prev + F1
        best = None
        t0 = time.time()
        nun = 0
        for S in range(hi, 1, -1):
            t1 = time.time()
            r, k, sp = solve_window(S, j, qs)
            if r:
                best = (S, k, sp)
                break
            nun += 1
            print(f"    j={j} S={S} UNSAT ({time.time()-t1:.0f}s)",
                  flush=True)
        S, k, sp = best
        prev = S
        print(f"  F_{j}({y}) = {S}   ({nun} refutations from cap {hi}, "
              f"{time.time()-t0:.0f}s)   witness k = {k}  openings at +{sp}"
              f"  gaps {[b - a for a, b in zip([0]+sp, sp+[S])]}",
              flush=True)


def fjone(y, j, cap):
    """Resume helper: exact F_j for a single depth, descending from cap
    (pass cap = last-refuted-S minus 1 after a kill, or F_{j-1} + F_1)."""
    qs = gears_of(y)
    print(f"EXACT F_{j}, machine {y}, descend from {cap}", flush=True)
    for S in range(cap, 1, -1):
        t1 = time.time()
        r, k, sp = solve_window(S, j, qs)
        if r:
            print(f"  F_{j}({y}) = {S}   witness k = {k}  openings at "
                  f"+{sp}  gaps {[b - a for a, b in zip([0]+sp, sp+[S])]}",
                  flush=True)
            return
        print(f"    j={j} S={S} UNSAT ({time.time()-t1:.0f}s)", flush=True)


def qj(y, jmax, a, smax=None):
    qs = gears_of(y)
    print(f"EXACT Q_j (middle gaps >= {a}), machine {y} - SAT, no scan",
          flush=True)
    for j in range(3, jmax + 1):
        best = None
        for S in range(smax or 300, 1, -1):
            r, k, sp = solve_window(S, j, qs, min_middle=a)
            if r:
                best = (S, k, sp)
                break
        if best is None:
            print(f"  Q_{j}({y}; a={a}) = 0 (no qualifying window)")
            continue
        S, k, sp = best
        print(f"  Q_{j}({y}; a={a}) = {S}   witness k = {k}  openings +{sp}"
              f"  gaps {[b - a2 for a2, b in zip([0]+sp, sp+[S])]}",
              flush=True)


def main():
    args = sys.argv[1:]
    def popopt(name, default=None):
        if name in args:
            i = args.index(name)
            val = int(args[i + 1])
            del args[i:i + 2]
            return val
        return default
    vmax = popopt("--vmax")
    vmin = popopt("--vmin", 1)
    smax = popopt("--smax")
    cmd = args[0]
    if cmd == "validate":
        validate()
    elif cmd == "predict":
        predict([int(x) for x in args[1:]], vmax, vmin)
    elif cmd == "fj":
        fj(int(args[1]), int(args[2]), smax)
    elif cmd == "fjone":
        fjone(int(args[1]), int(args[2]), int(args[3]))
    elif cmd == "qj":
        qj(int(args[1]), int(args[2]), int(args[3]), smax)
    elif cmd == "one":
        y, v = int(args[1]), int(args[2])
        r, k = solve_gap(v, gears_of(y))
        print(f"machine {y} v={v}: {r}  witness k={k}")
    elif cmd == "pair":
        # adjacent gap pair (u, v): window S = u+v, one interior opening
        # forced at position u.  Exact occurrence of d_i = u, d_{i+1} = v.
        y, u, v = int(args[1]), int(args[2]), int(args[3])
        qs = gears_of(y)
        inst = build_gap_instance(u + v, qs, spared_budget=1)
        if inst is None:
            print(f"machine {y} pair ({u},{v}): False (endpoint-blocked)")
            return
        clauses, phase, spare, pool = inst
        clauses.append([spare[u]])
        with SatSolver(bootstrap_with=clauses) as m:
            if not m.solve():
                print(f"machine {y} pair ({u},{v}): False")
                return
            model = set(l for l in m.get_model() if l > 0)
            res, mod = [], []
            for (q, a), var in phase.items():
                if var in model:
                    res.append((pow(6, -1, q) - a) % q)
                    mod.append(q)
            k = crt(res, mod)
            assert verify_window(k, u + v, [u], qs)
            print(f"machine {y} pair ({u},{v}): True  witness k={k}")


if __name__ == "__main__":
    main()

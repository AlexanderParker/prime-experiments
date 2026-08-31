"""ROUND 28, LP-DUALITY THREAD - THE CUT LOOP'S LIMIT, IN ONE LP.

THE ROUND-27 OPEN QUESTION.  At machine 43, width 117 (the increment width),
case (0,0,0) at k = 3, the case-split cut loop's LP maximum fell

    44.2578, 44.2083, ..., 43.4856   over fifteen passes, against the 43 it
                                     must fall below

- about 0.05 per pass and decelerating.  Round 27 could not say whether that is
CONVERGENCE IN PRINCIPLE THAT IS MERELY SLOW or a GENUINE ASYMPTOTE ABOVE THE
TARGET, because the only instrument was the loop itself.

THIS FILE REPLACES THE INSTRUMENT.  The cut loop's rows are drawn from the
family of EXACTLY VALID degree-l cuts at a position: vectors lam over the atom
masks with

    lam_0 + sum_{S subset x, S nonempty} lam_S  >=  1   for every nonempty x.

A point z survives EVERY such cut at position i exactly when its degree-<=l
moment vector at i EXTENDS to a probability distribution on the NONEMPTY
subsets of the free gears (`lp_degree_range.completable`).  So the loop's limit
- the value it is decelerating towards, with unbounded cut generation and exact
separation - is the optimum of the LIFTED program

    V*(machine, W, case)  =  max  sum_j frow_j z_j
      over z >= 0 and, for every position i of pos, p_i >= 0 on the 2^n - 1
      nonempty subsets, subject to
        (B) every block of z sums to 1,
        (L) every consistency link,
        (N) sum_x p_{i,x} = 1,
        (M) sum_{j : i in O_j, mask(S_j) = m} z_j  =  sum_{x superset m} p_{i,x}
            for every atom mask m of a subset of size 1 or 2.

That is ONE LP with 2^n extra columns per position, and it is the exact answer
to the round-27 question:

    V* <  |pos|   the cell is CERTIFIABLE by this species - the loop converges
                  in principle, and the dual of THIS LP hands over the cuts
                  that do it, so the loop need not be run at all;
    V* >= |pos|   the cell has a GENUINE ASYMPTOTE at or above the target - no
                  amount of cut generation can ever certify it, and the primal
                  optimum is an exhibited feasible point (a REFUTATION).

Both verdicts are made EXACT here, never on the float value:
  * the certified side seeds `R.rows` with the lifted duals (rationalised, and
    repaired to exact validity by raising lam_0 to the exact deficit) and then
    runs the ordinary `decide_star`, which produces the same exact rational
    dual certificate the loop would have;
  * the refuted side rationalises the lifted primal and verifies it IN THE
    POLYTOPE with `RelaxStar.verify` (exact block sums, exact links, exact
    completability at every position, exact row value >= |pos|).
The float LP is used only to FIND the objects; every claim is re-checked in
exact rational arithmetic.

    python research/cutlimit_r28.py PROBE <y> <W> <k> [ws]
    python research/cutlimit_r28.py MAP                 # the frontier map
    python research/cutlimit_r28.py GATE
"""
import json
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import star_case                                                    # noqa
from star_case import (RelaxStar, decide_star, rationalise_star,    # noqa
                       repair_links, witness_candidates)
from lp_degree_range import gears_of, ZERO, ONE, zeta_values        # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')


# ===================================================== the lifted formulation
def lifted(R, verbose=False, eps=0.0, slack_floor=False):
    """Solve the lifted LP.  Returns dict(val, z, mu, nu) where mu[i] is the
    dual vector over atom masks at position i and nu[i] the mass-row dual.

    eps > 0 forces every p_{i,x} >= eps.  That keeps the optimum's moment
    vector STRICTLY INSIDE the completability cone at every position, so the
    rationalised primal is still exactly completable - which is what turns an
    ASYMPTOTE reading into an exhibited, exactly verified refutation."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    n, l = R.n, R.l
    subs = R.subs                      # atom masks, subs[0] = 0
    T = len(subs)
    npos = len(R.pos)
    NX = (1 << n) - 1                  # nonempty subsets, x - 1 indexes them
    NZ = len(R.cols)
    NV = NZ + npos * NX
    if verbose:
        print("    lifted LP: %d z-cols + %d p-cols = %d vars"
              % (NZ, npos * NX, NV), flush=True)

    ri, ci, vv, beq = [], [], [], []
    r = 0
    # (B) blocks
    for S in R.subsets:
        lo, hi = R.block_span[S]
        for j in range(lo, hi):
            ri.append(r); ci.append(j); vv.append(1.0)
        beq.append(1.0); r += 1
    # (L) links
    for (par, kids) in R.links:
        for j in kids:
            ri.append(r); ci.append(j); vv.append(1.0)
        ri.append(r); ci.append(par); vv.append(-1.0)
        beq.append(0.0); r += 1
    # (M) moment rows, and (N) mass rows
    mrow = {}
    supers = _supersets(n, subs)       # per atom index t>=1: list of x (1..NX)
    for a, i in enumerate(R.pos):
        base = NZ + a * NX
        # (N)
        for x in range(1, NX + 1):
            ri.append(r); ci.append(base + x - 1); vv.append(1.0)
        beq.append(1.0)
        nurow = r
        r += 1
        # (M), one per atom of size 1 or 2
        byatom = {}
        for (j, si) in R.bypos[i]:
            byatom.setdefault(si, []).append(j)
        first = r
        for t in range(1, T):
            for j in byatom.get(t, ()):
                ri.append(r); ci.append(j); vv.append(1.0)
            for x in supers[t]:
                ri.append(r); ci.append(base + x - 1); vv.append(-1.0)
            beq.append(0.0); r += 1
        mrow[i] = (nurow, first)
    A_eq = coo_matrix((vv, (ri, ci)), shape=(r, NV))
    c = np.zeros(NV)
    for j, v in enumerate(R.frow):
        if v:
            c[j] = -float(v)
    t0 = time.time()
    bnd = [(0.0, None)] * NZ + [(eps, None)] * (npos * NX)
    A_ub = b_ub = None
    if slack_floor:
        # MAXIMISE THE COMPLETABILITY SLACK instead of the row: one extra
        # variable t, floored under every LOW-ORDER atom (|x| <= 2) and the
        # full atom - the columns of the incidence matrix that span the
        # degree-<=2 moment space - subject to the row already clearing |pos|.
        # A point with t > 0 has moments STRICTLY inside the completability
        # cone at every position, so its rationalisation survives exactly.
        c = np.zeros(NV + 1)
        c[NV] = -1.0
        bnd = bnd + [(0.0, None)]
        A_eq = coo_matrix((vv, (ri, ci)), shape=(r, NV + 1))
        ui, uc, uv2, ub = [], [], [], []
        ur = 0
        low = [x for x in range(1, NX + 1) if bin(x).count('1') <= 2]
        low.append(NX)
        for a in range(npos):
            base = NZ + a * NX
            for x in low:
                ui.append(ur); uc.append(base + x - 1); uv2.append(-1.0)
                ui.append(ur); uc.append(NV); uv2.append(1.0)
                ub.append(0.0); ur += 1
        for j, v in enumerate(R.frow):
            if v:
                ui.append(ur); uc.append(j); uv2.append(-float(v))
        ub.append(-float(R.frhs)); ur += 1
        A_ub = coo_matrix((uv2, (ui, uc)), shape=(ur, NV + 1))
        b_ub = np.array(ub)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array(beq),
                  bounds=bnd, method='highs')
    if res.status not in (0, 2):
        # HiGHS occasionally returns status 15 (model_status Unknown) on these
        # - a NUMERICAL failure, not a verdict, and it must never be recorded
        # as one.  Retry on the interior-point solver, then simplex.
        for meth in ('highs-ipm', 'highs-ds'):
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                          b_eq=np.array(beq), bounds=bnd, method=meth)
            if res.status in (0, 2):
                break
    if res.status != 0:
        return dict(val=None, status=res.status, message=res.message,
                    secs=time.time() - t0)
    if slack_floor:
        return dict(val=-res.fun, z=res.x[:NZ], mu=None, nu=None,
                    slack=-res.fun, secs=time.time() - t0, nvars=NV + 1,
                    nrows=r)
    marg = res.eqlin.marginals
    mu, nu = {}, {}
    for i in R.pos:
        nurow, first = mrow[i]
        nu[i] = float(marg[nurow])
        mu[i] = [0.0] + [float(marg[first + t - 1]) for t in range(1, T)]
    return dict(val=-res.fun, z=res.x[:NZ], mu=mu, nu=nu,
                secs=time.time() - t0, nvars=NV, nrows=r)


def lifted_mass(R, verbose=False):
    """THE COMPANION PROGRAM FOR THE EMPTY CASE.

    When the lifted polytope is empty the solver returns no duals, so there is
    nothing to seed the cut loop with - and the loop is then left to find its
    own way, which at machine 43 width 117 is exactly the fifteen-pass crawl
    round 27 recorded.  This program is the same one with the mass rows
    RELAXED: position i carries total atom mass s_i in [0,1] instead of exactly
    1, the recursion row is imposed as a HARD constraint, and sum_i s_i is
    maximised.  It is always feasible, its optimum is |pos| exactly when the
    lifted polytope is nonempty, and its duals carry the same per-position cut
    lam_i = mu_i / nu_i.  So the empty case gets cuts after all."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n = R.n
    subs = R.subs
    T = len(subs)
    npos = len(R.pos)
    NX = (1 << n) - 1
    NZ = len(R.cols)
    NV = NZ + npos * NX + npos          # z, p, s
    ri, ci, vv, beq = [], [], [], []
    r = 0
    for S in R.subsets:
        lo, hi = R.block_span[S]
        for j in range(lo, hi):
            ri.append(r); ci.append(j); vv.append(1.0)
        beq.append(1.0); r += 1
    for (par, kids) in R.links:
        for j in kids:
            ri.append(r); ci.append(j); vv.append(1.0)
        ri.append(r); ci.append(par); vv.append(-1.0)
        beq.append(0.0); r += 1
    supers = _supersets(n, subs)
    mrow = {}
    for a, i in enumerate(R.pos):
        base = NZ + a * NX
        for x in range(1, NX + 1):
            ri.append(r); ci.append(base + x - 1); vv.append(1.0)
        ri.append(r); ci.append(NZ + npos * NX + a); vv.append(-1.0)
        beq.append(0.0)
        nurow = r
        r += 1
        byatom = {}
        for (j, si) in R.bypos[i]:
            byatom.setdefault(si, []).append(j)
        first = r
        for t in range(1, T):
            for j in byatom.get(t, ()):
                ri.append(r); ci.append(j); vv.append(1.0)
            for x in supers[t]:
                ri.append(r); ci.append(base + x - 1); vv.append(-1.0)
            beq.append(0.0); r += 1
        mrow[i] = (nurow, first)
    A_eq = coo_matrix((vv, (ri, ci)), shape=(r, NV))
    ui, uc, uv2 = [], [], []
    for j, v in enumerate(R.frow):
        if v:
            ui.append(0); uc.append(j); uv2.append(-float(v))
    A_ub = coo_matrix((uv2, (ui, uc)), shape=(1, NV))
    c = np.zeros(NV)
    c[NZ + npos * NX:] = -1.0
    bnd = [(0.0, None)] * (NZ + npos * NX) + [(0.0, 1.0)] * npos
    t0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=np.array([-float(R.frhs)]), A_eq=A_eq,
                  b_eq=np.array(beq), bounds=bnd, method='highs')
    if res.status != 0:
        return dict(val=None, status=res.status, message=res.message,
                    secs=time.time() - t0)
    marg = res.eqlin.marginals
    mu, nu = {}, {}
    for i in R.pos:
        nurow, first = mrow[i]
        nu[i] = float(marg[nurow])
        mu[i] = [0.0] + [float(marg[first + t - 1]) for t in range(1, T)]
    return dict(val=-res.fun, mass=-res.fun, npos=npos, mu=mu, nu=nu,
                secs=time.time() - t0)


_SUP_CACHE = {}


def _supersets(n, subs):
    key = (n, len(subs))
    if key in _SUP_CACHE:
        return _SUP_CACHE[key]
    out = {}
    for t in range(1, len(subs)):
        m = subs[t]
        rest = [i for i in range(n) if not (m >> i) & 1]
        xs = []
        for k in range(1 << len(rest)):
            x = m
            for b, i in enumerate(rest):
                if (k >> b) & 1:
                    x |= 1 << i
            xs.append(x)
        out[t] = xs
    _SUP_CACHE[key] = out
    return out


# ======================================================= exact cut extraction
def lam_from_duals(R, mu_i, nu_i, den=10 ** 6):
    """Turn one position's lifted duals into an EXACTLY VALID cut.

    Dual feasibility of the lifted LP at the p-columns says, for every nonempty
    x,   nu_i <= sum_{t : atom_t subset x} mu_{i,t}.   Dividing by nu_i > 0
    gives a valid cut with lam_0 = 0.  Rounding can break that; the repair is
    exact and always available - RAISING lam_0 by d adds d to every subset sum,
    so setting lam_0 = max(0, 1 - min_x f[x]) restores validity by
    construction, and the resulting row is asserted valid before it is used."""
    if nu_i is None or nu_i <= 1e-12:
        return None
    lam = [ZERO] * len(R.subs)
    for t in range(1, len(R.subs)):
        v = mu_i[t] / nu_i
        if abs(v) > 1e-12:
            lam[t] = Fraction(round(v * den), den)
    f = zeta_values(lam, R.n, R.subs)
    lo = min(f[1:])
    if lo < ONE:
        lam[0] = ONE - lo
        f = zeta_values(lam, R.n, R.subs)
        lo = min(f[1:])
    if lo < ONE:
        return None
    if lam[0] >= ONE:
        return None                    # degenerate: the row says nothing
    return tuple(lam)


def seed_rows(R, L, den=10 ** 6):
    """Append the lifted duals' cuts to R.rows (validity asserted exactly)."""
    added = 0
    for i in R.pos:
        lam = lam_from_duals(R, L['mu'][i], L['nu'][i], den)
        if lam is None:
            continue
        f = zeta_values(lam, R.n, R.subs)
        assert min(f[1:]) >= ONE, "seeded row is not an exactly valid cut"
        R.rows.append((i, lam))
        added += 1
    return added


# ================================================================ the decider
def limit_decide(R, tag=None, tb=600.0, verbose=False, den=10 ** 6,
                 witness=True):
    """V* first; then the EXACT verdict on the side V* falls."""
    out = dict(npos=len(R.pos), ncols=len(R.cols), frhs=int(R.frhs))
    L = lifted(R, verbose=verbose)
    if L['val'] is None:
        if L.get('status') == 2:
            # THE STRONGEST READING THERE IS: the limit polytope is EMPTY, so
            # the recursion row is not even needed - level-2 consistency alone
            # excludes a fully blocked window.  V* = -infinity.
            out.update(side='CERTIFIABLE', vstar=None, empty=True,
                       lifted_secs=L['secs'])
            n0 = len(R.rows)
            try:
                Lm = lifted_mass(R)
                if Lm['val'] is not None:
                    out['mass'] = Lm['val']
                    out['seeded'] = seed_rows(R, Lm, den)
            except Exception as e:                            # noqa: BLE001
                out['mass_error'] = repr(e)[:200]
            out['rows0'] = n0
            tr = []
            v, info = decide_star(R, verbose=verbose, maxrounds=400, tag=tag,
                                  time_budget=tb, trace=tr)
            out.update(verdict=v, its=info.get('its'), ops=info.get('ops'),
                       secs=info.get('secs'),
                       traj=[t['lp_max'] for t in tr])
            if v == 'CERTIFIED':
                out['lhs'] = str(info['lhs'])
                out['rhs'] = str(info['rhs'])
            return out
        out.update(verdict='LP-FAIL', message=L.get('message'))
        return out
    out.update(vstar=L['val'], lifted_secs=L['secs'], nvars=L['nvars'])
    out['gap'] = L['val'] - float(R.frhs)
    if L['val'] < float(R.frhs) - 1e-7:
        out['side'] = 'CERTIFIABLE'
        n0 = len(R.rows)
        out['seeded'] = seed_rows(R, L, den)
        tr = []
        v, info = decide_star(R, verbose=verbose, maxrounds=400, tag=tag,
                              time_budget=tb, trace=tr)
        out.update(verdict=v, its=info.get('its'), ops=info.get('ops'),
                   lp_max=info.get('lp_max'), secs=info.get('secs'),
                   rows0=n0, traj=[t['lp_max'] for t in tr])
        if v == 'CERTIFIED':
            out['lhs'] = str(info['lhs'])
            out['rhs'] = str(info['rhs'])
        return out
    out['side'] = 'ASYMPTOTE'
    if not witness:
        out['verdict'] = 'ASYMPTOTE-FLOAT'
        return out
    # EXACT refutation.  Re-solve with an INTERIOR floor on p so the optimum's
    # moment vector is strictly completable, then rationalise and verify.
    tried = []
    Ls = lifted(R, slack_floor=True)
    out['slack'] = Ls.get('slack')
    for src, den in ((Ls, 10 ** 8), (Ls, 10 ** 10), (Ls, 10 ** 6),
                     (L, 10 ** 8)):
        Le = src
        if Le['val'] is None:
            tried.append(('-', None))
            continue
        tried.append((den, Le['val']))
        zx = repair_links(R, rationalise_star(R, list(Le['z']), den))
        cands = [('lifted-slack-primal d=%d' % den, zx)]
        cands += witness_candidates(R, list(Le['z']), None)
        for how, cand in cands:
            if cand is None:
                continue
            try:
                ver = R.verify(cand)
                out.update(verdict='REFUTED', how=how,
                           row_value=str(ver['row_value']),
                           row_rhs=str(ver['row_rhs']),
                           row_slack=str(ver['row_slack']))
                if tag:
                    star_case.save_wit(tag, R, cand,
                                       dict(verdict='REFUTED', how=how,
                                            row_value=str(ver['row_value']),
                                            row_rhs=str(ver['row_rhs'])))
                return out
            except AssertionError:
                continue
    out['verdict'] = 'ASYMPTOTE-NOWITNESS'
    out['eps_tried'] = tried
    return out


def cell(y, W, k, ws=None, tag=None, tb=600.0, verbose=False):
    g = gears_of(y)
    ws = tuple(ws if ws is not None else [0] * k)
    R = RelaxStar(g, W, g[:k], ws)
    out = dict(machine=y, W=W, k=k, ws=list(ws))
    out.update(limit_decide(R, tag=tag, tb=tb, verbose=verbose))
    return out


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'PROBE')
    star_case.OUT = R28
    os.makedirs(R28, exist_ok=True)
    if cmd == 'PROBE':
        y, W, k = int(a[1]), int(a[2]), int(a[3])
        ws = tuple(int(x) for x in a[4].split(',')) if len(a) > 4 else None
        tb = float(a[5]) if len(a) > 5 else 600.0
        t0 = time.time()
        o = cell(y, W, k, ws, tag=None, tb=tb, verbose=True)
        print(json.dumps({kk: v for kk, v in o.items() if kk != 'traj'},
                         indent=1))
        print("  [%.1fs]" % (time.time() - t0))
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

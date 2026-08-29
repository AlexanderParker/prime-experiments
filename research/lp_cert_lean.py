"""FORMALIST, ROUND 27.  Transcribe the LP thread's case-split certificates into
the shape the Lean side consumes, and RE-DERIVE every number independently.

WHY THIS FILE EXISTS.  My round-26 addendum (formalist.md R26.8) asked the LP
thread for a JSON emission of `research/data/r26/cert_*.pkl`.  At mid-round-27
`research/data/r27/` did not exist, so this lane produced the emission itself
from the round-26 pickles, per its own contingency.

WHAT IT CHECKS (nothing is taken on the pickle's word):
  1.  The relaxation is REBUILT from the primes alone (`star_case.RelaxStar`).
  2.  Every cut row of every case is asserted EQUAL TO `base_cut` - i.e. the
      certified cases used no separated cuts at all, so the only coverage rows
      are "position i is blocked by some free gear".  (If this ever fails the
      Lean encoding below is not applicable and the script says so.)
  3.  The dual combination is RECOMPUTED from this file's own formulas - not
      `star_case.certificate_star` - in exact integers after scaling by the
      common denominator D, and its `lhs < rhs` is asserted to agree with the
      recorded verdict digit for digit.
  4.  The recursion-row coefficients `n_ab` are RECOMPUTED from the closed form
      the Lean side will use (n = 0 above gear index 1; n = |P| at gear index 0;
      n = |P| - max_{r7} |P & hits(7,r7)| at gear index 1) and asserted equal to
      `RelaxStar.frow` column by column.  This is the step that lets the kernel
      avoid an 8.2M-evaluation max-cover check.
  5.  SOUNDNESS GATE on the recursion row itself: over random phase tuples,
      #covered positions + sum_ab n_ab <= sum_a |A_a|  (the lowest-blocker
      inequality the Lean proof discharges pointwise).

Run:
    python research/lp_cert_lean.py GATE            # all assertions
    python research/lp_cert_lean.py EMIT 23 48 5    # rung 19->23, gear 5 held
"""
import json
import os
import pickle
import random
import sys
from fractions import Fraction
from math import lcm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_degree_range import gears_of, base_cut, hits, teeth     # noqa: E402
from star_case import RelaxStar                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R26 = os.path.join(HERE, 'data', 'r26')
OUT = os.path.join(HERE, 'data', 'r27')

RUNGS = {
    # tag: (y, W, held gears, filename pattern, phase-tuple list)
    '19_23': (23, 48, (5,), 'cert_gate_m23_w48_h%d.pkl',
              [(a,) for a in range(5)]),
    '23_29': (29, 63, (5, 7), 'cert_rung2_m29_w63_h%d_%d.pkl',
              [(a, b) for a in range(5) for b in range(7)]),
    '29_31': (31, 74, (5, 7), 'cert_rung2_m31_w74_h%d_%d.pkl',
              [(a, b) for a in range(5) for b in range(7)]),
}


# --------------------------------------------------------------- combinatorics
def blocks(q, r, W):
    """Lean's `bg`: gear q at phase r blocks position x < W."""
    t = teeth(q)
    return lambda x: (r + x) % q in t


def pair_index(nfree):
    return [(a, b) for a in range(nfree) for b in range(a + 1, nfree)]


def ncoef(gears, pos, a, b, ra, rb):
    """THE CLOSED FORM THE KERNEL WILL USE.

    n_ab = |P| - mc, P = the positions of `pos` blocked by both gears, mc the
    largest number of them any single phase choice of the gears BELOW index a
    can cover.  a = 0: no lower gears, mc = 0.  a = 1: one lower gear, mc is an
    explicit max over its q phases.  a >= 2: mc = |P| (n = 0) - asserted below
    against `RelaxStar.frow`, never assumed."""
    qa, qb = gears[a], gears[b]
    fa, fb = blocks(qa, ra, 10 ** 9), blocks(qb, rb, 10 ** 9)
    P = [x for x in pos if fa(x) and fb(x)]
    if a == 0:
        return len(P)
    if a == 1:
        q0 = gears[0]
        mc = max(sum(1 for x in P if blocks(q0, s, 10 ** 9)(x))
                 for s in range(q0))
        return len(P) - mc
    return 0


# ------------------------------------------------------------- the transcript
def transcribe(y, W, held, fname, phtuples, verbose=True):
    gears = gears_of(y)
    free = gears[len(held):]
    nf = len(free)
    pairs = pair_index(nf)
    cases = []
    for idx, ws in enumerate(phtuples):
        path = os.path.join(R26, fname % ws)
        with open(path, 'rb') as f:
            d = pickle.load(f)
        assert tuple(d['held']) == tuple(held), (d['held'], held)
        assert tuple(d['ws']) == tuple(ws), (d['ws'], ws)
        assert d['W'] == W
        R = RelaxStar(gears, W, held=held, ws=ws, l=2)
        assert not R.dead
        assert tuple(R.gears) == tuple(free)
        pos = list(R.pos)
        npos = len(pos)

        # (2) every cut row is the base cut
        bc = base_cut(R.n, 2)
        assert [p for p, _ in d['rows']] == pos, 'row positions != pos'
        for p, lam in d['rows']:
            assert tuple(lam) == tuple(bc), ('row is not base_cut at pos %d' % p)

        # (4) n_ab from the closed form == RelaxStar.frow, column by column
        for j, (S, r, O) in enumerate(R.cols):
            if len(S) == 1:
                assert R.frow[j] == Fraction(len(O)), 'singleton frow'
            else:
                a, b = free.index(S[0]), free.index(S[1])
                n = ncoef(free, pos, a, b, r[0], r[1])
                assert R.frow[j] == -Fraction(n), \
                    ('frow mismatch', S, r, R.frow[j], n)

        # ---- scale to integers
        D = 1
        for v in list(d['y']) + list(d['nu']) + [d['yff']]:
            D = lcm(D, v.denominator)
        yv = [int(v * D) for v in d['y']]
        nuv = [int(v * D) for v in d['nu']]
        yff = int(d['yff'] * D)
        assert all(v >= 0 for v in yv) and yff >= 0

        # ---- link indexing, made explicit (my R26.8 spec)
        # links are appended pair-major: for each pair (qa,qb) in `pairs` order,
        # first qb links (drop=0, parent = the qb singleton at phase rb), then
        # qa links (drop=1, parent = the qa singleton at phase ra).
        base = []
        acc = 0
        for (a, b) in pairs:
            base.append(acc)
            acc += free[b] + free[a]
        assert acc == len(nuv), (acc, len(nuv))
        # cross-check the ordering against RelaxStar.links
        k = 0
        for pi, (a, b) in enumerate(pairs):
            S = (free[a], free[b])
            for rb in range(free[b]):
                par, kids = R.links[k]
                assert R.cols[par][0] == (free[b],) and R.cols[par][1] == (rb,)
                assert base[pi] + rb == k
                k += 1
            for ra in range(free[a]):
                par, kids = R.links[k]
                assert R.cols[par][0] == (free[a],) and R.cols[par][1] == (ra,)
                assert base[pi] + free[b] + ra == k
                k += 1
        assert k == len(R.links)

        # (3) recompute lhs / rhs from this file's own formulas
        def a_single(a, r):
            f = blocks(free[a], r, W)
            s = sum(yv[t] + yff for t, x in enumerate(pos) if f(x))
            for pi, (u, v) in enumerate(pairs):
                if u == a:
                    s -= nuv[base[pi] + free[v] + r]
                elif v == a:
                    s -= nuv[base[pi] + r]
            return s

        def a_pair(pi, ra, rb):
            a, b = pairs[pi]
            return (-yff * ncoef(free, pos, a, b, ra, rb)
                    + nuv[base[pi] + rb] + nuv[base[pi] + free[b] + ra])

        MS = [max(a_single(a, r) for r in range(free[a])) for a in range(nf)]
        MP = [max(a_pair(pi, ra, rb)
                  for ra in range(free[a]) for rb in range(free[b]))
              for pi, (a, b) in enumerate(pairs)]
        lhs = sum(MS) + sum(MP)
        rhs = sum(yv) + yff * npos
        assert Fraction(lhs, D) == d['info']['lhs'], (lhs, D, d['info']['lhs'])
        assert Fraction(rhs, D) == d['info']['rhs'], (rhs, D, d['info']['rhs'])
        assert lhs < rhs, 'certificate does not certify'

        # THE EXCEPTION LISTS.  At gear index 1 the coefficient n_ab is zero
        # for ~96% of columns (the one gear below can cover the whole overlap),
        # and n = 0 is sound with no max-cover evaluation at all.  Listing the
        # exceptions lets the kernel skip the 11-phase maximum on every other
        # column - measured 27x on that block, which is what makes the 35-case
        # rung affordable.
        exc = {}
        for pi, (a, b) in enumerate(pairs):
            if a != 1:
                continue
            keys = [ra * free[b] + rb
                    for ra in range(free[a]) for rb in range(free[b])
                    if ncoef(free, pos, a, b, ra, rb) != 0]
            exc[str(pi)] = keys
        cases.append(dict(ws=list(ws), pos=pos, D=D, y=yv, yff=yff, nu=nuv,
                          base=base, lhs=lhs, rhs=rhs, MS=MS, MP=MP, exc=exc))
        if verbose:
            print('  case %-8s pos %2d  D %3d  lhs %6d < rhs %6d  (%s < %s)'
                  % (ws, npos, D, lhs, rhs, d['info']['lhs'], d['info']['rhs']))
    return dict(y=y, W=W, held=list(held), free=list(free),
                teeth={str(q): list(teeth(q)) for q in gears},
                pairs=[list(p) for p in pairs], cases=cases)


# ----------------------------------------------------- (5) the soundness gate
def soundness_gate(y, W, held, ws, trials=400, seed=11):
    """Over random phase tuples of the FREE gears: the lowest-blocker
    inequality the Lean proof discharges pointwise,
        #covered + sum_ab n_ab  <=  sum_a |A_a| ,
    and (the harder half) n_ab <= #{x in A_a & A_b : no gear below a covers x}."""
    gears = gears_of(y)
    free = gears[len(held):]
    nf = len(free)
    pairs = pair_index(nf)
    R = RelaxStar(gears, W, held=held, ws=ws, l=2)
    pos = list(R.pos)
    rng = random.Random(seed)
    for _ in range(trials):
        r = [rng.randrange(q) for q in free]
        f = [blocks(free[a], r[a], W) for a in range(nf)]
        cov = [x for x in pos if any(f[a](x) for a in range(nf))]
        deg = sum(1 for x in pos for a in range(nf) if f[a](x))
        tot = 0
        for (a, b) in pairs:
            n = ncoef(free, pos, a, b, r[a], r[b])
            low = sum(1 for x in pos
                      if f[a](x) and f[b](x)
                      and not any(f[c](x) for c in range(a)))
            assert n <= low, ('lowest-blocker bound', a, b, n, low)
            tot += n
        assert len(cov) + tot <= deg, ('recursion row', len(cov), tot, deg)
    return trials


# ------------------------------------------------------------------------ CLI
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'GATE'
    os.makedirs(OUT, exist_ok=True)
    if cmd == 'GATE':
        for tag, (y, W, held, fn, phs) in RUNGS.items():
            print('rung %s  (y=%d, W=%d, held=%s, %d cases)'
                  % (tag, y, W, held, len(phs)))
            data = transcribe(y, W, held, fn, phs)
            path = os.path.join(OUT, 'cert_%s.json' % tag)
            with open(path, 'w') as f:
                json.dump(data, f)
            print('  -> %s (%d bytes)' % (path, os.path.getsize(path)))
            for ws in phs[:3]:
                n = soundness_gate(y, W, held, ws)
                print('  soundness gate at ws=%s: %d random tuples OK' % (ws, n))
        print('ALL ASSERTIONS PASSED')
    elif cmd == 'EMIT':
        y, W = int(sys.argv[2]), int(sys.argv[3])
        held = tuple(int(v) for v in sys.argv[4:])
        tag = [t for t, v in RUNGS.items()
               if v[0] == y and v[1] == W and v[2] == held][0]
        _, _, _, fn, phs = RUNGS[tag]
        data = transcribe(y, W, held, fn, phs)
        path = os.path.join(OUT, 'cert_%s.json' % tag)
        with open(path, 'w') as f:
            json.dump(data, f)
        print('wrote', path)
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

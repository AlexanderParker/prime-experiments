"""ROUND 27, LP-DUALITY THREAD - JSON EMISSION OF THE CASE-SPLIT CERTIFICATES
for the Lean side, to Formalist's R26.8 specification.

WHAT IS EMITTED, per rung, into research/data/r27/:

  layout_<rung>.json     the CASE-INDEPENDENT layout.  With no required-open
                         positions every gear's phase domain is all of Z_q, so
                         the column list (S, r) and the link list are IDENTICAL
                         in every case of the rung - only the position set and
                         the recursion row depend on ws.  Asserted, not assumed.
  cert_<rung>_h<ws>.json one file per case: pos, the cut rows, the dual weights
                         y / nu / yff, the recursion row, and the claimed
                         lhs / rhs.  INTEGERS ONLY - every rational is a
                         [num, den] pair.
  manifest_<rung>.json   the held gears, the list of held-phase tuples the case
                         files are indexed by, and the EXHAUSTIVENESS assertion
                         that this list is exactly prod(Z_q : q held).

ATOM INDEXING IS EXPLICIT.  `atoms` is a list of 1 + n + C(n,2) BITMASKS over
the FREE gears: bit i of a mask is `free_gears[i]`.  Entry t of every cut row
`lam` is the coefficient of `atoms[t]`.  Cut validity is stated directly on
those masks:  for every nonempty x < 2^n,  sum_{atoms[t] subset x} lam[t] >= 1.
`atom_gears` gives the same subsets as explicit gear lists.

THE CERTIFICATE INEQUALITY, in the emitted symbols:

    O_j    = { p in pos : every gear q of S_j blocks p at phase r_j,q }
    a_j    = sum_r  y[r] * lam^r[ index of mask(S_j) ]  * [ pos_r in O_j ]
           + yff * frow[j]
           + sum_{k : j in kids(link k)} nu[k]  -  sum_{k : j = par(link k)} nu[k]
    lhs    = sum over blocks S of  max_{j in block S} a_j
    rhs    = sum_r y[r] * (1 - lam^r[0])  +  yff * |pos|
    CERTIFICATE  iff  y >= 0, yff >= 0, every row an exactly valid cut, lhs < rhs.

THE EMISSION IS GATED (`python research/emit_certs_r27.py GATE`): every JSON is
re-loaded from disk in a clean pass, the layout is rebuilt from the PRIMES ALONE
(`RelaxStar`) and asserted equal to the file's, every O_j is recomputed from the
gear/phase data, the row validity is re-checked by the exact zeta transform, and
lhs / rhs are recomputed from the file's integers and asserted equal to the
claimed pair and to the round-26 published verdict.

    python research/emit_certs_r27.py EMIT     # write the JSON
    python research/emit_certs_r27.py GATE     # re-verify every file from disk
"""
import json
import os
import pickle
import sys
import time
from fractions import Fraction
from itertools import combinations, product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_degree_range import (gears_of, budget, hits, subsets_upto,  # noqa
                             _atom_tables, base_cut, ZERO, ONE)
from star_case import RelaxStar                                     # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
R26 = os.path.join(HERE, 'data', 'r26')
R27 = os.path.join(HERE, 'data', 'r27')

# rung name -> (machine, k, source pickle tag template, published lhs/rhs)
RUNGS = {
    '19_23': dict(y=23, k=1, tag='rung_m23_w48_h%s',
                  published=(Fraction(202, 7), Fraction(607, 21))),
    '29_31': dict(y=31, k=2, tag='rung2_m31_w74_h%s',
                  published=None),
}


def fr(x):
    """a Fraction as an integer pair."""
    x = Fraction(x)
    return [x.numerator, x.denominator]


def unfr(p):
    return Fraction(int(p[0]), int(p[1]))


def _tagname(spec, ws):
    return spec['tag'] % "_".join(str(w) for w in ws)


# ============================================================ layout emission
def layout_of(y, k):
    """the case-independent layout: free gears, atom masks, columns, links."""
    g = gears_of(y)
    free = g[k:]
    n = len(free)
    gidx = {q: i for i, q in enumerate(free)}
    subs, sidx = _atom_tables(n, 2)
    subsets = subsets_upto(free, 2)
    mask = {S: sum(1 << gidx[q] for q in S) for S in subsets}
    cols, block_span, tupidx = [], [], {}
    for S in subsets:
        lo = len(cols)
        for r in product(*[range(q) for q in S]):
            tupidx[(S, r)] = len(cols)
            cols.append([mask[S], list(r)])
        block_span.append([mask[S], lo, len(cols)])
    links = []
    for S in subsets:
        if len(S) < 2:
            continue
        for drop in range(len(S)):
            Sp = S[:drop] + S[drop + 1:]
            for rp in product(*[range(q) for q in Sp]):
                kids = [tupidx[(S, rp[:drop] + (v,) + rp[drop:])]
                        for v in range(S[drop])]
                links.append([tupidx[(Sp, rp)], kids])
    return dict(
        machine=y, k=k, full_gears=list(g), held_gears=list(g[:k]),
        free_gears=list(free), n=n, l=2,
        atoms=list(subs),
        atom_gears=[[free[i] for i in range(n) if (m >> i) & 1] for m in subs],
        atom_index_of_mask=[[m, i] for m, i in sorted(sidx.items())],
        cols=cols, block_span=block_span, links=links,
        column_order=("for S in subsets of free_gears of size 1 then 2 "
                      "(lexicographic by gear index), for r in "
                      "itertools.product(range(q) for q in S) - a column is "
                      "[mask(S), list(r)]"),
        link_order=("for S of size 2 (same order), for drop in 0,1, for rp in "
                    "product(range(q) for q in S minus drop) - a link is "
                    "[parent column, [child columns, one per phase of the "
                    "dropped gear]]"),
    )


# ====================================================== per-case cert emission
def emit_case(rung, spec, ws, layout):
    y, k = spec['y'], spec['k']
    g = gears_of(y)
    W = budget(y)
    src = os.path.join(R26, 'cert_%s.pkl' % _tagname(spec, ws))
    with open(src, 'rb') as fh:
        d = pickle.load(fh)
    assert tuple(d['held']) == tuple(g[:k]) and tuple(d['ws']) == tuple(ws)
    assert d['W'] == W and d['l'] == 2 and d['openpts'] == ()
    R = RelaxStar(g, W, g[:k], ws)
    # layout is case-independent - ASSERTED, not assumed
    fi = {q: i for i, q in enumerate(layout['free_gears'])}
    assert [[sum(1 << fi[q] for q in S), list(r)]
            for (S, r, _O) in R.cols] == layout['cols'], "column layout differs"
    assert [[p, list(kd)] for (p, kd) in R.links] == layout['links'], \
        "link layout differs"
    assert all(v.denominator == 1 for v in R.frow), "recursion row not integral"

    rows = [[int(i), [fr(v) for v in lam]] for (i, lam) in d['rows']]
    base = base_cut(R.n, 2)
    out = dict(
        schema='lp-case-split-certificate/1',
        rung=rung.replace('_', '->'), machine=y, W=W, l=2,
        full_gears=list(g), held_gears=list(g[:k]), ws=list(ws),
        free_gears=list(layout['free_gears']), n=R.n,
        atoms=list(layout['atoms']), atom_gears=layout['atom_gears'],
        block_span=layout['block_span'],
        n_cols=len(R.cols), n_links=len(R.links),
        pos=[int(i) for i in R.pos], n_pos=len(R.pos),
        frow=[int(v) for v in R.frow],
        frhs=fr(R.frhs),
        rows=rows,
        rows_all_base_cut=all(tuple(lam) == base for (_i, lam) in d['rows']),
        y=[fr(v) for v in d['y']],
        yff=fr(d['yff']),
        nu=[fr(v) for v in d['nu']],
        lhs=fr(d['info']['lhs']), rhs=fr(d['info']['rhs']),
        ops=int(d['info']['ops']), iterations=int(d['info']['its']),
        source_pickle=os.path.basename(src),
    )
    p = os.path.join(R27, 'cert_%s_h%s.json'
                     % (rung, "_".join(str(w) for w in ws)))
    with open(p, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    return p, out


def emit(rung):
    spec = RUNGS[rung]
    y, k = spec['y'], spec['k']
    g = gears_of(y)
    held = g[:k]
    os.makedirs(R27, exist_ok=True)
    layout = layout_of(y, k)
    lp = os.path.join(R27, 'layout_%s.json' % rung)
    with open(lp, 'w') as fh:
        json.dump(layout, fh, separators=(',', ':'))
    cases = [tuple(ws) for ws in product(*[range(q) for q in held])]
    files = []
    t0 = time.time()
    for ws in cases:
        p, _o = emit_case(rung, spec, ws, layout)
        files.append(os.path.basename(p))
    man = dict(
        schema='lp-case-split-manifest/1',
        rung=rung.replace('_', '->'), machine=y, W=budget(y),
        full_gears=list(g), held_gears=list(held), free_gears=list(g[k:]),
        n_cases=len(cases), held_phase_tuples=[list(w) for w in cases],
        exhaustiveness=("held_phase_tuples is exactly the cartesian product "
                        "prod_{q in held_gears} Z_q, in itertools.product "
                        "order; every configuration of the machine has its "
                        "held gears at exactly one of these tuples"),
        exhaustiveness_holds=(cases == [tuple(w) for w in
                                        product(*[range(q) for q in held])]
                              and len(cases) == _prod(held)),
        case_files=files, layout_file=os.path.basename(lp),
        claim=("F(machine %d) <= %d : every case carries an exact rational "
               "dual certificate, so no fully blocked window of width %d "
               "exists at machine %d" % (y, budget(y), budget(y), y)),
    )
    mp = os.path.join(R27, 'manifest_%s.json' % rung)
    with open(mp, 'w') as fh:
        json.dump(man, fh, indent=1)
    sz = sum(os.path.getsize(os.path.join(R27, f)) for f in files)
    print("  %s: %d case files (%.1f KB total, %.1f KB each), layout %.1f KB,"
          " manifest ok=%s  [%.0fs]"
          % (rung, len(files), sz / 1024.0, sz / 1024.0 / len(files),
             os.path.getsize(lp) / 1024.0, man['exhaustiveness_holds'],
             time.time() - t0), flush=True)
    return man


def _prod(xs):
    t = 1
    for x in xs:
        t *= x
    return t


# ======================================================================= GATE
def zeta_exact(lam, n, atoms):
    """f[x] = sum_{atoms[t] subset x} lam[t], all 2^n atoms, exact."""
    f = [ZERO] * (1 << n)
    for m, v in zip(atoms, lam):
        if v:
            f[m] += v
    for i in range(n):
        bit = 1 << i
        for x in range(1 << n):
            if x & bit:
                f[x] += f[x ^ bit]
    return f


def check_case_json(path, rebuild=True):
    """RE-VERIFY A CERTIFICATE FROM ITS JSON ALONE.  Everything is recomputed
    from the file's integers plus the primes; nothing is taken from the pickle
    and nothing from the emitting process."""
    with open(path) as fh:
        J = json.load(fh)
    with open(os.path.join(os.path.dirname(path),
                           'layout_%s.json'
                           % J['rung'].replace('->', '_'))) as fh:
        L = json.load(fh)
    n, W = J['n'], J['W']
    free = J['free_gears']
    held, ws = J['held_gears'], J['ws']
    atoms = J['atoms']
    assert atoms == L['atoms'] and free == L['free_gears']
    assert atoms == list(_atom_tables(n, 2)[0]), "atom masks are not canonical"
    assert J['atom_gears'] == [[free[i] for i in range(n) if (m >> i) & 1]
                              for m in atoms], "atom gear lists disagree"

    # --- the position set, from the primes
    blocked = set()
    for q, w in zip(held, ws):
        blocked |= set(hits(q, w, W))
    pos = sorted(set(range(W)) - blocked)
    assert pos == J['pos'], "position set does not match the held phases"
    assert unfr(J['frhs']) == Fraction(len(pos)), "frhs != |pos|"

    # --- the columns, and O_j recomputed from the gear/phase data
    cols = L['cols']
    assert len(cols) == J['n_cols']
    posset = frozenset(pos)
    O = []
    for (m, r) in cols:
        S = [free[i] for i in range(n) if (m >> i) & 1]
        assert len(S) == len(r)
        acc = None
        for q, rq in zip(S, r):
            h = hits(q, rq, W)
            acc = set(h) if acc is None else (acc & h)
        O.append(frozenset(acc) & posset)

    # --- row validity, exactly, on the emitted atom masks
    rows = J['rows']
    for (i, lamp) in rows:
        assert i in posset, "cut row at a position outside pos"
        lam = [unfr(v) for v in lamp]
        assert len(lam) == len(atoms)
        f = zeta_exact(lam, n, atoms)
        assert min(f[1:]) >= ONE, "invalid cut row at position %d" % i

    # --- dual weights
    yv = [unfr(v) for v in J['y']]
    nuv = [unfr(v) for v in J['nu']]
    yff = unfr(J['yff'])
    assert len(yv) == len(rows), "y length != row count"
    assert len(nuv) == len(L['links']) == J['n_links'], "nu length != links"
    assert all(v >= 0 for v in yv) and yff >= 0, "negative dual weight"

    # --- a_j, lhs, rhs - purely from the file's integers
    aidx = {m: t for t, m in enumerate(atoms)}
    a = [ZERO] * len(cols)
    for r, (i, lamp) in enumerate(rows):
        if not yv[r]:
            continue
        lam = [unfr(v) for v in lamp]
        for j, (m, _rr) in enumerate(cols):
            if i in O[j]:
                v = lam[aidx[m]]
                if v:
                    a[j] += yv[r] * v
    frow = J['frow']
    assert len(frow) == len(cols)
    if yff:
        for j, v in enumerate(frow):
            if v:
                a[j] += yff * v
    for kk, (par, kids) in enumerate(L['links']):
        if nuv[kk]:
            for j in kids:
                a[j] += nuv[kk]
            a[par] -= nuv[kk]
    lhs = ZERO
    for (_m, lo, hi) in J['block_span']:
        lhs += max(a[lo:hi])
    rhs = sum(yv[r] * (ONE - unfr(lamp[0]))
              for r, (_i, lamp) in enumerate(rows)) + yff * Fraction(len(pos))
    assert lhs == unfr(J['lhs']), ("lhs disagrees", lhs, J['lhs'])
    assert rhs == unfr(J['rhs']), ("rhs disagrees", rhs, J['rhs'])
    assert lhs < rhs, ("certificate does not close", lhs, rhs)

    # --- the recursion row against a rebuild from the primes alone
    if rebuild:
        R = RelaxStar(tuple(J['full_gears']), W, tuple(held), tuple(ws))
        assert [int(v) for v in R.frow] == frow, "recursion row differs"
        assert tuple(R.pos) == tuple(pos)
        assert [[p, list(kd)] for (p, kd) in R.links] == L['links']
    return lhs, rhs


def gate():
    t0 = time.time()
    print("=" * 78)
    print("GATE  round-27 JSON emission, re-verified from disk")
    print("=" * 78, flush=True)
    for rung, spec in RUNGS.items():
        mp = os.path.join(R27, 'manifest_%s.json' % rung)
        with open(mp) as fh:
            M = json.load(fh)
        held = M['held_gears']
        cases = [tuple(w) for w in M['held_phase_tuples']]
        assert cases == [tuple(w) for w in product(*[range(q) for q in held])]
        assert len(cases) == _prod(held) == M['n_cases']
        assert M['exhaustiveness_holds'] is True
        print("  %s: EXHAUSTIVENESS - %d held-phase tuples = prod%s = %d  GREEN"
              % (rung, len(cases), tuple(held), _prod(held)), flush=True)
        vals = []
        for ws in cases:
            p = os.path.join(R27, 'cert_%s_h%s.json'
                             % (rung, "_".join(str(w) for w in ws)))
            assert os.path.basename(p) in M['case_files'], p
            lhs, rhs = check_case_json(p)
            vals.append((ws, lhs, rhs))
        assert len(vals) == len(cases)
        lo = min(rhs - lhs for _w, lhs, rhs in vals)
        if spec['published'] is not None:
            # the round-26 table publishes the FIRST case's row
            plhs, prhs = spec['published']
            assert (vals[0][1], vals[0][2]) == (plhs, prhs), \
                ("does not reproduce the published row", vals[0])
            print("  %s: all %d cases re-verified from JSON, lhs < rhs in every"
                  " case; case %s reproduces the published round-26 row"
                  " %s < %s (min margin over cases %s)  GREEN"
                  % (rung, len(vals), str(vals[0][0]), plhs, prhs, lo),
                  flush=True)
        else:
            print("  %s: all %d cases re-verified from JSON, lhs < rhs in every"
                  " case (min margin %s; case %s: %s < %s)  GREEN"
                  % (rung, len(vals), lo, str(vals[0][0]), vals[0][1],
                     vals[0][2]), flush=True)
    print("\n  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0))


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'EMIT')
    if cmd == 'EMIT':
        os.makedirs(R27, exist_ok=True)
        for rung in RUNGS:
            emit(rung)
    elif cmd == 'GATE':
        gate()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

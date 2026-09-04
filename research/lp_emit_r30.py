"""ROUND 30, LP-DUALITY THREAD - THE 47 -> 53 RUNG EMITTED FOR THE KERNEL,
WITH THE MARGIN COLUMN AND THE MIRROR TRANSCRIPTION.

THE CLAIM.  Machine 53 has no fully blocked window of width 171, i.e.

    F(53) <= 171 = F(47) + 53          <-- (D) at 47 -> 53, hypothesis-free,

from the primes 5..53 and nothing else: one exact rational dual certificate per
case of a case split on the held gears' phases, the split exhaustive over their
product (a MIXED-k tree if any level-4 case had to be refined - the step
manifest asserts the partition either way).  This is the rung the
spectrum-plus-depth criterion cannot certify (F_6(47) = 177 > 171).

WHAT IS EMITTED, into research/data/r30/:

  layout_47_53_k<k>.json          the case-independent column / link / atom
                                  layout of level k (emit_certs_r27.layout_of)
  cert_47_53_k<k>_h<ws>.json      ONE FILE PER CASE, integers only - schema
                                  lp-case-split-certificate/2 (SPARSE, below)
  manifest_47_53_k<k>.json        the level: its held-phase tuples, the whole
                                  MARGIN COLUMN, min / max, ops
  manifest_47_53.json             THE STEP MANIFEST: the partition assertion
  research/lp_rungs_r30.txt       the human-readable margin column

SCHEMA 2 (sparse) - every field of round 29's schema 1 is recoverable by
`expand_v1`, and `check_case_json` of round 27 verifies the expansion
unchanged.  The differences, all made because a level-4 case has 51,691
columns and 3,060 links (a dense file would be ~1 MB):
    frow_nz      [[j, v], ...]   the NONZERO recursion-row coefficients (int);
                                 every other coefficient is 0
    nu_nz        [[t, [num, den]], ...]   the NONZERO link weights
    rows         as schema 1 when any row is not the base cut; otherwise the
                 field `rows_base_cut_positions` lists the row positions and
                 `base_cut` gives the one lam vector they all share
    mirror_of    present on a TRANSCRIBED certificate: the representative
                 case it was mirrored from (lp_mirror_r30, the theorem)

Half the certificates are TRANSCRIBED, not solved: one representative per
mirror orbit was decided by lp_tree_r30, and the other member's certificate
is the mirror transcription.  The gate does not care which is which - every
file is re-verified from its own integers plus the primes, the relaxation
rebuilt at ITS OWN held phases.

    uv run python research/lp_emit_r30.py EMIT  [workers]      (all levels)
    uv run python research/lp_emit_r30.py STEP                 (the partition)
    uv run python research/lp_emit_r30.py GATE  [workers] [sample_frac]
    uv run python research/lp_emit_r30.py TXT
"""
import json
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_certs_r27 import (fr, unfr, layout_of, check_case_json,   # noqa
                            zeta_exact, _prod)
from lp_degree_range import gears_of, hits, base_cut, _atom_tables, ZERO, ONE  # noqa
from star_case import RelaxStar                                       # noqa
from lp_mirror_r30 import mirror_cert, mirror_ws                      # noqa
from lp_tree_r30 import reps, cellpath, mirror as _mirror             # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
R30 = os.path.join(HERE, 'data', 'r30')

STEP = (47, 53)
Y, W = 53, 171
RUNG = '47_53'
CLAIM = ("F(53) <= 171 = F(47) + 53 : (D) at 47 -> 53, hypothesis-free - the "
         "rung the spectrum-plus-depth criterion cannot certify "
         "(F_6(47) = 177 > 171)")


def lname(k):
    return 'layout_%s_k%d.json' % (RUNG, k)


def cname(k, ws):
    return 'cert_%s_k%d_h%s.json' % (RUNG, k, "_".join(str(w) for w in ws))


def rungname(k):
    return '%s_k%d' % (RUNG, k)


# ================================================================ schema 2
def compress(J1):
    J = dict(J1)
    J['schema'] = 'lp-case-split-certificate/2'
    J['frow_nz'] = [[j, int(v)] for j, v in enumerate(J1['frow']) if v]
    del J['frow']
    J['nu_nz'] = [[t, v] for t, v in enumerate(J1['nu']) if v[0] != 0]
    del J['nu']
    if J1['rows_all_base_cut']:
        J['rows_base_cut_positions'] = [i for (i, _lam) in J1['rows']]
        J['base_cut'] = J1['rows'][0][1]
        del J['rows']
    J['sparse_is'] = ("frow_nz / nu_nz list the NONZERO entries of the "
                      "recursion row / link weights of schema 1, indexed by "
                      "column / link number in the layout file; every other "
                      "entry is 0.  rows_base_cut_positions + base_cut stand "
                      "for schema 1's rows when every row is the base cut.")
    return J


def expand_v1(J2):
    J = dict(J2)
    J['schema'] = 'lp-case-split-certificate/1'
    frow = [0] * J2['n_cols']
    for j, v in J2['frow_nz']:
        frow[j] = v
    J['frow'] = frow
    nu = [[0, 1]] * J2['n_links']
    for t, v in J2['nu_nz']:
        nu[t] = v
    J['nu'] = nu
    if 'rows' not in J2:
        J['rows'] = [[i, J2['base_cut']] for i in J2['rows_base_cut_positions']]
    for key in ('frow_nz', 'nu_nz', 'rows_base_cut_positions', 'base_cut',
                'sparse_is'):
        J.pop(key, None)
    return J


# ===================================================== the representative
def rep_cert_v1(k, ws, layout):
    """schema-1 dict of the decided representative case, from its pickle and
    a rebuild of the relaxation from the primes (layout asserted equal)"""
    g = gears_of(Y)
    src = os.path.join(R30, 'cert_cell_m%d_w%d_k%d_h%s.pkl'
                       % (Y, W, k, "_".join(str(w) for w in ws)))
    with open(src, 'rb') as fh:
        d = pickle.load(fh)
    assert tuple(d['held']) == tuple(g[:k]) and tuple(d['ws']) == tuple(ws)
    assert d['W'] == W and d['l'] == 2 and d['openpts'] == ()
    R = RelaxStar(g, W, g[:k], tuple(ws))
    fi = {q: i for i, q in enumerate(layout['free_gears'])}
    assert [[sum(1 << fi[q] for q in S), list(r)]
            for (S, r, _O) in R.cols] == layout['cols'], "column layout differs"
    assert [[p, list(kd)] for (p, kd) in R.links] == layout['links'], \
        "link layout differs"
    assert all(v.denominator == 1 for v in R.frow), "recursion row not integral"
    lhs, rhs = Fraction(d['info']['lhs']), Fraction(d['info']['rhs'])
    margin = rhs - lhs
    assert margin > 0, "source pickle does not close"
    base = base_cut(R.n, 2)
    with open(cellpath(Y, W, k, ws)) as fh:
        cell = json.load(fh)
    assert cell['verdict'] == 'CERTIFIED'
    return dict(
        schema='lp-case-split-certificate/1',
        rung=rungname(k).replace('_', '->'), machine=Y, W=W, l=2,
        step=list(STEP), kind='D-rung',
        full_gears=list(g), held_gears=list(g[:k]), ws=list(ws),
        free_gears=list(layout['free_gears']), n=R.n,
        atoms=list(layout['atoms']), atom_gears=layout['atom_gears'],
        block_span=layout['block_span'],
        n_cols=len(R.cols), n_links=len(R.links),
        pos=[int(i) for i in R.pos], n_pos=len(R.pos),
        frow=[int(v) for v in R.frow], frhs=fr(R.frhs),
        rows=[[int(i), [fr(v) for v in lam]] for (i, lam) in d['rows']],
        rows_all_base_cut=all(tuple(lam) == base for (_i, lam) in d['rows']),
        n_rows=len(d['rows']),
        y=[fr(v) for v in d['y']], yff=fr(d['yff']), nu=[fr(v) for v in d['nu']],
        lhs=fr(lhs), rhs=fr(rhs), margin=fr(margin),
        margin_is="margin = rhs - lhs, the per-case slack of the certificate "
                  "inequality sum_S max_{j in S} a_j < sum_r y_r (1-lam^r_0) "
                  "+ yff |pos|",
        ops=int(d['info']['ops']), iterations=int(d['info']['its']),
        method=cell.get('method'),
        representative=True,
        source_pickle=os.path.basename(src),
    )


def emit_one(args):
    k, ws = args
    with open(os.path.join(R30, lname(k))) as fh:
        layout = json.load(fh)
    J1 = rep_cert_v1(k, ws, layout)
    out = [(list(ws), J1['margin'], J1['rows_all_base_cut'],
            J1['iterations'], J1['ops'], cname(k, ws), True)]
    with open(os.path.join(R30, cname(k, ws)), 'w') as fh:
        json.dump(compress(J1), fh, separators=(',', ':'))
    mws = mirror_ws(ws, J1['held_gears'], W)
    if mws != list(ws):
        M1 = mirror_cert(J1, layout)
        M1['representative'] = False
        M1['ops'] = J1['ops']
        with open(os.path.join(R30, cname(k, mws)), 'w') as fh:
            json.dump(compress(M1), fh, separators=(',', ':'))
        out.append((mws, M1['margin'], M1['rows_all_base_cut'],
                    M1['iterations'], M1['ops'], cname(k, mws), False))
    return out


def certified_reps(k):
    out = []
    for ws in reps(Y, W, k):
        p = cellpath(Y, W, k, ws)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            if json.load(fh).get('verdict') == 'CERTIFIED':
                out.append(tuple(ws))
    return out


def emit_level(k, workers=3):
    g = gears_of(Y)
    held = g[:k]
    layout = layout_of(Y, k)
    with open(os.path.join(R30, lname(k)), 'w') as fh:
        json.dump(layout, fh, separators=(',', ':'))
    cs = certified_reps(k)
    if not cs:
        print("  level %d: no certified representatives on disk" % k)
        return None
    t0 = time.time()
    with Pool(workers) as pool:
        res = [r for rr in pool.map(emit_one, [(k, ws) for ws in cs],
                                    chunksize=2) for r in rr]
    res.sort(key=lambda r: r[0])
    margins = [Fraction(*r[1]) for r in res]
    tuples = [tuple(r[0]) for r in res]
    assert len(set(tuples)) == len(tuples)
    man = dict(
        schema='lp-case-split-manifest/2',
        rung=rungname(k).replace('_', '->'), machine=Y, W=W, k=k,
        step=list(STEP), kind='D-rung',
        full_gears=list(g), held_gears=list(held), free_gears=list(g[k:]),
        n_cases=len(res), held_phase_tuples=[list(t) for t in tuples],
        n_representatives=sum(1 for r in res if r[6]),
        n_transcribed=sum(1 for r in res if not r[6]),
        select=('all' if len(res) == _prod(held) else 'partial'),
        exhaustiveness=(
            ("held_phase_tuples is exactly the cartesian product "
             "prod_{q in held_gears} Z_q; every configuration of the machine "
             "has its held gears at exactly one of these tuples")
            if len(res) == _prod(held) else
            ("PARTIAL BY DESIGN: the step's exhaustiveness is stated in "
             "manifest_%s.json, which pairs the levels" % RUNG)),
        exhaustiveness_holds=(len(res) == _prod(held)
                              and sorted(tuples) == sorted(
                                  product(*[range(q) for q in held]))
                              if len(res) == _prod(held) else None),
        case_files=[r[5] for r in res],
        layout_file=lname(k),
        rows_all_base_cut=all(r[2] for r in res),
        iterations_max=max(r[3] for r in res),
        ops_total=sum(r[4] for r in res),
        margin_column=[[list(r[0]), r[1]] for r in res],
        margin_min=fr(min(margins)), margin_max=fr(max(margins)),
        margin_is="per case, rhs - lhs of the certificate inequality",
        mirror=("one representative per mirror orbit was decided; the other "
                "member's certificate is the mirror transcription of "
                "lp_mirror_r30 (MIRROR(ws) = (1 - W - ws) mod q per held "
                "gear); every file re-verifies from its own integers"),
        claim=CLAIM,
    )
    with open(os.path.join(R30, 'manifest_%s.json' % rungname(k)), 'w') as fh:
        json.dump(man, fh, indent=1)
    sz = sum(os.path.getsize(os.path.join(R30, f)) for f in man['case_files'])
    print("  level k=%d: %d case files (%d decided + %d transcribed; %.1f MB,"
          " %.1f KB each), layout %.1f KB;\n    all rows base cut = %s, max"
          " iterations = %d, ops = %d;\n    MARGIN COLUMN min %s max %s   [%.0fs]"
          % (k, len(res), man['n_representatives'], man['n_transcribed'],
             sz / 1048576.0, sz / 1024.0 / len(res),
             os.path.getsize(os.path.join(R30, lname(k))) / 1024.0,
             man['rows_all_base_cut'], man['iterations_max'],
             man['ops_total'], min(margins), max(margins), time.time() - t0),
          flush=True)
    return man


# ============================================================ the STEP
def step_manifest():
    """the partition: the certified tuples of every level, each expanded to
    the leaves of the deepest level, cover prod(Z_q : q held at the deepest
    level) EXACTLY ONCE"""
    g = gears_of(Y)
    parts = []
    for k in (4, 5, 6):
        p = os.path.join(R30, 'manifest_%s.json' % rungname(k))
        if os.path.exists(p):
            with open(p) as fh:
                parts.append(json.load(fh))
    assert parts
    kmax = max(m['k'] for m in parts)
    held = g[:kmax]
    leaves = {}
    for m in parts:
        for t in m['held_phase_tuples']:
            t = tuple(t)
            for ext in product(*[range(q) for q in g[len(t):kmax]]):
                leaf = t + ext
                assert leaf not in leaves, ("a leaf is covered twice", leaf)
                leaves[leaf] = m['k']
    full = set(product(*[range(q) for q in held]))
    assert set(leaves) == full, "the split is not exhaustive"
    mc = [Fraction(*x) for m in parts for (_w, x) in m['margin_column']]
    out = dict(
        schema='lp-case-split-step-manifest/1',
        step=list(STEP), machine=Y, W=W, kind='D-rung',
        W_is="W_inc = F(47) + 53 = 118 + 53 = 171 (the (D) budget width)",
        parts=[dict(rung=m['rung'], k=m['k'], held_gears=m['held_gears'],
                    n_cases=m['n_cases'],
                    manifest_file='manifest_%s.json' % rungname(m['k']),
                    layout_file=m['layout_file']) for m in parts],
        n_cases_total=sum(m['n_cases'] for m in parts),
        deepest_k=kmax,
        exhaustiveness=(
            "every certified tuple of every level, expanded to the phases of "
            "the gears up to the deepest level (%s), covers prod(Z_q) = %d "
            "leaves EXACTLY ONCE: no leaf twice, no leaf missing.  ASSERTED "
            "by lp_emit_r30.step_manifest." % (list(held), len(full))),
        exhaustiveness_holds=True,
        leaves_by_level={str(k): sum(1 for v in leaves.values() if v == k)
                         for k in sorted(set(leaves.values()))},
        margin_min=fr(min(mc)), margin_max=fr(max(mc)),
        rows_all_base_cut=all(m['rows_all_base_cut'] for m in parts),
        ops_total=sum(m['ops_total'] for m in parts),
        claim=CLAIM + " - every case of the split carries an exact rational "
              "dual certificate, so machine 53 has no fully blocked window "
              "of width 171.",
    )
    p = os.path.join(R30, 'manifest_%s.json' % RUNG)
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print("  STEP MANIFEST %s: %s cases, PARTITION ASSERTED over %d leaves"
          " (%s); margin min %s max %s; all rows base cut = %s; %d exact ops"
          % (os.path.basename(p),
             " + ".join(str(m['n_cases']) for m in parts), len(full),
             out['leaves_by_level'], min(mc), max(mc),
             out['rows_all_base_cut'], out['ops_total']), flush=True)
    return out


# ================================================================== GATE
def check_case_dict(J, L, rebuild=True):
    """`emit_certs_r27.check_case_json` on a schema-1 dict, with the a_j
    accumulation indexed by position (same arithmetic, fewer Python steps).
    Everything is recomputed from the dict's integers plus the primes."""
    n, Wd = J['n'], J['W']
    free = J['free_gears']
    held, ws = J['held_gears'], J['ws']
    atoms = J['atoms']
    assert atoms == L['atoms'] and free == L['free_gears']
    assert atoms == list(_atom_tables(n, 2)[0]), "atom masks are not canonical"
    blocked = set()
    for q, w in zip(held, ws):
        blocked |= set(hits(q, w, Wd))
    pos = sorted(set(range(Wd)) - blocked)
    assert pos == J['pos'], "position set does not match the held phases"
    assert unfr(J['frhs']) == Fraction(len(pos)), "frhs != |pos|"
    cols = L['cols']
    assert len(cols) == J['n_cols']
    posset = frozenset(pos)
    aidx = {m: t for t, m in enumerate(atoms)}
    bypos = {i: [] for i in pos}
    for j, (m, r) in enumerate(cols):
        S = [free[i] for i in range(n) if (m >> i) & 1]
        acc = None
        for q, rq in zip(S, r):
            h = hits(q, rq, Wd)
            acc = set(h) if acc is None else (acc & h)
        for i in acc:
            if i in posset:
                bypos[i].append((j, aidx[m]))
    rows = J['rows']
    for (i, lamp) in rows:
        assert i in posset, "cut row at a position outside pos"
        lam = [unfr(v) for v in lamp]
        assert len(lam) == len(atoms)
        f = zeta_exact(lam, n, atoms)
        assert min(f[1:]) >= ONE, "invalid cut row at position %d" % i
    yv = [unfr(v) for v in J['y']]
    nuv = [unfr(v) for v in J['nu']]
    yff = unfr(J['yff'])
    assert len(yv) == len(rows), "y length != row count"
    assert len(nuv) == len(L['links']) == J['n_links'], "nu length != links"
    assert all(v >= 0 for v in yv) and yff >= 0, "negative dual weight"
    a = [ZERO] * len(cols)
    for r, (i, lamp) in enumerate(rows):
        if not yv[r]:
            continue
        lam = [unfr(v) for v in lamp]
        for (j, t) in bypos[i]:
            if lam[t]:
                a[j] += yv[r] * lam[t]
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
    assert unfr(J['margin']) == rhs - lhs, "margin disagrees"
    if rebuild:
        R = RelaxStar(tuple(J['full_gears']), Wd, tuple(held), tuple(ws))
        assert [int(v) for v in R.frow] == frow, "recursion row differs"
        assert tuple(R.pos) == tuple(pos)
        assert [[p, list(kd)] for (p, kd) in R.links] == L['links']
    return lhs, rhs


def _gate_one(args):
    path, lpath, ref = args
    with open(path) as fh:
        J2 = json.load(fh)
    with open(lpath) as fh:
        L = json.load(fh)
    J = expand_v1(J2)
    lhs, rhs = check_case_dict(J, L, rebuild=True)
    if ref:
        # the round-27 REFERENCE checker on the expanded file, unchanged
        tmp = tempfile.mkdtemp(prefix='lp_gate_')
        try:
            shutil.copy(lpath, os.path.join(tmp, 'layout_%s.json'
                                            % J['rung'].replace('->', '_')))
            p = os.path.join(tmp, os.path.basename(path))
            with open(p, 'w') as fh:
                json.dump(J, fh, separators=(',', ':'))
            l2, r2 = check_case_json(p)
            assert (l2, r2) == (lhs, rhs)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    m = rhs - lhs
    return [J['ws'], [m.numerator, m.denominator], J['rows_all_base_cut'],
            J2.get('representative')]


def gate(workers=3, sample=0.02):
    t0 = time.time()
    print("=" * 78)
    print("GATE  round-30 emission, re-verified FROM DISK (schema 2 expanded")
    print("      to schema 1; relaxation rebuilt from the primes at each")
    print("      file's OWN held phases; every cut row re-checked by the")
    print("      exact zeta transform; lhs / rhs / margin recomputed from the")
    print("      file's own integers; the round-27 reference checker")
    print("      `check_case_json` re-run unchanged on every self-mirror case")
    print("      and a %.0f%% random sample)" % (100 * sample))
    print("=" * 78, flush=True)
    random.seed(30)
    for k in (4, 5, 6):
        mp = os.path.join(R30, 'manifest_%s.json' % rungname(k))
        if not os.path.exists(mp):
            continue
        with open(mp) as fh:
            M = json.load(fh)
        held = M['held_gears']
        cs = [tuple(w) for w in M['held_phase_tuples']]
        assert len(cs) == M['n_cases'] == len(set(cs))
        if M['select'] == 'all':
            assert sorted(cs) == sorted(product(*[range(q) for q in held]))
            assert len(cs) == _prod(held)
            assert M['exhaustiveness_holds'] is True
            print("  k=%d: EXHAUSTIVENESS - %d held-phase tuples = prod%s = %d"
                  "  GREEN" % (k, len(cs), tuple(held), _prod(held)),
                  flush=True)
        else:
            print("  k=%d: PARTIAL SPLIT - %d of prod%s = %d tuples (the step"
                  " manifest states the partition)"
                  % (k, len(cs), tuple(held), _prod(held)), flush=True)
        lpath = os.path.join(R30, M['layout_file'])
        jobs = []
        for ws in cs:
            p = os.path.join(R30, cname(k, ws))
            assert os.path.basename(p) in M['case_files']
            selfm = (list(_mirror(ws, held, W)) == list(ws))
            jobs.append((p, lpath, selfm or random.random() < sample))
        with Pool(workers) as pool:
            res = pool.map(_gate_one, jobs, chunksize=4)
        margins = [Fraction(r[1][0], r[1][1]) for r in res]
        col = {tuple(w): tuple(m) for (w, m) in M['margin_column']}
        assert all(col[tuple(r[0])] == tuple(r[1]) for r in res), \
            "margin column on disk disagrees with the re-verified margins"
        assert unfr(M['margin_min']) == min(margins)
        assert unfr(M['margin_max']) == max(margins)
        assert all(r[2] for r in res) == M['rows_all_base_cut']
        nref = sum(1 for j in jobs if j[2])
        ntr = sum(1 for r in res if r[3] is False)
        print("  k=%d: %d/%d cases re-verified from JSON, lhs < rhs in EVERY"
              " case (%d of them mirror-transcribed);\n       margin column"
              " min %s max %s; all rows base cut = %s; reference checker"
              " agreed on %d files  GREEN"
              % (k, len(res), len(cs), ntr, min(margins), max(margins),
                 all(r[2] for r in res), nref), flush=True)
    sp = os.path.join(R30, 'manifest_%s.json' % RUNG)
    if os.path.exists(sp):
        step_manifest()          # re-asserts the partition from the files
    print("\n  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0), flush=True)


# =================================================================== TXT
def txt():
    lines = []
    sp = os.path.join(R30, 'manifest_%s.json' % RUNG)
    if os.path.exists(sp):
        with open(sp) as fh:
            S = json.load(fh)
        lines.append("STEP 47 -> 53  machine 53, width 171 = F(47) + 53")
        lines.append("  CLAIM: %s" % S['claim'])
        lines.append("  parts: %s" % ", ".join(
            "k=%d %d cases" % (p['k'], p['n_cases']) for p in S['parts']))
        lines.append("  PARTITION: %s" % S['exhaustiveness'])
        lines.append("  margin min %s max %s; all rows base cut = %s; %d exact"
                     " ops" % (Fraction(*S['margin_min']),
                               Fraction(*S['margin_max']),
                               S['rows_all_base_cut'], S['ops_total']))
        lines.append("")
    for k in (4, 5, 6):
        mp = os.path.join(R30, 'manifest_%s.json' % rungname(k))
        if not os.path.exists(mp):
            continue
        with open(mp) as fh:
            M = json.load(fh)
        lines.append("LEVEL k = %d  held %s  %d cases (%d decided, %d mirror-"
                     "transcribed)" % (k, tuple(M['held_gears']), M['n_cases'],
                                       M['n_representatives'],
                                       M['n_transcribed']))
        lines.append("  all rows base cut = %s; max iterations %d; %d exact"
                     " certificate ops" % (M['rows_all_base_cut'],
                                           M['iterations_max'],
                                           M['ops_total']))
        col = [(tuple(w), Fraction(*m)) for (w, m) in M['margin_column']]
        hist = {}
        for _w, m in col:
            hist[m] = hist.get(m, 0) + 1
        lines.append("  MARGIN COLUMN min %s max %s" % (min(m for _w, m in col),
                                                        max(m for _w, m in col)))
        lines.append("  margin histogram: "
                     + ", ".join("%s x%d" % (kk, v)
                                 for kk, v in sorted(hist.items())))
        lines.append("")
        lines.append("    case %-22s margin" % ("(%s)" % ",".join(
            "w%d" % q for q in M['held_gears'])))
        for w, m in col:
            lines.append("    %-27s %s" % (str(w), str(m)))
        lines.append("")
    out = "\n".join(lines)
    p = os.path.join(HERE, 'lp_rungs_r30.txt')
    with open(p, 'w') as fh:
        fh.write(__doc__ + "\n\n" + out + "\n")
    print(out[:3000])
    print("  written to %s" % p)


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'GATE')
    os.makedirs(R30, exist_ok=True)
    if cmd == 'EMIT':
        w = int(a[1]) if len(a) > 1 else 3
        for k in (4, 5, 6):
            if certified_reps(k):
                emit_level(k, w)
        step_manifest()
    elif cmd == 'STEP':
        step_manifest()
    elif cmd == 'GATE':
        gate(int(a[1]) if len(a) > 1 else 3,
             float(a[2]) if len(a) > 2 else 0.02)
    elif cmd == 'TXT':
        txt()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

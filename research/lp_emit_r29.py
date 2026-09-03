"""ROUND 29, LP-DUALITY THREAD - THE 31->37 RUNG EMITTED FOR THE KERNEL,
WITH A MARGIN COLUMN.  (Formalist's round-28 ask, top item.)

THE ASK.  "31->37 at the smallest k that certifies, at any width, plus the
emission ... your margin column is exactly the number I want alongside it - it
tells me how much room the kernel has."

WHAT IS EMITTED, into research/data/r29/, in the SAME SCHEMA the kernel side
already parses (round-27 `cert_19_23_h*.json` / round-28 `cert_inc_*.json`,
consumed by `research/lp_cert_lean.py` and `research/lp_cert_inc_r28.py`):

  layout_31_37.json        the CASE-INDEPENDENT column / link / atom layout.
                           With no required-open positions every free gear's
                           phase domain is all of Z_q, so the column list and
                           the link list are identical in every case - ASSERTED
                           per case, not assumed.
  cert_31_37_h<w5>_<w7>_<w11>.json
                           one file per case, INTEGERS ONLY (every rational an
                           [num, den] pair): pos, the cut rows, the dual
                           weights y / nu / yff, the recursion row, the claimed
                           lhs / rhs, AND `margin` = rhs - lhs.
  manifest_31_37.json      held gears, the 385 held-phase tuples, the
                           exhaustiveness assertion, and the whole MARGIN
                           COLUMN (one entry per case, plus min / max).

THE CLAIM.  Machine 37 has no fully blocked window of width 95, i.e.

    F(37) <= 95 = F(31) + 37          <-- (D) at 31 -> 37,

with NO census hypothesis, NO period, NO word list: 385 exact rational dual
certificates over the primes 5..37, exhaustive over the phases of the three
held gears 5, 7, 11.

THE MARGIN COLUMN is the per-case slack `rhs - lhs` of the certificate
inequality - the quantity a kernel transcription has to preserve.  Round 28
emitted it for the six increment steps (1 -> 1/384); here it is for the rung.

k = 3 IS THE SMALLEST k THAT CERTIFIES, AND THE SMALLER ONES ARE REFUTED, NOT
STALLED - see `research/lp_cells_r29.py K12`, which decides every k = 1 and
k = 2 case of the same width by the lifted LP and exhibits an exact in-polytope
point wherever V* >= |pos|.  (Round 26 recorded k = 2 as a cut-loop stall at
40.994 against 40; a stall is an undecided cell, never a verdict.)

SOURCE.  The 385 exact certificates are round 26's
`research/data/r26/cert_rung3_m37_w95_h*.pkl`.  Nothing is trusted from them:
the gate rebuilds the relaxation FROM THE PRIMES, recomputes the position set
from the held phases, re-checks every cut row's validity by the exact zeta
transform over all 2^n atoms, and recomputes lhs / rhs / margin from the file's
own integers.

    uv run python research/lp_emit_r29.py EMIT   [workers]
    uv run python research/lp_emit_r29.py GATE   [workers]
    uv run python research/lp_emit_r29.py TXT
"""
import json
import os
import pickle
import sys
import time
from fractions import Fraction
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emit_certs_r27 as E27                                          # noqa
from emit_certs_r27 import fr, unfr, layout_of, check_case_json, _prod  # noqa
from lp_degree_range import gears_of, hits, base_cut                  # noqa
from star_case import RelaxStar                                       # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
R26 = os.path.join(HERE, 'data', 'r26')
R28 = os.path.join(HERE, 'data', 'r28')
R29 = os.path.join(HERE, 'data', 'r29')

# rung -> everything needed to rebuild and to name the source pickles.
RUNGS = {
    '31_37': dict(y=37, k=3, W=95, src=R26, tag='cert_rung3_m37_w95_h%s',
                  claim=("F(37) <= 95 = F(31) + 37 : (D) at 31 -> 37, "
                         "hypothesis-free"),
                  kind='D-rung', step=(31, 37)),
    # THE SEVENTH INCREMENT STEP, the first past the six literal ones.
    # W_inc(37 -> 41) = F_2(37) + s_min(41) = 90 + 14 = 104.  The split is
    # MIXED: 376 of the 385 k = 3 cases certify; the other 9 are each REFUTED
    # by an exact in-polytope point, so each is split on gear 13's phases into
    # 13 k = 4 children, and all 117 children certify.  The union is a
    # PARTITION of the configuration space, so the rung stands.
    'inc_37_41_k3': dict(y=41, k=3, W=104, src=R29,
                         tag='cert_cell_m41_w104_k3_h%s',
                         claim=("the 376 k = 3 cases of F(41) <= 104 that "
                                "certify at three held gears"),
                         kind='increment', step=(37, 41),
                         select='certified'),
    'inc_37_41_k4': dict(y=41, k=4, W=104, src=R29,
                         tag='cert_cell_m41_w104_k4_h%s',
                         claim=("the 117 k = 4 children of the 9 k = 3 cases "
                                "that do NOT certify"),
                         kind='increment', step=(37, 41),
                         select='ondisk'),
}


def emit_one(args):
    rung, ws = args
    spec = RUNGS[rung]
    y, k, W = spec['y'], spec['k'], spec['W']
    g = gears_of(y)
    with open(os.path.join(R29, 'layout_%s.json' % rung)) as fh:
        layout = json.load(fh)
    src = os.path.join(spec['src'],
                       spec['tag'] % "_".join(str(w) for w in ws) + '.pkl')
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
    out = dict(
        schema='lp-case-split-certificate/1',
        rung=rung.replace('_', '->'), machine=y, W=W, l=2,
        step=list(spec['step']), kind=spec['kind'],
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
        lhs=fr(lhs), rhs=fr(rhs),
        margin=fr(margin),
        margin_is="margin = rhs - lhs, the per-case slack of the certificate "
                  "inequality sum_S max_{j in S} a_j < sum_r y_r (1-lam^r_0) "
                  "+ yff |pos|",
        ops=int(d['info']['ops']), iterations=int(d['info']['its']),
        source_pickle=os.path.basename(src),
    )
    p = os.path.join(R29, 'cert_%s_h%s.json'
                     % (rung, "_".join(str(w) for w in ws)))
    with open(p, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    return (list(ws), fr(margin), out['rows_all_base_cut'], out['iterations'],
            out['ops'], os.path.basename(p))


def rung_cases(rung):
    """The held-phase tuples this rung's certificate files are indexed by.

    'all'        every tuple of prod(Z_q : q held) - a plain exhaustive split;
    'certified'  only the cases whose decided cell CERTIFIED (the rest are
                 refined one gear deeper by the sibling rung);
    'ondisk'     exactly the cells on disk (the refinement's children)."""
    spec = RUNGS[rung]
    y, k, W = spec['y'], spec['k'], spec['W']
    held = gears_of(y)[:k]
    allc = [tuple(ws) for ws in product(*[range(q) for q in held])]
    sel = spec.get('select', 'all')
    if sel == 'all':
        return allc
    out = []
    for ws in allc:
        p = os.path.join(R29, 'cell_m%d_w%d_k%d_h%s.json'
                         % (y, W, k, "_".join(str(x) for x in ws)))
        if not os.path.exists(p):
            if sel == 'ondisk':
                continue
            raise SystemExit("cell missing: %s" % p)
        with open(p) as fh:
            if json.load(fh).get('verdict') == 'CERTIFIED':
                out.append(ws)
    return out


def emit(rung, workers=3):
    spec = RUNGS[rung]
    y, k, W = spec['y'], spec['k'], spec['W']
    g = gears_of(y)
    held = g[:k]
    os.makedirs(R29, exist_ok=True)
    layout = layout_of(y, k)
    with open(os.path.join(R29, 'layout_%s.json' % rung), 'w') as fh:
        json.dump(layout, fh, separators=(',', ':'))
    cs = rung_cases(rung)
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(emit_one, [(rung, ws) for ws in cs], chunksize=4)
    margins = [Fraction(m[0], m[1]) for (_w, m, _b, _i, _o, _f) in res]
    man = dict(
        schema='lp-case-split-manifest/2',
        rung=rung.replace('_', '->'), machine=y, W=W, k=k,
        step=list(spec['step']), kind=spec['kind'],
        full_gears=list(g), held_gears=list(held), free_gears=list(g[k:]),
        n_cases=len(cs), held_phase_tuples=[list(w) for w in cs],
        select=spec.get('select', 'all'),
        exhaustiveness=(
            ("held_phase_tuples is exactly the cartesian product "
             "prod_{q in held_gears} Z_q, in itertools.product order; every "
             "configuration of the machine has its held gears at exactly one "
             "of these tuples")
            if spec.get('select', 'all') == 'all' else
            ("PARTIAL BY DESIGN: this file lists only part of the split; the "
             "step's exhaustiveness is stated in manifest_inc_%d_%d.json, "
             "which pairs this rung with its sibling at one more held gear"
             % spec['step'])),
        exhaustiveness_holds=(len(cs) == _prod(held)
                              if spec.get('select', 'all') == 'all' else None),
        case_files=[r[5] for r in res],
        layout_file='layout_%s.json' % rung,
        rows_all_base_cut=all(r[2] for r in res),
        iterations_max=max(r[3] for r in res),
        ops_total=sum(r[4] for r in res),
        margin_column=[[list(r[0]), r[1]] for r in res],
        margin_min=fr(min(margins)), margin_max=fr(max(margins)),
        margin_is="per case, rhs - lhs of the certificate inequality",
        claim=spec['claim'],
    )
    if spec['kind'] == 'increment':
        man['witness_file'] = 'witness_inc_%d_%d.json' % spec['step']
        man['W_is'] = ("W_inc = F_2(%d) + s_min(%d)" % spec['step'])
    with open(os.path.join(R29, 'manifest_%s.json' % rung), 'w') as fh:
        json.dump(man, fh, indent=1)
    sz = sum(os.path.getsize(os.path.join(R29, f)) for f in man['case_files'])
    print("  %s: %d case files (%.1f KB total, %.1f KB each), layout %.1f KB;"
          "\n    all rows base cut = %s, max iterations = %d, ops = %d;"
          "\n    MARGIN COLUMN min %s max %s   [%.0fs]"
          % (rung, len(cs), sz / 1024.0, sz / 1024.0 / len(cs),
             os.path.getsize(os.path.join(R29, 'layout_%s.json' % rung))
             / 1024.0, man['rows_all_base_cut'], man['iterations_max'],
             man['ops_total'], min(margins), max(margins), time.time() - t0),
          flush=True)
    return man


# ======================================================================= GATE
def _check(path):
    lhs, rhs = check_case_json(path)
    with open(path) as fh:
        J = json.load(fh)
    m = unfr(J['margin'])
    assert m == rhs - lhs, ("margin disagrees", m, rhs - lhs)
    assert m > 0, "margin not positive"
    return [J['ws'], [m.numerator, m.denominator], J['rows_all_base_cut']]


def gate(workers=3, only=None):
    t0 = time.time()
    print("=" * 78)
    print("GATE  round-29 emission, re-verified FROM DISK (relaxation rebuilt")
    print("      from the primes; every cut row re-checked by the exact zeta")
    print("      transform; lhs / rhs / margin recomputed from the file's own")
    print("      integers)")
    print("=" * 78, flush=True)
    todo = [only] if only else [r for r in RUNGS
                                if os.path.exists(os.path.join(
                                    R29, 'manifest_%s.json' % r))]
    for rung in todo:
        with open(os.path.join(R29, 'manifest_%s.json' % rung)) as fh:
            M = json.load(fh)
        held = M['held_gears']
        cs = [tuple(w) for w in M['held_phase_tuples']]
        assert len(cs) == M['n_cases'] == len(set(cs))
        if M.get('select', 'all') == 'all':
            assert cs == [tuple(w) for w in
                          product(*[range(q) for q in held])]
            assert len(cs) == _prod(held)
            assert M['exhaustiveness_holds'] is True
            print("  %s: EXHAUSTIVENESS - %d held-phase tuples = prod%s = %d"
                  " GREEN" % (rung, len(cs), tuple(held), _prod(held)),
                  flush=True)
        else:
            print("  %s: PARTIAL SPLIT - %d of prod%s = %d held-phase tuples"
                  " (exhaustiveness is stated for the STEP, see"
                  " manifest_inc_%d_%d.json)"
                  % (rung, len(cs), tuple(held), _prod(held), *M['step']),
                  flush=True)
        paths = [os.path.join(R29, 'cert_%s_h%s.json'
                              % (rung, "_".join(str(w) for w in ws)))
                 for ws in cs]
        for p in paths:
            assert os.path.basename(p) in M['case_files'], p
        with Pool(workers) as pool:
            res = pool.map(_check, paths, chunksize=4)
        margins = [Fraction(r[1][0], r[1][1]) for r in res]
        assert [r[1] for r in res] == [c[1] for c in M['margin_column']], \
            "margin column on disk disagrees with the re-verified margins"
        assert unfr(M['margin_min']) == min(margins)
        assert unfr(M['margin_max']) == max(margins)
        assert all(r[2] for r in res) == M['rows_all_base_cut']
        print("  %s: %d/%d cases re-verified from JSON, lhs < rhs in EVERY"
              " case;\n       margin column min %s max %s;"
              " all rows base cut = %s  GREEN"
              % (rung, len(res), len(cs), min(margins), max(margins),
                 all(r[2] for r in res)), flush=True)
    print("\n  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0), flush=True)


# ============================================ the STEP manifest (mixed split)
def step_manifest(M=37, q=41, parts=('inc_37_41_k3', 'inc_37_41_k4')):
    """One manifest for a step whose case split is MIXED in k.

    THE EXHAUSTIVENESS ARGUMENT, stated so a kernel can check it: the k = 3
    tuples listed in part 1, plus the k = 4 tuples listed in part 2, form a
    PARTITION of prod(Z_5 x Z_7 x Z_11) - every k = 3 tuple is either in part 1
    or is the 3-prefix of exactly 13 tuples of part 2, and no tuple is in both
    roles.  ASSERTED here from the two files' own lists."""
    from lp_degree_range import gears_of as _g
    mans = []
    for r in parts:
        with open(os.path.join(R29, 'manifest_%s.json' % r)) as fh:
            mans.append(json.load(fh))
    m3, m4 = mans
    g = _g(q)
    A = set(tuple(w) for w in m3['held_phase_tuples'])
    B = [tuple(w) for w in m4['held_phase_tuples']]
    pref = {}
    for w in B:
        pref.setdefault(w[:3], set()).add(w[3])
    assert all(v == set(range(g[3])) for v in pref.values()), \
        "a refined case does not carry ALL of the next gear's phases"
    assert not (A & set(pref)), "a case is both certified and refined"
    full = set(tuple(w) for w in product(*[range(p) for p in g[:3]]))
    assert A | set(pref) == full, "the split is not exhaustive"
    mc = [Fraction(*x) for (_w, x) in m3['margin_column']] + \
         [Fraction(*x) for (_w, x) in m4['margin_column']]
    out = dict(
        schema='lp-case-split-step-manifest/1',
        step=[M, q], machine=q, W=m3['W'], kind='increment',
        W_is="W_inc = F_2(%d) + s_min(%d) = 90 + 14 = %d" % (M, q, m3['W']),
        parts=[dict(rung=m['rung'], k=m['k'], held_gears=m['held_gears'],
                    n_cases=m['n_cases'],
                    manifest_file='manifest_%s.json' % r,
                    layout_file=m['layout_file']) for r, m in zip(parts, mans)],
        n_cases_total=m3['n_cases'] + m4['n_cases'],
        exhaustiveness=(
            "the %d k=3 tuples of part 1 and the %d k=4 tuples of part 2 "
            "PARTITION prod(Z_5 x Z_7 x Z_11): the 3-prefixes of part 2 are "
            "exactly the %d tuples part 1 omits, and each carries all 13 "
            "phases of gear 13.  ASSERTED by lp_emit_r29.step_manifest."
            % (m3['n_cases'], m4['n_cases'], len(pref))),
        exhaustiveness_holds=True,
        refined_k3_cases=[list(w) for w in sorted(pref)],
        margin_min=fr(min(mc)), margin_max=fr(max(mc)),
        rows_all_base_cut=(m3['rows_all_base_cut']
                           and m4['rows_all_base_cut']),
        ops_total=m3['ops_total'] + m4['ops_total'],
        witness_file='witness_inc_%d_%d.json' % (M, q),
        claim=("F(%d) <= %d = F_2(%d) + s_min(%d): every case of the mixed "
               "split carries an exact rational dual certificate, so machine "
               "%d has no fully blocked window of width %d.  With the witness "
               "file's lower half (an exhibited adjacent pair of machine %d "
               "of span F_2(%d) = 90) this is the INCREMENT LAW at %d -> %d."
               % (q, m3['W'], M, q, q, m3['W'], M, M, M, q)),
    )
    p = os.path.join(R29, 'manifest_inc_%d_%d.json' % (M, q))
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print("  STEP MANIFEST %s: %d + %d = %d cases, PARTITION ASSERTED;"
          " margin min %s max %s; all rows base cut = %s; %d exact ops"
          % (os.path.basename(p), m3['n_cases'], m4['n_cases'],
             out['n_cases_total'], min(mc), max(mc),
             out['rows_all_base_cut'], out['ops_total']), flush=True)
    return out


# =================================================================== WITNESS
def witness_at(M, s, a):
    """The exact-cover backtrack of `increment_cert_r27.witness_f2` run at ONE
    prescribed split (a, s - a).

    WHY THIS EXISTS, and it is a cost finding worth recording: the round-27
    routine sweeps `a` upward from 1, and at machine 37 the split a = 1 needs
    89 - 1 positions covered while 0, 1 and 90 stay open - which is
    UNREALISABLE (89 > F(37) = 88) and so has to be EXHAUSTED, not found.  That
    exhaustion did not finish in 600 s.  The maximiser is known from the
    project's own record - the m37 F_2 window is the pair (2, 88) - so the
    search is run at that split directly.  Nothing is assumed from the record:
    if the split is not realisable the backtrack returns None, and the object
    that is emitted is re-checked from its own numbers by CRT."""
    from lp_degree_range import hits, gears_of as _g
    g = _g(M)
    span = s + 1
    need = frozenset(set(range(1, s)) - {a})
    keep = (0, a, s)
    opts = {q: [(r, frozenset(hits(q, r, span))) for r in range(q)
                if not any(p in hits(q, r, span) for p in keep)] for q in g}

    def rec(covered, avail, chosen):
        if covered >= need:
            return chosen
        p = min(need - covered)
        for i, q in enumerate(avail):
            for (r, h) in opts[q]:
                if p in h:
                    out = rec(covered | h, avail[:i] + avail[i + 1:],
                              chosen + [(q, r)])
                    if out is not None:
                        return out
        return None

    got = rec(frozenset(), tuple(g), [])
    if got is None:
        return None
    ph = dict(got)
    for q in g:
        if q not in ph:
            ph[q] = opts[q][0][0]
    r = tuple(ph[q] for q in g)
    blocked = set()
    for q, rq in zip(g, r):
        blocked |= set(hits(q, rq, span))
    openp = sorted(set(range(span)) - blocked)
    if openp != [0, a, s]:
        return None
    return dict(machine=M, phases=list(r), gears=list(g), span=s,
                split=(a, s - a), openings=openp)


def witness(M=37, q=41, a=2):
    """The LOWER half of the increment law at M -> q': an EXPLICIT
    configuration of machine M realising an adjacent gap pair of sum F_2(M),
    as a phase vector, checked by CRT arithmetic on [0, F_2(M)] with no period
    scan.  Same object and same schema as round 28's `witness_inc_*.json`."""
    import increment_cert_r27 as IC
    from emit_inc_r28 import check_witness_json
    import emit_inc_r28 as E28
    from lp_degree_range import teeth
    IC.F2.setdefault(M, {37: 90}[M])          # F_2(37) = 90, project record
    w = witness_at(M, IC.F2[M], a) if a else IC.witness_f2(M)
    assert w is not None and IC.check_witness(w) and w['span'] == IC.F2[M]
    wj = dict(
        schema='lp-realisability-witness/1',
        machine=M, increment_step=[M, q], F2_claim=IC.F2[M],
        gears=list(w['gears']), phases=list(w['phases']),
        teeth={str(qq): list(teeth(qq)) for qq in w['gears']},
        span=w['span'], openings=list(w['openings']), split=list(w['split']),
        recipe=("blocked = union over gears q of {i in [0, span] : "
                "(i + phases[q]) mod q in teeth[q]}; the open positions of "
                "[0, span] are exactly `openings` = [0, a, span], so machine "
                "%d realises the adjacent gap pair (%d, %d) whose sum is "
                "F_2(%d) = %d.  No period scan is used or needed."
                % (M, w['split'][0], w['split'][1], M, IC.F2[M])),
    )
    p = os.path.join(R29, 'witness_inc_%d_%d.json' % (M, q))
    with open(p, 'w') as fh:
        json.dump(wj, fh, indent=1)
    E28.R28 = R29
    J = check_witness_json(p)
    print("  WITNESS  F_2(%d) >= %d  split %s  openings %s  phases %s"
          "   RE-CHECKED FROM DISK BY CRT  GREEN"
          % (M, J['F2_claim'], tuple(J['split']), J['openings'], J['phases']),
          flush=True)
    return p


# ======================================================================== TXT
def txt():
    """the persisted human-readable result: the margin column."""
    lines = []
    for rung in [r for r in RUNGS
                 if os.path.exists(os.path.join(R29,
                                                'manifest_%s.json' % r))]:
        with open(os.path.join(R29, 'manifest_%s.json' % rung)) as fh:
            M = json.load(fh)
        lines.append("RUNG %s   machine %d, width %d, k = %d, held gears %s"
                     % (M['rung'], M['machine'], M['W'], M['k'],
                        tuple(M['held_gears'])))
        lines.append("  CLAIM: %s" % M['claim'])
        lines.append("  %d cases = prod%s, exhaustive; all rows base cut = %s;"
                     " max iterations %d; %d exact certificate ops"
                     % (M['n_cases'], tuple(M['held_gears']),
                        M['rows_all_base_cut'], M['iterations_max'],
                        M['ops_total']))
        lines.append("  MARGIN COLUMN (per case: rhs - lhs), min %s / max %s"
                     % (Fraction(*M['margin_min']),
                        Fraction(*M['margin_max'])))
        lines.append("")
        lines.append("    case (w5,w7,w11)      margin      case ..."
                     "        margin")
        col = [(tuple(w), Fraction(*m)) for (w, m) in M['margin_column']]
        half = (len(col) + 1) // 2
        for i in range(half):
            a = col[i]
            b = col[i + half] if i + half < len(col) else None
            s = "    %-18s %-12s" % (str(a[0]), str(a[1]))
            if b:
                s += "  %-18s %-12s" % (str(b[0]), str(b[1]))
            lines.append(s)
        lines.append("")
        hist = {}
        for _w, m in col:
            hist[m] = hist.get(m, 0) + 1
        lines.append("  margin histogram: "
                     + ", ".join("%s x%d" % (k, v)
                                 for k, v in sorted(hist.items())))
        lines.append("")
    out = "\n".join(lines)
    p = os.path.join(HERE, 'lp_rungs_r29.txt')
    with open(p, 'w') as fh:
        fh.write(__doc__ + "\n\n" + out + "\n")
    print(out)
    print("  written to %s" % p)


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'GATE')
    os.makedirs(R29, exist_ok=True)
    if cmd == 'EMIT':
        rung = a[1]
        emit(rung, int(a[2]) if len(a) > 2 else 3)
    elif cmd == 'GATE':
        rest = a[1:]
        only = rest[0] if rest and not rest[0].isdigit() else None
        w = int(rest[-1]) if rest and rest[-1].isdigit() else 3
        gate(w, only)
    elif cmd == 'STEP':
        step_manifest()
    elif cmd == 'WITNESS':
        witness(int(a[1]) if len(a) > 1 else 37,
                int(a[2]) if len(a) > 2 else 41,
                int(a[3]) if len(a) > 3 else 2)
    elif cmd == 'TXT':
        txt()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

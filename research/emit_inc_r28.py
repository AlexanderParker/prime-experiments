"""ROUND 28, LP-DUALITY THREAD - JSON EMISSION OF THE INCREMENT-WIDTH
CERTIFICATES (and of the realisability witnesses) for the Lean side.

WHAT THIS IS FOR.  Formalist's round-28 item 1 consumes the six increment
certificates of round 27.  They transcribe my pickles independently
(`research/lp_cert_inc_r28.py`); this file is the SECOND SOURCE - the same
objects in the schema their round-26.8 spec named and my round-27 emission
used, so the two can be diffed as exact rationals the way `cert_19_23_h*.json`
was diffed last round.

THE CLAIM PER STEP.  For the step M -> q' at the increment width
W_inc = F_2(M) + s_min(q'):

  UPPER HALF (dual certificates, one per case, emitted here):
      no fully blocked window of width W_inc exists at machine q',
      i.e. F(q') <= W_inc = F_2(M) + s_min(q').
  LOWER HALF (a realisability witness, emitted here as `witness_<step>.json`):
      machine M has an adjacent gap PAIR summing to F_2(M) - an explicit phase
      vector with exactly three open positions 0 < a < F_2(M) in [0, F_2(M)],
      checked by CRT arithmetic with no period scan.
  Together: F(M + q') - F_2(M) <= s_min(q'), the increment law at that step.

EMITTED into research/data/r28/, per step:
  layout_inc_<step>.json      the case-independent column/link layout
  cert_inc_<step>_h<ws>.json  one file per case, INTEGERS ONLY
  manifest_inc_<step>.json    held-phase tuples + the exhaustiveness assertion
  witness_inc_<step>.json     the lower half, with its own re-check recipe

GATE (`python research/emit_inc_r28.py GATE`): every case JSON is re-verified
from disk by `emit_certs_r27.check_case_json` - relaxation rebuilt FROM THE
PRIMES, position set recomputed from the held phases, every cut row re-checked
valid by the exact zeta transform, lhs/rhs recomputed from the file's own
integers and asserted to close - and every witness JSON is re-checked by CRT
arithmetic from its own phase vector.

    python research/emit_inc_r28.py EMIT
    python research/emit_inc_r28.py GATE
"""
import json
import os
import pickle
import sys
import time
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emit_certs_r27 as E27                                        # noqa
from emit_certs_r27 import fr, layout_of, check_case_json, _prod     # noqa
from lp_degree_range import gears_of, hits, base_cut                 # noqa
from star_case import RelaxStar                                      # noqa
from increment_cert_r27 import (F2, smin, STEPS, w_inc,              # noqa
                                witness_f2, check_witness)

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.join(HERE, 'data', 'r27')
R28 = os.path.join(HERE, 'data', 'r28')

# step -> k (the held-gear count the round-27 run needed at the increment
# width).  Read off the pickles on disk, and ASSERTED against them.
KOF = {(11, 13): 1, (13, 17): 1, (17, 19): 1,
       (19, 23): 2, (23, 29): 2, (29, 31): 2}


def stepname(M, q):
    return "%d_%d" % (M, q)


def emit_case(M, q, ws, layout):
    y, W, k = q, w_inc(M, q), KOF[(M, q)]
    g = gears_of(y)
    tag = 'inc_m%d_w%d_h%s' % (y, W, "_".join(str(x) for x in ws))
    src = os.path.join(R27, 'cert_%s.pkl' % tag)
    with open(src, 'rb') as fh:
        d = pickle.load(fh)
    assert tuple(d['held']) == tuple(g[:k]) and tuple(d['ws']) == tuple(ws)
    assert d['W'] == W and d['l'] == 2 and d['openpts'] == ()
    R = RelaxStar(g, W, g[:k], ws)
    fi = {qq: i for i, qq in enumerate(layout['free_gears'])}
    assert [[sum(1 << fi[qq] for qq in S), list(r)]
            for (S, r, _O) in R.cols] == layout['cols'], "column layout differs"
    assert [[p, list(kd)] for (p, kd) in R.links] == layout['links'], \
        "link layout differs"
    assert all(v.denominator == 1 for v in R.frow), "recursion row not integral"
    rows = [[int(i), [fr(v) for v in lam]] for (i, lam) in d['rows']]
    base = base_cut(R.n, 2)
    out = dict(
        schema='lp-case-split-certificate/1',
        rung='inc_%d->%d' % (M, q), machine=y, W=W, l=2,
        increment_step=[M, q], F2_of_M=F2[M], s_min=smin(q),
        W_is=("W_inc = F_2(%d) + s_min(%d) = %d + %d"
              % (M, q, F2[M], smin(q))),
        full_gears=list(g), held_gears=list(g[:k]), ws=list(ws),
        free_gears=list(layout['free_gears']), n=R.n,
        atoms=list(layout['atoms']), atom_gears=layout['atom_gears'],
        block_span=layout['block_span'],
        n_cols=len(R.cols), n_links=len(R.links),
        pos=[int(i) for i in R.pos], n_pos=len(R.pos),
        frow=[int(v) for v in R.frow], frhs=fr(R.frhs),
        rows=rows,
        rows_all_base_cut=all(tuple(lam) == base for (_i, lam) in d['rows']),
        y=[fr(v) for v in d['y']], yff=fr(d['yff']),
        nu=[fr(v) for v in d['nu']],
        lhs=fr(d['info']['lhs']), rhs=fr(d['info']['rhs']),
        ops=int(d['info']['ops']), iterations=int(d['info']['its']),
        source_pickle=os.path.basename(src),
    )
    p = os.path.join(R28, 'cert_inc_%s_h%s.json'
                     % (stepname(M, q), "_".join(str(x) for x in ws)))
    with open(p, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    return p, out


def emit_step(M, q):
    y, k, W = q, KOF[(M, q)], w_inc(M, q)
    g = gears_of(y)
    held = g[:k]
    st = stepname(M, q)
    layout = layout_of(y, k)
    lp = os.path.join(R28, 'layout_inc_%s.json' % st)
    with open(lp, 'w') as fh:
        json.dump(layout, fh, separators=(',', ':'))
    cases = [tuple(ws) for ws in product(*[range(p) for p in held])]
    files = []
    t0 = time.time()
    for ws in cases:
        p, _o = emit_case(M, q, ws, layout)
        files.append(os.path.basename(p))
    man = dict(
        schema='lp-case-split-manifest/1',
        rung='inc_%d->%d' % (M, q), machine=y, W=W,
        increment_step=[M, q], F2_of_M=F2[M], s_min=smin(q),
        full_gears=list(g), held_gears=list(held), free_gears=list(g[k:]),
        n_cases=len(cases), held_phase_tuples=[list(w) for w in cases],
        exhaustiveness=("held_phase_tuples is exactly the cartesian product "
                        "prod_{q in held_gears} Z_q, in itertools.product "
                        "order; every configuration of the machine has its "
                        "held gears at exactly one of these tuples"),
        exhaustiveness_holds=(len(cases) == _prod(held)),
        case_files=files, layout_file=os.path.basename(lp),
        witness_file='witness_inc_%s.json' % st,
        claim=("F(%d) <= %d = F_2(%d) + s_min(%d): every case carries an exact "
               "rational dual certificate, so machine %d has no fully blocked "
               "window of width %d.  With the witness file's lower half this "
               "is the increment law at the step %d -> %d."
               % (y, W, M, q, y, W, M, q)),
    )
    mp = os.path.join(R28, 'manifest_inc_%s.json' % st)
    with open(mp, 'w') as fh:
        json.dump(man, fh, indent=1)
    # ---------------------------------------------------------- the witness
    w = witness_f2(M)
    assert w is not None and check_witness(w) and w['span'] == F2[M]
    wj = dict(
        schema='lp-realisability-witness/1',
        machine=M, increment_step=[M, q], F2_claim=F2[M],
        gears=list(w['gears']), phases=list(w['phases']),
        teeth={str(qq): list(__import__('lp_degree_range').teeth(qq))
               for qq in w['gears']},
        span=w['span'], openings=list(w['openings']), split=list(w['split']),
        recipe=("blocked = union over gears q of {i in [0, span] : "
                "(i + phases[q]) mod q in teeth[q]}; the open positions of "
                "[0, span] are exactly `openings` = [0, a, span], so machine "
                "%d realises the adjacent gap pair (%d, %d) whose sum is "
                "F_2(%d) = %d.  No period scan is used or needed."
               % (M, w['split'][0], w['split'][1], M, F2[M])),
    )
    with open(os.path.join(R28, 'witness_inc_%s.json' % st), 'w') as fh:
        json.dump(wj, fh, indent=1)
    sz = sum(os.path.getsize(os.path.join(R28, f)) for f in files)
    print("  inc %2d->%-2d  W_inc=%-3d k=%d  %3d case files (%.1f KB, %.1f KB "
          "each), layout %.1f KB, witness split %s  [%.0fs]"
          % (M, q, W, k, len(files), sz / 1024.0, sz / 1024.0 / len(files),
             os.path.getsize(lp) / 1024.0, tuple(w['split']),
             time.time() - t0), flush=True)
    return man


def check_witness_json(path):
    with open(path) as fh:
        J = json.load(fh)
    g, r, s = J['gears'], J['phases'], J['span']
    assert tuple(g) == gears_of(J['machine']), "gear list is not the machine's"
    blocked = set()
    for q, rq in zip(g, r):
        blocked |= set(hits(q, rq, s + 1))
    openp = sorted(set(range(s + 1)) - blocked)
    assert openp == list(J['openings']), (openp, J['openings'])
    assert len(openp) == 3 and openp[0] == 0 and openp[-1] == s
    assert openp[1] - openp[0] == J['split'][0]
    assert openp[2] - openp[1] == J['split'][1]
    assert sum(J['split']) == J['F2_claim'] == s
    return J


def gate():
    t0 = time.time()
    print("=" * 78)
    print("GATE  round-28 increment-width emission, re-verified from disk")
    print("=" * 78, flush=True)
    ncase = 0
    for (M, q) in STEPS:
        st = stepname(M, q)
        with open(os.path.join(R28, 'manifest_inc_%s.json' % st)) as fh:
            man = json.load(fh)
        held = man['held_gears']
        cases = [tuple(w) for w in man['held_phase_tuples']]
        assert cases == [tuple(w) for w in product(*[range(p) for p in held])]
        assert len(cases) == _prod(held) == man['n_cases']
        assert man['exhaustiveness_holds'] is True
        assert man['W'] == man['F2_of_M'] + man['s_min'], \
            "W is not the increment width"
        vals = []
        for ws in cases:
            p = os.path.join(R28, 'cert_inc_%s_h%s.json'
                             % (st, "_".join(str(x) for x in ws)))
            assert os.path.basename(p) in man['case_files'], p
            lhs, rhs = check_case_json(p)
            vals.append(rhs - lhs)
            ncase += 1
        J = check_witness_json(os.path.join(R28, 'witness_inc_%s.json' % st))
        print("  inc %2d->%-2d  W_inc = %d + %d = %d : %d/%d cases re-verified"
              " (min margin %s); witness F_2(%d) >= %d split %s  GREEN"
              % (M, q, man['F2_of_M'], man['s_min'], man['W'], len(vals),
                 len(cases), min(vals), M, J['F2_claim'], tuple(J['split'])),
              flush=True)
    print("\n  %d case certificates + 6 witnesses re-verified from JSON alone"
          % ncase)
    print("  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0))


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'EMIT')
    os.makedirs(R28, exist_ok=True)
    E27.R27 = R28                      # check_case_json reads layouts here
    if cmd == 'EMIT':
        for (M, q) in STEPS:
            emit_step(M, q)
    elif cmd == 'GATE':
        gate()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

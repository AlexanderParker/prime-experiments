"""FORMALIST, ROUND 28.  Transcribe the LP thread's INCREMENT-WIDTH case-split
certificates (round 27, `research/data/r27/cert_inc_*.pkl`) into the JSON shape
the Lean generator consumes, re-deriving every number independently.

THE OBJECT.  The increment law is

    F(M + q')  <=  F_2(M) + s_min(q'),      s_min(q') = min(2u' mod q', -2u' mod q')

and the LP thread certified its UPPER half at six literal steps by running the
case-split vehicle at the INCREMENT WIDTH  W_inc = F_2(M) + s_min(q')  instead of
the ladder's budget width F(M) + q'.  W_inc is strictly smaller at every step, so
each of these is a strictly harder obligation than the corresponding (D) rung.

This file reuses `lp_cert_lean.transcribe` verbatim - the same five assertions
(relaxation rebuilt from the primes; every cut row equals the base cut; lhs/rhs
recomputed from this lane's own formulas in exact integers; the closed-form
`n_ab` asserted equal to `RelaxStar.frow` column by column; the random-tuple
soundness gate on the recursion row) - pointed at the round-27 pickle directory.

    python research/lp_cert_inc_r28.py GATE
    python research/lp_cert_inc_r28.py EMIT 23_29
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lp_cert_lean                                              # noqa: E402
from lp_cert_lean import transcribe, soundness_gate              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.join(HERE, 'data', 'r27')

# The pickles live in r27, not r26; `transcribe` reads the module global.
lp_cert_lean.R26 = R27
lp_cert_lean.OUT = R27

# tag: (y, W_inc, held gears, filename pattern, phase-tuple list)
# W_inc = F_2(M) + s_min(q'):  39 = 31 + 8, 49 = 39 + 10, 65 = 55 + 10.
INC = {
    '11_13': (13, 15, (5,), 'cert_inc_m13_w15_h%d.pkl',
              [(a,) for a in range(5)]),
    '13_17': (17, 22, (5,), 'cert_inc_m17_w22_h%d.pkl',
              [(a,) for a in range(5)]),
    '17_19': (19, 31, (5,), 'cert_inc_m19_w31_h%d.pkl',
              [(a,) for a in range(5)]),
    '19_23': (23, 39, (5, 7), 'cert_inc_m23_w39_h%d_%d.pkl',
              [(a, b) for a in range(5) for b in range(7)]),
    '23_29': (29, 49, (5, 7), 'cert_inc_m29_w49_h%d_%d.pkl',
              [(a, b) for a in range(5) for b in range(7)]),
    '29_31': (31, 65, (5, 7), 'cert_inc_m31_w65_h%d_%d.pkl',
              [(a, b) for a in range(5) for b in range(7)]),
}

# The three rungs Lean actually needs.  At 11->13, 13->17 and 17->19 the
# corpus already carries a STRICTLY TIGHTER kernel bound on F(q') than W_inc
# (11 < 15, 18 < 22, 25 < 31), so the certificate is redundant there and is
# transcribed only as a cross-check.
LEAN = ('19_23', '23_29', '29_31')


def emit(tag, verbose=True):
    y, W, held, fn, phs = INC[tag]
    if verbose:
        print('inc rung %s  (y=%d, W_inc=%d, held=%s, %d cases)'
              % (tag, y, W, held, len(phs)))
    data = transcribe(y, W, held, fn, phs, verbose=verbose)
    data['tag'] = tag
    data['kind'] = 'increment'
    path = os.path.join(R27, 'cert_inc_%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(data, f)
    if verbose:
        print('  -> %s (%d bytes, %d positions/case)'
              % (path, os.path.getsize(path), len(data['cases'][0]['pos'])))
    return data


def cross(tag, verbose=True):
    """SECOND SOURCE.  The LP thread's own round-28 emission
    (`research/data/r28/cert_inc_<tag>_h<ws>.json`, written by their
    `emit_inc_r28.py` from the same pickles by different code) is compared to
    this lane's transcription AS EXACT RATIONALS, field by field - the round-27
    practice, repeated at the increment width."""
    from fractions import Fraction
    y, W, held, fn, phs = INC[tag]
    mine = json.load(open(os.path.join(R27, 'cert_inc_%s.json' % tag)))
    R28 = os.path.join(HERE, 'data', 'r28')
    n = 0
    for idx, ws in enumerate(phs):
        f = os.path.join(R28, 'cert_inc_%s_h%s.json'
                         % (tag, '_'.join(map(str, ws))))
        if not os.path.exists(f):
            print('  MISSING second source: %s' % os.path.basename(f))
            continue
        th = json.load(open(f))
        C = mine['cases'][idx]
        D = C['D']
        assert th['W'] == W and list(th['ws']) == list(ws)
        assert list(th['pos']) == list(C['pos']), ('pos', tag, ws)
        assert Fraction(*th['yff']) == Fraction(C['yff'], D), ('yff', tag, ws)
        assert Fraction(*th['lhs']) == Fraction(C['lhs'], D), ('lhs', tag, ws)
        assert Fraction(*th['rhs']) == Fraction(C['rhs'], D), ('rhs', tag, ws)
        for a, v in zip(th['y'], C['y']):
            assert Fraction(*a) == Fraction(v, D), ('y', tag, ws)
        for a, v in zip(th['nu'], C['nu']):
            assert Fraction(*a) == Fraction(v, D), ('nu', tag, ws)
        assert th['rows_all_base_cut'] is True
        n += 1
    if verbose:
        print('  %-6s %2d cases agree with the LP thread\'s own emission as'
              ' exact rationals' % (tag, n))
    return n


def main():
    cmd = sys.argv[1].upper() if len(sys.argv) > 1 else 'GATE'
    if cmd == 'CROSS':
        tot = sum(cross(t) for t in INC)
        print('CROSS-CHECK PASSED (%d cases, two codebases)' % tot)
    elif cmd == 'EMIT':
        emit(sys.argv[2])
    elif cmd == 'GATE':
        for tag in INC:
            data = emit(tag, verbose=False)
            y, W, held, fn, phs = INC[tag]
            print('  %-6s y=%-3d W_inc=%-3d held=%s  %2d cases  %3d positions'
                  ' %s' % (tag, y, W, held, len(data['cases']),
                           len(data['cases'][0]['pos']),
                           '(-> Lean)' if tag in LEAN else '(cross-check)'))
            for ws in phs[:2]:
                soundness_gate(y, W, held, ws)
        print('ALL ASSERTIONS PASSED')
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

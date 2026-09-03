"""FORMALIST, ROUND 29.  The 31 -> 37 case-split rung, transcribed and
independently re-derived.

The LP thread's round-29 emission (`research/data/r29/manifest_31_37.json`,
`layout_31_37.json`, 385 x `cert_31_37_h*_*_*.json`) announces

    F(37) <= 95 = F(31) + 37 : (D) at 31 -> 37, hypothesis-free,
    k = 3, held gears {5,7,11}, free gears {13,17,19,23,29,31,37},
    385 cases, rows_all_base_cut = True, iterations_max = 0.

The certificates themselves are the round-26 pickles
`research/data/r26/cert_rung3_m37_w95_h<a>_<b>_<c>.pkl`, which is what the
emission's `source_pickle` field names.  This driver reuses round 27's
`lp_cert_lean.transcribe` VERBATIM - so the 31 -> 37 rung is re-derived by
exactly the machinery that produced the 19->23, 23->29 and 29->31 rungs - and
adds a CROSS check against the round-29 JSON emission as exact rationals.

Everything is rebuilt from the primes: `RelaxStar` from the gear list, the
position set from the held phases, every cut row asserted equal to the base
cut, the recursion-row coefficients recomputed from the closed form the kernel
uses, and lhs/rhs recomputed in exact integers after scaling.

    uv run python research/lp_cert_case_r29.py ONE      # time a single case
    uv run python research/lp_cert_case_r29.py GATE     # all 385, + CROSS
    uv run python research/lp_cert_case_r29.py EMIT     # write cert_31_37.json
"""
import itertools
import json
import os
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lp_cert_lean as L                                        # noqa: E402

R29 = os.path.join(HERE, 'data', 'r29')
OUT = os.path.join(HERE, 'data', 'r29')

Y, W, HELD = 37, 95, (5, 7, 11)
PATTERN = 'cert_rung3_m37_w95_h%d_%d_%d.pkl'
PHASES = [(a, b, c) for a in range(5) for b in range(7) for c in range(11)]
TAG = '31_37'


def emission_case(ws):
    p = os.path.join(R29, 'cert_%s_h%d_%d_%d.json' % ((TAG,) + ws))
    with open(p) as f:
        return json.load(f)


def fr(v):
    return Fraction(v[0], v[1])


def cross(case, ws):
    """Compare my transcription against the LP thread's own JSON, exactly."""
    E = emission_case(ws)
    assert E['machine'] == Y and E['W'] == W, (E['machine'], E['W'])
    assert tuple(E['held_gears']) == HELD and tuple(E['ws']) == ws
    assert E['pos'] == case['pos'], 'pos mismatch'
    assert E['rows_all_base_cut'] is True
    D = case['D']
    assert [fr(v) for v in E['y']] == [Fraction(v, D) for v in case['y']], 'y'
    assert [fr(v) for v in E['nu']] == [Fraction(v, D) for v in case['nu']], 'nu'
    assert fr(E['yff']) == Fraction(case['yff'], D), 'yff'
    assert fr(E['lhs']) == Fraction(case['lhs'], D), 'lhs'
    assert fr(E['rhs']) == Fraction(case['rhs'], D), 'rhs'
    assert fr(E['margin']) == Fraction(case['rhs'] - case['lhs'], D), 'margin'
    return fr(E['margin'])


def run(phases, do_cross=True, verbose=True):
    t0 = time.time()
    data = L.transcribe(Y, W, HELD, PATTERN, phases, verbose=verbose)
    margins = []
    if do_cross:
        for i, ws in enumerate(phases):
            margins.append(cross(data['cases'][i], ws))
    return data, margins, time.time() - t0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'GATE'
    if cmd == 'ONE':
        data, margins, dt = run(PHASES[:1])
        print('one case in %.1f s; margin %s' % (dt, margins[0]))
        return
    if cmd in ('GATE', 'EMIT'):
        data, margins, dt = run(PHASES, verbose=(cmd == 'GATE'))
        assert len(data['cases']) == 385
        # exhaustiveness: the phase list IS the full cartesian product
        assert [tuple(c['ws']) for c in data['cases']] == \
            [tuple(t) for t in itertools.product(range(5), range(7), range(11))]
        print('385 cases transcribed and cross-checked in %.0f s' % dt)
        print('margin min %s  max %s' % (min(margins), max(margins)))
        n = L.soundness_gate(Y, W, HELD, PHASES[0])
        print('soundness gate at ws=%s: %d random tuples OK' % (PHASES[0], n))
        path = os.path.join(OUT, 'cert_%s.json' % TAG)
        with open(path, 'w') as f:
            json.dump(data, f)
        print('wrote %s (%d bytes)' % (path, os.path.getsize(path)))
        print('ALL ASSERTIONS PASSED')
        return
    print(__doc__)


if __name__ == '__main__':
    main()

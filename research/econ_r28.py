"""FORMALIST, ROUND 28, ITEM 3 - CASE-COUNT ECONOMICS.

Round-27 verdict 29: the case split's kernel limit is the CASE COUNT (k = 3 is
385 cases, ~8.6 h; k = 4 is 5,005 and out of reach).  Before any k = 3 rung is
attempted this round asks a narrower question: does a case module pay a FIXED
cost that batching several cases into one module would remove?

Every case module is its own lake target, so it is its own `lean.exe` process
and re-loads the whole mathlib import closure.  That cost is identical for all
of them and is paid `ncases` times.  This script builds three scratch modules -

    Econ0<fam>   imports only, no declarations         -> T0, the fixed cost
    Econ1<fam>   exactly one case body                 -> T1
    Econ5<fam>   five case bodies concatenated         -> T5

- from the ALREADY-GENERATED case files, so the bodies are byte-identical to
the ones in the ledger.  If T5 is near T0 + 5 (T1 - T0) the marginal cost is
additive and batching saves (B - 1) T0 per batch of B.

    python research/econ_r28.py 31        # write the scratch modules
    (then, SOLO, from proofs/: lake env lean Econ0_31.lean etc.)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.join(os.path.dirname(HERE), 'proofs')


def body(fam, ci):
    """The case body of proofs/IncCert<fam>C<ci>.lean, imports and namespace
    lines stripped (the mega-dry-file recipe: strip by CONTENT, never by line
    offsets - round-22's lesson)."""
    src = open(os.path.join(PROOFS, 'IncCert%dC%d.lean' % (fam, ci)),
               encoding='utf-8').read()
    lines = src.split('\n')
    # drop the block comment header
    i = lines.index('-/') + 1
    out = [ln for ln in lines[i:]
           if not ln.startswith('import ') and not ln.startswith('namespace ')
           and not ln.startswith('end ')]
    return '\n'.join(out)


def write(fam):
    hdr = ('/- SCRATCH (round 28, item 3): elaboration-cost measurement.\n'
           '   Bodies copied verbatim from the generated case modules.\n'
           '   DELETE after measuring. -/\nimport IncCert%dB\n\n'
           'namespace Econ%d\nopen IncCert%d\n' % (fam, fam, fam))
    for tag, cis in (('0', []), ('1', [0]), ('5', [0, 1, 2, 3, 4])):
        txt = hdr + '\n'.join(body(fam, ci) for ci in cis) + \
            '\nend Econ%d\n' % fam
        path = os.path.join(PROOFS, 'Econ%s_%d.lean' % (tag, fam))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(txt)
        print('wrote %s (%d lines)' % (path, txt.count('\n')))


if __name__ == '__main__':
    write(int(sys.argv[1]) if len(sys.argv) > 1 else 31)

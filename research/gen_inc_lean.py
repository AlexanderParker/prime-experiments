"""FORMALIST, ROUND 28.  Emit the Lean transcription of an INCREMENT-WIDTH
case-split rung (namespace `IncCert<y>`), from
`research/data/r27/cert_inc_<tag>.json` (written and independently re-derived by
`research/lp_cert_inc_r28.py`).

Identical machinery to `research/gen_case_lean.py` - the per-case module body is
that file's `gen_case` verbatim, so the two rung families share one soundness
skeleton - with a different namespace and a root whose headline theorem is the
INCREMENT LAW's upper half, `F(q') <= F_2(M) + s_min(q')`, rather than the
ladder's `F(q') <= F(M) + q'`.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_case_lean import gear_defs, gen_case                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.join(HERE, 'data', 'r27')
PROOFS = os.path.join(os.path.dirname(HERE), 'proofs')

# tag -> (old machine M, s_min(q'), F_2(M))
STEP = {'19_23': (19, 8, 31), '23_29': (23, 10, 39), '29_31': (29, 10, 55)}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else '19_23'
    M, smin, F2 = STEP[tag]
    with open(os.path.join(R27, 'cert_inc_%s.json' % tag)) as f:
        D = json.load(f)
    y, W, held, free = D['y'], D['W'], D['held'], D['free']
    assert W == F2 + smin, (W, F2, smin)
    NS = 'IncCert%d' % y
    ncase = len(D['cases'])
    names = []
    allg = sorted(set(held + free))

    # five free gears needs `CaseSplit.lowest5`, which lives in its own module
    # (see the header of proofs/CaseSplit5.lean).
    csmod = 'CaseSplit5' if len(free) == 5 else 'CaseSplit'
    B = ['/-',
         'INCREMENT-WIDTH CASE-SPLIT CERTIFICATES, step %s: the gears as'
         % tag.replace('_', '->'),
         'blocking predicates in (phase, offset) coordinates.  Gear q has teeth',
         'at the slot residues u and q - u with 6u = q -+ 1 (GENERATED).',
         '-/',
         'import %s' % csmod,
         '',
         'namespace %s' % NS,
         ''] + gear_defs(allg) + ['', 'end %s' % NS, '']
    with open(os.path.join(PROOFS, '%sB.lean' % NS), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(B))
    names.append('%sB' % NS)
    print('wrote proofs/%sB.lean' % NS)

    for ci in range(ncase):
        L = ['/-',
             'INCREMENT-WIDTH CERTIFICATE, step %s, case %d of %d (GENERATED'
             % (tag.replace('_', '->'), ci, ncase),
             'by research/gen_inc_lean.py from',
             'research/data/r27/cert_inc_%s.json, which re-derives every number'
             % tag,
             'from the primes alone).',
             '',
             'Machine %d, INCREMENT width %d = F_2(%d) + s_min(%d) = %d + %d,'
             % (y, W, M, y, F2, smin),
             'held gears %s at phases %s.  Free gears %s.'
             % (held, D['cases'][ci]['ws'], free),
             'All numbers are the LP thread\'s exact rational dual scaled by the',
             'case denominator %d.' % D['cases'][ci]['D'],
             '-/',
             'import %sB' % NS,
             '',
             'namespace %s' % NS,
             '']
        L += gen_case(D, ci)
        L += ['end %s' % NS, '']
        name = '%sC%d' % (NS, ci)
        with open(os.path.join(PROOFS, name + '.lean'), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(L))
        names.append(name)
    print('wrote %d case modules' % ncase)

    # ------------------------------------------------------------- the root
    orall = ' || '.join('gb%d (p %% %d) i' % (q, q) for q in allg)
    R = ['/-',
         'THE INCREMENT LAW AT %s, UPPER HALF, BY CASE-SPLIT LP DUALITY'
         % tag.replace('_', '->'),
         '(GENERATED root).',
         '',
         'The increment law (manager, round 26) is',
         '',
         '    F(M + q\')  <=  F_2(M) + s_min(q\'),   s_min = min(2u\' mod q\','
         ' -2u\' mod q\').',
         '',
         'Here M = %d, q\' = %d, s_min = %d and F_2(%d) = %d, so the width to'
         % (M, y, smin, M, F2),
         'certify is W_inc = %d.  This is STRICTLY SMALLER than the (D) ladder\'s'
         % W,
         'budget width F(%d) + %d, so it is a strictly harder obligation and is'
         % (M, y),
         'NOT implied by the corresponding rung.',
         '',
         'Every configuration of machine %d has its held gears %s at exactly one'
         % (y, held),
         'of the %d phase tuples, and each of those cases carries an exact dual'
         % ncase,
         'certificate of the restricted level-2 covering relaxation.  So no window',
         'of %d consecutive slots of machine %d is fully blocked.' % (W, y),
         '',
         'NO CENSUS HYPOTHESIS, NO PERIOD SCAN: the only inputs are the primes',
         'up to %d and %d integers per case.'
         % (y, len(D['cases'][0]['y']) + len(D['cases'][0]['nu']) + 1),
         '-/']
    for ci in range(ncase):
        R.append('import %sC%d' % (NS, ci))
    R += ['import Machine%d' % y, '', 'namespace %s' % NS, '']
    R.append('set_option maxHeartbeats 4000000')
    R.append('')
    R.append('/-- A slot that is not an opening of machine %d is blocked by one'
             % y)
    R.append('of its gears, in the certificate\'s (phase, offset) '
             'coordinates. -/')
    R.append('theorem blocked {p i : ℕ} (hp : 1 ≤ p) '
             '(h : ¬ Machine%d.Exposed%d (p + i)) :' % (y, y))
    R.append('    (%s) = true := by' % orall)
    for q in allg:
        R.append('  have e%d : (p %% %d + i) %% %d = (p + i) %% %d := by omega'
                 % (q, q, q, q))
    R.append('  simp only [%s, %s, Bool.or_eq_true, beq_iff_eq]'
             % (', '.join('gb%d' % q for q in allg),
                ', '.join('e%d' % q for q in allg)))
    R.append('  by_contra hcon')
    R.append('  push Not at hcon')
    R.append('  apply h')
    chain = [q for q in allg if q > 19]
    expr = '?_'
    for q in chain:
        expr = ('Machine%d.exposed%d_of (show 1 ≤ p + i by omega) (%s) ?_'
                % (q, q, expr))
    R.append('  refine %s' % expr)
    R.append('  · rw [Machine19.exposed19_iff (show 1 ≤ p + i by omega)]')
    R.append('    simp only [Machine19.expT, Bool.and_eq_true, bne_iff_ne, '
             'ne_eq, and_assoc]')
    R.append('    tauto')
    for q in chain:
        R.append('  · unfold Machine%d.Killed%d' % (q, q))
        R.append('    omega')
    R.append('')
    for ci in range(ncase):
        C = D['cases'][ci]
        eqs = ' '.join('(e%d : p %% %d = %d)' % (q, q, C['ws'][hi])
                       for hi, q in enumerate(held))
        R.append('theorem nocase%d {p : ℕ} %s' % (ci, eqs))
        R.append('    (hall : ∀ i, i < %d → (%s) = true) : False := by'
                 % (W, orall))
        args = ' '.join('(r%d := p %% %d)' % (a, free[a])
                        for a in range(len(free)))
        R.append('  refine nocov%d %s %s ?_'
                 % (ci, args, ' '.join('(by omega)' for _ in free)))
        R.append('  intro t ht')
        R.append('  have h3 := hall (q%d t) (plt%d t ht)' % (ci, ci))
        R.append('  rw [%s] at h3' % ', '.join('e%d' % q for q in held))
        R.append('  simp only [%s, Bool.false_or] at h3'
                 % ', '.join('pfree%d_%d t ht' % (ci, q) for q in held))
        R.append('  simpa only [%s] using h3'
                 % ', '.join('c%d_%d' % (ci, a) for a in range(len(free))))
        R.append('')
    R.append('/-- **`F(%d) <= %d` by the case split at the INCREMENT width**:'
             % (y, W))
    R.append('every window of %d consecutive slots contains an opening of' % W)
    R.append('machine %d. -/' % y)
    R.append('theorem no_run {p : ℕ} (hp : 1 ≤ p) :')
    R.append('    ∃ i < %d, Machine%d.Exposed%d (p + i) := by' % (W, y, y))
    R.append('  by_contra hc')
    R.append('  push Not at hc')
    R.append('  have hall : ∀ i, i < %d → (%s) = true :=' % (W, orall))
    R.append('    fun i hi => blocked hp (hc i hi)')
    for q in held:
        R.append('  have d%d : %s := by omega'
                 % (q, ' ∨ '.join('p %% %d = %d' % (q, v) for v in range(q))))
    lines = []

    def rec(hi, indent, ws):
        q = held[hi]
        lines.append(indent + 'rcases d%d with %s'
                     % (q, ' | '.join('e%d' % q for _ in range(q))))
        for v in range(q):
            if hi + 1 == len(held):
                ci = [k for k, C in enumerate(D['cases'])
                      if C['ws'] == ws + [v]][0]
                lines.append(indent + '· exact nocase%d %s hall'
                             % (ci, ' '.join('e%d' % qq for qq in held)))
            else:
                lines.append(indent + '· skip')
                rec(hi + 1, indent + '  ', ws + [v])
    rec(0, '  ', [])
    R += lines
    R.append('')
    R.append('theorem F_le (n : ℕ) : Machine%d.g%d n ≤ %d := by' % (y, y, W))
    R.append('  by_contra hcon')
    R.append('  obtain ⟨i, hi, hE⟩ := no_run (p := Machine%d.opSeq%d n + 1)'
             % (y, y))
    R.append('    (by have := Machine%d.opSeq%d_pos n; omega)' % (y, y))
    R.append('  have hgap : Machine%d.g%d n = Machine%d.opSeq%d (n + 1) '
             '- Machine%d.opSeq%d n := rfl' % (y, y, y, y, y, y))
    R.append('  have hlt := Machine%d.opSeq%d_lt_succ n' % (y, y))
    R.append('  exact Machine%d.opSeq%d_gap_empty n '
             '(Machine%d.opSeq%d n + 1 + i)' % (y, y, y, y))
    R.append('    (by omega) (by omega) hE')
    R.append('')
    R.append('/-- **THE INCREMENT LAW\'S UPPER HALF AT %s**: every gap of'
             % tag.replace('_', '->'))
    R.append('machine %d is at most `F_2(%d) + s_min(%d) = %d + %d = %d`.'
             % (y, M, y, F2, smin, W))
    R.append('No census hypothesis, no period scan - only the primes up to %d'
             % y)
    R.append('and the %d case certificates.  The matching LOWER half' % ncase)
    R.append('(`F_2(%d) >= %d`, a realisability statement no dual certificate' % (M, F2))
    R.append('can carry) is `Increment.f2_%d_ge`. -/' % M)
    R.append('theorem inc_%d_%d (n : ℕ) : Machine%d.g%d n ≤ %d + %d :='
             % (M, y, y, y, F2, smin))
    R.append('  F_le n')
    R.append('')
    R.append('end %s' % NS)
    R.append('')
    with open(os.path.join(PROOFS, '%s.lean' % NS), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(R))
    names.append(NS)
    print('wrote proofs/%s.lean  (%d cases)' % (NS, ncase))
    print('lakefile targets:', ', '.join('"%s"' % n for n in names))


if __name__ == '__main__':
    main()

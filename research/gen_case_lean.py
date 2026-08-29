"""FORMALIST, ROUND 27.  Emit the Lean transcription of a case-split rung.

Input: `research/data/r27/cert_<tag>.json` (written and independently verified
by `research/lp_cert_lean.py`).  Output: one `proofs/CaseCert*.lean` module per
case plus a root module carrying the rung.

The Lean side is generated, not hand-typed, because the only hand work worth
doing is the SOUNDNESS (in `proofs/CaseSplit.lean` and in the proof skeleton
below); the certificate itself is ~1,100 small integers per case and copying
those by hand is how transcription bugs happen.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.join(HERE, 'data', 'r27')
PROOFS = os.path.join(os.path.dirname(HERE), 'proofs')

TEETH = {5: (1, 4), 7: (6, 1), 11: (2, 9), 13: (11, 2), 17: (3, 14),
         19: (16, 3), 23: (4, 19), 29: (24, 5), 31: (26, 5), 37: (6, 31),
         41: (7, 34), 43: (36, 7)}


def zl(v):
    return '(%d)' % v if v < 0 else str(v)


def gear_defs(gears):
    out = []
    for q in gears:
        t0, t1 = TEETH[q]
        out.append('def gb%d (r i : ℕ) : Bool := '
                   '((r + i) %% %d == %d) || ((r + i) %% %d == %d)'
                   % (q, q, t0, q, t1))
    return out


def gen_case(D, ci):
    """Lean source for one case (as a list of lines), plus the case's own
    `nocov` theorem."""
    free = D['free']
    m = len(free)
    pairs = [tuple(p) for p in D['pairs']]
    C = D['cases'][ci]
    pos, y, nu, yff, base = C['pos'], C['y'], C['nu'], C['yff'], C['base']
    n = len(pos)
    L = []
    A = L.append
    A('/-! ### case %d: held gears at phases %s -/' % (ci, C['ws']))
    A('')
    A('def p%d : List ℕ := [%s]' % (ci, ', '.join(map(str, pos))))
    A('def q%d (t : ℕ) : ℕ := p%d.getD t 0' % (ci, ci))
    A('def n%d : ℕ := %d' % (ci, n))
    A('def yl%d : List ℤ := [%s]' % (ci, ', '.join(zl(v) for v in y)))
    A('def w%d (t : ℕ) : ℤ := yl%d.getD t 0' % (ci, ci))
    A('def ul%d : List ℤ := [%s]' % (ci, ', '.join(zl(v) for v in nu)))
    A('def u%d (k : ℕ) : ℤ := ul%d.getD k 0' % (ci, ci))
    A('')
    # indicators
    for a, q in enumerate(free):
        A('def c%d_%d (r t : ℕ) : Bool := gb%d r (q%d t)' % (ci, a, q, ci))
    A('')
    # weighted single sums
    for a, q in enumerate(free):
        A('def S%d_%d (r : ℕ) : ℤ := ∑ t ∈ Finset.range n%d, '
          '(w%d t + %d) * (if c%d_%d r t then 1 else 0)'
          % (ci, a, ci, ci, yff, ci, a))
    A('')
    # link sums per singleton
    for a, q in enumerate(free):
        terms = []
        for pi, (x, b) in enumerate(pairs):
            if x == a:
                terms.append('u%d (%d + r)' % (ci, base[pi] + free[b]))
            elif b == a:
                terms.append('u%d (%d + r)' % (ci, base[pi]))
        A('def L%d_%d (r : ℕ) : ℤ := %s' % (ci, a, ' + '.join(terms)))
    A('')
    for a, q in enumerate(free):
        A('def aS%d_%d (r : ℕ) : ℤ := S%d_%d r - L%d_%d r' % (ci, a, ci, a, ci, a))
        A('def MS%d_%d : ℤ := CaseSplit.mxr (aS%d_%d) %d'
          % (ci, a, ci, a, q - 1))
    A('')
    # pair coefficients
    for pi, (a, b) in enumerate(pairs):
        qa, qb = free[a], free[b]
        if a == 0:
            A('def N%d_%d (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n%d, '
              '(if c%d_%d ra t && c%d_%d rb t then 1 else 0)'
              % (ci, pi, ci, ci, a, ci, b))
        elif a == 1:
            A('def P%d_%d (ra rb : ℕ) : ℤ := ∑ t ∈ Finset.range n%d, '
              '(if c%d_%d ra t && c%d_%d rb t then 1 else 0)'
              % (ci, pi, ci, ci, a, ci, b))
            A('def C%d_%d (ra rb s : ℕ) : ℤ := ∑ t ∈ Finset.range n%d, '
              '(if c%d_%d ra t && c%d_%d rb t && c%d_0 s t then 1 else 0)'
              % (ci, pi, ci, ci, a, ci, b, ci))
            A('def M%d_%d (ra rb : ℕ) : ℤ := CaseSplit.mxr (C%d_%d ra rb) %d'
              % (ci, pi, ci, pi, free[0] - 1))
            A('def E%d_%d : List ℕ := [%s]'
              % (ci, pi, ', '.join(map(str, C['exc'][str(pi)]))))
            A('def N%d_%d (ra rb : ℕ) : ℤ := '
              'if E%d_%d.contains (ra * %d + rb) = true '
              'then P%d_%d ra rb - M%d_%d ra rb else 0'
              % (ci, pi, ci, pi, qb, ci, pi, ci, pi))
        else:
            A('def N%d_%d (_ra _rb : ℕ) : ℤ := 0' % (ci, pi))
        A('def aP%d_%d (ra rb : ℕ) : ℤ := -(%d) * N%d_%d ra rb '
          '+ u%d (%d + rb) + u%d (%d + ra)'
          % (ci, pi, yff, ci, pi, ci, base[pi], ci, base[pi] + qb))
        A('def MP%d_%d : ℤ := CaseSplit.mxr2 (aP%d_%d) %d %d'
          % (ci, pi, ci, pi, qa - 1, qb - 1))
    A('')
    A('def rhs%d : ℤ := (∑ t ∈ Finset.range n%d, w%d t) + %d * (n%d : ℤ)'
      % (ci, ci, ci, yff, ci))
    A('')
    # the finite checks
    A('set_option maxRecDepth 40000')
    A('set_option maxHeartbeats 4000000')
    A('')
    A('theorem wnn%d : ∀ t, t < n%d → (0 : ℤ) ≤ w%d t := by decide'
      % (ci, ci, ci))
    A('theorem plt%d : ∀ t, t < n%d → q%d t < %d := by decide'
      % (ci, ci, ci, D['W']))
    for hi, hq in enumerate(D['held']):
        A('theorem pfree%d_%d : ∀ t, t < n%d → gb%d %d (q%d t) = false := by '
          'decide' % (ci, hq, ci, hq, C['ws'][hi], ci))
    for a in range(m):
        A('theorem MSv%d_%d : MS%d_%d = %s := by decide +kernel'
          % (ci, a, ci, a, zl(C['MS'][a])))
    for pi in range(len(pairs)):
        A('theorem MPv%d_%d : MP%d_%d = %s := by decide +kernel'
          % (ci, pi, ci, pi, zl(C['MP'][pi])))
    A('theorem rhsv%d : rhs%d = %s := by decide +kernel'
      % (ci, ci, zl(C['rhs'])))
    A('')
    blocks = ' + '.join(['MS%d_%d' % (ci, a) for a in range(m)] +
                        ['MP%d_%d' % (ci, pi) for pi in range(len(pairs))])
    rwlist = ', '.join(['MSv%d_%d' % (ci, a) for a in range(m)] +
                       ['MPv%d_%d' % (ci, pi) for pi in range(len(pairs))] +
                       ['rhsv%d' % ci])
    A('/-- **The case-%d certificate**: the dual objective falls short of the'
      % ci)
    A('    recursion row\'s right-hand side.  Margin %d/%d.'
      % (C['rhs'] - C['lhs'], C['D']))
    A('    (Scaled by the common denominator %d: %d < %d.) -/'
      % (C['D'], C['lhs'], C['rhs']))
    A('theorem cert%d : %s < rhs%d := by' % (ci, blocks, ci))
    A('  rw [%s]' % rwlist)
    A('  decide')
    A('')
    # the degree / lowest-blocker helper definitions
    rs = ' '.join('r%d' % a for a in range(m))
    rsb = ' '.join('(r%d : ℕ)' % a for a in range(m))
    A('def Dg%d (%s t : ℕ) : ℤ := %s'
      % (ci, rs, ' + '.join('(if c%d_%d r%d t then 1 else 0)' % (ci, a, a)
                            for a in range(m))))
    for pi, (a, b) in enumerate(pairs):
        neg = ''.join('!c%d_%d r%d t && ' % (ci, e, e) for e in range(a))
        A('def Wl%d_%d (%s t : ℕ) : ℤ := if %sc%d_%d r%d t && c%d_%d r%d t '
          'then 1 else 0' % (ci, pi, rs, neg, ci, a, a, ci, b, b))
    A('')
    # ------------------------------------------------------------ soundness
    hyps = ' '.join('(h%d : r%d < %d)' % (a, a, free[a]) for a in range(m))
    covor = ' || '.join('c%d_%d r%d t' % (ci, a, a) for a in range(m))
    A('/-- **No configuration blocks the whole window in case %d.** -/' % ci)
    A('theorem nocov%d {%s : ℕ} %s' % (ci, rs, hyps))
    A('    (hcov : ∀ t, t < n%d → (%s) = true) : False := by' % (ci, covor))
    wsum = ' + '.join('Wl%d_%d %s t' % (ci, pi, rs)
                      for pi in range(len(pairs)))
    A('  have hpt : ∀ t ∈ Finset.range n%d, (1 : ℤ) + (%s) ≤ Dg%d %s t := by'
      % (ci, wsum, ci, rs))
    A('    intro t ht')
    A('    simp only [%s, Dg%d]'
      % (', '.join('Wl%d_%d' % (ci, pi) for pi in range(len(pairs))), ci))
    A('    exact CaseSplit.lowest%d %s (hcov t (Finset.mem_range.mp ht))'
      % (m, ' '.join('_' for _ in range(m))))
    A('  have hd1 : ∀ t ∈ Finset.range n%d, (1 : ℤ) ≤ Dg%d %s t := by'
      % (ci, ci, rs))
    A('    intro t ht')
    A('    simp only [Dg%d]' % ci)
    A('    exact CaseSplit.degpos%d %s (hcov t (Finset.mem_range.mp ht))'
      % (m, ' '.join('_' for _ in range(m))))
    lamsum = ' + '.join('(∑ t ∈ Finset.range n%d, Wl%d_%d %s t)' % (ci, ci, pi, rs)
                        for pi in range(len(pairs)))
    A('  have hsum : (n%d : ℤ) + (%s) ≤ ∑ t ∈ Finset.range n%d, Dg%d %s t := by'
      % (ci, lamsum, ci, ci, rs))
    A('    have h := Finset.sum_le_sum hpt')
    A('    simp only [Finset.sum_add_distrib, Finset.sum_const, '
      'Finset.card_range, nsmul_eq_mul, mul_one] at h')
    A('    exact h')
    # per-pair n <= Lambda
    for pi, (a, b) in enumerate(pairs):
        tgt = ('N%d_%d r%d r%d ≤ ∑ t ∈ Finset.range n%d, Wl%d_%d %s t'
               % (ci, pi, a, b, ci, ci, pi, rs))
        A('  have hn%d : %s := by' % (pi, tgt))
        if a == 0:
            A('    simp only [N%d_%d, Wl%d_%d, le_refl]' % (ci, pi, ci, pi))
        elif a == 1:
            A('    have hsp : ∀ t ∈ Finset.range n%d, Wl%d_%d %s t'
              % (ci, ci, pi, rs))
            A('        = (if c%d_%d r%d t && c%d_%d r%d t then (1:ℤ) else 0)'
              % (ci, a, a, ci, b, b))
            A('          - (if c%d_%d r%d t && c%d_%d r%d t && c%d_0 r0 t '
              'then (1:ℤ) else 0) := by'
              % (ci, a, a, ci, b, b, ci))
            A('      intro t _')
            A('      simp only [Wl%d_%d]' % (ci, pi))
            A('      exact CaseSplit.ind_low2 _ _ _')
            A('    have hnn : (0:ℤ) ≤ ∑ t ∈ Finset.range n%d, Wl%d_%d %s t := by'
              % (ci, ci, pi, rs))
            A('      apply Finset.sum_nonneg')
            A('      intro t _')
            A('      simp only [Wl%d_%d]' % (ci, pi))
            A('      exact CaseSplit.ind_nonneg _')
            A('    have hL : ∑ t ∈ Finset.range n%d, Wl%d_%d %s t'
              % (ci, ci, pi, rs))
            A('        = P%d_%d r%d r%d - C%d_%d r%d r%d r0 := by'
              % (ci, pi, a, b, ci, pi, a, b))
            A('      rw [Finset.sum_congr rfl hsp, Finset.sum_sub_distrib]')
            A('      simp only [P%d_%d, C%d_%d]' % (ci, pi, ci, pi))
            A('    have hm : C%d_%d r%d r%d r0 ≤ M%d_%d r%d r%d :='
              % (ci, pi, a, b, ci, pi, a, b))
            A('      CaseSplit.le_mxr (C%d_%d r%d r%d) %d r0 (by omega)'
              % (ci, pi, a, b, free[0] - 1))
            A('    simp only [N%d_%d]' % (ci, pi))
            A('    split')
            A('    · rw [hL]; omega')
            A('    · exact hnn')
        else:
            A('    simp only [N%d_%d]' % (ci, pi))
            A('    apply Finset.sum_nonneg')
            A('    intro t _')
            A('    simp only [Wl%d_%d]' % (ci, pi))
            A('    exact CaseSplit.ind_nonneg _')
    # the S identity
    Ssum = ' + '.join('S%d_%d r%d' % (ci, a, a) for a in range(m))
    A('  have hS : ∑ t ∈ Finset.range n%d, (w%d t + %d) * Dg%d %s t = %s := by'
      % (ci, ci, yff, ci, rs, Ssum))
    A('    simp only [%s, Dg%d, mul_add, Finset.sum_add_distrib]'
      % (', '.join('S%d_%d' % (ci, a) for a in range(m)), ci))
    A('  have hSD : ∑ t ∈ Finset.range n%d, (w%d t + %d) * Dg%d %s t'
      % (ci, ci, yff, ci, rs))
    A('      = (∑ t ∈ Finset.range n%d, w%d t * Dg%d %s t)'
      % (ci, ci, ci, rs))
    A('        + %d * (∑ t ∈ Finset.range n%d, Dg%d %s t) := by'
      % (yff, ci, ci, rs))
    A('    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum]')
    A('  have hwD : (∑ t ∈ Finset.range n%d, w%d t)' % (ci, ci))
    A('      ≤ ∑ t ∈ Finset.range n%d, w%d t * Dg%d %s t := by' % (ci, ci, ci, rs))
    A('    apply Finset.sum_le_sum')
    A('    intro t ht')
    A('    have h1 : (1:ℤ) ≤ Dg%d %s t := hd1 t ht' % (ci, rs))
    A('    have h2 : (0:ℤ) ≤ w%d t := wnn%d t (Finset.mem_range.mp ht)'
      % (ci, ci))
    A('    calc w%d t = w%d t * 1 := (mul_one _).symm' % (ci, ci))
    A('      _ ≤ w%d t * Dg%d %s t := by exact mul_le_mul_of_nonneg_left h1 h2'
      % (ci, ci, rs))
    # the aggregate identity
    aSsum = ' + '.join('aS%d_%d r%d' % (ci, a, a) for a in range(m))
    aPsum = ' + '.join('aP%d_%d r%d r%d' % (ci, pi, a, b)
                       for pi, (a, b) in enumerate(pairs))
    Nsum = ' + '.join('N%d_%d r%d r%d' % (ci, pi, a, b)
                      for pi, (a, b) in enumerate(pairs))
    A('  have hid : (%s) + (%s) = (%s) - %d * (%s) := by'
      % (aSsum, aPsum, Ssum, yff, Nsum))
    A('    simp only [%s, %s, %s]'
      % (', '.join('aS%d_%d' % (ci, a) for a in range(m)),
         ', '.join('aP%d_%d' % (ci, pi) for pi in range(len(pairs))),
         ', '.join('L%d_%d' % (ci, a) for a in range(m))))
    A('    ring')
    for a, q in enumerate(free):
        A('  have bS%d : aS%d_%d r%d ≤ MS%d_%d := '
          'CaseSplit.le_mxr (aS%d_%d) %d r%d (by omega)'
          % (a, ci, a, a, ci, a, ci, a, q - 1, a))
    for pi, (a, b) in enumerate(pairs):
        A('  have bP%d : aP%d_%d r%d r%d ≤ MP%d_%d := '
          'CaseSplit.le_mxr2 (aP%d_%d) %d %d r%d r%d (by omega) (by omega)'
          % (pi, ci, pi, a, b, ci, pi, ci, pi, free[a] - 1, free[b] - 1, a, b))
    A('  have hrhs : rhs%d = (∑ t ∈ Finset.range n%d, w%d t) + %d * (n%d : ℤ) := rfl'
      % (ci, ci, ci, yff, ci))
    A('  have hc := cert%d' % ci)
    facts = (['hsum', 'hS', 'hSD', 'hwD', 'hid', 'hrhs', 'hc'] +
             ['hn%d' % pi for pi in range(len(pairs))] +
             ['bS%d' % a for a in range(m)] +
             ['bP%d' % pi for pi in range(len(pairs))])
    A('  linarith [%s]' % ', '.join(facts))
    A('')
    return L


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else '19_23'
    with open(os.path.join(R27, 'cert_%s.json' % tag)) as f:
        D = json.load(f)
    y, W, held, free = D['y'], D['W'], D['held'], D['free']
    ncase = len(D['cases'])
    names = []
    B = ['/-',
         'CASE-SPLIT CERTIFICATES, rung %s: the gears as blocking predicates'
         % tag.replace('_', '->'),
         'in (phase, offset) coordinates.  Gear q has teeth at the slot',
         'residues u and q - u with 6u = q -+ 1 (GENERATED).',
         '-/',
         'import CaseSplit',
         '',
         'namespace CaseCert%d' % y,
         ''] + gear_defs(D['full_gears'] if 'full_gears' in D
                         else sorted(set(D['held'] + D['free']))) + \
        ['', 'end CaseCert%d' % y, '']
    with open(os.path.join(PROOFS, 'CaseCert%dB.lean' % y), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(B))
    names.append('CaseCert%dB' % y)
    print('wrote proofs/CaseCert%dB.lean' % y)
    for ci in range(ncase):
        L = ['/-',
             'CASE-SPLIT CERTIFICATE, rung %s, case %d of %d (GENERATED by'
             % (tag.replace('_', '->'), ci, ncase),
             'research/gen_case_lean.py from research/data/r27/cert_%s.json,'
             % tag,
             'which re-derives every number from the primes alone).',
             '',
             'Machine %d, window width %d, held gears %s at phases %s.'
             % (y, W, held, D['cases'][ci]['ws']),
             'Free gears %s.  All numbers are the LP thread\'s exact rational'
             % free,
             'dual scaled by the case denominator %d.' % D['cases'][ci]['D'],
             '-/',
             'import CaseCert%dB' % y,
             '',
             'namespace CaseCert%d' % y,
             '']
        L += gen_case(D, ci)
        L += ['end CaseCert%d' % y, '']
        name = 'CaseCert%dC%d' % (y, ci)
        with open(os.path.join(PROOFS, name + '.lean'), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(L))
        names.append(name)
        print('wrote proofs/%s.lean (%d lines)' % (name, len(L)))
    # ------------------------------------------------------------- the root
    allg = sorted(set(D['held'] + free))
    held = D['held']
    orall = ' || '.join('gb%d (p %% %d) i' % (q, q) for q in allg)
    R = ['/-',
         'THE %s RUNG BY CASE-SPLIT LP DUALITY (GENERATED root).'
         % tag.replace('_', '->'),
         '',
         'Every configuration of machine %d has its held gears %s at exactly'
         % (y, held),
         'one of the %d phase tuples, and each of those cases carries an exact'
         % ncase,
         'dual certificate of the restricted level-2 covering relaxation.  So',
         'no window of %d consecutive slots of machine %d is fully blocked.'
         % (W, y),
         '',
         'NO CENSUS HYPOTHESIS, NO PERIOD SCAN: the only inputs are the primes',
         'up to %d and %d integers per case.'
         % (y, len(D['cases'][0]['y']) + len(D['cases'][0]['nu']) + 1),
         '-/']
    for ci in range(ncase):
        R.append('import CaseCert%dC%d' % (y, ci))
    R += ['import Machine%d' % y, '', 'namespace CaseCert%d' % y, '']
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
    # the machines above 19 are defined by successive gear additions; 19 is
    # the one with a CRT-tuple characterisation.
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
    # per-case bridge
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
    # exhaustiveness
    R.append('/-- **`F(%d) <= %d` by the case split**: every window of %d'
             % (y, W, W))
    R.append('consecutive slots contains an opening of machine %d. -/' % y)
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
    prevF = {23: (19, 25), 29: (23, 34), 31: (29, 43)}[y]
    R.append('/-- **(D) at alpha = 3 at the %d->%d step, BY CASE-SPLIT LP'
             % (prevF[0], y))
    R.append('DUALITY**: every gap of machine %d is at most `F(%d) + %d = %d`.'
             % (y, prevF[0], y, W))
    R.append('No census hypothesis, no period scan - only the primes up to %d'
             % y)
    R.append('and the %d case certificates. -/' % ncase)
    R.append('theorem D_%d_%d_case (n : ℕ) : Machine%d.g%d n ≤ %d + %d :='
             % (prevF[0], y, y, y, prevF[1], y))
    R.append('  F_le n')
    R.append('')
    R.append('end CaseCert%d' % y)
    R.append('')
    with open(os.path.join(PROOFS, 'CaseCert%d.lean' % y), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(R))
    names.append('CaseCert%d' % y)
    print('wrote proofs/CaseCert%d.lean' % y)
    print('cases:', ncase)
    print('lakefile targets:', ', '.join('"%s"' % n for n in names))


if __name__ == '__main__':
    main()

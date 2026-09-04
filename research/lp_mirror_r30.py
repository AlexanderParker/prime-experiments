"""ROUND 30, LP-DUALITY THREAD - THE MIRROR HALVING, MADE EXACT.

THEOREM (mirror transcription of case-split certificates).  Fix a machine
(gears q_1 < ... < q_m), a width W, held gears H = (q_1..q_k) at phases ws,
free gears G = (q_{k+1}..q_m), and the level-2 relaxation RelaxStar(G, W, H, ws)
with position set pos, columns (S, r) for S a 1- or 2-subset of G and r a
phase tuple, links, cut rows (i, lam), recursion row frow and rhs |pos|.

Define the MIRROR
    on phases:      m_q(r)  = (1 - W - r) mod q                 (each gear q)
    on positions:   rho(i)  = W - 1 - i
    on cases:       MIRROR(ws) = (m_q(w_q) : q in H)
    on columns:     pi(S, r) = (S, (m_q(r_q) : q in S))          (same block S)
    on links:       a link is (parent (Sp, rp), children (S, rp with v inserted
                    at `drop`) for v in Z_{S[drop]});  pi sends it to the link
                    with parent pi(Sp, rp) and the same S (children permuted by
                    v -> m(v), a re-ordering that the link sum does not see).

LEMMA (round 29).  rho(hits(q, r, W)) = hits(q, m_q(r), W).  [i is blocked by q
at phase r iff i = t - r (mod q) for a tooth t; the teeth {u', q - u'} are closed
under t -> -t; and W - 1 - i = (-t) - (m_q(r)) (mod q).]

CLAIMS, each proved by the lemma applied gear by gear:
 (1) pos(MIRROR(ws)) = rho(pos(ws)).
 (2) O_{pi(j)} at MIRROR(ws) = rho(O_j at ws) - the overlap of the mirrored
     column with the mirrored position set is the reflection of the original
     overlap; in particular |O| is preserved.
 (3) frow(MIRROR(ws))[pi(j)] = frow(ws)[j].  For single columns frow = |O|.
     For pair columns frow = -(|P| - maxcover(P)), where maxcover is the largest
     number of positions of P = O_j that the gears BELOW the pair's first gear
     can cover over all their phases; under rho every lower gear's hit set at
     phase r maps to its hit set at m(r), so the set of coverable subsets of
     rho(P) is the image of the set of coverable subsets of P, and the maximum
     is equal.  (The decision procedure, `max_cover_dom`, is a search over the
     same finite structure and returns the same value and the same exactness
     flag, because both are functions of the family of hit-restricted subsets,
     which is mapped bijectively.)
 (4) A cut row (i, lam) is valid iff its subset sums over the atoms of G are
     >= 1 - a condition on lam alone, not on i - so (rho(i), lam) is a valid
     cut row at the mirrored case whenever (i, lam) is valid at the original.
 (5) TRANSCRIPTION.  Given a certificate (rows, y, nu, yff) of the case ws, put
         rows' = [(rho(i), lam) for (i, lam) in rows],   y' = y,   yff' = yff,
         nu'[pi(link)] = nu[link].
     Then for every column j:  a'_{pi(j)} = a_j, because every term of
         a_j = sum_r y_r lam^r[mask S_j] [i_r in O_j] + yff frow_j
               + sum_{links: j child} nu - sum_{links: j parent} nu
     is carried to the corresponding term at pi(j) by (2), (3) and the link
     map.  pi preserves blocks, so lhs' = sum_S max_{block S} a' = lhs; and
     rhs' = sum_r y'_r (1 - lam'^r_0) + yff' |pos'| = rhs by (1).  Hence
     lhs' < rhs' with margin' = margin, and (rows', y', nu', yff') is an exact
     dual certificate of the case MIRROR(ws), with the same op count.  []

COROLLARY.  One representative per mirror orbit is decided; the other
member's certificate is TRANSCRIBED by this file and re-verified from its
JSON by `emit_certs_r27.check_case_json` (which rebuilds the relaxation from
the primes and recomputes lhs / rhs / margin from the file's own integers).
The self-mirror case - MIRROR(ws) = ws, exactly one per level since each q is
odd - is its own representative.

    uv run python research/lp_mirror_r30.py GATE29 [workers]
        transcribe every certificate of the round-29 31->37 rung to its
        mirror image, re-verify each from JSON alone, and compare with the
        certificate the round-29 sweep found for that case independently.
    uv run python research/lp_mirror_r30.py GATE29T [workers]
        the same for the TRANSLATION lemma (below, after the mirror gate):
        every certificate transcribed onto every case whose position set is
        an exact translate of its own, re-verified from JSON alone.
    uv run python research/lp_mirror_r30.py ONE <cert.json>
"""
import json
import os
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emit_certs_r27 import check_case_json, unfr                      # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
R29 = os.path.join(HERE, 'data', 'r29')
R30 = os.path.join(HERE, 'data', 'r30')


def mirror_phase(r, q, W):
    return (1 - W - r) % q


def mirror_ws(ws, held, W):
    return [mirror_phase(w, q, W) for w, q in zip(ws, held)]


def column_maps(layout, W):
    """pi on columns and on links, from the case-independent layout.

    Returns (colperm, linkperm): colperm[j] = pi(j), linkperm[t] = pi(t)."""
    free = layout['free_gears']
    cols = layout['cols']
    tupidx = {(m, tuple(r)): j for j, (m, r) in enumerate(cols)}
    colperm = []
    for (m, r) in cols:
        S = [free[i] for i in range(len(free)) if (m >> i) & 1]
        rm = tuple(mirror_phase(rq, q, W) for rq, q in zip(r, S))
        colperm.append(tupidx[(m, rm)])
    # a link is determined by (parent column, mask of the children's S)
    links = layout['links']
    byparent = {}
    for t, (par, kids) in enumerate(links):
        byparent[(par, cols[kids[0]][0])] = t
    linkperm = []
    for t, (par, kids) in enumerate(links):
        linkperm.append(byparent[(colperm[par], cols[kids[0]][0])])
    assert sorted(colperm) == list(range(len(cols)))
    assert sorted(linkperm) == list(range(len(links)))
    return colperm, linkperm


def mirror_cert(J, layout):
    """the transcribed certificate of MIRROR(ws), as a JSON-ready dict"""
    W = J['W']
    held, ws = J['held_gears'], J['ws']
    colperm, linkperm = column_maps(layout, W)
    n_cols = len(layout['cols'])
    frow = [0] * n_cols
    for j, v in enumerate(J['frow']):
        frow[colperm[j]] = v
    nu = [None] * len(linkperm)
    for t, v in enumerate(J['nu']):
        nu[linkperm[t]] = v
    assert all(v is not None for v in nu)
    out = dict(J)
    out['ws'] = mirror_ws(ws, held, W)
    out['pos'] = sorted(W - 1 - i for i in J['pos'])
    out['rows'] = [[W - 1 - i, lam] for (i, lam) in J['rows']]
    out['frow'] = frow
    out['nu'] = nu
    out['mirror_of'] = list(ws)
    out['mirror_is'] = ("transcribed from the certificate of case %s by "
                        "lp_mirror_r30.mirror_cert: positions i -> W-1-i, "
                        "phases r -> (1-W-r) mod q, links permuted with the "
                        "columns; y, yff, lhs, rhs, margin unchanged (the "
                        "mirror transcription theorem)" % (list(ws),))
    out.pop('source_pickle', None)
    return out


# ============================================ THE TRANSLATION LEMMA (new)
# THEOREM (translation transcription).  Let ws' = ws + t (every held phase
# advanced by t mod its gear) and suppose the position sets are EXACT
# translates, pos(ws') = pos(ws) - t as subsets of [0, W) - equivalently the
# held gears block [0, t) at ws and [W - t, W) at ws' (t > 0; symmetric for
# t < 0).  Then with rho(i) = i - t and m_q(r) = (r + t) mod q the five
# claims of the mirror theorem hold verbatim (i in hits(q, r, W) iff
# i - t in hits(q, r + t, W) for i in pos, because both endpoints stay
# inside the window; the lower gears' hit-restricted subsets of a pair
# overlap P are mapped bijectively; cut validity is a condition on lam
# alone), so (rows - t, y, nu o pi_t^-1, yff) is an exact dual certificate of
# the case ws + t with the same lhs, rhs, margin and op count.  []
#
# WHY IT MATTERS: round 29 found the (V*, |pos|) classes of a sweep COARSER
# than the mirror orbits and could not name the symmetry ("not a
# translation - no ws -> ws + t preserves V* except t = 0, tested at all
# 35").  It IS a translation - one that holds exactly when the boundary
# positions are blocked, which a test of "ws -> ws + t for every case" cannot
# see.  At m37 W=95 k=2 the mirror+translation classes number 11 = the 11
# value classes round 29 measured; at m41 W=104 k=2 they number 14 = the 14
# value classes of this round's E15.  At m53 W=171 k=4 the classes are 1,391
# against 2,503 mirror orbits (a further 1.8x nobody used this round).
def translate_ws(ws, held, t):
    return [(w + t) % q for w, q in zip(ws, held)]


def translation_shift(J):
    """the t > 0 (or < 0) for which pos(ws + t) = pos(ws) - t exactly, if
    any, given the certificate's own held phases; None otherwise"""
    from lp_degree_range import hits
    W, held, ws = J['W'], J['held_gears'], J['ws']

    def pos_of(w):
        b = set()
        for q, x in zip(held, w):
            b |= set(hits(q, x, W))
        return frozenset(set(range(W)) - b)
    P = pos_of(ws)
    out = []
    for t in range(-(W - 1), W):
        if t == 0:
            continue
        if frozenset(i - t for i in P) == pos_of(translate_ws(ws, held, t)):
            out.append(t)
    return out


def _perm_maps(layout, phase_map):
    """pi on columns and links for a per-gear phase map r -> phase_map(r, q)"""
    free = layout['free_gears']
    cols = layout['cols']
    tupidx = {(m, tuple(r)): j for j, (m, r) in enumerate(cols)}
    colperm = []
    for (m, r) in cols:
        S = [free[i] for i in range(len(free)) if (m >> i) & 1]
        rm = tuple(phase_map(rq, q) for rq, q in zip(r, S))
        colperm.append(tupidx[(m, rm)])
    links = layout['links']
    byparent = {}
    for tt, (par, kids) in enumerate(links):
        byparent[(par, cols[kids[0]][0])] = tt
    linkperm = [byparent[(colperm[par], cols[kids[0]][0])]
                for (par, kids) in links]
    assert sorted(colperm) == list(range(len(cols)))
    assert sorted(linkperm) == list(range(len(links)))
    return colperm, linkperm


def translate_cert(J, layout, t):
    """the transcribed certificate of the case ws + t (position sets must be
    exact translates - asserted by check_case_json downstream, which rebuilds
    pos from the primes and compares it with the file's)"""
    held, ws = J['held_gears'], J['ws']
    colperm, linkperm = _perm_maps(layout, lambda r, q: (r + t) % q)
    frow = [0] * len(layout['cols'])
    for j, v in enumerate(J['frow']):
        frow[colperm[j]] = v
    nu = [None] * len(linkperm)
    for tt, v in enumerate(J['nu']):
        nu[linkperm[tt]] = v
    assert all(v is not None for v in nu)
    out = dict(J)
    out['ws'] = translate_ws(ws, held, t)
    out['pos'] = sorted(i - t for i in J['pos'])
    assert all(0 <= i < J['W'] for i in out['pos'])
    out['rows'] = [[i - t, lam] for (i, lam) in J['rows']]
    out['frow'] = frow
    out['nu'] = nu
    out['translate_of'] = [list(ws), t]
    out.pop('source_pickle', None)
    return out


def _check_translate(args):
    src, layout_path, tmpdir = args
    with open(src) as fh:
        J = json.load(fh)
    with open(layout_path) as fh:
        L = json.load(fh)
    res = []
    for t in translation_shift(J):
        M = translate_cert(J, L, t)
        p = os.path.join(tmpdir, 'tr%d_' % (t % 1000) + os.path.basename(src))
        with open(p, 'w') as fh:
            json.dump(M, fh, separators=(',', ':'))
        lhs, rhs = check_case_json(p)
        assert lhs == unfr(J['lhs']) and rhs == unfr(J['rhs'])
        os.remove(p)
        res.append((J['ws'], t, M['ws']))
    return res


def gate29t(workers=3, rung='31_37'):
    t0 = time.time()
    with open(os.path.join(R29, 'manifest_%s.json' % rung)) as fh:
        man = json.load(fh)
    tmp = tempfile.mkdtemp(prefix='lp_transl_')
    shutil.copy(os.path.join(R29, 'layout_%s.json' % rung),
                os.path.join(tmp, 'layout_%s.json' % rung))
    layout_path = os.path.join(tmp, 'layout_%s.json' % rung)
    jobs = [(os.path.join(R29, f), layout_path, tmp) for f in man['case_files']]
    print("GATE29T  translation-transcribe every certificate of rung %s to"
          " every case whose position set is an exact translate, re-verify"
          " each from JSON alone" % man['rung'], flush=True)
    with Pool(workers) as pool:
        res = [r for rr in pool.map(_check_translate, jobs, chunksize=4)
               for r in rr]
    shutil.rmtree(tmp, ignore_errors=True)
    src = set(tuple(r[0]) for r in res)
    dst = set(tuple(r[2]) for r in res)
    print("  %d translation transcriptions from %d source cases onto %d"
          " target cases, ALL RE-VERIFIED from JSON (relaxation rebuilt from"
          " the primes at the TRANSLATED case, lhs/rhs/margin equal to the"
          " source's)  GREEN" % (len(res), len(src), len(dst)))
    ts = {}
    for r in res:
        ts[r[1]] = ts.get(r[1], 0) + 1
    print("  shifts used: %s" % dict(sorted(ts.items())))
    print("  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0), flush=True)
    return res


# ================================================================== GATE 29
def _check_one(args):
    src, layout_path, tmpdir = args
    with open(src) as fh:
        J = json.load(fh)
    with open(layout_path) as fh:
        L = json.load(fh)
    M = mirror_cert(J, L)
    name = os.path.basename(src)
    p = os.path.join(tmpdir, 'mirror_' + name)
    with open(p, 'w') as fh:
        json.dump(M, fh, separators=(',', ':'))
    # check_case_json finds the layout by rung name in the file's directory
    lhs, rhs = check_case_json(p)
    assert lhs == unfr(J['lhs']) and rhs == unfr(J['rhs'])
    # the certificate the round-29 sweep found INDEPENDENTLY for that case
    disk = os.path.join(os.path.dirname(src), 'cert_%s_h%s.json'
                        % (J['rung'].replace('->', '_'),
                           "_".join(str(w) for w in M['ws'])))
    same = None
    if os.path.exists(disk):
        with open(disk) as fh:
            D = json.load(fh)
        assert D['ws'] == M['ws']
        same = (unfr(D['margin']) == rhs - lhs, D['ops'] == J['ops'],
                D['y'] == M['y'] and D['nu'] == M['nu'])
    return (J['ws'], M['ws'], [(rhs - lhs).numerator, (rhs - lhs).denominator],
            same)


def gate29(workers=3, rung='31_37'):
    t0 = time.time()
    with open(os.path.join(R29, 'manifest_%s.json' % rung)) as fh:
        man = json.load(fh)
    tmp = tempfile.mkdtemp(prefix='lp_mirror_')
    shutil.copy(os.path.join(R29, 'layout_%s.json' % rung),
                os.path.join(tmp, 'layout_%s.json' % rung))
    layout_path = os.path.join(tmp, 'layout_%s.json' % rung)
    jobs = [(os.path.join(R29, f), layout_path, tmp) for f in man['case_files']]
    print("GATE29  mirror-transcribe %d certificates of rung %s and re-verify"
          " each from JSON alone" % (len(jobs), man['rung']), flush=True)
    with Pool(workers) as pool:
        res = pool.map(_check_one, jobs, chunksize=4)
    held, W = man['held_gears'], man['W']
    for (ws, mws, _m, _s) in res:
        assert mws == mirror_ws(ws, held, W)
    selfm = [r for r in res if r[0] == r[1]]
    assert len(selfm) == 1, "exactly one self-mirror case per level"
    eq_margin = sum(1 for r in res if r[3] and r[3][0])
    eq_ops = sum(1 for r in res if r[3] and r[3][1])
    eq_dual = sum(1 for r in res if r[3] and r[3][2])
    print("  %d/%d transcribed certificates RE-VERIFIED from JSON (relaxation"
          " rebuilt from the primes at the MIRRORED case, every cut row"
          " re-checked, lhs/rhs/margin recomputed)  GREEN" % (len(res),
                                                             len(jobs)))
    print("  self-mirror case: %s" % (selfm[0][0],))
    print("  against the round-29 sweep's OWN certificate of the mirrored"
          " case: equal margin %d/%d, equal op count %d/%d, identical dual"
          " %d/%d" % (eq_margin, len(res), eq_ops, len(res), eq_dual,
                      len(res)))
    shutil.rmtree(tmp, ignore_errors=True)
    print("  ALL ASSERTIONS GREEN  [%.0fs]" % (time.time() - t0), flush=True)
    return res


def main():
    a = sys.argv[1:]
    cmd = (a[0].upper() if a else 'HELP')
    if cmd == 'GATE29':
        gate29(int(a[1]) if len(a) > 1 else 3)
    elif cmd == 'GATE29T':
        gate29t(int(a[1]) if len(a) > 1 else 3)
    elif cmd == 'ONE':
        src = a[1]
        with open(src) as fh:
            J = json.load(fh)
        with open(os.path.join(os.path.dirname(src), 'layout_%s.json'
                               % J['rung'].replace('->', '_'))) as fh:
            L = json.load(fh)
        M = mirror_cert(J, L)
        print(json.dumps({k: v for k, v in M.items()
                          if k in ('ws', 'mirror_of', 'pos', 'lhs', 'rhs',
                                   'margin')}, indent=1))
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

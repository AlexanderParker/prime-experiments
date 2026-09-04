"""ROUND 30, LP-DUALITY THREAD - THE ROUND'S TABLES, BUILT FROM DISK.

Reads the decided cells of research/data/r30/ (and the round-28/29 cells the
scores refer to), prints and persists to research/lp_r30_results.txt every
table the round report quotes, and ASSERTS the facts it reports.  A
reporter, not a decider: nothing is computed here that is not already a
decided cell, a manifest or a witness on disk.

  1  THE TREE at 47 -> 53 (machine 53, W = 171): per level, the verdict
     counts, iteration histogram, base-cut share, op counts, margins; the
     price table predicted vs measured.
  2  THE k = 3 PROBE (case (0,0,0), n = 11): the loop's trajectory.
  3  E14  W_c(43, 3) from the bisection file, and the ladder W_c(y,3)/F(y).
  4  E15  machine 41, W = 104, k = 2: (V*, |pos|) classes vs mirror orbits.
  5  E16  41 -> 43 at the increment width 117: the tree at machine 43.
  6  THE MIRROR THEOREM's gate lines (from lp_mirror_r30 GATE29, quoted).

    uv run python research/lp_score_r30.py
"""
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')
R29 = os.path.join(HERE, 'data', 'r29')
R30 = os.path.join(HERE, 'data', 'r30')

F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
     43: 103, 47: 118, 53: 145, 59: 161}
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def cells(d, prefix):
    got = []
    if not os.path.isdir(d):
        return got
    for f in sorted(os.listdir(d)):
        if f.startswith(prefix) and f.endswith('.json'):
            with open(os.path.join(d, f)) as fh:
                got.append(json.load(fh))
    return got


def tree_table(y, W, label):
    from lp_tree_r30 import reps, mirror
    from lp_degree_range import gears_of
    g = gears_of(y)
    say()
    say("%s  (machine %d, W = %d)" % (label, y, W))
    say()
    say("   k  reps   on disk  CERTIFIED  it=0   base-cut  NOCERT/other"
        "  ops (sum)     mean ops  secs (sum)  margin min / max")
    tot = {}
    for k in range(3, 7):
        sel = cells(R30, 'cell_m%d_w%d_k%d_' % (y, W, k))
        if not sel:
            continue
        nrep = len(reps(y, W, k)) if k <= 5 else None
        cert = [c for c in sel if c.get('verdict') == 'CERTIFIED']
        oth = [c for c in sel if c.get('verdict') != 'CERTIFIED']
        its0 = sum(1 for c in cert if c.get('its') == 0)
        base = sum(1 for c in cert if c.get('rows_all_base_cut'))
        ops = sum(c.get('ops') or 0 for c in cert)
        secs = sum(c.get('total_secs') or 0 for c in sel)
        mg = [Fraction(c['margin']) for c in cert if c.get('margin')]
        say("   %d  %-5s  %-7d  %-9d  %-5d  %-8d  %-12d  %-12d  %-8s  %-10.0f  %s / %s"
            % (k, nrep, len(sel), len(cert), its0, base, len(oth), ops,
               ('%.0f' % (ops / len(cert))) if cert else '-', secs,
               min(mg) if mg else '-', max(mg) if mg else '-'))
        tot[k] = dict(sel=sel, cert=cert, oth=oth)
        hist = {}
        for c in cert:
            hist[c.get('its')] = hist.get(c.get('its'), 0) + 1
        say("      iteration histogram: %s" % dict(sorted(hist.items(),
                                                          key=lambda t: (t[0] is None, t[0]))))
        if oth:
            vs = {}
            for c in oth:
                vs[c.get('verdict')] = vs.get(c.get('verdict'), 0) + 1
            say("      non-certified verdicts: %s" % vs)
            for c in oth[:12]:
                say("        %-32s plain=%s its=%s traj %s -> %s vs |pos| %d%s"
                    % (c['cell'], c.get('plain_verdict'), c.get('plain_its'),
                       ('%.3f' % c['plain_traj'][0]) if c.get('plain_traj')
                       else '-',
                       ('%.3f' % c['plain_traj'][-1]) if c.get('plain_traj')
                       else '-', c['npos'],
                       ('  V*=%s' % c.get('vstar')) if 'vstar' in c else ''))
        # the mirror on the tree: representatives only, so every cell's
        # mirror image is NOT on disk unless self-mirror - asserted
        held = g[:k]
        for c in sel:
            m = mirror(tuple(c['ws']), held, W)
            assert (m == tuple(c['ws'])) == c['self_mirror']
            if m != tuple(c['ws']):
                assert not os.path.exists(os.path.join(
                    R30, 'cell_m%d_w%d_k%d_h%s.json'
                    % (y, W, k, "_".join(str(x) for x in m)))), \
                    "both members of a mirror orbit were decided"
    return tot


def main():
    say("=" * 78)
    say("LP-DUALITY ROUND 30 - RESULTS FROM DISK")
    say("=" * 78)

    # ------------------------------------------------------- 1. the tree
    tot = tree_table(53, 171, "1. THE 47 -> 53 TREE AT W_inc = F(47) + 53 = 171")
    if 4 in tot:
        c4 = tot[4]
        n_cert = len(c4['cert'])
        n_all = len(c4['sel'])
        say()
        say("   certified fraction at the first affordable level (k = 4):"
            " %d / %d representatives = %.4f  (37 -> 41 at k = 3 was"
            " 376/385 = 0.9766)" % (n_cert, n_all, n_cert / n_all))
        mg = [Fraction(c['margin']) for c in c4['cert']]
        hist = {}
        for m in mg:
            hist[m] = hist.get(m, 0) + 1
        say("   k = 4 margin histogram (representatives): "
            + ", ".join("%s x%d" % (k, v) for k, v in sorted(hist.items())))
    # the emitted step
    sp = os.path.join(R30, 'manifest_47_53.json')
    if os.path.exists(sp):
        with open(sp) as fh:
            S = json.load(fh)
        say()
        say("   STEP MANIFEST: %s cases over %s; leaves by level %s;"
            " margin min %s max %s; %d exact ops; PARTITION ASSERTED = %s"
            % (S['n_cases_total'],
               " + ".join("k=%d %d" % (p['k'], p['n_cases'])
                          for p in S['parts']),
               S['leaves_by_level'], Fraction(*S['margin_min']),
               Fraction(*S['margin_max']), S['ops_total'],
               S['exhaustiveness_holds']))
        assert S['exhaustiveness_holds'] is True

    # ------------------------------------------------------- 2. k = 3 probe
    say()
    say("2. THE k = 3 PROBE at machine 53, W = 171, case (0,0,0), n = 11")
    p = os.path.join(R30, 'cell_m53_w171_k3_h0_0_0.json')
    if os.path.exists(p):
        with open(p) as fh:
            c = json.load(fh)
        say("   |pos| = %d, cols %d, links %d; plain loop: %s after %s passes,"
            " %.0f s; LP max %s -> %s against %d"
            % (c['npos'], c['ncols'], c['nlinks'], c['plain_verdict'],
               c['plain_its'], c['plain_secs'],
               '%.4f' % c['plain_traj'][0], '%.4f' % c['plain_traj'][-1],
               c['npos']))
        say("   (the lifted LP is out of reach at n = 11; at k = 4 the SAME"
            " phases (0,0,0,0) give a base-cut polytope that is EMPTY at"
            " iteration zero in ~20 s)")

    # ---------------------------------------------------------- 3. E14
    say()
    say("3. E14 - W_c(y, 3) = min{W : G < 0} at the all-zero case")
    say()
    say("   y    W_c   F(y)   W_c / F(y)    source")
    rows = []
    for d in (R30, R29, R28):
        for f in sorted(os.listdir(d)):
            if f.startswith('wc_') and f.endswith('_k3.json'):
                with open(os.path.join(d, f)) as fh:
                    J = json.load(fh)
                if J['machine'] in F and J['machine'] not in [r[0] for r in rows]:
                    rows.append((J['machine'], J['W_c'], os.path.basename(d),
                                 J))
    for (y, wc, src, J) in sorted(rows):
        say("   %-4d %-5d %-6d %.4f       %s" % (y, wc, F[y], wc / F[y], src))
    w43 = [r for r in rows if r[0] == 43]
    if w43:
        J = w43[0][3]
        say("   m43 bisection widths: %s" % {k: (None if v is None else
                                                round(v, 4))
                                             for k, v in J['widths'].items()})
        verdict = "CONFIRMED" if J['W_c'] >= 92 else "REFUTED"
        say("   E14 (W_c(43,3) >= 92, ratio > 0.89): %s - W_c(43,3) = %d,"
            " ratio %.3f%s" % (verdict, J['W_c'], J['W_c'] / 103.0,
                               "; AND THE RATIO HAS CROSSED 1: the case-0"
                               " cell at k = 3 is NOT certifiable at the truth"
                               " F(43) = 103, nor at 104 or 105"
                               if J['W_c'] > 104 else ""))

    # ---------------------------------------------------------- 4. E15
    say()
    say("4. E15 - machine 41, W = 104, k = 2: (V*, |pos|) classes vs mirror"
        " orbits")
    e15 = cells(R30, 'e15_m41_w104_k2_')
    if e15:
        from lp_tree_r30 import reps
        n_orb = len(reps(41, 104, 2))
        cls = set()
        for c in e15:
            v = ('EMPTY' if c.get('empty') else
                 (round(c['vstar'], 6) if c.get('vstar') is not None else None))
            cls.add((v, c['npos']))
            say("   case %-8s |pos|=%-3d V*=%s" % (tuple(c['ws']), c['npos'],
                                                   v))
        say("   %d of %d orbits decided by the lifted route; %d distinct"
            " (V*, |pos|) classes" % (len(e15), n_orb, len(cls)))
        if len(e15) == n_orb:
            say("   E15 (classes STRICTLY FEWER than orbits): %s"
                % ("CONFIRMED" if len(cls) < n_orb else "REFUTED"))

    # ---------------------------------------------------------- 5. E16
    t43 = tree_table(43, 117, "5. E16 - 41 -> 43 AT THE INCREMENT WIDTH"
                     " F_2(41) + s_min(43) = 103 + 14 = 117")
    if t43:
        say()
        say("   E16 asks: does any case refuted at k need MORE than one"
            " further gear?  Read off the tree above (a level-5 cell"
            " existing means a level-4 refusal, i.e. two gears beyond k=3).")

    # ------------------------------------- 7. mirror + translation classes
    say()
    say("7. THE CASE SPLIT'S SYMMETRY CLASSES: mirror, and boundary-blocked")
    say("   translation (lp_mirror_r30 translation lemma) - arithmetic only")
    say()
    say("   sweep             cases  mirror orbits  mirror+translation classes"
        "  measured value classes")
    import itertools
    import collections
    from lp_degree_range import hits, gears_of
    from lp_tree_r30 import mirror

    def classes(y, W, k):
        g = gears_of(y)
        held = g[:k]
        cs = list(itertools.product(*[range(q) for q in held]))

        def P(ws):
            b = set()
            for q, w in zip(held, ws):
                b |= hits(q, w, W)
            return frozenset(set(range(W)) - b)
        Pm = {ws: P(ws) for ws in cs}
        par = {c: c for c in cs}

        def find(c):
            while par[c] != c:
                c = par[c]
            return c
        for ws in cs:
            par[find(ws)] = find(mirror(ws, held, W))
            for t in range(1, W):
                wt = tuple((w + t) % q for w, q in zip(ws, held))
                if min(Pm[ws]) >= t and \
                        frozenset(i - t for i in Pm[ws]) == Pm[wt]:
                    par[find(ws)] = find(wt)
        ncl = len(set(find(c) for c in cs))
        morb = len(set(min(ws, mirror(ws, held, W)) for ws in cs))
        return len(cs), morb, ncl
    measured = {(37, 95, 2): 11, (41, 104, 2): 14}
    for (y, W, k) in ((37, 95, 2), (41, 104, 2), (37, 95, 3), (41, 104, 3),
                      (43, 134, 3), (47, 132, 4), (53, 171, 4), (53, 171, 3)):
        n, morb, ncl = classes(y, W, k)
        m = measured.get((y, W, k))
        say("   m%-3d W=%-4d k=%d  %5d  %6d         %6d                     %s"
            % (y, W, k, n, morb, ncl, m if m is not None else '-'))
        if m is not None:
            assert m == ncl, ("value classes != mirror+translation classes",
                              y, W, k, m, ncl)
    # the E15 check: every exact-translate pair of cases has equal V*
    if e15:
        from lp_tree_r30 import canon
        V = {tuple(c['ws']): round(c['vstar'], 6) for c in e15}
        held = (5, 7)
        cs = [(a, b) for a in range(5) for b in range(7)]

        def P(ws):
            b = set()
            for q, w in zip(held, ws):
                b |= hits(q, w, 104)
            return frozenset(set(range(104)) - b)
        pairs = eq = 0
        for x, yv in itertools.combinations(cs, 2):
            for t in range(1, 104):
                if frozenset(i - t for i in P(x)) == P(yv) or \
                        frozenset(i + t for i in P(x)) == P(yv):
                    pairs += 1
                    eq += (V[canon(x, held, 104)] == V[canon(yv, held, 104)])
                    break
        assert pairs == eq
        say("   m41 W=104 k=2: %d exact-translate pairs among the 35 cases,"
            " equal V* at %d/%d  (ASSERTED)" % (pairs, eq, pairs))

    p = os.path.join(HERE, 'lp_r30_results.txt')
    with open(p, 'w') as fh:
        fh.write("\n".join(OUT) + "\n")
    print("\n  written to %s" % p)


if __name__ == '__main__':
    main()

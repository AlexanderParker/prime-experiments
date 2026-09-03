"""ROUND 29, LP-DUALITY THREAD - THE ROUND'S TABLES, BUILT FROM DISK.

Reads every decided cell in research/data/r28/ and research/data/r29/ and prints
(and persists to research/lp_r29_results.txt) the five tables the round report
quotes.  Nothing is computed here that is not already a decided cell on disk:
this file is a reporter, not a decider, and it asserts the facts it reports.

  1  SMALLEST-k FOR 31->37 at the (D) budget width W = 95: the k = 1, 2, 3
     verdict tables, and which cells carry EXACT in-polytope refutations.
  2  THE MARGIN COLUMN of the emitted rung.
  3  E12: the offset V* - |pos| at the INCREMENT WIDTH, step by step, against
     the structural quantity W_inc - F(q').
  4  E11: every cell whose LIFTED POLYTOPE IS EMPTY, and whether it certified
     at iteration zero.
  5  RUNG TEN (43->47) at the increment width 132.

    uv run python research/lp_score_r29.py
"""
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
R28 = os.path.join(HERE, 'data', 'r28')
R29 = os.path.join(HERE, 'data', 'r29')

# F(y), exact, from the project corpus (lp_degree_range.F_EXACT for y <= 43).
F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91,
     43: 103, 47: 118, 53: 145, 59: 161}
F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
      41: 103, 43: 116, 53: 159}
STEPS_INC = [(11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (29, 31),
             (31, 37), (37, 41), (41, 43), (43, 47)]

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def cells(*dirs):
    """Every decided cell, DEDUPED by cell name - round 28's frontier map and
    this round's sweeps overlap at one cell (m41, W = 104, k = 3, case 0), and
    they agree (both EMPTY).  The later directory wins."""
    got = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith('cell_') and f.endswith('.json'):
                with open(os.path.join(d, f)) as fh:
                    c = json.load(fh)
                c['_dir'] = os.path.basename(d)
                got[c['cell']] = c
    return list(got.values())


def smin(q):
    from lp_degree_range import teeth
    u = teeth(q)[0]
    return min((2 * u) % q, (-2 * u) % q)


def main():
    C = cells(R28, R29)
    say("=" * 78)
    say("LP-DUALITY ROUND 29 - RESULTS FROM DISK  (%d decided cells)" % len(C))
    say("=" * 78)

    # ------------------------------------------------- 1. smallest k, 31->37
    say()
    say("1. 31 -> 37 AT THE (D) BUDGET WIDTH W = 95 - WHICH k CERTIFIES")
    say()
    say("   k  cases  CERTIFIED  ASYMPTOTE(exact refutation)  ASYMPTOTE(no"
        " witness)  other")
    for k in (1, 2, 3):
        sel = [c for c in C if c['machine'] == 37 and c['W'] == 95
               and c['k'] == k]
        if not sel:
            continue
        cert = [c for c in sel if c.get('verdict') == 'CERTIFIED']
        ref = [c for c in sel if c.get('verdict') == 'REFUTED']
        nw = [c for c in sel if c.get('verdict') == 'ASYMPTOTE-NOWITNESS']
        oth = [c for c in sel if c not in cert + ref + nw]
        say("  %2d  %5d  %9d  %27d  %20d  %5d"
            % (k, len(sel), len(cert), len(ref), len(nw), len(oth)))
        for c in ref[:3]:
            say("        exact refutation: %s  V* = %.4f >= |pos| = %d,"
                " row %s >= %s"
                % (c['cell'], c['vstar'], c['frhs'],
                   (c.get('row_value') or '')[:24] + '...', c.get('row_rhs')))
    say()
    say("   A cell with an EXACT in-polytope refutation PROVES the case-split")
    say("   vehicle at that k can never certify this rung, however many cuts")
    say("   are generated.  Round 26 had only a cut-loop STALL at k = 2 (LP")
    say("   max 40.994 against 40), which is an undecided cell.")

    # ------------------------------------------------------ 2. margin column
    for rung in ('31_37', 'inc_37_41_k3', 'inc_37_41_k4'):
        p = os.path.join(R29, 'manifest_%s.json' % rung)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            M = json.load(fh)
        if 'rung' not in M:
            continue
        say()
        say("2. EMITTED RUNG %s  (machine %d, W = %d, k = %d, held %s)"
            % (M['rung'], M['machine'], M['W'], M['k'],
               tuple(M['held_gears'])))
        say("   claim: %s" % M['claim'])
        col = [Fraction(*m) for (_w, m) in M['margin_column']]
        hist = {}
        for m in col:
            hist[m] = hist.get(m, 0) + 1
        say("   %d cases, exhaustive = %s; all rows base cut = %s;"
            " max iterations %d; %d exact ops"
            % (M['n_cases'], M['exhaustiveness_holds'],
               M['rows_all_base_cut'], M['iterations_max'], M['ops_total']))
        say("   MARGIN COLUMN (rhs - lhs per case): min %s, max %s"
            % (min(col), max(col)))
        say("   histogram: " + ", ".join("%s x%d" % (k, v)
                                         for k, v in sorted(hist.items())))
    p = os.path.join(R29, 'manifest_inc_37_41.json')
    if os.path.exists(p):
        with open(p) as fh:
            S = json.load(fh)
        say()
        say("2b. THE STEP MANIFEST FOR 37 -> 41 (a MIXED-k split)")
        say("    %s" % S['W_is'])
        for part in S['parts']:
            say("    part k=%d, held %s: %d cases"
                % (part['k'], tuple(part['held_gears']), part['n_cases']))
        say("    %d cases total; PARTITION ASSERTED: %s"
            % (S['n_cases_total'], S['exhaustiveness']))
        say("    margin min %s max %s; %d exact ops; witness %s"
            % (Fraction(*S['margin_min']), Fraction(*S['margin_max']),
               S['ops_total'], S['witness_file']))
        assert S['exhaustiveness_holds'] is True

    # ------------------------------------------------------------- 3. E12
    say()
    say("3. E12 - THE OFFSET AT THE INCREMENT WIDTH, AND WHAT IT TRACKS")
    say()
    say("   step      W_inc = F_2(M)+s_min   F(q')   W_inc - F(q')   G at the"
        " all-zero case")
    for (M, q) in STEPS_INC:
        if M not in F2:
            continue
        W = F2[M] + smin(q)
        gap = W - F[q]
        rows = [c for c in C if c['machine'] == q and c['W'] == W]
        txt = []
        for k in (1, 2, 3):
            r = [c for c in rows if c['k'] == k and all(x == 0
                                                        for x in c['ws'])]
            if not r:
                continue
            c = r[0]
            if c.get('empty'):
                txt.append("k=%d EMPTY(-inf)" % k)
            elif c.get('gap') is not None:
                txt.append("k=%d %+0.4f" % (k, c['gap']))
            elif c.get('method') == 'plain-loop':
                txt.append("k=%d certified" % k)
        say("   %2d->%-2d   %3d = %3d + %-3d       %4d    %+5d          %s"
            % (M, q, W, F2[M], smin(q), F[q], gap,
               "; ".join(txt) if txt else "-"))
    say()
    say("   The offset is NOT a function of the machine: it is a function of")
    say("   W_inc - F(q').  Where that is NEGATIVE (31 -> 37, -8) no sound")
    say("   method can certify and the offset must be positive; where it is")
    say("   positive the cell can be, and is, certifiable.")

    # ------------------------------------------------------------- 4. E11
    say()
    say("4. E11 - CELLS WHOSE LIFTED POLYTOPE IS EMPTY")
    emp = [c for c in C if c.get('empty') is True]
    z0 = [c for c in emp if c.get('its') == 0
          and c.get('verdict') == 'CERTIFIED']
    bad = [c for c in emp if c not in z0]
    say("   %d empty-polytope cells on disk (r28 + r29); %d certified at"
        " ITERATION ZERO once seeded; %d did not" % (len(emp), len(z0),
                                                     len(bad)))
    for c in bad[:8]:
        say("     exception: %s verdict=%s its=%s"
            % (c['cell'], c.get('verdict'), c.get('its')))
    new = [c for c in emp if c['_dir'] == 'r29']
    say("   of these, %d are ROUND-29 cells (the 'further cells' E11 asks"
        " for), %d of them at iteration zero"
        % (len(new), len([c for c in new if c.get('its') == 0
                          and c.get('verdict') == 'CERTIFIED'])))

    # --------------------------------------------------------- 5. rung ten
    say()
    say("5. RUNG TEN 43 -> 47 AT THE INCREMENT WIDTH W_inc = 116 + 16 = 132")
    say("   (Constructor's spectrum-plus-depth bound at this step is F_4(43)")
    say("   = 132, budget F(43) + 47 = 150, margin 18 - the SAME number.)")
    say()
    say("   k   case            |pos|   V*         verdict     ops      secs")
    for c in sorted([c for c in C if c['machine'] == 47 and c['W'] == 132],
                    key=lambda c: (c['k'], c['ws'])):
        say("   %d   %-14s  %-6s  %-10s %-11s %-8s %.0f"
            % (c['k'], str(tuple(c['ws'])), c.get('frhs'),
               'EMPTY' if c.get('empty') else
               ('%.4f' % c['vstar'] if c.get('vstar') is not None else '-'),
               c.get('verdict'), c.get('ops'), c.get('total_secs', 0)))

    # ------------------------------------------------------------ 6. INC41
    say()
    say("6. THE 37 -> 41 INCREMENT-WIDTH SWEEP (machine 41, W = 104, k = 3)")
    sel = [c for c in C if c['machine'] == 41 and c['W'] == 104
           and c['k'] == 3]
    cert = [c for c in sel if c.get('verdict') == 'CERTIFIED']
    say("   %d of 385 cases decided; %d CERTIFIED; %d other"
        % (len(sel), len(cert), len(sel) - len(cert)))
    for c in [c for c in sel if c.get('verdict') != 'CERTIFIED'][:8]:
        say("     %s  %s" % (c['cell'], c.get('verdict')))
    if sel:
        say("   iteration counts: max %s; methods: %s"
            % (max(c.get('its', 0) or 0 for c in sel),
               {m: len([c for c in sel if c.get('method') == m])
                for m in set(c.get('method') for c in sel)}))

    # --------------------------------------------------------------- W_c
    say()
    say("7. E9 - W_c(y, k) = min{W : G < 0} at the all-zero case")
    say()
    say("   y    k   W_c   F(y)   W_c / F(y)")
    for d in (R29, R28):
        for f in sorted(os.listdir(d)):
            if not f.startswith('wc_'):
                continue
            with open(os.path.join(d, f)) as fh:
                J = json.load(fh)
            if 'W_c' not in J or J['machine'] not in F:
                continue
            say("   %-4d %-3d %-5d %-6d %.4f   (%s)"
                % (J['machine'], J['k'], J['W_c'], F[J['machine']],
                   J['W_c'] / float(F[J['machine']]),
                   os.path.basename(d)))

    # ------------------------------------------------- 8. the mirror on the LP
    say()
    say("8. MIRROR EQUIVARIANCE OF THE CASE SPLIT (a lemma, gated here)")
    say("   reflect(hits(q,r,W)) = hits(q, (1-W-r) mod q, W), because the teeth")
    say("   {u', q-u'} are closed under t -> -t.  So the case at ws and the")
    say("   case at (1-W-ws) mod q have reflected position sets, isomorphic")
    say("   relaxations, and therefore equal V* and |pos|.")
    from lp_degree_range import gears_of, hits
    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        for W in (74, 95, 104, 132):
            for q in gears_of(y):
                for r in range(q):
                    ref = frozenset(W - 1 - i for i in hits(q, r, W))
                    assert ref == hits(q, (1 - W - r) % q, W), (q, r, W)
    say("   LEMMA ASSERTED for every gear of m11..m47 at W = 74, 95, 104, 132")
    for (y, W, k) in ((37, 95, 2), (41, 104, 3), (31, 74, 3), (47, 132, 5)):
        sel = [c for c in C if c['machine'] == y and c['W'] == W
               and c['k'] == k]
        if len(sel) < 2:
            continue
        # A cell decided by the FAST PATH records no V* (the lifted LP was
        # never run on it), so only cells whose verdict came from the lifted
        # route carry a comparable value.  |pos| is recorded for every cell.
        val = {tuple(c['ws']): ('EMPTY' if c.get('empty')
                                else (round(c['vstar'], 6)
                                      if c.get('vstar') is not None
                                      else None)) for c in sel}
        npos = {tuple(c['ws']): c['frhs'] for c in sel}
        held = gears_of(y)[:k]

        def mir(ws):
            return tuple((1 - W - w) % q for w, q in zip(ws, held))
        pairs = [(w, mir(w)) for w in val if mir(w) in val]
        okp = sum(1 for (w, m) in pairs if npos[m] == npos[w])
        assert okp == len(pairs), ("mirror breaks |pos|", y, W, k)
        vp = [(w, m) for (w, m) in pairs
              if val[w] is not None and val[m] is not None]
        okv = sum(1 for (w, m) in vp if val[m] == val[w])
        assert okv == len(vp), \
            ("mirror does not preserve the lifted value", y, W, k, okv,
             len(vp))
        vals = [v for v in val.values() if v is not None]
        classes = len(set(zip(vals, [npos[w] for w in val
                                     if val[w] is not None])))
        say("   m%d W=%d k=%d : %d cells, %d mirror pairs on disk; |pos| agrees"
            " %d/%d; V* agrees %d/%d on the %d cells the lifted LP decided%s"
            % (y, W, k, len(val), len(pairs), okp, len(pairs), okv, len(vp),
               len(vals),
               "   (VACUOUS - one value class)" if classes <= 1 else
               "   (%d distinct value classes)" % classes))

    p = os.path.join(HERE, 'lp_r29_results.txt')
    with open(p, 'w') as fh:
        fh.write("\n".join(OUT) + "\n")
    print("\n  written to %s" % p)


if __name__ == '__main__':
    main()

"""ROUND 28, LP-DUALITY THREAD - the round's tables, from the JSON on disk."""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R28 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'r28')

def cells():
    out = []
    for f in sorted(os.listdir(R28)):
        if f.startswith('cell_') and f.endswith('.json'):
            out.append(json.load(open(os.path.join(R28, f))))
    return out

def main():
    C = cells()
    print("=== A. the frontier grid (case 0, one cell per width) ===")
    g = [c for c in C if all(w == 0 for w in c['ws'])]
    g.sort(key=lambda c: (c['machine'], c['k'], c['W']))
    print("    y   k    W  |pos|     V*         G          G/W    verdict")
    for c in g:
        v, gp = c.get('vstar'), c.get('gap')
        print("  %4d %3d %4d %5s  %-10s %-10s %-8s %s"
              % (c['machine'], c['k'], c['W'], c.get('frhs'),
                 'EMPTY' if c.get('empty') else ('%.4f' % v if v is not None else '-'),
                 '-' if gp is None else '%+.4f' % gp,
                 '-' if gp is None else '%+.5f' % (gp / c['W']),
                 c.get('verdict')))
    print()
    print("=== B. the crossing widths W_c (bisection on the lifted value) ===")
    print("    y   k   W_c   F(y)  budget  W_c/F  single crossing")
    for f in sorted(os.listdir(R28)):
        if f.startswith('wc_') and f.endswith('.json'):
            d = json.load(open(os.path.join(R28, f)))
            print("  %4d %3d %5d %6s %7s  %5.3f  %s"
                  % (d['machine'], d['k'], d['wc'], d['F'], d['budget'],
                     d['wc'] / float(d['F']), d['single_crossing']))
    print()
    print("=== C. the padded step 31->37 (machine 37 case splits) ===")
    for (W, k) in sorted({(c['W'], c['k']) for c in C if c['machine'] == 37}):
        S = [c for c in C if c['machine'] == 37 and c['W'] == W and c['k'] == k]
        vc = collections.Counter(c.get('verdict') for c in S)
        emp = sum(1 for c in S if c.get('empty'))
        asy = [c for c in S if c.get('side') == 'ASYMPTOTE']
        ops = sum(c.get('ops') or 0 for c in S)
        line = ("  W=%-3d k=%d  %3d cells: %s; lifted polytope EMPTY in %d; "
                "%d exact ops" % (W, k, len(S), dict(vc), emp, ops))
        print(line)
        if asy:
            gaps = sorted(c['gap'] for c in asy)
            print("      NOT CERTIFIABLE in %d/%d cases (%.1f%%); "
                  "V* - |pos| ranges %+.3f .. %+.3f (median %+.3f)"
                  % (len(asy), len(S), 100.0 * len(asy) / len(S),
                     gaps[0], gaps[-1], gaps[len(gaps) // 2]))
    print()
    print("=== D. anything unresolved on disk ===")
    bad = [c for c in C if c.get('verdict') not in
           ('CERTIFIED', 'REFUTED', 'ASYMPTOTE-FLOAT')]
    for c in bad:
        print("  %-30s %s %s" % (c.get('cell'), c.get('side'), c.get('verdict')))
    if not bad:
        print("  none")

if __name__ == '__main__':
    main()

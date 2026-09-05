"""Item 1, shallow rungs: the separability distribution on the attaining 3-runs with v >= 6
at m13..m23, split trivial (v >= min(L,R), the peel bound) / hard (v < min(L,R)), with the C2
covering verdict of 2g.i.a beside it so the two quantities can be compared run by run.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r46'))
from sp_core import (gears_of, us_of, sieve, gap_stats, attaining_runs, sep_run, letters)
from gl_glue import glue

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def dist(vals):
    d = {}
    for x in vals:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def main(out, tops=(13, 17, 19, 23)):
    allrows = {}
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        P, blocked = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, blocked)
        att = attaining_runs(opens, gaps, N)
        rows = []
        for (x0, L, v, R) in att:
            r = sep_run(gears, us, x0, L, v, R)
            c2 = glue(gears, us, x0, L, v, R) is not None
            hard = v < min(L, R)
            shared = [gears[i] for i in range(len(gears)) if (r['sharedmask'] >> i) & 1]
            rows.append(dict(x0=x0, L=L, v=v, R=R, hard=hard, c2=c2, s=r['s'], u=r['u'],
                             sigma=round(r['sigma'], 4), ov=r['ov'], loss=r['loss'],
                             shared=shared,
                             sharedclass=[letters(g, v) for g in shared]))
        allrows[top] = rows
        hd = [r for r in rows if r['hard']]
        tr = [r for r in rows if not r['hard']]
        out.write(f"\n===== m{top}  F={F} F_2={F2}  attaining 3-runs v>=6: {len(rows)} "
                  f"(trivial {len(tr)}, hard {len(hd)})\n")
        for name, sub in (("TRIVIAL", tr), ("HARD", hd)):
            if not sub:
                continue
            out.write(f"  {name}: s dist {dist([r['s'] for r in sub])}; "
                      f"u dist {dist([r['u'] for r in sub])}; "
                      f"ov dist {dist([r['ov'] for r in sub])}; "
                      f"loss dist {dist([r['loss'] for r in sub])}\n")
            out.write(f"      sigma min/med/max = "
                      f"{min(r['sigma'] for r in sub):.3f}/"
                      f"{sorted(r['sigma'] for r in sub)[len(sub)//2]:.3f}/"
                      f"{max(r['sigma'] for r in sub):.3f}; "
                      f"C2 ok {sum(r['c2'] for r in sub)}/{len(sub)}\n")
            sh = {}
            for r in sub:
                for g, cl in zip(r['shared'], r['sharedclass']):
                    sh[(g, cl)] = sh.get((g, cl), 0) + 1
            out.write(f"      shared gears (gear, move-class): "
                      f"{dict(sorted(sh.items()))}\n")
        # cross-tab s against C2 on the hard runs
        ct = {}
        for r in hd:
            k = (r['s'], r['c2'])
            ct[k] = ct.get(k, 0) + 1
        out.write(f"  HARD cross-tab (s, C2ok) -> count: {dict(sorted(ct.items()))}\n")
        out.flush()
    with open(os.path.join(RES, 'sep_shallow.json'), 'w') as f:
        json.dump({str(k): v for k, v in allrows.items()}, f)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()

"""Item 5: the graded certificates and their loss.

Two graded certificates are compared on the same runs.

  SEP-c  (this branch)  a two-colouring of the gears used on the REAL flanks: left gears block
         x_1-1, x_1-2, ... and right gears block x_2+1, x_2+2, ...; the CRT point certifies
         F_2 >= a + b + 2 where a, b are the two blocked runs.  Loss c = (L+R) - (a+b+2).
         c = 0 exactly when the shared number s = 0.

  GLUE-c (2g.i.a's C2, graded)  the same colouring read on the GLUE instance (left base x_0+1,
         right base x_2-L+1): a = the longest run of covered offsets below the hole, b above.
         Certifies F_2 >= a + b + 2; loss 0 is exactly C2 success.

GLUE-c <= SEP-c always (the glue's options include the separation certificate's).  Reported per
machine on the attaining 3-runs with v >= 6, and at the resistant m29 run (18, 10, 30).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r46'))
from sp_core import (gears_of, us_of, sieve, gap_stats, attaining_runs, run_masks,
                     separability, letters)
from gl_glue import cov_pair, solve_cover

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def glue_loss(gears, us, x0, L, v, R):
    """max over two-colourings of the covered block around the hole; returns (loss, colouring)."""
    T, h, covL, covR = cov_pair(gears, us, x0, L, R, x0 + L + v)
    n = len(gears)
    best = (-1, None)
    for m in range(1 << n):
        acc = 0
        for i in range(n):
            acc |= covR[i] if (m >> i) & 1 else covL[i]
        a = 0
        while h - 1 - a >= 0 and (acc >> (h - 1 - a)) & 1:
            a += 1
        b = 0
        while h + 1 + b < T and (acc >> (h + 1 + b)) & 1:
            b += 1
        if a + b > best[0]:
            best = (a + b, m)
    return (L + R) - (best[0] + 2), best[1]


def dist(v):
    d = {}
    for x in v:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def main(out):
    import numpy as np
    for top in (13, 17, 19, 23):
        gears, us = gears_of(top), us_of(gears_of(top))
        P, b = sieve(gears, us)
        opens, gaps, F, F2, N = gap_stats(P, b)
        att = attaining_runs(opens, gaps, N)
        rows = []
        for (x0, L, v, R) in att:
            ml, mr, nl, nr = run_masks(gears, us, x0, L, v, R)
            sp = separability(ml, mr, nl, nr)
            gl, _ = glue_loss(gears, us, x0, L, v, R)
            rows.append((v < min(L, R), sp['loss'], gl, sp['s'], L, v, R, L + R))
        out.write(f"\n===== m{top} F={F} F_2={F2}: {len(rows)} attaining 3-runs v>=6\n")
        for nm, sel in (("TRIVIAL", False), ("HARD", True)):
            sub = [r for r in rows if r[0] == sel]
            if not sub:
                continue
            out.write(f"  {nm} ({len(sub)}): SEP-c loss {dist([r[1] for r in sub])}\n")
            out.write(f"  {' ' * len(nm)}   GLUE-c loss {dist([r[2] for r in sub])}; "
                      f"max GLUE-c {max(r[2] for r in sub)}, "
                      f"mean {sum(r[2] for r in sub)/len(sub):.2f}\n")
        out.flush()
    # the deep rungs: only the recorded resistant runs and the m29/m31 hard attaining sets
    for top, path in ((29, 'sep_deep_m29.json'), (31, 'sep_deep_m31.json')):
        gears, us = gears_of(top), us_of(gears_of(top))
        with open(os.path.join(RES, path)) as f:
            rows = json.load(f)
        out.write(f"\n===== m{top}: {len(rows)} attaining 3-runs v>=6\n")
        for nm, sel in (("TRIVIAL", False), ("HARD", True)):
            sub = [r for r in rows if r['hard'] == sel]
            if not sub:
                continue
            gls = [glue_loss(gears, us, r['x0'], r['L'], r['v'], r['R'])[0] for r in sub]
            out.write(f"  {nm} ({len(sub)}): SEP-c loss {dist([r['loss'] for r in sub])}\n")
            out.write(f"  {' ' * len(nm)}   GLUE-c loss {dist(gls)}; max {max(gls)}, "
                      f"mean {sum(gls)/len(gls):.2f}\n")
        out.flush()
    # the resistant case
    gears, us = gears_of(29), us_of(gears_of(29))
    x0, L, v, R = 278620515, 18, 10, 30
    ml, mr, nl, nr = run_masks(gears, us, x0, L, v, R)
    sp = separability(ml, mr, nl, nr)
    gl, mask = glue_loss(gears, us, x0, L, v, R)
    sh = [gears[i] for i in range(len(gears)) if (sp['sharedmask'] >> i) & 1]
    out.write(f"\n===== the resistant run: m29 (L,v,R) = (18,10,30) at x0 = {x0}\n"
              f"  shared number s = {sp['s']}, used u = {sp['u']}, sigma = {sp['sigma']:.3f}, "
              f"raw overlap ov = {sp['ov']}\n"
              f"  shared gears {sh}, move-classes {[letters(g, v) for g in sh]}\n"
              f"  SEP-c loss {sp['loss']}  (certifies F_2 >= {L+R-sp['loss']} against L+R = {L+R})\n"
              f"  GLUE-c loss {gl} (certifies F_2 >= {L+R-gl}); F_2(m29) = 55, F = 43\n")
    out.flush()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()

"""Item 1 at the deep rungs: the separability of the attaining 3-runs with v >= 6 at m29 and
m31.  The chunked full-period pass of r46/gl_deep is reused verbatim to find the runs (the
separability test itself needs only x0 mod g, so no period is ever held).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r46'))
from sp_core import gears_of, us_of, sep_run, letters
from gl_glue import glue
from gl_deep import run as deep_run

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def dist(vals):
    d = {}
    for x in vals:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def main(out, top):
    gears, P, joint, big, perv, thresh, F2, dt = deep_run(top, 8)
    us = us_of(gears)
    import numpy as np
    J2 = joint.reshape(128, 256)
    N = {}
    for v in range(1, 128):
        nz = np.flatnonzero(J2[v])
        if nz.size:
            N[v] = int(nz.max())
    F = max(N)
    att = []
    for vv, lst in sorted(perv.items()):
        for (s, x0, L, R) in lst:
            if s == N[vv]:
                att.append((x0, L, vv, R))
    rows = []
    for (x0, L, v, R) in att:
        r = sep_run(gears, us, x0, L, v, R)
        c2 = glue(gears, us, x0, L, v, R) is not None
        shared = [gears[i] for i in range(len(gears)) if (r['sharedmask'] >> i) & 1]
        rows.append(dict(x0=x0, L=L, v=v, R=R, hard=v < min(L, R), c2=c2, s=r['s'],
                         u=r['u'], sigma=round(r['sigma'], 4), ov=r['ov'], loss=r['loss'],
                         shared=shared, sharedclass=[letters(g, v) for g in shared]))
    out.write(f"\n===== m{top} P={P} F={F} F_2={F2} [{dt:.1f}s]  attaining 3-runs v>=6: "
              f"{len(rows)}\n")
    for nm, sub in (("TRIVIAL", [r for r in rows if not r['hard']]),
                    ("HARD", [r for r in rows if r['hard']])):
        if not sub:
            continue
        out.write(f"  {nm} ({len(sub)}): s dist {dist([r['s'] for r in sub])}; "
                  f"u dist {dist([r['u'] for r in sub])}; ov dist {dist([r['ov'] for r in sub])}; "
                  f"loss dist {dist([r['loss'] for r in sub])}\n")
        out.write(f"      sigma min/med/max = {min(r['sigma'] for r in sub):.3f}/"
                  f"{sorted(r['sigma'] for r in sub)[len(sub)//2]:.3f}/"
                  f"{max(r['sigma'] for r in sub):.3f}; C2 ok "
                  f"{sum(r['c2'] for r in sub)}/{len(sub)}\n")
        sh = {}
        for r in sub:
            for g, cl in zip(r['shared'], r['sharedclass']):
                sh[(g, cl)] = sh.get((g, cl), 0) + 1
        out.write(f"      shared gears (gear, move-class): {dict(sorted(sh.items()))}\n")
    ct = {}
    for r in rows:
        if r['hard']:
            ct[(r['s'], r['c2'])] = ct.get((r['s'], r['c2']), 0) + 1
    out.write(f"  HARD cross-tab (s, C2ok): {dict(sorted(ct.items()))}\n")
    out.flush()
    with open(os.path.join(RES, f'sep_deep_m{top}.json'), 'w') as f:
        json.dump(rows, f)


if __name__ == "__main__":
    top = int(sys.argv[1])
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    o = open(dest, 'a') if dest else sys.stdout
    main(o, top)
    if dest:
        o.close()

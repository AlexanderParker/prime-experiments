"""Is the real machine's gluability really a which-residues property?

2g.i.a: on the HARD attaining 3-runs the real machine glues at 62.5% against a pooled family
9.4%, the 99.6th percentile.  Before reading that as a residue coincidence, control for SIZE.
The glue has to cover L+R-1 offsets; the cap it certifies is F_2.  A run whose outer sum sits
well BELOW F_2 has slack -- fewer columns to buy -- so the rate must fall with

    slack := F_2 - (L + R)     and     depth := L + R - v   (the columns the run really asks for)

This script records, for every hard attaining 3-run of 200 random symmetric-tooth members and
of the real member at m17 and m19, the run's (v, L, R, F, F_2, slack), the C2 verdict and the
shared number s, and then compares the real machine with the family AT MATCHED SLACK.
"""
import sys, os, json, random
from math import prod
from sp_core import gears_of, us_of, gap_stats, attaining_runs, separability, run_masks
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'r46'))
from gl_glue import cov_pair, solve_cover
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PR = [5, 7, 11, 13, 17, 19]


def sieve_w(gears, ws):
    P = prod(gears)
    b = np.zeros(P, dtype=bool)
    for g, w in zip(gears, ws):
        b[w % g::g] = True
        b[(-w) % g::g] = True
    return P, b


def rows_for(gears, ws):
    P, b = sieve_w(gears, ws)
    opens, gaps, F, F2, N = gap_stats(P, b)
    out = []
    for (x0, L, v, R) in attaining_runs(opens, gaps, N):
        if v >= min(L, R):
            continue
        T, h, cL, cR = cov_pair(gears, ws, x0, L, R, x0 + L + v)
        ok = solve_cover(cL, cR, T, h) is not None
        ml, mr, nl, nr = run_masks(gears, ws, x0, L, v, R)
        r = separability(ml, mr, nl, nr)
        out.append(dict(v=v, L=L, R=R, sum=L + R, F=F, F2=F2, slack=F2 - (L + R),
                        c2=bool(ok), s=r['s'], loss=r['loss']))
    return F, F2, out


def rate(rows):
    return (sum(r['c2'] for r in rows), len(rows))


def main(out, nmem=200, seed=911911):
    rng = random.Random(seed)
    dump = {}
    for top in (17, 19):
        gears = [p for p in PR if p <= top]
        realw = tuple(min(u, g - u) for g, u in zip(gears, us_of(gears)))
        rF, rF2, rreal = rows_for(gears, realw)
        fam = []
        seen = {realw}
        while len(fam) < nmem:
            ws = tuple(rng.randrange(1, (g + 1) // 2) for g in gears)
            if ws in seen:
                continue
            seen.add(ws)
            F, F2, rr = rows_for(gears, ws)
            for r in rr:
                r['member'] = ws
            fam.extend(rr)
        ok, n = rate(rreal)
        fok, fn = rate(fam)
        out.write(f"\n===== m{top}: real F={rF} F_2={rF2}; hard attaining runs "
                  f"real {ok}/{n} = {100*ok/max(n,1):.1f}%, family pooled {fok}/{fn} = "
                  f"{100*fok/max(fn,1):.1f}%\n")
        out.write(f"  real runs (v, L+R, slack, C2, s): "
                  f"{[(r['v'], r['sum'], r['slack'], r['c2'], r['s']) for r in rreal]}\n")
        # matched on slack
        out.write("  C2 rate by slack = F_2 - (L+R):\n")
        allsl = sorted({r['slack'] for r in fam} | {r['slack'] for r in rreal})
        for sl in allsl:
            f = [r for r in fam if r['slack'] == sl]
            rl = [r for r in rreal if r['slack'] == sl]
            if not f and not rl:
                continue
            fo, fnn = rate(f)
            ro, rn = rate(rl)
            out.write(f"    slack {sl:3d}: family {fo:5d}/{fnn:5d} = "
                      f"{100*fo/max(fnn,1):5.1f}%   real {ro}/{rn}"
                      f"{'  <-- real runs here' if rn else ''}\n")
        # matched on v as well
        out.write("  C2 rate by (v, slack) for the cells the real machine occupies:\n")
        for key in sorted({(r['v'], r['slack']) for r in rreal}):
            f = [r for r in fam if (r['v'], r['slack']) == key]
            rl = [r for r in rreal if (r['v'], r['slack']) == key]
            fo, fnn = rate(f)
            ro, rn = rate(rl)
            out.write(f"    v={key[0]}, slack={key[1]}: family {fo}/{fnn} = "
                      f"{100*fo/max(fnn,1):.1f}%   real {ro}/{rn}\n")
        # per-member rates restricted to the real machine's slack values
        sls = {r['slack'] for r in rreal}
        per = {}
        for r in fam:
            if r['slack'] in sls:
                a = per.setdefault(r['member'], [0, 0])
                a[0] += r['c2']; a[1] += 1
        big = [(a[0] / a[1], a[1]) for a in per.values() if a[1] >= 4]
        if big:
            big.sort()
            realrate = ok / max(n, 1)
            above = sum(1 for x, _ in big if x >= realrate)
            out.write(f"  members with >=4 hard runs at the real machine's slack values: "
                      f"{len(big)}; median rate {big[len(big)//2][0]*100:.1f}%; "
                      f"{above} of them match or beat the real {100*realrate:.1f}% "
                      f"-> real percentile {100*(1-above/len(big)):.1f}\n")
        dump[str(top)] = dict(real=rreal, fam=fam)
        out.flush()
    with open(os.path.join(RES, 'sep_confound.json'), 'w') as f:
        json.dump({k: dict(real=v['real'],
                           fam=[{kk: vv for kk, vv in r.items() if kk != 'member'}
                                for r in v['fam']]) for k, v in dump.items()}, f)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()

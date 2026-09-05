"""Item 1, the family: 200 random symmetric-tooth members at m13, m17, m19 (teeth {w, -w},
w uniform in 1..(g-1)/2 -- the alignment-rules section 5 family), each on its full period,
scored exactly as the real member: the attaining 3-runs with v >= 6, split trivial / hard,
with the shared number s, the used number u and the separation index sigma.

The real machine is one member of this family and is scored the same way, so "are the real
machine's flanks more separable than a random member's" is answered directly.
"""
import sys, os, json, random
import numpy as np
from math import prod
from sp_core import (gears_of, us_of, gap_stats, attaining_runs, separability, run_masks)

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PR = [5, 7, 11, 13, 17, 19]


def sieve_w(gears, ws):
    P = prod(gears)
    b = np.zeros(P, dtype=bool)
    for g, w in zip(gears, ws):
        b[w % g::g] = True
        b[(-w) % g::g] = True
    return P, b


def score(gears, ws):
    P, b = sieve_w(gears, ws)
    opens, gaps, F, F2, N = gap_stats(P, b)
    att = attaining_runs(opens, gaps, N)
    tr, hd = [], []
    for (x0, L, v, R) in att:
        ml, mr, nl, nr = run_masks(gears, ws, x0, L, v, R)
        r = separability(ml, mr, nl, nr)
        rec = (r['s'], r['u'], r['sigma'], r['ov'], r['loss'], L, v, R)
        (hd if v < min(L, R) else tr).append(rec)
    return F, F2, tr, hd


def summ(rows):
    if not rows:
        return None
    s = [r[0] for r in rows]
    sg = [r[2] for r in rows]
    return dict(n=len(rows), s_mean=sum(s) / len(s), s_min=min(s), s_max=max(s),
                sep0=sum(1 for x in s if x == 0), sig_mean=sum(sg) / len(sg),
                sig_min=min(sg))


def main(out, nmem=200, seed=20260905):
    rng = random.Random(seed)
    dump = {}
    for top in (13, 17, 19):
        gears = [p for p in PR if p <= top]
        realw = tuple(min(u, g - u) for g, u in zip(gears, us_of(gears)))
        fsize = prod((g - 1) // 2 for g in gears)
        target = min(nmem, fsize - 1)   # realw is excluded from the draw
        rows = []
        seen = {realw}
        while len(rows) < target:
            ws = tuple(rng.randrange(1, (g + 1) // 2) for g in gears)
            if ws in seen:
                continue
            seen.add(ws)
            F, F2, tr, hd = score(gears, ws)
            rows.append((ws, F, F2, summ(tr), summ(hd)))
        rF, rF2, rtr, rhd = score(gears, realw)
        R_tr, R_hd = summ(rtr), summ(rhd)
        out.write(f"\n===== family m{top}: {target} random members (family size {fsize})\n")
        out.write(f"  REAL teeth {realw}: F={rF} F_2={rF2}\n")
        for nm, rr in (("trivial", R_tr), ("HARD", R_hd)):
            if rr:
                out.write(f"    real {nm}: n={rr['n']} s mean {rr['s_mean']:.2f} "
                          f"min {rr['s_min']} max {rr['s_max']} sep0 {rr['sep0']}; "
                          f"sigma mean {rr['sig_mean']:.3f} min {rr['sig_min']:.3f}\n")
        for nm, key, rr in (("trivial", 3, R_tr), ("HARD", 4, R_hd)):
            sub = [r for r in rows if r[key]]
            if not sub or rr is None:
                out.write(f"    {nm}: no comparable family rows\n")
                continue
            pooled_n = sum(r[key]['n'] for r in sub)
            pooled_s = sum(r[key]['s_mean'] * r[key]['n'] for r in sub) / pooled_n
            pooled_sep0 = sum(r[key]['sep0'] for r in sub)
            pooled_sig = sum(r[key]['sig_mean'] * r[key]['n'] for r in sub) / pooled_n
            means = sorted(r[key]['s_mean'] for r in sub)
            sigs = sorted(r[key]['sig_mean'] for r in sub)
            below_s = sum(1 for r in sub if r[key]['s_mean'] < rr['s_mean'])
            below_sig = sum(1 for r in sub if r[key]['sig_mean'] < rr['sig_mean'])
            out.write(f"    {nm} family: members {len(sub)}, runs {pooled_n}; "
                      f"pooled mean s {pooled_s:.2f} (real {rr['s_mean']:.2f}); "
                      f"separable runs {pooled_sep0}/{pooled_n} "
                      f"(real {rr['sep0']}/{rr['n']})\n")
            out.write(f"      member mean s   min/q1/med/q3/max = "
                      f"{means[0]:.2f}/{means[len(means)//4]:.2f}/"
                      f"{means[len(means)//2]:.2f}/{means[3*len(means)//4]:.2f}/"
                      f"{means[-1]:.2f};  members strictly BELOW the real mean s: "
                      f"{below_s}/{len(sub)}  -> real percentile "
                      f"{100*below_s/len(sub):.1f}\n")
            out.write(f"      member mean sigma min/q1/med/q3/max = "
                      f"{sigs[0]:.3f}/{sigs[len(sigs)//4]:.3f}/{sigs[len(sigs)//2]:.3f}/"
                      f"{sigs[3*len(sigs)//4]:.3f}/{sigs[-1]:.3f}; pooled {pooled_sig:.3f} "
                      f"(real {rr['sig_mean']:.3f}); members below: {below_sig}/{len(sub)}"
                      f"  -> real percentile {100*below_sig/len(sub):.1f}\n")
        dump[str(top)] = dict(real=dict(teeth=list(realw), F=rF, F2=rF2, tr=R_tr, hd=R_hd),
                              fam=[dict(teeth=list(r[0]), F=r[1], F2=r[2], tr=r[3], hd=r[4])
                                   for r in rows])
        out.flush()
    with open(os.path.join(RES, 'sep_family.json'), 'w') as f:
        json.dump(dump, f)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()

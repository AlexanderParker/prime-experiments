"""fd_mirror.py -- the mirror symmetry of the fields about the multiples of 6g (fields.md section 6; P6).

The mirror about 6gm is n -> 12gm - n.  A symmetric pair of F_j about 6gm is {n, 12gm - n} with both in F_j;
it is "through g" if g | n.  T5: the pairs through g are the dilates by g of the pairs {6m - i, 6m + i}
(i = 1, 5, 7, 11, ...) with both members in F_{j-1}; radius i = 1 is the column m itself (both-F_{j-1}).

(a) identity gate: symmetric F_2 pairs through g about 6gm (both members inside the section) = prime pairs
    {6m - i, 6m + i} with both members in the cofactor range of the section: 0 mismatches.
(b) growth: mean symmetric pairs per axis through g = 5 along the chain from base 3 (links 1, 2, 3) and the
    machine sights q = 7..53; the innermost-radius (twin) share.
(c) off the multiples of g: the symmetric-hit rate of F_2 about 6gm against the density of F_2 among the
    section's survivors (the null), per finer section, gears 5, 7, 11, 13.
Usage: uv run python fd_mirror.py
"""
import os, json, time
import numpy as np
from fd_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(RES, "mirror.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def omega_table(top, spf):
    mm = np.arange(1, top + 1, dtype=np.int64)
    mm = mm[(mm % 6 == 1) | (mm % 6 == 5)]
    omc, _, _ = omega_arrays(mm, spf)
    omtab = np.full(top + 1, -1, dtype=np.int8)
    omtab[mm] = omc
    return omtab


def mirror_census(lo, hi, g, omtab, j=2, off_axis=True, rcap=None, stride=1):
    """axes 6gm with the cofactor range (lo/g, hi/g]; for each m the symmetric F_j pairs through g about 6gm by
    radius i (both cofactors 6m -+ i in the range, both with Omega j-1), and, off the multiples of g, the
    symmetric-hit count of F_j among survivors n in (lo, hi] with g not | n, 12gm - n in (lo, hi].
    rcap caps the radius i; stride samples every stride-th axis (both used only on the 260M section)."""
    clo, chi = lo // g + 1, hi // g          # cofactors u with g u in (lo, hi]
    out = []
    for m in range((clo + 1) // 6 + 1, chi // 6 + 1, stride):
        c = 6 * m
        rmax = min(c - clo, chi - c)         # radius so that both 6m -+ i in [clo, chi]
        if rcap is not None:
            rmax = min(rmax, rcap)
        if rmax < 1:
            continue
        # i runs over 1, 5, 7, 11, 13, ... <= rmax
        i = np.arange(1, rmax + 1)
        i = i[(i % 6 == 1) | (i % 6 == 5)]
        a, b = omtab[c - i], omtab[c + i]
        both = (a == j - 1) & (b == j - 1)
        n_through = int(both.sum())
        twin1 = bool(both[0]) if len(i) and i[0] == 1 else False
        # count of F_{j-1} members among the cofactors on one side (the exposure of the through-g symmetry)
        exp_through = int((a == j - 1).sum())
        rec = dict(m=m, axis=6 * g * m, radii=len(i), through=n_through, exposure=exp_through, twin1=twin1)
        if off_axis:
            # off the multiples of g: survivors n in (lo, hi], g not | n, with 12gm - n in (lo, hi]
            A = 12 * g * m
            nlo, nhi = max(lo + 1, A - hi), min(hi, A - lo - 1)
            if nhi > nlo:
                n = np.arange(nlo, nhi + 1, dtype=np.int64)
                n = n[((n % 6 == 1) | (n % 6 == 5)) & (n % g != 0)]
                on = omtab[n] == j
                onm = omtab[A - n] == j
                rec["off_hits"] = int(on.sum())
                rec["off_sym"] = int((on & onm).sum())
        out.append(rec)
    return out


def main():
    t0 = time.time()
    results = {}
    say("# mirror census", time.ctime())
    secs = [finer_section(p) for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]]
    secs += construction_sections(3, 2) + construction_sections(5, 1) + construction_sections(7, 1)
    hi_all = max(int(s["cols"][-1]) * 6 + 2 for s in secs)
    spf = spf_table(hi_all + 2)
    omtab = omega_table(hi_all + 2, spf)
    say(f"omega table to {hi_all} in {time.time() - t0:.0f} s")

    # (a) identity gate + (c) off-axis rate, finer sections and small construction sections, gears 5, 7, 11, 13
    say("\n## (a) symmetric F_2 pairs through g (= prime pairs {6m-i, 6m+i} in the cofactor range) and (c) off-axis rate")
    say("| section | g | axes | through-g pairs (sum) | axes with a radius-1 pair (twins m) | exposure (primes on one side, sum) | through rate | off-axis hits | off-axis symmetric | off rate | F_2 density | z |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tab_a = []
    for s in secs:
        lo, hi = 6 * int(s["cols"][0]) - 2, 6 * int(s["cols"][-1]) + 1   # members in [lo+1, hi]
        n = np.arange(lo + 1, hi + 1, dtype=np.int64)
        n = n[(n % 6 == 1) | (n % 6 == 5)]
        dens = float((omtab[n] == 2).mean())
        for g in [5, 7, 11, 13]:
            if g not in s["gears"]:
                continue
            recs = mirror_census(lo, hi, g, omtab, j=2, off_axis=True)
            # second, independent count of the through-g pairs: directly on the members of the section divisible by g
            direct = 0
            for m_rec in recs:
                A = 12 * g * m_rec["m"]
                x = np.arange(lo + 1, hi + 1, dtype=np.int64)
                x = x[(x % g == 0) & ((x % 6 == 1) | (x % 6 == 5))]
                x = x[(A - x > lo) & (A - x <= hi) & (x < A - x)]
                direct += int(((omtab[x] == 2) & (omtab[A - x] == 2)).sum())
            thr = sum(r["through"] for r in recs)
            tw = sum(1 for r in recs if r["twin1"])
            expo = sum(r["exposure"] for r in recs)
            oh = sum(r.get("off_hits", 0) for r in recs)
            osy = sum(r.get("off_sym", 0) for r in recs)
            rate = osy / oh if oh else float("nan")
            z = (osy - oh * dens) / np.sqrt(oh * dens * (1 - dens)) if oh else float("nan")
            row = dict(section=s["name"], g=g, axes=len(recs), through=thr, direct=direct, twins=tw, exposure=expo,
                       off_hits=oh, off_sym=osy, off_rate=rate, density=dens, z=float(z))
            tab_a.append(row)
            say(f"| {s['name']} | {g} | {len(recs)} | {thr} (direct {direct}) | {tw} | {expo} | {thr / expo if expo else 0:.3f} | {oh} | {osy} | {rate:.3f} | {dens:.3f} | {z:+.2f} |")
    results["a_c"] = tab_a

    # (b) growth along the chain from base 3 and over the machine sights, through g = 5, F_2 and F_3
    say("\n## (b) growth: through g = 5, per axis 6.5.m with both cofactors in range; F_2 (from prime pairs) and F_3 (from F_2 pairs)")
    say("| range | axes | F_2 pairs total | mean per axis | max per axis | axes with radius-1 (twin m) | radius-1 share of pairs | F_3 pairs total | mean | axes with radius-1 (both-F_2 m) | radius-1 share |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    big = construction_sections(3, 3)
    ranges = [(s["name"], 6 * int(s["cols"][0]) - 2, 6 * int(s["cols"][-1]) + 1) for s in big[:2]]
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        qn = nextprime(q)
        ranges.append((f"sight q={q}", 4, qn * qn - 1))
    tab_b = []
    for name, lo, hi in ranges:
        row = dict(range=name)
        for j in (2, 3):
            recs = mirror_census(lo, hi, 5, omtab, j=j, off_axis=False)
            tot = sum(r["through"] for r in recs)
            tw = sum(1 for r in recs if r["twin1"])
            row[f"F{j}"] = dict(axes=len(recs), total=tot, mean=tot / len(recs) if recs else 0,
                                max=max((r["through"] for r in recs), default=0), radius1=tw,
                                radius1_share=tw / tot if tot else 0)
        tab_b.append(row)
        a, b = row["F2"], row["F3"]
        say(f"| {name} | {a['axes']} | {a['total']} | {a['mean']:.2f} | {a['max']} | {a['radius1']} | {a['radius1_share']:.3f} | {b['total']} | {b['mean']:.2f} | {b['radius1']} | {b['radius1_share']:.3f} |")
    # base 3 link 3 with a bigger table: cofactors up to hi/5 = 52M
    s3 = big[2]
    lo3, hi3 = 6 * int(s3["cols"][0]) - 2, 6 * int(s3["cols"][-1]) + 1
    spf3 = spf_table(hi3 // 5 + 10)
    omtab3 = omega_table(hi3 // 5 + 10, spf3)
    say(f"omega table to {hi3 // 5} in {time.time() - t0:.0f} s")
    # the 260M section: radius capped at 601 (100 columns each side in cofactor space), every 97th axis;
    # the same cap on links 1 and 2 for a like-for-like growth row
    say("\n## (b') growth with the radius capped at i <= 601 (like-for-like), through g = 5")
    say("| range | axes sampled | F_2 pairs | mean per axis | axes with radius-1 | radius-1 share | F_3 pairs | mean | radius-1 share |")
    say("|---|---|---|---|---|---|---|---|---|")
    for name, lo, hi, ot, stride in [(big[0]["name"], ranges[0][1], ranges[0][2], omtab, 1), (big[1]["name"], ranges[1][1], ranges[1][2], omtab, 1),
                                     (s3["name"], lo3, hi3, omtab3, 97)]:
        row = dict(range=name + " (cap 601)")
        for j in (2, 3):
            recs = mirror_census(lo, hi, 5, ot, j=j, off_axis=False, rcap=601, stride=stride)
            tot = sum(r["through"] for r in recs)
            tw = sum(1 for r in recs if r["twin1"])
            row[f"F{j}"] = dict(axes=len(recs), total=tot, mean=tot / len(recs) if recs else 0,
                                max=max((r["through"] for r in recs), default=0), radius1=tw, radius1_share=tw / tot if tot else 0)
        tab_b.append(row)
        a, b = row["F2"], row["F3"]
        say(f"| {row['range']} | {a['axes']} | {a['total']} | {a['mean']:.2f} | {a['radius1']} | {a['radius1_share']:.3f} | {b['total']} | {b['mean']:.2f} | {b['radius1_share']:.3f} |")
    results["b"] = tab_b

    # the radial profile of F_2 symmetry through 5 in base 3 link 2 and link 3: pairs at radius i (i <= 61), summed over axes
    say("\n## radial profile through g = 5: symmetric F_2 pairs at radius i, summed over the axes of the section")
    for name, lo, hi, ot, stride in [(big[1]["name"], ranges[1][1], ranges[1][2], omtab, 1), (s3["name"], lo3, hi3, omtab3, 97)]:
        clo, chi = lo // 5 + 1, hi // 5
        prof = {}
        axes = 0
        for m in range((clo + 1) // 6 + 1, chi // 6 + 1, stride):
            c = 6 * m
            rmax = min(c - clo, chi - c)
            if rmax < 1:
                continue
            axes += 1
            for i in [1, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43, 47, 49, 53, 55, 59, 61]:
                if i > rmax:
                    break
                if omtab is not None and ot[c - i] == 1 and ot[c + i] == 1:
                    prof[i] = prof.get(i, 0) + 1
        say(name, "axes", axes, "pairs by radius:", prof)
        results[f"profile_{name}"] = dict(axes=axes, prof=prof)

    json.dump(results, open(os.path.join(RES, "mirror.json"), "w"), indent=1, default=int)
    say("done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

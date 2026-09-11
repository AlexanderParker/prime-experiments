"""fd_mirror2.py -- second measurements for the mirror census (fields.md section 6).

(1) P6(b) in its pre-registered form: the UNCAPPED count of symmetric F_2 pairs through g = 5 per axis, on
    40 axes of base 3 link 3 sampled uniformly (every radius i with both cofactors in range), against the
    uncapped means at links 1 and 2 (2.00 and 46.83).
(2) the off-axis deviation: for base 3 link 2, base 5 link 1, base 7 link 1 and g in {5, 7, 11, 13}, the
    off-axis symmetric count against (i) the global-density null (as in fd_mirror.py), (ii) the local-density
    null (density of F_2 among survivors in 40 equal bins of the section, the mirror partner's bin), and
    (iii) split by whether n shares a prime factor with m (the axis's own factors), each with its own null.
Usage: uv run python fd_mirror2.py
"""
import os, json, time
import numpy as np
from math import gcd
from fd_common import *
from fd_mirror import omega_table

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOG = open(os.path.join(RES, "mirror2.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def main():
    t0 = time.time()
    out = {}
    say("# mirror, second measurements", time.ctime())
    # (1)
    s3 = construction_sections(3, 3)[2]
    lo3, hi3 = 6 * int(s3["cols"][0]) - 2, 6 * int(s3["cols"][-1]) + 1
    top = hi3 // 5 + 10
    spf3 = spf_table(top)
    ot = omega_table(top, spf3)
    clo, chi = lo3 // 5 + 1, hi3 // 5
    rng = np.random.default_rng(79)
    m_lo, m_hi = (clo + 1) // 6 + 1, chi // 6
    ms = sorted(rng.integers(m_lo, m_hi, size=40).tolist())
    counts = []
    say("## (1) uncapped symmetric F_2 pairs through 5 per axis, base 3 link 3, 40 sampled axes")
    say("| m | axis 30m | radii available | pairs (prime pairs {6m-i, 6m+i}) | radius-1 (twin m) |")
    say("|---|---|---|---|---|")
    for m in ms:
        c = 6 * m
        rmax = min(c - clo, chi - c)
        i = np.arange(1, rmax + 1, dtype=np.int64)
        i = i[(i % 6 == 1) | (i % 6 == 5)]
        both = (ot[c - i] == 1) & (ot[c + i] == 1)
        n = int(both.sum())
        counts.append(n)
        say(f"| {m} | {30 * m} | {len(i)} | {n} | {bool(both[0]) if len(i) and i[0] == 1 else False} |")
    say(f"mean {np.mean(counts):.1f}, min {min(counts)}, max {max(counts)}; links 1, 2 uncapped means 2.00, 46.83")
    out["uncapped_link3"] = dict(ms=ms, counts=counts, mean=float(np.mean(counts)))

    # (2)
    say("\n## (2) the off-axis symmetric rate against three nulls")
    say("| section | g | off-axis hits | symmetric | global null (z) | local null (z) | hits coprime to m: sym / local null (z) | hits sharing a factor with m: sym / local null (z) |")
    say("|---|---|---|---|---|---|---|---|")
    secs = construction_sections(3, 2)[1:] + construction_sections(5, 1) + construction_sections(7, 1)
    hi_all = max(6 * int(s["cols"][-1]) + 2 for s in secs)
    spf = spf_table(hi_all + 2)
    omt = omega_table(hi_all + 2, spf)
    tab = []
    for s in secs:
        lo, hi = 6 * int(s["cols"][0]) - 2, 6 * int(s["cols"][-1]) + 1
        n_all = np.arange(lo + 1, hi + 1, dtype=np.int64)
        n_all = n_all[(n_all % 6 == 1) | (n_all % 6 == 5)]
        isF2 = omt[n_all] == 2
        dens = float(isF2.mean())
        # local density in 40 equal bins of the number range
        nb = 40
        edges = np.linspace(lo, hi + 1, nb + 1)
        bidx = np.minimum(np.searchsorted(edges, n_all, side="right") - 1, nb - 1)
        dloc = np.array([isF2[bidx == b].mean() if (bidx == b).any() else dens for b in range(nb)])
        for g in [5, 7, 11, 13]:
            cl, ch = lo // g + 1, hi // g
            hits = sym = 0
            null_g = null_l = var_l = 0.0
            cop = dict(hits=0, sym=0, null=0.0, var=0.0)
            sha = dict(hits=0, sym=0, null=0.0, var=0.0)
            for m in range((cl + 1) // 6 + 1, ch // 6 + 1):
                A = 12 * g * m
                nlo, nhi = max(lo + 1, A - hi), min(hi, A - lo - 1)
                if nhi <= nlo:
                    continue
                n = np.arange(nlo, nhi + 1, dtype=np.int64)
                n = n[((n % 6 == 1) | (n % 6 == 5)) & (n % g != 0)]
                on = omt[n] == 2
                nn = n[on]
                mir = A - nn
                onm = omt[mir] == 2
                b = np.minimum(np.searchsorted(edges, mir, side="right") - 1, nb - 1)
                d = dloc[b]
                hits += len(nn)
                sym += int(onm.sum())
                null_g += len(nn) * dens
                null_l += float(d.sum())
                var_l += float((d * (1 - d)).sum())
                # share a factor with m?
                gm = np.array([gcd(int(x), m) > 1 for x in nn]) if m > 1 else np.zeros(len(nn), dtype=bool)
                for dct, sel in ((cop, ~gm), (sha, gm)):
                    dct["hits"] += int(sel.sum())
                    dct["sym"] += int(onm[sel].sum())
                    dct["null"] += float(d[sel].sum())
                    dct["var"] += float((d[sel] * (1 - d[sel])).sum())
            zg = (sym - null_g) / np.sqrt(hits * dens * (1 - dens)) if hits else float("nan")
            zl = (sym - null_l) / np.sqrt(var_l) if var_l > 0 else float("nan")
            zc = (cop["sym"] - cop["null"]) / np.sqrt(cop["var"]) if cop["var"] > 0 else float("nan")
            zs = (sha["sym"] - sha["null"]) / np.sqrt(sha["var"]) if sha["var"] > 0 else float("nan")
            row = dict(section=s["name"], g=g, hits=hits, sym=sym, null_global=null_g, z_global=float(zg), null_local=null_l,
                       z_local=float(zl), coprime=cop, sharing=sha, z_coprime=float(zc), z_sharing=float(zs))
            tab.append(row)
            say(f"| {s['name']} | {g} | {hits} | {sym} | {null_g:.1f} ({zg:+.2f}) | {null_l:.1f} ({zl:+.2f}) | "
                f"{cop['sym']} / {cop['null']:.1f} ({zc:+.2f}) | {sha['sym']} / {sha['null']:.1f} ({zs:+.2f}) |")
    out["off_axis"] = tab
    json.dump(out, open(os.path.join(RES, "mirror2.json"), "w"), indent=1, default=int)
    say("done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

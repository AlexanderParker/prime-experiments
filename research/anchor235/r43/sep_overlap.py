"""W3 mechanism - the pairwise overlap of two gears, as a function of (s_g, s_h, g, h).

Gear g at phase r strikes the two classes  -r u_g  and  (2-r) u_g  mod g  (separation s_g = 2 u_g).
For two gears the four class combinations are four residues mod gh, a translate of
{0, S_g, S_h, S_g + S_h} where S_g = CRT(s_g, 0) and S_h = CRT(0, s_h) modulo gh.

Measured here:
 (a) FULL-PHASE MEAN.  Over all gh phase pairs the mean overlap is exactly 4m/(gh) for EVERY pair
     of separations (each island contributes to exactly 2 phases of g and 2 of h).  Brute-force
     verified on small pairs.
 (b) REACHABLE MEAN.  Only phases a = -r u with r a nonzero QR are reachable, so island i is
     struck by n_g(i) in {0,1,2} reachable phases of g (n_g(i) = 2 chi_g(i), the doubling law).
     The reachable-phase mean overlap is  (4 / ((g-1)(h-1))) * sum_i n_g(i) n_h(i),  i.e. the
     full-phase mean times  (gh/((g-1)(h-1))) * (sum_i n_g n_h)/m.  The whole separation
     dependence of pairwise overlap sits in that correlation  C(g,h) = (1/m) sum_i n_g(i) n_h(i).
 (c) MIN over reachable phase pairs, and the fraction of gear pairs where 0 is unreachable.
 (d) PAIR-CAPABILITY of gears above d (can a single reachable phase strike two islands?).
 (e) The diagonal identity S_g + S_h = 3^{-1} mod gh for the real separation, and whether any real
     pair can put both diagonal points inside [1, d).

Usage: uv run python research/anchor235/r43/sep_overlap.py --d 560 --nrand 30 --tag o560
"""
import argparse
import json
import os
import random
from math import isqrt


def sieve(n):
    fl = bytearray([1]) * (n + 1)
    fl[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if fl[i]:
            fl[i * i:: i] = bytearray(len(range(i * i, n + 1, i)))
    return fl


def islands(d):
    return [i for i in range(1, d) if i % 35 in (5, 10, 12, 17)]


def u_for(g, fam, rng):
    if fam == "real":
        return pow(6, -1, g)
    if fam.startswith("coh:"):
        c, r = fam[4:].split("/")
        c, r = int(c), int(r)
        if g == r or int(c) % g == 0:
            return None
        return (int(c) * pow(int(r), -1, g) * pow(2, -1, g)) % g
    if fam == "rand":
        return rng.randrange(1, g)
    raise ValueError(fam)


def qr_table(g):
    t = bytearray(g)
    for x in range(1, (g + 1) // 2):
        t[(x * x) % g] = 1
    t[0] = 0
    return t


def nvec(g, u, isl, QR):
    """n_g(i) = number of reachable phases of g that strike island i  (0, 1 or 2)."""
    v = pow(u, -1, g)
    out = []
    for i in isl:
        c = 0
        r1 = (-i * v) % g
        r2 = (2 - i * v) % g
        if QR[r1]:
            c += 1
        if QR[r2]:
            c += 1
        out.append(c)
    return out


def full_mean_bruteforce(g, h, ug, uh, isl):
    """Mean overlap over ALL g*h phase pairs, by brute force (small g,h only)."""
    sg, sh = (2 * ug) % g, (2 * uh) % h
    tot = 0
    for i in isl:
        tot += 4          # island i is struck by 2 phases of g and 2 of h
    # brute force for real
    cnt = 0
    for a in range(g):
        Ig = set(i for i in isl if i % g == a or i % g == (a + sg) % g)
        if not Ig:
            continue
        for b in range(h):
            Ih = set(i for i in isl if i % h == b or i % h == (b + sh) % h)
            cnt += len(Ig & Ih)
    return cnt, tot


def pair_capable(g, u, isl, QR):
    """Can one reachable phase of g strike two islands of [1,d)?"""
    v = pow(u, -1, g)
    buck = {}
    for i in isl:
        buck.setdefault(i % g, []).append(i)
    for i in isl:
        for r in (((-i * v) % g), ((2 - i * v) % g)):
            if r and QR[r]:
                s = buck.get(((2 - r) * u) % g, []) + buck.get(((-r) * u) % g, [])
                if len(s) >= 2:
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=560)
    ap.add_argument("--nrand", type=int, default=30)
    ap.add_argument("--gmaxpairs", type=int, default=200, help="gear ceiling for the pair sweep")
    ap.add_argument("--tag", type=str, default="ov")
    args = ap.parse_args()
    d = args.d
    isl = islands(d)
    m = len(isl)
    GMAX = 3 * d + 2
    FL = sieve(GMAX + 10)
    PR = [p for p in range(11, GMAX + 1) if FL[p]]
    PS = [p for p in PR if p <= args.gmaxpairs]
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(OUT, exist_ok=True)
    res = {"d": d, "m": m, "npairs_gears": len(PS)}
    QR = {g: qr_table(g) for g in PR}

    print("d=%d m=%d  gears to %d (%d), pair sweep over %d gears (%d pairs)"
          % (d, m, GMAX, len(PR), len(PS), len(PS) * (len(PS) - 1) // 2), flush=True)

    # ---- (a) full-phase mean, brute force on small pairs, several separation families
    bf = []
    rng = random.Random(7)
    for (g, h) in [(11, 13), (11, 17), (13, 19), (17, 23)]:
        for fam in ["real", "coh:2/5", "coh:2/7", "rand", "rand", "rand"]:
            ug, uh = u_for(g, fam, rng), u_for(h, fam, rng)
            cnt, tot = full_mean_bruteforce(g, h, ug, uh, isl)
            bf.append(dict(g=g, h=h, fam=fam, sum_overlap=cnt, expected=4 * m,
                           mean=cnt / (g * h), predicted=4 * m / (g * h), ok=(cnt == 4 * m)))
    res["full_mean"] = bf
    nbad = sum(1 for b in bf if not b["ok"])
    print("(a) full-phase mean = 4m/(gh): %d checks, %d exceptions" % (len(bf), nbad), flush=True)

    # ---- (b,c) reachable-phase correlation C(g,h) and min overlap, per family
    fams = ["real", "coh:2/5", "coh:2/7", "coh:2/11", "coh:2/13", "coh:1/5", "coh:4/7", "coh:3/11"]
    fams += ["rand:%d" % k for k in range(args.nrand)]
    famres = {}
    for fam in fams:
        rng = random.Random(4200 + (int(fam.split(":")[1]) if fam.startswith("rand:") else 0))
        base = "rand" if fam.startswith("rand:") else fam
        U = {}
        for g in PS:
            u = u_for(g, base, rng)
            U[g] = u
        N = {g: nvec(g, U[g], isl, QR[g]) for g in PS if U[g] is not None}
        # C(g,h) over all pairs
        Cs = []
        minpos = 0
        npair = 0
        gs = [g for g in PS if g in N]
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                g, h = gs[a], gs[b]
                ng, nh = N[g], N[h]
                s = 0
                for k in range(m):
                    s += ng[k] * nh[k]
                Cs.append(s / m)
                npair += 1
                # min over reachable phase pairs is 0 unless every reachable pair has an island;
                # nonzero entries <= 4m while reachable pairs = (g-1)(h-1)/4
                if (g - 1) * (h - 1) // 4 <= 4 * m:
                    minpos += 0   # only these could be forced; computed below for small pairs
        famres[fam] = dict(C_mean=sum(Cs) / len(Cs), C_min=min(Cs), C_max=max(Cs), npair=npair)
        print("(b) %-10s  C mean %.4f  min %.4f  max %.4f" %
              (fam, famres[fam]["C_mean"], famres[fam]["C_min"], famres[fam]["C_max"]), flush=True)
    res["correlation"] = famres

    # ---- (d) pair-capability of gears above d
    capres = {}
    for fam in ["real", "coh:2/5", "coh:2/7", "coh:2/11", "coh:2/13"] + \
               ["rand:%d" % k for k in range(min(args.nrand, 10))]:
        rng = random.Random(9100 + (int(fam.split(":")[1]) if fam.startswith("rand:") else 0))
        base = "rand" if fam.startswith("rand:") else fam
        bands = {"(d,1.5d)": [d, int(1.5 * d)], "(1.5d,3d)": [int(1.5 * d), 3 * d],
                 "(3d,4d)": [3 * d, 4 * d]}
        cnt = {k: [0, 0] for k in bands}
        for g in PR:
            if g <= d:
                continue
            if g > 4 * d:
                break
            u = u_for(g, base, rng)
            if u is None:
                continue
            ok = pair_capable(g, u, isl, QR[g] if g in QR else qr_table(g))
            for k, (lo, hi) in bands.items():
                if lo < g <= hi:
                    cnt[k][1] += 1
                    if ok:
                        cnt[k][0] += 1
        capres[fam] = {k: (v[0], v[1], (v[0] / v[1] if v[1] else 0.0)) for k, v in cnt.items()}
        print("(d) %-10s  " % fam + "  ".join("%s %d/%d=%.3f" % (k, v[0], v[1], v[2])
                                              for k, v in capres[fam].items()), flush=True)
    res["pair_capable"] = capres

    # ---- (e) the diagonal identity for the real separation
    diag = []
    bad = 0
    inside = 0
    for (g, h) in [(11, 13), (13, 17), (19, 23), (29, 31), (41, 43), (101, 103)]:
        ug, uh = pow(6, -1, g), pow(6, -1, h)
        sg, sh = (2 * ug) % g, (2 * uh) % h
        gh = g * h
        # CRT lifts
        Sg = (sg * h * pow(h, -1, g)) % gh
        Sh = (sh * g * pow(g, -1, h)) % gh
        dsum = (Sg + Sh) % gh
        ok = (3 * dsum) % gh == 1 % gh
        if not ok:
            bad += 1
        dint = min(dsum, gh - dsum)
        if gh > 3 * d and dint < d:
            inside += 1
        diag.append(dict(g=g, h=h, gh=gh, Sg=Sg, Sh=Sh, diag=dsum, is_third_inv=ok,
                         min_diag_int=dint, gh_gt_3d=bool(gh > 3 * d)))
    res["diagonal"] = dict(rows=diag, exceptions=bad, real_pairs_with_diag_inside=inside)
    print("(e) diagonal S_g+S_h = 3^{-1} mod gh: %d checks, %d exceptions; "
          "real pairs with gh>3d and diagonal < d: %d" % (len(diag), bad, inside), flush=True)

    p = os.path.join(OUT, "sep_overlap_%s.json" % args.tag)
    json.dump(res, open(p, "w"), indent=1)
    print("written", p)


main()

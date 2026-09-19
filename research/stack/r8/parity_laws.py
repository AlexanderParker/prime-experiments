"""Task 2 (L1-L8 at P-, Q-, M'-centres) and task 4 (first-rung index) for centres s <= S_MAX.
Stretch members are factored per centre by a segmented sieve over the offsets j, |j| <= 2c-1,
s = 6c: gear p strikes offset j of member s^2+6j-1 iff 6j = -(s^2-1) mod p, and of member
s^2+6j+1 iff 6j = -(s^2+1) mod p. After dividing out every prime p <= s+1 what is left is 1 or
one prime (members are < (s+1)^2). Usage: python task2_laws.py [S_MAX]."""
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S_MAX = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100000
GEARS = (5, 7, 11, 13)
TWIN_CONST = 1.3203


def inv6_mod(ps):
    """6 * inv6 = 1 mod p for primes p >= 5 (p = 1 mod 6 -> (5p+1)/6, p = 5 mod 6 -> (p+1)/6)."""
    return np.where(ps % 6 == 1, (5 * ps + 1) // 6, (ps + 1) // 6)


def hits(s2pm1, J, ps, inv6):
    """for every prime in ps: first offset j >= -J with 6j = -(s2pm1) mod p, and the number of
    offsets in [-J, J] in that class."""
    r = ((-s2pm1) % ps) * inv6 % ps
    start = -J + ((r + J) % ps)
    n = np.where(start <= J, (J - start) // ps + 1, 0)
    return start, n


def strike_all(rem, om, start, n, ps):
    """divide every struck member by its striking prime, with multiplicity; om counts factors."""
    total = int(n.sum())
    idx = np.repeat(np.arange(len(ps)), n)
    within = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    J = (len(rem) - 1) // 2
    pos = start[idx] + ps[idx] * within + J
    pp = ps[idx]
    while len(pos):
        np.add.at(om, pos, 1)
        prod = np.ones(len(rem), dtype=np.int64)
        np.multiply.at(prod, pos, pp)
        rem //= prod
        keep = rem[pos] % pp == 0
        pos, pp = pos[keep], pp[keep]


def factor_stretch(s, ps_all, inv6_all):
    c = s // 6
    J = 2 * c - 1
    j = np.arange(-J, J + 1, dtype=np.int64)
    s2 = s * s
    A = s2 + 6 * j - 1
    B = s2 + 6 * j + 1
    m = np.searchsorted(ps_all, s + 1, side="right")
    ps, inv6 = ps_all[:m], inv6_all[:m]
    remA, remB = A.copy(), B.copy()
    omA = np.zeros(len(j), dtype=np.int16)
    omB = np.zeros(len(j), dtype=np.int16)
    stA, nA = hits(s2 - 1, J, ps, inv6)
    stB, nB = hits(s2 + 1, J, ps, inv6)
    strike_all(remA, omA, stA, nA, ps)
    strike_all(remB, omB, stB, nB, ps)
    omA = omA + (remA > 1)
    omB = omB + (remB > 1)
    # roughness to 61: struck by no gear <= 61
    rough = np.ones(len(j), dtype=bool)
    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61):
        rough &= (A % p != 0) & (B % p != 0)
    return j, A, B, omA, omB, rough, ps, nA, nB


def sets_from_omega(omA, omB):
    pa, pb = omA == 1, omB == 1
    P = pa & pb
    Q = (omA % 2 == 0) & (omB % 2 == 0)
    Mp = (omA % 2 == 1) & (omB % 2 == 1) & ~pa & ~pb
    return P, Q, Mp


def first_index(rung, rough, j):
    """1-based rank of the first rung among 61-rough offsets ordered by |j| (j < 0 first on ties)."""
    order = np.lexsort((j, np.abs(j)))
    order = order[rough[order]]
    hit = np.nonzero(rung[order])[0]
    return int(hit[0]) + 1 if len(hit) else None


def main():
    t0 = time.time()
    d = np.load(os.path.join(HERE, "sieve.npz"))
    spf, om = d["spf"], d["om"]
    sets = {"P": d["P"], "Q": d["Q"], "M'": d["Mp"], "Q61": d["Q61"]}
    K = len(sets["P"])
    kk = np.arange(1, K + 1, dtype=np.int64)
    Nspf = len(spf) - 1
    ar = np.arange(Nspf + 1, dtype=np.int32)
    ps_all = ar[(spf == ar) & (ar >= 5)].astype(np.int64)
    inv6_all = inv6_mod(ps_all)

    # ---------- L2, global, all centres s = 6k <= 10^7 (struck classes come from s alone) ----------
    print(f"=== L2 universal clearance: classes j mod g never struck, centres of each set, s <= {6*K} ===")
    print(f"{'set':>5} {'g':>3} {'#centres':>9} {'never-struck classes':>24} {'struck-class counts (class: #centres struck)'}")
    L2 = {}
    for name, flags in sets.items():
        ks = kk[flags]
        s = 6 * ks
        for g in GEARS:
            inv = int(inv6_mod(np.array([g]))[0])
            rA = ((-(s * s - 1)) % g) * inv % g
            rB = ((-(s * s + 1)) % g) * inv % g
            cnt = np.bincount(np.concatenate([rA, rB]), minlength=g)
            never = [c for c in range(g) if cnt[c] == 0]
            L2[(name, g)] = never
            print(f"{name:>5} {g:>3} {len(s):>9} {str(never):>24} {' '.join(f'{c}:{cnt[c]}' for c in range(g))}")
        if name == "P":
            for g in GEARS:
                print(f"    P centres with {g} | s^2-1: {s[(s*s-1) % g == 0].tolist()}")
        # split of Q-centres by whether g | s^2-1
        if name in ("Q", "M'"):
            for g in GEARS:
                inv = int(inv6_mod(np.array([g]))[0])
                div = (s * s - 1) % g == 0
                for lab, sel in (("g | s^2-1", div), ("g !| s^2-1", ~div)):
                    ss = s[sel]
                    rA = ((-(ss * ss - 1)) % g) * inv % g
                    rB = ((-(ss * ss + 1)) % g) * inv % g
                    cnt = np.bincount(np.concatenate([rA, rB]), minlength=g)
                    never = [c for c in range(g) if cnt[c] == 0]
                    print(f"{name:>5} {g:>3} {len(ss):>9} {str(never):>24} [{lab}]")

    # ---------- L7 parents unique, global: stretches of consecutive centres are disjoint ----------
    print(f"\n=== L7 parents unique (a rung lies in the stretch of at most one centre), s <= {6*K} ===")
    for name, flags in sets.items():
        s = 6 * kk[flags]
        lo, hi = (s - 1) ** 2, (s + 1) ** 2
        overlaps = int((hi[:-1] > lo[1:]).sum())
        gap = int((lo[1:] - hi[:-1]).min())
        print(f"  {name:>4}: {len(s)} centres, overlapping consecutive stretches = {overlaps}, "
              f"min gap between stretches = {gap}")

    # ---------- L8 region law, global ----------
    print(f"\n=== L8 region law: centres of each set between consecutive prime squares (p^2, q^2), p >= 5 ===")
    pr = ar[(spf == ar) & (ar >= 5) & (ar <= math.isqrt(6 * K))].astype(np.int64)
    dens = {}
    for name, flags in sets.items():
        s = 6 * kk[flags]
        dens[name] = len(s) / K
    for name, flags in sets.items():
        s = 6 * kk[flags]
        cnt = np.searchsorted(s, pr[1:] ** 2) - np.searchsorted(s, pr[:-1] ** 2)
        length = pr[1:] ** 2 - pr[:-1] ** 2
        gap = pr[1:] - pr[:-1]
        if name == "P":
            pred = TWIN_CONST * length / (2 * np.log(pr[:-1])) ** 2
            law = "1.3203 (q^2-p^2)/ln^2(p^2)"
        else:
            pred = dens[name] * length / 6
            law = f"d (q^2-p^2)/6, d = {dens[name]:.5f}"
        big = pr[:-1] >= 100  # regions above 10^4 for the ratio
        print(f"  {name:>4}: law {law}; regions p in [5, {pr[-2]}], {len(cnt)} regions")
        print(f"        overall sum(count)/sum(pred) = {cnt.sum()/pred.sum():.4f} "
              f"(p >= 100: {cnt[big].sum()/pred[big].sum():.4f})")
        print(f"        {'gap':>4} {'#regions':>8} {'mean count':>11} {'mean pred':>10} {'ratio':>7}")
        for gp in sorted(set(gap.tolist())):
            sel = (gap == gp) & big
            if sel.sum() == 0:
                continue
            print(f"        {gp:>4} {int(sel.sum()):>8} {cnt[sel].mean():>11.2f} {pred[sel].mean():>10.2f} "
                  f"{cnt[sel].sum()/pred[sel].sum():>7.4f}")

    # ---------- per-centre stretch computations, s <= S_MAX ----------
    print(f"\n=== per-centre laws L1, L3, L4, L5, L6 and task 4, centres s <= {S_MAX} ===")
    S_MIN = 100
    centres = {name: (6 * kk[flags])[(6 * kk[flags] <= S_MAX) & (6 * kk[flags] >= S_MIN)] for name, flags in sets.items()}
    print(f"  (centres restricted to {S_MIN} <= s <= {S_MAX})")
    for name, s in centres.items():
        print(f"  {name}: {len(s)} centres")
    # verify factoring against the global table on the first centres (members <= 10^7)
    checked = 0
    for name in ("P", "Q"):
        for s in centres[name][:40]:
            if (s + 1) ** 2 > Nspf:
                break
            j, A, B, omA, omB, rough, ps, nA, nB = factor_stretch(int(s), ps_all, inv6_all)
            assert (omA == om[A]).all() and (omB == om[B]).all(), s
            r2 = (spf[A] > 61) & (spf[B] > 61)
            assert (rough == r2).all(), s
            checked += 1
    print(f"  factoring cross-checked against the global table at {checked} centres: exact")

    rec = {name: defaultdict(list) for name in centres}
    L3_ok = L3_tot = 0
    for name, ss in centres.items():
        for s in ss:
            s = int(s)
            j, A, B, omA, omB, rough, ps, nA, nB = factor_stretch(s, ps_all, inv6_all)
            Pr, Qr, Mr = sets_from_omega(omA, omB)
            r = rec[name]
            r["s"].append(s)
            r["TP"].append(int(Pr.sum()))
            r["TQ"].append(int(Qr.sum()))
            r["TM"].append(int(Mr.sum()))
            r["TQ61"].append(int((Qr & rough).sum()))
            # L1: struck classes per gear g <= 13 - count distinct residues of struck j mod g
            for g in GEARS:
                gi = np.searchsorted(ps, g)
                struck = set()
                for memb in (A, B):
                    struck |= set((j[memb % g == 0] % g).tolist())
                r[f"L1_{g}"].append(len(struck))
            # L3: for rungs of the centre's own set, every gear g | j (g >= 5, g <= 61) has s' = s^2 mod g
            own = {"P": Pr, "Q": Qr, "M'": Mr, "Q61": Qr & rough}[name]
            jo = j[own]
            jo = jo[jo != 0]
            for g in ps[ps <= 61]:
                g = int(g)
                sel = jo % g == 0
                L3_tot += int(sel.sum())
                L3_ok += int((((s * s + 6 * jo[sel]) - s * s) % g == 0).sum())
            # L5: strikes by the prime factors of s-1 and s+1 (the twin gears at a twin centre)
            fac = set()
            for x in (s - 1, s + 1):
                while x > 1:
                    p = int(spf[x]); fac.add(p); x //= p
            fac = sorted(fac)
            n_off = 0
            for p in fac:
                n_off += int(((A % p) == 0).sum() + ((B % p) == 0).sum())
            r["L5_gears"].append(len(fac))
            r["L5_strikes"].append(n_off)  # at a twin centre: gears s-1 and s+1 strike j = 0 only -> 2
            # L6: primes p in (2s/3, s+1]: struck offsets per prime, both members
            top = ps > 2 * s / 3
            per = nA[top] + nB[top]
            r["L6_min"].append(int(per.min()))
            r["L6_max"].append(int(per.max()))
            # task 4: first rung index among 61-rough offsets, for P-rungs and Q-rungs
            r["idxP"].append(first_index(Pr, rough, j))
            r["idxQ"].append(first_index(Qr, rough, j))
            r["nrough"].append(int(rough.sum()))
        print(f"  {name}: done, {time.time()-t0:.0f}s")

    # ---------- L1 ----------
    print("\n=== L1: number of distinct struck residue classes j mod g, per gear, over all centres ===")
    for name, r in rec.items():
        print(f"  {name:>4}: " + ", ".join(f"g={g}: {sorted(set(r[f'L1_{g}']))}" for g in GEARS))

    # ---------- L3 ----------
    print(f"\n=== L3 phase lock: rung s' = s^2 + 6j, gear g | j (5 <= g <= 61): s' = s^2 mod g ===")
    print(f"  checks = {L3_tot}, satisfied = {L3_ok}, fraction = {L3_ok/L3_tot:.6f}")

    # ---------- L4 ----------
    print("\n=== L4: rung counts, cross table (rows: centre set; cols: rung set), mean count / law ===")
    print("  laws: P-rungs 1.3203 s/ln^2 s; Q-rungs d_Q (4c-1); M'-rungs d_M' (4c-1) with d the "
          "density among columns near s^2 (measured at the same heights from the global table)")
    # densities of Q, M' among columns as a function of height, from the global table (bins)
    def dens_at(flags, x):
        # density among columns with 6k+1 in [x/2, x]
        x = max(x, 1e5)
        lo, hi = max(1, int(x / 2) // 6), min(K, int(x) // 6)
        return flags[lo:hi].mean()
    for name, r in rec.items():
        s = np.array(r["s"], dtype=float)
        ncol = 4 * (s / 6) - 1
        for col, key in (("P", "TP"), ("Q", "TQ"), ("M'", "TM")):
            T = np.array(r[key], dtype=float)
            if col == "P":
                law = TWIN_CONST * s / np.log(s) ** 2
            else:
                flags = sets[col]
                dd = np.array([dens_at(flags, min(x * x, 6 * K)) for x in s])
                law = dd * ncol
            ratio = T / law
            print(f"  {name:>4}-centres, {col:>2}-rungs: n={len(s)}, mean T={T.mean():.3f}, mean T/law={ratio.mean():.4f} "
                  f"+- {ratio.std(ddof=1)/math.sqrt(len(s)):.4f}")
    print("\n  L4 by residue of s mod g (own-set rungs; ratio T/law, mean +- SE per class; "
          "max |dev| in SE units)")
    for name, r in rec.items():
        s = np.array(r["s"], dtype=float)
        ncol = 4 * (s / 6) - 1
        key = {"P": "TP", "Q": "TQ", "M'": "TM", "Q61": "TQ61"}[name]
        T = np.array(r[key], dtype=float)
        if name == "P":
            law = TWIN_CONST * s / np.log(s) ** 2
        else:
            dd = np.array([dens_at(sets[name], min(x * x, 6 * K)) for x in s])
            law = dd * ncol
        ratio = T / law
        # remove the drift of the density with height: divide by the mean ratio in the s-bin (8 bins by rank)
        order = np.argsort(s)
        nb = 8 if len(s) >= 80 else 1
        for b in range(nb):
            sel = order[b * len(s) // nb:(b + 1) * len(s) // nb]
            ratio[sel] = ratio[sel] / ratio[sel].mean()
        for g in GEARS:
            res = (np.array(r["s"]) % g)
            cells = []
            worst = 0
            for a in range(g):
                sel = res == a
                if sel.sum() < 2:
                    cells.append(f"{a}:-")
                    continue
                m = ratio[sel].mean()
                se = ratio[sel].std(ddof=1) / math.sqrt(sel.sum())
                dev = abs(m - ratio.mean()) / se if se > 0 else 0
                worst = max(worst, dev)
                cells.append(f"{a}:{m:.3f}+-{se:.3f}(n={sel.sum()})")
            print(f"  {name:>4} g={g:>2}: " + " ".join(cells) + f"  | max dev = {worst:.2f} SE")
        if name in ("Q", "M'"):
            for g in GEARS:
                sarr = np.array(r["s"])
                div = (sarr * sarr - 1) % g == 0
                out = []
                for lab, sel in ((f"{g}|s^2-1", div), (f"{g}!|s^2-1", ~div)):
                    if sel.sum() >= 2:
                        out.append(f"{lab}: {ratio[sel].mean():.4f}+-{ratio[sel].std(ddof=1)/math.sqrt(sel.sum()):.4f} (n={sel.sum()})")
                print(f"  {name:>4} split: " + "; ".join(out))

    # ---------- L5 ----------
    print("\n=== L5: strikes on the stretch by the prime factors of s-1 and s+1 ===")
    for name, r in rec.items():
        ng = np.array(r["L5_gears"]); st = np.array(r["L5_strikes"])
        print(f"  {name:>4}: #distinct prime factors of (s-1)(s+1): min {ng.min()} max {ng.max()} mean {ng.mean():.2f}; "
              f"struck offsets (both members, with j=0): min {st.min()} max {st.max()} mean {st.mean():.2f}; "
              f"centres with exactly 2 strikes (j = 0 only): {int((st == 2).sum())}/{len(st)}")

    # ---------- L6 ----------
    print("\n=== L6 top band: primes p in (2s/3, s+1], struck offsets per prime (both members) ===")
    for name, r in rec.items():
        print(f"  {name:>4}: min over centres of min = {min(r['L6_min'])}, max of max = {max(r['L6_max'])}")

    # ---------- task 4 ----------
    print(f"\n=== task 4: first rung index among 61-rough offsets (rank by |j|), s <= {S_MAX} ===")
    for name, r in rec.items():
        s = np.array(r["s"], dtype=float)
        for key, lab in (("idxP", "P-rungs"), ("idxQ", "Q-rungs")):
            idx = np.array([np.nan if v is None else v for v in r[key]], dtype=float)
            ok = ~np.isnan(idx)
            ratio = idx[ok] / np.log(s[ok]) ** 2
            am = int(np.argmax(ratio))
            print(f"  {name:>4}-centres, {lab}: n={len(s)}, no rung at {int((~ok).sum())} centres, "
                  f"mean index {idx[ok].mean():.2f}, max index {int(idx[ok].max())}, "
                  f"max index/ln^2 s = {ratio.max():.4f} at s = {int(s[ok][am])} (index {int(idx[ok][am])}), "
                  f"mean index/ln^2 s = {ratio.mean():.4f}")
            # tail against a geometric law with the measured rung density among rough offsets
            dens = (np.array(r["TP"]).sum() if key == "idxP" else np.array(r["TQ61"]).sum()) / np.array(r["nrough"]).sum()
            tail = []
            for n in (10, 20, 30, 40, 60):
                obs = int((idx[ok] > n).sum())
                exp = len(s) * (1 - dens) ** n
                tail.append(f">{n}: {obs} obs / {exp:.2f} geo")
            print(f"        rung density among rough offsets {dens:.4f}; tail counts: " + "; ".join(tail))
            if name in ("P", "Q") and key == ("idxP" if name == "P" else "idxQ"):
                smax = int(s[ok][am])
                j, A, B, omA, omB, rough, ps, nA, nB = factor_stretch(smax, ps_all, inv6_all)
                order = np.lexsort((j, np.abs(j)))
                order = order[rough[order]][: int(idx[ok][am])]
                print(f"        detail at s = {smax}: (j, Omega(s^2+6j-1), Omega(s^2+6j+1)) along the rough offsets: "
                      + " ".join(f"({int(j[i])},{int(omA[i])},{int(omB[i])})" for i in order))
        # rough offsets and rung density among them
        nr = np.array(r["nrough"], dtype=float)
        print(f"        61-rough offsets per centre: mean {nr.mean():.1f}; P-rungs/rough = "
              f"{np.array(r['TP']).sum()/nr.sum():.4f}, Q-rungs/rough = {np.array(r['TQ61']).sum()/nr.sum():.4f}")
    np.savez(os.path.join(HERE, f"rec_{S_MAX}.npz"), **{f"{n}__{k}": np.array([-1 if v is None else v for v in vals])
                                                         for n, r in rec.items() for k, vals in r.items()})
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

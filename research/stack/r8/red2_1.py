"""Red lane, script 1: claims 1-4 (tasks A, B, D) - exact, no sampling."""
import bisect
import math
import os
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
C2 = 0.6601618158468695739278121100145  # twin prime constant
TWO_C2 = 2 * C2

NSIEVE = 100_020_002  # (10^4+1)^2 = 100020001


def sieve_bool(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p * p :: p] = False
    return s


def main():
    print(f"sieving primes to {NSIEVE}", file=sys.stderr)
    isp = sieve_bool(NSIEVE)
    k = np.arange(1, NSIEVE // 6, dtype=np.int64)
    u = 6 * k
    m = isp[u - 1] & isp[u + 1]
    twins = u[m].astype(np.int64)
    del k, u, m, isp
    print(f"twin centres <= {NSIEVE}: {len(twins)}", file=sys.stderr)
    np.save(os.path.join(OUT, "twins.npy"), twins)
    tl = twins.tolist()

    print("=" * 70)
    print("PRELIM: twin centre counts")
    for lim in (10**4, 10**6, 7 * 10**6, NSIEVE):
        print(f"  pi_2centres(<= {lim:>12}) = {int(np.searchsorted(twins, lim, 'right')):>8}")

    # ---------------- Task A / claim 1: roots of the product forest to 10^6
    print("=" * 70)
    print("TASK A / CLAIM 1: roots of the product forest, u <= 10^6")
    U = 10**6
    nU = int(np.searchsorted(twins, U, "right"))
    roots = []
    need_nonsix = []          # non-roots not covered by any pair with s = 6
    need_not_prev6 = []       # non-roots covered by s=6 but not by t = largest t with 5(t-1) < u
    for i in range(nU):
        uu = tl[i]
        found = None
        found6 = None
        # s ranges over twin centres with (s-1)^2 < u  (since t >= s)
        for j in range(i):
            s = tl[j]
            if (s - 1) * (s - 1) >= uu:
                break
            # need twin t, s <= t < u, (s-1)(t-1) < u < (s+1)(t+1)
            lo = uu // (s + 1) - 2
            hi = uu // (s - 1) + 2
            a = bisect.bisect_left(tl, max(s, lo), j, i)
            b = bisect.bisect_right(tl, min(uu - 1, hi), j, i)
            for x in range(a, b):
                t = tl[x]
                if t < s or t >= uu:
                    continue
                if (s - 1) * (t - 1) < uu < (s + 1) * (t + 1):
                    if found is None:
                        found = (s, t)
                    if s == 6 and found6 is None:
                        found6 = (s, t)
            if found is not None and found6 is not None:
                break
        if found is None:
            roots.append(uu)
        else:
            if found6 is None:
                need_nonsix.append((uu, found))
    print(f"  roots found: {roots}")
    print(f"  #roots = {len(roots)}   (claim: exactly [6, 12, 18])")
    print(f"  non-roots needing a pair with s != 6: {len(need_nonsix)}")
    if need_nonsix:
        print(f"    first 20: {need_nonsix[:20]}")

    # ---------------- Task B: the multiplier-6 reformulation
    print("=" * 70)
    print("TASK B: u in some W(6,t) <=> 5(t-1) < u < 7(t+1)")
    # For each twin centre u >= 30, let t* = largest twin centre with 5(t-1) < u.
    # Covered by multiplier-6 windows  <=>  u < 7(t*+1).
    fails = []
    for i in range(nU):
        uu = tl[i]
        if uu < 30:
            continue
        # largest t (twin centre, t < u) with 5*(t-1) < u  ->  t < u/5 + 1
        lim = (uu - 1) // 5 + 1
        j = bisect.bisect_right(tl, lim, 0, i) - 1
        while j >= 0 and 5 * (tl[j] - 1) >= uu:
            j -= 1
        if j < 0:
            fails.append((uu, None))
            continue
        ts = tl[j]
        if not (uu < 7 * (ts + 1)):
            # not covered by the maximal such t; check every t
            cov = any(5 * (tl[x] - 1) < uu < 7 * (tl[x] + 1) for x in range(0, i))
            fails.append((uu, ts, cov))
    print(f"  twin centres 30 <= u <= 10^6 where t* fails to cover: {len(fails)}")
    if fails:
        print(f"    first 20: {fails[:20]}")

    # consecutive ratio of twin centres to 10^6
    tarr = twins[:nU]
    ratios = tarr[1:] / tarr[:-1]
    imax = int(np.argmax(ratios))
    print(f"  max consecutive ratio t'/t (t <= 10^6): {ratios[imax]:.6f} at t={tarr[imax]} -> t'={tarr[imax+1]}")
    print("  top 10 ratios:")
    order = np.argsort(-ratios)[:10]
    for o in order:
        o = int(o)
        cov = 5 * (int(tarr[o + 1]) - 1) < 7 * (int(tarr[o]) + 1)
        print(f"    t={tarr[o]:>8} t'={tarr[o+1]:>8}  ratio={ratios[o]:.6f}  windows overlap: {cov}")
    print(f"  #ratios >= 7/5 = 1.4 : {int((ratios >= 1.4).sum())}")
    # exact cover condition between consecutive twin centres t < t':
    #   every integer u in [7(t+1), ...) up to 5(t'-1) is uncovered.  Gap exists iff 5(t'-1) >= 7(t+1)
    gapmask = 5 * (tarr[1:] - 1) >= 7 * (tarr[:-1] + 1)
    print(f"  #consecutive pairs with a genuine uncovered integer gap: {int(gapmask.sum())}")
    if gapmask.any():
        idx = np.nonzero(gapmask)[0][:10]
        for o in idx:
            o = int(o)
            print(f"    uncovered integer interval [{7*(tarr[o]+1)}, {5*(tarr[o+1]-1)}] between t={tarr[o]}, t'={tarr[o+1]}")

    # ---------------- Claim 2: pairs 42 <= s <= t <= 10^4
    print("=" * 70)
    print("CLAIM 2: pairs 42 <= s <= t <= 10^4, rung counts of W(s,t)")
    base = [x for x in tl if 42 <= x <= 10**4]
    print(f"  twin centres in [42, 10^4]: {len(base)}  -> pairs = {len(base)*(len(base)+1)//2}")
    B = np.array(base, dtype=np.int64)
    best = []
    empties = 0
    minc = 10**9
    counts_all = []
    for ii in range(len(B)):
        s = int(B[ii])
        tt = B[ii:]
        lo = (s - 1) * (tt - 1)
        hi = (s + 1) * (tt + 1)
        c = np.searchsorted(twins, hi - 1, "right") - np.searchsorted(twins, lo, "right")
        counts_all.append(c)
        empties += int((c == 0).sum())
        mc = int(c.min())
        if mc < minc:
            minc = mc
        for jj in np.nonzero(c <= 3)[0]:
            best.append((int(c[jj]), s, int(tt[jj])))
    counts_all = np.concatenate(counts_all)
    print(f"  empty windows: {empties}")
    print(f"  minimum rung count: {minc}")
    best.sort()
    print("  all pairs with count <= 3:")
    for c, s, t in best[:40]:
        print(f"    count={c:>2}  (s,t)=({s},{t})   W=({(s-1)*(t-1)}, {(s+1)*(t+1)})")
    print(f"  total pairs={len(counts_all)}  mean count={counts_all.mean():.3f}")

    # ---------------- Claim 3: W(6,t) for twin centres t <= 10^6
    print("=" * 70)
    print("CLAIM 3: W(6,t) = (5(t-1), 7(t+1)) for twin centres t <= 10^6")
    lo6 = 5 * (tarr - 1)
    hi6 = 7 * (tarr + 1)
    n6 = np.searchsorted(twins, hi6 - 1, "right") - np.searchsorted(twins, lo6, "right")
    print(f"  t range: {tarr[0]} .. {tarr[-1]}  ({len(tarr)} twin centres)")
    print(f"  empty W(6,t): {int((n6 == 0).sum())}")
    print(f"  minimum count over all t <= 10^6: {int(n6.min())} at t={int(tarr[int(np.argmin(n6))])}")
    ordn = np.argsort(n6, kind="stable")[:15]
    print("  15 smallest counts:")
    for o in ordn:
        o = int(o)
        print(f"    t={int(tarr[o]):>8}  N_6={int(n6[o]):>3}  W=({int(lo6[o])},{int(hi6[o])})")
    for thr in (30, 42, 108, 1000):
        sub = n6[tarr >= thr]
        st = tarr[tarr >= thr]
        print(f"  restricted to t >= {thr}: min={int(sub.min())} at t={int(st[int(np.argmin(sub))])}")
    np.save(os.path.join(OUT, "n6.npy"), n6)
    np.save(os.path.join(OUT, "t6.npy"), tarr)

    # ---------------- Claim 4 / Task D
    print("=" * 70)
    print("CLAIM 4 / TASK D: st itself a twin centre")

    def r_exact(p):
        # A = Z_p \ {1,-1};  bad pairs ab = +-1 : exactly 2(p-3)
        good = (p - 2) ** 2 - 2 * (p - 3)
        return p * good / (p - 2) ** 3, good, (p - 2) ** 2

    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        r, g, tot = r_exact(p)
        print(f"    r_{p:<3} = {r:.6f}   P(good)={g}/{tot}={g/tot:.6f}   (1-2/p)={1-2/p:.6f}")
    # full product over all p >= 5 with tail
    def primes_upto(n):
        s = np.ones(n + 1, dtype=bool)
        s[:2] = False
        for q in range(2, int(n**0.5) + 1):
            if s[q]:
                s[q * q :: q] = False
        return np.nonzero(s)[0]

    ps = primes_upto(2_000_000)
    ps = ps[ps >= 5].astype(np.float64)
    K_partial = float(np.prod(p * ((p - 2) ** 2 - 2 * (p - 3)) / (p - 2) ** 3))
    # tail  p > 2e6 :  log r_p ~ -(2p-8)/(p-2)^3 ~ -2/p^2 ; sum_{p>X} 1/p^2 ~ 1/(X ln X)
    X = 2_000_000
    tail = math.exp(-2.0 / (X * math.log(X)))
    K = K_partial * tail
    print(f"    K = prod_{{p>=5}} r_p  (p <= 2e6) = {K_partial:.6f}, with tail = {K:.6f}")
    for P in (7, 13, 100, 1000, 10**5):
        sub = ps[ps <= P]
        val = float(np.prod(sub * ((sub - 2) ** 2 - 2 * (sub - 3)) / (sub - 2) ** 3))
        print(f"      partial product p <= {P:>7}: {val:.6f}")

    # observed / expected over the claim-2 pair set
    obs = 0
    exp_gen = 0.0
    diag_obs = 0
    tw_set_limit = int(twins[-1])
    obs_pairs = []
    for ii in range(len(B)):
        s = int(B[ii])
        tt = B[ii:]
        prod = s * tt
        assert prod.max() <= tw_set_limit
        pos = np.searchsorted(twins, prod)
        hit = (pos < len(twins)) & (twins[np.minimum(pos, len(twins) - 1)] == prod)
        obs += int(hit.sum())
        for jj in np.nonzero(hit)[0]:
            obs_pairs.append((s, int(tt[jj])))
        exp_gen += float((12 * C2 / np.log(prod.astype(np.float64)) ** 2).sum())
    print(f"    observed: st is a twin centre for {obs} of {len(counts_all)} pairs")
    print(f"    E_generic (st a random multiple of 6) = {exp_gen:.1f}")
    print(f"    observed/E_generic = {obs/exp_gen:.4f}")
    print(f"    K * E_generic = {K*exp_gen:.1f}   observed/(K*E_generic) = {obs/(K*exp_gen):.4f}")
    sd = math.sqrt(K * exp_gen)
    print(f"    Poisson sd on K*E = {sd:.2f}; z = {(obs - K*exp_gen)/sd:.3f}")
    sd0 = math.sqrt(exp_gen)
    print(f"    Poisson sd on E_generic = {sd0:.2f}; z = {(obs - exp_gen)/sd0:.3f}")
    ndiag = len(B)
    ediag = 0.0
    for s in B:
        ediag += 12 * C2 / math.log(float(s) * float(s)) ** 2
    print(f"    diagonal pairs s=t: {ndiag}; st = s^2, s^2-1=(s-1)(s+1) always composite -> observed 0")
    print(f"      E_generic charged to the diagonal = {ediag:.2f}")
    print(f"    off-diagonal: obs={obs}, E_generic={exp_gen-ediag:.1f}, ratio={obs/(exp_gen-ediag):.4f},"
          f" ratio/K={obs/((exp_gen-ediag)*K):.4f}")

    # empirical local factors on the actual pair multiset
    print("    empirical local factor per p over the actual 20301 pairs:")
    emp = 1.0
    th = 1.0
    for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
        good = 0
        tot = 0
        for ii in range(len(B)):
            s = int(B[ii])
            tt = B[ii:]
            res = (s * tt) % p
            good += int(((res != 1) & (res != p - 1)).sum())
            tot += len(tt)
        e = (good / tot) / (1 - 2 / p)
        r, _, _ = r_exact(p)
        emp *= e
        th *= r
        print(f"      p={p:<3} emp r_p={e:.5f}  model r_p={r:.5f}  running emp={emp:.5f} model={th:.5f}")

    print("=" * 70)
    print("done")


if __name__ == "__main__":
    main()

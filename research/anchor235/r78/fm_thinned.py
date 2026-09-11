"""fm_thinned.py -- P3, P4: the thinned monoids M_G, G = P_+ u T, T a subset of P_-, to N = 10^7.

For each thinning: the chain from c_1 = min G; on each section [c_k, c_{k+1}) (the last one a prefix to N) and on
each finer section [g^2, g'^2) (g, g' consecutive generators, g'^2 <= N):
  method (a), the machine: columns = j with 6j - 1 and 6j + 1 both in M_G; a column is OPEN if neither member is
      struck by a generator below the section's top generator (p_{k+1} for a section, g' for a finer section);
      twin gear pairs (a) = open columns;
  method (b), the closed form: twin prime pairs (t, t + 2) in the section with t in T.
Both classes at every scale: the count of generators of each class in [2^k, 2^{k+1}).
Output: results/thinned.json, results/thinned.log"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fm_common import primes_upto, spf_table, in_monoid_mask, chain_from, N_DEFAULT

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def thinnings(P):
    Pm = P[P % 6 == 5]
    Pp = P[P % 6 == 1]
    isP = np.zeros(int(P[-1]) + 3, dtype=bool)
    isP[P] = True
    W = Pm[isP[Pm + 2]]  # twin lowers
    Wset = set(int(t) for t in W)
    T = {}
    T["M: T={5}"] = np.array([5])
    T["T = P_- < 100"] = Pm[Pm < 100]
    T["T = P_- < 1000"] = Pm[Pm < 1000]
    T["T = P_- even index"] = Pm[0::2]
    T["T = P_- odd index"] = Pm[1::2]
    for r in (1, 2, 3, 4):
        T[f"T = P_- = {r} mod 5"] = Pm[Pm % 5 == r]
    for r in (1, 2, 3, 4, 5, 6):
        T[f"T = P_- = {r} mod 7"] = Pm[Pm % 7 == r]
    T["T = P_- \\ W (no twin lower)"] = np.array([t for t in Pm if int(t) not in Wset])
    T["T = W (twin lowers only)"] = W
    T["T = P_- = 23 mod 30"] = Pm[Pm % 30 == 23]
    T["T = P_- = 5 mod 42"] = Pm[Pm % 42 == 5]
    T["T = P_- \\ {29, 41}"] = np.array([t for t in Pm if t not in (29, 41)])
    # the strongest breaking thinning on the chain from 5: remove the twin lowers below 961
    T["T = P_- \\ (W < 961)"] = np.array([t for t in Pm if not (int(t) in Wset and t < 961)])
    T["T = P_- \\ (W < 841)"] = np.array([t for t in Pm if not (int(t) in Wset and t < 841)])
    # T_min: built greedily below (needs the chain); placeholder filled in main
    return T, Pm, Pp, W


def analyse(name, T, Pp, spf, P, W, N, log):
    G = np.union1d(Pp, T).astype(np.int64)
    gen = np.zeros(N + 1, dtype=bool)
    gen[G[G <= N]] = True
    inM = in_monoid_mask(spf, gen)
    isW = np.zeros(N + 3, dtype=bool)
    isW[W[W <= N]] = True
    Tset = np.zeros(N + 3, dtype=bool)
    Tset[T[T <= N]] = True
    twinlow = np.flatnonzero(isW & Tset)  # twin lowers in T
    chain = chain_from(G, int(G[0]), N)
    cuts = [c for c, p in chain] + [None]
    tops = [p for c, p in chain]
    # sections: [c_k, c_{k+1}) with top generator p_{k+1} = the generator at c_{k+1}; the last is a prefix to N
    secs = []
    for i, (c, p) in enumerate(chain):
        # section [c_k, c_{k+1}) with c_{k+1} = p_k^2 is struck by the generators below p_k (the square-root rule:
        # a member below p_k^2 with no generator factor below p_k is a generator); its own generators are >= p_k
        hi = chain[i + 1][0] if i + 1 < len(chain) else p * p
        top = p
        prefix = hi > N
        hi_eff = min(hi, N)
        rec = section_count(c, hi_eff, top, inM, gen, spf, twinlow, prefix)
        rec.update(dict(lo=int(c), hi=int(hi) if hi is not None else None, top_generator=int(top) if top else None,
                        prefix_to=int(N) if prefix else None))
        secs.append(rec)
    # finer sections [g^2, g'^2), g'^2 <= N
    Gs = G[G * G <= N]
    finer_fail = []
    finer_mismatch = 0
    n_finer = 0
    for g, g2 in zip(Gs, Gs[1:]):
        lo, hi = int(g * g), int(g2 * g2)
        rec = section_count(lo, hi, int(g2), inM, gen, spf, twinlow, False)
        n_finer += 1
        if rec["twin_a"] != rec["twin_b"]:
            finer_mismatch += 1
        if rec["twin_a"] == 0:
            finer_fail.append(dict(lo=lo, hi=hi, g=int(g), g2=int(g2), columns=rec["columns"]))
    # both classes at every scale
    scales = []
    for k in range(2, 24):
        lo, hi = 2 ** k, 2 ** (k + 1)
        nm = int(np.sum((T >= lo) & (T < hi)))
        npl = int(np.sum((Pp >= lo) & (Pp < hi)))
        scales.append((k, nm, npl))
    out = dict(name=name, n_generators_T=int(len(T[T <= N])), chain=[(int(c), int(p)) for c, p in chain],
               sections=secs, finer_sections=n_finer, finer_failing=len(finer_fail), finer_first_fail=finer_fail[:3],
               finer_mismatches=finer_mismatch, classes_by_dyadic=scales)
    line = json.dumps(out)
    print(line)
    log.write(line + "\n")
    log.flush()
    return out


def section_count(lo, hi, top, inM, gen, spf, twinlow, prefix):
    js = np.arange(lo // 6 + 1, (hi - 1) // 6 + 1, dtype=np.int64)
    L = 6 * js - 1
    R = 6 * js + 1
    inside = (L > lo) & (R < hi)
    js, L, R = js[inside], L[inside], R[inside]
    both = inM[L] & inM[R]
    js, L, R = js[both], L[both], R[both]
    # struck by a generator below top: the least prime factor (a generator, since the member is in M_G) is < top;
    # for a prefix (top unknown, >= sqrt(N)... ) use: struck iff composite, which is the square-root rule below top^2 > N
    if top is None:
        openL = spf[L] == L
        openR = spf[R] == R
    else:
        openL = spf[L] >= top
        openR = spf[R] >= top
    open_cols = js[openL & openR]
    twin_a = int(len(open_cols))
    twin_b = int(np.sum((twinlow > lo) & (twinlow + 2 < hi)))
    return dict(columns=int(len(js)), open=twin_a, twin_a=twin_a, twin_b=twin_b,
                first_open=[(int(6 * j - 1), int(6 * j + 1)) for j in open_cols[:3]],
                first_columns=[(int(6 * j - 1), int(6 * j + 1)) for j in js[:3]],
                struck_L_only=int(np.sum(~openL & openR)), struck_R_only=int(np.sum(openL & ~openR)),
                struck_both=int(np.sum(~openL & ~openR)), prefix=prefix)


def main():
    N = N_DEFAULT
    spf = spf_table(N)
    P = primes_upto(N).astype(np.int64)
    P = P[P >= 5]
    T, Pm, Pp, W = thinnings(P)
    # T_min: greedy one twin lower per section along the chain from 5 with P_+ present
    Tmin = [5]  # the base [5, 25) holds (5, 7) at its cut, as the real base [3, 9) holds (3, 5), (5, 7)
    c = 25
    while c <= N:
        Tarr = np.array(Tmin)
        G = np.union1d(Pp, Tarr)
        i = int(np.searchsorted(G, c))
        p = int(G[i])
        hi = p * p
        # section [c, hi): need a twin lower t in T with c < t and t + 2 < hi; add the least twin lower > c if none
        have = [t for t in Tmin if c < t and t + 2 < hi]
        if not have:
            t = int(W[(W > c) & (W + 2 < hi)][0])
            Tmin.append(t)
            Tmin.sort()
            continue  # the chain may change (the new t may be the least generator >= c)
        c = hi
    T["T_min (one twin lower per section)"] = np.array(Tmin)
    log = open(os.path.join(RES, "thinned.log"), "w")
    results = []
    for name, Tarr in T.items():
        Tarr = np.asarray(Tarr, dtype=np.int64)
        results.append(analyse(name, Tarr, Pp, spf, P, W, N, log))
    with open(os.path.join(RES, "thinned.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("T_min =", Tmin)
    print("total section mismatches (a vs b):",
          sum(1 for r in results for s in r["sections"] if s["twin_a"] != s["twin_b"]),
          "finer mismatches:", sum(r["finer_mismatches"] for r in results))


if __name__ == "__main__":
    main()

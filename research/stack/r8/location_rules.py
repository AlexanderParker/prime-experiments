"""Candidate location rules for the twin gap, tested per finer section [p^2, p'^2).

Column k = the slot (6k - 1, 6k + 1). A column is a twin iff both members are prime.
For each consecutive prime pair p < p' (p >= 5, p'^2 <= N), the finer section is the columns
k with p^2 <= 6k - 1 and 6k + 1 < p'^2. A candidate rule names ONE column of the section from
the primes at or below p; it succeeds if that column is a twin.

Candidates:
  sq1     : the first column after the square: k = (p^2 - 1)/6 + 1
  sqblind : the first column after the square at an offset i >= 1 blind to gears 5, 7, 11, 13
            (blind: -6i and 2 - 6i both non-squares mod g for each g in {5,7,11,13} with g < p)
  mid     : the middle column of the section
  midsq   : the column of the midpoint (p^2 + p'^2)/2 rounded to a column
  prodm2  : the column of p * p' - 2  (both members: p p' - 3? no: the column holding p p' - 2
            as its right member: 6k + 1 = p p' - 2 when p p' = 3 mod 6; else the column whose
            left member is p p' + 2 ... we take the column nearest to p p' on each side)
  mirror  : the column at radius 1 below the mirror axis 6 p m with m = p' (axis of the largest
            gear p and the next prime): k = p p' - k_p - 1 ... simplified: the columns adjacent
            to the column of p * p'  (k0 = (p p' -+ 1)/6, test k0 - 1 and k0 + 1)
  twinoff : the offset of the previous section's first twin (L_1(prev)) reused: k = a + L_1(prev)
Also measured: L_1(p), the first twin offset above the square, and whether it is blind to the
small gears (containment test).

Usage: uv run python location_rules.py N   (e.g. 10000000)
Results: research/stack/r8/results/location_rules_N.txt
"""
import sys, math
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)


def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def is_blind(i, gears):
    for g in gears:
        sq = {(x * x) % g for x in range(g)}
        if ((-6 * i) % g in sq) or ((2 - 6 * i) % g in sq):
            return False
    return True


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 7
    isp = sieve(N + 2)
    primes = np.nonzero(isp)[0]
    # twin columns: k with 6k-1 and 6k+1 prime
    K = (N - 1) // 6
    ks = np.arange(1, K + 1)
    twin = isp[6 * ks - 1] & isp[6 * ks + 1]  # index k-1
    def is_twin(k):
        return 1 <= k <= K and bool(twin[k - 1])
    cands = ["sq1", "sqblind", "mid", "midsq", "prod_lo", "prod_hi", "mirror_lo", "mirror_hi", "twinoff"]
    ok = {c: 0 for c in cands}; first_fail = {c: None for c in cands}; n_sec = 0
    L1_prev = 1; L1_blind_all = 0; L1_list = []
    ps = [int(p) for p in primes if p >= 5 and p * p <= N]
    for idx in range(len(ps) - 1):
        p, q = ps[idx], ps[idx + 1]
        if q * q > N: break
        a = (p * p - 1) // 6  # column whose right member is p^2
        lo, hi = a + 1, (q * q - 2) // 6  # columns with 6k-1 >= p^2+? and 6k+1 < q^2
        if hi < lo: continue
        n_sec += 1
        # first twin offset
        L1 = None
        for k in range(lo, hi + 1):
            if is_twin(k): L1 = k - a; break
        gears = [g for g in (5, 7, 11, 13) if g < p]
        if L1 is not None:
            L1_list.append((p, L1)); L1_blind_all += is_blind(L1, gears)
        tests = {}
        tests["sq1"] = a + 1
        kb = next((a + i for i in range(1, hi - a + 1) if is_blind(i, gears)), None)
        tests["sqblind"] = kb
        tests["mid"] = (lo + hi) // 2
        tests["midsq"] = ((p * p + q * q) // 2) // 6
        pq = p * q
        tests["prod_lo"] = (pq - 1) // 6 if pq % 6 == 1 else (pq + 1) // 6 - 1
        tests["prod_hi"] = tests["prod_lo"] + 1
        tests["mirror_lo"] = tests["prod_lo"] - 1
        tests["mirror_hi"] = tests["prod_hi"] + 1
        tests["twinoff"] = a + L1_prev
        for c in cands:
            k = tests[c]
            good = k is not None and lo <= k <= hi and is_twin(k)
            ok[c] += good
            if not good and first_fail[c] is None:
                first_fail[c] = (p, q, k)
        if L1 is not None: L1_prev = L1
    lines = [f"N = {N}, finer sections {n_sec}",
             "candidate | successes | fraction | first failure (p, p', column)"]
    for c in cands:
        lines.append(f"{c} | {ok[c]} | {ok[c]/n_sec:.4f} | {first_fail[c]}")
    lines.append(f"L_1(p) blind to the small gears (5,7,11,13 below p): {L1_blind_all} of {len(L1_list)}")
    lines.append("L_1 at p = 11..53: " + ", ".join(f"{p}:{L}" for p, L in L1_list if p <= 53))
    txt = "\n".join(lines); (RES / f"location_rules_{N}.txt").write_text(txt, encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()

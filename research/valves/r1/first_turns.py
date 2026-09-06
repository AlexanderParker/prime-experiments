"""The first two turns for every integer Q from 1 to Qmax: P_1, P_2 (twins in (Q, 2Q] and (2Q, 3Q]),
E_1, E_2 (q-smooth numbers there), T_m and B_m = T_m - P_m (the ember charges; proved: in turns 1, 2
every burnt charge has an ember member). Exact via prefix sums: in (Q, 2Q] a number is manifold-open iff
prime or q-smooth (a charge sP with P > Q and s >= 2 exceeds 2Q); the one edge case n + 2 = 2Q + 2 = 2(Q + 1)
is handled directly; in (2Q, 3Q + 2] open iff prime, q-smooth, or 2P with P prime > Q.

usage: uv run python research/valves/r1/first_turns.py Qmax
"""
import sys, json, os, math
import numpy as np

Qmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 5
N = 3 * Qmax + 4
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)


def sieve(n):
    is_p = np.ones(n + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = False
    return is_p


isprime = sieve(N + 2)
twin = np.zeros(N + 3, dtype=bool); twin[:N + 1] = isprime[:N + 1] & isprime[2:N + 3]
ctw = np.cumsum(twin)   # ctw[x] = twins with lower member <= x
tw_pos = np.flatnonzero(twin)


def smooth_mask(q):
    sm = np.ones(N + 3, dtype=bool); sm[0] = False
    rest = np.arange(N + 3, dtype=np.int64)
    for p in range(2, q + 1):
        if isprime[p]:
            while True:
                m = rest % p == 0
                m[0] = False
                if not m.any():
                    break
                rest[m] //= p
    return rest == 1


results = {}
Qs = np.arange(1, Qmax + 1, dtype=np.int64)
for q in (5, 7, 11, 13):
    smooth = smooth_mask(q)
    half_prime = np.zeros(N + 3, dtype=bool)
    ev = np.arange(0, N + 3, 2); half_prime[ev] = isprime[ev // 2]
    op1 = isprime | smooth                    # open in (Q, 2Q] for every Q (no 2P there)
    op2 = op1 | half_prime                    # open in (2Q, 3Q + 2] for every Q
    pair1 = np.zeros(N + 3, dtype=bool); pair1[:N + 1] = op1[:N + 1] & op1[2:N + 3]
    pair2 = np.zeros(N + 3, dtype=bool); pair2[:N + 1] = op2[:N + 1] & op2[2:N + 3]
    c1 = np.cumsum(pair1); c2 = np.cumsum(pair2); csm = np.cumsum(smooth)
    # turn 1: n in (Q, 2Q]; pairs with n <= 2Q - 2 use op1 for both; n = 2Q - 1, 2Q: partner 2Q + 1, 2Q + 2
    T1 = c1[2 * Qs - 2] - c1[Qs]
    e1 = op1[2 * Qs - 1] & op1[2 * Qs + 1]
    e2 = op1[2 * Qs] & (op1[2 * Qs + 2] | (half_prime[2 * Qs + 2] & isprime[Qs + 1]))
    T1 = T1 + e1.astype(np.int64) + e2.astype(np.int64)
    P1 = ctw[2 * Qs] - ctw[Qs]
    E1 = csm[2 * Qs] - csm[Qs]
    T2 = c2[3 * Qs] - c2[2 * Qs]
    P2 = ctw[3 * Qs] - ctw[2 * Qs]
    E2 = csm[3 * Qs] - csm[2 * Qs]
    B1 = T1 - P1; B2 = T2 - P2
    assert (B1 >= 0).all() and (B2 >= 0).all()
    assert (B1 <= 2 * E1).all() and (B2 <= 2 * E2).all()
    # direct check of the prefix-sum bookkeeping at a few Q by brute force
    for Qc in (30, 210, 1000, 2310, 10000, 12345):
        if Qc > Qmax:
            continue
        lo, hi = Qc + 1, 3 * Qc
        n = np.arange(lo, hi + 3)
        rest = n.copy()
        for p in range(2, q + 1):
            if isprime[p]:
                while True:
                    m = rest % p == 0
                    if not m.any():
                        break
                    rest[m] //= p
        opn = (rest == 1) | ((rest > Qc) & isprime[rest])
        pr = opn[:-2] & opn[2:]
        nn = n[:-2]
        t1 = int(pr[(nn > Qc) & (nn <= 2 * Qc)].sum()); t2 = int(pr[(nn > 2 * Qc) & (nn <= 3 * Qc)].sum())
        assert t1 == T1[Qc - 1] and t2 == T2[Qc - 1], (q, Qc, t1, T1[Qc - 1], t2, T2[Qc - 1])
    rng = (Qs >= 1000)
    i1 = int(np.argmin(np.where(rng, P1, 10 ** 9))); i2 = int(np.argmin(np.where(rng, P2, 10 ** 9)))

    def gap_at(Qv, m):
        lo, hi = m * Qv, (m + 1) * Qv
        i = np.searchsorted(tw_pos, lo, side="right"); j = np.searchsorted(tw_pos, hi, side="right")
        seg = tw_pos[max(i - 1, 0):j + 1]
        d = np.diff(seg)
        k = int(np.argmax(d))
        return {"largest_twin_gap_touching_turn": int(d[k]), "from": int(seg[k]), "to": int(seg[k + 1]), "twins_in_turn": int(j - i)}

    res = {"Qmax": Qmax,
           "P1_zero_Q": Qs[P1 == 0].tolist()[:50], "P2_zero_Q": Qs[P2 == 0].tolist()[:50],
           "n_P1_zero": int((P1 == 0).sum()), "n_P2_zero": int((P2 == 0).sum()),
           "B1_gt_P1_Q": Qs[B1 > P1].tolist()[-30:], "n_B1_gt_P1": int((B1 > P1).sum()), "last_B1_gt_P1": int(Qs[B1 > P1].max()) if (B1 > P1).any() else None,
           "B2_gt_P2_Q": Qs[B2 > P2].tolist()[-30:], "n_B2_gt_P2": int((B2 > P2).sum()), "last_B2_gt_P2": int(Qs[B2 > P2].max()) if (B2 > P2).any() else None,
           "B1_ge_P1_last": int(Qs[B1 >= P1].max()) if (B1 >= P1).any() else None,
           "n_B1_gt_P1_above_1000": int(((B1 > P1) & rng).sum()), "n_B2_gt_P2_above_1000": int(((B2 > P2) & rng).sum()),
           "minP1_above_1000": {"Q": int(Qs[i1]), "P1": int(P1[i1]), "B1": int(B1[i1]), "E1": int(E1[i1]), **gap_at(int(Qs[i1]), 1)},
           "minP2_above_1000": {"Q": int(Qs[i2]), "P2": int(P2[i2]), "B2": int(B2[i2]), "E2": int(E2[i2]), **gap_at(int(Qs[i2]), 2)},
           "maxB1_above_1000": {"Q": int(Qs[rng][np.argmax(B1[rng])]), "B1": int(B1[rng].max())},
           "maxB2_above_1000": {"Q": int(Qs[rng][np.argmax(B2[rng])]), "B2": int(B2[rng].max())},
           "max_B1_over_P1_above_1000": float((B1[rng] / np.maximum(P1[rng], 1)).max()),
           "max_B2_over_P2_above_1000": float((B2[rng] / np.maximum(P2[rng], 1)).max()),
           "E1_zero_count": int((E1 == 0).sum()), "E1_min_above_1000": int(E1[rng].min()), "E2_min_above_1000": int(E2[rng].min()),
           "B1_zero_frac_above_1000": float((B1[rng] == 0).mean()), "B2_zero_frac_above_1000": float((B2[rng] == 0).mean()),
           "P1_le_P2_frac_above_1000": float((P1[rng] <= P2[rng]).mean()),
           "min_P1_plus_P2_above_1000": {"Q": int(Qs[rng][np.argmin(P1[rng] + P2[rng])]), "P1+P2": int((P1[rng] + P2[rng]).min())},
           "table50": []}
    for k in range(50):
        Qv = int(round(10 ** (3 + 2 * k / 49)))
        i = Qv - 1
        res["table50"].append({"Q": Qv, "P1": int(P1[i]), "B1": int(B1[i]), "E1": int(E1[i]), "T1": int(T1[i]),
                               "P2": int(P2[i]), "B2": int(B2[i]), "E2": int(E2[i]), "T2": int(T2[i])})
    # small-Q table: Q = 1..60 exact
    res["small"] = [{"Q": int(Qv), "P1": int(P1[Qv - 1]), "B1": int(B1[Qv - 1]), "E1": int(E1[Qv - 1]), "P2": int(P2[Qv - 1]), "B2": int(B2[Qv - 1])} for Qv in range(1, 61)]
    results[q] = res
    print(f"q={q}: P1=0 at Q={res['P1_zero_Q']} ({res['n_P1_zero']} values); P2=0 at Q={res['P2_zero_Q'][:20]} ({res['n_P2_zero']})")
    print(f"  B1>P1: {res['n_B1_gt_P1']} values, last {res['last_B1_gt_P1']}, list tail {res['B1_gt_P1_Q'][-12:]}; above 1000: {res['n_B1_gt_P1_above_1000']}")
    print(f"  B2>P2: {res['n_B2_gt_P2']} values, last {res['last_B2_gt_P2']}, tail {res['B2_gt_P2_Q'][-12:]}; above 1000: {res['n_B2_gt_P2_above_1000']}")
    print(f"  min P1 above 1000: {res['minP1_above_1000']}")
    print(f"  min P2 above 1000: {res['minP2_above_1000']}")
    print(f"  max B1 {res['maxB1_above_1000']}, max B2 {res['maxB2_above_1000']}, max B1/P1 {res['max_B1_over_P1_above_1000']:.3f}, max B2/P2 {res['max_B2_over_P2_above_1000']:.3f}")
    print(f"  E1 min {res['E1_min_above_1000']} E2 min {res['E2_min_above_1000']}; B1=0 frac {res['B1_zero_frac_above_1000']:.3f} B2=0 frac {res['B2_zero_frac_above_1000']:.3f}; P1<=P2 frac {res['P1_le_P2_frac_above_1000']:.3f}; min P1+P2 {res['min_P1_plus_P2_above_1000']}")
with open(os.path.join(outdir, f"first_turns_{Qmax}.json"), "w") as f:
    json.dump(results, f, indent=1)

"""Valves from scratch: the engine (primes <= q) acting on the manifold's open set
(no prime factor in (q, Q]) in the quiet zone (Q, Q^2].

usage: uv run python research/valves/scratch/valves_scan.py q Q
Writes results/valves_q{q}_Q{Q}.json and a families CSV. Prints a bounded summary.
"""
import sys, json, math, os
from collections import Counter, defaultdict
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2])
N = Q * Q
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)


def sieve(n):
    is_p = np.ones(n + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = False
    return is_p


isprime = sieve(N + 2)
primes_all = np.flatnonzero(isprime)
engine = [int(p) for p in primes_all if p <= q]
gears_mid = primes_all[(primes_all > q) & (primes_all <= Q)]
qsharp = 1
for p in engine:
    qsharp *= p

# manifold-open indicator on [0, N+2]
mopen = np.ones(N + 3, dtype=bool)
for p in gears_mid:
    mopen[p::p] = False
mopen[0] = False

# charge pairs with lower member in (Q, N-2]
pair = (np.flatnonzero(mopen[Q + 1:N - 1] & mopen[Q + 3:N + 1]) + (Q + 1)).astype(np.int64)
n1 = pair; n2 = pair + 2


def air_fuel(x):
    r = x.copy()
    for p in engine:
        while True:
            m = (r % p == 0)
            if not m.any():
                break
            r[m] //= p
    return x // r, r


s1, f1 = air_fuel(n1); s2, f2 = air_fuel(n2)
# quiet-zone rule check: fuel is 1 or a prime above Q
bad_fuel = int(np.sum(~((f1 == 1) | ((f1 > Q) & isprime[f1])))) + int(np.sum(~((f2 == 1) | ((f2 > Q) & isprime[f2]))))

# twins in (Q, N-2]
tw = np.flatnonzero(isprime[Q + 1:N - 1] & isprime[Q + 3:N + 1]) + Q + 1
n_twins = int(len(tw))
pure = (s1 == 1) & (s2 == 1)
n_pure = int(pure.sum())
pure_eq_twins = bool(n_pure == n_twins and np.array_equal(pair[pure], tw))

# families
ss = np.stack([s1, s2], axis=1)
uk, inv, cnt = np.unique(ss, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
fam_s1 = uk[:, 0]; fam_s2 = uk[:, 1]
fam_keys = inv  # family index per pair
families = {(int(a), int(b)): int(c) for a, b, c in zip(fam_s1, fam_s2, cnt)}
ember_fam = {k_ for k_ in families if max(k_) > Q}
fuelled = {k_: v for k_, v in families.items() if max(k_) <= Q}

# P2: family shape and ports
g = np.gcd(fam_s1, fam_s2)
p2_gcd_bad = int(np.sum(~((g == 1) | (g == 2))))
p2_par_bad = int(np.sum((fam_s1 - fam_s2) % 2 != 0))
port = np.where(n1 % 2 == 0, 0, n1 % 6)   # 0 even, 1, 3, 5
port_pred = np.where(s1 % 2 == 0, 0, np.where(s1 % 3 == 0, 3, np.where(s2 % 3 == 0, 1, 5)))
p2_port_bad = int(np.sum(port != port_pred))
port_counts = {int(k): int(v) for k, v in zip(*np.unique(port, return_counts=True))}
port_fams = Counter(int(x) for x in np.where(fam_s1 % 2 == 0, 0, np.where(fam_s1 % 3 == 0, 3, np.where(fam_s2 % 3 == 0, 1, 5))))

# P3: imprint modulo q#
def imprint(a, b):
    """residues mod q# allowed for family (a, b)."""
    res = None
    mod = 1
    for p in engine:
        if a % p == 0:
            allowed = [0]
        elif b % p == 0:
            allowed = [(-2) % p]
        else:
            allowed = [r for r in range(p) if r != 0 and (r + 2) % p != 0]
        # CRT combine
        if res is None:
            res = allowed; mod = p
        else:
            new = []
            for r in res:
                for t in allowed:
                    # x = r mod mod, x = t mod p
                    x = r
                    while x % p != t:
                        x += mod
                    new.append(x)
            res = new; mod *= p
    return set(res)


imprint_size_pred = {}
for (a, b) in families:
    sz = 1
    for p in engine:
        if p > 2 and a % p != 0 and b % p != 0:
            sz *= (p - 2)
    imprint_size_pred[(a, b)] = sz

res_mod = n1 % qsharp
order = np.argsort(fam_keys, kind="stable")
res_sorted = res_mod[order]
bounds = np.concatenate([[0], np.cumsum(cnt)])
p3_contain_bad = 0; p3_attain_full = 0; p3_fams_checked = 0
p3_examples = []
per_class_stats = []
for fi, (a, b) in enumerate(zip(fam_s1, fam_s2)):
    a = int(a); b = int(b)
    members = res_sorted[bounds[fi]:bounds[fi + 1]]
    occ = set(int(x) for x in np.unique(members))
    allowed = imprint(a, b)
    p3_fams_checked += 1
    if not occ <= allowed:
        p3_contain_bad += 1
        if len(p3_examples) < 5:
            p3_examples.append(((a, b), sorted(occ - allowed)[:5]))
    if occ == allowed:
        p3_attain_full += 1
    assert len(allowed) == imprint_size_pred[(a, b)]
    if families[(a, b)] >= 20000:
        c = Counter(int(x) for x in members)
        vals = np.array([c.get(r, 0) for r in sorted(allowed)], dtype=float)
        mean = vals.mean()
        per_class_stats.append({"family": [a, b], "N": families[(a, b)], "classes": len(allowed),
                                "mean": mean, "min": int(vals.min()), "max": int(vals.max()),
                                "max_abs_dev_over_sqrt_mean": float(np.max(np.abs(vals - mean)) / math.sqrt(mean))})

# column coordinate: column port pairs, k = (n+1)/6
col = (port == 5)
k = (n1[col] + 1) // 6
cs1 = s1[col]; cs2 = s2[col]
p3_col_bad = 0
tooth = {}
for gg in engine:
    if gg < 5:
        continue
    inv6 = pow(6, -1, gg)
    tooth[gg] = inv6
    m1 = (cs1 % gg == 0); m2 = (cs2 % gg == 0)
    p3_col_bad += int(np.sum(k[m1] % gg != inv6)) + int(np.sum(k[m2] % gg != (-inv6) % gg))
    # unburnt at this gear: never on a tooth
    m0 = ~m1 & ~m2
    p3_col_bad += int(np.sum((k[m0] % gg == inv6) | (k[m0] % gg == (-inv6) % gg)))
n_col = int(col.sum())

# P4: stratum cap
stratum = (n1 - 1) // Q
p4_bad = int(np.sum((f1 > 1) & (s1 > stratum))) + int(np.sum((f2 > 1) & (s2 > ((n2 - 1) // Q))))
bottom = (stratum == 1)
bottom_burnt = ~pure & bottom
p4_bottom_bad = int(np.sum(bottom_burnt & ~((f1 == 1) | (f2 == 1))))
bottom_list = [[int(x), int(y), int(a), int(b)] for x, y, a, b in zip(n1[bottom_burnt], n2[bottom_burnt], s1[bottom_burnt], s2[bottom_burnt])]
# per stratum: total, pure, burnt (first 12 strata and overall)
strata = {}
for m in range(1, 13):
    sel = stratum == m
    strata[m] = {"charges": int(sel.sum()), "pure": int((sel & pure).sum()), "burnt": int((sel & ~pure).sum()),
                 "max_air": int(max(s1[sel].max() if sel.any() else 0, s2[sel].max() if sel.any() else 0)),
                 "families": int(len(np.unique(fam_keys[sel])))}

# P5: burnt charges between consecutive pure charges
burnt_pos = pair[~pure]
idx = np.searchsorted(burnt_pos, tw)
gaps = np.diff(tw)
counts = np.diff(idx)  # burnt charges strictly between tw[i] and tw[i+1] (burnt never equals a twin)
# distribution: by gap value, mean count and whether count is a function of gap
by_gap = defaultdict(list)
for gval, cval in zip(gaps.tolist(), counts.tolist()):
    by_gap[gval].append(cval)
gap_table = []
for gval in sorted(by_gap)[:40]:
    arr = np.array(by_gap[gval])
    gap_table.append([int(gval), int(len(arr)), float(arr.mean()), int(arr.min()), int(arr.max()), int(len(set(arr.tolist())))])
gap_functional = all(len(set(v)) == 1 for v in by_gap.values())
# by stratum of the left twin: mean burnt per unit gap
twin_stratum = (tw[:-1] - 1) // Q
rate_by_stratum = {}
for m in range(1, 13):
    sel = twin_stratum == m
    if sel.any():
        rate_by_stratum[m] = {"twin_gaps": int(sel.sum()), "burnt": int(counts[sel].sum()), "gap_sum": int(gaps[sel].sum()),
                              "rate": float(counts[sel].sum() / gaps[sel].sum()), "zero_frac": float(np.mean(counts[sel] == 0))}
# bottom-stratum gaps with a burnt charge inside: exact list
bot_sel = twin_stratum == 1
bot_nonzero = [[int(tw[i]), int(tw[i + 1]), int(counts[i])] for i in np.flatnonzero(bot_sel & (counts > 0))]
# nearest burnt charge to each twin (either side), distance distribution and air of nearest
right = burnt_pos[np.minimum(idx, len(burnt_pos) - 1)] - tw
left = tw - burnt_pos[np.maximum(idx - 1, 0)]
near = np.minimum(np.abs(right), np.abs(left))
near_hist = Counter(int(x) for x in near)
near_small = {d: near_hist.get(d, 0) for d in range(1, 13)}
# what sits at offsets +-1, +-2, +-3 from a twin (burnt charges), counts by offset and family
off_counts = {}
burnt_set_idx = {}
bs1 = s1[~pure]; bs2 = s2[~pure]
for off in (-3, -2, -1, 1, 2, 3):
    target = tw + off
    j = np.searchsorted(burnt_pos, target)
    j = np.minimum(j, len(burnt_pos) - 1)
    hit = burnt_pos[j] == target
    fams = Counter(zip(bs1[j[hit]].tolist(), bs2[j[hit]].tolist()))
    off_counts[off] = {"count": int(hit.sum()), "top_families": [[list(k_), v] for k_, v in fams.most_common(6)]}

# other view: engine-open columns in the quiet zone struck by the manifold
kk = np.arange((Q + 1) // 6 + 1, (N - 1) // 6 + 1, dtype=np.int64)
a_ = 6 * kk - 1; b_ = 6 * kk + 1
eopen = np.ones(len(kk), dtype=bool)
for p in engine:
    if p >= 5:
        eopen &= (a_ % p != 0) & (b_ % p != 0)
eo = int(eopen.sum())
emo = int((eopen & mopen[a_] & mopen[b_]).sum())

# manifold record in the quiet zone (largest gap between consecutive charges)
d = np.diff(pair)
rec = int(d.max()); rec_at = int(pair[int(d.argmax())])
gap4 = int(np.sum(d == 4))

# ONSET: first member of each family, its stratum, delay = stratum - max(s, s') (fuelled families only)
first_idx = np.full(len(uk), -1, dtype=np.int64)
np.minimum.at(first_idx, inv, np.arange(len(pair)))
# np.minimum.at with -1 initial is wrong; redo with a large initial
first_idx = np.full(len(uk), len(pair), dtype=np.int64)
np.minimum.at(first_idx, inv, np.arange(len(pair)))
onset_rows = []
delay_hist = Counter()
delay_by_max = defaultdict(list)
for fi in range(len(uk)):
    a = int(fam_s1[fi]); b = int(fam_s2[fi])
    if max(a, b) > Q:
        continue
    n0 = int(pair[first_idx[fi]])
    st = (n0 - 1) // Q
    d_ = st - max(a, b)
    delay_hist[d_] += 1
    delay_by_max[max(a, b)].append(d_)
    if max(a, b) <= 40:
        onset_rows.append([a, b, n0, st, d_])
onset_neg = sum(v for d_, v in delay_hist.items() if d_ < 0)
delay_zero_upto = 0
for M in sorted(delay_by_max):
    if all(d_ == 0 for d_ in delay_by_max[M]):
        delay_zero_upto = M
    else:
        break
# INVENTORY: admissible pairs = q-smooth (s, s'), gcd | 2, same parity, not (2, 2)
sm = [1]
for p in engine:
    sm = sorted({x * p ** e for x in sm for e in range(0, 40) if x * p ** e <= Q})
sm = [x for x in sm if x <= Q]
smset = set(sm)
adm = set()
for a in sm:
    for b in sm:
        # local solvability of s' y - s x = 2 in units: gcd | 2, same parity, and for even pairs
        # exactly one of s/2, s'/2 even (equivalently 4 divides exactly one of s, s')
        if (a - b) % 2 == 0 and math.gcd(a, b) in (1, 2) and (a % 2 == 1 or (a // 2 + b // 2) % 2 == 1):
            adm.add((a, b))
realised = set(fuelled)
not_adm = sorted(realised - adm)[:10]
n_not_adm = len(realised - adm)
inv_M = 0
for M in sorted({max(k_) for k_ in adm}):
    if all(k_ in realised for k_ in adm if max(k_) <= M):
        inv_M = M
    else:
        break
missing_small = sorted([k_ for k_ in adm if k_ not in realised and max(k_) <= 4 * inv_M])[:12]
# per-stratum fuelled family sets vs admissible with max <= m
strata_fams = {}
for m in range(1, 13):
    sel = stratum == m
    fams_m = {(int(a), int(b)) for a, b in zip(s1[sel], s2[sel]) if max(a, b) <= Q}
    adm_m = {k_ for k_ in adm if max(k_) <= m}
    strata_fams[m] = {"fuelled_families": len(fams_m), "admissible_le_m": len(adm_m), "equal": fams_m == adm_m,
                      "missing": sorted(adm_m - fams_m)[:8], "extra": sorted(fams_m - adm_m)[:8]}
# SPOKES: columns of mQ, m = 1..Q-1: always engine-open (mQ +- 1 coprime to q#) when q# | Q; twin iff both prime
spoke_m = np.arange(1, Q, dtype=np.int64)
spoke_engine_open = int(np.sum(np.gcd(spoke_m * Q - 1, qsharp) == 1) if Q % qsharp == 0 else -1)
spoke_twins = int(np.sum(isprime[spoke_m * Q - 1] & isprime[spoke_m * Q + 1]))
spoke_open_numbers = int(np.sum(mopen[spoke_m * Q - 1] & mopen[spoke_m * Q + 1]))

summary = {
    "q": q, "Q": Q, "qsharp": qsharp, "range": N,
    "charges": int(len(pair)), "families": len(families), "pure": n_pure, "twins": n_twins,
    "pure_eq_twins": pure_eq_twins, "bad_fuel": bad_fuel, "survivors_outside_11": int(np.sum(~pure & (s1 == 1) & (s2 == 1))),
    "P2": {"gcd_bad": p2_gcd_bad, "parity_bad": p2_par_bad, "port_bad": p2_port_bad, "port_counts": port_counts, "port_families": dict(port_fams)},
    "P3": {"families_checked": p3_fams_checked, "containment_bad": p3_contain_bad, "attain_full": p3_attain_full,
           "examples": p3_examples, "column_tooth_bad": p3_col_bad, "column_pairs": n_col, "per_class": per_class_stats,
           "imprint_size_11": imprint_size_pred[(1, 1)]},
    "P4": {"cap_bad": p4_bad, "bottom_burnt": len(bottom_list), "bottom_bad": p4_bottom_bad, "bottom_list": bottom_list[:60], "strata": strata},
    "P5": {"gap_functional": gap_functional, "gap_table": gap_table, "rate_by_stratum": rate_by_stratum,
           "bottom_gaps_with_burnt": bot_nonzero[:40], "bottom_gaps_with_burnt_count": len(bot_nonzero), "bottom_twin_gaps": int(bot_sel.sum()),
           "nearest_burnt_hist_1_12": near_small, "offsets": off_counts},
    "other_view": {"engine_open_columns": eo, "engine_and_manifold_open": emo},
    "record": {"quiet_zone_record": rec, "after": rec_at, "gap4": gap4},
    "top_families": [[list(k_), v] for k_, v in sorted(families.items(), key=lambda kv: -kv[1])[:30]],
    "ember_families": len(ember_fam), "fuelled_families": len(fuelled),
    "onset": {"delay_hist": {int(k_): int(v) for k_, v in sorted(delay_hist.items())}, "negative_delays": onset_neg,
              "delay_zero_for_all_max_upto": delay_zero_upto, "rows_max_le_40": onset_rows},
    "inventory": {"admissible_pairs_le_Q": len(adm), "realised_fuelled": len(realised), "realised_not_admissible": n_not_adm,
                  "examples_not_admissible": not_adm, "all_admissible_realised_upto_max": inv_M, "first_missing": missing_small,
                  "strata": strata_fams},
    "spokes": {"m_range": [1, Q - 1], "engine_open": spoke_engine_open, "manifold_open_both": spoke_open_numbers, "twins": spoke_twins},
}
with open(os.path.join(outdir, f"valves_q{q}_Q{Q}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=int)
with open(os.path.join(outdir, f"families_q{q}_Q{Q}.csv"), "w") as f:
    f.write("s,s2,count,imprint\n")
    for (a, b), c in sorted(families.items(), key=lambda kv: -kv[1]):
        f.write(f"{a},{b},{c},{imprint_size_pred[(a, b)]}\n")

print(json.dumps({k_: v for k_, v in summary.items() if k_ not in ("top_families",)}, default=int)[:6000])

"""The law table under free phase, on the raw line, on the quiet zone (Q, Q^2].

usage: uv run python research/valves/r3/phase_zero.py q Q model [M] [seed]

models (all on the same gear set, the primes in (q, Q]):
  real   : phase zero, gear g strikes the numbers g | n.
  rand   : every gear at an independent uniform random phase c_g; n struck iff n = c_g (mod g).
           On pairs the domino {c_g - 2, c_g} is kept.  (seed selects the vector)
  adv2   : the two-tooth free-phase adversary: phases chosen greedily so that the dominoes cover every
           pure-imprint pair (n, n + 2 both coprime to q#) in (Q, (M + 1) Q]; unused gears at phase 0;
           the sieve then acts on n itself over the whole zone.
  adv1   : the one-tooth adversary of valve_existence.md: the same phases, applied to the FUEL: n = s f
           (s its q-smooth part) is open iff f = 1 or (f > Q, f coprime to q#, f avoids every class c_g).
  parity : the parity adversary on the charges: the real open set restricted to the numbers with an even
           number of prime factors (Liouville +1); a prime is never in it.
  twinsieve : the saturated adversary: phase zero with the gear set enlarged by the twin members in (Q, Q^2]
           (every prime P > Q with P - 2 or P + 2 prime); its open set is the real one minus the charges whose
           fuel is a twin member; it has no pure pair by construction.

For the model's open set the script measures every law on record (see phase_zero.md): the pair-machine laws
(gap 4, run and chain ceilings, the gap census against the period density of L22), the zone laws (smooth
zone, quiet-zone rule both ways), saturation (closure under multiplication by air, closure under removing
air), the valve laws (imprint, port, inventory, onset, air cap, ember law), the count identities, the
per-turn densities and the record.  Output: results/pz_<model>_q<q>_Q<Q>[_s<seed>].json and a text summary.
"""
import sys, os, math, json, time, heapq
import numpy as np

t0 = time.time()
q = int(sys.argv[1]); Q = int(sys.argv[2]); model = sys.argv[3]
M = int(sys.argv[4]) if len(sys.argv) > 4 else 60
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)
tag = f"{model}_q{q}_Q{Q}" + (f"_s{seed}" if model == "rand" else "")
N = Q * Q + 2                       # pairs (n, n+2) with n <= Q^2 - 2 need numbers to Q^2; keep a margin


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


isprime = sieve(N + 2)
engine = [p for p in range(2, q + 1) if isprime[p]]
qsharp = math.prod(engine)
gears = np.flatnonzero(isprime[:Q + 1]); gears = [int(g) for g in gears if g > q]
qprime = gears[0]
log(f"q={q} Q={Q} model={model}: {len(gears)} gears in ({q}, {Q}], q' = {qprime}, engine {engine}, q# = {qsharp}")

# q-smooth numbers up to N
sm = [1]
for p in engine:
    new = []
    for s in sm:
        x = s * p
        while x <= N + 2:
            new.append(x); x *= p
    sm += new
smooth_all = np.array(sorted(sm), dtype=np.int64)
smooth_ge2 = smooth_all[smooth_all >= 2]
embers = smooth_all[(smooth_all > Q) & (smooth_all <= N + 2)]

# ---------------------------------------------------------------- phases
phases = {g: 0 for g in gears}
adv_info = {}
if model == "rand":
    rng = np.random.default_rng(seed)
    for g in gears:
        phases[g] = int(rng.integers(0, g))
elif model in ("adv1", "adv2"):
    coprime = np.ones(N + 3, dtype=bool)
    for p in engine:
        coprime[::p] = False
    hi = (M + 1) * Q
    cand = np.flatnonzero(coprime[Q + 1:hi - 1] & coprime[Q + 3:hi + 1]) + Q + 1
    cand = cand[cand <= hi]
    remaining = cand.copy(); n_cand = len(cand)
    heap = []
    for p in gears:
        h = np.bincount(remaining % p, minlength=p) + np.bincount((remaining + 2) % p, minlength=p)
        c = int(h.argmax()); heap.append((-int(h[c]), p, c))
    heapq.heapify(heap)
    used = {}
    while len(remaining) and heap:
        negb, p, c = heapq.heappop(heap)
        h = np.bincount(remaining % p, minlength=p) + np.bincount((remaining + 2) % p, minlength=p)
        c = int(h.argmax()); b = int(h[c])
        if heap and b < -heap[0][0]:
            heapq.heappush(heap, (-b, p, c)); continue
        used[p] = c
        kill = (remaining % p == c) | ((remaining + 2) % p == c)
        remaining = remaining[~kill]
    for p, c in used.items():
        phases[p] = c
    adv_info = {"candidates": int(n_cand), "gears_used": len(used), "nonzero_phase": sum(1 for c in used.values() if c),
                "survivors_in_M_turns": int(len(remaining)), "first_survivor": int(remaining.min()) if len(remaining) else None,
                "M": M}
    log(f"adversary: {n_cand} pure-imprint pairs in (Q, {hi}], {len(used)} of {len(gears)} gears used, "
        f"{adv_info['nonzero_phase']} at nonzero phase, survivors {len(remaining)}"
        + (f", first {int(remaining.min())} (turn {(int(remaining.min()) - 1) // Q})" if len(remaining) else ""))
    del coprime, cand, remaining

# ---------------------------------------------------------------- the open set on [0, N + 2]
ok = np.ones(N + 3, dtype=bool); ok[0] = False
if model in ("real", "rand", "adv2", "parity", "twinsieve"):
    for g in gears:
        ok[phases[g]::g] = False
elif model == "adv1":
    okF = np.ones(N + 3, dtype=bool)
    for g in gears:
        okF[phases[g]::g] = False
    for p in engine:
        okF[::p] = False
    okF[:Q + 1] = False
    F = np.flatnonzero(okF); del okF
    ok = np.zeros(N + 3, dtype=bool)
    for s in smooth_all:
        s = int(s)
        if s > (N + 2) // (Q + 1):
            break
        sub = F[F <= (N + 2) // s]
        ok[s * sub] = True
    ok[smooth_all[smooth_all <= N + 2]] = True     # embers, and the smooth zone below Q as the real one
    del F
else:
    raise SystemExit("model must be real, rand, adv2, adv1, parity")

# ---------------------------------------------------------------- decomposition of the open numbers in the zone


def decompose(idx):
    rest = idx.copy(); om = np.zeros(len(idx), dtype=np.int8)
    for p in engine:
        pos = np.flatnonzero(rest % p == 0)
        while len(pos):
            rest[pos] //= p; om[pos] += 1
            pos = pos[rest[pos] % p == 0]
    return idx // rest, rest, om


idx = np.flatnonzero(ok[Q + 1:Q * Q + 3]).astype(np.int64) + Q + 1
air, f, oms = decompose(idx)
if model == "parity":
    # Omega(n) = Omega(s) + [f > 1] for the real open set (f is 1 or a prime above Q)
    keep = ((oms + (f > 1)) % 2 == 0)
    ok[idx[~keep]] = False
    idx, air, f, oms = idx[keep], air[keep], f[keep], oms[keep]
    del keep
if model == "twinsieve":
    twin_member = isprime[np.minimum(f, N + 2)] & (isprime[np.minimum(f + 2, N + 2)] | isprime[np.maximum(f - 2, 0)]) & (f > Q)
    ok[idx[twin_member]] = False
    res_twin_gears = int(len(np.unique(f[twin_member])))
    idx, air, f, oms = idx[~twin_member], air[~twin_member], f[~twin_member], oms[~twin_member]
    del twin_member
log(f"open numbers in (Q, Q^2 + 2]: {len(idx)}")

res = {"q": q, "Q": Q, "model": model, "extra_gears": res_twin_gears if model == "twinsieve" else 0, "seed": seed if model == "rand" else None, "gears": len(gears), "qprime": qprime,
       "engine": engine, "adv": adv_info}

# ---------------------------------------------------------------- zone laws and saturation, on numbers
zone = idx <= Q * Q
T = int(zone.sum()); E = int((zone & (f == 1)).sum()); pure_num = int((zone & (air == 1)).sum())
res["T_numbers"] = T; res["embers"] = E; res["pure_fuel"] = pure_num
# smooth zone: open numbers in (1, Q] should be exactly the q-smooth ones
below = np.flatnonzero(ok[1:Q + 1]) + 1
sm_below = smooth_all[(smooth_all >= 1) & (smooth_all <= Q)]
res["smooth_zone"] = {"open_below_Q": int(len(below)), "smooth_below_Q": int(len(sm_below)),
                      "open_not_smooth": int(len(np.setdiff1d(below, sm_below))), "smooth_not_open": int(len(np.setdiff1d(sm_below, below)))}
# quiet-zone rule, "only if": every open number is s x P with P = 1 or a prime above Q
fz = f[zone]
bad_low = (fz > 1) & (fz <= Q)
bad_comp = (fz > Q) & ~isprime[np.minimum(fz, N + 2)]
res["quiet_only_if"] = {"open": T, "rough_part_in_(1,Q]": int(bad_low.sum()), "rough_part_composite_above_Q": int(bad_comp.sum())}
# SAT-down: for every open n with air > 1, its rough part is 1 or open
burnt_num = zone & (air > 1)
fb = f[burnt_num]
down_viol = ~((fb == 1) | ok[fb])
res["sat_down"] = {"tests": int(len(fb)), "violations": int(down_viol.sum())}
# SAT-up on the pure fuel: every s x f (s >= 2 smooth, f pure, s f <= Q^2) is open
pf = idx[zone & (air == 1)]
tests = 0; viol = 0
for s in smooth_ge2:
    s = int(s)
    if s > (Q * Q) // (Q + 1):
        break
    sub = pf[pf <= (Q * Q) // s]
    if not len(sub):
        continue
    tests += len(sub); viol += int((~ok[s * sub]).sum())
res["sat_up_pure"] = {"tests": tests, "violations": viol}
# closure under multiplication by air, all open n (the charge set as a module over the smooth numbers)
tests = 0; viol = 0; viol_odd = 0; tests_odd = 0
zidx = idx[zone]
for s in smooth_ge2:
    s = int(s)
    if s > (Q * Q) // (Q + 1):
        break
    sub = zidx[zidx <= (Q * Q) // s]
    if not len(sub):
        continue
    v = int((~ok[s * sub]).sum()); tests += len(sub); viol += v
    om_s = 0; x = s
    for p in engine:
        while x % p == 0:
            x //= p; om_s += 1
    if om_s % 2 == 1:
        tests_odd += len(sub); viol_odd += v
res["module_closure"] = {"tests": tests, "violations": viol, "tests_odd_omega_air": tests_odd, "violations_odd_omega_air": viol_odd}
del pf, zidx, fb, fz
log(f"numbers: T={T} embers={E} pure fuel={pure_num}; quiet only-if bad {res['quiet_only_if']}; sat_down {res['sat_down']}; "
    f"sat_up {res['sat_up_pure']}; module {res['module_closure']}; smooth zone {res['smooth_zone']}")

# the count identity as numbers: T - E = sum over smooth s < Q of N_pure(Q^2 / s)
pure_sorted = idx[zone & (air == 1)]
rhs = 0
for s in smooth_all:
    s = int(s)
    if s > (Q * Q) // (Q + 1):
        break
    rhs += int(np.searchsorted(pure_sorted, (Q * Q) // s, side="right"))
res["count_identity_numbers"] = {"T_minus_E": T - E, "sum_over_air_of_pure": rhs, "difference": T - E - rhs}
del pure_sorted

# ---------------------------------------------------------------- pairs
pair_mask = ok[idx + 2] & (idx <= Q * Q - 2)
n = idx[pair_mask]
pos2 = np.searchsorted(idx, n + 2)
s1 = air[pair_mask]; f1 = f[pair_mask]; o1 = oms[pair_mask]
s2 = air[pos2]; f2 = f[pos2]; o2 = oms[pos2]
turn = (n - 1) // Q
npairs = len(n)
pure = (s1 == 1) & (s2 == 1)
fuelled = (f1 > 1) & (f2 > 1)
burnt = ~pure
res["pairs"] = int(npairs); res["pure_pairs"] = int(pure.sum()); res["burnt_pairs"] = int(burnt.sum())
res["fuelled_pairs"] = int(fuelled.sum())
res["liouville_signed_pairs"] = int(((-1) ** ((o1 + (f1 > 1)).astype(np.int64) + (o2 + (f2 > 1)).astype(np.int64))).sum())
if model == "real":
    tw = isprime[n] & isprime[n + 2]
    res["pure_equals_twins"] = {"twins": int(tw.sum()), "pure": int(pure.sum()), "symmetric_difference": int((tw != pure).sum())}
# families
key = s1 * (N + 3) + s2
fam_keys, fam_counts = np.unique(key[fuelled], return_counts=True)
fam_s = fam_keys // (N + 3); fam_t = fam_keys % (N + 3)
families = {(int(a), int(b)): int(c) for a, b, c in zip(fam_s, fam_t, fam_counts)}
res["fuelled_families"] = len(families)
res["ember_families"] = int(len(np.unique(key[~fuelled])))
top = sorted(families.items(), key=lambda kv: -kv[1])[:12]
res["top_families"] = [[list(k), v] for k, v in top]
mirrored = sum(1 for (a, b) in families if (b, a) in families)
res["mirror_closure"] = {"families": len(families), "with_mirror": mirrored}
# inventory on fuelled pairs: gcd | 2, same parity, and if even then 4 divides exactly one
gg = np.gcd(s1[fuelled], s2[fuelled]); a = s1[fuelled]; b = s2[fuelled]
inv_bad = (gg > 2) | (a % 2 != b % 2) | ((a % 2 == 0) & ((a % 4 == 0) == (b % 4 == 0)))
res["inventory"] = {"fuelled_pairs": int(fuelled.sum()), "violations": int(inv_bad.sum())}
# imprint and port (congruence facts of n = s f with f coprime to q#)
imp_bad = 0
for p in engine:
    imp_bad += int(((n % p == 0) != (s1 % p == 0)).sum()) + int((((n + 2) % p == 0) != (s2 % p == 0)).sum())
port = n % 6
port_pred = np.where(s1 % 3 == 0, 3, np.where(s2 % 3 == 0, 1, 5))       # the odd ports; even iff 2 | s
port_bad = np.where(s1 % 2 == 0, n % 2 != 0, port != port_pred)
res["imprint_violations"] = imp_bad; res["port_violations"] = int(port_bad.sum())
# onset, air cap, ember law
onset_bad = fuelled & (np.maximum(s1, s2) > turn)
aircap_bad_num = int((zone & (f > 1) & (air > (idx - 1) // Q)).sum())
ember_bad = (turn <= 2) & burnt & fuelled
res["onset"] = {"fuelled_pairs": int(fuelled.sum()), "violations": int(onset_bad.sum()),
                "violations_turn1": int((onset_bad & (turn == 1)).sum()), "violations_turn2": int((onset_bad & (turn == 2)).sum())}
res["air_cap_numbers"] = {"fuelled_numbers": int((zone & (f > 1)).sum()), "violations": aircap_bad_num}
res["ember_law"] = {"burnt_pairs_turns_1_2": int(((turn <= 2) & burnt).sum()), "violations": int(ember_bad.sum()),
                    "burnt_turn1": int(((turn == 1) & burnt).sum()), "burnt_turn2": int(((turn == 2) & burnt).sum()),
                    "embers_turn1": int(((embers > Q) & (embers <= 2 * Q)).sum()), "embers_turn2": int(((embers > 2 * Q) & (embers <= 3 * Q)).sum())}
# per-turn ledger
Tm = np.bincount(turn, minlength=Q + 1); Pm = np.bincount(turn[pure], minlength=Q + 1)
Dn = np.bincount((idx[zone] - 1) // Q, minlength=Q + 1)
sel = [m for m in [1, 2, 3, 5, 9, 15, 30, 60, 100, 300, 1000, 3000, Q - 1] if m <= Q - 1]
res["per_turn"] = {str(m): {"open_numbers": int(Dn[m]), "density": Dn[m] / Q, "pairs": int(Tm[m]), "pure": int(Pm[m])} for m in sel}
res["turns_with_no_pure"] = int((Pm[1:Q] == 0).sum()); res["first_turn_with_pure"] = int(np.flatnonzero(Pm[1:Q] > 0)[0] + 1) if (Pm[1:Q] > 0).any() else None
res["last_turn_with_no_pure"] = int(np.flatnonzero(Pm[1:Q] == 0)[-1] + 1) if (Pm[1:Q] == 0).any() else None
# balance of the pure fuel on the engine's reduced classes
bal = {}
pf_all = idx[zone & (air == 1)]
for p in engine:
    if p == 2:
        continue
    h = np.bincount(pf_all % p, minlength=p)[1:]
    bal[p] = [int(h.min()), int(h.max())]
res["balance"] = bal
del pf_all
log(f"pairs {npairs}: pure {int(pure.sum())}, burnt {int(burnt.sum())}, fuelled {int(fuelled.sum())}, families {len(families)} "
    f"(mirrored {mirrored}); inventory viol {res['inventory']['violations']}; imprint viol {imp_bad}; port viol {res['port_violations']}; "
    f"onset viol {res['onset']['violations']}; air cap viol {aircap_bad_num}; ember viol {res['ember_law']['violations']}")

# ---------------------------------------------------------------- pair-machine laws on the zone
gp = np.diff(n)
maxd = int(gp.max())
hist = np.bincount(gp, minlength=min(maxd, 64) + 1)
res["gap_hist"] = {str(d): int(hist[d]) for d in range(1, min(64, maxd) + 1)}
res["gap4"] = int((gp == 4).sum())
imax = int(gp.argmax())
res["record"] = {"gap": maxd, "after": int(n[imax]), "turn": int(turn[imax]),
                 "twin_gap": bool(isprime[n[imax]] and isprime[n[imax] + 2] and isprime[n[imax + 1]] and isprime[n[imax + 1] + 2]) if model != "adv1" else None}
# runs of consecutive open numbers (step 1) and of one parity (step 2), from ok on the zone
oz = ok[Q + 1:Q * Q + 1]


def longest_run(b):
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    r = en - st
    if not len(r):
        return 0, 0
    return int(r.max()), int((r == r.max()).sum())


r1, c1 = longest_run(oz)
r2a, _ = longest_run(oz[0::2]); r2b, _ = longest_run(oz[1::2])
res["run_ceiling"] = {"longest_open_number_run": r1, "count": c1, "longest_open_pair_run_step1": max(r1 - 2, 0), "ceiling_qprime_minus_3": qprime - 3}
res["chain_ceiling"] = {"longest_one_parity_run": max(r2a, r2b), "longest_open_pair_chain_step2": max(max(r2a, r2b) - 1, 0), "ceiling_qprime_minus_2": qprime - 2}
del oz
log(f"gaps: max {maxd} after {int(n[imax])} (turn {int(turn[imax])}); gap4 {res['gap4']}; run {r1} (ceiling {qprime - 1} numbers); "
    f"chain {max(r2a, r2b)} (ceiling {qprime - 1} numbers)")

# ---------------------------------------------------------------- the gap census against the period density (L22 / W98)


def census_density(d, gears):
    """N_d / W for consecutive open pairs at distance d, by L22: sum over S subset of (0, d) of (-1)^|S| prod_g (1 - |E_g(S)| / g)."""
    small = [g for g in gears if g <= d + 2]
    big = [g for g in gears if g > d + 2]
    # prod over big gears of (1 - k / g) for k = 0 .. d + 3 (all offsets distinct mod g there)
    bigprod = []
    for k in range(0, d + 4):
        pr = 1.0
        for g in big:
            pr *= (1 - k / g)
        bigprod.append(pr)
    inner = list(range(1, d))
    total = 0.0
    for mask in range(1 << len(inner)):
        S = [inner[i] for i in range(len(inner)) if mask >> i & 1]
        offs = {0, 2, d, d + 2} | set(S) | {x + 2 for x in S}
        k = len(offs)
        term = bigprod[k]
        for g in small:
            e = len({(-x) % g for x in offs})
            term *= (1 - e / g)
        total += (-1) ** len(S) * term
    return total


dens = {}
for d in range(1, 15):
    dens[d] = census_density(d, gears)
length = Q * Q - Q
bands = [(1, 2), (3, 10), (11, 100), (101, 1000), (1001, Q - 1)]
census = {}
for d in range(1, 15):
    exp_all = dens[d] * length
    row = {"density": dens[d], "expected": exp_all, "measured": int(hist[d]) if d < len(hist) else 0,
           "z": ((int(hist[d]) if d < len(hist) else 0) - exp_all) / math.sqrt(max(exp_all, 1))}
    for lo, hi in bands:
        if lo > Q - 1:
            continue
        sel_b = (turn[:-1] >= lo) & (turn[:-1] <= min(hi, Q - 1))
        cnt = int((gp[sel_b] == d).sum()); L_b = (min(hi, Q - 1) - lo + 1) * Q
        row[f"band_{lo}_{hi}"] = {"measured": cnt, "expected": dens[d] * L_b, "ratio": cnt / (dens[d] * L_b) if dens[d] * L_b > 0 else None}
    census[d] = row
res["census"] = census
res["period_density_numbers"] = math.exp(sum(math.log1p(-1 / g) for g in gears))
res["period_density_pairs"] = math.exp(sum(math.log1p(-2 / g) for g in gears))
res["zone_density_numbers"] = T / length; res["zone_density_pairs"] = npairs / length
log("census (d: measured / expected, z): " + "; ".join(f"{d}: {census[d]['measured']}/{census[d]['expected']:.0f} ({census[d]['z']:+.1f})" for d in range(1, 15)))
log(f"density numbers zone {T / length:.4f} vs period {res['period_density_numbers']:.4f}; pairs zone {npairs / length:.5f} vs period {res['period_density_pairs']:.5f}")
res["seconds"] = time.time() - t0
with open(os.path.join(outdir, f"pz_{tag}.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=int)
log("done")

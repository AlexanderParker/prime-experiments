"""The core's real-phase leftover K_L(x) on a section (branch R4.d.i continued, round r5).

A section [lo, hi) of the chain from a base; slots (n, n + 2) with n = 5 mod 6, n = n0 + 6 j.
For every slot j the SMALLEST gear (prime >= 5 below the cut) dividing n or n + 2 is m[j]
(sentinel if none: the slot is a twin).  The core for a length L is the primes <= 6 L + 1, so the
slot is core-unstruck iff m[j] > 6 L + 1, and K_L(x) = sum_{i < L} [m[x + i] > 6 L + 1] is one
sliding sum per L.  Counterfactuals are other m arrays: random phases (each gear's two teeth
shifted by a random residue), or a non-coprime random integer gear set of the same sizes striking
its multiples.

usage:
  uv run python research/stack/r5/core_leftover.py sections          (Q1: the three sections at L*)
  uv run python research/stack/r5/core_leftover.py lscan             (Q2 + Q3: base 3, L = 50..2000)
  uv run python research/stack/r5/core_leftover.py free              (free-phase greedy records)
Memory: the base-3 section is 43.4 M slots; one m array is 87 MB (int16), a sliding pass ~350 MB.
"""
import sys, os, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
SENT = 32000  # sentinel: no gear below the cut strikes the slot (both members prime)


def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return s


def nextprime(n, isp):
    m = n + 1
    while not isp[m]: m += 1
    return m


def section(base, k, isp):
    cuts = [base]; firsts = [base]
    while len(cuts) <= k:
        c = firsts[-1] ** 2
        cuts.append(c)
        if len(cuts) <= k: firsts.append(nextprime(c - 1, isp) if isp[c] else nextprime(c, isp))
    return cuts[k - 1], cuts[k]


def build_m(n0, S, gears, phases=None):
    """m[j] = smallest gear striking slot j (n = n0 + 6 j), or SENT.  gears ascending;
    phases: optional array of shifts a_g (the gear strikes n = a_g and a_g - 2 mod g)."""
    m = np.full(S, SENT, dtype=np.int16)
    for idx in range(len(gears) - 1, -1, -1):  # descending: the last write is the smallest gear
        g = int(gears[idx]); a = 0 if phases is None else int(phases[idx])
        inv = pow(6, -1, g)
        for r in (a % g, (a - 2) % g):
            j0 = ((r - n0) * inv) % g
            m[j0::g] = g
    return m


def sliding(m, L, thr):
    """K_L(x) for every start x (int32)."""
    left = (m > thr).astype(np.int32)
    cs = np.cumsum(left, dtype=np.int32)
    K = cs[L - 1:].copy(); K[1:] -= cs[:-L]
    return K


def describe(K, L, n0):
    """distribution, minimum, positions, binomial comparison."""
    N = K.size
    hist = np.bincount(K)
    kmin = int(K.min()); kmax = int(K.max())
    mean = float(K.mean()); var = float(K.var())
    pos = np.flatnonzero(K == kmin)
    # clusters of consecutive minimum positions
    clusters = 1 + int((np.diff(pos) > 1).sum()) if pos.size else 0
    p = mean / L
    # binomial model: independent indicators with the same mean
    from math import lgamma, log, exp
    def binpmf(k):
        return exp(lgamma(L + 1) - lgamma(k + 1) - lgamma(L - k + 1) + k * log(p) + (L - k) * log(1 - p))
    binvar = L * p * (1 - p)
    cum = 0.0; ev_bin = None; bin_counts = {}
    for k in range(0, min(L, kmin + 12) + 1):
        pk = binpmf(k); cum += pk; bin_counts[k] = N * pk
        if ev_bin is None and N * cum >= 1.0: ev_bin = k
    # normal model with the measured variance: smallest k with N P(Z <= (k + 0.5 - mean)/sd) >= 1
    sd = math.sqrt(var); ev_norm = None
    for k in range(0, L + 1):
        z = (k + 0.5 - mean) / sd
        if N * 0.5 * math.erfc(-z / math.sqrt(2)) >= 1.0: ev_norm = k; break
    return dict(L=L, N=int(N), min=kmin, max=kmax, mean=mean, var=var, binvar=binvar,
                var_ratio=var / binvar, p=p, n_at_min=int(pos.size), clusters_at_min=clusters,
                first_min_x=int(pos[0]) if pos.size else None,
                first_min_n=int(n0 + 6 * pos[0]) if pos.size else None,
                min_positions_x=[int(v) for v in pos[:40]],
                hist={int(k): int(hist[k]) for k in range(hist.size) if hist[k]},
                ev_bin=ev_bin, ev_norm=ev_norm,
                bin_low_tail={k: v for k, v in bin_counts.items()})


def longest_run(m, thr):
    """R(thr): the longest run of consecutive slots all struck by gears <= thr (in slots)."""
    idx = np.flatnonzero(m > thr)
    if idx.size == 0: return int(m.size)
    gaps = np.diff(np.concatenate([[-1], idx, [m.size]])) - 1
    return int(gaps.max())


def greedy_cover(L, gears, order="asc"):
    """Free-phase greedy: each gear takes the residue striking the most still-open slots of
    [0, L).  Teeth in slot coordinates: j = a and j = a + d_g, d_g = -2 * 6^{-1} mod g.
    Returns the leftover count (0 = covered)."""
    unc = np.ones(L, dtype=np.float64)
    gs = list(gears) if order == "asc" else list(gears)[::-1]
    J = np.arange(L)
    for g in gs:
        g = int(g); d = (-2 * pow(6, -1, g)) % g
        r = J % g
        cnt = np.bincount(r, weights=unc, minlength=g)
        score = cnt + cnt[(np.arange(g) + d) % g]
        a = int(np.argmax(score))
        if score[a] <= 0: continue
        unc[(r == a) | (r == (a + d) % g)] = 0.0
        if not unc.any(): return 0
    return int(unc.sum())


def free_record(gears_all, Lmax, fixed_core=None):
    """largest L the greedy covers: with core(L) = gears <= 6L+1 (or a fixed core)."""
    best = 0
    L = 8
    while L <= Lmax:
        core = fixed_core if fixed_core is not None else [g for g in gears_all if g <= 6 * L + 1]
        left = min(greedy_cover(L, core, "asc"), greedy_cover(L, core, "desc"))
        if left == 0: best = L; L = int(L * 1.25) + 1
        else: break
    # refine between best and L
    lo, hi = best, L
    while hi - lo > 1:
        mid = (lo + hi) // 2
        core = fixed_core if fixed_core is not None else [g for g in gears_all if g <= 6 * mid + 1]
        left = min(greedy_cover(mid, core, "asc"), greedy_cover(mid, core, "desc"))
        if left == 0: lo = mid
        else: hi = mid
    return lo


def random_phases(gears, rng):
    return np.array([rng.integers(0, int(g)) for g in gears], dtype=np.int64)


def random_gearset(gears, rng, rel=0.05):
    """non-coprime random integers coprime to 6, within +-rel of each prime, distinct."""
    out = []; used = set(); shared = 0
    for g in gears:
        g = int(g); w = max(2, int(rel * g))
        for _ in range(1000):
            c = int(rng.integers(max(5, g - w), g + w + 1))
            if c % 2 and c % 3 and c not in used: break
        used.add(c); out.append(c)
    out = sorted(out)
    pairs = 0
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if math.gcd(out[i], out[j]) > 1: pairs += 1
    return np.array(out, dtype=np.int64), pairs


SECTIONS = {  # base, k, L*, LIMIT
    "base3": (3, 4, 579, 260467321 + 40),
    "base7": (7, 3, 254, 7946761 + 40),
    "base23": (23, 2, 153, 292681 + 40),
}


def load_section(name):
    base, k, Lstar, LIMIT = SECTIONS[name]
    isp = sieve(LIMIT)
    lo, hi = section(base, k, isp)
    n0 = lo + ((5 - lo) % 6)
    S = (hi - 3 - n0) // 6 + 1  # slots with n + 2 < hi (the slot (hi - 2, hi) holds the next cut, a square)
    gears = np.flatnonzero(isp[:lo])
    gears = gears[gears >= 5]
    twin = isp[n0:n0 + 6 * S:6][:S] & isp[n0 + 2:n0 + 2 + 6 * S:6][:S]
    return dict(base=base, k=k, Lstar=Lstar, lo=lo, hi=hi, n0=n0, S=S, gears=gears, twin=twin, isp=isp)


def cmd_sections():
    out = {}
    for name in ("base23", "base7", "base3"):
        t0 = time.time()
        sec = load_section(name); L = sec["Lstar"]; thr = 6 * L + 1
        gears = sec["gears"]; core = gears[gears <= thr]; tail = gears[gears > thr]
        m = build_m(sec["n0"], sec["S"], gears)
        # check: twins = slots with m == SENT
        assert bool(np.array_equal(m == SENT, sec["twin"])), "band structure check failed"
        K = sliding(m, L, thr)
        d = describe(K, L, sec["n0"])
        # the record stretch: the unique twin-free stretch of length L
        tw = sec["twin"].astype(np.int32); cs = np.cumsum(tw); O = cs[L - 1:].copy(); O[1:] -= cs[:-L]
        rec = np.flatnonzero(O == 0)
        d["record_positions_x"] = [int(v) for v in rec]
        d["record_K"] = [int(K[v]) for v in rec]
        d["record_n"] = [int(sec["n0"] + 6 * v) for v in rec]
        d["positions_K_le_recordK"] = int((K <= K[rec[0]]).sum()) if rec.size else None
        d["percentile_of_record"] = float((K <= K[rec[0]]).mean()) if rec.size else None
        # quiet part (n < thr^2: leftovers are twins) against the generic part
        qx = max(0, (thr * thr - sec["n0"]) // 6 + 1)
        d["quiet_slots"] = int(min(qx, sec["S"]))
        d["min_quiet"] = int(K[:qx].min()) if qx > 0 else None
        d["min_generic"] = int(K[qx:].min()) if qx < K.size else None
        d["min_positions_in_quiet"] = int((np.array(d["min_positions_x"]) < qx).sum())
        # tail strikes on the minimum stretches: leftovers not struck by the tail = twins there
        d["twins_on_min_stretches"] = [int(O[x]) for x in d["min_positions_x"][:10]]
        # core's own real-phase record at this core
        d["R_core"] = longest_run(m, thr)
        d["core_gears"] = int(core.size); d["tail_gears"] = int(tail.size); d["thr"] = thr
        d["lo"] = sec["lo"]; d["hi"] = sec["hi"]; d["S"] = sec["S"]
        # the free-phase side: greedy cover of [0, L) by core(L*) and the greedy free record
        d["greedy_left_at_Lstar_asc"] = greedy_cover(L, core, "asc")
        d["greedy_left_at_Lstar_desc"] = greedy_cover(L, core, "desc")
        d["greedy_free_record_fixed_core"] = free_record(gears, 20 * L, fixed_core=list(core))
        # random-phase counterfactuals (3 seeds) and the non-coprime integer set (2 seeds)
        cf = []
        for seed in range(3):
            rng = np.random.default_rng(1000 + seed)
            ph = random_phases(gears, rng)
            mc = build_m(sec["n0"], sec["S"], gears, ph)
            Kc = sliding(mc, L, thr); dc = describe(Kc, L, sec["n0"])
            cf.append(dict(kind="random_phase", seed=seed, min=dc["min"], mean=dc["mean"], var=dc["var"],
                           n_at_min=dc["n_at_min"], R_core=longest_run(mc, thr), ev_bin=dc["ev_bin"], ev_norm=dc["ev_norm"],
                           hist_low={k: v for k, v in dc["hist"].items() if k <= dc["min"] + 6}))
            del mc, Kc
        for seed in range(2):
            rng = np.random.default_rng(2000 + seed)
            gs, pairs = random_gearset(gears, rng)
            mc = build_m(sec["n0"], sec["S"], gs)
            Kc = sliding(mc, L, thr); dc = describe(Kc, L, sec["n0"])
            cf.append(dict(kind="random_integers", seed=seed, min=dc["min"], mean=dc["mean"], var=dc["var"],
                           n_at_min=dc["n_at_min"], R_core=longest_run(mc, thr), noncoprime_pairs=pairs,
                           n_gears=int(gs.size), core_size=int((gs <= thr).sum()),
                           sum_2_over_g_core=float(sum(2.0 / g for g in gs if g <= thr)),
                           sum_2_over_g_real=float(sum(2.0 / g for g in core)),
                           ev_bin=dc["ev_bin"], ev_norm=dc["ev_norm"],
                           hist_low={k: v for k, v in dc["hist"].items() if k <= dc["min"] + 6}))
            del mc, Kc
        d["counterfactuals"] = cf
        d["seconds"] = round(time.time() - t0, 1)
        out[name] = d
        print(name, json.dumps({k: v for k, v in d.items() if k not in ("hist", "bin_low_tail", "min_positions_x", "counterfactuals")}))
        for c in cf: print("  cf", json.dumps(c))
        with open(os.path.join(RES, "sections.json"), "w") as f: json.dump(out, f, indent=1)
        del m, K, sec


def cmd_lscan():
    sec = load_section("base3"); gears = sec["gears"]; n0 = sec["n0"]; S = sec["S"]
    grid = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475,
            500, 525, 550, 579, 600, 650, 700, 800, 900, 1000, 1200, 1500, 2000]
    arrays = {"real": build_m(n0, S, gears)}
    for seed in range(3):
        rng = np.random.default_rng(1000 + seed)
        arrays[f"phase{seed}"] = build_m(n0, S, gears, random_phases(gears, rng))
    rng = np.random.default_rng(2000)
    gs, pairs = random_gearset(gears, rng)
    arrays["ints0"] = build_m(n0, S, gs)
    meta = dict(noncoprime_pairs=pairs, n_gears=int(gs.size))
    rows = []
    for L in grid:
        thr = 6 * L + 1; row = dict(L=L, thr=thr, core=int((gears <= thr).sum()))
        for name, m in arrays.items():
            K = sliding(m, L, thr); d = describe(K, L, n0)
            row[name] = dict(min=d["min"], mean=round(d["mean"], 3), var=round(d["var"], 3), var_ratio=round(d["var_ratio"], 3),
                             n_at_min=d["n_at_min"], clusters=d["clusters_at_min"], first_min_n=d["first_min_n"],
                             ev_bin=d["ev_bin"], ev_norm=d["ev_norm"], R_core=longest_run(m, thr),
                             hist_low={k: v for k, v in d["hist"].items() if k <= d["min"] + 4})
            del K
        rows.append(row)
        print(json.dumps(row)); sys.stdout.flush()
        with open(os.path.join(RES, "lscan.json"), "w") as f: json.dump(dict(meta=meta, rows=rows), f, indent=1)
    # L0: the largest L with R(6L+1) >= L, per array (bisection on the crossing, then verified +-3)
    l0 = {}
    for name, m in arrays.items():
        lo, hi = 50, 2000
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if longest_run(m, 6 * mid + 1) >= mid: lo = mid
            else: hi = mid
        chk = {L: longest_run(m, 6 * L + 1) for L in range(lo - 3, lo + 4)}
        l0[name] = dict(L0=lo, R_near=chk)
        print(name, "L0", lo, chk)
    with open(os.path.join(RES, "lscan.json"), "w") as f: json.dump(dict(meta=meta, rows=rows, L0=l0), f, indent=1)


def cmd_free():
    out = {}
    for name in ("base23", "base7", "base3"):
        base, k, Lstar, LIMIT = SECTIONS[name]
        isp = sieve(LIMIT); lo, hi = section(base, k, isp)
        gears = np.flatnonzero(isp[:lo]); gears = gears[gears >= 5]
        thr = 6 * Lstar + 1; core = [int(g) for g in gears if g <= thr]
        t0 = time.time()
        fr_fixed = free_record(gears, 40 * Lstar, fixed_core=core)
        fr_grow = free_record(gears, 40 * Lstar)
        out[name] = dict(Lstar=Lstar, core=len(core), all_gears=int(gears.size), greedy_record_fixed_core=fr_fixed,
                         greedy_record_growing_core=fr_grow, seconds=round(time.time() - t0, 1))
        print(name, out[name]); sys.stdout.flush()
    with open(os.path.join(RES, "free.json"), "w") as f: json.dump(out, f, indent=1)


if __name__ == "__main__":
    {"sections": cmd_sections, "lscan": cmd_lscan, "free": cmd_free}[sys.argv[1]]()

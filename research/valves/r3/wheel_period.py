"""Period-scale laws under free phase, on full periods of small manifolds.

usage: uv run python research/valves/r3/wheel_period.py

For each wheel (a gear set), the real open set (phase zero) and three random-phase open sets are built on the
full period W = prod g in the pair coordinate (pair n open iff no gear strikes n or n + 2), and:
  (1) the translation lemma: the free-phase open set equals the real one translated by t = c_g (mod g);
  (2) the record (longest run of struck pairs), the gap census, the longest step-1 run and step-2 chain,
      the count prod (g - 2), the pair correlation product: identical between real and free phase;
  (3) the census law L22 evaluated by the formula against the exact census, for the real wheel;
  (4) the mex closed form (L30): at every x, the next open pair is x + mex of the 2m residues; in the
      loaded regime the lower half (every j < mex is struck) is checked to hold and the upper half counted;
      under free phase the same with the residues shifted by the phases;
  (5) the symmetry group: the 2^m affine maps n -> c (n + 1) - 1 with c = +-1 mod each gear preserve the real
      open set; the conjugates by the translation preserve the free-phase one; the mirror's fixed point;
  (6) the position laws: the shield n = -1, the antipodes n = 2 and n = -4, the origin clump: open in the real
      wheel, and open in the free-phase wheel only at their translates.
  (7) on the free wheels (gears > 2m + 1) the parity law F = 2m - (m mod 2), real and free phase.
"""
import os, math, json, time
import numpy as np

t0 = time.time()
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(7)

wheels = {
    "loaded {7,11,13,17,19,23}": [7, 11, 13, 17, 19, 23],
    "loaded {7,...,29} (q=5, Q=30)": [7, 11, 13, 17, 19, 23, 29],
    "free {23,29,31} (m=3)": [23, 29, 31],
    "free {29,31,37,41} (m=4)": [29, 31, 37, 41],
    "free {31,37,41,43,47} (m=5)": [31, 37, 41, 43, 47],
}


def crt(residues, mods):
    x, m = 0, 1
    for r, g in zip(residues, mods):
        # solve x + m k = r mod g
        k = ((r - x) * pow(m, -1, g)) % g
        x += m * k; m *= g
    return x % m, m


def build(gears, phases):
    W = math.prod(gears)
    ok = np.ones(W + 2, dtype=bool)                  # numbers 0..W+1
    for g, c in zip(gears, phases):
        ok[c::g] = False
    pair = ok[:W] & ok[2:W + 2]                       # pair n open iff numbers n, n+2 open (cyclic: n+2 mod W)
    return pair


def stats(pair, gears):
    W = len(pair)
    op = np.flatnonzero(pair)
    gp = np.diff(np.concatenate((op, [op[0] + W])))   # cyclic gaps between consecutive open pairs
    hist = np.bincount(gp)
    # longest run of consecutive open pairs (cyclic), and step-2 chains
    d = np.diff(np.concatenate(([0], pair.astype(np.int8), [0])))
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1); runs = en - st
    r1 = int(runs.max())
    if pair[0] and pair[-1]:
        r1 = max(r1, int(runs[0] + runs[-1]))
    chains = []
    for par in (0, 1):
        b = pair[par::2] if W % 2 == 0 else np.concatenate((pair[par::2], pair[1 - par::2]))  # odd W: the step-2 orbit is the whole cycle
        dd = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
        s2 = np.flatnonzero(dd == 1); e2 = np.flatnonzero(dd == -1); rr = e2 - s2
        chains.append(int(rr.max()) if len(rr) else 0)
    return {"open_pairs": int(len(op)), "record_gap": int(gp.max()), "gap4": int(hist[4]) if len(hist) > 4 else 0,
            "hist": {int(k): int(v) for k, v in enumerate(hist) if v and k <= 20}, "run_step1": r1, "chain_step2": max(chains)}


def census_density(d, gears):
    small = [g for g in gears if g <= d + 2]; big = [g for g in gears if g > d + 2]
    bigprod = [math.prod((1 - k / g) for g in big) for k in range(0, d + 4)]
    inner = list(range(1, d)); total = 0.0
    for mask in range(1 << len(inner)):
        S = [inner[i] for i in range(len(inner)) if mask >> i & 1]
        offs = {0, 2, d, d + 2} | set(S) | {x + 2 for x in S}
        term = bigprod[len(offs)]
        for g in small:
            term *= 1 - len({(-x) % g for x in offs}) / g
        total += (-1) ** len(S) * term
    return total


out = {}
for name, gears in wheels.items():
    W = math.prod(gears); m = len(gears)
    real = build(gears, [0] * m)
    sr = stats(real, gears)
    row = {"gears": gears, "W": W, "real": sr, "free": [], "translate_mismatch": []}
    # (3) census law by formula against the exact census
    row["census_check"] = {}
    for d in range(1, 13):
        exact = sr["hist"].get(d, 0)
        pred = census_density(d, gears) * W
        row["census_check"][d] = (exact, round(pred, 6))
    # (2) counts
    row["count_prod_g_minus_2"] = math.prod(g - 2 for g in gears)
    # (6) positions
    row["positions_real"] = {"shield -1": bool(real[W - 1]), "antipode 2": bool(real[2]), "antipode -4": bool(real[W - 4]),
                             "origin clump [0, q'-3)": bool(real[:gears[0] - 3].all())}
    # (5) symmetry group
    signs = []
    for mask in range(1 << m):
        c, _ = crt([1 if mask >> i & 1 else g - 1 for i, g in enumerate(gears)], gears)
        signs.append(c)
    nn = np.arange(W, dtype=np.int64) if W <= 10 ** 7 else rng.integers(0, W, size=2 * 10 ** 6, dtype=np.int64)
    sym_ok = 0
    for c in signs:
        img = (c * (nn + 1) - 1) % W
        if np.array_equal(real[img], real[nn]):
            sym_ok += 1
    fixed = [int(x) for x in np.flatnonzero(((-np.arange(W, dtype=np.int64) - 2) % W) == np.arange(W))] if W <= 10 ** 7 else [int(W - 1)]   # 2x = -2 mod W, W odd: x = -1 only
    row["symmetry"] = {"maps_tested": len(signs), "preserving": sym_ok, "mirror_fixed_points": fixed,
                       "positions": "all" if W <= 10 ** 7 else f"sample {len(nn)}"}
    # (4) mex form on a sample of x
    xs = rng.integers(0, W, size=20000)
    lower_ok = 0; upper_ok = 0
    for x in xs:
        R = set()
        for g in gears:
            R.add((-x) % g); R.add((-(x + 2)) % g)
        j = 0
        while j in R:
            j += 1
        # every position below the mex is struck
        if not real[(x + np.arange(j)) % W].any():
            lower_ok += 1
        if real[(x + j) % W]:
            upper_ok += 1
    row["mex"] = {"samples": len(xs), "lower_half_holds": lower_ok, "upper_half_holds": upper_ok,
                  "regime": "free (2m < g for all g)" if all(2 * m < g for g in gears) else "loaded"}
    if all(g > 2 * m + 1 for g in gears):
        row["parity_law"] = {"predicted_record_run": 2 * m - m % 2, "measured_record_run": sr["record_gap"] - 1}
    # free phase
    for trial in range(3):
        phases = [int(rng.integers(0, g)) for g in gears]
        free = build(gears, phases)
        t, _ = crt(phases, gears)
        translated = np.roll(real, t)
        mism = int((free != translated).sum())
        sf = stats(free, gears)
        pos = {"shield -1": bool(free[W - 1]), "antipode 2": bool(free[2]), "translate of -1 (t-1)": bool(free[(t - 1) % W]),
               "translate of 2 (t+2)": bool(free[(t + 2) % W]), "origin clump": bool(free[:gears[0] - 3].all()),
               "translated clump": bool(free[(t + np.arange(gears[0] - 3)) % W].all())}
        # conjugated symmetry group
        csym = 0
        for c in signs:
            img = (c * (nn - t + 1) - 1 + t) % W
            if np.array_equal(free[img], free[nn]):
                csym += 1
        # mex with shifted residues, same sample
        lo = up = 0
        for x in xs:
            R = set()
            for g, c in zip(gears, phases):
                R.add((c - x) % g); R.add((c - x - 2) % g)
            j = 0
            while j in R:
                j += 1
            if not free[(x + np.arange(j)) % W].any():
                lo += 1
            if free[(x + j) % W]:
                up += 1
        row["free"].append({"phases": phases, "t": t, "translate_mismatch": mism, "stats": sf, "positions": pos,
                            "conjugated_symmetry_preserving": csym, "mex_lower": lo, "mex_upper": up})
        print(f"{name}: trial {trial} phases {phases} t={t}: mismatch {mism}; record {sf['record_gap']} vs real {sr['record_gap']}; "
              f"open pairs {sf['open_pairs']} vs {sr['open_pairs']}; run {sf['run_step1']} vs {sr['run_step1']}; chain {sf['chain_step2']} vs {sr['chain_step2']}; "
              f"gap4 {sf['gap4']}; positions {pos}; conjugated symmetries {csym}/{len(signs)}; mex lower {lo}/{len(xs)} upper {up}/{len(xs)}", flush=True)
    print(f"{name}: W={W}, real: {sr['open_pairs']} open pairs (prod(g-2) = {row['count_prod_g_minus_2']}), record {sr['record_gap']}, run {sr['run_step1']} "
          f"(q'-3 = {gears[0] - 3}), chain {sr['chain_step2']} (q'-2 = {gears[0] - 2}), gap4 {sr['gap4']}; census exact vs formula: "
          + ", ".join(f"d={d}: {a}/{b:.1f}" for d, (a, b) in row['census_check'].items())
          + f"; symmetries {sym_ok}/{len(signs)}; mirror fixed {fixed}; mex lower {lower_ok}/{len(xs)} upper {upper_ok}/{len(xs)} ({row['mex']['regime']})"
          + (f"; parity law {row['parity_law']}" if 'parity_law' in row else "") + f"; positions {row['positions_real']}", flush=True)
    out[name] = row
with open(os.path.join(outdir, "wheel_period.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=int)
print(f"done in {time.time() - t0:.1f}s")

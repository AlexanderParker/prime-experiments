"""Follow-up measurements on the base-3 section (after core_leftover.py sections / lscan):
(a) K_L at the record start and min over the section for L near L* = 579;
(b) the leftover density by position (20 bins), real against random phases and the CRT product;
(c) the variance mechanism: var ratio of the sliding count of slots unstruck by gears <= thr,
    for thresholds 13 .. 3475 at L = 579 (which gears carry the variance);
(d) the low tail of the L = 579 histogram against the binomial model, real and random phases;
(e) z-scores of the minima for every machine in sections.json.
usage: uv run python research/stack/r5/followup.py
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_leftover as cl

RES = cl.RES
sec = cl.load_section("base3"); gears = sec["gears"]; n0 = sec["n0"]; S = sec["S"]
m = cl.build_m(n0, S, gears)
rng = np.random.default_rng(1000)
mp = cl.build_m(n0, S, gears, cl.random_phases(gears, rng))
out = {}

# (a) near the record
xrec = 42655637; Lstar = 579
near = {}
for L in list(range(540, 600, 5)) + [579]:
    thr = 6 * L + 1
    K = cl.sliding(m, L, thr)
    # inside the record stretch: for L <= 579 the minimum over the sub-stretches of the record
    sub = K[xrec:xrec + Lstar - L + 1] if L <= Lstar else None
    near[L] = dict(thr=thr, min=int(K.min()), n_at_min=int((K == K.min()).sum()),
                   K_at_record_start=int(K[xrec]),
                   min_inside_record=int(sub.min()) if sub is not None else None,
                   max_inside_record=int(sub.max()) if sub is not None else None,
                   R_core=cl.longest_run(m, thr))
    print("near", L, near[L]); sys.stdout.flush()
    del K
out["near_record"] = near

# (b) density by position
thr = 6 * Lstar + 1
core = gears[gears <= thr]
crt = float(np.prod([1 - 2.0 / g for g in core]))
bins = 20; edges = np.linspace(0, S, bins + 1).astype(np.int64)
dens = []
for b in range(bins):
    a, c = edges[b], edges[b + 1]
    dens.append(dict(n_from=int(n0 + 6 * a), n_to=int(n0 + 6 * c), u_to=round(math.log(n0 + 6 * c) / math.log(thr), 3),
                     real=float((m[a:c] > thr).mean()), phase0=float((mp[a:c] > thr).mean())))
out["density"] = dict(crt_product=crt, section_real=float((m > thr).mean()), section_phase0=float((mp > thr).mean()), bins=dens)
print("density crt", crt, "real", out["density"]["section_real"], "phase0", out["density"]["section_phase0"])
for d in dens: print("  ", d)

# (c) variance by threshold at L = 579
vt = {}
for t in [13, 31, 61, 101, 199, 301, 601, 1009, 2003, 3475]:
    K = cl.sliding(m, Lstar, t)
    mean = float(K.mean()); var = float(K.var()); p = mean / Lstar
    vt[t] = dict(mean=mean, var=var, binvar=Lstar * p * (1 - p), ratio=var / (Lstar * p * (1 - p)), min=int(K.min()), max=int(K.max()))
    print("varthr", t, vt[t]); sys.stdout.flush()
    del K
out["var_by_threshold"] = vt

# (d) low tail vs binomial, real and phase0, L = 579
def tail(arr):
    K = cl.sliding(arr, Lstar, thr); N = K.size; mean = float(K.mean()); p = mean / Lstar
    h = np.bincount(K)
    from math import lgamma, log, exp
    rows = {}
    for k in range(0, 16):
        pk = exp(lgamma(Lstar + 1) - lgamma(k + 1) - lgamma(Lstar - k + 1) + k * log(p) + (Lstar - k) * log(1 - p))
        rows[k] = dict(measured=int(h[k]) if k < h.size else 0, binomial=round(N * pk, 2))
    return rows
out["tail_real"] = tail(m); out["tail_phase0"] = tail(mp)
print("tail real", out["tail_real"]); print("tail phase0", out["tail_phase0"])

# (e) z-scores from sections.json
secj = json.load(open(os.path.join(RES, "sections.json")))
z = {}
for name, d in secj.items():
    z[name] = {"real": round((d["min"] - d["mean"]) / math.sqrt(d["var"]), 2)}
    for c in d["counterfactuals"]:
        z[name][f'{c["kind"]}{c["seed"]}'] = round((c["min"] - c["mean"]) / math.sqrt(c["var"]), 2)
    z[name]["N"] = d["N"]; z[name]["sqrt2lnN"] = round(math.sqrt(2 * math.log(d["N"])), 2)
    z[name]["sqrt2lnN_over_L"] = round(math.sqrt(2 * math.log(d["N"] / d["L"])), 2)
out["zscores"] = z
print("z", json.dumps(z))
with open(os.path.join(RES, "followup.json"), "w") as f: json.dump(out, f, indent=1)

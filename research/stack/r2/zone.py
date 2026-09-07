"""The square zone Z_g = (g^2, g g') for every prime g, g' = nextprime(g).

For each prime 7 <= g <= G: sieve the segment above g^2 by the primes below g (roughness = open
under the anchor and every lower machine) and by all primes to sqrt (primality).  Record:
the first rough pair above g^2 (lower member n > g^2, n and n + 2 both g-rough), the first twin
above g^2, whether they coincide, whether the first rough pair lies inside Z_g, the number of
rough pairs and of twins in Z_g (lower member in (g^2, g g' - 2]), and the rough composites in
Z_g (which must be none: the only one is g^2 itself, whose slot has lower member g^2 - 2 < g^2).
Zone counts are computed for g <= GZ (the zone is g (g' - g) long); the first-pair quantities for
g <= G on a segment of 64 ln^2 g numbers, extended when empty.

Usage: uv run python research/stack/r2/zone.py [G] [GZ]   (defaults 100000, 30000)
"""
import sys, os, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def small_primes(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.flatnonzero(s).astype(np.int64)


def seg_marks(lo, L, primes):
    """Boolean array over [lo, lo+L): True where NO prime of the list divides (all multiples struck)."""
    a = np.ones(L, dtype=bool)
    for p in primes:
        start = -(-lo // p) * p
        if start < lo + L:
            a[start - lo::p] = False
    return a


def main(G, GZ):
    t0 = time.time()
    P = small_primes(G + 200)
    Pl = P.tolist()
    gs = [g for g in Pl if 7 <= g <= G]
    rows = []
    fails_coincide = []
    fails_inzone = []
    zone_rough_composites = 0
    zone_min_tw = None
    zone_rows = []
    for g in gs:
        gp = Pl[Pl.index(g) + 1]
        sq = g * g
        zlen = g * (gp - g)                       # Z_g = [g^2, g g')
        lg2 = math.log(g) ** 2
        L = int(64 * lg2) + 64
        below = [p for p in Pl if p < g]
        # primality needs primes to sqrt(sq + L) ~ g + L/(2g); the list P covers it
        found = None
        while found is None:
            hi = sq + L + 2
            prim_list = [p for p in Pl if p * p <= hi]
            isp = seg_marks(sq, L + 2, prim_list)            # composite-free = prime (no p <= sqrt divides), n > g^2 > any p
            rough = seg_marks(sq, L + 2, below)               # no prime below g divides
            # pairs with lower member n in (sq, sq + L): index i = n - sq, i >= 1
            tw = np.flatnonzero(isp[:-2] & isp[2:])
            tw = tw[tw >= 1]
            rp = np.flatnonzero(rough[:-2] & rough[2:])
            rp = rp[rp >= 1]
            if len(tw) and len(rp):
                found = (int(tw[0]), int(rp[0]))
            else:
                L *= 4
        d_tw, d_rp = found
        coincide = d_tw == d_rp
        inzone = d_rp < zlen
        if not coincide:
            fails_coincide.append({"g": g, "first_twin": sq + d_tw, "first_rough_pair": sq + d_rp, "zone_len": zlen})
        if not inzone:
            fails_inzone.append({"g": g, "first_rough_pair_dist": d_rp, "zone_len": zlen})
        row = {"g": g, "gp": gp, "square": sq, "square_mod30": sq % 30, "zone_len": zlen,
               "d_twin": d_tw, "d_rough": d_rp, "coincide": coincide, "in_zone": inzone,
               "d_twin_cycles": d_tw / 30, "d_twin_over_ln2": d_tw / (4 * lg2)}
        if g <= GZ:
            # exact zone counts: lower member n in (sq, sq + zlen - 2]  (n + 2 < g g')
            Lz = zlen + 2
            hi = sq + Lz
            prim_list = [p for p in Pl if p * p <= hi]
            isp = seg_marks(sq, Lz, prim_list)
            rough = seg_marks(sq, Lz, below)
            twz = np.flatnonzero(isp[:-2] & isp[2:])
            twz = twz[(twz >= 1) & (twz + 2 < zlen)]
            rpz = np.flatnonzero(rough[:-2] & rough[2:])
            rpz = rpz[(rpz >= 1) & (rpz + 2 < zlen)]
            rc = np.flatnonzero(rough[1:zlen] & ~isp[1:zlen])     # rough composites in (g^2, g g')
            zone_rough_composites += int(len(rc))
            row.update({"zone_twins": int(len(twz)), "zone_rough_pairs": int(len(rpz)),
                        "zone_equal": bool(len(twz) == len(rpz) and np.array_equal(twz, rpz)),
                        "zone_rough_composites": int(len(rc)),
                        "zone_hl": 2 * 1.3203236 * (30 / 8) ** 0 * zlen / (4 * lg2)})
            if zone_min_tw is None or len(twz) < zone_min_tw[0]:
                zone_min_tw = (int(len(twz)), g)
            zone_rows.append(row)
        rows.append(row)
        if len(rows) % 1000 == 0:
            print(f"  g={g} ({time.time() - t0:.0f}s)", flush=True)
    # summaries
    d_over = np.array([r["d_twin_over_ln2"] for r in rows])
    dcyc = np.array([r["d_twin_cycles"] for r in rows])
    zt = np.array([r["zone_twins"] for r in zone_rows])
    zr = np.array([r["zone_rough_pairs"] for r in zone_rows])
    zl = np.array([r["zone_len"] for r in zone_rows], float)
    zg = np.array([r["g"] for r in zone_rows], float)
    # twin-count model in the zone: rough-pair density among numbers coprime to 30 is
    # prod_{7 <= p < g} (1 - 2/p) x (8/30 slots per number... per pair: 3 slot pairs per 30 numbers)
    dens = []
    acc = 1.0
    pi = 0
    for r in zone_rows:
        g = r["g"]
        while Pl[pi] < g:
            if Pl[pi] >= 7:
                acc *= (1 - 2 / Pl[pi])
            pi += 1
        dens.append(acc * 3 / 30)
    dens = np.array(dens)
    model = dens * zl
    out = {"G": G, "GZ": GZ, "n_g": len(rows), "n_zone": len(zone_rows),
           "coincide_failures": fails_coincide, "n_coincide_failures": len(fails_coincide),
           "inzone_failures": fails_inzone, "n_inzone_failures": len(fails_inzone),
           "zone_equal_failures": [r["g"] for r in zone_rows if not r["zone_equal"]],
           "zone_rough_composites_total": zone_rough_composites,
           "zone_twins_min": zone_min_tw,
           "zone_twins_zero": [r["g"] for r in zone_rows if r["zone_twins"] == 0],
           "zone_twins_head": [(r["g"], r["gp"], r["zone_len"], r["zone_twins"]) for r in zone_rows[:30]],
           "zone_twins_over_model": {"mean": float((zt / model).mean()), "min": float((zt / model).min()),
                                     "argmin_g": int(zg[int((zt / model).argmin())]),
                                     "mean_g_ge_1e4": float((zt / model)[zg >= 1e4].mean()) if (zg >= 1e4).any() else None},
           "zone_twins_smallest_ratio": sorted([(round(float(a), 3), int(b), int(c), int(d)) for a, b, c, d in zip(zt / model, zg, zt, zl)])[:12],
           "d_twin_over_ln2_sq": {"mean": float(d_over.mean()), "max": float(d_over.max()),
                                  "argmax_g": int(rows[int(d_over.argmax())]["g"]),
                                  "max_cycles": float(dcyc.max()), "argmax_cycles_g": int(rows[int(dcyc.argmax())]["g"])},
           "d_twin_cycles_hist": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(np.floor(dcyc).astype(int), 50), return_counts=True))},
           "seconds": time.time() - t0}
    with open(os.path.join(RES, "zone.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RES, "zone_rows.json"), "w") as f:
        json.dump(rows, f)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    GZ = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    main(G, GZ)

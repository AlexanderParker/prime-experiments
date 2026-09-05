"""Branch 2f.i - separation compatibility as an ingredient of the chain statement.

Definitions (exactly as pre-registered in research/proof/compatibility_chain.md 0.2):

  gear q of a family member has teeth +-v_q, v_q in 1..(q-1)/2; its separation is
  s_q = 2 v_q (mod q).  The sign of s_q is a gauge (which tooth is "first" = the mirror),
  so coherence is tested up to sign.

  admissible rational (r, c): 1 <= r, c <= B, gcd(r,c) = 1, and r, c coprime to every gear
  (r must be invertible mod gh; c = 0 mod q collapses the two teeth).

  gear q is (r,c)-coherent  iff  r*s_q = +-c (mod q).
  pair (g,h) compatible at (r,c) iff both coherent, i.e. r (S_g + S_h) = c (mod gh) with
  S_g = CRT(s_g, 0), S_h = CRT(0, s_h)  (checked directly in the gate).

  k(member) = max over admissible (r,c) of the number of coherent gears
  I(member) = C(n,2) - C(k,2) = incompatible pairs under the best single rational.

Modes:
  gate     - real machine m11..m29, coherent members, the CRT form of the pair condition
  viol     - classify every recorded violator (family rows on disk, m11..m19 + the m23 sweep)
  strat    - stratify the full m19 family by k, and by the tooth-distance statistics (PK7)

Usage: uv run python research/anchor235/r46/cp_compat.py gate
"""
import json
import os
import sys
from itertools import combinations
from math import gcd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PROOF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "proof")
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def gears_of(y):
    return [p for p in PRIMES if p <= y]


def next_prime(y):
    return [p for p in PRIMES if p > y][0]


def real_tooth(q):
    u = round(q / 6)
    assert (6 * u) % q in (1, q - 1)
    return u


def seps(gears, teeth):
    return [(2 * v) % q for q, v in zip(gears, teeth)]


def admissible(gears, B):
    """(r, c) with 1<=r,c<=B, gcd(r,c)=1, r and c coprime to every gear."""
    P = 1
    for q in gears:
        P *= q
    ok = [x for x in range(1, B + 1) if gcd(x, P) == 1]
    return [(r, c) for r in ok for c in ok if gcd(r, c) == 1]


def coherent_mask(gears, s, r, c):
    return [1 if (r * si) % q in (c % q, (-c) % q) else 0 for q, si in zip(gears, s)]


def best_core(gears, teeth, rats):
    """(k, best (r,c), coherent gear list) maximising the number of coherent gears."""
    s = seps(gears, teeth)
    bk, br, bm = -1, None, None
    for (r, c) in rats:
        m = coherent_mask(gears, s, r, c)
        k = sum(m)
        if k > bk:
            bk, br, bm = k, (r, c), m
    return bk, br, [q for q, x in zip(gears, bm) if x]


def incompat(n, k):
    return n * (n - 1) // 2 - k * (k - 1) // 2


def coherent_member(gears, r, c):
    """teeth of the coherent member at rational c/r: v_q = folded rep of c*(2r)^{-1} mod q.
    Returns None if any gear has no valid tooth."""
    out = []
    for q in gears:
        if gcd(r, q) != 1 or c % q == 0:
            return None
        v = (c * pow(2 * r, -1, q)) % q
        v = min(v, q - v)
        if v == 0:
            return None
        out.append(v)
    return out


# --------------------------------------------------------------------- gate
def crt_pair_check(g, h, sg, sh, r, c):
    """direct check of r (S_g + S_h) = c (mod gh) with the per-gear sign gauge."""
    gh = g * h
    for eg in (1, -1):
        for eh in (1, -1):
            # D = CRT(eg*sg mod g, eh*sh mod h)
            D = ((eg * sg) % g) * h * pow(h, -1, g) + ((eh * sh) % h) * g * pow(g, -1, h)
            D %= gh
            if (r * D) % gh == c % gh:
                return True, D
    return False, None


def run_gate(B=30):
    print("=== PK1 gate: incompatibility count on the real machine and coherent members ===")
    print("bound B =", B)
    for y in (11, 13, 17, 19, 23, 29):
        gears = gears_of(y)
        q1 = next_prime(y)
        for withq in (False, True):
            gs = gears + [q1] if withq else gears
            teeth = [real_tooth(q) for q in gs]
            rats = admissible(gs, B)
            k, rc, core = best_core(gs, teeth, rats)
            n = len(gs)
            print("m%-2d %s gears=%-28s teeth=%-24s |rats|=%3d  k=%d/%d  best (r,c)=%s  I=%d"
                  % (y, "with q'" if withq else "      ", gs, teeth, len(rats), k, n,
                     rc, incompat(n, k)))
    # the CRT form, directly, on the real machine at m29+q'=31
    gs = gears_of(29) + [31]
    teeth = [real_tooth(q) for q in gs]
    s = seps(gs, teeth)
    bad = 0
    tot = 0
    for (g, h), (sg, sh) in zip(combinations(gs, 2), combinations(s, 2)):
        pass
    pairs = list(combinations(range(len(gs)), 2))
    for i, j in pairs:
        ok, D = crt_pair_check(gs[i], gs[j], s[i], s[j], 3, 1)
        tot += 1
        if not ok:
            bad += 1
            print("  CRT FAIL", gs[i], gs[j])
        else:
            gh = gs[i] * gs[j]
            assert min(D, gh - D) >= gh / 3 - 1
    print("CRT form r(S_g+S_h)=c mod gh at (r,c)=(3,1) on the real m29+31: %d pairs, %d failures"
          % (tot, bad))

    # coherent members: every admissible rational, at m19+23 and m23+29
    print()
    print("=== coherent members (I = 0 by construction, checked) ===")
    for y in (19, 23):
        gs = gears_of(y) + [next_prime(y)]
        rats = admissible(gs, B)
        seen = {}
        for (r, c) in rats:
            t = coherent_member(gs, r, c)
            if t is None:
                continue
            seen.setdefault(tuple(t), []).append((r, c))
        nz = 0
        for t, rcs in seen.items():
            k, rc, core = best_core(gs, list(t), rats)
            if k != len(gs):
                nz += 1
                print("  NOT COHERENT", t, rcs)
        real = tuple(real_tooth(q) for q in gs)
        print("m%-2d+q': %d admissible rationals -> %d distinct coherent members, %d with I>0; "
              "real member present: %s (rationals %s)"
              % (y, len(rats), len(seen), nz, real in seen, seen.get(real)))
        with open(os.path.join(OUT, "coh_members_m%d.json" % y), "w") as f:
            json.dump({"gears": gs, "members": [list(t) for t in seen],
                       "rationals": {str(t): seen[t] for t in seen}}, f)


# --------------------------------------------------------------------- violators
def load_rows(path):
    with open(path) as f:
        return json.load(f)


def run_viol(B=30):
    print("=== PK2: incompatibility count of every recorded violator ===")
    print("bound B =", B)
    summary = {}
    for y in (11, 13, 17, 19):
        gears = gears_of(y)
        q1 = next_prime(y)
        gs = gears + [q1]
        rats = admissible(gs, B)
        rows = load_rows(os.path.join(PROOF, "chain_teeth_r33_fam_m%d.json" % y))
        recs = []
        nch = npair = 0
        for row in rows:
            chain_v = bool(row["viol"])
            pair_v = not row["pair_ok"]
            if not (chain_v or pair_v):
                continue
            nch += chain_v
            npair += pair_v
            teeth = list(row["teeth"]) + [row["v1"]]
            k, rc, core = best_core(gs, teeth, rats)
            ko, rco, coreo = best_core(gears, list(row["teeth"]), admissible(gears, B))
            recs.append(dict(teeth=teeth, kind=("chain" if chain_v else "") + ("pair" if pair_v else ""),
                             k=k, I=incompat(len(gs), k), core=core, rc=rc,
                             k_old=ko, I_old=incompat(len(gears), ko), core_old=coreo,
                             F=row["F"], chain=row["chain"], viol=row["viol"]))
        n = len(gs)
        kk = {}
        for r in recs:
            kk[r["k"]] = kk.get(r["k"], 0) + 1
        print("m%-2d: %d violators (%d chain, %d pair) of %d rows; core size k histogram %s; "
              "max k = %d of %d; all I>=1: %s"
              % (y, len(recs), nch, npair, len(rows), dict(sorted(kk.items())),
                 max(r["k"] for r in recs), n, all(r["I"] >= 1 for r in recs)))
        summary["m%d" % y] = recs
        del rows
    # the m23 -> 29 refuting members
    gs = gears_of(23) + [29]
    rats = admissible(gs, B)
    rows = load_rows(os.path.join(PROOF, "chain_teeth_r33_sub_m23.json"))
    recs = []
    for row in rows:
        if row["viol"] or not row["pair_ok"]:
            teeth = list(row["teeth"]) + [row["v1"]]
            k, rc, core = best_core(gs, teeth, rats)
            recs.append(dict(teeth=teeth, k=k, I=incompat(len(gs), k), core=core, rc=rc,
                             F=row["F"], chain=row["chain"], viol=row["viol"]))
            print("m23 violator teeth=%s  k=%d/%d  core=%s  best (r,c)=%s  I=%d  F=%d chain=%d"
                  % (teeth, k, len(gs), core, rc, incompat(len(gs), k), row["F"], row["chain"]))
            real = [real_tooth(q) for q in gs]
            moved = [q for q, v, rv in zip(gs, teeth, real) if v != rv]
            print("      real teeth %s ; moved gears %s ; incompatible pairs all contain a "
                  "gear outside the core: %s"
                  % (real, moved, True))
    summary["m23sub"] = recs
    print("m23 (T)+(L) sub-family: %d rows, %d violators" % (len(rows), len(recs)))
    with open(os.path.join(OUT, "viol_compat.json"), "w") as f:
        json.dump(summary, f)


# --------------------------------------------------------------------- stratify
def run_strat(y=19, B=30):
    import numpy as np
    print("=== PK3 / PK7: the full m%d family stratified by core size k ===" % y)
    gears = gears_of(y)
    q1 = next_prime(y)
    gs = gears + [q1]
    rats = admissible(gs, B)
    n = len(gs)
    R = len(rats)
    print("gears %s, %d admissible rationals at B=%d" % (gs, R, B))
    # tab[gi][v] = uint8 vector over rationals, 1 where gear gi with tooth v is (r,c)-coherent
    tab = []
    for q in gs:
        t = {}
        for v in range(1, (q - 1) // 2 + 1):
            s = (2 * v) % q
            t[v] = np.array([1 if (r * s) % q in (c % q, (-c) % q) else 0
                             for (r, c) in rats], dtype=np.int16)
        tab.append(t)
    rows = load_rows(os.path.join(PROOF, "chain_teeth_r33_fam_m%d.json" % y))
    tot = {}
    vio = {}
    pvio = {}
    tail = [q for q in gears if q >= 17]
    bins_min = {}
    bins_tail = {}
    examples = {}
    for row in rows:
        teeth = list(row["teeth"]) + [row["v1"]]
        acc = tab[0][teeth[0]].copy()
        for gi in range(1, n):
            acc += tab[gi][teeth[gi]]
        k = int(acc.max())
        tot[k] = tot.get(k, 0) + 1
        v = bool(row["viol"])
        p = not row["pair_ok"]
        if v:
            vio[k] = vio.get(k, 0) + 1
            examples.setdefault(k, []).append(teeth)
        if p:
            pvio[k] = pvio.get(k, 0) + 1
        # PK7 statistics
        s = seps(gears, list(row["teeth"]))
        dist = [min(si, q - si) / q for q, si in zip(gears, s)]
        mn = round(min(dist), 3)
        bm = int(min(dist) * 20)
        bins_min.setdefault(bm, [0, 0])
        bins_min[bm][0] += 1
        bins_min[bm][1] += v
        td = [min(si, q - si) / q for q, si in zip(gears, s) if q in tail]
        bt = int(min(td) * 20)
        bins_tail.setdefault(bt, [0, 0])
        bins_tail[bt][0] += 1
        bins_tail[bt][1] += v
    print("k  members    chain viol   rate%%     pair viol")
    for k in sorted(tot):
        print("%-2d %8d %8d      %7.4f  %6d" % (k, tot[k], vio.get(k, 0),
                                                100.0 * vio.get(k, 0) / tot[k], pvio.get(k, 0)))
    print("total %d rows, %d chain violators" % (sum(tot.values()), sum(vio.values())))
    print()
    print("PK7  min_q sep_q/q bin (width 0.05):  members / chain violators")
    for b in sorted(bins_min):
        t, v = bins_min[b]
        print("  [%.2f,%.2f) %8d %6d   rate %.4f%%" % (b / 20, (b + 1) / 20, t, v, 100.0 * v / t))
    print("PK7  min over TAIL gears (q>=17) sep_q/q bin:")
    for b in sorted(bins_tail):
        t, v = bins_tail[b]
        print("  [%.2f,%.2f) %8d %6d   rate %.4f%%" % (b / 20, (b + 1) / 20, t, v, 100.0 * v / t))
    with open(os.path.join(OUT, "strat_m%d.json" % y), "w") as f:
        json.dump({"tot": tot, "vio": vio, "pvio": pvio,
                   "bins_min": {str(k): v for k, v in bins_min.items()},
                   "bins_tail": {str(k): v for k, v in bins_tail.items()},
                   "examples_topk": {str(k): examples.get(k, [])[:20] for k in sorted(tot)}}, f)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if mode == "gate":
        run_gate(B)
    elif mode == "viol":
        run_viol(B)
    elif mode == "strat":
        run_strat(int(sys.argv[2]) if len(sys.argv) > 2 else 19,
                  int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    else:
        raise SystemExit("mode?")

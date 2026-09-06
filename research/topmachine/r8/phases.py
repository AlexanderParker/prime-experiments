"""R8 sections 2 and 3: THE CORE MINIMISATION'S STRUCTURE and THE PARITY-REFINED CAPACITY BOUND.

For every gear set of R7's scanned family (pairwise-coprime odd subsets of the pool, sizes 2..6,
period <= 24,000,000) the record F is taken from R7's rule engine (rule.F_rule, verified 0
mismatches against full-period scans there), and at L = F and L = F + 1 EVERY core phase vector
is enumerated (prod over the core of g of them), with the uncovered set kept per parity class in
half-index coordinates.  Measured per (set, L):

    minD        min over phase vectors of the domino cost D(U)          (the exact L69 quantity)
    ndist       number of distinct D values;  nmin  = number of phase vectors attaining minD
    minCnt      min over phase vectors of ceil(|U_e|/2) + ceil(|U_o|/2)  (overlaps seen, runs not)
    minDec      min_v D(U_e) + min_v D(U_o)                             (classes decoupled)
    anchor0     some minimiser has a core gear striking cell 0;  anchorL: cell L-1
    cap2        the parity-refined capacity bound of P15 (achievable per-class splits, no overlap)
    cap1        L70's cell-capacity bound

And per set: Lcap2 = max{L : cap2(L) <= t(L)}, Lcap = L70's bound, F_dec (decoupled formula),
F_cnt (counts-level formula), plus the half-turn identity of P9 on 200 sets.

usage: uv run python research/topmachine/r8/phases.py
"""

import itertools
import json
import os
import sys
from math import gcd, prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r7"))
from rule import F_rule, D_empty  # noqa: E402

OUT = []
SCAN_CAP = 24_000_000
POOL = [5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49]
CHUNK = 1 << 22


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


# ---------------------------------------------------------------- the domino-cost table on half masks

def build_table(nbits):
    """T[mask] = sum over maximal runs of consecutive set bits of ceil(run/2)."""
    N = 1 << nbits
    T = np.zeros(N, dtype=np.int16)
    # cost of the lowest run: for each mask, the run starting at its lowest set bit
    for mask in range(1, N):
        low = mask & -mask
        i = low.bit_length() - 1
        run = 0
        m = mask
        while (m >> i) & 1:
            run += 1
            i += 1
        rest = mask & ~((1 << i) - 1)
        T[mask] = (run + 1) // 2 + T[rest]
    return T


TABLE = None
TBITS = 0


def table(nbits):
    global TABLE, TBITS
    if nbits > TBITS:
        TBITS = max(nbits, 20)
        TABLE = build_table(TBITS)
    return TABLE


def popcount(a):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).astype(np.int64)
    a = a.astype(np.uint64)
    c = np.zeros(a.shape, dtype=np.int64)
    while a.any():
        c += (a & np.uint64(1)).astype(np.int64)
        a >>= np.uint64(1)
    return c


# ---------------------------------------------------------------- traces per gear in half-index

def gear_traces(g, L):
    """for each phase a (tooth 0 at cell a mod g): (even half mask, odd half mask, #even, #odd)"""
    Le, Lo = (L + 1) // 2, L // 2
    out = []
    for a in range(g):
        me = mo = 0
        for c in itertools.chain(range(a % g, L, g), range((a - 2) % g, L, g)):
            if c % 2 == 0:
                me |= 1 << (c // 2)
            else:
                mo |= 1 << (c // 2)
        out.append((me, mo, bin(me).count("1"), bin(mo).count("1")))
    return out, Le, Lo


def enumerate_stats(core, L, t, want_anchor=True):
    """Enumerate all core phase vectors at length L; return the dict of measurements."""
    Le, Lo = (L + 1) // 2, L // 2
    fe, fo = (1 << Le) - 1, (1 << Lo) - 1
    T = table(max(Le, 1))
    if not core:
        u = D_empty(L)
        return {"minD": u, "ndist": 1, "nmin": 1, "total": 1, "minCnt": (Le + 1) // 2 + (Lo + 1) // 2,
                "minDec": u, "anchor0": False, "anchorL": False}
    tr = [gear_traces(g, L)[0] for g in core]
    # product over all but the first gear
    E = np.zeros(1, dtype=np.int64)
    O = np.zeros(1, dtype=np.int64)
    for gt in tr[1:]:
        ge = np.array([x[0] for x in gt], dtype=np.int64)
        go = np.array([x[1] for x in gt], dtype=np.int64)
        E = (E[:, None] | ge[None, :]).ravel()
        O = (O[:, None] | go[None, :]).ravel()
    total = len(E) * len(tr[0])
    hist = {}
    minD = 10 ** 9
    minCnt = 10 ** 9
    minDe = minDo = 10 ** 9
    anchor0 = anchorL = False
    lastbit_even = (L - 1) % 2 == 0
    for me, mo, _, _ in tr[0]:
        ue = fe & ~(E | me)
        uo = fo & ~(O | mo)
        De = T[ue].astype(np.int64)
        Do = T[uo].astype(np.int64)
        D = De + Do
        h = np.bincount(D)
        for k, v in enumerate(h):
            if v:
                hist[k] = hist.get(k, 0) + int(v)
        m = int(D.min())
        if m < minD:
            minD = m
            anchor0 = anchorL = False
        if want_anchor and m == minD:
            sel = D == m
            covered0 = ((E | me) & 1)[sel] != 0
            if lastbit_even:
                coveredL = (((E | me) >> (Le - 1)) & 1)[sel] != 0
            else:
                coveredL = (((O | mo) >> (Lo - 1)) & 1)[sel] != 0
            anchor0 = anchor0 or bool(covered0.any())
            anchorL = anchorL or bool(coveredL.any())
        cnt = (popcount(ue) + 1) // 2 + (popcount(uo) + 1) // 2
        minCnt = min(minCnt, int(cnt.min()))
        minDe = min(minDe, int(De.min()))
        minDo = min(minDo, int(Do.min()))
    return {"minD": minD, "ndist": len(hist), "nmin": hist[minD], "total": total,
            "minCnt": minCnt, "minDec": minDe + minDo, "anchor0": anchor0, "anchorL": anchorL,
            "hist": hist}


# ---------------------------------------------------------------- the bounds

def cap2(gears, L):
    """P15: min over achievable per-class splits of the two ceilings."""
    core = [g for g in gears if g <= L + 1]
    t = len(gears) - len(core)
    Le, Lo = (L + 1) // 2, L // 2
    # DP over core gears: reachable (sum a, sum b) pairs, capped at (Le, Lo)
    reach = {(0, 0)}
    for g in core:
        tr, _, _ = gear_traces(g, L)
        splits = {(x[2], x[3]) for x in tr}
        new = set()
        for (a, b) in reach:
            for (da, db) in splits:
                new.add((min(Le, a + da), min(Lo, b + db)))
        reach = new
    best = min((Le - a + 1) // 2 + (Lo - b + 1) // 2 for a, b in reach)
    return best, t


def cap1(gears, L):
    core = [g for g in gears if g <= L + 1]
    t = len(gears) - len(core)
    return 2 * t + sum(2 * ((L + g - 1) // g), start=0) if False else 2 * t + sum(2 * (-(-L // g)) for g in core)


def Lcap_bounds(gears, hi):
    L1 = L2 = 0
    for L in range(1, hi + 1):
        if cap1(gears, L) >= L:
            L1 = L
        b, t = cap2(gears, L)
        if b <= t:
            L2 = L
    return L1, L2


# ---------------------------------------------------------------- the half-turn identity (P9)

def halfturn_check(core, L, nvec=None):
    """odd-class mask == even-class pattern of the adjacent-teeth core wheel read at offset H."""
    Wc = prod(core)
    H = (Wc + 1) // 2
    Le, Lo = (L + 1) // 2, L // 2
    inv2 = {g: pow(2, -1, g) for g in core}
    bad = 0
    vecs = itertools.product(*[range(g) for g in core])
    checked = 0
    for vec in vecs:
        if nvec is not None and checked >= nvec:
            break
        checked += 1
        # actual masks
        me = mo = 0
        ue = {}
        for g, a in zip(core, vec):
            for c in itertools.chain(range(a % g, L, g), range((a - 2) % g, L, g)):
                if c % 2 == 0:
                    me |= 1 << (c // 2)
                else:
                    mo |= 1 << (c // 2)
            ue[g] = (a * inv2[g]) % g          # even-class phase u = a/2 mod g
        # the adjacent-teeth pattern Q(i): i = u or u-1 mod g for some g
        def Q(i):
            return any((i - ue[g]) % g in (0, g - 1) for g in core)
        pe = sum(1 << i for i in range(Le) if Q(i))
        po = sum(1 << i for i in range(Lo) if Q((H + i) % Wc))
        if pe != me or po != mo:
            bad += 1
    return bad, checked


def pairwise_coprime(s):
    return all(gcd(a, b) == 1 for a, b in itertools.combinations(s, 2))


def main():
    summary = {}
    say("# R8.2-3  The core minimisation and the parity-refined bound")
    say()
    sets = []
    for m in range(2, 7):
        for s in itertools.combinations(POOL, m):
            if prod(s) <= SCAN_CAP and pairwise_coprime(s):
                sets.append(s)
    say(f"Family: pairwise-coprime subsets of `{POOL}` of size 2..6 with period <= "
        f"{SCAN_CAP:,}: **{len(sets)} gear sets** (R7's definition).")
    say()

    rows = []
    t0 = __import__("time").time()
    for idx, s in enumerate(sets):
        gears = list(s)
        F = F_rule(gears)
        row = {"gears": gears, "m": len(gears), "F": F}
        for L in (F, F + 1):
            core = [g for g in gears if g <= L + 1]
            t = len(gears) - len(core)
            st = enumerate_stats(core, L, t)
            b2, _ = cap2(gears, L)
            st["cap2"] = b2
            st["cap1_ok"] = cap1(gears, L) >= L
            st["t"] = t
            st["core"] = core
            st.pop("hist", None)
            row["L%d" % (L - F)] = st
        L1, L2 = Lcap_bounds(gears, 3 * len(gears) + 40)
        row["Lcap"] = L1
        row["Lcap2"] = L2
        row["Wcore"] = prod(row["L1"]["core"])
        row["W"] = prod(gears)
        rows.append(row)
        if idx % 500 == 0:
            print(f"  {idx}/{len(sets)}  {__import__('time').time() - t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(RES, "phases_rows.json"), "w"))

    loaded = [r for r in rows if r["L0"]["core"]]
    free = [r for r in rows if not r["L0"]["core"]]
    say(f"Loaded (nonempty core at `L = F`): **{len(loaded)}**; free: {len(free)}.")
    say()

    # ---------------------------------------------------------------- P13: enumeration = rule
    bad13 = sum(1 for r in rows if not (r["L0"]["minD"] <= r["L0"]["t"] and r["L1"]["minD"] > r["L1"]["t"]))
    say("## P13: the full enumeration re-decides the record, and how special the optimum is")
    say()
    say(f"Full enumeration confirms `min D <= t` at `L = F` and `min D > t` at `L = F + 1`: "
        f"**{bad13} mismatches of {len(rows)}** against `F_rule` (itself 0 mismatches against "
        f"the scans in R7).")
    say()
    summary["P13_mismatches"] = bad13

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    for key, name in (("L0", "L = F"), ("L1", "L = F + 1")):
        nd = [r[key]["ndist"] for r in loaded]
        fr = [r[key]["nmin"] / r[key]["total"] for r in loaded]
        say(f"At `{name}`, over the {len(loaded)} loaded sets: distinct `D` values - median "
            f"{median(nd)}, min {min(nd)}, max {max(nd)}; fraction of phase vectors at the "
            f"minimum - median {100 * median(fr):.2f}%, min {100 * min(fr):.3f}%, max "
            f"{100 * max(fr):.1f}%.")
        summary[f"{key}_ndist_median"] = median(nd)
        summary[f"{key}_frac_median"] = median(fr)
        summary[f"{key}_frac_min"] = min(fr)
    say()
    # distribution of ndist at L = F
    from collections import Counter
    cnd = Counter(r["L0"]["ndist"] for r in loaded)
    say("Distinct `D` values at `L = F` (loaded sets): " +
        ", ".join(f"{k}: {v}" for k, v in sorted(cnd.items())))
    say()
    # the most special optima
    worst = sorted(loaded, key=lambda r: r["L0"]["nmin"] / r["L0"]["total"])[:8]
    say("| gears | F | core | W_core | phase vectors | at the minimum | fraction | distinct D |")
    say("|---|---|---|---|---|---|---|---|")
    for r in worst:
        st = r["L0"]
        say(f"| {','.join(map(str, r['gears']))} | {r['F']} | {{{','.join(map(str, st['core']))}}} | "
            f"{r['Wcore']:,} | {st['total']:,} | {st['nmin']:,} | "
            f"{100 * st['nmin'] / st['total']:.3f}% | {st['ndist']} |")
    say()

    # ---------------------------------------------------------------- P14: the cost
    big = max(rows, key=lambda r: r["Wcore"])
    say("## P14: the cost is a scan of the core period")
    say()
    say(f"Largest core period in the family: `W_core = {big['Wcore']:,}` at "
        f"`{{{','.join(map(str, big['gears']))}}}` (`W = {big['W']:,}`, `F = {big['F']}`, core "
        f"`{{{','.join(map(str, big['L1']['core']))}}}` at `L = F + 1`).  Sum of `W_core` over the "
        f"family {sum(r['Wcore'] for r in rows):,} against sum of `W` "
        f"{sum(r['W'] for r in rows):,}; median `W / W_core` "
        f"{median([r['W'] / r['Wcore'] for r in rows]):.0f}.")
    say()
    summary["max_Wcore"] = big["Wcore"]

    # ---------------------------------------------------------------- P10: decoupling
    dec_fail = [r for r in loaded if r["L1"]["minDec"] <= r["L1"]["t"]]
    dec_gap = [r for r in loaded if r["L1"]["minDec"] < r["L1"]["minD"]]
    say("## P10: the parity classes do not minimise independently")
    say()
    say(f"At `L = F + 1`: `min_v D_e + min_v D_o < min_v (D_e + D_o)` on **{len(dec_gap)}** of "
        f"{len(loaded)} loaded sets; the decoupled criterion wrongly says coverable (so the "
        f"decoupled record formula over-estimates `F`) on **{len(dec_fail)}** sets.  Among them: "
        + ", ".join("{" + ",".join(map(str, r["gears"])) + "}" for r in dec_fail[:8]) + ".")
    say()
    summary["P10_decoupled_overestimates"] = len(dec_fail)
    summary["P10_decoupled_gap"] = len(dec_gap)

    # ---------------------------------------------------------------- P11: E2 run parity
    e2 = [r for r in loaded if r["L1"]["minCnt"] <= r["L1"]["t"]]
    e2gap = [r for r in loaded if r["L1"]["minCnt"] < r["L1"]["minD"]]
    say("## P11 (E2): the run structure matters at the optimum")
    say()
    say(f"At `L = F + 1`: `min_v [ceil(|U_e|/2) + ceil(|U_o|/2)] < min_v D` on **{len(e2gap)}** "
        f"loaded sets, and the counts-only criterion wrongly says coverable on **{len(e2)}**.  "
        f"Among them: " + ", ".join("{" + ",".join(map(str, r["gears"])) + "}" for r in e2[:8]) + ".")
    say()
    summary["P11_counts_overestimates"] = len(e2)

    # ---------------------------------------------------------------- P12: anchoring
    a0 = sum(1 for r in loaded if r["L0"]["anchor0"])
    aL = sum(1 for r in loaded if r["L0"]["anchorL"])
    say("## P12: anchoring at the window ends")
    say()
    say(f"At `L = F`, some minimiser has a core gear striking cell 0 on **{a0}** of {len(loaded)} "
        f"loaded sets ({len(loaded) - a0} exceptions); striking cell `L - 1` on **{aL}** "
        f"({len(loaded) - aL} exceptions).  Exceptions for cell 0: "
        + ", ".join("{" + ",".join(map(str, r["gears"])) + "}" for r in loaded if not r["L0"]["anchor0"])[:400] + ".")
    say()
    summary["P12_anchor0"] = a0
    summary["P12_anchorL"] = aL

    # ---------------------------------------------------------------- P15-17: the bound
    viol = sum(1 for r in rows if r["Lcap2"] < r["F"])
    order = sum(1 for r in rows if r["Lcap2"] > r["Lcap"])
    say("## P15-P17: the parity-refined capacity bound")
    say()
    say(f"`F_top <= Lcap2`: **{viol} violations** of {len(rows)}; `Lcap2 <= Lcap` (L70): "
        f"**{order} violations**.")
    say()
    slack2 = Counter(r["Lcap2"] - r["F"] for r in rows)
    slack1 = Counter(r["Lcap"] - r["F"] for r in rows)
    say("| slack | sets with `Lcap - F` = slack (L70) | sets with `Lcap2 - F` = slack (P15) |")
    say("|---|---|---|")
    for k in sorted(set(slack1) | set(slack2)):
        say(f"| {k} | {slack1.get(k, 0)} | {slack2.get(k, 0)} |")
    say()
    ex_free = sum(1 for r in free if r["Lcap2"] == r["F"])
    ex_loaded = sum(1 for r in loaded if r["Lcap2"] == r["F"])
    ex1_loaded = sum(1 for r in loaded if r["Lcap"] == r["F"])
    say(f"`Lcap2` exact on {ex_free} of {len(free)} free sets and on **{ex_loaded} of "
        f"{len(loaded)} loaded** sets (L70 exact on {ex1_loaded} loaded); maximum slack of "
        f"`Lcap2` {max(slack2)} against {max(slack1)} for L70.")
    say()
    summary["P15_violations"] = viol
    summary["P16_exact_free"] = ex_free
    summary["P16_exact_loaded"] = ex_loaded
    summary["P16_max_slack"] = max(slack2)
    for gs in ((7, 11, 13, 17), (13, 17, 19, 23, 29, 31, 37, 41), (7, 11, 13), (11, 13, 17, 19, 23)):
        L1, L2 = Lcap_bounds(list(gs), 60)
        F = F_rule(list(gs))
        say(f"`{{{','.join(map(str, gs))}}}`: `F = {F}`, `Lcap = {L1}`, `Lcap2 = {L2}`.")
    say()

    # the slack decomposition at L = F + 1
    loose = [r for r in loaded if r["L1"]["cap2"] <= r["L1"]["t"]]      # capacity says coverable
    by_overlap = [r for r in loose if r["L1"]["minCnt"] > r["L1"]["t"]]  # counts level closes it
    by_runs = [r for r in loose if r["L1"]["minCnt"] <= r["L1"]["t"]]   # only D closes it
    say(f"At `L = F + 1` the bound is loose (says coverable) on **{len(loose)}** loaded sets.  "
        f"Of these, the counts level (actual union sizes, so overlaps seen) already refuses "
        f"**{len(by_overlap)}** - the overlap kind; the remaining **{len(by_runs)}** are refused "
        f"only by the run structure `D` - the run-parity kind.")
    say("Overlap kind, examples: " + ", ".join("{" + ",".join(map(str, r["gears"])) + "}" for r in by_overlap[:6]))
    say("Run-parity kind, examples: " + ", ".join("{" + ",".join(map(str, r["gears"])) + "}" for r in by_runs[:6]))
    say()
    summary["P17_loose"] = len(loose)
    summary["P17_overlap"] = len(by_overlap)
    summary["P17_runs"] = len(by_runs)

    # ---------------------------------------------------------------- P9: half-turn identity
    say("## P9: the half-turn identity")
    say()
    small = [r for r in loaded if r["Wcore"] <= 20000][:200]
    badH = 0
    checked = 0
    for r in small:
        b, c = halfturn_check(r["L0"]["core"], r["F"])
        badH += b
        checked += c
    say(f"Odd-class mask against the even-class pattern of the adjacent-teeth core wheel read at "
        f"offset `H = (W_core + 1)/2`: **{badH} mismatches** over {checked:,} phase vectors of "
        f"{len(small)} loaded sets (every phase vector of each).")
    summary["P9_mismatches"] = badH
    summary["P9_vectors"] = checked

    json.dump(summary, open(os.path.join(RES, "phases.json"), "w"), indent=1)
    with open(os.path.join(RES, "phases.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()

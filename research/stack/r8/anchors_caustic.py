#!/usr/bin/env python3
"""Family C: caustic anchors (the run before a gear's first strike after a square).

Machine q (q prime): gears are the primes 5 <= h <= q (2 and 3 implicit, never
counted).  Window: q < n <= q*q.  A column is a pair (n, n+2) with n = 5 mod 6,
named by n.  Gear h is open at column n iff h does not divide n and h does not
divide n+2.

Anchor: a column a with a set K(a) of gears it is known open to by construction.
Certification: gear h is certified at column n by anchor a iff h in K(a) and
(n = a mod h or n = -a-2 mod h).  A column is fully certified iff every gear is
certified by some anchor.

Family C: for every ordered pair of distinct gears (g, h), c0 = g*g - 2; walk
c0 + 6i, i = 1, 2, ..., stop at the first i where h strikes (that i is e(g,h));
the anchors are c0 + 6i for 1 <= i < e(g,h), each with K = {h}.  Anchors are
deduplicated by column, merging K sets.

Base anchors (used only in the "C + base" figures): home a = -1 with K = all
gears; and a = g for every gear pair g, g+2 both gears, with K = all gears
except g and g+2.  Base anchors sharing a column with a family C anchor are
merged into it (same column, union of known-open gears).

Run: uv run python research/stack/r8/anchors_caustic.py
"""

import heapq
import time
from itertools import chain, compress, islice
from math import gcd
from operator import eq

from sympy import factorint, primerange

QS = [31, 101, 211, 401, 1009]
OUT = "research/stack/r8/results_anchors_caustic.md"


# ---------------------------------------------------------------- utilities
def sieve_upto(limit):
    sv = bytearray([1]) * (limit + 1)
    sv[0] = sv[1] = 0
    i = 2
    while i * i <= limit:
        if sv[i]:
            sv[i * i :: i] = bytearray(len(range(i * i, limit + 1, i)))
        i += 1
    return sv


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def first_strike(c0, h, inv6h):
    """smallest i >= 1 with h | c0+6i or h | c0+6i+2."""
    i1 = (-c0) % h * inv6h % h
    i2 = (-c0 - 2) % h * inv6h % h
    if i1 == 0:
        i1 = h
    if i2 == 0:
        i2 = h
    return i1 if i1 < i2 else i2


# ---------------------------------------------------------------- main work
def run(q):
    t0 = time.time()
    gears = list(primerange(5, q + 1))
    G = len(gears)
    gidx = {h: j for j, h in enumerate(gears)}
    bit = [1 << j for j in range(G)]
    full = (1 << G) - 1
    inv6 = {h: pow(6, -1, h) for h in gears}

    # ---- 1. build family C anchors (dedup by column, merge K) --------------
    anchors = {}          # column -> list of gear indices
    dirty = set()
    for g in gears:
        c0 = g * g - 2
        ev = []
        for h in gears:
            if h == g:
                continue
            ev.append((first_strike(c0, h, inv6[h]), gidx[h]))
        ev.sort(key=lambda t: -t[0])
        es = [t[0] for t in ev]      # descending
        arr = [t[1] for t in ev]
        maxe = es[0]
        cnt = len(es)                # arr[:cnt] = gears with e > i
        for i in range(1, maxe):
            while cnt > 0 and es[cnt - 1] <= i:
                cnt -= 1
            if cnt == 0:
                break
            col = c0 + 6 * i
            if col in anchors:
                anchors[col].extend(arr[:cnt])
                dirty.add(col)
            else:
                anchors[col] = arr[:cnt]
    for col in dirty:
        anchors[col] = sorted(set(anchors[col]))

    cols = sorted(anchors)
    n_anchors = len(cols)

    # ---- verification + family-C coverage ---------------------------------
    covC = [bytearray(h) for h in gears]
    fails = 0
    first_fail = None
    ksum = 0
    for col in cols:
        K = anchors[col]
        ksum += len(K)
        for j in K:
            h = gears[j]
            if col % h == 0 or (col + 2) % h == 0:
                fails += 1
                if first_fail is None:
                    first_fail = (col, h)
                continue
            r = col % h
            cv = covC[j]
            cv[r] = 1
            cv[(h - 2 - r) % h] = 1
    if fails:
        return {"q": q, "fatal": f"{fails} verification failures, first {first_fail}"}

    # ---- 2. coverage per gear --------------------------------------------
    covered_full = []
    first_uncov = None
    for j, h in enumerate(gears):
        # residues 0 and h-2 are never marked (anchors are open to h), so the
        # sum is exactly the number of covered residues among the h-2 valid ones
        c = sum(covC[j])
        if c == h - 2:
            covered_full.append(h)
        elif first_uncov is None:
            first_uncov = (h, c, h - 2)

    # ---- base anchors, merged in by column --------------------------------
    all_gears_K = list(range(G))
    base = {-1: all_gears_K}
    gear_pairs = [g for g in gears if (g + 2) in gidx]
    for g in gear_pairs:
        ex = {gidx[g], gidx[g + 2]}
        base[g] = [j for j in all_gears_K if j not in ex]
    merged = dict(anchors)
    for col, K in base.items():
        if col in merged:
            merged[col] = sorted(set(merged[col]) | set(K))
        else:
            merged[col] = list(K)

    mcols = sorted(merged)
    aidx = {c: i for i, c in enumerate(mcols)}
    M = len(mcols)

    # ---- coverage C+base, buckets (residue -> anchor ids), masks ----------
    covCB = [bytearray(h) for h in gears]
    buckets = [[[] for _ in range(h)] for h in gears]
    masks = [0] * M
    for aid in range(M):
        col = mcols[aid]
        m = 0
        for j in merged[col]:
            h = gears[j]
            r = col % h
            cv = covCB[j]
            cv[r] = 1
            cv[(h - 2 - r) % h] = 1
            buckets[j][r].append(aid)
            m |= bit[j]
        masks[aid] = m

    # ---- window scan ------------------------------------------------------
    maxcol = mcols[-1]
    sv = sieve_upto(max(q * q, maxcol + 2) + 2)
    n0 = q + 1
    while n0 % 6 != 5:
        n0 += 1
    hi = q * q - 2

    gz = list(enumerate(gears))
    twins = 0
    certC = 0
    certCB = 0
    struck_cert = 0
    cert_twins = []
    for n in range(n0, hi + 1, 6):
        ok = True
        for j, h in gz:
            if not covCB[j][n % h]:
                ok = False
                break
        istwin = sv[n] and sv[n + 2]
        if istwin:
            twins += 1
            if ok:
                certCB += 1
                cert_twins.append(n)
            okc = True
            for j, h in gz:
                if not covC[j][n % h]:
                    okc = False
                    break
            if okc:
                certC += 1
        elif ok:
            struck_cert += 1

    # ---- 5. walk length (greedy set cover over C + base anchors) ----------
    P = 1
    for h in gears:
        P *= h
    rmask = {1: 0, P: full}
    for j, h in enumerate(gears):
        rmask[h] = bit[j]

    def dmask(d):
        """bitmask of gears dividing d (d = 0 -> every gear)."""
        r = gcd(d, P)
        m = rmask.get(r)
        if m is None:
            m = 0
            for p in factorint(r):
                m |= bit[gidx[p]]
            rmask[r] = m
        return m

    rng = range(G)
    lens = []
    for n in cert_twins:
        chunks = []
        for j in rng:
            h = gears[j]
            b = buckets[j]
            r = n % h
            c = b[r]
            if c:
                chunks.append(c)
            t = h - 2 - r
            if t < 0:
                t += h
            if t != r:
                c = b[t]
                if c:
                    chunks.append(c)
        big = list(chain.from_iterable(chunks))
        big.sort()
        dups = set(compress(big, map(eq, big, islice(big, 1, None))))
        unc = full
        steps = 0
        if dups:
            heap = []
            for aid in dups:
                a = mcols[aid]
                m = masks[aid] & (dmask(abs(n - a)) | dmask(n + a + 2))
                c = m.bit_count()
                if c >= 2:
                    heap.append((-c, aid, m))
            heapq.heapify(heap)
            while heap:
                negc, aid, m = heap[0]
                c = (m & unc).bit_count()
                if c < -negc:
                    if c >= 2:
                        heapq.heapreplace(heap, (-c, aid, m))
                    else:
                        heapq.heappop(heap)
                    continue
                if c < 2:
                    break
                heapq.heappop(heap)
                unc &= ~m
                steps += 1
        steps += unc.bit_count()
        lens.append(steps)

    # ---- 6. anchors that are themselves twins, mean |K| -------------------
    a_twin = 0
    for col in cols:
        if col >= 0 and sv[col] and sv[col + 2]:
            a_twin += 1
    mean_k = ksum / n_anchors if n_anchors else 0.0

    return {
        "q": q,
        "gears": G,
        "anchors": n_anchors,
        "fails": fails,
        "cov_full": len(covered_full),
        "first_uncov": first_uncov,
        "twins": twins,
        "certC": certC,
        "certCB": certCB,
        "struck_cert": struck_cert,
        "wmin": min(lens) if lens else None,
        "wmed": median(lens),
        "wmax": max(lens) if lens else None,
        "a_twin": a_twin,
        "a_nontwin": n_anchors - a_twin,
        "mean_k": mean_k,
        "secs": time.time() - t0,
    }


# ---------------------------------------------------------------- side check
def side_check(q):
    """pairs (g, h) with h < g: g = m*h + r; report e(g,h) and the smallest
    s >= 1 with s*h - r*r > 0 and (s*h - r*r) = 0 mod 6.  No interpretation."""
    gears = list(primerange(5, q + 1))
    inv6 = {h: pow(6, -1, h) for h in gears}
    rows = []
    for g in gears:
        c0 = g * g - 2
        for h in gears:
            if h >= g:
                continue
            r = g % h
            e = first_strike(c0, h, inv6[h])
            s = None
            for cand in range(1, 6 * h + 7):
                v = cand * h - r * r
                if v > 0 and v % 6 == 0:
                    s = cand
                    break
            rows.append((g, h, r, e, s))
    return rows


# ---------------------------------------------------------------- reporting
HEAD = (
    "| q | anchors | verification failures | gears fully covered | "
    "first uncovered gear | twins certified (C alone) | twins certified (C + base) | "
    "total window twins | struck columns certified | walk length min/median/max | "
    "anchors twins / not twins | mean |K(a)| |"
)
SEP = "|" + "---|" * 12


def fmt_row(res):
    fu = res["first_uncov"]
    fu_s = "none" if fu is None else f"{fu[0]} ({fu[1]}/{fu[2]})"
    med = res["wmed"]
    med_s = "-" if med is None else (f"{med:.1f}" if med % 1 else f"{int(med)}")
    walk = (
        "-"
        if res["wmin"] is None
        else f"{res['wmin']} / {med_s} / {res['wmax']}"
    )
    return (
        f"| {res['q']} | {res['anchors']} | {res['fails']} | "
        f"{res['cov_full']}/{res['gears']} | {fu_s} | {res['certC']} | "
        f"{res['certCB']} | {res['twins']} | {res['struck_cert']} | {walk} | "
        f"{res['a_twin']} / {res['a_nontwin']} | {res['mean_k']:.2f} |"
    )


def main():
    rows = []
    for q in QS:
        res = run(q)
        if "fatal" in res:
            print(f"q = {q}: FATAL {res['fatal']}")
            return
        rows.append(res)
        print(fmt_row(res), f"   [{res['secs']:.1f}s]", flush=True)

    sc = side_check(101)
    first10 = sc[:10]

    lines = [
        "# Family C (caustic) anchors",
        "",
        "Anchors: for each ordered pair of distinct gears (g, h), the columns",
        "g*g - 2 + 6i for 1 <= i < e(g,h), where e(g,h) is the first i >= 1 at which",
        "h strikes; K = {h}; deduplicated by column with K sets merged.  Base anchors",
        "(home a = -1, K = all gears; gear pairs a = g with g, g+2 gears, K = all gears",
        "but g and g+2) are merged in by column for the 'C + base' figures.",
        "",
        HEAD,
        SEP,
    ]
    lines += [fmt_row(r) for r in rows]
    lines += [
        "",
        "## Side check (q = 101 only)",
        "",
        f"pairs (g, h) with h < g: {len(sc)}",
        "",
        "first 10 pairs, in order of g ascending then h ascending, as (g, h, r, e(g,h)):",
        "",
        "  " + ", ".join(f"({g}, {h}, {r}, {e})" for g, h, r, e, s in first10),
        "",
        "the smallest s >= 1 with s*h - r*r > 0 and (s*h - r*r) = 0 mod 6, same ten pairs:",
        "",
        "  " + ", ".join(
            ("no formula check" if s is None else f"(g={g}, h={h}) s = {s}")
            for g, h, r, e, s in first10
        ),
        "",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-10:]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

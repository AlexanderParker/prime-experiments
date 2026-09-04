"""W.t - the same representations across levels of the chain (T11).

Chain: g_0 = q; g_{n+1} = 6 k - 1 where k is the landing of the walk from g_n^2 under
{5..g_n} (the first opening at or above the column of g_n^2 - a twin pair).  Iterated while
g_n stays inside the prime table.

At every level the path is built in the OFFSET coordinate: gear p strikes offset i iff
i = (2 - g^2) 6^-1 or -g^2 6^-1 (mod p).  That gives the walk, the depth string, the
smallest-striker word and the hop offsets in one pass over the gears.

Reported per level: L, d, the start slot k_0 mod 5, the class g mod 30, the square gate,
the number of hop layers, the max depth, the word's first letters; and across levels: whether
any of these predicts the next level's value.

Writes results/pt_levels.txt.
Usage: uv run python research/anchor235/r38/pt_levels.py [--nchains 20] [--pmax 2000000]
"""
import argparse
import os
from collections import Counter

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def sieve_flags(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def level(g, P, INV6, I=2048):
    """path from g^2 under {5..g}, in the offset coordinate."""
    ng = int(np.searchsorted(P, g, side="right"))
    pp = P[:ng]
    iv = INV6[:ng]
    r = (g % pp) ** 2 % pp
    i_lo = ((2 - r) * iv) % pp
    i_hi = ((-r) * iv) % pp
    dep = np.zeros(I, dtype=np.int32)
    ms = np.zeros(I, dtype=np.int64)
    cand = np.flatnonzero((i_lo < I) | (i_hi < I))
    # descending so the smallest striker is written last
    for j in cand[::-1]:
        p = int(pp[j])
        for st in (int(i_lo[j]), int(i_hi[j])):
            if st < I:
                dep[st::p] += 1
                ms[st::p] = p
    L = int(np.argmax(dep == 0))
    return dict(g=int(g), L=L, dep=dep[:L + 1].copy(), word=ms[:L].copy(),
                d=int((2 * pow(6, -1, int(g))) % int(g)),
                k0=(g * g - 1) // 6, slot=((g * g - 1) // 6) % 5, g30=int(g) % 30,
                nlayers=len(set(int(v) for v in ms[:L])),
                depmax=int(dep[:L].max()) if L else 0,
                gate=bool(((int(g) * int(g) - 2) % pp != 0).all()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nchains", type=int, default=20)
    ap.add_argument("--pmax", type=int, default=2000000)
    a = ap.parse_args()
    log = open(os.path.join(OUT, "pt_levels.txt"), "w")

    def say(*xs):
        s = " ".join(str(x) for x in xs)
        print(s)
        log.write(s + "\n")

    fl = sieve_flags(a.pmax)
    P = np.flatnonzero(fl).astype(np.int64)
    P = P[P >= 5]
    say("prime table to %d: %d gears" % (a.pmax, P.size))
    INV6 = np.array([pow(6, -1, int(p)) for p in P], dtype=np.int64)
    PSET = set(int(x) for x in P)

    starts = [int(x) for x in P[:a.nchains]]
    chains = []
    for q in starts:
        ch = []
        g = q
        while g <= a.pmax // 2 and g in PSET:
            r = level(g, P, INV6)
            ch.append(r)
            k = r["k0"] + r["L"]
            gnext = 6 * k - 1
            if gnext > a.pmax:
                break
            g = gnext
        chains.append(ch)

    say("")
    say("=== the chains ===")
    say("start   levels   (g, L, d, slot, g mod 30, gate, hop layers, max depth)")
    for ch in chains:
        say(" %-6d %d  %s" % (ch[0]["g"], len(ch),
                              "  ".join("(%d, L=%d, d=%d, slot=%d, %d mod 30, gate=%s, lay=%d,"
                                        " dmax=%d)"
                                        % (r["g"], r["L"], r["d"], r["slot"], r["g30"],
                                           "Y" if r["gate"] else "N", r["nlayers"], r["depmax"])
                                        for r in ch)))

    pairs = [(ch[i], ch[i + 1]) for ch in chains for i in range(len(ch) - 1)]
    say("")
    say("=== across levels: does level n predict level n+1? (%d consecutive pairs) ===" % len(pairs))
    if pairs:
        Ln = np.array([x["L"] for x, _ in pairs], dtype=float)
        Ln1 = np.array([y["L"] for _, y in pairs], dtype=float)
        say("L_n vs L_{n+1}: correlation %.4f; L increases at %d of %d"
            % (float(np.corrcoef(Ln, Ln1)[0, 1]), int((Ln1 > Ln).sum()), len(pairs)))
        say("slot at level n -> slot at level n+1: %s"
            % dict(Counter((x["slot"], y["slot"]) for x, y in pairs)))
        say("   (the slot at level n+1 is decided by g_{n+1} mod 30 alone, T1)")
        say("class g mod 30 at n -> at n+1: %d distinct pairs from %d source classes"
            % (len({(x["g30"], y["g30"]) for x, y in pairs}),
               len({x["g30"] for x, y in pairs})))
        cnt = Counter((x["g30"], y["g30"]) for x, y in pairs)
        srcs = Counter(x["g30"] for x, y in pairs)
        det = sum(1 for s in srcs if len({t for (u, t) in cnt if u == s}) == 1)
        say("   source classes with a single successor: %d of %d (a function would need all)"
            % (det, len(srcs)))
        say("square gate at n -> at n+1: %s"
            % dict(Counter((x["gate"], y["gate"]) for x, y in pairs)))
        say("hop layers n vs n+1: correlation %.4f"
            % float(np.corrcoef([x["nlayers"] for x, _ in pairs],
                                [y["nlayers"] for _, y in pairs])[0, 1]))
        say("max depth n vs n+1: correlation %.4f"
            % float(np.corrcoef([x["depmax"] for x, _ in pairs],
                                [y["depmax"] for _, y in pairs])[0, 1]))
        wsame = sum(1 for x, y in pairs
                    if list(x["word"][:3]) == list(y["word"][:3]))
        say("first three letters of the word equal at consecutive levels: %d of %d"
            % (wsame, len(pairs)))

    say("")
    say("=== what DOES repeat: the frame ===")
    allr = [r for ch in chains for r in ch]
    say("levels computed: %d" % len(allr))
    say("levels with L < d (top gear strikes its own path once): %d of %d; exceptions %s"
        % (sum(1 for r in allr if r["L"] < r["d"]), len(allr),
           [(r["g"], r["L"], r["d"]) for r in allr if r["L"] >= r["d"]]))
    say("levels starting on slot 11|13 (k_0 = 2 mod 5): %d of %d"
        % (sum(1 for r in allr if r["slot"] == 2), len(allr)))
    say("levels whose offset 1 is blocked by gear 5: %d of %d"
        % (sum(1 for r in allr if (r["slot"] + 1) % 5 in (1, 4)), len(allr)))
    say("levels whose first column has depth 1 exactly when the square gate is open: %d of %d"
        % (sum(1 for r in allr if (int(r["dep"][0]) == 1) == r["gate"]), len(allr)))
    say("smallest-striker histogram over all levels' paths (top 8): %s"
        % Counter(int(v) for r in allr for v in r["word"]).most_common(8))
    tot = sum(len(r["word"]) for r in allr)
    n5 = sum(1 for r in allr for v in r["word"] if v == 5)
    say("gear-5 share of path columns across all levels: %d of %d = %.4f" % (n5, tot, n5 / tot))
    say("mean depth along the path by normalised position, pooled over levels:")
    prof = np.zeros(11)
    pn = np.zeros(11)
    for r in allr:
        L = r["L"]
        if L >= 3:
            b = (np.arange(L) * 10 // (L - 1))
            np.add.at(prof, b, r["dep"][:L])
            np.add.at(pn, b, 1)
    say("   " + " ".join("%.2f" % v for v in prof / np.maximum(pn, 1)))
    log.close()


if __name__ == "__main__":
    main()

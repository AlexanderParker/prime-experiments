"""Branch 5d.ii.i (prover, round 48), items 4 and 5: the window.

For every prime rung q with 7 <= q <= 997:
  window columns lo = q//6 + 1 .. W = (q'^2 - 1)/6 (q' the next prime);
  openings there are the twin-prime columns; F_W(q) = the longest blocked stretch
  with both ends inside the window.
For each DISTINCT window record stretch, at the first rung that holds it:
  hold  = number of gears of M = {5..q} that strike a column of it,
  cov   = its exact MINIMUM COVER among those gears (branch and bound),
  f     = the umbrella's forced count #{g <= q : g - a_g < F_W + 1},
  h_free= the free minimum, the least K such that SOME K gears at SOME phases block
          F_W - 1 consecutive columns; read off the certified F ladder, which is the
          inverse of h (initial segments optimal, gate in lattice.py).
Also, per rung, the gear-count inequality of item 5: h_free(W(q)) against n(M).

Usage: uv run python research/anchor235/r48/win.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cover_core import arcs  # noqa: E402

OUT = os.path.join(HERE, "results")

# the certified F ladder (max-gap convention), machine -> F, from
# research/proof/anchor_cycles.md line 28 and docs/proofs/08: F = 5 .. 161.
LADDER = [(5, 2), (7, 5), (11, 7), (13, 11), (17, 18), (19, 25), (23, 34),
          (29, 43), (31, 58), (37, 88), (41, 91), (43, 103), (47, 118),
          (53, 145), (59, 161)]
# recorded lower bounds past the certified ladder (research/proof/cov_spectrum.md)
LADDER_LB = [(61, 171), (67, 175), (71, 185)]


def sieve(n):
    b = bytearray([1]) * (n + 1)
    b[0] = b[1] = 0
    i = 2
    while i * i <= n:
        if b[i]:
            b[i * i::i] = bytearray(len(b[i * i::i]))
        i += 1
    return b


def h_free(S, gears_upto):
    """Least K with F({5..p_K}) >= S, from the ladder; '>=K' past its end."""
    for i, (y, f) in enumerate(LADDER):
        if f >= S:
            return i + 1, True
    for y, f in LADDER_LB:
        if f >= S:
            return len([1 for yy, _ in LADDER if yy <= y]) + \
                   len([1 for yy, _ in LADDER_LB if yy <= y]), True
    return len(LADDER) + len(LADDER_LB) + 1, False   # a lower bound only


def min_cover(cols, gears):
    """Exact minimum number of gears covering every column in cols."""
    idx = {c: i for i, c in enumerate(cols)}
    L = len(cols)
    full = (1 << L) - 1
    masks = []
    for g in gears:
        u = pow(6, -1, g)
        m = 0
        for c in cols:
            if c % g == u % g or c % g == (-u) % g:
                m |= 1 << idx[c]
        if m:
            masks.append(m)
    masks.sort(key=lambda m: -bin(m).count("1"))
    if not masks:
        return None, 0
    # per column, the masks that cover it
    bycol = [[m for m in masks if (m >> i) & 1] for i in range(L)]
    best = [len(masks) + 1]
    maxpop = max(bin(m).count("1") for m in masks)

    def rec(covered, used):
        if covered == full:
            best[0] = min(best[0], used)
            return
        todo = L - bin(covered).count("1")
        if used + -(-todo // maxpop) >= best[0]:
            return
        u = ~covered & full
        i = (u & -u).bit_length() - 1
        # branch on the least-covered uncovered column for a tighter tree
        cnt = {}
        uu = u
        while uu:
            b = uu & -uu
            j = b.bit_length() - 1
            uu ^= b
            cnt[j] = len(bycol[j])
        i = min(cnt, key=lambda j: cnt[j])
        for m in bycol[i]:
            if m & ~covered:
                rec(covered | m, used + 1)

    rec(0, 0)
    return best[0], len(masks)


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "window.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    NMAX = 1009 * 1009 + 10
    pr = sieve(NMAX)
    primes = [i for i in range(2, 1010) if pr[i]]
    TOP = (1009 * 1009 - 1) // 6
    twin = bytearray(TOP + 2)
    for k in range(1, TOP + 1):
        if pr[6 * k - 1] and pr[6 * k + 1]:
            twin[k] = 1
    opens = [k for k in range(1, TOP + 1) if twin[k]]

    rungs = [q for q in primes if 7 <= q <= 997]
    say("rung   q'      W       n(M)  F_W   x        f(F_W)  h_free(F_W)  "
        "h_free(W)  n(M)   ratio")
    seen = {}
    rows = []
    import bisect
    for q in rungs:
        qn = primes[primes.index(q) + 1]
        lo, W = q // 6 + 1, (qn * qn - 1) // 6
        i0 = bisect.bisect_left(opens, lo)
        i1 = bisect.bisect_right(opens, W)
        seg = opens[i0:i1]
        FW, x = 0, None
        for a, b in zip(seg, seg[1:]):
            if b - a > FW:
                FW, x = b - a, a
        n = sum(1 for g in primes if 5 <= g <= q)
        f = sum(1 for g in primes if 5 <= g <= q and arcs(g)[2] < FW + 1)
        hF, exactF = h_free(FW, q)
        hW, exactW = h_free(W, q)
        rows.append((q, qn, W, n, FW, x, f, hF, exactF, hW, exactW))
        say(f"{q:5d} {qn:5d} {W:8d}  {n:4d}  {FW:4d} {x:8d}  {f:5d}   "
            f"{'' if exactF else '>='}{hF:3d}        "
            f"{'' if exactW else '>='}{hW:3d}       {n:4d}   {hW/n:5.2f}")
        seen.setdefault((x, FW), q)

    say("")
    say("DISTINCT window record stretches, with their exact minimum cover")
    say("  x        F_W   first q   gears  hold   cov   f   h_free   cov/h_free  gate z")
    for (x, FW), q in sorted(seen.items(), key=lambda t: t[0][1]):
        gears = [g for g in primes if 5 <= g <= q]
        cols = list(range(x + 1, x + FW))
        cov, hold = min_cover(cols, gears)
        f = sum(1 for g in gears if arcs(g)[2] < FW + 1)
        hF, exactF = h_free(FW, q)
        z = int(((6 * (x + FW) + 1) ** 0.5))
        say(f"  {x:8d} {FW:4d}  {q:6d}  {len(gears):5d}  {hold:4d}  {cov:4d} "
            f"{f:4d}   {'' if exactF else '>='}{hF:3d}      "
            f"{cov/hF:5.2f}      {z:6d}")
    log.close()


if __name__ == "__main__":
    main()

"""E4's number: S_t(q) = the longest stretch of columns of machine q whose interior consecutive
hole gaps are all >= t (a stretch runs from just after a hole to just before a hole; its interior
holes are spaced >= t apart; t = 1 gives a whole period, t > max gap gives the record F(q) + 2
counting the bounding holes... we report the interior span = last interior hole - first interior
hole + flanks, the same quantity as G_k with the gap condition).

E4 (draft, final form) says: S_t(q) < q'^2/6 for q' the next gear and t = (q'-1)/3 (interior gaps of
a covered window are at least this). Pre-registered (tree node R5.f.xxxiv.g): S_t(q) falls at least
like F(q) x (mean gap)/t in t, and at t = ceil((q'-1)/3) it is below a third of q'^2/6 for every
q <= 19; the ratio S_t/(q'^2/6) falls with q.
"""
from math import prod, ceil

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})
def paint(gears, P):
    painted = bytearray(P)
    for g in gears:
        for t in teeth(g):
            painted[t::g] = b"\x01" * len(range(t, P, g))
    return painted

def S_t(gaps, t):
    """longest span over cyclic chains of consecutive gaps all >= t: span = sum of the chain's gaps
    plus the two flanking runs (the gaps before and after the chain, minus 1 each) - i.e. the window
    from just after the hole before the chain to just before the hole after it. A chain of zero gaps
    (a single hole) counts: run-hole-run."""
    H = len(gaps); best = 0
    g2 = gaps + gaps
    i = 0
    while i < H:
        if g2[i] >= t:
            j = i
            while j < i + H and g2[j] >= t: j += 1
            span = sum(g2[i:j])            # from hole i to hole j (j - i gaps)
            fb = g2[i - 1] - 1; fa = g2[j % H] - 1
            best = max(best, fb + span + 1 + fa)
            i = j
        else:
            # single hole with both neighbours gaps < t: run-hole-run
            best = max(best, g2[i - 1] - 1 + 1 + g2[i] - 1)
            i += 1
    return best

primes = [5, 7, 11, 13, 17, 19, 23, 29]
for k in range(1, 8):
    gears = primes[:k]; q2 = primes[k]; P = prod(gears)
    old = paint(gears, P)
    holes = [i for i in range(P) if not old[i]]; H = len(holes)
    gaps = [(holes[(i + 1) % H] - holes[i]) % P for i in range(H)]
    F = max(gaps) - 1
    t_star = ceil((q2 - 1) / 3)
    win = (q2 * q2 - 1) // 6 - (q2 + 7) // 6 + 1
    row = [(t, S_t(gaps, t)) for t in range(2, max(gaps) + 2)]
    print(f"machine 5..{gears[-1]}: F = {F}, max gap {max(gaps)}; next gear {q2}: t* = {t_star}, "
          f"S_t*(q) = {S_t(gaps, t_star)}, window {win}, ratio {S_t(gaps, t_star)/win:.3f}")
    print("   t -> S_t:", row)

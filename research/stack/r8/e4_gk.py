"""G_k(q): the longest span of k consecutive holes of machine q together with the two flanking
runs (k = 0 is the record F(q) itself; k = 1 the largest run-hole-run). Exact over one period.

Alignability (CRT over the joint period with a new gear q'): a chain of k consecutive holes with
gaps d_1..d_{k-1} can be placed with all k holes in q''s two tooth classes iff every partial sum
d_1 + ... + d_j is congruent to 0 or +inv3 modulo q', or every one to 0 or -inv3 (the chain starts in one
class and each step stays or crosses to the other; mixing the two signs is not realisable) (inv3 = the tooth distance). For k = 1
there is no condition: every single hole is alignable, so F(q') >= G_1(q) for every q' > q.
Pre-registered (tree node R5.f.xxxiv.e): (1) F(q') >= G_1(q) at every step, with equality at some
steps (predicted 13 and 19 from the e3 data); (2) F(q') = max over k of the largest alignable
chain span, and the alignability criterion reproduces F(q') for q' = 11..23; (3) G_k(q) grows
roughly linearly in k with slope about the mean hole gap plus mean run, far below (k+1)(F(q)+1).
"""
from math import prod

def inv(a, m): return pow(a, -1, m)
def teeth(g):
    c = inv(6, g); return sorted({c % g, (-c) % g})
def paint(gears, P):
    painted = bytearray(P)
    for g in gears:
        for t in teeth(g):
            painted[t::g] = b"\x01" * len(range(t, P, g))
    return painted

primes = [5, 7, 11, 13, 17, 19, 23]
for k in range(1, len(primes)):
    gears = primes[:k]; q2 = primes[k]; P = prod(gears)
    old = paint(gears, P)
    holes = [i for i in range(P) if not old[i]]; H = len(holes)
    gaps = [(holes[(i + 1) % H] - holes[i]) % P for i in range(H)]
    F = max(gaps) - 1
    # G_k: for chains of k consecutive holes starting at hole i: span = holes[i+k-1]-holes[i] + flanks
    G = {}
    for kk in range(1, 7):
        best = 0
        for i in range(H):
            span = sum(gaps[(i + j) % H] for j in range(kk - 1))  # distance from hole i to hole i+kk-1
            fb = gaps[(i - 1) % H] - 1; fa = gaps[(i + kk - 1) % H] - 1
            best = max(best, fb + span + 1 + fa)
        G[kk] = best
    # alignable chains for q2: partial sums in {0, +-inv3} mod q2
    d = inv(3, q2); okA = {0, d % q2}; okB = {0, (-d) % q2}
    Fq2 = F
    for kk in range(1, 7):
        for i in range(H):
            s_ = 0; goodA = goodB = True
            for j in range(kk - 1):
                s_ += gaps[(i + j) % H]
                if s_ % q2 not in okA: goodA = False
                if s_ % q2 not in okB: goodB = False
                if not (goodA or goodB): break
            if not (goodA or goodB): continue
            span = sum(gaps[(i + j) % H] for j in range(kk - 1))
            fb = gaps[(i - 1) % H] - 1; fa = gaps[(i + kk - 1) % H] - 1
            Fq2 = max(Fq2, fb + span + 1 + fa)
    # direct
    new = paint(gears + [q2], P * q2)
    z = new.find(b"\x00"); rot = new[z:] + new[:z]; run = mx = 0
    for x in rot:
        if x: run += 1; mx = max(mx, run)
        else: run = 0
    print(f"machine 5..{gears[-1]}: F = {F}, G_1..G_6 = {[G[j] for j in range(1, 7)]}; next gear {q2}: "
          f"predicted F({q2}) by alignable chains = {Fq2}, direct = {mx}, G_1 <= F({q2}): {G[1] <= mx}")

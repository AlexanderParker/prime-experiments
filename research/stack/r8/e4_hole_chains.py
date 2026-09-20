"""E4 as hole spacing: the record of machine q' equals the longest chain of consecutive holes of
machine q all lying in the two tooth classes of q' (so q' can fill every one of them), plus the
flanking runs. Exact period scans for 5..p, p <= 19 (q' <= 23).

Pre-registered (tree node R5.f.xxxiv.c): (1) the chain formula reproduces F(q') exactly for
q' = 7..23; (2) over a full period exactly 2/q' of the holes lie in q''s tooth classes (CRT) and the
same share of consecutive-hole PAIRS are both in the classes only to first order - the aligned
pairs are the rare joins; (3) the record chain uses k holes with k <= 2 ceil(F/q').
"""
from math import prod
from collections import Counter

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
    holes0 = [i for i in range(P) if not old[i]]
    # the joint period with the new gear: every hole class recurs q' times with distinct residues mod q'
    holes = [h + m * P for m in range(q2) for h in holes0]
    P = P * q2
    H = len(holes)
    # hole gaps (cyclic)
    gaps = [(holes[(i + 1) % H] - holes[i]) % P for i in range(H)]
    hist = Counter(gaps)
    T = set(teeth(q2))
    inT = [ (h % q2) in T for h in holes ]
    share = sum(inT) / H
    # consecutive pairs both aligned
    pairs_aligned = sum(1 for i in range(H) if inT[i] and inT[(i + 1) % H])
    # longest cyclic chain of consecutive aligned holes, and the record it yields:
    # span from the hole before the chain (exclusive) to the hole after (exclusive)
    best = 0; bestk = 0
    # iterate over chain starts
    i = 0; n = H
    # unroll twice for cyclic chains
    inT2 = inT + inT
    idx = 0
    while idx < n:
        if inT2[idx]:
            j = idx
            while j < idx + n and inT2[j]: j += 1
            kk = j - idx  # chain length in holes
            # run = from hole[idx-1]+1 to hole[j]-1 (cyclic) : length = (holes[j] - holes[idx-1]) - 1
            a = holes[(idx - 1) % n]; b = holes[j % n]
            span = (b - a) % P - 1
            if span > best: best, bestk = span, kk
            idx = j
        else: idx += 1
    # also the record with zero joins (an old run alone) = max gap - 1
    F_old = max(gaps) - 1
    Fq2 = max(best, F_old)
    # direct check
    new = paint(gears + [q2], P)
    # longest run in new pattern
    z = new.find(b"\x00"); rot = new[z:] + new[:z]; run = mx = 0
    for x in rot:
        if x: run += 1; mx = max(mx, run)
        else: run = 0
    print(f"machine 5..{gears[-1]} -> +{q2}: holes {H}, aligned share {share:.4f} (2/q' = {2/q2:.4f}), "
          f"aligned consecutive pairs {pairs_aligned} ({pairs_aligned/H:.4f} of holes), "
          f"longest aligned chain {bestk} holes -> record {Fq2}; direct scan F({q2}) = {mx}; "
          f"hole-gap histogram (gap: count) {sorted(hist.items())[:8]} ... max gap {max(gaps)}")

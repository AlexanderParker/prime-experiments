"""E4a / E4b: flanks and chains of aligned hole pairs, exact over the joint period, machines 5..q
with next gear q' (q' <= 23).

A consecutive hole pair (h_i, h_{i+1}) of machine q is ALIGNED for q' when both holes lie in the two
tooth classes of q'. Its flanks are the painted runs just before h_i and just after h_{i+1}. A chain
is a maximal run of consecutive aligned holes.
Pre-registered (tree node R5.f.xxxiv.d): (E4a) flank lengths beside aligned pairs have the same
distribution as beside all consecutive pairs (means within 5%, maxima equal or within 1); (E4b) the
chain-length histogram falls geometrically with ratio about the share of gaps that hit the two
aligned values; the record chain's flanks are the largest flanks among aligned pairs.
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
for k in range(2, len(primes)):
    gears = primes[:k]; q2 = primes[k]; P0 = prod(gears)
    old = paint(gears, P0)
    holes0 = [i for i in range(P0) if not old[i]]
    holes = [h + m * P0 for m in range(q2) for h in holes0]
    P = P0 * q2; H = len(holes)
    gaps = [(holes[(i + 1) % H] - holes[i]) % P for i in range(H)]
    # flank before hole i = gap[i-1] - 1 ; flank after hole i+1 = gap[i+1] - 1
    T = set(teeth(q2)); inT = [(h % q2) in T for h in holes]
    all_fl = []; al_fl = []; al_gaps = Counter()
    for i in range(H):
        fb = gaps[(i - 1) % H] - 1; fa = gaps[(i + 1) % H] - 1
        all_fl.append(max(fb, fa))
        if inT[i] and inT[(i + 1) % H]:
            al_fl.append(max(fb, fa)); al_gaps[gaps[i]] += 1
    # chains
    chains = Counter(); idx = 0; inT2 = inT + inT
    while idx < H:
        if inT2[idx]:
            j = idx
            while j < idx + H and inT2[j]: j += 1
            chains[j - idx] += 1; idx = j
        else: idx += 1
    Fq = max(gaps) - 1
    mean_all = sum(all_fl) / len(all_fl); mean_al = sum(al_fl) / len(al_fl) if al_fl else float('nan')
    print(f"5..{gears[-1]} -> +{q2}: F(q) = {Fq}; inv3 = {inv(3, q2)}, q'-inv3 = {q2 - inv(3, q2)}; "
          f"aligned pairs {len(al_fl)} by gap {sorted(al_gaps.items())}; "
          f"max flank: all {max(all_fl)}, aligned {max(al_fl) if al_fl else '-'}; "
          f"mean flank: all {mean_all:.3f}, aligned {mean_al:.3f}; chain lengths {sorted(chains.items())}")

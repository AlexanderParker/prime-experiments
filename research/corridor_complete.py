"""Round 17 lateral: (1) is the corridor COMPLETE at mod 35? (2) when does j=1
double padding first become affordable? (3) is p <= 2 provable?

COMPLETENESS LEMMA. A shape with n openings is blocked by gear q only if no
residue r survives. Gear q has 2 teeth, so it forbids at most 2n values of r
(one per tooth per offset). If 2n < q some r always survives, so gear q CANNOT
block. Constraints from distinct gears are independent by CRT, so the joint
feasible set is the product of the per-gear sets: a shape is corridor-feasible
iff it is feasible gear by gear. Hence

    only gears q <= 2n can ever block an n-opening shape.

n = 4 or 5 -> only gears 5, 7 -> the mod-35 test IS the whole corridor.
n = 6 -> gear 11 enters (mod 385); n = 7 -> gear 13 (mod 5005).

GENERALISED AP LEMMA. Openings at offsets i*q' for four values of i that are
DISTINCT mod 5 are impossible (their residues mod 5 are 4 distinct values, and
only 3 are exposed). The round-16 AP lemma is the case i = 0,1,2,3.
"""
E5 = {0, 2, 3}
E7 = {0, 2, 3, 4, 5}

def teeth(q):
    u = pow(6, -1, q)
    return {u % q, (-u) % q}

def feasible_gear(offsets, q):
    t = teeth(q)
    return [r for r in range(q) if all((r + d) % q not in t for d in offsets)]

def feasible(offsets, gears=(5, 7, 11, 13, 17, 19, 23)):
    return all(feasible_gear(offsets, q) for q in gears)

def blocking_gears(n):
    return [q for q in (5, 7, 11, 13, 17, 19, 23, 29, 31) if q <= 2 * n]

print("=" * 74)
print("PART 1: completeness - which gears can block an n-opening shape?")
for n in range(3, 9):
    print(f"  n={n} openings: gear q can block only if q <= 2n = {2*n}"
          f"  -> {blocking_gears(n)}")
print()
print("  the 37->41 j=1 shape has n=4 openings (0, 41, 55, 96):")
offs = [0, 41, 55, 96]
for q in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
    f = feasible_gear(offs, q)
    print(f"    gear {q:>2}: {len(f):>2}/{q} phases survive"
          + ("  <- can block in principle" if q <= 8 else ""))
print(f"  every gear leaves phases, and CRT multiplies them, so the shape is")
print(f"  GENUINELY FEASIBLE - no corridor at any modulus can kill it.")

print("=" * 74)
print("PART 3: is p <= 2 provable? three padded links, patterns (j1, j2)")
print("  links: pad, [j1 literals], pad, [j2 literals], pad")
def p3_offsets(qp, j1, j2, v):
    """offsets for three padded links with j1, j2 literal links between."""
    offs = [0]
    cur = 0
    cur += qp; offs.append(cur)             # pad 1
    val = v
    for _ in range(j1):
        cur += val; offs.append(cur); val = qp - val
    cur += qp; offs.append(cur)             # pad 2
    for _ in range(j2):
        cur += val; offs.append(cur); val = qp - val
    cur += qp; offs.append(cur)             # pad 3
    return offs

for (j1, j2) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    any_ok = []
    for qp in [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
               89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149]:
        u = pow(6, -1, qp); s = (2 * u) % qp
        for v in (s, qp - s):
            offs = p3_offsets(qp, j1, j2, v)
            n = len(offs)
            gs = blocking_gears(n)
            if feasible(offs, gs):
                any_ok.append((qp, v, n))
                break
    n_ex = len(p3_offsets(41, j1, j2, 14))
    mult = sorted({(d // 41) for d in p3_offsets(41, j1, j2, 14) if d % 41 == 0})
    ap = len({m % 5 for m in mult}) >= 4
    print(f"  ({j1},{j2}): n={n_ex} openings, pure q'-multiples at i={mult}"
          f"  4-distinct-mod-5: {ap}")
    print(f"        feasible for {len(any_ok)}/27 primes"
          + (f"  first: {any_ok[:3]}" if any_ok else "  -> IMPOSSIBLE FOR ALL"))

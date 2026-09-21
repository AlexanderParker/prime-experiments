"""The word construction on the FULL opening set of machine 5 (node c.xvi).

Gear 5 strikes columns n = 1, 4 mod 5, so each five columns leave a BLOCK of three open columns:
positions p = 1, 2, 3 at columns 5b + 0, 5b + 2, 5b + 3. Gear g >= 7 strikes column n iff
n = +-c_g mod g with c_g = 6^{-1} mod g, so it strikes block b at position p iff
b = 5^{-1}(+-c_g - e_p) mod g with (e_1, e_2, e_3) = (0, 2, 3).

PRE-REGISTERED, from the leg rule (a gear strikes two columns D apart iff g | 3D - 1, 3D + 1 or D):
 (1) the three open columns sit at distances 1 (p2-p3), 2 (p1-p2) and 3 (p1-p3); 3D -+ 1 is 2, 4 for
     D = 1, 5, 7 for D = 2, and 8, 10 for D = 3 - so among gears >= 7 ONLY GEAR 7 can strike two
     cells of one block, and only positions 1 and 2. Every gear >= 11 gives at most one cell per
     block.
 (2) hence a filled block needs at least two gears, exactly two only when 7 doubles there;
 (3) gear 7 doubles exactly at blocks b = 4 mod 7.
Then compute the block words (positions struck in blocks 0..g-1) for gears 7..61 and report the
structure: doubles, empties, longest run of consecutive non-empty blocks, and the block step
between the positions of one class.
"""
from sympy import primerange

E = (0, 2, 3)


def word(g):
    c = pow(6, -1, g)
    inv5 = pow(5, -1, g)
    w = [[] for _ in range(g)]
    for sign in (c, (-c) % g):
        for p, e in enumerate(E, start=1):
            b = (inv5 * (sign - e)) % g
            w[b].append(p)
    return [sorted(set(x)) for x in w]


print("PRE-REGISTERED CHECK (1) and (2): gears able to strike two cells of one block")
for g in primerange(7, 200):
    hits = [D for D in (1, 2, 3) if (3 * D - 1) % g == 0 or (3 * D + 1) % g == 0 or D % g == 0]
    if hits:
        print(f"  gear {g}: strikes column pairs at distance(s) {hits}")
print()
w7 = word(7)
print("gear 7 block word:", " | ".join("".join(map(str, x)) or "-" for x in w7))
print("  doubles at blocks", [b for b, x in enumerate(w7) if len(x) == 2], "(predicted [4])")
print()
print("block words, gears 11..61:")
for g in primerange(11, 62):
    w = word(g)
    nonempty = [b for b, x in enumerate(w) if x]
    best = run = 0
    for b in range(2 * g):
        run = run + 1 if w[b % g] else 0
        best = max(best, run)
    c = pow(6, -1, g)
    inv5 = pow(5, -1, g)
    step12 = (-2 * inv5) % g
    step23 = (-inv5) % g
    print(f"  {g:2d}: cells in {len(nonempty):2d} of {g:2d} blocks, longest run of non-empty blocks {best}, "
          f"class steps p1->p2 {min(step12, g-step12)}, p2->p3 {min(step23, g-step23)}; word "
          + " | ".join("".join(map(str, x)) or "-" for x in w))

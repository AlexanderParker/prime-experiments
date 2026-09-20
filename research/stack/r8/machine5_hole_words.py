"""Hole words (node c.viii follow-up). Over its hole period g, gear g's word lists the positions
(1..5) it strikes in each hole. Each class gives five holes in a progression with step 7^{-1} mod
g, positions 5, 4, 3, 2, 1 in that order along the progression. PRE-REGISTERED: (1) the two
classes coincide in a hole exactly at position pairs (p, p + d_g), d_g = 15^{-1} mod g nearer
representative, so the word has 5 - d_g double holes when d_g <= 4 and none otherwise; (2) the
word has g - 10 + (number of doubles) empty holes; (3) the word is a palindrome up to the mirror
h -> h0 - h with positions p -> 6 - p (the lap mirror k -> -k). Check g = 11..61 and print the
words for 11..31.
"""
from sympy import primerange
for g in primerange(11, 62):
    a = pow(30, -1, g); d = pow(15, -1, g); d = min(d, g - d)
    word = {h: sorted(p for p in range(1, 6) if (7*h+4+p) % g in (a, (-a) % g)) for h in range(g)}
    doubles = sum(len(v) == 2 for v in word.values()); empties = sum(len(v) == 0 for v in word.values())
    assert doubles == (5 - d if d <= 4 else 0), (g, d, doubles)
    assert empties == g - 10 + doubles, (g, empties)
    for v in word.values():
        if len(v) == 2: assert v[1] - v[0] == d
    # mirror: laps k -> -k sends hole h, position p to hole h', position 6-p with 7h+4+p = -(7h'+4+6-p) mod g
    mirror_ok = all(sorted(6 - p for p in word[h]) == word[(-(h) - 2) % g] for h in range(g))  # 7h+4+p + 7h'+10-p = 7(h+h')+14 = 0 mod g -> h' = -h-2
    assert mirror_ok, g
    if g <= 31:
        print(f"gear {g} (step {pow(7,-1,g)}, lap distance {d}): " + " | ".join("".join(map(str, word[h])) or "-" for h in range(g)))
print("checked g = 11..61: doubles = 5 - d (d <= 4), empties = g - 10 + doubles, word mirror-symmetric about hole -1 (h -> -h-2, p -> 6-p)")

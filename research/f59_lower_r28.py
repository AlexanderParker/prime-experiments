"""Round 28 (mechanic): re-verify the F(59) LOWER bound from the definition.

The pin's upper half is a scan; the lower half must be an exhibited object.
Round 27's witness is a word-legal 3-gap window of machine 53 of span 161, so
by the attainment theorem (Constructor r26) F(59) >= 161.  This re-checks it
from scratch at machine 53: the four offsets are openings, EVERY other slot of
the span is blocked gear by gear, and the middle gap is a legal kill letter for
gear 59.
"""
K = 2505673933219103747
OFF = [0, 10, 128, 161]
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
u = {q: pow(6, -1, q) for q in GEARS}


def is_open(x):
    return all(x % q not in (u[q] % q, (-u[q]) % q) for q in GEARS)


oset = set(OFF)
blocked = 0
for t in range(OFF[-1] + 1):
    o = is_open(K + t)
    assert o == (t in oset), ("mismatch at offset %d: open=%s" % (t, o))
    if not o:
        blocked += 1
gaps = [OFF[i + 1] - OFF[i] for i in range(3)]
s = (2 * pow(6, -1, 59)) % 59
mid = gaps[1]
assert mid % 59 in (0, s, (-s) % 59), "middle gap not a legal letter mod 59"
print("machine-53 address k = %d" % K)
print("  openings at k + %s ; all %d other slots of the span blocked "
      "(checked slot by slot, 14 gears)" % (OFF, blocked))
print("  gaps %s  span %d" % (gaps, sum(gaps)))
print("  middle %d = %d mod 59 (s = %d, so this is the letter 0: 2q' = 118 is "
      "TWO LAPS of padding)" % (mid, mid % 59, s))
print("  => F(59) >= 161 unconditionally (attainment theorem)")
print("ALL ASSERTIONS PASSED")

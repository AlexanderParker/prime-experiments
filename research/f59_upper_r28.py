"""Round 28 (mechanic): the UPPER half of F(59) = 161, re-asserted end to end.

The round-28 band (161, 178] at JMAX = 5 is checked by
research/gate_mechanic_r28.py part F.  Everything ABOVE 178 comes from round 27,
and the claim "F(59) = 161" is only as good as the CONTIGUITY of those bands:
a gap anywhere between 178 and the top cap would leave a span unrefuted.

This re-reads every band log, re-asserts that each band's workers TILE machine
23's period exactly, and re-asserts that the bands' intervals COVER (178, 260]
with no hole.  research/akill_bands_r27.py checks the six bands it was written
for and does NOT include (178, 184], which is the one that closes the hole
between 178 and 183 - so this is not a duplicate of it.

usage: <venv>/python research/f59_upper_r28.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R27 = os.path.join(HERE, "data", "r27")
NOPEN = 7952175
# (tag, lo, hi, JMAX) - each band answers "the maximum in (lo, hi], or lo"
BANDS = [("f59_b178_184_w", 178, 184, 7),
         ("f59_b183_194_w", 183, 194, 7),
         ("f59_b193_204_w", 193, 204, 7),
         ("f59_A_w", 203, 260, 7)]


def check(tag, lo, hi):
    logs = sorted(f for f in os.listdir(R27)
                  if f.startswith(tag) and f.endswith(".log"))
    assert logs, ("no logs for band", tag)
    iv, mx = [], []
    for f in logs:
        t = open(os.path.join(R27, f), errors="replace").read()
        assert "scan complete" in t, ("worker did not finish", f)
        m = re.search(r"WALKING start-opening indices \[([\d,]+), ([\d,]+)\)",
                      t)
        iv.append((int(m.group(1).replace(",", "")),
                   int(m.group(2).replace(",", ""))))
        mx += [int(x) for x in re.findall(r"max over J = (\d+)", t)]
    iv.sort()
    assert iv[0][0] == 0 and iv[-1][1] == NOPEN, ("not a tiling", tag, iv)
    for a, b in zip(iv, iv[1:]):
        assert a[1] == b[0], ("gap or overlap in the tiling", tag, a, b)
    assert set(mx) == {lo}, ("band NOT empty", tag, sorted(set(mx)))
    return len(logs)


def main():
    print("THE UPPER HALF OF F(59) = 161 - round-27 bands, re-read and "
          "re-asserted\n")
    cover = []
    for tag, lo, hi, j in BANDS:
        n = check(tag, lo, hi)
        print("  band (%3d, %3d] at JMAX = %d: %d workers TILE [0, %d), "
              "EMPTY" % (lo, hi, j, n, NOPEN))
        cover.append((lo, hi))
    cover.sort()
    top = cover[0][0]
    for lo, hi in cover:
        assert lo <= top, ("HOLE in the coverage below", lo, "top so far", top)
        top = max(top, hi)
    print("\n  the bands cover (%d, %d] with no hole, at depth <= 7"
          % (cover[0][0], top))
    print("  round-28's own band (161, 178] at JMAX = 5 closes (161, 178];")
    print("  JMAX = 5 is exhaustive there because A_kill(53->59) = 4 (N_5 = 0),")
    print("  so no word-legal window of 6 or 7 gaps exists at ANY span.")
    print("\n  => F(59) <= 161, and F(59) >= 161 by the exhibited machine-53")
    print("     window (research/f59_lower_r28.py).  F(59) = 161 EXACT,")
    print("     conditional only on the top cap: no word-legal window of")
    print("     machine 53 of span above %d at depth <= 7." % top)
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

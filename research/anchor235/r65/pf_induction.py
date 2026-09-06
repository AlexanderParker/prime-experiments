"""r65 / position-length frontier, part 4: the induction attempt, exactly.

The frontier theorem, conditional on the record law one machine down:

  a blocked stretch of {5..q} of length L starting at a column x with 6x - 1 > q is a blocked
  stretch of the effective machine E = {5..y}, y = the largest prime with y^2 <= 6(x+L-1)+1
  (proved: a member n > q with a prime factor p in [5,q] has n/p >= 5, whose least prime factor
  is a gear below sqrt(n)).  So L + 1 <= F(E), i.e.

      x  >=  ceil((y_L^2 - 1)/6) - L + 1  =:  bound(L),   y_L = least prime with F(y_L) >= L+1.

This script tabulates bound(L)/L -- the frontier constant the certified F ladder delivers --
against the measured R_min^>=(L)/L, and states what upper bound on F(y) each constant needs.

Self-contained, no dependencies beyond the standard library.
Run: uv run python research/anchor235/r65/pf_induction.py
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

FLADDER = [(5, 2), (7, 5), (11, 7), (13, 11), (17, 18), (19, 25), (23, 34), (29, 43), (31, 58)]

# measured R_min^>=(L) on full periods (pf_period.py, pf_period29.py)
MEAS = {
    7:  {1: 1, 2: 8, 3: 13, 4: 13},
    11: {1: 1, 2: 1, 3: 13, 4: 13, 5: 53, 6: 151},
    13: {1: 1, 2: 1, 3: 13, 4: 13, 5: 53, 6: 89, 7: 123, 8: 123, 9: 123, 10: 123},
    17: {1: 1, 2: 1, 3: 1, 4: 1, 5: 53, 6: 61, 7: 61, 8: 61, 9: 61, 10: 118, 11: 118,
         12: 118, 13: 118, 14: 118, 15: 118, 16: 118, 17: 118},
    19: {1: 1, 2: 1, 3: 1, 4: 1, 5: 53, 6: 59, 7: 59, 8: 59, 9: 59, 10: 59, 11: 59,
         12: 111, 13: 111, 14: 111, 15: 111, 16: 111, 17: 111, 18: 111, 19: 111, 20: 111,
         21: 111, 22: 111, 23: 111, 24: 111},
    23: {1: 1, 2: 1, 3: 1, 4: 1, 5: 53, 6: 59, 7: 59, 8: 59, 9: 59, 10: 59, 11: 59,
         12: 111, 13: 111, 14: 111, 15: 111, 16: 111, 17: 111, 18: 111, 19: 111, 20: 111,
         21: 111, 22: 111, 23: 111, 24: 111, 25: 40148, 26: 170034, 27: 190056,
         28: 396199, 29: 396199, 30: 1479278, 31: 2553844, 32: 5606403, 33: 12694429},
    29: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 59, 8: 59, 9: 59, 10: 59, 11: 59,
         12: 111, 13: 111, 14: 111, 15: 111, 16: 111, 17: 111, 18: 111, 19: 111, 20: 111,
         21: 111, 22: 111, 23: 111, 24: 111, 25: 5643, 26: 23278, 27: 23278, 28: 35564,
         29: 35564, 30: 102273, 31: 102273, 32: 102273, 33: 2278076, 34: 2278076,
         35: 2900801, 36: 2900801, 37: 6603768, 38: 144154491, 39: 144154491,
         40: 200906186, 41: 200906186, 42: 200906186},
}
D0 = {7: 2, 11: 3, 13: 3, 17: 5, 19: 5, 23: 5, 29: 7}


def y_for(L):
    for y, F in FLADDER:
        if F >= L + 1:
            return y, F
    return None, None


def bound(L):
    y, _ = y_for(L)
    if y is None:
        return None, None
    return -(-(y * y - 1) // 6) - L + 1, y


def main():
    lines = []
    W = lines.append
    W("THE RECORD LAW AS A FRONTIER CONSTANT")
    W("")
    W("F(y) against y^2/6 (the root's own scale):")
    W("    y     F(y)   y^2/6   F(y)/(y^2/6)")
    for y, F in FLADDER:
        W("  %3d %7d %8.2f %11.3f" % (y, F, y * y / 6, F / (y * y / 6)))
    W("")
    W("  max F(y)/(y^2/6) = 0.612 at y = 7, so the crude form 'F(y) <= y^2/(6(c+1)) for all y'")
    W("  is available only with c+1 = 1/0.612, i.e. c = 0.634.  The L-by-L form is stronger:")
    W("")
    W("bound(L) = ceil((y_L^2-1)/6) - L + 1, y_L the least prime with F(y_L) >= L+1:")
    W("     L   y_L   F(y_L)   bound(L)   bound(L)/L")
    best = None
    for L in range(1, 58):
        b, y = bound(L)
        if b is None:
            continue
        r = b / L
        if best is None or r < best[0]:
            best = (r, L, y)
        if L <= 12 or L in (17, 24, 33, 42, 57) or y != bound(L - 1)[1]:
            W("  %4d %5d %8d %10d %12.3f" % (L, y, dict(FLADDER)[y], b, r))
    W("")
    W("  MINIMUM bound(L)/L over L = 1..57: %.3f at L = %d (effective machine {5..%d})"
      % best)
    r6 = min((bound(L)[0] / L, L) for L in range(6, 58) if bound(L)[0])
    W("  minimum over L >= 6: %.3f at L = %d" % r6)
    W("")
    W("  So the certified ladder delivers the frontier theorem with c = %.2f for every L,"
      % best[0])
    W("  and with c = %.2f once L >= 6.  For c = 3 one would need F(y) <= y^2/24 at every y,"
      % r6[0])
    W("  which the ladder refutes at y = 5, 7, 11 (F/(y^2/24) = 1.92, 2.45, 1.39).")
    W("")
    W("MEASURED FRONTIER against the bound, full periods:")
    W("   machine   d_0   min R_min/L over L>=d_0  at L   min bound/L over L>=d_0  at L"
      "   truth/bound at the minimum")
    for q in sorted(MEAS):
        m = MEAS[q]
        d0 = D0[q]
        Ls = [L for L in m if L >= d0]
        tr = min((m[L] / L, L) for L in Ls)
        bd = min((bound(L)[0] / L, L) for L in Ls if bound(L)[0])
        W("   m%-8d %3d %22.3f %6d %24.3f %6d %12.2f"
          % (q, d0, tr[0], tr[1], bd[0], bd[1], tr[0] / bd[0]))
    W("")
    W("  and on the exception set L < d_0 the measured value is 1 at every machine, so the")
    W("  bound is violated there by a factor bound(L) (4, 7, 6, 5 at L = 1, 2, 3, 4).")
    W("")
    W("WHERE THE BOUND HAS CONTENT AT ALL:")
    W("  y = q at 6x = q^2, so the effective machine is a proper sub-machine only for")
    W("  x < (q^2-1)/6 -- the whole prefix except the new section (q^2, q'^2].  Above that,")
    W("  and everywhere deeper in the period, E = the whole machine and the bound reads")
    W("  L + 1 <= F(M): the root itself.  Example measured: the m29 record stretch sits at")
    W("  x = 200,906,186 where E = {5..34703}, 1197 times the machine.")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "pf_induction.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

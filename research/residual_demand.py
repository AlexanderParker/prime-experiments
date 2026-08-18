"""Round 18 lateral, part 2: THE RESIDUAL DEMAND CONSTRUCT.

The autocorrelation sigma(g) is the ENDPOINT half of a gap's arithmetic. The
other half is the INTERIOR: for a gap of exactly g, all g-1 interior slots must
be killed. The small gears kill some for free; the rest must be bought from the
big gears. So define, over phases r that keep both endpoints exposed,

    D(g) = min over admissible r of  #{ i in 1..g-1 : r+i exposed to 5 and 7 }

- the RESIDUAL DEMAND of the lag: how many interior slots the small gears
cannot kill, whatever the phase. A gap of value g needs D(g) kills bought from
gears >= 11, and each such gear q supplies at most 2*ceil((g-1)/q).

D is a relationship object (lag x small-gear coverage), computable in closed
form from mod-35 data alone, and it is the missing half of the erraticity.
"""
from math import prod
import numpy as np
from split_gap_law import primes
from exposed_autocorr import sigma, c_q, gap_hist

E35 = [k for k in range(35) if k % 5 not in (1, 4) and k % 7 not in (1, 6)]
E = set(E35)

def demand(g):
    """(min residual demand, #admissible phases mod 35)."""
    best, n = None, 0
    for r in range(35):
        if r % 35 not in E or (r + g) % 35 not in E:
            continue
        n += 1
        d = sum(1 for i in range(1, g) if (r + i) % 35 in E)
        best = d if best is None else min(best, d)
    return best, n

def supply(g, gears):
    """max interior kills purchasable from gears >= 11 in a span of g-1."""
    return sum(2 * ((g - 1 + q - 1) // q) for q in gears if q >= 11)

print("=" * 78)
print("PART 2: residual demand D(g) vs measured gap counts")
for y in (19, 23):
    cnt, gears = gap_hist(y)
    F = int(max(i for i in range(len(cnt)) if cnt[i]))
    print(f"  --- machine {y}, F = {F} ---")
    print(f"  {'g':>3} {'count':>7} {'sigma':>7} {'phases':>7} {'D(g)':>5} "
          f"{'supply':>7} {'slack':>6}  verdict")
    for g in range(20, min(F + 8, 46)):
        s = sigma(gears, g)
        d, n = demand(g)
        sup = supply(g, gears)
        slack = sup - d if d is not None else None
        v = ""
        if cnt[g] == 0:
            v = "ABSENT"
            if d is not None and slack is not None and slack < 0:
                v += " - demand exceeds supply (STRUCTURAL)"
            elif g > F:
                v += " - beyond F"
            else:
                v += " - feasible on this test, so rarity not structure"
        print(f"  {g:>3} {int(cnt[g]):>7} {s:>7.4f} {n:>7} "
              f"{str(d):>5} {sup:>7} {str(slack):>6}  {v}")

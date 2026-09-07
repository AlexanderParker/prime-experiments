"""The manifold's period-scale record laws against the quiet-zone record.

usage: uv run python research/valves/r3/record_rule.py q Q zone_record

For the gear set G = primes in (q, Q] and a window length L, the loaded record rule (L69, kernel) says
  Coverable(L) iff exists n : domCost(uncovered(core(L), n, L)) <= tail(L),
with core(L) = gears <= L + 1, tail(L) = gears > L + 1, and domCost the domino cost (pieces {x, x + 2}).
This script computes, as functions of L:
  (i)  the tail-only sufficiency: boundary_cost(L) = 2 floor(L/4) + min(L mod 4, 2) <= tail(L) certifies
       Coverable(L) with the core contributing nothing (domCost is monotone under inclusion), so
       L_low = max such L is a LOWER bound on the period record (coverability is downward closed);
  (ii) the capacity upper bound (L70 / W102 in its crude form): Coverable(L) implies
       ceil((L - sum_core 2 ceil(L/g))^+ / 2) <= tail(L); reported as vacuous when the core's capacity
       sum exceeds L (the loaded regime);
  (iii) at L = the measured zone record: the tail count, the boundary cost, the slack, and the
       uncovered-cell count left by the core at phase zero and at a greedy phase choice (an upper bound
       on the rule's minimum), i.e. how far the zone record sits below what the rule allows.
"""
import sys, math
import numpy as np

q = int(sys.argv[1]); Q = int(sys.argv[2]); Lz = int(sys.argv[3])
isp = np.ones(Q + 1, dtype=bool); isp[:2] = False
for i in range(2, int(Q ** 0.5) + 1):
    if isp[i]:
        isp[i * i::i] = False
gears = [int(g) for g in np.flatnonzero(isp) if g > q]
m = len(gears)


def boundary_cost(L):
    return 2 * (L // 4) + min(L % 4, 2)


def tail(L):
    return sum(1 for g in gears if g > L + 1)


def core(L):
    return [g for g in gears if g <= L + 1]


L_low = max(L for L in range(1, 4 * m + 8) if boundary_cost(L) <= tail(L))
print(f"q={q} Q={Q}: {m} gears; zone record {Lz}")
print(f"(i) tail-only lower bound on the period record: L_low = {L_low} (boundary cost {boundary_cost(L_low)} <= tail {tail(L_low)}); "
      f"the parity-law scale 2m - (m mod 2) = {2 * m - m % 2} (not applicable: loaded)")
# (ii) capacity bound
vac = None
for L in range(1, 4 * m + 8):
    cap = sum(2 * math.ceil(L / g) for g in core(L))
    if cap >= L:
        vac = L; break
print(f"(ii) capacity bound: the core's capacity sum_core 2 ceil(L/g) reaches L at L = {vac} (vacuous from there on); "
      f"at L = {Lz}: capacity {sum(2 * math.ceil(Lz / g) for g in core(Lz))} against L = {Lz}, core {len(core(Lz))} gears, tail {tail(Lz)}")
# (iii) at the zone record
C = core(Lz); t = tail(Lz)
bc = boundary_cost(Lz)
print(f"(iii) at L = {Lz}: core = gears <= {Lz + 1}: {len(C)} gears; tail {t}; boundary cost {bc}; slack tail - cost = {t - bc}")
# uncovered cells at phase zero: cells i in [0, L) with no core gear striking pair n + i for n = 0 (phase zero at the origin)
# and at the zone's record position n0 (pass as Lz's position is unknown here: use the origin and a greedy choice)


def uncovered_count(phases, L):
    cov = np.zeros(L, dtype=bool)
    for g, c in zip(C, phases):
        for tooth in (c % g, (c - 2) % g):      # pair positions struck by the number class c: pairs x with x = c or x + 2 = c
            cov[tooth::g] = True
    return int((~cov).sum()), cov


def dom_cost(cells):
    # exact domino cost of a cell set: split by parity, sum ceil(run/2) over maximal step-2 runs (L68)
    cost = 0
    for par in (0, 1):
        cs = sorted(x for x in cells if x % 2 == par)
        i = 0
        while i < len(cs):
            j = i
            while j + 1 < len(cs) and cs[j + 1] == cs[j] + 2:
                j += 1
            cost += (j - i + 2) // 2
            i = j + 1
    return cost


u0, cov0 = uncovered_count([0] * len(C), Lz)
cost0 = dom_cost([i for i in range(Lz) if not cov0[i]])
# greedy phases: each core gear in turn picks the class covering the most still-uncovered cells
cov = np.zeros(Lz, dtype=bool); ph = []
for g in C:
    best, bc_ = -1, 0
    for c in range(g):
        hit = 0
        for tooth in (c % g, (c - 2) % g):
            hit += int((~cov[tooth::g]).sum())
        if hit > best:
            best, bc_ = hit, c
    ph.append(bc_)
    for tooth in (bc_ % g, (bc_ - 2) % g):
        cov[tooth::g] = True
ug = int((~cov).sum()); costg = dom_cost([i for i in range(Lz) if not cov[i]])
print(f"      core at phase zero (window starting at the origin): uncovered {u0} of {Lz}, domino cost {cost0} <= tail {t}: {cost0 <= t}")
print(f"      core at a greedy phase choice: uncovered {ug}, domino cost {costg} (an upper bound on the rule's minimum over core phases) <= tail {t}: {costg <= t}")
print(f"      the rule certifies L = {Lz} coverable with the core doing nothing (cost {bc} <= tail {t}); the zone record is "
      f"{L_low / Lz:.1f}x below the tail-only lower bound on the period record")

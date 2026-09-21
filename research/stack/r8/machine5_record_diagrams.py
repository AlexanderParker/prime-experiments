"""Record diagrams (node c.xii follow-up). For q = 19..31 take the record walk's phases and print
the diagram: strikers of each lap, the laps each gear strikes (which class), classes never used,
coincidences (two gears on one lap), and at the end lap the nearest miss of every gear (how many
laps away its next strike is). Question: what stops the walk - a gear whose second class is spent
early, or a gear whose two classes both fall just outside.
"""
from sympy import primerange
PH = {19: {7: 3, 17: 11, 11: 1, 19: 3, 13: 5},
      23: {7: 3, 11: 2, 17: 1, 13: 6, 19: 1, 23: 1},
      29: {7: 3, 17: 11, 19: 9, 29: 24, 13: 5, 11: 6, 23: 20},
      31: {7: 3, 11: 2, 23: 10, 13: 12, 19: 6, 17: 12, 29: 18, 31: 11}}
for q, ph in PH.items():
    gears = sorted(ph); cls = {g: (pow(30, -1, g), (-pow(30, -1, g)) % g) for g in gears}
    def hits(i): return [(g, '+' if (ph[g] + i) % g == cls[g][0] else '-') for g in gears if (ph[g] + i) % g in cls[g]]
    L = 0
    while hits(L): L += 1
    print(f"q={q}: record {L}")
    print("  lap: strikers  ", " | ".join(f"{i}:" + ",".join(f"{g}{c}" for g, c in hits(i)) for i in range(L)))
    for g in gears:
        used = [(i, c) for i in range(L) for gg, c in hits(i) if gg == g]
        print(f"  gear {g:2d}: laps {[i for i, _ in used]} classes {''.join(c for _, c in used)}"
              + ("   (one class unused)" if len({c for _, c in used}) == 1 else ""))
    co = [i for i in range(L) if len(hits(i)) > 1]
    nxt = {g: min(((c - ph[g] - L) % g) for c in cls[g]) for g in gears}
    prv = {g: min(((ph[g] - 1 - c) % g) for c in cls[g]) for g in gears}
    print(f"  coincidences at laps {co}; end lap {L}: next strike of each gear in {nxt} laps; before lap 0: previous strike of each gear {prv} laps back")

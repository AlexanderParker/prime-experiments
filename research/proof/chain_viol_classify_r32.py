"""Prover B, round 32 -- classify the chain-statement violators of the tooth-counterfactual
family (chain_family_r32_viol_m<y>.json): which ingredients they still satisfy.
  degenerate old gear : v_q = (q-1)/2  (teeth adjacent: 2v = -1 mod q; antipode struck)
  degenerate incoming : v' = (q'-1)/2  (letters +-1)
  pinned incoming     : v' = round(q'/6)  (3a = q' -+ 1 holds)
Usage: uv run python research/proof/chain_viol_classify_r32.py 11 13 17 19
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PR = [5, 7, 11, 13, 17, 19, 23, 29]

for y in [int(t) for t in sys.argv[1:]]:
    gears = [p for p in PR if p <= y]
    q1 = [p for p in PR if p > y][0]
    vp = round(q1 / 6)
    real = tuple(round(q / 6) for q in gears)
    path = os.path.join(HERE, f"chain_family_r32_viol_m{y}.json")
    V = json.load(open(path))
    print(f"\nm{y} q'={q1}: {len(V)} violating rows")
    degold = [r for r in V if any(v == (q - 1) // 2 for q, v in zip(gears, r['teeth']))]
    deginc = [r for r in V if r['v1'] == (q1 - 1) // 2]
    clean = [r for r in V if r not in degold and r not in deginc]
    pinned = [r for r in V if r['v1'] == vp]
    pinned_clean = [r for r in pinned if r in clean]
    print(f"  degenerate old gear (adjacent teeth): {len(degold)}; degenerate incoming (a=1): {len(deginc)};"
          f" NON-degenerate (every gear's teeth non-adjacent, antipode open, a >= 2): {len(clean)}")
    print(f"  pinned incoming (3a = q' -+ 1): {len(pinned)}, of which non-degenerate: {len(pinned_clean)}")
    print(f"  pair statement holds at {sum(r['pair_ok'] for r in V)} of {len(V)}")
    # which old gears differ from the real teeth
    diff = Counter(sum(1 for a, b in zip(r['teeth'], real) if a != b) for r in V)
    print(f"  number of old gears with moved teeth, distribution: {dict(sorted(diff.items()))}")
    per_gear = Counter(q for r in V for q, a, b in zip(gears, r['teeth'], real) if a != b)
    print(f"  gears moved (count over violators): {dict(sorted(per_gear.items()))}")
    v57 = Counter((r['teeth'][0], r['teeth'][1]) for r in V)
    print(f"  (v_5, v_7) of violators: {dict(sorted(v57.items()))}  (real = (1,1))")
    # word shapes at the violating J
    shapes = Counter()
    bisF = 0
    cells = 0
    for r in V:
        for J, e in r['viol'].items():
            w, gL, gR, lit = r['argmax'][J]
            cells += 1
            q = q1
            cls = ''.join('0' if v % q == 0 else ('a' if v % q == r['a'] else 'b') for v in w)
            shapes[(int(J), cls, 'lit' if lit else 'pad')] += 1
            if any(v == r['F'] for v in w):
                bisF += 1
    print(f"  violating cells {cells}; a middle letter equal to F(M) in {bisF} of them")
    print("  shapes (J, class word, literal/padded): " + ", ".join(f"{k}:{n}" for k, n in sorted(shapes.items())))
    for tag, sel in [("pinned non-degenerate", pinned_clean), ("free non-degenerate", clean[:3])]:
        for r in sel[:3]:
            J = max(r['viol'], key=lambda J: r['viol'][J])
            w, gL, gR, lit = r['argmax'][J]
            print(f"  witness [{tag}]: teeth {tuple(r['teeth'])} v'={r['v1']} a={r['a']} F={r['F']} F_2={r['F2']}"
                  f" budget {r['F'] + q1}: J={J} span {r['F'] + q1 + r['viol'][J]} = ({gL}) + {w} + ({gR})"
                  f" {'literal' if lit else 'padded'}; L={r['L']}")

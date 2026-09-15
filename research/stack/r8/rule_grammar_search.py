"""Loop, iteration 22: every combination of step rules from a grammar (owner: not all
combinations of rules have been tried).

A rule is fixed across machines and reads no teeth: it names an origin, and one or two flips,
each with a mirror, periods and a direction, all drawn from the machine's own quantities.
  origins: home (-1, 1); the twin gear pairs (5,7), (11,13), (17,19), (29,31), (41,43) when
           inside the machine; the largest twin gear pair at most q; the largest at most sqrt q
  mirrors: B = the spiral base (lowest gears with product at most q/2); Pmax = the largest
           primorial at most q^2/2; Pmax' = the one below it; B*q; B*p (p the prime below q);
           B*g (g the first gear above the base); 6q; 6p; 30q; B*(q-p)/2 ... only integers:
           B*q, B*p, B*g, 6q, 6p, 30q, Pmax, Pmax', B, and B*q*p when at most q^2/2
  periods k: 1, 2, 3;  direction: up, down
One-flip rules: origin + mirror + k + direction.  Two-flip rules: two such steps in sequence
(landing of the first as the origin of the second).  Every rule is scored over the machines
11 to 2000: the number of machines where the landing is a twin inside the window.  Reported:
the best rules of each length, and the distribution.
usage: uv run python research/stack/r8/rule_grammar_search.py 2000
"""
import sys, itertools
import numpy as np
from sympy import primerange
from pathlib import Path

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = 4 * Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    twin = lambda n: 0 <= n < N - 2 and bool(sv[n] and sv[n + 2])
    onames = ['home', '(5,7)', '(11,13)', '(17,19)', '(29,31)', '(41,43)', 'top twin gears', 'top twin gears <= sqrt q']
    mnames = ['B', 'Pmax', "Pmax'", 'B*q', 'B*p', 'B*g', '6q', '6p', '30q', 'B*q*p']
    per_machine = []
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; B = 1
        for p in ps:
            if B * p <= q // 2: B *= p; base.append(p)
            else: break
        prims = []; P = 1
        for p in ps:
            P *= p
            if P <= q * q // 2: prims.append(P)
            else: break
        p_prev = ps[-2]; g = next(x for x in ps if x not in base)
        tg = [x for x in ps if x + 2 in ps]; tgs = [x for x in tg if x * x <= q]
        origins = {'home': -1, '(5,7)': 5 if q >= 7 else None, '(11,13)': 11 if q >= 13 else None, '(17,19)': 17 if q >= 19 else None,
                   '(29,31)': 29 if q >= 31 else None, '(41,43)': 41 if q >= 43 else None,
                   'top twin gears': tg[-1] if tg else None, 'top twin gears <= sqrt q': tgs[-1] if tgs else None}
        mirrors = {'B': B, 'Pmax': prims[-1], "Pmax'": prims[-2] if len(prims) > 1 else None, 'B*q': B * q, 'B*p': B * p_prev, 'B*g': B * g,
                   '6q': 6 * q, '6p': 6 * p_prev, '30q': 30 * q, 'B*q*p': B * q * p_prev if B * q * p_prev <= q * q // 2 else None}
        per_machine.append((q, origins, mirrors))
    steps = [(m, k, d) for m in mnames for k in (1, 2, 3) for d in (1, -1)]
    results = []
    for o in onames:
        for s1 in steps:
            score = 0
            for q, origins, mirrors in per_machine:
                O = origins[o]; M = mirrors[s1[0]]
                if O is None or M is None: continue
                L = O + 2 * s1[1] * M * s1[2]
                if q < L <= q * q - 2 and twin(L): score += 1
            results.append((score, o, s1, None))
    one_best = sorted(results, key=lambda r: -r[0])[:8]
    results2 = []
    for o in onames:
        for s1, s2 in itertools.product(steps, steps):
            score = 0
            for q, origins, mirrors in per_machine:
                O = origins[o]; M1 = mirrors[s1[0]]; M2 = mirrors[s2[0]]
                if O is None or M1 is None or M2 is None: continue
                L = O + 2 * s1[1] * M1 * s1[2] + 2 * s2[1] * M2 * s2[2]
                if q < L <= q * q - 2 and twin(L): score += 1
            results2.append((score, o, s1, s2))
    two_best = sorted(results2, key=lambda r: -r[0])[:12]
    scores2 = [r[0] for r in results2]
    out = [__doc__.strip(), "", f"machines {len(qs)} (11 to {Q}); one-flip rules {len(results)}, two-flip rules {len(results2)}", "",
           "best one-flip rules (machines landing on a twin in the window):"]
    for sc, o, s1, _ in one_best: out.append(f"   {sc:>4}  origin {o:<26} flip {s1[0]} x{s1[1]} {'up' if s1[2] > 0 else 'down'}")
    out.append(""); out.append("best two-flip rules:")
    for sc, o, s1, s2 in two_best: out.append(f"   {sc:>4}  origin {o:<26} {s1[0]} x{s1[1]} {'up' if s1[2] > 0 else 'down'}, then {s2[0]} x{s2[1]} {'up' if s2[2] > 0 else 'down'}")
    out.append(""); out.append(f"two-flip score distribution: max {max(scores2)}, rules above 100: {sum(s > 100 for s in scores2)}, above 150: {sum(s > 150 for s in scores2)}, above 200: {sum(s > 200 for s in scores2)}")
    Path("research/stack/r8/results_rule_grammar_search.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

"""Loop, iteration 23: a larger rule grammar, up to three flips.

Origins: home; the twin gear pairs (5,7) .. (101,103) inside the machine; the top twin gear pair;
the top one at most sqrt q; the top one at most q/2.
Mirrors (machine-varying): B*g1, B*g2, B*g3 (B the spiral base, g1 < g2 < g3 the first gears
above it), B*q, B*p, B*p2 (p, p2 the two primes below q), 6q, 6p, 6*g1, B*(q-p)/... only
integers: B*q, B*p, B*p2, B*g1, B*g2, B*g3, 6q, 6p, 6g1, B*q*g1 (when at most q^2/2), 2q, 2p.
Periods: 1, 2, 3, and two rules: 'enter' = the smallest k that puts the landing above q,
'top' = the largest k that keeps it at most q^2 - 2.  Directions up and down (rules 'enter'
and 'top' are up only).
Scored over the machines 11 to 2000; one, two and three flips (three flips restricted to the
mirrors B*g1, B*q, B*p, 6q, 6g1 and periods 1..2 to keep the count near 10^5).
usage: uv run python research/stack/r8/rule_grammar_search2.py 2000
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
    tgp = [5, 11, 17, 29, 41, 59, 71, 101]
    onames = ['home'] + [f'({p},{p+2})' for p in tgp] + ['top twin', 'top twin <= sqrt q', 'top twin <= q/2']
    mnames = ['B*q', 'B*p', 'B*p2', 'B*g1', 'B*g2', 'B*g3', '6q', '6p', '6g1', 'B*q*g1', '2q', '2p']
    per = []
    for q in qs:
        ps = list(primerange(2, q + 1)); base = []; B = 1
        for p in ps:
            if B * p <= q // 2: B *= p; base.append(p)
            else: break
        above = [x for x in ps if x not in base]; g1, g2, g3 = (above + [None, None, None])[:3]
        p1, p2 = ps[-2], ps[-3]
        tg = [x for x in ps if x + 2 in ps]
        origins = {'home': -1, 'top twin': tg[-1] if tg else None, 'top twin <= sqrt q': max([x for x in tg if x * x <= q], default=None), 'top twin <= q/2': max([x for x in tg if 2 * x <= q], default=None)}
        for p in tgp: origins[f'({p},{p+2})'] = p if p + 2 <= q else None
        mirrors = {'B*q': B * q, 'B*p': B * p1, 'B*p2': B * p2, 'B*g1': B * g1 if g1 else None, 'B*g2': B * g2 if g2 else None, 'B*g3': B * g3 if g3 else None,
                   '6q': 6 * q, '6p': 6 * p1, '6g1': 6 * g1 if g1 else None, 'B*q*g1': B * q * g1 if g1 and B * q * g1 <= q * q // 2 else None, '2q': 2 * q, '2p': 2 * p1}
        per.append((q, origins, mirrors))
    ks = [1, 2, 3, 'enter', 'top']
    def land(O, M, k, d, q):
        if k == 'enter':
            j = 1
            while O + 2 * j * M <= q: j += 1
            return O + 2 * j * M
        if k == 'top':
            j = (q * q - 2 - O) // (2 * M)
            return O + 2 * j * M if j >= 1 else None
        return O + 2 * k * M * d
    steps = [(m, k, d) for m in mnames for k in ks for d in ((1,) if k in ('enter', 'top') else (1, -1))]
    def score(o, seq):
        sc = 0
        for q, origins, mirrors in per:
            O = origins[o]
            if O is None: continue
            L = O; ok = True
            for m, k, d in seq:
                M = mirrors[m]
                if M is None: ok = False; break
                L = land(L, M, k, d, q)
                if L is None: ok = False; break
            if ok and q < L <= q * q - 2 and twin(L): sc += 1
        return sc
    out = [__doc__.strip(), "", f"machines {len(qs)}"]
    res1 = sorted(((score(o, [s]), o, [s]) for o in onames for s in steps), key=lambda r: -r[0])
    out.append(f"one flip: {len(steps) * len(onames)} rules; best:")
    for sc, o, seq in res1[:8]: out.append(f"   {sc:>4}  {o:<20} " + "; ".join(f"{m} x{k} {'up' if d > 0 else 'down'}" for m, k, d in seq))
    res2 = sorted(((score(o, [s1, s2]), o, [s1, s2]) for o in onames for s1 in steps for s2 in steps), key=lambda r: -r[0])
    out.append(f"two flips: {len(steps) ** 2 * len(onames)} rules; best:")
    for sc, o, seq in res2[:8]: out.append(f"   {sc:>4}  {o:<20} " + "; ".join(f"{m} x{k} {'up' if d > 0 else 'down'}" for m, k, d in seq))
    small = [(m, k, d) for m in ('B*g1', 'B*q', 'B*p', '6q', '6g1') for k in (1, 2, 'enter') for d in ((1,) if k == 'enter' else (1, -1))]
    res3 = sorted(((score(o, [s1, s2, s3]), o, [s1, s2, s3]) for o in ('home', '(29,31)', 'top twin <= sqrt q') for s1 in small for s2 in small for s3 in small), key=lambda r: -r[0])
    out.append(f"three flips (restricted): {len(small) ** 3 * 3} rules; best:")
    for sc, o, seq in res3[:8]: out.append(f"   {sc:>4}  {o:<20} " + "; ".join(f"{m} x{k} {'up' if d > 0 else 'down'}" for m, k, d in seq))
    Path("research/stack/r8/results_rule_grammar_search2.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

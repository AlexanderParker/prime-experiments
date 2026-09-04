"""Prover B, round 32 -- per-word slack of the chain statement on the r30 counted census.

For every realised legal word w of M (research/data/r30/occ_<y>_words.json, exact cyclic
census, key "w1 w2 ..." -> [occ, Phi, gL*, gR*]) the chain statement at depth J = len(w) + 2
is exactly   span(w) + Phi(w) <= F(M) + q'.   slack(w) = F + q' - span(w) - Phi(w).
Gate: max(F_2, max_w span(w) + Phi(w)) == F(M+q') (attainment identity, recorded F(M+q')).
Usage: uv run python research/proof/chain_slack_r32.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(HERE, "..", "data", "r30")
F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90}
FNEW = {11: 11, 13: 18, 17: 25, 19: 34, 23: 43, 29: 58, 31: 88, 37: 91}
QP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41}


def letters(q1):
    u = round(q1 / 6)
    assert (6 * u) % q1 in (1, q1 - 1)
    a = 2 * u
    return a, q1 - a


for y in sorted(F):
    q1 = QP[y]
    a, b = letters(q1)
    with open(os.path.join(DD, f"occ_{y}_words.json")) as fh:
        W = json.load(fh)
    rows = []
    for key, (occ, phi, gl, gr) in W.items():
        w = tuple(int(t) for t in key.split())
        cls = [(0 if v % q1 == 0 else (1 if v % q1 == a else -1)) for v in w]
        nz = [c for c in cls if c != 0]
        if any(x == y2 for x, y2 in zip(nz, nz[1:])):
            continue                      # same class twice: not T3-legal (census lists them too)
        span = sum(w)
        rows.append((F[y] + q1 - span - phi, w, occ, phi, gl, gr, span, 0 in cls))
    rows.sort()
    att = max([F2[y]] + [r[6] + r[3] for r in rows])
    print(f"m{y:<3} q'={q1:<3} a={a:<3} b={b:<3} F={F[y]:<3} F_2={F2[y]:<3} budget F+q'={F[y] + q1:<4}"
          f" words={len(rows):<3} GATE attainment max(F_2, max span+Phi)={att} == F(M+q')={FNEW[y]}"
          f" {'OK' if att == FNEW[y] else 'MISMATCH'}")
    print("   smallest slacks:")
    for s, w, occ, phi, gl, gr, span, pad in rows[:5]:
        print(f"     slack {s:>3}  w={w!s:<22} span={span:<3} Phi={phi:<3} argmax flanks ({gl},{gr})"
              f" occ={occ:<10} {'padded' if pad else 'literal'}  J={len(w) + 2}")
    named = {(a,): "(a)", (b,): "(b)", (q1,): "(q')", (a, b): "(a,b)", (b, a): "(b,a)", (a, b, a): "(a,b,a)"}
    got = {r[1]: r for r in rows}
    print("   named words: " + "; ".join(
        f"{nm} slack={got[w][0]} Phi={got[w][3]}" if w in got else f"{nm} not realised" for w, nm in named.items()))
    # the J = 3 literal cells: the statement is  Phi(v) <= F + q' - v  for each legal letter v
    lit1 = [(r[1][0], r[3], F[y] + q1 - r[1][0], r[0]) for r in rows if len(r[1]) == 1]
    print("   J=3 cells (letter v, Phi(v), need Phi <= F+q'-v, slack): " + "; ".join(
        f"v={v}: Phi={p} <= {need} (+{s})" for v, p, need, s in sorted(lit1)))

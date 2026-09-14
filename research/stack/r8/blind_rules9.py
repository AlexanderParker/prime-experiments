"""Loop, iteration 9: the sub-machine's landing plus one flip with the top gear (no primality).
E1 = the level-1 landing of the levels walk (base with 2 and 3 at every level, levels while
b_k >= 11), at most q.  Then one flip up with the fewest periods that enter the window:
  L1 mirror {3, q}: E1 + 6 k q       L2 {2,3,q}: E1 + 12 k q      L3 {base, q}: E1 + 2 k P q
  L4 {3, p} with p the prime below q  L5 {3, q} then {3, p} down: E1 + 6q - 6p
  L6 {2,3,q} then {2,3,p} down: E1 + 12 (q - p)
Machines 121 to 5000: twin / in window.  usage: uv run python research/stack/r8/blind_rules9.py 5000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
from levels_rule_and_final import levels_landing

def main():
    Q = int(sys.argv[1]); qs = list(primerange(121, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']; cnt = {n: [0, 0] for n in names}; ex = []
    for q in qs:
        ps = list(primerange(2, q + 1)); bounds, lands = levels_landing(q, ps)
        E1 = [L for k, b, bk, gk, L in lands if k == 1]
        if not E1: continue
        E1 = E1[0]; p = ps[-2]
        base = []; P = 1
        for x in ps:
            if P * x <= q // 2: P *= x; base.append(x)
            else: break
        def enter(step):
            k = 1
            while E1 + k * step <= q: k += 1
            return E1 + k * step
        Ds = {'L1': enter(6 * q), 'L2': enter(12 * q), 'L3': enter(2 * P * q), 'L4': enter(6 * p), 'L5': enter(6 * (q - p)) , 'L6': enter(12 * (q - p))}
        for n, E in Ds.items():
            inwin = q < E and E + 2 <= q * q; cnt[n][1] += inwin
            if inwin and sv[E] and sv[E + 2]: cnt[n][0] += 1
        if q in (499, 1999, 4999): ex.append((q, E1, Ds))
    out = [__doc__.strip(), "", f"machines {len(qs)} (121 to {Q}): rule: twin / in window"]
    for n in names: out.append(f"   {n} {cnt[n][0]:>4} / {cnt[n][1]:>4}")
    out += [f"   {e}" for e in ex]
    Path("research/stack/r8/results_blind_rules9.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

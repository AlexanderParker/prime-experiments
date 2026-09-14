"""Phase 2 by fields: the high-gear line in reach from the primorial spiral's landing E, and which
field takes each gear, field by field in the machine's order.

For each machine q (11 to the given bound), each high gear h (sqrt q < h <= q) and each direction,
the landing L = E + 6 d h (window only). The gears are consulted in order 5, 7, 11, ...: gear g
takes h iff g | L or g | L + 2 (the two forbidden classes). Stages recorded:
  reach      high gears in reach (both directions)
  after 5    those the field higher:5 leaves
  after 5,7  those higher:5 and higher:7 leave
  after sub  those every gear <= sqrt q leaves (the sub-machine's survivors)
  pass       those every gear <= q leaves (twins)
Also: for survivors of the sub-machine that a high gear g > sqrt q takes, the order of the struck
member (with g > sqrt q and L < q^2 the cofactor is below q^(3/2)).

usage: uv run python research/stack/r8/phase2_fields.py 2000
"""
import sys
import numpy as np
from sympy import primerange, factorint
from pathlib import Path

def main():
    Q = int(sys.argv[1])
    qs = list(primerange(11, Q + 1))
    N = Q * Q + 10
    sieve = np.ones(N + 1, dtype=bool); sieve[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]: sieve[i * i::i] = False
    def alt(xs): return sum(g if i % 2 == 0 else -g for i, g in enumerate(xs))
    out = [__doc__.strip(), ""]
    out.append(f"{'q':>5} {'E':>8} {'reach':>5} {'aft5':>5} {'aft57':>5} {'aftsub':>6} {'pass':>4}  first pass (h, dir)  high-field kills of sub survivors by order")
    tot = dict(reach=0, a5=0, a57=0, sub=0, ps=0)
    hi_order = {}
    zero_sub = []; zero_pass = []
    sub_taker = {}   # which field takes a sub-survivor (g > sqrt q)
    for q in qs:
        primes = list(primerange(2, q + 1)); base = []; P = 1
        for p in primes:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
        gears = [p for p in primes if p >= 5]
        r = int(q ** 0.5)
        sub = [g for g in gears if g <= r]
        high = [g for g in gears if g > r]
        c = dict(reach=0, a5=0, a57=0, sub=0, ps=0); first = None; ho = {}
        for d in (1, -1):
            for h in high:
                L = E + 6 * d * h
                if not (q < L and L + 2 <= q * q): continue
                c['reach'] += 1
                def takes(g): return L % g == 0 or (L + 2) % g == 0
                if takes(5): continue
                c['a5'] += 1
                if takes(7): continue
                c['a57'] += 1
                if any(takes(g) for g in sub if g > 7): continue
                c['sub'] += 1
                taker = next((g for g in high if g != h and takes(g)), None)
                if taker is None:
                    # h itself never divides L or L+2 (L = E + 6dh, E not 0 or -2 mod h unless h | E) -- check
                    if takes(h):
                        taker = h
                if taker is None:
                    assert sieve[L] and sieve[L + 2], (q, h, d, L)
                    c['ps'] += 1
                    if first is None: first = (h, '+' if d > 0 else '-')
                else:
                    m = L if L % taker == 0 else L + 2
                    j = sum(factorint(m).values())
                    ho[j] = ho.get(j, 0) + 1; hi_order[j] = hi_order.get(j, 0) + 1
                    sub_taker[taker] = sub_taker.get(taker, 0) + 1
        for k in tot: tot[k] += c[k]
        if c['sub'] == 0: zero_sub.append(q)
        if c['ps'] == 0: zero_pass.append(q)
        out.append(f"{q:>5} {E:>8} {c['reach']:>5} {c['a5']:>5} {c['a57']:>5} {c['sub']:>6} {c['ps']:>4}  {str(first):<20} {dict(sorted(ho.items()))}")
    out.append("")
    out.append(f"totals: reach {tot['reach']}, after 5 {tot['a5']}, after 5,7 {tot['a57']}, after sub-machine {tot['sub']}, pass {tot['ps']}")
    out.append(f"high-field kills of sub-machine survivors by order of the struck member: {dict(sorted(hi_order.items()))}")
    out.append(f"machines with no sub-machine survivor: {zero_sub}")
    out.append(f"machines with no passing high gear: {zero_pass}")
    top = sorted(sub_taker.items(), key=lambda kv: -kv[1])[:12]
    out.append(f"high fields taking sub-machine survivors, top 12 (gear: count): {top}")
    Path("research/stack/r8/results_phase2_fields.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[-6:]))

if __name__ == "__main__":
    main()

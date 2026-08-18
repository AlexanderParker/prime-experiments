"""Round 18, streaming version: repetition spectrum of the gap word."""
from math import prod
import numpy as np
from split_gap_law import primes

def scan(y, chunk=50_000_000):
    gears = primes(5, y)
    P = prod(gears)
    F = 0
    runmax = 0
    runmax_g = 0
    viol3 = viol5 = 0
    runhist = {}
    carry = None
    cur_g = None
    cur_len = 0
    a = 0
    def close_run(g, ln):
        nonlocal runmax, runmax_g, viol3, viol5
        runhist[ln] = runhist.get(ln, 0) + 1
        if ln > runmax:
            runmax, runmax_g = ln, g
        if ln >= 3 and g % 5 != 0:
            viol3 += 1
        if ln >= 5 and g % 35 != 0:
            viol5 += 1
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed).astype(np.int64) + a
        if carry is not None:
            o = np.concatenate(([carry], o))
        d = np.diff(o)
        if len(d):
            F = max(F, int(d.max()))
            ch = np.flatnonzero(np.diff(d) != 0)
            st = np.concatenate(([0], ch + 1))
            en = np.concatenate((ch, [len(d) - 1]))
            for s0, e0 in zip(st, en):
                g = int(d[s0]); ln = int(e0 - s0 + 1)
                if cur_g == g:
                    cur_len += ln
                else:
                    if cur_g is not None:
                        close_run(cur_g, cur_len)
                    cur_g, cur_len = g, ln
        carry = int(o[-1])
        a += S
    if cur_g is not None:
        close_run(cur_g, cur_len)
    return F, runmax, runmax_g, viol3, viol5, runhist

print(f"  {'y':>3} {'F':>4} {'F/y^2':>7} {'maxrun':>7} {'g':>4} {'5|g':>5} "
      f"{'viol(>=3,5!|g)':>15} {'viol(>=5,35!|g)':>16}")
for y in (13, 17, 19, 23, 29, 31):
    F, rm, g, v3, v5, hist = scan(y)
    print(f"  {y:>3} {F:>4} {F/y**2:>7.4f} {rm:>7} {g:>4} {str(g%5==0):>5} "
          f"{v3:>15} {v5:>16}")
    print(f"        equal-run length histogram: "
          f"{dict(sorted(hist.items())[:6])}")

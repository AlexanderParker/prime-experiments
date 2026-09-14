"""Loop, iteration 3: pick-up walks with fixed period rules (no search, no primality).

Rule C (slip inverse): descending over the gears above the base; at the step for gear g the
mirror is {base, g}; the periods k are chosen so that the PREVIOUS gear's phase lands at the
centre of its open arc, floor(prev/2): k = (centre - phase) * slip^{-1} mod prev, where slip =
2 P g mod prev (k in 1..prev; if the slip is 0 mod prev, k = 1); direction always up, or
alternating.  The column may leave the window; a final reduction is not applied.
Rule D (base-centre): same but the target is chosen so that ALL base gears are irrelevant (they
are carried) and the previous TWO gears are centred by solving for k modulo prev * prev2 (CRT),
when that k keeps the column below q^2.
Measured: machines 11 to 2000; landing in the window, twin, number of gears on a tooth at the
landing.  usage: uv run python research/stack/r8/pickup_walk_rules.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def inv(a, m):
    a %= m
    return pow(a, -1, m) if a and m > 1 and __import__('math').gcd(a, m) == 1 else None

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = 4 * Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    out = [__doc__.strip(), ""]
    for rule in ('C up', 'C alt', 'D up', 'D alt'):
        inwin = twin = 0; bad = []; ex = []
        for q in qs:
            ps = list(primerange(2, q + 1)); base = []; P = 1
            for p in ps:
                if P * p <= q // 2: P *= p; base.append(p)
                else: break
            gears = [p for p in ps if p not in base][::-1]
            n = -1; sign = 1
            for i, g in enumerate(gears):
                d = sign if rule.endswith('alt') else 1
                k = 1
                if i >= 1:
                    prev = gears[i - 1]; s = (2 * P * g * d) % prev; ph = n % prev
                    target = prev // 2
                    si = inv(s, prev)
                    if si is not None: k = ((target - ph) * si) % prev or prev
                    if rule.startswith('D') and i >= 2:
                        prev2 = gears[i - 2]; s2 = (2 * P * g * d) % prev2; ph2 = n % prev2; t2 = prev2 // 2
                        si2 = inv(s2, prev2)
                        if si is not None and si2 is not None:
                            # CRT: k = k1 mod prev, k = k2 mod prev2
                            k1 = ((target - ph) * si) % prev; k2 = ((t2 - ph2) * si2) % prev2
                            m1 = inv(prev, prev2)
                            k = (k1 + prev * (((k2 - k1) * m1) % prev2)) % (prev * prev2) or prev * prev2
                m = n + 2 * k * P * g * d
                if m > q * q: k = 1; m = n + 2 * P * g * d   # keep it in range when the rule overshoots
                n = m; sign = -sign
            ok = q < n <= q * q - 2
            inwin += ok
            if ok and sv[n] and sv[n + 2]: twin += 1
            teeth = [h for h in ps if h >= 5 and n % h in (0, h - 2)]
            if q in (31, 101, 499, 1999): ex.append((q, n, ok, len(teeth), teeth[:5]))
        out.append(f"rule {rule}: landing in the window {inwin} of {len(qs)}, twin {twin}; samples (q, landing, in window, gears on a tooth) {ex}")
    Path("research/stack/r8/results_pickup_walk_rules.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()

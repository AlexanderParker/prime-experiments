"""Round 20 lateral: THE DEPTH-SUM IDENTITY.

For the machine M with gears q in {5..y}, period P = prod q, let

    N2(g) = #{r mod P : r and r+g both open}  =  prod_q c_q(g)     (CRT, exact;
            c_q(g) the round-18 closed form: q-2 / q-3 / q-4)

and let W_j(g) = #{cyclic windows of j consecutive gaps summing to g}
(each opening starts exactly one j-window, so sum_g W_j(g) = #openings).

IDENTITY:  for every g >= 1,   sum_{j>=1} W_j(g)  =  N2(g).

Proof (one line): every ordered pair of openings (r, r+g) is the endpoint pair
of exactly one window - the one spanning the j gaps between them (j <= g since
gaps are >= 1); conversely every j-window with sum g is such a pair.

Consequence: the closed-form pair correlation is an exact SUM RULE over the
whole window-sum family {W_j} - the spectrum F_j = max{g : W_j(g) > 0} lives
inside an object with a closed form. In particular W_j(g) <= prod_q c_q(g)
for every depth j, from CRT arithmetic alone, no period scan.

Usage: python depth_identity.py [y ...]   (default 11 13 17 19 23)
Writes research/data/depth_identity_<y>.csv with columns g, N2, W_1..W_G.
"""
import sys
from math import prod
import numpy as np

G = 64  # verify identity for g = 1..G (j <= g <= G windows suffice: gaps >= 1)

def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))]

def c_q(q, g):
    u = pow(6, -1, q)
    if g % q == 0:
        return q - 2
    if g % q in ((2 * u) % q, (-2 * u) % q):
        return q - 3
    return q - 4

def c_q_brute(q, g):
    u = pow(6, -1, q)
    t = {u % q, (-u) % q}
    return sum(1 for r in range(q)
               if r not in t and (r + g) % q not in t)

def openings(y, chunk=40_000_000):
    """Yield opening positions over one full period, chunked."""
    gears = primes(5, y)
    P = prod(gears)
    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        yield np.flatnonzero(~killed).astype(np.int64) + a
        a += S

def run(y):
    gears = primes(5, y)
    P = prod(gears)
    # gap array over the full period, cyclic
    gaps_chunks = []
    first = last = None
    nopen = 0
    for o in openings(y):
        if o.size == 0:
            continue
        if first is None:
            first = o[0]
        else:
            gaps_chunks.append(np.array([o[0] - last], np.int64))
        if o.size > 1:
            gaps_chunks.append(np.diff(o))
        last = o[-1]
        nopen += o.size
    gaps_chunks.append(np.array([P - last + first], np.int64))  # wrap gap
    d = np.concatenate(gaps_chunks)
    assert d.sum() == P and d.size == nopen
    # W_j(g) for j = 1..G, g = 1..G, cyclic (append G leading gaps)
    dd = np.concatenate((d, d[:G])).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(dd)))
    W = np.zeros((G + 1, G + 1), np.int64)   # W[j][g]
    n = d.size
    for j in range(1, G + 1):
        s = cs[j:j + n] - cs[:n]
        s = s[s <= G]
        W[j] += np.bincount(s, minlength=G + 1)
    # the identity
    bad = 0
    for g in range(1, G + 1):
        N2 = prod(c_q(q, g) for q in gears)
        N2b = prod(c_q_brute(q, g) for q in gears)
        assert N2 == N2b, (y, g)          # closed form == brute force
        tot = int(W[1:, g].sum())
        if tot != N2:
            bad += 1
            print(f"  MISMATCH y={y} g={g}: sum_j W_j = {tot} vs N2 = {N2}")
    depthmax = max(j for j in range(1, G + 1) if W[j].sum() > 0)
    print(f"machine {y}: period {P}, openings {nopen}, "
          f"identity checked g=1..{G}: {G - bad} exact, {bad} mismatches"
          + ("  <-- FAIL" if bad else "   (all exact)"))
    assert bad == 0
    # persist for the renewal analysis
    import csv as _csv, os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", f"depth_identity_{y}.csv")
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["g", "N2"] + [f"W{j}" for j in range(1, G + 1)])
        for g in range(1, G + 1):
            N2 = prod(c_q(q, g) for q in gears)
            w.writerow([g, N2] + [int(W[j, g]) for j in range(1, G + 1)])
    return True

if __name__ == "__main__":
    ys = [int(a) for a in sys.argv[1:]] or [11, 13, 17, 19, 23]
    for y in ys:
        run(y)
    print("DEPTH-SUM IDENTITY: all machines exact.")

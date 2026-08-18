"""Round 18 lateral: THE EXPOSED-SET AUTOCORRELATION - a construct for the
"arithmetic selection" the search wrote off as "no smooth law, only histogram".

NEW OBJECT. Every part of the machine so far has been measured on its own:
gears, teeth, slots, gaps, words. The unmeasured relationship is
GEAR x LAG. For gear q let A_q = Z_q minus its two teeth {u, -u} be the
exposed set (|A_q| = q-2). Define its AUTOCORRELATION at lag g:

    c_q(g) = |{ r in A_q : r + g in A_q }|

i.e. how many phases keep BOTH ends of a lag-g pair exposed to gear q.

CLOSED FORM (derived, then verified): the union {u,-u} u ({u,-u} - g) has size
2 if q | g (the two teeth map onto themselves), 3 if g = +-2u mod q (one
coincidence - exactly the LITERAL-LINK condition of the padding work), and 4
otherwise. Hence

    c_q(g) = q-2   if q | g          (same tooth)
           = q-3   if g = +-2u_q     (opposite teeth - the literal link!)
           = q-4   otherwise         (generic)

So the three cases of the autocorrelation ARE the three tooth relationships.
The SELECTION FACTOR of a lag is the product over the machine's gears

    sigma(g) = prod_q  c_q(g) / (q-2),

which is 1 when every gear divides g and collapses fast otherwise - dominated
by the SMALL gears (q=5 contributes 1, 2/3 or 1/3; q=7 contributes 1, 4/5 or
3/5; large q contribute ~1). This is a closed-form arithmetic function, not
noise: it predicts which gap values are common and which are suppressed.

TEST: compare sigma(g) against measured gap-value histograms of real machines.
"""
from math import prod
import numpy as np
from split_gap_law import primes

def c_q(q, g):
    u = pow(6, -1, q)
    if g % q == 0:
        return q - 2
    if g % q in ((2 * u) % q, (-2 * u) % q):
        return q - 3
    return q - 4

def sigma(gears, g):
    s = 1.0
    for q in gears:
        s *= c_q(q, g) / (q - 2)
    return s

def verify_closed_form():
    bad = 0
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 31):
        u = pow(6, -1, q)
        A = set(range(q)) - {u % q, (-u) % q}
        for g in range(q * 2):
            brute = sum(1 for r in A if (r + g) % q in A)
            if brute != c_q(q, g):
                bad += 1
    return bad

def gap_hist(y, chunk=50_000_000, cap=200):
    gears = primes(5, y)
    P = prod(gears)
    cnt = np.zeros(cap + 1, np.int64)
    carry = None
    a = 0
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
        d = d[d <= cap]
        cnt += np.bincount(d, minlength=cap + 1)
        carry = int(o[-1])
        a += S
    return cnt, gears

print("=" * 76)
print("PART 1: the closed form")
print(f"  c_q(g) = q-2 if q|g;  q-3 if g = +-2u_q (the literal-link lag);"
      f"  q-4 else")
print(f"  brute-force verification over gears 5..31, all lags: "
      f"{verify_closed_form()} mismatches")
print("  gear 5 (u=1, 2u=2): c=3 if 5|g, 2 if g=+-2 mod 5, 1 otherwise")
print("  -> lags = +-1 mod 5 are SUPPRESSED BY A FACTOR 3 by gear 5 alone.")

print("=" * 76)
print("PART 2: measured gap histograms vs the selection factor")
for y in (19, 23):
    cnt, gears = gap_hist(y)
    tot = cnt.sum()
    print(f"  --- machine {y} (gears {gears}) ---")
    print(f"  {'g':>4} {'count':>9} {'g mod 5':>8} {'c_5':>4} {'c_7':>4} "
          f"{'sigma':>8} {'count/sigma':>12}")
    rows = []
    for g in range(20, 46):
        s = sigma(gears, g)
        rows.append((g, int(cnt[g]), g % 5, c_q(5, g), c_q(7, g), s,
                     cnt[g] / s if s else float('nan')))
    for g, c, gm5, c5, c7, s, ratio in rows:
        flag = "  <- ZERO" if c == 0 else ""
        print(f"  {g:>4} {c:>9} {gm5:>8} {c5:>4} {c7:>4} {s:>8.4f} "
              f"{ratio:>12.1f}{flag}")

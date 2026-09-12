"""The walk through the fields (owner, 2026-09-13): which fields does each step of the walk pass
through, and what does each field do to the landing.

Walk from home about the axis k M (M = 6 for S = {2, 3}): landing (2kM - 1, 2kM + 1) = with
M = 6 the column 2k, i.e. the pair (12k - 1, 12k + 1). The landing zone for machine q is the run
of k with q < 12k - 1 and 12k + 1 <= q^2. On that zone every field of the explorer restricts to
a set of painted k:
  multiples of h      k = -+12^-1 (mod h): two teeth per period h            (h a gear of the machine)
  squares             12k + 1 = g^2 for a gear g (right member only; 12k - 1 is never a square)
  higher:h            the landing member is h times a number with no smaller gear factor
  products:j          the landing member has exactly j prime factors
  blind for the zone  gears with no paint at all on the zone
Per machine: for each field, the painted k in the zone (count), exact per-period facts, which
fields are contained in which (higher:h and products:j inside the multiples rows), the k the
rule lands on and the fields it had to step past, and the joint fact: the rows of the multiples
fields are independent modulo the product of the gears (exact CRT count of survivors per full
period), the zone being one phase of that period.
Usage: uv run python walk_fields.py q [q ...]
"""
import sys
from math import isqrt
from sympy import primerange, isprime, factorint


def run(q):
    gears = list(primerange(5, q + 1))
    ks = [k for k in range(1, q * q // 12 + 2) if 12 * k - 1 > q and 12 * k + 1 <= q * q]
    print(f"\n== machine {q}: landing zone k in [{ks[0]}, {ks[-1]}] ({len(ks)} columns of the m-line)")
    # multiples rows
    print("field 'multiples of h' on the zone: h | teeth on k (classes mod h) | painted k in the zone | share | exact share per period 2/h")
    tot_painted = set(); rows = {}
    for h in gears:
        inv = pow(12, -1, h); t = sorted({inv % h, (-inv) % h})
        painted = [k for k in ks if (12 * k - 1) % h == 0 or (12 * k + 1) % h == 0]
        rows[h] = set(painted); tot_painted |= set(painted)
        if h <= 23 or h == gears[-1]:
            print(f"   {h} | {t} | {len(painted)} | {len(painted)/len(ks):.3f} | {2/h:.3f}")
    survivors = [k for k in ks if k not in tot_painted]
    print(f"   all rows together: painted {len(tot_painted)} of {len(ks)}; unpainted (the landings that are twins) {len(survivors)}; first {survivors[:5]}")
    # squares
    sq = [k for k in ks if isqrt(12 * k + 1) ** 2 == 12 * k + 1]
    print(f"field 'squares' on the zone: {len(sq)} k (12k + 1 = g^2): {[(k, isqrt(12*k+1)) for k in sq[:6]]}; each lies in the row of its g (contained in 'multiples')")
    # higher:h and products:j on the zone, containment
    hi = {}; pr = {}
    for k in ks:
        for n in (12 * k - 1, 12 * k + 1):
            if isprime(n): continue
            f = factorint(n); s = min(f); j = sum(f.values())
            hi.setdefault(s, set()).add(k); pr.setdefault(j, set()).add(k)
    cont = all(hi[h] <= rows[h] for h in hi if h in rows)
    print(f"field 'higher:h' on the zone (smallest gear h): rows {sorted(hi)[:8]}{' ...' if len(hi) > 8 else ''}; each contained in the multiples row of its h: {cont}; together they are exactly the painted set: {set().union(*hi.values()) == tot_painted}")
    print(f"field 'products:j' on the zone: " + ", ".join(f"j={j}: {len(v)}" for j, v in sorted(pr.items())) + f"; union = painted set: {set().union(*pr.values()) == tot_painted}")
    # blind gears for the zone (no paint at all)
    blind = [h for h in gears if not rows[h]]
    print(f"gears with no paint on the zone: {blind[:10]}{' ...' if len(blind) > 10 else ''} ({len(blind)} of {len(gears)}; a gear paints nothing when both its teeth miss the zone, only possible when h exceeds the zone length {len(ks)})")
    # the rule's landing and what it stepped past
    k0 = next(k for k in ks if k not in tot_painted)
    passed = [(k, [h for h in gears if k in rows[h]]) for k in ks if k < k0]
    print(f"the rule lands on k = {k0} ({12*k0-1}, {12*k0+1}); it stepped past {len(passed)} painted k: " + "; ".join(f"{k}: rows {hs}" for k, hs in passed[:6]) + (" ..." if len(passed) > 6 else ""))
    # exact CRT fact
    P = 1
    for h in gears: P *= h
    surv = 1
    for h in gears: surv *= (h - 2)
    print(f"exact joint fact: the rows are independent modulo the product of the gears: per full period of {P if P < 10**30 else 'about 10^%d' % (len(str(P))-1)} k there are exactly prod (h - 2) = {surv if surv < 10**30 else 'about 10^%d' % (len(str(surv))-1)} unpainted k; the zone is {len(ks)} k long, one phase of that period; unpainted found in the zone {len(survivors)} against the period share {len(ks) * surv / P:.1f}")


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))

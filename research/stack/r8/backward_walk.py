"""The walk backwards (owner, 2026-09-13): start from a found twin in the window (not a gear
pair) and flip to a pair of gears; find the rules that would make a walk possible.

A flip about the axis A/2 sends the column n to A - n - 2. It carries gear h (the openness of
the two columns to h agree) iff h divides A. The flip from the window twin n to the gear pair
(g, g+2) has A = n + g + 2, so it carries exactly the gears dividing n + g + 2, whatever axis
family it is filed under. Anchors = columns whose openness is known without search: the gear
pairs (g, g+2) with g + 2 <= q (both prime by construction, struck by g and g + 2 themselves,
open to every other gear) and the home column (-1, 1), open to every gear.
A gear h's openness at n is certified by an anchor m iff h divides n + m + 2 (n is the mirror of
m modulo h). Per machine:
  (a) window twin x gear pair: the gears carried by the one flip between them;
  (b) per gear h: the open residue classes mod h (n with n, n+2 not 0 mod h) that some anchor's
      mirror covers, and the classes no anchor covers;
  (c) per window twin: for each gear, an anchor certifying it, and whether every gear is
      certified (a full backward walk of one flip per gear).
Usage: uv run python backward_walk.py q [q ...]
"""
import sys
from sympy import primerange, isprime


def run(q):
    gears = list(primerange(5, q + 1))
    pairs = [g for g in primerange(5, q - 1) if isprime(g + 2) and g + 2 <= q]
    anchors = [-1] + pairs
    twins = [n for n in range(q + 1, q * q) if n % 6 == 5 and isprime(n) and isprime(n + 2)]
    print(f"\n== machine {q}: gears 5..{q}, gear pairs {[(g, g + 2) for g in pairs]}, home (-1, 1); window twins {len(twins)}")
    # (a)
    if q <= 13:
        print("(a) window twin -> gear pair: axis A = n + g + 2, carried gears (those dividing A)")
        for n in twins:
            print(f"   ({n}, {n+2}): " + "; ".join(f"-> ({g}, {g+2}) A = {n+g+2}, carries {[h for h in [2,3]+gears if (n+g+2) % h == 0]}" for g in pairs))
    # (b)
    print("(b) per gear h: open classes mod h covered by the anchors' mirrors / all open classes; missing classes")
    cover = {}
    for h in gears:
        opencls = [c for c in range(h) if c % h and (c + 2) % h]
        cov = set((-m - 2) % h for m in anchors if m % h and (m + 2) % h)  # anchors open to h only
        cover[h] = cov
        miss = [c for c in opencls if c not in cov]
        print(f"   h = {h}: {len(cov & set(opencls))}/{len(opencls)}" + (f"; missing {miss[:12]}{' ...' if len(miss) > 12 else ''}" if miss else " (complete)"))
    # (c)
    full = 0; partial = []
    for n in twins:
        cert = {}
        for h in gears:
            anc = [m for m in anchors if (n + m + 2) % h == 0 and m % h and (m + 2) % h]
            cert[h] = anc[0] if anc else None
        if all(v is not None for v in cert.values()): full += 1
        else: partial.append((n, [h for h in gears if cert[h] is None]))
    print(f"(c) window twins certified for every gear by one flip per gear from the anchors: {full} of {len(twins)}")
    if partial: print("    uncertified (twin: gears with no anchor): " + "; ".join(f"{n}: {hs}" for n, hs in partial[:8]) + (" ..." if len(partial) > 8 else ""))
    # which gears are the bottleneck
    bad = {}
    for n, hs in partial:
        for h in hs: bad[h] = bad.get(h, 0) + 1
    if bad: print(f"    gears lacking anchors, count of twins affected: {sorted(bad.items())[:10]}")


if __name__ == "__main__":
    for a in sys.argv[1:]: run(int(a))

"""Square anchors (family A) as a certification family, built and measured exactly as specified.

Machine q (q prime): gears are the primes 5 <= h <= q (2 and 3 implicit, never counted).
Window: integers n with q < n <= q*q.  Column = pair (n, n+2) with n = 5 mod 6, named by n.
A column is open to gear h iff h divides neither n nor n+2; a window twin is a column with
q < n and n+2 <= q*q open to every gear (equivalently n and n+2 both prime).

Anchor = a column a with a set K(a) of gears it is known open to by construction.
Certification: gear h is certified at column n by anchor a iff h in K(a) and
n = a mod h or n = -a-2 mod h (a mirror flip about an axis divisible by h maps a to -a-2 mod h
and keeps openness to h; two flips compose to a translation by a multiple of h).
A column is fully certified iff every gear of the machine is certified at it by some anchor.

Base anchors (the "+ base" figures): home a = -1 with K = all gears; and every gear pair a = g
with g and g+2 both gears, K = all gears except g and g+2.

Family A (square anchors): for every gear g, the columns g*g - 2 - 6j and g*g - 2 + 6j for
j = 1 .. g-1, excluding the columns where g divides a member, with K(a) = {g}.  g cannot divide
two numbers within 6(g-1) of g*g other than the multiples themselves, so these are known open to
g; the excluded j are taken from that claim (j = -2/6 mod g on the minus side, j = 2/6 mod g on
the plus side, the upper member never being hit) and the exclusion is then re-verified directly.
Anchors with n < 5 are dropped.

Usage: uv run python research/stack/r8/anchors_squares.py
Writes research/stack/r8/results_anchors_squares.md and prints the same table.
"""
import os
from statistics import median

from sympy import primerange

MACHINES = [31, 101, 211, 401, 1009]


def prime_flags(limit):
    """bytearray p with p[i] = 1 iff i is prime, for i <= limit (sympy primerange)."""
    flags = bytearray(limit + 1)
    for p in primerange(2, limit + 1):
        flags[p] = 1
    return flags


def build_family_a(gears):
    """The family A anchors as (a, g) pairs, g the single gear of K(a)."""
    anchors = []
    for g in gears:
        inv6 = pow(6, -1, g)
        j_minus = (-2 * inv6) % g          # the j where g divides the lower member, minus side
        j_plus = (2 * inv6) % g            # ... and on the plus side
        for j in range(1, g):
            if j != j_minus:
                n = g * g - 2 - 6 * j
                if n >= 5:
                    anchors.append((n, g))
            if j != j_plus:
                n = g * g - 2 + 6 * j
                if n >= 5:
                    anchors.append((n, g))
    return anchors


def run(q):
    gears = list(primerange(5, q + 1))
    gset = set(gears)
    ng = len(gears)

    # ---- 1. family A, and direct verification of every (anchor, gear) claim ------------------
    anchors_a = build_family_a(gears)
    failures = 0
    first_failure = None
    for a, g in anchors_a:
        if a % g == 0 or (a + 2) % g == 0:
            failures += 1
            if first_failure is None:
                first_failure = (a, g)
    if failures:
        return {"q": q, "failures": failures, "first_failure": first_failure}

    # ---- 2. coverage of each gear by family A alone ------------------------------------------
    covered_a = {h: set() for h in gears}
    for a, g in anchors_a:
        covered_a[g].add(a % g)
        covered_a[g].add((-a - 2) % g)
    full_gears = [h for h in gears if len(covered_a[h]) == h - 2]
    first_uncovered = next((h for h in gears if len(covered_a[h]) != h - 2), None)

    # ---- base anchors -------------------------------------------------------------------------
    base = [(-1, set(gears))]
    for g in gears:
        if g + 2 in gset:
            base.append((g, gset - {g, g + 2}))
    covered_base = {h: set() for h in gears}
    base_hit = {h: {} for h in gears}      # gear -> residue -> bitmask of base anchors
    for b, (a, K) in enumerate(base):
        for h in K:
            for r in (a % h, (-a - 2) % h):
                covered_base[h].add(r)
                base_hit[h][r] = base_hit[h].get(r, 0) | (1 << b)
    covered_all = {h: covered_a[h] | covered_base[h] for h in gears}

    # ---- the window's columns -----------------------------------------------------------------
    lo, hi = q + 1, q * q - 2            # q < n, n+2 <= q*q
    start = lo + ((5 - lo) % 6)
    ncols = (hi - start) // 6 + 1
    flags = prime_flags(q * q + 6 * q)   # covers every column and every anchor member

    def certified_mask(cert):
        """bytearray over column index: 1 iff every gear is certified there by residue set cert."""
        mask = bytearray([1]) * ncols
        for h in gears:
            inv6 = pow(6, -1, h)
            for r in range(h):
                if r in cert[h]:
                    continue
                i0 = ((r - start) * inv6) % h
                if i0 < ncols:
                    mask[i0::h] = bytearray(len(range(i0, ncols, h)))
        return mask

    mask_a = certified_mask(covered_a)
    mask_all = certified_mask(covered_all)

    twins = [i for i in range(ncols) if flags[start + 6 * i] and flags[start + 6 * i + 2]]
    total_twins = len(twins)
    twins_a = sum(1 for i in twins if mask_a[i])
    twins_all = sum(1 for i in twins if mask_all[i])

    # ---- 4. certified columns that are not open to every gear ---------------------------------
    struck_certified = sum(mask_all) - twins_all

    # ---- 5. walk length: greedy set cover over family A + base --------------------------------
    walks = []
    full = (1 << ng) - 1
    nb = len(base)
    for i in twins:
        if not mask_all[i]:
            continue
        n = start + 6 * i
        masks = [0] * nb
        for gi, h in enumerate(gears):
            m = base_hit[h].get(n % h, 0)
            while m:
                low = m & -m
                masks[low.bit_length() - 1] |= 1 << gi
                m ^= low
        uncov = full
        picks = 0
        while True:
            best, best_mask = 0, 0
            for m in masks:
                c = (m & uncov).bit_count()
                if c > best:
                    best, best_mask = c, m & uncov
            if best < 2:
                break
            uncov &= ~best_mask
            picks += 1
        # every gear still uncovered is certified by a single anchor of its own (family A,
        # or a base anchor covering just it), so the greedy tail is one anchor per gear
        walks.append(picks + uncov.bit_count())

    # ---- 6. family A anchors that are themselves twins -----------------------------------------
    anchor_twins = sum(1 for a, _ in anchors_a if flags[a] and flags[a + 2])

    return {
        "q": q,
        "anchors": len(anchors_a),
        "failures": failures,
        "first_failure": None,
        "full_gears": full_gears,
        "first_uncovered": first_uncovered,
        "twins_a": twins_a,
        "twins_all": twins_all,
        "total_twins": total_twins,
        "struck_certified": struck_certified,
        "walk": (min(walks), median(walks), max(walks)) if walks else None,
        "anchor_twins": anchor_twins,
        "anchor_nontwins": len(anchors_a) - anchor_twins,
        "ngears": ng,
    }


HEADER = (
    "| q | anchors | verification failures | gears fully covered | first uncovered gear | "
    "twins certified (A alone) | twins certified (A + base) | total window twins | "
    "struck columns certified | walk length min/median/max | anchors that are twins / not twins |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def main():
    rows = []
    for q in MACHINES:
        res = run(q)
        if res["failures"]:
            raise SystemExit("q=%d: %d verification failures, first %r"
                             % (q, res["failures"], res["first_failure"]))
        cov = ("all %d" % res["ngears"]) if len(res["full_gears"]) == res["ngears"] \
            else ("%d of %d" % (len(res["full_gears"]), res["ngears"]))
        mn, md, mx = res["walk"]
        md = int(md) if float(md).is_integer() else md
        rows.append("| %d | %d | %d | %s | %s | %d | %d | %d | %d | %s/%s/%s | %d / %d |"
                    % (res["q"], res["anchors"], res["failures"], cov,
                       "none" if res["first_uncovered"] is None else str(res["first_uncovered"]),
                       res["twins_a"], res["twins_all"], res["total_twins"],
                       res["struck_certified"], mn, md, mx,
                       res["anchor_twins"], res["anchor_nontwins"]))
        print(rows[-1], flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_anchors_squares.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Square anchors (family A): coverage, certification, walk length\n\n")
        f.write(HEADER + "\n" + "\n".join(rows) + "\n")
    print("\n" + HEADER + "\n" + "\n".join(rows))
    print("\nwritten: " + out)


if __name__ == "__main__":
    main()

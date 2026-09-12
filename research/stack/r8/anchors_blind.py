"""Family B (blind-class) anchors: build, verify, and measure certification.

Definitions used here (self-contained):

  Machine q (q prime).  Gears = primes 5 <= h <= q (2 and 3 implicit, never counted).
  Window = integers n with q < n <= q*q.
  Column = a pair (n, n+2) with n = 5 mod 6, named by n.  A column is OPEN TO gear h
  iff h does not divide n and h does not divide n+2.  A window twin is a column with
  q < n and n+2 <= q*q open to every gear (both members prime).

  An ANCHOR is a column a with a set K(a) of gears it is known open to by construction.
  Base anchors: home a = -1, K = all gears; and every gear pair a = g with g, g+2 both
  gears (g >= 5), K = all gears except g and g+2.

  CERTIFICATION: gear h is certified at column n by anchor a iff h in K(a) and
  (n = a mod h or n = -a-2 mod h).  A column is FULLY CERTIFIED iff every gear is
  certified by some anchor.

  FAMILY B: for every gear g >= 11, c0 = g*g - 2 (the column whose right member is g*g).
  Anchors are c0 + 6i for i >= 1 with i = 5, 10, 12 or 17 (mod 35) and c0 + 6i + 2 <= q*q;
  K(a) = {5, 7}.  (From a square origin gear 5 never strikes offsets i = 0, 2 mod 5 and
  gear 7 never strikes i = 3, 5 mod 7; those four classes are the joint ones mod 35.)

Usage: uv run python research/stack/r8/anchors_blind.py
"""
import statistics
from sympy import primerange, isprime

MACHINES = [31, 101, 211, 401, 1009]
BLIND_CLASSES = (5, 10, 12, 17)
BLIND_MOD = 35


# ---------------------------------------------------------------- construction

def gears_of(q):
    return list(primerange(5, q + 1))


def base_anchors(gears):
    """[(a, K(a))] -- home plus every gear pair."""
    gset = set(gears)
    out = [(-1, frozenset(gears))]
    for g in gears:
        if g + 2 in gset:
            out.append((g, frozenset(h for h in gears if h != g and h != g + 2)))
    return out


def family_b_anchors(q, gears):
    """Distinct columns c0 + 6i from every square g*g, g a gear >= 11."""
    qq = q * q
    anchors = set()
    for g in gears:
        if g < 11:
            continue
        c0 = g * g - 2
        imax = (qq - g * g) // 6          # c0 + 6i + 2 = g*g + 6i <= q*q
        for r in BLIND_CLASSES:
            i = r
            while i <= imax:
                anchors.add(c0 + 6 * i)
                i += BLIND_MOD
    return sorted(anchors)


# ---------------------------------------------------------------- certification

def allowed_residues(gears, base, bset):
    """gear h -> set of residues r mod h that some anchor with h in K(a) certifies."""
    allowed = {h: set() for h in gears}
    for a, K in base:
        for h in K:
            allowed[h].add(a % h)
            allowed[h].add((-a - 2) % h)
    for a in bset:
        for h in (5, 7):
            if h in allowed:
                allowed[h].add(a % h)
                allowed[h].add((-a - 2) % h)
    return allowed


def b_only_residues(gears, bset):
    allowed = {h: set() for h in (5, 7) if h in gears}
    for a in bset:
        for h in allowed:
            allowed[h].add(a % h)
            allowed[h].add((-a - 2) % h)
    return allowed


def is_open(n, gears):
    return all(n % h and (n + 2) % h for h in gears)


# ---------------------------------------------------------------- walk length

def greedy_walk(n, gears, base, b_keys):
    """Greedy set cover: anchors needed to certify every gear at n."""
    cands = []
    for a, K in base:
        s = frozenset(h for h in K if (n - a) % h == 0 or (n + a + 2) % h == 0)
        if s:
            cands.append(s)
    seen = set()
    for (r5, r7) in b_keys:
        s = set()
        if (n - r5) % 5 == 0 or (n + r5 + 2) % 5 == 0:
            s.add(5)
        if (n - r7) % 7 == 0 or (n + r7 + 2) % 7 == 0:
            s.add(7)
        s = frozenset(h for h in s if h in gears)
        if s and s not in seen:
            seen.add(s)
            cands.append(s)
    uncovered = set(gears)
    steps = 0
    while uncovered:
        best = max(cands, key=lambda s: len(s & uncovered), default=frozenset())
        if len(best & uncovered) == 0:
            return None
        uncovered -= best
        steps += 1
    return steps


# ---------------------------------------------------------------- per machine

def run(q):
    gears = gears_of(q)
    gset = set(gears)
    qq = q * q
    base = base_anchors(gears)
    bset = family_b_anchors(q, gears)

    # 1. verification: every family B anchor really is open to 5 and 7
    failures = 0
    first_fail = None
    for a in bset:
        for h in (5, 7):
            if a % h == 0 or (a + 2) % h == 0:
                failures += 1
                if first_fail is None:
                    first_fail = (a, h)
    if failures:
        return {"q": q, "failures": failures, "first_fail": first_fail}

    # 2. coverage by family B alone
    bres = b_only_residues(gset, bset)
    cov5 = (len(bres[5]), 5 - 2) if 5 in bres else None
    cov7 = (len(bres[7]), 7 - 2) if 7 in bres else None

    # primes up to q*q + 2
    limit = qq + 3
    sieve = bytearray([1]) * limit
    sieve[0] = sieve[1] = 0
    p = 2
    while p * p < limit:
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(range(p * p, limit, p)))
        p += 1

    # window columns: n = 5 mod 6, q < n, n + 2 <= q*q
    start = q + 1
    start += (5 - start % 6) % 6
    columns = list(range(start, qq - 1, 6))

    allowed = allowed_residues(gset, base, bset)
    gears_desc = sorted(gears, reverse=True)

    twins = [n for n in columns if sieve[n] and sieve[n + 2]]
    twin_set = set(twins)

    # 3a. twins certified for gears 5 and 7 by family B alone
    b57 = 0
    for n in twins:
        if all(n % h in bres[h] for h in bres):
            b57 += 1

    # 3b / 4. fully certified columns (B + base) over the whole window
    certified = []
    for n in columns:
        ok = True
        for h in gears_desc:
            if n % h not in allowed[h]:
                ok = False
                break
        if ok:
            certified.append(n)

    cert_twins = [n for n in certified if n in twin_set]
    struck_certified = sum(1 for n in certified if not is_open(n, gears))

    # 3c. family B alone, every gear
    b_alone_full = 0
    if all(h in (5, 7) for h in gears):
        for n in twins:
            if all(n % h in bres[h] for h in gears):
                b_alone_full += 1

    # 5. walk lengths on the fully certified twins
    b_keys = sorted({(a % 5, a % 7) for a in bset})
    walks = [greedy_walk(n, gset, base, b_keys) for n in cert_twins]
    walks = [w for w in walks if w is not None]
    if walks:
        walk = "%d / %g / %d" % (min(walks), statistics.median(walks), max(walks))
    else:
        walk = "n/a"

    # 6. nature of the anchors themselves
    a_twin = sum(1 for a in bset if isprime(a) and isprime(a + 2))
    a_open = sum(1 for a in bset if is_open(a, gears))

    return {
        "q": q,
        "gears": len(gears),
        "anchors": len(bset),
        "failures": 0,
        "cov5": cov5,
        "cov7": cov7,
        "b57": b57,
        "b_alone_full": b_alone_full,
        "cert_twins": len(cert_twins),
        "twins": len(twins),
        "struck_certified": struck_certified,
        "walk": walk,
        "a_twin": a_twin,
        "a_not_twin": len(bset) - a_twin,
        "a_open": a_open,
        "a_struck": len(bset) - a_open,
    }


def main():
    rows = []
    for q in MACHINES:
        r = run(q)
        if r["failures"]:
            print("q = %d: VERIFICATION FAILED, first failing anchor %r" % (q, r["first_fail"]))
            return
        rows.append(r)
        print("done q = %d: %d anchors, %d/%d twins certified"
              % (q, r["anchors"], r["cert_twins"], r["twins"]))

    head = ("| q | anchors | verification failures | coverage of 5 | coverage of 7 | "
            "twins certified for 5 and 7 (B alone) | twins certified (B + base) | "
            "total window twins | struck columns certified | walk length min/median/max | "
            "anchors twins / not twins | anchors machine-open / struck |")
    sep = "|" + "---|" * 12
    lines = [head, sep]
    for r in rows:
        c5 = "%d/%d" % r["cov5"] if r["cov5"] else "-"
        c7 = "%d/%d" % r["cov7"] if r["cov7"] else "-"
        lines.append(
            "| %d | %d | %d | %s | %s | %d | %d | %d | %d | %s | %d / %d | %d / %d |"
            % (r["q"], r["anchors"], r["failures"], c5, c7, r["b57"], r["cert_twins"],
               r["twins"], r["struck_certified"], r["walk"], r["a_twin"], r["a_not_twin"],
               r["a_open"], r["a_struck"]))
    table = "\n".join(lines)
    print()
    print(table)

    b_alone = ", ".join("q=%d: %d" % (r["q"], r["b_alone_full"]) for r in rows)
    out = ("# Family B (blind-class) anchors\n\n"
           "Built by `research/stack/r8/anchors_blind.py`.\n\n"
           + table + "\n\n"
           "Window twins fully certified by family B alone (all gears): " + b_alone + ".\n")
    with open("research/stack/r8/results_anchors_blind.md", "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()

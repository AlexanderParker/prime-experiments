"""xm_family_cover.py -- the tooth family at the section by exact-cover search instead of
enumeration (for the cuts whose family is too large to enumerate: p = 37 has 5.4e8 members,
p = 41 has 1.6e10).

Question: is there a tooth vector (v_g)_{g in {5..p}}, 1 <= v_g <= (g-1)/2, such that the gears
with teeth +-v_g strike every column a+1 .. b-1 of the section at the cut p?  Gear g with tooth v
strikes column k iff k mod g in {v, g - v}.  Search: pick the uncovered column with the fewest
(gear, tooth) options among the gears not yet assigned, branch, recurse; capacity bound as in
r70.  Enumerates ALL killers when --all is given (count and first few), else stops at the first.
Gate: reproduces xm_family.py's counts 15 at p = 17 and 6,030 at p = 29.
"""
import sys
import time

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
NEXT = {PRIMES[i]: PRIMES[i + 1] for i in range(len(PRIMES) - 1)}


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def search(p, count_all, limit_examples=3):
    pn = NEXT[p]
    a = (p * p - 1) // 6
    b = (pn * pn - 1) // 6
    cols = list(range(a + 1, b))
    n = len(cols)
    full = (1 << n) - 1
    gears = gears_of(p)
    per = {}
    for g in gears:
        rows = []
        for v in range(1, (g - 1) // 2 + 1):
            m = 0
            for i, k in enumerate(cols):
                if (k % g) in (v, g - v):
                    m |= 1 << i
            rows.append((v, m))
        per[g] = rows
    cov = {g: {} for g in gears}
    for g in gears:
        for v, m in per[g]:
            mm = m
            while mm:
                bb = mm & -mm
                cov[g].setdefault(bb.bit_length() - 1, []).append((v, m))
                mm ^= bb
    found = [0]
    examples = []
    assigned = {}
    nodes = [0]

    def rec(unc, rem):
        if unc == 0:
            # every remaining gear is free: multiply by its tooth count
            mult = 1
            for g in rem:
                mult *= (g - 1) // 2
            found[0] += mult
            if len(examples) < limit_examples:
                ex = dict(assigned)
                for g in rem:
                    ex[g] = 1
                examples.append([ex[g] for g in gears])
            return not count_all
        nodes[0] += 1
        need = bin(unc).count("1")
        tot = 0
        for g in rem:
            best = 0
            for v, m in per[g]:
                c = bin(m & unc).count("1")
                if c > best:
                    best = c
            tot += best
            if tot >= need:
                break
        if tot < need:
            return False
        bestp, bestopts = None, None
        u = unc
        while u:
            bb = u & -u
            pos = bb.bit_length() - 1
            u ^= bb
            opts = [(g, v, m) for g in rem for (v, m) in cov[g].get(pos, ())]
            if not opts:
                return False
            if bestopts is None or len(opts) < len(bestopts):
                bestp, bestopts = pos, opts
                if len(opts) <= 1:
                    break
        for g, v, m in bestopts:
            assigned[g] = v
            stop = rec(unc & ~m, tuple(x for x in rem if x != g))
            del assigned[g]
            if stop:
                return True
        return False

    t0 = time.time()
    rec(full, tuple(gears))
    return {"p": p, "section": (a + 1, b - 1), "columns": n, "killers": found[0],
            "examples": examples, "nodes": nodes[0], "secs": time.time() - t0,
            "note": "killers counted with multiplicity over gears left free at a leaf; a leaf "
                    "can be reached by several branches when a column is struck by two assigned "
                    "gears, so with --all the count is an UPPER bound unless it equals the "
                    "enumeration's"}


def main():
    args = [v for v in sys.argv[1:] if not v.startswith("--")]
    count_all = "--all" in sys.argv
    ps = [int(v) for v in args] or [17, 29, 37, 41]
    for p in ps:
        r = search(p, count_all)
        print(f"p = {p}: section {r['section']} ({r['columns']} columns): killers "
              f"{'>= ' if not count_all else ''}{r['killers']}; examples {r['examples']}; "
              f"nodes {r['nodes']:,d}; {r['secs']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

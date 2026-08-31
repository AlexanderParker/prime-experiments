"""Round 28 (constructor): THE DEEP-J WINDOWS, SCAN-FREE.

research/perj_window.py measures Q*_J exactly wherever a full-period source
reaches: every J at m11..m23 (direct cyclic scan) but only J <= 4 at m29, m31,
m37 (Mechanic's exact 4-tuple censuses).  The J = 5, 6 cells there - and every
cell at m41 and beyond, where no census exists at all - need a vehicle that
never touches the period.  This is that vehicle, and it is R80's triple sweep
generalised from J = 3 to any J.

WHY IT IS CHEAP, and this is the whole point of the construct.  A legal
J-window is

    (g_L, w_1, ..., w_{J-2}, g_R)

with every middle in {0, +-2c} mod q' and the nonzero classes alternating.  So
the middles are NOT free: at values <= F(M) there are only a handful of legal
letters (typically 3-5), and T3 forbids most sequences of them.  The only free
coordinates are the two flanks.  The candidate set is therefore O(F^2) per
legal middle word and the words are few - the deep-J program is the CHEAP end
of the CRT oracle, not the expensive one (a 6-tuple has 7 open points, so its
gear domains are far smaller than a pair's; cf. crt_dict's scope note).

FOUR SOUND FILTERS BEFORE THE SOLVER, in cost order.  Each one only ever
REMOVES candidates that cannot be realised, so the descent's first survivor is
still the true maximum.
  (F1) SPECTRUM: every sub-window of j consecutive gaps sums to <= F_j(M).
       F_1..F_4 are read EXACTLY off the 4-tuple census where one exists, and
       from a superset (hence an upper bound, hence sound as a filter) at m41.
  (F2) HOLES: every value must be a realised single gap of M.
  (F3) MIRROR (Lateral r25): occ(w) = occ(reverse w), so only the
       lexicographically smaller of each reverse pair is decided.
  (F4) PHASE SATURATION (Mechanic r26): no translate of the prefix-sum set
       fits inside some gear's exposed set.
Then research/crt_dict.py decide_cover, exact and scan-free.

Descending by span and stopping at the first realised window gives Q*_J
exactly, with every larger candidate refuted; stopping early gives a
gate-clean upper bound "Q*_J <= floor", which is the direction the per-J
analogue needs.

Usage:
  .venv/Scripts/python.exe research/perj_scanfree.py --y 29 --J 5 [--floor 40]
        [--workers 6] [--kind lit|pad|all] [--nodes 4000000]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
import crt_dict                                          # noqa: E402

KNOWN_F = crt_dict.KNOWN_F
KNOWN_F2 = crt_dict.KNOWN_F2
CENSUS = {23: "gap_tuples_23_4.csv", 29: "gap_tuples_29_4.csv",
          31: "gap_tuples_31_4.csv", 37: "gap_tuples_37_4.csv"}
SUPERSET41 = os.path.join(DDIR, "r27", "gap_tuples_41_4_screened_spancap.csv")


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def gears_of(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def exposed(g):
    c = pow(6, -1, g)
    return frozenset(r for r in range(g) if r != c % g and r != (-c) % g)


def ps_refuted(X, gears, E):
    for g in gears:
        Eg = E[g]
        xs = {x % g for x in X}
        if len(xs) > g - 2:
            return True
        if not any(all((t + x) % g in Eg for x in xs) for t in range(g)):
            return True
    return False


def gaps_scan(y):
    from math import prod
    gears = gears_of(y)
    P = prod(gears)
    ex = np.zeros(P, bool)
    for g in gears:
        u = pow(6, -1, g)
        ex[u % g::g] = True
        ex[(-u) % g::g] = True
    op = np.flatnonzero(~ex).astype(np.int64)
    return np.diff(np.concatenate([op, [op[0] + P]]))


# EXACT spectrum values on record that beat what a superset can give.
EXACT_FJ = {37: {3: 97}, 41: {2: 103, 4: 118}}


def spectrum(y):
    """F_1..F_4 of M, EXACT from the census, or an UPPER BOUND from a superset.

    Returns (dict j -> bound, set of realised single-gap values, exact_flag).
    A superset gives an upper bound on each F_j and a superset of the gap
    values - both are the sound direction for a filter that only prunes.
    """
    if y in CENSUS:
        path, exact = os.path.join(DDIR, CENSUS[y]), True
    elif y == 41:
        path, exact = SUPERSET41, False
    elif y <= 23:
        d = gaps_scan(y)
        F = {j: int(max(sum(d[(np.arange(len(d)) + i) % len(d)]
                            for i in range(j)))) for j in (1, 2, 3, 4)}
        vals = set(int(v) for v in np.unique(d))
        assert F[1] == KNOWN_F[y] and F[2] == KNOWN_F2[y], (y, F)
        return F, vals, True
    else:
        return None, None, None
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    F = {}
    for j in (1, 2, 3, 4):
        F[j] = int(max(arr[:, i:i + j].sum(axis=1).max()
                       for i in range(0, 5 - j)))
    vals = set(int(v) for v in np.unique(arr))
    assert F[1] == KNOWN_F[y], ("F gate", y, F[1])
    if exact:
        assert F[2] == KNOWN_F2[y], ("F_2 gate", y, F[2])
    for j, v in EXACT_FJ.get(y, {}).items():
        assert v <= F[j], ("exact F_%d above the superset bound" % j, y, v, F[j])
        F[j] = v                       # tighter AND exact
    return F, vals, exact


def legal_values(q1, a, b, F, vals):
    """Every legal middle VALUE <= F, tagged with its T3 class."""
    out = []
    for v in range(1, F + 1):
        if vals is not None and v not in vals:
            continue
        r = v % q1
        if r == 0:
            out.append((v, 0))
        elif r == a % q1:
            out.append((v, 1))
        elif r == b % q1:
            out.append((v, -1))
    return out


def middle_words(q1, a, b, F, vals, n, Fspec):
    """All T3-legal middle words of length n, pruned by the spectrum."""
    LV = legal_values(q1, a, b, F, vals)
    words = [[]]
    for _ in range(n):
        nxt = []
        for w in words:
            last = next((c for _, c in reversed(w) if c), 0)
            for v, c in LV:
                if c and c == last:
                    continue                      # T3 alternation
                cand = w + [(v, c)]
                vs = [x for x, _ in cand]
                if not spec_ok(vs, Fspec):
                    continue
                nxt.append(cand)
        words = nxt
    return [tuple(v for v, _ in w) for w in words]


def spec_ok(vs, Fspec):
    """(F1) every sub-window of j consecutive values sums to <= F_j."""
    for j in range(1, min(len(vs), max(Fspec)) + 1):
        lim = Fspec.get(j)
        if lim is None:
            continue
        for i in range(0, len(vs) - j + 1):
            if sum(vs[i:i + j]) > lim:
                return False
    return True


def canonical(t):
    return min(t, t[::-1])


def job(args):
    y, tup, nb = args
    t0 = time.time()
    try:
        ok = crt_dict.realised(y, tup, node_budget=nb)
        return tup, ok, time.time() - t0
    except Exception:
        return tup, None, time.time() - t0


def main():
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    y = opt("--y", 29)
    J = opt("--J", 5)
    workers = opt("--workers", 6)
    nb = opt("--nodes", 4_000_000)
    kind = opt("--kind", "all")
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    F = KNOWN_F[y]
    Fspec, vals, exact = spectrum(y)
    floor = opt("--floor", KNOWN_F2.get(y, 0) - 1)
    gears = gears_of(y)
    E = {g: exposed(g) for g in gears}
    print("=== machine %d -> q' = %d,  J = %d,  letters (%d,%d), s_min = %d"
          % (y, q1, J, a, b, min(a, b)))
    print("    spectrum filter F_1..F_4 = %s   [%s]"
          % ([Fspec[j] for j in (1, 2, 3, 4)],
             "EXACT census" if exact else "superset upper bound"))
    words = middle_words(q1, a, b, F, vals, J - 2, Fspec)
    if kind == "lit":
        words = [w for w in words if all(v % q1 for v in w)]
    elif kind == "pad":
        words = [w for w in words if any(v % q1 == 0 for v in w)]
    print("    legal middle words of length %d: %d  (kind=%s)"
          % (J - 2, len(words), kind))
    if not words:
        print("    => NO legal middle word exists at all: Q*_%d = -inf "
              "(the per-J program terminates at J = %d)" % (J, J - 1))
        return
    # build candidates
    cand = {}
    for w in words:
        ms = sum(w)
        for gL in range(1, F + 1):
            if vals is not None and gL not in vals:
                continue
            for gR in range(1, F + 1):
                if vals is not None and gR not in vals:
                    continue
                t = (gL,) + w + (gR,)
                if not spec_ok(list(t), Fspec):
                    continue
                s = gL + ms + gR
                if s <= floor:
                    continue
                c = canonical(t)
                cand[c] = s
    print("    candidates above floor %d after spectrum+hole+mirror filters:"
          " %d" % (floor, len(cand)))
    if not cand:
        print("    => every legal %d-window has span <= %d  (Q*_%d <= %d)"
              % (J, floor, J, floor))
        return
    live = []
    killed = 0
    for t, s in cand.items():
        X, acc = [0], 0
        for v in t:
            acc += v
            X.append(acc)
        if ps_refuted(X, gears, E):
            killed += 1
        else:
            live.append((s, t))
    print("    phase saturation refutes %d of %d for free; %d go to CRT"
          % (killed, len(cand), len(live)))
    if not live:
        print("    => every candidate above %d REFUTED; Q*_%d <= %d"
              % (floor, J, floor))
        return
    blocks = {}
    for s, t in live:
        blocks.setdefault(s, []).append(t)
    t0 = time.time()
    found = None
    nundec = 0
    with Pool(workers) as pool:
        for s in sorted(blocks, reverse=True):
            res = list(pool.imap_unordered(
                job, [(y, t, nb) for t in blocks[s]], chunksize=1))
            yes = [t for t, ok, dt in res if ok]
            und = [t for t, ok, dt in res if ok is None]
            nundec += len(und)
            worst = max(dt for t, ok, dt in res)
            print("      span %3d : %4d cand, %d realised, %d undecided, "
                  "worst %.1f s   [%.0f s]"
                  % (s, len(res), len(yes), len(und), worst,
                     time.time() - t0), flush=True)
            if yes:
                found = (s, sorted(yes)[0])
                break
    if found:
        s, t = found
        print("    => Q*_%d(%d) = %d   witness %s   (every larger candidate "
              "refuted, %d undecided)"
              % (J, y, s, ",".join(map(str, t)), nundec))
        print("       Delta_%d = %d - F_2 = %+d ;  palindrome: %s"
              % (J, s, s - KNOWN_F2[y], t == t[::-1]))
    else:
        print("    => all candidates above %d REFUTED (%d undecided) "
              "=> Q*_%d <= %d" % (floor, nundec, J, floor))
    print("    CRT %.0f s on %d workers" % (time.time() - t0, workers))


if __name__ == "__main__":
    main()

"""Round 30 (mechanic), probe (c): RECORD GENEALOGY.

Every record gap of machine y = M + q' (span F(y), slot k) is, by the merge
law, a window of M whose interior M-openings were all deleted by one phase of
q'.  Those interior openings o_1 < ... < o_L are the ANCESTOR window of M:
gaps (o_1 - k, ..., k + F - o_L), J' = L + 1 gaps.  Each gap of that window is
in turn a merged window of the machine below M, and so on down.  This file
computes the whole tree by residue arithmetic on the slot (slot k is blocked
by gear q iff k = +-6^{-1} mod q) - no scan, no transfer - and reports, per
level:
    machine, the lower window's gaps, how many openings the top gear deleted
    inside each gap, the deletion phase (slot mod q and which tooth each
    deleted opening sits on), and whether the lower window is a RECORD of its
    machine at its depth (span = F_{J'}(lower)) or a runner-up (deficit).
GENERATIONS = follow the LARGEST gap of the ancestor down one machine at a
time and count consecutive levels at which that gap is merged (>= 1 deleted
opening inside).  Pre-registered statements RR-SPECTRUM, RR-DEPTH, C3, C4
(research/data/r30/prereg_mechanic_r30.md) are scored from the printed tree.

Record windows: for y <= 29 every occurrence of the maximal gap is found by a
period scan (machine 29 from chain_depth_r29's memory-mapped opening list);
for y >= 31 the slots on record (C50, C44, C46, C48, C49, r26 anchors) are
used, and the F(43), F(47), F(53), F(59) records are obtained by LIFTING the
recorded lower-machine word-legal windows with the phase that deletes their
interiors - each lift is re-verified at the target machine slot by slot.

usage: uv run python research/genealogy_r30.py [--rank]
"""
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def gears(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def prev_machine(y):
    G = gears(y)
    return G[-2] if len(G) > 1 else None


def teeth(q):
    u = pow(6, -1, q)
    return u, (-u) % q


def is_open(k, y):
    return all(k % q not in teeth(q) for q in gears(y))


# F_J spectra on record (C11-UPDATE), exact
FJ = {
    11: {1: 7, 2: 11, 3: 16, 4: 21, 5: 25, 6: 28},
    13: {1: 11, 2: 16, 3: 23, 4: 26, 5: 28, 6: 31},
    17: {1: 18, 2: 25, 3: 28, 4: 33, 5: 35, 6: 40},
    19: {1: 25, 2: 31, 3: 35, 4: 38, 5: 47, 6: 50},
    23: {1: 34, 2: 39, 3: 50, 4: 58, 5: 65, 6: 77},
    29: {1: 43, 2: 55, 3: 65, 4: 70, 5: 85, 6: 90},
    31: {1: 58, 2: 68, 3: 85, 4: 90, 5: 92, 6: 97},
    37: {1: 88, 2: 90, 3: 97, 4: 105, 5: 113, 6: 120},
    41: {1: 91, 2: 103, 3: 110, 4: 118, 5: 128},
    43: {1: 103, 2: 116, 3: 125, 4: 132},
    47: {1: 118, 2: 134, 3: 145, 6: 177},
    53: {1: 145, 2: 159},
    59: {1: 161, 2: 173},
}


def openings_between(k, span, y):
    return [t for t in range(1, span) if is_open(k + t, y)]


def lift(y_low, k, offs, q):
    """lift the window (k, offs) of machine y_low to machine y_low + q with the
    phase that deletes every interior offset; re-verified at the target."""
    P = prod(gears(y_low))
    span = offs[-1]
    inner = offs[1:-1]
    y = q
    for t in range(q):
        kk = k + t * P
        if all(not is_open(kk + o, y) for o in inner) and is_open(kk, y) \
                and is_open(kk + span, y) \
                and all(not is_open(kk + s, y) for s in range(1, span)):
            return kk % prod(gears(y))
    return None


def tree(y, k, span, depth=0, out=None, maxdepth=12):
    """recursive genealogy of the gap/window [k, k+span] at machine y."""
    if out is None:
        out = []
    G = gears(y)
    q = G[-1]
    yl = prev_machine(y)
    if yl is None:
        return out
    ins = openings_between(k, span, yl)          # deleted by q inside
    L = len(ins)
    offs = [0] + ins + [span]
    gaps = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
    J = len(gaps)
    u, mu = teeth(q)
    tooth_word = "".join("+" if (k + o) % q == u else "-" for o in ins)
    rec = FJ.get(yl, {}).get(J)
    status = ("RECORD" if rec == span else f"runner-up by {rec - span}"
              if rec else "F_J unknown")
    out.append(dict(depth=depth, y=y, lower=yl, k=k, span=span, L=L,
                    gaps=gaps, phase=k % q, teeth=tooth_word, rec=rec,
                    status=status))
    pad = "    " * depth
    print(f"{pad}m{y} [{k}, +{span}]  <-  m{yl} window {gaps} (J={J}, "
          f"{L} deleted by {q} at slot mod {q} = {k % q}, teeth '{tooth_word}')"
          f"  {status} (F_{J}({yl}) = {rec})")
    if depth < maxdepth and yl > 11:
        pos = 0
        for gp in gaps:
            if gp > 1:
                sub = openings_between(k + pos, gp, prev_machine(yl))
                if sub:
                    tree(yl, k + pos, gp, depth + 1, out, maxdepth)
            pos += gp
    return out


def generations(y, k, span):
    """follow the largest gap of the ancestor down; count merged levels."""
    g = 0
    yy, kk, sp = y, k, span
    while True:
        yl = prev_machine(yy)
        if yl is None or yl < 11:
            return g
        ins = openings_between(kk, sp, yl)
        if not ins:
            return g
        g += 1
        offs = [0] + ins + [sp]
        gaps = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
        i = max(range(len(gaps)), key=lambda i: gaps[i])
        kk, sp, yy = kk + offs[i], gaps[i], yl


def scan_records(y):
    """every occurrence of the maximal gap of machine y (y <= 29)."""
    if y == 29:
        import chain_depth_r29 as CD
        X = np.memmap(os.path.join(CD.DATA, "x29.u32"), dtype=np.uint32,
                      mode="r")
        best, where = 0, []
        for a in range(0, CD.N29, 1 << 24):
            b = min(a + (1 << 24) + 1, CD.N29)
            x = X[a:b].astype(np.int64)
            d = np.diff(x)
            m = int(d.max())
            if m > best:
                best, where = m, []
            if m == best:
                where += [int(v) for v in x[:-1][d == m]]
        return best, sorted(set(where))
    G = gears(y)
    P = prod(G)
    w = np.ones(P, bool)
    for q in G:
        for u in teeth(q):
            w[u % q::q] = False
    X = np.flatnonzero(w).astype(np.int64)
    d = np.diff(np.concatenate([X, [X[0] + P]]))
    m = int(d.max())
    return m, [int(v) for v in X[d == m]]


RECORDS = [   # (machine, slot, span, source)
    (31, 21844264615, 58, "C50 chain pass"),
    (37, 1145973108145, 88, "C50 chain pass"),
    (41, 7244836295007, 91, "C50 chain pass"),
]
LIFTS = [     # (lower machine, slot, offsets, new gear, expected span)
    (41, 21157523372970, [0, 28, 103], 43, 103),
    (43, 18497829635337, [0, 85, 116, 118], 47, 118),
    (47, 82799441296736535, [0, 70, 105, 123, 145], 53, 145),
    (53, 2505673933219103747, [0, 10, 128, 161], 59, 161),
]
FJ_WINDOWS = [   # (machine, slot, offsets, label)
    (41, 21157523372970, [0, 28, 103], "F_2(41) = 103"),
    (41, 33044111735742 + 10, [0, 51, 53, 103, 118], "F_4(41) = 118 (inside the F_5 window)"),
    (41, 33044111735742, [0, 10, 61, 63, 113, 128], "F_5(41) = 128"),
    (43, 2161962392309552, [0, 31, 116], "F_2(43) = 116"),
    (43, 1595441702157105, [0, 67, 95, 125], "F_3(43) = 125"),
    (43, 280183736276020, [0, 18, 42, 50, 132], "F_4(43) = 132"),
    (47, 97575004641096768, [0, 54, 134], "F_2(47) = 134"),
    (47, 36068193854725102, [0, 28, 61, 145], "F_3(47) = 145"),
    (47, 46615676895423125, [0, 42, 70, 103, 107, 115, 177], "F_6(47) = 177"),
    (53, 327666424664536738, [0, 77, 159], "F_2(53) = 159"),
    (59, 307199471342884027665, [0, 100, 173], "F_2(59) = 173 A"),
    (59, 13260587016151412007, [0, 73, 173], "F_2(59) = 173 B"),
]


def verify_window(y, k, offs):
    span = offs[-1]
    oset = set(offs)
    for t in range(span + 1):
        assert is_open(k + t, y) == (t in oset), (y, k, t)


def main():
    print("=" * 78)
    print("RECORD GENEALOGY - the F ladder records 13..59")
    print("=" * 78)
    summary = []
    for y in (13, 17, 19, 23, 29):
        F, where = scan_records(y)
        print(f"\nmachine {y}: F = {F}, {len(where)} record windows at slots "
              f"{where[:6]}{'...' if len(where) > 6 else ''}")
        for k in where[:2]:              # both members of the first mirror pair
            t = tree(y, k, F)
            g = generations(y, k, F)
            print(f"    generations along the largest gap: {g}")
            summary.append((y, k, F, t, g))
    for y, k, F, src in RECORDS:
        assert is_open(k, y) and is_open(k + F, y)
        assert all(not is_open(k + s, y) for s in range(1, F))
        print(f"\nmachine {y}: F = {F} at slot {k} ({src}), verified")
        t = tree(y, k, F)
        g = generations(y, k, F)
        print(f"    generations along the largest gap: {g}")
        summary.append((y, k, F, t, g))
    for yl, k, offs, q, F in LIFTS:
        verify_window(yl, k, offs)
        kk = lift(yl, k, offs, q)
        assert kk is not None, ("lift failed", yl, k, q)
        print(f"\nmachine {q}: F = {F} at slot {kk} (lifted from the m{yl} "
              f"window {[offs[i+1]-offs[i] for i in range(len(offs)-1)]} at "
              f"slot {k}; verified at machine {q})")
        t = tree(q, kk, F)
        g = generations(q, kk, F)
        print(f"    generations along the largest gap: {g}")
        summary.append((q, kk, F, t, g))
    print("\n" + "=" * 78)
    print("SPECTRUM RECORDS F_J - their own genealogy")
    print("=" * 78)
    fj_summary = []
    for y, k, offs, lab in FJ_WINDOWS:
        verify_window(y, k, offs)
        print(f"\n{lab}: machine {y}, slot {k}, gaps "
              f"{[offs[i+1]-offs[i] for i in range(len(offs)-1)]}, verified")
        # each gap of the window is a merged window of the machine below
        yl = prev_machine(y)
        q = gears(y)[-1]
        rows = []
        for i in range(len(offs) - 1):
            a, b = offs[i], offs[i + 1]
            ins = openings_between(k + a, b - a, yl)
            sub = [0] + ins + [b - a]
            sg = [sub[j + 1] - sub[j] for j in range(len(sub) - 1)]
            rows.append((b - a, len(ins), sg))
            print(f"    gap {b-a:3d}: {len(ins)} m{yl}-openings deleted by {q} "
                  f"inside -> lower gaps {sg}")
        big = max(rows)
        g = generations(y, k + offs[[r[0] for r in rows].index(big[0])],
                        big[0])
        print(f"    largest gap {big[0]} is {'MERGED' if big[1] else 'INHERITED'}"
              f" at m{yl}; generations along it: {g}")
        fj_summary.append((lab, big[0], big[1] > 0, g))
    # ---------------------------------------------------------------- score
    print("\n" + "=" * 78)
    print("SCORE (pre-registered C1-C5)")
    print("=" * 78)
    steps = [s for s in summary if s[0] >= 29]
    seen = set()
    rr_spec = rr_depth = gen2 = gen3 = 0
    n = 0
    for y, k, F, t, g in steps:
        if y in seen:
            continue
        seen.add(y)
        n += 1
        anc = t[0]
        big = max(anc["gaps"])
        # is the largest gap of the ancestor merged at the machine below?
        i = anc["gaps"].index(big)
        pos = sum(anc["gaps"][:i])
        merged = bool(openings_between(k + pos, big, prev_machine(anc["lower"])))
        rr_spec += anc["status"] == "RECORD"
        rr_depth += merged
        gen2 += g >= 2
        gen3 += g >= 3
        print(f"  step {anc['lower']:2d} -> {y:2d}: ancestor {anc['gaps']} "
              f"{anc['status']:16s} largest gap {big} "
              f"{'MERGED' if merged else 'INHERITED'} below; generations {g}")
    print(f"\n  C1 RR-SPECTRUM: ancestor is the F_J(M) record at {rr_spec} of "
          f"{n} steps (predicted <= 3 of 9)")
    print(f"  C2 RR-DEPTH: largest gap merged below at {rr_depth} of {n} "
          f"(predicted >= 7 of 9)")
    print(f"  C3 generations >= 2 at {gen2} of {n}, >= 3 at {gen3} of {n} "
          f"(predicted all >= 2 from 29->31 on; >= 3 at >= 4)")
    m = sum(1 for _, _, mg, _ in fj_summary if mg)
    print(f"  C5 F_J records: largest gap merged below at {m} of "
          f"{len(fj_summary)} (predicted >= 8 of 10)")
    print("\nALL ASSERTIONS PASSED")


def rank_windows(M, J, S):
    """#J-windows of M (cyclic period) with span > S, and with span == S -
    the ancestor's RANK among M's own J-windows (pre-registered C6)."""
    if M == 29:
        import chain_depth_r29 as CD
        X = np.memmap(os.path.join(CD.DATA, "x29.u32"), dtype=np.uint32,
                      mode="r")
        above = equal = 0
        for a in range(0, CD.N29, 1 << 24):
            b = min(a + (1 << 24) + J, CD.N29)
            x = X[a:b].astype(np.int64)
            if b == CD.N29:
                x = np.concatenate([x, X[:J].astype(np.int64) + CD.P29])
            sp = x[J:] - x[:-J]
            above += int((sp > S).sum())
            equal += int((sp == S).sum())
        return above, equal
    G = gears(M)
    P = prod(G)
    w = np.ones(P, bool)
    for q in G:
        for u in teeth(q):
            w[u % q::q] = False
    X = np.flatnonzero(w).astype(np.int64)
    x = np.concatenate([X, X[:J] + P])
    sp = x[J:] - x[:-J]
    return int((sp > S).sum()), int((sp == S).sum())


def ranks():
    print("\nC6 - the ancestor's RANK among M's own J-windows by span "
          "(#windows strictly above it, #equal to it):")
    for M, J, S, label in ((13, 3, 18, "F(17) ancestor [5,11,2] / [5,6,7]"),
                           (17, 2, 25, "F(19) ancestor [7,18]"),
                           (17, 3, 25, "F(19) ancestor [7,13,5]"),
                           (19, 4, 34, "F(23) ancestor [4,8,15,7]"),
                           (23, 3, 43, "F(29) ancestor [10,10,23]"),
                           (29, 3, 58, "F(31) ancestor [18,10,30]")):
        a, e = rank_windows(M, J, S)
        print(f"  m{M} J={J} span {S}: {a} windows above, {e} equal   ({label})")


if __name__ == "__main__":
    if "--rank" in sys.argv:
        ranks()
    else:
        main()

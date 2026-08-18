"""Worked anatomy of a padded link - the units/frame reconciliation (r15).

Prints, for real openings of machine M with probe qp, the exact objects a
padded link consists of: slot addresses, BOTH members of each slot, the
slot-frame gap, and the same gap expressed in the two other frames in use
by the team. Also verifies that the two kills of a padded link really do
sit at the same tooth of qp (same residue mod qp), which is what "padded"
means.

Frames (this is the whole point of the script):
  * SLOT frame (k): slot k is the pair (6k-1, 6k+1). My censuses count
    gaps as differences of consecutive OPENINGS in k, so a padded link
    has slot-gap exactly qp.
  * INTEGER frame: the members themselves. Slot distance d becomes an
    integer distance 6d, always a multiple of 6.
  * ODD/ADJACENT frame: the corpus chain F(2,y) = 6,15,21,33,54,... is in
    this frame, where the unit is 2 integers. Slot distance d becomes 3d,
    always a multiple of 3 - which is the harvester's "for twins all gaps
    are divisible by 3". So a padded link costs qp in slots == 3qp in the
    adjacent frame == 6qp in integers. No contradiction: one object,
    three units. Cross-check available at every machine: F_adjacent =
    3 * F_slot (33 = 3*11 at y=13, 174 = 3*58 at y=31).

Usage: uv run python research/padded_link_anatomy.py y qp [--limit SLOTS]
       [--want SPAN]   (--want prints runs whose flanked span equals SPAN)
"""
import os
import sys
import numpy as np
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto


def show(k_prev, ks, k_next, qp, tag=""):
    """print full anatomy of a killed run ks (list of slot addresses)."""
    print(f"  {tag}")
    print(f"    flank opening before : k = {k_prev}")
    res = [k % qp for k in ks]
    for i, k in enumerate(ks):
        print(f"    killed opening {i}     : k = {k}   members "
              f"({6*k-1}, {6*k+1})   k mod {qp} = {res[i]}")
    print(f"    flank opening after  : k = {k_next}")
    gaps = [ks[i + 1] - ks[i] for i in range(len(ks) - 1)]
    print(f"    interior slot-gaps   : {gaps}")
    for g in gaps:
        kind = "PADDED (same tooth, one full lap)" if g % qp == 0 \
            else "literal"
        print(f"      gap {g} slots = {3*g} adjacent-frame = {6*g} "
              f"integers   -> {kind}")
    same = [i for i in range(len(res) - 1) if res[i] == res[i + 1]]
    print(f"    residues mod {qp}      : {res}"
          + (f"  -> SAME residue at link(s) {same}: one tooth, one lap"
             if same else ""))
    print(f"    (a link is padded iff its two openings share a residue mod"
          f" {qp}; which absolute residue is irrelevant - over the new"
          f" period q'*P_M every offset occurs, so the site fires exactly"
          f" once)")
    span = ks[-1] - ks[0]
    fl = k_next - k_prev
    print(f"    span {span} slots ({3*span} adj), flanked span {fl} slots "
          f"({3*fl} adj)")


def main():
    args = sys.argv[1:]
    limit = None
    want = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    if "--want" in args:
        i = args.index("--want")
        want = int(args[i + 1])
        del args[i:i + 2]
    y, qp = int(args[0]), int(args[1])
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, limit)
    uv = [pow(6, -1, g) for g in gears]
    s = (2 * pow(6, -1, qp)) % qp
    print(f"machine y={y} (gears {gears}), probe qp={qp}, "
          f"u={pow(6,-1,qp)}, s=2u mod qp={s}")
    print(f"period {P:.4e}, scanning {K:.3e}\n")
    shown = 0
    best = None
    tail = np.array([], dtype=np.int64)
    seg = 64_000_000
    for a in range(0, K, seg):
        b = min(K, a + seg)
        ex = np.zeros(b - a, bool)
        for g, u in zip(gears, uv):
            ex[(u - a) % g::g] = True
            ex[(-u - a) % g::g] = True
        ops = np.concatenate([tail,
                              np.flatnonzero(~ex).astype(np.int64) + a])
        if len(ops) < 4:
            tail = ops
            continue
        d = np.diff(ops)
        pad = np.flatnonzero(d == qp)
        for t in pad:
            if t < 1 or t + 2 >= len(ops):
                continue
            # maximal legal run containing this padded link
            lo = hi = t
            def letter(j):
                m = int(d[j]) % qp
                return 0 if m == 0 else (1 if m == s else
                                         (-1 if m == (qp - s) % qp else 9))
            if letter(t) == 9:
                continue
            seen = [letter(t)]
            while lo - 1 >= 0:
                L = letter(lo - 1)
                nz = [x for x in seen if x != 0]
                if L == 9 or (L != 0 and nz and L == nz[0]):
                    break
                seen.insert(0, L)
                lo -= 1
            while hi + 1 < len(d):
                L = letter(hi + 1)
                nz = [x for x in seen if x != 0]
                if L == 9 or (L != 0 and nz and L == nz[-1]):
                    break
                seen.append(L)
                hi += 1
            if lo < 1 or hi + 2 >= len(ops):
                continue
            ks = [int(x) for x in ops[lo:hi + 2]]
            fl = int(ops[hi + 2] - ops[lo - 1])
            if best is None or fl > best[0]:
                best = (fl, int(ops[lo - 1]), ks, int(ops[hi + 2]))
            if (want is None and shown < 2) or (want and fl == want
                                                and shown < 3):
                show(int(ops[lo - 1]), ks, int(ops[hi + 2]), qp,
                     tag=f"[padded link at slot {int(ops[t])}, "
                         f"flanked span {fl}]")
                shown += 1
                print()
        tail = ops[-12:]
    if best:
        print("=== largest flanked span found containing a padded link ===")
        show(best[1], best[2], best[3], qp, tag=f"flanked span {best[0]}")


if __name__ == "__main__":
    main()

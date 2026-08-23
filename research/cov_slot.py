"""Round 20 (mechanic): COV(M) in the SLOT frame - the exact coverability
spectrum by CRT, no period scan.

Slot k is blocked by gear q  iff  k = +-u_q (mod q), u_q = 6^{-1} mod q.
Fix a window's left opening at position 0.  Writing a_q = (u_q - k) mod q,
gear q blocks exactly the positions i with

    i = a_q  or  i = a_q + s_q  (mod q),   s_q = -2 u_q mod q,

with a_q in Z_q FREE (one choice per gear = the gear's phase).  By CRT every
choice of (a_q)_q is attained by some k in the period, so:

    gap v occurs at machine M
      <=>  exists (a_q): positions 1..v-1 all covered, 0 and v covered by
           NO gear.

This is EXACT occurrence, not a bound - the r17/r19 full-period hole lists
(machines 11..37) are the validation set, and machines 41/43/47/53, whose
periods (5e13..4e18) no scan reaches, become computable.

Usage:
  uv run python research/cov_slot.py validate          # machines 11..37
  uv run python research/cov_slot.py predict 41 43 47 53 [--vmax N]
  uv run python research/cov_slot.py one y v           # single query
"""
import os
import sys
import time
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")


def primes_upto(n):
    s = list(range(n + 1))
    for i in range(2, int(n ** 0.5) + 1):
        if s[i] == i:
            for j in range(i * i, n + 1, i):
                if s[j] == j:
                    s[j] = i
    return [i for i in range(2, n + 1) if s[i] == i]


# full-period measured truth (r17 hole_structure + r19 hist37, all exact)
MEASURED_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}
MEASURED_HOLES = {11: [], 13: [9], 17: [17], 19: [19, 24], 23: [24],
                  29: [41, 42], 31: [54, 56, 57],
                  37: [73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87]}


def gear_options(v, qs):
    """Per gear: list of distinct cover-masks over positions 1..v-1
    (bit i-1 <-> position i) from valid phases a (endpoints 0, v spared).
    Returns None if some gear cannot spare both endpoints."""
    opts = []
    for q in qs:
        u = pow(6, -1, q)
        s = (-2 * u) % q
        forb = {0 % q, v % q, (-s) % q, (v - s) % q}
        masks = set()
        for a in range(q):
            if a in forb:
                continue
            m = 0
            b = (a + s) % q
            for i in range(1, v):
                r = i % q
                if r == a or r == b:
                    m |= 1 << (i - 1)
            masks.add(m)
        if not masks:
            return None
        opts.append((q, sorted(masks, key=lambda m: -bin(m).count("1"))))
    return opts


def realizable(v, qs, node_budget=None, stats=None):
    """Exact: does a gap of exactly v occur at machine with gears qs?
    Returns True / False / None (budget exhausted)."""
    if v <= 1:
        return True
    full = (1 << (v - 1)) - 1
    opts = gear_options(v, qs)
    if opts is None:
        return False
    ngear = len(opts)
    # per gear: union of its masks, static max popcount, and pos -> masks
    union = [0] * ngear
    maxcov = [0] * ngear
    bypos = [dict() for _ in range(ngear)]
    for gi, (q, masks) in enumerate(opts):
        for m in masks:
            union[gi] |= m
            c = bin(m).count("1")
            if c > maxcov[gi]:
                maxcov[gi] = c
            mm = m
            while mm:
                b = mm & -mm
                bypos[gi].setdefault(b, []).append(m)
                mm ^= b
    nodes = [0]
    failed = {}

    def search(covered, used):
        if covered == full:
            return True
        key = (covered, used)
        if key in failed:
            return False
        nodes[0] += 1
        if node_budget and nodes[0] > node_budget:
            raise TimeoutError
        todo = full & ~covered
        # reachability + capacity prunes (static, cheap)
        reach = 0
        cap = 0
        for gi in range(ngear):
            if not used >> gi & 1:
                reach |= union[gi]
                cap += maxcov[gi]
        if todo & ~reach or cap < bin(todo).count("1"):
            failed[key] = 1
            return False
        pos_bit = todo & -todo
        for gi in range(ngear):
            if used >> gi & 1:
                continue
            for m in bypos[gi].get(pos_bit, ()):
                if search(covered | m, used | (1 << gi)):
                    return True
        failed[key] = 1
        return False

    try:
        r = search(0, 0)
    except TimeoutError:
        r = None
    if stats is not None:
        stats["nodes"] = nodes[0]
    return r


def spectrum(y, vmax, node_budget=20_000_000, verbose=True):
    qs = [p for p in primes_upto(y) if p >= 5]
    out = {}
    for v in range(1, vmax + 1):
        t0 = time.time()
        r = realizable(v, qs, node_budget)
        out[v] = r
        if verbose and (r is None or time.time() - t0 > 5):
            print(f"    v={v}: {r} ({time.time()-t0:.0f}s)", flush=True)
    return out


def validate():
    print("VALIDATION against full-period measured spectra (exact hole "
          "lists + F), machines 11..37:")
    all_ok = True
    for y in [11, 13, 17, 19, 23, 29, 31, 37]:
        F = MEASURED_F[y]
        t0 = time.time()
        cov = spectrum(y, F + 1, verbose=False)
        pred_holes = [v for v in range(1, F) if cov[v] is False]
        und = [v for v in cov if cov[v] is None]
        top_ok = (cov[F] is True)
        above = cov[F + 1]
        ok = (pred_holes == MEASURED_HOLES[y]) and top_ok and not und
        all_ok &= ok
        print(f"  machine {y:2d}: F={F}  pred holes {pred_holes} "
              f"vs measured {MEASURED_HOLES[y]}  top v={F} realizable: "
              f"{top_ok}  v=F+1: {above}  "
              f"{'AGREES' if ok else 'MISMATCH'}  ({time.time()-t0:.1f}s)")
        if und:
            print(f"    UNDECIDED (budget): {und}")
    print(f"  => {'ALL 8 MACHINES AGREE' if all_ok else 'MISMATCH SOMEWHERE'}")
    return all_ok


def predict(ys, vmax=None, node_budget=200_000_000):
    for y in ys:
        qs = [p for p in primes_upto(y) if p >= 5]
        P = prod(qs)
        vm = vmax or 140
        print(f"\nPREDICT machine {y} (gears {qs}), period {P:.4g} - "
              f"no scan possible; COV is exact by CRT.  v <= {vm}",
              flush=True)
        t0 = time.time()
        cov = spectrum(y, vm, node_budget)
        realized = [v for v in cov if cov[v] is True]
        und = [v for v in cov if cov[v] is None]
        F = max(realized)
        holes = [v for v in range(1, F) if cov[v] is False]
        print(f"  F({y}) = {F}   (largest realizable v; all v in "
              f"({F},{vm}] non-realizable: "
              f"{all(cov[v] is False for v in range(F+1, vm+1))})")
        print(f"  holes below F = {holes}")
        if und:
            print(f"  UNDECIDED (budget): {und}")
        print(f"  ({time.time()-t0:.0f}s)", flush=True)


def main():
    args = sys.argv[1:]
    vmax = None
    if "--vmax" in args:
        i = args.index("--vmax")
        vmax = int(args[i + 1])
        del args[i:i + 2]
    if not args or args[0] == "validate":
        validate()
    elif args[0] == "predict":
        predict([int(a) for a in args[1:]], vmax)
    elif args[0] == "one":
        y, v = int(args[1]), int(args[2])
        qs = [p for p in primes_upto(y) if p >= 5]
        t0 = time.time()
        print(f"machine {y}, v={v}: {realizable(v, qs)} "
              f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

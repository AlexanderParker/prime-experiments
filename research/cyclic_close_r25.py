"""Round 25 (mechanic): THE LINEAR-CLOSE DEFECT - diagnosis, exact correction,
and a source fix for the gap-pair census.

THE DEFECT (found by Lateral's parity law on first use, routed to this lane).
research/gap_pair_census.py streams [start, K) and takes np.diff of the
opening list.  A period is a CIRCLE: N openings carry N gaps, the last being
the WRAP gap from the final opening back round to the first.  The linear close
drops it, so every full-period ghist row in research/data/gap_pair_hist.csv
carries N-1 gaps.  Harmless for densities (relative error 1/N), fatal for any
exact identity - which is exactly what Lateral hit.

    machine   ghist total      prod(q-2)      short by
      11              134            135             1
      13            1,484          1,485             1
      17           22,274         22,275             1
      19          378,674        378,675             1
      23        7,952,174      7,952,175             1
      29      214,708,724    214,708,725             1
      31    6,226,553,024  6,226,553,025             1

THE MISSING GAP, IN CLOSED FORM (so the correction needs no rescan).  Slot 0 is
an opening at EVERY machine: gear q blocks k = +-u_q with u_q = 6^{-1} mod q,
and u_q is never 0.  The opening set is closed under k -> -k (the mirror law,
C18), so the largest opening is P - x_1 where x_1 is the smallest positive
opening.  Hence

    wrap gap  =  P - x_{N-1}  =  x_1  =  d_0,   the FIRST gap.

So the cyclic close adds exactly one more count at the value of the first gap -
and every affected census can be corrected in closed form, at any machine,
without touching its period.  Asserted below at every machine where the
openings are cheap to enumerate directly.

WHAT ELSE THE CLOSE OWES.  The gap histogram is short by one entry, but the
LAG-PAIR and RUN-MINIMUM tables are short by more, because they lose every
structure straddling the seam: with cyclic gaps d_0..d_{N-1},

    ghist    linear has d_0..d_{N-2}                    -> missing 1
    pair[j]  linear has (d_i, d_{i+j}), i <= N-2-j      -> missing j+1
    minh[m]  linear has min(d_i..d_{i+m-1}), i <= N-1-m -> missing m

All of them involve only gaps within LAGS+RUNS of the seam, so they are
computed here from the first and last few openings alone.

usage:
  python research/cyclic_close_r25.py check        # diagnose + assert closed form
  python research/cyclic_close_r25.py fix          # write the corrected CSV
"""
import csv
import os
import sys
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LAGS, RUNS = 5, 6
W = LAGS + RUNS + 4


def primes_upto(n):
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, n + 1, i):
                s[j] = False
    return [i for i in range(n + 1) if s[i]]


def gears(y):
    return [p for p in primes_upto(y) if p >= 5]


def is_open(k, gs):
    for q in gs:
        u = pow(6, -1, q)
        r = k % q
        if r == u % q or r == (-u) % q:
            return False
    return True


def head_tail(y, want=W, window=20000):
    """First and last `want` openings of machine y's period."""
    gs = gears(y)
    P = prod(gs)
    head = []
    k = 0
    while len(head) < want:
        if is_open(k, gs):
            head.append(k)
        k += 1
        assert k < window, (y, "head window too small")
    tail = []
    k = P - 1
    while len(tail) < want:
        if is_open(k, gs):
            tail.append(k)
        k -= 1
        assert P - k < window, (y, "tail window too small")
    return P, head, sorted(tail)


def seam_structures(y):
    """The structures a linear close drops, computed from the seam alone.

    Returns (wrap_gap, missing_pairs{j: [(a,b)]}, missing_runs{m: [minval]}).
    Local gap array g: g[t] = d_{N-W+t} for t < W-1, g[W-1] = d_{N-1} (the
    wrap), g[W-1+u] = d_{u-1} for u >= 1.
    """
    P, head, tail = head_tail(y)
    seq = tail + [x + P for x in head]
    g = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    wrap_local = W - 1
    assert g[wrap_local] == (P - tail[-1]) + head[0]

    def loc(i_off):
        """local index of cyclic gap d_{N-1+i_off} (i_off <= 0 tail, > 0 head)"""
        return wrap_local + i_off

    pairs = {}
    for j in range(1, LAGS + 1):
        # missing i = N-1-j .. N-1  -> offsets -j..0 relative to N-1
        pairs[j] = [(g[loc(off)], g[loc(off + j)])
                    for off in range(-j, 1)]
    runs = {}
    for m in range(2, RUNS + 1):
        # missing i = N-m .. N-1 -> offsets -(m-1)..0
        runs[m] = [min(g[loc(off + t)] for t in range(m))
                   for off in range(-(m - 1), 1)]
    return g[wrap_local], pairs, runs, head, tail, P


def check():
    print("=== 1. THE PARITY IDENTITY: every full-period table totals N ===")
    print("  (N openings on a circle carry N gaps, N lag-j pairs and N "
          "m-run minima)")
    hrows = list(csv.DictReader(open(os.path.join(DATA,
                 "gap_pair_hist.csv"))))
    jrows = list(csv.DictReader(open(os.path.join(DATA,
                 "gap_pair_joint.csv"))))
    tot = {}
    for r in hrows:
        if float(r["coverage"]) >= 1.0:
            tot.setdefault(int(r["y"]), {}).setdefault(
                r["kind"] + r["index"], 0)
            tot[int(r["y"])][r["kind"] + r["index"]] += int(r["count"])
    for r in jrows:
        if float(r["coverage"]) >= 1.0:
            tot.setdefault(int(r["y"]), {}).setdefault("lag" + r["lag"], 0)
            tot[int(r["y"])]["lag" + r["lag"]] += int(r["count"])
    shorts, bad = set(), []
    for y in sorted(tot):
        n = prod(q - 2 for q in gears(y))
        d = {k: n - v for k, v in tot[y].items()}
        shorts |= set(d.values())
        print(f"  m{y:<3} N = {n:>15,}   shortfalls " +
              " ".join(f"{k}:{v}" for k, v in sorted(d.items())))
        if set(d.values()) != {0}:
            bad.append((y, d))
    if not bad:
        print("  ALL TABLES CLOSED CYCLICALLY - every one totals N (asserted)")
    else:
        print("  LINEAR CLOSE STILL PRESENT (run `fix`): shortfalls above are "
              "the dropped seam structures - 1 gap, j+1 pairs at lag j, "
              "m minima at run length m")
        assert False, f"tables not cyclically closed: {bad[:2]}"

    print()
    print("=== 2. the missing gap is the FIRST gap (mirror law), closed form ===")
    for y in sorted(tot):
        wrap, pairs, runs, head, tail, P = seam_structures(y)
        d0 = head[1] - head[0]
        print(f"  m{y:<3} P = {P:>15,}  first opening {head[0]}  "
              f"last opening {tail[-1]:,}  wrap gap {wrap}  first gap {d0}"
              f"  {'AGREE' if wrap == d0 else 'DIFFER'}")
        assert head[0] == 0, (y, "slot 0 should always be an opening")
        assert tail[-1] == P - head[1], (y, "mirror law violated at the seam")
        assert wrap == d0, (y, wrap, d0)
    print("  wrap gap == first gap at every machine (asserted)")

    print()
    print("=== 3. what the close owes the pair and run tables ===")
    for y in sorted(tot):
        _, pairs, runs, _, _, _ = seam_structures(y)
        np_ = {j: len(v) for j, v in pairs.items()}
        nr = {m: len(v) for m, v in runs.items()}
        print(f"  m{y:<3} missing pairs per lag {np_}  missing runs per m {nr}")
    print("  (missing pair count at lag j is j+1; missing run count at m is m)")
    print()
    print("ALL ASSERTIONS PASSED")


def _bump(rows, fields, keyfields, deltas, path):
    """Add `deltas` (key tuple over keyfields -> count) into `rows`.

    Keys not already present are APPENDED (a seam structure can carry a
    (value) or (lag, gu, gv) cell that never occurs away from the seam).
    The original file is kept as <path>.linear.bak.
    """
    def key_of(r):
        return tuple(r[f] for f in keyfields) \
            if float(r["coverage"]) >= 1.0 else None

    idx = {}
    for i, r in enumerate(rows):
        k = key_of(r)
        if k is not None and k not in idx:
            idx[k] = i
    appended = 0
    for k, dv in deltas.items():
        if k in idx:
            rows[idx[k]] = dict(rows[idx[k]])
            rows[idx[k]]["count"] = str(int(rows[idx[k]]["count"]) + dv)
        else:
            row = dict(zip(keyfields, k))
            row["count"] = str(dv)
            rows.append(row)
            appended += 1
    if not os.path.exists(path + ".linear.bak"):
        os.replace(path, path + ".linear.bak")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.basename(path)}: {len(deltas)} cells corrected "
          f"({appended} newly created); pre-fix file kept as "
          f"{os.path.basename(path)}.linear.bak")


def fix():
    hp = os.path.join(DATA, "gap_pair_hist.csv")
    jp = os.path.join(DATA, "gap_pair_joint.csv")
    hrows = list(csv.DictReader(open(hp)))
    jrows = list(csv.DictReader(open(jp)))
    hf, jf = list(hrows[0].keys()), list(jrows[0].keys())
    full = sorted({int(r["y"]) for r in hrows if float(r["coverage"]) >= 1.0})
    covof = {}
    for r in hrows:
        if float(r["coverage"]) >= 1.0:
            covof[int(r["y"])] = r["coverage"]

    gdelta, mdelta, jdelta = {}, {}, {}
    for y in full:
        wrap, pairs, runs, _, _, _ = seam_structures(y)
        gdelta[(str(y), covof[y], "ghist", "0", str(wrap))] = 1
        for m, vals in runs.items():
            for v in vals:
                k = (str(y), covof[y], "minhist", str(m), str(v))
                mdelta[k] = mdelta.get(k, 0) + 1
        for j, ps in pairs.items():
            for a, b in ps:
                k = (str(y), covof[y], str(j), str(a), str(b))
                jdelta[k] = jdelta.get(k, 0) + 1

    hd = dict(gdelta)
    hd.update(mdelta)
    _bump(hrows, hf, ["y", "coverage", "kind", "index", "value"], hd, hp)
    _bump(jrows, jf, ["y", "coverage", "lag", "gu", "gv"], jdelta, jp)
    print()
    check()


if __name__ == "__main__":
    (fix if len(sys.argv) > 1 and sys.argv[1] == "fix" else check)()

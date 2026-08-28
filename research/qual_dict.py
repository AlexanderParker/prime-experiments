"""Round 25 (formalist): THE STRATIFIED QUALIFYING DICTIONARIES of a machine,
computed over its FULL period by a chunked sieve.

WHY THESE OBJECTS.  `proofs/Potential.lean`'s `merged_le_of_potential` turns
(D) at step M -> M + q' into three per-step conditions on M's gap word, with
NO depth quantifier.  Instantiating the potential by the qualifying tail
(the h11/h13/h17/h19 shape) reduces those three to

    (A)  every realised PAIR has sum <= B                      [depth 2]
    (Bj) every realised j-window whose interiors all qualify
         (gap >= floor) has sum <= B,   j = 3 .. K+1
    (C)  no K consecutive gaps all qualify                     [run bound]

with B = F(M) + q' and floor = 2u'' the next gear's tooth floor.  Each of
those is a statement about a SMALL explicit dictionary - the realised
j-windows with qualifying interiors - not about the machine's period.  So
this script computes, exactly and over the whole period:

  * the gap ladder F_1..F_8 and the pair dictionary,
  * for each j, the QUALIFYING j-window dictionary D_j = the set of realised
    j-tuples of consecutive gaps whose interior gaps are all >= floor,
  * the longest run of consecutive gaps >= floor,

and asserts the resulting Q_j ladder against the corpus values where known.
The dictionaries are written as CSV for direct transcription into Lean.

Memory: chunked, so machine 29's 1,078,282,205-slot period never allocates
more than ~100 MB at once (compute policy: memory is the binding constraint).

Usage:
    python research/qual_dict.py 23        # validation, 37,182,145 slots
    python research/qual_dict.py 29        # the target, 1,078,282,205 slots
"""
import os
import sys
import time
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

GEARS = {11: [5, 7, 11], 13: [5, 7, 11, 13], 17: [5, 7, 11, 13, 17],
         19: [5, 7, 11, 13, 17, 19], 23: [5, 7, 11, 13, 17, 19, 23],
         29: [5, 7, 11, 13, 17, 19, 23, 29],
         31: [5, 7, 11, 13, 17, 19, 23, 29, 31]}
# floor 2u'' set by the NEXT gear q''
NEXT = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37}
# F(M) from the corpus ladder
FCORPUS = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58}
JMAX = 10         # deepest window inspected (base-64 codes: 6*JMAX <= 63)
CHUNK = 60_000_000


def tooth_floor(qpp):
    u = pow(6, -1, qpp)
    return 2 * min(u % qpp, (-u) % qpp)


def gaps_of_period(y, chunk=CHUNK, jmax=JMAX):
    """Yield int64 arrays of consecutive gaps covering one full period,
    cyclically stitched.

    The CYCLIC SEAM is the trap here (mechanic's standing rule 18: a
    junction-straddling window seen by neither run).  The gap word is cyclic
    with `prod (q-2)` letters, so windows that START near the end of the
    period and continue past it must be seen too.  The final yield is
    therefore `[wrap gap] ++ (the first jmax-1 gaps of the period)`, which,
    prefixed by the carried tail, covers every straddling window.  Duplicated
    coverage is harmless (the consumer keeps sets and maxima)."""
    gears = GEARS[y]
    P = prod(gears)
    prev = None
    first = None
    head = None
    start = 0
    while start < P:
        stop = min(start + chunk, P)
        L = stop - start
        blocked = np.zeros(L, bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u % q, (-u) % q):
                off = (t - start) % q
                blocked[off::q] = True
        op = np.flatnonzero(~blocked).astype(np.int64) + start
        del blocked
        if op.size:
            if first is None:
                first = int(op[0])
            if prev is not None:
                op = np.concatenate(([prev], op))
            d = np.diff(op)
            if head is None:
                head = d[:jmax].copy()
            yield d
            prev = int(op[-1])
        start = stop
    # cyclic seam: the wrap gap, then the period's own first gaps
    yield np.concatenate((np.array([P + first - prev], dtype=np.int64), head))


def scan(y, jmax=JMAX, chunk=CHUNK):
    floor = tooth_floor(NEXT[y])
    tail = np.zeros(0, dtype=np.int64)
    Fj = [0] * (jmax + 1)
    dicts = {j: set() for j in range(2, jmax + 1)}
    pairs = set()
    maxrun = 0
    run = 0
    ngaps = 0
    t0 = time.time()
    for blk in gaps_of_period(y, chunk, jmax):
        g = np.concatenate((tail, blk)) if tail.size else blk
        ngaps += int(blk.size)
        n = g.size
        # running qualifying-run length across the block boundary handled by
        # recomputing on the overlap region only (tail carries jmax-1 gaps)
        q = g >= floor
        # F_j via sliding sums
        cs = np.concatenate(([0], np.cumsum(g)))
        for j in range(1, jmax + 1):
            if n >= j:
                Fj[j] = max(Fj[j], int((cs[j:] - cs[:-j]).max()))
        # qualifying dictionaries: windows of j gaps with interiors qualifying.
        # Tuples are encoded base 64 (every gap is < 64 at every machine
        # scanned here, asserted) so np.unique does the deduplication.
        assert int(g.max()) < 64, f"gap {int(g.max())} >= 64 breaks the encoding"
        for j in range(2, jmax + 1):
            if n < j:
                continue
            m = n - j + 1
            ok = np.ones(m, bool)
            for i in range(1, j - 1):
                ok &= q[i:i + m]
            idx = np.flatnonzero(ok)
            if idx.size:
                code = np.zeros(idx.size, dtype=np.int64)
                for k in range(j):
                    code += g[idx + k] << (6 * k)
                dicts[j].update(np.unique(code).tolist())
        if n >= 2:
            code = g[:-1] + (g[1:] << 6)
            pairs.update(np.unique(code).tolist())
        # longest qualifying run, vectorised
        if q.any():
            padded = np.concatenate(([False], q, [False]))
            d = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)
            maxrun = max(maxrun, int((ends - starts).max()))
        keep = min(jmax - 1, n)
        tail = g[n - keep:].copy()
    dt = time.time() - t0
    ngaps -= jmax          # the seam yield re-sends the period's first gaps
    assert ngaps == prod(q - 2 for q in GEARS[y]), (ngaps, y)
    return dict(y=y, floor=floor, Fj=Fj, dicts=dicts, pairs=pairs,
                maxrun=maxrun, ngaps=ngaps, secs=dt)


def decode(code, j):
    return tuple((code >> (6 * k)) & 63 for k in range(j))


def report(r):
    y, floor = r["y"], r["floor"]
    B = FCORPUS[y] + NEXT[y]
    print(f"machine {y}: floor 2u''({NEXT[y]}) = {floor}, "
          f"budget F({y}) + {NEXT[y]} = {B}, "
          f"{r['ngaps']:,} gaps, {r['secs']:.0f}s")
    print("  F_j ladder:", ", ".join(f"F_{j}={r['Fj'][j]}"
                                     for j in range(1, JMAX + 1)))
    print(f"  longest run of gaps >= {floor}: {r['maxrun']}")
    assert r["maxrun"] < JMAX - 1, "run longer than the chunk overlap"
    print(f"  realised pairs: {len(r['pairs']):,}, max pair sum = "
          f"{max(sum(decode(c, 2)) for c in r['pairs'])}")
    worst = {}
    for j in range(2, JMAX + 1):
        D = r["dicts"][j]
        if not D:
            print(f"  D_{j}: EMPTY  (no realised {j}-window with qualifying "
                  f"interiors)")
            continue
        s = max(sum(decode(c, j)) for c in D)
        worst[j] = s
        print(f"  D_{j}: {len(D):,} tuples, Q_{j} = max sum = {s}"
              f"   {'OK' if s <= B else 'OVER BUDGET'}")
    mq = max(worst.values())
    print(f"  => max_j Q_j = {mq}  vs budget {B}: "
          f"{'CERTIFIES' if mq <= B else 'FAILS'}  (margin {B - mq})")
    return worst


CORPUS_Q = {          # exact Q_j(M; floor) from formalist.md 2.19 / 12c
    23: {2: 39, 3: 43, 4: 50, 5: 55, 6: 60},
    19: {2: 31, 3: 35, 4: 37, 5: 38},
}


def main():
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    r = scan(y)
    worst = report(r)
    if y in CORPUS_Q:
        for j, v in CORPUS_Q[y].items():
            assert worst.get(j) == v, (
                f"GATE FAIL: Q_{j}({y}) = {worst.get(j)}, corpus says {v}")
        print(f"  GATE: Q_j ladder matches the corpus at machine {y} - OK")
    assert r["Fj"][1] == FCORPUS[y], (
        f"GATE FAIL: F({y}) = {r['Fj'][1]}, corpus says {FCORPUS[y]}")
    print(f"  GATE: F({y}) = {r['Fj'][1]} matches the corpus - OK")
    out = os.path.join(DATA, f"qualdict_{y}.csv")
    with open(out, "w") as f:
        f.write("j,tuple\n")
        for c in sorted(r["pairs"]):
            f.write("2," + " ".join(map(str, decode(c, 2))) + "\n")
        for j in range(3, JMAX + 1):
            for c in sorted(r["dicts"][j]):
                f.write(f"{j}," + " ".join(map(str, decode(c, j))) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

"""
LATERAL round 28 - THE m23 RUNG OF THE TOOTH-COUNTERFACTUAL TABLE (backlog U12(i)).

Round 27 measured the twin machine's percentile in the exhaustive counterfactual
family V(y) = prod_{q<=y} {1..(q-1)/2} (same gears, same mirror symmetry, same
period, same survivor count prod(q-2); only the tooth positions move) at
m11/13/17/19: 20.0 / 18.1 / 26.4 / 17.1 percent for F.  U12(i) asks whether the
~20% plateau HOLDS at the next machine or drifts toward 50%.

m23: |V(23)| = 142,560 sievings over P(23) = 37,182,145.  A direct full-period
sieve per member costs ~0.25 s (37 MB bool + a 8.0 M flatnonzero), i.e. ~10 h
single core.  This script uses the BLOCK DECOMPOSITION instead, which is ~5x
cheaper and shares work across the 11 choices of v_23:

    every m23 opening is x = o + j*P19 with o an m19 opening and j in [0,23),
    and x is blocked by gear 23 iff (o + j*P19) = +-v_23 (mod 23).

So for a fixed m19 tooth vector we sieve ONCE (378,675 openings), take
rr = O19 mod 23, and each of the 23 blocks of the m23 period is O19 with two
residue classes deleted.  Concatenating the 23 masked blocks in j order gives
the m23 opening set ALREADY SORTED, from which F and F_2 are two np.diff calls.

Emits F(m19), F_2(m19), F(m23), F_2(m23) for all 142,560 members, so the
percentile of the twin is available for F, for F_2, for the 19->23 increment
F(m23) - F_2(m19), and (as a free byproduct offered to the manager, whose item
U13 it is) for the budget slack F(m23) - F(m19) - 23.

Usage:
  python tooth_m23_r28.py --gate            # correctness gates vs a direct sieve
  python tooth_m23_r28.py --run --workers 6 # the full 142,560-member census
  python tooth_m23_r28.py --report          # percentiles from the emitted npy
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

GEARS19 = [5, 7, 11, 13, 17, 19]
Q23 = 23
P19 = 1
for _q in GEARS19:
    P19 *= _q
P23 = P19 * Q23
N19 = 1
for _q in GEARS19:
    N19 *= _q - 2
N23 = N19 * (Q23 - 2)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "r28")

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def sieve_openings(gears, vs, P):
    """Sorted opening set of the symmetric-teeth sieve, as int32."""
    blocked = np.zeros(P, dtype=bool)
    for q, v in zip(gears, vs):
        blocked[v % q::q] = True
        blocked[(-v) % q::q] = True
    return np.flatnonzero(~blocked).astype(np.int32)


def cyclic_F(op, P):
    """max cyclic 1-gap."""
    d = np.diff(op)
    return int(max(int(d.max()), P - int(op[-1]) + int(op[0])))


def cyclic_F2(op, P):
    """max cyclic 2-gap (o_{t+2} - o_t).

    No int64 cast: op is int32 with values < P < 2^31 and the differences are
    small positive numbers, so int32 arithmetic is exact.  The casts this
    replaces allocated two full copies per call and were what exhausted the
    commit limit on an 8-worker run."""
    d2 = op[2:] - op[:-2]
    w1 = P + int(op[0]) - int(op[-2])
    w2 = P + int(op[1]) - int(op[-1])
    return int(max(int(d2.max()), w1, w2))


def m23_from_m19(O19, rr, v23):
    """m23 opening set (sorted, int32) from the m19 opening set and gear 23's tooth.

    Reference implementation - materialises the whole 7.95 M array.  Used by the
    gate; the census uses m23_FF2 below, which never builds it."""
    c = P19 % Q23
    t0, t1 = v23 % Q23, (-v23) % Q23
    blocks = []
    for j in range(Q23):
        s0 = (t0 - j * c) % Q23
        s1 = (t1 - j * c) % Q23
        vals = O19[(rr != s0) & (rr != s1)]
        blocks.append(vals + np.int32(j * P19))
    return np.concatenate(blocks)


def m23_FF2(O19, rr, v23):
    """(F, F_2) of the m23 machine WITHOUT building its opening set.

    Block j of the m23 period is O19 with the two residue classes
    (+-v23 - j*P19) mod 23 deleted, shifted by j*P19, and the 23 blocks are
    already in ascending order.  So every 1-gap and 2-gap is either INTERNAL to
    a block (a diff of the masked O19, shift-invariant) or crosses one of the 23
    SEAMS (including the period's own wrap, block 22 -> block 0).  Working block
    by block keeps every array at 378,675 elements instead of 7,952,175 and
    removes the two 64 MB int64 casts the reference path paid per member.
    """
    c = P19 % Q23
    t0, t1 = v23 % Q23, (-v23) % Q23
    best1 = 0
    best2 = 0
    firsts = np.empty(Q23, dtype=np.int64)   # first two of each block, absolute
    firsts2 = np.empty(Q23, dtype=np.int64)
    lasts = np.empty(Q23, dtype=np.int64)    # last two of each block, absolute
    lasts2 = np.empty(Q23, dtype=np.int64)
    for j in range(Q23):
        s0 = (t0 - j * c) % Q23
        s1 = (t1 - j * c) % Q23
        v = O19[(rr != s0) & (rr != s1)]
        d = v[1:] - v[:-1]
        m = int(d.max())
        if m > best1:
            best1 = m
        m = int((v[2:] - v[:-2]).max())
        if m > best2:
            best2 = m
        off = j * P19
        firsts[j] = int(v[0]) + off
        firsts2[j] = int(v[1]) + off
        lasts[j] = int(v[-1]) + off
        lasts2[j] = int(v[-2]) + off
    # seams, including the cyclic one (block 22 -> block 0 shifted by P23)
    nxt_f = np.roll(firsts, -1).copy()
    nxt_f2 = np.roll(firsts2, -1).copy()
    nxt_f[-1] += P23
    nxt_f2[-1] += P23
    best1 = max(best1, int((nxt_f - lasts).max()))
    best2 = max(best2, int((nxt_f - lasts2).max()), int((nxt_f2 - lasts).max()))
    return best1, best2


def run_chunk(args):
    lo, hi = args
    space19 = [list(range(1, (q - 1) // 2 + 1)) for q in GEARS19]
    vecs = list(itertools.product(*space19))
    rows = []
    for idx in range(lo, hi):
        v19 = list(vecs[idx])
        O19 = sieve_openings(GEARS19, v19, P19)
        if O19.size != N19:
            raise AssertionError("m19 opening count %d != %d" % (O19.size, N19))
        F19 = cyclic_F(O19, P19)
        F219 = cyclic_F2(O19, P19)
        rr = (O19 % Q23).astype(np.int8)
        for v23 in range(1, (Q23 - 1) // 2 + 1):
            F23, F223 = m23_FF2(O19, rr, v23)
            rows.append((idx, v23, F19, F219, F23, F223))
    return np.array(rows, dtype=np.int64)


def do_gate():
    """Block decomposition vs a direct full-period m23 sieve, on sampled vectors."""
    rng = np.random.default_rng(2828)
    space19 = [list(range(1, (q - 1) // 2 + 1)) for q in GEARS19]
    vecs = list(itertools.product(*space19))
    gate(len(vecs) == 12960, "|V(19)| = 12960")
    gate(len(vecs) * 11 == 142560, "|V(23)| = 142560")
    gate(P23 == 37182145, "P(23) = 37,182,145")
    gate(N23 == 7952175, "N(23) = prod(q-2) = 7,952,175")
    true19 = [pow(6, -1, q) for q in GEARS19]
    true19 = [min(v % q, (-v) % q) for v, q in zip(true19, GEARS19)]
    gate(true19 == [1, 1, 2, 2, 3, 3], "twin m19 tooth vector is (1,1,2,2,3,3)")
    picks = [vecs.index(tuple(true19))] + [int(i) for i in rng.integers(0, len(vecs), 3)]
    for idx in picks:
        v19 = list(vecs[idx])
        O19 = sieve_openings(GEARS19, v19, P19)
        rr = (O19 % Q23).astype(np.int8)
        for v23 in (1, 4, 11):
            fast = m23_from_m19(O19, rr, v23)
            slow = sieve_openings(GEARS19 + [Q23], v19 + [v23], P23)
            gate(fast.size == slow.size and bool(np.array_equal(fast, slow)),
                 "block decomposition == direct sieve at v19=%s v23=%d "
                 "(%d openings)" % (tuple(v19), v23, fast.size))
            # F_2 sanity: F_2 >= F and F_2 <= 2F
            F, F2 = cyclic_F(fast, P23), cyclic_F2(fast, P23)
            gate(F <= F2 <= 2 * F, "F <= F_2 <= 2F at v19=%s v23=%d (%d, %d)"
                 % (tuple(v19), v23, F, F2))
            # the census path must agree with the materialised path exactly
            gate((F, F2) == m23_FF2(O19, rr, v23),
                 "block/seam census path == materialised path at v19=%s v23=%d "
                 "-> (F,F_2) = (%d,%d)" % (tuple(v19), v23, F, F2))
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)


SHARD = os.path.join(OUT, "m23_shards")


def run_shard(args):
    """One shard, with retries.

    This box runs six lanes at once and its physical RAM is the binding
    constraint: working sets get trimmed to ~11 MB and a 1.3 MB numpy allocation
    can fail transiently even with GB of commit headroom.  Two 8- and 4-worker
    runs of this census died that way.  A shard is cheap and idempotent, so
    retry it rather than losing the pool."""
    lo, hi = args
    path = os.path.join(SHARD, "s_%05d_%05d.npy" % (lo, hi))
    if os.path.exists(path):
        return path
    for attempt in range(8):
        try:
            arr = run_chunk((lo, hi))
            break
        except MemoryError:
            time.sleep(5 * (attempt + 1))
    else:
        raise MemoryError("shard %d-%d failed 8 times" % (lo, hi))
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)
    return path


TWIN_V23 = 4          # round(23/6) = 6^{-1} mod 23, folded
PSHARD = os.path.join(OUT, "m23_pinned")


def run_chunk_pinned(args):
    """v_23 PINNED to the twin's own value; only the m19 teeth vary.

    The full census (all 11 v_23 x 12,960 m19 vectors) needs ~2.7 h on this box
    under six-lane memory contention, which does not fit a round.  This is the
    deliberately-narrowed, self-contained object: the exhaustive family (B),
    exactly the column reported at every other step, at 1/11 the cost.
    """
    lo, hi = args
    space19 = [list(range(1, (q - 1) // 2 + 1)) for q in GEARS19]
    vecs = list(itertools.product(*space19))
    rows = []
    for idx in range(lo, hi):
        O19 = sieve_openings(GEARS19, list(vecs[idx]), P19)
        if O19.size != N19:
            raise AssertionError("m19 opening count wrong")
        rr = (O19 % Q23).astype(np.int8)
        F23, F223 = m23_FF2(O19, rr, TWIN_V23)
        rows.append((idx, TWIN_V23, cyclic_F(O19, P19), cyclic_F2(O19, P19),
                     F23, F223))
    return np.array(rows, dtype=np.int64)


def run_pshard(args):
    lo, hi = args
    path = os.path.join(PSHARD, "p_%05d_%05d.npy" % (lo, hi))
    if os.path.exists(path):
        return path
    # reuse the full shard if this range was already done by the full census
    full = os.path.join(SHARD, "s_%05d_%05d.npy" % (lo, hi))
    if os.path.exists(full):
        a = np.load(full)
        arr = a[a[:, 1] == TWIN_V23]
    else:
        for attempt in range(8):
            try:
                arr = run_chunk_pinned((lo, hi))
                break
            except MemoryError:
                time.sleep(5 * (attempt + 1))
        else:
            raise MemoryError("pinned shard %d-%d failed 8 times" % (lo, hi))
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)
    return path


def do_pinned(workers):
    import multiprocessing as mp
    os.makedirs(PSHARD, exist_ok=True)
    n = 12960
    step = 60
    chunks = [(i, min(i + step, n)) for i in range(0, n, step)]
    todo = [c for c in chunks
            if not os.path.exists(os.path.join(PSHARD, "p_%05d_%05d.npy" % c))]
    reuse = sum(1 for c in todo
                if os.path.exists(os.path.join(SHARD, "s_%05d_%05d.npy" % c)))
    print("pinned shards: %d total, %d to do (%d of them free from the partial "
          "full census)" % (len(chunks), len(todo), reuse), flush=True)
    t0 = time.time()
    if todo:
        with mp.Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(run_pshard, todo)):
                if (k + 1) % 20 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print("  %d/%d  %.0f s  eta %.0f s"
                          % (k + 1, len(todo), el,
                             el * (len(todo) - k - 1) / (k + 1)), flush=True)
    parts = [np.load(os.path.join(PSHARD, "p_%05d_%05d.npy" % c)) for c in chunks]
    res = np.concatenate(parts)
    res = res[np.argsort(res[:, 0])]
    np.save(os.path.join(OUT, "tooth_m23_pinned.npy"), res)
    print("wrote tooth_m23_pinned.npy  rows=%d  %.0f s"
          % (res.shape[0], time.time() - t0), flush=True)


def do_run(workers):
    """Resumable: each shard is a file; an interrupted run re-uses what exists."""
    import multiprocessing as mp
    os.makedirs(SHARD, exist_ok=True)
    n = 12960
    step = 60
    chunks = [(i, min(i + step, n)) for i in range(0, n, step)]
    todo = [c for c in chunks
            if not os.path.exists(os.path.join(SHARD, "s_%05d_%05d.npy" % c))]
    print("shards: %d total, %d already on disk, %d to do"
          % (len(chunks), len(chunks) - len(todo), len(todo)), flush=True)
    t0 = time.time()
    if todo:
        with mp.Pool(workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(run_shard, todo)):
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print("  %d/%d shards  %.0f s elapsed  eta %.0f s"
                          % (k + 1, len(todo), el,
                             el * (len(todo) - k - 1) / (k + 1)), flush=True)
    parts = [np.load(os.path.join(SHARD, "s_%05d_%05d.npy" % c)) for c in chunks]
    res = np.concatenate(parts)
    res = res[np.lexsort((res[:, 1], res[:, 0]))]
    np.save(os.path.join(OUT, "tooth_m23.npy"), res)
    print("wrote %s  rows=%d  %.0f s" % (os.path.join(OUT, "tooth_m23.npy"),
                                         res.shape[0], time.time() - t0), flush=True)


def pct(vals, x):
    vals = np.asarray(vals)
    return 100.0 * ((vals < x).sum() + 0.5 * (vals == x).sum()) / len(vals)


def do_report(pinned=False):
    if pinned:
        res = np.load(os.path.join(OUT, "tooth_m23_pinned.npy"))
    else:
        res = np.load(os.path.join(OUT, "tooth_m23.npy"))
    space19 = [list(range(1, (q - 1) // 2 + 1)) for q in GEARS19]
    vecs = list(itertools.product(*space19))
    true19 = [1, 1, 2, 2, 3, 3]
    ti = vecs.index(tuple(true19))
    gate(res.shape[0] == (12960 if pinned else 142560),
         "%d rows emitted" % res.shape[0])
    if pinned:
        gate(bool((res[:, 1] == TWIN_V23).all()),
             "every row has v_23 pinned to the twin's value %d" % TWIN_V23)
        gate(bool(np.array_equal(res[:, 0], np.arange(12960))),
             "the 12,960 m19 tooth vectors are each present exactly once")
    sel = (res[:, 0] == ti) & (res[:, 1] == 4)
    gate(int(sel.sum()) == 1, "the twin m23 member (v23 = 4) is present exactly once")
    twin = res[sel][0]
    F19t, F219t, F23t, F223t = int(twin[2]), int(twin[3]), int(twin[4]), int(twin[5])
    gate(F19t == 25, "twin F(m19) = 25 reproduces round 27")
    print("\n=== m23 rung, exhaustive over %s ==="
          % ("the PINNED family (B): all 12,960 m19 tooth vectors with v_23 "
             "fixed at the twin's own value 4" if pinned
             else "the FULL family |V(23)| = 142,560"))
    print("twin: F(19)=%d F_2(19)=%d F(23)=%d F_2(23)=%d" % (F19t, F219t, F23t, F223t))
    for name, col, tv in (("F(m23)", 4, F23t), ("F_2(m23)", 5, F223t)):
        v = res[:, col]
        print("  %-9s twin %-4d min %-4d median %-6.1f max %-4d  percentile %5.1f%%"
              % (name, tv, v.min(), float(np.median(v)), v.max(), pct(v, tv)))
    inc = res[:, 4] - res[:, 3]
    slack = res[:, 4] - res[:, 2] - Q23
    for name, v, tv in (("increment F(23)-F_2(19)", inc, F23t - F219t),
                        ("budget slack F(23)-F(19)-23", slack, F23t - F19t - Q23)):
        print("  %-27s twin %-5d min %-5d median %-7.1f max %-5d  percentile %5.1f%%"
              % (name, tv, v.min(), float(np.median(v)), v.max(), pct(v, tv)))
    # increment law over the family: s_min = min(2 v23, 23 - 2 v23)
    smin = np.minimum(2 * res[:, 1], Q23 - 2 * res[:, 1])
    viol = inc > smin
    print("  increment law F(23)-F_2(19) <= s_min(v23): violated by %d / %d = %.1f%%"
          % (int(viol.sum()), len(viol), 100.0 * viol.mean()))
    print("  twin: increment %d vs s_min %d -> %s"
          % (F23t - F219t, int(np.minimum(2 * 4, Q23 - 8)),
             "HOLDS" if F23t - F219t <= 15 else "FAILS"))
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.gate:
        do_gate()
    if a.pinned:
        do_pinned(a.workers)
    if a.run:
        do_run(a.workers)
    if a.report:
        do_report(pinned=a.pinned)
    return 0


if __name__ == "__main__":
    sys.exit(main())

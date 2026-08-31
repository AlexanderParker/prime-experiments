"""Round 28 (mechanic): Y_5 at ANY machine, by a fully streamed pass.

research/y5_m29_r28.py holds the whole gap sequence in memory (215 MB at m29).
That does not reach machine 31, whose period is 3.34e10 slots with 6.23e9
openings - the gap array alone would be 6.2 GB.  But nothing needs the array:
the only outputs are the DISTINCT 5-tuples, and there are of order a million of
them.  So sieve in blocks, carry the last four gaps across the block boundary,
pack each 5-window into one int64 (gaps at m31 are <= 58 < 64, so 5 x 6 bits
fit) and accumulate the distinct keys.  Memory is the block plus the key set.

VALIDATION FIRST (rule: no tool's numbers are used before it reproduces a known
anchor).  Run at m29 it must reproduce y5_m29_r28.py exactly: 208,668 distinct
5-tuples, X_5 = 9, Y_5 = 30, and an induced 4-tuple dictionary equal to the
round-25 full-period census.

usage: <venv>/python research/y5_stream_r28.py 29        (validation)
       <venv>/python research/y5_stream_r28.py 31
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from dict_transfer import load_dict                    # noqa: E402
from y5_r28 import saturated                           # noqa: E402

Y = int(sys.argv[1])
GEARS = [p for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37) if p <= Y]
P = 1
for q in GEARS:
    P *= q
NOPEN = 1
for q in GEARS:
    NOPEN *= (q - 2)
F1 = {29: 43, 31: 58}[Y]
SH = 6                                   # gaps < 64 at m29 and m31
assert F1 < (1 << SH)
# ROUND-28: block size is a COST/MEMORY dial and it is now an argument.
# The first m31 attempt ran at 1<<27 and died silently with no output at
# all - my own standing rule 33 (a job with no progress stride is
# indistinguishable from a hang) violated by a tool I wrote this round.
BLOCK = 1 << (int(sys.argv[2]) if len(sys.argv) > 2 else 27)
ONSET = {29: 41, 31: 53}[Y]              # onset(M -> next), C39


def window(keys, buf):
    """add every 5-window of `buf` (an int64 gap array) to `keys`."""
    if len(buf) < 5:
        return
    k = np.zeros(len(buf) - 4, np.int64)
    for t in range(5):
        k = (k << SH) | buf[t:len(buf) - 4 + t]
    keys.update(np.unique(k).tolist())


def stream():
    """(distinct packed 5-tuples, wrap gap, first gap) - CYCLICALLY CLOSED."""
    keys = set()
    carry = np.zeros(0, np.int64)         # trailing gaps not yet windowed
    head4 = None                          # the period's FIRST four gaps
    first_open = None
    first_gap = None
    tail = None
    ngap = gsum = gmax = 0
    import time
    t0 = time.time()
    nb = (P + BLOCK - 1) // BLOCK
    stride = max(1, nb // 20)
    for bi, lo in enumerate(range(0, P, BLOCK)):
        if bi % stride == 0:
            print("    block %d/%d  gaps %d  distinct 5-tuples %d  t=%.0fs"
                  % (bi, nb, ngap, len(keys), time.time() - t0), flush=True)
        hi = min(lo + BLOCK, P)
        ex = np.zeros(hi - lo, bool)
        for q in GEARS:
            u = pow(6, -1, q)
            for r in (u % q, (-u) % q):
                ex[(r - lo) % q::q] = True
        op = np.flatnonzero(~ex).astype(np.int64) + lo
        del ex
        if first_open is None:
            first_open = int(op[0])
        if tail is not None:
            op = np.concatenate([[tail], op])
        g = np.diff(op)
        tail = int(op[-1])
        del op
        if first_gap is None:
            first_gap = int(g[0])
        ngap += len(g)
        gsum += int(g.sum())
        gmax = max(gmax, int(g.max()))
        buf = np.concatenate([carry, g])
        del g
        if head4 is None and len(buf) >= 4:
            head4 = buf[:4].copy()
        window(keys, buf)
        carry = buf[-4:] if len(buf) >= 4 else buf
        del buf
    # CLOSE THE PERIOD.  The wrap gap runs from the last opening back to the
    # first one a period later; then the four 5-windows that STRADDLE the seam
    # are exactly the windows of  carry ++ [wrap] ++ head4.
    wrap = P + first_open - tail
    ngap += 1
    gsum += wrap
    gmax = max(gmax, wrap)
    window(keys, np.concatenate([carry, [wrap], head4]))
    assert ngap == NOPEN, (ngap, NOPEN)
    assert gsum == P, (gsum, P)
    assert first_open == 0, first_open
    assert wrap == first_gap, ("wrap gap must equal the FIRST GAP",
                               wrap, first_gap)
    assert gmax == F1, (gmax, F1)
    return keys, wrap, first_gap


def main():
    print("machine %d: period %d, openings %d - FULLY STREAMED"
          % (Y, P, NOPEN), flush=True)
    keys, wrap, first = stream()
    print("  cyclic close asserted (N gaps, sum = P, wrap = first = %d, "
          "max = %d = F(%d))" % (wrap, F1, Y), flush=True)
    print("  the four 5-windows straddling the seam ARE included (the windows "
          "of carry ++ [wrap] ++ head4),\n  so this is the EXACT 5-tuple "
          "dictionary, not a subset", flush=True)
    print("  distinct 5-tuples found: %d" % len(keys), flush=True)

    M6 = (1 << SH) - 1

    def unpack(k):
        return tuple((k >> (SH * (4 - i))) & M6 for i in range(5))

    # PERSIST the dictionary - it is a new object at m31 and nothing else on
    # disk can rebuild it without repeating the full-period pass.
    out = os.path.join(DATA, "r28", "gap_tuples_%d_5.csv" % Y)
    rows = sorted(unpack(k) for k in keys)
    rs = set(rows)
    assert all(t[::-1] in rs for t in rows), "NOT REVERSE-CLOSED"
    with open(out, "w") as fh:
        fh.write("g1,g2,g3,g4,g5" + chr(10))
        for t in rows:
            fh.write(",".join(map(str, t)) + chr(10))
    print("  wrote %s (%d tuples, reverse-closed, max span %d = F_5(%d))"
          % (out, len(rows), max(sum(t) for t in rows), Y), flush=True)

    D4 = set(load_dict(os.path.join(DATA, "gap_tuples_%d_4.csv" % Y)))
    ind = set()
    for k in keys:
        v = unpack(k)
        ind.add(v[:4])
        ind.add(v[1:])
    print("  CONTROL: induced 4-tuple dictionary %d vs the full-period census "
          "%d -> %s" % (len(ind), len(D4),
                        "EQUAL" if ind == D4 else "DIFFER by %d"
                        % len(D4 ^ ind)), flush=True)

    by_pref = defaultdict(list)
    for b in D4:
        by_pref[b[:3]].append(b[3])
    bestX = bestY = None
    witX = witY = None
    ncl = nsat = 0
    for a in D4:
        for last in by_pref.get(a[1:], ()):
            ncl += 1
            t = a + (last,)
            k = 0
            for v in t:
                k = (k << SH) | v
            if k in keys:
                continue
            sp = sum(t)
            if bestX is None or sp < bestX:
                bestX, witX = sp, t
            if saturated(t, Y) is not None:
                nsat += 1
                continue
            if bestY is None or sp < bestY:
                bestY, witY = sp, t
    print("\n  order-4 closure admits %d 5-walks; %d unrealised ones are "
          "phase-saturated" % (ncl, nsat))
    print("  X_5(%d) = %s  witness %s" % (Y, bestX, witX))
    print("  Y_5(%d) = %s  witness %s" % (Y, bestY, witY))
    print("  onset(%d -> next) = %d   ->   onset / Y_5 = %.3f"
          % (Y, ONSET, ONSET / bestY))


if __name__ == "__main__":
    main()

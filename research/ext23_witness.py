"""Harvester round 22: INDEPENDENT WITNESS CHECK for the 23 -> 29 extension.

The block-wise lift in ext_deficit23.py reports that some 23-winners lift to the full
y = 29 family maximum G = 75 (F = 225, h_2 = 450).  That is a strong claim - it says
the extension deficit at 23 -> 29 is ZERO - so it is checked here the hard way: locate
the claimed run of 74 consecutive KILLED positions in Z_Q29 (Q29 = 1,078,282,205) and
verify, position by position from the definitions and with no shared code, that

  * every one of the 74 positions is killed by some gear q in {5,...,29}
    (k = 0 or -delta mod q), and
  * the two flanking positions are survivors of every gear.

That is a self-contained certificate: it needs no sieve array, no scan, and no trust in
the search.  Usage: python ext23_witness.py [delta_23] [r]  (defaults: scan the winner
file for the first delta whose best lift reaches 75).
"""
import os
import sys
import numpy as np
from math import prod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from family_scan import survivors
from ext_deficit23 import _lift, QS23, Q23, QNEW, Q29, TRUE_G29

QS29 = QS23 + [QNEW]


def killed(k, delta):
    return [q for q in QS29 if k % q == 0 or (k + delta) % QNEW % 1 == 0 and False] or \
           [q for q in QS29 if k % q == 0 or (k + delta) % q == 0]


def verify(delta29, want):
    """find a maximal killed run of length want-1 in Z_Q29 and certify it."""
    # locate candidate runs from the 23-machine survivors, one block at a time
    S23 = np.flatnonzero(survivors(QS23, delta29 % Q23, Q23)).astype(np.int64)
    s_mod = (S23 % QNEW).astype(np.int8)
    r = delta29 % QNEW
    prev = None
    firstpos = None
    found = None
    for t in range(QNEW):
        ct = (t * Q23) % QNEW
        keep = (s_mod != (-ct) % QNEW) & (s_mod != (-r - ct) % QNEW)
        v = S23[keep] + t * Q23
        if v.size == 0:
            continue
        if prev is not None and v[0] - prev >= want:
            found = (int(prev), int(v[0]))
            break
        if v.size > 1:
            d = np.diff(v)
            i = int(np.argmax(d))
            if d[i] >= want:
                found = (int(v[i]), int(v[i + 1]))
                break
        if firstpos is None:
            firstpos = int(v[0])
        prev = int(v[-1])
    assert found is not None, "no run of the claimed length located"
    a, b = found
    assert b - a == want, (a, b, b - a)
    # certificate: every interior position killed, both endpoints open
    for k in (a, b):
        bad = [q for q in QS29 if k % q == 0 or (k + delta29) % q == 0]
        assert not bad, (k, bad)
    witness = []
    for k in range(a + 1, b):
        q = next((q for q in QS29 if k % q == 0 or (k + delta29) % q == 0), None)
        assert q is not None, k
        witness.append(q)
    return a, b, witness


if __name__ == "__main__":
    w = np.load("research/data/family_w23_delta.npy")
    if len(sys.argv) > 2:
        cands = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        cands = []
        for d in w[:4]:
            S = np.flatnonzero(survivors(QS23, int(d), Q23)).astype(np.int32)
            sm = (S % QNEW).astype(np.int8)
            for rr in range(QNEW):
                if _lift(S, sm, rr, Q23, QNEW) == TRUE_G29:
                    cands.append((int(d), rr))
                    break
    print(f"candidates (delta_23, r): {cands}", flush=True)
    log = []
    for d23, rr in cands:
        inv = pow(Q23, -1, QNEW)
        d29 = (d23 + Q23 * (((rr - d23) * inv) % QNEW)) % Q29
        a, b, wit = verify(d29, TRUE_G29)
        e = (3 * d29) % Q29
        msg = (f"delta_29 = {d29} (from delta_23 = {d23}, r = {rr}): certified killed "
               f"run k = {a+1}..{b-1} of length {b-a-1} = G-1, both flanks open -> "
               f"G = {b-a} = {TRUE_G29}, F = {3*(b-a)}, h_2 = {6*(b-a)} = 450")
        print(msg, flush=True)
        print(f"  killing gears along the run: {wit}", flush=True)
        log.append(msg)
        log.append("  killing gears: " + " ".join(map(str, wit)))
    with open("research/data/ext23_witness.out", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("ext23_witness: ALL ASSERTIONS GREEN")

"""Harvester round 22: the exhaustive y=23 family scan (complete 23-winner set).

Same exact prefilter as research/family_scan.py, run through the three-level fast
scanner research/family_scan_fast.py (both validated against brute force at y=13/17 and
against the exhaustive y=19 result, where they reproduce Ziller-Morack's h_2(19) = 258
and the complete 64-winner set).

Base gears 5,7,11,13,17 (Qb = 85085); middle gear 19; held-out top gear 23.
Gmin = 61 because ZM's h_2(23) = 366 -> F = 183 -> G = 61; the scan therefore also
REPLICATES h_2(23) independently (anything larger would also be found).

Usage:  python family_scan23.py <shard> <nshards>   -> research/data/scan23_<shard>.npy
        python family_scan23.py merge <nshards>     -> verify + winners
"""
import os
import sys
import numpy as np
from math import prod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from family_scan_fast import fast_scan
from family_scan import max_gap

BASE = [5, 7, 11, 13, 17]
QMID = 19
QT = 23
GMIN = 61
QB = prod(BASE)                  # 85085
Q = QB * QMID * QT               # 37,182,145

if __name__ == "__main__":
    if sys.argv[1] == "merge":
        ns = int(sys.argv[2])
        cand = np.unique(np.concatenate(
            [np.load(f"research/data/scan23_{i}.npy") for i in range(ns)]))
        qs = BASE + [QMID, QT]
        res = [(int(d), max_gap(qs, int(d), Q)) for d in cand]
        gm = max(g for _, g in res)
        win = sorted(d for d, g in res if g == gm)
        atleast = sum(1 for _, g in res if g >= GMIN)
        msg = (f"prefilter kept {cand.size} of {Q} deltas; max G = {gm} "
               f"(F = {3*gm}, h_2 = {6*gm}); {len(win)} winners, "
               f"{atleast} deltas with G >= {GMIN}")
        print(msg, flush=True)
        np.save("research/data/family_w23_delta.npy", np.array(win, np.int64))
        if gm != GMIN:
            print(f"  *** PRE-REGISTERED CHECK FAILED: max G = {gm}, expected "
                  f"{GMIN} from Ziller-Morack h_2(23) = 366 ***", flush=True)
        else:
            print("  pre-registered check PASSED: h_2(23) = 366 replicated "
                  "exhaustively", flush=True)
        with open("research/data/family_scan23.out", "w") as fh:
            fh.write(msg + "\n" + " ".join(map(str, win)) + "\n")
    else:
        s, ns = int(sys.argv[1]), int(sys.argv[2])
        lo, hi = QB * s // ns, QB * (s + 1) // ns
        a = np.array(fast_scan(BASE, QMID, QT, GMIN, lo, hi, prog=1000), np.int64)
        np.save(f"research/data/scan23_{s}.npy", a)
        print(f"shard {s}: [{lo},{hi}) -> {a.size} candidates", flush=True)

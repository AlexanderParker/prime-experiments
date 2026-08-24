"""Harvester round 22: reconcile the DIFFERENCE-representation winner counts with
Ziller-Morack's SEQUENCE counts.

ZM (arXiv:1706.03668, full_details.pdf Table 1, column nseq - "number of sequences of
maximum length", with exhaustive lists in the ancillary files remainders_2.txt /
permutations_2.txt / moduli_2.txt) report

    p_n     5  7  11  13  17  19  23  29  31  37  41  43  47  53 ...
    nseq    1  6   1   1   4   2   2  14   8   4   1   8   2  16 ...

The round-22 exhaustive scans count WINNING DIFFERENCES (deltas), which is a different
object: 8, 16, 64, 64 at y = 11, 13, 17, 19.  This script computes the covering
PATTERN of each record window - for every killed position, the smallest gear that kills
it - and counts distinct patterns up to reversal, which is what ZM's nseq counts.
"""
import os
import sys
import numpy as np
from math import prod
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from family_scan import survivors, scan

LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def patterns(qs, delta, Q, G):
    """for every maximal killed run of length G-1, the tuple of minimal killing
    gears, plus its reverse."""
    idx = np.flatnonzero(survivors(qs, delta, Q)).astype(np.int64)
    d = np.diff(np.append(idx, idx[0] + Q))
    out = []
    for i in np.flatnonzero(d == G):
        st = int(idx[i]) + 1
        pat = []
        for j in range(G - 1):
            k = (st + j) % Q
            pat.append(min(q for q in qs if k % q == 0 or (k + delta) % q == 0))
        out.append(tuple(pat))
    return out


def main():
    say("=== winner counts: DIFFERENCE representation vs ZM's SEQUENCE count ===")
    say("   y   winning deltas   record windows   distinct patterns   "
        "up to reversal   ZM nseq")
    zm_nseq = {11: 1, 13: 1, 17: 4, 19: 2, 23: 2}
    rows = []
    JOBS = [([5, 7], 11, 11, None), ([5, 7, 11], 13, 25, None),
            ([5, 7, 11, 13], 17, 32, None),
            ([5, 7, 11, 13, 17], 19, 43, "research/data/family_w19_delta.npy"),
            ([5, 7, 11, 13, 17, 19], 23, 61, "research/data/family_w23_delta.npy")]
    for qs_pre, qt, G, cache in JOBS:
        qs = qs_pre + [qt]
        Q = prod(qs)
        if cache and os.path.exists(cache):
            wins = [int(x) for x in np.load(cache)]
        elif cache:
            say(f"  y={qt}: winner set not computed yet - skipped")
            continue
        else:
            wins = [d for d, g in scan(qs_pre, qt, G)]
        pats, wins_windows = set(), 0
        for d in wins:
            ps = patterns(qs, d, Q, G)
            wins_windows += len(ps)
            pats.update(ps)
        canon = {min(p, p[::-1]) for p in pats}
        rows.append((qt, len(wins), wins_windows, len(pats), len(canon),
                     zm_nseq.get(qt)))
        say(f"  {qt:>3}   {len(wins):>13}   {wins_windows:>14}   {len(pats):>17}"
            f"   {len(canon):>14}   {zm_nseq.get(qt):>7}")
    for qt, nw, nwin, npat, ncan, nseq in rows:
        assert npat == nseq, (qt, npat, nseq)
        if npat == 1:
            assert ncan == 1, qt        # ZM: the single sequences are self-symmetric
    say("  EXACT MATCH at every computed y: the number of DISTINCT COVERING PATTERNS of "
        "the record windows equals ZM's nseq (reverses counted separately, exactly as "
        "they state).  y=11 and y=13 have a single self-symmetric pattern - ZM's own "
        "remark that the single sequences at n = 5, 6 are self-symmetric by default.")
    say("  So the two data sets are the same object in two representations, and this is "
        "an independent cross-check of both: many differences, few patterns.")
    with open("research/data/zm_seq_reconcile.out", "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print("zm_seq_reconcile: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

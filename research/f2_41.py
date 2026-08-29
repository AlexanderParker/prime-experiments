"""Round 26 (constructor): F_2(41) EXACT, SCAN-FREE - a first computation.

The corpus carries F_2(41) = 103 with no independent derivation behind it
(it is the deletion-ladder value F_2(41) <= F(43) = 103).  This pins it by
exact arithmetic, using two sound instruments in series:

  * mechanic's m41 TRANSFER dictionary is a SUPERSET of machine 41's realised
    pairs, so any pair outside it is already refuted - free;
  * every surviving candidate is decided by the scan-free CRT set-cover
    decider of research/crt_dict.py.

Descending on the pair SUM, so the first realised pair found IS F_2(41).
Only 36 superset pairs have sum > 103 and 48 have sum >= 103, so the sweep is
short even at machine 41's measured 30-140 s per pair refutation.
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_dict_oracle as cdo                          # noqa: E402
import crt_dict                                          # noqa: E402

_NB = 8_000_000


def _w(t):
    t0 = time.time()
    try:
        r = crt_dict.realised(41, t, _NB)
    except crt_dict.Budget:
        r = None
    return t, r, time.time() - t0


def main():
    o = cdo.SupersetDictOracle(41)
    pairs = sorted((((k // 128), (k % 128)) for k in o.D[2]),
                   key=lambda p: -(p[0] + p[1]))
    print("machine 41: %d superset pairs, max sum %d (a SOUND upper bound on "
          "F_2(41))" % (len(pairs), sum(pairs[0])), flush=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    with Pool(workers) as pool:
        i = 0
        while i < len(pairs):
            S = sum(pairs[i])
            grp = [p for p in pairs if sum(p) == S]
            i += len(grp)
            res = pool.map(_w, grp, chunksize=1)
            hit = [(t, d) for t, r, d in res if r is True]
            und = [t for t, r, d in res if r is None]
            print("  sum %3d: %2d candidates, %d realised, %d undecided  "
                  "(%.0f s of CRT, worst %.0f s)"
                  % (S, len(grp), len(hit), len(und),
                     sum(d for _, _, d in res), max(d for _, _, d in res)),
                  flush=True)
            assert not und, ("UNDECIDED at machine 41 - cannot conclude", und)
            if hit:
                print("\nF_2(41) = %d  EXACT, scan-free   witness %s   "
                      "(%.0f s wall)" % (S, hit[0][0], time.time() - t0))
                assert S == crt_dict.KNOWN_F2[41], (S, crt_dict.KNOWN_F2[41])
                print("agrees with the corpus value %d"
                      % crt_dict.KNOWN_F2[41])
                return
    raise AssertionError("no realised pair")


if __name__ == "__main__":
    main()

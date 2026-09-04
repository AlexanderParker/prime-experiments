"""Round 31 (constructor): L_pad(47) - the NON-BARE half of L at machine 47.

R98 pinned L(47) = 4 with the bare alternation (18,35,18,35), and refuted the
only two non-bare length-4 survivors ((35,18,35,53), (35,18,53,35)).  What is
not on record is whether any NON-BARE word of length 3 is realised at m47, i.e.
the value of L_pad(47) in the decomposition L = max(L_bare, L_pad).

Alphabet at m47 (q' = 53, a = 18, b = 35, F = 118): {18, 35, 53, 71, 88, 106}.
T3 alternation + the exact caps F_2 = 134, F_3 = 145 + phase saturation at every
gear leave TWELVE non-bare length-3 words; mirror canonicalisation (Lateral's
record-mirror theorem: the realised legal words are closed under reversal)
leaves SEVEN instances.  This decides all seven by crt_dict.decide_cover.

Usage:  uv run python research/lpad47_r31.py [--nodes N] [--workers 4]
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                              # noqa: E402

# the 7 mirror representatives of the 12 survivors
WORDS = [(18, 35, 53), (18, 53, 35), (35, 18, 53), (35, 18, 88),
         (35, 53, 53), (53, 35, 53), (35, 71, 35)]
MIRROR = {(18, 35, 53): (53, 35, 18), (18, 53, 35): (35, 53, 18),
          (35, 18, 53): (53, 18, 35), (35, 18, 88): (88, 18, 35),
          (35, 53, 53): (53, 53, 35), (53, 35, 53): (53, 35, 53),
          (35, 71, 35): (35, 71, 35)}


def job(args):
    w, nodes = args
    t0 = time.time()
    try:
        return w, crt_dict.realised(47, w, node_budget=nodes), time.time() - t0
    except Exception as e:                                    # noqa
        return w, None, time.time() - t0


def main():
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    nodes = opt("--nodes", 60_000_000)
    workers = opt("--workers", 4)
    print("machine 47 -> q' = 53: deciding the %d mirror representatives of "
          "the 12 non-bare\nlength-3 survivors, node budget %d, %d workers"
          % (len(WORDS), nodes, workers), flush=True)
    t0 = time.time()
    with Pool(workers) as p:
        res = sorted(p.imap_unordered(job, [(w, nodes) for w in WORDS]))
    tag = {True: "REALISED", False: "refuted", None: "UNDECIDED"}
    for w, ok, dt in res:
        print("  %-16s %-10s %.0f s   (mirror %s)"
              % (str(w), tag[ok], dt, MIRROR[w]), flush=True)
    yes = [w for w, ok, _ in res if ok]
    und = [w for w, ok, _ in res if ok is None]
    print()
    if yes:
        print("==> L_pad(47) >= 3, realised by %s (and its mirror)" % (yes[0],))
    elif not und:
        print("==> L_pad(47) <= 2 EXACT: all 12 non-bare length-3 words at m47 "
              "are refuted,\n    0 undecided.  So L(47) = 4 is carried by the "
              "BARE alternation alone and\n    L_pad(47) <= 2.")
    else:
        print("==> NOT DECIDED: %s undecided at the budget" % (und,))
    print("[%.0f s]" % (time.time() - t0))


if __name__ == "__main__":
    main()

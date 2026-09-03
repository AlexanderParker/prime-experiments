"""Round 29 (constructor): PIN L(47), the depth cap of the eleventh rung.

By R89, J_max(M) = L(M) + 2 and A_kill(M -> q') = L(M) + 1, with L the length of
the longest REALISED word-legal letter word.  At machine 47 (q' = 53, c = 9,
a = 18, b = 35) the legal letters <= F(47) = 118 are {18, 35, 53, 71, 88, 106},
and after T3 alternation, the exact spectrum caps F_1 = 118, F_2 = 134, F_3 = 145
(all on record), the caps F_4 <= F_3 + F_1, F_5 <= F_4 + F_1, phase saturation
and mirror canonicalisation, the ENTIRE decision is FOUR instances:

    length 4 : (18,35,18,35)  (35,18,35,53)  (35,18,53,35)      span 106/141/141
    length 5 : (35,18,35,18,35)                                 span 141

Mechanic's A_kill(47 -> 53) = 5 EXACT (round 25) predicts, via R89, that at least
one length-4 word is REALISED and the length-5 word is REFUTED.  That is a sharp
cross-check on a recorded value at a cost of four CRT calls.

Usage:  uv run python research/l47_words_r29.py
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                              # noqa: E402

WORDS = [(18, 35, 18, 35), (35, 18, 35, 53), (35, 18, 53, 35),
         (35, 18, 35, 18, 35)]
NODES = 40_000_000


def job(w):
    t0 = time.time()
    try:
        return w, crt_dict.realised(47, w, node_budget=NODES), time.time() - t0
    except Exception:
        return w, None, time.time() - t0


if __name__ == "__main__":
    print("machine 47 -> q' = 53 ; deciding %d candidate legal words at a "
          "%d-node budget" % (len(WORDS), NODES), flush=True)
    with Pool(4) as p:
        res = sorted(p.imap_unordered(job, WORDS))
    tag = {True: "REALISED", False: "refuted", None: "UNDECIDED"}
    for w, ok, dt in res:
        print("  %-22s %-10s %.0f s" % (str(w), tag[ok], dt), flush=True)
    four = [(w, ok) for w, ok, _ in res if len(w) == 4]
    five = [(w, ok) for w, ok, _ in res if len(w) == 5]
    print()
    if any(ok for _, ok in four) and all(ok is False for _, ok in five):
        print("==> L(47) = 4 EXACT, so J_max(47) = 6 and A_kill(47 -> 53) = 5")
        print("    - Mechanic's round-25 value CONFIRMED by an independent route")
        print("    (four CRT calls, no census, no period).")
    elif all(ok is False for _, ok in four + five):
        print("==> L(47) <= 3, which CONTRADICTS A_kill(47 -> 53) = 5.")
    else:
        print("==> not decided: %s"
              % [(w, tag[ok]) for w, ok, _ in res if ok is None])

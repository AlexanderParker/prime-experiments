"""Round 28 (mechanic): F_2(59) by the deepest lap-phase transfer yet (r = 8).

WHAT IT IS.  F_2(M) is the largest sum of two consecutive gaps of machine M.
It is computed here with the floor-1 lap-phase transfer - Q_J(TARGET; 1) =
F_J(TARGET) - from machine 23's period (37,182,145 slots) with EIGHT new gears
{29,31,37,41,43,47,53,59}, so machine 59 (period 1.96e19) is never built.  The
period ratio bought is 29*31*...*59 = 8.6e12.

WHY IT MATTERS.  The deletion ladder (K3) gives F_2(M) <= F(M + one more gear),
so F_2(59) is a LOWER bound on F(61) - it extends the corpus F ladder one rung
past where this lane took it in round 27 - and it is the left-hand side of the
manager's increment law at the next step: F(61) - F_2(59) <= s_min(61) = 20.

SEED AND CAP.  Seed at F(59) (which the round-28 pin computes exactly): every
2-window of span <= F(59) is unresolved and irrelevant, since F_2 >= F always.
Cap 220: the trivial cap is 2 F(59) and the measured F_2/F ratios are 1.02-1.17
across m31..m53, so 220 sits far above any plausible value while keeping the
walk cheap.  As always, the >= direction is an exhibited witness and is
unconditional; the <= direction is conditional on the span cap.

Usage:  python research/f2_59_r28.py run F59 [workers] [cap]
        python research/f2_59_r28.py show
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "r28")
J5 = os.path.join(HERE, "j5_multi.py")
PY = sys.executable
NOPEN = 7952175


def main():
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "show":
        for f in sorted(os.listdir(OUT)):
            if f.startswith("f2_59") and f.endswith(".log"):
                txt = open(os.path.join(OUT, f), errors="replace").read()
                print("%-26s %s %s" % (f, "DONE" if "scan complete" in txt
                                       else "RUNNING",
                                       re.findall(r"max over J = (\d+)", txt)))
        return
    LO = int(sys.argv[2])
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    HI = int(sys.argv[4]) if len(sys.argv) > 4 else 220
    edges = [i * NOPEN // W for i in range(W)] + [NOPEN]
    procs = []
    for i in range(W):
        log = os.path.join(OUT, "f2_59_s%d_c%d_w%d.log" % (LO, HI, i))
        fh = open(log, "w")
        procs.append((subprocess.Popen(
            [PY, "-u", J5, "23", "29,31,37,41,43,47,53,59", "61",
             "seed%d" % LO, str(HI), "2", "1", "plain",
             str(edges[i]), str(edges[i + 1])],
            stdout=fh, stderr=subprocess.STDOUT), fh, log))
        time.sleep(3)
    print("launched %d workers: F_2(59) in (%d, %d]" % (W, LO, HI), flush=True)
    t0 = time.time()
    for p, fh, _ in procs:
        p.wait()
        fh.close()
    best, wit = LO, None
    for _, _, log in procs:
        txt = open(log, encoding="utf-8", errors="replace").read()
        assert "scan complete" in txt, ("worker did not finish", log)
        for m in re.finditer(r"max over J = (\d+)", txt):
            if int(m.group(1)) > best:
                best = int(m.group(1))
        for m in re.finditer(r"J=(\d+): k=([\d,]+) span=(\d+)", txt):
            if int(m.group(3)) >= best:
                wit = m.group(0)
    el = time.time() - t0
    print("\nF_2(59) = %d   %s   [%.0fs x %d workers, span cap %d]"
          % (best, wit or "(EMPTY BAND - the seed stands)", el, W, HI))
    print("  deletion ladder: F(61) >= F_2(59) = %d, unconditional if the "
          "witness verifies at machine 59" % best)
    print("  increment law at 59 -> 61: predicts F(61) <= %d + 20 = %d"
          % (best, best + 20))


if __name__ == "__main__":
    main()

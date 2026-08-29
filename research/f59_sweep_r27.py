"""Round 27 (mechanic): F(59) EXACT by a DESCENDING-BAND word-legal sweep.

THE VEHICLE.  Constructor's round-26 attainment theorem makes the record law
an equality:  F(M + q') = max_J Q*_J(M; legal for q').  So machine 59's
maximal gap is computable on MACHINE 23's period by the lap-phase transfer
with r = 7 new gears {29,31,37,41,43,47,53} and the word-legal predicate for
q' = 59 - the machine 59 itself (period 1.96e19) is never built.

WHY BANDS.  j5_multi's cost is dominated by PHASE EXPANSION of windows whose
span exceeds the running best, so a run seeded at `lo` with span cap `hi`
expands only windows of span in (lo, hi] and answers exactly

    "the largest word-legal window span in (lo, hi], or `lo` if there is none".

Running bands in DESCENDING order therefore prices each band separately and
STOPS at the first non-empty one, whose maximum IS F(59) - every larger span
having already been refuted.  A single run seeded at the floor would pay for
every band at once and give no intermediate verdict.

Bands compose with no gap by construction: band i is (lo_i, hi_i] with
hi_{i+1} = lo_i.

Established going in (round 26): F(59) >= F_2(53) = 159 unconditionally
(deletion ladder on an exhibited witness).  (D) at 53->59 needs F(59) <= 204.

Usage:
  python research/f59_sweep_r27.py run  [workers]
  python research/f59_sweep_r27.py show
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "r27")
J5 = os.path.join(HERE, "j5_multi.py")
PY = sys.executable

NOPEN = 7952175                  # openings in machine 23's period
FLOOR = 158                      # F(59) >= 159 (r26), so seed one below
# The top band (203, 260] is run separately as f59_A_w*.log (it is the (D)
# decision on its own: budget = 204).  This sweep continues below it.
BANDS = [(193, 204), (183, 194), (173, 184),
         (168, 174), (163, 169), (FLOOR, 164)]


def run_band(lo, hi, W):
    tag = "f59_b%d_%d" % (lo, hi)
    edges = [i * NOPEN // W for i in range(W)] + [NOPEN]
    procs = []
    for i in range(W):
        log = os.path.join(OUT, "%s_w%d.log" % (tag, i))
        if os.path.exists(log):
            os.remove(log)
        fh = open(log, "w")
        procs.append((subprocess.Popen(
            [PY, "-u", J5, "23", "29,31,37,41,43,47,53", "59",
             "seed%d" % lo, str(hi), "7", "20", "legal",
             str(edges[i]), str(edges[i + 1])],
            stdout=fh, stderr=subprocess.STDOUT), fh, log))
    t0 = time.time()
    for p, fh, _ in procs:
        p.wait()
        fh.close()
    best, wit = lo, None
    for _, _, log in procs:
        txt = open(log, encoding="utf-8", errors="replace").read()
        assert "scan complete" in txt, ("worker did not finish", log)
        for m in re.finditer(r"J=(\d+): k=([\d,]+) span=(\d+)", txt):
            if int(m.group(3)) > best:
                best = int(m.group(3))
                wit = m.group(0)
        for m in re.finditer(r"max over J = (\d+)", txt):
            if int(m.group(1)) > best:
                best = int(m.group(1))
    return best, wit, time.time() - t0


def main():
    if sys.argv[1] == "show":
        for f in sorted(os.listdir(OUT)):
            if f.startswith("f59_b") and f.endswith(".log"):
                txt = open(os.path.join(OUT, f), errors="replace").read()
                mx = re.findall(r"max over J = (\d+)", txt)
                print("%-28s %s" % (f, mx))
        return
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(OUT, exist_ok=True)
    print("F(59) DESCENDING-BAND SWEEP, %d workers per band" % W, flush=True)
    print("  vehicle: max_J Q*_J(53; legal for 59) = F(59)  [record law, "
          "Constructor r26]", flush=True)
    for lo, hi in BANDS:
        best, wit, el = run_band(lo, hi, W)
        if best > lo:
            print("  band (%3d, %3d]: MAXIMUM %d  %s   [%.0fs x %d workers]"
                  % (lo, hi, best, wit or "", el, W), flush=True)
            print("\n==> F(59) = %d   (every band above is empty, so this is "
                  "the maximum)" % best, flush=True)
            print("    budget F(53) + 59 = 204 -> (D) at 53->59 %s"
                  % ("HOLDS" if best <= 204 else "FAILS"), flush=True)
            return
        print("  band (%3d, %3d]: EMPTY (no word-legal window of span in this "
              "band)   [%.0fs x %d workers]" % (lo, hi, el, W), flush=True)
    print("\n==> no word-legal window of span > %d; with the r26 theorem "
          "F(59) >= 159 this bounds F(59) <= %d" % (FLOOR, FLOOR), flush=True)


if __name__ == "__main__":
    main()

"""Round 28 (mechanic): PIN F(59) EXACTLY - the last band, at the depth the
question actually needs.

STATE GOING IN (round 27, C38):  161 <= F(59) <= 178.
  * every band above 178 is EMPTY at JMAX = 7 (four scans, all logged);
  * the J <= 3 maximum over (152, 184] is 161, exhibited at machine 53 by
    k = 2,505,673,933,219,103,747, gaps [10, 118, 33];
  * so the only spans left are (161, 178] and only at depth J >= 4.
Round 27 launched exactly that band at JMAX = 7 and KILLED it: no worker
reached its first progress stride in 35 minutes.

WHAT CHANGED - THE DEPTH CAP IS NOW A THEOREM, NOT A GUESS (rule 32).
A word-legal window of J gaps has J-1 INTERIOR OPENINGS, all deleted by one
phase of q' - i.e. a REALISED kill chain of ARITY J-1, whose word has J-2
letters (A_kill counts openings, not letters; rule 5's index trap).  So
Q*_J > 0 requires A_kill(53->59) >= J-1.  Round 27
closed that level: A_kill(53->59) = 4 EXACT, N_5 = 0.  Hence

    Q*_6 = Q*_7 = 0   and   max_{J<=5} Q*_J  =  max_J Q*_J  =  F(59)

so JMAX = 5 is not a scope choice, it is the whole question.  Measured on an
identical 20,000-index probe, alone: JMAX = 5 walks it in 57 s; JMAX = 7 does
not finish in 600 s.  The break condition is `lbmax > JMAX - 1`, so the cap
sets how deep every window is pursued.

The band answers exactly "the largest word-legal window span in (161, 178],
or 161 if there is none", and every larger span is already refuted, so its
answer IS F(59).

Usage:  python research/f59_pin_r28.py run [workers]
        python research/f59_pin_r28.py show
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

NOPEN = 7952175          # openings in machine 23's period
LO, HI, JMAX = 161, 178, 5
TAG = "f59_pin_%d_%d_J%d" % (LO, HI, JMAX)


def main():
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "show":
        best, wits = LO, []
        for f in sorted(os.listdir(OUT)):
            if not (f.startswith(TAG) and f.endswith(".log")):
                continue
            txt = open(os.path.join(OUT, f), errors="replace").read()
            done = "scan complete" in txt
            mx = re.findall(r"max over J = (\d+)", txt)
            print("%-34s %s %s" % (f, "DONE" if done else "RUNNING", mx))
            if done:
                for m in re.finditer(r"J=(\d+): k=([\d,]+) span=(\d+)", txt):
                    wits.append(m.group(0))
                for m in re.finditer(r"max over J = (\d+)", txt):
                    best = max(best, int(m.group(1)))
        print("\nbest so far %d; witnesses:" % best)
        for w in wits:
            print("   ", w)
        return

    W = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    edges = [i * NOPEN // W for i in range(W)] + [NOPEN]
    procs = []
    for i in range(W):
        log = os.path.join(OUT, "%s_w%d.log" % (TAG, i))
        fh = open(log, "w")
        procs.append((subprocess.Popen(
            [PY, "-u", J5, "23", "29,31,37,41,43,47,53", "59",
             "seed%d" % LO, str(HI), str(JMAX), "20", "legal",
             str(edges[i]), str(edges[i + 1])],
            stdout=fh, stderr=subprocess.STDOUT), fh, log))
        time.sleep(2)
    print("launched %d workers on band (%d, %d] at JMAX=%d" % (W, LO, HI, JMAX),
          flush=True)
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
    if best > LO:
        print("band (%d, %d] MAXIMUM %d  %s  [%.0fs x %d workers]"
              % (LO, HI, best, wit or "", el, W), flush=True)
    else:
        print("band (%d, %d] EMPTY  [%.0fs x %d workers]" % (LO, HI, el, W),
              flush=True)
    print("\n==> F(59) = %d   (every span above 178 refuted in r27; every span "
          "in (161,178] decided here)" % best, flush=True)
    print("    budget F(53) + 59 = 204 -> (D) at 53->59 %s"
          % ("HOLDS" if best <= 204 else "FAILS"), flush=True)
    print("    increment law: F(59) - F_2(53) = %d  vs s_min(59) = 20 -> %s"
          % (best - 159, "HOLDS" if best - 159 <= 20 else "FAILS"), flush=True)


if __name__ == "__main__":
    main()

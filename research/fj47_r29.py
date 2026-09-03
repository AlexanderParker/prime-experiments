"""Round 29 (mechanic), brief item (b): F_J(47) FOR J = 2..6, CRASH-PROOF.

WHAT.  The eleventh rung is 47 -> 53, budget F(47) + 53 = 171.  Constructor's
spectrum-plus-depth certificate needs the OLD machine's spectrum over
2 <= J <= J_max, and J_max(47) = A_kill(47 -> 53) + 1 = 6 (C23: A_kill = 5
EXACT, N_6 = 0), so Q*_7(47) is EMPTY BY THEOREM and the depth range is finite.
This driver computes F_J(47) for J = 2..6 by the floor-1 lap-phase transfer
from machine 23 with six new gears {29,31,37,41,43,47}.

SEED AND CAP, both deliberate and both stated with what they condition.
  seed 145 = F_3(47), the largest F_J(47) already on record, so the run resolves
    every J whose answer exceeds it and reports "145" for any J whose answer
    does not (and F_J is non-decreasing in J, so J = 4,5,6 are >= 145 by
    definition - the seed is exactly at the known floor, never above it).
  cap 290 = 2 F_3(47), which is at or above the SUBADDITIVITY ceiling of every
    depth in range (F_4 <= 2F_2 = 268, F_5 <= F_2 + F_3 = 279, F_6 <= 2F_3 =
    290).  Nothing here is span-conditional: a J-window of span above the cap
    would violate F_{a+b} <= F_a + F_b.

WHY SHARDED.  Round 28 lost seven workers and their launcher silently to
commit exhaustion, and round 29 lost six the same way.  Here the period is cut
into SHARDS of start-opening indices; a pool re-launches only shards whose log
does not already say "scan complete", so a kill costs at most the shards in
flight.  Shards also cap memory: j5_multi builds ext/res only over the index
window it walks.

MODE 'legal' runs the SAME shards with the WORD-LEGAL criterion instead of the
plain floor - Q*_J(47; legal for 53) rather than F_J(47) - which is the object
the attainment theorem (R68) says equals F(53) when maxed over J.  Same seed,
same cap, floor a = 2*round(53/6) = 18 (the smallest positive legal value, so
the mark-spacing pre-filter stays sound).

usage: uv run python research/fj47_r29.py run [workers] [shards] [legal]
       uv run python research/fj47_r29.py show [workers] [shards] [legal]
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEGAL = "legal" in sys.argv
SEED, CAP, JMAX = 145, 290, 6
for _a in sys.argv:                       # optional seedNNN override
    if _a.startswith("seed"):
        SEED = int(_a[4:])
TAG = ("q47" if LEGAL else "fj47") + ("" if SEED == 145 else "_s%d" % SEED)
OUT = os.path.join(HERE, "data", "r29", TAG)
J5 = os.path.join(HERE, "j5_multi.py")
PY = sys.executable
NOPEN = 7952175
FLOOR, KIND = ("18", "legal") if LEGAL else ("1", "plain")


def shard_edges(n):
    return [i * NOPEN // n for i in range(n)] + [NOPEN]


def log_of(i):
    return os.path.join(OUT, "sh%03d.log" % i)


def done(i):
    p = log_of(i)
    return os.path.exists(p) and "scan complete" in open(
        p, errors="replace").read()


def show(n):
    e = shard_edges(n)
    mx, nd = {}, 0
    for i in range(n):
        if not done(i):
            continue
        nd += 1
        txt = open(log_of(i), errors="replace").read()
        for J, v in re.findall(r"^\s+(\d)\s+(\d+)\s", txt, re.M):
            mx[int(J)] = max(mx.get(int(J), 0), int(v))
    cov = sum(e[i + 1] - e[i] for i in range(n) if done(i))
    print(f"  shards complete {nd}/{n}   start indices covered "
          f"{cov:,}/{NOPEN:,} ({100.0*cov/NOPEN:.1f}%)")
    S = "Q*" if LEGAL else "F"
    if mx:
        print("   " + "  ".join(f"{S}_{J}(47) >= {v}" if v > SEED
                                else f"{S}_{J}(47) <= {SEED} (at the seed)"
                                for J, v in sorted(mx.items())))
        top = max(mx.values())
        print(f"   max over J = {top}  vs budget F(47) + 53 = 171  ->  "
              f"{'CERTIFIES' if top <= 171 else 'FAILS by +%d' % (top - 171)}")
    return nd == n, mx


def run(workers, n):
    os.makedirs(OUT, exist_ok=True)
    e = shard_edges(n)
    todo = [i for i in range(n) if not done(i)]
    print(f"  {len(todo)} shards to run, {n - len(todo)} already complete, "
          f"{workers} workers", flush=True)
    live = []
    while todo or live:
        while todo and len(live) < workers:
            i = todo.pop(0)
            fh = open(log_of(i), "w")
            cmd = [PY, "-u", J5, "23", "29,31,37,41,43,47", "53",
                   "seed%d" % SEED, str(CAP), str(JMAX), FLOOR, KIND,
                   str(e[i]), str(e[i + 1])]
            try:
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            except OSError as ex:            # fork-table / commit exhaustion
                print(f"  shard {i}: Popen failed ({ex}); requeued", flush=True)
                fh.close()
                todo.append(i)
                time.sleep(20)
                continue
            live.append((i, p, fh))
            time.sleep(2)
        time.sleep(5)
        for rec in list(live):
            i, p, fh = rec
            if p.poll() is None:
                continue
            live.remove(rec)
            fh.close()
            if not done(i):
                print(f"  shard {i}: DIED (rc={p.returncode}); requeued",
                      flush=True)
                todo.append(i)
            else:
                ok, _ = show(n)
    print("  all shards complete" if all(done(i) for i in range(n))
          else "  STOPPED WITH SHARDS OUTSTANDING", flush=True)
    show(n)


if __name__ == "__main__":
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    if sys.argv[1] == "show":
        show(N)
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 4, N)

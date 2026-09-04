"""Round 30 (mechanic), brief item (d): THE INDEPENDENT max_J Q*_J(47; legal for 53).

Rung eleven (47 -> 53, budget F(47) + 53 = 171) is closed in round 29 only by
the attainment theorem R68 PLUS the corpus value F(53) = 145 - a control, not
a decision.  This driver computes the word-legal spectrum Q*_J(47; legal for
53), J <= 6, from MACHINE 23's period by the six-gear lap-phase transfer
(j5_multi.py, mode 'legal'), so that

    max_J Q*_J(47)  <=  171

is established with machine 53 NEVER BUILT and F(53) never consulted.

SEED 144, not 145 (standing rule 41: raise the seed to the QUESTION).  The
attainment theorem predicts max_J Q*_J(47) = F(53) = 145 with the J = 4
maximiser (70,35,18,22) (round 26's m47 window); seeding ONE below that value
makes the run two-sided - it must FIND 145 (lower half, with a witness) and
must find NOTHING above it (upper half) - and it decides the rung outright.
Any J whose true value is <= 144 is reported "<= 144 (at the seed)" and is a
bracket, not a value; that is the honest price of a seed.
Cap 290 = 2 F_3(47), at or above the subadditivity ceiling of every depth in
range (F_4 <= 2F_2 = 268, F_5 <= F_2 + F_3 = 279, F_6 <= 2F_3 = 290), so the
result is NOT span-conditional.  Depth cap 6 = J_max(47) = L(47) + 2 (R89,
L(47) = 4 exact) - Q*_7(47) is empty by theorem.

Sharded like round 29's fj47_r29.py (a kill costs only the shards in flight;
every shard writes its own log from the CHILD), children at HIGH priority
(measured ~10x on this box under other lanes' load), logs under
research/data/r30/q47_s144/.

usage: uv run python research/qstar47_r30.py run [workers] [shards]
       uv run python research/qstar47_r30.py show [workers] [shards]
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, CAP, JMAX = 144, 290, 6
OUT = os.path.join(HERE, "data", "r30", "q47_s%d" % SEED)
J5 = os.path.join(HERE, "j5_multi.py")
PY = sys.executable
NOPEN = 7952175                       # start openings of machine 23's period
FLOOR, KIND = "18", "legal"           # a = 2*round(53/6) = 18, word-legal
HIGH = 0x00000080                     # HIGH_PRIORITY_CLASS


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
    mx, nd, wit = {}, 0, {}
    for i in range(n):
        if not done(i):
            continue
        nd += 1
        txt = open(log_of(i), errors="replace").read()
        for J, v in re.findall(r"^\s+(\d)\s+(\d+)\s", txt, re.M):
            J, v = int(J), int(v)
            if v > mx.get(J, 0):
                mx[J] = v
                wit[J] = i
    cov = sum(e[i + 1] - e[i] for i in range(n) if done(i))
    print(f"  shards complete {nd}/{n}   start indices covered "
          f"{cov:,}/{NOPEN:,} ({100.0*cov/NOPEN:.1f}%)")
    if mx:
        print("   " + "  ".join(
            f"Q*_{J}(47) {'=' if nd == n else '>='} {v} [shard {wit[J]}]"
            if v > SEED else f"Q*_{J}(47) <= {SEED} (at the seed)"
            for J, v in sorted(mx.items())))
        top = max(mx.values())
        print(f"   max over J {'=' if nd == n else '>='} {top}  vs budget "
              f"F(47) + 53 = 171  ->  "
              f"{'CERTIFIES' if top <= 171 else 'FAILS by +%d' % (top - 171)}"
              f"{'' if nd == n else '   (PARTIAL)'}")
    return nd == n, mx


def run(workers, n):
    os.makedirs(OUT, exist_ok=True)
    e = shard_edges(n)
    todo = [i for i in range(n) if not done(i)]
    print(f"  {len(todo)} shards to run, {n - len(todo)} already complete, "
          f"{workers} workers, seed {SEED}, cap {CAP}, JMAX {JMAX}, "
          f"floor {FLOOR}, {KIND}", flush=True)
    live, t0 = [], time.time()
    while todo or live:
        while todo and len(live) < workers:
            i = todo.pop(0)
            fh = open(log_of(i), "w")
            cmd = [PY, "-u", J5, "23", "29,31,37,41,43,47", "53",
                   "seed%d" % SEED, str(CAP), str(JMAX), FLOOR, KIND,
                   str(e[i]), str(e[i + 1])]
            try:
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     creationflags=HIGH)
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
                print(f"  shard {i} complete at t={time.time()-t0:.0f}s; "
                      f"{len(todo)} queued, {len(live)} live", flush=True)
                show(n)
    print("  all shards complete" if all(done(i) for i in range(n))
          else "  STOPPED WITH SHARDS OUTSTANDING", flush=True)
    show(n)


if __name__ == "__main__":
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    if sys.argv[1] == "show":
        show(N)
    else:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 4, N)

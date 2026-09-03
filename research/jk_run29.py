"""jk_run29.py -- explicit-seed parallel driver for rust2/src/bin/jkcov6.rs.

HARVESTER lane, round 29.  Replaces research/jk_run.py for long runs.

WHY IT EXISTS.  jk_run.py's phase-2 split is sound ONLY IF no worker ever
improves on the shared incumbent.  Read out of jkcov6.rs: every node is pruned
when `feasible_to(cov, j, best + 1)` fails, so a worker whose `best` has risen
prunes MORE nodes ABOVE the split depth, visits FEWER split-depth nodes, and
its global `leafctr` numbering diverges from the other workers'.  The parts
`leafctr % nparts == part` then need not cover the tree.  jk_run.py's own
docstring says as much and its driver prints "rerun needed" -- but round 28's
j_3(P(23)) run hit exactly that case (two of fourteen workers beat the seed
219, reaching 227 and 232) and the result was never redone.

WHAT THIS DRIVER CHANGES:
  * --seed is EXPLICIT and mandatory.  No phase-1 guess; you pass the best
    verified witness you have, so the run starts at the value it must confirm.
  * THE PROTOCOL ASSERTION IS FATAL, not a printed warning.  If any worker
    reports m > seed the driver exits non-zero and says the seed to use next.
  * Per-worker result files on disk (kept from jk_run.py -- that part was
    already right), so a killed driver loses nothing and a rerun resumes.
  * Node counts are the reported cost (BENCHMARK PROTOCOL: ops, not wall time).
    Wall times are printed as a secondary column and are NOT comparable across
    runs made under different box load.

SOUNDNESS OF A CLEAN RUN.  If every worker is seeded at M and no worker ever
reports m > M, then no worker's `best` ever moved, so all workers pruned
identically above the split depth, their `leafctr` sequences agree, the parts
partition the split-depth children, and "no part contains a solution longer
than M" is a proof that none exists.  Together with a verified witness of
length M this is a two-sided exact answer.

Usage:
  uv run python research/jk_run29.py <k> <z> --seed M [--workers N] [--split D]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "rust2", "target", "release", "jkcov6.exe")


def parse_quiet(line):
    # k z jk m nodes secs STATUS verify
    f = line.split()
    return dict(k=int(f[0]), z=int(f[1]), jk=int(f[2]), m=int(f[3]),
                nodes=int(f[4]), secs=float(f[5]), status=f[6],
                verify=(f[7] == "true"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("z", type=int)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=0)
    ap.add_argument("--tag", default="r29")
    args = ap.parse_args()

    M = args.seed
    t0 = time.time()
    print(f"[{args.tag}] k={args.k} z={args.z}  CONFIRMING m <= {M} with "
          f"{args.workers} workers (split depth {args.split})", flush=True)

    base = os.path.join(ROOT, "research", "data",
                        f"jkpart29_k{args.k}_z{args.z}_M{M}_n{args.workers}")
    procs = []
    for i in range(args.workers):
        fn = f"{base}_p{i}.txt"
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            print(f"  worker {i:2d}: already on disk, skipped", flush=True)
            procs.append((i, None, fn))
            continue
        cmd = [BIN, str(args.k), str(args.z), "--seed", str(M),
               "--split", str(args.split), "--part", str(i),
               "--nparts", str(args.workers), "--quiet"]
        if args.lmax:
            cmd += ["--lmax", str(args.lmax)]
        fh = open(fn, "w")
        procs.append((i, subprocess.Popen(cmd, stdout=fh,
                                          stderr=subprocess.DEVNULL), fh))

    tot_nodes = 0
    ok = True
    beat = M
    for i, p, h in procs:
        fn = h if isinstance(h, str) else h.name
        if p is not None:
            p.wait()
            h.close()
            if p.returncode != 0:
                print(f"  worker {i} FAILED rc={p.returncode}", flush=True)
                ok = False
                continue
        so = open(fn).read().strip()
        if not so:
            print(f"  worker {i} produced no output", flush=True)
            ok = False
            continue
        r = parse_quiet(so)
        tot_nodes += r["nodes"]
        flag = ""
        if r["m"] > M:
            flag = "  <-- BEAT THE SEED"
            beat = max(beat, r["m"])
            ok = False
        if r["status"] != "EXACT":
            flag += "  <-- NOT EXACT"
            ok = False
        print(f"  worker {i:2d}: m={r['m']} nodes={r['nodes']:>15,} "
              f"{r['secs']:9.1f}s {r['status']}{flag}", flush=True)

    el = time.time() - t0
    D = {1: 2, 2: 6, 3: 6, 4: 30, 5: 30}.get(args.k)
    jk = D * (M + 1) if D else None
    print(f"[{args.tag}] RESULT k={args.k} z={args.z}  m = {M}  "
          f"j_k = {jk}  {'EXACT (CONFIRMED)' if ok else 'INVALID'}  "
          f"nodes={tot_nodes:,}  wall={el:.1f}s", flush=True)
    if not ok:
        if beat > M:
            print(f"[{args.tag}] PROTOCOL VIOLATION: a worker reached m={beat}. "
                  f"Delete the part files and rerun with --seed {beat}.",
                  flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

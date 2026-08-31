"""jk_run.py -- parallel driver for rust2/src/bin/jkcov6.rs.

HARVESTER lane, round 28.

TWO-PHASE PROTOCOL (this is what makes the split sound):
  Phase 1  one unseeded worker with a wall-clock cap finds a WITNESS of some
           run length M and verifies it (the binary re-checks its own witness
           by independent code before printing).
  Phase 2  N workers, each SEEDED at M and each taking a residue class of the
           branches at a fixed split depth, prove that nothing exceeds M.
           Because every worker starts with the same incumbent M and no worker
           ever improves on it, the pruning above the split depth is identical
           in all workers, so the union of the parts is the whole tree.
           If ANY worker reports a value > M the run is invalid and is redone
           with the larger seed (the driver asserts this).

Usage:
  python research/jk_run.py <k> <z> [--workers N] [--split D] [--phase1 SECS]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rust2", "target", "release", "jkcov6.exe")


def run(argv, timeout=None):
    p = subprocess.run([BIN] + [str(a) for a in argv], capture_output=True,
                       text=True, timeout=timeout)
    return p.stdout.strip(), p.returncode


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
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--phase1", type=int, default=60)
    ap.add_argument("--lmax", type=int, default=0)
    args = ap.parse_args()

    extra = ["--lmax", args.lmax] if args.lmax else []

    t0 = time.time()
    print(f"[phase 1] k={args.k} z={args.z}  witness search ({args.phase1}s cap)",
          flush=True)
    out, rc = run([args.k, args.z, "--secs", args.phase1, "--quiet"] + extra)
    if rc != 0:
        print("phase 1 failed:", out)
        sys.exit(1)
    r1 = parse_quiet(out)
    assert r1["verify"], "phase-1 witness did not verify"
    M = r1["m"]
    print(f"[phase 1] witness m = {M}  (j_k = {r1['jk']})  status={r1['status']}"
          f"  {r1['secs']:.1f}s", flush=True)
    if r1["status"] == "EXACT":
        print(f"RESULT k={args.k} z={args.z}  j_k = {r1['jk']}  m = {M}  "
              f"EXACT (single worker)  nodes={r1['nodes']}  "
              f"{time.time()-t0:.1f}s", flush=True)
        return

    print(f"[phase 2] proving m <= {M} with {args.workers} workers "
          f"(split depth {args.split})", flush=True)
    # EVERY WORKER WRITES ITS OWN RESULT FILE.  Round 28 lost a 35-minute run
    # because the driver died and took its pipes with it; on disk the parts
    # survive the driver, and a rerun can skip the parts already finished.
    tag = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                       f"jkpart_k{args.k}_z{args.z}_M{M}_n{args.workers}")
    procs = []
    for i in range(args.workers):
        fn = f"{tag}_p{i}.txt"
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
                                          stderr=subprocess.DEVNULL), fn))
    tot_nodes = 0
    ok = True
    for i, p, fn in procs:
        if p is not None:
            p.wait()
            if p.returncode != 0:
                print(f"  worker {i} FAILED rc={p.returncode}")
                ok = False
                continue
        so = open(fn).read().strip()
        if not so:
            print(f"  worker {i} produced no output")
            ok = False
            continue
        r = parse_quiet(so)
        tot_nodes += r["nodes"]
        print(f"  worker {i:2d}: m={r['m']} nodes={r['nodes']:>14,} "
              f"{r['secs']:8.1f}s {r['status']}", flush=True)
        if r["status"] != "EXACT":
            ok = False
        if r["m"] > M:
            print(f"  worker {i} BEAT THE SEED ({r['m']} > {M}) -- rerun needed")
            ok = False
    el = time.time() - t0
    print(f"RESULT k={args.k} z={args.z}  j_k = {(r1['jk']//(M+1))*(M+1)}  "
          f"m = {M}  {'EXACT' if ok else 'INCOMPLETE'}  "
          f"nodes={tot_nodes:,}  wall={el:.1f}s", flush=True)


if __name__ == "__main__":
    main()

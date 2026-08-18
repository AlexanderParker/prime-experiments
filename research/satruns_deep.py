"""Round-8 deep scan: the L=14 hunt, members to ~1e12. RESUMABLE.

Exact continuation of round 7's absolute saturated-run scan, processed in
chunks of 8e9 slots with results FLUSHED after every chunk (lesson from the
first attempt, which was killed at ~70% and lost all in-memory results).
State file data/satruns_deep_state.txt holds the last completed k; rerunning
resumes there. Each chunk overlaps the previous by 200 slots so runs
straddling a boundary are seen whole (dedupe on k_start when aggregating).

Context (Lateral round 8): saturated runs are unconditionally capped at 32
by the (5,7) CRT corridor, so the record hunt operates inside [13, 32].

Outputs (append, flushed per chunk):
  data/satruns_deep_ge10.csv   - every run L >= 10 (k_start, member, L)
  data/satruns_deep_renewal.csv- per (chunk, decade) counts L = 8..13+
  data/satruns_deep_state.txt  - last completed k (resume point)
Any L >= 13 is printed immediately; L >= 14 is the headline.

Usage: uv run python research/satruns_deep.py [K] [k_start]
Defaults: K = 167_000_000_000 (member 1.002e12), start = 11_999_999_800
(or the state file if further).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saturated_runs import scan, side_word

K_DEFAULT = 167_000_000_000
START_DEFAULT = 11_999_999_800
CHUNK = 8_000_000_000
DDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE = os.path.join(DDIR, "satruns_deep_state.txt")


def opencsv(name, header):
    path = os.path.join(DDIR, name)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a")
    if new:
        f.write(header + "\n")
    return f


def main():
    args = [int(a) for a in sys.argv[1:]]
    K = args[0] if args else K_DEFAULT
    k_start = args[1] if len(args) > 1 else START_DEFAULT
    os.makedirs(DDIR, exist_ok=True)
    if os.path.exists(STATE):
        done = int(open(STATE).read().strip())
        if done > k_start:
            k_start = done - 200  # overlap; dedupe on aggregation
            print(f"resuming from state file: k_start = {k_start}")
    t0 = time.time()
    mx_all = 0
    a = k_start
    while a <= K:
        b = min(K, a + CHUNK)
        starts, lens = scan(b, seg=128_000_000, k_start=a, progress_every=0)
        fg = opencsv("satruns_deep_ge10.csv", "k_start,member_start,L")
        sel = lens >= 10
        for k0, L in zip(starts[sel].tolist(), lens[sel].tolist()):
            fg.write(f"{k0},{6*k0-1},{L}\n")
        fg.close()
        fr = opencsv("satruns_deep_renewal.csv",
                     "k_from,k_to,decade,L8,L9,L10,L11,L12,L13plus")
        if len(starts):
            mem = 6 * starts - 1
            dec = np.floor(np.log10(mem.astype(float))).astype(int)
            for d in range(int(dec.min()), int(dec.max()) + 1):
                m = dec == d
                row = [int((m & (lens == L)).sum())
                       for L in (8, 9, 10, 11, 12)]
                r13 = int((m & (lens >= 13)).sum())
                fr.write(f"{a},{b},{d}," + ",".join(map(str, row))
                         + f",{r13}\n")
        fr.close()
        for k0, L in zip(starts[lens >= 13].tolist(),
                         lens[lens >= 13].tolist()):
            w = side_word(k0, L)
            tag = "  <<<< NEW RECORD (L >= 14)" if L >= 14 else ""
            print(f"  L>=13: k={k0} member={6*k0-1} L={L} word={w}{tag}",
                  flush=True)
        mx = int(lens.max()) if len(lens) else 0
        mx_all = max(mx_all, mx)
        with open(STATE, "w") as f:
            f.write(str(b))
        pct = 100 * (b - START_DEFAULT) / (K - START_DEFAULT)
        print(f"chunk done: k <= {b} ({pct:.1f}%), chunk max L = {mx}, "
              f"elapsed {time.time()-t0:.0f}s", flush=True)
        a = b + 1
    print(f"\nVERDICT: max L in scanned range = {mx_all}; members to "
          f"{6*K+1:.3e}; total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

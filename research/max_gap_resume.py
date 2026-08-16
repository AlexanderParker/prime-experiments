"""Resumable exact scan for F(L, y), the largest gap in the survivor pattern.

Full-period scans past y = 29 do not fit in one run: the period for y = 31 is
100,280,245,065 positions. This version checkpoints after every time budget, so a
later invocation continues where it stopped and the table can be extended a slice
at a time. Until a run reports COMPLETE, the number printed is a rigorous lower
bound on F(L, y) - the largest gap seen in the prefix scanned so far.

    python max_gap_resume.py 31        # one 90 second slice, then checkpoint
    python max_gap_resume.py 31 300    # a 300 second slice

State lives in max_gap_state_L{L}_y{y}.json next to this file.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_avoidance import odd_primes_upto

CHUNK = 1 << 26


def state_path(L, y):
    return Path(__file__).resolve().parent / f"max_gap_state_L{L}_y{y}.json"


def load(L, y, qs):
    path = state_path(L, y)
    if path.exists():
        return json.loads(path.read_text())
    period = 1
    for q in qs:
        period *= q
    return {"L": L, "y": y, "period": period, "start": 0,
            "biggest": 0, "first": None, "last": None}


def save(st):
    state_path(st["L"], st["y"]).write_text(json.dumps(st))


def run(y, seconds=90, L=2):
    qs = odd_primes_upto(y)
    st = load(L, y, qs)
    period, start = st["period"], st["start"]
    biggest, first, last = st["biggest"], st["first"], st["last"]

    t0 = time.time()
    while start < period and time.time() - t0 < seconds:
        size = min(CHUNK, period - start)
        alive = np.ones(size, dtype=bool)
        for q in qs:
            for r in range(L):
                alive[(r - start) % q :: q] = False
        idx = np.flatnonzero(alive)
        if idx.size:
            pos = idx + start
            if first is None:
                first = int(pos[0])
            if last is not None:
                biggest = max(biggest, int(pos[0]) - last)
            if idx.size > 1:
                biggest = max(biggest, int(np.diff(pos).max()))
            last = int(pos[-1])
        start += size

    st.update(start=start, biggest=biggest, first=first, last=last)
    save(st)

    done = start >= period
    if done and last is not None:
        biggest = max(biggest, (period - last) + first)
        st["biggest"] = biggest
        save(st)

    pct = 100.0 * start / period
    tag = "COMPLETE - exact" if done else "partial - lower bound"
    print(f"L={L} y={y}: scanned {start}/{period} ({pct:.2f}%) [{tag}]")
    print(f"F({L},{y}) {'=' if done else '>='} {biggest}, "
          f"so 2F {'=' if done else '>='} {2 * biggest}, "
          f"against y^2-y-2 = {y * y - y - 2}")
    return st


if __name__ == "__main__":
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 90
    run(y, secs)

"""Loop, entry 39: is the tail the same problem one machine down?  Descending settle walk
with mirror {2, 3, g}, K = 40, full memory.  The prefix = the steps of the free regime (top i
gears all above 2i); the tail = the remaining gears (below about q / 4.5) down to 5.  At each
tail step, among the eighty candidates inside the range: how many are open to the small
gears visited so far (the tail's own machine), how many to the big gears (the prefix's, kept
in memory), and how many to both (the keeping moves).  If the big gears rarely strike the
candidates the small gears leave open, the tail is the machine q/4.5's problem with the room
of the window (q, q^2].
usage: uv run python research/stack/r8/settle_tail.py lo hi
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path
sys.path.insert(0, "research/stack/r8")
import evolve_walk3 as E

def dist(ph, h): return min(ph, abs(ph - (h - 2)), h - ph if ph > h - 2 else h)

def main():
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); K = 40
    E.QMAX = hi
    N = hi * hi * 4 + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    qs = list(primerange(lo, hi + 1))
    out = [__doc__.strip(), ""]
    tot = dict(steps=0, cand=0, open_small=0, open_big=0, keeping=0, small_only_but_big_struck=0, no_keep=0)
    mins = []   # per machine: the minimum keeping count over the tail steps, and the gear where it occurs
    for q in qs:
        m = E.Machine(q, sv); P = 6; seq = [g for g in m.ps if g >= 5][::-1]
        free = 0
        while free < len(seq) and seq[free] > 2 * (free + 1): free += 1
        visited = []; L = -1; mk = (10 ** 9, None)
        for i, g in enumerate(seq):
            visited.append(g); nxt = seq[i + 1] if i + 1 < len(seq) else None
            big = [h for h in visited if h > seq[free - 1]] if free else []
            small = [h for h in visited if h not in big]
            cands = []
            for k in range(1, K + 1):
                for d in (1, -1):
                    n = L + 2 * k * P * g * d
                    if 0 <= n <= q * q - 2: cands.append(n)
            if i >= free:
                tot['steps'] += 1; tot['cand'] += len(cands)
                os_ = [n for n in cands if all(n % h not in (0, h - 2) for h in small)]
                ob = [n for n in cands if all(n % h not in (0, h - 2) for h in big)]
                keep = [n for n in os_ if all(n % h not in (0, h - 2) for h in big)]
                tot['open_small'] += len(os_); tot['open_big'] += len(ob); tot['keeping'] += len(keep)
                tot['small_only_but_big_struck'] += len(os_) - len(keep)
                if not keep: tot['no_keep'] += 1
                if len(keep) < mk[0]: mk = (len(keep), g)
            best = None
            for n in cands:
                on = sum(1 for h in visited if n % h in (0, h - 2))
                md = min(dist(n % h, h) for h in visited)
                fl = 0
                if on == 0 and nxt is not None:
                    fl = sum(1 for k2 in range(1, 21) for d2 in (1, -1) if 0 <= n + 2 * k2 * P * nxt * d2 <= q * q - 2 and all((n + 2 * k2 * P * nxt * d2) % h not in (0, h - 2) for h in visited + [nxt]))
                sc = (-on, fl, md)
                if best is None or sc > best[0]: best = (sc, n)
            L = best[1]
        mins.append((q, mk[0], mk[1]))
    s = tot['steps']
    out.append(f"machines {qs[0]}..{qs[-1]} ({len(qs)}); tail steps {s}; per tail step: candidates {tot['cand'] / s:.1f}, open to the small gears {tot['open_small'] / s:.2f}, open to the big gears {tot['open_big'] / s:.2f}, keeping (both) {tot['keeping'] / s:.2f}; candidates open to the small gears but struck by a big one {tot['small_only_but_big_struck'] / s:.2f} per step; tail steps with no keeping move {tot['no_keep']}")
    out.append(f"minimum keeping moves over the tail steps per machine (q, min, at gear): {sorted(mins, key=lambda r: r[1])[:12]}; overall minimum {min(r[1] for r in mins)}")
    Path(f"research/stack/r8/results_settle_tail_{lo}_{hi}.txt").write_text("\n".join(out), encoding="utf-8")
    print(out[-2]); print(out[-1])

if __name__ == "__main__":
    main()

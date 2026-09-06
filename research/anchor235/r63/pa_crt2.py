"""pa_crt2.py -- how the CRT search fails, and the same search run on the other two letters.

(1) At the minimal INFEASIBLE cell (a_L, r(a_L) + 1) of every rung:
      * is it killed by the gear-5 pair filter alone (gear 5 has no admissible class)?
      * what is the SHORTEST sub-interval of the run whose columns already cannot be covered?
        A short one is a local, proof-shaped obstruction; if the only uncoverable window is the
        whole run, the obstruction is global and there is no local certificate.
(2) The rows of the long letter b_L = q' - a_L and the padded letter q', by the same search,
    against the scanned values, and their excesses against F.

Outputs results/pa_crt2.txt / .json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
import pa_crt as C                                        # noqa: E402
from mf_core import u_of                                  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def unorm(g):
    w = u_of(g)
    return min(w, g - w)


class WinSearch(C.Search):
    """the same search restricted to covering only the columns of a sub-interval."""

    def __init__(self, gears, opens, S, lo, hi):
        self.gears = gears
        self.cover = [j for j in range(lo, hi + 1) if j not in opens]
        self.opts = {}
        self.union = {}
        for g in gears:
            out = {}
            for lam in range(g):
                if any(C.strike(g, lam, j) for j in opens):
                    continue
                m = 0
                for i, j in enumerate(self.cover):
                    if C.strike(g, lam, j):
                        m |= 1 << i
                out.setdefault(m, lam)
            self.opts[g] = list(out.items())
            u = 0
            for m in out:
                u |= m
            self.union[g] = u
        self.full = (1 << len(self.cover)) - 1
        self.nodes = 0
        self.sol = None


def gear5_dead(gears, v, a):
    S = v + a
    opens = {0, v, S}
    for g in gears:
        if not any(not any(C.strike(g, lam, o) for o in opens) for lam in range(g)):
            return g
    return None


def min_uncoverable_window(gears, v, a):
    """shortest sub-interval [lo, hi] of (0, S) whose columns alone cannot be covered."""
    S = v + a
    opens = {0, v, S}
    for length in range(1, S):
        for lo in range(1, S - length + 1):
            hi = lo + length - 1
            if all(j in opens for j in range(lo, hi + 1)):
                continue
            s = WinSearch(gears, opens, S, lo, hi)
            try:
                ok, _, _ = s.run()
            except RuntimeError:
                continue
            if not ok:
                return (length, lo, hi)
    return None


def row_top(gears, v, Fmax):
    for a in range(Fmax, 0, -1):
        try:
            ok, sol, n = C.feasible(gears, v, a)
        except RuntimeError:
            return None, None
        if ok:
            return a, sol
    return 0, None


def main():
    t0 = time.time()
    L = []
    W = L.append
    W("=== HOW THE CRT SEARCH FAILS, AND THE OTHER TWO LETTERS ===")

    RUNGS = [(13, [5, 7, 11], 7, 3),
             (17, [5, 7, 11, 13], 11, 7),
             (19, [5, 7, 11, 13, 17], 18, 12),
             (23, [5, 7, 11, 13, 17, 19], 25, 20),
             (29, [5, 7, 11, 13, 17, 19, 23], 34, 25),
             (31, [5, 7, 11, 13, 17, 19, 23, 29], 43, 35),
             (37, [5, 7, 11, 13, 17, 19, 23, 29, 31], 58, 46)]

    W("\n--- (1) the obstruction at the minimal infeasible cell a = r(a_L) + 1 ---")
    W("rung | a_L | a = r+1 | span | gear with NO admissible class | shortest uncoverable window")
    obs = []
    for qn, gears, F, rk in RUNGS:
        aL = 2 * unorm(qn)
        a = rk + 1
        t1 = time.time()
        dead = gear5_dead(gears, aL, a)
        win = None if dead else min_uncoverable_window(gears, aL, a)
        S = aL + a
        W(f"{qn:>4} | {aL:>3} | {a:>7} | {S:>4} | {dead if dead else 'none'} | "
          f"{'(killed by the pair filter)' if dead else (win if win else 'NONE -- only the whole run')}"
          f"  [{time.time()-t1:.1f}s]")
        obs.append(dict(q=qn, aL=aL, a=a, S=S, dead=dead, win=win))
    W("  a gear with no admissible class is the gear-5 PAIR FILTER (parent 4.i.a.i.a 2.3, cited);")
    W("  a short uncoverable window would be a local proof; 'NONE' means the infeasibility is a")
    W("  property of the whole run and has no interval certificate.")

    W("\n--- (2) the rows of the three letters by the same search ---")
    W("rung | letter | value | 3*letter | multiplier | search top | scanned | F | excess E")
    SCAN_B = {7: 0, 11: 0, 13: 0, 17: 5, 19: 7, 23: 13, 29: 15, 31: 27, 37: 40}
    SCAN_Q = {23: 5, 29: 8, 31: 14, 37: 30}
    rows = []
    for qn, gears, F, rk in RUNGS:
        e = 1 if qn % 6 == 5 else -1
        aL = 2 * unorm(qn)
        bL = qn - aL
        for name, v, mult, scan in (("a_L", aL, qn + e, rk),
                                    ("b_L", bL, 2 * qn - e, SCAN_B.get(qn)),
                                    ("q'", qn, 3 * qn, SCAN_Q.get(qn))):
            if v > F:
                W(f"{qn:>4} | {name:<3} | {v:>5} | {3*v:>8} | {mult:>10} | "
                  f"unrealisable (v > F) | {scan} | {F} | -")
                continue
            t1 = time.time()
            top, sol = row_top(gears, v, F)
            ok = (scan is None or top == scan)
            W(f"{qn:>4} | {name:<3} | {v:>5} | {3*v:>8} | {mult:>10} | {top:>10} | "
              f"{scan if scan is not None else '-':>7} | {F:>2} | {v + top - F:+d}"
              f"  {'' if ok else '<< MISMATCH'}  [{time.time()-t1:.1f}s]")
            rows.append(dict(q=qn, name=name, v=v, mult=mult, top=int(top),
                             scan=scan, F=F, E=int(v + top - F)))
    W("  the multiplier is 3 * letter: q'+eps for a_L, 2q'-eps for b_L, 3q' for the padded letter.")

    json.dump(dict(obs=obs, rows=rows), open(os.path.join(OUT, "pa_crt2.json"), "w"))
    txt = "\n".join(L)
    open(os.path.join(OUT, "pa_crt2.txt"), "w").write(txt)
    print(f"wrote {OUT}/pa_crt2.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    print(txt)


if __name__ == "__main__":
    main()

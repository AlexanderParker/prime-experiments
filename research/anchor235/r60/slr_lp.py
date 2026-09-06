"""slr_lp.py -- SCAN-FREE CERTIFICATES THAT THE SHORT-LETTER ROW IS EMPTY ABOVE r(a_L).

The row statement "M has no adjacent gap pair (a, a_L) with a > a_0" is exactly a family of
level-2 dictionary cells, and the LP lane's windowed vehicle decides such a cell by duality:
prescribe the three openings 0, a, a+a_L of a window of width a+a_L+1 and ask the composed
restricted covering LP whether every other position can be blocked
(docs/novel/restricted-covering-certificates.md RESULT 4; research/window_dict.py).

For each rung this script certifies every cell (a, a_L) with r(a_L) < a <= F(M), escalating the
number of held gears until the cell certifies, and checks TIGHTNESS by running the cell at
a = r(a_L), which is realised and must NOT certify.

Run:  uv run python research/anchor235/r60/slr_lp.py <y> <aL> <a_lo> <a_hi> [kmax]
"""
import os
import sys
import time
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "research"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "..", "..", "research"))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
from star_case import RelaxStar, decide_star, two_gap_geometry     # noqa: E402
from lp_degree_range import gears_of                               # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def cell(y, W, a, k, maxrounds=120, tb=25.0):
    """decide the cell (a, W-a) at machine y holding the k smallest gears.
    returns (verdict, ncases, ndead, ops, secs)"""
    g = gears_of(y)
    A, op = two_gap_geometry(W, a)
    held = g[:k]
    ops = 0
    dead = 0
    t0 = time.time()
    for ws in product(*[range(q) for q in held]):
        R = RelaxStar(g, A, held, ws, op)
        if R.dead:
            dead += 1
            del R
            continue
        v, info = decide_star(R, verbose=False, maxrounds=(6 if k == 0 else maxrounds),
                              time_budget=tb)
        ops += int(info.get('ops', 0) or 0)
        del R
        if v != 'CERTIFIED':
            return v, ws, dead, ops, time.time() - t0
    return 'CERTIFIED', None, dead, ops, time.time() - t0


def main():
    y = int(sys.argv[1])
    aL = int(sys.argv[2])
    lo = int(sys.argv[3])
    hi = int(sys.argv[4])
    kmax = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    kmin = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    lines = []
    W = lines.append
    W(f"=== machine {{5..{y}}}, row v = {aL}: certifying cells (a, {aL}) for a = {lo}..{hi} ===")
    W(f"gears {gears_of(y)}")
    tot_ops = 0
    verdicts = {}
    for a in range(lo, hi + 1):
        span = a + aL
        got = None
        for k in range(kmin, kmax + 1):
            v, ws, dead, ops, secs = cell(y, span, a, k)
            tot_ops += ops
            if v == 'CERTIFIED':
                got = (k, dead, ops, secs)
                W(f"  a = {a:>2} (span {span:>2}, split ({a},{aL})): CERTIFIED, "
                  f"{k} gears held, {dead} cases vacuous, {ops:,} exact ops, {secs:.1f}s")
                break
            W(f"  a = {a:>2} (span {span:>2}): k = {k} -> {v} at case {ws} "
              f"({ops:,} ops, {secs:.1f}s)")
        verdicts[a] = 'CERTIFIED' if got else 'STALLED'
        if not got:
            W(f"  a = {a:>2}: NOT CERTIFIED up to k = {kmax}")
    W(f"  total exact operations: {tot_ops:,}")
    W(f"  certified {sum(1 for v in verdicts.values() if v == 'CERTIFIED')} of {len(verdicts)}")
    txt = "\n".join(lines)
    print(txt)
    fn = os.path.join(OUT, f"slr_lp_m{y}_v{aL}_{lo}_{hi}.txt")
    open(fn, "w").write(txt)
    print("wrote", fn)


if __name__ == "__main__":
    main()

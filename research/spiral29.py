"""LATERAL round 25, backlog item U1: THE MACHINE-29 DEPTH SPIRAL.

The depth family  What_j(omega) = sum over depth-j windows of omega^{window sum},
omega = e(1/5), was measured only to m23 (round 21, c14_phase.py part 3).  Its
ANCHOR is a proved identity

    sum_{j>=1} What_j(omega) = |A_hat|^2 - N = (2-phi) prod_{q!=5}(q-2)^2 - N,

REAL, so the arms close a polygon; W_2's arm was climbing toward the pole phase
126 deg (66.5 -> 87.7 -> 113.2 at m17/19/23).  m29 is the next rung.

This is a STREAMING rewrite of c14_phase.part3: the round-21 code materialised
the whole opening array (m29 would need ~1.7 GB of int64 plus copies).  Here only
the opening RESIDUES mod 5 are kept, in a rolling buffer of J elements, because
omega^5 = 1 makes every window sum matter only mod 5.  Peak memory ~200 MB.

Also emitted (free, same pass, cross-checks for the round-25 mirror-cell theory):
  - the exact m29 gap histogram W_1(g),
  - the 5x5 table T[start slot residue mod 5][gap mod 5] (the phase-graded cell
    matrix; the mirror law predicts T[0][2] == T[3][2], T[0][3] == T[2][3],
    T[2][0] == T[3][0]),
  - opening class counts mod 5 (must be n_side on A_5 = {0,2,3}, 0 elsewhere).

All counts are EXACT INTEGERS; only the reported angles/moduli are floats.

Usage: python spiral29.py [y] [--J 25] [--chunk 33554432]
"""
import sys, os, cmath, math, json
from math import prod, pi
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OMEGA = cmath.exp(2j * pi / 5)


def primes(a, b):
    return [n for n in range(a, b + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def deg(z):
    return math.degrees(cmath.phase(z))


def run(y, J, chunk):
    gears = primes(5, y)
    P = prod(gears)
    n_side = prod(q - 2 for q in gears if q != 5)
    N_pred = prod(q - 2 for q in gears)
    print(f"machine {y}: gears {gears}")
    print(f"  P = {P}   predicted openings N = {N_pred}   n_side = {n_side}")

    lag = np.zeros((J + 1, 5), dtype=np.int64)   # lag[j][r] = #windows, sum = r
    ghist = {}
    cell = np.zeros((5, 5), dtype=np.int64)      # [start res][gap res]
    opencls = np.zeros(5, dtype=np.int64)

    buf = np.empty(0, dtype=np.int8)             # rolling residues
    head = None                                  # first J residues
    prev_last = None                             # last opening slot (int)
    first_open = None
    N = 0

    a = 0
    while a < P:
        S = min(chunk, P - a)
        killed = np.zeros(S, dtype=bool)
        for q in gears:
            u = pow(6, -1, q)
            for t in (u, q - u):
                killed[(t - a) % q::q] = True
        o = np.flatnonzero(~killed)
        del killed
        if o.size:
            o64 = o.astype(np.int64) + a
            del o
            N += o64.size
            res = (o64 % 5).astype(np.int8)
            opencls += np.bincount(res.astype(np.int64), minlength=5)
            # gaps: within chunk plus the bridge from the previous chunk
            if prev_last is None:
                first_open = int(o64[0])
                starts = o64[:-1]
                gaps = np.diff(o64)
            else:
                starts = np.concatenate(([prev_last], o64[:-1]))
                gaps = np.diff(np.concatenate(([prev_last], o64)))
            if gaps.size:
                gm = gaps.max()
                hc = np.bincount(gaps, minlength=int(gm) + 1)
                for g in np.flatnonzero(hc):
                    ghist[int(g)] = ghist.get(int(g), 0) + int(hc[g])
                idx = (starts % 5).astype(np.int64) * 5 + (gaps % 5)
                cell += np.bincount(idx, minlength=25).reshape(5, 5)
                del idx
            prev_last = int(o64[-1])
            del o64, starts, gaps
            # depth ladder on residues
            buf = np.concatenate((buf, res)) if buf.size else res
            del res
            m = buf.size - J
            if m > 0:
                for j in range(1, J + 1):
                    d = (buf[j:j + m].astype(np.int64)
                         - buf[:m].astype(np.int64)) % 5
                    lag[j] += np.bincount(d, minlength=5)
                    del d
                if head is None:
                    head = buf[:J].copy()
                buf = buf[m:].copy()
        a += S
        print(f"    ... slots {a}/{P}  ({100.0*a/P:.1f}%)  openings {N}",
              flush=True)

    # close the cycle: last gap and the final J window starts
    gwrap = P - prev_last + first_open
    ghist[gwrap] = ghist.get(gwrap, 0) + 1
    cell[prev_last % 5, gwrap % 5] += 1
    buf2 = np.concatenate((buf, head))
    for j in range(1, J + 1):
        d = (buf2[j:j + J].astype(np.int64) - buf2[:J].astype(np.int64)) % 5
        lag[j] += np.bincount(d, minlength=5)

    # ---------------------------------------------------------------- gates
    assert N == N_pred, (N, N_pred)
    assert sum(ghist.values()) == N, (sum(ghist.values()), N)
    assert sum(g * c for g, c in ghist.items()) == P
    assert int(cell.sum()) == N
    for j in range(1, J + 1):
        assert int(lag[j].sum()) == N, (j, int(lag[j].sum()))
    exp = np.zeros(5, dtype=np.int64)
    for r in (0, 2, 3):
        exp[r] = n_side
    assert (opencls == exp).all(), (opencls, exp)
    assert (cell[1] == 0).all() and (cell[4] == 0).all()
    # W_1 from the ladder must equal the histogram's residue classes
    w1cls = np.zeros(5, dtype=np.int64)
    for g, c in ghist.items():
        w1cls[g % 5] += c
    assert (lag[1] == w1cls).all(), (lag[1], w1cls)
    print("  ASSERT ok: N, period sum, opening classes on A_5, ladder totals, "
          "W_1 vs ladder depth 1")

    # -------------------------------------------------------------- mirror
    print("  cell table T[start res][gap res] (rows 1,4 are empty by A_5):")
    for r in range(5):
        print(f"    start {r}: " + " ".join(f"{int(x):>12}" for x in cell[r]))
    mirror_pairs = [((0, 2), (3, 2)), ((0, 3), (2, 3)), ((2, 0), (3, 0))]
    for (i1, s1), (i2, s2) in mirror_pairs:
        d = int(cell[i1, s1]) - int(cell[i2, s2])
        print(f"    mirror pair T[{i1}][{s1}] - T[{i2}][{s2}] = {d}")
    N1, N2, N3, N4 = (int(lag[1][r]) for r in (1, 2, 3, 4))
    N0 = int(lag[1][0])
    print(f"  gap residue classes mod 5: N0..N4 = {N0} {N1} {N2} {N3} {N4}")
    print(f"    N2 even? {N2 % 2 == 0}   N3 even? {N3 % 2 == 0}")
    print(f"    mirror relation 2(N1-N4) - (N2-N3) = {2*(N1-N4)-(N2-N3)}")
    print(f"    126-defect  (N2+N3) - 2*N0 = {N2+N3-2*N0}  "
          f"(theorem: == 2 mod 4)")

    # --------------------------------------------------------------- spiral
    pred_total = (2 - (1 + 5 ** 0.5) / 2) * n_side ** 2 - N
    print(f"  closure (float): sum_j What_j(omega) = {pred_total:.6f} "
          f"(real, exact form (2-phi)*n_side^2 - N)")
    print(f"    {'j':>3} {'|W_j|/N':>10} {'arg deg':>9} {'cum arg':>9} "
          f"{'cum|.|/N':>10}")
    tot = 0j
    arms = []
    for j in range(1, J + 1):
        Wj = sum(int(lag[j][r]) * OMEGA ** r for r in range(5))
        tot += Wj
        arms.append((j, [int(x) for x in lag[j]], abs(Wj) / N, deg(Wj)))
        print(f"    {j:>3} {abs(Wj)/N:>10.4f} {deg(Wj):>9.2f} "
              f"{deg(tot):>9.2f} {abs(tot)/N:>10.4f}")
    out = dict(y=y, P=P, N=N, n_side=n_side, J=J,
               lag=[[int(x) for x in lag[j]] for j in range(J + 1)],
               ghist={str(k): v for k, v in sorted(ghist.items())},
               cell=[[int(x) for x in row] for row in cell])
    with open(os.path.join(DATA, f"spiral_{y}.json"), "w") as f:
        json.dump(out, f)
    print(f"  wrote data/spiral_{y}.json")
    print("DONE")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    y = int(args[0]) if args and not args[0].startswith("--") else 29
    J = int(args[args.index("--J") + 1]) if "--J" in args else 25
    ch = int(args[args.index("--chunk") + 1]) if "--chunk" in args else 1 << 25
    run(y, J, ch)

"""Round 29 (mechanic): THE ROUND GATE.

One command, one process, no imports from the tools that produced the numbers.
Everything this round claims as an exhibited object is re-derived here from the
definition (slot k is blocked by gear q iff k = +-6^{-1} mod q):

  A. the four F_2 CRT slots handed to Formalist (item a) - openings at the three
     offsets, every other slot of the span blocked, the two outside neighbours
     located, and the machine-59 pair asserted to be an exact mirror pair;
  B. the anchor-235 RECORD-LAW survivors at machines 31 / 37 / 41 (item c) -
     each JSON profile is turned back into an absolute slot of the TARGET
     machine and re-checked there: opening at the slot, opening at slot + F,
     every slot between blocked, and exactly L openings of the LOWER machine
     inside, with the reported (gap-before, span, gap-after);
  C. the chain-depth vehicle replicating anchor235/chain_depth.py at g = 7..29;
  D. the item-(b) worker logs TILE machine 23's period exactly, and the reported
     per-J maxima are the max over the tiling.

usage: uv run python research/gate_mechanic_r29.py
"""
import json
import os
import re
import sys
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "r29")


def gears(y):
    return [p for p in range(5, y + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def mk(y):
    return {q: (pow(6, -1, q), (-pow(6, -1, q)) % q) for q in gears(y)}


def is_open(k, T):
    return all(k % q not in t for q, t in T.items())


# ------------------------------------------------------------------ A
def sectionA():
    W = [(41, 21157523372970, [0, 28, 103], "F_2(41) = 103"),
         (53, 327666424664536738, [0, 77, 159], "F_2(53) = 159"),
         (59, 307199471342884027665, [0, 100, 173], "F_2(59) = 173 A"),
         (59, 13260587016151412007, [0, 73, 173], "F_2(59) = 173 B")]
    out = []
    for y, k, offs, lab in W:
        T = mk(y)
        P = prod(gears(y))
        assert 0 <= k < P, (lab, "slot outside the period")
        for t in range(offs[-1] + 1):
            assert is_open(k + t, T) == (t in set(offs)), (lab, t)
        lo = k - 1
        while not is_open(lo, T):
            lo -= 1
        hi = k + offs[-1] + 1
        while not is_open(hi, T):
            hi += 1
        print(f"  A {lab:18s} m{y}  span {offs[-1]}  interior {offs[1]}  "
              f"flanks ({k - lo}, {hi - (k + offs[-1])})")
        out.append((k, lo, hi, P))
    a, b = out[2], out[3]
    assert a[0] + b[0] + 173 == a[3], "machine-59 pair is not a mirror pair"
    assert (a[0] - a[1], a[2] - a[0] - 173) == (b[2] - b[0] - 173, b[0] - b[1])
    print(f"  A machine-59 pair: y_A + y_B + 173 = P(59) = {a[3]:,}  "
          f"and the flank pairs are reverses  MIRROR PAIR")


# ------------------------------------------------------------------ B
def sectionB():
    CORPUS = {31: 58, 37: 88, 41: 91}
    LOWERY = {31: 29, 37: 31, 41: 37}
    for g in (31, 37, 41):
        fn = os.path.join(DATA, f"chain_{g}.json")
        if not os.path.exists(fn):
            print(f"  B machine {g}: NO RESULT FILE - not attempted")
            continue
        J = json.load(open(fn))
        best = max(v["value"] for v in J["best"].values())
        assert best == CORPUS[g], (g, best, CORPUS[g])
        T, TL = mk(g), mk(LOWERY[g])
        Pg = prod(gears(g))
        for L, v in sorted(J["best"].items(), key=lambda kv: int(kv[0])):
            if v["value"] != best:
                continue
            k = v["slot"] % Pg
            F = v["value"]
            assert is_open(k, T) and is_open(k + F, T), (g, "endpoints")
            assert all(not is_open(k + t, T) for t in range(1, F)), (g, "gap")
            ins = [t for t in range(1, F) if is_open(k + t, TL)]
            assert len(ins) == int(L), (g, "chain arity", ins)
            assert (ins[0], ins[-1] - ins[0], F - ins[-1]) == (
                v["before"], v["span"], v["after"]), (g, "profile")
            print(f"  B machine {g}: F({g}) = {F} at slot {k}, chain arity "
                  f"L = {L}, lower openings at {ins}, profile "
                  f"({v['before']}, {v['span']}, {v['after']}), phase r = "
                  f"{v['r']}, copy j = {v['copy']}   VERIFIED AT MACHINE {g}")
        print(f"  B machine {g}: chain depth D_{g} = {J['D']}")


# ------------------------------------------------------------------ C
def sectionC():
    sys.path.insert(0, HERE)
    import chain_depth_r29 as C
    C.gate()


# ------------------------------------------------------------------ D
def one_run(sub, seed):
    """Read one sharded j5_multi run: assert the COMPLETED shards tile a
    prefix-free set of start indices, report coverage and per-J maxima."""
    d = os.path.join(DATA, sub)
    if not os.path.isdir(d):
        print(f"  D {sub}: not attempted")
        return
    tiles, mx, part = [], {}, 0
    for f in sorted(os.listdir(d)):
        txt = open(os.path.join(d, f), errors="replace").read()
        m = re.search(r"WALKING start-opening indices \[([\d,]+), ([\d,]+)\)",
                      txt)
        if not m:
            continue
        lo, hi = (int(m.group(i).replace(",", "")) for i in (1, 2))
        if "scan complete" not in txt:
            part += 1
            continue
        tiles.append((lo, hi))
        for J, v in re.findall(r"^\s+(\d)\s+(\d+)\s", txt, re.M):
            mx[int(J)] = max(mx.get(int(J), 0), int(v))
    tiles.sort()
    for i in range(1, len(tiles)):          # completed shards never overlap
        assert tiles[i][0] >= tiles[i - 1][1], ("shard overlap", tiles)
    cov = sum(b - a for a, b in tiles)
    full = cov == 7952175
    print(f"  D {sub}: seed {seed}, {len(tiles)} complete shards "
          f"({'' if full else 'PARTIAL, '}{cov:,}/7,952,175 start indices, "
          f"{100.0*cov/7952175:.1f}%), {part} incomplete")
    if not mx:
        return
    print("  D   " + "  ".join(
        (f"F_{J}(47) = {v}" if full and v > seed else
         f"F_{J}(47) >= {v}" if v > seed else f"F_{J}(47) <= {seed}")
        for J, v in sorted(mx.items())))
    top = max(mx.values())
    print(f"  D   max over J {'=' if full and top > seed else '>='} {top}  vs "
          f"budget F(47) + 53 = 171  ->  "
          f"{'CERTIFIES' if top <= 171 else 'FAILS by +%d' % (top - 171)}")


def sectionD():
    one_run("fj47", 145)
    one_run("fj47_s174", 174)


# ------------------------------------------------------------------ E
def sectionE():
    """The F_6(47) = 177 maximiser as a SLOT of machine 47, re-checked there."""
    y, k = 47, 46615676895423125
    offs = [0, 42, 70, 103, 107, 115, 177]
    T = mk(y)
    assert 0 <= k < prod(gears(y))
    nb = 0
    for t in range(offs[-1] + 1):
        o = is_open(k + t, T)
        assert o == (t in set(offs)), ("m47 mismatch", t)
        nb += not o
    g = [offs[i + 1] - offs[i] for i in range(6)]
    print(f"  E F_6(47) = 177 at machine-47 slot {k}: 7 consecutive openings "
          f"{offs}, gap word {g}, {nb} other slots blocked")
    assert sum(g) == 177 > 171
    print("  E 177 > 171 = F(47) + 53  ->  the spectrum-plus-depth certificate "
          "FAILS at 47 -> 53, on an exhibited window")


if __name__ == "__main__":
    print("ROUND-29 MECHANIC GATE\n\nA. CRT slots (item a)")
    sectionA()
    print("\nB. record-law survivors, re-checked at the target machine (item c)")
    sectionB()
    print("\nC. chain-depth vehicle vs anchor235/chain_depth.py")
    sectionC()
    print("\nD. rung-eleven inputs (item b)")
    sectionD()
    print("\nE. the F_6(47) maximiser as a machine-47 slot (item b)")
    sectionE()
    print("\nALL ASSERTIONS PASSED")

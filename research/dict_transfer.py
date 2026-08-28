"""Round 24 (mechanic): THE DICTIONARY TRANSFER - the realised gap m-tuple
dictionary of machine M + q', computed from machine M's dictionary alone, by
PARTIAL SUMS MOD q', with NO scan of the new machine.

This is the construct named (and not built) in round 23's R23.I, aimed at the
object Constructor and Formalist both asked for.  It is the lap-phase transfer
(docs/novel/old-machine-spectrum.md) applied to DICTIONARIES instead of to
extremal values.

THE MECHANISM, in one paragraph.  Adding gear q' deletes the M-openings whose
slot is +-u' mod q' (u' = 6^{-1} mod q').  Anchor a window at an M'-opening
y_0 and write d_i = y_i - y_0 for the following M-openings; then y_i is
DELETED iff d_i mod q' lies in the two-element set {A, A - s}, where
s = 2u' mod q' and A = (u' - y_0) mod q' is FREE (CRT: y_0 mod q' takes every
value).  So the whole kill pattern of a window is a function of the window's
PARTIAL SUMS MOD q' and one free phase A - the gap word already carries it,
and nothing about the new machine's period is needed.  A 5-tuple of
consecutive M'-openings is therefore exactly: a walk of consecutive M-gaps,
a phase A with 0 not in {A, A-s}, in which exactly 3 interior openings
survive and every other interior is deleted.

WHAT MAKES IT FINITE, AND WHY THE DEPTH DOES NOT RUN AWAY.  Two kills are
>= min(s, q'-s) apart automatically (their offsets differ by 0 or +-s mod q'),
which is the kernel-checked T4 spacing bound; with the span cap
sum <= F_4(M') the number of merged M-gaps is bounded.  Better, the search
SHRINKS geometrically with depth: a kill step admits only the ~2/q' of the
dictionary's out-edges that land on the two kill residues, so the layer sizes
fall like (out-degree * 2/q')^k.  Measured layer profiles are printed.

WHAT IS EXACT AND WHAT IS A SUPERSET - stated plainly, because it decides how
the output may be used.  The transfer needs to know which WALKS of M-gaps are
realised, at depths beyond the dictionary's own order m.  We only have order
m, so we take the ORDER-m CLOSURE: a walk is admitted iff EVERY contiguous
m-window of it is in the dictionary.  A realised walk has all its m-windows
realised, so the closure is a SUPERSET - never a subset.  Hence

    output  CONTAINS  every realised m-tuple of machine M + q'.

That is exactly the shape Formalist asked for in round 23 ("E with
`hE : realised 4-tuples subset E` as a named hypothesis"), so a superset is
the right object, not a compromise - PROVIDED it is not much bigger than the
truth.  This tool measures that inflation at every step where the true
dictionary is known.

Usage:
  uv run python research/dict_transfer.py validate
  uv run python research/dict_transfer.py 31 37 --f4 105 --f1 88
  uv run python research/dict_transfer.py 37 41 --f4 145 --f1 91 \
        --in research/data/gap_tuples_37_4.csv
"""
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# F_1 = F(M') exact (corpus ladder) and F_4(M') caps.
# F_4 exact where scanned; above the wall from the DELETION-LADDER BOUND
# F_{r+1}(M) <= F(M + r more gears) (old-machine-spectrum.md Cor B).
F1_CAP = {29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118}
F4_CAP = {29: 70, 31: 90, 37: 105,
          41: 145,      # F_4(41) <= F(41+43+47+53) ... = F(53) = 145
          43: 175}      # F_4(43) <= F_3(43) + F(43) <= F(53) + 103


def load_dict(path):
    rows = open(path).read().strip().split("\n")[1:]
    return [tuple(int(x) for x in r.split(",")) for r in rows]


def induced(tuples, m):
    """All contiguous m-windows of the dictionary's tuples."""
    out = set()
    M = len(tuples[0])
    for t in tuples:
        for i in range(M - m + 1):
            out.add(t[i:i + m])
    return out


def build_next(tuples):
    """ctx (len 0..M-1) -> sorted list of admissible next gaps."""
    M = len(tuples[0])
    nxt = defaultdict(set)
    for m in range(1, M + 1):
        for t in induced(tuples, m):
            nxt[t[:-1]].add(t[-1])
    return {k: sorted(v) for k, v in nxt.items()}


def transfer(tuples, qp, f4cap, f1cap, out_m=4, node_cap=200_000_000,
             verbose=True):
    """Superset of machine (M + qp)'s realised out_m-tuple dictionary."""
    M = len(tuples[0])
    nxt = build_next(tuples)
    up = pow(6, -1, qp)
    s = (2 * up) % qp
    res = set()
    nodes = 0
    layer = defaultdict(int)          # #kills -> emitted (word, phase) pairs
    t0 = time.time()

    def step(ctx, d, nsurv, cur, out, nkill):
        nonlocal nodes
        for g in nxt.get(ctx, ()):
            d2 = d + g
            cur2 = cur + g
            if cur2 > f1cap or d2 > f4cap:
                continue
            nodes += 1
            c2 = (ctx + (g,))[-(M - 1):]
            r = d2 % qp
            if r == A or r == B:                    # deleted by q'
                step(c2, d2, nsurv, cur2, out, nkill + 1)
            else:                                   # survives
                o2 = out + (cur2,)
                if nsurv + 1 == out_m:
                    res.add(o2)
                    layer[nkill] += 1
                else:
                    step(c2, d2, nsurv + 1, 0, o2, nkill)

    for A in range(qp):
        B = (A - s) % qp
        if A == 0 or B == 0:          # y_0 itself must survive
            continue
        step((), 0, 0, 0, (), 0)
        if nodes > node_cap:
            raise RuntimeError(f"node cap {node_cap} exceeded at phase {A}")
    dt = time.time() - t0
    if verbose:
        print(f"  transfer + gear {qp} (u'={up}, s={s}), caps F_1<={f1cap} "
              f"F_{out_m}<={f4cap}: {len(res):,} {out_m}-tuples, "
              f"{nodes:,} DFS nodes, {dt:.0f}s")
        tot = sum(layer.values())
        prof = "  ".join(f"{k}:{layer[k]:,}({100*layer[k]/tot:.2f}%)"
                         for k in sorted(layer))
        print(f"    emissions by #deleted interiors: {prof}")
    return res, nodes, dict(layer)


VALIDATION = [
    (23, 29, "gap_tuples_23_4.csv", "gap_tuples_29_4.csv"),
    (29, 31, "gap_tuples_29_4.csv", "gap_tuples_31_4.csv"),
]


def validate():
    print("DICTIONARY TRANSFER validation: the transfer of machine M's exact "
          "4-tuple dictionary must CONTAIN machine (M+q')'s exact one.")
    ok = True
    for M, qp, fin, ftrue in VALIDATION:
        src = load_dict(os.path.join(DATA, fin))
        true = set(load_dict(os.path.join(DATA, ftrue)))
        print(f"\n  {M} -> {qp}: source dictionary {len(src):,} 4-tuples, "
              f"true target {len(true):,}")
        got, nodes, _ = transfer(src, qp, F4_CAP[qp], F1_CAP[qp])
        missing = true - got
        extra = got - true
        good = not missing
        ok &= good
        print(f"    contains the truth: {good} "
              f"(missing {len(missing)}), inflation "
              f"{len(got)/len(true):.4f}x ({len(extra):,} extra)")
        if missing:
            print(f"    MISSING EXAMPLES {sorted(missing)[:5]}")
    print("\n=> " + ("TRANSFER IS A VALID SUPERSET AT EVERY VALIDATION STEP"
                     if ok else "UNSOUND - do not use"))
    return ok


def main():
    args = sys.argv[1:]

    def popopt(name, default=None, cast=str):
        if name in args:
            i = args.index(name)
            v = cast(args[i + 1])
            del args[i:i + 2]
            return v
        return default

    if not args or args[0] == "validate":
        validate()
        return
    inp = popopt("--in", None, str)
    outp = popopt("--out", None, str)
    M, qp = int(args[0]), int(args[1])
    f4 = popopt("--f4", F4_CAP.get(qp), int)
    f1 = popopt("--f1", F1_CAP.get(qp), int)
    inp = inp or os.path.join(DATA, f"gap_tuples_{M}_4.csv")
    src = load_dict(inp)
    print(f"machine {M} -> {M + 0} + gear {qp}: source {inp} "
          f"({len(src):,} 4-tuples)")
    got, nodes, layer = transfer(src, qp, f4, f1)
    outp = outp or os.path.join(DATA, f"gap_tuples_{qp}_4_transfer.csv")
    with open(outp, "w") as f:
        f.write("g1,g2,g3,g4\n")
        for t in sorted(got):
            f.write(",".join(map(str, t)) + "\n")
    print(f"  wrote {outp}  ({len(got):,} tuples - a certified SUPERSET)")
    for m2 in (1, 2, 3):
        print(f"  induced {m2}-tuple dictionary: "
              f"{len(induced(sorted(got), m2)):,}")


if __name__ == "__main__":
    main()

"""Round 17 addendum (mechanic): THE HOLE STRUCTURE OF THE GAP SPECTRUM.

Rounds 14-15 established that padding supply is arithmetically selected, not
smooth, and that the operational rule is "look hist_M[v] up".  That is where
the analysis stopped.  This tool starts from there and treats the ABSENCES as
an object: for each machine, which values v < F(M) never occur as a gap, how
the surviving counts depend on v's residues rather than on v's size, and
whether an absence is inherited when the next gear is added.

Input: research/data/flank_envelope_gaphist.csv (full-period gap histograms
written by flank_envelope.py; a machine probed n times appears n times, so
rows are deduplicated by (y, gap)).

Usage: uv run python research/hole_structure.py
"""
import csv
import os
from collections import defaultdict

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load():
    gh = defaultdict(dict)
    with open(os.path.join(D, "flank_envelope_gaphist.csv")) as f:
        for r in csv.DictReader(f):
            y, g, c = int(r["y"]), int(r["gap"]), int(r["count"])
            if g not in gh[y]:
                gh[y][g] = c
    return gh


def main():
    gh = load()
    ys = sorted(gh)
    print("=" * 76)
    print("(1) THE HOLE LIST - values v < F(M) with hist_M[v] = 0, full period")
    print("=" * 76)
    holes = {}
    for y in ys:
        h = gh[y]
        F = max(h)
        holes[y] = [v for v in range(1, F) if h.get(v, 0) == 0]
        occ = sum(h.values())
        print(f"  machine {y:3d}: F = {F:3d}, {len(h)} distinct gap values, "
              f"{occ:,} gaps, {len(holes[y])} holes below F")
        print(f"     holes: {holes[y]}")

    print("=" * 76)
    print("(2) IS AN ABSENCE INHERITED?  hole at M vs the next machine M'")
    print("=" * 76)
    for a, b in zip(ys, ys[1:]):
        Fa, Fb = max(gh[a]), max(gh[b])
        ha = set(holes[a])
        hb = set(holes[b])
        # only values below BOTH maxima can be compared
        common = [v for v in range(1, min(Fa, Fb))]
        inh = [v for v in common if v in ha and v in hb]
        heal = [v for v in common if v in ha and v not in hb]
        new = [v for v in common if v not in ha and v in hb]
        print(f"  {a:3d} -> {b:3d}: below min(F) = {min(Fa,Fb)}: "
              f"{len(inh)} inherited, {len(heal)} healed, {len(new)} new")
        if heal:
            print(f"     HEALED (absent at {a}, present at {b}): {heal}")
        if new:
            print(f"     NEW    (present at {a}, absent at {b}): {new}")

    print("=" * 76)
    print("(3) DOES hist_M[v] DEPEND ON v's RESIDUES RATHER THAN v's SIZE?")
    print("    share of all gaps by v mod p, against the flat share 1/p")
    print("=" * 76)
    for y in ys:
        h = gh[y]
        tot = sum(h.values())
        line = []
        for p in (2, 3, 5, 7):
            by = defaultdict(int)
            for v, c in h.items():
                by[v % p] += c
            sh = [by[r] / tot for r in range(p)]
            line.append(f"mod {p}: " +
                        " ".join(f"{x*p:.2f}" for x in sh))
        print(f"  machine {y:3d}  " + " | ".join(line))
    print("  (each entry is the class's share x p, so 1.00 = flat)")

    print("=" * 76)
    print("(4) THE TOP OF THE SPECTRUM - counts of the largest values")
    print("=" * 76)
    for y in ys:
        h = gh[y]
        F = max(h)
        tail = [(v, h.get(v, 0)) for v in range(max(1, F - 12), F + 1)]
        print(f"  machine {y:3d} (F = {F}): " +
              " ".join(f"{v}:{c}" for v, c in tail))

    print("=" * 76)
    print("(5) HOLE RESIDUES - do the holes sit in particular classes?")
    print("=" * 76)
    for y in ys:
        if not holes[y]:
            continue
        for p in (5, 7):
            by = defaultdict(int)
            for v in holes[y]:
                by[v % p] += 1
            print(f"  machine {y:3d} holes mod {p}: " +
                  " ".join(f"{r}:{by[r]}" for r in range(p)))
    residue_model(gh, ys)


def residue_model(gh, ys):
    print("=" * 76)
    print("(6) THE RESIDUE LAW TEST: is hist_M[v] predicted by v's residues")
    print("    rather than by v?  Score R(v) = prod_p share_p(v mod p) over")
    print("    p = 2,3,5,7, fitted from that machine's own marginals; then")
    print("    rank every value in the top half of the spectrum by R and see")
    print("    where the HOLES land.")
    print("=" * 76)
    for y in ys:
        h = gh[y]
        F = max(h)
        tot = sum(h.values())
        f = {}
        for p in (2, 3, 5, 7):
            by = defaultdict(int)
            for v, c in h.items():
                by[v % p] += c
            f[p] = [by[r] * p / tot for r in range(p)]
        lo = F // 2
        vals = list(range(lo, F + 1))
        score = {}
        for v in vals:
            s_ = 1.0
            for p in (2, 3, 5, 7):
                s_ *= f[p][v % p]
            score[v] = s_
        order = sorted(vals, key=lambda v: score[v])
        hl = [v for v in vals if h.get(v, 0) == 0]
        pos = [(v, order.index(v) + 1) for v in hl]
        print(f"  machine {y:3d} (F = {F}, {len(vals)} values in [{lo},{F}]):")
        print("     lowest-scoring 6: " +
              " ".join(f"{v}(R={score[v]:.2f},n={h.get(v,0)})"
                       for v in order[:6]))
        print("     holes and their rank by R (1 = lowest score): " +
              (" ".join(f"v={v} rank {r}/{len(vals)}" for v, r in pos)
               or "none"))


if __name__ == "__main__":
    main()

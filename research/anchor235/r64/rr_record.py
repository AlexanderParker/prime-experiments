"""rr_record.py -- every configuration that realises a gap of a given size, enumerated exactly
from the residues, and the two neighbours of each.

Branch: research/proof/record_2run.md (node 4.i.a.i.a.1.a.i).

In tooth units lam_g = 6x (mod g) every gear has its teeth at +-1 (pinned_arithmetic.md 2.1), so

    gear g strikes the column at offset j from x   iff   lam_g + 6 j = +-1 (mod g).

A gap of size v starting at x is: offsets 0 and v open, every offset in (0, v) struck.  A choice
of one class lam_g per gear is a configuration and by CRT occurs at exactly one column of the
period, so the FULL assignments satisfying the above are in bijection with the gaps of size v:
their number is m(v) and, walking outward from each, we get that occurrence's two neighbours.
Nothing is scanned; this reaches machines whose period no scan can touch.

Enumeration: gears in a fixed order, each taking one of its admissible classes (those striking
neither offset 0 nor offset v), with two prunes -- the reachability prune (an uncovered column no
remaining gear can strike kills the node) and the capacity prune (uncovered columns more than the
remaining gears can strike).

Usage:  uv run python rr_record.py <top gear> [size] [nodecap]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
LADDER = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}


def strikes(g, lam, j):
    r = (lam + 6 * j) % g
    return r == 1 % g or r == (g - 1) % g


def gear_options(g, v):
    """admissible classes for gear g (missing offsets 0 and v), each with its interior mask."""
    out = []
    for lam in range(g):
        if strikes(g, lam, 0) or strikes(g, lam, v):
            continue
        m = 0
        for j in range(1, v):
            if strikes(g, lam, j):
                m |= 1 << (j - 1)
        out.append((m, lam))
    return out


class Enum:
    def __init__(self, gears, v, nodecap):
        self.gears = list(gears)
        self.v = v
        self.full = (1 << (v - 1)) - 1
        self.opts = {g: gear_options(g, v) for g in gears}
        self.reach = {g: 0 for g in gears}
        self.cap = {g: 0 for g in gears}
        for g in gears:
            for m, _ in self.opts[g]:
                self.reach[g] |= m
                self.cap[g] = max(self.cap[g], bin(m).count("1"))
        # order: most constrained first (fewest classes), ties by gear
        self.order = sorted(gears, key=lambda g: (len(self.opts[g]), g))
        k = len(self.order)
        self.suffix_reach = [0] * (k + 1)
        self.suffix_cap = [0] * (k + 1)
        for i in range(k - 1, -1, -1):
            g = self.order[i]
            self.suffix_reach[i] = self.suffix_reach[i + 1] | self.reach[g]
            self.suffix_cap[i] = self.suffix_cap[i + 1] + self.cap[g]
        self.nodes = 0
        self.nodecap = nodecap
        self.sols = []

    def run(self):
        self._dfs(0, 0, {})
        return self.sols, self.nodes

    def _dfs(self, i, covered, assign):
        self.nodes += 1
        if self.nodes > self.nodecap:
            raise RuntimeError("node cap")
        miss = self.full & ~covered
        if miss & ~self.suffix_reach[i]:
            return
        if bin(miss).count("1") > self.suffix_cap[i]:
            return
        if i == len(self.order):
            if covered == self.full:
                self.sols.append(dict(assign))
            return
        g = self.order[i]
        for m, lam in self.opts[g]:
            assign[g] = lam
            self._dfs(i + 1, covered | m, assign)
        assign.pop(g, None)


def neighbours(gears, lam, v, limit=400):
    """given a full configuration, the gaps immediately left of offset 0 and right of offset v."""
    R = None
    for t in range(1, limit):
        if not any(strikes(g, lam[g], v + t) for g in gears):
            R = t
            break
    L = None
    for t in range(1, limit):
        if not any(strikes(g, lam[g], -t) for g in gears):
            L = t
            break
    return L, R


def strikers(gears, lam, j):
    return [g for g in gears if strikes(g, lam[g], j)]


def report(top, v=None, nodecap=2_000_000_000):
    gears = [g for g in PRIMES if g <= top]
    if v is None:
        v = LADDER[top]
    t0 = time.time()
    e = Enum(gears, v, nodecap)
    sols, nodes = e.run()
    dt = time.time() - t0
    rows = []
    for lam in sols:
        L, R = neighbours(gears, lam, v)
        sole = sole_map(gears, lam, v)
        busy = {g for g in gears if sole[g]}
        # the columns of the two NEIGHBOUR gaps, and who strikes them
        outside = []
        for j in list(range(v + 1, v + R)) + list(range(-L + 1, 0)):
            st = strikers(gears, lam, j)
            outside.append({"j": j, "strikers": st,
                            "all_busy_inside": all(g in busy for g in st)})
        rows.append({
            "lam": {g: lam[g] for g in gears},
            "L": L, "R": R,
            "strikers_right": strikers(gears, lam, v + 1),
            "strikers_left": strikers(gears, lam, -1),
            "sole_strikers": sole,
            "n_busy": len(busy),
            "outside": outside,
        })
    n1 = max(max(r["L"], r["R"]) for r in rows) if rows else None
    Nsum = max(r["L"] + r["R"] for r in rows) if rows else None
    nbusy = [r["n_busy"] for r in rows]
    outs = [o for r in rows for o in r["outside"]]
    ends = [len(r["strikers_left"]) for r in rows if r["L"] > 1] + \
           [len(r["strikers_right"]) for r in rows if r["R"] > 1]
    d = {"gears": gears, "v": v, "m_v": len(rows), "n1": n1, "N": Nsum,
         "nodes": nodes, "seconds": dt,
         "classes_per_gear": {g: len(e.opts[g]) for g in gears},
         "all_gears_busy": all(nb == len(gears) for nb in nbusy),
         "outside_cols": len(outs),
         "outside_all_busy": sum(1 for o in outs if o["all_busy_inside"]),
         "end_striker_counts": ends,
         "rows": rows}
    return d


def sole_map(gears, lam, v):
    """interior columns struck by exactly one gear, per gear."""
    out = {g: [] for g in gears}
    for j in range(1, v):
        s = strikers(gears, lam, j)
        if len(s) == 1:
            out[s[0]].append(j)
    return {g: out[g] for g in gears}


if __name__ == "__main__":
    top = int(sys.argv[1])
    v = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000_000
    d = report(top, v, cap)
    tag = f"{top}_{d['v']}"
    with open(os.path.join(OUT, f"rec_{tag}.json"), "w") as f:
        json.dump(d, f, default=int)
    print(f"M={{5..{top}}} v={d['v']}  m(v)={d['m_v']}  n1={d['n1']}  N={d['N']}  "
          f"nodes={d['nodes']}  {d['seconds']:.1f}s")
    print("  classes per gear:", d["classes_per_gear"])
    print(f"  every gear busy (sole striker inside) at every occurrence: {d['all_gears_busy']}; "
          f"outside columns whose every striker is busy inside: "
          f"{d['outside_all_busy']}/{d['outside_cols']}; "
          f"strikers on the first outside column: {d['end_striker_counts']}")
    for r in d["rows"]:
        print(f"   L={r['L']} R={r['R']} lam={r['lam']}")
        print(f"      strikers just left = {r['strikers_left']}, "
              f"just right = {r['strikers_right']}")
        print("      sole strikers: "
              + " ".join(f"{g}:{len(v2)}" for g, v2 in r["sole_strikers"].items()))

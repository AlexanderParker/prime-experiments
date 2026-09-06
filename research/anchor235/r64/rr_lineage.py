"""rr_lineage.py -- the lineage of the record and of F_2, scan-free.

Branch: research/proof/record_2run.md (node 4.i.a.i.a.1.a.i).

A configuration of M + q' is a class lam_g per gear (tooth units, teeth at +-1).  Enumerating every
configuration that realises a prescribed run of openings gives every occurrence of that run in the
period; DROPPING the top gear q' from the same configuration and re-reading which offsets are open
gives the run's decomposition into gaps of M -- its fusion word (merge law, docs/proofs/05).  So
the lineage of F(M + q') and of F_2(M + q') is read off the same enumeration, with no period
scanned and no merge forest built.

Usage:  uv run python rr_lineage.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr_record import PRIMES, strikes, LADDER            # noqa: E402

OUT = os.path.join(HERE, "results")


def gear_options_run(g, opens, S):
    out = []
    for lam in range(g):
        if any(strikes(g, lam, o) for o in opens):
            continue
        m = 0
        for j in range(0, S + 1):
            if j in opens:
                continue
            if strikes(g, lam, j):
                m |= 1 << j
        out.append((m, lam))
    return out


class EnumRun:
    def __init__(self, gears, opens, S, nodecap):
        self.gears = list(gears)
        self.S = S
        self.opens = set(opens)
        self.full = 0
        for j in range(0, S + 1):
            if j not in self.opens:
                self.full |= 1 << j
        self.opts = {g: gear_options_run(g, opens, S) for g in gears}
        self.reach = {}
        self.cap = {}
        for g in gears:
            r = 0
            c = 0
            for m, _ in self.opts[g]:
                r |= m
                c = max(c, bin(m).count("1"))
            self.reach[g], self.cap[g] = r, c
        self.order = sorted(gears, key=lambda g: (len(self.opts[g]), g))
        k = len(self.order)
        self.sr = [0] * (k + 1)
        self.sc = [0] * (k + 1)
        for i in range(k - 1, -1, -1):
            g = self.order[i]
            self.sr[i] = self.sr[i + 1] | self.reach[g]
            self.sc[i] = self.sc[i + 1] + self.cap[g]
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
        if miss & ~self.sr[i]:
            return
        if bin(miss).count("1") > self.sc[i]:
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


def word_below(gears, lam, S, drop):
    """the offsets in [0, S] open for the machine WITHOUT gear `drop`, and the gap word."""
    sub = [g for g in gears if g != drop]
    op = [j for j in range(0, S + 1) if not any(strikes(g, lam[g], j) for g in sub)]
    return op, [op[i + 1] - op[i] for i in range(len(op) - 1)]


def rank_fraction(prof, p):
    """|{realised sizes >= p}| / |{realised sizes}| in the machine below (0 = the record)."""
    sizes = sorted(int(s) for s in prof["spec"])
    above = [s for s in sizes if s >= p]
    return (len(above) - 1) / len(sizes)


def main():
    prof = {}
    for f in ("prof_23.json", "prof_29.json", "prof_31.json"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            prof.update(json.load(open(p)))
    prof = {int(k): v for k, v in prof.items()}
    rows = []
    for i in range(3, len(PRIMES)):
        top = PRIMES[i]
        below = PRIMES[i - 1]
        if top not in LADDER or below not in prof:
            continue
        gears = PRIMES[:i + 1]
        F = LADDER[top]
        # --- the record's lineage
        e = EnumRun(gears, {0, F}, F, 3_000_000_000)
        sols, nodes = e.run()
        wordsF = sorted({tuple(word_below(gears, lam, F, top)[1]) for lam in sols})
        # --- F_2's lineage
        pr = prof.get(top)
        wordsF2 = None
        F2 = None
        if pr:
            F2 = pr["F2"]
            n1 = {int(k): v for k, v in pr["n1"].items()}
            vstar = max((v for v in n1 if v + n1[v] == F2))
            a = F2 - vstar
            e2 = EnumRun(gears, {0, vstar, F2}, F2, 3_000_000_000)
            s2, n2 = e2.run()
            wordsF2 = sorted({tuple(word_below(gears, lam, F2, top)[1]) for lam in s2})
            wordsF2 = (vstar, a, wordsF2, len(s2), n2)
        pb = prof[below]
        rows.append({
            "M": f"{{5..{below}}}", "q": top, "F_new": F, "F2_new": F2,
            "F_words": wordsF, "nodes": nodes,
            "F_ranks": [[round(rank_fraction(pb, p), 3) for p in w] for w in wordsF],
            "F2": wordsF2,
            "F2_ranks": ([[round(rank_fraction(pb, p), 3) for p in w] for w in wordsF2[2]]
                         if wordsF2 else None),
            "top3_below": sorted((int(s) for s in pb["spec"]), reverse=True)[:3],
        })
        r = rows[-1]
        print(f"\nM={r['M']} + {top}:  F(M+q') = {F}")
        print(f"  record fusion words (all occurrences, {len(sols)} configs, {nodes} nodes): "
              f"{r['F_words']}")
        print(f"  piece rank fractions in M: {r['F_ranks']}   top 3 sizes of M: "
              f"{r['top3_below']}")
        if wordsF2:
            print(f"  F_2(M+q') = {F2} at ({wordsF2[0]}, {wordsF2[1]}), "
                  f"{wordsF2[3]} configs, {wordsF2[4]} nodes")
            print(f"  F_2 fusion words: {wordsF2[2]}")
            print(f"  piece rank fractions in M: {r['F2_ranks']}")
    with open(os.path.join(OUT, "lineage.json"), "w") as f:
        json.dump(rows, f, default=str)


if __name__ == "__main__":
    main()

"""og_nearest.py -- P7: the distance from the real teeth to the nearest killer of the finer section.

d(p) = the least number of gears whose teeth must be moved from the real teeth +-k_g so that the family
member strikes every column of the section at the cut p. Search: for d = 1, 2, ..., every d-subset of
the gears; the other gears keep their real teeth and cover a fixed set; the d moved gears must cover the
rest with some choice of teeth (a small exact-cover search, teeth 1..(g-1)/2 each, the real tooth
excluded). Reports d(p), every nearest killer's moved gears (up to 20), and whether a tail gear
(g > section length) is among the moved gears in every nearest killer.
Cuts: p = 11..53 (cuts without any killer report d = infinity, decided by the same search to d = d_max
or by the known family verdicts of first_realisation.md 3.5: no killer at 7, 11, 13, 19, 23, 31).
Output: results/nearest.json
"""
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from og_common import finer_section, real_teeth

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def masks(gears, cols):
    per = {}
    for g in gears:
        rows = {}
        for v in range(1, (g - 1) // 2 + 1):
            m = 0
            for i, j in enumerate(cols):
                if (j % g) in (v, g - v):
                    m |= 1 << i
            rows[v] = m
        per[g] = rows
    return per


def cover_with(moved, per, rt, need, examples, gears_all, limit):
    """can the moved gears (each with a tooth != its real tooth) cover `need`? DFS; collects examples"""
    found = [0]

    def rec(idx, unc, chosen):
        if unc == 0:
            found[0] += 1
            if len(examples) < limit:
                examples.append(dict(chosen))
            return
        if idx == len(moved):
            return
        g = moved[idx]
        # capacity prune: remaining gears' best coverage
        rem = moved[idx:]
        cap = 0
        for h in rem:
            best = max(bin(m & unc).count("1") for v, m in per[h].items() if v != rt[h])
            cap += best
        if cap < bin(unc).count("1"):
            return
        for v, m in per[g].items():
            if v == rt[g]:
                continue
            chosen[g] = v
            rec(idx + 1, unc & ~m, chosen)
            del chosen[g]
            if found[0] and len(examples) >= limit:
                return

    rec(0, need, {})
    return found[0]


def nearest(p, d_max=6):
    a, b, cols, gears = finer_section(p)
    L = len(cols)
    full = (1 << L) - 1
    per = masks(gears, cols)
    rt_list = real_teeth(gears)
    rt = dict(zip(gears, rt_list))
    real_cover = 0
    for g in gears:
        real_cover |= per[g][rt[g]]
    twins = [cols[i] for i in range(L) if not (real_cover >> i) & 1]
    t0 = time.time()
    for d in range(1, d_max + 1):
        examples = []
        n_subsets_ok = 0
        subsets_ok = []
        for sub in itertools.combinations(gears, d):
            fixed = 0
            for g in gears:
                if g not in sub:
                    fixed |= per[g][rt[g]]
            need = full & ~fixed
            ex = []
            cnt = cover_with(list(sub), per, rt, need, ex, gears, 3)
            if cnt:
                n_subsets_ok += 1
                subsets_ok.append(list(sub))
                for e in ex:
                    examples.append(e)
        if n_subsets_ok:
            all_have_tail = all(any(g > L for g in s) for s in subsets_ok)
            return dict(p=p, section=(cols[0], cols[-1]), length=L, twins=twins, d=d,
                        n_moved_sets=n_subsets_ok, moved_sets=subsets_ok[:20],
                        all_nearest_move_a_tail_gear=all_have_tail,
                        examples=[{str(k): v for k, v in e.items()} for e in examples[:5]],
                        seconds=round(time.time() - t0, 1))
    return dict(p=p, section=(cols[0], cols[-1]), length=L, twins=twins, d=None, d_max_searched=d_max,
                seconds=round(time.time() - t0, 1))


def main():
    cuts = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    out = []
    for p in cuts:
        r = nearest(p, d_max=5 if p >= 37 else 6)
        print(f"p={p}: L={r['length']} twins={r['twins']} d={r['d']} moved sets={r.get('n_moved_sets')} "
              f"{r.get('moved_sets', '')[:6]} all move a tail gear={r.get('all_nearest_move_a_tail_gear')} "
              f"examples={r.get('examples', '')[:2]} ({r['seconds']}s)", flush=True)
        out.append(r)
        with open(os.path.join(RES, "nearest.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

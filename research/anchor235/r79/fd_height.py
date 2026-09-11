"""fd_height.py -- columns hit by exactly one field, at the origin against at height (fields.md section 4; P5).

Height: the four record runs of origin_mechanic.md table E (every column struck):
  m23 at column 12,694,429 (33 columns), m29 at 200,906,186 (42), m31 at 1,468,940,243 (57), m37 at 90,816,580,903 (87);
plus 20 random stretches of the same length at columns uniform in [x/2, 2x] as controls (sympy.factorint).
Origin: the finer sections p = 23..53 are in fields.json (fd_fields.py); here only the height side is computed and
the origin numbers are copied in for the table.
Usage: uv run python fd_height.py
"""
import os, json, time, random
import numpy as np
from collections import Counter
from fd_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(RES, "height.log"), "w")

RUNS = [(23, 12_694_429, 33), (29, 200_906_186, 42), (31, 1_468_940_243, 57), (37, 90_816_580_903, 87)]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def stretch_census(x, Lc):
    """column signatures of the stretch of Lc columns from x, by factorint"""
    sig = Counter()
    one = Counter()
    fields = Counter()
    sides = Counter()
    struck = 0
    open_ = 0
    for k in range(x, x + Lc):
        oms = []
        for side, n in ((-1, 6 * k - 1), (1, 6 * k + 1)):
            om, omm, lpf = factor_omega(n)
            if om >= 2:
                oms.append(om)
                fields[om] += 1
                sides[(om, side)] += 1
        if not oms:
            open_ += 1
            continue
        struck += 1
        c = tuple(sorted(oms))
        sig[c] += 1
        if len(set(c)) == 1:
            one[c[0]] += 1
    n1 = sum(one.values())
    return dict(struck=struck, open=open_, sig={"+".join(map(str, k)): v for k, v in sorted(sig.items())}, one=dict(one),
                one_share=n1 / struck if struck else None, one_F2_share=one[2] / n1 if n1 else None, fields=dict(fields),
                F2_right=sides[(2, 1)], F2_left=sides[(2, -1)])


def main():
    t0 = time.time()
    random.seed(79)
    results = {}
    say("# origin against height", time.ctime())
    say("| run | columns | open | struck | one-field share | F2 share of one-field | signatures | F2 L/R | controls: one-field share mean (min-max) | controls F2 share mean |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for q, x, Lc in RUNS:
        r = stretch_census(x, Lc)
        ctrl = []
        for _ in range(20):
            y = random.randint(x // 2, 2 * x)
            ctrl.append(stretch_census(y, Lc))
        cs = [c["one_share"] for c in ctrl if c["one_share"] is not None]
        cf = [c["one_F2_share"] for c in ctrl if c["one_F2_share"] is not None]
        results[f"m{q}"] = dict(run=r, controls=ctrl)
        say(f"| m{q} at {x} | {Lc} | {r['open']} | {r['struck']} | {r['one_share']:.3f} | {r['one_F2_share']:.3f} | {r['sig']} | {r['F2_left']}/{r['F2_right']} | "
            f"{np.mean(cs):.3f} ({min(cs):.3f}-{max(cs):.3f}) | {np.mean(cf):.3f} |")
    json.dump(results, open(os.path.join(RES, "height.json"), "w"), indent=1, default=int)
    say("done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

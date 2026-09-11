"""og_height.py -- P5: the same census at height, against the origin.

At the four record runs at height (length_face.md 1.1, 3.2): m23 33 columns at 12,694,429; m29 42 at
200,906,186; m31 57 at 1,468,940,243; m37 87 at 90,816,580,903: per column the least striker, the
quotient by it, whether the quotient is prime, the depth Omega(n) of the struck member (sympy factorint).
Controls: 100 random stretches of the same length at columns uniform in [x/2, 2x] around the record's x.
Origin: the finer section at the same engine (from og_census.py's method, recomputed here directly).
Output: results/height.json
"""
import json
import os
import random
import sys

from sympy import factorint

sys.path.insert(0, os.path.dirname(__file__))
from og_common import finer_section, gears_of, nextprime

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

RECORDS = {23: (12694429, 33), 29: (200906186, 42), 31: (1468940243, 57), 37: (90816580903, 87)}


def census_stretch(x, L, q):
    gears = gears_of(q)
    rows = []
    n_open = 0
    for j in range(x, x + L):
        best = None
        for s in (-1, 1):
            n = 6 * j + s
            for g in gears:
                if n % g == 0:
                    if best is None or g < best[0]:
                        best = (g, n)
                    break
        if best is None:
            n_open += 1
            continue
        g, n = best
        m = n // g
        f = factorint(int(n))
        depth = int(sum(int(v) for v in f.values()))
        fm = factorint(int(m)) if m > 1 else {}
        m_prime = bool(len(fm) == 1 and int(list(fm.values())[0]) == 1)
        rows.append(dict(j=int(j), g0=int(g), n=int(n), m=int(m), m_prime=m_prime, depth=depth,
                         below_cube=bool(n < g ** 3)))
    return rows, n_open


def summarize(rows):
    k = len(rows)
    if not k:
        return {}
    pq = sum(r["m_prime"] for r in rows)
    bc = sum(r["below_cube"] for r in rows)
    dh = {}
    for r in rows:
        dh[r["depth"]] = dh.get(r["depth"], 0) + 1
    return dict(struck=int(k), prime_quotient=int(pq), share=round(float(pq) / k, 3), below_cube=int(bc),
                depth_hist={int(a): int(b) for a, b in sorted(dh.items())},
                mean_depth=round(float(sum(r["depth"] for r in rows)) / k, 3),
                max_depth=int(max(r["depth"] for r in rows)))


def main():
    random.seed(77)
    out = {}
    for q, (x, L) in RECORDS.items():
        rows, n_open = census_stretch(x, L, q)
        rec = summarize(rows)
        rec["open"] = n_open
        # controls
        ctrl_rows = []
        ctrl_open = 0
        for _ in range(100):
            y = random.randint(x // 2, 2 * x)
            r2, o2 = census_stretch(y, L, q)
            ctrl_rows += r2
            ctrl_open += o2
        ctrl = summarize(ctrl_rows)
        ctrl["open_per_stretch"] = ctrl_open / 100
        # origin: the finer section at q
        a, b, cols, gears = finer_section(q)
        r3, o3 = census_stretch(cols[0], len(cols), q)
        orig = summarize(r3)
        orig["open"] = o3
        out[q] = dict(record=dict(x=x, L=L, **rec), controls=ctrl, origin=dict(cols=(cols[0], cols[-1]), **orig))
        print(f"q={q}: record x={x} L={L}: prime-quotient share {rec['share']} ({rec['prime_quotient']}/{rec['struck']}), "
              f"below cube {rec['below_cube']}, depth {rec['depth_hist']}; controls share {ctrl['share']} depth mean "
              f"{ctrl['mean_depth']}; ORIGIN cols {cols[0]}..{cols[-1]} share {orig['share']} "
              f"({orig['prime_quotient']}/{orig['struck']}), below cube {orig['below_cube']}, depth {orig['depth_hist']}",
              flush=True)
    with open(os.path.join(RES, "height.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

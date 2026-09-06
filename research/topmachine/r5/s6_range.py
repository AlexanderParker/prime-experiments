"""The range laws with a small gear: a fixed gear set {q'..97} on [1, 10^7]."""

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from common import teeth  # noqa: E402

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
          83, 89, 97]
N = 10_000_000


def main():
    out = []
    for qp in (2, 3, 5, 7, 11, 13):
        gears = [p for p in PRIMES if p >= qp]
        mask = np.ones(N, dtype=bool)
        for g in gears:
            for t in teeth(g):
                mask[t::g] = False
        dens = float(mask.mean())
        prodv = 1.0
        for g in gears:
            prodv *= (g - len(teeth(g))) / g
        pos = np.flatnonzero(mask)
        gaps = np.diff(pos)
        i = int(np.argmax(gaps))
        rec = {"q'": qp, "gears": len(gears), "density": dens, "crt_product": prodv,
               "ratio": dens / prodv, "record_run": int(gaps[i]) - 1,
               "record_at": int(pos[i]) + 1,
               "first_open": int(pos[0]), "gap4_count": int((gaps == 4).sum()),
               "gap1_count": int((gaps == 1).sum())}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    with open(__file__.rsplit("s6_")[0] + "results/s6_range.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

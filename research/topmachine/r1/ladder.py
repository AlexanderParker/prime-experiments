"""The top machine's ladder: F_top as gears are added one at a time, and the record's
composition.  Uses the validated covering formulation (cover.py)."""

import json
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from cover import F_cover, feasible, witness  # noqa: E402

PRIMES = [
    7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
    193, 197, 199, 211,
]


def main():
    out = {}

    # 1. the large-gear parity law: m gears all > 2m+1
    rows = []
    for start in range(0, 20):
        for m in range(2, 12):
            gs = PRIMES[start : start + m]
            if len(gs) < m or gs[0] <= 2 * m + 1:
                continue
            f, st = F_cover(gs)
            pred = 2 * m if m % 2 == 0 else 2 * m - 1
            rows.append({"gears": gs, "m": m, "F": f, "pred": pred, "ok": f == pred, "st": st})
    out["parity_law"] = rows
    bad = [r for r in rows if not r["ok"]]
    print("parity law: %d cases, %d exceptions" % (len(rows), len(bad)), bad[:4])

    # 2. ladders
    out["ladders"] = []
    for q0 in (7, 11, 13, 17, 19, 23):
        i0 = PRIMES.index(q0)
        row = []
        prev = None
        for k in range(1, 22):
            gs = PRIMES[i0 : i0 + k]
            if len(gs) < k:
                break
            t = time.time()
            f, st = F_cover(gs)
            if st != "exact":
                row.append({"gears": gs, "Q": gs[-1], "m": k, "F": f, "status": st})
                break
            inc = None if prev is None else f - prev
            row.append({"gears": gs, "Q": gs[-1], "m": k, "F": f, "increment": inc,
                        "two_m": 2 * k, "secs": round(time.time() - t, 2)})
            prev = f
            if time.time() - t > 120:
                break
        out["ladders"].append({"q0": q0, "rungs": row})
        print("q'=%d:" % q0, [(r["Q"], r.get("F"), r.get("increment")) for r in row])

    # 3. composition of the record
    out["composition"] = []
    for gs in [[7, 11, 13], [11, 13, 17, 19], [7, 11, 13, 17, 19],
               [11, 13, 17, 19, 23, 29, 31], [17, 19, 23, 29, 31, 37, 41],
               [7, 11, 13, 17, 19, 23, 29, 31], [13, 17, 19, 23, 29, 31, 37, 41, 43]]:
        L, st = F_cover(gs)
        w = witness(gs, L)
        pieces = {}
        for k, v in (w or {}).items():
            if k == "pool":
                pieces["pool"] = [[x for x in range(L) if (m >> x) & 1] for m in v]
            else:
                pieces[k] = [x for x in range(L) if (v >> x) & 1]
        out["composition"].append({"gears": gs, "L": L, "pieces": pieces})
        print("compose", gs, "L=", L, pieces)

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

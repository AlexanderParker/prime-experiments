"""pa_rung41.py -- the row of the letter one rung beyond every scan: M = {5..37}, q' = 41.

a_L(41) = 14 (3 * 14 = 42 = q' + 1), F({5..37}) = 88 (recorded).  The period of {5..37} is
1.24e12 columns, so r(14) has never been measured; the CRT search decides it scan-free.
The pinned letter predicts r(14) <= 88 + 3 - 14 = 77.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pa_crt as m

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
aL, F = 14, 88
out = []
t0 = time.time()
top = None
for a in range(F, 0, -1):
    t = time.time()
    try:
        ok, sol, n = m.feasible(GEARS, aL, a)
    except RuntimeError:
        out.append((a, None, m.NODECAP, None))
        print(a, "NODECAP", flush=True)
        continue
    out.append((a, bool(ok), int(n), round(time.time() - t, 2)))
    print(a, ok, n, round(time.time() - t, 2), f"cum {time.time()-t0:.0f}s", flush=True)
    if ok:
        top = a
        break
json.dump(dict(gears=GEARS, aL=aL, F=F, top=top, rows=out),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                            "pa_rung41.json"), "w"))
print("TOP", top, "pinned bound", F + 3 - aL, "excess", (aL + top - F) if top else None)

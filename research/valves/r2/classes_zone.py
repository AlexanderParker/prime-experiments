"""Pure charges (twins) per pure-imprint class mod q# over the WHOLE zone (Q, Q^2], and the mirror pairing
of the classes (E-P5). usage: uv run python research/valves/r2/classes_zone.py Q q1 q2 ...
"""
import sys, math, json, os
from collections import Counter
import numpy as np

Q = int(sys.argv[1]); qs = [int(x) for x in sys.argv[2:]] or [5, 7]
here = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(here, "results"); os.makedirs(outdir, exist_ok=True)
N = Q * Q + 4
s = np.ones(N + 1, dtype=bool); s[:2] = False
for i in range(2, int(N ** 0.5) + 1):
    if s[i]:
        s[i * i::i] = False
tw = np.flatnonzero(s[:-2] & s[2:])
tw = tw[(tw > Q) & (tw <= Q * Q)]
print(f"zone ({Q}, {Q*Q}]: {len(tw)} twins")
for q in qs:
    engine = [p for p in range(2, q + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
    qsharp = math.prod(engine)
    pc = [r for r in range(qsharp) if math.gcd(r, qsharp) == 1 and math.gcd(r + 2, qsharp) == 1]
    cnt = Counter((tw % qsharp).tolist())
    counts = {r: cnt.get(r, 0) for r in pc}
    vals = np.array(list(counts.values()))
    mm = []
    seen = set()
    for r in pc:
        sm = (-r - 2) % qsharp
        if sm != r and (sm, r) not in seen:
            seen.add((r, sm)); mm.append((r, sm, counts[r], counts[sm]))
    dev = max(abs(a - b) for _, _, a, b in mm) if mm else 0
    print(f"q={q} q#={qsharp}: {len(pc)} pure classes; twins per class min {vals.min()} max {vals.max()} mean {vals.mean():.1f} "
          f"(sqrt(mean) {math.sqrt(vals.mean()):.0f}); spoke class {qsharp-1}: {counts[qsharp-1]}; {len(mm)} mirror pairs, "
          f"max |count(r) - count(-r-2)| = {dev}; classes with 0 twins: {int((vals == 0).sum())}")
    if q == 5:
        print(f"  classes mod 30: {counts}")
    with open(os.path.join(outdir, f"classes_zone_Q{Q}_q{q}.json"), "w") as f:
        json.dump(dict(counts=counts, mirror=mm), f)

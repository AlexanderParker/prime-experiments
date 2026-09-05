"""Branch 5d.ii.i.a, item 1: A(7) and A(8), exhaustive over ALL primes >= 5
by the type reduction (arc_core).  Usage: uv run python .../arc_a7.py K L0"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import Level, RESULTS

K = int(sys.argv[1]); L = int(sys.argv[2])
out = open(os.path.join(RESULTS, f"arc_A{K}.txt"), "w")
while True:
    t = time.time()
    lv = Level(L, K)
    ok = lv.coverable()
    line = (f"K={K} L={L}: {'cover' if ok else 'NO COVER'}  {lv.nodes} nodes, "
            f"{len(lv.items)} item types, {time.time()-t:.1f}s")
    print(line, flush=True); out.write(line + "\n"); out.flush()
    if ok:
        w = "  witness: " + str(lv.witness)
        out.write(w + "\n"); out.flush()
    else:
        out.write(f"*** A({K}) = {L}\n"); out.flush()
        break
    L += 1
out.close()

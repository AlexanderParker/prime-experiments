"""Round 29 (constructor): decide the eight phase-saturation survivors among
machine 43's length-3 legal words, individually and at a large node budget.
L(43) <= 2 (hence A_kill(43 -> 47) <= 3, J_max(43) = 4, Q*_5(43) = -inf)
requires every one of them to be REFUTED."""
import sys, time, os
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import crt_dict

WORDS = [(16,47,47),(31,47,47),(31,47,63),(47,16,47),
         (47,31,47),(47,47,16),(47,47,31),(63,47,31)]

def job(w):
    t0 = time.time()
    try:
        return w, crt_dict.realised(43, w, node_budget=60_000_000), time.time()-t0
    except Exception as e:
        return w, None, time.time()-t0

if __name__ == "__main__":
    with Pool(4) as p:
        res = list(p.imap_unordered(job, WORDS))
    und = []
    for w, ok, dt in sorted(res):
        print("  %-16s %-10s %.0f s" % (str(w), {True:"REALISED", False:"refuted", None:"UNDECIDED"}[ok], dt), flush=True)
        if ok is None: und.append(w)
        assert ok is not True, ("REALISED - L(43) >= 3", w)
    print("undecided: %s" % und)
    if not und:
        print("==> L(43) = 2 CERTIFIED: A_kill(43->47) <= 3, J_max(43) = 4, Q*_5(43) = -inf")

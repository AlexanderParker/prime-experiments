import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lp_degree_range import decide, gears_of
W = int(sys.argv[1])
t0=time.time(); f,info = decide(gears_of(19), W, 2)
print("W=%d: %s [%.0fs]" % (W, "feasible" if f else "INFEASIBLE (%d ops)"%info['ops'], time.time()-t0), flush=True)

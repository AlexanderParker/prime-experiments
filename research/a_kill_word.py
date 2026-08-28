"""Round 24 (mechanic): decide ONE kill-chain word - parallel driver unit for
a_kill.py's k=4/k=5 levels (the f3_one.py pattern: one instance per process).

usage: python research/a_kill_word.py y g1,g2,...   [--cap N]
Prints REALISED (with machine-verified witness) or ZERO.  Exit code 0 always;
the result line is the datum.
"""
import sys, time
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
from cov_count import count_pattern

args = sys.argv[1:]
cap = 1
if "--cap" in args:
    i = args.index("--cap"); cap = int(args[i+1]); del args[i:i+2]
y = int(args[0])
w = tuple(int(x) for x in args[1].split(","))
opens, acc = [], 0
for g in w[:-1]:
    acc += g; opens.append(acc)
S = sum(w)
t0 = time.time()
n, wits, calls = count_pattern(y, S, tuple(opens), cap=cap)
tag = f"REALISED n>={n}" if n else "ZERO"
print(f"RESULT m{y} word {w} span {S}: {tag} ({calls} calls, "
      f"{time.time()-t0:.1f}s)" + (f" wit {wits[0]}" if wits else ""),
      flush=True)

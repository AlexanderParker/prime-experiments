"""Round 21 (mechanic): run_j(M; V(q')) - exact residue-qualifying run counts
by COV-COUNT word enumeration, for Constructor's deep-run ladder (R39/R44).

A depth-j qualifying run = j consecutive gaps each with value in
V(q') = {0, +2u', -2u'} mod q' (u' = round(q'/6)).  Method:
  1. legal values = {v in [1, F(M)] : v mod q' in V, v not a hole of M}
  2. spectrum prune: any word with span > F_{j}(M) has count 0 (theorem:
     a j-window of span S occurs only if S <= F_j; F_j exact from COV/census)
  3. every surviving word: exact occurrence count by projected model
     enumeration (cov_count.count_pattern), witnesses machine-verified.
Usage: python run_count.py y qprime j Fj_cap [--cap N]
Fj_cap = exact F_j(M) for the spectrum prune (97 for m37 j=3; 85 for m31 j=3).
"""
import sys
import time
from itertools import product

sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
from cov_count import count_pattern
from cov_sat import MEASURED_F, MEASURED_HOLES


def main():
    args = sys.argv[1:]
    cap = 200000
    if "--cap" in args:
        i = args.index("--cap"); cap = int(args[i+1]); del args[i:i+2]
    y, qp, j, fj = int(args[0]), int(args[1]), int(args[2]), int(args[3])
    F = MEASURED_F[y]
    holes = set(MEASURED_HOLES.get(y, []))
    up = round(qp / 6)
    V = {0, (2*up) % qp, (-2*up) % qp}
    legal = [v for v in range(1, F+1) if v % qp in V and v not in holes]
    print(f"run_{j}({y}; V({qp})): u'={up}, V={sorted(V)} mod {qp}, "
          f"legal values {legal}, F_{j}({y}) = {fj} (spectrum prune)", flush=True)
    words = list(product(legal, repeat=j))
    pruned = [w for w in words if sum(w) > fj]
    live = [w for w in words if sum(w) <= fj]
    print(f"{len(words)} words: {len(pruned)} ZERO by spectrum prune "
          f"(span > F_{j}), {len(live)} to decide", flush=True)
    total = 0
    nz = []
    for w in sorted(live, key=sum):
        opens = []
        acc = 0
        for g in w[:-1]:
            acc += g; opens.append(acc)
        S = sum(w)
        t0 = time.time()
        n, wits, calls = count_pattern(y, S, tuple(opens), cap=cap)
        if n:
            nz.append((w, n))
            total += n
            print(f"  word {w} span {S}: count = {n} "
                  f"({time.time()-t0:.0f}s) wit {wits[0]}", flush=True)
        else:
            print(f"  word {w} span {S}: 0 ({time.time()-t0:.0f}s)", flush=True)
    print(f"TOTAL run_{j}({y}; V({qp})) = {total} over {len(nz)} nonzero words "
          f"(+{len(pruned)} spectrum-zero, {len(live)-len(nz)} refuted)", flush=True)


if __name__ == "__main__":
    main()

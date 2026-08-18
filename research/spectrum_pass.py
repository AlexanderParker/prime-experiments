"""F_j spectrum pass (max sum of j consecutive gaps), no probe loops.

The Constructor's word identity needs F_j(M) tiers; this runs the machine
stream once and reports F_1..F_6 (and the qualifying-window budget scale).
Full period unless --limit given (partial = exact on the scanned prefix,
a LOWER bound on each F_j).

Usage: uv run python research/spectrum_pass.py y [--limit SLOTS]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuel_census import fuel


def main():
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    y = int(args[0])
    r = fuel(y, [], limit=limit)
    Fj = [int(r["Fj"][j]) for j in range(1, 7)]
    cov = r["K"] / r["P"]
    print(f"machine y={y}: period {r['P']:.3e}, scanned {r['K']:.3e} "
          f"({100*cov:.1f}%), openings {r['openings']}, {r['secs']:.0f}s")
    print("F_j (j=1..6): " + " ".join(map(str, Fj)))
    print("increments  : " + " ".join(str(Fj[i + 1] - Fj[i])
                                      for i in range(5)))
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    path = os.path.join(ddir, "spectra.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if new:
            f.write("y,period,scanned,coverage,openings,"
                    + ",".join(f"F{j}" for j in range(1, 7)) + "\n")
        f.write(f"{y},{r['P']},{r['K']},{cov:.4f},{r['openings']},"
                + ",".join(map(str, Fj)) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

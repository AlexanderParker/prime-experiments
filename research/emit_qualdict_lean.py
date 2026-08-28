"""Round 25 (formalist): emit the machine-29 qualifying dictionaries as Lean.

Reads research/data/qualdict_29.csv (written by research/qual_dict.py, which
gate-checks its own F_j and Q_j output against the corpus) and writes the
explicit `List` literals plus the `decide +kernel` bound checks.

The dictionaries are split one Lean module per depth j so that no single
module carries more than ~7k list elements (formalist.md 1.3: the limit is
per-declaration / per-process state, and lake gives each module its own
process).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs"))

# sharp bounds are read from the scan output itself (max window sum per depth),
# so the emitter cannot drift from the census it is transcribing.
QMAX = {}


def load(y):
    d = {}
    for line in open(os.path.join(DATA, f"qualdict_{y}.csv")).read().strip(
            ).split("\n")[1:]:
        j, t = line.split(",")
        d.setdefault(int(j), []).append(tuple(int(x) for x in t.split()))
    return d


def tup(t):
    return "(" + ", ".join(str(x) for x in t) + ")"


def emit(j, rows, y=29):
    ty = " × ".join(["ℕ"] * j)
    proj = " + ".join(["t.1"] + [f"t.2{'.2' * k}.1" for k in range(j - 2)]
                      + [f"t.{'2.' * (j - 1)}".rstrip(".")]) if j > 2 else \
        "t.1 + t.2"
    if j > 2:
        parts = ["t.1"]
        for k in range(1, j - 1):
            parts.append("t." + "2." * k + "1")
        parts.append("t." + "2." * (j - 1))
        parts[-1] = parts[-1].rstrip(".")
        proj = " + ".join(parts)
    body = ",\n  ".join(", ".join(tup(t) for t in rows[i:i + 6])
                        for i in range(0, len(rows), 6))
    q = QMAX[j]
    per = {11: 385, 13: 5005, 17: 85085, 19: 1616615, 23: 37182145,
           29: 1078282205, 31: 33426748355}[y]
    fl = {19: 8, 23: 10, 29: 10, 31: 12}[y]
    nxt = {19: 23, 23: 29, 29: 31, 31: 37}[y]
    return f'''/-
Machine {y}, depth {j}: the QUALIFYING {j}-WINDOW DICTIONARY - every window of
{j} consecutive machine-{y} gaps whose {max(j - 2, 0)} interior gaps are all
>= {fl} (the floor `2u''` of gear {nxt}).  {len(rows):,} tuples, measured over the
FULL period {per:,} by `research/qual_dict.py` (which gate-checks its
own output against the corpus ladder at machines 19 and 23).

NOT KERNEL-CHECKED, AND NOT CLAIMED TO BE: that this list CONTAINS every
realised qualifying {j}-window is the census hypothesis `E{j}` of
`Machine{y}.Census{y}`.  What IS kernel-checked here is the only thing the rung
consumes from it - that every listed window sums to at most {q}.
-/

import Machine{y}

namespace Machine{y}

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- The realised qualifying {j}-windows of machine {y} (census input). -/
def D{j} : List ({ty}) :=
  [{body}]

/-- Every listed qualifying {j}-window sums to at most `Q_{j}({y}; {fl}) = {q}`. -/
theorem D{j}_ok : D{j}.all (fun t => Nat.ble ({proj}) {q}) = true := by
  decide +kernel

end Machine{y}
'''


def main():
    y = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    d = load(y)
    for j, rows in sorted(d.items()):
        QMAX[j] = max(sum(t) for t in rows)
    print(f"machine {y}: sharp bounds read from the census "
          f"{ {j: QMAX[j] for j in sorted(QMAX)} }")
    for j, rows in sorted(d.items()):
        path = os.path.join(PROOFS, f"Machine{y}D{j}.lean")
        with open(path, "w", encoding="utf-8") as f:
            f.write(emit(j, rows, y))
        print(f"wrote {path}: {len(rows):,} tuples, "
              f"{os.path.getsize(path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()

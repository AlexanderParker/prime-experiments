"""Round 20 (constructor): THE EXACT QUALMAX CRITERION for (D).

By the merge law, every new gap of M+q' is a window sum of j consecutive old
gaps whose j-2 interiors are residue-qualifying (g mod q' in {0, +-2c} -
necessary for a chain, firing not required).  Hence, EXACTLY:

    F(M+q')  <=  max( F_2(M),  max_{j>=3} qualmax_j(M; q') )      (*)

with every quantity a full-period census value, and

    (D) at alpha = 3   <==   max(F_2, max_j qualmax_j)  <=  F + q'.

This check: (*) against the known F ladder at all seven steps 11->13 ..
31->37, and the (D) margins.  R31's suppression-corrected flatness is (*)
with qualmax_j replaced by the heuristic F_j - lambda(j-2)L; this version
has no fitted constant.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")

FLADDER = {13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88}

rows = {}
with open(os.path.join(DDIR, "tm_resid_runs.csv")) as f:
    for r in csv.DictReader(f):
        rows[int(r["y"])] = r

print("  step        F   F2  qm3  qm4  qm5  qm6  | crit  F(M+q')  slack | "
      "budget F+q'  margin  margin/q'")
exact_hits = 0
for y in sorted(rows):
    r = rows[y]
    q1 = int(r["qp"])
    if q1 not in FLADDER:
        continue
    F, F2 = int(r["F1"]), int(r["F2"])
    qms = [int(r[f"qm{j}"]) for j in range(3, 7)]
    crit = max([F2] + qms)
    Fnew = FLADDER[q1]
    assert Fnew <= crit, f"criterion violated at {y}->{q1}"
    budget = F + q1
    ok = crit <= budget
    assert ok, f"(D) fails at {y}->{q1}"
    if Fnew == crit:
        exact_hits += 1
    print(f"  {y:>3}->{q1:<3}  {F:>4} {F2:>4} " +
          " ".join(f"{q:>4}" for q in qms) +
          f"  | {crit:>4}  {Fnew:>6}  {crit - Fnew:>5} | {budget:>8}"
          f"  {budget - crit:>6}  {(budget - crit) / q1:>8.2f}")
print(f"\n(*) holds at all steps; EQUALITY crit = F(M+q') at {exact_hits} of "
      f"{sum(1 for y in rows if int(rows[y]['qp']) in FLADDER)} steps; "
      "(D) criterion max(F2, qm_j) <= F + q' holds at every measured step.")

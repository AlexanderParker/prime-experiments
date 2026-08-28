"""Round 25 (formalist): THE GATE on the machine-29 census input.

`Machine31.D_29_31`'s single hypothesis is `Machine29.Census29` - that the
15,860 tuples transcribed into `proofs/Machine29D2..D7.lean` contain every
realised qualifying window of machine 29, and that no six consecutive gaps
reach 10.  Nothing in the kernel can check that, so it gets the strongest
check outside one:

 1. CHUNK INDEPENDENCE.  The scan is chunked, and the failure mode that has
    bitten this project before is a window straddling a chunk junction seen
    by neither pass (mechanic's standing rule 18).  So the whole period is
    scanned TWICE with coprime, unrelated chunk sizes and the dictionaries
    must come out byte-identical.  A seam bug cannot survive that: the two
    runs put their junctions in completely different places.
 2. CYCLIC SEAM.  The gap word is cyclic; `qual_dict.gaps_of_period` closes
    the ring explicitly and the gap count is asserted equal to
    `prod (q - 2) = 214,708,725`.
 3. TRANSCRIPTION.  The Lean literals are parsed back out of
    `proofs/Machine29D*.lean` and compared, as sets, with the scan.
 4. CORPUS AGREEMENT.  Re-runs machines 19 and 23, where every number the
    scanner produces is already kernel-proved in this ledger.

Usage: python research/qual_dict_gate.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_dict as Q                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs"))


def dicts_of(r):
    out = {2: {Q.decode(c, 2) for c in r["pairs"]}}
    for j in range(3, Q.JMAX + 1):
        if r["dicts"][j]:
            out[j] = {Q.decode(c, j) for c in r["dicts"][j]}
    return out


def lean_dicts():
    out = {}
    for j in range(2, 8):
        p = os.path.join(PROOFS, f"Machine29D{j}.lean")
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8").read()
        body = src.split("[", 1)[1].split("]", 1)[0]
        out[j] = {tuple(int(x) for x in m.split(","))
                  for m in re.findall(r"\(([^()]*)\)", body)}
    return out


def main():
    ok = True

    print("(4) corpus agreement at the kernel-proved machines")
    for y in (19, 23):
        r = Q.scan(y)
        Q.report(r)
        for j, v in Q.CORPUS_Q.get(y, {}).items():
            got = max(sum(Q.decode(c, j)) for c in
                      (r["pairs"] if j == 2 else r["dicts"][j]))
            assert got == v, (y, j, got, v)
        assert r["Fj"][1] == Q.FCORPUS[y]
        print(f"  machine {y}: GREEN\n")

    print("(1)+(2) chunk independence and the cyclic seam at machine 29")
    a = Q.scan(29, chunk=40_000_000)
    b = Q.scan(29, chunk=23_456_789)
    da, db = dicts_of(a), dicts_of(b)
    assert a["ngaps"] == b["ngaps"] == 214_708_725, (a["ngaps"], b["ngaps"])
    for j in sorted(set(da) | set(db)):
        same = da.get(j, set()) == db.get(j, set())
        ok &= same
        print(f"  D_{j}: {len(da.get(j, ())):,} vs {len(db.get(j, ())):,}"
              f"  identical: {same}")
    assert a["Fj"] == b["Fj"], (a["Fj"], b["Fj"])
    assert a["maxrun"] == b["maxrun"] == 5, (a["maxrun"], b["maxrun"])
    print(f"  F_j identical: {a['Fj'][1:8]}; longest qualifying run "
          f"{a['maxrun']} (so depth >= 8 is vacuous)")

    print("\n(3) transcription: proofs/Machine29D*.lean vs the scan")
    L = lean_dicts()
    for j in sorted(da):
        same = L.get(j) == da[j]
        ok &= same
        print(f"  D_{j}: Lean {len(L.get(j, ())):,} tuples, "
              f"scan {len(da[j]):,}, identical: {same}")
    QMAX = {2: 55, 3: 65, 4: 68, 5: 71, 6: 71, 7: 71}
    for j, v in QMAX.items():
        got = max(sum(t) for t in L[j])
        assert got == v, (j, got, v)
        assert all(sum(t) <= v for t in L[j])
    print(f"  Lean-side maxima {QMAX} match the scan; "
          f"max_j Q_j = 71 <= 74 = F(29) + 31")

    print("\n=> " + ("ALL FOUR GATES GREEN" if ok else "GATE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

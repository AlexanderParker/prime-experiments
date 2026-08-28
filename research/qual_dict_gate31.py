"""Round 25 (formalist): the chunk-independence gate at MACHINE 31.

Same gate as `research/qual_dict_gate.py` part (1), run separately because a
machine-31 pass is ~24 minutes: rescan the whole 33,426,748,355-slot period
with an unrelated chunk size and require the dictionaries, the `F_j` ladder
and the qualifying run length to come out identical to the saved run.  A
window straddling a chunk junction (mechanic's standing rule 18) cannot
survive it, because the two runs put their junctions in different places.

Also re-checks the transcription against proofs/Machine31D*.lean if present.

Usage: python research/qual_dict_gate31.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qual_dict as Q                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROOFS = os.path.abspath(os.path.join(HERE, "..", "proofs"))

# the saved run (research/data/qualdict_31.csv, chunk 60,000,000)
EXPECT_SIZES = {2: 1253, 3: 8155, 4: 18566, 5: 13049, 6: 2120, 7: 42}
EXPECT_QMAX = {2: 68, 3: 85, 4: 90, 5: 91, 6: 90, 7: 88}
BUDGET = 58 + 37


def saved():
    d = {}
    for line in open(os.path.join(DATA, "qualdict_31.csv"),
                     encoding="utf-8").read().strip().split("\n")[1:]:
        j, t = line.split(",")
        d.setdefault(int(j), set()).add(tuple(int(x) for x in t.split()))
    return d


def lean_dicts():
    out = {}
    for j in range(2, 9):
        p = os.path.join(PROOFS, f"Machine31D{j}.lean")
        if not os.path.exists(p):
            continue
        body = open(p, encoding="utf-8").read().split("[", 1)[1].split("]", 1)[0]
        out[j] = {tuple(int(x) for x in m.split(","))
                  for m in re.findall(r"\(([^()]*)\)", body)}
    return out


def main():
    ok = True
    old = saved()
    for j, n in EXPECT_SIZES.items():
        assert len(old[j]) == n, (j, len(old[j]), n)
    print(f"saved run: {sum(EXPECT_SIZES.values()):,} tuples, "
          f"Q_j {EXPECT_QMAX}, budget {BUDGET}")

    print("\nrescanning machine 31 with an unrelated chunk size "
          "(37,000,001 vs 60,000,000)...")
    r = Q.scan(31, chunk=37_000_001)
    assert r["ngaps"] == 6_226_553_025, r["ngaps"]
    assert r["Fj"][1] == 58, r["Fj"][1]
    assert r["maxrun"] == 5, r["maxrun"]
    new = {2: {Q.decode(c, 2) for c in r["pairs"]}}
    for j in range(3, Q.JMAX + 1):
        if r["dicts"][j]:
            new[j] = {Q.decode(c, j) for c in r["dicts"][j]}

    for j in sorted(set(old) | set(new)):
        same = old.get(j, set()) == new.get(j, set())
        ok &= same
        print(f"  D_{j}: saved {len(old.get(j, ())):,} vs rescan "
              f"{len(new.get(j, ())):,}  identical: {same}")
    print(f"  gap count {r['ngaps']:,} = prod(q-2); F(31) = {r['Fj'][1]}; "
          f"longest qualifying run {r['maxrun']}")
    mq = max(max(sum(t) for t in new[j]) for j in new)
    print(f"  max_j Q_j = {mq} vs budget {BUDGET}: "
          f"{'CERTIFIES' if mq <= BUDGET else 'FAILS'} (margin {BUDGET - mq})")
    assert mq == 91, mq

    L = lean_dicts()
    if L:
        print("\ntranscription: proofs/Machine31D*.lean vs the rescan")
        for j in sorted(new):
            same = L.get(j) == new[j]
            ok &= same
            print(f"  D_{j}: Lean {len(L.get(j, ())):,}, scan "
                  f"{len(new[j]):,}, identical: {same}")
    else:
        print("\n(no proofs/Machine31D*.lean yet - transcription check skipped)")

    print("\n=> " + ("MACHINE-31 GATE GREEN" if ok else "GATE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Round 25 (mechanic): verify a dictionary-transfer SUPERSET against the
EXACT dictionary of the same machine, and report the inflation.

The transfer construct (research/dict_transfer.py, K4) is proved to emit a
SUPERSET of the target machine's realised m-tuple dictionary, and was validated
exhaustively at 23->29 and 29->31 (0 missing both times).  The 31->37 output
was produced in round 24 while the exact machine-37 scan was still in flight,
so it stood DRAFT-UNVERIFIED.  This closes that.

ANY missing tuple contradicts the construct's proof and its two exhaustive
validations and must be treated as a TOOL BUG, not as a datum - so the check
asserts, it does not report.

usage: python research/dict_containment_r25.py EXACT.csv TRANSFER.csv
"""
import sys


def load(path):
    with open(path) as fh:
        head = fh.readline()
        assert head.startswith("g1,"), (path, head)
        return {tuple(int(x) for x in ln.split(",")) for ln in fh if ln.strip()}


def main():
    exact_p, tr_p = sys.argv[1], sys.argv[2]
    exact, tr = load(exact_p), load(tr_p)
    print(f"  exact    {exact_p}: {len(exact):,} tuples")
    print(f"  transfer {tr_p}: {len(tr):,} tuples")
    missing = exact - tr
    print(f"  missing from the transfer (must be 0): {len(missing):,}")
    if missing:
        print(f"    e.g. {sorted(missing)[:10]}")
    print(f"  inflation = |transfer| / |exact| = "
          f"{len(tr) / len(exact):.2f}x")
    vx = sorted({v for t in exact for v in t})
    vt = sorted({v for t in tr for v in t})
    print(f"  distinct gap values: exact {len(vx)}, transfer {len(vt)} "
          f"(extra {sorted(set(vt) - set(vx))})")
    assert not missing, (
        f"{len(missing)} realised tuples missing from the certified superset "
        f"- this contradicts the transfer construct's proof; TOOL BUG")
    print("  CONTAINMENT VERIFIED (asserted): the transfer is a superset.")


if __name__ == "__main__":
    main()

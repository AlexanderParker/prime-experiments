"""rr_top37.py -- the top band of the spectrum of M = {5..37}, scan-free.

M = {5..37} has a period of 1.24e12 columns and has never been sieved.  The configuration
enumerator of rr_record.py decides, exactly and from the residues alone, how many gaps of each
size the period holds and what their neighbours are.  This sweeps the top band v >= 0.8 F = 71
(F = 88) plus a few cells above the record, and reports m(v), n1(v), N(v).

Usage: uv run python rr_top37.py <vlo> <vhi> [nodecap]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rr_record import report, OUT                        # noqa: E402


def main():
    vlo = int(sys.argv[1])
    vhi = int(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000_000
    rows = {}
    for v in range(vhi, vlo - 1, -1):
        try:
            d = report(37, v, cap)
        except RuntimeError:
            print(f"v={v}  NODE CAP {cap} exceeded -- undecided", flush=True)
            rows[v] = {"undecided": True}
            continue
        rows[v] = {"m": d["m_v"], "n1": d["n1"], "N": d["N"],
                   "nodes": d["nodes"], "s": round(d["seconds"], 1),
                   "all_busy": d["all_gears_busy"]}
        print(f"v={v}  m(v)={d['m_v']}  n1={d['n1']}  N={d['N']}  "
              f"Sig={None if d['n1'] is None else v + d['n1']}  "
              f"nodes={d['nodes']}  {d['seconds']:.1f}s  allbusy={d['all_gears_busy']}",
              flush=True)
        with open(os.path.join(OUT, "top37.json"), "w") as f:
            json.dump(rows, f, default=int)


if __name__ == "__main__":
    main()

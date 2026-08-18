"""Padding census (mechanic r14): zero-letter links in killed runs.

A killed run's links carry letters: +1 (spacing = s = 2u mod qp), -1
(spacing = qp - s), 0 (spacing = 0 mod qp - the two kills sit at the SAME
tooth, one full lap apart, which requires a gap of exactly qp (or 2qp,
...) in the old machine M). Legality = non-zero letters alternate, zeros
free; equivalently the running residue stays inside a 2-value window,
which is the condition already used by fuel_census.py (prefix-sum range
<= 1). So padded links were always inside N_k - this tool breaks them out.

Per (machine y, probe qp) it reports, over the full period unless limited:
  * padding SUPPLY: number of gaps of M equal to exactly qp (and 2qp),
    with F and the total gap count;
  * maximal legal runs classified by (k, z) = (openings killed, zero links
    used), with max flanked span per class - so one can read directly
    whether padding carries the step's record;
  * the DOUBLE-PADDED hunt: any legal run with z >= 2, with address and
    literal gap word (the event hunted this round).

Usage: uv run python research/padding_census.py y [q1 q2 ...]
       [--limit SLOTS] [--start SLOT]
Output: printed report + append to research/data/padding_census.csv
"""
import os
import sys
import time
import numpy as np
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragile_census import primes_upto
from prefix_census import is_prime

MAXK = 8


def _emit(S, ops, d, la, lb, q):
    """record a maximal legal link-run [la..lb] (kills ops[la..lb+1])."""
    k = lb - la + 2
    if k < 2:
        return
    z = int((d[la:lb + 1] % q == 0).sum())
    span = int(ops[lb + 1] - ops[la])
    if la >= 1 and lb + 2 < len(ops):
        fl = int(ops[lb + 2] - ops[la - 1])
    else:
        fl = span
    key = (min(k, MAXK), z)
    S["cls"][key] = S["cls"].get(key, 0) + 1
    if fl > S["best"].get(key, 0):
        S["best"][key] = fl
    if z >= 2 and len(S["dbl"]) < 25:
        S["dbl"].append((int(ops[la]), k, z,
                         tuple(int(x) for x in d[la:lb + 1])))


def padding(y, probes, limit=None, start=0, seg=64_000_000):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    K = P if limit is None else min(P, start + limit)
    uvals = [pow(6, -1, g) for g in gears]
    st = {}
    for q in probes:
        st[q] = dict(s=(2 * pow(6, -1, q)) % q, pad1=0, pad2=0, ngap=0,
                     F=0, cls={}, dbl=[], best={})
    tail = np.array([], dtype=np.int64)
    t0 = time.time()
    for a in range(start, K, seg):
        b = min(K, a + seg)
        ex = np.zeros(b - a, bool)
        for g, u in zip(gears, uvals):
            ex[(u - a) % g::g] = True
            ex[(-u - a) % g::g] = True
        op = np.flatnonzero(~ex).astype(np.int64) + a
        ops = np.concatenate([tail, op])
        if len(ops) < 3:
            tail = ops
            continue
        d = np.diff(ops)
        newlink = ops[1:] >= a
        for q, S in st.items():
            s = S["s"]
            S["ngap"] += int(newlink.sum())
            S["F"] = max(S["F"], int(d.max()))
            S["pad1"] += int(((d == q) & newlink).sum())
            S["pad2"] += int(((d == 2 * q) & newlink).sum())
            dm = d % q
            letter = np.full(len(d), 9, dtype=np.int8)
            letter[dm == 0] = 0
            letter[dm == s] = 1
            letter[dm == (q - s) % q] = -1
            legal = letter != 9
            idx = np.flatnonzero(legal)
            if len(idx) == 0:
                continue
            brk = np.flatnonzero(np.diff(idx) != 1)
            starts = np.concatenate([[0], brk + 1])
            ends = np.concatenate([brk, [len(idx) - 1]])
            for si, ei in zip(starts, ends):
                l0, l1 = int(idx[si]), int(idx[ei])
                if ops[l1 + 1] < a:
                    continue
                run_start = l0
                last_nz = 0
                for t in range(l0, l1 + 1):
                    lt = int(letter[t])
                    if lt != 0 and lt == last_nz:
                        _emit(S, ops, d, run_start, t - 1, q)
                        run_start = t
                        last_nz = lt
                    elif lt != 0:
                        last_nz = lt
                _emit(S, ops, d, run_start, l1, q)
        tail = ops[-(MAXK + 2):]
    return dict(y=y, P=P, K=K, start=start, probes=st,
                secs=time.time() - t0)


def main():
    args = sys.argv[1:]
    limit = None
    start = 0
    if "--limit" in args:
        i = args.index("--limit")
        limit = int(args[i + 1])
        del args[i:i + 2]
    if "--start" in args:
        i = args.index("--start")
        start = int(args[i + 1])
        del args[i:i + 2]
    y = int(args[0])
    probes = [int(a) for a in args[1:]]
    if not probes:
        p = y + 2
        while not is_prime(p):
            p += 2
        probes = [p]
    r = padding(y, probes, limit=limit, start=start)
    ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, "padding_census.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a")
    if new:
        f.write("y,q,coverage,F,total_gaps,gaps_eq_q,gaps_eq_2q,pad_share,"
                "runs_z0,runs_z1,runs_z2plus,maxspan_z0,maxspan_z1,"
                "maxspan_z2plus\n")
    cov = (r["K"] - r["start"]) / r["P"]
    print(f"machine y={y}: period {r['P']:.4e}, scanned "
          f"{(r['K'] - r['start']):.3e} ({100 * cov:.1f}%), {r['secs']:.0f}s")
    for q, S in r["probes"].items():
        z0 = sum(v for (k, z), v in S["cls"].items() if z == 0)
        z1 = sum(v for (k, z), v in S["cls"].items() if z == 1)
        z2 = sum(v for (k, z), v in S["cls"].items() if z >= 2)
        m0 = max([v for (k, z), v in S["best"].items() if z == 0] or [0])
        m1 = max([v for (k, z), v in S["best"].items() if z == 1] or [0])
        m2 = max([v for (k, z), v in S["best"].items() if z >= 2] or [0])
        share = S["pad1"] / max(S["ngap"], 1)
        print(f"  qp={q} s={S['s']}: F={S['F']} gaps={S['ngap']}  "
              f"SUPPLY gaps=={q}: {S['pad1']} ({share:.3e}); "
              f"gaps=={2 * q}: {S['pad2']}")
        print(f"    runs by padding: z=0 {z0}, z=1 {z1}, z>=2 {z2}  |  "
              f"max flanked span: z=0 {m0}, z=1 {m1}, z>=2 {m2}")
        for (k, z), v in sorted(S["cls"].items()):
            if z >= 1:
                print(f"      k={k} z={z}: {v} runs, max flanked span "
                      f"{S['best'][(k, z)]}")
        for addr, k, z, word in S["dbl"][:10]:
            print(f"      DOUBLE-PADDED: addr {addr} k={k} z={z} "
                  f"word {word}")
        f.write(f"{y},{q},{cov:.4f},{S['F']},{S['ngap']},{S['pad1']},"
                f"{S['pad2']},{share:.6e},{z0},{z1},{z2},{m0},{m1},{m2}\n")
    f.close()


if __name__ == "__main__":
    main()

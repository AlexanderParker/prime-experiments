"""Round 17 (mechanic): analysis of the flank-envelope census.

Reads the CSVs written by flank_envelope.py and answers, exactly:

 (A) the MONOTONE ENVELOPE - does max single flank fall with span?  Checked
     three ways: within a step's compatible word list, pooled per machine over
     all probed q', and unconditionally (all letters, any span) from the
     uncond CSV.  Every violation is exhibited.
 (B) the RARITY NULL - is the fall a sample-size effect?  For each word,
     compare the observed max flank / max flank sum with the exact prediction
     from drawing 2*occ flanks independently from the machine's own gap
     histogram, with the one-sided p-value P(max < obs).
 (C) the MARGIN TRAJECTORY - min over compatible words of
     (F + q' - span - FS_max), absolute and in units of q'.
 (D) the SPECTRUM CRITERION - F_{L+2}(M) <= F(M) + q', L = litcap(q') - 1.

Usage: uv run python research/envelope_analysis.py
"""
import os
import csv
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")


def load(name):
    with open(os.path.join(DDIR, name)) as f:
        return list(csv.DictReader(f))


def main():
    rows = load("flank_envelope_words.csv")
    for r in rows:
        for k in ("y", "qp", "ell", "span", "compat", "occ", "F", "Fell2",
                  "FSmax", "gL", "gR", "maxflank", "need3", "margin3",
                  "specceil", "specslack"):
            r[k] = int(r[k])
        r["addr_fs"] = int(r["addr_fs"])
        r["addr_mf"] = int(r["addr_mf"])

    print("=" * 78)
    print("(A1) MONOTONE ENVELOPE WITHIN EACH STEP'S COMPATIBLE WORD LIST")
    print("=" * 78)
    steps = defaultdict(list)
    for r in rows:
        if r["compat"]:
            steps[(r["y"], r["qp"])].append(r)
    nv = 0
    for key in sorted(steps):
        ws = sorted(steps[key], key=lambda r: r["span"])
        seq = [(w["span"], w["maxflank"]) for w in ws]
        bad = [(seq[i], seq[j]) for i in range(len(seq))
               for j in range(i + 1, len(seq))
               if seq[j][0] > seq[i][0] and seq[j][1] > seq[i][1]]
        nv += len(bad)
        print(f"  {key[0]:3d} -> {key[1]:3d}  F={ws[0]['F']:3d}  "
              f"(span, maxflank): " +
              " ".join(f"({s},{m})" for s, m in seq) +
              ("   VIOLATION " + str(bad) if bad else "   monotone"))
    print(f"  TOTAL within-step violations: {nv}")

    print("=" * 78)
    print("(A2) POOLED PER MACHINE (all probed q'): max flank vs span")
    print("=" * 78)
    mach = defaultdict(dict)
    for r in rows:
        if not r["compat"]:
            continue
        s = r["span"]
        cur = mach[r["y"]].get(s)
        if cur is None or r["maxflank"] > cur[0]:
            mach[r["y"]][s] = (r["maxflank"], r["qp"], r["word"], r["occ"],
                               r["addr_mf"])
    tot = 0
    for y in sorted(mach):
        seq = sorted(mach[y].items())
        bad = [(seq[i], seq[j]) for i in range(len(seq))
               for j in range(i + 1, len(seq)) if seq[j][1][0] > seq[i][1][0]]
        tot += len(bad)
        print(f"  machine {y}: " +
              " ".join(f"{s}:{v[0]}" for s, v in seq))
        for (s1, v1), (s2, v2) in bad:
            print(f"     VIOLATION  span {s1} -> maxflank {v1[0]} "
                  f"(w={v1[2]}, q'={v1[1]}, occ={v1[3]})   BUT   "
                  f"span {s2} -> maxflank {v2[0]} (w={v2[2]}, q'={v2[1]}, "
                  f"occ={v2[3]}, addr k={v2[4]:,})")
    print(f"  TOTAL pooled-per-machine violations: {tot}")

    print("=" * 78)
    print("(A3) UNCONDITIONAL ENVELOPE (any letters) - violation census")
    print("=" * 78)
    unc = load("flank_envelope_uncond.csv")
    U = defaultdict(dict)
    for r in unc:
        U[(int(r["y"]), int(r["ell"]))][int(r["span"])] = (
            int(r["maxflank"]), int(r["maxFS"]), int(r["count"]),
            int(r["addr_maxFS"]))
    for (y, ell) in sorted(U):
        seq = sorted(U[(y, ell)].items())
        bad = [(seq[i][0], seq[i][1][0], seq[j][0], seq[j][1][0])
               for i in range(len(seq)) for j in range(i + 1, len(seq))
               if seq[j][1][0] > seq[i][1][0]]
        # the maximal violation: biggest rise
        worst = max(bad, key=lambda t: (t[3] - t[1], t[2] - t[0]),
                    default=None)
        print(f"  machine {y} ell={ell}: {len(seq)} spans, "
              f"{len(bad)} violating pairs" +
              (f"; worst  E({worst[0]})={worst[1]} -> E({worst[2]})="
               f"{worst[3]}" if worst else ""))

    print("=" * 78)
    print("(B) RARITY NULL: observed extremes vs independent draws from the "
          "machine's own gap histogram")
    print("=" * 78)
    gh = defaultdict(lambda: np.zeros(256, np.int64))
    for r in load("flank_envelope_gaphist.csv"):
        gh[int(r["y"])][int(r["gap"])] += int(r["count"])
    print("   step        word      occ        F  span  obs_FS  null_FS  "
          "ceil  eff_null  obs-eff  p(<obs)   obs_mf null_mf")
    for r in sorted(rows, key=lambda r: (r["y"], r["qp"], r["span"])):
        if not r["compat"] or r["occ"] == 0:
            continue
        h = gh[r["y"]].astype(float)
        tot = h.sum()
        p = h / tot
        tail = np.cumsum(p[::-1])[::-1]
        n = 2 * r["occ"]
        # null median: largest g with 1-(1-tail[g])^n >= 0.5
        nullmf = max((g for g in range(1, 256)
                      if 1 - (1 - tail[g]) ** n >= 0.5), default=0)
        pmf = (1 - tail[min(255, r["maxflank"] + 1)]) ** n
        conv = np.convolve(p, p)
        ct = np.cumsum(conv[::-1])[::-1]
        nullfs = max((v for v in range(1, len(ct))
                      if 1 - (1 - ct[v]) ** r["occ"] >= 0.5), default=0)
        pfs = (1 - ct[min(len(ct) - 1, r["FSmax"] + 1)]) ** r["occ"]
        eff = min(nullfs, r["specceil"])
        print(f"  {r['y']:3d}->{r['qp']:3d} {r['word']:>10s} "
              f"{r['occ']:>10,} {r['F']:4d} {r['span']:5d}  {r['FSmax']:5d}"
              f"   {nullfs:6d} {r['specceil']:5d}  {eff:7d}  {r['FSmax']-eff:+7d}"
              f"  {pfs:7.4f}   {r['maxflank']:5d} {nullmf:6d}")

    print("=" * 78)
    print("(C) MARGIN TRAJECTORY (consecutive steps only)")
    print("=" * 78)
    cons = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41,
            41: 43}
    print("   step      F   q'   words  min margin3   /q'   binding word "
          "(span, FS_max)")
    for y in sorted(cons):
        key = (y, cons[y])
        if key not in steps:
            continue
        ws = [w for w in steps[key] if w["occ"] > 0]
        if not ws:
            continue
        b = min(ws, key=lambda w: w["margin3"])
        print(f"  {y:3d}->{cons[y]:3d} {b['F']:4d} {cons[y]:4d}  "
              f"{len(ws):4d}   {b['margin3']:+8d}   {b['margin3']/cons[y]:.3f}"
              f"   {b['word']:>12s} (span {b['span']}, FS_max {b['FSmax']})")

    print("=" * 78)
    print("(D) SPECTRUM CRITERION  F_{L+2}(M) <= F(M) + q',  L = litcap-1")
    print("=" * 78)
    import sys
    sys.path.insert(0, HERE)
    from flank_envelope import literal_cap
    spec = {}
    for r in load("flank_envelope_spectra.csv"):
        y = int(r["y"])
        if y not in spec or float(r["coverage"]) >= spec[y][0]:
            spec[y] = (float(r["coverage"]),
                       [int(r[f"F{j}"]) for j in range(1, 9)])
    print("   step     litcap  L  F_{L+2}   F+q'   verdict     (coverage)")
    for y in sorted(cons):
        q = cons[y]
        if y not in spec:
            continue
        cov, Fj = spec[y]
        L = literal_cap(q) - 1
        idx = L + 2
        v = Fj[idx - 1] if idx <= 8 else -1
        F = Fj[0]
        ok = "IMPLIES (D)" if 0 <= v <= F + q else "NOT implied"
        print(f"  {y:3d}->{q:3d}   {literal_cap(q):4d}  {L:2d}  "
              f"{v:6d}  {F+q:6d}   {ok:12s} ({cov:.4f})")


if __name__ == "__main__":
    main()

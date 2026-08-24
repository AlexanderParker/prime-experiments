"""Harvester round 22: THE FULL 19-WINNER SET and the extension-deficit ladder.

Round 21 left one micro-question open: the best clean extension of a family maximiser
falls short of the next machine's true maximum by 9 (13->17), 18 (17->19) and 36
(19->23) - a doubling on three points, but the 36 was LINEAGE-ONLY (computed from the
single known 19-argmax, because the full 19-winner set was believed out of reach:
2,424,922 differences, round 17).

This script closes that.  The delta reduction (research/delta_frame.py) plus the
held-out-top-gear prefilter (research/family_scan.py) make the exhaustive y=19 family
scan cheap, so:

  (1) all delta with G(delta) >= 43 at gears 5..19  ->  the complete 19-winner set
      (and an independent replication of Ziller-Morack's h_2(19) = 258);
  (2) the 3 | e branch settled exhaustively (a 3|e difference needs TWO killed runs
      of length >= 42 at a prescribed offset, so its delta must already be a winner);
  (3) the deficit ladder recomputed over COMPLETE winner sets at every rung:
      13->17, 17->19 (cross-checks of round 21) and 19->23 (the new number).

True family maxima (Ziller-Morack arXiv:1706.03668 Table 1, h_2 = 2F):
  y :   13    17    19    23        h_2 : 150  192  258  366
  F :   75    96   129   183        G = F/3 :  25   32   43   61
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import prod
from family_scan import scan, max_gap, survivors

G5_13 = [5, 7, 11, 13]
G5_17 = G5_13 + [17]
G5_19 = G5_17 + [19]
G5_23 = G5_19 + [23]
TRUE_G = {13: 25, 17: 32, 19: 43, 23: 61}


def extensions(qs_old, deltas_old, qnew):
    """For every delta_old in the list and every lift residue r mod qnew, the max gap
    of the extended machine.  Returns (best_G, list of (delta_old, r, G))."""
    Qo = prod(qs_old)
    Qn = Qo * qnew
    inv = pow(Qo, -1, qnew)
    out = []
    best = -1
    for d in deltas_old:
        idx = np.flatnonzero(survivors(qs_old, d, Qo)).astype(np.int64)
        full = (idx[None, :] + Qo * np.arange(qnew, dtype=np.int64)[:, None]).ravel()
        rem = full % qnew
        for r in range(qnew):
            keep = (rem != 0) & (rem != (-r) % qnew)
            v = full[keep]
            if v.size < 2:
                g = Qn
            else:
                g = int(np.diff(np.append(v, v[0] + Qn)).max())
            dn = (d + Qo * (((r - d) * inv) % qnew)) % Qn
            out.append((int(dn), int(d), r, g))
            best = max(best, g)
    return best, out


def killed_runs(qs, delta, Q, minlen):
    """start positions (in Z_Q) of maximal killed runs of length >= minlen."""
    idx = np.flatnonzero(survivors(qs, delta, Q)).astype(np.int64)
    d = np.diff(np.append(idx, idx[0] + Q))
    hits = np.flatnonzero(d - 1 >= minlen)
    return [(int(idx[i] + 1), int(d[i] - 1)) for i in hits]


def main():
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)

    # ---- 1. the complete winner sets -------------------------------------------
    say("=== EXHAUSTIVE FAMILY WINNER SETS (delta space, gears 5..y) ===")
    w13 = [d for d, g in scan([5, 7, 11], 13, TRUE_G[13])]
    say(f"  y=13: {len(w13)} winners at G = {TRUE_G[13]} (F = {3*TRUE_G[13]})")
    assert len(w13) == 16
    w17 = [d for d, g in scan(G5_13, 17, TRUE_G[17])]
    say(f"  y=17: {len(w17)} winners at G = {TRUE_G[17]} (F = {3*TRUE_G[17]})")
    assert len(w17) == 64
    say("  y=19: scanning 85085 prefilter classes ...")
    s19 = scan(G5_17, 19, TRUE_G[19], verbose=True, prog=20000)
    gmax19 = max(g for _, g in s19) if s19 else -1
    w19 = [d for d, g in s19 if g == gmax19]
    say(f"  y=19: max G = {gmax19} (F = {3*gmax19}, h_2 = {6*gmax19}) over the WHOLE "
        f"family; {len(w19)} winners, {len(s19)} deltas with G >= {TRUE_G[19]}")
    assert gmax19 == TRUE_G[19], gmax19
    np.save("research/data/family_w19_delta.npy", np.array(sorted(w19), np.int64))

    # ---- 2. the 3 | e branch ----------------------------------------------------
    # For 3 | e the survivors occupy TWO classes mod 3, so a gap of 3*G needs killed
    # runs of length >= G-1 in BOTH sub-lattices; both are translates of S_delta, so
    # delta must already be in the G >= 43 list.  Check those e directly.
    P19 = 3 * prod(G5_19)
    Q19 = prod(G5_19)
    best3, arg3 = -1, None
    for d, _ in s19:
        e = (3 * d) % Q19
        e = e + Q19 * ((-e * pow(Q19, -1, 3)) % 3)      # e = 0 mod 3, = 3d mod Q19
        assert e % 3 == 0 and (e - 3 * d) % Q19 == 0
        a = np.ones(P19, bool)
        for q in [3] + G5_19:
            a[0::q] = False
            a[(-e) % q::q] = False
        i2 = np.flatnonzero(a)
        f = int(np.diff(np.append(i2, i2[0] + P19)).max())
        if f > best3:
            best3, arg3 = f, int(e)
    say(f"  3 | e branch (exhaustive over the {len(s19)} candidate deltas): "
        f"best F = {best3} at e = {arg3}  vs  3 not dividing e: F = {3*gmax19}")
    assert best3 < 3 * gmax19

    # ---- 3. the deficit ladder over COMPLETE winner sets -------------------------
    say("")
    say("=== EXTENSION-DEFICIT LADDER (complete winner sets at every rung) ===")
    say("  step      #winners  best extension F   true max F   deficit")
    rows = []
    for (qs_old, wins, qnew, yold) in (([5, 7, 11, 13], w13, 17, 13),
                                       (G5_17, w17, 19, 17),
                                       (G5_19, w19, 23, 19)):
        best, out = extensions(qs_old, wins, qnew)
        true = TRUE_G[qnew]
        rows.append((yold, qnew, len(wins), 3 * best, 3 * true, 3 * (true - best)))
        say(f"  {yold:>2} -> {qnew:<3}  {len(wins):>8}   {3*best:>15}   "
            f"{3*true:>10}   {3*(true-best):>7}")
        if qnew == 23:
            np.save("research/data/ext19_to23.npy",
                    np.array([(a, b, c, g) for a, b, c, g in out], np.int64))
            top = sorted(out, key=lambda t: -t[3])[:6]
            say(f"      best 23-extensions (delta_new, delta_old, r, G): {top}")

    say("")
    say("  round-21 ladder was 9, 18, 36 (the 36 lineage-only).  Recomputed: "
        + ", ".join(str(r[5]) for r in rows))

    # ---- 4. anatomy of the best extension at 19->23 ------------------------------
    say("")
    say("=== ANATOMY: the record window and its best extension (19 -> 23) ===")
    bestrow = max(np.load("research/data/ext19_to23.npy").tolist(), key=lambda t: t[3])
    dn, do, r, g = bestrow
    Q23 = prod(G5_23)
    runs_old = killed_runs(G5_19, do, Q19, TRUE_G[19] - 1)
    runs_new = killed_runs(G5_23, dn, Q23, g - 1)
    say(f"  winner delta_19 = {do}: killed runs of length >= {TRUE_G[19]-1}: "
        f"{len(runs_old)}  (G = {TRUE_G[19]})")
    say(f"  best lift delta_23 = {dn} (r = {r}): G = {g} (F = {3*g}); "
        f"{len(runs_new)} maximal runs")
    # the old-machine gap word around the extended window
    st = runs_new[0][0]
    idx_old = np.flatnonzero(survivors(G5_19, do, Q19)).astype(np.int64)
    lo = st % Q19
    sel = np.sort(np.concatenate([idx_old, idx_old + Q19]))
    a = np.searchsorted(sel, lo) - 1
    b = np.searchsorted(sel, lo + g)
    word = np.diff(sel[a:b + 1]).tolist()
    say(f"  old-machine gaps fused by the lift: {word}  (sum {sum(word)}, "
        f"new gap {g})")

    with open("research/data/ext_deficit19.out", "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("ext_deficit19: ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()

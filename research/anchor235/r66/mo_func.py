"""mo_func.py -- assemble every candidate functional along the ladder and print the tables the
document reports: value at each machine, increment at each rung, the constant c(q') that would be
needed, and the verdict (does it bound F, is it monotone with c <= q').

Reads results/rungs.json (m5..m23 by period, one application of T at each), results/
ladder_K*_base23.json (the iterated ladder m29, m31, m37), results/excursion.json and
results/order.json.

Usage: uv run python research/anchor235/r66/mo_func.py [ladder_json]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mo_core import OUT, PRIMES, a_letter_floor

# corpus rows used where the instrument's depth ran out (alignment-rules.md 3.7, ladder_closure.md)
CORPUS_FJ = {29: [43, 55, 65, 70, 85, 90, 92, 97], 31: [58, 68, 85, 90, 92, 97, 104, 110],
             37: [88, 90, 97], 41: [91]}
CORPUS_QSTAR = {41: {1: 88, 2: 90, 3: 90, 4: 91}}
CORPUS_SECOND = {37: 85, 41: None}


def load():
    lad = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "ladder_K15_base23.json")
    rungs = json.load(open(os.path.join(OUT, "rungs.json")))
    ladder = json.load(open(lad)) if os.path.exists(lad) else []
    exc = json.load(open(os.path.join(OUT, "excursion.json")))
    order = json.load(open(os.path.join(OUT, "order.json"))) if os.path.exists(
        os.path.join(OUT, "order.json")) else {}
    M = {}
    for r in rungs:
        y = r["machine"]
        sp = {int(k): v for k, v in r["spec_full"].items()}
        vals = sorted(sp)
        M[y] = {"P": r["P"], "N": r["N"], "F": r["F"], "mu": r["mean_gap"], "Fj": r["Fj"],
                "second": vals[-2] if len(vals) > 1 else None,
                "top3": sum(vals[-3:]), "n_values": len(vals),
                "tails": r["tails"], "L": r["L"], "L_bare": r["L_bare"], "L_pad": r["L_pad"],
                "J_max": r["J_max"], "Qstar": {int(k): v for k, v in r["Qstar"].items()},
                "nJ": {int(k): v for k, v in r["nJ"].items()},
                "W": r["W"], "Z": r["Z"], "a_L": r["a_L"]}
    for r in ladder:
        y = r["machine"]
        sp = {int(k): v for k, v in r["spec_full"].items()}
        vals = sorted(sp)
        M[y] = {"P": r["P"], "N": r["N"], "F": r["F"], "mu": r["mean_gap"],
                "Fj": [x for x in r["Fj"]],
                "second": vals[-2] if len(vals) > 1 else None,
                "top3": sum(vals[-3:]), "n_values": len(vals), "tails": r["tails"],
                "L": r["L_next"], "L_bare": r["Lbare_next"], "L_pad": r["Lpad_next"],
                "J_max": r["J_max_next"],
                "Qstar_in": {int(k): v for k, v in r["Qstar"].items()},
                "nJ_in": {int(k): v for k, v in r["nJ"].items()},
                "W": r["W"], "Z": r["Z"], "a_L": r["a_L_next"]}
    for y, row in M.items():
        cj = CORPUS_FJ.get(y)
        if cj:
            fj = list(row["Fj"]) + [None] * 10
            row["Fj"] = [cj[i] if i < len(cj) and (fj[i] is None) else fj[i] for i in range(10)]
        if y in CORPUS_SECOND and CORPUS_SECOND[y]:
            row["second"] = CORPUS_SECOND[y]
    exd = {e["machine"]: e for e in exc}
    for y in M:
        if y in exd:
            M[y]["X"] = exd[y]["X"]
    return M, order


def main():
    M, order = load()
    ys = [y for y in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41] if y in M]
    print("\n== per machine ==")
    hdr = "y      N            P             F   mu      F_2..F_8"
    print(hdr)
    for y in ys:
        r = M[y]
        print(f"m{y:<3} {r['N']:>13,} {r['P']:>14,} {r['F']:>4} {r['mu']:.4f}  "
              f"{[x for x in r['Fj'][1:8]]}")

    print("\n== candidate (a): the record itself ==")
    print("rung  F(M)  q'  F(M+q')  budget  slack")
    for a, b in zip(ys, ys[1:]):
        print(f"{a}->{b}  {M[a]['F']:>4}  {b:>3}  {M[b]['F']:>6}  {M[a]['F']+b:>6}  "
              f"{M[a]['F']+b-M[b]['F']:>5}")

    print("\n== candidate (b): the F_J ladder, F_J(M+q') - F_J(M) against q' ==")
    print("rung   q'   " + "  ".join(f"J={j}" for j in range(1, 9)))
    for a, b in zip(ys, ys[1:]):
        cells = []
        for j in range(1, 9):
            u, v = M[a]["Fj"][j - 1], M[b]["Fj"][j - 1]
            if u is None or v is None:
                cells.append("   -")
            else:
                mark = "*" if v - u > b else " "
                cells.append(f"{v-u:>3}{mark}")
        print(f"{a}->{b}  {b:>3}  " + " ".join(cells))
    print("(* marks an increment above q': the F_J budget failing)")

    print("\n== candidate (b'): the second-largest realised value and the top-3 value sum ==")
    print("rung  q'  F2nd(M) F2nd(M+q') incr   top3(M) top3(M+q') incr")
    for a, b in zip(ys, ys[1:]):
        s1, s2 = M[a]["second"], M[b]["second"]
        t1, t2 = M[a]["top3"], M[b]["top3"]
        i1 = (s2 - s1) if (s1 and s2) else None
        print(f"{a}->{b} {b:>3}  {s1}   {s2}    {i1}    {t1}   {t2}   {t2-t1}"
              f"{'  * > q' if (t2 - t1) > b else ''}")

    print("\n== candidate (c): tail excess E_x and its densities ==")
    for x in ["2", "5", "10", "20", "30"]:
        print(f"  x = {x}:")
        for y in ys:
            t = M[y]["tails"].get(x)
            if t is None:
                continue
            print(f"    m{y:<3} E={t['E']:>16,}  count={t['count']:>14,}  "
                  f"E/N={t['E']/M[y]['N']:.6f}  E/P={t['E']/M[y]['P']:.6f}")

    print("\n== candidate (d): merge depth and the word length L ==")
    print("machine  L  L_bare  L_pad  J_max(next rung)")
    for y in ys:
        r = M[y]
        print(f"m{y:<5} {r['L']:>3} {r['L_bare']:>6} {r['L_pad']:>6} {r['J_max']:>6}")

    print("\n== candidate (e-i): the maximum excursion X = max_J (F_J - J mu) ==")
    print("machine  F     mu      X        X+mu   incr(X)   q'")
    prev = None
    for y in ys:
        r = M[y]
        if "X" not in r:
            continue
        inc = "" if prev is None else f"{r['X']-prev:.2f}"
        print(f"m{y:<5} {r['F']:>4} {r['mu']:.4f} {r['X']:>8.3f} {r['X']+r['mu']:>8.3f} "
              f"{inc:>9} {y:>4}")
        prev = r["X"]

    print("\n== candidate (e-ii): the letter-floor discount ==")
    print("rung  a_L  Q*_J                       Phi_letter(J>=2)  Phi_3(J>=3)  F(M)  argmax J")
    for a, b in zip(ys, ys[1:]):
        Q = M[a].get("Qstar") or M[b].get("Qstar_in") or CORPUS_QSTAR.get(b)
        if not Q:
            continue
        Q = {int(k): v for k, v in Q.items()}
        aL = a_letter_floor(b)
        cand = {J: Q[J] - max(J - 2, 0) * aL for J in Q if J >= 2}
        c3 = {J: Q[J] - (J - 2) * aL for J in Q if J >= 3}
        pl = max(cand.values()) if cand else None
        p3 = max(c3.values()) if c3 else None
        arg = [J for J in cand if cand[J] == pl]
        print(f"{a}->{b}  {aL:>3}  {Q}   {pl}   {p3}   {M[a]['F']}   {arg}")

    print("\n== candidate (e-iii): the scale-free triple named at node 4.i.b.ii ==")
    print("machine  W_1/N       W_2/N       W_3/N       Z_1/N       S=sum_{r>=2}C_r/N")
    for y in ys:
        r = M[y]
        W, Z, N = r["W"], r["Z"], r["N"]
        if len(W) < 4:
            continue
        S = sum(W[m] + Z[m] for m in range(1, len(W))) / N
        print(f"m{y:<5} {W[1]/N:.8f}  {W[2]/N:.3e}  {W[3]/N:.3e}  {Z[1]/N:.3e}  {S:.8f}")

    print("\n== candidate (e-iv): the de Bruijn relaxation B_k and the order of interaction ==")
    print("rung        budget  B_1   B_2   B_3   B_4   B_5   exact F(M+q')  least order")
    for name, v in order.items():
        y = int(name[1:])
        B = v["B"]
        cells = [str(B.get(str(k), "-")) for k in range(1, 6)]
        print(f"{name}->{v['gear']:<5} {v['budget']:>6}  " + "  ".join(f"{c:>4}" for c in cells) +
              f"   {M.get(v['gear'], {}).get('F', '?'):>6}        "
              f"{v['least_order_within_budget']}")


if __name__ == "__main__":
    main()

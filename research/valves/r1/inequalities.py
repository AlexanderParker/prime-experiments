"""Test the pre-registered candidate inequalities C1..C7 and the predictions V-P3, V-P5, V-P6 on the ten
ledgers, and emit the markdown ledger tables for the document.
usage: uv run python research/valves/r1/inequalities.py
"""
import json, os, glob, math

here = os.path.dirname(os.path.abspath(__file__))
res = os.path.join(here, "results")
runs = []
for fn in sorted(glob.glob(os.path.join(res, "ledger_q*_Q*.json"))):
    d = json.load(open(fn)); runs.append(d)
runs.sort(key=lambda d: (d["Q"], d["q"]))


def prod_p(q, f):
    v = 1.0
    for p in range(3, q + 1):
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            v *= f(p)
    return v


lines = []
tables = []
for d in runs:
    q, Q, rows = d["q"], d["Q"], d["rows"]
    sig_inf = 2.0 * prod_p(q, lambda p: p / (p - 2))
    c5 = 1 - 1 / sig_inf
    c6 = prod_p(q, lambda p: (p - 2) / (p - 1))
    C1 = [r["m"] for r in rows if r["P"] <= 0]
    C2 = [r["m"] for r in rows if r["Bember"] > 2 * r["E"]]
    C3 = [(r["m"], r["B"], r["P"]) for r in rows[:2] if r["B"] > r["P"]]
    C4 = [(rows[i]["m"], round(rows[i - 1]["B_over_T"], 4), round(rows[i]["B_over_T"], 4)) for i in range(1, len(rows)) if rows[i]["B_over_T"] < rows[i - 1]["B_over_T"]]
    C5 = [(r["m"], round(r["B_over_T"], 4)) for r in rows if r["B_over_T"] > c5]
    C6 = [(r["m"], round(r["P_over_A"], 4)) for r in rows if r["P_over_A"] is not None and r["P_over_A"] < c6]
    C7 = [(r["m"], r["P"], r["top_burnt"][0]) for r in rows if r["top_burnt"] and r["top_burnt"][0][1] > r["P"]]
    eq = [r["m"] for r in rows if r["families"] == r["A_adm"] and not r["missing"] and not r["extra"]]
    first_miss = next((r for r in rows if r["missing"]), None)
    r60 = rows[-1]
    tp = [r["T_over_P"] / r["Sigma_m"] for r in rows[49:60]]
    pa_dev = max((abs(r["P_over_A"] * r["Sigma1_m"] - 1) for r in rows[2:]), default=None)
    pa_dev_m = max(rows[2:], key=lambda r: abs(r["P_over_A"] * r["Sigma1_m"] - 1))["m"]
    bt_fuel = [r["Bfuel"] / r["T"] for r in rows]
    lines.append(f"### q = {q}, Q = {Q}")
    lines.append(f"- C1 P_m > 0: {'0 exceptions' if not C1 else 'FAILS at m = ' + str(C1)}; C2 B_ember <= 2E: {'0 exceptions' if not C2 else C2}; "
                 f"C3 (B_1 <= P_1, B_2 <= P_2): {'holds' if not C3 else 'FAILS ' + str(C3)}; "
                 f"C4 monotone B/T: {len(C4)} decreases{', first at m = ' + str(C4[0]) if C4 else ''}; "
                 f"C5 B/T <= {c5:.4f}: {'0 exceptions' if not C5 else str(len(C5)) + ' exceptions, first ' + str(C5[0])}; "
                 f"C6 P/A >= {c6:.4f}: {'0 exceptions' if not C6 else str(len(C6)) + ' exceptions, first ' + str(C6[0])}; "
                 f"C7 pure largest: {'0 exceptions' if not C7 else str(len(C7)) + ' exceptions, first ' + str(C7[0])}.")
    lines.append(f"- V-P3 families present = admissible with max <= m: equality at {len(eq)} of {len(rows)} turns; "
                 f"first miss at m = {first_miss['m'] if first_miss else None} ({first_miss['missing'][:4] if first_miss else ''}); missing over all turns {d['fam_missing_total']}, extra {d['fam_extra_total']}.")
    lines.append(f"- V-P5 at m = 60: B/T = {r60['B_over_T']:.4f} (pred before log correction {1 - 1/r60['Sigma_m']:.4f}, limit {c5:.4f}); "
                 f"T/P = {r60['T_over_P']:.3f} vs Sigma(60) = {r60['Sigma_m']:.3f}, ratio {r60['T_over_P']/r60['Sigma_m']:.3f}; mean ratio over m = 50..60: {sum(tp)/len(tp):.3f}; "
                 f"B_fuel/T at m = 60: {bt_fuel[-1]:.4f}; B/T at m = 1, 2, 3, 5, 9, 15, 30, 45, 60: " + ", ".join(f"{rows[i-1]['B_over_T']:.3f}" for i in (1, 2, 3, 5, 9, 15, 30, 45, 60)))
    lines.append(f"- V-P6 P/A against 1/Sigma_1(m): max relative deviation at m >= 3 is {pa_dev:.3f} (m = {pa_dev_m}); "
                 f"P/A at m = 1, 2, 3, 5, 9, 15, 30, 60: " + ", ".join(f"{rows[i-1]['P_over_A']:.3f}" for i in (1, 2, 3, 5, 9, 15, 30, 60)) +
                 "; predicted " + ", ".join(f"{1/rows[i-1]['Sigma1_m']:.3f}" for i in (1, 2, 3, 5, 9, 15, 30, 60)) + f"; limit {c6:.4f}.")
    # markdown table
    t = [f"#### Ledger q = {q}, Q = {Q} (turn m = (mQ, (m + 1)Q]; T total charges, B burnt = B_f fuelled-both + B_e ember-carrying, P pure = twins, E embers, pi primes, A primes with P + 2 open, fam/adm = fuelled families present / admissible with max <= m)", "",
         "| m | T | B | B_f | B_e | P | E | pi | A | fam/adm | B/T | P/A |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t.append(f"| {r['m']} | {r['T']} | {r['B']} | {r['Bfuel']} | {r['Bember']} | {r['P']} | {r['E']} | {r['pi']} | {r['A']} | {r['families']}/{r['A_adm']} | {r['B_over_T']:.3f} | {r['P_over_A']:.3f} |")
    t.append("")
    # families at m <= 12 with counts
    t.append("Burnt fuelled families per turn, m <= 12 (family: count): " + " ".join(
        f"m={r['m']}: " + ("{" + ", ".join(f"({k}) {v}" for k, v in r["fams_m_le12"].items()) + "}" if r["fams_m_le12"] else "{}") + ";" for r in rows[:12]))
    t.append("")
    tables.append("\n".join(t))

open(os.path.join(res, "inequalities.md"), "w").write("\n".join(lines) + "\n\n" + "\n".join(tables))
print("\n".join(lines))

"""Mechanism at the dead-cycle record of M + q' (question 3 of branch 7a).

For every run attaining F_c(M+q') (from cycle_record_<q'>.json): each cycle's six numbers with the
gears of M+q' dividing them; which slots q' kills ALONE (the interior openings of M the new gear
kills); their spacing in columns (class mod q': 0 = padded, +-2u' = literal), in multipliers
m = (30j+e)/q', and in cycles; the distinct-gear count per dead cycle.

Usage: uv run python research/anchor235/r34/cycle_mechanism.py [q' ...]   (default 11 13 17 19 23 29)
Writes research/anchor235/r34/results/mechanism_<q'>.txt
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
E = (11, 13, 17, 19, 29, 31)
OFFS = (2, 3, 5)


def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return [int(p) for p in np.flatnonzero(s)]


def analyse(qn):
    with open(os.path.join(RES, f"cycle_record_{qn}.json")) as f:
        rec = json.load(f)
    gears = rec["gears"]                     # 7..qn
    lower = [g for g in gears if g < qn]
    u = pow(6, -1, qn); d = (2 * u) % qn
    lines = [f"machine {{5..{qn}}} = M + {qn}, M = {{5..{lower[-1] if lower else 5}}}; u' = {u}, 2u' = {d}, "
             f"q'-2u' = {qn - d}; F_c = {rec['F_c']['max']} at {rec['F_c']['n_max']} runs; F = {rec['F']}"]
    lat_ok = 0; lat_n = 0; classes = {}; mdiffs = {}; gearcount = {}; nq_sole = []
    for (j0, L) in rec["F_c"]["runs"]:
        lines.append(f"\nrun of {L} dead cycles j = {j0}..{j0+L-1} (columns {5*j0+2}..{5*(j0+L-1)+5}, numbers {30*j0+11}..{30*(j0+L-1)+31})")
        kills = []      # (j, e, column, m) of q'-sole kills
        for j in range(j0, j0 + L):
            row = []
            forced = set(); allg = set(); m_open_slots = 0
            for i, e in enumerate(E):
                n = 30 * j + e
                divs = [g for g in gears if n % g == 0]
                allg |= set(divs)
                if len(divs) == 1: forced.add(divs[0])
                row.append(f"{n}:{'/'.join(map(str, divs)) if divs else '-'}")
                if divs == [qn]:
                    kills.append((j, e, 5 * j + OFFS[i // 2], n // qn))
            # every slot blocked (dead cycle); slot blocked under M?  (both members free of lower gears -> M-open slot)
            for t in range(3):
                a, b = 30 * j + E[2 * t], 30 * j + E[2 * t + 1]
                assert any(a % g == 0 or b % g == 0 for g in gears), (j, t)
                if all(a % g and b % g for g in lower): m_open_slots += 1
            gearcount[len(allg)] = gearcount.get(len(allg), 0) + 1
            lines.append(f"  j={j} (j mod 7 = {j % 7}): " + "  ".join(row) +
                         f"   | gears {sorted(allg)} ({len(allg)} distinct; forced {sorted(forced)}); M-open slots here: {m_open_slots}")
        nq_sole.append(len(kills))
        if kills:
            lines.append(f"  q'={qn} sole kills: " + ", ".join(f"j={j} e={e} col={c} m={m}" for j, e, c, m in kills))
        for (j1, e1, c1, m1), (j2, e2, c2, m2) in zip(kills, kills[1:]):
            dc = c2 - c1; cl = dc % qn
            name = "0 (padded)" if cl == 0 else ("+2u'" if cl == d else ("-2u'" if cl == (qn - d) % qn else f"ILLEGAL {cl}"))
            classes[name] = classes.get(name, 0) + 1
            mdiffs[m2 - m1] = mdiffs.get(m2 - m1, 0) + 1
            lat_n += 1
            lat_ok += int((30 * (j2 - j1) + e2 - e1) == qn * (m2 - m1))
            lines.append(f"    consecutive kills: columns {c1}->{c2} (diff {dc} = {name} mod q'), cycles {j1}->{j2} (diff {j2-j1}), "
                         f"multipliers {m1}->{m2} (diff {m2-m1}); identity 30*dj + de = q'*dm: {30*(j2-j1)+e2-e1} = {qn*(m2-m1)}")
    lines.append(f"\nSUMMARY {{5..{qn}}}: q'-sole kills per record run {nq_sole}; consecutive-kill classes {classes}; "
                 f"multiplier differences {mdiffs}; lattice identity {lat_ok}/{lat_n}; distinct killing gears per dead cycle {dict(sorted(gearcount.items()))}")
    txt = "\n".join(lines)
    with open(os.path.join(RES, f"mechanism_{qn}.txt"), "w") as f:
        f.write(txt + "\n")
    print(lines[0]); print(lines[-1])
    return {"q": qn, "sole_kills": nq_sole, "classes": classes, "mdiffs": mdiffs, "lattice": (lat_ok, lat_n), "gearcount": gearcount}


if __name__ == "__main__":
    qs = [int(a) for a in sys.argv[1:]] or [11, 13, 17, 19, 23, 29]
    out = [analyse(q) for q in qs]
    with open(os.path.join(RES, "mechanism_summary.json"), "w") as f:
        json.dump(out, f, indent=1)

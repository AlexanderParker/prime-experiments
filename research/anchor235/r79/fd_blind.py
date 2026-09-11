"""fd_blind.py -- the section against each field: blind sets (fields.md section 5; P3, P7).

T6: a member of F_j (j >= 3) below p'^2 has least prime factor < p'^(2/j), so F_j is blind to every column
open under the gears < p'^(2/j).  The square field is empty in every finer section and equals the squares of
the primes in (p_k, p_{k+1}) in a construction section.  On the cube-core-open columns (open under the gears
< p'^(2/3)) every strike is F_2 = g x r, g >= p'^(2/3), r prime in [g, p'^2 / g).

Per section: for each j >= 3 the confinement check (0 exceptions); the square field; the cube-core, its open
columns, the F_2 pins on them (g, r), the twins among them; and the per-gear location of the pins.
Usage: uv run python fd_blind.py
"""
import os, json, time
import numpy as np
from collections import Counter
from fd_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(RES, "blind.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def blind_census(s, spf, list_pins=False):
    section_census(s, spf)
    cols, L, R = s["cols"], s["L"], s["R"]
    omL, omR, lpfL, lpfR = s["omL"], s["omR"], s["lpfL"], s["lpfR"]
    pn = s["pn"]
    J = int(max(omL.max(), omR.max()))
    r = dict(name=s["name"], columns=len(cols), pn=pn, J=J)
    # confinement of F_j, j >= 3: least factor < pn^(2/j); equivalently the column is struck by a gear < pn^(2/j)
    conf = {}
    for j in range(3, J + 1):
        bound = pn ** (2.0 / j)
        exc = 0
        cnt = 0
        for om, lpf in ((omL, lpfL), (omR, lpfR)):
            sel = om == j
            cnt += int(sel.sum())
            exc += int((lpf[sel] >= bound).sum())
        conf[j] = dict(members=cnt, bound=round(bound, 2), gears_below=[g for g in s["gears"] if g < bound], exceptions=exc)
    r["confinement"] = conf
    # the square field
    sq = [int(x) for x in R[omR == 2] if int(round(np.sqrt(x))) ** 2 == x]
    r["squares"] = sq
    # the cube-core
    bound3 = pn ** (2.0 / 3)
    core = [g for g in s["gears"] if g < bound3]
    tail = [g for g in s["gears"] if g >= bound3]
    core_open = ~struck_columns(cols, core)
    twin = (omL == 1) & (omR == 1)
    n_open = int(core_open.sum())
    n_twin = int(twin.sum())
    # strikes on core-open columns: every composite member there must be F_2 with lpf >= bound3
    exc = 0
    pins = []
    for i in np.flatnonzero(core_open):
        for x, om, lpf, side in ((L[i], omL[i], lpfL[i], -1), (R[i], omR[i], lpfR[i], 1)):
            if om >= 2:
                if om != 2 or lpf < bound3:
                    exc += 1
                pins.append((int(cols[i]), side, int(lpf), int(x // lpf)))
    r["cube_core"] = dict(bound=round(bound3, 2), core=core, tail_count=len(tail), open=n_open, twins=n_twin,
                          ratio=n_open / n_twin if n_twin else None, pins=len(pins), exceptions=exc,
                          pinned_cols=len(set(p[0] for p in pins)))
    # per gear: how many pins, and the cofactor range actually used against the available primes in [g, pn^2/g)
    per_g = Counter(p[2] for p in pins)
    r["pins_per_gear"] = {int(g): int(per_g.get(g, 0)) for g in tail}
    if list_pins:
        r["pin_list"] = pins
    # twins from the cube-core reading: core-open and not pinned
    pinned = set(p[0] for p in pins)
    tw2 = sum(1 for i in np.flatnonzero(core_open) if int(cols[i]) not in pinned)
    r["cube_core"]["twins_from_pins"] = tw2
    return r


def main():
    t0 = time.time()
    results = []
    say("# blind sets", time.ctime())
    secs = [finer_section(p) for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]]
    secs += construction_sections(3, 2) + construction_sections(5, 2) + construction_sections(7, 2) + construction_sections(13, 1)
    hi = max(int(s["cols"][-1]) * 6 + 2 for s in secs)
    spf = spf_table(hi + 2)
    say("| section | p' | J | confinement per j>=3: members (bound; gears below; exceptions) | squares in section | cube-core gears (bound) | core-open | twins | open/twins | pins | pinned cols | exceptions | twins from pins | pins per tail gear |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in secs:
        r = blind_census(s, spf, list_pins=(len(s["cols"]) < 200))
        results.append(r)
        conf = "; ".join(f"j={j}: {d['members']} ({d['bound']}; {d['gears_below']}; exc {d['exceptions']})" for j, d in r["confinement"].items())
        cc = r["cube_core"]
        sqs = r["squares"] if len(r["squares"]) <= 8 else f"{len(r['squares'])} squares, first {r['squares'][:3]}"
        say(f"| {r['name']} | {r['pn']} | {r['J']} | {conf} | {sqs} | {cc['core']} ({cc['bound']}) | {cc['open']} | {cc['twins']} | "
            f"{cc['ratio']:.2f} | {cc['pins']} | {cc['pinned_cols']} | {cc['exceptions']} | {cc['twins_from_pins']} | {r['pins_per_gear']} |")
        if "pin_list" in r:
            say("   pins (column, side, g, r):", r["pin_list"])
    json.dump(results, open(os.path.join(RES, "blind.json"), "w"), indent=1, default=int)
    say("done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

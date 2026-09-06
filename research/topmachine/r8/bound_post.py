"""R8 post-processing of phases.py's rows: vacuity of the capacity bounds, slack by core density,
the minimisers under reflection, and the anchoring exceptions' mechanism.

usage: uv run python research/topmachine/r8/bound_post.py   (after phases.py)
"""

import json
import os
import sys
from collections import Counter
from fractions import Fraction
from math import prod

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "r7"))
from phases import cap1, cap2, gear_traces, enumerate_stats  # noqa: E402
from rule import F_rule  # noqa: E402

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def main():
    rows = json.load(open(os.path.join(RES, "phases_rows.json")))
    loaded = [r for r in rows if r["L0"]["core"]]
    say("# R8 post: vacuity, slack, mirror pairs, anchoring exceptions")
    say()

    # ---------------------------------------------------------------- vacuity
    for r in rows:
        g = r["gears"]
        hi = 3 * len(g) + 40
        r["vac1"] = cap1(g, hi) >= hi
        b, t = cap2(g, hi)
        r["vac2"] = b <= t
        r["dens"] = sum(Fraction(2, x) for x in g)
    v1 = sum(r["vac1"] for r in rows)
    v2 = sum(r["vac2"] for r in rows)
    say("## Vacuity: when the capacity criterion holds at every length")
    say()
    say(f"L70's criterion holds at `L = 3m + 40` (bound vacuous) on **{v1}** of {len(rows)} sets; "
        f"P15's on **{v2}**.")
    dens_vac = [r["dens"] for r in rows if r["vac2"]]
    dens_non = [r["dens"] for r in rows if not r["vac2"]]
    say(f"Gear density `sum 2/g`: vacuous sets have min {float(min(dens_vac)):.4f}, "
        f"non-vacuous sets have max {float(max(dens_non)):.4f}.")
    ge1_nonvac = sum(1 for r in rows if not r["vac2"] and r["dens"] >= 1)
    lt1_vac = sum(1 for r in rows if r["vac2"] and r["dens"] < 1)
    say(f"Sets with `sum 2/g >= 1` that are NOT vacuous: {ge1_nonvac}; vacuous sets with "
        f"`sum 2/g < 1`: {lt1_vac}.")
    if lt1_vac:
        ex = [r for r in rows if r["vac2"] and r["dens"] < 1][:6]
        say("  examples: " + "; ".join(f"{{{','.join(map(str, r['gears']))}}} density {float(r['dens']):.4f} F {r['F']}" for r in ex))
    say()

    # ---------------------------------------------------------------- slack, non-vacuous
    nv = [r for r in rows if not r["vac2"]]
    nv_loaded = [r for r in nv if r["L0"]["core"]]
    nv1 = [r for r in rows if not r["vac1"]]
    say("## Slack over the non-vacuous sets")
    say()
    s2 = Counter(r["Lcap2"] - r["F"] for r in nv)
    s1 = Counter(r["Lcap"] - r["F"] for r in nv1)
    say("| slack | L70, sets (of the non-vacuous for L70) | P15, sets (of the non-vacuous for P15) |")
    say("|---|---|---|")
    for k in sorted(set(s1) | set(s2)):
        say(f"| {k} | {s1.get(k, 0)} | {s2.get(k, 0)} |")
    say()
    say(f"P15 non-vacuous: {len(nv)} sets ({len(nv_loaded)} loaded); exact on "
        f"**{s2.get(0, 0)}** of them ({sum(1 for r in nv_loaded if r['Lcap2'] == r['F'])} of the "
        f"{len(nv_loaded)} loaded); maximum slack **{max(s2)}**.  L70 non-vacuous: {len(nv1)} sets, "
        f"exact on {s1.get(0, 0)}, maximum slack {max(s1)}.")
    say()
    worst = sorted(nv, key=lambda r: -(r["Lcap2"] - r["F"]))[:8]
    say("| gears | F | Lcap | Lcap2 | core at F+1 | density |")
    say("|---|---|---|---|---|---|")
    for r in worst:
        say(f"| {','.join(map(str, r['gears']))} | {r['F']} | {r['Lcap']} | {r['Lcap2']} | "
            f"{{{','.join(map(str, r['L1']['core']))}}} | {float(r['dens']):.3f} |")
    say()
    # the loose kinds again, restricted to non-vacuous
    loose = [r for r in nv_loaded if r["L1"]["cap2"] <= r["L1"]["t"]]
    ov = [r for r in loose if r["L1"]["minCnt"] > r["L1"]["t"]]
    say(f"Among the non-vacuous loaded sets, the bound is loose at `L = F + 1` on {len(loose)}: "
        f"{len(ov)} of the overlap kind, {len(loose) - len(ov)} of the run-parity kind.")
    say()

    # ---------------------------------------------------------------- density bands
    say("## Slack by core density at the deciding length")
    say()
    say("| core density rho = sum_{g <= F+2} 2/g | sets | loaded | Lcap2 = F | Lcap2 - F median | Lcap2 - F max | Lcap2 >= 3m+39 (beyond the scan) |")
    say("|---|---|---|---|---|---|---|")
    bands = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0), (1.0, 9)]
    for lo, hi in bands:
        sel = [r for r in rows if lo <= sum(2 / g for g in r["L1"]["core"]) < hi]
        if not sel:
            continue
        sl = sorted(r["Lcap2"] - r["F"] for r in sel)
        beyond = sum(1 for r in sel if r["Lcap2"] >= 3 * len(r["gears"]) + 39)
        say(f"| [{lo}, {hi}) | {len(sel)} | {sum(1 for r in sel if r['L0']['core'])} | "
            f"{sum(1 for r in sel if r['Lcap2'] == r['F'])} | {sl[len(sl) // 2]} | {sl[-1]} | {beyond} |")
    say()
    say("The same for L70 (`Lcap`):")
    say()
    say("| core density | sets | Lcap = F | Lcap - F median | Lcap - F max |")
    say("|---|---|---|---|---|")
    for lo, hi in bands:
        sel = [r for r in rows if lo <= sum(2 / g for g in r["L1"]["core"]) < hi]
        if not sel:
            continue
        sl = sorted(r["Lcap"] - r["F"] for r in sel)
        say(f"| [{lo}, {hi}) | {len(sel)} | {sum(1 for r in sel if r['Lcap'] == r['F'])} | {sl[len(sl) // 2]} | {sl[-1]} |")
    say()

    # ---------------------------------------------------------------- minimisers under reflection
    say("## The minimisers under reflection")
    say()
    odd = [r for r in loaded if r["L0"]["nmin"] % 2 == 1]
    one = [r for r in loaded if r["L0"]["nmin"] == 1]
    two = [r for r in loaded if r["L0"]["nmin"] == 2]
    say(f"At `L = F`: exactly one minimising phase vector (necessarily self-mirror) on **{len(one)}** "
        f"of {len(loaded)} loaded sets; exactly two on **{len(two)}**; an odd count on {len(odd)}.  "
        f"Examples with one: " + ", ".join("{" + ",".join(map(str, r["gears"])) + "}" for r in one[:6]) + ".")
    say()

    # ---------------------------------------------------------------- anchoring exceptions
    say("## The anchoring exceptions, mechanism")
    say()
    exc = [r for r in loaded if not r["L0"]["anchor0"]]
    cores = Counter(tuple(r["L0"]["core"]) for r in exc)
    say(f"{len(exc)} exceptions; their cores at `L = F`: " +
        ", ".join(f"{{{','.join(map(str, c))}}} x{n}" for c, n in cores.most_common()))
    fs = Counter((tuple(r["L0"]["core"]), r["F"], r["L0"]["t"]) for r in exc)
    say("(core, F, t) triples: " + ", ".join(f"{{{','.join(map(str, c))}}} F={F} t={t} x{n}" for (c, F, t), n in fs.most_common()))
    say()
    # one worked instance: list the minimising phases of the core
    r = exc[0]
    g = r["gears"]
    F = r["F"]
    core = r["L0"]["core"]
    say(f"Instance `{{{','.join(map(str, g))}}}`, `F = {F}`, core `{{{','.join(map(str, core))}}}`, "
        f"`t = {r['L0']['t']}`, `min D = {r['L0']['minD']}`:")
    say()
    say("| phase a of the core gear (tooth 0 at cell a) | trace in [0, F) | U_even | U_odd | D |")
    say("|---|---|---|---|---|")
    L = F
    for gg in core:
        tr, Le, Lo = gear_traces(gg, L)
        for a, (me, mo, ne, no) in enumerate(tr):
            cells = sorted([2 * i for i in range(Le) if (me >> i) & 1] + [2 * i + 1 for i in range(Lo) if (mo >> i) & 1])
            ue = [2 * i for i in range(Le) if not (me >> i) & 1]
            uo = [2 * i + 1 for i in range(Lo) if not (mo >> i) & 1]
            from rule import domino_cost_set
            D = domino_cost_set(ue) + domino_cost_set(uo)
            say(f"| {a} | {cells} | {ue} | {uo} | {D}{' (min)' if D == r['L0']['minD'] else ''} |")
    say()

    json.dump({"vac1": v1, "vac2": v2, "nonvac2": len(nv), "exact_nonvac2": s2.get(0, 0),
               "max_slack_nonvac2": max(s2), "one_minimiser": len(one), "two_minimisers": len(two),
               "anchor_exceptions": len(exc)},
              open(os.path.join(RES, "bound_post.json"), "w"), indent=1)
    with open(os.path.join(RES, "bound_post.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()

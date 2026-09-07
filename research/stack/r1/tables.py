"""Markdown tables from results/stack_<q>.json for research/proof/stacked_squares.md.
Usage: uv run python research/stack/r1/tables.py [q ...] > research/stack/r1/results/tables.md
"""
import sys, os, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def f(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def pat(p):
    return "".join("x" if b else "." for b in p) if p else "-"


def cls(c):
    if not c or c.get("cycles", 0) == 0:
        return None
    return c


def main(qs):
    data = {q: json.load(open(os.path.join(RES, f"stack_{q}.json"))) for q in qs}
    out = []
    P = out.append

    P("### T1. The machines (claim B)\n")
    P("| q | q# | cycles | k | g_k | g_k^2 | gears | count | below q# | square cycles | boundary | band cycles | band numbers |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            gears = f"{m['first_gear']}..{m['last_gear']}" if m["first_gear"] else "none"
            if m["final"]:
                gears = f"{m['first_gear']}..(q#)"
            cb = m.get("cycles_band")
            band = f"[{cb[0]}, {cb[1]})" if cb and cb[1] > cb[0] else "empty"
            bn = f"[{m['band_lo']}, {m['band_hi']})" if m.get("band_hi") and m["band_hi"] > m["band_lo"] else "empty"
            P(f"| {q} | {d['qsharp']} | {d['J']} | {m['k']} | {m['g']} | {m['square']} | {gears} | {m['n_gears']} | {m['n_gears_below_qsharp']} | "
              f"[0, {m['cycles_square'][1]}) | {f(m.get('boundary_cycle'))} | {band} | {bn} |")
    P("")

    P("### T2. Claim A verified (square part: new strikes are home strikes; band: open under 1..k = twin)\n")
    P("| q | k | new strikes below g_k^2 | home | not home | gears in slot classes | g_k^2 mod 30 | g_k^2 in cycle | g_k^2 struck as new | band slots (3 slots) | band twins (3 slots) | mismatches | boundary cycle numbers | struck by k | new by k |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            P(f"| {q} | {m['k']} | {m['claimA_square_new_strikes']} | {m['claimA_square_home']} | {m['claimA_square_new_not_home']} | {m['gears_in_slot_classes']} | "
              f"{m['square_mod_30']} | {f(m.get('square_cycle'))} | {m['square_struck_new']} | {m.get('band_slots_by_slot')} | {m.get('band_twins_by_slot')} | {m.get('claimA_band_mismatch')} | "
              f"{m.get('boundary_numbers')} | {pat(m.get('boundary_pattern'))} | {pat(m.get('boundary_new_pattern'))} |")
    P("")

    P("### T3. Cycle classification per machine and part (the machine's own strikes, echoes included)\n")
    P("| q | k | part | cycles | open | closed | mixed (2 open / 1 open) | slot strikes 11-13 / 17-19 / 29-31 | first closed | last open | first open | last closed | longest closed run (len @ j, ties) | longest open run | longest non-open run | CRT open | CRT closed |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            crt = m.get("crt")
            for part, key in (("square", "square_class"), ("band", "band_class"), ("all", "all_class")):
                c = cls(m.get(key))
                if c is None:
                    continue
                n = c["cycles"]
                eo = f"{crt['cycle_open'] * n:.1f}" if crt else "-"
                ec = f"{crt['cycle_closed'] * n:.1f}" if crt else "-"
                lc = c["longest_closed_run"]; lo = c["longest_open_run"]; ln = c["longest_nonopen_run"]
                P(f"| {q} | {m['k']} | {part} | {n} | {c['open']} | {c['closed']} | {c['mixed']} ({c['two_open']} / {c['one_open']}) | "
                  f"{c['slot_strikes'][0]} / {c['slot_strikes'][1]} / {c['slot_strikes'][2]} | {f(c['first_closed'])} | {f(c['last_open'])} | {f(c['first_open'])} | {f(c['last_closed'])} | "
                  f"{lc['length']} @ {f(lc['start'])} ({lc['count_at_max']}) | {lo['length']} @ {f(lo['start'])} | {ln['length']} @ {f(ln['start'])} | {eo} | {ec} |")
    P("")

    P("### T3b. The same on the band for the machine's NEW strikes (smallest prime factor in the machine) and for the JOINT strikes of machines 1..k\n")
    P("| q | k | strikes | cycles | open | closed | mixed | slot strikes | first open | last open | first closed | longest closed run | longest open run | longest non-open run |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            for lab, key in (("new", "band_class_new"), ("joint 1..k", "band_class_joint"), ("new (all)", "all_class_new")):
                c = cls(m.get(key))
                if c is None:
                    continue
                lc = c["longest_closed_run"]; lo = c["longest_open_run"]; ln = c["longest_nonopen_run"]
                P(f"| {q} | {m['k']} | {lab} | {c['cycles']} | {c['open']} | {c['closed']} | {c['mixed']} | {c['slot_strikes']} | {f(c['first_open'])} | {f(c['last_open'])} | {f(c['first_closed'])} | "
                  f"{lc['length']} @ {f(lc['start'])} ({lc['count_at_max']}) | {lo['length']} @ {f(lo['start'])} | {ln['length']} @ {f(ln['start'])} |")
    P("")

    P("### T4. The band profile in u = ln x / ln g_k: what the machine's open numbers are made of\n")
    P("| q | k | u | cycles | number open | prime | smooth x prime | smooth | model prime | model mixed | CRT per number | slot open | CRT slot | own/CRT | cycle open | cycle closed | twins | joint/product |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            crt = m.get("crt")
            for p in m.get("band_profile") or []:
                mo = p.get("model")
                P(f"| {q} | {m['k']} | {p['u'][0]:.2f}-{p['u'][1]:.2f} | {p['cycles']} | {p['number_open']:.3f} | {p['number_prime']:.3f} | {p['number_mixed']:.3f} | {p['number_smooth']:.3f} | "
                  f"{f(mo['prime']) if mo else '-'} | {f(mo['mixed']) if mo else '-'} | {f(crt['mertens_1']) if crt else '-'} | {p['own_slot_open']:.3f} | {f(crt['slot_open'][0]) if crt else '-'} | {p['own_over_crt']:.2f} | "
                  f"{p['cycle_open']:.4f} | {p['cycle_closed']:.3f} | {p['twins']} | {f(p['ratio'], 2)} |")
    P("")

    P("### T5. The connection to machine 1: open fractions per machine on each band, jointly open slots (= twins), the independence ratio\n")
    P("| q | k | band slots | open fraction by machine 1..k | product | joint (twins) | joint fraction | ratio | cycles open by machine 1..k | jointly open cycles |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            if m.get("band_slot_open_fraction_by_machine") is None:
                continue
            fr = m["band_slot_open_fraction_by_machine"]
            n = sum(m["band_slots_by_slot"]) if m.get("band_slots_by_slot") else 3 * (m["cycles_band"][1] - m["cycles_band"][0])
            P(f"| {q} | {m['k']} | {3 * (m['cycles_band'][1] - m['cycles_band'][0])} | {', '.join(f'{x:.4f}' for x in fr)} | {math.prod(fr):.5f} | {m['band_joint_open_slots']} | {m['band_joint_open_fraction']:.5f} | {f(m['band_independence_ratio'], 3)} | "
              f"{m['band_cycle_open_by_machine']} | {m['band_cycle_open_joint']} |")
    P("")

    P("### T6. Cross-machine relations\n")
    P("| q | k | band k | primes in band k | gears of k+1 (below q#) | equal | twins in band k | double-home slots of k+1 | mismatch | closed j of k mapped by the square | landing on closed j' of k+1 | rate | closed density of k+1 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for c in d["cross"]:
            sm = c.get("square_map")
            P(f"| {q} | {c['k']} | [{c['band'][0]}, {c['band'][1]}) | {c['primes_in_band']} | {c['gears_of_next_below_qsharp']} | {c['gear_sets_equal']} | {c['twins_in_band']} | {c['double_home_slots_of_next']} | {c['mismatch']} | "
              f"{sm['closed_of_k_mapped'] if sm else '-'} | {sm['landing_on_closed_of_next'] if sm else '-'} | {f(sm['rate']) if sm else '-'} | {f(sm['closed_density_of_next']) if sm else '-'} |")
    P("")

    P("### T7. The top cycle inside the period (j = q#/30 - 1) and the cycle at q# (j = q#/30), per machine (x = struck, . = open, six numbers 30j + 11, 13, 17, 19, 29, 31)\n")
    P("| q | k | top cycle j | numbers | struck by k | class | cycle at q# j | numbers | struck by k | class | new by k |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    def cl(p):
        sl = [p[0] or p[1], p[2] or p[3], p[4] or p[5]]
        s = sum(sl)
        return "open" if s == 0 else ("closed" if s == 3 else "mixed")
    for q, d in data.items():
        for m in d["machines"]:
            t = m["top_cycle_pattern"]; a = m["cycle_at_qsharp"]
            P(f"| {q} | {m['k']} | {t['j']} | {t['numbers'][0]}..{t['numbers'][5]} | {pat(t['own'])} | {cl(t['own'])} | {a['j']} | {a['numbers'][0]}..{a['numbers'][5]} | {pat(a['own'])} | {cl(a['own'])} | {pat(a['new'])} |")
    P("")

    P("### T8. Distinct gears striking a closed cycle of the band (sample of closed cycles, machines k >= 2)\n")
    P("| q | k | closed cycles sampled | minimum gears | histogram (gears: cycles) | max numbers struck in one cycle |")
    P("|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            if m.get("band_closed_gear_count_hist"):
                P(f"| {q} | {m['k']} | {m['band_closed_sampled']} | {m['band_closed_min_gears']} | {m['band_closed_gear_count_hist']} | {m.get('band_max_numbers_struck_per_cycle')} |")
    P("")

    P("### T9. CRT densities of each machine's gear set (full-period values) against the half law\n")
    P("| q | k | gears | prod(1 - 1/p) | prod(1 - 2/p) (slot open) | cycle open prod(1 - 6/p) | cycle closed | cycle mixed | 1/2 | 1/4 | 1/64 | 27/64 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q, d in data.items():
        for m in d["machines"]:
            crt = m.get("crt")
            if crt:
                P(f"| {q} | {m['k']} | {m['n_gears']} | {crt['mertens_1']:.4f} | {crt['slot_open'][0]:.4f} | {crt['cycle_open']:.5f} | {crt['cycle_closed']:.4f} | {crt['cycle_mixed']:.4f} | 0.5 | 0.25 | {1/64:.5f} | {27/64:.4f} |")
    P("")
    for q, d in data.items():
        if d.get("final_home") is not None:
            P(f"q = {q}: final machine home strikes {d['final_home']}, echoes {d['final_echo']}, seconds {d['seconds']:.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    qs = [int(a) for a in sys.argv[1:]] or [5, 7, 11, 13, 17, 19, 23]
    main(qs)

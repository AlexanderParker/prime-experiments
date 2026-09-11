"""xm_mech.py -- the mechanism of the first realisations: for every step of the staircase
x_min(p, .) (results/xm_scan_m{p}.json from xm_scan.py), the run's kill map and what fixes its
position.

For the run of length r at x = x_min(p, r):
  kill map      which gears strike each column x + o, o in [0, r);
  used gears    the gears striking at least one column of the run;
  forced gears  U = the gears that are the SOLE striker of some column of the run (their residue
                is fixed by that column in every cover of the run); M_U = prod U;
  the forced class   r_U = x mod M_U;  every y = r_U (mod M_U) has the forced gears striking the
                same offsets;  k* = (x - r_U) / M_U = the number of earlier members of the class,
                each of which FAILS to be a run (else x_min would be smaller); for the earliest
                failing members the report names the first uncovered offset and the gears that
                struck that offset at x (all non-forced there);
  coincidences  the columns of the run struck by both members of a twin-gear pair (g, g+2),
                g >= 5, listed per pair.
Also the section row: the staircase step at which r first reaches l_p = (p'^2 - p^2)/6.
"""
import argparse
import json
import os

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
NEXT = {PRIMES[i]: PRIMES[i + 1] for i in range(len(PRIMES) - 1)}


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def u_of(g):
    return pow(6, -1, g)


def strikes(g, k):
    u = u_of(g)
    return (k % g) in (u % g, (-u) % g)


def kill_map(x, r, gears):
    return [[g for g in gears if strikes(g, x + o)] for o in range(r)]


def crt_class(x, U):
    M = 1
    for g in U:
        M *= g
    return x % M, M


def analyse_run(x, r, gears, max_fail_report=3):
    km = kill_map(x, r, gears)
    used = sorted({g for col in km for g in col})
    forced = sorted({col[0] for col in km if len(col) == 1})
    sole_cols = {g: [o for o, col in enumerate(km) if col == [g]] for g in forced}
    rU, MU = crt_class(x, forced)
    kstar = (x - rU) // MU
    fails = []
    # the earliest failing members of the forced class (bounded report)
    j = 0
    y = rU
    while y < x and len(fails) < max_fail_report:
        # first offset not struck at y
        first_unc = None
        for o in range(r):
            if not any(strikes(g, y + o) for g in gears):
                first_unc = o
                break
        fails.append({"y": y, "j": j, "first_uncovered": first_unc,
                      "struck_at_x_by": km[first_unc] if first_unc is not None else None})
        j += 1
        y += MU
    coinc = {}
    for g in gears:
        if g + 2 in gears:
            cols = [o for o, col in enumerate(km) if g in col and (g + 2) in col]
            if cols:
                coinc[f"{g},{g + 2}"] = cols
    multiplicity = [len(col) for col in km]
    return {"x": x, "r": r, "used": used, "n_used": len(used), "forced": forced,
            "n_forced": len(forced), "M_U": MU, "r_U": rU, "k_star": kstar,
            "least_of_class": (kstar == 0), "fails": fails, "coincidences": coinc,
            "sole_offsets": {str(g): v for g, v in sole_cols.items()},
            "mean_multiplicity": sum(multiplicity) / r, "max_multiplicity": max(multiplicity),
            "kill_map": [[int(g) for g in col] for col in km] if r <= 160 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ps", nargs="+", type=int)
    ap.add_argument("--fail-report", type=int, default=3)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    allout = {}
    for p in a.ps:
        with open(os.path.join(here, "results", f"xm_scan_m{p}.json")) as f:
            sc = json.load(f)
        gears = gears_of(p)
        pn = NEXT[p]
        lp = (pn * pn - p * p) // 6
        rows = []
        for (x, r) in sc["stair"]:
            rows.append(analyse_run(x, r, gears, a.fail_report))
        sec = next((row for row in rows if row["r"] >= lp), None)
        out = {"p": p, "p_next": pn, "l_p": lp, "a": (p * p - 1) // 6, "b": (pn * pn - 1) // 6,
               "P": sc["P"], "scanned_to": sc["scanned_to"], "done": sc["done"],
               "rows": rows, "section_row": sec}
        allout[p] = out
        print(f"\n=== m{p}: gears {gears}, P = {sc['P']:,d}, l_p = {lp}, b = {out['b']} ===")
        print(f"{'r':>4} {'x_min':>16} {'x/P':>9} {'used':>5} {'forced':>6} {'M_U':>16} "
              f"{'least?':>6} {'k*':>10} {'meanmult':>8} {'coinc pairs (g>=11)':>20}")
        for row in rows:
            cp = [k for k in row["coincidences"] if not k.startswith("5,")]
            print(f"{row['r']:4d} {row['x']:16,d} {row['x'] / sc['P']:9.6f} {row['n_used']:5d} "
                  f"{row['n_forced']:6d} {row['M_U']:16,d} {str(row['least_of_class']):>6} "
                  f"{row['k_star']:10,d} {row['mean_multiplicity']:8.3f} {','.join(cp):>20}")
        if sec is not None:
            print(f"section row: r = {sec['r']} at x = {sec['x']:,d}; forced {sec['forced']}; "
                  f"M_U = {sec['M_U']:,d}; k* = {sec['k_star']:,d}")
            for fl in sec["fails"]:
                print(f"   earlier class member y = {fl['y']:,d} (j = {fl['j']}): first uncovered "
                      f"offset {fl['first_uncovered']}, struck at x by {fl['struck_at_x_by']}")
        else:
            print(f"section row: l_p = {lp} not reached in the scan (gmax {sc['gmax']}, scanned "
                  f"to {sc['scanned_to']:,d})")
    with open(os.path.join(here, "results", "xm_mech.json"), "w") as f:
        json.dump(allout, f)


if __name__ == "__main__":
    main()

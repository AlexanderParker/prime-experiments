"""rh_collide.py -- the collision laws of docs/proofs/21 on the PULLBACK (twisted separations
2 u_g q'^-1 mod g) at every rung m19..m53: the arc floor (c(g,h;L) = 0 for 2 <= L <=
max(a'_g, a'_h)) with its exception count against the real-teeth control; the shared-arc pairs of
the pullback and Theorem 2 on them; onsets; the rich-direction coincidence k(g,h;n) with its
+4 law; and the pair (5,7)'s stacking deficit against the full D(n) of rh_omega.py.

    uv run python research/anchor235/r68/rh_collide.py
"""
import json
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import OUT, arc, gears_of, min_g, next_gear, pair_deficits, pair_pattern, sep, twisted_sep, window_counts

RUNGS = [19, 23, 29, 31, 37, 41, 43, 47, 53]
LMAX = 60


def floor_test(gears, seps):
    """Per pair: arcs, onset, floor exceptions (L in [2, max arc] with c > 0), shared-arc check."""
    rows = {}
    n_exc_pairs = 0
    n_inst = 0
    n_exc = 0
    shared = []
    for g, h in combinations(gears, 2):
        ag, ah = arc(seps[g], g), arc(seps[h], h)
        Lmax = max(LMAX, max(ag, ah) + 2)
        c, k = pair_deficits(g, seps[g], h, seps[h], Lmax)
        onset = next((L for L in range(2, Lmax + 1) if c[L - 1] > 0), None)
        exc = [L for L in range(2, max(ag, ah) + 1) if c[L - 1] > 0]
        n_inst += max(ag, ah) - 1
        n_exc += len(exc)
        n_exc_pairs += bool(exc)
        row = {"a_g": ag, "a_h": ah, "onset": onset, "floor_exc": exc, "c": c[:30], "k": k[:30]}
        if ag == ah:
            row["shared_arc_c_at_a+1"] = c[ag]      # c(g,h; a+1), index a+1-1
            shared.append((g, h, ag, c[ag]))
        rows[f"{g},{h}"] = row
    return rows, n_exc_pairs, n_inst, n_exc, shared


def plus4_check(gears, seps, gh_max=400):
    bad = 0
    tested = 0
    for g, h in combinations(gears, 2):
        if g * h > gh_max:
            continue
        P = g * h
        c, k = pair_deficits(g, seps[g], h, seps[h], 2 * P)
        for n in range(1, P + 1):
            tested += 1
            if k[n + P - 1] != k[n - 1] + 4:
                bad += 1
            if c[n + P - 1] != c[n - 1] + 4:
                bad += 1
    return tested, bad


def main():
    omega = json.load(open(os.path.join(OUT, "omega.json")))
    res = {}
    for y in RUNGS:
        q = next_gear(y)
        gears = gears_of(y)
        tw = {g: twisted_sep(g, q) for g in gears}
        real = {g: sep(g) for g in gears}
        rows, ep, ni, ne, shared = floor_test(gears, tw)
        rrows, rep, rni, rne, rshared = floor_test(gears, real)
        t4, b4 = plus4_check(gears, tw)
        # rich direction: first n with k > 0 against first n with both min_g, min_h > 0
        rich_exc = 0
        rich_rows = {}
        for g, h in combinations(gears, 2):
            k = rows[f"{g},{h}"]["k"]
            first_k = next((n for n in range(1, 31) if k[n - 1] > 0), None)
            first_both = next((n for n in range(1, 31) if min_g(g, tw[g], n) > 0 and min_g(h, tw[h], n) > 0), None)
            rich_rows[f"{g},{h}"] = {"first_k_pos": first_k, "first_both_forced": first_both}
            if first_k != first_both:
                rich_exc += 1
        # the pair (5,7): its own stacking deficit against the full D(n)
        Dfull = omega[f"m{y}"]["D"]
        U, A, B = pair_pattern(5, tw[5], 7, tw[7])
        d57 = {}
        agree = 0
        for n in range(1, 21):
            jm = int(window_counts(U, n).min())
            m5, m7 = min_g(5, tw[5], n), min_g(7, tw[7], n)
            d57[n] = jm - max(m5, m7)
            agree += (d57[n] == Dfull[str(n)])
        twins = [(g, h) for g, h in combinations(gears, 2) if h == g + 2]
        twin_shared = [(g, h) for g, h, a, c in shared if h == g + 2]
        res[f"m{y}"] = {"q": q, "twisted_sep": tw, "arcs": {g: arc(tw[g], g) for g in gears},
                        "pairs": rows, "floor_exc_pairs": ep, "floor_instances": ni, "floor_exc": ne,
                        "real_floor_exc": rne, "real_floor_instances": rni,
                        "shared_arc_pairs": shared, "twin_pairs": twins, "twin_shared": twin_shared,
                        "plus4_tested": t4, "plus4_bad": b4, "rich_first_exc": rich_exc, "rich_rows": rich_rows,
                        "D57": d57, "D_full": {int(k): v for k, v in Dfull.items()}, "D_agree_n": agree}
        arcs1 = [g for g in gears if arc(tw[g], g) == 1]
        print(f"m{y} -> {q}: arcs {res[f'm{y}']['arcs']}; arc-1 gears {arcs1}")
        print(f"   arc floor on the pullback: {ne} exceptions in {ni} instances over {len(rows)} pairs "
              f"({ep} pairs fail); real teeth control: {rne} in {rni}")
        print(f"   shared-arc pairs (twisted): {shared}  [Theorem 2 c(a+1) >= 1 at all: "
              f"{all(c >= 1 for _, _, _, c in shared)}]; twin pairs {twins}, twins sharing an arc {twin_shared}")
        print(f"   +4 law (c and k, pairs with gh <= 400): {b4} violations in {t4} instances")
        print(f"   rich first-coincidence = first both-forced: {rich_exc} exceptions in {len(rich_rows)} pairs")
        print(f"   D(5,7)(n) n=1..20: {[d57[n] for n in range(1, 21)]}")
        print(f"   D_full(n)  n=1..20: {[Dfull[str(n)] for n in range(1, 21)]}  agree at {agree}/20")
        fails = {k: v for k, v in rows.items() if v["floor_exc"]}
        for k, v in list(fails.items())[:8]:
            print(f"      floor fails at pair {k}: arcs {v['a_g']},{v['a_h']} onset {v['onset']} exc L {v['floor_exc']}")
    with open(os.path.join(OUT, "collide.json"), "w") as f:
        json.dump(res, f, indent=1, default=int)


if __name__ == "__main__":
    main()

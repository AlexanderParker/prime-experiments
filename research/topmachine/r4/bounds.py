"""The bounds in use: the covering bound's exact death point, the first-hit model above the
zone, the tail fraction, and the saturation test on a fixed region above the zone.

usage: uv run python research/topmachine/r4/bounds.py
"""

import json
import os
from math import exp, isqrt, log

import numpy as np

from zone import primes_upto, range_record, smooth_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = []
MERTENS = 0.2614972128


def say(s=""):
    print(s)
    OUT.append(str(s))


def L_star(q):
    """The length at which sum_{q < g <= L} 1/g reaches 1/2 (Mertens); the covering bound
    L <= 2m/(1 - 2 H_S) is non-vacuous exactly while the record is below this."""
    s = sum(1.0 / int(p) for p in primes_upto(q))
    return exp(exp(0.5 + s - MERTENS))


def H_S(q, F, Q):
    return sum(1.0 / int(p) for p in primes_upto(min(F, Q)) if p > q)


def zone_record_from_smooth(q, Q, openings):
    """The zone record predicted from the finite smooth-pair list plus the first opening
    at or after the largest smooth pair <= Q - 2."""
    sp = [n for n in smooth_pairs(q, Q) if n <= Q - 2]
    if not sp:
        return None, None
    best, at = 0, None
    for a, b in zip(sp, sp[1:]):
        if b - a - 1 > best:
            best, at = b - a - 1, a + 1
    nxt = min(o for o in openings if o > sp[-1])
    if nxt - sp[-1] - 1 > best:
        best, at = nxt - sp[-1] - 1, sp[-1] + 1
    return best, at


def main():
    say("# The bounds in use")
    say()

    # ---------- the exact law for the zone record, and the two-part record ----------
    say("## L47  the zone record from the finite smooth-pair list (exact law)")
    say()
    say("| q | N | Q | zone record measured | at | predicted from the smooth list | at | "
        "agrees |")
    say("|---|---|---|---|---|---|---|---|")
    bad = 0
    tot = 0
    rows = []
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 37):
        for N in (10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8):
            Q = isqrt(N)
            gears = [int(p) for p in primes_upto(Q) if p > q]
            st = range_record(gears, N, Q)
            ops = st["open_le_Q"] + [st["first_above_Q"]]
            pred, pat = zone_record_from_smooth(q, Q, ops)
            ok = (pred == st["best_zone"][0]) and (pat == st["best_zone"][1])
            tot += 1
            bad += 0 if ok else 1
            rows.append({"q": q, "N": N, "Q": Q, "m": len(gears),
                         "zone": st["best_zone"][0], "zone_at": st["best_zone"][1],
                         "pred": pred, "pred_at": pat, "ok": ok,
                         "F": st["best"][0], "above": st["best_above"][0],
                         "density": st["count"] / N})
            say(f"| {q} | 1e{len(str(N))-1} | {Q:,} | {st['best_zone'][0]:,} | "
                f"{st['best_zone'][1]:,} | {pred:,} | {pat:,} | {'yes' if ok else '**NO**'} |")
    say()
    say(f"**{bad} exceptions of {tot}.**")
    say()

    # ---------- the covering bound's death point ----------
    say("## L49  the covering bound 2m/(1 - 2 H_S): where it dies, exactly")
    say()
    say("| q | L*(q) | N | Q | m | 2m | record F | H_S | covering bound | alive? | "
        "F < L*(q)? |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        q, N, Q, F, m = r["q"], r["N"], r["Q"], r["F"], r["m"]
        Ls = L_star(q)
        H = H_S(q, F, Q)
        b = 2 * m / (1 - 2 * H) if H < 0.5 else None
        say(f"| {q} | {Ls:.0f} | 1e{len(str(N))-1} | {Q:,} | {m} | {2*m} | {F:,} | "
            f"{H:.3f} | {('%.0f' % b) if b else 'vacuous'} | "
            f"{'alive' if b else 'dead'} | {'yes' if F < Ls else 'no'} |")
    say()

    # ---------- the record against 2m, and m log m ----------
    say("## L50  the in-use record against the large-gear parity value 2m")
    say()
    say("| q | N | m | 2m | F_range | F / 2m | Q - s(q) - 1 (proved) | F / (Q) |")
    say("|---|---|---|---|---|---|---|---|")
    sq = {q: max(smooth_pairs(q, 10 ** 13)) for q in (5, 7, 11, 13, 17, 19, 23, 29, 37)}
    for r in rows:
        lb = r["Q"] - sq[r["q"]] - 1
        say(f"| {r['q']} | 1e{len(str(r['N']))-1} | {r['m']} | {2*r['m']} | {r['F']:,} | "
            f"{r['F'] / (2*r['m']):.2f} | {lb if lb > 0 else '-'} | "
            f"{r['F'] / r['Q']:.3f} |")
    say()

    # ---------- the tail fraction in use ----------
    say("## L51  in use the machine is nearly all core")
    say()
    say("| q | N | Q | m | proved F_top >= | tail gears <= | tail fraction |")
    say("|---|---|---|---|---|---|---|")
    for r in rows:
        lb = max(r["F"], r["Q"] - sq[r["q"]] - 1)
        gears = [int(p) for p in primes_upto(r["Q"]) if p > r["q"]]
        tail = len([g for g in gears if g > lb + 1])
        say(f"| {r['q']} | 1e{len(str(r['N']))-1} | {r['Q']:,} | {r['m']} | {lb:,} | "
            f"{tail} | {tail / r['m']:.3f} |")
    say()

    # ---------- the first-hit model above the zone ----------
    say("## L52  above the zone: the record against the first-hit (Poisson) model")
    say()
    say("| q | N | Q | density p | above-zone record A | first-hit model "
        "ln(Np)/(-ln(1-p)) | A / model |")
    say("|---|---|---|---|---|---|---|")
    for r in rows:
        p = r["density"]
        model = log(r["N"] * p) / (-log(1 - p))
        say(f"| {r['q']} | 1e{len(str(r['N']))-1} | {r['Q']:,} | {p:.5f} | {r['above']:,} | "
            f"{model:.0f} | {r['above'] / model:.2f} |")
    say()

    # ---------- saturation on a fixed region strictly above every zone ----------
    say("## P10c  a fixed test region (10^6, 10^7], gears (q, Q] with Q <= 10^6: "
        "no saturation")
    say()
    say("| q | Q | m | max walk on (10^6, 10^7] | at | openings on the region |")
    say("|---|---|---|---|---|---|")
    N = 10 ** 7
    for q in (5, 19):
        for Q in (3162, 10 ** 4, 31623, 10 ** 5, 3 * 10 ** 5, 10 ** 6):
            gears = [int(p) for p in primes_upto(Q) if p > q]
            st = range_record(gears, N, 10 ** 6)
            say(f"| {q} | {Q:,} | {len(gears)} | {st['best_above'][0]:,} | "
                f"{st['best_above'][1]:,} | {st['count']:,} |")
        say()

    json.dump(rows, open(os.path.join(RES, "bounds.json"), "w"), indent=1)
    with open(os.path.join(RES, "bounds.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()

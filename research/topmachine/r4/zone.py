"""The in-use machine: the gear zone, the smooth-pair identity, and the range record.

Gear set G = primes in (q, Q], pair coordinate, range [1, N].  Everything exact.

P4  for n <= Q - 2 the pair n is open iff n and n + 2 are both q-smooth
P5  F_range(N) >= Q - s(q),  s(q) the largest q-smooth pair
P8  where the record block sits
P9  the record of the walk above the gear zone

usage: uv run python research/topmachine/r4/zone.py
"""

import json
import os
from math import isqrt, prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CHUNK = 1 << 23

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def primes_upto(n):
    if n < 2:
        return np.array([], dtype=np.int64)
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, isqrt(n) + 1):
        if s[p]:
            s[p * p:: p] = False
    return np.flatnonzero(s).astype(np.int64)


def smooth_numbers(q, limit):
    """All n <= limit whose prime factors are all <= q.  Sorted."""
    ps = [int(p) for p in primes_upto(q)]
    out = [1]
    for p in ps:
        new = []
        for v in out:
            w = v * p
            while w <= limit:
                new.append(w)
                w *= p
        out.extend(new)
    out.sort()
    return out


def smooth_pairs(q, limit):
    """All n <= limit with n and n + 2 both q-smooth."""
    sm = smooth_numbers(q, limit + 2)
    st = set(sm)
    return [n for n in sm if n <= limit and (n + 2) in st]


def openings_scan(gears, N, callback):
    """Walk [0, N] in chunks, calling callback(positions) with open pair positions."""
    gears = np.asarray(gears, dtype=np.int64)
    base = 0
    while base <= N:
        n = min(CHUNK, N + 1 - base)
        a = np.ones(n, dtype=bool)
        for g in gears:
            g = int(g)
            a[(-base) % g:: g] = False
            a[(-2 - base) % g:: g] = False
        idx = np.flatnonzero(a)
        if len(idx):
            callback(idx.astype(np.int64) + base)
        base += n


def range_record(gears, N, Q, lo=1):
    """Exact record data over [lo, N].

    Returns dict with the overall record (block start, length), the record among blocks
    starting at or below Q (the gear zone), the record among blocks starting above Q,
    the list of openings <= Q, and the first opening above Q.
    """
    state = {
        "prev": None,
        "best": (0, None),
        "best_zone": (0, None),
        "best_above": (0, None),
        "open_le_Q": [],
        "first_above_Q": None,
        "count": 0,
    }

    def cb(pos):
        st = state
        st["count"] += len(pos)
        le = pos[pos <= Q]
        if len(le):
            st["open_le_Q"].extend(int(v) for v in le)
        if st["first_above_Q"] is None:
            ab = pos[pos > Q]
            if len(ab):
                st["first_above_Q"] = int(ab[0])
        prev = st["prev"]
        arr = pos if prev is None else np.concatenate([[prev], pos])
        if len(arr) >= 2:
            starts = arr[:-1] + 1
            lens = arr[1:] - arr[:-1] - 1
            k = int(np.argmax(lens))
            if int(lens[k]) > st["best"][0]:
                st["best"] = (int(lens[k]), int(starts[k]))
            sel = starts <= Q
            if sel.any():
                lz = lens[sel]
                kk = int(np.argmax(lz))
                if int(lz[kk]) > st["best_zone"][0]:
                    st["best_zone"] = (int(lz[kk]), int(starts[sel][kk]))
            sel = starts > Q
            if sel.any():
                la = lens[sel]
                kk = int(np.argmax(la))
                if int(la[kk]) > st["best_above"][0]:
                    st["best_above"] = (int(la[kk]), int(starts[sel][kk]))
        st["prev"] = int(pos[-1])

    openings_scan(gears, N, cb)
    return state


def main():
    rows = []
    say("# The in-use machine: the gear zone and the record")
    say()

    # ---------- P4, P5, P8: the identity and the record, at Q = isqrt(N) ----------
    say("## P4/P5/P8  gears (q, Q], Q = floor(sqrt N)")
    say()
    say("| q | N | Q | m | open pairs <= Q | q-smooth pairs <= Q | identity | "
        "F_range | at | zone record | at | above-zone record | at | first opening > Q |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 37):
        for N in (10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8):
            Q = isqrt(N)
            gears = [int(p) for p in primes_upto(Q) if p > q]
            if not gears:
                continue
            st = range_record(gears, N, Q)
            sp = [n for n in smooth_pairs(q, Q) if n <= Q - 2]
            op = [n for n in st["open_le_Q"] if 1 <= n <= Q - 2]
            ident = "0 exceptions" if op == sp else (
                f"MISMATCH {sorted(set(op) ^ set(sp))[:6]}")
            rows.append({
                "q": q, "N": N, "Q": Q, "m": len(gears),
                "n_open_le_Q": len(op), "n_smooth_pairs": len(sp),
                "identity_ok": op == sp,
                "F_range": st["best"][0], "F_at": st["best"][1],
                "zone": st["best_zone"][0], "zone_at": st["best_zone"][1],
                "above": st["best_above"][0], "above_at": st["best_above"][1],
                "first_above_Q": st["first_above_Q"],
                "largest_smooth_pair_le_Q": max(sp) if sp else None,
            })
            say(f"| {q} | 1e{len(str(N))-1} | {Q:,} | {len(gears)} | {len(op)} | {len(sp)} | "
                f"{ident} | {st['best'][0]:,} | {st['best'][1]:,} | "
                f"{st['best_zone'][0]:,} | {st['best_zone'][1]:,} | "
                f"{st['best_above'][0]:,} | {st['best_above'][1]:,} | "
                f"{st['first_above_Q']:,} |")
    say()

    # ---------- s(q): the global largest smooth pair ----------
    say("## P5/P11  the Stormer set: q-smooth pairs (n, n+2)")
    say()
    say("| q | pairs found <= 1e13 | largest s(q) | the pair | all pairs (n) |")
    say("|---|---|---|---|---|")
    sq = {}
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 37):
        lim = 10 ** 13
        sp = smooth_pairs(q, lim)
        sq[q] = max(sp)
        show = ", ".join(str(v) for v in sp) if len(sp) <= 24 else (
            ", ".join(str(v) for v in sp[:12]) + ", ..., "
            + ", ".join(str(v) for v in sp[-6:]))
        say(f"| {q} | {len(sp)} | **{max(sp):,}** | ({max(sp):,}, {max(sp)+2:,}) | {show} |")
    say()

    # ---------- the proved lower bound against the truth ----------
    say("## P5/P7  the proved lower bound Q - s(q) against the measured record")
    say()
    say("| q | N | Q | s(q) | Q - s(q) | measured F_range | ratio | record block start | "
        "s(q) + 1 | P8 |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = sq[r["q"]]
        lb = r["Q"] - s
        ok = "yes" if r["F_at"] == s + 1 else "no"
        rat = r["F_range"] / lb if lb > 0 else float("nan")
        say(f"| {r['q']} | 1e{len(str(r['N']))-1} | {r['Q']:,} | {s:,} | "
            f"{lb:,} | {r['F_range']:,} | {rat:.4f} | {r['F_at']:,} | {s+1:,} | {ok} |")
        r["s_q"] = s
        r["lower_bound"] = lb
    say()

    json.dump({"rows": rows, "s_q": sq}, open(os.path.join(RES, "zone.json"), "w"), indent=1)
    with open(os.path.join(RES, "zone.out"), "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()

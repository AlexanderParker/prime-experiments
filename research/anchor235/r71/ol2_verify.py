"""ol2_verify.py -- independent re-check of the decisive numbers of a finished rung.

The decisive direction of a lower bound on B_k is a set of YES verdicts (the binding word's
k-subwindows are realised), and a YES can be CERTIFIED rather than trusted:

  * the covering search is re-run here in a few lines of its own, returning the phase t_g of every
    gear rather than a boolean;
  * t_g = (u_g - x) mod g, so CRT turns the phase vector into an explicit COLUMN x of the machine;
  * the window is then verified at that column directly against the strike rule
    (column c is struck by gear g iff c = +-u_g mod g) -- no covering argument, no instrument, no
    dictionary.  The output is an integer the reader can check by hand.

So a lower bound B_k >= S comes out of this script as: one explicit column of M per k-subwindow of
the witness, plus the arithmetic of the fusion at phase z of q'.

The second half is the EXACTNESS audit.  Every fusing J-word of span above a floor is
re-enumerated and its k-subwindows looked up in the memo.  A word survives the audit only if one
of its k-subwindows is decided NOT realised.  A word with no False and a missing verdict is an
UNDECIDED -- the descending scan would have skipped it silently, so B_k would be a lower bound and
not the exact value -- and is reported, and with `--resolve` put to the full solver.

Usage: uv run python research/anchor235/r71/ol2_verify.py <y> <k> <floor> [procs] [--resolve]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ol2_core import (F2_CAP_ONLY, F2_RECORD, F_RECORD, OUT, d_of, decide_many,  # noqa: E402
                      enumerate_fusing, fj_cap, gears_upto, letter_class, load_memo, memo,
                      next_gear, save_memo, u_of)

NAMES = {0: "PAD", 1: "UP", 2: "DOWN", 3: "BAD"}


def fuses(word, q, z):
    """The word's interior offsets struck and both flanks unstruck, at phase z of q'."""
    d = d_of(q)
    struck = {0, d}
    o, off = 0, []
    for g in word:
        o += g
        off.append(o)
    if (z % q) in struck:
        return False
    for oo in off[:-1]:
        if (oo + z) % q not in struck:
            return False
    return (off[-1] + z) % q not in struck


# --------------------------------------------------------------- a witness, not a boolean

def find_phases(offsets, gears, budget=20_000_000):
    """Return {gear: phase} realising the window, or None.  Own search, written here: filter each
    gear's phases by the OPEN offsets, then cover the CLOSED offsets, always branching on the
    closed offset with the fewest remaining ways of being struck."""
    S = int(offsets[-1])
    opens = set(int(o) for o in offsets)
    closed = [o for o in range(S + 1) if o not in opens]
    full = (1 << (S + 1)) - 1
    cmask = 0
    for o in closed:
        cmask |= 1 << o
    per = {}
    for g in gears:
        d = d_of(g)
        bad = set()
        for o in opens:
            bad.add(o % g)
            bad.add((o - d) % g)
        ms = []
        for t in range(g):
            if t in bad:
                continue
            m = 0
            for o in closed:
                if o % g == t or o % g == (t + d) % g:
                    m |= 1 << o
            ms.append((t, m))
        if not ms:
            return None
        per[g] = ms
    nodes = [0]

    def rec(unc, rem, assign):
        if unc == 0:
            return dict(assign)
        nodes[0] += 1
        if nodes[0] > budget:
            raise RuntimeError("witness search budget exceeded")
        need = bin(unc).count("1")
        tot = 0
        for g in rem:
            tot += max(bin(m & unc).count("1") for _, m in per[g])
            if tot >= need:
                break
        else:
            if tot < need:
                return None
        best = None
        u = unc
        while u:
            b = u & -u
            p = b.bit_length() - 1
            u ^= b
            opts = [(g, t, m) for g in rem for t, m in per[g] if m >> p & 1]
            if not opts:
                return None
            if best is None or len(opts) < len(best):
                best = opts
                if len(opts) <= 1:
                    break
        for g, t, m in best:
            r = rec(unc & ~m, tuple(x for x in rem if x != g), assign + [(g, t)])
            if r is not None:
                return r
        return None

    assert full  # span is positive
    res = rec(cmask, tuple(gears), [])
    if res is None:
        return None
    # gears not needed for the cover are free: park each on a phase that strikes nothing open
    # (every phase in per[g] has that property by construction)
    for g in gears:
        res.setdefault(g, per[g][0][0])
    return res


def crt(pairs):
    """x with x = r (mod m) for each (r, m), the moduli pairwise coprime."""
    x, M = 0, 1
    for r, m in pairs:
        # solve x + M*t = r (mod m)
        t = ((r - x) * pow(M, -1, m)) % m
        x += M * t
        M *= m
    return x % M, M


def certify_word(gaps, gears):
    """Find an explicit COLUMN of the machine at which the window (gaps) occurs, and verify it
    there directly against the strike rule.  Returns the certificate or None."""
    off, o = [0], 0
    for g in gaps:
        o += int(g)
        off.append(o)
    ph = find_phases(off, gears)
    if ph is None:
        return None
    # gear g strikes the offsets o = (+-u_g - x) mod g, and the solver's two strike classes are
    # {t_g, t_g + d_g} with d_g = 2 u_g, so t_g = (-u_g - x) mod g and t_g + d_g = (u_g - x).
    # Hence x = -u_g - t_g (mod g), and CRT over the gears names the column.
    x, M = crt([((-u_of(g) - ph[g]) % g, g) for g in gears])
    S = off[-1]
    openset = set(off)

    def struck(c):
        return any(c % g in (u_of(g) % g, (-u_of(g)) % g) for g in gears)

    ok_open = all(not struck(x + oo) for oo in off)
    ok_closed = all(struck(x + c) for c in range(S + 1) if c not in openset)
    return {"gaps": [int(g) for g in gaps], "span": S, "column": int(x), "period": int(M),
            "phases": {str(g): int(ph[g]) for g in gears},
            "all_openings_open": bool(ok_open), "all_others_struck": bool(ok_closed),
            "verified": bool(ok_open and ok_closed)}


def subwindows(word, k):
    if len(word) <= k:
        return [tuple(word)]
    return [tuple(word[i:i + k]) for i in range(len(word) - k + 1)]


def main():
    y = int(sys.argv[1])
    k = int(sys.argv[2])
    floor = int(sys.argv[3])
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    resolve = "--resolve" in sys.argv
    gears = gears_upto(y)
    q = next_gear(y)
    FM = F_RECORD[y]
    Fnext = F_RECORD[q]
    cap2 = F2_RECORD.get(y, F2_CAP_ONLY.get(y, Fnext))
    CAPS = {j: fj_cap(y, j) for j in range(1, 9)}
    print(f"=== verify m{y} -> {q};  k = {k}, floor = {floor} ===", flush=True)
    print(f"u = {u_of(q)}, d = {d_of(q)}, F(m{y}) = {FM}, budget = {FM + q}, "
          f"certified F(m{q}) = {Fnext}, caps {CAPS}", flush=True)
    n0 = load_memo(os.path.join(OUT, f"memo_m{y}.json"))
    print(f"memo: {n0:,} verdicts on file", flush=True)

    rep = json.load(open(os.path.join(OUT, f"step_{y}_{q}.json")))
    out = {"y": y, "q": q, "k": k, "floor": floor, "gears": gears}

    # ---------------------------------------------------------------- 1. certified witnesses
    out["witnesses"] = {}
    todo = []
    for kk in (k, k - 1):
        for J, r in (rep.get(f"per_J_{kk}") or {}).items():
            if r.get("subsumed") or r.get("word") is None:
                continue
            todo.append((f"B_{kk} term J={J}", [int(x) for x in r["word"]], r["phase"], kk))
    rw = rep.get("record_witness")
    if rw:
        todo.append(("record witness", [int(x) for x in rw["word"]], rw["phase"], None))
    for label, w, z, kk in todo:
        t0 = time.time()
        rec = {"word": w, "phase": z, "span": sum(w), "fuses_at_phase": fuses(w, q, z),
               "letters": [NAMES[letter_class(g, q)] for g in w]}
        def safe(t):
            try:
                return certify_word(list(t), gears)
            except RuntimeError:
                return "budget exceeded"

        if kk is None:                                   # the record witness is realised itself
            rec["certificate"] = safe(w)
        else:
            rec["subwindow_certificates"] = {",".join(map(str, t)): safe(t)
                                             for t in subwindows(w, kk)}
            rec["word_itself"] = safe(w)
        rec["secs"] = round(time.time() - t0, 1)
        out["witnesses"][label] = rec
        def say(c):
            if c is None:
                return "NOT realised"
            if not isinstance(c, dict):
                return str(c)
            return f"realised at column {c['column']}, verified {c['verified']}"

        if kk is None:
            print(f"  {label}: {w} span {sum(w)} phase {z}; fuses {rec['fuses_at_phase']}; "
                  f"{say(rec['certificate'])}", flush=True)
        else:
            print(f"  {label}: {w} span {sum(w)} phase {z}; fuses {rec['fuses_at_phase']}; "
                  f"letters {rec['letters']}", flush=True)
            for s, c in rec["subwindow_certificates"].items():
                print(f"      {kk}-subwindow ({s}) span "
                      f"{sum(int(x) for x in s.split(','))}: {say(c)}", flush=True)
            print(f"      the word itself: {say(rec['word_itself'])}", flush=True)
        with open(os.path.join(OUT, f"verify_{y}_{q}_k{k}.json"), "w") as f:
            json.dump(out, f, indent=1)

    # ---------------------------------------------------------------- 2. exactness audit
    D1 = list(range(1, FM + 1))
    path = os.path.join(OUT, f"dict_m{y}.json")
    dd = json.load(open(path)) if os.path.exists(path) else {}
    if dd.get("D1"):
        D1 = [int(v) for v in dd["D1"]]
    D2 = set((a, b) for a in D1 for b in D1 if a + b <= cap2)
    Jmax = rep["J_max"]
    M = memo()
    audit = {}
    for J in range(k + 1, Jmax + 1):
        t0 = time.time()
        words = enumerate_fusing(J, D1, D2, q)
        above = [w for w in words if sum(w) > floor]
        surv, unknown = [], []
        for w in above:
            subs = subwindows(w, k)
            if any(sum(t) > (CAPS.get(k) or 10 ** 9) for t in subs):
                continue                      # certified cap: not realised, no solver call
            v = [M.get(t) for t in subs]
            if any(x is False for x in v):
                continue                      # refuted, correctly skipped by the scan
            (surv if all(x is True for x in v) else unknown).append(w)
        audit[J] = {"fusing_words": len(words), "above_floor": len(above),
                    "n_admissible_above_floor": len(surv),
                    "admissible_above_floor": [list(w) for w in surv[:20]],
                    "undecided_above_floor": len(unknown),
                    "undecided_examples": [list(w) for w in unknown[:20]],
                    "secs": round(time.time() - t0, 1)}
        print(f"  audit J={J}: {len(words):,} fusing words, {len(above):,} above {floor}; "
              f"{len(surv)} level-{k} admissible (must be 0), {len(unknown)} undecided",
              flush=True)
        if unknown and resolve:
            need = sorted({t for w in unknown for t in subwindows(w, k) if M.get(t) is None})
            print(f"    resolving {len(need):,} undecided {k}-windows with the full solver",
                  flush=True)
            decide_many(need, gears, procs=procs)
            save_memo(os.path.join(OUT, f"memo_m{y}.json"))
            surv2, unk2 = [], []
            for w in unknown:
                v = [M.get(t) for t in subwindows(w, k)]
                if any(x is False for x in v):
                    continue
                (surv2 if all(x is True for x in v) else unk2).append(w)
            audit[J]["after_resolve"] = {"admissible": len(surv2), "still_undecided": len(unk2),
                                         "admissible_words": [list(w) for w in surv2[:20]],
                                         "undecided_words": [list(w) for w in unk2[:20]]}
            print(f"    after resolve: {len(surv2)} admissible, {len(unk2)} still undecided",
                  flush=True)
        del words, above
        out["audit"] = {str(j): audit[j] for j in audit}
        with open(os.path.join(OUT, f"verify_{y}_{q}_k{k}.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(f"written: verify_{y}_{q}_k{k}.json", flush=True)


if __name__ == "__main__":
    main()

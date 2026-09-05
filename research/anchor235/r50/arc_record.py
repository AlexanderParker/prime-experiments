"""Branch 5d.ii.i.a, item 5: the record law's LETTERS are the arcs of the new gear,
and the increment at twin rungs.

For each rung q -> q' the record of M' = {5..q'} is a run of L = F(M') - 1 struck
columns.  A witness cover of that run assigns one phase to each gear; the columns of
the run that NO gear of M = {5..q} strikes are exactly the interior openings of M that
q' has to strike, and the differences between consecutive such columns are the letters
of the word (docs/proofs/05 (F): a = 2 u_{q'} = a_{q'} and b = q' - a_{q'}).  This
script extracts them.  The word grammar itself is a recorded theorem, cited not
re-derived; what is measured here is WHICH letter each rung's record uses and how the
flanks sit around it.

Usage: uv run python research/anchor235/r50/arc_record.py [QMAX]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arc_core import arc, F_of, primes_upto, RESULTS  # noqa: E402

FLADDER = {5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
           37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}


def cover_witness(L, gears):
    """One phase assignment covering 0..L-1, or None.  Same search as r48
    cover_core, with the phases kept."""
    gears = sorted(gears)
    n = len(gears)
    full = (1 << L) - 1
    masks, dsep, cap = [], [], []
    for g in gears:
        d = pow(3, -1, g)
        dsep.append(d)
        ms = []
        for o in range(g):
            m = 0
            for i in range(o, L, g):
                m |= 1 << i
            for i in range((o + d) % g, L, g):
                m |= 1 << i
            ms.append(m)
        masks.append(ms)
        cap.append(max(bin(m).count("1") for m in ms))
    fail = set()
    chosen = {}

    def search(covered, avail):
        if covered == full:
            return True
        key = (covered, avail)
        if key in fail:
            return False
        u = ~covered & full
        todo = bin(u).count("1")
        tot, a = 0, avail
        while a:
            b = a & -a
            tot += cap[b.bit_length() - 1]
            a ^= b
        if tot < todo:
            fail.add(key)
            return False
        pos = (u & -u).bit_length() - 1
        a = avail
        while a:
            b = a & -a
            i = b.bit_length() - 1
            a ^= b
            g, d = gears[i], dsep[i]
            for o in {pos % g, (pos - d) % g}:
                if search(covered | masks[i][o], avail ^ b):
                    chosen[gears[i]] = o
                    return True
        fail.add(key)
        return False

    return dict(chosen) if search(0, (1 << n) - 1) else None


def main():
    QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "arc_record.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    ps = [p for p in primes_upto(QMAX) if p >= 5]
    rows = []
    say("=== the record word at each rung: flanks and letters ===")
    say("  q'  a_q'  b_q'   F   word (M-gaps of the record run)          "
        "letters (differences of the struck openings)")
    for i in range(1, len(ps)):
        q, qp = ps[i - 1], ps[i]
        M = ps[:i]
        Mp = ps[:i + 1]
        F = FLADDER[qp]
        L = F - 1
        t = time.time()
        ch = cover_witness(L, Mp)
        if ch is None:
            say(f"  {qp}: no cover at L={L} - ladder value wrong?")
            continue
        # columns struck by no gear of M
        struckM = 0
        for g in M:
            d = pow(3, -1, g)
            o = ch[g]
            for x in range(o % g, L, g):
                struckM |= 1 << x
            for x in range((o + d) % g, L, g):
                struckM |= 1 << x
        interior = [x for x in range(L) if not (struckM >> x & 1)]
        pts = [-1] + interior + [L]
        word = [pts[j + 1] - pts[j] for j in range(len(pts) - 1)]
        letters = [interior[j + 1] - interior[j] for j in range(len(interior) - 1)]
        a = arc(qp)
        b = qp - a
        say(f"  {qp:2d}   {a:3d}  {b:4d}  {F:3d}   {str(word):38s} "
            f"{str(letters):24s}  [{time.time()-t:.0f}s]")
        rows.append({"q": q, "qp": qp, "a": a, "b": b, "F": F, "word": word,
                     "letters": letters,
                     "letters_are_arc": all(x % qp in (a % qp, b % qp)
                                            for x in letters)})

    say("")
    say("=== increments, twin rungs against the rest ===")
    say("   q -> q'   twin?   F(q)  F(q')  increment  inc/q'   a_q'  new arc?")
    inc = []
    for i in range(1, len(ps)):
        q, qp = ps[i - 1], ps[i]
        tw = (qp - q == 2)
        d = FLADDER[qp] - FLADDER[q]
        newarc = arc(qp) != arc(q)
        say(f"  {q:3d} -> {qp:3d}   {'TWIN' if tw else '    '}   "
            f"{FLADDER[q]:4d}  {FLADDER[qp]:5d}   {d:6d}    {d/qp:6.3f}  "
            f"{arc(qp):4d}   {'yes' if newarc else 'NO (shared with q)'}")
        inc.append({"q": q, "qp": qp, "twin": tw, "inc": d, "ratio": d / qp,
                    "newarc": newarc})
    tw = [r["ratio"] for r in inc if r["twin"]]
    nt = [r["ratio"] for r in inc if not r["twin"]]
    say(f"  twin rungs   inc/q': {['%.3f' % x for x in tw]}   "
        f"min {min(tw):.3f} max {max(tw):.3f} mean {sum(tw)/len(tw):.3f}")
    say(f"  other rungs  inc/q': {['%.3f' % x for x in nt]}   "
        f"min {min(nt):.3f} max {max(nt):.3f} mean {sum(nt)/len(nt):.3f}")
    say(f"  separated? {'YES' if max(tw) < min(nt) or max(nt) < min(tw) else 'NO'}")

    json.dump({"words": rows, "increments": inc},
              open(os.path.join(RESULTS, "arc_record.json"), "w"))
    log.close()


if __name__ == "__main__":
    main()

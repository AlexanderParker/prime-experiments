"""Round 24 (mechanic): A_kill(M -> q') - the kill-chain (fuel) arity cap,
decided by CRT+SAT word refutation at machines beyond every scan.

WHY.  Round 23 showed the WORD-FREE criterion max_J Q_J(M; 2u') <= F(M) + q'
FAILS at 43->47 (152 vs 150) and 47->53 (177 vs 171), and that both failures
live at depths J = 6, 7 ONLY.  The merge law never needs a depth above
k_max + 1, where k_max = A_kill(M -> q') is the longest chain of consecutive
M-openings that ONE phase of gear q' can delete.  So a proven k_max <= 4 at
those two steps restores the criterion.  k_max is 3 at both steps below
(37->41 and 41->43, r20).

THE OBJECT.  A k-chain is k CONSECUTIVE openings of M all lying in the two
teeth {c - u', c + u'} of gear q' for one phase c.  Writing the k-1
consecutive gaps as a word, each gap must satisfy

    v mod q'  in  V = {0, +s, -s},   s = 2 u' mod q',  u' = 6^{-1} mod q'

(letter 0 / +1 / -1) and the letter word's PREFIX SUMS must have range <= 1
(the two teeth are one apart in the +-s lattice).  That is exactly
fuel_census.py's window condition, and the T3 alternation law in Constructor's
two-teeth-kill-spacing.md.  N_k = # realised such k-tuples per period;
k_max = max{k : N_k > 0}.

THE METHOD.  For each legal word, "is it realised?" is a pure CRT+SAT
question (cov_count.count_pattern): the word's openings exposed, every other
interior slot blocked, one free phase per gear.  No period scan - machine 43's
period is 2.18e15 and machine 47's is 1.02e17.

PRUNES, all theorems:
  * spectrum:   any contiguous t-block of the word has span <= F_t(M).
                F_2(M) <= F(M + q') and F_3(M) <= F(M + q' + q'') come free
                from the DELETION-LADDER BOUND F_{r+1}(M) <= F(M + r gears)
                (old-machine-spectrum.md Corollary B) plus the corpus ladder.
  * holes:      a gap value that is a hole of M cannot appear (holes only
                prune, so an INCOMPLETE hole list is sound).
  * overlap:    (Constructor R45) a realised m-word has every contiguous
                (m-1)-sub-word realised, so level m only tests words whose
                sub-words survived level m-1.

Usage:
  uv run --with python-sat python research/a_kill.py validate
  uv run --with python-sat python research/a_kill.py run 43 47 --kmax 4
  uv run --with python-sat python research/a_kill.py run 47 53 --kmax 4
Options: --decide (cap counts at 1: only 0 vs >0, much faster),
         --cap N, --caps "103,118,145" (F_1,F_2,F_3,... upper bounds).
"""
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cov_count import count_pattern            # noqa: E402
from cov_sat import MEASURED_HOLES             # noqa: E402

# Exact F(M) (corpus ladder, complete to 53 after r23's F(47) = 118).
F_EXACT = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58,
           37: 88, 41: 91, 43: 103, 47: 118, 53: 145}

# Known hole lists.  INCOMPLETE lists are sound (holes only prune).
HOLES = dict(MEASURED_HOLES)
HOLES[41] = [84, 87, 89]          # r20 COV-SAT, full spectrum at m41
HOLES[43] = [102]                 # r21: one hole observed below F(43) = 103
HOLES[47] = []                    # no hole list computed

# F_t(M) upper bounds.  F_1 exact; F_2, F_3 from the deletion-ladder bound
# F_{r+1}(M) <= F(M + r more gears) with the corpus F ladder.
DEFAULT_CAPS = {
    37: [88, 90, 97, 105, 113, 120],        # exact full-period spectrum (r21)
    # F_2(41)=103, F_3(41)=110 (r24 exact), F_4(41)=118 (r27 exact, floor-1
    # lap-phase transfer at cap 150 > the deletion-ladder cap 145, so the
    # value is NOT span-capped)
    41: [91, 103, 110, 118],
    43: [103, 118, 145],                    # F_2<=F(47), F_3<=F(53)
    47: [118, 145, 263],                    # F_2<=F(53); F_3<=F_2+F_1
    # ROUND 27, step 53 -> 59.  The corpus F ladder STOPS at 53, so the
    # deletion-ladder cap F_2(53) <= F(59) is unavailable and this lane's own
    # F_2(53) = 159 (C30) has a `<=` side that is CONDITIONAL on a span cap.
    # A verdict must not inherit that condition, so the caps here are the
    # UNCONDITIONAL ones F_2 <= 2 F_1 and F_3 <= 3 F_1.  Measured price of
    # unconditionality (research/akill_53_59_plan_r27.py): the k=3 level goes
    # from 6 reverse classes to 15.  Deeper levels are pruned adaptively by
    # the overlap lemma from the REALISED words actually found, so the
    # worst-case counts in that script are not the campaign's cost.
    53: [145, 290, 435],
}


def letters_of(word, qp, s):
    """Letter (0/+1/-1) per gap; None if some gap is not in V."""
    out = []
    for v in word:
        r = v % qp
        if r == 0:
            out.append(0)
        elif r == s:
            out.append(1)
        elif r == (-s) % qp:
            out.append(-1)
        else:
            return None
    return out


def window_valid(letters):
    """Prefix-sum range <= 1 (the two teeth are one step apart)."""
    p = 0
    lo = hi = 0
    for L in letters:
        p += L
        lo = min(lo, p)
        hi = max(hi, p)
    return hi - lo <= 1


def span_ok(word, caps):
    """Every contiguous t-block has span <= caps[t-1]."""
    m = len(word)
    for t in range(1, m + 1):
        if t > len(caps):
            continue
        cap = caps[t - 1]
        for i in range(0, m - t + 1):
            if sum(word[i:i + t]) > cap:
                return False
    return True


def legal_values(y, qp):
    s = (2 * pow(6, -1, qp)) % qp
    V = {0, s, (-s) % qp}
    F = F_EXACT[y]
    holes = set(HOLES.get(y, []))
    vals = [v for v in range(1, F + 1) if v % qp in V and v not in holes]
    return s, sorted(V), vals


def enumerate_words(y, qp, nlet, caps):
    """All legal nlet-letter words (residue + window + span prunes)."""
    s, V, vals = legal_values(y, qp)
    out = []
    for w in product(vals, repeat=nlet):
        L = letters_of(w, qp, s)
        if L is None or not window_valid(L):
            continue
        if not span_ok(w, caps):
            continue
        out.append(w)
    return s, V, vals, out


def decide_level(y, words, cap, log):
    """Return {word: count}.  cap=1 => decision only (0 or >=1)."""
    res = {}
    for w in words:
        opens = []
        acc = 0
        for g in w[:-1]:
            acc += g
            opens.append(acc)
        S = sum(w)
        t0 = time.time()
        n, wits, calls = count_pattern(y, S, tuple(opens), cap=cap)
        res[w] = n
        tag = ("REALISED" + (" (>=cap)" if n >= cap else f" count={n}")
               if n else "zero")
        log(f"    word {w} span {S}: {tag} "
            f"({calls} calls, {time.time()-t0:.1f}s)"
            + (f" wit {wits[0]}" if wits else ""))
    return res


def sub_ok(w, prev):
    """Overlap lemma: every contiguous (len-1)-sub-word must be realised."""
    m = len(w)
    for i in range(m):
        sub = w[:i] + w[i + 1:]
        if i not in (0, m - 1):
            continue                       # only contiguous sub-words
        if prev.get(sub, 0) == 0:
            return False
    return True


def run(y, qp, kmax, cap, caps, logf=None):
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    s, V, vals = legal_values(y, qp)
    log(f"A_kill({y} -> {qp}): u'={pow(6,-1,qp)%qp} s={s} V={V} mod {qp}, "
        f"F({y})={F_EXACT[y]}, holes {sorted(HOLES.get(y, []))}")
    log(f"  legal gap values (v mod {qp} in V, v <= F, not a hole): {vals}")
    log(f"  span caps F_1..F_{len(caps)}: {caps}")
    prev = None
    totals = {}
    for k in range(3, kmax + 2):          # k-tuple <-> (k-1)-letter word
        nlet = k - 1
        s_, V_, vals_, words = enumerate_words(y, qp, nlet, caps)
        if prev is not None:
            words = [w for w in words if sub_ok(w, prev)]
        log(f"  === k={k} ({nlet}-letter words): {len(words)} to decide "
            f"after residue+window+span+overlap prunes ===")
        t0 = time.time()
        res = decide_level(y, words, cap, log)
        nz = {w: n for w, n in res.items() if n}
        tot = sum(nz.values())
        totals[k] = (len(words), len(nz), tot)
        log(f"  N_{k}({y}->{qp}): {len(nz)} realised words of {len(words)} "
            f"tested, total count {tot}"
            + (" (counts capped)" if cap and any(n >= cap
                                                 for n in nz.values())
               else " EXACT")
            + f"   [{time.time()-t0:.0f}s]")
        for w, n in sorted(nz.items()):
            log(f"      realised: {w} count {n}")
        prev = res
        if not nz:
            log(f"  ==> N_{k} = 0, so A_kill({y}->{qp}) = k_max <= {k-1}")
            break
    return totals, lines


ANCHOR_NOTE = """
ANCHORS (full-period fuel scan, machine 37, r20/r21 - see the round-24
correction: the published N_k were the THIRD RANGE only; the true
full-period values are the sums over the three chained runs):
    N_3(37->41) = 300 + 1173 + 1579 = 3052      (up to <= 2 tuples lost at
                                                 each of the two junctions)
    N_4(37->41) = 0                             (k_max = 3)
"""


def validate(cap=100000):
    print(ANCHOR_NOTE)
    caps = DEFAULT_CAPS[37]
    totals, _ = run(37, 41, 4, cap, caps)
    n3 = totals[3][2]
    n4 = totals.get(4, (0, 0, 0))[2]
    print(f"\n  SAT enumeration: N_3(37->41) = {n3}  vs scan 3052 "
          f"(diff {n3 - 3052}); N_4 = {n4} vs scan 0")
    ok = (n4 == 0) and abs(n3 - 3052) <= 4
    print("=> " + ("ANCHORS AGREE" if ok else "MISMATCH - do not use"))
    return ok


def main():
    args = sys.argv[1:]

    def popopt(name, default=None, cast=str):
        if name in args:
            i = args.index(name)
            v = cast(args[i + 1])
            del args[i:i + 2]
            return v
        return default

    decide = "--decide" in args
    if decide:
        args.remove("--decide")
    cap = popopt("--cap", 1 if decide else 100000, int)
    capstr = popopt("--caps", None, str)
    kmax = popopt("--kmax", 4, int)
    logp = popopt("--log", None, str)
    if not args:
        print(__doc__)
        return
    if args[0] == "validate":
        validate()
        return
    y, qp = int(args[1]), int(args[2])
    caps = ([int(x) for x in capstr.split(",")] if capstr
            else DEFAULT_CAPS[y])
    f = open(logp, "a") if logp else None
    run(y, qp, kmax, cap, caps, logf=f)
    if f:
        f.close()


if __name__ == "__main__":
    main()

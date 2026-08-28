"""Round 25 (mechanic) GATE: A_kill(47->53) = 5, and the criterion repair FAILS.

Round 24 left the 47->53 fuel cap open and recorded "nothing seen contradicts
k_max = 3".  Round 24's orchestrator then ran to completion unattended and its
log carries realised words at k=4 AND k=5 - the first 5-chain anywhere in the
project.  That overturns a standing expectation, so standing rule 15 applies:
an anchor on the answer does not test the predicate.  EVERYTHING below is
re-derived here from the DEFINITION in plain integer arithmetic - no SAT for
the witnesses, no numpy, no import of a_kill.py / cov_count.py / j5_multi.py
except for the two k=6 refutations, which are genuine UNSAT/structural facts
and are re-run rather than re-read.

WHAT IS ASSERTED (five parts):

  A. The ten k=4 / k=5 realised witnesses.  A k-chain at M -> q' is k
     CONSECUTIVE openings of machine M that ONE phase of gear q' deletes.  For
     each claimed address k0 and gap word:
       (1) OCCURRENCE - k0 + p_i are openings of machine M and EVERY other
           slot of the span is blocked (gear-by-gear modular arithmetic).
       (2) KILLABILITY - some residue r mod q' puts every p_i on a tooth
           {+u', -u'} of gear q', u' = 6^{-1} mod q'.
       (3) JOINT REALISABILITY - gcd(P(M), q') = 1, so k* = CRT(k0, r) is an
           address where occurrence AND kill both hold; re-verified there from
           scratch.
  B. The k=6 candidate list.  Word enumeration (residue legality, prefix-sum
     window validity, span caps, overlap lemma) re-implemented here and
     asserted to be EXACTLY the two words the round-25 run decided.
  C. Both k=6 words are ZERO (re-decided, not re-read).  One is refuted with
     no SAT call at all, by a fact this gate proves separately: gear 5 has no
     admissible phase, i.e. every residue mod 5 is forbidden by the exposed
     set.  => N_6 = 0 => A_kill(47->53) = 5 EXACT.
  D. Q_6(47; 18) >= 174 at machine 47, from the r=6 lap-phase transfer's own
     witness, translated to a machine-47 address by CRT and checked directly:
     7 openings, every interior slot blocked, every middle gap >= 18.
  E. THE CONSEQUENCE.  The merge law consumes depths j <= k_max + 1 = 6.
     Q_6 = 174 > 171 = F(47) + 53.  The depth-capped word-free criterion is
     NOT restored at 47->53 - the restoration threshold was k_max <= 4 and the
     truth is 5.  (D) at 47->53 itself is UNAFFECTED and still true by
     arithmetic: F(53) = 145 <= 171.

usage: <venv-sat>/python research/akill_verify_r25.py
       <venv>/python research/akill_verify_r25.py --nosat   (skip part C)
"""
import sys
from itertools import product
from math import prod

# ------------------------------------------------------------------ machine

def primes_upto(n):
    s = [True] * (n + 1)
    s[0] = s[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            for j in range(i * i, n + 1, i):
                s[j] = False
    return [i for i in range(n + 1) if s[i]]


def gears(y):
    return [p for p in primes_upto(y) if p >= 5]


def is_open(k, gs):
    """Slot k survives the machine with gear set gs (gear q blocks +-6^-1)."""
    for q in gs:
        u = pow(6, -1, q)
        r = k % q
        if r == u % q or r == (-u) % q:
            return False
    return True


def positions(word):
    pos = [0]
    for g in word:
        pos.append(pos[-1] + g)
    return pos


def check_occurrence(y, k0, word):
    """Consecutive openings of machine y with exactly this gap word at k0."""
    gs = gears(y)
    pos = positions(word)
    posset = set(pos)
    for t in range(pos[-1] + 1):
        o = is_open(k0 + t, gs)
        if t in posset and not o:
            return False, f"slot k0+{t} should be an OPENING but is blocked"
        if t not in posset and o:
            return False, f"slot k0+{t} should be BLOCKED but is an opening"
    return True, f"{len(pos)} consecutive m{y} openings, span {pos[-1]}"


def check_killable(qp, word):
    """Shifts r for which one phase of gear qp deletes the whole chain."""
    u = pow(6, -1, qp)
    teeth = {u % qp, (-u) % qp}
    pos = positions(word)
    return [r for r in range(qp)
            if all((r + p) % qp in teeth for p in pos)], teeth


def crt(res, mod):
    x, M = 0, 1
    for r, m in zip(res, mod):
        x += M * ((r - x) * pow(M % m, -1, m) % m)
        M *= m
    return x % M, M


# ---------------------------------------------------------- A. the witnesses

WITNESSES = [                                    # (y, qp, k0, word)
    (47, 53, 93687953862430707, (18, 35, 18)),
    (47, 53, 45325141048971367, (18, 35, 53)),
    (47, 53, 6455343315713442, (18, 53, 35)),
    (47, 53, 56130763926877505, (35, 18, 35)),
    (47, 53, 84714628477715547, (35, 18, 53)),
    (47, 53, 68482738008881507, (35, 53, 18)),
    (47, 53, 98565777296016922, (53, 18, 35)),
    (47, 53, 56423873174590617, (53, 35, 18)),
    (47, 53, 97539948280322747, (18, 35, 18, 35)),
    (47, 53, 51380738915461847, (35, 18, 35, 18)),
]


def verify_witness(y, qp, k0, word):
    P = prod(gears(y))
    ok1, msg1 = check_occurrence(y, k0, word)
    hits, teeth = check_killable(qp, word)
    ok2 = bool(hits)
    ok3, msg3, kstar = False, "not attempted", None
    if ok1 and ok2:
        kstar, _ = crt([k0 % P, hits[0]], [P, qp])
        ok3a, m3a = check_occurrence(y, kstar, word)
        u = pow(6, -1, qp)
        ok3b = all((kstar + p) % qp in {u % qp, (-u) % qp}
                   for p in positions(word))
        ok3 = ok3a and ok3b
        msg3 = (f"k* = {kstar:,}: occurrence {'OK' if ok3a else 'FAIL'}, "
                f"all {len(positions(word))} members on a tooth of {qp}: "
                f"{'OK' if ok3b else 'FAIL'}")
    print(f"  k={len(word)+1} word {word} span {sum(word)}")
    print(f"    (1) occurrence at k0 = {k0:,}: {'OK' if ok1 else 'FAIL'}"
          f" - {msg1}")
    print(f"    (2) one phase of {qp} kills all: {'OK' if ok2 else 'FAIL'}"
          f" - teeth {sorted(teeth)}, admissible shifts {hits}")
    print(f"    (3) {msg3}")
    return ok1 and ok2 and ok3


# ------------------------------------------- B. independent word enumeration

F_EXACT = {37: 88, 41: 91, 43: 103, 47: 118, 53: 145}
CAPS_47 = [118, 145, 263]          # F_1 exact; F_2 <= F(53); F_3 <= F_2 + F_1
HOLES_47 = []                      # no hole list computed at m47 (sound: only prunes)


def legal_values(y, qp):
    s = (2 * pow(6, -1, qp)) % qp
    V = {0, s, (-s) % qp}
    return s, [v for v in range(1, F_EXACT[y] + 1)
               if v % qp in V and v not in HOLES_47]


def word_legal(w, qp, s, caps):
    """Residue legality + prefix-sum window validity + span caps."""
    p = lo = hi = 0
    for v in w:
        r = v % qp
        L = 0 if r == 0 else (1 if r == s else (-1 if r == (-s) % qp else None))
        if L is None:
            return False
        p += L
        lo, hi = min(lo, p), max(hi, p)
    if hi - lo > 1:
        return False
    for t in range(1, min(len(w), len(caps)) + 1):
        for i in range(len(w) - t + 1):
            if sum(w[i:i + t]) > caps[t - 1]:
                return False
    return True


def enumerate_level(y, qp, nlet, realised_prev):
    """Legal nlet-letter words whose two contiguous sub-words are realised."""
    s, vals = legal_values(y, qp)
    out = []
    for w in product(vals, repeat=nlet):
        if not word_legal(w, qp, s, CAPS_47):
            continue
        if realised_prev is not None and (w[1:] not in realised_prev
                                          or w[:-1] not in realised_prev):
            continue
        out.append(w)
    return out


# ------------------------------------------------- C. the k=6 refutations

def gear5_forbids_all(word):
    """Exact structural refutation: gear q has NO admissible phase when every
    residue mod q is forbidden by the exposed set X (a phase a of gear q blocks
    {a, a+s}, s = -2u mod q, so a is forbidden if a == x or a == x - s for some
    exposed x).  Reported for the gear with the smallest modulus that fires."""
    X = set(positions(word))
    for q in gears(47):
        u = pow(6, -1, q)
        s = (-2 * u) % q
        forb = {x % q for x in X} | {(x - s) % q for x in X}
        if len(forb) == q:
            return q
    return None


# ------------------------------------------------------- D. the Q_6 witness
# research/data/j5_multi_23_r6.log, the r=6 lap-phase transfer:
#   J=6: k=2,970,028 span=174 phases=(12,19,10,18,34,25) marks=(1,4,8,14,24)
Q6_OLD, Q6_NEW = 23, [29, 31, 37, 41, 43, 47]
Q6_X0, Q6_SPAN, Q6_A = 2970028, 174, 18
Q6_PHASES, Q6_MARKIDX = [12, 19, 10, 18, 34, 25], [1, 4, 8, 14, 24]


def verify_q6():
    """Phase c for new gear q means the window sits in lap j with
    c = -j*P mod q, because slot x + j*P is blocked by q iff x = c +- u.
    CRT over the six new gears gives j, hence the machine-47 address."""
    old = gears(Q6_OLD)
    P = prod(old)
    assert is_open(Q6_X0, old), "x0 is not an opening of machine 23"
    assert is_open(Q6_X0 + Q6_SPAN, old), "window end is not an m23 opening"
    interior = [x for x in range(Q6_X0 + 1, Q6_X0 + Q6_SPAN)
                if is_open(x, old)]
    marks = [interior[t] - Q6_X0 for t in Q6_MARKIDX]
    js = [(-c * pow(P % q, -1, q)) % q for q, c in zip(Q6_NEW, Q6_PHASES)]
    j, _ = crt(js, Q6_NEW)
    k = Q6_X0 + j * P
    offs = [0] + marks + [Q6_SPAN]
    g47 = gears(47)
    offset_set = set(offs)
    for t in range(Q6_SPAN + 1):
        o = is_open(k + t, g47)
        assert o == (t in offset_set), \
            f"machine-47 mismatch at offset {t}: open={o}"
    gapw = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
    mids = gapw[1:-1]
    assert sum(gapw) == Q6_SPAN
    assert len(gapw) == 6, ("expected a 6-gap window", gapw)
    assert all(g >= Q6_A for g in mids), (mids, Q6_A)
    print(f"  machine-47 address k = {k:,}")
    print(f"    7 openings at k + {offs}, all {Q6_SPAN - 6} other interior "
          f"slots blocked (checked slot by slot)")
    print(f"    gaps {gapw} sum {sum(gapw)}; middles {mids} all >= {Q6_A}")
    return k, sum(gapw)


# ------------------------------------------------------------------- main

def main():
    nosat = "--nosat" in sys.argv
    print(__doc__.split("usage:")[0].strip().splitlines()[0])
    print()
    print("=== A. the ten realised k=4 / k=5 witnesses at 47->53 ===")
    bad = [w for y, qp, k0, w in WITNESSES
           if not verify_witness(y, qp, k0, w)]
    assert not bad, f"WITNESS FAILURES: {bad}"
    real4 = {w for _, _, _, w in WITNESSES if len(w) == 3}
    real5 = {w for _, _, _, w in WITNESSES if len(w) == 4}
    print(f"  10/10 verified => N_4(47->53) >= {len(real4)}, "
          f"N_5(47->53) >= {len(real5)}  => A_kill(47->53) >= 5")

    print()
    print("=== B. the k=6 candidate list, re-enumerated independently ===")
    lvl4 = enumerate_level(47, 53, 3, None)
    lvl5 = enumerate_level(47, 53, 4, real4)
    lvl6 = enumerate_level(47, 53, 5, real5)
    print(f"  legal 3-letter words (k=4): {len(lvl4)}")
    print(f"  legal 4-letter words surviving the overlap prune (k=5): "
          f"{len(lvl5)}")
    print(f"  legal 5-letter words surviving the overlap prune (k=6): "
          f"{len(lvl6)} -> {lvl6}")
    assert set(lvl6) == {(18, 35, 18, 35, 18), (35, 18, 35, 18, 35)}, lvl6
    assert real4 <= set(lvl4) and real5 <= set(lvl5), "realised word not legal"
    print("  matches the decided list exactly (asserted)")

    print()
    print("=== C. both k=6 words are ZERO ===")
    q = gear5_forbids_all((18, 35, 18, 35, 18))
    assert q == 5, q
    print(f"  (18, 35, 18, 35, 18): ZERO BY THEOREM - gear {q} has no "
          f"admissible phase (every residue mod {q} is forbidden by the "
          f"exposed set), so some exposed slot is blocked.  No SAT needed.")
    if nosat:
        print("  (35, 18, 35, 18, 35): SKIPPED (--nosat); "
              "run under the SAT venv to re-decide")
    else:
        sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
        from cov_count import count_pattern
        w = (35, 18, 35, 18, 35)
        n, wits, calls = count_pattern(47, sum(w), tuple(positions(w)[1:-1]),
                                       cap=1)
        print(f"  {w}: count {n} ({calls} SAT call(s)) - "
              f"{'ZERO' if n == 0 else 'REALISED ' + str(wits)}")
        assert n == 0, (w, n, wits)
    print("  => N_6(47->53) = 0  =>  A_kill(47->53) = 5 EXACT")

    print()
    print("=== D. Q_6(47; 18) >= 174 at machine 47 ===")
    kq, span = verify_q6()
    assert span == 174

    print()
    print("=== E. the consequence ===")
    budget = F_EXACT[47] + 53
    depth = 5 + 1
    print(f"  merge law consumes depths j <= k_max + 1 = {depth}")
    print(f"  Q_{depth}(47; 18) >= {span}  vs budget F(47) + 53 = {budget}"
          f"  ->  EXCEEDS by {span - budget}")
    assert span > budget
    print("  => the DEPTH-CAPPED word-free criterion is NOT restored at "
          "47->53.")
    print("     The round-24 restoration threshold was k_max <= 4; the exact "
          "value is 5.")
    print(f"  (D) at 47->53 is untouched and true: F(53) = 145 <= {budget}.")
    print()
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

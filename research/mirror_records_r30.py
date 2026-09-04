"""
LATERAL round 30, BLOCK C - THE MIRROR AS AN EXACT SYMMETRY OF EVERY RECORD ON FILE.

STATEMENT.  Machine y, gears the primes 5..y, period P = prod q, mirror k -> -k.
A window is (address k, span s, offsets 0 = o_0 < o_1 < ... < o_J = s) with the
o_i the openings in [k, k+s] and every other slot of the span blocked.  Then
    k' = (P - k - s) mod P
is an opening, the openings in [k', k'+s] are EXACTLY the reversed offsets
s - o_J < ... < s - o_0, the two flanks reverse (gap below k' = gap above k+s and
gap above k'+s = gap below k), and for any gear q'' not dividing P the residue
of an interior opening maps r -> (P - r) mod q''.  The pair's addresses sum to
P - s exactly.  Proof: k blocked by q iff k = +-u_q (mod q); the tooth pair is
closed under negation and q | P, so k + t is open iff P - k - t is open. []

IN TRANSFER COORDINATES (a window of machine y0, period P0, lifted by phases c_q
of the new gears q, where slot k+t of the lift is blocked by q iff
(k + t) = c_q +- u_q mod q):  x = k + j P0 with c_q = -j P0 mod q.  The mirror
x -> P - x - s has k' = P0 - k - s and j' = P/P0 - 1 - j, hence
    c'_q = -(j' P0) = (1 + j) P0 = P0 - c_q   (mod q).

This script re-checks every exact record window on file at its own machine
from the definition, then checks its mirror partner the same way, and gates
the transfer-coordinate map on the transfer witnesses.

Usage: uv run python research/mirror_records_r30.py
"""
import json
import os
import sys
from math import prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R29 = os.path.join(HERE, "data", "r29")
NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def gears(y):
    return [p for p in range(5, y + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def teeth(y):
    return {q: (pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in gears(y)}


def is_open(k, T):
    return all(k % q not in t for q, t in T.items())


def window_at(y, k, s, T):
    """Offsets of the openings in [k, k+s] (asserting k, k+s open) and the
    two outside flanks."""
    offs = [t for t in range(s + 1) if is_open(k + t, T)]
    assert offs and offs[0] == 0 and offs[-1] == s, ("window ends not open", y, k)
    lo = k - 1
    while not is_open(lo, T):
        lo -= 1
    hi = k + s + 1
    while not is_open(hi, T):
        hi += 1
    return offs, k - lo, hi - (k + s)


def check_record(label, y, k, offs, next_gear=None):
    T = teeth(y)
    P = prod(gears(T and list(T))) if False else prod(list(T))
    s = offs[-1]
    got, below, above = window_at(y, k, s, T)
    gate(got == list(offs),
         "%s: window re-verified at machine %d from the definition (offsets %s, "
         "flanks %d/%d)" % (label, y, offs, below, above))
    kp = (P - k - s) % P
    rev = [s - o for o in reversed(offs)]
    got2, below2, above2 = window_at(y, kp, s, T)
    gate(got2 == rev,
         "%s: MIRROR slot k' = P - k - s = %d carries the REVERSED offsets %s"
         % (label, kp, rev))
    gate((below2, above2) == (above, below),
         "%s: the flanks reverse: (%d, %d) -> (%d, %d)"
         % (label, below, above, below2, above2))
    gate(kp != k, "%s: the partner is a DIFFERENT slot (k' != k)" % label)
    gate(k + kp + s == P or k + kp + s == 2 * P,
         "%s: k + k' + s = %s x P  (both addresses in [0, P))"
         % (label, (k + kp + s) // P))
    if next_gear is not None:
        q2 = next_gear
        r1 = [(k + o) % q2 for o in offs]
        r2 = [(kp + o) % q2 for o in rev]
        gate(r2 == [(P - r) % q2 for r in reversed(r1)],
             "%s: interior residues mod %d map r -> P - r (%s -> %s)"
             % (label, q2, r1, r2))
    return kp


def crt(res, mod):
    x, M = 0, 1
    for r, m in zip(res, mod):
        x += M * (((r - x) * pow(M % m, -1, m)) % m)
        M *= m
    return x % M, M


def lift(k, P0, NEW, phases):
    """Transfer coordinates (k, phases) -> full slot x."""
    js = [(-c * pow(P0 % q, -1, q)) % q for q, c in zip(NEW, phases)]
    j, M = crt(js, NEW)
    return (k + j * P0) % (P0 * M)


def phases_of(x, P0, NEW):
    k = x % P0
    j = x // P0
    return k, tuple((-j * P0) % q for q in NEW)


def main():
    print("ROUND-30 LATERAL - the mirror on every exact record on file\n")
    records = []
    # Mechanic's round-29 CRT slots (crt_slots_r29.py)
    records += [
        ("F_2(41)=103", 41, 21157523372970, [0, 28, 103]),
        ("F_2(53)=159", 53, 327666424664536738, [0, 77, 159]),
        ("F_2(59)=173 A", 59, 307199471342884027665, [0, 100, 173]),
        ("F_2(59)=173 B", 59, 13260587016151412007, [0, 73, 173]),
    ]
    # Mechanic's round-28 witnesses (witness_gate_r28.py)
    records += [
        ("F(59)>=161 @m53", 53, 2505673933219103747, [0, 10, 128, 161]),
        ("F_2(43)=116", 43, 2161962392309552, [0, 31, 116]),
        ("F_3(43)=125", 43, 1595441702157105, [0, 67, 95, 125]),
        ("F_4(43)=132", 43, 280183736276020, [0, 18, 42, 50, 132]),
        ("F_5(41)=128 a", 41, 33044111735742, [0, 10, 61, 63, 113, 128]),
        ("F_5(41)=128 b", 41, 17664265518665, [0, 15, 65, 67, 118, 128]),
        ("F_3(47)=145", 47, 36068193854725102, [0, 28, 61, 145]),
    ]
    # LP thread's witness F_2(37) >= 90: phases -> slot by CRT
    w = json.load(open(os.path.join(R29, "witness_inc_37_41.json")))
    G = w["gears"]
    y37, _ = crt(w["phases"], G)
    for q, ph in zip(G, w["phases"]):
        assert y37 % q == ph
    records.append(("F_2(37)>=90 (LP)", 37, y37, w["openings"]))
    # Mechanic's F_6(47) = 177 transfer witness (witness47_r29.py)
    P23 = prod(gears(23))
    ex = np.zeros(P23, bool)
    for q in gears(23):
        u = pow(6, -1, q)
        ex[u % q::q] = True
        ex[(-u) % q::q] = True
    op23 = np.flatnonzero(~ex).astype(np.int64)
    K, PH, MARKS, SPAN = 26216680, (3, 21, 29, 26, 26, 27), (5, 10, 16, 17, 19), 177
    NEW47 = [29, 31, 37, 41, 43, 47]
    i = int(np.searchsorted(op23, K))
    j = int(np.searchsorted(op23, K + SPAN))
    assert op23[i] == K and op23[j] == K + SPAN
    interior = [int(v) - K for v in op23[i + 1:j]]
    offs47 = [0] + [interior[m] for m in MARKS] + [SPAN]
    x47 = lift(K, P23, NEW47, PH)
    records.append(("F_6(47)=177", 47, x47, offs47))
    # Mechanic's record survivors at the target machine (chain_*.json)
    chain_recs = []
    for g in (31, 37, 41):
        d = json.load(open(os.path.join(R29, "chain_%d.json" % g)))
        for Lk, b in sorted(d["best"].items()):
            val = b["before"] + b["span"] + b["after"]
            assert val == b["value"]
            chain_recs.append(("chain_%d L=%s value=%d" % (g, Lk, val), g,
                               int(b["slot"]), [0, val]))
    records += chain_recs

    next_of = {31: 37, 37: 41, 41: 43, 43: 47, 47: 53, 53: 59, 59: 61}
    partners = {}
    for label, y, k, offs in records:
        print("\n--- %s : machine %d, k = %d, span %d, J = %d" %
              (label, y, k, offs[-1], len(offs) - 1))
        partners[label] = check_record(label, y, k, offs, next_of.get(y))

    print("\n=== C2: the recorded pairs are single mirror orbits ===")
    P59 = prod(gears(59))
    gate(partners["F_2(59)=173 A"] == 13260587016151412007,
         "F_2(59): witness B IS witness A's mirror, k_B = P(59) - k_A - 173")
    gate(partners["F_5(41)=128 a"] == 17664265518665,
         "F_5(41): witness b IS witness a's mirror in MACHINE-41 coordinates")

    print("\n=== C3: the mirror in TRANSFER coordinates ===")
    P47 = prod(gears(47))
    Kp = P23 - K - SPAN
    PHp = tuple((P23 - c) % q for q, c in zip(NEW47, PH))
    xp = lift(Kp, P23, NEW47, PHp)
    gate(xp == (P47 - x47 - SPAN) % P47,
         "F_6(47): (k, c) -> (P23 - k - s, P23 - c mod q) lifts to the mirror "
         "slot: k' = %d, phases %s -> x' = %d" % (Kp, PHp, xp))
    # the F_5(41) pair in transfer coordinates from machine 23
    NEW41 = [29, 31, 37, 41]
    ka, pa = phases_of(33044111735742, P23, NEW41)
    kb, pb = phases_of(17664265518665, P23, NEW41)
    gate(kb == P23 - ka - 128 and pb == tuple((P23 - c) % q for q, c in zip(NEW41, pa)),
         "F_5(41): the pair's transfer coordinates are (k, c) and "
         "(P23 - k - s, P23 - c): k = %d/%d, c = %s/%s" % (ka, kb, pa, pb))
    gate(ka == 4834937 and kb == 32347080,
         "F_5(41): machine-23 starts are round 28's 4,834,937 and 32,347,080")
    # the F_2(59) pair in transfer coordinates from machine 23 (8 new gears)
    NEW59 = [29, 31, 37, 41, 43, 47, 53, 59]
    ka, pa = phases_of(307199471342884027665, P23, NEW59)
    kb, pb = phases_of(13260587016151412007, P23, NEW59)
    gate(kb == P23 - ka - 173 and pb == tuple((P23 - c) % q for q, c in zip(NEW59, pa)),
         "F_2(59): the pair's transfer coordinates are mirrored the same way: "
         "k = %d/%d, phases %s / %s" % (ka, kb, pa, pb))
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

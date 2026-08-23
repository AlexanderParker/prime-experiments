"""Merge-law h_2 ladder test: frame mapping, scanner validation, assertions.

Companion to docs/novel/merge-law-h2-test.md and rust3/src/bin/h2ladder.rs.
This file holds the exact frame mapping and the correctness battery; the heavy
exact runs (h_2(19), h_2(23), twin rungs 31..43) live in the rust bin:

    cargo build --release --manifest-path rust3/Cargo.toml
    rust3/target/release/h2ladder twin | h2_17 | h2_19 | sample19 200 | h2_23
    rust3/target/release/h2ladder twin37 | twin41 | twin43

THE MAPPING (extracted from research/jacobsthal_family.py, which reproduced
Ziller-Morack's 18, 30, 66, 150, 192):

  * ZM frame: integers, modulus p_n# (2 and 3 included), pairs (m, m+2e') for
    ANY even difference 2e'. h_2(n) = j_2(p_n#).
  * Halved frame: m = 2n+1 absorbs gear 2 (factor 2 on all lengths). Gear
    q in {3, 5, ..., y} blocks position n = 0 and n = -e (mod q), where e is
    the reduced (halved) difference.  h_2 = 2 * max over e of F_e(y).
  * h_2 is the max over ALL difference classes e, NOT the twin class: the twin
    difference (e = 1) sits at the 13th-21st percentile of its family
    (harvester round 17).  2*F_1(13) = 66 while h_2(13) = 150.
  * Twin slot frame: for 3 not dividing e... for e = 1 gear 3 confines
    survivors to one class mod 3, so F_1 = F_adjacent = 3 * F_slot, where the
    slot frame is slot k = (6k-1, 6k+1), gears 5..y, gear q blocks
    k = +-u (mod q), 6u = 1 (mod q).  F(2,y) of the corpus is F_adjacent.

THE MERGE LAW, one scanner for both frames: adding a coprime gear q' deletes,
per lap, the openings whose positions lie in a two-element residue set
{c, c+s} mod q'; the deleted pair shifts per lap and every c occurs
(gcd(P, q') = 1), so a run of consecutive openings is deleted together in some
lap IFF its positions all lie in such a set.  s is the tooth separation:
e mod q' in the halved frame (teeth {0, -e}), 2u mod q' in the slot frame
(teeth {u, -u}).  F(M+q') = max over plain old gaps and merged run spans -
read off the OLD word alone, new period never constructed.
"""

import sys
import time
from math import prod, gcd

import numpy as np

# h_2(p_n) for p_n = 2..73: Ziller-Morack arXiv:1706.03668 Table 1 (verbatim
# from docs/novel/paired-jacobsthal-values.md section 6).
ZM = {2: 2, 3: 6, 5: 18, 7: 30, 11: 66, 13: 150, 17: 192, 19: 258, 23: 366,
      29: 450, 31: 570, 37: 708, 41: 894, 43: 1044, 47: 1284, 53: 1422,
      59: 1656, 61: 1902, 67: 2190, 71: 2460, 73: 2622}

# Corpus fixed-twin ladder, adjacent frame (= 3 * slot frame).
F2Y = {7: 15, 11: 21, 13: 33, 17: 54, 19: 75, 23: 102, 29: 129, 31: 174,
       37: 264, 41: 273, 43: 309}


def odd_primes(lo, hi):
    return [p for p in range(lo, hi + 1)
            if p > 1 and all(p % k for k in range(2, int(p**0.5) + 1))]


# ---------------------------------------------------------------------------
# Frames.
# ---------------------------------------------------------------------------

def h2_word(gears, e, P):
    """Halved-frame survivors for reduced difference e (jacobsthal_family.py)."""
    a = np.ones(P, bool)
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
    return np.flatnonzero(a)


def F_of(gears, e, P):
    idx = h2_word(gears, e, P)
    if idx.size < 2:
        return 0
    return int(np.diff(np.append(idx, idx[0] + P)).max())


def slot_word(y):
    """Slot-frame twin openings: gears 5..y, teeth +-6^{-1} mod q."""
    gears = odd_primes(5, y)
    P = prod(gears)
    a = np.ones(P, bool)
    for q in gears:
        u = pow(6, -1, q)
        a[u % q::q] = False
        a[(-u) % q::q] = False
    return np.flatnonzero(a), P


def max_cyclic_gap(idx, P):
    return int(np.diff(np.append(idx, idx[0] + P)).max())


# ---------------------------------------------------------------------------
# The merge-law scanner (mirror of ChainScan in rust3/src/bin/h2ladder.rs).
# ---------------------------------------------------------------------------

class ChainScan:
    """Exact chain condition as a streaming automaton: maximal runs of
    consecutive openings with residues inside a two-element set {c, c+s}."""

    def __init__(self, q, s):
        self.q, self.s = q, s % q
        self.best = 0
        self.started = False

    def push(self, x):
        q, s = self.q, self.s
        r = x % q
        if not self.started:
            self.started = True
            self.last_x = x
            self.val_a, self.has_b = r, False
            self.run_prev_x = x
            self.last_val, self.block_prev_x = r, x
            return
        g = x - self.last_x
        if g > self.best:
            self.best = g
        fits = (r == self.val_a
                or (self.has_b and r == self.val_b)
                or (not self.has_b
                    and (r == (self.val_a + s) % q or self.val_a == (r + s) % q)))
        if fits:
            if not self.has_b and r != self.val_a:
                self.val_b, self.has_b = r, True
            if r != self.last_val:
                self.last_val, self.block_prev_x = r, self.last_x
        else:
            span = x - self.run_prev_x
            if span > self.best:
                self.best = span
            if r == (self.last_val + s) % q or self.last_val == (r + s) % q:
                self.val_a, self.val_b, self.has_b = self.last_val, r, True
                self.run_prev_x = self.block_prev_x
            else:
                self.val_a, self.has_b = r, False
                self.run_prev_x = self.last_x
            self.last_val, self.block_prev_x = r, self.last_x
        self.last_x = x


def merge_F(positions, P, q, s, margin=256):
    """F(M + q) from the old word alone (positions sorted in [0, P))."""
    sc = ChainScan(q, s)
    for x in positions:
        sc.push(int(x))
    for x in positions[:margin]:
        sc.push(int(x) + P)
    return sc.best


# ---------------------------------------------------------------------------
# 1. Mapping: reproduce ZM h_2 exhaustively at small y, and the frame chain.
# ---------------------------------------------------------------------------

def part1_mapping():
    print("1. MAPPING - h_2 = 2 * max_e F_e (halved frame), exhaustive:")
    for y in (5, 7, 11, 13):
        gears = odd_primes(3, y)
        P = prod(gears)
        best = max(F_of(gears, e, P) for e in range(1, P // 2 + 1))
        h2 = 2 * best
        twin = F_of(gears, 1, P)
        print(f"   y={y:>2}: h_2 = {h2:>3} (ZM {ZM[y]:>3}) "
              f"{'MATCH' if h2 == ZM[y] else 'MISMATCH <-- HEADLINE'};"
              f"  twin class 2*F_1 = {2 * twin:>3} (h_2 is NOT the twin object)")
        assert h2 == ZM[y], f"h_2({y}) mismatch"
    # the twin frame chain: F_1 (halved) = F_adjacent = 3 * F_slot
    print("   twin frame chain F_1 = 3 * F_slot:")
    for y in (7, 11, 13, 17):
        gears = odd_primes(3, y)
        P = prod(gears)
        f1 = F_of(gears, 1, P)
        idx, Ps = slot_word(y)
        fs = max_cyclic_gap(idx, Ps)
        print(f"   y={y:>2}: F_1 = {f1:>3} = 3*{fs:>2}; corpus F(2,{y}) = {F2Y[y]:>3}")
        assert f1 == 3 * fs == F2Y[y]
    # the non-identity, stated numerically
    assert 2 * F_of(odd_primes(3, 13), 1, 15015) == 66 != ZM[13]
    print("   => a twin-only ladder computes F(2,y), NOT h_2; h_2 needs the whole family.\n")


# ---------------------------------------------------------------------------
# 2. Scanner battery: merge value == direct construction, both frames,
#    including degenerate teeth (q | e collapses, s = 0).
# ---------------------------------------------------------------------------

def part2_battery():
    print("2. SCANNER BATTERY - merge == direct construction:")
    cases = 0
    # halved h_2 frame
    for base in ([3, 5], [3, 5, 7], [3, 5, 7, 11]):
        P = prod(base)
        for qp in (7, 11, 13, 17, 19, 23):
            if qp in base:
                continue
            es = set(range(1, 40)) | {P, qp, 2 * qp, 3 * qp, 105, 210, P * qp // 2}
            for e in sorted(es):
                word = h2_word(base, e, P)
                if word.size < 2:
                    continue
                got = merge_F(word, P, qp, e % qp)
                want = F_of(base + [qp], e, P * qp)
                assert got == want, (
                    f"MERGE LAW WRONG (halved frame) base={base} q'={qp} e={e}: "
                    f"merge {got} vs construction {want}  <-- HEADLINE")
                cases += 1
    # twin slot frame, including the doc 4a table values
    doc_table = {(11, 13): 11, (11, 17): 16, (13, 17): 18, (13, 23): 18,
                 (17, 19): 25, (17, 29): 26, (19, 23): 34, (19, 31): 37}
    for y in (11, 13, 17, 19):
        word, P = slot_word(y)
        for qp in (13, 17, 19, 23, 29, 31, 37):
            if qp <= y:
                continue
            s = (2 * pow(6, -1, qp)) % qp
            got = merge_F(word, P, qp, s)
            # direct construction of the extended machine
            gears = odd_primes(5, y) + [qp]
            Pn = prod(gears)
            a = np.ones(Pn, bool)
            for q in gears:
                u = pow(6, -1, q)
                a[u % q::q] = False
                a[(-u) % q::q] = False
            want = max_cyclic_gap(np.flatnonzero(a), Pn)
            assert got == want, (
                f"MERGE LAW WRONG (slot frame) M({y}) + {qp}: {got} vs {want} <-- HEADLINE")
            if (y, qp) in doc_table:
                assert got == doc_table[(y, qp)], f"doc 4a table mismatch at {(y, qp)}"
            cases += 1
    print(f"   {cases} (machine, gear, class) cases, exact agreement in every one.\n")


# ---------------------------------------------------------------------------
# 3. Twin slot ladder 17 -> 29 in Python (the rust bin continues to 43).
# ---------------------------------------------------------------------------

def thin(word, P, qp):
    """Full new word: qp laps, two residue classes (the teeth) deleted per lap."""
    u = pow(6, -1, qp)
    t1, t2 = u % qp, (-u) % qp
    out = []
    for lap in range(qp):
        x = word + lap * P
        r = x % qp
        out.append(x[(r != t1) & (r != t2)])
    return np.concatenate(out), P * qp


def part3_twin_ladder():
    print("3. TWIN SLOT LADDER 17 -> 29 (merge value, then word carried forward):")
    word, P = slot_word(17)
    for qp in (19, 23, 29):
        s = (2 * pow(6, -1, qp)) % qp
        t0 = time.time()
        f = merge_F(word, P, qp, s)
        dt = time.time() - t0
        ok = 3 * f == F2Y[qp]
        print(f"   +{qp}: merge F_slot = {f} (adjacent {3 * f}), corpus {F2Y[qp]} "
              f"=> {'MATCH' if ok else 'MISMATCH <-- HEADLINE'}  ({dt:.2f}s scan)")
        assert ok
        word, P = thin(word, P, qp)
        assert max_cyclic_gap(word, P) == f, "thinned word max must equal merge value"
    print("   (rungs 31, 37, 41, 43: rust3 h2ladder twin / twin37 / twin41 / twin43)\n")


# ---------------------------------------------------------------------------
# 4. Assertions for the heavy rust results (recorded outputs of h2ladder).
# ---------------------------------------------------------------------------

RUST_RESULTS = {
    # value: (computed by rust3 h2ladder subcommand, ZM / corpus reference)
    "h_2(17) exhaustive": (192, ZM[17]),
    "h_2(19) by merge": (258, ZM[19]),
    "h_2(23) by merge, unpruned, all 2,424,932 words": (366, ZM[23]),
    "F(2,31) merge=construction": (174, F2Y[31]),
    "F(2,37) merge, period never built": (264, F2Y[37]),
    "F(2,41) merge, period never built": (273, F2Y[41]),
    # F(2,43): the merge run (h2ladder twin43, ~2-4 h idle) was terminated for
    # the round-21 machine handover; 309 stands on the covering search alone.
}


def part4_rust():
    print("4. HEAVY EXACT VALUES (rust3/src/bin/h2ladder.rs, asserted there too):")
    for name, (got, want) in RUST_RESULTS.items():
        assert got == want, f"{name}: {got} vs {want}"
        print(f"   {name}: {got} == {want}  MATCH")
    print()


# ---------------------------------------------------------------------------
# 5. Operation counts (machine-independent primary metric; mirrors
#    `h2ladder ops`, whose instrumented counters verify these closed forms).
#
# One op = one elementary visit.
#   merge path   = generation visits (lap walks producing the old word,
#                  deletions included) + scanner pushes (chain-condition
#                  checks: one letter fed to one scanner) + base-word sieve
#                  (strikes + cells) where the path sieves a base word.
#   construction = sieve strikes (teeth per gear x P/q, summed) + P cells
#                  scanned to read the gaps.
# ---------------------------------------------------------------------------

def part5_ops():
    print("5. OPERATION COUNTS (closed forms; instrumented verification in `h2ladder ops`):")
    A = lambda y: prod(q - 2 for q in odd_primes(5, y))
    P = lambda y: prod(odd_primes(5, y))
    strikes = lambda y: sum(2 * P(y) // q for q in odd_primes(5, y))
    assert A(17) == 22275 and A(29) == 214_708_725 and A(41) == 8_499_244_879_125
    rungs = {19: (strikes(17) + P(17), A(17)), 23: (19 * A(17), A(19)),
             29: (23 * A(19), A(23)), 31: (29 * A(23), A(29)),
             37: (31 * A(29), A(31)), 41: (37 * 31 * A(29), A(37)),
             43: (41 * 37 * 31 * A(29), A(41))}
    print("   twin rung: merge ops (gen + pushes) vs construction ops (strikes + P cells)")
    for qp, (gen, pushes) in rungs.items():
        con = strikes(qp) + P(qp)
        print(f"   +{qp:>2}: merge {gen + pushes:>17,} vs construction {con:>21,}"
              f"   ratio {con / (gen + pushes):7.1f}")
    # h_2 family: sum over classes c of A_c(17) = prod(q - (1 if q|c else 2))
    g17 = odd_primes(3, 17)
    P17 = prod(g17)
    sum_a = sieve17 = 0
    for c in range(P17 // 2 + 1):
        a, s = 1, 0
        for q in g17:
            k = 1 if c % q == 0 else 2
            a *= q - k
            s += k * (P17 // q)
        sum_a += a
        sieve17 += s + P17
    assert sum_a == 4_246_778_880
    m19 = sieve17 + 10 * sum_a          # 10 scanners (s and 19-s coincide)
    m23 = sieve17 + 361 * sum_a + 12 * 324 * sum_a
    P19, P23 = P17 * 19, P17 * 19 * 23
    def con_h2(Pl, gears):
        n = Pl // 2
        return sum((Pl // q) * (2 * n - n // q) for q in gears) + n * Pl
    c19, c23 = con_h2(P19, odd_primes(3, 19)), con_h2(P23, odd_primes(3, 23))
    print(f"   h_2 17->19: merge {m19:,} vs construction {c19:,} ops (ratio {c19 / m19:.1f})")
    print(f"   h_2 19->23: merge {m23:,} vs construction {c23:,} ops (ratio {c23 / m23:.1f})")
    print()


if __name__ == "__main__":
    part1_mapping()
    part2_battery()
    part3_twin_ladder()
    part4_rust()
    part5_ops()
    print("ALL ASSERTIONS PASSED.")
    sys.exit(0)

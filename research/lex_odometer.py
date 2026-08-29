"""
LATERAL round 27 - the 2n-gap reordering law: PROOF + GATES.

The claim (human's sort-step idea, manager probe, docs/novel/two-n-gap-reordering.md):
sort the machine's openings by CRT phase vector in lex order; the adjacent
differences take EXACTLY 2n distinct values at n gears.

This script proves it constructively and gates every step.

THE THEOREM (see the docstring of part A for the argument):
  Under CRT the opening set O is the product set A_1 x ... x A_n,
  A_i = Z_{q_i} minus {+-6^{-1} mod q_i}.  Lex order on phase vectors is the
  mixed-radix ODOMETER on digit vectors (j_1..j_n), j_i indexing A_i sorted.
  The lex successor increments digit i (the last non-maximal digit) and wraps
  every digit below it, so the VALUE difference is
      D(i, delta) = CRT( 0 for i'<i ; delta for i ; w_{i'} for i'>i )
  with delta = a_i^{(j_i+1)} - a_i^{(j_i)} a consecutive difference of the
  sorted set A_i and w_{i'} = a_{i'}^{(0)} - a_{i'}^{(m-1)} = -max(A_{i'}) mod q.
  Because coordinates below i are 0, the carry position i is recoverable from
  the difference, so distinct (i, delta) give distinct differences and
      #distinct differences = sum_i d_i,   d_i = #distinct consecutive
                                                differences of sorted A_i.
  For the twin machine every A_i is Z_q minus two residues that are NEVER
  adjacent (adjacency needs 2u = +-1, i.e. 3 = +-1 mod q, impossible for q>=5)
  and never 0, so the sorted differences of A_i are exactly {1, 2}: d_i = 2 and
  the count is 2n.  The cyclic closure is free (part C).

Usage:
  python lex_odometer.py --parts ABCDEF
"""
import argparse
import itertools
import sys
from math import gcd

GEARS = [5, 7, 11, 13, 17, 19, 23, 29]

NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def teeth(q):
    """slot k is blocked by q iff 6k = +-1 mod q, i.e. k = +-6^{-1}."""
    u = pow(6, -1, q)
    return (u % q, (-u) % q)


def exposed(q):
    t = set(teeth(q))
    return [a for a in range(q) if a not in t]


def crt(residues, mods):
    """residues[i] mod mods[i] -> value mod prod(mods)."""
    x, M = 0, 1
    for r, m in zip(residues, mods):
        # solve x' = x (mod M), x' = r (mod m)
        g = gcd(M, m)
        assert g == 1
        inv = pow(M, -1, m)
        k = ((r - x) * inv) % m
        x = x + M * k
        M *= m
    return x % M


def idempotents(mods):
    """E_i with E_i = 1 mod mods[i], 0 mod the others."""
    P = 1
    for m in mods:
        P *= m
    out = []
    for i, m in enumerate(mods):
        Mi = P // m
        out.append(Mi * pow(Mi, -1, m) % P)
    return out


def openings(gears):
    """natural-order (sorted) list of openings in Z_P."""
    P = 1
    for q in gears:
        P *= q
    tset = [set(teeth(q)) for q in gears]
    out = []
    for k in range(P):
        ok = True
        for q, t in zip(gears, tset):
            if k % q in t:
                ok = False
                break
        if ok:
            out.append(k)
    return out, P


def lex_sequence(gears):
    """openings sorted by phase vector (lex over the given gear order)."""
    A = [sorted(exposed(q)) for q in gears]
    mods = list(gears)
    seq = []
    for tup in itertools.product(*A):
        seq.append(crt(list(tup), mods))
    return seq


def distinct_diffs(seq, P, cyclic=True):
    d = set()
    for i in range(len(seq) - 1):
        d.add((seq[i + 1] - seq[i]) % P)
    if cyclic:
        d.add((seq[0] - seq[-1]) % P)
    return d


# ---------------------------------------------------------------- part A
def partA():
    """The 2n law, direct verification, natural (increasing) gear order."""
    print("\n=== PART A: the 2n law, increasing gear order ===")
    for n in range(1, 7):
        gears = GEARS[:n]
        seq = lex_sequence(gears)
        P = 1
        for q in gears:
            P *= q
        dl = distinct_diffs(seq, P, cyclic=False)
        dc = distinct_diffs(seq, P, cyclic=True)
        # cross-check the lex sequence is a permutation of the true opening set
        if P <= 3000000:
            nat, _ = openings(gears)
            gate(sorted(seq) == nat, "gears %s: lex sequence == opening set (N=%d)"
                 % (gears, len(nat)))
        print("    gears %-28s n=%d  linear %d  cyclic %d"
              % (gears, n, len(dl), len(dc)))
        gate(len(dl) == 2 * n, "gears %s: linear distinct diffs == 2n = %d" % (gears, 2 * n))
        gate(len(dc) == 2 * n, "gears %s: cyclic distinct diffs == 2n = %d" % (gears, 2 * n))


# ---------------------------------------------------------------- part B
def predicted_set(gears):
    """The closed-form value set and multiplicities (P2)."""
    A = [sorted(exposed(q)) for q in gears]
    mods = list(gears)
    n = len(gears)
    w = [(-a[-1]) % q for a, q in zip(A, mods)]
    m = [len(a) for a in A]
    out = {}
    for i in range(n):
        # consecutive differences of A_i
        steps = [A[i][j + 1] - A[i][j] for j in range(m[i] - 1)]
        lower = 1
        for i2 in range(i):
            lower *= m[i2]
        for delta in sorted(set(steps)):
            res = [0] * n
            res[i] = delta
            for i2 in range(i + 1, n):
                res[i2] = w[i2]
            val = crt(res, mods)
            out[val] = out.get(val, 0) + steps.count(delta) * lower
    return out


def partB():
    """Closed-form value set AND multiplicities, no enumeration of the order."""
    print("\n=== PART B: closed-form values and multiplicities ===")
    for n in range(1, 7):
        gears = GEARS[:n]
        P = 1
        for q in gears:
            P *= q
        seq = lex_sequence(gears)
        obs = {}
        for i in range(len(seq) - 1):
            d = (seq[i + 1] - seq[i]) % P
            obs[d] = obs.get(d, 0) + 1
        pred = predicted_set(gears)
        gate(set(pred) == set(obs), "gears %s: predicted VALUE SET exact" % (gears,))
        gate(pred == obs, "gears %s: predicted MULTIPLICITIES exact" % (gears,))
        # the closed forms for the step counts
        for i, q in enumerate(gears):
            A = sorted(exposed(q))
            steps = [A[j + 1] - A[j] for j in range(len(A) - 1)]
            n2 = steps.count(2)
            n1 = steps.count(1)
            gate(set(steps) == {1, 2}, "gear %d: sorted exposed steps are exactly {1,2}" % q)
            gate(n2 == (1 if q in (5, 7) else 2),
                 "gear %d: #2-steps = %d (1 at q=5,7 else 2)" % (q, n2))
            gate(n1 == q - 3 - n2, "gear %d: #1-steps = q-3-#2 = %d" % (q, n1))
        print("    gears %-28s multiplicity table verified (%d values)" % (gears, len(pred)))
        if n <= 4:
            for v in sorted(pred):
                print("        D = %-12d mult %d" % (v, pred[v]))


# ---------------------------------------------------------------- part C
def partC():
    """Cyclic closure is free: the wrap difference is already in the 2n set."""
    print("\n=== PART C: the cyclic wrap costs nothing ===")
    for n in range(1, 7):
        gears = GEARS[:n]
        P = 1
        for q in gears:
            P *= q
        seq = lex_sequence(gears)
        wrap = (seq[0] - seq[-1]) % P
        A1 = sorted(exposed(gears[0]))
        w1 = (-A1[-1]) % gears[0]
        pred = predicted_set(gears)
        res = [0] * n
        res[0] = w1
        A = [sorted(exposed(q)) for q in gears]
        for i2 in range(1, n):
            res[i2] = (-A[i2][-1]) % gears[i2]
        gate(wrap == crt(res, list(gears)),
             "gears %s: wrap difference == CRT(w_1..w_n)" % (gears,))
        gate(w1 in (1, 2), "gear %d: w_1 = -max(A_1) = %d is in {1,2}" % (gears[0], w1))
        gate(wrap in pred, "gears %s: wrap value is already one of the 2n" % (gears,))
        print("    gears %-28s w_1 = %d  ->  wrap = D(1,%d)" % (gears, w1, w1))


# ---------------------------------------------------------------- part D
def partD():
    """Order-independence (P1): the law holds for EVERY gear ordering."""
    print("\n=== PART D: order-independence over all gear permutations ===")
    for n in (2, 3, 4):
        base = GEARS[:n]
        P = 1
        for q in base:
            P *= q
        bad = 0
        for perm in itertools.permutations(base):
            seq = lex_sequence(list(perm))
            dc = distinct_diffs(seq, P, cyclic=True)
            dl = distinct_diffs(seq, P, cyclic=False)
            if len(dc) != 2 * n or len(dl) != 2 * n:
                bad += 1
        gate(bad == 0, "all %d orderings of %s give exactly 2n = %d"
             % (len(list(itertools.permutations(base))), base, 2 * n))
    # a sample at n = 5, 6 (full period 5*7*11*13*17 = 85085 etc.)
    import random
    random.seed(27)
    for n in (5, 6):
        base = GEARS[:n]
        P = 1
        for q in base:
            P *= q
        perms = [tuple(base)] + [tuple(random.sample(base, n)) for _ in range(11)]
        bad = 0
        for perm in perms:
            seq = lex_sequence(list(perm))
            if len(distinct_diffs(seq, P, cyclic=True)) != 2 * n:
                bad += 1
        gate(bad == 0, "12 sampled orderings of %s give exactly 2n = %d" % (base, 2 * n))
    # AND the SET of values does depend on the order (the law does not)
    s1 = set(predicted_set([5, 7, 11]))
    s2 = set(predicted_set([11, 7, 5]))
    gate(s1 != s2, "the 2n VALUES depend on the ordering (only the COUNT is invariant)")


# ---------------------------------------------------------------- part E
def synth_lex_count(mods, removes):
    """Generic sieve: gear mods[i] removes residue set removes[i] (0 excluded)."""
    A = [sorted(set(range(m)) - set(r)) for m, r in zip(mods, removes)]
    P = 1
    for m in mods:
        P *= m
    seq = [crt(list(t), list(mods)) for t in itertools.product(*A)]
    return len(distinct_diffs(seq, P, cyclic=False)), len(distinct_diffs(seq, P, cyclic=True)), A


def partE():
    """2n is NOT a product-set fact - it is 'two non-adjacent teeth per gear'."""
    print("\n=== PART E: the general law, and why the twin machine gives 2 per gear ===")
    cases = [
        ("two non-adjacent (the machine)", [11, 13], [[2, 9], [2, 11]]),
        ("two ADJACENT removed", [11, 13], [[1, 2], [2, 11]]),
        ("three removed, spread", [11, 13], [[1, 2, 5], [2, 11]]),
        ("one removed", [11, 13], [[3], [4]]),
        ("four spread", [17, 13], [[2, 5, 9, 13], [2, 11]]),
    ]
    for name, mods, rem in cases:
        lin, cyc, A = synth_lex_count(mods, rem)
        d = []
        for a in A:
            d.append(len(set(a[j + 1] - a[j] for j in range(len(a) - 1))))
        print("    %-32s mods %s  d_i %s  sum %d   linear %d  cyclic %d"
              % (name, mods, d, sum(d), lin, cyc))
        gate(lin == sum(d), "%s: linear count == sum d_i = %d" % (name, sum(d)))
    # the falsifier: a sieve with sum d_i != 2n
    lin, cyc, A = synth_lex_count([11, 13], [[1, 2, 5], [2, 11]])
    gate(lin != 4, "a 2-gear sieve with a 3-point removal gives %d != 2n = 4" % lin)
    # the non-adjacency fact for the real machine, at scale
    bad = [q for q in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                       67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
           if abs(teeth(q)[0] - teeth(q)[1]) % q in (1, q - 1) or 0 in teeth(q)]
    gate(not bad, "no gear 5..113 has adjacent teeth or a tooth at 0")


# ---------------------------------------------------------------- part F
def partF():
    """The digital-sequence form: an explicit closed-form bijection [0,N) -> O."""
    print("\n=== PART F: the odometer / van der Corput closed form ===")
    for n in range(2, 6):
        gears = GEARS[:n]
        mods = list(gears)
        A = [sorted(exposed(q)) for q in gears]
        m = [len(a) for a in A]
        E = idempotents(mods)
        P = 1
        for q in gears:
            P *= q
        N = 1
        for mm in m:
            N *= mm
        # Phi(t) = sum_i A_i[digit_i(t)] * E_i mod P, digits mixed-radix, i=1 most
        # significant.
        vals = []
        for t in range(N):
            x = t
            digs = [0] * n
            for i in range(n - 1, -1, -1):
                digs[i] = x % m[i]
                x //= m[i]
            v = 0
            for i in range(n):
                v += A[i][digs[i]] * E[i]
            vals.append(v % P)
        seq = lex_sequence(gears)
        gate(vals == seq, "gears %s: Phi(t) reproduces the lex enumeration exactly" % (gears,))
        nat = sorted(vals)
        gate(len(set(vals)) == N, "gears %s: Phi is injective (N = %d)" % (gears, N))
        gaps = [nat[i + 1] - nat[i] for i in range(N - 1)] + [P - nat[-1] + nat[0]]
        F = max(gaps)
        print("    gears %-24s N=%-8d P=%-9d F=%-4d dispersion F/P = %.6f  N*F/P = %.4f"
              % (gears, N, P, F, F / P, N * F / P))


# ---------------------------------------------------------------- part G
def partG():
    """The shuffle: what is F as a statistic of the phase-order -> natural-order
    permutation?  Measurements, and the dual count (P6)."""
    print("\n=== PART G: the shuffle statistics ===")
    print("    y      n   N        lex-diff  nat-diff   inv-diff   F    F@lexjump")
    for n in range(2, 7):
        gears = GEARS[:n]
        P = 1
        for q in gears:
            P *= q
        seq = lex_sequence(gears)
        N = len(seq)
        nat = sorted(seq)
        # natural-order distinct differences (the gap spectrum)
        natd = set(nat[i + 1] - nat[i] for i in range(N - 1))
        natd.add(P - nat[-1] + nat[0])
        # the inverse shuffle: lex-index displacement between natural neighbours
        pos = {v: t for t, v in enumerate(seq)}
        invd = set()
        for i in range(N - 1):
            invd.add((pos[nat[i + 1]] - pos[nat[i]]) % N)
        lexd = distinct_diffs(seq, P, cyclic=True)
        gaps = [nat[i + 1] - nat[i] for i in range(N - 1)] + [P - nat[-1] + nat[0]]
        F = max(gaps)
        # the lex-index displacement across the record gap
        r = gaps.index(F)
        if r == N - 1:
            jump = (pos[nat[0]] - pos[nat[-1]]) % N
        else:
            jump = (pos[nat[r + 1]] - pos[nat[r]]) % N
        print("    %-6s %d   %-8d %-9d %-10d %-10d %-4d %d"
              % (gears[-1], n, N, len(lexd), len(natd), len(invd), F, jump))
        gate(len(lexd) == 2 * n, "gears %s: lex side has exactly 2n = %d" % (gears, 2 * n))
        if n >= 4:
            gate(len(invd) > 10 * n,
                 "gears %s: inverse shuffle is NOT small (%d displacements)" % (gears, len(invd)))
    print("\n    READING: the lex order has a 2n-element difference set and an")
    print("    intractable index permutation; the natural order has the trivial")
    print("    permutation and an intractable difference set (the gap histogram,")
    print("    F its maximum).  All the machine's difficulty is the shuffle.")


# ---------------------------------------------------------------- part H
def runs_law(q, T):
    """THE STEP-TYPE LAW.  A_i's sorted consecutive differences are exactly
    {L+1 : L an INTERIOR maximal run of the removed set T} together with 1
    whenever any two survivors are adjacent.  'Interior' = the run contains
    neither 0 nor q-1 (a run at either end has no survivor on one side of it,
    so it generates no difference).  Hence

        d_i = #distinct interior run lengths  (+1 if a 1-step survives).
    """
    T = set(T)
    A = sorted(set(range(q)) - T)
    steps = set(A[j + 1] - A[j] for j in range(len(A) - 1))
    lengths, n_int, r = set(), 0, 0
    start = None
    for x in range(q):
        if x in T:
            if r == 0:
                start = x
            r += 1
        else:
            if r and start != 0:
                lengths.add(r)
                n_int += 1
            r = 0
    # a trailing run reaches q-1: not interior
    pred = set(L + 1 for L in lengths)
    if (len(A) - 1) - n_int > 0:
        pred.add(1)
    return steps, pred


def partH():
    """THE DEFLATION: the 2n count is BLIND to the arithmetic that sets F.

    The count depends on each gear only through 'how many distinct maximal-run
    lengths does the removed set have', which is 1 for ANY two-element removal
    that is not a terminal adjacent pair.  So every admissible re-choice of the
    teeth - which moves F by a factor of two - leaves the count at exactly 2n.
    """
    print("\n=== PART H: is 2n sensitive to anything that moves F? ===")
    import random
    random.seed(1927)
    # H1: the run-length law for the count, over arbitrary removals
    bad = 0
    for _ in range(400):
        q = random.choice([11, 13, 17, 19, 23])
        k = random.randint(1, 4)
        T = random.sample(range(q), k)
        steps, pred = runs_law(q, T)
        if steps != pred:
            bad += 1
    gate(bad == 0, "run-length law for d_i verified on 400 random removals")

    # H2: tooth re-randomisation at a fixed machine - count fixed, F swings
    mods = [5, 7, 11, 13]
    n = len(mods)
    P = 1
    for q in mods:
        P *= q
    trueT = [list(teeth(q)) for q in mods]
    # sample from exactly the class the real teeth belong to: two removed
    # residues whose step-type count d_i is 2 (part-E law).
    samples = [trueT]
    while len(samples) < 60:
        cand = []
        for q in mods:
            while True:
                a, b = random.sample(range(q), 2)
                if len(runs_law(q, [a, b])[0]) == 2:
                    break
            cand.append([a, b])
        samples.append(cand)
    counts, Fs = set(), []
    for T in samples:
        lin, cyc, A = synth_lex_count(mods, T)
        counts.add((lin, cyc))
        seq = [crt(list(t), mods) for t in itertools.product(*A)]
        nat = sorted(seq)
        g = [nat[i + 1] - nat[i] for i in range(len(nat) - 1)] + [P - nat[-1] + nat[0]]
        Fs.append(max(g))
    gate(set(c[0] for c in counts) == {2 * n},
         "60 tooth choices at mods %s: LINEAR distinct-diff count is 2n = %d for EVERY one"
         % (mods, 2 * n))
    print("    (cyclic counts over the same 60: %s - the free wrap closure of part C"
          % sorted(set(c[1] for c in counts)))
    print("     is a property of the machine's OWN teeth, not of every 2-point sieve)")
    gate(max(Fs) > 1.4 * min(Fs),
         "the same 60 choices move F over [%d, %d] - a factor %.2f"
         % (min(Fs), max(Fs), max(Fs) / min(Fs)))
    print("    mods %s: count fixed at %d; F ranges %d..%d (true machine F = %d)"
          % (mods, 2 * n, min(Fs), max(Fs), Fs[0]))

    # H3: the single exception - a TERMINAL adjacent pair costs one value
    lin, cyc, A = synth_lex_count([5, 7, 11], [[1, 4], [1, 6], [9, 10]])
    gate(lin == 5, "teeth {9,10} at gear 11 (terminal adjacent pair): count 5, not 6")
    print("    the only way to leave 2n is to remove {q-2,q-1} - a fact about where")
    print("    you cut the cycle, not about the sieve.")

    # H4: not even primes are needed
    lin, cyc, A = synth_lex_count([8, 9, 25], [[2, 5], [2, 7], [4, 20]])
    gate(lin == 6, "coprime NON-PRIME moduli [8,9,25] also give 2n = 6 (linear)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCDEFGH")
    a = ap.parse_args()
    fns = {"A": partA, "B": partB, "C": partC, "D": partD, "E": partE,
           "F": partF, "G": partG, "H": partH}
    for p in a.parts:
        fns[p]()
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

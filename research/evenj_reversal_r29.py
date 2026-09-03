"""
LATERAL round 29, BLOCK B - THE EVEN-J REVERSAL SYMMETRY, AND WHAT IT BUYS.

Round 28 killed the mirror as an ODD-J lever's only home: at J = 5 the maximiser
IS a palindrome, at J = 3 and J = 4 it is a reversal PAIR.  This script states the
symmetry as a map on windows, proves what it preserves, and settles what it gives
for EVEN J.

THE MAP.  Openings o_0 = 0 < ... < o_{N-1}, N = prod(q-2) ODD, o_{N-t} = P - o_t.
The mirror k -> -k carries the depth-J window W_t = [o_t, o_{t+J}] to W_{N-t-J},
i.e. on indices it is the involution R_J(t) = -(t+J) (mod N).  It preserves
depth J, span, the number of interior openings (J-1), sends the gap WORD to its
REVERSAL, and sends a killing residue r mod q' to -(r + span).  R_J has exactly
one fixed point (N odd): the SELF-MIRROR window, centred on slot 0 for J even and
on the antipode for J odd.

THE THEOREM THIS SCRIPT GATES (new here; the J=2 case is the known one):

  For every J >= 3 the self-mirror depth-J window is NEVER WORD-LEGAL.
    J ODD:  its central middle is the gap straddling the antipode, which has
            length 1 (lateral 7.3: both antipodal slots are openings).  1 is a
            legal letter iff 1 = 0 or +-2u' (mod q'); 2u' = 2*6^{-1} = 3^{-1},
            so that needs 3 = +-1 (mod q'), i.e. q' | 2 or q' | 4 - impossible.
    J EVEN >= 4: its two CENTRAL middles are both equal to d_0, the machine's
            first gap.  T3 forbids two equal NONZERO classes in a row, and
            0 < d_0 < q' forbids both being padded (= 0 mod q').
    J = 2:  no middles at all, so the self-mirror 2-window (d_0, d_0) IS legal.
            This is the one depth where the lever needs a hypothesis, and there
            it is exactly d_0 != F.

CONSEQUENCE.  R_J is FIXED-POINT-FREE on the word-legal family at every J >= 3,
so EVERY span count over that family is EVEN, with no exceptional class and no
census - which upgrades lateral round 26's item 7.5 (a 66-cell check) to a
theorem, and discharges Formalist's `hexc` at every depth >= 3.

Usage: python evenj_reversal_r29.py [--upto 23] [--maxj 7]
"""
import argparse
import sys

import numpy as np

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]
NGATE = 0


def gate(cond, msg):
    global NGATE
    NGATE += 1
    if not cond:
        print("ASSERT FAIL: " + msg)
        raise AssertionError(msg)
    print("  ASSERT ok: " + msg)


def machine(gears):
    P = 1
    for q in gears:
        P *= q
    v = {q: min(pow(6, -1, q) % q, (-pow(6, -1, q)) % q) for q in gears}
    blocked = np.zeros(P, dtype=bool)
    for q in gears:
        blocked[v[q] % q::q] = True
        blocked[(-v[q]) % q::q] = True
    op = np.flatnonzero(~blocked).astype(np.int64)
    return P, op, v


def legal_mask(g, qp):
    """(legal, cls) for the gap array g at incoming gear q'."""
    u = pow(6, -1, qp) % qp
    A = (2 * u) % qp
    B = qp - A
    r = g % qp
    cls = np.zeros(g.size, dtype=np.int8)
    cls[r == A] = 1
    cls[r == B] = -1
    legal = (r == 0) | (r == A) | (r == B)
    return legal, cls, A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, default=23)
    ap.add_argument("--maxj", type=int, default=7)
    a = ap.parse_args()

    for gi in range(2, len(PRIMES)):
        y = PRIMES[gi]
        if y > a.upto:
            break
        gears = PRIMES[:gi + 1]
        qp = PRIMES[gi + 1]
        P, op, v = machine(gears)
        N = op.size
        Nref = 1
        for q in gears:
            Nref *= q - 2
        gate(N == Nref, "m%d: prod(q-2) = %d openings" % (y, N))
        gate(N % 2 == 1, "m%d: N is ODD (so R_J has exactly one fixed point)" % y)
        g = np.empty(N, dtype=np.int64)
        g[:-1] = op[1:] - op[:-1]
        g[-1] = P - op[-1] + op[0]
        d0 = int(g[0])
        print("\n===== machine {5..%d} -> q' = %d :  P = %d, N = %d, d_0 = %d ====="
              % (y, qp, P, N, d0))
        legal, cls, A, B = legal_mask(g, qp)
        gate(0 < d0 < qp, "m%d: 0 < d_0 = %d < q' = %d (kills the padded branch "
                          "of the even-J theorem)" % (y, d0, qp))
        # the antipodal gap is 1
        ant = (P + 1) // 2
        NGATE_local = int(np.searchsorted(op, ant))
        gate(op[NGATE_local] == ant and op[NGATE_local - 1] == ant - 1,
             "m%d: both antipodal slots (P-+1)/2 are openings, so the antipodal "
             "gap is 1" % y)
        gate(1 not in (0, A, B),
             "m%d: 1 is NOT a legal letter mod q'=%d (legal classes {0,%d,%d})"
             % (y, qp, A, B))
        gate(int(g[NGATE_local - 1]) == 1, "m%d: the gap at the antipode is 1" % y)

        inv2 = pow(2, -1, N)
        ext = np.concatenate([g, g[:a.maxj + 2]])
        pre = np.zeros(ext.size + 1, dtype=np.int64)
        np.cumsum(ext, out=pre[1:])
        lex = np.concatenate([legal, legal[:a.maxj + 2]])
        cex = np.concatenate([cls, cls[:a.maxj + 2]])

        for J in range(2, a.maxj + 1):
            t0 = (-J * inv2) % N
            word = [int(ext[(t0 + i) % N]) for i in range(J)]
            span = sum(word)
            # the self-mirror window really is self-mirror
            gate(((-(t0 + J)) % N) == t0,
                 "m%d J=%d: R_J fixes t0 = %d and nothing else" % (y, J, t0))
            gate(word == word[::-1],
                 "m%d J=%d: the self-mirror window's word IS a palindrome %s"
                 % (y, J, word))
            mids = word[1:-1]
            if J >= 3:
                mc = [(0 if m % qp == 0 else (1 if m % qp == A
                                              else (-1 if m % qp == B else 9)))
                      for m in mids]
                t2 = all(c != 9 for c in mc)
                nz = [c for c in mc if c != 0]
                t3 = all(nz[i] != nz[i + 1] for i in range(len(nz) - 1))
                gate(not (t2 and t3),
                     "m%d J=%d: SELF-MIRROR WINDOW IS NOT WORD-LEGAL "
                     "(middles %s, classes %s)" % (y, J, mids, mc))
                if J % 2 == 0:
                    gate(mids[len(mids) // 2 - 1] == mids[len(mids) // 2] == d0,
                         "m%d J=%d (even): the two CENTRAL middles are both d_0 = %d"
                         % (y, J, d0))
                else:
                    gate(mids[len(mids) // 2] == 1,
                         "m%d J=%d (odd): the CENTRAL middle is the antipodal "
                         "gap = 1" % (y, J))
            else:
                gate(word == [d0, d0],
                     "m%d J=2: the self-mirror 2-window is (d_0, d_0) = %s and "
                     "IS word-legal (no middles) - the one depth needing hexc"
                     % (y, word))

            # ---- the word-legal family at this depth, and its parity ----
            idx = np.arange(N)
            valid = np.ones(N, dtype=bool)
            last = np.zeros(N, dtype=np.int8)
            for k in range(J - 2):
                j = idx + 1 + k
                cj = cex[j]
                compat = (cj == 0) | (last == 0) | (last != cj)
                valid &= lex[j] & compat
                last = np.where(cj != 0, cj, last)
            nleg = int(valid.sum())
            spans = (pre[idx + J] - pre[idx])[valid]
            if nleg == 0:
                print("   J=%d: word-legal family EMPTY" % J)
                continue
            # reversal closure of the family (as an index involution)
            tt = idx[valid]
            rr = (-(tt + J)) % N
            gate(bool(np.all(valid[rr])),
                 "m%d J=%d: the word-legal family is CLOSED under R_J (%d windows)"
                 % (y, J, nleg))
            gate(bool(np.all((pre[rr + J] - pre[rr]) == spans)),
                 "m%d J=%d: R_J preserves SPAN on the whole legal family" % (y, J))
            bc = np.bincount(spans)
            odd = np.flatnonzero(bc % 2 == 1)
            if J >= 3:
                gate(odd.size == 0,
                     "m%d J=%d: EVERY span count over the word-legal family is "
                     "EVEN - no exceptional class (%d windows, %d spans)"
                     % (y, J, nleg, int((bc > 0).sum())))
            else:
                gate(odd.size == 1 and int(odd[0]) == 2 * d0,
                     "m%d J=2: exactly ONE odd span count, at 2*d_0 = %d"
                     % (y, 2 * d0))
            # palindromes in the legal family, split literal / padded
            pal = np.ones(nleg, dtype=bool)
            for i in range(J // 2):
                pal &= (ext[tt + i] == ext[tt + (J - 1 - i)])
            npal = int(pal.sum())
            padded = np.zeros(nleg, dtype=bool)
            for k in range(J - 2):
                padded |= (ext[tt + 1 + k] % qp == 0)
            npal_lit = int((pal & ~padded).sum())
            npal_pad = int((pal & padded).sum())
            if J >= 4 and J % 2 == 0:
                gate(npal_lit == 0,
                     "m%d J=%d (even): ZERO LITERAL word-legal palindromes "
                     "(Constructor Theorem B), padded palindromes %d"
                     % (y, J, npal_pad))
            Q = int(spans.max())
            nmax = int((spans == Q).sum())
            wmax = [int(ext[tt[int(np.argmax(spans))] + i]) for i in range(J)]
            if J >= 3:
                gate(nmax % 2 == 0,
                     "m%d J=%d: the number of windows attaining Q*_J = %d is "
                     "EVEN (%d)" % (y, J, Q, nmax))
            print("   J=%d: %8d legal windows (%d palindromic: %d literal, %d "
                  "padded), Q*_%d = %3d, attained %d times, a maximiser %s%s"
                  % (J, nleg, npal, npal_lit, npal_pad, J, Q, nmax, wmax,
                     "  PALINDROME" if wmax == wmax[::-1] else ""))
    print("\nALL %d ASSERTION GATES PASSED" % NGATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

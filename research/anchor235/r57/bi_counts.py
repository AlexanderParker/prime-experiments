"""bi_counts.py -- the count side: the branching identity, its closed forms, and the moments.

For every rung M -> M + q' (M = m5 .. m29) this computes, three independent ways where possible:

  route A (closed form)  C_r = W_{r-1} + Z_{r-1} from the old machine's LETTER dictionary alone,
                         then n_J = C_{J-1} - 2 C_J + C_{J+1};
  route B (local weight) n_J = sum over positions of eps_J, the number of copies in which the
                         J-run at that position fuses into exactly one new gap;
  route C (direct build) the order histogram of the tiled period, copy by copy, no words.

and checks the moment identities
      sum_J n_J = (q'-2) N,  sum_J J n_J = q' N,  mean = q'/(q'-2),
      sum_J J^2 n_J = 2 sum_{r>=0} C_r - q' N = q' N + 4 N + 2 sum_{r>=2} C_r.

Usage:  uv run python research/anchor235/r57/bi_counts.py [maxtop]
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_core import (OUT, PRIMES, u_of, machine_gaps, letters, chain_counts, word_counts,
                     orders_from_chains, eps_of_window, residues_mod, orders_direct, wrap_pad)


def eps_orders(lt, q, jmax, chunk=20_000_000):
    """route B: n_J = sum_n eps_J(n), for J = 1 .. jmax."""
    N = lt.size
    lp = wrap_pad(lt, jmax)
    out = []
    for J in range(1, jmax + 1):
        tot = 0
        pos = 0
        while pos < N:
            end = min(pos + chunk, N)
            lts = [lp[pos + k:end + k] for k in range(J)]
            tot += int(eps_of_window(lts, q).sum())
            pos = end
        out.append(tot)
    return out


def main():
    maxtop = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    lines = []
    W = lines.append
    W("bi_counts -- the branching identity, closed forms and moments")
    W("")
    hdr = ("rung | N_old | q' | L | J_max | C_0 | C_1 | C_2 | C_3 | C_4 | C_5 | "
           "A_0 | A_d | n_1..n_6 (route A) | routeB == A | routeC == A")
    W(hdr)
    rows = []
    for k in range(len(PRIMES) - 1):
        top, q = PRIMES[k], PRIMES[k + 1]
        if top > maxtop:
            break
        t0 = time.time()
        P, f0, gaps = machine_gaps(top)
        N = gaps.size
        lt = letters(gaps, q)
        C = chain_counts(lt, q, N, rmax=10)
        nJ = orders_from_chains(C)
        # L = longest realised legal word
        L = 0
        while True:
            Wm, Zm = word_counts(lt, L + 1)
            if Wm == 0:
                break
            L += 1
            if L > 12:
                break
        jmax = L + 2
        spec = np.bincount(gaps)                 # the gap spectrum of M, by value
        vals = np.arange(spec.size, dtype=np.int64)
        d = (2 * u_of(q)) % q
        A0 = int(spec[(vals % q == 0) & (vals > 0)].sum())
        Ad = int(spec[(vals % q == d) | (vals % q == (-d) % q)].sum())
        nB = eps_orders(lt, q, max(jmax + 1, 6))
        # route C: direct build on the tiled period
        res = residues_mod(gaps, q, r0=f0)
        histC = orders_direct(res, q, P)
        del res
        nC = [int(histC[J]) for J in range(1, len(nB) + 1)]
        okB = all(nB[i] == nJ[i] for i in range(len(nB)))
        okC = all(nC[i] == nJ[i] for i in range(len(nC)))
        Nnew = sum(nJ)
        sumJ = sum((i + 1) * n for i, n in enumerate(nJ))
        sumJ2 = sum((i + 1) ** 2 * n for i, n in enumerate(nJ))
        pred2 = q * N + 4 * N + 2 * sum(C[2:])
        rows.append(dict(top=top, q=q, N=N, L=L, jmax=jmax, C=C, A0=A0, Ad=Ad, lt=lt,
                         nJ=nJ, nB=nB, nC=nC, okB=okB, okC=okC, Nnew=Nnew,
                         sumJ=sumJ, sumJ2=sumJ2, pred2=pred2, P=P,
                         maxorder=max(i + 1 for i, n in enumerate(nJ) if n > 0),
                         secs=time.time() - t0))
        W(f"{top}->{q} | {N} | {q} | {L} | {jmax} | " +
          " | ".join(str(x) for x in C[:6]) +
          f" | {A0} | {Ad} | " + " ".join(str(x) for x in nJ[:6]) +
          f" | {okB} | {okC}   [{time.time()-t0:.1f}s]")
        print(W.__self__[-1], flush=True)
    W("")
    W("== closed forms of C_2, C_3, C_4 from the residue dictionaries ==")
    W("rung | C_2 | 2A_0 + A_d | C_3 | 2 Z_2 + W2mixed | C_4 | 2 Z_3 + W3mixed")
    for row in rows:
        top, q, C = row["top"], row["q"], row["C"]
        lt = row["lt"]
        W2, Z2 = word_counts(lt, 2)
        W3, Z3 = word_counts(lt, 3)
        W(f"{top}->{q} | {C[2]} | {2*row['A0'] + row['Ad']} | {C[3]} | {W2 + Z2} | "
          f"{C[4]} | {W3 + Z3}")
    W("")
    W("== moments ==")
    W("rung | N_new | (q'-2)N | sum J | q'N | mean | q'/(q'-2) | sum J^2 | "
      "q'N+4N+2 sum_{r>=2} C_r | E[J^2] | Var | teeth-free part 2(q'-4)/(q'-2)^2 | S")
    for row in rows:
        q, N = row["q"], row["N"]
        S = sum(row["C"][2:]) / N
        mean = row["sumJ"] / row["Nnew"]
        eJ2 = row["sumJ2"] / row["Nnew"]
        var = eJ2 - mean ** 2
        W(f"{row['top']}->{q} | {row['Nnew']} | {(q-2)*N} | {row['sumJ']} | {q*N} | "
          f"{mean:.9f} | {q/(q-2):.9f} | {row['sumJ2']} | {row['pred2']} | {eJ2:.9f} | "
          f"{var:.9f} | {2*(q-4)/(q-2)**2:.9f} | {S:.9f}")
    W("")
    W("== max order vs L + 2, and the chain depth ==")
    W("rung | max order | L + 2 | D = max r with C_r > 0 | L + 1")
    for row in rows:
        D = max(r for r, c in enumerate(row["C"]) if c > 0)
        W(f"{row['top']}->{row['q']} | {row['maxorder']} | {row['jmax']} | {D} | {row['L']+1}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "bi_counts.txt"), "w").write(txt)
    print("\nwrote results/bi_counts.txt")


if __name__ == "__main__":
    main()

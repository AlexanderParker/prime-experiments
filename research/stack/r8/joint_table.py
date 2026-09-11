"""The joint field-index table of a section (fields_construction.md F4, step 5).

Column k = (6k - 1, 6k + 1). Field index of a member = its number of prime factors with
multiplicity (all >= 5 on S). For a section [lo, hi) (columns with lo <= 6k - 1 and
6k + 1 < hi), the table T[i][j] = number of columns whose left member has index i and right
member index j. Marginals = single-field counts. Independence prediction for a cell:
(row sum)(column sum)/N. The twin cell is (1, 1). Reported per section: N, the table for
i, j <= 4, each cell's observed/independent ratio, the (1, 1) cell's ratio and z.

Usage: uv run python joint_table.py lo hi [lo hi ...]
"""
import sys, math
import numpy as np


def omega_table(N):
    """Omega(n) (with multiplicity) for 0..N by a sieve of least prime factors."""
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, int(N ** 0.5) + 1):
        if spf[i] == 0:
            spf[i*i::i][spf[i*i::i] == 0] = i
    om = np.zeros(N + 1, dtype=np.int64)
    for n in range(2, N + 1):
        p = spf[n] if spf[n] else n
        om[n] = 1 + om[n // p]
    return om


def main():
    args = [int(x) for x in sys.argv[1:]]
    N = max(args) + 2
    om = omega_table(N)
    for lo, hi in zip(args[::2], args[1::2]):
        k0 = (lo + 1) // 6 + (1 if (lo + 1) % 6 else 0); k1 = (hi - 2) // 6
        ks = np.arange(k0, k1 + 1); L = om[6 * ks - 1]; R = om[6 * ks + 1]
        n = len(ks); K = 5
        T = np.zeros((K + 1, K + 1), dtype=np.int64)
        for i in range(1, K + 1):
            for j in range(1, K + 1):
                T[i, j] = int(np.sum((L == i) & (R == j)))
        row = T.sum(axis=1); col = T.sum(axis=0)
        print(f"section [{lo}, {hi}): columns {n}; left marginals {row[1:].tolist()}; right marginals {col[1:].tolist()}")
        print("  observed / independent, cells (i, j) for i, j = 1..4:")
        for i in range(1, 5):
            cells = []
            for j in range(1, 5):
                e = row[i] * col[j] / n
                cells.append(f"{T[i,j]}/{e:.1f}={T[i,j]/e:.2f}" if e > 0 else "-")
            print("   ", " | ".join(cells))
        e11 = row[1] * col[1] / n
        print(f"  twin cell (1,1): observed {T[1,1]}, independent {e11:.1f}, ratio {T[1,1]/e11:.3f}, z {(T[1,1]-e11)/math.sqrt(e11):+.2f}")


if __name__ == "__main__":
    main()

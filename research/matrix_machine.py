"""THE MACHINE AS ONE MATRIX ALGEBRA - a working linear-algebra implementation
of the project's discovered laws (round-20 implementation task).

Objects (all exact integer matrices unless a section says FLOAT):

  V_q  = Z^q            per-gear residue space
  D_q  = diag(teeth)    blocking projector of gear q  (teeth {u, -u}, 6u=1 mod q)
  E_q  = I - D_q        exposure projector
  S_q  = cyclic shift   S_q e_r = e_{r+1 mod q}
  CRT  : Z_P = (x) V_q  (Kronecker product; P = prod q); slot k <-> (k mod q)_q

Laws that OPERATE in this algebra (each verified below with assertions):

  1. open count  |E(M)| = trace( (x)_q E_q ) = prod_q trace(E_q) = prod (q-2)
     corridor / autocorrelation law:  c_q(g) = trace(E_q S_q^g E_q S_q^-g)
       = (C_q C_q^T)[g,0]  where C_q = sum_r a_q(r) S_q^r  (circulant of the
       exposure indicator) - Lateral's three-case law {q-2, q-3, q-4} as an
       integer matrix identity; mod-35 corridor = the 35x35 case E_5 (x) E_7.
     depth-sum identity: sum_j W_j(g) = trace(E_P S_P^g E_P S_P^-g)
       = prod_q c_q(g)  (trace of Kronecker = product of traces).
  2. F(M) = nilpotency index of B S, B = I - (x)E_q, with the exact Kronecker
     splitting  B S = (x)_q S_q - (x)_q (E_q S_q);  verified by exact matrix
     powering at machines 11, 13, 17.
  3. merge law as operator: adding gear q' = lift by tensoring (E_P (x) E_q'),
     delete via D_q'; F(M+q') = nilpotency index of B'S' on the tensored
     space, computed WITHOUT materialising the CRT-recombined period.
  4. paired-Holt transfer matrix: the explicit matrix T with
     T[g, w] = coef(w) [sum(w)=g] maps the old word-population vector to the
     new gap histogram exactly (4 rungs); the square word-level matrix H is
     block-triangular by word length with diagonal coef_diag(w), so its
     eigenvalues are EXACT RATIONALS after normalising by (q'-2), generically
     (q'-2j-2)/(q'-2) at word length j.
  5. DFT diagonalisation: Lateral's closed-form spectrum hat_q(j) =
     -2cos(2 pi j u / q) is the eigenvalue list of the circulant C_q built in
     (1); gear 5's characteristic polynomial factors EXACTLY as
     (x-3)(x^2-x-1)^2, so the largest non-Perron eigenvalue is phi (golden
     ratio) and the machine-independent spectral gap phi/3 is an exact
     eigenvalue statement.

Benchmark protocol: OPERATION COUNTS, not wall time - counters in OPS.
Boundaries (recorded, not softened): no spectral gap in the exact frame (the
renewal operator is a permutation); the aggregated gap chain is NOT Markov
(no fixed-order transfer matrix on gap values carries the deep-run law);
B = I - (x)E_q is NOT itself a Kronecker product (blocking is the complement
of a product), which is why nilpotency needs the full space.

Run:  uv run python research/matrix_machine.py
"""
import numpy as np
import scipy.sparse as sp
import sympy
from sympy import symbols, expand
from math import prod, cos, pi
from fractions import Fraction
from collections import Counter

OPS = Counter()          # benchmark protocol: explicit operation counters

# ---------------------------------------------------------------- utilities
def primes_upto(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for p in range(2, int(n ** .5) + 1):
        if s[p]:
            s[p * p::p] = False
    return [int(p) for p in np.flatnonzero(s)]

def teeth(q):
    u = pow(6, -1, q)
    return u % q, (-u) % q

def exposed_vec(q):
    """per-gear exposure indicator a_q on Z_q (integer 0/1 vector)."""
    a = np.ones(q, np.int64)
    t1, t2 = teeth(q)
    a[t1] = a[t2] = 0
    return a

def E_mat(q):
    return np.diag(exposed_vec(q))

def S_mat(q):
    """cyclic shift, S e_r = e_{r+1}."""
    S = np.zeros((q, q), np.int64)
    for r in range(q):
        S[(r + 1) % q, r] = 1
    return S

def sieve_openings(gears):
    """direct census route (for cross-checks): boolean exposure on Z_P."""
    P = prod(gears)
    a = np.ones(P, bool)
    OPS['census_sieve_init'] += P
    for q in gears:
        t1, t2 = teeth(q)
        a[t1::q] = False
        a[t2::q] = False
        OPS['census_sieve_marks'] += 2 * (P // q + 1)
    return a

def gap_word(exposed_bool):
    idx = np.flatnonzero(exposed_bool)
    P = len(exposed_bool)
    OPS['census_scan'] += P
    return np.diff(np.append(idx, idx[0] + P)), idx

def crt_perm(gears):
    """perm[k] = row-major Kronecker flat index of (k mod q1, ..., k mod qm)."""
    P = prod(gears)
    perm = np.zeros(P, np.int64)
    for k in range(P):
        f = 0
        for q in gears:
            f = f * q + (k % q)
        perm[k] = f
    return perm

def kron_all(mats):
    M = mats[0]
    for A in mats[1:]:
        M = np.kron(M, A)
    return M

def c_closed(q, g, T):
    """closed-form autocorrelation of the exposed set (Lateral r19 law),
    general two- or one-tooth set T."""
    ts = sorted(T)
    if len(ts) == 1:
        return q - 1 if g % q == 0 else q - 2
    d = (ts[1] - ts[0]) % q
    gm = g % q
    if gm == 0:
        return q - 2
    if gm in (d, (-d) % q):
        return q - 3
    return q - 4

# known F ladder in the slot (k) frame, gears 5..y
F_KNOWN = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34}

print("=" * 78)
print("PART 1 - RESIDUE STATE AS TENSOR: traces compute the census numbers")
print("=" * 78)

# --- open count as a trace of a Kronecker product --------------------------
for y in (11, 13, 17):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    # matrix route: trace((x)E_q) = prod trace(E_q); count the diagonal reads
    tr = 1
    for q in gears:
        tr *= int(np.trace(E_mat(q)))
        OPS['trace_open_count'] += q
    census = int(sieve_openings(gears).sum())
    assert tr == census == prod(q - 2 for q in gears), (y, tr, census)
    print(f"machine {y}: open count = trace((x)_q E_q) = "
          f"prod(q-2) = {tr}  == census {census}  "
          f"[trace ops {sum(gears)} vs census ops ~{2*P}]")

# --- CRT alignment: the Kronecker diagonal IS the sieve --------------------
gears = [5, 7, 11]; P = prod(gears)
EP = kron_all([E_mat(q) for q in gears])          # 385 x 385 exact
perm = crt_perm(gears)
sieve = sieve_openings(gears)
assert all(EP[perm[k], perm[k]] == int(sieve[k]) for k in range(P))
print(f"machine 11: diag((x)E_q) under the CRT index map == the sieve "
      f"indicator, all {P} slots (exact)")

# --- the corridor mod 35 as a 35x35 matrix ---------------------------------
g35 = [5, 7]; P35 = 35
perm35 = crt_perm(g35)
E35 = np.zeros((35, 35), np.int64)
K35 = kron_all([E_mat(5), E_mat(7)])
for k in range(35):
    E35[k, k] = K35[perm35[k], perm35[k]]         # E_5 (x) E_7 in slot order
S35 = S_mat(35)
S35inv = S35.T
acc = np.eye(35, dtype=np.int64)
ok = 0
for g in range(0, 70):
    Sg = np.linalg.matrix_power(S35, g % 35)
    M = E35 @ Sg @ E35 @ Sg.T                     # E S^g E S^-g
    OPS['corridor_matmul'] += 4 * 35 ** 3
    t = int(np.trace(M))
    want = c_closed(5, g, set(teeth(5))) * c_closed(7, g, set(teeth(7)))
    assert t == want, (g, t, want)
    ok += 1
print(f"corridor mod 35: trace(E S^g E S^-g) == c_5(g)*c_7(g) for g=0..69 "
      f"({ok}/70 exact) - the round-18 admissible-endpoint-phase law is a "
      f"matrix trace")

# --- depth-sum identity as the SAME trace at machine level -----------------
for y in (11, 13):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    gaps, idx = gap_word(sieve_openings(gears))
    A = len(gaps)
    Gmax = 40
    # census side: W_j(g) by direct window sums
    W = Counter()
    ext = np.concatenate([gaps, gaps])
    csum = np.concatenate([[0], np.cumsum(ext)])
    for i in range(A):
        j = 1
        while True:
            ssum = csum[i + j] - csum[i]
            if ssum > Gmax:
                break
            W[(j, int(ssum))] += 1
            j += 1
            OPS['depth_census'] += 1
    for g in range(1, Gmax + 1):
        lhs = sum(W[(j, g)] for j in range(1, g + 1))
        # matrix side: product of per-gear traces trace(E_q S_q^g E_q S_q^-g)
        rhs = 1
        for q in gears:
            Sg = np.linalg.matrix_power(S_mat(q), g % q)
            rhs *= int(np.trace(E_mat(q) @ Sg @ E_mat(q) @ Sg.T))
            OPS['depth_trace'] += q
        assert lhs == rhs, (y, g, lhs, rhs)
    print(f"machine {y}: depth-sum identity sum_j W_j(g) == "
          f"prod_q trace(E_q S^g E_q S^-g) for g=1..{Gmax} (exact; "
          f"trace ops ~{Gmax*sum(gears)} vs window census ops "
          f"{OPS['depth_census']})")

print()
print("=" * 78)
print("PART 2 - SHIFT + BLOCK ALGEBRA: F(M) = nilpotency index of B S")
print("=" * 78)

# --- the exact Kronecker splitting of BS at machine 11 (dense, 385x385) ----
gears = [5, 7, 11]; P = prod(gears)
perm = crt_perm(gears); inv = np.argsort(perm)
Pm = np.zeros((P, P), np.int64)
for k in range(P):
    Pm[perm[k], k] = 1                             # CRT permutation matrix
SP = S_mat(P)
kron_S = kron_all([S_mat(q) for q in gears])
assert np.array_equal(kron_S, Pm @ SP @ Pm.T), "S_P != (x)S_q under CRT"
print(f"machine 11: S_P == Perm^T ((x)_q S_q) Perm exactly (the slot shift "
      f"IS the tensor of per-gear cyclic shifts)")
kron_E = kron_all([E_mat(q) for q in gears])
kron_ES = kron_all([E_mat(q) @ S_mat(q) for q in gears])
BS_tensor = kron_S - kron_ES                      # B S = (x)S - (x)(ES)
EPfull = Pm.T @ kron_E @ Pm
BSfull = (np.eye(P, dtype=np.int64) - EPfull) @ SP
assert np.array_equal(BSfull, Pm.T @ BS_tensor @ Pm)
print(f"machine 11: B S == (x)_q S_q - (x)_q (E_q S_q) exactly - the "
      f"nilpotent operator is a DIFFERENCE of two Kronecker products")
print(f"  (B itself is NOT a tensor product: blocking is the complement of "
      f"a product - the recorded structural boundary)")

# --- nilpotency by exact matrix powering (dense at 385) --------------------
N = BSfull.copy()
Nm = N.copy(); m = 1
while Nm.any():
    Nm = np.sign(Nm @ N)          # entries stay 0/1; sign() guards overflow
    OPS['nilpotent_dense_matmul'] += P ** 3
    m += 1
assert m == F_KNOWN[11], (m, F_KNOWN[11])
print(f"machine 11 (dim {P}): (BS)^m = 0 first at m = {m} == F = "
      f"{F_KNOWN[11]}  [dense int matrix powering, {m-1} products]")

# --- machines 13, 17: sparse exact powering (nnz <= P per power) -----------
for y in (13, 17):
    gears = [p for p in primes_upto(y) if p >= 5]
    P = prod(gears)
    ex = sieve_openings(gears)
    b = (~ex).astype(np.int64)
    # N = B S as a sparse matrix: column k -> row (k+1) iff slot k+1 blocked
    rows = (np.arange(P) + 1) % P
    keep = b[rows] == 1
    N = sp.csr_matrix((np.ones(keep.sum(), np.int64),
                       (rows[keep], np.arange(P)[keep])), shape=(P, P))
    Nm = N.copy(); m = 1
    ops = N.nnz
    while Nm.nnz:
        Nm = Nm @ N
        Nm.data[:] = np.minimum(Nm.data, 1)
        ops += Nm.nnz
        OPS['nilpotent_sparse_ops'] += Nm.nnz
        m += 1
    assert m == F_KNOWN[y], (y, m, F_KNOWN[y])
    print(f"machine {y} (dim {P}): (BS)^m = 0 first at m = {m} == F = "
          f"{F_KNOWN[y]}  [sparse exact powering, {m-1} products, "
          f"{ops} nnz-ops total; dense would cost {(m-1)*P**3:.1e}]")

print()
print("=" * 78)
print("PART 3 - MERGE LAW AS OPERATOR: lift by (x) V_q', delete via D_q'")
print("=" * 78)

# adding gear q': E_new = E_P (x) E_q', S_new = S_P (x) S_q' (CRT coords).
# state lives on a P x q' grid - the Kronecker factorisation operating;
# the CRT-recombined 1-D period of length P*q' is NEVER materialised.
MERGE_STEPS = [([5, 7, 11], 13, 11),
               ([5, 7, 11, 13], 17, 18),
               ([5, 7, 11, 13, 17, 19], 23, 34)]
for gears, qp, F_expect in MERGE_STEPS:
    P = prod(gears)
    expP = sieve_openings(gears)                  # old machine's state
    aq = exposed_vec(qp).astype(bool)             # E_q' diagonal
    blocked = ~(expP[:, None] & aq[None, :])      # B' = I - E_P (x) E_q'
    v = np.ones((P, qp), bool)
    m = 0
    while v.any():
        v = np.roll(v, -1, axis=0)                # S_P factor
        v = np.roll(v, -1, axis=1)                # S_q' factor
        v &= blocked                              # B' projector
        m += 1
        OPS['merge_operator_elops'] += 3 * P * qp
    assert m == F_expect, (gears, qp, m, F_expect)
    print(f"gears {gears} + {qp}: F(M+q') = nilpotency index of "
          f"(I - E_P(x)E_q')(S_P(x)S_q') = {m} == known {F_expect}   "
          f"[state {P} x {qp}, {m} iterations, "
          f"{3*P*qp*m:.2e} element ops]")

print()
print("=" * 78)
print("PART 4 - PAIRED-HOLT TRANSFER MATRIX (exact; op-counted vs census)")
print("=" * 78)

def openings_slot(gears):
    return np.flatnonzero(sieve_openings(gears))

def openings_e(gears, e):
    P = prod(gears)
    a = np.ones(P, bool)
    OPS['census_sieve_init'] += P
    for q in gears:
        a[0::q] = False
        a[(-e) % q::q] = False
        OPS['census_sieve_marks'] += 2 * (P // q + 1)
    return np.flatnonzero(a)

def word_counts(gaps, Lmax):
    n = len(gaps)
    ext = np.concatenate([gaps, gaps[:Lmax]])
    cnt = Counter()
    for j in range(1, Lmax + 1):
        for i in range(n):
            cnt[tuple(int(x) for x in ext[i:i + j])] += 1
            OPS['matrix_word_pass'] += 1
    return cnt

def coef_partition(word, cuts, q, T):
    """#{r in Z_q' : boundary points not in T, interior points in T} for the
    fusion of `word` with block boundaries after positions in `cuts`.
    Set arithmetic; ops counted."""
    pts = [0]
    s = 0
    for x in word:
        s += x
        pts.append(s)
    nb = set(cuts) | {0, len(word)}
    cand = None
    for i, p in enumerate(pts):
        if i in nb:
            continue
        allowed = {(t - p) % q for t in T}
        OPS['matrix_coef_setops'] += len(T)
        cand = allowed if cand is None else cand & allowed
        if not cand:
            return 0
    if cand is None:
        cand = set(range(q))
        OPS['matrix_coef_setops'] += q
    for i in sorted(nb):
        cand -= {(t - pts[i]) % q for t in T}
        OPS['matrix_coef_setops'] += len(T)
        if not cand:
            return 0
    return len(cand)

def coef_full(word, q, T):
    return coef_partition(word, (), q, T)

def hist_matrix_route(old_gaps, q, T, Lmax=8):
    """the transfer matrix T[g,w] = coef(w)[sum w = g] applied to the old
    word-population vector; returns predicted new histogram Counter."""
    wc = word_counts(old_gaps, Lmax)
    words = sorted(wc)
    col = {w: i for i, w in enumerate(words)}
    n_old = np.array([wc[w] for w in words], dtype=np.int64)
    rows, cols, vals, sums = [], [], [], {}
    gvals = sorted({sum(w) for w in words})
    grow = {g: i for i, g in enumerate(gvals)}
    for w in words:
        c = coef_full(w, q, T)
        if c:
            rows.append(grow[sum(w)]); cols.append(col[w]); vals.append(c)
    T_hist = sp.csr_matrix((np.array(vals, np.int64), (rows, cols)),
                           shape=(len(gvals), len(words)))
    n_new = T_hist @ n_old                         # THE matrix operating
    OPS['matrix_matvec'] += T_hist.nnz
    pred = Counter({g: int(n_new[grow[g]]) for g in gvals if n_new[grow[g]]})
    return pred, T_hist, len(words)

RUNGS = []
G0 = [5, 7, 11, 13]
RUNGS.append(("slot [5,7,11,13] -> +17", openings_slot(G0), prod(G0),
              openings_slot(G0 + [17]), prod(G0 + [17]), 17,
              set(teeth(17))))
G1 = G0 + [17]
RUNGS.append(("slot [..17] -> +19", openings_slot(G1), prod(G1),
              openings_slot(G1 + [19]), prod(G1 + [19]), 19,
              set(teeth(19))))
Ge = [3, 5, 7, 11, 13]
RUNGS.append(("family e=344 -> +17", openings_e(Ge, 344), prod(Ge),
              openings_e(Ge + [17], 344), prod(Ge + [17]), 17,
              {0, (-344) % 17}))
RUNGS.append(("collapse e=102 -> +17 (Holt 1-residue)",
              openings_e(Ge, 102), prod(Ge),
              openings_e(Ge + [17], 102), prod(Ge + [17]), 17, {0}))

for name, old_idx, Pold, new_idx, Pnew, q, T in RUNGS:
    gaps_old = np.diff(np.append(old_idx, old_idx[0] + Pold))
    gaps_new = np.diff(np.append(new_idx, new_idx[0] + Pnew))
    hist_new = Counter(int(g) for g in gaps_new)
    OPS['census_scan'] += Pnew
    ops_before = (OPS['matrix_word_pass'] + OPS['matrix_coef_setops'] +
                  OPS['matrix_matvec'])
    pred, T_hist, nwords = hist_matrix_route(gaps_old, q, T)
    ops_matrix = (OPS['matrix_word_pass'] + OPS['matrix_coef_setops'] +
                  OPS['matrix_matvec']) - ops_before
    allg = sorted(set(hist_new) | set(pred))
    for g in allg:
        assert pred[g] == hist_new[g], (name, g, pred[g], hist_new[g])
    # diagonal = c-law
    for g in allg:
        assert coef_full((g,), q, T) == c_closed(q, g, T), (name, g)
    print(f"{name}: matrix T ({T_hist.shape[0]} gap values x {nwords} words, "
          f"nnz {T_hist.nnz}) @ n_old == new histogram EXACT, every gap "
          f"value; diagonal == c-law")
    print(f"   op count, matrix route: {ops_matrix}  vs census route "
          f"(sieve+scan the new period): {2*Pnew} -> ratio "
          f"x{2*Pnew/max(ops_matrix,1):.0f}")

# --- word-level square matrix H: triangular, exact rational eigenvalues ----
print()
name, old_idx, Pold, new_idx, Pnew, q, T = RUNGS[0]
gaps_old = np.diff(np.append(old_idx, old_idx[0] + Pold))
gaps_new = np.diff(np.append(new_idx, new_idx[0] + Pnew))
Lmax_e = 4
wc = word_counts(gaps_old, Lmax_e)
words = sorted(wc, key=lambda w: (-len(w), w))    # by DECREASING length
H = {}                                            # H[w_target][w_source]
from itertools import combinations
for wsrc in words:
    j = len(wsrc)
    for ncuts in range(j):
        for cuts in combinations(range(1, j), ncuts):
            c = coef_partition(wsrc, cuts, q, T)
            if not c:
                continue
            bounds = [0] + list(cuts) + [j]
            wtgt = tuple(sum(wsrc[bounds[i]:bounds[i+1]])
                         for i in range(len(bounds) - 1))
            H.setdefault(wtgt, {})[wsrc] = \
                H.setdefault(wtgt, {}).get(wsrc, 0) + c
# structural triangularity: target length < source length except identity
for wt, row in H.items():
    for ws in row:
        assert len(wt) < len(ws) or wt == ws, (wt, ws)
print(f"H (rung 1, q'={q}): block-triangular by word length - same-length "
      f"entries are ONLY the diagonal ({len(words)} words, lengths <= "
      f"{Lmax_e}); eigenvalues therefore = diagonal, EXACTLY")

# verify H actually operates: predict new word populations of length <= 2
LmaxP = 9
wcP = word_counts(gaps_old, LmaxP)
predw = Counter()
contrib_at_max = 0
for wsrc, n in wcP.items():
    j = len(wsrc)
    for ncuts in (0, 1):
        for cuts in combinations(range(1, j), ncuts):
            c = coef_partition(wsrc, cuts, q, T)
            if not c:
                continue
            bounds = [0] + list(cuts) + [j]
            wtgt = tuple(sum(wsrc[bounds[i]:bounds[i+1]])
                         for i in range(len(bounds) - 1))
            predw[wtgt] += c * n
            if j == LmaxP:
                contrib_at_max += 1
assert contrib_at_max == 0, "no closure at Lmax - increase Lmax"
new_wc = word_counts(gaps_new, 2)
keys = {w for w in set(predw) | set(new_wc) if len(w) <= 2}
for w in keys:
    assert predw[w] == new_wc[w], (w, predw[w], new_wc[w])
print(f"H @ n_old == new word populations for ALL {len(keys)} words of "
      f"length <= 2 (pairs included), exact; zero contributions at source "
      f"length {LmaxP} = closure certificate")

# exact rational eigenvalues and the (q'-2j-2)/(q'-2) eigen-scale
diag_by_len = {}
for w in words:
    d = H.get(w, {}).get(w, 0)
    diag_by_len.setdefault(len(w), Counter())[d] += 1
print(f"eigenvalues of H, exact (normalised by q'-2 = {q-2}):")
for j in sorted(diag_by_len):
    cnt = diag_by_len[j]
    generic = q - 2 * (j + 1)
    # exact law: coef_diag(w) = q' - #distinct{(t - p) mod q'} >= q'-2(j+1),
    # equality iff the j+1 window points are in general position mod q'
    assert all(d >= generic for d in cnt), (j, cnt, generic)
    attained = generic in cnt
    dom = cnt.most_common(1)[0][0]
    vals = {d: Fraction(d, q - 2) for d in sorted(cnt)}
    pretty = ", ".join(f"{d} -> {v} (x{cnt[d]})" for d, v in vals.items())
    print(f"  length {j}: {pretty}")
    print(f"    floor value q'-2(j+1) = {generic}, normalised "
          f"{Fraction(generic, q-2)} == (q'-2j-2)/(q'-2): "
          f"{'ATTAINED' if attained else 'NOT attained'}"
          f"{' and modal' if dom == generic else f'; modal value is {dom} (residue collisions mod q'' at this depth)'}")

print()
print("=" * 78)
print("PART 5 - DFT DIAGONALISATION: the closed-form spectrum is the "
      "eigenvalue list of the circulant built in part 1")
print("=" * 78)

x = symbols('x')
for q in (5, 7, 11, 13):
    a = exposed_vec(q)
    S = S_mat(q)
    C = sum(int(a[r]) * np.linalg.matrix_power(S, r) for r in range(q))
    # circulant identity: C = circulant(a)
    assert all(C[i, j] == a[(i - j) % q] for i in range(q) for j in range(q))
    # Wiener-Khinchin as an integer matrix identity: C C^T = sum_g c(g) S^g
    CCt = C @ C.T
    for g in range(q):
        assert CCt[g, 0] == c_closed(q, g, set(teeth(q))), (q, g)
        # and it equals the part-1 trace form
        Sg = np.linalg.matrix_power(S, g)
        E = E_mat(q)
        assert int(np.trace(E @ Sg @ E @ Sg.T)) == CCt[g, 0]
    # FLOAT check: the DFT diagonalises C with Lateral's closed form
    u = pow(6, -1, q)
    F = np.exp(2j * pi * np.outer(np.arange(q), np.arange(q)) / q)
    D = np.conj(F.T) @ C @ F / q
    hat = np.array([q - 2 if j == 0 else -2 * cos(2 * pi * j * u / q)
                    for j in range(q)])
    off = D - np.diag(np.diag(D))
    assert np.abs(off).max() < 1e-9
    assert np.abs(np.diag(D).real - hat).max() < 1e-9
    print(f"gear {q}: C_q = sum_r a(r) S^r (circulant of the exposure "
          f"indicator); C C^T = sum_g c_q(g) S^g exact (c-law = "
          f"autocorrelation row); DFT diagonalises C to hat_q(j) = "
          f"-2cos(2 pi j u/q) [float check < 1e-9]")

# --- gear 5: the golden eigenvalue, EXACT ----------------------------------
a5 = exposed_vec(5)
C5 = sum(int(a5[r]) * np.linalg.matrix_power(S_mat(5), r) for r in range(5))
cp = sympy.Matrix(C5.tolist()).charpoly(x).as_expr()
target = (x - 3) * (x ** 2 - x - 1) ** 2
assert expand(cp - target) == 0
print(f"\ngear 5 EXACT: charpoly(C_5) = (x-3)(x^2-x-1)^2  "
      f"[sympy, integer matrix]")
print(f"  -> eigenvalues: 3 (Perron, = q-2) and the roots of x^2-x-1 = "
      f"phi, 1-phi, each twice")
print(f"  -> largest non-Perron |eigenvalue| = phi EXACTLY; spectral gap "
      f"phi/3 is an exact eigenvalue statement for gear 5")

# machine level: C_35 == CRT-conjugated C_5 (x) C_7 (exact), and the
# phi/3 bound over full character enumeration (FLOAT, labeled)
a35 = np.ones(35, np.int64)
for q in (5, 7):
    t1, t2 = teeth(q)
    a35[t1::q] = 0; a35[t2::q] = 0
C35 = np.array([[a35[(i - j) % 35] for j in range(35)] for i in range(35)],
               np.int64)
K = np.kron(C5, sum(int(exposed_vec(7)[r]) *
                    np.linalg.matrix_power(S_mat(7), r) for r in range(7)))
Pm35 = np.zeros((35, 35), np.int64)
for k in range(35):
    Pm35[perm35[k], k] = 1
assert np.array_equal(C35, Pm35.T @ K @ Pm35)
print(f"machine 35: C_35 == Perm^T (C_5 (x) C_7) Perm exactly - the "
      f"machine circulant is the tensor of gear circulants")

for gearset in ([5, 7], [5, 7, 11], [5, 7, 11, 13]):
    ratios = []
    for q in gearset:
        u = pow(6, -1, q)
        r = np.array([1.0 if j == 0 else
                      abs(-2 * cos(2 * pi * j * u / q)) / (q - 2)
                      for j in range(q)])
        ratios.append(r)
    # all characters = product over gears of local ratios (Kronecker eigs)
    M = ratios[0]
    for r in ratios[1:]:
        M = np.outer(M, r).ravel()
    M_sorted = np.sort(M)[::-1]
    phi = (1 + 5 ** .5) / 2
    assert abs(M_sorted[0] - 1) < 1e-12            # DC
    assert abs(M_sorted[1] - phi / 3) < 1e-12      # the golden gap
    # attained only by gear-5's +-2 mode: multiplicity exactly 2
    assert abs(M_sorted[2] - phi / 3) < 1e-12 and M_sorted[3] < phi / 3 - 1e-6
    print(f"machine {gearset}: max non-DC |eig|/DC over all "
          f"{len(M)} characters = phi/3 = {phi/3:.6f}, multiplicity 2 "
          f"(gear 5's +-2 mode only) [float enumeration, labeled]")

print()
print("=" * 78)
print("OP-COUNT LEDGER (benchmark protocol: operations, not wall time)")
print("=" * 78)
for k in sorted(OPS):
    print(f"  {k:32s} {OPS[k]:>15,}")
print()
print("ALL ASSERTIONS PASSED - the matrix formulation OPERATES:")
print(" traces -> open count, corridor, depth-sum identity;")
print(" nilpotency of BS -> F at machines 11/13/17;")
print(" lift-tensor-delete -> merge law F(M+q') at 3 steps (11, 18, 34);")
print(" paired-Holt matrix -> exact new histograms at 4 rungs + pair words;")
print(" circulant diagonalisation -> closed-form spectrum, golden gap "
      "phi/3 exact at gear 5.")
print()
print("BOUNDARIES (Constructor r20 refutations, unchanged): the exact frame")
print("has NO spectral gap (the renewal operator is a permutation); the")
print("aggregated gap chain is NOT Markov (fixed-order transfer matrices")
print("over-predict deep qualifying runs by growing factors); B = I - (x)E_q")
print("is not a Kronecker product, so nilpotency needs the joint space.")

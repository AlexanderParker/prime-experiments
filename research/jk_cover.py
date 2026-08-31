"""jk_cover.py  --  exact maximal-covering search for the k-class Jacobsthal family.

HARVESTER lane, round 28.

OBJECT.  For k >= 1 and modulus m = P(z) = prod of primes <= z,

    j_k(P(z)) - 1  =  the longest interval coverable by choosing, at each prime
                      p <= z, a set S_p of residue classes mod p with
                      |S_p| <= min(k, p-1).                       (jk-family.md)

k = 1 is the ordinary Jacobsthal function (A048669), k = 2 is Ziller-Morack's
paired h_2 (A288815 carries h_2/2).

THIS FILE is the reference implementation: a plain, slow, obviously-correct
Python engine, used to (i) validate the covering restatement directly against
the DEFINITION by brute force at tiny (k, z), (ii) reproduce published values,
and (iii) cross-check the fast Rust engine (rust2/src/bin/jkcover.rs).

Two independent engines live here:
    dfs_maxrun()  -- branch and bound over "which prime covers the leftmost
                     uncovered position", maximising the run directly.
    sat_cover()   -- a CNF encoding decided by an off-the-shelf SAT solver
                     (python-sat, .venv-sat).  Completely different code path.

Run:  python research/jk_cover.py            (gate: all assertions)
      python research/jk_cover.py solve K Z  (one value)
"""
from __future__ import annotations

import sys
import time
from itertools import combinations


# --------------------------------------------------------------------------
# primes
# --------------------------------------------------------------------------
def primes_upto(n: int) -> list[int]:
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(range(i * i, n + 1, i)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def caps_for(k: int, z: int):
    ps = primes_upto(z)
    return ps, [min(k, p - 1) for p in ps]


# --------------------------------------------------------------------------
# ENGINE 0 -- brute force straight from the DEFINITION (tiny cases only)
# --------------------------------------------------------------------------
def jk_bruteforce_definition(k: int, z: int, emax: int | None = None) -> int:
    """j_k(P(z)) computed from the DEFINITION, not the covering restatement.

    j_k(m) = max over admissible k-tuples E = (0=E_0 <= ... <= E_{k-1}) of the
    largest cyclic gap between consecutive n with gcd(prod(n+E_i), m) = 1.

    Exhaustive over all tuples with entries < emax (default: the modulus).
    Only usable for z <= 7.
    """
    ps = primes_upto(z)
    P = 1
    for p in ps:
        P *= p
    if emax is None:
        emax = P

    best = 0
    # E_0 = 0 wlog (translation); entries strictly increasing wlog (a repeated
    # entry is the same condition, i.e. a smaller k)
    for r in range(1, k + 1):
        for tail in combinations(range(1, emax), r - 1):
            E = (0,) + tail
            killed = bytearray(P)
            for p in ps:
                cls = set((-e) % p for e in E)
                for c in cls:
                    killed[c::p] = b"\x01" * len(range(c, P, p))
            surv = [n for n in range(P) if not killed[n]]
            if not surv:
                continue
            g = max(
                (surv[i + 1] - surv[i]) for i in range(len(surv) - 1)
            ) if len(surv) > 1 else 0
            g = max(g, surv[0] + P - surv[-1])
            if g > best:
                best = g
    return best


# --------------------------------------------------------------------------
# ENGINE 1 -- branch and bound on the covering restatement
# --------------------------------------------------------------------------
class DFS:
    def __init__(self, k: int, z: int, lmax: int):
        self.ps, self.caps = caps_for(k, z)
        self.n = len(self.ps)
        self.lmax = lmax
        self.masks = []
        for p in self.ps:
            mm = []
            for r in range(p):
                m = 0
                for j in range(r, lmax, p):
                    m |= 1 << j
                mm.append(m)
            self.masks.append(mm)
        self.nodes = 0
        self.best = 0
        self.bestsol = None

    # ---- upper bound on the reachable run length from this state ----------
    def _ub(self, covered: int, j: int, free: list[int]) -> int:
        """Largest L that could still be reached: the first uncovered position
        x >= j at which the residual capacity provably runs out."""
        ps = self.ps
        n = self.n
        t = 0
        x = j
        lmax = self.lmax
        while x < lmax:
            if not ((covered >> x) & 1):
                t += 1
                M = x - j + 1
                cap = 0
                for i in range(n):
                    fi = free[i]
                    if fi:
                        p = ps[i]
                        cap += fi * ((M + p - 1) // p)
                        if cap >= t:
                            break
                if cap < t:
                    return x
            x += 1
        return lmax

    def run(self, covered: int, cnt: list[int], used: list[list[int]]):
        self.nodes += 1
        lmax = self.lmax
        # leftmost uncovered position
        j = 0
        while j < lmax and ((covered >> j) & 1):
            j += 1
        if j >= lmax:
            if lmax > self.best:
                self.best = lmax
                self.bestsol = [list(u) for u in used]
            return
        if j > self.best:
            self.best = j
            self.bestsol = [list(u) for u in used]
        free = [self.caps[i] - cnt[i] for i in range(self.n)]
        if not any(free):
            return
        if self._ub(covered, j, free) <= self.best:
            return
        # branch: which prime covers position j
        order = []
        for i in range(self.n):
            if free[i]:
                r = j % self.ps[i]
                nc = covered | self.masks[i][r]
                order.append((-bin(nc ^ covered).count("1"), i, r, nc))
        order.sort()
        for _, i, r, nc in order:
            cnt[i] += 1
            used[i].append(r)
            self.run(nc, cnt, used)
            used[i].pop()
            cnt[i] -= 1


def dfs_maxrun(k: int, z: int, lmax: int):
    """Return (L, solution) where L is the longest coverable interval."""
    d = DFS(k, z, lmax)
    d.run(0, [0] * d.n, [[] for _ in range(d.n)])
    if d.best >= lmax:
        raise RuntimeError(f"lmax={lmax} too small for k={k}, z={z}")
    return d.best, d.bestsol, d.nodes


def jk_covering(k: int, z: int, lmax: int | None = None) -> int:
    if lmax is None:
        lmax = 8
        while True:
            try:
                L, _, _ = dfs_maxrun(k, z, lmax)
                return L + 1
            except RuntimeError:
                lmax *= 2
    L, _, _ = dfs_maxrun(k, z, lmax)
    return L + 1


def verify_solution(k: int, z: int, L: int, sol) -> bool:
    """Independent check that `sol` really covers [0, L-1] within the caps."""
    ps, caps = caps_for(k, z)
    for i, u in enumerate(sol):
        if len(u) > caps[i]:
            return False
        if len(set(u)) != len(u):
            return False
        for r in u:
            if not (0 <= r < ps[i]):
                return False
    for j in range(L):
        ok = False
        for i, u in enumerate(sol):
            if (j % ps[i]) in u:
                ok = True
                break
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------
# ENGINE 2 -- SAT
# --------------------------------------------------------------------------
def sat_cover(k: int, z: int, L: int, solver_name: str = "cadical195"):
    """Is [0, L-1] coverable?  Returns (bool, solution-or-None)."""
    from pysat.formula import CNF, IDPool
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Solver

    ps, caps = caps_for(k, z)
    pool = IDPool()
    v = {}
    for i, p in enumerate(ps):
        for r in range(p):
            v[(i, r)] = pool.id(("x", i, r))
    cnf = CNF()
    for i, p in enumerate(ps):
        lits = [v[(i, r)] for r in range(p)]
        if caps[i] >= p:
            continue
        enc = CardEnc.atmost(lits=lits, bound=caps[i], vpool=pool,
                             encoding=EncType.seqcounter)
        cnf.extend(enc.clauses)
    for j in range(L):
        cnf.append([v[(i, j % p)] for i, p in enumerate(ps)])
    with Solver(name=solver_name, bootstrap_with=cnf) as s:
        sat = s.solve()
        if not sat:
            return False, None
        model = set(l for l in s.get_model() if l > 0)
        sol = [[r for r in range(p) if v[(i, r)] in model]
               for i, p in enumerate(ps)]
    return True, sol


def jk_sat(k: int, z: int, lo: int = 1, hi: int | None = None,
           solver_name: str = "cadical195") -> int:
    """j_k(P(z)) by SAT: linear scan upward from lo."""
    L = lo
    while True:
        ok, _ = sat_cover(k, z, L + 1, solver_name)
        if not ok:
            return L + 1
        L += 1
        if hi is not None and L > hi:
            raise RuntimeError("hi exceeded")


# --------------------------------------------------------------------------
# published reference values (OEIS records read first-hand 2026-08-29)
# --------------------------------------------------------------------------
PRIMES30 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
            67, 71, 73]
# A048670 #164 Jul 11 2026 -- ordinary Jacobsthal at primorials
A048670 = [2, 4, 6, 10, 14, 22, 26, 34, 40, 46, 58, 66, 74, 90, 100, 106, 118,
           132, 152, 174, 190]
# A288815 #19 Apr 12 2026 -- paired Jacobsthal at primorials (= 6*A072753 + 6)
A288815 = [2, 6, 18, 30, 66, 150, 192, 258, 366, 450, 570, 708, 894, 1044,
           1284, 1422, 1656, 1902, 2190, 2460, 2622]
J1_REF = dict(zip(PRIMES30, A048670))
J2_REF = dict(zip(PRIMES30, A288815))
# computed in-round by rust2/src/bin/jkcov6.rs
J3_REF = {3: 6, 5: 24, 7: 78, 11: 180, 13: 306, 17: 612, 19: 972}


def rust_value(k, z, secs=None):
    """Run the fast engine and return (j_k, exact?) or None if unavailable."""
    import os
    import subprocess
    exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "rust2", "target", "release", "jkcov6.exe")
    if not os.path.exists(exe):
        return None
    cmd = [exe, str(k), str(z), "--quiet"]
    if secs:
        cmd += ["--secs", str(secs)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    f = p.stdout.split()
    return int(f[2]), f[6] == "EXACT", f[7] == "true"


def _gate():
    t0 = time.time()
    import functools
    print("=" * 74)
    print("jk_cover.py  --  gate")
    print("=" * 74)

    # A. the covering restatement AGAINST THE DEFINITION, brute force
    print("\n[A] covering restatement vs the definition (brute force)")
    for k in (1, 2, 3):
        for z in (2, 3, 5, 7):
            if z == 7 and k == 3:
                emax = 40
            else:
                emax = None
            a = jk_bruteforce_definition(k, z, emax)
            b = jk_covering(k, z)
            print(f"    k={k} z={z:2d}   definition={a:4d}   covering={b:4d}"
                  f"   {'OK' if a == b else 'MISMATCH'}")
            assert a == b, (k, z, a, b)

    # B. published values, PYTHON reference engine (unreduced lattice)
    print("\n[B] published values -- python reference engine (no reduction,")
    print("    no canonical-form rule; the slowest and most obviously correct)")
    for z in (2, 3, 5, 7, 11, 13, 17):
        got = jk_covering(1, z)
        val = J1_REF[z]
        print(f"    j_1(P({z:2d})) = {got:4d}   (A048670: {val})"
              f"   {'OK' if got == val else 'MISMATCH'}")
        assert got == val, (z, got, val)
    for z in (2, 3, 5, 7, 11):
        got = jk_covering(2, z)
        val = J2_REF[z]
        print(f"    j_2(P({z:2d})) = {got:4d}   (A288815: {val})"
              f"   {'OK' if got == val else 'MISMATCH'}")
        assert got == val, (z, got, val)
    for z in (3, 5, 7):
        got = jk_covering(3, z)
        val = J3_REF[z]
        print(f"    j_3(P({z:2d})) = {got:4d}   (round 28: {val})"
              f"   {'OK' if got == val else 'MISMATCH'}")
        assert got == val, (z, got, val)

    # C. SAT engine agrees with the DFS engine
    print("\n[C] SAT engine vs DFS engine")
    print("    (python-sat lives in .venv-sat; the gate shells out to it)")
    import os
    import subprocess
    satpy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".venv-sat", "Scripts", "python.exe")
    me = os.path.abspath(__file__)
    if os.path.exists(satpy):
        r = subprocess.run([satpy, me, "satcheck"], capture_output=True, text=True)
        print(r.stdout.rstrip())
        assert r.returncode == 0, "SAT cross-check FAILED\n" + r.stderr
    else:
        print("    .venv-sat not present -- SKIPPED")

    # C2. the REDUCED rust engine agrees with the unreduced python engine, and
    #     reproduces the published tables far past where python can reach.
    print("\n[C2] rust engine (reduced lattice + canonical form) vs everything")
    if rust_value(1, 7) is None:
        print("    jkcov6.exe not built -- SKIPPED")
    else:
        for k, z, ref in ([(1, z, J1_REF[z]) for z in
                           (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43)]
                          + [(2, z, J2_REF[z]) for z in
                             (3, 5, 7, 11, 13, 17, 19, 23)]
                          + [(3, z, J3_REF[z]) for z in (3, 5, 7, 11, 13, 17, 19)]):
            v, exact, ver = rust_value(k, z)
            tag = "A048670" if k == 1 else ("A288815" if k == 2 else "round 28")
            print(f"    k={k} z={z:2d}  rust={v:5d}  {tag}={ref:5d}  "
                  f"exact={exact} witness={ver}  "
                  f"{'OK' if v == ref and exact else 'MISMATCH'}")
            assert v == ref and exact, (k, z, v, ref)

    # D. solutions verify independently
    print("\n[D] witnesses re-verified by independent code")
    for k, z in ((1, 19), (2, 17), (3, 7)):
        L, sol, _ = dfs_maxrun(k, z, 200)
        ok = verify_solution(k, z, L, sol)
        print(f"    k={k} z={z:2d}   L={L:4d}   witness {'VERIFIES' if ok else 'BAD'}")
        assert ok

    print(f"\njk_cover: ALL ASSERTIONS GREEN   ({time.time()-t0:.1f} s)")


def _satcheck():
    """Run in .venv-sat: the SAT engine must agree with the DFS engine."""
    for k, z in ((1, 13), (1, 17), (2, 11), (3, 5), (3, 7)):
        d = jk_covering(k, z)
        s = jk_sat(k, z, lo=max(1, d - 6))
        print(f"    k={k} z={z:2d}   dfs={d:4d}  sat={s:4d}"
              f"   {'OK' if d == s else 'MISMATCH'}")
        assert d == s, (k, z, d, s)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "satcheck":
        _satcheck()
        sys.exit(0)
    if len(sys.argv) >= 4 and sys.argv[1] == "solve":
        k = int(sys.argv[2]); z = int(sys.argv[3])
        lmax = int(sys.argv[4]) if len(sys.argv) > 4 else None
        t0 = time.time()
        if lmax is None:
            val = jk_covering(k, z)
            print(f"j_{k}(P({z})) = {val}   ({time.time()-t0:.1f} s)")
        else:
            L, sol, nodes = dfs_maxrun(k, z, lmax)
            print(f"j_{k}(P({z})) = {L+1}   L={L}  nodes={nodes}  "
                  f"({time.time()-t0:.1f} s)")
            print("witness:", sol)
    else:
        _gate()

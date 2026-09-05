"""Branch 5d.ii.i (prover, round 48): the summary tables the document quotes.

  T1  per machine: n, F, S_sat (where the umbrella count saturates), F/S_sat,
      W(q)/S_sat, max_S (f - h), the span where f first exceeds h, and
      F - S_max(n-1) (how far below the record a gear can first be spared).
  T2  the two ladders side by side: S_max^M(K) inside the machine, the F ladder,
      and the free adversarial ladder A(K).
  T3  the window: cov(actual stretch) against h_free from the A ladder.
Usage: uv run python research/anchor235/r48/summary.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cover_core import arcs  # noqa: E402

OUT = os.path.join(HERE, "results")
GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31]
LADDER = {2: 5, 3: 7, 4: 11, 5: 18, 6: 25, 7: 34, 8: 43, 9: 58, 10: 88, 11: 91,
          12: 103, 13: 118, 14: 145, 15: 161}


def main():
    log = open(os.path.join(OUT, "summary.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say("T1. THE UMBRELLA SATURATES; h DOES NOT.")
    say("  q   n  F   W(q)  S_sat  F/S_sat  W/S_sat  max(f-h)  first f>h  F-S_max(n-1)")
    ps = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for i, q in enumerate(GEARS[1:], start=2):
        d = json.load(open(os.path.join(OUT, f"machine_{q}.json")))
        n, F = d["n"], d["F"]
        rows = d["h_f"]
        qn = ps[ps.index(q) + 1]
        W = (qn * qn - 1) // 6
        gears = GEARS[:n]
        S_sat = max(arcs(g)[2] for g in gears) - 1
        gap = max(f - h for S, h, f in rows)
        first = next((S for S, h, f in rows if f > h), None)
        smax = d["S_max"]
        spare = F - smax[str(n - 1)] if str(n - 1) in smax else F - smax[n - 1]
        say(f" {q:3d} {n:3d} {F:3d} {W:6d} {S_sat:6d}  {F/S_sat:7.2f} {W/S_sat:8.2f}"
            f"  {gap:8d}  {str(first):>9}  {spare:12d}")

    say("")
    say("T2. THE TWO LADDERS.  S_max^M(K) = best K-subset of the machine {5..31};")
    say("    F ladder = F({5..p_K}); A(K) = best K gears anywhere (pool 5..149).")
    # A(K): exact for K <= 6 (exhaustive over the primes 5..149, adversary2.py);
    # K >= 7 are certified lower bounds from the exact {5..31} lattice.
    adv = {"A": {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 58,
             10: 88, 11: 91, 12: 103, 13: 118, 14: 145, 15: 161}}
    # (K >= 10 use A(K) >= F({5..p_K}), the initial segment being one choice)
    d31 = json.load(open(os.path.join(OUT, "machine_31.json")))
    say("   K  S_max^{5..31}(K)   F({5..p_K})   A(K)   A/F ladder")
    for K in range(1, 10):
        s = d31["S_max"].get(str(K), d31["S_max"].get(K))
        lad = LADDER.get(K, 0)
        a = adv["A"].get(K, "-")
        r = f"{a/lad:5.2f}" if adv and isinstance(a, int) and lad else "  -  "
        say(f"  {K:2d}   {s:12d}   {lad:11d}   {str(a):>5}   {r}")

    say("")
    say("T3. THE WINDOW RECORD: what covers it, against what could.")
    say("    (cov = exact minimum cover of the real stretch with the real phases;")
    say("     h_A = least K with A(K) >= F_W, the free minimum; both exact where A is)")
    A = dict(adv["A"])
    tab = [(12, 5, 7, 2), (52, 6, 17, 3), (58, 12, 19, 6), (110, 25, 23, 6),
           (397, 28, 47, 10), (980, 35, 73, 11), (2233, 47, 113, 14),
           (3090, 62, 137, 17), (4070, 83, 157, 20), (10383, 105, 241, 22),
           (31318, 154, 433, 27), (114742, 168, 829, 34), (141725, 242, 919, 32)]
    say("     x        F_W  first q   cov   h_A   cov/h_A")
    for x, FW, q, cov in tab:
        hA = min((K for K in sorted(A) if A[K] >= FW), default=None)
        exact = hA is not None and hA <= 6
        tag = f"{'' if exact else '>='}{cov/hA:6.2f}" if hA else "     -"
        say(f"  {x:8d} {FW:5d}  {q:6d}  {cov:4d}  "
            f"{(str(hA) + (' exact' if exact else ' <=')) if hA else '> 9':>10}  {tag}")
    log.close()


if __name__ == "__main__":
    main()

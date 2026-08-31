"""Round 28 (mechanic): re-verify EVERY witness this round produced, from the
definition, at its own machine, slot by slot.

The round's new exact values are each an upper bound from a scan plus a LOWER
bound from an exhibited window.  The scans are gated elsewhere; this gates the
exhibited half - one place, one command, no imports from the tools that found
them.

usage: <venv>/python research/witness_gate_r28.py
"""

def gears(y):
    return [p for p in range(5, y + 1)
            if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def check(y, k, offs, label):
    G = gears(y)
    u = {q: pow(6, -1, q) for q in G}
    oset = set(offs)
    nb = 0
    for t in range(offs[-1] + 1):
        op = all((k + t) % q not in (u[q] % q, (-u[q]) % q) for q in G)
        assert op == (t in oset), ("mismatch", label, t, op)
        if not op:
            nb += 1
    gaps = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
    assert sum(gaps) == offs[-1]
    print("  %-14s m%-2d k=%-22d gaps %-22s span %3d, %3d other slots blocked"
          % (label, y, k, str(gaps), sum(gaps), nb))
    return sum(gaps)


W = [
    (53, 2505673933219103747, [0, 10, 128, 161], "F(59)>=161"),
    (43, 2161962392309552, [0, 31, 116], "F_2(43)=116"),
    (43, 1595441702157105, [0, 67, 95, 125], "F_3(43)=125"),
    (43, 280183736276020, [0, 18, 42, 50, 132], "F_4(43)=132"),
    (41, 33044111735742, [0, 10, 61, 63, 113, 128], "F_5(41)=128"),
    (41, 17664265518665, [0, 15, 65, 67, 118, 128], "F_5(41) mirror"),
    (47, 36068193854725102, [0, 28, 61, 145], "F_3(47)=145"),
]
print("ROUND-28 WITNESS GATE - every exhibited window, re-checked from the "
      "definition\n")
spans = [check(*w) for w in W]
assert spans == [161, 116, 125, 132, 128, 128, 145]
# the F_5(41) pair must be an exact mirror pair in machine-23 coordinates
assert 4834937 + 32347080 == 5 * 7 * 11 * 13 * 17 * 19 * 23 - 128
print("\n  the two F_5(41) maximisers are an exact MIRROR PAIR: "
      "4,834,937 + 32,347,080 = P(23) - 128")
print("  and their gap words are exact reverses: [10,51,2,50,15] / "
      "[15,50,2,51,10]")
print("\nALL ASSERTIONS PASSED")

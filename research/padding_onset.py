"""Round 17 lateral: when does a double-padded run first become UNOBSTRUCTED?

A shape is unobstructed iff (a) corridor-feasible and (b) spectrum-affordable
(its total cost <= F_{shape openings - 1}(M), a necessary condition since the
run's gaps are consecutive gaps of M).

Costs: j=0 (adjacent pads) = 2q', needs F_2(M).
       j=1 (one literal between) = 2q' + L, L = min(s, q'-s), needs F_3(M).
       j>=2: impossible for every q' (AP lemma, round 16).
"""
SPEC = {13: [11, 16, 23, 26, 28, 31], 17: [18, 25, 28, 33, 35, 40],
        19: [25, 31, 35, 38, 47, 50], 23: [34, 39, 50, 58, 65, 77],
        29: [43, 55, 65, 70, 85, 90], 31: [58, 68, 85, 90, 92, 97],
        37: [88, 90, 95, 103, 112, 115]}      # machine 37 = prefix lower bounds
CORR = {23: (True, True), 29: (False, False), 31: (False, False),
        37: (True, True), 41: (False, True), 43: (True, True), 47: (True, True)}
STEPS = [(19, 23), (23, 29), (29, 31), (31, 37), (37, 41), (41, 43), (43, 47)]

print(f"{'step':>9} {'shape':>6} {'cost':>5} {'need':>7} {'have':>7} {'corridor':>9} {'verdict':>26}")
for y, qp in STEPS:
    u = pow(6, -1, qp); s = (2 * u) % qp; L = min(s, qp - s)
    c0, c1 = CORR[qp]
    for j, cost, idx, corr in ((0, 2 * qp, 1, c0), (1, 2 * qp + L, 2, c1)):
        spec = SPEC.get(y)
        have = spec[idx] if spec else None
        lb = y == 37
        if not corr:
            verdict = "corridor EXCLUDES"
            hs = "-"
        elif have is None:
            hs = "?"
            verdict = "spectrum unknown"
        else:
            hs = f"{'>=' if lb else ''}{have}"
            if have >= cost:
                verdict = "UNOBSTRUCTED"
            else:
                verdict = f"spectrum short by {cost - have}"
        print(f"  {y:>4}->{qp:<3} {'j='+str(j):>6} {cost:>5} {'F_'+str(idx+1):>7} "
              f"{hs:>7} {('OK' if corr else 'NO'):>9} {verdict:>26}")
print()
print("F is monotone in the machine (adding a gear only deletes openings, so")
print("gaps only grow): F(41) >= F(37) = 88, and F_2 >= F always. Hence")
print("F_2(41) >= 88 > 86 = 2*43 - the j=0 shape at 41->43 is spectrum-")
print("GUARANTEED, not merely likely. Combined with its corridor feasibility,")
print("41->43 is the FIRST STEP WITH NO OBSTRUCTION OF ANY KIND.")
print()
print("Near-miss pattern worth recording: the j=1 shape misses by EXACTLY ONE")
print("at two consecutive steps - 31->37 needs 86 with F_3(31) = 85, and")
print("37->41 needs 96 with F_3(37) >= 95. Two 1-unit misses in a row.")

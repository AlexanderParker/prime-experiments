"""Round 13 lateral: predictions for 37->41 and 41->43, stated IN ADVANCE.

Framework (exact): F(M+q') = max_w [span(w) + FS_max(w;M)], excess = F_new - F2
= max_{w nonempty} [span(w) - deficit(w)], deficit(w) = F2 - FS_max(w).

Literal words of q' have spans: k=2 -> {s, q'-s} = {(q'-1)/3 or (2q'+1)/3 and
its complement}; k=3 -> q'; k=4 -> q'+s or 2q'-s; k=5 -> 2q'; k=6 -> 2q'+s.
(Cap 6 = harvester/constructor theorem.)

H-SAT : deficits grow with span fast enough that the SHORT k=2 word keeps
        winning; excess ~ min(s,q'-s) - d, d ~ 6-8.
H-CLIMB: machines grow, long words become abundant, deficits per unit span
        shrink (deficit ~ 2.52*span/lambda, lambda = mean gap, growing only
        Mertens-slowly), so the winner migrates to longer words.
"""
def wordspans(qp):
    u = pow(6, -1, qp); s = (2 * u) % qp
    return s, {2: sorted({s, qp - s}), 3: [qp], 4: sorted({qp + s, 2 * qp - s}),
               5: [2 * qp], 6: [2 * qp + s]}

KNOWN = [(13, 17, 11, 16, 18), (17, 19, 18, 25, 25), (19, 23, 25, 31, 34),
         (23, 29, 34, 39, 43), (29, 31, 43, 55, 58), (31, 37, 58, 68, 88)]
print("RETRODICTION: does the winner's span exceed the SHORT k=2 span?")
print(f"  {'step':>9} {'q':>3} {'F2':>4} {'F_new':>5} {'excess':>6} "
      f"{'short':>5} {'long':>5} {'verdict':>28}")
for y, qp, Fo, F2, Fn in KNOWN:
    s, sp = wordspans(qp)
    short, long_ = sp[2][0], sp[2][1]
    exc = Fn - F2
    v = ("winner span >= %d > short %d -> NOT the short word" % (exc, short)
         if exc > short else "consistent with short word")
    print(f"  {y:>4}->{qp:<3} {qp:>3} {F2:>4} {Fn:>5} {exc:>6} "
          f"{short:>5} {long_:>5} {v:>28}")
print()
print("PREDICTIONS (stated in advance; mechanic's machine-37/41 census falsifies one)")
for y, qp, Fo, F2 in [(37, 41, 88, 90), (41, 43, None, None)]:
    s, sp = wordspans(qp)
    short, long_ = sp[2][0], sp[2][1]
    print(f"  step {y}->{qp}: s={s}, k=2 spans {{{short},{long_}}}, "
          f"k=3 span {sp[3][0]}, k=4 spans {sp[4]}, k=6 span {sp[6][0]}")
    if F2:
        print(f"    H-SAT  : excess ~ {short} - (6..8) = {short-8}..{short-6}"
              f"  -> F({qp}) ~ {F2+short-8}..{F2+short-6} "
              f"(excess/q' {(short-8)/qp:.2f}..{(short-6)/qp:.2f})")
        print(f"    H-CLIMB: excess ~ {long_} - (8..12) = {long_-12}..{long_-8}"
              f"  -> F({qp}) ~ {F2+long_-12}..{F2+long_-8} "
              f"(excess/q' {(long_-12)/qp:.2f}..{(long_-8)/qp:.2f})")
        print(f"    DISCRIMINATOR: F({qp}) <= {F2+short-4} favours SAT; "
              f">= {F2+long_-14} favours CLIMB")
    else:
        print(f"    (needs F2(41); shapes: H-SAT excess ~ {short-8}..{short-6}, "
              f"H-CLIMB excess ~ {long_-12}..{long_-8})")
print()
print("ASYMPTOTIC CEILING (unconditional, from the cap-6 theorem):")
for qp in (41, 43, 101, 1009):
    s, sp = wordspans(qp)
    print(f"  q'={qp:>4}: longest literal span = 2q'+s = {sp[6][0]} "
          f"= {sp[6][0]/qp:.2f} q'  -> excess/q' can never exceed {sp[6][0]/qp:.2f}"
          f" (deficits >= 0 only if FS_max <= F2)")

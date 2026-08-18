"""Round 16 lateral: corridor feasibility for FULL padded-run shapes.

A run with p = 2 padded links separated by j literal links has openings

    o, o+A, o+A+v_1, ..., o+A+sum(v), o+A+sum(v)+B

with A, B = a*q', b*q' (a,b >= 1) the padded links and v_1..v_j the literal
links, whose CLASSES alternate: a link = +2u (mod q') goes tooth -u -> +u, a
link = -2u goes +u -> -u, and a padded link (= 0) keeps the tooth. Minimal
literal values are s = 2u mod q' and q'-s.

Every opening lies in E, the 15-residue exposed set mod 35 (avoid the teeth of
gears 5 and 7). So the shape is CORRIDOR-FEASIBLE iff some r in E puts all of
its cumulative offsets in E mod 35. That is a pure residue test - no spectrum,
so it is immune to the machine-37 F_j values being prefix lower bounds.

Outputs: (a) feasibility of the minimal p=2 shapes for j = 0..4 over the primes;
(b) whether feasibility is a function of q' mod 210; (c) the 37->41 knife-edge
shape; (d) banked predictions for 41->43 and 43->47.
"""
E = sorted(k for k in range(35) if k % 5 not in (1, 4) and k % 7 not in (1, 6))
Eset = set(E)

def feasible(offsets):
    """Is there r in E with r + every offset in E (mod 35)?"""
    return [r for r in E if all((r + d) % 35 in Eset for d in offsets)]

def shape_offsets(qp, j, start_s, a=1, b=1):
    """Cumulative offsets of the p=2, j-literal-link minimal shape."""
    u = pow(6, -1, qp)
    s = (2 * u) % qp
    lits = []
    cur = s if start_s else qp - s
    for _ in range(j):
        lits.append(cur)
        cur = qp - cur                      # letters alternate
    offs = [0, a * qp]
    for v in lits:
        offs.append(offs[-1] + v)
    offs.append(offs[-1] + b * qp)
    return offs, lits

def part_a(primes_list, jmax=4):
    print("=" * 78)
    print("PART A: corridor feasibility of minimal p=2 shapes (a=b=1)")
    print(f"  E mod 35 ({len(E)}): {E}")
    print(f"  {'q':>4} {'q%35':>5} {'q%210':>6} " +
          " ".join(f"j={j:<3}" for j in range(jmax + 1)))
    table = {}
    for qp in primes_list:
        cells = []
        for j in range(jmax + 1):
            ok = False
            for start_s in (True, False):
                offs, _ = shape_offsets(qp, j, start_s)
                if feasible(offs):
                    ok = True
                    break
            cells.append(ok)
        table[qp] = cells
        print(f"  {qp:>4} {qp%35:>5} {qp%210:>6} " +
              " ".join(("YES  " if c else ".    ") for c in cells))
    print("  ('.' = corridor-IMPOSSIBLE: no phase admits the shape at all)")
    return table

def part_b(table):
    print("=" * 78)
    print("PART B: is feasibility a function of q' mod 210?")
    byres = {}
    clash = 0
    for qp, cells in table.items():
        r = qp % 210
        if r in byres and byres[r][1] != cells:
            clash += 1
            print(f"  CLASH at residue {r}: q'={byres[r][0]} {byres[r][1]} "
                  f"vs q'={qp} {cells}")
        byres.setdefault(r, (qp, cells))
    print(f"  distinct residues seen {len(byres)}, clashes {clash} -> "
          f"{'feasibility IS a function of q mod 210' if not clash else 'NOT a function of q mod 210'}")
    imp = {j: 0 for j in range(len(next(iter(table.values()))))}
    for cells in table.values():
        for j, c in enumerate(cells):
            if not c:
                imp[j] += 1
    n = len(table)
    print("  share of primes for which the shape is corridor-IMPOSSIBLE:")
    for j, c in imp.items():
        print(f"    j={j}: {c}/{n} = {100*c/n:.0f}%")

def part_c():
    print("=" * 78)
    print("PART C: the 37->41 knife-edge shape (j=1, needs F_3(37) >= 96)")
    qp = 41
    for start_s in (True, False):
        offs, lits = shape_offsets(qp, 1, start_s)
        r = feasible(offs)
        print(f"  literal link {lits[0]:>2}: offsets {offs} -> mod 35 "
              f"{[d % 35 for d in offs]}, total {offs[-1]}")
        print(f"    corridor phases: {r if r else 'NONE - IMPOSSIBLE'}")
    print("  VERDICT: the corridor does NOT settle the knife-edge - the")
    print("  (41,14,41) shape is corridor-feasible, so F_3(37) still decides.")

def part_d():
    print("=" * 78)
    print("PART D: banked predictions for the next steps")
    known_F = {37: 88, 43: 103}     # slot units; F(43)=103 from corpus F(2,43)=309
    for y, qp, note in [(37, 41, "census running"),
                        (41, 43, "banked"), (43, 47, "banked")]:
        print(f"  step {y}->{qp} (q' mod 35 = {qp%35}):")
        for j in range(0, 3):
            best = None
            for start_s in (True, False):
                offs, lits = shape_offsets(qp, j, start_s)
                r = feasible(offs)
                if r:
                    if best is None or offs[-1] < best[0]:
                        best = (offs[-1], lits, r)
            if best is None:
                print(f"    j={j}: corridor IMPOSSIBLE -> shape excluded "
                      f"unconditionally")
            else:
                tot, lits, r = best
                print(f"    j={j}: corridor OK (phases {r[:4]}), cheapest total "
                      f"{tot} -> needs F_{j+2}({y}) >= {tot}")

if __name__ == "__main__":
    ps = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
          101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
          167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233]
    t = part_a(ps)
    part_b(t)
    part_c()
    part_d()
